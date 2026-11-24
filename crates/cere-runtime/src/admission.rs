//! Admission primitive (m1 design note L4, m1close design note Q1/Q2, m3 design note Q5-Q7): global backend
//! in-flight cap, per-scope in-flight cap, per-worker in-flight caps, one bounded queue shared by all scopes,
//! weighted deficit round robin across scopes, priorities within a scope, and deadlines. The worker is chosen at
//! the grant. HTTP mapping (429/503) is the caller's.
//!
//! Every waiter carries a [`Target`]: `Any` worker of a deployment, a session's `Pinned` worker within a
//! deployment, or `Only` its own worker (a readiness smoke). A waiter is grantable when the global cap, its
//! scope's cap and an allowed worker's cap all have room. Each worker belongs to one deployment (M5 D6, fixed at
//! construction). A pin to an eligible worker of the waiter's deployment allows only that worker (strict
//! stickiness); `Any` and a pin to an ineligible or unknown worker, or to one of another deployment, allow every
//! eligible worker of the deployment; `Only` allows its worker in any state but lost and never moves.
//! Eligible means `ready` (not joining, draining, suspect or lost), judged at the grant on the worker's last
//! pushed [`WorkerView`] aged by the worker's own rule ([`WorkerCap::aging`], [`WorkerView::state_at`]; within
//! its lease when the view has one), and not marked suspect since its last sighting. Among the allowed workers
//! with room the grant takes the fewest in flight, ties by worker id.
//!
//! Grants follow deficit round robin with a row as the cost unit (Q1; M2.6 D12): each scope's turn is a quantum
//! of `weight` rows, after which the scope moves to the back of the round; the scope earliest in the round that
//! has a grantable waiter goes first (ties: earliest waiter). Every grant's rows count against the quantum,
//! including an orphan's charge (1 row), so weight 1 with cost-1 waiters is the M1 least-recently-served order.
//! Within a scope the next waiter is the grantable one served first by priority: `high`, then `normal`, then
//! `low`, FIFO within a level; a waiter queued longer than the aging bound is served as `high` (Q2).
//!
//! A waiter charges `cost` rows (a native request with `n` choices charges `n`, every other request 1) to the
//! global, scope and worker counts. Dispatch picks the head - the next waiter in that order among those with at
//! least one row of room - and grants it only when all `cost` rows fit; otherwise nothing is granted until they
//! do (head-of-line reservation, D12), so a cost-c waiter never starves behind a cost-1 stream. A waiter whose
//! cost exceeds a cap outright is never head (it would block the queue forever); it waits out its deadline.
//!
//! A native replica of a multi-replica deployment (m32n design note D6, D11) also has a KV pool
//! ([`Admission::set_pool`]); its waiters carry a [`Route`]: the pool tokens they reserve and an optional preferred
//! worker (prefix affinity). Such a worker is allowed only while its free pool holds the reservation; the grant
//! takes the pin, else the preferred worker, else the least loaded (fewest in flight, then most free pool tokens,
//! then id). A reservation no allowed worker's pool can ever hold is never head, like an oversized cost. Workers
//! without a pool (every other worker and the single native form) are unchanged.
//!
//! Dispatch runs after every enqueue, release, worker update, and at the next worker-aging or lease deadline (a
//! timer), so [`Admission::acquire`] grants at once iff dispatch picks the new waiter; otherwise it queues (or
//! fails `QueueFull` when `queue_depth` waiters already wait). A waiter whose deadline passes, or whose future is
//! dropped, leaves the queue and is never granted a slot. A permit keeps its rows charged until it is dropped, so
//! a caller keeps it until the backend stream has really closed (L6) or an orphan abort is settled (L3).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;
use tokio::sync::oneshot;
use tokio::time::Instant;

use crate::app::unix_ms;
use crate::config::{AdmissionConfig, ApiConfig, RuntimeConfig, DEFAULT_AGING_MS};
use crate::pool::{
    Class, WorkerPool, DEFAULT_BATCH_FLOOR, DEFAULT_CLASS, DEFAULT_OWNER_WAIT_CAP_MS,
};
use crate::tier::{Tier, DEFAULT_ALLOWED_TIERS};
use crate::worker::{next_aging_ms, Aging, WorkerHealth, WorkerState, WorkerView};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionLimits {
    pub max_inflight: u64,
    pub per_scope_inflight: u64,
    pub queue_depth: u64,
}

impl AdmissionLimits {
    pub fn from_config(admission: &AdmissionConfig) -> Self {
        Self {
            max_inflight: admission.max_backend_inflight,
            per_scope_inflight: admission.per_scope_inflight,
            queue_depth: admission.queue_depth,
        }
    }
}

/// A request's priority within its scope (`X-Cere-Priority`, Q2). The derived order is the service order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    High,
    Normal,
    Low,
}

impl Priority {
    /// `high`, `normal` or `low`, case-insensitive.
    pub fn parse(value: &str) -> Option<Self> {
        [Self::High, Self::Normal, Self::Low]
            .into_iter()
            .find(|p| value.eq_ignore_ascii_case(p.as_str()))
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Normal => "normal",
            Self::Low => "low",
        }
    }
}

/// Scheduling inputs beyond the caps: each scope's DRR weight (Q1; unlisted scopes weigh 1) and the queue
/// wait after which a waiter is served as [`Priority::High`] (Q2), plus the team-pool settings
/// (net-teleport design note 4.4): each scope's [`ScopePolicy`], the batch floor and the owner wait cap.
#[derive(Debug, Clone)]
pub struct SchedulePolicy {
    weights: HashMap<Arc<str>, u64>,
    aging: Duration,
    scopes: HashMap<Arc<str>, ScopePolicy>,
    /// Batch's share of the grants made while both classes wait, in thousandths.
    batch_floor_permille: u64,
    owner_wait_cap_ms: u64,
}

/// One scope's team-pool settings (C13, C17, C18); the default is an unlisted scope's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopePolicy {
    /// The class of a request that names none.
    pub class: Class,
    /// Replaces `admission.per_scope_inflight` for this scope.
    pub per_scope_inflight: Option<u64>,
    /// The cap on this scope's in-flight `batch` rows.
    pub max_batch_inflight: Option<u64>,
    /// The worker tiers this scope may be granted on ([`WorkerPool::tier_for`]).
    pub allowed_tiers: Vec<Tier>,
}

impl Default for ScopePolicy {
    fn default() -> Self {
        Self {
            class: DEFAULT_CLASS,
            per_scope_inflight: None,
            max_batch_inflight: None,
            allowed_tiers: DEFAULT_ALLOWED_TIERS.to_vec(),
        }
    }
}

impl Default for SchedulePolicy {
    fn default() -> Self {
        Self::new([], Duration::from_millis(DEFAULT_AGING_MS))
    }
}

impl SchedulePolicy {
    pub fn new(weights: impl IntoIterator<Item = (String, u64)>, aging: Duration) -> Self {
        Self {
            weights: weights
                .into_iter()
                .map(|(scope, weight)| (Arc::from(scope), weight.max(1)))
                .collect(),
            aging,
            scopes: HashMap::new(),
            batch_floor_permille: permille(DEFAULT_BATCH_FLOOR),
            owner_wait_cap_ms: DEFAULT_OWNER_WAIT_CAP_MS,
        }
    }

    /// `[[auth.scopes]]` `weight`, `class`, `per_scope_inflight`, `max_batch_inflight` and `allowed_tiers`, and
    /// `admission.aging_ms`, `batch_floor` and `owner_wait_cap_ms`.
    pub fn from_config(config: &RuntimeConfig) -> Self {
        let mut policy = Self::new(
            config
                .auth
                .scopes
                .iter()
                .map(|s| (s.scope_id.clone(), s.weight())),
            Duration::from_millis(config.admission.aging_ms()),
        )
        .with_pool_limits(
            config.admission.batch_floor(),
            Duration::from_millis(config.admission.owner_wait_cap_ms()),
        );
        for s in &config.auth.scopes {
            policy = policy.with_scope(
                &s.scope_id,
                ScopePolicy {
                    class: s.class(),
                    per_scope_inflight: s.per_scope_inflight,
                    max_batch_inflight: s.max_batch_inflight,
                    allowed_tiers: s.allowed_tiers().to_vec(),
                },
            );
        }
        policy
    }

    /// Sets `scope`'s team-pool settings.
    pub fn with_scope(mut self, scope: &str, policy: ScopePolicy) -> Self {
        self.scopes.insert(Arc::from(scope), policy);
        self
    }

    /// Sets the batch floor (a share of grants, 0..=1) and the owner wait cap.
    pub fn with_pool_limits(mut self, batch_floor: f64, owner_wait_cap: Duration) -> Self {
        self.batch_floor_permille = permille(batch_floor);
        self.owner_wait_cap_ms = owner_wait_cap.as_millis() as u64;
        self
    }

    fn weight(&self, scope: &str) -> u64 {
        self.weights.get(scope).copied().unwrap_or(1)
    }

    fn scope(&self, scope: &str) -> Option<&ScopePolicy> {
        self.scopes.get(scope)
    }

    fn class(&self, scope: &str) -> Class {
        self.scope(scope).map_or(DEFAULT_CLASS, |s| s.class)
    }

    fn tiers(&self, scope: &str) -> &[Tier] {
        self.scope(scope)
            .map_or(DEFAULT_ALLOWED_TIERS, |s| &s.allowed_tiers)
    }

    fn batch_cap(&self, scope: &str) -> u64 {
        self.scope(scope)
            .and_then(|s| s.max_batch_inflight)
            .unwrap_or(u64::MAX)
    }
}

/// Thousandths of a share, clamped to 0..=1000.
fn permille(share: f64) -> u64 {
    (share.clamp(0.0, 1.0) * 1000.0).round() as u64
}

/// One grant in the batch floor's credit ([`State::floor_credit`]).
const FLOOR_UNIT: i64 = 1000;

/// Team-pool grant counts since start (net-teleport design note 4.4; the `cere_admission_pool_grants_total`
/// family): grants made while both classes had a candidate, by the class granted ([`Class::index`]), and grants
/// owner priority (Q-T6) gave an owner's waiter on its own worker ahead of DRR.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PoolGrants {
    pub contested: [u64; 2],
    pub owner_priority: u64,
}

/// The request deadline budget: the client's `X-Cere-Deadline-Ms` (or `api.default_deadline_ms`) clamped to
/// `api.max_deadline_ms`. Queue wait counts against it.
pub fn deadline_budget_ms(requested: Option<u64>, api: &ApiConfig) -> u64 {
    requested
        .unwrap_or(api.default_deadline_ms)
        .min(api.max_deadline_ms)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionError {
    /// `queue_depth` waiters already queued (429, before generation).
    QueueFull,
    /// The deadline passed before a slot was granted; no slot was ever charged.
    DeadlineExceeded,
    /// Admission is closed for shutdown (503).
    Closed,
    /// Team pools (D-T7): the deadline passed while no eligible worker of the request's deployment was allowed
    /// for its scope (tier or owner-only window); the request never spills to a disallowed worker.
    NoAllowedWorker,
}

impl AdmissionError {
    pub fn code(self) -> &'static str {
        match self {
            Self::QueueFull => "queue_full",
            Self::DeadlineExceeded => "deadline_exceeded",
            Self::Closed => "shutting_down",
            Self::NoAllowedWorker => crate::pool::REASON_NO_ALLOWED_WORKER,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionSnapshot {
    pub inflight: u64,
    pub queued: u64,
    pub closed: bool,
}

/// Where a waiter may be granted (Q5); `usize` is the request's deployment (M5 D6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// Any eligible worker of the deployment.
    Any(usize),
    /// A session's sticky worker: only it while it is eligible and belongs to the deployment, otherwise any
    /// eligible worker of the deployment.
    Pinned(Arc<str>, usize),
    /// Only this worker, in any state but lost (a readiness smoke); never moved to another worker.
    Only(Arc<str>),
}

/// A worker grants go to, with its backend in-flight cap (Q6), the aging rule its health is judged by (M3.5
/// Q7: a remote worker's own, a local worker's the registry's) and the deployment it serves (M5 D6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerCap {
    pub id: String,
    pub cap: u64,
    pub aging: Aging,
    pub deployment: usize,
}

/// What a native multi-replica request adds to its rows (m32n design note D6, D11): its KV pool reservation and
/// the card whose prefix-affinity map holds its key. The default (no tokens, no preference) is every other
/// request's.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Route {
    /// Pool tokens reserved on the granted worker (`min(context, ceil64(prompt + max_tokens))`, F8).
    pub tokens: u64,
    /// D11 (2): preferred over least-loaded when eligible and fitting; never over a pin.
    pub prefer: Option<Arc<str>>,
    /// Team pools (C17): the request's class (`X-Cere-Class`); `None` -> its scope's default class.
    pub class: Option<Class>,
}

/// One worker's cap and charged slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerLoad {
    pub id: Arc<str>,
    pub cap: u64,
    pub inflight: u64,
    /// The in-flight of the scope asked about.
    pub scope_inflight: u64,
    /// The worker's KV pool tokens (0 = untracked) and the tokens its grants reserve (m32n design note D6).
    pub pool: u64,
    pub reserved: u64,
}

#[derive(Debug, Clone)]
pub struct Admission {
    shared: Arc<Shared>,
}

#[derive(Debug)]
struct Shared {
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    /// The caps and scheduling policy, under the same lock as the counts so a live reload
    /// ([`Admission::reconfigure`]) swaps them atomically between grants.
    limits: AdmissionLimits,
    policy: SchedulePolicy,
    inflight: u64,
    per_scope: HashMap<Arc<str>, u64>,
    queues: HashMap<Arc<str>, VecDeque<Waiter>>,
    queued: u64,
    /// Each served scope's place in the round and what is left of its quantum (Q1).
    turns: HashMap<Arc<str>, Turn>,
    /// Quanta spent so far: the place a scope takes at the back of the round.
    rounds: u64,
    next_waiter: u64,
    closed: bool,
    /// Sorted by id, the grant's tie-break.
    workers: Vec<WorkerSlot>,
    /// The worker-aging deadline (unix ms) a dispatch timer is armed for.
    timer_ms: Option<u64>,
    /// [`AgingClock::Wall`] outside the unit tests.
    clock: AgingClock,
    /// Team pools (C18): each scope's in-flight `batch` rows.
    batch_per_scope: HashMap<Arc<str>, u64>,
    /// Team pools (C17): over the grants made while both classes had a candidate ("contested"), the batch floor's
    /// share of them minus the batch grants among them, in thousandths of a grant. A contested grant goes to batch
    /// when it would leave the credit positive, so after every contested grant batch holds at least
    /// `ceil(batch_floor x contested grants)` of them.
    floor_credit: i64,
    /// Grants counted for the team-pool metrics ([`Admission::pool_grants`]).
    pool_grants: PoolGrants,
}

/// The unix-ms clock worker aging and leases are judged on (Q7): the wall clock that stamps worker sightings. The
/// unit tests' `on_tokio_clock` runs it on tokio's clock instead, so a paused runtime steps aging in virtual time
/// and no test asserts a wall-clock window between a sighting and a grant.
#[derive(Debug, Default, Clone, Copy)]
enum AgingClock {
    #[default]
    Wall,
    /// `base_ms` at tokio instant `at`, advanced by tokio's clock since.
    #[cfg(test)]
    Tokio { base_ms: u64, at: Instant },
}

impl AgingClock {
    fn now_ms(self) -> u64 {
        match self {
            Self::Wall => unix_ms(),
            #[cfg(test)]
            Self::Tokio { base_ms, at } => base_ms + at.elapsed().as_millis() as u64,
        }
    }
}

/// A scope's deficit-round-robin turn. A scope never served has place 0 and a fresh quantum.
#[derive(Debug, Default, Clone, Copy)]
struct Turn {
    place: u64,
    deficit: u64,
}

/// What a waiter asked for; a re-queued grant keeps it (Q7).
#[derive(Debug, Clone)]
struct Ticket {
    id: u64,
    priority: Priority,
    queued_at: Instant,
    target: Target,
    /// Rows charged on grant (D12).
    cost: u64,
    /// m32n design note D6/D11: pool tokens reserved on grant and the preferred worker.
    route: Route,
    /// Team pools (C17): resolved at the ticket (the header's, else the scope's default).
    class: Class,
}

#[derive(Debug)]
struct Waiter {
    ticket: Ticket,
    tx: oneshot::Sender<Permit>,
}

#[derive(Debug)]
struct WorkerSlot {
    id: Arc<str>,
    cap: u64,
    aging: Aging,
    deployment: usize,
    inflight: u64,
    /// m32n design note D6: KV pool tokens (0 = untracked: the single form) and the tokens grants reserve.
    pool: u64,
    reserved: u64,
    per_scope: HashMap<Arc<str>, u64>,
    /// The last pushed view; joining at epoch 0 until the first update.
    view: WorkerView,
    /// Set by `mark_suspect`: the sighting it judged. A newer sighting or another epoch clears it.
    suspect_mark: Option<u64>,
    /// The aged state, refreshed at the start of each dispatch.
    state: WorkerState,
    /// Team pools (net-teleport design note 4.4): owner, tier and owner-only windows.
    team: WorkerPool,
    /// Since when (unix ms) the worker's local owner-only switch reads on (its heartbeat), if it does.
    switch_since: Option<u64>,
    /// Since when the worker is owner-only (a window or the switch), refreshed at the start of each dispatch.
    owner_only: Option<u64>,
}

impl WorkerSlot {
    fn new(worker: WorkerCap) -> Self {
        Self {
            id: Arc::from(worker.id),
            cap: worker.cap,
            aging: worker.aging,
            deployment: worker.deployment,
            inflight: 0,
            pool: 0,
            reserved: 0,
            per_scope: HashMap::new(),
            view: WorkerView {
                health: WorkerHealth::Joining,
                last_seen_ms: 0,
                epoch: 0,
                draining: false,
                lease_until_ms: None,
            },
            suspect_mark: None,
            state: WorkerState::Joining,
            team: WorkerPool::default(),
            switch_since: None,
            owner_only: None,
        }
    }

    /// The pushed view with a suspect mark applied.
    fn marked_view(&self) -> WorkerView {
        match self.view.health {
            WorkerHealth::Ready if self.suspect_mark.is_some() => WorkerView {
                health: WorkerHealth::Suspect,
                ..self.view
            },
            _ => self.view,
        }
    }

    fn state_at(&self, now_ms: u64) -> WorkerState {
        self.marked_view().state_at(now_ms, self.aging)
    }

    /// The next instant the slot's state changes without an update: an aging deadline, the end of its lease or an
    /// owner-only window's start or end.
    fn next_aging_ms(&self, now_ms: u64) -> Option<u64> {
        let view = self.marked_view();
        let aged = next_aging_ms(view.health, view.last_seen_ms, now_ms, self.aging);
        let lease = view.lease_until_ms.filter(|&until| until > now_ms);
        let window = self.team.next_window_change(now_ms);
        aged.into_iter().chain(lease).chain(window).min()
    }

    /// Since when the worker is owner-only at `now_ms`: the earlier of a window's start and the switch's.
    fn owner_only_at(&self, now_ms: u64) -> Option<u64> {
        let window = self.team.window_since(now_ms);
        window.into_iter().chain(self.switch_since).min()
    }

    /// Team pools (4.1, 4.4): `scope` (allowed `tiers`) may be granted here: the worker's tier for it is allowed,
    /// and the worker is not owner-only or `scope` owns it.
    fn admits(&self, scope: &str, tiers: &[Tier]) -> bool {
        crate::tier::allows(tiers, self.team.tier_for(scope))
            && (self.owner_only.is_none() || self.team.is_owner(scope))
    }

    fn has_room(&self, rows: u64) -> bool {
        self.inflight + rows <= self.cap
    }

    /// m32n design note D6: an untracked pool (0) holds anything; a tracked one its free tokens.
    fn pool_holds(&self, tokens: u64) -> bool {
        self.pool == 0 || self.reserved + tokens <= self.pool
    }

    fn free_pool(&self) -> u64 {
        self.pool.saturating_sub(self.reserved)
    }

    fn charge(&mut self, scope: &Arc<str>, rows: u64, tokens: u64) {
        self.inflight += rows;
        self.reserved += tokens;
        *self.per_scope.entry(scope.clone()).or_insert(0) += rows;
    }

    fn uncharge(&mut self, scope: &str, rows: u64, tokens: u64) {
        self.inflight -= rows;
        self.reserved -= tokens;
        uncount(&mut self.per_scope, scope, rows);
    }
}

fn uncount(per_scope: &mut HashMap<Arc<str>, u64>, scope: &str, rows: u64) {
    if let Some(n) = per_scope.get_mut(scope) {
        *n -= rows;
        if *n == 0 {
            per_scope.remove(scope);
        }
    }
}

/// A waiter that entered the queue: granted by the dispatch that followed, or still waiting.
enum Entry {
    Granted(Permit),
    Queued(Arc<str>, u64, oneshot::Receiver<Permit>),
}

/// The next grant [`State::choose`] found: a waiter (its scope and queue index), the worker owner priority
/// reserved it (`None`: picked as usual) and whether both classes had a candidate (the batch floor counts it).
struct Choice {
    scope: Arc<str>,
    index: usize,
    worker: Option<usize>,
    contested: bool,
}

/// A ticket's priority as served: a waiter queued longer than `aging` counts as high (Q2).
fn effective(ticket: &Ticket, now: Instant, aging: Duration) -> Priority {
    if now.saturating_duration_since(ticket.queued_at) > aging {
        Priority::High
    } else {
        ticket.priority
    }
}

/// What a grant's worker means for its scope (net-teleport design note 4.4), for the usage row and the work cap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantTeam {
    /// The worker's owner scope; `None` for an unowned worker.
    pub owner: Option<Arc<str>>,
    /// The worker's tier for the granted scope ([`WorkerPool::tier_for`]).
    pub tier: Tier,
    /// Q-T6/Q-T8: a non-owner's work on an owned worker ends within `admission.owner_wait_cap_ms` of its grant.
    pub work_cap: Option<Duration>,
}

/// One owned worker's owner-only state ([`Admission::owner_only_states`]): `(worker, owner, since)`, `since` the
/// unix ms its current window or switch span began, `None` outside one.
pub type OwnerOnlyState = (Arc<str>, Arc<str>, Option<u64>);

impl State {
    fn scope_inflight(&self, scope: &str) -> u64 {
        self.per_scope.get(scope).copied().unwrap_or(0)
    }

    /// `scope`'s in-flight cap: its `per_scope_inflight` override (C18), else `admission.per_scope_inflight`.
    fn scope_cap(&self, scope: &str) -> u64 {
        self.policy
            .scope(scope)
            .and_then(|s| s.per_scope_inflight)
            .unwrap_or(self.limits.per_scope_inflight)
    }

    /// The `batch` rows `scope` may still be granted under its `max_batch_inflight` (C18).
    fn batch_room(&self, scope: &str) -> u64 {
        let held = self.batch_per_scope.get(scope).copied().unwrap_or(0);
        self.policy.batch_cap(scope).saturating_sub(held)
    }

    fn worker_index(&self, id: &str) -> Option<usize> {
        self.workers.iter().position(|w| *w.id == *id)
    }

    /// Charges the ticket's rows to `scope`, the global count and `worker`, and spends them from the scope's
    /// quantum; a spent quantum moves the scope to the back of the round.
    fn charge(
        &mut self,
        shared: &Arc<Shared>,
        scope: Arc<str>,
        worker: Option<usize>,
        ticket: Ticket,
    ) -> Permit {
        let rows = ticket.cost;
        self.inflight += rows;
        *self.per_scope.entry(scope.clone()).or_insert(0) += rows;
        if ticket.class == Class::Batch {
            *self.batch_per_scope.entry(scope.clone()).or_insert(0) += rows;
        }
        let turn = self.turns.entry(scope.clone()).or_default();
        if turn.deficit == 0 {
            turn.deficit = self.policy.weight(&scope);
        }
        turn.deficit = turn.deficit.saturating_sub(rows);
        if turn.deficit == 0 {
            self.rounds += 1;
            turn.place = self.rounds;
        }
        let owner_cap = Duration::from_millis(self.policy.owner_wait_cap_ms);
        let (worker, epoch, team) = match worker {
            Some(i) => {
                let w = &mut self.workers[i];
                w.charge(&scope, rows, ticket.route.tokens);
                let team = GrantTeam {
                    owner: w.team.owner.clone(),
                    tier: w.team.tier_for(&scope),
                    work_cap: (w.team.owner.is_some() && !w.team.is_owner(&scope))
                        .then_some(owner_cap),
                };
                (Some(w.id.clone()), w.view.epoch, Some(team))
            }
            None => (None, 0, None),
        };
        Permit {
            shared: shared.clone(),
            scope,
            worker,
            epoch,
            team,
            ticket,
            armed: true,
        }
    }

    fn uncharge(&mut self, scope: &str, worker: Option<&str>, ticket: &Ticket) {
        let rows = ticket.cost;
        self.inflight -= rows;
        uncount(&mut self.per_scope, scope, rows);
        if ticket.class == Class::Batch {
            uncount(&mut self.batch_per_scope, scope, rows);
        }
        if let Some(i) = worker.and_then(|id| self.worker_index(id)) {
            self.workers[i].uncharge(scope, rows, ticket.route.tokens);
        }
    }

    /// The worker a waiter of `scope` with `target` is granted on now, if any (Q5), among those `fits` accepts:
    /// the pin while it is eligible, else `prefer` (m32n design note D11 (2)) when allowed, else the least
    /// loaded. For `Any` and `Pinned`, eligible includes the team-pool rules ([`WorkerSlot::admits`]): a worker
    /// of a disallowed tier, or owner-only for another scope, is never picked (D-T7: no spill).
    fn pick(
        &self,
        scope: &str,
        target: &Target,
        prefer: Option<&str>,
        fits: impl Fn(&WorkerSlot) -> bool,
    ) -> Option<usize> {
        let tiers = self.policy.tiers(scope);
        let ready = |i: usize| self.workers[i].state == WorkerState::Ready;
        let serves = |d: usize| {
            move |i: usize| {
                ready(i) && self.workers[i].deployment == d && self.workers[i].admits(scope, tiers)
            }
        };
        let route = |d: usize| {
            prefer
                .and_then(|id| self.worker_index(id))
                .filter(|&i| serves(d)(i) && fits(&self.workers[i]))
                .or_else(|| self.least_loaded(serves(d), &fits))
        };
        match target {
            Target::Only(id) => self
                .worker_index(id)
                .filter(|&i| self.workers[i].state != WorkerState::Lost && fits(&self.workers[i])),
            Target::Pinned(id, d) => match self.worker_index(id).filter(|&i| serves(*d)(i)) {
                Some(i) => Some(i).filter(|&i| fits(&self.workers[i])),
                None => route(*d),
            },
            Target::Any(d) => route(*d),
        }
    }

    /// D-T7: a waiter of `scope` with `target` whose deadline passed was refused by the team-pool rules rather than
    /// load: every worker of its deployment has a tier `scope` may not use, or some are ready and no ready one admits
    /// it (tier or another scope's owner-only window; spec 4.1: "no allowed eligible worker"). `Only` targets are
    /// never refused so.
    fn denied(&self, scope: &str, target: &Target) -> bool {
        let d = match target {
            Target::Any(d) | Target::Pinned(_, d) => *d,
            Target::Only(_) => return false,
        };
        let tiers = self.policy.tiers(scope);
        let deployment = || self.workers.iter().filter(move |w| w.deployment == d);
        if deployment().next().is_none() {
            return false;
        }
        if !deployment().any(|w| crate::tier::allows(tiers, w.team.tier_for(scope))) {
            return true;
        }
        let mut ready = deployment()
            .filter(|w| w.state == WorkerState::Ready)
            .peekable();
        ready.peek().is_some() && ready.all(|w| !w.admits(scope, tiers))
    }

    /// The allowed worker `fits` accepts with the fewest in flight, then the most free pool tokens (D11 (3); equal
    /// for untracked pools), ties by id.
    fn least_loaded(
        &self,
        allowed: impl Fn(usize) -> bool,
        fits: impl Fn(&WorkerSlot) -> bool,
    ) -> Option<usize> {
        self.workers
            .iter()
            .enumerate()
            .filter(|&(i, w)| allowed(i) && fits(w))
            .min_by(|(_, a), (_, b)| {
                (a.inflight, std::cmp::Reverse(a.free_pool()), &a.id).cmp(&(
                    b.inflight,
                    std::cmp::Reverse(b.free_pool()),
                    &b.id,
                ))
            })
            .map(|(i, _)| i)
    }

    /// The waiters of `scope`'s `queue` served next, one per [`Class`] (indexed by [`Class::index`]) - highest
    /// effective priority (a waiter queued longer than `aging` counts as high), then earliest - among those with a
    /// row of room on an allowed worker whose cost and pool reservation some allowed worker and the caps can ever
    /// hold (a `batch` waiter also within the scope's batch room, C18), as `(level, id, index)`.
    fn next_candidates(
        &self,
        scope: &str,
        queue: &VecDeque<Waiter>,
        now: Instant,
        aging: Duration,
    ) -> [Option<(Priority, u64, usize)>; 2] {
        let (max_inflight, scope_cap) = (self.limits.max_inflight, self.scope_cap(scope));
        let batch_room = self.batch_room(scope);
        let mut best = [None, None];
        for (index, w) in queue.iter().enumerate() {
            let (cost, tokens) = (w.ticket.cost, w.ticket.route.tokens);
            let target = &w.ticket.target;
            let candidate = cost <= max_inflight
                && cost <= scope_cap
                && (w.ticket.class != Class::Batch || cost <= batch_room)
                && self.pick(scope, target, None, |s| s.has_room(1)).is_some()
                && self.pick(scope, target, None, |s| cost <= s.cap).is_some()
                && self
                    .pick(scope, target, None, |s| s.pool == 0 || tokens <= s.pool)
                    .is_some();
            if !candidate {
                continue;
            }
            let key = (effective(&w.ticket, now, aging), w.ticket.id, index);
            let slot = &mut best[w.ticket.class.index()];
            if slot.is_none_or(|b| key < b) {
                *slot = Some(key);
            }
        }
        best
    }

    /// Owner priority (Q-T6): the waiter of `scope` that a worker `scope` owns can take now - in either class,
    /// highest effective priority then earliest - as `(level, id, index, worker)`. `None` when `scope` owns no
    /// worker with room for one.
    fn owner_candidate(
        &self,
        scope: &str,
        queue: &VecDeque<Waiter>,
        now: Instant,
        aging: Duration,
    ) -> Option<(Priority, u64, usize, usize)> {
        if !self
            .workers
            .iter()
            .any(|w| w.team.is_owner(scope) && w.has_room(1))
        {
            return None;
        }
        let room = self
            .scope_cap(scope)
            .saturating_sub(self.scope_inflight(scope));
        let global = self.limits.max_inflight.saturating_sub(self.inflight);
        let batch_room = self.batch_room(scope);
        queue
            .iter()
            .enumerate()
            .filter(|(_, w)| {
                let cost = w.ticket.cost;
                cost <= room
                    && cost <= global
                    && (w.ticket.class != Class::Batch || cost <= batch_room)
            })
            .filter_map(|(index, w)| {
                let (cost, tokens) = (w.ticket.cost, w.ticket.route.tokens);
                let fits = |s: &WorkerSlot| s.has_room(cost) && s.pool_holds(tokens);
                let worker = self
                    .workers
                    .iter()
                    .filter(|o| o.team.is_owner(scope))
                    .find_map(|o| {
                        self.pick(scope, &w.ticket.target, Some(&o.id), fits)
                            .filter(|&i| self.workers[i].team.is_owner(scope))
                    })?;
                Some((effective(&w.ticket, now, aging), w.ticket.id, index, worker))
            })
            .min()
    }

    /// The next grant, if any waiter is a candidate: first an owner's waiter for its own worker (Q-T6), else the
    /// class served - interactive, but batch once the floor it accrued over contested grants reaches a whole
    /// grant (C17) - and in it the scope earliest in the round (DRR by weight) with its candidate.
    fn choose(&self, now: Instant, aging: Duration) -> Option<Choice> {
        let open = || {
            self.queues
                .iter()
                .filter(|(scope, _)| self.scope_inflight(scope) < self.scope_cap(scope))
        };
        let owner = open()
            .filter_map(|(scope, queue)| {
                let (level, id, index, worker) = self.owner_candidate(scope, queue, now, aging)?;
                Some((level, id, index, worker, scope))
            })
            .min();
        if let Some((_, _, index, worker, scope)) = owner {
            return Some(Choice {
                scope: scope.clone(),
                index,
                worker: Some(worker),
                contested: false,
            });
        }
        // A class's best candidate so far: (round place, waiter id, queue index, scope).
        type Best<'a> = Option<(u64, u64, usize, &'a Arc<str>)>;
        let mut best: [Best<'_>; 2] = [None, None];
        for (scope, queue) in open() {
            let place = self.turns.get(scope).map_or(0, |t| t.place);
            for (slot, candidate) in best
                .iter_mut()
                .zip(self.next_candidates(scope, queue, now, aging))
            {
                let Some((_, id, index)) = candidate else {
                    continue;
                };
                let key = (place, id, index, scope);
                if slot.is_none_or(|b| key < b) {
                    *slot = Some(key);
                }
            }
        }
        let [interactive, batch] = best;
        let contested = interactive.is_some() && batch.is_some();
        let floor_due = self.floor_credit + self.policy.batch_floor_permille as i64 > 0;
        let (_, _, index, scope) = if contested && floor_due {
            batch
        } else {
            interactive.or(batch)
        }?;
        Some(Choice {
            scope: scope.clone(),
            index,
            worker: None,
            contested,
        })
    }

    /// Grants every waiter it can: the head ([`Self::choose`]) when its rows fit the global, scope and a worker's
    /// room; a head that does not fit stops the grants until it does (D12). Then arms the aging timer.
    fn dispatch(&mut self, shared: &Arc<Shared>) {
        let (limits, aging) = (self.limits, self.policy.aging);
        let now = Instant::now();
        let now_ms = self.clock.now_ms();
        for w in &mut self.workers {
            w.state = w.state_at(now_ms);
            w.owner_only = w.owner_only_at(now_ms);
        }
        while self.inflight < limits.max_inflight {
            let Some(choice) = self.choose(now, aging) else {
                break;
            };
            let scope = choice.scope;
            let ticket = &self.queues[&scope][choice.index].ticket;
            let cost = ticket.cost;
            if self.inflight + cost > limits.max_inflight
                || self.scope_inflight(&scope) + cost > self.scope_cap(&scope)
            {
                break;
            }
            let tokens = ticket.route.tokens;
            let fits = |s: &WorkerSlot| s.has_room(cost) && s.pool_holds(tokens);
            let picked = choice.worker.or_else(|| {
                self.pick(&scope, &ticket.target, ticket.route.prefer.as_deref(), fits)
            });
            let Some(worker) = picked else {
                break;
            };
            let queue = self
                .queues
                .get_mut(&scope)
                .expect("selected scope is queued");
            let waiter = queue
                .remove(choice.index)
                .expect("selected waiter is queued");
            if queue.is_empty() {
                self.queues.remove(&scope);
            }
            self.queued -= 1;
            if choice.contested {
                self.floor_credit += self.policy.batch_floor_permille as i64;
                if waiter.ticket.class == Class::Batch {
                    self.floor_credit -= FLOOR_UNIT;
                }
            }
            let class = waiter.ticket.class;
            let permit = self.charge(shared, scope, Some(worker), waiter.ticket);
            match waiter.tx.send(permit) {
                Ok(()) if choice.contested => self.pool_grants.contested[class.index()] += 1,
                Ok(()) if choice.worker.is_some() => self.pool_grants.owner_priority += 1,
                Ok(()) => {}
                Err(mut permit) => {
                    // The waiter went away between its last poll and now: undo without re-entering the lock.
                    permit.armed = false;
                    self.uncharge(&permit.scope, permit.worker.as_deref(), &permit.ticket);
                }
            }
        }
        self.arm_timer(shared, now_ms);
    }

    /// While waiters queue, arms one timer that re-runs dispatch at the next worker-aging deadline (Q7): a worker
    /// aging to suspect re-routes the waiters pinned to it, and nothing else would run dispatch then.
    fn arm_timer(&mut self, shared: &Arc<Shared>, now_ms: u64) {
        if self.queued == 0 {
            return;
        }
        let Some(at) = self
            .workers
            .iter()
            .filter_map(|w| w.next_aging_ms(now_ms))
            .min()
        else {
            return;
        };
        if self
            .timer_ms
            .is_some_and(|armed| armed > now_ms && armed <= at)
        {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        self.timer_ms = Some(at);
        let shared = Arc::downgrade(shared);
        runtime.spawn(async move {
            tokio::time::sleep(Duration::from_millis(at - now_ms)).await;
            if let Some(shared) = shared.upgrade() {
                let mut st = lock(&shared);
                if st.timer_ms == Some(at) {
                    st.timer_ms = None;
                }
                st.dispatch(&shared);
            }
        });
    }

    /// Queues `ticket` for `scope` and runs dispatch. A new arrival (`bounded`) that dispatch did not grant at
    /// once fails `QueueFull` when `queue_depth` waiters already wait.
    fn enter(
        &mut self,
        shared: &Arc<Shared>,
        scope: Arc<str>,
        ticket: Ticket,
        deadline: Instant,
        bounded: bool,
    ) -> Result<Entry, AdmissionError> {
        if self.closed {
            return Err(AdmissionError::Closed);
        }
        if Instant::now() >= deadline {
            return Err(AdmissionError::DeadlineExceeded);
        }
        let id = ticket.id;
        let (tx, mut rx) = oneshot::channel();
        self.queues
            .entry(scope.clone())
            .or_default()
            .push_back(Waiter { ticket, tx });
        self.queued += 1;
        self.dispatch(shared);
        if let Ok(permit) = rx.try_recv() {
            return Ok(Entry::Granted(permit));
        }
        if bounded && self.queued > self.limits.queue_depth {
            self.remove_waiter(shared, &scope, id);
            return Err(AdmissionError::QueueFull);
        }
        Ok(Entry::Queued(scope, id, rx))
    }

    /// Removes waiter `id` of `scope` if it still queues, then dispatches: with D12's head-of-line reservation a
    /// leaving waiter may have been the head that held the others back (M26R-01).
    fn remove_waiter(&mut self, shared: &Arc<Shared>, scope: &str, id: u64) {
        let Some(queue) = self.queues.get_mut(scope) else {
            return;
        };
        let before = queue.len();
        queue.retain(|w| w.ticket.id != id);
        let removed = (before - queue.len()) as u64;
        if queue.is_empty() {
            self.queues.remove(scope);
        }
        self.queued -= removed;
        if removed > 0 {
            self.dispatch(shared);
        }
    }

    fn mark_suspect(&mut self, worker: &str) {
        if let Some(i) = self.worker_index(worker) {
            let w = &mut self.workers[i];
            w.suspect_mark = Some(w.view.last_seen_ms);
        }
    }

    /// A new ticket of `scope`; its class is the route's, else the scope's default (C17).
    fn ticket(
        &mut self,
        scope: &str,
        priority: Priority,
        target: Target,
        cost: u64,
        route: Route,
    ) -> Ticket {
        self.next_waiter += 1;
        let class = route.class.unwrap_or_else(|| self.policy.class(scope));
        Ticket {
            id: self.next_waiter,
            priority,
            queued_at: Instant::now(),
            target,
            cost,
            route,
            class,
        }
    }
}

impl Admission {
    pub fn new(limits: AdmissionLimits) -> Self {
        Self::with_policy(limits, SchedulePolicy::default())
    }

    pub fn with_policy(limits: AdmissionLimits, policy: SchedulePolicy) -> Self {
        Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    limits,
                    policy,
                    ..State::default()
                }),
            }),
        }
    }

    pub fn limits(&self) -> AdmissionLimits {
        self.lock().limits
    }

    /// A live config reload (`crate::reload`): new caps and scheduling policy for every grant from now on.
    /// Charged permits keep their rows; queued waiters are re-dispatched under the new limits (a raised cap
    /// grants the waiters it makes room for, a lowered one grants nothing until enough permits drop).
    pub fn reconfigure(&self, limits: AdmissionLimits, policy: SchedulePolicy) {
        let mut st = self.lock();
        st.limits = limits;
        st.policy = policy;
        st.dispatch(&self.shared);
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        lock(&self.shared)
    }

    /// Sets the workers grants go to, each with its in-flight cap and aging rule (Q6, Q7, M3.5 Q7). Called once at
    /// construction, before the first grant. A worker reads joining until its first [`Self::update_worker`].
    pub fn set_workers(&self, workers: Vec<WorkerCap>) {
        let mut slots: Vec<WorkerSlot> = workers.into_iter().map(WorkerSlot::new).collect();
        slots.sort_by(|a, b| a.id.cmp(&b.id));
        let mut st = self.lock();
        st.workers = slots;
        st.dispatch(&self.shared);
    }

    /// M3.5 Q11: sets worker `id`'s in-flight cap (a remote worker's cap follows its inventory). Charged slots
    /// stay charged; a higher cap grants the waiters it makes room for. Unknown ids are ignored.
    pub fn set_cap(&self, id: &str, cap: u64) {
        let mut st = self.lock();
        let Some(i) = st.worker_index(id) else {
            return;
        };
        st.workers[i].cap = cap;
        st.dispatch(&self.shared);
    }

    /// m32n design note D6: sets worker `id`'s KV pool tokens (a native replica's `kv_pool_tokens`); a grant on
    /// it then needs its [`Route::tokens`] free in the pool. 0 (the default) leaves the pool untracked. Unknown ids
    /// are ignored.
    pub fn set_pool(&self, id: &str, pool: u64) {
        let mut st = self.lock();
        let Some(i) = st.worker_index(id) else {
            return;
        };
        st.workers[i].pool = pool;
        st.dispatch(&self.shared);
    }

    /// Pushes worker `id`'s registry view (every readiness publish, Q7); eligibility is judged on it, aged, at
    /// each grant. A newer sighting or another epoch clears a [`Self::mark_suspect`] mark. Unknown ids are
    /// ignored.
    pub fn update_worker(&self, id: &str, view: WorkerView) {
        let mut st = self.lock();
        let Some(i) = st.worker_index(id) else {
            return;
        };
        let w = &mut st.workers[i];
        if w.suspect_mark
            .is_some_and(|seen| view.last_seen_ms > seen || view.epoch != w.view.epoch)
        {
            w.suspect_mark = None;
        }
        w.view = view;
        st.dispatch(&self.shared);
    }

    /// Takes worker `id` out of new grants (it reads suspect) until an update with a newer sighting (Q7).
    pub fn mark_suspect(&self, id: &str) {
        let mut st = self.lock();
        st.mark_suspect(id);
        st.dispatch(&self.shared);
    }

    /// Waits until `deadline` for `cost` rows (D12: a native request's `n`, otherwise 1) for `scope` at
    /// `priority` on a worker `target` allows.
    pub async fn acquire(
        &self,
        scope: &str,
        priority: Priority,
        deadline: Instant,
        target: Target,
        cost: u64,
    ) -> Result<Permit, AdmissionError> {
        self.acquire_routed(scope, priority, deadline, target, cost, Route::default())
            .await
    }

    /// [`Self::acquire`] with a pool reservation and a preferred worker (m32n design note D11): the grant goes to
    /// the pin, else the preferred worker, else the least loaded, among the allowed workers whose rows and pool
    /// both hold the request.
    pub async fn acquire_routed(
        &self,
        scope: &str,
        priority: Priority,
        deadline: Instant,
        target: Target,
        cost: u64,
        route: Route,
    ) -> Result<Permit, AdmissionError> {
        let entry = {
            let mut st = self.lock();
            let ticket = st.ticket(scope, priority, target, cost, route);
            st.enter(&self.shared, Arc::from(scope), ticket, deadline, true)?
        };
        self.wait(entry, deadline).await
    }

    /// Q7: `begin_work` rejected `permit`'s worker. Marks that worker suspect (so the waiter cannot return to
    /// it), releases the slot and queues the same waiter again - its id, queue time, priority and target, so it
    /// goes ahead of later waiters - then waits until `deadline` for a new grant. Never fails `QueueFull` (it was
    /// admitted); the caller re-queues once.
    pub async fn requeue(
        &self,
        mut permit: Permit,
        deadline: Instant,
    ) -> Result<Permit, AdmissionError> {
        permit.armed = false;
        let entry = {
            let mut st = self.lock();
            if let Some(worker) = permit.worker.as_deref() {
                st.mark_suspect(worker);
            }
            st.uncharge(&permit.scope, permit.worker.as_deref(), &permit.ticket);
            let entered = st.enter(
                &self.shared,
                permit.scope.clone(),
                permit.ticket.clone(),
                deadline,
                false,
            );
            if entered.is_err() {
                // Closed or past its deadline: the freed slot still goes to the next waiter.
                st.dispatch(&self.shared);
            }
            entered?
        };
        self.wait(entry, deadline).await
    }

    async fn wait(&self, entry: Entry, deadline: Instant) -> Result<Permit, AdmissionError> {
        let (scope, id, rx) = match entry {
            Entry::Granted(permit) => return Ok(permit),
            Entry::Queued(scope, id, rx) => (scope, id, rx),
        };
        let guard = WaitGuard {
            shared: &self.shared,
            scope,
            id,
        };
        match tokio::time::timeout_at(deadline, rx).await {
            Ok(Ok(permit)) => Ok(permit),
            Ok(Err(_)) => Err(AdmissionError::Closed),
            Err(_) if guard.denied() => Err(AdmissionError::NoAllowedWorker),
            Err(_) => Err(AdmissionError::DeadlineExceeded),
        }
    }

    /// Charges a slot for `scope` regardless of caps and closure: a prior-epoch orphan whose backend request may
    /// still run (L3). `worker` is the orphan's worker; `None` (or an unknown id) charges the global and scope
    /// counts only. New work waits until the counts fall below the caps again.
    pub fn charge_orphan(&self, scope: &str, worker: Option<&str>) -> Permit {
        let mut st = self.lock();
        let index = worker.and_then(|id| st.worker_index(id));
        // Never re-queued: the ticket's target is never read.
        let deployment = index.map_or(0, |i| st.workers[i].deployment);
        let ticket = st.ticket(
            scope,
            Priority::Normal,
            Target::Any(deployment),
            1,
            Route::default(),
        );
        st.charge(&self.shared, Arc::from(scope), index, ticket)
    }

    /// Stops admission for shutdown: queued waiters and later calls get `Closed`; held permits stay charged.
    pub fn close(&self) {
        let waiters = {
            let mut st = self.lock();
            st.closed = true;
            st.queued = 0;
            std::mem::take(&mut st.queues)
        };
        drop(waiters);
    }

    pub fn snapshot(&self) -> AdmissionSnapshot {
        let st = self.lock();
        AdmissionSnapshot {
            inflight: st.inflight,
            queued: st.queued,
            closed: st.closed,
        }
    }

    pub fn scope_inflight(&self, scope: &str) -> u64 {
        self.lock().scope_inflight(scope)
    }

    /// Each worker's cap and charged slots, in total and for `scope`, sorted by id.
    pub fn worker_loads(&self, scope: &str) -> Vec<WorkerLoad> {
        self.lock()
            .workers
            .iter()
            .map(|w| WorkerLoad {
                id: w.id.clone(),
                cap: w.cap,
                inflight: w.inflight,
                scope_inflight: w.per_scope.get(scope).copied().unwrap_or(0),
                pool: w.pool,
                reserved: w.reserved,
            })
            .collect()
    }

    /// Team pools: sets worker `id`'s owner, tier and owner-only windows (C13, C15, C19). Unknown ids are ignored;
    /// a worker without a call is unowned, tier `own`, with no windows.
    pub fn set_worker_pool(&self, id: &str, team: WorkerPool) {
        let mut st = self.lock();
        let Some(i) = st.worker_index(id) else {
            return;
        };
        st.workers[i].team = team;
        st.dispatch(&self.shared);
    }

    /// Team pools: worker `id`'s local owner-only switch as its last heartbeat reported it (C19). Turning it on
    /// stamps the span's start; off ends it. Unknown ids are ignored.
    pub fn set_owner_only(&self, id: &str, on: bool) {
        let mut st = self.lock();
        let now_ms = st.clock.now_ms();
        let Some(i) = st.worker_index(id) else {
            return;
        };
        let w = &mut st.workers[i];
        match (on, w.switch_since) {
            (true, None) => w.switch_since = Some(now_ms),
            (false, Some(_)) => w.switch_since = None,
            _ => return,
        }
        st.dispatch(&self.shared);
    }

    /// Every owned worker's [`OwnerOnlyState`] now, with `now_ms` (the usage report's opt-out hours,
    /// net-teleport design note 4.4; [`crate::pool::OwnerOnlyTracker`]).
    pub fn owner_only_states(&self) -> (u64, Vec<OwnerOnlyState>) {
        let st = self.lock();
        let now_ms = st.clock.now_ms();
        let states = st
            .workers
            .iter()
            .filter_map(|w| Some((w.id.clone(), w.team.owner.clone()?, w.owner_only_at(now_ms))))
            .collect();
        (now_ms, states)
    }

    /// Team-pool grant counts since start ([`PoolGrants`]).
    pub fn pool_grants(&self) -> PoolGrants {
        self.lock().pool_grants
    }
}

fn lock(shared: &Shared) -> MutexGuard<'_, State> {
    shared.state.lock().unwrap_or_else(|e| e.into_inner())
}

/// Removes a waiter that stops waiting (deadline, cancellation) so it never holds queue depth or a grant.
struct WaitGuard<'a> {
    shared: &'a Arc<Shared>,
    scope: Arc<str>,
    id: u64,
}

impl Drop for WaitGuard<'_> {
    fn drop(&mut self) {
        lock(self.shared).remove_waiter(self.shared, &self.scope, self.id);
    }
}

impl WaitGuard<'_> {
    /// D-T7: the waiter, still queued, is refused by the team-pool rules rather than load ([`State::denied`]).
    fn denied(&self) -> bool {
        let st = lock(self.shared);
        st.queues
            .get(&self.scope)
            .and_then(|q| q.iter().find(|w| w.ticket.id == self.id))
            .is_some_and(|w| st.denied(&self.scope, &w.ticket.target))
    }
}

/// One grant's charged rows (global, scope and worker). Dropping it releases them and grants the next waiter.
#[derive(Debug)]
pub struct Permit {
    shared: Arc<Shared>,
    scope: Arc<str>,
    /// `None` for a global-only orphan charge.
    worker: Option<Arc<str>>,
    epoch: u64,
    /// `None` for a global-only orphan charge.
    team: Option<GrantTeam>,
    ticket: Ticket,
    armed: bool,
}

impl Permit {
    pub fn scope(&self) -> &str {
        &self.scope
    }

    /// The worker the slot is charged to; empty for a global-only orphan charge.
    pub fn worker(&self) -> &str {
        self.worker.as_deref().unwrap_or("")
    }

    /// The worker's epoch in its last pushed view at the grant; 0 for a global-only orphan charge.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The request's class (C17).
    pub fn class(&self) -> Class {
        self.ticket.class
    }

    /// The granted worker's owner and tier for this scope, and the non-owner work cap; `None` for an orphan
    /// charge.
    pub fn team(&self) -> Option<&GrantTeam> {
        self.team.as_ref()
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        if self.armed {
            let mut st = lock(&self.shared);
            st.uncharge(&self.scope, self.worker.as_deref(), &self.ticket);
            st.dispatch(&self.shared);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::{Aging, WorkerHealth, WorkerView};
    use futures_util::FutureExt;
    use std::future::Future;
    use std::pin::Pin;
    use std::time::Duration;

    impl Admission {
        /// Worker aging and leases on tokio's clock from now: under `start_paused` they move only as virtual time
        /// does, so a sighting `n` ms old stays exactly that old until the test (or an armed timer) advances.
        fn on_tokio_clock(self) -> Self {
            self.lock().clock = AgingClock::Tokio {
                base_ms: unix_ms(),
                at: Instant::now(),
            };
            self
        }

        /// Now on the clock aging is judged on.
        fn now_ms(&self) -> u64 {
            self.lock().clock.now_ms()
        }
    }

    fn limits(max_inflight: u64, per_scope_inflight: u64, queue_depth: u64) -> AdmissionLimits {
        AdmissionLimits {
            max_inflight,
            per_scope_inflight,
            queue_depth,
        }
    }

    fn far() -> Instant {
        Instant::now() + Duration::from_secs(60)
    }

    /// Worker aging that never fires.
    const NEVER: Aging = Aging {
        suspect_after_ms: u64::MAX,
        lost_after_ms: u64::MAX,
    };

    fn view(health: WorkerHealth, last_seen_ms: u64) -> WorkerView {
        WorkerView {
            health,
            last_seen_ms,
            epoch: 1,
            draining: false,
            lease_until_ms: None,
        }
    }

    fn ready() -> WorkerView {
        view(WorkerHealth::Ready, unix_ms())
    }

    /// `health`, last seen `ago` ms before now on `adm`'s aging clock.
    fn seen_ago(adm: &Admission, health: WorkerHealth, ago: u64) -> WorkerView {
        view(health, adm.now_ms() - ago)
    }

    /// Registers `workers` as `(id, cap)` under `aging`, each one ready now on `adm`'s aging clock.
    fn with_workers(adm: Admission, workers: &[(&str, u64)], aging: Aging) -> Admission {
        let caps = workers
            .iter()
            .map(|&(id, cap)| WorkerCap {
                id: id.to_owned(),
                cap,
                aging,
                deployment: 0,
            })
            .collect();
        adm.set_workers(caps);
        for &(id, _) in workers {
            adm.update_worker(id, seen_ago(&adm, WorkerHealth::Ready, 0));
        }
        adm
    }

    /// The single-worker shape of the scheduling tests: one ready worker whose cap never binds.
    fn single(adm: Admission) -> Admission {
        let cap = adm.limits().max_inflight;
        with_workers(adm, &[("w", cap)], NEVER)
    }

    /// Ready workers `a` and `b` that never age.
    fn two(limits: AdmissionLimits, cap_a: u64, cap_b: u64) -> Admission {
        with_workers(Admission::new(limits), &[("a", cap_a), ("b", cap_b)], NEVER)
    }

    fn pinned(id: &str) -> Target {
        Target::Pinned(Arc::from(id), 0)
    }

    fn only(id: &str) -> Target {
        Target::Only(Arc::from(id))
    }

    type Acquire = Pin<Box<dyn Future<Output = Result<Permit, AdmissionError>> + Send>>;

    fn waiter(adm: &Admission, scope: &str, priority: Priority, target: Target) -> Acquire {
        let (adm, scope) = (adm.clone(), scope.to_owned());
        Box::pin(async move { adm.acquire(&scope, priority, far(), target, 1).await })
    }

    /// A `normal` grant that must happen on the first poll.
    fn granted(adm: &Admission, scope: &str, target: Target) -> Permit {
        adm.acquire(scope, Priority::Normal, far(), target, 1)
            .now_or_never()
            .expect("granted at once")
            .unwrap()
    }

    /// `(worker, cap, in-flight, in-flight of scope)` per worker.
    fn loads(adm: &Admission, scope: &str) -> Vec<(String, u64, u64, u64)> {
        adm.worker_loads(scope)
            .into_iter()
            .map(|l| (l.id.to_string(), l.cap, l.inflight, l.scope_inflight))
            .collect()
    }

    #[tokio::test]
    async fn global_and_per_scope_caps() {
        let adm = single(Admission::new(limits(2, 1, 4)));
        let a = adm
            .acquire("a", Priority::Normal, far(), Target::Any(0), 1)
            .await
            .unwrap();
        // Scope a is at its cap; b gets the second global slot.
        let mut a2 = Box::pin(adm.acquire("a", Priority::Normal, far(), Target::Any(0), 1));
        assert!(futures_util::poll!(&mut a2).is_pending());
        let b = adm
            .acquire("b", Priority::Normal, far(), Target::Any(0), 1)
            .await
            .unwrap();
        // Global cap reached: c queues even though its scope is idle.
        let mut c = Box::pin(adm.acquire("c", Priority::Normal, far(), Target::Any(0), 1));
        assert!(futures_util::poll!(&mut c).is_pending());
        assert_eq!(
            adm.snapshot(),
            AdmissionSnapshot {
                inflight: 2,
                queued: 2,
                closed: false
            }
        );
        assert_eq!((adm.scope_inflight("a"), adm.scope_inflight("b")), (1, 1));
        // Releasing b frees a global slot; a2 is still blocked by its scope cap, c gets it.
        drop(b);
        let c = c.await.unwrap();
        assert_eq!(c.scope(), "c");
        assert!(futures_util::poll!(&mut a2).is_pending());
        drop(a);
        let a2 = a2.await.unwrap();
        assert_eq!(adm.snapshot().inflight, 2);
        drop((a2, c));
        assert_eq!(
            adm.snapshot(),
            AdmissionSnapshot {
                inflight: 0,
                queued: 0,
                closed: false
            }
        );
    }

    #[tokio::test]
    async fn two_scopes_five_requests_interleave_round_robin() {
        let adm = single(Admission::new(limits(1, 1, 16)));
        let order = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let first = adm
            .acquire("a", Priority::Normal, far(), Target::Any(0), 1)
            .await
            .unwrap();
        let mut handles = Vec::new();
        let labels: Vec<String> = (2..=5)
            .map(|i| format!("a{i}"))
            .chain((1..=5).map(|i| format!("b{i}")))
            .collect();
        for label in labels {
            let (adm, order) = (adm.clone(), order.clone());
            let mut fut = Box::pin(async move {
                let scope = &label[..1];
                let permit = adm
                    .acquire(scope, Priority::Normal, far(), Target::Any(0), 1)
                    .await
                    .unwrap();
                order.lock().await.push(label.clone());
                tokio::task::yield_now().await;
                drop(permit);
            });
            // Enqueue in submission order.
            assert!(futures_util::poll!(&mut fut).is_pending());
            handles.push(tokio::spawn(fut));
        }
        assert_eq!(adm.snapshot().queued, 9);
        drop(first);
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(
            *order.lock().await,
            ["b1", "a2", "b2", "a3", "b3", "a4", "b4", "a5", "b5"]
        );
    }

    /// Queues `labels` (scope = first character) in order behind a held slot, releases it, and returns the
    /// grant order. `priority_of` gives each label's priority.
    async fn grant_order(
        adm: &Admission,
        held: Permit,
        labels: Vec<String>,
        priority_of: impl Fn(&str) -> Priority,
    ) -> Vec<String> {
        let order = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for label in labels {
            let (adm, order) = (adm.clone(), order.clone());
            let priority = priority_of(&label);
            let mut fut = Box::pin(async move {
                let scope = &label[..1];
                let permit = adm
                    .acquire(scope, priority, far(), Target::Any(0), 1)
                    .await
                    .unwrap();
                order.lock().await.push(label.clone());
                tokio::task::yield_now().await;
                drop(permit);
            });
            assert!(futures_util::poll!(&mut fut).is_pending());
            handles.push(tokio::spawn(fut));
        }
        drop(held);
        for h in handles {
            h.await.unwrap();
        }
        Arc::try_unwrap(order)
            .expect("every waiter task finished")
            .into_inner()
    }

    fn weighted(weights: &[(&str, u64)], aging: Duration) -> SchedulePolicy {
        SchedulePolicy::new(weights.iter().map(|(s, w)| ((*s).to_owned(), *w)), aging)
    }

    /// Q1: under saturation, weights 3:1 give 3:1 grants (±1) over 40 grants.
    #[tokio::test]
    async fn weighted_scopes_share_grants_three_to_one() {
        let policy = weighted(&[("a", 3), ("b", 1)], Duration::from_secs(60));
        let adm = single(Admission::with_policy(limits(1, 1, 100), policy));
        let held = adm
            .acquire("c", Priority::Normal, far(), Target::Any(0), 1)
            .await
            .unwrap();
        let labels = (0..40)
            .map(|i| format!("a{i}"))
            .chain((0..40).map(|i| format!("b{i}")))
            .collect();
        let order = grant_order(&adm, held, labels, |_| Priority::Normal).await;
        let a = order[..40].iter().filter(|l| l.starts_with('a')).count();
        assert!((29..=31).contains(&a), "a got {a} of 40: {order:?}");
        // FIFO within a scope.
        let a_order: Vec<&String> = order.iter().filter(|l| l.starts_with('a')).collect();
        let expected: Vec<String> = (0..40).map(|i| format!("a{i}")).collect();
        assert_eq!(a_order, expected.iter().collect::<Vec<_>>());
    }

    /// Q2: within one scope, high before normal before low, FIFO within a level.
    #[tokio::test]
    async fn priority_levels_within_a_scope() {
        let adm = single(Admission::new(limits(1, 1, 16)));
        let held = adm
            .acquire("a", Priority::Normal, far(), Target::Any(0), 1)
            .await
            .unwrap();
        let labels = ["a-n1", "a-l1", "a-h1", "a-n2", "a-h2"]
            .map(str::to_owned)
            .to_vec();
        let order = grant_order(&adm, held, labels, |l| match &l[2..3] {
            "h" => Priority::High,
            "l" => Priority::Low,
            _ => Priority::Normal,
        })
        .await;
        assert_eq!(order, ["a-h1", "a-h2", "a-n1", "a-n2", "a-l1"]);
    }

    /// Q2: a waiter queued longer than `aging_ms` is served as high, ahead of newer highs; a low waiter that
    /// has not aged yet still waits for both. On the paused tokio clock (waiters age on it): the first low ages by
    /// exactly the 80 ms slept, and no time passes between the newer waiters' enqueue and their grants however slow
    /// the host.
    #[tokio::test(start_paused = true)]
    async fn aged_low_waiter_is_served_before_newer_highs() {
        let policy = weighted(&[], Duration::from_millis(50));
        let adm = single(Admission::with_policy(limits(1, 1, 16), policy));
        let held = adm
            .acquire("a", Priority::Normal, far(), Target::Any(0), 1)
            .await
            .unwrap();
        let order = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let spawn = |label: &'static str, priority| {
            let (adm, order) = (adm.clone(), order.clone());
            Box::pin(async move {
                let permit = adm
                    .acquire("a", priority, far(), Target::Any(0), 1)
                    .await
                    .unwrap();
                order.lock().await.push(label);
                tokio::task::yield_now().await;
                drop(permit);
            })
        };
        let mut low = spawn("low", Priority::Low);
        assert!(futures_util::poll!(&mut low).is_pending());
        let low = tokio::spawn(low);
        tokio::time::sleep(Duration::from_millis(80)).await;
        let mut handles = vec![low];
        for (label, priority) in [
            ("fresh-low", Priority::Low),
            ("high1", Priority::High),
            ("high2", Priority::High),
        ] {
            let mut waiter = spawn(label, priority);
            assert!(futures_util::poll!(&mut waiter).is_pending());
            handles.push(tokio::spawn(waiter));
        }
        drop(held);
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(*order.lock().await, ["low", "high1", "high2", "fresh-low"]);
    }

    /// priority design note on the production config (`prod-t4.toml`: 16 rows since BC7, the one `frontend` scope,
    /// `aging_ms` 30 s): frontend sends main sessions as `high` and subagents and helpers as `low`. At the row cap,
    /// each `high` queued behind a waiting `low` gets the next freed row while the `low` is younger than `aging_ms`;
    /// the first row freed after that is the `low`'s, though a new `high` queues every turn (starvation bound
    /// `aging_ms` + one turn). Paused tokio clock: a row frees every `aging / 8`.
    #[tokio::test(start_paused = true)]
    async fn prod_high_is_admitted_before_a_queued_low_and_the_low_never_starves() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../configs/runtime/prod-t4.toml"
        );
        let config = RuntimeConfig::load(path).unwrap();
        assert_eq!(config.admission.aging_ms, Some(30_000));
        let limits = AdmissionLimits::from_config(&config.admission);
        let policy = SchedulePolicy::from_config(&config);
        let aging = policy.aging;
        let rows = limits.max_inflight.min(limits.per_scope_inflight);
        assert_eq!(rows, 16);
        let adm = single(Admission::with_policy(limits, policy));
        let mut held: VecDeque<Permit> = (0..rows)
            .map(|_| granted(&adm, "frontend", Target::Any(0)))
            .collect();
        let mut low = waiter(&adm, "frontend", Priority::Low, Target::Any(0));
        assert!(futures_util::poll!(&mut low).is_pending());
        let queued_at = Instant::now();
        let turn = aging / 8;
        let mut highs_first = 0;
        loop {
            assert!(highs_first < 64, "low starved");
            let mut high = waiter(&adm, "frontend", Priority::High, Target::Any(0));
            assert!(futures_util::poll!(&mut high).is_pending());
            tokio::time::sleep(turn).await;
            drop(held.pop_front());
            if let std::task::Poll::Ready(permit) = futures_util::poll!(&mut low) {
                let waited = queued_at.elapsed();
                assert!(
                    waited > aging && waited <= aging + turn,
                    "low granted after {waited:?} (aging {aging:?}, turn {turn:?})"
                );
                assert!(futures_util::poll!(&mut high).is_pending());
                held.push_back(permit.unwrap());
                break;
            }
            match futures_util::poll!(&mut high) {
                std::task::Poll::Ready(permit) => held.push_back(permit.unwrap()),
                std::task::Poll::Pending => {
                    panic!("a freed row went to neither the high nor the low")
                }
            }
            highs_first += 1;
        }
        assert_eq!(highs_first, 8);
    }

    /// Weights and priorities never bypass the caps or the queue bound.
    #[tokio::test]
    async fn weights_keep_caps_and_queue_depth() {
        let policy = weighted(&[("a", 100)], Duration::from_secs(60));
        let adm = single(Admission::with_policy(limits(2, 1, 2), policy));
        let a = adm
            .acquire("a", Priority::Normal, far(), Target::Any(0), 1)
            .await
            .unwrap();
        let mut a2 = Box::pin(adm.acquire("a", Priority::High, far(), Target::Any(0), 1));
        assert!(futures_util::poll!(&mut a2).is_pending());
        let b = adm
            .acquire("b", Priority::Normal, far(), Target::Any(0), 1)
            .await
            .unwrap();
        let mut c = Box::pin(adm.acquire("c", Priority::Normal, far(), Target::Any(0), 1));
        assert!(futures_util::poll!(&mut c).is_pending());
        let err = adm
            .acquire("a", Priority::High, far(), Target::Any(0), 1)
            .now_or_never()
            .unwrap()
            .unwrap_err();
        assert_eq!(err, AdmissionError::QueueFull);
        // b's slot frees a global slot; a is still at its scope cap, so c gets it despite a's weight.
        drop(b);
        let c = c.await.unwrap();
        assert!(futures_util::poll!(&mut a2).is_pending());
        drop(a);
        let a2 = a2.await.unwrap();
        assert_eq!(adm.snapshot().inflight, 2);
        drop((a2, c));
    }

    #[test]
    fn priority_header_values() {
        assert_eq!(Priority::parse("HIGH"), Some(Priority::High));
        assert_eq!(Priority::parse("normal"), Some(Priority::Normal));
        assert_eq!(Priority::parse("Low"), Some(Priority::Low));
        assert_eq!(Priority::parse("urgent"), None);
        assert_eq!(Priority::parse(""), None);
    }

    #[tokio::test]
    async fn queue_full_rejects_immediately() {
        let adm = single(Admission::new(limits(1, 1, 1)));
        let _held = adm
            .acquire("a", Priority::Normal, far(), Target::Any(0), 1)
            .await
            .unwrap();
        let mut queued = Box::pin(adm.acquire("b", Priority::Normal, far(), Target::Any(0), 1));
        assert!(futures_util::poll!(&mut queued).is_pending());
        let err = adm
            .acquire("c", Priority::Normal, far(), Target::Any(0), 1)
            .now_or_never()
            .unwrap()
            .unwrap_err();
        assert_eq!(err, AdmissionError::QueueFull);
        assert_eq!(err.code(), "queue_full");
    }

    #[tokio::test]
    async fn expiry_while_queued_never_grants() {
        let adm = single(Admission::new(limits(1, 1, 4)));
        let held = adm
            .acquire("a", Priority::Normal, far(), Target::Any(0), 1)
            .await
            .unwrap();
        let deadline = Instant::now() + Duration::from_millis(30);
        let err = adm
            .acquire("b", Priority::Normal, deadline, Target::Any(0), 1)
            .await
            .unwrap_err();
        assert_eq!(err, AdmissionError::DeadlineExceeded);
        assert_eq!(adm.snapshot().queued, 0);
        // The expired waiter is gone: releasing the slot leaves it free rather than granting a ghost.
        drop(held);
        assert_eq!(
            adm.snapshot(),
            AdmissionSnapshot {
                inflight: 0,
                queued: 0,
                closed: false
            }
        );
        // An already-passed deadline never charges a slot.
        let past = Instant::now() - Duration::from_millis(1);
        assert_eq!(
            adm.acquire("b", Priority::Normal, past, Target::Any(0), 1)
                .await
                .unwrap_err(),
            AdmissionError::DeadlineExceeded
        );
        assert_eq!(adm.snapshot().inflight, 0);
    }

    #[tokio::test]
    async fn dropped_waiter_frees_queue_depth_and_permit_drop_releases() {
        let adm = single(Admission::new(limits(1, 1, 1)));
        let held = adm
            .acquire("a", Priority::Normal, far(), Target::Any(0), 1)
            .await
            .unwrap();
        let mut waiter = Box::pin(adm.acquire("b", Priority::Normal, far(), Target::Any(0), 1));
        assert!(futures_util::poll!(&mut waiter).is_pending());
        drop(waiter);
        assert_eq!(adm.snapshot().queued, 0);
        let mut next = Box::pin(adm.acquire("b", Priority::Normal, far(), Target::Any(0), 1));
        assert!(futures_util::poll!(&mut next).is_pending());
        drop(held);
        let next = next.await.unwrap();
        assert_eq!(adm.snapshot().inflight, 1);
        drop(next);
        assert_eq!(adm.snapshot().inflight, 0);
    }

    #[tokio::test]
    async fn orphan_charge_holds_slot_beyond_cap_and_close_rejects() {
        let adm = single(Admission::new(limits(1, 1, 4)));
        let orphan_a = adm.charge_orphan("a", None);
        let orphan_b = adm.charge_orphan("b", None);
        assert_eq!(adm.snapshot().inflight, 2);
        let mut fresh = Box::pin(adm.acquire("c", Priority::Normal, far(), Target::Any(0), 1));
        assert!(futures_util::poll!(&mut fresh).is_pending());
        drop(orphan_a);
        // Still at the cap (1 of 1) after one orphan settles.
        assert!(futures_util::poll!(&mut fresh).is_pending());
        drop(orphan_b);
        let fresh = fresh.await.unwrap();
        let mut queued = Box::pin(adm.acquire("d", Priority::Normal, far(), Target::Any(0), 1));
        assert!(futures_util::poll!(&mut queued).is_pending());
        adm.close();
        assert_eq!(queued.await.unwrap_err(), AdmissionError::Closed);
        assert_eq!(
            adm.acquire("e", Priority::Normal, far(), Target::Any(0), 1)
                .await
                .unwrap_err(),
            AdmissionError::Closed
        );
        assert_eq!(
            adm.snapshot(),
            AdmissionSnapshot {
                inflight: 1,
                queued: 0,
                closed: true
            }
        );
        drop(fresh);
        assert_eq!(adm.snapshot().inflight, 0);
    }

    #[test]
    fn deadline_budget_defaults_and_clamps() {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../configs/runtime/sglang-t4-qwen38.toml"
        ))
        .unwrap();
        let api: ApiConfig = toml::from_str::<toml::Value>(&text).unwrap()["api"]
            .clone()
            .try_into()
            .unwrap();
        assert_eq!(deadline_budget_ms(None, &api), api.default_deadline_ms);
        assert_eq!(deadline_budget_ms(Some(5), &api), 5);
        assert_eq!(
            deadline_budget_ms(Some(u64::MAX), &api),
            api.max_deadline_ms
        );
    }

    /// Q5 (review M3-05): behind a waiter pinned to a full worker, an unpinned waiter of the same scope, and one of
    /// another scope, are granted on the other worker at once.
    #[tokio::test]
    async fn waiters_behind_a_pin_to_a_full_worker_are_granted_elsewhere_at_once() {
        let adm = two(limits(4, 4, 8), 1, 2);
        let on_a = granted(&adm, "s", pinned("a"));
        assert_eq!(on_a.worker(), "a");
        let mut stuck = waiter(&adm, "s", Priority::Normal, pinned("a"));
        assert!(futures_util::poll!(&mut stuck).is_pending());
        let same_scope = granted(&adm, "s", Target::Any(0));
        let other_scope = granted(&adm, "t", Target::Any(0));
        assert_eq!((same_scope.worker(), other_scope.worker()), ("b", "b"));
        assert!(futures_util::poll!(&mut stuck).is_pending());
        drop(on_a);
        assert_eq!(stuck.await.unwrap().worker(), "a");
    }

    /// Q5 (review M3-05): a high-priority waiter pinned to a full worker does not block a normal waiter of its
    /// scope that is grantable on another worker.
    #[tokio::test]
    async fn high_priority_pin_to_a_full_worker_does_not_block_a_grantable_normal_waiter() {
        let adm = two(limits(4, 4, 8), 1, 1);
        let on_a = granted(&adm, "s", pinned("a"));
        let on_b = granted(&adm, "s", pinned("b"));
        let mut high = waiter(&adm, "s", Priority::High, pinned("a"));
        assert!(futures_util::poll!(&mut high).is_pending());
        let mut normal = waiter(&adm, "s", Priority::Normal, Target::Any(0));
        assert!(futures_util::poll!(&mut normal).is_pending());
        drop(on_b);
        let normal = normal.await.unwrap();
        assert_eq!(normal.worker(), "b");
        assert!(futures_util::poll!(&mut high).is_pending());
        drop(on_a);
        assert_eq!(high.await.unwrap().worker(), "a");
        drop(normal);
    }

    /// Q5, Q8: a pin to an eligible worker is strict: the waiter waits for its full worker while the other one
    /// has room.
    #[tokio::test]
    async fn strict_pin_waits_for_its_worker() {
        let adm = two(limits(4, 4, 8), 1, 4);
        let on_a = granted(&adm, "s", pinned("a"));
        let mut next = waiter(&adm, "s", Priority::Normal, pinned("a"));
        assert!(futures_util::poll!(&mut next).is_pending());
        assert_eq!(
            loads(&adm, "s"),
            [("a".to_owned(), 1, 1, 1), ("b".to_owned(), 4, 0, 0)]
        );
        drop(on_a);
        assert_eq!(next.await.unwrap().worker(), "a");
    }

    /// Ported from `routing.rs`: a session stays on its eligible worker despite load; unpinned work goes to the
    /// least-loaded worker.
    #[tokio::test]
    async fn pin_holds_despite_load_and_unpinned_work_goes_least_loaded() {
        let adm = two(limits(8, 8, 8), 4, 4);
        let b1 = granted(&adm, "s", pinned("b"));
        let b2 = granted(&adm, "s", pinned("b"));
        assert_eq!((b1.worker(), b2.worker()), ("b", "b"));
        assert_eq!(granted(&adm, "s", Target::Any(0)).worker(), "a");
    }

    /// Ported from `routing.rs`: a draining, suspect, joining, lost or unknown pinned worker gets no new work;
    /// the waiter re-routes, also while it waits for its full worker.
    #[tokio::test]
    async fn ineligible_pin_re_routes() {
        let adm = two(limits(8, 8, 8), 4, 1);
        let seen = unix_ms();
        for b in [
            WorkerView {
                draining: true,
                ..view(WorkerHealth::Ready, seen)
            },
            view(WorkerHealth::Suspect, seen),
            view(WorkerHealth::Joining, seen),
            view(WorkerHealth::Lost, seen),
        ] {
            adm.update_worker("b", b);
            assert_eq!(granted(&adm, "s", pinned("b")).worker(), "a");
        }
        assert_eq!(granted(&adm, "s", pinned("gone")).worker(), "a");
        adm.update_worker("b", ready());
        let on_b = granted(&adm, "s", pinned("b"));
        let mut next = waiter(&adm, "s", Priority::Normal, pinned("b"));
        assert!(futures_util::poll!(&mut next).is_pending());
        adm.update_worker("b", view(WorkerHealth::Suspect, unix_ms()));
        assert_eq!(next.await.unwrap().worker(), "a");
        drop(on_b);
    }

    /// Ported from `routing.rs`: workers not seen healthy for `suspect_after_ms` take no work until a fresh
    /// sighting.
    #[tokio::test]
    async fn aged_workers_take_no_work_until_seen_again() {
        let aging = Aging {
            suspect_after_ms: 1_000,
            lost_after_ms: 60_000,
        };
        let adm = with_workers(
            Admission::new(limits(4, 4, 4)),
            &[("a", 1), ("b", 1)],
            aging,
        );
        let stale = view(WorkerHealth::Ready, unix_ms() - 5_000);
        adm.update_worker("a", stale);
        adm.update_worker("b", stale);
        let mut next = waiter(&adm, "s", Priority::Normal, Target::Any(0));
        assert!(futures_util::poll!(&mut next).is_pending());
        adm.update_worker("b", ready());
        assert_eq!(next.await.unwrap().worker(), "b");
    }

    /// Q5: among the allowed workers with room, the fewest in flight wins, ties by worker id.
    #[tokio::test]
    async fn least_in_flight_then_lowest_id() {
        let adm = two(limits(8, 8, 8), 4, 4);
        let first = granted(&adm, "s", Target::Any(0));
        let second = granted(&adm, "s", Target::Any(0));
        let third = granted(&adm, "t", Target::Any(0));
        assert_eq!(
            [first.worker(), second.worker(), third.worker()],
            ["a", "b", "a"]
        );
        let extra_b = granted(&adm, "s", pinned("b"));
        drop(first);
        assert_eq!(granted(&adm, "s", Target::Any(0)).worker(), "a");
        drop((second, third, extra_b));
    }

    /// Q5 (review M3-01): a smoke (`Only`) goes to its worker whatever that worker's state except lost - joining
    /// next to a ready or a full worker, suspect - and never to another worker; unpinned work never lands on a
    /// joining worker.
    #[tokio::test]
    async fn smoke_runs_only_on_its_worker_in_any_state_but_lost() {
        let adm = two(limits(4, 4, 8), 1, 1);
        let seen = unix_ms();
        let at = |health, epoch| WorkerView {
            epoch,
            ..view(health, seen)
        };
        adm.update_worker("b", at(WorkerHealth::Joining, 2));
        // `a` is ready and free: the smoke still goes to `b`.
        let smoke = granted(&adm, "smoke", only("b"));
        assert_eq!((smoke.worker(), smoke.epoch()), ("b", 2));
        // `b` is full: a second smoke waits although `a` is free; unpinned work takes `a`.
        let mut second = waiter(&adm, "smoke", Priority::Normal, only("b"));
        assert!(futures_util::poll!(&mut second).is_pending());
        let on_a = granted(&adm, "s", Target::Any(0));
        assert_eq!(on_a.worker(), "a");
        // `a` is full: unpinned work waits rather than land on the joining `b`.
        let mut any = waiter(&adm, "s", Priority::Normal, Target::Any(0));
        assert!(futures_util::poll!(&mut any).is_pending());
        drop(smoke);
        let second = second.await.unwrap();
        assert_eq!(second.worker(), "b");
        assert!(futures_util::poll!(&mut any).is_pending());
        drop(second);
        assert!(futures_util::poll!(&mut any).is_pending());
        adm.update_worker("b", at(WorkerHealth::Suspect, 2));
        assert_eq!(granted(&adm, "smoke", only("b")).worker(), "b");
        // Lost: the smoke waits until `b` re-registers at a new epoch.
        adm.update_worker("b", at(WorkerHealth::Lost, 2));
        let mut third = waiter(&adm, "smoke", Priority::Normal, only("b"));
        assert!(futures_util::poll!(&mut third).is_pending());
        adm.update_worker("b", at(WorkerHealth::Joining, 3));
        let third = third.await.unwrap();
        assert_eq!((third.worker(), third.epoch()), ("b", 3));
        drop(on_a);
        assert_eq!(any.await.unwrap().worker(), "a");
    }

    /// Q7 (review M3-04): a waiter pinned to a full worker whose probes stop is granted on the other worker once
    /// the pin ages to suspect, with no other admission event and before its deadline. On the paused tokio clock:
    /// only the armed aging timer moves time, so the pin is fresh at the first poll however slow the host.
    #[tokio::test(start_paused = true)]
    async fn aging_deadline_moves_a_pinned_waiter_off_a_silent_worker() {
        let aging = Aging {
            suspect_after_ms: 2_000,
            lost_after_ms: 60_000,
        };
        let adm = with_workers(
            Admission::new(limits(4, 4, 8)).on_tokio_clock(),
            &[("a", 1), ("b", 1)],
            aging,
        );
        let start = Instant::now();
        // `a` was last seen 1.8 s ago: it reads suspect 200 ms from now.
        adm.update_worker("a", seen_ago(&adm, WorkerHealth::Ready, 1_800));
        let on_a = granted(&adm, "s", pinned("a"));
        let deadline = start + Duration::from_millis(1_500);
        let mut moved = Box::pin(adm.acquire("s", Priority::Normal, deadline, pinned("a"), 1));
        assert!(futures_util::poll!(&mut moved).is_pending());
        let moved = moved.await.unwrap();
        assert_eq!(moved.worker(), "b");
        assert!(start.elapsed() >= Duration::from_millis(200));
        drop(on_a);
    }

    /// M3.5 Q7: each worker ages by its own `WorkerCap::aging`: a remote worker (suspect 5 s, lost a further
    /// 10 s) takes no new work 6 s after its last sighting and is lost (not even a smoke) at 16 s, while a local
    /// worker on the M3 rule (suspect 15 s, lost a further 60 s) seen as long ago is still eligible at 6 s and
    /// only suspect at 16 s.
    #[tokio::test(start_paused = true)]
    async fn each_worker_ages_by_its_own_rule() {
        let remote = Aging {
            suspect_after_ms: 5_000,
            lost_after_ms: 10_000,
        };
        let m3 = Aging {
            suspect_after_ms: 15_000,
            lost_after_ms: 60_000,
        };
        let adm = Admission::new(limits(8, 8, 8)).on_tokio_clock();
        adm.set_workers(vec![
            WorkerCap {
                id: "a".into(),
                cap: 4,
                aging: remote,
                deployment: 0,
            },
            WorkerCap {
                id: "b".into(),
                cap: 4,
                aging: m3,
                deployment: 0,
            },
        ]);
        let seen = |ago: u64| {
            let view = seen_ago(&adm, WorkerHealth::Ready, ago);
            adm.update_worker("a", view);
            adm.update_worker("b", view);
        };
        seen(4_000);
        assert_eq!(granted(&adm, "s", pinned("a")).worker(), "a");
        seen(6_000);
        assert_eq!(granted(&adm, "s", pinned("a")).worker(), "b");
        assert_eq!(granted(&adm, "smoke", only("a")).worker(), "a");
        seen(16_000);
        let mut smoke = waiter(&adm, "smoke", Priority::Normal, only("a"));
        assert!(futures_util::poll!(&mut smoke).is_pending());
        assert_eq!(granted(&adm, "smoke", only("b")).worker(), "b");
        let mut any = waiter(&adm, "s", Priority::Normal, Target::Any(0));
        assert!(futures_util::poll!(&mut any).is_pending());
    }

    /// M3.5 Q7: a worker whose view carries `lease_until_ms` is eligible only before it; the lease deadline
    /// re-runs dispatch, so a waiter pinned to the full leased worker moves to another worker when the lease
    /// runs out, with no other admission event and before its deadline. On the paused tokio clock, like
    /// `aging_deadline_moves_a_pinned_waiter_off_a_silent_worker`.
    #[tokio::test(start_paused = true)]
    async fn lease_bounds_eligibility() {
        let adm = with_workers(
            Admission::new(limits(4, 4, 8)).on_tokio_clock(),
            &[("a", 1), ("b", 1)],
            NEVER,
        );
        let leased = |until_ms| WorkerView {
            lease_until_ms: Some(until_ms),
            ..seen_ago(&adm, WorkerHealth::Ready, 0)
        };
        adm.update_worker("a", leased(adm.now_ms() + 60_000));
        assert_eq!(granted(&adm, "s", pinned("a")).worker(), "a");
        adm.update_worker("a", leased(adm.now_ms()));
        assert_eq!(granted(&adm, "s", pinned("a")).worker(), "b");
        let start = Instant::now();
        adm.update_worker("a", leased(adm.now_ms() + 200));
        let on_a = granted(&adm, "s", pinned("a"));
        assert_eq!(on_a.worker(), "a");
        let deadline = start + Duration::from_millis(1_500);
        let mut moved = Box::pin(adm.acquire("s", Priority::Normal, deadline, pinned("a"), 1));
        assert!(futures_util::poll!(&mut moved).is_pending());
        let moved = moved.await.unwrap();
        assert_eq!(moved.worker(), "b");
        assert!(start.elapsed() >= Duration::from_millis(200));
        drop(on_a);
    }

    /// M3.5 Q11: `set_cap` changes a worker's in-flight cap at run time: a higher cap grants the waiters it makes
    /// room for, a lower one keeps charged slots and holds new grants back until in-flight drops below it;
    /// unknown ids are ignored.
    #[tokio::test]
    async fn set_cap_changes_a_workers_capacity() {
        let adm = with_workers(Admission::new(limits(8, 8, 8)), &[("w", 2)], NEVER);
        let first = granted(&adm, "s", Target::Any(0));
        let second = granted(&adm, "s", Target::Any(0));
        let mut third = waiter(&adm, "s", Priority::Normal, Target::Any(0));
        assert!(futures_util::poll!(&mut third).is_pending());
        adm.set_cap("w", 3);
        let third = third.await.unwrap();
        adm.set_cap("w", 1);
        assert_eq!(loads(&adm, "s"), vec![("w".to_owned(), 1, 3, 3)]);
        drop((first, second));
        let mut fourth = waiter(&adm, "s", Priority::Normal, Target::Any(0));
        assert!(futures_util::poll!(&mut fourth).is_pending());
        drop(third);
        let fourth = fourth.await.unwrap();
        adm.set_cap("gone", 5);
        assert_eq!(loads(&adm, "s"), vec![("w".to_owned(), 1, 1, 1)]);
        drop(fourth);
    }

    /// Q6, Q9: an orphan charged to a worker counts against that worker and the global cap; a global-only charge
    /// (no known worker) counts against the global cap alone. Both may exceed the caps.
    #[tokio::test]
    async fn orphan_charges_count_per_worker_or_globally() {
        let adm = two(limits(4, 4, 8), 1, 1);
        let orphan = adm.charge_orphan("s", Some("a"));
        assert_eq!(orphan.worker(), "a");
        let mut on_a = waiter(&adm, "s", Priority::Normal, pinned("a"));
        assert!(futures_util::poll!(&mut on_a).is_pending());
        assert_eq!(granted(&adm, "s", Target::Any(0)).worker(), "b");
        drop(orphan);
        assert_eq!(on_a.await.unwrap().worker(), "a");

        let adm = two(limits(2, 2, 8), 2, 2);
        let global = adm.charge_orphan("t", None);
        assert_eq!(global.worker(), "");
        let first = granted(&adm, "s", Target::Any(0));
        assert_eq!(first.worker(), "a");
        // The global cap is full although both workers have room.
        let mut next = waiter(&adm, "s", Priority::Normal, Target::Any(0));
        assert!(futures_util::poll!(&mut next).is_pending());
        assert_eq!(
            loads(&adm, "t"),
            [("a".to_owned(), 2, 1, 0), ("b".to_owned(), 2, 0, 0)]
        );
        drop(global);
        let next = next.await.unwrap();
        assert_eq!(next.worker(), "b");
        let beyond = [
            adm.charge_orphan("s", Some("b")),
            adm.charge_orphan("s", Some("b")),
        ];
        assert_eq!(adm.snapshot().inflight, 4);
        assert_eq!(
            loads(&adm, "s"),
            [("a".to_owned(), 2, 1, 1), ("b".to_owned(), 2, 3, 3)]
        );
        drop((beyond, first, next));
        assert_eq!(adm.snapshot().inflight, 0);
    }

    /// Q7: `mark_suspect` keeps a worker out of new grants until an update with a newer sighting.
    #[tokio::test]
    async fn mark_suspect_holds_until_a_newer_sighting() {
        let adm = two(limits(8, 8, 8), 4, 4);
        let seen = unix_ms();
        adm.update_worker("a", view(WorkerHealth::Ready, seen));
        adm.mark_suspect("a");
        assert_eq!(granted(&adm, "s", pinned("a")).worker(), "b");
        adm.update_worker("a", view(WorkerHealth::Ready, seen));
        assert_eq!(granted(&adm, "s", Target::Any(0)).worker(), "b");
        adm.update_worker("a", view(WorkerHealth::Ready, seen + 1));
        assert_eq!(granted(&adm, "s", Target::Any(0)).worker(), "a");
    }

    /// Q7: a grant whose `begin_work` was rejected is re-queued: its worker turns suspect, its slot frees, and the
    /// same waiter is granted on another worker ahead of a later waiter.
    #[tokio::test]
    async fn requeue_moves_the_waiter_off_the_rejected_worker_ahead_of_later_waiters() {
        let adm = two(limits(4, 4, 8), 1, 1);
        adm.update_worker(
            "b",
            WorkerView {
                epoch: 5,
                ..ready()
            },
        );
        let first = granted(&adm, "s", Target::Any(0));
        assert_eq!((first.worker(), first.epoch()), ("a", 1));
        let on_b = granted(&adm, "s", Target::Any(0));
        let mut later = waiter(&adm, "s", Priority::Normal, Target::Any(0));
        assert!(futures_util::poll!(&mut later).is_pending());
        let mut again = Box::pin(adm.requeue(first, far()));
        assert!(futures_util::poll!(&mut again).is_pending());
        assert_eq!(
            adm.snapshot(),
            AdmissionSnapshot {
                inflight: 1,
                queued: 2,
                closed: false
            }
        );
        drop(on_b);
        let again = again.await.unwrap();
        assert_eq!((again.worker(), again.epoch()), ("b", 5));
        assert!(futures_util::poll!(&mut later).is_pending());
    }

    fn xorshift(state: &mut u64, n: u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state % n
    }

    /// Q6: under seeded random interleavings of arrivals (pinned, targeted, unpinned; every priority), releases,
    /// health updates and suspect marks, no worker exceeds its cap, no scope its cap, the total never exceeds the
    /// global cap, and every waiter is served once everything is released.
    #[tokio::test]
    async fn worker_scope_and_global_caps_hold_under_random_interleavings() {
        const GLOBAL: u64 = 5;
        const PER_SCOPE: u64 = 3;
        let caps = [("a", 2), ("b", 4)];
        for seed in 1..=8u64 {
            let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let adm = with_workers(Admission::new(limits(GLOBAL, PER_SCOPE, 12)), &caps, NEVER);
            let mut held: Vec<Permit> = Vec::new();
            let mut pending: Vec<Acquire> = Vec::new();
            // Sightings only move forward, so an update after `mark_suspect` clears the mark.
            let mut seen = unix_ms();
            for _ in 0..500 {
                // A fresh task poll per step: tokio's cooperative budget would otherwise hold back ready grants.
                tokio::task::yield_now().await;
                match xorshift(&mut rng, 6) {
                    0 | 1 => {
                        let scope = ["s", "t", "u"][xorshift(&mut rng, 3) as usize];
                        let priority = [Priority::High, Priority::Normal, Priority::Low]
                            [xorshift(&mut rng, 3) as usize];
                        let target = match xorshift(&mut rng, 4) {
                            0 => pinned("a"),
                            1 => pinned("b"),
                            2 => only("b"),
                            _ => Target::Any(0),
                        };
                        pending.push(waiter(&adm, scope, priority, target));
                    }
                    2 if !held.is_empty() => {
                        let i = xorshift(&mut rng, held.len() as u64) as usize;
                        held.swap_remove(i);
                    }
                    3 => {
                        let id = caps[xorshift(&mut rng, 2) as usize].0;
                        let health = [
                            WorkerHealth::Ready,
                            WorkerHealth::Suspect,
                            WorkerHealth::Joining,
                        ][xorshift(&mut rng, 3) as usize];
                        seen += 1;
                        adm.update_worker(id, view(health, seen));
                    }
                    4 => adm.mark_suspect(caps[xorshift(&mut rng, 2) as usize].0),
                    _ => {}
                }
                let mut waiting = Vec::with_capacity(pending.len());
                for mut w in pending.drain(..) {
                    match w.as_mut().now_or_never() {
                        Some(Ok(permit)) => held.push(permit),
                        Some(Err(e)) => assert_eq!(e, AdmissionError::QueueFull),
                        None => waiting.push(w),
                    }
                }
                pending = waiting;
                assert!(adm.snapshot().inflight <= GLOBAL);
                for (id, cap) in caps {
                    let on = held.iter().filter(|p| p.worker() == id).count() as u64;
                    assert!(on <= cap, "seed {seed}: {on} on {id}");
                }
                for l in adm.worker_loads("s") {
                    assert!(l.inflight <= l.cap, "seed {seed}: {l:?}");
                }
                for scope in ["s", "t", "u"] {
                    let n = held.iter().filter(|p| p.scope() == scope).count() as u64;
                    assert!(n <= PER_SCOPE && adm.scope_inflight(scope) <= PER_SCOPE);
                }
            }
            seen += 1;
            for (id, _) in caps {
                adm.update_worker(id, view(WorkerHealth::Ready, seen));
            }
            while !pending.is_empty() {
                tokio::task::yield_now().await;
                held.clear();
                let before = pending.len();
                let mut waiting = Vec::with_capacity(before);
                for mut w in pending.drain(..) {
                    match w.as_mut().now_or_never() {
                        Some(result) => held.push(result.unwrap()),
                        None => waiting.push(w),
                    }
                }
                assert!(
                    waiting.len() < before,
                    "seed {seed}: no waiter served: {:?} {:?}",
                    adm.snapshot(),
                    adm.worker_loads("s")
                );
                pending = waiting;
            }
            held.clear();
            assert_eq!(
                adm.snapshot(),
                AdmissionSnapshot {
                    inflight: 0,
                    queued: 0,
                    closed: false
                }
            );
        }
    }

    /// Ready workers `(id, cap, deployment)` that never age (M5 D6).
    fn deployed(limits: AdmissionLimits, workers: &[(&str, u64, usize)]) -> Admission {
        let adm = Admission::new(limits);
        adm.set_workers(
            workers
                .iter()
                .map(|&(id, cap, deployment)| WorkerCap {
                    id: id.to_owned(),
                    cap,
                    aging: NEVER,
                    deployment,
                })
                .collect(),
        );
        for &(id, _, _) in workers {
            adm.update_worker(id, ready());
        }
        adm
    }

    #[test]
    fn any_grants_only_workers_of_its_deployment() {
        // `a0` and `a1` serve deployment 0, `b0` deployment 1; least-loaded runs inside the deployment.
        let adm = deployed(
            limits(16, 16, 16),
            &[("a0", 4, 0), ("a1", 4, 0), ("b0", 4, 1)],
        );
        let first = granted(&adm, "s", Target::Any(1));
        assert_eq!(first.worker(), "b0");
        let mut held = Vec::new();
        for _ in 0..4 {
            held.push(granted(&adm, "s", Target::Any(0)));
        }
        let mut workers: Vec<&str> = held.iter().map(Permit::worker).collect();
        workers.sort_unstable();
        assert_eq!(workers, ["a0", "a0", "a1", "a1"]);
        // `b0` has room but serves the other deployment: deployment 1 is never granted on `a*`.
        held.push(granted(&adm, "s", Target::Any(1)));
        assert_eq!(held.last().unwrap().worker(), "b0");
    }

    #[tokio::test]
    async fn per_deployment_caps_bind_per_worker() {
        // Unequal entry caps (A 1, B 3): A's second request waits although B's worker has room.
        let adm = deployed(limits(16, 16, 16), &[("a0", 1, 0), ("b0", 3, 1)]);
        let _a = granted(&adm, "s", Target::Any(0));
        let mut a2 = waiter(&adm, "s", Priority::Normal, Target::Any(0));
        assert!(a2.as_mut().now_or_never().is_none());
        let b: Vec<Permit> = (0..3).map(|_| granted(&adm, "s", Target::Any(1))).collect();
        assert!(b.iter().all(|p| p.worker() == "b0"));
        let mut b4 = waiter(&adm, "s", Priority::Normal, Target::Any(1));
        assert!(b4.as_mut().now_or_never().is_none());
        assert_eq!(
            loads(&adm, "s"),
            [("a0".to_owned(), 1, 1, 1), ("b0".to_owned(), 3, 3, 3)]
        );
    }

    #[test]
    fn a_pin_outside_the_deployment_repins_inside_it() {
        // The session's owner `a0` (deployment 0) is Ready with room, but the turn is for deployment 1.
        let adm = deployed(limits(16, 16, 16), &[("a0", 4, 0), ("b0", 4, 1)]);
        let permit = granted(&adm, "s", Target::Pinned(Arc::from("a0"), 1));
        assert_eq!(permit.worker(), "b0");
        // Inside its deployment the pin stays strict.
        let permit = granted(&adm, "s", Target::Pinned(Arc::from("a0"), 0));
        assert_eq!(permit.worker(), "a0");
    }

    /// Two native replicas `a` and `b` (m32n design note D16: cap 4 each) with a 200,000-token pool each.
    fn cards() -> Admission {
        let adm = two(limits(8, 8, 32), 4, 4);
        adm.set_pool("a", 200_000);
        adm.set_pool("b", 200_000);
        adm
    }

    fn route(tokens: u64, prefer: Option<&str>) -> Route {
        Route {
            tokens,
            prefer: prefer.map(Arc::from),
            class: None,
        }
    }

    fn routed(adm: &Admission, target: Target, route: Route) -> Acquire {
        let adm = adm.clone();
        Box::pin(async move {
            adm.acquire_routed("s", Priority::Normal, far(), target, 1, route)
                .await
        })
    }

    fn granted_routed(adm: &Admission, target: Target, route: Route) -> Permit {
        routed(adm, target, route)
            .now_or_never()
            .expect("granted at once")
            .unwrap()
    }

    /// `(worker, in-flight, reserved pool tokens)` per worker.
    fn pools(adm: &Admission) -> Vec<(String, u64, u64)> {
        adm.worker_loads("s")
            .into_iter()
            .map(|l| (l.id.to_string(), l.inflight, l.reserved))
            .collect()
    }

    /// R6/R8 (D6, D11): a card whose pool cannot hold the reservation is not eligible even when it has the fewest
    /// requests in flight; the reservation is released with the permit.
    #[test]
    fn a_card_whose_pool_cannot_hold_the_reservation_is_not_eligible() {
        let adm = cards();
        let big = granted_routed(&adm, Target::Any(0), route(200_000, None));
        assert_eq!(big.worker(), "a");
        let b1 = granted_routed(&adm, Target::Any(0), route(1_024, None));
        let b2 = granted_routed(&adm, Target::Any(0), route(1_024, None));
        assert_eq!((b1.worker(), b2.worker()), ("b", "b"));
        // Fewest in flight is `a` (1 < 2), but its pool is full.
        let short = granted_routed(&adm, Target::Any(0), route(64, None));
        assert_eq!(short.worker(), "b");
        assert_eq!(
            pools(&adm),
            [("a".to_owned(), 1, 200_000), ("b".to_owned(), 3, 2_112)]
        );
        drop(big);
        assert_eq!(pools(&adm)[0], ("a".to_owned(), 0, 0));
    }

    /// R6 (D6, D7): two 200K requests run one per card; a third queues (never truncated) and is granted when one
    /// finishes, on that card.
    #[tokio::test]
    async fn two_full_reservations_run_one_per_card_and_a_third_queues() {
        let adm = cards();
        let first = granted_routed(&adm, Target::Any(0), route(200_000, None));
        let second = granted_routed(&adm, Target::Any(0), route(200_000, None));
        assert_eq!((first.worker(), second.worker()), ("a", "b"));
        let mut third = routed(&adm, Target::Any(0), route(200_000, None));
        assert!(futures_util::poll!(&mut third).is_pending());
        drop(second);
        let third = third.await.unwrap();
        assert_eq!(third.worker(), "b");
    }

    /// R8 (D11 (3)): least-loaded is fewest in flight, then most free pool tokens, then the worker id.
    #[test]
    fn least_loaded_ties_go_to_the_most_free_pool_then_the_id() {
        let adm = cards();
        let a = granted_routed(&adm, Target::Any(0), route(100_000, None));
        let b = granted_routed(&adm, Target::Any(0), route(10_000, None));
        assert_eq!((a.worker(), b.worker()), ("a", "b"));
        // One in flight each: `b` has more free pool.
        let next = granted_routed(&adm, Target::Any(0), route(64, None));
        assert_eq!(next.worker(), "b");
    }

    /// R8 (D11 (2)): the preferred (affinity) card wins over least-loaded when it is eligible; a preferred card
    /// without room falls back to least-loaded; a pin always wins over a preference.
    #[test]
    fn the_preferred_card_wins_only_when_eligible_and_never_over_a_pin() {
        let adm = cards();
        let held = granted_routed(&adm, Target::Any(0), route(64, Some("b")));
        assert_eq!(held.worker(), "b");
        // `b` has 1 in flight, `a` 0: the preference still wins.
        let again = granted_routed(&adm, Target::Any(0), route(64, Some("b")));
        assert_eq!(again.worker(), "b");
        let full = granted_routed(&adm, Target::Any(0), route(199_872, Some("b")));
        assert_eq!(full.worker(), "b");
        // `b`'s pool is full: the preference is ignored.
        let spill = granted_routed(&adm, Target::Any(0), route(64, Some("b")));
        assert_eq!(spill.worker(), "a");
        let pin = granted_routed(&adm, pinned("a"), route(64, Some("b")));
        assert_eq!(pin.worker(), "a");
    }

    /// R8 (D11): a pinned request whose card has no pool room waits for that card (strict pin), even though the
    /// other card is free.
    #[tokio::test]
    async fn a_pin_waits_for_its_own_pool() {
        let adm = cards();
        let big = granted_routed(&adm, pinned("a"), route(200_000, None));
        assert_eq!(big.worker(), "a");
        let mut turn = routed(&adm, pinned("a"), route(64, None));
        assert!(futures_util::poll!(&mut turn).is_pending());
        drop(big);
        assert_eq!(turn.await.unwrap().worker(), "a");
    }

    /// D11 "never stuck": a reservation no allowed card can ever hold is never the head, so it does not hold the
    /// queue (it waits out its deadline); requests behind it are granted.
    #[tokio::test]
    async fn a_reservation_no_card_can_ever_hold_never_blocks_the_queue() {
        let adm = cards();
        let mut huge = routed(&adm, Target::Any(0), route(200_001, None));
        assert!(futures_util::poll!(&mut huge).is_pending());
        let short = granted_routed(&adm, Target::Any(0), route(64, None));
        assert_eq!(short.worker(), "a");
    }

    /// The single native form and every other worker: an untracked pool (0) ignores reservations, so the grant
    /// order is the M3.1 one (fewest in flight, then id).
    #[test]
    fn an_untracked_pool_ignores_reservations() {
        let adm = two(limits(8, 8, 32), 4, 4);
        let a = granted_routed(&adm, Target::Any(0), route(500_000, None));
        let b = granted_routed(&adm, Target::Any(0), route(500_000, None));
        let c = granted_routed(&adm, Target::Any(0), route(1, None));
        assert_eq!((a.worker(), b.worker(), c.worker()), ("a", "b", "a"));
    }
}
