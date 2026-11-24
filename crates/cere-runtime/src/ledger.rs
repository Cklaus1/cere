//! Bounded request ledger (m1 design note L1, L2 outcomes, L6 transitions, L7 status) and the durable owner of
//! sessions (S1–S3) and recovery (L3).
//!
//! Identity is (auth scope, request id). The request id is the backend rid the caller generated (unique per
//! process incarnation); it is the only handle for transitions. An optional idempotency key indexes
//! (scope, key) -> rid (Q2: absent key -> no dedupe). A request that ends never sent to the backend (of this
//! incarnation before `running`; with the dispatch journaled, any without a `dispatched` record) releases its
//! key, so a retry runs as new (journaled as `key_released`).
//!
//! States (FAILURE_MODEL.md): `accepted → queued → running → completed | failed`; `accepted | queued →
//! cancelled` (no backend call); `running → cancel_pending → cancelled | completed | failed`. Any non-terminal
//! state may fail (e.g. `worker_restart`, shutdown before dispatch). Exactly one terminal per request.
//!
//! Durability: `admitted` is journaled at [`Ledger::accept`], before the caller may reach the backend; every
//! terminal is journaled (with the session revision it commits, atomically). `queued`/`running`/
//! `cancel_pending` are memory-only: after a restart every non-terminal record is failed `worker_restart` and
//! returned as an [`Orphan`] whose rid must be aborted. Session leases are memory-only for the same reason.
//! A terminal that cannot be journaled still ends the request in memory (`failed: journal_error`) and poisons
//! the journal, so every later write and admission fails until a restart recovers from disk
//! ([`Ledger::is_journal_poisoned`]).
//!
//! Workers (m3 design note Q4, Q9-Q11): each worker id has an epoch, monotonic across restarts (journaled
//! `worker_epoch`; every configured local worker registers at its highest + 1 on open, a remote worker only at the
//! epoch its agent registers, [`Ledger::accept_worker_epoch`], M3.5 Q4). With configured workers an admission is
//! journaled `dispatch_journaled` and [`Ledger::dispatched`] journals the worker immediately before the backend
//! call, so recovery knows where each orphan's backend work may be ([`OrphanWorker`]). Operator drain intent is
//! journaled per worker id and survives every compaction ([`Ledger::set_drain`], M3.5 Q13), as is operator
//! residency intent ([`Ledger::set_residency`], m36 design note Q8).
//!
//! Session owners (m4 design note Q2, Q3, Q5-Q7): a session's owner worker and generation are durable. A grant
//! that changes the owner (the lease holder's turn on another worker) rides on that grant's `dispatched` record
//! ([`Ledger::dispatched`]), an operator migration on a `session_owner` record ([`Ledger::migrate_session`]);
//! each applies in memory only after its write. A request records the owner generation it holds from
//! [`Ledger::accept`] (the lease lock) and its own grant-time change; its commit is refused
//! `session_owner_superseded` once the session's generation moved past it ([`Ledger::held_owner`]).
//!
//! Bounds: at most `max_records` records (non-terminal plus terminal within `terminal_retention_ms`, the
//! documented client retry horizon). A full ledger rejects new requests with [`LedgerError::Overloaded`] before
//! admission, unless `evict_terminal_when_full` (ops-hardening design note OH-03) lets it evict its oldest
//! terminal record first, dropped as retention would drop it (its id is then unknown and its key loses idempotent
//! replay), and never an in-flight record or one still holding its slot. Eviction writes only records every
//! earlier release replays: a keyed victim is journaled as `key_expired` before the new admission (replay drops the
//! terminal record holding that key), an unkeyed one not at all (replay evicts down to the bound again). An expired
//! or evicted keyed record leaves a tombstone (bounded FIFO of `max_records` entries) so a late
//! retry gets `idempotency_key_expired` instead of regenerating. Retained unary bodies are memory-only and bounded
//! in total bytes; the oldest are dropped first (their requests then answer `response_not_retained`).
//!
//! All methods take `now_ms` (wall-clock Unix milliseconds) explicitly; retention survives restarts because
//! records carry their timestamps.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use crate::api_types::{CacheReport, RequestUsage};
use crate::config::{parse_loopback_base_url, RuntimeConfig};
use crate::digest::{random_bytes, Digest, DigestKeys, Salt};
use crate::journal::{
    AdmittedRecord, DispatchedRecord, Journal, Node, OrphanHeldRecord, OrphanReleaseCause,
    OrphanReleasedRecord, OwnerRef, Record, Residency, SessionAdvance, SessionOwnerRecord,
    SessionRef, TerminalRecord, WorkerEpochRecord,
};
use crate::labels::RequestLabels;
use crate::pool::Class;
use crate::session::{ForkParent, Revision, SessionError, SessionLimits, SessionStore};

/// Journal lines allowed beyond twice the last compacted size before the next write compacts.
pub const DEFAULT_COMPACTION_SLACK: u64 = 1024;

/// Terminal reason recorded for requests of a previous process incarnation.
pub const REASON_WORKER_RESTART: &str = "worker_restart";

/// Terminal reason of an in-flight request whose worker's epoch was superseded (M3 Q10). Like
/// [`REASON_WORKER_RESTART`], an unreleased one is charged again by a restart until its original bound.
pub const REASON_WORKER_EPOCH_SUPERSEDED: &str = "worker_epoch_superseded";

/// Terminal reason (and client code) of a session turn whose owner generation was superseded by a migration
/// after its grant (m4 design note Q6, Q7): its commit is refused, or it was cancelled by a forced takeover.
pub const REASON_SESSION_OWNER_SUPERSEDED: &str = "session_owner_superseded";

/// Terminal reason recorded in memory when the terminal record could not be journaled (the journal is then
/// poisoned); equals [`LedgerError::Journal`]'s code.
pub const REASON_JOURNAL_ERROR: &str = "journal_error";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LedgerLimits {
    pub max_records: u64,
    pub terminal_retention_ms: u64,
    /// Total bytes of retained unary response bodies.
    pub max_retained_bytes: u64,
    pub compaction_slack: u64,
    pub sessions: SessionLimits,
    /// A full ledger evicts its oldest terminal record to admit a new request (`ledger.evict_terminal_when_full`).
    pub evict_terminal_when_full: bool,
}

impl LedgerLimits {
    /// Ledger bounds from the runtime config. The retained-body budget is `api.max_request_body_bytes` (the
    /// config has no dedicated knob).
    pub fn from_config(config: &RuntimeConfig) -> Self {
        Self {
            max_records: config.ledger.max_records,
            terminal_retention_ms: config.ledger.terminal_retention_ms,
            max_retained_bytes: config.api.max_request_body_bytes,
            compaction_slack: DEFAULT_COMPACTION_SLACK,
            sessions: SessionLimits::from_config(&config.ledger),
            evict_terminal_when_full: config.ledger.evict_terminal_when_full(),
        }
    }
}

/// Occupancy of the ledger ([`Ledger::ledger_stats`]): `live` non-terminal records, `terminal` records still
/// retained, the `capacity` (`max_records`) both share, and the terminal records evicted by this incarnation to
/// admit new requests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LedgerStats {
    pub live: u64,
    pub terminal: u64,
    pub capacity: u64,
    pub evictions: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestState {
    Accepted,
    Queued,
    Running,
    CancelPending,
    Completed,
    Failed,
    Cancelled,
}

impl RequestState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Queued => "queued",
            Self::Running => "running",
            Self::CancelPending => "cancel_pending",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalState {
    Completed,
    Failed,
    Cancelled,
}

impl From<TerminalState> for RequestState {
    fn from(t: TerminalState) -> Self {
        match t {
            TerminalState::Completed => Self::Completed,
            TerminalState::Failed => Self::Failed,
            TerminalState::Cancelled => Self::Cancelled,
        }
    }
}

/// L7 timings in milliseconds since acceptance (`None` = not reached), except `queue_wait_ms`: the time the request
/// spent in the `queued` state less `encode_ms` (beat-sglang design note OQ4), so the host stages before it
/// (request digest, the `admitted` journal write and its fsync, a wait for the ledger lock) and the native pre-grant
/// encode stay out of it; they still count in every other timing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timings {
    pub queue_wait_ms: Option<u64>,
    /// The pre-grant encode stage of a pool-routed native request (its encode-slot wait included); `None` for
    /// every other request. Journaled only when set, so a journal without native requests stays readable by an
    /// older `cered`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encode_ms: Option<u64>,
    pub first_generated_ms: Option<u64>,
    pub first_visible_ms: Option<u64>,
    pub total_ms: Option<u64>,
    pub client_backpressure_ms: u64,
}

/// A retained unary response (memory-only, never journaled).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetainedResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

/// Session part of an admission (S2): the writer's model, its expected revision and the digest of
/// `messages[..count]` under the session's salt, where `count` is the expected revision's `message_count`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionClaim {
    pub session_id: String,
    /// The turn's model (M5 D7): a session bound to another refuses the turn; its commit binds an unbound one.
    pub model: String,
    pub expected_revision: u64,
    pub prefix_digest: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptRequest {
    pub scope: String,
    pub rid: String,
    pub key: Option<String>,
    pub digest: Digest,
    pub stream: bool,
    /// M2.14: the API surface; `None` for chat.
    pub surface: Option<String>,
    pub session: Option<SessionClaim>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcceptOutcome {
    /// New record in state `accepted`, journaled; the caller proceeds to admission.
    Accepted,
    /// L2: same key and digest, unary, body retained → replay it; nothing is executed.
    Replay {
        rid: String,
        response: Arc<RetainedResponse>,
    },
}

/// Inputs of a successful completion.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Completion {
    /// Unary response to retain for replay (ignored for streamed requests, which are never replayable).
    pub retained: Option<RetainedResponse>,
    /// New session revision (S2): `message_count` and digest of the committed messages incl. the assistant turn.
    pub session_commit: Option<Revision>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CancelOutcome {
    /// Was accepted/queued: now `cancelled` with no backend call; the caller stops waiting for admission.
    Cancelled,
    /// Was running: now `cancel_pending`; the caller must `abort_request(rid)` and settle the terminal.
    AbortRequired,
    /// Already `cancel_pending` (idempotent repeat).
    AlreadyPending,
    /// Already terminal (idempotent repeat).
    AlreadyTerminal(RequestState),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerError {
    /// L1 bound reached (503 retryable, before admission).
    Overloaded,
    /// L2: terminal, body not retained (409; status URL from `rid`).
    ResponseNotRetained {
        rid: String,
    },
    /// L2: not terminal yet (409).
    RequestInProgress {
        rid: String,
    },
    /// L2: the original was streamed (409).
    StreamNotReplayable {
        rid: String,
    },
    /// L2: the keyed record outlived its retention (409; never regenerated).
    IdempotencyKeyExpired,
    /// L2: same key, different digest (409).
    IdempotencyConflict,
    /// Unknown rid, or a rid of another scope.
    NotFound,
    /// The transition is not allowed from the current state.
    InvalidTransition {
        from: RequestState,
        to: RequestState,
    },
    /// A second terminal for the same request.
    AlreadyTerminal {
        state: RequestState,
    },
    /// The rid is already recorded.
    DuplicateRid,
    /// Q10: the dispatch named a worker epoch the ledger already superseded; the request was failed
    /// `worker_epoch_superseded` instead and no backend saw it.
    WorkerEpochSuperseded,
    /// M3.5 Q4: a registered epoch not above the worker's journaled one; nothing was journaled.
    StaleWorkerEpoch {
        journaled: u64,
    },
    Session(SessionError),
    /// Journal I/O or corruption; the operation had no effect.
    Journal(String),
}

impl LedgerError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Overloaded => "ledger_full",
            Self::ResponseNotRetained { .. } => "response_not_retained",
            Self::RequestInProgress { .. } => "request_in_progress",
            Self::StreamNotReplayable { .. } => "stream_not_replayable",
            Self::IdempotencyKeyExpired => "idempotency_key_expired",
            Self::IdempotencyConflict => "idempotency_conflict",
            Self::NotFound => "request_not_found",
            Self::InvalidTransition { .. } => "invalid_transition",
            Self::AlreadyTerminal { .. } => "already_terminal",
            Self::DuplicateRid => "duplicate_request_id",
            Self::WorkerEpochSuperseded => REASON_WORKER_EPOCH_SUPERSEDED,
            Self::StaleWorkerEpoch { .. } => "stale_epoch",
            Self::Session(e) => e.code(),
            Self::Journal(_) => "journal_error",
        }
    }
}

impl From<SessionError> for LedgerError {
    fn from(e: SessionError) -> Self {
        Self::Session(e)
    }
}

impl From<io::Error> for LedgerError {
    fn from(e: io::Error) -> Self {
        Self::Journal(e.to_string())
    }
}

/// L7 status view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestStatus {
    pub scope: String,
    pub rid: String,
    pub key: Option<String>,
    pub state: RequestState,
    pub reason: Option<String>,
    /// Always 1 in M1 (no retries).
    pub attempt: u32,
    /// Process epoch (journal incarnation) the request was admitted under.
    pub epoch: u64,
    pub stream: bool,
    /// M2.14: the API surface; `None` for chat.
    pub surface: Option<String>,
    pub session_id: Option<String>,
    pub timings: Timings,
    /// What the backend reported (Q1/Q5); unavailable until then.
    pub usage: RequestUsage,
    /// Worker of the request's last `dispatched` record (M3 Q11); `None` before dispatch and for pre-M3 entries.
    pub worker_id: Option<String>,
    /// That worker's epoch at dispatch; `None` with `worker_id`.
    pub worker_epoch: Option<u64>,
    /// A2: the request's client and workload ([`Ledger::set_labels`]); `None` before and for replayed entries.
    pub labels: Option<RequestLabels>,
    /// Team pools (net-teleport design note 4.4, C17): the request's class ([`Ledger::set_class`]); `None`
    /// before and for replayed entries.
    pub class: Option<Class>,
}

/// Per-scope terminal counters since the ledger was opened (process start), for `GET /v1/cere/stats` (Q5).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScopeStats {
    /// Terminals by state; every terminal state is present.
    pub requests: BTreeMap<&'static str, u64>,
    /// Sum of reported `prompt_tokens`.
    pub prompt_tokens_total: u64,
    /// Sum of `cached_tokens` over requests with `cache_report: "reported"`.
    pub cached_tokens_total: u64,
    pub cache_reported_requests: u64,
    pub cache_unavailable_requests: u64,
    /// m32n design note D18: the same terminals by the worker they were dispatched to (never-dispatched ones
    /// are not counted here). Not serialized: `GET /v1/cere/stats` reports them under its own `workers.<id>`.
    #[serde(skip)]
    pub workers: BTreeMap<String, CardStats>,
}

/// m32n design note D18: one scope's terminals on one worker.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct CardStats {
    pub requests: u64,
    /// Sum of reported `completion_tokens`.
    pub completion_tokens: u64,
    /// Sum of reported `cached_tokens`.
    pub cached_tokens: u64,
}

impl Default for ScopeStats {
    fn default() -> Self {
        Self {
            requests: [
                TerminalState::Completed,
                TerminalState::Failed,
                TerminalState::Cancelled,
            ]
            .into_iter()
            .map(|t| (RequestState::from(t).as_str(), 0))
            .collect(),
            prompt_tokens_total: 0,
            cached_tokens_total: 0,
            cache_reported_requests: 0,
            cache_unavailable_requests: 0,
            workers: BTreeMap::new(),
        }
    }
}

impl ScopeStats {
    fn count(&mut self, state: TerminalState, usage: &RequestUsage, worker: Option<&str>) {
        *self
            .requests
            .entry(RequestState::from(state).as_str())
            .or_default() += 1;
        self.prompt_tokens_total += usage.prompt_tokens.unwrap_or(0);
        self.cached_tokens_total += usage.cached_tokens.unwrap_or(0);
        match usage.cache_report {
            CacheReport::Reported => self.cache_reported_requests += 1,
            CacheReport::Unavailable => self.cache_unavailable_requests += 1,
        }
        if let Some(worker) = worker {
            let card = self.workers.entry(worker.to_owned()).or_default();
            card.requests += 1;
            card.completion_tokens += usage.completion_tokens.unwrap_or(0);
            card.cached_tokens += usage.cached_tokens.unwrap_or(0);
        }
    }
}

/// Where a worker's backend is reached (M3.5 Q11): `cered`'s own loopback backend at `base_url`, or a federated
/// agent at `endpoint`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerTarget {
    Local { base_url: String },
    Remote { endpoint: String },
}

impl WorkerTarget {
    fn local_url(&self) -> Option<&str> {
        match self {
            Self::Local { base_url } => Some(base_url),
            Self::Remote { .. } => None,
        }
    }
}

/// A request of a previous incarnation (failed `worker_restart` on open) or a held terminal charged again after a
/// restart; [`Orphan::worker`] says where its rid must be aborted and which cap holds its slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Orphan {
    pub scope: String,
    pub rid: String,
    pub epoch: u64,
    /// When the `admitted` record was written: the start of the backend slot's retention bound (L3).
    pub admitted_ms: u64,
    /// `(worker id, worker epoch)` of the rid's last `dispatched` record (the Q11 `worker=<id>@<epoch>` log);
    /// `None` when it was never dispatched or admitted before M3.
    pub dispatched: Option<(String, u64)>,
    /// Where its backend work may be, given the workers configured at the open (M3 Q9).
    pub worker: OrphanWorker,
}

impl Orphan {
    fn of(rid: &str, e: &Entry, configured: &BTreeMap<String, WorkerTarget>) -> Self {
        let worker = match &e.worker {
            Some(w) => match &w.target {
                WorkerTarget::Local { base_url } => {
                    match configured.get(&w.id).and_then(WorkerTarget::local_url) {
                        Some(url) if same_base_url(url, base_url) => OrphanWorker::Known {
                            id: w.id.clone(),
                            base_url: url.to_owned(),
                        },
                        _ => OrphanWorker::Unconfigured {
                            base_url: base_url.clone(),
                        },
                    }
                }
                WorkerTarget::Remote { endpoint } => OrphanWorker::Remote {
                    id: w.id.clone(),
                    endpoint: endpoint.clone(),
                },
            },
            None if e.dispatch_journaled => OrphanWorker::NotDispatched,
            None => OrphanWorker::Legacy,
        };
        Self {
            scope: e.scope.clone(),
            rid: rid.to_owned(),
            epoch: e.epoch,
            admitted_ms: e.accepted_ms,
            dispatched: e.worker.as_ref().map(|w| (w.id.clone(), w.epoch)),
            worker,
        }
    }
}

/// Where a recovered orphan's backend work may be (M3 Q9), from its last `dispatched` record and the workers
/// configured at the open.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrphanWorker {
    /// Dispatched to a configured local worker that serves the journaled `base_url` (normalised; this is the
    /// configured one): aborted on, charged to and load-polled against that worker.
    Known { id: String, base_url: String },
    /// Dispatched to a local worker id that is not configured as local, or whose configured `base_url` differs
    /// from the journaled one: charged to no worker (global cap) until its retention bound and aborted on every
    /// configured local worker serving the journaled `base_url` (none if no worker does).
    Unconfigured { base_url: String },
    /// Dispatched to remote worker `id` at the journaled `endpoint` (M3.5 Q15): while `id` is configured as remote
    /// at that endpoint, charged to and aborted (`CancelWork`) on that worker until its agent reports the operation
    /// `cleanup_acknowledged` ([`Ledger::release_remote_orphan`]) or its retention bound; otherwise charged to no
    /// worker (global cap) until its retention bound and aborted nowhere.
    Remote { id: String, endpoint: String },
    /// Admitted before M3 (no `dispatch_journaled`): with one configured local worker, that worker's orphan as
    /// before; otherwise charged to no worker (global cap) until its retention bound and aborted on every local
    /// worker.
    Legacy,
    /// An M3 admission without a `dispatched` record never reached a backend: released at once
    /// ([`OrphanReleaseCause::NotDispatched`]), never held or aborted.
    NotDispatched,
}

/// Which admission cap holds a recovered orphan's slot (M3 Q9).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrphanCharge {
    /// That worker's cap (and the global cap).
    Worker(String),
    /// The global cap only.
    Global,
    /// Nothing holds a slot: the request never reached a backend.
    NotHeld,
}

impl OrphanWorker {
    /// The cap holding the orphan's slot until its release, given the configured workers.
    pub fn charged_to(&self, configured: &BTreeMap<String, WorkerTarget>) -> OrphanCharge {
        match self {
            Self::Known { id, .. } => OrphanCharge::Worker(id.clone()),
            Self::Remote { id, endpoint } if serves_endpoint(configured, id, endpoint) => {
                OrphanCharge::Worker(id.clone())
            }
            Self::Legacy => {
                let mut ids = local_ids(configured);
                match (ids.next(), ids.next()) {
                    (Some(only), None) => OrphanCharge::Worker(only.clone()),
                    _ => OrphanCharge::Global,
                }
            }
            Self::Unconfigured { .. } | Self::Remote { .. } => OrphanCharge::Global,
            Self::NotDispatched => OrphanCharge::NotHeld,
        }
    }

    /// Configured worker ids that get the orphan's abort (a local worker's `/abort_request`, a remote worker's
    /// `CancelWork`), in id order.
    pub fn abort_targets(&self, configured: &BTreeMap<String, WorkerTarget>) -> Vec<String> {
        match self {
            Self::Known { id, .. } => vec![id.clone()],
            Self::Remote { id, endpoint } if serves_endpoint(configured, id, endpoint) => {
                vec![id.clone()]
            }
            Self::Remote { .. } | Self::NotDispatched => Vec::new(),
            Self::Unconfigured { base_url } => configured
                .iter()
                .filter(|(_, t)| {
                    t.local_url()
                        .is_some_and(|url| same_base_url(url, base_url))
                })
                .map(|(id, _)| id.clone())
                .collect(),
            Self::Legacy => local_ids(configured).cloned().collect(),
        }
    }
}

fn local_ids(configured: &BTreeMap<String, WorkerTarget>) -> impl Iterator<Item = &String> {
    configured
        .iter()
        .filter(|(_, t)| t.local_url().is_some())
        .map(|(id, _)| id)
}

/// `id` is configured as a remote worker at `endpoint` (equal as socket addresses when both parse, else as
/// strings).
fn serves_endpoint(configured: &BTreeMap<String, WorkerTarget>, id: &str, endpoint: &str) -> bool {
    match configured.get(id) {
        Some(WorkerTarget::Remote { endpoint: current }) => {
            match (
                current.parse::<SocketAddr>(),
                endpoint.parse::<SocketAddr>(),
            ) {
                (Ok(a), Ok(b)) => a == b,
                _ => current == endpoint,
            }
        }
        _ => false,
    }
}

/// Equal after normalisation: `parse_loopback_base_url` admits only a literal host and an explicit port, so its
/// `(host, port)` is the normal form. A URL it rejects equals nothing.
fn same_base_url(a: &str, b: &str) -> bool {
    match (parse_loopback_base_url(a), parse_loopback_base_url(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

#[derive(Debug)]
pub struct Recovery {
    pub epoch: u64,
    /// Sorted by rid (fixed-width rids sort chronologically).
    pub orphans: Vec<Orphan>,
    /// Terminals `worker_restart` (an earlier open's orphans) or `worker_epoch_superseded` (M3 Q10), and any
    /// terminal whose H2 hold on a remote worker was journaled (`orphan_held`, M3.5 Q15), whose slot was still
    /// held when that incarnation ended (no `orphan_released` record) and whose `admitted_ms +
    /// terminal_retention_ms` has not passed (m1close Q7): charged again until that original bound. Sorted by
    /// rid.
    pub recharged: Vec<Orphan>,
    pub truncated_tail_bytes: u64,
}

/// A non-terminal request (for shutdown drain, L8, and the Q10 supersede).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InFlight {
    pub scope: String,
    pub rid: String,
    pub state: RequestState,
    /// `(worker id, worker epoch)` of the request's last `dispatched` record; `None` before dispatch.
    pub worker: Option<(String, u64)>,
    /// When the `admitted` record was written: the start of the retention bound of an orphaned slot.
    pub admitted_ms: u64,
}

/// Session view for callers. `salt` is internal (needed to compute prefix digests); never return it to clients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionView {
    pub session_id: String,
    pub revision: u64,
    pub current: Revision,
    pub salt: Salt,
    pub parent: Option<ForkParent>,
    /// The durable owner (m4 design note Q3) at `owner_generation` (0 iff `None`).
    pub pinned_worker: Option<String>,
    pub owner_generation: u64,
    /// The bound model (M5 D7); `None` until the first commit.
    pub model: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForkView {
    pub child: SessionView,
    pub parent_session_id: String,
    pub parent_revision: u64,
}

/// The session owner a request holds (m4 design note Q2): its generation as of the request's `accept` (the lease
/// lock) or its own grant-time owner change, and that owner's worker while it is still the session's (`None` once
/// the generation was superseded or the session is gone, and for an unowned session).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeldOwner {
    pub worker: Option<String>,
    pub owner_generation: u64,
}

/// A session's owner for operators (m4 design note Q4 `GET /v1/cere/sessions/{id}/owner`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnerView {
    pub session_id: String,
    pub worker: Option<String>,
    pub owner_generation: u64,
    pub revision: u64,
}

/// What a migration may move a session to (m4 design note Q5 step 4): a configured, Ready worker, with the index
/// of the deployment it serves (M5 D8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerCheck {
    /// Not a configured worker id.
    Unknown,
    /// Configured, not Ready (joining, draining, suspect, lost).
    Ineligible(usize),
    Eligible(usize),
}

impl WorkerCheck {
    /// The worker's deployment index; `None` for an unknown worker.
    pub fn deployment(self) -> Option<usize> {
        match self {
            Self::Unknown => None,
            Self::Ineligible(d) | Self::Eligible(d) => Some(d),
        }
    }
}

/// An operator migration of a session's owner (m4 design note Q4, Q5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MigrateRequest<'a> {
    pub session_id: &'a str,
    pub to_worker: &'a str,
    /// The CAS: the session's owner generation the operator read.
    pub expected_owner_generation: u64,
    /// Q6: take over from an owner that is not eligible, breaking a held lease.
    pub force: bool,
}

/// A published migration (replay mode: the destination reconstructs from the next turn's messages).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Migrated {
    pub session_id: String,
    pub scope: String,
    pub worker: String,
    pub owner_generation: u64,
    pub revision: u64,
    /// The owner before the move.
    pub previous: Option<String>,
    /// Q6: the rid of the turn whose lease a forced takeover broke; the caller cancels it
    /// `session_owner_superseded`.
    pub superseded: Option<String>,
}

/// Why a migration was refused (m4 design note Q5 order); nothing changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrateError {
    /// Unknown or expired session (404 `session_not_found`).
    NotFound,
    /// The owner generation is not the expected one (409 `owner_generation_conflict`, with the current owner).
    Conflict {
        worker: Option<String>,
        owner_generation: u64,
    },
    /// `to_worker` is not a configured worker (404 `worker_not_found`).
    WorkerNotFound,
    /// `to_worker` serves another model than the session's (M5 D8; 409 `model_mismatch`).
    ModelMismatch,
    /// `to_worker` is not Ready (409 `worker_not_eligible`).
    WorkerNotEligible,
    /// `to_worker` already owns the session (409 `already_owner`).
    AlreadyOwner,
    /// `force` against an eligible owner (409 `owner_healthy`).
    OwnerHealthy,
    /// A turn holds the lease and `force` is not set (409 `session_busy`).
    SessionBusy,
    /// The `session_owner` record could not be written (500); the journal is not poisoned.
    Journal(LedgerError),
}

#[derive(Debug)]
struct Entry {
    scope: String,
    key: Option<String>,
    digest: Digest,
    epoch: u64,
    stream: bool,
    surface: Option<String>,
    session: Option<SessionRef>,
    state: RequestState,
    reason: Option<String>,
    accepted_ms: u64,
    /// When `mark_queued` moved it to `queued` (OQ4: its queue wait starts there). Memory-only: `None` before and
    /// for replayed entries.
    queued_ms: Option<u64>,
    terminal_ms: Option<u64>,
    timings: Timings,
    usage: RequestUsage,
    retained: Option<Arc<RetainedResponse>>,
    /// Set once a recovered orphan's slot was released (journaled `orphan_released`, m1close Q7).
    orphan_release: Option<(u64, OrphanReleaseCause)>,
    /// `(at_ms, remote worker id)` of the journaled start of an H2 hold of the request's slot (`orphan_held`,
    /// M3.5 Q15): while no release follows, a restart charges the slot again.
    orphan_held: Option<(u64, String)>,
    /// Admitted by an M3 ledger: without a `dispatched` record the request never reached a backend.
    dispatch_journaled: bool,
    /// Q11: the request's last `dispatched` record; `None` before dispatch and for pre-M3 entries.
    worker: Option<EntryWorker>,
    /// m4 design note Q2/Q7: the session owner generation this request holds (set at `accept` under the lease
    /// lock, raised by its own grant-time owner change); its commit needs it to still be the session's.
    /// Memory-only: replayed entries hold 0 and never commit (recovery ends them).
    held_generation: u64,
    /// The session turn's model (M5 D7), bound by its commit. Memory-only like `held_generation`.
    model: Option<String>,
    /// A2: the client and workload set at acceptance ([`Ledger::set_labels`]). Memory-only: replayed entries hold
    /// `None`.
    labels: Option<RequestLabels>,
    /// Team pools (C17): the class set at acceptance ([`Ledger::set_class`]). Memory-only like `labels`.
    class: Option<Class>,
}

impl Entry {
    fn since(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.accepted_ms)
    }

    /// OQ4: the queue wait of a request that leaves the queue at `now_ms`: the time since `mark_queued` (since its
    /// acceptance for one never queued, a request replayed from the journal included) less its pre-grant encode.
    fn queue_wait(&self, now_ms: u64) -> u64 {
        let since = now_ms.saturating_sub(self.queued_ms.unwrap_or(self.accepted_ms));
        since.saturating_sub(self.timings.encode_ms.unwrap_or(0))
    }

    /// A terminal record whose admission slot outlived its terminal and is not released yet: an earlier open's
    /// orphan, a superseded request or an H2 hold, charged until `accepted_ms + retention_ms`.
    fn holds_slot(&self, retention_ms: u64, now_ms: u64) -> bool {
        let held = self.orphan_held.is_some()
            || (self.state == RequestState::Failed
                && matches!(
                    self.reason.as_deref(),
                    Some(REASON_WORKER_RESTART | REASON_WORKER_EPOCH_SUPERSEDED)
                ));
        held && self.orphan_release.is_none()
            && self.accepted_ms.saturating_add(retention_ms) > now_ms
    }
}

/// A `dispatched` record without its rid.
#[derive(Debug)]
struct EntryWorker {
    id: String,
    epoch: u64,
    target: WorkerTarget,
    dispatched_ms: u64,
}

impl EntryWorker {
    /// The record's worker; its address fields must match its node (Q11: `base_url` iff local, `endpoint` iff
    /// remote).
    fn replayed(d: DispatchedRecord) -> Result<Self, String> {
        let target = match (d.node, d.base_url, d.endpoint) {
            (Node::Local, Some(base_url), None) => WorkerTarget::Local { base_url },
            (Node::Remote, None, Some(endpoint)) => WorkerTarget::Remote { endpoint },
            (Node::Local, ..) => {
                return Err(format!(
                    "dispatched rid {}: node local needs base_url and no endpoint",
                    d.rid
                ))
            }
            (Node::Remote, ..) => {
                return Err(format!(
                    "dispatched rid {}: node remote needs endpoint and no base_url",
                    d.rid
                ))
            }
        };
        Ok(Self {
            id: d.worker,
            epoch: d.worker_epoch,
            target,
            dispatched_ms: d.at_ms,
        })
    }

    fn record(&self, rid: &str) -> DispatchedRecord {
        let (node, base_url, endpoint) = match &self.target {
            WorkerTarget::Local { base_url } => (Node::Local, Some(base_url.clone()), None),
            WorkerTarget::Remote { endpoint } => (Node::Remote, None, Some(endpoint.clone())),
        };
        DispatchedRecord {
            rid: rid.to_owned(),
            worker: self.id.clone(),
            worker_epoch: self.epoch,
            node,
            base_url,
            endpoint,
            // A re-emitted record: the session snapshot carries the owner (m4 design note Q3c).
            session_owner: None,
            at_ms: self.dispatched_ms,
        }
    }
}

#[derive(Debug)]
pub struct Ledger {
    limits: LedgerLimits,
    keys: Arc<DigestKeys>,
    journal: Journal,
    entries: HashMap<String, Entry>,
    by_key: HashMap<(String, String), String>,
    /// (expiry time, rid) of terminal records in terminal order.
    expiry: VecDeque<(u64, String)>,
    tombstones: HashMap<(String, String), u64>,
    tombstone_order: VecDeque<(String, String)>,
    retained_order: VecDeque<String>,
    retained_bytes: u64,
    sessions: SessionStore,
    /// Terminals written by this incarnation, by scope (Q5); replayed records are not counted.
    stats: HashMap<String, ScopeStats>,
    /// When this incarnation opened the ledger.
    opened_ms: u64,
    /// Highest journaled epoch per worker id ever seen, with its record's time (Q4).
    worker_epochs: BTreeMap<String, (u64, u64)>,
    /// Workers were configured at the open: admissions are journaled `dispatch_journaled` (M3 Q9).
    dispatch_journaled: bool,
    /// Worker ids with operator drain intent, with the time of their `worker_drain` record (M3.5 Q13).
    drains: BTreeMap<String, u64>,
    /// Latest operator residency intent per worker id: `(desired, intent_seq, at_ms)` of its last
    /// `worker_residency` record (m36 design note Q8).
    residency: BTreeMap<String, (Residency, u64, u64)>,
    /// Terminal records evicted by this incarnation to admit new requests.
    evictions: u64,
    /// Terminal records in `entries`, kept with them for [`Ledger::ledger_stats`].
    terminal: u64,
}

impl Ledger {
    /// Opens the journal under `state_dir` (epoch + 1, fsync'd), rebuilds requests, sessions, worker epochs,
    /// drain and residency intents, applies retention at `now_ms`, registers each configured local worker (`workers`: id ->
    /// target) at its highest journaled epoch + 1 (default 1; Q4; a remote worker's epoch is only ever the one its
    /// agent registered, M3.5 Q4), fails + journals every prior-epoch non-terminal request as `worker_restart`,
    /// classifies orphans and re-charged terminals against `workers` (Q9, M3.5 Q15; an M3 request never
    /// dispatched is released at once, `not_dispatched`), and compacts the journal to the recovered live state,
    /// which makes the registrations durable before the open returns. With no configured workers nothing
    /// registers and admissions are journaled as before M3 (orphans of such a ledger are `Legacy`).
    pub fn open(
        limits: LedgerLimits,
        keys: Arc<DigestKeys>,
        state_dir: &Path,
        workers: &BTreeMap<String, WorkerTarget>,
        now_ms: u64,
    ) -> Result<(Self, Recovery), LedgerError> {
        let (journal, replay) = Journal::open(state_dir, now_ms)?;
        let mut ledger = Self {
            limits,
            keys,
            journal,
            entries: HashMap::new(),
            by_key: HashMap::new(),
            expiry: VecDeque::new(),
            tombstones: HashMap::new(),
            tombstone_order: VecDeque::new(),
            retained_order: VecDeque::new(),
            retained_bytes: 0,
            sessions: SessionStore::new(limits.sessions),
            stats: HashMap::new(),
            opened_ms: now_ms,
            worker_epochs: BTreeMap::new(),
            dispatch_journaled: !workers.is_empty(),
            drains: BTreeMap::new(),
            residency: BTreeMap::new(),
            evictions: 0,
            terminal: 0,
        };
        for (index, record) in replay.records.into_iter().enumerate() {
            ledger.replay(record).map_err(|why| {
                LedgerError::Journal(format!("journal record {}: {why}", index + 1))
            })?;
        }
        // `purge` stops at the first unexpired head, so the queue must be in expiry order; replay pushes in
        // file order, and compaction writes requests in acceptance order (review finding 5).
        ledger.expiry.make_contiguous().sort_unstable();
        ledger.purge(now_ms);
        ledger.shed(now_ms)?;
        for id in local_ids(workers) {
            let epoch = ledger.next_worker_epoch(id);
            ledger.worker_epochs.insert(id.clone(), (epoch, now_ms));
        }
        let retention_ms = ledger.limits.terminal_retention_ms;
        let mut orphans = Vec::new();
        let mut recharged = Vec::new();
        for (rid, e) in &ledger.entries {
            if !e.state.is_terminal() {
                orphans.push(Orphan::of(rid, e, workers));
                continue;
            }
            // Its slot outlived its terminal: an earlier open's orphan, a superseded request, or an H2 hold.
            if e.holds_slot(retention_ms, now_ms) {
                recharged.push(Orphan::of(rid, e, workers));
            }
        }
        orphans.sort_by(|a, b| a.rid.cmp(&b.rid));
        // A never-dispatched terminal whose release was lost (a crash inside the previous open) holds nothing.
        let (lost_release, mut recharged): (Vec<Orphan>, Vec<Orphan>) = recharged
            .into_iter()
            .partition(|o| o.worker == OrphanWorker::NotDispatched);
        recharged.sort_by(|a, b| a.rid.cmp(&b.rid));
        for orphan in &orphans {
            ledger.terminate(
                &orphan.rid,
                TerminalState::Failed,
                Some(REASON_WORKER_RESTART),
                None,
                None,
                now_ms,
            )?;
        }
        for orphan in orphans.iter().chain(&lost_release) {
            if orphan.worker == OrphanWorker::NotDispatched {
                ledger.release_orphan(&orphan.rid, OrphanReleaseCause::NotDispatched, now_ms)?;
            }
        }
        let live = ledger.live_records(now_ms);
        ledger.journal.compact(&live)?;
        let epoch = ledger.journal.epoch();
        Ok((
            ledger,
            Recovery {
                epoch,
                orphans,
                recharged,
                truncated_tail_bytes: replay.truncated_tail_bytes,
            },
        ))
    }

    fn replay(&mut self, record: Record) -> Result<(), String> {
        match record {
            Record::Epoch { .. } => {}
            Record::WorkerEpoch(w) => {
                if self
                    .worker_epochs
                    .get(&w.worker)
                    .is_some_and(|&(epoch, _)| epoch >= w.epoch)
                {
                    return Err(format!(
                        "worker {} epoch {} not increasing",
                        w.worker, w.epoch
                    ));
                }
                self.worker_epochs.insert(w.worker, (w.epoch, w.at_ms));
            }
            Record::Admitted(a) => {
                if self.entries.contains_key(&a.rid) {
                    return Err(format!("duplicate admitted rid {}", a.rid));
                }
                if let Some(key) = &a.key {
                    self.by_key
                        .insert((a.scope.clone(), key.clone()), a.rid.clone());
                }
                self.entries.insert(
                    a.rid,
                    Entry {
                        scope: a.scope,
                        key: a.key,
                        digest: a.digest,
                        epoch: a.epoch,
                        stream: a.stream,
                        surface: a.surface,
                        session: a.session,
                        state: RequestState::Accepted,
                        reason: None,
                        accepted_ms: a.at_ms,
                        queued_ms: None,
                        terminal_ms: None,
                        timings: Timings::default(),
                        usage: RequestUsage::UNAVAILABLE,
                        retained: None,
                        orphan_release: None,
                        orphan_held: None,
                        dispatch_journaled: a.dispatch_journaled,
                        worker: None,
                        // Replayed requests never commit (recovery ends them).
                        held_generation: 0,
                        model: None,
                        labels: None,
                        class: None,
                    },
                );
            }
            Record::Dispatched(d) => {
                let entry = self
                    .entries
                    .get_mut(&d.rid)
                    .ok_or_else(|| format!("dispatched for unknown rid {}", d.rid))?;
                if entry.state.is_terminal() {
                    return Err(format!("dispatched after the terminal of rid {}", d.rid));
                }
                // M4 Q3a: the grant's owner change applies if it is newer than the session's owner; a purged
                // session (compaction keeps request records whose session expired) ignores it.
                if let Some(o) = &d.session_owner {
                    if self
                        .sessions
                        .stored(&o.session_id)
                        .is_some_and(|s| o.owner_generation > s.owner_generation)
                    {
                        self.sessions
                            .set_owner(&o.session_id, &d.worker, o.owner_generation);
                    }
                }
                entry.worker = Some(EntryWorker::replayed(d)?);
            }
            Record::Terminal(t) => {
                let entry = self
                    .entries
                    .get_mut(&t.rid)
                    .ok_or_else(|| format!("terminal for unknown rid {}", t.rid))?;
                if entry.state.is_terminal() {
                    return Err(format!("second terminal for rid {}", t.rid));
                }
                entry.state = t.state.into();
                self.terminal += 1;
                entry.reason = t.reason;
                entry.timings = t.timings;
                // Absent in records written before usage was journaled.
                entry.usage = t.usage.unwrap_or(RequestUsage::UNAVAILABLE);
                entry.terminal_ms = Some(t.at_ms);
                if t.key_released {
                    if let Some(key) = entry.key.take() {
                        let index = (entry.scope.clone(), key);
                        if self.by_key.get(&index) == Some(&t.rid) {
                            self.by_key.remove(&index);
                        }
                    }
                }
                if let Some(adv) = t.session {
                    self.sessions
                        .apply_advance(
                            &adv.session_id,
                            adv.revision,
                            adv.entry,
                            adv.model.as_deref(),
                            t.at_ms,
                        )
                        .map_err(|e| format!("session advance {}: {}", adv.session_id, e.code()))?;
                }
                self.expiry.push_back((
                    t.at_ms.saturating_add(self.limits.terminal_retention_ms),
                    t.rid,
                ));
            }
            Record::Session(s) => self.sessions.insert_record(s),
            Record::SessionOwner(r) => {
                let s = self
                    .sessions
                    .stored(&r.session_id)
                    .ok_or_else(|| format!("session owner {}: session_not_found", r.session_id))?;
                if r.revision != s.revision() || r.digest != s.current().digest {
                    return Err(format!("session owner {}: revision mismatch", r.session_id));
                }
                if r.owner_generation > s.owner_generation {
                    self.sessions
                        .set_owner(&r.session_id, &r.worker, r.owner_generation);
                    self.sessions.touch(&r.session_id, r.at_ms);
                }
            }
            Record::KeyExpired { scope, key, at_ms } => {
                let index = (scope, key);
                match self.by_key.get(&index).cloned() {
                    // OH-03: a full ledger evicted the terminal record holding the key (`evict`).
                    Some(rid) if self.entries[&rid].state.is_terminal() => {
                        self.unexpire(&rid);
                        self.drop_record(&rid, at_ms);
                    }
                    Some(rid) => return Err(format!("key_expired for in-flight rid {rid}")),
                    None => self.add_tombstone(index.0, index.1, at_ms),
                }
            }
            Record::OrphanReleased(o) => {
                let entry = self
                    .entries
                    .get_mut(&o.rid)
                    .ok_or_else(|| format!("orphan_released for unknown rid {}", o.rid))?;
                entry.orphan_release = Some((o.at_ms, o.cause));
            }
            Record::OrphanHeld(o) => {
                let entry = self
                    .entries
                    .get_mut(&o.rid)
                    .ok_or_else(|| format!("orphan_held for unknown rid {}", o.rid))?;
                entry.orphan_held = Some((o.at_ms, o.worker));
            }
            Record::WorkerDrain {
                worker,
                draining,
                at_ms,
            } => {
                if draining {
                    self.drains.insert(worker, at_ms);
                } else {
                    self.drains.remove(&worker);
                }
            }
            Record::WorkerResidency {
                worker,
                desired,
                intent_seq,
                at_ms,
            } => {
                self.residency.insert(worker, (desired, intent_seq, at_ms));
            }
        }
        Ok(())
    }

    /// The epoch of this incarnation (the local worker's epoch).
    pub fn epoch(&self) -> u64 {
        self.journal.epoch()
    }

    pub fn digest_keys(&self) -> &Arc<DigestKeys> {
        &self.keys
    }

    pub fn limits(&self) -> &LedgerLimits {
        &self.limits
    }

    pub fn ledger_stats(&self) -> LedgerStats {
        debug_assert_eq!(
            self.terminal,
            self.entries
                .values()
                .filter(|e| e.state.is_terminal())
                .count() as u64,
            "maintained terminal count"
        );
        LedgerStats {
            live: self.entries.len() as u64 - self.terminal,
            terminal: self.terminal,
            capacity: self.limits.max_records,
            evictions: self.evictions,
        }
    }

    /// True once a journal write failure poisoned the journal: no new work may be admitted (fail closed).
    pub fn is_journal_poisoned(&self) -> bool {
        self.journal.is_poisoned()
    }

    /// Team pools (net-teleport design note 4.4, C18): per-scope live-session caps replacing
    /// `ledger.max_sessions_per_scope` for the scopes listed.
    pub fn set_session_scope_caps(&mut self, caps: HashMap<String, u64>) {
        self.sessions.set_scope_caps(caps);
    }

    /// The highest journaled epoch of every worker id ever seen (Q4), including workers not configured now.
    pub fn worker_epochs(&self) -> BTreeMap<String, u64> {
        self.worker_epochs
            .iter()
            .map(|(id, &(epoch, _))| (id.clone(), epoch))
            .collect()
    }

    /// The highest journaled epoch of `worker`; `None` for an id never journaled (a remote worker before its
    /// first accepted registration, M3.5 Q4).
    pub fn worker_epoch(&self, worker: &str) -> Option<u64> {
        self.worker_epochs.get(worker).map(|&(epoch, _)| epoch)
    }

    /// Journals `worker`'s next epoch (highest + 1, default 1) and returns it (Q4: a new `instance_id`, a rejoin
    /// after Lost). On a journal error the epoch is unchanged.
    pub fn bump_worker_epoch(&mut self, worker: &str, now_ms: u64) -> Result<u64, LedgerError> {
        let epoch = self.next_worker_epoch(worker);
        self.journal_worker_epoch(worker, epoch, now_ms)?;
        Ok(epoch)
    }

    /// M3.5 Q4: journals `epoch`, minted by `worker`'s agent and registered, as the worker's current epoch. An
    /// epoch not above the journaled one is refused ([`LedgerError::StaleWorkerEpoch`]) and nothing is
    /// journaled; on a journal error the epoch is unchanged.
    pub fn accept_worker_epoch(
        &mut self,
        worker: &str,
        epoch: u64,
        now_ms: u64,
    ) -> Result<(), LedgerError> {
        if let Some(journaled) = self.worker_epoch(worker).filter(|&j| epoch <= j) {
            return Err(LedgerError::StaleWorkerEpoch { journaled });
        }
        self.journal_worker_epoch(worker, epoch, now_ms)
    }

    fn journal_worker_epoch(
        &mut self,
        worker: &str,
        epoch: u64,
        now_ms: u64,
    ) -> Result<(), LedgerError> {
        self.write(
            &Record::WorkerEpoch(WorkerEpochRecord {
                worker: worker.to_owned(),
                epoch,
                at_ms: now_ms,
            }),
            now_ms,
        )?;
        self.worker_epochs
            .insert(worker.to_owned(), (epoch, now_ms));
        Ok(())
    }

    fn next_worker_epoch(&self, worker: &str) -> u64 {
        self.worker_epochs
            .get(worker)
            .map_or(1, |&(epoch, _)| epoch + 1)
    }

    /// M3.5 Q13: journals operator drain intent (`draining`) or its clearing (undrain) for any worker id, in any
    /// health and whether registered or not. A call that changes nothing journals nothing; on a journal error
    /// the intent is unchanged.
    pub fn set_drain(
        &mut self,
        worker: &str,
        draining: bool,
        now_ms: u64,
    ) -> Result<(), LedgerError> {
        if self.drains.contains_key(worker) == draining {
            return Ok(());
        }
        self.write(
            &Record::WorkerDrain {
                worker: worker.to_owned(),
                draining,
                at_ms: now_ms,
            },
            now_ms,
        )?;
        if draining {
            self.drains.insert(worker.to_owned(), now_ms);
        } else {
            self.drains.remove(worker);
        }
        Ok(())
    }

    /// Worker ids with journaled drain intent (M3.5 Q13), restored on open.
    pub fn drain_intents(&self) -> BTreeSet<String> {
        self.drains.keys().cloned().collect()
    }

    /// m36 design note Q8: journals a new residency intent for `worker` (each call is one; `intent_seq` is the
    /// worker's previous one + 1, 1 for the first), fsynced before the in-memory intent changes. On a journal
    /// error the intent is unchanged. Returns the new `intent_seq`.
    pub fn set_residency(
        &mut self,
        worker: &str,
        desired: Residency,
        now_ms: u64,
    ) -> Result<u64, LedgerError> {
        let intent_seq = self.residency.get(worker).map_or(0, |&(_, seq, _)| seq) + 1;
        self.write(
            &Record::WorkerResidency {
                worker: worker.to_owned(),
                desired,
                intent_seq,
                at_ms: now_ms,
            },
            now_ms,
        )?;
        self.residency
            .insert(worker.to_owned(), (desired, intent_seq, now_ms));
        Ok(intent_seq)
    }

    /// The latest residency intent per worker id as `(desired, intent_seq)` (m36 design note Q8), restored on
    /// open.
    pub fn residency_intents(&self) -> BTreeMap<String, (Residency, u64)> {
        self.residency
            .iter()
            .map(|(worker, &(desired, seq, _))| (worker.clone(), (desired, seq)))
            .collect()
    }

    /// L2 lookup + L1 bound + S2 claim, then journal `admitted` and record `accepted` (taking the session lease).
    pub fn accept(
        &mut self,
        req: AcceptRequest,
        now_ms: u64,
    ) -> Result<AcceptOutcome, LedgerError> {
        self.purge(now_ms);
        if let Some(key) = &req.key {
            let index = (req.scope.clone(), key.clone());
            if let Some(rid) = self.by_key.get(&index) {
                let e = &self.entries[rid];
                let rid = rid.clone();
                return if e.digest != req.digest {
                    Err(LedgerError::IdempotencyConflict)
                } else if !e.state.is_terminal() {
                    Err(LedgerError::RequestInProgress { rid })
                } else if e.stream {
                    Err(LedgerError::StreamNotReplayable { rid })
                } else if let Some(response) = &e.retained {
                    Ok(AcceptOutcome::Replay {
                        response: response.clone(),
                        rid,
                    })
                } else {
                    Err(LedgerError::ResponseNotRetained { rid })
                };
            }
            if self.tombstones.contains_key(&index) {
                return Err(LedgerError::IdempotencyKeyExpired);
            }
        }
        if self.entries.contains_key(&req.rid) {
            return Err(LedgerError::DuplicateRid);
        }
        // L1: a full ledger rejects, or (OH-03) names its oldest evictable terminal record, evicted only once the
        // session claim passed.
        let victim = if self.entries.len() as u64 >= self.limits.max_records {
            Some(
                self.eviction_candidate(now_ms)
                    .ok_or(LedgerError::Overloaded)?,
            )
        } else {
            None
        };
        if let Some(c) = &req.session {
            self.sessions.check_claim(
                &req.scope,
                &c.session_id,
                &c.model,
                c.expected_revision,
                &c.prefix_digest,
                now_ms,
            )?;
        }
        if let Some(rid) = victim {
            self.evict(&rid, now_ms)?;
            self.evictions += 1;
        }
        let session = req.session.as_ref().map(|c| SessionRef {
            session_id: c.session_id.clone(),
            expected_revision: c.expected_revision,
        });
        let epoch = self.journal.epoch();
        self.write(
            &Record::Admitted(AdmittedRecord {
                scope: req.scope.clone(),
                rid: req.rid.clone(),
                key: req.key.clone(),
                digest: req.digest,
                epoch,
                stream: req.stream,
                surface: req.surface.clone(),
                session: session.clone(),
                at_ms: now_ms,
                dispatch_journaled: self.dispatch_journaled,
            }),
            now_ms,
        )?;
        // Q2: the owner generation the request holds, read under the same lock that takes the lease.
        let mut held_generation = 0;
        if let Some(c) = &req.session {
            self.sessions.acquire(&c.session_id, &req.rid);
            held_generation = self
                .sessions
                .stored(&c.session_id)
                .map_or(0, |s| s.owner_generation);
        }
        if let Some(key) = &req.key {
            self.by_key
                .insert((req.scope.clone(), key.clone()), req.rid.clone());
        }
        self.entries.insert(
            req.rid,
            Entry {
                scope: req.scope,
                key: req.key,
                digest: req.digest,
                epoch,
                stream: req.stream,
                surface: req.surface,
                session,
                state: RequestState::Accepted,
                reason: None,
                accepted_ms: now_ms,
                queued_ms: None,
                terminal_ms: None,
                timings: Timings::default(),
                usage: RequestUsage::UNAVAILABLE,
                retained: None,
                orphan_release: None,
                orphan_held: None,
                dispatch_journaled: self.dispatch_journaled,
                worker: None,
                held_generation,
                model: req.session.map(|c| c.model),
                labels: None,
                class: None,
            },
        );
        Ok(AcceptOutcome::Accepted)
    }

    fn entry_mut(&mut self, rid: &str) -> Result<&mut Entry, LedgerError> {
        self.entries.get_mut(rid).ok_or(LedgerError::NotFound)
    }

    fn step(
        &mut self,
        rid: &str,
        from: RequestState,
        to: RequestState,
    ) -> Result<&mut Entry, LedgerError> {
        let e = self.entry_mut(rid)?;
        if e.state != from {
            return Err(LedgerError::InvalidTransition { from: e.state, to });
        }
        e.state = to;
        Ok(e)
    }

    /// `accepted → queued` (waiting for an admission permit) at `now_ms`, where its queue wait starts (OQ4).
    pub fn mark_queued(&mut self, rid: &str, now_ms: u64) -> Result<(), LedgerError> {
        let e = self.step(rid, RequestState::Accepted, RequestState::Queued)?;
        e.queued_ms = Some(now_ms);
        Ok(())
    }

    /// The pre-grant encode stage of the queued request `rid` took `ms` (OQ4): it is reported as `encode_ms` and
    /// left out of `queue_wait_ms`. Refused for a request no longer queued (it ended meanwhile; nothing to report).
    pub fn record_encode(&mut self, rid: &str, ms: u64) -> Result<(), LedgerError> {
        let e = self.entry_mut(rid)?;
        if e.state != RequestState::Queued {
            return Err(LedgerError::InvalidTransition {
                from: e.state,
                to: e.state,
            });
        }
        e.timings.encode_ms = Some(ms);
        Ok(())
    }

    /// `queued → running` (permit held; the backend call may start).
    pub fn mark_running(&mut self, rid: &str, now_ms: u64) -> Result<(), LedgerError> {
        let e = self.step(rid, RequestState::Queued, RequestState::Running)?;
        e.timings.queue_wait_ms = Some(e.queue_wait(now_ms));
        Ok(())
    }

    /// Q9: journals that the running (or cancel-pending) request `rid` is sent to `worker` at `worker_epoch` via
    /// `target` (a local `base_url` or, M3.5 Q11, a remote agent `endpoint`) next, immediately before the backend
    /// call, and records it as the request's worker (Q11). A journal error returns it with nothing changed: the
    /// caller fails the request before any backend sees it.
    /// m4 design note Q2/Q3a (m3 Q8): a session turn whose request holds the lease and is granted on another
    /// worker than the session's owner moves the owner there at the next generation; the change rides on this
    /// record (`session_owner`) and applies, with the request's held generation, only after the write.
    /// Q10: an epoch below the worker's journaled one was superseded (the worker was relaunched or rejoined
    /// after the grant): the request fails `worker_epoch_superseded` here and `WorkerEpochSuperseded` returns.
    pub fn dispatched(
        &mut self,
        rid: &str,
        worker: &str,
        worker_epoch: u64,
        target: &WorkerTarget,
        now_ms: u64,
    ) -> Result<(), LedgerError> {
        let e = self.entries.get(rid).ok_or(LedgerError::NotFound)?;
        if !matches!(e.state, RequestState::Running | RequestState::CancelPending) {
            return Err(LedgerError::InvalidTransition {
                from: e.state,
                to: e.state,
            });
        }
        if self
            .worker_epochs
            .get(worker)
            .is_some_and(|&(current, _)| worker_epoch < current)
        {
            // A journal error still settles the terminal in memory (and poisons the journal).
            let _ = self.fail(rid, REASON_WORKER_EPOCH_SUPERSEDED, None, now_ms);
            return Err(LedgerError::WorkerEpochSuperseded);
        }
        let dispatched = EntryWorker {
            id: worker.to_owned(),
            epoch: worker_epoch,
            target: target.clone(),
            dispatched_ms: now_ms,
        };
        let owner_change = e.session.as_ref().and_then(|s| {
            self.sessions
                .owner_change(&s.session_id, rid, worker)
                .map(|owner_generation| OwnerRef {
                    session_id: s.session_id.clone(),
                    owner_generation,
                })
        });
        let mut record = dispatched.record(rid);
        record.session_owner = owner_change.clone();
        self.write(&Record::Dispatched(record), now_ms)?;
        if let Some(o) = &owner_change {
            self.sessions
                .set_owner(&o.session_id, worker, o.owner_generation);
        }
        let e = self.entry_mut(rid)?;
        e.worker = Some(dispatched);
        if let Some(o) = owner_change {
            e.held_generation = o.owner_generation;
        }
        Ok(())
    }

    /// First token received from the backend (first call wins).
    pub fn note_first_generated(&mut self, rid: &str, now_ms: u64) -> Result<(), LedgerError> {
        let e = self.entry_mut(rid)?;
        if e.timings.first_generated_ms.is_none() {
            e.timings.first_generated_ms = Some(e.since(now_ms));
        }
        Ok(())
    }

    /// First token handed to the client (first call wins).
    pub fn note_first_visible(&mut self, rid: &str, now_ms: u64) -> Result<(), LedgerError> {
        let e = self.entry_mut(rid)?;
        if e.timings.first_visible_ms.is_none() {
            e.timings.first_visible_ms = Some(e.since(now_ms));
        }
        Ok(())
    }

    /// Adds time spent blocked on a full client stream buffer.
    pub fn add_backpressure(&mut self, rid: &str, ms: u64) -> Result<(), LedgerError> {
        let e = self.entry_mut(rid)?;
        e.timings.client_backpressure_ms = e.timings.client_backpressure_ms.saturating_add(ms);
        Ok(())
    }

    /// The backend's usage for a dispatched (`running`/`cancel_pending`) request, journaled with its terminal.
    pub fn record_usage(&mut self, rid: &str, usage: RequestUsage) -> Result<(), LedgerError> {
        let e = self.entry_mut(rid)?;
        if !matches!(e.state, RequestState::Running | RequestState::CancelPending) {
            return Err(LedgerError::InvalidTransition {
                from: e.state,
                to: e.state,
            });
        }
        e.usage = usage;
        Ok(())
    }

    /// Team pools (C17): the class `rid` is scheduled in (memory only, never journaled). Unknown rids are ignored.
    pub fn set_class(&mut self, rid: &str, class: Class) {
        if let Some(e) = self.entries.get_mut(rid) {
            e.class = Some(class);
        }
    }

    /// A2: labels `rid` with its client and workload (memory only, never journaled). Unknown rids are ignored.
    pub fn set_labels(&mut self, rid: &str, labels: RequestLabels) {
        if let Some(e) = self.entries.get_mut(rid) {
            e.labels = Some(labels);
        }
    }

    /// Scoped, idempotent cancel request (L6, L4 running expiry, L5 backpressure). `reason` is journaled only
    /// when the request is cancelled here (accepted/queued).
    pub fn request_cancel(
        &mut self,
        scope: &str,
        rid: &str,
        reason: &str,
        now_ms: u64,
    ) -> Result<CancelOutcome, LedgerError> {
        let state = match self.entries.get(rid) {
            Some(e) if e.scope == scope => e.state,
            _ => return Err(LedgerError::NotFound),
        };
        match state {
            RequestState::Accepted | RequestState::Queued => {
                self.cancel(rid, reason, now_ms)?;
                Ok(CancelOutcome::Cancelled)
            }
            RequestState::Running => {
                self.step(rid, RequestState::Running, RequestState::CancelPending)?;
                Ok(CancelOutcome::AbortRequired)
            }
            RequestState::CancelPending => Ok(CancelOutcome::AlreadyPending),
            terminal => Ok(CancelOutcome::AlreadyTerminal(terminal)),
        }
    }

    /// `running | cancel_pending → completed`, committing the session revision if any.
    pub fn complete(
        &mut self,
        rid: &str,
        completion: Completion,
        now_ms: u64,
    ) -> Result<(), LedgerError> {
        self.terminate(
            rid,
            TerminalState::Completed,
            None,
            completion.retained,
            completion.session_commit,
            now_ms,
        )
    }

    /// Any non-terminal state `→ failed` with a reason code. A unary failure body may be retained for replay.
    pub fn fail(
        &mut self,
        rid: &str,
        reason: &str,
        retained: Option<RetainedResponse>,
        now_ms: u64,
    ) -> Result<(), LedgerError> {
        self.terminate(
            rid,
            TerminalState::Failed,
            Some(reason),
            retained,
            None,
            now_ms,
        )
    }

    /// `accepted | queued | cancel_pending → cancelled` with a reason code (`running` must pass through
    /// `cancel_pending`).
    pub fn cancel(&mut self, rid: &str, reason: &str, now_ms: u64) -> Result<(), LedgerError> {
        self.terminate(
            rid,
            TerminalState::Cancelled,
            Some(reason),
            None,
            None,
            now_ms,
        )
    }

    /// Journals that a recovered orphan's admission slot was released (m1close Q7), so a later restart does not
    /// charge it again. A no-op for a rid that is gone (past retention) or already released. A failed write
    /// leaves the orphan unreleased on disk: a restart charges it again until its original bound.
    pub fn release_orphan(
        &mut self,
        rid: &str,
        cause: OrphanReleaseCause,
        now_ms: u64,
    ) -> Result<(), LedgerError> {
        if self
            .entries
            .get(rid)
            .is_none_or(|e| e.orphan_release.is_some())
        {
            return Ok(());
        }
        let record = Record::OrphanReleased(OrphanReleasedRecord {
            rid: rid.to_owned(),
            at_ms: now_ms,
            cause,
        });
        self.write(&record, now_ms)?;
        if let Some(e) = self.entries.get_mut(rid) {
            e.orphan_release = Some((now_ms, cause));
        }
        Ok(())
    }

    /// M3.5 Q15: journals the release (`cleanup_acknowledged`) of a terminal request whose last `dispatched`
    /// record names remote `worker`, once its agent reported the operation `cleanup_acknowledged`; the caller
    /// then drops the slot charged to that worker. Returns whether a release was journaled: `false` for a rid
    /// that is gone, not terminal, dispatched elsewhere (or never), or already released. A failed write leaves the
    /// orphan unreleased on disk (a restart charges it again until its original bound).
    pub fn release_remote_orphan(
        &mut self,
        rid: &str,
        worker: &str,
        now_ms: u64,
    ) -> Result<bool, LedgerError> {
        let held = self.entries.get(rid).is_some_and(|e| {
            e.state.is_terminal()
                && e.orphan_release.is_none()
                && e.worker.as_ref().is_some_and(|w| {
                    w.id == worker && matches!(w.target, WorkerTarget::Remote { .. })
                })
        });
        if !held {
            return Ok(false);
        }
        self.release_orphan(rid, OrphanReleaseCause::CleanupAcknowledged, now_ms)?;
        Ok(true)
    }

    /// M3.5 Q15 / H2: journals `orphan_held` when a hold of the slot of a request whose last `dispatched` record
    /// names remote `worker` starts (its agent may still hold the work), so a restart before the release charges
    /// that slot again until the request's original bound. The hold runs in its own task, started when the
    /// executor drops the slot, which is before or after the executor journals the request's terminal depending
    /// on the path (a work stream that fails before its first event drops the slot first; a relayed stream's
    /// reader drops it after handing the end to the executor): the record may precede the terminal, replay
    /// accepts both orders, and only a terminal request is charged again (a non-terminal one is an orphan of
    /// the next open anyway). Returns whether a hold was journaled: `false` for a rid that is gone, dispatched
    /// elsewhere (or never), already held or already released. A failed write leaves the hold unjournaled: a
    /// restart then does not charge the slot again.
    pub fn hold_remote_orphan(
        &mut self,
        rid: &str,
        worker: &str,
        now_ms: u64,
    ) -> Result<bool, LedgerError> {
        let holdable = self.entries.get(rid).is_some_and(|e| {
            e.orphan_held.is_none()
                && e.orphan_release.is_none()
                && e.worker.as_ref().is_some_and(|w| {
                    w.id == worker && matches!(w.target, WorkerTarget::Remote { .. })
                })
        });
        if !holdable {
            return Ok(false);
        }
        let record = Record::OrphanHeld(OrphanHeldRecord {
            rid: rid.to_owned(),
            worker: worker.to_owned(),
            at_ms: now_ms,
        });
        self.write(&record, now_ms)?;
        if let Some(e) = self.entries.get_mut(rid) {
            e.orphan_held = Some((now_ms, worker.to_owned()));
        }
        Ok(true)
    }

    fn terminate(
        &mut self,
        rid: &str,
        state: TerminalState,
        reason: Option<&str>,
        retained: Option<RetainedResponse>,
        commit: Option<Revision>,
        now_ms: u64,
    ) -> Result<(), LedgerError> {
        let e = self.entries.get(rid).ok_or(LedgerError::NotFound)?;
        let from = e.state;
        if from.is_terminal() {
            return Err(LedgerError::AlreadyTerminal { state: from });
        }
        let allowed = match state {
            TerminalState::Completed => {
                matches!(from, RequestState::Running | RequestState::CancelPending)
            }
            TerminalState::Failed => true,
            TerminalState::Cancelled => from != RequestState::Running,
        };
        if !allowed {
            return Err(LedgerError::InvalidTransition {
                from,
                to: state.into(),
            });
        }
        let advance = match (&commit, &e.session) {
            (None, _) => None,
            (Some(_), None) => return Err(LedgerError::Session(SessionError::NotFound)),
            (Some(entry), Some(s)) => {
                let revision = self.sessions.check_commit(
                    &s.session_id,
                    rid,
                    s.expected_revision,
                    entry,
                    e.held_generation,
                )?;
                Some(SessionAdvance {
                    session_id: s.session_id.clone(),
                    revision,
                    entry: *entry,
                    model: e.model.clone(),
                })
            }
        };
        let mut timings = e.timings;
        if timings.queue_wait_ms.is_none() {
            timings.queue_wait_ms = Some(e.queue_wait(now_ms));
        }
        timings.total_ms = Some(e.since(now_ms));
        let session = e.session.clone();
        let usage = e.usage;
        // Review finding 2: a request that was never sent to the backend (queue full, queued deadline, shutdown,
        // worker not ready, cancel while queued) has nothing to replay or protect, so its key is released and a
        // retry runs as a new request. With the dispatch journaled (M3 Q9) that is every request without a
        // `dispatched` record, of any incarnation and state (review J2: one that ended at dispatch, superseded or
        // on a journal error, and the restart's `worker_restart` of one that never got that far). Before M3 only a
        // request of this incarnation that never reached `running` is provably unsent: a prior-epoch request
        // (replayed as `accepted`) may have been dispatched and keeps its key.
        let never_sent = if e.dispatch_journaled {
            e.worker.is_none()
        } else {
            matches!(from, RequestState::Accepted | RequestState::Queued) && e.epoch == self.epoch()
        };
        let key_released = e.key.is_some() && never_sent;
        let written = self.write(
            &Record::Terminal(TerminalRecord {
                rid: rid.to_owned(),
                state,
                reason: reason.map(str::to_owned),
                timings,
                session: advance.clone(),
                at_ms: now_ms,
                key_released,
                usage: Some(usage),
            }),
            now_ms,
        );
        // Fail closed (review finding 10): the request ends anyway, as `failed: journal_error` without a session
        // advance or a retained body, and the poisoned journal refuses every later write, so no new work is
        // admitted. On disk it stays `admitted` without a terminal: the next incarnation fails it
        // `worker_restart` and aborts its rid. Both say failed and neither commits the turn or regenerates.
        let (state, reason, advance, retained) = if written.is_ok() {
            (state, reason, advance, retained)
        } else {
            self.journal.poison();
            (
                TerminalState::Failed,
                Some(REASON_JOURNAL_ERROR),
                None,
                None,
            )
        };
        let e = self.entries.get_mut(rid).expect("checked above");
        e.state = state.into();
        self.terminal += 1;
        e.reason = reason.map(str::to_owned);
        e.timings = timings;
        e.terminal_ms = Some(now_ms);
        self.stats.entry(e.scope.clone()).or_default().count(
            state,
            &usage,
            e.worker.as_ref().map(|w| w.id.as_str()),
        );
        if key_released && written.is_ok() {
            if let Some(key) = e.key.take() {
                self.by_key.remove(&(e.scope.clone(), key));
            }
        }
        let stream = e.stream;
        match (advance, session) {
            (Some(adv), _) => self
                .sessions
                .apply_advance(
                    &adv.session_id,
                    adv.revision,
                    adv.entry,
                    adv.model.as_deref(),
                    now_ms,
                )
                .expect("checked by check_commit under the lease"),
            (None, Some(s)) => self.sessions.release(&s.session_id, rid),
            (None, None) => {}
        }
        self.expiry.push_back((
            now_ms.saturating_add(self.limits.terminal_retention_ms),
            rid.to_owned(),
        ));
        if let (Some(response), false) = (retained, stream) {
            self.retain(rid, response);
        }
        written
    }

    fn retain(&mut self, rid: &str, response: RetainedResponse) {
        let size = response.body.len() as u64;
        if size > self.limits.max_retained_bytes {
            return;
        }
        while self.retained_bytes + size > self.limits.max_retained_bytes {
            let Some(old) = self.retained_order.pop_front() else {
                break;
            };
            if let Some(r) = self.entries.get_mut(&old).and_then(|e| e.retained.take()) {
                self.retained_bytes -= r.body.len() as u64;
            }
        }
        if let Some(e) = self.entries.get_mut(rid) {
            e.retained = Some(Arc::new(response));
            self.retained_bytes += size;
            self.retained_order.push_back(rid.to_owned());
        }
    }

    /// L7: the status of `rid` if it belongs to `scope` (another scope → `None`, same as unknown).
    pub fn status(&self, scope: &str, rid: &str) -> Option<RequestStatus> {
        let e = self.entries.get(rid).filter(|e| e.scope == scope)?;
        Some(RequestStatus {
            scope: e.scope.clone(),
            rid: rid.to_owned(),
            key: e.key.clone(),
            state: e.state,
            reason: e.reason.clone(),
            attempt: 1,
            epoch: e.epoch,
            stream: e.stream,
            surface: e.surface.clone(),
            session_id: e.session.as_ref().map(|s| s.session_id.clone()),
            timings: e.timings,
            usage: e.usage,
            worker_id: e.worker.as_ref().map(|w| w.id.clone()),
            worker_epoch: e.worker.as_ref().map(|w| w.epoch),
            labels: e.labels.clone(),
            class: e.class,
        })
    }

    /// [`Ledger::status`] as of `now_ms`: records past their terminal retention are purged first (as
    /// [`Ledger::accept`] does), so an expired request is unknown even when nothing was accepted since.
    pub fn status_at(&mut self, scope: &str, rid: &str, now_ms: u64) -> Option<RequestStatus> {
        self.purge(now_ms);
        self.status(scope, rid)
    }

    /// Q5: terminals of `scope` written since [`Ledger::opened_ms`] (all zero for a scope with none).
    pub fn stats(&self, scope: &str) -> ScopeStats {
        self.stats.get(scope).cloned().unwrap_or_default()
    }

    /// When this incarnation opened the ledger (Unix ms): the start of [`Ledger::stats`].
    pub fn opened_ms(&self) -> u64 {
        self.opened_ms
    }

    /// When `rid`'s `admitted` record was written: the start of its orphaned slot's retention bound (L3, H2);
    /// `None` once the rid is gone.
    pub fn admitted_ms(&self, rid: &str) -> Option<u64> {
        self.entries.get(rid).map(|e| e.accepted_ms)
    }

    /// Every non-terminal request, sorted by rid.
    pub fn in_flight(&self) -> Vec<InFlight> {
        let mut out: Vec<InFlight> = self
            .entries
            .iter()
            .filter(|(_, e)| !e.state.is_terminal())
            .map(|(rid, e)| InFlight {
                scope: e.scope.clone(),
                rid: rid.clone(),
                state: e.state,
                worker: e.worker.as_ref().map(|w| (w.id.clone(), w.epoch)),
                admitted_ms: e.accepted_ms,
            })
            .collect();
        out.sort_by(|a, b| a.rid.cmp(&b.rid));
        out
    }

    /// Records currently held (non-terminal + terminal within retention) as of the last operation.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// S1: a new session at revision 0 with a fresh salt lineage, journaled.
    pub fn create_session(&mut self, scope: &str, now_ms: u64) -> Result<SessionView, LedgerError> {
        let salt = Salt::random()?;
        let genesis = self.keys.scope_key(scope).prefix_digest(&salt, &[]);
        let session = self
            .sessions
            .new_root(scope, new_session_id()?, salt, genesis, now_ms)?;
        self.write(&Record::Session(session.record()), now_ms)?;
        let id = session.session_id.clone();
        self.sessions.insert(session);
        self.session(scope, &id, now_ms)
    }

    /// S3/Q5: fork `parent_id` at `revision` into a new session whose revision 0 is the parent's revision.
    pub fn fork_session(
        &mut self,
        scope: &str,
        parent_id: &str,
        revision: u64,
        now_ms: u64,
    ) -> Result<ForkView, LedgerError> {
        let child =
            self.sessions
                .new_fork(scope, parent_id, revision, new_session_id()?, now_ms)?;
        self.write(&Record::Session(child.record()), now_ms)?;
        let id = child.session_id.clone();
        self.sessions.insert(child);
        Ok(ForkView {
            child: self.session(scope, &id, now_ms)?,
            parent_session_id: parent_id.to_owned(),
            parent_revision: revision,
        })
    }

    /// A live session of `scope` (another scope or expired → `session_not_found`).
    pub fn session(
        &self,
        scope: &str,
        session_id: &str,
        now_ms: u64,
    ) -> Result<SessionView, LedgerError> {
        let s = self.sessions.get(scope, session_id, now_ms)?;
        Ok(SessionView {
            session_id: s.session_id.clone(),
            revision: s.revision(),
            current: s.current(),
            salt: s.salt,
            parent: s.parent.clone(),
            pinned_worker: s.pinned_worker.clone(),
            owner_generation: s.owner_generation,
            model: s.model.clone(),
        })
    }

    /// m4 design note Q2: the session owner as request `rid` holds it (`None` for a request without a session
    /// and an unknown rid). Read on the same guard right after [`Self::accept`] (the turn's routing target) and
    /// after [`Self::dispatched`] (whether its grant changed the owner).
    pub fn held_owner(&self, rid: &str) -> Option<HeldOwner> {
        let e = self.entries.get(rid)?;
        let session = e.session.as_ref()?;
        let worker = self
            .sessions
            .stored(&session.session_id)
            .filter(|s| s.owner_generation == e.held_generation)
            .and_then(|s| s.pinned_worker.clone());
        Some(HeldOwner {
            worker,
            owner_generation: e.held_generation,
        })
    }

    /// m4 design note Q4: the owner of a live session of any scope (expired → `session_not_found`).
    pub fn session_owner(&self, session_id: &str, now_ms: u64) -> Result<OwnerView, LedgerError> {
        let s = self.sessions.get_any(session_id, now_ms)?;
        Ok(OwnerView {
            session_id: s.session_id.clone(),
            worker: s.pinned_worker.clone(),
            owner_generation: s.owner_generation,
            revision: s.revision(),
        })
    }

    /// m4 design note Q5 (2)-(8), replay mode: moves the owner of `req.session_id` to `req.to_worker` at the next
    /// generation. Refusals, in this order, change nothing: unknown or expired session; a generation other than
    /// the expected one; `to_worker` unknown, serving another model (M5 D8: its deployment is not the bound
    /// model's, `deployment_of_model`, or for an unbound session not its owner's; an unbound session without owner
    /// skips this), not eligible, or already the owner; with `force` an eligible owner (no owner counts as not
    /// eligible), without it a held lease. Then the current revision and digest are journaled with the new owner
    /// (`session_owner`, fsync'd; a write error changes nothing and does not poison the ledger), and the owner, the
    /// session's TTL and (Q6, force) the broken lease apply.
    pub fn migrate_session(
        &mut self,
        req: MigrateRequest<'_>,
        check: &dyn Fn(&str) -> WorkerCheck,
        deployment_of_model: &dyn Fn(&str) -> Option<usize>,
        now_ms: u64,
    ) -> Result<Migrated, MigrateError> {
        self.purge(now_ms);
        let s = self
            .sessions
            .get_any(req.session_id, now_ms)
            .map_err(|_| MigrateError::NotFound)?;
        if s.owner_generation != req.expected_owner_generation {
            return Err(MigrateError::Conflict {
                worker: s.pinned_worker.clone(),
                owner_generation: s.owner_generation,
            });
        }
        let to = check(req.to_worker);
        let Some(to_deployment) = to.deployment() else {
            return Err(MigrateError::WorkerNotFound);
        };
        let session_deployment = match &s.model {
            Some(model) => Some(deployment_of_model(model)),
            None => s
                .pinned_worker
                .as_deref()
                .and_then(|w| check(w).deployment())
                .map(Some),
        };
        if session_deployment.is_some_and(|d| d != Some(to_deployment)) {
            return Err(MigrateError::ModelMismatch);
        }
        if to == WorkerCheck::Ineligible(to_deployment) {
            return Err(MigrateError::WorkerNotEligible);
        }
        if s.pinned_worker.as_deref() == Some(req.to_worker) {
            return Err(MigrateError::AlreadyOwner);
        }
        if req.force {
            if s.pinned_worker
                .as_deref()
                .is_some_and(|w| matches!(check(w), WorkerCheck::Eligible(_)))
            {
                return Err(MigrateError::OwnerHealthy);
            }
        } else if s.lease.is_some() {
            return Err(MigrateError::SessionBusy);
        }
        let owner_generation = s.owner_generation + 1;
        let revision = s.revision();
        let record = SessionOwnerRecord {
            session_id: s.session_id.clone(),
            worker: req.to_worker.to_owned(),
            owner_generation,
            revision,
            digest: s.current().digest,
            at_ms: now_ms,
        };
        let scope = s.scope.clone();
        let previous = s.pinned_worker.clone();
        let superseded = s.lease.clone();
        self.write(&Record::SessionOwner(record), now_ms)
            .map_err(MigrateError::Journal)?;
        self.sessions
            .set_owner(req.session_id, req.to_worker, owner_generation);
        self.sessions.touch(req.session_id, now_ms);
        if let Some(holder) = &superseded {
            self.sessions.release(req.session_id, holder);
        }
        Ok(Migrated {
            session_id: req.session_id.to_owned(),
            scope,
            worker: req.to_worker.to_owned(),
            owner_generation,
            revision,
            previous,
            superseded,
        })
    }

    /// Compacts first when due (so a failed compaction leaves the operation unapplied), then appends.
    fn write(&mut self, record: &Record, now_ms: u64) -> Result<(), LedgerError> {
        if self.journal.needs_compaction(self.limits.compaction_slack) {
            self.purge(now_ms);
            let live = self.live_records(now_ms);
            self.journal.compact(&live)?;
        }
        self.journal.append(record)?;
        Ok(())
    }

    /// Records that rebuild the current state: the highest epoch of every worker id ever seen, the drain intent of
    /// every draining worker id (M3.5 Q13), the latest residency intent per worker id (m36 design note Q8),
    /// requests (admitted, the last `dispatched`, the terminal without
    /// session advance, then the orphan release or else an unreleased hold), tombstones, then session snapshots
    /// (which already contain every committed revision).
    fn live_records(&self, now_ms: u64) -> Vec<Record> {
        let mut rids: Vec<(&String, &Entry)> = self.entries.iter().collect();
        rids.sort_by(|a, b| (a.1.accepted_ms, a.0).cmp(&(b.1.accepted_ms, b.0)));
        let mut out: Vec<Record> = self
            .worker_epochs
            .iter()
            .map(|(worker, &(epoch, at_ms))| {
                Record::WorkerEpoch(WorkerEpochRecord {
                    worker: worker.clone(),
                    epoch,
                    at_ms,
                })
            })
            .collect();
        out.extend(
            self.drains
                .iter()
                .map(|(worker, &at_ms)| Record::WorkerDrain {
                    worker: worker.clone(),
                    draining: true,
                    at_ms,
                }),
        );
        out.extend(
            self.residency
                .iter()
                .map(
                    |(worker, &(desired, intent_seq, at_ms))| Record::WorkerResidency {
                        worker: worker.clone(),
                        desired,
                        intent_seq,
                        at_ms,
                    },
                ),
        );
        for (rid, e) in rids {
            out.push(Record::Admitted(AdmittedRecord {
                scope: e.scope.clone(),
                rid: rid.clone(),
                key: e.key.clone(),
                digest: e.digest,
                epoch: e.epoch,
                stream: e.stream,
                surface: e.surface.clone(),
                session: e.session.clone(),
                at_ms: e.accepted_ms,
                dispatch_journaled: e.dispatch_journaled,
            }));
            if let Some(w) = &e.worker {
                out.push(Record::Dispatched(w.record(rid)));
            }
            if let (Some(state), Some(at_ms)) = (terminal_of(e.state), e.terminal_ms) {
                out.push(Record::Terminal(TerminalRecord {
                    rid: rid.clone(),
                    state,
                    reason: e.reason.clone(),
                    timings: e.timings,
                    session: None,
                    at_ms,
                    key_released: false,
                    usage: Some(e.usage),
                }));
            }
            match (&e.orphan_held, e.orphan_release) {
                (_, Some((at_ms, cause))) => {
                    out.push(Record::OrphanReleased(OrphanReleasedRecord {
                        rid: rid.clone(),
                        at_ms,
                        cause,
                    }))
                }
                // A hold not released yet: the next open charges its slot again.
                (Some((at_ms, worker)), None) => out.push(Record::OrphanHeld(OrphanHeldRecord {
                    rid: rid.clone(),
                    worker: worker.clone(),
                    at_ms: *at_ms,
                })),
                (None, None) => {}
            }
        }
        for index in &self.tombstone_order {
            out.push(Record::KeyExpired {
                scope: index.0.clone(),
                key: index.1.clone(),
                at_ms: self.tombstones[index],
            });
        }
        out.extend(
            self.sessions
                .records(now_ms)
                .into_iter()
                .map(Record::Session),
        );
        out
    }

    /// Drops terminal records past retention and expired sessions.
    fn purge(&mut self, now_ms: u64) {
        while let Some((expires, _)) = self.expiry.front() {
            if *expires > now_ms {
                break;
            }
            let (_, rid) = self.expiry.pop_front().expect("front exists");
            self.drop_record(&rid, now_ms);
        }
        self.sessions.purge_expired(now_ms);
    }

    /// The oldest terminal record (first in `expiry`) a full ledger may evict: none without
    /// `evict_terminal_when_full`, and never one that still holds its slot.
    fn eviction_candidate(&self, now_ms: u64) -> Option<String> {
        if !self.limits.evict_terminal_when_full {
            return None;
        }
        let retention_ms = self.limits.terminal_retention_ms;
        self.expiry
            .iter()
            .map(|(_, rid)| rid)
            .find(|rid| {
                self.entries
                    .get(*rid)
                    .is_some_and(|e| !e.holds_slot(retention_ms, now_ms))
            })
            .cloned()
    }

    /// Replay (OH-03): unkeyed victims leave no record, so a ledger over its bound after replay and retention evicts
    /// again as `accept` did, oldest evictable terminal record first.
    fn shed(&mut self, now_ms: u64) -> Result<(), LedgerError> {
        while self.entries.len() as u64 > self.limits.max_records {
            let Some(rid) = self.eviction_candidate(now_ms) else {
                break;
            };
            self.evict(&rid, now_ms)?;
        }
        Ok(())
    }

    /// Drops terminal record `rid` as retention would; a keyed one is first journaled as the `key_expired` its
    /// tombstone becomes at compaction, which every release replays.
    fn evict(&mut self, rid: &str, now_ms: u64) -> Result<(), LedgerError> {
        let e = &self.entries[rid];
        if let Some(key) = &e.key {
            let record = Record::KeyExpired {
                scope: e.scope.clone(),
                key: key.clone(),
                at_ms: now_ms,
            };
            self.write(&record, now_ms)?;
        }
        self.unexpire(rid);
        self.drop_record(rid, now_ms);
        Ok(())
    }

    fn unexpire(&mut self, rid: &str) {
        if let Some(i) = self.expiry.iter().position(|(_, r)| r == rid) {
            self.expiry.remove(i);
        }
    }

    /// Removes a record and its retained body; a keyed one leaves a tombstone at `at_ms`.
    fn drop_record(&mut self, rid: &str, at_ms: u64) {
        let Some(e) = self.entries.remove(rid) else {
            return;
        };
        if e.state.is_terminal() {
            self.terminal -= 1;
        }
        if let Some(r) = e.retained {
            self.retained_bytes -= r.body.len() as u64;
            self.retained_order.retain(|x| x != rid);
        }
        if let Some(key) = e.key {
            self.by_key.remove(&(e.scope.clone(), key.clone()));
            self.add_tombstone(e.scope, key, at_ms);
        }
    }

    fn add_tombstone(&mut self, scope: String, key: String, at_ms: u64) {
        let index = (scope, key);
        if self.tombstones.insert(index.clone(), at_ms).is_none() {
            self.tombstone_order.push_back(index);
        }
        while self.tombstone_order.len() as u64 > self.limits.max_records {
            if let Some(old) = self.tombstone_order.pop_front() {
                self.tombstones.remove(&old);
            }
        }
    }
}

fn terminal_of(state: RequestState) -> Option<TerminalState> {
    match state {
        RequestState::Completed => Some(TerminalState::Completed),
        RequestState::Failed => Some(TerminalState::Failed),
        RequestState::Cancelled => Some(TerminalState::Cancelled),
        _ => None,
    }
}

fn new_session_id() -> io::Result<String> {
    let bytes = random_bytes::<16>()?;
    let mut id = String::with_capacity(5 + 32);
    id.push_str("sess_");
    for b in bytes {
        id.push_str(&format!("{b:02x}"));
    }
    Ok(id)
}
