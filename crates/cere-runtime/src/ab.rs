//! Live A/B traffic split of the runtime-layer perf levers (ab-split design note AB-01..AB-06).
//!
//! `[deployments.native.ab]` names arms, each an id, a weight and runtime-layer lever choices (`drafter`,
//! `draft_policy`). Only `bit_identical` implementations of `runtime` levers are admitted ([`check_arm_lever`]), so an
//! arm changes speed, never output: greedy speculation still equals plain decoding on every arm. A request's arm is a
//! salted HMAC-SHA256 of its session key ([`SessionKey`]: the scope and `X-Cere-Session-Id`, else the request id)
//! mapped onto the weights ([`Plan::assign`]); a session keeps the arm it first got while the process runs and that
//! arm stays configured ([`AbState::assign`]), so a reload of the arms reaches new sessions only. The operator may
//! name an arm (`X-Cere-Ab-Arm` with `X-Cere-Operator-Authorization`, for tests). The arm's choices go over the
//! active levers where the request's `cere_work.flags` are built (`crate::native::serve::encode`), as the profile's
//! do. `GET /v1/cere/stats` reports per-arm counters and summaries under `ab` ([`AbState::render`]), by arm id only:
//! no prompt, output, token id or session id is kept or reported.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use hmac::{Hmac, Mac};
use serde_json::{json, Map, Value};
use sha2::Sha256;

use crate::app::App;
use crate::config::{AbConfig, NativeConfig, RuntimeConfig};
use crate::levers::{
    self, DraftPolicy, Drafter, GrammarDrafts, Layer, LeverSet, LeverSpec, Numerics,
};
use crate::native::serve::Samples;
use crate::reload::Apply;

type HmacSha256 = Hmac<Sha256>;

/// AB-02: lever `name` at implementation `id` may be an arm's choice: a registered `runtime`-layer lever (an
/// executor-layer one is process-global, switched only between requests by the dev lever route) at a registered
/// `bit_identical` implementation (a `changing` one would change live users' outputs). The error starts with the
/// lever name (the caller prefixes its path).
pub fn check_arm_lever(registry: &[LeverSpec], name: &str, id: &str) -> Result<(), String> {
    let spec = registry
        .iter()
        .find(|l| l.name == name)
        .ok_or_else(|| format!("{name}: unknown lever"))?;
    if spec.layer != Layer::Runtime {
        return Err(format!(
            "{name}: an executor-layer lever is process-global; an arm takes runtime-layer levers only"
        ));
    }
    let Some(implementation) = spec.impls.iter().find(|i| i.id == id) else {
        let known: Vec<&str> = spec.impls.iter().map(|i| i.id).collect();
        return Err(format!(
            "{name}: unknown implementation {id:?} (registered: {})",
            known.join(", ")
        ));
    };
    if implementation.numerics != Numerics::BitIdentical {
        return Err(format!(
            "{name}: {id:?} is a changing implementation; an arm takes bit-identical implementations only"
        ));
    }
    Ok(())
}

/// The runtime-layer levers one arm sets; the levers it leaves out keep the request's active set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArmLevers {
    pub drafter: Option<Drafter>,
    pub draft_policy: Option<DraftPolicy>,
    /// GrammarSpec GS-04: only "off" passes [`check_arm_lever`] ("on" is changing), so an arm of a `-v49` deployment
    /// can decode its constrained requests plain.
    pub grammar_drafts: Option<GrammarDrafts>,
}

impl ArmLevers {
    /// AB-02: an arm's `levers` table, each entry passing [`check_arm_lever`] against [`levers::REGISTRY`], and the
    /// profile's levers with the arm's over them runnable by the model (`LeverSet::from_table`: lookup drafts and the
    /// adaptive policy need `speculative_draft_tokens = 4`). The error starts with the lever name.
    pub fn from_table(
        arm: &BTreeMap<String, String>,
        profile: &BTreeMap<String, String>,
        speculative_draft_tokens: u32,
    ) -> Result<ArmLevers, String> {
        for (name, id) in arm {
            check_arm_lever(levers::REGISTRY, name, id)?;
        }
        let mut merged = profile.clone();
        merged.extend(arm.iter().map(|(k, v)| (k.clone(), v.clone())));
        let set = LeverSet::from_table(&merged, speculative_draft_tokens)?;
        let mut choices = ArmLevers::default();
        for name in arm.keys() {
            match name.as_str() {
                "drafter" => choices.drafter = Some(set.drafter),
                "draft_policy" => choices.draft_policy = Some(set.draft_policy),
                "grammar_drafts" => choices.grammar_drafts = Some(set.grammar_drafts),
                other => return Err(format!("{other}: no per-request path takes this lever")),
            }
        }
        Ok(choices)
    }

    /// `active` with this arm's choices over it.
    pub fn apply(&self, mut active: LeverSet) -> LeverSet {
        if let Some(drafter) = self.drafter {
            active.drafter = drafter;
        }
        if let Some(policy) = self.draft_policy {
            active.draft_policy = policy;
        }
        if let Some(grammar) = self.grammar_drafts {
            active.grammar_drafts = grammar;
        }
        active
    }
}

/// One resolved arm.
#[derive(Debug)]
pub struct PlanArm {
    pub id: Arc<str>,
    pub weight: u32,
    pub levers: ArmLevers,
    /// The arm's `levers` table as configured (reported next to its stats).
    pub table: BTreeMap<String, String>,
}

/// A deployment's resolved `[deployments.native.ab]`; the live settings hold one per entry that has the section.
#[derive(Debug)]
pub struct Plan {
    salt: String,
    arms: Vec<PlanArm>,
    /// The weights' sum, above 0.
    total: u64,
}

/// What a request's arm is derived from (AB-03).
#[derive(Debug, Clone, Copy)]
pub enum SessionKey<'a> {
    /// A session turn: the scope and its `X-Cere-Session-Id` (session ids are scope-local).
    Session { scope: &'a str, session_id: &'a str },
    /// Any other request: its request id, so each such request is a session of its own.
    Request(&'a str),
}

impl SessionKey<'_> {
    /// HMAC-SHA256 keyed by `salt` over a version tag, the key kind and its length-framed fields.
    fn digest(&self, salt: &str) -> [u8; 32] {
        fn frame(mac: &mut HmacSha256, bytes: &[u8]) {
            mac.update(&(bytes.len() as u64).to_le_bytes());
            mac.update(bytes);
        }
        let mut mac =
            HmacSha256::new_from_slice(salt.as_bytes()).expect("HMAC accepts any key length");
        mac.update(b"cere-ab-v1\0");
        match self {
            SessionKey::Session { scope, session_id } => {
                mac.update(b"s");
                frame(&mut mac, scope.as_bytes());
                frame(&mut mac, session_id.as_bytes());
            }
            SessionKey::Request(rid) => {
                mac.update(b"r");
                frame(&mut mac, rid.as_bytes());
            }
        }
        mac.finalize().into_bytes().into()
    }
}

impl Plan {
    /// The plan of a validated `[deployments.native.ab]` (`None` without the section).
    pub fn from_native(native: &NativeConfig) -> Result<Option<Plan>, String> {
        let Some(ab) = &native.ab else {
            return Ok(None);
        };
        Self::new(ab, &native.levers, native.speculative_draft_tokens).map(Some)
    }

    pub fn new(
        ab: &AbConfig,
        profile: &BTreeMap<String, String>,
        speculative_draft_tokens: u32,
    ) -> Result<Plan, String> {
        let arms = ab
            .arms
            .iter()
            .map(|arm| {
                Ok(PlanArm {
                    id: Arc::from(arm.id.as_str()),
                    weight: arm.weight,
                    levers: ArmLevers::from_table(&arm.levers, profile, speculative_draft_tokens)
                        .map_err(|e| format!("ab arm {}: {e}", arm.id))?,
                    table: arm.levers.clone(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let total = arms.iter().map(|a| u64::from(a.weight)).sum();
        if total == 0 {
            return Err("ab arms weights must sum above 0".to_owned());
        }
        Ok(Plan {
            salt: ab.salt.clone(),
            arms,
            total,
        })
    }

    pub fn arms(&self) -> &[PlanArm] {
        &self.arms
    }

    /// The index of arm `id`.
    pub fn position(&self, id: &str) -> Option<usize> {
        self.arms.iter().position(|a| &*a.id == id)
    }

    /// AB-03: the arm of `key` by the hash alone: the digest's first 8 bytes (little-endian) as a point in
    /// [0, total) by multiply-shift, then the arm whose cumulative weight range holds it (a 0-weight arm none).
    pub fn assign(&self, key: SessionKey<'_>) -> usize {
        self.pick(&key.digest(&self.salt))
    }

    fn pick(&self, digest: &[u8; 32]) -> usize {
        let mut head = [0u8; 8];
        head.copy_from_slice(&digest[..8]);
        let x = u64::from_le_bytes(head);
        let point = ((u128::from(x) * u128::from(self.total)) >> 64) as u64;
        let mut end = 0u64;
        for (index, arm) in self.arms.iter().enumerate() {
            end += u64::from(arm.weight);
            if point < end {
                return index;
            }
        }
        unreachable!("the point lies below the weights' sum")
    }
}

/// How many session -> arm entries one entry remembers; past it the oldest is forgotten (that session's next turn
/// takes the hash's arm under the current plan).
pub const STICKY_SESSIONS: usize = 65_536;

/// The sessions' remembered arms, by key digest (the session id itself is not kept), oldest first in `order`.
#[derive(Default)]
struct Sticky {
    arms: HashMap<[u8; 32], Arc<str>>,
    order: VecDeque<[u8; 32]>,
}

/// AB-04: one arm's counters and summaries since the process started (or since the arm was configured).
#[derive(Default)]
struct ArmCounters {
    /// Requests the native path started on this arm.
    requests: u64,
    completed: u64,
    cancelled: u64,
    /// Requests that failed: a refused encode or a generation that ended with an error.
    errors: u64,
    /// Completion tokens of completed requests.
    completion_tokens: u64,
    /// Speculate rounds and accepted drafts of every ticket of the arm's requests.
    spec_verify_steps: u64,
    spec_accepted_drafts: u64,
    /// Per completed request: acceptance -> first committed token, in ms.
    ttft_ms: Summary,
    /// Per completed request with decode tokens: tokens after the first over first -> last committed token.
    decode_tok_s: Summary,
    /// Per completed request that speculated: completion tokens per speculate round (m29 design note D13 ii).
    accept_length: Summary,
}

/// A count, a mean over every sample and p50/p99 over the most recent ones.
#[derive(Default)]
struct Summary {
    count: u64,
    sum: f64,
    recent: Samples,
}

impl Summary {
    fn push(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.recent.push(value);
    }

    fn render(&self) -> Value {
        let (p50, p99) = self.recent.percentiles();
        let mean = if self.count == 0 {
            Value::Null
        } else {
            json!(self.sum / self.count as f64)
        };
        json!({ "count": self.count, "mean": mean, "p50": p50, "p99": p99 })
    }
}

/// One arm's stats; a request holds it from acceptance to its terminal.
#[derive(Default)]
pub struct ArmStats(Mutex<ArmCounters>);

impl ArmStats {
    fn lock(&self) -> MutexGuard<'_, ArmCounters> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn render(&self, arm: &PlanArm) -> Value {
        let c = self.lock();
        json!({
            "weight": arm.weight,
            "levers": arm.table,
            "requests": c.requests,
            "completed": c.completed,
            "cancelled": c.cancelled,
            "errors": c.errors,
            "completion_tokens": c.completion_tokens,
            "spec_verify_steps": c.spec_verify_steps,
            "spec_accepted_drafts": c.spec_accepted_drafts,
            "ttft_ms": c.ttft_ms.render(),
            "decode_tok_s": c.decode_tok_s.render(),
            "accept_length": c.accept_length.render(),
        })
    }
}

/// A request's arm: its id, its lever choices and its stats.
pub struct Arm {
    pub id: Arc<str>,
    pub levers: ArmLevers,
    stats: Arc<ArmStats>,
    accepted: Instant,
}

impl Arm {
    /// The native path starts the request on this arm (counted in `requests`).
    pub(crate) fn start(&self) -> Arc<RequestTally> {
        self.stats.lock().requests += 1;
        Arc::new(RequestTally {
            stats: self.stats.clone(),
            accepted: self.accepted,
            verify_steps: AtomicU64::new(0),
            accepted_drafts: AtomicU64::new(0),
            timing: Mutex::new(Timing::default()),
        })
    }
}

/// How a request on an arm ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum End {
    Completed { completion_tokens: u64 },
    Cancelled,
    Error,
}

/// The committed-token span of a request's choices.
#[derive(Default)]
struct Timing {
    first: Option<Instant>,
    last: Option<Instant>,
    decode_tokens: u64,
}

/// One running request's measures, shared by its choices; folded into the arm's stats once, at its end.
pub(crate) struct RequestTally {
    stats: Arc<ArmStats>,
    accepted: Instant,
    verify_steps: AtomicU64,
    accepted_drafts: AtomicU64,
    timing: Mutex<Timing>,
}

impl RequestTally {
    /// A ticket's speculate rounds and accepted drafts.
    pub(crate) fn spec(&self, verify_steps: u64, accepted_drafts: u64) {
        self.verify_steps.fetch_add(verify_steps, Ordering::Relaxed);
        self.accepted_drafts
            .fetch_add(accepted_drafts, Ordering::Relaxed);
    }

    /// A choice's committed span: its first and last committed token instants and its token count (>= 1).
    pub(crate) fn choice(&self, first: Instant, last: Instant, tokens: u64) {
        let mut t = self.timing.lock().unwrap_or_else(|e| e.into_inner());
        t.first = Some(t.first.map_or(first, |f| f.min(first)));
        t.last = Some(t.last.map_or(last, |l| l.max(last)));
        t.decode_tokens += tokens.saturating_sub(1);
    }

    pub(crate) fn finish(&self, end: End) {
        let steps = self.verify_steps.load(Ordering::Relaxed);
        let accepted = self.accepted_drafts.load(Ordering::Relaxed);
        let timing = std::mem::take(&mut *self.timing.lock().unwrap_or_else(|e| e.into_inner()));
        let mut c = self.stats.lock();
        c.spec_verify_steps += steps;
        c.spec_accepted_drafts += accepted;
        match end {
            End::Completed { completion_tokens } => {
                c.completed += 1;
                c.completion_tokens += completion_tokens;
                if let Some(first) = timing.first {
                    let ttft = first.saturating_duration_since(self.accepted);
                    c.ttft_ms.push(ttft.as_secs_f64() * 1e3);
                    let span = timing.last.map(|l| l.saturating_duration_since(first));
                    if let Some(span) = span.filter(|s| timing.decode_tokens > 0 && !s.is_zero()) {
                        c.decode_tok_s
                            .push(timing.decode_tokens as f64 / span.as_secs_f64());
                    }
                }
                if steps > 0 {
                    c.accept_length
                        .push(completion_tokens as f64 / steps as f64);
                }
            }
            End::Cancelled => c.cancelled += 1,
            End::Error => c.errors += 1,
        }
    }
}

/// The live A/B state of one `[[deployments]]` entry.
#[derive(Default)]
struct Entry {
    stats: BTreeMap<Arc<str>, Arc<ArmStats>>,
    sticky: Sticky,
    /// The salt and arm ids of the plan the last [`AbState::retain`] installed (`None`: no reload has changed the
    /// entry's `ab` section, so every request runs the startup plan). A request that read its plan before a reload
    /// reaches [`AbState::assign`] / [`AbState::arm`] after that reload's `retain`; checked under the same lock, it
    /// records nothing for an arm (or, for a session, a salt) the entry no longer configures (AB-05).
    configured: Option<(String, Vec<Arc<str>>)>,
}

impl Entry {
    /// Arm `id` of `plan` may be recorded: no reload yet, or the current plan configures the arm (and, with
    /// `salted`, under `plan`'s salt).
    fn configures(&self, plan: &Plan, id: &str, salted: bool) -> bool {
        self.configured.as_ref().is_none_or(|(salt, ids)| {
            (!salted || *salt == plan.salt) && ids.iter().any(|i| &**i == id)
        })
    }
}

/// AB-03/AB-04: the per-arm stats and the session -> arm memory of every `[[deployments]]` entry.
pub struct AbState {
    entries: Vec<Mutex<Entry>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl AbState {
    pub fn new(deployments: usize) -> Self {
        Self {
            entries: (0..deployments)
                .map(|_| Mutex::new(Entry::default()))
                .collect(),
        }
    }

    /// AB-03: the arm index of a request on entry `deployment` under `plan`. A session turn keeps the arm its
    /// session first got while that arm is still configured (whatever the weights now say); otherwise, and for a
    /// request without a session, the hash picks, and a session remembers the pick (unless a reload has since
    /// removed the arm or changed the salt).
    pub fn assign(&self, deployment: usize, plan: &Plan, key: SessionKey<'_>) -> usize {
        let SessionKey::Session { .. } = key else {
            return plan.assign(key);
        };
        let digest = key.digest(&plan.salt);
        let mut entry = lock(&self.entries[deployment]);
        if let Some(index) = entry
            .sticky
            .arms
            .get(&digest)
            .and_then(|id| plan.position(id))
        {
            return index;
        }
        let index = plan.pick(&digest);
        let id = &plan.arms[index].id;
        if !entry.configures(plan, id, true) {
            return index;
        }
        let sticky = &mut entry.sticky;
        if sticky.arms.insert(digest, id.clone()).is_none() {
            sticky.order.push_back(digest);
            if sticky.order.len() > STICKY_SESSIONS {
                if let Some(oldest) = sticky.order.pop_front() {
                    sticky.arms.remove(&oldest);
                }
            }
        }
        index
    }

    /// The request's [`Arm`]: arm `index` of `plan` with the entry's stats of that arm id (stats of their own, kept
    /// nowhere, for an arm a reload has removed since the request read `plan`).
    pub fn arm(&self, deployment: usize, plan: &Plan, index: usize) -> Arm {
        let arm = &plan.arms[index];
        let mut entry = lock(&self.entries[deployment]);
        let stats = if entry.configures(plan, &arm.id, false) {
            entry.stats.entry(arm.id.clone()).or_default().clone()
        } else {
            Arc::default()
        };
        Arm {
            id: arm.id.clone(),
            levers: arm.levers,
            stats,
            accepted: Instant::now(),
        }
    }

    /// AB-05 (reload): entry `deployment` now runs `plan`. The stats and remembered sessions of arms it no longer
    /// configures are dropped; a new salt forgets every session (a new experiment).
    fn retain(&self, deployment: usize, plan: Option<&Plan>, salt_changed: bool) {
        let mut entry = lock(&self.entries[deployment]);
        let kept = |id: &Arc<str>| plan.is_some_and(|p| p.position(id).is_some());
        entry.stats.retain(|id, _| kept(id));
        if salt_changed || plan.is_none() {
            entry.sticky = Sticky::default();
        } else {
            let Sticky { arms, order } = &mut entry.sticky;
            arms.retain(|_, id| kept(id));
            order.retain(|digest| arms.contains_key(digest));
        }
        entry.configured = Some(plan.map_or_else(Default::default, |p| {
            (
                p.salt.clone(),
                p.arms.iter().map(|a| a.id.clone()).collect(),
            )
        }));
    }

    /// AB-04: the `ab` object of `GET /v1/cere/stats`: per served model with a plan, per configured arm (plan order),
    /// its weight, levers, counters and summaries. `None` when no entry has a plan (the key is then absent).
    pub fn render(&self, plans: &[Option<Arc<Plan>>], models: &[&str]) -> Option<Value> {
        let mut out = Map::new();
        for (index, plan) in plans.iter().enumerate() {
            let Some(plan) = plan else { continue };
            let entry = lock(&self.entries[index]);
            let mut arms = Map::new();
            for arm in &plan.arms {
                let value = match entry.stats.get(&arm.id) {
                    Some(s) => s.render(arm),
                    None => ArmStats::default().render(arm),
                };
                arms.insert(arm.id.to_string(), value);
            }
            out.insert(models[index].to_owned(), Value::Object(arms));
        }
        (!out.is_empty()).then_some(Value::Object(out))
    }
}

/// AB-05: the reload hook of `[deployments.native.ab]`: arms a candidate drops lose their stats and remembered
/// sessions, a new salt forgets every session; new requests take the new plan with the new settings.
pub(crate) fn reload_hook(
    _app: &App,
    running: &RuntimeConfig,
    candidate: &RuntimeConfig,
) -> Result<Option<Apply>, String> {
    let ab = |c: &RuntimeConfig, i: usize| -> Option<AbConfig> {
        c.deployments
            .get(i)
            .and_then(|d| d.native.as_ref())
            .and_then(|n| n.ab.clone())
    };
    let mut changes = Vec::new();
    for index in 0..candidate.deployments.len() {
        let (before, after) = (ab(running, index), ab(candidate, index));
        if before == after {
            continue;
        }
        let plan = match candidate.deployments[index].native.as_ref() {
            Some(native) => Plan::from_native(native)?,
            None => None,
        };
        let salt_changed = before.as_ref().map(|a| &a.salt) != after.as_ref().map(|a| &a.salt);
        changes.push((index, plan, salt_changed));
    }
    if changes.is_empty() {
        return Ok(None);
    }
    Ok(Some(Box::new(move |app: &App| {
        for (index, plan, salt_changed) in &changes {
            app.ab.retain(*index, plan.as_ref(), *salt_changed);
        }
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AbArmConfig;
    use crate::levers::LeverClass;

    /// One test arm: id, weight and its `levers` table.
    type ArmSpec<'a> = (&'a str, u32, &'a [(&'a str, &'a str)]);

    fn plan(arms: &[ArmSpec<'_>]) -> Plan {
        let ab = AbConfig {
            salt: "exp-1".to_owned(),
            arms: arms
                .iter()
                .map(|(id, weight, levers)| AbArmConfig {
                    id: (*id).to_owned(),
                    weight: *weight,
                    levers: levers
                        .iter()
                        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                        .collect(),
                })
                .collect(),
        };
        Plan::new(&ab, &BTreeMap::new(), 4).unwrap()
    }

    const RUNTIME_CHANGING: &[LeverSpec] = &[LeverSpec {
        name: "toy_drafter",
        slot: "spec.accept",
        layer: Layer::Runtime,
        class: LeverClass::Hot,
        default: "a",
        family: None,
        impls: &[
            levers::bit_identical("a", "x::a"),
            levers::changing("b", "x::b", true),
        ],
    }];

    #[test]
    fn arm_levers_are_runtime_layer_and_bit_identical_only() {
        assert_eq!(
            check_arm_lever(levers::REGISTRY, "k2_gemm", "cublaslt").unwrap_err(),
            "k2_gemm: an executor-layer lever is process-global; an arm takes runtime-layer levers only"
        );
        assert_eq!(
            check_arm_lever(levers::REGISTRY, "drafter", "ngram").unwrap_err(),
            "drafter: unknown implementation \"ngram\" (registered: mtp, lookup, mtp+lookup, mtp+lookup+tail, mtp+lookup+long, mtp+lookup+chain)"
        );
        assert_eq!(
            check_arm_lever(levers::REGISTRY, "spec", "x").unwrap_err(),
            "spec: unknown lever"
        );
        assert_eq!(
            check_arm_lever(RUNTIME_CHANGING, "toy_drafter", "b").unwrap_err(),
            "toy_drafter: \"b\" is a changing implementation; an arm takes bit-identical implementations only"
        );
        check_arm_lever(RUNTIME_CHANGING, "toy_drafter", "a").unwrap();
        // grammar-spec design note GS-04: grammar_drafts "on" is the shipped registry's one changing runtime-layer
        // implementation, so it is refused as an arm's choice; every bit-identical one may be an arm's choice.
        assert_eq!(
            check_arm_lever(levers::REGISTRY, "grammar_drafts", "on").unwrap_err(),
            "grammar_drafts: \"on\" is a changing implementation; an arm takes bit-identical implementations only"
        );
        for lever in levers::REGISTRY
            .iter()
            .filter(|l| l.layer == Layer::Runtime)
        {
            for i in lever.impls {
                let checked = check_arm_lever(levers::REGISTRY, lever.name, i.id);
                assert_eq!(
                    checked.is_ok(),
                    i.numerics == Numerics::BitIdentical,
                    "{}/{}",
                    lever.name,
                    i.id
                );
            }
        }
    }

    #[test]
    fn arm_choices_go_over_the_active_set() {
        let profile = BTreeMap::from([("drafter".to_owned(), "lookup".to_owned())]);
        let arm = BTreeMap::from([("draft_policy".to_owned(), "adaptive".to_owned())]);
        let levers = ArmLevers::from_table(&arm, &profile, 4).unwrap();
        assert_eq!(
            levers,
            ArmLevers {
                drafter: None,
                draft_policy: Some(DraftPolicy::Adaptive),
                grammar_drafts: None,
            }
        );
        let active = LeverSet {
            drafter: Drafter::MtpLookup,
            ..LeverSet::default()
        };
        let applied = levers.apply(active);
        assert_eq!(applied.drafter, Drafter::MtpLookup);
        assert_eq!(applied.draft_policy, DraftPolicy::Adaptive);
        assert_eq!(ArmLevers::default().apply(active), active);
        let grammar = BTreeMap::from([("grammar_drafts".to_owned(), "off".to_owned())]);
        let on = LeverSet {
            grammar_drafts: GrammarDrafts::On,
            ..active
        };
        let plain = ArmLevers::from_table(&grammar, &profile, 4).unwrap();
        assert_eq!(plain.apply(on).grammar_drafts, GrammarDrafts::Off);
        assert_eq!(
            ArmLevers::from_table(&arm, &BTreeMap::new(), 0).unwrap_err(),
            "draft_policy: \"adaptive\" needs speculative_draft_tokens = 4"
        );
    }

    #[test]
    fn assignment_is_deterministic_per_key_and_salt() {
        let p = plan(&[("a", 1, &[]), ("b", 1, &[])]);
        let key = SessionKey::Session {
            scope: "dev",
            session_id: "sess_1",
        };
        let first = p.assign(key);
        for _ in 0..10 {
            assert_eq!(p.assign(key), first);
        }
        // The same session id in another scope is another session; a request id is its own key.
        let picks: Vec<usize> = (0..64)
            .map(|i| {
                p.assign(SessionKey::Session {
                    scope: "dev",
                    session_id: &format!("sess_{i}"),
                })
            })
            .collect();
        assert!(picks.contains(&0) && picks.contains(&1), "{picks:?}");
        let r = SessionKey::Request("rid_1");
        assert_eq!(p.assign(r), p.assign(r));
        // A 0-weight arm takes no session.
        let drained = plan(&[("a", 0, &[]), ("b", 5, &[])]);
        for i in 0..256 {
            assert_eq!(drained.assign(SessionKey::Request(&format!("r{i}"))), 1);
        }
    }

    #[test]
    fn weights_split_sessions_within_a_bound() {
        // 20,000 keys over weights 1:3: each arm's share within 1.5 points of its weight's (binomial sd ~0.31 points,
        // so the bound is ~5 sd and the keys are fixed, so the test is deterministic).
        let p = plan(&[
            ("control", 1, &[]),
            ("adaptive", 3, &[("draft_policy", "adaptive")]),
        ]);
        let n = 20_000;
        let mut counts = [0u32; 2];
        for i in 0..n {
            counts[p.assign(SessionKey::Request(&format!("req-{i:08}")))] += 1;
        }
        let share = f64::from(counts[1]) / f64::from(n);
        assert!((share - 0.75).abs() < 0.015, "{counts:?}");
        // Another salt reshuffles: a different split of the same keys (~37.5% move at 1:3).
        let mut other = plan(&[("control", 1, &[]), ("adaptive", 3, &[])]);
        other.salt = "exp-2".to_owned();
        let moved = (0..1000)
            .map(|i| format!("req-{i:08}"))
            .filter(|k| p.assign(SessionKey::Request(k)) != other.assign(SessionKey::Request(k)))
            .count();
        assert!(moved > 250, "{moved} of 1000 keys moved with a new salt");
    }

    #[test]
    fn a_session_keeps_its_arm_across_a_reweight_until_its_arm_is_removed() {
        let state = AbState::new(1);
        let before = plan(&[("a", 1, &[]), ("b", 1, &[])]);
        let keys: Vec<String> = (0..64).map(|i| format!("s{i}")).collect();
        fn key(id: &str) -> SessionKey<'_> {
            SessionKey::Session {
                scope: "dev",
                session_id: id,
            }
        }
        let first: Vec<usize> = keys
            .iter()
            .map(|k| state.assign(0, &before, key(k)))
            .collect();
        // All weight to "b": remembered sessions keep their arm, a new session goes to "b".
        let after = plan(&[("a", 0, &[]), ("b", 1, &[])]);
        state.retain(0, Some(&after), false);
        for (k, want) in keys.iter().zip(&first) {
            assert_eq!(state.assign(0, &after, key(k)), *want, "{k}");
        }
        assert_eq!(state.assign(0, &after, key("new")), 1);
        // "a" removed: its sessions take the hash's arm under the new plan.
        let only_b = plan(&[("b", 1, &[])]);
        state.retain(0, Some(&only_b), false);
        for k in &keys {
            assert_eq!(state.assign(0, &only_b, key(k)), 0, "{k}");
        }
    }

    #[test]
    fn stats_count_per_arm_and_drop_removed_arms() {
        let state = AbState::new(1);
        let p = plan(&[("a", 1, &[]), ("b", 1, &[("draft_policy", "adaptive")])]);
        let a = state.arm(0, &p, 0);
        let tally = a.start();
        tally.spec(4, 6);
        let t0 = Instant::now();
        tally.choice(t0, t0 + std::time::Duration::from_millis(100), 11);
        tally.finish(End::Completed {
            completion_tokens: 11,
        });
        state.arm(0, &p, 1).start().finish(End::Error);
        state.arm(0, &p, 1).start().finish(End::Cancelled);
        let plans = [Some(Arc::new(p))];
        let out = state.render(&plans, &["m"]).unwrap();
        let a = &out["m"]["a"];
        assert_eq!(a["requests"], 1);
        assert_eq!(a["completed"], 1);
        assert_eq!(a["completion_tokens"], 11);
        assert_eq!(a["spec_verify_steps"], 4);
        assert_eq!(a["spec_accepted_drafts"], 6);
        assert_eq!(a["accept_length"]["mean"], 11.0 / 4.0);
        assert_eq!(a["ttft_ms"]["count"], 1);
        assert_eq!(a["decode_tok_s"]["count"], 1);
        let rate = a["decode_tok_s"]["mean"].as_f64().unwrap();
        assert!((rate - 100.0).abs() < 1e-6, "{rate}");
        let b = &out["m"]["b"];
        assert_eq!(b["requests"], 2);
        assert_eq!(b["errors"], 1);
        assert_eq!(b["cancelled"], 1);
        assert_eq!(b["levers"], json!({"draft_policy": "adaptive"}));
        assert_eq!(
            b["ttft_ms"],
            json!({"count": 0, "mean": null, "p50": null, "p99": null})
        );
        assert!(state.render(&[None], &["m"]).is_none());
        let only_b = plan(&[("b", 1, &[])]);
        state.retain(0, Some(&only_b), false);
        assert_eq!(lock(&state.entries[0]).stats.len(), 1);
    }

    /// AB-05 (review L5): a request that read its plan before a reload removed its arm (its `assign`/`arm` run after
    /// the reload's `retain`) neither re-creates the removed arm's stats nor remembers its session on that arm, so
    /// re-adding the arm id later starts it from zero and the session takes the hash's arm.
    #[test]
    fn retain_then_stale_arm_does_not_resurrect() {
        fn key(id: &str) -> SessionKey<'_> {
            SessionKey::Session {
                scope: "dev",
                session_id: id,
            }
        }
        let state = AbState::new(1);
        let stale = plan(&[("a", 1, &[]), ("b", 1, &[])]);
        let ids: Vec<String> = (0..64).map(|i| format!("s{i}")).collect();
        let id = ids.iter().find(|k| stale.assign(key(k)) == 1).unwrap();
        state.retain(0, Some(&plan(&[("a", 1, &[])])), false);
        let index = state.assign(0, &stale, key(id));
        assert_eq!(index, 1);
        state.arm(0, &stale, index).start().finish(End::Cancelled);
        // "b" is back, at weight 0: a remembered session would keep it, a new one hashes to "a".
        let readded = plan(&[("a", 1, &[]), ("b", 0, &[])]);
        state.retain(0, Some(&readded), false);
        assert_eq!(state.assign(0, &readded, key(id)), 0);
        let out = state.render(&[Some(Arc::new(readded))], &["m"]).unwrap();
        assert_eq!(out["m"]["b"]["requests"], 0);
        assert_eq!(out["m"]["b"]["cancelled"], 0);
        // A new salt (a new experiment): a request still on the old plan's salt leaves no remembered session.
        let mut salted = plan(&[("a", 1, &[]), ("b", 1, &[])]);
        salted.salt = "exp-2".to_owned();
        state.retain(0, Some(&salted), true);
        state.assign(0, &stale, key(id));
        assert!(lock(&state.entries[0]).sticky.arms.is_empty());
        state.assign(0, &salted, key(id));
        assert_eq!(lock(&state.entries[0]).sticky.arms.len(), 1);
    }
}
