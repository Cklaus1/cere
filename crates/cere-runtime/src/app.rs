//! Process state of `cered` (m1 design note Wave C): the merged primitives wired together, startup recovery
//! (L3) and graceful shutdown (L8).
//!
//! Workers (m3 design note C2, C3): one [`Replica`] per configured backend, in configuration order, each
//! identified by its device (`scheduler.devices`, Q3). Each replica has its own backend client, readiness
//! probe and smoke, and every readiness publish feeds that worker's record in the registry and admission
//! ([`App::publish_worker`], Q7). Worker epochs (Q4) start at the ledger's registration (highest journaled
//! epoch + 1) and move only on that worker's own new `instance_id` or rejoin after lost, journaled before its
//! re-smoke.
//!
//! Startup: open the ledger (epoch + 1, prior-epoch requests failed `worker_restart`, one worker epoch per
//! configured worker), charge one admission slot per recovered orphan on the cap that holds it (Q9), send its
//! `/abort_request` to every target at once (each retried every second until acknowledged or the slot is
//! released), and hold the slot until `ledger.terminal_retention_ms` has passed since its `admitted` record,
//! whatever the aborts' outcome (the backend may ignore the abort: SGLang drops a stream's state when the client
//! connection closes but keeps generating), and start the readiness probes. With a replica's `load_endpoint`
//! set, an aborted orphan charged to that worker is released early once its backend load `L` is at most `C`,
//! the number of `cered` streams confirmed inside that worker's scheduler, on two consecutive polls (m1close Q6).
//! Every release is journaled (`orphan_released`); an orphan still held when the process ended is charged again
//! by the next start until its original bound (Q7), and so is a remote request whose H2 hold was journaled
//! (`orphan_held`) and not released. An orphan dispatched to a configured remote worker at its configured
//! endpoint (M3.5 Q15) is charged to that worker and held by [`RemoteOrphan`] instead: reconciliation cancels its
//! operation, and its agent's `cleanup_acknowledged` in this term or the same bound releases it.
//! Shutdown: stop admission, drain in-flight requests for `api.shutdown_drain_ms`, cancel the rest (`shutdown`;
//! each executor aborts on its own worker), wait a bounded time for their backend streams to close, and report
//! clean only if every request reached its terminal through its own executor, no backend stream is still open on
//! any worker (counted per worker: its charged slots minus its orphans whose abort was sent and that only wait for
//! their release, m3 design note C6) and no terminal write was lost (journal not poisoned).
//!
//! A replica's backend is an SGLang server over HTTP or the in-process native executor ([`Backend`], spec m22
//! R4; exactly one replica). A native backend's executor lived inside the previous process, so its recovered
//! orphans hold no backend work: they fail `worker_restart` and are not charged.

use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::admission::{
    Admission, AdmissionLimits, AdmissionSnapshot, Permit, SchedulePolicy, WorkerCap,
};
use crate::api_types::DeploymentIdentity;
use crate::auth::Authenticator;
use crate::config::{
    BackendKind, PrefixCacheIsolation, RemoteWorkerConfig, ReplicaConfig, RuntimeConfig,
    ScopeCredential,
};
use crate::digest::DigestKeys;
use crate::fed::client::{AuthorityBase, ClientTiming, WorkerClient};
use crate::fed::orphan::RemoteOrphan;
use crate::fed::registry::{
    ModelView, RegistryListener, RemoteRegistry, RemoteState, RemoteTiming, ResidencyError,
};
use crate::fed::remote::{self, RemoteCall, RemoteWorker};
use crate::fed::tls::Pki;
use crate::journal::{OrphanReleaseCause, Residency};
use crate::ledger::{
    CancelOutcome, Ledger, LedgerError, LedgerLimits, MigrateError, MigrateRequest, Migrated,
    Orphan, OrphanCharge, OrphanWorker, RequestState, WorkerCheck, WorkerTarget,
    REASON_SESSION_OWNER_SUPERSEDED, REASON_WORKER_EPOCH_SUPERSEDED,
};
use crate::levers::{self, LeverSet};
use crate::native::detok::Vocab;
use crate::native::driver::{Driver, Model, Session};
use crate::native::export::{self, ExportError, Exported, StateIdentity};
use crate::native::ffi::ExecutionCapabilities;
use crate::native::sampling::GenerationDefaults;
use crate::native::serve::SessionQueue;
use crate::native::sidecar::{GrammarSidecar, Sidecar};
use crate::readiness::{
    Problem, Readiness, ReadinessSnapshot, RemoteDriver, RemoteProblems, PROBE_INTERVAL,
};
use crate::reload::{self, ReloadError, Reloaded, Settings};
use crate::resume::ResumeStore;
use crate::sglang::{Rid, RidGenerator, SglangHttp, Timeouts};
use crate::worker::{Aging, WorkerHealth, WorkerRegistry, WorkerState};

/// Terminal/cancel reason codes set by the server.
pub const REASON_CANCEL_REQUESTED: &str = "cancel_requested";
pub const REASON_CLIENT_DISCONNECTED: &str = "client_disconnected";
pub const REASON_CLIENT_BACKPRESSURE: &str = "client_backpressure";
pub const REASON_DEADLINE_EXCEEDED: &str = "deadline_exceeded";
pub const REASON_SHUTDOWN: &str = "shutdown";
pub const REASON_CANCEL_UNACKNOWLEDGED: &str = "cancel_unacknowledged";
pub const REASON_QUEUE_FULL: &str = "queue_full";
pub const REASON_BACKEND_ABORT: &str = "backend_abort";
pub const REASON_WORKER_NOT_READY: &str = "worker_not_ready";
/// H2/D5: an in-flight remote request whose worker turned Lost.
pub const REASON_WORKER_LOST: &str = "worker_lost";
/// Readiness problem field while the ledger's journal is poisoned.
pub const PROBLEM_JOURNAL_POISONED: &str = "journal_poisoned";

/// Retry period of a failed orphan `/abort_request` at startup.
const ORPHAN_ABORT_RETRY: Duration = Duration::from_secs(1);
/// Shortest gap between two orphan load polls (m1close Q6: two consecutive polls at least 1 s apart).
const ORPHAN_LOAD_POLL_MIN: Duration = Duration::from_secs(1);
/// Poll period while waiting for the drain and settle phases of shutdown.
const SHUTDOWN_POLL: Duration = Duration::from_millis(20);
/// Extra settle time after the cancel-acknowledgement bound during shutdown.
const SHUTDOWN_SETTLE_GRACE: Duration = Duration::from_secs(1);

/// Protocol timing bounds of the server. Not operator policy, so not in the runtime config; tests shorten them.
#[derive(Debug, Clone)]
pub struct Tuning {
    pub backend: Timeouts,
    /// Deadline of the readiness smoke request.
    pub smoke_deadline: Duration,
    /// After `/abort_request`, how long the backend stream may take to end before the request fails
    /// `cancel_unacknowledged` (its admission slot stays charged until the stream closes).
    pub cancel_ack_timeout: Duration,
    /// Idle period after which an SSE stream gets a `: keepalive` comment.
    pub keepalive_interval: Duration,
    /// Readiness probe period of a local worker. Production keeps Q8's [`PROBE_INTERVAL`]; tests shorten it.
    pub probe_interval: Duration,
    /// A local worker not probed healthy for this long reads suspect. Production: three [`PROBE_INTERVAL`]s; a
    /// test probing fast keeps a long one, so a stall of the test host does not age its workers.
    pub suspect_after: Duration,
    /// Period of a remote worker's readiness driver between registry round bumps (its Q8 probe round: residency,
    /// describe, reconcile, smoke). Production keeps [`PROBE_INTERVAL`]; tests shorten it apart from
    /// `probe_interval`.
    pub remote_interval: Duration,
    /// The clock orphan retention reads: the ledger opens at it (an orphan already past `admitted_ms +
    /// terminal_retention_ms` is released at open) and a held orphan is released when it reaches that bound.
    /// Production reads the system clock; a test of the hold moves a [`ManualClock`].
    pub retention_clock: RetentionClock,
}

/// See [`Tuning::retention_clock`].
#[derive(Debug, Clone, Default)]
pub enum RetentionClock {
    /// The system clock (`unix_ms`).
    #[default]
    Wall,
    /// A test's clock: reads and moves only by [`ManualClock::set`].
    Manual(Arc<ManualClock>),
}

impl RetentionClock {
    /// Unix milliseconds on this clock.
    pub fn now_ms(&self) -> u64 {
        match self {
            Self::Wall => unix_ms(),
            Self::Manual(clock) => clock.now_ms(),
        }
    }

    /// Resolves once this clock reads `at_ms` or later.
    async fn reached(&self, at_ms: u64) {
        match self {
            Self::Wall => {
                tokio::time::sleep(Duration::from_millis(at_ms.saturating_sub(unix_ms()))).await;
            }
            Self::Manual(clock) => {
                // `self` owns the sender, so the wait ends only at `at_ms`.
                let _ = clock.now.subscribe().wait_for(|&now| now >= at_ms).await;
            }
        }
    }
}

/// A settable retention clock for tests ([`RetentionClock::Manual`]).
#[derive(Debug)]
pub struct ManualClock {
    now: watch::Sender<u64>,
}

impl ManualClock {
    /// A clock reading `now_ms`.
    pub fn new(now_ms: u64) -> Self {
        Self {
            now: watch::Sender::new(now_ms),
        }
    }

    pub fn now_ms(&self) -> u64 {
        *self.now.borrow()
    }

    /// Moves the clock to `now_ms` and wakes every hold waiting on it.
    pub fn set(&self, now_ms: u64) {
        self.now.send_replace(now_ms);
    }
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            backend: Timeouts {
                connect: Duration::from_secs(2),
                control: Duration::from_secs(10),
                // Covers prefill of a long prompt before the first streamed byte.
                stream_idle: Duration::from_secs(300),
            },
            smoke_deadline: Duration::from_secs(60),
            cancel_ack_timeout: Duration::from_secs(5),
            keepalive_interval: Duration::from_secs(15),
            probe_interval: PROBE_INTERVAL,
            suspect_after: 3 * PROBE_INTERVAL,
            remote_interval: PROBE_INTERVAL,
            retention_clock: RetentionClock::Wall,
        }
    }
}

/// The backend `cered` serves (spec m22 R4, `deployments.kind`), or a remote worker's (M3.5 Q11).
#[derive(Clone)]
pub(crate) enum Backend {
    /// `sglang_http`: a backend-managed SGLang server.
    SglangHttp(SglangHttp),
    /// `native_cuda`: the in-process executor, available once readiness filled the slot.
    Native(NativeSlot),
    /// A `[[federation.workers]]` agent, reached over the worker RPC.
    Remote(RemoteWorker),
}

impl Backend {
    pub(crate) fn sglang(&self) -> Option<&SglangHttp> {
        match self {
            Backend::SglangHttp(client) => Some(client),
            Backend::Native(_) | Backend::Remote(_) => None,
        }
    }

    /// The loaded native backend; `None` for `sglang_http` and before (or after a failed) native load.
    pub(crate) fn native(&self) -> Option<Arc<NativeBackend>> {
        match self {
            Backend::SglangHttp(_) | Backend::Remote(_) => None,
            Backend::Native(slot) => slot.get(),
        }
    }
}

/// Where readiness publishes the native backend: empty until the load, every identity check and the SMOKE
/// passed (`/readyz` ready), then filled. A native replica of a multi-replica deployment can be unloaded and
/// loaded again by the operator (m32n design note D10); the single form is filled once for the life of the
/// process. Clones share the slot.
#[derive(Clone, Default)]
pub struct NativeSlot(Arc<Mutex<SlotState>>);

#[derive(Default)]
struct SlotState {
    backend: Option<Arc<NativeBackend>>,
    /// D10: released by the operator; readiness reports the replica `unloaded` and loads nothing.
    unloaded: bool,
    /// D10: the operator asked for a load; the next probe runs the load again.
    reload: bool,
    /// D9 (b)/N4: the load after a failed re-smoke failed too; only a `cered` restart recovers the card.
    faulted: bool,
}

impl NativeSlot {
    pub fn get(&self) -> Option<Arc<NativeBackend>> {
        lock(&self.0).backend.clone()
    }

    /// Fills an empty slot; an unloaded one stays empty.
    pub(crate) fn fill(&self, backend: NativeBackend) {
        let mut slot = lock(&self.0);
        if slot.backend.is_none() && !slot.unloaded {
            slot.backend = Some(Arc::new(backend));
        }
    }

    /// D10 `unload`: takes the backend (released once its last request holding it ends) and keeps the slot empty
    /// until [`NativeSlot::request_load`].
    pub(crate) fn unload(&self) -> Option<Arc<NativeBackend>> {
        let mut slot = lock(&self.0);
        slot.unloaded = true;
        slot.reload = false;
        slot.backend.take()
    }

    /// D10 `load`: the next probe re-runs the load (a loaded slot is released first). A faulted slot stays as
    /// it is ([`NativeSlot::is_faulted`]).
    pub(crate) fn request_load(&self) -> Option<Arc<NativeBackend>> {
        let mut slot = lock(&self.0);
        if slot.faulted {
            return None;
        }
        slot.unloaded = false;
        slot.reload = true;
        slot.backend.take()
    }

    /// D9 (b): takes the backend of a card whose re-smoke failed (released once its last request holding it ends),
    /// so the reload that follows fills the slot again.
    pub(crate) fn release(&self) -> Option<Arc<NativeBackend>> {
        lock(&self.0).backend.take()
    }

    pub(crate) fn is_unloaded(&self) -> bool {
        lock(&self.0).unloaded
    }

    /// D9 (b)/N4: marks the card lost to a sticky fault (its one reload failed).
    pub(crate) fn set_faulted(&self) {
        lock(&self.0).faulted = true;
    }

    pub(crate) fn is_faulted(&self) -> bool {
        lock(&self.0).faulted
    }

    /// Whether a load was requested since the last call.
    pub(crate) fn take_reload(&self) -> bool {
        std::mem::take(&mut lock(&self.0).reload)
    }
}

/// The loaded and smoked native executor (spec m22 "Native path contract").
pub struct NativeBackend {
    /// The driver thread that owns the executor handle; every host call goes through it.
    pub driver: Driver,
    /// The sealed model.
    pub model: Model,
    /// The tokenizer sidecar: prompt token IDs per request.
    pub sidecar: Sidecar,
    /// The grammar process (spec m28 D7): `Some` iff the profile advertises a grammar output. The native probe
    /// reports its `health()` as the advisory `grammar` problem (respawning or down) without making the worker not
    /// ready; the tokenizer sidecar is unaffected.
    pub grammar: Option<GrammarSidecar>,
    /// The detokenizer's table (artifact `tokenizer.json`, loaded once with the model).
    pub vocab: Vocab,
    /// What the executor reported for the sealed model (equal to the deployment capabilities file).
    pub capabilities: ExecutionCapabilities,
    /// BC1 leftover (batch-concurrency design note "Prefill selection signature"): the sealed model's prefill
    /// selection digest as lowercase hex (`cere_model_prefill_signature`, read right after the seal); `None` when the
    /// library predates the call or its device has no selection. `/readyz` and `/v1/cere/stats` report it.
    pub prefill_signature: Option<String>,
    /// The model's stop tokens (artifact `generation_config.json`).
    pub stop_token_ids: Vec<u32>,
    /// The model's `vocab_size` (the logits row width): `top_k` "all" and the `logit_bias` id bound (m27 D8).
    pub vocab_size: u32,
    /// The model's sampling defaults (artifact `generation_config.json`, m27 D10): absent request fields take them.
    pub generation_defaults: GenerationDefaults,
    /// Context limit of every session (`deployments.native.max_context_tokens`).
    pub max_context_tokens: u64,
    /// The readiness instance id of this load.
    pub instance_id: String,
    /// Requests submitted to the executor (the SMOKE excluded; refusals and context errors never count), counted
    /// once their choices are queued (U1); `GET /v1/cere/stats` reports it as `native_submissions`.
    pub submissions: AtomicU64,
    /// Speculate rounds (`cere_completion.verify_steps`) of every completed ticket since the load (m29 design note
    /// D11); `GET /v1/cere/stats` reports it as `native_spec_verify_steps`. 0 without speculation.
    pub spec_verify_steps: AtomicU64,
    /// Accepted drafts (`cere_completion.accepted_drafts`) of every completed ticket since the load; `GET
    /// /v1/cere/stats` reports it as `native_spec_accepted_drafts`.
    pub spec_accepted_drafts: AtomicU64,
    /// m23 design note PI-02/PI-04: the active levers, read back from the executor at load and after every hot
    /// swap (`cere_model_selection`). A request copies them once (its `drafter` picks the `cere_work.flags` draft bits,
    /// m29l design note D6); the write lock is held across a swap, so a request sees one whole set.
    pub levers: RwLock<LeverSet>,
    /// PI-04: `deployments.native.lever_hot_swap`; `POST /v1/cere/levers` is refused without it.
    pub lever_hot_swap: bool,
    /// `grammar_step` wall times and Constrain->commit gaps of constrained requests (spec m28 D8); `GET
    /// /v1/cere/stats` reports them as `grammar` when the backend has a grammar process.
    pub(crate) grammar_stats: crate::native::serve::GrammarStats,
    /// U1 preemption (M2.11 D8) when `deployments.native.preempt` is set.
    pub(crate) preemption: Option<crate::native::serve::Preemption>,
    /// Library, profile and state layout an exported session carries (M2.11 D3, D9).
    pub identity: StateIdentity,
    /// Session creates waiting for KV pages (ledger CR1): pages are reserved at create, so a request the pool cannot
    /// cover yet is queued here until a session is released.
    pub(crate) sessions: SessionQueue,
    /// m32n design note D3/D18: the card this replica runs on (`GPU-<uuid>`).
    pub device_uuid: String,
    /// D9 (b): set when a request's executor call ended with an internal or unknown status or the driver stopped;
    /// the readiness probe of a multi-replica deployment then re-smokes the card and, if that fails, loads it once
    /// more.
    pub(crate) fault: std::sync::atomic::AtomicBool,
    /// D11 (2)/D13: the prefix-affinity keys this card's cache holds (in memory only).
    pub(crate) affinity: crate::native::affinity::AffinityMap,
    /// D18: tokens this card committed since its load.
    pub(crate) committed: AtomicU64,
    /// D18: requests whose PREFILL reported `cached_tokens` > 0 (prefix-cache hits) since the load.
    pub(crate) prefix_hits: AtomicU64,
    /// D18: when and at which `committed` count [`NativeBackend::decode_rate`] last read the card.
    pub(crate) rate_mark: Mutex<(Instant, u64)>,
    /// m32n design note D22 (3): how this load ran (`GET /v1/cere/cards` `load`).
    pub(crate) load: LoadReport,
    /// loop-guard design note: `deployments.native.loop_guard` and its counters; `None` (the default) is off.
    pub(crate) loop_guard: Option<crate::native::loop_guard::LoopGuard>,
    /// NetTeleport S3: the recent prompts a graceful stop parks (`[state] park_on_stop`); `None` when off.
    pub(crate) parked: Option<crate::native::park::Recorder>,
}

/// m32n design note D22 (3): how a native load ran. A replica's load after its first runs while the other card
/// serves, in the background: its ALLOCATE (and the executor threads started there) at nice 19 on 2 CPUs, its
/// staging threads at nice 19 through 2 staging buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LoadReport {
    pub(crate) background: bool,
    /// The nice value of the load's staging threads.
    pub(crate) nice: i32,
    /// The nice value and CPU count of the thread ALLOCATE ran on.
    pub(crate) allocate_nice: i32,
    pub(crate) allocate_cpus: usize,
    /// Staging buffers of the pass: the cap on copies in flight.
    pub(crate) staging_slots: usize,
    /// The most staging buffers' worth of bytes in flight during the pass.
    pub(crate) copies_in_flight_max: u64,
}

impl NativeBackend {
    /// m32n design note D18: tokens per second this card committed since the previous call (since its load on
    /// the first).
    pub(crate) fn decode_rate(&self) -> f64 {
        let now = Instant::now();
        let committed = self.committed.load(Ordering::Acquire);
        let (at, before) = std::mem::replace(&mut *lock(&self.rate_mark), (now, committed));
        let secs = now.duration_since(at).as_secs_f64();
        if secs > 0.0 {
            committed.saturating_sub(before) as f64 / secs
        } else {
            0.0
        }
    }

    /// U3 (M2.11 D9): the idle `session`'s state in this backend's envelope, in a locked, self-zeroing buffer.
    pub fn export_state(&self, session: &Session) -> Result<Exported, ExportError> {
        export::export(&self.driver, session, &self.identity)
    }

    /// U3 (M2.11 D9): a fresh session holding `exported`'s state; a mismatched envelope is refused before import.
    pub fn import_state(&self, exported: &Exported) -> Result<Session, ExportError> {
        export::import(
            &self.driver,
            &self.model,
            self.max_context_tokens,
            &self.identity,
            exported,
        )
    }

    /// NetTeleport S1 (D-T3): this backend's `state_compat_id` under its active levers.
    pub fn state_compat_id(&self) -> [u8; 32] {
        crate::native::teleport::compat_id(
            &self.identity.library_sha256,
            &self.identity.layout_id,
            &self.active_levers(),
        )
    }

    /// NetTeleport S1: the deepest cached prefix state of `tokens` under `namespace`, in a locked buffer, with its
    /// token count (BUSY retried until `deadline`). Blocking.
    pub fn export_prefix_state(
        &self,
        namespace: [u8; 16],
        tokens: &[u32],
        max_blob_bytes: u64,
        deadline: Instant,
    ) -> Result<(export::LockedBuf, u64), crate::native::teleport::Refusal> {
        crate::native::teleport::export_image(
            &self.driver,
            &self.model,
            &self.identity,
            namespace,
            tokens,
            max_blob_bytes,
            deadline,
        )
    }

    /// NetTeleport S1: inserts an opened prefix state image under `namespace`; its token count. Blocking.
    pub fn import_prefix_state(
        &self,
        namespace: [u8; 16],
        image: &export::LockedBuf,
        deadline: Instant,
    ) -> Result<u64, crate::native::teleport::Refusal> {
        crate::native::teleport::import_image(
            &self.driver,
            &self.model,
            &self.identity,
            namespace,
            image,
            deadline,
        )
    }

    /// m23 design note PI-02: the active levers, copied once per request (no lookup on the hot path).
    pub fn active_levers(&self) -> LeverSet {
        *self.levers.read().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Why `POST /v1/cere/levers` (m23 design note PI-04) refused a swap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeverSwapError {
    /// No native backend is loaded (`sglang_http`, or before the native load).
    NotLoaded,
    /// The deployment does not set `lever_hot_swap` (403 `lever_hot_swap_disabled`).
    Disabled,
    /// An unknown lever or impl, a value the deployment cannot run, or an unknown or out-of-range launch param
    /// (400 `invalid_lever`).
    Invalid(String),
    /// A build param (SelfTune D6, O292 D-9): fixed at library build (409 `lever_build_param`).
    BuildParam(String),
    /// A restart-class lever would change (409 `lever_restart_only`).
    RestartOnly(&'static str),
    /// A request is in flight (409 `requests_in_flight`).
    InFlight,
    /// The executor refused otherwise, or reported a set outside the registry.
    Executor(String),
}

/// One configured backend worker (m3 design note C3): its own backend client, readiness probe and smoke.
pub struct Replica {
    /// The worker id: the replica's device (Q3).
    worker_id: String,
    /// The index of the `[[deployments]]` entry it serves (M5 D5).
    deployment: usize,
    /// The configured `base_url`, journaled with every `dispatched` record (Q9); empty for `native_cuda`.
    base_url: String,
    pub(crate) backend: Backend,
    readiness: Readiness,
    /// The replica's `load_endpoint` (m1close Q6), polled while a recovered orphan is charged to this worker.
    load_endpoint: Option<String>,
    /// m1close Q6 `C`: `cered` streams confirmed inside this replica's scheduler.
    pub(crate) confirmed: ConfirmedStreams,
    /// Orphans charged to this worker whose abort was sent and that only wait for their release (excluded from
    /// its open backend streams at shutdown).
    retained: Arc<AtomicU64>,
    /// The worker's current epoch and the backend identity behind it; also serialises its publishes.
    track: Mutex<WorkerTrack>,
}

/// A worker's current epoch (Q4) and the backend identity behind it.
struct WorkerTrack {
    epoch: u64,
    /// The first `instance_id` confirmed at `epoch`, kept through outages: a backend that comes back with
    /// another one was relaunched.
    instance_id: Option<String>,
}

impl Replica {
    /// The worker id (the replica's device).
    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    /// The index of the `[[deployments]]` entry this worker serves (M5 D5).
    pub fn deployment(&self) -> usize {
        self.deployment
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// This replica's readiness (tests force probes through it).
    pub fn readiness(&self) -> &Readiness {
        &self.readiness
    }

    /// The worker's current epoch (Q4).
    pub fn epoch(&self) -> u64 {
        lock(&self.track).epoch
    }

    /// The backend identity confirmed at the worker's current epoch, if any yet.
    pub fn instance_id(&self) -> Option<String> {
        lock(&self.track).instance_id.clone()
    }
}

/// One served model (M5 D5): a `[[deployments]]` entry's served name and `cere` identity (its chat validation policy
/// is live, in [`Settings`]).
pub(crate) struct Served {
    pub(crate) model: String,
    pub(crate) identity: DeploymentIdentity,
    /// The deployment's context limit (`max_model_len` of its `/v1/models` card; null without one).
    pub(crate) max_context_tokens: Option<u64>,
}

pub struct App {
    /// The startup config: the authority for every `restart` key ([`crate::reload::KEYS`]).
    pub(crate) config: RuntimeConfig,
    /// The applied settings; a live reload ([`App::reload`]) swaps the `Arc`, a request keeps the one it took.
    settings: RwLock<Arc<Settings>>,
    /// Serializes reloads.
    reload_lock: Mutex<()>,
    pub(crate) tuning: Tuning,
    pub(crate) auth: Authenticator,
    /// Every served model in `[[deployments]]` order (M5 D5); never empty.
    pub(crate) served: Vec<Served>,
    /// `config_hash()` of the startup config, the `cere` identity hash; a live reload never moves it.
    pub(crate) config_hash: String,
    /// The workers in configuration order; never empty.
    replicas: Vec<Replica>,
    pub(crate) rids: Arc<RidGenerator>,
    /// Q2: the `cache_salt` of each auth scope, when `prefix_cache_isolation = "cache_salt"`; empty otherwise.
    pub(crate) cache_salts: HashMap<String, String>,
    pub(crate) admission: Admission,
    /// Continue-mode stream buffers (m1close Q3-Q5).
    pub(crate) resume: Arc<ResumeStore>,
    /// ab-split design note AB-03/AB-04: per entry, the A/B arms' stats and the sessions' arms.
    pub(crate) ab: crate::ab::AbState,
    ledger: Arc<Mutex<Ledger>>,
    /// Worker health and active work, fed by every readiness publish ([`App::publish_worker`]).
    workers: Mutex<WorkerRegistry>,
    /// Cancel signal of every executor that has not finished, by rid.
    active: Mutex<HashMap<String, watch::Sender<Option<&'static str>>>>,
    stopping: AtomicBool,
    epoch: u64,
    /// Orphans charged to no worker (Q9) whose abort was sent and that only wait for their release; a worker's
    /// own are counted on its [`Replica`].
    unattributed_retained: Arc<AtomicU64>,
    background: Mutex<Vec<JoinHandle<()>>>,
    /// M3.5 remote workers (`[federation]`), their registry service and its bound address.
    remote: Option<Arc<RemoteRegistry>>,
    /// The remote workers in `[[federation.workers]]` order (Q11), dispatched to like replicas.
    remotes: Vec<RemoteWorker>,
    registry_listener: Mutex<Option<RegistryListener>>,
    federation_addr: Option<SocketAddr>,
    /// The failed checks of each remote worker's readiness round (its driver writes, `/readyz` reads).
    remote_problems: Arc<RemoteProblems>,
    /// Track LoRA: the native entry's adapter registry, when `[deployments.native.lora]` is set.
    pub(crate) lora: Option<Arc<crate::native::lora::Lora>>,
    /// WeightLab (`-v57`): the native entry's research registry, when `[deployments.native.research]` is set.
    pub(crate) research: Option<Arc<crate::native::research::Research>>,
    /// ops-hardening design note OH-05: what `GET /metrics` exports beyond the live reads.
    pub(crate) metrics: crate::metrics::Metrics,
    /// net-teleport design note 4.4 (S7): one usage row per request terminal; `None` with `usage.enabled = false`.
    pub(crate) usage: Option<crate::usage::UsageJournal>,
    /// NetTeleport S3 (L1 warm restart): the parked-state store when `[state] park_on_stop` is set; the shutdown
    /// parks the recorded sessions' prefix states through it after the drain.
    pub(crate) park: Option<Arc<crate::native::park::Park>>,
    /// net-teleport design note 4.4 (S6): open owner-only spans, recorded into `usage` as they end.
    pub(crate) owner_only_spans: Mutex<crate::pool::OwnerOnlyTracker>,
}

/// Why [`App::set_drain_intent`] changed nothing.
#[derive(Debug)]
pub enum DrainError {
    /// Neither a configured local nor a configured remote worker.
    UnknownWorker,
    /// The journal write failed; the intent is unchanged.
    Journal(LedgerError),
}

/// m32n design note D18: a native replica's card as configured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeCard {
    pub device_uuid: String,
    /// The executor's decode rows (the profile's `decode_batch_max`).
    pub rows_max: u64,
    /// The card's KV pool in tokens.
    pub kv_pool_tokens: u64,
}

/// Why [`App::native_residency`] changed nothing (m32n design note D10).
#[derive(Debug, PartialEq, Eq)]
pub enum NativeResidencyError {
    /// `unload` of a replica without the operator's drain intent (409 `worker_not_draining`).
    NotDraining,
    /// `unload` while admission still holds a request on the replica (409 `requests_in_flight`).
    InFlight,
    /// `load` of a replica whose reload after a sticky fault failed (409 `worker_faulted`, N4).
    Faulted,
}

impl App {
    /// Opens the ledger and starts recovery aborts and the readiness probes. Must run inside a tokio runtime.
    /// `operator_token` enables the drain routes (M3.5 Q13); `None` disables them.
    pub fn open(
        config: RuntimeConfig,
        credentials: &[ScopeCredential],
        operator_token: Option<crate::config::SecretToken>,
        tuning: Tuning,
    ) -> Result<Arc<Self>, String> {
        let mut served = Vec::with_capacity(config.deployments.len());
        for (index, entry) in config.deployments.iter().enumerate() {
            let identity = DeploymentIdentity::from_config(&config, index)
                .ok_or("cered serves a live backend kind with a pinned deployment")?;
            let model = entry
                .served_model_name
                .clone()
                .ok_or_else(|| format!("cered needs deployments[{index}].served_model_name"))?;
            served.push(Served {
                model,
                identity,
                max_context_tokens: entry.lock.as_ref().map(|d| d.profile.max_context_tokens),
            });
        }
        let settings = Settings::new(config.clone(), 0)?;
        let config_hash = settings.config_hash.clone();
        let configured = configured_replicas(&config)?;
        let caps = (0..config.deployments.len())
            .map(|index| worker_cap(&config, index))
            .collect::<Result<Vec<u64>, String>>()?;
        // M5 D2(e): every remote worker serves the entry its `expect.served_model_name` names.
        let remote_deployments = config
            .federation
            .iter()
            .flat_map(|fed| &fed.workers)
            .map(|w| {
                config.deployment_of_worker(&w.worker_id).ok_or_else(|| {
                    format!(
                        "federation worker {} serves no configured deployment",
                        w.worker_id
                    )
                })
            })
            .collect::<Result<Vec<usize>, String>>()?;
        // Q2: the coordinator's identity is checked before any state is touched.
        let pki = match &config.federation {
            Some(fed) => Some(
                Pki::load(&fed.ca_file, &fed.cert_file, &fed.key_file, &fed.principal)
                    .map_err(|e| format!("federation PKI: {e}"))?,
            ),
            None => None,
        };
        let state_dir = config.ledger.state_dir.clone();
        let keys = DigestKeys::open(&state_dir)
            .map_err(|e| format!("cannot open server secret in {}: {e}", state_dir.display()))?;
        let park = crate::native::park::Park::open(&config)
            .map_err(|e| format!("cannot open the park key in {}: {e}", state_dir.display()))?
            .map(Arc::new);
        let cache_salts = match config.deployments[0].prefix_cache_isolation() {
            PrefixCacheIsolation::CacheSalt => config
                .auth
                .scopes
                .iter()
                .map(|s| (s.scope_id.clone(), keys.cache_salt(&s.scope_id)))
                .collect(),
            PrefixCacheIsolation::None => HashMap::new(),
        };
        // Q9: device -> base_url; the ledger registers each local worker at its next epoch (Q4). Remote workers
        // are known by endpoint and never bumped: the coordinator journals only epochs their agents register.
        let mut base_urls: BTreeMap<String, WorkerTarget> = configured
            .iter()
            .map(|(_, r)| {
                let target = WorkerTarget::Local {
                    base_url: r.base_url.clone(),
                };
                (r.device.clone(), target)
            })
            .collect();
        for w in config.federation.iter().flat_map(|fed| &fed.workers) {
            let target = WorkerTarget::Remote {
                endpoint: w.endpoint.clone(),
            };
            base_urls.insert(w.worker_id.clone(), target);
        }
        let (mut ledger, recovery) = Ledger::open(
            LedgerLimits::from_config(&config),
            Arc::new(keys),
            &state_dir,
            &base_urls,
            tuning.retention_clock.now_ms(),
        )
        .map_err(|e| format!("cannot open ledger in {}: {e:?}", state_dir.display()))?;
        // net-teleport design note 4.4 (C18): per-scope live-session caps.
        ledger.set_session_scope_caps(crate::pool::session_caps(&config));
        let epoch = recovery.epoch;
        let worker_epochs = ledger.worker_epochs();
        // Q13: operator drain intent survives restarts.
        let drain_intents = ledger.drain_intents();
        // m36 design note Q8: so does residency intent (the registry's `desired` and `intent_seq`).
        let residency_intents = ledger.residency_intents();
        let rids = Arc::new(RidGenerator::new(epoch));
        // Q7/Q10: one aging rule for the registry and admission.
        let aging = Aging {
            suspect_after_ms: tuning.suspect_after.as_millis() as u64,
            lost_after_ms: config.scheduler.worker_lost_after().as_millis() as u64,
        };
        let admission = Admission::with_policy(
            AdmissionLimits::from_config(&config.admission),
            SchedulePolicy::from_config(&config),
        );
        // P7: a remote worker's cap is `max_inflight` until its inventory is read (the view pump lowers it).
        let remote_caps = config.federation.iter().flat_map(|fed| {
            let aging = RemoteTiming::from_config(fed).aging();
            fed.workers
                .iter()
                .zip(&remote_deployments)
                .map(move |(w, &deployment)| WorkerCap {
                    id: w.worker_id.clone(),
                    cap: w.max_inflight,
                    aging,
                    deployment,
                })
        });
        // M5 D6: each local worker's cap is its own entry's.
        admission.set_workers(
            configured
                .iter()
                .map(|&(deployment, ref r)| WorkerCap {
                    id: r.device.clone(),
                    cap: caps[deployment],
                    aging,
                    deployment,
                })
                .chain(remote_caps)
                .collect(),
        );
        // net-teleport design note 4.4: worker owners, tiers and owner-only windows.
        crate::pool::configure_workers(
            &admission,
            &config,
            configured.iter().map(|(_, r)| r.device.as_str()),
        );
        // m32n design note D6/D11: each native replica of a multi-replica deployment admits by its KV pool.
        if let Some(pool) = crate::readiness::replica_pool_tokens(&config) {
            for (_, r) in &configured {
                admission.set_pool(&r.device, pool);
            }
        }
        // Review A1 pass 2: a probe blocks on its re-smoke, and the probe's publishes are the worker's sightings,
        // so the smoke (queue wait included) gives up within half the suspect + lost bound. A Suspect worker whose
        // re-smoke waits for one of its own slots is then sighted again before it could age lost (Q4 rule 3).
        let smoke_deadline = tuning.smoke_deadline.min(Duration::from_millis(
            aging.suspect_after_ms.saturating_add(aging.lost_after_ms) / 2,
        ));
        let now = unix_ms();
        let mut workers = WorkerRegistry::new(aging);
        let remote_ids: Vec<&str> = config
            .federation
            .iter()
            .flat_map(|fed| fed.workers.iter().map(|w| w.worker_id.as_str()))
            .collect();
        for id in drain_intents
            .iter()
            .filter(|id| !remote_ids.contains(&id.as_str()))
        {
            workers.set_drain_intent(id, true);
        }
        let mut replicas = Vec::with_capacity(configured.len());
        // m32n design note R3: the native replicas open and hash-check the one library once.
        let library = crate::readiness::SharedLibrary::default();
        // Track LoRA: the native entry's adapter registry (the single form only, so one executor holds the pool).
        let lora = config
            .deployments
            .iter()
            .find_map(|d| d.native.as_ref()?.lora.clone())
            .map(|l| Arc::new(crate::native::lora::Lora::new(l)));
        // WeightLab: the native entry's research registry (the single form only: config refuses replicas).
        let research = config.deployments.iter().find_map(|d| {
            let n = d.native.as_ref()?;
            let r = n.research.clone()?;
            let scopes = config
                .auth
                .scopes
                .iter()
                .map(|s| s.scope_id.clone())
                .collect();
            Some(Arc::new(crate::native::research::Research::new(
                r,
                n.max_context_tokens,
                scopes,
            )))
        });
        let mut ordinal = 0;
        for (deployment, r) in configured {
            let worker_epoch = worker_epochs
                .get(&r.device)
                .copied()
                .ok_or_else(|| format!("the ledger registered no epoch for worker {}", r.device))?;
            let (backend, readiness) = match config.deployments[deployment].kind {
                BackendKind::NativeCuda => {
                    let slot = NativeSlot::default();
                    let readiness = Readiness::native(
                        &config,
                        slot.clone(),
                        worker_epoch,
                        ordinal,
                        library.clone(),
                        lora.clone(),
                        research.clone(),
                        park.clone(),
                    )?;
                    ordinal += 1;
                    (Backend::Native(slot), readiness)
                }
                _ => {
                    let client =
                        SglangHttp::new(&r.base_url, tuning.backend).map_err(|e| e.to_string())?;
                    let readiness = Readiness::new(
                        client.clone(),
                        &r.device,
                        &config,
                        deployment,
                        rids.clone(),
                        admission.clone(),
                        smoke_deadline,
                    )?;
                    (Backend::SglangHttp(client), readiness)
                }
            };
            workers
                .register(&r.device, worker_epoch, now)
                .map_err(|e| format!("cannot register worker {}: {e:?}", r.device))?;
            replicas.push(Replica {
                worker_id: r.device,
                deployment,
                base_url: r.base_url,
                backend,
                readiness: readiness.with_probe_interval(tuning.probe_interval),
                load_endpoint: r.load_endpoint,
                confirmed: ConfirmedStreams::default(),
                retained: Arc::default(),
                track: Mutex::new(WorkerTrack {
                    epoch: worker_epoch,
                    instance_id: None,
                }),
            });
        }
        if recovery.truncated_tail_bytes > 0 {
            eprintln!(
                "cered recovery: cut a torn journal tail of {} bytes",
                recovery.truncated_tail_bytes
            );
        }
        let retention_ms = config.ledger.terminal_retention_ms;
        let unattributed_retained = Arc::new(AtomicU64::new(0));
        let ledger = Arc::new(Mutex::new(ledger));
        // M3.5 D1: one client per remote worker, for its readiness driver and its dispatch (Q11).
        let remote_clients = match (&config.federation, &pki) {
            (Some(fed), Some(pki)) => fed
                .workers
                .iter()
                .map(|w| {
                    WorkerClient::connect(
                        &w.endpoint,
                        pki,
                        &w.principal,
                        ClientTiming::from_config(fed),
                    )
                    .map_err(|e| format!("federation worker {}: {e}", w.worker_id))
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => Vec::new(),
        };
        // M3.5 P6: the registry service of the remote workers, under this process epoch as the term (Q6). M5 D8:
        // each managed worker is gated on its own deployment's lock.
        let remote_locks: BTreeMap<String, String> = config
            .federation
            .iter()
            .flat_map(|fed| &fed.workers)
            .zip(&remote_deployments)
            .map(|(w, &d)| (w.worker_id.clone(), served[d].identity.lock_sha256.clone()))
            .collect();
        let (remote, registry_listener) = match (&config.federation, &pki) {
            (Some(fed), Some(pki)) => {
                let registry = Arc::new(RemoteRegistry::new(
                    fed,
                    epoch,
                    &drain_intents,
                    &residency_intents,
                    &remote_locks,
                ));
                let listener =
                    RegistryListener::spawn(registry.clone(), Arc::downgrade(&ledger), fed, pki)?;
                (Some(registry), Some(listener))
            }
            _ => (None, None),
        };
        let remotes: Vec<RemoteWorker> = match (&config.federation, &remote) {
            (Some(fed), Some(registry)) => fed
                .workers
                .iter()
                .zip(&remote_clients)
                .map(|(worker, client)| {
                    RemoteWorker::new(
                        worker.clone(),
                        client.clone(),
                        &fed.cluster_id,
                        &fed.coordinator_id,
                        registry.clone(),
                    )
                })
                .collect(),
            _ => Vec::new(),
        };
        let federation_addr = registry_listener.as_ref().map(RegistryListener::local_addr);
        let mut background = Vec::with_capacity(recovery.orphans.len() + recovery.recharged.len());
        // M5 D5: `model=` names the model of the dispatched worker's deployment (`-` when there is none) on every
        // startup recovery line.
        let model_of = |dispatched: Option<&(String, u64)>| {
            dispatched
                .and_then(|(w, _)| config.deployment_of_worker(w))
                .map_or("-", |d| served[d].model.as_str())
        };
        let usage = crate::usage::UsageJournal::open(&config)?;
        for orphan in &recovery.orphans {
            if let Some(u) = &usage {
                u.record(crate::usage::UsageRow::of_orphan(
                    &config,
                    orphan,
                    unix_ms(),
                ));
            }
            let model = model_of(orphan.dispatched.as_ref());
            eprintln!(
                "cered request rid={} scope={} worker={} epoch={} state=failed reason=worker_restart model={}",
                orphan.rid,
                orphan.scope,
                worker_field(orphan.dispatched.as_ref()),
                orphan.epoch,
                model
            );
        }
        let native = config.deployments[0].kind == BackendKind::NativeCuda;
        if !native {
            for orphan in &recovery.recharged {
                eprintln!(
                    "cered recovery: re-charged rid={} worker={} scope={} model={} (held when the previous incarnation stopped)",
                    orphan.rid,
                    worker_field(orphan.dispatched.as_ref()),
                    orphan.scope,
                    model_of(orphan.dispatched.as_ref())
                );
            }
        }
        let orphans = recovery
            .orphans
            .into_iter()
            .chain(recovery.recharged.iter().cloned());
        // One load poller per worker that holds an orphan and has a `load_endpoint`; each stops once every hold
        // listening to it (each owning a receiver) has ended.
        let mut loads: HashMap<String, watch::Receiver<LoadPoll>> = HashMap::new();
        for orphan in orphans {
            let worker_log = worker_field(orphan.dispatched.as_ref());
            let model = model_of(orphan.dispatched.as_ref());
            if native {
                eprintln!(
                    "cered recovery: not charged rid={} worker={worker_log} scope={} model={model} (the native executor ended with its process)",
                    orphan.rid, orphan.scope
                );
                continue;
            }
            // The orphan's backend request may still run: keep its slot charged on the cap that holds it (Q9)
            // until the retention bound since its admission, or until the Q6 load check of its worker releases it
            // once its abort was sent (spec L3). A never-dispatched one was released by the ledger already.
            let worker = match orphan.worker.charged_to(&base_urls) {
                OrphanCharge::NotHeld => {
                    eprintln!(
                        "cered recovery: released rid={} worker={worker_log} cause={} model={model}",
                        orphan.rid,
                        OrphanReleaseCause::NotDispatched.as_str()
                    );
                    continue;
                }
                OrphanCharge::Worker(id) => Some(id),
                OrphanCharge::Global => None,
            };
            if matches!(orphan.worker, OrphanWorker::Unconfigured { .. }) {
                eprintln!(
                    "cered recovery: rid={} worker={worker_log} unconfigured model={model}",
                    orphan.rid
                );
            }
            // M3.5 Q15: charged to a remote worker, the orphan waits for its agent's cleanup report in this term
            // or its retention bound ([`RemoteOrphan`]); reconciliation sends its CancelWork.
            let remote_index = worker
                .as_deref()
                .and_then(|id| remotes.iter().position(|r| r.worker_id() == id));
            if let (Some(i), Some(registry), Some(fed)) =
                (remote_index, &remote, &config.federation)
            {
                let period = Duration::from_millis(config.ledger.orphan_load_poll_ms());
                let hold = RemoteOrphan {
                    worker_id: remotes[i].worker_id().to_owned(),
                    client: remote_clients[i].clone(),
                    registry: registry.clone(),
                    cluster_id: fed.cluster_id.clone(),
                    coordinator_id: fed.coordinator_id.clone(),
                    ledger: Arc::downgrade(&ledger),
                    retained: remotes[i].retained().clone(),
                    period: period.max(ORPHAN_LOAD_POLL_MIN),
                    release_ms: orphan.admitted_ms.saturating_add(retention_ms),
                };
                let permit = admission.charge_orphan(&orphan.scope, Some(&hold.worker_id));
                background.push(tokio::spawn(hold.run(orphan, permit)));
                continue;
            }
            let replica = worker
                .as_deref()
                .and_then(|id| replicas.iter().find(|r| r.worker_id == id));
            let load = replica.and_then(|r| {
                let client = r.backend.sglang()?;
                let endpoint = r.load_endpoint.clone()?;
                let rx = loads.entry(r.worker_id.clone()).or_insert_with(|| {
                    let (tx, rx) = watch::channel(LoadPoll::default());
                    let period = Duration::from_millis(config.ledger.orphan_load_poll_ms());
                    background.push(tokio::spawn(poll_load(
                        client.clone(),
                        endpoint,
                        period.max(ORPHAN_LOAD_POLL_MIN),
                        r.confirmed.clone(),
                        tx,
                    )));
                    rx
                });
                Some(rx.clone())
            });
            let clients = orphan
                .worker
                .abort_targets(&base_urls)
                .iter()
                .filter_map(|id| replicas.iter().find(|r| &r.worker_id == id))
                .filter_map(|r| r.backend.sglang().cloned())
                .collect();
            let permit = admission.charge_orphan(&orphan.scope, worker.as_deref());
            let hold = OrphanHold {
                clients,
                ledger: Arc::downgrade(&ledger),
                retained: replica
                    .map_or_else(|| unattributed_retained.clone(), |r| r.retained.clone()),
                release_ms: orphan.admitted_ms.saturating_add(retention_ms),
                clock: tuning.retention_clock.clone(),
                load,
            };
            background.push(tokio::spawn(hold.run(orphan, permit)));
        }
        drop(loads);
        let resume = Arc::new(ResumeStore::new(&config.api, retention_ms));
        let ab = crate::ab::AbState::new(served.len());
        let metrics = crate::metrics::Metrics::new(&config.telemetry.clients);
        let app = Arc::new(Self {
            config_hash,
            auth: Authenticator::new(
                credentials,
                operator_token.as_ref(),
                config.auth.keyless_scope.as_deref(),
            ),
            config,
            settings: RwLock::new(Arc::new(settings)),
            reload_lock: Mutex::new(()),
            tuning,
            served,
            replicas,
            rids,
            cache_salts,
            admission,
            resume,
            ab,
            ledger,
            workers: Mutex::new(workers),
            active: Mutex::new(HashMap::new()),
            stopping: AtomicBool::new(false),
            epoch,
            unattributed_retained,
            background: Mutex::new(background),
            remote,
            remotes,
            registry_listener: Mutex::new(registry_listener),
            federation_addr,
            remote_problems: Arc::new(RemoteProblems::default()),
            lora,
            research,
            metrics,
            usage,
            park,
            owner_only_spans: Mutex::new(crate::pool::OwnerOnlyTracker::new(unix_ms())),
        });
        // S6: owner-only (opt-out) spans into the usage journal.
        if let Some(task) = crate::pool::spawn_span_recorder(&app) {
            lock(&app.background).push(task);
        }
        // Hooks before the first probe, so no publish escapes the registry and admission. Weak: a probe task
        // never keeps the `App` alive.
        for (index, replica) in app.replicas.iter().enumerate() {
            let weak = Arc::downgrade(&app);
            replica.readiness.on_publish(move |snapshot| {
                if let Some(app) = weak.upgrade() {
                    app.publish_worker(index, snapshot);
                }
            });
        }
        let probes: Vec<_> = app.replicas.iter().map(|r| r.readiness.spawn()).collect();
        lock(&app.background).extend(probes);
        // M3.5 Q9, Q10: one readiness driver per remote worker, woken by its round bumps and every probe interval;
        // P7: and the view pump that feeds admission every remote worker's view and cap. M3.6 Q8, M5 D8: each
        // driver's residency calls name its own deployment's lock.
        if let (Some(registry), Some(fed)) = (&app.remote, &app.config.federation) {
            let mut drivers = Vec::with_capacity(fed.workers.len());
            for ((worker, client), &d) in fed
                .workers
                .iter()
                .zip(remote_clients)
                .zip(&remote_deployments)
            {
                let deployment_lock: [u8; 32] =
                    crate::digest::unhex(&app.served[d].identity.lock_sha256)
                        .and_then(|bytes| bytes.try_into().ok())
                        .ok_or("the deployment lock is not a hex SHA-256")?;
                drivers.push(
                    RemoteDriver {
                        worker: worker.clone(),
                        cluster_id: fed.cluster_id.clone(),
                        coordinator_id: fed.coordinator_id.clone(),
                        registry: registry.clone(),
                        client,
                        ledger: Arc::downgrade(&app.ledger),
                        rids: app.rids.clone(),
                        smoke_deadline,
                        interval: app.tuning.remote_interval,
                        problems: app.remote_problems.clone(),
                        admission: app.admission.clone(),
                        deployment_lock,
                    }
                    .spawn(),
                );
            }
            lock(&app.background).extend(drivers);
            let pump = remote::spawn_view_pump(
                registry.clone(),
                app.admission.clone(),
                fed.workers.clone(),
                Arc::downgrade(&app.ledger),
                fed.heartbeat_interval() / 2,
            );
            lock(&app.background).push(pump);
        }
        Ok(app)
    }

    /// The process epoch (journal incarnation).
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The remote workers of `[federation]` (M3.5), `None` without federation.
    pub fn remote_workers(&self) -> Option<&RemoteRegistry> {
        self.remote.as_deref()
    }

    /// The bound address of the registry service, `None` without federation.
    pub fn federation_addr(&self) -> Option<SocketAddr> {
        self.federation_addr
    }

    /// Every remote worker's configuration and state at `now_ms`, in `[[federation.workers]]` order.
    pub fn remote_entries(&self, now_ms: u64) -> Vec<(&RemoteWorkerConfig, RemoteState)> {
        let (Some(registry), Some(fed)) = (&self.remote, &self.config.federation) else {
            return Vec::new();
        };
        fed.workers.iter().zip(registry.list(now_ms)).collect()
    }

    /// The remote worker `worker_id`, if configured (Q11).
    pub(crate) fn remote(&self, worker_id: &str) -> Option<&RemoteWorker> {
        self.remotes.iter().find(|r| r.worker_id() == worker_id)
    }

    /// Every remote worker, in `[[federation.workers]]` order (Q11).
    pub(crate) fn remotes(&self) -> &[RemoteWorker] {
        &self.remotes
    }

    /// A grant on remote worker `worker_id` at `epoch` (Q11): the worker, the authority of its work from its
    /// registry state now, and the instance id of its inventory. `None` unless the worker is Ready at that epoch
    /// (the grant is then rejected like a local `begin_work`).
    pub(crate) fn remote_grant(
        &self,
        worker_id: &str,
        epoch: u64,
    ) -> Option<(RemoteWorker, AuthorityBase, Option<String>)> {
        let worker = self.remote(worker_id)?;
        let state = self.remote.as_deref()?.get(worker_id, unix_ms())?;
        let at = worker.authority(&state, epoch)?;
        let instance_id = state.inventory.and_then(|i| i.instance_id);
        Some((worker.clone(), at, instance_id))
    }

    /// H2: the slot (`permit`) of `call` on remote `worker`, whose work may still hold the agent's resources, is
    /// held until the agent reports the operation cleaned up or the request's retention bound
    /// ([`remote::hold_orphan`]). Outside a runtime it is released at once.
    pub(crate) fn hold_remote_orphan(
        &self,
        worker: &RemoteWorker,
        call: Arc<RemoteCall>,
        permit: Permit,
    ) {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let period = Duration::from_millis(self.config.ledger.orphan_load_poll_ms());
        let task = runtime.spawn(remote::hold_orphan(
            call,
            permit,
            Arc::downgrade(&self.ledger),
            worker.retained().clone(),
            period.max(ORPHAN_LOAD_POLL_MIN),
            self.config.ledger.terminal_retention_ms,
        ));
        let mut background = lock(&self.background);
        // One hold per held call: the finished ones are dropped here, so the list stays bounded.
        background.retain(|t| !t.is_finished());
        background.push(task);
    }

    /// The failed readiness checks of remote worker `worker_id` in its round `round`.
    pub fn remote_problems(&self, worker_id: &str, round: u64) -> Vec<Problem> {
        self.remote_problems.get(worker_id, round)
    }

    /// Q13: journals and applies operator drain intent (`draining`) or its clearing for a configured local or
    /// remote worker, in any state. A local worker's admission view follows at once; a remote worker's intent is
    /// sent as `desired_state` in its heartbeat replies and survives Lost and every higher epoch.
    pub fn set_drain_intent(&self, worker_id: &str, draining: bool) -> Result<(), DrainError> {
        if let Some(remote) = self.remote.as_deref().filter(|r| r.contains(worker_id)) {
            remote
                .set_drain_intent(&self.ledger, worker_id, draining, unix_ms())
                .map_err(DrainError::Journal)?;
            return Ok(());
        }
        if self.replica(worker_id).is_none() {
            return Err(DrainError::UnknownWorker);
        }
        // Lock order: ledger, then workers; the admission view is pushed under the ledger lock, so concurrent
        // drains and undrains apply in journal order.
        let mut ledger = self.ledger();
        ledger
            .set_drain(worker_id, draining, unix_ms())
            .map_err(DrainError::Journal)?;
        let view = {
            let mut workers = self.workers();
            workers.set_drain_intent(worker_id, draining);
            workers.get(worker_id).map(|w| w.view())
        };
        if let Some(view) = view {
            self.admission.update_worker(worker_id, view);
        }
        drop(ledger);
        Ok(())
    }

    /// m36 design note Q8-Q10: journals a new residency intent (`load` = resident, `unload` = unloaded) for a
    /// remote worker and pushes its admission view before returning, so an `unloaded` worker takes no new grant
    /// once the route answers (review M36-05). A local worker's model belongs to its backend (`Unmanaged`).
    pub fn set_residency(
        &self,
        worker_id: &str,
        desired: Residency,
    ) -> Result<ModelView, ResidencyError> {
        let remote = self
            .remote
            .as_deref()
            .zip(self.config.federation.as_ref())
            .and_then(|(registry, fed)| {
                fed.workers
                    .iter()
                    .find(|w| w.worker_id == worker_id)
                    .map(|w| (registry, w))
            });
        let Some((registry, worker)) = remote else {
            return Err(if self.replica(worker_id).is_some() {
                ResidencyError::Unmanaged
            } else {
                ResidencyError::UnknownWorker
            });
        };
        let model = registry.set_residency(&self.ledger, worker_id, desired, unix_ms())?;
        remote::push_view(registry, &self.admission, worker);
        Ok(model)
    }

    /// m32n design note D10: `unload` releases the executor of a local native replica of a multi-replica
    /// deployment (drained, nothing in flight) and keeps it unloaded; `load` makes its next probe run the load
    /// again (Ready at epoch + 1 on success). Answers the replica's new native state (`unloaded` or `loading`);
    /// `None` for every other worker, which [`Self::set_residency`] handles.
    pub(crate) fn native_residency(
        &self,
        worker_id: &str,
        desired: Residency,
    ) -> Option<Result<&'static str, NativeResidencyError>> {
        let replica = self.replica(worker_id)?;
        let Backend::Native(slot) = &replica.backend else {
            return None;
        };
        if !self.routes_by_pool(replica.deployment) {
            return None;
        }
        Some(match desired {
            Residency::Unloaded => {
                if !self.workers().drain_intended(worker_id) {
                    return Some(Err(NativeResidencyError::NotDraining));
                }
                let inflight = self
                    .admission
                    .worker_loads("")
                    .iter()
                    .find(|w| &*w.id == worker_id)
                    .map_or(0, |w| w.inflight);
                if inflight > 0 {
                    return Some(Err(NativeResidencyError::InFlight));
                }
                drop(slot.unload());
                eprintln!("cered worker: {worker_id}: native replica unloaded by the operator");
                Ok("unloaded")
            }
            Residency::Resident => {
                if slot.is_faulted() {
                    return Some(Err(NativeResidencyError::Faulted));
                }
                drop(slot.request_load());
                eprintln!(
                    "cered worker: {worker_id}: native replica load requested by the operator"
                );
                Ok("loading")
            }
        })
    }

    /// m4 design note Q4-Q6 (design-m4.md §3): moves the owner of `req.session_id` to `req.to_worker` in replay
    /// mode through [`Ledger::migrate_session`], whose refusals change nothing. The destination (and, with
    /// `force`, the current owner) is checked on the ledger guard that publishes the move: a configured worker
    /// is eligible only while Ready. A forced takeover that broke a held lease cancels that turn
    /// `session_owner_superseded` on the same guard (inline, not through [`Self::cancel`], which would lock the
    /// ledger again), so no other writer sees the session between the move and the cancel.
    pub fn migrate_session(&self, req: MigrateRequest<'_>) -> Result<Migrated, MigrateError> {
        let now = unix_ms();
        // Lock order: a remote worker's state is read before the ledger guard (the remote registry locks its
        // workers, then the ledger); a local worker's under it (ledger, then workers).
        let remote: HashMap<String, WorkerState> = self
            .remote_entries(now)
            .into_iter()
            .map(|(config, state)| (config.worker_id.clone(), state.state))
            .collect();
        let check = |worker: &str| {
            let Some(deployment) = self.config.deployment_of_worker(worker) else {
                return WorkerCheck::Unknown;
            };
            let state = match remote.get(worker) {
                Some(state) => Some(*state),
                None if self.replica(worker).is_some() => {
                    self.local_worker_state(worker).map(|(state, _)| state)
                }
                None => return WorkerCheck::Unknown,
            };
            if state == Some(WorkerState::Ready) {
                WorkerCheck::Eligible(deployment)
            } else {
                WorkerCheck::Ineligible(deployment)
            }
        };
        let deployment_of_model = |model: &str| self.deployment_of_model(model);
        let (migrated, cancelled) = {
            let mut ledger = self.ledger();
            let migrated = ledger.migrate_session(req, &check, &deployment_of_model, now)?;
            let mut cancelled = None;
            if let Some(rid) = &migrated.superseded {
                let outcome = ledger.request_cancel(
                    &migrated.scope,
                    rid,
                    REASON_SESSION_OWNER_SUPERSEDED,
                    now,
                );
                if let Ok(outcome @ (CancelOutcome::Cancelled | CancelOutcome::AbortRequired)) =
                    outcome
                {
                    if let Some(tx) = self.active().get(rid) {
                        tx.send_replace(Some(REASON_SESSION_OWNER_SUPERSEDED));
                    }
                    if outcome == CancelOutcome::Cancelled {
                        cancelled = Some(rid.clone());
                    }
                }
            }
            (migrated, cancelled)
        };
        if let Some(rid) = &cancelled {
            self.log_terminal(&migrated.scope, rid);
        }
        eprintln!(
            "cered event=session_owner session={} worker={} generation={} cause={}",
            migrated.session_id,
            migrated.worker,
            migrated.owner_generation,
            if req.force { "force" } else { "migrate" }
        );
        Ok(migrated)
    }

    /// The applied settings (live keys); a request takes them once, at acceptance.
    pub fn settings(&self) -> Arc<Settings> {
        self.settings
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// A live config reload ([`crate::reload`]): re-reads the file the server started from, validates it, and
    /// either applies every change at once (each changed key `live`, every hook prepared) or nothing. Without a
    /// change it is a no-op: generation and hash stay. Logs one stderr line either way.
    pub fn reload(&self) -> Result<Reloaded, ReloadError> {
        let _serial = lock(&self.reload_lock);
        let result = self.reload_locked();
        match &result {
            Ok(r) if r.applied => eprintln!(
                "cered config reload generation={} config_hash={} changed={}",
                r.generation,
                r.config_hash,
                r.changed.join(",")
            ),
            Ok(r) => eprintln!(
                "cered config reload unchanged generation={} config_hash={}",
                r.generation, r.config_hash
            ),
            Err(e) => eprintln!("cered config reload refused: {e}"),
        }
        result
    }

    fn reload_locked(&self) -> Result<Reloaded, ReloadError> {
        if self.is_stopping() {
            return Err(ReloadError::Stopping);
        }
        let source = self
            .config
            .source
            .as_ref()
            .ok_or(ReloadError::Unavailable)?;
        let candidate = source
            .read()
            .map_err(|e| ReloadError::Invalid(e.to_string()))?;
        let running = self.settings();
        let changed = reload::diff(&running.config, &candidate);
        let refused = reload::refused(&changed);
        if !refused.is_empty() {
            return Err(ReloadError::Restart(refused));
        }
        if changed.is_empty() {
            return Ok(Reloaded {
                applied: false,
                generation: running.generation,
                config_hash: running.config_hash.clone(),
                changed,
            });
        }
        let mut applies = Vec::with_capacity(reload::HOOKS.len());
        for hook in reload::HOOKS {
            match (hook.prepare)(self, &running.config, &candidate) {
                Ok(Some(apply)) => applies.push(apply),
                Ok(None) => {}
                Err(reason) => {
                    return Err(ReloadError::Hook {
                        name: hook.name,
                        reason,
                    })
                }
            }
        }
        let next = Arc::new(
            Settings::new(candidate, running.generation + 1).map_err(ReloadError::Invalid)?,
        );
        for apply in applies {
            apply(self);
        }
        self.admission.reconfigure(
            AdmissionLimits::from_config(&next.config.admission),
            SchedulePolicy::from_config(&next.config),
        );
        *self.settings.write().unwrap_or_else(|e| e.into_inner()) = next.clone();
        Ok(Reloaded {
            applied: true,
            generation: next.generation,
            config_hash: next.config_hash.clone(),
            changed,
        })
    }

    /// Every served model in `[[deployments]]` order (M5 D5).
    pub(crate) fn served(&self) -> &[Served] {
        &self.served
    }

    /// The `cere` identity of entry `index` (M5 D5); panics past the last entry.
    pub fn deployment(&self, index: usize) -> &DeploymentIdentity {
        &self.served[index].identity
    }

    /// The index of the entry serving `model`, if served (M5 D5).
    pub fn deployment_of_model(&self, model: &str) -> Option<usize> {
        self.served.iter().position(|s| s.model == model)
    }

    /// Track LoRA: the entry whose native backend holds the adapter pool, when `[deployments.native.lora]` is set.
    pub(crate) fn lora_deployment(&self) -> Option<usize> {
        self.lora.as_ref()?;
        self.config
            .deployments
            .iter()
            .position(|d| d.native.as_ref().is_some_and(|n| n.lora.is_some()))
    }

    /// Track LoRA: the entry serving adapter `name` and a lease on it (held by the request until its sessions are
    /// released), if loaded.
    pub(crate) fn lora_adapter(
        &self,
        name: &str,
    ) -> Option<(usize, Arc<crate::native::lora::Adapter>)> {
        let deployment = self.lora_deployment()?;
        Some((deployment, self.lora.as_ref()?.lease(name)?))
    }

    /// Whether some worker of entry `deployment` (local or remote) is Ready now (M5 D5: else its requests answer
    /// today's 503 `not_ready`).
    pub(crate) fn deployment_ready(&self, deployment: usize) -> bool {
        self.replicas
            .iter()
            .filter(|r| r.deployment == deployment)
            .any(|r| {
                matches!(
                    self.worker_state(&r.worker_id),
                    Some((WorkerState::Ready, _))
                )
            })
            || self.remote_entries(unix_ms()).iter().any(|(config, s)| {
                s.state == WorkerState::Ready
                    && self.config.deployment_of_worker(&config.worker_id) == Some(deployment)
            })
    }

    /// The workers in configuration order (never empty).
    pub fn replicas(&self) -> &[Replica] {
        &self.replicas
    }

    /// The worker `worker_id`, if configured.
    pub fn replica(&self, worker_id: &str) -> Option<&Replica> {
        self.replicas.iter().find(|r| r.worker_id == worker_id)
    }

    /// The position of worker `worker_id` in [`Self::replicas`], if configured.
    pub(crate) fn replica_index(&self, worker_id: &str) -> Option<usize> {
        self.replicas.iter().position(|r| r.worker_id == worker_id)
    }

    /// Replica `index`. Panics past the last replica.
    pub(crate) fn replica_at(&self, index: usize) -> &Replica {
        &self.replicas[index]
    }

    /// Replica `index`'s readiness (tests force probes through it). Panics past the last replica.
    pub fn readiness(&self, index: usize) -> &Readiness {
        &self.replicas[index].readiness
    }

    /// m1close Q6 `C` summed over the workers: `cered` streams confirmed inside a backend scheduler right now (a
    /// content chunk received, no finish chunk, terminal or abort yet, and the relay not waiting for client
    /// room).
    pub fn confirmed_streams(&self) -> u64 {
        self.replicas.iter().map(|r| r.confirmed.count()).sum()
    }

    /// A loaded native backend (`native_cuda` once readiness passed): the first replica's that is loaded; `None`
    /// for `sglang_http`. Every replica runs the same library, profile and levers (m32n design note D2).
    pub fn native(&self) -> Option<Arc<NativeBackend>> {
        self.replicas.iter().find_map(|r| r.backend.native())
    }

    /// Every loaded native backend, in replica order (counters are summed over the cards).
    pub fn natives(&self) -> Vec<Arc<NativeBackend>> {
        self.replicas
            .iter()
            .filter_map(|r| r.backend.native())
            .collect()
    }

    /// m32n design note D11/D12: whether entry `deployment` routes by KV pool and prefix affinity (a native
    /// deployment with `[[deployments.native.replicas]]`); the single form and every other kind do not.
    pub(crate) fn routes_by_pool(&self, deployment: usize) -> bool {
        self.config.deployments[deployment]
            .native
            .as_ref()
            .is_some_and(|n| n.replicas.is_some())
    }

    /// m32n design note D18: the card of worker `worker_id` when it is a native replica of a multi-replica
    /// deployment; `None` for the single form and every other worker.
    pub(crate) fn native_card(&self, worker_id: &str) -> Option<NativeCard> {
        let replica = self.replica(worker_id)?;
        if !matches!(replica.backend, Backend::Native(_))
            || !self.routes_by_pool(replica.deployment)
        {
            return None;
        }
        let entry = &self.config.deployments[replica.deployment];
        let device_uuid = entry
            .native
            .as_ref()?
            .replica_list(&self.config.scheduler.devices)
            .into_iter()
            .find(|r| r.device == worker_id)?
            .device_uuid;
        Some(NativeCard {
            device_uuid,
            rows_max: entry.lock.as_ref()?.profile.decode_batch_max,
            kv_pool_tokens: crate::readiness::replica_pool_tokens(&self.config)?,
        })
    }

    /// D12: a Ready, loaded native replica of entry `deployment`, whose sidecar encodes a request before its grant
    /// (every replica runs the same tokenizer lock); `None` while none is.
    pub(crate) fn live_native(&self, deployment: usize) -> Option<Arc<NativeBackend>> {
        self.replicas
            .iter()
            .filter(|r| r.deployment == deployment)
            .filter(|r| {
                matches!(
                    self.local_worker_state(&r.worker_id),
                    Some((WorkerState::Ready, _))
                )
            })
            .find_map(|r| r.backend.native())
    }

    /// D11 (2), D13: the worker of entry `deployment` whose card holds affinity `key` (the first, in replica order).
    pub(crate) fn affinity_worker(
        &self,
        deployment: usize,
        key: &crate::native::affinity::Key,
    ) -> Option<Arc<str>> {
        self.replicas
            .iter()
            .filter(|r| r.deployment == deployment)
            .find(|r| r.backend.native().is_some_and(|n| n.affinity.holds(key)))
            .map(|r| Arc::from(r.worker_id.as_str()))
    }

    /// Process-level readiness problems (m3 design note Q11): `journal_poisoned` once a journal write failure
    /// poisoned the ledger (no admission can succeed until a restart recovers from disk).
    pub fn process_problems(&self) -> Vec<Problem> {
        if !self.ledger().is_journal_poisoned() {
            return Vec::new();
        }
        vec![Problem {
            field: PROBLEM_JOURNAL_POISONED.to_owned(),
            detail: "a journal write failed; no request can be admitted until cered restarts"
                .to_owned(),
        }]
    }

    /// The worker's lifecycle state (aged now, Q10) and epoch in the registry; a remote worker's from its
    /// coordinator-side registry (epoch 0 before its first accepted Register).
    pub fn worker_state(&self, worker_id: &str) -> Option<(WorkerState, u64)> {
        if let Some(registry) = self.remote.as_deref().filter(|r| r.contains(worker_id)) {
            return registry
                .get(worker_id, unix_ms())
                .map(|s| (s.state, s.epoch.unwrap_or(0)));
        }
        self.local_worker_state(worker_id)
    }

    /// A local worker's lifecycle state (aged now) and epoch; `None` for an id that is not a local worker.
    fn local_worker_state(&self, worker_id: &str) -> Option<(WorkerState, u64)> {
        let workers = self.workers();
        let aging = workers.aging_of(worker_id);
        workers
            .get(worker_id)
            .map(|w| (w.state_at(unix_ms(), aging), w.epoch))
    }

    /// Why no new request can be admitted now: `None` once the journal is healthy and at least one worker is
    /// ready (Q10), else the process problems or every worker's state and readiness problem fields.
    pub(crate) fn not_serving(&self) -> Option<String> {
        let problems = self.process_problems();
        if !problems.is_empty() {
            return Some(fields(&problems));
        }
        let states: Vec<_> = self
            .replicas
            .iter()
            .map(|r| self.worker_state(&r.worker_id))
            .collect();
        let remotes = self.remote_entries(unix_ms());
        if states
            .iter()
            .any(|s| matches!(s, Some((WorkerState::Ready, _))))
            || remotes.iter().any(|(_, s)| s.state == WorkerState::Ready)
        {
            return None;
        }
        let why: Vec<String> = self
            .replicas
            .iter()
            .zip(states)
            .map(|(r, state)| {
                let problems = r.readiness.snapshot().problems;
                let detail = if problems.is_empty() {
                    state.map_or("unknown", |(s, _)| s.as_str()).to_owned()
                } else {
                    fields(&problems)
                };
                format!("{}: {detail}", r.worker_id)
            })
            .chain(remotes.iter().map(|(config, s)| {
                let problems = self.remote_problems(&config.worker_id, s.round);
                let detail = if problems.is_empty() {
                    s.state.as_str().to_owned()
                } else {
                    fields(&problems)
                };
                format!("{}: {detail}", config.worker_id)
            }))
            .collect();
        Some(why.join("; "))
    }

    /// The publish hook of replica `index` (m3 design note Q4, Q7), run before each of its readiness snapshots
    /// becomes visible. A confirmed `instance_id` other than the one behind the worker's epoch (a relaunch), or a
    /// confirmed identity on a lost epoch (a rejoin), first journals the worker's next epoch and re-registers it
    /// there (joining), so the smoke that follows and every later grant use the new epoch. Then the probe outcome
    /// is observed into the registry (a joining worker stays joining until it is ready; a not-ready probe that
    /// confirmed the epoch's own `instance_id` is a sighting: suspect, but not aging lost) and the worker's view
    /// is pushed to admission. A bump supersedes the old epoch's in-flight requests ([`App::supersede`]).
    fn publish_worker(&self, index: usize, snapshot: &ReadinessSnapshot) {
        let replica = &self.replicas[index];
        let id = replica.worker_id.as_str();
        let mut track = lock(&replica.track);
        let now = unix_ms();
        let health = {
            let workers = self.workers();
            let aging = workers.aging_of(id);
            workers.get(id).map(|w| w.health_at(now, aging))
        };
        let confirmed = snapshot.instance_id.as_deref();
        let cause = match (track.instance_id.as_deref(), confirmed) {
            (Some(known), Some(new)) if known != new => Some("relaunch"),
            (_, Some(_)) if health == Some(WorkerHealth::Lost) => Some("rejoin"),
            _ => None,
        };
        // Q4 rule 3 (review A1): every pin matched and the identity behind the epoch answers, so the worker is
        // seen even while its re-smoke waits for one of its own slots (or failed): suspect, never aged lost.
        let sighted =
            cause.is_none() && confirmed.is_some() && track.instance_id.as_deref() == confirmed;
        let mut bump_failed = false;
        if let Some(cause) = cause {
            let old = track.epoch;
            // Bound first: a guard in the scrutinee would hold the ledger through the arms (`supersede`).
            let bumped = self.ledger().bump_worker_epoch(id, now);
            match bumped {
                Ok(epoch) => {
                    // A higher epoch always registers (joining).
                    let _ = self.workers().register(id, epoch, now);
                    *track = WorkerTrack {
                        epoch,
                        instance_id: None,
                    };
                    eprintln!(
                        "cered worker: {id} epoch {old} -> {epoch} ({cause}, instance_id={})",
                        confirmed.unwrap_or("-")
                    );
                    self.supersede(replica, old, cause == "relaunch", now);
                }
                Err(e) => {
                    // Not ready at an epoch that no longer names its backend; the next publish retries.
                    eprintln!(
                        "cered worker: cannot journal a new epoch for {id} ({cause}): {}",
                        e.code()
                    );
                    bump_failed = true;
                }
            }
        }
        if !bump_failed && track.instance_id.is_none() {
            track.instance_id = confirmed.map(str::to_owned);
        }
        let healthy = snapshot.ready && !bump_failed;
        let mut workers = self.workers();
        let joining = workers
            .get(id)
            .is_some_and(|w| w.health == WorkerHealth::Joining);
        if healthy || !joining {
            // A lost epoch is recorded lost here; it rejoins at a new epoch once its identity is confirmed.
            let _ = if !healthy && sighted {
                workers.sighted(id, track.epoch, now)
            } else {
                workers.observe(id, track.epoch, healthy, now)
            };
        }
        let view = workers.get(id).map(|w| w.view());
        drop(workers);
        if let Some(view) = view {
            self.admission.update_worker(id, view);
        }
    }

    /// Q10: worker `replica`'s epoch `old` was superseded (`relaunched`: a new `instance_id`; else a rejoin).
    /// Every request still running there fails `worker_epoch_superseded` (a `cancel_pending` one is its
    /// executor's to settle: it sent the abort already). Each is failed and its executor signalled under one
    /// ledger lock, which a cancel takes to signal too: the signal is never overwritten, and the executor answers
    /// the client without a terminal or an abort of its own. The terminal is logged and `/abort_request` goes
    /// to the worker. A relaunched backend holds none of the old epoch's work: the rid is released at once. A
    /// rejoined one may still run it: the rid stays charged to the worker and is held like a recovered orphan
    /// (aborted, then released at its retention bound or by the Q6 load check).
    fn supersede(&self, replica: &Replica, old: u64, relaunched: bool, now: u64) {
        let id = replica.worker_id.as_str();
        let superseded: Vec<_> = {
            let mut ledger = self.ledger();
            let stale = ledger.in_flight().into_iter().filter(|r| {
                r.state == RequestState::Running
                    && r.worker.as_ref().is_some_and(|(w, e)| w == id && *e == old)
            });
            let superseded: Vec<_> = stale
                .filter(|r| {
                    // An error is a journal failure: the ledger is poisoned and recorded `journal_error`.
                    let failed = ledger.fail(&r.rid, REASON_WORKER_EPOCH_SUPERSEDED, None, now);
                    failed.is_ok()
                })
                .collect();
            let active = self.active();
            for r in &superseded {
                if let Some(tx) = active.get(&r.rid) {
                    tx.send_replace(Some(REASON_WORKER_EPOCH_SUPERSEDED));
                }
            }
            superseded
        };
        let client = replica.backend.sglang();
        let worker = (id.to_owned(), old);
        // One load poller for this worker's holds; it stops once every hold listening to it has ended.
        let load = match (&replica.load_endpoint, client) {
            (Some(endpoint), Some(client)) if !relaunched && !superseded.is_empty() => {
                let (tx, rx) = watch::channel(LoadPoll::default());
                let period = Duration::from_millis(self.config.ledger.orphan_load_poll_ms());
                lock(&self.background).push(tokio::spawn(poll_load(
                    client.clone(),
                    endpoint.clone(),
                    period.max(ORPHAN_LOAD_POLL_MIN),
                    replica.confirmed.clone(),
                    tx,
                )));
                Some(rx)
            }
            _ => None,
        };
        for request in superseded {
            self.log_terminal(&request.scope, &request.rid);
            if relaunched {
                let cause = OrphanReleaseCause::Relaunch;
                let released = self.ledger().release_orphan(&request.rid, cause, now);
                let worker = worker_field(Some(&worker));
                if let Err(e) = released {
                    eprintln!(
                        "cered recovery: cannot journal the release rid={} worker={worker}: {e:?}",
                        request.rid
                    );
                }
                eprintln!(
                    "cered recovery: released rid={} worker={worker} cause={}",
                    request.rid,
                    cause.as_str()
                );
                if let (Some(client), Ok(rid)) = (client.cloned(), Rid::parse(&request.rid)) {
                    tokio::spawn(async move {
                        if let Err(e) = client.abort(&rid).await {
                            eprintln!("cered abort rid={rid} worker={worker} failed: {e}");
                        }
                    });
                }
                continue;
            }
            let hold = OrphanHold {
                clients: client.into_iter().cloned().collect(),
                ledger: Arc::downgrade(&self.ledger),
                retained: replica.retained.clone(),
                release_ms: request
                    .admitted_ms
                    .saturating_add(self.config.ledger.terminal_retention_ms),
                clock: self.tuning.retention_clock.clone(),
                load: load.clone(),
            };
            let permit = self.admission.charge_orphan(&request.scope, Some(id));
            let orphan = Orphan {
                scope: request.scope,
                rid: request.rid,
                epoch: self.epoch,
                admitted_ms: request.admitted_ms,
                dispatched: Some(worker.clone()),
                worker: OrphanWorker::Known {
                    id: id.to_owned(),
                    base_url: replica.base_url.clone(),
                },
            };
            lock(&self.background).push(tokio::spawn(hold.run(orphan, permit)));
        }
    }

    pub fn admission_snapshot(&self) -> AdmissionSnapshot {
        self.admission.snapshot()
    }

    /// Requests the native executor accepted since process start (`GET /v1/cere/stats`); 0 for `sglang_http`
    /// and before the native load.
    pub fn native_submissions(&self) -> u64 {
        self.natives()
            .iter()
            .map(|n| n.submissions.load(Ordering::Acquire))
            .sum()
    }

    /// Speculate rounds of the native executor since the load (m29 design note D11, `GET /v1/cere/stats`); 0 for
    /// `sglang_http`, before the native load and without speculation.
    pub fn native_spec_verify_steps(&self) -> u64 {
        self.natives()
            .iter()
            .map(|n| n.spec_verify_steps.load(Ordering::Acquire))
            .sum()
    }

    /// Drafts the native executor's speculate rounds accepted since the load (`GET /v1/cere/stats`).
    pub fn native_spec_accepted_drafts(&self) -> u64 {
        self.natives()
            .iter()
            .map(|n| n.spec_accepted_drafts.load(Ordering::Acquire))
            .sum()
    }

    /// Generations the native executor preempted (M2.11 D8, `GET /v1/cere/stats`); 0 without `preempt`.
    pub fn native_preemptions(&self) -> u64 {
        self.natives()
            .iter()
            .filter_map(|n| n.preemption.as_ref().map(|p| p.preemptions()))
            .sum()
    }

    /// Generations that used their session up and grew it to their whole budget (sessions design note SE-09,
    /// `GET /v1/cere/stats`); 0 without `reserve_output_tokens`.
    pub fn native_grows(&self) -> u64 {
        self.natives()
            .iter()
            .filter_map(|n| n.preemption.as_ref().map(|p| p.grows()))
            .sum()
    }

    /// loop-guard design note LG-04: the `native_loop_guard` object of `GET /v1/cere/stats`
    /// (`repetition_stops`, `tool_repeat_refusals`, summed over the cards); `None` unless a loaded native backend
    /// has `[deployments.native.loop_guard]`.
    pub fn native_loop_guard(&self) -> Option<serde_json::Value> {
        let natives = self.natives();
        crate::native::loop_guard::LoopGuard::stats(
            natives.iter().filter_map(|n| n.loop_guard.as_ref()),
        )
    }

    /// The `grammar` object of `GET /v1/cere/stats` (spec m28 D8): p50/p99 of `grammar_step` wall time and of the
    /// Constrain->commit gap, in ms (`null` before a sample); `None` unless the native backend has a grammar process.
    pub fn native_grammar_stats(&self) -> Option<serde_json::Value> {
        self.native()
            .filter(|n| n.grammar.is_some())
            .map(|n| n.grammar_stats.render())
    }

    /// m23 design note PI-05: the active levers of the loaded native backend (every registry lever, the executor
    /// layer as `cere_model_selection` reported it); `None` for `sglang_http` and before the native load.
    pub fn native_levers(&self) -> Option<BTreeMap<String, String>> {
        self.native().map(|n| n.active_levers().to_table())
    }

    /// BC1 leftover: the loaded native backend's prefill selection signature (lowercase hex); `None` for
    /// `sglang_http`, before the native load and when the library reports none.
    pub fn native_prefill_signature(&self) -> Option<String> {
        self.native().and_then(|n| n.prefill_signature.clone())
    }

    /// PI-04: the dev hot swap. `changes` (lever name -> impl id) applies over the active set, and `params` (lever
    /// name -> param -> value, SelfTune D6, O292 D-9) are checked against each lever's next impl: a launch param in
    /// its registry range is accepted (each registered range is the one value the impl's selection carries), a build
    /// param or an out-of-range or unknown one is refused. The checks run in the route's refusal order: the flag, the
    /// registry and the model's drafts, the params, the restart class, then no request in flight (admission first;
    /// the executor's `CERE_ERR_BUSY` is the same refusal). Swaps are serialized by the levers' write lock, held until
    /// the executor's set is read back.
    pub fn swap_levers(
        &self,
        changes: &BTreeMap<String, String>,
        params: &BTreeMap<String, BTreeMap<String, u64>>,
    ) -> Result<BTreeMap<String, String>, LeverSwapError> {
        let native = self.native().ok_or(LeverSwapError::NotLoaded)?;
        if !native.lever_hot_swap {
            return Err(LeverSwapError::Disabled);
        }
        let mut active = native
            .levers
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let mut table = active.to_table();
        for (name, id) in changes {
            if levers::lever(name).is_none() {
                return Err(LeverSwapError::Invalid(format!("{name}: unknown lever")));
            }
            table.insert(name.clone(), id.clone());
        }
        let drafts = u32::from(native.capabilities.speculative_draft_tokens);
        let next = LeverSet::from_table(&table, drafts).map_err(LeverSwapError::Invalid)?;
        for (name, values) in params {
            let id = next
                .get(name)
                .ok_or_else(|| LeverSwapError::Invalid(format!("{name}: unknown lever")))?;
            for (param, value) in values {
                levers::check_hot_param(name, id, param, *value).map_err(|r| match r {
                    levers::ParamRefusal::Build(m) => LeverSwapError::BuildParam(m),
                    levers::ParamRefusal::Invalid(m) => LeverSwapError::Invalid(m),
                })?;
            }
        }
        for lever in levers::REGISTRY {
            if lever.class == levers::LeverClass::Restart
                && next.get(lever.name) != active.get(lever.name)
            {
                return Err(LeverSwapError::RestartOnly(lever.name));
            }
        }
        if self.admission.snapshot().inflight > 0 {
            return Err(LeverSwapError::InFlight);
        }
        match native.driver.set_levers(&native.model, &next) {
            Ok(()) => {}
            Err(e) if e.status() == Some(crate::native::ffi::StatusError::Busy) => {
                return Err(LeverSwapError::InFlight)
            }
            Err(e) => return Err(LeverSwapError::Executor(e.to_string())),
        }
        let raw = native
            .driver
            .levers(&native.model)
            .map_err(|e| LeverSwapError::Executor(e.to_string()))?;
        *active = LeverSet::from_ffi(&raw, &next).map_err(LeverSwapError::Executor)?;
        Ok(active.to_table())
    }

    pub fn is_stopping(&self) -> bool {
        self.stopping.load(Ordering::SeqCst)
    }

    pub(crate) fn ledger(&self) -> MutexGuard<'_, Ledger> {
        lock(&self.ledger)
    }

    /// The worker registry, the authority `begin_work` checks at dispatch (tests move a worker's epoch here
    /// between a grant and its `begin_work`, M3 Q7).
    pub fn workers(&self) -> MutexGuard<'_, WorkerRegistry> {
        lock(&self.workers)
    }

    fn active(&self) -> MutexGuard<'_, HashMap<String, watch::Sender<Option<&'static str>>>> {
        lock(&self.active)
    }

    /// Registers a new executor's cancel signal.
    pub(crate) fn register_active(&self, rid: &str) -> watch::Receiver<Option<&'static str>> {
        let (tx, rx) = watch::channel(None);
        self.active().insert(rid.to_owned(), tx);
        rx
    }

    pub(crate) fn unregister_active(&self, rid: &str) {
        self.active().remove(rid);
    }

    /// Scoped cancel (L6 endpoint, deadline, backpressure, disconnect, shutdown). A request that was running
    /// is now `cancel_pending`; its executor is signalled and settles the terminal after the backend abort.
    pub(crate) fn cancel(
        &self,
        scope: &str,
        rid: &str,
        reason: &'static str,
    ) -> Result<CancelOutcome, LedgerError> {
        // Signalled under the ledger lock, like a supersede (Q10): a cancel never overwrites that signal.
        let outcome = {
            let mut ledger = self.ledger();
            let outcome = ledger.request_cancel(scope, rid, reason, unix_ms())?;
            if matches!(
                outcome,
                CancelOutcome::Cancelled | CancelOutcome::AbortRequired
            ) {
                if let Some(tx) = self.active().get(rid) {
                    tx.send_replace(Some(reason));
                }
            }
            outcome
        };
        if outcome == CancelOutcome::Cancelled {
            self.log_terminal(scope, rid);
        }
        Ok(outcome)
    }

    /// One stderr line per request terminal: ids, state, reason, timings and the reported token counts (Q5);
    /// never content. Then the request's `cered access` line when it waited for this terminal (its client left
    /// first, [`crate::metrics`]).
    pub(crate) fn log_terminal(&self, scope: &str, rid: &str) {
        let Some(s) = self.ledger().status(scope, rid) else {
            return;
        };
        self.metrics.observe_terminal(&s);
        if let Some(u) = &self.usage {
            u.record(crate::usage::UsageRow::of_terminal(
                &self.config,
                &s,
                unix_ms(),
            ));
        }
        let deferred = self.metrics.deferred_access(&s);
        let t = s.timings;
        let u = s.usage;
        let ms = |v: Option<u64>| v.map_or_else(|| "-".to_owned(), |v| v.to_string());
        // M5 D5: the model is the one the request's worker serves; `-` before any dispatch. It follows `stream=` so
        // the Q11 field order (`worker=` right after `scope=`, then state, reason) parsers rely on stays.
        let model = s
            .worker_id
            .as_deref()
            .and_then(|w| self.config.deployment_of_worker(w))
            .map_or("-", |d| self.served[d].model.as_str());
        // A2: ` client=<c> workload=<w>` only with `[[telemetry.clients]]`, so a config without rules logs the
        // line byte-identical; before `research_plan=`, which stays last.
        let client = self.metrics.labels_of(&s).map_or_else(String::new, |l| {
            format!(" client={} workload={}", l.client, l.workload)
        });
        let worker = s.worker_id.zip(s.worker_epoch);
        // WeightLab (WL-C): ` research_plan=<sha256|->` last, only with `[deployments.native.research]`, so the
        // default-off line is byte-identical and the Q11 field order stays.
        let research = self.research.as_ref().map_or_else(String::new, |r| {
            format!(
                " research_plan={}",
                r.take_applied(&s.rid).as_deref().unwrap_or("-")
            )
        });
        eprintln!(
            "cered request rid={} scope={} worker={} state={} reason={} stream={} model={} queue_wait_ms={} \
             first_generated_ms={} first_visible_ms={} total_ms={} client_backpressure_ms={} prompt_tokens={} \
             cached_tokens={} cache_report={}{}{}",
            s.rid,
            s.scope,
            worker_field(worker.as_ref()),
            s.state.as_str(),
            s.reason.as_deref().unwrap_or("-"),
            s.stream,
            model,
            ms(t.queue_wait_ms),
            ms(t.first_generated_ms),
            ms(t.first_visible_ms),
            ms(t.total_ms),
            t.client_backpressure_ms,
            ms(u.prompt_tokens),
            ms(u.cached_tokens),
            u.cache_report.as_str(),
            client,
            research,
        );
        if let Some(line) = deferred {
            eprintln!("{line}");
        }
    }

    /// L8. Returns true only if every in-flight request settled through its executor and no backend stream
    /// (including recovery orphans) is still open.
    pub async fn shutdown(&self) -> bool {
        self.stopping.store(true, Ordering::SeqCst);
        // Queued requests are cancelled (`shutdown`) without a backend call; running ones may drain.
        self.admission.close();
        let drain = Duration::from_millis(self.settings().config.api.shutdown_drain_ms);
        self.wait_for(drain, || self.ledger().in_flight().is_empty())
            .await;
        // Bind first: a guard temporary in the `for` head would live through the loop, and `cancel` locks again.
        let remaining = self.ledger().in_flight();
        for request in remaining {
            let _ = self.cancel(&request.scope, &request.rid, REASON_SHUTDOWN);
        }
        self.wait_for(
            self.tuning.cancel_ack_timeout + SHUTDOWN_SETTLE_GRACE,
            || self.ledger().in_flight().is_empty() && self.open_backend() == 0,
        )
        .await;
        let forced = self.ledger().in_flight();
        for request in &forced {
            // A forced terminal: the executor did not settle in time; this stop is not reported as clean.
            let failed = self
                .ledger()
                .fail(&request.rid, REASON_SHUTDOWN, None, unix_ms());
            if failed.is_ok() {
                self.log_terminal(&request.scope, &request.rid);
            }
        }
        // NetTeleport S3: the executor drained; park the recorded sessions' prefix states (bounded by
        // `state.park_deadline_ms`) before the process exits.
        if let Some(park) = &self.park {
            self.park_states(park).await;
        }
        // The registry service answers heartbeats while requests drain; it stops before the ledger is released.
        let registry_listener = lock(&self.registry_listener).take();
        if let Some(listener) = registry_listener {
            listener.stop().await;
        }
        let open_backend = self.open_backend();
        // Await the aborted tasks: an orphan hold may be journaling a release, and the next incarnation needs
        // the state-dir lock this process holds until the ledger is dropped.
        let background: Vec<_> = lock(&self.background).drain(..).collect();
        for task in &background {
            task.abort();
        }
        for task in background {
            let _ = task.await;
        }
        // S6: the owner-only spans still open end now.
        crate::pool::record_owner_only_spans(self, true);
        // S7: every queued usage row is written and synced before the next incarnation starts.
        if let Some(u) = &self.usage {
            u.close();
        }
        // A poisoned journal lost at least one terminal write: the disk does not hold what memory reported.
        let journal_poisoned = self.ledger().is_journal_poisoned();
        let clean = forced.is_empty() && open_backend == 0 && !journal_poisoned;
        eprintln!(
            "cered shutdown clean={clean} forced_terminals={} open_backend_streams={open_backend} journal_poisoned={journal_poisoned}",
            forced.len()
        );
        clean
    }

    /// NetTeleport S3: [`crate::native::park::Park::park`] of every loaded native backend's recorded prompts under
    /// the live `[state]` limits, on the blocking pool.
    async fn park_states(&self, park: &Arc<crate::native::park::Park>) {
        let settings = self.settings();
        let state = &settings.config.state;
        let limits = crate::native::park::Limits {
            max_bytes: state.park_max_bytes(),
            max_blob_bytes: state.max_blob_bytes(),
            deadline: Instant::now() + Duration::from_millis(state.park_deadline_ms()),
            tier: settings.config.server.tier(),
        };
        for backend in self.replicas.iter().filter_map(|r| r.backend.native()) {
            let Some(recorded) = backend.parked.as_ref().map(|r| r.take()) else {
                continue;
            };
            let park = park.clone();
            let parked =
                tokio::task::spawn_blocking(move || park.park(&backend, recorded, &limits)).await;
            if let Err(e) = parked {
                eprintln!("cered state park: the park task ended without a result: {e}");
            }
        }
    }

    /// Admission slots of backend work that may still be open, counted per worker (review M3-12): each worker's
    /// charged slots except its orphans that were aborted (a remote worker's: held after their cancel or end)
    /// and only wait out their release, plus the slots charged to no worker (Q9 orphans) except those retained
    /// the same way.
    fn open_backend(&self) -> u64 {
        // Charges before retained counts: a hold uncounts itself before it releases its slot, so a release in
        // between shows the slot as open, never an open slot as retained.
        let mut unattributed = self.admission.snapshot().inflight;
        let mut open = 0;
        for load in self.admission.worker_loads("") {
            unattributed = unattributed.saturating_sub(load.inflight);
            let retained = match self.replica(&load.id) {
                Some(r) => r.retained.load(Ordering::SeqCst),
                None => self
                    .remote(&load.id)
                    .map_or(0, |r| r.retained().load(Ordering::SeqCst)),
            };
            open += load.inflight.saturating_sub(retained);
        }
        let retained = self.unattributed_retained.load(Ordering::SeqCst);
        open + unattributed.saturating_sub(retained)
    }

    async fn wait_for(&self, bound: Duration, done: impl Fn() -> bool) {
        let until = tokio::time::Instant::now() + bound;
        while !done() && tokio::time::Instant::now() < until {
            tokio::time::sleep(SHUTDOWN_POLL).await;
        }
    }
}

/// m1close Q6 `C`: the number of streams confirmed inside the backend scheduler, a lower bound of `cered`'s
/// share of the backend load. A stream counts while it holds a [`ConfirmedStream`]: from a backend content
/// chunk until its finish chunk, its terminal or its abort, except while its relay waits for client room (it
/// reads nothing then, so the backend may finish unseen; the next content chunk counts it again). Unary,
/// queued, in-transit, draining and smoke requests never count, so they can only delay an orphan's release.
#[derive(Clone, Default)]
pub(crate) struct ConfirmedStreams(Arc<AtomicU64>);

impl ConfirmedStreams {
    pub(crate) fn enter(&self) -> ConfirmedStream {
        self.0.fetch_add(1, Ordering::SeqCst);
        ConfirmedStream(self.0.clone())
    }

    fn count(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

/// One confirmed stream; dropping it uncounts the stream.
pub(crate) struct ConfirmedStream(Arc<AtomicU64>);

impl Drop for ConfirmedStream {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The latest orphan load poll: its sequence number and how many consecutive polls (ending with it) saw
/// `L ≤ C`. An error or `L > C` resets the streak.
#[derive(Clone, Copy, Default)]
struct LoadPoll {
    seq: u64,
    clear_streak: u64,
}

/// Polls `endpoint` every `period` while any orphan hold listens. `C` is sampled before and after `L` and the
/// smaller value is used, so a stream confirmed or finished while `L` was read never counts in its favour.
async fn poll_load(
    client: SglangHttp,
    endpoint: String,
    period: Duration,
    confirmed: ConfirmedStreams,
    tx: watch::Sender<LoadPoll>,
) {
    let mut poll = LoadPoll::default();
    let mut failing = false;
    loop {
        tokio::select! {
            _ = tokio::time::sleep(period) => {}
            _ = tx.closed() => return,
        }
        let before = confirmed.count();
        let load = client.load(&endpoint).await;
        let c = before.min(confirmed.count());
        poll.seq += 1;
        poll.clear_streak = match &load {
            Ok(l) if *l <= c => poll.clear_streak + 1,
            Ok(_) => 0,
            Err(e) => {
                if !failing {
                    eprintln!("cered recovery: orphan load poll failed ({e}); orphans stay held");
                }
                0
            }
        };
        failing = load.is_err();
        tx.send_replace(poll);
    }
}

/// One recovered orphan's charged slot (L3, m1close Q6/Q7).
struct OrphanHold {
    /// The backends of the orphan's abort targets (M3 Q9); none when no configured worker serves its URL.
    clients: Vec<SglangHttp>,
    /// Weak: a hold never keeps the ledger (and its state-dir lock) alive past the `App`.
    ledger: Weak<Mutex<Ledger>>,
    /// The retained count of the worker charged with the orphan, or of the slots charged to no worker.
    retained: Arc<AtomicU64>,
    /// `admitted_ms + terminal_retention_ms`: the latest release, never extended by a restart.
    release_ms: u64,
    /// The clock `release_ms` is read on ([`Tuning::retention_clock`]).
    clock: RetentionClock,
    /// Load polls of the worker charged with the orphan, when that replica has a `load_endpoint`.
    load: Option<watch::Receiver<LoadPoll>>,
}

impl OrphanHold {
    /// Aborts the orphan on every target at once ([`abort_orphan`]) and holds `permit` until its release bound:
    /// the retention bound, whatever the aborts' outcome (a dead target never holds the slot past it), or
    /// earlier, once every abort was sent, two consecutive load polls after them that saw `L ≤ C`. Journals the
    /// release.
    async fn run(self, orphan: Orphan, permit: Permit) {
        let worker = worker_field(orphan.dispatched.as_ref());
        let rid = match Rid::parse(&orphan.rid) {
            Ok(rid) => rid,
            Err(e) => {
                eprintln!(
                    "cered recovery: cannot abort rid={} worker={worker}: {e}",
                    orphan.rid
                );
                return;
            }
        };
        let Self {
            clients,
            ledger,
            retained,
            release_ms,
            clock,
            load,
        } = self;
        // Dropped once the slot is released: an abort still failing then is not retried.
        let (held, released) = watch::channel(());
        let aborts: Vec<_> = clients
            .into_iter()
            .map(|client| {
                tokio::spawn(abort_orphan(
                    client,
                    rid.clone(),
                    worker.clone(),
                    released.clone(),
                ))
            })
            .collect();
        drop(released);
        // Counted retained (aborted, waiting only for the release) once every abort was sent.
        let mut counted = false;
        let cause = tokio::select! {
            () = clock.reached(release_ms) => OrphanReleaseCause::Retention,
            () = async {
                futures_util::future::join_all(aborts).await;
                eprintln!(
                    "cered recovery: aborted rid={rid} worker={worker}; slot held until unix_ms={release_ms}"
                );
                retained.fetch_add(1, Ordering::SeqCst);
                counted = true;
                load_cleared(load).await;
            } => OrphanReleaseCause::Load,
        };
        if let Some(ledger) = ledger.upgrade() {
            if let Err(e) = lock(&ledger).release_orphan(&orphan.rid, cause, clock.now_ms()) {
                eprintln!(
                    "cered recovery: cannot journal the release rid={rid} worker={worker}: {e:?}"
                );
            }
        }
        eprintln!(
            "cered recovery: released rid={rid} worker={worker} cause={}",
            cause.as_str()
        );
        // Uncount first: shutdown may briefly see the slot as open, never an open slot as retained.
        if counted {
            retained.fetch_sub(1, Ordering::SeqCst);
        }
        drop(permit);
        drop(held);
    }
}

/// Sends `/abort_request` for an orphan's `rid` to one of its targets, retrying every second until the backend
/// accepts it or the orphan's slot was released (`released` closed). An attempt in flight at the release still
/// completes, so every target gets at least one.
async fn abort_orphan(
    client: SglangHttp,
    rid: Rid,
    worker: String,
    mut released: watch::Receiver<()>,
) {
    loop {
        let Err(e) = client.abort(&rid).await else {
            return;
        };
        if released.has_changed().is_err() {
            eprintln!(
                "cered recovery: abort failed rid={rid} worker={worker} ({e}); slot released, not retried"
            );
            return;
        }
        eprintln!("cered recovery: abort failed rid={rid} worker={worker} ({e}); retrying");
        tokio::select! {
            () = tokio::time::sleep(ORPHAN_ABORT_RETRY) => {}
            _ = released.changed() => return,
        }
    }
}

/// The Q11 `worker=` log field: `<id>@<epoch>` of a request's last dispatch, `-` before dispatch.
pub(crate) fn worker_field(worker: Option<&(String, u64)>) -> String {
    worker.map_or_else(|| "-".to_owned(), |(id, epoch)| format!("{id}@{epoch}"))
}

/// Resolves after two consecutive polls with `L ≤ C`, both started after this call; never without polls.
async fn load_cleared(load: Option<watch::Receiver<LoadPoll>>) {
    let Some(mut load) = load else {
        return std::future::pending().await;
    };
    let start = load.borrow_and_update().seq;
    loop {
        if load.changed().await.is_err() {
            return std::future::pending().await;
        }
        let poll = *load.borrow_and_update();
        if poll.seq >= start + 2 && poll.clear_streak >= 2 {
            return;
        }
    }
}

/// The configured workers (m3 design note C3), each with the index of its `[[deployments]]` entry (M5 D5), in
/// entry order: an entry's `replica_list` for `sglang_http` (the single form is one replica on
/// `scheduler.devices[0]`; a remote-only entry of several has none); `native_cuda` (exactly one entry): its
/// `native.replica_list` (m32n design note D2: one worker on `scheduler.devices[0]`, or one per
/// `[[deployments.native.replicas]]` entry), without a base URL.
fn configured_replicas(config: &RuntimeConfig) -> Result<Vec<(usize, ReplicaConfig)>, String> {
    let devices = &config.scheduler.devices;
    let mut replicas = Vec::new();
    for (index, entry) in config.deployments.iter().enumerate() {
        let listed = match entry.kind {
            BackendKind::SglangHttp => entry.replica_list(devices),
            BackendKind::NativeCuda => entry
                .native
                .as_ref()
                .map(|n| n.replica_list(devices))
                .unwrap_or_default()
                .into_iter()
                .map(|r| ReplicaConfig {
                    device: r.device,
                    base_url: String::new(),
                    load_endpoint: None,
                })
                .collect(),
            BackendKind::Unconfigured | BackendKind::CpuReference => {
                return Err("cered serves only sglang_http and native_cuda backends".to_owned())
            }
        };
        replicas.extend(listed.into_iter().map(|r| (index, r)));
    }
    if replicas.is_empty() {
        return Err(
            "cered needs a backend on scheduler.devices (deployments.base_url or [[deployments.replicas]])"
                .to_owned(),
        );
    }
    Ok(replicas)
}

/// The backend in-flight cap (Q6) of every local worker of entry `index`, as config validation bounds it:
/// `expect.max_running_requests` for `sglang_http`, the profile's validated rows (`decode_batch_max`) for
/// `native_cuda`, plus `max_parked_sessions` with `preempt` (M2.11 D8).
fn worker_cap(config: &RuntimeConfig, index: usize) -> Result<u64, String> {
    let entry = &config.deployments[index];
    match entry.kind {
        BackendKind::NativeCuda => {
            let parked = entry
                .native
                .as_ref()
                .filter(|n| n.preempt)
                .map_or(0, |n| n.max_parked_sessions);
            entry
                .lock
                .as_ref()
                .map(|d| d.profile.decode_batch_max.saturating_add(parked))
                .ok_or_else(|| "cered needs the pinned deployment".to_owned())
        }
        _ => entry
            .expect
            .as_ref()
            .map(|e| e.max_running_requests)
            .ok_or_else(|| format!("cered needs deployments[{index}].expect")),
    }
}

/// The problem fields, comma separated.
fn fields(problems: &[Problem]) -> String {
    problems
        .iter()
        .map(|p| p.field.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(crate) fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}
