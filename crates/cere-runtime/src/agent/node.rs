//! The running node agent (m35 design note Q3-Q8, Q10a, Q12, Q14; harness "Agent (AGENT)"): startup, the shared
//! node state every task and handler works on, the admission contract (Q7a), the fences (term Q6, generation Q5,
//! incarnation Q4) and the mTLS listener.
//!
//! [`Agent::start`] locks the `state_dir` once per process, loads the PKI, opens the op journal, bumps and persists
//! the worker epoch (old-epoch operations become terminal `worker_restart` and are aborted on the backend), binds
//! `listen` and starts three tasks: the listener (one tonic `Router` with `WorkerService` and
//! `WorkerInferenceService`, `tls::Authz` before any state), the Register/Heartbeat link ([`crate::agent::link`])
//! and the backend identity probe ([`crate::agent::identity`]).
//!
//! Lock order: [`Shared::journal`] before the node state. A fence takes the journal lock first, so an operation
//! accepted under the journal lock either precedes the fence (and is ended by it) or sees the fenced state.
//!
//! Events go to stderr as `cere-worker event=<kind> key=value ...` and into a bounded in-memory ring
//! ([`AgentHandle::events`]); they carry ids, counters and reason codes only, never prompt or output text.
//!
//! Managed mode (m36 design note Q3-Q7, design-m36-agent.md §3): with `[backend.launch]` the agent owns the
//! backend's lifecycle. Start checks the privilege rule, reconciles a recorded group of a previous process and
//! starts the [`Supervisor`]; `LoadModel` launches the backend (`unloaded`/`failed` -> `loading` once
//! `backend_child.json` exists -> `ready` on the first confirmed identity of a live child), `DrainModel` and
//! `UnloadModel` quiesce it (`draining` -> `unloading` -> `unloaded`), a child exit, the startup deadline or a
//! served-model mismatch fail it. Every model transition takes the journal lock first; lifecycle ops are
//! cleanup-acknowledged at their terminal and never count as live ops. Fences end only work ops and never touch
//! the child.

use std::collections::VecDeque;
use std::fmt;
use std::fs::File;
use std::io;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use cere_proto::worker::v2 as pb;
use pb::worker_inference_service_server::WorkerInferenceServiceServer;
use pb::worker_service_server::WorkerServiceServer;
use tokio::sync::{mpsc, watch, Notify};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::{Channel, Endpoint, Server};
use tonic::{Code, Status};

use crate::agent::config::{AgentConfig, AgentLaunch};
use crate::agent::identity::{self, BackendIdentity};
use crate::agent::journal::{Accepted, OpError, OpIdentity, OpJournal, OpKey, OpLimits, OpView};
use crate::agent::link;
use crate::agent::ops::Ops;
use crate::agent::service::WorkerApi;
use crate::agent::state::AgentState;
use crate::agent::supervisor::{
    self, ChildExit, ChildRecord, LaunchSpec, ReconcileError, Reconciled, Supervisor,
};
use crate::digest::{hex, random_bytes};
use crate::fed::quota::PrincipalQuota;
use crate::fed::tls::{FedError, Pki};
use crate::fed::wire::{self, AdmissionState, Inventory, InventoryModel, ModelState};
use crate::journal::{ensure_private_dir, lock_dir};
use crate::sglang::{Rid, SglangHttp, Timeouts};

/// Most event lines kept in memory for [`AgentHandle::events`].
const EVENT_RING: usize = 4096;
/// Bound of [`AgentHandle::shutdown`]'s wait for each task.
const SHUTDOWN_WAIT: Duration = Duration::from_secs(5);

/// Unix time in milliseconds.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Why [`Agent::start`] failed; nothing was served or registered.
#[derive(Debug)]
pub enum AgentError {
    /// The state dir cannot be created or locked (another `cere-worker` holds it).
    StateDir {
        path: PathBuf,
        source: io::Error,
    },
    /// The op journal or the durable state cannot be opened or written.
    State {
        path: PathBuf,
        source: io::Error,
    },
    Pki(FedError),
    /// The coordinator endpoint or the listener's TLS cannot be configured.
    Tls(String),
    Bind {
        addr: SocketAddr,
        source: io::Error,
    },
    /// Managed mode: the privilege rule (`check_privileges`), the program check (`check_program`) or the supervisor
    /// thread (design-m36-agent.md §2.3).
    Launch(String),
    /// Managed mode: start reconciliation left processes of the recorded group; `cere-worker` exits 1.
    Reconcile(String),
    /// Managed mode: start reconciliation could not read, parse or remove `backend_child.json` or list `/proc`
    /// (or its task failed); `cere-worker` exits 2 like every other startup error.
    ReconcileIo(String),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::StateDir { path, source } => {
                write!(f, "state dir {}: {source}", path.display())
            }
            AgentError::State { path, source } => {
                write!(f, "agent state in {}: {source}", path.display())
            }
            AgentError::Pki(e) => write!(f, "pki: {e}"),
            AgentError::Tls(e) => write!(f, "tls: {e}"),
            AgentError::Bind { addr, source } => write!(f, "cannot bind {addr}: {source}"),
            AgentError::Launch(e) => write!(f, "backend launch: {e}"),
            AgentError::Reconcile(e) | AgentError::ReconcileIo(e) => {
                write!(f, "backend reconcile: {e}")
            }
        }
    }
}

impl std::error::Error for AgentError {}

/// The event sink: one stderr line per event, the latest [`EVENT_RING`] kept in memory.
pub struct EventSink {
    ring: Mutex<VecDeque<String>>,
}

impl EventSink {
    fn new() -> EventSink {
        EventSink {
            ring: Mutex::new(VecDeque::new()),
        }
    }

    /// Emits `cere-worker event=<kind> <fields>`; `fields` are `key=value` pairs without prompt or output text.
    pub fn emit(&self, kind: &str, fields: fmt::Arguments<'_>) {
        let line = format!("cere-worker event={kind} {fields}");
        eprintln!("{line}");
        let mut ring = lock(&self.ring);
        if ring.len() == EVENT_RING {
            ring.pop_front();
        }
        ring.push_back(line);
    }

    pub fn recent(&self) -> Vec<String> {
        lock(&self.ring).iter().cloned().collect()
    }
}

/// The exact `DescribeResources` bytes and their hash (Q10a).
#[derive(Debug, PartialEq, Eq)]
pub struct InventoryDoc {
    pub bytes: Vec<u8>,
    pub sha256: [u8; 32],
    /// `InventoryReply.inventory_generation`: +1 per content change.
    pub generation: u64,
}

#[derive(Debug, Default)]
struct Lease {
    /// The last accepted `(coordinator_term, lease_sequence)`.
    last: (u64, u64),
    until: Option<Instant>,
    expired_logged: bool,
}

#[derive(Debug)]
struct NodeState {
    durable: AgentState,
    /// The latest confirmed backend identity of this process.
    identity: Option<BackendIdentity>,
    inventory: Option<Arc<InventoryDoc>>,
    lease: Lease,
    /// `(term, generation)` from the last applied reply to a heartbeat sent after the latest admission fence.
    reconciled: Option<(u64, u64)>,
    draining: bool,
    poisoned: bool,
    /// +1 per admission fence (lease expiry, term, generation, incarnation): replies to heartbeats sent before the
    /// latest one never set `reconciled`.
    view_fence: u64,
    /// An accepted Register under the current term and epoch.
    registered: bool,
    /// Last heartbeat sequence sent in this epoch.
    sequence: u64,
    last_accepted: Option<Instant>,
    /// `attached` without `[backend.launch]`; the Q4 lifecycle state otherwise (m36 design note Q4).
    model: ModelState,
    /// A launch is in progress (§3.5): the reported state stays `unloaded`/`failed` until `backend_child.json`
    /// exists (Q3).
    launching: bool,
    /// The running group of this process (set once `spawn` returned).
    child: Option<ChildRecord>,
    /// `loading` fails `model_load_failed` at this instant (`startup_timeout_ms`).
    load_deadline: Option<Instant>,
    /// An accepted Unload waits for quiescence (or a child exit) to end in `unloaded`.
    unload_pending: bool,
}

/// SubmitWork or CancelWork: the ops fences end and the live-op count covers (design-m36-agent.md §3.1).
pub(crate) fn work_op(method: &str) -> bool {
    method == wire::METHOD_SUBMIT || method == wire::METHOD_CANCEL
}

impl From<&AgentLaunch> for LaunchSpec {
    fn from(launch: &AgentLaunch) -> LaunchSpec {
        LaunchSpec {
            argv: launch.argv.clone(),
            env: launch.env.clone(),
            stop_timeout: Duration::from_millis(launch.stop_timeout_ms),
            uid: launch.uid,
            gid: launch.gid,
        }
    }
}

impl NodeState {
    fn admission(&self, now: Instant) -> AdmissionState {
        if self.poisoned {
            return AdmissionState::Poisoned;
        }
        let term = self.durable.coordinator_term;
        let generation = self.durable.model_generation;
        let live = self.lease.until.is_some_and(|until| now < until) && self.lease.last.0 == term;
        if live && !self.draining && generation >= 1 && self.reconciled == Some((term, generation))
        {
            AdmissionState::Open
        } else {
            AdmissionState::Fenced
        }
    }
}

/// One heartbeat as sent: the reply is judged against it.
#[derive(Debug, Clone)]
pub(crate) struct Beat {
    pub epoch: u64,
    pub sequence: u64,
    pub term: u64,
    pub generation: u64,
    pub admission: AdmissionState,
    pub inventory_sha256: [u8; 32],
    pub sent: Instant,
    pub sent_unix_ms: u64,
    /// Team pools (C19): the agent serves only its owner now (switch or window).
    pub owner_only: bool,
    view_fence: u64,
}

/// The node state shared by the listener, the link, the probe and (track WORK) the operation runtime.
pub struct Shared {
    pub config: AgentConfig,
    /// `RegisterRequest.boot_id`: 128 random bits (hex) per process.
    pub boot_id: String,
    /// The loopback backend client for work and aborts.
    pub backend: SglangHttp,
    /// The op journal (Q12). Lock it before any other node state.
    pub journal: Mutex<OpJournal>,
    pub events: EventSink,
    /// +1 per fence that ends operations (term, generation, incarnation); operation tasks watch it.
    pub fence: watch::Sender<u64>,
    /// Per-principal pool of unary control calls (Q14).
    pub quota: PrincipalQuota,
    /// Team pools (net-teleport design note 4.4, C19): the owner, windows and local switch.
    pub owner_only: crate::pool::AgentOwnerOnly,
    /// Set once at shutdown: the listener, the link, the probe and the operation runtime stop, and open
    /// `StreamWork` streams end `UNAVAILABLE worker_restart` (H2).
    pub(crate) stop: watch::Sender<bool>,
    state: Mutex<NodeState>,
    /// Managed mode: the backend supervisor (`Some` until [`AgentHandle::shutdown`] takes it; design §3.1).
    supervisor: Mutex<Option<Arc<Supervisor>>>,
    /// Wakes the lifecycle task when a new `load_deadline` is set.
    lifecycle_wake: Notify,
    /// The `state_dir` lock, held as long as anything can write the journal.
    _lock: File,
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, NodeState> {
        lock(&self.state)
    }

    pub fn epoch(&self) -> u64 {
        self.state().durable.worker_epoch
    }

    pub fn generation(&self) -> u64 {
        self.state().durable.model_generation
    }

    /// The highest coordinator term accepted (0 before the first).
    pub fn term(&self) -> u64 {
        self.state().durable.coordinator_term
    }

    /// The confirmed backend identity of this process (`None` before the first confirmation).
    pub fn identity(&self) -> Option<BackendIdentity> {
        self.state().identity.clone()
    }

    /// The current inventory (`None` before the first confirmed backend).
    pub fn inventory(&self) -> Option<Arc<InventoryDoc>> {
        self.state().inventory.clone()
    }

    /// Q7a: `poisoned` | `open` (lease live, not draining, reconciled = current generation under the lease's
    /// term) | `fenced`.
    pub fn admission_state(&self) -> AdmissionState {
        self.state().admission(Instant::now())
    }

    /// Q6 checks of a coordinator call (after `Authz` and the quota): [`Shared::check_target`], then
    /// [`Shared::fences`]. Never call it holding [`Shared::journal`] (a term fence takes it).
    pub fn authorize(
        &self,
        authority: Option<&pb::Authority>,
        check_generation: bool,
    ) -> Result<(), Status> {
        let authority = self.check_target(authority)?;
        self.fences(authority, check_generation)
    }

    /// The identity checks of [`Shared::authorize`] that are no fences (m36 design note Q6): a missing
    /// authority or target is `INVALID_ARGUMENT malformed_request`, then `wrong_cluster`, `wrong_worker`,
    /// `wrong_coordinator`.
    pub fn check_target<'a>(
        &self,
        authority: Option<&'a pb::Authority>,
    ) -> Result<&'a pb::Authority, Status> {
        let malformed = || wire::status(Code::InvalidArgument, wire::MALFORMED_REQUEST);
        let refuse = |reason| Err(wire::status(Code::FailedPrecondition, reason));
        let authority = authority.ok_or_else(malformed)?;
        let target = authority.target.as_ref().ok_or_else(malformed)?;
        if target.cluster_id != self.config.cluster_id {
            return refuse(wire::WRONG_CLUSTER);
        }
        if target.worker_id != self.config.worker_id {
            return refuse(wire::WRONG_WORKER);
        }
        if authority.coordinator_id != self.config.coordinator.coordinator_id {
            return refuse(wire::WRONG_COORDINATOR);
        }
        Ok(authority)
    }

    /// The fences of a checked authority: `stale_epoch` (target epoch not the current one), a lower term
    /// `stale_term` (a higher one fences, then the call continues) and, with `check_generation`,
    /// `model_generation` not the current one `stale_generation`. Never call it holding [`Shared::journal`].
    pub fn fences(&self, authority: &pb::Authority, check_generation: bool) -> Result<(), Status> {
        let refuse = |reason| Err(wire::status(Code::FailedPrecondition, reason));
        let epoch = authority.target.as_ref().map_or(0, |t| t.worker_epoch);
        if epoch != self.epoch() {
            return refuse(wire::STALE_EPOCH);
        }
        let term = self.term();
        if authority.coordinator_term < term {
            return refuse(wire::STALE_TERM);
        }
        if authority.coordinator_term > term {
            self.adopt_term(authority.coordinator_term);
        }
        if check_generation && authority.model_generation != Some(self.generation()) {
            return refuse(wire::STALE_GENERATION);
        }
        Ok(())
    }

    /// `[backend.launch]` is configured: the agent owns the model lifecycle (m36 design note Q3).
    pub fn managed(&self) -> bool {
        self.config.backend.launch.is_some()
    }

    /// Poisons the agent (admission `poisoned` for the life of the process); one `poisoned` event.
    pub fn poison(&self, reason: &str) {
        let mut st = self.state();
        self.poison_locked(&mut st, reason);
    }

    fn poison_locked(&self, st: &mut NodeState, reason: &str) {
        if st.poisoned {
            return;
        }
        st.poisoned = true;
        self.rebuild_inventory(st);
        self.events
            .emit("poisoned", format_args!("reason={reason}"));
    }

    fn persist(&self, st: &mut NodeState) {
        if st.durable.store(&self.config.state_dir).is_err() {
            self.poison_locked(st, "state_write");
        }
    }

    /// Re-encodes the inventory; a content change bumps `inventory_generation`. Its `admission_state` is
    /// `poisoned` or `open` only: the live open/fenced state travels in each heartbeat, because a hash change
    /// sends the worker back to Joining (Q7a) and an open/fenced flip in the hash would never settle. Attached
    /// mode (m36 design note Q7): model state `attached`, no lock, built once an identity is confirmed. Managed
    /// mode (§3.9): always built; the pins and `instance_id` are present iff an identity is confirmed and the
    /// model is `ready`, `draining` or `unloading`; the lock is the approved one; a launch reports the prior state.
    fn rebuild_inventory(&self, st: &mut NodeState) {
        let managed = self.managed();
        let confirmed = match &st.identity {
            Some(id)
                if !managed
                    || matches!(
                        st.model,
                        ModelState::Ready | ModelState::Draining | ModelState::Unloading
                    ) =>
            {
                Some(id)
            }
            _ if !managed => return,
            _ => None,
        };
        let generation = st.durable.model_generation;
        let name = &self.config.backend.served_model_name;
        let instance_id = confirmed.map(|id| id.instance_id.clone());
        let inventory = Inventory {
            schema_version: wire::INVENTORY_SCHEMA_VERSION,
            worker_id: self.config.worker_id.clone(),
            served_model_name: name.clone(),
            tp_size: confirmed.map(|id| id.tp_size),
            kv_cache_dtype: confirmed.map(|id| id.kv_cache_dtype.clone()),
            max_running_requests: confirmed.map(|id| id.max_running_requests),
            instance_id: instance_id.clone(),
            model_generation: generation,
            admission_state: if st.poisoned {
                AdmissionState::Poisoned
            } else {
                AdmissionState::Open
            },
            gpus: Vec::new(),
            models: vec![InventoryModel {
                served_model_name: name.clone(),
                lock_sha256: self.config.backend.lock_sha256.clone(),
                state: st.model,
                instance_id,
                model_generation: generation,
            }],
            artifacts: Vec::new(),
        };
        let bytes = match inventory.to_json_bytes() {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("cere-worker: inventory not built: {e}");
                return;
            }
        };
        if st.inventory.as_ref().is_some_and(|doc| doc.bytes == bytes) {
            return;
        }
        let next = st.inventory.as_ref().map_or(1, |doc| doc.generation + 1);
        st.inventory = Some(Arc::new(InventoryDoc {
            sha256: Inventory::sha256(&bytes),
            bytes,
            generation: next,
        }));
    }

    /// Logs the lease expiry once and fences the admission view.
    fn check_expiry(&self, st: &mut NodeState, now: Instant) {
        if st.lease.expired_logged || st.lease.until.is_none_or(|until| now < until) {
            return;
        }
        st.lease.expired_logged = true;
        st.reconciled = None;
        st.view_fence += 1;
        self.events.emit(
            "lease",
            format_args!("state=expired sequence={}", st.lease.last.1),
        );
    }

    /// Accepts a higher coordinator term (Q6): persisted, admission fenced, a new Register needed; every
    /// nonterminal operation of an older term terminal `coordinator_restart` and aborted. Adopting the first term
    /// (none accepted before) is not a fence.
    fn adopt_term(&self, term: u64) {
        let mut journal = lock(&self.journal);
        let old = {
            let mut st = self.state();
            let old = st.durable.coordinator_term;
            if term <= old {
                return;
            }
            st.durable.coordinator_term = term;
            st.registered = false;
            st.reconciled = None;
            st.view_fence += 1;
            self.persist(&mut st);
            old
        };
        if old != 0 {
            self.events.emit(
                "fence",
                format_args!(
                    "reason={} term={term} previous_term={old}",
                    wire::COORDINATOR_RESTART
                ),
            );
        }
        let ended = self.end_ops(&mut journal, wire::COORDINATOR_RESTART, |op| {
            op.coordinator_term < term && work_op(op.method)
        });
        drop(journal);
        self.after_fence(wire::COORDINATOR_RESTART, &ended);
    }

    /// An in-process incarnation jump to `epoch` (Q4): persisted, sequence 0, a new Register; every nonterminal
    /// work op of an older epoch terminal `worker_restart` and aborted. Lifecycle ops and the child are untouched
    /// (design-m36-agent.md §3.8); [`Shared::restart_ops`] stays for [`Agent::start`].
    fn jump_epoch(&self, epoch: u64) {
        let mut journal = lock(&self.journal);
        {
            let mut st = self.state();
            if epoch <= st.durable.worker_epoch {
                return;
            }
            st.durable.worker_epoch = epoch;
            st.sequence = 0;
            st.registered = false;
            st.last_accepted = None;
            st.reconciled = None;
            st.view_fence += 1;
            self.persist(&mut st);
        }
        let ended = self.end_ops(&mut journal, wire::WORKER_RESTART, |op| {
            op.worker_epoch < epoch && work_op(op.method)
        });
        drop(journal);
        self.after_fence(wire::WORKER_RESTART, &ended);
    }

    /// `OpJournal::restart(epoch)`; on a journal failure the agent is poisoned and the doomed operations are
    /// still aborted.
    fn restart_ops(&self, journal: &mut OpJournal, epoch: u64) -> Vec<(OpView, bool)> {
        let doomed = nonterminal(journal, |op| op.worker_epoch < epoch);
        match journal.restart(epoch, now_ms()) {
            Ok(_) => doomed.into_iter().map(|op| (op, true)).collect(),
            Err(_) => {
                self.poison("journal_write");
                doomed.into_iter().map(|op| (op, false)).collect()
            }
        }
    }

    /// Makes every nonterminal operation matching `fenced` terminal `reason`; `(op, journaled)` per operation.
    fn end_ops(
        &self,
        journal: &mut OpJournal,
        reason: &str,
        fenced: impl Fn(&OpView) -> bool,
    ) -> Vec<(OpView, bool)> {
        let doomed = nonterminal(journal, fenced);
        let now = now_ms();
        let mut out = Vec::with_capacity(doomed.len());
        for op in doomed {
            let journaled = journal.mark_terminal(&op.key, reason, now).is_ok();
            if !journaled {
                self.poison("journal_write");
            }
            out.push((op, journaled));
        }
        out
    }

    /// Terminal events for the journaled ends, the fence counter, and a backend abort per ended SubmitWork rid.
    fn after_fence(&self, reason: &str, ended: &[(OpView, bool)]) {
        let mut aborts = Vec::new();
        for (op, journaled) in ended {
            if *journaled {
                self.events.emit(
                    "terminal",
                    format_args!(
                        "op={}/{} reason={reason}",
                        op.key.scope_id, op.key.operation_id
                    ),
                );
            }
            if op.method == wire::METHOD_SUBMIT {
                if let Ok(rid) = Rid::parse(&op.request_id) {
                    aborts.push(rid);
                }
            }
        }
        self.fence.send_modify(|n| *n += 1);
        for rid in aborts {
            let backend = self.backend.clone();
            tokio::spawn(async move {
                let _ = backend.abort(&rid).await;
            });
        }
    }

    /// A confirmed backend identity (Q5): the first one of the process builds the inventory; an `instance_id`
    /// other than the one behind the current generation bumps the durable generation, fences admission and ends
    /// every nonterminal work op of an older generation `model_generation_superseded` (aborted; a lifecycle op is
    /// never superseded, M36-01). A probe outage followed by the same `instance_id` changes nothing. Managed mode
    /// (§3.6): ignored unless [`confirmable`] and the recorded child is alive (a foreign listener is never
    /// confirmed); in `loading` the model becomes `ready` and pending Loads end `""`, under the one journal lock.
    pub(crate) fn confirm(&self, identity: BackendIdentity) {
        let mut journal = lock(&self.journal);
        let bumped = {
            let mut st = self.state();
            if self.managed()
                && (!confirmable(st.model, st.load_deadline, Instant::now())
                    || !st.child.is_some_and(|child| Supervisor::alive(&child)))
            {
                return;
            }
            if st.identity.as_ref() == Some(&identity) {
                return;
            }
            let same = st.durable.model_generation >= 1
                && st.durable.instance_id.as_deref() == Some(identity.instance_id.as_str());
            if !same {
                st.durable.model_generation += 1;
                st.durable.instance_id = Some(identity.instance_id.clone());
                st.reconciled = None;
                st.view_fence += 1;
                self.persist(&mut st);
            }
            st.identity = Some(identity);
            if st.model == ModelState::Loading {
                st.load_deadline = None;
                self.set_model(&mut st, ModelState::Ready);
                self.end_lifecycle(&mut journal, &mut st, wire::METHOD_LOAD, "");
            }
            self.rebuild_inventory(&mut st);
            (!same).then(|| {
                (
                    st.durable.model_generation,
                    st.durable.instance_id.clone().unwrap_or_default(),
                )
            })
        };
        let Some((generation, instance_id)) = bumped else {
            return;
        };
        self.events.emit(
            "generation",
            format_args!("generation={generation} instance_id={instance_id}"),
        );
        let ended = self.end_ops(&mut journal, wire::MODEL_GENERATION_SUPERSEDED, |op| {
            op.model_generation < generation && work_op(op.method)
        });
        drop(journal);
        self.after_fence(wire::MODEL_GENERATION_SUPERSEDED, &ended);
    }

    /// Per link iteration: journal upkeep (tombstones, compaction; a failed write poisons), the lazy lease
    /// expiry, and the self-lost rule (no accepted reply for `lost_after_ms` -> incarnation jump).
    pub(crate) fn tick(&self) {
        let now = Instant::now();
        let poisoned = {
            let mut journal = lock(&self.journal);
            journal.expire(now_ms());
            journal.poisoned()
        };
        if poisoned {
            self.poison("journal_write");
        }
        let lost_after = Duration::from_millis(self.config.limits.lost_after_ms);
        let lost = {
            let mut st = self.state();
            self.check_expiry(&mut st, now);
            st.last_accepted
                .is_some_and(|at| now.duration_since(at) >= lost_after)
                .then_some(st.durable.worker_epoch)
        };
        if let Some(epoch) = lost {
            self.jump_epoch(epoch + 1);
        }
    }

    /// The next heartbeat (sequence +1), or `None` while not registered.
    pub(crate) fn next_beat(&self) -> Option<Beat> {
        let owner_only = self.owner_only.active(now_ms());
        let now = Instant::now();
        let mut st = self.state();
        if !st.registered {
            return None;
        }
        let sha = st.inventory.as_ref()?.sha256;
        st.sequence += 1;
        Some(Beat {
            epoch: st.durable.worker_epoch,
            sequence: st.sequence,
            term: st.durable.coordinator_term,
            generation: st.durable.model_generation,
            admission: st.admission(now),
            inventory_sha256: sha,
            sent: now,
            sent_unix_ms: now_ms(),
            owner_only,
            view_fence: st.view_fence,
        })
    }

    /// The Register to send (epoch, inventory hash), or `None` before the first confirmed backend.
    pub(crate) fn register_target(&self) -> Option<(u64, [u8; 32])> {
        let st = self.state();
        let sha = st.inventory.as_ref()?.sha256;
        Some((st.durable.worker_epoch, sha))
    }

    /// A Register reply for `epoch` (Q4, Q6).
    pub(crate) fn on_register(&self, epoch: u64, reply: &pb::RegisterReply) {
        {
            let st = self.state();
            if epoch != st.durable.worker_epoch
                || reply.coordinator_term < st.durable.coordinator_term
            {
                // A jump raced it, or a lower term: refused.
                return;
            }
        }
        if reply.accepted {
            self.events.emit(
                "register",
                format_args!(
                    "term={} accepted=true epoch={epoch}",
                    reply.coordinator_term
                ),
            );
        } else {
            self.events.emit(
                "register",
                format_args!(
                    "term={} accepted=false epoch={epoch} reason={}",
                    reply.coordinator_term, reply.reason_code
                ),
            );
        }
        self.adopt_term(reply.coordinator_term);
        if reply.accepted {
            let mut st = self.state();
            if st.durable.worker_epoch == epoch {
                st.registered = true;
                st.last_accepted = Some(Instant::now());
            }
            return;
        }
        match reply.reason_code.as_str() {
            wire::EPOCH_LOST => self.jump_epoch(epoch + 1),
            wire::STALE_EPOCH => self.jump_epoch(reply.journaled_worker_epoch.max(epoch) + 1),
            _ => {}
        }
    }

    /// A heartbeat reply (Q4, Q6, Q7, Q7a, Q13).
    pub(crate) fn on_heartbeat(&self, beat: &Beat, reply: &pb::HeartbeatReply) {
        let now = Instant::now();
        let term = reply.coordinator_term;
        {
            let st = self.state();
            if beat.epoch != st.durable.worker_epoch || term < st.durable.coordinator_term {
                return;
            }
        }
        if term > beat.term {
            // Q6 fence first; the grant of an older term view is not applied, a Register follows.
            self.adopt_term(term);
            self.state().registered = false;
            return;
        }
        if !reply.accepted {
            match reply.reason_code.as_str() {
                wire::EPOCH_LOST => self.jump_epoch(beat.epoch + 1),
                wire::REPLAYED_HEARTBEAT => {}
                // unknown_worker, stale_epoch (the Register answers with the journaled epoch), stale_term, ...
                _ => self.state().registered = false,
            }
            return;
        }
        let mut st = self.state();
        st.last_accepted = Some(now);
        self.check_expiry(&mut st, now);
        let pair = (term, reply.lease_sequence);
        if pair <= st.lease.last {
            return;
        }
        st.lease.last = pair;
        st.lease.until = Some(beat.sent + Duration::from_millis(reply.admission_lease_ms));
        st.lease.expired_logged = false;
        self.events.emit(
            "lease",
            format_args!(
                "state=granted sequence={} until_ms={}",
                reply.lease_sequence,
                beat.sent_unix_ms.saturating_add(reply.admission_lease_ms)
            ),
        );
        st.draining = reply.desired_state == wire::DESIRED_DRAINING;
        if beat.view_fence == st.view_fence {
            st.reconciled = reply.reconciled_generation.map(|g| (term, g));
        }
    }
}

/// What an accepted new lifecycle key does (design-m36-agent.md §3.4).
enum Transition {
    /// Joins the pending op of the same method; it ends with it.
    Join,
    /// `launching = true`, then the launch task (§3.5).
    Launch,
    /// Terminal `""` at once (Load in `ready`; Unload in `unloaded`/`failed`, `failed` -> `unloaded`).
    EndNow,
    /// `ready` -> `draining` (`unload`: an Unload waits for it), then the quiescence check.
    Drain { unload: bool },
    /// Already `draining`: an Unload sets `unload_pending`; the quiescence check, never a restarted drain.
    Quiesce { unload: bool },
    /// Load in `draining`: back to `ready`, pending Drains (and Unloads) end `cancel_requested`.
    Undrain,
}

/// What a returned stop of the recorded group leaves (design-m36-agent.md §3.6, §3.7).
#[derive(Debug, PartialEq, Eq)]
struct AfterStop {
    model: ModelState,
    /// The terminal reason of the pending Unloads (`None`: none can be pending).
    unload: Option<&'static str>,
    /// The terminal reason of the pending Drains.
    drain: &'static str,
    /// The stop did not confirm the group gone: the child stays recorded.
    keep_child: bool,
}

/// A confirmed stop drops the child: a pending Unload ends `""` in `unloaded`, else the model is `failed`; pending
/// Drains end `backend_exited`. A stop error (survivors, an undeletable record, the supervisor gone) is no confirmed
/// stop: the child stays recorded, the model is `failed` and pending Unloads and Drains end `poisoned`.
fn after_stop(stopped: &Result<(), supervisor::StopError>, unload_pending: bool) -> AfterStop {
    match (stopped, unload_pending) {
        (Err(_), _) => AfterStop {
            model: ModelState::Failed,
            unload: Some(wire::POISONED),
            drain: wire::POISONED,
            keep_child: true,
        },
        (Ok(()), true) => AfterStop {
            model: ModelState::Unloaded,
            unload: Some(""),
            drain: wire::BACKEND_EXITED,
            keep_child: false,
        },
        (Ok(()), false) => AfterStop {
            model: ModelState::Failed,
            unload: None,
            drain: wire::BACKEND_EXITED,
            keep_child: false,
        },
    }
}

/// The node state still matches the identity checked by [`Shared::fences`] (a fence ran in between).
fn recheck(st: &NodeState, identity: &OpIdentity) -> Result<(), Status> {
    let refuse = |reason| Err(wire::status(Code::FailedPrecondition, reason));
    if identity.worker_epoch != st.durable.worker_epoch {
        return refuse(wire::STALE_EPOCH);
    }
    if identity.coordinator_term != st.durable.coordinator_term {
        return refuse(wire::STALE_TERM);
    }
    if identity.model_generation != st.durable.model_generation {
        return refuse(wire::STALE_GENERATION);
    }
    Ok(())
}

/// The `<ip>:<port>` of `[backend] base_url` (`http://<loopback literal>:<port>`, validated at load).
fn backend_addr(base_url: &str) -> Option<SocketAddr> {
    base_url
        .strip_prefix("http://")?
        .split('/')
        .next()?
        .parse()
        .ok()
}

/// The managed model lifecycle (m36 design note Q3-Q7, design-m36-agent.md §3.3-§3.7). Lock order journal ->
/// state; the supervisor is called outside both.
impl Shared {
    fn supervisor(&self) -> Option<Arc<Supervisor>> {
        lock(&self.supervisor).clone()
    }

    /// Stops the recorded group; `Ok` when nothing runs (or the supervisor is gone at shutdown).
    async fn stop_child(&self) -> Result<(), supervisor::StopError> {
        match self.supervisor() {
            Some(sup) => sup.stop().await,
            None => Ok(()),
        }
    }

    /// The model becomes `model` (one `model` event per change) and the inventory is rebuilt.
    fn set_model(&self, st: &mut NodeState, model: ModelState) {
        if st.model != model {
            st.model = model;
            self.events
                .emit("model", format_args!("state={}", model.as_str()));
        }
        self.rebuild_inventory(st);
    }

    /// Every pending op of `method` terminal `reason` and cleanup-acknowledged (lifecycle ops are acknowledged at
    /// their terminal, Q4); a journal failure poisons the agent.
    fn end_lifecycle(
        &self,
        journal: &mut OpJournal,
        st: &mut NodeState,
        method: &str,
        reason: &str,
    ) {
        let now = now_ms();
        for op in nonterminal(journal, |op| op.method == method) {
            let journaled = journal
                .mark_terminal(&op.key, reason, now)
                .and_then(|()| journal.mark_cleanup(&op.key, now));
            match journaled {
                Ok(()) => self.events.emit(
                    "terminal",
                    format_args!(
                        "op={}/{} reason={reason}",
                        op.key.scope_id, op.key.operation_id
                    ),
                ),
                Err(_) => self.poison_locked(st, "journal_write"),
            }
        }
    }

    /// Every terminal SubmitWork op not yet cleanup-acknowledged is acknowledged: the backend that held it is gone
    /// (Q4; no `/get_load` poll of a stopped backend).
    fn clean_work_terminals(&self, journal: &mut OpJournal, st: &mut NodeState) {
        let (all, _) = journal.list(None, usize::MAX);
        let now = now_ms();
        for op in all.into_iter().filter(|op| {
            op.method == wire::METHOD_SUBMIT && op.terminal && !op.cleanup_acknowledged
        }) {
            match journal.mark_cleanup(&op.key, now) {
                Ok(()) => self.events.emit(
                    "cleanup",
                    format_args!("op={}/{}", op.key.scope_id, op.key.operation_id),
                ),
                Err(_) => self.poison_locked(st, "journal_write"),
            }
        }
    }

    /// Start: terminal lifecycle ops of an earlier process are cleanup-acknowledged (design §3.2).
    fn ack_lifecycle_terminals(&self, journal: &mut OpJournal) {
        let (all, _) = journal.list(None, usize::MAX);
        let now = now_ms();
        for op in all
            .into_iter()
            .filter(|op| !work_op(op.method) && op.terminal && !op.cleanup_acknowledged)
        {
            if journal.mark_cleanup(&op.key, now).is_err() {
                self.poison("journal_write");
            }
        }
    }

    /// The status of a refused journal accept; a write failure poisons the agent.
    pub(crate) fn op_error(&self, error: OpError) -> Status {
        match error {
            OpError::Conflict => wire::status(Code::AlreadyExists, wire::OPERATION_CONFLICT),
            OpError::Full => wire::status(Code::ResourceExhausted, wire::JOURNAL_FULL),
            OpError::Poisoned => wire::status(Code::Unavailable, wire::POISONED),
            OpError::Unknown => wire::status(Code::NotFound, wire::UNKNOWN_OPERATION),
            OpError::Io(_) => {
                self.poison("journal_write");
                wire::status(Code::Unavailable, wire::POISONED)
            }
        }
    }

    /// A new SubmitWork key, under the journal lock (§3.8): `draining` -> `UNAVAILABLE model_draining`; no
    /// serving model (`ready`, or `attached`, and no launch running) -> `UNAVAILABLE backend_unconfirmed`.
    pub(crate) fn work_refusal(&self) -> Result<(), Status> {
        let st = self.state();
        match st.model {
            ModelState::Draining => Err(wire::status(Code::Unavailable, wire::MODEL_DRAINING)),
            ModelState::Ready | ModelState::Attached if !st.launching => Ok(()),
            _ => Err(wire::status(Code::Unavailable, wire::BACKEND_UNCONFIRMED)),
        }
    }

    /// The identity probe runs (§3.6): always in attached mode; in managed mode while the model is `loading`,
    /// `ready` or `draining`.
    pub(crate) fn probe_wanted(&self) -> bool {
        !self.managed()
            || matches!(
                self.state().model,
                ModelState::Loading | ModelState::Ready | ModelState::Draining
            )
    }

    /// The probe saw another served model: a `loading` model fails `model_load_failed` at once (§3.6).
    pub(crate) async fn name_mismatch(&self) {
        let child = {
            let st = self.state();
            if self.managed() && st.model == ModelState::Loading {
                st.child
            } else {
                None
            }
        };
        if let Some(child) = child {
            self.fail(child, wire::MODEL_LOAD_FAILED).await;
        }
    }

    fn join_lifecycle(
        &self,
        journal: &mut OpJournal,
        key: &OpKey,
        identity: OpIdentity,
    ) -> Result<Accepted, Status> {
        let accepted = journal
            .accept(
                key.clone(),
                identity,
                &key.operation_id,
                1,
                wire::WORK_CAP_MS,
                now_ms(),
            )
            .map_err(|e| self.op_error(e))?;
        self.events.emit(
            "accept",
            format_args!(
                "op={}/{} fresh={}",
                key.scope_id, key.operation_id, accepted.fresh
            ),
        );
        Ok(accepted)
    }

    /// Q6 steps 5-7 of a `LoadModel`/`DrainModel`/`UnloadModel` whose shape, admission, mode and digest
    /// [`WorkerApi`] checked. A journaled key is answered by [`OpJournal::accept`] (its receipt, or
    /// `operation_conflict`) without fences or state checks. A new key: [`Shared::fences`] (generation 0 before the
    /// first confirmed backend), `lock_not_approved`, `unsupported_placement`, a stale `instance_id`
    /// (`stale_generation`), then under the journal lock `UNAVAILABLE poisoned` (a stop poisoned the agent after the
    /// admission check) and the §3.4 table; `model_busy` is refused before the journal.
    pub(crate) fn lifecycle(
        self: &Arc<Self>,
        req: &pb::ModelLifecycleRequest,
        key: OpKey,
        identity: OpIdentity,
    ) -> Result<Accepted, Status> {
        let method = identity.method;
        {
            let mut journal = lock(&self.journal);
            if journal.get(&key).is_some() {
                return self.join_lifecycle(&mut journal, &key, identity);
            }
        }
        let refuse = |reason| Err(wire::status(Code::FailedPrecondition, reason));
        let authority = self.check_target(req.authority.as_ref())?;
        self.fences(authority, true)?;
        let approved = self.config.backend.lock_bytes();
        if approved.as_ref().map(<[u8; 32]>::as_slice)
            != Some(req.deployment_lock_sha256.as_slice())
        {
            return refuse(wire::LOCK_NOT_APPROVED);
        }
        if !req.device_ids.is_empty() {
            return refuse(wire::UNSUPPORTED_PLACEMENT);
        }
        if !req.instance_id.is_empty()
            && (method == wire::METHOD_LOAD
                || self.identity().map(|id| id.instance_id).as_deref()
                    != Some(req.instance_id.as_str()))
        {
            return refuse(wire::STALE_GENERATION);
        }

        let mut journal = lock(&self.journal);
        if journal.get(&key).is_some() {
            return self.join_lifecycle(&mut journal, &key, identity);
        }
        let transition = {
            let st = self.state();
            recheck(&st, &identity)?;
            if st.poisoned {
                return Err(wire::status(Code::Unavailable, wire::POISONED));
            }
            let loading = st.launching || st.model == ModelState::Loading;
            let busy = || Err(wire::status(Code::FailedPrecondition, wire::MODEL_BUSY));
            match (method, st.model) {
                (wire::METHOD_LOAD, _) if loading => Transition::Join,
                (wire::METHOD_LOAD, ModelState::Unloaded | ModelState::Failed) => {
                    Transition::Launch
                }
                (wire::METHOD_LOAD, ModelState::Ready) => Transition::EndNow,
                (wire::METHOD_LOAD, ModelState::Draining) => Transition::Undrain,
                (_, _) if loading => return busy(),
                (wire::METHOD_DRAIN, ModelState::Ready) => Transition::Drain { unload: false },
                (wire::METHOD_DRAIN, ModelState::Draining) => Transition::Quiesce { unload: false },
                (wire::METHOD_UNLOAD, ModelState::Unloaded | ModelState::Failed) => {
                    Transition::EndNow
                }
                (wire::METHOD_UNLOAD, ModelState::Ready) => Transition::Drain { unload: true },
                (wire::METHOD_UNLOAD, ModelState::Draining) => Transition::Quiesce { unload: true },
                (wire::METHOD_UNLOAD, ModelState::Unloading) => Transition::Join,
                _ => return busy(),
            }
        };
        let accepted = self.join_lifecycle(&mut journal, &key, identity)?;
        let mut st = self.state();
        let mut launch = false;
        let mut quiesce = false;
        match transition {
            Transition::Join => {}
            Transition::Launch => {
                st.launching = true;
                launch = true;
            }
            Transition::EndNow => {
                if st.model == ModelState::Failed {
                    self.set_model(&mut st, ModelState::Unloaded);
                }
                self.end_lifecycle(&mut journal, &mut st, method, "");
            }
            Transition::Drain { unload } => {
                st.unload_pending |= unload;
                self.set_model(&mut st, ModelState::Draining);
                quiesce = true;
            }
            Transition::Quiesce { unload } => {
                st.unload_pending |= unload;
                quiesce = true;
            }
            Transition::Undrain => {
                st.unload_pending = false;
                self.set_model(&mut st, ModelState::Ready);
                self.end_lifecycle(
                    &mut journal,
                    &mut st,
                    wire::METHOD_DRAIN,
                    wire::CANCEL_REQUESTED,
                );
                self.end_lifecycle(
                    &mut journal,
                    &mut st,
                    wire::METHOD_UNLOAD,
                    wire::CANCEL_REQUESTED,
                );
                self.end_lifecycle(&mut journal, &mut st, wire::METHOD_LOAD, "");
            }
        }
        drop(st);
        drop(journal);
        if launch {
            tokio::spawn(Arc::clone(self).launch());
        }
        if quiesce {
            self.quiesce();
        }
        Ok(accepted)
    }

    /// §3.5, one task per launch: a listener already on the `base_url` port -> `failed`, Loads end
    /// `backend_port_in_use`, no spawn; a failed spawn -> stop (a half-started group), `failed`, Loads end
    /// `model_load_failed`; else `loading` with the child recorded and the startup deadline armed.
    async fn launch(self: Arc<Self>) {
        let rpc = Duration::from_millis(self.config.limits.rpc_timeout_ms);
        if let Some(addr) = backend_addr(&self.config.backend.base_url) {
            let probe = tokio::time::timeout(rpc, tokio::net::TcpStream::connect(addr)).await;
            if matches!(probe, Ok(Ok(_))) {
                self.launch_failed(wire::BACKEND_PORT_IN_USE);
                return;
            }
        }
        let spawned = match self.supervisor() {
            Some(sup) => sup.spawn().await.map_err(|e| e.to_string()),
            None => Err("supervisor stopped".to_owned()),
        };
        let child = match spawned {
            Ok(child) => child,
            Err(e) => {
                eprintln!("cere-worker: backend spawn failed: {e}");
                if let Err(e) = self.stop_child().await {
                    eprintln!("cere-worker: backend stop failed: {e}");
                    self.poison("backend_stop");
                }
                self.launch_failed(wire::MODEL_LOAD_FAILED);
                return;
            }
        };
        let startup = self
            .config
            .backend
            .launch
            .as_ref()
            .map_or(0, |launch| launch.startup_timeout_ms);
        {
            let _journal = lock(&self.journal);
            let mut st = self.state();
            st.child = Some(child);
            st.launching = false;
            st.load_deadline = Some(Instant::now() + Duration::from_millis(startup));
            self.set_model(&mut st, ModelState::Loading);
        }
        self.events.emit(
            "backend",
            format_args!("state=spawned pid={} pgid={}", child.pid, child.pgid),
        );
        self.lifecycle_wake.notify_one();
        // An exit before `child` was set reached the lifecycle task as a foreign record; the unreaped leader shows
        // it here.
        if !Supervisor::alive(&child) {
            self.fail(child, wire::MODEL_LOAD_FAILED).await;
        }
    }

    fn launch_failed(&self, reason: &str) {
        let mut journal = lock(&self.journal);
        let mut st = self.state();
        st.launching = false;
        self.set_model(&mut st, ModelState::Failed);
        self.end_lifecycle(&mut journal, &mut st, wire::METHOD_LOAD, reason);
    }

    /// §3.6 fail: the startup deadline is cleared (a confirmation during the stop is ignored), the group is stopped
    /// (outside the lock), then under the journal lock [`Shared::settle_stop`] with pending Loads ending `reason`.
    /// Nothing happens unless `child` is still current.
    async fn fail(&self, child: ChildRecord, reason: &str) {
        {
            let mut st = self.state();
            if st.child != Some(child) {
                return;
            }
            st.load_deadline = None;
        }
        self.events.emit(
            "backend",
            format_args!("state=failing pgid={} reason={reason}", child.pgid),
        );
        let stopped = self.stop_child().await;
        let mut journal = lock(&self.journal);
        let mut st = self.state();
        if st.child != Some(child) {
            return;
        }
        self.settle_stop(&mut journal, &mut st, stopped, Some(reason));
    }

    /// A returned stop of the recorded group, under the journal lock ([`after_stop`]): the identity and the startup
    /// deadline are dropped, the model and the pending Unloads, Loads (`load`) and Drains end as decided. A confirmed
    /// stop drops the child and cleanup-acknowledges the terminal SubmitWork ops; a stop that cannot confirm the
    /// group gone poisons the agent (`backend_stop`) and keeps the child recorded.
    fn settle_stop(
        &self,
        journal: &mut OpJournal,
        st: &mut NodeState,
        stopped: Result<(), supervisor::StopError>,
        load: Option<&str>,
    ) {
        let after = after_stop(&stopped, st.unload_pending);
        if let Err(e) = stopped {
            eprintln!("cere-worker: backend stop failed: {e}");
            self.poison_locked(st, "backend_stop");
        }
        if !after.keep_child {
            st.child = None;
        }
        st.identity = None;
        st.load_deadline = None;
        st.unload_pending = false;
        self.set_model(st, after.model);
        if let Some(reason) = after.unload {
            self.end_lifecycle(journal, st, wire::METHOD_UNLOAD, reason);
        }
        if let Some(reason) = load {
            self.end_lifecycle(journal, st, wire::METHOD_LOAD, reason);
        }
        self.end_lifecycle(journal, st, wire::METHOD_DRAIN, after.drain);
        if !after.keep_child {
            self.clean_work_terminals(journal, st);
        }
    }

    /// §3.7, after every Drain/Unload accept and every SubmitWork cleanup: in `draining` with no live op the pending
    /// Drains end `""`; with an Unload pending the model goes `unloading`, the group is stopped and the model ends
    /// `unloaded` with the pending Unloads `""`. A recorded leader that is already dead or exiting does not quiesce:
    /// its ops ended because it died, and its exit fails the model (§3.6, pending Drains `backend_exited`).
    pub(crate) fn quiesce(self: &Arc<Self>) {
        {
            let mut journal = lock(&self.journal);
            let mut st = self.state();
            if st.model != ModelState::Draining || journal.live_ops() != 0 {
                return;
            }
            if st.child.is_some_and(|child| !Supervisor::alive(&child)) {
                return;
            }
            self.end_lifecycle(&mut journal, &mut st, wire::METHOD_DRAIN, "");
            if !st.unload_pending {
                return;
            }
            self.set_model(&mut st, ModelState::Unloading);
        }
        let shared = Arc::clone(self);
        tokio::spawn(async move { shared.finish_unload().await });
    }

    async fn finish_unload(&self) {
        let stopped = self.stop_child().await;
        let mut journal = lock(&self.journal);
        let mut st = self.state();
        if st.model != ModelState::Unloading {
            return;
        }
        self.settle_stop(&mut journal, &mut st, stopped, None);
    }

    /// A leader exit (§3.6): of the current child in `loading` -> fail `model_load_failed`, in `ready`/`draining`
    /// -> fail `backend_exited`; any other record or state is ignored (a stop does not report exits).
    async fn on_child_exit(&self, exit: ChildExit) {
        let reason = {
            let st = self.state();
            if st.child != Some(exit.record) {
                return;
            }
            match st.model {
                ModelState::Loading => wire::MODEL_LOAD_FAILED,
                ModelState::Ready | ModelState::Draining => wire::BACKEND_EXITED,
                _ => return,
            }
        };
        let how = match (exit.code, exit.signal) {
            (Some(code), _) => format!("code={code}"),
            (None, Some(signal)) => format!("signal={signal}"),
            (None, None) => "code=unknown".to_owned(),
        };
        self.events.emit(
            "backend",
            format_args!("state=exited pgid={} {how}", exit.record.pgid),
        );
        self.fail(exit.record, reason).await;
    }

    /// The startup deadline passed: a `loading` model fails `model_load_failed`.
    async fn on_deadline(&self) {
        let child = {
            let mut st = self.state();
            if st.load_deadline.is_none_or(|at| Instant::now() < at) {
                return;
            }
            st.load_deadline = None;
            if st.model == ModelState::Loading {
                st.child
            } else {
                None
            }
        };
        if let Some(child) = child {
            self.fail(child, wire::MODEL_LOAD_FAILED).await;
        }
    }
}

/// The lifecycle task (§3.6): child exits from the supervisor, the startup deadline, shutdown.
async fn supervise(
    shared: Arc<Shared>,
    mut exits: mpsc::UnboundedReceiver<ChildExit>,
    stop: watch::Receiver<bool>,
) {
    let stop = stopped(stop);
    tokio::pin!(stop);
    loop {
        let deadline = shared.state().load_deadline;
        let timer = async move {
            match deadline {
                Some(at) => tokio::time::sleep_until(at.into()).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            () = &mut stop => return,
            exit = exits.recv() => match exit {
                Some(exit) => shared.on_child_exit(exit).await,
                None => return,
            },
            () = timer => shared.on_deadline().await,
            () = shared.lifecycle_wake.notified() => {}
        }
    }
}

fn nonterminal(journal: &OpJournal, fenced: impl Fn(&OpView) -> bool) -> Vec<OpView> {
    let (all, _) = journal.list(None, usize::MAX);
    all.into_iter()
        .filter(|op| !op.terminal && fenced(op))
        .collect()
}

/// Resolves once `stop` is set (or its sender is gone).
pub(crate) async fn stopped(mut stop: watch::Receiver<bool>) {
    let _ = stop.wait_for(|stop| *stop).await;
}

/// The node agent.
pub struct Agent;

impl Agent {
    /// Starts the agent: state-dir lock, PKI, op journal, epoch +1 (persisted before anything is served or
    /// registered; older-epoch operations terminal `worker_restart` and aborted), in managed mode the privilege
    /// rule, start reconciliation of a recorded group and the supervisor (design-m36-agent.md §3.2), listener,
    /// link, probe and the lifecycle task.
    pub async fn start(config: AgentConfig) -> Result<AgentHandle, AgentError> {
        let dir = config.state_dir.clone();
        let state_err = |source| AgentError::State {
            path: dir.clone(),
            source,
        };
        let lock_file = ensure_private_dir(&dir)
            .and_then(|()| lock_dir(&dir, "cere-worker"))
            .map_err(|source| AgentError::StateDir {
                path: dir.clone(),
                source,
            })?;
        let pki = Pki::load(
            &config.ca_file,
            &config.cert_file,
            &config.key_file,
            &config.principal,
        )
        .map_err(AgentError::Pki)?;
        let limits = OpLimits {
            max_operations: config.journal.max_operations,
            retry_horizon_ms: config.journal.retry_horizon_ms,
            rpc_timeout_ms: config.limits.rpc_timeout_ms,
        };
        let (journal, _) = OpJournal::open(&dir, limits, now_ms()).map_err(state_err)?;
        let mut durable = AgentState::load(&dir).map_err(state_err)?;
        durable.worker_epoch += 1;
        durable.store(&dir).map_err(state_err)?;
        let mut managed = None;
        if let Some(launch) = &config.backend.launch {
            // SAFETY: geteuid has no preconditions and cannot fail.
            let euid = unsafe { libc::geteuid() };
            launch
                .check_privileges(euid)
                .and_then(|()| launch.check_program())
                .map_err(|e| AgentError::Launch(e.to_string()))?;
            let stop_timeout = Duration::from_millis(launch.stop_timeout_ms);
            let reconcile_dir = dir.clone();
            let reconciled = tokio::task::spawn_blocking(move || {
                supervisor::reconcile_at_start(&reconcile_dir, stop_timeout)
            })
            .await
            .map_err(|e| AgentError::ReconcileIo(e.to_string()))?
            .map_err(|e| match e {
                ReconcileError::Survivors { .. } => AgentError::Reconcile(e.to_string()),
                ReconcileError::Io(_) => AgentError::ReconcileIo(e.to_string()),
            })?;
            if let Reconciled::Killed { processes } = reconciled {
                eprintln!("cere-worker event=backend state=reconciled killed={processes}");
            }
            let (sup, exits) = Supervisor::start(dir.clone(), LaunchSpec::from(launch))
                .map_err(|e| AgentError::Launch(format!("backend supervisor: {e}")))?;
            managed = Some((Arc::new(sup), exits));
        }
        let channel = coordinator_channel(&config, &pki)?;
        let mut server = Server::builder()
            .tls_config(pki.server_tls())
            .map_err(|e| AgentError::Tls(e.to_string()))?;
        let listener = tokio::net::TcpListener::bind(config.listen)
            .await
            .map_err(|source| AgentError::Bind {
                addr: config.listen,
                source,
            })?;
        let local_addr = listener.local_addr().map_err(|source| AgentError::Bind {
            addr: config.listen,
            source,
        })?;
        let backend = SglangHttp::new(
            &config.backend.base_url,
            Timeouts {
                connect: Duration::from_secs(2),
                control: Duration::from_millis(config.limits.rpc_timeout_ms),
                // Covers prefill of a long prompt before the first streamed byte (as `cered`).
                stream_idle: Duration::from_secs(300),
            },
        )
        .map_err(|e| AgentError::Tls(format!("backend client: {e}")))?;
        let probe = identity::probe_client(
            &config.backend.base_url,
            Duration::from_millis(config.limits.heartbeat_interval_ms),
        );
        let boot_id = hex(&random_bytes::<16>().map_err(state_err)?);
        let epoch = durable.worker_epoch;
        let model = if managed.is_some() {
            ModelState::Unloaded
        } else {
            ModelState::Attached
        };
        let (supervisor, exits) = match managed {
            Some((sup, exits)) => (Some(sup), Some(exits)),
            None => (None, None),
        };
        let shared = Arc::new(Shared {
            boot_id,
            backend,
            journal: Mutex::new(journal),
            events: EventSink::new(),
            fence: watch::Sender::new(0),
            quota: PrincipalQuota::new(config.limits.control_quota),
            owner_only: crate::pool::AgentOwnerOnly::new(
                config.owner.clone(),
                config.owner_only_windows(),
                &config.state_dir,
            ),
            stop: watch::Sender::new(false),
            state: Mutex::new(NodeState {
                durable,
                identity: None,
                inventory: None,
                lease: Lease::default(),
                reconciled: None,
                draining: false,
                poisoned: false,
                view_fence: 0,
                registered: false,
                sequence: 0,
                last_accepted: None,
                model,
                launching: false,
                child: None,
                load_deadline: None,
                unload_pending: false,
            }),
            supervisor: Mutex::new(supervisor),
            lifecycle_wake: Notify::new(),
            _lock: lock_file,
            config,
        });
        let ended = {
            let mut journal = lock(&shared.journal);
            let ended = shared.restart_ops(&mut journal, epoch);
            shared.ack_lifecycle_terminals(&mut journal);
            let mut st = shared.state();
            // Managed: start reconciliation stopped any recorded group (a confirmed stop), so every terminal
            // SubmitWork op is acknowledged now instead of polling a dead backend's `/get_load` (Q4).
            if shared.managed() {
                shared.clean_work_terminals(&mut journal, &mut st);
            }
            shared.rebuild_inventory(&mut st);
            ended
        };
        shared.after_fence(wire::WORKER_RESTART, &ended);

        let ops = Ops::start(Arc::clone(&shared));
        let api = WorkerApi::new(Arc::clone(&shared), Arc::clone(&ops));
        let router = server
            .add_service(
                WorkerServiceServer::new(api.clone())
                    .max_decoding_message_size(wire::CONTROL_MESSAGE_MAX),
            )
            .add_service(
                WorkerInferenceServiceServer::new(api)
                    .max_decoding_message_size(wire::INFERENCE_MESSAGE_MAX),
            );
        let serve = router.serve_with_incoming_shutdown(
            TcpListenerStream::new(listener),
            stopped(shared.stop.subscribe()),
        );
        let ops_stop = shared.stop.subscribe();
        let mut tasks = vec![
            tokio::spawn(async move {
                if let Err(e) = serve.await {
                    eprintln!("cere-worker: listener error: {e}");
                }
            }),
            tokio::spawn(link::run(
                Arc::clone(&shared),
                channel,
                shared.stop.subscribe(),
            )),
            tokio::spawn(identity::run(
                Arc::clone(&shared),
                probe,
                shared.stop.subscribe(),
            )),
            tokio::spawn(async move {
                stopped(ops_stop).await;
                ops.join().await;
            }),
        ];
        if let Some(exits) = exits {
            tasks.push(tokio::spawn(supervise(
                Arc::clone(&shared),
                exits,
                shared.stop.subscribe(),
            )));
        }
        Ok(AgentHandle {
            shared,
            local_addr,
            tasks,
        })
    }
}

/// The channel to `[coordinator] endpoint` (mTLS, the coordinator principal as the expected peer name, every
/// call bounded by `rpc_timeout_ms`, HTTP/2 keepalive at the heartbeat interval).
fn coordinator_channel(config: &AgentConfig, pki: &Pki) -> Result<Channel, AgentError> {
    let rpc = Duration::from_millis(config.limits.rpc_timeout_ms);
    let endpoint = Endpoint::from_shared(config.coordinator.endpoint.clone())
        .map_err(|e| AgentError::Tls(format!("coordinator endpoint: {e}")))?
        .tls_config(pki.client_tls(&config.coordinator.principal))
        .map_err(|e| AgentError::Tls(format!("coordinator tls: {e}")))?
        .connect_timeout(rpc)
        .timeout(rpc)
        .http2_keep_alive_interval(Duration::from_millis(config.limits.heartbeat_interval_ms))
        .keep_alive_timeout(rpc)
        .keep_alive_while_idle(true);
    Ok(endpoint.connect_lazy())
}

/// A running agent.
pub struct AgentHandle {
    shared: Arc<Shared>,
    local_addr: SocketAddr,
    tasks: Vec<JoinHandle<()>>,
}

impl AgentHandle {
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn epoch(&self) -> u64 {
        self.shared.epoch()
    }

    pub fn generation(&self) -> u64 {
        self.shared.generation()
    }

    pub fn term(&self) -> u64 {
        self.shared.term()
    }

    pub fn admission_state(&self) -> AdmissionState {
        self.shared.admission_state()
    }

    /// The latest event lines (at most 4096), oldest first.
    pub fn events(&self) -> Vec<String> {
        self.shared.events.recent()
    }

    pub fn shared(&self) -> &Arc<Shared> {
        &self.shared
    }

    /// Stops the listener (graceful: open streams end `UNAVAILABLE worker_restart`), the link, the probe, the
    /// lifecycle task and the operation runtime (each task gets a bounded wait), then the backend group (§3.2).
    pub async fn shutdown(mut self) {
        self.shared.stop.send_replace(true);
        for task in std::mem::take(&mut self.tasks) {
            let abort = task.abort_handle();
            if tokio::time::timeout(SHUTDOWN_WAIT, task).await.is_err() {
                abort.abort();
            }
        }
        let supervisor = lock(&self.shared.supervisor).take();
        if let Some(sup) = supervisor {
            match Arc::try_unwrap(sup) {
                Ok(sup) => sup.shutdown().await,
                // A launch or stop task still holds it: stop the group; the last drop ends the thread.
                Err(sup) => {
                    if let Err(e) = sup.stop().await {
                        eprintln!("cere-worker: backend stop failed: {e}");
                    }
                }
            }
        }
    }
}

impl Drop for AgentHandle {
    fn drop(&mut self) {
        self.shared.stop.send_replace(true);
        for task in &self.tasks {
            task.abort();
        }
    }
}

/// Whether a managed agent may confirm a backend identity in `model` (§3.6). In `loading` only before the startup
/// deadline instant: the deadline is final even before the lifecycle task wakes to fail the load, and no deadline
/// means a fail cleared it and is stopping the group.
fn confirmable(model: ModelState, load_deadline: Option<Instant>, now: Instant) -> bool {
    match model {
        ModelState::Loading => load_deadline.is_some_and(|at| now < at),
        ModelState::Ready | ModelState::Draining => true,
        ModelState::Attached
        | ModelState::Unloaded
        | ModelState::Unloading
        | ModelState::Failed => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::supervisor::StopError;

    #[test]
    fn a_confirmed_stop_drops_the_child_and_ends_a_pending_unload_unloaded() {
        assert_eq!(
            after_stop(&Ok(()), true),
            AfterStop {
                model: ModelState::Unloaded,
                unload: Some(""),
                drain: wire::BACKEND_EXITED,
                keep_child: false,
            }
        );
        assert_eq!(
            after_stop(&Ok(()), false),
            AfterStop {
                model: ModelState::Failed,
                unload: None,
                drain: wire::BACKEND_EXITED,
                keep_child: false,
            }
        );
    }

    #[test]
    fn every_stop_error_keeps_the_child_and_ends_unloads_and_drains_poisoned() {
        let record = ChildRecord {
            pid: 4242,
            pgid: 4242,
            start_time_ticks: 7,
        };
        let poisoned = AfterStop {
            model: ModelState::Failed,
            unload: Some(wire::POISONED),
            drain: wire::POISONED,
            keep_child: true,
        };
        for unload_pending in [true, false] {
            for error in [
                StopError::Survivors(record),
                StopError::Io(io::Error::from(io::ErrorKind::PermissionDenied)),
                StopError::Closed,
            ] {
                assert_eq!(
                    after_stop(&Err(error), unload_pending),
                    poisoned,
                    "unload_pending={unload_pending}"
                );
            }
        }
    }

    /// §3.6 / Q4: the startup deadline is final. A confirmation in `loading` counts only before the deadline
    /// instant, even when the lifecycle task has not yet woken to fail the load; without a deadline (a fail
    /// cleared it) it never counts. `ready`/`draining` always count; every other state never does.
    #[test]
    fn a_confirmation_counts_only_before_the_startup_deadline() {
        let now = Instant::now();
        let later = now + Duration::from_millis(1);
        assert!(confirmable(ModelState::Loading, Some(later), now));
        assert!(!confirmable(ModelState::Loading, Some(now), now));
        assert!(!confirmable(ModelState::Loading, Some(now), later));
        assert!(!confirmable(ModelState::Loading, None, now));
        for model in [ModelState::Ready, ModelState::Draining] {
            assert!(confirmable(model, None, now), "{model:?}");
        }
        for model in [
            ModelState::Attached,
            ModelState::Unloaded,
            ModelState::Unloading,
            ModelState::Failed,
        ] {
            assert!(!confirmable(model, Some(later), now), "{model:?}");
        }
    }
}
