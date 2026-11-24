//! In-process stub of the M3.5 node agent (`WorkerService` + `WorkerInferenceService`) for the coordinator tests
//! (tracks CREG/CREM, build-loop-m35 design note): enforces the agent's authority, journal and lease rules.
//!
//! The rules are the real agent's (build-loop-m35 design note "Wire behaviour", the agreed "AGENT wire" details):
//! refusal order authz -> quota -> cluster, worker, epoch, coordinator_id, term (lower `stale_term`, higher fences
//! and continues) -> generation (SubmitWork, CancelWork, StreamWork; unset = `stale_generation`) -> malformed
//! (`malformed_request`) -> digest (`payload_digest_mismatch`) -> journal (dedupe or `operation_conflict`) ->
//! admission (`lease_expired`). Reads (DescribeResources, ListOperations, GetOperation) carry only target,
//! coordinator_id and term. CancelWork targets the SubmitWork op of `authority.scope_id` with the same request id
//! and attempt; a wrong ticket is `ticket_mismatch`. Admission is `open` only while the lease of the last accepted
//! heartbeat is live and its reply carried `reconciled_generation` = the current generation (Q7a).
//!
//! [`StubAgent::link`] plays the agent's Register/Heartbeat loop against the coordinator's registry listener. The
//! test scripts the inventory, the reported inventory hash, journaled operations, how work streams end and a gate
//! on `ListOperations`, and reads back every call and every `CancelWork`.
//!
//! [`StubAgent::serve_chat`] (track DISPATCH) makes it run each accepted chat Submit against a backend (an
//! in-process `cere-mock-sglang`) like the agent's op runtime: a unary chat is one event carrying the response
//! JSON then the terminal event; a streamed chat is one event per parsed chunk (`serde_json::to_vec` of the
//! chunk, harness H3a) then the terminal event (`""` at `[DONE]`, else the backend error's API code); sequence =
//! offset + 1, the terminal next; a second attach is `operation_conflict`. A fresh CancelWork of a live run aborts
//! the rid at the backend and pushes the `cancel_requested` terminal; the operation is cleanup-acknowledged once
//! the backend call ended. Faults the test injects: scripted Submit refusals, a lost Submit reply (journaled
//! first), a held Submit reply, a held cleanup, a broken event stream, a terminal reason and a paused link.
//!
//! Lifecycle (track COORD-DRIVE, m36 design note Q6, Q7; design-m36-coord design note §2.4): `LoadModel`,
//! `DrainModel` and `UnloadModel` record every call past authorization (its request, so its op id and
//! Authority), answer a journaled key's stored receipt before any fence (Q6 order), then answer per a script of
//! [`Lifecycle`] answers per method (a receipt by default). Lifecycle ops are served by `GetOperation` and
//! `ListOperations` with their `method`; fences and a new generation end only SubmitWork/CancelWork ops. The
//! described model is settable ([`StubAgent::set_model`]) so a test can walk the coordinator's state machine.

use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use cere_proto::worker::v2 as pb;
use cere_runtime::fed::quota::PrincipalQuota;
use cere_runtime::fed::tls::{Authz, Pki};
use cere_runtime::fed::wire::{self, AdmissionState, Inventory, InventoryModel, ModelState};
use cere_runtime::readiness::REMOTE_SMOKE_SCOPE;
use cere_runtime::sglang::{Rid, SglangHttp, Timeouts};
use pb::worker_inference_service_server::{WorkerInferenceService, WorkerInferenceServiceServer};
use pb::worker_registry_service_client::WorkerRegistryServiceClient;
use pb::worker_service_server::{WorkerService, WorkerServiceServer};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::{TcpListenerStream, UnboundedReceiverStream};
use tokio_stream::Stream;
use tonic::transport::Server;
use tonic::{Code, Request, Response, Status};

use super::{Federation, StateDir, FED_CLUSTER, FED_COORDINATOR, MODEL};

/// The instance id of the stub's backend.
pub const STUB_INSTANCE_ID: &str = "sglang-00stubagent0001";

/// One journaled operation, keyed `(scope_id, operation_id)` like the agent's op journal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StubOp {
    pub scope_id: String,
    pub operation_id: String,
    pub method: &'static str,
    pub request_id: String,
    pub attempt: u64,
    pub ticket: String,
    pub worker_epoch: u64,
    pub model_generation: u64,
    pub coordinator_term: u64,
    pub payload_sha256: Vec<u8>,
    pub terminal: bool,
    pub reason: String,
    pub cleanup_acknowledged: bool,
}

impl StubOp {
    /// A live SubmitWork op `scope_id/operation_id` (request id = operation id, attempt 1).
    pub fn submit(
        scope_id: &str,
        operation_id: &str,
        epoch: u64,
        generation: u64,
        term: u64,
    ) -> StubOp {
        StubOp {
            scope_id: scope_id.to_owned(),
            operation_id: operation_id.to_owned(),
            method: wire::METHOD_SUBMIT,
            request_id: operation_id.to_owned(),
            attempt: 1,
            ticket: format!("ticket-{scope_id}-{operation_id}"),
            worker_epoch: epoch,
            model_generation: generation,
            coordinator_term: term,
            payload_sha256: vec![0; 32],
            terminal: false,
            reason: String::new(),
            cleanup_acknowledged: false,
        }
    }

    fn status(&self) -> pb::OperationStatus {
        let phase = match (self.terminal, self.cleanup_acknowledged) {
            (_, true) => wire::PHASE_RELEASED,
            (true, false) => wire::PHASE_TERMINAL,
            (false, false) => wire::PHASE_ACCEPTED,
        };
        pb::OperationStatus {
            operation_id: self.operation_id.clone(),
            phase: phase.to_owned(),
            terminal: self.terminal,
            cleanup_acknowledged: self.cleanup_acknowledged,
            model_generation: self.model_generation,
            completion_ticket: self.ticket.clone(),
            reason_code: self.reason.clone(),
            coordinator_term: self.coordinator_term,
            worker_epoch: self.worker_epoch,
            scope_id: self.scope_id.clone(),
            request_id: self.request_id.clone(),
            attempt: self.attempt,
            method: self.method.to_owned(),
        }
    }
}

/// The scripted answer of one lifecycle call (a fresh key that passed the fences and the digest).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    /// Journal the op and answer its receipt; the op stays live and the model unchanged.
    Receipt,
    /// Refuse `code reason` before the journal.
    Refuse(Code, &'static str),
    /// Journal the op, then answer a reset connection (the receipt is dropped); a retry of the same key gets the
    /// stored receipt.
    DropReceipt,
    /// Journal the op, answer its receipt, then describe the model in each of `states` in turn ([`WALK_STEP`]
    /// apart; entering `ready` confirms a new backend instance under a new generation), then end the op with
    /// `reason`.
    Walk {
        states: &'static [ModelState],
        reason: &'static str,
    },
}

/// The time between two states of a [`Lifecycle::Walk`].
pub const WALK_STEP: Duration = Duration::from_millis(150);

fn is_lifecycle(method: &str) -> bool {
    matches!(
        method,
        wire::METHOD_LOAD | wire::METHOD_DRAIN | wire::METHOD_UNLOAD
    )
}

/// One recorded lifecycle call: the method and the request as received.
pub type LifecycleCall = (&'static str, pb::ModelLifecycleRequest);

/// How a `StreamWork` of a live op ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkOutcome {
    /// One event carrying a unary 1-token chat response (`finish_reason` `length`), then the terminal event.
    Complete,
    /// The stream fails `UNAVAILABLE worker_restart` before any event; the op stays live.
    Break,
    /// Only the terminal event, carrying this reason; the op turns terminal with it.
    End(&'static str),
}

/// Every call that passed authorization, by method name, in arrival order.
pub type Calls = Vec<&'static str>;

/// A protocol violation the next served `StreamWork` commits right after its first event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Violation {
    /// An event above the 64 KiB bound.
    Oversize,
    /// An event whose `committed_token_offset` skips ahead.
    OffsetGap,
    /// The stream ends cleanly without a terminal event.
    EndWithoutTerminal,
    /// An event whose payload is not JSON.
    Undecodable,
    /// The stream fails with this (non-UNAVAILABLE) status code.
    Status(Code),
}

/// A served chat run: its events until `StreamWork` attaches, and the signal a fresh `CancelWork` raises.
struct Run {
    events: Option<mpsc::UnboundedReceiver<pb::WorkEvent>>,
    cancel: watch::Sender<bool>,
}

struct State {
    worker_id: String,
    epoch: u64,
    generation: u64,
    /// The highest coordinator term accepted (0 = none yet).
    term: u64,
    /// The last heartbeat reply carried `reconciled_generation` = the generation it was sent with.
    reconciled: bool,
    lease_until: Option<Instant>,
    /// The last heartbeat sequence sent: kept across a re-register within the epoch (like the real agent),
    /// reset by a new incarnation ([`StubAgent::restart`]).
    sequence: u64,
    /// The last accepted heartbeat reply's `desired_state` was `DRAINING` (the real agent fences on it).
    draining: bool,
    /// Reported as the heartbeat's `owner_only` (team pools, C19); the stub does not refuse work on it.
    owner_only: bool,
    inventory: Inventory,
    inventory_bytes: Vec<u8>,
    /// Reported in Register and heartbeats instead of the hash of `inventory_bytes`.
    reported_sha256: Option<[u8; 32]>,
    ops: BTreeMap<(String, String), StubOp>,
    next_ticket: u64,
    /// How the work of every scope but the readiness smoke's ends.
    outcome: WorkOutcome,
    /// How the readiness smoke's work ([`REMOTE_SMOKE_SCOPE`]) ends: apart from `outcome`, so a chat scripted to
    /// break never fails a re-smoke that runs meanwhile (the worker would stay Joining and the chat queued).
    smoke_outcome: WorkOutcome,
    calls: Calls,
    submits: Vec<pb::SubmitWorkRequest>,
    /// Every authorized SubmitWork with a matching digest, refused ones and journal retries included.
    submit_attempts: Vec<pb::SubmitWorkRequest>,
    cancels: Vec<pb::CancelWorkRequest>,
    /// The backend every accepted Submit runs against ([`StubAgent::serve_chat`]).
    backend: Option<SglangHttp>,
    runs: BTreeMap<(String, String), Run>,
    /// Scripted refusals, one per following SubmitWork.
    refusals: VecDeque<(Code, &'static str)>,
    /// How many following SubmitWork calls lose their reply after the journal step.
    lost_replies: usize,
    /// Every SubmitWork the journal answered: request id and whether it was a fresh accept.
    accepts: Vec<(String, bool)>,
    /// Finished runs keep `cleanup_acknowledged` unset while this holds.
    hold_cleanup: bool,
    held: Vec<(String, String)>,
    violation: Option<Violation>,
    /// Scripted answers per lifecycle method, one per following fresh call ([`Lifecycle::Receipt`] when empty).
    lifecycle_answers: BTreeMap<&'static str, VecDeque<Lifecycle>>,
    /// Every lifecycle call past authorization, in arrival order.
    lifecycle_calls: Vec<LifecycleCall>,
}

impl State {
    fn open(&self) -> bool {
        self.reconciled
            && !self.draining
            && self.lease_until.is_some_and(|until| Instant::now() < until)
    }

    fn admission(&self) -> AdmissionState {
        if self.open() {
            AdmissionState::Open
        } else {
            AdmissionState::Fenced
        }
    }

    fn reported(&self) -> [u8; 32] {
        self.reported_sha256
            .unwrap_or_else(|| Inventory::sha256(&self.inventory_bytes))
    }

    /// A higher coordinator term fences: admission closes and every live SubmitWork/CancelWork op of an older
    /// term ends `coordinator_restart` (lifecycle ops end only through the model state machine, Q6).
    fn fence(&mut self, term: u64) {
        self.term = term;
        self.reconciled = false;
        for op in self.ops.values_mut() {
            if !op.terminal && !is_lifecycle(op.method) && op.coordinator_term < term {
                op.terminal = true;
                op.reason = wire::COORDINATOR_RESTART.to_owned();
            }
        }
    }

    fn describe(&mut self, inventory: Inventory) {
        self.inventory_bytes = inventory.to_json_bytes().unwrap();
        self.inventory = inventory;
    }

    /// A new model generation (Q5): admission closes, the live SubmitWork/CancelWork ops of the old generation
    /// end `model_generation_superseded`, and the inventory names the generation.
    fn set_generation(&mut self, generation: u64) {
        self.generation = generation;
        self.reconciled = false;
        for op in self.ops.values_mut() {
            if !op.terminal && !is_lifecycle(op.method) && op.model_generation != generation {
                op.terminal = true;
                op.reason = wire::MODEL_GENERATION_SUPERSEDED.to_owned();
            }
        }
        let mut inventory = self.inventory.clone();
        inventory.model_generation = generation;
        for model in &mut inventory.models {
            model.model_generation = generation;
        }
        self.describe(inventory);
    }

    /// Describes the managed model in `state` under `lock` (Q7): the backend identity fields name `instance_id`
    /// in `ready`/`draining`/`unloading` and are null otherwise.
    fn set_model(&mut self, state: ModelState, lock: Option<String>, instance_id: &str) {
        let mut inventory = self.inventory.clone();
        let identity = matches!(
            state,
            ModelState::Ready | ModelState::Draining | ModelState::Unloading
        );
        let instance = identity.then(|| instance_id.to_owned());
        inventory.tp_size = identity.then_some(1);
        inventory.kv_cache_dtype = identity.then(|| "fp8_e4m3".to_owned());
        inventory.max_running_requests = identity.then_some(4);
        inventory.instance_id = instance.clone();
        inventory.models[0].instance_id = instance;
        inventory.models[0].lock_sha256 = lock;
        inventory.models[0].state = state;
        self.describe(inventory);
    }

    /// The described backend instance, or the stub's default one.
    fn instance_id(&self) -> String {
        self.inventory.models[0]
            .instance_id
            .clone()
            .unwrap_or_else(|| STUB_INSTANCE_ID.to_owned())
    }

    /// A journaled SubmitWork answer, unless the reply is to be lost: then the caller sees a reset connection.
    fn reply<T>(&mut self, answer: Result<T, Status>) -> Result<T, Status> {
        if answer.is_ok() && self.lost_replies > 0 {
            self.lost_replies -= 1;
            return Err(Status::unavailable(
                "connection reset by peer (stub: reply lost)",
            ));
        }
        answer
    }
}

struct Shared {
    state: Mutex<State>,
    authz: Authz,
    quota: PrincipalQuota,
    /// `true` lets `ListOperations` answer; `false` holds every call until it turns `true`.
    list_gate: watch::Sender<bool>,
    /// `true` lets a fresh SubmitWork accept answer; `false` holds its reply (journaled first) until `true`.
    submit_gate: watch::Sender<bool>,
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn refuse(code: Code, reason: &str) -> Status {
    wire::status(code, reason)
}

fn malformed() -> Status {
    refuse(Code::InvalidArgument, wire::MALFORMED_REQUEST)
}

/// Which authority form a method takes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    /// DescribeResources, ListOperations: target, coordinator_id and term only.
    Read,
    /// GetOperation: a read that may name the scope of the queried op.
    Get,
    /// SubmitWork, CancelWork, StreamWork: scope, operation id, budget and the current generation.
    Op,
}

#[derive(Clone)]
struct Service {
    shared: Arc<Shared>,
}

impl Service {
    /// Authorization and the control quota, before any state.
    fn admit<T>(&self, req: &Request<T>) -> Result<cere_runtime::fed::quota::QuotaGuard, Status> {
        let principal = self.shared.authz.coordinator_caller(req)?;
        self.shared.quota.try_acquire(&principal)
    }

    /// The target, coordinator and term checks, then the generation and the form of the authority.
    fn authority(
        &self,
        authority: &Option<pb::Authority>,
        form: Form,
        method: &'static str,
    ) -> Result<pb::Authority, Status> {
        let a = authority.as_ref().ok_or_else(malformed)?;
        let target = a.target.as_ref().ok_or_else(malformed)?;
        let mut st = self.shared.state();
        let precondition = |reason| Err(refuse(Code::FailedPrecondition, reason));
        if target.cluster_id != FED_CLUSTER {
            return precondition(wire::WRONG_CLUSTER);
        }
        if target.worker_id != st.worker_id {
            return precondition(wire::WRONG_WORKER);
        }
        if target.worker_epoch != st.epoch {
            return precondition(wire::STALE_EPOCH);
        }
        if a.coordinator_id != FED_COORDINATOR {
            return precondition(wire::WRONG_COORDINATOR);
        }
        if a.coordinator_term < st.term {
            return precondition(wire::STALE_TERM);
        }
        if a.coordinator_term > st.term {
            st.fence(a.coordinator_term);
        }
        if form == Form::Op && a.model_generation != Some(st.generation) {
            return precondition(wire::STALE_GENERATION);
        }
        let well_formed = match form {
            Form::Read => {
                a.scope_id.is_empty()
                    && a.operation_id.is_empty()
                    && a.payload_sha256.is_empty()
                    && a.model_generation.is_none()
            }
            Form::Get => {
                a.operation_id.is_empty()
                    && a.payload_sha256.is_empty()
                    && a.model_generation.is_none()
            }
            Form::Op => !a.scope_id.is_empty() && !a.operation_id.is_empty(),
        };
        if !well_formed {
            return Err(malformed());
        }
        st.calls.push(method);
        Ok(a.clone())
    }

    fn receipt(operation_id: &str, ticket: &str) -> Response<pb::OperationReceipt> {
        Response::new(pb::OperationReceipt {
            accepted: true,
            operation_id: operation_id.to_owned(),
            completion_ticket: ticket.to_owned(),
            reason_code: String::new(),
        })
    }

    /// The dedupe rule of the op journal: the same key with the same identity answers the original receipt, any
    /// other identity is `operation_conflict`.
    fn dedupe(
        st: &State,
        key: &(String, String),
        method: &'static str,
        a: &pb::Authority,
        request_id: &str,
        attempt: u64,
    ) -> Option<Result<Response<pb::OperationReceipt>, Status>> {
        let op = st.ops.get(key)?;
        let same = op.method == method
            && op.payload_sha256 == a.payload_sha256
            && op.request_id == request_id
            && op.attempt == attempt
            && op.worker_epoch == st.epoch
            && Some(op.model_generation) == a.model_generation
            && op.coordinator_term == a.coordinator_term;
        Some(if same {
            Ok(Self::receipt(&op.operation_id, &op.ticket))
        } else {
            Err(refuse(Code::AlreadyExists, wire::OPERATION_CONFLICT))
        })
    }

    /// A lifecycle call (Q6 order): authz and quota, the request shape, the record, the journal lookup (a
    /// journaled key with the same identity answers its stored receipt whatever the current epoch, term or
    /// generation; another identity is `operation_conflict`), then for a new key the fences, the digest and the
    /// scripted answer.
    fn lifecycle(
        &self,
        req: Request<pb::ModelLifecycleRequest>,
        method: &'static str,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        let _slot = self.admit(&req)?;
        let r = req.into_inner();
        let a = r.authority.as_ref().ok_or_else(malformed)?;
        let target = a.target.as_ref().ok_or_else(malformed)?;
        if a.scope_id.is_empty()
            || a.operation_id.is_empty()
            || a.payload_sha256.len() != 32
            || r.deployment_lock_sha256.len() != 32
            || !r.device_ids.is_empty()
        {
            return Err(malformed());
        }
        let key = (a.scope_id.clone(), a.operation_id.clone());
        {
            let mut st = self.shared.state();
            st.lifecycle_calls.push((method, r.clone()));
            if let Some(op) = st.ops.get(&key) {
                let same = op.method == method
                    && op.payload_sha256 == a.payload_sha256
                    && op.worker_epoch == target.worker_epoch
                    && Some(op.model_generation) == a.model_generation
                    && op.coordinator_term == a.coordinator_term;
                return if same {
                    Ok(Self::receipt(&op.operation_id, &op.ticket))
                } else {
                    Err(refuse(Code::AlreadyExists, wire::OPERATION_CONFLICT))
                };
            }
        }
        let a = self.authority(&r.authority, Form::Op, method)?;
        let digest = wire::lifecycle_sha256(
            method,
            &a.scope_id,
            &r.deployment_lock_sha256,
            &r.instance_id,
            &r.device_ids,
        );
        if a.payload_sha256 != digest {
            return Err(refuse(Code::InvalidArgument, wire::PAYLOAD_DIGEST_MISMATCH));
        }
        let mut st = self.shared.state();
        let answer = st
            .lifecycle_answers
            .get_mut(method)
            .and_then(VecDeque::pop_front)
            .unwrap_or(Lifecycle::Receipt);
        if let Lifecycle::Refuse(code, reason) = answer {
            return Err(refuse(code, reason));
        }
        st.next_ticket += 1;
        let ticket = format!("lifecycle-ticket-{}", st.next_ticket);
        let op = StubOp {
            scope_id: a.scope_id.clone(),
            operation_id: a.operation_id.clone(),
            method,
            request_id: String::new(),
            attempt: 0,
            ticket: ticket.clone(),
            worker_epoch: st.epoch,
            model_generation: st.generation,
            coordinator_term: a.coordinator_term,
            payload_sha256: a.payload_sha256.clone(),
            terminal: false,
            reason: String::new(),
            cleanup_acknowledged: false,
        };
        st.ops.insert(key.clone(), op);
        match answer {
            Lifecycle::DropReceipt => Err(Status::unavailable(
                "connection reset by peer (stub: receipt dropped)",
            )),
            Lifecycle::Walk { states, reason } => {
                tokio::spawn(walk(self.shared.clone(), key, states, reason));
                Ok(Self::receipt(&a.operation_id, &ticket))
            }
            Lifecycle::Receipt | Lifecycle::Refuse(..) => {
                Ok(Self::receipt(&a.operation_id, &ticket))
            }
        }
    }
}

/// [`Lifecycle::Walk`]: one described state per [`WALK_STEP`] (entering `ready` confirms a new instance under a
/// new generation); the op ends with `reason` at the last state, cleanup-acknowledged (Q4).
async fn walk(
    shared: Arc<Shared>,
    key: (String, String),
    states: &'static [ModelState],
    reason: &'static str,
) {
    for (index, &state) in states.iter().enumerate() {
        tokio::time::sleep(WALK_STEP).await;
        let mut st = shared.state();
        let lock = st.inventory.models[0].lock_sha256.clone();
        if state == ModelState::Ready {
            let generation = st.generation + 1;
            st.set_generation(generation);
            st.set_model(state, lock, &format!("sglang-{generation:016x}"));
        } else {
            let instance = st.instance_id();
            st.set_model(state, lock, &instance);
        }
        if index + 1 == states.len() {
            if let Some(op) = st.ops.get_mut(&key) {
                op.terminal = true;
                op.reason = reason.to_owned();
                op.cleanup_acknowledged = true;
            }
        }
    }
}

/// The unary chat response of a 1-token greedy chat.
fn unary_response(rid: &str) -> Value {
    json!({
        "id": rid,
        "object": "chat.completion",
        "model": MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "p"},
            "finish_reason": "length",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })
}

type EventStream = Pin<Box<dyn Stream<Item = Result<pb::WorkEvent, Status>> + Send>>;

/// The events of one served run: `committed_token_offset` = event index from 0, `sequence` = offset + 1.
struct Emit {
    tx: mpsc::UnboundedSender<pb::WorkEvent>,
    offset: u64,
    template: pb::WorkEvent,
}

impl Emit {
    fn push(&mut self, payload: Vec<u8>, terminal: bool, reason: &str) {
        let _ = self.tx.send(pb::WorkEvent {
            sequence: self.offset + 1,
            committed_token_offset: self.offset,
            bounded_output_event_json: payload,
            terminal,
            reason_code: reason.to_owned(),
            ..self.template.clone()
        });
        self.offset += 1;
    }
}

/// One accepted chat Submit run against the stub's backend, like the agent's op runtime.
struct Chat {
    shared: Arc<Shared>,
    key: (String, String),
    backend: SglangHttp,
    rid: Rid,
    emit: Emit,
    cancelled: watch::Receiver<bool>,
}

/// Resolves once a fresh `CancelWork` raised the run's signal (never if the signal is gone).
async fn cancelled(mut signal: watch::Receiver<bool>) {
    if signal.wait_for(|c| *c).await.is_err() {
        std::future::pending::<()>().await;
    }
}

impl Chat {
    async fn run(mut self, body: Value, stream: bool, deadline: Duration) {
        let cancel = cancelled(self.cancelled.clone());
        tokio::pin!(cancel);
        if stream {
            self.run_stream(&body, deadline, cancel).await;
        } else {
            let backend = self.backend.clone();
            let call = backend.chat(&body, deadline);
            tokio::pin!(call);
            let result = tokio::select! {
                result = &mut call => Some(result),
                () = &mut cancel => None,
            };
            match result {
                Some(Ok(response)) => {
                    self.emit
                        .push(serde_json::to_vec(&response).unwrap(), false, "");
                    self.finish("");
                }
                Some(Err(e)) => self.finish(e.op_reason()),
                None => {
                    let _ = self.backend.abort(&self.rid).await;
                    self.finish(wire::CANCEL_REQUESTED);
                    let _ = call.await;
                }
            }
        }
        // The backend call ended: the operation's resources are released.
        let mut st = self.shared.state();
        if st.hold_cleanup {
            st.held.push(self.key.clone());
        } else if let Some(op) = st.ops.get_mut(&self.key) {
            op.cleanup_acknowledged = true;
        }
    }

    async fn run_stream(
        &mut self,
        body: &Value,
        deadline: Duration,
        mut cancel: Pin<&mut impl Future<Output = ()>>,
    ) {
        let backend = self.backend.clone();
        let opened = tokio::select! {
            opened = backend.chat_stream(body, deadline) => Some(opened),
            () = &mut cancel => None,
        };
        let mut chunks = match opened {
            Some(Ok(chunks)) => chunks,
            Some(Err(e)) => return self.finish(e.op_reason()),
            None => {
                let _ = self.backend.abort(&self.rid).await;
                return self.finish(wire::CANCEL_REQUESTED);
            }
        };
        loop {
            let chunk = tokio::select! {
                chunk = chunks.next_chunk() => Some(chunk),
                () = &mut cancel => None,
            };
            match chunk {
                Some(Ok(Some(chunk))) => {
                    self.emit
                        .push(serde_json::to_vec(&chunk).unwrap(), false, "")
                }
                Some(Ok(None)) => return self.finish(""),
                Some(Err(e)) => return self.finish(e.op_reason()),
                None => {
                    let _ = self.backend.abort(&self.rid).await;
                    self.finish(wire::CANCEL_REQUESTED);
                    while let Ok(Some(_)) = chunks.next_chunk().await {}
                    return;
                }
            }
        }
    }

    /// The op turns terminal with `reason` (unless something ended it first) and the terminal event goes out.
    fn finish(&mut self, reason: &str) {
        if let Some(op) = self.shared.state().ops.get_mut(&self.key) {
            if !op.terminal {
                op.terminal = true;
                op.reason = reason.to_owned();
            }
        }
        self.emit.push(Vec::new(), true, reason);
    }
}

/// The `StreamWork` stream of a served run, committing `violation` right after the first event.
fn attach(
    mut events: mpsc::UnboundedReceiver<pb::WorkEvent>,
    violation: Option<Violation>,
) -> EventStream {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        let mut forwarded = 0;
        while let Some(event) = events.recv().await {
            if let Some(violation) = violation.filter(|_| forwarded == 1) {
                let bad = match violation {
                    Violation::Oversize => Ok(pb::WorkEvent {
                        bounded_output_event_json: vec![b' '; wire::WORK_EVENT_MAX + 1],
                        ..event
                    }),
                    Violation::OffsetGap => Ok(pb::WorkEvent {
                        committed_token_offset: event.committed_token_offset + 5,
                        ..event
                    }),
                    Violation::EndWithoutTerminal => return,
                    Violation::Undecodable => Ok(pb::WorkEvent {
                        bounded_output_event_json: b"{not json".to_vec(),
                        ..event
                    }),
                    Violation::Status(code) => Err(Status::new(code, "stub: injected failure")),
                };
                let _ = tx.send(bad);
                return;
            }
            if tx.send(Ok(event)).is_err() {
                return;
            }
            forwarded += 1;
        }
    });
    Box::pin(UnboundedReceiverStream::new(rx))
}

#[tonic::async_trait]
impl WorkerService for Service {
    async fn describe_resources(
        &self,
        req: Request<pb::InventoryRequest>,
    ) -> Result<Response<pb::InventoryReply>, Status> {
        let _slot = self.admit(&req)?;
        self.authority(&req.get_ref().authority, Form::Read, "DescribeResources")?;
        let st = self.shared.state();
        Ok(Response::new(pb::InventoryReply {
            bounded_resource_inventory_json: st.inventory_bytes.clone(),
            inventory_generation: st.generation,
        }))
    }

    async fn load_model(
        &self,
        req: Request<pb::ModelLifecycleRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.lifecycle(req, wire::METHOD_LOAD)
    }

    async fn drain_model(
        &self,
        req: Request<pb::ModelLifecycleRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.lifecycle(req, wire::METHOD_DRAIN)
    }

    async fn unload_model(
        &self,
        req: Request<pb::ModelLifecycleRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.lifecycle(req, wire::METHOD_UNLOAD)
    }

    async fn cancel_work(
        &self,
        req: Request<pb::CancelWorkRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        let _slot = self.admit(&req)?;
        let r = req.get_ref();
        let a = self.authority(&r.authority, Form::Op, "CancelWork")?;
        if a.payload_sha256.len() != 32 || r.request_id.is_empty() || r.attempt == 0 {
            return Err(malformed());
        }
        let digest = wire::payload_sha256(
            wire::METHOD_CANCEL,
            &a.scope_id,
            &r.request_id,
            r.attempt,
            b"",
            r.completion_ticket.as_bytes(),
        );
        if a.payload_sha256 != digest {
            return Err(refuse(Code::InvalidArgument, wire::PAYLOAD_DIGEST_MISMATCH));
        }
        let mut st = self.shared.state();
        let key = (a.scope_id.clone(), a.operation_id.clone());
        if let Some(answer) =
            Service::dedupe(&st, &key, wire::METHOD_CANCEL, &a, &r.request_id, r.attempt)
        {
            return answer;
        }
        let st = &mut *st;
        let generation = st.generation;
        let target = st.ops.values_mut().find(|op| {
            op.method == wire::METHOD_SUBMIT
                && op.scope_id == a.scope_id
                && op.request_id == r.request_id
                && op.attempt == r.attempt
        });
        let Some(target) = target
            .filter(|op| op.ticket == r.completion_ticket && op.model_generation == generation)
        else {
            return Err(refuse(Code::FailedPrecondition, wire::TICKET_MISMATCH));
        };
        let target_key = (target.scope_id.clone(), target.operation_id.clone());
        if !target.terminal {
            target.terminal = true;
            target.reason = wire::CANCEL_REQUESTED.to_owned();
            match st.runs.get(&target_key) {
                // A served run: it aborts the rid at the backend, pushes the `cancel_requested` terminal and
                // acknowledges cleanup once the backend call ended.
                Some(run) => {
                    run.cancel.send_replace(true);
                }
                None => target.cleanup_acknowledged = true,
            }
        }
        st.next_ticket += 1;
        let ticket = format!("cancel-ticket-{}", st.next_ticket);
        let op = StubOp {
            scope_id: a.scope_id.clone(),
            operation_id: a.operation_id.clone(),
            method: wire::METHOD_CANCEL,
            request_id: r.request_id.clone(),
            attempt: r.attempt,
            ticket: ticket.clone(),
            worker_epoch: st.epoch,
            model_generation: generation,
            coordinator_term: a.coordinator_term,
            payload_sha256: a.payload_sha256.clone(),
            terminal: true,
            reason: String::new(),
            cleanup_acknowledged: true,
        };
        st.ops.insert(key, op);
        st.cancels.push(r.clone());
        Ok(Service::receipt(&a.operation_id, &ticket))
    }

    async fn get_operation(
        &self,
        req: Request<pb::OperationQuery>,
    ) -> Result<Response<pb::OperationStatus>, Status> {
        let _slot = self.admit(&req)?;
        let r = req.get_ref();
        let a = self.authority(&r.authority, Form::Get, "GetOperation")?;
        let st = self.shared.state();
        let mut found = st.ops.values().filter(|op| {
            op.operation_id == r.queried_operation_id
                && (a.scope_id.is_empty() || op.scope_id == a.scope_id)
        });
        match (found.next(), found.next()) {
            (Some(op), None) => Ok(Response::new(op.status())),
            _ => Err(refuse(Code::NotFound, wire::UNKNOWN_OPERATION)),
        }
    }

    async fn list_operations(
        &self,
        req: Request<pb::ListOperationsRequest>,
    ) -> Result<Response<pb::ListOperationsReply>, Status> {
        let _slot = self.admit(&req)?;
        let r = req.get_ref();
        self.authority(&r.authority, Form::Read, "ListOperations")?;
        let limit = r.limit as usize;
        if !(1..=wire::LIST_PAGE_MAX).contains(&limit) {
            return Err(malformed());
        }
        let mut gate = self.shared.list_gate.subscribe();
        let _ = gate.wait_for(|open| *open).await;
        let st = self.shared.state();
        let after = (r.after_scope_id.clone(), r.after_operation_id.clone());
        let mut page = st
            .ops
            .range(after..)
            .filter(|(key, _)| key.0 != r.after_scope_id || key.1 != r.after_operation_id)
            .map(|(_, op)| op.status());
        let operations: Vec<_> = page.by_ref().take(limit).collect();
        let more = page.next().is_some();
        Ok(Response::new(pb::ListOperationsReply { operations, more }))
    }

    type StreamWorkStream = EventStream;

    async fn stream_work(
        &self,
        req: Request<pb::WorkStreamRequest>,
    ) -> Result<Response<EventStream>, Status> {
        self.shared.authz.coordinator_caller(&req)?;
        let r = req.get_ref();
        let a = self.authority(&r.authority, Form::Op, "StreamWork")?;
        let mut st = self.shared.state();
        let st = &mut *st;
        let outcome = if a.scope_id == REMOTE_SMOKE_SCOPE {
            st.smoke_outcome
        } else {
            st.outcome
        };
        let target = Some(pb::WorkerIdentity {
            cluster_id: FED_CLUSTER.to_owned(),
            worker_id: st.worker_id.clone(),
            worker_epoch: st.epoch,
        });
        let key = (a.scope_id.clone(), a.operation_id.clone());
        let Some(op) = st.ops.get_mut(&key) else {
            return Err(refuse(Code::NotFound, wire::UNKNOWN_OPERATION));
        };
        if op.method != wire::METHOD_SUBMIT
            || op.request_id != r.request_id
            || op.attempt != r.attempt
            || op.ticket != r.completion_ticket
        {
            return Err(refuse(Code::FailedPrecondition, wire::TICKET_MISMATCH));
        }
        if let Some(run) = st.runs.get_mut(&key) {
            // A served run streams its events once; a second attach is a conflict.
            let Some(events) = run.events.take() else {
                return Err(refuse(Code::AlreadyExists, wire::OPERATION_CONFLICT));
            };
            return Ok(Response::new(attach(events, st.violation.take())));
        }
        let event = |sequence: u64, payload: Vec<u8>, terminal: bool, reason: &str| pb::WorkEvent {
            worker: target.clone(),
            coordinator_term: a.coordinator_term,
            scope_id: key.0.clone(),
            request_id: r.request_id.clone(),
            attempt: r.attempt,
            sequence,
            completion_ticket: r.completion_ticket.clone(),
            committed_token_offset: sequence - 1,
            bounded_output_event_json: payload,
            terminal,
            reason_code: reason.to_owned(),
        };
        let events = if op.terminal {
            vec![Ok(event(1, Vec::new(), true, &op.reason))]
        } else {
            match outcome {
                WorkOutcome::Complete => {
                    op.terminal = true;
                    op.cleanup_acknowledged = true;
                    let body = serde_json::to_vec(&unary_response(&r.request_id)).unwrap();
                    vec![
                        Ok(event(1, body, false, "")),
                        Ok(event(2, Vec::new(), true, "")),
                    ]
                }
                WorkOutcome::Break => {
                    vec![Err(refuse(Code::Unavailable, wire::WORKER_RESTART))]
                }
                WorkOutcome::End(reason) => {
                    op.terminal = true;
                    op.reason = reason.to_owned();
                    op.cleanup_acknowledged = true;
                    vec![Ok(event(1, Vec::new(), true, reason))]
                }
            }
        };
        Ok(Response::new(Box::pin(tokio_stream::iter(events))))
    }

    type StreamTelemetryStream =
        Pin<Box<dyn Stream<Item = Result<pb::TelemetryEnvelope, Status>> + Send>>;

    async fn stream_telemetry(
        &self,
        _: Request<pb::TelemetryRequest>,
    ) -> Result<Response<Self::StreamTelemetryStream>, Status> {
        Err(Status::unimplemented(""))
    }
}

#[tonic::async_trait]
impl WorkerInferenceService for Service {
    async fn submit_work(
        &self,
        req: Request<pb::SubmitWorkRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        let _slot = self.admit(&req)?;
        let r = req.get_ref();
        let a = self.authority(&r.authority, Form::Op, "SubmitWork")?;
        if a.payload_sha256.len() != 32 || r.attempt == 0 {
            return Err(malformed());
        }
        let work_unit: Value =
            serde_json::from_slice(&r.work_unit_json).map_err(|_| malformed())?;
        let unit_ok = work_unit.as_object().is_some_and(|unit| {
            unit.len() == 2
                && unit.get("kind") == Some(&json!("chat"))
                && unit["stream"].is_boolean()
        });
        let body: Value =
            serde_json::from_slice(&r.bounded_inference_input_json).map_err(|_| malformed())?;
        let rid_ok = body.get("rid").and_then(Value::as_str) == Some(r.request_id.as_str())
            && Rid::parse(&r.request_id).is_ok();
        if !unit_ok || !rid_ok {
            return Err(malformed());
        }
        let digest = wire::payload_sha256(
            wire::METHOD_SUBMIT,
            &a.scope_id,
            &r.request_id,
            r.attempt,
            &r.work_unit_json,
            &r.bounded_inference_input_json,
        );
        if a.payload_sha256 != digest {
            return Err(refuse(Code::InvalidArgument, wire::PAYLOAD_DIGEST_MISMATCH));
        }
        // The state guard ends with the journal step, before a held reply waits.
        let answer = {
            let mut st = self.shared.state();
            st.submit_attempts.push(r.clone());
            if let Some((code, reason)) = st.refusals.pop_front() {
                return Err(refuse(code, reason));
            }
            let key = (a.scope_id.clone(), a.operation_id.clone());
            if let Some(answer) =
                Service::dedupe(&st, &key, wire::METHOD_SUBMIT, &a, &r.request_id, r.attempt)
            {
                if answer.is_ok() {
                    st.accepts.push((r.request_id.clone(), false));
                }
                return st.reply(answer);
            }
            if !st.open() {
                return Err(refuse(Code::Unavailable, wire::LEASE_EXPIRED));
            }
            // A new key at or above the backend's `max_running_requests` live ops is refused (a retry of a held key
            // was answered by the journal above).
            let live = st
                .ops
                .values()
                .filter(|op| op.method == wire::METHOD_SUBMIT && !op.cleanup_acknowledged)
                .count() as u64;
            if st
                .inventory
                .max_running_requests
                .is_none_or(|cap| live >= cap)
            {
                return Err(refuse(Code::ResourceExhausted, wire::QUEUE_FULL));
            }
            st.next_ticket += 1;
            let ticket = format!("ticket-{}", st.next_ticket);
            let op = StubOp {
                ticket: ticket.clone(),
                request_id: r.request_id.clone(),
                attempt: r.attempt,
                payload_sha256: a.payload_sha256.clone(),
                ..StubOp::submit(
                    &a.scope_id,
                    &a.operation_id,
                    st.epoch,
                    st.generation,
                    a.coordinator_term,
                )
            };
            st.ops.insert(key.clone(), op);
            st.submits.push(r.clone());
            st.accepts.push((r.request_id.clone(), true));
            if let Some(backend) = st.backend.clone() {
                let (tx, events) = mpsc::unbounded_channel();
                let (cancel, cancelled) = watch::channel(false);
                let run = Run {
                    events: Some(events),
                    cancel,
                };
                st.runs.insert(key.clone(), run);
                let emit = Emit {
                    tx,
                    offset: 0,
                    template: pb::WorkEvent {
                        worker: Some(pb::WorkerIdentity {
                            cluster_id: FED_CLUSTER.to_owned(),
                            worker_id: st.worker_id.clone(),
                            worker_epoch: st.epoch,
                        }),
                        coordinator_term: a.coordinator_term,
                        scope_id: a.scope_id.clone(),
                        request_id: r.request_id.clone(),
                        attempt: r.attempt,
                        completion_ticket: ticket.clone(),
                        ..Default::default()
                    },
                };
                let chat = Chat {
                    shared: self.shared.clone(),
                    key,
                    backend,
                    rid: Rid::parse(&r.request_id).map_err(|_| malformed())?,
                    emit,
                    cancelled,
                };
                let stream = work_unit["stream"] == json!(true);
                let deadline = Duration::from_millis(a.remaining_budget_ms.max(1));
                tokio::spawn(chat.run(body, stream, deadline));
            }
            st.reply(Ok(Service::receipt(&a.operation_id, &ticket)))
        };
        let mut gate = self.shared.submit_gate.subscribe();
        let _ = gate.wait_for(|open| *open).await;
        answer
    }
}

/// A running stub agent: its mTLS listener and the shared state the test scripts.
pub struct StubAgent {
    shared: Arc<Shared>,
    addr: SocketAddr,
    principal: String,
    _dir: StateDir,
    stop: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
    link: Option<JoinHandle<()>>,
}

/// The default (attached-mode) inventory of `worker_id` at `generation`: pins of [`Federation::toml`].
pub fn inventory(worker_id: &str, generation: u64) -> Inventory {
    Inventory {
        schema_version: wire::INVENTORY_SCHEMA_VERSION,
        worker_id: worker_id.to_owned(),
        served_model_name: MODEL.to_owned(),
        tp_size: Some(1),
        kv_cache_dtype: Some("fp8_e4m3".to_owned()),
        max_running_requests: Some(4),
        instance_id: Some(STUB_INSTANCE_ID.to_owned()),
        model_generation: generation,
        admission_state: AdmissionState::Fenced,
        gpus: Vec::new(),
        models: vec![InventoryModel {
            served_model_name: MODEL.to_owned(),
            lock_sha256: None,
            state: ModelState::Attached,
            instance_id: Some(STUB_INSTANCE_ID.to_owned()),
            model_generation: generation,
        }],
        artifacts: Vec::new(),
    }
}

impl StubAgent {
    /// Serves `worker_id@epoch` (generation 1) on a loopback port with a leaf for `principal` under `fed`'s CA,
    /// authorizing only the coordinator principal.
    pub async fn start(
        fed: &Federation,
        worker_id: &str,
        principal: &str,
        epoch: u64,
    ) -> StubAgent {
        let dir = StateDir::new();
        std::fs::create_dir_all(dir.path()).unwrap();
        let ca = fed.pki.write_ca(dir.path());
        let (cert, key) = fed.pki.leaf(principal).write(dir.path());
        let pki = Pki::load(&ca, &cert, &key, principal).unwrap();
        let inventory = inventory(worker_id, 1);
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                worker_id: worker_id.to_owned(),
                epoch,
                generation: 1,
                term: 0,
                reconciled: false,
                lease_until: None,
                draining: false,
                owner_only: false,
                inventory_bytes: inventory.to_json_bytes().unwrap(),
                inventory,
                reported_sha256: None,
                ops: BTreeMap::new(),
                next_ticket: 0,
                outcome: WorkOutcome::Complete,
                smoke_outcome: WorkOutcome::Complete,
                calls: Vec::new(),
                submits: Vec::new(),
                submit_attempts: Vec::new(),
                cancels: Vec::new(),
                backend: None,
                runs: BTreeMap::new(),
                refusals: VecDeque::new(),
                lost_replies: 0,
                accepts: Vec::new(),
                hold_cleanup: false,
                held: Vec::new(),
                sequence: 0,
                violation: None,
                lifecycle_answers: BTreeMap::new(),
                lifecycle_calls: Vec::new(),
            }),
            authz: Authz::agent_side(FED_COORDINATOR.to_owned()),
            quota: PrincipalQuota::new(64),
            list_gate: watch::Sender::new(true),
            submit_gate: watch::Sender::new(true),
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let service = Service {
            shared: shared.clone(),
        };
        let router = Server::builder()
            .tls_config(pki.server_tls())
            .unwrap()
            .add_service(
                WorkerServiceServer::new(service.clone())
                    .max_decoding_message_size(wire::CONTROL_MESSAGE_MAX),
            )
            .add_service(
                WorkerInferenceServiceServer::new(service)
                    .max_decoding_message_size(wire::INFERENCE_MESSAGE_MAX),
            );
        let (stop, stopped) = oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            let incoming = TcpListenerStream::new(listener);
            let _ = router
                .serve_with_incoming_shutdown(incoming, async {
                    let _ = stopped.await;
                })
                .await;
        });
        StubAgent {
            shared,
            addr,
            principal: principal.to_owned(),
            _dir: dir,
            stop: Some(stop),
            task: Some(task),
            link: None,
        }
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.shared.state()
    }

    /// Starts the agent's Register/Heartbeat loop against `fed`'s registry listener, one heartbeat per
    /// `interval`. A refused heartbeat registers again.
    pub fn link(&mut self, fed: &Federation, interval: Duration) {
        let mut client =
            WorkerRegistryServiceClient::new(fed.channel(fed.agent_tls(&self.principal)));
        let shared = self.shared.clone();
        self.link = Some(tokio::spawn(async move {
            let mut registered = false;
            loop {
                let (worker, reported, generation, admission, term, sequence, owner_only) = {
                    let mut st = shared.state();
                    if registered {
                        st.sequence += 1;
                    }
                    let worker = pb::WorkerIdentity {
                        cluster_id: FED_CLUSTER.to_owned(),
                        worker_id: st.worker_id.clone(),
                        worker_epoch: st.epoch,
                    };
                    (
                        worker,
                        st.reported(),
                        st.generation,
                        st.admission(),
                        st.term,
                        st.sequence,
                        st.owner_only,
                    )
                };
                if !registered {
                    let reply = client
                        .register(pb::RegisterRequest {
                            proposed: Some(worker),
                            boot_id: "00112233445566778899aabbccddeeff".to_owned(),
                            protocol_revision: cere_proto::PROTOCOL_REVISION,
                            inventory_sha256: reported.to_vec(),
                        })
                        .await;
                    if let Ok(reply) = reply.map(Response::into_inner) {
                        if reply.accepted {
                            let mut st = shared.state();
                            if reply.coordinator_term > st.term {
                                st.fence(reply.coordinator_term);
                            }
                            registered = true;
                        }
                    }
                } else {
                    let reply = client
                        .heartbeat(pb::HeartbeatRequest {
                            worker: Some(worker),
                            sequence,
                            observed_coordinator_term: term,
                            inventory_sha256: reported.to_vec(),
                            model_generation: generation,
                            admission_state: admission.as_str().to_owned(),
                            owner_only,
                        })
                        .await;
                    if let Ok(reply) = reply.map(Response::into_inner) {
                        let mut st = shared.state();
                        if reply.accepted {
                            st.lease_until = Some(
                                Instant::now() + Duration::from_millis(reply.admission_lease_ms),
                            );
                            st.draining = reply.desired_state == wire::DESIRED_DRAINING;
                            st.reconciled = reply.reconciled_generation == Some(generation)
                                && generation == st.generation;
                        } else {
                            st.reconciled = false;
                            registered = false;
                        }
                    }
                }
                tokio::time::sleep(interval).await;
            }
        }));
    }

    /// The inventory currently described.
    pub fn inventory(&self) -> Inventory {
        self.state().inventory.clone()
    }

    /// The exact bytes `DescribeResources` answers.
    pub fn inventory_bytes(&self) -> Vec<u8> {
        self.state().inventory_bytes.clone()
    }

    /// Describes `inventory` from now on (and reports its hash).
    pub fn set_inventory(&self, inventory: Inventory) {
        self.state().describe(inventory);
    }

    /// Describes the managed model (m36 design note Q7) in `state` under `lock` from now on; the identity fields
    /// name the current backend instance (or [`STUB_INSTANCE_ID`]) in `ready`/`draining`/`unloading`.
    pub fn set_model(&self, state: ModelState, lock: &str) {
        let mut st = self.state();
        let instance = st.instance_id();
        st.set_model(state, Some(lock.to_owned()), &instance);
    }

    /// The next fresh `method` call (a key not journaled yet, past the fences) answers `answer`; queued per
    /// method, [`Lifecycle::Receipt`] once the queue is empty.
    pub fn script_lifecycle(&self, method: &'static str, answer: Lifecycle) {
        self.state()
            .lifecycle_answers
            .entry(method)
            .or_default()
            .push_back(answer);
    }

    /// Every lifecycle call past authorization, in arrival order (journaled-key retries included).
    pub fn lifecycle_calls(&self) -> Vec<LifecycleCall> {
        self.state().lifecycle_calls.clone()
    }

    /// Ends the journaled op `scope_id/operation_id` with `reason`, cleanup-acknowledged.
    pub fn end_op(&self, scope_id: &str, operation_id: &str, reason: &str) {
        let mut st = self.state();
        let op = st
            .ops
            .get_mut(&(scope_id.to_owned(), operation_id.to_owned()))
            .expect("a journaled op");
        op.terminal = true;
        op.reason = reason.to_owned();
        op.cleanup_acknowledged = true;
    }

    /// Reports `sha256` in Register and heartbeats instead of the described bytes' hash (`None` = the real one).
    pub fn report_sha256(&self, sha256: Option<[u8; 32]>) {
        self.state().reported_sha256 = sha256;
    }

    /// Reports `on` as the heartbeat's `owner_only` from the next heartbeat on (team pools, C19).
    pub fn set_owner_only(&self, on: bool) {
        self.state().owner_only = on;
    }

    /// A new model generation (Q5): admission closes, the live SubmitWork/CancelWork ops of the old generation
    /// end `model_generation_superseded`, and the inventory names the generation.
    pub fn set_generation(&self, generation: u64) {
        self.state().set_generation(generation);
    }

    pub fn generation(&self) -> u64 {
        self.state().generation
    }

    /// The highest coordinator term the stub accepted.
    pub fn term(&self) -> u64 {
        self.state().term
    }

    /// A new incarnation at `epoch` (an agent restart): the next heartbeat is refused and the link registers
    /// again; the lease and reconciliation of the old incarnation are gone.
    pub fn restart(&self, epoch: u64) {
        let mut st = self.state();
        st.epoch = epoch;
        st.reconciled = false;
        st.lease_until = None;
        st.draining = false;
        st.sequence = 0;
    }

    pub fn admission(&self) -> AdmissionState {
        self.state().admission()
    }

    /// Journals `op` as if an earlier coordinator had submitted it.
    pub fn add_op(&self, op: StubOp) {
        let key = (op.scope_id.clone(), op.operation_id.clone());
        self.state().ops.insert(key, op);
    }

    pub fn op(&self, scope_id: &str, operation_id: &str) -> Option<StubOp> {
        self.state()
            .ops
            .get(&(scope_id.to_owned(), operation_id.to_owned()))
            .cloned()
    }

    /// How the next `StreamWork` of a live op ends, the readiness smoke's excepted ([`Self::set_smoke_outcome`]).
    pub fn set_outcome(&self, outcome: WorkOutcome) {
        self.state().outcome = outcome;
    }

    /// How the next `StreamWork` of a live readiness smoke op ([`REMOTE_SMOKE_SCOPE`]) ends.
    pub fn set_smoke_outcome(&self, outcome: WorkOutcome) {
        self.state().smoke_outcome = outcome;
    }

    /// `true` holds every `ListOperations` until `hold_list(false)`.
    pub fn hold_list(&self, hold: bool) {
        self.shared.list_gate.send_replace(!hold);
    }

    pub fn calls(&self) -> Calls {
        self.state().calls.clone()
    }

    pub fn count(&self, method: &str) -> usize {
        self.state().calls.iter().filter(|m| **m == method).count()
    }

    pub fn submits(&self) -> Vec<pb::SubmitWorkRequest> {
        self.state().submits.clone()
    }

    /// Every SubmitWork past the digest check: refused, accepted and retried ones, in arrival order.
    pub fn submit_attempts(&self) -> Vec<pb::SubmitWorkRequest> {
        self.state().submit_attempts.clone()
    }

    pub fn cancels(&self) -> Vec<pb::CancelWorkRequest> {
        self.state().cancels.clone()
    }

    /// Every following accepted Submit runs against the SGLang backend at `base_url` (an in-process
    /// `cere-mock-sglang`), streaming its output as work events.
    pub fn serve_chat(&self, base_url: &str) {
        let timeouts = Timeouts {
            connect: Duration::from_secs(1),
            control: Duration::from_secs(2),
            stream_idle: Duration::from_secs(10),
        };
        self.state().backend = Some(SglangHttp::new(base_url, timeouts).unwrap());
    }

    /// The next SubmitWork that passes authorization and validation is refused `code reason`.
    pub fn refuse_next_submit(&self, code: Code, reason: &'static str) {
        self.state().refusals.push_back((code, reason));
    }

    /// The next `n` SubmitWork calls the journal answers lose their reply (the connection resets after the journal
    /// write).
    pub fn lose_submit_replies(&self, n: usize) {
        self.state().lost_replies = n;
    }

    /// `true` holds the reply of every following fresh SubmitWork accept (the op is journaled and its run started)
    /// until `hold_submit_replies(false)`.
    pub fn hold_submit_replies(&self, hold: bool) {
        self.shared.submit_gate.send_replace(!hold);
    }

    /// Every SubmitWork the journal answered, in order: request id and whether it was a fresh accept.
    pub fn accepts(&self) -> Vec<(String, bool)> {
        self.state().accepts.clone()
    }

    /// `true` keeps every run that ends from now on unacknowledged; `false` acknowledges the held ones.
    pub fn hold_cleanup(&self, hold: bool) {
        let mut st = self.state();
        st.hold_cleanup = hold;
        if !hold {
            for key in std::mem::take(&mut st.held) {
                if let Some(op) = st.ops.get_mut(&key) {
                    op.cleanup_acknowledged = true;
                }
            }
        }
    }

    /// The next served `StreamWork` commits `violation` after its first event.
    pub fn set_violation(&self, violation: Violation) {
        self.state().violation = Some(violation);
    }

    /// Stops the Register/Heartbeat loop (the agent goes silent; its lease runs out).
    pub fn unlink(&mut self) {
        if let Some(link) = self.link.take() {
            link.abort();
        }
    }
}

impl Drop for StubAgent {
    fn drop(&mut self) {
        if let Some(link) = self.link.take() {
            link.abort();
        }
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}
