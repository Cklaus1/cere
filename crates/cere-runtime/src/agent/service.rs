//! The agent's `WorkerService` and `WorkerInferenceService` handlers (m35 design note Q8, Q11, Q12, Q14, H1, H3).
//! Track WORK (build-loop-m35 design note).
//!
//! Order per call: `tls::Authz` (the coordinator principal) -> the per-principal pool (unary calls: the control
//! quota; `StreamWork`: the stream pool, live-op bound + 1, H1) -> [`Shared::authorize`] -> message validation
//! (`INVALID_ARGUMENT malformed_request`, H3(c)) -> admission (refused only when poisoned: `UNAVAILABLE poisoned`)
//! -> the recomputed `payload_sha256` (`INVALID_ARGUMENT payload_digest_mismatch`) -> the op journal and the
//! operation runtime ([`crate::agent::ops`]). A journaled SubmitWork key is answered by the journal in any other
//! admission state; a new key needs `open` (`UNAVAILABLE lease_expired` while fenced), checked by
//! [`Ops::submit`] under the journal lock. `DescribeResources` before a confirmed backend identity is
//! `UNAVAILABLE backend_unconfirmed` in attached mode; a managed agent always answers (m36 design note Q7).
//! `LoadModel`, `DrainModel` and `UnloadModel` (m36 design note Q6, design-m36-agent.md §3.3): the control
//! quota -> shape (`malformed_request`) and target -> admission -> attached mode `FAILED_PRECONDITION
//! model_unmanaged` -> the recomputed `wire::lifecycle_sha256` -> [`Shared::lifecycle`] (journal lookup, fences,
//! lock/placement/instance checks, the state table). `StreamTelemetry` is `UNIMPLEMENTED`.

use std::sync::Arc;

use cere_proto::worker::v2 as pb;
use pb::worker_inference_service_server::WorkerInferenceService;
use pb::worker_service_server::WorkerService;
use serde_json::value::RawValue;
use serde_json::Value;
use tonic::{Code, Request, Response, Status};

use crate::agent::journal::{Accepted, OpIdentity, OpKey, OpView};
use crate::agent::node::Shared;
use crate::agent::ops::{lock, NewCancel, NewSubmit, Ops, RpcStream, StreamPool};
use crate::fed::quota::QuotaGuard;
use crate::fed::tls::Authz;
use crate::fed::wire::{self, AdmissionState};
use crate::sglang::Rid;

fn malformed() -> Status {
    wire::status(Code::InvalidArgument, wire::MALFORMED_REQUEST)
}

/// The authority of an authorized call (`Shared::authorize` already refused a missing one).
fn authority(authority: Option<&pb::Authority>) -> Result<&pb::Authority, Status> {
    authority.ok_or_else(malformed)
}

/// The operation key of an op call; an empty scope or operation id is malformed.
fn op_key(authority: &pb::Authority) -> Result<OpKey, Status> {
    if authority.scope_id.is_empty() || authority.operation_id.is_empty() {
        return Err(malformed());
    }
    Ok(OpKey {
        scope_id: authority.scope_id.clone(),
        operation_id: authority.operation_id.clone(),
    })
}

/// `payload_sha256` must be exactly 32 bytes.
fn claimed_digest(authority: &pb::Authority) -> Result<[u8; 32], Status> {
    authority
        .payload_sha256
        .as_slice()
        .try_into()
        .map_err(|_| malformed())
}

fn check_digest(claimed: &[u8; 32], computed: &[u8; 32]) -> Result<(), Status> {
    if claimed != computed {
        return Err(wire::status(
            Code::InvalidArgument,
            wire::PAYLOAD_DIGEST_MISMATCH,
        ));
    }
    Ok(())
}

fn identity(method: &'static str, authority: &pb::Authority, digest: [u8; 32]) -> OpIdentity {
    OpIdentity {
        method,
        payload_sha256: digest,
        worker_epoch: authority
            .target
            .as_ref()
            .map_or(0, |target| target.worker_epoch),
        model_generation: authority.model_generation.unwrap_or_default(),
        coordinator_term: authority.coordinator_term,
    }
}

fn receipt(key: &OpKey, accepted: Accepted) -> pb::OperationReceipt {
    pb::OperationReceipt {
        accepted: true,
        operation_id: key.operation_id.clone(),
        completion_ticket: accepted.ticket,
        reason_code: String::new(),
    }
}

/// `OperationStatus` per the proto: `phase` `accepted` (nonterminal), `terminal` (not cleanup-acknowledged) or
/// `released`.
fn status_of(view: OpView) -> pb::OperationStatus {
    let phase = match (view.terminal, view.cleanup_acknowledged) {
        (false, _) => wire::PHASE_ACCEPTED,
        (true, false) => wire::PHASE_TERMINAL,
        (true, true) => wire::PHASE_RELEASED,
    };
    pb::OperationStatus {
        operation_id: view.key.operation_id,
        phase: phase.to_owned(),
        terminal: view.terminal,
        cleanup_acknowledged: view.cleanup_acknowledged,
        model_generation: view.model_generation,
        completion_ticket: view.ticket,
        reason_code: view.reason,
        coordinator_term: view.coordinator_term,
        worker_epoch: view.worker_epoch,
        scope_id: view.key.scope_id,
        request_id: view.request_id,
        attempt: view.attempt,
        method: view.method.to_owned(),
    }
}

/// The listener's handlers (one value per service, sharing the node state, the op runtime and the stream pool).
#[derive(Clone)]
pub(crate) struct WorkerApi {
    shared: Arc<Shared>,
    ops: Arc<Ops>,
    authz: Authz,
    streams: StreamPool,
}

impl WorkerApi {
    pub(crate) fn new(shared: Arc<Shared>, ops: Arc<Ops>) -> WorkerApi {
        WorkerApi {
            authz: Authz::agent_side(shared.config.coordinator.principal.clone()),
            shared,
            ops,
            streams: StreamPool::default(),
        }
    }

    fn unimplemented<T, R>(&self, req: &Request<T>) -> Result<R, Status> {
        self.authz.coordinator_caller(req)?;
        Err(Status::unimplemented(""))
    }

    /// Authz, then a slot of the caller's unary control pool (held for the call).
    fn control<T>(&self, req: &Request<T>) -> Result<QuotaGuard, Status> {
        let principal = self.authz.coordinator_caller(req)?;
        self.shared.quota.try_acquire(&principal)
    }

    /// Q7a: every call is served unless the agent is poisoned; a new SubmitWork key also needs `open`
    /// ([`Ops::submit`]).
    fn admit(&self) -> Result<(), Status> {
        match self.shared.admission_state() {
            AdmissionState::Poisoned => Err(wire::status(Code::Unavailable, wire::POISONED)),
            AdmissionState::Open | AdmissionState::Fenced => Ok(()),
        }
    }

    /// One entry for `LoadModel`/`DrainModel`/`UnloadModel` (design-m36-agent.md §3.3, steps 1-4 here).
    fn lifecycle(
        &self,
        method: &'static str,
        req: Request<pb::ModelLifecycleRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        let _slot = self.control(&req)?;
        let req = req.into_inner();
        let auth = authority(req.authority.as_ref())?;
        if auth.target.is_none() || req.deployment_lock_sha256.len() != 32 {
            return Err(malformed());
        }
        let claimed = claimed_digest(auth)?;
        let key = op_key(auth)?;
        self.shared.check_target(Some(auth))?;
        self.admit()?;
        if !self.shared.managed() {
            return Err(wire::status(
                Code::FailedPrecondition,
                wire::MODEL_UNMANAGED,
            ));
        }
        let computed = wire::lifecycle_sha256(
            method,
            &key.scope_id,
            &req.deployment_lock_sha256,
            &req.instance_id,
            &req.device_ids,
        );
        check_digest(&claimed, &computed)?;
        let accepted =
            self.shared
                .lifecycle(&req, key.clone(), identity(method, auth, computed))?;
        Ok(Response::new(receipt(&key, accepted)))
    }
}

#[tonic::async_trait]
impl WorkerService for WorkerApi {
    async fn describe_resources(
        &self,
        req: Request<pb::InventoryRequest>,
    ) -> Result<Response<pb::InventoryReply>, Status> {
        let _slot = self.control(&req)?;
        self.shared
            .authorize(req.get_ref().authority.as_ref(), false)?;
        let doc = self
            .shared
            .inventory()
            .ok_or_else(|| wire::status(Code::Unavailable, wire::BACKEND_UNCONFIRMED))?;
        Ok(Response::new(pb::InventoryReply {
            bounded_resource_inventory_json: doc.bytes.clone(),
            inventory_generation: doc.generation,
        }))
    }

    async fn load_model(
        &self,
        req: Request<pb::ModelLifecycleRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.lifecycle(wire::METHOD_LOAD, req)
    }

    async fn drain_model(
        &self,
        req: Request<pb::ModelLifecycleRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.lifecycle(wire::METHOD_DRAIN, req)
    }

    async fn unload_model(
        &self,
        req: Request<pb::ModelLifecycleRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        self.lifecycle(wire::METHOD_UNLOAD, req)
    }

    /// CancelWork under its own operation id (digest over `completion_ticket`); served while fenced.
    async fn cancel_work(
        &self,
        req: Request<pb::CancelWorkRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        let _slot = self.control(&req)?;
        let req = req.into_inner();
        self.shared.authorize(req.authority.as_ref(), true)?;
        let authority = authority(req.authority.as_ref())?;
        let claimed = claimed_digest(authority)?;
        let key = op_key(authority)?;
        if req.request_id.is_empty() {
            return Err(malformed());
        }
        self.admit()?;
        let computed = wire::payload_sha256(
            wire::METHOD_CANCEL,
            &key.scope_id,
            &req.request_id,
            req.attempt,
            b"",
            req.completion_ticket.as_bytes(),
        );
        check_digest(&claimed, &computed)?;
        let accepted = self.ops.cancel(NewCancel {
            key: key.clone(),
            identity: identity(wire::METHOD_CANCEL, authority, computed),
            request_id: req.request_id.clone(),
            attempt: req.attempt,
            ticket: req.completion_ticket.clone(),
            remaining_budget_ms: authority.remaining_budget_ms,
        })?;
        Ok(Response::new(receipt(&key, accepted)))
    }

    /// The op (`authority.scope_id`, `queried_operation_id`); an empty scope names the unique op with that id.
    async fn get_operation(
        &self,
        req: Request<pb::OperationQuery>,
    ) -> Result<Response<pb::OperationStatus>, Status> {
        let _slot = self.control(&req)?;
        let req = req.into_inner();
        self.shared.authorize(req.authority.as_ref(), false)?;
        let authority = authority(req.authority.as_ref())?;
        if req.queried_operation_id.is_empty() {
            return Err(malformed());
        }
        let unknown = || wire::status(Code::NotFound, wire::UNKNOWN_OPERATION);
        let journal = lock(&self.shared.journal);
        let view = if authority.scope_id.is_empty() {
            let (all, _) = journal.list(None, usize::MAX);
            let mut found = all
                .into_iter()
                .filter(|op| op.key.operation_id == req.queried_operation_id);
            let first = found.next().ok_or_else(unknown)?;
            if found.next().is_some() {
                return Err(malformed());
            }
            first
        } else {
            journal
                .get(&OpKey {
                    scope_id: authority.scope_id.clone(),
                    operation_id: req.queried_operation_id.clone(),
                })
                .ok_or_else(unknown)?
        };
        drop(journal);
        Ok(Response::new(status_of(view)))
    }

    /// At most `limit` (1..=256) statuses strictly after (`after_scope_id`, `after_operation_id`); an empty pair
    /// starts at the beginning.
    async fn list_operations(
        &self,
        req: Request<pb::ListOperationsRequest>,
    ) -> Result<Response<pb::ListOperationsReply>, Status> {
        let _slot = self.control(&req)?;
        let req = req.into_inner();
        self.shared.authorize(req.authority.as_ref(), false)?;
        let limit = req.limit as usize;
        if !(1..=wire::LIST_PAGE_MAX).contains(&limit) {
            return Err(malformed());
        }
        let after =
            (!req.after_scope_id.is_empty() || !req.after_operation_id.is_empty()).then(|| OpKey {
                scope_id: req.after_scope_id.clone(),
                operation_id: req.after_operation_id.clone(),
            });
        let (page, more) = lock(&self.shared.journal).list(after.as_ref(), limit);
        Ok(Response::new(pb::ListOperationsReply {
            operations: page.into_iter().map(status_of).collect(),
            more,
        }))
    }

    type StreamWorkStream = RpcStream<pb::WorkEvent>;

    /// Attaches to the SubmitWork op (`authority.scope_id`, `authority.operation_id`) whose request id, attempt and
    /// ticket match (else `ticket_mismatch`), replaying from `after_sequence`.
    async fn stream_work(
        &self,
        req: Request<pb::WorkStreamRequest>,
    ) -> Result<Response<Self::StreamWorkStream>, Status> {
        let principal = self.authz.coordinator_caller(&req)?;
        let bound = self
            .shared
            .identity()
            .map_or(0, |identity| identity.max_running_requests);
        let slot = self
            .streams
            .try_acquire(&principal, bound.saturating_add(1) as usize)?;
        let req = req.into_inner();
        self.shared.authorize(req.authority.as_ref(), true)?;
        let key = op_key(authority(req.authority.as_ref())?)?;
        let view = lock(&self.shared.journal)
            .get(&key)
            .filter(|op| op.method == wire::METHOD_SUBMIT)
            .ok_or_else(|| wire::status(Code::NotFound, wire::UNKNOWN_OPERATION))?;
        if view.ticket != req.completion_ticket
            || view.request_id != req.request_id
            || view.attempt != req.attempt
        {
            return Err(wire::status(
                Code::FailedPrecondition,
                wire::TICKET_MISMATCH,
            ));
        }
        let stream = self.ops.attach(&view, req.after_sequence, slot)?;
        Ok(Response::new(stream))
    }

    type StreamTelemetryStream = RpcStream<pb::TelemetryEnvelope>;

    async fn stream_telemetry(
        &self,
        req: Request<pb::TelemetryRequest>,
    ) -> Result<Response<Self::StreamTelemetryStream>, Status> {
        self.unimplemented(&req)
    }
}

#[tonic::async_trait]
impl WorkerInferenceService for WorkerApi {
    /// SubmitWork: `work_unit_json` = `{"kind":"chat","stream":<bool>}`; the body's `rid` must be the CERE rid
    /// `request_id`.
    async fn submit_work(
        &self,
        req: Request<pb::SubmitWorkRequest>,
    ) -> Result<Response<pb::OperationReceipt>, Status> {
        let _slot = self.control(&req)?;
        let req = req.into_inner();
        self.shared.authorize(req.authority.as_ref(), true)?;
        let authority = authority(req.authority.as_ref())?;
        let claimed = claimed_digest(authority)?;
        let key = op_key(authority)?;
        let rid = Rid::parse(&req.request_id).map_err(|_| malformed())?;
        let unit: Value = serde_json::from_slice(&req.work_unit_json).map_err(|_| malformed())?;
        let stream = match (
            unit.get("kind").and_then(Value::as_str),
            unit.get("stream").and_then(Value::as_bool),
        ) {
            (Some("chat"), Some(stream)) => stream,
            _ => return Err(malformed()),
        };
        let body: Value =
            serde_json::from_slice(&req.bounded_inference_input_json).map_err(|_| malformed())?;
        if body.get("rid").and_then(Value::as_str) != Some(req.request_id.as_str()) {
            return Err(malformed());
        }
        let tools = body.get("tools").is_some_and(|tools| !tools.is_null());
        // M2.4 D5: the backend gets these bytes, not a re-serialization (key order and number spelling of the
        // client's `tools` survive).
        let body: Box<RawValue> =
            serde_json::from_slice(&req.bounded_inference_input_json).map_err(|_| malformed())?;
        self.admit()?;
        let computed = wire::payload_sha256(
            wire::METHOD_SUBMIT,
            &key.scope_id,
            &req.request_id,
            req.attempt,
            &req.work_unit_json,
            &req.bounded_inference_input_json,
        );
        check_digest(&claimed, &computed)?;
        let accepted = self.ops.submit(NewSubmit {
            key: key.clone(),
            identity: identity(wire::METHOD_SUBMIT, authority, computed),
            request_id: req.request_id.clone(),
            rid,
            attempt: req.attempt,
            remaining_budget_ms: authority.remaining_budget_ms,
            body,
            stream,
            tools,
        })?;
        Ok(Response::new(receipt(&key, accepted)))
    }
}
