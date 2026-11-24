//! The coordinator's client of one agent's `WorkerService` and `WorkerInferenceService` (m35 design note Q8,
//! Q12, Q14; build-loop-m35 design note "Coordinator (CREG -> CREM)", D1, D4).
//!
//! One lazily connected mTLS channel per remote worker: `https://<endpoint>` with `Pki::client_tls(principal)`,
//! HTTP/2 keepalive every heartbeat interval with the RPC timeout as its timeout, also while idle. Every call
//! carries an [`Authority`](pb::Authority) built from an [`AuthorityBase`]: reads (DescribeResources,
//! ListOperations, GetOperation) name only the target, the coordinator and its term; operations (SubmitWork,
//! CancelWork, StreamWork) add the scope, the operation id, the Q12 digest, the remaining budget and the agent's
//! heartbeat generation. Refusals come back as the agent's `tonic::Status` (reason = message, `wire::reason`).

use std::time::Duration;

use cere_proto::worker::v2 as pb;
use pb::worker_inference_service_client::WorkerInferenceServiceClient;
use pb::worker_service_client::WorkerServiceClient;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Status, Streaming};

use crate::config::FederationConfig;
use crate::fed::tls::{FedError, Pki};
use crate::fed::wire;

/// The channel timings (build-loop-m35 design note "Wire behaviour" Channels).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClientTiming {
    /// HTTP/2 keepalive interval (= the heartbeat interval).
    pub heartbeat_interval: Duration,
    /// Connect timeout, keepalive timeout and the deadline of every unary call.
    pub rpc_timeout: Duration,
}

impl ClientTiming {
    pub fn from_config(fed: &FederationConfig) -> ClientTiming {
        ClientTiming {
            heartbeat_interval: fed.heartbeat_interval(),
            rpc_timeout: fed.rpc_timeout(),
        }
    }
}

/// Who calls whom: the agent's registered incarnation under this coordinator's term (creg-notes "AGENT wire" 2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorityBase {
    pub cluster_id: String,
    pub worker_id: String,
    pub worker_epoch: u64,
    pub coordinator_id: String,
    pub coordinator_term: u64,
    /// The agent's generation from its last accepted heartbeat; operations carry it (`None` = no heartbeat yet,
    /// which the agent refuses as `stale_generation`).
    pub model_generation: Option<u64>,
}

impl AuthorityBase {
    /// The read form: target, coordinator id and term only.
    pub fn read(&self) -> pb::Authority {
        pb::Authority {
            target: Some(pb::WorkerIdentity {
                cluster_id: self.cluster_id.clone(),
                worker_id: self.worker_id.clone(),
                worker_epoch: self.worker_epoch,
            }),
            coordinator_id: self.coordinator_id.clone(),
            coordinator_term: self.coordinator_term,
            ..Default::default()
        }
    }

    /// The operation form: the read form plus scope, operation id, digest, budget and generation.
    pub fn op(
        &self,
        scope_id: &str,
        operation_id: &str,
        payload_sha256: Vec<u8>,
        remaining_budget_ms: u64,
    ) -> pb::Authority {
        pb::Authority {
            scope_id: scope_id.to_owned(),
            operation_id: operation_id.to_owned(),
            payload_sha256,
            remaining_budget_ms,
            model_generation: self.model_generation,
            ..self.read()
        }
    }
}

/// One submitted unit of work as the agent journals it (Q12): the operation key, the request it serves and the
/// two digested byte strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Work {
    pub scope_id: String,
    pub operation_id: String,
    pub request_id: String,
    pub attempt: u64,
    pub work_unit_json: Vec<u8>,
    pub input_json: Vec<u8>,
}

/// The Submit operation a cancel or stream names: its scope, request, attempt and completion ticket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpRef {
    pub scope_id: String,
    pub request_id: String,
    pub attempt: u64,
    pub completion_ticket: String,
}

/// A connected (lazily) client of one agent; clones share the channel.
#[derive(Debug, Clone)]
pub struct WorkerClient {
    control: WorkerServiceClient<Channel>,
    inference: WorkerInferenceServiceClient<Channel>,
    rpc_timeout: Duration,
}

impl WorkerClient {
    /// A client of the agent at `endpoint` (the configured `https://host:port`) whose server certificate must
    /// name `peer_principal`. Connects on the first call; must run inside a tokio runtime.
    pub fn connect(
        endpoint: &str,
        pki: &Pki,
        peer_principal: &str,
        timing: ClientTiming,
    ) -> Result<WorkerClient, FedError> {
        let channel_error = |reason: String| FedError::Channel {
            endpoint: endpoint.to_owned(),
            reason,
        };
        let channel = Endpoint::from_shared(endpoint.to_owned())
            .map_err(|e| channel_error(e.to_string()))?
            .connect_timeout(timing.rpc_timeout)
            .http2_keep_alive_interval(timing.heartbeat_interval)
            .keep_alive_timeout(timing.rpc_timeout)
            .keep_alive_while_idle(true)
            .tls_config(pki.client_tls(peer_principal))
            .map_err(|e| channel_error(e.to_string()))?
            .connect_lazy();
        Ok(WorkerClient {
            control: WorkerServiceClient::new(channel.clone())
                .max_decoding_message_size(wire::CONTROL_MESSAGE_MAX)
                .max_encoding_message_size(wire::CONTROL_MESSAGE_MAX),
            inference: WorkerInferenceServiceClient::new(channel)
                .max_decoding_message_size(wire::INFERENCE_MESSAGE_MAX)
                .max_encoding_message_size(wire::INFERENCE_MESSAGE_MAX),
            rpc_timeout: timing.rpc_timeout,
        })
    }

    fn unary<T>(&self, message: T) -> Request<T> {
        let mut request = Request::new(message);
        request.set_timeout(self.rpc_timeout);
        request
    }

    /// `DescribeResources`; the reply holds the exact inventory bytes (hash them, never a re-encoding).
    pub async fn describe(&self, at: &AuthorityBase) -> Result<pb::InventoryReply, Status> {
        let request = self.unary(pb::InventoryRequest {
            authority: Some(at.read()),
        });
        Ok(self
            .control
            .clone()
            .describe_resources(request)
            .await?
            .into_inner())
    }

    /// Every operation the agent holds, following `ListOperations` pages of [`wire::LIST_PAGE_MAX`] by the
    /// `(scope_id, operation_id)` cursor of each page's last entry.
    pub async fn list_all(&self, at: &AuthorityBase) -> Result<Vec<pb::OperationStatus>, Status> {
        let mut all = Vec::new();
        let mut after = (String::new(), String::new());
        loop {
            let request = self.unary(pb::ListOperationsRequest {
                authority: Some(at.read()),
                after_scope_id: after.0.clone(),
                after_operation_id: after.1.clone(),
                limit: wire::LIST_PAGE_MAX as u32,
            });
            let page = self
                .control
                .clone()
                .list_operations(request)
                .await?
                .into_inner();
            let more = page.more;
            if page.operations.len() > wire::LIST_PAGE_MAX {
                return Err(Status::out_of_range("ListOperations page over the limit"));
            }
            if let Some(last) = page.operations.last() {
                after = (last.scope_id.clone(), last.operation_id.clone());
            } else if more {
                return Err(Status::internal("ListOperations: an empty page with more"));
            }
            all.extend(page.operations);
            if !more {
                return Ok(all);
            }
        }
    }

    /// `GetOperation` of `queried_operation_id` in `scope_id` (empty: the unique op with that id).
    pub async fn get_operation(
        &self,
        at: &AuthorityBase,
        scope_id: &str,
        queried_operation_id: &str,
    ) -> Result<pb::OperationStatus, Status> {
        let mut authority = at.read();
        authority.scope_id = scope_id.to_owned();
        let request = self.unary(pb::OperationQuery {
            authority: Some(authority),
            queried_operation_id: queried_operation_id.to_owned(),
        });
        Ok(self
            .control
            .clone()
            .get_operation(request)
            .await?
            .into_inner())
    }

    /// `SubmitWork` of `work` (Q12): the RPC deadline is min(rpc timeout, `remaining_budget_ms`); the budget
    /// rides in the authority. A retry with the same `work` returns the original receipt.
    pub async fn submit(
        &self,
        at: &AuthorityBase,
        work: &Work,
        remaining_budget_ms: u64,
    ) -> Result<pb::OperationReceipt, Status> {
        let digest = wire::payload_sha256(
            wire::METHOD_SUBMIT,
            &work.scope_id,
            &work.request_id,
            work.attempt,
            &work.work_unit_json,
            &work.input_json,
        );
        let mut request = Request::new(pb::SubmitWorkRequest {
            authority: Some(at.op(
                &work.scope_id,
                &work.operation_id,
                digest.to_vec(),
                remaining_budget_ms,
            )),
            request_id: work.request_id.clone(),
            attempt: work.attempt,
            work_unit_json: work.work_unit_json.clone(),
            bounded_inference_input_json: work.input_json.clone(),
        });
        request.set_timeout(
            self.rpc_timeout
                .min(Duration::from_millis(remaining_budget_ms)),
        );
        Ok(self
            .inference
            .clone()
            .submit_work(request)
            .await?
            .into_inner())
    }

    /// `CancelWork` of `target` under the cancel operation id `operation_id` (Q12: `<rid>/cancel/<n>`).
    pub async fn cancel(
        &self,
        at: &AuthorityBase,
        operation_id: &str,
        target: &OpRef,
    ) -> Result<pb::OperationReceipt, Status> {
        let digest = wire::payload_sha256(
            wire::METHOD_CANCEL,
            &target.scope_id,
            &target.request_id,
            target.attempt,
            b"",
            target.completion_ticket.as_bytes(),
        );
        let budget_ms = self.rpc_timeout.as_millis() as u64;
        let request = self.unary(pb::CancelWorkRequest {
            authority: Some(at.op(&target.scope_id, operation_id, digest.to_vec(), budget_ms)),
            request_id: target.request_id.clone(),
            attempt: target.attempt,
            completion_ticket: target.completion_ticket.clone(),
        });
        Ok(self
            .control
            .clone()
            .cancel_work(request)
            .await?
            .into_inner())
    }

    /// `LoadModel`/`DrainModel`/`UnloadModel` (`method`, m36 design note Q6, Q8) under `authority` (built once
    /// per call and reused unchanged on a retry, so a journaled key answers its receipt): the coordinator's
    /// deployment `lock`, no instance id, no device ids.
    pub async fn lifecycle(
        &self,
        method: &'static str,
        authority: pb::Authority,
        lock: &[u8; 32],
    ) -> Result<pb::OperationReceipt, Status> {
        let request = self.unary(pb::ModelLifecycleRequest {
            authority: Some(authority),
            deployment_lock_sha256: lock.to_vec(),
            instance_id: String::new(),
            device_ids: Vec::new(),
        });
        let mut control = self.control.clone();
        let reply = match method {
            wire::METHOD_LOAD => control.load_model(request).await?,
            wire::METHOD_DRAIN => control.drain_model(request).await?,
            wire::METHOD_UNLOAD => control.unload_model(request).await?,
            other => {
                return Err(Status::invalid_argument(format!(
                    "{other} is not a lifecycle method"
                )))
            }
        };
        Ok(reply.into_inner())
    }

    /// `StreamWork` of the Submit operation `operation_id` from `after_sequence` (D4): the stream's deadline is
    /// `remaining_budget_ms` capped at [`wire::WORK_CAP_MS`]; liveness comes from the keepalive, not the RPC
    /// timeout.
    pub async fn stream(
        &self,
        at: &AuthorityBase,
        operation_id: &str,
        target: &OpRef,
        after_sequence: u64,
        remaining_budget_ms: u64,
    ) -> Result<Streaming<pb::WorkEvent>, Status> {
        let budget_ms = remaining_budget_ms.min(wire::WORK_CAP_MS);
        let mut request = Request::new(pb::WorkStreamRequest {
            authority: Some(at.op(&target.scope_id, operation_id, Vec::new(), budget_ms)),
            request_id: target.request_id.clone(),
            attempt: target.attempt,
            completion_ticket: target.completion_ticket.clone(),
            after_sequence,
        });
        request.set_timeout(Duration::from_millis(budget_ms));
        Ok(self
            .control
            .clone()
            .stream_work(request)
            .await?
            .into_inner())
    }
}
