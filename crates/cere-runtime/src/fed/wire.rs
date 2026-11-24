//! Wire vocabulary of the worker RPC shared by coordinator and agent (m35 design note Q8, Q10a, Q12, Q14;
//! m36 design note Q6, Q7, Q12): reason codes, the status/reason error channel, the scoped payload digests,
//! admission and model states, the bounded resource inventory and the message limits.

use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// Reason codes. WorkerService/WorkerInferenceService refusals are gRPC statuses whose message is exactly one of
// these; registry refusals carry them in `reason_code` (Q8).
pub const STALE_TERM: &str = "stale_term";
pub const STALE_EPOCH: &str = "stale_epoch";
pub const STALE_GENERATION: &str = "stale_generation";
pub const WRONG_CLUSTER: &str = "wrong_cluster";
pub const WRONG_WORKER: &str = "wrong_worker";
pub const WRONG_COORDINATOR: &str = "wrong_coordinator";
pub const EPOCH_LOST: &str = "epoch_lost";
pub const UNKNOWN_WORKER: &str = "unknown_worker";
pub const UNSUPPORTED_REVISION: &str = "unsupported_revision";
pub const REPLAYED_HEARTBEAT: &str = "replayed_heartbeat";
pub const LEASE_EXPIRED: &str = "lease_expired";
pub const QUEUE_FULL: &str = "queue_full";
pub const JOURNAL_FULL: &str = "journal_full";
pub const POISONED: &str = "poisoned";
/// The agent has no confirmed backend identity yet, so it has no inventory to describe (`DescribeResources`).
pub const BACKEND_UNCONFIRMED: &str = "backend_unconfirmed";
pub const OPERATION_CONFLICT: &str = "operation_conflict";
pub const PAYLOAD_DIGEST_MISMATCH: &str = "payload_digest_mismatch";
pub const TICKET_MISMATCH: &str = "ticket_mismatch";
pub const UNKNOWN_OPERATION: &str = "unknown_operation";
pub const QUOTA_EXHAUSTED: &str = "quota_exhausted";
pub const PERMISSION_DENIED_REASON: &str = "permission_denied";
/// `tls::peer_principal`: no CA-verified peer certificate, or one that breaks the Q2 SAN rule.
pub const UNAUTHENTICATED_PEER: &str = "unauthenticated_peer";
pub const WORKER_RESTART: &str = "worker_restart";
pub const COORDINATOR_RESTART: &str = "coordinator_restart";
pub const MODEL_GENERATION_SUPERSEDED: &str = "model_generation_superseded";
pub const CLIENT_BACKPRESSURE: &str = "client_backpressure";
pub const CANCEL_REQUESTED: &str = "cancel_requested";
pub const DEADLINE_EXCEEDED: &str = "deadline_exceeded";
/// INVALID_ARGUMENT for a request that breaks its message contract: missing authority or target, `payload_sha256`
/// not 32 bytes, an unparsable `work_unit_json`, a body whose `rid` is not `request_id`, a list limit outside
/// 1..=[`LIST_PAGE_MAX`].
pub const MALFORMED_REQUEST: &str = "malformed_request";
/// `FAILED_PRECONDITION`: a lifecycle request names another lock than the agent's `[backend] lock_sha256`.
pub const LOCK_NOT_APPROVED: &str = "lock_not_approved";
/// `FAILED_PRECONDITION`: a lifecycle RPC to an agent without `[backend.launch]` (attached mode).
pub const MODEL_UNMANAGED: &str = "model_unmanaged";
/// `FAILED_PRECONDITION`, before the journal: a lifecycle call the model state cannot take now (Load in
/// `unloading`, Unload in `loading`, Drain outside `ready`/`draining`).
pub const MODEL_BUSY: &str = "model_busy";
/// `FAILED_PRECONDITION`: a lifecycle request with non-empty `device_ids`.
pub const UNSUPPORTED_PLACEMENT: &str = "unsupported_placement";
/// `UNAVAILABLE`: a new SubmitWork key while the model is `draining`.
pub const MODEL_DRAINING: &str = "model_draining";
/// `UNAVAILABLE`: a SubmitWork of a scope other than the worker's owner while it is owner-only (its local switch or
/// an `owner_only` window, net-teleport design note 4.4 C19).
pub const OWNER_ONLY: &str = "owner_only";
/// Terminal reason of a LoadModel whose backend exited, timed out or named another model before `ready`.
pub const MODEL_LOAD_FAILED: &str = "model_load_failed";
/// The supervised backend exited in `ready`/`draining`: terminal reason of pending DrainModel ops.
pub const BACKEND_EXITED: &str = "backend_exited";
/// Terminal reason of a LoadModel that found something already listening on the backend port.
pub const BACKEND_PORT_IN_USE: &str = "backend_port_in_use";

// `OperationStatus.phase` (Q8): accepted and not terminal; terminal and not cleanup-acknowledged; released.
pub const PHASE_ACCEPTED: &str = "accepted";
pub const PHASE_TERMINAL: &str = "terminal";
pub const PHASE_RELEASED: &str = "released";

/// Decode limit of `WorkerRegistryService` and `WorkerService` (Q14).
pub const CONTROL_MESSAGE_MAX: usize = 1 << 20;
/// Decode limit of `WorkerInferenceService` (Q14).
pub const INFERENCE_MESSAGE_MAX: usize = 8 << 20;
/// Most operation statuses in one `ListOperations` page (Q8).
pub const LIST_PAGE_MAX: usize = 256;
/// Largest `WorkEvent.bounded_output_event_json` (Q11).
pub const WORK_EVENT_MAX: usize = 64 << 10;
/// Largest `bounded_resource_inventory_json` (Q10a).
pub const INVENTORY_MAX: usize = 64 << 10;
/// Cap of the worker-local work deadline: receipt instant + min(remaining budget, this) (proto `Authority`).
pub const WORK_CAP_MS: u64 = 600_000;

/// Journaled methods; the first input of `payload_sha256` (Q12) or `lifecycle_sha256` (M3.6 Q6).
pub const METHOD_SUBMIT: &str = "SubmitWork";
pub const METHOD_CANCEL: &str = "CancelWork";
pub const METHOD_LOAD: &str = "LoadModel";
pub const METHOD_DRAIN: &str = "DrainModel";
pub const METHOD_UNLOAD: &str = "UnloadModel";

/// `HeartbeatReply.desired_state` of a draining worker (Q13).
pub const DESIRED_DRAINING: &str = "DRAINING";

/// A refusal on the status channel: `code` with the reason code as the whole message.
pub fn status(code: tonic::Code, reason: &str) -> tonic::Status {
    tonic::Status::new(code, reason)
}

/// The reason code of a refusal built by `status`.
pub fn reason(status: &tonic::Status) -> &str {
    status.message()
}

/// Q12 operation digest: SHA-256 over the u64 big-endian length-prefixed concatenation of method, scope_id,
/// request_id, attempt (decimal ASCII), work_unit_json and `tail` (`bounded_inference_input_json` for SubmitWork,
/// `completion_ticket` for CancelWork). Budget, term, epoch and generation are deliberately not inputs.
pub fn payload_sha256(
    method: &str,
    scope_id: &str,
    request_id: &str,
    attempt: u64,
    work_unit_json: &[u8],
    tail: &[u8],
) -> [u8; 32] {
    let attempt = attempt.to_string();
    length_prefixed_sha256([
        method.as_bytes(),
        scope_id.as_bytes(),
        request_id.as_bytes(),
        attempt.as_bytes(),
        work_unit_json,
        tail,
    ])
}

/// M3.6 Q6 lifecycle digest (`LoadModel`/`DrainModel`/`UnloadModel`): SHA-256 over the u64 big-endian
/// length-prefixed concatenation of method, scope_id, deployment_lock_sha256 (the raw bytes), instance_id and then
/// each device_id. Term, epoch and generation are not inputs.
pub fn lifecycle_sha256(
    method: &str,
    scope_id: &str,
    deployment_lock_sha256: &[u8],
    instance_id: &str,
    device_ids: &[String],
) -> [u8; 32] {
    length_prefixed_sha256(
        [
            method.as_bytes(),
            scope_id.as_bytes(),
            deployment_lock_sha256,
            instance_id.as_bytes(),
        ]
        .into_iter()
        .chain(device_ids.iter().map(String::as_bytes)),
    )
}

fn length_prefixed_sha256<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// Agent admission state reported in every heartbeat and in the inventory (Q7a).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AdmissionState {
    Open,
    Fenced,
    Poisoned,
}

impl AdmissionState {
    pub fn as_str(self) -> &'static str {
        match self {
            AdmissionState::Open => "open",
            AdmissionState::Fenced => "fenced",
            AdmissionState::Poisoned => "poisoned",
        }
    }

    pub fn parse(text: &str) -> Option<AdmissionState> {
        match text {
            "open" => Some(AdmissionState::Open),
            "fenced" => Some(AdmissionState::Fenced),
            "poisoned" => Some(AdmissionState::Poisoned),
            _ => None,
        }
    }
}

/// An entry type with no values: `gpus` and `artifacts` must be empty arrays (M3.5 Q10a; artifact locality is out
/// of scope in M3.6).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoEntry {}

/// The model state an agent describes (m36 design note Q4, Q7): `attached` for an agent without
/// `[backend.launch]`, the Q4 lifecycle states otherwise. Shared by agent and coordinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelState {
    Attached,
    Unloaded,
    Loading,
    Ready,
    Draining,
    Unloading,
    Failed,
}

impl ModelState {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelState::Attached => "attached",
            ModelState::Unloaded => "unloaded",
            ModelState::Loading => "loading",
            ModelState::Ready => "ready",
            ModelState::Draining => "draining",
            ModelState::Unloading => "unloading",
            ModelState::Failed => "failed",
        }
    }

    pub fn parse(text: &str) -> Option<ModelState> {
        match text {
            "attached" => Some(ModelState::Attached),
            "unloaded" => Some(ModelState::Unloaded),
            "loading" => Some(ModelState::Loading),
            "ready" => Some(ModelState::Ready),
            "draining" => Some(ModelState::Draining),
            "unloading" => Some(ModelState::Unloading),
            "failed" => Some(ModelState::Failed),
            _ => None,
        }
    }
}

/// A nullable field whose key must be present (serde would otherwise default a missing `Option` to `None`).
fn nullable<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

/// The worker's one model (Q7): its approved lock (`null` in attached mode), state, and the confirmed backend
/// instance (`null` while unconfirmed).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventoryModel {
    pub served_model_name: String,
    #[serde(deserialize_with = "nullable")]
    pub lock_sha256: Option<String>,
    pub state: ModelState,
    #[serde(deserialize_with = "nullable")]
    pub instance_id: Option<String>,
    pub model_generation: u64,
}

/// Inventory `schema_version` (M3.6 Q7; version 1 is refused).
pub const INVENTORY_SCHEMA_VERSION: u64 = 2;

/// `bounded_resource_inventory_json` (M3.6 Q7; `schemas/worker_inventory.schema.json`). The four backend identity
/// fields are all set or all null: set in attached mode and in `ready`, null in `unloaded`/`loading`/`failed`,
/// either in `draining`/`unloading`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inventory {
    pub schema_version: u64,
    pub worker_id: String,
    pub served_model_name: String,
    #[serde(deserialize_with = "nullable")]
    pub tp_size: Option<u64>,
    #[serde(deserialize_with = "nullable")]
    pub kv_cache_dtype: Option<String>,
    #[serde(deserialize_with = "nullable")]
    pub max_running_requests: Option<u64>,
    #[serde(deserialize_with = "nullable")]
    pub instance_id: Option<String>,
    pub model_generation: u64,
    pub admission_state: AdmissionState,
    pub gpus: Vec<NoEntry>,
    pub models: Vec<InventoryModel>,
    pub artifacts: Vec<NoEntry>,
}

/// Longest identifier string in an inventory (the schema's `maxLength`).
const INVENTORY_STRING_MAX: usize = 256;

#[derive(Debug)]
pub enum InventoryError {
    /// The encoded document exceeds `INVENTORY_MAX` bytes.
    TooLarge(usize),
    Json(serde_json::Error),
    Invalid(&'static str),
}

impl fmt::Display for InventoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InventoryError::TooLarge(len) => {
                write!(f, "inventory is {len} bytes, above {INVENTORY_MAX}")
            }
            InventoryError::Json(e) => write!(f, "inventory JSON: {e}"),
            InventoryError::Invalid(what) => write!(f, "inventory: {what}"),
        }
    }
}

impl std::error::Error for InventoryError {}

fn check_name(value: &str, what: &'static str) -> Result<(), InventoryError> {
    if value.is_empty() || value.len() > INVENTORY_STRING_MAX {
        return Err(InventoryError::Invalid(what));
    }
    Ok(())
}

/// 64 lowercase hex characters (a SHA-256 in the repository's lock notation).
fn is_lock_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl Inventory {
    /// The worker's one model; `None` only for an inventory that was never validated.
    pub fn model(&self) -> Option<&InventoryModel> {
        self.models.first()
    }

    fn validate(&self) -> Result<(), InventoryError> {
        if self.schema_version != INVENTORY_SCHEMA_VERSION {
            return Err(InventoryError::Invalid("schema_version must be 2"));
        }
        check_name(&self.worker_id, "worker_id must be 1..=256 bytes")?;
        check_name(
            &self.served_model_name,
            "served_model_name must be 1..=256 bytes",
        )?;
        let confirmed = match (
            self.tp_size,
            &self.kv_cache_dtype,
            self.max_running_requests,
            &self.instance_id,
        ) {
            (Some(tp_size), Some(kv_cache_dtype), Some(max_running_requests), Some(instance_id)) => {
                if tp_size == 0 {
                    return Err(InventoryError::Invalid("tp_size must be >= 1"));
                }
                if max_running_requests == 0 {
                    return Err(InventoryError::Invalid("max_running_requests must be >= 1"));
                }
                check_name(kv_cache_dtype, "kv_cache_dtype must be 1..=256 bytes")?;
                check_name(instance_id, "instance_id must be 1..=256 bytes")?;
                true
            }
            (None, None, None, None) => false,
            _ => {
                return Err(InventoryError::Invalid(
                    "tp_size, kv_cache_dtype, max_running_requests and instance_id must be all set or all null",
                ))
            }
        };
        let [model] = self.models.as_slice() else {
            return Err(InventoryError::Invalid("models must have exactly one item"));
        };
        if model.served_model_name != self.served_model_name
            || model.instance_id != self.instance_id
            || model.model_generation != self.model_generation
        {
            return Err(InventoryError::Invalid(
                "models[0] must repeat served_model_name, instance_id and model_generation",
            ));
        }
        match (model.state, &model.lock_sha256) {
            (ModelState::Attached, None) => {}
            (ModelState::Attached, Some(_)) => {
                return Err(InventoryError::Invalid(
                    "models[0].lock_sha256 must be null in attached mode",
                ))
            }
            (_, Some(lock)) if is_lock_hex(lock) => {}
            (_, _) => {
                return Err(InventoryError::Invalid(
                    "models[0].lock_sha256 must be 64 lowercase hex in managed mode",
                ))
            }
        }
        let identity_ok = match model.state {
            ModelState::Attached | ModelState::Ready => confirmed,
            ModelState::Unloaded | ModelState::Loading | ModelState::Failed => !confirmed,
            ModelState::Draining | ModelState::Unloading => true,
        };
        if !identity_ok {
            return Err(InventoryError::Invalid(
                "backend identity fields must be set exactly in attached/ready (either in draining/unloading)",
            ));
        }
        Ok(())
    }

    /// The exact bytes sent as `bounded_resource_inventory_json`; refused above `INVENTORY_MAX`.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, InventoryError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(InventoryError::Json)?;
        if bytes.len() > INVENTORY_MAX {
            return Err(InventoryError::TooLarge(bytes.len()));
        }
        Ok(bytes)
    }

    /// Parses a received inventory: size bound first, then unknown fields and value bounds.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Inventory, InventoryError> {
        if bytes.len() > INVENTORY_MAX {
            return Err(InventoryError::TooLarge(bytes.len()));
        }
        let inventory: Inventory = serde_json::from_slice(bytes).map_err(InventoryError::Json)?;
        inventory.validate()?;
        Ok(inventory)
    }

    /// `inventory_sha256`: SHA-256 of the exact inventory bytes as sent (never of a re-encoding).
    pub fn sha256(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }
}
