//! Public API shapes shared by validation and the server: the OpenAI-style error body, the `cere` response
//! extension (m1 design note A3, A4, S6) and the per-request usage record (m14 design note Q1, Q5).
//! HTTP-framework agnostic; statuses are plain `u16`.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::config::{RemoteWorkerConfig, RuntimeConfig};
use crate::digest::hex;
use crate::fed::registry::{Desired, ModelView, RemoteState};
use crate::fed::wire::{Inventory, ModelState};
use crate::journal::Node;
use crate::readiness::{Problem, ReadinessSnapshot, SmokeStatus};
use crate::sglang::BackendError;
use crate::worker::WorkerState;

pub const INVALID_REQUEST: &str = "invalid_request_error";
pub const SERVER_ERROR: &str = "server_error";

/// An API failure rendered as `{"error":{"message","type","code","param"}}` with HTTP `status`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiError {
    pub status: u16,
    pub error_type: &'static str,
    pub code: &'static str,
    pub message: String,
    /// Request field path (`messages[2].content`, `stream_options.foo`) when one field is at fault.
    pub param: Option<String>,
}

impl ApiError {
    pub fn new(
        status: u16,
        error_type: &'static str,
        code: &'static str,
        message: impl Into<String>,
        param: Option<String>,
    ) -> Self {
        Self {
            status,
            error_type,
            code,
            message: message.into(),
            param,
        }
    }

    /// 400: the field is valid OpenAI but outside the CERE subset (or unknown).
    pub fn unsupported_parameter(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(
            400,
            INVALID_REQUEST,
            "unsupported_parameter",
            message,
            Some(param.into()),
        )
    }

    /// 400: the field is in the subset but its value is not acceptable.
    pub fn invalid_value(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(
            400,
            INVALID_REQUEST,
            "invalid_value",
            message,
            Some(param.into()),
        )
    }

    /// 400: a required field is absent.
    pub fn missing_parameter(param: impl Into<String>) -> Self {
        let param = param.into();
        Self::new(
            400,
            INVALID_REQUEST,
            "missing_required_parameter",
            format!("missing required parameter '{param}'"),
            Some(param),
        )
    }

    /// 400: the body is not a JSON object.
    pub fn invalid_json(message: impl Into<String>) -> Self {
        Self::new(400, INVALID_REQUEST, "invalid_json", message, None)
    }

    /// 400 `invalid_value` on the request's constraint field (`tool_choice` for a tool constraint) when the
    /// backend could not compile its grammar (M2.8 D6): fixed text, never the backend's.
    pub fn grammar_compile(param: impl Into<String>) -> Self {
        Self::invalid_value(param, GRAMMAR_COMPILE_MESSAGE)
    }

    /// The body `{"error":{…}}`.
    pub fn body(&self) -> Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
                "param": self.param,
            }
        })
    }
}

/// The fixed text of a grammar compile refusal (M2.8 D6).
pub const GRAMMAR_COMPILE_MESSAGE: &str = "the request's grammar could not be compiled";

/// The `context_length_exceeded` refusal: prompt plus output budget above the session's context limit.
pub const CONTEXT_LENGTH_EXCEEDED: &str = "context_length_exceeded";

/// A prompt that does not fit the context with its output budget: the counts behind a native
/// [`CONTEXT_LENGTH_EXCEEDED`] or an SGLang `context_overflow` refusal, from which a compat surface words its own
/// error (ApiFit AF-L3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextExcess {
    /// Prompt tokens.
    pub prompt: u64,
    /// The request's output budget (`max_tokens`).
    pub max_tokens: u64,
    /// The context limit the request was held to.
    pub context: u64,
}

impl ContextExcess {
    /// The native refusal: 400 [`CONTEXT_LENGTH_EXCEEDED`] on `messages`.
    pub fn error(self) -> ApiError {
        ApiError::new(
            400,
            INVALID_REQUEST,
            CONTEXT_LENGTH_EXCEEDED,
            format!(
                "the prompt ({} tokens) plus max_tokens ({}) exceeds the context limit of {} tokens",
                self.prompt, self.max_tokens, self.context
            ),
            Some("messages".to_owned()),
        )
    }

    /// The counts of a context refusal, read back from its code and message: the native wording of
    /// [`Self::error`], or SGLang's `tokenizer_manager.py` wording inside a `context_overflow` ("... maximum context
    /// length of C tokens. You requested a total of T tokens: N tokens from the input messages and K tokens for the
    /// completion. ..."). `None` for any other error or wording.
    pub fn of(code: &str, message: &str) -> Option<Self> {
        let [prompt, max_tokens, context] = match code {
            CONTEXT_LENGTH_EXCEEDED => [
                "the prompt (",
                "plus max_tokens (",
                "exceeds the context limit of ",
            ],
            "context_overflow" => [
                " tokens: ",
                " tokens from the input messages and ",
                "maximum context length of ",
            ],
            _ => return None,
        }
        .map(|marker| number_after(message, marker));
        Some(Self {
            prompt: prompt?,
            max_tokens: max_tokens?,
            context: context?,
        })
    }
}

/// The decimal number right after the first `marker` in `text`.
fn number_after(text: &str, marker: &str) -> Option<u64> {
    let rest = &text[text.find(marker)? + marker.len()..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Client-facing view of a backend failure. Only a context overflow carries the backend's text (token counts);
/// other backend messages stay internal.
impl From<&BackendError> for ApiError {
    fn from(error: &BackendError) -> Self {
        match error {
            BackendError::ContextOverflow { message } => Self::new(
                400,
                INVALID_REQUEST,
                "context_overflow",
                format!("prompt plus output budget exceeds the backend context length: {message}"),
                Some("messages".to_owned()),
            ),
            BackendError::Connect(_) => Self::new(
                503,
                SERVER_ERROR,
                "backend_unavailable",
                "backend unreachable",
                None,
            ),
            BackendError::Timeout(_) => Self::new(
                504,
                SERVER_ERROR,
                "backend_timeout",
                "backend timed out",
                None,
            ),
            BackendError::Status { status, .. } => Self::new(
                502,
                SERVER_ERROR,
                "backend_error",
                format!("backend returned HTTP {status}"),
                None,
            ),
            BackendError::ToolSchema { index } => Self::invalid_value(
                match index {
                    Some(i) => format!("tools[{i}].function.parameters"),
                    None => "tools".to_owned(),
                },
                "a tool's parameters are not a valid JSON schema",
            ),
            BackendError::ToolsUnsupported => Self::unsupported_parameter(
                "tools",
                "this model's worker backend does not run the qwen3_coder tool parser",
            ),
            // The request's constraint field is named by the caller that knows the request
            // (`ApiError::grammar_compile`); the backend text never reaches the client.
            BackendError::GrammarCompile => Self::new(
                400,
                INVALID_REQUEST,
                "invalid_value",
                GRAMMAR_COMPILE_MESSAGE,
                None,
            ),
            BackendError::Protocol(_) => Self::new(
                502,
                SERVER_ERROR,
                "backend_protocol_error",
                "backend response was malformed or incomplete",
                None,
            ),
            BackendError::Internal(_) => {
                Self::new(500, SERVER_ERROR, "internal_error", "internal error", None)
            }
        }
    }
}

/// Whether the backend reported prefix-cache hits for this request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheReport {
    Reported,
    Unavailable,
}

impl CacheReport {
    /// The serialized name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reported => "reported",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Token counts of one request as the backend reported them (Q1/Q5): the status API `usage`, the journal
/// Terminal record and the stderr terminal line. `None` = not reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub cache_report: CacheReport,
}

impl RequestUsage {
    /// Nothing reported (no backend `usage` object).
    pub const UNAVAILABLE: Self = Self {
        prompt_tokens: None,
        completion_tokens: None,
        cached_tokens: None,
        cache_report: CacheReport::Unavailable,
    };

    /// Q1. `usage` is the backend's `usage` object, if any. A reported `prompt_tokens_details.cached_tokens`
    /// is passed through. When it is missing and `missing_means_zero` (the readiness-verified pin
    /// `expect.enable_cache_report = true` on a request that was not cancelled: SGLang omits the details on a
    /// miss), it is `0`/reported. Otherwise, and always without a `usage` object (failures, cut streams),
    /// nothing is inferred: `null`/unavailable. A cancelled request is not inferred because an aborted
    /// request's usage may omit the details for a request that never reported them (decided in M1.4 H1).
    pub fn from_backend(usage: Option<&Value>, missing_means_zero: bool) -> Self {
        let Some(usage) = usage.filter(|u| u.is_object()) else {
            return Self::UNAVAILABLE;
        };
        let cached_tokens = match reported_cached_tokens(usage) {
            Some(cached) => Some(cached),
            None if missing_means_zero => Some(0),
            None => None,
        };
        Self {
            prompt_tokens: usage.get("prompt_tokens").and_then(Value::as_u64),
            completion_tokens: usage.get("completion_tokens").and_then(Value::as_u64),
            cached_tokens,
            cache_report: if cached_tokens.is_some() {
                CacheReport::Reported
            } else {
                CacheReport::Unavailable
            },
        }
    }
}

/// Identity fields of the `cere` extension for one `[[deployments]]` entry, fixed at startup (M5 D5);
/// `config_hash` is the global one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeploymentIdentity {
    pub recipe_id: String,
    pub lock_sha256: String,
    pub profile_id: String,
    pub config_hash: String,
    pub execution_mode: String,
    /// `deployments.expect.enable_cache_report` (readiness keeps the backend equal to it): Q1's pin.
    pub cache_report_pinned: bool,
}

impl DeploymentIdentity {
    /// `None` without a loaded deployment (a `deployments.kind` with no pinned deployment: neither `sglang_http`
    /// nor `native_cuda`). Reads entry `index`.
    pub fn from_config(config: &RuntimeConfig, index: usize) -> Option<Self> {
        let entry = &config.deployments[index];
        let deployment = entry.lock.as_ref()?;
        Some(Self {
            recipe_id: deployment.recipe.recipe_id.clone(),
            lock_sha256: deployment.lock_sha256.clone(),
            profile_id: deployment.profile.profile_id.clone(),
            config_hash: config.config_hash(),
            execution_mode: deployment.profile.execution_mode.clone(),
            cache_report_pinned: entry.expect.as_ref().is_some_and(|e| e.enable_cache_report),
        })
    }
}

/// Per-request identity fields of the `cere` extension. The worker fields are set when admission grants the
/// request a worker (m3 design note Q3, Q11) and are `None` before dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttemptIdentity {
    pub request_id: String,
    pub attempt: u32,
    /// The granted worker (its device).
    pub worker_id: Option<String>,
    /// The granted worker's epoch at the grant (Q4).
    pub worker_epoch: Option<u64>,
    /// The process epoch (journal incarnation).
    pub epoch: u64,
    /// `ReadinessSnapshot::instance_id` of the granted worker's backend at the grant.
    pub instance_id: Option<String>,
}

/// The `cere` object added to a chat response (A4). `cached_tokens`/`cache_report` follow Q1
/// ([`RequestUsage::from_backend`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CereExtension {
    pub request_id: String,
    pub attempt: u32,
    pub worker_id: Option<String>,
    pub worker_epoch: Option<u64>,
    pub epoch: u64,
    pub recipe_id: String,
    pub lock_sha256: String,
    pub instance_id: Option<String>,
    pub profile_id: String,
    pub config_hash: String,
    pub execution_mode: String,
    pub cached_tokens: Option<u64>,
    pub cache_report: CacheReport,
}

impl CereExtension {
    /// `usage` is the request's [`RequestUsage`] (Q1).
    pub fn new(
        deployment: &DeploymentIdentity,
        attempt: &AttemptIdentity,
        usage: &RequestUsage,
    ) -> Self {
        Self {
            request_id: attempt.request_id.clone(),
            attempt: attempt.attempt,
            worker_id: attempt.worker_id.clone(),
            worker_epoch: attempt.worker_epoch,
            epoch: attempt.epoch,
            recipe_id: deployment.recipe_id.clone(),
            lock_sha256: deployment.lock_sha256.clone(),
            instance_id: attempt.instance_id.clone(),
            profile_id: deployment.profile_id.clone(),
            config_hash: deployment.config_hash.clone(),
            execution_mode: deployment.execution_mode.clone(),
            cached_tokens: usage.cached_tokens,
            cache_report: usage.cache_report,
        }
    }
}

/// `usage.prompt_tokens_details.cached_tokens` when the backend reports it; never inferred.
pub fn reported_cached_tokens(usage: &Value) -> Option<u64> {
    usage
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(Value::as_u64)
}

/// One `/readyz` worker entry (m35 design note P11; build-loop-m35 design note "CREG split"; m36 design note
/// Q11): the same 15 keys for a local and a remote worker. A local worker has node `local` and no remote fields
/// (`model` null); a remote worker without an accepted Register in this term is `joining` with no incarnation
/// fields.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReadyzWorker {
    pub state: &'static str,
    pub epoch: Option<u64>,
    pub ready: bool,
    pub instance_id: Option<String>,
    pub served_model_name: String,
    pub smoke: SmokeStatus,
    pub problems: Vec<Problem>,
    pub probed_at_unix_ms: Option<u64>,
    pub node: Node,
    /// Remote: the generation of the last accepted heartbeat.
    pub model_generation: Option<u64>,
    /// Remote: what is left of the coordinator-side lease view (0 once expired); null before the first grant.
    pub lease_ms_remaining: Option<u64>,
    /// Remote: the term it registered under.
    pub term: Option<u64>,
    /// Remote: the inventory read by DescribeResources in this round.
    pub inventory: Option<Inventory>,
    /// Remote: lowercase hex SHA-256 of the exact DescribeResources bytes.
    pub inventory_sha256: Option<String>,
    /// Remote: the last described model and the operator's residency intent (m36 design note Q11).
    pub model: Option<ReadyzModel>,
}

/// The `/readyz` `model` object of a remote worker (m36 design note Q11), also the `model` of the load/unload
/// answers (Q9): `managed` null before the first describe since this start, `state` the last described one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReadyzModel {
    pub managed: Option<bool>,
    pub desired: Desired,
    pub state: Option<ModelState>,
    pub lock_sha256: Option<String>,
}

impl From<ModelView> for ReadyzModel {
    fn from(view: ModelView) -> Self {
        Self {
            managed: view.managed,
            desired: view.desired,
            state: view.state,
            lock_sha256: view.lock_sha256,
        }
    }
}

impl ReadyzWorker {
    /// A local worker: its latest readiness snapshot and its registry state.
    pub fn local(
        snapshot: ReadinessSnapshot,
        state: &'static str,
        epoch: u64,
        ready: bool,
    ) -> Self {
        Self {
            state,
            epoch: Some(epoch),
            ready,
            instance_id: snapshot.instance_id,
            served_model_name: snapshot.served_model_name,
            smoke: snapshot.smoke,
            problems: snapshot.problems,
            probed_at_unix_ms: snapshot.probed_at_unix_ms,
            node: Node::Local,
            model_generation: None,
            lease_ms_remaining: None,
            term: None,
            inventory: None,
            inventory_sha256: None,
            model: None,
        }
    }

    /// A remote worker `config` in `state` at `now_ms`, with the problems of its current round. `instance_id` is
    /// the agent-reported backend id; `served_model_name` the pinned one.
    pub fn remote(
        config: &RemoteWorkerConfig,
        state: RemoteState,
        problems: Vec<Problem>,
        now_ms: u64,
    ) -> Self {
        Self {
            state: state.state.as_str(),
            epoch: state.epoch,
            ready: state.state == WorkerState::Ready,
            instance_id: state.inventory.as_ref().and_then(|i| i.instance_id.clone()),
            served_model_name: config.expect.served_model_name.clone(),
            smoke: state.smoke,
            problems,
            probed_at_unix_ms: state.last_heartbeat_ms,
            node: Node::Remote,
            model_generation: state.generation,
            lease_ms_remaining: state
                .lease_until_ms
                .map(|until| until.saturating_sub(now_ms)),
            term: state.term,
            inventory: state.inventory,
            inventory_sha256: state.inventory_sha256.map(|sha| hex(&sha)),
            model: Some(state.model.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(details: Value) -> Value {
        json!({"prompt_tokens": 7577, "completion_tokens": 4, "total_tokens": 7581, "prompt_tokens_details": details})
    }

    /// Q1: hit, miss with usage, no usage, pin false.
    #[test]
    fn q1_cached_token_semantics() {
        let hit = usage(json!({"cached_tokens": 7552}));
        let miss = usage(Value::Null);
        let reported = |cached| RequestUsage {
            prompt_tokens: Some(7577),
            completion_tokens: Some(4),
            cached_tokens: Some(cached),
            cache_report: CacheReport::Reported,
        };
        assert_eq!(RequestUsage::from_backend(Some(&hit), true), reported(7552));
        assert_eq!(RequestUsage::from_backend(Some(&miss), true), reported(0));
        let mut absent = miss.clone();
        absent
            .as_object_mut()
            .unwrap()
            .remove("prompt_tokens_details");
        assert_eq!(RequestUsage::from_backend(Some(&absent), true), reported(0));
        for pinned in [true, false] {
            assert_eq!(
                RequestUsage::from_backend(None, pinned),
                RequestUsage::UNAVAILABLE
            );
            assert_eq!(
                RequestUsage::from_backend(Some(&Value::Null), pinned),
                RequestUsage::UNAVAILABLE
            );
        }
        // Pin false: unchanged from M1.3 (a reported count passes through, a missing one is not inferred).
        assert_eq!(
            RequestUsage::from_backend(Some(&hit), false),
            reported(7552)
        );
        assert_eq!(
            RequestUsage::from_backend(Some(&miss), false),
            RequestUsage {
                cached_tokens: None,
                cache_report: CacheReport::Unavailable,
                ..reported(0)
            }
        );
        assert_eq!(
            serde_json::to_value(reported(0)).unwrap(),
            json!({"prompt_tokens": 7577, "completion_tokens": 4, "cached_tokens": 0, "cache_report": "reported"})
        );
    }
}
