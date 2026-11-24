//! `POST /v1/chat/completions` subset (m1 design note A3, Q7): strict validation of the raw JSON body
//! against the config and the F5 profile, the explicit backend body, and finish classification.
//!
//! Every field outside the subset, known to OpenAI or not, is rejected with 400 `unsupported_parameter` and
//! `param` = the field path. Nothing is dropped, defaulted silently or clamped: the output budget is always
//! forwarded explicitly, and a `length` finish below that budget is a failure (`backend_clamped_output`).

use cere_core::capabilities::SAMPLING_PARAMS;
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::value::RawValue;
use serde_json::{json, Map, Value};

use cere_core::capabilities::GRAMMAR_OUTPUTS;

use crate::api_types::{ApiError, INVALID_REQUEST};
use crate::config::{BackendKind, RuntimeConfig};
use crate::constraint::{self, known_keys, CONSTRAINT_FIELDS, SCHEMA_PARAM};
use crate::document;
use crate::image::{self, ImageInfo};
use crate::sglang::{BackendError, Rid};

/// The M2.4 fields, the seven M2.7 sampling fields (D8, gated by the profile's `sampling_params`), the M2.12
/// fields (D1, gated by `thinking_toggle`/`token_controls`) and the M2.8 constraint fields (D3).
const ALLOWED_FIELDS: [&str; 32] = [
    "chat_template_kwargs",
    "ebnf",
    "frequency_penalty",
    "ignore_eos",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "messages",
    "min_p",
    "min_tokens",
    "model",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "reasoning_effort",
    "regex",
    "repetition_penalty",
    "response_format",
    "seed",
    "separate_reasoning",
    "stop",
    "stop_token_ids",
    "stream",
    "stream_options",
    "temperature",
    "thinking_budget",
    "tool_choice",
    "tools",
    "top_k",
    "top_logprobs",
    "top_p",
];

/// ApiLenient D1 (api-lenient design note): the mapped parameters, each rewritten to its canonical field before
/// validation (D2): `chat_template_kwargs.preserve_thinking` and `preserve_thinking` true = absent (D3);
/// `chat_template_kwargs.reasoning_effort` -> `reasoning_effort`; `chat_template_kwargs.thinking` and
/// `enable_thinking` -> `chat_template_kwargs.enable_thinking` (Q5); `messages[].role` developer -> system (Q5).
/// An unequal pair is 400 `conflicting_parameters`. The entries are also the counter's `param` labels.
pub const MAPPED_PARAMS: [&str; 6] = [
    "chat_template_kwargs.preserve_thinking",
    "chat_template_kwargs.reasoning_effort",
    "chat_template_kwargs.thinking",
    "enable_thinking",
    "messages.role.developer",
    "preserve_thinking",
];

/// ApiLenient D1/D4: the inert chat parameters: type-checked ([`inert_value`]), dropped before normalization and
/// reported (`X-Cere-Ignored-Params`, `cere.ignored_params`, `cere_request_params_total`). No output contract.
pub const INERT_PARAMS: [&str; 9] = [
    "metadata",
    "modalities",
    "prompt_cache_key",
    "prompt_cache_retention",
    "safety_identifier",
    "service_tier",
    "store",
    "stream_options.include_obfuscation",
    "user",
];

/// ApiLenient D4: the Anthropic fields accepted and not enforced (before ApiLenient: logged to stderr only).
pub const ANTHROPIC_INERT_PARAMS: [&str; 3] = [
    "betas",
    "output_config.task_budget",
    "thinking.budget_tokens",
];

/// ApiLenient D4: the counter's `param` label for a reported path outside the lists (Anthropic's unknown top-level
/// fields, Q6), so the label set stays bounded.
pub const UNLISTED_PARAM: &str = "unlisted";

/// ApiLenient AL-4: reserved CERE extension names no deployment implements yet, with the spec that defines each:
/// 400 `unsupported_parameter` on every surface, never inert.
pub const RESERVED_EXTENSIONS: [(&str, &str); 1] =
    [("deterministic", "tasks/batch-concurrency-spec.md BC-D6")];

/// ApiLenient AL-4 (D9): known OpenAI chat fields that stay refused, with why and what works instead.
const REJECTED_FIELDS: [(&str, &str); 8] = [
    ("audio", "audio output is not supported: this deployment outputs text only; omit audio"),
    ("function_call", "function_call is the deprecated functions API and is not supported; send tool_choice"),
    ("functions", "functions is the deprecated functions API and is not supported; send tools"),
    ("prediction", "predicted outputs are not supported; omit prediction (decoding is unchanged without it)"),
    ("reasoning", "the reasoning object is not supported on chat completions; send reasoning_effort and chat_template_kwargs.enable_thinking"),
    ("thinking", "the thinking object is not supported on chat completions; send chat_template_kwargs.enable_thinking and reasoning_effort"),
    ("verbosity", "verbosity is not supported; ask for the length in the prompt or cap it with max_completion_tokens"),
    ("web_search_options", "web search is not supported; pass search results in messages or define a search tool"),
];

/// The counter label of a reported parameter path (ApiLenient D4): the path itself when a D1 list holds it, else
/// [`UNLISTED_PARAM`].
pub fn param_label(path: &str) -> &'static str {
    MAPPED_PARAMS
        .iter()
        .chain(&INERT_PARAMS)
        .chain(&ANTHROPIC_INERT_PARAMS)
        .find(|listed| **listed == path)
        .copied()
        .unwrap_or(UNLISTED_PARAM)
}

/// ApiLenient AL-4: the 400 for a reserved CERE extension name, or `None`.
pub fn reserved_extension(key: &str) -> Option<ApiError> {
    RESERVED_EXTENSIONS
        .iter()
        .find(|(name, _)| *name == key)
        .map(|(name, spec)| {
            ApiError::unsupported_parameter(
                *name,
                format!(
                    "{name} is a reserved CERE extension ({spec}) that this deployment does not implement; omit it"
                ),
            )
        })
}

/// ApiLenient D4: checks one non-null inert top-level field (`key` from [`INERT_PARAMS`]): Ok when the value is
/// inert, 400 `invalid_value` for a wrong type, 400 `unsupported_parameter` for a value with an effect cered does
/// not implement (`store: true`, `service_tier` other than auto/default, `modalities` other than `["text"]`).
pub fn inert_value(key: &str, value: &Value) -> Result<(), ApiError> {
    let wrong = |what: &str| {
        Err(ApiError::invalid_value(
            key,
            format!("{key} must be {what}"),
        ))
    };
    match (key, value) {
        ("user" | "safety_identifier" | "prompt_cache_key" | "prompt_cache_retention", Value::String(_)) => Ok(()),
        ("user" | "safety_identifier" | "prompt_cache_key" | "prompt_cache_retention", _) => wrong("a string"),
        ("metadata", Value::Object(map)) if map.values().all(Value::is_string) => Ok(()),
        ("metadata", _) => wrong("an object of strings"),
        ("store", Value::Bool(false)) => Ok(()),
        ("store", Value::Bool(true)) => Err(ApiError::unsupported_parameter(
            key,
            "store: true needs a response store; cered is stateless (store: false is accepted and ignored)",
        )),
        ("store", _) => wrong("a boolean"),
        ("service_tier", Value::String(tier)) if tier == "auto" || tier == "default" => Ok(()),
        ("service_tier", Value::String(tier)) => Err(ApiError::unsupported_parameter(
            key,
            format!("service_tier '{tier}' is not supported; 'auto' and 'default' are accepted and ignored"),
        )),
        ("service_tier", _) => wrong("a string"),
        ("modalities", Value::Array(items)) if items.len() == 1 && items[0] == "text" => Ok(()),
        ("modalities", Value::Array(_)) => Err(ApiError::unsupported_parameter(
            key,
            "this deployment outputs text only; modalities [\"text\"] is accepted and ignored",
        )),
        ("modalities", _) => wrong("an array"),
        _ => Err(ApiError::unsupported_parameter(
            key,
            format!("parameter '{key}' is not supported"),
        )),
    }
}

/// M2.12 D2: the 400 for a thinking field on a profile without `thinking_toggle`.
fn thinking_refused(param: &str) -> ApiError {
    ApiError::unsupported_parameter(
        param,
        format!("this deployment's profile does not declare thinking_toggle: {param} is not supported (M2.12)"),
    )
}

/// ApiLenient D2/D4: a chat body after alias canonicalization and inert removal. The validator reads it through
/// [`Canonical::get`] and never sees an alias or an inert field; the client's body is not copied.
struct Canonical<'a> {
    object: &'a Map<String, Value>,
    /// The canonical values of the top-level keys rewritten here.
    replaced: Map<String, Value>,
    /// Top-level keys handled here (aliases and inert fields).
    removed: Vec<&'static str>,
    ignored: Vec<&'static str>,
    mapped: Vec<&'static str>,
}

impl<'a> Canonical<'a> {
    /// `key` of the canonical body (`null` = absent).
    fn get(&self, key: &str) -> Option<&Value> {
        if self.removed.contains(&key) {
            return None;
        }
        self.replaced
            .get(key)
            .or_else(|| self.object.get(key))
            .filter(|value| !value.is_null())
    }

    fn keys(&self) -> impl Iterator<Item = &'a String> + '_ {
        self.object
            .keys()
            .filter(|key| !self.removed.contains(&key.as_str()))
    }
}

/// 400 `conflicting_parameters` for two spellings of one field with unequal values (ApiLenient D2).
fn conflicting(alias: &str, canonical: &str, a: &Value, b: &Value) -> ApiError {
    ApiError::new(
        400,
        INVALID_REQUEST,
        "conflicting_parameters",
        format!("{alias} ({a}) and {canonical} ({b}) are the same parameter with different values; send one"),
        Some(alias.to_owned()),
    )
}

/// `logit_bias` values: OpenAI's range (D8, Q6).
const MAX_LOGIT_BIAS: f64 = 100.0;

/// D2 bounds of a `tools` list.
const MAX_TOOLS: usize = 128;
const MAX_TOOL_NAME_CHARS: usize = 64;
/// Containers from a tool's `parameters` object (depth 1) down to its deepest value.
const MAX_PARAMETERS_NESTING: usize = 32;
/// M2.12 D13: SGLang's `reasoning_effort` strings (a number in `[0, MAX_REASONING_EFFORT]` is accepted too).
const REASONING_EFFORTS: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];
const MAX_REASONING_EFFORT: f64 = 0.99;
/// M2.12 D13: the efforts the pinned Qwen chat template renders with thinking on; it raises on any other.
const THINKING_EFFORTS: [&str; 3] = ["low", "medium", "xhigh"];
/// M2.12 D1: at most this many `stop_token_ids`.
const MAX_STOP_TOKEN_IDS: usize = 32;
/// Audit F4: at most this many `stop` strings. The native matcher runs every one over every committed token and
/// the sidecar tokenizes each, so the cost grows with the count; OpenAI itself allows 4.
const MAX_STOP_STRINGS: usize = 16;
/// Audit F4: at most this many bytes per `stop` string (UTF-8). A longer one is never a natural delimiter, and the
/// matcher holds a window as long as the longest.
const MAX_STOP_BYTES: usize = 256;
/// The parser `tool_calls` needs on an SGLang deployment (D1): the one the native detector reproduces.
pub const TOOL_CALL_PARSER: &str = "qwen3_coder";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One call of an assistant message (D3): `index` is accepted and dropped, `type` is always `"function"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolCallFunction,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// The exact client string (D7: digested as sent, never re-serialized).
    pub arguments: String,
}

/// A validated message (D3). Optional fields are serialized only when present, so a pre-M2.4 message serializes
/// (and digests) exactly as before.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ChatMessage {
    pub role: Role,
    /// `None` = an assistant `content: null` (only with tool calls). Serialized as `""`, which is how SGLang
    /// renders it (`serving_chat.py:1281-1283`), so `null` and `""` forward, render and digest alike (D7).
    #[serde(serialize_with = "content_or_empty")]
    pub content: Option<Content>,
    /// O113: an assistant message's thinking as the client sent it (non-empty); the chat template renders it into
    /// the turn's `<think>` block on both paths. Bound by the request digest, not by session prefix digests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// A plain `{role, content}` message.
    pub fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(Content::Text(content.into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// The message with a text-only parts array joined into one string, which the chat template renders
    /// identically (m213 F4: `render_content` concatenates the `text` items); `None` when it holds an image.
    pub fn joined(&self) -> Option<Self> {
        let Some(Content::Parts(parts)) = &self.content else {
            return Some(self.clone());
        };
        let mut text = String::new();
        for part in parts {
            match part {
                ContentPart::Text { text: t } => text.push_str(t),
                ContentPart::ImageUrl { .. } => return None,
            }
        }
        Some(Self {
            content: Some(Content::Text(text)),
            ..self.clone()
        })
    }
}

/// A message's `content` (m213 D2): a string, or (user and tool messages, VisionLong C7) an array of text and image parts,
/// serialized as sent so it forwards value-exact (D7) and digests byte-sensitive (D6).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

/// `image_url` of an image part: the data URL as sent, `detail` only when sent (only `"auto"` is accepted), and
/// the header-derived grid (never serialized).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip)]
    pub image: ImageInfo,
}

fn content_or_empty<S: Serializer>(content: &Option<Content>, s: S) -> Result<S::Ok, S::Error> {
    match content {
        Some(content) => content.serialize(s),
        None => s.serialize_str(""),
    }
}

/// The assistant message a completed turn commits (D7): its text and its calls, so the client's verbatim echo
/// of the reply digests equal.
pub fn assistant_reply(content: String, tool_calls: Vec<ToolCall>) -> ChatMessage {
    ChatMessage {
        role: Role::Assistant,
        content: Some(Content::Text(content)),
        reasoning_content: None,
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        tool_call_id: None,
    }
}

/// Tool calls of a response (D7): unary `message.tool_calls` or the stream's `delta.tool_calls` fragments,
/// assembled per `index` - `id` and `name` from the first fragment that has them, `arguments` concatenated in
/// arrival order (a fragment without `index` counts at its position).
#[derive(Debug, Default)]
pub struct ToolCallAssembly {
    calls: std::collections::BTreeMap<u64, ToolCall>,
}

impl ToolCallAssembly {
    /// Adds the fragments of one `tool_calls` array; returns whether it held any.
    pub fn push(&mut self, tool_calls: Option<&Value>) -> bool {
        let Some(fragments) = tool_calls.and_then(Value::as_array) else {
            return false;
        };
        for (position, fragment) in fragments.iter().enumerate() {
            let index = fragment
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or(position as u64);
            let text = |pointer: &str| fragment.pointer(pointer).and_then(Value::as_str);
            let call = self.calls.entry(index).or_insert_with(|| ToolCall {
                id: String::new(),
                kind: "function".to_owned(),
                function: ToolCallFunction {
                    name: String::new(),
                    arguments: String::new(),
                },
            });
            if call.id.is_empty() {
                call.id = text("/id").unwrap_or_default().to_owned();
            }
            if call.function.name.is_empty() {
                call.function.name = text("/function/name").unwrap_or_default().to_owned();
            }
            call.function
                .arguments
                .push_str(text("/function/arguments").unwrap_or_default());
        }
        !fragments.is_empty()
    }

    pub fn into_calls(self) -> Vec<ToolCall> {
        self.calls.into_values().collect()
    }
}

/// `tool_choice` (D2, M2.8 D4): `auto` (the default with tools), `none` (validated, never rendered; the default
/// without tools), `required` or a named function. Serializes in canonical form (`"auto"`, `"none"`,
/// `"required"`, `{"function":{"name":...},"type":"function"}`): the digest input (D13 entry for M2.14).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}

impl Serialize for ToolChoice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::None => serializer.serialize_str("none"),
            Self::Required => serializer.serialize_str("required"),
            Self::Named(name) => {
                json!({"type": "function", "function": {"name": name}}).serialize(serializer)
            }
        }
    }
}

/// Decoding mode a request needs from the profile: `temperature: 0` is greedy, anything else (including the
/// OpenAI default when `temperature` is absent) is sampled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SamplerMode {
    Greedy,
    Sampled,
}

impl SamplerMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::Sampled => "sampled",
        }
    }
}

/// A validated request.
#[derive(Clone, Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// Output budget forwarded as `max_completion_tokens`: the client's value or `api.default_max_output_tokens`.
    pub max_output_tokens: u64,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    /// `stop` as a list (a single string becomes one element); empty = none.
    pub stop: Vec<String>,
    pub stream: bool,
    /// The client asked for the trailing usage chunk. The backend is always asked for it on streams.
    pub include_usage: bool,
    pub sampler: SamplerMode,
    /// `tools` exactly as received (D4): the bytes of the request's subtree, forwarded untouched.
    pub tools: Option<Box<RawValue>>,
    pub tool_choice: ToolChoice,
    /// The client's `tool_choice` bytes when sent (M2.8 D6): a `required`/named choice is forwarded as these.
    pub tool_choice_raw: Option<Box<RawValue>>,
    /// Some tool has `strict: true` (with `auto`, a tool constraint, D4).
    pub strict_tools: bool,
    /// M2.8 D3/D6: the constraint field as the client's bytes; `None` when absent, `null` or `{type:"text"}`.
    /// At most one of the three is `Some`.
    pub response_format: Option<Box<RawValue>>,
    pub regex: Option<Box<RawValue>>,
    pub ebnf: Option<Box<RawValue>>,
    pub parallel_tool_calls: Option<bool>,
    pub logprobs: bool,
    pub top_logprobs: Option<u32>,
    /// m213 D5/D11: the context limit of the profile's image decode combination when the request has an image
    /// part (`None` without one); a native image session is created at it.
    #[serde(skip)]
    pub image_context_tokens: Option<u64>,
    /// VisionLong C3: an image request may speculate (the profile has the image `verify` combination for its
    /// sampler mode); `false` without an image part.
    #[serde(skip)]
    pub image_speculation: bool,
    /// VisionLong C3: an image request may carry a prefix capture hint (the profile has an image `prefill`
    /// combination); `false` without an image part.
    #[serde(skip)]
    pub image_capture: bool,
    /// M2.7 D8 sampling fields, each only when sent (and declared by the profile's `sampling_params`).
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub seed: Option<u64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub repetition_penalty: Option<f64>,
    /// `logit_bias` as received (keys are canonical decimal token ids, values numbers in [-100, 100]).
    pub logit_bias: Option<Map<String, Value>>,
    /// M2.12 D1: `chat_template_kwargs.enable_thinking` as sent (`None` = absent).
    pub enable_thinking: Option<bool>,
    /// ApiLenient AL-2 (D3): `Some(false)` when the request sent `preserve_thinking: false` (either spelling): the
    /// template drops the think blocks of the assistant turns before the last user query. `true` is the template's
    /// default and canonicalizes to `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preserve_thinking: Option<bool>,
    /// M2.12 D13: `reasoning_effort` as sent (a string or a number).
    pub reasoning_effort: Option<Value>,
    /// M2.12 D4/D13: thinking resolved as SGLang does: the explicit flag, else `reasoning_effort != "none"`, else
    /// the profile default (off on every profile, Q4 a).
    pub thinking: bool,
    /// M2.12 D4: `separate_reasoning` as sent (absent = true).
    pub separate_reasoning: Option<bool>,
    /// M2.12 D1/D8: `stop_token_ids` as sent, as a sorted set (SGLang collapses duplicates).
    pub stop_token_ids: Option<Vec<u32>>,
    /// M2.12 D10: `min_tokens` as sent.
    pub min_tokens: Option<u64>,
    /// M2.12 D9: `ignore_eos` as sent.
    pub ignore_eos: Option<bool>,
    /// ApiFit AF-L7: `thinking_budget` as sent (native, thinking on; enforced by `native/serve.rs`).
    pub thinking_budget: Option<u64>,
    /// Choices to generate (M2.6 D11): 1 unless a native profile with `decode_batch_max` > 1 allows more.
    pub n: u32,
    /// Track LoRA: the adapter the request's `model` named (a lease on it), set by the chat front after validation;
    /// `None` for a base-model request.
    #[serde(skip)]
    pub adapter: Option<std::sync::Arc<crate::native::lora::Adapter>>,
    /// WeightLab: the research plan the request's scope is bound to (held until the request ends), set by the chat
    /// front after acceptance; `None` without `[deployments.native.research]` or a binding.
    #[serde(skip)]
    pub research: Option<std::sync::Arc<crate::native::research::Applied>>,
    /// ApiLenient D4: the inert parameters the request carried ([`INERT_PARAMS`]), dropped before validation:
    /// reported, never forwarded or digested.
    #[serde(skip)]
    pub ignored_params: Vec<&'static str>,
    /// ApiLenient D2: the aliases the request spelled ([`MAPPED_PARAMS`]), rewritten to their canonical fields:
    /// counted, never digested.
    #[serde(skip)]
    pub mapped_params: Vec<&'static str>,
}

impl ChatRequest {
    /// M2.6 D11 (OQ9): n > 1 is refused on a session turn (a turn commits one assistant reply) and in continue
    /// mode (resume buffers hold one choice).
    pub fn check_choices(&self, session: bool, continue_mode: bool) -> Result<(), ApiError> {
        if self.n > 1 && (session || continue_mode) {
            return Err(ApiError::unsupported_parameter(
                "n",
                "n > 1 is not supported on a session turn or with X-Cere-On-Disconnect: continue",
            ));
        }
        Ok(())
    }

    /// The raw bytes of `tools` as received (D4), when the request has tools.
    pub fn tools_json(&self) -> Option<&str> {
        self.tools.as_deref().map(RawValue::get)
    }

    /// Every image of the request in message and part order (m213 D4: history included).
    pub fn images(&self) -> impl Iterator<Item = &ImageInfo> {
        self.messages
            .iter()
            .filter_map(|m| match &m.content {
                Some(Content::Parts(parts)) => Some(parts),
                _ => None,
            })
            .flatten()
            .filter_map(|part| match part {
                ContentPart::ImageUrl { image_url } => Some(&image_url.image),
                ContentPart::Text { .. } => None,
            })
    }

    /// Whether the tools are rendered into the prompt: tools present and `tool_choice` not `none` (F7, M2.8 D4).
    pub fn render_tools(&self) -> bool {
        self.tools.is_some() && self.tool_choice != ToolChoice::None
    }

    /// Canonical form of every semantically relevant field (input to the Q3 request digest). D7: `tools` enters
    /// as its raw bytes (a string, so key order and number spelling count); defaults are canonical (absent and
    /// `"auto"` equal with tools, `parallel_tool_calls: true` and `logprobs: false` omitted), so a pre-M2.4 body
    /// normalizes as before. M2.7 D8: a sampling field enters only when it is not at its default (`top_k` -1,
    /// `min_p` and the additive penalties 0, `repetition_penalty` 1, `logit_bias` empty) and `seed` whenever given,
    /// so a pre-M2.7 body normalizes as before; `logit_bias` values enter as numbers, so `1` and `1.0` digest equal.
    /// M2.8: `tool_choice` is canonical (a named choice digests the same in any key order).
    pub fn normalized(&self) -> Value {
        let mut normalized = json!({
            "model": self.model,
            "messages": self.messages,
            "max_completion_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop,
            "stream": self.stream,
            "include_usage": self.include_usage,
        });
        if let Some(tools) = self.tools_json() {
            normalized["tools"] = json!(tools);
            if self.tool_choice != ToolChoice::Auto {
                normalized["tool_choice"] = json!(self.tool_choice);
            }
        }
        // M2.8 D6: the constraint field enters as its raw bytes (a string); `{type:"text"}` is absent.
        if let Some((name, value)) = self.constraint_json() {
            normalized[name] = json!(value.get());
        }
        if self.parallel_tool_calls == Some(false) {
            normalized["parallel_tool_calls"] = json!(false);
        }
        if self.logprobs {
            normalized["logprobs"] = json!(true);
        }
        if let Some(top) = self.top_logprobs {
            normalized["top_logprobs"] = json!(top);
        }
        if let Some(top_k) = self.top_k.filter(|k| *k != -1) {
            normalized["top_k"] = json!(top_k);
        }
        for (name, value, default) in [
            ("min_p", self.min_p, 0.0),
            ("frequency_penalty", self.frequency_penalty, 0.0),
            ("presence_penalty", self.presence_penalty, 0.0),
            ("repetition_penalty", self.repetition_penalty, 1.0),
        ] {
            if let Some(value) = value.filter(|v| *v != default) {
                normalized[name] = json!(value);
            }
        }
        if let Some(seed) = self.seed {
            normalized["seed"] = json!(seed);
        }
        if let Some(bias) = self.logit_bias.as_ref().filter(|b| !b.is_empty()) {
            normalized["logit_bias"] = bias
                .iter()
                .map(|(id, value)| (id.clone(), json!(value.as_f64())))
                .collect::<Map<_, _>>()
                .into();
        }
        // M2.12 D6: each field only when it differs from its default, so a pre-M2.12 body normalizes as before.
        if self.thinking {
            normalized["enable_thinking"] = json!(true);
        }
        if let Some(effort) = &self.reasoning_effort {
            normalized["reasoning_effort"] = effort.clone();
        }
        if self.separate_reasoning == Some(false) {
            normalized["separate_reasoning"] = json!(false);
        }
        if let Some(ids) = self.stop_token_ids.as_ref().filter(|ids| !ids.is_empty()) {
            normalized["stop_token_ids"] = json!(ids);
        }
        if let Some(min) = self.min_tokens.filter(|min| *min > 0) {
            normalized["min_tokens"] = json!(min);
        }
        // ApiLenient AL-2: only `false` (D3), so every earlier body normalizes as before.
        if let Some(preserve) = self.preserve_thinking {
            normalized["chat_template_kwargs"] = json!({ "preserve_thinking": preserve });
        }
        if self.ignore_eos == Some(true) {
            normalized["ignore_eos"] = json!(true);
        }
        // ApiFit AF-L7: only when sent, so every earlier body normalizes as before.
        if let Some(budget) = self.thinking_budget {
            normalized["thinking_budget"] = json!(budget);
        }
        // M2.6: only n > 1 enters, so every earlier body normalizes as before.
        if self.n != 1 {
            normalized["n"] = json!(self.n);
        }
        normalized
    }

    /// The exact body for SGLang: explicit budget and `rid`; streams always request the usage chunk, which the
    /// clamp check and committed-token accounting need (drop it towards a client that did not ask). D5: `tools`
    /// are the client's bytes, with an explicit `tool_choice`; the other M2.4 fields, the M2.7 sampling fields (D9)
    /// and the M2.12 fields (D3, D8-D10) only when sent.
    pub fn backend_body(&self, rid: &Rid) -> BackendBody {
        let mut body = Map::new();
        body.insert("rid".to_owned(), json!(rid.as_str()));
        body.insert("model".to_owned(), json!(self.model));
        body.insert("messages".to_owned(), json!(self.messages));
        body.insert(
            "max_completion_tokens".to_owned(),
            json!(self.max_output_tokens),
        );
        if let Some(temperature) = self.temperature {
            body.insert("temperature".to_owned(), json!(temperature));
        }
        if let Some(top_p) = self.top_p {
            body.insert("top_p".to_owned(), json!(top_p));
        }
        if !self.stop.is_empty() {
            body.insert("stop".to_owned(), json!(self.stop));
        }
        body.insert("stream".to_owned(), json!(self.stream));
        if self.stream {
            body.insert(
                "stream_options".to_owned(),
                json!({ "include_usage": true }),
            );
        }
        // Grammar-free bodies keep their pre-M2.8 bytes (C10): `auto`/`none` stay canonical fields.
        if self.tools.is_some() && matches!(self.tool_choice, ToolChoice::Auto | ToolChoice::None) {
            body.insert("tool_choice".to_owned(), json!(self.tool_choice));
        }
        if let Some(parallel) = self.parallel_tool_calls {
            body.insert("parallel_tool_calls".to_owned(), json!(parallel));
        }
        if self.logprobs {
            body.insert("logprobs".to_owned(), json!(true));
        }
        if let Some(top) = self.top_logprobs {
            body.insert("top_logprobs".to_owned(), json!(top));
        }
        if let Some(top_k) = self.top_k {
            body.insert("top_k".to_owned(), json!(top_k));
        }
        for (name, value) in [
            ("min_p", self.min_p),
            ("frequency_penalty", self.frequency_penalty),
            ("presence_penalty", self.presence_penalty),
            ("repetition_penalty", self.repetition_penalty),
        ] {
            if let Some(value) = value {
                body.insert(name.to_owned(), json!(value));
            }
        }
        if let Some(seed) = self.seed {
            body.insert("seed".to_owned(), json!(seed));
        }
        if let Some(bias) = &self.logit_bias {
            body.insert("logit_bias".to_owned(), Value::Object(bias.clone()));
        }
        // ApiLenient AL-2: `preserve_thinking` only when false (D3), so every earlier body keeps its bytes.
        let mut kwargs = Map::new();
        if let Some(enable) = self.enable_thinking {
            kwargs.insert("enable_thinking".to_owned(), json!(enable));
        }
        if let Some(preserve) = self.preserve_thinking {
            kwargs.insert("preserve_thinking".to_owned(), json!(preserve));
        }
        if !kwargs.is_empty() {
            body.insert("chat_template_kwargs".to_owned(), Value::Object(kwargs));
        }
        if let Some(separate) = self.separate_reasoning {
            body.insert("separate_reasoning".to_owned(), json!(separate));
        }
        if let Some(effort) = &self.reasoning_effort {
            body.insert("reasoning_effort".to_owned(), effort.clone());
        }
        if let Some(ids) = &self.stop_token_ids {
            body.insert("stop_token_ids".to_owned(), json!(ids));
        }
        if let Some(min) = self.min_tokens {
            body.insert("min_tokens".to_owned(), json!(min));
        }
        if let Some(ignore) = self.ignore_eos {
            body.insert("ignore_eos".to_owned(), json!(ignore));
        }
        if let Some(budget) = self.thinking_budget {
            body.insert("thinking_budget".to_owned(), json!(budget));
        }
        let mut raw = Vec::new();
        if let Some(tools) = &self.tools {
            raw.push(("tools", tools.clone()));
            if matches!(
                self.tool_choice,
                ToolChoice::Required | ToolChoice::Named(_)
            ) {
                if let Some(choice) = &self.tool_choice_raw {
                    raw.push(("tool_choice", choice.clone()));
                }
            }
        }
        raw.extend(
            self.constraint_json()
                .map(|(name, value)| (name, value.to_owned())),
        );
        BackendBody { fields: body, raw }
    }

    /// The constraint field and its raw bytes, when the request has one (D6).
    pub fn constraint_json(&self) -> Option<(&'static str, &RawValue)> {
        [
            ("response_format", &self.response_format),
            ("regex", &self.regex),
            ("ebnf", &self.ebnf),
        ]
        .into_iter()
        .find_map(|(name, value)| value.as_deref().map(|value| (name, value)))
    }

    /// The constraint field (D3): `response_format`, `regex` or `ebnf`; at most one is present.
    pub fn constraint_field(&self) -> Option<&'static str> {
        self.constraint_json().map(|(name, _)| name)
    }

    /// A tool constraint (D4): tools with `tool_choice` `required` or named, or `auto` and a strict tool.
    pub fn tool_constrained(&self) -> bool {
        self.tools.is_some()
            && match self.tool_choice {
                ToolChoice::Required | ToolChoice::Named(_) => true,
                ToolChoice::Auto => self.strict_tools,
                ToolChoice::None => false,
            }
    }

    /// The D10 grammar output the request needs: `json_schema` (`json_object` and `json_schema`), `regex`,
    /// `ebnf` or `tool_calls_constrained`; `None` for a grammar-free request.
    pub fn grammar_output(&self) -> Option<&'static str> {
        match self.constraint_field() {
            Some(field) => Some(grammar_output_of(field)),
            None => self.tool_constrained().then_some(TOOL_CALLS_CONSTRAINED),
        }
    }
}

/// The validated M2.12 thinking fields (D1/D4/D13).
struct Thinking {
    enable_thinking: Option<bool>,
    preserve_thinking: Option<bool>,
    reasoning_effort: Option<Value>,
    separate_reasoning: Option<bool>,
    resolved: bool,
}

/// The SGLang chat body (D5): JSON fields plus the client's raw subtrees (`tools`, and M2.8's constraint fields
/// and a `required`/named `tool_choice`, D6), which serialize verbatim (a `RawValue` is written as is) after the
/// fields, on the local HTTP call and into the remote Submit's `input_json` alike.
#[derive(Clone, Debug)]
pub struct BackendBody {
    pub fields: Map<String, Value>,
    pub raw: Vec<(&'static str, Box<RawValue>)>,
}

impl Serialize for BackendBody {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.fields.len() + self.raw.len()))?;
        for (key, value) in &self.fields {
            map.serialize_entry(key, value)?;
        }
        for (key, value) in &self.raw {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

/// The client's subtrees the request keeps as bytes (D4, M2.8 D6); a second parse of the body takes them.
#[derive(Default, Deserialize)]
struct RawFields {
    tools: Option<Box<RawValue>>,
    tool_choice: Option<Box<RawValue>>,
    response_format: Option<Box<RawValue>>,
    regex: Option<Box<RawValue>>,
    ebnf: Option<Box<RawValue>>,
}

const TOOL_CALLS_CONSTRAINED: &str = "tool_calls_constrained";

/// The D10 grammar output of a constraint field: `response_format` (`json_object` and `json_schema`) is
/// `json_schema`.
fn grammar_output_of(field: &str) -> &'static str {
    match field {
        "regex" => "regex",
        "ebnf" => "ebnf",
        _ => "json_schema",
    }
}

/// `tool_choice` (D2, M2.8 D4): `"required"` needs `tools`; a named `{type:"function", function:{name}}` needs
/// the name among `tools`. Any other value -> 400 `invalid_value` `tool_choice`.
fn tool_choice(value: Option<&Value>, tools: Option<&Value>) -> Result<ToolChoice, ApiError> {
    let invalid = |message: String| Err(ApiError::invalid_value("tool_choice", message));
    let object = match value {
        None => return Ok(ToolChoice::Auto),
        Some(Value::String(choice)) => {
            return match choice.as_str() {
                "auto" => Ok(ToolChoice::Auto),
                "none" => Ok(ToolChoice::None),
                "required" if tools.is_some() => Ok(ToolChoice::Required),
                "required" => invalid("tool_choice \"required\" needs tools".to_owned()),
                _ => invalid(
                    "tool_choice must be \"auto\", \"none\", \"required\" or a named function"
                        .to_owned(),
                ),
            }
        }
        Some(Value::Object(object)) => object,
        Some(_) => {
            return invalid(
                "tool_choice must be \"auto\", \"none\", \"required\" or a named function"
                    .to_owned(),
            )
        }
    };
    let name = Some(object)
        .filter(|o| o.len() == 2 && o.get("type").and_then(Value::as_str) == Some("function"))
        .and_then(|o| o.get("function"))
        .and_then(Value::as_object)
        .filter(|function| function.len() == 1)
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str);
    let Some(name) = name else {
        return invalid(
            "a named tool_choice is {\"type\": \"function\", \"function\": {\"name\": string}}"
                .to_owned(),
        );
    };
    let listed = tools
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|tool| tool.pointer("/function/name").and_then(Value::as_str) == Some(name));
    if listed {
        Ok(ToolChoice::Named(name.to_owned()))
    } else {
        invalid(format!("tool_choice names '{name}', which is not in tools"))
    }
}

/// `404 model_not_found` for a model id this `cered` does not serve (chat `model`, `GET /v1/models/{id}`);
/// `served` lists every served model in config order (M5 D5).
pub fn model_not_found(model: &str, served: &[String]) -> ApiError {
    ApiError::new(
        404,
        INVALID_REQUEST,
        "model_not_found",
        format!(
            "model '{model}' is not served here; served models: {}",
            served.join(", ")
        ),
        Some("model".to_owned()),
    )
}

/// Validation inputs taken once from the config and the pinned profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatPolicy {
    pub served_model_name: String,
    /// Every served model in config order (the `model_not_found` message, M5 D5).
    pub served_models: Vec<String>,
    pub max_messages: u64,
    pub default_max_output_tokens: u64,
    pub max_output_tokens: u64,
    pub greedy: bool,
    pub sampled: bool,
    /// D1: `tools` are accepted for a sampler mode iff the profile has its `decode`/`tool_calls` combination
    /// and, on `sglang_http`, the backend's pinned `tool_call_parser` is [`TOOL_CALL_PARSER`].
    pub tools_greedy: bool,
    pub tools_sampled: bool,
    /// M2.8 D10: the (grammar output, sampler mode) pairs of the profile's `decode` combinations; a constrained
    /// request needs its pair.
    pub grammar_decodes: Vec<(&'static str, SamplerMode)>,
    /// M2.8 D5: the profile's `json_schema_keywords`, the only keywords a constraint's schema may use.
    pub json_schema_keywords: Vec<String>,
    /// D12: the profile's `max_top_logprobs`; `None` refuses `logprobs`.
    pub max_top_logprobs: Option<u32>,
    /// M2.7 D8: the profile's `sampling_params`; a sampling field outside it is 400 `unsupported_parameter`.
    pub sampling_params: Vec<String>,
    /// `deployments.kind = "native_cuda"` (spec m22 "Native path contract"): names the native path in refusals.
    pub native: bool,
    /// m213 D5: image parts are accepted for a sampler mode iff the profile has its `decode` combination with
    /// `modality: "image"`.
    pub images_greedy: bool,
    pub images_sampled: bool,
    /// m213 Q3 (a): the deployment has `[[federation.workers]]`, so a request may be granted to a remote worker;
    /// image bytes never leave the host, so images are refused on it.
    pub remote_workers: bool,
    /// m213 D11: the smallest `max_context_tokens` of the profile's image decode combinations (`None` without one).
    pub image_context_tokens: Option<u64>,
    /// VisionLong C3: the profile's image `verify` combinations per sampler mode (output `text`): an image
    /// session keeps speculation. Without one (the -v6 lab profile) it runs with `NO_SPECULATION` (m213 D14).
    pub image_speculation_greedy: bool,
    pub image_speculation_sampled: bool,
    /// VisionLong C3: the profile has an image `prefill` combination: an image session may capture its prefix
    /// state at the last message boundary (m210 D4 (c)). Without one (the -v6 lab profile) capture stays off.
    pub image_capture: bool,
    /// M2.12 D2: the profile's `thinking_toggle` (absent = false: the thinking fields are refused).
    pub thinking_toggle: bool,
    /// M2.12 D2: the profile's `token_controls` (absent = empty: all three refused).
    pub token_controls: Vec<String>,
    /// M2.12 D5: the profile's `vocab_size`; `None` refuses `stop_token_ids`.
    pub vocab_size: Option<u32>,
    /// M2.6 D11: the largest `n` - a native profile's `decode_batch_max` (its rows), otherwise 1 (OQ4).
    pub max_choices: u64,
}

impl ChatPolicy {
    /// Needs entry `deployment`'s `served_model_name` and its loaded deployment profile.
    pub fn from_config(config: &RuntimeConfig, deployment: usize) -> Result<Self, String> {
        let entry = &config.deployments[deployment];
        let served_model_name = entry.served_model_name.clone().ok_or_else(|| {
            format!("chat validation needs deployments[{deployment}].served_model_name")
        })?;
        let profile = &entry
            .lock
            .as_ref()
            .ok_or("chat validation needs a pinned deployment profile")?
            .profile;
        let decodes = |mode: SamplerMode, modality: &str, output: &str| {
            profile.validated_combinations.iter().any(|c| {
                c.operation == "decode"
                    && c.modality == modality
                    && c.output == output
                    && c.sampler_mode == mode.as_str()
            })
        };
        let image_op = |operation: &str, sampler_mode: &str| {
            profile.validated_combinations.iter().any(|c| {
                c.operation == operation
                    && c.modality == "image"
                    && c.output == "text"
                    && c.sampler_mode == sampler_mode
            })
        };
        let native = entry.kind == BackendKind::NativeCuda;
        let parser = native
            || entry
                .expect
                .as_ref()
                .is_some_and(|e| e.tool_call_parser == TOOL_CALL_PARSER);
        let remote_workers = config
            .federation
            .iter()
            .flat_map(|f| &f.workers)
            .any(|w| w.expect.served_model_name == served_model_name);
        Ok(Self {
            served_model_name,
            served_models: config
                .deployments
                .iter()
                .filter_map(|d| d.served_model_name.clone())
                .collect(),
            max_messages: config.api.max_messages,
            default_max_output_tokens: config.api.default_max_output_tokens,
            max_output_tokens: config.api.max_output_tokens,
            greedy: decodes(SamplerMode::Greedy, "text", "text"),
            sampled: decodes(SamplerMode::Sampled, "text", "text"),
            tools_greedy: parser && decodes(SamplerMode::Greedy, "text", "tool_calls"),
            tools_sampled: parser && decodes(SamplerMode::Sampled, "text", "tool_calls"),
            grammar_decodes: GRAMMAR_OUTPUTS
                .iter()
                .flat_map(|output| {
                    [SamplerMode::Greedy, SamplerMode::Sampled]
                        .into_iter()
                        .filter(|mode| decodes(*mode, "text", output))
                        .map(|mode| (*output, mode))
                })
                .collect(),
            json_schema_keywords: profile.json_schema_keywords.clone(),
            max_top_logprobs: profile.max_top_logprobs,
            sampling_params: profile.sampling_params.clone().unwrap_or_default(),
            max_choices: if native {
                profile.decode_batch_max.max(1)
            } else {
                1
            },
            native,
            images_greedy: decodes(SamplerMode::Greedy, "image", "text"),
            images_sampled: decodes(SamplerMode::Sampled, "image", "text"),
            remote_workers,
            image_context_tokens: profile
                .validated_combinations
                .iter()
                .filter(|c| c.operation == "decode" && c.modality == "image")
                .map(|c| c.max_context_tokens)
                .min(),
            image_speculation_greedy: image_op("verify", SamplerMode::Greedy.as_str()),
            image_speculation_sampled: image_op("verify", SamplerMode::Sampled.as_str()),
            image_capture: image_op("prefill", "not_applicable"),
            thinking_toggle: profile.thinking_toggle == Some(true),
            token_controls: profile.token_controls.clone().unwrap_or_default(),
            vocab_size: profile.vocab_size,
        })
    }

    /// Validates a parsed request body. The first failing field is reported. `tools` is the re-serialized subtree;
    /// [`Self::validate_bytes`] keeps the client's bytes.
    pub fn validate(&self, body: &Value) -> Result<ChatRequest, ApiError> {
        self.checked(body, None)
    }

    /// [`Self::validate`] of `body`, parsed from `bytes`, keeping `tools` as the client's exact bytes (D4): a
    /// second parse of the same bytes takes the `tools` subtree as an owned `RawValue`. Every HTTP request goes
    /// here.
    pub fn validate_bytes(&self, body: &Value, bytes: &[u8]) -> Result<ChatRequest, ApiError> {
        self.checked(body, Some(bytes))
    }

    /// ApiLenient D2/D4 (api-lenient design note): drops and checks the inert fields, then rewrites the
    /// aliases of [`MAPPED_PARAMS`] (except `role: developer`, mapped by `message`) to their canonical fields.
    fn canonicalize(&self, canon: &mut Canonical) -> Result<(), ApiError> {
        let object = canon.object;
        let present = |key: &str| object.get(key).filter(|value| !value.is_null());
        for key in INERT_PARAMS.into_iter().filter(|key| !key.contains('.')) {
            if let Some(value) = present(key) {
                inert_value(key, value)?;
                canon.ignored.push(key);
            }
        }
        if let Some(Value::Object(options)) = present("stream_options") {
            if let Some(value) = options.get("include_obfuscation") {
                if !value.is_null() && !value.is_boolean() {
                    return Err(ApiError::invalid_value(
                        "stream_options.include_obfuscation",
                        "stream_options.include_obfuscation must be a boolean",
                    ));
                }
                if !value.is_null() {
                    canon.ignored.push("stream_options.include_obfuscation");
                }
                let mut options = options.clone();
                options.remove("include_obfuscation");
                canon
                    .replaced
                    .insert("stream_options".to_owned(), Value::Object(options));
            }
        }

        let kwargs = match present("chat_template_kwargs") {
            None => None,
            Some(Value::Object(kwargs)) => Some(kwargs),
            // Not an object: `thinking` refuses it; the top-level aliases are checked below all the same.
            Some(_) => None,
        };
        let in_kwargs = |key: &str| {
            kwargs
                .and_then(|kwargs| kwargs.get(key))
                .filter(|value| !value.is_null())
        };
        // (path, value) of each spelling sent, canonical first.
        let spelled = |pairs: &[(&'static str, Option<&Value>)]| {
            pairs
                .iter()
                .filter_map(|(path, value)| value.map(|value| (*path, value.clone())))
                .collect::<Vec<_>>()
        };
        let enable = spelled(&[
            (
                "chat_template_kwargs.enable_thinking",
                in_kwargs("enable_thinking"),
            ),
            ("chat_template_kwargs.thinking", in_kwargs("thinking")),
            ("enable_thinking", present("enable_thinking")),
        ]);
        let preserve = spelled(&[
            (
                "chat_template_kwargs.preserve_thinking",
                in_kwargs("preserve_thinking"),
            ),
            ("preserve_thinking", present("preserve_thinking")),
        ]);
        let effort = spelled(&[
            ("reasoning_effort", present("reasoning_effort")),
            (
                "chat_template_kwargs.reasoning_effort",
                in_kwargs("reasoning_effort"),
            ),
        ]);
        let aliases: Vec<&'static str> = enable
            .iter()
            .chain(&preserve)
            .chain(&effort)
            .map(|(path, _)| *path)
            .filter(|path| MAPPED_PARAMS.contains(path))
            .collect();
        let kwargs_aliased = ["thinking", "preserve_thinking", "reasoning_effort"]
            .into_iter()
            .any(|key| kwargs.is_some_and(|kwargs| kwargs.contains_key(key)));
        if aliases.is_empty() && !kwargs_aliased {
            return Ok(());
        }
        if let Some(alias) = aliases.first().filter(|_| !self.thinking_toggle) {
            return Err(thinking_refused(alias));
        }
        // Booleans, each checked under the path it was sent as.
        for (path, value) in enable.iter().chain(&preserve) {
            if !value.is_boolean() {
                return Err(ApiError::invalid_value(
                    *path,
                    format!("{path} must be a boolean"),
                ));
            }
        }
        for spellings in [&enable, &preserve, &effort] {
            if let Some((first, a)) = spellings.first() {
                if let Some((path, b)) = spellings[1..].iter().find(|(_, b)| b != a) {
                    return Err(conflicting(path, first, b, a));
                }
            }
        }
        // D3: true is the template's default rendering (absent); false is kept for the template (AL-2), and in
        // `chat_template_kwargs` it is the canonical spelling, not an alias.
        let preserve_false = preserve
            .first()
            .is_some_and(|(_, value)| *value == Value::Bool(false));
        let aliases: Vec<&'static str> = aliases
            .into_iter()
            .filter(|path| !(preserve_false && *path == "chat_template_kwargs.preserve_thinking"))
            .collect();
        canon.mapped.extend(aliases);
        if kwargs.is_some() || present("chat_template_kwargs").is_none() {
            let mut kwargs = kwargs.cloned().unwrap_or_default();
            kwargs.remove("thinking");
            kwargs.remove("preserve_thinking");
            kwargs.remove("reasoning_effort");
            if let Some((_, value)) = enable.first() {
                kwargs.insert("enable_thinking".to_owned(), value.clone());
            }
            if preserve_false {
                kwargs.insert("preserve_thinking".to_owned(), Value::Bool(false));
            }
            canon
                .replaced
                .insert("chat_template_kwargs".to_owned(), Value::Object(kwargs));
        }
        if let Some((_, value)) = effort.first() {
            canon
                .replaced
                .insert("reasoning_effort".to_owned(), value.clone());
        }
        Ok(())
    }

    fn checked(&self, body: &Value, bytes: Option<&[u8]>) -> Result<ChatRequest, ApiError> {
        let object = body
            .as_object()
            .ok_or_else(|| ApiError::invalid_json("request body must be a JSON object"))?;
        // ApiLenient D1: the top-level aliases and inert fields never reach the allowlist below.
        let mut canon = Canonical {
            object,
            replaced: Map::new(),
            removed: INERT_PARAMS
                .into_iter()
                .filter(|key| !key.contains('.'))
                .chain(["enable_thinking", "preserve_thinking"])
                .filter(|key| object.contains_key(*key))
                .collect(),
            ignored: Vec::new(),
            mapped: Vec::new(),
        };
        // M2.7 D8: a sampling field the profile does not declare is refused like an unknown one, `null` included,
        // so a profile without `sampling_params` answers exactly as before M2.7.
        let undeclared = |key: &str| {
            SAMPLING_PARAMS.contains(&key) && !self.sampling_params.iter().any(|name| name == key)
        };
        if let Some(key) = canon
            .keys()
            .find(|key| !ALLOWED_FIELDS.contains(&key.as_str()) || undeclared(key))
        {
            if let Some(refusal) = reserved_extension(key) {
                return Err(refusal);
            }
            let message = if undeclared(key) {
                format!("parameter '{key}' is not supported by this deployment's profile (sampling_params)")
            } else if let Some((_, why)) = REJECTED_FIELDS.iter().find(|(name, _)| name == key) {
                (*why).to_owned()
            } else {
                format!("parameter '{key}' is not supported")
            };
            return Err(ApiError::unsupported_parameter(key.as_str(), message));
        }
        self.canonicalize(&mut canon)?;
        let ignored = std::mem::take(&mut canon.ignored);
        let mut mapped = std::mem::take(&mut canon.mapped);
        // `null` means absent for every optional field.
        let field = |key: &str| canon.get(key);

        let model = match field("model") {
            None => return Err(ApiError::missing_parameter("model")),
            Some(Value::String(model)) => model,
            Some(_) => return Err(ApiError::invalid_value("model", "model must be a string")),
        };
        if *model != self.served_model_name {
            return Err(model_not_found(model, &self.served_models));
        }
        let messages = self.messages(field("messages"))?;
        let developer = field("messages")
            .and_then(Value::as_array)
            .is_some_and(|items| {
                items
                    .iter()
                    .any(|item| item.get("role").and_then(Value::as_str) == Some("developer"))
            });
        let n = match field("n") {
            None => 1,
            Some(n) => n
                .as_u64()
                .filter(|n| (1..=self.max_choices).contains(n))
                .ok_or_else(|| {
                    ApiError::unsupported_parameter(
                        "n",
                        if self.max_choices == 1 {
                            "only n=1 is supported".to_owned()
                        } else {
                            format!("n must be an integer from 1 to {}", self.max_choices)
                        },
                    )
                })? as u32,
        };
        let stream = match field("stream") {
            None => false,
            Some(Value::Bool(stream)) => *stream,
            Some(_) => {
                return Err(ApiError::invalid_value(
                    "stream",
                    "stream must be a boolean",
                ))
            }
        };
        let include_usage = stream_options(field("stream_options"), stream)?;
        let max_output_tokens = match (field("max_tokens"), field("max_completion_tokens")) {
            (Some(_), Some(_)) => {
                return Err(ApiError::new(
                    400,
                    INVALID_REQUEST,
                    "conflicting_parameters",
                    "send max_tokens or max_completion_tokens, not both",
                    Some("max_completion_tokens".to_owned()),
                ))
            }
            (Some(value), None) => self.budget("max_tokens", value)?,
            (None, Some(value)) => self.budget("max_completion_tokens", value)?,
            (None, None) => self.default_max_output_tokens,
        };
        let temperature = bounded_number(field("temperature"), "temperature", |t| {
            (0.0..=2.0).contains(&t)
        })?;
        let top_p = bounded_number(field("top_p"), "top_p", |p| p > 0.0 && p <= 1.0)?;
        // Native stop strings are matched in `cered` since M2.4 (D11, `native/stop.rs`).
        let stop = stop_list(field("stop"))?;
        let sampler = if temperature == Some(0.0) {
            SamplerMode::Greedy
        } else {
            SamplerMode::Sampled
        };
        let supported = match sampler {
            SamplerMode::Greedy => self.greedy,
            SamplerMode::Sampled => self.sampled,
        };
        if !supported {
            return Err(ApiError::unsupported_parameter(
                "temperature",
                format!(
                    "the deployment profile has no validated {} decode combination",
                    sampler.as_str()
                ),
            ));
        }
        let image = first_image(&messages);
        if let Some(at) = &image {
            let images = match sampler {
                SamplerMode::Greedy => self.images_greedy,
                SamplerMode::Sampled => self.images_sampled,
            };
            if !images {
                return Err(ApiError::unsupported_parameter(
                    at.as_str(),
                    format!(
                        "the deployment profile has no validated {} image decode combination",
                        sampler.as_str()
                    ),
                ));
            }
        }
        // M2.7 D8: native serves `top_p` (its refusal went with the sampler); the ranges are SGLang's (F7) and
        // `logit_bias` values OpenAI's (Q6). Mapping `top_k` -1 / > vocab to the vocabulary is routing (C6).
        let top_k = match field("top_k") {
            None => None,
            Some(value) => Some(value.as_i64().filter(|k| *k == -1 || *k >= 1).ok_or_else(
                || ApiError::invalid_value("top_k", "top_k must be -1 or an integer >= 1"),
            )?),
        };
        let min_p = bounded_number(field("min_p"), "min_p", |p| (0.0..=1.0).contains(&p))?;
        let seed = match field("seed") {
            None => None,
            Some(value) => Some(value.as_u64().filter(|s| *s < 1 << 63).ok_or_else(|| {
                ApiError::invalid_value("seed", "seed must be an integer from 0 to 2^63-1")
            })?),
        };
        let penalty = |name: &str| bounded_number(field(name), name, |p| (-2.0..=2.0).contains(&p));
        let frequency_penalty = penalty("frequency_penalty")?;
        let presence_penalty = penalty("presence_penalty")?;
        let repetition_penalty =
            bounded_number(field("repetition_penalty"), "repetition_penalty", |r| {
                r > 0.0 && r <= 2.0
            })?;
        let logit_bias = field("logit_bias").map(logit_bias).transpose()?;
        let tools = tools_value(field("tools"))?;
        let tool_choice = tool_choice(field("tool_choice"), tools)?;
        let parallel_tool_calls = match field("parallel_tool_calls") {
            None => None,
            Some(Value::Bool(parallel)) => Some(*parallel),
            Some(_) => {
                return Err(ApiError::invalid_value(
                    "parallel_tool_calls",
                    "parallel_tool_calls must be a boolean",
                ))
            }
        };
        let (constraint_field, strict) = self.constraint(field, tools, &tool_choice, sampler)?;
        if tools.is_some() {
            let supported = match sampler {
                SamplerMode::Greedy => self.tools_greedy,
                SamplerMode::Sampled => self.tools_sampled,
            };
            if !supported {
                return Err(ApiError::unsupported_parameter(
                    "tools",
                    format!(
                        "the deployment profile has no validated {} tool_calls decode combination",
                        sampler.as_str()
                    ),
                ));
            }
        }
        let logprobs = match field("logprobs") {
            None => false,
            Some(Value::Bool(logprobs)) => *logprobs,
            Some(_) => {
                return Err(ApiError::invalid_value(
                    "logprobs",
                    "logprobs must be a boolean",
                ))
            }
        };
        if logprobs && self.max_top_logprobs.is_none() {
            return Err(ApiError::unsupported_parameter(
                "logprobs",
                if self.native {
                    "this deployment does not return logprobs (native logprobs arrive with M2.7)"
                } else {
                    "this deployment's profile does not declare max_top_logprobs: logprobs are not returned"
                },
            ));
        }
        let top_logprobs = match field("top_logprobs") {
            None => None,
            Some(value) => {
                let max = self.max_top_logprobs.filter(|_| logprobs).ok_or_else(|| {
                    ApiError::invalid_value("top_logprobs", "top_logprobs requires logprobs: true")
                })?;
                let top = value
                    .as_u64()
                    .filter(|top| *top <= u64::from(max))
                    .ok_or_else(|| {
                        ApiError::invalid_value(
                            "top_logprobs",
                            format!("top_logprobs must be an integer from 0 to {max}"),
                        )
                    })?;
                Some(top as u32)
            }
        };
        let raw = if ["tools", "tool_choice"]
            .into_iter()
            .chain(CONSTRAINT_FIELDS)
            .any(|name| field(name).is_some())
        {
            match bytes {
                Some(bytes) => serde_json::from_slice::<RawFields>(bytes)
                    .map_err(|e| ApiError::invalid_json(e.to_string()))?,
                None => {
                    let raw = |name: &str| {
                        field(name).map(|value| {
                            serde_json::value::to_raw_value(value).expect("JSON values serialize")
                        })
                    };
                    RawFields {
                        tools: raw("tools"),
                        tool_choice: raw("tool_choice"),
                        response_format: raw("response_format"),
                        regex: raw("regex"),
                        ebnf: raw("ebnf"),
                    }
                }
            }
        } else {
            RawFields::default()
        };
        let constrains = |name: &str| constraint_field == Some(name);
        let thinking = self.thinking(&field)?;
        let stop_token_ids = self.stop_token_ids(field("stop_token_ids"))?;
        let min_tokens = self.min_tokens(field("min_tokens"), max_output_tokens)?;
        let ignore_eos = match field("ignore_eos") {
            None => None,
            Some(_) if !self.controls("ignore_eos") => return Err(self.not_declared("ignore_eos")),
            Some(Value::Bool(ignore)) => Some(*ignore),
            Some(_) => {
                return Err(ApiError::invalid_value(
                    "ignore_eos",
                    "ignore_eos must be a boolean",
                ))
            }
        };
        let thinking_budget =
            self.thinking_budget(field("thinking_budget"), thinking.resolved, logprobs)?;
        if developer {
            mapped.push("messages.role.developer");
        }
        // Both lists in D1 list order (sorted), as the header reports them.
        mapped.sort_unstable();
        Ok(ChatRequest {
            model: model.clone(),
            messages,
            max_output_tokens,
            temperature,
            top_p,
            stop,
            stream,
            include_usage,
            sampler,
            tools: raw.tools.filter(|_| tools.is_some()),
            tool_choice,
            tool_choice_raw: raw.tool_choice,
            strict_tools: !strict.is_empty(),
            response_format: raw
                .response_format
                .filter(|_| constrains("response_format")),
            regex: raw.regex.filter(|_| constrains("regex")),
            ebnf: raw.ebnf.filter(|_| constrains("ebnf")),
            parallel_tool_calls,
            logprobs,
            top_logprobs,
            image_speculation: image.is_some()
                && match sampler {
                    SamplerMode::Greedy => self.image_speculation_greedy,
                    SamplerMode::Sampled => self.image_speculation_sampled,
                },
            image_capture: image.is_some() && self.image_capture,
            image_context_tokens: image.and(self.image_context_tokens),
            top_k,
            min_p,
            seed,
            frequency_penalty,
            presence_penalty,
            repetition_penalty,
            logit_bias,
            enable_thinking: thinking.enable_thinking,
            preserve_thinking: thinking.preserve_thinking,
            reasoning_effort: thinking.reasoning_effort,
            thinking: thinking.resolved,
            separate_reasoning: thinking.separate_reasoning,
            stop_token_ids,
            min_tokens,
            ignore_eos,
            thinking_budget,
            n,
            adapter: None,
            research: None,
            ignored_params: ignored,
            mapped_params: mapped,
        })
    }

    /// The M2.8 constraint of a request (D3/D4/D5/D10): at most one of `response_format`/`regex`/`ebnf`, never
    /// together with a tool constraint, gated by the profile's `decode` combination for its grammar output, then
    /// the D5 walk of its schemas. Returns the constraint field and the strict tools' indices.
    fn constraint<'a>(
        &self,
        field: impl Fn(&str) -> Option<&'a Value>,
        tools: Option<&Value>,
        tool_choice: &ToolChoice,
        sampler: SamplerMode,
    ) -> Result<(Option<&'static str>, Vec<usize>), ApiError> {
        let mut present = Vec::new();
        for name in CONSTRAINT_FIELDS {
            let Some(value) = field(name) else {
                continue;
            };
            let constrains = if name == "response_format" {
                constraint::response_format(value)?
            } else {
                constraint::grammar_text(value, name)?;
                true
            };
            if constrains {
                present.push(name);
            }
        }
        if let [_, .., later] = present[..] {
            return Err(ApiError::invalid_value(
                later,
                "at most one of response_format, regex and ebnf may constrain a request",
            ));
        }
        let constraint_field = present.first().copied();
        let strict: Vec<usize> = tools
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
            .filter(|(_, tool)| tool.pointer("/function/strict") == Some(&Value::Bool(true)))
            .map(|(index, _)| index)
            .collect();
        let tool_constrained = tools.is_some()
            && match tool_choice {
                ToolChoice::Required | ToolChoice::Named(_) => true,
                ToolChoice::Auto => !strict.is_empty(),
                ToolChoice::None => false,
            };
        // N1: a tool constraint and an output constraint never combine.
        if tool_constrained && constraint_field.is_some() {
            return Err(ApiError::invalid_value(
                "tool_choice",
                "a tool constraint (tool_choice required or named, or a strict tool) cannot be combined with \
                 response_format, regex or ebnf",
            ));
        }
        let (param, output) = match constraint_field {
            Some(name) => (name, grammar_output_of(name)),
            None if tool_constrained => ("tool_choice", TOOL_CALLS_CONSTRAINED),
            None => return Ok((None, strict)),
        };
        if !self
            .grammar_decodes
            .iter()
            .any(|(o, mode)| *o == output && *mode == sampler)
        {
            return Err(ApiError::unsupported_parameter(
                param,
                format!(
                    "the deployment profile has no validated {} {output} decode combination",
                    sampler.as_str()
                ),
            ));
        }
        if constraint_field == Some("response_format") {
            if let Some(schema) =
                field("response_format").and_then(|format| format.pointer("/json_schema/schema"))
            {
                // Number ranges: only a native deployment's grammar sidecar compiles them exactly (B7).
                constraint::walk_schema(
                    schema,
                    &self.json_schema_keywords,
                    SCHEMA_PARAM,
                    self.native,
                )?;
            }
        }
        if tool_constrained {
            let items = tools
                .and_then(Value::as_array)
                .map_or(&[][..], Vec::as_slice);
            for &index in &strict {
                if let Some(parameters) = items[index]
                    .pointer("/function/parameters")
                    .filter(|p| !p.is_null())
                {
                    constraint::walk_schema(
                        parameters,
                        &self.json_schema_keywords,
                        &format!("tools[{index}].function.parameters"),
                        false,
                    )?;
                }
            }
        }
        Ok((constraint_field, strict))
    }

    /// Whether the profile's `token_controls` lists `control` (M2.12 D2).
    fn controls(&self, control: &str) -> bool {
        self.token_controls.iter().any(|c| c == control)
    }

    /// 400 `unsupported_parameter` for an M2.12 field the profile does not declare (D2), naming the milestone that
    /// brings it: M2.7 for native `min_tokens` (D10), M2.12 otherwise.
    fn not_declared(&self, param: &str) -> ApiError {
        let message = if self.native && param == "min_tokens" {
            "native min_tokens needs the M2.7 logits hook; it is not supported yet".to_owned()
        } else {
            format!("this deployment's profile does not declare {param} (M2.12)")
        };
        ApiError::unsupported_parameter(param, message)
    }

    /// M2.12 D1/D4/D13: `chat_template_kwargs` (`enable_thinking`, and ApiLenient AL-2's `preserve_thinking`, false
    /// after canonicalization), `separate_reasoning`, `reasoning_effort`, all gated by the profile's
    /// `thinking_toggle`, and the resolved thinking flag.
    fn thinking<'a>(
        &self,
        field: &impl Fn(&str) -> Option<&'a Value>,
    ) -> Result<Thinking, ApiError> {
        let refused = thinking_refused;
        let (enable_thinking, preserve_thinking) = match field("chat_template_kwargs") {
            None => (None, None),
            Some(Value::Object(kwargs)) => {
                if let Some(key) = kwargs
                    .keys()
                    .find(|key| !matches!(key.as_str(), "enable_thinking" | "preserve_thinking"))
                {
                    return Err(ApiError::unsupported_parameter(
                        format!("chat_template_kwargs.{key}"),
                        format!(
                            "chat_template_kwargs.{key} is not read by this deployment's chat template; it reads \
                             enable_thinking (alias thinking), reasoning_effort and preserve_thinking"
                        ),
                    ));
                }
                if !kwargs.is_empty() && !self.thinking_toggle {
                    return Err(refused("chat_template_kwargs"));
                }
                let flag = |key: &str| match kwargs.get(key) {
                    None => Ok(None),
                    Some(Value::Bool(value)) => Ok(Some(*value)),
                    Some(_) => Err(ApiError::invalid_value(
                        format!("chat_template_kwargs.{key}"),
                        format!("{key} must be a boolean"),
                    )),
                };
                // D3: `preserve_thinking: true` is the template's default (absent).
                (
                    flag("enable_thinking")?,
                    flag("preserve_thinking")?.filter(|preserve| !preserve),
                )
            }
            Some(_) => {
                return Err(ApiError::invalid_value(
                    "chat_template_kwargs",
                    "chat_template_kwargs must be an object",
                ))
            }
        };
        let separate_reasoning = match field("separate_reasoning") {
            None => None,
            Some(_) if !self.thinking_toggle => return Err(refused("separate_reasoning")),
            Some(Value::Bool(separate)) => Some(*separate),
            Some(_) => {
                return Err(ApiError::invalid_value(
                    "separate_reasoning",
                    "separate_reasoning must be a boolean",
                ))
            }
        };
        let reasoning_effort = match field("reasoning_effort") {
            None => None,
            Some(_) if !self.thinking_toggle => return Err(refused("reasoning_effort")),
            Some(effort) => {
                let accepted = match effort {
                    Value::String(name) => REASONING_EFFORTS.contains(&name.as_str()),
                    Value::Number(number) => number
                        .as_f64()
                        .is_some_and(|n| (0.0..=MAX_REASONING_EFFORT).contains(&n)),
                    _ => false,
                };
                if !accepted {
                    return Err(ApiError::invalid_value(
                        "reasoning_effort",
                        format!(
                            "reasoning_effort must be one of {} or a number from 0 to {MAX_REASONING_EFFORT}",
                            REASONING_EFFORTS.join(", ")
                        ),
                    ));
                }
                Some(effort.clone())
            }
        };
        let resolved = enable_thinking.unwrap_or_else(|| {
            reasoning_effort
                .as_ref()
                .is_some_and(|effort| effort.as_str() != Some("none"))
        });
        if resolved {
            if let Some(effort) = &reasoning_effort {
                if !effort
                    .as_str()
                    .is_some_and(|name| THINKING_EFFORTS.contains(&name))
                {
                    return Err(ApiError::invalid_value(
                        "reasoning_effort",
                        format!(
                            "with thinking on, the model's chat template accepts reasoning_effort {} only",
                            THINKING_EFFORTS.join(", ")
                        ),
                    ));
                }
            }
        }
        Ok(Thinking {
            enable_thinking,
            preserve_thinking,
            reasoning_effort,
            separate_reasoning,
            resolved,
        })
    }

    /// ApiFit AF-L7 (api-fit design note): `thinking_budget`, a CERE extension: the most output tokens the
    /// thinking may take, `</think>` included, on a native profile with `thinking_toggle` and thinking resolved on.
    /// Not with `logprobs`: the closing tokens CERE inserts at the budget are not sampled.
    fn thinking_budget(
        &self,
        value: Option<&Value>,
        thinking: bool,
        logprobs: bool,
    ) -> Result<Option<u64>, ApiError> {
        let Some(value) = value else {
            return Ok(None);
        };
        if !self.thinking_toggle || !self.native {
            return Err(ApiError::unsupported_parameter(
                "thinking_budget",
                "thinking_budget needs a native deployment whose profile declares thinking_toggle",
            ));
        }
        let budget = value.as_u64().filter(|b| *b >= 1).ok_or_else(|| {
            ApiError::invalid_value(
                "thinking_budget",
                "thinking_budget must be a positive integer",
            )
        })?;
        if !thinking {
            return Err(ApiError::invalid_value(
                "thinking_budget",
                "thinking_budget needs thinking on (chat_template_kwargs.enable_thinking: true)",
            ));
        }
        if logprobs {
            return Err(ApiError::unsupported_parameter(
                "thinking_budget",
                "thinking_budget is not supported with logprobs",
            ));
        }
        Ok(Some(budget))
    }

    /// M2.12 D1/D5: `stop_token_ids`, 0..=32 integers below the profile's `vocab_size`, as a sorted set.
    fn stop_token_ids(&self, value: Option<&Value>) -> Result<Option<Vec<u32>>, ApiError> {
        let Some(value) = value else {
            return Ok(None);
        };
        let vocab = match self.vocab_size {
            Some(vocab) if self.controls("stop_token_ids") => vocab,
            _ => return Err(self.not_declared("stop_token_ids")),
        };
        let invalid = || {
            ApiError::invalid_value(
                "stop_token_ids",
                format!(
                    "stop_token_ids must be an array of at most {MAX_STOP_TOKEN_IDS} integers from 0 to {}",
                    vocab - 1
                ),
            )
        };
        let items = value.as_array().ok_or_else(invalid)?;
        if items.len() > MAX_STOP_TOKEN_IDS {
            return Err(invalid());
        }
        let ids = items
            .iter()
            .map(|item| {
                item.as_u64()
                    .filter(|id| *id < u64::from(vocab))
                    .map(|id| id as u32)
                    .ok_or_else(invalid)
            })
            .collect::<Result<std::collections::BTreeSet<u32>, _>>()?;
        Ok(Some(ids.into_iter().collect()))
    }

    /// M2.12 D1/D10: `min_tokens`, an integer from 0 to the request's output budget.
    fn min_tokens(&self, value: Option<&Value>, budget: u64) -> Result<Option<u64>, ApiError> {
        let Some(value) = value else {
            return Ok(None);
        };
        if self.native || !self.controls("min_tokens") {
            return Err(self.not_declared("min_tokens"));
        }
        value
            .as_u64()
            .filter(|min| *min <= budget)
            .map(Some)
            .ok_or_else(|| {
                ApiError::invalid_value(
                    "min_tokens",
                    format!("min_tokens must be an integer from 0 to the output budget {budget}"),
                )
            })
    }

    fn messages(&self, value: Option<&Value>) -> Result<Vec<ChatMessage>, ApiError> {
        let items = match value {
            None => return Err(ApiError::missing_parameter("messages")),
            Some(Value::Array(items)) => items,
            Some(_) => {
                return Err(ApiError::invalid_value(
                    "messages",
                    "messages must be an array",
                ))
            }
        };
        if items.is_empty() {
            return Err(ApiError::invalid_value(
                "messages",
                "messages must not be empty",
            ));
        }
        if items.len() as u64 > self.max_messages {
            return Err(ApiError::invalid_value(
                "messages",
                format!("at most {} messages are allowed", self.max_messages),
            ));
        }
        let mut images = ImageBudget {
            policy: self,
            count: 0,
            tokens: 0,
            documents: document::Budget::default(),
        };
        items
            .iter()
            .enumerate()
            .map(|(index, item)| message(index, item, &mut images).map(|m| in_place(index, m)))
            .collect()
    }

    fn budget(&self, name: &str, value: &Value) -> Result<u64, ApiError> {
        let tokens = value
            .as_u64()
            .filter(|tokens| *tokens >= 1)
            .ok_or_else(|| {
                ApiError::invalid_value(name, format!("{name} must be a positive integer"))
            })?;
        if tokens > self.max_output_tokens {
            return Err(ApiError::invalid_value(
                name,
                format!(
                    "{name} {tokens} exceeds the maximum of {}",
                    self.max_output_tokens
                ),
            ));
        }
        Ok(tokens)
    }
}

/// The marks around a mid-conversation system message's content (ApiFit AF-L1, api-fit design note).
pub const MID_SYSTEM_OPEN: &str = "<system-message>\n";
pub const MID_SYSTEM_CLOSE: &str = "\n</system-message>";

/// ApiFit AF-L1: the pinned chat template accepts a system message only as `messages[0]` and raises on any later one
/// (F1; SGLang and vLLM refuse it too, F3). A system message at a later index renders in place as a user message
/// holding its content between [`MID_SYSTEM_OPEN`] and [`MID_SYSTEM_CLOSE`]: the prompt before it is unchanged (a
/// valid cached prefix, F7) and the marks carry its system authority. A CERE extension; every other message, and a
/// request whose only system message is first, is unchanged. Folding is idempotent (the result is a user message).
fn in_place(index: usize, message: ChatMessage) -> ChatMessage {
    match &message.content {
        Some(Content::Text(text)) if index > 0 && message.role == Role::System => {
            ChatMessage::text(
                Role::User,
                format!("{MID_SYSTEM_OPEN}{text}{MID_SYSTEM_CLOSE}"),
            )
        }
        _ => message,
    }
}

/// One message (D3). system/user `{role, content: string, name?}`; assistant `{role, content: string|null,
/// tool_calls?, reasoning_content?, name?}` (`content: null` needs calls; `reasoning_content`, `tool_calls` and
/// `name` may be `null` = absent); tool `{role, content: string, tool_call_id, name?}`. O113 (owner 2026-10-17;
/// tool messages: Main 2026-10-17): a string `reasoning_content` is kept for the chat template (`""` = absent, which
/// renders the same); `name` is accepted and dropped on every role (the pinned template never reads it).
fn message(index: usize, item: &Value, images: &mut ImageBudget) -> Result<ChatMessage, ApiError> {
    const KEYS: [&str; 6] = [
        "content",
        "name",
        "reasoning_content",
        "role",
        "tool_call_id",
        "tool_calls",
    ];
    let path = |key: &str| format!("messages[{index}].{key}");
    let object = item.as_object().ok_or_else(|| {
        ApiError::invalid_value(format!("messages[{index}]"), "a message must be an object")
    })?;
    if let Some(key) = object.keys().find(|key| !KEYS.contains(&key.as_str())) {
        return Err(ApiError::unsupported_parameter(
            path(key),
            format!("message field '{key}' is not supported"),
        ));
    }
    let role = match object.get("role") {
        None | Some(Value::Null) => return Err(ApiError::missing_parameter(path("role"))),
        Some(Value::String(role)) => match role.as_str() {
            // ApiLenient Q5: `developer` is OpenAI's newer name for system (the Responses surface maps it too).
            "system" | "developer" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => {
                return Err(ApiError::unsupported_parameter(
                    path("role"),
                    format!("role '{other}' is not supported; use system (or developer), user, assistant or tool"),
                ))
            }
        },
        Some(_) => {
            return Err(ApiError::invalid_value(
                path("role"),
                "role must be a string",
            ))
        }
    };
    let allowed: &[&str] = match role {
        Role::System | Role::User => &["content", "name", "role"],
        Role::Assistant => &["content", "name", "reasoning_content", "role", "tool_calls"],
        Role::Tool => &["content", "name", "role", "tool_call_id"],
    };
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(ApiError::unsupported_parameter(
            path(key),
            format!("message field '{key}' is not supported for this role"),
        ));
    }
    if !matches!(
        object.get("name"),
        None | Some(Value::Null | Value::String(_))
    ) {
        return Err(ApiError::invalid_value(
            path("name"),
            "name must be a string",
        ));
    }
    let reasoning_content = match object.get("reasoning_content") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) => (!text.is_empty()).then(|| text.clone()),
        Some(_) => {
            return Err(ApiError::invalid_value(
                path("reasoning_content"),
                "reasoning_content must be a string",
            ))
        }
    };
    let tool_calls = match object.get("tool_calls") {
        None | Some(Value::Null) => None,
        Some(Value::Array(calls)) if !calls.is_empty() => Some(
            calls
                .iter()
                .enumerate()
                .map(|(call, item)| tool_call(&path(&format!("tool_calls[{call}]")), item))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Some(_) => {
            return Err(ApiError::invalid_value(
                path("tool_calls"),
                "tool_calls must be a non-empty array",
            ))
        }
    };
    let tool_call_id = match (role, object.get("tool_call_id")) {
        (Role::Tool, None | Some(Value::Null)) => {
            return Err(ApiError::missing_parameter(path("tool_call_id")))
        }
        (Role::Tool, Some(Value::String(id))) => Some(id.clone()),
        (Role::Tool, Some(_)) => {
            return Err(ApiError::invalid_value(
                path("tool_call_id"),
                "tool_call_id must be a string",
            ))
        }
        _ => None,
    };
    let content = match object.get("content") {
        None => return Err(ApiError::missing_parameter(path("content"))),
        Some(Value::String(content)) => Some(Content::Text(content.clone())),
        Some(Value::Null) if role == Role::Assistant && tool_calls.is_some() => None,
        Some(Value::Array(parts)) if role == Role::User => Some(Content::Parts(content_parts(
            &path("content"),
            parts,
            images,
        )?)),
        // A tool result's parts: VisionLong C7 (screenshots, kept as parts and rendered inside its
        // `<tool_response>`) and DocInput DI-02 (text and file parts, a document a tool read). An array without an
        // image part renders as the joined string, exactly as a string content would.
        Some(Value::Array(parts)) if role == Role::Tool => {
            let parts = content_parts(&path("content"), parts, images)?;
            if parts
                .iter()
                .any(|part| matches!(part, ContentPart::ImageUrl { .. }))
            {
                Some(Content::Parts(parts))
            } else {
                let mut text = String::new();
                for part in parts {
                    if let ContentPart::Text { text: t } = part {
                        text.push_str(&t);
                    }
                }
                Some(Content::Text(text))
            }
        }
        Some(Value::Array(_)) => {
            return Err(ApiError::unsupported_parameter(
                path("content"),
                "a content array is accepted only in user and tool messages; this role's content is a string",
            ))
        }
        Some(_) => {
            return Err(ApiError::unsupported_parameter(
                path("content"),
                "message content is a string or (user and tool messages) an array of parts (null only with tool_calls)",
            ))
        }
    };
    Ok(ChatMessage {
        role,
        content,
        reasoning_content,
        tool_calls,
        tool_call_id,
    })
}

/// The images a request has accepted so far (m213 D4: counted across the whole history) and the D5/Q3 gate.
struct ImageBudget<'a> {
    policy: &'a ChatPolicy,
    count: usize,
    tokens: u32,
    /// DocInput DI-05: the pages and rendered text of the request's documents (history included).
    documents: document::Budget,
}

impl ImageBudget<'_> {
    /// D5/Q3 (a): refused before the URL is looked at, so the reason is the deployment, not the image.
    fn admit(&self, at: &str) -> Result<(), ApiError> {
        if self.policy.remote_workers {
            return Err(ApiError::unsupported_parameter(
                at,
                "images are not accepted on a deployment with federation workers (image bytes stay on this host)",
            ));
        }
        if !(self.policy.images_greedy || self.policy.images_sampled) {
            return Err(ApiError::unsupported_parameter(
                at,
                "the deployment profile has no validated image decode combination",
            ));
        }
        Ok(())
    }

    /// D4 / VisionLong C2: at most [`image::MAX_IMAGES`] images and [`image::MAX_TOKENS_PER_IMAGE`] tokens each;
    /// their tokens together fit the profile's image context (the prompt check then counts the text too).
    fn count(&mut self, at: &str, image: &ImageInfo) -> Result<(), ApiError> {
        if self.count == image::MAX_IMAGES {
            return Err(ApiError::invalid_value(
                at,
                format!(
                    "at most {} images are allowed per request (history included)",
                    image::MAX_IMAGES
                ),
            ));
        }
        let tokens = image.tokens();
        if tokens > image::MAX_TOKENS_PER_IMAGE {
            return Err(ApiError::invalid_value(
                at,
                format!(
                    "the image needs {tokens} tokens ({}x{} after resize); at most {} are allowed per image",
                    image.resized_width,
                    image.resized_height,
                    image::MAX_TOKENS_PER_IMAGE
                ),
            ));
        }
        let context = self.policy.image_context_tokens.unwrap_or(0);
        if u64::from(self.tokens) + u64::from(tokens) > context {
            return Err(ApiError::invalid_value(
                at,
                format!(
                    "the images need {} tokens; the context holds {context}",
                    self.tokens + tokens,
                ),
            ));
        }
        self.count += 1;
        self.tokens += tokens;
        Ok(())
    }
}

/// A user or tool message's parts array (m213 D2): 1..=64 parts, each `{type: "text", text}`, `{type: "image_url",
/// image_url: {url, detail?}}` or (DocInput DI-02) `{type: "file", file: {file_data, filename?}}`, which becomes a
/// text part holding the rendered document; every other type or key is refused with its path.
fn content_parts(
    at: &str,
    parts: &[Value],
    images: &mut ImageBudget,
) -> Result<Vec<ContentPart>, ApiError> {
    if parts.is_empty() || parts.len() > image::MAX_PARTS {
        return Err(ApiError::invalid_value(
            at,
            format!("a content array holds 1 to {} parts", image::MAX_PARTS),
        ));
    }
    let mut out: Vec<ContentPart> = Vec::with_capacity(parts.len());
    for (index, part) in parts.iter().enumerate() {
        let at = format!("{at}[{index}]");
        let part = match part.get("type").and_then(Value::as_str) {
            Some("file") => {
                let rendered = file_part(&at, part, &mut images.documents)?;
                let previous = match out.last() {
                    Some(ContentPart::Text { text }) => text.as_str(),
                    _ => "",
                };
                ContentPart::Text {
                    text: document::after(previous, rendered),
                }
            }
            _ => content_part(&at, part, images)?,
        };
        out.push(part);
    }
    Ok(out)
}

/// DocInput DI-02: `{type: "file", file: {file_data, filename?}}` with a `data:application/pdf;base64,` or
/// `data:text/plain;base64,` URL, rendered to text ([`document`]); `filename` is the document's title. Uploaded
/// files (`file_id`) are refused: CERE has no Files API.
fn file_part(at: &str, part: &Value, documents: &mut document::Budget) -> Result<String, ApiError> {
    let path = |key: &str| format!("{at}.{key}");
    let object = part
        .as_object()
        .ok_or_else(|| ApiError::invalid_value(at, "a content part must be an object"))?;
    if let Some(key) = object
        .keys()
        .find(|key| !["file", "type"].contains(&key.as_str()))
    {
        return Err(ApiError::unsupported_parameter(
            path(key),
            format!("content part field '{key}' is not supported"),
        ));
    }
    let at = path("file");
    let path = |key: &str| format!("{at}.{key}");
    let file = match object.get("file") {
        None | Some(Value::Null) => return Err(ApiError::missing_parameter(&at)),
        Some(Value::Object(file)) => file,
        Some(_) => {
            return Err(ApiError::invalid_value(
                &at,
                "file must be an object {file_data, filename?}",
            ))
        }
    };
    if file.contains_key("file_id") {
        return Err(ApiError::unsupported_parameter(
            path("file_id"),
            "uploaded files (file_id) are not supported; send the file as file_data (a base64 data URL)",
        ));
    }
    if let Some(key) = file
        .keys()
        .find(|key| !["file_data", "filename"].contains(&key.as_str()))
    {
        return Err(ApiError::unsupported_parameter(
            path(key),
            format!("file field '{key}' is not supported"),
        ));
    }
    let title = match file.get("filename") {
        None | Some(Value::Null) => None,
        Some(Value::String(name)) => Some(name.as_str()),
        Some(_) => {
            return Err(ApiError::invalid_value(
                path("filename"),
                "filename must be a string",
            ))
        }
    };
    let data = match file.get("file_data") {
        None | Some(Value::Null) => return Err(ApiError::missing_parameter(path("file_data"))),
        Some(Value::String(data)) => data,
        Some(_) => {
            return Err(ApiError::invalid_value(
                path("file_data"),
                "file_data must be a string",
            ))
        }
    };
    let at = path("file_data");
    let (media, payload) = document::data_url(data).map_err(|e| document_error(&at, e))?;
    documents
        .base64(
            media,
            payload,
            document::Meta {
                title,
                context: None,
            },
        )
        .map_err(|e| document_error(&at, e))
}

/// A document the deployment cannot read is unsupported; one that is broken or over a limit is an invalid value.
fn document_error(at: &str, e: document::DocumentError) -> ApiError {
    if e.is_unsupported() {
        ApiError::unsupported_parameter(at, e.to_string())
    } else {
        ApiError::invalid_value(at, e.to_string())
    }
}

fn content_part(at: &str, part: &Value, images: &mut ImageBudget) -> Result<ContentPart, ApiError> {
    let path = |key: &str| format!("{at}.{key}");
    let object = part
        .as_object()
        .ok_or_else(|| ApiError::invalid_value(at, "a content part must be an object"))?;
    let kind = match object.get("type") {
        None | Some(Value::Null) => return Err(ApiError::missing_parameter(path("type"))),
        Some(Value::String(kind)) => kind.as_str(),
        Some(_) => {
            return Err(ApiError::invalid_value(
                path("type"),
                "type must be a string",
            ))
        }
    };
    let allowed = match kind {
        "text" => ["text", "type"],
        "image_url" => ["image_url", "type"],
        other => {
            return Err(ApiError::unsupported_parameter(
                path("type"),
                format!(
                    "content part type '{other}' is not supported; use text, image_url or file"
                ),
            ))
        }
    };
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(ApiError::unsupported_parameter(
            path(key),
            format!("content part field '{key}' is not supported"),
        ));
    }
    if kind == "text" {
        return match object.get("text") {
            None | Some(Value::Null) => Err(ApiError::missing_parameter(path("text"))),
            Some(Value::String(text)) => Ok(ContentPart::Text { text: text.clone() }),
            Some(_) => Err(ApiError::invalid_value(
                path("text"),
                "text must be a string",
            )),
        };
    }
    images.admit(at)?;
    let at = path("image_url");
    let path = |key: &str| format!("{at}.{key}");
    let object = match object.get("image_url") {
        None | Some(Value::Null) => return Err(ApiError::missing_parameter(&at)),
        Some(Value::Object(object)) => object,
        Some(_) => {
            return Err(ApiError::invalid_value(
                &at,
                "image_url must be an object {url, detail?}",
            ))
        }
    };
    if let Some(key) = object
        .keys()
        .find(|key| !["detail", "url"].contains(&key.as_str()))
    {
        return Err(ApiError::unsupported_parameter(
            path(key),
            format!("image_url field '{key}' is not supported"),
        ));
    }
    let url = match object.get("url") {
        None | Some(Value::Null) => return Err(ApiError::missing_parameter(path("url"))),
        Some(Value::String(url)) => url,
        Some(_) => return Err(ApiError::invalid_value(path("url"), "url must be a string")),
    };
    let detail = match object.get("detail") {
        None | Some(Value::Null) => None,
        Some(Value::String(detail)) if detail == "auto" => Some(detail.clone()),
        Some(Value::String(_)) => {
            return Err(ApiError::unsupported_parameter(
                path("detail"),
                "only detail \"auto\" is supported",
            ))
        }
        Some(_) => {
            return Err(ApiError::invalid_value(
                path("detail"),
                "detail must be a string",
            ))
        }
    };
    let image = image::parse_data_url(url)
        .map_err(|e| ApiError::invalid_value(path("url"), e.to_string()))?;
    images.count(&path("url"), &image)?;
    Ok(ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: url.clone(),
            detail,
            image,
        },
    })
}

/// The path of the request's first image part, if any.
pub fn first_image(messages: &[ChatMessage]) -> Option<String> {
    messages
        .iter()
        .enumerate()
        .find_map(|(index, m)| match &m.content {
            Some(Content::Parts(parts)) => parts
                .iter()
                .position(|p| matches!(p, ContentPart::ImageUrl { .. }))
                .map(|part| format!("messages[{index}].content[{part}]")),
            _ => None,
        })
}

/// One `tool_calls[i]` of an assistant message (D3): `{id, index?, type?: "function", function: {name,
/// arguments}}`; `arguments` must parse to an object and is kept as the exact string.
fn tool_call(at: &str, item: &Value) -> Result<ToolCall, ApiError> {
    let path = |key: &str| format!("{at}.{key}");
    let object = item
        .as_object()
        .ok_or_else(|| ApiError::invalid_value(at, "a tool call must be an object"))?;
    known_keys(object, at, &["function", "id", "index", "type"])?;
    let id = required_string(object, &path("id"), "id")?;
    if let Some(index) = object.get("index").filter(|v| !v.is_null()) {
        if index.as_u64().is_none() {
            return Err(ApiError::invalid_value(
                path("index"),
                "index must be an integer >= 0",
            ));
        }
    }
    match object.get("type") {
        None | Some(Value::Null) => {}
        Some(Value::String(kind)) if kind == "function" => {}
        Some(_) => {
            return Err(ApiError::invalid_value(
                path("type"),
                "type must be \"function\"",
            ))
        }
    }
    let at = path("function");
    let path = |key: &str| format!("{at}.{key}");
    let function = match object.get("function") {
        None | Some(Value::Null) => return Err(ApiError::missing_parameter(at)),
        Some(Value::Object(function)) => function,
        Some(_) => return Err(ApiError::invalid_value(at, "function must be an object")),
    };
    known_keys(function, &at, &["arguments", "name"])?;
    let name = required_string(function, &path("name"), "name")?;
    let arguments = required_string(function, &path("arguments"), "arguments")?;
    if !matches!(
        serde_json::from_str::<Value>(&arguments),
        Ok(Value::Object(_))
    ) {
        return Err(ApiError::invalid_value(
            path("arguments"),
            "arguments must be a JSON object string",
        ));
    }
    Ok(ToolCall {
        id,
        kind: "function".to_owned(),
        function: ToolCallFunction { name, arguments },
    })
}

/// `tools` (D2), checked shape by shape; the subtree itself is what the request keeps.
fn tools_value(value: Option<&Value>) -> Result<Option<&Value>, ApiError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let items = match value.as_array() {
        Some(items) if (1..=MAX_TOOLS).contains(&items.len()) => items,
        _ => {
            return Err(ApiError::invalid_value(
                "tools",
                format!("tools must be an array of 1 to {MAX_TOOLS} tools"),
            ))
        }
    };
    let mut names = std::collections::HashSet::new();
    for (index, item) in items.iter().enumerate() {
        let at = format!("tools[{index}]");
        let path = |key: &str| format!("{at}.{key}");
        let object = item
            .as_object()
            .ok_or_else(|| ApiError::invalid_value(&at, "a tool must be an object"))?;
        known_keys(object, &at, &["function", "type"])?;
        match object.get("type") {
            None | Some(Value::Null) => return Err(ApiError::missing_parameter(path("type"))),
            Some(Value::String(kind)) if kind == "function" => {}
            Some(_) => {
                return Err(ApiError::invalid_value(
                    path("type"),
                    "type must be \"function\"",
                ))
            }
        }
        let at = path("function");
        let path = |key: &str| format!("{at}.{key}");
        let function = match object.get("function") {
            None | Some(Value::Null) => return Err(ApiError::missing_parameter(at)),
            Some(Value::Object(function)) => function,
            Some(_) => return Err(ApiError::invalid_value(at, "function must be an object")),
        };
        known_keys(
            function,
            &at,
            &["description", "name", "parameters", "strict"],
        )?;
        let name = required_string(function, &path("name"), "name")?;
        let valid = (1..=MAX_TOOL_NAME_CHARS).contains(&name.len())
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
        if !valid {
            return Err(ApiError::invalid_value(
                path("name"),
                format!("name must match ^[A-Za-z0-9_-]{{1,{MAX_TOOL_NAME_CHARS}}}$"),
            ));
        }
        if !names.insert(name) {
            return Err(ApiError::invalid_value(
                path("name"),
                "tool names must be unique",
            ));
        }
        match function.get("description") {
            None | Some(Value::Null | Value::String(_)) => {}
            Some(_) => {
                return Err(ApiError::invalid_value(
                    path("description"),
                    "description must be a string",
                ))
            }
        }
        match function.get("parameters") {
            None | Some(Value::Null) => {}
            Some(parameters @ Value::Object(_))
                if constraint::depth(parameters) <= MAX_PARAMETERS_NESTING => {}
            Some(_) => {
                return Err(ApiError::invalid_value(
                    path("parameters"),
                    format!(
                        "parameters must be a JSON schema object nested at most \
                         {MAX_PARAMETERS_NESTING} deep"
                    ),
                ))
            }
        }
        // D2: a boolean (SGLang's `Function.strict: bool` refuses null); M2.8 D4: `strict: true` constrains the
        // call to the tool's parameters schema.
        match function.get("strict") {
            None | Some(Value::Bool(_)) => {}
            Some(_) => {
                return Err(ApiError::invalid_value(
                    path("strict"),
                    "strict must be a boolean",
                ))
            }
        }
    }
    Ok(Some(value))
}

fn required_string(object: &Map<String, Value>, path: &str, key: &str) -> Result<String, ApiError> {
    match object.get(key) {
        None | Some(Value::Null) => Err(ApiError::missing_parameter(path)),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(ApiError::invalid_value(
            path,
            format!("{key} must be a string"),
        )),
    }
}

fn stream_options(value: Option<&Value>, stream: bool) -> Result<bool, ApiError> {
    let Some(value) = value else {
        return Ok(false);
    };
    let options = value.as_object().ok_or_else(|| {
        ApiError::invalid_value("stream_options", "stream_options must be an object")
    })?;
    if !stream {
        return Err(ApiError::invalid_value(
            "stream_options",
            "stream_options is only allowed when stream is true",
        ));
    }
    if let Some(key) = options.keys().find(|key| *key != "include_usage") {
        return Err(ApiError::unsupported_parameter(
            format!("stream_options.{key}"),
            format!("stream option '{key}' is not supported"),
        ));
    }
    match options.get("include_usage") {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(include)) => Ok(*include),
        Some(_) => Err(ApiError::invalid_value(
            "stream_options.include_usage",
            "include_usage must be a boolean",
        )),
    }
}

fn bounded_number(
    value: Option<&Value>,
    name: &str,
    in_range: impl Fn(f64) -> bool,
) -> Result<Option<f64>, ApiError> {
    let Some(value) = value else {
        return Ok(None);
    };
    value
        .as_f64()
        .filter(|number| in_range(*number))
        .map(Some)
        .ok_or_else(|| ApiError::invalid_value(name, format!("{name} is out of range")))
}

/// `logit_bias` (D8, Q6): an object of canonical decimal token ids (`0` or no leading zero, a `u32`) to numbers in
/// [-100, 100]. Whether an id is below the vocabulary is not known here (the validator holds no vocabulary): SGLang
/// checks it (F7) and the native route maps it (C6).
fn logit_bias(value: &Value) -> Result<Map<String, Value>, ApiError> {
    let invalid = |message: String| ApiError::invalid_value("logit_bias", message);
    let Value::Object(bias) = value else {
        return Err(invalid(
            "logit_bias must be an object of token ids to numbers".to_owned(),
        ));
    };
    for (id, value) in bias {
        let canonical = id.bytes().all(|b| b.is_ascii_digit())
            && (id == "0" || !id.starts_with('0'))
            && id.parse::<u32>().is_ok();
        if !canonical {
            return Err(invalid(format!(
                "logit_bias key '{id}' must be a decimal token id"
            )));
        }
        if !value
            .as_f64()
            .is_some_and(|v| (-MAX_LOGIT_BIAS..=MAX_LOGIT_BIAS).contains(&v))
        {
            return Err(invalid(format!(
                "logit_bias['{id}'] must be a number from -100 to 100"
            )));
        }
    }
    Ok(bias.clone())
}

fn stop_list(value: Option<&Value>) -> Result<Vec<String>, ApiError> {
    let non_empty = |text: &str, param: String| {
        if text.is_empty() {
            Err(ApiError::invalid_value(
                param,
                "stop strings must not be empty",
            ))
        } else if text.len() > MAX_STOP_BYTES {
            Err(ApiError::invalid_value(
                param,
                format!("stop strings must be at most {MAX_STOP_BYTES} bytes"),
            ))
        } else {
            Ok(text.to_owned())
        }
    };
    match value {
        None => Ok(Vec::new()),
        Some(Value::String(stop)) => Ok(vec![non_empty(stop, "stop".to_owned())?]),
        Some(Value::Array(stops)) if stops.len() > MAX_STOP_STRINGS => {
            Err(ApiError::invalid_value(
                "stop",
                format!("stop must hold at most {MAX_STOP_STRINGS} strings"),
            ))
        }
        Some(Value::Array(stops)) => stops
            .iter()
            .enumerate()
            .map(|(index, stop)| {
                let param = format!("stop[{index}]");
                match stop.as_str() {
                    Some(stop) => non_empty(stop, param),
                    None => Err(ApiError::invalid_value(
                        param,
                        "stop entries must be strings",
                    )),
                }
            })
            .collect(),
        Some(_) => Err(ApiError::invalid_value(
            "stop",
            "stop must be a string or a list of strings",
        )),
    }
}

/// How a backend generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    Stop,
    /// SGLang's `tool_calls` (a natural stop with at least one parsed call): a success like `Stop` (D6).
    ToolCalls,
    /// The full forwarded budget was generated.
    Length,
    /// loop-guard design note LG-D1: the native loop guard ended a repeating output: a success like `Stop` (the
    /// output is complete as far as it goes; the client decides what to do with a loop).
    Repetition,
    Abort,
    /// `length` before the forwarded budget: the backend clamped the output. The request fails.
    BackendClampedOutput {
        completion_tokens: u64,
        budget: u64,
    },
}

impl Finish {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::ToolCalls => "tool_calls",
            Self::Length => "length",
            Self::Repetition => "repetition",
            Self::Abort => "abort",
            Self::BackendClampedOutput { .. } => "backend_clamped_output",
        }
    }
}

/// Classifies a backend `finish_reason` given `usage.completion_tokens` and the forwarded budget. A `length`
/// finish without a token count cannot rule out a clamp and is a protocol error.
pub fn classify_finish(
    finish_reason: Option<&str>,
    completion_tokens: Option<u64>,
    budget: u64,
) -> Result<Finish, BackendError> {
    match finish_reason {
        Some("stop") => Ok(Finish::Stop),
        Some("tool_calls") => Ok(Finish::ToolCalls),
        Some("abort") => Ok(Finish::Abort),
        Some("repetition") => Ok(Finish::Repetition),
        Some("length") => match completion_tokens {
            Some(completion_tokens) if completion_tokens < budget => {
                Ok(Finish::BackendClampedOutput {
                    completion_tokens,
                    budget,
                })
            }
            Some(_) => Ok(Finish::Length),
            None => Err(BackendError::Protocol(
                "finish_reason length without usage.completion_tokens".to_owned(),
            )),
        },
        Some(other) => Err(BackendError::Protocol(format!(
            "unexpected finish_reason {other:?}"
        ))),
        None => Err(BackendError::Protocol("missing finish_reason".to_owned())),
    }
}

/// [`classify_finish`] for a unary response of `n` choices: exactly `n` choices, classified by
/// [`classify_choices`] with `usage.completion_tokens`.
pub fn classify_response(response: &Value, budget: u64, n: u32) -> Result<Finish, BackendError> {
    let choices = match response.get("choices").and_then(Value::as_array) {
        Some(choices) if choices.len() as u64 == u64::from(n) => choices,
        _ if n == 1 => {
            return Err(BackendError::Protocol(
                "response must carry exactly one choice".to_owned(),
            ))
        }
        _ => {
            return Err(BackendError::Protocol(format!(
                "response must carry exactly {n} choices"
            )))
        }
    };
    let reasons = choices
        .iter()
        .map(|choice| choice.get("finish_reason").and_then(Value::as_str));
    classify_choices(
        reasons,
        response
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64),
        budget,
    )
}

/// [`classify_finish`] over the choices' `finish_reason`s, given `completion_tokens` summed over the choices
/// (M2.6 D11, F9). A `length` choice generated its whole budget unless the backend clamped it, so the clamp
/// check bounds the total by `budget` per `length` choice (for one choice: the choice's own count). The
/// request's finish: any error, else `abort`, else a clamp, else the first choice's.
pub fn classify_choices<'a>(
    reasons: impl Iterator<Item = Option<&'a str>> + Clone,
    completion_tokens: Option<u64>,
    budget: u64,
) -> Result<Finish, BackendError> {
    let lengths = reasons.clone().filter(|r| *r == Some("length")).count() as u64;
    let length_budget = budget.saturating_mul(lengths.max(1));
    let mut request = None;
    for reason in reasons {
        let finish = classify_finish(reason, completion_tokens, length_budget)?;
        request = Some(match (request, finish) {
            (Some(Finish::Abort), _) | (_, Finish::Abort) => Finish::Abort,
            (Some(clamp @ Finish::BackendClampedOutput { .. }), _)
            | (_, clamp @ Finish::BackendClampedOutput { .. }) => clamp,
            (Some(first), _) | (None, first) => first,
        });
    }
    request.ok_or_else(|| BackendError::Protocol("no choice to classify".to_owned()))
}
