//! HTTP client for one pinned SGLang server (m1 design note A1, F4) and the prefix-free request id (Q1).
//!
//! SGLang matches `POST /abort_request {"rid": r}` by **prefix**: every live request whose rid starts with `r` is
//! aborted, and an empty `r` aborts everything. [`Rid`] is fixed width, so no rid is a proper prefix of another,
//! and it can never be empty. The client has no way to send `abort_all`.
//!
//! Timeouts are explicit: a connect timeout on every call, a whole-exchange timeout on control calls, the
//! caller's deadline on chat calls, and an idle timeout between stream reads (keepalive comments count as reads).
//! A unary chat sends no bytes until it finishes, so only its deadline bounds it.

use std::fmt;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, Value};

const RID_PREFIX: &str = "cere-";
const RID_LEN: usize = RID_PREFIX.len() + 20 + 1 + 20 + 1 + 16;
/// Longest backend error message kept in a [`BackendError`].
const MAX_ERROR_MESSAGE_CHARS: usize = 512;

/// Prefix-free backend request id: `cere-<20-digit epoch>-<20-digit seq>-<16 lowercase hex nonce>` (63 bytes).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Rid(String);

/// A string that is not a [`Rid`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RidError(String);

impl fmt::Display for RidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not a CERE rid: {}", self.0)
    }
}

impl std::error::Error for RidError {}

impl Rid {
    pub fn new(epoch: u64, seq: u64, nonce: u64) -> Self {
        Self(format!("{RID_PREFIX}{epoch:020}-{seq:020}-{nonce:016x}"))
    }

    /// Accepts exactly the [`Rid::new`] format.
    pub fn parse(text: &str) -> Result<Self, RidError> {
        let invalid = || RidError(text.chars().take(RID_LEN + 8).collect());
        if text.len() != RID_LEN {
            return Err(invalid());
        }
        let rest = text.strip_prefix(RID_PREFIX).ok_or_else(invalid)?;
        let bytes = rest.as_bytes();
        let digits = |range: std::ops::Range<usize>| bytes[range].iter().all(u8::is_ascii_digit);
        let hex = bytes[42..]
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b));
        if !(digits(0..20) && bytes[20] == b'-' && digits(21..41) && bytes[41] == b'-' && hex) {
            return Err(invalid());
        }
        // 20 decimal digits can exceed u64::MAX; such a rid is never generated.
        if rest[0..20].parse::<u64>().is_err() || rest[21..41].parse::<u64>().is_err() {
            return Err(invalid());
        }
        Ok(Self(text.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn epoch(&self) -> u64 {
        self.field(0..20)
    }

    pub fn seq(&self) -> u64 {
        self.field(21..41)
    }

    pub fn nonce(&self) -> u64 {
        u64::from_str_radix(&self.0[RID_PREFIX.len() + 42..], 16).expect("validated rid nonce")
    }

    fn field(&self, range: std::ops::Range<usize>) -> u64 {
        let rest = &self.0[RID_PREFIX.len()..];
        rest[range].parse().expect("validated rid field")
    }
}

impl fmt::Display for Rid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for Rid {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Rid {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Rid::parse(&text).map_err(serde::de::Error::custom)
    }
}

/// Issues rids for one worker epoch: a per-process random nonce and a monotonic sequence starting at 0.
#[derive(Debug)]
pub struct RidGenerator {
    epoch: u64,
    nonce: u64,
    next_seq: AtomicU64,
}

impl RidGenerator {
    /// Draws the process nonce from the OS-seeded std hasher keys, mixed with the pid and the clock.
    pub fn new(epoch: u64) -> Self {
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_u32(std::process::id());
        hasher.write_u128(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        );
        Self::with_nonce(epoch, hasher.finish())
    }

    pub fn with_nonce(epoch: u64, nonce: u64) -> Self {
        Self {
            epoch,
            nonce,
            next_seq: AtomicU64::new(0),
        }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn next_rid(&self) -> Rid {
        Rid::new(
            self.epoch,
            self.next_seq.fetch_add(1, Ordering::Relaxed),
            self.nonce,
        )
    }
}

/// Failure of one backend call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendError {
    /// TCP connect failed or timed out.
    Connect(String),
    /// The call's deadline, control timeout or stream idle timeout elapsed.
    Timeout(String),
    /// Non-2xx answer (or in-stream error object) other than a context overflow.
    Status { status: u16, message: String },
    /// The backend rejected the prompt plus output budget as longer than its context length.
    ContextOverflow { message: String },
    /// The backend refused `tools[index].function.parameters` as a JSON schema (M2.4 D2: an SGLang 400 whose
    /// message starts `Tool {i} function`). `None`: the index did not survive (the remote path).
    ToolSchema { index: Option<usize> },
    /// A tools request on a backend whose `tool_call_parser` is not [`crate::validate::TOOL_CALL_PARSER`] (M2.4
    /// D1 on a remote worker: the agent refuses it before any backend call).
    ToolsUnsupported,
    /// The backend could not compile the request's grammar (M2.8 D6: an SGLang 400 whose message starts
    /// `Failed to compile` or `Grammar compilation failed`). The backend text, which may quote the client's
    /// schema, is dropped.
    GrammarCompile,
    /// Malformed or truncated response: bad JSON, a dropped connection, a stream without `data: [DONE]`.
    Protocol(String),
    /// cered's own invariant or dependency failed mid-generation (M2.8 D8: the grammar process lost or answered
    /// `accepted: false`): 500 `internal_error`; the detail stays internal.
    Internal(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connect(m) => write!(f, "backend connect failed: {m}"),
            Self::Timeout(m) => write!(f, "backend timed out: {m}"),
            Self::Status { status, message } => write!(f, "backend HTTP {status}: {message}"),
            Self::ContextOverflow { message } => write!(f, "backend context overflow: {message}"),
            Self::ToolSchema { index: Some(i) } => {
                write!(f, "backend refused the parameters schema of tool {i}")
            }
            Self::ToolSchema { index: None } => {
                write!(f, "backend refused a tool parameters schema")
            }
            Self::ToolsUnsupported => write!(f, "backend does not run the qwen3_coder tool parser"),
            Self::GrammarCompile => write!(f, "backend could not compile the request's grammar"),
            Self::Protocol(m) => write!(f, "backend protocol error: {m}"),
            Self::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for BackendError {}

impl BackendError {
    /// The op reason a worker agent ends a failed backend call with: the error's API code, except a grammar
    /// compile refusal (M2.8 D6), whose `invalid_value` would read as a tool schema refusal on the coordinator.
    pub fn op_reason(&self) -> &'static str {
        match self {
            Self::GrammarCompile => GRAMMAR_COMPILE_REASON,
            _ => crate::api_types::ApiError::from(self).code,
        }
    }
}

/// Client timeouts; every one is explicit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timeouts {
    /// TCP connect, every call.
    pub connect: Duration,
    /// Whole exchange of `/health`, `/v1/models`, `/get_server_info` and `/abort_request`.
    pub control: Duration,
    /// Longest wait for the next bytes of a chat stream.
    pub stream_idle: Duration,
}

/// Client for one SGLang `base_url` (validated loopback by the config loader).
#[derive(Clone, Debug)]
pub struct SglangHttp {
    client: reqwest::Client,
    base_url: String,
    timeouts: Timeouts,
}

impl SglangHttp {
    pub fn new(base_url: &str, timeouts: Timeouts) -> Result<Self, BackendError> {
        Self::build(base_url, timeouts, usize::MAX)
    }

    /// The same backend through a client that keeps no idle connection, so every call connects anew and its
    /// failure is the backend's state now. A kept-alive connection the backend closed reads open until the
    /// client's connection task sees the close; a request written on it before then ends unanswered, a
    /// `Protocol` error, where a fresh connect to the gone backend is refused, a `Connect` error.
    pub fn fresh_connections(&self) -> Result<Self, BackendError> {
        Self::build(&self.base_url, self.timeouts, 0)
    }

    fn build(base_url: &str, timeouts: Timeouts, max_idle: usize) -> Result<Self, BackendError> {
        let client = reqwest::Client::builder()
            .connect_timeout(timeouts.connect)
            .pool_max_idle_per_host(max_idle)
            // The backend is loopback-only; an environment proxy must never see its traffic.
            .no_proxy()
            .build()
            .map_err(|e| BackendError::Protocol(format!("cannot build HTTP client: {e}")))?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_owned(),
            timeouts,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// `GET /health` → 200.
    pub async fn health(&self) -> Result<(), BackendError> {
        let response = self
            .client
            .get(self.url("/health"))
            .timeout(self.timeouts.control)
            .send()
            .await
            .map_err(transport)?;
        check_status(response).await.map(drop)
    }

    /// Model ids listed by `GET /v1/models`.
    pub async fn models(&self) -> Result<Vec<String>, BackendError> {
        let body = self.get_json("/v1/models").await?;
        let data = body
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| BackendError::Protocol("/v1/models without a data array".to_owned()))?;
        data.iter()
            .map(|model| {
                model
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| BackendError::Protocol("/v1/models entry without id".to_owned()))
            })
            .collect()
    }

    /// `GET /get_server_info` as a JSON object.
    pub async fn server_info(&self) -> Result<Value, BackendError> {
        let body = self.get_json("/get_server_info").await?;
        if !body.is_object() {
            return Err(BackendError::Protocol(
                "/get_server_info is not a JSON object".to_owned(),
            ));
        }
        Ok(body)
    }

    /// Requests inside the backend scheduler: `L = Σ_dp (num_reqs + num_waiting_reqs)` over the per-dp-rank
    /// list of `GET <endpoint>` (SGLang `/get_load`, m1close Q6). Strict: a missing or non-integer field is a
    /// protocol error, never a zero, because an under-count could release an orphan's slot early.
    pub async fn load(&self, endpoint: &str) -> Result<u64, BackendError> {
        let body = self.get_json(endpoint).await?;
        let ranks = body
            .as_array()
            .filter(|ranks| !ranks.is_empty())
            .ok_or_else(|| BackendError::Protocol(format!("{endpoint} is not a non-empty list")))?;
        ranks.iter().try_fold(0u64, |total, rank| {
            let count = |field: &str| {
                rank.get(field).and_then(Value::as_u64).ok_or_else(|| {
                    BackendError::Protocol(format!("{endpoint} entry without integer {field}"))
                })
            };
            Ok(total
                .saturating_add(count("num_reqs")?)
                .saturating_add(count("num_waiting_reqs")?))
        })
    }

    /// Unary `POST /v1/chat/completions` with a complete backend body (`stream:false`, explicit `rid`).
    pub async fn chat<B: Serialize + ?Sized>(
        &self,
        body: &B,
        deadline: Duration,
    ) -> Result<Value, BackendError> {
        let response = self.post_chat(body, deadline).await?;
        let value: Value = response.json().await.map_err(transport)?;
        if let Some(error) = embedded_error(&value) {
            return Err(error);
        }
        Ok(value)
    }

    /// Streaming `POST /v1/chat/completions` (`stream:true`); returns once the 200 headers arrive.
    pub async fn chat_stream<B: Serialize + ?Sized>(
        &self,
        body: &B,
        deadline: Duration,
    ) -> Result<ChatStream, BackendError> {
        let response = self.post_chat(body, deadline).await?;
        Ok(ChatStream {
            response,
            parser: SseParser::default(),
            idle: self.timeouts.stream_idle,
            done: false,
        })
    }

    /// `POST /abort_request {"rid": rid}` for exactly one rid; never `abort_all`.
    pub async fn abort(&self, rid: &Rid) -> Result<(), BackendError> {
        let response = self
            .client
            .post(self.url("/abort_request"))
            .timeout(self.timeouts.control)
            .json(&json!({ "rid": rid.as_str() }))
            .send()
            .await
            .map_err(transport)?;
        check_status(response).await.map(drop)
    }

    /// `POST <path>` of a token-count endpoint (`/v1/messages/count_tokens`, M2.14 D3) with the client's request
    /// bytes; its `input_tokens`.
    pub async fn count_tokens(&self, path: &str, body: &[u8]) -> Result<u64, BackendError> {
        let response = self
            .client
            .post(self.url(path))
            .timeout(self.timeouts.control)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body.to_vec())
            .send()
            .await
            .map_err(transport)?;
        let value: Value = check_status(response)
            .await?
            .json()
            .await
            .map_err(transport)?;
        value["input_tokens"]
            .as_u64()
            .ok_or_else(|| BackendError::Protocol(format!("{path} without integer input_tokens")))
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }

    async fn get_json(&self, path: &str) -> Result<Value, BackendError> {
        let response = self
            .client
            .get(self.url(path))
            .timeout(self.timeouts.control)
            .send()
            .await
            .map_err(transport)?;
        check_status(response)
            .await?
            .json()
            .await
            .map_err(transport)
    }

    async fn post_chat<B: Serialize + ?Sized>(
        &self,
        body: &B,
        deadline: Duration,
    ) -> Result<reqwest::Response, BackendError> {
        let response = self
            .client
            .post(self.url("/v1/chat/completions"))
            .timeout(deadline)
            .json(body)
            .send()
            .await
            .map_err(transport)?;
        check_status(response).await
    }
}

/// An open chat stream. Stop reading after the first `Err` or `Ok(None)`.
#[derive(Debug)]
pub struct ChatStream {
    response: reqwest::Response,
    parser: SseParser,
    idle: Duration,
    done: bool,
}

impl ChatStream {
    /// Next parsed `data:` chunk; `Ok(None)` once `data: [DONE]` arrives. A connection that ends before
    /// `[DONE]` is a [`BackendError::Protocol`]; an in-stream error object is mapped like an HTTP error.
    pub async fn next_chunk(&mut self) -> Result<Option<Value>, BackendError> {
        if self.done {
            return Ok(None);
        }
        loop {
            if let Some(data) = self.parser.next_data()? {
                if data == "[DONE]" {
                    self.done = true;
                    return Ok(None);
                }
                let chunk: Value = serde_json::from_str(&data).map_err(|e| {
                    BackendError::Protocol(format!("stream chunk is not JSON: {e}"))
                })?;
                if let Some(error) = embedded_error(&chunk) {
                    return Err(error);
                }
                return Ok(Some(chunk));
            }
            match tokio::time::timeout(self.idle, self.response.chunk()).await {
                Err(_) => {
                    return Err(BackendError::Timeout(format!(
                        "no stream bytes for {} ms",
                        self.idle.as_millis()
                    )))
                }
                Ok(Err(e)) => return Err(transport(e)),
                Ok(Ok(None)) => {
                    return Err(BackendError::Protocol(
                        "stream ended without data: [DONE]".to_owned(),
                    ))
                }
                Ok(Ok(Some(bytes))) => self.parser.push(&bytes),
            }
        }
    }
}

/// Incremental `text/event-stream` decoder: bytes may split anywhere, including inside UTF-8 sequences.
/// Yields each event's `data` (multi-line data joined with `\n`); comments and other fields are skipped.
#[derive(Debug, Default)]
pub(crate) struct SseParser {
    buf: Vec<u8>,
    /// Bytes of `buf` already consumed.
    pos: usize,
    data: Option<String>,
}

impl SseParser {
    pub(crate) fn push(&mut self, bytes: &[u8]) {
        self.buf.drain(..self.pos);
        self.pos = 0;
        self.buf.extend_from_slice(bytes);
    }

    /// Data of the next complete event, or `None` until more bytes arrive.
    pub(crate) fn next_data(&mut self) -> Result<Option<String>, BackendError> {
        while let Some(offset) = self.buf[self.pos..].iter().position(|&b| b == b'\n') {
            let start = self.pos;
            self.pos += offset + 1;
            let mut line = &self.buf[start..start + offset];
            if let [head @ .., b'\r'] = line {
                line = head;
            }
            if line.is_empty() {
                if let Some(data) = self.data.take() {
                    return Ok(Some(data));
                }
                continue;
            }
            let line = std::str::from_utf8(line)
                .map_err(|_| BackendError::Protocol("stream line is not UTF-8".to_owned()))?;
            let value = if line == "data" {
                ""
            } else if let Some(value) = line.strip_prefix("data:") {
                value.strip_prefix(' ').unwrap_or(value)
            } else {
                // Comment (keepalive) or a field CERE does not use.
                continue;
            };
            match &mut self.data {
                Some(data) => {
                    data.push('\n');
                    data.push_str(value);
                }
                None => self.data = Some(value.to_owned()),
            }
        }
        Ok(None)
    }
}

fn transport(error: reqwest::Error) -> BackendError {
    if error.is_timeout() {
        BackendError::Timeout(error.to_string())
    } else if error.is_connect() {
        BackendError::Connect(error.to_string())
    } else {
        BackendError::Protocol(error.to_string())
    }
}

async fn check_status(response: reqwest::Response) -> Result<reqwest::Response, BackendError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let text = response.text().await.map_err(transport)?;
    let message = serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|body| error_message(&body).map(str::to_owned))
        .unwrap_or(text);
    Err(status_error(status.as_u16(), &message))
}

/// SGLang error objects: `{"object":"error","message",…,"code"}` or `{"error":{"message",…,"code"}}`.
fn error_message(body: &Value) -> Option<&str> {
    body.pointer("/error/message")
        .or_else(|| {
            (body.get("object").and_then(Value::as_str) == Some("error"))
                .then(|| body.get("message"))
                .flatten()
        })
        .and_then(Value::as_str)
}

/// SGLang's grammar compile refusals (M2.8 F13): the message starts with one of these.
const GRAMMAR_COMPILE_PREFIXES: [&str; 2] = ["Failed to compile", "Grammar compilation failed"];

/// The op reason a worker agent ends a grammar compile refusal with, which the coordinator maps back to
/// [`BackendError::GrammarCompile`] (M2.8 D6).
pub const GRAMMAR_COMPILE_REASON: &str = "grammar_compile_failed";

/// An error object delivered inside a 200 body or SSE chunk.
fn embedded_error(body: &Value) -> Option<BackendError> {
    let message = error_message(body)?;
    let code = body
        .pointer("/error/code")
        .or_else(|| body.get("code"))
        .and_then(Value::as_u64)
        .and_then(|code| u16::try_from(code).ok())
        .unwrap_or(500);
    Some(status_error(code, message))
}

/// SGLang's tool schema refusal (anchored at the message start, which then quotes the client's schema) is a
/// [`BackendError::ToolSchema`]; a 400 starting with a grammar compile prefix (M2.8 D6, F13) is a
/// [`BackendError::GrammarCompile`]; else a 400 naming the context length (the tokenizer's check) or "exceeds
/// the maximum allowed length" (the scheduler's check, delivered in-stream after HTTP 200) is a context overflow;
/// nothing else is.
fn status_error(status: u16, message: &str) -> BackendError {
    let message: String = message.chars().take(MAX_ERROR_MESSAGE_CHARS).collect();
    if let Some(index) = tool_schema_index(status, &message) {
        return BackendError::ToolSchema { index: Some(index) };
    }
    if status == 400
        && GRAMMAR_COMPILE_PREFIXES
            .iter()
            .any(|p| message.starts_with(p))
    {
        return BackendError::GrammarCompile;
    }
    let lower = message.to_ascii_lowercase();
    if status == 400
        && (lower.contains("context length")
            || lower.contains("exceeds the maximum allowed length"))
    {
        BackendError::ContextOverflow { message }
    } else {
        BackendError::Status { status, message }
    }
}

/// SGLang's tool schema refusal (`serving_chat.py:868,871`): a 400 whose message starts `Tool {i} function`.
fn tool_schema_index(status: u16, message: &str) -> Option<usize> {
    let rest = message.strip_prefix("Tool ").filter(|_| status == 400)?;
    let digits = rest.find(|c: char| !c.is_ascii_digit())?;
    if digits == 0 || !rest[digits..].starts_with(" function") {
        return None;
    }
    rest[..digits].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const STREAM: &str = ": keepalive\r\n\r\ndata: {\"a\":1}\n\nevent: x\ndata: {\"b\":\ndata: 2}\n\n: ping\n\ndata:{\"c\":\"\u{e9}\"}\r\n\r\ndata: [DONE]\n\n";

    fn drain(parser: &mut SseParser, out: &mut Vec<String>) {
        while let Some(data) = parser.next_data().unwrap() {
            out.push(data);
        }
    }

    #[test]
    fn sse_parser_is_split_invariant() {
        let expected = vec![
            "{\"a\":1}".to_owned(),
            "{\"b\":\n2}".to_owned(),
            "{\"c\":\"\u{e9}\"}".to_owned(),
            "[DONE]".to_owned(),
        ];
        let bytes = STREAM.as_bytes();
        for split in 0..=bytes.len() {
            let mut parser = SseParser::default();
            let mut out = Vec::new();
            parser.push(&bytes[..split]);
            drain(&mut parser, &mut out);
            parser.push(&bytes[split..]);
            drain(&mut parser, &mut out);
            assert_eq!(out, expected, "split at {split}");
        }
        let mut parser = SseParser::default();
        let mut out = Vec::new();
        for byte in bytes {
            parser.push(std::slice::from_ref(byte));
            drain(&mut parser, &mut out);
        }
        assert_eq!(out, expected, "byte at a time");
    }

    #[test]
    fn sse_parser_waits_for_the_blank_line() {
        let mut parser = SseParser::default();
        parser.push(b"data: {\"a\":1}\n");
        assert_eq!(parser.next_data().unwrap(), None);
        parser.push(b"\n");
        assert_eq!(parser.next_data().unwrap().as_deref(), Some("{\"a\":1}"));
    }

    #[test]
    fn error_objects_map_to_typed_errors() {
        let overflow = json!({"object":"error","message":"The input (9 tokens) is longer than the model's context length (8 tokens).","type":"BadRequestError","param":null,"code":400});
        assert!(matches!(
            embedded_error(&overflow),
            Some(BackendError::ContextOverflow { .. })
        ));
        let nested = json!({"error":{"message":"boom","type":"InternalServerError","code":503}});
        assert_eq!(
            embedded_error(&nested),
            Some(BackendError::Status {
                status: 503,
                message: "boom".to_owned()
            })
        );
        assert_eq!(embedded_error(&json!({"object":"chat.completion"})), None);
        assert_eq!(
            status_error(500, "context length"),
            BackendError::Status {
                status: 500,
                message: "context length".to_owned()
            }
        );
    }

    /// SGLang's schema refusal quotes the client's schema, so its text may hold "context length": the anchored
    /// tool-schema prefix wins.
    #[test]
    fn tool_schema_refusal_quoting_context_length_is_a_tool_schema() {
        let message = r#"Tool 2 function has invalid 'parameters' schema: {'type': 'context length'} is not valid"#;
        assert_eq!(
            status_error(400, message),
            BackendError::ToolSchema { index: Some(2) }
        );
        assert!(matches!(
            status_error(400, "x Tool 2 function context length"),
            BackendError::ContextOverflow { .. }
        ));
    }

    /// A request body that is empty at once.
    struct NoBody;

    impl hyper::body::Body for NoBody {
        type Data = &'static [u8];
        type Error = std::convert::Infallible;

        fn poll_frame(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Result<hyper::body::Frame<Self::Data>, Self::Error>>> {
            std::task::Poll::Ready(None)
        }

        fn is_end_stream(&self) -> bool {
            true
        }
    }

    /// A peer that takes every write and never answers, so a sent request stays in flight.
    struct SilentPeer;

    impl hyper::rt::Read for SilentPeer {
        fn poll_read(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
            _: hyper::rt::ReadBufCursor<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            std::task::Poll::Pending
        }
    }

    impl hyper::rt::Write for SilentPeer {
        fn poll_write(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
            buf: &[u8],
        ) -> std::task::Poll<std::io::Result<usize>> {
            std::task::Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn poll_shutdown(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            std::task::Poll::Ready(Ok(()))
        }
    }

    /// One poll of the connection task, which never ends: the peer never answers.
    async fn poll_connection(
        connection: &mut hyper::client::conn::http1::Connection<SilentPeer, NoBody>,
        exhaust_budget: bool,
    ) {
        std::future::poll_fn(|cx| {
            if exhaust_budget {
                while let std::task::Poll::Ready(restore) = tokio::task::coop::poll_proceed(cx) {
                    restore.made_progress();
                }
            }
            let connection = std::pin::Pin::new(&mut *connection);
            assert!(std::future::Future::poll(connection, cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
    }

    /// PlacementHang (O304): [`SglangHttp`]'s client, like every `reqwest::Client`, pools HTTP/1 connections
    /// through hyper-util's legacy pool, which returns a connection to the idle set once hyper reports it ready.
    /// hyper < 1.12 could report ready with a response still streaming (a stale "want", hyperium/hyper#4207): the
    /// pool then gave that connection to the next request (a chat, `/abort_request`, `/health`), which waited
    /// behind the whole stream. A connection must stay not-ready from the request until its response ends. The
    /// stale signal comes from the connection task finding its queue empty just before a request lands (a
    /// preempted task, how it hung under load) or from tokio's coop budget making a queue that holds the request
    /// report `Pending`; the budget path is deterministic, so this drives that one (hyper's own regression test,
    /// hyperium/hyper#4208).
    #[tokio::test]
    async fn a_connection_with_a_request_in_flight_is_not_ready() {
        let (mut sender, mut connection) = hyper::client::conn::http1::handshake(SilentPeer)
            .await
            .unwrap();
        poll_connection(&mut connection, false).await;
        assert!(sender.is_ready(), "an idle connection is ready");

        let _response = sender.send_request(hyper::Request::new(NoBody));
        assert!(!sender.is_ready(), "a queued request makes it busy");
        poll_connection(&mut connection, true).await;
        tokio::task::yield_now().await;
        poll_connection(&mut connection, false).await;
        assert!(
            !sender.is_ready(),
            "the connection reports ready with a request in flight"
        );
    }
}
