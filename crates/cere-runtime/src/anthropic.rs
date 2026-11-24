//! `POST /v1/messages` and `POST /v1/messages/count_tokens` (m214 design note D3): the Anthropic Messages
//! surface over the shared chat pipeline.
//!
//! The request is validated as SGLang's `protocol.py` models validate it (pydantic messages `<loc>: <msg>`) and
//! converted as SGLang's `_convert_to_chat_completion_request` converts it, into a body of CERE's chat subset:
//! block lists become the string the pinned chat template renders from them (its `render_content` concatenates
//! text items), `tools[].input_schema` keeps the client's bytes, and a `tool_use` input becomes `arguments` as
//! Python `json.dumps` writes it. `thinking` disabled/absent is the deployment default (no `chat_template_kwargs`).
//! Capability-keyed fields convert as SGLang converts them on a profile that declares the capability and refuse
//! after SGLang's own conversion errors elsewhere (ApiFit, api-fit design note AF-L6): `top_k` (M2.7
//! `sampling_params`), `tool_choice` any/tool -> `required` / a named function (M2.8 `tool_calls_constrained`),
//! `thinking` enabled/adaptive and `output_config.effort` low/medium (M2.12 `thinking_toggle`). Image blocks
//! (M2.13) always refuse.
//! Responses, SSE events and error envelopes mirror SGLang's `serving.py` / `http_server.py`.

use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use serde::Serialize;
use serde_json::value::RawValue;
use serde_json::{json, Map, Value};

use crate::api_types::ContextExcess;
use crate::app::App;
use crate::document;
use crate::pyjson::python_json;
use crate::surface::{
    CompatApi, CompatKind, ConvertCtx, ErrorView, OwnedError, SurfaceRender, SurfaceStream,
};
use crate::validate::ChatPolicy;

const INVALID_REQUEST: &str = "invalid_request_error";

/// The surface's entry points for `server::serve_compat` / `server::compat_gate`.
pub static ANTHROPIC: CompatApi = CompatApi {
    kind: CompatKind::Anthropic,
    error_body,
    convert,
};

/// Installs the Anthropic routes (registered iff some deployment sets `api_compat_anthropic`).
pub fn routes(router: Router<Arc<App>>) -> Router<Arc<App>> {
    router
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
}

async fn messages(State(app): State<Arc<App>>, headers: HeaderMap, body: Body) -> Response {
    crate::server::serve_compat(&ANTHROPIC, app, headers, body).await
}

/// `count_tokens` renders and encodes the converted conversation as the chat path does for
/// `usage.prompt_tokens` (native), or asks SGLang's own endpoint (`sglang_http`); errors in the envelope.
async fn count_tokens(State(app): State<Arc<App>>, headers: HeaderMap, body: Body) -> Response {
    use crate::server::{bytes_response, compat_gate, compat_prompt_tokens, status_of};
    let (_, bytes, deployment) = match compat_gate(&ANTHROPIC, &app, &headers, body).await {
        Ok(front) => front,
        Err(response) => return response,
    };
    // Audit F5: the conversion renders the request's documents; with any, it runs off the async workers.
    let converted = {
        let settings = app.settings();
        let bytes = bytes.clone();
        let carries = crate::document::bytes_carry_documents(&bytes);
        crate::document::run(carries, move || {
            convert_count_tokens(&bytes, &settings.policies[deployment])
        })
        .await
        .unwrap_or_else(|refused| Err(crate::server::offload_owned_error(refused)))
    };
    let chat = match converted {
        Ok(chat) => chat,
        Err(e) => {
            let view = e.view();
            return bytes_response(status_of(view.status), error_body(&view));
        }
    };
    match compat_prompt_tokens(&app, deployment, &chat, "/v1/messages/count_tokens", &bytes).await {
        Ok(n) => bytes_response(axum::http::StatusCode::OK, count_tokens_body(n)),
        Err(e) => bytes_response(
            status_of(e.status),
            error_body(&ErrorView {
                status: e.status,
                kind: e.error_type,
                code: e.code,
                message: &e.message,
                param: e.param.as_deref(),
            }),
        ),
    }
}

fn convert(
    bytes: &[u8],
    ctx: &ConvertCtx,
) -> Result<(Vec<u8>, Arc<dyn SurfaceRender>), OwnedError> {
    let (chat, echo) = convert_request(bytes, ctx.policy)?;
    Ok((chat, Arc::new(echo)))
}

// ---------------------------------------------------------------------------------------------------------------
// Errors

/// SGLang's `ERROR_TYPE_MAP` (`serving.py`): Anthropic's `error.type` by HTTP status, else `api_error`.
fn error_type(status: u16) -> &'static str {
    match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        408 => "request_timeout_error",
        429 => "rate_limit_error",
        503 => "overloaded_error",
        _ => "api_error",
    }
}

/// SGLang's `_scrub_error_message`: 5xx is always generic; 4xx keeps the message without traceback lines,
/// truncated to 500 characters.
fn scrub(message: &str, status: u16) -> String {
    if status >= 500 {
        return "Internal server error".to_owned();
    }
    if message.is_empty() {
        return "Request failed".to_owned();
    }
    let kept: Vec<&str> = message
        .lines()
        .filter(|line| !line.starts_with("Traceback") && !line.contains("File \"/"))
        .collect();
    let cleaned = kept.join("\n");
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        return "Request failed".to_owned();
    }
    if cleaned.chars().count() > 500 {
        return format!("{}…", cleaned.chars().take(500).collect::<String>());
    }
    cleaned.to_owned()
}

#[derive(Serialize)]
struct ErrorPayload<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    message: &'a str,
}

#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    error: ErrorPayload<'a>,
}

/// The client-visible message: Anthropic's own context refusal text for a prompt that does not fit (ApiFit AF-L3,
/// api-fit design note F12; a coding agent compacts on the first form and lowers `max_tokens` on the second),
/// else SGLang's scrubbed message.
fn client_message(e: &ErrorView) -> String {
    match ContextExcess::of(e.code, e.message) {
        Some(x) if x.prompt > x.context => {
            format!("prompt is too long: {} tokens > {} maximum", x.prompt, x.context)
        }
        Some(x) => format!(
            "input length and `max_tokens` exceed context limit: {} + {} > {}, decrease input length or \
             `max_tokens` and try again",
            x.prompt, x.max_tokens, x.context
        ),
        None => scrub(e.message, e.status),
    }
}

/// The Anthropic error envelope `{"type":"error","error":{"type","message"}}` (SGLang `_error_response`).
pub fn error_body(e: &ErrorView) -> Vec<u8> {
    let message = client_message(e);
    let envelope = ErrorEnvelope {
        kind: "error",
        error: ErrorPayload {
            kind: error_type(e.status),
            message: &message,
        },
    };
    serde_json::to_vec(&envelope).unwrap_or_default()
}

fn error(code: &str, message: String, param: Option<String>) -> OwnedError {
    OwnedError {
        status: 400,
        kind: INVALID_REQUEST.to_owned(),
        code: code.to_owned(),
        message,
        param,
    }
}

/// A pydantic validation error, rendered as SGLang's handler renders it: `<loc>: <msg>`.
fn field_error(code: &str, loc: &str, msg: &str) -> OwnedError {
    error(code, format!("{loc}: {msg}"), Some(loc.to_owned()))
}

fn missing(loc: &str) -> OwnedError {
    field_error("missing_parameter", loc, "Field required")
}

fn invalid(loc: &str, msg: &str) -> OwnedError {
    field_error("invalid_value", loc, msg)
}

fn value_error(loc: &str, msg: &str) -> OwnedError {
    invalid(loc, &format!("Value error, {msg}"))
}

/// Pydantic's literal error: `Input should be 'a', 'b' or 'c'`.
fn literal(loc: &str, options: &[&str]) -> OwnedError {
    let quoted: Vec<String> = options.iter().map(|o| format!("'{o}'")).collect();
    let expected = match quoted.split_last() {
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} or {last}", rest.join(", ")),
        None => String::new(),
    };
    invalid(loc, &format!("Input should be {expected}"))
}

/// A CERE refusal of a field this deployment lacks the capability for (ruling 2).
fn unsupported(param: String, message: String) -> OwnedError {
    error("unsupported_parameter", message, Some(param))
}

// ---------------------------------------------------------------------------------------------------------------
// Validation (protocol.py)

type Fields = Map<String, Value>;
type Checked<T> = Result<T, OwnedError>;

/// A present, non-null optional field.
fn opt<'a>(object: &'a Fields, key: &str) -> Option<&'a Value> {
    object.get(key).filter(|v| !v.is_null())
}

fn at(loc: &str, key: &str) -> String {
    if loc.is_empty() {
        key.to_owned()
    } else {
        format!("{loc}.{key}")
    }
}

fn check_str<'a>(value: &'a Value, loc: &str) -> Checked<&'a str> {
    value
        .as_str()
        .ok_or_else(|| invalid(loc, "Input should be a valid string"))
}

fn check_int(value: &Value, loc: &str) -> Checked<i128> {
    if let Some(i) = value.as_i64() {
        return Ok(i128::from(i));
    }
    if let Some(u) = value.as_u64() {
        return Ok(i128::from(u));
    }
    match value.as_f64() {
        Some(f) if f.fract() == 0.0 && f.abs() < 1e18 => Ok(f as i128),
        Some(_) => Err(invalid(
            loc,
            "Input should be a valid integer, got a number with a fractional part",
        )),
        None => Err(invalid(loc, "Input should be a valid integer")),
    }
}

fn check_number(value: &Value, loc: &str) -> Checked<f64> {
    value
        .as_f64()
        .ok_or_else(|| invalid(loc, "Input should be a valid number"))
}

fn check_bool(value: &Value, loc: &str) -> Checked<()> {
    value
        .as_bool()
        .map(|_| ())
        .ok_or_else(|| invalid(loc, "Input should be a valid boolean"))
}

fn check_list<'a>(value: &'a Value, loc: &str) -> Checked<&'a Vec<Value>> {
    value
        .as_array()
        .ok_or_else(|| invalid(loc, "Input should be a valid list"))
}

fn check_dict<'a>(value: &'a Value, loc: &str) -> Checked<&'a Fields> {
    value
        .as_object()
        .ok_or_else(|| invalid(loc, "Input should be a valid dictionary"))
}

fn check_model<'a>(value: &'a Value, loc: &str, model: &str) -> Checked<&'a Fields> {
    value.as_object().ok_or_else(|| {
        invalid(
            loc,
            &format!("Input should be a valid dictionary or instance of {model}"),
        )
    })
}

/// A required field: absent is `Field required`, `null` fails the type check.
fn required<'a>(object: &'a Fields, key: &str, loc: &str) -> Checked<&'a Value> {
    object.get(key).ok_or_else(|| missing(&at(loc, key)))
}

fn opt_str(object: &Fields, key: &str, loc: &str) -> Checked<()> {
    match opt(object, key) {
        Some(v) => check_str(v, &at(loc, key)).map(|_| ()),
        None => Ok(()),
    }
}

fn str_list(value: &Value, loc: &str) -> Checked<()> {
    for (i, item) in check_list(value, loc)?.iter().enumerate() {
        check_str(item, &format!("{loc}.{i}"))?;
    }
    Ok(())
}

const BLOCK_TAGS: [&str; 8] = [
    "text",
    "image",
    "tool_use",
    "tool_result",
    "tool_reference",
    "search_result",
    "thinking",
    "redacted_thinking",
];

/// One `AnthropicContentBlock` (discriminated on `type`).
fn check_block(value: &Value, loc: &str) -> Checked<()> {
    let object = value.as_object().ok_or_else(|| {
        invalid(
            loc,
            "Input should be a valid dictionary or object to extract fields from",
        )
    })?;
    let tag = match object.get("type").and_then(Value::as_str) {
        Some(tag) => tag,
        None => {
            return Err(invalid(
                loc,
                "Unable to extract tag using discriminator 'type'",
            ))
        }
    };
    // DocInput DI-01: SGLang has no `document` block; it is checked apart so the unknown-tag message stays SGLang's.
    if tag == "document" {
        return check_document(object, &at(loc, tag));
    }
    if !BLOCK_TAGS.contains(&tag) {
        let expected: Vec<String> = BLOCK_TAGS.iter().map(|t| format!("'{t}'")).collect();
        return Err(invalid(
            loc,
            &format!(
                "Input tag '{tag}' found using 'type' does not match any of the expected tags: {}",
                expected.join(", ")
            ),
        ));
    }
    let loc = at(loc, tag);
    match tag {
        "text" => check_str(required(object, "text", &loc)?, &at(&loc, "text")).map(|_| ()),
        "image" => match opt(object, "source") {
            Some(Value::String(_)) | None => Ok(()),
            Some(v) => check_dict(v, &at(&loc, "source")).map(|_| ()),
        },
        "tool_use" => {
            check_str(required(object, "id", &loc)?, &at(&loc, "id"))?;
            check_str(required(object, "name", &loc)?, &at(&loc, "name"))?;
            match object.get("input") {
                None => Ok(()),
                Some(v) => check_dict(v, &at(&loc, "input")).map(|_| ()),
            }
        }
        "tool_result" => {
            opt_str(object, "tool_use_id", &loc)?;
            opt_str(object, "id", &loc)?;
            match opt(object, "content") {
                None | Some(Value::String(_)) => {}
                Some(Value::Array(items)) => {
                    for (i, item) in items.iter().enumerate() {
                        check_block(item, &format!("{loc}.content.{i}"))?;
                    }
                }
                Some(_) => {
                    return Err(invalid(
                        &at(&loc, "content"),
                        "Input should be a valid string",
                    ))
                }
            }
            match opt(object, "is_error") {
                Some(v) => check_bool(v, &at(&loc, "is_error")),
                None => Ok(()),
            }
        }
        "tool_reference" => {
            opt_str(object, "name", &loc)?;
            opt_str(object, "tool_name", &loc)?;
            opt_str(object, "id", &loc)
        }
        "search_result" => match opt(object, "content") {
            Some(v) => {
                for (i, item) in check_list(v, &at(&loc, "content"))?.iter().enumerate() {
                    check_dict(item, &format!("{loc}.content.{i}"))?;
                }
                Ok(())
            }
            None => Ok(()),
        },
        "thinking" => {
            check_str(required(object, "thinking", &loc)?, &at(&loc, "thinking"))?;
            opt_str(object, "signature", &loc)
        }
        _ => opt_str(object, "data", &loc),
    }
}

const DOCUMENT_SOURCES: [&str; 5] = ["base64", "text", "content", "url", "file"];

/// An Anthropic `document` block: `{source, title?, context?, citations?}` with a `base64` (`data`, `media_type`),
/// `text` (`data`), `content` (a string or blocks), `url` or `file` (`file_id`) source. Shape only; what CERE
/// can read is decided at conversion ([`document_text`]).
fn check_document(object: &Fields, loc: &str) -> Checked<()> {
    let source = check_dict(required(object, "source", loc)?, &at(loc, "source"))?;
    let loc_source = at(loc, "source");
    let kind = check_str(
        required(source, "type", &loc_source)?,
        &at(&loc_source, "type"),
    )?;
    if !DOCUMENT_SOURCES.contains(&kind) {
        return Err(literal(&at(&loc_source, "type"), &DOCUMENT_SOURCES));
    }
    match kind {
        "base64" => {
            check_str(
                required(source, "data", &loc_source)?,
                &at(&loc_source, "data"),
            )?;
            check_str(
                required(source, "media_type", &loc_source)?,
                &at(&loc_source, "media_type"),
            )?;
        }
        "text" => {
            check_str(
                required(source, "data", &loc_source)?,
                &at(&loc_source, "data"),
            )?;
            opt_str(source, "media_type", &loc_source)?;
        }
        "content" => match required(source, "content", &loc_source)? {
            Value::String(_) => {}
            content => {
                let loc_content = at(&loc_source, "content");
                for (i, item) in check_list(content, &loc_content)?.iter().enumerate() {
                    check_block(item, &format!("{loc_content}.{i}"))?;
                }
            }
        },
        "url" => {
            check_str(
                required(source, "url", &loc_source)?,
                &at(&loc_source, "url"),
            )?;
        }
        _ => {
            check_str(
                required(source, "file_id", &loc_source)?,
                &at(&loc_source, "file_id"),
            )?;
        }
    }
    opt_str(object, "title", loc)?;
    opt_str(object, "context", loc)?;
    if let Some(citations) = opt(object, "citations") {
        let citations = check_dict(citations, &at(loc, "citations"))?;
        if let Some(enabled) = opt(citations, "enabled") {
            check_bool(enabled, &at(&at(loc, "citations"), "enabled"))?;
        }
    }
    Ok(())
}

/// `Union[str, list[AnthropicContentBlock]]`.
fn check_content(value: &Value, loc: &str) -> Checked<()> {
    match value {
        Value::String(_) => Ok(()),
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                check_block(item, &format!("{loc}.{i}"))?;
            }
            Ok(())
        }
        _ => Err(invalid(&at(loc, "str"), "Input should be a valid string")),
    }
}

fn check_messages(object: &Fields) -> Checked<()> {
    let messages = required(object, "messages", "")?;
    for (i, message) in check_list(messages, "messages")?.iter().enumerate() {
        let loc = format!("messages.{i}");
        let message = check_model(message, &loc, "AnthropicMessage")?;
        let role = required(message, "role", &loc)?;
        if !matches!(role.as_str(), Some("user" | "assistant" | "system")) {
            return Err(literal(&at(&loc, "role"), &["user", "assistant", "system"]));
        }
        check_content(required(message, "content", &loc)?, &at(&loc, "content"))?;
    }
    Ok(())
}

/// `AnthropicThinkingParam` and its shape validator.
fn check_thinking(value: &Value) -> Checked<()> {
    let thinking = check_model(value, "thinking", "AnthropicThinkingParam")?;
    let kind = required(thinking, "type", "thinking")?;
    let kind = match kind.as_str() {
        Some(k @ ("enabled" | "disabled" | "adaptive")) => k,
        _ => {
            return Err(literal(
                "thinking.type",
                &["enabled", "disabled", "adaptive"],
            ))
        }
    };
    let budget = match opt(thinking, "budget_tokens") {
        Some(v) => Some(check_int(v, "thinking.budget_tokens")?),
        None => None,
    };
    let display = opt(thinking, "display");
    if let Some(d) = display {
        if !matches!(d.as_str(), Some("summarized" | "omitted")) {
            return Err(literal("thinking.display", &["summarized", "omitted"]));
        }
    }
    match (kind, budget) {
        ("enabled", None) => Err(value_error(
            "thinking",
            "thinking.budget_tokens is required when thinking.type is 'enabled'",
        )),
        ("enabled", Some(b)) if b < 1024 => Err(value_error(
            "thinking",
            &format!("thinking.budget_tokens must be >= 1024 (got {b})"),
        )),
        ("disabled", Some(_)) => Err(value_error(
            "thinking",
            "thinking.budget_tokens is not allowed when thinking.type is 'disabled'",
        )),
        ("disabled", None) if display.is_some() => Err(value_error(
            "thinking",
            "thinking.display is not allowed when thinking.type is 'disabled'",
        )),
        ("adaptive", Some(_)) => Err(value_error(
            "thinking",
            "thinking.budget_tokens is not allowed when thinking.type is 'adaptive'",
        )),
        _ => Ok(()),
    }
}

fn check_tool_choice(value: &Value) -> Checked<()> {
    let choice = check_model(value, "tool_choice", "AnthropicToolChoice")?;
    let kind = required(choice, "type", "tool_choice")?;
    if !matches!(kind.as_str(), Some("auto" | "any" | "tool" | "none")) {
        return Err(literal(
            "tool_choice.type",
            &["auto", "any", "tool", "none"],
        ));
    }
    opt_str(choice, "name", "tool_choice")
}

/// `_tool_discriminator`: missing/`custom` type and unknown types are custom tools.
fn tool_family(tool: &Value) -> &'static str {
    match tool.get("type").and_then(Value::as_str) {
        Some(t) if t.starts_with("web_search_") => "web_search",
        Some(t) if t.starts_with("computer_") => "computer",
        Some(t) if t.starts_with("bash_") => "bash",
        Some(t) if t.starts_with("text_editor_") => "text_editor",
        _ => "custom",
    }
}

fn check_tools(value: &Value) -> Checked<()> {
    for (i, tool) in check_list(value, "tools")?.iter().enumerate() {
        let family = tool_family(tool);
        let loc = format!("tools.{i}.{family}");
        let object = check_model(tool, &loc, "AnthropicCustomTool")?;
        match family {
            "custom" => {
                if let Some(kind) = opt(object, "type") {
                    if kind.as_str() != Some("custom") {
                        return Err(literal(&at(&loc, "type"), &["custom"]));
                    }
                }
                check_str(required(object, "name", &loc)?, &at(&loc, "name"))?;
                opt_str(object, "description", &loc)?;
                check_dict(
                    required(object, "input_schema", &loc)?,
                    &at(&loc, "input_schema"),
                )?;
                if let Some(v) = opt(object, "defer_loading") {
                    check_bool(v, &at(&loc, "defer_loading"))?;
                }
            }
            "web_search" => {
                let kind = object.get("type").and_then(Value::as_str).unwrap_or("");
                let digits = &kind["web_search_".len()..];
                if digits.len() != 8 || !digits.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(invalid(
                        &at(&loc, "type"),
                        "String should match pattern '^web_search_\\d{8}$'",
                    ));
                }
                if let Some(name) = opt(object, "name") {
                    if name.as_str() != Some("web_search") {
                        return Err(literal(&at(&loc, "name"), &["web_search"]));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

const EFFORTS: [&str; 6] = ["minimal", "low", "medium", "high", "xhigh", "max"];

fn check_output_config(value: &Value) -> Checked<()> {
    let config = check_model(value, "output_config", "AnthropicOutputConfig")?;
    if let Some(effort) = opt(config, "effort") {
        if !effort.as_str().is_some_and(|e| EFFORTS.contains(&e)) {
            return Err(literal("output_config.effort", &EFFORTS));
        }
    }
    if let Some(budget) = opt(config, "task_budget") {
        let loc = "output_config.task_budget";
        let budget = check_model(budget, loc, "AnthropicTaskBudget")?;
        if required(budget, "type", loc)?.as_str() != Some("tokens") {
            return Err(literal(&at(loc, "type"), &["tokens"]));
        }
        let total = check_int(required(budget, "total", loc)?, &at(loc, "total"))?;
        if total <= 0 {
            return Err(invalid(&at(loc, "total"), "Input should be greater than 0"));
        }
        if let Some(remaining) = opt(budget, "remaining") {
            if check_int(remaining, &at(loc, "remaining"))? < 0 {
                return Err(invalid(
                    &at(loc, "remaining"),
                    "Input should be greater than or equal to 0",
                ));
            }
        }
    }
    Ok(())
}

/// `AnthropicMessagesRequest` (`count == false`) or `AnthropicCountTokensRequest`, field by field in model order;
/// the first error is reported.
fn check_request(object: &Fields, count: bool) -> Checked<()> {
    let model = check_str(required(object, "model", "")?, "model")?;
    check_messages(object)?;
    if !count {
        let max_tokens = check_int(required(object, "max_tokens", "")?, "max_tokens")?;
        if let Some(v) = opt(object, "metadata") {
            check_dict(v, "metadata")?;
        }
        if let Some(v) = opt(object, "stop_sequences") {
            str_list(v, "stop_sequences")?;
        }
        if let Some(v) = opt(object, "stream") {
            check_bool(v, "stream")?;
        }
        if model.is_empty() {
            return Err(value_error("model", "Model is required"));
        }
        if max_tokens <= 0 {
            return Err(value_error("max_tokens", "max_tokens must be positive"));
        }
    }
    if let Some(v) = opt(object, "system") {
        check_content(v, "system")?;
    }
    if !count {
        if let Some(v) = opt(object, "temperature") {
            check_number(v, "temperature")?;
        }
    }
    if let Some(v) = opt(object, "thinking") {
        check_thinking(v)?;
    }
    if let Some(v) = opt(object, "tool_choice") {
        check_tool_choice(v)?;
    }
    if let Some(v) = opt(object, "tools") {
        check_tools(v)?;
    }
    if !count {
        if let Some(v) = opt(object, "top_k") {
            check_int(v, "top_k")?;
        }
        if let Some(v) = opt(object, "top_p") {
            check_number(v, "top_p")?;
        }
    }
    if let Some(v) = opt(object, "output_config") {
        check_output_config(v)?;
    }
    if let Some(v) = opt(object, "betas") {
        str_list(v, "betas")?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// Conversion (`_convert_to_chat_completion_request`)

#[derive(Serialize)]
struct ChatFunction<'a> {
    description: &'a str,
    name: &'a str,
    parameters: Box<RawValue>,
}

#[derive(Serialize)]
struct ChatTool<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    function: ChatFunction<'a>,
}

/// The converted body, in CERE's chat subset.
#[derive(Serialize)]
struct ChatBody<'a> {
    messages: Vec<Value>,
    model: &'a str,
    max_tokens: &'a Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<&'a Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    /// M2.7 sampling (`top_k` in the profile's `sampling_params`), as sent.
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<&'a Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ChatTool<'a>>,
    /// `auto`/`none`, or with M2.8 `required` / `{"type": "function", "function": {"name"}}`.
    tool_choice: Value,
    /// `{"enable_thinking": true}` for `thinking` enabled/adaptive (M2.12), else absent (deployment default).
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<Value>,
    /// `output_config.effort` low/medium (M2.12 D13).
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    /// ApiFit AF-L7: `thinking.budget_tokens` on a native deployment (CERE's `thinking_budget` chat field).
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<u64>,
}

/// The raw-byte view of a JSON object (keys -> exact value bytes), for the subtrees whose bytes or key order the
/// conversion keeps.
type RawObject<'a> = HashMap<String, &'a RawValue>;

fn raw_object(raw: &RawValue) -> RawObject<'_> {
    serde_json::from_str(raw.get()).unwrap_or_default()
}

fn raw_list(raw: Option<&RawValue>) -> Vec<&RawValue> {
    raw.and_then(|raw| serde_json::from_str(raw.get()).ok())
        .unwrap_or_default()
}

/// The first capability refusal met while converting; raised after SGLang's own conversion errors (ruling 2).
#[derive(Default)]
struct Refusals(Option<OwnedError>);

impl Refusals {
    fn note(&mut self, refusal: impl FnOnce() -> OwnedError) {
        if self.0.is_none() {
            self.0 = Some(refusal());
        }
    }
}

/// `_convert_anthropic_image_source_to_openai_part` yields a part.
fn image_part(source: Option<&Value>) -> bool {
    match source {
        Some(Value::Object(source)) => {
            if source.get("type").and_then(Value::as_str) == Some("base64") {
                source
                    .get("data")
                    .and_then(Value::as_str)
                    .is_some_and(|d| !d.is_empty())
            } else {
                source
                    .get("url")
                    .and_then(Value::as_str)
                    .is_some_and(|u| !u.is_empty())
            }
        }
        _ => false,
    }
}

fn refuse_image(refusals: &mut Refusals, param: String) {
    refusals.note(|| {
        unsupported(
            param,
            "image blocks need the M2.13 image-input capability, which this deployment does not have".to_owned(),
        )
    });
}

/// DocInput DI-01: a `document` block rendered as text ([`document`]): a base64 PDF or text document, a text
/// source, or a content source's text blocks. `None` once a refusal is noted (the first refusal answers, so later
/// documents are not read): citations, URL and Files API sources, unreadable or over-limit documents.
fn document_text(
    block: &Fields,
    loc: &str,
    documents: &mut document::Budget,
    refusals: &mut Refusals,
) -> Option<String> {
    if refusals.0.is_some() {
        return None;
    }
    let field = |key: &str| block.get(key).unwrap_or(&Value::Null);
    if field("citations")["enabled"] == Value::Bool(true) {
        refusals.note(|| {
            unsupported(
                format!("{loc}.citations"),
                "document citations are not supported; send the document without citations.enabled"
                    .to_owned(),
            )
        });
        return None;
    }
    let meta = document::Meta {
        title: field("title").as_str(),
        context: field("context").as_str(),
    };
    let source = field("source");
    let rendered = match source["type"].as_str().unwrap_or_default() {
        "base64" => {
            let media_type = source["media_type"].as_str().unwrap_or_default();
            match document::Media::of(media_type) {
                Some(media) => {
                    documents.base64(media, source["data"].as_str().unwrap_or_default(), meta)
                }
                None => Err(document::DocumentError::MediaType(
                    media_type.chars().take(64).collect(),
                )),
            }
        }
        "text" => documents.text(source["data"].as_str().unwrap_or_default(), meta),
        "content" => {
            let text = match &source["content"] {
                Value::Array(items) => {
                    let mut texts = Vec::new();
                    for (k, item) in items.iter().enumerate() {
                        match item["type"].as_str() {
                            Some("text") => texts.push(item["text"].as_str().unwrap_or_default()),
                            Some("image") if image_part(item.get("source")) => {
                                refuse_image(refusals, format!("{loc}.source.content[{k}]"));
                                return None;
                            }
                            _ => {}
                        }
                    }
                    texts.join("\n")
                }
                content => content.as_str().unwrap_or_default().to_owned(),
            };
            documents.text(&text, meta)
        }
        "url" => {
            refusals.note(|| {
                unsupported(
                    format!("{loc}.source"),
                    "document URL sources are not fetched; send the document as a base64 or text source".to_owned(),
                )
            });
            return None;
        }
        _ => {
            refusals.note(|| {
                unsupported(
                    format!("{loc}.source"),
                    "Files API document sources (file_id) are not supported; send the document as a base64 or \
                     text source"
                        .to_owned(),
                )
            });
            return None;
        }
    };
    match rendered {
        Ok(text) => Some(text),
        Err(e) => {
            let code = if e.is_unsupported() {
                "unsupported_parameter"
            } else {
                "invalid_value"
            };
            refusals.note(|| error(code, e.to_string(), Some(format!("{loc}.source"))));
            None
        }
    }
}

/// `_text_from_search_result`.
fn search_text(item: &Fields) -> String {
    let mut parts = Vec::new();
    if let Some(title) = item.get("title").filter(|t| truthy(t)) {
        parts.push(format!("Title: {}", py_str_of(title)));
    }
    match item.get("source") {
        Some(Value::Object(source)) => {
            let text = source
                .get("url")
                .filter(|v| truthy(v))
                .or_else(|| source.get("text").filter(|v| truthy(v)));
            if let Some(text) = text {
                parts.push(format!("Source: {}", py_str_of(text)));
            }
        }
        Some(source) if truthy(source) => parts.push(format!("Source: {}", py_str_of(source))),
        _ => {}
    }
    let mut content = Vec::new();
    match item.get("content") {
        Some(Value::String(text)) => content.push(text.clone()),
        Some(Value::Array(items)) => {
            for part in items {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(text) = part.get("text").filter(|t| truthy(t)) {
                        content.push(py_str_of(text));
                    }
                }
            }
        }
        _ => {}
    }
    if !content.is_empty() {
        parts.push(format!("Content: {}", content.join("\n")));
    }
    parts.join("\n")
}

/// Python truthiness of a JSON value.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python `str()` of a JSON scalar as it appears in an f-string (strings verbatim).
fn py_str_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `_convert_tool_result_content`: the tool message contents and the joined text. Every part the pinned template
/// renders as text is flattened into one string (its `render_content` concatenates items); image and
/// `tool_reference` parts refuse.
fn tool_result_content(
    content: Option<&Value>,
    loc: &str,
    documents: &mut document::Budget,
    refusals: &mut Refusals,
) -> (String, String) {
    let Some(Value::Array(items)) = content else {
        let text = match content {
            Some(Value::String(s)) => s.clone(),
            _ => String::new(),
        };
        return (text.clone(), text);
    };
    let mut texts = Vec::new();
    for (k, item) in items.iter().enumerate() {
        let Some(item) = item.as_object() else {
            continue;
        };
        match item.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        texts.push(text.to_owned());
                    }
                }
            }
            Some("image") => {
                if image_part(item.get("source")) {
                    refuse_image(refusals, format!("{loc}[{k}]"));
                }
            }
            Some("tool_reference") => {
                let named = ["tool_name", "name"].iter().any(|key| {
                    item.get(*key)
                        .and_then(Value::as_str)
                        .is_some_and(|n| !n.is_empty())
                });
                if named {
                    refusals.note(|| {
                        unsupported(
                            format!("{loc}[{k}]"),
                            "tool_reference blocks (deferred tools) are not supported".to_owned(),
                        )
                    });
                }
            }
            Some("search_result") => {
                let text = search_text(item);
                if !text.is_empty() {
                    texts.push(text);
                }
            }
            Some("document") => {
                if let Some(text) = document_text(item, &format!("{loc}[{k}]"), documents, refusals)
                {
                    let previous = texts.last().map_or("", String::as_str);
                    texts.push(document::after(previous, text));
                }
            }
            _ => {}
        }
    }
    (texts.concat(), texts.join("\n"))
}

/// The qwen3 reasoning parser's `wrap_reasoning_history`: prior-turn thinking re-wrapped in its own tokens.
fn wrap_reasoning_history(text: &str) -> String {
    format!("<think>{text}\n</think>")
}

/// Request body -> (chat body bytes, echo). `count` converts a count_tokens request (`max_tokens` 1, as SGLang's
/// `handle_count_tokens`). The deployment's `policy` keys the capability fields: M2.12's `thinking_toggle`
/// (`thinking`, `output_config.effort`), M2.7's `top_k` in `sampling_params`, and M2.8's `tool_calls_constrained`
/// decodes (`tool_choice` any/tool).
fn convert_body(
    bytes: &[u8],
    count: bool,
    policy: &ChatPolicy,
) -> Result<(Vec<u8>, Echo), OwnedError> {
    let thinking = policy.thinking_toggle;
    let top_k_capable = policy.sampling_params.iter().any(|p| p == "top_k");
    let named_capable = policy
        .grammar_decodes
        .iter()
        .any(|(output, _)| *output == "tool_calls_constrained");
    let body: Value = serde_json::from_slice(bytes)
        .map_err(|e| error("invalid_json", format!("invalid JSON body: {e}"), None))?;
    let object = body.as_object().ok_or_else(|| {
        error(
            "invalid_json",
            "request body must be a JSON object".to_owned(),
            None,
        )
    })?;
    check_request(object, count)?;
    // ApiLenient AL-4: a reserved CERE extension name is refused here too (SGLang's model drops unknown keys).
    if let Some(e) = object
        .keys()
        .find_map(|key| crate::validate::reserved_extension(key))
    {
        return Err(error(e.code, e.message, e.param));
    }
    // ApiLenient D4/Q6: what the conversion drops stays dropped, but is reported (header + counter).
    let mut ignored: Vec<String> = object
        .iter()
        .filter(|(key, value)| !value.is_null() && !CONVERTED_FIELDS.contains(&key.as_str()))
        .map(|(key, _)| reported_name(key))
        .collect();
    if !count && opt(object, "metadata").is_some() {
        ignored.push("metadata".to_owned());
    }
    let text = std::str::from_utf8(bytes).unwrap_or_default();
    let raw: RawObject = serde_json::from_str(text).unwrap_or_default();
    let mut refusals = Refusals::default();
    let mut documents = document::Budget::default();

    // System: the top-level system, then the inline system messages before the first turn (the pinned template only
    // accepts a leading system message, so SGLang merges them). ApiFit AF-L2: an inline system message after a turn
    // stays at its position; chat validation renders it in place (`validate.rs` `in_place`).
    let mut system_parts: Vec<String> = Vec::new();
    match opt(object, "system") {
        Some(Value::String(system)) if !system.trim().is_empty() => {
            system_parts.push(system.clone())
        }
        Some(Value::Array(blocks)) => {
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        if !text.is_empty() {
                            system_parts.push(text.to_owned());
                        }
                    }
                }
            }
        }
        _ => {}
    }
    let messages = object["messages"].as_array().cloned().unwrap_or_default();
    let leading = messages
        .iter()
        .take_while(|message| message["role"] == "system")
        .count();
    for message in &messages[..leading] {
        if let Some(text) = inline_system_text(&message["content"]) {
            system_parts.push(text);
        }
    }
    let mut chat_messages: Vec<Value> = Vec::new();
    if !system_parts.is_empty() {
        chat_messages.push(json!({"role": "system", "content": system_parts.join("\n")}));
    }
    let raw_messages = raw_list(raw.get("messages").copied());
    for (i, message) in messages.iter().enumerate() {
        let role = message["role"].as_str().unwrap_or_default();
        if role == "system" {
            if i >= leading {
                if let Some(text) = inline_system_text(&message["content"]) {
                    chat_messages.push(json!({"role": "system", "content": text}));
                }
            }
            continue;
        }
        let blocks = match &message["content"] {
            Value::String(content) => {
                chat_messages.push(json!({"role": role, "content": content}));
                continue;
            }
            Value::Array(blocks) => blocks,
            _ => continue,
        };
        let raw_blocks = raw_list(
            raw_messages
                .get(i)
                .and_then(|m| raw_object(m).get("content").copied()),
        );
        let mut parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        if role == "assistant" {
            if blocks.iter().any(|b| b["type"] == "redacted_thinking") {
                return Err(error(
                    "invalid_value",
                    "Anthropic redacted_thinking history is not supported".to_owned(),
                    Some(format!("messages[{i}].content")),
                ));
            }
            let thinking: Vec<&str> = blocks
                .iter()
                .filter(|b| b["type"] == "thinking")
                .filter_map(|b| b["thinking"].as_str())
                .filter(|t| !t.is_empty())
                .collect();
            if !thinking.is_empty() {
                parts.push(wrap_reasoning_history(&thinking.join("\n")));
            }
        }
        for (j, block) in blocks.iter().enumerate() {
            let loc = format!("messages[{i}].content[{j}]");
            match block["type"].as_str().unwrap_or_default() {
                "text" => parts.push(block["text"].as_str().unwrap_or_default().to_owned()),
                "image" => {
                    if image_part(block.get("source")) {
                        refuse_image(&mut refusals, loc);
                    }
                }
                "search_result" => {
                    let text = block.as_object().map(search_text).unwrap_or_default();
                    if !text.is_empty() {
                        parts.push(text);
                    }
                }
                "document" => {
                    let text = block
                        .as_object()
                        .and_then(|b| document_text(b, &loc, &mut documents, &mut refusals));
                    if let Some(text) = text {
                        let previous = parts.last().map_or("", String::as_str);
                        parts.push(document::after(previous, text));
                    }
                }
                "tool_use" => {
                    // SGLang mints `call_<uuid4>` for an empty id; a deterministic id keeps the request digest
                    // (idempotency) stable.
                    let id = match block["id"].as_str() {
                        Some(id) if !id.is_empty() => id.to_owned(),
                        _ => format!("call_m{i}b{j}"),
                    };
                    let input = raw_blocks
                        .get(j)
                        .and_then(|b| raw_object(b).get("input").copied());
                    let arguments = match input {
                        Some(input) => python_json(input.get(), true).unwrap_or_default(),
                        None => "{}".to_owned(),
                    };
                    tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {"name": block["name"], "arguments": arguments},
                    }));
                }
                "tool_result" => {
                    let (content, text) = tool_result_content(
                        opt_value(block, "content"),
                        &loc,
                        &mut documents,
                        &mut refusals,
                    );
                    let id = [block.get("tool_use_id"), block.get("id")]
                        .into_iter()
                        .flatten()
                        .filter_map(Value::as_str)
                        .find(|id| !id.is_empty())
                        .unwrap_or_default();
                    if role == "user" {
                        if !parts.is_empty() {
                            chat_messages.push(json!({"role": "user", "content": parts.concat()}));
                            parts.clear();
                        }
                        chat_messages
                            .push(json!({"role": "tool", "tool_call_id": id, "content": content}));
                    } else {
                        parts.push(format!("Tool result: {text}"));
                    }
                }
                _ => {}
            }
        }
        let mut chat_message = json!({ "role": role });
        if !tool_calls.is_empty() {
            chat_message["tool_calls"] = Value::Array(tool_calls.clone());
        }
        if !parts.is_empty() {
            chat_message["content"] = json!(parts.concat());
        } else if !tool_calls.is_empty() {
            chat_message["content"] = Value::Null;
        } else if role == "user" {
            continue;
        } else {
            chat_message["content"] = json!("");
        }
        chat_messages.push(chat_message);
    }

    // Logged-only fields, as SGLang. ApiFit AF-L7: `thinking.budget_tokens` of `enabled` thinking is the chat body's
    // `thinking_budget` on a native deployment with the thinking capability (enforced by `native/serve.rs`).
    let budget_tokens = opt(object, "thinking")
        .filter(|t| t["type"] == "enabled")
        .and_then(|t| t.get("budget_tokens"))
        .filter(|b| !b.is_null());
    let thinking_budget = budget_tokens
        .and_then(Value::as_u64)
        .filter(|_| !count && thinking && policy.native);
    if budget_tokens
        .filter(|_| thinking_budget.is_none())
        .is_some()
    {
        ignored.push("thinking.budget_tokens".to_owned());
    }
    let output_config = opt(object, "output_config");
    if output_config
        .and_then(|c| opt_value(c, "task_budget"))
        .is_some()
    {
        ignored.push("output_config.task_budget".to_owned());
    }
    if opt(object, "betas").filter(|b| truthy(b)).is_some() {
        ignored.push("betas".to_owned());
    }
    ignored.sort_unstable();
    ignored.dedup();

    // Tools: server tools are skipped; custom tools keep the client's `input_schema` bytes.
    let raw_tools = raw_list(raw.get("tools").copied());
    let mut tools: Vec<ChatTool> = Vec::new();
    if let Some(Value::Array(items)) = opt(object, "tools") {
        for (i, tool) in items.iter().enumerate() {
            if tool_family(tool) != "custom" {
                eprintln!(
                    "cered anthropic: skipping server tool {} (type {})",
                    tool["name"], tool["type"]
                );
                continue;
            }
            if tool["defer_loading"] == true {
                refusals.note(|| {
                    unsupported(
                        format!("tools[{i}].defer_loading"),
                        "deferred tools (defer_loading) are not supported".to_owned(),
                    )
                });
            }
            let schema = raw_tools
                .get(i)
                .and_then(|t| raw_object(t).get("input_schema").copied());
            let has_type = tool["input_schema"].get("type").is_some();
            let parameters = match schema {
                Some(schema) if has_type => schema.to_owned(),
                Some(schema) => with_object_type(schema.get())?,
                None => continue,
            };
            tools.push(ChatTool {
                kind: "function",
                function: ChatFunction {
                    description: tool["description"].as_str().unwrap_or_default(),
                    name: tool["name"].as_str().unwrap_or_default(),
                    parameters,
                },
            });
        }
    }

    // tool_choice: SGLang's own errors first, then any/tool as SGLang converts them (`required`, a named function)
    // on a profile with M2.8's constrained tool calls, refused elsewhere.
    let mut tool_choice = json!(if tools.is_empty() { "none" } else { "auto" });
    if let Some(choice) = opt(object, "tool_choice") {
        let kind = choice["type"].as_str().unwrap_or_default();
        if kind == "none" {
            tool_choice = json!("none");
        } else if !tools.is_empty() {
            if kind == "tool" {
                let name = choice["name"].as_str();
                if !tools.iter().any(|t| Some(t.function.name) == name) {
                    return Err(error(
                        "invalid_value",
                        format!(
                            "tool_choice references tool {} but it is not in the forwarded tools list \
                             (server-side Anthropic tools cannot be selected)",
                            python_repr_str(name)
                        ),
                        Some("tool_choice.name".to_owned()),
                    ));
                }
            }
            if kind == "any" && named_capable {
                tool_choice = json!("required");
            } else if kind == "tool" && named_capable {
                tool_choice = json!({"type": "function", "function": {"name": choice["name"]}});
            } else if kind == "any" || kind == "tool" {
                refusals.note(|| {
                    unsupported(
                        "tool_choice.type".to_owned(),
                        format!(
                            "tool_choice {kind:?} needs the M2.8 required/named tool_choice capability, which this \
                             deployment does not have"
                        ),
                    )
                });
            }
        } else if kind == "any" || kind == "tool" {
            return Err(error(
                "invalid_value",
                format!(
                    "tool_choice='{kind}' requires at least one custom tool; all supplied tools were server-side \
                     Anthropic built-ins which the OpenAI-compatible backend cannot invoke"
                ),
                Some("tool_choice.type".to_owned()),
            ));
        }
    }

    // Capability-keyed fields (ruling 2), in a fixed order after every SGLang error.
    if !count && !top_k_capable && opt(object, "top_k").is_some() {
        return Err(unsupported(
            "top_k".to_owned(),
            "top_k needs the M2.7 sampling capability, which this deployment does not have"
                .to_owned(),
        ));
    }
    let refusal = match refusals.0.take() {
        Some(r) if r.param.as_deref() != Some("tool_choice.type") => return Err(r),
        choice => choice,
    };
    // M2.14 D3 with M2.12: `thinking` enabled/adaptive -> `enable_thinking: true`; disabled/absent is the deployment
    // default. `output_config.effort` low/medium -> `reasoning_effort` (m212 D13: effort != none => thinking on);
    // SGLang's other efforts (`xhigh` rewritten to `max`) are ones the pinned template rejects.
    let mut chat_template_kwargs = None;
    if let Some(kind) = opt(object, "thinking").and_then(|t| t["type"].as_str()) {
        if kind != "disabled" {
            if !thinking {
                return Err(unsupported(
                    "thinking.type".to_owned(),
                    format!(
                        "thinking {kind:?} needs the M2.12 thinking capability, which this deployment does not \
                         have"
                    ),
                ));
            }
            chat_template_kwargs = Some(json!({"enable_thinking": true}));
        }
    }
    let reasoning_effort = match output_config.and_then(|c| c["effort"].as_str()) {
        Some(effort @ ("low" | "medium")) if thinking => Some(effort),
        Some(effort @ ("low" | "medium")) => {
            return Err(unsupported(
                "output_config.effort".to_owned(),
                format!(
                    "output_config.effort {effort:?} needs the M2.12 thinking capability, which this deployment \
                     does not have"
                ),
            ))
        }
        Some(effort) => {
            return Err(error(
                "invalid_value",
                format!("output_config.effort {effort:?} is not supported; use \"low\" or \"medium\" (m212 D13)"),
                Some("output_config.effort".to_owned()),
            ))
        }
        None => None,
    };
    if let Some(refusal) = refusal {
        return Err(refusal);
    }

    let one = json!(1);
    let stream = !count && object.get("stream").and_then(Value::as_bool) == Some(true);
    let model = object["model"].as_str().unwrap_or_default();
    let chat = ChatBody {
        messages: chat_messages,
        model,
        max_tokens: if count { &one } else { &object["max_tokens"] },
        stop: if count {
            None
        } else {
            opt(object, "stop_sequences").filter(|s| s.as_array().is_some_and(|s| !s.is_empty()))
        },
        stream,
        stream_options: stream.then(|| json!({"include_usage": true})),
        temperature: if count {
            None
        } else {
            opt(object, "temperature").and_then(Value::as_f64)
        },
        top_p: if count {
            None
        } else {
            opt(object, "top_p").and_then(Value::as_f64)
        },
        top_k: if count { None } else { opt(object, "top_k") },
        tools,
        tool_choice,
        chat_template_kwargs,
        reasoning_effort,
        thinking_budget,
    };
    let chat = serde_json::to_vec(&chat).map_err(|e| {
        error(
            "invalid_value",
            format!("request could not be converted: {e}"),
            None,
        )
    })?;
    let mut echo = Echo::for_model(model);
    echo.ignored = ignored;
    Ok((chat, echo))
}

fn opt_value<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.get(key).filter(|v| !v.is_null())
}

/// `_extract_system_text`: a system message's text blocks, each trimmed, joined by newlines.
fn inline_system_text(content: &Value) -> Option<String> {
    match content {
        Value::String(s) => Some(s.trim().to_owned()).filter(|s| !s.is_empty()),
        Value::Array(blocks) => {
            let texts: Vec<&str> = blocks
                .iter()
                .filter(|b| b["type"] == "text")
                .filter_map(|b| b["text"].as_str())
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .collect();
            (!texts.is_empty()).then(|| texts.join("\n"))
        }
        _ => None,
    }
}

/// `input_schema` without a top-level `type` gains `"type": "object"` (protocol.py `_ensure_object_schema`, which
/// appends the key); the client's other bytes are kept.
fn with_object_type(schema: &str) -> Result<Box<RawValue>, OwnedError> {
    let trimmed = schema.trim_end();
    let inner = trimmed.strip_suffix('}').unwrap_or(trimmed);
    let empty = inner
        .trim_start()
        .strip_prefix('{')
        .is_some_and(|rest| rest.trim().is_empty());
    let text = if empty {
        format!("{inner}\"type\":\"object\"}}")
    } else {
        format!("{inner},\"type\":\"object\"}}")
    };
    RawValue::from_string(text).map_err(|e| {
        error(
            "invalid_value",
            format!("input_schema could not be completed: {e}"),
            None,
        )
    })
}

/// Python `repr()` of an optional string, as SGLang's `{tool_name!r}`.
fn python_repr_str(value: Option<&str>) -> String {
    match value {
        None => "None".to_owned(),
        Some(s) => {
            let quote = if s.contains('\'') && !s.contains('"') {
                '"'
            } else {
                '\''
            };
            let mut out = String::new();
            out.push(quote);
            for c in s.chars() {
                match c {
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if c == quote => {
                        out.push('\\');
                        out.push(c);
                    }
                    c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                        let _ = write!(out, "\\x{:02x}", c as u32);
                    }
                    c => out.push(c),
                }
            }
            out.push(quote);
            out
        }
    }
}

/// A `/v1/messages` request -> the chat body bytes and the request's echo, for a deployment of `policy`.
pub fn convert_request(bytes: &[u8], policy: &ChatPolicy) -> Result<(Vec<u8>, Echo), OwnedError> {
    convert_body(bytes, false, policy)
}

/// A `/v1/messages/count_tokens` request -> the chat body whose rendered prompt is counted (SGLang builds a
/// messages request with `max_tokens=1` and converts it).
pub fn convert_count_tokens(bytes: &[u8], policy: &ChatPolicy) -> Result<Vec<u8>, OwnedError> {
    convert_body(bytes, true, policy).map(|(chat, _)| chat)
}

/// The count_tokens response body `{"input_tokens": n}`.
pub fn count_tokens_body(input_tokens: u64) -> Vec<u8> {
    serde_json::to_vec(&json!({ "input_tokens": input_tokens })).unwrap_or_default()
}

// ---------------------------------------------------------------------------------------------------------------
// Output (`_convert_response`, `_generate_anthropic_stream`)

/// `msg_<32 hex>`, as SGLang's `msg_{uuid4().hex}` (per-run, `volatile` in the goldens).
fn random_hex() -> String {
    let a = RandomState::new().build_hasher().finish();
    let b = RandomState::new().build_hasher().finish();
    format!("{a:016x}{b:016x}")
}

/// The request's echo: what the renders need from the request.
pub struct Echo {
    model: String,
    /// ApiLenient D4: the fields the conversion dropped (sorted), for `X-Cere-Ignored-Params` and the counter.
    ignored: Vec<String>,
}

impl Echo {
    /// The echo of a request for `model`.
    pub fn for_model(model: &str) -> Self {
        Echo {
            model: model.to_owned(),
            ignored: Vec::new(),
        }
    }
}

/// The top-level fields the conversion reads ([`check_request`], [`convert_body`]); `metadata` and `betas` are
/// reported by their own rules, every other key is dropped and reported (ApiLenient Q6).
const CONVERTED_FIELDS: [&str; 15] = [
    "betas",
    "max_tokens",
    "messages",
    "metadata",
    "model",
    "output_config",
    "stop_sequences",
    "stream",
    "system",
    "temperature",
    "thinking",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
];

/// A dropped key as the header reports it: verbatim when it is a plain field name, else
/// [`crate::validate::UNLISTED_PARAM`] (the header carries no client bytes it cannot hold).
fn reported_name(key: &str) -> String {
    let plain = !key.is_empty()
        && key.len() <= 64
        && key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.');
    if plain {
        key.to_owned()
    } else {
        crate::validate::UNLISTED_PARAM.to_owned()
    }
}

#[derive(Serialize, Default)]
struct Usage {
    #[serde(skip_serializing_if = "Option::is_none")]
    input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_read_input_tokens: Option<u64>,
}

/// `_anthropic_usage_from_openai(include_input=True, include_output=True)`.
fn usage_of(usage: Option<&Value>) -> Usage {
    let Some(usage) = usage.filter(|u| u.is_object()) else {
        return Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            cache_read_input_tokens: None,
        };
    };
    let prompt = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    Usage {
        input_tokens: Some(prompt.saturating_sub(cached)),
        output_tokens: Some(usage["completion_tokens"].as_u64().unwrap_or(0)),
        cache_read_input_tokens: (cached > 0).then_some(cached),
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Block<'a> {
    Thinking {
        thinking: &'a str,
    },
    Text {
        text: &'a str,
    },
    ToolUse {
        id: &'a str,
        name: &'a str,
        input: Box<RawValue>,
    },
}

#[derive(Serialize)]
struct Message<'a> {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    content: Vec<Block<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<&'static str>,
    usage: Usage,
}

/// `STOP_REASON_MAP`; anything else (abort, ...) is `end_turn`. A matched stop string is `end_turn` with no
/// `stop_sequence`, as SGLang (ruling 3).
fn stop_reason(finish: Option<&str>) -> &'static str {
    match finish.unwrap_or("stop") {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    }
}

/// A tool call's `arguments` as the `input` object (`json.loads`, key order kept); not an object -> `{}`.
fn tool_input(arguments: Option<&str>) -> Box<RawValue> {
    let compact = arguments
        .filter(|a| a.trim_start().starts_with('{'))
        .and_then(|a| python_json(a, false))
        .unwrap_or_else(|| "{}".to_owned());
    RawValue::from_string(compact).unwrap_or_else(|_| empty_object())
}

fn empty_object() -> Box<RawValue> {
    RawValue::from_string("{}".to_owned()).expect("{} is JSON")
}

fn render_message(chat: &Value, fallback_model: &str) -> Vec<u8> {
    let model = chat["model"].as_str().unwrap_or(fallback_model);
    let Some(choice) = chat["choices"].get(0) else {
        let message = Message {
            id: format!("msg_{}", random_hex()),
            kind: "message",
            role: "assistant",
            content: vec![Block::Text { text: "" }],
            model,
            stop_reason: Some("end_turn"),
            usage: usage_of(None),
        };
        return serde_json::to_vec(&message).unwrap_or_default();
    };
    let message = &choice["message"];
    let mut content = Vec::new();
    if let Some(thinking) = message["reasoning_content"]
        .as_str()
        .filter(|t| !t.is_empty())
    {
        content.push(Block::Thinking { thinking });
    }
    if let Some(text) = message["content"].as_str().filter(|t| !t.is_empty()) {
        content.push(Block::Text { text });
    }
    if let Some(calls) = message["tool_calls"].as_array() {
        for call in calls {
            content.push(Block::ToolUse {
                id: call["id"].as_str().unwrap_or_default(),
                name: call["function"]["name"].as_str().unwrap_or_default(),
                input: tool_input(call["function"]["arguments"].as_str()),
            });
        }
    }
    if content.is_empty() {
        content.push(Block::Text { text: "" });
    }
    let message = Message {
        id: format!("msg_{}", random_hex()),
        kind: "message",
        role: "assistant",
        content,
        model,
        stop_reason: Some(stop_reason(choice["finish_reason"].as_str())),
        usage: usage_of(chat.get("usage")),
    };
    serde_json::to_vec(&message).unwrap_or_default()
}

impl SurfaceRender for Echo {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn ignored_params(&self) -> &[String] {
        &self.ignored
    }

    fn render_unary(&self, chat: &Value, _rid: &str) -> Vec<u8> {
        render_message(chat, &self.model)
    }

    fn error_body(&self, e: &ErrorView) -> Vec<u8> {
        error_body(e)
    }

    fn stream(&self) -> Box<dyn SurfaceStream> {
        Box::new(Stream {
            model: self.model.clone(),
            index: 0,
            open: None,
            output_tokens: None,
            start_input_tokens: None,
            final_input: None,
            finish: None,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Thinking,
    Text,
    ToolUse,
}

/// One request's SSE renderer (SGLang's `_generate_anthropic_stream` state).
struct Stream {
    model: String,
    index: usize,
    open: Option<Kind>,
    output_tokens: Option<u64>,
    /// ApiFit AF-L8: `message_start.usage.input_tokens` as sent (the prompt count when the backend knew it at
    /// open, else 0).
    start_input_tokens: Option<u64>,
    /// The final usage's `(input_tokens, cache_read_input_tokens)` (prompt minus prefix-cache hits, the hits).
    final_input: Option<(u64, u64)>,
    finish: Option<String>,
}

fn sse(out: &mut Vec<u8>, event: &str, data: &impl Serialize) {
    out.extend_from_slice(b"event: ");
    out.extend_from_slice(event.as_bytes());
    out.extend_from_slice(b"\ndata: ");
    let _ = serde_json::to_writer(&mut *out, data);
    out.extend_from_slice(b"\n\n");
}

impl Stream {
    fn close(&mut self, out: &mut Vec<u8>) {
        if self.open.take().is_some() {
            sse(
                out,
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": self.index}),
            );
            self.index += 1;
        }
    }

    /// Opens a block of `kind` (closing the open one if its kind differs, or always with `force_new`).
    fn ensure(&mut self, kind: Kind, block: Block, force_new: bool, out: &mut Vec<u8>) {
        if self.open.is_some() && (force_new || self.open != Some(kind)) {
            self.close(out);
        }
        if self.open.is_none() {
            #[derive(Serialize)]
            struct Start<'a> {
                #[serde(rename = "type")]
                kind: &'static str,
                index: usize,
                content_block: Block<'a>,
            }
            sse(
                out,
                "content_block_start",
                &Start {
                    kind: "content_block_start",
                    index: self.index,
                    content_block: block,
                },
            );
            self.open = Some(kind);
        }
    }

    fn delta(&self, delta: Value, out: &mut Vec<u8>) {
        sse(
            out,
            "content_block_delta",
            &json!({"type": "content_block_delta", "index": self.index, "delta": delta}),
        );
    }

    /// One chunk choice's delta: thinking, then tool calls, then text, as SGLang orders them.
    fn choice(&mut self, choice: &Value, out: &mut Vec<u8>) {
        if let Some(finish) = choice["finish_reason"].as_str() {
            self.finish = Some(finish.to_owned());
        }
        let delta = &choice["delta"];
        if let Some(thinking) = delta["reasoning_content"]
            .as_str()
            .filter(|t| !t.is_empty())
        {
            self.ensure(Kind::Thinking, Block::Thinking { thinking: "" }, false, out);
            self.delta(json!({"type": "thinking_delta", "thinking": thinking}), out);
        }
        if let Some(calls) = delta["tool_calls"].as_array() {
            for call in calls {
                let function = &call["function"];
                let arguments = function["arguments"].as_str().filter(|a| !a.is_empty());
                if let Some(name) = function["name"].as_str().filter(|n| !n.is_empty()) {
                    let minted;
                    let id = match call["id"].as_str() {
                        Some(id) if !id.is_empty() => id,
                        _ => {
                            minted = format!("toolu_{}", random_hex());
                            &minted
                        }
                    };
                    let block = Block::ToolUse {
                        id,
                        name,
                        input: empty_object(),
                    };
                    self.ensure(Kind::ToolUse, block, true, out);
                    if let Some(arguments) = arguments {
                        self.delta(
                            json!({"type": "input_json_delta", "partial_json": arguments}),
                            out,
                        );
                    }
                } else if let Some(arguments) = arguments {
                    if self.open == Some(Kind::ToolUse) {
                        self.delta(
                            json!({"type": "input_json_delta", "partial_json": arguments}),
                            out,
                        );
                    }
                }
            }
        }
        if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            self.ensure(Kind::Text, Block::Text { text: "" }, false, out);
            self.delta(json!({"type": "text_delta", "text": text}), out);
        }
    }

    fn note_usage(&mut self, usage: Option<&Value>) {
        if let Some(tokens) = usage.and_then(|u| u["completion_tokens"].as_u64()) {
            self.output_tokens = Some(tokens);
        }
        if let Some(prompt) = usage.and_then(|u| u["prompt_tokens"].as_u64()) {
            let cached = usage
                .and_then(|u| u["prompt_tokens_details"]["cached_tokens"].as_u64())
                .unwrap_or(0);
            self.final_input = Some((prompt.saturating_sub(cached), cached));
        }
    }

    /// ApiFit AF-L8: the `message_delta` usage. Anthropic's final usage is cumulative; the input fields enter when
    /// they differ from `message_start`'s (the prompt count was unknown at open, or prefix-cache hits moved
    /// tokens to `cache_read_input_tokens`), so a stream whose `message_start` was exact renders as SGLang.
    fn delta_usage(&self) -> Value {
        let mut usage = json!({"output_tokens": self.output_tokens.unwrap_or(0)});
        if let Some((input, cached)) = self.final_input {
            if cached > 0 || self.start_input_tokens != Some(input) {
                usage["input_tokens"] = json!(input);
                if cached > 0 {
                    usage["cache_read_input_tokens"] = json!(cached);
                }
            }
        }
        usage
    }
}

/// A chunk's first choice; a bare choice (`finish_reason`/`delta` at the top) is its own choice.
fn first_choice(chunk: &Value) -> Option<&Value> {
    match chunk.get("choices") {
        Some(choices) => choices.get(0),
        None => Some(chunk),
    }
}

impl SurfaceStream for Stream {
    /// `message_start`: ApiFit AF-L8, `input_tokens` is the prompt's token count when the backend knows it at open
    /// (a local native generation: the encoded prompt), else 0 (SGLang and remote workers report it only with the
    /// final usage, which `message_delta` then carries). Prefix-cache hits are known only after the PREFILL:
    /// `message_delta` moves them to `cache_read_input_tokens`.
    fn open(&mut self, prompt_tokens: Option<u64>) -> Vec<u8> {
        self.start_input_tokens = Some(prompt_tokens.unwrap_or(0));
        #[derive(Serialize)]
        struct Start<'a> {
            #[serde(rename = "type")]
            kind: &'static str,
            message: Message<'a>,
        }
        let mut out = Vec::new();
        sse(
            &mut out,
            "message_start",
            &Start {
                kind: "message_start",
                message: Message {
                    id: format!("msg_{}", random_hex()),
                    kind: "message",
                    role: "assistant",
                    content: Vec::new(),
                    model: &self.model,
                    stop_reason: None,
                    usage: Usage {
                        input_tokens: self.start_input_tokens,
                        output_tokens: Some(0),
                        cache_read_input_tokens: None,
                    },
                },
            },
        );
        out
    }

    fn chunk(&mut self, chunk: &Value) -> Vec<u8> {
        let mut out = Vec::new();
        self.note_usage(chunk.get("usage"));
        if let Some(choice) = chunk.get("choices").and_then(|c| c.get(0)) {
            self.choice(choice, &mut out);
        }
        out
    }

    fn complete(&mut self, finish: &Value, usage: Option<&Value>) -> Vec<u8> {
        let mut out = Vec::new();
        if let Some(choice) = first_choice(finish) {
            self.choice(choice, &mut out);
        }
        self.note_usage(usage);
        self.close(&mut out);
        sse(
            &mut out,
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason(self.finish.as_deref())},
                "usage": self.delta_usage(),
            }),
        );
        sse(&mut out, "message_stop", &json!({"type": "message_stop"}));
        out
    }

    /// SGLang's `_flush_on_error`: close the open block, the error event, `message_stop`.
    fn fail(&mut self, e: &ErrorView) -> Vec<u8> {
        let mut out = Vec::new();
        self.close(&mut out);
        let message = client_message(e);
        sse(
            &mut out,
            "error",
            &ErrorEnvelope {
                kind: "error",
                error: ErrorPayload {
                    kind: error_type(e.status),
                    message: &message,
                },
            },
        );
        sse(&mut out, "message_stop", &json!({"type": "message_stop"}));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_without_type_gains_object_type() {
        assert_eq!(
            with_object_type("{ }").unwrap().get(),
            "{ \"type\":\"object\"}"
        );
        assert_eq!(
            with_object_type(r#"{"properties": {}} "#).unwrap().get(),
            r#"{"properties": {},"type":"object"}"#
        );
    }
}
