//! The stateless OpenAI Responses surface (m214 design note D4): `POST /v1/responses` as a host-side translation
//! into the chat pipeline, mirroring SGLang's non-harmony `OpenAIServingResponses` for the subset below.
//!
//! * [`convert`]: the Responses request to the CERE chat body (validated by the chat validator afterwards) plus
//!   the [`Echo`] the response renders. `input` as a string or items (messages, `function_call`,
//!   `function_call_output`), `instructions`, function tools, `tool_choice` `auto`/`none`, `max_output_tokens`,
//!   `temperature`, `top_p`, `stream`. Stateless: `previous_response_id`, `background: true`, built-in tools and
//!   `store: true` are refused; an absent `store` is echoed `false`. Capability-keyed values (`tool_choice`
//!   `required`/named: M2.8; image parts: M2.13) are refused (TrackM214 ruling 2); `reasoning.effort` converts on a
//!   profile declaring M2.12's `thinking_toggle` and is refused elsewhere.
//! * [`render_unary`]: the `response` object of SGLang's unary path from a chat completion.
//! * [`ResponsesStream`]: the SSE events of `responses_stream_generator_non_harmony` from chat chunks.
//! * [`error_body`]: SGLang's error envelope `{"error":{message, type, param: null, code: <status>}}`.
//! * [`Responses`]: the request's [`SurfaceRender`] (unary body, error envelope, [`ResponsesStream`] as the
//!   relay's [`SurfaceStream`]).
//!
//! Bodies are [`Node`] trees: this crate's `serde_json` sorts object keys, and SGLang's key order (and the
//! client's `tools[].parameters` order, which reaches the chat template) must survive.
use std::fmt;
use std::io;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::{Path as UrlPath, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;

use serde::de::{Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};
use sha2::{Digest, Sha256};

use crate::api_types::{ApiError, INVALID_REQUEST, SERVER_ERROR};
use crate::app::App;
use crate::surface::{
    CompatApi, CompatKind, ConvertCtx, ErrorView, OwnedError, SurfaceRender, SurfaceStream,
};
use crate::validate::ChatPolicy;

/// An order-preserving JSON value (object members keep their wire order).
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Null,
    Bool(bool),
    Num(Number),
    Str(String),
    Arr(Vec<Node>),
    Obj(Vec<(String, Node)>),
}

impl Node {
    /// The member `key` of an object.
    pub fn get(&self, key: &str) -> Option<&Node> {
        match self {
            Node::Obj(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Node::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Node::Num(n) => n.as_u64(),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Node::Num(n) => n.as_f64(),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Node::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// RFC 6901 lookup, as `serde_json::Value::pointer`.
    pub fn pointer(&self, pointer: &str) -> Option<&Node> {
        let mut node = self;
        for token in tokens(pointer)? {
            node = match node {
                Node::Obj(members) => members.iter().find(|(k, _)| *k == token).map(|(_, v)| v)?,
                Node::Arr(items) => items.get(token.parse::<usize>().ok()?)?,
                _ => return None,
            };
        }
        Some(node)
    }

    pub fn pointer_mut(&mut self, pointer: &str) -> Option<&mut Node> {
        let mut node = self;
        for token in tokens(pointer)? {
            node = match node {
                Node::Obj(members) => members
                    .iter_mut()
                    .find(|(k, _)| *k == token)
                    .map(|(_, v)| v)?,
                Node::Arr(items) => items.get_mut(token.parse::<usize>().ok()?)?,
                _ => return None,
            };
        }
        Some(node)
    }

    /// The tree of a `serde_json::Value` (object members in the `Value`'s order).
    pub fn from_value(value: &Value) -> Node {
        match value {
            Value::Null => Node::Null,
            Value::Bool(b) => Node::Bool(*b),
            Value::Number(n) => Node::Num(n.clone()),
            Value::String(s) => Node::Str(s.clone()),
            Value::Array(items) => Node::Arr(items.iter().map(Node::from_value).collect()),
            Value::Object(map) => Node::Obj(
                map.iter()
                    .map(|(k, v)| (k.clone(), Node::from_value(v)))
                    .collect(),
            ),
        }
    }

    /// The compact JSON text.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("a Node always serializes")
    }

    fn is_null(&self) -> bool {
        matches!(self, Node::Null)
    }
}

fn tokens(pointer: &str) -> Option<Vec<String>> {
    if pointer.is_empty() {
        return Some(Vec::new());
    }
    let rest = pointer.strip_prefix('/')?;
    Some(
        rest.split('/')
            .map(|t| t.replace("~1", "/").replace("~0", "~"))
            .collect(),
    )
}

fn obj(members: Vec<(&str, Node)>) -> Node {
    Node::Obj(
        members
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
    )
}

fn s(text: &str) -> Node {
    Node::Str(text.to_owned())
}

fn n(value: u64) -> Node {
    Node::Num(Number::from(value))
}

impl Serialize for Node {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Node::Null => serializer.serialize_unit(),
            Node::Bool(b) => serializer.serialize_bool(*b),
            Node::Num(n) => n.serialize(serializer),
            Node::Str(s) => serializer.serialize_str(s),
            Node::Arr(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Node::Obj(members) => {
                let mut map = serializer.serialize_map(Some(members.len()))?;
                for (k, v) in members {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Node {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct NodeVisitor;
        impl<'de> Visitor<'de> for NodeVisitor {
            type Value = Node;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_unit<E>(self) -> Result<Node, E> {
                Ok(Node::Null)
            }
            fn visit_none<E>(self) -> Result<Node, E> {
                Ok(Node::Null)
            }
            fn visit_bool<E>(self, b: bool) -> Result<Node, E> {
                Ok(Node::Bool(b))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Node, E> {
                Ok(Node::Num(v.into()))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Node, E> {
                Ok(Node::Num(v.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Node, E> {
                Number::from_f64(v)
                    .map(Node::Num)
                    .ok_or_else(|| E::custom("non-finite number"))
            }
            fn visit_str<E>(self, v: &str) -> Result<Node, E> {
                Ok(Node::Str(v.to_owned()))
            }
            fn visit_string<E>(self, v: String) -> Result<Node, E> {
                Ok(Node::Str(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Node, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(Node::Arr(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Node, A::Error> {
                let mut members: Vec<(String, Node)> = Vec::new();
                while let Some((k, v)) = map.next_entry::<String, Node>()? {
                    // A repeated key keeps its last value, as serde_json's map.
                    match members.iter_mut().find(|(key, _)| *key == k) {
                        Some(slot) => slot.1 = v,
                        None => members.push((k, v)),
                    }
                }
                Ok(Node::Obj(members))
            }
        }
        deserializer.deserialize_any(NodeVisitor)
    }
}

/// SGLang's per-object ids: `resp_`/`msg_`/`rs_` + 32 hex digits (`random_uuid()`), `fc_` + 8, `call_` + 24.
/// Derived from a 16-byte seed and a counter (SHA-256), so a seeded generator is reproducible in tests.
pub struct Ids {
    seed: [u8; 16],
    counter: u64,
}

impl Ids {
    pub fn seeded(seed: [u8; 16]) -> Self {
        Self { seed, counter: 0 }
    }

    /// Seeded from the kernel CSPRNG.
    pub fn random() -> io::Result<Self> {
        Ok(Self::seeded(crate::digest::random_bytes::<16>()?))
    }

    fn hex(&mut self, digits: usize) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.seed);
        hasher.update(self.counter.to_le_bytes());
        self.counter += 1;
        let digest = hasher.finalize();
        let mut out = String::with_capacity(64);
        for byte in digest.iter() {
            out.push_str(&format!("{byte:02x}"));
        }
        out.truncate(digits);
        out
    }

    fn id(&mut self, prefix: &str, digits: usize) -> String {
        format!("{prefix}{}", self.hex(digits))
    }
}

/// The request values the response object echoes.
#[derive(Clone, Debug)]
pub struct Echo {
    pub model: String,
    instructions: Node,
    max_output_tokens: Node,
    temperature: Node,
    top_p: Node,
    tool_choice: Node,
    /// The function tools in SGLang's echo form (`{type, name, description, parameters, strict, tools}`).
    tools: Vec<Node>,
    /// ApiLenient D4: the inert fields the request carried (sorted), for `X-Cere-Ignored-Params` and the counter.
    pub ignored: Vec<String>,
}

/// A converted request: the CERE chat body (`stream` included), validated by the chat validator next.
#[derive(Clone, Debug)]
pub struct Converted {
    pub chat: Vec<u8>,
    pub echo: Echo,
    pub stream: bool,
}

/// SGLang's `ResponsesRequest.to_sampling_params` headroom: `max_new_tokens = max_output_tokens - 2`.
const OUTPUT_HEADROOM: u64 = 2;

const FIELDS: [&str; 19] = [
    "background",
    "input",
    "instructions",
    "max_output_tokens",
    "metadata",
    "model",
    "previous_response_id",
    "prompt_cache_key",
    "prompt_cache_retention",
    "reasoning",
    "safety_identifier",
    "service_tier",
    "store",
    "stream",
    "temperature",
    "tool_choice",
    "tools",
    "top_p",
    "user",
];

/// ApiLenient D4: the inert Responses fields (`validate::inert_value` rules; `store: false` is checked below).
const INERT_FIELDS: [&str; 6] = [
    "metadata",
    "prompt_cache_key",
    "prompt_cache_retention",
    "safety_identifier",
    "service_tier",
    "user",
];

fn stateless(param: &str, what: &str) -> ApiError {
    ApiError::unsupported_parameter(
        param,
        format!("{what} needs a response store; cered's /v1/responses is stateless"),
    )
}

/// `field` of `object` (`null` = absent).
fn field<'a>(object: &'a Node, key: &str) -> Option<&'a Node> {
    object.get(key).filter(|v| !v.is_null())
}

fn known(object: &[(String, Node)], at: &str, keys: &[&str]) -> Result<(), ApiError> {
    match object.iter().find(|(k, _)| !keys.contains(&k.as_str())) {
        Some((key, _)) => Err(ApiError::unsupported_parameter(
            format!("{at}{key}"),
            format!("field '{at}{key}' is not supported"),
        )),
        None => Ok(()),
    }
}

fn boolean(object: &Node, key: &str) -> Result<Option<bool>, ApiError> {
    match field(object, key) {
        None => Ok(None),
        Some(Node::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(ApiError::invalid_value(
            key,
            format!("{key} must be a boolean"),
        )),
    }
}

/// The Responses request `body` as a CERE chat body for a deployment of `policy`. Native (`policy.native`): an
/// absent temperature/top_p stays absent (native samples them with the model's `generation_config.json` defaults,
/// m27 D10); SGLang-backed bodies carry SGLang's effective `temperature` 0.7 and `top_p` 1.0 when absent
/// (TrackM214 ruling 3). `reasoning.effort` needs the profile's M2.12 `thinking_toggle`.
pub fn convert(body: &[u8], policy: &ChatPolicy) -> Result<Converted, ApiError> {
    let native = policy.native;
    let request: Node = serde_json::from_slice(body)
        .map_err(|e| ApiError::invalid_json(format!("request body is not valid JSON: {e}")))?;
    let Node::Obj(members) = &request else {
        return Err(ApiError::invalid_json("request body must be a JSON object"));
    };
    if let Some((key, _)) = members.iter().find(|(k, _)| !FIELDS.contains(&k.as_str())) {
        if let Some(refusal) = crate::validate::reserved_extension(key) {
            return Err(refusal);
        }
        return Err(ApiError::unsupported_parameter(
            key.as_str(),
            format!("parameter '{key}' is not supported"),
        ));
    }
    if field(&request, "previous_response_id").is_some() {
        return Err(stateless("previous_response_id", "previous_response_id"));
    }
    if boolean(&request, "background")? == Some(true) {
        return Err(stateless("background", "background: true"));
    }
    let mut ignored: Vec<String> = Vec::new();
    match boolean(&request, "store")? {
        Some(true) => return Err(stateless("store", "store: true")),
        Some(false) => ignored.push("store".to_owned()),
        None => {}
    }
    for key in INERT_FIELDS {
        if let Some(value) = field(&request, key) {
            let value: serde_json::Value = serde_json::from_slice(&value.to_bytes())
                .map_err(|e| ApiError::invalid_json(e.to_string()))?;
            crate::validate::inert_value(key, &value)?;
            ignored.push(key.to_owned());
        }
    }
    ignored.sort_unstable();
    let stream = boolean(&request, "stream")?.unwrap_or(false);
    let model = match field(&request, "model") {
        None => return Err(ApiError::missing_parameter("model")),
        Some(Node::Str(model)) => model.clone(),
        Some(_) => return Err(ApiError::invalid_value("model", "model must be a string")),
    };
    let instructions = match field(&request, "instructions") {
        None => None,
        Some(Node::Str(text)) => Some(text.clone()),
        Some(_) => {
            return Err(ApiError::invalid_value(
                "instructions",
                "instructions must be a string",
            ))
        }
    };
    let tools = tools(field(&request, "tools"))?;
    let tool_choice = tool_choice(field(&request, "tool_choice"), !tools.is_empty())?;
    let effort = reasoning(field(&request, "reasoning"), policy.thinking_toggle)?;
    let mut messages = input(field(&request, "input"))?;
    if let Some(text) = &instructions {
        // SGLang's `_construct_input_messages`: the instructions become the system message; a leading system
        // message of the input is appended to it.
        match messages.first_mut() {
            Some(Node::Obj(first)) if first[0].1.as_str() == Some("system") => {
                if let Some((_, Node::Str(content))) =
                    first.iter_mut().find(|(k, _)| k == "content")
                {
                    *content = format!("{text}\n\n{content}");
                }
            }
            _ => messages.insert(0, obj(vec![("role", s("system")), ("content", s(text))])),
        }
    }
    let max_output_tokens = match field(&request, "max_output_tokens") {
        None => None,
        Some(value) => match value.as_u64() {
            Some(tokens) if tokens > OUTPUT_HEADROOM => Some(tokens),
            _ => {
                return Err(ApiError::invalid_value(
                    "max_output_tokens",
                    format!(
                        "max_output_tokens must be an integer greater than {OUTPUT_HEADROOM} (SGLang reserves \
                         {OUTPUT_HEADROOM} tokens of the budget)"
                    ),
                ))
            }
        },
    };
    let number = |key: &str| match field(&request, key) {
        None => Ok(None),
        Some(value @ Node::Num(_)) => Ok(Some(value.clone())),
        Some(_) => Err(ApiError::invalid_value(
            key,
            format!("{key} must be a number"),
        )),
    };
    let temperature = number("temperature")?;
    let top_p = number("top_p")?;

    let mut chat = vec![("model", s(&model)), ("messages", Node::Arr(messages))];
    if let Some(tokens) = max_output_tokens {
        chat.push(("max_completion_tokens", n(tokens - OUTPUT_HEADROOM)));
    }
    let sampling = |value: &Option<Node>, sglang_default: f64| match value {
        Some(value) => Some(value.clone()),
        None if native => None,
        None => Number::from_f64(sglang_default).map(Node::Num),
    };
    if let Some(value) = sampling(&temperature, 0.7) {
        chat.push(("temperature", value));
    }
    if let Some(value) = sampling(&top_p, 1.0) {
        chat.push(("top_p", value));
    }
    chat.push(("stream", Node::Bool(stream)));
    if !tools.is_empty() {
        chat.push((
            "tools",
            Node::Arr(tools.iter().map(|tool| tool.chat.clone()).collect()),
        ));
        chat.push(("tool_choice", tool_choice.clone()));
    }
    // SGLang (`serving_responses.py:585`, m214 F4): the effort as `reasoning_effort`, thinking on as
    // `chat_template_kwargs.enable_thinking` (CERE's chat subset has no `thinking` alias key, m212 D1).
    if let Some(effort) = effort {
        chat.push((
            "chat_template_kwargs",
            obj(vec![("enable_thinking", Node::Bool(true))]),
        ));
        chat.push(("reasoning_effort", Node::Str(effort)));
    }
    Ok(Converted {
        chat: obj(chat).to_bytes(),
        echo: Echo {
            model,
            instructions: instructions.map_or(Node::Null, Node::Str),
            max_output_tokens: max_output_tokens.map_or(Node::Null, n),
            temperature: temperature.unwrap_or(Node::Null),
            top_p: top_p.unwrap_or(Node::Null),
            tool_choice,
            tools: tools.into_iter().map(|tool| tool.echo).collect(),
            ignored,
        },
        stream,
    })
}

struct Tool {
    chat: Node,
    echo: Node,
}

/// Function tools `{type: "function", name, description?, parameters?, strict?}`; built-in tools are refused.
/// The chat validator checks the converted shape (names, parameters, `strict`).
fn tools(value: Option<&Node>) -> Result<Vec<Tool>, ApiError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let Node::Arr(items) = value else {
        return Err(ApiError::invalid_value("tools", "tools must be an array"));
    };
    let mut out = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let at = format!("tools[{index}]");
        let Node::Obj(members) = item else {
            return Err(ApiError::invalid_value(&at, "a tool must be an object"));
        };
        match field(item, "type").and_then(Node::as_str) {
            Some("function") => {}
            Some(_) => return Err(stateless(&format!("{at}.type"), "built-in tools")),
            None => return Err(ApiError::missing_parameter(format!("{at}.type"))),
        }
        known(
            members,
            &format!("{at}."),
            &["description", "name", "parameters", "strict", "type"],
        )?;
        let name = field(item, "name")
            .ok_or_else(|| ApiError::missing_parameter(format!("{at}.name")))?
            .clone();
        let optional = |key: &str| field(item, key).cloned();
        let mut function = vec![];
        if let Some(description) = optional("description") {
            function.push(("description", description));
        }
        function.push(("name", name.clone()));
        if let Some(parameters) = optional("parameters") {
            function.push(("parameters", parameters));
        }
        let strict = optional("strict").unwrap_or(Node::Bool(false));
        function.push(("strict", strict.clone()));
        out.push(Tool {
            chat: obj(vec![("type", s("function")), ("function", obj(function))]),
            echo: obj(vec![
                ("type", s("function")),
                ("name", name),
                ("description", optional("description").unwrap_or(Node::Null)),
                ("parameters", optional("parameters").unwrap_or(Node::Null)),
                ("strict", strict),
                ("tools", Node::Null),
            ]),
        });
    }
    Ok(out)
}

/// `tool_choice` as SGLang maps it: absent -> `auto`; `auto`/`none` pass; `required` and named need M2.8.
fn tool_choice(value: Option<&Node>, has_tools: bool) -> Result<Node, ApiError> {
    match value {
        None => Ok(s("auto")),
        Some(Node::Str(choice)) if choice == "auto" || choice == "none" => Ok(s(choice)),
        Some(Node::Str(choice)) if choice == "required" && !has_tools => Err(ApiError::invalid_value(
            "tool_choice",
            "tool_choice=\"required\" requires at least one tool with type=\"function\"; other built-in tool \
             types cannot be forced.",
        )),
        Some(Node::Str(choice)) if choice == "required" => Err(m28()),
        Some(choice @ Node::Obj(_))
            if choice.get("type").and_then(Node::as_str) == Some("function")
                && choice.get("name").and_then(Node::as_str).is_some() =>
        {
            Err(m28())
        }
        Some(_) => Err(ApiError::invalid_value(
            "tool_choice",
            "tool_choice must be \"auto\", \"none\", \"required\" or {\"type\": \"function\", \"name\": ...}",
        )),
    }
}

fn m28() -> ApiError {
    ApiError::unsupported_parameter(
        "tool_choice",
        "tool_choice \"required\" and named functions are not supported (planned for M2.8); use \"auto\" or \"none\"",
    )
}

/// `reasoning`: `effort` maps to `reasoning_effort` per m212 D13 (low, medium, xhigh), which needs the profile's
/// M2.12 `thinking_toggle`; summaries are not rendered. Returns the effort to forward.
fn reasoning(value: Option<&Node>, thinking: bool) -> Result<Option<String>, ApiError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let Node::Obj(members) = value else {
        return Err(ApiError::invalid_value(
            "reasoning",
            "reasoning must be an object",
        ));
    };
    known(members, "reasoning.", &["effort", "summary"])?;
    let effort = match field(value, "effort").map(Node::as_str) {
        None => None,
        Some(Some(effort @ ("low" | "medium" | "xhigh"))) if thinking => Some(effort.to_owned()),
        Some(Some("low" | "medium" | "xhigh")) => {
            return Err(ApiError::unsupported_parameter(
                "reasoning.effort",
                "reasoning.effort needs the deployment's reasoning-effort capability (M2.12)",
            ))
        }
        Some(_) => {
            return Err(ApiError::invalid_value(
                "reasoning.effort",
                "reasoning.effort must be \"low\", \"medium\" or \"xhigh\"",
            ))
        }
    };
    if field(value, "summary").is_some() {
        return Err(ApiError::unsupported_parameter(
            "reasoning.summary",
            "reasoning summaries are not supported",
        ));
    }
    Ok(effort)
}

/// `input` as chat messages: a string is one user message; items are messages (`developer` becomes `system`, as
/// SGLang normalizes it), `function_call` (an assistant message with one tool call) and `function_call_output`
/// (a tool message).
fn input(value: Option<&Node>) -> Result<Vec<Node>, ApiError> {
    let items = match value {
        None => return Err(ApiError::missing_parameter("input")),
        Some(Node::Str(text)) => {
            return Ok(vec![obj(vec![("role", s("user")), ("content", s(text))])])
        }
        Some(Node::Arr(items)) => items,
        Some(_) => {
            return Err(ApiError::invalid_value(
                "input",
                "input must be a string or an array of items",
            ))
        }
    };
    items
        .iter()
        .enumerate()
        .map(|(index, item)| input_item(&format!("input[{index}]"), item))
        .collect()
}

fn input_item(at: &str, item: &Node) -> Result<Node, ApiError> {
    let Node::Obj(members) = item else {
        return Err(ApiError::invalid_value(
            at,
            "an input item must be an object",
        ));
    };
    let string = |key: &str| match field(item, key) {
        None => Err(ApiError::missing_parameter(format!("{at}.{key}"))),
        Some(Node::Str(text)) => Ok(text.clone()),
        Some(_) => Err(ApiError::invalid_value(
            format!("{at}.{key}"),
            format!("{key} must be a string"),
        )),
    };
    match field(item, "type").and_then(Node::as_str) {
        None | Some("message") => {
            known(members, &format!("{at}."), &["content", "role", "type"])?;
            let role = match string("role")?.as_str() {
                "developer" | "system" => "system",
                "user" => "user",
                "assistant" => "assistant",
                other => {
                    let message = format!("role '{other}' is not supported");
                    return Err(ApiError::invalid_value(format!("{at}.role"), message));
                }
            };
            let content = content(&format!("{at}.content"), field(item, "content"))?;
            Ok(obj(vec![("role", s(role)), ("content", content)]))
        }
        Some("function_call") => {
            known(
                members,
                &format!("{at}."),
                &["arguments", "call_id", "id", "name", "status", "type"],
            )?;
            let call = obj(vec![
                ("id", s(&string("call_id")?)),
                ("type", s("function")),
                (
                    "function",
                    obj(vec![
                        ("name", s(&string("name")?)),
                        ("arguments", s(&string("arguments")?)),
                    ]),
                ),
            ]);
            Ok(obj(vec![
                ("role", s("assistant")),
                ("content", Node::Null),
                ("tool_calls", Node::Arr(vec![call])),
            ]))
        }
        Some("function_call_output") => {
            known(
                members,
                &format!("{at}."),
                &["call_id", "id", "output", "status", "type"],
            )?;
            Ok(obj(vec![
                ("role", s("tool")),
                ("content", s(&string("output")?)),
                ("tool_call_id", s(&string("call_id")?)),
            ]))
        }
        Some(other) => Err(ApiError::unsupported_parameter(
            format!("{at}.type"),
            format!("input items of type '{other}' are not supported"),
        )),
    }
}

/// Message content: a string, or text parts (`input_text`/`output_text`) joined into one string, as the chat
/// template's `render_content` concatenates text items. Image parts need the M2.13 capability.
fn content(at: &str, value: Option<&Node>) -> Result<Node, ApiError> {
    match value {
        None => Err(ApiError::missing_parameter(at)),
        Some(Node::Str(text)) => Ok(s(text)),
        Some(Node::Arr(parts)) => {
            let mut text = String::new();
            for (index, part) in parts.iter().enumerate() {
                let at = format!("{at}[{index}]");
                let Node::Obj(members) = part else {
                    return Err(ApiError::invalid_value(
                        &at,
                        "a content part must be an object",
                    ));
                };
                match field(part, "type").and_then(Node::as_str) {
                    Some("input_text" | "output_text") => {}
                    Some("input_image" | "input_file") => {
                        return Err(ApiError::unsupported_parameter(
                            format!("{at}.type"),
                            "image and file inputs are not supported (images need the M2.13 capability)",
                        ))
                    }
                    _ => {
                        return Err(ApiError::invalid_value(
                            format!("{at}.type"),
                            "type must be \"input_text\" or \"output_text\"",
                        ))
                    }
                }
                known(
                    members,
                    &format!("{at}."),
                    &["annotations", "text", "type", "logprobs"],
                )?;
                match field(part, "text") {
                    Some(Node::Str(part)) => text.push_str(part),
                    _ => return Err(ApiError::missing_parameter(format!("{at}.text"))),
                }
            }
            Ok(Node::Str(text))
        }
        Some(_) => Err(ApiError::invalid_value(
            at,
            "content must be a string or an array of text parts",
        )),
    }
}

/// SGLang's error envelope (`http_server.py`): `type` carries the CERE error code, `code` the HTTP status.
pub fn error_body(error: &ApiError) -> Node {
    envelope(error.status, error.code, &error.message)
}

/// [`error_body`] from its parts (an error the chat pipeline rendered: `kind` is its `error.code`).
pub fn envelope(status: u16, kind: &str, message: &str) -> Node {
    obj(vec![(
        "error",
        obj(vec![
            ("message", s(message)),
            ("type", s(kind)),
            ("param", Node::Null),
            ("code", n(u64::from(status))),
        ]),
    )])
}

/// `GET /v1/responses/{id}` and `POST /v1/responses/{id}/cancel`: no response is ever stored.
pub fn stateless_not_found(id: &str) -> ApiError {
    ApiError::new(
        404,
        INVALID_REQUEST,
        "not_found",
        format!("Response with id '{id}' not found: cered's /v1/responses is stateless and stores no responses"),
        None,
    )
}

/// `usage` of a chat completion (or the stream's usage chunk) as Responses usage.
fn usage(chat: Option<&Value>) -> Node {
    let get = |pointer: &str| {
        chat.and_then(|u| u.pointer(pointer))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let prompt = get("/prompt_tokens");
    let completion = get("/completion_tokens");
    let total = match get("/total_tokens") {
        0 => prompt + completion,
        total => total,
    };
    let reasoning = match get("/reasoning_tokens") {
        0 => get("/completion_tokens_details/reasoning_tokens"),
        tokens => tokens,
    };
    obj(vec![
        ("input_tokens", n(prompt)),
        (
            "input_tokens_details",
            obj(vec![
                (
                    "cached_tokens",
                    n(get("/prompt_tokens_details/cached_tokens")),
                ),
                ("cache_write_tokens", n(0)),
            ]),
        ),
        ("output_tokens", n(completion)),
        (
            "output_tokens_details",
            obj(vec![("reasoning_tokens", n(reasoning))]),
        ),
        ("total_tokens", n(total)),
    ])
}

fn output_text(text: &str) -> Node {
    obj(vec![
        ("annotations", Node::Arr(vec![])),
        ("text", s(text)),
        ("type", s("output_text")),
        ("logprobs", Node::Null),
    ])
}

fn message_item(id: &str, content: Vec<Node>, status: &str) -> Node {
    obj(vec![
        ("id", s(id)),
        ("content", Node::Arr(content)),
        ("role", s("assistant")),
        ("status", s(status)),
        ("type", s("message")),
    ])
}

fn reasoning_item(id: &str, text: Option<&str>, status: Node) -> Node {
    let content = text.map_or_else(Vec::new, |text| {
        vec![obj(vec![("text", s(text)), ("type", s("reasoning_text"))])]
    });
    obj(vec![
        ("id", s(id)),
        ("summary", Node::Arr(vec![])),
        ("type", s("reasoning")),
        ("content", Node::Arr(content)),
        ("encrypted_content", Node::Null),
        ("status", status),
    ])
}

fn call_item(arguments: &str, call_id: &str, name: &str, id: &str, status: &str) -> Node {
    obj(vec![
        ("arguments", s(arguments)),
        ("call_id", s(call_id)),
        ("name", s(name)),
        ("type", s("function_call")),
        ("id", s(id)),
        ("status", s(status)),
    ])
}

/// `from_request`'s `_is_text_only`: some output, all of it messages.
fn text_only(output: &[Node]) -> bool {
    !output.is_empty()
        && output
            .iter()
            .all(|item| item.get("type").and_then(Node::as_str) == Some("message"))
}

/// `(status, incomplete_details)` of a finish reason (`_status_from_finish_reason`).
fn status(finish: Option<&str>) -> (&'static str, Node) {
    match finish {
        Some("length") => ("incomplete", obj(vec![("reason", s("max_output_tokens"))])),
        _ => ("completed", Node::Null),
    }
}

/// The unary `response` object (SGLang's `ResponsesResponse.from_request(...).model_dump()` key order) of a chat
/// completion.
pub fn render_unary(echo: &Echo, ids: &mut Ids, chat: &Value) -> Node {
    let id = ids.id("resp_", 32);
    let choice = chat.pointer("/choices/0");
    let message = choice.and_then(|c| c.get("message"));
    let text = |key: &str| {
        message
            .and_then(|m| m.get(key))
            .and_then(Value::as_str)
            .filter(|t| !t.is_empty())
    };
    let mut output = Vec::new();
    if let Some(reasoning) = text("reasoning_content") {
        output.push(reasoning_item(
            &ids.id("rs_", 32),
            Some(reasoning),
            Node::Null,
        ));
    }
    if let Some(content) = text("content") {
        output.push(message_item(
            &ids.id("msg_", 32),
            vec![output_text(content)],
            "completed",
        ));
    }
    let calls = message
        .and_then(|m| m.get("tool_calls"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for call in calls {
        let get = |pointer: &str| call.pointer(pointer).and_then(Value::as_str);
        let call_id = match get("/id") {
            Some(id) => id.to_owned(),
            None => ids.id("call_", 24),
        };
        output.push(call_item(
            get("/function/arguments").unwrap_or_default(),
            &call_id,
            get("/function/name").unwrap_or_default(),
            &ids.id("fc_", 8),
            "completed",
        ));
    }
    let (status, incomplete) = status(
        choice
            .and_then(|c| c.get("finish_reason"))
            .and_then(Value::as_str),
    );
    let text_format = if text_only(&output) {
        obj(vec![("format", obj(vec![("type", s("text"))]))])
    } else {
        Node::Null
    };
    let created = chat.get("created").map_or(Node::Null, Node::from_value);
    obj(vec![
        ("id", s(&id)),
        ("object", s("response")),
        ("created_at", created),
        ("model", s(&echo.model)),
        ("output", Node::Arr(output)),
        ("status", s(status)),
        ("usage", usage(chat.get("usage"))),
        ("parallel_tool_calls", Node::Bool(true)),
        ("tool_choice", echo.tool_choice.clone()),
        ("tools", Node::Arr(echo.tools.clone())),
        ("error", Node::Null),
        ("incomplete_details", incomplete),
        ("instructions", echo.instructions.clone()),
        ("max_output_tokens", echo.max_output_tokens.clone()),
        ("previous_response_id", Node::Null),
        (
            "reasoning",
            obj(vec![("effort", Node::Null), ("summary", Node::Null)]),
        ),
        ("store", Node::Bool(false)),
        ("temperature", echo.temperature.clone()),
        ("text", text_format),
        ("top_p", echo.top_p.clone()),
        ("truncation", s("disabled")),
        ("user", Node::Null),
        ("metadata", obj(vec![])),
    ])
}

/// One SSE event: `event: <name>` and its `data` object.
#[derive(Clone, Debug)]
pub struct Event {
    pub name: &'static str,
    pub data: Node,
}

impl Event {
    /// The wire frame (`_send_event`: compact JSON in the event model's key order).
    pub fn frame(&self) -> String {
        let data = String::from_utf8(self.data.to_bytes()).expect("JSON is UTF-8");
        format!("event: {}\ndata: {data}\n\n", self.name)
    }
}

struct OpenItem {
    id: String,
    index: u64,
    text: String,
}

struct CallState {
    chat_index: u64,
    item: OpenItem,
    call_id: String,
    name: String,
    done: bool,
}

/// The SSE events of `responses_stream_generator_non_harmony` (non-harmony, no reasoning summary), fed with the
/// chat stream's chunks.
pub struct ResponsesStream {
    echo: Echo,
    ids: Ids,
    id: String,
    created: u64,
    sequence: u64,
    next_index: u64,
    reasoning: Option<OpenItem>,
    message: Option<OpenItem>,
    calls: Vec<CallState>,
    emitted: Vec<Node>,
    finish: Option<String>,
    usage: Option<Value>,
}

impl ResponsesStream {
    pub fn new(echo: Echo, mut ids: Ids, created: u64) -> Self {
        let id = ids.id("resp_", 32);
        Self {
            echo,
            ids,
            id,
            created,
            sequence: 0,
            next_index: 0,
            reasoning: None,
            message: None,
            calls: Vec::new(),
            emitted: Vec::new(),
            finish: None,
            usage: None,
        }
    }

    fn event(&mut self, name: &'static str, mut members: Vec<(&str, Node)>) -> Event {
        // The SDK event models' fields are alphabetical: `sequence_number` goes before the first later one.
        let at = members
            .iter()
            .position(|(k, _)| *k > "sequence_number")
            .unwrap_or(members.len());
        members.insert(at, ("sequence_number", n(self.sequence)));
        members.push(("type", s(name)));
        self.sequence += 1;
        Event {
            name,
            data: obj(members),
        }
    }

    /// The SDK `Response` (stream events re-validate the dump, so its field order) with `tools` emptied
    /// (`_sanitize_response_dict`).
    fn response(&self, status: &str, output: Vec<Node>, usage: Node) -> Node {
        let (_, incomplete) = match status {
            "incomplete" => self::status(Some("length")),
            _ => self::status(None),
        };
        let text = if text_only(&output) {
            obj(vec![
                ("format", obj(vec![("type", s("text"))])),
                ("verbosity", Node::Null),
            ])
        } else {
            Node::Null
        };
        obj(vec![
            ("id", s(&self.id)),
            (
                "created_at",
                Number::from_f64(self.created as f64).map_or(Node::Null, Node::Num),
            ),
            ("error", Node::Null),
            ("incomplete_details", incomplete),
            ("instructions", self.echo.instructions.clone()),
            ("metadata", obj(vec![])),
            ("model", s(&self.echo.model)),
            ("object", s("response")),
            ("output", Node::Arr(output)),
            ("parallel_tool_calls", Node::Bool(true)),
            ("temperature", self.echo.temperature.clone()),
            ("tool_choice", self.echo.tool_choice.clone()),
            ("tools", Node::Arr(vec![])),
            ("top_p", self.echo.top_p.clone()),
            ("background", Node::Null),
            ("conversation", Node::Null),
            ("max_output_tokens", self.echo.max_output_tokens.clone()),
            ("max_tool_calls", Node::Null),
            ("previous_response_id", Node::Null),
            ("prompt", Node::Null),
            ("prompt_cache_key", Node::Null),
            (
                "reasoning",
                obj(vec![
                    ("effort", Node::Null),
                    ("generate_summary", Node::Null),
                    ("summary", Node::Null),
                ]),
            ),
            ("safety_identifier", Node::Null),
            ("service_tier", Node::Null),
            ("status", s(status)),
            ("text", text),
            ("top_logprobs", Node::Null),
            ("truncation", s("disabled")),
            ("usage", usage),
            ("user", Node::Null),
            ("store", Node::Bool(false)),
        ])
    }

    /// `response.created` and `response.in_progress`.
    pub fn open(&mut self) -> Vec<Event> {
        let initial = self.response("in_progress", Vec::new(), Node::Null);
        vec![
            self.event("response.created", vec![("response", initial.clone())]),
            self.event("response.in_progress", vec![("response", initial)]),
        ]
    }

    fn open_item(&mut self, prefix: &str, digits: usize) -> OpenItem {
        let item = OpenItem {
            id: self.ids.id(prefix, digits),
            index: self.next_index,
            text: String::new(),
        };
        self.next_index += 1;
        item
    }

    fn close_reasoning(&mut self, events: &mut Vec<Event>) {
        let Some(item) = self.reasoning.take() else {
            return;
        };
        let done = self.event(
            "response.reasoning_text.done",
            vec![
                ("content_index", n(0)),
                ("item_id", s(&item.id)),
                ("output_index", n(item.index)),
                ("text", s(&item.text)),
            ],
        );
        events.push(done);
        let completed = reasoning_item(&item.id, Some(&item.text), s("completed"));
        let done = self.event(
            "response.output_item.done",
            vec![("item", completed.clone()), ("output_index", n(item.index))],
        );
        events.push(done);
        self.emitted.push(completed);
    }

    fn close_message(&mut self, events: &mut Vec<Event>) {
        let Some(item) = self.message.take() else {
            return;
        };
        let part = output_text(&item.text);
        let common = |extra: Vec<(&'static str, Node)>| {
            let mut members = vec![("content_index", n(0)), ("item_id", s(&item.id))];
            members.extend(extra);
            members
        };
        let text_done = self.event(
            "response.output_text.done",
            common(vec![
                ("logprobs", Node::Arr(vec![])),
                ("output_index", n(item.index)),
                ("text", s(&item.text)),
            ]),
        );
        let part_done = self.event(
            "response.content_part.done",
            common(vec![
                ("output_index", n(item.index)),
                ("part", part.clone()),
            ]),
        );
        let completed = message_item(&item.id, vec![part], "completed");
        let item_done = self.event(
            "response.output_item.done",
            vec![("item", completed.clone()), ("output_index", n(item.index))],
        );
        events.extend([text_done, part_done, item_done]);
        self.emitted.push(completed);
    }

    fn close_call(&mut self, position: usize, events: &mut Vec<Event>) {
        if self.calls[position].done {
            return;
        }
        self.calls[position].done = true;
        let call = &self.calls[position];
        let (id, index, arguments, call_id, name) = (
            call.item.id.clone(),
            call.item.index,
            call.item.text.clone(),
            call.call_id.clone(),
            call.name.clone(),
        );
        let done = self.event(
            "response.function_call_arguments.done",
            vec![
                ("arguments", s(&arguments)),
                ("item_id", s(&id)),
                ("name", s(&name)),
                ("output_index", n(index)),
            ],
        );
        let completed = call_item(&arguments, &call_id, &name, &id, "completed");
        let item_done = self.event(
            "response.output_item.done",
            vec![("item", completed.clone()), ("output_index", n(index))],
        );
        events.extend([done, item_done]);
        self.emitted.push(completed);
    }

    fn close_calls(&mut self, except: Option<usize>, events: &mut Vec<Event>) {
        for position in 0..self.calls.len() {
            if Some(position) != except {
                self.close_call(position, events);
            }
        }
    }

    fn reasoning_delta(&mut self, delta: &str, events: &mut Vec<Event>) {
        self.close_message(events);
        if self.reasoning.is_none() {
            let item = self.open_item("rs_", 32);
            let added = self.event(
                "response.output_item.added",
                vec![
                    ("item", reasoning_item(&item.id, None, s("in_progress"))),
                    ("output_index", n(item.index)),
                ],
            );
            events.push(added);
            self.reasoning = Some(item);
        }
        let item = self.reasoning.as_mut().expect("opened above");
        item.text.push_str(delta);
        let (id, index) = (item.id.clone(), item.index);
        let event = self.event(
            "response.reasoning_text.delta",
            vec![
                ("content_index", n(0)),
                ("delta", s(delta)),
                ("item_id", s(&id)),
                ("output_index", n(index)),
            ],
        );
        events.push(event);
    }

    fn text_delta(&mut self, delta: &str, events: &mut Vec<Event>) {
        // `_should_emit_normal_text_as_message`: whitespace between tool calls is not a message.
        let in_progress = self.calls.iter().any(|c| !c.done);
        if delta.is_empty() || (in_progress && delta.trim().is_empty()) {
            return;
        }
        self.close_reasoning(events);
        self.close_calls(None, events);
        if self.message.is_none() {
            let item = self.open_item("msg_", 32);
            let added = self.event(
                "response.output_item.added",
                vec![
                    ("item", message_item(&item.id, vec![], "in_progress")),
                    ("output_index", n(item.index)),
                ],
            );
            let part = self.event(
                "response.content_part.added",
                vec![
                    ("content_index", n(0)),
                    ("item_id", s(&item.id)),
                    ("output_index", n(item.index)),
                    ("part", output_text("")),
                ],
            );
            events.extend([added, part]);
            self.message = Some(item);
        }
        let item = self.message.as_mut().expect("opened above");
        item.text.push_str(delta);
        let (id, index) = (item.id.clone(), item.index);
        let event = self.event(
            "response.output_text.delta",
            vec![
                ("content_index", n(0)),
                ("delta", s(delta)),
                ("item_id", s(&id)),
                ("logprobs", Node::Arr(vec![])),
                ("output_index", n(index)),
            ],
        );
        events.push(event);
    }

    fn call_deltas(&mut self, calls: &[&Value], events: &mut Vec<Event>) {
        if calls.is_empty() {
            return;
        }
        self.close_reasoning(events);
        self.close_message(events);
        for call in calls {
            let chat_index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
            let get = |pointer: &str| call.pointer(pointer).and_then(Value::as_str);
            let position = match self
                .calls
                .iter()
                .position(|c| c.chat_index == chat_index && !c.done)
            {
                Some(position) => position,
                None => {
                    self.close_calls(None, events);
                    let item = self.open_item("fc_", 8);
                    let call_id = match get("/id") {
                        Some(id) => id.to_owned(),
                        None => self.ids.id("call_", 24),
                    };
                    let name = get("/function/name").unwrap_or_default().to_owned();
                    let added = self.event(
                        "response.output_item.added",
                        vec![
                            (
                                "item",
                                call_item("", &call_id, &name, &item.id, "in_progress"),
                            ),
                            ("output_index", n(item.index)),
                        ],
                    );
                    events.push(added);
                    self.calls.push(CallState {
                        chat_index,
                        item,
                        call_id,
                        name,
                        done: false,
                    });
                    self.calls.len() - 1
                }
            };
            let Some(arguments) = get("/function/arguments").filter(|a| !a.is_empty()) else {
                continue;
            };
            let state = &mut self.calls[position];
            state.item.text.push_str(arguments);
            let (id, index) = (state.item.id.clone(), state.item.index);
            let event = self.event(
                "response.function_call_arguments.delta",
                vec![
                    ("delta", s(arguments)),
                    ("item_id", s(&id)),
                    ("output_index", n(index)),
                ],
            );
            events.push(event);
        }
    }

    /// The events of one chat chunk (`chat.completion.chunk`; a `usage` chunk only records the usage).
    pub fn chunk(&mut self, chunk: &Value) -> Vec<Event> {
        let mut events = Vec::new();
        if let Some(usage) = chunk.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(usage.clone());
        }
        let Some(choice) = chunk.pointer("/choices/0") else {
            return events;
        };
        let delta = choice.get("delta");
        let text = |key: &str| {
            delta
                .and_then(|d| d.get(key))
                .and_then(Value::as_str)
                .unwrap_or_default()
        };
        let reasoning = text("reasoning_content");
        if !reasoning.is_empty() {
            self.reasoning_delta(reasoning, &mut events);
        }
        // Continuing arguments precede this delta's text; a newly opened call follows it.
        let calls: Vec<&Value> = delta
            .and_then(|d| d.get("tool_calls"))
            .and_then(Value::as_array)
            .map(|calls| calls.iter().collect())
            .unwrap_or_default();
        let continuing = |call: &&Value| {
            let index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
            self.calls.iter().any(|c| c.chat_index == index && !c.done)
        };
        let (continuing, opening): (Vec<&Value>, Vec<&Value>) =
            calls.into_iter().partition(continuing);
        self.call_deltas(&continuing, &mut events);
        self.text_delta(text("content"), &mut events);
        self.call_deltas(&opening, &mut events);
        if let Some(finish) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish = Some(finish.to_owned());
        }
        events
    }

    /// The closing events and `response.completed` (`usage`: the stream's final usage when the caller holds it,
    /// else the usage chunk's).
    pub fn complete(&mut self, usage: Option<&Value>) -> Vec<Event> {
        let mut events = Vec::new();
        self.close_reasoning(&mut events);
        self.close_message(&mut events);
        self.close_calls(None, &mut events);
        let usage = self::usage(usage.or(self.usage.as_ref()));
        let (status, _) = status(self.finish.as_deref());
        let response = self.response(status, self.emitted.clone(), usage);
        events.push(self.event("response.completed", vec![("response", response)]));
        events
    }

    /// `response.failed` (SGLang's stream exception path: no closing events, empty output).
    pub fn fail(&mut self) -> Vec<Event> {
        let response = self.response("failed", Vec::new(), Node::Null);
        vec![self.event("response.failed", vec![("response", response)])]
    }
}

/// The Responses surface of one request: renders the chat pipeline's results (M2.14 plumbing hook).
pub struct Responses {
    echo: Echo,
    /// The request's id seed (unary or stream: one render per request).
    seed: [u8; 16],
}

impl Responses {
    /// The renderer of a converted request, its ids seeded from the kernel CSPRNG.
    pub fn new(echo: Echo) -> io::Result<Self> {
        Ok(Self {
            echo,
            seed: crate::digest::random_bytes::<16>()?,
        })
    }
}

fn frames(events: Vec<Event>) -> Vec<u8> {
    events.iter().flat_map(|e| e.frame().into_bytes()).collect()
}

impl SurfaceRender for Responses {
    fn name(&self) -> &'static str {
        "responses"
    }

    fn ignored_params(&self) -> &[String] {
        &self.echo.ignored
    }

    fn render_unary(&self, chat: &Value, _rid: &str) -> Vec<u8> {
        render_unary(&self.echo, &mut Ids::seeded(self.seed), chat).to_bytes()
    }

    fn error_body(&self, e: &ErrorView) -> Vec<u8> {
        compat_error_body(e)
    }

    fn stream(&self) -> Box<dyn SurfaceStream> {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        Box::new(ResponsesStream::new(
            self.echo.clone(),
            Ids::seeded(self.seed),
            created,
        ))
    }
}

impl SurfaceStream for ResponsesStream {
    fn open(&mut self, _prompt_tokens: Option<u64>) -> Vec<u8> {
        frames(ResponsesStream::open(self))
    }

    fn chunk(&mut self, chunk: &Value) -> Vec<u8> {
        frames(ResponsesStream::chunk(self, chunk))
    }

    /// `finish`: the held finish chunk (or its choice), whose delta and finish reason are rendered first.
    fn complete(&mut self, finish: &Value, usage: Option<&Value>) -> Vec<u8> {
        let mut events = if finish.get("choices").is_some() {
            ResponsesStream::chunk(self, finish)
        } else {
            ResponsesStream::chunk(self, &serde_json::json!({ "choices": [finish] }))
        };
        events.extend(ResponsesStream::complete(self, usage));
        frames(events)
    }

    fn fail(&mut self, _e: &ErrorView) -> Vec<u8> {
        frames(ResponsesStream::fail(self))
    }
}

/// The surface's entry points for `server::serve_compat` (registered by [`routes`]).
pub static RESPONSES: CompatApi = CompatApi {
    kind: CompatKind::Responses,
    error_body: compat_error_body,
    convert: compat_convert,
};

fn compat_error_body(e: &ErrorView) -> Vec<u8> {
    envelope(e.status, e.code, e.message).to_bytes()
}

fn owned(e: ApiError) -> OwnedError {
    OwnedError {
        status: e.status,
        kind: e.error_type.to_owned(),
        code: e.code.to_owned(),
        message: e.message,
        param: e.param,
    }
}

/// [`convert`] for the deployment the request's `model` selected (native: an absent temperature stays absent).
#[allow(clippy::type_complexity)]
fn compat_convert(
    body: &[u8],
    ctx: &ConvertCtx,
) -> Result<(Vec<u8>, Arc<dyn SurfaceRender>), OwnedError> {
    let converted = convert(body, ctx.policy).map_err(owned)?;
    let render = Responses::new(converted.echo).map_err(|e| {
        owned(ApiError::new(
            500,
            SERVER_ERROR,
            "internal_error",
            format!("no response id seed: {e}"),
            None,
        ))
    })?;
    Ok((converted.chat, Arc::new(render)))
}

/// `POST /v1/responses`, `GET /v1/responses/{id}` and `POST /v1/responses/{id}/cancel` (installed by
/// `server::router` iff some deployment sets `api_compat_responses`).
pub fn routes(router: Router<Arc<App>>) -> Router<Arc<App>> {
    router
        .route("/v1/responses", post(create))
        .route("/v1/responses/{id}", get(not_stored))
        .route("/v1/responses/{id}/cancel", post(not_stored))
}

async fn create(State(app): State<Arc<App>>, headers: HeaderMap, body: Body) -> Response {
    crate::server::serve_compat(&RESPONSES, app, headers, body).await
}

/// Retrieve and cancel: bearer-authenticated, then always 404 `not_found` (no response is stored), whatever the
/// serving state or the deployments' flags.
async fn not_stored(
    State(app): State<Arc<App>>,
    UrlPath(id): UrlPath<String>,
    headers: HeaderMap,
) -> Response {
    if let Err(refused) = crate::server::compat_authenticate(&RESPONSES, &app, &headers).await {
        return refused;
    }
    let body = error_body(&stateless_not_found(&id)).to_bytes();
    (
        StatusCode::NOT_FOUND,
        [(header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response()
}
