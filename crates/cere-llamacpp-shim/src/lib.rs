//! SGLang-surface adapter in front of a llama.cpp `llama-server` (Phase 1 of the AMD AMD track,
//! amd-amd design note).
//!
//! `cere-worker` identifies and drives its loopback backend through the SGLang HTTP subset documented in
//! `crates/cere-mock-sglang/src/lib.rs`. `llama-server` speaks OpenAI chat but not that subset, so this shim sits
//! between them on loopback:
//!
//! - `GET /health` → 200 when the upstream `/health` answers 200, else 503.
//! - `GET /v1/models` → the configured `served_model_name`, `max_model_len` = `context_length`.
//! - `GET /get_server_info` → the configured info file (what the operator pins: the real llama.cpp build in
//!   `version`, the GGUF in `model_path`, ...) plus this launch's `startup_time` and `random_seed`. The shim never
//!   reports SGLang values: a deployment that fronts llama.cpp pins llama.cpp values (AGENTS.md 6).
//! - `GET /get_load` → `[{"dp_rank":0,"num_reqs":r,"num_waiting_reqs":w}]` over the shim's live chat requests,
//!   `r` capped at `max_running_requests` (llama-server's `--parallel`).
//! - `POST /abort_request {"rid": r, "abort_all": b}` → 200. Ends every live request whose rid starts with `r`
//!   (SGLang's prefix predicate; empty `r` or `abort_all` ends all) by dropping its upstream connection, which
//!   llama-server treats as a cancel. The client sees `finish_reason: "abort"`.
//! - `POST /v1/chat/completions` → unary or SSE, forwarded after [`upstream_body`]; responses are reshaped by
//!   [`reshape_response`] / [`reshape_chunk`].
//!
//! Prefix-cache isolation: llama.cpp has no `cache_salt`. A request that carries one is sent with
//! `cache_prompt: false` (no reuse, so no cross-scope reuse) unless the shim runs with `trust_single_scope`, for a
//! lab with one scope. `cached_tokens` is reported from llama-server's `timings.cache_n`.

use serde_json::{json, Map, Value};

/// SGLang-only request fields llama-server does not understand; dropped before forwarding.
pub const SGLANG_ONLY_FIELDS: &[&str] = &[
    "rid",
    "cache_salt",
    "max_completion_tokens",
    "separate_reasoning",
    "return_hidden_states",
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
];

/// The upstream request body: `rid`/SGLang-only fields removed, `max_completion_tokens` mapped to `max_tokens`
/// (it wins, as in SGLang), usage always requested on streams (the shim re-emits it only if the client asked),
/// and `cache_prompt` disabled for salted requests unless `trust_single_scope`.
pub fn upstream_body(mut body: Map<String, Value>, trust_single_scope: bool) -> Map<String, Value> {
    if let Some(n) = body
        .get("max_completion_tokens")
        .cloned()
        .filter(|v| !v.is_null())
    {
        body.insert("max_tokens".into(), n);
    }
    let salted = body.get("cache_salt").is_some_and(|v| !v.is_null());
    for key in SGLANG_ONLY_FIELDS {
        body.remove(*key);
    }
    if salted && !trust_single_scope {
        body.insert("cache_prompt".into(), Value::Bool(false));
    }
    if body.get("stream").and_then(Value::as_bool) == Some(true) {
        body.insert("stream_options".into(), json!({"include_usage": true}));
    }
    body
}

/// Request fields llama-server would silently ignore (it skips unknown keys), so serving them would return
/// unconstrained output for a constrained request. They are refused instead (AGENTS.md 6): SGLang's top-level
/// `regex` and `ebnf`. `response_format` (json_object / json_schema) is forwarded: llama-server enforces it itself.
pub const REFUSED_FIELDS: &[&str] = &["regex", "ebnf"];

/// The first non-null field of `body` the shim refuses, if any.
pub fn refused_field(body: &Map<String, Value>) -> Option<&'static str> {
    REFUSED_FIELDS
        .iter()
        .copied()
        .find(|k| body.get(*k).is_some_and(|v| !v.is_null()))
}

/// How the shim serves a request's `tools` (CERE's qwen3_coder semantics, as the native path: SGLang renders the
/// tools and runs its own parser; llama-server's own parser is stricter and is not used).
#[derive(Debug, Clone, PartialEq)]
pub enum ToolPlan {
    /// No tools, or `tool_choice: "none"` (SGLang renders no tools and parses nothing): forwarded without tools.
    None,
    /// `tools` with `tool_choice` absent or `"auto"`: the tools go to the template as `chat_template_kwargs.tools`
    /// (llama-server renders them byte-identically there, but then leaves the reply unparsed) and the reply is
    /// parsed by CERE's qwen3_coder detector. Holds the client's tools, for the conversion's schema lookup.
    Parse(Vec<Value>),
}

/// The tool plan of `body`, or the refusal: constrained tool choices (`required`, named: the profile declares no
/// `tool_calls_constrained`) and `parallel_tool_calls: false` are refused rather than served differently from
/// SGLang.
pub fn tool_plan(body: &Map<String, Value>) -> Result<ToolPlan, String> {
    let tools = match body.get("tools") {
        None | Some(Value::Null) => return Ok(ToolPlan::None),
        Some(Value::Array(t)) if t.is_empty() => return Ok(ToolPlan::None),
        Some(Value::Array(t)) => t.clone(),
        Some(_) => return Err("tools must be an array".into()),
    };
    match body.get("tool_choice") {
        None | Some(Value::Null) => {}
        Some(Value::String(c)) if c == "auto" => {}
        Some(Value::String(c)) if c == "none" => return Ok(ToolPlan::None),
        Some(_) => {
            return Err("tool_choice other than auto/none is not supported by this backend".into())
        }
    }
    if body.get("parallel_tool_calls") == Some(&Value::Bool(false)) {
        return Err("parallel_tool_calls: false is not supported by this backend".into());
    }
    Ok(ToolPlan::Parse(tools))
}

/// `body` for llama-server under `plan`: `tools`, `tool_choice` and `parallel_tool_calls` removed; with
/// [`ToolPlan::Parse`], the tools placed in `chat_template_kwargs.tools` (other kwargs kept).
pub fn apply_tool_plan(mut body: Map<String, Value>, plan: &ToolPlan) -> Map<String, Value> {
    for key in ["tools", "tool_choice", "parallel_tool_calls"] {
        body.remove(key);
    }
    if let ToolPlan::Parse(tools) = plan {
        let kwargs = body
            .entry("chat_template_kwargs")
            .or_insert_with(|| json!({}));
        if !kwargs.is_object() {
            *kwargs = json!({});
        }
        kwargs["tools"] = Value::Array(tools.clone());
    }
    body
}

/// One parsed call, in SGLang's unary shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub index: usize,
    pub name: String,
    pub arguments: String,
}

/// The unary response with the qwen3_coder parse applied to `choices[0]`: `content` = the normal text (null when
/// empty), `tool_calls` = the calls, and a `stop` finish becomes `tool_calls` when a call went out
/// (`serving_chat.py:2035-2038`; the native path's `whole`).
pub fn with_tool_calls(mut resp: Value, normal: &str, calls: &[ToolCall]) -> Value {
    if let Some(choice) = resp.pointer_mut("/choices/0") {
        let message = &mut choice["message"];
        message["content"] = if normal.is_empty() {
            Value::Null
        } else {
            json!(normal)
        };
        message["tool_calls"] = if calls.is_empty() {
            Value::Null
        } else {
            Value::Array(
                calls
                    .iter()
                    .map(|c| {
                        json!({"id": c.id, "index": c.index, "type": "function",
                               "function": {"name": c.name, "arguments": c.arguments}})
                    })
                    .collect(),
            )
        };
        if !calls.is_empty() && choice["finish_reason"] == "stop" {
            choice["finish_reason"] = json!("tool_calls");
        }
    }
    resp
}

/// Streaming with tools: takes `choices[0].delta.content` and `choices[0].finish_reason` out of a relayed chunk
/// (content to the tool parser, the finish to be re-emitted after the call items), leaving nulls.
pub fn take_content_and_finish(chunk: &mut Value) -> (String, Option<Value>) {
    let Some(choice) = chunk.pointer_mut("/choices/0") else {
        return (String::new(), None);
    };
    let content = match choice["delta"].get_mut("content").map(Value::take) {
        Some(Value::String(c)) => c,
        _ => String::new(),
    };
    let finish = choice
        .get_mut("finish_reason")
        .map(Value::take)
        .filter(|f| !f.is_null());
    (content, finish)
}

/// Whether a relayed chunk still carries something for the client (role, content or reasoning).
pub fn chunk_has_payload(chunk: &Value) -> bool {
    let d = &chunk["choices"][0]["delta"];
    ["role", "content", "reasoning_content"]
        .iter()
        .any(|k| d.get(*k).is_some_and(|v| !v.is_null() && v != ""))
}

/// A chunk like `skeleton` (its id, object, created, model) whose only choice carries `delta` and `finish`.
pub fn chunk_like(skeleton: &Value, delta: Value, finish: Value) -> Value {
    json!({"id": skeleton["id"], "object": skeleton["object"], "created": skeleton["created"],
           "model": skeleton["model"],
           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
}

/// One streamed tool-call item in SGLang's shape: `id` and `name` on a call's first item, null after.
pub fn stream_call(
    id: Option<String>,
    index: i64,
    name: Option<String>,
    arguments: String,
) -> Value {
    json!({"role": null, "content": null, "reasoning_content": null,
           "tool_calls": [{"id": id, "index": index, "type": "function",
                           "function": {"name": name, "arguments": arguments}}]})
}

/// Whether the client asked for a usage chunk on its stream.
pub fn wants_usage(body: &Map<String, Value>) -> bool {
    body.get("stream_options")
        .and_then(|o| o.get("include_usage"))
        .and_then(Value::as_bool)
        == Some(true)
}

/// `prompt_tokens_details` from llama-server's `timings.cache_n`: `{"cached_tokens": n}` when the cache report is
/// on and `n > 0`, else null (SGLang reports a miss as null).
fn cache_details(timings: Option<&Value>, cache_report: bool) -> Value {
    match timings
        .and_then(|t| t.get("cache_n"))
        .and_then(Value::as_u64)
    {
        Some(n) if cache_report && n > 0 => json!({"cached_tokens": n}),
        _ => Value::Null,
    }
}

fn fix_usage(usage: &mut Value, timings: Option<&Value>, cache_report: bool) {
    if let Some(u) = usage.as_object_mut() {
        u.insert(
            "prompt_tokens_details".into(),
            cache_details(timings, cache_report),
        );
    }
}

/// A unary upstream response in SGLang shape: `id` = rid, `model` = the served model name (llama-server reports
/// its own `--alias`, which need not match), `timings` dropped, `prompt_tokens_details` set.
pub fn reshape_response(mut resp: Value, rid: &str, model: &str, cache_report: bool) -> Value {
    let timings = resp.as_object_mut().and_then(|o| o.remove("timings"));
    if let Some(o) = resp.as_object_mut() {
        o.insert("id".into(), Value::String(rid.to_owned()));
        o.insert("model".into(), Value::String(model.to_owned()));
        if let Some(usage) = o.get_mut("usage") {
            fix_usage(usage, timings.as_ref(), cache_report);
        }
    }
    resp
}

/// One upstream SSE chunk in SGLang shape. Returns the chunk to relay (if any) and the usage it carried (if any).
/// SGLang sends usage only in a separate `choices: []` chunk after the finish chunk, which is what `cered`
/// accepts, so usage riding on a content/finish chunk is split off here and re-emitted by the caller at the end.
pub fn reshape_chunk(
    mut chunk: Value,
    rid: &str,
    model: &str,
    cache_report: bool,
) -> (Option<Value>, Option<Value>) {
    let Some(o) = chunk.as_object_mut() else {
        return (None, None);
    };
    let timings = o.remove("timings");
    o.insert("id".into(), Value::String(rid.to_owned()));
    o.insert("model".into(), Value::String(model.to_owned()));
    let usage = o.remove("usage").filter(|u| !u.is_null()).map(|mut u| {
        fix_usage(&mut u, timings.as_ref(), cache_report);
        u
    });
    let no_choices = o
        .get("choices")
        .and_then(Value::as_array)
        .is_none_or(Vec::is_empty);
    if no_choices {
        return (None, usage); // a bare usage chunk: re-emitted in canonical form at the end
    }
    (Some(chunk), usage)
}

/// The canonical trailing usage chunk.
pub fn usage_chunk(rid: &str, model: &str, usage: Value) -> Value {
    json!({"id": rid, "object": "chat.completion.chunk", "model": model, "choices": [], "usage": usage})
}

/// The finish chunk sent when a stream is aborted.
pub fn abort_chunk(rid: &str, model: &str) -> Value {
    json!({"id": rid, "object": "chat.completion.chunk", "model": model,
           "choices": [{"index": 0, "delta": {"role": null, "content": null, "reasoning_content": null},
                        "finish_reason": "abort", "matched_stop": null}]})
}

/// The body returned for an aborted unary request.
pub fn abort_response(rid: &str, model: &str, prompt_tokens: u64) -> Value {
    json!({"id": rid, "object": "chat.completion", "model": model,
           "choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                        "finish_reason": "abort", "matched_stop": null}],
           "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens,
                     "prompt_tokens_details": null}})
}

/// Whether a chunk generated output (content, reasoning or a tool call): counted for an aborted stream's usage.
pub fn generated(chunk: &Value) -> bool {
    let delta = chunk.pointer("/choices/0/delta");
    let text = |k: &str| {
        delta
            .and_then(|d| d.get(k))
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty())
    };
    text("content")
        || text("reasoning_content")
        || delta
            .and_then(|d| d.get("tool_calls"))
            .and_then(Value::as_array)
            .is_some_and(|c| !c.is_empty())
}

/// SGLang's abort predicate: `rid` starts with `prefix`; an empty prefix or `abort_all` matches everything.
pub fn abort_matches(rid: &str, prefix: &str, abort_all: bool) -> bool {
    abort_all || prefix.is_empty() || rid.starts_with(prefix)
}

/// `/get_load` over `live` requests with `max_running` slots.
pub fn load(live: usize, max_running: usize) -> Value {
    let running = live.min(max_running);
    json!([{"dp_rank": 0, "num_reqs": running, "num_waiting_reqs": live - running}])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    #[test]
    fn upstream_body_drops_sglang_fields_and_maps_budget() {
        let b = upstream_body(
            obj(json!({"rid": "cere-1", "model": "m", "max_tokens": 9,
            "max_completion_tokens": 4, "messages": [], "separate_reasoning": true})),
            false,
        );
        assert_eq!(b.get("max_tokens"), Some(&json!(4)));
        assert!(
            b.get("rid").is_none()
                && b.get("max_completion_tokens").is_none()
                && b.get("separate_reasoning").is_none()
        );
        assert!(
            b.get("cache_prompt").is_none(),
            "unsalted requests keep llama's default cache use"
        );
    }

    #[test]
    fn salted_request_disables_cache_unless_single_scope() {
        let body = obj(json!({"cache_salt": "scope-a", "messages": []}));
        assert_eq!(
            upstream_body(body.clone(), false).get("cache_prompt"),
            Some(&json!(false))
        );
        assert!(upstream_body(body, true).get("cache_prompt").is_none());
    }

    #[test]
    fn null_budget_does_not_override() {
        let b = upstream_body(
            obj(json!({"max_tokens": 9, "max_completion_tokens": null})),
            false,
        );
        assert_eq!(b.get("max_tokens"), Some(&json!(9)));
    }

    #[test]
    fn stream_always_requests_usage_upstream() {
        let client = obj(json!({"stream": true}));
        assert!(!wants_usage(&client));
        assert!(wants_usage(&upstream_body(client, false)));
    }

    #[test]
    fn unenforceable_constraints_are_refused() {
        let b = |v: Value| v.as_object().unwrap().clone();
        assert_eq!(refused_field(&b(json!({"regex": "[0-9]+"}))), Some("regex"));
        assert_eq!(
            refused_field(&b(json!({"ebnf": "root ::= \"a\""}))),
            Some("ebnf")
        );
        assert_eq!(
            refused_field(&b(json!({"regex": null, "messages": []}))),
            None
        );
        let rf = json!({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}});
        assert_eq!(refused_field(&b(rf.clone())), None);
        // forwarded unchanged
        assert_eq!(
            upstream_body(b(rf.clone()), false)["response_format"],
            rf["response_format"]
        );
    }

    #[test]
    fn tool_plans() {
        let b = |v: Value| v.as_object().unwrap().clone();
        let tools = json!([{"type": "function", "function": {"name": "f"}}]);
        assert_eq!(tool_plan(&b(json!({}))), Ok(ToolPlan::None));
        assert_eq!(tool_plan(&b(json!({"tools": []}))), Ok(ToolPlan::None));
        assert_eq!(
            tool_plan(&b(json!({"tools": tools, "tool_choice": "none"}))),
            Ok(ToolPlan::None)
        );
        let plan = tool_plan(&b(json!({"tools": tools, "tool_choice": "auto"}))).unwrap();
        assert_eq!(plan, ToolPlan::Parse(tools.as_array().unwrap().clone()));
        assert!(tool_plan(&b(json!({"tools": tools, "tool_choice": "required"}))).is_err());
        assert!(tool_plan(&b(
            json!({"tools": tools, "tool_choice": {"type": "function"}})
        ))
        .is_err());
        assert!(tool_plan(&b(json!({"tools": tools, "parallel_tool_calls": false}))).is_err());
        assert!(matches!(
            tool_plan(&b(json!({"tools": tools, "stream": true}))),
            Ok(ToolPlan::Parse(_))
        ));
        assert!(tool_plan(&b(json!({"tools": "x"}))).is_err());
        let up = apply_tool_plan(
            b(
                json!({"tools": tools, "tool_choice": "auto", "chat_template_kwargs": {"enable_thinking": false}}),
            ),
            &plan,
        );
        assert!(up.get("tools").is_none() && up.get("tool_choice").is_none());
        assert_eq!(
            up["chat_template_kwargs"],
            json!({"enable_thinking": false, "tools": tools})
        );
        let none = apply_tool_plan(
            b(json!({"tools": tools, "tool_choice": "none"})),
            &ToolPlan::None,
        );
        assert!(none.get("tools").is_none() && none.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn tool_calls_reshape_the_unary_reply() {
        let resp = json!({"choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "x <tool_call>...", "reasoning_content": "r"}}]});
        let call = ToolCall {
            id: "call_1".into(),
            index: 0,
            name: "f".into(),
            arguments: "{\"a\": 1}".into(),
        };
        let r = with_tool_calls(resp.clone(), "x ", std::slice::from_ref(&call));
        assert_eq!(r["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(r["choices"][0]["message"]["content"], "x ");
        assert_eq!(r["choices"][0]["message"]["reasoning_content"], "r");
        assert_eq!(
            r["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"a\": 1}"
        );
        let r = with_tool_calls(resp.clone(), "", std::slice::from_ref(&call));
        assert_eq!(r["choices"][0]["message"]["content"], Value::Null);
        let r = with_tool_calls(resp, "plain", &[]);
        assert_eq!(r["choices"][0]["finish_reason"], "stop");
        assert_eq!(r["choices"][0]["message"]["tool_calls"], Value::Null);
    }

    #[test]
    fn stream_tool_chunk_helpers() {
        let mut c = json!({"id": "r", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "choices": [{"index": 0, "delta": {"content": "hi <tool"}, "finish_reason": null}]});
        assert_eq!(
            take_content_and_finish(&mut c),
            ("hi <tool".to_owned(), None)
        );
        assert_eq!(c["choices"][0]["delta"]["content"], Value::Null);
        assert!(!chunk_has_payload(&c));
        let mut f = json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]});
        assert_eq!(
            take_content_and_finish(&mut f),
            (String::new(), Some(json!("stop")))
        );
        let r = json!({"choices": [{"delta": {"reasoning_content": "think"}}]});
        assert!(chunk_has_payload(&r));
        let item = chunk_like(
            &c,
            stream_call(Some("call_x".into()), 0, Some("f".into()), String::new()),
            Value::Null,
        );
        assert_eq!(item["id"], "r");
        assert_eq!(
            item["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "f"
        );
        let next = stream_call(None, 0, None, "{".into());
        assert_eq!(next["tool_calls"][0]["id"], Value::Null);
        assert_eq!(next["tool_calls"][0]["function"]["arguments"], "{");
    }

    #[test]
    fn unary_response_reshaped() {
        let r = reshape_response(
            json!({"id": "chatcmpl-x", "model": "upstream-alias", "choices": [], "timings": {"cache_n": 12},
            "usage": {"prompt_tokens": 20, "completion_tokens": 3}}),
            "cere-7",
            "Served-Name",
            true,
        );
        assert_eq!(r["id"], "cere-7");
        assert_eq!(r["model"], "Served-Name");
        assert!(r.get("timings").is_none());
        assert_eq!(
            r["usage"]["prompt_tokens_details"],
            json!({"cached_tokens": 12})
        );
        let miss = reshape_response(
            json!({"usage": {}, "timings": {"cache_n": 0}}),
            "r",
            "m",
            true,
        );
        assert_eq!(miss["usage"]["prompt_tokens_details"], Value::Null);
        let off = reshape_response(
            json!({"usage": {}, "timings": {"cache_n": 5}}),
            "r",
            "m",
            false,
        );
        assert_eq!(off["usage"]["prompt_tokens_details"], Value::Null);
    }

    #[test]
    fn usage_on_finish_chunk_is_split_off() {
        let (relay, usage) = reshape_chunk(
            json!({"id": "x", "model": "upstream-alias", "choices": [{"index": 0, "delta": {},
            "finish_reason": "stop"}], "usage": {"completion_tokens": 3}, "timings": {"cache_n": 2}}),
            "cere-1",
            "Served-Name",
            true,
        );
        let relay = relay.unwrap();
        assert_eq!(relay["id"], "cere-1");
        assert_eq!(relay["model"], "Served-Name");
        assert!(relay.get("usage").is_none() && relay.get("timings").is_none());
        assert_eq!(usage.unwrap()["prompt_tokens_details"]["cached_tokens"], 2);
    }

    #[test]
    fn bare_usage_chunk_is_held_for_the_end() {
        let (relay, usage) = reshape_chunk(
            json!({"choices": [], "usage": {"completion_tokens": 1}}),
            "r",
            "m",
            false,
        );
        assert!(relay.is_none());
        assert_eq!(usage.unwrap()["completion_tokens"], 1);
        assert_eq!(usage_chunk("r", "m", json!({}))["choices"], json!([]));
    }

    #[test]
    fn abort_prefix_predicate() {
        assert!(abort_matches("cere-10", "cere-1", false));
        assert!(!abort_matches("cere-2", "cere-1", false));
        assert!(abort_matches("x", "", false) && abort_matches("x", "y", true));
    }

    #[test]
    fn load_caps_running_at_slots() {
        assert_eq!(
            load(6, 4),
            json!([{"dp_rank": 0, "num_reqs": 4, "num_waiting_reqs": 2}])
        );
        assert_eq!(
            load(0, 4),
            json!([{"dp_rank": 0, "num_reqs": 0, "num_waiting_reqs": 0}])
        );
    }

    #[test]
    fn generated_detects_output() {
        assert!(generated(
            &json!({"choices": [{"delta": {"content": "a"}}]})
        ));
        assert!(generated(
            &json!({"choices": [{"delta": {"reasoning_content": "t"}}]})
        ));
        assert!(!generated(
            &json!({"choices": [{"delta": {"role": "assistant", "content": ""}}]})
        ));
    }

    #[test]
    fn abort_shapes() {
        assert_eq!(
            abort_chunk("r", "m")["choices"][0]["finish_reason"],
            "abort"
        );
        assert_eq!(abort_response("r", "m", 3)["usage"]["total_tokens"], 3);
    }
}
