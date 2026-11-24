//! DocInput D9 at the request layer (doc-input design note DI-06): a request without documents takes the
//! unchanged path. For a fixture set of document-free chat and `/v1/messages` bodies on the shipped `-v29` profile
//! (`configs/runtime/native-g0-spec.toml`), the Anthropic conversion's chat bytes, the validated request, its
//! digest input (`normalized()`) and the tokenizer sidecar's `encode_chat` request line equal
//! `tests/fixtures/native/document_identity.json`, recorded by this test on a `git archive` of origin/main without
//! DocInput (last `fc48e036`; a main that changes the validated request is re-recorded the same way).
//! The sidecar (`scripts/native/tokenizer_sidecar.py`), the template and the tokenizer are unchanged, so an equal
//! request line renders the same prompt bytes and token ids.
//!
//! Record: `CERE_DOC_IDENTITY_RECORD=<path> cargo test -p cere-runtime --test document_identity` writes the
//! outputs to `<path>` instead of comparing.
use cere_runtime::anthropic;
use cere_runtime::config::RuntimeConfig;
use cere_runtime::validate::{ChatPolicy, ChatRequest};
use serde_json::{json, Map, Value};

fn policy() -> ChatPolicy {
    let config = RuntimeConfig::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/runtime/native-g0-spec.toml"),
    )
    .unwrap();
    ChatPolicy::from_config(&config, 0).unwrap()
}

const TOOLS: &str = r#"[{"type":"function","function":{"name":"Read","description":"Read a file","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}}]"#;

fn chat_bodies() -> Vec<(&'static str, Value)> {
    let tools: Value = serde_json::from_str(TOOLS).unwrap();
    vec![
        (
            "chat_string",
            json!({"model": "Qwen3.8-Flash-Next", "messages": [{"role": "user", "content": "hi"}]}),
        ),
        (
            "chat_turns",
            json!({"model": "Qwen3.8-Flash-Next", "temperature": 0, "max_tokens": 64, "stop": ["END"],
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "a"},
                         {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]}),
        ),
        (
            "chat_text_parts",
            json!({"model": "Qwen3.8-Flash-Next", "temperature": 0, "messages": [
            {"role": "user", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}]}),
        ),
        (
            "chat_tools",
            json!({"model": "Qwen3.8-Flash-Next", "temperature": 0, "tools": tools, "tool_choice": "auto",
            "messages": [
                {"role": "user", "content": "read a.txt"},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "Read", "arguments": "{\"file_path\":\"a.txt\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "PDF file read: a.txt"},
                {"role": "user", "content": "thanks"}]}),
        ),
        (
            "chat_unicode_stream",
            json!({"model": "Qwen3.8-Flash-Next", "stream": true,
            "stream_options": {"include_usage": true}, "stop": ["é", "x y"],
            "messages": [{"role": "user", "content": "é😀\n\"q\" <document> </page>"}]}),
        ),
        (
            "chat_sampled",
            json!({"model": "Qwen3.8-Flash-Next", "temperature": 0.7, "top_p": 0.9,
            "max_completion_tokens": 32, "messages": [{"role": "user", "content": "x"}]}),
        ),
        (
            "chat_thinking",
            json!({"model": "Qwen3.8-Flash-Next", "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": true}, "messages": [{"role": "user", "content": "x"}]}),
        ),
        (
            "chat_json_object",
            json!({"model": "Qwen3.8-Flash-Next", "temperature": 0,
            "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": "x"}]}),
        ),
    ]
}

fn anthropic_bodies() -> Vec<(&'static str, Value)> {
    let tools = json!([{"name": "Read", "description": "Read a file",
        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}]);
    let read = |content: Value| {
        json!([
            {"role": "user", "content": "read a.txt"},
            {"role": "assistant", "content": [{"type": "text", "text": "Reading."},
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a.txt"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": content}]}
        ])
    };
    vec![
        (
            "anthropic_string",
            json!({"model": "Qwen3.8-Flash-Next", "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]}),
        ),
        (
            "anthropic_text_blocks",
            json!({"model": "Qwen3.8-Flash-Next", "max_tokens": 64, "temperature": 0,
            "system": "be brief", "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]},
                         {"role": "assistant", "content": "ok"}, {"role": "user", "content": "three"}]}),
        ),
        (
            "anthropic_system_blocks",
            json!({"model": "Qwen3.8-Flash-Next", "max_tokens": 64, "temperature": 0,
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "messages": [{"role": "user", "content": "x"}]}),
        ),
        (
            "anthropic_tool_result_string",
            json!({"model": "Qwen3.8-Flash-Next", "max_tokens": 64, "temperature": 0,
            "tools": tools, "messages": read(json!("file text"))}),
        ),
        (
            "anthropic_tool_result_text_blocks",
            json!({"model": "Qwen3.8-Flash-Next", "max_tokens": 64,
            "temperature": 0, "tools": tools,
            "messages": read(json!([{"type": "text", "text": "PDF file read: a.txt"}, {"type": "text", "text": "more"}]))}),
        ),
        (
            "anthropic_tool_result_error",
            json!({"model": "Qwen3.8-Flash-Next", "max_tokens": 64, "temperature": 0,
            "tools": tools, "messages": [
                {"role": "user", "content": "read a.txt"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a.txt"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": true, "content": "no such file"},
                    {"type": "text", "text": "try b.txt"}]}]}),
        ),
        (
            "anthropic_tool_choice_stream",
            json!({"model": "Qwen3.8-Flash-Next", "max_tokens": 64, "stream": true,
            "tools": tools, "tool_choice": {"type": "auto"}, "messages": [{"role": "user", "content": "x"}]}),
        ),
    ]
}

/// The tokenizer sidecar's `encode_chat` request as `native/serve.rs::render` and `native/sidecar.rs::encode_chat`
/// build it: messages joined to strings where text-only, the request's tools, render flag, stops and kwargs.
fn sidecar_line(request: &ChatRequest) -> String {
    let messages = request
        .messages
        .iter()
        .map(|m| m.joined().unwrap_or_else(|| m.clone()))
        .collect::<Vec<_>>();
    let mut line = json!({
        "op": "encode_chat",
        "messages": messages,
        "render_tools": request.render_tools(),
        "stop": request.stop,
    });
    if let Some(tools) = request.tools_json() {
        line["tools_json"] = json!(tools);
    }
    if request.thinking {
        line["enable_thinking"] = json!(true);
    }
    if let Some(effort) = &request.reasoning_effort {
        line["reasoning_effort"] = effort.clone();
    }
    if let Some(preserve) = request.preserve_thinking {
        line["preserve_thinking"] = json!(preserve);
    }
    String::from_utf8(serde_json::to_vec(&line).unwrap()).unwrap()
}

fn outcome(policy: &ChatPolicy, chat: &[u8]) -> Map<String, Value> {
    let mut out = Map::new();
    let value: Value = serde_json::from_slice(chat).unwrap();
    match policy.validate_bytes(&value, chat) {
        Ok(request) => {
            out.insert(
                "validated".into(),
                json!(serde_json::to_string(&request).unwrap()),
            );
            out.insert(
                "normalized".into(),
                json!(serde_json::to_string(&request.normalized()).unwrap()),
            );
            out.insert("sidecar".into(), json!(sidecar_line(&request)));
        }
        Err(e) => {
            out.insert(
                "error".into(),
                json!(format!(
                    "{} {} {:?} {}",
                    e.status, e.code, e.param, e.message
                )),
            );
        }
    }
    out
}

fn outcomes() -> Map<String, Value> {
    let policy = policy();
    let mut all = Map::new();
    for (name, body) in chat_bodies() {
        all.insert(
            name.into(),
            Value::Object(outcome(&policy, &serde_json::to_vec(&body).unwrap())),
        );
    }
    for (name, body) in anthropic_bodies() {
        let out = match anthropic::convert_request(&serde_json::to_vec(&body).unwrap(), &policy) {
            Ok((chat, _)) => {
                let mut out = outcome(&policy, &chat);
                out.insert("converted".into(), json!(String::from_utf8(chat).unwrap()));
                out
            }
            Err(e) => Map::from_iter([(
                "error".to_owned(),
                json!(format!(
                    "{} {} {:?} {}",
                    e.status, e.code, e.param, e.message
                )),
            )]),
        };
        all.insert(name.into(), Value::Object(out));
    }
    all
}

#[test]
fn document_free_requests_take_the_unchanged_path() {
    let got = outcomes();
    if let Ok(path) = std::env::var("CERE_DOC_IDENTITY_RECORD") {
        std::fs::write(path, serde_json::to_vec_pretty(&got).unwrap()).unwrap();
        return;
    }
    let golden: Map<String, Value> = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/native/document_identity.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        golden.keys().collect::<Vec<_>>(),
        "fixture names"
    );
    for (name, want) in &golden {
        let have = &got[name];
        assert_eq!(
            have.as_object().unwrap().keys().collect::<Vec<_>>(),
            want.as_object().unwrap().keys().collect::<Vec<_>>(),
            "{name}: outcome fields"
        );
        for (field, value) in want.as_object().unwrap() {
            assert_eq!(&have[field], value, "{name}: {field}");
        }
    }
    // The battery covers both accepted paths, not only refusals.
    let served = golden
        .values()
        .filter(|v| v.get("sidecar").is_some())
        .count();
    assert!(served >= 12, "only {served} fixtures are served");
}
