//! M2.14 C4 (D3): `POST /v1/messages` and `POST /v1/messages/count_tokens` through `cered`, on the mock SGLang
//! (answering with the golden chat completions and chunks of `tests/fixtures/native/anthropic_goldens.json`) and
//! on the native toy executor ([`native_server`]): the default-off 404, unary and SSE, bearer auth, the Anthropic
//! error envelope, the capability refusals, `count_tokens` equal to the chat path's `usage.prompt_tokens`, cancel
//! on disconnect, the journal record, and per-surface idempotency, replay and resume.
mod native_server;
mod native_support;
mod support;

use std::time::Duration;

use cere_mock_sglang::{MockConfig, Scripted, ServerInfo};
use cere_runtime::anthropic;
use native_server::{Case, Native, TOKEN};
use serde_json::{json, Value};
use support::{http, journal, Harness, MODEL, TOKEN_A};

fn goldens() -> Value {
    let path = support::repo_root().join("tests/fixtures/native/anthropic_goldens.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn case<'a>(doc: &'a Value, section: &str, name: &str) -> &'a Value {
    doc[section]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("no {section} case {name}"))
}

/// The test-only switch (N6: no shipped config enables the surface).
const ENABLE: (&str, &str) = (
    "allow_cross_scope_prefix_cache = true",
    "allow_cross_scope_prefix_cache = true\napi_compat_anthropic = true",
);

async fn enabled(mock: MockConfig, tweaks: &[(&str, &str)]) -> Harness {
    let mut all = vec![ENABLE];
    all.extend_from_slice(tweaks);
    Harness::start(
        MockConfig {
            record_bodies: true,
            ..mock
        },
        &all,
    )
    .await
}

fn messages(h: &Harness, body: &Value) -> reqwest::RequestBuilder {
    h.post("/v1/messages", TOKEN_A).json(body)
}

fn count_tokens(h: &Harness, body: &Value) -> reqwest::RequestBuilder {
    h.post("/v1/messages/count_tokens", TOKEN_A).json(body)
}

async fn json_of(response: reqwest::Response) -> (u16, Value) {
    let status = response.status().as_u16();
    let bytes = response.bytes().await.unwrap();
    let value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("{status}: {e}: {}", String::from_utf8_lossy(&bytes)));
    (status, value)
}

/// SGLang's Anthropic envelope, exactly `{"type":"error","error":{"type", "message"}}`.
fn assert_envelope(status: u16, body: &Value, want: u16, kind: &str) {
    assert_eq!(status, want, "{body}");
    assert_eq!(
        body.as_object().unwrap().keys().collect::<Vec<_>>(),
        ["error", "type"],
        "{body}"
    );
    assert_eq!(body["type"], "error", "{body}");
    let error = body["error"].as_object().unwrap();
    assert_eq!(
        error.keys().collect::<Vec<_>>(),
        ["message", "type"],
        "{body}"
    );
    assert_eq!(error["type"], kind, "{body}");
    assert!(error["message"].is_string(), "{body}");
}

/// The chat body `cered` converts an Anthropic request into (the same conversation on the chat route), for the
/// SGLang Qwen deployment's policy.
fn converted(anthropic: &Value) -> Value {
    let config = cere_runtime::config::RuntimeConfig::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs/runtime/sglang-t4-qwen38.toml"),
    )
    .unwrap();
    let policy = cere_runtime::validate::ChatPolicy::from_config(&config, 0).unwrap();
    let (chat, _) =
        anthropic::convert_request(&serde_json::to_vec(anthropic).unwrap(), &policy).unwrap();
    serde_json::from_slice(&chat).unwrap()
}

/// The golden chunks plus the usage chunk SGLang sends last (cered always asks for it).
fn scripted_stream(case: &Value) -> Scripted {
    let mut chunks: Vec<Value> = case["chunks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let mut c = c.clone();
            c.as_object_mut().unwrap().remove("usage");
            c
        })
        .collect();
    chunks.push(json!({
        "id": "fixture-rid", "object": "chat.completion.chunk", "created": 1791656033u64,
        "model": MODEL, "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17}
    }));
    Scripted {
        response: Value::Null,
        chunks,
    }
}

/// Whole SSE frames `event: <type>\ndata: <json>\n\n`.
fn frames(text: &str) -> Vec<(String, Value)> {
    assert!(text.ends_with("\n\n"), "trailing partial frame: {text:?}");
    text.split_terminator("\n\n")
        .map(|frame| {
            let (event, data) = frame
                .strip_prefix("event: ")
                .and_then(|rest| rest.split_once("\ndata: "))
                .unwrap_or_else(|| panic!("event + data frame: {frame:?}"));
            (event.to_owned(), serde_json::from_str(data).unwrap())
        })
        .collect()
}

fn names(events: &[(String, Value)]) -> Vec<&str> {
    events.iter().map(|(n, _)| n.as_str()).collect()
}

/// The concatenated `text_delta`s of a stream.
fn streamed_text(events: &[(String, Value)]) -> String {
    events
        .iter()
        .filter_map(|(_, d)| d.pointer("/delta/text").and_then(Value::as_str))
        .collect()
}

/// Reads an open stream up to its first `content_block_delta` frame.
async fn read_to_delta(r: &mut reqwest::Response) {
    let mut seen = Vec::new();
    while !String::from_utf8_lossy(&seen).contains("content_block_delta") {
        match r.chunk().await.unwrap() {
            Some(bytes) => seen.extend_from_slice(&bytes),
            None => panic!("the stream ended early: {}", String::from_utf8_lossy(&seen)),
        }
    }
}

fn rid_of(response: &reqwest::Response) -> String {
    response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned()
}

// ---------------------------------------------------------------------------------------------------------
// The mock SGLang

#[tokio::test]
async fn capability_off_is_todays_404() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let before = h.mock.dispatched_rids().len();
    let unknown = h.post("/v1/no-such-route", TOKEN_A).send().await.unwrap();
    let (want_status, want_body) = (unknown.status(), unknown.bytes().await.unwrap());
    assert_eq!(want_status, 404);
    let body = json!({"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": "Hello"}]});
    for path in ["/v1/messages", "/v1/messages/count_tokens"] {
        let r = h.post(path, TOKEN_A).json(&body).send().await.unwrap();
        assert_eq!(r.status(), want_status, "{path}");
        assert_eq!(r.bytes().await.unwrap(), want_body, "{path}");
    }
    assert_eq!(h.mock.dispatched_rids().len(), before);
}

#[tokio::test]
async fn unary_and_stream_through_sglang() {
    let h = enabled(MockConfig::default(), &[]).await;
    let doc = goldens();
    for name in ["text", "tool_use", "stop_sequence"] {
        let case = case(&doc, "unary", name);
        h.mock.set_scripted(Some(Scripted {
            response: case["chat"].clone(),
            chunks: Vec::new(),
        }));
        let r = messages(&h, &case["anthropic"]).send().await.unwrap();
        assert!(r.headers().contains_key("x-cere-request-id"), "{name}");
        assert_eq!(r.headers()["content-type"], "application/json", "{name}");
        let (status, got) = json_of(r).await;
        assert_eq!(status, 200, "{name}: {got}");
        let (mut got, mut want) = (
            json!({ "message": got }),
            json!({ "message": case["message"] }),
        );
        for pointer in case["volatile"].as_array().unwrap() {
            for v in [&mut got, &mut want] {
                if let Some(v) = v.pointer_mut(pointer.as_str().unwrap()) {
                    *v = Value::Null;
                }
            }
        }
        assert_eq!(got, want, "{name}");
        // The backend got the converted chat body.
        let sent: Value = serde_json::from_slice(h.mock.received_bodies().last().unwrap()).unwrap();
        assert_eq!(sent["messages"], case["openai"]["messages"], "{name}");
    }
    for name in ["text", "tool_use"] {
        let case = case(&doc, "streams", name);
        h.mock.set_scripted(Some(scripted_stream(case)));
        let r = messages(&h, &case["anthropic"]).send().await.unwrap();
        assert_eq!(r.status(), 200, "{name}");
        assert_eq!(r.headers()["content-type"], "text/event-stream", "{name}");
        let text = r.text().await.unwrap();
        assert!(!text.contains("[DONE]"), "{name}: {text}");
        assert!(!text.contains("\"cere\""), "{name}: {text}");
        let got = frames(&text);
        let want: Vec<(String, Value)> = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| (e[0].as_str().unwrap().to_owned(), e[1].clone()))
            .collect();
        assert_eq!(names(&got), names(&want), "{name}: SSE event order");
        // D3 with ApiFit AF-L8: an SGLang stream opens before any count exists (input_tokens 0); the final
        // `message_delta` carries the backend's output tokens and the prompt count it reported.
        assert_eq!(got[0].1["message"]["usage"]["input_tokens"], 0, "{name}");
        let delta = &got[got.len() - 2].1;
        assert_eq!(delta["usage"]["output_tokens"], 7, "{name}: {delta}");
        let prompt = want[0].1["message"]["usage"]["input_tokens"].clone();
        assert_eq!(delta["usage"]["input_tokens"], prompt, "{name}: {delta}");
    }
}

/// The backend body without its per-request `rid`.
fn without_rid(bytes: &[u8]) -> Value {
    let mut body: Value = serde_json::from_slice(bytes).unwrap();
    body.as_object_mut().unwrap().remove("rid");
    body
}

/// ApiLenient AL-3/AL-4 (D4, Q6): dropped Anthropic fields stay dropped (the converted body is the one without
/// them) and are reported by `X-Cere-Ignored-Params`; the reserved `deterministic` is refused.
#[tokio::test]
async fn dropped_fields_are_reported_and_reserved_names_refused() {
    let h = enabled(MockConfig::default(), &[]).await;
    let doc = goldens();
    let case = case(&doc, "unary", "text");
    h.mock.set_scripted(Some(Scripted {
        response: case["chat"].clone(),
        chunks: Vec::new(),
    }));
    let r = messages(&h, &case["anthropic"]).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert!(!r.headers().contains_key("x-cere-ignored-params"));
    r.bytes().await.unwrap();
    let plain = without_rid(h.mock.received_bodies().last().unwrap());
    let mut body = case["anthropic"].clone();
    body["betas"] = json!(["interleaved-thinking-2025-05-14"]);
    body["container"] = json!("c-1");
    let r = messages(&h, &body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["x-cere-ignored-params"], "betas,container");
    r.bytes().await.unwrap();
    let sent = without_rid(h.mock.received_bodies().last().unwrap());
    assert_eq!(sent, plain, "the converted body is unchanged");
    body["deterministic"] = json!(true);
    let (status, got) = json_of(messages(&h, &body).send().await.unwrap()).await;
    assert_envelope(status, &got, 400, "invalid_request_error");
    let message = got["error"]["message"].as_str().unwrap();
    assert!(message.contains("reserved CERE extension"), "{got}");
}

#[tokio::test]
async fn count_tokens_equals_the_chat_prompt_tokens_on_sglang() {
    let h = enabled(MockConfig::default(), &[]).await;
    let doc = goldens();
    for name in ["text", "system_tools"] {
        let request = &case(&doc, "count_tokens", name)["anthropic"];
        let before = h.mock.dispatched_rids().len();
        let r = count_tokens(&h, request).send().await.unwrap();
        assert_eq!(r.headers()["content-type"], "application/json", "{name}");
        let (status, counted) = json_of(r).await;
        assert_eq!(status, 200, "{name}: {counted}");
        let n = counted["input_tokens"].as_u64().unwrap();
        assert_eq!(counted, json!({ "input_tokens": n }), "{name}");
        assert!(n > 0, "{name}");
        assert_eq!(
            h.mock.dispatched_rids().len(),
            before,
            "{name}: no chat dispatch"
        );
        // The same conversation on the chat route and on /v1/messages.
        let mut conversation = request.clone();
        conversation["max_tokens"] = json!(4);
        let (status, chat) = json_of(h.chat(&converted(&conversation)).send().await.unwrap()).await;
        assert_eq!(status, 200, "{name}: {chat}");
        assert_eq!(chat["usage"]["prompt_tokens"], n, "{name}: {chat}");
        let (status, message) = json_of(messages(&h, &conversation).send().await.unwrap()).await;
        assert_eq!(status, 200, "{name}: {message}");
        assert_eq!(message["usage"]["input_tokens"], n, "{name}: {message}");
    }
}

#[tokio::test]
async fn bearer_auth_refusals_and_the_anthropic_envelope() {
    let h = enabled(MockConfig::default(), &[]).await;
    let before = h.mock.dispatched_rids().len();
    let doc = goldens();
    let hello = json!({"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": "Hello"}]});
    for path in ["/v1/messages", "/v1/messages/count_tokens"] {
        // Bearer only: no token, or the Anthropic `x-api-key` header (N6 has not approved it), is 401.
        let r = h.http.post(h.url(path)).json(&hello).send().await.unwrap();
        let (status, body) = json_of(r).await;
        assert_envelope(status, &body, 401, "authentication_error");
        let r = h
            .http
            .post(h.url(path))
            .header("x-api-key", TOKEN_A)
            .json(&hello)
            .send()
            .await
            .unwrap();
        let (status, body) = json_of(r).await;
        assert_envelope(status, &body, 401, "authentication_error");
        // Malformed JSON, a protocol violation, an unserved model.
        let r = h
            .post(path, TOKEN_A)
            .header("content-type", "application/json")
            .body("{")
            .send()
            .await
            .unwrap();
        let (status, body) = json_of(r).await;
        assert_envelope(status, &body, 400, "invalid_request_error");
        let r = h
            .post(path, TOKEN_A)
            .json(&json!({"model": MODEL, "messages": "nope"}))
            .send()
            .await
            .unwrap();
        let (status, body) = json_of(r).await;
        assert_envelope(status, &body, 400, "invalid_request_error");
        let mut unserved = hello.clone();
        unserved["model"] = json!("no-such-model");
        let (status, body) =
            json_of(h.post(path, TOKEN_A).json(&unserved).send().await.unwrap()).await;
        assert_envelope(status, &body, 404, "not_found_error");
    }
    // Refusals that remain (ApiFit AF-L6): image blocks (M2.13) and an effort outside m212 D13.
    for name in ["image_block", "effort_high"] {
        let request = &case(&doc, "requests", name)["anthropic"];
        let (status, body) = json_of(messages(&h, request).send().await.unwrap()).await;
        assert_envelope(status, &body, 400, "invalid_request_error");
    }
    assert_eq!(h.mock.dispatched_rids().len(), before);
}

/// Audit F4: `stop_sequences` meet the chat route's caps (16 strings of 256 bytes): over them is the Anthropic
/// `invalid_request_error` envelope and nothing reaches the backend.
#[tokio::test]
async fn stop_sequences_over_the_caps_are_refused_before_the_backend() {
    let h = enabled(MockConfig::default(), &[]).await;
    let before = h.mock.dispatched_rids().len();
    let with = |stop: Value| {
        json!({"model": MODEL, "max_tokens": 8, "stop_sequences": stop,
               "messages": [{"role": "user", "content": "Hello"}]})
    };
    for stop in [json!(vec!["s"; 17]), json!(["x".repeat(257)])] {
        let (status, body) = json_of(messages(&h, &with(stop)).send().await.unwrap()).await;
        assert_envelope(status, &body, 400, "invalid_request_error");
    }
    assert_eq!(h.mock.dispatched_rids().len(), before);
}

/// ApiFit AF-L6: on a profile declaring M2.7's `top_k` and M2.8's `tool_calls_constrained` (the harness's SGLang Qwen
/// profile), `top_k` and `tool_choice` any/tool reach the backend as SGLang converts them (`top_k` as sent,
/// `required`, a named function).
#[tokio::test]
async fn top_k_and_tool_choice_reach_the_backend_on_a_capable_profile() {
    let h = enabled(MockConfig::default(), &[]).await;
    let doc = goldens();
    h.mock.set_scripted(Some(Scripted {
        response: case(&doc, "unary", "text")["chat"].clone(),
        chunks: Vec::new(),
    }));
    for (name, field) in [
        ("top_k", "top_k"),
        ("tool_choice_any", "tool_choice"),
        ("tool_choice_tool", "tool_choice"),
    ] {
        let golden = case(&doc, "requests", name);
        let (status, body) =
            json_of(messages(&h, &golden["anthropic"]).send().await.unwrap()).await;
        assert_eq!(status, 200, "{name}: {body}");
        let sent: Value = serde_json::from_slice(h.mock.received_bodies().last().unwrap()).unwrap();
        assert!(!golden["openai"][field].is_null(), "{name}");
        assert_eq!(sent[field], golden["openai"][field], "{name}: {sent}");
    }
}

/// M2.12 wiring of D3 (the harness profile declares `thinking_toggle`): `thinking` enabled/adaptive reaches the
/// backend as `chat_template_kwargs.enable_thinking: true` and `output_config.effort` low/medium as
/// `reasoning_effort` (effort != none => thinking on, m212 D13), as SGLang converts them; `budget_tokens` is not
/// forwarded (not enforced).
#[tokio::test]
async fn thinking_and_effort_reach_the_backend_on_a_thinking_profile() {
    let h = enabled(MockConfig::default(), &[]).await;
    let doc = goldens();
    h.mock.set_scripted(Some(Scripted {
        response: case(&doc, "unary", "text")["chat"].clone(),
        chunks: Vec::new(),
    }));
    for (name, kwargs, effort) in [
        (
            "thinking_enabled_budget",
            json!({"enable_thinking": true}),
            Value::Null,
        ),
        (
            "thinking_adaptive",
            json!({"enable_thinking": true}),
            Value::Null,
        ),
        (
            "thinking_display_omitted",
            json!({"enable_thinking": true}),
            Value::Null,
        ),
        ("effort_low", Value::Null, json!("low")),
        ("effort_medium", Value::Null, json!("medium")),
    ] {
        let golden = case(&doc, "requests", name);
        let (status, body) =
            json_of(messages(&h, &golden["anthropic"]).send().await.unwrap()).await;
        assert_eq!(status, 200, "{name}: {body}");
        let sent: Value = serde_json::from_slice(h.mock.received_bodies().last().unwrap()).unwrap();
        assert_eq!(sent["chat_template_kwargs"], kwargs, "{name}: {sent}");
        assert_eq!(sent["reasoning_effort"], effort, "{name}: {sent}");
        assert_eq!(
            sent["chat_template_kwargs"], golden["openai"]["chat_template_kwargs"],
            "{name}"
        );
        assert_eq!(
            sent["reasoning_effort"], golden["openai"]["reasoning_effort"],
            "{name}"
        );
        assert!(sent.get("budget_tokens").is_none(), "{name}: {sent}");
    }
}

#[tokio::test]
async fn queue_full_is_429_and_not_ready_is_503_in_the_envelope() {
    let h = enabled(
        MockConfig {
            token_latency: Duration::from_millis(30),
            natural_output_tokens: 8,
            ..MockConfig::default()
        },
        &[
            ("max_backend_inflight = 2", "max_backend_inflight = 1"),
            ("per_scope_inflight = 2", "per_scope_inflight = 1"),
            ("queue_depth = 64", "queue_depth = 1"),
        ],
    )
    .await;
    let hello = json!({"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": "Hello"}]});
    // The first request stays at its finish until the 429 is seen: the second one queues behind it however late
    // it arrives.
    h.mock.hold_finish(true);
    let first = tokio::spawn(h.chat(&support::chat_body(false)).send());
    until("one in flight", || h.app.admission_snapshot().inflight == 1).await;
    until("the backend call", || h.mock.in_flight() == 1).await;
    let second = tokio::spawn(h.chat(&support::chat_body(false)).send());
    until("one queued", || h.app.admission_snapshot().queued == 1).await;
    let (status, body) = json_of(messages(&h, &hello).send().await.unwrap()).await;
    assert_envelope(status, &body, 429, "rate_limit_error");
    h.mock.hold_finish(false);
    assert_eq!(first.await.unwrap().unwrap().status(), 200);
    assert_eq!(second.await.unwrap().unwrap().status(), 200);

    // A worker whose backend drifted is not ready: 503 on both routes.
    h.mock.set_server_info_override(Some(ServerInfo {
        kv_cache_dtype: "bf16".to_owned(),
        ..ServerInfo::default()
    }));
    h.app.readiness(0).probe().await;
    let before = h.mock.dispatched_rids().len();
    for path in ["/v1/messages", "/v1/messages/count_tokens"] {
        let (status, body) =
            json_of(h.post(path, TOKEN_A).json(&hello).send().await.unwrap()).await;
        assert_envelope(status, &body, 503, "overloaded_error");
    }
    assert_eq!(h.mock.dispatched_rids().len(), before);
}

/// Waits for `done` (bounded like `Harness::terminal`, for a loaded host; the bound is no part of any assertion).
async fn until(what: &str, done: impl Fn() -> bool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while !done() {
        assert!(
            std::time::Instant::now() < deadline,
            "never reached: {what}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test]
async fn client_disconnect_cancels_and_aborts() {
    let h = enabled(
        MockConfig {
            token_latency: Duration::from_millis(40),
            natural_output_tokens: 100,
            ..MockConfig::default()
        },
        &[],
    )
    .await;
    let body = json!({
        "model": MODEL, "max_tokens": 100, "stream": true,
        "messages": [{"role": "user", "content": "Hello"}],
    });
    let mut r = messages(&h, &body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let rid = rid_of(&r);
    read_to_delta(&mut r).await;
    drop(r);
    let record = h.terminal(&rid).await;
    assert_eq!(record["state"], "cancelled", "{record}");
    assert_eq!(record["reason"], "client_disconnected", "{record}");
    until("the backend abort", || {
        h.mock.abort_requests().iter().any(|a| a.rid == rid)
    })
    .await;
}

/// The journal records of `rid`, without the fields that differ between any two requests.
fn journaled(h: &Harness, rid: &str) -> Vec<Value> {
    journal(h.state_dir.path())
        .into_iter()
        .filter(|r| r["rid"] == rid)
        .map(|mut r| {
            let record = r.as_object_mut().unwrap();
            for volatile in ["rid", "at_ms", "digest", "timings"] {
                record.remove(volatile);
            }
            r
        })
        .collect()
}

#[tokio::test]
async fn journal_record_is_the_chats_but_the_surface() {
    let h = enabled(MockConfig::default(), &[]).await;
    let request = json!({"model": MODEL, "max_tokens": 4, "messages": [{"role": "user", "content": "Hello"}]});
    let chat = h.chat(&converted(&request)).send().await.unwrap();
    assert_eq!(chat.status(), 200);
    let chat_rid = rid_of(&chat);
    chat.bytes().await.unwrap();
    let message = messages(&h, &request).send().await.unwrap();
    assert_eq!(message.status(), 200);
    let message_rid = rid_of(&message);
    message.bytes().await.unwrap();
    h.terminal(&chat_rid).await;
    h.terminal(&message_rid).await;
    let chat = journaled(&h, &chat_rid);
    let mut message = journaled(&h, &message_rid);
    assert_eq!(message[0]["type"], "admitted", "{message:?}");
    assert_eq!(message[0]["surface"], "anthropic", "{message:?}");
    assert!(chat[0].get("surface").is_none(), "{chat:?}");
    message[0].as_object_mut().unwrap().remove("surface");
    assert_eq!(message, chat);
}

#[tokio::test]
async fn idempotency_replay_and_resume_are_per_surface() {
    let h = enabled(MockConfig::default(), &[]).await;
    let doc = goldens();
    let text = case(&doc, "unary", "text");
    h.mock.set_scripted(Some(Scripted {
        response: text["chat"].clone(),
        chunks: Vec::new(),
    }));
    let request = &text["anthropic"];
    // An Anthropic replay returns the Anthropic body, byte for byte, without a second dispatch.
    let first = messages(&h, request)
        .header("Idempotency-Key", "a-1")
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), 200);
    let rid = first.headers()["x-cere-request-id"].clone();
    let first = first.bytes().await.unwrap();
    let dispatched = h.mock.dispatched_rids().len();
    let replay = messages(&h, request)
        .header("Idempotency-Key", "a-1")
        .send()
        .await
        .unwrap();
    assert_eq!(replay.status(), 200);
    assert_eq!(replay.headers()["x-cere-request-id"], rid);
    assert_eq!(replay.bytes().await.unwrap(), first);
    assert_eq!(h.mock.dispatched_rids().len(), dispatched);
    let body: Value = serde_json::from_slice(&first).unwrap();
    assert_eq!(body["type"], "message", "{body}");
    // The same key on chat then on Anthropic (and the other way round) conflicts.
    h.mock.set_scripted(None);
    let chat = h
        .chat(&converted(request))
        .header("Idempotency-Key", "c-1")
        .send()
        .await
        .unwrap();
    assert_eq!(chat.status(), 200);
    let (status, body) = json_of(
        messages(&h, request)
            .header("Idempotency-Key", "c-1")
            .send()
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, 409, "{body}");
    assert_envelope(status, &body, 409, "api_error");
    let chat = h
        .chat(&converted(request))
        .header("Idempotency-Key", "a-1")
        .send()
        .await
        .unwrap();
    assert_eq!(chat.status(), 409);
    let chat: Value = chat.json().await.unwrap();
    assert_eq!(chat["error"]["code"], "idempotency_conflict", "{chat}");
    // An Anthropic stream cannot be resumed (non-chat surface).
    let stream = case(&doc, "streams", "text");
    h.mock.set_scripted(Some(scripted_stream(stream)));
    let streamed = messages(&h, &stream["anthropic"]).send().await.unwrap();
    assert_eq!(streamed.status(), 200);
    let rid = rid_of(&streamed);
    streamed.text().await.unwrap();
    let resume = h
        .get(&format!("/v1/cere/requests/{rid}/events"), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(resume.status(), 409);
    let resume: Value = resume.json().await.unwrap();
    assert_eq!(resume["error"]["code"], "stream_not_replayable", "{resume}");
}

// ---------------------------------------------------------------------------------------------------------
// The native toy executor

/// `server_native`'s `TEXT` prompt: the stub sidecar encodes it as 716 tokens ending in `1`, so the toy's
/// output is [2235, 2 ("ab"), 780, ...] with no stop in reach (`hold_after=2` holds at "ab").
fn text_prompt() -> String {
    let mut s = "cere-native-sentinel-prompt".to_owned();
    while s.len() < 714 {
        s.push('x');
    }
    s.push('1');
    s
}

fn native_case(device_id: &str) -> Case {
    Case {
        device_id: device_id.to_owned(),
        compat: &["api_compat_anthropic"],
        ..Case::matching()
    }
}

fn native_post(native: &Native, path: &str, body: &Value) -> reqwest::RequestBuilder {
    http()
        .post(native.cered.url(path))
        .bearer_auth(TOKEN)
        .json(body)
}

#[tokio::test]
async fn native_unary_stream_and_count_tokens() {
    let native = Native::ready(native_case("toy")).await;
    // The native deployment is greedy-only: temperature 0, as on the chat route.
    let conversation = json!([{"role": "user", "content": text_prompt()}]);
    let request =
        json!({"model": MODEL, "max_tokens": 4, "temperature": 0, "messages": conversation});
    let (status, chat) = json_of(native.chat(&converted(&request)).send().await.unwrap()).await;
    assert_eq!(status, 200, "{chat}");
    let text = chat["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(text, "ab", "{chat}");
    // A matched stop reports `end_turn`, as SGLang does (owner ruling 4).
    let stop_reason = match chat["choices"][0]["finish_reason"].as_str() {
        Some("length") => "max_tokens",
        _ => "end_turn",
    };
    let output_tokens = &chat["usage"]["completion_tokens"];

    let (status, unary) = json_of(
        native_post(&native, "/v1/messages", &request)
            .send()
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, 200, "{unary}");
    assert_eq!(unary["type"], "message", "{unary}");
    assert_eq!(unary["role"], "assistant", "{unary}");
    assert_eq!(unary["model"], MODEL, "{unary}");
    assert_eq!(
        unary["content"],
        json!([{"type": "text", "text": text}]),
        "{unary}"
    );
    assert_eq!(unary["stop_reason"], stop_reason, "{unary}");
    assert_eq!(
        unary["usage"]["input_tokens"], chat["usage"]["prompt_tokens"],
        "{unary}"
    );
    assert_eq!(&unary["usage"]["output_tokens"], output_tokens, "{unary}");
    assert!(unary.get("cere").is_none(), "{unary}");

    let mut streamed = request.clone();
    streamed["stream"] = json!(true);
    let r = native_post(&native, "/v1/messages", &streamed)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["content-type"], "text/event-stream");
    let events = frames(&r.text().await.unwrap());
    let names = names(&events);
    assert_eq!(names.first(), Some(&"message_start"), "{names:?}");
    assert_eq!(names.last(), Some(&"message_stop"), "{names:?}");
    assert_eq!(streamed_text(&events), text);
    let delta = &events[events.len() - 2].1;
    assert_eq!(delta["delta"]["stop_reason"], stop_reason, "{delta}");
    assert_eq!(&delta["usage"]["output_tokens"], output_tokens, "{delta}");

    let submissions = native.submissions().await;
    let count = json!({"model": MODEL, "messages": conversation});
    let r = native_post(&native, "/v1/messages/count_tokens", &count)
        .send()
        .await
        .unwrap();
    let (status, counted) = json_of(r).await;
    assert_eq!(status, 200, "{counted}");
    assert_eq!(
        counted,
        json!({ "input_tokens": chat["usage"]["prompt_tokens"] })
    );
    assert_eq!(native.submissions().await, submissions, "nothing submitted");
}

/// m27 D10 (Q4 (a)): a native `/v1/messages` request without `temperature` (as typical Anthropic clients send it)
/// is served, sampled with the model's `generation_config.json` defaults (the fixture's 1.0/20/0.95); the toy
/// records the params of every item. Before M2.7 D10 it was 400 `unsupported_parameter` (O24-1).
#[tokio::test]
async fn native_absent_temperature_samples_with_the_model_defaults() {
    let t = native_support::Tmp::new("anthropic-defaults");
    let record = t.join("steps.record");
    let native = Native::ready(native_case(&format!("toy,record={}", record.display()))).await;
    let body = json!({
        "model": MODEL, "max_tokens": 3,
        "messages": [{"role": "user", "content": text_prompt()}],
    });
    let (status, unary) = json_of(
        native_post(&native, "/v1/messages", &body)
            .send()
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, 200, "{unary}");
    // The readiness SMOKE's greedy step comes first; the request's items (up to 3: a random seed may sample the
    // toy EOS early) follow.
    let steps = std::fs::read_to_string(&record).unwrap();
    let sampled: Vec<&str> = steps
        .lines()
        .filter(|s| !s.contains("sampling=null"))
        .collect();
    assert!((1..=3).contains(&sampled.len()), "{steps}");
    for step in sampled {
        for want in [
            "temperature=1 ",
            "top_k=20 ",
            "top_p=0.949999988 ",
            "min_p=0 ",
        ] {
            assert!(step.contains(want), "{want}: {step}");
        }
    }
}

#[tokio::test]
async fn native_disconnect_cancels_the_generation() {
    let native = Native::ready(native_case("toy,hold_after=2")).await;
    let body = json!({
        "model": MODEL, "max_tokens": 16, "stream": true,
        "temperature": 0, "messages": [{"role": "user", "content": text_prompt()}],
    });
    let mut r = native_post(&native, "/v1/messages", &body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let rid = rid_of(&r);
    read_to_delta(&mut r).await;
    drop(r);
    let status = native.terminal(&rid).await;
    assert_eq!(
        (&status["state"], &status["reason"]),
        (&json!("cancelled"), &json!("client_disconnected")),
        "{status}"
    );
}

/// `server_native`'s prompt of `len` tokens (the stub sidecar's count) ending in `last`.
fn sized_prompt(len: usize, last: char) -> String {
    let mut s = "cere-native-sentinel-prompt".to_owned();
    while s.len() < len - 2 {
        s.push('x');
    }
    s.push(last);
    s
}

/// ApiFit AF-06 through cered on the native toy (200,000-token context): a prompt over the context is Anthropic's
/// `prompt is too long: N tokens > M maximum`; a prompt that fits but not with `max_tokens` is Anthropic's `input
/// length and max_tokens` form; unary and streamed, exactly the Anthropic envelope; the chat route keeps
/// `context_length_exceeded` and its message.
#[tokio::test]
async fn native_context_refusals_use_anthropics_wording() {
    let native = Native::ready(native_case("toy")).await;
    let cases = [
        (
            200_010,
            1,
            "prompt is too long: 200010 tokens > 200000 maximum".to_owned(),
            "the prompt (200010 tokens) plus max_tokens (1) exceeds the context limit of 200000 tokens",
        ),
        (
            199_996,
            5,
            "input length and `max_tokens` exceed context limit: 199996 + 5 > 200000, decrease input length or \
             `max_tokens` and try again"
                .to_owned(),
            "the prompt (199996 tokens) plus max_tokens (5) exceeds the context limit of 200000 tokens",
        ),
    ];
    for (len, max_tokens, anthropic_message, chat_message) in cases {
        let body = json!({
            "model": MODEL, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": sized_prompt(len, '1')}],
        });
        let want = json!({"type": "error", "error": {"type": "invalid_request_error", "message": anthropic_message}});
        let r = native_post(&native, "/v1/messages", &body)
            .send()
            .await
            .unwrap();
        let (status, got) = json_of(r).await;
        assert_envelope(status, &got, 400, "invalid_request_error");
        assert_eq!(got, want, "{len}");
        let mut streamed = body.clone();
        streamed["stream"] = json!(true);
        let r = native_post(&native, "/v1/messages", &streamed)
            .send()
            .await
            .unwrap();
        if r.status() == 400 {
            let (status, got) = json_of(r).await;
            assert_envelope(status, &got, 400, "invalid_request_error");
            assert_eq!(got, want, "{len} streamed");
        } else {
            let events = frames(&r.text().await.unwrap());
            let error = events.iter().find(|(name, _)| name == "error").unwrap();
            assert_eq!(error.1, want, "{len} streamed");
        }
        let (status, chat) = json_of(native.chat(&converted(&body)).send().await.unwrap()).await;
        assert_eq!(status, 400, "{chat}");
        assert_eq!(chat["error"]["code"], "context_length_exceeded", "{chat}");
        assert_eq!(chat["error"]["message"], chat_message, "{chat}");
    }
}

/// ApiFit (Main 2026-10-21, `coding-agent`): the request a coding agent sends a model it does not know: fixed-budget
/// thinking (`thinking {type: enabled, budget_tokens}`), no effort or `context_management`, `max_tokens` 32,000,
/// system blocks with `cache_control`, metadata, tools, a mid-conversation system message, streamed. Accepted on a
/// thinking deployment whose output limit is 32,768; the backend gets thinking on and the reminder in place.
#[tokio::test]
async fn coding_agent_fixed_budget_request_shape_is_accepted() {
    let h = enabled(MockConfig::default(), &[]).await;
    let doc = goldens();
    h.mock
        .set_scripted(Some(scripted_stream(case(&doc, "streams", "text"))));
    let body = json!({
        "model": MODEL,
        "max_tokens": 32000,
        "stream": true,
        "thinking": {"type": "enabled", "budget_tokens": 31999},
        "metadata": {"user_id": "fixture-user"},
        "system": [
            {"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Be terse."},
        ],
        "tools": [{"name": "get_weather", "description": "Weather for a city",
                   "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi."}]},
            {"role": "system", "content": "<system-reminder>The date changed.</system-reminder>"},
            {"role": "user", "content": "Next"},
        ],
    });
    let r = messages(&h, &body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["content-type"], "text/event-stream");
    let events = frames(&r.text().await.unwrap());
    assert_eq!(names(&events).last(), Some(&"message_stop"), "{events:?}");
    let sent: Value = serde_json::from_slice(h.mock.received_bodies().last().unwrap()).unwrap();
    assert_eq!(
        sent["chat_template_kwargs"],
        json!({"enable_thinking": true}),
        "{sent}"
    );
    assert_eq!(sent["max_completion_tokens"], 32000, "{sent}");
    assert_eq!(
        sent["messages"],
        json!([
            {"role": "system", "content": "You are a coding agent.\nBe terse."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": "<system-message>\n<system-reminder>The date changed.</system-reminder>\n</system-message>"},
            {"role": "user", "content": "Next"},
        ]),
        "{sent}"
    );
}

/// ApiFit AF-11 / AF-L6 natively: the `coding-agent` request shape through cered over the speculating toy (the
/// `-v29` combinations: sampled and greedy decode and verify): no `temperature` (sampled with the model defaults),
/// `thinking` enabled with a budget, a mid-conversation system message, tools with `tool_choice` absent, `any`
/// (`required`, constrained, run without speculation) and `tool`; unary and streamed.
#[tokio::test]
async fn native_coding_agent_request_shape_is_served() {
    let native = Native::ready(Case {
        speculation: true,
        ..native_case("toy")
    })
    .await;
    let tools = json!([{"name": "get_weather", "description": "Weather for a city",
                        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                                         "required": ["city"]}}]);
    for tool_choice in [
        Value::Null,
        json!({"type": "any"}),
        json!({"type": "tool", "name": "get_weather"}),
    ] {
        for stream in [false, true] {
            let mut body = json!({
                "model": MODEL, "max_tokens": 16, "stream": stream,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "metadata": {"user_id": "fixture-user"},
                "system": [{"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral"}}],
                "tools": tools,
                "messages": [
                    {"role": "user", "content": text_prompt()},
                    {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                    {"role": "system", "content": "<system-reminder>r</system-reminder>"},
                    {"role": "user", "content": "next"},
                ],
            });
            if !tool_choice.is_null() {
                body["tool_choice"] = tool_choice.clone();
            }
            let r = native_post(&native, "/v1/messages", &body)
                .send()
                .await
                .unwrap();
            if stream {
                assert_eq!(r.status(), 200, "{tool_choice}");
                let events = frames(&r.text().await.unwrap());
                assert_eq!(
                    names(&events).last(),
                    Some(&"message_stop"),
                    "{tool_choice}: {events:?}"
                );
            } else {
                let (status, got) = json_of(r).await;
                assert_eq!(status, 200, "{tool_choice}: {got}");
                assert_eq!(got["type"], "message", "{got}");
            }
        }
    }
}

/// ApiFit AF-13 (AF-L8): on a native deployment `message_start.usage.input_tokens` is the real prompt count (the
/// count_tokens answer), and the final usage of stream and unary splits it into `input_tokens` +
/// `cache_read_input_tokens` (prefix-cache hits) with the cumulative `output_tokens` in `message_delta`.
#[tokio::test]
async fn native_stream_usage_follows_anthropic_semantics() {
    let native = Native::ready(native_case("toy")).await;
    let body = json!({"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": text_prompt()}]});
    let (status, counted) = json_of(
        native_post(&native, "/v1/messages/count_tokens", &body)
            .send()
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, 200, "{counted}");
    let prompt = counted["input_tokens"].as_u64().unwrap();
    assert!(prompt > 0);
    let input_of = |usage: &Value| {
        usage["input_tokens"].as_u64().unwrap()
            + usage["cache_read_input_tokens"].as_u64().unwrap_or(0)
    };
    // Twice: the second request may hit the first's prefix, which moves tokens to cache_read_input_tokens.
    for round in 0..2 {
        let mut streamed = body.clone();
        streamed["stream"] = json!(true);
        let r = native_post(&native, "/v1/messages", &streamed)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        let events = frames(&r.text().await.unwrap());
        let start = &events[0].1["message"]["usage"];
        assert_eq!(start["input_tokens"], prompt, "round {round}: {start}");
        assert_eq!(start["output_tokens"], 0, "round {round}: {start}");
        let delta = &events.iter().find(|(e, _)| e == "message_delta").unwrap().1["usage"];
        let output = delta["output_tokens"].as_u64().unwrap();
        assert!(output > 0, "round {round}: {delta}");
        if delta.get("input_tokens").is_some() {
            assert_eq!(input_of(delta), prompt, "round {round}: {delta}");
            assert!(
                delta["cache_read_input_tokens"].as_u64().unwrap() > 0,
                "round {round}: {delta}"
            );
        }
        let (status, unary) = json_of(
            native_post(&native, "/v1/messages", &body)
                .send()
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(status, 200, "{unary}");
        assert_eq!(input_of(&unary["usage"]), prompt, "round {round}: {unary}");
        assert_eq!(
            unary["usage"]["output_tokens"], output,
            "round {round}: {unary}"
        );
    }
}

/// ApiFit AF-15 on the Anthropic surface (AF-L7): `thinking.budget_tokens` of `enabled` thinking binds on a native
/// deployment with the thinking capability. This tokenizer spells `</think>` with an ID the toy does not reach, so
/// adaptive thinking (no budget) makes the whole output one thinking block; with `budget_tokens` B the message has
/// a thinking block (B-2 kept tokens and the closing newline) and a text block after it, unary and stream alike,
/// and a stream that ends with `message_stop`.
#[tokio::test]
async fn native_thinking_budget_tokens_bind() {
    let vocab = native_support::FIXTURE_VOCAB;
    let added: Vec<Value> = (0..vocab)
        .map(|id| {
            let content = match vocab - id {
                1 => "</think>".to_owned(),
                2 => "Ċ".to_owned(),
                3 => "ĊĊ".to_owned(),
                _ => format!("[{id}]"),
            };
            json!({"id": id, "content": content, "special": false})
        })
        .collect();
    let tokenizer = serde_json::to_vec(&json!({
        "version": "1.0",
        "model": {"type": "BPE", "vocab": {}},
        "added_tokens": added,
        "decoder": {"type": "ByteLevel"},
    }))
    .unwrap();
    let native = Native::ready(Case {
        tokenizer: Some(tokenizer),
        capabilities_patch: json!({"thinking_toggle": true}),
        ..native_case("toy")
    })
    .await;
    // Anthropic's smallest budget is 1024.
    let request = |budget: Option<u64>, stream: bool| {
        let thinking = match budget {
            Some(budget) => json!({"type": "enabled", "budget_tokens": budget}),
            None => json!({"type": "adaptive"}),
        };
        json!({"model": MODEL, "max_tokens": 1100, "temperature": 0, "stream": stream, "thinking": thinking,
               "messages": [{"role": "user", "content": text_prompt()}]})
    };
    let kinds = |message: &Value| -> Vec<String> {
        message["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["type"].as_str().unwrap().to_owned())
            .collect()
    };
    // Without a budget the whole output is one thinking block.
    let (status, free) = json_of(
        native_post(&native, "/v1/messages", &request(None, false))
            .send()
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, 200, "{free}");
    assert_eq!(kinds(&free), ["thinking"], "{free}");
    let (status, cut) = json_of(
        native_post(&native, "/v1/messages", &request(Some(1024), false))
            .send()
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, 200, "{cut}");
    assert_eq!(kinds(&cut), ["thinking", "text"], "{cut}");
    let thinking = cut["content"][0]["thinking"].as_str().unwrap();
    assert_eq!(thinking.trim_end().matches('[').count(), 1022, "{cut}");
    assert!(free["content"][0]["thinking"]
        .as_str()
        .unwrap()
        .starts_with(thinking.trim_end()));
    assert_eq!(
        cut["usage"]["output_tokens"], free["usage"]["output_tokens"],
        "{cut}"
    );
    let r = native_post(&native, "/v1/messages", &request(Some(1024), true))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let events = frames(&r.text().await.unwrap());
    assert_eq!(names(&events).last(), Some(&"message_stop"), "{events:?}");
    let blocks: Vec<&str> = events
        .iter()
        .filter(|(e, _)| e == "content_block_start")
        .map(|(_, d)| d["content_block"]["type"].as_str().unwrap())
        .collect();
    assert_eq!(blocks, ["thinking", "text"], "{events:?}");
    let delta = &events.iter().find(|(e, _)| e == "message_delta").unwrap().1;
    assert_eq!(
        delta["usage"]["output_tokens"], cut["usage"]["output_tokens"],
        "{delta}"
    );
}
