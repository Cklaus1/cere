//! C5 (spec m214 D4): the stateless OpenAI Responses surface against `tests/fixtures/native/responses_goldens.json`,
//! recorded in the pinned image from SGLang's own `/v1/responses` route (non-harmony `OpenAIServingResponses`) by
//! `scripts/native/gen_api_fixtures.py`.
//!
//! Golden rules (the generator's `masked` self-check): values at a case's `volatile` JSON pointers are per run and
//! compared as null; `openai` is `model_dump(exclude_unset)`, so its nulls are droppable; `cere_override` marks a
//! case where the spec's rule wins over SGLang's recorded output (`{status, type}` refusal or an `echo` patch). Key
//! order is compared (the parse is order-preserving, [`Node`]). TrackM214 ruling 4: the stream `response.completed`
//! usage is not derivable from the recorded chat chunks (SGLang counts engine outputs), so it is volatile here;
//! SGLang's pydantic errors are compared by status/type only (their messages carry a Python traceback cered never
//! echoes).
mod support;

use cere_mock_sglang::{MockConfig, Scripted};
use cere_runtime::api_types::ApiError;
use cere_runtime::responses::{self, Event, Ids, Node, ResponsesStream};
use cere_runtime::validate::ChatPolicy;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use support::{Harness, TOKEN_A};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn goldens() -> Node {
    let path = root().join("tests/fixtures/native/responses_goldens.json");
    serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap()
}

fn cases<'a>(doc: &'a Node, section: &str) -> &'a [Node] {
    match doc.get(section) {
        Some(Node::Arr(items)) => items,
        _ => panic!("golden section {section} missing"),
    }
}

fn name(case: &Node) -> &str {
    case.get("name").and_then(Node::as_str).unwrap()
}

fn bytes(node: &Node) -> Vec<u8> {
    serde_json::to_vec(node).unwrap()
}

/// `node` with the value at each JSON pointer replaced by null (absent pointers skipped, as `masked`).
fn masked(node: &Node, pointers: &[String]) -> Node {
    let mut out = node.clone();
    for pointer in pointers {
        if let Some(slot) = out.pointer_mut(pointer) {
            *slot = Node::Null;
        }
    }
    out
}

fn volatile(case: &Node) -> Vec<String> {
    match case.get("volatile") {
        Some(Node::Arr(items)) => items
            .iter()
            .map(|p| p.as_str().unwrap().to_owned())
            .collect(),
        _ => Vec::new(),
    }
}

/// Object members whose value is null removed, recursively (`exclude_unset` nulls are droppable).
fn without_nulls(node: &Node) -> Node {
    match node {
        Node::Obj(members) => Node::Obj(
            members
                .iter()
                .filter(|(_, v)| !matches!(v, Node::Null))
                .map(|(k, v)| (k.clone(), without_nulls(v)))
                .collect(),
        ),
        Node::Arr(items) => Node::Arr(items.iter().map(without_nulls).collect()),
        other => other.clone(),
    }
}

/// The relative pointers of a case rebased under `prefix` (`/response/id` -> `/id` for prefix `/response`).
fn under(pointers: &[String], prefix: &str) -> Vec<String> {
    pointers
        .iter()
        .filter_map(|p| p.strip_prefix(prefix).map(str::to_owned))
        .collect()
}

/// A policy that accepts every chat feature the converted bodies use (greedy and sampled, tools, both backends).
fn policy(native: bool) -> ChatPolicy {
    ChatPolicy {
        served_model_name: "Qwen3.8-Flash-Next".to_owned(),
        served_models: vec!["Qwen3.8-Flash-Next".to_owned()],
        max_messages: 256,
        default_max_output_tokens: 4096,
        max_output_tokens: 32768,
        greedy: true,
        sampled: true,
        tools_greedy: true,
        tools_sampled: true,
        grammar_decodes: Vec::new(),
        json_schema_keywords: Vec::new(),
        max_top_logprobs: None,
        sampling_params: Vec::new(),
        max_choices: 1,
        native,
        // The Responses adapter refuses `input_image` itself (M2.14), so no image combination is needed.
        images_greedy: false,
        images_sampled: false,
        remote_workers: false,
        image_context_tokens: None,
        image_speculation_greedy: false,
        image_speculation_sampled: false,
        image_capture: false,
        // The converted bodies carry no M2.12 field (`reasoning.effort` is refused before conversion).
        thinking_toggle: false,
        token_controls: Vec::new(),
        vocab_size: None,
    }
}

fn expected_error(case: &Node) -> Option<(u16, String)> {
    let pick = |n: &Node| {
        Some((
            n.get("status")?.as_u64()? as u16,
            n.get("type")?.as_str()?.to_owned(),
        ))
    };
    if let Some(over) = case
        .get("cere_override")
        .filter(|o| o.get("status").is_some())
    {
        return pick(over);
    }
    case.get("error").and_then(pick)
}

/// A refusal matches the golden's status and type. SGLang's own 400s (FastAPI's `Bad Request` for pydantic
/// errors, `invalid_request_error` from the serving layer) are a CERE field error: `invalid_value` or
/// `missing_required_parameter` (TrackM214 ruling 4: status/type only).
fn assert_error(case: &str, got: &ApiError, status: u16, kind: &str) {
    let codes: &[&str] = match kind {
        "Bad Request" | "invalid_request_error" => &["invalid_value", "missing_required_parameter"],
        kind => &[kind],
    };
    assert!(
        got.status == status && codes.contains(&got.code),
        "{case}: {} {} {}",
        got.status,
        got.code,
        got.message
    );
    assert!(
        !got.message.contains("File \"") && !got.message.contains(".py"),
        "{case}: a refusal never echoes a Python traceback"
    );
}

/// Messages with text-part content joined into one string (the adapters' rule: the chat template concatenates
/// text items, so the prompt is the same).
fn joined_parts(messages: &Node) -> Node {
    let Node::Arr(items) = messages else {
        panic!("messages is not an array")
    };
    let join = |message: &Node| match message {
        Node::Obj(members) => Node::Obj(
            members
                .iter()
                .map(|(k, v)| match (k.as_str(), v) {
                    ("content", Node::Arr(parts)) => (
                        k.clone(),
                        Node::Str(
                            parts
                                .iter()
                                .map(|p| p.get("text").unwrap().as_str().unwrap())
                                .collect(),
                        ),
                    ),
                    _ => (k.clone(), v.clone()),
                })
                .collect(),
        ),
        other => other.clone(),
    };
    Node::Arr(items.iter().map(join).collect())
}

/// The chat completion every request case's engine produced (`"OK"`, EOS, prompt_tokens 10).
fn ok_chat() -> Value {
    json!({
        "id": "fixture-rid", "object": "chat.completion", "created": 1791656033u64,
        "model": "Qwen3.8-Flash-Next",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
    })
}

fn render(case: &Node, chat: &Value) -> Node {
    let converted = responses::convert(&bytes(case.get("responses").unwrap()), &policy(false))
        .unwrap_or_else(|e| panic!("{}: {e:?}", name(case)));
    responses::render_unary(&converted.echo, &mut Ids::seeded([7; 16]), chat)
}

fn patched(case: &Node, response: &Node) -> Node {
    let mut response = response.clone();
    if let Some(Node::Obj(echo)) = case.get("cere_override").and_then(|o| o.get("echo")) {
        for (key, value) in echo {
            *response.pointer_mut(&format!("/{key}")).unwrap() = value.clone();
        }
    }
    response
}

#[test]
fn request_conversion_matches_sglang() {
    let doc = goldens();
    let mut checked = 0;
    for case in cases(&doc, "requests") {
        let case_name = name(case);
        let body = bytes(case.get("responses").unwrap());
        let result = responses::convert(&body, &policy(false));
        if let Some((status, kind)) = expected_error(case) {
            let error = result
                .err()
                .unwrap_or_else(|| panic!("{case_name}: accepted"));
            assert_error(case_name, &error, status, &kind);
            if case.get("cere_override").is_none() && kind == "invalid_request_error" {
                let message = case.pointer("/error/message").unwrap().as_str().unwrap();
                assert_eq!(error.message, message, "{case_name}");
            }
            checked += 1;
            continue;
        }
        let converted = result.unwrap_or_else(|e| panic!("{case_name}: {e:?}"));
        // The CERE chat body carries SGLang's converted messages, tools and tool_choice (nulls droppable, text
        // parts joined; without tools CERE sends no tool_choice where SGLang's default is "none").
        let chat: Node = serde_json::from_slice(&converted.chat).unwrap();
        let openai = without_nulls(case.get("openai").unwrap());
        assert_eq!(
            without_nulls(chat.get("messages").unwrap()),
            joined_parts(openai.get("messages").unwrap()),
            "{case_name}: messages"
        );
        assert_eq!(chat.get("tools"), openai.get("tools"), "{case_name}: tools");
        if chat.get("tools").is_some() {
            assert_eq!(
                chat.get("tool_choice"),
                openai.get("tool_choice"),
                "{case_name}"
            );
        } else {
            assert!(chat.get("tool_choice").is_none(), "{case_name}");
        }
        // The CERE chat body passes the chat validator on both backends. Native (M2.7): `top_p` is served (D8),
        // and a body without `temperature` keeps it absent and passes: native samples it with the model's
        // `generation_config.json` defaults (D10, Q4 (a); O24-1 closed).
        for native in [false, true] {
            let converted = responses::convert(&body, &policy(native)).unwrap();
            let parsed: Value = serde_json::from_slice(&converted.chat).unwrap();
            policy(native)
                .validate_bytes(&parsed, &converted.chat)
                .unwrap_or_else(|e| panic!("{case_name} native={native}: {e:?}"));
        }
        // The unary body for the case's engine output ("OK").
        let response = case.get("response").unwrap();
        let pointers = under(&volatile(case), "/response");
        assert_eq!(
            masked(&render(case, &ok_chat()), &pointers),
            masked(&patched(case, response), &pointers),
            "{case_name}: response body"
        );
        checked += 1;
    }
    assert_eq!(checked, 32);
}

/// A native profile that validates greedy combinations only (`sampled: false`): a Responses request without
/// `temperature` is sampled (m27 D10), so it is refused there exactly as a chat request without it, and
/// `temperature: 0` passes (docs/API.md "Sampling"; review ReviewApi P2, resolved by the spec's "as on chat" rule).
#[test]
fn native_greedy_only_profile_needs_temperature_zero() {
    let greedy_only = ChatPolicy {
        sampled: false,
        tools_sampled: false,
        ..policy(true)
    };
    for (body, refused) in [
        (
            json!({"model": "Qwen3.8-Flash-Next", "input": "Hello"}),
            true,
        ),
        (
            json!({"model": "Qwen3.8-Flash-Next", "input": "Hello", "temperature": 0}),
            false,
        ),
    ] {
        let converted =
            responses::convert(&serde_json::to_vec(&body).unwrap(), &policy(true)).unwrap();
        let parsed: Value = serde_json::from_slice(&converted.chat).unwrap();
        assert!(parsed.get("temperature").is_none() == refused, "{parsed}");
        let checked = greedy_only.validate_bytes(&parsed, &converted.chat);
        if refused {
            let error = checked.unwrap_err();
            assert_eq!(
                (error.code, error.param.as_deref()),
                ("unsupported_parameter", Some("temperature"))
            );
        } else {
            checked.unwrap();
        }
    }
}

#[test]
fn chat_body_mirrors_sglang_sampling() {
    // TrackM214 ruling 3: SGLang-backed bodies carry SGLang's effective sampling (max_new_tokens = max_output_tokens
    // - 2, temperature 0.7 and top_p 1.0 when absent); native keeps an absent temperature absent.
    let doc = goldens();
    for case in cases(&doc, "unary").iter().chain(cases(&doc, "streams")) {
        if expected_error(case).is_some() {
            continue;
        }
        let case_name = name(case);
        let request = case.get("responses").unwrap();
        let converted = responses::convert(&bytes(request), &policy(false)).unwrap();
        let chat: Value = serde_json::from_slice(&converted.chat).unwrap();
        let sampling = case.get("sampling_params").unwrap();
        let num = |n: &Node, key: &str| n.get(key).and_then(Node::as_f64);
        assert_eq!(
            chat["temperature"].as_f64(),
            num(sampling, "temperature"),
            "{case_name}"
        );
        assert_eq!(
            chat["top_p"].as_f64(),
            num(sampling, "top_p"),
            "{case_name}"
        );
        if request.get("max_output_tokens").is_some() {
            assert_eq!(
                chat["max_completion_tokens"].as_f64(),
                num(sampling, "max_new_tokens"),
                "{case_name}"
            );
        } else {
            assert!(chat.get("max_completion_tokens").is_none(), "{case_name}");
        }
        let native: Value = serde_json::from_slice(
            &responses::convert(&bytes(request), &policy(true))
                .unwrap()
                .chat,
        )
        .unwrap();
        assert_eq!(
            native.get("temperature").and_then(Value::as_f64),
            num(request, "temperature"),
            "{case_name}"
        );
        assert_eq!(
            native.get("top_p").and_then(Value::as_f64),
            num(request, "top_p")
        );
        assert_eq!(
            native["max_completion_tokens"],
            chat["max_completion_tokens"]
        );
        assert_eq!(
            chat["stream"],
            json!(case.get("chunks").is_some()),
            "{case_name}"
        );
    }
}

#[test]
fn unary_bodies_match_sglang() {
    let doc = goldens();
    let mut checked = 0;
    for case in cases(&doc, "unary") {
        let case_name = name(case);
        let request = bytes(case.get("responses").unwrap());
        if let Some((status, kind)) = expected_error(case) {
            assert_error(
                case_name,
                &responses::convert(&request, &policy(false)).unwrap_err(),
                status,
                &kind,
            );
            continue;
        }
        let chat: Value = serde_json::to_value(case.get("chat").unwrap()).unwrap();
        let pointers = under(&volatile(case), "/response");
        let got = render(case, &chat);
        assert_eq!(
            masked(&got, &pointers),
            masked(case.get("response").unwrap(), &pointers),
            "{case_name}"
        );
        // Ids are SGLang-shaped: resp_/msg_/rs_ + 32 hex, fc_ + 8 hex, call_ + 24 hex.
        let id = got.get("id").unwrap().as_str().unwrap();
        assert!(id.starts_with("resp_") && id.len() == 37, "{id}");
        checked += 1;
    }
    assert_eq!(checked, 7);
}

fn run_stream(case: &Node) -> Vec<Event> {
    let request = bytes(case.get("responses").unwrap());
    let converted = responses::convert(&request, &policy(false)).unwrap();
    assert!(converted.stream);
    let mut stream = ResponsesStream::new(converted.echo, Ids::seeded([9; 16]), 1791656033);
    let mut events = stream.open();
    let Some(Node::Arr(chunks)) = case.get("chunks") else {
        panic!("{}: no chunks", name(case))
    };
    for chunk in chunks {
        events.extend(stream.chunk(&serde_json::to_value(chunk).unwrap()));
    }
    events.extend(stream.complete(None));
    events
}

#[test]
fn stream_events_match_sglang() {
    let doc = goldens();
    let mut checked = 0;
    for case in cases(&doc, "streams") {
        let case_name = name(case);
        if let Some((status, kind)) = expected_error(case) {
            let request = bytes(case.get("responses").unwrap());
            assert_error(
                case_name,
                &responses::convert(&request, &policy(false)).unwrap_err(),
                status,
                &kind,
            );
            continue;
        }
        let events = run_stream(case);
        let got = Node::Arr(
            events
                .iter()
                .map(|e| Node::Arr(vec![Node::Str(e.name.to_owned()), e.data.clone()]))
                .collect(),
        );
        let Some(Node::Arr(want)) = case.get("events") else {
            panic!()
        };
        let mut pointers = under(&volatile(case), "/events");
        pointers.push(format!("/{}/1/response/usage", want.len() - 1));
        assert_eq!(
            masked(&got, &pointers),
            masked(case.get("events").unwrap(), &pointers),
            "{case_name}"
        );
        // One id per output item across its events.
        let mut ids = std::collections::BTreeMap::new();
        for event in &events {
            let index = event.data.get("output_index").and_then(Node::as_u64);
            let id = event
                .data
                .get("item_id")
                .or_else(|| event.data.pointer("/item/id"))
                .and_then(Node::as_str);
            if let (Some(index), Some(id)) = (index, id) {
                assert_eq!(*ids.entry(index).or_insert(id), id, "{case_name}");
            }
        }
        // SSE frames: `event: <type>` then `data: <json>` in the event's key order.
        let frame = events[0].frame();
        assert!(frame.starts_with("event: response.created\ndata: {\"response\":{\"id\":\"resp_"));
        assert!(frame.ends_with("\"sequence_number\":0,\"type\":\"response.created\"}\n\n"));
        checked += 1;
    }
    assert_eq!(checked, 4);
}

#[test]
fn stream_usage_and_failure() {
    let converted = responses::convert(
        br#"{"model":"m","input":"Hi","stream":true}"#,
        &policy(false),
    )
    .unwrap();
    let mut stream = ResponsesStream::new(converted.echo.clone(), Ids::seeded([1; 16]), 5);
    stream.open();
    stream.chunk(
        &json!({"choices": [{"index": 0, "delta": {"content": "Hey"}, "finish_reason": "stop"}]}),
    );
    stream.chunk(&json!({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}));
    let done = stream.complete(None);
    let last = done.last().unwrap();
    assert_eq!(last.name, "response.completed");
    let usage = serde_json::to_value(last.data.pointer("/response/usage").unwrap()).unwrap();
    assert_eq!(
        usage,
        json!({"input_tokens": 4, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
               "output_tokens": 2, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 6})
    );

    let mut stream = ResponsesStream::new(converted.echo, Ids::seeded([2; 16]), 5);
    let opened = stream.open();
    stream.chunk(&json!({"choices": [{"index": 0, "delta": {"content": "Hey"}}]}));
    let failed = stream.fail();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].name, "response.failed");
    let response = failed[0].data.get("response").unwrap();
    assert_eq!(response.get("status").unwrap().as_str(), Some("failed"));
    assert_eq!(response.get("output"), Some(&Node::Arr(vec![])));
    assert_eq!(
        response.get("id"),
        opened[0].data.pointer("/response/id"),
        "the failed response keeps the stream's id"
    );
}

#[test]
fn error_envelope_is_sglangs() {
    let error = ApiError::unsupported_parameter("store", "store: true needs a response store");
    let body =
        String::from_utf8(serde_json::to_vec(&responses::error_body(&error)).unwrap()).unwrap();
    assert_eq!(
        body,
        r#"{"error":{"message":"store: true needs a response store","type":"unsupported_parameter","param":null,"code":400}}"#
    );
    let doc = goldens();
    for route in cases(&doc, "routes") {
        let path = route.get("path").unwrap().as_str().unwrap();
        let id = path
            .trim_start_matches("/v1/responses/")
            .trim_end_matches("/cancel");
        let error = responses::stateless_not_found(id);
        assert_eq!((error.status, error.code), (404, "not_found"));
        assert!(error.message.contains("stateless"), "{}", error.message);
    }
}

#[test]
fn refusals_beyond_the_goldens() {
    let refuse = |body: Value| {
        responses::convert(&serde_json::to_vec(&body).unwrap(), &policy(false)).unwrap_err()
    };
    let base = || json!({"model": "Qwen3.8-Flash-Next", "input": "Hello"});
    // Fields outside the subset are refused, never dropped (as chat).
    let mut body = base();
    body["text"] = json!({"format": {"type": "json_object"}});
    assert_eq!((refuse(body).code, 400), ("unsupported_parameter", 400));
    // Image parts need the M2.13 capability.
    let mut body = base();
    body["input"] =
        json!([{"role": "user", "content": [{"type": "input_image", "image_url": "data:"}]}]);
    assert_eq!(refuse(body).code, "unsupported_parameter");
    // SGLang's headroom of 2 leaves no output below 3.
    let mut body = base();
    body["max_output_tokens"] = json!(2);
    assert_eq!(refuse(body).code, "invalid_value");
    // Reasoning summaries are not rendered.
    let mut body = base();
    body["reasoning"] = json!({"summary": "auto"});
    assert_eq!(refuse(body).code, "unsupported_parameter");
    // A JSON body that is not an object.
    assert_eq!(
        responses::convert(b"[1]", &policy(false)).unwrap_err().code,
        "invalid_json"
    );
}

// ---------------------------------------------------------------------------------------------------------
// Through cered (the mock SGLang answers with each golden's recorded chat completion or chunks)

/// The test-only switch (TrackM214 ruling 1: no shipped config enables the surface).
const ENABLE: (&str, &str) = (
    "allow_cross_scope_prefix_cache = true",
    "allow_cross_scope_prefix_cache = true\napi_compat_responses = true",
);

async fn enabled() -> Harness {
    let mock = MockConfig {
        record_bodies: true,
        ..MockConfig::default()
    };
    Harness::start(mock, &[ENABLE]).await
}

fn case_named<'a>(doc: &'a Node, section: &str, wanted: &str) -> &'a Node {
    cases(doc, section)
        .iter()
        .find(|c| name(c) == wanted)
        .unwrap_or_else(|| panic!("no {section} case {wanted}"))
}

fn post(h: &Harness, body: &Node) -> reqwest::RequestBuilder {
    h.post("/v1/responses", TOKEN_A)
        .header("content-type", "application/json")
        .body(bytes(body))
}

async fn node_of(response: reqwest::Response) -> (u16, Node) {
    let status = response.status().as_u16();
    let body = response.bytes().await.unwrap();
    let node = serde_json::from_slice(&body)
        .unwrap_or_else(|e| panic!("{status}: {e}: {}", String::from_utf8_lossy(&body)));
    (status, node)
}

/// SGLang's envelope, exactly `{"error":{message, type, param: null, code: <status>}}`.
fn assert_envelope(status: u16, body: &Node, want: u16, kind: &str) {
    assert_eq!(status, want, "{body:?}");
    let Some(Node::Obj(error)) = body.get("error") else {
        panic!("no error object: {body:?}")
    };
    let keys: Vec<&str> = error.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["message", "type", "param", "code"], "{body:?}");
    assert_eq!(
        body.pointer("/error/type").and_then(Node::as_str),
        Some(kind)
    );
    assert_eq!(body.pointer("/error/param"), Some(&Node::Null));
    assert_eq!(
        body.pointer("/error/code").and_then(Node::as_u64),
        Some(u64::from(want))
    );
}

/// The golden chunks plus the usage chunk SGLang sends last (cered always asks for it).
fn scripted_stream(case: &Node) -> Scripted {
    let mut chunks: Vec<Value> = match case.get("chunks") {
        Some(Node::Arr(chunks)) => chunks
            .iter()
            .map(|c| serde_json::to_value(c).unwrap())
            .collect(),
        _ => panic!("{}: no chunks", name(case)),
    };
    chunks.push(json!({
        "id": "fixture-rid", "object": "chat.completion.chunk", "created": 1791656033u64,
        "model": "Qwen3.8-Flash-Next", "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17}
    }));
    Scripted {
        response: Value::Null,
        chunks,
    }
}

/// `event: <type>` + `data: <json>` frames (comments and other lines are not Responses frames).
fn frames(text: &str) -> Vec<(String, Node)> {
    assert!(text.ends_with("\n\n"), "trailing partial frame: {text:?}");
    text.split("\n\n")
        .filter(|raw| !raw.is_empty() && !raw.starts_with(':'))
        .map(|raw| {
            let (event, data) = raw
                .strip_prefix("event: ")
                .and_then(|rest| rest.split_once("\ndata: "))
                .unwrap_or_else(|| panic!("event + data frame: {raw:?}"));
            assert!(!data.contains('\n'), "{raw:?}");
            (event.to_owned(), serde_json::from_str(data).unwrap())
        })
        .collect()
}

#[tokio::test]
async fn capability_off_is_todays_404() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let before = h.mock.dispatched_rids().len();
    let unknown = h.post("/v1/no-such-route", TOKEN_A).send().await.unwrap();
    let (want_status, want_body) = (unknown.status(), unknown.bytes().await.unwrap());
    assert_eq!(want_status, 404);
    let body = json!({"model": "Qwen3.8-Flash-Next", "input": "Hello"});
    for request in [
        h.post("/v1/responses", TOKEN_A).json(&body),
        h.get("/v1/responses/resp_fixture0", TOKEN_A),
        h.post("/v1/responses/resp_fixture0/cancel", TOKEN_A),
    ] {
        let r = request.send().await.unwrap();
        assert_eq!(r.status(), want_status);
        assert_eq!(r.bytes().await.unwrap(), want_body);
    }
    assert_eq!(h.mock.dispatched_rids().len(), before);
}

#[tokio::test]
async fn unary_through_cered_matches_sglang() {
    let h = enabled().await;
    let doc = goldens();
    for wanted in ["text", "function_call", "text_then_function_call"] {
        let case = case_named(&doc, "unary", wanted);
        h.mock.set_scripted(Some(Scripted {
            response: serde_json::to_value(case.get("chat").unwrap()).unwrap(),
            chunks: Vec::new(),
        }));
        let r = post(&h, case.get("responses").unwrap())
            .send()
            .await
            .unwrap();
        assert!(r.headers().contains_key("x-cere-request-id"), "{wanted}");
        assert_eq!(r.headers()["content-type"], "application/json", "{wanted}");
        let (status, got) = node_of(r).await;
        assert_eq!(status, 200, "{wanted}: {got:?}");
        let pointers = under(&volatile(case), "/response");
        let want = patched(case, case.get("response").unwrap());
        assert_eq!(
            masked(&got, &pointers),
            masked(&want, &pointers),
            "{wanted}"
        );
        // The backend got the converted chat body with SGLang's effective sampling (TrackM214 ruling 3).
        let sent: Value = serde_json::from_slice(h.mock.received_bodies().last().unwrap()).unwrap();
        let openai = serde_json::to_value(case.get("openai").unwrap()).unwrap();
        assert_eq!(sent["messages"], openai["messages"], "{wanted}");
        assert_eq!(sent["temperature"], json!(0.7), "{wanted}");
        assert_eq!(sent["top_p"], json!(1.0), "{wanted}");
    }
}

#[tokio::test]
async fn stream_through_cered_has_sglangs_events() {
    let h = enabled().await;
    let doc = goldens();
    for wanted in ["text", "function_call", "text_then_function_call"] {
        let case = case_named(&doc, "streams", wanted);
        h.mock.set_scripted(Some(scripted_stream(case)));
        let r = post(&h, case.get("responses").unwrap())
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "{wanted}");
        assert_eq!(r.headers()["content-type"], "text/event-stream", "{wanted}");
        let text = r.text().await.unwrap();
        assert!(
            !text.contains("[DONE]"),
            "{wanted}: SGLang's Responses stream has no [DONE]"
        );
        let got = frames(&text);
        let Some(Node::Arr(want)) = case.get("events") else {
            panic!()
        };
        let names = |events: &[(String, Node)]| -> Vec<String> {
            events.iter().map(|(n, _)| n.clone()).collect()
        };
        let want: Vec<(String, Node)> = want
            .iter()
            .map(|e| {
                let Node::Arr(pair) = e else { panic!() };
                (pair[0].as_str().unwrap().to_owned(), pair[1].clone())
            })
            .collect();
        assert_eq!(names(&got), names(&want), "{wanted}: SSE event order");
        let mut pointers = under(&volatile(case), "/events");
        pointers.push(format!("/{}/1/response/usage", want.len() - 1));
        let as_node = |events: &[(String, Node)]| {
            Node::Arr(
                events
                    .iter()
                    .map(|(n, d)| Node::Arr(vec![Node::Str(n.clone()), d.clone()]))
                    .collect(),
            )
        };
        assert_eq!(
            masked(&as_node(&got), &pointers),
            masked(&as_node(&want), &pointers),
            "{wanted}"
        );
        // The final usage is cered's (the backend's usage chunk).
        let usage = got
            .last()
            .unwrap()
            .1
            .pointer("/response/usage/output_tokens");
        assert_eq!(usage.and_then(Node::as_u64), Some(7), "{wanted}");
    }
}

#[tokio::test]
async fn refusals_use_sglangs_envelope() {
    let h = enabled().await;
    let before = h.mock.dispatched_rids().len();
    let doc = goldens();
    for wanted in [
        "store_true",
        "previous_response_id",
        "background",
        "builtin_tool",
    ] {
        let case = case_named(&doc, "requests", wanted);
        let (status, body) = node_of(
            post(&h, case.get("responses").unwrap())
                .send()
                .await
                .unwrap(),
        )
        .await;
        assert_envelope(status, &body, 400, "unsupported_parameter");
    }
    // Stateless retrieve and cancel (M214S-04/M214S-18).
    for request in [
        h.get("/v1/responses/resp_fixture0", TOKEN_A),
        h.post("/v1/responses/resp_fixture0/cancel", TOKEN_A),
    ] {
        let (status, body) = node_of(request.send().await.unwrap()).await;
        assert_envelope(status, &body, 404, "not_found");
        let message = body
            .pointer("/error/message")
            .and_then(Node::as_str)
            .unwrap();
        assert!(message.contains("stateless"), "{message}");
    }
    // Malformed JSON, an unserved model and a missing token, in the same envelope.
    let r = h
        .post("/v1/responses", TOKEN_A)
        .header("content-type", "application/json")
        .body("{")
        .send()
        .await
        .unwrap();
    let (status, body) = node_of(r).await;
    assert_envelope(status, &body, 400, "invalid_json");
    let unserved = json!({"model": "no-such-model", "input": "Hello"});
    let r = h
        .post("/v1/responses", TOKEN_A)
        .json(&unserved)
        .send()
        .await
        .unwrap();
    let (status, body) = node_of(r).await;
    assert_eq!(status, 404, "{body:?}");
    assert_envelope(
        status,
        &body,
        404,
        body.pointer("/error/type").and_then(Node::as_str).unwrap(),
    );
    let r = h
        .http
        .post(h.url("/v1/responses"))
        .json(&json!({"model": "Qwen3.8-Flash-Next", "input": "Hello"}))
        .send()
        .await
        .unwrap();
    let (status, body) = node_of(r).await;
    assert_envelope(status, &body, 401, "unauthorized");
    assert_eq!(h.mock.dispatched_rids().len(), before);
}

/// M2.12 wiring of D4 (the harness profile declares `thinking_toggle`): `reasoning.effort` low/medium/xhigh
/// reaches the backend as `reasoning_effort` with thinking on (`chat_template_kwargs.enable_thinking: true`, the
/// key SGLang sets next to its `thinking` alias); other efforts stay 400 `invalid_value` (m212 D13).
#[tokio::test]
async fn reasoning_effort_reaches_the_backend_on_a_thinking_profile() {
    let h = enabled().await;
    let doc = goldens();
    h.mock.set_scripted(Some(Scripted {
        response: serde_json::to_value(case_named(&doc, "unary", "text").get("chat").unwrap())
            .unwrap(),
        chunks: Vec::new(),
    }));
    for wanted in [
        "reasoning_effort_low",
        "reasoning_effort_medium",
        "reasoning_effort_xhigh",
    ] {
        let case = case_named(&doc, "requests", wanted);
        let (status, body) = node_of(
            post(&h, case.get("responses").unwrap())
                .send()
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(status, 200, "{wanted}: {body:?}");
        let sent: Value = serde_json::from_slice(h.mock.received_bodies().last().unwrap()).unwrap();
        let openai = serde_json::to_value(case.get("openai").unwrap()).unwrap();
        assert_eq!(
            sent["reasoning_effort"], openai["reasoning_effort"],
            "{wanted}: {sent}"
        );
        assert_eq!(
            sent["chat_template_kwargs"],
            json!({"enable_thinking": true}),
            "{wanted}: {sent}"
        );
        assert_eq!(
            openai["chat_template_kwargs"]["enable_thinking"],
            json!(true),
            "{wanted}"
        );
    }
    let before = h.mock.dispatched_rids().len();
    for wanted in [
        "reasoning_effort_none",
        "reasoning_effort_minimal",
        "reasoning_effort_high",
        "reasoning_effort_max",
    ] {
        let case = case_named(&doc, "requests", wanted);
        let (status, body) = node_of(
            post(&h, case.get("responses").unwrap())
                .send()
                .await
                .unwrap(),
        )
        .await;
        assert_envelope(status, &body, 400, "invalid_value");
    }
    assert_eq!(h.mock.dispatched_rids().len(), before);
}

#[tokio::test]
async fn idempotency_is_per_surface() {
    let h = enabled().await;
    let doc = goldens();
    let text = case_named(&doc, "unary", "text");
    h.mock.set_scripted(Some(Scripted {
        response: serde_json::to_value(text.get("chat").unwrap()).unwrap(),
        chunks: Vec::new(),
    }));
    let request = text.get("responses").unwrap();
    // A Responses replay returns the Responses body, byte for byte, without a second dispatch.
    let first = post(&h, request)
        .header("Idempotency-Key", "r-1")
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), 200);
    let rid = first.headers()["x-cere-request-id"].clone();
    let first = first.bytes().await.unwrap();
    let dispatched = h.mock.dispatched_rids().len();
    let replay = post(&h, request)
        .header("Idempotency-Key", "r-1")
        .send()
        .await
        .unwrap();
    assert_eq!(replay.status(), 200);
    assert_eq!(replay.headers()["x-cere-request-id"], rid);
    assert_eq!(replay.bytes().await.unwrap(), first);
    assert_eq!(h.mock.dispatched_rids().len(), dispatched);
    let body: Node = serde_json::from_slice(&first).unwrap();
    assert_eq!(body.get("object").and_then(Node::as_str), Some("response"));
    // The key of a Responses request reused on chat, and of a chat request reused on Responses, conflicts.
    let chat = h
        .chat(&support::chat_body(false))
        .header("Idempotency-Key", "r-1")
        .send()
        .await
        .unwrap();
    assert_eq!(chat.status(), 409);
    let chat: Value = chat.json().await.unwrap();
    assert_eq!(chat["error"]["code"], "idempotency_conflict", "{chat}");
    h.mock.set_scripted(None);
    let chat = h
        .chat(&support::chat_body(false))
        .header("Idempotency-Key", "c-1")
        .send()
        .await
        .unwrap();
    assert_eq!(chat.status(), 200);
    let r = post(&h, request)
        .header("Idempotency-Key", "c-1")
        .send()
        .await
        .unwrap();
    let (status, body) = node_of(r).await;
    assert_envelope(status, &body, 409, "idempotency_conflict");
    // A streamed Responses request is not replayable, and its events cannot be resumed (non-chat surface).
    let stream = case_named(&doc, "streams", "text");
    h.mock.set_scripted(Some(scripted_stream(stream)));
    let streamed = post(&h, stream.get("responses").unwrap())
        .header("Idempotency-Key", "s-1")
        .send()
        .await
        .unwrap();
    assert_eq!(streamed.status(), 200);
    let rid = streamed.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    streamed.text().await.unwrap();
    let again = post(&h, stream.get("responses").unwrap())
        .header("Idempotency-Key", "s-1")
        .send()
        .await
        .unwrap();
    let (status, body) = node_of(again).await;
    assert_envelope(status, &body, 409, "stream_not_replayable");
    let resume = h
        .get(&format!("/v1/cere/requests/{rid}/events"), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(resume.status(), 409);
    let resume: Value = resume.json().await.unwrap();
    assert_eq!(resume["error"]["code"], "stream_not_replayable", "{resume}");
}

/// The plumbing hook (`SurfaceRender`/`SurfaceStream`): the relay holds the finish chunk back and hands it to
/// `complete`, so the SSE bytes still carry SGLang's event order.
#[test]
fn surface_hooks_render_the_held_finish_chunk() {
    use cere_runtime::surface::{ErrorView, SurfaceRender};
    let doc = goldens();
    for wanted in ["text", "function_call", "text_then_function_call"] {
        let case = case_named(&doc, "streams", wanted);
        let converted =
            responses::convert(&bytes(case.get("responses").unwrap()), &policy(false)).unwrap();
        let renderer = responses::Responses::new(converted.echo).unwrap();
        assert_eq!(renderer.name(), "responses");
        let mut stream = renderer.stream();
        let mut sse = stream.open(None);
        let Some(Node::Arr(chunks)) = case.get("chunks") else {
            panic!()
        };
        let (finish, passthrough) = chunks.split_last().unwrap();
        for chunk in passthrough {
            sse.extend(stream.chunk(&serde_json::to_value(chunk).unwrap()));
        }
        let usage = json!({"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17});
        sse.extend(stream.complete(&serde_json::to_value(finish).unwrap(), Some(&usage)));
        let got = frames(&String::from_utf8(sse).unwrap());
        let Some(Node::Arr(want)) = case.get("events") else {
            panic!()
        };
        let want: Vec<&str> = want
            .iter()
            .map(|e| e.pointer("/0").and_then(Node::as_str).unwrap())
            .collect();
        let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, want, "{wanted}");
        let last = &got.last().unwrap().1;
        assert_eq!(
            last.pointer("/response/usage/output_tokens")
                .and_then(Node::as_u64),
            Some(7)
        );
        let error = ErrorView {
            status: 503,
            kind: "server_error",
            code: "backend_unavailable",
            message: "gone",
            param: None,
        };
        let body: Node = serde_json::from_slice(&renderer.error_body(&error)).unwrap();
        assert_envelope(503, &body, 503, "backend_unavailable");
        let failed = frames(&String::from_utf8(renderer.stream().fail(&error)).unwrap());
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0, "response.failed");
    }
}

/// ApiFit AF-05 (api-fit design note): `instructions` and a leading system item merge as before (SGLang); a
/// `system` or `developer` input item after a turn reaches chat validation at its position and renders in place as a
/// marked user message (AF-L1), on both backends.
#[test]
fn af05_mid_conversation_system_items_render_in_place() {
    let body = json!({
        "model": "Qwen3.8-Flash-Next",
        "instructions": "Be terse.",
        "input": [
            {"role": "system", "content": "Leading."},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "developer", "content": "r1"},
            {"role": "user", "content": "n"},
            {"role": "system", "content": "r2"},
        ],
    });
    for native in [false, true] {
        let converted =
            responses::convert(&serde_json::to_vec(&body).unwrap(), &policy(native)).unwrap();
        let parsed: Value = serde_json::from_slice(&converted.chat).unwrap();
        let request = policy(native)
            .validate_bytes(&parsed, &converted.chat)
            .unwrap_or_else(|e| panic!("native={native}: {e:?}"));
        let messages = request.normalized()["messages"].clone();
        let pairs: Vec<(&str, &str)> = messages
            .as_array()
            .unwrap()
            .iter()
            .map(|m| (m["role"].as_str().unwrap(), m["content"].as_str().unwrap()))
            .collect();
        assert_eq!(
            pairs,
            [
                ("system", "Be terse.\n\nLeading."),
                ("user", "q"),
                ("assistant", "a"),
                ("user", "<system-message>\nr1\n</system-message>"),
                ("user", "n"),
                ("user", "<system-message>\nr2\n</system-message>"),
            ],
            "native={native}"
        );
    }
}
