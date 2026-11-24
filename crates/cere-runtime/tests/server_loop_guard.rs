//! loop-guard design note: the native loop guard through the real `cered` over the CPU toy executor
//! ([`native_server`]). The stub sidecar encodes a user message as `[1]` plus its UTF-8 bytes and the toy's next
//! token depends only on the last token and the position, so `TEXT`'s output IDs are known; a tokenizer that
//! decodes each of them to the same piece makes the toy's output a flat loop, the default tokenizer (`[id]` pieces
//! through `distinct_tokenizer`, or the fixture's) an output that never repeats.
mod native_server;
mod native_support;
mod support;

use native_server::{Case, Native, TOKEN};
use native_support::{toy_tokens, FIXTURE_VOCAB};
use serde_json::{json, Value};
use support::{check_stream, http, Event, SseReader, MODEL};

const SENTINEL: &str = "cere-native-sentinel-prompt";
/// `server_native`'s `TEXT`: 716 prompt tokens ending in `1`; no stop ID within 64 output tokens.
const TEXT: (usize, char) = (716, '1');
/// One loop unit: 17 bytes whose smallest period is 17.
const PIECE: &str = "loop-check-again;";
/// LG-D1 fires once 128 bytes of copies end the text: at the 8th piece (136 bytes; 7 pieces are 119).
const STOP_AT: usize = 8;

fn guard() -> Value {
    json!({"min_period_bytes": 16, "max_period_bytes": 64, "min_repeats": 4, "min_span_bytes": 128})
}

fn prompt((len, last): (usize, char)) -> String {
    let mut s = SENTINEL.to_owned();
    while s.len() < len - 2 {
        s.push('x');
    }
    s.push(last);
    s
}

fn body(max_tokens: u64, stream: bool) -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt(TEXT)}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    })
}

/// [`body`] streamed, with the usage chunk (the tests' locals named `body` shadow the function).
fn body_stream(max_tokens: u64) -> Value {
    let mut b = body(max_tokens, true);
    b["stream_options"] = json!({"include_usage": true});
    b
}

/// What the toy commits for `TEXT` within `n` tokens.
fn toy_output(n: usize) -> Vec<u32> {
    let ids = toy_tokens(TEXT.1 as u32, TEXT.0 as u64, n, FIXTURE_VOCAB);
    assert!(!ids.iter().any(|id| [6, 7].contains(id)), "a stop in reach");
    ids
}

/// A ByteLevel `tokenizer.json` in which every toy ID decodes to `[id]` except the first `n` output IDs of
/// `TEXT`, which all decode to [`PIECE`].
fn looping_tokenizer(n: usize) -> Vec<u8> {
    let looped = toy_output(n);
    let added: Vec<Value> = (0..FIXTURE_VOCAB)
        .map(|id| {
            let content = if looped.contains(&id) {
                PIECE.to_owned()
            } else {
                format!("[{id}]")
            };
            json!({"id": id, "content": content, "special": false})
        })
        .collect();
    serde_json::to_vec(&json!({
        "version": "1.0",
        "model": {"type": "BPE", "vocab": {}},
        "added_tokens": added,
        "decoder": {"type": "ByteLevel"},
    }))
    .unwrap()
}

fn case(loop_guard: Value, looping: bool) -> Case {
    Case {
        tokenizer: looping.then(|| looping_tokenizer(64)),
        loop_guard,
        compat: &["api_compat_anthropic"],
        ..Case::matching()
    }
}

async fn unary(native: &Native, body: &Value) -> (u16, Value) {
    let response = native.chat(body).send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

/// The streamed content and the terminal chunk.
async fn streamed(native: &Native, body: &Value) -> (String, Value) {
    let response = native.chat(body).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let events = SseReader::new(response).rest().await;
    let terminal = check_stream(&events);
    let text = events
        .iter()
        .filter_map(|e| match e {
            Event::Data(v) => v
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str),
            Event::Done => None,
        })
        .collect();
    (text, terminal)
}

async fn stats(native: &Native) -> Value {
    let (status, body) = native.get("/v1/cere/stats").await;
    assert_eq!(status, 200, "{body}");
    body
}

/// LG-02: a flat loop finishes with `repetition` (`matched_stop` null) at the first token where the bar holds,
/// unary and SSE alike, keeping the text and tokens up to there; the request completes; the Anthropic surface
/// reports `end_turn`; the stats count the three stops.
#[tokio::test]
async fn a_looping_output_finishes_with_repetition() {
    let native = Native::ready(case(guard(), true)).await;
    let (status, body) = unary(&native, &body(64, false)).await;
    assert_eq!(status, 200, "{body}");
    let choice = &body["choices"][0];
    assert_eq!(
        (&choice["finish_reason"], &choice["matched_stop"]),
        (&json!("repetition"), &Value::Null),
        "{body}"
    );
    assert_eq!(
        choice["message"]["content"],
        PIECE.repeat(STOP_AT),
        "{body}"
    );
    assert_eq!(body["usage"]["completion_tokens"], STOP_AT, "{body}");
    assert_eq!(body["cere"]["state"], "completed", "{body}");

    let sse = body_stream(64);
    let (text, terminal) = streamed(&native, &sse).await;
    assert_eq!(text, PIECE.repeat(STOP_AT));
    assert_eq!(
        terminal["choices"][0]["finish_reason"], "repetition",
        "{terminal}"
    );
    assert_eq!(
        terminal["choices"][0]["matched_stop"],
        Value::Null,
        "{terminal}"
    );
    assert_eq!(terminal["cere"]["state"], "completed", "{terminal}");

    let anthropic = json!({"model": MODEL, "max_tokens": 64, "temperature": 0,
                           "messages": [{"role": "user", "content": prompt(TEXT)}]});
    let response = http()
        .post(native.cered.url("/v1/messages"))
        .bearer_auth(TOKEN)
        .json(&anthropic)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let message: Value = response.json().await.unwrap();
    assert_eq!(message["stop_reason"], "end_turn", "{message}");
    assert_eq!(
        message["content"],
        json!([{"type": "text", "text": PIECE.repeat(STOP_AT)}]),
        "{message}"
    );
    assert_eq!(message["usage"]["output_tokens"], STOP_AT, "{message}");

    assert_eq!(
        stats(&native).await["native_loop_guard"],
        json!({"repetition_stops": 3, "tool_repeat_refusals": 0})
    );
}

/// LG-02: without the table (every shipped profile) the same loop runs to its budget and the stats carry no
/// `native_loop_guard`.
#[tokio::test]
async fn without_the_table_a_loop_runs_to_its_budget() {
    let native = Native::ready(case(Value::Null, true)).await;
    let (status, body) = unary(&native, &body(64, false)).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["choices"][0]["finish_reason"], "length", "{body}");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        PIECE.repeat(64),
        "{body}"
    );
    let stats = stats(&native).await;
    assert!(stats.get("native_loop_guard").is_none(), "{stats}");
}

/// LG-02: an output that does not repeat is the same with the guard on as off (content, finish, usage; unary and
/// SSE), and nothing is counted.
#[tokio::test]
async fn a_non_looping_output_is_unchanged_by_the_guard() {
    let mut outputs = Vec::new();
    for loop_guard in [Value::Null, guard(), json!({})] {
        let native = Native::ready(case(loop_guard.clone(), false)).await;
        let (status, body) = unary(&native, &body(48, false)).await;
        assert_eq!(status, 200, "{body}");
        let sse = body_stream(48);
        let (text, terminal) = streamed(&native, &sse).await;
        outputs.push((
            body["choices"].clone(),
            body["usage"]["completion_tokens"].clone(),
            text,
            terminal["choices"].clone(),
        ));
        if !loop_guard.is_null() {
            assert_eq!(
                stats(&native).await["native_loop_guard"],
                json!({"repetition_stops": 0, "tool_repeat_refusals": 0})
            );
        }
    }
    assert_eq!(outputs[0].0[0]["finish_reason"], "length");
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(outputs[0], outputs[2]);
}

const ORDINALS: [&str; 10] = [
    "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth",
];

/// The race `qwen2` conversation: `pairs` turns of the same Bash call (its `description` relabelled) with the same
/// result.
fn tool_loop(pairs: usize) -> Value {
    let mut messages = vec![json!({"role": "user", "content": prompt(TEXT)})];
    for (k, label) in ORDINALS.iter().take(pairs).enumerate() {
        let id = format!("call_{k:024}");
        let arguments =
            json!({"command": "python3 -m pytest -q", "description": format!("{label} check")});
        messages.push(
            json!({"role": "assistant", "content": null, "tool_calls": [{
                "id": id, "type": "function",
                "function": {"name": "Bash", "arguments": arguments.to_string()},
            }]}),
        );
        messages.push(json!({"role": "tool", "tool_call_id": id, "content": "51 passed in 0.12s"}));
    }
    json!({
        "model": MODEL,
        "messages": messages,
        "tools": [{"type": "function", "function": {
            "name": "Bash",
            "description": "Run a command",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string"}, "description": {"type": "string"}}},
        }}],
        "max_tokens": 4,
        "temperature": 0,
    })
}

/// LG-02: `tool_repeat_run` identical call -> result pairs ending the conversation are refused (400
/// `loop_detected`) before any encode or submission; one fewer is served; without the table both are served.
#[tokio::test]
async fn repeated_tool_calls_are_refused_with_loop_detected() {
    let native = Native::ready(case(json!({"tool_repeat_run": 10}), false)).await;
    let submissions = native.submissions().await;
    let (status, body) = unary(&native, &tool_loop(10)).await;
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["error"]["code"], "loop_detected", "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    assert_eq!(body["error"]["param"], "messages", "{body}");
    assert_eq!(native.submissions().await, submissions);
    let (status, body) = unary(&native, &tool_loop(9)).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        stats(&native).await["native_loop_guard"],
        json!({"repetition_stops": 0, "tool_repeat_refusals": 1})
    );

    let off = Native::ready(case(Value::Null, false)).await;
    let (status, body) = unary(&off, &tool_loop(10)).await;
    assert_eq!(status, 200, "{body}");
}
