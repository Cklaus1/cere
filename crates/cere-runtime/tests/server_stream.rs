//! Streaming relay: slow consumers (L5), usage-chunk hiding, cancel endpoint, client disconnect and an
//! unacknowledged abort (L6); per-index choices (M2.6 D11).
mod native_server;
mod native_support;
mod support;

use std::time::Duration;

use cere_mock_sglang::{Fault, MockConfig};
use serde_json::{json, Value};
use support::*;

fn slow(token_ms: u64, tokens: u32) -> MockConfig {
    MockConfig {
        token_latency: Duration::from_millis(token_ms),
        natural_output_tokens: tokens,
        ..MockConfig::default()
    }
}

fn long_stream(tokens: u64) -> Value {
    let mut body = chat_body(true);
    body["max_completion_tokens"] = tokens.into();
    body
}

fn aborted_rids(h: &Harness) -> Vec<String> {
    h.mock.abort_requests().into_iter().map(|a| a.rid).collect()
}

async fn open_stream(h: &Harness, body: &Value) -> (String, SseReader) {
    let response = h.chat(body).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    (rid, SseReader::new(response))
}

#[tokio::test]
async fn l5_slow_consumer_is_cancelled_for_backpressure() {
    let h = Harness::start_socket(
        slow(1, 1_000_000),
        &[
            ("stream_buffer_events = 256", "stream_buffer_events = 1"),
            ("stall_timeout_ms = 30000", "stall_timeout_ms = 200"),
        ],
        test_tuning(),
        Some(4096),
    )
    .await;
    let (head, mut socket, mut body) = raw_chat(&h.base, &long_stream(32_768)).await;
    assert!(head.starts_with("HTTP/1.0 200"), "{head}");
    let rid = head
        .lines()
        .find_map(|l| l.strip_prefix("x-cere-request-id: "))
        .expect("request id header")
        .to_owned();
    // Do not read: the socket buffers and the one-event channel fill, the relay stalls and gives up.
    let record = h.terminal(&rid).await;
    assert_eq!(record["state"], "cancelled", "{record}");
    assert_eq!(record["reason"], "client_backpressure");
    assert!(
        record["timings"]["client_backpressure_ms"]
            .as_u64()
            .unwrap()
            >= 200
    );
    assert_eq!(aborted_rids(&h), vec![rid.clone()]);

    // Reading late still yields a well-formed stream with exactly one terminal.
    tokio::io::AsyncReadExt::read_to_end(&mut socket, &mut body)
        .await
        .unwrap();
    let events = parse_events(std::str::from_utf8(&body).unwrap());
    let terminal = check_stream(&events);
    assert_eq!(terminal["cere"]["state"], "cancelled");
    assert_eq!(terminal["cere"]["reason"], "client_backpressure");
    assert_eq!(terminal["error"]["code"], "client_backpressure");
    for _ in 0..100 {
        if h.app.admission_snapshot().inflight == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the slot was not released after the backend stream ended");
}

/// Review finding 6: a relay blocked on a full client buffer still honours a cancel (endpoint, shutdown,
/// deadline) at once: the backend gets its abort, the request settles without waiting out `stall_timeout_ms`.
#[tokio::test]
async fn l6_cancel_while_blocked_on_a_full_client_buffer() {
    let h = Harness::start_socket(
        slow(1, 1_000_000),
        &[("stream_buffer_events = 256", "stream_buffer_events = 1")],
        test_tuning(),
        Some(4096),
    )
    .await;
    let (head, mut socket, mut body) = raw_chat(&h.base, &long_stream(32_768)).await;
    assert!(head.starts_with("HTTP/1.0 200"), "{head}");
    let rid = head
        .lines()
        .find_map(|l| l.strip_prefix("x-cere-request-id: "))
        .expect("request id header")
        .to_owned();
    // Do not read: at 1 ms per token the socket buffers and the one-event channel fill within milliseconds and
    // the relay blocks (stall bound 30 s).
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(h.mock.in_flight(), 1);
    let cancelled_at = std::time::Instant::now();
    let cancel = h
        .post(&format!("/v1/cere/requests/{rid}/cancel"), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(cancel.status(), 202);
    let record = h.terminal(&rid).await;
    assert!(
        cancelled_at.elapsed() < Duration::from_secs(3),
        "settled {:?} after the cancel",
        cancelled_at.elapsed()
    );
    assert_eq!(record["state"], "cancelled", "{record}");
    assert_eq!(record["reason"], "cancel_requested");
    assert_eq!(aborted_rids(&h), vec![rid.clone()]);

    tokio::io::AsyncReadExt::read_to_end(&mut socket, &mut body)
        .await
        .unwrap();
    let events = parse_events(std::str::from_utf8(&body).unwrap());
    let terminal = check_stream(&events);
    assert_eq!(terminal["cere"]["state"], "cancelled");
    assert_eq!(terminal["cere"]["reason"], "cancel_requested");
}

/// Review finding 6, L8: SIGTERM while the relay is blocked on a slow reader still aborts the rid and settles the
/// request through its executor (a clean stop, no forced terminal).
#[tokio::test]
async fn l8_shutdown_while_blocked_on_a_full_client_buffer_aborts() {
    let mut h = Harness::start_socket(
        slow(1, 1_000_000),
        &[
            ("stream_buffer_events = 256", "stream_buffer_events = 1"),
            ("shutdown_drain_ms = 30000", "shutdown_drain_ms = 300"),
        ],
        test_tuning(),
        Some(4096),
    )
    .await;
    let (head, mut socket, mut body) = raw_chat(&h.base, &long_stream(32_768)).await;
    let rid = head
        .lines()
        .find_map(|l| l.strip_prefix("x-cere-request-id: "))
        .expect("request id header")
        .to_owned();
    tokio::time::sleep(Duration::from_millis(500)).await;
    // The client reads only after the shutdown's settle bound (drain 300 ms + ack 400 ms + 1 s grace) has
    // passed, so its connection can close and the HTTP server can stop.
    let reader = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(2500)).await;
        tokio::io::AsyncReadExt::read_to_end(&mut socket, &mut body)
            .await
            .unwrap();
        body
    });
    assert!(h.stop().await, "shutdown was forced");
    let events = parse_events(std::str::from_utf8(&reader.await.unwrap()).unwrap());
    assert_eq!(check_stream(&events)["cere"]["reason"], "shutdown");
    assert_eq!(aborted_rids(&h), vec![rid.clone()]);
    let records = journal(h.state_dir.path());
    assert!(non_terminal(&records).is_empty());
    let terminal = records
        .iter()
        .find(|r| r["type"] == "terminal" && r["rid"] == rid.as_str())
        .expect("terminal record");
    assert_eq!(terminal["state"], "cancelled", "{terminal}");
    assert_eq!(terminal["reason"], "shutdown", "{terminal}");
}

#[tokio::test]
async fn l5_usage_chunk_only_when_requested() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    for include in [None, Some(false)] {
        let mut body = chat_body(true);
        if let Some(include) = include {
            body["stream_options"] = json!({"include_usage": include});
        }
        let (_, mut reader) = open_stream(&h, &body).await;
        let events = reader.rest().await;
        let finish = check_stream(&events);
        assert_eq!(finish["cere"]["state"], "completed");
        assert_eq!(
            &events[events.len() - 2].data()["cere"],
            &finish["cere"],
            "finish is last before [DONE]"
        );
        assert!(
            events.iter().all(|e| match e {
                Event::Data(d) => d["choices"] != json!([]) || d["cere"]["event"] == "accepted",
                Event::Done => true,
            }),
            "no usage chunk: {events:?}"
        );
    }
}

#[tokio::test]
async fn l6_cancel_endpoint_mid_stream() {
    let h = Harness::start(slow(40, 100), &[]).await;
    let (rid, mut reader) = open_stream(&h, &long_stream(100)).await;
    let accepted = reader.next().await.unwrap();
    assert_eq!(accepted.data()["cere"]["request_id"], rid.as_str());
    // Wait for a token so the request is running on the backend.
    let token = reader.next().await.unwrap();
    assert!(token.data()["cere"]["seq"].as_u64().unwrap() >= 1);

    let cancel = h
        .post(&format!("/v1/cere/requests/{rid}/cancel"), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(cancel.status(), 202);
    assert_eq!(
        cancel.json::<Value>().await.unwrap(),
        json!({"request_id": rid, "state": "cancel_pending"})
    );
    let mut events = vec![accepted, token];
    events.extend(reader.rest().await);
    let terminal = check_stream(&events);
    assert_eq!(terminal["cere"]["state"], "cancelled");
    assert_eq!(terminal["cere"]["reason"], "cancel_requested");
    assert_eq!(
        aborted_rids(&h),
        vec![rid.clone()],
        "exactly this rid was aborted"
    );
    let record = h.terminal(&rid).await;
    assert_eq!(record["state"], "cancelled");
    assert_eq!(record["reason"], "cancel_requested");

    let again = h
        .post(&format!("/v1/cere/requests/{rid}/cancel"), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(again.status(), 200);
    assert_eq!(again.json::<Value>().await.unwrap()["state"], "cancelled");
    assert_eq!(
        aborted_rids(&h).len(),
        1,
        "a repeated cancel sends no second abort"
    );
}

#[tokio::test]
async fn l6_client_disconnect_cancels_and_aborts() {
    let h = Harness::start(slow(40, 100), &[]).await;
    let (rid, mut reader) = open_stream(&h, &long_stream(100)).await;
    reader.next().await.unwrap();
    reader.next().await.unwrap();
    drop(reader);
    let record = h.terminal(&rid).await;
    assert_eq!(record["state"], "cancelled");
    assert_eq!(record["reason"], "client_disconnected");
    assert_eq!(aborted_rids(&h), vec![rid]);
}

/// An abort the backend ignores fails the request `cancel_unacknowledged` once the ack bound runs out, and the
/// slot stays charged while the backend still generates: a follow-up request waits in the queue until the ignored
/// generation ends. The generation is held at its finish (`MockHandle::hold_finish`) until the follow-up is seen
/// queued, so neither outcome races its natural end on a loaded host.
#[tokio::test]
async fn l6_unacknowledged_abort_keeps_the_slot_charged() {
    let h = Harness::start_tuned(
        slow(40, 30),
        &[
            ("max_backend_inflight = 2", "max_backend_inflight = 1"),
            ("per_scope_inflight = 2", "per_scope_inflight = 1"),
        ],
        short_ack(test_tuning()),
    )
    .await;
    h.mock.set_fault(Fault::IgnoreAbort);
    h.mock.hold_finish(true);
    let (rid, mut reader) = open_stream(&h, &long_stream(30)).await;
    let mut events = vec![reader.next().await.unwrap(), reader.next().await.unwrap()];
    let cancel = h
        .post(&format!("/v1/cere/requests/{rid}/cancel"), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(cancel.status(), 202);
    events.extend(reader.rest().await);
    let terminal = check_stream(&events);
    assert_eq!(terminal["cere"]["state"], "failed");
    assert_eq!(terminal["cere"]["reason"], "cancel_unacknowledged");
    let record = h.terminal(&rid).await;
    assert_eq!(record["state"], "failed");
    assert_eq!(record["reason"], "cancel_unacknowledged");

    // The backend still generates: the slot stays charged, so a follow-up request waits in the queue.
    assert_eq!(h.mock.in_flight(), 1);
    assert_eq!(h.app.admission_snapshot().inflight, 1);
    h.mock.set_fault(Fault::None);
    let dispatched = h.mock.dispatched_rids().len();
    let follow = tokio::spawn(h.chat(&chat_body(false)).send());
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while h.app.admission_snapshot().queued == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the follow-up never queued"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(
        h.mock.dispatched_rids().len(),
        dispatched,
        "the follow-up reached the backend beside the ignored stream"
    );
    h.mock.hold_finish(false);
    let follow = follow.await.unwrap().unwrap();
    assert_eq!(follow.status(), 200);
    assert_eq!(h.mock.in_flight(), 0);
    assert_eq!(h.mock.max_in_flight(), 1);
}

/// Review finding 12: a slow `/abort_request` does not stretch the cancel-acknowledgement bound. The ack window
/// runs from the cancel; the executor keeps watching the backend while the abort call is in flight. The abort call
/// is held open (`MockHandle::hold_aborts`) and the ignored generation at its finish until the terminal arrived,
/// so only the ack bound can end the request, whatever the host's speed.
#[tokio::test]
async fn l6_slow_abort_call_does_not_stretch_the_ack_bound() {
    let h = Harness::start_tuned(slow(40, 100), &[], short_ack(test_tuning())).await;
    h.mock.set_fault(Fault::IgnoreAbort);
    h.mock.hold_finish(true);
    h.mock.hold_aborts(true);
    let (rid, mut reader) = open_stream(&h, &long_stream(100)).await;
    let mut events = vec![reader.next().await.unwrap(), reader.next().await.unwrap()];
    let cancel = h
        .post(&format!("/v1/cere/requests/{rid}/cancel"), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(cancel.status(), 202);
    events.extend(reader.rest().await);
    let terminal = check_stream(&events);
    assert_eq!(terminal["cere"]["state"], "failed");
    assert_eq!(terminal["cere"]["reason"], "cancel_unacknowledged");
    // The held abort is still delivered, for exactly this rid.
    let held = || -> Vec<String> { h.mock.held_aborts().into_iter().map(|a| a.rid).collect() };
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while held().is_empty() {
        assert!(
            std::time::Instant::now() < deadline,
            "abort never delivered"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(held(), vec![rid]);
    h.mock.hold_aborts(false);
    h.mock.hold_finish(false);
}

/// m1close R6/Q12: the SSE error event's `cere.usage` carries the backend's counts with the status API's
/// values and null rules, without `stream_options.include_usage`. A clamp fault ends the stream at `length`
/// below the budget (`backend_clamped_output`) with a usage chunk.
#[tokio::test]
async fn r6_error_event_carries_the_status_usage() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    h.mock.set_fault(Fault::ClampOutput { tokens: 3 });
    let (rid, mut reader) = open_stream(&h, &long_stream(8)).await;
    let events = reader.rest().await;
    let terminal = check_stream(&events);
    assert_eq!(terminal["cere"]["reason"], "backend_clamped_output");
    let usage = &terminal["cere"]["usage"];
    assert!(usage["prompt_tokens"].is_u64(), "{terminal}");
    assert_eq!(usage["completion_tokens"], 3, "{terminal}");
    let (_, status) = h.status(&rid, TOKEN_A).await;
    assert_eq!(status["state"], "failed");
    assert_eq!(*usage, status["usage"], "{status}");
}

/// m1close R4/Q9: SGLang's scheduler rejects an over-long prompt after the 200 (an in-stream error object,
/// code 400, "... exceeds the maximum allowed length"). The client already has its 200, so it gets the SSE
/// terminal error event `context_overflow` and `[DONE]`; the status API records the same terminal as a
/// pre-stream HTTP 400 context overflow.
#[tokio::test]
async fn r4_in_stream_max_allowed_length_is_context_overflow() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    h.mock.set_fault(Fault::StreamError {
        code: 400,
        message: "Input length (199994 tokens) exceeds the maximum allowed length (199994 tokens)."
            .to_owned(),
    });
    let (rid, mut reader) = open_stream(&h, &long_stream(1)).await;
    let events = reader.rest().await;
    let terminal = check_stream(&events);
    assert_eq!(terminal["error"]["code"], "context_overflow", "{terminal}");
    assert!(
        terminal["error"]["message"]
            .as_str()
            .unwrap()
            .contains("exceeds the maximum allowed length"),
        "{terminal}"
    );
    assert_eq!(terminal["cere"]["state"], "failed");
    assert_eq!(terminal["cere"]["reason"], "context_overflow");
    let record = h.terminal(&rid).await;
    assert_eq!(
        (&record["state"], &record["reason"]),
        (&json!("failed"), &json!("context_overflow")),
        "{record}"
    );

    // The tokenizer form (HTTP 400 before any byte: prompt + budget over the mock's 200000-token context)
    // records the same terminal.
    h.mock.set_fault(Fault::None);
    let mut long = long_stream(8);
    long["messages"][0]["content"] = "w ".repeat(200_000).into();
    let response = h.chat(&long).send().await.unwrap();
    assert_eq!(response.status(), 400);
    let early = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], "context_overflow", "{body}");
    let early = h.terminal(&early).await;
    assert_eq!(
        (&early["state"], &early["reason"]),
        (&record["state"], &record["reason"]),
        "{early}"
    );
}

/// C2 (M2.8 D6): a grammar compile failure delivered inside the stream (SGLang's in-stream error object, code
/// 400) ends the stream with `invalid_value` on the constraint field and a fixed message; the unary request gets
/// the same text as an HTTP 400 and the same client error.
#[tokio::test]
async fn c2_in_stream_grammar_compile_failure_is_invalid_value() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    h.mock.set_fault(Fault::StreamError {
        code: 400,
        message: "Grammar compilation failed: secret".to_owned(),
    });
    let mut body = long_stream(4);
    body["ebnf"] = json!("root ::= \"a\"");
    let (rid, mut reader) = open_stream(&h, &body).await;
    let events = reader.rest().await;
    let terminal = check_stream(&events);
    assert_eq!(terminal["error"]["code"], "invalid_value", "{terminal}");
    assert_eq!(terminal["error"]["param"], "ebnf", "{terminal}");
    assert!(!terminal.to_string().contains("secret"), "{terminal}");
    assert_eq!(terminal["cere"]["state"], "failed");
    assert_eq!(terminal["cere"]["reason"], "invalid_value");
    let record = h.terminal(&rid).await;
    assert_eq!(
        (&record["state"], &record["reason"]),
        (&json!("failed"), &json!("invalid_value")),
        "{record}"
    );

    let mut unary = chat_body(false);
    unary["ebnf"] = json!("root ::= \"a\"");
    let response = h.chat(&unary).send().await.unwrap();
    assert_eq!(response.status(), 400);
    let error: Value = response.json().await.unwrap();
    assert_eq!(
        (&error["error"]["code"], &error["error"]["param"]),
        (&json!("invalid_value"), &json!("ebnf")),
        "{error}"
    );
    assert_eq!(error["error"]["message"], terminal["error"]["message"]);
}

/// Polls until `App::confirmed_streams()` is `want` (10 s bound).
async fn wait_confirmed(h: &Harness, want: u64) {
    let until = tokio::time::Instant::now() + Duration::from_secs(10);
    while h.app.confirmed_streams() != want {
        assert!(
            tokio::time::Instant::now() < until,
            "confirmed streams stayed {} (want {want})",
            h.app.confirmed_streams()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// m1close Q6: `C` stays a lower bound of cered's share of the backend load. A relay blocked waiting for client
/// room cannot see the backend finish, so its stream leaves `C` for the wait; the next backend chunk after
/// the wait counts it again. A stream whose client keeps reading counts throughout.
#[tokio::test]
async fn q6_a_relay_blocked_on_its_client_is_not_confirmed() {
    let h = Harness::start_socket(slow(1, 1_000_000), &[], test_tuning(), Some(4096)).await;
    let (_, mut reading) = open_stream(&h, &long_stream(32_768)).await;
    let reader = tokio::spawn(async move { while reading.next().await.is_some() {} });
    let (head, mut stalled, _) = raw_chat(&h.base, &long_stream(32_768)).await;
    assert!(head.starts_with("HTTP/1.0 200"), "{head}");
    // Do not read: the socket buffers and the client channel fill, the relay blocks (stall bound 30 s).
    wait_confirmed(&h, 1).await;
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(
            h.app.confirmed_streams(),
            1,
            "the blocked relay counted again"
        );
    }
    assert_eq!(h.mock.in_flight(), 2);
    // Reading again unblocks the relay: its next backend chunk counts it again.
    let drain = tokio::spawn(async move {
        let mut buf = [0u8; 4096];
        while tokio::io::AsyncReadExt::read(&mut stalled, &mut buf)
            .await
            .is_ok_and(|n| n > 0)
        {}
    });
    wait_confirmed(&h, 2).await;
    drain.abort();
    reader.abort();
}

fn tool_chunk(delta: Value, finish: Value) -> Value {
    json!({
        "id": "5a6b7c8d", "object": "chat.completion.chunk", "created": 1_760_000_000, "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "logprobs": null, "finish_reason": finish, "matched_stop": null}]
    })
}

/// SGLang's tool-call stream (F3): role chunk, the call's head, an `arguments` fragment with logprobs, and a
/// finish chunk that still carries the last fragment; then the usage chunk.
fn tool_call_chunks() -> Vec<Value> {
    let mut fragment = tool_chunk(
        json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\": \"Pa"}}]}),
        Value::Null,
    );
    fragment["choices"][0]["logprobs"] = json!({"content": [
        {"token": "Pa", "logprob": -0.25, "bytes": null, "top_logprobs": []}
    ]});
    vec![
        tool_chunk(json!({"role": "assistant", "content": ""}), Value::Null),
        tool_chunk(
            json!({"tool_calls": [{"id": "call_0123456789abcdef01234567", "index": 0, "type": "function",
                                   "function": {"name": "get_weather", "arguments": ""}}]}),
            Value::Null,
        ),
        fragment,
        tool_chunk(
            json!({"tool_calls": [{"index": 0, "function": {"arguments": "ris\"}"}}]}),
            json!("tool_calls"),
        ),
        json!({"id": "5a6b7c8d", "object": "chat.completion.chunk", "created": 1_760_000_000, "model": MODEL,
               "choices": [], "usage": {"prompt_tokens": 30, "completion_tokens": 9, "total_tokens": 39}}),
    ]
}

/// C8 stream (D5-D7): deltas relayed value-equal with their `index`, the request ends `completed` with TTFT on a
/// tool delta, the call assembled across chunks (including the held finish chunk) is committed, and the
/// stream-assembled message echoed back continues at revision + 1.
#[tokio::test]
async fn c8_streamed_tool_call_relays_commits_and_continues() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let chunks = tool_call_chunks();
    h.mock.set_scripted(Some(cere_mock_sglang::Scripted {
        response: Value::Null,
        chunks: chunks.clone(),
    }));
    let r = h.post("/v1/cere/sessions", TOKEN_A).send().await.unwrap();
    let session = r.json::<Value>().await.unwrap()["session_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let tools = json!([{"type": "function", "function": {"name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]);
    let user = json!({"role": "user", "content": "Weather in Paris?"});
    let body = json!({"model": MODEL, "messages": [user], "max_completion_tokens": 16, "stream": true,
                      "tools": tools});
    let response = h
        .chat(&body)
        .header("X-Cere-Session-Id", &session)
        .header("X-Cere-Expected-Revision", "0")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let events = SseReader::new(response).rest().await;
    let data: Vec<Value> = events
        .iter()
        .filter_map(|e| match e {
            Event::Data(v) if v["cere"]["event"] != "accepted" => Some(v.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(data.len(), 4, "{data:?}");
    for (got, want) in data.iter().zip(&chunks) {
        let mut got = got.clone();
        got.as_object_mut().unwrap().remove("cere");
        assert_eq!(&got, want);
    }
    let finish = &data[3]["cere"];
    assert_eq!(finish["state"], "completed", "{finish}");
    assert_eq!(finish["revision"], 1, "{finish}");
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "completed", "{status}");
    assert!(status["timings"]["first_generated_ms"].is_u64(), "{status}");

    // The client assembles the message from the deltas and echoes it.
    let assembled = json!({"role": "assistant", "content": null, "tool_calls": [
        {"id": "call_0123456789abcdef01234567", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
    ]});
    h.mock.set_scripted(None);
    let next = json!({"model": MODEL, "max_completion_tokens": 4, "tools": tools, "messages": [
        user, assembled,
        {"role": "tool", "tool_call_id": "call_0123456789abcdef01234567", "content": "18 C"}
    ]});
    let r = h
        .chat(&next)
        .header("X-Cere-Session-Id", &session)
        .header("X-Cere-Expected-Revision", "1")
        .send()
        .await
        .unwrap();
    let code = r.status();
    let body: Value = r.json().await.unwrap();
    assert_eq!(code, 200, "{body}");
    assert_eq!(body["cere"]["revision"], 2, "{body}");
}

/// One SSE body with the per-request and per-run values masked (`id`, `created`, the `cere` request id and
/// timings, the scratch config's hash); every other byte is compared.
fn masked_events(raw: &str) -> String {
    let mut out = String::new();
    for event in raw.split("\n\n").filter(|e| !e.is_empty()) {
        let (id, data) = event.split_once("\ndata: ").unwrap_or(("", event));
        let data = data.strip_prefix("data: ").unwrap_or(data);
        if data == "[DONE]" {
            out.push_str(&format!("{id} [DONE]\n"));
            continue;
        }
        let mut v: Value = serde_json::from_str(data).unwrap_or_else(|e| panic!("{e}: {data}"));
        assert_eq!(serde_json::to_string(&v).unwrap(), data, "not canonical");
        for key in ["id", "created"] {
            if v.get(key).is_some() {
                v[key] = json!("<masked>");
            }
        }
        if let Some(cere) = v.get_mut("cere").and_then(Value::as_object_mut) {
            for key in [
                "request_id",
                "config_hash",
                "instance_id",
                "since_unix_ms",
                "ttft_ms",
                "e2e_ms",
                "queue_ms",
            ] {
                if cere.contains_key(key) {
                    cere.insert(key.to_owned(), json!("<masked>"));
                }
            }
        }
        out.push_str(&format!("{id} {v}\n"));
    }
    out
}

/// M2.6 C7 regression guard: single-choice relay bodies (a generated stream with and without the usage chunk,
/// and the scripted tool-call stream) are byte-equal to the bodies recorded on the merge base, apart from the
/// values [`masked_events`] names.
#[tokio::test]
async fn single_choice_relay_bodies_equal_merge_base_goldens() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let mut got = String::new();
    for (name, include_usage) in [("generated", false), ("generated_usage", true)] {
        let mut body = chat_body(true);
        body["stream_options"] = json!({"include_usage": include_usage});
        let response = h.chat(&body).send().await.unwrap();
        assert_eq!(response.status(), 200, "{name}");
        got.push_str(&format!(
            "# {name}\n{}",
            masked_events(&response.text().await.unwrap())
        ));
    }
    h.mock.set_scripted(Some(cere_mock_sglang::Scripted {
        response: Value::Null,
        chunks: tool_call_chunks(),
    }));
    let mut body = json!({"model": MODEL, "messages": [{"role": "user", "content": "Weather in Paris?"}],
                          "max_completion_tokens": 16, "stream": true,
                          "stream_options": {"include_usage": true}});
    body["tools"] = json!([{"type": "function", "function": {"name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]);
    let response = h.chat(&body).send().await.unwrap();
    assert_eq!(response.status(), 200);
    got.push_str(&format!(
        "# scripted_tool_call\n{}",
        masked_events(&response.text().await.unwrap())
    ));
    assert!(
        got == RELAY_GOLDEN,
        "single-choice relay bodies differ:\n{got}"
    );
}

/// The scripted stream of a single-choice request whose chunk `extra` comes before the finish chunk; the request
/// fails `backend_protocol_error` with the SSE error event.
async fn assert_scripted_protocol_error(extra: Value) {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let usage = json!({"id": "5a6b7c8d", "object": "chat.completion.chunk", "created": 1_760_000_000,
                       "model": MODEL, "choices": [],
                       "usage": {"prompt_tokens": 30, "completion_tokens": 2, "total_tokens": 32}});
    let chunks = vec![
        tool_chunk(json!({"role": "assistant", "content": ""}), Value::Null),
        tool_chunk(json!({"content": "Hi"}), Value::Null),
        extra,
        tool_chunk(json!({"content": null}), json!("stop")),
        usage,
    ];
    h.mock.set_scripted(Some(cere_mock_sglang::Scripted {
        response: Value::Null,
        chunks,
    }));
    let (rid, mut reader) = open_stream(&h, &long_stream(16)).await;
    let events = reader.rest().await;
    let terminal = check_stream(&events);
    assert_eq!(terminal["cere"]["state"], "failed", "{events:?}");
    assert_eq!(
        terminal["cere"]["reason"], "backend_protocol_error",
        "{events:?}"
    );
    let (_, status) = h.status(&rid, TOKEN_A).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], "backend_protocol_error", "{status}");
}

/// M2.6 C7 (D11): a chunk of choice 1 on a request of one choice is a backend protocol error.
#[tokio::test]
async fn c7_chunk_outside_the_choices_fails() {
    let mut chunk = tool_chunk(json!({"content": " there"}), Value::Null);
    chunk["choices"][0]["index"] = json!(1);
    assert_scripted_protocol_error(chunk).await;
}

/// M2.6 C7 (D11): a second finish chunk of the same choice is a backend protocol error.
#[tokio::test]
async fn c7_duplicate_finish_fails() {
    assert_scripted_protocol_error(tool_chunk(json!({"content": null}), json!("length"))).await;
}

/// M2.6 C7 (D11): a native `n` = 3 stream on a one-event client buffer: the terminal needs n + 2 free events
/// (three finish chunks, the usage chunk, `[DONE]`). It completes with the finishes in `index` order after the
/// choices' chunks, the last carrying the terminal `cere`, then one usage chunk summing the choices.
#[tokio::test]
async fn c7_native_three_choices_relay_one_finish_each() {
    let case = native_server::Case {
        stream_buffer_events: Some(1),
        ..native_server::Case::matching()
    };
    let native = native_server::Native::ready(case).await;
    let body = json!({"model": MODEL, "messages": [{"role": "user", "content": "rows"}], "max_tokens": 6,
                      "temperature": 0, "stream": true, "stream_options": {"include_usage": true}, "n": 3});
    let response = native.chat(&body).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let events = SseReader::new(response).rest().await;
    assert!(matches!(events.last(), Some(Event::Done)), "{events:?}");
    let data: Vec<&Value> = events
        .iter()
        .filter_map(|e| match e {
            Event::Data(v) => Some(v),
            Event::Done => None,
        })
        .collect();
    let finishes: Vec<usize> = data
        .iter()
        .enumerate()
        .filter(|(_, d)| !d["choices"][0]["finish_reason"].is_null())
        .map(|(i, _)| i)
        .collect();
    assert_eq!(finishes.len(), 3, "{events:?}");
    let last = data.len() - 1;
    assert_eq!(finishes, [last - 3, last - 2, last - 1], "{events:?}");
    for (index, &at) in finishes.iter().enumerate() {
        assert_eq!(data[at]["choices"][0]["index"], index, "{events:?}");
    }
    assert_eq!(data[last - 1]["cere"]["state"], "completed", "{events:?}");
    let usage = &data[last]["usage"];
    let status = native.terminal(&rid).await;
    assert_eq!(status["state"], "completed", "{status}");
    assert_eq!(
        status["usage"]["completion_tokens"], usage["completion_tokens"],
        "{status}"
    );
    assert_eq!(
        usage["completion_tokens"].as_u64().unwrap() % 3,
        0,
        "{usage}"
    );
}

/// Recorded on `bl/m26-rust`'s merge base `cfe1a3a` (pre-M2.6 relay).
const RELAY_GOLDEN: &str = r#"# generated
id: 0 {"cere":{"attempt":1,"cache_report":"unavailable","cached_tokens":null,"config_hash":"<masked>","epoch":1,"event":"accepted","execution_mode":"backend_managed","instance_id":"<masked>","lock_sha256":"eb182556781d898eb7a0cd8780ce20000586eb10f8114c6cd81393381ec236ec","profile_id":"sglang-t4-qwen38-flash-next-v1","recipe_id":"qwen38-flash-next","request_id":"<masked>","seq":0,"worker_epoch":1,"worker_id":"gpu1"},"choices":[],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk"}
id: 1 {"cere":{"committed_tokens":null,"seq":1},"choices":[{"delta":{"content":"","reasoning_content":null,"role":"assistant"},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 2 {"cere":{"committed_tokens":null,"seq":2},"choices":[{"delta":{"content":"tok0","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 3 {"cere":{"committed_tokens":null,"seq":3},"choices":[{"delta":{"content":" tok1","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 4 {"cere":{"committed_tokens":null,"seq":4},"choices":[{"delta":{"content":" tok2","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 5 {"cere":{"committed_tokens":null,"seq":5},"choices":[{"delta":{"content":" tok3","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 6 {"cere":{"committed_tokens":null,"seq":6},"choices":[{"delta":{"content":" tok4","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 7 {"cere":{"committed_tokens":null,"seq":7},"choices":[{"delta":{"content":" tok5","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 8 {"cere":{"committed_tokens":null,"seq":8},"choices":[{"delta":{"content":" tok6","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 9 {"cere":{"committed_tokens":null,"seq":9},"choices":[{"delta":{"content":" tok7","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 10 {"cere":{"attempt":1,"cache_report":"unavailable","cached_tokens":null,"committed_tokens":null,"config_hash":"<masked>","epoch":1,"execution_mode":"backend_managed","instance_id":"<masked>","lock_sha256":"eb182556781d898eb7a0cd8780ce20000586eb10f8114c6cd81393381ec236ec","profile_id":"sglang-t4-qwen38-flash-next-v1","recipe_id":"qwen38-flash-next","request_id":"<masked>","seq":10,"state":"completed","worker_epoch":1,"worker_id":"gpu1"},"choices":[{"delta":{"content":null,"reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":"length","index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
 [DONE]
# generated_usage
id: 0 {"cere":{"attempt":1,"cache_report":"unavailable","cached_tokens":null,"config_hash":"<masked>","epoch":1,"event":"accepted","execution_mode":"backend_managed","instance_id":"<masked>","lock_sha256":"eb182556781d898eb7a0cd8780ce20000586eb10f8114c6cd81393381ec236ec","profile_id":"sglang-t4-qwen38-flash-next-v1","recipe_id":"qwen38-flash-next","request_id":"<masked>","seq":0,"worker_epoch":1,"worker_id":"gpu1"},"choices":[],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk"}
id: 1 {"cere":{"committed_tokens":null,"seq":1},"choices":[{"delta":{"content":"","reasoning_content":null,"role":"assistant"},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 2 {"cere":{"committed_tokens":null,"seq":2},"choices":[{"delta":{"content":"tok0","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 3 {"cere":{"committed_tokens":null,"seq":3},"choices":[{"delta":{"content":" tok1","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 4 {"cere":{"committed_tokens":null,"seq":4},"choices":[{"delta":{"content":" tok2","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 5 {"cere":{"committed_tokens":null,"seq":5},"choices":[{"delta":{"content":" tok3","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 6 {"cere":{"committed_tokens":null,"seq":6},"choices":[{"delta":{"content":" tok4","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 7 {"cere":{"committed_tokens":null,"seq":7},"choices":[{"delta":{"content":" tok5","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 8 {"cere":{"committed_tokens":null,"seq":8},"choices":[{"delta":{"content":" tok6","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 9 {"cere":{"committed_tokens":null,"seq":9},"choices":[{"delta":{"content":" tok7","reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 10 {"cere":{"attempt":1,"cache_report":"unavailable","cached_tokens":null,"committed_tokens":null,"config_hash":"<masked>","epoch":1,"execution_mode":"backend_managed","instance_id":"<masked>","lock_sha256":"eb182556781d898eb7a0cd8780ce20000586eb10f8114c6cd81393381ec236ec","profile_id":"sglang-t4-qwen38-flash-next-v1","recipe_id":"qwen38-flash-next","request_id":"<masked>","seq":10,"state":"completed","worker_epoch":1,"worker_id":"gpu1"},"choices":[{"delta":{"content":null,"reasoning_content":null,"role":null,"tool_calls":null},"finish_reason":"length","index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":null}
id: 11 {"cere":{"committed_tokens":null,"seq":11},"choices":[],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":{"completion_tokens":8,"prompt_tokens":2,"prompt_tokens_details":null,"reasoning_tokens":0,"total_tokens":10}}
 [DONE]
# scripted_tool_call
id: 0 {"cere":{"attempt":1,"cache_report":"unavailable","cached_tokens":null,"config_hash":"<masked>","epoch":1,"event":"accepted","execution_mode":"backend_managed","instance_id":"<masked>","lock_sha256":"eb182556781d898eb7a0cd8780ce20000586eb10f8114c6cd81393381ec236ec","profile_id":"sglang-t4-qwen38-flash-next-v1","recipe_id":"qwen38-flash-next","request_id":"<masked>","seq":0,"worker_epoch":1,"worker_id":"gpu1"},"choices":[],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk"}
id: 1 {"cere":{"committed_tokens":null,"seq":1},"choices":[{"delta":{"content":"","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk"}
id: 2 {"cere":{"committed_tokens":null,"seq":2},"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"","name":"get_weather"},"id":"call_0123456789abcdef01234567","index":0,"type":"function"}]},"finish_reason":null,"index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk"}
id: 3 {"cere":{"committed_tokens":null,"seq":3},"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\"city\": \"Pa"},"index":0}]},"finish_reason":null,"index":0,"logprobs":{"content":[{"bytes":null,"logprob":-0.25,"token":"Pa","top_logprobs":[]}]},"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk"}
id: 4 {"cere":{"attempt":1,"cache_report":"unavailable","cached_tokens":null,"committed_tokens":null,"config_hash":"<masked>","epoch":1,"execution_mode":"backend_managed","instance_id":"<masked>","lock_sha256":"eb182556781d898eb7a0cd8780ce20000586eb10f8114c6cd81393381ec236ec","profile_id":"sglang-t4-qwen38-flash-next-v1","recipe_id":"qwen38-flash-next","request_id":"<masked>","seq":4,"state":"completed","worker_epoch":1,"worker_id":"gpu1"},"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"ris\"}"},"index":0}]},"finish_reason":"tool_calls","index":0,"logprobs":null,"matched_stop":null}],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk"}
id: 5 {"cere":{"committed_tokens":null,"seq":5},"choices":[],"created":"<masked>","id":"<masked>","model":"Qwen3.8-Flash-Next","object":"chat.completion.chunk","usage":{"completion_tokens":9,"prompt_tokens":30,"total_tokens":39}}
 [DONE]
"#;
