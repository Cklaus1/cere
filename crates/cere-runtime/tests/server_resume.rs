//! m1close R2 (Q3-Q5): `X-Cere-On-Disconnect: continue` streams detach instead of cancelling, and
//! `GET /v1/cere/requests/{rid}/events` replays the retained events after a cursor, then follows live.
mod support;

use std::time::Duration;

use cere_mock_sglang::MockConfig;
use serde_json::{json, Value};
use support::*;

const CONTINUE: (&str, &str) = ("X-Cere-On-Disconnect", "continue");

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

fn seq(event: &Event) -> u64 {
    event.data()["cere"]["seq"].as_u64().expect("seq")
}

fn aborted_rids(h: &Harness) -> Vec<String> {
    h.mock.abort_requests().into_iter().map(|a| a.rid).collect()
}

/// How often the mock saw `rid` (the readiness smoke is dispatched too).
fn dispatches(h: &Harness, rid: &str) -> usize {
    h.mock
        .dispatched_rids()
        .iter()
        .filter(|r| *r == rid)
        .count()
}

/// Opens a continue-mode stream; returns its rid and the original response as a reader.
async fn open_continue(h: &Harness, body: &Value) -> (String, SseReader) {
    let response = h
        .chat(body)
        .header(CONTINUE.0, CONTINUE.1)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    (rid, SseReader::new(response))
}

/// `GET /v1/cere/requests/{rid}/events` with an optional query and extra headers.
async fn events(
    h: &Harness,
    rid: &str,
    query: &str,
    token: &str,
    headers: &[(&str, &str)],
) -> reqwest::Response {
    let mut request = h.get(&format!("/v1/cere/requests/{rid}/events{query}"), token);
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    request.send().await.unwrap()
}

/// Resumes `rid` after `after` and returns the reader (asserting 200).
async fn resume(h: &Harness, rid: &str, after: u64) -> SseReader {
    let response = events(h, rid, &format!("?after={after}"), TOKEN_A, &[]).await;
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(response.headers()["x-cere-request-id"], rid);
    SseReader::new(response)
}

async fn refused(response: reqwest::Response, status: u16, code: &str) {
    assert_eq!(response.status(), status);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], code, "{body}");
}

/// Reads the original stream up to and including `seq == k`.
async fn read_through(reader: &mut SseReader, k: u64) -> Vec<Event> {
    let mut seen = Vec::new();
    loop {
        let event = reader.next().await.expect("stream ended early");
        let at = seq(&event);
        seen.push(event);
        if at == k {
            return seen;
        }
    }
}

#[tokio::test]
async fn r2_disconnect_then_resume_replays_once_and_follows_live() {
    let h = Harness::start(slow(30, 40), &[]).await;
    let (rid, mut original) = open_continue(&h, &long_stream(40)).await;
    let mut seen = read_through(&mut original, 3).await;
    drop(original);
    // Detached, not cancelled: the request keeps generating.
    tokio::time::sleep(Duration::from_millis(150)).await;
    let (_, status) = h.status(&rid, TOKEN_A).await;
    assert_eq!(status["state"], "running", "{status}");
    let mut reader = resume(&h, &rid, 3).await;
    let replayed = reader.rest().await;
    assert_eq!(seq(&replayed[0]), 4, "replay starts right after the cursor");
    let terminal = check_stream_tail(&replayed, 3);
    assert_eq!(terminal["cere"]["state"], "completed");
    seen.extend(replayed);
    let seqs: Vec<u64> = seen
        .iter()
        .filter(|e| **e != Event::Done)
        .map(seq)
        .collect();
    assert_eq!(
        seqs,
        (0..seqs.len() as u64).collect::<Vec<_>>(),
        "exactly once"
    );
    check_stream(&seen);
    assert_eq!(dispatches(&h, &rid), 1, "no regeneration");
    assert!(aborted_rids(&h).is_empty());
    assert_eq!(h.terminal(&rid).await["state"], "completed");
}

#[tokio::test]
async fn r2_after_wins_over_last_event_id() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let (rid, mut original) = open_continue(&h, &long_stream(8)).await;
    check_stream(&original.rest().await);
    let response = events(&h, &rid, "?after=5", TOKEN_A, &[("Last-Event-ID", "2")]).await;
    let replay = SseReader::new(response).rest().await;
    assert_eq!(seq(&replay[0]), 6);
    check_stream_tail(&replay, 5);
    let response = events(&h, &rid, "", TOKEN_A, &[("Last-Event-ID", "2")]).await;
    let replay = SseReader::new(response).rest().await;
    assert_eq!(seq(&replay[0]), 3);
    check_stream_tail(&replay, 2);
    // Neither: from the accepted event.
    let response = events(&h, &rid, "", TOKEN_A, &[]).await;
    check_stream(&SseReader::new(response).rest().await);
    let response = events(&h, &rid, "?after=x", TOKEN_A, &[]).await;
    refused(response, 400, "invalid_value").await;
}

#[tokio::test]
async fn r2_codes_for_unknown_other_scope_and_non_continue() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let (rid, mut original) = open_continue(&h, &chat_body(true)).await;
    original.rest().await;
    refused(
        events(&h, &rid, "", TOKEN_B, &[]).await,
        404,
        "request_not_found",
    )
    .await;
    let unknown = "cere-00000000000000000001-00000000000000000001-0123456789abcdef";
    refused(
        events(&h, unknown, "", TOKEN_A, &[]).await,
        404,
        "request_not_found",
    )
    .await;
    let unauthenticated = h
        .http
        .get(h.url(&format!("/v1/cere/requests/{rid}/events")))
        .send()
        .await
        .unwrap();
    assert_eq!(unauthenticated.status(), 401);

    // A stream without the header, and a unary request with it, are not resumable.
    let response = h.chat(&chat_body(true)).send().await.unwrap();
    let plain = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    SseReader::new(response).rest().await;
    refused(
        events(&h, &plain, "", TOKEN_A, &[]).await,
        409,
        "not_resumable",
    )
    .await;
    let response = h
        .chat(&chat_body(false))
        .header(CONTINUE.0, CONTINUE.1)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let unary = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    refused(
        events(&h, &unary, "", TOKEN_A, &[]).await,
        409,
        "not_resumable",
    )
    .await;

    let response = h
        .chat(&chat_body(true))
        .header(CONTINUE.0, "sometimes")
        .send()
        .await
        .unwrap();
    refused(response, 400, "invalid_value").await;
}

/// A rid from the previous incarnation is still known to the ledger, but its buffer died with the process.
#[tokio::test]
async fn r2_prior_epoch_rid_is_cursor_expired() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let (rid, mut original) = open_continue(&h, &chat_body(true)).await;
    original.rest().await;
    assert_eq!(h.terminal(&rid).await["state"], "completed");
    let h = h.restart().await;
    let (code, status) = h.status(&rid, TOKEN_A).await;
    assert_eq!(code, 200, "{status}");
    assert_ne!(status["epoch"], h.app.epoch());
    refused(
        events(&h, &rid, "", TOKEN_A, &[]).await,
        410,
        "cursor_expired",
    )
    .await;
}

/// Q4 expiry codes with short retentions: buffer released → 410; past the ledger's terminal retention → 404,
/// even when no request was accepted since (nothing purged the ledger) and the buffer's tombstone is gone.
#[tokio::test]
async fn r2_ledger_expired_rid_is_not_found_after_its_tombstone() {
    let h = Harness::start(
        MockConfig::default(),
        &[
            (
                "shutdown_drain_ms = 30000",
                "shutdown_drain_ms = 30000\nresume_retention_ms = 1000",
            ),
            (
                "terminal_retention_ms = 3600000",
                "terminal_retention_ms = 2000",
            ),
        ],
    )
    .await;
    let (rid, mut original) = open_continue(&h, &chat_body(true)).await;
    original.rest().await;
    assert_eq!(h.terminal(&rid).await["state"], "completed");
    tokio::time::sleep(Duration::from_millis(1200)).await;
    // Releases the buffer now; its tombstone lasts the ledger retention from here, past the ledger's own expiry.
    refused(
        events(&h, &rid, "", TOKEN_A, &[]).await,
        410,
        "cursor_expired",
    )
    .await;
    tokio::time::sleep(Duration::from_millis(2100)).await;
    refused(
        events(&h, &rid, "", TOKEN_A, &[]).await,
        404,
        "request_not_found",
    )
    .await;
}

/// Per-request bound: the oldest events go first, so an early cursor gets 410 while a late one still works.
#[tokio::test]
async fn r2_per_request_overflow_expires_early_cursors() {
    let h = Harness::start(
        slow(2, 200),
        &[(
            "shutdown_drain_ms = 30000",
            "shutdown_drain_ms = 30000\nresume_buffer_bytes = 4096",
        )],
    )
    .await;
    let (rid, mut original) = open_continue(&h, &long_stream(200)).await;
    let all = original.rest().await;
    let terminal = check_stream(&all);
    assert_eq!(terminal["cere"]["state"], "completed");
    let last = seq(&all[all.len() - 2]);
    refused(
        events(&h, &rid, "", TOKEN_A, &[]).await,
        410,
        "cursor_expired",
    )
    .await;
    refused(
        events(&h, &rid, "?after=3", TOKEN_A, &[]).await,
        410,
        "cursor_expired",
    )
    .await;
    let tail = resume(&h, &rid, last - 1).await.rest().await;
    assert_eq!(tail.len(), 2, "{tail:?}");
    check_stream_tail(&tail, last - 1);
}

/// Process bound: a growing buffer evicts the oldest terminal buffer first. One backend slot, so the total may
/// equal the per-request bound.
#[tokio::test]
async fn r2_total_cap_evicts_the_oldest_terminal_buffer() {
    let h = Harness::start(
        slow(1, 1000),
        &[
            (
                "shutdown_drain_ms = 30000",
                "shutdown_drain_ms = 30000\nresume_buffer_bytes = 65536\n\
                 resume_buffer_total_bytes = 65536",
            ),
            ("max_backend_inflight = 2", "max_backend_inflight = 1"),
            ("per_scope_inflight = 2", "per_scope_inflight = 1"),
        ],
    )
    .await;
    let (old, mut reader) = open_continue(&h, &long_stream(4)).await;
    check_stream(&reader.rest().await);
    resume(&h, &old, 0).await.rest().await;
    let (big, mut reader) = open_continue(&h, &long_stream(1000)).await;
    let all = reader.rest().await;
    check_stream(&all);
    refused(
        events(&h, &old, "?after=0", TOKEN_A, &[]).await,
        410,
        "cursor_expired",
    )
    .await;
    let last = seq(&all[all.len() - 2]);
    check_stream_tail(&resume(&h, &big, last - 1).await.rest().await, last - 1);
}

/// One live reader: the superseded one gets `: superseded` and a close, never `[DONE]` or an error event.
#[tokio::test]
async fn r2_new_reader_supersedes_the_old_one() {
    let h = Harness::start(slow(30, 30), &[]).await;
    let (rid, mut original) = open_continue(&h, &long_stream(30)).await;
    read_through(&mut original, 2).await;
    let mut second = resume(&h, &rid, 2).await;
    let mut raws = Vec::new();
    while let Some(raw) = original.next_raw().await {
        raws.push(raw);
    }
    assert_eq!(
        raws.last().map(String::as_str),
        Some(": superseded"),
        "{raws:?}"
    );
    for raw in &raws[..raws.len() - 1] {
        if raw.starts_with(':') {
            continue;
        }
        let event = parse_event(raw);
        assert_ne!(event, Event::Done, "{raws:?}");
        assert!(event.data().get("error").is_none(), "{raws:?}");
    }
    let rest = second.rest().await;
    assert_eq!(check_stream_tail(&rest, 2)["cere"]["state"], "completed");
    assert_eq!(h.terminal(&rid).await["state"], "completed");
}

/// A stalled continue-mode client is detached, not cancelled `client_backpressure` (compare
/// `l5_slow_consumer_is_cancelled_for_backpressure`), and can collect the whole stream later.
#[tokio::test]
async fn r2_stalled_continue_client_detaches() {
    let h = Harness::start_socket(
        slow(1, 2000),
        &[
            ("stream_buffer_events = 256", "stream_buffer_events = 1"),
            ("stall_timeout_ms = 30000", "stall_timeout_ms = 200"),
        ],
        test_tuning(),
        Some(4096),
    )
    .await;
    let (head, _socket, _) = raw_chat_with(&h.base, &long_stream(2000), &[CONTINUE]).await;
    assert!(head.starts_with("HTTP/1.0 200"), "{head}");
    let rid = head
        .lines()
        .find_map(|l| l.strip_prefix("x-cere-request-id: "))
        .expect("request id header")
        .to_owned();
    // Never read the socket.
    let record = h.terminal(&rid).await;
    assert_eq!(record["state"], "completed", "{record}");
    assert_eq!(record["timings"]["client_backpressure_ms"], 0, "{record}");
    assert!(aborted_rids(&h).is_empty());
    let response = events(&h, &rid, "", TOKEN_A, &[]).await;
    let all = SseReader::new(response).rest().await;
    assert_eq!(check_stream(&all)["cere"]["state"], "completed");
}

/// A disconnect while still queued (no `accepted` event yet) cancels in continue mode too.
#[tokio::test]
async fn r2_queued_disconnect_still_cancels() {
    let h = Harness::start(
        slow(40, 30),
        &[
            ("max_backend_inflight = 2", "max_backend_inflight = 1"),
            ("per_scope_inflight = 2", "per_scope_inflight = 1"),
        ],
    )
    .await;
    let (first, holder) = open_continue(&h, &long_stream(30)).await;
    let request = h
        .chat(&long_stream(30))
        .header(CONTINUE.0, CONTINUE.1)
        .send();
    let waiting = tokio::spawn(request);
    for _ in 0..200 {
        if h.app.admission_snapshot().queued == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(h.app.admission_snapshot().queued, 1);
    waiting.abort();
    let queued = journal(h.state_dir.path())
        .into_iter()
        .filter(|r| r["type"] == "admitted" && r["rid"] != first.as_str())
        .map(|r| r["rid"].as_str().unwrap().to_owned())
        .next()
        .expect("the queued request was admitted to the ledger");
    let record = h.terminal(&queued).await;
    assert_eq!(record["state"], "cancelled", "{record}");
    assert_eq!(record["reason"], "client_disconnected");
    assert_eq!(h.app.admission_snapshot().queued, 0);
    // The detached original runs on to its natural end; only it ever reached the backend.
    drop(holder);
    assert_eq!(h.terminal(&first).await["state"], "completed");
    assert_eq!(dispatches(&h, &first), 1);
    assert_eq!(dispatches(&h, &queued), 0);
}

/// Detached work keeps its own deadline: it is cancelled `deadline_exceeded` and the backend aborted. The deadline
/// is a tokio `Instant` of this runtime, so the test stops the clock (paused, and a blocking task keeps it from
/// auto-advancing) until the stream is open and detached: the 400 ms cannot pass before cered opens the stream,
/// however slow the host. The cancel bound ([`test_tuning`]'s) is long: the mock always honours an abort of an
/// open stream, so only a dropped abort could miss it.
#[tokio::test]
async fn r2_detached_request_is_cancelled_at_its_deadline() {
    let h = Harness::start(slow(40, 1000), &[]).await;
    let (thaw, frozen) = std::sync::mpsc::channel::<()>();
    let holder = tokio::task::spawn_blocking(move || frozen.recv());
    tokio::time::pause();
    let response = h
        .chat(&long_stream(1000))
        .header(CONTINUE.0, CONTINUE.1)
        .header("X-Cere-Deadline-Ms", "400")
        .send()
        .await
        .unwrap();
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let mut original = SseReader::new(response);
    original.next().await.unwrap();
    drop(original);
    tokio::time::resume();
    drop(thaw);
    holder.await.unwrap().unwrap_err();
    let record = h.terminal(&rid).await;
    assert_eq!(record["state"], "cancelled", "{record}");
    assert_eq!(record["reason"], "deadline_exceeded");
    assert_eq!(aborted_rids(&h), vec![rid.clone()]);
    let response = events(&h, &rid, "", TOKEN_A, &[]).await;
    let terminal = check_stream(&SseReader::new(response).rest().await);
    assert_eq!(terminal["error"]["code"], "deadline_exceeded", "{terminal}");
    let (_, status) = h.status(&rid, TOKEN_A).await;
    assert_eq!(terminal["cere"]["usage"], status["usage"]);
}

/// A keyed retry of a continue stream is still `stream_not_replayable`, and names the resume URL.
#[tokio::test]
async fn r2_keyed_retry_names_the_events_url() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let keyed = |h: &Harness| {
        h.chat(&chat_body(true))
            .header(CONTINUE.0, CONTINUE.1)
            .header("Idempotency-Key", "resume-key-1")
    };
    let response = keyed(&h).send().await.unwrap();
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    SseReader::new(response).rest().await;
    h.terminal(&rid).await;
    let retry = keyed(&h).send().await.unwrap();
    assert_eq!(retry.status(), 409);
    assert_eq!(retry.headers()["x-cere-request-id"], rid.as_str());
    let body: Value = retry.json().await.unwrap();
    assert_eq!(
        body["error"],
        json!({
            "message": "the original request was streamed and cannot be replayed",
            "type": "invalid_request_error",
            "code": "stream_not_replayable",
            "param": null,
            "status_url": format!("/v1/cere/requests/{rid}"),
            "events_url": format!("/v1/cere/requests/{rid}/events"),
        })
    );
    assert_eq!(dispatches(&h, &rid), 1, "no regeneration");
}

/// Q4 before the terminal: a keyed retry of a still-running continue stream is `stream_not_replayable` with the
/// resume URL too (not `request_in_progress`), and the URL resumes it.
#[tokio::test]
async fn r2_keyed_retry_of_a_running_stream_names_the_events_url() {
    let h = Harness::start(slow(30, 40), &[]).await;
    let keyed = |h: &Harness| {
        h.chat(&long_stream(40))
            .header(CONTINUE.0, CONTINUE.1)
            .header("Idempotency-Key", "resume-key-running")
    };
    let response = keyed(&h).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let mut original = SseReader::new(response);
    read_through(&mut original, 1).await;
    drop(original);
    let retry = keyed(&h).send().await.unwrap();
    assert_eq!(retry.status(), 409);
    assert_eq!(retry.headers()["x-cere-request-id"], rid.as_str());
    let body: Value = retry.json().await.unwrap();
    let (_, status) = h.status(&rid, TOKEN_A).await;
    assert_eq!(
        status["state"], "running",
        "retried before the terminal: {status}"
    );
    assert_eq!(body["error"]["code"], "stream_not_replayable", "{body}");
    assert_eq!(
        body["error"]["status_url"],
        format!("/v1/cere/requests/{rid}")
    );
    let url = body["error"]["events_url"].as_str().expect("events_url");
    assert_eq!(url, format!("/v1/cere/requests/{rid}/events"));
    let rest = resume(&h, &rid, 1).await.rest().await;
    assert_eq!(check_stream_tail(&rest, 1)["cere"]["state"], "completed");
    assert_eq!(dispatches(&h, &rid), 1, "no regeneration");
}

/// Critical path: while the producer keeps emitting, a reader disconnects and resumes from its last seen seq
/// again and again. Across all connections every seq arrives exactly once, in order, up to `[DONE]`.
#[tokio::test]
async fn r2_repeated_resumes_while_producing_deliver_each_seq_once() {
    let h = Harness::start(slow(4, 400), &[]).await;
    let (rid, mut reader) = open_continue(&h, &long_stream(400)).await;
    let mut delivered: Vec<u64> = Vec::new();
    let mut done = false;
    let mut connections = 1;
    let mut take = 1usize;
    while !done {
        for _ in 0..take {
            match reader.next().await.expect("connection ended before [DONE]") {
                Event::Done => {
                    done = true;
                    break;
                }
                event => delivered.push(seq(&event)),
            }
        }
        if !done {
            drop(reader);
            reader = resume(&h, &rid, *delivered.last().unwrap()).await;
            connections += 1;
            take = take % 7 + 1;
        }
    }
    assert!(connections > 20, "{connections} connections");
    assert_eq!(
        delivered,
        (0..delivered.len() as u64).collect::<Vec<_>>(),
        "no gap, no duplicate"
    );
    assert_eq!(h.terminal(&rid).await["state"], "completed");
    assert_eq!(dispatches(&h, &rid), 1);
}
