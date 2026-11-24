//! M3.1 C6 (m3 design note Q10, review M3-12): two `sglang_http` replicas, worker `gpu0` on mock A and
//! `gpu1` on mock B. Every request runs on the worker it was granted and on no other; a cancel, a client
//! disconnect, a deadline and a shutdown of a request on B each send `/abort_request` to B only; under saturation
//! neither mock ever holds more requests than the worker cap, readiness smokes included; the shutdown counts open
//! backend streams per worker (A's retained orphan neither hides nor stands for an open stream on B); and mock A
//! going away mid-stream fails only A's streams (`backend_protocol_error`, as today: the connection drops
//! mid-body) while B finishes its own and takes the new work, `/readyz` staying 200 with A suspect, then lost.
mod support;

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use cere_mock_sglang::{Fault, MockConfig, MockHandle, MockSglang, ServerInfo};
use cere_runtime::config::RuntimeConfig;
use serde_json::Value;
use support::*;

/// `mocks[i]` serves device `gpu<i>` ([`Harness::start_replicas`]).
const A: &str = "gpu0";
const B: &str = "gpu1";

/// Global and per-scope caps above the sum of both worker caps (`expect.max_running_requests` = 4 each), so the
/// worker caps are the ones that bind.
const WIDE_CAPS: [(&str, &str); 2] = [
    ("max_backend_inflight = 2", "max_backend_inflight = 8"),
    ("per_scope_inflight = 2", "per_scope_inflight = 8"),
];

const DRAIN_300MS: (&str, &str) = ("shutdown_drain_ms = 30000", "shutdown_drain_ms = 300");

/// `scheduler.worker_lost_after_ms` at its 1 s floor.
const LOST_AFTER_1S: (&str, &str) = (
    "mode = \"replica\"",
    "mode = \"replica\"\nworker_lost_after_ms = 1000",
);

/// Output tokens of a stream that holds A only while the next request is granted (20 ms per token).
const SHORT: u64 = 25;
/// Output tokens of a request that runs until something ends it (80 s at 20 ms per token).
const LONG: u64 = 4_000;

fn slow(token_ms: u64, tokens: u32) -> MockConfig {
    MockConfig {
        token_latency: Duration::from_millis(token_ms),
        natural_output_tokens: tokens,
        ..MockConfig::default()
    }
}

/// Two mocks emitting one token per 20 ms.
async fn slow_mocks() -> Vec<MockHandle> {
    let mut out = Vec::new();
    for _ in 0..2 {
        out.push(MockSglang::start(slow(20, 5_000)).await.unwrap());
    }
    out
}

fn body(stream: bool, tokens: u64) -> Value {
    let mut body = chat_body(stream);
    body["max_completion_tokens"] = tokens.into();
    body
}

fn aborted(mock: &MockHandle) -> Vec<String> {
    mock.abort_requests().into_iter().map(|a| a.rid).collect()
}

fn dispatched(mock: &MockHandle) -> BTreeSet<String> {
    mock.dispatched_rids().into_iter().collect()
}

/// Sends a streamed chat and reads its `accepted` event, which goes out once the backend stream opened; returns
/// the rid, the granted worker and the reader.
async fn open_stream(request: reqwest::RequestBuilder) -> (String, String, SseReader) {
    let response = request.send().await.unwrap();
    assert_eq!(response.status(), 200);
    let rid = response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let mut reader = SseReader::new(response);
    let accepted = reader.next().await.unwrap();
    let cere = &accepted.data()["cere"];
    assert_eq!(cere["event"], "accepted", "{cere}");
    assert_eq!(cere["request_id"], rid.as_str(), "{cere}");
    let worker = cere["worker_id"]
        .as_str()
        .expect("granted worker")
        .to_owned();
    (rid, worker, reader)
}

/// A short stream on A, the least loaded worker on an id tie: while it is open the next request is granted B. It
/// ends by itself after [`SHORT`] tokens, with no abort.
async fn hold_a(h: &Harness) -> (String, SseReader) {
    let (rid, worker, reader) = open_stream(h.chat(&body(true, SHORT))).await;
    assert_eq!(worker, A, "an idle tie goes to the lowest worker id");
    (rid, reader)
}

/// Reads an opened stream (its `accepted` event consumed) to the end: it finished by its budget on `worker`.
async fn finishes(h: &Harness, rid: &str, worker: &str, mut reader: SseReader) {
    let events = reader.rest().await;
    let terminal = check_stream_tail(&events, 0);
    assert_eq!(
        terminal["choices"][0]["finish_reason"], "length",
        "{terminal}"
    );
    assert_eq!(terminal["cere"]["state"], "completed", "{terminal}");
    let status = h.terminal(rid).await;
    assert_eq!(status["state"], "completed", "{status}");
    assert_eq!(status["worker_id"], worker, "{status}");
}

/// A long stream on B, with A's holding stream already finished: A has no request, B one.
async fn stream_on_b(h: &Harness, request: reqwest::RequestBuilder) -> (String, SseReader) {
    let (held, holder) = hold_a(h).await;
    let (rid, worker, reader) = open_stream(request).await;
    assert_eq!(worker, B, "A was busy");
    finishes(h, &held, A, holder).await;
    assert_eq!(h.replica(1).in_flight(), 1);
    assert_eq!(h.mock.in_flight(), 0);
    (rid, reader)
}

/// The isolation every abort test ends with: `rid` ran on B only, B got exactly its abort, A got none.
fn aborted_on_b_only(h: &Harness, rid: &str) {
    assert!(dispatched(h.replica(1)).contains(rid));
    assert!(!dispatched(&h.mock).contains(rid));
    assert_eq!(aborted(h.replica(1)), [rid]);
    assert_eq!(aborted(&h.mock), Vec::<String>::new());
}

async fn wait_aborted(mock: &MockHandle, rid: &str) {
    for _ in 0..500 {
        if mock.abort_requests().iter().any(|a| a.rid == rid) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("{rid} was never aborted");
}

/// Streams and unary requests at once: each runs on the worker named in its response, that worker's mock saw its
/// rid and the other mock never did (no request is answered by two workers); both workers get work.
#[tokio::test]
async fn c6_requests_land_on_the_granted_worker_only() {
    let h = Harness::start_replicas(slow_mocks().await, &WIDE_CAPS).await;
    let smokes = [dispatched(&h.mock), dispatched(h.replica(1))];
    let mut streams = Vec::new();
    for _ in 0..3 {
        streams.push(open_stream(h.chat(&body(true, 10))).await);
    }
    let unary: Vec<_> = (0..3)
        .map(|_| tokio::spawn(h.chat(&body(false, 10)).send()))
        .collect();
    let mut granted = Vec::new();
    for (rid, worker, mut reader) in streams {
        let terminal = check_stream_tail(&reader.rest().await, 0);
        assert_eq!(terminal["cere"]["worker_id"], worker.as_str(), "{terminal}");
        granted.push((rid, worker));
    }
    for response in futures_util::future::join_all(unary).await {
        let response = response.unwrap().unwrap();
        assert_eq!(response.status(), 200);
        let reply: Value = response.json().await.unwrap();
        let cere = &reply["cere"];
        assert_eq!(cere["state"], "completed", "{reply}");
        granted.push((
            cere["request_id"].as_str().unwrap().to_owned(),
            cere["worker_id"].as_str().unwrap().to_owned(),
        ));
    }
    let seen = [dispatched(&h.mock), dispatched(h.replica(1))];
    for (rid, worker) in &granted {
        let (on, off) = match worker.as_str() {
            A => (0, 1),
            B => (1, 0),
            other => panic!("unknown worker {other}"),
        };
        assert!(seen[on].contains(rid), "{rid} not on its worker {worker}");
        assert!(
            !seen[off].contains(rid),
            "{rid} also reached the other worker"
        );
        assert_eq!(h.terminal(rid).await["worker_id"], worker.as_str());
    }
    for (i, worker) in [A, B].into_iter().enumerate() {
        let mine: BTreeSet<&String> = granted
            .iter()
            .filter(|(_, w)| w == worker)
            .map(|(rid, _)| rid)
            .collect();
        assert!(!mine.is_empty(), "no request ran on {worker}");
        let fresh: BTreeSet<&String> = seen[i].difference(&smokes[i]).collect();
        assert_eq!(fresh, mine, "{worker} saw exactly its granted requests");
    }
}

/// `POST /v1/cere/requests/{id}/cancel` of a stream on B aborts it on B only.
#[tokio::test]
async fn c6_cancel_on_b_aborts_on_b_only() {
    let h = Harness::start_replicas(slow_mocks().await, &WIDE_CAPS).await;
    let (rid, mut reader) = stream_on_b(&h, h.chat(&body(true, LONG))).await;
    let cancel = h
        .post(&format!("/v1/cere/requests/{rid}/cancel"), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert_eq!(cancel.status(), 202);
    let terminal = check_stream_tail(&reader.rest().await, 0);
    assert_eq!(terminal["cere"]["state"], "cancelled", "{terminal}");
    assert_eq!(terminal["cere"]["reason"], "cancel_requested", "{terminal}");
    aborted_on_b_only(&h, &rid);
    assert_eq!(h.terminal(&rid).await["worker_id"], B);
}

/// A client that goes away from a stream on B gets it aborted on B only.
#[tokio::test]
async fn c6_client_disconnect_on_b_aborts_on_b_only() {
    let h = Harness::start_replicas(slow_mocks().await, &WIDE_CAPS).await;
    let (rid, reader) = stream_on_b(&h, h.chat(&body(true, LONG))).await;
    drop(reader);
    wait_aborted(h.replica(1), &rid).await;
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "cancelled", "{status}");
    assert_eq!(status["reason"], "client_disconnected", "{status}");
    aborted_on_b_only(&h, &rid);
    assert_eq!(status["worker_id"], B, "{status}");
}

/// A request on B that runs past its deadline (unary: the executor's drive loop) is aborted on B only.
#[tokio::test]
async fn c6_deadline_on_b_aborts_on_b_only() {
    let h = Harness::start_replicas(slow_mocks().await, &WIDE_CAPS).await;
    let (held, holder) = hold_a(&h).await;
    let request = h
        .chat(&body(false, LONG))
        .header("X-Cere-Deadline-Ms", "1500")
        .send();
    let call = tokio::spawn(request);
    for _ in 0..200 {
        if h.replica(1).in_flight() == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(h.replica(1).in_flight(), 1, "the request did not reach B");
    finishes(&h, &held, A, holder).await;
    let response = call.await.unwrap().unwrap();
    assert_eq!(response.status(), 504);
    let reply: Value = response.json().await.unwrap();
    let cere = &reply["cere"];
    assert_eq!(cere["state"], "cancelled", "{reply}");
    assert_eq!(cere["reason"], "deadline_exceeded", "{reply}");
    assert_eq!(cere["worker_id"], B, "{reply}");
    aborted_on_b_only(&h, cere["request_id"].as_str().unwrap());
}

/// The L8 shutdown (SIGTERM's path) with a stream open on B aborts it on B only and stops clean.
#[tokio::test]
async fn c6_shutdown_on_b_aborts_on_b_only() {
    let mut h = Harness::start_replicas(
        slow_mocks().await,
        &[WIDE_CAPS[0], WIDE_CAPS[1], DRAIN_300MS],
    )
    .await;
    let (rid, mut reader) = stream_on_b(&h, h.chat(&body(true, LONG))).await;
    let rest = tokio::spawn(async move { reader.rest().await });
    assert!(h.stop().await, "shutdown was forced");
    let terminal = check_stream_tail(&rest.await.unwrap(), 0);
    assert_eq!(terminal["cere"]["state"], "cancelled", "{terminal}");
    assert_eq!(terminal["cere"]["reason"], "shutdown", "{terminal}");
    aborted_on_b_only(&h, &rid);
    let records = journal(h.state_dir.path());
    assert!(non_terminal(&records).is_empty());
}

/// Makes worker `index` forget its backend identity, then probes it again: the probe smokes, under an admission
/// slot of that worker (readiness waits for it like any request).
async fn resmoke(h: &Harness, index: usize) {
    let mock = h.replica(index);
    mock.set_server_info_override(Some(ServerInfo {
        max_running_requests: 2,
        kv_cache_dtype: "auto".to_owned(),
        ..ServerInfo::default()
    }));
    assert!(!h.app.readiness(index).probe().await.ready);
    mock.set_server_info_override(None);
    let probed = h.app.readiness(index).probe().await;
    assert!(probed.ready, "{:?}", probed.problems);
}

/// More concurrent requests (two scopes) than the worker caps (2 each; the global cap may not exceed their sum)
/// together, with a re-smoke of each worker while the queue is full (the other worker takes all grants then):
/// each mock saw at most its cap at any time, and reached it, smokes included.
#[tokio::test]
async fn c6_saturation_keeps_each_mock_within_its_cap() {
    let mut mocks = Vec::new();
    for _ in 0..2 {
        mocks.push(
            MockSglang::start(MockConfig {
                server_info: ServerInfo {
                    max_running_requests: 2,
                    ..ServerInfo::default()
                },
                ..slow(40, 10)
            })
            .await
            .unwrap(),
        );
    }
    let tweaks = [
        ("max_backend_inflight = 2", "max_backend_inflight = 4"),
        ("max_running_requests = 4", "max_running_requests = 2"),
    ];
    let h = Harness::start_replicas(mocks, &tweaks).await;
    let calls: Vec<_> = (0..16)
        .map(|i| {
            let token = if i % 2 == 0 { TOKEN_A } else { TOKEN_B };
            tokio::spawn(
                h.post("/v1/chat/completions", token)
                    .json(&body(false, 10))
                    .send(),
            )
        })
        .collect();
    for _ in 0..400 {
        let s = h.app.admission_snapshot();
        if s.inflight == 4 && s.queued > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let s = h.app.admission_snapshot();
    assert!(s.inflight == 4 && s.queued > 0, "never saturated: {s:?}");
    resmoke(&h, 0).await;
    resmoke(&h, 1).await;
    let mut rids = BTreeSet::new();
    for call in calls {
        let response = call.await.unwrap().unwrap();
        assert_eq!(response.status(), 200);
        let reply: Value = response.json().await.unwrap();
        rids.insert(reply["cere"]["request_id"].as_str().unwrap().to_owned());
    }
    assert_eq!(rids.len(), 16);
    for i in 0..2 {
        let mock = h.replica(i);
        assert_eq!(
            mock.max_in_flight(),
            2,
            "mock {i} above (or never at) its cap"
        );
        let smokes = dispatched(mock).difference(&rids).count();
        assert!(smokes >= 2, "mock {i}: the start smoke and the re-smoke");
    }
}

/// Mock A shut down mid-stream: A's streams fail through their own stream (the connection dropped mid-body:
/// `backend_protocol_error`, Q10 "as today"), B's run to their end, new work goes to B, `/readyz` stays 200 with
/// A suspect, and A is lost after `scheduler.worker_lost_after_ms`. The probe period is the default test one (5 s),
/// so neither worker ages suspect (15 s unseen) while the streams open, however slow the host. The test probes A
/// itself just before and just after the shutdown, so A reads suspect for the 16 s after the healthy probe and the
/// suspect checks come first, then the streams' ends, then A's aging to lost.
#[tokio::test]
async fn c6_mock_a_down_mid_stream_fails_only_a() {
    let mut h = Harness::start_replicas(
        slow_mocks().await,
        &[WIDE_CAPS[0], WIDE_CAPS[1], LOST_AFTER_1S],
    )
    .await;
    // Least loaded, ties by id: A, B, A, B. B's streams end by their budget (~1 s); A's would run for minutes.
    let mut a_streams = Vec::new();
    let mut b_streams = Vec::new();
    for i in 0..4 {
        let tokens = if i % 2 == 0 { LONG } else { 50 };
        let (rid, worker, mut reader) = open_stream(h.chat(&body(true, tokens))).await;
        let token = reader.next().await.unwrap();
        let seq = token.data()["cere"]["seq"].as_u64().unwrap();
        let expected = if i % 2 == 0 { A } else { B };
        assert_eq!(worker, expected, "stream {i}");
        if i % 2 == 0 {
            a_streams.push((rid, reader, seq));
        } else {
            b_streams.push((rid, reader, seq));
        }
    }
    // A healthy probe of A just before the shutdown: A ages lost 16 s after it.
    assert!(h.app.readiness(0).probe().await.ready);
    let a = std::mem::replace(
        &mut h.mock,
        MockSglang::start(MockConfig::default()).await.unwrap(),
    );
    a.shutdown().await.unwrap();

    // A's next probe fails: suspect at once (no aging needed).
    assert!(!h.app.readiness(0).probe().await.ready);
    let (status, ready) = readyz_until(&h, "gpu0 suspect", |b| {
        b["workers"][A]["state"] == "suspect"
    })
    .await;
    assert_eq!(status, 200, "{ready}");
    assert_eq!(ready["ready"], true, "{ready}");
    assert_eq!(ready["workers"][B]["state"], "ready", "{ready}");
    assert_eq!(ready["workers"][A]["ready"], false, "{ready}");

    let before = dispatched(h.replica(1));
    for _ in 0..3 {
        let response = h.chat(&body(false, 5)).send().await.unwrap();
        assert_eq!(response.status(), 200);
        let reply: Value = response.json().await.unwrap();
        assert_eq!(reply["cere"]["worker_id"], B, "{reply}");
        let rid = reply["cere"]["request_id"].as_str().unwrap();
        assert!(dispatched(h.replica(1)).contains(rid));
        assert!(!before.contains(rid));
    }

    for (rid, mut reader, seq) in a_streams {
        let terminal = check_stream_tail(&reader.rest().await, seq);
        assert_eq!(terminal["cere"]["state"], "failed", "{terminal}");
        assert_eq!(
            terminal["cere"]["reason"], "backend_protocol_error",
            "{terminal}"
        );
        assert_eq!(
            terminal["error"]["code"], "backend_protocol_error",
            "{terminal}"
        );
        let status = h.terminal(&rid).await;
        assert_eq!(status["state"], "failed", "{status}");
        assert_eq!(status["worker_id"], A, "{status}");
    }
    for (rid, mut reader, seq) in b_streams {
        let terminal = check_stream_tail(&reader.rest().await, seq);
        assert_eq!(
            terminal["choices"][0]["finish_reason"], "length",
            "{terminal}"
        );
        assert_eq!(terminal["cere"]["state"], "completed", "{terminal}");
        assert_eq!(h.terminal(&rid).await["worker_id"], B);
    }
    assert_eq!(aborted(h.replica(1)), Vec::<String>::new());

    // Lost once unseen for the suspect bound (`suspect_after`) plus `worker_lost_after_ms` (1 s): within 16 s of
    // the healthy probe before the shutdown.
    let lost_by = Instant::now() + test_tuning().suspect_after + Duration::from_secs(5);
    let ready = loop {
        let (status, ready) = readyz(&h).await;
        assert_eq!(status, 200, "{ready}");
        if ready["workers"][A]["state"] == "lost" {
            break ready;
        }
        assert!(
            Instant::now() < lost_by,
            "/readyz never showed gpu0 lost: {ready}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert_eq!(ready["workers"][B]["state"], "ready", "{ready}");
}

/// Mock A gone before its next probe (5 s period): A is still ready, so the next request is granted A (least
/// loaded, lowest id) and fails `backend_unavailable` (the connection is refused) on A, through its own response.
/// Nothing is retried on B (Q10).
#[tokio::test]
async fn c6_new_work_on_a_down_fails_unavailable_and_is_not_retried_on_b() {
    let mut h = Harness::start_replicas(slow_mocks().await, &WIDE_CAPS).await;
    let b_before = dispatched(h.replica(1));
    let a = std::mem::replace(
        &mut h.mock,
        MockSglang::start(MockConfig::default()).await.unwrap(),
    );
    a.shutdown().await.unwrap();
    let response = h.chat(&body(false, 5)).send().await.unwrap();
    assert_eq!(response.status(), 503);
    let reply: Value = response.json().await.unwrap();
    assert_eq!(reply["error"]["code"], "backend_unavailable", "{reply}");
    let cere = &reply["cere"];
    assert_eq!(cere["state"], "failed", "{reply}");
    assert_eq!(cere["reason"], "backend_unavailable", "{reply}");
    assert_eq!(cere["worker_id"], A, "{reply}");
    assert_eq!(dispatched(h.replica(1)), b_before);
    assert_eq!(aborted(h.replica(1)), Vec::<String>::new());
    let rid = cere["request_id"].as_str().unwrap();
    assert_eq!(h.terminal(rid).await["worker_id"], A);
}

/// Polls the binary's `/readyz` until both workers are ready.
async fn both_ready(cered: &Cered) {
    let http = http();
    for _ in 0..500 {
        if let Ok(r) = http.get(cered.url("/readyz")).send().await {
            let body: Value = r.json().await.unwrap_or_default();
            if body["workers"][A]["ready"] == true && body["workers"][B]["ready"] == true {
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("the workers never both became ready");
}

/// The real binary on A and B, started over a recovered orphan of A (its abort sent, its slot retained for the
/// retention bound), with one stream on B (A is the more loaded worker) when SIGTERM arrives; `b_fault` is B's
/// fault for that stream. Returns the exit code, stderr, the orphan's and the stream's rids and terminal event,
/// the state dir and both mocks.
struct SigtermRun {
    code: Option<i32>,
    stderr: String,
    orphan: String,
    rid: String,
    terminal: Value,
    state: StateDir,
    a: MockHandle,
    b: MockHandle,
}

async fn sigterm_with_orphan_on_a(b_fault: Fault) -> SigtermRun {
    let a = MockSglang::start(MockConfig::default()).await.unwrap();
    let b = MockSglang::start(slow(50, 2_000)).await.unwrap();
    let state = StateDir::new();
    let text = replicas_config_text(
        &[a.base_url(), b.base_url()],
        state.path(),
        free_port(),
        &[DRAIN_300MS],
    );
    let config = ConfigFile::new(&text);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let parsed = RuntimeConfig::from_toml_str(&text, &repo_root()).unwrap();
    let orphan = seed_orphan(&parsed, "b", now);
    let mut cered = Cered::spawn(&config.path, &[]).await.unwrap();
    both_ready(&cered).await;
    wait_aborted(&a, &orphan).await;
    b.set_fault(b_fault);
    let request = http()
        .post(cered.url("/v1/chat/completions"))
        .bearer_auth(TOKEN_A)
        .json(&body(true, 2_000));
    let (rid, worker, mut reader) = open_stream(request).await;
    assert_eq!(worker, B, "A holds the orphan's slot");
    assert!(reader.next().await.is_some());
    cered.signal("TERM");
    let events = reader.rest().await;
    assert_eq!(events.last(), Some(&Event::Done));
    let terminal = events[events.len() - 2].data().clone();
    let (code, stderr) = cered.exit(Duration::from_secs(20)).await;
    SigtermRun {
        code,
        stderr,
        orphan,
        rid,
        terminal,
        state,
        a,
        b,
    }
}

/// SIGTERM with a stream on B: its abort goes to B only (A only ever got the orphan's recovery abort) and the
/// stop is clean: B's stream closed and A's retained orphan is not an open stream of A.
#[tokio::test(flavor = "multi_thread")]
async fn c6_sigterm_aborts_on_b_only_and_a_retained_orphan_is_not_open() {
    let run = sigterm_with_orphan_on_a(Fault::None).await;
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    assert!(
        run.stderr
            .contains("cered shutdown clean=true forced_terminals=0 open_backend_streams=0"),
        "{}",
        run.stderr
    );
    assert_eq!(
        run.terminal["cere"]["reason"], "shutdown",
        "{}",
        run.terminal
    );
    assert_eq!(aborted(&run.b), [run.rid.as_str()]);
    assert_eq!(aborted(&run.a), [run.orphan.as_str()]);
    assert!(non_terminal(&journal(run.state.path())).is_empty());
}

/// SIGTERM with a stream on B whose backend ignores the abort: the stop is not clean and reports exactly B's
/// open stream (A's retained orphan does not count, B's stream is not hidden behind A).
#[tokio::test(flavor = "multi_thread")]
async fn c6_sigterm_counts_open_backend_streams_per_worker() {
    let run = sigterm_with_orphan_on_a(Fault::IgnoreAbort).await;
    assert_eq!(run.code, Some(1), "{}", run.stderr);
    assert!(
        run.stderr
            .contains("cered shutdown clean=false forced_terminals=0 open_backend_streams=1"),
        "{}",
        run.stderr
    );
    assert_eq!(
        run.terminal["cere"]["reason"], "cancel_unacknowledged",
        "{}",
        run.terminal
    );
    assert_eq!(aborted(&run.b), [run.rid.as_str()]);
    assert_eq!(aborted(&run.a), [run.orphan.as_str()]);
    let records = journal(run.state.path());
    assert!(non_terminal(&records).is_empty());
    let terminal = records
        .iter()
        .find(|r| r["type"] == "terminal" && r["rid"] == run.rid.as_str())
        .expect("terminal record");
    assert_eq!(terminal["reason"], "cancel_unacknowledged", "{terminal}");
}
