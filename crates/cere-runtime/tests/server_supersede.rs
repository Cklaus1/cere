//! M3 Q10 epoch supersede (m3 design note C6): when a worker's epoch moves on (a relaunch, or a rejoin after
//! it was lost), every request still in flight on its old epoch fails `worker_epoch_superseded` exactly once,
//! its `/abort_request` reaches that worker exactly once, and the other worker's requests are untouched. After
//! a relaunch the rid's slot is released at once (journaled `orphan_released` cause `relaunch`, never charged
//! again by a restart); after a rejoin the same backend may still run it, so it stays charged to the worker
//! until its release, like a recovered orphan.

mod support;

use std::time::Duration;

use cere_mock_sglang::{MockConfig, MockHandle, MockSglang, ServerInfo};
use cere_runtime::app::Tuning;
use serde_json::Value;
use support::*;

const SUPERSEDED: &str = "worker_epoch_superseded";

/// Room for four requests at once, two per worker (each worker's cap is 4).
const INFLIGHT_8: [(&str, &str); 2] = [
    ("max_backend_inflight = 2", "max_backend_inflight = 8"),
    ("per_scope_inflight = 2", "per_scope_inflight = 8"),
];

/// `scheduler.worker_lost_after_ms` at its 1 s floor.
const LOST_AFTER_1S: (&str, &str) = (
    "mode = \"replica\"",
    "mode = \"replica\"\nworker_lost_after_ms = 1000",
);

/// `scheduler.worker_lost_after_ms` = 3 s.
const LOST_AFTER_3S: (&str, &str) = (
    "mode = \"replica\"",
    "mode = \"replica\"\nworker_lost_after_ms = 3000",
);

const RETENTION_6S: (&str, &str) = (
    "terminal_retention_ms = 3600000",
    "terminal_retention_ms = 6000",
);

/// A mock generating `tokens` tokens 20 ms apart (2 000 tokens: 40 s, longer than any test waits).
async fn mock(tokens: u32) -> MockHandle {
    MockSglang::start(MockConfig {
        token_latency: Duration::from_millis(20),
        natural_output_tokens: tokens,
        ..MockConfig::default()
    })
    .await
    .unwrap()
}

fn long_body(stream: bool) -> Value {
    let mut body = chat_body(stream);
    body["max_completion_tokens"] = 2_000.into();
    body
}

/// Opens a stream as scope `a`; returns its rid, the worker its `accepted` event names, and the reader
/// positioned after that event.
async fn open_stream(h: &Harness) -> (String, String, SseReader) {
    let response = h.chat(&long_body(true)).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let mut reader = SseReader::new(response);
    let accepted = reader.next().await.expect("the accepted event");
    let cere = &accepted.data()["cere"];
    assert_eq!(cere["event"], "accepted", "{cere}");
    let field = |name: &str| cere[name].as_str().unwrap().to_owned();
    (field("request_id"), field("worker_id"), reader)
}

/// The events after `accepted` up to the end of the body (10 s bound).
async fn rest(reader: &mut SseReader, what: &str) -> Vec<Event> {
    tokio::time::timeout(Duration::from_secs(10), reader.rest())
        .await
        .unwrap_or_else(|_| panic!("{what} did not end"))
}

/// The first rid `mock` received that is not in `before` (5 s bound).
async fn new_rid(mock: &MockHandle, before: &[String]) -> String {
    for _ in 0..500 {
        if let Some(rid) = mock
            .dispatched_rids()
            .into_iter()
            .find(|r| !before.contains(r))
        {
            return rid;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the mock never received a new request");
}

/// The rids of every `/abort_request` `mock` received, sorted.
fn aborted(mock: &MockHandle) -> Vec<String> {
    let mut rids: Vec<String> = mock.abort_requests().into_iter().map(|a| a.rid).collect();
    rids.sort();
    rids
}

/// Waits (5 s bound) until `mock` received an abort for each of `rids`, then for twice the cancel ack bound its
/// tests run under ([`short_ack`]), so a second abort for any of them would have arrived; returns every aborted
/// rid, sorted.
async fn settled_aborts(mock: &MockHandle, rids: &[&str]) -> Vec<String> {
    for _ in 0..500 {
        let got = aborted(mock);
        if rids.iter().all(|r| got.iter().any(|g| g == r)) {
            tokio::time::sleep(2 * SHORT_CANCEL_ACK).await;
            return aborted(mock);
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("aborts of {rids:?} never arrived: {:?}", aborted(mock));
}

/// Journal records of `kind` for `rid`.
fn records(h: &Harness, kind: &str, rid: &str) -> Vec<Value> {
    journal(h.state_dir.path())
        .into_iter()
        .filter(|r| r["type"] == kind && r["rid"] == rid)
        .collect()
}

/// Asserts the status API and the journal hold exactly one terminal for `rid`: `failed` with
/// `worker_epoch_superseded`, on `gpu0` at its old epoch 1.
async fn assert_superseded(h: &Harness, rid: &str) {
    let (code, status) = h.status(rid, TOKEN_A).await;
    assert_eq!(code, 200, "{status}");
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], SUPERSEDED, "{status}");
    assert_eq!(status["worker_id"], "gpu0", "{status}");
    assert_eq!(status["worker_epoch"], 1, "{status}");
    let terminals = records(h, "terminal", rid);
    assert_eq!(terminals.len(), 1, "one terminal for {rid}: {terminals:?}");
    assert_eq!(terminals[0]["state"], "failed", "{terminals:?}");
    assert_eq!(terminals[0]["reason"], SUPERSEDED, "{terminals:?}");
}

/// Asserts a stream's terminal is the `worker_epoch_superseded` error event, naming gpu0 at its old epoch 1.
fn assert_superseded_event(events: &[Event]) {
    let terminal = check_stream_tail(events, 0);
    assert_eq!(terminal["error"]["code"], SUPERSEDED, "{terminal}");
    assert_eq!(terminal["cere"]["state"], "failed", "{terminal}");
    assert_eq!(terminal["cere"]["reason"], SUPERSEDED, "{terminal}");
    assert_eq!(terminal["cere"]["worker_id"], "gpu0", "{terminal}");
    assert_eq!(terminal["cere"]["worker_epoch"], 1, "{terminal}");
}

/// `GET /v1/cere/stats` as scope `a`: `(cap, inflight)` of `worker`.
async fn worker_stats(h: &Harness, worker: &str) -> (u64, u64) {
    let body: Value = h
        .get("/v1/cere/stats", TOKEN_A)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let w = &body["workers"][worker];
    let field = |name: &str| {
        w[name]
            .as_u64()
            .unwrap_or_else(|| panic!("{worker}.{name}: {body}"))
    };
    (field("cap"), field("inflight"))
}

/// A relaunch of gpu0 (same URL, new `instance_id`) supersedes its epoch 1: its in-flight stream and unary end
/// `worker_epoch_superseded` (SSE error event, 503), each rid is aborted on gpu0 exactly once, both are released
/// at once (`orphan_released` cause `relaunch`), and gpu1's two streams finish normally.
#[tokio::test(flavor = "multi_thread")]
async fn relaunch_supersedes_only_that_workers_in_flight_requests() {
    let h = Harness::start_replicas_tuned(
        vec![mock(2_000).await, mock(150).await],
        &INFLIGHT_8,
        short_ack(fast_tuning()),
    )
    .await;
    // Least in flight, ties by id: gpu0, gpu1, gpu0, gpu1, each charged before the next is sent.
    let (a_stream, worker, mut a_reader) = open_stream(&h).await;
    assert_eq!(worker, "gpu0");
    let (b_first, worker, mut b_first_reader) = open_stream(&h).await;
    assert_eq!(worker, "gpu1");
    let before = h.mock.dispatched_rids();
    let unary = tokio::spawn(h.chat(&long_body(false)).send());
    let a_unary = new_rid(&h.mock, &before).await;
    let (b_second, worker, mut b_second_reader) = open_stream(&h).await;
    assert_eq!(worker, "gpu1");
    let (_, body) = readyz(&h).await;
    assert_eq!(body["workers"]["gpu0"]["epoch"], 1, "{body}");

    h.mock.set_server_info_override(Some(launched(7, 12.5)));

    let events = rest(&mut a_reader, "the gpu0 stream").await;
    assert_superseded_event(&events);
    let response = tokio::time::timeout(Duration::from_secs(10), unary)
        .await
        .expect("the gpu0 unary did not end")
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), 503);
    assert_eq!(
        response.headers()["x-cere-request-id"].to_str().unwrap(),
        a_unary
    );
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], SUPERSEDED, "{body}");
    assert_eq!(body["cere"]["state"], "failed", "{body}");
    assert_eq!(body["cere"]["reason"], SUPERSEDED, "{body}");
    assert_eq!(body["cere"]["worker_id"], "gpu0", "{body}");
    assert_eq!(body["cere"]["worker_epoch"], 1, "{body}");
    // gpu1's streams were running through the supersede and finish normally.
    for rid in [&b_first, &b_second] {
        let (_, status) = h.status(rid, TOKEN_A).await;
        assert_eq!(status["state"], "running", "{status}");
    }
    for reader in [&mut b_first_reader, &mut b_second_reader] {
        let events = rest(reader, "a gpu1 stream").await;
        let terminal = check_stream_tail(&events, 0);
        assert_eq!(
            terminal["choices"][0]["finish_reason"], "stop",
            "{terminal}"
        );
        assert_eq!(terminal["cere"]["state"], "completed", "{terminal}");
        assert_eq!(terminal["cere"]["worker_id"], "gpu1", "{terminal}");
    }

    let mut expected = vec![a_stream.clone(), a_unary.clone()];
    expected.sort();
    assert_eq!(
        settled_aborts(&h.mock, &[&a_stream, &a_unary]).await,
        expected
    );
    assert_eq!(aborted(h.replica(1)), Vec::<String>::new());
    for rid in [&a_stream, &a_unary] {
        assert_superseded(&h, rid).await;
        let released = records(&h, "orphan_released", rid);
        assert_eq!(released.len(), 1, "{released:?}");
        assert_eq!(released[0]["cause"], "relaunch", "{released:?}");
    }
    for rid in [&b_first, &b_second] {
        assert_eq!(h.terminal(rid).await["state"], "completed");
        assert!(records(&h, "orphan_released", rid).is_empty());
    }
    let (_, body) = readyz_until(&h, "gpu0 ready at epoch 2", |b| {
        b["workers"]["gpu0"]["epoch"] == 2 && b["workers"]["gpu0"]["state"] == "ready"
    })
    .await;
    assert_eq!(body["workers"]["gpu1"]["epoch"], 1, "{body}");
    // Nothing stays charged: the relaunched backend holds none of the old epoch's work.
    for _ in 0..500 {
        if h.app.admission_snapshot().inflight == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(h.app.admission_snapshot().inflight, 0);

    let h = h.restart().await;
    assert_eq!(
        h.app.admission_snapshot().inflight,
        0,
        "a released superseded rid was charged again"
    );
}

/// gpu0 is lost (drift past `worker_lost_after_ms`) while a stream runs there, then rejoins with the same
/// backend at epoch 2: the stream ends `worker_epoch_superseded`, its abort reaches gpu0 once, and its slot
/// stays charged to gpu0 after the executor let go of it. The hold counts as retained (aborted, waiting only
/// for its release), so a shutdown meanwhile is clean; the next incarnation charges it to gpu0 again until the
/// retention bound since its admission releases it.
#[tokio::test(flavor = "multi_thread")]
async fn rejoin_keeps_the_superseded_rid_charged_to_its_worker_until_released() {
    let mut h = Harness::start_replicas_tuned(
        vec![mock(2_000).await, mock(2_000).await],
        &[LOST_AFTER_1S, RETENTION_6S],
        short_ack(probe_aging(fast_tuning())),
    )
    .await;
    let (rid, worker, mut reader) = open_stream(&h).await;
    assert_eq!(worker, "gpu0");

    h.mock
        .set_server_info_override(Some(cere_mock_sglang::ServerInfo {
            kv_cache_dtype: "bf16".to_owned(),
            ..cere_mock_sglang::ServerInfo::default()
        }));
    readyz_until(&h, "gpu0 lost", |b| b["workers"]["gpu0"]["state"] == "lost").await;
    h.mock.set_server_info_override(None);

    let events = rest(&mut reader, "the gpu0 stream").await;
    assert_superseded_event(&events);
    assert_eq!(settled_aborts(&h.mock, &[&rid]).await, [rid.clone()]);
    assert_superseded(&h, &rid).await;
    readyz_until(&h, "gpu0 rejoined at epoch 2", |b| {
        b["workers"]["gpu0"]["epoch"] == 2 && b["workers"]["gpu0"]["state"] == "ready"
    })
    .await;
    // The backend generation ended (the executor's slot is gone with it); the rid is still charged to gpu0.
    for _ in 0..500 {
        if h.mock.in_flight() == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(h.mock.in_flight(), 0);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        records(&h, "orphan_released", &rid).is_empty(),
        "released before the check"
    );
    assert_eq!(worker_stats(&h, "gpu0").await, (4, 1));
    assert_eq!(worker_stats(&h, "gpu1").await, (4, 0));
    assert_eq!(h.app.admission_snapshot().inflight, 1);

    assert!(
        h.stop().await,
        "the held superseded rid counted as an open backend stream"
    );
    let h = h.restart().await;
    assert_eq!(worker_stats(&h, "gpu0").await, (4, 1));
    let admitted = records(&h, "admitted", &rid)[0]["at_ms"].as_u64().unwrap();
    let mut released = Vec::new();
    for _ in 0..800 {
        released = records(&h, "orphan_released", &rid);
        if !released.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(released.len(), 1, "never released: {released:?}");
    assert_eq!(released[0]["cause"], "retention", "{released:?}");
    assert!(released[0]["at_ms"].as_u64().unwrap() >= admitted + 6_000);
    for _ in 0..100 {
        if h.app.admission_snapshot().inflight == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(worker_stats(&h, "gpu0").await, (4, 0));
}

/// Review A1 (Q4 rule 3), pass 2: a Suspect worker whose probes confirm the same `instance_id` keeps its epoch
/// while its re-smoke waits for a slot, also when the configured smoke deadline (10 s, as production's 60 s vs a
/// 15 s + `worker_lost_after_ms` bound) outlasts the suspect + lost bound (300 ms + 3 s). gpu0 (cap 1) runs a 4 s
/// stream whose finish the mock holds; one drifted probe resets its smoke, and the re-smoke cannot get gpu0's only
/// slot while the stream holds it. Each re-smoke gives up within half that bound (1.65 s) and the probe publishes
/// a sighting: gpu0 stays suspect, never ages lost, never rejoins at epoch 2. The stream ends right after a
/// re-smoke starts waiting, so that smoke gets the slot with nearly all of its 1.65 s, runs, and gpu0 is ready
/// again at epoch 1 with nothing aborted. The lost bound is 3 s, not the 1 s floor: the first sighting after the
/// drift comes one drift window plus one capped smoke after the last one, and the other half of a 1.3 s bound
/// (650 ms for the drift window and the probe's own requests) is not enough on a loaded host.
#[tokio::test(flavor = "multi_thread")]
async fn a_suspect_worker_seen_with_its_instance_keeps_its_epoch_while_its_smoke_waits() {
    let cap_1 = || async {
        MockSglang::start(MockConfig {
            token_latency: Duration::from_millis(20),
            natural_output_tokens: 200,
            server_info: ServerInfo {
                max_running_requests: 1,
                ..ServerInfo::default()
            },
            ..MockConfig::default()
        })
        .await
        .unwrap()
    };
    let h = Harness::start_replicas_tuned(
        vec![cap_1().await, cap_1().await],
        &[
            ("max_running_requests = 4", "max_running_requests = 1"),
            LOST_AFTER_3S,
        ],
        Tuning {
            smoke_deadline: Duration::from_secs(10),
            ..probe_aging(fast_tuning())
        },
    )
    .await;
    // The stream ends when the test lets it (below), not wherever in a re-smoke's wait its last token falls: a
    // smoke granted the slot with less of its deadline left than its chat takes times out and aborts its rid.
    h.mock.hold_finish(true);
    let (rid, worker, mut reader) = open_stream(&h).await;
    assert_eq!(worker, "gpu0");

    // A loaded host (gate-98b): a probe outlasts gpu0's 300 ms suspect bound, so gpu0 reads suspect by aging, no
    // problem published, while that probe is still in flight. The drift below is set in that window, so "suspect"
    // alone does not say a probe read it: the wait is for the drift's own problem, after which the smoke is reset.
    h.mock.set_info_delay(Duration::from_millis(500));
    readyz_until(&h, "gpu0 suspect by aging", |b| {
        let gpu0 = &b["workers"]["gpu0"];
        gpu0["state"] == "suspect" && gpu0["problems"].as_array().is_some_and(Vec::is_empty)
    })
    .await;
    h.mock.set_info_delay(Duration::ZERO);
    h.mock.set_server_info_override(Some(ServerInfo {
        max_running_requests: 1,
        kv_cache_dtype: "bf16".to_owned(),
        ..ServerInfo::default()
    }));
    readyz_until(&h, "gpu0 suspect on the drift", |b| {
        let gpu0 = &b["workers"]["gpu0"];
        gpu0["state"] == "suspect"
            && gpu0["problems"]
                .as_array()
                .is_some_and(|p| p.iter().any(|p| p["field"] == "kv_cache_dtype"))
    })
    .await;
    h.mock.set_server_info_override(None);

    let held = tokio::time::Instant::now() + Duration::from_secs(30);
    while h.mock.finish_held() == 0 {
        assert!(
            tokio::time::Instant::now() < held,
            "the gpu0 stream never reached its finish"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // A probe publishes only after its re-smoke gave up, so a re-smoke queued after a later publish than this one
    // has just started its wait.
    let published = h.app.readiness(0).snapshot().probed_at_unix_ms;
    let fresh = tokio::time::Instant::now() + Duration::from_secs(10);
    while h.app.readiness(0).snapshot().probed_at_unix_ms == published
        || h.app.admission_snapshot().queued == 0
    {
        assert!(
            tokio::time::Instant::now() < fresh,
            "no re-smoke waited for gpu0's slot"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    h.mock.hold_finish(false);
    let events = rest(&mut reader, "the gpu0 stream").await;
    let terminal = check_stream_tail(&events, 0);
    assert_eq!(
        terminal["choices"][0]["finish_reason"], "stop",
        "{terminal}"
    );
    let (code, status) = h.status(&rid, TOKEN_A).await;
    assert_eq!(code, 200, "{status}");
    assert_eq!(status["state"], "completed", "{status}");
    assert_eq!(status["worker_epoch"], 1, "{status}");
    readyz_until(&h, "gpu0 ready at epoch 1", |b| {
        b["workers"]["gpu0"]["state"] == "ready"
    })
    .await;
    let gpu0_epochs: Vec<u64> = journal(h.state_dir.path())
        .into_iter()
        .filter(|r| r["type"] == "worker_epoch" && r["worker"] == "gpu0")
        .map(|r| r["epoch"].as_u64().unwrap())
        .collect();
    assert_eq!(gpu0_epochs, [1], "gpu0 rejoined at a new epoch");
    assert!(aborted(&h.mock).is_empty(), "{:?}", aborted(&h.mock));
}

/// Review A1 pass 3: a re-smoke whose chat hangs to the capped deadline sends `/abort_request` for its rid, and a
/// slow abort (900 ms, under the 2 s control timeout) must not delay the probe's next sighting. gpu0 (cap 2) runs
/// a 4 s stream; the re-smoke gets the second slot, hangs 650 ms (half of 300 ms + 1 s), and its abort takes
/// 900 ms: awaited inline that is 1.55 s without a sighting, past the 1.3 s bound. gpu0 stays at epoch 1 and the
/// stream completes.
#[tokio::test(flavor = "multi_thread")]
async fn a_slow_smoke_abort_does_not_age_a_suspect_worker_lost() {
    let cap_2 = || async {
        MockSglang::start(MockConfig {
            token_latency: Duration::from_millis(20),
            natural_output_tokens: 200,
            server_info: ServerInfo {
                max_running_requests: 2,
                ..ServerInfo::default()
            },
            ..MockConfig::default()
        })
        .await
        .unwrap()
    };
    let h = Harness::start_replicas_tuned(
        vec![cap_2().await, cap_2().await],
        &[
            ("max_running_requests = 4", "max_running_requests = 2"),
            LOST_AFTER_1S,
        ],
        probe_aging(fast_tuning()),
    )
    .await;
    let (rid, worker, mut reader) = open_stream(&h).await;
    assert_eq!(worker, "gpu0");
    h.mock
        .set_fault(cere_mock_sglang::Fault::HangBeforeResponse);
    h.mock.set_abort_delay(Duration::from_millis(900));
    h.mock.set_server_info_override(Some(ServerInfo {
        max_running_requests: 2,
        kv_cache_dtype: "bf16".to_owned(),
        ..ServerInfo::default()
    }));
    readyz_until(&h, "gpu0 suspect", |b| {
        b["workers"]["gpu0"]["state"] == "suspect"
    })
    .await;
    h.mock.set_server_info_override(None);

    let events = rest(&mut reader, "the gpu0 stream").await;
    let terminal = check_stream_tail(&events, 0);
    assert_eq!(
        terminal["choices"][0]["finish_reason"], "stop",
        "{terminal}"
    );
    let (code, status) = h.status(&rid, TOKEN_A).await;
    assert_eq!(code, 200, "{status}");
    assert_eq!(status["state"], "completed", "{status}");
    let gpu0_epochs: Vec<u64> = journal(h.state_dir.path())
        .into_iter()
        .filter(|r| r["type"] == "worker_epoch" && r["worker"] == "gpu0")
        .map(|r| r["epoch"].as_u64().unwrap())
        .collect();
    assert_eq!(gpu0_epochs, [1], "gpu0 rejoined at a new epoch");
}
