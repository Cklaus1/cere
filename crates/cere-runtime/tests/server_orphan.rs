//! m1close R3 (Q6, Q7): a recovered orphan's admission slot is released early once the backend load `L`
//! (`/get_load`) is at most `C`, the number of `cered` streams confirmed inside the scheduler, on two
//! consecutive polls; every release is journaled as `orphan_released`, and an orphan still held when `cered`
//! stops is charged again by the next incarnation until its original retention bound. In-process `cered`
//! against the mock; CI evidence only.

mod support;

use std::sync::Arc;
use std::time::Duration;

use cere_mock_sglang::MockConfig;
use cere_runtime::app::{ManualClock, RetentionClock, Tuning};
use serde_json::{json, Value};
use support::*;

/// Enables the Q6 load release with the shortest poll period (1 s); retention stays at the t4 hour.
const LOAD_RELEASE: [(&str, &str); 2] = [
    (
        "shared_backend_tenancy = \"shared_acknowledged\"",
        "shared_backend_tenancy = \"shared_acknowledged\"\nload_endpoint = \"/get_load\"",
    ),
    (
        "terminal_retention_ms = 3600000",
        "terminal_retention_ms = 3600000\norphan_load_poll_ms = 1000",
    ),
];

/// The Q7 restart test's retention bound; the test moves its clock across it (`Tuning::retention_clock`).
const RETENTION_8S: [(&str, &str); 1] = [(
    "terminal_retention_ms = 3600000",
    "terminal_retention_ms = 8000",
)];

/// Long generations (20 ms per token, 2 000 tokens), so a stream stays inside the "scheduler" for the test.
fn slow_mock() -> MockConfig {
    MockConfig {
        token_latency: Duration::from_millis(20),
        natural_output_tokens: 2_000,
        ..MockConfig::default()
    }
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

async fn wait_aborted(h: &Harness, rid: &str) {
    for _ in 0..400 {
        if h.mock.abort_requests().iter().any(|a| a.rid == rid) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("orphan {rid} was never aborted");
}

/// Waits (bounded) until admission counts exactly `inflight` slots; returns the time it did.
async fn wait_inflight(h: &Harness, inflight: u64, bound: Duration) -> u64 {
    let until = tokio::time::Instant::now() + bound;
    while h.app.admission_snapshot().inflight != inflight {
        assert!(
            tokio::time::Instant::now() < until,
            "admission never reached {inflight}: {:?}",
            h.app.admission_snapshot()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    unix_ms()
}

/// The mock's `/get_load` total, read the way `cered` reads it.
async fn backend_load(h: &Harness) -> u64 {
    let ranks: Vec<Value> = h
        .http
        .get(format!("{}/get_load", h.mock.base_url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    ranks
        .iter()
        .map(|r| r["num_reqs"].as_u64().unwrap() + r["num_waiting_reqs"].as_u64().unwrap())
        .sum()
}

/// The orphan's backend request still generating after `cered` aborted it (the live L3 finding: the abort
/// arrived when SGLang no longer mapped the rid): a stream opened directly on the mock under the orphan's rid.
async fn orphan_keeps_running(h: &Harness, rid: &str) -> reqwest::Response {
    let response = h
        .http
        .post(format!("{}/v1/chat/completions", h.mock.base_url()))
        .json(&json!({
            "rid": rid,
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello there"}],
            "max_completion_tokens": 2_000,
            "stream": true,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    response
}

/// Starts a `cered` stream as scope `a`; resolves once its 200 headers arrived (after the backend's).
fn cered_stream(h: &Harness) -> tokio::task::JoinHandle<reqwest::Response> {
    let mut body = chat_body(true);
    body["max_completion_tokens"] = 2_000.into();
    let request = h.chat(&body);
    tokio::spawn(async move {
        let response = request.send().await.unwrap();
        assert_eq!(response.status(), 200);
        response
    })
}

async fn wait_confirmed(h: &Harness, confirmed: u64) {
    let until = tokio::time::Instant::now() + Duration::from_secs(10);
    while h.app.confirmed_streams() != confirmed {
        assert!(
            tokio::time::Instant::now() < until,
            "confirmed streams never reached {confirmed}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn released(h: &Harness) -> Vec<(String, String)> {
    journal(h.state_dir.path())
        .iter()
        .filter(|r| r["type"] == "orphan_released")
        .map(|r| {
            (
                r["rid"].as_str().unwrap().to_owned(),
                r["cause"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

/// Q6 + Q7: an idle backend (`L = 0 = C`) releases the orphan on the second poll, long before the hour of
/// retention, and journals `cause = "load"`; the next incarnation does not charge it again.
#[tokio::test(flavor = "multi_thread")]
async fn q6_idle_backend_releases_the_orphan_by_load_and_a_restart_keeps_it_released() {
    let admitted_ms = unix_ms();
    let mut rid = String::new();
    let h = Harness::start_seeded(slow_mock(), &LOAD_RELEASE, |config| {
        rid = seed_orphan(config, "a", admitted_ms);
    })
    .await;
    wait_aborted(&h, &rid).await;
    let at = wait_inflight(&h, 0, Duration::from_secs(10)).await;
    assert!(
        at >= admitted_ms + 1_000,
        "released {} ms after admission: two polls are at least 1 s apart",
        at - admitted_ms
    );
    assert_eq!(released(&h), [(rid.clone(), "load".to_owned())]);
    let h = h.restart().await;
    assert_eq!(
        h.app.admission_snapshot().inflight,
        0,
        "released orphan re-charged"
    );
    assert_eq!(released(&h), [(rid, "load".to_owned())]);
}

/// Q6 critical path: the backend load lags a dispatched request (no load count, no chunk yet) while the orphan
/// still runs. The naive rule `L ≤ dispatched in flight` would release the orphan; the Q6 rule holds it because
/// `C` only counts streams confirmed by a backend chunk. The orphan stays held while `L > C` (orphan plus a
/// confirmed `cered` stream) and is released once the orphan really ended (`L = 1 = C`).
#[tokio::test(flavor = "multi_thread")]
async fn q6_lagging_load_holds_the_orphan_where_the_naive_rule_releases() {
    let mut rid = String::new();
    let h = Harness::start_seeded(slow_mock(), &LOAD_RELEASE, |config| {
        rid = seed_orphan(config, "a", unix_ms());
    })
    .await;
    wait_aborted(&h, &rid).await;
    let orphan = orphan_keeps_running(&h, &rid).await;
    h.mock.set_load_lag(Duration::from_secs(4));
    let stream = cered_stream(&h);
    wait_inflight(&h, 2, Duration::from_secs(5)).await;
    // Three polls (1 s apart) while the dispatched stream is in transit.
    tokio::time::sleep(Duration::from_millis(3_000)).await;
    let load = backend_load(&h).await;
    let held_orphans = 1;
    let dispatched = h.app.admission_snapshot().inflight - held_orphans;
    assert_eq!((load, dispatched), (1, 1));
    assert!(load <= dispatched, "the naive rule would release here");
    assert_eq!(h.app.confirmed_streams(), 0);
    assert_eq!(
        h.app.admission_snapshot().inflight,
        2,
        "orphan released while L > C"
    );
    assert!(released(&h).is_empty());
    // The lag ends: the stream is confirmed, but the orphan still runs (L = 2 > C = 1).
    let response = stream.await.unwrap();
    let stream = tokio::spawn(async move {
        let mut reader = SseReader::new(response);
        while reader.next().await.is_some() {}
    });
    wait_confirmed(&h, 1).await;
    tokio::time::sleep(Duration::from_millis(2_500)).await;
    assert_eq!(backend_load(&h).await, 2);
    assert_eq!(
        h.app.admission_snapshot().inflight,
        2,
        "orphan released while L > C"
    );
    // The orphan ends: L = 1 = C on two polls releases it.
    drop(orphan);
    wait_inflight(&h, 1, Duration::from_secs(10)).await;
    assert_eq!(released(&h), [(rid, "load".to_owned())]);
    assert_eq!(h.app.confirmed_streams(), 1);
    stream.abort();
}

/// Q7: a restart during the hold charges the orphan again until its ORIGINAL `admitted_ms +
/// terminal_retention_ms` (never extended by the restart), then journals `cause = "retention"`. Retention reads
/// the test's clock (`Tuning::retention_clock`, kept across the restart), so neither incarnation's setup spends
/// the bound (RtFlake2 L1; it failed "test setup too slow" on a loaded host, rt-flake design note F6).
#[tokio::test(flavor = "multi_thread")]
async fn q7_restart_during_the_hold_recharges_until_the_original_bound() {
    let admitted_ms = unix_ms();
    let clock = Arc::new(ManualClock::new(admitted_ms));
    let tuning = Tuning {
        retention_clock: RetentionClock::Manual(clock.clone()),
        ..test_tuning()
    };
    let mut rid = String::new();
    let h = Harness::start_seeded_tuned(slow_mock(), &RETENTION_8S, tuning, |config| {
        rid = seed_orphan(config, "a", admitted_ms);
    })
    .await;
    wait_aborted(&h, &rid).await;
    clock.set(admitted_ms + 2_000);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(h.app.admission_snapshot().inflight, 1);
    let h = h.restart().await;
    assert_eq!(
        h.app.admission_snapshot().inflight,
        1,
        "orphan not re-charged"
    );
    assert!(released(&h).is_empty());
    wait_aborted(&h, &rid).await;
    clock.set(admitted_ms + 7_999);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        h.app.admission_snapshot().inflight,
        1,
        "orphan released before the original 8 000 ms bound"
    );
    clock.set(admitted_ms + 8_000);
    wait_inflight(&h, 0, Duration::from_secs(10)).await;
    assert_eq!(released(&h), [(rid, "retention".to_owned())]);
}
