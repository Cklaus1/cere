//! L4 on the request path: global cap, queue bound, deadlines while queued and while running.
mod support;

use std::sync::Arc;
use std::time::Duration;

use cere_mock_sglang::{Fault, MockConfig, ServerInfo};
use cere_runtime::app::{ManualClock, RetentionClock, Tuning};
use serde_json::Value;
use support::*;

const CAP_ONE: [(&str, &str); 2] = [
    ("max_backend_inflight = 2", "max_backend_inflight = 1"),
    ("per_scope_inflight = 2", "per_scope_inflight = 1"),
];

fn slow(token_ms: u64, tokens: u32) -> MockConfig {
    MockConfig {
        token_latency: Duration::from_millis(token_ms),
        natural_output_tokens: tokens,
        ..MockConfig::default()
    }
}

/// A unary request in the background; resolves to (status, rid, body).
fn spawn_chat(
    h: &Harness,
    body: Value,
    deadline_ms: Option<u64>,
) -> tokio::task::JoinHandle<(u16, String, Value)> {
    let mut request = h.chat(&body);
    if let Some(ms) = deadline_ms {
        request = request.header("X-Cere-Deadline-Ms", ms.to_string());
    }
    tokio::spawn(async move {
        let r = request.send().await.unwrap();
        let status = r.status().as_u16();
        let rid = r
            .headers()
            .get("x-cere-request-id")
            .map(|v| v.to_str().unwrap().to_owned())
            .unwrap_or_default();
        (status, rid, r.json().await.unwrap())
    })
}

async fn wait_inflight(h: &Harness, inflight: u64, queued: u64) {
    for _ in 0..200 {
        let s = h.app.admission_snapshot();
        if s.inflight == inflight && s.queued == queued {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!(
        "admission never reached {inflight}/{queued}: {:?}",
        h.app.admission_snapshot()
    );
}

/// Admission counts a request in flight before its backend call arrives at the mock, so a test that samples
/// `dispatched_rids` must first wait for the mock to see the in-flight request (CI raced this).
async fn wait_backend_live(h: &Harness, live: usize) {
    for _ in 0..200 {
        if h.mock.in_flight() == live {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!(
        "the mock never saw {live} live requests: {}",
        h.mock.in_flight()
    );
}

#[tokio::test]
async fn l4_second_request_queues_then_runs() {
    let h = Harness::start(slow(30, 8), &CAP_ONE).await;
    let base = h.mock.max_in_flight();
    assert_eq!(base, 1, "the readiness smoke ran alone");
    let first = spawn_chat(&h, chat_body(false), None);
    wait_inflight(&h, 1, 0).await;
    let second = spawn_chat(&h, chat_body(false), None);
    wait_inflight(&h, 1, 1).await;
    let (s1, _, _) = first.await.unwrap();
    let (s2, rid2, _) = second.await.unwrap();
    assert_eq!((s1, s2), (200, 200));
    assert_eq!(
        h.mock.max_in_flight(),
        1,
        "the backend never saw two requests at once"
    );
    let status = h.terminal(&rid2).await;
    assert!(
        status["timings"]["queue_wait_ms"].as_u64().unwrap() >= 100,
        "{status}"
    );
}

/// Review finding 13, L4: 2 scopes x 5 requests at per-scope cap 1 (global cap 1) dispatch round-robin through
/// the HTTP path: the scope served least recently goes first, FIFO within a scope.
#[tokio::test]
async fn l4_two_scopes_interleave_round_robin() {
    let h = Harness::start(slow(20, 1_000), &CAP_ONE).await;
    let spawn_as = |token: &'static str, tokens: u32| {
        let mut body = chat_body(false);
        body["max_completion_tokens"] = tokens.into();
        let request = h.post("/v1/chat/completions", token).json(&body);
        tokio::spawn(async move {
            let r = request.send().await.unwrap();
            assert_eq!(r.status(), 200);
            let rid = r.headers()["x-cere-request-id"]
                .to_str()
                .unwrap()
                .to_owned();
            (token, rid)
        })
    };
    // a0 runs (60 tokens, about 1.2 s); a1..a4 then b0..b4 (2 tokens each) queue in that order behind it.
    let mut calls = vec![spawn_as(TOKEN_A, 60)];
    wait_inflight(&h, 1, 0).await;
    for (i, token) in [TOKEN_A; 4].into_iter().chain([TOKEN_B; 5]).enumerate() {
        calls.push(spawn_as(token, 2));
        wait_inflight(&h, 1, i as u64 + 1).await;
    }
    let mut scope_of = std::collections::HashMap::new();
    for call in calls {
        let (token, rid) = call.await.unwrap();
        scope_of.insert(rid, if token == TOKEN_A { "a" } else { "b" });
    }
    let order: Vec<&str> = h
        .mock
        .dispatched_rids()
        .iter()
        .filter_map(|rid| scope_of.get(rid).copied())
        .collect();
    assert_eq!(
        order,
        ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        "dispatch order by scope"
    );
}

/// Review finding 1: a request that waits in the queue longer than the worker's suspect bound (three probe
/// periods) is still served while the background probe keeps reporting the backend ready.
#[tokio::test]
async fn l4_long_queue_wait_still_dispatches_on_a_ready_worker() {
    let mut tuning = test_tuning();
    tuning.probe_interval = Duration::from_millis(100);
    let h = Harness::start_tuned(slow(50, 20), &CAP_ONE, probe_aging(tuning)).await;
    let mut long = chat_body(false);
    long["max_completion_tokens"] = 20.into();
    let first = spawn_chat(&h, long, None);
    wait_inflight(&h, 1, 0).await;
    let second = spawn_chat(&h, chat_body(false), None);
    wait_inflight(&h, 1, 1).await;
    assert_eq!(first.await.unwrap().0, 200);
    let (status, rid, body) = second.await.unwrap();
    assert_eq!(status, 200, "{body}");
    let record = h.terminal(&rid).await;
    assert!(
        record["timings"]["queue_wait_ms"].as_u64().unwrap() >= 300,
        "queued past the suspect bound: {record}"
    );
}

#[tokio::test]
async fn l4_queue_full_is_429_without_dispatch() {
    let h = Harness::start(
        slow(30, 8),
        &[
            CAP_ONE[0],
            CAP_ONE[1],
            ("queue_depth = 64", "queue_depth = 1"),
        ],
    )
    .await;
    let first = spawn_chat(&h, chat_body(false), None);
    wait_inflight(&h, 1, 0).await;
    wait_backend_live(&h, 1).await;
    let second = spawn_chat(&h, chat_body(false), None);
    wait_inflight(&h, 1, 1).await;
    let dispatched = h.mock.dispatched_rids().len();
    let r = h
        .chat(&chat_body(false))
        .header("Idempotency-Key", "retry-me")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 429);
    let rid = r.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    assert_eq!(
        r.json::<Value>().await.unwrap()["error"]["code"],
        "queue_full"
    );
    assert_eq!(
        h.mock.dispatched_rids().len(),
        dispatched,
        "a rejected request is never dispatched"
    );
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "failed");
    assert_eq!(status["reason"], "queue_full");
    assert_eq!(first.await.unwrap().0, 200);
    assert_eq!(second.await.unwrap().0, 200);
    // Review finding 2: the rejection did not consume the key; the retry under it runs.
    let retry = h
        .chat(&chat_body(false))
        .header("Idempotency-Key", "retry-me")
        .send()
        .await
        .unwrap();
    let retry_status = retry.status();
    assert_eq!(retry_status, 200, "{}", retry.text().await.unwrap());
}

#[tokio::test]
async fn l4_deadline_while_queued_is_504_without_dispatch() {
    let h = Harness::start(slow(40, 10), &CAP_ONE).await;
    let first = spawn_chat(&h, chat_body(false), None);
    wait_inflight(&h, 1, 0).await;
    wait_backend_live(&h, 1).await;
    let dispatched = h.mock.dispatched_rids().len();
    let (status, rid, body) = spawn_chat(&h, chat_body(false), Some(150)).await.unwrap();
    assert_eq!(status, 504);
    assert_eq!(body["error"]["code"], "deadline_exceeded");
    assert_eq!(h.mock.dispatched_rids().len(), dispatched);
    let record = h.terminal(&rid).await;
    assert_eq!(record["state"], "cancelled");
    assert_eq!(record["reason"], "deadline_exceeded");
    assert_eq!(first.await.unwrap().0, 200);
    assert!(h.mock.abort_requests().is_empty());
}

#[tokio::test]
async fn l4_deadline_while_running_aborts() {
    let h = Harness::start(slow(50, 40), &[]).await;
    let mut body = chat_body(false);
    body["max_completion_tokens"] = 40.into();
    let (status, rid, body) = spawn_chat(&h, body, Some(300)).await.unwrap();
    assert_eq!(status, 504);
    assert_eq!(body["error"]["code"], "deadline_exceeded");
    assert_eq!(body["cere"]["state"], "cancelled");
    assert_eq!(body["cere"]["reason"], "deadline_exceeded");
    let aborts: Vec<String> = h.mock.abort_requests().into_iter().map(|a| a.rid).collect();
    assert_eq!(aborts, vec![rid.clone()]);
    let record = h.terminal(&rid).await;
    assert_eq!(record["state"], "cancelled");
    assert_eq!(record["reason"], "deadline_exceeded");
    assert_eq!(h.app.admission_snapshot().inflight, 0);
}

/// Review finding 4: on the deadline path an abort the backend ignores fails `cancel_unacknowledged` (not
/// `cancelled`), and the slot stays charged while the backend still generates (held at its finish, so its natural
/// end cannot come first on a loaded host).
#[tokio::test]
async fn l4_deadline_with_ignored_abort_fails_cancel_unacknowledged() {
    let h = Harness::start_tuned(slow(40, 60), &[], short_ack(test_tuning())).await;
    h.mock.set_fault(Fault::IgnoreAbort);
    h.mock.hold_finish(true);
    let mut body = chat_body(false);
    body["max_completion_tokens"] = 60.into();
    let (status, rid, body) = spawn_chat(&h, body, Some(300)).await.unwrap();
    assert_eq!(status, 502, "{body}");
    assert_eq!(body["error"]["code"], "cancel_unacknowledged");
    assert_eq!(body["cere"]["state"], "failed");
    assert_eq!(body["cere"]["reason"], "cancel_unacknowledged");
    assert_eq!(h.mock.in_flight(), 1, "the backend still generates");
    assert_eq!(h.app.admission_snapshot().inflight, 1, "slot still charged");
    let aborts: Vec<String> = h.mock.abort_requests().into_iter().map(|a| a.rid).collect();
    assert_eq!(aborts, vec![rid.clone()]);
    let record = h.terminal(&rid).await;
    assert_eq!(record["state"], "failed");
    assert_eq!(record["reason"], "cancel_unacknowledged");
}

/// Makes the readiness probe forget the backend identity, so the next probe smokes again.
async fn forget_identity(h: &Harness) {
    h.mock.set_server_info_override(Some(ServerInfo {
        kv_cache_dtype: "auto".to_owned(),
        ..ServerInfo::default()
    }));
    assert!(!h.app.readiness(0).probe().await.ready);
    h.mock.set_server_info_override(None);
}

/// Review finding 9: the readiness smoke is backend work and waits for an admission slot like any request.
#[tokio::test]
async fn a2_smoke_runs_under_the_backend_cap() {
    let h = Harness::start(slow(50, 20), &CAP_ONE).await;
    let mut long = chat_body(false);
    long["max_completion_tokens"] = 20.into();
    let first = spawn_chat(&h, long, None);
    wait_inflight(&h, 1, 0).await;
    forget_identity(&h).await;
    let smoked = h.app.readiness(0).probe().await;
    assert!(smoked.ready, "{:?}", smoked.problems);
    assert_eq!(first.await.unwrap().0, 200);
    assert_eq!(
        h.mock.max_in_flight(),
        1,
        "the smoke ran beside a request at cap 1"
    );
}

/// Review finding 9: a smoke that does not complete (here: times out) is aborted by its rid. The abort runs
/// detached (review A1 pass 3): the probe returns at once while the smoke's slot stays held until the abort
/// returns.
#[tokio::test]
async fn a2_failed_smoke_is_aborted() {
    let mut tuning = test_tuning();
    tuning.smoke_deadline = Duration::from_millis(300);
    let h = Harness::start_tuned(MockConfig::default(), &[], tuning).await;
    h.mock.set_fault(Fault::HangBeforeResponse);
    h.mock.set_abort_delay(Duration::from_millis(300));
    forget_identity(&h).await;
    let failed = h.app.readiness(0).probe().await;
    assert!(!failed.ready);
    assert_eq!(failed.problems[0].field, "smoke", "{:?}", failed.problems);
    assert_eq!(
        h.app.admission_snapshot().inflight,
        1,
        "the smoke's slot is held while its abort runs"
    );
    let smoke_rid = h.mock.dispatched_rids().last().unwrap().clone();
    wait_aborted(&h, &smoke_rid).await;
    let aborts: Vec<String> = h.mock.abort_requests().into_iter().map(|a| a.rid).collect();
    assert_eq!(aborts, vec![smoke_rid]);
    for _ in 0..400 {
        if h.app.admission_snapshot().inflight == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("the smoke's slot was not released after its abort");
}

const RETENTION_2S: [(&str, &str); 1] = [(
    "terminal_retention_ms = 3600000",
    "terminal_retention_ms = 2000",
)];

/// A retention bound the l3 hold test moves its clock across (`Tuning::retention_clock`).
const RETENTION_6S: [(&str, &str); 1] = [(
    "terminal_retention_ms = 3600000",
    "terminal_retention_ms = 6000",
)];

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

async fn wait_aborted(h: &Harness, rid: &str) {
    for _ in 0..400 {
        if h.mock
            .abort_requests()
            .iter()
            .any(|a| a.rid == rid && !a.abort_all)
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("orphan {rid} was never aborted");
}

/// L3 (live finding): SGLang ignores an abort that arrives after the client connection closed and keeps
/// generating, so a recovered orphan's slot stays charged after its abort until the retention bound since its
/// admission has passed. Retention reads the test's clock, held at the admission while `cered` opens and aborts
/// the orphan: on the wall clock a loaded host's setup spent the bound before the first check (RtFlake2 L1).
#[tokio::test]
async fn l3_orphan_slot_held_after_abort_until_retention_passes() {
    let admitted_ms = unix_ms();
    let clock = Arc::new(ManualClock::new(admitted_ms));
    let tuning = Tuning {
        retention_clock: RetentionClock::Manual(clock.clone()),
        ..test_tuning()
    };
    let mut rid = String::new();
    let h = Harness::start_seeded_tuned(MockConfig::default(), &RETENTION_6S, tuning, |config| {
        rid = seed_orphan(config, "a", admitted_ms);
    })
    .await;
    wait_aborted(&h, &rid).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        h.app.admission_snapshot().inflight,
        1,
        "orphan released at its abort"
    );
    clock.set(admitted_ms + 5_999);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        h.app.admission_snapshot().inflight,
        1,
        "orphan released before its retention bound"
    );
    clock.set(admitted_ms + 6_000);
    wait_inflight(&h, 0, 0).await;
}

/// An orphan admitted longer ago than the retention bound (cered was down) is released right after its abort.
#[tokio::test]
async fn l3_old_orphan_released_after_abort() {
    let admitted_ms = unix_ms() - 10_000;
    let mut rid = String::new();
    let h = Harness::start_seeded(MockConfig::default(), &RETENTION_2S, |config| {
        rid = seed_orphan(config, "a", admitted_ms);
    })
    .await;
    wait_aborted(&h, &rid).await;
    wait_inflight(&h, 0, 0).await;
}

/// Queues one unary request as `token` with an optional `X-Cere-Priority`; resolves to (label, rid).
fn spawn_labelled(
    h: &Harness,
    label: &'static str,
    token: &'static str,
    priority: Option<&'static str>,
    tokens: u32,
) -> tokio::task::JoinHandle<(&'static str, String)> {
    let mut body = chat_body(false);
    body["max_completion_tokens"] = tokens.into();
    let mut request = h.post("/v1/chat/completions", token).json(&body);
    if let Some(priority) = priority {
        request = request.header("X-Cere-Priority", priority);
    }
    tokio::spawn(async move {
        let r = request.send().await.unwrap();
        assert_eq!(r.status(), 200);
        let rid = r.headers()["x-cere-request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        (label, rid)
    })
}

/// The labels of `calls` in the order the mock saw their backend requests.
async fn dispatch_order(
    h: &Harness,
    calls: Vec<tokio::task::JoinHandle<(&'static str, String)>>,
) -> Vec<&'static str> {
    let mut label_of = std::collections::HashMap::new();
    for call in calls {
        let (label, rid) = call.await.unwrap();
        label_of.insert(rid, label);
    }
    h.mock
        .dispatched_rids()
        .iter()
        .filter_map(|rid| label_of.get(rid).copied())
        .collect()
}

/// m1close R1/Q1 through HTTP: scope `a` with weight 3 gets three grants per turn to `b`'s one. The first
/// request's immediate grant counts against `a`'s quantum.
#[tokio::test]
async fn r1_weighted_scopes_through_http() {
    let mut tweaks = CAP_ONE.to_vec();
    tweaks.push((
        "scope_id = \"a\"\ntoken_env = \"TOK_A\"",
        "scope_id = \"a\"\ntoken_env = \"TOK_A\"\nweight = 3",
    ));
    let h = Harness::start(slow(20, 1_000), &tweaks).await;
    let mut calls = vec![spawn_labelled(&h, "a", TOKEN_A, None, 60)];
    wait_inflight(&h, 1, 0).await;
    for (i, (label, token)) in [("a", TOKEN_A); 4]
        .into_iter()
        .chain([("b", TOKEN_B); 5])
        .enumerate()
    {
        calls.push(spawn_labelled(&h, label, token, None, 2));
        wait_inflight(&h, 1, i as u64 + 1).await;
    }
    assert_eq!(
        dispatch_order(&h, calls).await,
        ["a", "a", "a", "b", "a", "a", "b", "b", "b", "b"],
        "dispatch order by scope"
    );
}

/// m1close R1/Q2 through HTTP: `X-Cere-Priority` (case-insensitive) orders one scope's queue high, normal,
/// low; absent means normal.
#[tokio::test]
async fn r1_priority_header_orders_a_scope_queue() {
    let h = Harness::start(slow(20, 1_000), &CAP_ONE).await;
    let mut calls = vec![spawn_labelled(&h, "first", TOKEN_A, None, 60)];
    wait_inflight(&h, 1, 0).await;
    for (i, (label, priority)) in [
        ("low", Some("low")),
        ("normal", None),
        ("high", Some("HIGH")),
        ("normal2", Some("Normal")),
    ]
    .into_iter()
    .enumerate()
    {
        calls.push(spawn_labelled(&h, label, TOKEN_A, priority, 2));
        wait_inflight(&h, 1, i as u64 + 1).await;
    }
    assert_eq!(
        dispatch_order(&h, calls).await,
        ["first", "high", "normal", "normal2", "low"]
    );
}

/// m1close R1/Q2: an unknown priority is a 400 before anything is accepted or dispatched.
#[tokio::test]
async fn r1_invalid_priority_is_rejected() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let dispatched = h.mock.dispatched_rids();
    let r = h
        .chat(&chat_body(false))
        .header("X-Cere-Priority", "urgent")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    assert_eq!(body["error"]["param"], "X-Cere-Priority", "{body}");
    assert_eq!(h.mock.dispatched_rids(), dispatched);
}

/// m1close Q2: neither `X-Cere-Priority` nor `X-Cere-On-Disconnect` enters the idempotency digest, so a keyed
/// retry with other values replays the original instead of failing `idempotency_conflict`.
#[tokio::test]
async fn r1_scheduling_headers_are_not_digested() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let first = h
        .chat(&chat_body(false))
        .header("Idempotency-Key", "k-sched")
        .header("X-Cere-Priority", "high")
        .header("X-Cere-On-Disconnect", "continue")
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), 200);
    let rid = first.headers()["x-cere-request-id"].clone();
    let body: Value = first.json().await.unwrap();
    let dispatched = h.mock.dispatched_rids();
    let retry = h
        .chat(&chat_body(false))
        .header("Idempotency-Key", "k-sched")
        .header("X-Cere-Priority", "low")
        .send()
        .await
        .unwrap();
    assert_eq!(retry.status(), 200);
    assert_eq!(retry.headers()["x-cere-request-id"], rid);
    assert_eq!(retry.json::<Value>().await.unwrap(), body);
    assert_eq!(h.mock.dispatched_rids(), dispatched, "no regeneration");
}

/// M2.6 C5 (D12): rows as admission cost, driven synchronously through the public `Admission` API. Each grant is
/// recorded as `label@worker`; a release drops the oldest held permit and polls every waiter in queue order.
mod rows {
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use cere_runtime::admission::{
        Admission, AdmissionError, AdmissionLimits, Permit, Priority, SchedulePolicy, Target,
        WorkerCap,
    };
    use cere_runtime::worker::{Aging, WorkerHealth, WorkerView};
    use futures_util::task::noop_waker_ref;
    use futures_util::FutureExt;
    use tokio::time::Instant;

    type Acquire = Pin<Box<dyn Future<Output = Result<Permit, AdmissionError>> + Send>>;

    const NEVER: Aging = Aging {
        suspect_after_ms: u64::MAX,
        lost_after_ms: u64::MAX,
    };

    fn admission(
        max_inflight: u64,
        per_scope_inflight: u64,
        weights: &[(&str, u64)],
        workers: &[(&str, u64)],
    ) -> Admission {
        let limits = AdmissionLimits {
            max_inflight,
            per_scope_inflight,
            queue_depth: 64,
        };
        let policy = SchedulePolicy::new(
            weights.iter().map(|(s, w)| ((*s).to_owned(), *w)),
            Duration::from_secs(3_600),
        );
        let adm = Admission::with_policy(limits, policy);
        adm.set_workers(
            workers
                .iter()
                .map(|&(id, cap)| WorkerCap {
                    id: id.to_owned(),
                    cap,
                    aging: NEVER,
                    deployment: 0,
                })
                .collect(),
        );
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        for &(id, _) in workers {
            adm.update_worker(
                id,
                WorkerView {
                    health: WorkerHealth::Ready,
                    last_seen_ms: now_ms,
                    epoch: 1,
                    draining: false,
                    lease_until_ms: None,
                },
            );
        }
        adm
    }

    fn acquire(
        adm: &Admission,
        scope: &str,
        priority: Priority,
        target: Target,
        cost: u64,
    ) -> Acquire {
        let (adm, scope) = (adm.clone(), scope.to_owned());
        let deadline = Instant::now() + Duration::from_secs(3_600);
        Box::pin(async move { adm.acquire(&scope, priority, deadline, target, cost).await })
    }

    struct Sim {
        adm: Admission,
        held: VecDeque<(String, Permit)>,
        waiting: Vec<(String, Acquire)>,
        trace: Vec<String>,
    }

    impl Sim {
        fn new(adm: Admission) -> Self {
            Self {
                adm,
                held: VecDeque::new(),
                waiting: Vec::new(),
                trace: Vec::new(),
            }
        }

        /// Queues `label` (scope = its first character) at cost 1 and polls it once.
        fn enqueue(&mut self, label: &str, priority: Priority, target: Target) {
            self.enqueue_rows(label, priority, target, 1);
        }

        /// Queues `label` charging `cost` rows and polls it once.
        fn enqueue_rows(&mut self, label: &str, priority: Priority, target: Target, cost: u64) {
            let fut = acquire(&self.adm, &label[..1], priority, target, cost);
            self.waiting.push((label.to_owned(), fut));
            self.poll();
        }

        fn poll(&mut self) {
            let mut cx = Context::from_waker(noop_waker_ref());
            let mut still = Vec::new();
            for (label, mut fut) in std::mem::take(&mut self.waiting) {
                match fut.poll_unpin(&mut cx) {
                    Poll::Ready(Ok(permit)) => {
                        self.trace.push(format!("{label}@{}", permit.worker()));
                        self.held.push_back((label, permit));
                    }
                    Poll::Ready(Err(e)) => panic!("{label}: {e:?}"),
                    Poll::Pending => still.push((label, fut)),
                }
            }
            self.waiting = still;
        }

        /// Drops the oldest held permit (or `label`'s) and polls every waiter.
        fn release(&mut self, label: Option<&str>) {
            let index = match label {
                Some(l) => self.held.iter().position(|(h, _)| h == l).expect("held"),
                None => 0,
            };
            self.held.remove(index);
            self.poll();
        }
    }

    fn pinned(id: &str) -> Target {
        Target::Pinned(Arc::from(id), 0)
    }

    /// Regression (recorded on the merge base before costs existed): a mixed cost-1 workload - weighted scopes,
    /// priorities, pins, two workers, global cap below the worker sum - grants in exactly the same order.
    #[tokio::test]
    async fn cost_one_trace_equals_merge_base() {
        let adm = admission(3, 2, &[("a", 2)], &[("w1", 2), ("w2", 2)]);
        let mut sim = Sim::new(adm);
        use Priority::{High, Low, Normal};
        sim.enqueue("a0", Normal, Target::Any(0));
        sim.enqueue("b0", Normal, pinned("w1"));
        sim.enqueue("c0", Normal, Target::Any(0));
        for (label, priority, target) in [
            ("a1", Normal, Target::Any(0)),
            ("b1", Low, Target::Any(0)),
            ("a2", High, pinned("w2")),
            ("c1", Normal, pinned("w1")),
            ("b2", High, Target::Any(0)),
            ("a3", Low, Target::Any(0)),
            ("c2", Normal, Target::Any(0)),
            ("a4", Normal, pinned("w1")),
            ("b3", Normal, Target::Any(0)),
            ("c3", High, Target::Any(0)),
        ] {
            sim.enqueue(label, priority, target);
        }
        let mut step = 0;
        while !sim.held.is_empty() {
            sim.release(None);
            step += 1;
            if step == 3 {
                sim.enqueue("a5", High, Target::Any(0));
                sim.enqueue("b4", Normal, pinned("w2"));
            }
        }
        assert!(sim.waiting.is_empty());
        assert_eq!(
            sim.trace,
            [
                "a0@w1", "b0@w1", "c0@w2", "a2@w2", "b2@w1", "c3@w1", "a5@w2", "a1@w1", "b3@w1",
                "c2@w2", "a4@w1", "a3@w1", "b4@w2", "c1@w1", "b1@w1"
            ]
        );
    }

    fn loads(adm: &Admission, scope: &str) -> Vec<(String, u64, u64, u64)> {
        adm.worker_loads(scope)
            .into_iter()
            .map(|l| (l.id.to_string(), l.cap, l.inflight, l.scope_inflight))
            .collect()
    }

    /// C5: a cost-3 waiter with 2 free rows waits; it is granted once 3 rows are free; the global, scope and
    /// worker loads each charge 3; its release frees 3.
    #[tokio::test]
    async fn cost_three_waits_for_three_free_rows_and_charges_three() {
        let mut sim = Sim::new(admission(4, 4, &[], &[("w", 4)]));
        sim.enqueue("z0", Priority::Normal, Target::Any(0));
        sim.enqueue("z1", Priority::Normal, Target::Any(0));
        sim.enqueue_rows("a0", Priority::Normal, Target::Any(0), 3);
        assert_eq!(sim.trace, ["z0@w", "z1@w"], "2 free rows: cost 3 waits");
        assert_eq!(
            (sim.adm.snapshot().inflight, sim.adm.snapshot().queued),
            (2, 1)
        );
        sim.release(Some("z0"));
        assert_eq!(sim.trace, ["z0@w", "z1@w", "a0@w"], "granted at 3 free");
        assert_eq!(sim.adm.snapshot().inflight, 4);
        assert_eq!(sim.adm.scope_inflight("a"), 3);
        assert_eq!(loads(&sim.adm, "a"), [("w".to_owned(), 4, 4, 3)]);
        sim.release(Some("a0"));
        assert_eq!(sim.adm.snapshot().inflight, 1);
        assert_eq!(sim.adm.scope_inflight("a"), 0);
        assert_eq!(loads(&sim.adm, "a"), [("w".to_owned(), 4, 1, 0)]);
    }

    /// C5 head-of-line reservation: a cost-4 waiter amid a continuous cost-1 stream (one new cost-1 waiter per
    /// release) is granted as soon as 4 rows are free, and no cost-1 waiter is granted after it became head.
    #[tokio::test]
    async fn cost_four_head_is_not_starved_by_a_cost_one_stream() {
        let mut sim = Sim::new(admission(4, 4, &[], &[("w", 4)]));
        for i in 0..4 {
            sim.enqueue(&format!("s{i}"), Priority::Normal, Target::Any(0));
        }
        sim.enqueue_rows("b0", Priority::Normal, Target::Any(0), 4);
        assert_eq!(sim.trace.len(), 4, "full: b0 queues as head");
        for i in 0..4 {
            sim.enqueue(&format!("t{i}"), Priority::High, Target::Any(0));
            sim.release(None);
            if i < 3 {
                assert_eq!(
                    sim.trace.len(),
                    4,
                    "no cost-1 grant past the head: {:?}",
                    sim.trace
                );
            }
        }
        assert_eq!(sim.trace[4], "b0@w", "granted once 4 rows freed");
        assert_eq!(sim.trace.len(), 5);
        assert_eq!(sim.adm.snapshot().inflight, 4);
        sim.release(Some("b0"));
        assert_eq!(&sim.trace[5..], ["t0@w", "t1@w", "t2@w", "t3@w"]);
    }

    /// D12: the DRR quantum counts rows. Scope `a` (weight 4) spends 3 + 1 rows on a0, a1 and moves to the back,
    /// so the one row a1 frees goes to `b0`, not `a2` (with a request as the unit, `a` would still hold 2 of its
    /// quantum and `a2`, the earlier waiter, would come first).
    #[tokio::test]
    async fn drr_quantum_counts_rows() {
        let mut sim = Sim::new(admission(4, 4, &[("a", 4)], &[("w", 4)]));
        sim.enqueue_rows("c0", Priority::Normal, Target::Any(0), 4);
        sim.enqueue_rows("a0", Priority::Normal, Target::Any(0), 3);
        for label in ["a1", "a2", "b0", "b1"] {
            sim.enqueue(label, Priority::Normal, Target::Any(0));
        }
        sim.release(Some("c0"));
        assert_eq!(sim.trace, ["c0@w", "a0@w", "a1@w"]);
        sim.release(Some("a1"));
        assert_eq!(sim.trace, ["c0@w", "a0@w", "a1@w", "b0@w"]);
    }

    /// A cost no cap can ever hold (5 rows at 4) is never head of line: it waits for its deadline while cost-1
    /// waiters behind it are granted (validation bounds `n` by the profile, D11; this guards a misconfiguration).
    #[tokio::test]
    async fn cost_above_every_cap_does_not_block_the_queue() {
        let mut sim = Sim::new(admission(4, 4, &[], &[("w", 4)]));
        sim.enqueue_rows("a0", Priority::High, Target::Any(0), 5);
        sim.enqueue("b0", Priority::Normal, Target::Any(0));
        assert_eq!(sim.trace, ["b0@w"]);
        assert_eq!(sim.adm.snapshot().queued, 1);
    }

    /// M26R-01: when the blocking cost-4 head stops waiting (deadline, disconnect, dropped future), the cost-1
    /// waiter it held back is granted at once on the free row, without waiting for an unrelated event.
    #[tokio::test]
    async fn head_leaving_regrants_the_waiters_behind_it() {
        let mut sim = Sim::new(admission(4, 4, &[], &[("w", 4)]));
        for i in 0..3 {
            sim.enqueue(&format!("s{i}"), Priority::Normal, Target::Any(0));
        }
        sim.enqueue_rows("b0", Priority::Normal, Target::Any(0), 4);
        sim.enqueue("c0", Priority::Normal, Target::Any(0));
        assert_eq!(sim.trace.len(), 3, "c0 waits behind the head b0");
        let index = sim.waiting.iter().position(|(l, _)| l == "b0").unwrap();
        drop(sim.waiting.remove(index));
        assert_eq!(
            sim.adm.snapshot().queued,
            0,
            "b0 left; c0 granted on the free row"
        );
        sim.poll();
        assert_eq!(sim.trace, ["s0@w", "s1@w", "s2@w", "c0@w"]);
        assert_eq!(sim.adm.snapshot().inflight, 4);
    }
}
