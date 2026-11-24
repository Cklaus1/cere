//! M3.1 sessions and dispatch over two replicas (m3 design note C5 / Q8, Q7, C7 / Q9): a fork inherits the
//! parent's sticky worker; a turn pinned to a suspect worker runs on the other one and re-pins there; a full
//! Ready pinned worker makes the turn wait instead of moving; a `begin_work` rejection re-queues once onto the
//! other worker and a second one fails `worker_not_ready`; a journal failure at `dispatched` fails the request
//! before any backend sees it.
mod support;

use std::time::Duration;

use cere_mock_sglang::{MockConfig, MockHandle, MockSglang, ServerInfo};
use cere_runtime::app::Tuning;
use cere_runtime::ledger::DEFAULT_COMPACTION_SLACK;
use serde_json::{json, Value};
use support::*;
use tokio::task::JoinHandle;

/// The mock's reply to a turn with `max_completion_tokens = 4`.
const REPLY: &str = "tok0 tok1 tok2 tok3";

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn rid_of(response: &reqwest::Response) -> String {
    response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned()
}

async fn create_session(h: &Harness) -> String {
    let r = h.post("/v1/cere/sessions", TOKEN_A).send().await.unwrap();
    assert_eq!(r.status(), 201);
    let body: Value = r.json().await.unwrap();
    body["session_id"].as_str().unwrap().to_owned()
}

/// A session turn at `revision` whose history is `messages`.
fn turn(h: &Harness, session: &str, revision: u64, messages: &[&str]) -> reqwest::RequestBuilder {
    let messages: Vec<Value> = messages
        .iter()
        .enumerate()
        .map(|(i, content)| {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            json!({"role": role, "content": content})
        })
        .collect();
    h.chat(&json!({"model": MODEL, "messages": messages, "max_completion_tokens": 4}))
        .header("X-Cere-Session-Id", session)
        .header("X-Cere-Expected-Revision", revision.to_string())
}

/// Sends a request that must succeed; returns its rid and the worker that served it, after checking that
/// worker's mock received the rid and the other mock did not.
async fn served(h: &Harness, request: reqwest::RequestBuilder) -> (String, String) {
    let r = request.send().await.unwrap();
    assert_eq!(r.status(), 200);
    served_reply(h, r.json().await.unwrap())
}

fn served_reply(h: &Harness, reply: Value) -> (String, String) {
    let rid = reply["cere"]["request_id"].as_str().unwrap().to_owned();
    let worker = reply["cere"]["worker_id"].as_str().unwrap().to_owned();
    for i in 0..h.replica_count() {
        let seen = h.replica(i).dispatched_rids().contains(&rid);
        assert_eq!(seen, worker == format!("gpu{i}"), "replica {i}: {reply}");
    }
    (rid, worker)
}

/// Polls until `done` holds (5 s bound).
async fn until(what: &str, done: impl Fn() -> bool) {
    for _ in 0..1_000 {
        if done() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("timed out waiting for {what}");
}

/// Waits until no admission slot is charged (a finished request's permit drops just after its response).
async fn idle(h: &Harness) {
    until("no request in flight", || {
        h.app.admission_snapshot().inflight == 0
    })
    .await;
}

/// Two mocks generating a token every 5 ms with no natural end, serving `info`: a bounded request finishes in
/// a few tens of ms, a [`Held`] one runs until it is cancelled (it honours the abort, so its slot frees).
async fn generating_mocks(info: ServerInfo) -> Vec<MockHandle> {
    let mut out = Vec::new();
    for _ in 0..2 {
        let config = MockConfig {
            server_info: info.clone(),
            token_latency: Duration::from_millis(5),
            natural_output_tokens: 1_000_000,
            ..MockConfig::default()
        };
        out.push(MockSglang::start(config).await.unwrap());
    }
    out
}

/// A long non-session generation on `gpu0` (over [`generating_mocks`]) until [`Held::release`].
struct Held {
    rid: String,
    call: JoinHandle<reqwest::Result<reqwest::Response>>,
}

impl Held {
    /// Both workers must be idle: the unpinned request then goes to `gpu0` (least in flight, id tie-break).
    async fn on_gpu0(h: &Harness) -> Self {
        let mock = h.replica(0);
        let before = mock.dispatched_rids().len();
        let mut body = chat_body(false);
        body["max_completion_tokens"] = 30_000.into();
        let call = tokio::spawn(h.chat(&body).send());
        until("the held request on gpu0", || {
            mock.dispatched_rids().len() > before
        })
        .await;
        Self {
            rid: mock.dispatched_rids()[before].clone(),
            call,
        }
    }

    async fn release(self, h: &Harness) {
        let cancel = h
            .post(&format!("/v1/cere/requests/{}/cancel", self.rid), TOKEN_A)
            .send()
            .await
            .unwrap();
        assert!(cancel.status().is_success(), "{}", cancel.status());
        self.call.await.unwrap().unwrap();
        idle(h).await;
    }
}

/// C5: a fork inherits the parent's pin. The parent is pinned to `gpu1` (its first turn ran while `gpu0` was
/// busy); the fork's turn lands on `gpu1` although both workers are idle and an unpinned request goes to `gpu0`.
#[tokio::test]
async fn c5_fork_inherits_the_parent_pin() {
    let h = Harness::start_replicas(generating_mocks(ServerInfo::default()).await, &[]).await;
    let held = Held::on_gpu0(&h).await;
    let parent = create_session(&h).await;
    let (_, worker) = served(&h, turn(&h, &parent, 0, &["hi"])).await;
    assert_eq!(
        worker, "gpu1",
        "the parent's first turn avoids the busy gpu0"
    );
    held.release(&h).await;

    let fork = h
        .post(&format!("/v1/cere/sessions/{parent}/fork"), TOKEN_A)
        .json(&json!({"revision": 1}))
        .send()
        .await
        .unwrap();
    assert_eq!(fork.status(), 201);
    let fork: Value = fork.json().await.unwrap();
    let child = fork["session_id"].as_str().unwrap();
    let revision = fork["revision"].as_u64().unwrap();
    let (_, worker) = served(&h, turn(&h, child, revision, &["hi", REPLY, "branch"])).await;
    assert_eq!(
        worker, "gpu1",
        "the fork's turn runs on the parent's worker"
    );
    idle(&h).await;
    let (_, worker) = served(&h, h.chat(&chat_body(false))).await;
    assert_eq!(worker, "gpu0", "an unpinned request goes to the idle gpu0");
}

/// C5: a turn on a session pinned to a suspect worker runs on the other worker and re-pins the session there:
/// after the first worker is ready again, the next turn stays on the new worker. Readiness moves only on the
/// probes the test runs ([`quiet_replicas`]): under `fast_tuning`'s 100 ms background probes, a stall of the
/// test process over 300 ms aged both workers suspect, and the last unpinned request went to whichever worker a
/// probe cleared first (gpu1 at load 91, `queue_wait_ms=523`; rt-flake design note F8).
#[tokio::test]
async fn c5_suspect_pin_moves_the_turn_and_re_pins() {
    let h = quiet_replicas().await;
    let session = create_session(&h).await;
    let (_, worker) = served(&h, turn(&h, &session, 0, &["hi"])).await;
    assert_eq!(worker, "gpu0");

    h.replica(0).set_server_info_override(Some(ServerInfo {
        kv_cache_dtype: "bf16".to_owned(),
        ..ServerInfo::default()
    }));
    assert!(!h.app.readiness(0).probe().await.ready);
    readyz_until(&h, "gpu0 suspect", |b| {
        b["workers"]["gpu0"]["state"] == "suspect"
    })
    .await;
    let (_, worker) = served(&h, turn(&h, &session, 1, &["hi", REPLY, "more"])).await;
    assert_eq!(worker, "gpu1", "a suspect pin is not eligible");

    h.replica(0).set_server_info_override(None);
    assert!(h.app.readiness(0).probe().await.ready);
    readyz_until(&h, "gpu0 ready again", |b| {
        b["workers"]["gpu0"]["state"] == "ready"
    })
    .await;
    idle(&h).await;
    let history = ["hi", REPLY, "more", REPLY, "again"];
    let (_, worker) = served(&h, turn(&h, &session, 2, &history)).await;
    assert_eq!(worker, "gpu1", "the moved turn re-pinned the session");
    idle(&h).await;
    let (_, worker) = served(&h, h.chat(&chat_body(false))).await;
    assert_eq!(worker, "gpu0", "an unpinned request goes to the idle gpu0");
}

/// C5: a Ready pinned worker that is full (cap 1, held by another request) makes the turn wait, although the
/// other worker has room; it is dispatched to the pinned worker once the slot frees.
#[tokio::test]
async fn c5_full_ready_pin_waits_instead_of_moving() {
    let info = ServerInfo {
        max_running_requests: 1,
        ..ServerInfo::default()
    };
    let h = Harness::start_replicas(
        generating_mocks(info).await,
        &[("max_running_requests = 4", "max_running_requests = 1")],
    )
    .await;
    let session = create_session(&h).await;
    let (_, worker) = served(&h, turn(&h, &session, 0, &["hi"])).await;
    assert_eq!(worker, "gpu0");
    idle(&h).await;

    let held = Held::on_gpu0(&h).await;
    let gpu1_before = h.replica(1).dispatched_rids().len();
    let call = tokio::spawn(turn(&h, &session, 1, &["hi", REPLY, "more"]).send());
    until("the turn queued", || h.app.admission_snapshot().queued == 1).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(!call.is_finished(), "the turn waits for its pinned worker");
    assert_eq!(h.app.admission_snapshot().queued, 1);
    assert_eq!(h.replica(1).dispatched_rids().len(), gpu1_before);

    held.release(&h).await;
    let r = call.await.unwrap().unwrap();
    assert_eq!(r.status(), 200);
    let (_, worker) = served_reply(&h, r.json().await.unwrap());
    assert_eq!(
        worker, "gpu0",
        "dispatched to the pinned worker after a slot freed"
    );
    assert_eq!(h.replica(1).dispatched_rids().len(), gpu1_before);
}

/// Probes only on demand during the test (the background loop's next tick is a minute away), so the registry
/// edits below are not overwritten by a readiness publish before the request is granted.
fn quiet_tuning() -> Tuning {
    Tuning {
        probe_interval: Duration::from_secs(60),
        ..test_tuning()
    }
}

/// Starts two replicas and lets every background probe that is already running finish first (each replica's
/// probes are serialized in FIFO order).
async fn quiet_replicas() -> Harness {
    let h = Harness::start_replicas_tuned(mocks(2).await, &[], quiet_tuning()).await;
    for i in 0..h.replica_count() {
        assert!(h.app.readiness(i).probe().await.ready);
    }
    h
}

/// Moves `worker`'s epoch in the registry only (joining there, or `ready`): admission keeps its last pushed
/// view, so it still grants the worker at the old epoch, and that grant's `begin_work` is rejected.
fn move_registry_epoch(h: &Harness, worker: &str, ready: bool) {
    let mut workers = h.app.workers();
    let epoch = workers.get(worker).unwrap().epoch + 1;
    workers.register(worker, epoch, unix_ms()).unwrap();
    if ready {
        workers.observe(worker, epoch, true, unix_ms()).unwrap();
    }
}

/// Q7: the grant goes to `gpu0` (both idle, id tie-break), whose `begin_work` rejects it; the request is
/// re-queued once, granted on `gpu1` and served there at its epoch.
#[tokio::test]
async fn q7_rejected_grant_requeues_onto_the_other_worker() {
    let h = quiet_replicas().await;
    move_registry_epoch(&h, "gpu0", false);
    let r = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let reply: Value = r.json().await.unwrap();
    assert_eq!(reply["cere"]["worker_epoch"], 1, "{reply}");
    let (rid, worker) = served_reply(&h, reply);
    assert_eq!(worker, "gpu1");
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "completed", "{status}");
}

/// Q7: no other worker is eligible (`gpu1` is suspect), so the re-queued request goes back to `gpu0` once a
/// publish clears its suspect mark; that second grant is rejected too, which fails the request
/// `worker_not_ready` (503) before any backend sees it.
#[tokio::test]
async fn q7_second_rejection_fails_worker_not_ready() {
    let h = quiet_replicas().await;
    h.replica(1).set_server_info_override(Some(ServerInfo {
        kv_cache_dtype: "bf16".to_owned(),
        ..ServerInfo::default()
    }));
    assert!(!h.app.readiness(1).probe().await.ready);
    // `gpu0` is ready in the registry (so the readiness gate passes) at an epoch admission has not seen.
    move_registry_epoch(&h, "gpu0", true);
    let call = tokio::spawn(h.chat(&chat_body(false)).send());
    // Granted `gpu0` at epoch 1, rejected, re-queued: `gpu0` is marked suspect and `gpu1` is suspect.
    until("the re-queued request", || {
        h.app.admission_snapshot().queued == 1
    })
    .await;
    // The publish pushes `gpu0`'s registry view (epoch 2), which clears the mark and grants it there at once;
    // the registry moves on before the granted request runs (this runtime is single-threaded and nothing is
    // awaited in between), so its `begin_work` is rejected again.
    assert!(h.app.readiness(0).probe().await.ready);
    move_registry_epoch(&h, "gpu0", false);
    let r = call.await.unwrap().unwrap();
    assert_eq!(r.status(), 503);
    let rid = rid_of(&r);
    let err: Value = r.json().await.unwrap();
    assert_eq!(err["error"]["code"], "not_ready", "{err}");
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], "worker_not_ready", "{status}");
    assert_eq!(status["worker_id"], Value::Null, "{status}");
    for i in 0..2 {
        assert!(
            !h.replica(i).dispatched_rids().contains(&rid),
            "replica {i}"
        );
    }
}

fn journal_lines(h: &Harness) -> u64 {
    std::fs::read_to_string(h.state_dir.path().join("journal.jsonl"))
        .unwrap()
        .lines()
        .count() as u64
}

/// C7 (Q9): the `dispatched` record is the write that needs a compaction, and the compaction fails (its temp
/// path is a directory): the request fails `journal_error` (500) and neither mock receives its rid.
#[tokio::test]
async fn c7_journal_failure_at_dispatch_reaches_no_backend() {
    let h = Harness::start_replicas(mocks(2).await, &[]).await;
    // The open compacted the journal to these lines and nothing was appended since (fresh state dir).
    let compacted = journal_lines(&h);
    // A write compacts first once the file has more than `2 × compacted + slack` lines.
    let full = 2 * compacted + DEFAULT_COMPACTION_SLACK;
    std::fs::create_dir(h.state_dir.path().join("journal.jsonl.compact.tmp")).unwrap();
    // Fill to exactly `full` lines: a unary request journals `admitted`, `dispatched` and its terminal; a
    // session create journals one record.
    let requests = (full - compacted) / 3;
    for _ in 0..requests {
        let r = h.chat(&chat_body(false)).send().await.unwrap();
        assert_eq!(r.status(), 200);
    }
    until("every filler terminal journaled", || {
        journal_lines(&h) == compacted + 3 * requests
    })
    .await;
    for _ in 0..(full - compacted) % 3 {
        create_session(&h).await;
    }
    assert_eq!(journal_lines(&h), full);
    let dispatched: Vec<usize> = (0..2)
        .map(|i| h.replica(i).dispatched_rids().len())
        .collect();

    // `admitted` is the last line that fits; `dispatched` needs the failing compaction.
    let r = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 500);
    let rid = rid_of(&r);
    let err: Value = r.json().await.unwrap();
    assert_eq!(err["error"]["code"], "journal_error", "{err}");
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], "journal_error", "{status}");
    assert_eq!(status["worker_id"], Value::Null, "{status}");
    for (i, &before) in dispatched.iter().enumerate() {
        assert_eq!(h.replica(i).dispatched_rids().len(), before, "replica {i}");
    }
    let records = journal(h.state_dir.path());
    assert!(records
        .iter()
        .any(|r| r["type"] == "admitted" && r["rid"] == rid.as_str()));
    assert!(!records
        .iter()
        .any(|r| r["rid"] == rid.as_str() && r["type"] != "admitted"));
}
