//! M3.1 (m3 design note C2, C3, C8 `/readyz`): one `cered` routes over N independent `sglang_http` replicas,
//! one worker per device. The worker id is the device; each replica is probed and smoked against its own backend
//! and publishes into the registry and admission; a worker's epoch moves only on its own new launch or on a
//! rejoin after lost; `/readyz` reports process-level problems plus one entry per worker (Q11).
mod support;

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;

use cere_mock_sglang::{MockConfig, MockSglang, ServerInfo};
use serde_json::{json, Value};
use support::*;

/// Q11 top-level `/readyz` keys.
const READYZ_KEYS: [&str; 5] = ["problems", "ready", "stopping", "tenancy", "workers"];
/// Q11 per-worker `/readyz` keys (m35 design note P11: local and remote alike; m36 design note Q11 adds
/// `model`), sorted.
const READYZ_WORKER_KEYS: [&str; 15] = [
    "epoch",
    "instance_id",
    "inventory",
    "inventory_sha256",
    "lease_ms_remaining",
    "model",
    "model_generation",
    "node",
    "probed_at_unix_ms",
    "problems",
    "ready",
    "served_model_name",
    "smoke",
    "state",
    "term",
];

/// `scheduler.worker_lost_after_ms` at its 1 s floor.
const LOST_AFTER_1S: (&str, &str) = (
    "mode = \"replica\"",
    "mode = \"replica\"\nworker_lost_after_ms = 1000",
);

fn keys(value: &Value) -> Vec<&str> {
    value
        .as_object()
        .unwrap_or_else(|| panic!("not an object: {value}"))
        .keys()
        .map(String::as_str)
        .collect()
}

fn problem_fields(problems: &Value) -> Vec<&str> {
    problems
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["field"].as_str().unwrap())
        .collect()
}

/// Journaled `worker_epoch` records of `worker`, in file order.
fn worker_epochs(state_dir: &Path, worker: &str) -> Vec<u64> {
    journal(state_dir)
        .iter()
        .filter(|r| r["type"] == "worker_epoch" && r["worker"] == worker)
        .map(|r| r["epoch"].as_u64().unwrap())
        .collect()
}

/// Asserts one Q11 worker entry that is ready and smoked, and returns its epoch.
fn assert_ready_worker(body: &Value, worker: &str) -> u64 {
    let w = &body["workers"][worker];
    assert_eq!(keys(w), READYZ_WORKER_KEYS, "{worker}: {body}");
    assert_eq!(w["state"], "ready", "{worker}: {body}");
    assert_eq!(w["ready"], true, "{worker}: {body}");
    assert_eq!(w["served_model_name"], MODEL, "{worker}: {body}");
    assert_eq!(w["problems"], json!([]), "{worker}: {body}");
    assert!(w["model"].is_null(), "a local worker: {worker}: {body}");
    let instance = w["instance_id"].as_str().unwrap();
    assert!(instance.starts_with("sglang-"), "{worker}: {body}");
    assert_eq!(
        w["smoke"],
        json!({"state": "passed", "instance_id": instance}),
        "{worker}: {body}"
    );
    assert!(
        w["probed_at_unix_ms"].as_u64().is_some(),
        "{worker}: {body}"
    );
    let epoch = w["epoch"].as_u64().unwrap();
    assert!(epoch >= 1, "{worker}: {body}");
    epoch
}

/// C8: the single form reports exactly the Q11 shape, one worker named by the device (`gpu1` in the t4 profile).
#[tokio::test]
async fn c8_readyz_single_form_is_the_q11_shape() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    let (status, body) = readyz(&h).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(keys(&body), READYZ_KEYS, "{body}");
    assert_eq!(body["ready"], true, "{body}");
    assert_eq!(body["stopping"], false, "{body}");
    assert_eq!(body["problems"], json!([]), "{body}");
    assert!(body["tenancy"].is_object(), "{body}");
    assert_eq!(keys(&body["workers"]), ["gpu1"], "{body}");
    assert_eq!(assert_ready_worker(&body, "gpu1"), 1, "{body}");
}

/// C3: two replicas, both ready in `/readyz`, each smoked against its own backend (one smoke per mock).
#[tokio::test]
async fn c3_two_replicas_are_both_ready() {
    let h = Harness::start_replicas(mocks(2).await, &[]).await;
    let (status, body) = readyz(&h).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(keys(&body), READYZ_KEYS, "{body}");
    assert_eq!(keys(&body["workers"]), ["gpu0", "gpu1"], "{body}");
    assert_eq!(assert_ready_worker(&body, "gpu0"), 1, "{body}");
    assert_eq!(assert_ready_worker(&body, "gpu1"), 1, "{body}");
    assert_ne!(
        body["workers"]["gpu0"]["instance_id"], body["workers"]["gpu1"]["instance_id"],
        "{body}"
    );
    for i in 0..2 {
        assert_eq!(h.replica(i).dispatched_rids().len(), 1, "replica {i}");
    }
}

/// C3/Q10: a pin drift on one replica makes only that worker suspect (no epoch change), `/readyz` stays 200 on
/// the other; the same instance returning before `worker_lost_after_ms` is ready again at the same epoch.
#[tokio::test]
async fn c3_drift_suspects_only_that_worker() {
    let h = Harness::start_replicas_tuned(mocks(2).await, &[], fast_tuning()).await;
    let (_, before) = readyz(&h).await;
    let old_instance = before["workers"]["gpu1"]["instance_id"].clone();
    h.replica(1).set_server_info_override(Some(ServerInfo {
        kv_cache_dtype: "bf16".to_owned(),
        ..ServerInfo::default()
    }));
    let (status, body) = readyz_until(&h, "gpu1 suspect", |b| {
        b["workers"]["gpu1"]["state"] == "suspect"
    })
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["ready"], true, "{body}");
    assert_eq!(body["problems"], json!([]), "{body}");
    assert_eq!(assert_ready_worker(&body, "gpu0"), 1, "{body}");
    let w = &body["workers"]["gpu1"];
    assert_eq!(keys(w), READYZ_WORKER_KEYS, "{body}");
    assert_eq!(w["ready"], false, "{body}");
    assert_eq!(w["epoch"], 1, "{body}");
    assert_eq!(problem_fields(&w["problems"]), ["kv_cache_dtype"], "{body}");

    // Chat still works: it lands on the ready worker only.
    let r = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let reply: Value = r.json().await.unwrap();
    assert_eq!(reply["cere"]["worker_id"], "gpu0", "{reply}");

    h.replica(1).set_server_info_override(None);
    let (status, body) = readyz_until(&h, "gpu1 ready again", |b| {
        b["workers"]["gpu1"]["state"] == "ready"
    })
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(assert_ready_worker(&body, "gpu1"), 1, "{body}");
    assert_eq!(body["workers"]["gpu1"]["instance_id"], old_instance);
    assert_eq!(worker_epochs(h.state_dir.path(), "gpu1"), [1]);
}

/// C2/Q4: a new `instance_id` on one replica bumps only that worker's epoch, journaled at detection before its
/// re-smoke (the mock holds the smoke while `/readyz` shows it pending); the other worker keeps epoch 1.
#[tokio::test]
async fn c2_new_instance_bumps_only_that_worker_before_its_resmoke() {
    let h = Harness::start_replicas_tuned(mocks(2).await, &[], fast_tuning()).await;
    let (_, before) = readyz(&h).await;
    let old_instance = before["workers"]["gpu1"]["instance_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let smokes = h.replica(1).dispatched_rids().len();

    h.replica(1).set_load_lag(Duration::from_millis(800));
    h.replica(1)
        .set_server_info_override(Some(launched(7, 12.5)));
    let (status, pending) = readyz_until(&h, "gpu1 re-smoke pending", |b| {
        let w = &b["workers"]["gpu1"];
        w["smoke"]["state"] == "pending"
            && w["instance_id"].is_string()
            && w["instance_id"] != old_instance.as_str()
    })
    .await;
    // The bump is already journaled and visible while the smoke is held.
    assert_eq!(
        worker_epochs(h.state_dir.path(), "gpu1"),
        [1, 2],
        "{pending}"
    );
    assert_eq!(pending["workers"]["gpu1"]["epoch"], 2, "{pending}");
    assert_eq!(pending["workers"]["gpu1"]["ready"], false, "{pending}");
    assert_ne!(pending["workers"]["gpu1"]["state"], "ready", "{pending}");
    assert_eq!(status, 200, "the other worker serves: {pending}");
    assert_eq!(assert_ready_worker(&pending, "gpu0"), 1, "{pending}");

    let (_, after) = readyz_until(&h, "gpu1 ready under the new launch", |b| {
        b["workers"]["gpu1"]["state"] == "ready"
    })
    .await;
    assert_eq!(assert_ready_worker(&after, "gpu1"), 2, "{after}");
    assert_eq!(
        after["workers"]["gpu1"]["instance_id"],
        pending["workers"]["gpu1"]["instance_id"]
    );
    assert_eq!(assert_ready_worker(&after, "gpu0"), 1, "{after}");
    assert_eq!(
        h.replica(1).dispatched_rids().len(),
        smokes + 1,
        "one re-smoke"
    );
    assert_eq!(worker_epochs(h.state_dir.path(), "gpu0"), [1]);
    assert_eq!(worker_epochs(h.state_dir.path(), "gpu1"), [1, 2]);

    // Work dispatched to gpu1 now names its new epoch: both mocks hold each request, so two concurrent ones
    // take one worker each (least loaded).
    h.replica(0).set_load_lag(Duration::from_millis(300));
    h.replica(1).set_load_lag(Duration::from_millis(300));
    let calls: Vec<_> = (0..2).map(|_| h.chat(&chat_body(false)).send()).collect();
    let mut seen = BTreeSet::new();
    for r in futures_util::future::join_all(calls).await {
        let reply: Value = r.unwrap().json().await.unwrap();
        let worker = reply["cere"]["worker_id"].as_str().unwrap().to_owned();
        let expected = if worker == "gpu1" { 2 } else { 1 };
        assert_eq!(reply["cere"]["worker_epoch"], expected, "{reply}");
        seen.insert(worker);
    }
    assert_eq!(seen, BTreeSet::from(["gpu0".to_owned(), "gpu1".to_owned()]));
}

/// C2/Q4 rule 4: a worker that aged to lost and then is ready again rejoins at a bumped, journaled epoch.
#[tokio::test]
async fn c2_lost_then_ready_rejoins_at_a_bumped_epoch() {
    let h =
        Harness::start_replicas_tuned(mocks(2).await, &[LOST_AFTER_1S], probe_aging(fast_tuning()))
            .await;
    h.replica(1).set_server_info_override(Some(ServerInfo {
        kv_cache_dtype: "bf16".to_owned(),
        ..ServerInfo::default()
    }));
    let (status, body) =
        readyz_until(&h, "gpu1 lost", |b| b["workers"]["gpu1"]["state"] == "lost").await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["workers"]["gpu1"]["epoch"], 1, "{body}");
    h.replica(1).set_server_info_override(None);
    let (_, body) = readyz_until(&h, "gpu1 rejoined", |b| {
        b["workers"]["gpu1"]["state"] == "ready"
    })
    .await;
    assert_eq!(assert_ready_worker(&body, "gpu1"), 2, "{body}");
    assert_eq!(assert_ready_worker(&body, "gpu0"), 1, "{body}");
    assert_eq!(worker_epochs(h.state_dir.path(), "gpu1"), [1, 2]);
    assert_eq!(worker_epochs(h.state_dir.path(), "gpu0"), [1]);
}

/// C3: under concurrent load both replicas receive requests; each response names the worker whose mock saw
/// its rid, at that worker's epoch, and `epoch` stays the process epoch.
#[tokio::test]
async fn c3_requests_land_on_both_replicas() {
    let mut replicas = Vec::new();
    for _ in 0..2 {
        replicas.push(
            MockSglang::start(MockConfig {
                token_latency: Duration::from_millis(20),
                ..MockConfig::default()
            })
            .await
            .unwrap(),
        );
    }
    let h = Harness::start_replicas(replicas, &[]).await;
    let smokes: Vec<usize> = (0..2)
        .map(|i| h.replica(i).dispatched_rids().len())
        .collect();
    let mut replies = Vec::new();
    for _round in 0..3 {
        let calls: Vec<_> = (0..2).map(|_| h.chat(&chat_body(false)).send()).collect();
        for r in futures_util::future::join_all(calls).await {
            let r = r.unwrap();
            assert_eq!(r.status(), 200);
            replies.push(r.json::<Value>().await.unwrap());
        }
    }
    let mut per_worker = [0usize; 2];
    for reply in &replies {
        let cere = &reply["cere"];
        let rid = cere["request_id"].as_str().unwrap();
        let index = match cere["worker_id"].as_str().unwrap() {
            "gpu0" => 0,
            "gpu1" => 1,
            other => panic!("unknown worker {other}: {reply}"),
        };
        assert!(
            h.replica(index).dispatched_rids().iter().any(|d| d == rid),
            "{reply}"
        );
        assert_eq!(cere["worker_epoch"], 1, "{reply}");
        assert_eq!(cere["epoch"], h.app.epoch(), "{reply}");
        per_worker[index] += 1;
    }
    assert!(per_worker.iter().all(|&n| n > 0), "{per_worker:?}");
    for i in 0..2 {
        assert_eq!(
            h.replica(i).dispatched_rids().len() - smokes[i],
            per_worker[i],
            "replica {i}"
        );
    }
}

/// C2: neither the M1 local worker id constant nor a quoted local worker id is left (`grep -rnw` over
/// `crates/cere-runtime`, `scripts/live` and `tests`), except the recorded allowlist: the `transport` rules of
/// `scripts/cere_contracts.py:129-133` (outside the scanned trees) and their `tests/` fixtures, i.e. a line under
/// `tests/` that names `transport`; and a line naming the `"node"` key, whose value is a node kind (M3.5 `/readyz`
/// `node`, m35 design note Q10), not a worker id.
#[test]
fn c2_no_local_worker_id_is_left() {
    let root = repo_root();
    // Built so this file never matches itself.
    let patterns = [
        concat!("LOCAL_", "WORKER_ID"),
        concat!("\"", "lo", "cal", "\""),
    ];
    let allowed = |path: &str, line: &str| {
        (path.starts_with("tests/") && line.contains("transport")) || line.contains("\"node\"")
    };
    let mut hits = Vec::new();
    for dir in ["crates/cere-runtime", "tests"] {
        let mut stack = vec![root.join(dir)];
        while let Some(path) = stack.pop() {
            if path.is_dir() {
                for entry in std::fs::read_dir(&path).unwrap() {
                    stack.push(entry.unwrap().path());
                }
                continue;
            }
            let text = String::from_utf8_lossy(&std::fs::read(&path).unwrap()).into_owned();
            let relative = path.strip_prefix(&root).unwrap().display().to_string();
            for (number, line) in text.lines().enumerate() {
                if patterns.iter().any(|p| word_match(line, p)) && !allowed(&relative, line) {
                    hits.push(format!("{relative}:{}: {line}", number + 1));
                }
            }
        }
    }
    assert!(hits.is_empty(), "{}", hits.join("\n"));
}

/// `grep -w`: `pattern` occurs with no word character right before or after it.
fn word_match(line: &str, pattern: &str) -> bool {
    let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
    line.match_indices(pattern).any(|(at, _)| {
        !word(line[..at].chars().next_back()) && !word(line[at + pattern.len()..].chars().next())
    })
}
