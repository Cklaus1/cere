//! Remote readiness (m35 design note Q7a, Q9, Q10, Q10a, P11; build-loop-m35 design note "CREG split"): an
//! in-process federated `cered` dials an in-process stub agent (`support::stub_agent`, which enforces the agent's
//! wire rules) while the stub's link loop registers and heartbeats as the agent. Joining -> Ready runs
//! DescribeResources (hash and pins), reconciliation (ListOperations + CancelWork of unowned live ops), the
//! remote smoke (SubmitWork + StreamWork) and only then Ready; `/readyz` shows every worker with the 15 keys
//! (m36 design note Q11: `model` is null for a local worker, the remote model object otherwise).

mod support;

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cere_mock_sglang::MockConfig;
use cere_runtime::app::Tuning;
use cere_runtime::fed::registry::{RemoteRegistry, RemoteState};
use cere_runtime::fed::wire::{self, Inventory};
use cere_runtime::journal::Node;
use cere_runtime::sglang::Rid;
use cere_runtime::worker::WorkerState;
use serde_json::{json, Value};
use support::stub_agent::{StubAgent, StubOp, WorkOutcome, STUB_INSTANCE_ID};
use support::{fast_tuning, readyz, readyz_until, Federation, Harness, MODEL};

const A: &str = "worker-a";
const PRINCIPAL_A: &str = "cere-worker-a";
const TIMINGS: &str =
    "heartbeat_interval_ms = 100\nadmission_lease_ms = 1000\nsuspect_after_ms = 2000\n\
                       lost_after_ms = 4000\n";
const LEASE_MS: u64 = 1000;
/// The stub agent's heartbeat period.
const HEARTBEAT: Duration = Duration::from_millis(50);
const WORKER_KEYS: [&str; 15] = [
    "state",
    "epoch",
    "ready",
    "instance_id",
    "served_model_name",
    "smoke",
    "problems",
    "probed_at_unix_ms",
    "node",
    "model_generation",
    "lease_ms_remaining",
    "term",
    "inventory",
    "inventory_sha256",
    "model",
];
const REMOTE_ONLY_KEYS: [&str; 6] = [
    "model_generation",
    "lease_ms_remaining",
    "term",
    "inventory",
    "inventory_sha256",
    "model",
];

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// `[[federation.workers]]` for `worker_id` dialled at the stub, pinned like [`Federation::toml`].
fn worker_toml(worker_id: &str, principal: &str, stub: &StubAgent) -> String {
    format!(
        "[[federation.workers]]\nworker_id = \"{worker_id}\"\nendpoint = \"https://{}\"\n\
         principal = \"{principal}\"\nmax_inflight = 4\n\n[federation.workers.expect]\n\
         served_model_name = \"{MODEL}\"\ntp_size = 1\nkv_cache_dtype = \"fp8_e4m3\"\n\n",
        stub.addr()
    )
}

/// A federated `cered` (probing every 100 ms) whose remote worker `A` is the stub; the stub is not linked yet.
async fn start(fed: &Federation, stub: &StubAgent) -> Harness {
    start_tuned(fed, stub, fast_tuning()).await
}

async fn start_tuned(fed: &Federation, stub: &StubAgent, tuning: Tuning) -> Harness {
    let mut text = fed.toml(TIMINGS, &[]);
    text.push_str(&worker_toml(A, PRINCIPAL_A, stub));
    Harness::start_federated(MockConfig::default(), &text, tuning).await
}

/// The stub (scripted by `script` before anything dials it), a `cered` dialling it, then the stub's link loop.
async fn world(script: impl FnOnce(&StubAgent)) -> (Federation, StubAgent, Harness) {
    let fed = Federation::new();
    let mut stub = StubAgent::start(&fed, A, PRINCIPAL_A, 1).await;
    script(&stub);
    let h = start(&fed, &stub).await;
    stub.link(&fed, HEARTBEAT);
    (fed, stub, h)
}

fn registry(h: &Harness) -> &RemoteRegistry {
    h.app.remote_workers().expect("[federation] configured")
}

fn state(h: &Harness) -> RemoteState {
    registry(h)
        .get(A, now_ms())
        .expect("configured remote worker")
}

fn remote(body: &Value) -> &Value {
    &body["workers"][A]
}

fn keys(entry: &Value) -> BTreeSet<&str> {
    entry
        .as_object()
        .unwrap_or_else(|| panic!("worker entry {entry}"))
        .keys()
        .map(String::as_str)
        .collect()
}

fn problem_fields(entry: &Value) -> Vec<&str> {
    entry["problems"]
        .as_array()
        .unwrap_or_else(|| panic!("problems of {entry}"))
        .iter()
        .map(|p| p["field"].as_str().unwrap())
        .collect()
}

async fn until_ready(h: &Harness) -> Value {
    let (_, body) = readyz_until(h, "worker-a ready", |b| remote(b)["state"] == "ready").await;
    remote(&body).clone()
}

async fn until_problem(h: &Harness, field: &str) -> Value {
    let (_, body) = readyz_until(h, &format!("a {field} problem"), |b| {
        problem_fields(remote(b)).contains(&field)
    })
    .await;
    remote(&body).clone()
}

#[tokio::test(flavor = "multi_thread")]
async fn joining_to_ready_end_to_end_with_the_exact_readyz_entries() {
    let fed = Federation::new();
    let mut stub = StubAgent::start(&fed, A, PRINCIPAL_A, 1).await;
    let h = start(&fed, &stub).await;
    let term = h.app.epoch();

    // Before an accepted Register: joining with every incarnation field null.
    let (_, body) = readyz(&h).await;
    let before = remote(&body);
    assert_eq!(keys(before), BTreeSet::from(WORKER_KEYS), "{before}");
    assert_eq!(before["state"], "joining");
    assert_eq!(before["ready"], false);
    assert_eq!(before["node"], json!(Node::Remote));
    for key in [
        "epoch",
        "term",
        "model_generation",
        "lease_ms_remaining",
        "inventory",
        "inventory_sha256",
        "instance_id",
        "probed_at_unix_ms",
    ] {
        assert!(before[key].is_null(), "{key} of {before}");
    }
    assert_eq!(before["smoke"], json!({"state": "pending"}));
    assert_eq!(
        before["model"],
        json!({"managed": null, "desired": "none", "state": null, "lock_sha256": null}),
        "never described: {before}"
    );
    assert_eq!(before["served_model_name"], MODEL);
    // Every local entry has the 15 keys, node local and the remote-only keys null (`model` too, Q11).
    let locals: Vec<_> = body["workers"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(id, _)| id.as_str() != A)
        .collect();
    assert_eq!(locals.len(), 1, "{body}");
    for (id, entry) in locals {
        assert_eq!(keys(entry), BTreeSet::from(WORKER_KEYS), "{id}: {entry}");
        assert_eq!(entry["node"], json!(Node::Local), "{id}: {entry}");
        assert_eq!(entry["state"], "ready", "{id}: {entry}");
        for key in REMOTE_ONLY_KEYS {
            assert!(entry[key].is_null(), "{id}: {key} of {entry}");
        }
    }
    // The top-level readiness counts local workers only.
    assert_eq!(body["ready"], true, "{body}");

    stub.link(&fed, HEARTBEAT);
    let entry = until_ready(&h).await;
    assert_eq!(keys(&entry), BTreeSet::from(WORKER_KEYS), "{entry}");
    let bytes = stub.inventory_bytes();
    assert_eq!(entry["epoch"], 1, "{entry}");
    assert_eq!(entry["ready"], true, "{entry}");
    assert_eq!(entry["node"], json!(Node::Remote), "{entry}");
    assert_eq!(entry["term"], term, "{entry}");
    assert_eq!(entry["model_generation"], 1, "{entry}");
    assert_eq!(entry["instance_id"], STUB_INSTANCE_ID, "{entry}");
    assert_eq!(entry["served_model_name"], MODEL, "{entry}");
    assert_eq!(
        entry["smoke"],
        json!({"state": "passed", "instance_id": STUB_INSTANCE_ID}),
        "{entry}"
    );
    assert_eq!(entry["problems"], json!([]), "{entry}");
    assert_eq!(
        entry["model"],
        json!({"managed": false, "desired": "none", "state": "attached", "lock_sha256": null}),
        "the stub's attached-mode inventory: {entry}"
    );
    assert_eq!(
        entry["inventory"],
        serde_json::to_value(stub.inventory()).unwrap()
    );
    assert_eq!(entry["inventory_sha256"], hex(&Inventory::sha256(&bytes)));
    let lease = entry["lease_ms_remaining"].as_u64().unwrap();
    assert!(lease <= LEASE_MS, "{entry}");
    let probed = entry["probed_at_unix_ms"].as_u64().unwrap();
    assert!(probed <= now_ms() && probed + 10_000 > now_ms(), "{entry}");

    // Describe, reconcile (one page, nothing to cancel), then the smoke.
    assert_eq!(
        stub.calls(),
        [
            "DescribeResources",
            "ListOperations",
            "SubmitWork",
            "StreamWork"
        ]
    );
    let submits = stub.submits();
    let smoke = &submits[0];
    let authority = smoke.authority.as_ref().unwrap();
    assert_eq!(authority.scope_id, "cere-smoke");
    assert_eq!(authority.operation_id, smoke.request_id);
    assert_eq!(authority.coordinator_term, term);
    assert_eq!(authority.model_generation, Some(1));
    assert!(authority.remaining_budget_ms > 0 && authority.remaining_budget_ms <= 10_000);
    assert_eq!(smoke.attempt, 1);
    let unit: Value = serde_json::from_slice(&smoke.work_unit_json).unwrap();
    assert_eq!(unit, json!({"kind": "chat", "stream": false}));
    assert_eq!(Rid::parse(&smoke.request_id).unwrap().epoch(), term);
    let body: Value = serde_json::from_slice(&smoke.bounded_inference_input_json).unwrap();
    assert_eq!(body["rid"], smoke.request_id);
    assert_eq!(body["model"], MODEL);
    assert_eq!(body["max_completion_tokens"], 1);
    assert_eq!(body["temperature"], 0);
    assert_eq!(body["stream"], false);
    assert!(stub.cancels().is_empty());
    let s = state(&h);
    assert_eq!(s.reconciled_generation, Some(1));
    assert_eq!(
        s.inventory_sha256,
        Some(Inventory::sha256(&bytes)),
        "the hash of the exact described bytes"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn inventory_hash_mismatch_keeps_joining_until_the_agent_reports_the_described_bytes() {
    let (_fed, stub, h) = world(|stub| stub.report_sha256(Some([9; 32]))).await;
    let entry = until_problem(&h, "inventory_sha256").await;
    assert_eq!(entry["state"], "joining", "{entry}");
    assert_eq!(entry["ready"], false, "{entry}");
    assert!(entry["inventory"].is_null(), "not recorded: {entry}");
    assert_eq!(stub.count("ListOperations"), 0);
    assert_eq!(stub.count("SubmitWork"), 0);

    stub.report_sha256(None);
    let entry = until_ready(&h).await;
    assert_eq!(entry["problems"], json!([]), "{entry}");
    assert!(stub.count("DescribeResources") >= 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn pin_mismatch_keeps_joining_with_a_problem() {
    let (_fed, stub, h) = world(|stub| {
        let mut inventory = stub.inventory();
        inventory.tp_size = Some(2);
        stub.set_inventory(inventory);
    })
    .await;
    let entry = until_problem(&h, "tp_size").await;
    assert_eq!(entry["state"], "joining", "{entry}");
    assert_eq!(entry["ready"], false, "{entry}");
    assert_eq!(entry["inventory"]["tp_size"], 2, "{entry}");
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(stub.count("ListOperations"), 0);
    assert_eq!(stub.count("SubmitWork"), 0);
    assert_eq!(remote(&readyz(&h).await.1)["state"], "joining");

    // The agent's backend changes to the pinned shape: a new hash, described again, then Ready.
    let mut inventory = stub.inventory();
    inventory.tp_size = Some(1);
    stub.set_inventory(inventory);
    let entry = until_ready(&h).await;
    assert_eq!(entry["problems"], json!([]), "{entry}");
}

#[tokio::test(flavor = "multi_thread")]
async fn unowned_live_operations_are_cancelled_with_the_pinned_id_before_ready() {
    let fed = Federation::new();
    let mut stub = StubAgent::start(&fed, A, PRINCIPAL_A, 1).await;
    let h = start(&fed, &stub).await;
    let term = h.app.epoch();
    // 300 finished ops fill more than one page; the unowned live op sorts after them.
    for n in 0..300 {
        stub.add_op(StubOp {
            terminal: true,
            cleanup_acknowledged: true,
            ..StubOp::submit("s0", &format!("op-{n:03}"), 1, 1, term)
        });
    }
    let live = StubOp::submit("s1", "orphan-rid", 1, 1, term);
    stub.add_op(live.clone());
    stub.link(&fed, HEARTBEAT);

    until_ready(&h).await;
    assert!(stub.count("ListOperations") >= 2, "{:?}", stub.calls());
    let cancels = stub.cancels();
    assert_eq!(cancels.len(), 1, "{cancels:?}");
    let cancel = &cancels[0];
    let authority = cancel.authority.as_ref().unwrap();
    assert_eq!(authority.scope_id, "s1");
    assert_eq!(authority.operation_id, format!("orphan-rid/cancel/{term}"));
    assert_eq!(authority.model_generation, Some(1));
    assert_eq!(authority.coordinator_term, term);
    assert_eq!(cancel.request_id, live.request_id);
    assert_eq!(cancel.attempt, live.attempt);
    assert_eq!(cancel.completion_ticket, live.ticket);
    let cancelled = stub.op("s1", "orphan-rid").unwrap();
    assert!(cancelled.terminal);
    assert_eq!(cancelled.reason, wire::CANCEL_REQUESTED);
    let calls = stub.calls();
    let at = |m: &str| calls.iter().position(|c| *c == m).unwrap();
    assert!(at("CancelWork") < at("SubmitWork"), "{calls:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn smoke_failure_keeps_joining_and_cancels_the_smoke() {
    let (_fed, stub, h) = world(|stub| stub.set_smoke_outcome(WorkOutcome::Break)).await;
    let (_, body) = readyz_until(&h, "a failed smoke", |b| {
        remote(b)["smoke"]["state"] == "failed"
    })
    .await;
    let entry = remote(&body);
    assert_eq!(entry["state"], "joining", "{entry}");
    assert_eq!(entry["ready"], false, "{entry}");
    assert!(problem_fields(entry).contains(&"smoke"), "{entry}");
    let first = stub.submits()[0].clone();
    // The detached CancelWork of the failed smoke.
    let mut cancels = stub.cancels();
    for _ in 0..250 {
        if !cancels.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        cancels = stub.cancels();
    }
    let cancel = cancels
        .iter()
        .find(|c| c.request_id == first.request_id)
        .unwrap_or_else(|| panic!("no CancelWork of the smoke: {cancels:?}"));
    let authority = cancel.authority.as_ref().unwrap();
    assert_eq!(authority.scope_id, "cere-smoke");
    assert_eq!(
        authority.operation_id,
        format!("{}/cancel/1", first.request_id)
    );
    let smoke_op = stub.op("cere-smoke", &first.request_id).unwrap();
    assert_eq!(cancel.completion_ticket, smoke_op.ticket);
    assert!(smoke_op.terminal);

    stub.set_smoke_outcome(WorkOutcome::Complete);
    let entry = until_ready(&h).await;
    assert_eq!(entry["smoke"]["state"], "passed", "{entry}");
    let last = stub.submits().last().unwrap().clone();
    assert_ne!(
        last.request_id, first.request_id,
        "every smoke has its own rid"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn generation_change_while_ready_rejoins_and_reruns_every_check() {
    let (_fed, stub, h) = world(|_| {}).await;
    until_ready(&h).await;
    let round = state(&h).round;
    assert_eq!(stub.count("DescribeResources"), 1);

    // Held in reconciliation, so the Joining entry is observable.
    stub.hold_list(true);
    stub.set_generation(2);
    let (_, body) = readyz_until(&h, "joining at generation 2", |b| {
        remote(b)["state"] == "joining" && remote(b)["model_generation"] == 2
    })
    .await;
    let entry = remote(&body);
    assert_eq!(entry["ready"], false, "{entry}");
    assert_eq!(entry["smoke"], json!({"state": "pending"}), "{entry}");
    assert!(state(&h).round > round);

    stub.hold_list(false);
    let entry = until_ready(&h).await;
    assert_eq!(entry["model_generation"], 2, "{entry}");
    assert_eq!(entry["inventory"]["model_generation"], 2, "{entry}");
    assert_eq!(
        entry["inventory_sha256"],
        hex(&Inventory::sha256(&stub.inventory_bytes()))
    );
    assert_eq!(stub.count("DescribeResources"), 2);
    assert_eq!(stub.count("SubmitWork"), 2);
    assert_eq!(state(&h).reconciled_generation, Some(2));
    let smoke = stub.submits().last().unwrap().clone();
    assert_eq!(smoke.authority.unwrap().model_generation, Some(2));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_result_computed_in_an_old_round_is_dropped() {
    let (_fed, stub, h) = world(|stub| stub.hold_list(true)).await;
    for _ in 0..500 {
        if stub.count("ListOperations") > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(stub.count("ListOperations"), 1, "{:?}", stub.calls());
    let round = state(&h).round;
    let next = registry(&h).enter_joining(A, round).unwrap();
    assert_eq!(next, round + 1);
    stub.hold_list(false);

    until_ready(&h).await;
    let s = state(&h);
    assert_eq!(s.round, next);
    assert_eq!(s.reconciled_generation, Some(1));
    // The reconciliation of the old round was dropped and the new round described and reconciled again.
    assert_eq!(stub.count("DescribeResources"), 2, "{:?}", stub.calls());
    assert_eq!(stub.count("ListOperations"), 2, "{:?}", stub.calls());
    assert_eq!(stub.count("SubmitWork"), 1, "{:?}", stub.calls());
}

/// Q13 (live L2 `e_drain_partition`): the agent fences while draining (`WORKER_PROTOCOL.md` Q7a), so its
/// `fenced` reports keep a draining remote worker Draining in its round instead of sending it back to Joining; a
/// new incarnation of a draining worker passes Q10 without the smoke (the agent refuses SubmitWork while draining)
/// and reads draining; undrain starts a new round whose smoke runs before Ready.
#[tokio::test(flavor = "multi_thread")]
async fn a_draining_worker_stays_draining_across_a_new_epoch_and_undrain_smokes_again() {
    let (_fed, stub, h) = world(|_| {}).await;
    until_ready(&h).await;
    let round = state(&h).round;
    h.app.set_drain_intent(A, true).unwrap();
    for _ in 0..250 {
        if stub.admission() == wire::AdmissionState::Fenced {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(stub.admission(), wire::AdmissionState::Fenced);
    // Ten heartbeats reporting `fenced`.
    tokio::time::sleep(HEARTBEAT * 10).await;
    let s = state(&h);
    assert_eq!((s.state, s.round), (WorkerState::Draining, round), "{s:?}");
    assert_eq!(remote(&readyz(&h).await.1)["state"], "draining");

    stub.restart(2);
    let (_, body) = readyz_until(&h, "worker-a draining at epoch 2", |b| {
        remote(b)["state"] == "draining" && remote(b)["epoch"] == 2
    })
    .await;
    let entry = remote(&body);
    assert_eq!(entry["ready"], false, "{entry}");
    assert_eq!(stub.count("SubmitWork"), 1, "{:?}", stub.calls());
    assert_eq!(state(&h).reconciled_generation, Some(1));

    h.app.set_drain_intent(A, false).unwrap();
    let entry = until_ready(&h).await;
    assert_eq!(entry["epoch"], 2, "{entry}");
    assert_eq!(entry["smoke"]["state"], "passed", "{entry}");
    assert_eq!(stub.count("SubmitWork"), 2, "{:?}", stub.calls());
    let smoke = stub.submits().last().unwrap().clone();
    assert_eq!(smoke.authority.unwrap().target.unwrap().worker_epoch, 2);
}

/// O182 (rt-flake2 design note L2): the remote driver follows the agent's heartbeats, not its period. With a
/// period no step of this test reaches (an hour), the worker joins (the first heartbeat's generation starts the
/// reconcile, the agent's `open` report the smoke), a new incarnation of a draining worker reads draining (no report
/// to wait for after the reconcile) and undrain smokes it Ready again.
#[tokio::test(flavor = "multi_thread")]
async fn the_remote_driver_follows_heartbeats_not_its_period() {
    let fed = Federation::new();
    let mut stub = StubAgent::start(&fed, A, PRINCIPAL_A, 1).await;
    let tuning = Tuning {
        remote_interval: Duration::from_secs(3600),
        ..fast_tuning()
    };
    let h = start_tuned(&fed, &stub, tuning).await;
    stub.link(&fed, HEARTBEAT);
    let entry = until_ready(&h).await;
    assert_eq!(entry["smoke"]["state"], "passed", "{entry}");

    h.app.set_drain_intent(A, true).unwrap();
    stub.restart(2);
    let (_, body) = readyz_until(&h, "worker-a draining at epoch 2", |b| {
        remote(b)["state"] == "draining" && remote(b)["epoch"] == 2
    })
    .await;
    assert_eq!(remote(&body)["ready"], false, "{body}");

    h.app.set_drain_intent(A, false).unwrap();
    let entry = until_ready(&h).await;
    assert_eq!(entry["epoch"], 2, "{entry}");
    assert_eq!(stub.count("SubmitWork"), 2, "{:?}", stub.calls());
}

/// Review K2: the problems of the worker's Joining round outlive it. A worker that fails to join with a pin
/// problem and then goes Lost (its agent silent) still shows the pin problem on `/readyz` across later probe
/// passes (100 ms interval), which refresh only the residency-owned problems.
#[tokio::test(flavor = "multi_thread")]
async fn a_lost_worker_keeps_the_problems_of_its_joining_round() {
    let (_fed, mut stub, h) = world(|stub| {
        let mut inventory = stub.inventory();
        inventory.tp_size = Some(2);
        stub.set_inventory(inventory);
    })
    .await;
    let entry = until_problem(&h, "tp_size").await;
    assert_eq!(entry["state"], "joining", "{entry}");
    let round = state(&h).round;

    stub.unlink();
    readyz_until(&h, "worker-a lost", |b| remote(b)["state"] == "lost").await;
    // At least three probe passes of the Lost worker.
    tokio::time::sleep(Duration::from_millis(350)).await;
    let (_, body) = readyz(&h).await;
    let entry = remote(&body);
    assert_eq!(entry["state"], "lost", "{entry}");
    assert_eq!(state(&h).round, round);
    assert_eq!(problem_fields(entry), ["tp_size"], "{entry}");
}
