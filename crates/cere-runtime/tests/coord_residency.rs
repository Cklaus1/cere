//! P5 coordinator residency (m36 design note Q2, Q8-Q11; design-m36-coord design note §2.2-2.3, §3).
//!
//! State half: the operator routes `POST /v1/cere/workers/{id}/load|unload` (drain auth model; 404
//! `worker_not_found`, 409 `model_unmanaged` for a local or attached worker, 409 `worker_not_described` before the
//! first describe, 202 `{worker_id, desired, intent_seq, model}` after the fsynced `worker_residency` record), the
//! registry's model view (kept across `join()`, Lost, a re-register and, for the intent, a coordinator restart),
//! the Q10 Ready gate of a managed worker, and an unload that makes a Ready worker ineligible before the route
//! answers.
//!
//! Drive half (§3.1, §3.2): the readiness driver issues exactly the Q8 table's `LoadModel`/`UnloadModel` calls
//! (none without an intent, for `failed` under the intent that saw it fail, for an inventory of an older epoch or
//! under another lock, which `/readyz` names `lock_mismatch`), at most one per (intent, epoch, term), retried with
//! the same op id and Authority until receipted, then polled to its terminal; a managed worker is Ready only
//! while its model is `ready` under the coordinator's lock and wanted, and is never pin-checked or smoked
//! otherwise.
//!
//! Two harnesses: a raw `WorkerRegistryService` client acting as the agent (as `coord_registry.rs`; the driver's
//! endpoint dials nothing, so the test drives the registry's transitions itself), and the in-process stub agent
//! describing a managed model under the coordinator's deployment lock (as `coord_dispatch.rs`).

mod support;

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cere_mock_sglang::{MockConfig, MockHandle, MockSglang};
use cere_proto::worker::v2 as pb;
use cere_runtime::fed::registry::{Desired, RemoteError, RemoteRegistry, RemoteState};
use cere_runtime::fed::wire::{self, Inventory, ModelState};
use cere_runtime::readiness::{SmokeStatus, REMOTE_SMOKE_SCOPE};
use cere_runtime::worker::WorkerState;
use pb::worker_registry_service_client::WorkerRegistryServiceClient;
use serde_json::{json, Value};
use support::stub_agent::{self, Lifecycle, StubAgent};
use support::{
    chat_body, fast_tuning, journal, lock_sha256_of, readyz, readyz_until, remote_only_b_text,
    replicas_config_text, test_tuning, Federation, Harness, StateDir, FEDERATION_BODY_TWEAK,
    FED_CLUSTER, MODEL, MODEL_B, OPERATOR_TOKEN, OPERATOR_TWEAK, TOKEN_A,
};
use tonic::transport::Channel;
use tonic::Code;

const A: &str = "worker-a";
const PRINCIPAL_A: &str = "cere-worker-a";
/// Raw-client harness timings (absolute ages): Lost 800 ms after the last sighting.
const FAST: &str =
    "heartbeat_interval_ms = 100\nadmission_lease_ms = 200\nsuspect_after_ms = 400\n\
                    lost_after_ms = 800\n";
/// Stub harness timings (as `coord_dispatch.rs`).
const TIMINGS: &str =
    "heartbeat_interval_ms = 100\nadmission_lease_ms = 1000\nsuspect_after_ms = 2000\n\
                       lost_after_ms = 4000\n";
/// The stub agent's heartbeat period.
const HEARTBEAT: Duration = Duration::from_millis(50);

type Client = WorkerRegistryServiceClient<Channel>;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn registry(h: &Harness) -> &RemoteRegistry {
    h.app.remote_workers().expect("[federation] configured")
}

fn state(h: &Harness) -> RemoteState {
    registry(h)
        .get(A, now_ms())
        .expect("configured remote worker")
}

fn local_id(h: &Harness) -> String {
    h.app.replicas()[0].worker_id().to_owned()
}

fn lock(h: &Harness) -> String {
    h.app.deployment(0).lock_sha256.clone()
}

async fn call(h: &Harness, worker: &str, action: &str, auth: Option<&str>) -> (u16, Value) {
    let mut request = h
        .http
        .post(h.url(&format!("/v1/cere/workers/{worker}/{action}")));
    if let Some(value) = auth {
        request = request.header("authorization", value);
    }
    let response = request.send().await.unwrap();
    let status = response.status().as_u16();
    let text = response.text().await.unwrap();
    (status, serde_json::from_str(&text).unwrap_or(Value::Null))
}

async fn operate(h: &Harness, worker: &str, action: &str) -> (u16, Value) {
    call(h, worker, action, Some(&format!("Bearer {OPERATOR_TOKEN}"))).await
}

fn error_code(body: &Value) -> &str {
    body["error"]["code"].as_str().unwrap_or_default()
}

fn residency_records(h: &Harness) -> Vec<Value> {
    journal(h.state_dir.path())
        .into_iter()
        .filter(|r| r["type"] == "worker_residency")
        .collect()
}

/// The stub's attached inventory turned managed: `state` under `lock` (identity fields per Q7).
fn managed(generation: u64, state: ModelState, lock: &str) -> Inventory {
    let mut inventory = stub_agent::inventory(A, generation);
    if !matches!(
        state,
        ModelState::Ready | ModelState::Draining | ModelState::Unloading
    ) {
        inventory.tp_size = None;
        inventory.kv_cache_dtype = None;
        inventory.max_running_requests = None;
        inventory.instance_id = None;
        inventory.models[0].instance_id = None;
    }
    inventory.models[0].lock_sha256 = Some(lock.to_owned());
    inventory.models[0].state = state;
    inventory
}

// ---------------------------------------------------------------------------------------------------------
// Raw registry client acting as the agent

/// A federated `cered` with the operator token and remote worker `A`, whose endpoint dials nothing.
async fn raw_start() -> (Harness, Federation) {
    let fed = Federation::new();
    let text = fed.toml(FAST, &[(A, PRINCIPAL_A)]);
    let h = Harness::start_federated_with(
        MockConfig::default(),
        &[OPERATOR_TWEAK],
        &text,
        test_tuning(),
    )
    .await;
    (h, fed)
}

fn agent(fed: &Federation) -> Client {
    WorkerRegistryServiceClient::new(fed.channel(fed.agent_tls(PRINCIPAL_A)))
}

fn identity(epoch: u64) -> Option<pb::WorkerIdentity> {
    Some(pb::WorkerIdentity {
        cluster_id: FED_CLUSTER.into(),
        worker_id: A.into(),
        worker_epoch: epoch,
    })
}

async fn register(client: &mut Client, epoch: u64) {
    let reply = client
        .register(pb::RegisterRequest {
            proposed: identity(epoch),
            boot_id: "0123456789abcdef0123456789abcdef".into(),
            protocol_revision: cere_proto::PROTOCOL_REVISION,
            inventory_sha256: vec![7; 32],
        })
        .await
        .unwrap()
        .into_inner();
    assert!(reply.accepted, "{reply:?}");
}

async fn heartbeat(client: &mut Client, h: &Harness, epoch: u64, sequence: u64) {
    let reply = client
        .heartbeat(pb::HeartbeatRequest {
            worker: identity(epoch),
            sequence,
            observed_coordinator_term: h.app.epoch(),
            inventory_sha256: vec![7; 32],
            model_generation: 1,
            admission_state: "open".into(),
            owner_only: false,
        })
        .await
        .unwrap()
        .into_inner();
    assert!(reply.accepted, "{reply:?}");
}

/// Records `inventory` as described in the current round (the driver's `record_inventory`).
fn describe(h: &Harness, inventory: Inventory) {
    let round = state(h).round;
    let sha = Inventory::sha256(&inventory.to_json_bytes().unwrap());
    registry(h)
        .record_inventory(A, round, inventory, sha)
        .unwrap();
}

/// Every M3.5 precondition of Ready in the current round (reconciled generation 1, an `open` heartbeat with a
/// live lease, a passed smoke), then `mark_ready`'s answer.
async fn try_ready(
    h: &Harness,
    client: &mut Client,
    epoch: u64,
    sequence: u64,
) -> Result<(), RemoteError> {
    let reg = registry(h);
    let round = state(h).round;
    reg.mark_reconciled(A, round, 1).unwrap();
    heartbeat(client, h, epoch, sequence).await;
    reg.set_smoke(
        A,
        round,
        SmokeStatus::Passed {
            instance_id: "remote-instance".into(),
        },
    )
    .unwrap();
    reg.mark_ready(A, round, now_ms())
}

#[tokio::test(flavor = "multi_thread")]
async fn without_an_operator_token_load_and_unload_are_404() {
    let fed = Federation::new();
    let text = fed.toml(FAST, &[(A, PRINCIPAL_A)]);
    let h = Harness::start_federated(MockConfig::default(), &text, test_tuning()).await;
    for action in ["load", "unload"] {
        for auth in [
            None,
            Some("Bearer secret-a"),
            Some("Bearer secret-operator"),
        ] {
            let (status, body) = call(&h, A, action, auth).await;
            assert_eq!(status, 404, "{action} {auth:?}: {body}");
            assert_eq!(error_code(&body), "not_found", "{body}");
        }
    }
    assert!(residency_records(&h).is_empty());
}

/// Q9: the drain auth model, the worker id, then the model checks; nothing is journaled on a refusal.
#[tokio::test(flavor = "multi_thread")]
async fn routes_check_the_bearer_the_worker_and_the_described_model() {
    let (h, fed) = raw_start().await;
    let local = local_id(&h);
    for action in ["load", "unload"] {
        for worker in [A, local.as_str(), "gpu9"] {
            let (status, body) = call(&h, worker, action, Some(&format!("Bearer {TOKEN_A}"))).await;
            assert_eq!(status, 403, "{action} {worker}: {body}");
            assert_eq!(error_code(&body), "forbidden", "{body}");
        }
        for auth in [None, Some("Bearer wrong"), Some("bearer secret-operator")] {
            let (status, body) = call(&h, A, action, auth).await;
            assert_eq!(status, 401, "{action} {auth:?}: {body}");
            assert_eq!(error_code(&body), "unauthorized", "{body}");
        }
        let (status, body) = operate(&h, "gpu9", action).await;
        assert_eq!(status, 404, "{body}");
        assert_eq!(error_code(&body), "worker_not_found", "{body}");
        // A local worker's backend owns its model (F8).
        let (status, body) = operate(&h, &local, action).await;
        assert_eq!(status, 409, "{body}");
        assert_eq!(error_code(&body), "model_unmanaged", "{body}");
        // Never described since this start: neither before nor after its Register.
        let (status, body) = operate(&h, A, action).await;
        assert_eq!(status, 409, "{body}");
        assert_eq!(error_code(&body), "worker_not_described", "{body}");
    }
    let mut client = agent(&fed);
    register(&mut client, 1).await;
    let (status, body) = operate(&h, A, "load").await;
    assert_eq!(
        (status, error_code(&body)),
        (409, "worker_not_described"),
        "{body}"
    );
    // Last described in attached mode.
    describe(&h, stub_agent::inventory(A, 1));
    for action in ["load", "unload"] {
        let (status, body) = operate(&h, A, action).await;
        assert_eq!(status, 409, "{body}");
        assert_eq!(error_code(&body), "model_unmanaged", "{body}");
    }
    assert!(residency_records(&h).is_empty());
    assert_eq!(state(&h).model.desired, Desired::None);
}

/// M36-13 / Q8: `managed`, the last described state and lock, `desired` and `intent_seq` survive `join()`, Lost
/// and a re-register at a higher epoch (whose inventory epoch then lags); a coordinator restart restores the
/// journaled intent. The routes answer for a Lost worker (partitioned).
#[tokio::test(flavor = "multi_thread")]
async fn the_model_view_survives_join_lost_reregister_and_restart() {
    let (h, fed) = raw_start().await;
    let lock = lock(&h);
    let mut client = agent(&fed);
    register(&mut client, 1).await;
    let before = state(&h);
    assert_eq!(
        (
            before.model.managed,
            before.model.state,
            before.inventory_epoch
        ),
        (None, None, None)
    );
    describe(&h, managed(1, ModelState::Loading, &lock));
    let described = state(&h);
    assert_eq!(described.model.managed, Some(true));
    assert_eq!(described.model.state, Some(ModelState::Loading));
    assert_eq!(described.model.lock_sha256.as_deref(), Some(lock.as_str()));
    assert_eq!(described.inventory_epoch, Some(1));
    let (status, body) = operate(&h, A, "load").await;
    assert_eq!(status, 202, "{body}");
    assert_eq!(
        body,
        json!({
            "worker_id": A,
            "desired": "resident",
            "intent_seq": 1,
            "model": {"managed": true, "desired": "resident", "state": "loading", "lock_sha256": lock},
        })
    );
    let view = (
        Some(true),
        Desired::Resident,
        1,
        Some(ModelState::Loading),
        Some(lock.clone()),
    );
    let model_of = |s: &RemoteState| {
        (
            s.model.managed,
            s.model.desired,
            s.model.intent_seq,
            s.model.state,
            s.model.lock_sha256.clone(),
        )
    };

    // join(): a new round clears the inventory, not the model view.
    let round = state(&h).round;
    registry(&h).enter_joining(A, round).unwrap();
    let joined = state(&h);
    assert_eq!(joined.round, round + 1);
    assert!(joined.inventory.is_none());
    assert_eq!(model_of(&joined), view);

    // Lost (no heartbeat for the absolute bound): kept, and the routes still answer.
    tokio::time::sleep(Duration::from_millis(900)).await;
    let lost = state(&h);
    assert_eq!(lost.state, WorkerState::Lost);
    assert_eq!(model_of(&lost), view);
    let (status, body) = operate(&h, A, "unload").await;
    assert_eq!(status, 202, "{body}");
    assert_eq!(body["intent_seq"], 2, "{body}");
    assert_eq!(body["model"]["desired"], "unloaded", "{body}");
    let view = (
        Some(true),
        Desired::Unloaded,
        2,
        Some(ModelState::Loading),
        Some(lock.clone()),
    );
    assert_eq!(model_of(&state(&h)), view);

    // A new incarnation: kept, with the inventory epoch of the last describe.
    register(&mut client, 2).await;
    let rejoined = state(&h);
    assert_eq!(
        (rejoined.state, rejoined.epoch),
        (WorkerState::Joining, Some(2))
    );
    assert_eq!(model_of(&rejoined), view);
    assert_eq!(rejoined.inventory_epoch, Some(1));

    let records = residency_records(&h);
    assert_eq!(records.len(), 2, "{records:?}");
    assert_eq!(records[1]["desired"], "unloaded", "{records:?}");
    assert_eq!(records[1]["intent_seq"], 2, "{records:?}");

    // A coordinator restart restores the intent; the model is unknown until the next describe.
    drop(client);
    let h = h.restart().await;
    let restored = state(&h);
    assert_eq!(restored.model.desired, Desired::Unloaded);
    assert_eq!(restored.model.intent_seq, 2);
    assert_eq!(restored.model.managed, None);
}

/// Q10: a managed worker is Ready only with model state `ready`, the coordinator's lock and `desired !=
/// unloaded`; an attached worker keeps the M3.5 rule.
#[tokio::test(flavor = "multi_thread")]
async fn mark_ready_gates_a_managed_worker_on_state_lock_and_desire() {
    let (h, fed) = raw_start().await;
    let lock = lock(&h);
    let mut client = agent(&fed);
    register(&mut client, 1).await;
    let mut sequence = 1;
    for (inventory, why) in [
        (managed(1, ModelState::Loading, &lock), "loading"),
        (managed(1, ModelState::Unloaded, &lock), "unloaded"),
        (
            managed(1, ModelState::Ready, &"0".repeat(64)),
            "another lock",
        ),
    ] {
        describe(&h, inventory);
        let refused = try_ready(&h, &mut client, 1, sequence).await;
        sequence += 1;
        assert_eq!(refused, Err(RemoteError::ModelNotReady), "{why}");
        assert_eq!(state(&h).state, WorkerState::Joining, "{why}");
    }
    // desired = unloaded refuses a ready model under the coordinator's lock.
    let (status, body) = operate(&h, A, "unload").await;
    assert_eq!(status, 202, "{body}");
    describe(&h, managed(1, ModelState::Ready, &lock));
    let refused = try_ready(&h, &mut client, 1, sequence).await;
    sequence += 1;
    assert_eq!(refused, Err(RemoteError::ModelNotReady), "desired unloaded");
    let (status, body) = operate(&h, A, "load").await;
    assert_eq!(status, 202, "{body}");
    assert_eq!(try_ready(&h, &mut client, 1, sequence).await, Ok(()));
    sequence += 1;
    assert_eq!(state(&h).state, WorkerState::Ready);

    // Attached: Ready as before (no lock, no residency intent).
    let round = state(&h).round;
    registry(&h).enter_joining(A, round).unwrap();
    describe(&h, stub_agent::inventory(A, 1));
    assert_eq!(try_ready(&h, &mut client, 1, sequence).await, Ok(()));
}

// ---------------------------------------------------------------------------------------------------------
// Stub agent describing a managed model

/// A federated `cered` (operator token, probing every 100 ms) whose local worker is drained and whose remote
/// worker `A` is the linked stub agent (chat served by `mock`) describing its managed model in a given state
/// under the coordinator's deployment `lock`.
struct Managed {
    h: Harness,
    stub: StubAgent,
    lock: String,
    _fed: Federation,
    _mock: MockHandle,
}

async fn managed_start(state: ModelState) -> Managed {
    managed_start_in(state, 0).await
}

/// [`managed_start`] with worker `A` in `deployment`: 0 is the coordinator's single entry ([`MODEL`]), 1 a second,
/// remote-only entry [`MODEL_B`] (M5 D8) whose model the stub describes under that entry's lock.
async fn managed_start_in(state: ModelState, deployment: usize) -> Managed {
    let model = match deployment {
        0 => MODEL,
        _ => MODEL_B,
    };
    let fed = Federation::new();
    let mut stub = StubAgent::start(&fed, A, PRINCIPAL_A, 1).await;
    let mock = MockSglang::start(MockConfig::default()).await.unwrap();
    stub.serve_chat(&mock.base_url());
    let workers = format!(
        "{}[[federation.workers]]\nworker_id = \"{A}\"\nendpoint = \"https://{}\"\n\
         principal = \"{PRINCIPAL_A}\"\nmax_inflight = 4\n\n[federation.workers.expect]\n\
         served_model_name = \"{model}\"\ntp_size = 1\nkv_cache_dtype = \"fp8_e4m3\"\n\n",
        fed.toml(TIMINGS, &[]),
        stub.addr()
    );
    let h = match deployment {
        0 => {
            Harness::start_federated_with(
                MockConfig::default(),
                &[OPERATOR_TWEAK],
                &workers,
                fast_tuning(),
            )
            .await
        }
        _ => {
            // Two entries refuse `base_url` (M5 D2(b)): A in the replicas form, then B remote-only.
            let local = MockSglang::start(MockConfig::default()).await.unwrap();
            let state_dir = StateDir::new();
            let mut config = replicas_config_text(
                &[local.base_url()],
                state_dir.path(),
                18390,
                &[FEDERATION_BODY_TWEAK, OPERATOR_TWEAK],
            );
            config.push('\n');
            config.push_str(&remote_only_b_text());
            config.push_str(&workers);
            Harness::start_config(vec![local], state_dir, config, fast_tuning()).await
        }
    };
    let lock = h.app.deployment(deployment).lock_sha256.clone();
    if deployment != 0 {
        let mut inventory = stub_agent::inventory(A, 1);
        inventory.served_model_name = MODEL_B.to_owned();
        inventory.models[0].served_model_name = MODEL_B.to_owned();
        stub.set_inventory(inventory);
    }
    stub.set_model(state, &lock);
    h.app.set_drain_intent(&local_id(&h), true).unwrap();
    stub.link(&fed, HEARTBEAT);
    Managed {
        h,
        stub,
        lock,
        _fed: fed,
        _mock: mock,
    }
}

/// Q9/Q10 (review M36-05): `unload` of a Ready managed worker takes it out of admission before the 202: right
/// after the answer the registry view is not Ready and a new request is not granted to it (the local worker is
/// drained, so nothing else serves it). `load` is the next intent and the worker becomes Ready again.
#[tokio::test(flavor = "multi_thread")]
async fn unload_makes_a_ready_worker_ineligible_before_the_route_answers() {
    let Managed {
        h,
        stub,
        lock,
        _fed,
        _mock,
    } = managed_start(ModelState::Ready).await;
    let (_, body) = readyz_until(&h, "managed worker-a ready", |b| {
        b["workers"][A]["state"] == "ready"
    })
    .await;
    assert_eq!(
        body["workers"][A]["model"],
        json!({"managed": true, "desired": "none", "state": "ready", "lock_sha256": lock})
    );
    let r = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 200, "served by worker-a while Ready");
    let work = |stub: &StubAgent| {
        stub.submits()
            .iter()
            .filter(|s| s.authority.as_ref().unwrap().scope_id != REMOTE_SMOKE_SCOPE)
            .count()
    };
    let served = work(&stub);
    assert_eq!(served, 1);

    let (status, body) = operate(&h, A, "unload").await;
    assert_eq!(status, 202, "{body}");
    assert_eq!(
        body,
        json!({
            "worker_id": A,
            "desired": "unloaded",
            "intent_seq": 1,
            "model": {"managed": true, "desired": "unloaded", "state": "ready", "lock_sha256": lock},
        })
    );
    let view = registry(&h).worker_view(A).expect("registered");
    let aging = registry(&h).timing().aging();
    assert_ne!(view.state_at(now_ms(), aging), WorkerState::Ready);
    assert_ne!(state(&h).state, WorkerState::Ready);
    let r = h
        .chat(&chat_body(false))
        .header("X-Cere-Deadline-Ms", "500")
        .send()
        .await
        .unwrap();
    assert_ne!(
        r.status(),
        200,
        "no grant to worker-a after the unload answered"
    );
    assert_eq!(
        work(&stub),
        served,
        "no SubmitWork after the unload answered"
    );
    let records = residency_records(&h);
    assert_eq!(records.len(), 1, "{records:?}");
    assert_eq!(
        (
            &records[0]["worker"],
            &records[0]["desired"],
            &records[0]["intent_seq"]
        ),
        (&json!(A), &json!("unloaded"), &json!(1))
    );

    let (status, body) = operate(&h, A, "load").await;
    assert_eq!(status, 202, "{body}");
    assert_eq!(
        (&body["desired"], &body["intent_seq"]),
        (&json!("resident"), &json!(2))
    );
    readyz_until(&h, "worker-a ready again", |b| {
        b["workers"][A]["state"] == "ready"
    })
    .await;
    let r = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 200, "served by worker-a after load");
    assert_eq!(work(&stub), served + 1);
}

// ---------------------------------------------------------------------------------------------------------
// The residency driver (design-m36-coord design note §3.1, §3.2)

/// Six probe rounds of the 100 ms interval: long enough for any call the driver would make to show.
const ROUNDS: Duration = Duration::from_millis(600);
/// The scope of every residency call (m36 design note Q8).
const CONTROL_SCOPE: &str = "_cere_control";

/// The op id of the residency call of `intent_seq` at `epoch` in this coordinator's term (Q8).
fn op_id(h: &Harness, intent_seq: u64, epoch: u64, verb: &str) -> String {
    format!("residency-{}-{intent_seq}-{epoch}-{verb}", h.app.epoch())
}

fn load(h: &Harness, intent_seq: u64, epoch: u64) -> (&'static str, String) {
    (wire::METHOD_LOAD, op_id(h, intent_seq, epoch, "load"))
}

fn unload(h: &Harness, intent_seq: u64, epoch: u64) -> (&'static str, String) {
    (wire::METHOD_UNLOAD, op_id(h, intent_seq, epoch, "unload"))
}

/// Every lifecycle call the stub received, as `(method, op id)`.
fn calls(stub: &StubAgent) -> Vec<(&'static str, String)> {
    stub.lifecycle_calls()
        .into_iter()
        .map(|(method, request)| (method, request.authority.unwrap().operation_id))
        .collect()
}

/// [`calls`] after [`ROUNDS`].
async fn calls_after_rounds(stub: &StubAgent) -> Vec<(&'static str, String)> {
    tokio::time::sleep(ROUNDS).await;
    calls(stub)
}

/// Every lifecycle call as received, once there are `n` and [`ROUNDS`] more passed.
async fn requests_after(stub: &StubAgent, n: usize) -> Vec<stub_agent::LifecycleCall> {
    for _ in 0..500 {
        if stub.lifecycle_calls().len() >= n {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    tokio::time::sleep(ROUNDS).await;
    stub.lifecycle_calls()
}

/// Waits until `/readyz` shows worker-a's model described in `state`.
async fn described(h: &Harness, state: ModelState) -> Value {
    let state = serde_json::to_value(state).unwrap();
    let (_, body) = readyz_until(h, &format!("model {state}"), |b| {
        b["workers"][A]["model"]["state"] == state
    })
    .await;
    body
}

/// The `field`s of worker-a's `/readyz` problems.
fn problem_fields(body: &Value) -> Vec<&str> {
    body["workers"][A]["problems"]
        .as_array()
        .map(|problems| {
            problems
                .iter()
                .filter_map(|p| p["field"].as_str())
                .collect()
        })
        .unwrap_or_default()
}

/// The readiness smokes worker-a ran so far.
fn smokes(stub: &StubAgent) -> usize {
    stub.submits()
        .iter()
        .filter(|s| s.authority.as_ref().unwrap().scope_id == REMOTE_SMOKE_SCOPE)
        .count()
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

async fn intent(h: &Harness, action: &str) {
    let (status, body) = operate(h, A, action).await;
    assert_eq!(status, 202, "{action}: {body}");
}

/// Q8, `desired = none`: whatever the agent describes is adopted; the driver never calls.
#[tokio::test(flavor = "multi_thread")]
async fn without_an_intent_the_driver_never_calls() {
    let m = managed_start(ModelState::Unloaded).await;
    for state in [
        ModelState::Unloaded,
        ModelState::Loading,
        ModelState::Ready,
        ModelState::Draining,
        ModelState::Unloading,
        ModelState::Failed,
    ] {
        m.stub.set_model(state, &m.lock);
        described(&m.h, state).await;
        assert_eq!(
            calls_after_rounds(&m.stub).await,
            vec![],
            "none + {state:?}"
        );
    }
}

/// Q8, `desired = resident`: `LoadModel` only for `unloaded` or `draining`, once per intent; nothing for
/// `loading`, `ready`, `unloading`, nor for `failed` under the intent that saw it fail (no implicit retry,
/// MODEL_LIFECYCLE:104); the next `load` is a new intent with its own op id.
#[tokio::test(flavor = "multi_thread")]
async fn a_resident_intent_loads_only_an_unloaded_or_draining_model() {
    let m = managed_start(ModelState::Loading).await;
    described(&m.h, ModelState::Loading).await;
    intent(&m.h, "load").await;
    for state in [
        ModelState::Loading,
        ModelState::Ready,
        ModelState::Unloading,
        ModelState::Failed,
    ] {
        m.stub.set_model(state, &m.lock);
        described(&m.h, state).await;
        assert_eq!(
            calls_after_rounds(&m.stub).await,
            vec![],
            "resident + {state:?}"
        );
    }
    m.stub.set_model(ModelState::Unloaded, &m.lock);
    described(&m.h, ModelState::Unloaded).await;
    let first = load(&m.h, 1, 1);
    assert_eq!(calls_after_rounds(&m.stub).await, vec![first.clone()]);
    m.stub.set_model(ModelState::Draining, &m.lock);
    described(&m.h, ModelState::Draining).await;
    assert_eq!(
        calls_after_rounds(&m.stub).await,
        vec![first.clone()],
        "one Load per intent"
    );
    intent(&m.h, "load").await;
    assert_eq!(
        calls_after_rounds(&m.stub).await,
        vec![first, load(&m.h, 2, 1)],
        "resident + draining under a new intent"
    );
}

/// Q8, `desired = unloaded`: `UnloadModel` for `ready`, `loading`, `draining` and `failed`; nothing for
/// `unloaded` or `unloading`.
#[tokio::test(flavor = "multi_thread")]
async fn an_unloaded_intent_unloads_every_live_or_failed_model() {
    let m = managed_start(ModelState::Ready).await;
    let mut expected = Vec::new();
    for (seq, (state, call)) in [
        (ModelState::Ready, true),
        (ModelState::Loading, true),
        (ModelState::Draining, true),
        (ModelState::Failed, true),
        (ModelState::Unloaded, false),
        (ModelState::Unloading, false),
    ]
    .into_iter()
    .enumerate()
    {
        m.stub.set_model(state, &m.lock);
        described(&m.h, state).await;
        intent(&m.h, "unload").await;
        if call {
            expected.push(unload(&m.h, seq as u64 + 1, 1));
        }
        assert_eq!(
            calls_after_rounds(&m.stub).await,
            expected,
            "unloaded + {state:?}"
        );
    }
}

/// Q8: no call for a model under another lock (`/readyz` problem `lock_mismatch`, the worker stays Joining) nor
/// for an inventory described in an older worker epoch; the intent's call goes out once the coordinator's lock
/// is described, and after an agent restart once the new incarnation is described.
#[tokio::test(flavor = "multi_thread")]
async fn a_foreign_lock_or_an_older_epoch_inventory_gets_no_call() {
    let m = managed_start(ModelState::Unloaded).await;
    let other = "0".repeat(64);
    m.stub.set_model(ModelState::Unloaded, &other);
    readyz_until(&m.h, "the foreign lock", |b| {
        b["workers"][A]["model"]["lock_sha256"] == other.as_str()
    })
    .await;
    intent(&m.h, "load").await;
    let (_, body) = readyz_until(&m.h, "lock_mismatch", |b| {
        problem_fields(b) == ["lock_mismatch"]
    })
    .await;
    assert_eq!(body["workers"][A]["state"], "joining", "{body}");
    assert_eq!(calls_after_rounds(&m.stub).await, vec![], "lock mismatch");
    m.stub.set_model(ModelState::Unloaded, &m.lock);
    let first = load(&m.h, 1, 1);
    requests_after(&m.stub, 1).await;
    assert_eq!(calls(&m.stub), vec![first.clone()]);

    // An agent restart whose inventory cannot be described yet: the model view is still epoch 1's.
    m.stub.report_sha256(Some([9; 32]));
    m.stub.restart(2);
    readyz_until(&m.h, "epoch 2 undescribed", |b| {
        b["workers"][A]["epoch"] == 2 && problem_fields(b).contains(&"inventory_sha256")
    })
    .await;
    intent(&m.h, "load").await;
    assert_eq!(
        calls_after_rounds(&m.stub).await,
        vec![first.clone()],
        "stale-epoch inventory"
    );
    m.stub.report_sha256(None);
    requests_after(&m.stub, 2).await;
    assert_eq!(calls(&m.stub), vec![first, load(&m.h, 2, 2)]);
}

/// M5 D8 (C4): with two deployments (two locks), worker A of the second, remote-only entry is driven under its
/// own deployment's lock: the managed `LoadModel` (and its Authority payload) carries B's lock, never the first
/// entry's; the worker is Ready once its model is ready under B's lock and then serves B's requests with B's
/// identity, while a model ready under the first entry's lock is a foreign lock for it.
#[tokio::test(flavor = "multi_thread")]
async fn m5_a_managed_load_carries_the_workers_deployment_lock() {
    let m = managed_start_in(ModelState::Unloaded, 1).await;
    assert_eq!(m.lock, lock_sha256_of("gemma-4-e4b-it"));
    assert_eq!(lock(&m.h), lock_sha256_of("qwen38-flash-next"));
    assert_ne!(m.lock, lock(&m.h));
    described(&m.h, ModelState::Unloaded).await;
    intent(&m.h, "load").await;
    let requests = requests_after(&m.stub, 1).await;
    let (method, request) = &requests[0];
    let lock_b = unhex(&m.lock);
    assert_eq!(*method, wire::METHOD_LOAD);
    assert_eq!(request.deployment_lock_sha256, lock_b);
    assert_eq!(
        request.authority.as_ref().unwrap().payload_sha256,
        wire::lifecycle_sha256(wire::METHOD_LOAD, CONTROL_SCOPE, &lock_b, "", &[]).to_vec()
    );

    m.stub.set_model(ModelState::Ready, &m.lock);
    readyz_until(&m.h, "worker-a ready under B's lock", |b| {
        b["workers"][A]["state"] == "ready"
    })
    .await;
    let mut body = chat_body(false);
    body["model"] = json!(MODEL_B);
    let r = m.h.chat(&body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let reply: Value = r.json().await.unwrap();
    assert_eq!(reply["cere"]["worker_id"], A, "{reply}");
    assert_eq!(reply["cere"]["lock_sha256"], m.lock.as_str(), "{reply}");

    m.stub.set_model(ModelState::Ready, &lock(&m.h));
    readyz_until(&m.h, "lock_mismatch under A's lock", |b| {
        problem_fields(b) == ["lock_mismatch"]
    })
    .await;
}

/// Q8: a Load whose receipt was dropped after the agent journaled it, and one refused `model_busy`, are retried
/// on later rounds with the same op id and the identical Authority (the journal answers the stored receipt);
/// a receipted call is polled, never re-sent, across rounds and after its terminal (one call per intent, epoch
/// and term). The Authority names `_cere_control`, the op id, the lifecycle digest of the coordinator's lock,
/// the term, the epoch and the heartbeat generation. After an agent restart (a new epoch) the same intent
/// reloads under a new op id and the worker becomes Ready.
#[tokio::test(flavor = "multi_thread")]
async fn a_residency_call_is_retried_unchanged_until_receipted_and_never_resent() {
    let m = managed_start(ModelState::Unloaded).await;
    described(&m.h, ModelState::Unloaded).await;
    m.stub
        .script_lifecycle(wire::METHOD_LOAD, Lifecycle::DropReceipt);
    intent(&m.h, "load").await;
    let requests = requests_after(&m.stub, 2).await;
    assert_eq!(requests.len(), 2, "the retry got the stored receipt");
    assert_eq!(requests[0], requests[1], "the retry is the same request");
    let (method, request) = &requests[0];
    let lock = unhex(&m.lock);
    assert_eq!(*method, wire::METHOD_LOAD);
    assert_eq!(request.deployment_lock_sha256, lock);
    assert_eq!(request.instance_id, "");
    assert!(request.device_ids.is_empty());
    let authority = request.authority.clone().unwrap();
    let target = authority.target.clone().unwrap();
    assert_eq!((target.worker_id.as_str(), target.worker_epoch), (A, 1));
    assert_eq!(authority.coordinator_term, m.h.app.epoch());
    assert_eq!(authority.scope_id, CONTROL_SCOPE);
    assert_eq!(authority.operation_id, op_id(&m.h, 1, 1, "load"));
    assert_eq!(
        authority.payload_sha256,
        wire::lifecycle_sha256(wire::METHOD_LOAD, CONTROL_SCOPE, &lock, "", &[]).to_vec()
    );
    assert_eq!(authority.model_generation, Some(1));
    assert_eq!(authority.remaining_budget_ms, wire::WORK_CAP_MS);

    // The terminal: re-described (still unloaded), and no second Load for this intent.
    m.stub.end_op(CONTROL_SCOPE, &authority.operation_id, "");
    assert_eq!(calls_after_rounds(&m.stub).await.len(), 2);

    m.stub.script_lifecycle(
        wire::METHOD_LOAD,
        Lifecycle::Refuse(Code::FailedPrecondition, wire::MODEL_BUSY),
    );
    intent(&m.h, "load").await;
    let requests = requests_after(&m.stub, 4).await;
    assert_eq!(requests.len(), 4, "one retry after model_busy");
    assert_eq!(requests[2], requests[3], "the retry is the same request");
    assert_eq!(calls(&m.stub)[2..], [load(&m.h, 2, 1), load(&m.h, 2, 1)]);

    // A new epoch: the same intent reloads under a new op id.
    m.stub.script_lifecycle(
        wire::METHOD_LOAD,
        Lifecycle::Walk {
            states: &[ModelState::Loading, ModelState::Ready],
            reason: "",
        },
    );
    m.stub.restart(2);
    readyz_until(&m.h, "worker-a ready at epoch 2", |b| {
        b["workers"][A]["state"] == "ready" && b["workers"][A]["epoch"] == 2
    })
    .await;
    assert_eq!(calls_after_rounds(&m.stub).await[4..], [load(&m.h, 2, 2)]);
}

/// Q8, MODEL_LIFECYCLE:104: a Load that ends `model_load_failed` is shown as `/readyz` problem `residency` and is
/// not retried; the operator's next `load` is a new intent and so the explicit reset: a new Load, then Ready.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_load_waits_for_the_next_intent() {
    let m = managed_start(ModelState::Unloaded).await;
    described(&m.h, ModelState::Unloaded).await;
    m.stub.script_lifecycle(
        wire::METHOD_LOAD,
        Lifecycle::Walk {
            states: &[ModelState::Loading, ModelState::Failed],
            reason: wire::MODEL_LOAD_FAILED,
        },
    );
    intent(&m.h, "load").await;
    let (_, body) = readyz_until(&m.h, "the failed load", |b| {
        b["workers"][A]["model"]["state"] == "failed" && problem_fields(b) == ["residency"]
    })
    .await;
    let detail = body["workers"][A]["problems"][0]["detail"]
        .as_str()
        .unwrap();
    assert!(detail.contains(wire::MODEL_LOAD_FAILED), "{detail}");
    let first = load(&m.h, 1, 1);
    assert_eq!(calls_after_rounds(&m.stub).await, vec![first.clone()]);
    let (_, body) = readyz(&m.h).await;
    assert_eq!(body["workers"][A]["state"], "joining", "{body}");
    assert_eq!(problem_fields(&body), ["residency"], "{body}");

    m.stub.script_lifecycle(
        wire::METHOD_LOAD,
        Lifecycle::Walk {
            states: &[ModelState::Loading, ModelState::Ready],
            reason: "",
        },
    );
    intent(&m.h, "load").await;
    let (_, body) = readyz_until(&m.h, "worker-a ready", |b| {
        b["workers"][A]["state"] == "ready"
    })
    .await;
    assert_eq!(problem_fields(&body), Vec::<&str>::new(), "{body}");
    assert_eq!(calls(&m.stub), vec![first, load(&m.h, 2, 1)]);
}

/// Q10 (review M36-03): a managed worker is Ready only while its model is `ready` under the coordinator's lock
/// and `desired != unloaded`; otherwise it stays Joining without pin or smoke problems (a foreign lock shows only
/// `lock_mismatch`) and is never smoked.
#[tokio::test(flavor = "multi_thread")]
async fn a_managed_worker_is_ready_only_while_its_model_is_ready_under_the_lock_and_wanted() {
    let m = managed_start(ModelState::Loading).await;
    described(&m.h, ModelState::Loading).await;
    tokio::time::sleep(ROUNDS).await;
    let (_, body) = readyz(&m.h).await;
    assert_eq!(body["workers"][A]["state"], "joining", "{body}");
    assert_eq!(problem_fields(&body), Vec::<&str>::new(), "loading: {body}");
    assert_eq!(smokes(&m.stub), 0, "no smoke while loading");

    m.stub.set_model(ModelState::Ready, &m.lock);
    readyz_until(&m.h, "worker-a ready", |b| {
        b["workers"][A]["state"] == "ready"
    })
    .await;
    let smoked = smokes(&m.stub);
    assert!(smoked >= 1);

    m.stub.set_model(ModelState::Ready, &"0".repeat(64));
    readyz_until(&m.h, "lock_mismatch", |b| {
        problem_fields(b) == ["lock_mismatch"]
    })
    .await;
    tokio::time::sleep(ROUNDS).await;
    let (_, body) = readyz(&m.h).await;
    assert_eq!(body["workers"][A]["state"], "joining", "{body}");
    assert_eq!(problem_fields(&body), ["lock_mismatch"], "{body}");
    assert_eq!(smokes(&m.stub), smoked, "no smoke under a foreign lock");

    m.stub.set_model(ModelState::Ready, &m.lock);
    readyz_until(&m.h, "worker-a ready again", |b| {
        b["workers"][A]["state"] == "ready"
    })
    .await;
    let smoked = smokes(&m.stub);

    intent(&m.h, "unload").await;
    tokio::time::sleep(ROUNDS).await;
    let (_, body) = readyz(&m.h).await;
    assert_eq!(body["workers"][A]["state"], "joining", "{body}");
    assert_eq!(body["workers"][A]["model"]["state"], "ready", "{body}");
    assert_eq!(
        problem_fields(&body),
        Vec::<&str>::new(),
        "unwanted: {body}"
    );
    assert_eq!(smokes(&m.stub), smoked, "no smoke for an unwanted model");
    assert_eq!(calls(&m.stub), vec![unload(&m.h, 1, 1)]);
}

/// Q10 applies to managed workers only (review K1): a journaled `desired = unloaded` (forward-only) does not fence
/// the same worker id once it describes an attached model. After the operator unloads a managed worker, the agent
/// switches to attached mode: the worker becomes Ready, a session is admitted to it, and `/readyz` keeps the
/// journaled intent next to the attached description.
#[tokio::test(flavor = "multi_thread")]
async fn a_journaled_unload_does_not_fence_a_worker_described_attached_later() {
    let m = managed_start(ModelState::Ready).await;
    readyz_until(&m.h, "managed worker-a ready", |b| {
        b["workers"][A]["state"] == "ready"
    })
    .await;
    intent(&m.h, "unload").await;
    requests_after(&m.stub, 1).await;
    assert_eq!(calls(&m.stub), vec![unload(&m.h, 1, 1)]);

    m.stub
        .set_inventory(stub_agent::inventory(A, m.stub.generation()));
    let (_, body) = readyz_until(&m.h, "attached worker-a ready", |b| {
        b["workers"][A]["state"] == "ready" && b["workers"][A]["model"]["state"] == "attached"
    })
    .await;
    assert_eq!(
        body["workers"][A]["model"],
        json!({"managed": false, "desired": "unloaded", "state": "attached", "lock_sha256": null}),
        "{body}"
    );
    assert_eq!(state(&m.h).model.desired, Desired::Unloaded);
    let view = registry(&m.h).worker_view(A).expect("registered");
    let aging = registry(&m.h).timing().aging();
    assert_eq!(view.state_at(now_ms(), aging), WorkerState::Ready);
    let r = m.h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 200, "served by the attached worker-a");
    let served = m
        .stub
        .submits()
        .iter()
        .filter(|s| s.authority.as_ref().unwrap().scope_id != REMOTE_SMOKE_SCOPE)
        .count();
    assert_eq!(served, 1, "the session was admitted to worker-a");
}

/// Review K2: on a pass of a non-Joining worker the residency-owned problems are refreshed. A refused load leaves
/// a `residency` problem; the agent then loads on its own and the worker becomes Ready still showing it (the call of
/// intent 1 ended refused). The operator's next `load` supersedes that call (nothing to call for a `ready` model),
/// and the problem clears while the worker stays Ready in the same round.
#[tokio::test(flavor = "multi_thread")]
async fn a_residency_problem_on_a_ready_worker_clears_once_superseded() {
    let m = managed_start(ModelState::Unloaded).await;
    described(&m.h, ModelState::Unloaded).await;
    m.stub.script_lifecycle(
        wire::METHOD_LOAD,
        Lifecycle::Refuse(Code::PermissionDenied, "not_authorized"),
    );
    intent(&m.h, "load").await;
    readyz_until(&m.h, "the refused load", |b| {
        problem_fields(b) == ["residency"]
    })
    .await;

    m.stub.set_model(ModelState::Ready, &m.lock);
    let (_, body) = readyz_until(&m.h, "worker-a ready", |b| {
        b["workers"][A]["state"] == "ready"
    })
    .await;
    assert_eq!(problem_fields(&body), ["residency"], "{body}");
    let round = state(&m.h).round;
    tokio::time::sleep(ROUNDS).await;
    let (_, body) = readyz(&m.h).await;
    assert_eq!(problem_fields(&body), ["residency"], "{body}");

    intent(&m.h, "load").await;
    let (_, body) = readyz_until(&m.h, "the residency problem cleared", |b| {
        problem_fields(b).is_empty()
    })
    .await;
    assert_eq!(body["workers"][A]["state"], "ready", "{body}");
    assert_eq!(
        state(&m.h).round,
        round,
        "cleared without a new Joining round"
    );
    assert_eq!(calls(&m.stub), vec![load(&m.h, 1, 1)]);
}
