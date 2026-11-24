//! P8 operator drain API (m35 design note Q13; build-loop-m35 design note "Operator API", D3):
//! `POST /v1/cere/workers/{worker_id}/drain|undrain` behind the optional `[auth] operator_token_env` bearer token,
//! for local workers and remote (`[federation]`) ids, journaled and restored on restart.

mod support;

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cere_mock_sglang::{MockConfig, MockSglang};
use cere_proto::worker::v2 as pb;
use cere_runtime::fed::wire;
use pb::worker_registry_service_client::WorkerRegistryServiceClient;
use serde_json::{json, Value};
use support::{
    chat_body, config_text, journal, readyz, readyz_until, test_tuning, Cered, ConfigFile,
    Federation, Harness, FED_CLUSTER, OPERATOR_TOKEN, OPERATOR_TWEAK, TOKEN_A,
};

const LOCAL: &str = "gpu1";
const REMOTE: &str = "worker-a";
const PRINCIPAL: &str = "cere-worker-a";

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

async fn operate(h: &Harness, worker: &str, draining: bool) -> (u16, Value) {
    let action = if draining { "drain" } else { "undrain" };
    call(h, worker, action, Some(&format!("Bearer {OPERATOR_TOKEN}"))).await
}

fn drain_records(h: &Harness, worker: &str) -> Vec<bool> {
    journal(h.state_dir.path())
        .iter()
        .filter(|r| r["type"] == "worker_drain" && r["worker"] == worker)
        .map(|r| r["draining"].as_bool().unwrap())
        .collect()
}

fn error_code(body: &Value) -> &str {
    body["error"]["code"].as_str().unwrap_or_default()
}

#[tokio::test(flavor = "multi_thread")]
async fn without_an_operator_token_both_routes_are_404() {
    let h = Harness::start(MockConfig::default(), &[]).await;
    for action in ["drain", "undrain"] {
        for auth in [
            None,
            Some("Bearer secret-a"),
            Some("Bearer secret-operator"),
        ] {
            let (status, body) = call(&h, LOCAL, action, auth).await;
            assert_eq!(status, 404, "{action} {auth:?}: {body}");
            assert_eq!(error_code(&body), "not_found", "{body}");
        }
    }
    assert!(drain_records(&h, LOCAL).is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn operator_routes_check_the_bearer_and_the_worker_id() {
    let h = Harness::start(MockConfig::default(), &[OPERATOR_TWEAK]).await;
    for action in ["drain", "undrain"] {
        // A client scope token is authenticated but not authorized, also for an unknown worker.
        for worker in [LOCAL, "gpu9"] {
            let (status, body) = call(&h, worker, action, Some(&format!("Bearer {TOKEN_A}"))).await;
            assert_eq!(status, 403, "{action} {worker}: {body}");
            assert_eq!(error_code(&body), "forbidden", "{body}");
        }
        for auth in [
            None,
            Some("Bearer wrong"),
            Some("Bearer "),
            Some("Basic secret-operator"),
            Some("bearer secret-operator"),
        ] {
            let (status, body) = call(&h, LOCAL, action, auth).await;
            assert_eq!(status, 401, "{action} {auth:?}: {body}");
            assert_eq!(error_code(&body), "unauthorized", "{body}");
        }
        let auth = format!("Bearer {OPERATOR_TOKEN}");
        let (status, body) = call(&h, "gpu9", action, Some(&auth)).await;
        assert_eq!(status, 404, "{body}");
        assert_eq!(error_code(&body), "worker_not_found", "{body}");
    }
    assert!(drain_records(&h, LOCAL).is_empty());
    assert!(drain_records(&h, "gpu9").is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn local_drain_stops_new_grants_lets_in_flight_finish_and_survives_restart() {
    let mut mocks = Vec::new();
    for _ in 0..2 {
        let config = MockConfig {
            token_latency: Duration::from_millis(60),
            ..MockConfig::default()
        };
        mocks.push(MockSglang::start(config).await.unwrap());
    }
    // Once one worker drains, every new chat needs the other one Ready. A worker reads suspect once it goes
    // 3 probe intervals without a healthy probe; with `fast_tuning` that is 300 ms, which a loaded host misses
    // (`backend not ready: gpu0: draining; gpu1: suspect`). `test_tuning` gives 15 s, and nothing here waits
    // on a probe (undrain only clears the intent).
    let h = Harness::start_replicas(mocks, &[OPERATOR_TWEAK]).await;
    let smokes: Vec<usize> = (0..2)
        .map(|i| h.replica(i).dispatched_rids().len())
        .collect();

    // One request in flight; drain the worker that took it.
    let in_flight = tokio::spawn(h.chat(&chat_body(false)).send());
    let mut busy = None;
    for _ in 0..200 {
        busy = (0..2).find(|&i| h.replica(i).dispatched_rids().len() > smokes[i]);
        if busy.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let busy = busy.expect("the request reached a backend");
    let device = format!("gpu{busy}");
    let (status, body) = operate(&h, &device, true).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body, json!({"worker_id": device, "draining": true}));
    // Idempotent: the same reply again.
    let (status, again) = operate(&h, &device, true).await;
    assert_eq!((status, &again), (200, &body));

    let (_, ready) = readyz(&h).await;
    assert_eq!(ready["workers"][&device]["state"], "draining", "{ready}");
    assert_eq!(ready["workers"][&device]["ready"], false, "{ready}");
    let response = in_flight.await.unwrap().unwrap();
    assert_eq!(response.status().as_u16(), 200, "in-flight work finishes");
    for _ in 0..4 {
        let r = h.chat(&chat_body(false)).send().await.unwrap();
        let status = r.status().as_u16();
        let text = r.text().await.unwrap();
        assert_eq!(status, 200, "{text}");
    }
    assert_eq!(
        h.replica(busy).dispatched_rids().len(),
        smokes[busy] + 1,
        "no new grant on the drained worker"
    );
    assert_eq!(drain_records(&h, &device).first(), Some(&true));

    // Restored on restart.
    let h = h.restart().await;
    let (_, ready) = readyz(&h).await;
    assert_eq!(ready["workers"][&device]["state"], "draining", "{ready}");

    // Undrain: ready again once the checks pass.
    let (status, body) = operate(&h, &device, false).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body, json!({"worker_id": device, "draining": false}));
    let (status, again) = operate(&h, &device, false).await;
    assert_eq!((status, &again), (200, &body));
    readyz_until(&h, "undrained worker ready", |b| {
        b["workers"][&device]["state"] == "ready" && b["workers"][&device]["ready"] == true
    })
    .await;
    let records = drain_records(&h, &device);
    assert_eq!(records.last(), Some(&false), "{records:?}");
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn identity(epoch: u64) -> Option<pb::WorkerIdentity> {
    Some(pb::WorkerIdentity {
        cluster_id: FED_CLUSTER.into(),
        worker_id: REMOTE.into(),
        worker_epoch: epoch,
    })
}

fn heartbeat_req(term: u64, sequence: u64) -> pb::HeartbeatRequest {
    pb::HeartbeatRequest {
        worker: identity(1),
        sequence,
        observed_coordinator_term: term,
        inventory_sha256: vec![7; 32],
        model_generation: 1,
        admission_state: "open".into(),
        owner_only: false,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn remote_drain_is_journaled_and_sent_as_desired_state() {
    let fed = Federation::new();
    let text = fed.toml(
        "heartbeat_interval_ms = 100\nadmission_lease_ms = 200\nsuspect_after_ms = 400\n\
         lost_after_ms = 800\n",
        &[(REMOTE, PRINCIPAL)],
    );
    let h = Harness::start_federated_with(
        MockConfig::default(),
        &[OPERATOR_TWEAK],
        &text,
        test_tuning(),
    )
    .await;
    // Before the agent's first Register.
    let (status, body) = operate(&h, REMOTE, true).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body, json!({"worker_id": REMOTE, "draining": true}));
    let (status, again) = operate(&h, REMOTE, true).await;
    assert_eq!((status, &again), (200, &body));
    assert_eq!(drain_records(&h, REMOTE).first(), Some(&true));
    let registry = h.app.remote_workers().unwrap();
    assert!(registry.get(REMOTE, now_ms()).unwrap().draining);

    let mut agent = WorkerRegistryServiceClient::new(fed.channel(fed.agent_tls(PRINCIPAL)));
    let reply = agent
        .register(pb::RegisterRequest {
            proposed: identity(1),
            boot_id: "0123456789abcdef0123456789abcdef".into(),
            protocol_revision: cere_proto::PROTOCOL_REVISION,
            inventory_sha256: vec![7; 32],
        })
        .await
        .unwrap()
        .into_inner();
    assert!(reply.accepted, "{reply:?}");
    let term = h.app.epoch();
    let reply = agent
        .heartbeat(heartbeat_req(term, 1))
        .await
        .unwrap()
        .into_inner();
    assert!(reply.accepted, "{reply:?}");
    assert_eq!(reply.desired_state, wire::DESIRED_DRAINING);

    let (status, body) = operate(&h, REMOTE, false).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body, json!({"worker_id": REMOTE, "draining": false}));
    let reply = agent
        .heartbeat(heartbeat_req(term, 2))
        .await
        .unwrap()
        .into_inner();
    assert!(reply.accepted, "{reply:?}");
    assert_eq!(reply.desired_state, "");
    assert_eq!(drain_records(&h, REMOTE).last(), Some(&false));
}

/// `cered` resolves the operator token at startup: a configured but missing or empty variable refuses to start
/// (exit 2, as for scope tokens); with it set, a drain over HTTP survives a process restart on the same state dir.
#[tokio::test(flavor = "multi_thread")]
async fn cered_resolves_the_operator_token_and_keeps_drain_across_restart() {
    let mock = MockSglang::start(MockConfig::default()).await.unwrap();
    let state_dir = support::StateDir::new();
    let text = config_text(
        &mock.base_url(),
        state_dir.path(),
        support::free_port(),
        &[OPERATOR_TWEAK],
    );
    let config = ConfigFile::new(&text);
    for value in [None, Some("")] {
        let (code, stderr) = Cered::spawn(&config.path, &[("TOK_OP", value)])
            .await
            .err()
            .expect("refused without the operator token");
        assert_eq!(code, Some(2), "{stderr}");
        assert!(stderr.contains("TOK_OP"), "{stderr}");
    }

    let env = [("TOK_OP", Some(OPERATOR_TOKEN))];
    let mut cered = Cered::spawn(&config.path, &env).await.unwrap();
    cered.wait_ready().await;
    let http = support::http();
    let drain = http
        .post(cered.url(&format!("/v1/cere/workers/{LOCAL}/drain")))
        .bearer_auth(OPERATOR_TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(drain.status().as_u16(), 200);
    let body: Value = drain.json().await.unwrap();
    assert_eq!(body, json!({"worker_id": LOCAL, "draining": true}));
    cered.signal("TERM");
    let (code, stderr) = cered.exit(Duration::from_secs(20)).await;
    assert_eq!(code, Some(0), "{stderr}");

    let mut cered = Cered::spawn(&config.path, &env).await.unwrap();
    let mut state = Value::Null;
    for _ in 0..250 {
        let body: Value = http
            .get(cered.url("/readyz"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        state = body["workers"][LOCAL]["state"].clone();
        if state == "draining" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(state, "draining");
    cered.signal("TERM");
    let (code, stderr) = cered.exit(Duration::from_secs(20)).await;
    assert_eq!(code, Some(0), "{stderr}");
}
