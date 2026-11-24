//! M5 multi-model `cered` over in-process mocks (m5 design note C3, C4): deployment A ([`MODEL`]) on `gpu0` and
//! `gpu1`, deployment B ([`MODEL_B`]) on `gpu2`. The request's `model` selects the deployment (D5), admission and
//! least-loaded placement stay inside it (D6), each worker's readiness checks its own deployment's pins (D9),
//! sessions bind to the model of their first commit (D7) and migration stays inside one deployment (D8).
mod support;

use std::time::Duration;

use cere_mock_sglang::{Fault, MockConfig, MockHandle, MockSglang, ServerInfo};
use cere_runtime::app::Tuning;
use reqwest::StatusCode;
use serde_json::{json, Value};
use support::*;

const LOCK_A: &str = "qwen38-flash-next";
const LOCK_B: &str = "gemma-4-e4b-it";
const PROFILE_B: &str = "sglang-gemma-4-e4b-it-v1";

/// A's two mocks (`token_latency` per token) and B's one, in device order.
async fn mocks_ab(token_latency: Duration) -> Vec<MockHandle> {
    let mut out = Vec::new();
    for info in [
        ServerInfo::default(),
        ServerInfo::default(),
        server_info_b(),
    ] {
        out.push(
            MockSglang::start(MockConfig {
                server_info: info,
                token_latency,
                ..MockConfig::default()
            })
            .await
            .unwrap(),
        );
    }
    out
}

fn config_ab(mocks: &[MockHandle], state_dir: &StateDir, tweaks: &[(&str, &str)]) -> String {
    let urls: Vec<String> = mocks.iter().map(MockHandle::base_url).collect();
    multimodel_config_text(&urls[..2], &urls[2..], state_dir.path(), 18391, tweaks)
}

async fn start_ab(tweaks: &[(&str, &str)], tuning: Tuning) -> Harness {
    start_ab_mocks(mocks_ab(Duration::ZERO).await, tweaks, tuning).await
}

async fn start_ab_mocks(
    mocks: Vec<MockHandle>,
    tweaks: &[(&str, &str)],
    tuning: Tuning,
) -> Harness {
    let state_dir = StateDir::new();
    let config = config_ab(&mocks, &state_dir, tweaks);
    Harness::start_config(mocks, state_dir, config, tuning).await
}

fn body(model: &str, messages: &[(&str, &str)]) -> Value {
    let messages: Vec<Value> = messages
        .iter()
        .map(|(role, content)| json!({"role": role, "content": content}))
        .collect();
    json!({"model": model, "messages": messages, "max_completion_tokens": 4})
}

async fn answer(request: reqwest::RequestBuilder) -> (u16, Value) {
    let r = request.send().await.unwrap();
    let status = r.status().as_u16();
    let text = r.text().await.unwrap();
    let body = serde_json::from_str(&text)
        .unwrap_or_else(|_| panic!("{status} answered a non-JSON body {text:?}"));
    (status, body)
}

async fn chat(h: &Harness, model: &str) -> (u16, Value) {
    answer(h.chat(&body(model, &[("user", "hello")]))).await
}

async fn create(h: &Harness) -> String {
    let (status, body) = answer(h.post("/v1/cere/sessions", TOKEN_A)).await;
    assert_eq!(status, 201, "{body}");
    body["session_id"].as_str().unwrap().to_owned()
}

/// A session turn at `revision` on `model`.
async fn turn(
    h: &Harness,
    session: &str,
    revision: u64,
    model: &str,
    messages: &[(&str, &str)],
) -> (u16, Value) {
    answer(
        h.chat(&body(model, messages))
            .header("X-Cere-Session-Id", session)
            .header("X-Cere-Expected-Revision", revision.to_string()),
    )
    .await
}

async fn view(h: &Harness, session: &str) -> Value {
    let (status, body) = answer(h.get(&format!("/v1/cere/sessions/{session}"), TOKEN_A)).await;
    assert_eq!(status, 200, "{body}");
    body
}

async fn owner(h: &Harness, session: &str) -> Value {
    let (status, body) = answer(h.get(
        &format!("/v1/cere/sessions/{session}/owner"),
        OPERATOR_TOKEN,
    ))
    .await;
    assert_eq!(status, 200, "{body}");
    body
}

async fn migrate(h: &Harness, session: &str, to: &str, expected: u64) -> (u16, Value) {
    answer(
        h.post(
            &format!("/v1/cere/sessions/{session}/migrate"),
            OPERATOR_TOKEN,
        )
        .json(&json!({"to_worker": to, "expected_owner_generation": expected, "force": false})),
    )
    .await
}

#[track_caller]
fn assert_error(got: &(u16, Value), status: u16, code: &str, param: Option<&str>) {
    let (got_status, body) = got;
    assert_eq!(*got_status, status, "{body}");
    assert_eq!(body["error"]["code"], code, "{body}");
    assert_eq!(body["error"]["param"].as_str(), param, "{body}");
}

/// The `cere` identity of a reply served by deployment `lock` (profile `profile`) on one of `workers`.
#[track_caller]
fn assert_served(
    h: &Harness,
    reply: &Value,
    model: &str,
    lock: &str,
    profile: &str,
    workers: &[&str],
) {
    let cere = &reply["cere"];
    assert_eq!(reply["model"], model, "{reply}");
    assert_eq!(cere["lock_sha256"], lock_sha256_of(lock), "{reply}");
    assert_eq!(cere["profile_id"], profile, "{reply}");
    assert_eq!(
        cere["config_hash"],
        h.app.deployment(0).config_hash,
        "{reply}"
    );
    let worker = cere["worker_id"].as_str().unwrap();
    assert!(
        workers.contains(&worker),
        "{worker} not in {workers:?}: {reply}"
    );
    let index: usize = worker.strip_prefix("gpu").unwrap().parse().unwrap();
    let rid = cere["request_id"].as_str().unwrap();
    assert!(
        h.replica(index).dispatched_rids().iter().any(|d| d == rid),
        "{reply}"
    );
}

/// C3 (D5, D6, D9): under concurrent load A's requests land on both of A's workers and never on B's, B's only on
/// `gpu2`; each reply carries its deployment's model, lock, recipe and profile and the global config hash; the
/// `usage()` cache report follows each deployment's `enable_cache_report` pin (A off, B on).
#[tokio::test]
async fn c3_each_model_is_served_only_by_its_own_workers() {
    let h = start_ab_mocks(
        mocks_ab(Duration::from_millis(20)).await,
        &[],
        test_tuning(),
    )
    .await;
    let smokes: Vec<usize> = (0..3)
        .map(|i| h.replica(i).dispatched_rids().len())
        .collect();
    let mut per_worker = [0usize; 3];
    for _round in 0..3 {
        let calls: Vec<_> = [MODEL, MODEL, MODEL_B]
            .into_iter()
            .map(|model| {
                let request = h.chat(&body(model, &[("user", "hello")])).send();
                async move { (model, request.await.unwrap()) }
            })
            .collect();
        for (model, r) in futures_util::future::join_all(calls).await {
            assert_eq!(r.status(), StatusCode::OK);
            let reply: Value = r.json().await.unwrap();
            if model == MODEL {
                assert_served(
                    &h,
                    &reply,
                    MODEL,
                    LOCK_A,
                    &h.app.deployment(0).profile_id,
                    &["gpu0", "gpu1"],
                );
                assert_eq!(reply["cere"]["cache_report"], "unavailable", "{reply}");
            } else {
                assert_served(&h, &reply, MODEL_B, LOCK_B, PROFILE_B, &["gpu2"]);
                assert_eq!(reply["cere"]["cache_report"], "reported", "{reply}");
                assert_eq!(reply["cere"]["cached_tokens"], 0, "{reply}");
            }
            let worker = reply["cere"]["worker_id"].as_str().unwrap();
            per_worker[worker
                .strip_prefix("gpu")
                .unwrap()
                .parse::<usize>()
                .unwrap()] += 1;
        }
    }
    assert_ne!(lock_sha256_of(LOCK_A), lock_sha256_of(LOCK_B));
    assert_ne!(h.app.deployment(0).recipe_id, h.app.deployment(1).recipe_id);
    assert_eq!(per_worker[0] + per_worker[1], 6, "{per_worker:?}");
    assert!(
        per_worker[0] > 0 && per_worker[1] > 0,
        "least-loaded inside A: {per_worker:?}"
    );
    assert_eq!(per_worker[2], 3, "{per_worker:?}");
    for i in 0..3 {
        assert_eq!(
            h.replica(i).dispatched_rids().len() - smokes[i],
            per_worker[i],
            "replica {i}"
        );
    }
}

/// C3 (D6): every local worker's cap is its own entry's `expect.max_running_requests`, wired through `App::open`.
/// With B capped at 1 and A at 4 (global cap 9 = 2 × 4 + 1 × 1), eight concurrent A requests run four on each of
/// A's workers while three concurrent B requests reach B's worker one at a time (the others queue); all succeed.
#[tokio::test]
async fn c3_each_local_worker_takes_its_own_entrys_cap() {
    let latency = Duration::from_millis(150);
    let mut mocks = mocks_ab(latency).await;
    let b_info = ServerInfo {
        max_running_requests: 1,
        ..server_info_b()
    };
    mocks[2] = MockSglang::start(MockConfig {
        server_info: b_info,
        token_latency: latency,
        ..MockConfig::default()
    })
    .await
    .unwrap();
    let state_dir = StateDir::new();
    let admission = "max_backend_inflight = 2\nper_scope_inflight = 2";
    let mut doc: toml::Table = config_ab(
        &mocks,
        &state_dir,
        &[(
            admission,
            "max_backend_inflight = 9\nper_scope_inflight = 9",
        )],
    )
    .parse()
    .unwrap();
    doc["deployments"][1]["expect"]["max_running_requests"] = toml::Value::Integer(1);
    let config = toml::to_string(&doc).unwrap();
    let h = Harness::start_config(mocks, state_dir, config, test_tuning()).await;
    let calls: Vec<_> = std::iter::repeat_n(MODEL, 8)
        .chain(std::iter::repeat_n(MODEL_B, 3))
        .map(|model| {
            let request = h.chat(&body(model, &[("user", "hello")])).send();
            async move { (model, request.await.unwrap()) }
        })
        .collect();
    for (model, r) in futures_util::future::join_all(calls).await {
        let status = r.status();
        let reply: Value = r.json().await.unwrap();
        assert_eq!(status, StatusCode::OK, "{model}: {reply}");
        assert_eq!(reply["model"], model, "{reply}");
    }
    assert_eq!(h.replica(2).max_in_flight(), 1, "B's worker cap is B's own");
    assert_eq!(
        [h.replica(0).max_in_flight(), h.replica(1).max_in_flight()],
        [4, 4],
        "A's workers keep A's cap"
    );
}

/// C3 (D5): an unserved model is `404 model_not_found` naming every served model in config order, for chat and
/// for the model route.
#[tokio::test]
async fn c3_an_unknown_model_is_404_naming_both_models() {
    let h = start_ab(&[], test_tuning()).await;
    let message = format!("model 'other' is not served here; served models: {MODEL}, {MODEL_B}");
    let got = chat(&h, "other").await;
    assert_error(&got, 404, "model_not_found", Some("model"));
    assert_eq!(got.1["error"]["message"], message.as_str());
    let got = answer(h.get("/v1/models/other", TOKEN_A)).await;
    assert_error(&got, 404, "model_not_found", Some("model"));
    assert_eq!(got.1["error"]["message"], message.as_str());
}

/// C3 (D5): `/v1/models` lists one card per deployment in config order, each with its `max_model_len`;
/// `/v1/models/{B}` is B's card.
#[tokio::test]
async fn c3_models_lists_both_cards_in_config_order() {
    let h = start_ab(&[], test_tuning()).await;
    let card = |id: &str, len: u64| json!({"id": id, "object": "model", "created": 0, "owned_by": "cere", "max_model_len": len});
    let (status, list) = answer(h.get("/v1/models", TOKEN_A)).await;
    assert_eq!(status, 200, "{list}");
    assert_eq!(
        list,
        json!({"object": "list", "data": [card(MODEL, 200_000), card(MODEL_B, 32_768)]})
    );
    let (status, one) = answer(h.get(&format!("/v1/models/{MODEL_B}"), TOKEN_A)).await;
    assert_eq!((status, one), (200, card(MODEL_B, 32_768)));
}

/// C3 (D5, D9): with both of A's workers out (their pins drift) A answers 503 `not_ready` while B still serves
/// and `/readyz` stays 200 (B's worker is ready).
#[tokio::test]
async fn c3_a_down_is_503_while_b_serves() {
    let h = start_ab(&[], fast_tuning()).await;
    for i in 0..2 {
        h.replica(i).set_server_info_override(Some(ServerInfo {
            kv_cache_dtype: "bf16".to_owned(),
            ..ServerInfo::default()
        }));
    }
    let (status, body) = readyz_until(&h, "A's workers out", |b| {
        ["gpu0", "gpu1"]
            .iter()
            .all(|w| b["workers"][w]["ready"] == false)
    })
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["workers"]["gpu2"]["ready"], true, "{body}");
    let got = chat(&h, MODEL).await;
    assert_error(&got, 503, "not_ready", None);
    let (status, reply) = chat(&h, MODEL_B).await;
    assert_eq!(status, 200, "{reply}");
    assert_served(&h, &reply, MODEL_B, LOCK_B, PROFILE_B, &["gpu2"]);
}

/// M2.14 (m214 design note D4, review ReviewApi P3): A without and B with `api_compat_responses`. The flag check
/// applies only to a served model; an unserved one is the chat path's `model_not_found` naming the served models
/// (never "A is not served" for a model the client did not ask for). Retrieve and cancel are 404 stateless whatever
/// the deployments' flags or A's serving state.
#[tokio::test]
async fn m214_responses_on_b_only_gates_by_the_requested_model() {
    let mocks = mocks_ab(Duration::ZERO).await;
    let state_dir = StateDir::new();
    let mut doc: toml::Table = config_ab(&mocks, &state_dir, &[]).parse().unwrap();
    doc["deployments"].as_array_mut().unwrap()[1]
        .as_table_mut()
        .unwrap()
        .insert("api_compat_responses".to_owned(), true.into());
    let h = Harness::start_config(
        mocks,
        state_dir,
        toml::to_string(&doc).unwrap(),
        fast_tuning(),
    )
    .await;
    let responses = |model: &str| {
        h.post("/v1/responses", TOKEN_A)
            .json(&json!({"model": model, "input": "hello", "temperature": 0}))
    };
    let (status, body) = answer(responses(MODEL)).await;
    assert_eq!(status, 404, "{body}");
    assert_eq!(
        body["error"]["message"],
        format!("model {MODEL} is not served on the responses API").as_str()
    );
    let (status, body) = answer(responses("other")).await;
    assert_eq!(status, 404, "{body}");
    assert_eq!(
        body["error"]["message"],
        format!("model 'other' is not served here; served models: {MODEL}, {MODEL_B}").as_str()
    );
    let stateless = |h: &Harness| {
        [
            h.get("/v1/responses/resp_x", TOKEN_A),
            h.post("/v1/responses/resp_x/cancel", TOKEN_A),
        ]
    };
    for request in stateless(&h) {
        let (status, body) = answer(request).await;
        assert_eq!(status, 404, "{body}");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains("stateless"), "{message}");
    }
    // A down (its pins drift): retrieve and cancel are still 404 stateless, not A's 503.
    for i in 0..2 {
        h.replica(i).set_server_info_override(Some(ServerInfo {
            kv_cache_dtype: "bf16".to_owned(),
            ..ServerInfo::default()
        }));
    }
    readyz_until(&h, "A's workers out", |b| {
        ["gpu0", "gpu1"]
            .iter()
            .all(|w| b["workers"][w]["ready"] == false)
    })
    .await;
    for request in stateless(&h) {
        let (status, body) = answer(request).await;
        assert_eq!(status, 404, "{body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("stateless"),
            "{body}"
        );
    }
    let (status, body) = answer(h.http.get(h.url("/v1/responses/resp_x"))).await;
    assert_eq!(status, 401, "{body}");
}

/// C3 (D6): a session whose owner is outside the request's deployment (an A turn failed after dispatch, so the
/// session is owned by an A worker but unbound) re-pins inside B on a B turn, which binds it to B.
#[tokio::test]
async fn c3_an_owner_outside_the_deployment_re_pins_inside_it() {
    let h = start_ab(&[OPERATOR_TWEAK], test_tuning()).await;
    let id = create(&h).await;
    for i in 0..2 {
        h.replica(i).set_fault(Fault::ServerError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            remaining: 1,
        });
    }
    let (status, failed) = turn(&h, &id, 0, MODEL, &[("user", "hi")]).await;
    assert_ne!(status, 200, "{failed}");
    let before = owner(&h, &id).await;
    let a_owner = before["worker"].as_str().unwrap().to_owned();
    assert!(["gpu0", "gpu1"].contains(&a_owner.as_str()), "{before}");
    assert_eq!(
        view(&h, &id).await["model"],
        Value::Null,
        "the failed turn binds nothing"
    );
    let (status, reply) = turn(&h, &id, 0, MODEL_B, &[("user", "hi")]).await;
    assert_eq!(status, 200, "{reply}");
    assert_served(&h, &reply, MODEL_B, LOCK_B, PROFILE_B, &["gpu2"]);
    assert_eq!(owner(&h, &id).await["worker"], "gpu2");
    assert_eq!(view(&h, &id).await["model"], MODEL_B);
}

/// C4 (D7): the first commit binds the session to A; a B turn is 409 `model_mismatch` (param `model`) with
/// nothing journaled, then the next A turn commits; a fork inherits the binding (B refused, A served).
#[tokio::test]
async fn c4_an_a_bound_session_refuses_b_and_its_fork_inherits() {
    const REPLY: &str = "tok0 tok1 tok2 tok3";
    let h = start_ab(&[], test_tuning()).await;
    let id = create(&h).await;
    assert_eq!(
        view(&h, &id).await["model"],
        Value::Null,
        "unbound until the first commit"
    );
    let (status, reply) = turn(&h, &id, 0, MODEL, &[("user", "hi")]).await;
    assert_eq!(status, 200, "{reply}");
    assert_eq!(
        view(&h, &id).await,
        json!({"session_id": id, "revision": 1, "message_count": 2, "model": MODEL, "parent": null})
    );
    let history = [("user", "hi"), ("assistant", REPLY), ("user", "more")];
    let records = journal(h.state_dir.path()).len();
    let dispatched: Vec<usize> = (0..3)
        .map(|i| h.replica(i).dispatched_rids().len())
        .collect();
    let got = turn(&h, &id, 1, MODEL_B, &history).await;
    assert_error(&got, 409, "model_mismatch", Some("model"));
    assert_eq!(
        journal(h.state_dir.path()).len(),
        records,
        "nothing journaled"
    );
    let after: Vec<usize> = (0..3)
        .map(|i| h.replica(i).dispatched_rids().len())
        .collect();
    assert_eq!(after, dispatched, "nothing dispatched");
    let (status, reply) = turn(&h, &id, 1, MODEL, &history).await;
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply["cere"]["revision"], 2, "{reply}");

    let (status, fork) = answer(
        h.post(&format!("/v1/cere/sessions/{id}/fork"), TOKEN_A)
            .json(&json!({"revision": 1})),
    )
    .await;
    assert_eq!(status, 201, "{fork}");
    let child = fork["session_id"].as_str().unwrap().to_owned();
    assert_eq!(view(&h, &child).await["model"], MODEL, "a fork inherits");
    let rev = fork["revision"].as_u64().unwrap();
    let got = turn(&h, &child, rev, MODEL_B, &history).await;
    assert_error(&got, 409, "model_mismatch", Some("model"));
    let (status, reply) = turn(&h, &child, rev, MODEL, &history).await;
    assert_eq!(status, 200, "{reply}");
}

/// C4 (D7): a B binding survives a restart (an A turn is still refused); a later config that no longer serves B
/// answers the bound session's B turn 404 `model_not_found` (validation precedes the session check).
#[tokio::test]
async fn c4_binding_survives_restart_and_a_removed_model_is_404() {
    let h = start_ab(&[], test_tuning()).await;
    let id = create(&h).await;
    let (status, reply) = turn(&h, &id, 0, MODEL_B, &[("user", "hi")]).await;
    assert_eq!(status, 200, "{reply}");
    let history = [
        ("user", "hi"),
        ("assistant", "tok0 tok1 tok2 tok3"),
        ("user", "more"),
    ];
    let h = h.restart().await;
    assert_eq!(view(&h, &id).await["model"], MODEL_B);
    let got = turn(&h, &id, 1, MODEL, &history).await;
    assert_error(&got, 409, "model_mismatch", Some("model"));

    let urls: Vec<String> = (0..2).map(|i| h.replica(i).base_url()).collect();
    let only_a = replicas_config_text(&urls, h.state_dir.path(), 18391, &[]);
    let h = h.restart_with_config(only_a).await;
    assert_eq!(view(&h, &id).await["model"], MODEL_B);
    let got = turn(&h, &id, 1, MODEL_B, &history).await;
    assert_error(&got, 404, "model_not_found", Some("model"));
    assert_eq!(
        got.1["error"]["message"],
        format!("model '{MODEL_B}' is not served here; served models: {MODEL}")
    );
}

/// C4 (D8): migrating an A-bound session to B's worker is 409 `model_mismatch` (param `model`) and changes
/// nothing; an unknown worker is still `worker_not_found`; a move inside A is unchanged; an unbound ownerless
/// session skips the model check.
#[tokio::test]
async fn c4_migrate_stays_inside_the_deployment() {
    let h = start_ab(&[OPERATOR_TWEAK], test_tuning()).await;
    let id = create(&h).await;
    let (status, reply) = turn(&h, &id, 0, MODEL, &[("user", "hi")]).await;
    assert_eq!(status, 200, "{reply}");
    let before = owner(&h, &id).await;
    let a_owner = before["worker"].as_str().unwrap().to_owned();
    let other_a = if a_owner == "gpu0" { "gpu1" } else { "gpu0" };
    assert_error(
        &migrate(&h, &id, "gpu9", 1).await,
        404,
        "worker_not_found",
        None,
    );
    assert_error(
        &migrate(&h, &id, "gpu2", 1).await,
        409,
        "model_mismatch",
        Some("model"),
    );
    assert_eq!(
        owner(&h, &id).await,
        before,
        "a refused move changes nothing"
    );
    let (status, moved) = migrate(&h, &id, other_a, 1).await;
    assert_eq!(status, 200, "{moved}");
    assert_eq!(moved["worker"], other_a, "{moved}");
    assert_eq!(moved["owner_generation"], 2, "{moved}");

    let fresh = create(&h).await;
    let (status, moved) = migrate(&h, &fresh, "gpu2", 0).await;
    assert_eq!(status, 200, "{moved}");
    assert_eq!(moved["worker"], "gpu2", "{moved}");
    let (status, reply) = turn(&h, &fresh, 0, MODEL_B, &[("user", "hi")]).await;
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply["cere"]["worker_id"], "gpu2", "{reply}");
}
