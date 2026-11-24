//! WeightLab (`-v57`, weightlab design note WL-01, WL-04, WL-11, WL-14): the research routes and plan-bound
//! requests over the CPU toy executor. The toy writes every selected capture row of a bound session as a function of
//! its position (research/capture.cpp `write_toy_row`: an I32 column `j` of layer `l` at position `p` is
//! `(p + l + j) % 512`), so a capture's bytes are checked here exactly; a capture-only plan changes no token (the
//! bound scope's output equals the unbound scope's). The TEXT prompt is 716 tokens (`server_native.rs` TEXT).
mod native_server;
mod native_support;
mod support;

use native_server::{Case, Native, OPERATOR_TOKEN, TOKEN};
use serde_json::{json, Value};
use support::{http, MODEL};

const SENTINEL: &str = "cere-native-sentinel-prompt";
/// Prompt length (in the stub's tokens) and last character (`server_native.rs` TEXT).
const TEXT: (usize, char) = (716, '1');
/// The second scope's token (`CERE_TOKEN_NATIVE_B`, scope `native-b`).
const TOKEN_B: &str = "secret-b";
const PLANS: &str = "/v1/cere/research/plans";

fn prompt() -> String {
    let mut s = SENTINEL.to_owned();
    while s.len() < TEXT.0 - 2 {
        s.push('x');
    }
    s.push(TEXT.1);
    s
}

/// `[deployments.native.research]` with `host` bytes of host ring (and per-request cap) and `max_plans`.
fn research_case(host: u64, max_plans: u32) -> Case {
    Case {
        research: json!({
            "device_staging_bytes": 1u64 << 20,
            "host_buffer_bytes": host,
            "max_capture_bytes_per_request": host,
            "max_plans": max_plans,
            "max_vectors": 4,
        }),
        ..Case::matching()
    }
}

/// The capture plan of the routing point at layers 0 and 3: prompt positions [0, 4) and 2 generated positions.
fn route_plan() -> Value {
    json!({"target": "coding_output_volume", "capture": [{
        "point": "native.layers.L.moe_route", "layers": [0, 3],
        "positions": {"prompt": [[0, 4]], "generated": 2}}]})
}

async fn operator(
    native: &Native,
    method: reqwest::Method,
    path: &str,
    bearer: Option<&str>,
    body: Option<&Value>,
) -> (u16, Vec<u8>) {
    let mut request = http().request(method, native.cered.url(path));
    if let Some(bearer) = bearer {
        request = request.bearer_auth(bearer);
    }
    if let Some(body) = body {
        request = request.json(body);
    }
    let response = request.send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.bytes().await.unwrap().to_vec())
}

async fn post(native: &Native, path: &str, body: &Value) -> (u16, Value) {
    let (status, bytes) = operator(
        native,
        reqwest::Method::POST,
        path,
        Some(OPERATOR_TOKEN),
        Some(body),
    )
    .await;
    (status, serde_json::from_slice(&bytes).unwrap())
}

async fn delete(native: &Native, path: &str) -> (u16, Value) {
    let (status, bytes) = operator(
        native,
        reqwest::Method::DELETE,
        path,
        Some(OPERATOR_TOKEN),
        None,
    )
    .await;
    (status, serde_json::from_slice(&bytes).unwrap())
}

async fn get(native: &Native, path: &str) -> (u16, Vec<u8>) {
    operator(
        native,
        reqwest::Method::GET,
        path,
        Some(OPERATOR_TOKEN),
        None,
    )
    .await
}

fn code(body: &Value) -> &str {
    body["error"]["code"].as_str().unwrap_or_default()
}

/// Registers `plan`; returns `(plan_id, plan_sha256)`.
async fn register(native: &Native, plan: &Value) -> (String, String) {
    let (status, body) = post(native, PLANS, plan).await;
    assert_eq!(status, 200, "{body}");
    (
        body["plan_id"].as_str().unwrap().to_owned(),
        body["plan_sha256"].as_str().unwrap().to_owned(),
    )
}

async fn bind(native: &Native, scope: &str, plan_id: &str) -> (u16, Value) {
    post(
        native,
        "/v1/cere/research/bindings",
        &json!({"scope": scope, "plan_id": plan_id}),
    )
    .await
}

/// A unary chat of `token`'s scope generating 2 tokens: `(status, request id, research header, body)`.
async fn chat(native: &Native, token: &str) -> (u16, String, Option<String>, Value) {
    let response = http()
        .post(native.cered.url("/v1/chat/completions"))
        .bearer_auth(token)
        .json(&json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt()}],
            "max_tokens": 2,
            "temperature": 0,
        }))
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    let header = |name: &str| {
        response
            .headers()
            .get(name)
            .map(|v| v.to_str().unwrap().to_owned())
    };
    let rid = header("x-cere-request-id").unwrap_or_default();
    let plan = header("x-cere-research-plan");
    (status, rid, plan, response.json().await.unwrap())
}

/// The `cered request` line of `rid` (written at its terminal, which may follow the reply).
async fn request_line(native: &Native, rid: &str) -> String {
    let prefix = format!("cered request rid={rid} ");
    for _ in 0..400 {
        if let Some(line) = native
            .cered
            .stderr_so_far()
            .lines()
            .find(|l| l.starts_with(&prefix))
        {
            return line.to_owned();
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    panic!("no request line for {rid}");
}

/// `(header, data)` of a safetensors file.
fn safetensors(file: &[u8]) -> (Value, &[u8]) {
    let n = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
    (
        serde_json::from_slice(&file[8..8 + n]).unwrap(),
        &file[8 + n..],
    )
}

/// The routes exist only with the table (404 without it); with it they take the operator bearer: none or an
/// unknown one 401, a client scope token 403 (as every operator route, m35 design note Q13).
#[tokio::test]
async fn research_routes_need_the_table_and_the_operator_bearer() {
    let plain = Native::ready(Case::matching()).await;
    let (status, _) = operator(
        &plain,
        reqwest::Method::POST,
        PLANS,
        Some(OPERATOR_TOKEN),
        Some(&route_plan()),
    )
    .await;
    assert_eq!(status, 404);
    // Default off: no research header and no `research_plan=` on the request line.
    let (status, rid, plan, body) = chat(&plain, TOKEN).await;
    assert_eq!((status, plan), (200, None), "{body}");
    let line = request_line(&plain, &rid).await;
    assert!(!line.contains("research_plan="), "{line}");
    drop(plain);

    let native = Native::ready(research_case(1 << 20, 2)).await;
    for (bearer, want) in [(None, 401), (Some("not-a-token"), 401), (Some(TOKEN), 403)] {
        for (method, path) in [
            (reqwest::Method::POST, PLANS),
            (reqwest::Method::DELETE, "/v1/cere/research/plans/plan_1"),
            (reqwest::Method::POST, "/v1/cere/research/bindings"),
            (
                reqwest::Method::DELETE,
                "/v1/cere/research/bindings/native-a",
            ),
            (reqwest::Method::GET, "/v1/cere/research/captures"),
            (reqwest::Method::GET, "/v1/cere/research/captures/x"),
            (reqwest::Method::POST, "/v1/cere/research/vectors?name=v"),
            (reqwest::Method::DELETE, "/v1/cere/research/vectors/x"),
        ] {
            let (status, _) =
                operator(&native, method.clone(), path, bearer, Some(&json!({}))).await;
            assert_eq!(status, want, "{method} {path} with {bearer:?}");
        }
    }
}

/// A bound scope's request runs under its plan: the reply names the plan, its capture is listed with the manifest
/// (positions: the prompt range and the generated rows from P - 1), drained once as a safetensors file whose rows
/// are the toy's, and its tokens equal the unbound scope's (a capture-only plan). The request line ends with
/// `research_plan=<sha256>`, an unbound scope's with `research_plan=-`.
#[tokio::test]
async fn a_bound_scope_captures_and_drains_once_and_its_tokens_are_unchanged() {
    let native = Native::ready(research_case(1 << 20, 2)).await;
    let (plan_id, sha) = register(&native, &route_plan()).await;
    let (status, body) = bind(&native, "native-a", &plan_id).await;
    assert_eq!(status, 200, "{body}");

    let (status, rid, plan, bound) = chat(&native, TOKEN).await;
    assert_eq!(status, 200, "{bound}");
    assert_eq!(plan.as_deref(), Some(sha.as_str()));
    let (status, base_rid, plan, base) = chat(&native, TOKEN_B).await;
    assert_eq!((status, plan), (200, None), "{base}");
    assert_eq!(
        bound["choices"], base["choices"],
        "a capture-only plan changes no token"
    );
    assert_eq!(bound["usage"], base["usage"], "{bound}");
    assert_eq!(bound["usage"]["completion_tokens"], json!(2), "{bound}");
    native.terminal(&rid).await;

    let (status, list) = get(&native, "/v1/cere/research/captures?scope=native-a").await;
    assert_eq!(status, 200);
    let list: Value = serde_json::from_slice(&list).unwrap();
    let captures = list["captures"].as_array().unwrap();
    assert_eq!(captures.len(), 1, "{list}");
    let manifest = &captures[0];
    assert_eq!(
        (
            &manifest["request_id"],
            &manifest["scope"],
            &manifest["plan_id"],
            &manifest["plan_sha256"],
            &manifest["prefix_lookup"],
        ),
        (
            &json!(rid),
            &json!("native-a"),
            &json!(plan_id),
            &json!(sha),
            &json!("skipped"),
        ),
        "{manifest}"
    );
    let (status, none) = get(&native, "/v1/cere/research/captures?scope=native-b").await;
    assert_eq!(status, 200);
    assert_eq!(
        serde_json::from_slice::<Value>(&none).unwrap(),
        json!({"captures": []})
    );

    let (status, file) = get(&native, &format!("/v1/cere/research/captures/{rid}")).await;
    assert_eq!(status, 200);
    assert_eq!(manifest["bytes"], json!(file.len()));
    let (header, data) = safetensors(&file);
    assert_eq!(header["__metadata__"]["plan_sha256"], json!(sha));
    let tensors = manifest["tensors"].as_array().unwrap();
    // Two parts (ids, weights) at two layers.
    assert_eq!(tensors.len(), 4, "{manifest}");
    let want: Vec<u64> = vec![0, 1, 2, 3, TEXT.0 as u64 - 1, TEXT.0 as u64];
    for t in tensors {
        let positions: Vec<u64> = serde_json::from_value(t["positions"].clone()).unwrap();
        assert_eq!(positions, want, "{t}");
        assert_eq!(t["shape"], json!([want.len(), 10]), "{t}");
        let name = t["name"].as_str().unwrap();
        let offsets = &header[name]["data_offsets"];
        let (a, b) = (
            offsets[0].as_u64().unwrap() as usize,
            offsets[1].as_u64().unwrap() as usize,
        );
        if t["dtype"] == "I32" {
            let layer = t["layer"].as_u64().unwrap();
            let got: Vec<i32> = data[a..b]
                .chunks(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let expect: Vec<i32> = want
                .iter()
                .flat_map(|p| (0..10).map(move |j| ((p + layer + j) % 512) as i32))
                .collect();
            assert_eq!(got, expect, "{name}");
        }
    }
    // Drained once.
    let (status, again) = get(&native, &format!("/v1/cere/research/captures/{rid}")).await;
    let again: Value = serde_json::from_slice(&again).unwrap();
    assert_eq!((status, code(&again)), (404, "research_capture_not_found"));

    let line = request_line(&native, &rid).await;
    assert!(line.ends_with(&format!(" research_plan={sha}")), "{line}");
    let line = request_line(&native, &base_rid).await;
    assert!(line.ends_with(" research_plan=-"), "{line}");
}

/// Every refusal of the routes (spec "Experiment API"): 400 `invalid_research_plan` and
/// `research_budget_exceeded`, 507 `research_plans_full`, 404 `research_plan_not_found` /
/// `research_scope_not_found` / `research_binding_not_found` / `research_capture_not_found`, 409
/// `research_scope_bound` and `research_plan_busy` (a bound plan), and the plan deletes once unbound.
#[tokio::test]
async fn research_route_refusals() {
    let native = Native::ready(research_case(1 << 20, 1)).await;
    let (status, body) = post(
        &native,
        PLANS,
        &json!({"target": "coding_output_volume", "capture": [{"point": "native.layers.L.nope"}]}),
    )
    .await;
    assert_eq!(
        (status, code(&body)),
        (400, "invalid_research_plan"),
        "{body}"
    );
    // 2,000 prompt rows of a 5,120-byte point: above the 1 MiB cap.
    let (status, body) = post(
        &native,
        PLANS,
        &json!({"target": "coding_output_volume", "capture": [{
            "point": "native.final_mix.out", "positions": {"prompt": [[0, 2000]]}}]}),
    )
    .await;
    assert_eq!(
        (status, code(&body)),
        (400, "research_budget_exceeded"),
        "{body}"
    );
    let (plan_id, _) = register(&native, &route_plan()).await;
    let (status, body) = post(&native, PLANS, &route_plan()).await;
    assert_eq!(
        (status, code(&body)),
        (507, "research_plans_full"),
        "{body}"
    );

    for (scope, plan, want) in [
        ("native-a", "plan_99", "research_plan_not_found"),
        (
            "no-such-scope",
            plan_id.as_str(),
            "research_scope_not_found",
        ),
    ] {
        let (status, body) = bind(&native, scope, plan).await;
        assert_eq!((status, code(&body)), (404, want), "{body}");
    }
    let (status, body) = bind(&native, "native-a", &plan_id).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = bind(&native, "native-a", &plan_id).await;
    assert_eq!(
        (status, code(&body)),
        (409, "research_scope_bound"),
        "{body}"
    );
    let (status, body) = delete(&native, &format!("{PLANS}/{plan_id}")).await;
    assert_eq!((status, code(&body)), (409, "research_plan_busy"), "{body}");
    let (status, body) = delete(&native, "/v1/cere/research/bindings/native-b").await;
    assert_eq!(
        (status, code(&body)),
        (404, "research_binding_not_found"),
        "{body}"
    );
    let (status, body) = delete(&native, "/v1/cere/research/bindings/native-a").await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = delete(&native, &format!("{PLANS}/{plan_id}")).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = delete(&native, &format!("{PLANS}/{plan_id}")).await;
    assert_eq!(
        (status, code(&body)),
        (404, "research_plan_not_found"),
        "{body}"
    );
    for path in [
        "/v1/cere/research/captures/nope",
        "/v1/cere/research/captures?after=nope",
    ] {
        let (status, body) = get(&native, path).await;
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            (status, code(&body)),
            (404, "research_capture_not_found"),
            "{path}: {body}"
        );
    }
}

/// WL-04 admission: a request whose bound does not fit what undrained captures leave of the host ring is 503
/// `research_buffer_full` and runs nothing; after the drain the next one runs.
#[tokio::test]
async fn a_request_past_the_host_buffer_is_refused_until_a_drain() {
    let native = Native::ready(research_case(1 << 20, 2)).await;
    // 110 prompt rows + 2 generated of 5,120 bytes: 573,440 bytes, two of which exceed 1 MiB.
    let plan = json!({"target": "coding_output_volume", "capture": [{
        "point": "native.final_mix.out", "positions": {"prompt": [[0, 110]], "generated": 2}}]});
    let (plan_id, _) = register(&native, &plan).await;
    let (status, body) = bind(&native, "native-a", &plan_id).await;
    assert_eq!(status, 200, "{body}");
    let (status, first, _, body) = chat(&native, TOKEN).await;
    assert_eq!(status, 200, "{body}");
    native.terminal(&first).await;
    let (status, _, _, body) = chat(&native, TOKEN).await;
    assert_eq!(
        (status, code(&body)),
        (503, "research_buffer_full"),
        "{body}"
    );
    let (status, _) = get(&native, &format!("/v1/cere/research/captures/{first}")).await;
    assert_eq!(status, 200);
    let (status, _, _, body) = chat(&native, TOKEN).await;
    assert_eq!(status, 200, "{body}");
}

/// WL-01: a research config on a library without the six research calls (the pre-staging toy) fails the load,
/// naming them; the same library loads without the table.
#[tokio::test]
async fn research_is_refused_at_load_without_the_library_calls() {
    let old = native_support::toy_library_prestaging();
    let old_case = |research: Value| Case {
        library_pin: native_server::file_sha256(&old),
        library: old.clone(),
        research,
        ..Case::matching()
    };
    Native::ready(old_case(Value::Null)).await;
    let refused = Native::start(old_case(research_case(1 << 20, 2).research)).await;
    let (status, body) = refused.settled().await;
    assert_ne!(status, 200, "{body}");
    assert!(
        body.to_string()
            .contains("research needs a library exporting cere_model_research_reserve"),
        "{body}"
    );
}

/// The toy's step record (`toy,record=PATH`): one `key=value` map per committed step, grouped by session in
/// first-step order.
fn sessions(path: &std::path::Path) -> Vec<Vec<std::collections::BTreeMap<String, String>>> {
    let mut out: Vec<Vec<std::collections::BTreeMap<String, String>>> = Vec::new();
    for line in std::fs::read_to_string(path).unwrap_or_default().lines() {
        let step: std::collections::BTreeMap<String, String> = line
            .split(' ')
            .map(|kv| {
                let (k, v) = kv.split_once('=').unwrap();
                (k.to_owned(), v.to_owned())
            })
            .collect();
        match out.iter_mut().find(|s| s[0]["session"] == step["session"]) {
            Some(s) => s.push(step),
            None => out.push(vec![step]),
        }
    }
    out
}

/// WL-11 (our rule: every plan session is pinned, capture-only or intervening): with `preempt_every = 4` a
/// session of a bound scope is never parked or snapshotted (as a LoRA adapter session): its 16 tokens are one
/// session of the toy's record and no preemption is counted; the unbound scope's same request is preempted at
/// every 4 tokens (four sessions, three preemptions).
#[tokio::test]
async fn every_plan_session_capture_only_or_intervening_is_never_parked_or_snapshotted() {
    let intervening = json!({"target": "coding_output_volume", "interventions": [{"kind": "ablate_zero",
        "point": "native.layers.L.attn_mix.out", "layers": [0], "positions": {"generated": true}}]});
    for (label, plan) in [("capture-only", route_plan()), ("intervening", intervening)] {
        let t = native_support::Tmp::new("research-pinned");
        let record = t.join("steps.record");
        let native = Native::ready(Case {
            device_id: format!("toy,record={}", record.display()),
            snapshot: Some(native_server::SnapshotProfile {
                preempt: true,
                preempt_min_tokens: 512,
                inflight: 2,
                preempt_every: 4,
                rows: 1,
                parked: 1,
                reserve: 0,
            }),
            ..research_case(1 << 20, 2)
        })
        .await;
        // The readiness smoke's sessions come first in the record.
        let smoke = sessions(&record).len();
        let (plan_id, _) = register(&native, &plan).await;
        let (status, body) = bind(&native, "native-a", &plan_id).await;
        assert_eq!(status, 200, "{label}: {body}");
        let long = |token: &'static str| {
            http()
                .post(native.cered.url("/v1/chat/completions"))
                .bearer_auth(token)
                .json(&json!({
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt()}],
                    "max_tokens": 16,
                    "temperature": 0,
                }))
                .send()
        };

        let response = long(TOKEN).await.unwrap();
        assert_eq!(response.status(), 200, "{label}");
        assert!(
            response.headers().contains_key("x-cere-research-plan"),
            "{label}"
        );
        let bound: Value = response.json().await.unwrap();
        let pinned = sessions(&record);
        let lens: Vec<usize> = pinned[smoke..].iter().map(Vec::len).collect();
        assert_eq!(lens, [16], "{label}: a plan session was parked: {pinned:?}");
        assert_eq!(native.preemptions().await, 0, "{label}");

        let response = long(TOKEN_B).await.unwrap();
        assert_eq!(response.status(), 200, "{label}");
        assert!(
            !response.headers().contains_key("x-cere-research-plan"),
            "{label}"
        );
        let unbound: Value = response.json().await.unwrap();
        let all = sessions(&record);
        let lens: Vec<usize> = all[smoke + 1..].iter().map(Vec::len).collect();
        assert_eq!(
            lens, [4; 4],
            "{label}: the unbound control was not preempted: {all:?}"
        );
        assert_eq!(native.preemptions().await, 3, "{label}");
        assert_eq!(
            bound["usage"]["completion_tokens"], unbound["usage"]["completion_tokens"],
            "{label}"
        );
    }
}

/// The shipped live plans (scripts/research/plans: WL-12 steering, WL-16's no-op plan) register and bind through the
/// executor's plan_load once `@steer` names an uploaded [1, 2,560] BF16 vector (as wl_live.py substitutes it), and so
/// does one plan per expert kind (their `cere_research_op.top_k` is 0: research/plan.cpp `load_op`).
#[tokio::test]
async fn the_shipped_live_plans_register_and_bind() {
    let native = Native::ready(research_case(1 << 20, 5)).await;
    let data: Vec<u8> = (0..2_560).flat_map(|_| 0x3f80u16.to_le_bytes()).collect();
    let mut head = serde_json::to_vec(
        &json!({"v": {"dtype": "BF16", "shape": [1, 2_560], "data_offsets": [0, data.len()]}}),
    )
    .unwrap();
    while head.len() % 8 != 0 {
        head.push(b' ');
    }
    let mut file = (head.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&head);
    file.extend_from_slice(&data);
    let response = http()
        .post(native.cered.url("/v1/cere/research/vectors?name=steer"))
        .bearer_auth(OPERATOR_TOKEN)
        .body(file)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let vector: Value = response.json().await.unwrap();
    let id = vector["vector_id"].as_str().unwrap();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/research/plans");
    for (name, scope) in [
        ("wl12-steer.json", "native-a"),
        ("wl16-noop.json", "native-b"),
    ] {
        let text = std::fs::read_to_string(dir.join(name)).unwrap();
        let plan: Value = serde_json::from_str(&text.replace("@steer", id)).unwrap();
        let (status, body) = post(&native, PLANS, &plan).await;
        assert_eq!(status, 200, "{name}: {body}");
        let (status, body) = bind(&native, scope, body["plan_id"].as_str().unwrap()).await;
        assert_eq!(status, 200, "{name}: {body}");
    }
    let (router, route) = (
        "native.layers.L.gemm.moe_router.out",
        "native.layers.L.moe_route",
    );
    let pos = json!({"prompt": [[0, 4]], "generated": true});
    for op in [
        json!({"kind": "expert_mask", "point": router, "layers": "all", "positions": pos, "experts": [3, 7]}),
        json!({"kind": "expert_bias", "point": router, "layers": [0], "positions": pos,
               "experts": [1], "weights": [-2.0]}),
        json!({"kind": "expert_force", "point": route, "layers": [3], "positions": pos,
               "experts": [1, 2], "weights": [0.5, 0.5]}),
    ] {
        let plan = json!({"target": "coding_output_volume", "interventions": [op]});
        let (status, body) = post(&native, PLANS, &plan).await;
        assert_eq!(status, 200, "{op}: {body}");
    }
}

/// server-info design note SI-03: research captures export prompt-derived activations, so a research cered reports
/// them on `GET /get_server_info` (`debug_tensor_dump_output_folder`, SGLang's tensor-dump key) and a vetting client
/// refuses it; the same toy without the table reports none.
#[tokio::test]
async fn server_info_reports_research_captures() {
    let native = Native::ready(research_case(1 << 20, 2)).await;
    let (status, info) = native.get("/get_server_info").await;
    assert_eq!(status, 200, "{info}");
    assert_eq!(
        info["debug_tensor_dump_output_folder"], "/v1/cere/research/captures",
        "{info}"
    );
    assert_eq!(info["cere"]["research"], true, "{info}");
    assert_eq!(info["log_requests"], false, "{info}");
    assert_eq!(
        support::client_refusal(&info, MODEL, false).as_deref(),
        Some("the server may export prompt data (debug_tensor_dump_output_folder)"),
        "{info}"
    );
}
