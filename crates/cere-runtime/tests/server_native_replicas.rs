//! m32n design note: one `cered` serving one native deployment with two replicas (`gpu0`, `gpu1`) over the CPU toy
//! executor, the fixture artifact and the stub sidecar ([`native_server`]). The stub encodes a user message as `[1]`
//! plus its UTF-8 bytes, so a prompt's token count is its length. A toy with `hold_after=N` holds a request after
//! its N-th generated token (until cancelled or `hold_ms`), which keeps a row and its pool reservation charged; a
//! `device_id` of `a;b` gives each replica its own toy options.
mod native_server;
mod native_support;
mod support;

use std::time::Duration;

use native_server::{
    Case, Native, DEVICE_UUID, OPERATOR_TOKEN, SECOND_UUID, SECOND_WORKER, TOKEN, WORKER,
};
use native_support::Tmp;
use serde_json::{json, Value};
use support::{http, MODEL};

/// Starts every test prompt (as the single-card tests' prompts do).
const SENTINEL: &str = "cere-native-sentinel-prompt";
/// A toy that holds every request after its second token for up to a minute.
const HOLD: &str = "toy,hold_after=2,hold_ms=60000";

/// A user message of `len` prompt tokens.
fn body(len: usize, max_tokens: u64, stream: bool) -> Value {
    let mut content = SENTINEL.to_owned();
    while content.len() < len - 1 {
        content.push('x');
    }
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    })
}

fn rid(response: &reqwest::Response) -> String {
    response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned()
}

/// The worker the request `rid` was granted, once it is dispatched.
async fn granted(native: &Native, rid: &str) -> String {
    for _ in 0..800 {
        let (status, body) = native.get(&format!("/v1/cere/requests/{rid}")).await;
        assert_eq!(status, 200, "{body}");
        if let Some(worker) = body["worker_id"].as_str() {
            return worker.to_owned();
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("request {rid} was never dispatched");
}

/// Starts a streamed request and waits for its grant: the response (held open: dropping it cancels the request)
/// and the granted worker.
async fn hold(native: &Native, body: &Value) -> (reqwest::Response, String) {
    let response = native.chat(body).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let worker = granted(native, &rid(&response)).await;
    (response, worker)
}

/// A unary request's status and body.
async fn answer(native: &Native, body: &Value) -> (u16, Value) {
    let response = native.chat(body).send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

/// R2/R3 (D2, D3): both replicas load from one library, pass their own identity checks and SMOKE, and are Ready
/// in `/readyz`, whose `capacity` (R12, C8) counts both cards; a request completes on one of them.
#[tokio::test]
async fn r2_both_replicas_are_ready_and_serve() {
    let native = Native::ready(Case::replicas()).await;
    let (status, body) = native.settled().await;
    assert_eq!(status, 200, "{body}");
    for worker in [WORKER, SECOND_WORKER] {
        assert_eq!(body["workers"][worker]["state"], "ready", "{body}");
        assert_eq!(body["workers"][worker]["ready"], true, "{body}");
    }
    assert_ne!(
        body["workers"][WORKER]["instance_id"], body["workers"][SECOND_WORKER]["instance_id"],
        "{body}"
    );
    let rows = rows_max(&native).await;
    assert_eq!(body["capacity"], capacity(2, rows), "{body}");
    let (status, got) = answer(&native, &body_unary()).await;
    assert_eq!(status, 200, "{got}");
    assert!(
        [WORKER, SECOND_WORKER].contains(&got["cere"]["worker_id"].as_str().unwrap()),
        "{got}"
    );
}

/// What the toy prints when a create meets a shared segment this process is still creating (host/host_share.cpp,
/// answered BUSY).
const STILL_CREATING: &str = "this process is still creating it";
/// What `cered` prints when a replica waits for the sibling replica that holds the shared segment (readiness.rs).
const WAITING: &str = "waiting for the replica that is loading the shared PLE segment";

/// The published shared segments (`cere-host-<sha256>`, not their `.lock` / `.tmp` companions) in the
/// deployment's `host_share_dir`.
fn published_segments(native: &Native) -> Vec<String> {
    let config: toml::Table =
        toml::from_str(&std::fs::read_to_string(&native.config_path).unwrap()).unwrap();
    let dir = config["deployments"][0]["native"]["host_share_dir"]
        .as_str()
        .unwrap()
        .to_owned();
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.starts_with("cere-host-") && !n.ends_with(".lock") && !n.ends_with(".tmp"))
        .collect()
}

/// D21 (G-3 attempt 1): the replicas load at once and share the deployment's one PLE segment (`host_share_dir`).
/// Both toys hold their staged puts on a gate, so the replica that creates the segment is still uploading (the
/// segment unsealed) when the other reaches ALLOCATE: that one waits for the seal, then attaches, and both are
/// Ready over one published segment. Before the fix its create answered BUSY and it stayed refused.
#[tokio::test]
async fn d21_a_replica_waits_for_its_sibling_to_seal_the_shared_segment_then_attaches() {
    let t = Tmp::new("d21-segment");
    let gate = gate(&t);
    let native = Native::start(Case {
        device_id: format!("toy,staging_gate={},hold_ms=120000", gate.display()),
        host_share: true,
        ..Case::replicas()
    })
    .await;
    let until = std::time::Instant::now() + Duration::from_secs(60);
    loop {
        let log = native.cered.stderr_so_far();
        if log.contains(WAITING) || log.contains(STILL_CREATING) {
            break;
        }
        assert!(
            std::time::Instant::now() < until,
            "no replica reached ALLOCATE while the other held the unsealed segment: {log}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    std::fs::remove_file(&gate).unwrap();
    let (status, body) = native.settled().await;
    let log = native.cered.stderr_so_far();
    assert_eq!(status, 200, "{body}\n{log}");
    for worker in [WORKER, SECOND_WORKER] {
        assert_eq!(body["workers"][worker]["ready"], true, "{body}\n{log}");
    }
    assert!(!log.contains(STILL_CREATING), "{log}");
    assert_eq!(log.matches(WAITING).count(), 1, "{log}");
    assert_eq!(published_segments(&native).len(), 1, "{log}");
    let rows = rows_max(&native).await;
    assert_eq!(body["capacity"], capacity(2, rows), "{body}");
}

/// D21: serialized, a sibling replica never makes a create BUSY, so a BUSY create is a real busy segment and is
/// neither retried nor waited out: `gpu1`'s toy answers its first create BUSY (`create_busy=1`), its load fails
/// at once with that error and `gpu1` stays not ready, while `gpu0` (which created the segment) serves.
#[tokio::test]
async fn d21_a_busy_create_fails_the_load_at_once() {
    let native = Native::start(Case {
        device_id: "toy;toy,create_busy=1".to_owned(),
        host_share: true,
        ..Case::replicas()
    })
    .await;
    let (status, body) = native.settled().await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["workers"][WORKER]["ready"], true, "{body}");
    assert_eq!(body["workers"][SECOND_WORKER]["ready"], false, "{body}");
    let problems = body["workers"][SECOND_WORKER]["problems"].to_string();
    assert!(
        problems.contains("cere_model_create") && problems.contains("busy"),
        "{body}"
    );
    // Later probes load nothing more: the refusal stands.
    let probed = |body: &Value| body["workers"][SECOND_WORKER]["probed_at_unix_ms"].as_u64();
    let after = readyz_until(&native, "a later probe of gpu1", |later| {
        probed(later) > probed(&body)
    })
    .await;
    assert_eq!(after["workers"][SECOND_WORKER]["ready"], false, "{after}");
    let (status, got) = answer(&native, &body_unary()).await;
    assert_eq!(status, 200, "{got}");
    assert_eq!(got["cere"]["worker_id"], WORKER, "{got}");
}

/// The KV pool of each card of [`Case::replicas`] (`configs/runtime/native-2gpu.toml`).
const CARD_POOL: u64 = 200_000;

/// The `rows_max` both cards report in the scope view of `GET /v1/cere/stats` (the profile's
/// `decode_batch_max`).
async fn rows_max(native: &Native) -> u64 {
    let (status, stats) = native.get("/v1/cere/stats").await;
    assert_eq!(status, 200, "{stats}");
    let rows = stats["workers"][WORKER]["rows_max"].as_u64().unwrap();
    assert!(rows > 0, "{stats}");
    assert_eq!(stats["workers"][SECOND_WORKER]["rows_max"], rows, "{stats}");
    rows
}

/// The R12 `/readyz` `capacity` with `ready` of the two cards ready.
fn capacity(ready: u64, rows: u64) -> Value {
    json!({
        "workers_ready": ready,
        "workers_configured": 2,
        "rows": { "ready": ready * rows, "configured": 2 * rows },
        "kv_pool_tokens": { "ready": ready * CARD_POOL, "configured": 2 * CARD_POOL },
    })
}

fn body_unary() -> Value {
    body(600, 4, false)
}

/// R8 (D11 (3)): with no pin and no affinity a request goes to the least-loaded card: the first to `gpu0` (the
/// tie goes to the worker id), the second, while the first holds a row there, to `gpu1`.
#[tokio::test]
async fn r8_concurrent_requests_spread_over_both_cards() {
    let native = Native::ready(Case {
        device_id: HOLD.to_owned(),
        ..Case::replicas()
    })
    .await;
    let (first, on_first) = hold(&native, &body(600, 16, true)).await;
    let (second, on_second) = hold(&native, &body(600, 16, true)).await;
    assert_eq!(
        (on_first.as_str(), on_second.as_str()),
        (WORKER, SECOND_WORKER)
    );
    drop((first, second));
}

/// R6 (D6, D11): admission charges each card's KV pool with the request's reservation (`ceil64(prompt +
/// max_tokens)`). `gpu0` holds one 150K request, `gpu1` two short ones: by rows `gpu0` is the least loaded, but
/// its pool cannot hold a second 150K reservation, so that request goes to `gpu1`, whose pool can.
#[tokio::test]
async fn r6_a_reservation_goes_to_the_card_whose_pool_holds_it() {
    let native = Native::ready(Case {
        device_id: HOLD.to_owned(),
        ..Case::replicas()
    })
    .await;
    let (long, on_long) = hold(&native, &body(150_000, 16, true)).await;
    assert_eq!(on_long, WORKER);
    let (short_a, on_a) = hold(&native, &body(600, 16, true)).await;
    // Rows tie 1-1: the most free pool tokens win.
    let (short_b, on_b) = hold(&native, &body(600, 16, true)).await;
    assert_eq!(
        (on_a.as_str(), on_b.as_str()),
        (SECOND_WORKER, SECOND_WORKER)
    );
    let (second_long, on_second_long) = hold(&native, &body(150_000, 16, true)).await;
    assert_eq!(on_second_long, SECOND_WORKER);
    drop((long, short_a, short_b, second_long));
}

/// A two-message request (`history` tokens, then a last message of `last` tokens): the prefix cache captures the
/// history (m210 design note D4).
fn two_messages(history: usize, last: usize, max_tokens: u64) -> Value {
    let mut h = SENTINEL.to_owned();
    while h.len() < history - 1 {
        h.push('x');
    }
    json!({
        "model": MODEL,
        "messages": [
            {"role": "user", "content": h},
            {"role": "user", "content": format!("1{}", "q".repeat(last - 2))},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    })
}

/// R8 (D11 (2), D12, D13): on a cache-on deployment a repeated prompt goes back to the card that captured its
/// prefix, ahead of least-loaded. The first run lands on `gpu1` (`gpu0` holds a request); once both cards are idle
/// the tie would go to `gpu0`, but the affinity key sends the repeat to `gpu1`, where it hits its prefix. Another
/// scope's identical prompt has another key (its `cache_salt`), so it routes by load only, to `gpu0`.
#[tokio::test]
async fn r8_a_repeated_prefix_returns_to_its_card() {
    let t = Tmp::new("r8-affinity");
    let gate = gate(&t);
    let native = Native::ready(Case {
        // `gpu0` holds a request past its second token while the gate exists; the gate goes once the busy request
        // is cancelled, so `gpu0`'s next request runs to its end whether or not the busy one reached its hold
        // (`gpu1` never holds).
        device_id: format!("{HOLD},hold_gate={};toy", gate.display()),
        prefix_state_slots: Some(16),
        ..Case::replicas()
    })
    .await;
    let (busy, on_busy) = hold(&native, &body(600, 16, true)).await;
    assert_eq!(on_busy, WORKER);
    let request = two_messages(9_000, 100, 4);
    let (status, first) = answer(&native, &request).await;
    assert_eq!(status, 200, "{first}");
    assert_eq!(first["cere"]["worker_id"], SECOND_WORKER, "{first}");
    let busy_rid = rid(&busy);
    drop(busy);
    assert_eq!(native.terminal(&busy_rid).await["state"], "cancelled");
    std::fs::remove_file(&gate).unwrap();
    let (status, repeat) = answer(&native, &request).await;
    assert_eq!(status, 200, "{repeat}");
    assert_eq!(repeat["cere"]["worker_id"], SECOND_WORKER, "{repeat}");
    assert_eq!(
        repeat["usage"]["prompt_tokens_details"]["cached_tokens"],
        json!((9_100 - 1) / 64 * 64),
        "{repeat}"
    );
    let other = http()
        .post(native.cered.url("/v1/chat/completions"))
        .bearer_auth("secret-b")
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(other.status(), 200);
    let other: Value = other.json().await.unwrap();
    assert_eq!(other["cere"]["worker_id"], WORKER, "{other}");
}

/// D12: a constrained request is encoded (and its grammar opened) on the first live replica before its grant; when
/// another card is granted, its grammar is opened again in that card's grammar process, so the masks it runs
/// under are its own. `gpu0` holds a request, the constrained one lands on `gpu1` and commits exactly its script
/// (the stub's `seq:100,7`: id 100, then the stop id 7).
#[tokio::test]
async fn d12_a_grammar_is_reopened_on_the_granted_card() {
    let native = Native::ready(Case {
        device_id: format!("{HOLD},hold_count=1;toy"),
        ..Case::replicas()
    })
    .await;
    let (busy, on_busy) = hold(&native, &body(600, 16, true)).await;
    assert_eq!(on_busy, WORKER);
    let mut request = body(716, 8, false);
    request["regex"] = json!("seq:100,7");
    let (status, got) = answer(&native, &request).await;
    assert_eq!(status, 200, "{got}");
    assert_eq!(got["cere"]["worker_id"], SECOND_WORKER, "{got}");
    assert_eq!(got["choices"][0]["matched_stop"], 7, "{got}");
    assert_eq!(got["usage"]["completion_tokens"], 2, "{got}");
    drop(busy);
}

/// `POST path` with the operator token: status and JSON body.
async fn operator(native: &Native, path: &str) -> (u16, Value) {
    let response = http()
        .post(native.cered.url(path))
        .bearer_auth(OPERATOR_TOKEN)
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap_or(Value::Null))
}

/// `/readyz` once `until` holds of it (bounded: a few probe periods, the load included).
async fn readyz_until(native: &Native, what: &str, until: impl Fn(&Value) -> bool) -> Value {
    let mut last = Value::Null;
    for _ in 0..1500 {
        let response = http()
            .get(native.cered.url("/readyz"))
            .send()
            .await
            .unwrap();
        last = response.json().await.unwrap();
        if until(&last) {
            return last;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("{what}: never reached; last /readyz {last}");
}

/// A gate file a toy holds its sessions on (after `hold_after` tokens) until the test removes it.
fn gate(t: &Tmp) -> std::path::PathBuf {
    let path = t.join("hold.gate");
    std::fs::write(&path, "").unwrap();
    path
}

/// Ends a held stream: removes its gate and reads the stream to its end, which must be a clean `[DONE]`.
async fn finish(gate: &std::path::Path, held: reqwest::Response) {
    std::fs::remove_file(gate).unwrap();
    let text = held.text().await.unwrap();
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
    assert!(!text.contains("\"error\""), "{text}");
}

/// Audit F6: on a multi-replica deployment a request's encode before its grant answers to the request's deadline.
/// With the stub's encode held for 3 s, a request with a 500 ms deadline is `504 deadline_exceeded` long before the
/// encode returns (it used to wait it out, and then queue), and nothing was dispatched; once the encode is quick
/// again the same request is served.
#[tokio::test]
async fn f6_a_deadline_ends_the_encode_before_the_grant() {
    let t = Tmp::new("f6-encode-deadline");
    let slow = t.join("slow-encode");
    let native = Native::ready(Case {
        sidecar_args: vec![
            "--slow-encode-if".to_owned(),
            slow.display().to_string(),
            "3".to_owned(),
        ],
        ..Case::replicas()
    })
    .await;
    std::fs::write(&slow, "").unwrap();
    let started = std::time::Instant::now();
    let response = native
        .chat(&body(600, 8, false))
        .header("x-cere-deadline-ms", "500")
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    let refused: Value = response.json().await.unwrap();
    assert_eq!(status, 504, "{refused}");
    assert_eq!(refused["error"]["code"], "deadline_exceeded", "{refused}");
    assert!(
        started.elapsed() < Duration::from_millis(2500),
        "the request waited for the encode: {:?}",
        started.elapsed()
    );
    std::fs::remove_file(&slow).unwrap();
    let (status, served) = answer(&native, &body(600, 8, false)).await;
    assert_eq!(status, 200, "{served}");
}

/// R10 (D9 (b), D10): a sticky device fault on `gpu1` (every forward after its fourth fails) fails the request
/// there; the next probe re-smokes the card, the re-smoke fails, the executor is released and loaded once more,
/// and `gpu1` is Ready again at epoch + 1 with a new `instance_id`. `gpu0`'s stream, held over the whole fault
/// and reload, completes cleanly at its unchanged epoch.
#[tokio::test]
async fn r10_a_sticky_fault_reloads_the_card_at_a_new_epoch_while_the_other_serves() {
    let t = Tmp::new("r10-fault");
    let gate = gate(&t);
    let native = Native::ready(Case {
        device_id: format!(
            "toy,hold_after=2,hold_ms=120000,hold_gate={};toy,fail_forward=3,fail_sticky=1",
            gate.display()
        ),
        ..Case::replicas()
    })
    .await;
    let (_, before) = native.settled().await;
    let (held, on_held) = hold(&native, &body(600, 16, true)).await;
    assert_eq!(on_held, WORKER);
    let held_rid = rid(&held);
    let (status, got) = answer(&native, &body(600, 16, false)).await;
    assert_ne!(status, 200, "{got}");
    let epoch = |body: &Value, w: &str| body["workers"][w]["epoch"].as_u64().unwrap();
    let b_epoch = epoch(&before, SECOND_WORKER);
    let after = readyz_until(&native, "gpu1 reloaded at epoch + 1", |body| {
        body["workers"][SECOND_WORKER]["ready"] == true && epoch(body, SECOND_WORKER) > b_epoch
    })
    .await;
    assert_eq!(epoch(&after, SECOND_WORKER), b_epoch + 1, "{after}");
    assert_ne!(
        after["workers"][SECOND_WORKER]["instance_id"],
        before["workers"][SECOND_WORKER]["instance_id"],
        "{after}"
    );
    a_unchanged(&before, &after);
    finish(&gate, held).await;
    assert_eq!(native.terminal(&held_rid).await["state"], "completed");
}

/// `gpu0` is still Ready at its first epoch and `instance_id` in `after`.
fn a_unchanged(before: &Value, after: &Value) {
    let a = |body: &Value, key: &str| body["workers"][WORKER][key].clone();
    assert_eq!(a(after, "ready"), true, "{after}");
    assert_eq!(a(after, "epoch"), a(before, "epoch"), "{after}");
    assert_eq!(a(after, "instance_id"), a(before, "instance_id"), "{after}");
}

/// R10 (D9 (b), N4): when the one reload after a sticky fault fails too (here the artifact is gone), `gpu1` stays
/// not ready and an operator `load` is refused 409 `worker_faulted` (only a `cered` restart recovers the card);
/// `gpu0`'s held stream completes and new requests keep landing on `gpu0`.
#[tokio::test]
async fn r10_a_failed_reload_leaves_the_card_faulted_while_the_other_serves() {
    let t = Tmp::new("r10-reload-fails");
    let gate = gate(&t);
    let native = Native::ready(Case {
        device_id: format!(
            "toy,hold_after=2,hold_ms=120000,hold_gate={};toy,fail_forward=3,fail_sticky=1",
            gate.display()
        ),
        ..Case::replicas()
    })
    .await;
    let (_, before) = native.settled().await;
    let (held, on_held) = hold(&native, &body(600, 16, true)).await;
    assert_eq!(on_held, WORKER);
    let held_rid = rid(&held);
    let moved = t.join("artifact-moved");
    std::fs::rename(&native.artifact, &moved).unwrap();
    let (status, got) = answer(&native, &body(600, 16, false)).await;
    assert_ne!(status, 200, "{got}");
    // The fault probe publishes `fault` while it re-smokes, then the failed reload's own problems.
    let failed = readyz_until(&native, "gpu1's reload failed", |body| {
        body["workers"][SECOND_WORKER]["problems"]
            .as_array()
            .is_some_and(|p| !p.is_empty() && p.iter().all(|p| p["field"] != "fault"))
    })
    .await;
    assert_eq!(failed["workers"][SECOND_WORKER]["ready"], false, "{failed}");
    // C8/R12: `/readyz` stays 200 with half the capacity.
    let (status, now) = native.settled().await;
    assert_eq!(status, 200, "{now}");
    assert_eq!(
        now["capacity"],
        capacity(1, rows_max(&native).await),
        "{now}"
    );
    let (status, refused) =
        operator(&native, &format!("/v1/cere/workers/{SECOND_WORKER}/load")).await;
    assert_eq!(status, 409, "{refused}");
    assert_eq!(refused["error"]["code"], "worker_faulted", "{refused}");
    // With the artifact back, later probes still load nothing.
    std::fs::rename(&moved, &native.artifact).unwrap();
    let probed = |body: &Value| body["workers"][SECOND_WORKER]["probed_at_unix_ms"].as_u64();
    let after = readyz_until(&native, "a later probe of gpu1", |body| {
        probed(body) > probed(&failed)
    })
    .await;
    assert_eq!(after["workers"][SECOND_WORKER]["ready"], false, "{after}");
    a_unchanged(&before, &after);
    finish(&gate, held).await;
    assert_eq!(native.terminal(&held_rid).await["state"], "completed");
    let (status, got) = answer(&native, &body_unary()).await;
    assert_eq!(status, 200, "{got}");
    assert_eq!(got["cere"]["worker_id"], WORKER, "{got}");
}

/// R17 (D10): `unload` of `gpu1` needs its drain (409 `worker_not_draining`) and no admitted request on it (409
/// `requests_in_flight`); once drained and idle it answers 202 `unloaded` and `/readyz` reports the replica
/// `unloaded`; `load` answers 202 `loading` and `gpu1` comes back at epoch + 1 with a new `instance_id` (Ready
/// after `undrain`). `gpu0`'s stream, held over the whole cycle, completes cleanly at its unchanged epoch. D22 (3):
/// the startup loads ran in the foreground; the reload of `gpu1` ran at background priority through 2 staging
/// slots, its ALLOCATE on a thread at nice 19 on 2 CPUs (`GET /v1/cere/cards` `load`), and `gpu0`'s load report is
/// unchanged by it.
#[tokio::test]
async fn r17_unload_and_load_return_the_card_at_a_new_epoch_while_the_other_serves() {
    let t = Tmp::new("r17-unload");
    let gate = gate(&t);
    let held_toy = format!(
        "toy,hold_after=2,hold_ms=120000,hold_gate={}",
        gate.display()
    );
    let native = Native::ready(Case {
        device_id: held_toy,
        ..Case::replicas()
    })
    .await;
    let (_, before) = native.settled().await;
    let (status, startup) = cards(&native, Some(OPERATOR_TOKEN)).await;
    assert_eq!(status, 200, "{startup}");
    for worker in [WORKER, SECOND_WORKER] {
        let load = &startup["cards"][worker]["load"];
        assert_eq!(load["background"], false, "{startup}");
        // `cered` inherits the test's nice value and CPUs.
        assert_eq!(load["nice"], own_nice(), "{startup}");
        assert_eq!(load["allocate_nice"], own_nice(), "{startup}");
        assert_eq!(load["allocate_cpus"], own_cpus(), "{startup}");
    }
    let (held, on_held) = hold(&native, &body(600, 16, true)).await;
    assert_eq!(on_held, WORKER);
    let held_rid = rid(&held);
    let (on_b, on_b_worker) = hold(&native, &body(600, 16, true)).await;
    assert_eq!(on_b_worker, SECOND_WORKER);
    let path = |action: &str| format!("/v1/cere/workers/{SECOND_WORKER}/{action}");

    let (status, body) = operator(&native, &path("unload")).await;
    assert_eq!(status, 409, "{body}");
    assert_eq!(body["error"]["code"], "worker_not_draining", "{body}");
    let (status, body) = operator(&native, &path("drain")).await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = operator(&native, &path("unload")).await;
    assert_eq!(status, 409, "{body}");
    assert_eq!(body["error"]["code"], "requests_in_flight", "{body}");
    drop(on_b);
    let mut unloaded = Value::Null;
    for _ in 0..1500 {
        let (status, body) = operator(&native, &path("unload")).await;
        if status == 202 {
            unloaded = body;
            break;
        }
        assert_eq!(body["error"]["code"], "requests_in_flight", "{body}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        unloaded,
        json!({ "worker_id": SECOND_WORKER, "native": "unloaded" })
    );
    let down = readyz_until(&native, "gpu1 reported unloaded", |body| {
        body["workers"][SECOND_WORKER]["problems"]
            .as_array()
            .is_some_and(|p| p.iter().any(|p| p["field"] == "unloaded"))
    })
    .await;
    assert_eq!(down["workers"][SECOND_WORKER]["ready"], false, "{down}");

    let (status, body) = operator(&native, &path("load")).await;
    assert_eq!(status, 202, "{body}");
    assert_eq!(
        body,
        json!({ "worker_id": SECOND_WORKER, "native": "loading" })
    );
    let b_epoch = before["workers"][SECOND_WORKER]["epoch"].as_u64().unwrap();
    let loaded = readyz_until(&native, "gpu1 loaded at epoch + 1", |body| {
        let b = &body["workers"][SECOND_WORKER];
        b["epoch"].as_u64() == Some(b_epoch + 1)
            && b["problems"].as_array().is_some_and(Vec::is_empty)
    })
    .await;
    assert_ne!(
        loaded["workers"][SECOND_WORKER]["instance_id"],
        before["workers"][SECOND_WORKER]["instance_id"],
        "{loaded}"
    );
    let (status, reloaded) = cards(&native, Some(OPERATOR_TOKEN)).await;
    assert_eq!(status, 200, "{reloaded}");
    let load = &reloaded["cards"][SECOND_WORKER]["load"];
    assert_eq!(load["background"], true, "{reloaded}");
    assert_eq!(load["nice"], 19, "{reloaded}");
    assert_eq!(load["allocate_nice"], 19, "{reloaded}");
    assert_eq!(load["allocate_cpus"], own_cpus().min(2), "{reloaded}");
    assert_eq!(load["staging_slots"], 2, "{reloaded}");
    let copies = load["copies_in_flight_max"].as_u64().unwrap();
    assert!((1..=2).contains(&copies), "{reloaded}");
    assert_eq!(
        reloaded["cards"][WORKER]["load"], startup["cards"][WORKER]["load"],
        "{reloaded}"
    );
    let (status, body) = operator(&native, &path("undrain")).await;
    assert_eq!(status, 200, "{body}");
    let after = readyz_until(&native, "gpu1 ready after undrain", |body| {
        body["workers"][SECOND_WORKER]["ready"] == true
    })
    .await;
    assert_eq!(
        after["workers"][SECOND_WORKER]["epoch"],
        b_epoch + 1,
        "{after}"
    );
    a_unchanged(&before, &after);
    finish(&gate, held).await;
    assert_eq!(native.terminal(&held_rid).await["state"], "completed");
}

/// `GET /v1/cere/cards` with the bearer `token` (None: no `Authorization`): status and JSON body.
async fn cards(native: &Native, token: Option<&str>) -> (u16, Value) {
    let mut request = http().get(native.cered.url("/v1/cere/cards"));
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    let response = request.send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap_or(Value::Null))
}

/// The calling thread's nice value (Linux: a per-thread attribute; a spawned `cered` starts with it).
fn own_nice() -> i32 {
    // SAFETY: reads the calling thread's own nice value.
    unsafe { libc::getpriority(libc::PRIO_PROCESS, libc::gettid() as libc::id_t) }
}

/// How many CPUs the calling thread may run on (Linux: a per-thread mask; a spawned `cered` starts with it).
fn own_cpus() -> usize {
    cere_runtime::native::driver::allowed_cpus().unwrap().len()
}

/// Every key of a `GET /v1/cere/cards` entry (D18; `load`: D22 (3)), sorted.
const CARD_KEYS: [&str; 20] = [
    "active_rows",
    "committed_tokens",
    "decode_tokens_per_s",
    "device_uuid",
    "epoch",
    "fifo_waiters",
    "free_mib",
    "kv_pool_tokens",
    "load",
    "loaded",
    "open_sessions",
    "preemptions",
    "prefix_cache",
    "prefix_hits",
    "reserved_pool_tokens",
    "rows_max",
    "spec_accept_length",
    "spec_accepted_drafts",
    "spec_verify_steps",
    "state",
];

fn keys(value: &Value) -> Vec<&str> {
    value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect()
}

/// R12 (D18): the scope view of `GET /v1/cere/stats` gives each card its `device_uuid`, `rows_max`,
/// `kv_pool_tokens` and the caller scope's `requests`, `completion_tokens` and `cached_tokens` there; another
/// scope's request changes nothing in it. The operator's `GET /v1/cere/cards` (401 without a bearer, 403 for a
/// client token) has every D18 field per card: committed tokens and the decode rate since the previous read,
/// prefix hits (a repeated prompt hits its prefix on the card its affinity sends it to) and idle rows, pool,
/// sessions and waiters.
#[tokio::test]
async fn r12_per_card_stats_for_the_scope_and_the_operator() {
    let native = Native::ready(Case {
        prefix_state_slots: Some(16),
        ..Case::replicas()
    })
    .await;
    let scope_cards = |stats: &Value| {
        (
            stats["workers"][WORKER].clone(),
            stats["workers"][SECOND_WORKER].clone(),
        )
    };
    let (status, before) = native.get("/v1/cere/stats").await;
    assert_eq!(status, 200, "{before}");
    let rows = before["workers"][WORKER]["rows_max"].as_u64().unwrap();
    for (worker, uuid) in [(WORKER, DEVICE_UUID), (SECOND_WORKER, SECOND_UUID)] {
        let card = &before["workers"][worker];
        assert_eq!(card["device_uuid"], uuid, "{before}");
        assert_eq!(card["rows_max"], rows, "{before}");
        assert_eq!(card["kv_pool_tokens"], CARD_POOL, "{before}");
        for counter in ["requests", "completion_tokens", "cached_tokens"] {
            assert_eq!(card[counter], 0, "{before}");
        }
    }

    // Another scope's request (on `gpu0`, the tie) changes nothing in this scope's view.
    let other = http()
        .post(native.cered.url("/v1/chat/completions"))
        .bearer_auth("secret-b")
        .json(&body_unary())
        .send()
        .await
        .unwrap();
    assert_eq!(other.status(), 200);
    let other: Value = other.json().await.unwrap();
    assert_eq!(other["cere"]["worker_id"], WORKER, "{other}");
    let (_, after_other) = native.get("/v1/cere/stats").await;
    assert_eq!(
        scope_cards(&after_other),
        scope_cards(&before),
        "{after_other}"
    );

    // This scope's prompt, then its repeat, which hits its prefix on the same card.
    let request = two_messages(9_000, 100, 4);
    let (status, first) = answer(&native, &request).await;
    assert_eq!(status, 200, "{first}");
    let card = first["cere"]["worker_id"].as_str().unwrap().to_owned();
    let (status, repeat) = answer(&native, &request).await;
    assert_eq!(status, 200, "{repeat}");
    assert_eq!(repeat["cere"]["worker_id"], card.as_str(), "{repeat}");
    let cached = repeat["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap();
    assert!(cached > 0, "{repeat}");
    let completion = |r: &Value| r["usage"]["completion_tokens"].as_u64().unwrap();
    let (_, stats) = native.get("/v1/cere/stats").await;
    let mine = &stats["workers"][card.as_str()];
    assert_eq!(mine["requests"], 2, "{stats}");
    assert_eq!(
        mine["completion_tokens"],
        completion(&first) + completion(&repeat),
        "{stats}"
    );
    assert_eq!(mine["cached_tokens"], cached, "{stats}");

    for (token, code) in [(None, 401), (Some(TOKEN), 403)] {
        let (status, body) = cards(&native, token).await;
        assert_eq!(status, code, "{body}");
    }
    let (_, ready) = native.settled().await;
    let (status, view) = cards(&native, Some(OPERATOR_TOKEN)).await;
    assert_eq!(status, 200, "{view}");
    assert_eq!(keys(&view["cards"]), [WORKER, SECOND_WORKER], "{view}");
    for (worker, uuid) in [(WORKER, DEVICE_UUID), (SECOND_WORKER, SECOND_UUID)] {
        let c = &view["cards"][worker];
        assert_eq!(keys(c), CARD_KEYS, "{view}");
        assert_eq!(c["device_uuid"], uuid, "{view}");
        assert_eq!(c["state"], "ready", "{view}");
        assert_eq!(c["epoch"], ready["workers"][worker]["epoch"], "{view}");
        assert_eq!(c["loaded"], true, "{view}");
        assert_eq!(c["rows_max"], rows, "{view}");
        assert_eq!(c["kv_pool_tokens"], CARD_POOL, "{view}");
        for idle in [
            "active_rows",
            "reserved_pool_tokens",
            "open_sessions",
            "fifo_waiters",
            "preemptions",
            "spec_verify_steps",
            "spec_accepted_drafts",
        ] {
            assert_eq!(c[idle], 0, "{idle}: {view}");
        }
        assert_eq!(c["spec_accept_length"], Value::Null, "{view}");
        assert_eq!(c["free_mib"], Value::Null, "{view}");
    }
    let on_card = &view["cards"][card.as_str()];
    assert_eq!(on_card["prefix_hits"], 1, "{view}");
    // The card's prefix cache counters (cere_model_prefix_stats) agree with its hit count; the idle card's are 0.
    assert_eq!(on_card["prefix_cache"]["hits"], 1, "{view}");
    assert!(
        on_card["prefix_cache"]["captures"].as_u64().unwrap() >= 1,
        "{view}"
    );
    assert_eq!(
        on_card["prefix_cache"]["alt_edges_enabled"], false,
        "{view}"
    );
    for (worker, c) in view["cards"].as_object().unwrap() {
        assert!(c["prefix_cache"].is_object(), "{worker}: {view}");
        assert_eq!(
            c["prefix_cache"]["hits"], c["prefix_hits"],
            "{worker}: {view}"
        );
    }
    // Every token this scope's two requests committed on the card (and any of the other scope's there).
    let committed = on_card["committed_tokens"].as_u64().unwrap();
    assert!(
        committed >= completion(&first) + completion(&repeat),
        "{view}"
    );
    assert!(
        on_card["decode_tokens_per_s"].as_f64().unwrap() > 0.0,
        "{view}"
    );
    // Nothing committed since that read: the rate is 0.
    let (_, again) = cards(&native, Some(OPERATOR_TOKEN)).await;
    assert_eq!(
        again["cards"][card.as_str()]["decode_tokens_per_s"],
        0.0,
        "{again}"
    );
    assert_eq!(
        again["cards"][card.as_str()]["committed_tokens"],
        committed,
        "{again}"
    );
}

/// R12, one-GPU unchanged: the single form's `/readyz` has no `capacity`, its scope view of `GET /v1/cere/stats`
/// only the Q11 worker keys, and the operator's `GET /v1/cere/cards` no card.
#[tokio::test]
async fn r12_the_single_form_reports_no_cards() {
    let native = Native::ready(Case::matching()).await;
    let (status, ready) = native.settled().await;
    assert_eq!(status, 200, "{ready}");
    assert_eq!(
        keys(&ready),
        ["problems", "ready", "stopping", "tenancy", "workers"],
        "{ready}"
    );
    let (status, stats) = native.get("/v1/cere/stats").await;
    assert_eq!(status, 200, "{stats}");
    assert_eq!(keys(&stats["workers"]), [WORKER], "{stats}");
    assert_eq!(
        keys(&stats["workers"][WORKER]),
        ["cap", "epoch", "inflight", "state"],
        "{stats}"
    );
    let (status, view) = cards(&native, Some(OPERATOR_TOKEN)).await;
    assert_eq!(status, 200, "{view}");
    assert_eq!(view, json!({ "cards": {} }));
}
