//! Placement fault matrix (m36 design note P8 cases (a)-(d); design-m36-agent.md §3, design-m36-coord.md §3):
//! the real `cered` and a real managed `cere-worker` that launches the built `cere-mock-sglang` itself
//! ([`World::start_managed`]: rcgen PKI, shrunk Q7 timings, the backend copied into a 0755 [`support::mock_bin`]
//! dir and dropped to uid/gid 65534 when root). The agent registers with model `unloaded`; operator `load` and
//! `unload` (Q9) drive the coordinator's desired residency (Q8), which issues `LoadModel`/`DrainModel`/
//! `UnloadModel` to the agent. The coordinator's local worker `gpu1` serves the test model, so sessions can pin
//! to it before `worker-a` turns Ready and after it leaves.
//!
//! The coordinator runs with the production probe period (`readiness::PROBE_INTERVAL`); the global and per-scope
//! admission caps are raised so held streams on `gpu1` and work on `worker-a` fit side by side.
mod support;

use std::path::PathBuf;
use std::time::Duration;

use cere_mock_sglang::MockConfig;
use cere_runtime::readiness::PROBE_INTERVAL;
use serde_json::{json, Value};
use support::fed_world::*;
use support::mock_bin::group;
use support::*;

/// The coordinator's local worker.
const LOCAL: &str = "gpu1";
/// Output tokens of a remote stream that must still run when `unload` answers: 5 s at 20 ms per token.
const REMOTE_STREAM_TOKENS: u64 = 250;

/// `worker-a`'s launched backend generates a token every 20 ms with no natural end; `gpu1` every 5 ms, so a
/// [`Hold`] runs until it is cancelled. Global admission 6 and 3 per scope (local + remote caps are 4 + 4).
async fn start() -> World {
    let options = Options {
        backend: MockConfig {
            server_info: backend_info(1, 4),
            token_latency: Duration::from_millis(20),
            natural_output_tokens: 100_000,
            ..MockConfig::default()
        },
        ..Options::default()
    };
    let managed = Managed {
        local: MockConfig {
            token_latency: Duration::from_millis(5),
            natural_output_tokens: 1_000_000,
            ..MockConfig::default()
        },
        cered_tweaks: vec![
            ("max_backend_inflight = 2", "max_backend_inflight = 6"),
            ("per_scope_inflight = 2", "per_scope_inflight = 3"),
        ],
    };
    World::start_managed(options, managed).await
}

/// `GET /v1/cere/requests/{rid}` with `token`.
async fn request_status(w: &World, token: &str, rid: &str) -> Value {
    let r = w
        .http
        .get(w.cered.url(&format!("/v1/cere/requests/{rid}")))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    r.json().await.unwrap()
}

/// A scope-`b` stream on `gpu1` that generates until [`Hold::release`].
struct Hold {
    rid: String,
    sse: SseReader,
}

impl Hold {
    /// Starts it and waits for its first output; it must run on `gpu1`.
    async fn on_local(w: &World) -> Hold {
        let mut body = chat_body(true);
        body["max_completion_tokens"] = 30_000.into();
        let r = w
            .http
            .post(w.cered.url("/v1/chat/completions"))
            .bearer_auth(TOKEN_B)
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        let rid = r.headers()["x-cere-request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let mut sse = SseReader::new(r);
        sse.next().await.unwrap();
        sse.next().await.unwrap();
        let status = request_status(w, TOKEN_B, &rid).await;
        assert_eq!(status["worker_id"], LOCAL, "{status}");
        Hold { rid, sse }
    }

    /// Cancels it and reads its stream to the end.
    async fn release(mut self, w: &World) {
        let r = w
            .http
            .post(
                w.cered
                    .url(&format!("/v1/cere/requests/{}/cancel", self.rid)),
            )
            .bearer_auth(TOKEN_B)
            .send()
            .await
            .unwrap();
        assert!(r.status().is_success(), "{}", r.status());
        self.sse.rest().await;
    }
}

/// A scope-`a` session and its committed history (user and assistant messages alternating).
struct Session {
    id: String,
    history: Vec<String>,
}

impl Session {
    async fn new(w: &World) -> Session {
        let r = w
            .http
            .post(w.cered.url("/v1/cere/sessions"))
            .bearer_auth(TOKEN_A)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 201);
        let body: Value = r.json().await.unwrap();
        Session {
            id: body["session_id"].as_str().unwrap().to_owned(),
            history: Vec::new(),
        }
    }

    /// One 4-token turn appending `text`; returns the worker that served it.
    async fn turn(&mut self, w: &World, text: &str) -> String {
        let mut messages: Vec<Value> = self
            .history
            .iter()
            .enumerate()
            .map(|(i, content)| {
                let role = if i % 2 == 0 { "user" } else { "assistant" };
                json!({"role": role, "content": content})
            })
            .collect();
        messages.push(json!({"role": "user", "content": text}));
        let revision = self.history.len() / 2;
        let r = w
            .http
            .post(w.cered.url("/v1/chat/completions"))
            .bearer_auth(TOKEN_A)
            .header("X-Cere-Session-Id", &self.id)
            .header("X-Cere-Expected-Revision", revision.to_string())
            .json(&json!({"model": MODEL, "messages": messages, "max_completion_tokens": 4}))
            .send()
            .await
            .unwrap();
        let status = r.status();
        let reply: Value = r.json().await.unwrap();
        assert_eq!(status, 200, "{reply}");
        let content = reply["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_owned();
        self.history.push(text.to_owned());
        self.history.push(content);
        reply["cere"]["worker_id"].as_str().unwrap().to_owned()
    }
}

/// Operator `load`, then waits until `A` is Ready at the agent's epoch with model `ready` under the coordinator's
/// lock.
async fn load(w: &World) -> Seen {
    let body = w.residency("load").await;
    assert_eq!(body["desired"], "resident", "{body}");
    let seen = w.until_ready().await;
    let entry = w.entry().await;
    assert_eq!(entry["model"]["managed"], true, "{entry}");
    assert_eq!(entry["model"]["desired"], "resident", "{entry}");
    assert_eq!(entry["model"]["state"], "ready", "{entry}");
    assert_eq!(entry["model"]["lock_sha256"], deployment_lock(), "{entry}");
    seen
}

/// The coordinator's `dispatched` records naming `A`.
fn dispatched_to_a(w: &World) -> usize {
    journal(w.cered_state.path())
        .iter()
        .filter(|r| r["type"] == "dispatched" && r["worker"] == A)
        .count()
}

/// The agent's `LoadModel` operations (the coordinator's `residency-<term>-<intent_seq>-<epoch>-load` ids in the
/// `_cere_control` scope) it accepted fresh.
fn load_calls(w: &World) -> usize {
    w.agent
        .events()
        .lines()
        .filter(|l| {
            l.contains("event=accept op=_cere_control/residency-")
                && l.contains("-load ")
                && l.contains(" fresh=true")
        })
        .count()
}

/// The pgid of `backend_child.json`.
fn child_pgid(w: &World) -> i32 {
    let text = std::fs::read_to_string(w.child_file()).unwrap();
    let child: Value = serde_json::from_str(&text).unwrap();
    child["pgid"].as_i64().unwrap() as i32
}

/// `backend_child.json`'s `(pid, start_time_ticks)`: the identity of the launched child.
fn child_identity(w: &World) -> (i32, u64) {
    let text = std::fs::read_to_string(w.child_file()).unwrap();
    let child: Value = serde_json::from_str(&text).unwrap();
    (
        child["pid"].as_i64().unwrap() as i32,
        child["start_time_ticks"].as_u64().unwrap(),
    )
}

/// Backends the agent launched.
fn spawned(w: &World) -> usize {
    w.agent.count("event=backend state=spawned")
}

/// The authority the coordinator presents to `A` now: the agent's epoch, the term and the inventory's model
/// generation on `/readyz` (the agent fences a lifecycle call at another generation `stale_generation`).
async fn current(w: &World) -> Seen {
    let entry = w.entry().await;
    assert_eq!(entry["epoch"], w.agent.epoch, "{entry}");
    Seen {
        epoch: w.agent.epoch,
        term: entry["term"].as_u64().unwrap_or_else(|| panic!("{entry}")),
        generation: entry["inventory"]["model_generation"]
            .as_u64()
            .unwrap_or_else(|| panic!("{entry}")),
    }
}

/// A probe `load` of `lock` presenting `pem`, as op `op` in the lifecycle scope under `seen`'s authority, sent
/// `repeat` times on one connection; returns its lines.
async fn probe_load(
    w: &World,
    pem: &(PathBuf, PathBuf),
    seen: Seen,
    op: &str,
    lock: &str,
    repeat: u64,
) -> Vec<Value> {
    let mut call = vec!["load".to_owned()];
    call.extend(authority_flags(seen, seen.generation));
    call.extend(args(&[
        "--op-id",
        op,
        "--lock",
        lock,
        "--repeat",
        &repeat.to_string(),
    ]));
    let lines = w.probe_agent(pem, call).await;
    assert_eq!(lines.len(), repeat as usize, "{lines:?}");
    lines
}

/// A lock the coordinator never approved.
fn other_lock() -> String {
    let lock = "ab".repeat(32);
    assert_ne!(lock, deployment_lock());
    lock
}

/// P8 (a), MULTI_NODE.md:99 and MODEL_LIFECYCLE.md:116: operator `load` makes the managed worker Ready; while
/// streams are held open on `gpu1`, concurrent new sessions are granted to `worker-a`, and a session pinned to
/// `gpu1` before the load stays there although `worker-a` has fewer requests in flight.
#[tokio::test(flavor = "multi_thread")]
async fn operator_load_routes_new_sessions_to_the_loaded_worker() {
    let w = start().await;
    let entry = w.entry().await;
    assert_ne!(entry["state"], "ready", "{entry}");
    assert_eq!(entry["model"]["state"], "unloaded", "{entry}");

    let mut pinned = Session::new(&w).await;
    assert_eq!(pinned.turn(&w, "before the load").await, LOCAL);
    let holds = [Hold::on_local(&w).await, Hold::on_local(&w).await];

    let seen = load(&w).await;
    assert_eq!(seen.epoch, w.agent.epoch);
    assert_eq!(w.mock_bin().running().len(), 1);

    let (mut one, mut two) = (Session::new(&w).await, Session::new(&w).await);
    let (first, second) = tokio::join!(one.turn(&w, "one"), two.turn(&w, "two"));
    assert_eq!((first.as_str(), second.as_str()), (A, A));

    assert_eq!(pinned.turn(&w, "after the load").await, LOCAL);
    assert_eq!(one.turn(&w, "one again").await, A);
    for hold in holds {
        hold.release(&w).await;
    }
}

/// P8 (b), MODEL_LIFECYCLE.md:28 (`DRAINING` admits no new work but lets active work finish) and :30 (two-phase
/// unload): operator `unload` while two remote streams run; they complete, nothing is dispatched to `worker-a`
/// after the route answers, a session pinned there re-pins to `gpu1` on its next turn, and once the model is
/// `unloaded` no launched process and no `backend_child.json` remain.
#[tokio::test(flavor = "multi_thread")]
async fn unload_lets_remote_streams_finish_and_repins_sessions() {
    let w = start().await;
    let holds = [Hold::on_local(&w).await, Hold::on_local(&w).await];
    load(&w).await;

    let mut session = Session::new(&w).await;
    assert_eq!(session.turn(&w, "on worker-a").await, A);
    let mut streams = Vec::new();
    for _ in 0..2 {
        let (rid, sse, events) = w.stream_chat(REMOTE_STREAM_TOKENS).await;
        let status = w.status(&rid).await;
        assert_eq!(status["worker_id"], A, "{status}");
        streams.push((rid, sse, events));
    }

    let body = w.residency("unload").await;
    assert_eq!(body["desired"], "unloaded", "{body}");
    let granted = dispatched_to_a(&w);
    let entry = w.entry().await;
    assert_ne!(entry["state"], "ready", "{entry}");

    assert_eq!(session.turn(&w, "after the unload").await, LOCAL);
    for (rid, mut sse, events) in streams {
        let after = events[1].data()["cere"]["seq"].as_u64().unwrap();
        let rest = sse.rest().await;
        let terminal = check_stream_tail(&rest, after);
        assert_eq!(
            terminal["choices"][0]["finish_reason"], "length",
            "{terminal}"
        );
        let status = w.status(&rid).await;
        assert_eq!(status["state"], "completed", "{status}");
        assert_eq!(status["worker_id"], A, "{status}");
    }
    assert_eq!(session.turn(&w, "pinned to gpu1 now").await, LOCAL);
    for hold in holds {
        hold.release(&w).await;
    }

    w.until_model("unloaded", ready_bound()).await;
    let bin = w.mock_bin();
    until("no launched backend left", short_bound(), || {
        bin.running().is_empty().then_some(())
    })
    .await;
    assert!(!w.child_file().exists(), "{}", w.child_file().display());
    assert_eq!(dispatched_to_a(&w), granted);
    let events = w.agent.events();
    let drained = events.find("event=model state=draining");
    let unloaded = events.rfind("event=model state=unloaded");
    assert!(
        drained.is_some() && drained < unloaded,
        "draining, then unloaded:\n{events}"
    );
    let entry = w.entry().await;
    assert_ne!(entry["state"], "ready", "{entry}");
    assert_eq!(entry["model"]["desired"], "unloaded", "{entry}");
}

/// P8 (c), MODEL_LIFECYCLE.md:104 (a failed instance is not retried implicitly): a launch that exits at once
/// leaves the model `failed`; the coordinator issues no further `LoadModel` over three probe rounds; the next
/// operator `load` (launch script restored) makes the worker Ready.
#[tokio::test(flavor = "multi_thread")]
async fn a_failed_load_is_not_retried_until_the_next_operator_load() {
    let w = start().await;
    w.break_backend();
    let body = w.residency("load").await;
    assert_eq!(body["intent_seq"], 1, "{body}");
    let entry = w.until_model("failed", ready_bound()).await;
    assert_ne!(entry["state"], "ready", "{entry}");
    assert_eq!(load_calls(&w), 1, "{}", w.agent.events());
    let spawned = w.agent.count("event=backend state=spawned");

    tokio::time::sleep(PROBE_INTERVAL * 3 + Duration::from_secs(1)).await;
    assert_eq!(load_calls(&w), 1, "{}", w.agent.events());
    assert_eq!(w.agent.count("event=backend state=spawned"), spawned);
    let entry = w.entry().await;
    assert_eq!(entry["model"]["state"], "failed", "{entry}");
    assert!(w.mock_bin().running().is_empty());

    w.restore_backend();
    let seen = load(&w).await;
    assert_eq!(seen.epoch, w.agent.epoch);
    assert_eq!(load_calls(&w), 2, "{}", w.agent.events());
    assert_eq!(w.mock_bin().running().len(), 1);
}

/// P8 (d), MODEL_LIFECYCLE.md:105 (worker epochs fence after restarts) and ADVERSARIAL_TEST_PLAN.md:19 (a new
/// model generation after a backend restart): with `desired=resident`, restarting the agent reloads the model
/// at a new epoch and generation, and the old backend's process group is gone.
#[tokio::test(flavor = "multi_thread")]
async fn an_agent_restart_reloads_the_resident_model() {
    let mut w = start().await;
    let before = load(&w).await;
    let old_pgid = child_pgid(&w);
    assert!(!group(old_pgid).is_empty());

    w.restart_agent().await;
    let after = w.until_ready().await;
    assert!(after.epoch > before.epoch, "{before:?} -> {after:?}");
    assert!(
        after.generation > before.generation,
        "{before:?} -> {after:?}"
    );
    let entry = w.entry().await;
    assert_eq!(entry["model"]["state"], "ready", "{entry}");
    assert_eq!(entry["model"]["desired"], "resident", "{entry}");
    assert!(group(old_pgid).is_empty(), "{:?}", group(old_pgid));
    assert_ne!(child_pgid(&w), old_pgid);
    assert_eq!(w.mock_bin().running().len(), 1);
    assert_eq!(load_calls(&w), 1, "{}", w.agent.events());
}

/// P8 (e), ADVERSARIAL_TEST_PLAN.md:18 (restart the coordinator: durable term fence, reconcile before scheduling)
/// and MODEL_LIFECYCLE.md:116 (the owning worker performs the physical load): `cered` SIGKILLed and restarted
/// with `worker-a` resident restores `desired` and `intent_seq` from its journal; `worker-a` turns Ready again
/// at the next term on the same backend, which is not launched a second time and gets no fresh `LoadModel`.
#[tokio::test(flavor = "multi_thread")]
async fn a_coordinator_restart_keeps_the_resident_model() {
    let mut w = start().await;
    let before = load(&w).await;
    let child = child_identity(&w);
    assert_eq!(w.mock_bin().running(), vec![child.0]);

    w.restart_cered().await;
    let after = w.until_ready().await;
    assert!(after.term > before.term, "{before:?} -> {after:?}");
    assert_eq!(after.epoch, before.epoch);
    assert_eq!(after.generation, before.generation);
    let entry = w.entry().await;
    assert_eq!(entry["model"]["managed"], true, "{entry}");
    assert_eq!(entry["model"]["desired"], "resident", "{entry}");
    assert_eq!(entry["model"]["state"], "ready", "{entry}");
    assert_eq!(entry["model"]["lock_sha256"], deployment_lock(), "{entry}");

    assert_eq!(child_identity(&w), child);
    assert_eq!(w.mock_bin().running(), vec![child.0]);
    assert_eq!(spawned(&w), 1, "{}", w.agent.events());
    assert_eq!(load_calls(&w), 1, "{}", w.agent.events());

    // The restored intent_seq is 1: the next operator intent is 2.
    let body = w.residency("unload").await;
    assert_eq!(body["intent_seq"], 2, "{body}");
    assert_eq!(body["desired"], "unloaded", "{body}");
}

/// P8 (f), ADVERSARIAL_TEST_PLAN.md:21 (spoofed principal: deny before model loading) and MODEL_LIFECYCLE.md:16
/// (no silent model change): a coordinator-principal `LoadModel` of a lock the coordinator never approved is
/// refused `lock_not_approved`, and the worker's own principal calling `LoadModel` is refused
/// `PERMISSION_DENIED`; while unloaded nothing is launched, and while resident the recorded child keeps its pid
/// and start time.
#[tokio::test(flavor = "multi_thread")]
async fn forged_load_calls_are_refused_and_launch_nothing() {
    let w = start().await;
    let refuse = |label: &'static str| {
        let w = &w;
        async move {
            let seen = current(w).await;
            let forged = &probe_load(
                w,
                &w.coordinator_pem,
                seen,
                &format!("forged-{label}"),
                &other_lock(),
                1,
            )
            .await[0];
            assert_prompt_refusal(forged, "load", "FAILED_PRECONDITION", "lock_not_approved");
            let own = &probe_load(
                w,
                &w.worker_pem,
                seen,
                &format!("own-{label}"),
                &deployment_lock(),
                1,
            )
            .await[0];
            assert_eq!(own["code"], "PERMISSION_DENIED", "{own}");
            assert_eq!(own["accepted"], false, "{own}");
        }
    };

    refuse("unloaded").await;
    tokio::time::sleep(Duration::from_millis(HEARTBEAT_MS * 4)).await;
    assert_eq!(spawned(&w), 0, "{}", w.agent.events());
    assert!(w.mock_bin().running().is_empty());
    assert!(!w.child_file().exists(), "{}", w.child_file().display());
    let entry = w.entry().await;
    assert_eq!(entry["model"]["state"], "unloaded", "{entry}");

    load(&w).await;
    let child = child_identity(&w);
    refuse("resident").await;
    tokio::time::sleep(Duration::from_millis(HEARTBEAT_MS * 4)).await;
    assert_eq!(child_identity(&w), child);
    assert_eq!(w.mock_bin().running(), vec![child.0]);
    assert_eq!(spawned(&w), 1, "{}", w.agent.events());
    let entry = w.entry().await;
    assert_eq!(entry["state"], "ready", "{entry}");
    assert_eq!(entry["model"]["state"], "ready", "{entry}");
}

/// P8 (g), ADVERSARIAL_TEST_PLAN.md:16 (a dropped load acknowledgement retried: the same operation and ticket, no
/// second physical side effect) and MODEL_LIFECYCLE.md:8 (concurrent loads coalesce into one): while unloaded,
/// two `LoadModel` ops race, each sent twice (duplicate: the second copy answers the first copy's ticket); an op
/// that reaches the agent while the model is `loading` joins the one load and succeeds, one that arrives after it
/// turned `ready` carries a generation the confirmation superseded (m36-spec Q4/Q5) and is refused
/// `stale_generation`; after Ready each accepted call is replayed with its original authority (its receipt dropped)
/// and answers the original receipt. Exactly one backend is ever launched.
#[tokio::test(flavor = "multi_thread")]
async fn duplicate_concurrent_and_replayed_loads_launch_one_child() {
    let w = start().await;
    let seen = current(&w).await;
    let lock = deployment_lock();
    let ops = ["load-one", "load-two"];
    let (one, two) = tokio::join!(
        probe_load(&w, &w.coordinator_pem, seen, ops[0], &lock, 2),
        probe_load(&w, &w.coordinator_pem, seen, ops[1], &lock, 2),
    );
    let mut accepted = Vec::new();
    for (op, lines) in ops.into_iter().zip([&one, &two]) {
        if lines[0]["accepted"] == true {
            let ticket = accepted_ticket(&lines[0], "load");
            assert_eq!(accepted_ticket(&lines[1], "load"), ticket, "{lines:?}");
            accepted.push((op, ticket));
        } else {
            // Its first copy reached the agent after the other op's whole load (PlacementHang F2, host load 34).
            assert!(
                w.agent.count("event=model state=ready") > 0,
                "{}",
                w.agent.events()
            );
            for line in lines {
                assert_prompt_refusal(line, "load", "FAILED_PRECONDITION", "stale_generation");
            }
        }
    }
    assert!(!accepted.is_empty(), "{one:?} {two:?}");

    w.agent
        .until_event("event=model state=ready", ready_bound())
        .await;
    for (op, ticket) in &accepted {
        let replayed = probe_load(&w, &w.coordinator_pem, seen, op, &lock, 1).await;
        assert_eq!(
            &accepted_ticket(&replayed[0], "load"),
            ticket,
            "{replayed:?}"
        );
    }
    for (op, _) in &accepted {
        let status = w.operation(seen, "_cere_control", op).await;
        assert!(
            status.terminal && status.reason_code.is_empty(),
            "{status:?}"
        );
    }

    tokio::time::sleep(Duration::from_millis(HEARTBEAT_MS * 4)).await;
    assert_eq!(spawned(&w), 1, "{}", w.agent.events());
    let fresh = w
        .agent
        .events()
        .lines()
        .filter(|l| l.contains("event=accept op=_cere_control/load-") && l.contains(" fresh=true"))
        .count();
    assert_eq!(fresh, accepted.len(), "{}", w.agent.events());
    assert_eq!(w.mock_bin().running().len(), 1);
}

/// P8 (h), MODEL_LIFECYCLE.md:25 (`LOADING`/`READY` -> `FAILED`) and :104 (a failed instance is not retried
/// implicitly): the launched backend SIGKILLed while Ready with a remote stream in flight leaves the model
/// `failed` and `worker-a` not Ready; the stream ends `backend_protocol_error`; no `LoadModel` follows over three
/// probe rounds, and the next operator `load` makes the worker Ready again.
#[tokio::test(flavor = "multi_thread")]
async fn a_backend_crash_fails_the_model_until_the_operator_loads() {
    let w = start().await;
    let holds = [Hold::on_local(&w).await, Hold::on_local(&w).await];
    load(&w).await;
    let (rid, mut sse, events) = w.stream_chat(REMOTE_STREAM_TOKENS).await;
    let status = w.status(&rid).await;
    assert_eq!(status["worker_id"], A, "{status}");

    let (pid, _) = child_identity(&w);
    // SAFETY: plain kill(2) of the backend this test's agent launched (its pid from backend_child.json).
    assert_eq!(unsafe { libc::kill(pid, libc::SIGKILL) }, 0);

    let after = events.last().unwrap().data()["cere"]["seq"]
        .as_u64()
        .unwrap();
    let rest = sse.rest().await;
    let terminal = check_stream_tail(&rest, after);
    assert_eq!(
        terminal["error"]["code"], "backend_protocol_error",
        "{terminal}"
    );
    let status = w.status(&rid).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], "backend_protocol_error", "{status}");

    let entry = w.until_model("failed", ready_bound()).await;
    assert_ne!(entry["state"], "ready", "{entry}");
    assert_eq!(load_calls(&w), 1, "{}", w.agent.events());
    assert_eq!(spawned(&w), 1, "{}", w.agent.events());

    tokio::time::sleep(PROBE_INTERVAL * 3 + Duration::from_secs(1)).await;
    assert_eq!(load_calls(&w), 1, "{}", w.agent.events());
    assert_eq!(spawned(&w), 1, "{}", w.agent.events());
    assert!(w.mock_bin().running().is_empty());
    let entry = w.entry().await;
    assert_eq!(entry["model"]["state"], "failed", "{entry}");
    assert_ne!(entry["state"], "ready", "{entry}");

    load(&w).await;
    assert_eq!(load_calls(&w), 2, "{}", w.agent.events());
    assert_eq!(spawned(&w), 2, "{}", w.agent.events());
    assert_eq!(w.mock_bin().running().len(), 1);
    for hold in holds {
        hold.release(&w).await;
    }
}
