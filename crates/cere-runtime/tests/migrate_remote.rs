//! M4 P3 (m4 design note row P3, Q4-Q7; design-m4 design note §4): session ownership migration with a remote
//! owner, over the real `cered` and the real `cere-worker` of [`World::start_with_local`] (rcgen PKI, shrunk Q7
//! timings, both federation directions through relays a test black-holes to partition the nodes). The
//! coordinator's local worker `gpu1` serves the test model, so the remote `worker-a` and `gpu1` are both Ready
//! and a session can be moved between them. A session's first turn is pinned on `worker-a` by draining `gpu1`
//! for that turn (the live harness's `m_owner` does the same).
//!
//! One world per test (attached agent: no managed mock binary is copied).
mod support;

use std::time::Duration;

use cere_mock_sglang::MockConfig;
use serde_json::{json, Value};
use support::fed_world::*;
use support::relay::Block;
use support::*;

/// The coordinator's local worker.
const LOCAL: &str = "gpu1";
const SUPERSEDED: &str = "session_owner_superseded";
const WORKER_LOST: &str = "worker_lost";
/// Output tokens of a turn that must still be running when the partition and the forced takeover happen: ten
/// minutes at 20 ms per token, far beyond the test.
const LONG_TOKENS: u64 = 30_000;

/// `worker-a`'s backend generates a token every 20 ms with no natural end within any turn's budget (a 4-token
/// turn takes ~80 ms, a [`LONG_TOKENS`] one runs until it is aborted); `gpu1` serves the test model.
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
    World::start_with_local(options, MockConfig::default()).await
}

/// Status and JSON body of a response (a non-JSON body fails with the text; a failed send with the whole error
/// chain, which `reqwest`'s `Display` leaves out).
async fn answer(request: reqwest::RequestBuilder) -> (u16, Value) {
    let r = request.send().await.unwrap_or_else(|e| {
        let mut chain = format!("{e:?}");
        let mut source = std::error::Error::source(&e);
        while let Some(cause) = source {
            chain.push_str(&format!("\n  caused by: {cause:?}"));
            source = cause.source();
        }
        panic!("send failed: {chain}")
    });
    let status = r.status().as_u16();
    let text = r.text().await.unwrap();
    let body = serde_json::from_str(&text)
        .unwrap_or_else(|_| panic!("{status} answered a non-JSON body {text:?}"));
    (status, body)
}

/// `GET /v1/cere/sessions/{id}/owner` with the operator token.
async fn owner(w: &World, session: &str) -> (u16, Value) {
    answer(
        w.http
            .get(w.cered.url(&format!("/v1/cere/sessions/{session}/owner")))
            .bearer_auth(OPERATOR_TOKEN),
    )
    .await
}

/// `POST /v1/cere/sessions/{id}/migrate` with the operator token.
async fn migrate(w: &World, session: &str, body: Value) -> (u16, Value) {
    answer(
        w.http
            .post(w.cered.url(&format!("/v1/cere/sessions/{session}/migrate")))
            .bearer_auth(OPERATOR_TOKEN)
            .json(&body),
    )
    .await
}

fn to(worker: &str, expected: u64, force: bool) -> Value {
    json!({"to_worker": worker, "expected_owner_generation": expected, "force": force})
}

/// The owner read's `200` answer.
fn owned(session: &str, worker: Option<&str>, generation: u64, revision: u64) -> (u16, Value) {
    (
        200,
        json!({"session_id": session, "worker": worker, "owner_generation": generation, "revision": revision}),
    )
}

/// The migrate route's `200` answer.
fn moved(session: &str, worker: &str, generation: u64, revision: u64) -> (u16, Value) {
    (
        200,
        json!({
            "session_id": session,
            "worker": worker,
            "owner_generation": generation,
            "revision": revision,
            "mode": "replay",
        }),
    )
}

#[track_caller]
fn assert_refused(got: &(u16, Value), status: u16, code: &str) {
    let (got_status, body) = got;
    assert_eq!(*got_status, status, "{body}");
    assert_eq!(body["error"]["code"], code, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
}

/// Drains or undrains the local worker through the operator route.
async fn operate_local(w: &World, action: &str) {
    let (status, body) = answer(
        w.http
            .post(w.cered.url(&format!("/v1/cere/workers/{LOCAL}/{action}")))
            .bearer_auth(OPERATOR_TOKEN),
    )
    .await;
    assert_eq!(status, 200, "{action} {LOCAL}: {body}");
}

/// The worker whose backend received `rid`: exactly one of `worker-a`'s and `gpu1`'s.
fn receiver(w: &World, rid: &str) -> &'static str {
    let remote = w.backend_requests(rid);
    let local = w
        .local
        .dispatched_rids()
        .iter()
        .filter(|r| *r == rid)
        .count();
    match (remote, local) {
        (1, 0) => A,
        (0, 1) => LOCAL,
        _ => panic!("{rid}: {remote} backend requests on {A}, {local} on {LOCAL}"),
    }
}

/// A scope-`a` session and its committed history (user and assistant messages alternating).
struct Session {
    id: String,
    messages: Vec<Value>,
    revision: u64,
}

impl Session {
    async fn create(w: &World) -> Session {
        let (status, body) = answer(
            w.http
                .post(w.cered.url("/v1/cere/sessions"))
                .bearer_auth(TOKEN_A),
        )
        .await;
        assert_eq!(status, 201, "{body}");
        Session {
            id: body["session_id"].as_str().unwrap().to_owned(),
            messages: Vec::new(),
            revision: 0,
        }
    }

    /// The next turn at the current revision: the committed history and a new user message.
    fn request(&self, w: &World, max_tokens: u64, stream: bool) -> reqwest::RequestBuilder {
        let mut messages = self.messages.clone();
        messages.push(json!({"role": "user", "content": format!("q{}", self.revision)}));
        w.http
            .post(w.cered.url("/v1/chat/completions"))
            .bearer_auth(TOKEN_A)
            .header("X-Cere-Session-Id", &self.id)
            .header("X-Cere-Expected-Revision", self.revision.to_string())
            .json(&json!({
                "model": MODEL,
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "stream": stream,
            }))
    }

    /// A unary 4-token turn that must complete on `worker` and advance the revision by one: its `cere` object,
    /// the request status (`worker_id`) and the backend that received the rid all name `worker`.
    async fn turn_on(&mut self, w: &World, worker: &str) {
        let (status, reply) = answer(self.request(w, 4, false)).await;
        assert_eq!(status, 200, "{reply}");
        let cere = &reply["cere"];
        let rid = cere["request_id"].as_str().unwrap();
        assert_eq!(cere["worker_id"], worker, "{reply}");
        assert_eq!(cere["revision"], self.revision + 1, "{reply}");
        let state = w.status(rid).await;
        assert_eq!(state["state"], "completed", "{state}");
        assert_eq!(state["worker_id"], worker, "{state}");
        assert_eq!(receiver(w, rid), worker);
        let content = reply["choices"][0]["message"]["content"].clone();
        assert!(content.is_string(), "{reply}");
        self.messages
            .push(json!({"role": "user", "content": format!("q{}", self.revision)}));
        self.messages
            .push(json!({"role": "assistant", "content": content}));
        self.revision += 1;
    }

    /// The first turn, on `worker-a`: `gpu1` is drained for it, so admission grants only `worker-a`, which the
    /// grant pins (owner generation 1); then `gpu1` is undrained and Ready again.
    async fn first_turn_on_a(&mut self, w: &World) {
        assert_eq!(self.revision, 0);
        operate_local(w, "drain").await;
        self.turn_on(w, A).await;
        operate_local(w, "undrain").await;
        w.until_local_ready().await;
        assert_eq!(owner(w, &self.id).await, owned(&self.id, Some(A), 1, 1));
    }

    /// `GET /v1/cere/sessions/{id}` (client view): the revision.
    async fn read_revision(&self, w: &World) -> u64 {
        let (status, body) = answer(
            w.http
                .get(w.cered.url(&format!("/v1/cere/sessions/{}", self.id)))
                .bearer_auth(TOKEN_A),
        )
        .await;
        assert_eq!(status, 200, "{body}");
        body["revision"].as_u64().unwrap()
    }
}

/// P3: a session pinned on the remote `worker-a` (generation 1) migrates to the local `gpu1` (generation 2) and
/// its next turn is served there at revision 2; it migrates back (generation 3) and the next turn is served by
/// `worker-a` at revision 3. Each move keeps the revision; no turn moves the owner again.
#[tokio::test(flavor = "multi_thread")]
async fn p3_a_remote_owner_migrates_to_the_local_worker_and_back() {
    let w = start().await;
    let mut s = Session::create(&w).await;
    assert_eq!(owner(&w, &s.id).await, owned(&s.id, None, 0, 0));
    s.first_turn_on_a(&w).await;

    assert_eq!(
        migrate(&w, &s.id, to(LOCAL, 1, false)).await,
        moved(&s.id, LOCAL, 2, 1)
    );
    assert_eq!(owner(&w, &s.id).await, owned(&s.id, Some(LOCAL), 2, 1));
    s.turn_on(&w, LOCAL).await;
    assert_eq!(s.revision, 2);
    assert_eq!(owner(&w, &s.id).await, owned(&s.id, Some(LOCAL), 2, 2));

    assert_eq!(
        migrate(&w, &s.id, to(A, 2, false)).await,
        moved(&s.id, A, 3, 2)
    );
    s.turn_on(&w, A).await;
    assert_eq!(s.revision, 3);
    assert_eq!(owner(&w, &s.id).await, owned(&s.id, Some(A), 3, 3));
}

/// P3 (Q6): `worker-a` owns the session and runs a long streamed turn when the relays cut it off. Once `/readyz`
/// reads it `suspect` (not eligible, not yet Lost: the turn still holds the session lease), a migration without
/// `force` is `409 session_busy` (never `owner_healthy`: the owner is not eligible), and a forced one to `gpu1`
/// answers 200 at generation 2. The in-flight turn ends without advancing the revision, with reason
/// `session_owner_superseded` or `worker_lost` (whichever ends it); the next turn runs on `gpu1` at the next
/// revision.
#[tokio::test(flavor = "multi_thread")]
async fn p3_a_partitioned_remote_owner_is_force_migrated() {
    let w = start().await;
    let mut s = Session::create(&w).await;
    s.first_turn_on_a(&w).await;

    let r = s.request(&w, LONG_TOKENS, true).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let rid = r.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let mut sse = SseReader::new(r);
    let mut events = vec![sse.next().await.unwrap(), sse.next().await.unwrap()];
    let running = w.status(&rid).await;
    assert_eq!(running["worker_id"], A, "{running}");
    assert_eq!(w.backend_requests(&rid), 1);

    w.to_coordinator.block(Block::Refuse);
    w.to_agent.block(Block::Hold);
    let entry = w
        .until_entry(
            "worker-a not ready",
            Duration::from_millis(LOST_MS) + short_bound(),
            |e| e["state"] != "ready",
        )
        .await;
    // Suspect: the owner is not eligible while its turn still holds the lease (Lost would end the turn first).
    assert_eq!(entry["state"], "suspect", "{entry}");

    let busy = migrate(&w, &s.id, to(LOCAL, 1, false)).await;
    assert_refused(&busy, 409, "session_busy");
    assert_eq!(
        migrate(&w, &s.id, to(LOCAL, 1, true)).await,
        moved(&s.id, LOCAL, 2, 1)
    );

    let rest = tokio::time::timeout(short_bound(), sse.rest())
        .await
        .expect("the superseded turn did not end");
    events.extend(rest);
    let terminal = check_stream(&events);
    let code = terminal["error"]["code"].as_str().unwrap_or_default();
    assert!(code == SUPERSEDED || code == WORKER_LOST, "{terminal}");
    let ended = w.status(&rid).await;
    assert_eq!(ended["reason"], code, "{ended}");
    let state = ended["state"].as_str().unwrap();
    assert!(
        state == "cancelled" || state == "failed",
        "{ended}: the turn must not complete"
    );
    eprintln!("p3 force: the superseded turn ended {state} {code}");
    assert_eq!(s.read_revision(&w).await, 1);
    assert_eq!(owner(&w, &s.id).await, owned(&s.id, Some(LOCAL), 2, 1));

    s.turn_on(&w, LOCAL).await;
    assert_eq!(s.revision, 2);
    assert_eq!(owner(&w, &s.id).await, owned(&s.id, Some(LOCAL), 2, 2));

    w.to_coordinator.heal();
    w.to_agent.heal();
}

/// P3 (Q3): `worker-a` owns the session at generation 3 (pinned, moved to `gpu1` and back). `cered` is killed and
/// started again while the agent cannot reach it: the owner read answers `worker-a` at generation 3 from the
/// journal before `worker-a` registered in the new term. After the heal, once `/readyz` reads `worker-a` Ready,
/// the next turn runs on `worker-a` at the next revision with no re-pin (generation still 3).
#[tokio::test(flavor = "multi_thread")]
async fn p3_a_remote_owner_survives_a_cered_restart() {
    let mut w = start().await;
    let mut s = Session::create(&w).await;
    s.first_turn_on_a(&w).await;
    assert_eq!(
        migrate(&w, &s.id, to(LOCAL, 1, false)).await,
        moved(&s.id, LOCAL, 2, 1)
    );
    assert_eq!(
        migrate(&w, &s.id, to(A, 2, false)).await,
        moved(&s.id, A, 3, 1)
    );

    w.to_coordinator.block(Block::Refuse);
    w.restart_cered().await;
    assert_eq!(owner(&w, &s.id).await, owned(&s.id, Some(A), 3, 1));
    let entry = w.entry().await;
    assert_ne!(entry["state"], "ready", "{entry}");
    w.to_coordinator.heal();

    w.until_ready().await;
    s.turn_on(&w, A).await;
    assert_eq!(s.revision, 2);
    assert_eq!(owner(&w, &s.id).await, owned(&s.id, Some(A), 3, 2));
}
