//! M4 P2 operator session migration (m4 design note Q4-Q7, design-m4 design note §3) over two mock replicas
//! `gpu0`/`gpu1`: `POST /v1/cere/sessions/{id}/migrate` moves a session's durable owner to another Ready worker
//! in replay mode (the destination reconstructs from the next turn's messages), `GET /v1/cere/sessions/{id}/owner`
//! reads it. Refusals follow the Q5 order and change nothing; a forced takeover from an ineligible owner fences
//! the turn that held the lease (`session_owner_superseded`), whether its late completion reaches the commit or
//! its abort lands first.

mod support;

use std::collections::HashMap;
use std::future::Future;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cere_mock_sglang::{Fault, MockConfig, MockHandle, MockSglang, ServerInfo};
use cere_runtime::ledger::DEFAULT_COMPACTION_SLACK;
use serde_json::{json, Value};
use support::*;
use tokio::task::JoinHandle;

/// The mock's reply to a turn with `max_completion_tokens = 4`.
const REPLY: &str = "tok0 tok1 tok2 tok3";
const SUPERSEDED: &str = "session_owner_superseded";
const CACHE_REPORT_TWEAK: (&str, &str) =
    ("enable_cache_report = false", "enable_cache_report = true");
const TTL_TWEAK: (&str, &str) = ("session_ttl_ms = 86400000", "session_ttl_ms = 1000");
/// The `session_ttl_ms` of [`TTL_TWEAK`].
const TTL_MS: u64 = 1_000;

/// Probes every 5 s ([`test_tuning`]): no test here waits on a probe (ineligibility comes from a drain, expiry
/// from the wall clock).
async fn harness(mocks: Vec<MockHandle>) -> Harness {
    Harness::start_replicas_tuned(mocks, &[OPERATOR_TWEAK], test_tuning()).await
}

/// Two mocks generating a token every 5 ms with no natural end: a bounded turn finishes in a few tens of ms, an
/// unbounded one runs until it is aborted (it honours the abort unless the mock's fault says otherwise).
async fn generating_mocks() -> Vec<MockHandle> {
    let mut out = Vec::new();
    for _ in 0..2 {
        let config = MockConfig {
            token_latency: Duration::from_millis(5),
            natural_output_tokens: 1_000_000,
            ..MockConfig::default()
        };
        out.push(MockSglang::start(config).await.unwrap());
    }
    out
}

/// Two mocks reporting prefix-cache hits (`cached_tokens`), for [`CACHE_REPORT_TWEAK`].
async fn reporting_mocks() -> Vec<MockHandle> {
    let mut out = Vec::new();
    for _ in 0..2 {
        let config = MockConfig {
            cache_granularity: 1,
            server_info: ServerInfo {
                enable_cache_report: true,
                ..ServerInfo::default()
            },
            ..MockConfig::default()
        };
        out.push(MockSglang::start(config).await.unwrap());
    }
    out
}

async fn create_session(h: &Harness) -> String {
    let r = h.post("/v1/cere/sessions", TOKEN_A).send().await.unwrap();
    assert_eq!(r.status(), 201);
    let body: Value = r.json().await.unwrap();
    body["session_id"].as_str().unwrap().to_owned()
}

/// The history of a turn at `revision`: every earlier turn was `q<i>` answered [`REPLY`], then `q<revision>`.
fn history(revision: u64) -> Vec<Value> {
    let mut messages = Vec::new();
    for i in 0..revision {
        messages.push(json!({"role": "user", "content": format!("q{i}")}));
        messages.push(json!({"role": "assistant", "content": REPLY}));
    }
    messages.push(json!({"role": "user", "content": format!("q{revision}")}));
    messages
}

fn turn_with(
    h: &Harness,
    session: &str,
    revision: u64,
    max_tokens: u64,
    stream: bool,
) -> reqwest::RequestBuilder {
    h.chat(&json!({
        "model": MODEL,
        "messages": history(revision),
        "max_completion_tokens": max_tokens,
        "stream": stream,
    }))
    .header("X-Cere-Session-Id", session)
    .header("X-Cere-Expected-Revision", revision.to_string())
}

/// A unary turn at `revision` that commits [`REPLY`].
fn turn(h: &Harness, session: &str, revision: u64) -> reqwest::RequestBuilder {
    turn_with(h, session, revision, 4, false)
}

/// Sends a turn that must complete; returns its `cere` object (see [`completed`]).
async fn served(h: &Harness, request: reqwest::RequestBuilder) -> Value {
    completed(h, &answer(request).await)
}

/// A completed turn's answer: returns its `cere` object after checking that exactly the worker it names received
/// the rid.
fn completed(h: &Harness, (status, reply): &(u16, Value)) -> Value {
    assert_eq!(*status, 200, "{reply}");
    let cere = reply["cere"].clone();
    let rid = cere["request_id"].as_str().unwrap().to_owned();
    let worker = cere["worker_id"].as_str().unwrap();
    for i in 0..h.replica_count() {
        let seen = h.replica(i).dispatched_rids().contains(&rid);
        assert_eq!(seen, worker == format!("gpu{i}"), "replica {i}: {reply}");
    }
    cere
}

/// Status and JSON body of a response (a non-JSON body fails with the text).
async fn answer(request: reqwest::RequestBuilder) -> (u16, Value) {
    let r = request.send().await.unwrap();
    let status = r.status().as_u16();
    let text = r.text().await.unwrap();
    let body = serde_json::from_str(&text)
        .unwrap_or_else(|_| panic!("{status} answered a non-JSON body {text:?}"));
    (status, body)
}

fn owner_path(session: &str) -> String {
    format!("/v1/cere/sessions/{session}/owner")
}

fn migrate_path(session: &str) -> String {
    format!("/v1/cere/sessions/{session}/migrate")
}

async fn owner(h: &Harness, session: &str) -> (u16, Value) {
    answer(h.get(&owner_path(session), OPERATOR_TOKEN)).await
}

async fn migrate(h: &Harness, session: &str, body: Value) -> (u16, Value) {
    answer(h.post(&migrate_path(session), OPERATOR_TOKEN).json(&body)).await
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
    let kind = if status >= 500 {
        "server_error"
    } else {
        "invalid_request_error"
    };
    assert_eq!(body["error"]["type"], kind, "{body}");
}

/// Drain or undrain `worker` through the operator route.
async fn operate(h: &Harness, worker: &str, action: &str) {
    let (status, body) = answer(h.post(
        &format!("/v1/cere/workers/{worker}/{action}"),
        OPERATOR_TOKEN,
    ))
    .await;
    assert_eq!(status, 200, "{body}");
}

/// Polls until `done` holds (10 s bound: a hang, not a timing assertion).
async fn until(what: &str, done: impl Fn() -> bool) {
    for _ in 0..2_000 {
        if done() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("timed out waiting for {what}");
}

/// A session turn the mock behind `gpu<worker>` received and may still be generating: its rid and the client call
/// (the status and the whole body text: JSON for unary, the SSE text for a stream). A `stream` turn also waits for
/// the response headers: a stream that ends before `cered` committed its `200` is answered with a plain error
/// status instead (a runner stall between dispatch and the headers made the fence's forced move land there:
/// `409` where the test expects the SSE error event).
struct Running {
    rid: String,
    call: JoinHandle<(u16, String)>,
}

async fn start_turn(
    h: &Harness,
    request: reqwest::RequestBuilder,
    worker: usize,
    stream: bool,
) -> Running {
    let mock = h.replica(worker);
    let before = mock.dispatched_rids().len();
    let (headers_tx, headers_rx) = tokio::sync::oneshot::channel();
    let call = tokio::spawn(async move {
        let r = request.send().await.unwrap();
        let _ = headers_tx.send(());
        (r.status().as_u16(), r.text().await.unwrap())
    });
    until("the turn on its worker", || {
        mock.dispatched_rids().len() > before
    })
    .await;
    if stream {
        tokio::time::timeout(Duration::from_secs(10), headers_rx)
            .await
            .expect("the stream's first event within 10 s")
            .unwrap();
    }
    Running {
        rid: mock.dispatched_rids()[before].clone(),
        call,
    }
}

/// P2 happy path: the first turn pins `gpu0` (owner generation 1); a migration to `gpu1` answers generation 2 in
/// replay mode with the revision unchanged; the next turn runs on `gpu1` with no cache hit and continues the
/// revision; the move is journaled once (`session_owner`) and the turn after it does not move the owner again.
#[tokio::test(flavor = "multi_thread")]
async fn p2_migrate_moves_the_owner_and_the_next_turn_replays_on_the_destination() {
    let h = Harness::start_replicas_tuned(
        reporting_mocks().await,
        &[OPERATOR_TWEAK, CACHE_REPORT_TWEAK],
        test_tuning(),
    )
    .await;
    let sid = create_session(&h).await;
    assert_eq!(owner(&h, &sid).await, owned(&sid, None, 0, 0));

    let first = served(&h, turn(&h, &sid, 0)).await;
    assert_eq!(first["worker_id"], "gpu0", "{first}");
    assert_eq!(first["revision"], 1, "{first}");
    assert_eq!(owner(&h, &sid).await, owned(&sid, Some("gpu0"), 1, 1));

    let reply = migrate(
        &h,
        &sid,
        json!({"to_worker": "gpu1", "expected_owner_generation": 1}),
    )
    .await;
    assert_eq!(reply, moved(&sid, "gpu1", 2, 1));
    assert_eq!(owner(&h, &sid).await, owned(&sid, Some("gpu1"), 2, 1));

    let next = served(&h, turn(&h, &sid, 1)).await;
    assert_eq!(next["worker_id"], "gpu1", "{next}");
    assert_eq!(next["revision"], 2, "{next}");
    assert_eq!(next["cache_report"], "reported", "{next}");
    assert_eq!(next["cached_tokens"], 0, "{next}");
    assert_eq!(owner(&h, &sid).await, owned(&sid, Some("gpu1"), 2, 2));

    let records = journal(h.state_dir.path());
    let moves: Vec<&Value> = records
        .iter()
        .filter(|r| r["type"] == "session_owner")
        .collect();
    assert_eq!(moves.len(), 1, "{moves:?}");
    let record = moves[0];
    assert_eq!(
        (
            &record["session_id"],
            &record["worker"],
            &record["owner_generation"],
            &record["revision"]
        ),
        (&json!(sid), &json!("gpu1"), &json!(2), &json!(1)),
        "{record}"
    );
    let next_rid = next["request_id"].as_str().unwrap();
    let dispatched = records
        .iter()
        .find(|r| r["type"] == "dispatched" && r["rid"] == next_rid)
        .unwrap();
    assert!(dispatched.get("session_owner").is_none(), "{dispatched}");
}

/// The routes' auth model and the migrate body contract: without an operator token they do not exist; a missing
/// or unknown bearer is 401, a client token 403 (before the body is read); each malformed body is a 400 with its
/// code and `param`. Nothing changes.
#[tokio::test(flavor = "multi_thread")]
async fn p2_auth_and_body_refusals_change_nothing() {
    let disabled = Harness::start(MockConfig::default(), &[]).await;
    let sid = create_session(&disabled).await;
    for token in [OPERATOR_TOKEN, TOKEN_A] {
        let got = answer(disabled.get(&owner_path(&sid), token)).await;
        assert_refused(&got, 404, "not_found");
        let got = answer(
            disabled
                .post(&migrate_path(&sid), token)
                .json(&to("gpu1", 0, false)),
        )
        .await;
        assert_refused(&got, 404, "not_found");
    }

    let h = harness(mocks(2).await).await;
    let sid = create_session(&h).await;
    let body = to("gpu1", 0, false);
    let unauthenticated = [
        h.http.get(h.url(&owner_path(&sid))),
        h.http.post(h.url(&migrate_path(&sid))).json(&body),
        h.get(&owner_path(&sid), "not-a-token"),
        h.post(&migrate_path(&sid), "not-a-token").json(&body),
        // Auth runs before the body is read.
        h.post(&migrate_path(&sid), "not-a-token").body("not json"),
    ];
    for request in unauthenticated {
        let r = request.send().await.unwrap();
        assert_eq!(r.status(), 401);
        assert_eq!(r.headers()["www-authenticate"], "Bearer");
        let body: Value = r.json().await.unwrap();
        assert_eq!(body["error"]["code"], "unauthorized", "{body}");
    }
    for request in [
        h.get(&owner_path(&sid), TOKEN_A),
        h.post(&migrate_path(&sid), TOKEN_A).json(&body),
        h.post(&migrate_path(&sid), TOKEN_B).body("not json"),
    ] {
        assert_refused(&answer(request).await, 403, "forbidden");
    }

    let bad: [(&str, &str, Option<&str>); 14] = [
        ("", "invalid_json", None),
        ("not json", "invalid_json", None),
        ("[1]", "invalid_json", None),
        ("\"gpu1\"", "invalid_json", None),
        ("{}", "missing_required_parameter", Some("to_worker")),
        (
            r#"{"expected_owner_generation": 0}"#,
            "missing_required_parameter",
            Some("to_worker"),
        ),
        (
            r#"{"to_worker": "gpu1"}"#,
            "missing_required_parameter",
            Some("expected_owner_generation"),
        ),
        (
            r#"{"to_worker": "gpu1", "expected_owner_generation": 0, "extra": 1}"#,
            "unsupported_parameter",
            Some("extra"),
        ),
        (
            r#"{"to_worker": 1, "expected_owner_generation": 0}"#,
            "invalid_value",
            Some("to_worker"),
        ),
        (
            r#"{"to_worker": null, "expected_owner_generation": 0}"#,
            "invalid_value",
            Some("to_worker"),
        ),
        (
            r#"{"to_worker": "gpu1", "expected_owner_generation": -1}"#,
            "invalid_value",
            Some("expected_owner_generation"),
        ),
        (
            r#"{"to_worker": "gpu1", "expected_owner_generation": 1.5}"#,
            "invalid_value",
            Some("expected_owner_generation"),
        ),
        (
            r#"{"to_worker": "gpu1", "expected_owner_generation": "0"}"#,
            "invalid_value",
            Some("expected_owner_generation"),
        ),
        (
            r#"{"to_worker": "gpu1", "expected_owner_generation": 0, "force": "yes"}"#,
            "invalid_value",
            Some("force"),
        ),
    ];
    for (raw, code, param) in bad {
        let request = h
            .post(&migrate_path(&sid), OPERATOR_TOKEN)
            .header("content-type", "application/json")
            .body(raw);
        let got = answer(request).await;
        assert_refused(&got, 400, code);
        assert_eq!(got.1["error"]["param"], json!(param), "{raw}: {}", got.1);
    }
    let null_force = r#"{"to_worker": "gpu1", "expected_owner_generation": 0, "force": null}"#;
    let got = answer(
        h.post(&migrate_path(&sid), OPERATOR_TOKEN)
            .header("content-type", "application/json")
            .body(null_force),
    )
    .await;
    assert_refused(&got, 400, "invalid_value");
    assert_eq!(owner(&h, &sid).await, owned(&sid, None, 0, 0));
    assert!(
        journal(h.state_dir.path())
            .iter()
            .all(|r| r["type"] != "session_owner"),
        "a refusal journals nothing"
    );
}

/// Every Q5 refusal, in order, with the owner unchanged after each: unknown session; a stale generation (409
/// `owner_generation_conflict` naming the current owner inside `error`, also before any owner exists); unknown
/// destination; a drained destination; the current owner; a held lease without `force`; `force` against an
/// eligible owner. A CAS retry after a success reads the move.
#[tokio::test(flavor = "multi_thread")]
async fn p2_refusals_follow_the_q5_order_and_change_nothing() {
    let h = harness(generating_mocks().await).await;
    let sid = create_session(&h).await;

    let unknown = "sess_00000000000000000000000000000000";
    assert_refused(&owner(&h, unknown).await, 404, "session_not_found");
    assert_refused(
        &migrate(&h, unknown, to("gpu9", 0, true)).await,
        404,
        "session_not_found",
    );

    let conflict = migrate(&h, &sid, to("gpu1", 1, false)).await;
    assert_refused(&conflict, 409, "owner_generation_conflict");
    assert_eq!(conflict.1["error"]["worker"], Value::Null, "{}", conflict.1);
    assert_eq!(conflict.1["error"]["owner_generation"], 0, "{}", conflict.1);
    assert_eq!(owner(&h, &sid).await, owned(&sid, None, 0, 0));

    let first = served(&h, turn(&h, &sid, 0)).await;
    assert_eq!(first["worker_id"], "gpu0", "{first}");
    let unchanged = owned(&sid, Some("gpu0"), 1, 1);
    assert_eq!(owner(&h, &sid).await, unchanged);

    // The generation is checked before the destination.
    let conflict = migrate(&h, &sid, to("gpu9", 0, false)).await;
    assert_refused(&conflict, 409, "owner_generation_conflict");
    assert_eq!(conflict.1["error"]["worker"], "gpu0", "{}", conflict.1);
    assert_eq!(conflict.1["error"]["owner_generation"], 1, "{}", conflict.1);
    assert_eq!(owner(&h, &sid).await, unchanged);

    assert_refused(
        &migrate(&h, &sid, to("gpu9", 1, false)).await,
        404,
        "worker_not_found",
    );
    assert_eq!(owner(&h, &sid).await, unchanged);

    operate(&h, "gpu1", "drain").await;
    assert_refused(
        &migrate(&h, &sid, to("gpu1", 1, true)).await,
        409,
        "worker_not_eligible",
    );
    assert_eq!(owner(&h, &sid).await, unchanged);
    operate(&h, "gpu1", "undrain").await;

    assert_refused(
        &migrate(&h, &sid, to("gpu0", 1, false)).await,
        409,
        "already_owner",
    );
    assert_eq!(owner(&h, &sid).await, unchanged);

    // A turn holds the lease on gpu0.
    let held = start_turn(&h, turn_with(&h, &sid, 1, 30_000, false), 0, false).await;
    assert_refused(
        &migrate(&h, &sid, to("gpu1", 1, false)).await,
        409,
        "session_busy",
    );
    assert_eq!(owner(&h, &sid).await, unchanged);
    // Force needs an owner that is not eligible, whether or not a turn holds the lease.
    assert_refused(
        &migrate(&h, &sid, to("gpu1", 1, true)).await,
        409,
        "owner_healthy",
    );
    assert_eq!(owner(&h, &sid).await, unchanged);
    let cancel = h
        .post(&format!("/v1/cere/requests/{}/cancel", held.rid), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert!(cancel.status().is_success(), "{}", cancel.status());
    let (status, _) = held.call.await.unwrap();
    assert_eq!(status, 409);
    assert_refused(
        &migrate(&h, &sid, to("gpu1", 1, true)).await,
        409,
        "owner_healthy",
    );
    assert_eq!(owner(&h, &sid).await, unchanged);

    let request = to("gpu1", 1, false);
    assert_eq!(
        migrate(&h, &sid, request.clone()).await,
        moved(&sid, "gpu1", 2, 1)
    );
    // The operator's retry of the same CAS reads the move.
    let retry = migrate(&h, &sid, request).await;
    assert_refused(&retry, 409, "owner_generation_conflict");
    assert_eq!(retry.1["error"]["worker"], "gpu1", "{}", retry.1);
    assert_eq!(retry.1["error"]["owner_generation"], 2, "{}", retry.1);
    assert_eq!(owner(&h, &sid).await, owned(&sid, Some("gpu1"), 2, 1));
    let moves = journal(h.state_dir.path())
        .iter()
        .filter(|r| r["type"] == "session_owner")
        .count();
    assert_eq!(moves, 1, "only the success is journaled");
}

/// Both routes answer `503 shutting_down` once the process is stopping, before auth.
#[tokio::test(flavor = "multi_thread")]
async fn p2_routes_answer_shutting_down_while_stopping() {
    let h = harness(mocks(2).await).await;
    let sid = create_session(&h).await;
    // The L8 shutdown marks the process stopping; the HTTP server keeps serving until `serve` stops it.
    assert!(
        h.app.shutdown().await,
        "nothing in flight: a clean shutdown"
    );
    for request in [
        h.get(&owner_path(&sid), OPERATOR_TOKEN),
        h.post(&migrate_path(&sid), OPERATOR_TOKEN)
            .json(&to("gpu1", 0, false)),
        h.http.post(h.url(&migrate_path(&sid))).body("not json"),
    ] {
        assert_refused(&answer(request).await, 503, "shutting_down");
    }
}

/// `cered`'s clock (`app::unix_ms`): a session expires once it reads `updated_ms + session_ttl_ms` or later.
fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Sleeps until [`unix_ms`] reads `at` or later.
async fn sleep_until_unix(at: u64) {
    loop {
        let now = unix_ms();
        if now >= at {
            return;
        }
        tokio::time::sleep(Duration::from_millis(at - now)).await;
    }
}

/// The instant `cered` stamped `session`'s TTL with in its `kind` journal record: `session` (the creation's
/// `updated_ms`) or `session_owner` (the move's `at_ms`).
fn stamped(h: &Harness, session: &str, kind: &str) -> u64 {
    let field = if kind == "session" {
        "updated_ms"
    } else {
        "at_ms"
    };
    let records = journal(h.state_dir.path());
    let record = records
        .iter()
        .find(|r| r["type"] == kind && r["session_id"] == session)
        .unwrap_or_else(|| panic!("no {kind} record of {session}: {records:?}"));
    record[field].as_u64().unwrap()
}

/// An answer of `cered` and the wall-clock interval of its call: `cered` read its clock for it at or after `sent`
/// and at or before `answered` (unix ms).
struct Timed {
    got: (u16, Value),
    sent: u64,
    answered: u64,
}

async fn timed(call: impl Future<Output = (u16, Value)>) -> Timed {
    let sent = unix_ms();
    let got = call.await;
    Timed {
        got,
        sent,
        answered: unix_ms(),
    }
}

impl Timed {
    /// Whether the session the call named was alive for it, its TTL ending at `expiry` (unix ms). Answered before
    /// the expiry, it was; sent at or after it, it was not, and the answer is `404 session_not_found` (checked
    /// here). A call that straddles the expiry (a runner stall) tells by its answer.
    fn alive(&self, expiry: u64) -> bool {
        let (status, body) = &self.got;
        if self.answered < expiry || (self.sent < expiry && *status != 404) {
            return true;
        }
        let (sent, answered) = (self.sent, self.answered);
        assert_eq!(
            *status, 404,
            "sent {sent}, answered {answered}, expiry {expiry}: {body}"
        );
        assert_refused(&self.got, 404, "session_not_found");
        false
    }
}

/// Trials before [`p2_expired_session_is_refused_and_a_force_migrate_refreshes_the_ttl`] gives up: a runner stall
/// that lands the move or a call naming `kept` past its TTL voids a trial (the next one starts over with new
/// sessions).
const TTL_TRIALS: u32 = 5;

/// An expired session is refused (`404 session_not_found`, with or without `force`, and on the owner read);
/// a migration refreshes the session's TTL from the instant it applies: a session moved after its creation is
/// still alive when the creation-based TTL has passed, and takes its next turn on the destination. Expiry is
/// judged on the instants `cered` stamped (journal) against the wall-clock interval of each call, never on the
/// runner keeping a schedule (CI 3a09d7c2: a fixed 1.1 s schedule left the calls after it 400 ms before `kept`'s
/// refreshed TTL ended; a runner stall past that got the read `404`, correctly).
#[tokio::test(flavor = "multi_thread")]
async fn p2_expired_session_is_refused_and_a_force_migrate_refreshes_the_ttl() {
    let h =
        Harness::start_replicas_tuned(mocks(2).await, &[OPERATOR_TWEAK, TTL_TWEAK], test_tuning())
            .await;
    for _ in 0..TTL_TRIALS {
        if ttl_trial(&h).await {
            return;
        }
    }
    panic!("every one of {TTL_TRIALS} trials stalled past kept's TTL");
}

/// One trial of the TTL test: `stale` and `kept` are created and `kept` is force-moved to `gpu1` halfway
/// through its TTL; once both creation-based TTLs have passed, `kept` is still owned by `gpu1` and takes its
/// next turn there, and `stale` is refused. False when a stall voided it (the move or a call naming `kept` landed
/// past its TTL).
async fn ttl_trial(h: &Harness) -> bool {
    let stale = create_session(h).await;
    let kept = create_session(h).await;
    let born = stamped(h, &kept, "session");
    let created = born.max(stamped(h, &stale, "session"));

    sleep_until_unix(created + TTL_MS / 2).await;
    // No owner yet counts as not eligible: force is allowed.
    let moving = timed(migrate(h, &kept, to("gpu1", 0, true))).await;
    if !moving.alive(born + TTL_MS) {
        return false;
    }
    assert_eq!(moving.got, moved(&kept, "gpu1", 1, 0));
    // The move stamps the TTL when it applies, not at the creation.
    let at = stamped(h, &kept, "session_owner");
    assert!(
        (moving.sent..=moving.answered).contains(&at),
        "moved at {at}, called in [{}, {}]",
        moving.sent,
        moving.answered
    );

    // Past both creation-based TTLs: `kept` lives on until `at + TTL_MS`.
    sleep_until_unix(created + TTL_MS).await;
    let read = timed(owner(h, &kept)).await;
    if !read.alive(at + TTL_MS) {
        return false;
    }
    assert_eq!(read.got, owned(&kept, Some("gpu1"), 1, 0));
    let next = timed(answer(turn(h, &kept, 0))).await;
    if !next.alive(at + TTL_MS) {
        return false;
    }
    let next = completed(h, &next.got);
    assert_eq!(next["worker_id"], "gpu1", "{next}");
    assert_eq!(next["revision"], 1, "{next}");

    assert_refused(&owner(h, &stale).await, 404, "session_not_found");
    for force in [false, true] {
        assert_refused(
            &migrate(h, &stale, to("gpu1", 0, force)).await,
            404,
            "session_not_found",
        );
    }
    true
}

/// M4-00: the owner a turn targets is read on the guard that takes its lease, never from its earlier session
/// view. A migration racing each turn either commits before the turn's accept (the turn runs on the new owner,
/// with no re-pin) or is refused `session_busy` while the turn holds the lease; it never lands inside a turn,
/// and no turn runs on an owner other than the one current at its `admitted` record. Checked over the journal
/// for every interleaving the races produce.
#[tokio::test(flavor = "multi_thread")]
async fn p2_m4_00_a_migrate_racing_a_turn_is_honoured_at_accept() {
    const RACES: u64 = 16;
    let h = harness(mocks(2).await).await;
    let sid = create_session(&h).await;
    served(&h, turn(&h, &sid, 0)).await;
    let mut accepted_moves = 0;
    for revision in 1..=RACES {
        let (status, before) = owner(&h, &sid).await;
        assert_eq!(status, 200, "{before}");
        let from = before["worker"].as_str().unwrap();
        let generation = before["owner_generation"].as_u64().unwrap();
        let other = if from == "gpu0" { "gpu1" } else { "gpu0" };
        let (cere, migrated) = tokio::join!(
            served(&h, turn(&h, &sid, revision)),
            migrate(&h, &sid, to(other, generation, false)),
        );
        assert_eq!(cere["revision"], revision + 1, "{cere}");
        match migrated.0 {
            200 => accepted_moves += 1,
            _ => assert_refused(&migrated, 409, "session_busy"),
        }
    }

    let records = journal(h.state_dir.path());
    let mut current: Option<String> = None;
    let mut holder: Option<String> = None;
    let mut targets: HashMap<String, Option<String>> = HashMap::new();
    let mut moves = 0;
    for r in &records {
        let rid = r["rid"].as_str().unwrap_or_default().to_owned();
        match r["type"].as_str().unwrap() {
            "admitted" if r["session"]["session_id"] == sid.as_str() => {
                assert_eq!(holder, None, "{r}");
                targets.insert(rid.clone(), current.clone());
                holder = Some(rid);
            }
            "dispatched" if targets.contains_key(&rid) => {
                let worker = r["worker"].as_str().unwrap().to_owned();
                match &targets[&rid] {
                    Some(owner) => {
                        assert_eq!(
                            &worker, owner,
                            "the turn targets its accept-time owner: {r}"
                        );
                        assert!(r.get("session_owner").is_none(), "no re-pin: {r}");
                    }
                    None => {
                        assert_eq!(r["session_owner"]["owner_generation"], 1, "{r}");
                        current = Some(worker);
                    }
                }
            }
            "terminal" if holder.as_deref() == Some(rid.as_str()) => {
                assert_eq!(r["state"], "completed", "{r}");
                holder = None;
            }
            "session_owner" if r["session_id"] == sid.as_str() => {
                assert_eq!(holder, None, "a migration committed inside a turn: {r}");
                current = Some(r["worker"].as_str().unwrap().to_owned());
                moves += 1;
            }
            _ => {}
        }
    }
    assert_eq!(targets.len() as u64, RACES + 1);
    assert_eq!(moves, accepted_moves);
    let (_, last) = owner(&h, &sid).await;
    assert_eq!(last["worker"], json!(current), "{last}");
    assert_eq!(last["owner_generation"], 1 + accepted_moves, "{last}");
}

/// The fence (Q6, Q7): owner `gpu0` is drained (ineligible) while its turn runs; a forced move to `gpu1` breaks
/// the lease and cancels the turn. `late`: the mock ignores the abort and the completion reaches the commit,
/// which refuses it (`failed`); otherwise the abort lands first (`cancelled`). Either way the client sees `409
/// invalid_request_error session_owner_superseded`, the revision does not move, and the next turn runs on
/// `gpu1` at the next revision. The late completion (released once the move is checked) lands inside
/// [`test_tuning`]'s long cancel bound, so the request ends at its commit rather than `cancel_unacknowledged`.
async fn fence(stream: bool, late: bool) {
    let h = harness(generating_mocks().await).await;
    let sid = create_session(&h).await;
    let first = served(&h, turn(&h, &sid, 0)).await;
    assert_eq!(first["worker_id"], "gpu0", "{first}");
    assert_eq!(owner(&h, &sid).await, owned(&sid, Some("gpu0"), 1, 1));

    if late {
        // The mock ignores the abort and holds the turn's finish until the move is checked, so its completion
        // reaches the commit after the move however long the drain and the move take (it was 2 s of generation:
        // a runner stall past it committed the turn first). The abort is sent from its own task and lands late
        // (a slow control plane, as on a loaded host: it reached gpu0 after the terminal).
        h.replica(0).set_fault(Fault::IgnoreAbort);
        h.replica(0).set_abort_delay(Duration::from_millis(300));
        h.replica(0).hold_finish(true);
    }
    let max_tokens = if late { 4 } else { 30_000 };
    let held = start_turn(&h, turn_with(&h, &sid, 1, max_tokens, stream), 0, stream).await;
    operate(&h, "gpu0", "drain").await;
    assert_eq!(
        migrate(&h, &sid, to("gpu1", 1, true)).await,
        moved(&sid, "gpu1", 2, 1)
    );
    let aborted = |h: &Harness| {
        h.replica(0)
            .abort_requests()
            .iter()
            .any(|a| a.rid == held.rid)
    };
    if late {
        assert_eq!(
            h.replica(0).in_flight(),
            1,
            "the held turn is still live after the move"
        );
        // The abort runs on its own task, ordered with nothing the client sees: it must reach gpu0 while the
        // turn is still held, before its completion can reach cered (which then has no abort left to send).
        until("the move aborted the turn on gpu0", || aborted(&h)).await;
        h.replica(0).hold_finish(false);
    }

    let state = if late { "failed" } else { "cancelled" };
    let (status, text) = held.call.await.unwrap();
    let (error, cere) = if stream {
        assert_eq!(status, 200, "{text}");
        let events = parse_events(&text);
        assert!(matches!(events.last(), Some(Event::Done)), "{text}");
        let last = events[events.len() - 2].data().clone();
        (last["error"].clone(), last["cere"].clone())
    } else {
        assert_eq!(status, 409, "{text}");
        let body: Value = serde_json::from_str(&text).unwrap();
        (body["error"].clone(), body["cere"].clone())
    };
    assert_eq!(error["code"], SUPERSEDED, "{text}");
    assert_eq!(error["type"], "invalid_request_error", "{text}");
    assert_eq!(
        (&cere["state"], &cere["reason"]),
        (&json!(state), &json!(SUPERSEDED)),
        "{text}"
    );

    let terminal = h.terminal(&held.rid).await;
    assert_eq!(terminal["state"], state, "{terminal}");
    assert_eq!(terminal["reason"], SUPERSEDED, "{terminal}");
    assert_eq!(terminal["worker_id"], "gpu0", "{terminal}");
    assert!(aborted(&h), "the move aborted the turn on gpu0");
    assert_eq!(owner(&h, &sid).await, owned(&sid, Some("gpu1"), 2, 1));

    let next = served(&h, turn(&h, &sid, 1)).await;
    assert_eq!(next["worker_id"], "gpu1", "{next}");
    assert_eq!(next["revision"], 2, "{next}");
    assert_eq!(owner(&h, &sid).await, owned(&sid, Some("gpu1"), 2, 2));
}

#[tokio::test(flavor = "multi_thread")]
async fn p2_fence_late_unary_completion_fails_superseded() {
    fence(false, true).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn p2_fence_late_stream_completion_fails_superseded() {
    fence(true, true).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn p2_fence_unary_abort_first_cancels_superseded() {
    fence(false, false).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn p2_fence_stream_abort_first_cancels_superseded() {
    fence(true, false).await;
}

fn journal_lines(h: &Harness) -> u64 {
    std::fs::read_to_string(h.state_dir.path().join("journal.jsonl"))
        .unwrap()
        .lines()
        .count() as u64
}

/// Q5 (7), F9 over HTTP: the `session_owner` record is the write that needs a compaction, and the compaction
/// fails (its temp path is a directory): the migrate answers `500 server_error internal_error`, the journal and
/// the owner are unchanged and `/readyz` does not report `journal_poisoned`. Once the fault is removed, the next
/// turn runs on the old owner at the next revision and a migrate with the same expected generation applies.
#[tokio::test(flavor = "multi_thread")]
async fn p2_a_migrate_journal_failure_answers_500_and_changes_nothing() {
    let h = harness(mocks(2).await).await;
    // The open compacted the journal to these lines and nothing was appended since (fresh state dir).
    let compacted = journal_lines(&h);
    // A write compacts first once the file has more than `2 × compacted + slack` lines.
    let full = 2 * compacted + DEFAULT_COMPACTION_SLACK;
    let sid = create_session(&h).await;
    let first = served(&h, turn(&h, &sid, 0)).await;
    assert_eq!(first["worker_id"], "gpu0", "{first}");
    h.terminal(first["request_id"].as_str().unwrap()).await;
    assert_eq!(owner(&h, &sid).await, owned(&sid, Some("gpu0"), 1, 1));
    // Fill to `full + 1` lines, so the next write compacts first: a unary request journals `admitted`,
    // `dispatched` and its terminal; a session create journals one record (the scope's session cap bounds how
    // many creates can fill).
    let start = journal_lines(&h);
    let requests = (full + 1 - start) / 3;
    for _ in 0..requests {
        let r = h.chat(&chat_body(false)).send().await.unwrap();
        let status = r.status();
        assert_eq!(status, 200, "{}", r.text().await.unwrap());
    }
    until("every filler terminal journaled", || {
        journal_lines(&h) == start + 3 * requests
    })
    .await;
    while journal_lines(&h) <= full {
        create_session(&h).await;
    }
    assert_eq!(journal_lines(&h), full + 1);
    let blocker = h.state_dir.path().join("journal.jsonl.compact.tmp");
    std::fs::create_dir(&blocker).unwrap();
    let before = std::fs::read(h.state_dir.path().join("journal.jsonl")).unwrap();

    assert_refused(
        &migrate(&h, &sid, to("gpu1", 1, false)).await,
        500,
        "internal_error",
    );
    assert_eq!(
        std::fs::read(h.state_dir.path().join("journal.jsonl")).unwrap(),
        before
    );
    assert_eq!(owner(&h, &sid).await, owned(&sid, Some("gpu0"), 1, 1));
    let (status, ready) = readyz(&h).await;
    assert_eq!(status, 200, "{ready}");
    assert_eq!(ready["ready"], true, "{ready}");
    assert!(
        !ready["problems"]
            .as_array()
            .is_some_and(|p| p.iter().any(|p| p["field"] == "journal_poisoned")),
        "{ready}"
    );

    std::fs::remove_dir(&blocker).unwrap();
    let next = served(&h, turn(&h, &sid, 1)).await;
    assert_eq!(next["worker_id"], "gpu0", "{next}");
    assert_eq!(next["revision"], 2, "{next}");
    assert_eq!(owner(&h, &sid).await, owned(&sid, Some("gpu0"), 1, 2));
    assert_eq!(
        migrate(&h, &sid, to("gpu1", 1, false)).await,
        moved(&sid, "gpu1", 2, 2)
    );
    assert_eq!(owner(&h, &sid).await, owned(&sid, Some("gpu1"), 2, 2));
}

/// Two generating mocks (as [`generating_mocks`]) reporting a backend cap of 1, for [`CAP_ONE_TWEAK`].
async fn cap_one_mocks() -> Vec<MockHandle> {
    let mut out = Vec::new();
    for _ in 0..2 {
        let config = MockConfig {
            token_latency: Duration::from_millis(5),
            natural_output_tokens: 1_000_000,
            server_info: ServerInfo {
                max_running_requests: 1,
                ..ServerInfo::default()
            },
            ..MockConfig::default()
        };
        out.push(MockSglang::start(config).await.unwrap());
    }
    out
}

const CAP_ONE_TWEAK: (&str, &str) = ("max_running_requests = 4", "max_running_requests = 1");

/// The fence of a turn that holds the lease but is still queued (`CancelOutcome::Cancelled`): owner `gpu0` is
/// drained, `gpu1` (cap 1) runs another request, so the session's next turn waits for `gpu1`'s slot. A forced
/// move to `gpu1` cancels it before dispatch: the client sees `409 session_owner_superseded`, the request ends
/// `cancelled` with that reason (one terminal record), no backend saw its rid and the revision does not move;
/// once `gpu1` is free the next turn runs there at the next revision.
async fn queued_fence(stream: bool) {
    let h = Harness::start_replicas_tuned(
        cap_one_mocks().await,
        &[OPERATOR_TWEAK, CAP_ONE_TWEAK],
        test_tuning(),
    )
    .await;
    let sid = create_session(&h).await;
    let first = served(&h, turn(&h, &sid, 0)).await;
    assert_eq!(first["worker_id"], "gpu0", "{first}");
    assert_eq!(owner(&h, &sid).await, owned(&sid, Some("gpu0"), 1, 1));
    until("no request in flight", || {
        h.app.admission_snapshot().inflight == 0
    })
    .await;

    operate(&h, "gpu0", "drain").await;
    let mut body = chat_body(false);
    body["max_completion_tokens"] = 30_000.into();
    let blocker = start_turn(&h, h.chat(&body), 1, false).await;

    let call = tokio::spawn(turn_with(&h, &sid, 1, 4, stream).send());
    until("the turn queued", || h.app.admission_snapshot().queued == 1).await;
    let records = journal(h.state_dir.path());
    let admitted: Vec<&str> = records
        .iter()
        .filter(|r| {
            r["type"] == "admitted"
                && r["session"]["session_id"] == sid.as_str()
                && r["session"]["expected_revision"] == 1
        })
        .map(|r| r["rid"].as_str().unwrap())
        .collect();
    assert_eq!(admitted.len(), 1, "{records:?}");
    let rid = admitted[0].to_owned();
    let (code, status) = h.status(&rid, TOKEN_A).await;
    assert_eq!(code, 200, "{status}");
    assert_eq!(status["state"], "queued", "{status}");

    assert_eq!(
        migrate(&h, &sid, to("gpu1", 1, true)).await,
        moved(&sid, "gpu1", 2, 1)
    );
    // Cancelled before dispatch: answered from the settled status (the error and the request id, no `cere`
    // object, as every pre-dispatch terminal).
    let r = call.await.unwrap().unwrap();
    let code = r.status().as_u16();
    let header = r.headers().get("x-cere-request-id").cloned();
    let text = r.text().await.unwrap();
    assert_eq!(code, 409, "{text}");
    assert_eq!(
        header.as_ref().and_then(|v| v.to_str().ok()),
        Some(rid.as_str())
    );
    let reply: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(reply["error"]["code"], SUPERSEDED, "{text}");
    assert_eq!(reply["error"]["type"], "invalid_request_error", "{text}");

    let terminal = h.terminal(&rid).await;
    assert_eq!(terminal["state"], "cancelled", "{terminal}");
    assert_eq!(terminal["reason"], SUPERSEDED, "{terminal}");
    assert_eq!(terminal["worker_id"], Value::Null, "{terminal}");
    for i in 0..2 {
        assert!(
            !h.replica(i).dispatched_rids().contains(&rid),
            "replica {i} saw the cancelled turn"
        );
    }
    let terminals = journal(h.state_dir.path())
        .iter()
        .filter(|r| r["type"] == "terminal" && r["rid"] == rid.as_str())
        .count();
    assert_eq!(terminals, 1, "the cancel is journaled once");
    assert_eq!(owner(&h, &sid).await, owned(&sid, Some("gpu1"), 2, 1));

    let cancel = h
        .post(
            &format!("/v1/cere/requests/{}/cancel", blocker.rid),
            TOKEN_A,
        )
        .send()
        .await
        .unwrap();
    assert!(cancel.status().is_success(), "{}", cancel.status());
    blocker.call.await.unwrap();
    until("no request in flight", || {
        h.app.admission_snapshot().inflight == 0
    })
    .await;
    let next = served(&h, turn(&h, &sid, 1)).await;
    assert_eq!(next["worker_id"], "gpu1", "{next}");
    assert_eq!(next["revision"], 2, "{next}");
    assert_eq!(owner(&h, &sid).await, owned(&sid, Some("gpu1"), 2, 2));
}

#[tokio::test(flavor = "multi_thread")]
async fn p2_force_cancels_a_queued_turn_superseded() {
    queued_fence(false).await;
    queued_fence(true).await;
}

/// The real `cered` logs every owner change of a migration (`cause=migrate`, and `cause=force` for a forced
/// takeover) next to the grant-time pin (`cause=pin`).
#[tokio::test(flavor = "multi_thread")]
async fn p2_cered_logs_each_owner_change() {
    let m0 = MockSglang::start(MockConfig::default()).await.unwrap();
    let m1 = MockSglang::start(MockConfig::default()).await.unwrap();
    let state_dir = StateDir::new();
    let text = replicas_config_text(
        &[m0.base_url(), m1.base_url()],
        state_dir.path(),
        free_port(),
        &[OPERATOR_TWEAK],
    );
    let config = ConfigFile::new(&text);
    let mut cered = Cered::spawn(&config.path, &[("TOK_OP", Some(OPERATOR_TOKEN))])
        .await
        .unwrap();
    let client = http();
    let mut ready = false;
    for _ in 0..500 {
        let body: Value = client
            .get(cered.url("/readyz"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if body["workers"]["gpu0"]["ready"] == true && body["workers"]["gpu1"]["ready"] == true {
            ready = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(ready, "both workers ready");

    let created: Value = client
        .post(cered.url("/v1/cere/sessions"))
        .bearer_auth(TOKEN_A)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let sid = created["session_id"].as_str().unwrap().to_owned();
    let r = client
        .post(cered.url("/v1/chat/completions"))
        .bearer_auth(TOKEN_A)
        .header("X-Cere-Session-Id", &sid)
        .header("X-Cere-Expected-Revision", "0")
        .json(&json!({"model": MODEL, "messages": history(0), "max_completion_tokens": 4}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let reply: Value = r.json().await.unwrap();
    let pinned = reply["cere"]["worker_id"].as_str().unwrap().to_owned();
    let other = if pinned == "gpu0" { "gpu1" } else { "gpu0" };
    let operator = |method: reqwest::Method, path: String| {
        client
            .request(method, cered.url(&path))
            .bearer_auth(OPERATOR_TOKEN)
    };
    let (status, body) =
        answer(operator(reqwest::Method::POST, migrate_path(&sid)).json(&to(other, 1, false)))
            .await;
    assert_eq!(status, 200, "{body}");
    let (status, body) = answer(operator(
        reqwest::Method::POST,
        format!("/v1/cere/workers/{other}/drain"),
    ))
    .await;
    assert_eq!(status, 200, "{body}");
    let (status, body) =
        answer(operator(reqwest::Method::POST, migrate_path(&sid)).json(&to(&pinned, 2, true)))
            .await;
    assert_eq!(status, 200, "{body}");

    cered.signal("TERM");
    let (code, stderr) = cered.exit(Duration::from_secs(20)).await;
    assert_eq!(code, Some(0), "{stderr}");
    for line in [
        format!("cered event=session_owner session={sid} worker={pinned} generation=1 cause=pin"),
        format!(
            "cered event=session_owner session={sid} worker={other} generation=2 cause=migrate"
        ),
        format!("cered event=session_owner session={sid} worker={pinned} generation=3 cause=force"),
    ] {
        assert!(
            stderr.lines().any(|l| l == line),
            "missing {line:?} in:\n{stderr}"
        );
    }
}
