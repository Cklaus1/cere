//! Remote dispatch, happy path (m35 design note Q11, P7; build-loop-m35 design note "CREM split", track DA): an
//! in-process `cered` dispatches chat to a remote worker, the in-process stub agent (`support::stub_agent`), which
//! runs every accepted Submit against a second mock SGLang. The local worker never turns ready (its mock serves
//! another model than the pin), so only the remote worker takes requests, except in the sticky-session test where
//! both are ready. Unary and streamed chat give the local path's bytes after normalising ids, timings and `cere`
//! identity fields; each request reaches the agent's journal once; the request status names the remote worker and
//! epoch; the `dispatched` record carries node and endpoint; the remote worker's admission cap follows its
//! inventory and `max_inflight`; no grant after the lease view expired; the remote smoke takes an admission slot;
//! `/readyz` is ready with only a remote worker ready; sessions stick across a local and a remote worker.
//!
//! Submit faults (H2, track DB1): a lost Submit reply is retried with the same operation; a suspect-class refusal
//! (`lease_expired`, `queue_full`) re-queues the request; a joining-class refusal (`stale_term`) takes the worker
//! back to Joining and re-queues; a second refusal from a worker already tried fails `worker_not_ready`; any other
//! refusal is `backend_protocol_error`. A managed agent's model refusals (`model_draining`, `backend_unconfirmed`,
//! `backend_exited`) are suspect-class: the request re-queues to the local gpu0 and completes there
//! (m36 design note Q10, review M36-05).
//!
//! Cancel, violations and terminals (H2, track DB2): the cancel endpoint and a client disconnect mid-stream each
//! send one `CancelWork` (`<rid>/cancel/1`, the receipt's ticket); a cancel before the receipt sends it once the
//! receipt arrived; a naturally completed work is never cancelled; a streamed protocol violation (an oversize
//! event, an offset gap, the end without a terminal, an unparsable event) fails `backend_protocol_error` and
//! cancels the work once; a terminal reason of the work reaches the client as its error code. A client body at
//! `config::REMOTE_BODY_MAX` fits one SubmitWork and is served (Q14).
//!
//! Orphan hold (H2, track DB3; these harnesses keep the local worker drained instead of unready, since they tune
//! `[ledger]`): a cancelled remote work keeps its slot, charged to the worker, until the agent reports its
//! operation `cleanup_acknowledged` (the release is journaled), or until its retention bound; a naturally
//! completed work holds nothing.
mod support;

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cere_mock_sglang::{Fault, MockConfig, MockHandle, MockSglang, ServerInfo};
use cere_runtime::fed::registry::RemoteState;
use cere_runtime::fed::wire;
use cere_runtime::readiness::REMOTE_SMOKE_SCOPE;
use cere_runtime::worker::WorkerState;
use serde_json::{json, Value};
use support::stub_agent::{StubAgent, Violation, WorkOutcome, STUB_INSTANCE_ID};
use support::*;
use tokio::task::JoinHandle;
use tonic::Code;

const A: &str = "worker-a";
const PRINCIPAL_A: &str = "cere-worker-a";
const TIMINGS: &str =
    "heartbeat_interval_ms = 100\nadmission_lease_ms = 1000\nsuspect_after_ms = 2000\n\
                       lost_after_ms = 4000\n";
/// The stub agent's heartbeat period.
const HEARTBEAT: Duration = Duration::from_millis(50);
/// The placeholder of every normalised field.
const NORMALISED: &str = "<normalised>";

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// `[[federation.workers]]` for `A` dialled at the stub, pinned like [`Federation::toml`].
fn worker_toml(stub: &StubAgent, max_inflight: u64) -> String {
    format!(
        "[[federation.workers]]\nworker_id = \"{A}\"\nendpoint = \"https://{}\"\n\
         principal = \"{PRINCIPAL_A}\"\nmax_inflight = {max_inflight}\n\n[federation.workers.expect]\n\
         served_model_name = \"{MODEL}\"\ntp_size = 1\nkv_cache_dtype = \"fp8_e4m3\"\n\n",
        stub.addr()
    )
}

/// A mock serving another model than the pin: the local worker never passes readiness.
fn unready_local() -> MockConfig {
    MockConfig {
        server_info: ServerInfo {
            served_model_name: "another-model".to_owned(),
            ..ServerInfo::default()
        },
        ..MockConfig::default()
    }
}

/// The remote side of a test: the stub agent serving chat on its own mock.
struct Remote {
    fed: Federation,
    stub: StubAgent,
    mock: MockHandle,
}

impl Remote {
    /// `script` runs on the stub before anything dials it.
    async fn start(mock: MockConfig, script: impl FnOnce(&StubAgent)) -> Remote {
        let fed = Federation::new();
        let stub = StubAgent::start(&fed, A, PRINCIPAL_A, 1).await;
        let mock = MockSglang::start(mock).await.unwrap();
        stub.serve_chat(&mock.base_url());
        script(&stub);
        Remote { fed, stub, mock }
    }

    fn config(&self, max_inflight: u64) -> String {
        let mut text = self.fed.toml(TIMINGS, &[]);
        text.push_str(&worker_toml(&self.stub, max_inflight));
        text
    }

    fn link(&mut self) {
        self.stub.link(&self.fed, HEARTBEAT);
    }
}

/// A federated `cered` whose only ready worker will be the linked remote `A` (`max_inflight` 4).
async fn remote_only(remote: &mut Remote) -> Harness {
    let h =
        Harness::start_federated_unready(unready_local(), &remote.config(4), fast_tuning()).await;
    remote.link();
    until_remote_ready(&h).await;
    h
}

/// A federated `cered` with a ready local worker on a [`generating`] mock beside the linked remote `A`
/// (`max_inflight` 4), under [`fast_remote_tuning`]: `A` joins within a few 100 ms driver rounds, while the local
/// worker probes every 5 s. These tests route between the two and wait on no local probe.
async fn local_and_remote(remote: &mut Remote) -> Harness {
    let h = Harness::start_federated(generating(), &remote.config(4), fast_remote_tuning()).await;
    remote.link();
    until_remote_ready(&h).await;
    h
}

async fn until_remote_ready(h: &Harness) -> Value {
    let (_, body) =
        readyz_until(h, "worker-a ready", |b| b["workers"][A]["state"] == "ready").await;
    body
}

fn state(h: &Harness) -> RemoteState {
    h.app
        .remote_workers()
        .expect("[federation] configured")
        .get(A, now_ms())
        .expect("configured remote worker")
}

fn rid_of(response: &reqwest::Response) -> String {
    response.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned()
}

/// `value` with the fields that differ between two harnesses or two requests replaced: the completion id and
/// creation time, and the `cere` identity (request, worker, epochs, instance, config hash) and timings.
fn normalised(mut value: Value) -> Value {
    for key in ["id", "created"] {
        if value.get(key).is_some() {
            value[key] = json!(NORMALISED);
        }
    }
    if let Some(cere) = value.get_mut("cere").and_then(Value::as_object_mut) {
        for key in [
            "request_id",
            "worker_id",
            "worker_epoch",
            "epoch",
            "instance_id",
            "config_hash",
            "timings",
        ] {
            if cere.contains_key(key) {
                cere.insert(key.to_owned(), json!(NORMALISED));
            }
        }
    }
    value
}

fn normalised_events(events: &[Event]) -> Vec<Value> {
    events
        .iter()
        .map(|e| match e {
            Event::Data(value) => normalised(value.clone()),
            Event::Done => json!("[DONE]"),
        })
        .collect()
}

/// The remote identity of a `cere` object: worker `A` at its registered epoch, the agent-reported instance.
fn assert_remote_identity(h: &Harness, cere: &Value, rid: &str) {
    assert_eq!(cere["request_id"], rid, "{cere}");
    assert_eq!(cere["worker_id"], A, "{cere}");
    assert_eq!(cere["worker_epoch"], json!(state(h).epoch), "{cere}");
    assert_eq!(cere["epoch"], h.app.epoch(), "{cere}");
    assert_eq!(cere["instance_id"], STUB_INSTANCE_ID, "{cere}");
}

/// The agent journaled exactly one fresh accept of `rid`.
fn assert_one_accept(stub: &StubAgent, rid: &str) {
    let accepts: Vec<(String, bool)> = stub
        .accepts()
        .into_iter()
        .filter(|(request, _)| request == rid)
        .collect();
    assert_eq!(accepts, vec![(rid.to_owned(), true)]);
}

/// The request status names the remote worker and its epoch.
async fn assert_remote_status(h: &Harness, rid: &str) {
    let status = h.terminal(rid).await;
    assert_eq!(status["state"], "completed", "{status}");
    assert_eq!(status["worker_id"], A, "{status}");
    assert_eq!(status["worker_epoch"], json!(state(h).epoch), "{status}");
}

/// The `dispatched` record of `rid` carries node `remote`, the endpoint and no base URL.
fn assert_remote_dispatched(h: &Harness, stub: &StubAgent, rid: &str) {
    let records = journal(h.state_dir.path());
    let dispatched: Vec<&Value> = records
        .iter()
        .filter(|r| r["type"] == "dispatched" && r["rid"] == rid)
        .collect();
    assert_eq!(dispatched.len(), 1, "{records:?}");
    let record = dispatched[0];
    assert_eq!(record["worker"], A, "{record}");
    assert_eq!(record["worker_epoch"], json!(state(h).epoch), "{record}");
    assert_eq!(record["node"], "remote", "{record}");
    assert_eq!(
        record["endpoint"],
        format!("https://{}", stub.addr()),
        "{record}"
    );
    assert!(record.get("base_url").is_none(), "{record}");
}

/// Q11/P7: a unary chat through the remote worker answers the local path's body (normalised), with the remote
/// identity in `cere`; the agent journaled it once, the status names the remote worker and epoch, and the
/// `dispatched` record carries node and endpoint.
#[tokio::test(flavor = "multi_thread")]
async fn unary_chat_through_the_remote_worker_matches_the_local_path() {
    let local = Harness::start(MockConfig::default(), &[]).await;
    let r = local.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let expected: Value = r.json().await.unwrap();

    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h = remote_only(&mut remote).await;
    let r = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let rid = rid_of(&r);
    let content_type = r.headers()["content-type"].clone();
    let body: Value = r.json().await.unwrap();
    assert_remote_identity(&h, &body["cere"], &rid);
    assert_eq!(normalised(body), normalised(expected));
    assert_eq!(content_type, "application/json");
    assert!(remote.mock.dispatched_rids().contains(&rid));
    assert!(h.mock.dispatched_rids().is_empty());

    assert_one_accept(&remote.stub, &rid);
    assert_remote_status(&h, &rid).await;
    assert_remote_dispatched(&h, &remote.stub, &rid);
    let submit = remote
        .stub
        .submits()
        .into_iter()
        .find(|s| s.request_id == rid)
        .unwrap();
    assert_eq!(submit.attempt, 1);
    let unit: Value = serde_json::from_slice(&submit.work_unit_json).unwrap();
    assert_eq!(unit, json!({"kind": "chat", "stream": false}));
    let input: Value = serde_json::from_slice(&submit.bounded_inference_input_json).unwrap();
    let authority = submit.authority.as_ref().unwrap();
    assert_eq!(authority.operation_id, rid);
    assert_eq!(authority.scope_id, "a");
    assert_eq!(input["rid"], rid.as_str());
    assert_eq!(input["stream"], false);
}

/// Q11/P7: a streamed chat through the remote worker gives the local path's SSE events (same frames, `cere`
/// fields, usage, `[DONE]`) after normalising; the agent journaled it once.
#[tokio::test(flavor = "multi_thread")]
async fn streamed_chat_through_the_remote_worker_matches_the_local_path() {
    let local = Harness::start(MockConfig::default(), &[]).await;
    let r = local.chat(&chat_body(true)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let expected = SseReader::new(r).rest().await;
    check_stream(&expected);

    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h = remote_only(&mut remote).await;
    let r = h.chat(&chat_body(true)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let rid = rid_of(&r);
    assert_eq!(r.headers()["content-type"], "text/event-stream");
    let events = SseReader::new(r).rest().await;
    let terminal = check_stream(&events);
    assert_remote_identity(&h, &terminal["cere"], &rid);
    assert_eq!(normalised_events(&events), normalised_events(&expected));

    assert_one_accept(&remote.stub, &rid);
    assert_remote_status(&h, &rid).await;
    assert_remote_dispatched(&h, &remote.stub, &rid);
    let submit = remote
        .stub
        .submits()
        .into_iter()
        .find(|s| s.request_id == rid)
        .unwrap();
    let unit: Value = serde_json::from_slice(&submit.work_unit_json).unwrap();
    assert_eq!(unit, json!({"kind": "chat", "stream": true}));
}

async fn remote_cap(h: &Harness) -> Value {
    let r = h.get("/v1/cere/stats", TOKEN_A).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    body["workers"][A].clone()
}

/// P7: the remote worker's admission cap is `max_inflight` until its inventory is read, then the smaller of the
/// inventory's `max_running_requests` and `max_inflight`.
#[tokio::test(flavor = "multi_thread")]
async fn admission_cap_follows_the_inventory_and_max_inflight() {
    // The inventory is the smaller.
    let mut remote = Remote::start(MockConfig::default(), |stub| {
        let mut inventory = stub.inventory();
        inventory.max_running_requests = Some(2);
        stub.set_inventory(inventory);
    })
    .await;
    let h =
        Harness::start_federated_unready(unready_local(), &remote.config(4), fast_tuning()).await;
    let before = remote_cap(&h).await;
    assert_eq!(
        before["cap"], 4,
        "max_inflight before the inventory: {before}"
    );
    remote.link();
    until_remote_ready(&h).await;
    let after = remote_cap(&h).await;
    assert_eq!(
        after["cap"], 2,
        "the inventory's max_running_requests: {after}"
    );
    assert_eq!(after["state"], "ready", "{after}");
    assert_eq!(after["epoch"], json!(state(&h).epoch), "{after}");

    // `max_inflight` is the smaller.
    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h =
        Harness::start_federated_unready(unready_local(), &remote.config(3), fast_tuning()).await;
    remote.link();
    until_remote_ready(&h).await;
    let after = remote_cap(&h).await;
    assert_eq!(
        after["cap"], 3,
        "max_inflight under the inventory's 4: {after}"
    );
}

/// P7/Q7: once the coordinator-side lease view of the remote worker expired (its agent went silent), no request
/// is granted to it: the request is refused and the agent sees no Submit.
#[tokio::test(flavor = "multi_thread")]
async fn no_grant_once_the_lease_view_expired() {
    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h = remote_only(&mut remote).await;
    let r = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 200, "served while the lease holds");

    remote.stub.unlink();
    // A heartbeat in flight at the unlink may still renew the lease once: wait out the lease view current at
    // each look until it is no longer Ready.
    for _ in 0..10 {
        let current = state(&h);
        if current.state != WorkerState::Ready {
            break;
        }
        let until = current.lease_until_ms.expect("a lease view while ready");
        while now_ms() <= until {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
    assert_ne!(state(&h).state, WorkerState::Ready);
    let submits = remote.stub.submits().len();
    let r = h
        .chat(&chat_body(false))
        .header("X-Cere-Deadline-Ms", "500")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["error"]["code"], "not_ready", "{body}");
    assert_eq!(
        remote.stub.submits().len(),
        submits,
        "no Submit after the lease"
    );
}

/// P7: the remote smoke waits for an admission slot of its worker like the local smoke: while it runs (the remote
/// backend answers slowly), one slot is charged although no request is in flight.
#[tokio::test(flavor = "multi_thread")]
async fn the_remote_smoke_takes_an_admission_slot() {
    let slow = MockConfig {
        token_latency: Duration::from_millis(300),
        ..MockConfig::default()
    };
    let mut remote = Remote::start(slow, |_| {}).await;
    let h =
        Harness::start_federated_unready(unready_local(), &remote.config(4), fast_tuning()).await;
    assert_eq!(h.app.admission_snapshot().inflight, 0);
    remote.link();
    let mut charged = false;
    for _ in 0..2_500 {
        let smoking = remote.stub.submits().iter().any(|s| {
            s.authority
                .as_ref()
                .is_some_and(|a| a.scope_id == REMOTE_SMOKE_SCOPE)
        });
        if smoking
            && state(&h).state == WorkerState::Joining
            && h.app.admission_snapshot().inflight == 1
        {
            charged = true;
            break;
        }
        if state(&h).state == WorkerState::Ready {
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    assert!(charged, "the smoke ran without an admission slot");
    until_remote_ready(&h).await;
    for _ in 0..250 {
        if h.app.admission_snapshot().inflight == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("the smoke's slot was never released");
}

/// P11: top-level `/readyz` is ready (200) when the only Ready worker is remote.
#[tokio::test(flavor = "multi_thread")]
async fn readyz_is_ready_with_only_a_remote_worker_ready() {
    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h =
        Harness::start_federated_unready(unready_local(), &remote.config(4), fast_tuning()).await;
    let (status, body) = readyz(&h).await;
    assert_eq!(status, 503, "{body}");
    assert_eq!(body["ready"], false, "{body}");
    remote.link();
    let body = until_remote_ready(&h).await;
    let local = h.app.replicas()[0].worker_id();
    assert_ne!(body["workers"][local]["state"], "ready", "{body}");
    let (status, body) = readyz(&h).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["ready"], true, "{body}");
}

async fn create_session(h: &Harness) -> String {
    let r = h.post("/v1/cere/sessions", TOKEN_A).send().await.unwrap();
    assert_eq!(r.status(), 201);
    let body: Value = r.json().await.unwrap();
    body["session_id"].as_str().unwrap().to_owned()
}

/// A session turn at `revision` whose history is `messages`.
fn turn(h: &Harness, session: &str, revision: u64, messages: &[&str]) -> reqwest::RequestBuilder {
    let messages: Vec<Value> = messages
        .iter()
        .enumerate()
        .map(|(i, content)| {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            json!({"role": role, "content": content})
        })
        .collect();
    h.chat(&json!({"model": MODEL, "messages": messages, "max_completion_tokens": 4}))
        .header("X-Cere-Session-Id", session)
        .header("X-Cere-Expected-Revision", revision.to_string())
}

/// Sends a turn that must succeed; returns the worker that served it and the reply text, after checking that
/// only that worker's mock received the rid.
async fn served(
    h: &Harness,
    remote: &Remote,
    request: reqwest::RequestBuilder,
) -> (String, String) {
    let r = request.send().await.unwrap();
    assert_eq!(r.status(), 200);
    let reply: Value = r.json().await.unwrap();
    let rid = reply["cere"]["request_id"].as_str().unwrap().to_owned();
    let worker = reply["cere"]["worker_id"].as_str().unwrap().to_owned();
    assert_eq!(
        remote.mock.dispatched_rids().contains(&rid),
        worker == A,
        "{reply}"
    );
    assert_eq!(
        h.mock.dispatched_rids().contains(&rid),
        worker != A,
        "{reply}"
    );
    let text = reply["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_owned();
    (worker, text)
}

/// Waits until no admission slot is charged (a finished request's permit drops just after its response).
async fn idle(h: &Harness) {
    for _ in 0..1_000 {
        if h.app.admission_snapshot().inflight == 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("a request stayed in flight");
}

/// A long unpinned generation on the local worker until it is cancelled. Waits for the outcome (the mock saw
/// it, or the call ended without reaching it), not a wall bound a loaded host can miss; [`HELD`] only catches a
/// request that does neither.
async fn hold_local(h: &Harness) -> (String, JoinHandle<reqwest::Result<reqwest::Response>>) {
    let before = h.mock.dispatched_rids().len();
    let mut body = chat_body(false);
    body["max_completion_tokens"] = 30_000.into();
    let call = tokio::spawn(h.chat(&body).send());
    let settled = tokio::time::timeout(HELD, async {
        while h.mock.dispatched_rids().len() <= before && !call.is_finished() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    assert!(
        settled.is_ok(),
        "the held request neither reached the local worker nor ended"
    );
    if h.mock.dispatched_rids().len() <= before {
        let ended = call.await.unwrap();
        panic!("the held request ended without reaching the local worker: {ended:?}");
    }
    (h.mock.dispatched_rids()[before].clone(), call)
}

/// Bounds [`hold_local`]'s wait: it only catches a request that hangs before the local worker.
const HELD: Duration = Duration::from_secs(60);

/// Q8 across nodes: a session whose first turn ran on the remote worker (the local one was busy) stays there, and
/// one whose first turn ran on the local worker stays there, although an unpinned request goes to the local
/// worker (least in flight, then the lowest id).
#[tokio::test(flavor = "multi_thread")]
async fn sessions_stick_across_a_local_and_a_remote_worker() {
    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h = local_and_remote(&mut remote).await;
    let local = h.app.replicas()[0].worker_id().to_owned();

    let (held, call) = hold_local(&h).await;
    let on_remote = create_session(&h).await;
    let (worker, remote_reply) = served(&h, &remote, turn(&h, &on_remote, 0, &["hi"])).await;
    assert_eq!(worker, A, "the first turn avoids the busy local worker");
    let cancel = h
        .post(&format!("/v1/cere/requests/{held}/cancel"), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert!(cancel.status().is_success(), "{}", cancel.status());
    call.await.unwrap().unwrap();
    idle(&h).await;

    let on_local = create_session(&h).await;
    let (worker, local_reply) = served(&h, &remote, turn(&h, &on_local, 0, &["hi"])).await;
    assert_eq!(worker, local, "an idle tie goes to the lowest id");
    idle(&h).await;

    let history = ["hi", remote_reply.as_str(), "more"];
    let (worker, _) = served(&h, &remote, turn(&h, &on_remote, 1, &history)).await;
    assert_eq!(worker, A, "the remote pin holds");
    idle(&h).await;
    let history = ["hi", local_reply.as_str(), "more"];
    let (worker, _) = served(&h, &remote, turn(&h, &on_local, 1, &history)).await;
    assert_eq!(worker, local, "the local pin holds");
}

/// The SubmitWork calls the stub authorized so far (refused ones and the smoke's included).
fn submit_calls(stub: &StubAgent) -> usize {
    stub.count("SubmitWork")
}

/// H2: a Submit whose reply was lost (the agent journaled it) is retried with the same operation inside the
/// budget; the journal answers the retry with the original receipt and the request is served.
#[tokio::test(flavor = "multi_thread")]
async fn a_lost_submit_reply_is_retried_with_the_same_operation() {
    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h = remote_only(&mut remote).await;
    remote.stub.lose_submit_replies(1);
    let r = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let rid = rid_of(&r);
    let accepts: Vec<(String, bool)> = remote
        .stub
        .accepts()
        .into_iter()
        .filter(|(request, _)| *request == rid)
        .collect();
    assert_eq!(accepts, vec![(rid.clone(), true), (rid.clone(), false)]);
    assert_remote_status(&h, &rid).await;
}

/// H2: a suspect-class refusal (`lease_expired`, `queue_full`) re-queues the request in admission; the next
/// grant's Submit is accepted and the request is served.
#[tokio::test(flavor = "multi_thread")]
async fn a_suspect_refusal_requeues_the_request() {
    for (code, reason) in [
        (Code::Unavailable, wire::LEASE_EXPIRED),
        (Code::ResourceExhausted, wire::QUEUE_FULL),
    ] {
        let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
        let h = remote_only(&mut remote).await;
        remote.stub.refuse_next_submit(code, reason);
        let before = submit_calls(&remote.stub);
        let r = h.chat(&chat_body(false)).send().await.unwrap();
        assert_eq!(r.status(), 200, "{reason}");
        let rid = rid_of(&r);
        assert_eq!(submit_calls(&remote.stub) - before, 2, "{reason}");
        assert_one_accept(&remote.stub, &rid);
        assert_remote_status(&h, &rid).await;
    }
}

/// H2: a joining-class refusal (`stale_term`) takes the worker back to Joining (a new round); once it is Ready
/// again the re-queued request is served.
#[tokio::test(flavor = "multi_thread")]
async fn a_joining_refusal_rejoins_the_worker_and_requeues() {
    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h = remote_only(&mut remote).await;
    let round = state(&h).round;
    remote
        .stub
        .refuse_next_submit(Code::FailedPrecondition, wire::STALE_TERM);
    let r = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let rid = rid_of(&r);
    assert!(state(&h).round > round, "the refusal started a new round");
    assert_one_accept(&remote.stub, &rid);
    assert_remote_status(&h, &rid).await;
}

/// H2: a second refusal from a worker the request was already refused by fails it `worker_not_ready` (503).
#[tokio::test(flavor = "multi_thread")]
async fn a_second_refusal_from_the_same_worker_fails_not_ready() {
    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h = remote_only(&mut remote).await;
    remote
        .stub
        .refuse_next_submit(Code::Unavailable, wire::LEASE_EXPIRED);
    remote
        .stub
        .refuse_next_submit(Code::Unavailable, wire::LEASE_EXPIRED);
    let before = submit_calls(&remote.stub);
    let r = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 503);
    let rid = rid_of(&r);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["error"]["code"], "not_ready", "{body}");
    assert_eq!(submit_calls(&remote.stub) - before, 2);
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], "worker_not_ready", "{status}");
}

/// H2: a refusal outside the suspect and joining classes (`operation_conflict`, `payload_digest_mismatch`) is a
/// protocol failure: 502 `backend_protocol_error`, without a retry.
#[tokio::test(flavor = "multi_thread")]
async fn other_refusals_are_backend_protocol_errors() {
    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h = remote_only(&mut remote).await;
    for (code, reason) in [
        (Code::AlreadyExists, wire::OPERATION_CONFLICT),
        (Code::InvalidArgument, wire::PAYLOAD_DIGEST_MISMATCH),
    ] {
        remote.stub.refuse_next_submit(code, reason);
        let before = submit_calls(&remote.stub);
        let r = h.chat(&chat_body(false)).send().await.unwrap();
        assert_eq!(r.status(), 502, "{reason}");
        let rid = rid_of(&r);
        let body: Value = r.json().await.unwrap();
        assert_eq!(body["error"]["code"], "backend_protocol_error", "{body}");
        assert_eq!(submit_calls(&remote.stub) - before, 1, "{reason}");
        let status = h.terminal(&rid).await;
        assert_eq!(status["state"], "failed", "{status}");
    }
}

/// m36 design note Q10 (review M36-05): a SubmitWork the remote agent refuses because its model is not serving
/// (`model_draining`, `backend_unconfirmed`, `backend_exited`) is suspect-class: the request re-queues to the
/// local gpu0 (granted to worker-a first because gpu0 is busy) and completes there; worker-a journaled nothing.
#[tokio::test(flavor = "multi_thread")]
async fn a_model_refusal_requeues_the_request_to_the_local_worker() {
    for (code, reason) in [
        (Code::Unavailable, wire::MODEL_DRAINING),
        (Code::Unavailable, wire::BACKEND_UNCONFIRMED),
        (Code::Unavailable, wire::BACKEND_EXITED),
    ] {
        let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
        let h = local_and_remote(&mut remote).await;
        let local = h.app.replicas()[0].worker_id().to_owned();
        let (held, call) = hold_local(&h).await;
        remote.stub.refuse_next_submit(code, reason);
        let before = submit_calls(&remote.stub);
        let r = h.chat(&chat_body(false)).send().await.unwrap();
        assert_eq!(r.status(), 200, "{reason}");
        let rid = rid_of(&r);
        let reply: Value = r.json().await.unwrap();
        assert_eq!(
            reply["cere"]["worker_id"],
            json!(local),
            "{reason}: {reply}"
        );
        assert_eq!(
            submit_calls(&remote.stub) - before,
            1,
            "{reason}: one refused Submit on worker-a"
        );
        assert!(
            remote
                .stub
                .accepts()
                .iter()
                .all(|(request, _)| *request != rid),
            "{reason}: worker-a journaled nothing"
        );
        assert!(h.mock.dispatched_rids().contains(&rid), "{reason}");
        let status = h.terminal(&rid).await;
        assert_eq!(status["state"], "completed", "{reason}: {status}");
        cancel(&h, &held).await;
        call.await.unwrap().unwrap();
    }
}

/// A mock that generates until the request is cancelled (one token per 5 ms).
fn generating() -> MockConfig {
    MockConfig {
        token_latency: Duration::from_millis(5),
        natural_output_tokens: 1_000_000,
        ..MockConfig::default()
    }
}

/// A chat that runs until it is cancelled on a [`generating`] mock.
fn long_body(stream: bool) -> Value {
    let mut body = chat_body(stream);
    body["max_completion_tokens"] = 30_000.into();
    body
}

async fn cancel(h: &Harness, rid: &str) {
    let r = h
        .post(&format!("/v1/cere/requests/{rid}/cancel"), TOKEN_A)
        .send()
        .await
        .unwrap();
    assert!(r.status().is_success(), "{}", r.status());
}

/// The rid of the first Submit of scope `a` after the first `before` ones, once the stub journaled it.
async fn submitted_after(stub: &StubAgent, before: usize) -> String {
    for _ in 0..1_000 {
        let submits = stub.submits();
        let submit = submits
            .iter()
            .skip(before)
            .find(|s| s.authority.as_ref().is_some_and(|a| a.scope_id == "a"));
        if let Some(submit) = submit {
            return submit.request_id.clone();
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("no Submit reached the stub");
}

/// Waits for the `CancelWork` of `rid`, then checks it stays the only one: operation `<rid>/cancel/1` in scope
/// `a`, naming the Submit by request id, attempt 1 and its receipt's ticket.
async fn assert_one_cancel(stub: &StubAgent, rid: &str) {
    let cancels = || -> Vec<_> {
        stub.cancels()
            .into_iter()
            .filter(|c| c.request_id == rid)
            .collect()
    };
    for _ in 0..400 {
        if !cancels().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    // Room for a second one to arrive.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let cancels = cancels();
    assert_eq!(cancels.len(), 1, "{cancels:?}");
    let cancel = &cancels[0];
    let authority = cancel.authority.as_ref().unwrap();
    assert_eq!(authority.operation_id, format!("{rid}/cancel/1"));
    assert_eq!(authority.scope_id, "a");
    assert_eq!(cancel.attempt, 1);
    let submit = stub.op("a", rid).expect("the Submit op");
    assert_eq!(cancel.completion_ticket, submit.ticket);
}

/// H2: the cancel endpoint mid-stream sends one `CancelWork` with the receipt's ticket; the agent's
/// `cancel_requested` terminal acknowledges it and the stream ends `request_cancelled`.
#[tokio::test(flavor = "multi_thread")]
async fn the_cancel_endpoint_mid_stream_cancels_the_remote_work() {
    let mut remote = Remote::start(generating(), |_| {}).await;
    let h = remote_only(&mut remote).await;
    let r = h.chat(&long_body(true)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let rid = rid_of(&r);
    let mut sse = SseReader::new(r);
    let mut events = vec![sse.next().await.unwrap(), sse.next().await.unwrap()];
    cancel(&h, &rid).await;
    events.extend(sse.rest().await);
    let terminal = check_stream(&events);
    assert_eq!(terminal["error"]["code"], "request_cancelled", "{terminal}");
    assert_eq!(terminal["cere"]["state"], "cancelled", "{terminal}");
    assert_one_cancel(&remote.stub, &rid).await;
}

/// H2: a client gone mid-stream cancels the remote work with one `CancelWork`; the request ends
/// `client_disconnected`.
#[tokio::test(flavor = "multi_thread")]
async fn a_client_disconnect_mid_stream_cancels_the_remote_work() {
    let mut remote = Remote::start(generating(), |_| {}).await;
    let h = remote_only(&mut remote).await;
    let r = h.chat(&long_body(true)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let rid = rid_of(&r);
    let mut sse = SseReader::new(r);
    sse.next().await.unwrap();
    sse.next().await.unwrap();
    drop(sse);
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "cancelled", "{status}");
    assert_eq!(status["reason"], "client_disconnected", "{status}");
    assert_one_cancel(&remote.stub, &rid).await;
}

/// H2: a cancel while the Submit's receipt is outstanding sends no `CancelWork` (it has no ticket); the receipt
/// sends it, and the request ends `request_cancelled`.
#[tokio::test(flavor = "multi_thread")]
async fn a_cancel_before_the_receipt_is_sent_once_the_receipt_arrives() {
    let mut remote = Remote::start(generating(), |_| {}).await;
    let h = remote_only(&mut remote).await;
    remote.stub.hold_submit_replies(true);
    let before = remote.stub.submits().len();
    let call = tokio::spawn(h.chat(&long_body(false)).send());
    let rid = submitted_after(&remote.stub, before).await;
    cancel(&h, &rid).await;
    // The executor takes the cancel while the receipt is held.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(remote.stub.cancels().is_empty());
    remote.stub.hold_submit_replies(false);
    let r = call.await.unwrap().unwrap();
    assert_eq!(r.status(), 409);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["error"]["code"], "request_cancelled", "{body}");
    assert_eq!(body["cere"]["state"], "cancelled", "{body}");
    assert_one_cancel(&remote.stub, &rid).await;
}

/// H2: a work that reached its natural terminal is never cancelled, unary or streamed.
#[tokio::test(flavor = "multi_thread")]
async fn a_naturally_completed_remote_work_is_not_cancelled() {
    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h = remote_only(&mut remote).await;
    for stream in [false, true] {
        let r = h.chat(&chat_body(stream)).send().await.unwrap();
        assert_eq!(r.status(), 200);
        let rid = rid_of(&r);
        if stream {
            check_stream(&SseReader::new(r).rest().await);
        } else {
            let _: Value = r.json().await.unwrap();
        }
        assert_remote_status(&h, &rid).await;
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    let cancels = remote.stub.cancels();
    assert!(cancels.is_empty(), "{cancels:?}");
}

/// H2: a protocol violation of a streamed remote work (an event above 64 KiB, an offset gap, the end without a
/// terminal, an unparsable event; the stub commits it at the second event) fails the stream
/// `backend_protocol_error` and cancels the work once.
#[tokio::test(flavor = "multi_thread")]
async fn a_stream_violation_fails_the_request_and_cancels_the_work() {
    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h = remote_only(&mut remote).await;
    for violation in [
        Violation::Oversize,
        Violation::OffsetGap,
        Violation::EndWithoutTerminal,
        Violation::Undecodable,
    ] {
        remote.stub.set_violation(violation);
        let r = h.chat(&chat_body(true)).send().await.unwrap();
        assert_eq!(r.status(), 200, "{violation:?}");
        let rid = rid_of(&r);
        let events = SseReader::new(r).rest().await;
        let terminal = check_stream(&events);
        assert_eq!(
            terminal["error"]["code"], "backend_protocol_error",
            "{violation:?}: {terminal}"
        );
        let status = h.terminal(&rid).await;
        assert_eq!(status["state"], "failed", "{violation:?}: {status}");
        assert_eq!(
            status["reason"], "backend_protocol_error",
            "{violation:?}: {status}"
        );
        assert_one_cancel(&remote.stub, &rid).await;
    }
}

/// H2: a terminal reason of the remote work (`worker_restart`, `model_generation_superseded`) reaches the client
/// as that error code, 503, and is the request's terminal reason.
#[tokio::test(flavor = "multi_thread")]
async fn a_terminal_reason_reaches_the_client_as_its_code() {
    // The stub serves no chat: each StreamWork ends as scripted.
    let fed = Federation::new();
    let mut stub = StubAgent::start(&fed, A, PRINCIPAL_A, 1).await;
    let mut config = fed.toml(TIMINGS, &[]);
    config.push_str(&worker_toml(&stub, 4));
    let h = Harness::start_federated_unready(unready_local(), &config, fast_tuning()).await;
    stub.link(&fed, HEARTBEAT);
    until_remote_ready(&h).await;
    for reason in [wire::WORKER_RESTART, wire::MODEL_GENERATION_SUPERSEDED] {
        stub.set_outcome(WorkOutcome::End(reason));
        let r = h.chat(&chat_body(false)).send().await.unwrap();
        assert_eq!(r.status(), 503, "{reason}");
        let rid = rid_of(&r);
        let body: Value = r.json().await.unwrap();
        assert_eq!(body["error"]["code"], reason, "{body}");
        assert_eq!(body["cere"]["reason"], reason, "{body}");
        let status = h.terminal(&rid).await;
        assert_eq!(status["state"], "failed", "{status}");
        assert_eq!(status["reason"], reason, "{status}");
    }
}

/// Q14: a client body exactly at `config::REMOTE_BODY_MAX` (the federated harness's `api.max_request_body_bytes`)
/// is granted to the remote worker and served: its SubmitWork, the re-serialized body plus authority, ids and work
/// unit, stays under `wire::INFERENCE_MESSAGE_MAX`.
#[tokio::test(flavor = "multi_thread")]
async fn a_body_at_the_federation_bound_fits_one_submit_work() {
    use prost::Message;

    // The stub serves no chat: each StreamWork completes with a 1-token unary response (`finish_reason` `length`),
    // so the request asks for one token.
    let fed = Federation::new();
    let mut stub = StubAgent::start(&fed, A, PRINCIPAL_A, 1).await;
    let mut config = fed.toml(TIMINGS, &[]);
    config.push_str(&worker_toml(&stub, 4));
    let h = Harness::start_federated_unready(unready_local(), &config, fast_tuning()).await;
    stub.link(&fed, HEARTBEAT);
    until_remote_ready(&h).await;

    let bound = cere_runtime::config::REMOTE_BODY_MAX as usize;
    let mut body = chat_body(false);
    body["max_completion_tokens"] = json!(1);
    let pad = bound - serde_json::to_vec(&body).unwrap().len() + "hello there".len();
    body["messages"][0]["content"] = json!("x".repeat(pad));
    assert_eq!(serde_json::to_vec(&body).unwrap().len(), bound);
    let r = h.chat(&body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let rid = rid_of(&r);
    assert_one_accept(&stub, &rid);
    let submit = stub
        .submits()
        .into_iter()
        .find(|s| s.request_id == rid)
        .unwrap();
    assert!(submit.bounded_inference_input_json.len() >= bound);
    assert!(submit.encoded_len() <= wire::INFERENCE_MESSAGE_MAX);
    assert_remote_status(&h, &rid).await;

    // One byte more is refused at HTTP admission, before any grant.
    body["messages"][0]["content"] = json!("x".repeat(pad + 1));
    let r = h.chat(&body).send().await.unwrap();
    assert_eq!(r.status(), 413);
}

/// A federated `cered` whose only worker taking requests will be the linked remote `A` with cap 1: the local
/// worker is drained. Orphan holds poll every second (the minimum) and terminals are retained `retention_ms`.
async fn remote_holding(remote: &mut Remote, retention_ms: u64) -> Harness {
    let ledger = format!("terminal_retention_ms = {retention_ms}\norphan_load_poll_ms = 1000");
    let h = Harness::start_federated_with(
        MockConfig::default(),
        &[("terminal_retention_ms = 3600000", &ledger)],
        &remote.config(1),
        fast_tuning(),
    )
    .await;
    let local = h.app.replicas()[0].worker_id().to_owned();
    h.app.set_drain_intent(&local, true).unwrap();
    remote.link();
    until_remote_ready(&h).await;
    h
}

/// A streamed request on a [`generating`] mock, cancelled through the endpoint after its first two events; its
/// rid once the stream ended `request_cancelled`.
async fn cancelled_mid_stream(h: &Harness) -> String {
    let r = h.chat(&long_body(true)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let rid = rid_of(&r);
    let mut sse = SseReader::new(r);
    let mut events = vec![sse.next().await.unwrap(), sse.next().await.unwrap()];
    cancel(h, &rid).await;
    events.extend(sse.rest().await);
    let terminal = check_stream(&events);
    assert_eq!(terminal["error"]["code"], "request_cancelled", "{terminal}");
    rid
}

/// The journaled `orphan_released` records of `rid`.
fn orphan_releases(h: &Harness, rid: &str) -> Vec<Value> {
    journal(h.state_dir.path())
        .into_iter()
        .filter(|r| r["type"] == "orphan_released" && r["rid"] == rid)
        .collect()
}

/// H2/P7: a cancelled remote work holds its slot, charged to the worker, until the agent reports the operation
/// `cleanup_acknowledged`: with the worker's cap at 1 the next request stays queued and the stats show the held
/// slot; once cleanup is acknowledged the next poll journals the release and the queued request is served.
#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_remote_work_holds_its_slot_until_cleanup_acknowledged() {
    let mut remote = Remote::start(generating(), |stub| stub.hold_cleanup(true)).await;
    let h = remote_holding(&mut remote, 3_600_000).await;
    let rid = cancelled_mid_stream(&h).await;
    assert_one_cancel(&remote.stub, &rid).await;
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "cancelled", "{status}");
    let submits = remote.stub.submits().len();
    let next = tokio::spawn(h.chat(&chat_body(false)).send());
    // Two polls see the cleanup unacknowledged.
    tokio::time::sleep(Duration::from_millis(2_200)).await;
    assert!(!next.is_finished(), "the next request was answered");
    assert_eq!(
        remote.stub.submits().len(),
        submits,
        "the next request was granted"
    );
    assert_eq!(h.app.admission_snapshot().queued, 1);
    let held = remote_cap(&h).await;
    assert_eq!(held["cap"], 1, "{held}");
    assert_eq!(held["inflight"], 1, "the held slot: {held}");
    assert!(orphan_releases(&h, &rid).is_empty());
    remote.stub.hold_cleanup(false);
    let r = tokio::time::timeout(Duration::from_secs(5), next)
        .await
        .expect("served once cleanup was acknowledged")
        .unwrap()
        .unwrap();
    assert_eq!(r.status(), 200);
    let released = orphan_releases(&h, &rid);
    assert_eq!(released.len(), 1, "{released:?}");
    assert_eq!(released[0]["cause"], "cleanup_acknowledged", "{released:?}");
}

/// P7: a hold whose cleanup is never acknowledged ends at the request's retention bound (`admitted_ms +
/// terminal_retention_ms`): the queued request is served then and the release is journaled `retention`.
#[tokio::test(flavor = "multi_thread")]
async fn an_unacknowledged_hold_is_released_at_the_retention_bound() {
    let mut remote = Remote::start(generating(), |stub| stub.hold_cleanup(true)).await;
    let h = remote_holding(&mut remote, 1_000).await;
    let admitted = now_ms();
    let rid = cancelled_mid_stream(&h).await;
    let r = tokio::time::timeout(Duration::from_secs(5), h.chat(&chat_body(false)).send())
        .await
        .expect("served at the bound")
        .unwrap();
    assert_eq!(r.status(), 200);
    assert!(
        now_ms() >= admitted + 1_000,
        "served before the retention bound"
    );
    let op = remote.stub.op("a", &rid).expect("the Submit op");
    assert!(!op.cleanup_acknowledged, "{op:?}");
    let released = orphan_releases(&h, &rid);
    assert_eq!(released.len(), 1, "{released:?}");
    assert_eq!(released[0]["cause"], "retention", "{released:?}");
}

/// P7: a remote work that reached its natural terminal holds nothing: with the worker's cap at 1 and cleanup
/// never acknowledged, each next request is served at once, unary or streamed.
#[tokio::test(flavor = "multi_thread")]
async fn a_natural_completion_holds_no_slot() {
    let mut remote = Remote::start(MockConfig::default(), |stub| stub.hold_cleanup(true)).await;
    let h = remote_holding(&mut remote, 3_600_000).await;
    for stream in [false, true, false] {
        let r = tokio::time::timeout(Duration::from_secs(5), h.chat(&chat_body(stream)).send())
            .await
            .expect("served at once")
            .unwrap();
        assert_eq!(r.status(), 200);
        let rid = rid_of(&r);
        if stream {
            check_stream(&SseReader::new(r).rest().await);
        } else {
            let _: Value = r.json().await.unwrap();
        }
        assert_remote_status(&h, &rid).await;
        assert!(orphan_releases(&h, &rid).is_empty());
    }
}

/// C1 (D2/D5) remote path: SGLang's tool schema 400 on the remote worker ends the op with the agent reason
/// `invalid_value`, which reaches the client as `invalid_value` `tools` without backend text: unary as HTTP 400,
/// streamed (the remote stream is open before the agent's first event) as the terminal error event.
#[tokio::test(flavor = "multi_thread")]
async fn c1_a_remote_tool_schema_refusal_is_invalid_value() {
    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h = remote_only(&mut remote).await;
    remote.mock.set_fault(Fault::Reject {
        code: 400,
        message: "Tool 0 function has invalid 'parameters' schema: secret".to_owned(),
    });
    let tools =
        json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]);
    let mut body = chat_body(false);
    body["tools"] = tools.clone();
    let r = h.chat(&body).send().await.unwrap();
    assert_eq!(r.status(), 400);
    let error: Value = r.json().await.unwrap();
    assert_eq!(error["error"]["code"], "invalid_value", "{error}");
    assert_eq!(error["error"]["param"], "tools", "{error}");
    assert!(!error.to_string().contains("secret"), "{error}");

    let mut body = chat_body(true);
    body["tools"] = tools;
    let r = h.chat(&body).send().await.unwrap();
    let events = SseReader::new(r).rest().await;
    let error = events
        .iter()
        .rev()
        .find_map(|e| match e {
            Event::Data(v) if v.get("error").is_some() => Some(v.clone()),
            _ => None,
        })
        .expect("a terminal error event");
    assert_eq!(error["error"]["code"], "invalid_value", "{error}");
    assert_eq!(error["error"]["param"], "tools", "{error}");
    assert!(!error.to_string().contains("secret"), "{error}");
}

/// C2 (M2.8 D6) remote path: SGLang's grammar compile 400 on the remote worker ends the op with the agent reason
/// `grammar_compile_failed`, which reaches the client as `invalid_value` on the constraint field with fixed text:
/// unary as HTTP 400, streamed as the terminal error event.
#[tokio::test(flavor = "multi_thread")]
async fn c2_a_remote_grammar_compile_failure_is_invalid_value() {
    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h = remote_only(&mut remote).await;
    remote.mock.set_fault(Fault::Reject {
        code: 400,
        message: "Failed to compile json grammar: secret".to_owned(),
    });
    for (stream, field, value) in [
        (false, "response_format", json!({"type": "json_object"})),
        (true, "regex", json!("[0-9]+")),
    ] {
        let mut body = chat_body(stream);
        body[field] = value;
        let r = h.chat(&body).send().await.unwrap();
        let error = if stream {
            let events = SseReader::new(r).rest().await;
            events
                .iter()
                .rev()
                .find_map(|e| match e {
                    Event::Data(v) if v.get("error").is_some() => Some(v.clone()),
                    _ => None,
                })
                .expect("a terminal error event")
        } else {
            assert_eq!(r.status(), 400);
            r.json().await.unwrap()
        };
        assert_eq!(error["error"]["code"], "invalid_value", "{error}");
        assert_eq!(error["error"]["param"], field, "{error}");
        assert!(!error.to_string().contains("secret"), "{error}");
    }
}

/// D2 "on a request with tools" (M24 Z2 R1-12), remote path: a tool-free request whose remote work ends
/// `invalid_value` is an ordinary backend failure (502 `backend_error`), not a `tools` refusal.
#[tokio::test(flavor = "multi_thread")]
async fn c1_a_remote_invalid_value_without_tools_is_a_backend_error() {
    let fed = Federation::new();
    let mut stub = StubAgent::start(&fed, A, PRINCIPAL_A, 1).await;
    let mut config = fed.toml(TIMINGS, &[]);
    config.push_str(&worker_toml(&stub, 4));
    let h = Harness::start_federated_unready(unready_local(), &config, fast_tuning()).await;
    stub.link(&fed, HEARTBEAT);
    until_remote_ready(&h).await;
    stub.set_outcome(WorkOutcome::End("invalid_value"));
    let r = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 502);
    let error: Value = r.json().await.unwrap();
    assert_eq!(error["error"]["code"], "backend_error", "{error}");
}

/// D1 remote path (M24 Z2 R1-06): an agent whose backend does not run `qwen3_coder` ends a tools Submit
/// `unsupported_parameter`; the client gets 400 `unsupported_parameter` naming `tools`, not a 502.
#[tokio::test(flavor = "multi_thread")]
async fn d1_a_remote_tools_refusal_is_unsupported_parameter() {
    let fed = Federation::new();
    let mut stub = StubAgent::start(&fed, A, PRINCIPAL_A, 1).await;
    let mut config = fed.toml(TIMINGS, &[]);
    config.push_str(&worker_toml(&stub, 4));
    let h = Harness::start_federated_unready(unready_local(), &config, fast_tuning()).await;
    stub.link(&fed, HEARTBEAT);
    until_remote_ready(&h).await;
    stub.set_outcome(WorkOutcome::End("unsupported_parameter"));
    let mut body = chat_body(false);
    body["tools"] = json!([{"type": "function", "function": {"name": "f"}}]);
    let r = h.chat(&body).send().await.unwrap();
    assert_eq!(r.status(), 400);
    let rid = rid_of(&r);
    let error: Value = r.json().await.unwrap();
    assert_eq!(error["error"]["code"], "unsupported_parameter", "{error}");
    assert_eq!(error["error"]["param"], "tools", "{error}");
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "failed", "{status}");

    // The remote stream opens before the agent's terminal event: the refusal is the SSE error event.
    let mut body = chat_body(true);
    body["tools"] = json!([{"type": "function", "function": {"name": "f"}}]);
    let r = h.chat(&body).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let events = SseReader::new(r).rest().await;
    let error = events
        .iter()
        .rev()
        .find_map(|e| match e {
            Event::Data(v) if v.get("error").is_some() => Some(v.clone()),
            _ => None,
        })
        .expect("a terminal error event");
    assert_eq!(error["error"]["code"], "unsupported_parameter", "{error}");
    assert_eq!(error["error"]["param"], "tools", "{error}");
}

/// C2 (D4/D5) remote path: every Submit attempt of a tools request carries the client's `tools` bytes verbatim -
/// key order, `1.0`, `1E5`, 2^70, `\u00e9` - in its input: the first, refused `queue_full` (the request waits in
/// admission again and is re-submitted), the re-submission after the next grant and its retry after a lost reply.
#[tokio::test(flavor = "multi_thread")]
async fn c2_every_remote_submit_carries_the_tools_bytes() {
    const TOOLS: &str = r#"[{"type":"function","function":{"name":"f","parameters":{"type":"object","x-a":1.0,"x-b":1E5,"x-c":1180591620717411303424,"x-d":"Z\u00e9"}}}]"#;
    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h = remote_only(&mut remote).await;
    remote
        .stub
        .refuse_next_submit(Code::ResourceExhausted, wire::QUEUE_FULL);
    remote.stub.lose_submit_replies(1);
    let before = remote.stub.submit_attempts().len();
    let body = format!(
        r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}],"max_completion_tokens":4,"tools":{TOOLS}}}"#
    );
    let r = h
        .post("/v1/chat/completions", TOKEN_A)
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let rid = rid_of(&r);
    let attempts: Vec<_> = remote
        .stub
        .submit_attempts()
        .split_off(before)
        .into_iter()
        .filter(|s| s.request_id == rid)
        .collect();
    assert_eq!(attempts.len(), 3, "refused, lost reply, retried");
    for submit in attempts {
        let input = String::from_utf8(submit.bounded_inference_input_json).unwrap();
        assert!(input.contains(&format!(r#""tools":{TOOLS}"#)), "{input}");
    }
}
