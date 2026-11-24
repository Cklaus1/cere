//! M3.5 H2 orphans of remote work: a Submit that reached the agent may hold the agent's resources even when its
//! receipt came too late, so its slot stays charged until the agent reports cleanup; the stub agent keeps its
//! heartbeat sequence across a same-epoch reconnect like the real agent; and an in-flight remote request whose
//! worker turns Lost fails `worker_lost`, one whose worker registers a higher epoch `worker_epoch_superseded`,
//! and the new incarnation's cleanup report releases its slot.
//!
//! Parking (H2): a work stream that breaks on the transport (`UNAVAILABLE worker_restart`) parks the request
//! (no output, slot charged, client stream open) until the worker turns Lost (`worker_lost`), registers a higher
//! epoch (`worker_epoch_superseded`), reconnects at the same epoch (`backend_protocol_error`) or the request's
//! deadline passes (`deadline_exceeded`); any other status fails at once and cancels the work.
mod support;

use std::future::Future;
use std::io::Write;
use std::time::Duration;

use cere_mock_sglang::{MockConfig, MockHandle, ServerInfo};
use cere_runtime::fed::registry::RemoteState;
use cere_runtime::worker::WorkerState;
use serde_json::Value;
use support::stub_agent::{StubAgent, Violation, WorkOutcome};
use support::*;
use tonic::Code;

const A: &str = "worker-a";
const PRINCIPAL_A: &str = "cere-worker-a";
/// Federation timings of the tests that age the worker: an unlinked stub reads Lost within 3 s.
const AGING: &str =
    "heartbeat_interval_ms = 100\nadmission_lease_ms = 1000\nsuspect_after_ms = 2000\n\
                       lost_after_ms = 3000\n";
/// Federation timings of the tests whose worker never ages: a lease no host stall outlasts. Under [`AGING`] a
/// stub heartbeat starved for 1 s (a loaded host) expires the lease, cered's view or the stub's own, so the next
/// heartbeat returns the worker to Joining (Q7a) and a parked stream ends `backend_protocol_error`; starved for
/// 3 s, the worker is Lost before the test used it.
const STEADY: &str =
    "heartbeat_interval_ms = 100\nadmission_lease_ms = 60000\nsuspect_after_ms = 120000\n\
                       lost_after_ms = 180000\n";
/// Federation timings of the tests that wait for the worker to turn Lost: an unlinked stub reads Lost within 3 s,
/// and the lease runs almost as long, so only a stub heartbeat starved for 2.8-3 s (not 1-3 s as under
/// [`AGING`]) expires the lease without the worker turning Lost, returning it to Joining and ending a parked
/// stream `backend_protocol_error` before the stub was unlinked.
const LOSING: &str =
    "heartbeat_interval_ms = 100\nadmission_lease_ms = 2800\nsuspect_after_ms = 2900\n\
                       lost_after_ms = 3000\n";
/// The stub agent's heartbeat period.
const HEARTBEAT: Duration = Duration::from_millis(50);
/// The deadline of a request whose test waits for another terminal reason (the worker Lost, a higher epoch, a
/// same-epoch reconnect, a protocol violation, a cleanup report): if that reason never comes, the request ends
/// `deadline_exceeded` here, so its outcome tells the paths apart, not a wall-clock bound a loaded host can miss.
const OUTCOME_DEADLINE_MS: &str = "30000";
/// Bounds a wait for the end of a request that [`OUTCOME_DEADLINE_MS`] (or a shorter budget) ends: it only
/// catches a request that never ends.
const ENDED: Duration = Duration::from_secs(60);

/// The remote side of a test: the stub agent serving chat on its own mock, or (no mock) a stub whose every
/// `StreamWork` ends as [`StubAgent::set_outcome`] scripts.
struct Remote {
    fed: Federation,
    stub: StubAgent,
    _mock: Option<MockHandle>,
}

impl Remote {
    async fn start(mock: MockConfig, script: impl FnOnce(&StubAgent)) -> Remote {
        let fed = Federation::new();
        let stub = StubAgent::start(&fed, A, PRINCIPAL_A, 1).await;
        let mock = cere_mock_sglang::MockSglang::start(mock).await.unwrap();
        stub.serve_chat(&mock.base_url());
        script(&stub);
        Remote {
            fed,
            stub,
            _mock: Some(mock),
        }
    }

    /// The stub agent serving no chat: each `StreamWork` ends as scripted (a 1-token unary response by default).
    async fn scripted() -> Remote {
        let fed = Federation::new();
        let stub = StubAgent::start(&fed, A, PRINCIPAL_A, 1).await;
        Remote {
            fed,
            stub,
            _mock: None,
        }
    }

    fn config(&self, timings: &str, max_inflight: u64) -> String {
        let mut text = self.fed.toml(timings, &[]);
        text.push_str(&format!(
            "[[federation.workers]]\nworker_id = \"{A}\"\nendpoint = \"https://{}\"\n\
             principal = \"{PRINCIPAL_A}\"\nmax_inflight = {max_inflight}\n\n[federation.workers.expect]\n\
             served_model_name = \"{MODEL}\"\ntp_size = 1\nkv_cache_dtype = \"fp8_e4m3\"\n\n",
            self.stub.addr()
        ));
        text
    }
}

fn state(h: &Harness) -> RemoteState {
    h.app
        .remote_workers()
        .expect("[federation] configured")
        .get(A, now_ms())
        .expect("configured remote worker")
}

async fn until_remote_ready(h: &Harness) {
    readyz_until(h, "worker-a ready", |b| b["workers"][A]["state"] == "ready").await;
}

/// A federated `cered` with `timings` ([`AGING`] or [`STEADY`]) whose only worker taking requests is the linked
/// remote `A` with cap 1 (the local worker is drained); orphan holds poll every second and terminals are retained
/// an hour.
async fn remote_holding(remote: &mut Remote, timings: &str) -> Harness {
    let ledger = "terminal_retention_ms = 3600000\norphan_load_poll_ms = 1000";
    let h = Harness::start_federated_with(
        MockConfig::default(),
        &[("terminal_retention_ms = 3600000", ledger)],
        &remote.config(timings, 1),
        fast_tuning(),
    )
    .await;
    let local = h.app.replicas()[0].worker_id().to_owned();
    h.app.set_drain_intent(&local, true).unwrap();
    remote.stub.link(&remote.fed, HEARTBEAT);
    until_remote_ready(&h).await;
    h
}

fn orphan_releases(h: &Harness, rid: &str) -> Vec<Value> {
    journal(h.state_dir.path())
        .into_iter()
        .filter(|r| r["type"] == "orphan_released" && r["rid"] == rid)
        .collect()
}

/// The budget of a [`late_receipt`] chat (`X-Cere-Deadline-Ms`).
const LATE_BUDGET_MS: &str = "500";

/// How many chats [`late_receipt`] sends at most.
const LATE_CHATS: usize = 10;

/// Whether the slot of every request is free: none holds a permit (a held Submit receipt or an H2 hold does) or
/// waits for one.
fn idle(h: &Harness) -> bool {
    let snapshot = h.app.admission_snapshot();
    snapshot.inflight == 0 && snapshot.queued == 0
}

/// A unary chat whose Submit the agent journaled, its receipt held (`hold_submit_replies`), answered (not 200)
/// once its [`LATE_BUDGET_MS`] budget ran out: its rid. The budget runs from the chat's arrival, so it also covers
/// the fsyncs of the request's `admitted` and `dispatched` journal records: a host whose disk stalls them past the
/// budget ends the chat before its Submit goes out (or before the agent journaled it). Such a chat is checked (it
/// ended unserved, then its slot is free again: no Submit held it) and the next one is sent, up to [`LATE_CHATS`]
/// chats. A Submit the agent journals only after the chat was answered is that chat's too: its receipt also came
/// after the budget.
async fn late_receipt(h: &Harness, stub: &StubAgent) -> String {
    for _ in 0..LATE_CHATS {
        let before = stub.accepts().len();
        let r = bounded(
            h,
            stub,
            "late receipt: no answer at the budget",
            h.chat(&chat_body(false))
                .header("X-Cere-Deadline-Ms", LATE_BUDGET_MS)
                .send(),
        )
        .await
        .unwrap();
        assert_ne!(r.status(), 200);
        let rid = r.headers()["x-cere-request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let journaled = || !stub.accepts()[before..].is_empty();
        if !journaled() {
            let status = h.terminal(&rid).await;
            assert_ne!(status["state"], "completed", "{status}");
            bounded(
                h,
                stub,
                "late receipt: an expired chat's slot is free",
                async {
                    while !journaled() && !idle(h) {
                        tokio::time::sleep(Duration::from_millis(20)).await;
                    }
                },
            )
            .await;
        }
        if journaled() {
            assert_eq!(
                stub.accepts()[before..].to_vec(),
                vec![(rid.clone(), true)],
                "the agent journaled the Submit"
            );
            return rid;
        }
    }
    panic!("the agent journaled the Submit of none of {LATE_CHATS} chats");
}

/// H2/DB3: a Submit whose receipt arrives after the request's budget ran out was still journaled by the agent:
/// its slot stays charged (cap 1: the next request stays queued) until the agent reports the operation's cleanup,
/// then the hold is released and the queued request is served (within its [`OUTCOME_DEADLINE_MS`]).
#[tokio::test(flavor = "multi_thread")]
async fn a_late_receipt_holds_the_slot_until_cleanup() {
    let mut remote = Remote::start(MockConfig::default(), |stub| stub.hold_cleanup(true)).await;
    let h = remote_holding(&mut remote, STEADY).await;
    remote.stub.hold_submit_replies(true);
    let rid = late_receipt(&h, &remote.stub).await;
    remote.stub.hold_submit_replies(false);
    assert!(remote.stub.op("a", &rid).is_some());
    let submits = remote.stub.submits().len();
    let next = tokio::spawn(
        h.chat(&chat_body(false))
            .header("X-Cere-Deadline-Ms", OUTCOME_DEADLINE_MS)
            .send(),
    );
    until(
        &h,
        &remote.stub,
        "late receipt: the next request queued",
        || {
            h.app.admission_snapshot().queued == 1
                || next.is_finished()
                || remote.stub.submits().len() != submits
        },
    )
    .await;
    // Two polls see the cleanup unacknowledged.
    let polls = || {
        remote
            .stub
            .calls()
            .iter()
            .filter(|c| **c == "GetOperation")
            .count()
    };
    let seen = polls();
    until(&h, &remote.stub, "late receipt: two orphan polls", || {
        polls() >= seen + 2
    })
    .await;
    assert!(!next.is_finished(), "the next request was answered");
    assert_eq!(
        remote.stub.submits().len(),
        submits,
        "the next request was granted"
    );
    assert_eq!(h.app.admission_snapshot().queued, 1);
    assert!(orphan_releases(&h, &rid).is_empty());
    remote.stub.hold_cleanup(false);
    let r = tokio::time::timeout(ENDED, next)
        .await
        .expect("the next request ended")
        .unwrap()
        .unwrap();
    assert_eq!(r.status(), 200);
    let released = orphan_releases(&h, &rid);
    assert_eq!(released.len(), 1, "{released:?}");
    assert_eq!(released[0]["cause"], "cleanup_acknowledged", "{released:?}");
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

/// The stub keeps its heartbeat sequence within an epoch: after the link stops until the lease view expired, a
/// relink at the same epoch registers again and its heartbeats are accepted (the worker is ready again).
#[tokio::test(flavor = "multi_thread")]
async fn a_same_epoch_relink_is_ready_again() {
    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h =
        Harness::start_federated_unready(unready_local(), &remote.config(AGING, 4), fast_tuning())
            .await;
    remote.stub.link(&remote.fed, HEARTBEAT);
    until_remote_ready(&h).await;
    remote.stub.unlink();
    until_lease_expired(&h).await;
    remote.stub.link(&remote.fed, HEARTBEAT);
    until_remote_ready(&h).await;
}

/// Waits (the stub's link stopped) until the worker's lease view expired: it no longer reads Ready.
async fn until_lease_expired(h: &Harness) {
    for _ in 0..10 {
        let current = state(h);
        if current.state != WorkerState::Ready {
            break;
        }
        let until = current.lease_until_ms.expect("a lease view while ready");
        tokio::time::sleep(Duration::from_millis(until.saturating_sub(now_ms()) + 10)).await;
    }
    assert_ne!(state(h).state, WorkerState::Ready);
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

/// The rid of the first Submit after the first `before` ones, once the stub journaled it.
async fn submitted_after(h: &Harness, stub: &StubAgent, before: usize) -> String {
    until(h, stub, "no Submit reached the stub", || {
        stub.submits().len() > before
    })
    .await;
    stub.submits()[before].request_id.clone()
}

/// Polls `ready` every 5 ms until it holds, within [`STEP`] (a hang reports as [`bounded`] does).
async fn until(h: &Harness, stub: &StubAgent, stage: &str, mut ready: impl FnMut() -> bool) {
    bounded(h, stub, stage, async {
        while !ready() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
}

/// H2/D5: the worker turns Lost while the request's Submit waits for its receipt: the request fails
/// `worker_lost` (503) at once, which is its terminal reason.
#[tokio::test(flavor = "multi_thread")]
async fn lost_while_the_submit_is_in_flight_fails_worker_lost() {
    let mut remote = Remote::start(MockConfig::default(), |_| {}).await;
    let h = remote_holding(&mut remote, LOSING).await;
    remote.stub.hold_submit_replies(true);
    let before = remote.stub.submits().len();
    let call = tokio::spawn(
        h.chat(&chat_body(false))
            .header("X-Cere-Deadline-Ms", OUTCOME_DEADLINE_MS)
            .send(),
    );
    let rid = submitted_after(&h, &remote.stub, before).await;
    remote.stub.unlink();
    let r = tokio::time::timeout(ENDED, call)
        .await
        .expect("the request ended")
        .unwrap()
        .unwrap();
    assert_eq!(r.status(), 503);
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["error"]["code"], "worker_lost", "{body}");
    assert_eq!(body["cere"]["reason"], "worker_lost", "{body}");
    assert_eq!(state(&h).state, WorkerState::Lost);
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], "worker_lost", "{status}");
}

/// A streamed request (deadline [`OUTCOME_DEADLINE_MS`]) on a [`generating`] mock, after its first two events
/// (the second carries output); its rid and the events read so far.
async fn streaming(h: &Harness) -> (String, SseReader, Vec<Event>) {
    let r = h
        .chat(&long_body(true))
        .header("X-Cere-Deadline-Ms", OUTCOME_DEADLINE_MS)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let rid = r.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let mut sse = SseReader::new(r);
    let events = vec![sse.next().await.unwrap(), sse.next().await.unwrap()];
    (rid, sse, events)
}

/// The rest of a stream, checked: its terminal event. The request's deadline ends it; [`ENDED`] only catches
/// a stream that never ends.
async fn stream_terminal(mut sse: SseReader, mut events: Vec<Event>) -> Value {
    let rest = tokio::time::timeout(ENDED, sse.rest())
        .await
        .expect("the stream ended");
    events.extend(rest);
    check_stream(&events)
}

/// H2/D5: the worker turns Lost while its work streams (the work itself keeps running): the stream ends with
/// the error `worker_lost`, the request's terminal reason.
#[tokio::test(flavor = "multi_thread")]
async fn lost_while_streaming_fails_worker_lost() {
    let mut remote = Remote::start(generating(), |_| {}).await;
    let h = remote_holding(&mut remote, LOSING).await;
    let (rid, sse, events) = streaming(&h).await;
    remote.stub.unlink();
    let terminal = stream_terminal(sse, events).await;
    assert_eq!(terminal["error"]["code"], "worker_lost", "{terminal}");
    assert_eq!(state(&h).state, WorkerState::Lost);
    assert_eq!(terminal["cere"]["state"], "failed", "{terminal}");
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], "worker_lost", "{status}");
}

/// H2/D5: the agent restarts at a higher epoch and registers it while a work streams: the stream ends with the
/// error `worker_epoch_superseded`, the request's terminal reason. (The new incarnation's reconciliation would
/// cancel the old work; it waits at `ListOperations`, so only the registration ends the stream.) The slot stays
/// held until that reconciliation cancelled the old work and the agent reports it `cleanup_acknowledged` at the
/// new epoch: the hold's poll, under the worker's current authority, then releases it long before the retention
/// bound. Cap 1: the new incarnation's smoke waits for that slot, so the worker is ready again (and serves the
/// next request) only after the release.
#[tokio::test(flavor = "multi_thread")]
async fn a_higher_epoch_while_streaming_fails_worker_epoch_superseded() {
    let mut remote = Remote::start(generating(), |_| {}).await;
    let h = remote_holding(&mut remote, STEADY).await;
    let (rid, sse, events) = streaming(&h).await;
    remote.stub.hold_list(true);
    remote.stub.restart(2);
    let terminal = stream_terminal(sse, events).await;
    assert_eq!(state(&h).epoch, Some(2));
    assert_eq!(
        terminal["error"]["code"], "worker_epoch_superseded",
        "{terminal}"
    );
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], "worker_epoch_superseded", "{status}");
    assert!(orphan_releases(&h, &rid).is_empty());
    remote.stub.hold_list(false);
    until_remote_ready(&h).await;
    let op = remote.stub.op("a", &rid).expect("the old work");
    assert!(op.cleanup_acknowledged, "{op:?}");
    let released = orphan_releases(&h, &rid);
    assert_eq!(released.len(), 1, "{released:?}");
    assert_eq!(released[0]["cause"], "cleanup_acknowledged", "{released:?}");
    let r = h.chat(&chat_body(false)).send().await.unwrap();
    assert_eq!(r.status(), 200);
}

/// How long a parked stream is watched for silence.
const PARKED: Duration = Duration::from_millis(500);

/// The bound of each step of [`parked`] before its stream is silent, and of each [`until`] wait. It only catches a
/// hang: without it a stuck step would hold the test until the runner kills it.
const STEP: Duration = Duration::from_secs(30);

/// Awaits `step` within [`STEP`]. On a timeout `stage`, the worker's registry state, the admission snapshot and the
/// stub's calls go to stderr first, unbuffered and outside the test's output capture (a runtime that cannot shut
/// down never reports the panic), then it panics naming `stage`.
async fn bounded<T>(
    h: &Harness,
    stub: &StubAgent,
    stage: &str,
    step: impl Future<Output = T>,
) -> T {
    match tokio::time::timeout(STEP, step).await {
        Ok(out) => out,
        Err(_) => {
            let report = format!(
                "{stage} within {STEP:?}: worker {:?}; admission {:?}; stub calls {:?}",
                state(h),
                h.app.admission_snapshot(),
                stub.calls()
            );
            let _ = writeln!(std::io::stderr(), "coord_orphan: {report}");
            panic!("{report}");
        }
    }
}

/// What [`try_parked`] saw of a streamed request whose work stream breaks.
enum Park {
    /// Silent for [`PARKED`], then still running: its rid, its stream and the events read so far.
    Parked(String, SseReader, Vec<Event>),
    /// Ended before it was seen parked: its rid and all its events.
    Ended(String, Vec<Event>),
}

/// A streamed request whose work stream breaks on the transport (`UNAVAILABLE worker_restart`, the agent's
/// shutdown status): after its `accepted` event the client stream stays open and silent while the request runs
/// (it parks). `deadline_ms` sets its budget.
async fn try_parked(h: &Harness, stub: &StubAgent, deadline_ms: &str) -> Park {
    stub.set_outcome(WorkOutcome::Break);
    let request = h
        .chat(&chat_body(true))
        .header("X-Cere-Deadline-Ms", deadline_ms);
    let r = bounded(h, stub, "parked: no response to the chat", request.send())
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let rid = r.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let mut sse = SseReader::new(r);
    let first = bounded(h, stub, "parked: no first event", sse.next()).await;
    let mut events = vec![first.unwrap()];
    if let Ok(next) = tokio::time::timeout(PARKED, sse.next()).await {
        events.extend(next);
        events.extend(rest(sse).await);
        return Park::Ended(rid, events);
    }
    let (_, status) = bounded(
        h,
        stub,
        "parked: no request status",
        h.status(&rid, TOKEN_A),
    )
    .await;
    if status["state"] != "running" {
        events.extend(rest(sse).await);
        return Park::Ended(rid, events);
    }
    Park::Parked(rid, sse, events)
}

/// The rest of `sse`, within [`ENDED`].
async fn rest(mut sse: SseReader) -> Vec<Event> {
    tokio::time::timeout(ENDED, sse.rest())
        .await
        .expect("the stream ended")
}

/// [`try_parked`] with the deadline [`OUTCOME_DEADLINE_MS`]: the request must park. Its rid and the events read
/// so far.
async fn parked(h: &Harness, stub: &StubAgent) -> (String, SseReader, Vec<Event>) {
    match try_parked(h, stub, OUTCOME_DEADLINE_MS).await {
        Park::Parked(rid, sse, events) => (rid, sse, events),
        Park::Ended(rid, events) => panic!("the parked stream {rid} went on: {events:?}"),
    }
}

/// Waits for the `CancelWork` of `rid`, then checks it stays the only one (`<rid>/cancel/1`).
async fn assert_one_cancel(h: &Harness, stub: &StubAgent, rid: &str) {
    let cancels = || -> Vec<_> {
        stub.cancels()
            .into_iter()
            .filter(|c| c.request_id == rid)
            .collect()
    };
    until(h, stub, "no CancelWork of the request", || {
        !cancels().is_empty()
    })
    .await;
    // Room for a second one to arrive.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let cancels = cancels();
    assert_eq!(cancels.len(), 1, "{cancels:?}");
    let authority = cancels[0].authority.as_ref().unwrap();
    assert_eq!(authority.operation_id, format!("{rid}/cancel/1"));
}

/// H2: a broken work stream parks the request, its slot still charged (cap 1: the next request stays queued),
/// until the worker turns Lost: the stream then ends with the error `worker_lost`, the request's terminal reason.
#[tokio::test(flavor = "multi_thread")]
async fn a_broken_stream_parks_until_the_worker_is_lost() {
    let mut remote = Remote::scripted().await;
    let h = remote_holding(&mut remote, LOSING).await;
    let (rid, sse, events) = parked(&h, &remote.stub).await;
    let submits = remote.stub.submits().len();
    let next = tokio::spawn(h.chat(&chat_body(false)).send());
    until(&h, &remote.stub, "the next request queued", || {
        h.app.admission_snapshot().queued == 1
            || next.is_finished()
            || remote.stub.submits().len() != submits
    })
    .await;
    tokio::time::sleep(PARKED).await;
    assert!(!next.is_finished(), "the next request was answered");
    assert_eq!(h.app.admission_snapshot().queued, 1);
    assert_eq!(
        remote.stub.submits().len(),
        submits,
        "the next request was granted"
    );
    remote.stub.unlink();
    let terminal = stream_terminal(sse, events).await;
    assert_eq!(terminal["error"]["code"], "worker_lost", "{terminal}");
    assert_eq!(state(&h).state, WorkerState::Lost);
    assert_eq!(terminal["cere"]["state"], "failed", "{terminal}");
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], "worker_lost", "{status}");
    next.abort();
}

/// H2/D5: a parked request whose worker registers a higher epoch ends `worker_epoch_superseded`.
#[tokio::test(flavor = "multi_thread")]
async fn a_broken_stream_parks_until_a_higher_epoch_registers() {
    let mut remote = Remote::scripted().await;
    let h = remote_holding(&mut remote, STEADY).await;
    let (rid, sse, events) = parked(&h, &remote.stub).await;
    remote.stub.restart(2);
    let terminal = stream_terminal(sse, events).await;
    assert_eq!(state(&h).epoch, Some(2));
    assert_eq!(
        terminal["error"]["code"], "worker_epoch_superseded",
        "{terminal}"
    );
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], "worker_epoch_superseded", "{status}");
}

/// H2: a parked request whose worker reconnects at the same epoch (its lease view expired, not Lost, then its
/// first heartbeat starts a new round) ends `backend_protocol_error`: M3.5 does not resume a work stream.
#[tokio::test(flavor = "multi_thread")]
async fn a_same_epoch_reconnect_ends_a_parked_stream_backend_protocol_error() {
    let mut remote = Remote::scripted().await;
    let h = remote_holding(&mut remote, AGING).await;
    let (rid, sse, events) = parked(&h, &remote.stub).await;
    remote.stub.unlink();
    until_lease_expired(&h).await;
    assert_ne!(state(&h).state, WorkerState::Lost);
    remote.stub.link(&remote.fed, HEARTBEAT);
    let terminal = stream_terminal(sse, events).await;
    let current = state(&h);
    assert_eq!(current.epoch, Some(1));
    assert_ne!(current.state, WorkerState::Lost);
    assert_eq!(
        terminal["error"]["code"], "backend_protocol_error",
        "{terminal}"
    );
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], "backend_protocol_error", "{status}");
}

/// The budgets [`a_parked_stream_ends_at_the_request_deadline`] tries, in order. A budget that runs out before
/// its stream was seen parked (a loaded host) is checked (it ended `deadline_exceeded`) and the next, twice as
/// long, is tried.
const PARK_BUDGETS_MS: [&str; 4] = ["2000", "4000", "8000", "16000"];

/// H2: a parked request ends at its deadline like a local one, `deadline_exceeded` (never
/// `cancel_unacknowledged`), and its work is cancelled once.
#[tokio::test(flavor = "multi_thread")]
async fn a_parked_stream_ends_at_the_request_deadline() {
    let mut remote = Remote::scripted().await;
    let h = remote_holding(&mut remote, STEADY).await;
    for budget in PARK_BUDGETS_MS {
        let (rid, events) = match try_parked(&h, &remote.stub, budget).await {
            Park::Parked(rid, sse, events) => {
                let terminal = stream_terminal(sse, events).await;
                assert_eq!(terminal["error"]["code"], "deadline_exceeded", "{terminal}");
                assert_eq!(terminal["cere"]["state"], "cancelled", "{terminal}");
                let status = h.terminal(&rid).await;
                assert_eq!(status["state"], "cancelled", "{status}");
                assert_eq!(status["reason"], "deadline_exceeded", "{status}");
                assert_one_cancel(&h, &remote.stub, &rid).await;
                return;
            }
            Park::Ended(rid, events) => (rid, events),
        };
        let terminal = check_stream(&events);
        assert_eq!(terminal["error"]["code"], "deadline_exceeded", "{terminal}");
        let status = h.terminal(&rid).await;
        assert_eq!(status["reason"], "deadline_exceeded", "{status}");
        until(&h, &remote.stub, "an expired chat's slot is free", || {
            idle(&h)
        })
        .await;
    }
    panic!("no budget of {PARK_BUDGETS_MS:?} outlasted the park checks");
}

/// H2: a work stream that fails with a status outside the transport class is a protocol violation: the request
/// fails `backend_protocol_error` at once (no park) and its work is cancelled once. The outcome tells the two paths
/// apart, not a wall-clock bound a loaded host can miss: a park would end `deadline_exceeded` at
/// [`OUTCOME_DEADLINE_MS`] (under [`STEADY`] the stub's heartbeats keep the worker Ready and no reconnect can come
/// before the 60 s lease expires, so nothing else ends a parked stream).
#[tokio::test(flavor = "multi_thread")]
async fn a_non_transport_status_fails_at_once_and_cancels_the_work() {
    let mut remote = Remote::start(generating(), |_| {}).await;
    let h = remote_holding(&mut remote, STEADY).await;
    remote
        .stub
        .set_violation(Violation::Status(Code::FailedPrecondition));
    let r = h
        .chat(&long_body(true))
        .header("X-Cere-Deadline-Ms", OUTCOME_DEADLINE_MS)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let rid = r.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let events = tokio::time::timeout(ENDED, SseReader::new(r).rest())
        .await
        .expect("the stream ended");
    let terminal = check_stream(&events);
    assert_eq!(
        terminal["error"]["code"], "backend_protocol_error",
        "{terminal}"
    );
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], "backend_protocol_error", "{status}");
    assert_one_cancel(&h, &remote.stub, &rid).await;
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
