//! WORK track (m35 design note row P5, data half; harness "AGENT split"): the agent's `SubmitWork`,
//! `CancelWork`, `GetOperation`, `ListOperations` and `StreamWork` handlers and its operation runtime, driven by a
//! tonic client that presents the coordinator's leaf, against the in-process stub coordinator (lease and fences)
//! and an in-process `cere-mock-sglang` backend.

mod support;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use cere_mock_sglang::{MockConfig, MockHandle, MockSglang, ServerInfo};
use cere_proto::worker::v2 as pb;
use cere_runtime::agent::config::AgentConfig;
use cere_runtime::agent::journal::{OpKey, OpView, OPERATIONS_FILE};
use cere_runtime::agent::{Agent, AgentHandle};
use cere_runtime::fed::tls::Pki;
use cere_runtime::fed::wire::{self, AdmissionState};
use cere_runtime::sglang::Rid;
use pb::worker_inference_service_client::WorkerInferenceServiceClient;
use pb::worker_service_client::WorkerServiceClient;
use serde_json::{json, Value};
use support::pki::TestPki;
use support::stub_coordinator::{StubConfig, StubCoordinator};
use support::StateDir;
use tonic::transport::{Channel, Endpoint};
use tonic::{Code, Status, Streaming};

const CLUSTER: &str = "lab-test";
const WORKER: &str = "worker-a";
const WORKER_PRINCIPAL: &str = "cere-worker-a";
const COORD_PRINCIPAL: &str = "cere-coordinator";
const COORD_ID: &str = "coord-1";
const MODEL: &str = "Qwen3.8-Flash-Next";
const TERM: u64 = 5;
/// The stub coordinator's admission lease, with [`LOST_AFTER`] and the agent's `lost_after_ms` in [`LIMITS`]: a
/// lease no host stall outlasts (as `agent_lifecycle.rs`). A 500 ms lease expired when a loaded host starved the
/// agent's heartbeats, and the next SubmitWork was refused `UNAVAILABLE lease_expired` instead of the refusal under
/// test (`a_full_journal_refuses_new_keys`, CI load 19-57; rt-flake design note F9).
const LEASE_MS: u64 = 60_000;
/// The stub coordinator's Lost bound (no accepted heartbeat for this long).
const LOST_AFTER: Duration = Duration::from_secs(180);
/// The lease [`expire_lease`] switches the stub to before it falls silent.
const SHORT_LEASE_MS: u64 = 500;
const LIMITS: &str = "heartbeat_interval_ms = 100\nlost_after_ms = 180000\nrpc_timeout_ms = 300\n";
/// The prompt of every test body: it must never reach an event line or the op journal.
const PROMPT: &str = "work prompt secret";
const SCOPE: &str = "dev";
const CHAT_STREAM: &[u8] = br#"{"kind":"chat","stream":true}"#;
const CHAT_UNARY: &[u8] = br#"{"kind":"chat","stream":false}"#;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

async fn eventually(what: &str, mut done: impl FnMut() -> bool) {
    let until = Instant::now() + Duration::from_secs(10);
    while !done() {
        assert!(Instant::now() < until, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn confirmed_info(seed: u64) -> ServerInfo {
    let mut info = support::launched(seed, 1.5);
    info.extra.insert("tp_size".into(), json!(1));
    info
}

/// Backend pacing: per-token delay and the natural output length.
#[derive(Clone, Copy)]
struct Pace {
    latency_ms: u64,
    natural: u32,
}

const FAST: Pace = Pace {
    latency_ms: 0,
    natural: 100_000,
};
const SLOW: Pace = Pace {
    latency_ms: 30,
    natural: 100_000,
};

struct Fixture {
    root: StateDir,
    pki: TestPki,
    ca: PathBuf,
    cert: PathBuf,
    key: PathBuf,
    stub: StubCoordinator,
    mock: MockHandle,
}

impl Fixture {
    async fn new(pace: Pace) -> Fixture {
        Fixture::with(confirmed_info(1), pace).await
    }

    async fn with(info: ServerInfo, pace: Pace) -> Fixture {
        // The op journal's fsyncs run in the link loop beside the heartbeats (O146).
        let root = StateDir::on_tmpfs();
        let pki_dir = root.path().join("pki");
        let pki = TestPki::new();
        let ca = pki.write_ca(&pki_dir);
        let (cert, key) = pki.leaf(WORKER_PRINCIPAL).write(&pki_dir);
        let coord = load_pki(&pki, &pki_dir, &ca, COORD_PRINCIPAL);
        let stub = StubCoordinator::start(
            &coord,
            StubConfig {
                cluster_id: CLUSTER.into(),
                term: TERM,
                admission_lease_ms: LEASE_MS,
                lost_after: LOST_AFTER,
                bindings: BTreeMap::from([(WORKER_PRINCIPAL.to_owned(), WORKER.to_owned())]),
            },
        )
        .await;
        stub.set_reconcile(true);
        let mock = MockSglang::start(MockConfig {
            server_info: info,
            token_latency: Duration::from_millis(pace.latency_ms),
            natural_output_tokens: pace.natural,
            record_bodies: true,
            ..MockConfig::default()
        })
        .await
        .unwrap();
        Fixture {
            root,
            pki,
            ca,
            cert,
            key,
            stub,
            mock,
        }
    }

    fn state_dir(&self) -> PathBuf {
        self.root.path().join("state")
    }

    fn config(&self, limits: &str, journal: &str) -> AgentConfig {
        let text = format!(
            "cluster_id = \"{CLUSTER}\"\nworker_id = \"{WORKER}\"\nlisten = \"127.0.0.1:{}\"\n\
             principal = \"{WORKER_PRINCIPAL}\"\nca_file = \"{}\"\ncert_file = \"{}\"\nkey_file = \"{}\"\n\
             state_dir = \"{}\"\n\n[coordinator]\nendpoint = \"{}\"\nprincipal = \"{COORD_PRINCIPAL}\"\n\
             coordinator_id = \"{COORD_ID}\"\n\n[backend]\nbase_url = \"{}\"\nserved_model_name = \"{MODEL}\"\n\n\
             [journal]\n{journal}\n[limits]\n{limits}",
            support::free_port(),
            self.ca.display(),
            self.cert.display(),
            self.key.display(),
            self.state_dir().display(),
            self.stub.endpoint(),
            self.mock.base_url(),
        );
        AgentConfig::from_toml_str(&text, None).unwrap()
    }

    /// An agent whose admission is open (registered, lease live, reconciled).
    async fn open_agent(&self, limits: &str, journal: &str) -> AgentHandle {
        let agent = Agent::start(self.config(limits, journal)).await.unwrap();
        open(&agent).await;
        agent
    }

    fn channel(&self, agent: &AgentHandle, principal: &str) -> Channel {
        self.endpoint(agent, principal).connect_lazy()
    }

    fn endpoint(&self, agent: &AgentHandle, principal: &str) -> Endpoint {
        let dir = self.root.path().join("pki");
        let pki = load_pki(&self.pki, &dir, &self.ca, principal);
        Endpoint::from_shared(format!("https://{}", agent.local_addr()))
            .unwrap()
            .timeout(Duration::from_secs(30))
            .tls_config(pki.client_tls(WORKER_PRINCIPAL))
            .unwrap()
    }

    /// A coordinator `WorkerService` client whose HTTP/2 receive window is 16 KiB per stream: a stream it does not
    /// read stays open on the agent while the op's output exceeds the window.
    fn narrow_control(&self, agent: &AgentHandle) -> WorkerServiceClient<Channel> {
        let endpoint = self
            .endpoint(agent, COORD_PRINCIPAL)
            .initial_stream_window_size(16 * 1024);
        WorkerServiceClient::new(endpoint.connect_lazy())
    }

    /// The coordinator's clients.
    fn clients(&self, agent: &AgentHandle) -> Clients {
        let channel = self.channel(agent, COORD_PRINCIPAL);
        Clients {
            control: WorkerServiceClient::new(channel.clone()),
            inference: WorkerInferenceServiceClient::new(channel),
        }
    }

    fn aborted(&self, rid: &str) -> bool {
        self.mock
            .abort_requests()
            .iter()
            .any(|a| a.rid == rid && !a.abort_all)
    }

    fn dispatched(&self, rid: &str) -> usize {
        self.mock
            .dispatched_rids()
            .iter()
            .filter(|r| r.as_str() == rid)
            .count()
    }
}

struct Clients {
    control: WorkerServiceClient<Channel>,
    inference: WorkerInferenceServiceClient<Channel>,
}

fn load_pki(pki: &TestPki, dir: &Path, ca: &Path, principal: &str) -> Pki {
    let (cert, key) = pki.leaf(principal).write(dir);
    Pki::load(ca, &cert, &key, principal).unwrap()
}

async fn open(agent: &AgentHandle) {
    eventually("admission open", || {
        agent.admission_state() == AdmissionState::Open
    })
    .await;
}

/// Expires the agent's lease: one accepted heartbeat grants [`SHORT_LEASE_MS`], then the stub falls silent, so the
/// lease ends within [`SHORT_LEASE_MS`] of that beat. Everything the test did before ran under the long lease.
async fn expire_lease(fx: &Fixture, agent: &AgentHandle) {
    fx.stub.set_admission_lease_ms(SHORT_LEASE_MS);
    let mark = fx.stub.heartbeats().len();
    eventually("a short grant", || fx.stub.heartbeats().len() > mark).await;
    fx.stub.set_silent(true);
    eventually("fenced", || {
        agent.admission_state() == AdmissionState::Fenced
    })
    .await;
}

fn rid(seq: u64) -> String {
    Rid::new(1, seq, 0xbeef).as_str().to_owned()
}

fn key(scope: &str, op: &str) -> OpKey {
    OpKey {
        scope_id: scope.into(),
        operation_id: op.into(),
    }
}

fn view(agent: &AgentHandle, scope: &str, op: &str) -> Option<OpView> {
    agent.shared().journal.lock().unwrap().get(&key(scope, op))
}

fn all_ops(agent: &AgentHandle) -> Vec<OpView> {
    agent
        .shared()
        .journal
        .lock()
        .unwrap()
        .list(None, usize::MAX)
        .0
}

fn events_with(agent: &AgentHandle, needle: &str) -> usize {
    agent
        .events()
        .iter()
        .filter(|line| line.contains(needle))
        .count()
}

/// The read authority (no op fields, no generation) at the agent's current epoch and term.
fn read_authority(agent: &AgentHandle) -> pb::Authority {
    pb::Authority {
        target: Some(pb::WorkerIdentity {
            cluster_id: CLUSTER.into(),
            worker_id: WORKER.into(),
            worker_epoch: agent.epoch(),
        }),
        coordinator_id: COORD_ID.into(),
        coordinator_term: agent.term(),
        ..Default::default()
    }
}

fn op_authority(agent: &AgentHandle, scope: &str, op: &str, digest: [u8; 32]) -> pb::Authority {
    pb::Authority {
        scope_id: scope.into(),
        operation_id: op.into(),
        payload_sha256: digest.to_vec(),
        remaining_budget_ms: 60_000,
        model_generation: Some(agent.generation()),
        ..read_authority(agent)
    }
}

fn body(rid: &str, stream: bool, max_tokens: u64) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "rid": rid,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }))
    .unwrap()
}

/// A well-formed SubmitWork for (`SCOPE`, `op`) with the correct digest.
fn submit(
    agent: &AgentHandle,
    op: &str,
    rid: &str,
    stream: bool,
    max_tokens: u64,
) -> pb::SubmitWorkRequest {
    submit_in(agent, SCOPE, op, rid, stream, max_tokens)
}

fn submit_in(
    agent: &AgentHandle,
    scope: &str,
    op: &str,
    rid: &str,
    stream: bool,
    max_tokens: u64,
) -> pb::SubmitWorkRequest {
    let unit = if stream { CHAT_STREAM } else { CHAT_UNARY };
    let input = body(rid, stream, max_tokens);
    let digest = wire::payload_sha256(wire::METHOD_SUBMIT, scope, rid, 1, unit, &input);
    pb::SubmitWorkRequest {
        authority: Some(op_authority(agent, scope, op, digest)),
        request_id: rid.into(),
        attempt: 1,
        work_unit_json: unit.to_vec(),
        bounded_inference_input_json: input,
    }
}

/// Recomputes the digest after a test edited the request.
fn redigest(req: &mut pb::SubmitWorkRequest) {
    let auth = req.authority.as_mut().unwrap();
    auth.payload_sha256 = wire::payload_sha256(
        wire::METHOD_SUBMIT,
        &auth.scope_id,
        &req.request_id,
        req.attempt,
        &req.work_unit_json,
        &req.bounded_inference_input_json,
    )
    .to_vec();
}

fn cancel(agent: &AgentHandle, target_rid: &str, ticket: &str) -> pb::CancelWorkRequest {
    let op = format!("{target_rid}/cancel/1");
    let digest = wire::payload_sha256(
        wire::METHOD_CANCEL,
        SCOPE,
        target_rid,
        1,
        b"",
        ticket.as_bytes(),
    );
    pb::CancelWorkRequest {
        authority: Some(op_authority(agent, SCOPE, &op, digest)),
        request_id: target_rid.into(),
        attempt: 1,
        completion_ticket: ticket.into(),
    }
}

fn attach(
    agent: &AgentHandle,
    op: &str,
    rid: &str,
    ticket: &str,
    after: u64,
) -> pb::WorkStreamRequest {
    pb::WorkStreamRequest {
        authority: Some(op_authority(agent, SCOPE, op, [0; 32])),
        request_id: rid.into(),
        attempt: 1,
        completion_ticket: ticket.into(),
        after_sequence: after,
    }
}

async fn accepted(c: &mut Clients, req: pb::SubmitWorkRequest) -> pb::OperationReceipt {
    let receipt = c.inference.submit_work(req).await.unwrap().into_inner();
    assert!(receipt.accepted, "{receipt:?}");
    assert!(receipt.reason_code.is_empty());
    assert_eq!(receipt.completion_ticket.len(), 32, "{receipt:?}");
    receipt
}

fn assert_refused<T: std::fmt::Debug>(result: Result<T, Status>, code: Code, reason: &str) {
    match result {
        Err(status) => {
            assert_eq!(status.code(), code, "{status:?}");
            assert_eq!(wire::reason(&status), reason, "{status:?}");
        }
        Ok(ok) => panic!("expected {code:?} {reason}, got {ok:?}"),
    }
}

/// Reads `stream` to its end: the events and the closing error, if any.
async fn drain(mut stream: Streaming<pb::WorkEvent>) -> (Vec<pb::WorkEvent>, Option<Status>) {
    let mut events = Vec::new();
    loop {
        match tokio::time::timeout(Duration::from_secs(20), stream.message())
            .await
            .expect("the stream stalled")
        {
            Ok(Some(event)) => events.push(event),
            Ok(None) => return (events, None),
            Err(status) => return (events, Some(status)),
        }
    }
}

async fn stream_all(
    c: &mut Clients,
    req: pb::WorkStreamRequest,
) -> (Vec<pb::WorkEvent>, Option<Status>) {
    drain(c.control.stream_work(req).await.unwrap().into_inner()).await
}

/// Sequence numbers consecutive from `first`, offsets = sequence - 1, exactly one terminal event, last, with an
/// empty body; returns its reason.
fn check_events(events: &[pb::WorkEvent], first: u64) -> String {
    assert!(!events.is_empty());
    for (i, event) in events.iter().enumerate() {
        assert_eq!(event.sequence, first + i as u64, "{events:?}");
        assert_eq!(event.committed_token_offset, event.sequence - 1);
        assert_eq!(event.terminal, i + 1 == events.len(), "{events:?}");
    }
    let last = events.last().unwrap();
    assert!(last.bounded_output_event_json.is_empty());
    last.reason_code.clone()
}

/// The streamed text of the non-terminal chunk events.
fn streamed_text(events: &[pb::WorkEvent]) -> String {
    events
        .iter()
        .filter(|e| !e.terminal)
        .filter_map(|e| {
            let chunk: Value = serde_json::from_slice(&e.bounded_output_event_json).unwrap();
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

fn terminal_events(agent: &AgentHandle, op: &str) -> usize {
    events_with(
        agent,
        &format!("cere-worker event=terminal op={SCOPE}/{op} reason="),
    )
}

/// Every refusal before acceptance has its code and reason and changes nothing: no journal record, no backend
/// call, no `accept` event. Load/Drain/Unload/StreamTelemetry are `UNIMPLEMENTED`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn submit_refusals_change_nothing() {
    let fx = Fixture::new(FAST).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let good = || submit(&agent, "op-1", &r, true, 4);
    let edit = |f: &dyn Fn(&mut pb::SubmitWorkRequest)| {
        let mut req = good();
        f(&mut req);
        req
    };
    let cases: Vec<(pb::SubmitWorkRequest, Code, &str)> = vec![
        (
            edit(&|q| q.authority = None),
            Code::InvalidArgument,
            wire::MALFORMED_REQUEST,
        ),
        (
            edit(&|q| q.authority.as_mut().unwrap().target = None),
            Code::InvalidArgument,
            wire::MALFORMED_REQUEST,
        ),
        (
            edit(&|q| {
                q.authority
                    .as_mut()
                    .unwrap()
                    .target
                    .as_mut()
                    .unwrap()
                    .cluster_id = "other".into()
            }),
            Code::FailedPrecondition,
            wire::WRONG_CLUSTER,
        ),
        (
            edit(&|q| {
                q.authority
                    .as_mut()
                    .unwrap()
                    .target
                    .as_mut()
                    .unwrap()
                    .worker_id = "worker-b".into()
            }),
            Code::FailedPrecondition,
            wire::WRONG_WORKER,
        ),
        (
            edit(&|q| {
                q.authority
                    .as_mut()
                    .unwrap()
                    .target
                    .as_mut()
                    .unwrap()
                    .worker_epoch -= 1
            }),
            Code::FailedPrecondition,
            wire::STALE_EPOCH,
        ),
        (
            edit(&|q| q.authority.as_mut().unwrap().coordinator_id = "coord-2".into()),
            Code::FailedPrecondition,
            wire::WRONG_COORDINATOR,
        ),
        (
            edit(&|q| q.authority.as_mut().unwrap().coordinator_term -= 1),
            Code::FailedPrecondition,
            wire::STALE_TERM,
        ),
        (
            edit(&|q| q.authority.as_mut().unwrap().model_generation = Some(9)),
            Code::FailedPrecondition,
            wire::STALE_GENERATION,
        ),
        (
            edit(&|q| q.authority.as_mut().unwrap().model_generation = None),
            Code::FailedPrecondition,
            wire::STALE_GENERATION,
        ),
        (
            edit(&|q| {
                q.authority
                    .as_mut()
                    .unwrap()
                    .payload_sha256
                    .pop()
                    .map(drop)
                    .unwrap()
            }),
            Code::InvalidArgument,
            wire::MALFORMED_REQUEST,
        ),
        (
            edit(&|q| {
                q.work_unit_json = b"not json".to_vec();
                redigest(q);
            }),
            Code::InvalidArgument,
            wire::MALFORMED_REQUEST,
        ),
        (
            edit(&|q| {
                q.bounded_inference_input_json = body(&rid(2), true, 4);
                redigest(q);
            }),
            Code::InvalidArgument,
            wire::MALFORMED_REQUEST,
        ),
        (
            edit(&|q| q.bounded_inference_input_json = body(&r, true, 5)),
            Code::InvalidArgument,
            wire::PAYLOAD_DIGEST_MISMATCH,
        ),
    ];
    for (req, code, reason) in cases {
        assert_refused(c.inference.submit_work(req).await, code, reason);
    }
    // Authz first: a principal other than the coordinator's.
    let mut stranger = WorkerInferenceServiceClient::new(fx.channel(&agent, "cere-worker-b"));
    assert_refused(
        stranger.submit_work(good()).await,
        Code::PermissionDenied,
        wire::PERMISSION_DENIED_REASON,
    );
    assert!(all_ops(&agent).is_empty());
    assert!(fx.mock.dispatched_rids().is_empty());
    assert_eq!(events_with(&agent, "event=accept"), 0);

    // Attached mode (m36 design note Q6): a well-formed lifecycle call is FAILED_PRECONDITION model_unmanaged,
    // nothing journaled.
    let lifecycle = |method: &str, op: &str| {
        let lock = [0xab; 32];
        let digest = wire::lifecycle_sha256(method, "_cere_control", &lock, "", &[]);
        pb::ModelLifecycleRequest {
            authority: Some(op_authority(&agent, "_cere_control", op, digest)),
            deployment_lock_sha256: lock.to_vec(),
            ..Default::default()
        }
    };
    assert_refused(
        c.control
            .load_model(lifecycle(wire::METHOD_LOAD, "load-1"))
            .await,
        Code::FailedPrecondition,
        wire::MODEL_UNMANAGED,
    );
    assert_refused(
        c.control
            .drain_model(lifecycle(wire::METHOD_DRAIN, "drain-1"))
            .await,
        Code::FailedPrecondition,
        wire::MODEL_UNMANAGED,
    );
    assert_refused(
        c.control
            .unload_model(lifecycle(wire::METHOD_UNLOAD, "unload-1"))
            .await,
        Code::FailedPrecondition,
        wire::MODEL_UNMANAGED,
    );
    assert!(all_ops(&agent).is_empty());
    let telemetry = pb::TelemetryRequest {
        authority: Some(read_authority(&agent)),
        after_sequence: 0,
    };
    assert_eq!(
        c.control
            .stream_telemetry(telemetry)
            .await
            .unwrap_err()
            .code(),
        Code::Unimplemented
    );
    agent.shutdown().await;
}

/// Submit + stream: consecutive sequences from 1, offsets = index, one terminal event ("" = natural completion)
/// with an empty body and no `[DONE]` frame; the journal holds accepted, terminal and cleanup; events carry the op
/// but never prompt or output text, and neither does the journal file.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn submit_then_stream_delivers_the_chunks_and_a_terminal_event() {
    let fx = Fixture::new(FAST).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let receipt = accepted(&mut c, submit(&agent, "op-1", &r, true, 5)).await;
    assert_eq!(receipt.operation_id, "op-1");
    let (events, closed) = stream_all(
        &mut c,
        attach(&agent, "op-1", &r, &receipt.completion_ticket, 0),
    )
    .await;
    assert!(closed.is_none(), "{closed:?}");
    assert_eq!(check_events(&events, 1), "");
    assert_eq!(streamed_text(&events), "tok0 tok1 tok2 tok3 tok4");
    for event in &events {
        assert_eq!(event.scope_id, SCOPE);
        assert_eq!(event.request_id, r);
        assert_eq!(event.attempt, 1);
        assert_eq!(event.completion_ticket, receipt.completion_ticket);
        assert_eq!(event.coordinator_term, TERM);
        let worker = event.worker.as_ref().unwrap();
        assert_eq!(
            (
                worker.cluster_id.as_str(),
                worker.worker_id.as_str(),
                worker.worker_epoch
            ),
            (CLUSTER, WORKER, agent.epoch())
        );
        assert!(event.bounded_output_event_json.len() <= wire::WORK_EVENT_MAX);
        assert!(!String::from_utf8_lossy(&event.bounded_output_event_json).contains("[DONE]"));
    }
    eventually("cleanup acknowledged", || {
        view(&agent, SCOPE, "op-1").is_some_and(|v| v.cleanup_acknowledged)
    })
    .await;
    let v = view(&agent, SCOPE, "op-1").unwrap();
    assert!(v.terminal && v.reason.is_empty());
    assert_eq!(v.ticket, receipt.completion_ticket);
    assert_eq!(fx.dispatched(&r), 1);
    assert_eq!(
        events_with(&agent, "cere-worker event=accept op=dev/op-1 fresh=true"),
        1
    );
    assert_eq!(terminal_events(&agent, "op-1"), 1);
    assert_eq!(
        events_with(&agent, "cere-worker event=cleanup op=dev/op-1"),
        1
    );
    for line in agent.events() {
        assert!(!line.contains(PROMPT) && !line.contains("tok1"), "{line}");
    }
    let journal = std::fs::read_to_string(fx.state_dir().join("operations.jsonl")).unwrap();
    assert!(!journal.contains(PROMPT) && !journal.contains("tok1"));
    let status = c
        .control
        .get_operation(pb::OperationQuery {
            authority: Some(read_authority(&agent)),
            queried_operation_id: "op-1".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(status.phase, wire::PHASE_RELEASED);
    agent.shutdown().await;
}

/// Unary chat: one non-terminal event with the whole JSON response, then the terminal event.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unary_chat_is_one_event_then_the_terminal() {
    let fx = Fixture::new(FAST).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let receipt = accepted(&mut c, submit(&agent, "op-1", &r, false, 3)).await;
    let (events, closed) = stream_all(
        &mut c,
        attach(&agent, "op-1", &r, &receipt.completion_ticket, 0),
    )
    .await;
    assert!(closed.is_none());
    assert_eq!(events.len(), 2, "{events:?}");
    assert_eq!(check_events(&events, 1), "");
    let response: Value = serde_json::from_slice(&events[0].bounded_output_event_json).unwrap();
    assert_eq!(
        response["choices"][0]["message"]["content"],
        "tok0 tok1 tok2"
    );
    agent.shutdown().await;
}

/// A retried Submit (same key and identity) returns the original receipt without a second backend call and never
/// renews the deadline (the budget is not part of the identity); a changed payload under the same key is
/// `ALREADY_EXISTS operation_conflict`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_duplicate_submit_returns_the_original_receipt() {
    let fx = Fixture::new(SLOW).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let first = accepted(&mut c, submit(&agent, "op-1", &r, true, 20)).await;
    let deadline = view(&agent, SCOPE, "op-1").unwrap().deadline_ms;
    let mut retry = submit(&agent, "op-1", &r, true, 20);
    retry.authority.as_mut().unwrap().remaining_budget_ms = 500_000;
    let second = accepted(&mut c, retry).await;
    assert_eq!(second, first);
    assert_eq!(view(&agent, SCOPE, "op-1").unwrap().deadline_ms, deadline);
    assert_eq!(
        events_with(&agent, "event=accept op=dev/op-1 fresh=false"),
        1
    );

    let mut conflict = submit(&agent, "op-1", &r, true, 21);
    redigest(&mut conflict);
    assert_refused(
        c.inference.submit_work(conflict).await,
        Code::AlreadyExists,
        wire::OPERATION_CONFLICT,
    );
    let (events, _) = stream_all(
        &mut c,
        attach(&agent, "op-1", &r, &first.completion_ticket, 0),
    )
    .await;
    assert_eq!(check_events(&events, 1), "");
    assert_eq!(fx.dispatched(&r), 1);
    assert_eq!(all_ops(&agent).len(), 1);
    agent.shutdown().await;
}

/// CancelWork (its own op id, journaled) ends the target `cancel_requested` once and aborts its rid; a repeat
/// returns the same receipt; a cancel of an unknown op or with a wrong ticket is `ticket_mismatch` and changes
/// nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_ends_the_target_once_and_is_idempotent() {
    let fx = Fixture::new(SLOW).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let receipt = accepted(&mut c, submit(&agent, "op-1", &r, true, 200)).await;
    let ticket = receipt.completion_ticket.clone();

    let unknown = rid(9);
    assert_refused(
        c.control
            .cancel_work(cancel(&agent, &unknown, &ticket))
            .await,
        Code::FailedPrecondition,
        wire::TICKET_MISMATCH,
    );
    assert_refused(
        c.control
            .cancel_work(cancel(&agent, &r, &"0".repeat(32)))
            .await,
        Code::FailedPrecondition,
        wire::TICKET_MISMATCH,
    );
    assert_eq!(all_ops(&agent).len(), 1);

    let first = c
        .control
        .cancel_work(cancel(&agent, &r, &ticket))
        .await
        .unwrap()
        .into_inner();
    assert!(first.accepted);
    assert_eq!(first.operation_id, format!("{r}/cancel/1"));
    let again = c
        .control
        .cancel_work(cancel(&agent, &r, &ticket))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(again, first);
    let target = view(&agent, SCOPE, "op-1").unwrap();
    assert!(target.terminal);
    assert_eq!(target.reason, wire::CANCEL_REQUESTED);
    let cancel_op = view(&agent, SCOPE, &format!("{r}/cancel/1")).unwrap();
    assert_eq!(cancel_op.method, wire::METHOD_CANCEL);
    assert!(cancel_op.terminal && cancel_op.cleanup_acknowledged);
    eventually("the rid aborted", || fx.aborted(&r)).await;
    let (events, closed) = stream_all(&mut c, attach(&agent, "op-1", &r, &ticket, 0)).await;
    assert!(closed.is_none());
    assert_eq!(
        check_events(&events, events[0].sequence),
        wire::CANCEL_REQUESTED
    );
    eventually("the cancelled op cleaned up", || {
        view(&agent, SCOPE, "op-1").is_some_and(|v| v.cleanup_acknowledged)
    })
    .await;
    assert_eq!(terminal_events(&agent, "op-1"), 1);
    assert_eq!(
        events_with(
            &agent,
            &format!("event=accept op=dev/{r}/cancel/1 fresh=false")
        ),
        1
    );
    agent.shutdown().await;
}

/// StreamWork: wrong ticket `ticket_mismatch`, unknown op `unknown_operation`, a second concurrent attach
/// `operation_conflict`; after a broken stream a re-attach replays from `after_sequence`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_attach_rules_and_replay_after_a_broken_stream() {
    let fx = Fixture::new(SLOW).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let receipt = accepted(&mut c, submit(&agent, "op-1", &r, true, 40)).await;
    let ticket = receipt.completion_ticket.clone();
    assert_refused(
        c.control
            .stream_work(attach(&agent, "op-1", &r, &"f".repeat(32), 0))
            .await,
        Code::FailedPrecondition,
        wire::TICKET_MISMATCH,
    );
    assert_refused(
        c.control
            .stream_work(attach(&agent, "op-9", &r, &ticket, 0))
            .await,
        Code::NotFound,
        wire::UNKNOWN_OPERATION,
    );
    let mut first = c
        .control
        .stream_work(attach(&agent, "op-1", &r, &ticket, 0))
        .await
        .unwrap()
        .into_inner();
    for n in 1..=3 {
        let event = first.message().await.unwrap().unwrap();
        assert_eq!(event.sequence, n);
    }
    assert_refused(
        c.control
            .stream_work(attach(&agent, "op-1", &r, &ticket, 0))
            .await,
        Code::AlreadyExists,
        wire::OPERATION_CONFLICT,
    );
    drop(first);
    let until = Instant::now() + Duration::from_secs(10);
    let second = loop {
        match c
            .control
            .stream_work(attach(&agent, "op-1", &r, &ticket, 3))
            .await
        {
            Ok(stream) => break stream.into_inner(),
            Err(status) if wire::reason(&status) == wire::OPERATION_CONFLICT => {
                assert!(Instant::now() < until, "the broken stream never detached");
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Err(status) => panic!("{status:?}"),
        }
    };
    let (events, closed) = drain(second).await;
    assert!(closed.is_none());
    assert_eq!(check_events(&events, 4), "");
    agent.shutdown().await;
}

/// Q14: a terminal op a stream has attached to is dropped from the live-op table (its output buffer with it) once
/// it is cleaned up and no stream is attached, even when its stream broke before the terminal. A later attach is
/// answered like one for an op of an earlier process: just the terminal event, after `after_sequence`. (An op no
/// stream has attached to yet keeps its buffer: `submit_then_stream_delivers_the_chunks_and_a_terminal_event`.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cleaned_up_op_frees_its_buffer() {
    let fx = Fixture::new(SLOW).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let receipt = accepted(&mut c, submit(&agent, "op-1", &r, true, 40)).await;
    let ticket = receipt.completion_ticket.clone();
    let mut first = c
        .control
        .stream_work(attach(&agent, "op-1", &r, &ticket, 0))
        .await
        .unwrap()
        .into_inner();
    for n in 1..=3 {
        assert_eq!(first.message().await.unwrap().unwrap().sequence, n);
    }
    drop(first);
    eventually("op-1 terminal and cleaned up", || {
        view(&agent, SCOPE, "op-1").is_some_and(|v| v.terminal && v.cleanup_acknowledged)
    })
    .await;
    let until = Instant::now() + Duration::from_secs(10);
    let second = loop {
        match c
            .control
            .stream_work(attach(&agent, "op-1", &r, &ticket, 3))
            .await
        {
            Ok(stream) => break stream.into_inner(),
            Err(status) if wire::reason(&status) == wire::OPERATION_CONFLICT => {
                assert!(Instant::now() < until, "the broken stream never detached");
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Err(status) => panic!("{status:?}"),
        }
    };
    let (events, closed) = drain(second).await;
    assert!(closed.is_none(), "{closed:?}");
    assert_eq!(
        events.len(),
        1,
        "the buffer was replayed: {} events",
        events.len()
    );
    assert_eq!(check_events(&events, 4), "");
    agent.shutdown().await;
}

/// The worker-local deadline (receipt + remaining budget) ends the op `deadline_exceeded` and aborts it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_deadline_ends_the_op() {
    let fx = Fixture::new(SLOW).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let mut req = submit(&agent, "op-1", &r, true, 500);
    req.authority.as_mut().unwrap().remaining_budget_ms = 400;
    let receipt = accepted(&mut c, req).await;
    let (events, closed) = stream_all(
        &mut c,
        attach(&agent, "op-1", &r, &receipt.completion_ticket, 0),
    )
    .await;
    assert!(closed.is_none());
    assert_eq!(check_events(&events, 1), wire::DEADLINE_EXCEEDED);
    assert_eq!(
        view(&agent, SCOPE, "op-1").unwrap().reason,
        wire::DEADLINE_EXCEEDED
    );
    eventually("the rid aborted", || fx.aborted(&r)).await;
    assert_eq!(terminal_events(&agent, "op-1"), 1);
    agent.shutdown().await;
}

/// Nobody drains the output: the buffer fills, the backend read stops, and after `output_stall_ms` the op ends
/// `client_backpressure` (aborted); a later attach gets only the terminal event.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn undrained_output_ends_the_op_client_backpressure() {
    let fx = Fixture::new(FAST).await;
    let limits = format!("{LIMITS}output_buffer_bytes = 1024\noutput_stall_ms = 300\n");
    let agent = fx.open_agent(&limits, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let receipt = accepted(&mut c, submit(&agent, "op-1", &r, true, 5000)).await;
    eventually("client_backpressure", || {
        view(&agent, SCOPE, "op-1").is_some_and(|v| v.terminal)
    })
    .await;
    assert_eq!(
        view(&agent, SCOPE, "op-1").unwrap().reason,
        wire::CLIENT_BACKPRESSURE
    );
    eventually("the rid aborted", || fx.aborted(&r)).await;
    let (events, closed) = stream_all(
        &mut c,
        attach(&agent, "op-1", &r, &receipt.completion_ticket, 0),
    )
    .await;
    assert!(closed.is_none());
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(
        check_events(&events, events[0].sequence),
        wire::CLIENT_BACKPRESSURE
    );
    agent.shutdown().await;
}

/// An output event larger than `WORK_EVENT_MAX` fails the op `backend_protocol_error`. The mock echoes the
/// request's `model` in its response, so a long model name makes a one-token answer oversize.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_oversize_event_fails_the_op() {
    let fx = Fixture::new(FAST).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let mut req = submit(&agent, "op-1", &r, false, 1);
    let mut input: Value = serde_json::from_slice(&req.bounded_inference_input_json).unwrap();
    input["model"] = json!("m".repeat(wire::WORK_EVENT_MAX));
    req.bounded_inference_input_json = serde_json::to_vec(&input).unwrap();
    redigest(&mut req);
    let receipt = accepted(&mut c, req).await;
    let (events, closed) = stream_all(
        &mut c,
        attach(&agent, "op-1", &r, &receipt.completion_ticket, 0),
    )
    .await;
    assert!(closed.is_none());
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(check_events(&events, 1), "backend_protocol_error");
    agent.shutdown().await;
}

/// Live ops are bounded by the backend's `max_running_requests`: one more is `RESOURCE_EXHAUSTED queue_full`
/// (journal unchanged) while a retry of a live op still returns its receipt.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn submits_beyond_the_backend_cap_are_queue_full() {
    let fx = Fixture::new(SLOW).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let cap = ServerInfo::default().max_running_requests as u64;
    let mut first = None;
    for n in 1..=cap {
        let receipt = accepted(
            &mut c,
            submit(&agent, &format!("op-{n}"), &rid(n), true, 300),
        )
        .await;
        first.get_or_insert(receipt);
    }
    assert_refused(
        c.inference
            .submit_work(submit(&agent, "op-x", &rid(99), true, 3))
            .await,
        Code::ResourceExhausted,
        wire::QUEUE_FULL,
    );
    assert_eq!(all_ops(&agent).len() as u64, cap);
    assert_eq!(fx.dispatched(&rid(99)), 0);
    let retry = accepted(&mut c, submit(&agent, "op-1", &rid(1), true, 300)).await;
    assert_eq!(Some(retry), first);
    agent.shutdown().await;
}

/// `max_operations` live + retained: a new key beyond it is `RESOURCE_EXHAUSTED journal_full`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_journal_refuses_new_keys() {
    let fx = Fixture::new(FAST).await;
    let agent = fx.open_agent(LIMITS, "max_operations = 1\n").await;
    let mut c = fx.clients(&agent);
    let receipt = accepted(&mut c, submit(&agent, "op-1", &rid(1), false, 1)).await;
    let (events, _) = stream_all(
        &mut c,
        attach(&agent, "op-1", &rid(1), &receipt.completion_ticket, 0),
    )
    .await;
    assert_eq!(check_events(&events, 1), "");
    assert_refused(
        c.inference
            .submit_work(submit(&agent, "op-2", &rid(2), false, 1))
            .await,
        Code::ResourceExhausted,
        wire::JOURNAL_FULL,
    );
    assert_eq!(fx.dispatched(&rid(2)), 0);
    agent.shutdown().await;
}

/// H1: the stream pool is the live-op bound + 1; a `StreamWork` beyond it is `RESOURCE_EXHAUSTED quota_exhausted`
/// while `CancelWork` (control pool) still succeeds. Streams nobody reads stay open under HTTP/2 flow control.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_stream_pool_refuses_streams_but_not_cancel() {
    let mut info = confirmed_info(1);
    info.max_running_requests = 1;
    let fx = Fixture::with(info, FAST).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let mut narrow = fx.narrow_control(&agent);
    let mut held = Vec::new();
    let mut tickets = Vec::new();
    for n in 1..=2u64 {
        let op = format!("op-{n}");
        let receipt = accepted(&mut c, submit(&agent, &op, &rid(n), true, 1500)).await;
        let stream = narrow
            .stream_work(attach(&agent, &op, &rid(n), &receipt.completion_ticket, 0))
            .await
            .unwrap()
            .into_inner();
        held.push(stream);
        eventually("the op completed", || {
            view(&agent, SCOPE, &op).is_some_and(|v| v.cleanup_acknowledged)
        })
        .await;
        tickets.push(receipt.completion_ticket);
    }
    assert_refused(
        c.control
            .stream_work(attach(&agent, "op-2", &rid(2), &tickets[1], 0))
            .await,
        Code::ResourceExhausted,
        wire::QUOTA_EXHAUSTED,
    );
    let receipt = c
        .control
        .cancel_work(cancel(&agent, &rid(2), &tickets[1]))
        .await
        .unwrap()
        .into_inner();
    assert!(receipt.accepted);
    drop(held);
    agent.shutdown().await;
}

/// Lease expiry refuses Submit (`UNAVAILABLE lease_expired`) but in-flight work finishes and Cancel is served.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lease_expiry_refuses_submit_but_in_flight_work_finishes() {
    let fx = Fixture::new(SLOW).await;
    let limits = "heartbeat_interval_ms = 100\nlost_after_ms = 10000\nrpc_timeout_ms = 300\n";
    let agent = fx.open_agent(limits, "").await;
    let mut c = fx.clients(&agent);
    let one = accepted(&mut c, submit(&agent, "op-1", &rid(1), true, 40)).await;
    let two = accepted(&mut c, submit(&agent, "op-2", &rid(2), true, 400)).await;
    expire_lease(&fx, &agent).await;
    assert_refused(
        c.inference
            .submit_work(submit(&agent, "op-3", &rid(3), true, 3))
            .await,
        Code::Unavailable,
        wire::LEASE_EXPIRED,
    );
    assert!(view(&agent, SCOPE, "op-3").is_none());
    let cancelled = c
        .control
        .cancel_work(cancel(&agent, &rid(2), &two.completion_ticket))
        .await
        .unwrap()
        .into_inner();
    assert!(cancelled.accepted);
    let (events, _) = stream_all(
        &mut c,
        attach(&agent, "op-1", &rid(1), &one.completion_ticket, 0),
    )
    .await;
    assert_eq!(check_events(&events, 1), "");
    assert_eq!(streamed_text(&events).matches("tok").count(), 40);
    assert_eq!(
        view(&agent, SCOPE, "op-2").unwrap().reason,
        wire::CANCEL_REQUESTED
    );
    agent.shutdown().await;
}

/// Q11: a SubmitWork retry of a journaled key is answered by the journal in every admission state but `poisoned`.
/// While fenced (lease expired) the same request gets the original receipt (`fresh=false`, no second backend call)
/// and a changed payload under that key `operation_conflict`; a new key is still `UNAVAILABLE lease_expired`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_submit_retry_while_fenced_gets_the_original_receipt() {
    let fx = Fixture::new(SLOW).await;
    let limits = "heartbeat_interval_ms = 100\nlost_after_ms = 10000\nrpc_timeout_ms = 300\n";
    let agent = fx.open_agent(limits, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let first = accepted(&mut c, submit(&agent, "op-1", &r, true, 40)).await;
    expire_lease(&fx, &agent).await;
    let retry = accepted(&mut c, submit(&agent, "op-1", &r, true, 40)).await;
    assert_eq!(retry, first);
    assert_eq!(
        events_with(&agent, "event=accept op=dev/op-1 fresh=false"),
        1
    );
    let mut conflict = submit(&agent, "op-1", &r, true, 41);
    redigest(&mut conflict);
    assert_refused(
        c.inference.submit_work(conflict).await,
        Code::AlreadyExists,
        wire::OPERATION_CONFLICT,
    );
    assert_refused(
        c.inference
            .submit_work(submit(&agent, "op-2", &rid(2), true, 3))
            .await,
        Code::Unavailable,
        wire::LEASE_EXPIRED,
    );
    assert!(view(&agent, SCOPE, "op-2").is_none());
    assert_eq!(agent.admission_state(), AdmissionState::Fenced);
    let (events, closed) = stream_all(
        &mut c,
        attach(&agent, "op-1", &r, &first.completion_ticket, 0),
    )
    .await;
    assert!(closed.is_none(), "{closed:?}");
    assert_eq!(check_events(&events, 1), "");
    assert_eq!(fx.dispatched(&r), 1);
    assert_eq!(fx.dispatched(&rid(2)), 0);
    agent.shutdown().await;
}

/// A higher coordinator term ends live ops `coordinator_restart` exactly once (LINK journals the terminal; the op
/// runtime delivers it on the open stream without a second terminal record or event).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_term_fence_ends_live_ops_once() {
    let fx = Fixture::new(SLOW).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let receipt = accepted(&mut c, submit(&agent, "op-1", &r, true, 300)).await;
    let mut stream = c
        .control
        .stream_work(attach(&agent, "op-1", &r, &receipt.completion_ticket, 0))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(stream.message().await.unwrap().unwrap().sequence, 1);
    fx.stub.restart();
    let (mut events, closed) = drain(stream).await;
    assert!(closed.is_none());
    let reason = check_events(&events, 2);
    assert_eq!(reason, wire::COORDINATOR_RESTART);
    events.clear();
    assert_eq!(
        view(&agent, SCOPE, "op-1").unwrap().reason,
        wire::COORDINATOR_RESTART
    );
    eventually("the rid aborted", || fx.aborted(&r)).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(terminal_events(&agent, "op-1"), 1);
    agent.shutdown().await;
}

/// A backend relaunch (new `instance_id`) bumps the generation and ends live ops `model_generation_superseded`
/// exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_backend_relaunch_ends_live_ops_once() {
    let fx = Fixture::new(SLOW).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let receipt = accepted(&mut c, submit(&agent, "op-1", &r, true, 300)).await;
    let mut stream = c
        .control
        .stream_work(attach(&agent, "op-1", &r, &receipt.completion_ticket, 0))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(stream.message().await.unwrap().unwrap().sequence, 1);
    fx.mock.set_server_info_override(Some(confirmed_info(2)));
    let (events, closed) = drain(stream).await;
    assert!(closed.is_none());
    assert_eq!(check_events(&events, 2), wire::MODEL_GENERATION_SUPERSEDED);
    assert_eq!(agent.generation(), 2);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(terminal_events(&agent, "op-1"), 1);
    agent.shutdown().await;
}

/// H2: at shutdown an open StreamWork ends with `UNAVAILABLE worker_restart`, never a clean end of stream.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_ends_an_open_stream_with_worker_restart() {
    let fx = Fixture::new(SLOW).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let receipt = accepted(&mut c, submit(&agent, "op-1", &r, true, 300)).await;
    let mut stream = c
        .control
        .stream_work(attach(&agent, "op-1", &r, &receipt.completion_ticket, 0))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(stream.message().await.unwrap().unwrap().sequence, 1);
    let reader = tokio::spawn(drain(stream));
    tokio::time::timeout(Duration::from_secs(10), agent.shutdown())
        .await
        .expect("shutdown hung on the open stream");
    let (events, closed) = reader.await.unwrap();
    assert!(events.iter().all(|e| !e.terminal), "{events:?}");
    let status = closed.expect("a clean end of stream without a terminal event");
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_eq!(wire::reason(&status), wire::WORKER_RESTART);
}

/// An agent restart makes previous-epoch nonterminal ops terminal `worker_restart` and aborts them on the
/// backend; with no `load_endpoint` an unheld op is cleanup-acknowledged at its worker-local deadline (H3(b)).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_restart_ends_previous_epoch_ops_and_cleans_them_up_at_the_deadline() {
    let fx = Fixture::new(SLOW).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let mut req = submit(&agent, "op-1", &r, true, 300);
    req.authority.as_mut().unwrap().remaining_budget_ms = 2_500;
    accepted(&mut c, req).await;
    eventually("dispatched", || fx.dispatched(&r) == 1).await;
    agent.shutdown().await;
    assert!(!view_on_disk(&fx, "op-1").terminal);

    let agent = Agent::start(fx.config(LIMITS, "")).await.unwrap();
    let v = view(&agent, SCOPE, "op-1").unwrap();
    assert!(v.terminal);
    assert_eq!(v.reason, wire::WORKER_RESTART);
    eventually("the rid aborted", || fx.aborted(&r)).await;
    let mut c = fx.clients(&agent);
    let status = c
        .control
        .get_operation(pb::OperationQuery {
            authority: Some(read_authority(&agent)),
            queried_operation_id: "op-1".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(status.reason_code, wire::WORKER_RESTART);
    assert!(status.terminal);
    let deadline = v.deadline_ms;
    if now_ms() < deadline {
        assert!(!status.cleanup_acknowledged);
        assert_eq!(status.phase, wire::PHASE_TERMINAL);
    }
    eventually("cleanup at the deadline", || {
        view(&agent, SCOPE, "op-1").is_some_and(|v| v.cleanup_acknowledged)
    })
    .await;
    assert!(now_ms() >= deadline);
    // The worker's event line reaches the captured stderr after the view shows the acknowledgement.
    eventually("the cleanup event", || {
        events_with(&agent, "cere-worker event=cleanup op=dev/op-1") >= 1
    })
    .await;
    assert_eq!(
        events_with(&agent, "cere-worker event=cleanup op=dev/op-1"),
        1
    );
    agent.shutdown().await;
}

fn view_on_disk(fx: &Fixture, op: &str) -> OpView {
    use cere_runtime::agent::journal::{OpJournal, OpLimits};
    let limits = OpLimits {
        max_operations: 4096,
        retry_horizon_ms: 3_600_000,
        rpc_timeout_ms: 300,
    };
    let (journal, _) = OpJournal::open(&fx.state_dir(), limits, now_ms()).unwrap();
    journal.get(&key(SCOPE, op)).unwrap()
}

/// GetOperation and ListOperations: statuses per the proto, pages of at most `limit` strictly after the cursor
/// in (scope, op) order, a limit outside 1..=256 `malformed_request`, an unknown op `unknown_operation`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_and_list_operations() {
    let fx = Fixture::new(FAST).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let mut tickets = BTreeMap::new();
    for (n, (scope, op)) in [("b", "op-0"), ("a", "op-2"), ("a", "op-1")]
        .into_iter()
        .enumerate()
    {
        let r = rid(n as u64 + 1);
        let receipt = accepted(&mut c, submit_in(&agent, scope, op, &r, false, 1)).await;
        tickets.insert((scope, op), (r, receipt.completion_ticket));
    }
    let list = |after: (&str, &str), limit: u32| pb::ListOperationsRequest {
        authority: Some(read_authority(&agent)),
        after_scope_id: after.0.into(),
        after_operation_id: after.1.into(),
        limit,
    };
    let page = c
        .control
        .list_operations(list(("", ""), 2))
        .await
        .unwrap()
        .into_inner();
    let keys: Vec<(String, String)> = page
        .operations
        .iter()
        .map(|s| (s.scope_id.clone(), s.operation_id.clone()))
        .collect();
    assert_eq!(
        keys,
        [("a".into(), "op-1".into()), ("a".into(), "op-2".into())]
    );
    assert!(page.more);
    let rest = c
        .control
        .list_operations(list(("a", "op-2"), 2))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(rest.operations.len(), 1);
    assert_eq!(rest.operations[0].scope_id, "b");
    assert!(!rest.more);
    for limit in [0, 257] {
        assert_refused(
            c.control.list_operations(list(("", ""), limit)).await,
            Code::InvalidArgument,
            wire::MALFORMED_REQUEST,
        );
    }

    let get = |scope: &str, op: &str| pb::OperationQuery {
        authority: Some(pb::Authority {
            scope_id: scope.into(),
            ..read_authority(&agent)
        }),
        queried_operation_id: op.into(),
    };
    let s = c
        .control
        .get_operation(get("a", "op-1"))
        .await
        .unwrap()
        .into_inner();
    let (r, ticket) = &tickets[&("a", "op-1")];
    assert_eq!(s.operation_id, "op-1");
    assert_eq!(s.scope_id, "a");
    assert_eq!(&s.request_id, r);
    assert_eq!(s.attempt, 1);
    assert_eq!(&s.completion_ticket, ticket);
    assert_eq!(s.method, wire::METHOD_SUBMIT);
    assert_eq!(s.coordinator_term, TERM);
    assert_eq!(s.worker_epoch, agent.epoch());
    assert_eq!(s.model_generation, agent.generation());
    assert!([
        wire::PHASE_ACCEPTED,
        wire::PHASE_TERMINAL,
        wire::PHASE_RELEASED
    ]
    .contains(&s.phase.as_str()));
    // Empty scope: the unique op with that id.
    let s = c
        .control
        .get_operation(get("", "op-0"))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(s.scope_id, "b");
    assert_refused(
        c.control.get_operation(get("a", "op-9")).await,
        Code::NotFound,
        wire::UNKNOWN_OPERATION,
    );
    agent.shutdown().await;
}

/// A journal write failure (here: the compaction that follows an op's records) poisons the agent: Submit and
/// Cancel are `UNAVAILABLE poisoned` with no backend call.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_journal_write_failure_poisons_the_agent() {
    let fx = Fixture::new(FAST).await;
    let agent = fx.open_agent(LIMITS, "max_operations = 2\n").await;
    std::fs::create_dir(fx.state_dir().join("operations.jsonl.compact.tmp")).unwrap();
    let mut c = fx.clients(&agent);
    let receipt = accepted(&mut c, submit(&agent, "op-1", &rid(1), false, 1)).await;
    eventually("poisoned", || {
        agent.admission_state() == AdmissionState::Poisoned
    })
    .await;
    assert_eq!(
        events_with(&agent, "cere-worker event=poisoned reason=journal_write"),
        1
    );
    assert_refused(
        c.inference
            .submit_work(submit(&agent, "op-2", &rid(2), false, 1))
            .await,
        Code::Unavailable,
        wire::POISONED,
    );
    assert_refused(
        c.control
            .cancel_work(cancel(&agent, &rid(1), &receipt.completion_ticket))
            .await,
        Code::Unavailable,
        wire::POISONED,
    );
    assert_eq!(fx.dispatched(&rid(2)), 0);
    agent.shutdown().await;
}

/// A journal write failure inside CancelWork (here: the target's `cancel_requested` terminal) poisons the agent
/// and the target still ends locally, like any op end: its stream ends `cancel_requested` and its rid is
/// aborted, with no `terminal` event line for the unjournaled terminal. The failure needs a file-size limit, so
/// it runs in a child process.
#[test]
fn a_failed_cancel_write_still_ends_the_target() {
    if std::env::var_os(CANCEL_WRITE_FAILURE_ENV).is_some() {
        return;
    }
    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            "cancel_write_failure_child",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CANCEL_WRITE_FAILURE_ENV, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "child failed: {:?}\n{stdout}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("CANCEL-WRITE-FAILURE-CHILD-OK"), "{stdout}");
}

const CANCEL_WRITE_FAILURE_ENV: &str = "CERE_AGENT_CANCEL_WRITE_FAILURE";

/// Child of [`a_failed_cancel_write_still_ends_the_target`]; inert unless `CERE_AGENT_CANCEL_WRITE_FAILURE` is
/// set.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_write_failure_child() {
    if std::env::var_os(CANCEL_WRITE_FAILURE_ENV).is_none() {
        return;
    }
    let fx = Fixture::new(SLOW).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let receipt = accepted(&mut c, submit(&agent, "op-1", &r, true, 400)).await;
    let ticket = receipt.completion_ticket.clone();
    let mut stream = c
        .control
        .stream_work(attach(&agent, "op-1", &r, &ticket, 0))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(stream.message().await.unwrap().unwrap().sequence, 1);
    let cancel_op = format!("{r}/cancel/1");
    limit_journal_to_the_cancel_records(&fx, "op-1", &cancel_op);
    let cancelled = c
        .control
        .cancel_work(cancel(&agent, &r, &ticket))
        .await
        .unwrap()
        .into_inner();
    assert!(cancelled.accepted);
    assert_eq!(agent.admission_state(), AdmissionState::Poisoned);
    assert_eq!(
        events_with(&agent, "cere-worker event=poisoned reason=journal_write"),
        1
    );
    let own = view(&agent, SCOPE, &cancel_op).unwrap();
    assert!(own.terminal && own.cleanup_acknowledged, "{own:?}");
    assert!(
        !view(&agent, SCOPE, "op-1").unwrap().terminal,
        "the target's terminal write failed"
    );
    let (events, closed) = drain(stream).await;
    assert!(closed.is_none(), "{closed:?}");
    assert_eq!(check_events(&events, 2), wire::CANCEL_REQUESTED);
    eventually("the rid aborted", || fx.aborted(&r)).await;
    assert_eq!(terminal_events(&agent, "op-1"), 0);
    let mut stdout = std::io::stdout();
    stdout
        .write_all(b"CANCEL-WRITE-FAILURE-CHILD-OK\n")
        .unwrap();
    stdout.flush().unwrap();
    agent.shutdown().await;
}

/// Caps this process's file size so the journal takes the three records of the fresh cancel `cancel_op` of
/// `target` (accepted, terminal, cleanup) and then only a 20-byte short write; `SIGXFSZ` is ignored, so that next
/// append fails `EFBIG`. The cancel's accepted line is the target's with the operation id and method swapped.
fn limit_journal_to_the_cancel_records(fx: &Fixture, target: &str, cancel_op: &str) {
    let text = std::fs::read_to_string(fx.state_dir().join(OPERATIONS_FILE)).unwrap();
    let target_accepted = text
        .lines()
        .find(|line| {
            line.contains(r#""type":"accepted""#)
                && line.contains(&format!(r#""operation_id":"{target}""#))
        })
        .unwrap();
    let accepted = target_accepted.len() + 1 + cancel_op.len() + wire::METHOD_CANCEL.len()
        - target.len()
        - wire::METHOD_SUBMIT.len();
    let line = |record: Value| serde_json::to_vec(&record).unwrap().len() + 1;
    let at_ms = now_ms();
    let terminal = line(json!({
        "type": "terminal", "scope_id": SCOPE, "operation_id": cancel_op, "reason": "", "at_ms": at_ms,
    }));
    let cleanup = line(json!({
        "type": "cleanup", "scope_id": SCOPE, "operation_id": cancel_op, "at_ms": at_ms,
    }));
    let limit = libc::rlimit {
        rlim_cur: (text.len() + accepted + terminal + cleanup + 20) as u64,
        rlim_max: libc::RLIM_INFINITY,
    };
    // SAFETY: plain syscalls on a valid struct and a valid handler.
    unsafe {
        assert_ne!(libc::signal(libc::SIGXFSZ, libc::SIG_IGN), libc::SIG_ERR);
        assert_eq!(libc::setrlimit(libc::RLIMIT_FSIZE, &limit), 0);
    }
}

/// Q8: every agent refusal is a `wire` reason. Before the backend identity is confirmed (here never: the backend
/// reports no `tp_size`) DescribeResources, once authorized, is `UNAVAILABLE backend_unconfirmed`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn describe_without_a_confirmed_backend_is_backend_unconfirmed() {
    let fx = Fixture::with(support::launched(1, 1.5), FAST).await;
    let agent = Agent::start(fx.config(LIMITS, "")).await.unwrap();
    let mut c = fx.clients(&agent);
    let mut wrong = read_authority(&agent);
    wrong.target.as_mut().unwrap().cluster_id = "other".into();
    assert_refused(
        c.control
            .describe_resources(pb::InventoryRequest {
                authority: Some(wrong),
            })
            .await,
        Code::FailedPrecondition,
        wire::WRONG_CLUSTER,
    );
    assert_refused(
        c.control
            .describe_resources(pb::InventoryRequest {
                authority: Some(read_authority(&agent)),
            })
            .await,
        Code::Unavailable,
        wire::BACKEND_UNCONFIRMED,
    );
    assert_eq!(agent.admission_state(), AdmissionState::Fenced);
    agent.shutdown().await;
}

/// C2 (D5): the agent forwards the Submit's body bytes, not a re-serialization - the `tools` subtree keeps its
/// key order and number spellings (`1.0`, `1E5`, 2^70, `\u00e9`) on the way to SGLang.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c2_the_agent_forwards_the_body_bytes_verbatim() {
    let fx = Fixture::new(FAST).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    let r = rid(1);
    let mut req = submit(&agent, "op-1", &r, false, 3);
    let input = format!(
        r#"{{"rid":"{r}","model":"m","messages":[{{"role":"user","content":"{PROMPT}"}}],"max_completion_tokens":3,"temperature":0,"stream":false,"tools":[{{"type":"function","function":{{"name":"f","parameters":{{"type":"object","x-a":1.0,"x-b":1E5,"x-c":1180591620717411303424,"x-d":"Z\u00e9"}}}}}}],"tool_choice":"auto"}}"#
    );
    req.bounded_inference_input_json = input.clone().into_bytes();
    redigest(&mut req);
    let before = fx.mock.received_bodies().len();
    let receipt = accepted(&mut c, req).await;
    let (events, _) = stream_all(
        &mut c,
        attach(&agent, "op-1", &r, &receipt.completion_ticket, 0),
    )
    .await;
    assert_eq!(check_events(&events, 1), "");
    let sent = fx.mock.received_bodies().split_off(before);
    assert_eq!(sent, [input.into_bytes()]);
    agent.shutdown().await;
}

/// C1 (D2/D5) remote half: SGLang's tool schema 400 ends the op with the API code `invalid_value` as its
/// reason, which the coordinator maps back to a client 400.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c1_a_tool_schema_refusal_ends_the_op_invalid_value() {
    let fx = Fixture::new(FAST).await;
    let agent = fx.open_agent(LIMITS, "").await;
    let mut c = fx.clients(&agent);
    fx.mock.set_fault(cere_mock_sglang::Fault::Reject {
        code: 400,
        message: "Tool 0 function has invalid 'parameters' schema: bad".to_owned(),
    });
    let r = rid(1);
    let receipt = accepted(&mut c, submit(&agent, "op-1", &r, true, 3)).await;
    let (events, _) = stream_all(
        &mut c,
        attach(&agent, "op-1", &r, &receipt.completion_ticket, 0),
    )
    .await;
    assert_eq!(check_events(&events, 1), "invalid_value");
    agent.shutdown().await;
}

/// D1 on the remote path (M24 Z2 R1-06): an agent whose backend reports a `tool_call_parser` other than
/// `qwen3_coder` (or none) ends a tools Submit `unsupported_parameter` before any backend call (the coordinator
/// maps it to a client 400 naming `tools`); tool-free Submits are still served.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn d1_a_backend_without_the_qwen3_coder_parser_refuses_tools() {
    for parser in [Some("hermes"), None] {
        let mut info = confirmed_info(1);
        info.tool_call_parser = parser.map(str::to_owned);
        let fx = Fixture::with(info, FAST).await;
        let agent = fx.open_agent(LIMITS, "").await;
        let mut c = fx.clients(&agent);
        let before = fx.mock.received_bodies().len();
        for (n, stream) in [(1, false), (2, true)] {
            let (r, op) = (rid(n), format!("op-{n}"));
            let mut req = submit(&agent, &op, &r, stream, 3);
            let mut input: Value =
                serde_json::from_slice(&req.bounded_inference_input_json).unwrap();
            input["tools"] = json!([{"type": "function", "function": {"name": "f"}}]);
            req.bounded_inference_input_json = serde_json::to_vec(&input).unwrap();
            redigest(&mut req);
            let receipt = accepted(&mut c, req).await;
            let (events, _) = stream_all(
                &mut c,
                attach(&agent, &op, &r, &receipt.completion_ticket, 0),
            )
            .await;
            assert_eq!(
                check_events(&events, 1),
                "unsupported_parameter",
                "{parser:?} stream {stream}"
            );
        }
        assert_eq!(fx.mock.received_bodies().len(), before, "{parser:?}");
        let r = rid(3);
        let receipt = accepted(&mut c, submit(&agent, "op-3", &r, false, 3)).await;
        let (events, _) = stream_all(
            &mut c,
            attach(&agent, "op-3", &r, &receipt.completion_ticket, 0),
        )
        .await;
        assert_eq!(check_events(&events, 1), "", "{parser:?}");
        assert_eq!(fx.mock.received_bodies().len(), before + 1, "{parser:?}");
        agent.shutdown().await;
    }
}
