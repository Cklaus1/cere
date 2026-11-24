//! Coordinator restart with remote work (m35 design note Q15, P7; build-loop-m35 design note track ORPHAN-B): a
//! request that a crashed `cered` left `dispatched` to remote worker `A` is recovered by the next start as failed
//! `worker_restart` and held as an orphan charged to `A`'s cap. The hold asks the agent for the request's
//! operation (`GetOperation`, every second) only once the agent registered in the new term: a
//! `cleanup_acknowledged` report, or an operation the agent no longer knows, releases the slot, journaled
//! `cleanup_acknowledged`; without one the slot is released at the retention bound, journaled `retention`. An
//! orphan whose journaled endpoint is no longer `A`'s is charged to no worker until its retention bound, and its
//! agent is never asked. A live remote orphan (H2: its client went away while the agent kept the operation) still
//! held when `cered` restarts is charged to `A` again by the next start the same way.
//!
//! The seeded journal plays the crashed incarnation; the in-process stub agent (`support::stub_agent`) journals
//! the request's operation under that term and fences it (terminal `coordinator_restart`, cleanup not
//! acknowledged) when the new term registers. The local worker never turns ready, so only `A` takes requests. The
//! P7 smoke takes a slot of `A`: at cap 1 the orphan keeps `A` out of Ready until its release, so the queueing
//! test runs `A` at cap 2 with the other slot busy.
mod support;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cere_mock_sglang::{MockConfig, MockHandle, MockSglang, ServerInfo};
use cere_runtime::config::RuntimeConfig;
use cere_runtime::fed::wire;
use serde_json::Value;
use support::stub_agent::{StubAgent, StubOp};
use support::*;

const A: &str = "worker-a";
const PRINCIPAL_A: &str = "cere-worker-a";
const TIMINGS: &str =
    "heartbeat_interval_ms = 100\nadmission_lease_ms = 1000\nsuspect_after_ms = 2000\n\
                       lost_after_ms = 4000\n";
/// The stub agent's heartbeat period.
const HEARTBEAT: Duration = Duration::from_millis(50);
/// The orphan hold's poll period (`orphan_load_poll_ms`, at its minimum).
const POLL: Duration = Duration::from_millis(1_000);
/// The seeded request's scope (token A).
const SCOPE: &str = "a";
/// The stub agent's epoch and model generation, before and after the restart.
const AGENT_EPOCH: u64 = 1;
const GENERATION: u64 = 1;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
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

/// The stub agent's backend: a chat runs until its `max_completion_tokens` (5 ms per token).
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

/// `[[federation.workers]]` for `A` dialled at the stub with cap `max_inflight`, pinned like [`Federation::toml`].
fn worker_toml(stub: &StubAgent, max_inflight: u64) -> String {
    format!(
        "[[federation.workers]]\nworker_id = \"{A}\"\nendpoint = \"https://{}\"\n\
         principal = \"{PRINCIPAL_A}\"\nmax_inflight = {max_inflight}\n\n[federation.workers.expect]\n\
         served_model_name = \"{MODEL}\"\ntp_size = 1\nkv_cache_dtype = \"fp8_e4m3\"\n\n",
        stub.addr()
    )
}

/// Journals, in `config`'s state dir, one request of scope [`SCOPE`] admitted at `admitted_ms` and dispatched to
/// `A` at [`AGENT_EPOCH`] through `endpoint` (`None` = `A`'s configured one) that never reached a terminal, as a
/// crashed `cered` with `config`'s workers left it. Returns the rid and that incarnation's epoch (its term).
fn seed_remote(config: &RuntimeConfig, endpoint: Option<&str>, admitted_ms: u64) -> (String, u64) {
    use cere_runtime::digest::{DigestKeys, RequestDigestInput};
    use cere_runtime::ledger::{AcceptOutcome, AcceptRequest, Ledger, LedgerLimits, WorkerTarget};
    let dir = &config.ledger.state_dir;
    let keys = Arc::new(DigestKeys::open(dir).unwrap());
    let fed = config.federation.as_ref().expect("[federation] configured");
    let remote = |endpoint: &str| WorkerTarget::Remote {
        endpoint: endpoint.to_owned(),
    };
    let mut workers: BTreeMap<String, WorkerTarget> = config.deployments[0]
        .replica_list(&config.scheduler.devices)
        .into_iter()
        .map(|r| {
            (
                r.device,
                WorkerTarget::Local {
                    base_url: r.base_url,
                },
            )
        })
        .collect();
    for w in &fed.workers {
        workers.insert(w.worker_id.clone(), remote(&w.endpoint));
    }
    let (mut ledger, recovery) = Ledger::open(
        LedgerLimits::from_config(config),
        keys,
        dir,
        &workers,
        admitted_ms,
    )
    .unwrap();
    let rid = format!("cere-{:020}-{:020}-0123456789abcdef", recovery.epoch, 1);
    let digest = ledger
        .digest_keys()
        .scope_key(SCOPE)
        .request_digest(&RequestDigestInput {
            request: &chat_body(false),
            session_id: None,
            expected_revision: None,
            served_model: MODEL,
            deadline_ms: 1_000,
            config_hash: "seed",
            surface: None,
        });
    let request = AcceptRequest {
        scope: SCOPE.into(),
        rid: rid.clone(),
        key: None,
        digest,
        stream: false,
        surface: None,
        session: None,
    };
    assert_eq!(
        ledger.accept(request, admitted_ms),
        Ok(AcceptOutcome::Accepted)
    );
    ledger.mark_queued(&rid, admitted_ms).unwrap();
    ledger.mark_running(&rid, admitted_ms).unwrap();
    let configured = &fed.workers[0].endpoint;
    let target = remote(endpoint.unwrap_or(configured));
    ledger
        .dispatched(&rid, A, AGENT_EPOCH, &target, admitted_ms)
        .unwrap();
    (rid, recovery.epoch)
}

/// A `cered` restarted over a seeded journal ([`seed_remote`]), with the unlinked stub agent `A` serving chat.
struct Restarted {
    fed: Federation,
    stub: StubAgent,
    /// The stub agent's backend ([`generating`]).
    _mock: MockHandle,
    h: Harness,
    rid: String,
    /// The crashed incarnation's term.
    old_term: u64,
    admitted_ms: u64,
}

impl Restarted {
    /// The stub agent `A` (cap `max_inflight`, 1 or 2) and a federated `cered` opened over the journal of
    /// [`seed_remote`] (`endpoint`, admitted now). Orphan holds poll every second, terminals are retained
    /// `retention_ms`, and the global and scope caps (2, or 4 at cap 2) never bind before `A`'s.
    async fn start(retention_ms: u64, endpoint: Option<&str>, max_inflight: u64) -> Restarted {
        let fed = Federation::new();
        let stub = StubAgent::start(&fed, A, PRINCIPAL_A, AGENT_EPOCH).await;
        let mock = MockSglang::start(generating()).await.unwrap();
        stub.serve_chat(&mock.base_url());
        let mut config = fed.toml(TIMINGS, &[]);
        config.push_str(&worker_toml(&stub, max_inflight));
        let ledger = format!("terminal_retention_ms = {retention_ms}\norphan_load_poll_ms = 1000");
        let mut tweaks = vec![("terminal_retention_ms = 3600000", ledger.as_str())];
        if max_inflight == 2 {
            tweaks.push(("max_backend_inflight = 2", "max_backend_inflight = 4"));
            tweaks.push(("per_scope_inflight = 2", "per_scope_inflight = 4"));
        }
        let admitted_ms = now_ms();
        let mut seeded = None;
        let h = Harness::start_federated_seeded(
            unready_local(),
            &tweaks,
            &config,
            fast_tuning(),
            |config| seeded = Some(seed_remote(config, endpoint, admitted_ms)),
        )
        .await;
        let (rid, old_term) = seeded.unwrap();
        assert!(h.app.epoch() > old_term, "a new term");
        Restarted {
            fed,
            stub,
            _mock: mock,
            h,
            rid,
            old_term,
            admitted_ms,
        }
    }

    /// The live SubmitWork operation of the seeded request, as the agent journaled it in the crashed term.
    fn live_op(&self) -> StubOp {
        StubOp::submit(SCOPE, &self.rid, AGENT_EPOCH, GENERATION, self.old_term)
    }

    fn link(&mut self) {
        self.stub.link(&self.fed, HEARTBEAT);
    }

    async fn until_ready(&self) {
        until_ready(&self.h).await;
    }

    async fn load(&self) -> Value {
        load(&self.h).await
    }

    /// The journaled `orphan_released` records of the seeded request.
    fn releases(&self) -> Vec<Value> {
        records(&self.h, "orphan_released", &self.rid)
    }

    /// Waits (`bound`) until the seeded request's release is journaled; returns the records.
    async fn until_released(&self, bound: Duration) -> Vec<Value> {
        let until = tokio::time::Instant::now() + bound;
        loop {
            let released = self.releases();
            if !released.is_empty() {
                return released;
            }
            assert!(
                tokio::time::Instant::now() < until,
                "the orphan was not released within {bound:?}"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// A unary chat, answered within 10 s.
    async fn served(&self) {
        let r = tokio::time::timeout(
            Duration::from_secs(10),
            self.h.chat(&chat_body(false)).send(),
        )
        .await
        .expect("served")
        .unwrap();
        assert_eq!(r.status(), 200);
    }
}

async fn until_ready(h: &Harness) {
    readyz_until(h, "worker-a ready", |b| b["workers"][A]["state"] == "ready").await;
}

/// `A`'s admission view in `/v1/cere/stats`: its cap and the in-flight of scope `a` on it.
async fn load(h: &Harness) -> Value {
    let r = h.get("/v1/cere/stats", TOKEN_A).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    body["workers"][A].clone()
}

/// The journaled records of type `kind` of request `rid`.
fn records(h: &Harness, kind: &str, rid: &str) -> Vec<Value> {
    journal(h.state_dir.path())
        .into_iter()
        .filter(|r| r["type"] == kind && r["rid"] == rid)
        .collect()
}

/// Q15: the recovered request reads failed `worker_restart` and its orphan holds one of `A`'s two slots; with the
/// other one busy, a new request stays queued while the agent, registered in the new term, reports the operation
/// fenced but not cleaned up. Once the agent reports `cleanup_acknowledged`, the next poll journals the release
/// (`cleanup_acknowledged`) and the queued request is served.
#[tokio::test(flavor = "multi_thread")]
async fn a_remote_orphan_is_held_until_the_agent_acknowledges_its_cleanup() {
    let mut r = Restarted::start(3_600_000, None, 2).await;
    r.stub.add_op(r.live_op());
    let status = r.h.terminal(&r.rid).await;
    assert_eq!(status["state"], "failed", "{status}");
    assert_eq!(status["reason"], "worker_restart", "{status}");
    let held = r.load().await;
    assert_eq!(held["cap"], 2, "{held}");
    assert_eq!(held["inflight"], 1, "the orphan's slot: {held}");
    r.link();
    r.until_ready().await;
    let busy = r.h.chat(&long_body(true)).send().await.unwrap();
    assert_eq!(busy.status(), 200);
    let mut busy = SseReader::new(busy);
    assert!(busy.next().await.is_some(), "the busy request runs");
    assert_eq!(r.load().await["inflight"], 2);
    let next = tokio::spawn(r.h.chat(&chat_body(false)).send());
    // At least one poll in the new term sees the cleanup unacknowledged.
    tokio::time::sleep(2 * POLL + Duration::from_millis(200)).await;
    assert!(!next.is_finished(), "the next request was answered");
    assert!(r.stub.count("GetOperation") >= 1, "{:?}", r.stub.calls());
    assert!(r.releases().is_empty(), "{:?}", r.releases());
    assert_eq!(r.load().await["inflight"], 2);
    let op = r.stub.op(SCOPE, &r.rid).expect("the Submit op");
    assert!(op.terminal, "fenced by the new term: {op:?}");
    assert_eq!(op.reason, wire::COORDINATOR_RESTART, "{op:?}");
    assert!(!op.cleanup_acknowledged, "{op:?}");
    r.stub.add_op(StubOp {
        cleanup_acknowledged: true,
        ..op
    });
    let response = tokio::time::timeout(Duration::from_secs(10), next)
        .await
        .expect("served once cleanup was acknowledged")
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), 200);
    let released = r.releases();
    assert_eq!(released.len(), 1, "{released:?}");
    assert_eq!(released[0]["cause"], "cleanup_acknowledged", "{released:?}");
    drop(busy);
}

/// Q15: an agent that never acknowledges the cleanup of the fenced operation (no run holds it any more, so nothing
/// acknowledges it) leaves the slot held until the retention bound (`admitted_ms + terminal_retention_ms`): the
/// release is journaled `retention`, and `A` (cap 1) then turns Ready and serves.
#[tokio::test(flavor = "multi_thread")]
async fn an_unacknowledged_remote_orphan_is_released_at_the_retention_bound() {
    let mut r = Restarted::start(1_000, None, 1).await;
    r.stub.add_op(r.live_op());
    r.link();
    let released = r.until_released(Duration::from_secs(5)).await;
    assert_eq!(released.len(), 1, "{released:?}");
    assert_eq!(released[0]["cause"], "retention", "{released:?}");
    assert!(released[0]["at_ms"].as_u64().unwrap() >= r.admitted_ms + 1_000);
    r.until_ready().await;
    r.served().await;
    let op = r.stub.op(SCOPE, &r.rid).expect("the Submit op");
    assert!(!op.cleanup_acknowledged, "{op:?}");
    assert_eq!(r.releases().len(), 1);
}

/// Q15: nothing releases the orphan before the agent's accepted Register in the new term, even an agent that
/// already cleaned the operation up: the unlinked agent is never asked, past two poll periods. Once it registers,
/// the next poll releases the slot, and `A` (cap 1) turns Ready and serves.
#[tokio::test(flavor = "multi_thread")]
async fn no_release_before_the_agent_registers_in_the_new_term() {
    let mut r = Restarted::start(3_600_000, None, 1).await;
    // The work ended by itself after the crash and the agent released its resources.
    r.stub.add_op(StubOp {
        terminal: true,
        cleanup_acknowledged: true,
        ..r.live_op()
    });
    tokio::time::sleep(2 * POLL + Duration::from_millis(200)).await;
    assert!(r.releases().is_empty(), "{:?}", r.releases());
    assert_eq!(r.load().await["inflight"], 1);
    assert!(r.stub.calls().is_empty(), "{:?}", r.stub.calls());
    let linked = now_ms();
    r.link();
    let released = r.until_released(Duration::from_secs(5)).await;
    assert_eq!(released.len(), 1, "{released:?}");
    assert_eq!(released[0]["cause"], "cleanup_acknowledged", "{released:?}");
    assert!(
        released[0]["at_ms"].as_u64().unwrap() >= linked,
        "released before the Register: {released:?}"
    );
    assert_eq!(r.stub.term(), r.h.app.epoch());
    r.until_ready().await;
    r.served().await;
}

/// Q15: an agent that no longer knows the operation (`GetOperation` NOT_FOUND `unknown_operation`, e.g. its journal
/// was reset) has nothing left to clean up: the first poll after its Register releases the slot, journaled
/// `cleanup_acknowledged`.
#[tokio::test(flavor = "multi_thread")]
async fn an_operation_unknown_to_the_agent_releases_the_orphan() {
    let mut r = Restarted::start(3_600_000, None, 1).await;
    assert_eq!(r.load().await["inflight"], 1);
    r.link();
    let released = r.until_released(Duration::from_secs(5)).await;
    assert_eq!(released.len(), 1, "{released:?}");
    assert_eq!(released[0]["cause"], "cleanup_acknowledged", "{released:?}");
    assert!(r.stub.op(SCOPE, &r.rid).is_none());
    assert!(r.stub.count("GetOperation") >= 1, "{:?}", r.stub.calls());
    r.until_ready().await;
    r.served().await;
}

/// Q15 / M3 Q9: an orphan whose journaled endpoint is not `A`'s configured one is charged to no worker (`A`'s
/// cap stays free) until its retention bound, journaled `retention`; its agent is never asked about it.
#[tokio::test(flavor = "multi_thread")]
async fn an_orphan_at_another_endpoint_is_held_globally_until_the_retention_bound() {
    let mut r = Restarted::start(3_000, Some("https://127.0.0.1:9"), 1).await;
    r.stub.add_op(StubOp {
        terminal: true,
        cleanup_acknowledged: true,
        ..r.live_op()
    });
    let status = r.h.terminal(&r.rid).await;
    assert_eq!(status["reason"], "worker_restart", "{status}");
    assert_eq!(r.load().await["inflight"], 0, "charged to {A}");
    assert_eq!(
        r.h.app.admission_snapshot().inflight,
        1,
        "the global charge"
    );
    r.link();
    let released = r.until_released(Duration::from_secs(8)).await;
    assert_eq!(released.len(), 1, "{released:?}");
    assert_eq!(released[0]["cause"], "retention", "{released:?}");
    assert!(released[0]["at_ms"].as_u64().unwrap() >= r.admitted_ms + 3_000);
    assert_eq!(r.stub.count("GetOperation"), 0, "{:?}", r.stub.calls());
}

/// Q15 / H2: a remote request whose client went away (`client_disconnected`) while the agent keeps its operation
/// uncleaned holds its slot on `A`, and the hold is journaled when it starts. `cered` restarts during that hold:
/// the next start charges the slot to `A` again (cap 2, the other slot busy: a new request waits) while the agent,
/// registered in the new term, still reports the operation not cleaned up. Its `cleanup_acknowledged` then
/// releases the slot, journaled once, and the waiting request is served.
#[tokio::test(flavor = "multi_thread")]
async fn a_held_remote_orphan_is_charged_again_after_a_restart() {
    let fed = Federation::new();
    let mut stub = StubAgent::start(&fed, A, PRINCIPAL_A, AGENT_EPOCH).await;
    let mock = MockSglang::start(generating()).await.unwrap();
    stub.serve_chat(&mock.base_url());
    let mut config = fed.toml(TIMINGS, &[]);
    config.push_str(&worker_toml(&stub, 2));
    let tweaks = [
        (
            "terminal_retention_ms = 3600000",
            "terminal_retention_ms = 3600000\norphan_load_poll_ms = 1000",
        ),
        ("max_backend_inflight = 2", "max_backend_inflight = 4"),
        ("per_scope_inflight = 2", "per_scope_inflight = 4"),
    ];
    let h =
        Harness::start_federated_seeded(unready_local(), &tweaks, &config, fast_tuning(), |_| {})
            .await;
    stub.link(&fed, HEARTBEAT);
    until_ready(&h).await;
    stub.hold_cleanup(true);
    let r = h.chat(&long_body(true)).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let rid = r.headers()["x-cere-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let mut sse = SseReader::new(r);
    assert!(sse.next().await.is_some(), "the request runs");
    assert!(sse.next().await.is_some(), "the request runs");
    drop(sse);
    let status = h.terminal(&rid).await;
    assert_eq!(status["state"], "cancelled", "{status}");
    assert_eq!(status["reason"], "client_disconnected", "{status}");
    // The hold polls the agent once it started.
    let until = tokio::time::Instant::now() + Duration::from_secs(5);
    while stub.count("GetOperation") == 0 {
        assert!(
            tokio::time::Instant::now() < until,
            "no poll: {:?}",
            stub.calls()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(load(&h).await["inflight"], 1, "the held slot");
    let op = stub.op(SCOPE, &rid).expect("the Submit op");
    assert!(op.terminal && !op.cleanup_acknowledged, "{op:?}");

    let h = h.restart().await;
    let status = h.terminal(&rid).await;
    assert_eq!(status["reason"], "client_disconnected", "{status}");
    assert_eq!(load(&h).await["inflight"], 1, "charged to {A} again");
    assert_eq!(records(&h, "orphan_released", &rid), Vec::<Value>::new());
    until_ready(&h).await;
    let busy = h.chat(&long_body(true)).send().await.unwrap();
    assert_eq!(busy.status(), 200);
    let mut busy = SseReader::new(busy);
    assert!(busy.next().await.is_some(), "the busy request runs");
    assert_eq!(load(&h).await["inflight"], 2);
    let polled = stub.count("GetOperation");
    let next = tokio::spawn(h.chat(&chat_body(false)).send());
    // At least one poll in the new term sees the cleanup unacknowledged.
    tokio::time::sleep(2 * POLL + Duration::from_millis(200)).await;
    assert!(!next.is_finished(), "the next request was answered");
    assert!(stub.count("GetOperation") > polled, "{:?}", stub.calls());
    assert_eq!(records(&h, "orphan_released", &rid), Vec::<Value>::new());
    stub.hold_cleanup(false);
    let response = tokio::time::timeout(Duration::from_secs(10), next)
        .await
        .expect("served once cleanup was acknowledged")
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), 200);
    let released = records(&h, "orphan_released", &rid);
    assert_eq!(released.len(), 1, "{released:?}");
    assert_eq!(released[0]["cause"], "cleanup_acknowledged", "{released:?}");
    drop(busy);
}
