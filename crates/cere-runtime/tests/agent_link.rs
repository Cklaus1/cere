//! LINK track (m35 design note row P5, control half; harness "AGENT split"): the `cere-worker` agent's durable
//! state, backend identity and generation, Register/Heartbeat loop, lease, term fence, epoch jumps, admission state,
//! events, listener (Authz first, `DescribeResources`) and binary, against the in-process stub coordinator
//! (`support::stub_coordinator`, which enforces the coordinator's rules) and `cere-mock-sglang`.
//!
//! Operations are seeded straight into the agent's journal (`Shared::journal`): SubmitWork is track WORK's.

mod support;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use cere_mock_sglang::{MockConfig, MockHandle, MockSglang, ServerInfo};
use cere_proto::worker::v2 as pb;
use cere_runtime::agent::config::AgentConfig;
use cere_runtime::agent::journal::{OpIdentity, OpJournal, OpKey, OpLimits, OpView};
use cere_runtime::agent::state::AgentState;
use cere_runtime::agent::{Agent, AgentHandle};
use cere_runtime::fed::tls::Pki;
use cere_runtime::fed::wire::{self, AdmissionState, Inventory, ModelState};
use cere_runtime::sglang::Rid;
use pb::worker_service_client::WorkerServiceClient;
use serde_json::json;
use support::pki::TestPki;
use support::stub_coordinator::{StubConfig, StubCoordinator};
use support::StateDir;
use tonic::transport::{Channel, Endpoint};
use tonic::Code;

const CLUSTER: &str = "lab-test";
const WORKER: &str = "worker-a";
const WORKER_PRINCIPAL: &str = "cere-worker-a";
const COORD_PRINCIPAL: &str = "cere-coordinator";
const COORD_ID: &str = "coord-1";
const MODEL: &str = "Qwen3.8-Flash-Next";
/// The stub coordinator's first term.
const TERM: u64 = 5;
const LEASE_MS: u64 = 500;
/// Heartbeat 100 ms, RPC timeout 300 ms, lost after 3 s unless a test needs it shorter.
const LIMITS: &str = "heartbeat_interval_ms = 100\nlost_after_ms = 3000\nrpc_timeout_ms = 300\n";

/// A backend `/get_server_info` the agent confirms: pins (`tp_size` as an extra key, the mock does not model it)
/// and a launch (`random_seed`, `startup_time`); a new `seed` is a relaunched backend (new `instance_id`).
fn confirmed_info(seed: u64) -> ServerInfo {
    let mut info = support::launched(seed, 1.5);
    info.extra.insert("tp_size".into(), json!(1));
    info
}

/// No `tp_size`: the agent cannot confirm the identity.
fn unconfirmed_info() -> ServerInfo {
    support::launched(1, 1.5)
}

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
    async fn new() -> Fixture {
        Fixture::with(confirmed_info(1)).await
    }

    async fn with(info: ServerInfo) -> Fixture {
        let root = StateDir::new();
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
                lost_after: Duration::from_secs(3),
                bindings: BTreeMap::from([(WORKER_PRINCIPAL.to_owned(), WORKER.to_owned())]),
            },
        )
        .await;
        let mock = MockSglang::start(MockConfig {
            server_info: info,
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

    fn durable(&self) -> AgentState {
        AgentState::load(&self.state_dir()).unwrap()
    }

    fn toml(&self, port: u16, limits: &str, journal: &str) -> String {
        format!(
            "cluster_id = \"{CLUSTER}\"\nworker_id = \"{WORKER}\"\nlisten = \"127.0.0.1:{port}\"\n\
             principal = \"{WORKER_PRINCIPAL}\"\nca_file = \"{}\"\ncert_file = \"{}\"\nkey_file = \"{}\"\n\
             state_dir = \"{}\"\n\n[coordinator]\nendpoint = \"{}\"\nprincipal = \"{COORD_PRINCIPAL}\"\n\
             coordinator_id = \"{COORD_ID}\"\n\n[backend]\nbase_url = \"{}\"\nserved_model_name = \"{MODEL}\"\n\n\
             [journal]\n{journal}\n[limits]\n{limits}",
            self.ca.display(),
            self.cert.display(),
            self.key.display(),
            self.state_dir().display(),
            self.stub.endpoint(),
            self.mock.base_url(),
        )
    }

    fn config(&self, limits: &str, journal: &str) -> AgentConfig {
        AgentConfig::from_toml_str(&self.toml(support::free_port(), limits, journal), None).unwrap()
    }

    async fn start(&self) -> AgentHandle {
        self.start_with(LIMITS, "").await
    }

    async fn start_with(&self, limits: &str, journal: &str) -> AgentHandle {
        Agent::start(self.config(limits, journal)).await.unwrap()
    }

    /// A `WorkerService` client presenting a leaf for `principal`.
    fn client(&self, agent: &AgentHandle, principal: &str) -> WorkerServiceClient<Channel> {
        let dir = self.root.path().join("pki");
        let pki = load_pki(&self.pki, &dir, &self.ca, principal);
        let channel = Endpoint::from_shared(format!("https://{}", agent.local_addr()))
            .unwrap()
            .timeout(Duration::from_secs(5))
            .tls_config(pki.client_tls(WORKER_PRINCIPAL))
            .unwrap()
            .connect_lazy();
        WorkerServiceClient::new(channel)
    }

    /// An accepted Register at `epoch` that the agent has processed: it heartbeats at that epoch only after it
    /// handled the reply (and logged its `register` event); the stub records the Register before the agent sees it.
    fn registered(&self, epoch: u64) -> bool {
        self.stub
            .registers()
            .iter()
            .any(|r| r.accepted && r.epoch == epoch)
            && self.stub.heartbeats().iter().any(|h| h.epoch == epoch)
    }

    fn aborted(&self, rid: &Rid) -> bool {
        self.mock
            .abort_requests()
            .iter()
            .any(|a| a.rid == rid.as_str() && !a.abort_all)
    }
}

fn load_pki(pki: &TestPki, dir: &Path, ca: &Path, principal: &str) -> Pki {
    let (cert, key) = pki.leaf(principal).write(dir);
    Pki::load(ca, &cert, &key, principal).unwrap()
}

fn identity(agent: &AgentHandle, digest: u8) -> OpIdentity {
    OpIdentity {
        method: wire::METHOD_SUBMIT,
        payload_sha256: [digest; 32],
        worker_epoch: agent.epoch(),
        model_generation: agent.generation(),
        coordinator_term: agent.term(),
    }
}

/// Accepts a SubmitWork op at the agent's current epoch, generation and term.
fn seed(agent: &AgentHandle, op: &str, seq: u64) -> (OpKey, Rid) {
    let rid = Rid::new(agent.epoch(), seq, 0xa11ce);
    let key = OpKey {
        scope_id: "dev".into(),
        operation_id: op.into(),
    };
    let identity = identity(agent, seq as u8);
    agent
        .shared()
        .journal
        .lock()
        .unwrap()
        .accept(key.clone(), identity, rid.as_str(), 1, 60_000, now_ms())
        .unwrap();
    (key, rid)
}

fn op(agent: &AgentHandle, key: &OpKey) -> OpView {
    agent.shared().journal.lock().unwrap().get(key).unwrap()
}

fn terminal_with(agent: &AgentHandle, key: &OpKey, reason: &str) -> bool {
    let view = op(agent, key);
    view.terminal && view.reason == reason
}

fn events_with(agent: &AgentHandle, needle: &str) -> usize {
    agent
        .events()
        .iter()
        .filter(|line| line.contains(needle))
        .count()
}

fn authority(agent: &AgentHandle, term: u64) -> pb::Authority {
    pb::Authority {
        target: Some(pb::WorkerIdentity {
            cluster_id: CLUSTER.into(),
            worker_id: WORKER.into(),
            worker_epoch: agent.epoch(),
        }),
        coordinator_id: COORD_ID.into(),
        coordinator_term: term,
        ..Default::default()
    }
}

fn describe(authority: Option<pb::Authority>) -> pb::InventoryRequest {
    pb::InventoryRequest { authority }
}

async fn open(fx: &Fixture, agent: &AgentHandle) {
    eventually("admission open", || {
        agent.admission_state() == AdmissionState::Open
    })
    .await;
    assert!(fx.registered(agent.epoch()));
}

/// Q4: the epoch is +1 per start and on disk before the agent registers; Register waits for the first confirmed
/// backend identity (the inventory needs its `instance_id`), which is generation 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn epoch_is_durable_before_register_and_register_waits_for_a_confirmed_backend() {
    let fx = Fixture::with(unconfirmed_info()).await;
    let agent = fx.start().await;
    assert_eq!(agent.epoch(), 1);
    assert_eq!(fx.durable().worker_epoch, 1);
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(
        fx.stub.registers().is_empty(),
        "registered before a confirmed backend"
    );
    assert_eq!(agent.generation(), 0);
    assert_eq!(agent.admission_state(), AdmissionState::Fenced);

    fx.mock.set_server_info_override(Some(confirmed_info(1)));
    eventually("Register at epoch 1", || fx.registered(1)).await;
    let first = fx.stub.registers()[0].clone();
    assert_eq!(first.epoch, 1);
    assert_eq!(first.boot_id.len(), 32);
    assert!(first.boot_id.bytes().all(|b| b.is_ascii_hexdigit()));
    assert_eq!(agent.generation(), 1);
    assert_eq!(fx.durable().model_generation, 1);
    assert_eq!(
        events_with(
            &agent,
            "cere-worker event=register term=5 accepted=true epoch=1"
        ),
        1
    );
    assert_eq!(
        events_with(
            &agent,
            "cere-worker event=generation generation=1 instance_id=sglang-"
        ),
        1
    );
    agent.shutdown().await;

    let agent = fx.start().await;
    assert_eq!(agent.epoch(), 2);
    assert_eq!(fx.durable().worker_epoch, 2);
    assert_eq!(agent.generation(), 1, "the generation is durable");
    eventually("Register at epoch 2", || fx.registered(2)).await;
    let second = fx.stub.registers().last().unwrap().clone();
    assert_ne!(second.boot_id, first.boot_id);
    assert_eq!(fx.stub.journaled_epoch(WORKER), 2);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        agent.generation(),
        1,
        "same backend instance, same generation"
    );
    assert_eq!(events_with(&agent, "event=generation"), 0);
    agent.shutdown().await;
}

/// The state dir is locked once per process: a second agent on it is refused and bumps nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_agent_on_the_same_state_dir_is_refused() {
    let fx = Fixture::new().await;
    let first = fx.start().await;
    let refused = Agent::start(fx.config(LIMITS, "")).await;
    let err = refused.err().expect("second agent started").to_string();
    assert!(err.contains("locked"), "{err}");
    assert_eq!(fx.durable().worker_epoch, 1);
    first.shutdown().await;
    let again = fx.start().await;
    assert_eq!(again.epoch(), 2);
    again.shutdown().await;
}

/// Q7/Q7a: a grant opens admission once reconciled; silence lets the lease expire (event, fenced); a grant after
/// expiry restores the lease but not admission until `reconciled_generation` again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lease_grant_and_expiry_drive_admission() {
    let fx = Fixture::new().await;
    fx.stub.set_reconcile(true);
    let agent = fx.start().await;
    open(&fx, &agent).await;
    let granted = agent
        .events()
        .into_iter()
        .find(|l| l.starts_with("cere-worker event=lease state=granted sequence="))
        .expect("no lease grant event");
    assert!(granted.contains(" until_ms="), "{granted}");
    eventually("an open heartbeat", || {
        fx.stub
            .heartbeats()
            .iter()
            .any(|h| h.admission_state == "open" && h.model_generation == 1)
    })
    .await;

    fx.stub.set_reconcile(false);
    fx.stub.set_silent(true);
    let silent_at = Instant::now();
    eventually("fenced after lease expiry", || {
        agent.admission_state() == AdmissionState::Fenced
    })
    .await;
    assert!(silent_at.elapsed() < Duration::from_millis(LEASE_MS + 400));
    eventually("lease expired event", || {
        events_with(&agent, "cere-worker event=lease state=expired sequence=") == 1
    })
    .await;

    let grants = events_with(&agent, "event=lease state=granted");
    fx.stub.set_silent(false);
    eventually("a grant after expiry", || {
        events_with(&agent, "event=lease state=granted") > grants
    })
    .await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        agent.admission_state(),
        AdmissionState::Fenced,
        "a grant after expiry must not reopen admission"
    );
    fx.stub.set_reconcile(true);
    open(&fx, &agent).await;
    agent.shutdown().await;
}

/// Q7: a grant counts only if `(term, lease_sequence)` is above the last accepted pair: a replayed or older grant
/// is ignored, `(t+1, 1)` after `(t, 500)` is accepted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replayed_or_older_grants_are_ignored_and_a_new_term_restarts_the_sequence() {
    let fx = Fixture::new().await;
    fx.stub.set_reconcile(true);
    let agent = fx.start().await;
    open(&fx, &agent).await;
    fx.stub.force_lease_sequence(500);
    eventually("grant 500", || {
        events_with(&agent, "event=lease state=granted sequence=500 ") == 1
    })
    .await;
    let answered = |seq: u64| {
        fx.stub
            .heartbeats()
            .iter()
            .filter(|h| h.accepted && h.lease_sequence == seq)
            .count()
    };
    let replayed = fx.stub.replay_last_grant();
    assert!(replayed >= 500);
    eventually("the replayed grant", || answered(replayed) == 2).await;
    eventually("the grant after the replay", || {
        events_with(
            &agent,
            &format!("event=lease state=granted sequence={} ", replayed + 1),
        ) == 1
    })
    .await;
    assert_eq!(
        events_with(
            &agent,
            &format!("event=lease state=granted sequence={replayed} ")
        ),
        1,
        "replayed grant applied"
    );
    let before_older = agent.events().len();
    let hb_mark = fx.stub.heartbeats().len();
    fx.stub.force_lease_sequence(3);
    eventually("older grants 3 and 4", || {
        fx.stub.heartbeats()[hb_mark..]
            .iter()
            .any(|h| h.accepted && h.lease_sequence == 4)
    })
    .await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(
        !agent.events()[before_older..]
            .iter()
            .any(|l| l.contains("state=granted")),
        "an older grant was applied"
    );
    // Older grants renew nothing: the lease runs out.
    eventually("fenced", || {
        agent.admission_state() == AdmissionState::Fenced
    })
    .await;

    let mark = agent.events().len();
    fx.stub.restart();
    eventually("grant (t+1, 1)", || {
        agent.events()[mark..]
            .iter()
            .any(|l| l.starts_with("cere-worker event=lease state=granted sequence=1 "))
    })
    .await;
    assert_eq!(agent.term(), TERM + 1);
    agent.shutdown().await;
}

/// Q6: a higher term fences: older-term ops terminal `coordinator_restart` and aborted, the term persisted, a new
/// Register; a reply or a call at a lower term is refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_higher_term_fences_older_term_ops_and_a_lower_term_is_refused() {
    let fx = Fixture::new().await;
    fx.stub.set_reconcile(true);
    let agent = fx.start().await;
    open(&fx, &agent).await;
    assert_eq!(agent.term(), TERM);
    assert_eq!(
        events_with(&agent, "event=fence"),
        0,
        "adopting the first term is no fence"
    );
    let (key, rid) = seed(&agent, "op-1", 1);

    fx.stub.restart();
    eventually("coordinator_restart", || {
        terminal_with(&agent, &key, wire::COORDINATOR_RESTART)
    })
    .await;
    eventually("abort of the fenced op", || fx.aborted(&rid)).await;
    assert_eq!(agent.term(), TERM + 1);
    assert_eq!(fx.durable().coordinator_term, TERM + 1);
    assert_eq!(
        events_with(
            &agent,
            "cere-worker event=fence reason=coordinator_restart term=6"
        ),
        1
    );
    assert_eq!(
        events_with(
            &agent,
            "cere-worker event=terminal op=dev/op-1 reason=coordinator_restart"
        ),
        1
    );
    eventually("Register at the new term", || {
        fx.stub
            .registers()
            .iter()
            .any(|r| r.accepted && r.term == TERM + 1)
    })
    .await;
    open(&fx, &agent).await;

    let fences = events_with(&agent, "event=fence");
    let mark = agent.events().len();
    let hb_mark = fx.stub.heartbeats().len();
    fx.stub.force_reply_term(TERM);
    eventually("the stale reply", || {
        fx.stub.heartbeats()[hb_mark..]
            .iter()
            .any(|h| h.term == TERM && h.accepted)
    })
    .await;
    let stale = fx.stub.heartbeats()[hb_mark..]
        .iter()
        .find(|h| h.term == TERM)
        .unwrap()
        .lease_sequence;
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert_eq!(agent.term(), TERM + 1);
    assert_eq!(events_with(&agent, "event=fence"), fences);
    let applied = format!("state=granted sequence={stale} ");
    assert!(
        !agent.events()[mark..].iter().any(|l| l.contains(&applied)),
        "a lower-term grant was applied"
    );

    let mut client = fx.client(&agent, COORD_PRINCIPAL);
    let status = client
        .describe_resources(describe(Some(authority(&agent, TERM))))
        .await
        .unwrap_err();
    assert_eq!(status.code(), Code::FailedPrecondition);
    assert_eq!(wire::reason(&status), wire::STALE_TERM);
    agent.shutdown().await;
}

/// Q4: `epoch_lost` jumps to epoch + 1 in process: old ops `worker_restart` and aborted, the epoch durable, the
/// heartbeat sequence restarted, a new Register.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn epoch_lost_jumps_one_epoch_and_restarts_old_ops() {
    let fx = Fixture::new().await;
    fx.stub.set_reconcile(true);
    let agent = fx.start().await;
    open(&fx, &agent).await;
    let (key, rid) = seed(&agent, "op-1", 1);

    fx.stub.mark_lost(WORKER);
    eventually("epoch 2", || agent.epoch() == 2).await;
    eventually("worker_restart", || {
        terminal_with(&agent, &key, wire::WORKER_RESTART)
    })
    .await;
    eventually("abort of the old op", || fx.aborted(&rid)).await;
    assert_eq!(fx.durable().worker_epoch, 2);
    eventually("Register at epoch 2", || fx.registered(2)).await;
    eventually("a heartbeat at epoch 2", || {
        fx.stub.heartbeats().iter().any(|h| h.epoch == 2)
    })
    .await;
    let first = fx
        .stub
        .heartbeats()
        .into_iter()
        .find(|h| h.epoch == 2)
        .unwrap();
    assert_eq!(first.sequence, 1, "the sequence restarts per epoch");
    open(&fx, &agent).await;
    agent.shutdown().await;
}

/// Q4: `stale_epoch` jumps to the journaled epoch + 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stale_epoch_jumps_to_the_journaled_epoch_plus_one() {
    let fx = Fixture::new().await;
    fx.stub.set_journaled_epoch(WORKER, 7);
    // An op of epoch 1 (this agent's first incarnation) already in the journal.
    let key = OpKey {
        scope_id: "dev".into(),
        operation_id: "op-1".into(),
    };
    let rid = Rid::new(1, 1, 0xa11ce);
    {
        let limits = OpLimits {
            max_operations: 4096,
            retry_horizon_ms: 600_000,
            rpc_timeout_ms: 300,
        };
        let (mut journal, _) = OpJournal::open(&fx.state_dir(), limits, now_ms()).unwrap();
        let identity = OpIdentity {
            method: wire::METHOD_SUBMIT,
            payload_sha256: [1; 32],
            worker_epoch: 1,
            model_generation: 1,
            coordinator_term: TERM,
        };
        journal
            .accept(key.clone(), identity, rid.as_str(), 1, 60_000, now_ms())
            .unwrap();
    }
    let agent = fx.start().await;
    assert_eq!(agent.epoch(), 1);
    eventually("epoch 8", || agent.epoch() == 8).await;
    eventually("worker_restart", || {
        terminal_with(&agent, &key, wire::WORKER_RESTART)
    })
    .await;
    eventually("abort of the old op", || fx.aborted(&rid)).await;
    assert_eq!(fx.durable().worker_epoch, 8);
    eventually("Register at epoch 8", || fx.registered(8)).await;
    let registers = fx.stub.registers();
    assert_eq!(registers[0].epoch, 1);
    assert_eq!(registers[0].reason, wire::STALE_EPOCH);
    agent.shutdown().await;
}

/// Q4: no accepted reply for `lost_after_ms` jumps the epoch (the agent closes admission on its own).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_accepted_reply_for_lost_after_jumps_the_epoch() {
    let fx = Fixture::new().await;
    fx.stub.set_reconcile(true);
    let agent = fx
        .start_with(
            "heartbeat_interval_ms = 100\nlost_after_ms = 1000\nrpc_timeout_ms = 300\n",
            "",
        )
        .await;
    open(&fx, &agent).await;
    let (key, rid) = seed(&agent, "op-1", 1);
    fx.stub.set_silent(true);
    let silent_at = Instant::now();
    eventually("epoch 2", || agent.epoch() == 2).await;
    assert!(
        silent_at.elapsed() >= Duration::from_millis(800),
        "jumped after {:?}",
        silent_at.elapsed()
    );
    assert_eq!(agent.admission_state(), AdmissionState::Fenced);
    eventually("worker_restart", || {
        terminal_with(&agent, &key, wire::WORKER_RESTART)
    })
    .await;
    eventually("abort of the old op", || fx.aborted(&rid)).await;
    assert_eq!(fx.durable().worker_epoch, 2);
    fx.stub.set_silent(false);
    eventually("Register at epoch 2", || fx.registered(2)).await;
    assert_eq!(agent.epoch(), 2, "one jump per silence");
    agent.shutdown().await;
}

/// Q5: a backend relaunch (new `instance_id`) bumps the durable generation, fails older-generation ops
/// `model_generation_superseded` (aborted) and changes the inventory hash; an outage followed by the same
/// `instance_id` keeps the generation and the ops.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_backend_relaunch_bumps_the_generation_and_an_outage_does_not() {
    let fx = Fixture::new().await;
    fx.stub.set_reconcile(true);
    let agent = fx.start().await;
    open(&fx, &agent).await;
    assert_eq!(agent.generation(), 1);
    let first_id = fx.durable().instance_id.unwrap();
    let (old, old_rid) = seed(&agent, "old", 1);
    let sha1 = fx
        .stub
        .heartbeats()
        .last()
        .unwrap()
        .inventory_sha256
        .clone();

    fx.mock.set_server_info_override(Some(confirmed_info(2)));
    eventually("generation 2", || agent.generation() == 2).await;
    eventually("model_generation_superseded", || {
        terminal_with(&agent, &old, wire::MODEL_GENERATION_SUPERSEDED)
    })
    .await;
    eventually("abort of the superseded op", || fx.aborted(&old_rid)).await;
    let durable = fx.durable();
    assert_eq!(durable.model_generation, 2);
    assert_ne!(durable.instance_id.as_deref(), Some(first_id.as_str()));
    assert_eq!(
        events_with(
            &agent,
            "cere-worker event=generation generation=2 instance_id=sglang-"
        ),
        1
    );
    eventually(
        "a heartbeat at generation 2 with a new inventory hash",
        || {
            fx.stub
                .heartbeats()
                .iter()
                .any(|h| h.model_generation == 2 && h.inventory_sha256 != sha1)
        },
    )
    .await;
    open(&fx, &agent).await;

    let (new, _) = seed(&agent, "new", 2);
    let mut renamed = confirmed_info(2);
    renamed.served_model_name = "other-model".into();
    fx.mock.set_server_info_override(Some(renamed));
    tokio::time::sleep(Duration::from_millis(400)).await;
    fx.mock.set_server_info_override(Some(confirmed_info(2)));
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(agent.generation(), 2);
    assert_eq!(fx.durable().model_generation, 2);
    assert!(!op(&agent, &new).terminal, "an outage must not end the ops");
    assert_eq!(events_with(&agent, "event=generation"), 2);
    agent.shutdown().await;
}

/// `desired_state = "DRAINING"` closes admission (reported `fenced`) until the intent is cleared.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn draining_closes_admission() {
    let fx = Fixture::new().await;
    fx.stub.set_reconcile(true);
    let agent = fx.start().await;
    open(&fx, &agent).await;
    fx.stub.set_draining(true);
    eventually("fenced while draining", || {
        agent.admission_state() == AdmissionState::Fenced
    })
    .await;
    let mark = fx.stub.heartbeats().len();
    eventually("a fenced heartbeat", || {
        fx.stub.heartbeats()[mark..]
            .iter()
            .any(|h| h.admission_state == "fenced")
    })
    .await;
    fx.stub.set_draining(false);
    open(&fx, &agent).await;
    agent.shutdown().await;
}

/// Q2/Q8/Q10a: `Authz` refuses a non-coordinator principal before any state (a higher term in its call fences
/// nothing); the coordinator's `DescribeResources` returns the inventory whose SHA-256 the agent sends in Register
/// and every heartbeat; the authority is checked (Q6).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn describe_resources_authorizes_first_and_matches_the_heartbeat_hash() {
    let fx = Fixture::new().await;
    let agent = fx.start().await;
    eventually("registered", || fx.registered(1)).await;
    eventually("an accepted heartbeat", || {
        fx.stub.heartbeats().iter().any(|h| h.accepted)
    })
    .await;

    let mut intruder = fx.client(&agent, "cere-intruder");
    let status = intruder
        .describe_resources(describe(Some(authority(&agent, TERM + 7))))
        .await
        .unwrap_err();
    assert_eq!(status.code(), Code::PermissionDenied);
    assert_eq!(wire::reason(&status), wire::PERMISSION_DENIED_REASON);
    assert_eq!(
        agent.term(),
        TERM,
        "a refused caller reached the term check"
    );

    let mut client = fx.client(&agent, COORD_PRINCIPAL);
    let reply = client
        .describe_resources(describe(Some(authority(&agent, TERM))))
        .await
        .unwrap()
        .into_inner();
    let bytes = reply.bounded_resource_inventory_json;
    let sha = Inventory::sha256(&bytes).to_vec();
    assert!(reply.inventory_generation >= 1);
    let last = fx
        .stub
        .heartbeats()
        .into_iter()
        .rev()
        .find(|h| h.accepted)
        .unwrap();
    assert_eq!(last.inventory_sha256, sha);
    assert_eq!(fx.stub.registers()[0].inventory_sha256, sha);
    let inventory = Inventory::from_json_bytes(&bytes).unwrap();
    assert_eq!(inventory.worker_id, WORKER);
    assert_eq!(inventory.served_model_name, MODEL);
    assert_eq!(inventory.tp_size, Some(1));
    assert_eq!(inventory.kv_cache_dtype.as_deref(), Some("fp8_e4m3"));
    assert_eq!(inventory.max_running_requests, Some(4));
    assert_eq!(inventory.model_generation, 1);
    assert_eq!(inventory.instance_id, fx.durable().instance_id);
    assert!(inventory.instance_id.is_some());
    assert_eq!(inventory.models.len(), 1);
    assert_eq!(inventory.models[0].instance_id, inventory.instance_id);
    assert_eq!(inventory.models[0].state, ModelState::Attached);
    assert_eq!(inventory.models[0].lock_sha256, None);

    let refused = |mutate: fn(&mut pb::Authority)| {
        let mut a = authority(&agent, TERM);
        mutate(&mut a);
        describe(Some(a))
    };
    for (request, code, reason) in [
        (
            describe(None),
            Code::InvalidArgument,
            wire::MALFORMED_REQUEST,
        ),
        (
            refused(|a| a.target.as_mut().unwrap().cluster_id = "other".into()),
            Code::FailedPrecondition,
            wire::WRONG_CLUSTER,
        ),
        (
            refused(|a| a.target.as_mut().unwrap().worker_id = "worker-b".into()),
            Code::FailedPrecondition,
            wire::WRONG_WORKER,
        ),
        (
            refused(|a| a.target.as_mut().unwrap().worker_epoch += 1),
            Code::FailedPrecondition,
            wire::STALE_EPOCH,
        ),
        (
            refused(|a| a.coordinator_id = "coord-2".into()),
            Code::FailedPrecondition,
            wire::WRONG_COORDINATOR,
        ),
    ] {
        let status = client.describe_resources(request).await.unwrap_err();
        assert_eq!((status.code(), wire::reason(&status)), (code, reason));
    }
    agent.shutdown().await;
}

/// A journal write failure poisons the agent: admission `poisoned`, one `poisoned` event, reported in heartbeats.
/// The failure is real: the compaction temp path is a directory (fails also for root), and `max_operations = 1`
/// makes the second record trigger a compaction.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_journal_write_failure_poisons_the_agent() {
    let fx = Fixture::new().await;
    fx.stub.set_reconcile(true);
    let agent = fx.start_with(LIMITS, "max_operations = 1\n").await;
    open(&fx, &agent).await;
    std::fs::create_dir(fx.state_dir().join("operations.jsonl.compact.tmp")).unwrap();
    let (key, _) = seed(&agent, "op-1", 1);
    agent
        .shared()
        .journal
        .lock()
        .unwrap()
        .mark_terminal(&key, "", now_ms())
        .unwrap();
    eventually("poisoned", || {
        agent.admission_state() == AdmissionState::Poisoned
    })
    .await;
    assert_eq!(
        events_with(&agent, "cere-worker event=poisoned reason=journal_write"),
        1
    );
    eventually("a poisoned heartbeat", || {
        fx.stub
            .heartbeats()
            .iter()
            .any(|h| h.admission_state == "poisoned")
    })
    .await;
    agent.shutdown().await;
}

fn worker_binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_cere-worker"))
}

/// The binary: `--check-config` (0 / 2, no PEM opened), `--version`, the stdout startup line, SIGTERM -> exit 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_binary_checks_config_prints_the_startup_line_and_stops_on_sigterm() {
    let fx = Fixture::new().await;
    let port = support::free_port();
    let good = fx.root.path().join("agent.toml");
    std::fs::write(&good, fx.toml(port, LIMITS, "")).unwrap();
    let out = worker_binary()
        .arg("--check-config")
        .arg(&good)
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Shape only: PEM paths that do not exist still pass.
    let missing = fx.root.path().join("missing-pems.toml");
    std::fs::write(
        &missing,
        fx.toml(port, LIMITS, "")
            .replace(&fx.ca.display().to_string(), "/nonexistent/ca.pem"),
    )
    .unwrap();
    let out = worker_binary()
        .arg("--check-config")
        .arg(&missing)
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bad = fx.root.path().join("bad.toml");
    std::fs::write(
        &bad,
        fx.toml(port, LIMITS, "")
            .replace(&format!("cluster_id = \"{CLUSTER}\"\n"), ""),
    )
    .unwrap();
    let out = worker_binary()
        .arg("--check-config")
        .arg(&bad)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(!out.stderr.is_empty());

    let out = worker_binary().arg("--version").output().unwrap();
    assert_eq!(out.status.code(), Some(0));
    let version = std::fs::read_to_string(support::repo_root().join("PACKAGE_VERSION")).unwrap();
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        format!("cere-worker {}\n", version.trim())
    );

    let out = worker_binary().output().unwrap();
    assert_eq!(out.status.code(), Some(2), "no arguments is a usage error");

    // First start: the generation is whatever the first probe reached by the time of the line (0 or 1); the
    // second start reads the durable generation 1 and its backend is the same instance.
    let (child, stderr, line) = spawn_worker(&good).await;
    let prefix = format!("cere-worker listening on 127.0.0.1:{port} worker={WORKER}@1 generation=");
    assert!(
        line == format!("{prefix}0\n") || line == format!("{prefix}1\n"),
        "{line:?}"
    );
    eventually("the binary registers", || fx.registered(1)).await;
    let (code, stderr) = stop_worker(child, stderr).await;
    assert_eq!(code, Some(0), "{stderr}");
    assert!(
        stderr.contains("cere-worker event=register term=5 accepted=true epoch=1"),
        "{stderr}"
    );

    let (child, stderr, line) = spawn_worker(&good).await;
    assert_eq!(
        line,
        format!("cere-worker listening on 127.0.0.1:{port} worker={WORKER}@2 generation=1\n")
    );
    eventually("the binary registers again", || fx.registered(2)).await;
    let (code, stderr) = stop_worker(child, stderr).await;
    assert_eq!(code, Some(0), "{stderr}");
}

/// Spawns `cere-worker --config <path>` and reads its startup line.
async fn spawn_worker(
    config: &Path,
) -> (std::process::Child, std::thread::JoinHandle<String>, String) {
    use std::io::{BufRead, BufReader, Read};

    let mut child = worker_binary()
        .arg("--config")
        .arg(config)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stderr = {
        let mut pipe = child.stderr.take().unwrap();
        std::thread::spawn(move || {
            let mut text = String::new();
            let _ = pipe.read_to_string(&mut text);
            text
        })
    };
    let stdout = child.stdout.take().unwrap();
    let line = tokio::time::timeout(
        Duration::from_secs(20),
        tokio::task::spawn_blocking(move || {
            let mut line = String::new();
            BufReader::new(stdout).read_line(&mut line).unwrap();
            line
        }),
    )
    .await
    .expect("cere-worker printed no startup line in time")
    .unwrap();
    (child, stderr, line)
}

/// SIGTERM, then a bounded wait for the exit; returns the exit code and the whole stderr.
async fn stop_worker(
    mut child: std::process::Child,
    stderr: std::thread::JoinHandle<String>,
) -> (Option<i32>, String) {
    let ok = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .unwrap()
        .success();
    assert!(ok);
    let until = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < until,
            "cere-worker did not stop on SIGTERM"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    (status.code(), stderr.join().unwrap())
}
