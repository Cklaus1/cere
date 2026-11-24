//! M3.6 P4 (m36 design note Q3-Q7, design-m36-agent.md §3): the managed model lifecycle over the agent's mTLS
//! listener. The backend is a `/bin/sh` script that `exec`s the built `cere-mock-sglang` with
//! `--startup-delay-ms`, so `loading` is observable; every test runs the agent in-process against the stub
//! coordinator.
//!
//! The mock binary (`<target>/debug/cere-mock-sglang`; `make ci` builds it, else `cargo build -p
//! cere-mock-sglang`) is copied into the test's 0755 temp dir: as root the child drops to uid/gid 65534 (the
//! `check_privileges` rule), which cannot reach a binary under a 0750 home.

mod support;

use std::collections::BTreeMap;
use std::fs::Permissions;
use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use cere_mock_sglang::{MockConfig, MockHandle, MockSglang, ServerInfo};
use cere_proto::worker::v2 as pb;
use cere_runtime::agent::config::AgentConfig;
use cere_runtime::agent::journal::{OpKey, OpView, OPERATIONS_FILE};
use cere_runtime::agent::supervisor::{ChildRecord, CHILD_FILE};
use cere_runtime::agent::{Agent, AgentHandle};
use cere_runtime::fed::tls::Pki;
use cere_runtime::fed::wire::{self, AdmissionState, Inventory};
use cere_runtime::sglang::Rid;
use pb::worker_inference_service_client::WorkerInferenceServiceClient;
use pb::worker_service_client::WorkerServiceClient;
use serde_json::{json, Value};
use support::mock_bin;
use support::pki::TestPki;
use support::stub_coordinator::{StubConfig, StubCoordinator};
use support::StateDir;
use tonic::transport::{Channel, Endpoint};
use tonic::{Code, Status};

const CLUSTER: &str = "lab-test";
const WORKER: &str = "worker-a";
const WORKER_PRINCIPAL: &str = "cere-worker-a";
const COORD_PRINCIPAL: &str = "cere-coordinator";
const COORD_ID: &str = "coord-1";
const MODEL: &str = "Qwen3.8-Flash-Next";
const TERM: u64 = 5;
/// No test here ages the link (a Lost worker is `mark_lost`): the agent's self-lost bound, like the stub's
/// [`LEASE_MS`] and [`LOST_AFTER`], is one no host stall outlasts.
const LIMITS: &str = "heartbeat_interval_ms = 100\nlost_after_ms = 180000\nrpc_timeout_ms = 300\n";
/// The stub coordinator's admission lease. A short one (500 ms) expired when a loaded host starved the agent's
/// heartbeats for about a second, and the next SubmitWork was refused `UNAVAILABLE lease_expired`
/// (`lifecycle_ops_never_count_toward_queue_full`, CI of bc68dcab).
const LEASE_MS: u64 = 60_000;
/// The stub coordinator's Lost bound (no accepted heartbeat for this long).
const LOST_AFTER: Duration = Duration::from_secs(180);
/// The coordinator's lifecycle scope (COORD-DRIVE call shape).
const CONTROL_SCOPE: &str = "_cere_control";
const SCOPE: &str = "dev";
/// The approved `[backend] lock_sha256`.
const LOCK: [u8; 32] = [0xab; 32];
const UNLIMITED: u64 = 60_000;

async fn eventually(what: &str, mut done: impl FnMut() -> bool) {
    let until = Instant::now() + Duration::from_secs(20);
    while !done() {
        assert!(Instant::now() < until, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn confirmed_info(seed: u64) -> ServerInfo {
    let mut info = support::launched(seed, 1.5);
    info.extra.insert("tp_size".into(), json!(1));
    info
}

fn load_pki(pki: &TestPki, dir: &Path, ca: &Path, principal: &str) -> Pki {
    let (cert, key) = pki.leaf(principal).write(dir);
    Pki::load(ca, &cert, &key, principal).unwrap()
}

fn running_as_root() -> bool {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { libc::geteuid() == 0 }
}

fn chmod(path: &Path, mode: u32) {
    std::fs::set_permissions(path, Permissions::from_mode(mode)).unwrap();
}

/// The managed backend: a free loopback port and a 0755 dir holding the mock copy and the scripts.
struct Fixture {
    root: StateDir,
    pki: TestPki,
    ca: PathBuf,
    cert: PathBuf,
    key: PathBuf,
    stub: StubCoordinator,
    port: u16,
    bin: PathBuf,
    /// The in-process backend of an attached-mode agent.
    attached: Option<MockHandle>,
}

impl Fixture {
    async fn new() -> Fixture {
        let root = StateDir::new();
        std::fs::create_dir_all(root.path()).unwrap();
        chmod(root.path(), 0o755);
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
        let bin = root.path().join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        chmod(&bin, 0o755);
        Fixture {
            root,
            pki,
            ca,
            cert,
            key,
            stub,
            port: support::free_port(),
            bin,
            attached: None,
        }
    }

    /// An attached-mode fixture: the backend runs in-process, no `[backend.launch]`.
    async fn attached() -> Fixture {
        let mut f = Fixture::new().await;
        let mock = MockSglang::start(MockConfig {
            server_info: confirmed_info(1),
            ..MockConfig::default()
        })
        .await
        .unwrap();
        f.attached = Some(mock);
        f
    }

    fn state_dir(&self) -> PathBuf {
        self.root.path().join("state")
    }

    fn child_file(&self) -> PathBuf {
        self.state_dir().join(CHILD_FILE)
    }

    /// Writes `bin/backend.sh` (0755) with `body` after the shebang, from a child process (O30: a sibling test's
    /// fork never holds it open for writing, which would make the agent's exec fail ETXTBSY).
    fn script(&self, body: &str) -> PathBuf {
        let path = self.bin.join("backend.sh");
        let text = format!("#!/bin/sh\n{body}\n");
        support::exec_file::write_file(&path, text.as_bytes(), 0o755);
        path
    }

    /// The fixture's copy of the built mock; every backend a test launches executes it ([`Fixture::children`]).
    fn mock_path(&self) -> PathBuf {
        self.bin.join("cere-mock-sglang")
    }

    /// [`Fixture::mock_path`], placed first ([`mock_bin::place`]).
    fn placed_mock(&self) -> PathBuf {
        let mock = self.mock_path();
        if !mock.exists() {
            mock_bin::place(&mock);
        }
        mock
    }

    /// The placed mock's command line on the fixture port: `delay_ms` before it listens, `latency_ms` before each
    /// output token, serving `info`.
    fn mock_command(&self, info: &ServerInfo, delay_ms: u64, latency_ms: u64) -> String {
        let mock = self.placed_mock();
        let info_file = self.bin.join("server_info.json");
        std::fs::write(&info_file, serde_json::to_vec(info).unwrap()).unwrap();
        chmod(&info_file, 0o644);
        format!(
            "{} --port {} --server-info {} --startup-delay-ms {delay_ms} --token-latency-ms {latency_ms}",
            mock.display(),
            self.port,
            info_file.display()
        )
    }

    /// A script that `exec`s the mock copy on the fixture port after `delay_ms`, serving `info`.
    fn mock_script(&self, info: &ServerInfo, delay_ms: u64) -> PathBuf {
        self.script(&format!("exec {}", self.mock_command(info, delay_ms, 0)))
    }

    /// A ready mock whose every output token takes `latency_ms`: a SubmitWork op stays live for 4 tokens.
    fn slow_mock_script(&self, latency_ms: u64) -> PathBuf {
        self.script(&format!(
            "exec {}",
            self.mock_command(&confirmed_info(1), 0, latency_ms)
        ))
    }

    /// A group leader (`/bin/sh`) that ignores SIGTERM and outlives the mock by 30 s: a stop waits the whole
    /// `stop_timeout_ms` before its SIGKILL, so `unloading` is observable.
    fn stubborn_script(&self, delay_ms: u64) -> PathBuf {
        self.script(&format!(
            "trap '' TERM\n{}\nsleep 30",
            self.mock_command(&confirmed_info(1), delay_ms, 0)
        ))
    }

    fn base_config(&self, listen: u16, base_url: &str, backend_extra: &str) -> String {
        format!(
            "cluster_id = \"{CLUSTER}\"\nworker_id = \"{WORKER}\"\nlisten = \"127.0.0.1:{listen}\"\n\
             principal = \"{WORKER_PRINCIPAL}\"\nca_file = \"{}\"\ncert_file = \"{}\"\nkey_file = \"{}\"\n\
             state_dir = \"{}\"\n\n[coordinator]\nendpoint = \"{}\"\nprincipal = \"{COORD_PRINCIPAL}\"\n\
             coordinator_id = \"{COORD_ID}\"\n\n[limits]\n{LIMITS}\n[backend]\nbase_url = \"{base_url}\"\n\
             served_model_name = \"{MODEL}\"\n{backend_extra}",
            self.ca.display(),
            self.cert.display(),
            self.key.display(),
            self.state_dir().display(),
            self.stub.endpoint(),
        )
    }

    /// The TOML of a managed agent on `listen` launching `script`.
    fn managed_text(&self, script: &Path, startup_timeout_ms: u64, listen: u16) -> String {
        self.managed_text_with(script, startup_timeout_ms, 2000, listen)
    }

    /// [`Fixture::managed_text`] with `stop_timeout_ms`.
    fn managed_text_with(
        &self,
        script: &Path,
        startup_timeout_ms: u64,
        stop_timeout_ms: u64,
        listen: u16,
    ) -> String {
        let ids = if running_as_root() {
            "uid = 65534\ngid = 65534\n"
        } else {
            ""
        };
        let extra = format!(
            "lock_sha256 = \"{}\"\n\n[backend.launch]\nargv = [\"/bin/sh\", \"{}\"]\n\
             startup_timeout_ms = {startup_timeout_ms}\nstop_timeout_ms = {stop_timeout_ms}\n{ids}",
            hex(&LOCK),
            script.display()
        );
        self.base_config(listen, &format!("http://127.0.0.1:{}", self.port), &extra)
    }

    /// A managed config launching `script`.
    fn managed(&self, script: &Path, startup_timeout_ms: u64) -> AgentConfig {
        let text = self.managed_text(script, startup_timeout_ms, support::free_port());
        AgentConfig::from_toml_str(&text, None).unwrap()
    }

    fn attached_config(&self) -> AgentConfig {
        let mock = self.attached.as_ref().unwrap();
        let text = self.base_config(support::free_port(), &mock.base_url(), "");
        AgentConfig::from_toml_str(&text, None).unwrap()
    }

    /// An mTLS channel to the agent on `addr` as `principal`.
    fn channel_as(&self, addr: SocketAddr, principal: &str) -> Channel {
        let dir = self.root.path().join("pki");
        let pki = load_pki(&self.pki, &dir, &self.ca, principal);
        Endpoint::from_shared(format!("https://{addr}"))
            .unwrap()
            .timeout(Duration::from_secs(30))
            .tls_config(pki.client_tls(WORKER_PRINCIPAL))
            .unwrap()
            .connect_lazy()
    }

    fn channel(&self, agent: &AgentHandle) -> Channel {
        self.channel_as(agent.local_addr(), COORD_PRINCIPAL)
    }

    /// The recorded backend group.
    fn record(&self) -> ChildRecord {
        serde_json::from_slice(&std::fs::read(self.child_file()).unwrap()).unwrap()
    }

    /// The journal file's bytes.
    fn journal_bytes(&self) -> Vec<u8> {
        std::fs::read(self.state_dir().join(OPERATIONS_FILE)).unwrap_or_default()
    }

    fn control(&self, agent: &AgentHandle) -> WorkerServiceClient<Channel> {
        WorkerServiceClient::new(self.channel(agent))
    }

    fn inference(&self, agent: &AgentHandle) -> WorkerInferenceServiceClient<Channel> {
        WorkerInferenceServiceClient::new(self.channel(agent))
    }

    /// Live processes executing the fixture's mock copy (the launched backend), by `/proc/<pid>/exe`. Not by
    /// `--port <fixture port>` in the argv: the port is reserved only in this network namespace, and a
    /// `make ci-docker` container on the same host runs these tests in its own over the same port window, so its
    /// mock can name the same number (O48).
    fn children(&self) -> usize {
        mock_bin::running(&self.mock_path()).len()
    }

    /// Starts a managed agent and waits for its Register (term adopted).
    async fn start(&self, config: AgentConfig) -> AgentHandle {
        let agent = Agent::start(config).await.unwrap();
        eventually("registered", || agent.term() == TERM).await;
        agent
    }
}

fn view(agent: &AgentHandle, scope: &str, op: &str) -> Option<OpView> {
    agent.shared().journal.lock().unwrap().get(&OpKey {
        scope_id: scope.into(),
        operation_id: op.into(),
    })
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

fn read_authority(agent: &AgentHandle) -> pb::Authority {
    authority_at(agent.epoch(), agent.term())
}

/// A read authority for (`epoch`, `term`) (an agent run as a subprocess).
fn authority_at(epoch: u64, term: u64) -> pb::Authority {
    pb::Authority {
        target: Some(pb::WorkerIdentity {
            cluster_id: CLUSTER.into(),
            worker_id: WORKER.into(),
            worker_epoch: epoch,
        }),
        coordinator_id: COORD_ID.into(),
        coordinator_term: term,
        ..Default::default()
    }
}

/// A lifecycle call as COORD-DRIVE sends it: scope `_cere_control`, the Q6 digest, the work cap as budget and the
/// agent's current generation.
fn lifecycle(
    agent: &AgentHandle,
    method: &str,
    op: &str,
    lock: &[u8],
    instance_id: &str,
    device_ids: &[&str],
) -> pb::ModelLifecycleRequest {
    let base = pb::Authority {
        model_generation: Some(agent.generation()),
        ..read_authority(agent)
    };
    lifecycle_on(base, method, op, lock, instance_id, device_ids)
}

/// [`lifecycle`] on `base` (target, term and generation).
fn lifecycle_on(
    base: pb::Authority,
    method: &str,
    op: &str,
    lock: &[u8],
    instance_id: &str,
    device_ids: &[&str],
) -> pb::ModelLifecycleRequest {
    let device_ids: Vec<String> = device_ids.iter().map(|d| (*d).to_owned()).collect();
    let digest = wire::lifecycle_sha256(method, CONTROL_SCOPE, lock, instance_id, &device_ids);
    pb::ModelLifecycleRequest {
        authority: Some(pb::Authority {
            scope_id: CONTROL_SCOPE.into(),
            operation_id: op.into(),
            payload_sha256: digest.to_vec(),
            remaining_budget_ms: wire::WORK_CAP_MS,
            ..base
        }),
        deployment_lock_sha256: lock.to_vec(),
        instance_id: instance_id.into(),
        device_ids,
    }
}

fn load(agent: &AgentHandle, op: &str) -> pb::ModelLifecycleRequest {
    lifecycle(agent, wire::METHOD_LOAD, op, &LOCK, "", &[])
}

fn drain(agent: &AgentHandle, op: &str) -> pb::ModelLifecycleRequest {
    lifecycle(agent, wire::METHOD_DRAIN, op, &LOCK, "", &[])
}

fn unload(agent: &AgentHandle, op: &str) -> pb::ModelLifecycleRequest {
    lifecycle(agent, wire::METHOD_UNLOAD, op, &LOCK, "", &[])
}

/// A well-formed unary SubmitWork for (`SCOPE`, `op`).
fn submit(agent: &AgentHandle, op: &str, seq: u64) -> pb::SubmitWorkRequest {
    let base = pb::Authority {
        model_generation: Some(agent.generation()),
        ..read_authority(agent)
    };
    submit_on(base, op, seq)
}

/// [`submit`] on `base` (target, term and generation; an agent run as a subprocess).
fn submit_on(base: pb::Authority, op: &str, seq: u64) -> pb::SubmitWorkRequest {
    let rid = Rid::new(1, seq, 0xbeef).as_str().to_owned();
    let unit = br#"{"kind":"chat","stream":false}"#;
    let input = serde_json::to_vec(&json!({
        "rid": rid,
        "messages": [{"role": "user", "content": "lifecycle prompt"}],
        "max_completion_tokens": 4,
        "stream": false,
    }))
    .unwrap();
    let digest = wire::payload_sha256(wire::METHOD_SUBMIT, SCOPE, &rid, 1, unit, &input);
    pb::SubmitWorkRequest {
        authority: Some(pb::Authority {
            scope_id: SCOPE.into(),
            operation_id: op.into(),
            payload_sha256: digest.to_vec(),
            remaining_budget_ms: UNLIMITED,
            ..base
        }),
        request_id: rid,
        attempt: 1,
        work_unit_json: unit.to_vec(),
        bounded_inference_input_json: input,
    }
}

fn refused(
    result: Result<tonic::Response<pb::OperationReceipt>, Status>,
    code: Code,
    reason: &str,
) {
    let status = result.expect_err("refused");
    assert_eq!(status.code(), code, "{status:?}");
    assert_eq!(status.message(), reason, "{status:?}");
}

async fn inventory(c: &mut WorkerServiceClient<Channel>, agent: &AgentHandle) -> Value {
    inventory_for(c, read_authority(agent)).await
}

/// `DescribeResources`, the bytes validated by [`Inventory::from_json_bytes`] (Q7: the identity fields are set
/// exactly as the model state allows).
async fn inventory_for(c: &mut WorkerServiceClient<Channel>, authority: pb::Authority) -> Value {
    let reply = c
        .describe_resources(pb::InventoryRequest {
            authority: Some(authority),
        })
        .await
        .unwrap()
        .into_inner();
    let bytes = &reply.bounded_resource_inventory_json;
    if let Err(e) = Inventory::from_json_bytes(bytes) {
        panic!(
            "invalid inventory {e:?}: {}",
            String::from_utf8_lossy(bytes)
        );
    }
    serde_json::from_slice(bytes).unwrap()
}

fn model_state(inventory: &Value) -> &str {
    inventory["models"][0]["state"].as_str().unwrap()
}

/// Polls `DescribeResources` until the model reports `state`.
async fn wait_state(
    c: &mut WorkerServiceClient<Channel>,
    agent: &AgentHandle,
    state: &str,
) -> Value {
    wait_state_for(c, read_authority(agent), state).await
}

/// [`wait_state`] for an agent read at `authority` (a `cere-worker` process).
async fn wait_state_for(
    c: &mut WorkerServiceClient<Channel>,
    authority: pb::Authority,
    state: &str,
) -> Value {
    let until = Instant::now() + Duration::from_secs(20);
    loop {
        let inv = inventory_for(c, authority.clone()).await;
        if model_state(&inv) == state {
            return inv;
        }
        assert!(
            Instant::now() < until,
            "timed out waiting for {state}: {inv}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// The op ends with `reason` and is cleanup-acknowledged at its terminal (lifecycle ops, Q4).
async fn ends(agent: &AgentHandle, op: &str, reason: &str) {
    eventually(&format!("{op} terminal"), || {
        view(agent, CONTROL_SCOPE, op).is_some_and(|v| v.terminal)
    })
    .await;
    let v = view(agent, CONTROL_SCOPE, op).unwrap();
    assert_eq!(v.reason, reason, "{op}");
    assert!(v.cleanup_acknowledged, "{op} cleanup-acknowledged");
    assert_eq!(v.request_id, op, "request_id = operation_id");
    assert_eq!(v.attempt, 1);
}

/// Model state events (`event=model state=<s>`) in order.
fn model_events(agent: &AgentHandle) -> Vec<String> {
    agent
        .events()
        .iter()
        .filter_map(|line| line.split("event=model state=").nth(1))
        .map(|rest| rest.split_whitespace().next().unwrap_or("").to_owned())
        .collect()
}

/// Inventory facts that hold in a confirmed state (`ready`, `draining`, `unloading`).
fn assert_confirmed(inv: &Value, generation: u64) {
    assert_eq!(inv["served_model_name"], MODEL);
    assert_eq!(inv["tp_size"], 1);
    assert_eq!(inv["max_running_requests"], 4);
    assert!(inv["kv_cache_dtype"].is_string(), "{inv}");
    let instance = inv["instance_id"].as_str().expect("instance_id");
    assert_eq!(inv["models"][0]["instance_id"], instance);
    assert_eq!(inv["models"][0]["served_model_name"], MODEL);
    assert_eq!(inv["models"][0]["lock_sha256"], hex(&LOCK));
    assert_eq!(inv["models"][0]["model_generation"], generation);
    assert_eq!(inv["model_generation"], generation);
}

/// Inventory facts of an unconfirmed state (`unloaded`, `loading`, `failed`).
fn assert_unconfirmed(inv: &Value) {
    assert_eq!(inv["served_model_name"], MODEL);
    for key in [
        "tp_size",
        "kv_cache_dtype",
        "max_running_requests",
        "instance_id",
    ] {
        assert!(inv[key].is_null(), "{key} absent: {inv}");
    }
    assert!(inv["models"][0]["instance_id"].is_null(), "{inv}");
    assert_eq!(inv["models"][0]["served_model_name"], MODEL);
    assert_eq!(inv["models"][0]["lock_sha256"], hex(&LOCK));
}

/// A managed agent whose model is `ready` (mock without startup delay).
async fn ready_agent(f: &Fixture) -> (AgentHandle, WorkerServiceClient<Channel>) {
    ready_agent_with(f, &f.mock_script(&confirmed_info(1), 0)).await
}

/// A managed agent launching `script` whose model is `ready`.
async fn ready_agent_with(
    f: &Fixture,
    script: &Path,
) -> (AgentHandle, WorkerServiceClient<Channel>) {
    let agent = f.start(f.managed(script, UNLIMITED)).await;
    let mut c = f.control(&agent);
    c.load_model(load(&agent, "load-ready")).await.unwrap();
    wait_state(&mut c, &agent, "ready").await;
    ends(&agent, "load-ready", "").await;
    // The reconciled generation 1 arrives with the next heartbeat reply.
    eventually("admission open", || {
        agent.admission_state() == AdmissionState::Open
    })
    .await;
    (agent, c)
}

/// The lifecycle op is journaled and not terminal.
fn pending(agent: &AgentHandle, op: &str) -> bool {
    view(agent, CONTROL_SCOPE, op).is_some_and(|v| !v.terminal)
}

/// The lifecycle op is terminal `reason` and cleanup-acknowledged right now (no wait).
fn ended_at_once(agent: &AgentHandle, op: &str, reason: &str) {
    let v = view(agent, CONTROL_SCOPE, op).expect("journaled");
    assert!(v.terminal, "{op} terminal at once");
    assert_eq!(v.reason, reason, "{op}");
    assert!(v.cleanup_acknowledged, "{op} cleanup-acknowledged");
}

/// A fresh SubmitWork op (`SCOPE`, `op`) that is accepted and still live.
async fn live_op(
    work: &mut WorkerInferenceServiceClient<Channel>,
    agent: &AgentHandle,
    op: &str,
    seq: u64,
) {
    let receipt = work
        .submit_work(submit(agent, op, seq))
        .await
        .unwrap()
        .into_inner();
    assert!(receipt.accepted);
    assert!(!view(agent, SCOPE, op).unwrap().terminal, "{op} live");
}

/// The SubmitWork op ends with `reason`.
async fn work_ends(agent: &AgentHandle, op: &str, reason: &str) {
    eventually(&format!("{op} terminal"), || {
        view(agent, SCOPE, op).is_some_and(|v| v.terminal)
    })
    .await;
    assert_eq!(view(agent, SCOPE, op).unwrap().reason, reason, "{op}");
}

/// Index of the first event line containing `needle`.
fn event_at(agent: &AgentHandle, needle: &str) -> usize {
    let events = agent.events();
    events
        .iter()
        .position(|line| line.contains(needle))
        .unwrap_or_else(|| panic!("no event {needle:?} in {events:#?}"))
}

fn spawns(agent: &AgentHandle) -> usize {
    agent
        .events()
        .iter()
        .filter(|line| line.contains("event=backend state=spawned"))
        .count()
}

/// `/proc/<pid>/stat` fields after the command name (index 0 is field 3, `state`); `None` once it is gone.
fn proc_stat(pid: i32) -> Option<Vec<String>> {
    let text = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let (_, rest) = text.rsplit_once(')')?;
    Some(rest.split_whitespace().map(str::to_owned).collect())
}

/// The process start time in clock ticks (field 22).
fn start_ticks(pid: i32) -> Option<u64> {
    proc_stat(pid)?.get(19)?.parse().ok()
}

/// Pids in process group `pgid` (field 5), zombies included.
fn group_members(pgid: i32) -> Vec<i32> {
    std::fs::read_dir("/proc")
        .unwrap()
        .filter_map(Result::ok)
        .filter_map(|e| e.file_name().to_str()?.parse::<i32>().ok())
        .filter(|pid| proc_stat(*pid).and_then(|s| s.get(2)?.parse::<i32>().ok()) == Some(pgid))
        .collect()
}

fn sigkill(pid: i32) {
    // SAFETY: kill(2) has no memory preconditions.
    assert_eq!(unsafe { libc::kill(pid, libc::SIGKILL) }, 0, "kill {pid}");
}

/// SIGKILLs process group `pgid` on drop (a failed assertion must not leave the backend group behind).
struct GroupGuard(i32);

impl Drop for GroupGuard {
    fn drop(&mut self) {
        // SAFETY: kill(2) has no memory preconditions; ESRCH (already gone) is the expected outcome.
        unsafe { libc::kill(-self.0, libc::SIGKILL) };
    }
}

/// A started process, SIGKILLed and reaped on drop.
struct Reaped(std::process::Child);

impl Reaped {
    /// `spawn` returns once the child has exec'd, so its `/proc/<pid>/exe` already names `program`.
    fn spawn(program: &Path, args: &[String]) -> Reaped {
        Reaped(
            std::process::Command::new(program)
                .args(args)
                .stdin(std::process::Stdio::null())
                .spawn()
                .unwrap(),
        )
    }
}

impl Drop for Reaped {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// O48: [`Fixture::children`] counts the processes executing the fixture's mock copy. Another process naming the
/// fixture port is no child: a `make ci-docker` container on the same host runs this binary in its own network
/// namespace over the same port window, so its mock can carry the same `--port` while this test's backend is gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn children_are_the_fixtures_mock_copy_not_any_process_naming_its_port() {
    let f = Fixture::new().await;
    // The startup delay comes before the bind: neither mock listens during the test.
    let args = [
        "--port".to_owned(),
        f.port.to_string(),
        "--startup-delay-ms".to_owned(),
        "600000".to_owned(),
    ];
    let foreign = Reaped::spawn(&mock_bin::built(), &args);
    assert_eq!(f.children(), 0, "the built mock on the fixture port");
    let own = Reaped::spawn(&f.placed_mock(), &args);
    assert_eq!(f.children(), 1, "the fixture's copy");
    drop(own);
    assert_eq!(f.children(), 0, "the copy is reaped");
    drop(foreign);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn load_reaches_ready_with_a_joined_second_key_and_one_child() {
    let f = Fixture::new().await;
    let script = f.mock_script(&confirmed_info(1), 1500);
    let agent = f.start(f.managed(&script, UNLIMITED)).await;
    let mut c = f.control(&agent);
    let inv = inventory(&mut c, &agent).await;
    assert_eq!(model_state(&inv), "unloaded");
    assert_unconfirmed(&inv);
    assert_eq!(agent.generation(), 0);

    let first = c
        .load_model(load(&agent, "load-1"))
        .await
        .unwrap()
        .into_inner();
    assert!(first.accepted);
    assert_eq!(first.operation_id, "load-1");
    let inv = wait_state(&mut c, &agent, "loading").await;
    assert_unconfirmed(&inv);
    assert!(f.child_file().exists(), "loading only after the record");
    let second = c
        .load_model(load(&agent, "load-2"))
        .await
        .unwrap()
        .into_inner();
    assert!(second.accepted);
    assert!(
        !view(&agent, CONTROL_SCOPE, "load-2").unwrap().terminal,
        "joins the load"
    );
    // A retry of the first key returns its receipt.
    let again = c
        .load_model(load(&agent, "load-1"))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(again.completion_ticket, first.completion_ticket);

    let inv = wait_state(&mut c, &agent, "ready").await;
    assert_eq!(agent.generation(), 1);
    assert_confirmed(&inv, 1);
    ends(&agent, "load-1", "").await;
    ends(&agent, "load-2", "").await;
    assert_eq!(f.children(), 1, "exactly one launched backend");
    assert_eq!(model_events(&agent), ["loading", "ready"]);

    agent.shutdown().await;
    assert_eq!(f.children(), 0, "shutdown stops the child");
    assert!(!f.child_file().exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unload_without_live_ops_stops_the_child_and_removes_its_record() {
    let f = Fixture::new().await;
    let (agent, mut c) = ready_agent(&f).await;
    assert_eq!(f.children(), 1);
    let receipt = c
        .unload_model(unload(&agent, "unload-1"))
        .await
        .unwrap()
        .into_inner();
    assert!(receipt.accepted);
    ends(&agent, "unload-1", "").await;
    let inv = wait_state(&mut c, &agent, "unloaded").await;
    assert_unconfirmed(&inv);
    assert_eq!(f.children(), 0, "the child is stopped");
    assert!(!f.child_file().exists(), "{CHILD_FILE} removed");
    assert_eq!(
        model_events(&agent),
        ["loading", "ready", "draining", "unloading", "unloaded"]
    );
    agent.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn attached_mode_refuses_lifecycle_calls_model_unmanaged() {
    let f = Fixture::attached().await;
    let agent = f.start(f.attached_config()).await;
    let mut c = f.control(&agent);
    refused(
        c.load_model(load(&agent, "load-1")).await,
        Code::FailedPrecondition,
        wire::MODEL_UNMANAGED,
    );
    refused(
        c.drain_model(drain(&agent, "drain-1")).await,
        Code::FailedPrecondition,
        wire::MODEL_UNMANAGED,
    );
    refused(
        c.unload_model(unload(&agent, "unload-1")).await,
        Code::FailedPrecondition,
        wire::MODEL_UNMANAGED,
    );
    assert!(all_ops(&agent).is_empty(), "nothing journaled");
    let inv = wait_state(&mut c, &agent, "attached").await;
    assert!(inv["models"][0]["lock_sha256"].is_null());
    agent.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn malformed_and_unapproved_lifecycle_calls_change_nothing() {
    let f = Fixture::new().await;
    let script = f.mock_script(&confirmed_info(1), 0);
    let agent = f.start(f.managed(&script, UNLIMITED)).await;
    let mut c = f.control(&agent);
    refused(
        c.load_model(lifecycle(
            &agent,
            wire::METHOD_LOAD,
            "l-1",
            &[0xcd; 32],
            "",
            &[],
        ))
        .await,
        Code::FailedPrecondition,
        wire::LOCK_NOT_APPROVED,
    );
    refused(
        c.load_model(lifecycle(
            &agent,
            wire::METHOD_LOAD,
            "l-2",
            &LOCK[..31],
            "",
            &[],
        ))
        .await,
        Code::InvalidArgument,
        wire::MALFORMED_REQUEST,
    );
    refused(
        c.load_model(lifecycle(
            &agent,
            wire::METHOD_LOAD,
            "l-3",
            &LOCK,
            "",
            &["gpu0"],
        ))
        .await,
        Code::FailedPrecondition,
        wire::UNSUPPORTED_PLACEMENT,
    );
    refused(
        c.unload_model(lifecycle(
            &agent,
            wire::METHOD_UNLOAD,
            "u-1",
            &[0xcd; 32],
            "",
            &[],
        ))
        .await,
        Code::FailedPrecondition,
        wire::LOCK_NOT_APPROVED,
    );
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(all_ops(&agent).is_empty(), "nothing journaled");
    assert!(!f.child_file().exists(), "no child");
    assert_eq!(f.children(), 0);
    assert_eq!(model_state(&inventory(&mut c, &agent).await), "unloaded");
    agent.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn submit_work_is_refused_while_unloaded_and_while_draining() {
    let f = Fixture::new().await;
    let script = f.mock_script(&confirmed_info(1), 0);
    let agent = f.start(f.managed(&script, UNLIMITED)).await;
    let mut work = f.inference(&agent);
    refused(
        work.submit_work(submit(&agent, "w-1", 1)).await,
        Code::Unavailable,
        wire::BACKEND_UNCONFIRMED,
    );
    let mut c = f.control(&agent);
    c.load_model(load(&agent, "load-1")).await.unwrap();
    wait_state(&mut c, &agent, "ready").await;
    c.drain_model(drain(&agent, "drain-1")).await.unwrap();
    ends(&agent, "drain-1", "").await;
    let inv = wait_state(&mut c, &agent, "draining").await;
    assert_confirmed(&inv, 1);
    refused(
        work.submit_work(submit(&agent, "w-2", 2)).await,
        Code::Unavailable,
        wire::MODEL_DRAINING,
    );
    assert!(
        all_ops(&agent)
            .iter()
            .all(|op| op.method != wire::METHOD_SUBMIT),
        "no SubmitWork journaled"
    );
    agent.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_backend_exiting_during_load_fails_the_model() {
    let f = Fixture::new().await;
    let script = f.script("exit 3");
    let agent = f.start(f.managed(&script, UNLIMITED)).await;
    let mut c = f.control(&agent);
    c.load_model(load(&agent, "load-1")).await.unwrap();
    ends(&agent, "load-1", wire::MODEL_LOAD_FAILED).await;
    let inv = wait_state(&mut c, &agent, "failed").await;
    assert_unconfirmed(&inv);
    eventually("record removed", || !f.child_file().exists()).await;
    assert_eq!(agent.generation(), 0);
    agent.shutdown().await;
}

/// M36-02: the Load receipt was lost after `ready`; the retry carries the original Authority (generation 0, now
/// stale) and gets the original receipt, not `stale_generation` and not a second child.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_load_retried_after_ready_with_its_original_authority_returns_the_original_receipt() {
    let f = Fixture::new().await;
    let script = f.mock_script(&confirmed_info(1), 0);
    let agent = f.start(f.managed(&script, UNLIMITED)).await;
    let mut c = f.control(&agent);
    let original = load(&agent, "load-1");
    assert_eq!(
        original.authority.as_ref().unwrap().model_generation,
        Some(0)
    );
    let first = c.load_model(original.clone()).await.unwrap().into_inner();
    wait_state(&mut c, &agent, "ready").await;
    ends(&agent, "load-1", "").await;
    assert_eq!(agent.generation(), 1);

    let again = c.load_model(original).await.unwrap().into_inner();
    assert_eq!(again, first, "the original receipt");
    ended_at_once(&agent, "load-1", "");
    assert_eq!(all_ops(&agent).len(), 1);
    assert_eq!(f.children(), 1, "one child");
    assert_eq!(spawns(&agent), 1);
    assert_eq!(model_events(&agent), ["loading", "ready"]);
    agent.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pre_bound_backend_port_fails_the_load_without_a_spawn() {
    let f = Fixture::new().await;
    let squatter = std::net::TcpListener::bind(("127.0.0.1", f.port)).unwrap();
    let script = f.mock_script(&confirmed_info(1), 0);
    let agent = f.start(f.managed(&script, UNLIMITED)).await;
    let mut c = f.control(&agent);
    c.load_model(load(&agent, "load-1")).await.unwrap();
    ends(&agent, "load-1", wire::BACKEND_PORT_IN_USE).await;
    let inv = wait_state(&mut c, &agent, "failed").await;
    assert_unconfirmed(&inv);
    assert!(!f.child_file().exists(), "no {CHILD_FILE}");
    assert_eq!(f.children(), 0);
    assert_eq!(spawns(&agent), 0, "no spawn");
    assert_eq!(agent.generation(), 0);
    drop(squatter);
    agent.shutdown().await;
}

/// §3.6: a listener on `base_url` that this agent did not launch is never confirmed, over >= 3 identity probe
/// periods (one heartbeat interval, 100 ms, each).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_foreign_listener_while_unloaded_is_never_confirmed() {
    let f = Fixture::new().await;
    let script = f.mock_script(&confirmed_info(1), 0);
    let agent = f.start(f.managed(&script, UNLIMITED)).await;
    let mut c = f.control(&agent);
    let foreign = MockSglang::start(MockConfig {
        port: f.port,
        server_info: confirmed_info(1),
        ..MockConfig::default()
    })
    .await
    .unwrap();
    assert_eq!(foreign.addr().port(), f.port);
    let until = Instant::now() + Duration::from_millis(600);
    while Instant::now() < until {
        let inv = inventory(&mut c, &agent).await;
        assert_eq!(model_state(&inv), "unloaded");
        assert_unconfirmed(&inv);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(agent.generation(), 0);
    assert!(agent.shared().identity().is_none(), "no identity");
    assert!(model_events(&agent).is_empty());
    assert!(all_ops(&agent).is_empty());
    drop(foreign);
    agent.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unload_waits_for_a_live_op_then_stops_the_child() {
    let f = Fixture::new().await;
    let (agent, mut c) = ready_agent_with(&f, &f.slow_mock_script(700)).await;
    let mut work = f.inference(&agent);
    live_op(&mut work, &agent, "w-1", 1).await;
    c.unload_model(unload(&agent, "unload-1")).await.unwrap();
    let inv = wait_state(&mut c, &agent, "draining").await;
    assert_confirmed(&inv, 1);
    assert!(pending(&agent, "unload-1"), "waits for the live op");
    assert!(!view(&agent, SCOPE, "w-1").unwrap().terminal);
    assert_eq!(f.children(), 1, "the child runs while the op is live");

    work_ends(&agent, "w-1", "").await;
    ends(&agent, "unload-1", "").await;
    let inv = wait_state(&mut c, &agent, "unloaded").await;
    assert_unconfirmed(&inv);
    assert_eq!(f.children(), 0);
    assert!(!f.child_file().exists());
    assert_eq!(
        model_events(&agent),
        ["loading", "ready", "draining", "unloading", "unloaded"]
    );
    assert!(
        event_at(&agent, &format!("event=cleanup op={SCOPE}/w-1"))
            < event_at(&agent, "event=model state=unloading"),
        "the child is stopped only after the op's cleanup"
    );
    agent.shutdown().await;
}

/// §3.4: Unload/Drain in `loading` and Load/Drain in `unloading` -> `model_busy`, refused before the journal.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lifecycle_calls_during_a_transition_are_refused_model_busy() {
    let f = Fixture::new().await;
    let script = f.mock_script(&confirmed_info(1), 60_000);
    let agent = f.start(f.managed(&script, UNLIMITED)).await;
    let mut c = f.control(&agent);
    c.load_model(load(&agent, "load-1")).await.unwrap();
    wait_state(&mut c, &agent, "loading").await;
    let before = f.journal_bytes();
    assert!(!before.is_empty());
    refused(
        c.unload_model(unload(&agent, "unload-1")).await,
        Code::FailedPrecondition,
        wire::MODEL_BUSY,
    );
    refused(
        c.drain_model(drain(&agent, "drain-1")).await,
        Code::FailedPrecondition,
        wire::MODEL_BUSY,
    );
    assert_eq!(f.journal_bytes(), before, "nothing journaled");
    assert!(view(&agent, CONTROL_SCOPE, "unload-1").is_none());
    assert!(pending(&agent, "load-1"));
    assert_eq!(model_state(&inventory(&mut c, &agent).await), "loading");
    agent.shutdown().await;

    // `unloading` lasts `stop_timeout_ms`: the group leader ignores SIGTERM.
    let f = Fixture::new().await;
    let (agent, mut c) = ready_agent_with(&f, &f.stubborn_script(0)).await;
    c.unload_model(unload(&agent, "unload-1")).await.unwrap();
    wait_state(&mut c, &agent, "unloading").await;
    let before = f.journal_bytes();
    refused(
        c.load_model(load(&agent, "load-2")).await,
        Code::FailedPrecondition,
        wire::MODEL_BUSY,
    );
    refused(
        c.drain_model(drain(&agent, "drain-2")).await,
        Code::FailedPrecondition,
        wire::MODEL_BUSY,
    );
    assert_eq!(f.journal_bytes(), before, "nothing journaled");
    assert!(view(&agent, CONTROL_SCOPE, "load-2").is_none());
    ends(&agent, "unload-1", "").await;
    wait_state(&mut c, &agent, "unloaded").await;
    assert_eq!(f.children(), 0);
    agent.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn load_while_draining_cancels_the_drain() {
    let f = Fixture::new().await;
    let (agent, mut c) = ready_agent_with(&f, &f.slow_mock_script(10_000)).await;
    let mut work = f.inference(&agent);
    live_op(&mut work, &agent, "w-1", 1).await;
    c.drain_model(drain(&agent, "drain-1")).await.unwrap();
    wait_state(&mut c, &agent, "draining").await;
    assert!(pending(&agent, "drain-1"), "held by the live op");

    let receipt = c
        .load_model(load(&agent, "load-2"))
        .await
        .unwrap()
        .into_inner();
    assert!(receipt.accepted);
    ended_at_once(&agent, "load-2", "");
    ended_at_once(&agent, "drain-1", wire::CANCEL_REQUESTED);
    let inv = wait_state(&mut c, &agent, "ready").await;
    assert_confirmed(&inv, 1);
    assert!(
        !view(&agent, SCOPE, "w-1").unwrap().terminal,
        "the live op is untouched"
    );
    assert_eq!(f.children(), 1);
    assert_eq!(spawns(&agent), 1);
    live_op(&mut work, &agent, "w-2", 2).await;
    agent.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unload_in_failed_ends_unloaded_at_once() {
    let f = Fixture::new().await;
    let script = f.script("exit 3");
    let agent = f.start(f.managed(&script, UNLIMITED)).await;
    let mut c = f.control(&agent);
    c.load_model(load(&agent, "load-1")).await.unwrap();
    ends(&agent, "load-1", wire::MODEL_LOAD_FAILED).await;
    wait_state(&mut c, &agent, "failed").await;
    eventually("record removed", || !f.child_file().exists()).await;

    let receipt = c
        .unload_model(unload(&agent, "unload-1"))
        .await
        .unwrap()
        .into_inner();
    assert!(receipt.accepted);
    ended_at_once(&agent, "unload-1", "");
    let inv = inventory(&mut c, &agent).await;
    assert_eq!(model_state(&inv), "unloaded");
    assert_unconfirmed(&inv);
    assert!(model_events(&agent).ends_with(&["failed".to_owned(), "unloaded".to_owned()]));
    assert_eq!(f.children(), 0);
    agent.shutdown().await;
}

/// §3.6: the backend dies under a pending Drain and a live SubmitWork: `failed`, the Drain ends `backend_exited`,
/// the op `backend_protocol_error`, and nothing is respawned.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_child_crash_while_serving_fails_the_model_without_a_respawn() {
    let f = Fixture::new().await;
    let (agent, mut c) = ready_agent_with(&f, &f.slow_mock_script(10_000)).await;
    let mut work = f.inference(&agent);
    live_op(&mut work, &agent, "w-1", 1).await;
    c.drain_model(drain(&agent, "drain-1")).await.unwrap();
    wait_state(&mut c, &agent, "draining").await;
    assert!(pending(&agent, "drain-1"));

    sigkill(f.record().pid);
    ends(&agent, "drain-1", wire::BACKEND_EXITED).await;
    work_ends(&agent, "w-1", "backend_protocol_error").await;
    let inv = wait_state(&mut c, &agent, "failed").await;
    assert_unconfirmed(&inv);
    eventually("record removed", || !f.child_file().exists()).await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(f.children(), 0, "no respawn");
    assert_eq!(spawns(&agent), 1);
    assert_eq!(model_state(&inventory(&mut c, &agent).await), "failed");
    agent.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_crash_with_a_pending_unload_ends_unloaded() {
    let f = Fixture::new().await;
    let (agent, mut c) = ready_agent_with(&f, &f.slow_mock_script(10_000)).await;
    let mut work = f.inference(&agent);
    live_op(&mut work, &agent, "w-1", 1).await;
    c.unload_model(unload(&agent, "unload-1")).await.unwrap();
    wait_state(&mut c, &agent, "draining").await;
    assert!(pending(&agent, "unload-1"));

    sigkill(f.record().pid);
    ends(&agent, "unload-1", "").await;
    let inv = wait_state(&mut c, &agent, "unloaded").await;
    assert_unconfirmed(&inv);
    work_ends(&agent, "w-1", "backend_protocol_error").await;
    eventually("record removed", || !f.child_file().exists()).await;
    assert_eq!(f.children(), 0);
    assert_eq!(spawns(&agent), 1);
    agent.shutdown().await;
}

/// Replaces `backend_child.json` with a directory: the next stop kills the group but cannot remove the record
/// (`StopError::Io`, also as root).
fn block_record_removal(f: &Fixture) {
    std::fs::remove_file(f.child_file()).unwrap();
    std::fs::create_dir(f.child_file()).unwrap();
}

/// The agent is poisoned by an unconfirmed stop: one `backend_stop` event, the model `failed`, and a new lifecycle
/// key is `UNAVAILABLE poisoned` before any spawn.
async fn poisoned_by_the_stop(agent: &AgentHandle, c: &mut WorkerServiceClient<Channel>) {
    eventually("poisoned", || {
        agent.admission_state() == AdmissionState::Poisoned
    })
    .await;
    assert_eq!(
        agent
            .events()
            .iter()
            .filter(|line| line.contains("event=poisoned reason=backend_stop"))
            .count(),
        1
    );
    let inv = wait_state(c, agent, "failed").await;
    assert_unconfirmed(&inv);
    assert_eq!(inv["admission_state"], "poisoned");
    refused(
        c.load_model(load(agent, "load-2")).await,
        Code::Unavailable,
        wire::POISONED,
    );
    assert!(view(agent, CONTROL_SCOPE, "load-2").is_none());
    assert_eq!(spawns(agent), 1);
}

/// §3.6/§3.7: a stop that cannot confirm the group gone (here `StopError::Io`) is no confirmed stop: the agent is
/// poisoned, the model ends `failed` (never `unloaded`) and the pending Unload, or the pending Drain of a crashed
/// backend, ends `poisoned`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unconfirmed_stop_poisons_the_agent_and_fails_the_model() {
    let f = Fixture::new().await;
    let (agent, mut c) = ready_agent(&f).await;
    block_record_removal(&f);
    c.unload_model(unload(&agent, "unload-1")).await.unwrap();
    ends(&agent, "unload-1", wire::POISONED).await;
    poisoned_by_the_stop(&agent, &mut c).await;
    assert!(!model_events(&agent).contains(&"unloaded".to_owned()));
    eventually("the group is gone", || f.children() == 0).await;
    agent.shutdown().await;

    let f = Fixture::new().await;
    let (agent, mut c) = ready_agent_with(&f, &f.slow_mock_script(10_000)).await;
    let mut work = f.inference(&agent);
    live_op(&mut work, &agent, "w-1", 1).await;
    c.drain_model(drain(&agent, "drain-1")).await.unwrap();
    wait_state(&mut c, &agent, "draining").await;
    assert!(pending(&agent, "drain-1"));
    let pid = f.record().pid;
    block_record_removal(&f);
    sigkill(pid);
    ends(&agent, "drain-1", wire::POISONED).await;
    poisoned_by_the_stop(&agent, &mut c).await;
    eventually("the group is gone", || f.children() == 0).await;
    agent.shutdown().await;
}

/// Q4: only SubmitWork ops count toward the `max_running_requests` cap (4 in `confirmed_info`): a lifecycle call
/// at the cap is admitted, and a pending DrainModel (journaled, not cleanup-acknowledged) holds no live-op slot.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lifecycle_ops_never_count_toward_queue_full() {
    let f = Fixture::new().await;
    let (agent, mut c) = ready_agent_with(&f, &f.slow_mock_script(10_000)).await;
    let mut work = f.inference(&agent);
    for seq in 1..=3 {
        live_op(&mut work, &agent, &format!("w-{seq}"), seq).await;
    }
    for op in ["load-2", "load-3"] {
        let receipt = c.load_model(load(&agent, op)).await.unwrap().into_inner();
        assert!(receipt.accepted);
        ended_at_once(&agent, op, "");
    }
    live_op(&mut work, &agent, "w-4", 4).await;
    refused(
        work.submit_work(submit(&agent, "w-5", 5)).await,
        Code::ResourceExhausted,
        wire::QUEUE_FULL,
    );
    let receipt = c
        .load_model(load(&agent, "load-4"))
        .await
        .unwrap()
        .into_inner();
    assert!(receipt.accepted, "a lifecycle call at the cap is admitted");
    ended_at_once(&agent, "load-4", "");
    assert_eq!(agent.shared().journal.lock().unwrap().live_ops(), 4);
    c.drain_model(drain(&agent, "drain-1")).await.unwrap();
    wait_state(&mut c, &agent, "draining").await;
    let v = view(&agent, CONTROL_SCOPE, "drain-1").expect("journaled");
    assert!(
        !v.terminal && !v.cleanup_acknowledged,
        "the drain waits for the live ops"
    );
    assert_eq!(
        agent.shared().journal.lock().unwrap().live_ops(),
        4,
        "the pending drain holds no live-op slot"
    );
    agent.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn load_fails_by_startup_timeout_and_by_a_served_model_name_mismatch() {
    let f = Fixture::new().await;
    let script = f.mock_script(&confirmed_info(1), 60_000);
    // The minimum `startup_timeout_ms`; the mock would listen only after 60 s.
    let agent = f.start(f.managed(&script, 1000)).await;
    let mut c = f.control(&agent);
    c.load_model(load(&agent, "load-1")).await.unwrap();
    ends(&agent, "load-1", wire::MODEL_LOAD_FAILED).await;
    let inv = wait_state(&mut c, &agent, "failed").await;
    assert_unconfirmed(&inv);
    eventually("record removed", || !f.child_file().exists()).await;
    assert_eq!(f.children(), 0, "the timed-out child is stopped");
    assert_eq!(agent.generation(), 0);
    agent.shutdown().await;

    let f = Fixture::new().await;
    let mut info = confirmed_info(1);
    info.served_model_name = "another-model".into();
    let script = f.mock_script(&info, 0);
    let agent = f.start(f.managed(&script, UNLIMITED)).await;
    let mut c = f.control(&agent);
    c.load_model(load(&agent, "load-1")).await.unwrap();
    ends(&agent, "load-1", wire::MODEL_LOAD_FAILED).await;
    let inv = wait_state(&mut c, &agent, "failed").await;
    assert_unconfirmed(&inv);
    eventually("record removed", || !f.child_file().exists()).await;
    assert_eq!(f.children(), 0, "the mismatched child is stopped");
    assert_eq!(agent.generation(), 0);
    agent.shutdown().await;
}

/// §3.6: the startup deadline fails the load while the stop of the group still runs. The leader ignores SIGTERM
/// (the ignored disposition survives the `exec`), so the mock binds 2 s after the spawn, inside the 4 s stop; the
/// probe sees it while the model is still `loading`, yet it is never confirmed: no `ready`, generation 0, the Load
/// ends `model_load_failed`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_backend_binding_during_the_deadline_stop_is_never_confirmed() {
    let f = Fixture::new().await;
    let script = f.script(&format!(
        "trap '' TERM\nexec {}",
        f.mock_command(&confirmed_info(1), 2000, 0)
    ));
    let text = f.managed_text_with(&script, 1000, 4000, support::free_port());
    let agent = f
        .start(AgentConfig::from_toml_str(&text, None).unwrap())
        .await;
    let mut c = f.control(&agent);
    c.load_model(load(&agent, "load-1")).await.unwrap();
    let backend = SocketAddr::from(([127, 0, 0, 1], f.port));
    let mut bound_while_loading = false;
    let until = Instant::now() + Duration::from_secs(20);
    loop {
        let inv = inventory(&mut c, &agent).await;
        assert_ne!(model_state(&inv), "ready", "{inv}");
        assert_eq!(agent.generation(), 0, "{inv}");
        match model_state(&inv) {
            "failed" => break,
            "loading" => {
                bound_while_loading |= tokio::net::TcpStream::connect(backend).await.is_ok();
            }
            _ => {}
        }
        assert!(
            Instant::now() < until,
            "timed out waiting for failed: {inv}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(bound_while_loading, "the backend listened during the stop");
    ends(&agent, "load-1", wire::MODEL_LOAD_FAILED).await;
    assert_eq!(model_events(&agent), ["loading", "failed"]);
    assert_eq!(agent.generation(), 0);
    assert!(agent.shared().identity().is_none(), "no identity");
    assert_unconfirmed(&inventory(&mut c, &agent).await);
    eventually("record removed", || !f.child_file().exists()).await;
    assert_eq!(f.children(), 0, "the stop ended with SIGKILL");
    agent.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_worker_principal_calling_load_model_is_permission_denied() {
    let f = Fixture::new().await;
    let script = f.mock_script(&confirmed_info(1), 0);
    let agent = f.start(f.managed(&script, UNLIMITED)).await;
    let mut as_worker =
        WorkerServiceClient::new(f.channel_as(agent.local_addr(), WORKER_PRINCIPAL));
    let status = as_worker
        .load_model(load(&agent, "load-1"))
        .await
        .expect_err("refused");
    assert_eq!(status.code(), Code::PermissionDenied, "{status:?}");
    assert!(all_ops(&agent).is_empty(), "nothing journaled");
    assert!(f.journal_bytes().is_empty());
    assert!(!f.child_file().exists());
    assert_eq!(f.children(), 0);
    let mut c = f.control(&agent);
    assert_eq!(model_state(&inventory(&mut c, &agent).await), "unloaded");
    agent.shutdown().await;
}

/// Q4/§3.8: a term adoption and an in-process epoch jump end work ops only; a pending Load and its child survive.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fences_leave_a_pending_lifecycle_op_and_the_child_untouched() {
    let f = Fixture::new().await;
    let script = f.mock_script(&confirmed_info(1), 60_000);
    let agent = f.start(f.managed(&script, UNLIMITED)).await;
    let mut c = f.control(&agent);
    c.load_model(load(&agent, "load-1")).await.unwrap();
    wait_state(&mut c, &agent, "loading").await;
    let record = f.record();
    assert_eq!(start_ticks(record.pid), Some(record.start_time_ticks));

    f.stub.mark_lost(WORKER);
    eventually("epoch 2", || agent.epoch() == 2).await;
    f.stub.restart();
    eventually("term adopted", || agent.term() == TERM + 1).await;
    eventually("registered again", || {
        f.stub
            .heartbeats()
            .iter()
            .any(|h| h.epoch == 2 && h.term == TERM + 1)
    })
    .await;

    assert!(pending(&agent, "load-1"), "the Load is not fenced");
    assert_eq!(f.record(), record, "same record");
    assert_eq!(
        start_ticks(record.pid),
        Some(record.start_time_ticks),
        "same process"
    );
    assert_eq!(f.children(), 1);
    assert_eq!(spawns(&agent), 1);
    let inv = inventory(&mut c, &agent).await;
    assert_eq!(model_state(&inv), "loading");
    assert_unconfirmed(&inv);
    agent.shutdown().await;
}

/// Q7/§3.9 through `Inventory::from_json_bytes` (every `inventory` call validates the identity-field rule).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_inventory_follows_q7_in_every_model_state() {
    let f = Fixture::new().await;
    let script = f.stubborn_script(1500);
    let agent = f.start(f.managed(&script, UNLIMITED)).await;
    let mut c = f.control(&agent);
    let inv = inventory(&mut c, &agent).await;
    assert_eq!(model_state(&inv), "unloaded");
    assert_unconfirmed(&inv);

    c.load_model(load(&agent, "load-1")).await.unwrap();
    assert_unconfirmed(&wait_state(&mut c, &agent, "loading").await);
    let inv = wait_state(&mut c, &agent, "ready").await;
    let first = agent.generation();
    assert_eq!(first, 1);
    assert_confirmed(&inv, first);

    c.drain_model(drain(&agent, "drain-1")).await.unwrap();
    assert_confirmed(&wait_state(&mut c, &agent, "draining").await, 1);
    ends(&agent, "drain-1", "").await;

    c.unload_model(unload(&agent, "unload-1")).await.unwrap();
    assert_confirmed(&wait_state(&mut c, &agent, "unloading").await, 1);
    ends(&agent, "unload-1", "").await;
    assert_unconfirmed(&wait_state(&mut c, &agent, "unloaded").await);

    c.load_model(load(&agent, "load-2")).await.unwrap();
    let inv = wait_state(&mut c, &agent, "ready").await;
    let generation = agent.generation();
    assert_eq!(
        generation,
        first + 1,
        "the relaunched backend is a new instance"
    );
    assert_confirmed(&inv, generation);
    sigkill(f.record().pid);
    assert_unconfirmed(&wait_state(&mut c, &agent, "failed").await);
    assert_eq!(
        model_events(&agent),
        [
            "loading",
            "ready",
            "draining",
            "unloading",
            "unloaded",
            "loading",
            "ready",
            "failed"
        ]
    );
    agent.shutdown().await;
}

/// The `cere-worker` binary on `config` (SIGKILLed on drop) and its epoch from the startup line.
struct Worker {
    child: std::process::Child,
    epoch: u64,
}

impl Worker {
    async fn spawn(config: &Path) -> Worker {
        use std::io::{BufRead, BufReader};

        let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_cere-worker"))
            .arg("--config")
            .arg(config)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
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
        let epoch = line
            .split_whitespace()
            .find_map(|word| word.strip_prefix(&format!("worker={WORKER}@")))
            .and_then(|epoch| epoch.parse().ok())
            .unwrap_or_else(|| panic!("startup line {line:?}"));
        Worker { child, epoch }
    }

    async fn wait_exit(&mut self) -> std::process::ExitStatus {
        let until = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < until, "cere-worker did not exit");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Registered at `epoch` with a heartbeat after it (the term is adopted).
fn registered(f: &Fixture, epoch: u64) -> bool {
    f.stub
        .registers()
        .iter()
        .any(|r| r.accepted && r.epoch == epoch)
        && f.stub.heartbeats().iter().any(|h| h.epoch == epoch)
}

/// §3.2: the agent dies (SIGKILL) while loading; the backend leader dies with it (parent-death signal) but a
/// process it forked survives in the old group. The next start on the same `state_dir` kills that group, removes
/// the record, starts `unloaded`, and the old Load ends `worker_restart`, cleanup-acknowledged.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_agent_restart_kills_the_old_group_and_ends_the_load_worker_restart() {
    let f = Fixture::new().await;
    let mock = f.mock_command(&confirmed_info(1), 60_000, 0);
    let script = f.script(&format!("sleep 300 &\nexec {mock}"));
    let listen = support::free_port();
    let config = f.root.path().join("agent.toml");
    std::fs::write(&config, f.managed_text(&script, UNLIMITED, listen)).unwrap();
    let addr = SocketAddr::from(([127, 0, 0, 1], listen));
    let mut c = WorkerServiceClient::new(f.channel_as(addr, COORD_PRINCIPAL));

    let mut first = Worker::spawn(&config).await;
    let epoch = first.epoch;
    eventually("registered", || registered(&f, epoch)).await;
    let base = pb::Authority {
        model_generation: Some(0),
        ..authority_at(epoch, TERM)
    };
    c.load_model(lifecycle_on(
        base,
        wire::METHOD_LOAD,
        "load-1",
        &LOCK,
        "",
        &[],
    ))
    .await
    .unwrap();
    eventually("the record", || f.child_file().exists()).await;
    let record = f.record();
    let _group = GroupGuard(record.pgid);
    // The record and the group appear inside `supervisor.spawn()` (the record is renamed into place, then the
    // directory is fsynced); the agent reports `loading` only once that call has returned (design-m36-agent.md
    // §3.5 step 3; while launching it reports the prior state, `unloaded`). Wait on the agent's own state.
    wait_state_for(&mut c, authority_at(epoch, TERM), "loading").await;
    eventually("leader and sleep in the group", || {
        group_members(record.pgid).len() == 2
    })
    .await;

    first.child.kill().unwrap();
    first.wait_exit().await;
    eventually("the leader died with the agent", || {
        start_ticks(record.pid).is_none()
    })
    .await;
    assert_eq!(
        group_members(record.pgid).len(),
        1,
        "the forked sleep outlives the agent"
    );
    assert!(f.child_file().exists());

    let second = Worker::spawn(&config).await;
    assert!(second.epoch > epoch);
    eventually("the old group is gone", || {
        group_members(record.pgid).is_empty()
    })
    .await;
    assert!(!f.child_file().exists(), "{CHILD_FILE} removed");
    eventually("registered again", || registered(&f, second.epoch)).await;
    let read = authority_at(second.epoch, TERM);
    let inv = inventory_for(&mut c, read.clone()).await;
    assert_eq!(model_state(&inv), "unloaded");
    assert_unconfirmed(&inv);
    let status = c
        .get_operation(pb::OperationQuery {
            authority: Some(pb::Authority {
                scope_id: CONTROL_SCOPE.into(),
                ..read
            }),
            queried_operation_id: "load-1".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(status.terminal);
    assert_eq!(status.reason_code, wire::WORKER_RESTART);
    assert!(status.cleanup_acknowledged, "cleanup-acknowledged");
    assert_eq!(f.children(), 0);
    drop(second);
}

/// `text` with `[backend] load_endpoint = "/get_load"` (H3(b): ops of an earlier process are released by the
/// `/get_load` rule).
fn with_load_endpoint(text: &str) -> String {
    let out = text.replacen(
        "lock_sha256 = ",
        "load_endpoint = \"/get_load\"\nlock_sha256 = ",
        1,
    );
    assert_ne!(out, text);
    out
}

/// Q4/§3.2: the agent is SIGKILLed with a SubmitWork in flight and its backend dies with it. The next start on the
/// same `state_dir` reconciles the group (a confirmed stop), ends the op `worker_restart` and cleanup-acknowledges
/// it at once: nothing polls the dead backend's `/get_load` for it (`load_endpoint` is set), and it never counts
/// toward `queue_full` of the next child.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restart_after_a_kill_cleanup_acknowledges_the_old_childs_work() {
    let f = Fixture::new().await;
    // The first agent's stdout is a pipe the test stops reading after the startup line.
    let mock = f.mock_command(&confirmed_info(1), 0, 10_000);
    let script = f.script(&format!("exec {mock} >/dev/null"));
    let listen = support::free_port();
    let config = f.root.path().join("agent.toml");
    std::fs::write(
        &config,
        with_load_endpoint(&f.managed_text(&script, UNLIMITED, listen)),
    )
    .unwrap();
    let addr = SocketAddr::from(([127, 0, 0, 1], listen));
    let channel = f.channel_as(addr, COORD_PRINCIPAL);
    let mut c = WorkerServiceClient::new(channel.clone());

    let mut first = Worker::spawn(&config).await;
    let epoch = first.epoch;
    eventually("registered", || registered(&f, epoch)).await;
    let base = pb::Authority {
        model_generation: Some(0),
        ..authority_at(epoch, TERM)
    };
    c.load_model(lifecycle_on(
        base,
        wire::METHOD_LOAD,
        "load-1",
        &LOCK,
        "",
        &[],
    ))
    .await
    .unwrap();
    eventually("the record", || f.child_file().exists()).await;
    let record = f.record();
    let _group = GroupGuard(record.pgid);
    eventually("admission open at generation 1", || {
        f.stub
            .heartbeats()
            .iter()
            .any(|h| h.epoch == epoch && h.model_generation == 1 && h.admission_state == "open")
    })
    .await;
    let mut work = WorkerInferenceServiceClient::new(channel);
    let base = pb::Authority {
        model_generation: Some(1),
        ..authority_at(epoch, TERM)
    };
    let receipt = work
        .submit_work(submit_on(base, "w-1", 1))
        .await
        .unwrap()
        .into_inner();
    assert!(receipt.accepted);
    first.child.kill().unwrap();
    first.wait_exit().await;
    eventually("the backend died with the agent", || {
        start_ticks(record.pid).is_none()
    })
    .await;
    drop(first);

    let text = with_load_endpoint(&f.managed_text(&script, UNLIMITED, support::free_port()));
    let agent = f
        .start(AgentConfig::from_toml_str(&text, None).unwrap())
        .await;
    assert!(!f.child_file().exists(), "{CHILD_FILE} reconciled");
    let mut c = f.control(&agent);
    let status = c
        .get_operation(pb::OperationQuery {
            authority: Some(pb::Authority {
                scope_id: SCOPE.into(),
                ..read_authority(&agent)
            }),
            queried_operation_id: "w-1".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(status.terminal);
    assert_eq!(status.reason_code, wire::WORKER_RESTART);
    assert!(status.cleanup_acknowledged, "cleanup-acknowledged at start");
    assert_eq!(agent.shared().journal.lock().unwrap().live_ops(), 0);

    c.load_model(load(&agent, "load-2")).await.unwrap();
    wait_state(&mut c, &agent, "ready").await;
    eventually("admission open", || {
        agent.admission_state() == AdmissionState::Open
    })
    .await;
    let mut work = f.inference(&agent);
    for seq in 2..=5 {
        live_op(&mut work, &agent, &format!("w-{seq}"), seq).await;
    }
    refused(
        work.submit_work(submit(&agent, "w-6", 6)).await,
        Code::ResourceExhausted,
        wire::QUEUE_FULL,
    );
    agent.shutdown().await;
    assert_eq!(f.children(), 0);
}

/// Runs `cere-worker --config <config>` to its exit; a worker still running after 30 s (it started instead of
/// refusing) is killed and fails the test rather than hanging it.
async fn run_worker(config: &Path) -> std::process::Output {
    let config = config.to_owned();
    tokio::task::spawn_blocking(move || {
        let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_cere-worker"))
            .arg("--config")
            .arg(&config)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while child.try_wait().unwrap().is_none() {
            if std::time::Instant::now() >= deadline {
                child.kill().unwrap();
                let out = child.wait_with_output().unwrap();
                panic!(
                    "cere-worker did not exit in time; stderr: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        child.wait_with_output().unwrap()
    })
    .await
    .unwrap()
}

/// design-m36-agent.md §2.3: only a recorded group that survives start reconciliation exits 1 (here an unreaped
/// zombie of this test: SIGKILL cannot remove it); a `backend_child.json` that cannot be parsed exits 2 like every
/// other startup error. Both exit before the startup line and keep the record.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn start_reconciliation_exits_1_on_survivors_and_2_on_a_corrupt_record() {
    use std::os::unix::process::CommandExt;

    let f = Fixture::new().await;
    let script = f.mock_script(&confirmed_info(1), 0);
    let config = f.root.path().join("agent.toml");
    std::fs::write(
        &config,
        f.managed_text(&script, UNLIMITED, support::free_port()),
    )
    .unwrap();
    std::fs::create_dir_all(f.state_dir()).unwrap();
    chmod(&f.state_dir(), 0o700);

    let mut zombie = std::process::Command::new("/bin/sh")
        .args(["-c", "exit 0"])
        .process_group(0)
        .spawn()
        .unwrap();
    let pid = zombie.id() as i32;
    eventually("the zombie", || {
        proc_stat(pid).is_some_and(|fields| fields[0] == "Z")
    })
    .await;
    let record = ChildRecord {
        pid,
        pgid: pid,
        start_time_ticks: start_ticks(pid).unwrap(),
    };
    std::fs::write(f.child_file(), serde_json::to_vec(&record).unwrap()).unwrap();
    let out = run_worker(&config).await;
    zombie.wait().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(out.stdout.is_empty(), "no startup line");
    assert_eq!(f.record(), record, "the next start reconciles again");

    std::fs::write(f.child_file(), b"{").unwrap();
    let out = run_worker(&config).await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains(CHILD_FILE), "{stderr}");
    assert!(out.stdout.is_empty(), "no startup line");
    assert_eq!(std::fs::read(f.child_file()).unwrap(), b"{");
}

/// m36 design note P2/Q13: `--check-config` checks only that `argv[0]` is absolute; the agent start checks that
/// it exists as an executable regular file, so a typo exits 2 at start instead of failing the first `LoadModel`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_missing_or_non_executable_program_exits_2_at_start() {
    use std::os::unix::fs::PermissionsExt;

    let f = Fixture::new().await;
    let script = f.mock_script(&confirmed_info(1), 0);
    let plain = f.root.path().join("not-executable");
    std::fs::write(&plain, b"#!/bin/sh\n").unwrap();
    std::fs::set_permissions(&plain, std::fs::Permissions::from_mode(0o644)).unwrap();
    let missing = f.root.path().join("missing-program");
    for program in [missing, plain, f.root.path().to_path_buf()] {
        let text = f
            .managed_text(&script, UNLIMITED, support::free_port())
            .replace(
                "argv = [\"/bin/sh\"",
                &format!("argv = [{:?}", program.to_str().unwrap()),
            );
        let config = f.root.path().join("agent.toml");
        std::fs::write(&config, text).unwrap();
        let out = run_worker(&config).await;
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{program:?}: {stderr}");
        assert!(stderr.contains("argv[0]"), "{program:?}: {stderr}");
        assert!(out.stdout.is_empty(), "no startup line");
    }
}
