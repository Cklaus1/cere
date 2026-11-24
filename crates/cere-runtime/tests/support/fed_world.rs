//! Two-node world on one host, shared by `tests/federation.rs` (m35 design note P9; build-loop-m35 design note
//! track MATRIX-A): the real `cered` and the real `cere-worker` with an rcgen cluster PKI ([`super::pki`]) and
//! shrunk Q7 timings. Both directions of the federation link run through a loopback [`Relay`] that a test
//! black-holes to partition the nodes. `cere-rpc-probe` makes the forged and duplicate calls; [`Seen`] carries the
//! epoch, term and generation they present. The agent attaches to an in-process `cere_mock_sglang`; the
//! coordinator's local worker never turns ready (its mock serves another model), so every client request goes to
//! the remote worker, and [`World::start`] returns once `A` is Ready.
//!
//! [`World::start_managed`] (m36 design note P8) runs the agent in managed mode instead: it launches the built
//! mock from a [`MockBin`] on the coordinator's `LoadModel`, and the local worker (`gpu1`) serves the test model.

use std::io::{BufRead, BufReader, ErrorKind, Read};
use std::net::SocketAddr;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, Command, Stdio};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use cere_mock_sglang::{MockConfig, MockHandle, MockSglang, ServerInfo};
use cere_proto::worker::v2 as pb;
use cere_runtime::agent::supervisor::CHILD_FILE;
use cere_runtime::fed::wire;
use pb::worker_service_client::WorkerServiceClient;
use serde_json::{json, Value};
use tonic::transport::{Channel, Endpoint};

use super::mock_bin::{launch_ids, MockBin};
use super::relay::Relay;
use super::*;

pub const A: &str = "worker-a";
pub const PRINCIPAL_A: &str = "cere-worker-a";
/// Shrunk Q7 timings, valid for both validators (`heartbeat < lease < suspect < lost`) and shared by both binaries.
/// Both state dirs are on a tmpfs ([`StateDir::on_tmpfs`]): each journal record is synced on the path a heartbeat
/// waits on, and a disk sync stalled by other writers (seconds on the shared CI host) would outlast them.
pub const HEARTBEAT_MS: u64 = 250;
pub const LEASE_MS: u64 = 1500;
pub const SUSPECT_MS: u64 = 2000;
pub const LOST_MS: u64 = 4000;
pub const RPC_TIMEOUT_MS: u64 = 2000;
/// The probe's per-call deadline.
pub const PROBE_TIMEOUT_MS: u64 = 5000;
/// The scope of every operation the tests submit to the agent directly.
pub const SCOPE: &str = "probe";
/// A refusal answers at once: well below every configured timing.
pub const PROMPT_REFUSAL_MS: u64 = 1000;

/// Bound of a wait for a Joining -> Ready pass, a pure guard: `cered` runs its remote driver with the default tuning,
/// which follows the agent's heartbeats (O182, rt-flake2 design note D3: a join takes about a second) and retries
/// a failed check every `readiness::PROBE_INTERVAL`; a local worker's probe also runs at that interval.
pub fn ready_bound() -> Duration {
    cere_runtime::readiness::PROBE_INTERVAL * 4
}

/// Bound of a short wait (an event, a backend request, an exit): a few heartbeats.
pub fn short_bound() -> Duration {
    Duration::from_millis(HEARTBEAT_MS * 40)
}

/// Polls `check` every 20 ms until it yields a value; panics with `what` after `bound`.
pub async fn until<T>(what: &str, bound: Duration, mut check: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + bound;
    loop {
        if let Some(value) = check() {
            return value;
        }
        assert!(Instant::now() < deadline, "{what}: not within {bound:?}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// A CERE-shaped rid (`sglang::Rid`) for the operations the tests submit to the agent directly.
pub fn rid(n: u64) -> String {
    format!("cere-{:020}-{n:020}-{n:016x}", 999)
}

/// The agent's backend: the live server info with a fixed launch (one `instance_id` per `seed`), the
/// `max_running_requests` bound and `tp_size` (an extra key: the mock does not model it).
pub fn backend_info(seed: u64, max_running_requests: u32) -> ServerInfo {
    let mut info = ServerInfo {
        max_running_requests,
        ..launched(seed, seed as f64)
    };
    info.extra.insert("tp_size".into(), json!(1));
    info
}

/// The managed agent's launch script in its [`MockBin`] (`[backend.launch] argv = ["/bin/sh", <it>]`).
pub const BACKEND_SCRIPT: &str = "backend.sh";
/// The server-info file the launched mock serves.
const SERVER_INFO_FILE: &str = "server-info.json";

/// The coordinator's deployment lock: the SHA-256 of the lock file its `backend.deployment` names, hex.
pub fn deployment_lock() -> String {
    use sha2::{Digest, Sha256};
    let bytes =
        std::fs::read(repo_root().join("configs/deployments/qwen38-flash-next.lock.json")).unwrap();
    format!("{:x}", Sha256::digest(bytes))
}

/// The launch script: `exec`s the [`MockBin`] copy on `config`'s port with its token latency and natural output
/// length, serving the server-info file (so the backend runs as the script's process, in its group).
fn backend_script(bin: &MockBin, config: &MockConfig) -> String {
    format!(
        "#!/bin/sh\nexec {} --port {} --server-info {} --token-latency-ms {} --natural-output-tokens {}\n",
        bin.binary.display(),
        config.port,
        bin.path().join(SERVER_INFO_FILE).display(),
        config.token_latency.as_millis(),
        config.natural_output_tokens,
    )
}

/// Writes the server-info file and the launch script into `bin`; returns the agent's managed `[backend]` keys
/// and `[backend.launch]` table (m36 design note Q3; uid/gid 65534 when this process is root).
fn managed_backend(bin: &MockBin, config: &MockConfig) -> String {
    bin.write(
        SERVER_INFO_FILE,
        &serde_json::to_string(&config.server_info).unwrap(),
        0o644,
    );
    let script = bin.write(BACKEND_SCRIPT, &backend_script(bin, config), 0o755);
    let ids = match launch_ids() {
        Some(id) => format!("uid = {id}\ngid = {id}\n"),
        None => String::new(),
    };
    format!(
        "lock_sha256 = \"{}\"\n\n[backend.launch]\nargv = [\"/bin/sh\", \"{}\"]\n\
         startup_timeout_ms = 20000\nstop_timeout_ms = 2000\n{ids}",
        deployment_lock(),
        script.display(),
    )
}

/// A running `cere-worker --config <path>`; its stderr (the `event=` lines) is collected while it runs.
pub struct Agent {
    pub child: Child,
    /// The epoch of its startup line.
    pub epoch: u64,
    events: Arc<Mutex<EventLog>>,
}

/// The agent's stderr pipe and the lines read from it. Every read of the log first drains the pipe, so a line the
/// agent wrote before a reply the test has seen is in the log (RtFlake2 L3: a background reader alone lagged an
/// `event=accept` line behind its SubmitWork reply under load).
struct EventLog {
    /// The read end, non-blocking.
    pipe: ChildStderr,
    text: String,
    /// Bytes after the last newline read.
    partial: Vec<u8>,
    closed: bool,
}

impl EventLog {
    fn new(pipe: ChildStderr) -> EventLog {
        let fd = pipe.as_raw_fd();
        // SAFETY: `fd` is the open read end `pipe` owns; F_GETFL/F_SETFL change only its status flags.
        let set = unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFL);
            flags >= 0 && libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) == 0
        };
        assert!(set, "O_NONBLOCK on the agent's stderr pipe");
        EventLog {
            pipe,
            text: String::new(),
            partial: Vec::new(),
            closed: false,
        }
    }

    /// Reads every byte the pipe holds now and appends its complete lines.
    fn drain(&mut self) {
        let mut buf = [0u8; 8192];
        while !self.closed {
            match self.pipe.read(&mut buf) {
                Ok(0) => self.closed = true,
                Ok(n) => self.partial.extend_from_slice(&buf[..n]),
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(e) => panic!("reading the agent's stderr: {e}"),
            }
        }
        if let Some(end) = self.partial.iter().rposition(|&b| b == b'\n') {
            let lines: Vec<u8> = self.partial.drain(..=end).collect();
            self.text.push_str(&String::from_utf8_lossy(&lines));
        }
    }

    /// Drains `log` every 20 ms until the pipe closes, so it never fills while no test reads the log.
    fn pump(log: &Mutex<EventLog>) {
        loop {
            {
                let mut log = log.lock().unwrap_or_else(PoisonError::into_inner);
                log.drain();
                if log.closed {
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Agent {
    pub async fn spawn(config: &Path) -> Agent {
        let mut child = Command::new(env!("CARGO_BIN_EXE_cere-worker"))
            .arg("--config")
            .arg(config)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let events = Arc::new(Mutex::new(EventLog::new(child.stderr.take().unwrap())));
        let sink = Arc::clone(&events);
        std::thread::spawn(move || EventLog::pump(&sink));
        let stdout = child.stdout.take().unwrap();
        let line = tokio::time::timeout(
            short_bound(),
            tokio::task::spawn_blocking(move || {
                let mut stdout = BufReader::new(stdout);
                let mut line = String::new();
                stdout.read_line(&mut line).unwrap();
                // Keep draining: a managed agent's backend inherits this pipe and exits on a write error.
                std::thread::spawn(move || std::io::copy(&mut stdout, &mut std::io::sink()));
                line
            }),
        )
        .await
        .expect("cere-worker printed no startup line in time")
        .unwrap();
        let epoch = line
            .split_once(&format!(" worker={A}@"))
            .and_then(|(_, rest)| rest.split_once(' '))
            .and_then(|(epoch, _)| epoch.parse().ok());
        let mut agent = Agent {
            child,
            epoch: 0,
            events,
        };
        match epoch {
            Some(epoch) => agent.epoch = epoch,
            None => panic!(
                "unexpected startup line {line:?}; stderr: {}",
                agent.events()
            ),
        }
        agent
    }

    /// Every line the agent has written so far.
    pub fn events(&self) -> String {
        let mut log = self.events.lock().unwrap_or_else(PoisonError::into_inner);
        log.drain();
        log.text.clone()
    }

    /// Event lines containing `needle`.
    pub fn count(&self, needle: &str) -> usize {
        self.events().lines().filter(|l| l.contains(needle)).count()
    }

    /// Waits until an event line contains `needle`.
    pub async fn until_event(&self, needle: &str, bound: Duration) {
        let what = format!("cere-worker event {needle:?}");
        until(&what, bound, || (self.count(needle) > 0).then_some(())).await;
    }

    /// The index of the first event line that is exactly `cere-worker <event>` (an op's `contains` needle also
    /// matches its cancel operations' lines).
    pub fn position(&self, event: &str) -> Option<usize> {
        let line = format!("cere-worker {event}");
        self.events().lines().position(|l| l == line)
    }

    /// Waits until an event line is exactly `cere-worker <event>`.
    pub async fn until_line(&self, event: &str, bound: Duration) {
        let what = format!("cere-worker event line {event:?}");
        until(&what, bound, || self.position(event).map(|_| ())).await;
    }

    /// SIGTERM, then a bounded wait for the exit; returns the exit code.
    pub async fn stop(&mut self) -> Option<i32> {
        let ok = Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status()
            .unwrap()
            .success();
        assert!(ok, "kill -TERM failed");
        let child = &mut self.child;
        until("cere-worker exits on SIGTERM", short_bound(), || {
            child.try_wait().unwrap()
        })
        .await
        .code()
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The per-test knobs of a [`World`].
pub struct Options {
    /// The agent's backend (its port is picked by the world).
    pub backend: MockConfig,
    /// The agent's `limits.output_buffer_bytes`.
    pub output_buffer_bytes: usize,
    /// The agent's `limits.output_stall_ms`.
    pub output_stall_ms: u64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            backend: MockConfig {
                server_info: backend_info(1, 4),
                ..MockConfig::default()
            },
            output_buffer_bytes: 1 << 20,
            output_stall_ms: LOST_MS,
        }
    }
}

/// The extra knobs of [`World::start_managed`].
pub struct Managed {
    /// The coordinator's local worker (`gpu1`).
    pub local: MockConfig,
    /// Extra edits of the coordinator's config text (applied after the world's own).
    pub cered_tweaks: Vec<(&'static str, &'static str)>,
}

impl Default for Managed {
    /// A local worker serving the test model, so it turns Ready.
    fn default() -> Self {
        Managed {
            local: MockConfig::default(),
            cered_tweaks: Vec::new(),
        }
    }
}

/// The remote worker as one `/readyz` reading saw it while Ready.
#[derive(Clone, Copy, Debug)]
pub struct Seen {
    pub epoch: u64,
    pub term: u64,
    pub generation: u64,
}

/// `cered` (coordinator) and `cere-worker` (agent `A`) linked over two relays: `to_agent` carries the
/// coordinator's dial of the agent (`[[federation.workers]] endpoint`), `to_coordinator` the agent's dial of the
/// registry (`[coordinator] endpoint`).
pub struct World {
    pub fed: Federation,
    /// PEMs of a coordinator-principal leaf (the probe acts as the coordinator).
    pub coordinator_pem: (PathBuf, PathBuf),
    /// PEMs of the agent's own leaf (`PRINCIPAL_A`).
    pub worker_pem: (PathBuf, PathBuf),
    pub backend_config: MockConfig,
    /// `None` only while [`World::restart_backend`] swaps it.
    pub backend: Option<MockHandle>,
    pub local: MockHandle,
    pub to_agent: Relay,
    pub to_coordinator: Relay,
    pub agent_addr: SocketAddr,
    pub cered_state: StateDir,
    pub cered_config: ConfigFile,
    pub cered: Cered,
    pub _agent_state: StateDir,
    pub agent_config: ConfigFile,
    pub agent: Agent,
    /// The managed agent's launch directory ([`World::start_managed`]); after `agent`, so a backend the agent
    /// leaves behind is killed when the world drops.
    pub mock_bin: Option<MockBin>,
    pub http: reqwest::Client,
}

impl World {
    /// Starts both binaries and waits until the remote worker is Ready.
    pub async fn start(options: Options) -> World {
        let world = Self::launch(options, None, Managed::default(), false).await;
        world.until_ready().await;
        world
    }

    /// [`World::start`] with the coordinator's local worker (`gpu1`) on `local` instead of a mock serving another
    /// model: given the test model it is a second Ready worker beside `A` (m4 design note P3), with no managed
    /// [`MockBin`]. Waits until both `A` and the local worker are Ready.
    pub async fn start_with_local(options: Options, local: MockConfig) -> World {
        let managed = Managed {
            local,
            cered_tweaks: Vec::new(),
        };
        let world = Self::launch(options, None, managed, true).await;
        world.until_ready().await;
        world.until_local_ready().await;
        world
    }

    /// Starts both binaries with a managed agent (m36 design note Q3): its `[backend.launch]` runs
    /// [`BACKEND_SCRIPT`] in a [`MockBin`], which `exec`s the built mock on the backend port with
    /// `options.backend`'s server info, token latency and natural output length, and its `lock_sha256` is the
    /// coordinator's deployment lock; the local worker runs `managed.local`. Waits until `A` is described at the
    /// agent's epoch with model `unloaded` and the local worker is Ready.
    pub async fn start_managed(options: Options, managed: Managed) -> World {
        let world = Self::launch(options, Some(MockBin::new()), managed, false).await;
        world.until_model("unloaded", ready_bound()).await;
        world.until_local_ready().await;
        world
    }

    /// Without `mock_bin` (attached mode) the local worker serves another model unless `local_serves`;
    /// `extra.local` is its config when managed or `local_serves`.
    async fn launch(
        options: Options,
        mock_bin: Option<MockBin>,
        extra: Managed,
        local_serves: bool,
    ) -> World {
        let fed = Federation::new();
        let coordinator_pem = fed.pki.leaf(FED_COORDINATOR).write(fed.dir.path());
        let worker_pem = fed.pki.leaf(PRINCIPAL_A).write(fed.dir.path());
        let backend_config = MockConfig {
            port: free_port(),
            ..options.backend
        };
        // The port is this process's ([`free_port`]), so attached mode's mock binds it now and `restart_backend`
        // rebinds it; the managed agent launches its backend on it.
        let backend = match mock_bin {
            Some(_) => None,
            None => Some(MockSglang::start(backend_config.clone()).await.unwrap()),
        };
        let backend_url = format!("http://127.0.0.1:{}", backend_config.port);
        let managed = match &mock_bin {
            Some(bin) => managed_backend(bin, &backend_config),
            None => String::new(),
        };
        let local = MockSglang::start(if mock_bin.is_some() || local_serves {
            extra.local
        } else {
            MockConfig {
                server_info: ServerInfo {
                    served_model_name: "another-model".to_owned(),
                    ..ServerInfo::default()
                },
                ..MockConfig::default()
            }
        })
        .await
        .unwrap();
        let agent_addr = SocketAddr::from(([127, 0, 0, 1], free_port()));
        let to_agent = Relay::start(agent_addr).await;
        let to_coordinator = Relay::start(fed.listen).await;

        let cered_state = StateDir::on_tmpfs();
        let mut tweaks = vec![
            FEDERATION_BODY_TWEAK,
            OPERATOR_TWEAK,
            // Orphan holds poll the agent every second (DB3), so a release follows its cleanup quickly.
            (
                "terminal_retention_ms = 3600000",
                "terminal_retention_ms = 3600000\norphan_load_poll_ms = 1000",
            ),
        ];
        tweaks.extend(extra.cered_tweaks);
        let mut text = config_text(&local.base_url(), cered_state.path(), free_port(), &tweaks);
        text.push('\n');
        text.push_str(&fed.toml(
            &format!(
                "heartbeat_interval_ms = {HEARTBEAT_MS}\nadmission_lease_ms = {LEASE_MS}\n\
                 suspect_after_ms = {SUSPECT_MS}\nlost_after_ms = {LOST_MS}\nrpc_timeout_ms = {RPC_TIMEOUT_MS}\n"
            ),
            &[],
        ));
        text.push_str(&format!(
            "[[federation.workers]]\nworker_id = \"{A}\"\nendpoint = \"https://{}\"\n\
             principal = \"{PRINCIPAL_A}\"\nmax_inflight = 4\n\n[federation.workers.expect]\n\
             served_model_name = \"{MODEL}\"\ntp_size = 1\nkv_cache_dtype = \"fp8_e4m3\"\n",
            to_agent.addr()
        ));
        let cered_config = ConfigFile::new(&text);
        let cered = Cered::spawn(&cered_config.path, &[("TOK_OP", Some(OPERATOR_TOKEN))])
            .await
            .unwrap_or_else(|(code, stderr)| panic!("cered exited {code:?}: {stderr}"));

        let agent_state = StateDir::on_tmpfs();
        let agent_config = ConfigFile::new(&format!(
            "cluster_id = \"{FED_CLUSTER}\"\nworker_id = \"{A}\"\nlisten = \"{agent_addr}\"\n\
             principal = \"{PRINCIPAL_A}\"\nca_file = \"{}\"\ncert_file = \"{}\"\nkey_file = \"{}\"\n\
             state_dir = \"{}\"\n\n[coordinator]\nendpoint = \"https://{}\"\nprincipal = \"{FED_COORDINATOR}\"\n\
             coordinator_id = \"{FED_COORDINATOR}\"\n\n[backend]\nbase_url = \"{}\"\nload_endpoint = \"/get_load\"\n\
             served_model_name = \"{MODEL}\"\n{managed}\n[limits]\nheartbeat_interval_ms = {HEARTBEAT_MS}\n\
             lost_after_ms = {LOST_MS}\nrpc_timeout_ms = {RPC_TIMEOUT_MS}\noutput_buffer_bytes = {}\n\
             output_stall_ms = {}\n",
            fed.ca.display(),
            worker_pem.0.display(),
            worker_pem.1.display(),
            agent_state.path().display(),
            to_coordinator.addr(),
            backend_url,
            options.output_buffer_bytes,
            options.output_stall_ms,
        ));
        let agent = Agent::spawn(&agent_config.path).await;
        World {
            fed,
            coordinator_pem,
            worker_pem,
            backend_config,
            backend,
            local,
            to_agent,
            to_coordinator,
            agent_addr,
            cered_state,
            cered_config,
            cered,
            _agent_state: agent_state,
            agent_config,
            agent,
            mock_bin,
            http: http(),
        }
    }

    pub fn backend(&self) -> &MockHandle {
        self.backend.as_ref().unwrap()
    }

    /// The managed agent's launch directory.
    pub fn mock_bin(&self) -> &MockBin {
        self.mock_bin.as_ref().expect("a managed world")
    }

    /// The agent's `state_dir/backend_child.json` (the launched child's pid, pgid and start time).
    pub fn child_file(&self) -> PathBuf {
        self._agent_state.path().join(CHILD_FILE)
    }

    /// Replaces [`BACKEND_SCRIPT`] with one that exits 3 at once: the next launch fails.
    pub fn break_backend(&self) {
        self.mock_bin()
            .write(BACKEND_SCRIPT, "#!/bin/sh\nexit 3\n", 0o755);
    }

    /// Restores the [`BACKEND_SCRIPT`] that `exec`s the mock.
    pub fn restore_backend(&self) {
        let bin = self.mock_bin();
        bin.write(
            BACKEND_SCRIPT,
            &backend_script(bin, &self.backend_config),
            0o755,
        );
    }

    /// The whole `/readyz` body.
    pub async fn readyz(&self) -> Value {
        self.http
            .get(self.cered.url("/readyz"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    /// Waits until the coordinator's local worker (`gpu1`) is Ready.
    pub async fn until_local_ready(&self) {
        let deadline = Instant::now() + ready_bound();
        loop {
            let body = self.readyz().await;
            if body["workers"]["gpu1"]["state"] == "ready" {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "gpu1 not ready within {:?}: {body}",
                ready_bound()
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Waits until `A` was described at the agent's current epoch with model state `state` (m36 design note
    /// Q9 `/readyz` `model`); returns the entry.
    pub async fn until_model(&self, state: &str, bound: Duration) -> Value {
        let epoch = self.agent.epoch;
        self.until_entry(&format!("worker-a model {state}"), bound, |e| {
            e["epoch"] == epoch && e["model"]["state"] == state
        })
        .await
    }

    /// `POST /v1/cere/workers/A/{action}` (`load` or `unload`, m36 design note Q9) with the operator token:
    /// 202 with `{worker_id, desired, intent_seq, model}`; returns the body.
    pub async fn residency(&self, action: &str) -> Value {
        let r = self
            .http
            .post(self.cered.url(&format!("/v1/cere/workers/{A}/{action}")))
            .bearer_auth(OPERATOR_TOKEN)
            .send()
            .await
            .unwrap();
        let status = r.status();
        let body: Value = r.json().await.unwrap();
        assert_eq!(status, 202, "{action}: {body}");
        assert_eq!(body["worker_id"], A, "{body}");
        body
    }

    /// Shuts the agent's backend down and starts a new instance on the same port serving `info`.
    pub async fn restart_backend(&mut self, info: ServerInfo) {
        self.backend.take().unwrap().shutdown().await.unwrap();
        self.backend_config.server_info = info;
        let config = self.backend_config.clone();
        let deadline = Instant::now() + short_bound();
        let fresh = loop {
            match MockSglang::start(config.clone()).await {
                Ok(handle) => break handle,
                Err(e) => assert!(Instant::now() < deadline, "rebind backend: {e}"),
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        self.backend = Some(fresh);
    }

    /// Stops the agent (SIGTERM, exit 0) and starts it again on the same config and state dir: a new epoch.
    pub async fn restart_agent(&mut self) {
        let old = self.agent.epoch;
        assert_eq!(self.agent.stop().await, Some(0), "{}", self.agent.events());
        self.agent = Agent::spawn(&self.agent_config.path).await;
        assert!(self.agent.epoch > old, "{} -> {}", old, self.agent.epoch);
    }

    /// SIGKILLs `cered` (no shutdown path runs) and starts it again on the same config and state dir: a new
    /// process epoch, the coordinator's next term. `http` is replaced: its pool may still hold an idle keep-alive
    /// connection to the killed process whose FIN the client has not processed yet (a loaded host), and a request
    /// written on it is answered RST (`ConnectionReset`, which hyper does not retry: the request was sent), O31.
    pub async fn restart_cered(&mut self) {
        let old = self.cered.epoch;
        self.cered.kill();
        self.http = http();
        self.cered = Cered::spawn(&self.cered_config.path, &[("TOK_OP", Some(OPERATOR_TOKEN))])
            .await
            .unwrap_or_else(|(code, stderr)| panic!("cered exited {code:?}: {stderr}"));
        assert!(self.cered.epoch > old, "{} -> {}", old, self.cered.epoch);
    }

    /// The `/readyz` entry of `A`.
    pub async fn entry(&self) -> Value {
        let body: Value = self
            .http
            .get(self.cered.url("/readyz"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        body["workers"][A].clone()
    }

    /// Polls `/readyz` until `A`'s entry satisfies `done`; returns that entry.
    pub async fn until_entry(
        &self,
        what: &str,
        bound: Duration,
        done: impl Fn(&Value) -> bool,
    ) -> Value {
        let deadline = Instant::now() + bound;
        loop {
            let entry = self.entry().await;
            if done(&entry) {
                return entry;
            }
            assert!(
                Instant::now() < deadline,
                "{what}: not within {bound:?}; last entry {entry}\nagent events:\n{}\ncered remote events:\n{}",
                self.agent.events(),
                self.cered_remote_events()
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// `cered`'s stderr lines about remote workers (`cered remote ...`) so far.
    pub fn cered_remote_events(&self) -> String {
        self.cered
            .stderr_so_far()
            .lines()
            .filter(|l| l.starts_with("cered remote "))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Waits until `A` is Ready with a passed smoke at the agent's current epoch.
    pub async fn until_ready(&self) -> Seen {
        let epoch = self.agent.epoch;
        let entry = self
            .until_entry("worker-a ready", ready_bound(), |e| {
                e["state"] == "ready" && e["epoch"] == epoch
            })
            .await;
        assert_eq!(entry["smoke"]["state"], "passed", "{entry}");
        Seen {
            epoch,
            term: entry["term"].as_u64().unwrap(),
            generation: entry["model_generation"].as_u64().unwrap(),
        }
    }

    /// Waits until every operation the agent accepted fresh has its cleanup recorded (Q12), so its live-op bound
    /// is free.
    pub async fn settled(&self) {
        let agent = &self.agent;
        until("agent operations cleaned up", short_bound(), || {
            (agent.count(" fresh=true") == agent.count("event=cleanup ")).then_some(())
        })
        .await;
    }

    /// `POST /v1/cere/workers/A/{action}` with the operator token.
    pub async fn operate(&self, action: &str) -> Value {
        let r = self
            .http
            .post(self.cered.url(&format!("/v1/cere/workers/{A}/{action}")))
            .bearer_auth(OPERATOR_TOKEN)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "{action}");
        r.json().await.unwrap()
    }

    /// A client chat through `cered`; returns its rid after asserting it completed on `A` at `epoch`.
    pub async fn chat(&self, epoch: u64) -> String {
        let r = self
            .http
            .post(self.cered.url("/v1/chat/completions"))
            .bearer_auth(TOKEN_A)
            .json(&chat_body(false))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        let rid = r.headers()["x-cere-request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let body: Value = r.json().await.unwrap();
        assert!(
            body["choices"][0]["message"]["content"].is_string(),
            "{body}"
        );
        let r = self
            .http
            .get(self.cered.url(&format!("/v1/cere/requests/{rid}")))
            .bearer_auth(TOKEN_A)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        let status: Value = r.json().await.unwrap();
        assert_eq!(status["state"], "completed", "{status}");
        assert_eq!(status["worker_id"], A, "{status}");
        assert_eq!(status["worker_epoch"], epoch, "{status}");
        rid
    }

    /// The probe against the agent presenting `pem` (signed by `ca`'s authority, trusting the cluster CA).
    pub async fn probe_agent(&self, pem: &(PathBuf, PathBuf), call: Vec<String>) -> Vec<Value> {
        let endpoint = format!("https://{}", self.agent_addr);
        probe(&self.fed.ca, pem, &endpoint, PRINCIPAL_A, call).await
    }

    /// The probe against the coordinator's registry presenting `pem`.
    pub async fn probe_registry(&self, pem: &(PathBuf, PathBuf), call: Vec<String>) -> Vec<Value> {
        let endpoint = format!("https://{}", self.fed.listen);
        probe(&self.fed.ca, pem, &endpoint, FED_COORDINATOR, call).await
    }

    /// A probe `submit` as the coordinator at `seen`'s epoch and term with `generation`.
    pub async fn submit(
        &self,
        seen: Seen,
        generation: u64,
        op: &str,
        rid: &str,
        extra: &[&str],
    ) -> Vec<Value> {
        let mut call = vec!["submit".to_owned()];
        call.extend(authority_flags(seen, generation));
        call.extend(args(&[
            "--scope",
            SCOPE,
            "--op-id",
            op,
            "--request-id",
            rid,
        ]));
        call.extend(args(extra));
        self.probe_agent(&self.coordinator_pem, call).await
    }

    /// A probe `cancel` of (`SCOPE`, `rid`, attempt 1) as the coordinator, as op `op`.
    pub async fn cancel(
        &self,
        seen: Seen,
        generation: u64,
        op: &str,
        rid: &str,
        ticket: &str,
    ) -> Value {
        self.cancel_in(SCOPE, seen, generation, op, rid, ticket)
            .await
    }

    /// A probe `cancel` of (`scope`, `rid`, attempt 1) as the coordinator, as op `op`.
    pub async fn cancel_in(
        &self,
        scope: &str,
        seen: Seen,
        generation: u64,
        op: &str,
        rid: &str,
        ticket: &str,
    ) -> Value {
        let mut call = vec!["cancel".to_owned()];
        call.extend(authority_flags(seen, generation));
        call.extend(args(&[
            "--scope",
            scope,
            "--op-id",
            op,
            "--request-id",
            rid,
            "--ticket",
            ticket,
        ]));
        let lines = self.probe_agent(&self.coordinator_pem, call).await;
        assert_eq!(lines.len(), 1, "{lines:?}");
        lines[0].clone()
    }

    /// Backend chat requests that carried `rid`.
    pub fn backend_requests(&self, rid: &str) -> usize {
        self.backend()
            .dispatched_rids()
            .iter()
            .filter(|r| *r == rid)
            .count()
    }

    /// Backend `/abort_request` calls for `rid`.
    pub fn backend_aborts(&self, rid: &str) -> usize {
        self.backend()
            .abort_requests()
            .iter()
            .filter(|a| a.rid == rid)
            .count()
    }

    /// A coordinator-principal channel to the agent (straight, not through the relay) whose per-stream HTTP/2
    /// window is 16 KiB: a `StreamWork` stream nobody reads stays open on the agent once its output exceeds it.
    pub fn narrow_channel(&self) -> Channel {
        let leaf = self.fed.pki.leaf(FED_COORDINATOR);
        Endpoint::from_shared(format!("https://{}", self.agent_addr))
            .unwrap()
            .timeout(Duration::from_millis(PROBE_TIMEOUT_MS))
            .initial_stream_window_size(16 * 1024)
            .tls_config(tls_with(self.fed.pki.ca_pem(), &leaf, PRINCIPAL_A))
            .unwrap()
            .connect_lazy()
    }

    /// A coordinator-principal channel to the agent (straight, not through the relay), connected (TCP, mTLS,
    /// HTTP/2) before it returns: a later call on it is one RPC, with no process start or connection setup, so it
    /// lands inside a window the test opened (fed-load-flake design note F3).
    pub async fn connected_channel(&self) -> Channel {
        let leaf = self.fed.pki.leaf(FED_COORDINATOR);
        Endpoint::from_shared(format!("https://{}", self.agent_addr))
            .unwrap()
            .timeout(Duration::from_millis(PROBE_TIMEOUT_MS))
            .tls_config(tls_with(self.fed.pki.ca_pem(), &leaf, PRINCIPAL_A))
            .unwrap()
            .connect()
            .await
            .expect("the agent accepts the coordinator principal")
    }

    /// `GetOperation` of (`scope`, `op`) as the coordinator at `seen`, straight to the agent (not through the
    /// relay).
    pub async fn operation(&self, seen: Seen, scope: &str, op: &str) -> pb::OperationStatus {
        let leaf = self.fed.pki.leaf(FED_COORDINATOR);
        let channel = Endpoint::from_shared(format!("https://{}", self.agent_addr))
            .unwrap()
            .timeout(Duration::from_millis(PROBE_TIMEOUT_MS))
            .tls_config(tls_with(self.fed.pki.ca_pem(), &leaf, PRINCIPAL_A))
            .unwrap()
            .connect_lazy();
        operation_on(channel, seen, scope, op).await
    }

    /// A streamed client chat through `cered` asking for `max_tokens`, after its `accepted` event and its first
    /// output event: its rid, the reader and the events read so far.
    pub async fn stream_chat(&self, max_tokens: u64) -> (String, SseReader, Vec<Event>) {
        let mut body = chat_body(true);
        body["max_completion_tokens"] = max_tokens.into();
        let r = self
            .http
            .post(self.cered.url("/v1/chat/completions"))
            .bearer_auth(TOKEN_A)
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
        let events = vec![sse.next().await.unwrap(), sse.next().await.unwrap()];
        (rid, sse, events)
    }

    /// `GET /v1/cere/requests/{rid}` as scope `a`.
    pub async fn status(&self, rid: &str) -> Value {
        let r = self
            .http
            .get(self.cered.url(&format!("/v1/cere/requests/{rid}")))
            .bearer_auth(TOKEN_A)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        r.json().await.unwrap()
    }

    /// `POST /v1/cere/requests/{rid}/cancel` as scope `a`: the HTTP status and the body.
    pub async fn client_cancel(&self, rid: &str) -> (u16, Value) {
        let r = self
            .http
            .post(self.cered.url(&format!("/v1/cere/requests/{rid}/cancel")))
            .bearer_auth(TOKEN_A)
            .send()
            .await
            .unwrap();
        (r.status().as_u16(), r.json().await.unwrap())
    }

    /// Scope `a`'s slots on `A` (`/v1/cere/stats` `inflight`): its running requests and held orphans.
    pub async fn inflight(&self) -> u64 {
        let body: Value = self
            .http
            .get(self.cered.url("/v1/cere/stats"))
            .bearer_auth(TOKEN_A)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        body["workers"][A]["inflight"].as_u64().unwrap()
    }

    /// Polls [`World::inflight`] until it is `n`.
    pub async fn until_inflight(&self, n: u64, bound: Duration) {
        let deadline = Instant::now() + bound;
        loop {
            let now = self.inflight().await;
            if now == n {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "scope a slots on {A}: {now}, not {n} within {bound:?}"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// The coordinator journal's `kind` records of `rid`.
    pub fn records(&self, kind: &str, rid: &str) -> Vec<Value> {
        journal(self.cered_state.path())
            .into_iter()
            .filter(|r| r["type"] == kind && r["rid"] == rid)
            .collect()
    }

    /// Scope `a`'s rids the coordinator journaled `admitted` (accepted), in journal order.
    pub fn admitted(&self) -> Vec<String> {
        journal(self.cered_state.path())
            .into_iter()
            .filter(|r| r["type"] == "admitted" && r["scope"] == "a")
            .map(|r| r["rid"].as_str().unwrap().to_owned())
            .collect()
    }

    /// For a burst of `burst` scope `a` requests admitted after the rids `before`, sent while the backend holds
    /// every generation at its finish ([`MockHandle::hold_finish`]): waits until `held` of them are held there at
    /// once and every other one is journaled without a `dispatched` record and is `queued`; returns those rids.
    pub async fn until_held_and_queued(
        &self,
        before: &[String],
        burst: usize,
        held: usize,
        bound: Duration,
    ) -> Vec<String> {
        let deadline = Instant::now() + bound;
        loop {
            let records = journal(self.cered_state.path());
            let dispatched: Vec<&str> = records
                .iter()
                .filter(|r| r["type"] == "dispatched")
                .filter_map(|r| r["rid"].as_str())
                .collect();
            let admitted: Vec<&str> = records
                .iter()
                .filter(|r| r["type"] == "admitted" && r["scope"] == "a")
                .filter_map(|r| r["rid"].as_str())
                .filter(|rid| !before.iter().any(|b| b == rid))
                .collect();
            let waiting: Vec<String> = admitted
                .iter()
                .filter(|rid| !dispatched.contains(rid))
                .map(|rid| rid.to_string())
                .collect();
            let on_backend = self.backend().finish_held();
            if admitted.len() == burst && on_backend == held && waiting.len() == burst - held {
                let mut states = Vec::new();
                for rid in &waiting {
                    states.push(self.status(rid).await["state"].clone());
                }
                if states.iter().all(|s| s == "queued") {
                    return waiting;
                }
            }
            assert!(
                Instant::now() < deadline,
                "{held} of {burst} held on the backend and the rest queued: not within {bound:?} \
                 (held {on_backend}, admitted {admitted:?}, not dispatched {waiting:?})"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Waits until the coordinator journaled the release of `rid`'s held slot; returns the release records.
    pub async fn until_released(&self, rid: &str, bound: Duration) -> Vec<Value> {
        until(&format!("the slot of {rid} released"), bound, || {
            let released = self.records("orphan_released", rid);
            (!released.is_empty()).then_some(released)
        })
        .await
    }
}

pub fn args(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

/// The probe's worker-service authority flags for `A` at `seen`'s epoch and term.
pub fn authority_flags(seen: Seen, generation: u64) -> Vec<String> {
    args(&[
        "--cluster-id",
        FED_CLUSTER,
        "--worker-id",
        A,
        "--epoch",
        &seen.epoch.to_string(),
        "--coordinator-id",
        FED_COORDINATOR,
        "--term",
        &seen.term.to_string(),
        "--generation",
        &generation.to_string(),
    ])
}

/// Runs `cere-rpc-probe` (trusting `ca`, presenting `pem`, expecting `peer` at `endpoint`) to completion; every
/// call must have printed its line (exit 0). Returns the lines.
pub async fn probe(
    ca: &Path,
    pem: &(PathBuf, PathBuf),
    endpoint: &str,
    peer: &str,
    call: Vec<String>,
) -> Vec<Value> {
    let mut argv = vec![
        "--ca".to_owned(),
        ca.display().to_string(),
        "--cert".to_owned(),
        pem.0.display().to_string(),
        "--key".to_owned(),
        pem.1.display().to_string(),
        "--endpoint".to_owned(),
        endpoint.to_owned(),
        "--peer".to_owned(),
        peer.to_owned(),
        "--timeout-ms".to_owned(),
        PROBE_TIMEOUT_MS.to_string(),
    ];
    argv.extend(call.iter().cloned());
    let out = tokio::task::spawn_blocking(move || {
        Command::new(env!("CARGO_BIN_EXE_cere-rpc-probe"))
            .args(&argv)
            .output()
            .unwrap()
    })
    .await
    .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "probe {call:?}: {stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// `line` is the `call` outcome `code`/`reason`.
pub fn assert_outcome(line: &Value, call: &str, code: &str, reason: &str) {
    assert_eq!(
        (
            line["call"].as_str(),
            line["code"].as_str(),
            line["reason"].as_str()
        ),
        (Some(call), Some(code), Some(reason)),
        "{line}"
    );
}

/// `line` is a refusal `code`/`reason` that answered at once.
pub fn assert_prompt_refusal(line: &Value, call: &str, code: &str, reason: &str) {
    assert_outcome(line, call, code, reason);
    assert_eq!(line["accepted"], false, "{line}");
    assert!(
        line["ms"].as_u64().unwrap() < PROMPT_REFUSAL_MS,
        "a refusal must not wait: {line}"
    );
}

/// `line` is an accepted receipt; returns its ticket.
pub fn accepted_ticket(line: &Value, call: &str) -> String {
    assert_outcome(line, call, "OK", "");
    assert_eq!(line["accepted"], true, "{line}");
    let ticket = line["ticket"].as_str().unwrap().to_owned();
    assert!(!ticket.is_empty(), "{line}");
    ticket
}

/// The coordinator authority of a `StreamWork`/`SubmitWork` for (`SCOPE`, `op`) at `seen`.
pub fn op_authority(seen: Seen, op: &str, digest: [u8; 32]) -> pb::Authority {
    scoped_authority(seen, SCOPE, op, digest)
}

/// The coordinator authority of an operation (`scope`, `op`) at `seen`.
pub fn scoped_authority(seen: Seen, scope: &str, op: &str, digest: [u8; 32]) -> pb::Authority {
    pb::Authority {
        target: Some(pb::WorkerIdentity {
            cluster_id: FED_CLUSTER.to_owned(),
            worker_id: A.to_owned(),
            worker_epoch: seen.epoch,
        }),
        coordinator_id: FED_COORDINATOR.to_owned(),
        coordinator_term: seen.term,
        scope_id: scope.to_owned(),
        operation_id: op.to_owned(),
        payload_sha256: digest.to_vec(),
        remaining_budget_ms: 60_000,
        model_generation: Some(seen.generation),
    }
}

/// A chat SubmitWork for (`SCOPE`, `op`, `rid`) asking for `max_tokens`, streamed or unary.
pub fn chat_submit(
    seen: Seen,
    op: &str,
    rid: &str,
    max_tokens: u64,
    stream: bool,
) -> pb::SubmitWorkRequest {
    let unit = serde_json::to_vec(&json!({"kind": "chat", "stream": stream})).unwrap();
    let input = serde_json::to_vec(&json!({
        "rid": rid,
        "messages": [{"role": "user", "content": "fill the output buffer"}],
        "max_completion_tokens": max_tokens,
        "stream": stream,
    }))
    .unwrap();
    let digest = wire::payload_sha256(wire::METHOD_SUBMIT, SCOPE, rid, 1, &unit, &input);
    pb::SubmitWorkRequest {
        authority: Some(op_authority(seen, op, digest)),
        request_id: rid.to_owned(),
        attempt: 1,
        work_unit_json: unit,
        bounded_inference_input_json: input,
    }
}

/// `GetOperation` of (`scope`, `op`) as the coordinator at `seen` on `channel`.
pub async fn operation_on(
    channel: Channel,
    seen: Seen,
    scope: &str,
    op: &str,
) -> pb::OperationStatus {
    WorkerServiceClient::new(channel)
        .get_operation(pb::OperationQuery {
            authority: Some(scoped_authority(seen, scope, op, [0; 32])),
            queried_operation_id: op.to_owned(),
        })
        .await
        .unwrap_or_else(|s| panic!("GetOperation {scope}/{op}: {s:?}"))
        .into_inner()
}
