//! The `cere-worker` node agent config (`schemas/worker_agent_config.schema.json`; m35 design note Q16, Q12, Q14,
//! harness revision H1). Loading is strict like the runtime config: serde rejects unknown keys, then every schema
//! constraint and every semantic rule of `scripts/cere_contracts.py` is checked here; the shared case table
//! `tests/fixtures/worker_agent_config_invalid.json` keeps both verdicts identical. Validation is shape-only: the PEMs
//! are never opened, so `cere-worker --check-config` needs no key material. Managed mode (m36 design note Q3, Q13)
//! adds `[backend] lock_sha256` and `[backend.launch]`; `--check-config` checks their shape only (argv[0] must be
//! absolute, its existence is checked at start).
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer};

use crate::config::{
    absolute_path, chars, find_checkout, id, in_checkout, listen_literal, load_endpoint_path,
    parse_loopback_base_url, parse_tailnet_endpoint, pem_paths, principal, range,
    served_model_name, ConfigError,
};

/// The agent's operation journal (Q12).
pub const DEFAULT_MAX_OPERATIONS: usize = 4096;
/// The tombstone horizon; at least the 600 s work cap (Q12).
pub const DEFAULT_RETRY_HORIZON_MS: u64 = 600_000;
/// `dev-tailnet.yaml` `control_plane` values (Q7) and the Q14 control quota and output buffer.
pub const DEFAULT_HEARTBEAT_INTERVAL_MS: u64 = 1000;
pub const DEFAULT_LOST_AFTER_MS: u64 = 15_000;
pub const DEFAULT_RPC_TIMEOUT_MS: u64 = 5000;
pub const DEFAULT_CONTROL_QUOTA: usize = 64;
pub const DEFAULT_OUTPUT_BUFFER_BYTES: usize = 1 << 20;
/// `[backend.launch]` defaults (m36 design note Q3).
pub const DEFAULT_STARTUP_TIMEOUT_MS: u64 = 600_000;
pub const DEFAULT_STOP_TIMEOUT_MS: u64 = 30_000;
/// `[backend.launch]` bounds (Q3): argv and env entry counts, and the byte cap of an argv entry or env value.
const MAX_LAUNCH_ITEMS: usize = 64;
const MAX_LAUNCH_BYTES: usize = 4096;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentConfig {
    pub cluster_id: String,
    pub worker_id: String,
    /// A canonical `<ip>:<port>` literal: loopback, 100.64.0.0/10 or RFC1918 (a bridge container), never a wildcard.
    #[serde(deserialize_with = "agent_listen")]
    pub listen: SocketAddr,
    /// The worker's mTLS principal (the only dNSName SAN of its leaf, Q2).
    pub principal: String,
    pub ca_file: PathBuf,
    pub cert_file: PathBuf,
    pub key_file: PathBuf,
    /// Durable agent state (journal, epoch counter); outside any checkout.
    pub state_dir: PathBuf,
    pub coordinator: AgentCoordinator,
    pub backend: AgentBackend,
    #[serde(default)]
    pub journal: AgentJournal,
    #[serde(default)]
    pub limits: AgentLimits,
    /// Team pools (net-teleport design note 4.4, C19): the person scope owning this worker; with it the agent
    /// refuses other scopes' SubmitWork while owner-only (an `owner_only` window or `cere-worker owner-only on`).
    /// Absent -> unowned (never owner-only).
    #[serde(default)]
    pub owner: Option<String>,
    /// Owner-only windows, e.g. `"Mon-Fri 09:00-17:00 Europe/Berlin"` ([`crate::pool::Window`]); needs `owner`.
    #[serde(default)]
    pub owner_only: Option<Vec<String>>,
}

impl AgentConfig {
    /// The parsed `owner_only` windows (validated at load).
    pub fn owner_only_windows(&self) -> Vec<crate::pool::Window> {
        self.owner_only
            .iter()
            .flatten()
            .map(|w| crate::pool::Window::parse(w).expect("owner_only validated at load"))
            .collect()
    }
}

fn agent_listen<'de, D: Deserializer<'de>>(d: D) -> Result<SocketAddr, D::Error> {
    listen_literal(d, "listen", true)
}

/// The coordinator the agent registers with.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentCoordinator {
    /// `https://<literal 100.64.0.0/10, 127.0.0.0/8 or [::1]>:<port>`.
    pub endpoint: String,
    /// The coordinator's mTLS principal; never the agent's own.
    pub principal: String,
    pub coordinator_id: String,
}

impl AgentCoordinator {
    /// The validated endpoint as a socket address.
    pub fn endpoint_addr(&self) -> SocketAddr {
        parse_tailnet_endpoint("coordinator.endpoint", &self.endpoint)
            .expect("endpoint validated at load")
    }
}

/// The agent's one loopback backend: attached (no `launch`) or managed (m36 design note Q3).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentBackend {
    /// `http://<127.0.0.0/8 literal or [::1]>:<port>` (the runtime config's loopback rule).
    pub base_url: String,
    #[serde(default)]
    pub load_endpoint: Option<String>,
    pub served_model_name: String,
    /// The approved deployment lock (64 lowercase hex); set exactly when `launch` is (attached mode has no lock, Q7).
    #[serde(default)]
    pub lock_sha256: Option<String>,
    /// `[backend.launch]`: the backend the agent spawns and supervises (Q3).
    #[serde(default)]
    pub launch: Option<AgentLaunch>,
}

impl AgentBackend {
    /// The approved lock as raw bytes; `Some` iff `lock_sha256` is set (validated as 64 lowercase hex).
    pub fn lock_bytes(&self) -> Option<[u8; 32]> {
        let hex = self.lock_sha256.as_deref()?.as_bytes();
        let nibble = |c: u8| match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            _ => None,
        };
        if hex.len() != 64 {
            return None;
        }
        let mut out = [0u8; 32];
        for (byte, pair) in out.iter_mut().zip(hex.chunks_exact(2)) {
            *byte = (nibble(pair[0])? << 4) | nibble(pair[1])?;
        }
        Some(out)
    }
}

/// `[backend.launch]` (m36 design note Q3): argv, extra environment (the child's PATH is fixed), timeouts and the
/// credentials the child runs as.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentLaunch {
    pub argv: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default = "default_startup_timeout_ms")]
    pub startup_timeout_ms: u64,
    #[serde(default = "default_stop_timeout_ms")]
    pub stop_timeout_ms: u64,
    #[serde(default)]
    pub uid: Option<u32>,
    #[serde(default)]
    pub gid: Option<u32>,
}

fn default_startup_timeout_ms() -> u64 {
    DEFAULT_STARTUP_TIMEOUT_MS
}

fn default_stop_timeout_ms() -> u64 {
    DEFAULT_STOP_TIMEOUT_MS
}

impl AgentLaunch {
    /// P2/Q13: `--check-config` checks only that `argv[0]` is absolute; the agent start checks that it names a
    /// regular file with an execute bit (symlinks followed), so a typo fails at start, not at the first load.
    /// Whether the dropped `uid`/`gid` can execute it is only known at spawn (`model_load_failed`).
    pub fn check_program(&self) -> Result<(), ConfigError> {
        use std::os::unix::fs::PermissionsExt;
        let program = &self.argv[0];
        match std::fs::metadata(program) {
            Ok(meta) if meta.is_file() && meta.permissions().mode() & 0o111 != 0 => Ok(()),
            Ok(_) => Err(ConfigError::new(format!(
                "backend.launch.argv[0] {program:?} is not an executable regular file"
            ))),
            Err(e) => Err(ConfigError::new(format!(
                "backend.launch.argv[0] {program:?} cannot be read: {e}"
            ))),
        }
    }

    /// Q3: an agent running as root (`euid` 0) must drop the child to an explicit `uid` and `gid`; any other agent
    /// must set neither (it cannot drop privileges: the child's `setgroups` would fail, so every spawn would). Not
    /// part of `--check-config` (shape only); the agent checks it at start.
    pub fn check_privileges(&self, euid: u32) -> Result<(), ConfigError> {
        if euid == 0 && (self.uid.is_none() || self.gid.is_none()) {
            return Err(ConfigError::new(
                "backend.launch.uid and backend.launch.gid are required when cere-worker runs as root",
            ));
        }
        if euid != 0 && (self.uid.is_some() || self.gid.is_some()) {
            return Err(ConfigError::new(
                "backend.launch.uid and backend.launch.gid must be unset when cere-worker does not run as root",
            ));
        }
        Ok(())
    }

    /// The Q3 shape rules (`schemas/worker_agent_config.schema.json` `backend.launch`).
    fn validate(&self) -> Result<(), ConfigError> {
        if !(1..=MAX_LAUNCH_ITEMS).contains(&self.argv.len()) {
            return Err(ConfigError::new(format!(
                "backend.launch.argv item count must be 1..{MAX_LAUNCH_ITEMS}"
            )));
        }
        if self.argv.iter().any(|a| a.is_empty() || !bounded_bytes(a)) {
            return Err(ConfigError::new(format!(
                "backend.launch.argv entries must be 1..{MAX_LAUNCH_BYTES} bytes without NUL"
            )));
        }
        if !self.argv[0].starts_with('/') {
            return Err(ConfigError::new(
                "backend.launch.argv[0] must be an absolute path",
            ));
        }
        if self.env.len() > MAX_LAUNCH_ITEMS {
            return Err(ConfigError::new(format!(
                "backend.launch.env must have at most {MAX_LAUNCH_ITEMS} entries"
            )));
        }
        for (key, value) in &self.env {
            if !env_key(key) {
                return Err(ConfigError::new(
                    "backend.launch.env key does not match its pattern",
                ));
            }
            if key == "PATH" {
                return Err(ConfigError::new("backend.launch.env must not set PATH"));
            }
            if !bounded_bytes(value) {
                return Err(ConfigError::new(format!(
                    "backend.launch.env values must be at most {MAX_LAUNCH_BYTES} bytes without NUL"
                )));
            }
        }
        range(
            "backend.launch.startup_timeout_ms",
            self.startup_timeout_ms,
            1000,
            3_600_000,
        )?;
        range(
            "backend.launch.stop_timeout_ms",
            self.stop_timeout_ms,
            100,
            600_000,
        )?;
        // 0 is root and u32::MAX the "no change" sentinel of setresuid/setresgid.
        for (field, id) in [
            ("backend.launch.uid", self.uid),
            ("backend.launch.gid", self.gid),
        ] {
            if let Some(id) = id {
                range(field, u64::from(id), 1, u64::from(u32::MAX - 1))?;
            }
        }
        if self.uid.is_some() != self.gid.is_some() {
            return Err(ConfigError::new(
                "backend.launch.uid and backend.launch.gid must be set together",
            ));
        }
        Ok(())
    }
}

/// At most `MAX_LAUNCH_BYTES` bytes and no NUL (an argv entry or env value handed to `execve`).
fn bounded_bytes(value: &str) -> bool {
    value.len() <= MAX_LAUNCH_BYTES && !value.contains('\0')
}

/// `^[A-Za-z_][A-Za-z0-9_]{0,127}$`.
fn env_key(key: &str) -> bool {
    let bytes = key.as_bytes();
    matches!(bytes.first(), Some(c) if c.is_ascii_alphabetic() || *c == b'_')
        && bytes.len() <= 128
        && bytes
            .iter()
            .all(|c| c.is_ascii_alphanumeric() || *c == b'_')
}

/// `[journal]` (Q12).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AgentJournal {
    pub max_operations: usize,
    pub retry_horizon_ms: u64,
}

impl Default for AgentJournal {
    fn default() -> Self {
        Self {
            max_operations: DEFAULT_MAX_OPERATIONS,
            retry_horizon_ms: DEFAULT_RETRY_HORIZON_MS,
        }
    }
}

/// `[limits]` (Q7, Q14). No stream-pool key: the `StreamWork` pool is the live-op bound (the backend's
/// `max_running_requests`) + 1, learned at run time (harness revision H1).
#[derive(Debug, Clone, Deserialize)]
#[serde(from = "RawLimits")]
pub struct AgentLimits {
    pub heartbeat_interval_ms: u64,
    pub lost_after_ms: u64,
    pub rpc_timeout_ms: u64,
    /// Per-principal pool of unary control calls.
    pub control_quota: usize,
    /// Per-operation output buffer, at most 1 MiB.
    pub output_buffer_bytes: usize,
    /// Undrained `StreamWork` output cancels the operation `client_backpressure` after this long; defaults to
    /// `rpc_timeout_ms`.
    pub output_stall_ms: u64,
}

impl Default for AgentLimits {
    fn default() -> Self {
        RawLimits::default().into()
    }
}

/// `[limits]` as written: `output_stall_ms` defaults to the effective `rpc_timeout_ms`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields, default)]
struct RawLimits {
    heartbeat_interval_ms: u64,
    lost_after_ms: u64,
    rpc_timeout_ms: u64,
    control_quota: usize,
    output_buffer_bytes: usize,
    output_stall_ms: Option<u64>,
}

impl Default for RawLimits {
    fn default() -> Self {
        Self {
            heartbeat_interval_ms: DEFAULT_HEARTBEAT_INTERVAL_MS,
            lost_after_ms: DEFAULT_LOST_AFTER_MS,
            rpc_timeout_ms: DEFAULT_RPC_TIMEOUT_MS,
            control_quota: DEFAULT_CONTROL_QUOTA,
            output_buffer_bytes: DEFAULT_OUTPUT_BUFFER_BYTES,
            output_stall_ms: None,
        }
    }
}

impl From<RawLimits> for AgentLimits {
    fn from(raw: RawLimits) -> Self {
        Self {
            heartbeat_interval_ms: raw.heartbeat_interval_ms,
            lost_after_ms: raw.lost_after_ms,
            rpc_timeout_ms: raw.rpc_timeout_ms,
            control_quota: raw.control_quota,
            output_buffer_bytes: raw.output_buffer_bytes,
            output_stall_ms: raw.output_stall_ms.unwrap_or(raw.rpc_timeout_ms),
        }
    }
}

impl AgentConfig {
    /// Reads and validates the agent config at `path`. The agent needs no checkout: the repository bound on
    /// `state_dir` and the PEM paths applies only when an ancestor of the config holds `PACKAGE_VERSION`.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::new(format!("cannot read {}: {e}", path.display())))?;
        Self::from_toml_str(&text, find_checkout(path)?.as_deref())
    }

    /// Parses and validates TOML text; `checkout` (if any) bounds `state_dir` and the PEM paths.
    pub fn from_toml_str(text: &str, checkout: Option<&Path>) -> Result<Self, ConfigError> {
        let config: AgentConfig =
            toml::from_str(text).map_err(|e| ConfigError::new(e.to_string().trim_end()))?;
        config.validate(checkout)?;
        Ok(config)
    }

    /// Every schema constraint, then the semantic rules (Q16, Q12, Q14, H1). Shape only: no file is opened.
    pub fn validate(&self, checkout: Option<&Path>) -> Result<(), ConfigError> {
        id("cluster_id", &self.cluster_id)?;
        id("worker_id", &self.worker_id)?;
        principal("principal", &self.principal)?;
        for (field, path) in [
            ("ca_file", &self.ca_file),
            ("cert_file", &self.cert_file),
            ("key_file", &self.key_file),
            ("state_dir", &self.state_dir),
        ] {
            absolute_path(field, path)?;
        }
        let c = &self.coordinator;
        chars("coordinator.endpoint", &c.endpoint, 1, 128)?;
        principal("coordinator.principal", &c.principal)?;
        id("coordinator.coordinator_id", &c.coordinator_id)?;
        let b = &self.backend;
        parse_loopback_base_url(&b.base_url)?;
        if let Some(path) = &b.load_endpoint {
            load_endpoint_path("backend.load_endpoint", path)?;
        }
        served_model_name("backend.served_model_name", &b.served_model_name)?;
        match (&b.lock_sha256, &b.launch) {
            (Some(_), Some(launch)) => {
                if b.lock_bytes().is_none() {
                    return Err(ConfigError::new(
                        "backend.lock_sha256 must be 64 lowercase hex",
                    ));
                }
                launch.validate()?;
            }
            (None, Some(_)) => {
                return Err(ConfigError::new(
                    "backend.launch requires backend.lock_sha256",
                ))
            }
            (Some(_), None) => {
                return Err(ConfigError::new(
                    "backend.lock_sha256 requires backend.launch",
                ))
            }
            (None, None) => {}
        }
        let j = &self.journal;
        range(
            "journal.max_operations",
            j.max_operations as u64,
            1,
            1_000_000,
        )?;
        range(
            "journal.retry_horizon_ms",
            j.retry_horizon_ms,
            600_000,
            86_400_000,
        )?;
        let l = &self.limits;
        range(
            "limits.heartbeat_interval_ms",
            l.heartbeat_interval_ms,
            100,
            600_000,
        )?;
        range("limits.lost_after_ms", l.lost_after_ms, 100, 3_600_000)?;
        range("limits.rpc_timeout_ms", l.rpc_timeout_ms, 100, 600_000)?;
        range("limits.control_quota", l.control_quota as u64, 1, 4096)?;
        range(
            "limits.output_buffer_bytes",
            l.output_buffer_bytes as u64,
            1,
            1 << 20,
        )?;
        range("limits.output_stall_ms", l.output_stall_ms, 1, 3_600_000)?;
        parse_tailnet_endpoint("coordinator.endpoint", &c.endpoint)?;
        if self.principal == c.principal {
            return Err(ConfigError::new("principal equals coordinator.principal"));
        }
        if l.heartbeat_interval_ms >= l.lost_after_ms {
            return Err(ConfigError::new(
                "limits.heartbeat_interval_ms must be below limits.lost_after_ms",
            ));
        }
        if l.output_stall_ms > l.lost_after_ms {
            return Err(ConfigError::new(
                "limits.output_stall_ms must not exceed limits.lost_after_ms",
            ));
        }
        if checkout.is_some_and(|root| in_checkout(&self.state_dir, root)) {
            return Err(ConfigError::new(
                "state_dir must be outside any repository checkout",
            ));
        }
        if let Some(owner) = &self.owner {
            if !crate::pool::is_scope_id(owner) {
                return Err(ConfigError::new("owner does not match its pattern"));
            }
            if owner == crate::readiness::REMOTE_SMOKE_SCOPE {
                return Err(ConfigError::new(
                    "owner must not be the remote readiness smoke scope",
                ));
            }
        }
        if let Some(windows) = &self.owner_only {
            if self.owner.is_none() {
                return Err(ConfigError::new("owner_only needs an owner"));
            }
            if !(1..=crate::pool::MAX_WINDOWS).contains(&windows.len()) {
                return Err(ConfigError::new(format!(
                    "owner_only item count must be 1..{}",
                    crate::pool::MAX_WINDOWS
                )));
            }
            for text in windows {
                crate::pool::Window::parse(text).map_err(ConfigError::new)?;
            }
        }
        pem_paths(
            "",
            [&self.ca_file, &self.cert_file, &self.key_file],
            checkout,
            &self.state_dir,
        )
    }
}
