//! Shared harness of the `server_*` integration tests: an in-process `cered` (App + router on an ephemeral
//! port) in front of `cere_mock_sglang`, plus a config writer for tests that spawn the real binary.
#![allow(dead_code)]

pub mod pki;
// Owner-registered for Wave A2 (build-loop-m35 design note): AGENT owns stub_coordinator, CREG/CREM own stub_agent.
pub mod stub_agent;
pub mod stub_coordinator;
// Owner-registered for Wave A3 (build-loop-m35 design note): MATRIX owns relay.
pub mod relay;
// Track MATRIX (m36 design note P8): the two-node world of federation.rs, shared with placement.rs, and the
// built mock a managed agent launches.
pub mod exec_file;
pub mod fed_world;
pub mod mock_bin;
pub mod ports;

pub use ports::free_port;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cere_mock_sglang::{MockConfig, MockHandle, MockSglang, ServerInfo};
use cere_runtime::app::{App, RetentionClock, Tuning};
use cere_runtime::config::{ConfigSource, RuntimeConfig};
use cere_runtime::server;
use cere_runtime::sglang::Timeouts;
use serde_json::{json, Value};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

pub const MODEL: &str = "Qwen3.8-Flash-Next";
/// Deployment B of the M5 multi-deployment tests ([`multimodel_config_text`]).
pub const MODEL_B: &str = "gemma-4-e4b-it";
pub const TOKEN_A: &str = "secret-a";
pub const TOKEN_B: &str = "secret-b";
/// The operator bearer token of the drain routes (m35 design note Q13), resolved from `TOK_OP`; enabled by
/// [`OPERATOR_TWEAK`].
pub const OPERATOR_TOKEN: &str = "secret-operator";
pub const OPERATOR_TWEAK: (&str, &str) = (
    "[[auth.scopes]]\nscope_id = \"a\"",
    "[auth]\noperator_token_env = \"TOK_OP\"\n\n[[auth.scopes]]\nscope_id = \"a\"",
);

/// The live profile's 8 MiB request body bound lowered to `config::REMOTE_BODY_MAX`, which `[federation]` requires;
/// every federated config text gets it ([`Harness::start_federated`] and kin apply it themselves).
pub const FEDERATION_BODY_TWEAK: (&str, &str) = (
    "max_request_body_bytes = 8388608",
    "max_request_body_bytes = 8323072",
);

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// A fresh absolute state directory outside the repository, removed on drop.
pub struct StateDir(pub PathBuf);

impl StateDir {
    pub fn new() -> Self {
        Self::under(&std::env::temp_dir())
    }

    /// A fresh state directory on a tmpfs: `/dev/shm` when it is a writable one (Linux; a Docker container's
    /// 64 MiB default), else the temp dir of [`StateDir::new`]. For a process whose journal writes sit on its
    /// liveness path: every record is synced, and on a disk shared with other writers (the CI container's overlay
    /// `/tmp` on the host disk) one sync can take seconds, longer than the shrunk Q7 timings of [`fed_world`] or
    /// the wall-clock retention and aging bounds of an in-process [`Harness`] (whose state dirs live here).
    pub fn on_tmpfs() -> Self {
        let shm = Path::new("/dev/shm");
        if is_writable_tmpfs(shm) {
            Self::under(shm)
        } else {
            Self::new()
        }
    }

    fn under(base: &Path) -> Self {
        sweep_dead_once(base);
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        Self(base.join(format!(
            "cered-test-{}-{}-{nanos}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

/// `path` is a directory on a tmpfs that this process may create entries in.
fn is_writable_tmpfs(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c_path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    let mut fs = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: statfs(2) and access(2) on a NUL-terminated path; `fs` is read only after statfs filled it (0).
    unsafe {
        libc::statfs(c_path.as_ptr(), fs.as_mut_ptr()) == 0
            && fs.assume_init().f_type == libc::TMPFS_MAGIC
            && libc::access(c_path.as_ptr(), libc::W_OK | libc::X_OK) == 0
    }
}

impl Drop for StateDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// [`sweep_dead`] of `base`, once per process and base.
fn sweep_dead_once(base: &Path) {
    static SWEPT: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
    let mut swept = SWEPT.lock().unwrap_or_else(|e| e.into_inner());
    if !swept.iter().any(|p| p == base) {
        swept.push(base.to_path_buf());
        sweep_dead(base);
    }
}

/// Removes the `cered-test-<pid>-*` entries of `base` whose pid names no process. [`StateDir`] and [`ConfigFile`]
/// remove their directory on drop, which a test process killed by a signal (a `timeout`, a stress loop's kill, an
/// OOM kill) never runs: its directories stay in `/dev/shm`, the temp dir and the checkout, and `integrate.py
/// precheck` refuses on them. The next test process of the host removes them. A live pid (a test still running, or
/// a reused pid) keeps its entries; an entry this process cannot remove (another user's) is left.
pub fn sweep_dead(base: &Path) -> Vec<PathBuf> {
    let mut removed = Vec::new();
    let Ok(entries) = std::fs::read_dir(base) else {
        return removed;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name
            .to_str()
            .and_then(|n| n.strip_prefix("cered-test-"))
            .and_then(|rest| rest.split_once('-'))
            .and_then(|(pid, _)| pid.parse::<libc::pid_t>().ok())
        else {
            continue;
        };
        if pid <= 0 || pid_alive(pid) {
            continue;
        }
        let path = entry.path();
        let gone = if path.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        if gone.is_ok() {
            removed.push(path);
        }
    }
    removed
}

/// `kill(pid, 0)` finds the process (EPERM: another user's, alive).
fn pid_alive(pid: libc::pid_t) -> bool {
    // SAFETY: signal 0 only checks that `pid` exists and may be signalled; nothing is sent.
    let rc = unsafe { libc::kill(pid, 0) };
    rc == 0 || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

#[test]
fn sweep_dead_removes_only_the_entries_of_dead_pids() {
    let base = StateDir::new();
    std::fs::create_dir_all(base.path()).unwrap();
    let mut child = std::process::Command::new("true").spawn().unwrap();
    let dead = child.id();
    child.wait().unwrap(); // reaped: its pid names no process
    let me = std::process::id();
    let names = [
        format!("cered-test-{dead}-0-123"),
        format!("cered-test-{dead}-1.cere"),
        format!("cered-test-{me}-0-123"),
        "cered-test-abc".to_owned(),
        format!("other-{dead}-0"),
    ];
    for name in &names {
        std::fs::create_dir_all(base.path().join(name).join("sub")).unwrap();
    }
    let mut removed = sweep_dead(base.path());
    removed.sort();
    assert_eq!(
        removed,
        vec![base.path().join(&names[0]), base.path().join(&names[1])]
    );
    for (name, kept) in names.iter().zip([false, false, true, true, true]) {
        assert_eq!(base.path().join(name).exists(), kept, "{name}");
    }
}

/// The live config pointed at `base_url` and `state_dir`, with two scopes (`a` = TOK_A, `b` = TOK_B), then
/// `tweak` applied to the TOML text (each tweak must change something).
pub fn config_text(base_url: &str, state_dir: &Path, port: u16, tweaks: &[(&str, &str)]) -> String {
    render_config(
        &[(
            "base_url = \"http://127.0.0.1:18080\"",
            &format!("base_url = \"{base_url}\""),
        )],
        state_dir,
        port,
        tweaks,
    )
}

/// [`config_text`] in the `[[deployments.replicas]]` form (m3 design note Q2): replica `i` is device `gpu<i>` at
/// `base_urls[i]`, `scheduler.devices` lists exactly those devices and `deployments[0].base_url` is gone.
pub fn replicas_config_text(
    base_urls: &[String],
    state_dir: &Path,
    port: u16,
    tweaks: &[(&str, &str)],
) -> String {
    let devices: Vec<String> = (0..base_urls.len()).map(|i| format!("gpu{i}")).collect();
    let mut tables = String::new();
    for (device, base_url) in devices.iter().zip(base_urls) {
        tables.push_str(&format!(
            "[[deployments.replicas]]\ndevice = \"{device}\"\nbase_url = \"{base_url}\"\n\n"
        ));
    }
    let devices = devices
        .iter()
        .map(|device| format!("\"{device}\""))
        .collect::<Vec<_>>()
        .join(", ");
    render_config(
        &[
            ("base_url = \"http://127.0.0.1:18080\"\n", ""),
            ("devices = [\"gpu1\"]", &format!("devices = [{devices}]")),
            ("[model_management]", &format!("{tables}[model_management]")),
        ],
        state_dir,
        port,
        tweaks,
    )
}

/// The second `[[deployments]]` entry of `configs/runtime/multimodel-dev.toml` (model [`MODEL_B`]) without its
/// replicas, with the test configs' tenancy and prefix-cache flags (equal across entries, D2(g)).
fn entry_b() -> toml::Table {
    let text =
        std::fs::read_to_string(repo_root().join("configs/runtime/multimodel-dev.toml")).unwrap();
    let doc: toml::Table = text.parse().unwrap();
    let mut entry = doc["deployments"][1].as_table().unwrap().clone();
    assert_eq!(entry["served_model_name"].as_str(), Some(MODEL_B));
    entry.remove("replicas");
    let a: toml::Table = config_text("http://127.0.0.1:1", Path::new("/state"), 1, &[])
        .parse()
        .unwrap();
    for key in ["shared_backend_tenancy", "allow_cross_scope_prefix_cache"] {
        entry.insert(key.to_owned(), a["deployments"][0][key].clone());
    }
    entry
}

/// [`MODEL_B`]'s `[[deployments]]` entry as TOML text with no local replica (a remote-only entry, M5 D2(b)), to
/// put before the `[federation]` text of a federated config.
pub fn remote_only_b_text() -> String {
    let doc = toml::Table::from_iter([(
        "deployments".to_owned(),
        toml::Value::Array(vec![entry_b().into()]),
    )]);
    toml::to_string(&doc).unwrap()
}

/// Two deployments (M5 D1): A is [`replicas_config_text`] over `a_urls` (devices `gpu0..`), B is
/// `multimodel-dev.toml`'s [`MODEL_B`] entry with one local replica per `b_urls` (devices after A's).
/// `tweaks` apply to A's text.
pub fn multimodel_config_text(
    a_urls: &[String],
    b_urls: &[String],
    state_dir: &Path,
    port: u16,
    tweaks: &[(&str, &str)],
) -> String {
    let mut doc: toml::Table = replicas_config_text(a_urls, state_dir, port, tweaks)
        .parse()
        .unwrap();
    let devices: Vec<String> = (a_urls.len()..a_urls.len() + b_urls.len())
        .map(|i| format!("gpu{i}"))
        .collect();
    let replicas: Vec<toml::Value> = devices
        .iter()
        .zip(b_urls)
        .map(|(device, url)| {
            toml::Value::Table(toml::Table::from_iter([
                ("device".to_owned(), device.as_str().into()),
                ("base_url".to_owned(), url.as_str().into()),
            ]))
        })
        .collect();
    let mut b = entry_b();
    b.insert("replicas".to_owned(), toml::Value::Array(replicas));
    doc["deployments"].as_array_mut().unwrap().push(b.into());
    let all = doc["scheduler"]["devices"].as_array_mut().unwrap();
    all.extend(devices.into_iter().map(toml::Value::from));
    toml::to_string(&doc).unwrap()
}

/// The `/get_server_info` of B's launch: every pin of `multimodel-dev.toml`'s B `expect`, no quantization,
/// speculative decoding or chat template.
pub fn server_info_b() -> ServerInfo {
    let entry = entry_b();
    let expect = entry["expect"].as_table().unwrap();
    let text = |key: &str| expect[key].as_str().unwrap().to_owned();
    ServerInfo {
        version: text("version"),
        model_path: text("model_path"),
        tokenizer_path: text("tokenizer_path"),
        served_model_name: MODEL_B.to_owned(),
        chat_template: None,
        quantization: None,
        kv_cache_dtype: text("kv_cache_dtype"),
        speculative_algorithm: None,
        speculative_num_steps: None,
        speculative_eagle_topk: None,
        speculative_num_draft_tokens: None,
        reasoning_parser: Some(text("reasoning_parser")),
        tool_call_parser: Some(text("tool_call_parser")),
        allow_auto_truncate: expect["allow_auto_truncate"].as_bool().unwrap(),
        enable_cache_report: expect["enable_cache_report"].as_bool().unwrap(),
        context_length: expect["context_length"].as_integer().unwrap() as u64,
        max_running_requests: expect["max_running_requests"].as_integer().unwrap() as u32,
        ..ServerInfo::default()
    }
}

/// The hex SHA-256 of `configs/deployments/<prefix>.lock.json` (a deployment's `lock_sha256`).
pub fn lock_sha256_of(prefix: &str) -> String {
    use sha2::{Digest, Sha256};
    let bytes =
        std::fs::read(repo_root().join(format!("configs/deployments/{prefix}.lock.json"))).unwrap();
    format!("{:x}", Sha256::digest(bytes))
}

/// The live profile with `backend` edits, the test bind/state/scopes edits, then `tweaks`.
fn render_config(
    backend: &[(&str, &str)],
    state_dir: &Path,
    port: u16,
    tweaks: &[(&str, &str)],
) -> String {
    let mut text =
        std::fs::read_to_string(repo_root().join("configs/runtime/sglang-t4-qwen38.toml")).unwrap();
    let mut replace = |from: &str, to: &str| {
        assert!(text.contains(from), "config has no {from:?}");
        text = text.replace(from, to);
    };
    replace(
        "bind = \"127.0.0.1:18390\"",
        &format!("bind = \"127.0.0.1:{port}\""),
    );
    for (from, to) in backend {
        replace(from, to);
    }
    replace(
        "state_dir = \"/var/lib/cere/sglang-t4\"",
        &format!("state_dir = \"{}\"", state_dir.display()),
    );
    replace(
        "[[auth.scopes]]\nscope_id = \"dev\"\ntoken_env = \"CERE_TOKEN_DEV\"",
        "[[auth.scopes]]\nscope_id = \"a\"\ntoken_env = \"TOK_A\"\n\n[[auth.scopes]]\nscope_id = \"b\"\ntoken_env = \"TOK_B\"",
    );
    replace(
        "allow_cross_scope_prefix_cache = false",
        "allow_cross_scope_prefix_cache = true",
    );
    for (from, to) in tweaks {
        replace(from, to);
    }
    text
}

#[test]
fn replicas_config_text_renders_the_replicas_form() {
    let urls = [
        "http://127.0.0.1:1".to_owned(),
        "http://127.0.0.1:2".to_owned(),
    ];
    let state_dir = Path::new("/tmp/cered-replicas");
    let tweak = [("queue_depth = 64", "queue_depth = 8")];
    let mut doc: toml::Table = replicas_config_text(&urls, state_dir, 18391, &tweak)
        .parse()
        .unwrap();
    let replicas = doc["deployments"][0]
        .as_table_mut()
        .unwrap()
        .remove("replicas")
        .unwrap();
    assert_eq!(
        replicas,
        toml::Value::Array(
            [("gpu0", &urls[0]), ("gpu1", &urls[1])]
                .into_iter()
                .map(|(device, url)| {
                    toml::Value::Table(toml::Table::from_iter([
                        ("device".to_owned(), device.into()),
                        ("base_url".to_owned(), url.as_str().into()),
                    ]))
                })
                .collect()
        )
    );
    let devices = doc["scheduler"]
        .as_table_mut()
        .unwrap()
        .remove("devices")
        .unwrap();
    assert_eq!(
        devices,
        toml::Value::Array(vec!["gpu0".into(), "gpu1".into()])
    );
    // Everything else is the single form's text, without `deployments[0].base_url`.
    let mut single: toml::Table = config_text(&urls[0], state_dir, 18391, &tweak)
        .parse()
        .unwrap();
    single["deployments"][0]
        .as_table_mut()
        .unwrap()
        .remove("base_url");
    single["scheduler"]
        .as_table_mut()
        .unwrap()
        .remove("devices");
    assert_eq!(doc, single);
}

/// Shrunk production timings. The cancel ack bound is only a guard against an abort that never ends its
/// generation: a test that asserts an acknowledged cancel (`cancelled`) must not depend on a loaded host
/// acknowledging inside a short window. A test whose backend never acknowledges runs under [`short_ack`].
/// Local workers age to Suspect after three probe periods (15 s), as in production.
pub fn test_tuning() -> Tuning {
    Tuning {
        backend: Timeouts {
            connect: Duration::from_secs(1),
            control: Duration::from_secs(2),
            stream_idle: Duration::from_secs(10),
        },
        smoke_deadline: Duration::from_secs(10),
        cancel_ack_timeout: Duration::from_secs(20),
        keepalive_interval: Duration::from_secs(15),
        probe_interval: Duration::from_secs(5),
        suspect_after: Duration::from_secs(15),
        remote_interval: Duration::from_secs(5),
        retention_clock: RetentionClock::Wall,
    }
}

/// The cancel ack bound of [`short_ack`].
pub const SHORT_CANCEL_ACK: Duration = Duration::from_millis(400);

/// `tuning` with the [`SHORT_CANCEL_ACK`] bound, for a test whose backend never acknowledges the abort: the bound
/// expiring is the event it waits for.
pub fn short_ack(tuning: Tuning) -> Tuning {
    Tuning {
        cancel_ack_timeout: SHORT_CANCEL_ACK,
        ..tuning
    }
}

pub fn http() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

/// [`test_tuning`] probing every 100 ms (local probes and remote driver rounds), so a drift or a relaunch shows
/// within a few polls. Local workers keep the 15 s suspect bound: aged by three 100 ms periods, a stall of a
/// loaded host over 300 ms aged every replica to Suspect, and a request waited, failed `503 not_ready` or went to
/// the other worker (rt-flake design note F8). A test whose event is a local worker aging takes [`probe_aging`].
pub fn fast_tuning() -> Tuning {
    Tuning {
        probe_interval: Duration::from_millis(100),
        remote_interval: Duration::from_millis(100),
        ..test_tuning()
    }
}

/// `tuning` whose local workers age to Suspect after three probe periods, as in production, for a test whose event
/// is that aging: a worker reading suspect while its probe hangs, Lost after `scheduler.worker_lost_after_ms`, or
/// a smoke capped by half the suspect + lost bound.
pub fn probe_aging(tuning: Tuning) -> Tuning {
    Tuning {
        suspect_after: 3 * tuning.probe_interval,
        ..tuning
    }
}

/// [`test_tuning`] with remote driver rounds every 100 ms: a remote worker joins within a few rounds, while a local
/// worker keeps the 5 s probe.
pub fn fast_remote_tuning() -> Tuning {
    Tuning {
        remote_interval: Duration::from_millis(100),
        ..test_tuning()
    }
}

/// `n` default mocks, for [`Harness::start_replicas`].
pub async fn mocks(n: usize) -> Vec<MockHandle> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(MockSglang::start(MockConfig::default()).await.unwrap());
    }
    out
}

/// A server info reporting a backend launch: a new `(random_seed, startup_time)` is a new `instance_id`.
pub fn launched(random_seed: u64, total: f64) -> ServerInfo {
    ServerInfo {
        random_seed: Some(random_seed),
        startup_time: Some(json!({ "total": total })),
        ..ServerInfo::default()
    }
}

/// The prompt-export launch keys a vetting client requires reported as `null`/`false` (Chief's `local-sglang`
/// provider, `packages/micode_adapter/sglang.py` `EXPORT_FLAGS`, sglang 0.5.19), copied here rather than taken from
/// `cere_runtime::server_info` so the tests compare cered against the client, not against itself.
pub const CLIENT_EXPORT_FLAGS: [&str; 8] = [
    "kv_events",
    "kv_events_config",
    "hicache_storage_backend",
    "debug_tensor_dump_output_folder",
    "enable_trace",
    "log_requests_target",
    "spec_trace_dir",
    "msprobe_dump_config",
];

/// server-info design note SI-06: the vetting rule of Chief's `local-sglang` provider (`check_server`) over a
/// `GET /get_server_info` body, for a route serving `model` (`lan`: its `accepted_exposure` is `lan`). `None` when
/// the client would send a prompt, else the reason it refuses.
pub fn client_refusal(info: &Value, model: &str, lan: bool) -> Option<String> {
    let Some(info) = info.as_object() else {
        return Some("not an sglang server".into());
    };
    if info.get("served_model_name") != Some(&json!(model)) {
        return Some("the server serves a different model than the route".into());
    }
    let host = match info.get("host") {
        Some(Value::String(host)) if !host.is_empty() => host,
        _ => return Some("the server reports no listen address".into()),
    };
    if !["127.0.0.1", "::1", "localhost"].contains(&host.as_str()) && !lan {
        return Some("the server listens beyond loopback".into());
    }
    if info.get("log_requests") != Some(&json!(false)) {
        return Some("the server may log requests".into());
    }
    let truthy = |v: &Value| match v {
        Value::Null | Value::Bool(false) => false,
        Value::String(s) => !s.is_empty(),
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::Bool(true) => true,
    };
    if info.get("crash_dump_folder").is_some_and(truthy) {
        return Some("the server keeps crash dumps".into());
    }
    let exporting: Vec<&str> = CLIENT_EXPORT_FLAGS
        .into_iter()
        .filter(|flag| !matches!(info.get(*flag), Some(Value::Null | Value::Bool(false))))
        .collect();
    if !exporting.is_empty() {
        return Some(format!(
            "the server may export prompt data ({})",
            exporting.join(", ")
        ));
    }
    None
}

pub async fn readyz(h: &Harness) -> (u16, Value) {
    let response = h.http.get(h.url("/readyz")).send().await.unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

/// Polls `/readyz` until `done` holds for its body (10 s bound); returns the status and body that satisfied it.
pub async fn readyz_until(h: &Harness, what: &str, done: impl Fn(&Value) -> bool) -> (u16, Value) {
    let mut last = Value::Null;
    for _ in 0..500 {
        let (status, body) = readyz(h).await;
        if done(&body) {
            return (status, body);
        }
        last = body;
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("/readyz never showed {what}: {last}");
}

/// A started `mock`, a state dir and the live config at them with [`FEDERATION_BODY_TWEAK`] and `tweaks`, then
/// `appended` (`[federation]`).
async fn federated_config(
    mock: MockConfig,
    tweaks: &[(&str, &str)],
    appended: &str,
) -> (MockHandle, StateDir, String) {
    let mock = MockSglang::start(mock).await.unwrap();
    let state_dir = StateDir::on_tmpfs();
    let tweaks: Vec<(&str, &str)> = std::iter::once(FEDERATION_BODY_TWEAK)
        .chain(tweaks.iter().copied())
        .collect();
    let mut config = config_text(&mock.base_url(), state_dir.path(), 18390, &tweaks);
    config.push('\n');
    config.push_str(appended);
    (mock, state_dir, config)
}

pub struct Harness {
    /// The mock behind the only backend ([`Harness::start`]) or behind replica `gpu0`
    /// ([`Harness::start_replicas`]); [`Harness::replica`] reaches every replica.
    pub mock: MockHandle,
    pub app: Arc<App>,
    pub base: String,
    pub http: reqwest::Client,
    pub state_dir: StateDir,
    /// The mocks behind replicas `gpu1..`, in device order; empty for the single form.
    more_mocks: Vec<MockHandle>,
    config: String,
    /// Holds `cered.toml`, the config text the app started from: its reload source.
    config_dir: StateDir,
    tuning: Tuning,
    /// Opened waiting for the local workers to turn ready; [`Harness::restart`] reopens the same way.
    wait_local: bool,
    stop: Option<oneshot::Sender<()>>,
    join: Option<JoinHandle<std::io::Result<bool>>>,
}

impl Harness {
    pub async fn start(mock: MockConfig, tweaks: &[(&str, &str)]) -> Self {
        Self::start_tuned(mock, tweaks, test_tuning()).await
    }

    pub async fn start_tuned(mock: MockConfig, tweaks: &[(&str, &str)], tuning: Tuning) -> Self {
        Self::start_socket(mock, tweaks, tuning, None).await
    }

    /// `send_buffer` sets SO_SNDBUF on the listener (accepted sockets inherit it), so a client that stops
    /// reading blocks the server after a few KiB instead of several MiB of kernel buffering.
    pub async fn start_socket(
        mock: MockConfig,
        tweaks: &[(&str, &str)],
        tuning: Tuning,
        send_buffer: Option<u32>,
    ) -> Self {
        let mock = MockSglang::start(mock).await.unwrap();
        let state_dir = StateDir::on_tmpfs();
        let config = config_text(&mock.base_url(), state_dir.path(), 18390, tweaks);
        Self::open(mock, Vec::new(), state_dir, config, tuning, send_buffer).await
    }

    /// `cered` over N independent replicas in the `[[deployments.replicas]]` form ([`replicas_config_text`]):
    /// `mocks[i]` serves device `gpu<i>`.
    pub async fn start_replicas(mocks: Vec<MockHandle>, tweaks: &[(&str, &str)]) -> Self {
        Self::start_replicas_tuned(mocks, tweaks, test_tuning()).await
    }

    pub async fn start_replicas_tuned(
        mocks: Vec<MockHandle>,
        tweaks: &[(&str, &str)],
        tuning: Tuning,
    ) -> Self {
        let base_urls: Vec<String> = mocks.iter().map(MockHandle::base_url).collect();
        let state_dir = StateDir::on_tmpfs();
        let config = replicas_config_text(&base_urls, state_dir.path(), 18390, tweaks);
        let mut mocks = mocks.into_iter();
        let mock = mocks.next().expect("at least one replica");
        Self::open(mock, mocks.collect(), state_dir, config, tuning, None).await
    }

    /// Like `start`, but `seed` runs against the parsed config before the server opens its state dir.
    pub async fn start_seeded(
        mock: MockConfig,
        tweaks: &[(&str, &str)],
        seed: impl FnOnce(&RuntimeConfig),
    ) -> Self {
        Self::start_seeded_tuned(mock, tweaks, test_tuning(), seed).await
    }

    /// [`Harness::start_seeded`] under `tuning`.
    pub async fn start_seeded_tuned(
        mock: MockConfig,
        tweaks: &[(&str, &str)],
        tuning: Tuning,
        seed: impl FnOnce(&RuntimeConfig),
    ) -> Self {
        let mock = MockSglang::start(mock).await.unwrap();
        let state_dir = StateDir::on_tmpfs();
        let config = config_text(&mock.base_url(), state_dir.path(), 18390, tweaks);
        seed(&RuntimeConfig::from_toml_str(&config, &repo_root()).unwrap());
        Self::open(mock, Vec::new(), state_dir, config, tuning, None).await
    }

    /// Like `start_tuned` with `appended` (a `[federation]` section, see [`Federation::toml`]) added to the
    /// config text.
    pub async fn start_federated(mock: MockConfig, appended: &str, tuning: Tuning) -> Self {
        Self::start_federated_with(mock, &[], appended, tuning).await
    }

    /// [`Harness::start_federated`] with `tweaks` applied to the config text before `appended` is added.
    pub async fn start_federated_with(
        mock: MockConfig,
        tweaks: &[(&str, &str)],
        appended: &str,
        tuning: Tuning,
    ) -> Self {
        let (mock, state_dir, config) = federated_config(mock, tweaks, appended).await;
        Self::open(mock, Vec::new(), state_dir, config, tuning, None).await
    }

    /// [`Harness::start_federated`] without waiting for the local worker to turn ready (track DA: a local
    /// worker whose mock never passes readiness, so only remote workers take requests).
    pub async fn start_federated_unready(mock: MockConfig, appended: &str, tuning: Tuning) -> Self {
        let (mock, state_dir, config) = federated_config(mock, &[], appended).await;
        Self::open_with(mock, Vec::new(), state_dir, config, tuning, None, false).await
    }

    /// `cered` over an already rendered `config` (M5 multi-deployment tests): `mocks[0]` is [`Harness::mock`],
    /// `mocks[i]` is [`Harness::replica`]`(i)`; waits for the local workers like [`Harness::start`].
    pub async fn start_config(
        mocks: Vec<MockHandle>,
        state_dir: StateDir,
        config: String,
        tuning: Tuning,
    ) -> Self {
        let mut mocks = mocks.into_iter();
        let mock = mocks.next().expect("at least one mock");
        Self::open(mock, mocks.collect(), state_dir, config, tuning, None).await
    }

    async fn open(
        mock: MockHandle,
        more_mocks: Vec<MockHandle>,
        state_dir: StateDir,
        config: String,
        tuning: Tuning,
        send_buffer: Option<u32>,
    ) -> Self {
        Self::open_with(
            mock,
            more_mocks,
            state_dir,
            config,
            tuning,
            send_buffer,
            true,
        )
        .await
    }

    async fn open_with(
        mock: MockHandle,
        more_mocks: Vec<MockHandle>,
        state_dir: StateDir,
        config: String,
        tuning: Tuning,
        send_buffer: Option<u32>,
        wait_local: bool,
    ) -> Self {
        let config_dir = StateDir::new();
        std::fs::create_dir_all(config_dir.path()).unwrap();
        let config_path = config_dir.path().join("cered.toml");
        std::fs::write(&config_path, &config).unwrap();
        let mut parsed = RuntimeConfig::from_toml_str(&config, &repo_root()).unwrap();
        parsed.source = Some(ConfigSource {
            path: config_path,
            repo_root: repo_root(),
        });
        let env = |name: &str| -> Option<std::ffi::OsString> {
            match name {
                "TOK_A" => Some(TOKEN_A.into()),
                "TOK_B" => Some(TOKEN_B.into()),
                "TOK_OP" => Some(OPERATOR_TOKEN.into()),
                _ => None,
            }
        };
        let credentials = parsed.resolve_auth_with(env).unwrap();
        let operator = parsed.resolve_operator_token_with(env).unwrap();
        let app = App::open(parsed, &credentials, operator, tuning.clone()).unwrap();
        if wait_local {
            wait_ready(&app).await;
        }
        let socket = tokio::net::TcpSocket::new_v4().unwrap();
        if let Some(bytes) = send_buffer {
            socket.set_send_buffer_size(bytes).unwrap();
        }
        socket.bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let listener = socket.listen(64).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let (stop, stopped) = oneshot::channel::<()>();
        let join = tokio::spawn(server::serve(app.clone(), listener, async move {
            let _ = stopped.await;
        }));
        Self {
            mock,
            app,
            base,
            http: http(),
            state_dir,
            more_mocks,
            config,
            config_dir,
            tuning,
            wait_local,
            stop: Some(stop),
            join: Some(join),
        }
    }

    /// Runs the L8 shutdown; returns whether it was clean.
    pub async fn stop(&mut self) -> bool {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        match self.join.take() {
            Some(join) => join.await.unwrap().unwrap(),
            None => true,
        }
    }

    /// [`Harness::restart`] under another config text (same state dir and mocks).
    pub async fn restart_with_config(mut self, config: String) -> Self {
        self.config = config;
        self.restart().await
    }

    /// The config text this incarnation started from.
    pub fn started_config(&self) -> &str {
        &self.config
    }

    /// Overwrites the file this incarnation started from (its live reload source) with `text`.
    pub fn write_config(&self, text: &str) {
        std::fs::write(self.config_dir.path().join("cered.toml"), text).unwrap();
    }

    /// `POST /v1/cere/config/reload` under the operator token: (status, body).
    pub async fn reload(&self) -> (u16, Value) {
        let r = self
            .post("/v1/cere/config/reload", OPERATOR_TOKEN)
            .send()
            .await
            .unwrap();
        let status = r.status().as_u16();
        (status, r.json().await.unwrap())
    }

    /// `GET /v1/cere/stats` of scope `a`.
    pub async fn stats(&self) -> Value {
        let r = self.get("/v1/cere/stats", TOKEN_A).send().await.unwrap();
        assert_eq!(r.status(), 200);
        r.json().await.unwrap()
    }

    /// Stops this server and opens a new incarnation on the same state dir and mocks, waiting for the local workers
    /// as the first one did. The old `App` holds the state-dir lock (a real process releases it on exit), so this
    /// waits until the harness owns the last reference (executor tasks may still be unwinding after the drain) and
    /// drops it before reopening.
    pub async fn restart(mut self) -> Self {
        self.stop().await;
        let Harness {
            mock,
            more_mocks,
            app,
            state_dir,
            config,
            tuning,
            wait_local,
            ..
        } = self;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while Arc::strong_count(&app) > 1 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "App still referenced after stop"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        drop(app);
        Self::open_with(
            mock, more_mocks, state_dir, config, tuning, None, wait_local,
        )
        .await
    }

    /// The mock behind replica `gpu<index>` (`0` = [`Harness::mock`]).
    pub fn replica(&self, index: usize) -> &MockHandle {
        match index {
            0 => &self.mock,
            _ => &self.more_mocks[index - 1],
        }
    }

    /// How many backend mocks this harness runs.
    pub fn replica_count(&self) -> usize {
        1 + self.more_mocks.len()
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    pub fn get(&self, path: &str, token: &str) -> reqwest::RequestBuilder {
        self.http.get(self.url(path)).bearer_auth(token)
    }

    pub fn post(&self, path: &str, token: &str) -> reqwest::RequestBuilder {
        self.http.post(self.url(path)).bearer_auth(token)
    }

    pub fn chat(&self, body: &Value) -> reqwest::RequestBuilder {
        self.post("/v1/chat/completions", TOKEN_A).json(body)
    }

    pub async fn status(&self, rid: &str, token: &str) -> (u16, Value) {
        let r = self
            .get(&format!("/v1/cere/requests/{rid}"), token)
            .send()
            .await
            .unwrap();
        let status = r.status().as_u16();
        (status, r.json().await.unwrap())
    }

    /// Polls the request status until it is terminal (within 30 s: a bound for a hang, not a timing assertion, so
    /// the suite holds when test binaries run in parallel).
    pub async fn terminal(&self, rid: &str) -> Value {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while std::time::Instant::now() < deadline {
            let (code, body) = self.status(rid, TOKEN_A).await;
            assert_eq!(code, 200, "{body}");
            if ["completed", "failed", "cancelled"].contains(&body["state"].as_str().unwrap()) {
                return body;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("request {rid} did not reach a terminal state");
    }
}

/// An HTTP/1.0 chat request over a socket with a tiny receive buffer, answered close-delimited (no chunked
/// framing). Returns the response head and the connected stream, positioned after the head; the caller
/// decides when (or whether) to read the body.
pub async fn raw_chat(base: &str, body: &Value) -> (String, tokio::net::TcpStream, Vec<u8>) {
    raw_chat_with(base, body, &[]).await
}

/// [`raw_chat`] with extra request headers.
pub async fn raw_chat_with(
    base: &str,
    body: &Value,
    headers: &[(&str, &str)],
) -> (String, tokio::net::TcpStream, Vec<u8>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let addr = base.strip_prefix("http://").unwrap().parse().unwrap();
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    socket.set_recv_buffer_size(2048).unwrap();
    let mut stream = socket.connect(addr).await.unwrap();
    let payload = body.to_string();
    let extra = headers
        .iter()
        .fold(String::new(), |mut out, (name, value)| {
            out.push_str(&format!("{name}: {value}\r\n"));
            out
        });
    let request = format!(
        "POST /v1/chat/completions HTTP/1.0\r\nHost: {addr}\r\nAuthorization: Bearer {TOKEN_A}\r\n\
         {extra}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
        payload.len()
    );
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    let mut chunk = [0u8; 256];
    loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8(buf[..i].to_vec()).unwrap();
            let rest = buf[i + 4..].to_vec();
            return (head, stream, rest);
        }
        let n = stream.read(&mut chunk).await.unwrap();
        assert!(n > 0, "connection closed before the response head");
        buf.extend_from_slice(&chunk[..n]);
    }
}

/// Splits a complete `text/event-stream` body into `data:` events (comments skipped).
pub fn parse_events(text: &str) -> Vec<Event> {
    assert!(
        text.is_empty() || text.ends_with("\n\n"),
        "trailing partial event"
    );
    text.split("\n\n")
        .filter(|raw| !raw.is_empty() && !raw.starts_with(':'))
        .map(parse_event)
        .collect()
}

/// One raw SSE event: every JSON event is `id: <seq>` (its `cere.seq`, m1close Q4) then its `data:` line;
/// `[DONE]` is a bare `data:` line.
pub fn parse_event(raw: &str) -> Event {
    if raw == "data: [DONE]" {
        return Event::Done;
    }
    let (id, data) = raw
        .strip_prefix("id: ")
        .and_then(|rest| rest.split_once("\ndata: "))
        .unwrap_or_else(|| panic!("id + data event: {raw:?}"));
    let value: Value = serde_json::from_str(data).unwrap();
    assert_eq!(
        Some(id.parse::<u64>().unwrap()),
        value["cere"]["seq"].as_u64(),
        "{raw}"
    );
    Event::Data(value)
}

/// Probes every replica until each is ready (2 s bound per replica).
pub async fn wait_ready(app: &App) {
    for (index, replica) in app.replicas().iter().enumerate() {
        let mut ready = false;
        for _ in 0..100 {
            if app.readiness(index).probe().await.ready {
                ready = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(
            ready,
            "worker {} never became ready: {:?}",
            replica.worker_id(),
            replica.readiness().snapshot()
        );
    }
}

pub fn chat_body(stream: bool) -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello there"}],
        "max_completion_tokens": 8,
        "stream": stream,
    })
}

/// Reads `text/event-stream` events (`data:` payloads and `:` comments) from a response.
pub struct SseReader {
    response: reqwest::Response,
    buf: String,
}

impl SseReader {
    pub fn new(response: reqwest::Response) -> Self {
        Self {
            response,
            buf: String::new(),
        }
    }

    /// The next raw event text (without the trailing blank line); `None` at the end of the body.
    pub async fn next_raw(&mut self) -> Option<String> {
        loop {
            if let Some(i) = self.buf.find("\n\n") {
                let event = self.buf[..i].to_owned();
                self.buf.drain(..i + 2);
                return Some(event);
            }
            match self.response.chunk().await.unwrap() {
                Some(bytes) => self.buf.push_str(std::str::from_utf8(&bytes).unwrap()),
                None => {
                    assert!(self.buf.is_empty(), "trailing partial event {:?}", self.buf);
                    return None;
                }
            }
        }
    }

    /// The next `data:` event (comments skipped): `Some(Data(json))`, `Some(Done)` or `None` at the end.
    pub async fn next(&mut self) -> Option<Event> {
        loop {
            let raw = self.next_raw().await?;
            if !raw.starts_with(':') {
                return Some(parse_event(&raw));
            }
        }
    }

    /// Every remaining event up to the end of the body.
    pub async fn rest(&mut self) -> Vec<Event> {
        let mut out = Vec::new();
        while let Some(e) = self.next().await {
            out.push(e);
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Data(Value),
    Done,
}

impl Event {
    pub fn data(&self) -> &Value {
        match self {
            Event::Data(v) => v,
            Event::Done => panic!("expected a data event, got [DONE]"),
        }
    }
}

/// Checks the stream framing contract on a complete event list: seq strictly increasing from the accepted
/// event (0), exactly one terminal (a finish chunk or an error event), followed only by an optional usage chunk
/// (`choices: []`) and then `[DONE]`, which is last (L5 ordering). Returns the terminal.
pub fn check_stream(events: &[Event]) -> Value {
    let first = events[0].data();
    assert_eq!(first["cere"]["event"], "accepted");
    assert_eq!(first["cere"]["seq"], 0);
    check_stream_tail(&events[1..], 0)
}

/// [`check_stream`] for the events after a reader already consumed the first ones, the last of which had
/// `seq` `after_seq`.
pub fn check_stream_tail(events: &[Event], after_seq: u64) -> Value {
    assert!(matches!(events.last(), Some(Event::Done)), "{events:?}");
    let body = &events[..events.len() - 1];
    let mut last_seq = after_seq;
    let mut terminals = Vec::new();
    for (index, event) in body.iter().enumerate() {
        let data = event.data();
        let seq = data["cere"]["seq"].as_u64().expect("seq");
        assert!(seq > last_seq, "seq {seq} after {last_seq}");
        last_seq = seq;
        let finished = data
            .pointer("/choices/0/finish_reason")
            .is_some_and(|f| !f.is_null());
        if finished || data.get("error").is_some() {
            terminals.push((index, data.clone()));
        }
    }
    assert_eq!(terminals.len(), 1, "{events:?}");
    let (index, terminal) = terminals.pop().unwrap();
    match &body[index + 1..] {
        [] => {}
        [usage] => assert_eq!(
            usage.data()["choices"],
            serde_json::json!([]),
            "only a usage chunk may follow the terminal: {events:?}"
        ),
        _ => panic!("events after the terminal: {events:?}"),
    }
    terminal
}

/// Every `cere` identity field of A4 and M3 Q11, checked against the app: the worker is one of its replicas,
/// at that worker's current epoch and backend identity; `epoch` is the process epoch.
pub fn assert_identity(cere: &Value, app: &App, rid: &str) {
    assert_eq!(cere["request_id"], rid);
    assert_eq!(cere["attempt"], 1);
    let worker = cere["worker_id"]
        .as_str()
        .expect("worker_id after dispatch");
    let replica = app
        .replica(worker)
        .unwrap_or_else(|| panic!("unknown worker_id {worker}"));
    assert_eq!(cere["worker_epoch"], replica.epoch());
    assert_eq!(cere["epoch"], app.epoch());
    assert_eq!(cere["execution_mode"], "backend_managed");
    let instance = replica.readiness().snapshot().instance_id.unwrap();
    assert_eq!(cere["instance_id"], instance.as_str());
    for field in ["recipe_id", "lock_sha256", "profile_id", "config_hash"] {
        assert!(
            cere[field].as_str().is_some_and(|s| !s.is_empty()),
            "{field} missing in {cere}"
        );
    }
}

/// A config file for the real binary. `RuntimeConfig::load` needs the file inside the checkout (the
/// deployment paths are repo-relative), so it lives in a git-ignored `*.cere/` directory removed on drop.
pub struct ConfigFile {
    dir: PathBuf,
    pub path: PathBuf,
}

impl ConfigFile {
    pub fn new(text: &str) -> Self {
        sweep_dead_once(&repo_root());
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = repo_root().join(format!(
            "cered-test-{}-{}.cere",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cered.toml");
        std::fs::write(&path, text).unwrap();
        Self { dir, path }
    }
}

impl Drop for ConfigFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The M3.5 coordinator principal, cluster and coordinator ids of [`Federation::toml`].
pub const FED_COORDINATOR: &str = "cere-coordinator";
pub const FED_CLUSTER: &str = "lab-cere";

/// An rcgen cluster CA with the coordinator's PEM files in a temp dir outside the checkout, a loopback registry
/// listen address, and the `[federation]` text pointing at them (m35 design note Q2, Q16).
pub struct Federation {
    pub pki: pki::TestPki,
    pub dir: StateDir,
    pub ca: PathBuf,
    pub listen: std::net::SocketAddr,
    cert: PathBuf,
    key: PathBuf,
}

impl Federation {
    pub fn new() -> Self {
        let pki = pki::TestPki::new();
        let dir = StateDir::new();
        std::fs::create_dir_all(dir.path()).unwrap();
        let ca = pki.write_ca(dir.path());
        let (cert, key) = pki.leaf(FED_COORDINATOR).write(dir.path());
        let listen = std::net::SocketAddr::from(([127, 0, 0, 1], free_port()));
        Self {
            pki,
            dir,
            ca,
            listen,
            cert,
            key,
        }
    }

    /// `[federation]` with `keys` (extra `key = value` lines, e.g. timings) and one `[[federation.workers]]`
    /// per `(worker_id, principal)`, each pinned to the test model and dialled at an unused loopback port.
    pub fn toml(&self, keys: &str, workers: &[(&str, &str)]) -> String {
        let mut text = format!(
            "[federation]\ncluster_id = \"{FED_CLUSTER}\"\ncoordinator_id = \"{FED_COORDINATOR}\"\n\
             listen = \"{}\"\nprincipal = \"{FED_COORDINATOR}\"\nca_file = \"{}\"\ncert_file = \"{}\"\n\
             key_file = \"{}\"\n{keys}\n",
            self.listen,
            self.ca.display(),
            self.cert.display(),
            self.key.display()
        );
        for (worker_id, principal) in workers {
            text.push_str(&format!(
                "[[federation.workers]]\nworker_id = \"{worker_id}\"\nendpoint = \"https://127.0.0.1:{}\"\n\
                 principal = \"{principal}\"\nmax_inflight = 4\n\n[federation.workers.expect]\n\
                 served_model_name = \"{MODEL}\"\ntp_size = 1\nkv_cache_dtype = \"fp8_e4m3\"\n\n",
                free_port()
            ));
        }
        text
    }

    /// Client TLS of an agent holding a leaf for `principal` under this CA, expecting the coordinator.
    pub fn agent_tls(&self, principal: &str) -> tonic::transport::ClientTlsConfig {
        let leaf = self.pki.leaf(principal);
        tls_with(self.pki.ca_pem(), &leaf, FED_COORDINATOR)
    }

    /// A lazily connected channel to the registry listener (a handshake refusal shows on the first call).
    pub fn channel(&self, tls: tonic::transport::ClientTlsConfig) -> tonic::transport::Channel {
        tonic::transport::Endpoint::from_shared(format!("https://{}", self.listen))
            .unwrap()
            .timeout(Duration::from_secs(10))
            .tls_config(tls)
            .unwrap()
            .connect_lazy()
    }
}

/// Client TLS trusting `ca_pem`, presenting `leaf`, expecting the server principal `peer`.
pub fn tls_with(ca_pem: &str, leaf: &pki::Leaf, peer: &str) -> tonic::transport::ClientTlsConfig {
    tonic::transport::ClientTlsConfig::new()
        .ca_certificate(tonic::transport::Certificate::from_pem(ca_pem))
        .identity(tonic::transport::Identity::from_pem(
            leaf.cert_pem(),
            leaf.key_pem(),
        ))
        .domain_name(peer)
}

/// A running `cered --config <path>` child process.
pub struct Cered {
    child: std::process::Child,
    pub base: String,
    pub epoch: u64,
    /// Collects stderr until the process closes it ([`Cered::stderr_so_far`] reads it meanwhile).
    stderr: Option<std::thread::JoinHandle<()>>,
    stderr_text: Arc<Mutex<String>>,
}

impl Cered {
    /// Spawns the binary with `env` (TOK_A/TOK_B unless overridden) and waits for the startup line.
    /// `Err((exit code, stderr))` when the process exits before printing it.
    pub async fn spawn(
        config: &Path,
        env: &[(&str, Option<&str>)],
    ) -> Result<Self, (Option<i32>, String)> {
        Self::spawn_binary(Path::new(env!("CARGO_BIN_EXE_cered")), config, env).await
    }

    /// [`Cered::spawn`] of another `cered` build, e.g. an earlier release (a rollback test).
    pub async fn spawn_binary(
        binary: &Path,
        config: &Path,
        env: &[(&str, Option<&str>)],
    ) -> Result<Self, (Option<i32>, String)> {
        use std::io::{BufRead, Read};
        use std::process::{Command, Stdio};
        let mut command = Command::new(binary);
        command
            .arg("--config")
            .arg(config)
            .env("TOK_A", TOKEN_A)
            .env("TOK_B", TOKEN_B)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (name, value) in env {
            match value {
                Some(value) => command.env(name, value),
                None => command.env_remove(name),
            };
        }
        let mut child = command.spawn().unwrap();
        let stderr_text = Arc::new(Mutex::new(String::new()));
        let stderr = {
            let mut pipe = child.stderr.take().unwrap();
            let text = stderr_text.clone();
            std::thread::spawn(move || {
                let mut chunk = [0u8; 4096];
                while let Ok(n) = pipe.read(&mut chunk) {
                    if n == 0 {
                        break;
                    }
                    text.lock()
                        .unwrap()
                        .push_str(&String::from_utf8_lossy(&chunk[..n]));
                }
            })
        };
        let stdout = child.stdout.take().unwrap();
        let line = tokio::time::timeout(
            Duration::from_secs(20),
            tokio::task::spawn_blocking(move || {
                let mut line = String::new();
                std::io::BufReader::new(stdout)
                    .read_line(&mut line)
                    .unwrap();
                line
            }),
        )
        .await
        .expect("cered printed no startup line in time")
        .unwrap();
        if line.is_empty() {
            let code = child.wait().unwrap().code();
            stderr.join().unwrap();
            let text = std::mem::take(&mut *stderr_text.lock().unwrap());
            return Err((code, text));
        }
        let rest = line
            .trim_end()
            .strip_prefix("cered listening on ")
            .unwrap_or_else(|| panic!("unexpected startup line {line:?}"));
        let (base, epoch) = rest.split_once(" epoch=").unwrap();
        Ok(Self {
            child,
            base: base.to_owned(),
            epoch: epoch.parse().unwrap(),
            stderr: Some(stderr),
            stderr_text,
        })
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Everything the process wrote to stderr so far (it keeps running).
    pub fn stderr_so_far(&self) -> String {
        self.stderr_text.lock().unwrap().clone()
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    pub async fn wait_ready(&self) {
        let http = http();
        for _ in 0..500 {
            if let Ok(r) = http.get(self.url("/readyz")).send().await {
                if r.status() == 200 {
                    return;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("cered never became ready");
    }

    pub fn signal(&self, name: &str) {
        let ok = std::process::Command::new("kill")
            .args([format!("-{name}"), self.pid().to_string()])
            .status()
            .unwrap()
            .success();
        assert!(ok, "kill -{name} failed");
    }

    /// SIGKILL: no shutdown path runs.
    pub fn kill(&mut self) {
        self.child.kill().unwrap();
        self.child.wait().unwrap();
    }

    /// Waits (bounded) for the process to exit; returns its exit code and everything it wrote to stderr.
    pub async fn exit(&mut self, bound: Duration) -> (Option<i32>, String) {
        let until = std::time::Instant::now() + bound;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                if let Some(t) = self.stderr.take() {
                    t.join().unwrap();
                }
                let stderr = self.stderr_text.lock().unwrap().clone();
                return (status.code(), stderr);
            }
            assert!(
                std::time::Instant::now() < until,
                "cered did not exit in {bound:?}"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

impl Drop for Cered {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Every journal record of `state_dir`, parsed.
pub fn journal(state_dir: &Path) -> Vec<Value> {
    std::fs::read_to_string(state_dir.join("journal.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// Rids admitted in the journal without a terminal record.
pub fn non_terminal(records: &[Value]) -> Vec<String> {
    let terminal: std::collections::HashSet<&str> = records
        .iter()
        .filter(|r| r["type"] == "terminal")
        .map(|r| r["rid"].as_str().unwrap())
        .collect();
    records
        .iter()
        .filter(|r| r["type"] == "admitted")
        .map(|r| r["rid"].as_str().unwrap())
        .filter(|rid| !terminal.contains(rid))
        .map(str::to_owned)
        .collect()
}

/// Journals a prior-epoch request admitted at `admitted_ms` and dispatched to `config`'s first replica that never
/// reached a terminal (a crashed `cered`), in `config`'s state dir; returns its rid. The next `App::open` recovers
/// it as that worker's orphan (M3 Q9).
pub fn seed_orphan(config: &RuntimeConfig, scope: &str, admitted_ms: u64) -> String {
    use cere_runtime::digest::{DigestKeys, RequestDigestInput};
    use cere_runtime::ledger::{AcceptOutcome, AcceptRequest, Ledger, LedgerLimits, WorkerTarget};
    let dir = &config.ledger.state_dir;
    let keys = Arc::new(DigestKeys::open(dir).unwrap());
    let replicas = config.deployments[0].replica_list(&config.scheduler.devices);
    let local = |base_url: &str| WorkerTarget::Local {
        base_url: base_url.to_owned(),
    };
    let workers: std::collections::BTreeMap<String, WorkerTarget> = replicas
        .iter()
        .map(|r| (r.device.clone(), local(&r.base_url)))
        .collect();
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
        .scope_key(scope)
        .request_digest(&RequestDigestInput {
            request: &chat_body(true),
            session_id: None,
            expected_revision: None,
            served_model: MODEL,
            deadline_ms: 1_000,
            config_hash: "seed",
            surface: None,
        });
    let request = AcceptRequest {
        scope: scope.into(),
        rid: rid.clone(),
        key: None,
        digest,
        stream: true,
        surface: None,
        session: None,
    };
    assert_eq!(
        ledger.accept(request, admitted_ms),
        Ok(AcceptOutcome::Accepted)
    );
    let first = &replicas[0];
    let worker_epoch = ledger.worker_epochs()[&first.device];
    ledger.mark_queued(&rid, admitted_ms).unwrap();
    ledger.mark_running(&rid, admitted_ms).unwrap();
    ledger
        .dispatched(
            &rid,
            &first.device,
            worker_epoch,
            &local(&first.base_url),
            admitted_ms,
        )
        .unwrap();
    rid
}

impl Harness {
    /// [`Harness::start_federated_unready`] with `tweaks`, whose `seed` runs against the parsed config before the
    /// server opens its state dir (track ORPHAN-B: the journal a crashed federated `cered` left).
    pub async fn start_federated_seeded(
        mock: MockConfig,
        tweaks: &[(&str, &str)],
        appended: &str,
        tuning: Tuning,
        seed: impl FnOnce(&RuntimeConfig),
    ) -> Self {
        let (mock, state_dir, config) = federated_config(mock, tweaks, appended).await;
        seed(&RuntimeConfig::from_toml_str(&config, &repo_root()).unwrap());
        Self::open_with(mock, Vec::new(), state_dir, config, tuning, None, false).await
    }
}
