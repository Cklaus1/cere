//! M3.5 P3: the coordinator's `[federation]` and `[auth] operator_token_env` (m35 design note Q16, Q7, Q13, Q14)
//! and the `cere-worker` agent config (Q16, Q12, Q14; harness revision H1). Every refused case is a row of a shared
//! table (`tests/fixtures/runtime_config_invalid.json`, `tests/fixtures/worker_agent_config_invalid.json`) that
//! tests/test_cere_contracts.py also runs through the Python mirror; here each row is checked for its Rust error.
use std::ffi::OsString;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use cere_runtime::agent::config::AgentConfig;
use cere_runtime::config::RuntimeConfig;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const FEDERATION: &str = "configs/runtime/federation-dev.toml";
const LIVE: &str = "configs/runtime/sglang-t4-qwen38.toml";
const AGENT: &str = "configs/worker/node-b-mock.toml";
/// The managed-mode agent profile (m36 design note Q3, Q13); the base of every `[backend.launch]` refusal row.
const MANAGED: &str = "configs/worker/node-b-managed.toml";

const RUNTIME_CASES: [(&str, &str); 49] = [
    ("fed_listen_wildcard", "federation.listen must not be a wildcard address"),
    ("fed_listen_ipv6_wildcard", "federation.listen must not be a wildcard address"),
    ("fed_listen_public", "federation.listen host must be a 100.64.0.0/10 or loopback literal"),
    ("fed_listen_rfc1918", "federation.listen host must be a 100.64.0.0/10 or loopback literal"),
    ("fed_listen_cgnat_edge", "federation.listen host must be a 100.64.0.0/10 or loopback literal"),
    ("fed_listen_hostname", "federation.listen must be a canonical <ip>:<port> literal with a port 1..65535"),
    ("fed_listen_leading_zero", "federation.listen must be a canonical <ip>:<port> literal with a port 1..65535"),
    ("fed_listen_port_zero", "federation.listen must be a canonical <ip>:<port> literal with a port 1..65535"),
    ("fed_listen_no_port", "federation.listen must be a canonical <ip>:<port> literal with a port 1..65535"),
    ("fed_endpoint_http", "federation.workers.endpoint must use https://"),
    ("fed_endpoint_public", "federation.workers.endpoint host must be a literal 100.64.0.0/10, 127.0.0.0/8 or [::1] address"),
    ("fed_endpoint_rfc1918", "federation.workers.endpoint host must be a literal 100.64.0.0/10, 127.0.0.0/8 or [::1] address"),
    ("fed_endpoint_hostname", "federation.workers.endpoint host must be a literal 100.64.0.0/10, 127.0.0.0/8 or [::1] address"),
    ("fed_endpoint_wildcard", "federation.workers.endpoint host must be a literal 100.64.0.0/10, 127.0.0.0/8 or [::1] address"),
    ("fed_endpoint_trailing_slash", "federation.workers.endpoint needs an explicit port 1..65535 and nothing after it"),
    ("fed_endpoint_no_port", "federation.workers.endpoint needs an explicit port 1..65535 and nothing after it"),
    ("fed_duplicate_worker_id", "duplicate federation worker_id worker-a"),
    ("fed_worker_id_is_device", "federation worker_id equals a scheduler device: gpu0"),
    ("fed_duplicate_principal", "duplicate federation worker principal cere-worker-a"),
    ("fed_principal_is_coordinator", "federation worker principal equals the coordinator principal"),
    ("fed_principal_pattern", "federation.principal does not match its pattern"),
    ("fed_worker_principal_pattern", "federation.workers.principal does not match its pattern"),
    ("fed_lease_not_above_heartbeat", "federation timings must satisfy heartbeat_interval_ms < admission_lease_ms < lost_after_ms"),
    ("fed_lease_not_below_lost", "federation timings must satisfy heartbeat_interval_ms < admission_lease_ms < lost_after_ms"),
    ("fed_suspect_not_above_heartbeat", "federation suspect_after_ms must exceed heartbeat_interval_ms"),
    ("fed_lost_not_above_suspect", "federation lost_after_ms must exceed suspect_after_ms"),
    ("fed_lease_not_below_suspect", "federation admission_lease_ms must be below suspect_after_ms"),
    ("fed_heartbeat_range", "federation.heartbeat_interval_ms must be"),
    ("fed_rpc_timeout_range", "federation.rpc_timeout_ms must be"),
    ("fed_rpc_quota_zero", "federation.rpc_quota_per_principal must be"),
    ("fed_max_inflight_zero", "federation.workers.max_inflight must be"),
    ("fed_global_cap_exceeded", "admission max_backend_inflight exceeds local + remote workers x min(expect.max_running_requests, every max_inflight)"),
    ("fed_global_cap_uses_min_max_inflight", "admission max_backend_inflight exceeds local + remote workers x min(expect.max_running_requests, every max_inflight)"),
    ("fed_pem_relative", "federation.ca_file must be absolute"),
    ("fed_pem_in_checkout", "federation.cert_file must be outside any repository checkout"),
    ("fed_pem_under_state_dir", "federation.key_file must not be under the state_dir"),
    ("fed_pem_dotdot", "federation.ca_file component does not match its pattern"),
    ("fed_missing_cluster_id", "missing field `cluster_id`"),
    ("fed_missing_coordinator_id", "missing field `coordinator_id`"),
    ("fed_cluster_id_pattern", "federation.cluster_id does not match its pattern"),
    ("fed_unknown_key", "unknown field `stream_pool`"),
    ("fed_workers_empty", "federation.workers item count out of range"),
    ("fed_worker_expect_missing", "missing field `expect`"),
    ("fed_worker_tp_size_zero", "federation.workers.expect.tp_size must be"),
    ("fed_needs_sglang_http", "[federation] requires backend kind sglang_http"),
    ("fed_body_above_remote_max", "api.max_request_body_bytes must be <= 8323072 with [federation]"),
    ("operator_token_env_pattern", "auth.operator_token_env does not match its pattern"),
    ("operator_token_env_is_scope_env", "auth.operator_token_env must differ from every scope token_env"),
    ("fed_worker_matches_no_deployment", "federation worker worker-a expect.served_model_name other-model matches no deployments served_model_name"),
];

const AGENT_CASES: [(&str, &str); 67] = [
    (
        "agent_listen_wildcard",
        "listen must not be a wildcard address",
    ),
    (
        "agent_listen_ipv6_wildcard",
        "listen must not be a wildcard address",
    ),
    (
        "agent_listen_public",
        "listen host must be a loopback, 100.64.0.0/10 or RFC1918 literal",
    ),
    (
        "agent_listen_link_local",
        "listen host must be a loopback, 100.64.0.0/10 or RFC1918 literal",
    ),
    (
        "agent_listen_hostname",
        "listen must be a canonical <ip>:<port> literal with a port 1..65535",
    ),
    (
        "agent_listen_port_zero",
        "listen must be a canonical <ip>:<port> literal with a port 1..65535",
    ),
    (
        "agent_endpoint_http",
        "coordinator.endpoint must use https://",
    ),
    (
        "agent_endpoint_rfc1918",
        "coordinator.endpoint host must be a literal 100.64.0.0/10, 127.0.0.0/8 or [::1] address",
    ),
    (
        "agent_endpoint_hostname",
        "coordinator.endpoint host must be a literal 100.64.0.0/10, 127.0.0.0/8 or [::1] address",
    ),
    (
        "agent_endpoint_no_port",
        "coordinator.endpoint needs an explicit port 1..65535 and nothing after it",
    ),
    (
        "agent_principal_is_coordinator",
        "principal equals coordinator.principal",
    ),
    (
        "agent_principal_pattern",
        "principal does not match its pattern",
    ),
    (
        "agent_worker_id_pattern",
        "worker_id does not match its pattern",
    ),
    (
        "agent_retry_horizon_below_work_cap",
        "journal.retry_horizon_ms must be 600000..86400000",
    ),
    (
        "agent_max_operations_zero",
        "journal.max_operations must be 1..1000000",
    ),
    (
        "agent_output_stall_above_lost",
        "limits.output_stall_ms must not exceed limits.lost_after_ms",
    ),
    (
        "agent_default_output_stall_above_lost",
        "limits.output_stall_ms must not exceed limits.lost_after_ms",
    ),
    (
        "agent_heartbeat_not_below_lost",
        "limits.heartbeat_interval_ms must be below limits.lost_after_ms",
    ),
    (
        "agent_output_buffer_above_1mib",
        "limits.output_buffer_bytes must be 1..1048576",
    ),
    (
        "agent_control_quota_zero",
        "limits.control_quota must be 1..4096",
    ),
    ("agent_pem_relative", "ca_file must be absolute"),
    (
        "agent_pem_dotdot",
        "ca_file component does not match its pattern",
    ),
    (
        "agent_pem_in_checkout",
        "cert_file must be outside any repository checkout",
    ),
    (
        "agent_pem_under_state_dir",
        "key_file must not be under the state_dir",
    ),
    (
        "agent_state_dir_in_checkout",
        "state_dir must be outside any repository checkout",
    ),
    (
        "agent_state_dir_is_checkout",
        "state_dir must be outside any repository checkout",
    ),
    (
        "agent_backend_not_loopback",
        "backend base_url host must be a literal 127.0.0.0/8 address or [::1]",
    ),
    ("agent_backend_https", "backend base_url must use http://"),
    (
        "agent_backend_trailing_slash",
        "backend base_url needs an explicit port 1..65535 and nothing after it",
    ),
    (
        "agent_load_endpoint_pattern",
        "backend.load_endpoint does not match its pattern",
    ),
    ("agent_missing_cluster_id", "missing field `cluster_id`"),
    (
        "agent_missing_coordinator_id",
        "missing field `coordinator_id`",
    ),
    ("agent_unknown_key", "unknown field `stream_pool`"),
    ("agent_unknown_limits_key", "unknown field `stream_pool`"),
    (
        "agent_launch_without_lock",
        "backend.launch requires backend.lock_sha256",
    ),
    (
        "agent_lock_without_launch",
        "backend.lock_sha256 requires backend.launch",
    ),
    (
        "agent_lock_uppercase",
        "backend.lock_sha256 must be 64 lowercase hex",
    ),
    (
        "agent_lock_short",
        "backend.lock_sha256 must be 64 lowercase hex",
    ),
    (
        "agent_launch_argv_empty",
        "backend.launch.argv item count must be 1..64",
    ),
    (
        "agent_launch_argv_relative",
        "backend.launch.argv[0] must be an absolute path",
    ),
    (
        "agent_launch_argv_too_many",
        "backend.launch.argv item count must be 1..64",
    ),
    (
        "agent_launch_argv_nul",
        "backend.launch.argv entries must be 1..4096 bytes without NUL",
    ),
    (
        "agent_launch_argv0_nul",
        "backend.launch.argv entries must be 1..4096 bytes without NUL",
    ),
    (
        "agent_launch_argv_too_long",
        "backend.launch.argv entries must be 1..4096 bytes without NUL",
    ),
    (
        "agent_launch_argv_empty_entry",
        "backend.launch.argv entries must be 1..4096 bytes without NUL",
    ),
    (
        "agent_launch_env_too_many",
        "backend.launch.env must have at most 64 entries",
    ),
    (
        "agent_launch_env_key_pattern",
        "backend.launch.env key does not match its pattern",
    ),
    (
        "agent_launch_env_path",
        "backend.launch.env must not set PATH",
    ),
    (
        "agent_launch_env_value_too_long",
        "backend.launch.env values must be at most 4096 bytes without NUL",
    ),
    (
        "agent_launch_env_value_nul",
        "backend.launch.env values must be at most 4096 bytes without NUL",
    ),
    (
        "agent_launch_startup_timeout_low",
        "backend.launch.startup_timeout_ms must be 1000..3600000",
    ),
    (
        "agent_launch_startup_timeout_high",
        "backend.launch.startup_timeout_ms must be 1000..3600000",
    ),
    (
        "agent_launch_stop_timeout_low",
        "backend.launch.stop_timeout_ms must be 100..600000",
    ),
    (
        "agent_launch_stop_timeout_high",
        "backend.launch.stop_timeout_ms must be 100..600000",
    ),
    (
        "agent_launch_uid_zero",
        "backend.launch.uid must be 1..4294967294",
    ),
    (
        "agent_launch_uid_max",
        "backend.launch.uid must be 1..4294967294",
    ),
    (
        "agent_launch_gid_zero",
        "backend.launch.gid must be 1..4294967294",
    ),
    (
        "agent_launch_gid_max",
        "backend.launch.gid must be 1..4294967294",
    ),
    (
        "agent_launch_uid_without_gid",
        "backend.launch.uid and backend.launch.gid must be set together",
    ),
    (
        "agent_launch_gid_without_uid",
        "backend.launch.uid and backend.launch.gid must be set together",
    ),
    ("agent_launch_unknown_key", "unknown field `cwd`"),
    ("agent_owner_pattern", "owner does not match its pattern"),
    (
        "agent_owner_smoke_scope",
        "owner must not be the remote readiness smoke scope",
    ),
    (
        "agent_owner_only_without_owner",
        "owner_only needs an owner",
    ),
    (
        "agent_owner_only_empty",
        "owner_only item count must be 1..16",
    ),
    (
        "agent_owner_only_bad_time",
        "times are HH:MM (00:00..24:00)",
    ),
    ("agent_owner_only_unknown_zone", "Mars/Olympus"),
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// RFC 7396 JSON merge patch; `${REPO_ROOT}` in strings becomes the repository root.
fn merge_patch(target: Value, patch: &Value) -> Value {
    match patch {
        Value::Object(patch) => {
            let mut out = match target {
                Value::Object(map) => map,
                _ => serde_json::Map::new(),
            };
            for (key, value) in patch {
                if value.is_null() {
                    out.remove(key);
                } else {
                    let current = out.remove(key).unwrap_or(Value::Null);
                    out.insert(key.clone(), merge_patch(current, value));
                }
            }
            Value::Object(out)
        }
        Value::String(s) => Value::String(s.replace("${REPO_ROOT}", root().to_str().unwrap())),
        other => other.clone(),
    }
}

fn patched_toml(base: &str, patch: &Value) -> String {
    let text = std::fs::read_to_string(root().join(base)).unwrap();
    let table: toml::Table = toml::from_str(&text).unwrap();
    toml::to_string(&merge_patch(serde_json::to_value(table).unwrap(), patch)).unwrap()
}

fn runtime(base: &str, patch: &Value) -> Result<RuntimeConfig, String> {
    RuntimeConfig::from_toml_str(&patched_toml(base, patch), &root())
        .map_err(|e| e.message().to_owned())
}

/// One shared-table runtime case: `patch`, then `entry_patch` merged into `deployments[0]` (M5 D4). A top-level
/// `backend` key in `patch` (the pre-M5 shape) is refused.
fn runtime_case(case: &Value) -> Result<RuntimeConfig, String> {
    let name = case["name"].as_str().unwrap();
    assert!(
        case["patch"].get("backend").is_none(),
        "case {name}: patch.backend is the pre-M5 shape; use entry_patch"
    );
    let text = std::fs::read_to_string(root().join(case["base"].as_str().unwrap())).unwrap();
    let table: toml::Table = toml::from_str(&text).unwrap();
    let mut value = merge_patch(serde_json::to_value(table).unwrap(), &case["patch"]);
    if let Some(entry) = case.get("entry_patch") {
        let first = &mut value["deployments"][0];
        *first = merge_patch(first.take(), entry);
    }
    RuntimeConfig::from_toml_str(&toml::to_string(&value).unwrap(), &root())
        .map_err(|e| e.message().to_owned())
}

#[test]
#[should_panic(expected = "patch.backend is the pre-M5 shape")]
fn runtime_case_with_patch_backend_is_refused() {
    let _ = runtime_case(&json!({
        "name": "old_shape", "base": LIVE, "patch": {"backend": {"base_url": null}}
    }));
}

fn agent(patch: &Value) -> Result<AgentConfig, String> {
    agent_from(AGENT, patch)
}

fn agent_from(base: &str, patch: &Value) -> Result<AgentConfig, String> {
    AgentConfig::from_toml_str(&patched_toml(base, patch), Some(&root()))
        .map_err(|e| e.message().to_owned())
}

fn cases(fixture: &str) -> Vec<Value> {
    serde_json::from_str(&std::fs::read_to_string(root().join(fixture)).unwrap()).unwrap()
}

/// A scratch directory removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "cere-config-federation-{}-{tag}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn federation_profile_loads() {
    let config = RuntimeConfig::load(root().join(FEDERATION)).unwrap();
    let fed = config.federation.as_ref().expect("[federation]");
    assert_eq!(
        (
            fed.cluster_id.as_str(),
            fed.coordinator_id.as_str(),
            fed.principal.as_str()
        ),
        ("lab-cere", "cere-coordinator", "cere-coordinator")
    );
    assert_eq!(fed.listen, "100.72.2.2:7443".parse::<SocketAddr>().unwrap());
    assert_eq!(
        fed.ca_file,
        Path::new("/tmp/cere-fed-PLACEHOLDER/pki/ca.pem")
    );
    assert_eq!(fed.workers.len(), 1);
    let worker = &fed.workers[0];
    assert_eq!(
        (
            worker.worker_id.as_str(),
            worker.principal.as_str(),
            worker.max_inflight
        ),
        ("worker-a", "cere-worker-a", 4)
    );
    assert_eq!(
        worker.endpoint_addr(),
        "100.71.1.1:7443".parse::<SocketAddr>().unwrap()
    );
    assert_eq!(
        (
            worker.expect.served_model_name.as_str(),
            worker.expect.tp_size,
            worker.expect.kv_cache_dtype.as_str()
        ),
        ("Qwen3.8-Flash-Next", 1, "fp8_e4m3")
    );
    assert_eq!(
        config.auth.operator_token_env.as_deref(),
        Some("CERE_OPERATOR_TOKEN")
    );
}

#[test]
fn configs_without_federation_have_none() {
    let live = RuntimeConfig::load(root().join(LIVE)).unwrap();
    assert!(live.federation.is_none());
    assert!(live.auth.operator_token_env.is_none());
}

/// The federation profile with every optional timing and the quota left out.
fn without_timings() -> Value {
    json!({"federation": {
        "heartbeat_interval_ms": null, "suspect_after_ms": null, "lost_after_ms": null,
        "admission_lease_ms": null, "rpc_timeout_ms": null, "rpc_quota_per_principal": null}})
}

#[test]
fn federation_timings_default_to_dev_tailnet() {
    let fed = runtime(FEDERATION, &without_timings())
        .unwrap()
        .federation
        .unwrap();
    let ms = Duration::from_millis;
    assert_eq!(
        (
            fed.heartbeat_interval(),
            fed.suspect_after(),
            fed.lost_after(),
            fed.admission_lease(),
            fed.rpc_timeout()
        ),
        (ms(1000), ms(5000), ms(15_000), ms(4000), ms(5000))
    );
    assert_eq!(fed.rpc_quota(), 64);
}

/// `config_hash` is the digest of the TOML as written (`canonical_digest` in `scripts/cere_contracts.py`): a
/// defaulted federation timing is not serialized, so Rust and Python agree on configs that leave it out.
#[test]
fn config_hash_covers_the_federation_toml_as_written() {
    let text = patched_toml(FEDERATION, &without_timings());
    let config = RuntimeConfig::from_toml_str(&text, &root()).unwrap();
    let table: toml::Table = toml::from_str(&text).unwrap();
    let canonical = serde_json::to_string(&serde_json::to_value(table).unwrap()).unwrap();
    assert_eq!(
        config.config_hash(),
        format!("{:x}", Sha256::digest(canonical.as_bytes()))
    );
}

#[test]
fn listen_accepts_tailnet_and_loopback_literals() {
    for listen in [
        "100.64.0.1:7443",
        "100.127.255.254:7443",
        "127.0.0.1:7443",
        "127.8.9.10:1",
        "[::1]:65535",
    ] {
        let config = runtime(FEDERATION, &json!({"federation": {"listen": listen}}))
            .unwrap_or_else(|e| panic!("{listen}: {e}"));
        assert_eq!(
            config.federation.unwrap().listen,
            listen.parse::<SocketAddr>().unwrap()
        );
    }
}

#[test]
fn endpoint_accepts_tailnet_and_loopback_literals() {
    for (endpoint, addr) in [
        ("https://100.64.0.1:1", "100.64.0.1:1"),
        ("https://127.0.0.1:7443", "127.0.0.1:7443"),
        ("https://[::1]:7443", "[::1]:7443"),
    ] {
        let worker = json!({"worker_id": "worker-a", "endpoint": endpoint, "principal": "cere-worker-a",
            "max_inflight": 4, "expect": {"served_model_name": "Qwen3.8-Flash-Next", "tp_size": 1,
            "kv_cache_dtype": "fp8_e4m3"}});
        let config = runtime(FEDERATION, &json!({"federation": {"workers": [worker]}}))
            .unwrap_or_else(|e| panic!("{endpoint}: {e}"));
        let fed = config.federation.unwrap();
        assert_eq!(
            fed.workers[0].endpoint_addr(),
            addr.parse::<SocketAddr>().unwrap()
        );
    }
}

#[test]
fn global_cap_counts_local_and_remote_workers() {
    // 2 workers x min(4, 4) = 8 is the committed cap; a smaller remote max_inflight lowers the per-worker cap.
    assert!(runtime(FEDERATION, &json!({})).is_ok());
    let worker = json!({"worker_id": "worker-a", "endpoint": "https://100.71.1.1:7443",
        "principal": "cere-worker-a", "max_inflight": 3, "expect": {"served_model_name": "Qwen3.8-Flash-Next",
        "tp_size": 1, "kv_cache_dtype": "fp8_e4m3"}});
    let ok = json!({"admission": {"max_backend_inflight": 6, "per_scope_inflight": 6},
        "federation": {"workers": [worker]}});
    assert!(runtime(FEDERATION, &ok).is_ok());
}

/// With `[federation]` a request body must fit one SubmitWork under `wire::INFERENCE_MESSAGE_MAX` with a 64 KiB
/// envelope; without it the 8 MiB body the committed configs use stays valid.
#[test]
fn federation_bounds_the_request_body_by_the_remote_message_cap() {
    let body = |n: u64| json!({"api": {"max_request_body_bytes": n}});
    assert_eq!(
        cere_runtime::config::REMOTE_BODY_MAX,
        cere_runtime::fed::wire::INFERENCE_MESSAGE_MAX as u64 - 65_536
    );
    assert!(runtime(FEDERATION, &body(8_323_072)).is_ok());
    let err = runtime(FEDERATION, &body(8_323_073)).unwrap_err();
    assert!(
        err.contains("api.max_request_body_bytes must be <= 8323072 with [federation]"),
        "{err}"
    );
    assert!(runtime(LIVE, &body(8_388_608)).is_ok());
}

#[test]
fn refused_runtime_cases_hit_their_rule() {
    let table = cases("tests/fixtures/runtime_config_invalid.json");
    let named: Vec<&str> = table
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .filter(|n| n.starts_with("fed_") || n.starts_with("operator_"))
        .collect();
    assert_eq!(named, RUNTIME_CASES.map(|(n, _)| n));
    for (name, needle) in RUNTIME_CASES {
        let case = table.iter().find(|c| c["name"] == name).unwrap();
        let err = runtime_case(case).err();
        let err = err.unwrap_or_else(|| panic!("{name} accepted"));
        assert!(err.contains(needle), "{name}: {err}");
    }
}

#[test]
fn operator_token_resolves_from_the_environment() {
    let config = RuntimeConfig::load(root().join(FEDERATION)).unwrap();
    let env = |pairs: &'static [(&'static str, &'static str)]| {
        move |name: &str| {
            pairs
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| OsString::from(v))
        }
    };
    let token = config
        .resolve_operator_token_with(env(&[
            ("CERE_TOKEN_DEV", "client"),
            ("CERE_OPERATOR_TOKEN", "op"),
        ]))
        .unwrap()
        .expect("configured");
    assert_eq!(token.expose(), "op");
    assert!(!format!("{token:?}").contains("op\""));
    let missing = config
        .resolve_operator_token_with(env(&[("CERE_TOKEN_DEV", "client")]))
        .unwrap_err();
    assert!(
        missing.message().contains("CERE_OPERATOR_TOKEN"),
        "{missing}"
    );
    let empty = config
        .resolve_operator_token_with(env(&[
            ("CERE_TOKEN_DEV", "client"),
            ("CERE_OPERATOR_TOKEN", ""),
        ]))
        .unwrap_err();
    assert!(empty.message().contains("missing or empty"), "{empty}");
    let shared = config
        .resolve_operator_token_with(env(&[
            ("CERE_TOKEN_DEV", "same"),
            ("CERE_OPERATOR_TOKEN", "same"),
        ]))
        .unwrap_err();
    assert!(
        shared
            .message()
            .contains("operator token equals a scope token"),
        "{shared}"
    );
    // Absent key: the operator routes are disabled, whatever the environment holds.
    let live = RuntimeConfig::load(root().join(LIVE)).unwrap();
    assert!(live
        .resolve_operator_token_with(env(&[
            ("CERE_TOKEN_DEV", "client"),
            ("CERE_OPERATOR_TOKEN", "op")
        ]))
        .unwrap()
        .is_none());
}

#[test]
fn agent_profile_loads() {
    let config = AgentConfig::load(&root().join(AGENT)).unwrap();
    assert_eq!(
        (
            config.cluster_id.as_str(),
            config.worker_id.as_str(),
            config.principal.as_str()
        ),
        ("lab-cere", "worker-a", "cere-worker-a")
    );
    assert_eq!(
        config.listen,
        "100.71.1.1:7443".parse::<SocketAddr>().unwrap()
    );
    assert_eq!(config.state_dir, Path::new("/var/lib/cere-worker/worker-a"));
    assert_eq!(
        config.key_file,
        Path::new("/tmp/cere-fed-PLACEHOLDER/pki/worker-a-key.pem")
    );
    assert_eq!(
        (
            config.coordinator.endpoint.as_str(),
            config.coordinator.principal.as_str(),
            config.coordinator.coordinator_id.as_str()
        ),
        (
            "https://100.72.2.2:7443",
            "cere-coordinator",
            "cere-coordinator"
        )
    );
    assert_eq!(
        config.coordinator.endpoint_addr(),
        "100.72.2.2:7443".parse::<SocketAddr>().unwrap()
    );
    assert_eq!(config.backend.base_url, "http://127.0.0.1:18317");
    assert_eq!(config.backend.load_endpoint.as_deref(), Some("/get_load"));
    assert_eq!(config.backend.served_model_name, "Qwen3.8-Flash-Next");
    assert_eq!(
        (
            config.journal.max_operations,
            config.journal.retry_horizon_ms
        ),
        (4096, 600_000)
    );
    let l = &config.limits;
    assert_eq!(
        (
            l.heartbeat_interval_ms,
            l.lost_after_ms,
            l.rpc_timeout_ms,
            l.control_quota,
            l.output_buffer_bytes,
            l.output_stall_ms
        ),
        (1000, 15_000, 5000, 64, 1 << 20, 5000)
    );
}

#[test]
fn agent_journal_and_limits_default() {
    let config = agent(&json!({"journal": null, "limits": null})).unwrap();
    assert_eq!(
        (
            config.journal.max_operations,
            config.journal.retry_horizon_ms
        ),
        (4096, 600_000)
    );
    let l = &config.limits;
    assert_eq!(
        (
            l.heartbeat_interval_ms,
            l.lost_after_ms,
            l.rpc_timeout_ms,
            l.control_quota,
            l.output_buffer_bytes,
            l.output_stall_ms
        ),
        (1000, 15_000, 5000, 64, 1 << 20, 5000)
    );
    // output_stall_ms follows rpc_timeout_ms when absent (Q14).
    let config =
        agent(&json!({"limits": {"output_stall_ms": null, "rpc_timeout_ms": 7000}})).unwrap();
    assert_eq!(config.limits.output_stall_ms, 7000);
}

#[test]
fn agent_listen_accepts_loopback_tailnet_and_rfc1918_literals() {
    for listen in [
        "127.0.0.1:7443",
        "[::1]:7443",
        "100.64.0.1:7443",
        "10.1.2.3:7443",
        "172.16.0.1:7443",
        "172.31.255.254:7443",
        "192.168.1.10:7443",
    ] {
        let config = agent(&json!({"listen": listen})).unwrap_or_else(|e| panic!("{listen}: {e}"));
        assert_eq!(config.listen, listen.parse::<SocketAddr>().unwrap());
    }
}

#[test]
fn refused_agent_cases_hit_their_rule() {
    let table = cases("tests/fixtures/worker_agent_config_invalid.json");
    let named: Vec<&str> = table.iter().map(|c| c["name"].as_str().unwrap()).collect();
    assert_eq!(named, AGENT_CASES.map(|(n, _)| n));
    for (name, needle) in AGENT_CASES {
        let case = table.iter().find(|c| c["name"] == name).unwrap();
        // `[backend.launch]` rows patch the managed profile (m36 design note Q13), every other row the mock one.
        let base = if name.starts_with("agent_launch_") || name.starts_with("agent_lock_") {
            MANAGED
        } else {
            AGENT
        };
        assert_eq!(case["base"], base, "{name}");
        let err = agent_from(base, &case["patch"])
            .err()
            .unwrap_or_else(|| panic!("{name} accepted"));
        assert!(err.contains(needle), "{name}: {err}");
    }
}

#[test]
fn agent_config_outside_any_checkout_is_accepted() {
    // `cere-worker` needs no checkout: with no PACKAGE_VERSION above the config, no repo-root bound applies, so
    // even a state_dir under this repository passes (the operator's choice on a worker host).
    let scratch = Scratch::new("outside");
    let state_dir = root().join("agent-state");
    let path = scratch.0.join("agent.toml");
    std::fs::write(
        &path,
        patched_toml(AGENT, &json!({"state_dir": state_dir.to_str().unwrap()})),
    )
    .unwrap();
    let config = AgentConfig::load(&path).unwrap();
    assert_eq!(config.state_dir, state_dir);
}

#[test]
fn agent_config_inside_a_checkout_bounds_state_dir() {
    // The nearest ancestor holding PACKAGE_VERSION is the checkout: state_dir and PEMs must be outside it.
    let scratch = Scratch::new("inside");
    std::fs::write(scratch.0.join("PACKAGE_VERSION"), "0.0.0\n").unwrap();
    std::fs::create_dir_all(scratch.0.join("configs/worker")).unwrap();
    let path = scratch.0.join("configs/worker/agent.toml");
    let checkout = scratch.0.canonicalize().unwrap();
    for (patch, needle) in [
        (
            json!({"state_dir": checkout.join("state").to_str().unwrap()}),
            "state_dir must be outside any repository checkout",
        ),
        (
            json!({"ca_file": checkout.join("ca.pem").to_str().unwrap()}),
            "ca_file must be outside any repository checkout",
        ),
    ] {
        std::fs::write(&path, patched_toml(AGENT, &patch)).unwrap();
        let err = AgentConfig::load(&path).unwrap_err();
        assert!(err.message().contains(needle), "{err}");
    }
    std::fs::write(&path, patched_toml(AGENT, &json!({}))).unwrap();
    AgentConfig::load(&path).unwrap();
}

#[test]
fn agent_check_config_never_opens_the_pems() {
    // The committed placeholders do not exist; validation is shape-only (Q16).
    assert!(!Path::new("/tmp/cere-fed-PLACEHOLDER").exists());
    let config = AgentConfig::load(&root().join(AGENT)).unwrap();
    config.validate(Some(&root())).unwrap();
    config.validate(None).unwrap();
}

#[test]
fn agent_load_reports_an_unreadable_file() {
    let err = AgentConfig::load(Path::new("/nonexistent/cere/agent.toml")).unwrap_err();
    assert!(
        err.message()
            .contains("cannot read /nonexistent/cere/agent.toml"),
        "{err}"
    );
}

#[test]
fn managed_profile_loads_with_launch_defaults() {
    // m36 design note Q3, Q13: the committed managed profile pins the federation-dev deployment lock and launches
    // the mock on the base_url port with the operator's server-info file, as uid/gid 65534.
    let config = AgentConfig::load(&root().join(MANAGED)).unwrap();
    let lock =
        std::fs::read(root().join("configs/deployments/qwen38-flash-next.lock.json")).unwrap();
    let digest: [u8; 32] = Sha256::digest(&lock).into();
    assert_eq!(config.backend.lock_bytes(), Some(digest));
    let launch = config.backend.launch.as_ref().unwrap();
    assert_eq!(
        launch.argv[0],
        "/var/lib/cere/fed-PLACEHOLDER/bin/cere-mock-sglang"
    );
    let port = config.backend.base_url.rsplit(':').next().unwrap();
    let flag = |name: &str| {
        let at = launch.argv.iter().position(|a| a == name).unwrap();
        launch.argv[at + 1].as_str()
    };
    assert_eq!(flag("--port"), port);
    assert_eq!(
        flag("--server-info"),
        "/var/lib/cere/fed-PLACEHOLDER/server-info.json"
    );
    assert!(launch.env.is_empty());
    assert_eq!(
        (launch.startup_timeout_ms, launch.stop_timeout_ms),
        (600_000, 30_000)
    );
    assert_eq!((launch.uid, launch.gid), (Some(65534), Some(65534)));
    // Attached mode (the mock profile) has neither a lock nor a launch table (Q7).
    let attached = AgentConfig::load(&root().join(AGENT)).unwrap();
    assert_eq!(attached.backend.lock_bytes(), None);
    assert!(attached.backend.launch.is_none());
}

#[test]
fn gemma_agent_profiles_load() {
    // n3 design note D5, C3: the managed Gemma profile pins the federation-gemma deployment lock and launches SGLang
    // on its base_url host/port under its served_model_name as 65534, without PATH (the agent fixes it).
    let config =
        AgentConfig::load(&root().join("configs/worker/node-b-gemma-managed.toml")).unwrap();
    let lock = std::fs::read(root().join("configs/deployments/gemma-4-e4b-it.lock.json")).unwrap();
    let digest: [u8; 32] = Sha256::digest(&lock).into();
    assert_eq!(config.backend.lock_bytes(), Some(digest));
    let launch = config.backend.launch.as_ref().unwrap();
    assert_eq!(launch.argv[0], "/opt/sglang/bin/python3");
    let flag = |name: &str| {
        let at = launch.argv.iter().position(|a| a == name).unwrap();
        launch.argv[at + 1].as_str()
    };
    let url = format!("http://{}:{}", flag("--host"), flag("--port"));
    assert_eq!(config.backend.base_url, url);
    assert_eq!(
        flag("--served-model-name"),
        config.backend.served_model_name
    );
    assert!(!launch.env.contains_key("PATH"));
    assert_eq!(launch.env["HOME"], "/tmp/cere-sglang-home");
    assert_eq!(
        (launch.startup_timeout_ms, launch.stop_timeout_ms),
        (900_000, 30_000)
    );
    assert_eq!((launch.uid, launch.gid), (Some(65534), Some(65534)));
    let attached = AgentConfig::load(&root().join("configs/worker/node-b-gemma.toml")).unwrap();
    assert_eq!(attached.backend.base_url, config.backend.base_url);
    assert_eq!(
        attached.backend.served_model_name,
        config.backend.served_model_name
    );
    assert_eq!(attached.backend.lock_bytes(), None);
    assert!(attached.backend.launch.is_none());
}

#[test]
fn managed_launch_bounds_are_inclusive() {
    let long = "a".repeat(4096);
    let argv: Vec<String> = std::iter::once("/bin/true".to_owned())
        .chain((1..64).map(|_| long.clone()))
        .collect();
    let env: serde_json::Map<String, Value> =
        (0..64).map(|i| (format!("_V{i}"), json!(long))).collect();
    let config = agent_from(
        MANAGED,
        &json!({"backend": {"launch": {
            "argv": argv, "env": env, "startup_timeout_ms": 1000, "stop_timeout_ms": 600_000,
            "uid": 4_294_967_294u32, "gid": 1,
        }}}),
    )
    .unwrap();
    let launch = config.backend.launch.unwrap();
    assert_eq!((launch.argv.len(), launch.env.len()), (64, 64));
    agent_from(
        MANAGED,
        &json!({"backend": {"launch": {"startup_timeout_ms": 3_600_000, "stop_timeout_ms": 100}}}),
    )
    .unwrap();
}

#[test]
fn launch_privileges_rule() {
    // m36 design note Q3: a root agent must drop the child to an explicit uid and gid; a non-root agent cannot
    // drop to any (`setgroups` needs root), so it is refused with either set and the child inherits its credentials.
    let config = AgentConfig::load(&root().join(MANAGED)).unwrap();
    let both = config.backend.launch.unwrap();
    both.check_privileges(0).unwrap();
    let err = both.check_privileges(1000).unwrap_err();
    assert!(
        err.message().contains(
            "backend.launch.uid and backend.launch.gid must be unset when cere-worker does not run as root"
        ),
        "{err}"
    );
    let mut neither = both.clone();
    (neither.uid, neither.gid) = (None, None);
    let err = neither.check_privileges(0).unwrap_err();
    assert!(
        err.message().contains(
            "backend.launch.uid and backend.launch.gid are required when cere-worker runs as root"
        ),
        "{err}"
    );
    neither.check_privileges(1000).unwrap();
    for (uid, gid) in [(Some(65534), None), (None, Some(65534))] {
        let mut half = both.clone();
        (half.uid, half.gid) = (uid, gid);
        assert!(half.check_privileges(0).is_err());
        assert!(half.check_privileges(1000).is_err());
    }
}
