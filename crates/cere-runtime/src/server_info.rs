//! `GET /get_server_info` of `cered` (server-info design note SI-01..SI-06): SGLang's server-info route, answered
//! with `cered`'s own configuration, so a client that vets an SGLang server before it sends a prompt (Chief's
//! `local-sglang` provider: served model, listen address, request logging, crash dumps, prompt export) can vet CERE
//! the same way. Every value is read from the running process (the startup config, the bound listener, the
//! registries); none is a constant chosen to pass a check.
//!
//! The retention keys use SGLang's names and meanings (`log_requests`: request content written to a log;
//! `crash_dump_folder`: requests dumped on a crash; the [`EXPORT_KEYS`]: prompt-derived data exported or kept
//! outside the serving path). `cered` writes metadata-only lines and journals ([`crate::journal`], the access and
//! request lines, [`crate::usage`]); `telemetry` must be `metadata` without `raw_content` ([`crate::config`]), so
//! content is never logged. A worker `cered` does not run in process (a `sglang_http` backend, a federation worker)
//! has retention settings `cered` cannot see: then a key whose `cered` value is "none" is not reported as none, it is
//! `null` (`log_requests`) or absent (every other retention key), and `cere.unverified_workers` names the workers.

use std::net::SocketAddr;

use serde_json::{json, Map, Value};

use crate::app::{App, Backend};

/// The SGLang 0.5.19 launch keys that keep or export prompt-derived data (SGLang reports `null`/`false` when off).
pub const EXPORT_KEYS: [&str; 8] = [
    "kv_events",
    "kv_events_config",
    "hicache_storage_backend",
    "debug_tensor_dump_output_folder",
    "enable_trace",
    "log_requests_target",
    "spec_trace_dir",
    "msprobe_dump_config",
];

/// Where research captures leave the process (WeightLab, `[deployments.native.research]`): activations of the
/// bound requests, drained over HTTP by the operator.
const RESEARCH_CAPTURES: &str = "/v1/cere/research/captures";

/// The `GET /get_server_info` body of `app` serving on `listen`.
pub(crate) fn report(app: &App, listen: SocketAddr) -> Value {
    let config = &app.config;
    let served = app.served();
    let unverified = unverified_workers(app);
    let verified = unverified.is_empty();
    let raw_content = config.telemetry.raw_content;

    let mut info = Map::new();
    // As `cered --version` prints it.
    info.insert(
        "version".into(),
        json!(format!(
            "cered {}; package v{}",
            env!("CARGO_PKG_VERSION"),
            include_str!("../../../PACKAGE_VERSION").trim()
        )),
    );
    // One served model: its name, as SGLang reports it; several (M5): every name, in config order.
    let names: Vec<&str> = served.iter().map(|s| s.model.as_str()).collect();
    info.insert(
        "served_model_name".into(),
        match names.as_slice() {
            [one] => json!(one),
            many => json!(many),
        },
    );
    info.insert(
        "context_length".into(),
        match served {
            [one] => json!(one.max_context_tokens),
            _ => Value::Null,
        },
    );
    info.insert(
        "max_running_requests".into(),
        json!(app.settings().config.admission.max_backend_inflight),
    );
    info.insert("host".into(), json!(listen.ip().to_string()));
    info.insert("port".into(), json!(listen.port()));

    // `cered`'s own value of a retention key, or absent when it is "none" and an unverified worker may differ.
    let mut retention = |key: &str, own: Value| {
        let none = own.is_null() || own == Value::Bool(false);
        if verified || !none {
            info.insert(key.into(), own);
        }
    };
    retention("crash_dump_folder", Value::Null);
    retention(
        "log_requests_target",
        json!(raw_content.then_some("stderr")),
    );
    retention("kv_events", Value::Null);
    retention("kv_events_config", Value::Null);
    retention(
        "hicache_storage_backend",
        json!(config.state.nvme_spill_enabled.then_some("nvme")),
    );
    retention(
        "debug_tensor_dump_output_folder",
        json!(app.research.is_some().then_some(RESEARCH_CAPTURES)),
    );
    retention("enable_trace", json!(false));
    retention("spec_trace_dir", Value::Null);
    retention("msprobe_dump_config", Value::Null);
    info.insert(
        "log_requests".into(),
        if raw_content || verified {
            json!(raw_content)
        } else {
            Value::Null
        },
    );

    info.insert(
        "cere".into(),
        json!({
            "config_hash": app.config_hash,
            "telemetry_level": config.telemetry.level,
            "raw_content": raw_content,
            "usage_journal": app.usage.is_some(),
            "host_spill_enabled": config.state.host_spill_enabled,
            "research": app.research.is_some(),
            "federation_listen": app.federation_addr().map(|a| a.to_string()),
            "unverified_workers": unverified,
        }),
    );
    Value::Object(info)
}

/// The workers whose retention settings `cered` cannot see, as `<worker_id>: <kind>[ <base_url>]`.
fn unverified_workers(app: &App) -> Vec<String> {
    let local = app
        .replicas()
        .iter()
        .filter_map(|replica| match &replica.backend {
            Backend::Native(_) => None,
            Backend::SglangHttp(_) => Some(format!(
                "{}: sglang_http {}",
                replica.worker_id(),
                replica.base_url()
            )),
            Backend::Remote(_) => Some(format!("{}: remote", replica.worker_id())),
        });
    let remote = app
        .remotes()
        .iter()
        .map(|r| format!("{}: remote", r.worker_id()));
    local.chain(remote).collect()
}
