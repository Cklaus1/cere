//! Backend identity (m35 design note Q5, Q10a): the agent probes its loopback backend (`GET /get_server_info` and
//! `GET /v1/models`, as `readiness` does) every heartbeat interval in a task of its own, with a probe timeout of one
//! heartbeat interval, so the heartbeat never waits on it.
//!
//! A probe **confirms** an identity when the backend answers both endpoints, `/get_server_info` names the
//! configured `served_model_name`, `/v1/models` lists it, and the pins (`tp_size`, `kv_cache_dtype`,
//! `max_running_requests`) and the launch (`random_seed`, `startup_time`) are present. Its `instance_id` is
//! `"sglang-"` + the first 8 bytes (lowercase hex) of the SHA-256 over the canonical JSON of `{base_url,
//! served_model_name, pins, launch}`, where `launch` is the lowercase hex SHA-256 of the canonical JSON array
//! `[random_seed, startup_time]` (the `readiness` launch rule). A relaunched backend therefore has a new
//! `instance_id` and a probe outage followed by the same backend has the same one; what the agent does with a
//! change (the generation bump and its fence) lives in [`crate::agent::node::Shared`].
//!
//! Managed mode (m36 design note Q5, design-m36-agent.md §3.6): the probe runs only while the model is
//! `loading`, `ready` or `draining`, and a backend naming another served model fails a load at once
//! ([`ProbeError::NameMismatch`]).

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::sync::watch;
use tokio::time::MissedTickBehavior;

use crate::agent::node::{stopped, Shared};
use crate::digest::{canonical_json, hex};
use crate::sglang::{SglangHttp, Timeouts};

/// Longest identifier string the inventory carries (`schemas/worker_inventory.schema.json` `maxLength`).
const NAME_MAX: usize = 256;

/// One confirmed backend identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendIdentity {
    pub instance_id: String,
    pub served_model_name: String,
    pub tp_size: u64,
    pub kv_cache_dtype: String,
    /// The backend's live-op bound (Q14; harness revision H1).
    pub max_running_requests: u64,
    /// `/get_server_info` `tool_call_parser` (`None`: absent or null). Not part of the instance id: the agent
    /// refuses a tools Submit unless it is `qwen3_coder` (M2.4 D1, M24 Z2 R1-06).
    pub tool_call_parser: Option<String>,
}

/// The probe client: every call (connect and exchange) bounded by one heartbeat interval.
pub fn probe_client(base_url: &str, heartbeat_interval: Duration) -> SglangHttp {
    SglangHttp::new(
        base_url,
        Timeouts {
            connect: heartbeat_interval,
            control: heartbeat_interval,
            stream_idle: heartbeat_interval,
        },
    )
    .expect("the backend base_url is validated at config load")
}

/// Why a probe confirmed nothing (never contains prompt text).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeError {
    /// `/get_server_info` names another `served_model_name`: in managed mode a load fails at once (§3.6).
    NameMismatch(String),
    /// Unreachable, incomplete or not listed; the next interval probes again.
    Unconfirmed(String),
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProbeError::NameMismatch(s) | ProbeError::Unconfirmed(s) => f.write_str(s),
        }
    }
}

impl From<String> for ProbeError {
    fn from(problem: String) -> ProbeError {
        ProbeError::Unconfirmed(problem)
    }
}

impl From<&str> for ProbeError {
    fn from(problem: &str) -> ProbeError {
        ProbeError::Unconfirmed(problem.to_owned())
    }
}

/// The probe task: one probe per heartbeat interval, bounded by that interval; each confirmed identity goes to
/// [`Shared::confirm`]. An unconfirmed probe changes nothing (the identity and generation are kept) and is logged
/// once per distinct reason. In managed mode an interval without a model to probe ([`Shared::probe_wanted`]) is
/// skipped and a name mismatch goes to [`Shared::name_mismatch`].
pub(crate) async fn run(shared: Arc<Shared>, client: SglangHttp, stop: watch::Receiver<bool>) {
    let interval = Duration::from_millis(shared.config.limits.heartbeat_interval_ms);
    let mut tick = tokio::time::interval(interval);
    tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let stop = stopped(stop);
    tokio::pin!(stop);
    let mut last_problem: Option<String> = None;
    loop {
        tokio::select! {
            () = &mut stop => return,
            _ = tick.tick() => {}
        }
        if !shared.probe_wanted() {
            continue;
        }
        let outcome = tokio::select! {
            () = &mut stop => return,
            outcome = tokio::time::timeout(
                interval,
                probe(&client, &shared.config.backend.served_model_name),
            ) => outcome,
        };
        match outcome.unwrap_or_else(|_| Err("probe timed out".into())) {
            Ok(identity) => {
                last_problem = None;
                shared.confirm(identity);
            }
            Err(problem) => {
                if matches!(problem, ProbeError::NameMismatch(_)) {
                    shared.name_mismatch().await;
                }
                let problem = problem.to_string();
                if last_problem.as_deref() != Some(problem.as_str()) {
                    eprintln!("cere-worker: backend identity not confirmed: {problem}");
                    last_problem = Some(problem);
                }
            }
        }
    }
}

/// Probes the backend once; `Err` names why the identity is not confirmed (never contains prompt text).
pub async fn probe(
    client: &SglangHttp,
    served_model_name: &str,
) -> Result<BackendIdentity, ProbeError> {
    let info = client
        .server_info()
        .await
        .map_err(|e| format!("/get_server_info: {e}"))?;
    let models = client
        .models()
        .await
        .map_err(|e| format!("/v1/models: {e}"))?;
    identify(client.base_url(), served_model_name, &info, &models)
}

/// The identity rule over one `/get_server_info` body and the `/v1/models` ids.
pub fn identify(
    base_url: &str,
    served_model_name: &str,
    info: &Value,
    models: &[String],
) -> Result<BackendIdentity, ProbeError> {
    let reported = info
        .get("served_model_name")
        .and_then(Value::as_str)
        .ok_or("/get_server_info without served_model_name")?;
    if reported != served_model_name {
        return Err(ProbeError::NameMismatch(format!(
            "backend serves {reported:?}, configured {served_model_name:?}"
        )));
    }
    if !models.iter().any(|m| m == served_model_name) {
        return Err(format!("/v1/models does not list {served_model_name:?}").into());
    }
    let positive = |key: &str| {
        info.get(key)
            .and_then(Value::as_u64)
            .filter(|&n| n >= 1)
            .ok_or_else(|| format!("/get_server_info without a positive integer {key}"))
    };
    let tp_size = positive("tp_size")?;
    let max_running_requests = positive("max_running_requests")?;
    let kv_cache_dtype = info
        .get("kv_cache_dtype")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty() && s.len() <= NAME_MAX)
        .ok_or("/get_server_info without a kv_cache_dtype")?
        .to_owned();
    if served_model_name.is_empty() || served_model_name.len() > NAME_MAX {
        return Err("served_model_name must be 1..=256 bytes".into());
    }
    let present = |key: &str| info.get(key).filter(|v| !v.is_null());
    let (Some(seed), Some(startup)) = (present("random_seed"), present("startup_time")) else {
        return Err("/get_server_info without random_seed and startup_time".into());
    };
    let launch = hex(&Sha256::digest(
        canonical_json(&json!([seed, startup])).as_bytes(),
    ));
    let identity = json!({
        "base_url": base_url,
        "served_model_name": served_model_name,
        "pins": {
            "tp_size": tp_size,
            "kv_cache_dtype": kv_cache_dtype,
            "max_running_requests": max_running_requests,
        },
        "launch": launch,
    });
    let digest = Sha256::digest(canonical_json(&identity).as_bytes());
    Ok(BackendIdentity {
        instance_id: format!("sglang-{}", hex(&digest[..8])),
        served_model_name: served_model_name.to_owned(),
        tp_size,
        kv_cache_dtype,
        max_running_requests,
        tool_call_parser: info
            .get("tool_call_parser")
            .and_then(Value::as_str)
            .map(str::to_owned),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "http://127.0.0.1:18317";
    const MODEL: &str = "m";

    fn info(seed: u64) -> Value {
        json!({
            "served_model_name": MODEL,
            "tp_size": 1,
            "kv_cache_dtype": "fp8_e4m3",
            "max_running_requests": 4,
            "random_seed": seed,
            "startup_time": {"total": 1.5},
        })
    }

    fn models() -> Vec<String> {
        vec![MODEL.to_owned()]
    }

    #[test]
    fn instance_id_follows_the_launch_and_nothing_else_of_a_probe() {
        let a = identify(BASE, MODEL, &info(1), &models()).unwrap();
        assert!(a.instance_id.starts_with("sglang-"));
        assert_eq!(a.instance_id.len(), "sglang-".len() + 16);
        assert_eq!(identify(BASE, MODEL, &info(1), &models()).unwrap(), a);
        let mut noisy = info(1);
        noisy["version"] = json!("other");
        assert_eq!(identify(BASE, MODEL, &noisy, &models()).unwrap(), a);
        let b = identify(BASE, MODEL, &info(2), &models()).unwrap();
        assert_ne!(b.instance_id, a.instance_id);
        let mut pins = info(1);
        pins["max_running_requests"] = json!(8);
        assert_ne!(
            identify(BASE, MODEL, &pins, &models()).unwrap().instance_id,
            a.instance_id
        );
    }

    #[test]
    fn an_unconfirmed_probe_names_its_reason() {
        for key in [
            "tp_size",
            "kv_cache_dtype",
            "max_running_requests",
            "random_seed",
        ] {
            let mut body = info(1);
            body.as_object_mut().unwrap().remove(key);
            assert!(identify(BASE, MODEL, &body, &models()).is_err(), "{key}");
        }
        let mut other = info(1);
        other["served_model_name"] = json!("other");
        assert!(matches!(
            identify(BASE, MODEL, &other, &models()),
            Err(ProbeError::NameMismatch(_))
        ));
        assert!(matches!(
            identify(BASE, MODEL, &info(1), &[]),
            Err(ProbeError::Unconfirmed(_))
        ));
        let mut zero = info(1);
        zero["tp_size"] = json!(0);
        assert!(identify(BASE, MODEL, &zero, &models()).is_err());
    }
}
