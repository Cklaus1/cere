//! Live config reload of a running `cered` (`POST /v1/cere/config/reload`, docs/RUNBOOK.md "Live config reload").
//!
//! [`KEYS`] is the single classification of every config key: `live` keys take effect on a running server,
//! `restart` keys (memory sizing, identity, pins, bound sockets, credentials) only through a restart. A key without
//! a row is `restart` (fail-closed).
//!
//! A reload re-reads the file the server started from ([`crate::config::ConfigSource`]), validates it fully, diffs
//! every leaf against the running config ([`diff`]) and refuses the whole reload, nothing applied, when any change
//! is not live ([`refused`]). Otherwise every [`HOOKS`] entry prepares its change (any refusal again refuses the
//! whole reload), then the prepared changes, the admission caps and the [`Settings`] snapshot are swapped in
//! under one reload lock and the generation moves by one. A request takes its [`Settings`] once, at acceptance:
//! requests in flight finish on the settings they started with, new ones use the new snapshot. Admission caps
//! apply at grant time, so queued requests are granted under the caps in force when they are granted.
//!
//! `config_hash` (the `cere` identity of every response, journaled) stays the startup config's; the effective
//! config's hash and the reload generation are reported by `GET /v1/cere/stats` and logged on every reload.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::app::App;
use crate::config::RuntimeConfig;
use crate::validate::ChatPolicy;

/// Whether a running server may take a change of a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyClass {
    /// Applied by a reload; new requests see it.
    Live,
    /// Refused by a reload; takes effect only through a restart.
    Restart,
}

impl KeyClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Restart => "restart",
        }
    }
}

/// One row of [`KEYS`]: a key path (`[]` for any array index), its class and the code fact that decides it.
#[derive(Debug, Clone, Copy)]
pub struct KeyRule {
    pub path: &'static str,
    pub class: KeyClass,
    pub why: &'static str,
}

const fn live(path: &'static str, why: &'static str) -> KeyRule {
    KeyRule {
        path,
        class: KeyClass::Live,
        why,
    }
}

const fn restart(path: &'static str, why: &'static str) -> KeyRule {
    KeyRule {
        path,
        class: KeyClass::Restart,
        why,
    }
}

const IDENTITY: &str =
    "served identity: the deployment, profile and pins the start validated and loaded";
const BACKEND: &str = "backend shape the start probed and smoked (readiness, routes, cache salts)";
const EXPECT: &str = "the backend's expected identity, checked by readiness at start";
const NATIVE_LOAD: &str =
    "read once when the executor loads (memory, pins, sidecar, loaded backend)";
const FIXED: &str = "fixed by validation to one value; no running reader";
const RESIDENCY: &str = "model residency policy; no running reader, taken at start";
const LEDGER: &str = "sizes or opens the ledger and its journal at start";
const FEDERATION: &str = "registry service, PKI and remote workers built at start";
const CREDENTIALS: &str = "tokens resolved from the environment at start";
const LEVER_LOAD: &str = "lever selection passed to the executor at load; hot levers swap via POST /v1/cere/levers (dev)";
const AB_PLAN: &str = "A/B plan of the settings a request is accepted under; per-request draft flags only, \
                       sessions keep their arm while it is configured, so changes reach new sessions";
const LOOP_GUARD: &str =
    "the loaded backend's loop guard is built at load (tasks/loop-guard-spec.md LG-03)";
const CLIENT_RULES: &str =
    "request labels and their metric series are built at start (roadmap v0.7 A2)";

/// Every config key, classified against the code that reads it (docs/RUNBOOK.md "Live config reload" renders this
/// table). `deployments[].lock` is the pinned deployment files (lock, recipe, capabilities) behind
/// `deployments[].deployment`, compared by content so an edited lock under the same path is refused too.
pub const KEYS: &[KeyRule] = &[
    restart(
        "schema_version",
        "config revision; the whole file's meaning",
    ),
    restart("planning_only", "process mode chosen at start"),
    restart("server.bind", "the listening socket is bound at start"),
    restart("server.tier", "trust-tier policy taken at start"),
    restart("deployments[].mode", IDENTITY),
    restart("deployments[].kind", IDENTITY),
    restart("deployments[].profile_id", IDENTITY),
    restart("deployments[].base_url", BACKEND),
    restart("deployments[].served_model_name", IDENTITY),
    restart("deployments[].deployment", IDENTITY),
    restart("deployments[].lock", IDENTITY),
    restart("deployments[].shared_backend_tenancy", BACKEND),
    restart("deployments[].allow_cross_scope_prefix_cache", BACKEND),
    restart(
        "deployments[].prefix_cache_isolation",
        "per-scope cache salts derived at start",
    ),
    restart(
        "deployments[].api_compat_anthropic",
        "compat routes are mounted at start",
    ),
    restart(
        "deployments[].api_compat_responses",
        "compat routes are mounted at start",
    ),
    restart("deployments[].expect.version", EXPECT),
    restart("deployments[].expect.model_path", EXPECT),
    restart("deployments[].expect.tokenizer_path", EXPECT),
    restart("deployments[].expect.chat_template", EXPECT),
    restart("deployments[].expect.quantization", EXPECT),
    restart("deployments[].expect.kv_cache_dtype", EXPECT),
    restart("deployments[].expect.speculative_algorithm", EXPECT),
    restart("deployments[].expect.speculative_num_steps", EXPECT),
    restart("deployments[].expect.speculative_eagle_topk", EXPECT),
    restart("deployments[].expect.speculative_num_draft_tokens", EXPECT),
    restart("deployments[].expect.reasoning_parser", EXPECT),
    restart("deployments[].expect.tool_call_parser", EXPECT),
    restart("deployments[].expect.allow_auto_truncate", EXPECT),
    restart("deployments[].expect.enable_cache_report", EXPECT),
    restart("deployments[].expect.context_length", EXPECT),
    restart(
        "deployments[].expect.max_running_requests",
        "sizes the worker's in-flight cap at start",
    ),
    restart("deployments[].expect.image_digest", EXPECT),
    restart("deployments[].load_endpoint", BACKEND),
    restart(
        "deployments[].replicas[].device",
        "workers and their epochs are registered at start",
    ),
    restart("deployments[].replicas[].base_url", BACKEND),
    restart("deployments[].replicas[].load_endpoint", BACKEND),
    restart("deployments[].native.library", NATIVE_LOAD),
    restart(
        "deployments[].native.library_sha256",
        "lock_native pin of the loaded library",
    ),
    restart("deployments[].native.artifact_dir", NATIVE_LOAD),
    restart(
        "deployments[].native.expected_manifest_sha256",
        "artifact pin checked at load",
    ),
    restart("deployments[].native.device_uuid", NATIVE_LOAD),
    restart(
        "deployments[].native.replicas[].device",
        "workers and their epochs are registered at start",
    ),
    restart("deployments[].native.replicas[].device_uuid", NATIVE_LOAD),
    restart(
        "deployments[].native.max_context_tokens",
        "sizes the executor's sessions at load",
    ),
    restart(
        "deployments[].native.kv_pool_tokens",
        "sizes the KV pool at load",
    ),
    restart(
        "deployments[].native.sidecar",
        "the tokenizer sidecar process is spawned at load",
    ),
    restart(
        "deployments[].native.sidecar_deadline_ms",
        "copied into the tokenizer sidecar client when the process is spawned at load",
    ),
    restart("deployments[].native.image_digest", NATIVE_LOAD),
    restart(
        "deployments[].native.snapshot_host_bytes",
        "sizes the pinned host snapshot arena at load",
    ),
    restart(
        "deployments[].native.prefix_state_slots",
        "sizes the prefix cache's device state pool at load",
    ),
    restart(
        "deployments[].native.prefix_end_state",
        "set on the model once after cere_model_create, before any ticket",
    ),
    restart(
        "deployments[].native.prefix_alt_edges",
        "set on the model once after cere_model_create, before any ticket",
    ),
    restart(
        "deployments[].native.preempt",
        "preemption state is built at load",
    ),
    restart(
        "deployments[].native.preempt_min_tokens",
        "copied into the loaded backend's preemption policy",
    ),
    restart(
        "deployments[].native.max_parked_sessions",
        "sizes the parked-session arena at load",
    ),
    restart(
        "deployments[].native.preempt_every",
        "copied into the loaded backend's preemption policy",
    ),
    restart(
        "deployments[].native.reserve_output_tokens",
        "copied into the loaded backend's preemption policy",
    ),
    restart(
        "deployments[].native.host_share_dir",
        "executor device options at load",
    ),
    restart(
        "deployments[].native.keep_host_share",
        "executor device options at load",
    ),
    restart(
        "deployments[].native.speculative_draft_tokens",
        "sizes the executor's draft rows at load",
    ),
    // m23 design note PI-02/PI-04: lever selections are copied into the executor at load. A config reload never
    // swaps them (an executor swap can be refused BUSY after a reload hook has prepared, and Apply is infallible);
    // the hot levers swap through the dev route POST /v1/cere/levers instead.
    restart("deployments[].native.levers", LEVER_LOAD),
    restart("deployments[].native.levers.k2_gemm", LEVER_LOAD),
    restart("deployments[].native.levers.prefill_attention", LEVER_LOAD),
    restart("deployments[].native.levers.decode_step", LEVER_LOAD),
    restart("deployments[].native.levers.moe_gemm", LEVER_LOAD),
    restart("deployments[].native.levers.draft_tokens", LEVER_LOAD),
    restart("deployments[].native.levers.verify_attention", LEVER_LOAD),
    restart("deployments[].native.levers.k2_weights", LEVER_LOAD),
    restart("deployments[].native.levers.drafter", LEVER_LOAD),
    restart("deployments[].native.levers.draft_policy", LEVER_LOAD),
    restart("deployments[].native.levers.grammar_drafts", LEVER_LOAD),
    restart("deployments[].native.levers.sdar_gemm", LEVER_LOAD),
    restart(
        "deployments[].native.lever_hot_swap",
        "admits the dev lever route; fixed at start",
    ),
    restart(
        "deployments[].native.family",
        "the family the library is built for; checked at load",
    ),
    // ab-split design note AB-05: the A/B arms are read per request from the live settings (the arm's runtime-layer
    // levers only go into that request's `cere_work.flags`, never into the executor), so a reload takes them; the
    // `ab` hook drops the stats and remembered sessions of removed arms (all sessions on a new salt).
    live("deployments[].native.ab.salt", AB_PLAN),
    live("deployments[].native.ab.arms[].id", AB_PLAN),
    live("deployments[].native.ab.arms[].weight", AB_PLAN),
    live("deployments[].native.ab.arms[].levers", AB_PLAN),
    live("deployments[].native.ab.arms[].levers.drafter", AB_PLAN),
    live(
        "deployments[].native.ab.arms[].levers.draft_policy",
        AB_PLAN,
    ),
    live(
        "deployments[].native.ab.arms[].levers.grammar_drafts",
        AB_PLAN,
    ),
    restart(
        "deployments[].native.executor_memory_limit_bytes",
        "executor memory budget at load",
    ),
    // Track LoRA: the pool is reserved once at load; adapters move through the load/unload routes, not a reload.
    restart(
        "deployments[].native.lora.max_adapters",
        "the adapter pool is reserved at load",
    ),
    restart(
        "deployments[].native.lora.max_rank",
        "the adapter pool is reserved at load",
    ),
    restart(
        "deployments[].native.lora.adapter_root",
        "fixed at start; adapters load through POST /v1/load_lora_adapter",
    ),
    restart(
        "deployments[].native.lora.preload[].name",
        "loaded once at load; adapters load through POST /v1/load_lora_adapter",
    ),
    restart(
        "deployments[].native.lora.preload[].path",
        "loaded once at load; adapters load through POST /v1/load_lora_adapter",
    ),
    // WeightLab: the research reserve is made once at load; plans move through the research routes, not a reload.
    restart(
        "deployments[].native.research.device_staging_bytes",
        "the capture staging is reserved at load",
    ),
    restart(
        "deployments[].native.research.host_buffer_bytes",
        "the capture ring is allocated at load",
    ),
    restart(
        "deployments[].native.research.max_capture_bytes_per_request",
        "fixed at start; plans are checked against it at registration",
    ),
    restart(
        "deployments[].native.research.max_plans",
        "the plan slots are reserved at load",
    ),
    restart(
        "deployments[].native.research.max_vectors",
        "fixed at start; vectors upload through POST /v1/cere/research/vectors",
    ),
    restart(
        "deployments[].native.decode_rounds_per_prefill_chunk",
        "the executor's round policy is set once after cere_executor_create; O79 PF1: decode rounds per prefill \
         chunk while decode rows are ready (1..64, absent = 1), bit-identical (row invariance), trades a prefill's \
         TTFT for the running rows' tok/s; see tasks/m210-spec.md PF1",
    ),
    restart(
        "deployments[].native.prefill_slice_ms",
        "the executor's prefill slice is set once after cere_executor_create, before the model is created; MixedPD: \
         a prefill call pauses at a layer boundary after about this many ms while decode rows are ready (1..10000, \
         absent = off), bit-identical; see tasks/mixed-pd-spec.md",
    ),
    restart(
        "deployments[].native.decode_share_pct",
        "the executor's decode share is set once after cere_executor_create; S1 G4: while two or more prompts wait \
         they prefill back to back, with one left decode rows get this percent of its prefill wall time (1..1000, \
         absent = off), bit-identical; see tasks/beat-sglang-spec.md",
    ),
    restart(
        "deployments[].native.prefill_age_calls",
        "the executor's prefill aging cap is set once after cere_executor_create; BC4 (b): without a decode share a \
         prompt overtaken this many times by younger prompts' prefill calls runs next (1..1024, absent = the \
         default 8), bit-identical; see tasks/batch-concurrency-spec.md",
    ),
    restart("deployments[].native.loop_guard.min_period_bytes", LOOP_GUARD),
    restart("deployments[].native.loop_guard.max_period_bytes", LOOP_GUARD),
    restart("deployments[].native.loop_guard.min_repeats", LOOP_GUARD),
    restart("deployments[].native.loop_guard.min_span_bytes", LOOP_GUARD),
    restart("deployments[].native.loop_guard.tool_repeat_run", LOOP_GUARD),
    restart("deployments[].native.loop_guard.ignore_arg_keys", LOOP_GUARD),
    restart("scheduler.mode", FIXED),
    restart(
        "scheduler.devices",
        "workers and their epochs are registered at start",
    ),
    restart(
        "scheduler.worker_lost_after_ms",
        "the worker registry's aging rule is built at start",
    ),
    restart(
        "state.host_spill_enabled",
        "preemption's host spill is decided at load",
    ),
    restart("state.nvme_spill_enabled", FIXED),
    restart("state.allow_cross_device_resume", FIXED),
    live(
        "state.max_blob_bytes",
        "read by each state export/import from the settings it starts under",
    ),
    live(
        "state.busy_wait_ms",
        "read by each state export/import from the settings it starts under",
    ),
    restart(
        "state.park_on_stop",
        "the park store and its prompt recorder are built at start",
    ),
    restart(
        "state.park_max_sessions",
        "the prompt recorder's bound is fixed at start",
    ),
    live(
        "state.park_max_bytes",
        "read by the graceful stop from the settings it stops under",
    ),
    live(
        "state.park_deadline_ms",
        "read by the graceful stop from the settings it stops under",
    ),
    restart("telemetry.level", FIXED),
    restart("telemetry.raw_content", FIXED),
    restart("telemetry.clients[].client", CLIENT_RULES),
    restart("telemetry.clients[].workload", CLIENT_RULES),
    restart("telemetry.clients[].scope", CLIENT_RULES),
    restart("telemetry.clients[].priority_header", CLIENT_RULES),
    restart("safety.fail_on_unknown_capability", FIXED),
    restart("safety.allow_tool_execution", FIXED),
    restart("safety.allow_power_changes", FIXED),
    restart("safety.allow_cloud_fallback", FIXED),
    restart("model_management.load_on_demand", RESIDENCY),
    restart("model_management.single_flight_loads", FIXED),
    restart("model_management.unload_idle", RESIDENCY),
    restart("model_management.idle_unload_seconds", RESIDENCY),
    restart("model_management.min_resident_seconds", RESIDENCY),
    restart("model_management.max_loaded_models_per_device", RESIDENCY),
    restart("model_management.max_concurrent_loads", RESIDENCY),
    restart("model_management.eviction_policy", RESIDENCY),
    restart("model_management.drain_timeout_ms", RESIDENCY),
    restart("model_management.preload_recipe_ids", RESIDENCY),
    restart(
        "api.max_request_body_bytes",
        "sizes the ledger's retained request bytes and the sidecar's input bound at start",
    ),
    live(
        "api.max_messages",
        "chat validation policy, taken per request",
    ),
    live(
        "api.default_max_output_tokens",
        "chat validation policy, taken per request",
    ),
    live(
        "api.max_output_tokens",
        "chat validation policy, taken per request",
    ),
    live(
        "api.default_deadline_ms",
        "deadline budget, taken per request",
    ),
    live("api.max_deadline_ms", "deadline budget, taken per request"),
    live(
        "api.stream_buffer_events",
        "a stream's channel bound, taken per request",
    ),
    live(
        "api.stall_timeout_ms",
        "a stream's stall bound, taken per request",
    ),
    live("api.shutdown_drain_ms", "read when the shutdown starts"),
    restart("api.resume_buffer_bytes", "sizes the resume store at start"),
    restart(
        "api.resume_buffer_total_bytes",
        "sizes the resume store at start",
    ),
    restart(
        "api.resume_retention_ms",
        "the resume store's retention is set at start",
    ),
    live(
        "admission.max_backend_inflight",
        "admission cap, swapped between grants",
    ),
    live(
        "admission.per_scope_inflight",
        "admission cap, swapped between grants",
    ),
    live(
        "admission.queue_depth",
        "admission cap, swapped between grants",
    ),
    live(
        "admission.aging_ms",
        "admission scheduling policy, swapped between grants",
    ),
    live(
        "admission.batch_floor",
        "admission scheduling policy, swapped between grants",
    ),
    live(
        "admission.owner_wait_cap_ms",
        "admission scheduling policy, swapped between grants",
    ),
    restart("auth.scopes[].scope_id", CREDENTIALS),
    restart("auth.scopes[].token_env", CREDENTIALS),
    restart("auth.scopes[].token_envs", CREDENTIALS),
    live(
        "auth.scopes[].weight",
        "admission DRR weight, swapped between grants",
    ),
    restart(
        "auth.scopes[].allowed_tiers",
        "trust-tier policy taken at start",
    ),
    live(
        "auth.scopes[].class",
        "admission scheduling policy, swapped between grants",
    ),
    live(
        "auth.scopes[].per_scope_inflight",
        "admission cap, swapped between grants",
    ),
    live(
        "auth.scopes[].max_batch_inflight",
        "admission cap, swapped between grants",
    ),
    restart("auth.scopes[].max_sessions_per_scope", LEDGER),
    restart("auth.operator_token_env", CREDENTIALS),
    restart("auth.keyless_scope", CREDENTIALS),
    restart("ledger.state_dir", LEDGER),
    restart("ledger.max_records", LEDGER),
    restart("ledger.terminal_retention_ms", LEDGER),
    restart("ledger.max_sessions_per_scope", LEDGER),
    restart("ledger.session_ttl_ms", LEDGER),
    restart("ledger.max_revisions_per_session", LEDGER),
    restart(
        "ledger.orphan_load_poll_ms",
        "orphan load pollers are spawned at start",
    ),
    restart("ledger.evict_terminal_when_full", LEDGER),
    restart("usage.enabled", "the usage journal writer is started at start"),
    restart(
        "usage.retention_days",
        "the usage journal writer is started at start",
    ),
    restart("federation.cluster_id", FEDERATION),
    restart("federation.coordinator_id", FEDERATION),
    restart("federation.listen", FEDERATION),
    restart("federation.principal", FEDERATION),
    restart("federation.ca_file", FEDERATION),
    restart("federation.cert_file", FEDERATION),
    restart("federation.key_file", FEDERATION),
    restart("federation.heartbeat_interval_ms", FEDERATION),
    restart("federation.suspect_after_ms", FEDERATION),
    restart("federation.lost_after_ms", FEDERATION),
    restart("federation.admission_lease_ms", FEDERATION),
    restart("federation.rpc_timeout_ms", FEDERATION),
    restart("federation.rpc_quota_per_principal", FEDERATION),
    restart("federation.workers[].worker_id", FEDERATION),
    restart("federation.workers[].endpoint", FEDERATION),
    restart("federation.workers[].principal", FEDERATION),
    restart("federation.workers[].max_inflight", FEDERATION),
    restart("federation.workers[].expect.served_model_name", FEDERATION),
    restart("federation.workers[].expect.tp_size", FEDERATION),
    restart("federation.workers[].expect.kv_cache_dtype", FEDERATION),
    restart("federation.workers[].owner", FEDERATION),
    restart("federation.workers[].tier", FEDERATION),
    restart("federation.workers[].owner_only", FEDERATION),
];

/// The [`KEYS`] row of a concrete leaf path (`deployments[0].native.library` matches `deployments[].native.library`).
pub fn rule(path: &str) -> Option<&'static KeyRule> {
    let pattern = normalize(path);
    KEYS.iter().find(|r| r.path == pattern)
}

/// `path` with every array index removed (`a[3].b` -> `a[].b`).
fn normalize(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    let mut in_index = false;
    for c in path.chars() {
        match c {
            '[' => {
                in_index = true;
                out.push('[');
            }
            ']' => {
                in_index = false;
                out.push(']');
            }
            _ if in_index => {}
            _ => out.push(c),
        }
    }
    out
}

/// Every leaf of `config`, by concrete path: objects descend by key, arrays of objects by index, anything else
/// (scalars, scalar arrays) is one leaf. An absent optional key has no leaf. Each pinned deployment adds
/// `deployments[i].lock`, the content of its deployment files.
fn leaves(config: &RuntimeConfig) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    let value = serde_json::to_value(config).expect("runtime config serializes to JSON");
    flatten(String::new(), value, &mut out);
    for (i, entry) in config.deployments.iter().enumerate() {
        if let Some(d) = &entry.lock {
            let files = json!({
                "lock_sha256": d.lock_sha256,
                "recipe": serde_json::to_value(&d.recipe).expect("recipe serializes"),
                "capabilities": serde_json::to_value(&d.profile).expect("capabilities serialize"),
            });
            out.insert(format!("deployments[{i}].lock"), files);
        }
    }
    out
}

fn flatten(path: String, value: Value, out: &mut BTreeMap<String, Value>) {
    match value {
        Value::Object(map) => {
            for (key, v) in map {
                let child = if path.is_empty() {
                    key
                } else {
                    format!("{path}.{key}")
                };
                flatten(child, v, out);
            }
        }
        Value::Array(items) if !items.is_empty() && items.iter().all(Value::is_object) => {
            for (i, v) in items.into_iter().enumerate() {
                flatten(format!("{path}[{i}]"), v, out);
            }
        }
        other => {
            out.insert(path, other);
        }
    }
}

/// The concrete leaf paths whose value differs between `running` and `candidate` (present on one side only
/// counts), sorted.
pub fn diff(running: &RuntimeConfig, candidate: &RuntimeConfig) -> Vec<String> {
    let (a, b) = (leaves(running), leaves(candidate));
    let mut changed: Vec<String> = a
        .iter()
        .filter(|(path, value)| b.get(*path) != Some(*value))
        .map(|(path, _)| path.clone())
        .chain(b.keys().filter(|path| !a.contains_key(*path)).cloned())
        .collect();
    changed.sort();
    changed
}

/// The changes of `changed` (from [`diff`]) a running server cannot take: `restart` keys and keys without a row.
pub fn refused(changed: &[String]) -> Vec<String> {
    changed
        .iter()
        .filter(|path| rule(path).map(|r| r.class) != Some(KeyClass::Live))
        .cloned()
        .collect()
}

/// The applied settings: the effective config (startup config at generation 0), its hash, its reload generation
/// and what is derived from it per request. A request holds one `Arc<Settings>` from acceptance to its terminal.
#[derive(Debug)]
pub struct Settings {
    pub generation: u64,
    /// `config_hash()` of [`Self::config`].
    pub config_hash: String,
    pub config: RuntimeConfig,
    /// Chat validation of each `[[deployments]]` entry, in config order.
    pub(crate) policies: Vec<ChatPolicy>,
    /// ab-split design note AB-01: each entry's A/B plan (`None` without `[deployments.native.ab]`), in config
    /// order; a request takes its arm from the snapshot it was accepted under.
    pub(crate) ab_plans: Vec<Option<Arc<crate::ab::Plan>>>,
}

impl Settings {
    pub(crate) fn new(config: RuntimeConfig, generation: u64) -> Result<Self, String> {
        let policies = (0..config.deployments.len())
            .map(|index| ChatPolicy::from_config(&config, index))
            .collect::<Result<Vec<_>, String>>()?;
        let ab_plans = config
            .deployments
            .iter()
            .map(|d| match &d.native {
                Some(native) => crate::ab::Plan::from_native(native).map(|p| p.map(Arc::new)),
                None => Ok(None),
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            generation,
            config_hash: config.config_hash(),
            config,
            policies,
            ab_plans,
        })
    }
}

/// A prepared change of a reload hook, applied after every hook prepared and before the settings swap; it cannot
/// fail.
pub type Apply = Box<dyn FnOnce(&App) + Send>;

/// A subsystem with live state of its own (e.g. a hot executor lever) that a reload must reconfigure. `prepare`
/// sees the running and the candidate config (already validated, every changed key `live`), has no side effect,
/// and answers `Ok(None)` when it has nothing to change, `Ok(Some(apply))` with the change, or `Err(reason)` to
/// refuse the whole reload (409 `reload_hook_refused` naming the hook; nothing is applied).
#[derive(Clone, Copy)]
pub struct Hook {
    pub name: &'static str,
    pub prepare: fn(&App, &RuntimeConfig, &RuntimeConfig) -> Result<Option<Apply>, String>,
}

/// Every reload hook, prepared and applied in this order. A key a hook reconfigures is a `live` row of [`KEYS`].
pub const HOOKS: &[Hook] = &[Hook {
    name: "ab",
    prepare: crate::ab::reload_hook,
}];

/// A reload that took effect, or found nothing to change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reloaded {
    /// False when the file has no change against the running config (generation and hash unchanged).
    pub applied: bool,
    pub generation: u64,
    pub config_hash: String,
    /// The changed leaf paths.
    pub changed: Vec<String>,
}

/// Why a reload changed nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReloadError {
    /// The server did not start from a file (an in-process config).
    Unavailable,
    /// The server is shutting down.
    Stopping,
    /// The file is unreadable or invalid.
    Invalid(String),
    /// These changed keys need a restart.
    Restart(Vec<String>),
    /// A [`HOOKS`] entry refused.
    Hook { name: &'static str, reason: String },
}

impl std::fmt::Display for ReloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable => f.write_str("the server did not start from a config file"),
            Self::Stopping => f.write_str("the server is shutting down"),
            Self::Invalid(e) => write!(f, "invalid config: {e}"),
            Self::Restart(keys) => write!(
                f,
                "restart-class keys changed, nothing applied: {}",
                keys.join(", ")
            ),
            Self::Hook { name, reason } => write!(f, "reload hook {name} refused: {reason}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_matches_any_index() {
        assert_eq!(
            normalize("deployments[12].replicas[0].device"),
            "deployments[].replicas[].device"
        );
        assert_eq!(rule("auth.scopes[3].weight").unwrap().class, KeyClass::Live);
        assert_eq!(
            rule("deployments[0].native.kv_pool_tokens").unwrap().class,
            KeyClass::Restart
        );
        assert!(rule("no.such.key").is_none());
    }

    #[test]
    fn keys_are_unique() {
        let mut paths: Vec<&str> = KEYS.iter().map(|r| r.path).collect();
        paths.sort_unstable();
        let len = paths.len();
        paths.dedup();
        assert_eq!(paths.len(), len, "a KEYS path is listed twice");
    }
}
