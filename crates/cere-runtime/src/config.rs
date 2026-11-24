//! Runtime configuration revision 2 (`schemas/runtime_config.schema.json`).
//!
//! Loading is strict: unknown keys are rejected by serde, then every schema constraint (patterns, ranges,
//! consts, conditionals) and every semantic rule of `scripts/cere_contracts.py` is checked here. The shared
//! case table `tests/fixtures/runtime_config_invalid.json` keeps the Python and Rust verdicts identical.
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::OsString;
use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::time::Duration;

use cere_core::capabilities::{ExecutionCapabilities, SAMPLING_PARAMS};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

use crate::levers::{Family, LeverSet, Target};
use crate::tier::Tier;

pub const SCHEMA_VERSION: &str = "2";

/// `scheduler.worker_lost_after_ms` when absent (M3 Q10).
pub const DEFAULT_WORKER_LOST_AFTER_MS: u64 = 60_000;

/// Any reason a runtime configuration (or its deployment files, or its auth secrets) is unusable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(String);

impl ConfigError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid runtime config: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    pub schema_version: String,
    pub planning_only: bool,
    pub server: ServerConfig,
    /// `[[deployments]]` (M5 D1): 1..=16 entries; the request's `model` selects one by `served_model_name`.
    pub deployments: Vec<DeploymentConfig>,
    pub model_management: ModelManagementConfig,
    pub scheduler: SchedulerConfig,
    pub state: StateConfig,
    pub telemetry: TelemetryConfig,
    pub safety: SafetyConfig,
    pub api: ApiConfig,
    pub admission: AdmissionConfig,
    pub auth: AuthConfig,
    pub ledger: LedgerConfig,
    /// M3.5 federated workers (m35 design note Q16). Optional so configs without it keep their `config_hash`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub federation: Option<FederationConfig>,
    /// net-teleport design note 4.4 (S7): the usage journal. Optional so configs without it keep their
    /// `config_hash`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageConfig>,
    /// The file this config was read from, re-read by a live reload (`crate::reload`). Never serialized, so it is
    /// not part of `config_hash`.
    #[serde(skip)]
    pub source: Option<ConfigSource>,
}

/// A config file and the repository root its relative paths resolve against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigSource {
    pub path: PathBuf,
    pub repo_root: PathBuf,
}

impl ConfigSource {
    /// Reads, validates and resolves the file again ([`RuntimeConfig::from_toml_str`]); the result keeps this
    /// source.
    pub fn read(&self) -> Result<RuntimeConfig, ConfigError> {
        let text = std::fs::read_to_string(&self.path)
            .map_err(|e| ConfigError::new(format!("cannot read {}: {e}", self.path.display())))?;
        let mut config = RuntimeConfig::from_toml_str(&text, &self.repo_root)?;
        config.source = Some(self.clone());
        Ok(config)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// `127.0.0.1:<port>`.
    pub bind: String,
    /// Trust tier of this install for its owner (net-teleport design note 4.1); absent → `own`. Optional so
    /// configs without it keep their `config_hash`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<Tier>,
}

impl ServerConfig {
    pub fn tier(&self) -> Tier {
        self.tier.unwrap_or(crate::tier::DEFAULT_TIER)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendMode {
    BackendManaged,
    Native,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Unconfigured,
    CpuReference,
    SglangHttp,
    /// The in-process ABI v3 executor (`libcere_executor.so`) with the tokenizer sidecar (spec m22 R4).
    NativeCuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharedBackendTenancy {
    Exclusive,
    SharedAcknowledged,
}

/// How scopes sharing one backend radix cache are kept apart (Q2). Absent in the config means `none`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixCacheIsolation {
    /// One shared cache namespace for every scope.
    None,
    /// Every backend chat request carries a per-scope `cache_salt` derived from the server secret.
    CacheSalt,
}

/// One `[[deployments]]` entry (M5 D1): a served model with its backend kind, lock and workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentConfig {
    pub mode: BackendMode,
    pub kind: BackendKind,
    pub profile_id: String,
    /// `sglang_http` single form only: `http://<127.0.0.0/8 literal or [::1]>:<port>`, nothing else. A single
    /// `sglang_http` entry sets exactly one of `base_url` and `replicas` (M3 Q2); several entries refuse it (M5 D2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// `sglang_http` only: independent replicas of this deployment, one per scheduler device (M3 Q2).
    /// Optional so configs without it keep their `config_hash`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replicas: Option<Vec<ReplicaConfig>>,
    /// Live kinds (`sglang_http`, `native_cuda`): the `model` a chat request must name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_model_name: Option<String>,
    /// Live kinds: repo-relative prefix of `<prefix>.{recipe,lock,capabilities}.json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deployment: Option<String>,
    pub shared_backend_tenancy: SharedBackendTenancy,
    pub allow_cross_scope_prefix_cache: bool,
    /// Optional so configs that omit it keep their `config_hash` (Q3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_cache_isolation: Option<PrefixCacheIsolation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expect: Option<BackendExpect>,
    /// `sglang_http` single form only: the backend's load path (`/get_load`) polled to release recovered orphans
    /// early (m1close Q6); absent → orphans are held for the retention bound only. With `replicas` it moves to
    /// each replica.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_endpoint: Option<String>,
    /// `native_cuda` only: the executor library, artifact, device and sidecar. Optional so configs without it
    /// keep their `config_hash` (m1close Q3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native: Option<NativeConfig>,
    /// M2.14 D3: serve `POST /v1/messages` and `/v1/messages/count_tokens` for this deployment. Default off (N6);
    /// omitted when false so configs without it keep their `config_hash`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub api_compat_anthropic: bool,
    /// M2.14 D4: serve `/v1/responses` for this deployment. Default off; omitted when false (as above).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub api_compat_responses: bool,
    /// This entry's pinned recipe, lock and capability profile; loaded for live kinds. Not part of the TOML.
    #[serde(skip)]
    pub lock: Option<Deployment>,
}

impl DeploymentConfig {
    /// The effective isolation: absent means `none`.
    pub fn prefix_cache_isolation(&self) -> PrefixCacheIsolation {
        self.prefix_cache_isolation
            .unwrap_or(PrefixCacheIsolation::None)
    }

    /// The backend workers in configuration order (M3 Q3): the `replicas` array, or the single form as one
    /// replica on `devices[0]` with the backend's `base_url` and `load_endpoint`. Empty for kinds without a
    /// `base_url` (planning kinds, `native_cuda`).
    pub fn replica_list(&self, devices: &[String]) -> Vec<ReplicaConfig> {
        match (&self.replicas, &self.base_url, devices.first()) {
            (Some(replicas), _, _) => replicas.clone(),
            (None, Some(base_url), Some(device)) => vec![ReplicaConfig {
                device: device.clone(),
                base_url: base_url.clone(),
                load_endpoint: self.load_endpoint.clone(),
            }],
            _ => Vec::new(),
        }
    }
}

/// One `[[deployments.replicas]]` entry: an independent `sglang_http` server of its deployment (M3 Q2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplicaConfig {
    /// The worker id: one of `scheduler.devices`.
    pub device: String,
    /// Same form as `deployments.base_url`.
    pub base_url: String,
    /// Same form as `deployments.load_endpoint`, for this replica.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_endpoint: Option<String>,
}

/// Pinned `GET /get_server_info` values. Field names are the SGLang keys, except `image_digest`
/// (recorded evidence; the backend does not report it).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendExpect {
    pub version: String,
    pub model_path: String,
    pub tokenizer_path: String,
    /// Absent pins `null` (the model's bundled template); TOML has no null. Likewise for `quantization` and the
    /// speculative keys below (absent = the backend must report null).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    pub kv_cache_dtype: String,
    /// The four speculative keys are all present or all absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_algorithm: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_num_steps: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_eagle_topk: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_num_draft_tokens: Option<u64>,
    pub reasoning_parser: String,
    pub tool_call_parser: String,
    pub allow_auto_truncate: bool,
    pub enable_cache_report: bool,
    pub context_length: u64,
    pub max_running_requests: u64,
    pub image_digest: String,
}

/// `[deployments.native]` (spec m22 "Native path contract"): what `cered` loads in-process and the identity readiness
/// holds it to. Paths are the ones the `cered` process sees (inside the native container).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeConfig {
    /// Absolute path of `libcere_executor.so`, opened with `dlopen` after its SHA-256 matched.
    pub library: String,
    /// SHA-256 of the library file; equals the deployment lock's `backend_build_sha256`.
    pub library_sha256: String,
    /// Absolute path of the prepared M2.1 artifact, opened in stamp mode.
    pub artifact_dir: String,
    /// Manifest SHA-256 the artifact must have; equals the deployment lock's `artifact_sha256`.
    pub expected_manifest_sha256: String,
    /// `GPU-<uuid>` the process is bound to (`CUDA_VISIBLE_DEVICES`); the executor uses its device 0. Set exactly
    /// when `replicas` is absent (m32n design note D2); skipped when absent, so configs keep their `config_hash`.
    /// An AMD GPU (amd-amd design note PAR-6: the HIP executor) is bound as `hip:<ordinal>` instead, the process's
    /// `HIP_VISIBLE_DEVICES` (gfx1151 reports no usable UUID); `CUDA_VISIBLE_DEVICES` must then be unset, since HIP
    /// honors it too. The Vulkan executor (vulkan-amd design note D2, `kind` stays `native_cuda`) is bound as
    /// `vk:<ordinal>`: the ordinal is the executor's device id (Vulkan physical device `<ordinal>`), no
    /// `*_VISIBLE_DEVICES` variable applies. Single form only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_uuid: Option<String>,
    /// m32n design note D1/D2: `[[deployments.native.replicas]]`, one in-process executor per card of this one
    /// deployment. Every other native field is deployment-level, so identical per replica by construction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replicas: Option<Vec<NativeReplicaConfig>>,
    /// Context limit of every session; equals the capability profile's `max_context_tokens`.
    pub max_context_tokens: u64,
    /// KV pool size in tokens (m25 design note D4; ledger D 72): absent or 0 = the derived pool (max_sessions x
    /// ceil(context / 64) x 64); non-zero is a multiple of 64 and at least `max_context_tokens`. The upper bound is the
    /// executor's support table (UNSUPPORTED at model create). Skipped when absent, so configs without it keep their
    /// `config_hash`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_pool_tokens: Option<u64>,
    /// argv of the tokenizer sidecar (`scripts/native/tokenizer_sidecar.py`), spawned without a shell.
    pub sidecar: Vec<String>,
    /// Audit N1: the longest one tokenizer sidecar call (request write plus reply) may take before the child is
    /// killed, in milliseconds; 0 = the default ([`crate::native::sidecar::DEFAULT_IO_DEADLINE`], 120 s). Skipped
    /// when 0, so configs without it keep their `config_hash`.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub sidecar_deadline_ms: u64,
    /// Recorded evidence only: digest of the container image that provides CUDA user space and the tokenizer.
    pub image_digest: String,
    /// M2.11 D8 host-RAM snapshot arena (`cere_model_desc.snapshot_host_bytes`, ledger D/M3); 0 = none (shipped).
    /// Above 0 exactly when the profile declares `native_state_snapshot` (ledger C 56).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub snapshot_host_bytes: u64,
    /// M2.11 D8 U1 preemption: needs the arena and `[state] host_spill_enabled`; lets in-flight work exceed the
    /// profile's rows by `max_parked_sessions`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub preempt: bool,
    /// Committed tokens a victim needs, and the spacing between two preemptions of one generation (D8).
    #[serde(
        default = "default_preempt_min_tokens",
        skip_serializing_if = "is_default_preempt_min_tokens"
    )]
    pub preempt_min_tokens: u64,
    /// Snapshots the arena holds at once, each at the context limit (D8).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub max_parked_sessions: u64,
    /// D13 dev knob: preempt the oldest running generation after every N committed tokens; 0 = off. Only the
    /// dev/live `<profile id>-snapshot` profile admits it (ledger P1, LR5).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub preempt_every: u32,
    /// sessions design note SE-09 optimistic reservation: a generation's first session holds its prompt plus this
    /// many output tokens (64-token multiple); one that uses it up parks itself (U1) and resumes in a session of its
    /// whole budget. 0 = off (the session holds its whole budget). Needs `preempt` and `max_parked_sessions` ≥
    /// `admission.max_backend_inflight`, so every running generation can always park.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub reserve_output_tokens: u64,
    /// Opportunity O47 (build-loop-ple design note): the host directory (a tmpfs every lane on the host mounts, e.g.
    /// `/dev/shm/cere-host`) where the executor shares the artifact's HOST_PINNED region (PLE table, scale, index)
    /// with the other native processes of the same artifact; absent = the private pinned block. An absolute path
    /// without commas (it rides in the executor's device id). Skipped when absent, so configs keep their `config_hash`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_share_dir: Option<String>,
    /// M2.10 D6 (`cere_model_desc.prefix_state_slots`, ledger D 104): device state slots of the native prefix cache;
    /// 0 = cache off (default, M2.5's eager page take), at most [`MAX_PREFIX_STATE_SLOTS`]. Skipped when 0, so
    /// configs without it keep their `config_hash`.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub prefix_state_slots: u64,
    /// m210 design note D15⁗′ stage 2a (`cere_model_set_prefix_end_state`, set once after `cere_model_create`): a
    /// drafting PREFILL also captures its end state (two state slots, class-pinned resume). Opt-in prefix profiles
    /// only; requires `prefix_state_slots > 0`. Skipped when false, so configs without it keep their `config_hash`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub prefix_end_state: bool,
    /// m210 design note D17 (`cere_model_set_prefix_alt_edges`, set once after `cere_model_create`): a session whose
    /// path runs through another session's edge over other KV pages stores its states on an alternative edge over its
    /// own pages instead of being refused (M210S-39). Opt-in prefix profiles only; requires `prefix_state_slots > 0`.
    /// Skipped when false, so configs without it keep their `config_hash`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub prefix_alt_edges: bool,
    /// m29 design note D11 (I1): MTP draft tokens per speculate round; 0 = off (today's behaviour, not serialized,
    /// so configs keep their `config_hash`), 4 = on (the MTP draft uploaded to DEVICE with the artifact's token map).
    /// On family `inkling2_loop` the on value is a window of 2..8 verify rows instead (inkling design note D-I-S9B:
    /// the MTP drafter on DEVICE, no token map, the unfed token plus 1..7 drafts). Any other value is refused at
    /// load. Non-zero exactly when the profile declares `native_speculative_batch`.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub speculative_draft_tokens: u32,
    /// m23 design note PI-02: `[deployments.native.levers]`, lever name -> impl id of `crate::levers::REGISTRY`
    /// (absent levers take the registry default; the drafter of m29l design note D6 is the `drafter` lever).
    /// Skipped when empty, so configs without it keep their `config_hash`; when present it is part of it.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub levers: BTreeMap<String, String>,
    /// PI-04: the dev/bench flag that admits `POST /v1/cere/levers` (the hot swap); shipped profiles never set it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub lever_hot_swap: bool,
    /// SelfTune ST4 S3a: the model family the library is built for (its descriptor's `model_type`), `qwen4_exp`
    /// (absent, not serialized, so configs keep their `config_hash`) or `sdar` (DICE-8B). On `sdar` every lever is its
    /// default and the features an SDAR library refuses are refused here by name ([`check_family`]).
    #[serde(default, skip_serializing_if = "Family::is_qwen4_exp")]
    pub family: Family,
    /// m29 design note M29S2-02 (ledger M1): the executor's device-byte limit (`cere_executor_options
    /// .memory_budget_bytes`); absent = [`crate::readiness::NATIVE_MEMORY_BUDGET_BYTES`] (80 GiB, every profile
    /// before M2.9). A whole MiB, at least that default: a profile raises it to its measured peak + 512 MiB, at most
    /// [`crate::readiness::NATIVE_MEMORY_CEILING_BYTES`] (m23 design note P8: above it the executor could cross the
    /// GPU0 free floor, 10,240 MiB, which the live gates check at the measured peak). Skipped when absent (`config_hash`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executor_memory_limit_bytes: Option<u64>,
    /// Opportunity O79 (m210 design note PF1, `cere_executor_set_decode_rounds`): decode rounds the executor runs
    /// after each prefill chunk while decode rows are ready, 1..=[`MAX_DECODE_ROUNDS_PER_CHUNK`]; absent = the
    /// executor default 1 (the kinds alternate) and the call is not made, so libraries predating it keep loading.
    /// Skipped when absent (`config_hash`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_rounds_per_prefill_chunk: Option<u64>,
    /// MixedPD (mixed-pd design note, `cere_executor_set_prefill_slice`): layer-sliced prefill, a prefill call pauses
    /// at a layer boundary after about this many ms while decode rows are ready, 1..=[`MAX_PREFILL_SLICE_MS`];
    /// absent = every call runs whole and the call is not made, so libraries predating it keep loading. Set before
    /// the model is created. Skipped when absent (`config_hash`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_slice_ms: Option<u64>,
    /// S1 G4 (beat-sglang design note, `cere_executor_set_decode_share`): decode share, while 2+ prompts wait they
    /// prefill back to back; with one left, decode rows get this percent of its prefill wall time,
    /// 1..=[`MAX_DECODE_SHARE_PCT`]; absent = the decode rounds per chunk stay and the call is not made, so libraries
    /// predating it keep loading. Set right after the executor is created. Skipped when absent (`config_hash`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_share_pct: Option<u64>,
    /// BC4 (b) (batch-concurrency design note, `cere_executor_set_prefill_age`): the aging cap of the
    /// fewest-tokens-left prefill order, a prompt overtaken this many times by younger prompts' prefill calls runs
    /// next, 1..=[`MAX_PREFILL_AGE_CALLS`]; absent = the call is not made and the executor default 8 stays. Set right
    /// after the executor is created. Skipped when absent (`config_hash`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_age_calls: Option<u64>,
    /// M2.16 L4 (m216 design note): keep the published shared segment after this process detaches
    /// (`host_share_keep=1`), so a restart attaches instead of reading and pinning the PLE again; requires
    /// `host_share_dir`. Removal: docs/RUNBOOK.md "Shared PLE host segment". Skipped when false (`config_hash` kept).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub keep_host_share: bool,
    /// ab-split design note AB-01: `[deployments.native.ab]`, the live A/B split of the runtime-layer levers. Absent
    /// in every shipped profile (not serialized, so configs keep their `config_hash`); present, the per-request
    /// path takes each request's arm choices over the active levers (`crate::ab`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ab: Option<AbConfig>,
    /// Track LoRA (`-v50`): `[deployments.native.lora]`, PEFT LoRA adapters served on the base model, loaded and
    /// unloaded at runtime (`POST /v1/load_lora_adapter`, `POST /v1/unload_lora_adapter`). Absent in every other
    /// profile (not serialized, so configs keep their `config_hash`); a request that names no adapter is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<LoraConfig>,
    /// WeightLab (`-v57`, weightlab design note WL-01): `[deployments.native.research]`, opt-in capture and
    /// intervention plans (`/v1/cere/research/*`, operator bearer). Absent in every other profile (not serialized, so
    /// configs keep their `config_hash`); absent, no research symbol is called and nothing changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub research: Option<ResearchConfig>,
    /// loop-guard design note: `[deployments.native.loop_guard]`, the native repetition guard (LG-D1 in-response
    /// repetition, LG-D2 repeated tool calls across turns). Absent in every shipped profile (not serialized, so
    /// configs keep their `config_hash`) and off; present, its detectors run with the table's thresholds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loop_guard: Option<LoopGuardConfig>,
}

/// One `[[deployments.native.replicas]]` entry (m32n design note D2): the worker id (one of `scheduler.devices`)
/// and the card it runs on. Replica `i` uses executor device ordinal `i`, so `CUDA_VISIBLE_DEVICES` must list the
/// replicas' UUIDs in replica order (D3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeReplicaConfig {
    pub device: String,
    pub device_uuid: String,
}

/// `[deployments.native.lora]`: the executor's adapter pool (`cere_model_lora_reserve`, sized at load) and where
/// adapters may be loaded from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoraConfig {
    /// Adapters resident at once, 1..=`CERE_LORA_MAX_ADAPTERS` (64).
    pub max_adapters: u32,
    /// The largest rank an adapter may have, 1..=`CERE_LORA_MAX_RANK` (256).
    pub max_rank: u32,
    /// Absolute directory every adapter path must resolve inside (symlinks resolved).
    pub adapter_root: String,
    /// Adapters loaded at startup, before the backend reports ready; a failed one fails the load.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub preload: Vec<LoraPreload>,
}

/// One `[[deployments.native.lora.preload]]` adapter: its served name and its PEFT directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoraPreload {
    pub name: String,
    pub path: String,
}

/// `[deployments.native.research]` (WeightLab WL-01): the executor's research reserve (`cere_model_research_reserve`
/// at model create, before seal) and the per-request capture cap. All five keys are required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResearchConfig {
    /// Device capture staging, 1 MiB..=1 GiB, counted against the memory budget at create.
    pub device_staging_bytes: u64,
    /// Pinned host capture ring shared by every request, 1 MiB..=16 GiB.
    pub host_buffer_bytes: u64,
    /// A plan whose capture upper bound exceeds this is refused at registration, 4096..=`host_buffer_bytes`.
    pub max_capture_bytes_per_request: u64,
    /// Plans registered at once, 1..=`CERE_RESEARCH_MAX_PLANS` (64).
    pub max_plans: u32,
    /// Uploaded vectors held at once, 1..=1024.
    pub max_vectors: u32,
}

/// WeightLab WL-18: the production bind port a research config may never take (`configs/runtime/prod-t4.toml`).
pub const PRODUCTION_PORT: &str = "18080";
/// WeightLab WL-18: the env var naming the GPU1 production marker (as `scripts/live/_common.py`).
pub const PRODUCTION_MARKER_ENV: &str = "CERE_GPU1_PRODUCTION_MARKER";
/// The default production marker path (`scripts/live/_common.py` `PRODUCTION_MARKER`).
pub const PRODUCTION_MARKER: &str = "/var/lib/cere/gpu1.production";

/// The production marker's path: [`PRODUCTION_MARKER_ENV`] when set and non-empty, else [`PRODUCTION_MARKER`].
pub fn production_marker_path() -> PathBuf {
    match std::env::var_os(PRODUCTION_MARKER_ENV) {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => PathBuf::from(PRODUCTION_MARKER),
    }
}

/// The `uuid` of the production marker at `path` (`scripts/live/_common.py` `gpu1_production`): `None` when the
/// file is missing; an unreadable or malformed marker, or one without a string `uuid`, reads `uuid` "unknown"
/// there, which equals no device uuid, so it is `None` here too.
pub fn production_marker_uuid(path: &Path) -> Option<String> {
    let raw = std::fs::read(path).ok()?;
    let record: serde_json::Value = serde_json::from_slice(&raw).ok()?;
    record
        .get("uuid")?
        .as_str()
        .filter(|u| !u.is_empty())
        .map(str::to_owned)
}

/// m32n design note D2 (out of scope: more than two cards): the replica count of `[[deployments.native.replicas]]`.
pub const NATIVE_REPLICAS: usize = 2;

/// ab-split design note AB-01: `[deployments.native.ab]`. `salt` keys the session hash (a label, not a secret:
/// a new salt reshuffles every session); `arms` are 1..=[`AB_MAX_ARMS`] arms with unique ids.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbConfig {
    pub salt: String,
    pub arms: Vec<AbArmConfig>,
}

/// One arm: `id` (the only name the stats report), `weight` (its share of new sessions; 0 takes none) and its
/// runtime-layer lever choices (lever name -> bit-identical implementation id; absent levers keep the active set).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbArmConfig {
    pub id: String,
    pub weight: u32,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub levers: BTreeMap<String, String>,
}

/// loop-guard design note: `[deployments.native.loop_guard]`. Absent keys take [`LoopGuardConfig::default`]; a
/// detector's 0 (`min_repeats`, `tool_repeat_run`) turns it off, and a table with both off is refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoopGuardConfig {
    /// LG-D1: the shortest repeating unit in bytes of output text; a suffix whose smallest period is shorter is
    /// never flagged (`=====`, `0, 0, 0`, `x = 0\n` runs).
    pub min_period_bytes: u32,
    /// LG-D1: the longest repeating unit checked; bounds the per-token window.
    pub max_period_bytes: u32,
    /// LG-D1: consecutive copies of the unit that end the output; 0 = LG-D1 off.
    pub min_repeats: u32,
    /// LG-D1: bytes the copies must cover together.
    pub min_span_bytes: u32,
    /// LG-D2: identical assistant tool call -> tool result pairs ending the conversation that refuse the request;
    /// 0 = LG-D2 off.
    pub tool_repeat_run: u32,
    /// LG-D2: top-level argument keys left out of the call comparison (race `qwen2`: Bash's `description`).
    pub ignore_arg_keys: Vec<String>,
}

impl Default for LoopGuardConfig {
    fn default() -> Self {
        Self {
            min_period_bytes: 16,
            max_period_bytes: 1024,
            min_repeats: 4,
            min_span_bytes: 512,
            tool_repeat_run: 10,
            ignore_arg_keys: vec!["description".to_owned()],
        }
    }
}

/// LG-D1: the most bytes one in-response check scans (`max(min_span_bytes, min_repeats x max_period_bytes)`).
pub const LOOP_GUARD_MAX_WINDOW_BYTES: u64 = 65_536;

impl LoopGuardConfig {
    /// LG-D1: the bytes of output text one check scans.
    pub fn window_bytes(&self) -> u64 {
        u64::from(self.min_span_bytes)
            .max(u64::from(self.min_repeats) * u64::from(self.max_period_bytes))
    }
}

/// AB-01: the most arms one deployment splits between.
pub const AB_MAX_ARMS: usize = 16;
/// AB-01: the largest arm weight.
pub const AB_MAX_WEIGHT: u64 = 1_000_000;

/// Upper bound of `[deployments.native] prefix_state_slots` (the header's `CERE_MAX_PREFIX_STATE_SLOTS`).
pub const MAX_PREFIX_STATE_SLOTS: u64 = crate::native::ffi::MAX_PREFIX_STATE_SLOTS;

/// Upper bound of `[deployments.native] decode_rounds_per_prefill_chunk` (the header's
/// `CERE_MAX_DECODE_ROUNDS_PER_CHUNK`).
pub const MAX_DECODE_ROUNDS_PER_CHUNK: u64 = crate::native::ffi::MAX_DECODE_ROUNDS_PER_CHUNK as u64;

/// Upper bound of `[deployments.native] prefill_slice_ms` (the header's `CERE_MAX_PREFILL_SLICE_MS`).
pub const MAX_PREFILL_SLICE_MS: u64 = crate::native::ffi::MAX_PREFILL_SLICE_MS as u64;

/// Upper bound of `[deployments.native] decode_share_pct` (the header's `CERE_MAX_DECODE_SHARE_PCT`).
pub const MAX_DECODE_SHARE_PCT: u64 = crate::native::ffi::MAX_DECODE_SHARE_PCT as u64;

/// Upper bound of `[deployments.native] prefill_age_calls` (the header's `CERE_MAX_PREFILL_AGE_CALLS`).
pub const MAX_PREFILL_AGE_CALLS: u64 = crate::native::ffi::MAX_PREFILL_AGE_CALLS as u64;

/// m29 design note D11: the `speculative_draft_tokens` values a config may set (off, MTP chain 3/1/4).
pub const SPECULATIVE_DRAFT_TOKENS: [u32; 2] = [0, 4];

/// inkling design note D-I-S9B: the on values of a `family = "inkling2_loop"` config (0 is off): a verify window of
/// 2..8 rows, the unfed token plus 1..7 drafts of the MTP drafter's block (its Markov head drafts at most 7).
pub const INKLING_DRAFT_WINDOWS: std::ops::RangeInclusive<u32> = 2..=8;

impl NativeConfig {
    /// PI-02: the profile's resolved lever set (validated at load, so a loaded config never fails here).
    pub fn lever_set(&self) -> Result<LeverSet, String> {
        LeverSet::from_table(&self.levers, self.speculative_draft_tokens)
    }

    /// The native workers in replica order (m32n design note D2): `replicas`, or the single form as one replica on
    /// `devices[0]` bound to `device_uuid`. Empty only for a config validation refuses.
    pub fn replica_list(&self, devices: &[String]) -> Vec<NativeReplicaConfig> {
        match (&self.replicas, &self.device_uuid, devices.first()) {
            (Some(replicas), _, _) => replicas.clone(),
            (None, Some(uuid), Some(device)) => vec![NativeReplicaConfig {
                device: device.clone(),
                device_uuid: uuid.clone(),
            }],
            _ => Vec::new(),
        }
    }

    /// `cere_model_desc.snapshot_host_bytes` as cered passes it (M2.10 D9, ledger M3). Only with the prefix cache
    /// on (`prefix_state_slots > 0`) is the arena also the cache's host tier (executor gate `snapshot_host_bytes > 0
    /// && prefix_state_slots > 0`, no ABI flag), so then `[state] host_spill_enabled = false` passes 0. With the
    /// cache off the arena holds only M2.11 session snapshots and is passed as configured (zeroing it would take
    /// M2.11's snapshot from a spill-less arena config).
    pub fn desc_snapshot_host_bytes(&self, host_spill_enabled: bool) -> u64 {
        if self.prefix_state_slots > 0 && !host_spill_enabled {
            0
        } else {
            self.snapshot_host_bytes
        }
    }

    /// The executor memory limit in bytes: `executor_memory_limit_bytes`, or the 80 GiB default.
    pub fn executor_memory_limit(&self) -> u64 {
        self.executor_memory_limit_bytes
            .unwrap_or(crate::readiness::NATIVE_MEMORY_BUDGET_BYTES)
    }
}

const DEFAULT_PREEMPT_MIN_TOKENS: u64 = 512;

fn default_preempt_min_tokens() -> u64 {
    DEFAULT_PREEMPT_MIN_TOKENS
}

fn is_default_preempt_min_tokens(v: &u64) -> bool {
    *v == DEFAULT_PREEMPT_MIN_TOKENS
}

fn is_zero<T: Default + PartialEq>(v: &T) -> bool {
    *v == T::default()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionPolicy {
    LruPriority,
    ManualOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManagementConfig {
    pub load_on_demand: bool,
    pub single_flight_loads: bool,
    pub unload_idle: bool,
    pub idle_unload_seconds: u64,
    pub min_resident_seconds: u64,
    pub max_loaded_models_per_device: u64,
    pub max_concurrent_loads: u64,
    pub eviction_policy: EvictionPolicy,
    pub drain_timeout_ms: u64,
    pub preload_recipe_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerConfig {
    pub mode: String,
    pub devices: Vec<String>,
    /// Suspect longer than this → Lost (M3 Q10). Optional so configs without it keep their `config_hash`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_lost_after_ms: Option<u64>,
}

impl SchedulerConfig {
    /// The effective Suspect → Lost bound: absent means 60 s.
    pub fn worker_lost_after(&self) -> Duration {
        Duration::from_millis(
            self.worker_lost_after_ms
                .unwrap_or(DEFAULT_WORKER_LOST_AFTER_MS),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateConfig {
    pub host_spill_enabled: bool,
    pub nvme_spill_enabled: bool,
    pub allow_cross_device_resume: bool,
    /// NetTeleport S1 (net-teleport design note 4.3): the largest state image `POST /v1/cere/state/export` and
    /// `/import` move; absent → 4 GiB. Optional so configs without it keep their `config_hash`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_blob_bytes: Option<u64>,
    /// NetTeleport S1 (D-T1): how long a state export or import waits for an idle executor before it refuses
    /// `state_busy`; absent → 30,000 ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub busy_wait_ms: Option<u64>,
    /// NetTeleport S3 (net-teleport design note 4.2 L1, C7): park the hot prefix states of recently active sessions
    /// sealed under `<ledger.state_dir>/parked/` at a graceful stop and import them at the next start; absent → off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub park_on_stop: Option<bool>,
    /// NetTeleport S3: how many recent distinct prompts the recorder keeps (and a stop parks at most); absent → 8.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub park_max_sessions: Option<u64>,
    /// NetTeleport S3: the image bytes one stop parks at most; absent → 16 GiB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub park_max_bytes: Option<u64>,
    /// NetTeleport S3: how long a stop parks after the drain, most recent first; absent → 10,000 ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub park_deadline_ms: Option<u64>,
}

impl StateConfig {
    pub fn max_blob_bytes(&self) -> u64 {
        self.max_blob_bytes
            .unwrap_or(crate::native::teleport::DEFAULT_MAX_BLOB_BYTES)
    }

    pub fn busy_wait_ms(&self) -> u64 {
        self.busy_wait_ms
            .unwrap_or(crate::native::teleport::DEFAULT_BUSY_WAIT_MS)
    }

    pub fn park_on_stop(&self) -> bool {
        self.park_on_stop.unwrap_or(false)
    }

    pub fn park_max_sessions(&self) -> u64 {
        self.park_max_sessions
            .unwrap_or(crate::native::park::DEFAULT_MAX_SESSIONS)
    }

    pub fn park_max_bytes(&self) -> u64 {
        self.park_max_bytes
            .unwrap_or(crate::native::park::DEFAULT_MAX_BYTES)
    }

    pub fn park_deadline_ms(&self) -> u64 {
        self.park_deadline_ms
            .unwrap_or(crate::native::park::DEFAULT_DEADLINE_MS)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryConfig {
    pub level: String,
    pub raw_content: bool,
    /// Roadmap v0.7 A2: the rules that label each chat request with a client and a workload in its log lines and
    /// `GET /metrics` ([`crate::labels`]); first match wins, none → `other`. Optional so configs without it keep
    /// their `config_hash`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub clients: Vec<ClientRule>,
}

/// One `[[telemetry.clients]]` rule: a request of `scope` (any scope when absent) whose `X-Cere-Priority` presence
/// equals `priority_header` (either when absent), or that names `client` in `X-Cere-Client`, is `client` of
/// `workload`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientRule {
    pub client: String,
    pub workload: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority_header: Option<bool>,
}

/// At most this many `[[telemetry.clients]]` rules: the metrics' label set stays small.
pub const MAX_CLIENT_RULES: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SafetyConfig {
    pub fail_on_unknown_capability: bool,
    pub allow_tool_execution: bool,
    pub allow_power_changes: bool,
    pub allow_cloud_fallback: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiConfig {
    pub max_request_body_bytes: u64,
    pub max_messages: u64,
    pub default_max_output_tokens: u64,
    pub max_output_tokens: u64,
    pub default_deadline_ms: u64,
    pub max_deadline_ms: u64,
    pub stream_buffer_events: u64,
    pub stall_timeout_ms: u64,
    pub shutdown_drain_ms: u64,
    /// Per continue-mode stream: bytes of SSE events kept for resume (Q5); absent → 4 MiB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_buffer_bytes: Option<u64>,
    /// All resume buffers together (Q5); absent → 64 MiB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_buffer_total_bytes: Option<u64>,
    /// How long a resume buffer outlives its request's terminal (Q5); absent → 60 s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_retention_ms: Option<u64>,
}

pub const DEFAULT_RESUME_BUFFER_BYTES: u64 = 4_194_304;
pub const DEFAULT_RESUME_BUFFER_TOTAL_BYTES: u64 = 67_108_864;
pub const DEFAULT_RESUME_RETENTION_MS: u64 = 60_000;
pub const DEFAULT_AGING_MS: u64 = 30_000;
pub const DEFAULT_SCOPE_WEIGHT: u64 = 1;

impl ApiConfig {
    pub fn resume_buffer_bytes(&self) -> u64 {
        self.resume_buffer_bytes
            .unwrap_or(DEFAULT_RESUME_BUFFER_BYTES)
    }

    pub fn resume_buffer_total_bytes(&self) -> u64 {
        self.resume_buffer_total_bytes
            .unwrap_or(DEFAULT_RESUME_BUFFER_TOTAL_BYTES)
    }

    pub fn resume_retention_ms(&self) -> u64 {
        self.resume_retention_ms
            .unwrap_or(DEFAULT_RESUME_RETENTION_MS)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionConfig {
    pub max_backend_inflight: u64,
    pub per_scope_inflight: u64,
    pub queue_depth: u64,
    /// Queue wait after which a waiter is served as `high` priority (Q2); absent → 30 s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aging_ms: Option<u64>,
    /// Team pools (net-teleport design note 4.4, C17): the share of grants batch keeps while both classes wait;
    /// absent -> 0.1 (`crate::pool::DEFAULT_BATCH_FLOOR`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_floor: Option<f64>,
    /// Team pools (Q-T6, Q-T8): how long a non-owner job on an owned worker may keep running once its owner waits
    /// for the worker or an owner-only window holds it; absent -> 120 s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner_wait_cap_ms: Option<u64>,
}

impl AdmissionConfig {
    pub fn aging_ms(&self) -> u64 {
        self.aging_ms.unwrap_or(DEFAULT_AGING_MS)
    }

    pub fn batch_floor(&self) -> f64 {
        self.batch_floor.unwrap_or(crate::pool::DEFAULT_BATCH_FLOOR)
    }

    pub fn owner_wait_cap_ms(&self) -> u64 {
        self.owner_wait_cap_ms
            .unwrap_or(crate::pool::DEFAULT_OWNER_WAIT_CAP_MS)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthConfig {
    pub scopes: Vec<AuthScope>,
    /// Name of the env var holding the operator bearer token of the drain routes (M3.5 Q13); absent → the operator
    /// routes are disabled. Optional so configs without it keep their `config_hash`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operator_token_env: Option<String>,
    /// The configured scope a client request without an `Authorization` header belongs to (g9-prod design note
    /// F2: frontend's substitute proxy sends none). A present header must still name a token (else 401), and the
    /// operator routes ignore it. `server.bind` is always loopback, so only local processes reach it. Absent →
    /// every client request needs a bearer. Optional so configs without it keep their `config_hash`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keyless_scope: Option<String>,
}

/// A scope and the *name* of the environment variable holding its bearer token (never the token).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthScope {
    pub scope_id: String,
    pub token_env: String,
    /// Deficit-round-robin weight of the scope's admission queue (Q1/Q2); absent → 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<u64>,
    /// Tiers this person's state and work may go to (net-teleport design note 4.1, Q-T5); absent → `["own"]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_tiers: Option<Vec<Tier>>,
    /// Team pools (C14): the env vars of this person's further device tokens; each authenticates as this scope like
    /// `token_env`. Absent -> `token_env` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_envs: Option<Vec<String>>,
    /// Team pools (C17): the class of this scope's requests without `X-Cere-Class`; absent -> `interactive`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<crate::pool::Class>,
    /// Team pools (C18): this scope's in-flight cap instead of `admission.per_scope_inflight`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_scope_inflight: Option<u64>,
    /// Team pools (C18): this scope's live-session cap instead of `ledger.max_sessions_per_scope`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_sessions_per_scope: Option<u64>,
    /// Team pools (C18, Q-T7): this person's cap on in-flight `batch` rows; absent -> only the scope cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_batch_inflight: Option<u64>,
}

impl AuthScope {
    pub fn weight(&self) -> u64 {
        self.weight.unwrap_or(DEFAULT_SCOPE_WEIGHT)
    }

    pub fn allowed_tiers(&self) -> &[Tier] {
        self.allowed_tiers
            .as_deref()
            .unwrap_or(crate::tier::DEFAULT_ALLOWED_TIERS)
    }

    /// Every env var naming one of this scope's tokens: `token_env`, then `token_envs`.
    pub fn token_env_names(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.token_env.as_str())
            .chain(self.token_envs.iter().flatten().map(String::as_str))
    }

    pub fn class(&self) -> crate::pool::Class {
        self.class.unwrap_or(crate::pool::DEFAULT_CLASS)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LedgerConfig {
    pub state_dir: PathBuf,
    pub max_records: u64,
    pub terminal_retention_ms: u64,
    pub max_sessions_per_scope: u64,
    pub session_ttl_ms: u64,
    pub max_revisions_per_session: u64,
    /// Period of the orphan load poll (m1close Q6, with `deployments.load_endpoint`); absent → 5 s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orphan_load_poll_ms: Option<u64>,
    /// A full ledger evicts its oldest terminal record (never an in-flight one) to admit a new request
    /// (ops-hardening design note OH-03); absent → false (reject `ledger_full`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evict_terminal_when_full: Option<bool>,
}

pub const DEFAULT_ORPHAN_LOAD_POLL_MS: u64 = 5000;

impl LedgerConfig {
    pub fn orphan_load_poll_ms(&self) -> u64 {
        self.orphan_load_poll_ms
            .unwrap_or(DEFAULT_ORPHAN_LOAD_POLL_MS)
    }

    pub fn evict_terminal_when_full(&self) -> bool {
        self.evict_terminal_when_full.unwrap_or(false)
    }
}

/// `[usage]` (net-teleport design note 4.4, S7): the durable usage journal in `<ledger.state_dir>/usage/`
/// ([`crate::usage`]). Every key is optional.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UsageConfig {
    /// Absent → true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Days of day files kept, today included (Q-T9); absent → [`DEFAULT_USAGE_RETENTION_DAYS`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retention_days: Option<u64>,
}

pub const DEFAULT_USAGE_RETENTION_DAYS: u64 = 90;

impl RuntimeConfig {
    /// Whether `cered` writes the usage journal (`usage.enabled`, default true).
    pub fn usage_enabled(&self) -> bool {
        self.usage.as_ref().and_then(|u| u.enabled).unwrap_or(true)
    }

    pub fn usage_retention_days(&self) -> u64 {
        self.usage
            .as_ref()
            .and_then(|u| u.retention_days)
            .unwrap_or(DEFAULT_USAGE_RETENTION_DAYS)
    }

    fn validate_usage(&self) -> Result<(), ConfigError> {
        match self.usage.as_ref().and_then(|u| u.retention_days) {
            Some(days) => range("usage.retention_days", days, 1, 3650),
            None => Ok(()),
        }
    }
}

/// `[federation]`: the coordinator side of M3.5 (m35 design note Q16). The timings default to
/// `configs/cluster/dev-tailnet.yaml` `control_plane` (Q7), the registry quota to Q14's 64; they are optional so a
/// config that leaves them out keeps its `config_hash` equal to `canonical_digest` of its TOML. Read them through
/// the accessors.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FederationConfig {
    pub cluster_id: String,
    pub coordinator_id: String,
    /// A canonical `<ip>:<port>` literal; the host is in 100.64.0.0/10 or loopback, never a wildcard.
    #[serde(deserialize_with = "federation_listen")]
    pub listen: SocketAddr,
    /// The coordinator's mTLS principal (the only dNSName SAN of its leaf, Q2).
    pub principal: String,
    pub ca_file: PathBuf,
    pub cert_file: PathBuf,
    pub key_file: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heartbeat_interval_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suspect_after_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lost_after_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admission_lease_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rpc_timeout_ms: Option<u64>,
    /// Per-principal unary quota of the coordinator's registry service (Q14).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rpc_quota_per_principal: Option<usize>,
    pub workers: Vec<RemoteWorkerConfig>,
}

/// `dev-tailnet.yaml` `control_plane` values (Q7) and the Q14 quota.
pub const DEFAULT_HEARTBEAT_INTERVAL_MS: u64 = 1000;
pub const DEFAULT_SUSPECT_AFTER_MS: u64 = 5000;
pub const DEFAULT_LOST_AFTER_MS: u64 = 15_000;
pub const DEFAULT_ADMISSION_LEASE_MS: u64 = 4000;
pub const DEFAULT_RPC_TIMEOUT_MS: u64 = 5000;
pub const DEFAULT_RPC_QUOTA: usize = 64;

/// Largest `api.max_request_body_bytes` with `[federation]`: a request granted to a remote worker travels as one
/// SubmitWork (the re-serialized backend body plus authority, ids and the work unit) through a channel capped at
/// `wire::INFERENCE_MESSAGE_MAX` both ways, so 64 KiB of that cap is kept for the envelope. A larger body would pass
/// HTTP admission and fail only when granted to a remote worker. Mirrored in `scripts/cere_contracts.py`.
pub const REMOTE_BODY_MAX: u64 = (crate::fed::wire::INFERENCE_MESSAGE_MAX - 65_536) as u64;

impl FederationConfig {
    fn heartbeat_interval_ms(&self) -> u64 {
        self.heartbeat_interval_ms
            .unwrap_or(DEFAULT_HEARTBEAT_INTERVAL_MS)
    }

    fn suspect_after_ms(&self) -> u64 {
        self.suspect_after_ms.unwrap_or(DEFAULT_SUSPECT_AFTER_MS)
    }

    fn lost_after_ms(&self) -> u64 {
        self.lost_after_ms.unwrap_or(DEFAULT_LOST_AFTER_MS)
    }

    fn admission_lease_ms(&self) -> u64 {
        self.admission_lease_ms
            .unwrap_or(DEFAULT_ADMISSION_LEASE_MS)
    }

    fn rpc_timeout_ms(&self) -> u64 {
        self.rpc_timeout_ms.unwrap_or(DEFAULT_RPC_TIMEOUT_MS)
    }

    /// The effective heartbeat interval (Q7).
    pub fn heartbeat_interval(&self) -> Duration {
        Duration::from_millis(self.heartbeat_interval_ms())
    }

    /// The effective Ready -> Suspect bound, by coordinator receive time (Q7).
    pub fn suspect_after(&self) -> Duration {
        Duration::from_millis(self.suspect_after_ms())
    }

    /// The effective Suspect -> Lost bound (Q7).
    pub fn lost_after(&self) -> Duration {
        Duration::from_millis(self.lost_after_ms())
    }

    /// The effective admission lease granted per heartbeat (Q7).
    pub fn admission_lease(&self) -> Duration {
        Duration::from_millis(self.admission_lease_ms())
    }

    /// The effective per-RPC deadline (Q7).
    pub fn rpc_timeout(&self) -> Duration {
        Duration::from_millis(self.rpc_timeout_ms())
    }

    /// The effective per-principal registry quota (Q14).
    pub fn rpc_quota(&self) -> usize {
        self.rpc_quota_per_principal.unwrap_or(DEFAULT_RPC_QUOTA)
    }
}

fn federation_listen<'de, D: Deserializer<'de>>(d: D) -> Result<SocketAddr, D::Error> {
    listen_literal(d, "federation.listen", false)
}

/// Deserializes a listen literal checked by [`parse_listen`]; the error carries the rule's bare message.
pub(crate) fn listen_literal<'de, D: Deserializer<'de>>(
    d: D,
    field: &str,
    rfc1918: bool,
) -> Result<SocketAddr, D::Error> {
    let text = String::deserialize(d)?;
    parse_listen(field, &text, rfc1918).map_err(|e| serde::de::Error::custom(e.message()))
}

/// `[[federation.workers]]`: one remote worker the coordinator dials.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteWorkerConfig {
    /// Unique; never a `scheduler.devices` id.
    pub worker_id: String,
    /// `https://<literal 100.64.0.0/10, 127.0.0.0/8 or [::1]>:<port>`.
    pub endpoint: String,
    /// The worker's mTLS principal: unique, never the coordinator's.
    pub principal: String,
    /// Coordinator-side cap on this worker's in-flight streams (>= 1, harness revision H1).
    pub max_inflight: u64,
    pub expect: RemoteExpect,
    /// Team pools (net-teleport design note 4.4, C15): the person scope owning this worker; absent -> unowned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    /// This worker's trust tier for everyone but its owner (4.1, C13; `own` for the owner); absent -> `own`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<Tier>,
    /// Owner-only windows (C19), e.g. `"Mon-Fri 09:00-17:00 Europe/Berlin"` ([`crate::pool::Window`]); needs `owner`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner_only: Option<Vec<String>>,
}

impl RemoteWorkerConfig {
    /// The validated endpoint as a socket address.
    pub fn endpoint_addr(&self) -> SocketAddr {
        parse_tailnet_endpoint("federation.workers.endpoint", &self.endpoint)
            .expect("endpoint validated at load")
    }

    pub fn tier(&self) -> Tier {
        self.tier.unwrap_or(crate::tier::DEFAULT_TIER)
    }

    /// The worker's pool policy: owner, tier and the parsed windows (validated at load).
    pub fn pool(&self) -> crate::pool::WorkerPool {
        crate::pool::WorkerPool {
            owner: self.owner.as_deref().map(std::sync::Arc::from),
            tier: self.tier(),
            windows: self
                .owner_only
                .iter()
                .flatten()
                .map(|w| crate::pool::Window::parse(w).expect("owner_only validated at load"))
                .collect(),
        }
    }
}

/// Inventory pins a remote worker must report before Ready (Q10).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteExpect {
    pub served_model_name: String,
    pub tp_size: u64,
    pub kv_cache_dtype: String,
}

/// `schemas/model_recipe.schema.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRecipe {
    pub schema_version: String,
    pub recipe_id: String,
    pub model_ref: String,
    pub backend_profile_id: String,
    pub context_tokens: u64,
    pub required_operations: Vec<String>,
    #[serde(default)]
    pub adapters: Vec<RecipeAdapter>,
    pub residency: RecipeResidency,
    pub sampling_defaults: SamplingDefaults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeAdapter {
    pub id: String,
    pub scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeResidency {
    pub priority: u64,
    pub keep_warm_seconds: u64,
    pub pinned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingDefaults {
    pub temperature: f64,
    pub top_p: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<u64>,
}

/// `schemas/deployment_lock.schema.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentLock {
    pub schema_version: String,
    pub lock_id: String,
    pub recipe_sha256: String,
    pub artifact_sha256: String,
    pub model_content_sha256: String,
    pub tokenizer_sha256: String,
    pub template_sha256: String,
    pub backend_build_sha256: String,
    pub backend_profile_id: String,
    pub effective_context_tokens: u64,
    pub effective_operations: Vec<String>,
    #[serde(default)]
    pub adapter_sha256s: Vec<String>,
}

/// The pinned deployment identity referenced by `deployments.deployment`.
#[derive(Debug, Clone)]
pub struct Deployment {
    pub recipe: ModelRecipe,
    pub lock: DeploymentLock,
    /// SHA-256 (lowercase hex) of the lock file bytes.
    pub lock_sha256: String,
    pub profile: ExecutionCapabilities,
}

/// A bearer token resolved from the environment. `Debug` never prints the value.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretToken(String);

impl SecretToken {
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretToken(<redacted>)")
    }
}

#[derive(Debug, Clone)]
pub struct ScopeCredential {
    pub scope_id: String,
    pub token: SecretToken,
}

impl RuntimeConfig {
    /// Reads, validates and resolves the deployment of the config at `path`. The repository root is the
    /// nearest ancestor of the config file that contains `PACKAGE_VERSION`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::new(format!("cannot read {}: {e}", path.display())))?;
        let repo_root = find_repo_root(path)?;
        let mut config = Self::from_toml_str(&text, &repo_root)?;
        config.source = Some(ConfigSource {
            path: path.to_path_buf(),
            repo_root,
        });
        Ok(config)
    }

    /// Parses and validates TOML text; `repo_root` resolves each `deployments.deployment` and bounds
    /// `ledger.state_dir`.
    pub fn from_toml_str(text: &str, repo_root: &Path) -> Result<Self, ConfigError> {
        let mut config: RuntimeConfig =
            toml::from_str(text).map_err(|e| ConfigError::new(e.to_string().trim_end()))?;
        config.validate(repo_root)?;
        for i in 0..config.deployments.len() {
            if let Some(prefix) = &config.deployments[i].deployment {
                let lock = load_deployment(repo_root, prefix)?;
                config.check_deployment(i, &lock)?;
                config.deployments[i].lock = Some(lock);
            }
        }
        Ok(config)
    }

    /// The index of the entry serving `served_model_name`.
    pub fn deployment_by_model(&self, served_model_name: &str) -> Option<usize> {
        self.deployments
            .iter()
            .position(|d| d.served_model_name.as_deref() == Some(served_model_name))
    }

    /// The index of the entry a local worker (`scheduler.devices` id) belongs to: with one entry every device is
    /// its own, with several the entry whose `replicas` lists the device.
    pub fn deployment_of_device(&self, device: &str) -> Option<usize> {
        if !self.scheduler.devices.iter().any(|d| d == device) {
            return None;
        }
        if self.deployments.len() == 1 {
            return Some(0);
        }
        self.deployments
            .iter()
            .position(|d| d.replicas.iter().flatten().any(|r| r.device == device))
    }

    /// The index of the entry a worker belongs to: a local device ([`Self::deployment_of_device`]) or a remote
    /// `[[federation.workers]]` id, bound by its `expect.served_model_name` (M5 D2(e)).
    pub fn deployment_of_worker(&self, worker_id: &str) -> Option<usize> {
        self.deployment_of_device(worker_id).or_else(|| {
            let fed = self.federation.as_ref()?;
            let worker = fed.workers.iter().find(|w| w.worker_id == worker_id)?;
            self.deployment_by_model(&worker.expect.served_model_name)
        })
    }

    /// SHA-256 (lowercase hex) of the canonical JSON (sorted keys, no whitespace) of the effective config.
    /// Equals `canonical_digest` in `scripts/cere_contracts.py` applied to the same TOML.
    pub fn config_hash(&self) -> String {
        let value = serde_json::to_value(self).expect("runtime config serializes to JSON");
        let canonical = serde_json::to_string(&value).expect("JSON value serializes");
        sha256_hex(canonical.as_bytes())
    }

    /// Resolves every scope token from the process environment; a missing, empty, non-UTF-8 or shared
    /// token is a startup failure.
    pub fn resolve_auth(&self) -> Result<Vec<ScopeCredential>, ConfigError> {
        self.resolve_auth_with(|name| std::env::var_os(name))
    }

    /// `resolve_auth` with an explicit environment lookup.
    pub fn resolve_auth_with(
        &self,
        lookup: impl Fn(&str) -> Option<OsString>,
    ) -> Result<Vec<ScopeCredential>, ConfigError> {
        let mut seen = HashSet::new();
        let mut out = Vec::with_capacity(self.auth.scopes.len());
        for scope in &self.auth.scopes {
            for env in scope.token_env_names() {
                let missing = || {
                    ConfigError::new(format!(
                        "auth token env var {env} for scope {} is missing or empty",
                        scope.scope_id
                    ))
                };
                let value = lookup(env).ok_or_else(missing)?;
                let token = value.into_string().map_err(|_| {
                    ConfigError::new(format!("auth token env var {env} is not UTF-8"))
                })?;
                if token.is_empty() {
                    return Err(missing());
                }
                if !seen.insert(token.clone()) {
                    return Err(ConfigError::new(format!(
                        "auth token env var {env} for scope {} equals another configured token",
                        scope.scope_id
                    )));
                }
                out.push(ScopeCredential {
                    scope_id: scope.scope_id.clone(),
                    token: SecretToken(token),
                });
            }
        }
        Ok(out)
    }

    /// Resolves the operator bearer token of the drain routes (M3.5 Q13) from the process environment. `None` when
    /// `auth.operator_token_env` is absent (the operator routes are disabled); a missing, empty or non-UTF-8 value,
    /// or one equal to a scope token, is a startup failure.
    pub fn resolve_operator_token(&self) -> Result<Option<SecretToken>, ConfigError> {
        self.resolve_operator_token_with(|name| std::env::var_os(name))
    }

    /// `resolve_operator_token` with an explicit environment lookup.
    pub fn resolve_operator_token_with(
        &self,
        lookup: impl Fn(&str) -> Option<OsString>,
    ) -> Result<Option<SecretToken>, ConfigError> {
        let Some(env) = &self.auth.operator_token_env else {
            return Ok(None);
        };
        let missing = || {
            ConfigError::new(format!(
                "auth operator token env var {env} is missing or empty"
            ))
        };
        let token = lookup(env)
            .ok_or_else(missing)?
            .into_string()
            .map_err(|_| {
                ConfigError::new(format!("auth operator token env var {env} is not UTF-8"))
            })?;
        if token.is_empty() {
            return Err(missing());
        }
        let shared = self
            .auth
            .scopes
            .iter()
            .flat_map(AuthScope::token_env_names)
            .any(|e| lookup(e).is_some_and(|v| v == token.as_str()));
        if shared {
            return Err(ConfigError::new("operator token equals a scope token"));
        }
        Ok(Some(SecretToken(token)))
    }

    fn validate(&self, repo_root: &Path) -> Result<(), ConfigError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ConfigError::new(format!(
                "schema_version {:?} unsupported; expected {SCHEMA_VERSION:?}",
                self.schema_version
            )));
        }
        self.validate_server()?;
        self.validate_research_isolation(
            production_marker_uuid(&production_marker_path()).as_deref(),
        )?;
        self.validate_deployments()?;
        self.validate_model_management()?;
        self.validate_scheduler()?;
        self.validate_fixed_policy()?;
        self.validate_limits()?;
        self.validate_auth()?;
        self.validate_clients()?;
        self.validate_ledger(repo_root)?;
        self.validate_federation(repo_root)?;
        self.validate_usage()?;
        self.validate_lora_workers()?;
        self.validate_live_kind()
    }

    /// WeightLab WL-18 (cered part): research must never reach production. With a `[deployments.native.research]`
    /// table, refuses the production bind port, any native `device_uuid` (replicas included) equal to the production
    /// marker's `uuid` (`marker_uuid`; `None` = no marker, no uuid check) and an unset `[auth] operator_token_env`.
    pub fn validate_research_isolation(
        &self,
        marker_uuid: Option<&str>,
    ) -> Result<(), ConfigError> {
        for (i, d) in self.deployments.iter().enumerate() {
            let Some(n) = d.native.as_ref().filter(|n| n.research.is_some()) else {
                continue;
            };
            let p = format!("deployments[{i}].native.research");
            if self.server.bind.rsplit_once(':').map(|(_, port)| port) == Some(PRODUCTION_PORT) {
                return Err(ConfigError::new(format!(
                    "{p} is refused on the production bind port {PRODUCTION_PORT} (server.bind {})",
                    self.server.bind
                )));
            }
            if let Some(marker) = marker_uuid {
                let uuids = n
                    .device_uuid
                    .iter()
                    .chain(n.replicas.iter().flatten().map(|r| &r.device_uuid));
                for uuid in uuids {
                    if uuid == marker {
                        return Err(ConfigError::new(format!(
                            "{p} is refused on the production GPU: device_uuid {uuid} equals the production marker's \
                             uuid"
                        )));
                    }
                }
            }
            if self.auth.operator_token_env.is_none() {
                return Err(ConfigError::new(format!(
                    "{p} requires [auth] operator_token_env (the research routes take the operator bearer)"
                )));
            }
        }
        Ok(())
    }

    /// Track LoRA and WeightLab: an adapter or a research plan lives in one local executor, so no remote worker may
    /// serve a LoRA or research entry.
    fn validate_lora_workers(&self) -> Result<(), ConfigError> {
        let Some(fed) = &self.federation else {
            return Ok(());
        };
        for w in &fed.workers {
            if let Some(d) = self.deployment_of_worker(&w.worker_id) {
                let Some(n) = self.deployments[d].native.as_ref() else {
                    continue;
                };
                for (table, present) in [
                    ("lora", n.lora.is_some()),
                    ("research", n.research.is_some()),
                ] {
                    if present {
                        return Err(ConfigError::new(format!(
                            "deployments[{d}].native.{table} is refused with the federation worker {}",
                            w.worker_id
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_server(&self) -> Result<(), ConfigError> {
        let port = self
            .server
            .bind
            .strip_prefix("127.0.0.1:")
            .filter(|p| (1..=5).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_digit()))
            .ok_or_else(|| ConfigError::new("server.bind must be 127.0.0.1:<port>"))?;
        match port.parse::<u32>() {
            Ok(1..=65535) => Ok(()),
            _ => Err(ConfigError::new("server.bind port must be 1..65535")),
        }
    }

    /// The schema constraints of every `[[deployments]]` entry (M5 D3): 1..=16 entries, each the old backend shape.
    fn validate_deployments(&self) -> Result<(), ConfigError> {
        items("deployments", self.deployments.len(), 1, 16)?;
        for (i, b) in self.deployments.iter().enumerate() {
            let p = format!("deployments[{i}]");
            id(&format!("{p}.profile_id"), &b.profile_id)?;
            if let Some(url) = &b.base_url {
                parse_loopback_base_url(url)?;
            }
            if let Some(name) = &b.served_model_name {
                served_model_name(&format!("{p}.served_model_name"), name)?;
            }
            if let Some(prefix) = &b.deployment {
                let name = prefix.strip_prefix("configs/deployments/").ok_or_else(|| {
                    ConfigError::new(format!("{p}.deployment must be configs/deployments/<name>"))
                })?;
                pattern(&format!("{p}.deployment"), name, 128, is_lower_alnum, |c| {
                    is_lower_alnum(c) || c == b'_' || c == b'-'
                })?;
            }
            if let Some(expect) = &b.expect {
                validate_expect(expect)?;
            }
            if let Some(path) = &b.load_endpoint {
                load_endpoint_path(&format!("{p}.load_endpoint"), path)?;
            }
            if let Some(replicas) = &b.replicas {
                items(&format!("{p}.replicas"), replicas.len(), 1, 64)?;
                for r in replicas {
                    id(&format!("{p}.replicas.device"), &r.device)?;
                    parse_loopback_base_url(&r.base_url)?;
                    if let Some(path) = &r.load_endpoint {
                        load_endpoint_path(&format!("{p}.replicas.load_endpoint"), path)?;
                    }
                }
            }
            if let Some(native) = &b.native {
                validate_native(&p, native)?;
            }
        }
        Ok(())
    }

    fn validate_model_management(&self) -> Result<(), ConfigError> {
        let mm = &self.model_management;
        if !mm.single_flight_loads {
            return Err(ConfigError::new(
                "model_management.single_flight_loads must be true",
            ));
        }
        range(
            "model_management.idle_unload_seconds",
            mm.idle_unload_seconds,
            0,
            604_800,
        )?;
        range(
            "model_management.min_resident_seconds",
            mm.min_resident_seconds,
            0,
            604_800,
        )?;
        range(
            "model_management.max_loaded_models_per_device",
            mm.max_loaded_models_per_device,
            1,
            64,
        )?;
        range(
            "model_management.max_concurrent_loads",
            mm.max_concurrent_loads,
            1,
            16,
        )?;
        range(
            "model_management.drain_timeout_ms",
            mm.drain_timeout_ms,
            1,
            3_600_000,
        )?;
        id_list(
            "model_management.preload_recipe_ids",
            &mm.preload_recipe_ids,
            0,
        )?;
        if mm.unload_idle && mm.idle_unload_seconds < mm.min_resident_seconds {
            return Err(ConfigError::new(
                "idle unload must not undercut minimum residency",
            ));
        }
        let slots = mm.max_loaded_models_per_device * self.scheduler.devices.len() as u64;
        if mm.preload_recipe_ids.len() as u64 > slots {
            return Err(ConfigError::new(
                "preload set exceeds configured model slots",
            ));
        }
        Ok(())
    }

    fn validate_scheduler(&self) -> Result<(), ConfigError> {
        let s = &self.scheduler;
        if s.mode != "replica" {
            return Err(ConfigError::new("scheduler.mode must be replica"));
        }
        id_list("scheduler.devices", &s.devices, 1)?;
        if let Some(ms) = s.worker_lost_after_ms {
            range("scheduler.worker_lost_after_ms", ms, 1000, 3_600_000)?;
        }
        Ok(())
    }

    fn validate_fixed_policy(&self) -> Result<(), ConfigError> {
        if self.state.allow_cross_device_resume {
            return Err(ConfigError::new("cross-device resume not implemented"));
        }
        // NetTeleport S1: 64 KiB..64 GiB of image, 0..600 s of busy wait.
        if let Some(bytes) = self.state.max_blob_bytes {
            range("state.max_blob_bytes", bytes, 1 << 16, 64 << 30)?;
        }
        if let Some(ms) = self.state.busy_wait_ms {
            range("state.busy_wait_ms", ms, 0, 600_000)?;
        }
        // NetTeleport S3: 1..256 recorded prompts, 64 KiB..1 TiB of parked images, 0..120 s of park after the
        // drain; parking runs on the single native form only (one executor holds the prefix cache).
        if let Some(n) = self.state.park_max_sessions {
            range("state.park_max_sessions", n, 1, 256)?;
        }
        if let Some(bytes) = self.state.park_max_bytes {
            range("state.park_max_bytes", bytes, 1 << 16, 1 << 40)?;
        }
        if let Some(ms) = self.state.park_deadline_ms {
            range("state.park_deadline_ms", ms, 0, 120_000)?;
        }
        if self.state.park_on_stop() {
            let single_native = self.deployments.first().is_some_and(|d| {
                d.kind == BackendKind::NativeCuda
                    && d.native.as_ref().is_some_and(|n| n.replicas.is_none())
            });
            if !single_native {
                return Err(ConfigError::new(
                    "state.park_on_stop needs deployments[0] native_cuda in the single form (no replicas)",
                ));
            }
        }
        if self.telemetry.level != "metadata" || self.telemetry.raw_content {
            return Err(ConfigError::new(
                "telemetry must be metadata-only without raw content",
            ));
        }
        let s = &self.safety;
        if !s.fail_on_unknown_capability
            || s.allow_tool_execution
            || s.allow_power_changes
            || s.allow_cloud_fallback
        {
            return Err(ConfigError::new("safety settings are fixed by the schema"));
        }
        Ok(())
    }

    /// `[[telemetry.clients]]` (A2): at most [`MAX_CLIENT_RULES`], labels `^[a-z0-9][a-z0-9._-]{0,31}$` other than
    /// `other` (the label of no match), and a rule's scope a configured auth scope.
    fn validate_clients(&self) -> Result<(), ConfigError> {
        let rules = &self.telemetry.clients;
        if rules.len() > MAX_CLIENT_RULES {
            return Err(ConfigError::new(format!(
                "telemetry.clients lists at most {MAX_CLIENT_RULES} rules"
            )));
        }
        let label = |field: &str, value: &str| -> Result<(), ConfigError> {
            pattern(field, value, 32, is_lower_alnum, |c| {
                is_lower_alnum(c) || c == b'.' || c == b'_' || c == b'-'
            })?;
            if value == crate::labels::OTHER {
                return Err(ConfigError::new(format!(
                    "{field} {value:?} is reserved for requests no rule matches"
                )));
            }
            Ok(())
        };
        for r in rules {
            label("telemetry.clients.client", &r.client)?;
            label("telemetry.clients.workload", &r.workload)?;
            if let Some(scope) = &r.scope {
                if !self.auth.scopes.iter().any(|s| &s.scope_id == scope) {
                    return Err(ConfigError::new(format!(
                        "telemetry.clients scope {scope:?} is not an auth scope"
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_limits(&self) -> Result<(), ConfigError> {
        let a = &self.api;
        range(
            "api.max_request_body_bytes",
            a.max_request_body_bytes,
            1024,
            67_108_864,
        )?;
        range("api.max_messages", a.max_messages, 1, 10_000)?;
        range(
            "api.default_max_output_tokens",
            a.default_max_output_tokens,
            1,
            131_072,
        )?;
        range("api.max_output_tokens", a.max_output_tokens, 1, 131_072)?;
        range(
            "api.default_deadline_ms",
            a.default_deadline_ms,
            1,
            86_400_000,
        )?;
        range("api.max_deadline_ms", a.max_deadline_ms, 1, 86_400_000)?;
        range(
            "api.stream_buffer_events",
            a.stream_buffer_events,
            1,
            65_536,
        )?;
        range("api.stall_timeout_ms", a.stall_timeout_ms, 1, 600_000)?;
        range("api.shutdown_drain_ms", a.shutdown_drain_ms, 0, 600_000)?;
        if let Some(bytes) = a.resume_buffer_bytes {
            range("api.resume_buffer_bytes", bytes, 4096, 268_435_456)?;
        }
        if let Some(bytes) = a.resume_buffer_total_bytes {
            range(
                "api.resume_buffer_total_bytes",
                bytes,
                65_536,
                4_294_967_296,
            )?;
        }
        if let Some(ms) = a.resume_retention_ms {
            range("api.resume_retention_ms", ms, 1000, 3_600_000)?;
        }
        if a.default_max_output_tokens > a.max_output_tokens {
            return Err(ConfigError::new(
                "api default_max_output_tokens exceeds max_output_tokens",
            ));
        }
        if a.default_deadline_ms > a.max_deadline_ms {
            return Err(ConfigError::new(
                "api default_deadline_ms exceeds max_deadline_ms",
            ));
        }
        let ad = &self.admission;
        range(
            "admission.max_backend_inflight",
            ad.max_backend_inflight,
            1,
            4096,
        )?;
        range(
            "admission.per_scope_inflight",
            ad.per_scope_inflight,
            1,
            4096,
        )?;
        range("admission.queue_depth", ad.queue_depth, 1, 10_000)?;
        if let Some(ms) = ad.aging_ms {
            range("admission.aging_ms", ms, 1000, 3_600_000)?;
        }
        if let Some(floor) = ad.batch_floor {
            if !(0.0..=0.5).contains(&floor) {
                return Err(ConfigError::new(
                    "admission.batch_floor must be within 0.0..=0.5",
                ));
            }
        }
        if let Some(ms) = ad.owner_wait_cap_ms {
            range("admission.owner_wait_cap_ms", ms, 1000, 600_000)?;
        }
        if ad.per_scope_inflight > ad.max_backend_inflight {
            return Err(ConfigError::new(
                "per-scope in-flight cap exceeds the global backend cap",
            ));
        }
        // Every backend slot can hold a full per-request resume buffer, so live continue streams alone never
        // exceed the total (resume.rs keeps a self-eviction only as the backstop).
        if a.resume_buffer_total_bytes() < a.resume_buffer_bytes() * ad.max_backend_inflight {
            return Err(ConfigError::new(
                "api resume_buffer_total_bytes is below resume_buffer_bytes x admission.max_backend_inflight",
            ));
        }
        Ok(())
    }

    fn validate_auth(&self) -> Result<(), ConfigError> {
        let scopes = &self.auth.scopes;
        if scopes.is_empty() || scopes.len() > 64 {
            return Err(ConfigError::new("auth.scopes must list 1..64 scopes"));
        }
        let (mut ids, mut envs) = (HashSet::new(), HashSet::new());
        for s in scopes {
            pattern(
                "auth.scopes.scope_id",
                &s.scope_id,
                64,
                is_lower_alnum,
                |c| is_lower_alnum(c) || c == b'_' || c == b'-',
            )?;
            for env in s.token_env_names() {
                pattern(
                    "auth.scopes.token_env",
                    env,
                    64,
                    |c| c.is_ascii_uppercase(),
                    |c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'_',
                )?;
                if !envs.insert(env) {
                    return Err(ConfigError::new(
                        "one token_env cannot authenticate two scopes",
                    ));
                }
            }
            if let Some(extra) = &s.token_envs {
                items("auth.scopes.token_envs", extra.len(), 1, 16)?;
            }
            if let Some(cap) = s.per_scope_inflight {
                range("auth.scopes.per_scope_inflight", cap, 1, 4096)?;
                if cap > self.admission.max_backend_inflight {
                    return Err(ConfigError::new(
                        "per-scope in-flight cap exceeds the global backend cap",
                    ));
                }
            }
            if let Some(cap) = s.max_sessions_per_scope {
                range("auth.scopes.max_sessions_per_scope", cap, 1, 1_000_000)?;
            }
            if let Some(cap) = s.max_batch_inflight {
                range("auth.scopes.max_batch_inflight", cap, 1, 4096)?;
            }
            if let Some(weight) = s.weight {
                range("auth.scopes.weight", weight, 1, 100)?;
            }
            if let Some(tiers) = &s.allowed_tiers {
                let distinct: HashSet<_> = tiers.iter().collect();
                if tiers.is_empty() || distinct.len() != tiers.len() {
                    return Err(ConfigError::new(
                        "auth.scopes.allowed_tiers must list 1..3 distinct tiers",
                    ));
                }
            }
            if !ids.insert(&s.scope_id) {
                return Err(ConfigError::new("duplicate auth scope_id"));
            }
        }
        if let Some(env) = &self.auth.operator_token_env {
            pattern(
                "auth.operator_token_env",
                env,
                64,
                |c| c.is_ascii_uppercase(),
                |c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'_',
            )?;
            if envs.contains(env.as_str()) {
                return Err(ConfigError::new(
                    "auth.operator_token_env must differ from every scope token_env",
                ));
            }
        }
        if let Some(scope) = &self.auth.keyless_scope {
            if !ids.contains(scope) {
                return Err(ConfigError::new(
                    "auth.keyless_scope must name a configured auth scope",
                ));
            }
        }
        Ok(())
    }

    fn validate_ledger(&self, repo_root: &Path) -> Result<(), ConfigError> {
        let l = &self.ledger;
        absolute_path("ledger.state_dir", &l.state_dir)?;
        if in_checkout(&l.state_dir, repo_root) {
            return Err(ConfigError::new(
                "ledger state_dir must be outside the repository",
            ));
        }
        range("ledger.max_records", l.max_records, 1, 1_000_000)?;
        range(
            "ledger.terminal_retention_ms",
            l.terminal_retention_ms,
            1000,
            604_800_000,
        )?;
        range(
            "ledger.max_sessions_per_scope",
            l.max_sessions_per_scope,
            1,
            1_000_000,
        )?;
        range(
            "ledger.session_ttl_ms",
            l.session_ttl_ms,
            1000,
            2_592_000_000,
        )?;
        range(
            "ledger.max_revisions_per_session",
            l.max_revisions_per_session,
            1,
            100_000,
        )?;
        if let Some(ms) = l.orphan_load_poll_ms {
            range("ledger.orphan_load_poll_ms", ms, 1000, 600_000)?;
        }
        Ok(())
    }

    /// `[federation]` (M3.5 Q16, Q7, Q14): the backend kind and the request body bound `REMOTE_BODY_MAX`, then the
    /// schema constraints, then the listen, endpoint, PEM, timing and identity rules. The global cap with remote
    /// workers and each worker's deployment (M5 D2(e)) are checked with the `sglang_http` rules.
    fn validate_federation(&self, repo_root: &Path) -> Result<(), ConfigError> {
        let Some(fed) = &self.federation else {
            return Ok(());
        };
        if self
            .deployments
            .iter()
            .any(|d| d.kind != BackendKind::SglangHttp)
        {
            return Err(ConfigError::new(
                "[federation] requires backend kind sglang_http",
            ));
        }
        if self.api.max_request_body_bytes > REMOTE_BODY_MAX {
            return Err(ConfigError::new(format!(
                "api.max_request_body_bytes must be <= {REMOTE_BODY_MAX} with [federation] (one SubmitWork under the \
                 remote message cap)"
            )));
        }
        id("federation.cluster_id", &fed.cluster_id)?;
        id("federation.coordinator_id", &fed.coordinator_id)?;
        principal("federation.principal", &fed.principal)?;
        for (field, path) in [
            ("federation.ca_file", &fed.ca_file),
            ("federation.cert_file", &fed.cert_file),
            ("federation.key_file", &fed.key_file),
        ] {
            absolute_path(field, path)?;
        }
        range(
            "federation.heartbeat_interval_ms",
            fed.heartbeat_interval_ms(),
            100,
            600_000,
        )?;
        for (field, ms) in [
            ("federation.suspect_after_ms", fed.suspect_after_ms()),
            ("federation.lost_after_ms", fed.lost_after_ms()),
            ("federation.admission_lease_ms", fed.admission_lease_ms()),
        ] {
            range(field, ms, 100, 3_600_000)?;
        }
        range(
            "federation.rpc_timeout_ms",
            fed.rpc_timeout_ms(),
            100,
            600_000,
        )?;
        range(
            "federation.rpc_quota_per_principal",
            fed.rpc_quota() as u64,
            1,
            4096,
        )?;
        items("federation.workers", fed.workers.len(), 1, 64)?;
        for w in &fed.workers {
            id("federation.workers.worker_id", &w.worker_id)?;
            chars("federation.workers.endpoint", &w.endpoint, 1, 128)?;
            principal("federation.workers.principal", &w.principal)?;
            range("federation.workers.max_inflight", w.max_inflight, 1, 4096)?;
            served_model_name(
                "federation.workers.expect.served_model_name",
                &w.expect.served_model_name,
            )?;
            range("federation.workers.expect.tp_size", w.expect.tp_size, 1, 64)?;
            pattern(
                "federation.workers.expect.kv_cache_dtype",
                &w.expect.kv_cache_dtype,
                64,
                is_alnum,
                |c| is_alnum(c) || b"_.-".contains(&c),
            )?;
        }
        pem_paths(
            "federation.",
            [&fed.ca_file, &fed.cert_file, &fed.key_file],
            Some(repo_root),
            &self.ledger.state_dir,
        )?;
        let (heartbeat, suspect, lost, lease) = (
            fed.heartbeat_interval_ms(),
            fed.suspect_after_ms(),
            fed.lost_after_ms(),
            fed.admission_lease_ms(),
        );
        if !(heartbeat < lease && lease < lost) {
            return Err(ConfigError::new(
                "federation timings must satisfy heartbeat_interval_ms < admission_lease_ms < lost_after_ms",
            ));
        }
        if suspect <= heartbeat {
            return Err(ConfigError::new(
                "federation suspect_after_ms must exceed heartbeat_interval_ms",
            ));
        }
        if lost <= suspect {
            return Err(ConfigError::new(
                "federation lost_after_ms must exceed suspect_after_ms",
            ));
        }
        if lease >= suspect {
            return Err(ConfigError::new(
                "federation admission_lease_ms must be below suspect_after_ms",
            ));
        }
        let (mut ids, mut principals) = (HashSet::new(), HashSet::new());
        for w in &fed.workers {
            parse_tailnet_endpoint("federation.workers.endpoint", &w.endpoint)?;
            if !ids.insert(&w.worker_id) {
                return Err(ConfigError::new(format!(
                    "duplicate federation worker_id {}",
                    w.worker_id
                )));
            }
            if self.scheduler.devices.contains(&w.worker_id) {
                return Err(ConfigError::new(format!(
                    "federation worker_id equals a scheduler device: {}",
                    w.worker_id
                )));
            }
            if w.principal == fed.principal {
                return Err(ConfigError::new(
                    "federation worker principal equals the coordinator principal",
                ));
            }
            if !principals.insert(&w.principal) {
                return Err(ConfigError::new(format!(
                    "duplicate federation worker principal {}",
                    w.principal
                )));
            }
        }
        self.validate_team_pool(fed)
    }

    /// Team pools (net-teleport design note 4.4): a worker's `owner` names a configured scope, `owner_only` needs
    /// an owner and parses ([`crate::pool::Window`]: its zone must exist in the system database), and with an owned
    /// worker no scope may be the remote readiness smoke's (the agent lets that scope through owner-only).
    fn validate_team_pool(&self, fed: &FederationConfig) -> Result<(), ConfigError> {
        let scopes: HashSet<&str> = self
            .auth
            .scopes
            .iter()
            .map(|s| s.scope_id.as_str())
            .collect();
        for w in &fed.workers {
            if let Some(owner) = &w.owner {
                if !scopes.contains(owner.as_str()) {
                    return Err(ConfigError::new(format!(
                        "federation worker {} owner must name a configured auth scope",
                        w.worker_id
                    )));
                }
            }
            let Some(windows) = &w.owner_only else {
                continue;
            };
            if w.owner.is_none() {
                return Err(ConfigError::new(format!(
                    "federation worker {} owner_only needs an owner",
                    w.worker_id
                )));
            }
            items(
                "federation.workers.owner_only",
                windows.len(),
                1,
                crate::pool::MAX_WINDOWS,
            )?;
            for text in windows {
                crate::pool::Window::parse(text).map_err(ConfigError::new)?;
            }
        }
        if fed.workers.iter().any(|w| w.owner.is_some())
            && scopes.contains(crate::readiness::REMOTE_SMOKE_SCOPE)
        {
            return Err(ConfigError::new(format!(
                "auth scope {} is reserved for the remote readiness smoke when a federation worker has an owner",
                crate::readiness::REMOTE_SMOKE_SCOPE
            )));
        }
        Ok(())
    }

    /// Schema conditionals plus the per-kind semantic rules that need no deployment files. Mode `native` and kind
    /// `native_cuda` imply each other. Planning kinds and `native_cuda` allow exactly one entry (M5 D2(h)).
    fn validate_live_kind(&self) -> Result<(), ConfigError> {
        for b in &self.deployments {
            if (b.mode == BackendMode::Native) != (b.kind == BackendKind::NativeCuda) {
                return Err(ConfigError::new(
                    "backend mode native requires kind native_cuda, and native_cuda requires mode native",
                ));
            }
        }
        if self.deployments.len() > 1
            && self
                .deployments
                .iter()
                .any(|d| d.kind != BackendKind::SglangHttp)
        {
            return Err(ConfigError::new(
                "more than one deployment requires every entry to be sglang_http (native_cuda, unconfigured and cpu_reference allow exactly one)",
            ));
        }
        match self.deployments[0].kind {
            BackendKind::Unconfigured | BackendKind::CpuReference => self.validate_planning_kind(),
            BackendKind::SglangHttp => self.validate_sglang_http(),
            BackendKind::NativeCuda => self.validate_native_cuda(),
        }
    }

    fn validate_planning_kind(&self) -> Result<(), ConfigError> {
        let b = &self.deployments[0];
        if b.base_url.is_some()
            || b.replicas.is_some()
            || b.served_model_name.is_some()
            || b.deployment.is_some()
            || b.expect.is_some()
        {
            return Err(ConfigError::new(
                "base_url, replicas, served_model_name, deployment and expect need a live backend kind",
            ));
        }
        if b.load_endpoint.is_some() {
            return Err(ConfigError::new(
                "deployments[0].load_endpoint is sglang_http-only",
            ));
        }
        if b.native.is_some() {
            return Err(ConfigError::new(
                "deployments[0].native is native_cuda-only",
            ));
        }
        if !self.planning_only {
            return Err(ConfigError::new(
                "planning_only=false requires a live backend kind",
            ));
        }
        Ok(())
    }

    /// Every entry is `sglang_http` (M5 D2): one entry keeps the M3 rules (exactly one of `base_url` and
    /// `replicas`); several entries refuse `base_url` and make `replicas` optional. Then the cross-entry rules:
    /// unique names and locks (c), disjoint replica devices covering `scheduler.devices` (d), each remote worker
    /// bound to the entry serving its `expect.served_model_name` (e), a worker per entry (f), equal cache keys (g),
    /// the summed global cap (j) and the cross-scope rule per entry (i).
    fn validate_sglang_http(&self) -> Result<(), ConfigError> {
        let multi = self.deployments.len() > 1;
        let mut expects = Vec::with_capacity(self.deployments.len());
        for (i, b) in self.deployments.iter().enumerate() {
            let p = format!("deployments[{i}]");
            if multi {
                if b.base_url.is_some() {
                    return Err(ConfigError::new(format!(
                        "{p}.base_url is refused with more than one deployment; use {p}.replicas"
                    )));
                }
            } else if b.base_url.is_some() == b.replicas.is_some() {
                return Err(ConfigError::new(format!(
                    "sglang_http requires exactly one of {p}.base_url and {p}.replicas"
                )));
            }
            let complete = b.served_model_name.is_some() && b.deployment.is_some();
            let Some(expect) = b.expect.as_ref().filter(|_| complete) else {
                return Err(ConfigError::new(
                    "sglang_http requires served_model_name, deployment and expect",
                ));
            };
            if b.native.is_some() {
                return Err(ConfigError::new(format!("{p}.native is native_cuda-only")));
            }
            if multi && b.load_endpoint.is_some() {
                return Err(ConfigError::new(format!(
                    "{p}.load_endpoint is refused with more than one deployment; set it per replica"
                )));
            }
            expects.push(expect);
        }
        self.validate_live_common()?;
        let (mut names, mut locks) = (HashSet::new(), HashSet::new());
        for b in &self.deployments {
            let name = b.served_model_name.as_deref().unwrap_or_default();
            if !names.insert(name) {
                return Err(ConfigError::new(format!(
                    "duplicate deployments served_model_name {name}"
                )));
            }
            let lock = b.deployment.as_deref().unwrap_or_default();
            if !locks.insert(lock) {
                return Err(ConfigError::new(format!(
                    "duplicate deployments deployment {lock}"
                )));
            }
        }
        let local = if self.deployments[0].base_url.is_some() {
            vec![self.single_worker()?]
        } else {
            self.validate_replicas()?
        };
        let remote = self.remote_workers_by_deployment()?;
        for (i, (l, r)) in local.iter().zip(&remote).enumerate() {
            if *l == 0 && r.is_empty() {
                return Err(ConfigError::new(format!(
                    "deployments[{i}] has no worker (no local replica and no remote worker)"
                )));
            }
        }
        let first = &self.deployments[0];
        for (i, b) in self.deployments.iter().enumerate().skip(1) {
            if b.shared_backend_tenancy != first.shared_backend_tenancy
                || b.allow_cross_scope_prefix_cache != first.allow_cross_scope_prefix_cache
                || b.prefix_cache_isolation() != first.prefix_cache_isolation()
            {
                return Err(ConfigError::new(format!(
                    "deployments[{i}] shared_backend_tenancy, allow_cross_scope_prefix_cache and prefix_cache_isolation must equal deployments[0]'s"
                )));
            }
        }
        // Global cap ≤ Σ_entries N_e × min(per-worker cap of e); every replica's cap is its entry's
        // `expect.max_running_requests` (M3 Q6). With `[federation]`, N_e also counts e's remote workers and each of
        // their `max_inflight` bounds e's per-worker cap (M3.5 Q16, harness revision H1, M5 D2(j)).
        let cap: u64 = expects
            .iter()
            .zip(local.iter().zip(&remote))
            .map(|(expect, (l, r))| {
                let per_worker = r
                    .iter()
                    .map(|w| w.max_inflight)
                    .fold(expect.max_running_requests, u64::min);
                (l + r.len() as u64) * per_worker
            })
            .sum();
        if self.admission.max_backend_inflight > cap {
            let formula = if self.federation.is_some() {
                "local + remote workers x min(expect.max_running_requests, every max_inflight)"
            } else {
                "replicas x expect.max_running_requests"
            };
            let summed = if multi {
                ", summed over deployments"
            } else {
                ""
            };
            return Err(ConfigError::new(format!(
                "admission max_backend_inflight exceeds {formula}{summed}"
            )));
        }
        for b in &self.deployments {
            self.validate_cross_scope(b)?;
        }
        Ok(())
    }

    /// The cross-scope rule of a backend radix cache (Q2; M2.10 D2 for the native cache): with more than one
    /// auth scope, scopes share prefixes only by `allow_cross_scope_prefix_cache`, else every scope needs its own
    /// namespace (`prefix_cache_isolation = "cache_salt"`). Shared by `validate_sglang_http` (every entry) and
    /// `validate_native_cuda` (when `prefix_state_slots > 0`; a cache-off native worker shares nothing).
    fn validate_cross_scope(&self, b: &DeploymentConfig) -> Result<(), ConfigError> {
        if self.auth.scopes.len() > 1
            && !b.allow_cross_scope_prefix_cache
            && b.prefix_cache_isolation() != PrefixCacheIsolation::CacheSalt
        {
            return Err(ConfigError::new(
                "more than one auth scope on a shared radix cache requires allow_cross_scope_prefix_cache or prefix_cache_isolation = \"cache_salt\"",
            ));
        }
        Ok(())
    }

    /// The single backend form is one worker on `scheduler.devices[0]` (M3 Q3), so it lists exactly one device.
    fn single_worker(&self) -> Result<u64, ConfigError> {
        if self.scheduler.devices.len() != 1 {
            return Err(ConfigError::new(
                "single backend form needs exactly one scheduler device",
            ));
        }
        Ok(1)
    }

    /// `[[deployments.replicas]]` (M3 Q2, M5 D2(b)(d)): `load_endpoint` moves per replica; devices are unique within
    /// and across entries, at least one entry has a replica, and together they are exactly `scheduler.devices`;
    /// base URLs are unique after normalisation. `parse_loopback_base_url` admits only a literal host and an
    /// explicit port without a trailing slash, so its `(host, port)` is the normal form. Returns each entry's
    /// local worker count.
    fn validate_replicas(&self) -> Result<Vec<u64>, ConfigError> {
        let mut owners: HashMap<&str, usize> = HashMap::new();
        let mut urls = HashSet::new();
        let mut counts = Vec::with_capacity(self.deployments.len());
        for (i, b) in self.deployments.iter().enumerate() {
            let replicas = b.replicas.as_deref().unwrap_or_default();
            if !replicas.is_empty() && b.load_endpoint.is_some() {
                return Err(ConfigError::new(format!(
                    "deployments[{i}].load_endpoint is refused with deployments[{i}].replicas; set it per replica"
                )));
            }
            for r in replicas {
                if let Some(j) = owners.insert(r.device.as_str(), i) {
                    return Err(ConfigError::new(if j == i {
                        format!("duplicate replica device {}", r.device)
                    } else {
                        format!(
                            "replica device {} belongs to deployments[{j}] and deployments[{i}]",
                            r.device
                        )
                    }));
                }
                if !urls.insert(parse_loopback_base_url(&r.base_url)?) {
                    return Err(ConfigError::new(format!(
                        "duplicate replica base_url {}",
                        r.base_url
                    )));
                }
                if !self.scheduler.devices.contains(&r.device) {
                    return Err(ConfigError::new(format!(
                        "replica device is not in scheduler.devices: {}",
                        r.device
                    )));
                }
            }
            counts.push(replicas.len() as u64);
        }
        if owners.is_empty() {
            return Err(ConfigError::new(
                "at least one deployment needs a local replica",
            ));
        }
        if self.scheduler.devices.len() != owners.len() {
            return Err(ConfigError::new(
                "scheduler.devices must list exactly the replica devices",
            ));
        }
        Ok(counts)
    }

    /// `[[federation.workers]]` grouped by entry (M5 D2(e)): each belongs to the entry whose `served_model_name`
    /// equals its `expect.served_model_name`.
    fn remote_workers_by_deployment(&self) -> Result<Vec<Vec<&RemoteWorkerConfig>>, ConfigError> {
        let mut out = vec![Vec::new(); self.deployments.len()];
        for w in self.federation.iter().flat_map(|f| &f.workers) {
            let i = self
                .deployment_by_model(&w.expect.served_model_name)
                .ok_or_else(|| {
                    ConfigError::new(format!(
                        "federation worker {} expect.served_model_name {} matches no deployments served_model_name",
                        w.worker_id, w.expect.served_model_name
                    ))
                })?;
            out[i].push(w);
        }
        Ok(out)
    }

    /// `native_cuda` (spec m22 "Native path contract" → Config): no HTTP backend to reach or pin, so `base_url`,
    /// `expect` and `load_endpoint` are refused. `prefix_cache_isolation` is accepted (M2.10 D2: `cache_salt`
    /// namespaces each scope's sessions) and, with `prefix_state_slots > 0`, held to the shared cross-scope rule.
    /// One executor binds one device per process, so `replicas` is refused and one device is listed (M3 Q13).
    fn validate_native_cuda(&self) -> Result<(), ConfigError> {
        let b = &self.deployments[0];
        if b.replicas.is_some() {
            return Err(ConfigError::new(
                "native_cuda refuses deployments[0].replicas",
            ));
        }
        if b.base_url.is_some() || b.expect.is_some() || b.load_endpoint.is_some() {
            return Err(ConfigError::new(
                "native_cuda refuses base_url, expect and load_endpoint",
            ));
        }
        if b.served_model_name.is_none() || b.deployment.is_none() || b.native.is_none() {
            return Err(ConfigError::new(
                "native_cuda requires served_model_name, deployment and deployments[0].native",
            ));
        }
        // M2.11 D8 (F6): a preempted generation's snapshot is host-RAM spill.
        if b.native.as_ref().is_some_and(|n| n.preempt) && !self.state.host_spill_enabled {
            return Err(ConfigError::new(
                "deployments[0].native.preempt requires [state] host_spill_enabled",
            ));
        }
        if b.native.as_ref().is_some_and(|n| n.prefix_state_slots > 0) {
            self.validate_cross_scope(b)?;
        }
        self.validate_live_common()?;
        self.native_workers().map(|_| ())
    }

    /// The local native workers (m32n design note D2): the single form is one worker on `scheduler.devices[0]`;
    /// with `[[deployments.native.replicas]]` every replica device is one of `scheduler.devices`, and together they
    /// are exactly `scheduler.devices`.
    fn native_workers(&self) -> Result<u64, ConfigError> {
        let Some(replicas) = self.deployments[0]
            .native
            .as_ref()
            .and_then(|n| n.replicas.as_ref())
        else {
            return self.single_worker();
        };
        for r in replicas {
            if !self.scheduler.devices.contains(&r.device) {
                return Err(ConfigError::new(format!(
                    "native replica device is not in scheduler.devices: {}",
                    r.device
                )));
            }
        }
        if self.scheduler.devices.len() != replicas.len() {
            return Err(ConfigError::new(
                "scheduler.devices must list exactly the native replica devices",
            ));
        }
        Ok(replicas.len() as u64)
    }

    /// Rules every live kind shares: a live config is not planning-only and manages no model lifecycle itself.
    fn validate_live_common(&self) -> Result<(), ConfigError> {
        if self.planning_only {
            return Err(ConfigError::new(
                "a live backend kind requires planning_only=false",
            ));
        }
        let mm = &self.model_management;
        if mm.load_on_demand
            || mm.unload_idle
            || mm.eviction_policy != EvictionPolicy::ManualOnly
            || !mm.preload_recipe_ids.is_empty()
        {
            return Err(ConfigError::new(
                "a live backend kind requires load_on_demand=false, unload_idle=false, eviction_policy=manual_only, preload_recipe_ids=[]",
            ));
        }
        Ok(())
    }

    /// Cross-checks between entry `i` of the validated config and its deployment files.
    fn check_deployment(&self, i: usize, d: &Deployment) -> Result<(), ConfigError> {
        let b = &self.deployments[i];
        let p = format!("deployments[{i}]");
        if d.profile.profile_id != b.profile_id {
            return Err(ConfigError::new(
                "backend profile_id differs from the deployment profile",
            ));
        }
        let mode = match b.mode {
            BackendMode::BackendManaged => "backend_managed",
            BackendMode::Native => "native",
        };
        if d.profile.execution_mode != mode {
            return Err(ConfigError::new(
                "deployment execution mode differs from backend mode",
            ));
        }
        match (&b.expect, &b.native) {
            (Some(expect), None) => {
                check_sglang_deployment(expect, d)?;
                // Each replica's backend cap is one worker's cap (M3 Q6).
                if expect.max_running_requests > d.profile.decode_batch_max {
                    return Err(ConfigError::new(
                        "expect.max_running_requests exceeds the profile's validated rows",
                    ));
                }
            }
            (None, Some(native)) => {
                check_native_deployment(&p, native, d, self.state.host_spill_enabled)?;
                // Each native worker's cap is the profile's validated rows (M3 Q6, Q13), plus the sessions U1 parks
                // in the snapshot arena (M2.11 D8); m32n design note D16: the global cap is replicas x that.
                let parked = if native.preempt {
                    native.max_parked_sessions
                } else {
                    0
                };
                let workers = native.replicas.as_ref().map_or(1, |r| r.len() as u64);
                if self.admission.max_backend_inflight
                    > workers.saturating_mul(d.profile.decode_batch_max.saturating_add(parked))
                {
                    return Err(ConfigError::new(if workers == 1 {
                        "admission max_backend_inflight exceeds the profile's validated rows (+ max_parked_sessions with preempt)"
                    } else {
                        "admission max_backend_inflight exceeds native replicas x the profile's validated rows (+ max_parked_sessions with preempt)"
                    }));
                }
                // sessions design note SE-09: a generation that used its reserved session up parks itself to grow
                // it, so every generation in flight must be able to park at once (no wait for a park slot that only
                // another waiting grower could free).
                if native.reserve_output_tokens > 0
                    && native.max_parked_sessions < self.admission.max_backend_inflight
                {
                    return Err(ConfigError::new(format!(
                        "{p}.native.reserve_output_tokens requires max_parked_sessions >= admission.max_backend_inflight"
                    )));
                }
            }
            _ => {
                return Err(ConfigError::new(format!(
                    "a deployment needs exactly one of {p}.expect and {p}.native"
                )))
            }
        }
        if !d.lock.effective_operations.iter().any(|op| op == "chat") {
            return Err(ConfigError::new("deployment lock does not provide chat"));
        }
        Ok(())
    }
}

/// `sglang_http`: the deployment pins the served model path, the backend image and the context.
fn check_sglang_deployment(expect: &BackendExpect, d: &Deployment) -> Result<(), ConfigError> {
    if d.recipe.model_ref != format!("local://{}", expect.model_path.trim_start_matches('/')) {
        return Err(ConfigError::new(
            "deployment recipe does not pin expect.model_path",
        ));
    }
    if format!("sha256:{}", d.lock.backend_build_sha256) != expect.image_digest {
        return Err(ConfigError::new(
            "deployment lock does not pin expect.image_digest",
        ));
    }
    if d.profile.max_context_tokens != expect.context_length {
        return Err(ConfigError::new(
            "deployment context differs from expect.context_length",
        ));
    }
    Ok(())
}

/// SelfTune ST4 S3a (fail-first): what a `family = "sdar"` deployment refuses, by key name. An SDAR library (DICE-8B)
/// runs one session, eager, greedy, with no draft head, prefix cache, snapshot, LoRA or research points (ST4 plan
/// D4), and the AMD HIP port is built for the Qwen family only. Its levers are refused in
/// [`LeverSet::check_target`]. A `qwen4_exp` deployment passes unchanged.
/// Inkling 2 Mini (inkling design note S0, D-I5): a `family = "inkling2_loop"` deployment refuses the same keys
/// except decode rows > 1, a Vulkan device (S11 is the Vulkan port) and the MTP drafter (D-I-S9B: its window is
/// validated with the other values, [`INKLING_DRAFT_WINDOWS`]; CUDA only: the Vulkan port has no drafter): the prefix
/// cache and snapshots of its rings and conv states (S6), preemption, LoRA and research (out of scope), the lever hot
/// swap, a hip device (no HIP executor), and a profile advertising a grammar output (S6 v1: auto tool calls only).
fn check_family(
    p: &str,
    native: &NativeConfig,
    d: &Deployment,
    uuids: &[&str],
) -> Result<(), ConfigError> {
    let family = native.family;
    if family == Family::Qwen4Exp {
        return Ok(());
    }
    let refused = |key: &str| {
        Err(ConfigError::new(format!(
            "{key} is refused with {p}.native.family = \"{}\"",
            family.name()
        )))
    };
    if native.speculative_draft_tokens != 0 && family != Family::Inkling2Loop {
        return refused(&format!("{p}.native.speculative_draft_tokens"));
    }
    if native.prefix_state_slots > 0 {
        return refused(&format!("{p}.native.prefix_state_slots"));
    }
    if native.snapshot_host_bytes > 0 || d.profile.native_state_snapshot {
        return refused(&format!("{p}.native.snapshot_host_bytes"));
    }
    if native.preempt {
        return refused(&format!("{p}.native.preempt"));
    }
    if native.lora.is_some() {
        return refused(&format!("{p}.native.lora"));
    }
    if native.research.is_some() {
        return refused(&format!("{p}.native.research"));
    }
    if native.lever_hot_swap {
        return refused(&format!("{p}.native.lever_hot_swap"));
    }
    if family == Family::Inkling2Loop {
        if uuids
            .iter()
            .any(|u| Target::arch_of(u) == crate::levers::Arch::Gfx1151)
        {
            return refused(&format!(
                "{p}.native.device_uuid on the hip arch (hip:<ordinal>)"
            ));
        }
        if native.speculative_draft_tokens != 0
            && uuids
                .iter()
                .any(|u| Target::arch_of(u) != crate::levers::Arch::Sm120a)
        {
            return refused(&format!(
                "{p}.native.speculative_draft_tokens on a non-CUDA device"
            ));
        }
        // S6 v1 serves auto tool calls only: the grammar sidecar builds qwen3_coder structural tags, and a
        // from-start JSON grammar forbids Inkling's framing tokens (<|message_model|><|content_text|>).
        if crate::native::advertises_grammar(&d.profile) {
            return refused("profile structured outputs / tool_calls_constrained");
        }
        return Ok(());
    }
    if uuids
        .iter()
        .any(|u| Target::arch_of(u) != crate::levers::Arch::Sm120a)
    {
        return refused(&format!(
            "{p}.native.device_uuid on a non-CUDA arch (hip:<ordinal> or vk:<ordinal>)"
        ));
    }
    if d.profile.decode_batch_max > 1 {
        return refused("profile decode_batch_max > 1");
    }
    Ok(())
}

/// `native_cuda`: the lock pins the executor library and the artifact manifest; the profile pins the context.
fn check_native_deployment(
    p: &str,
    native: &NativeConfig,
    d: &Deployment,
    host_spill_enabled: bool,
) -> Result<(), ConfigError> {
    if d.lock.backend_build_sha256 != native.library_sha256 {
        return Err(ConfigError::new(format!(
            "deployment lock backend_build_sha256 differs from {p}.native.library_sha256"
        )));
    }
    if d.lock.artifact_sha256 != native.expected_manifest_sha256 {
        return Err(ConfigError::new(format!(
            "deployment lock artifact_sha256 differs from {p}.native.expected_manifest_sha256"
        )));
    }
    if d.profile.max_context_tokens != native.max_context_tokens {
        return Err(ConfigError::new(format!(
            "deployment context differs from {p}.native.max_context_tokens"
        )));
    }
    // SelfTune D3 (fail-first): every selected lever impl was proven on the device's arch and covers the profile's
    // decode rows and context.
    let levers = native
        .lever_set()
        .map_err(|e| ConfigError::new(format!("{p}.native.levers.{e}")))?;
    let uuids: Vec<&str> = match (&native.device_uuid, &native.replicas) {
        (Some(uuid), _) => vec![uuid.as_str()],
        (None, Some(replicas)) => replicas.iter().map(|r| r.device_uuid.as_str()).collect(),
        (None, None) => Vec::new(),
    };
    check_family(p, native, d, &uuids)?;
    for uuid in uuids {
        let target = Target {
            arch: Target::arch_of(uuid),
            max_rows: d.profile.decode_batch_max,
            max_context: native.max_context_tokens,
            speculative_draft_tokens: native.speculative_draft_tokens,
            family: native.family,
        };
        levers
            .check_target(&target)
            .map_err(|e| ConfigError::new(format!("{p}.native.levers.{e}")))?;
    }
    // M2.11: the executor reports `native_state_snapshot` exactly when it has an arena (ledger C 56), so a config
    // that disagrees with its profile could never pass readiness. The arena is the one cered passes (M2.10 D9).
    if (native.desc_snapshot_host_bytes(host_spill_enabled) > 0) != d.profile.native_state_snapshot
    {
        return Err(ConfigError::new(format!(
            "{p}.native.snapshot_host_bytes > 0 must equal the profile's native_state_snapshot"
        )));
    }
    // m29 design note D11/D12 (I3): the executor reports `native_speculative_batch` exactly when the model has a
    // draft, so a config that disagrees with its profile could never pass readiness.
    if (native.speculative_draft_tokens > 0) != d.profile.native_speculative_batch {
        return Err(ConfigError::new(format!(
            "{p}.native.speculative_draft_tokens > 0 must equal the profile's native_speculative_batch"
        )));
    }
    // D8 composition (I2) was refused with decode_batch_max > 1 until rows could leave a batch for a park: a row's
    // DECODE ends on its own cancel, a snapshot runs on the driver thread between rounds, and a restored session joins
    // the next round like a fresh one (sessions design note SE-09; M2.6 D6 row invariance).
    let parked_bytes = crate::native::snapshot::arena_slot_bytes(
        native.max_context_tokens,
        native.speculative_draft_tokens > 0,
    )
    .and_then(|bytes| bytes.checked_mul(native.max_parked_sessions));
    if parked_bytes.is_none_or(|bytes| bytes > native.snapshot_host_bytes) {
        return Err(ConfigError::new(format!(
            "{p}.native.max_parked_sessions x the snapshot bytes at max_context_tokens exceed snapshot_host_bytes"
        )));
    }
    // D13 (ledger P1 suffix rule, LR5): the dev knob runs only on the dev/live snapshot profile.
    if native.preempt_every > 0
        && !(d.profile.native_state_snapshot && d.profile.profile_id.ends_with("-snapshot"))
    {
        return Err(ConfigError::new(format!(
            "{p}.native.preempt_every requires the dev/live <profile id>-snapshot profile"
        )));
    }
    // D13: the knob forces U1 preemptions, so it needs U1.
    if native.preempt_every > 0 && !native.preempt {
        return Err(ConfigError::new(format!(
            "{p}.native.preempt_every requires preempt = true"
        )));
    }
    // M211R-04: a forced preemption parks the victim, so it needs room for one parked session.
    if native.preempt_every > 0 && native.max_parked_sessions == 0 {
        return Err(ConfigError::new(format!(
            "{p}.native.preempt_every requires max_parked_sessions >= 1"
        )));
    }
    // SE-09: the reserve is whole 64-token pages (M2.5 D4) and the grow is a U1 park.
    if native.reserve_output_tokens % 64 != 0 {
        return Err(ConfigError::new(format!(
            "{p}.native.reserve_output_tokens must be a multiple of 64"
        )));
    }
    if native.reserve_output_tokens > 0 && !native.preempt {
        return Err(ConfigError::new(format!(
            "{p}.native.reserve_output_tokens requires preempt = true"
        )));
    }
    Ok(())
}

/// Nearest ancestor of `config_path` containing a `PACKAGE_VERSION` file.
pub fn find_repo_root(config_path: &Path) -> Result<PathBuf, ConfigError> {
    find_checkout(config_path)?
        .ok_or_else(|| ConfigError::new("config file is not inside a CERE repository checkout"))
}

/// [`find_repo_root`] for configs that need no checkout (the `cere-worker` agent): `None` when no ancestor holds
/// `PACKAGE_VERSION`.
pub fn find_checkout(config_path: &Path) -> Result<Option<PathBuf>, ConfigError> {
    let canonical = config_path
        .canonicalize()
        .map_err(|e| ConfigError::new(format!("cannot resolve {}: {e}", config_path.display())))?;
    Ok(canonical
        .ancestors()
        .skip(1)
        .find(|dir| dir.join("PACKAGE_VERSION").is_file())
        .map(Path::to_path_buf))
}

/// A canonical `<IPv4>:<port>` or `[<IPv6>]:<port>` literal, port 1..65535, never a wildcard; the host is a loopback
/// or 100.64.0.0/10 (Tailscale CGNAT) literal, plus RFC1918 when `rfc1918` (the agent in a bridge container, M3.5
/// Q16). Mirrors `parse_listen` in `scripts/cere_contracts.py`.
pub fn parse_listen(field: &str, text: &str, rfc1918: bool) -> Result<SocketAddr, ConfigError> {
    let bad = || {
        ConfigError::new(format!(
            "{field} must be a canonical <ip>:<port> literal with a port 1..65535"
        ))
    };
    let (host, port) = text.rsplit_once(':').ok_or_else(bad)?;
    let port = literal_port(port).ok_or_else(bad)?;
    let ip = match host.strip_prefix('[').and_then(|h| h.strip_suffix(']')) {
        Some(v6) => v6
            .parse::<Ipv6Addr>()
            .ok()
            .filter(|ip| ip.to_string() == v6)
            .map(IpAddr::V6),
        None => host
            .parse::<Ipv4Addr>()
            .ok()
            .filter(|ip| ip.to_string() == host)
            .map(IpAddr::V4),
    }
    .ok_or_else(bad)?;
    if ip.is_unspecified() {
        return Err(ConfigError::new(format!(
            "{field} must not be a wildcard address"
        )));
    }
    let allowed = match ip {
        IpAddr::V6(v6) => v6 == Ipv6Addr::LOCALHOST,
        IpAddr::V4(v4) => v4.is_loopback() || is_tailnet(v4) || (rfc1918 && v4.is_private()),
    };
    if !allowed {
        let kinds = if rfc1918 {
            "a loopback, 100.64.0.0/10 or RFC1918"
        } else {
            "a 100.64.0.0/10 or loopback"
        };
        return Err(ConfigError::new(format!(
            "{field} host must be {kinds} literal"
        )));
    }
    Ok(SocketAddr::new(ip, port))
}

/// `https://<literal 100.64.0.0/10, 127.0.0.0/8 or [::1]>:<port 1..65535>`, nothing after the port: a federation
/// peer is dialled by address and authenticated by its principal (M3.5 Q2, Q16). Mirrors `parse_tailnet_endpoint`
/// in `scripts/cere_contracts.py`.
pub fn parse_tailnet_endpoint(field: &str, url: &str) -> Result<SocketAddr, ConfigError> {
    let rest = url
        .strip_prefix("https://")
        .ok_or_else(|| ConfigError::new(format!("{field} must use https://")))?;
    let (host, port) = rest
        .rsplit_once(':')
        .and_then(|(host, port)| Some((host, literal_port(port)?)))
        .ok_or_else(|| {
            ConfigError::new(format!(
                "{field} needs an explicit port 1..65535 and nothing after it"
            ))
        })?;
    if host == "[::1]" {
        return Ok(SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), port));
    }
    match host.parse::<Ipv4Addr>() {
        Ok(ip) if ip.to_string() == host && (ip.is_loopback() || is_tailnet(ip)) => {
            Ok(SocketAddr::new(IpAddr::V4(ip), port))
        }
        _ => Err(ConfigError::new(format!(
            "{field} host must be a literal 100.64.0.0/10, 127.0.0.0/8 or [::1] address"
        ))),
    }
}

/// A decimal port `[1-9][0-9]{0,4}` in 1..65535.
fn literal_port(port: &str) -> Option<u16> {
    let digits_ok = matches!(port.as_bytes().first(), Some(b'1'..=b'9'))
        && port.len() <= 5
        && port.bytes().all(|b| b.is_ascii_digit());
    port.parse::<u16>().ok().filter(|_| digits_ok)
}

/// Tailscale's CGNAT range 100.64.0.0/10.
fn is_tailnet(ip: Ipv4Addr) -> bool {
    let [a, b, ..] = ip.octets();
    a == 100 && b & 0xc0 == 64
}

/// The schema's absolute path pattern `^(/[A-Za-z0-9_][A-Za-z0-9._-]{0,254}){1,64}$` (no `.`/`..` components).
pub(crate) fn absolute_path(field: &str, path: &Path) -> Result<(), ConfigError> {
    let text = path
        .to_str()
        .ok_or_else(|| ConfigError::new(format!("{field} must be UTF-8")))?;
    let components: Vec<&str> = text
        .strip_prefix('/')
        .ok_or_else(|| ConfigError::new(format!("{field} must be absolute")))?
        .split('/')
        .collect();
    if components.len() > 64 {
        return Err(ConfigError::new(format!("{field} has too many components")));
    }
    for component in components {
        pattern(
            &format!("{field} component"),
            component,
            255,
            |c| is_alnum(c) || c == b'_',
            |c| is_alnum(c) || b"._-".contains(&c),
        )?;
    }
    Ok(())
}

/// Whether `path` is the checkout `root` or inside it, lexically against `root` and its canonical form.
pub(crate) fn in_checkout(path: &Path, root: &Path) -> bool {
    let canonical = root.canonicalize().ok();
    std::iter::once(root)
        .chain(canonical.as_deref())
        .any(|r| path.starts_with(r))
}

/// The PEM rules by shape only, nothing is opened (M3.5 Q16): outside the checkout (when there is one) and not
/// under the state dir. `paths` are `ca_file`, `cert_file`, `key_file`, already checked by [`absolute_path`].
pub(crate) fn pem_paths(
    prefix: &str,
    paths: [&PathBuf; 3],
    checkout: Option<&Path>,
    state_dir: &Path,
) -> Result<(), ConfigError> {
    for (key, path) in ["ca_file", "cert_file", "key_file"].into_iter().zip(paths) {
        if checkout.is_some_and(|root| in_checkout(path, root)) {
            return Err(ConfigError::new(format!(
                "{prefix}{key} must be outside any repository checkout"
            )));
        }
        if path.starts_with(state_dir) {
            return Err(ConfigError::new(format!(
                "{prefix}{key} must not be under the state_dir"
            )));
        }
    }
    Ok(())
}

/// An mTLS principal: a lowercase DNS name of at most 253 bytes (M3.5 Q2), the schema's principal pattern.
pub(crate) fn principal(field: &str, value: &str) -> Result<(), ConfigError> {
    let label_ok = |label: &str| {
        let b = label.as_bytes();
        (1..=63).contains(&b.len())
            && is_lower_alnum(b[0])
            && is_lower_alnum(b[b.len() - 1])
            && b.iter().all(|&c| is_lower_alnum(c) || c == b'-')
    };
    if (1..=253).contains(&value.len()) && value.split('.').all(label_ok) {
        Ok(())
    } else {
        Err(ConfigError::new(format!(
            "{field} does not match its pattern"
        )))
    }
}

/// The schema's served model name pattern `^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$`.
pub(crate) fn served_model_name(field: &str, value: &str) -> Result<(), ConfigError> {
    pattern(field, value, 128, is_alnum, |c| {
        is_alnum(c) || b"_.:/-".contains(&c)
    })
}

/// `http://<127.0.0.0/8 literal or [::1]>:<port 1..65535>`; no userinfo, path, query, fragment or
/// trailing slash. Mirrors `parse_loopback_base_url` in `scripts/cere_contracts.py`.
pub fn parse_loopback_base_url(url: &str) -> Result<(String, u16), ConfigError> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| ConfigError::new("backend base_url must use http://"))?;
    let (host, port) = rest.rsplit_once(':').ok_or_else(bad_port)?;
    let port = literal_port(port).ok_or_else(bad_port)?;
    if host == "[::1]" {
        return Ok(("::1".to_owned(), port));
    }
    match host.parse::<Ipv4Addr>() {
        Ok(ip) if ip.is_loopback() && ip.to_string() == host => Ok((host.to_owned(), port)),
        _ => Err(ConfigError::new(
            "backend base_url host must be a literal 127.0.0.0/8 address or [::1]",
        )),
    }
}

fn bad_port() -> ConfigError {
    ConfigError::new("backend base_url needs an explicit port 1..65535 and nothing after it")
}

/// A backend load path such as `/get_load`.
pub(crate) fn load_endpoint_path(field: &str, path: &str) -> Result<(), ConfigError> {
    pattern(
        field,
        path,
        128,
        |c| c == b'/',
        |c| is_alnum(c) || b"_./-".contains(&c),
    )
}

fn validate_expect(e: &BackendExpect) -> Result<(), ConfigError> {
    pattern("expect.version", &e.version, 64, is_alnum, |c| {
        is_alnum(c) || b".+_-".contains(&c)
    })?;
    for (field, value) in [
        ("expect.model_path", &e.model_path),
        ("expect.tokenizer_path", &e.tokenizer_path),
    ] {
        pattern(
            field,
            value,
            1024,
            |c| c == b'/',
            |c| is_alnum(c) || b"._/-".contains(&c),
        )?;
    }
    if let Some(template) = &e.chat_template {
        if !(1..=4096).contains(&template.chars().count()) {
            return Err(ConfigError::new(
                "expect.chat_template must be 1..4096 characters",
            ));
        }
    }
    let speculative = [
        ("speculative_algorithm", e.speculative_algorithm.is_some()),
        ("speculative_num_steps", e.speculative_num_steps.is_some()),
        ("speculative_eagle_topk", e.speculative_eagle_topk.is_some()),
        (
            "speculative_num_draft_tokens",
            e.speculative_num_draft_tokens.is_some(),
        ),
    ];
    if speculative.iter().any(|(_, set)| *set) {
        let missing: Vec<&str> = speculative
            .iter()
            .filter(|(_, set)| !set)
            .map(|(name, _)| *name)
            .collect();
        if !missing.is_empty() {
            return Err(ConfigError::new(format!(
                "expect speculative keys are all present or all absent; missing: {}",
                missing.join(", ")
            )));
        }
    }
    for (field, value) in [
        ("expect.quantization", e.quantization.as_ref()),
        ("expect.kv_cache_dtype", Some(&e.kv_cache_dtype)),
        (
            "expect.speculative_algorithm",
            e.speculative_algorithm.as_ref(),
        ),
        ("expect.reasoning_parser", Some(&e.reasoning_parser)),
        ("expect.tool_call_parser", Some(&e.tool_call_parser)),
    ] {
        let Some(value) = value else { continue };
        pattern(field, value, 64, is_alnum, |c| {
            is_alnum(c) || b"_.-".contains(&c)
        })?;
    }
    for (field, value, max) in [
        ("expect.speculative_num_steps", e.speculative_num_steps, 64),
        (
            "expect.speculative_eagle_topk",
            e.speculative_eagle_topk,
            64,
        ),
        (
            "expect.speculative_num_draft_tokens",
            e.speculative_num_draft_tokens,
            256,
        ),
    ] {
        if let Some(value) = value {
            range(field, value, 1, max)?;
        }
    }
    range("expect.context_length", e.context_length, 1, 16_777_216)?;
    range(
        "expect.max_running_requests",
        e.max_running_requests,
        1,
        4096,
    )?;
    if e.allow_auto_truncate {
        return Err(ConfigError::new("expect.allow_auto_truncate must be false"));
    }
    match e.image_digest.strip_prefix("sha256:") {
        Some(hex) if is_sha256_hex(hex) => Ok(()),
        _ => Err(ConfigError::new(
            "expect.image_digest must be sha256:<64 lowercase hex>",
        )),
    }
}

/// The schema constraints of `[deployments.native]`; `p` is the entry's `deployments[<i>]`.
fn validate_native(p: &str, n: &NativeConfig) -> Result<(), ConfigError> {
    for (field, value) in [("library", &n.library), ("artifact_dir", &n.artifact_dir)] {
        pattern(
            &format!("{p}.native.{field}"),
            value,
            1024,
            |c| c == b'/',
            |c| is_alnum(c) || b"._/-".contains(&c),
        )?;
    }
    sha256_field(&format!("{p}.native.library_sha256"), &n.library_sha256)?;
    sha256_field(
        &format!("{p}.native.expected_manifest_sha256"),
        &n.expected_manifest_sha256,
    )?;
    match (&n.device_uuid, &n.replicas) {
        (Some(uuid), None)
            if is_gpu_uuid(uuid) || hip_ordinal(uuid).is_some() || vk_ordinal(uuid).is_some() =>
        {}
        (Some(_), None) => {
            return Err(ConfigError::new(format!(
                "{p}.native.device_uuid must be GPU-<8-4-4-4-12 lowercase hex>, hip:<ordinal> or vk:<ordinal>"
            )))
        }
        (Some(_), Some(_)) => {
            return Err(ConfigError::new(format!(
                "{p}.native.device_uuid is refused with {p}.native.replicas; set it per replica"
            )))
        }
        (None, None) => {
            return Err(ConfigError::new(format!(
                "{p}.native requires exactly one of device_uuid and replicas"
            )))
        }
        (None, Some(replicas)) => {
            if replicas.len() != NATIVE_REPLICAS {
                return Err(ConfigError::new(format!(
                    "{p}.native.replicas must list exactly {NATIVE_REPLICAS} replicas"
                )));
            }
            // The dev hot swap is one executor's (PI-04); the replicas' levers are deployment-level (D2).
            if n.lever_hot_swap {
                return Err(ConfigError::new(format!(
                    "{p}.native.lever_hot_swap is refused with {p}.native.replicas"
                )));
            }
            let (mut devices, mut uuids) = (HashSet::new(), HashSet::new());
            for r in replicas {
                id(&format!("{p}.native.replicas.device"), &r.device)?;
                if !is_gpu_uuid(&r.device_uuid) {
                    return Err(ConfigError::new(format!(
                        "{p}.native.replicas.device_uuid must be GPU-<8-4-4-4-12 lowercase hex>"
                    )));
                }
                if !devices.insert(r.device.as_str()) {
                    return Err(ConfigError::new(format!(
                        "duplicate native replica device {}",
                        r.device
                    )));
                }
                if !uuids.insert(r.device_uuid.as_str()) {
                    return Err(ConfigError::new(format!(
                        "duplicate native replica device_uuid {}",
                        r.device_uuid
                    )));
                }
            }
        }
    }
    range(
        &format!("{p}.native.max_context_tokens"),
        n.max_context_tokens,
        1,
        16_777_216,
    )?;
    match n.kv_pool_tokens {
        None | Some(0) => {}
        Some(pool) if pool % 64 != 0 => {
            return Err(ConfigError::new(format!(
                "{p}.native.kv_pool_tokens must be 0 or a multiple of 64"
            )))
        }
        Some(pool) if pool < n.max_context_tokens => {
            return Err(ConfigError::new(format!(
                "{p}.native.kv_pool_tokens must be 0 or at least {p}.native.max_context_tokens"
            )))
        }
        Some(_) => {}
    }
    range(
        &format!("{p}.native.prefix_state_slots"),
        n.prefix_state_slots,
        0,
        MAX_PREFIX_STATE_SLOTS,
    )?;
    if let Some(rounds) = n.decode_rounds_per_prefill_chunk {
        range(
            &format!("{p}.native.decode_rounds_per_prefill_chunk"),
            rounds,
            1,
            MAX_DECODE_ROUNDS_PER_CHUNK,
        )?;
    }
    if let Some(slice_ms) = n.prefill_slice_ms {
        range(
            &format!("{p}.native.prefill_slice_ms"),
            slice_ms,
            1,
            MAX_PREFILL_SLICE_MS,
        )?;
    }
    if let Some(share_pct) = n.decode_share_pct {
        range(
            &format!("{p}.native.decode_share_pct"),
            share_pct,
            1,
            MAX_DECODE_SHARE_PCT,
        )?;
    }
    if let Some(calls) = n.prefill_age_calls {
        range(
            &format!("{p}.native.prefill_age_calls"),
            calls,
            1,
            MAX_PREFILL_AGE_CALLS,
        )?;
    }
    if n.prefix_end_state && n.prefix_state_slots == 0 {
        return Err(ConfigError::new(format!(
            "{p}.native.prefix_end_state requires prefix_state_slots > 0"
        )));
    }
    if n.prefix_alt_edges && n.prefix_state_slots == 0 {
        return Err(ConfigError::new(format!(
            "{p}.native.prefix_alt_edges requires prefix_state_slots > 0"
        )));
    }
    if let Some(dir) = &n.host_share_dir {
        pattern(
            &format!("{p}.native.host_share_dir"),
            dir,
            1024,
            |c| c == b'/',
            |c| is_alnum(c) || b"._/-".contains(&c),
        )?;
    }
    if n.keep_host_share && n.host_share_dir.is_none() {
        return Err(ConfigError::new(format!(
            "{p}.native.keep_host_share requires {p}.native.host_share_dir"
        )));
    }
    let sidecar = format!("{p}.native.sidecar");
    items(&sidecar, n.sidecar.len(), 1, 64)?;
    let argv = |c: u8| is_alnum(c) || b"._/=:+-".contains(&c);
    for arg in &n.sidecar {
        pattern(&sidecar, arg, 1024, argv, argv)?;
    }
    if n.sidecar_deadline_ms != 0 {
        range(
            &format!("{p}.native.sidecar_deadline_ms"),
            n.sidecar_deadline_ms,
            1_000,
            3_600_000,
        )?;
    }
    range(
        &format!("{p}.native.preempt_min_tokens"),
        n.preempt_min_tokens,
        1,
        16_777_216,
    )?;
    if n.preempt && n.snapshot_host_bytes == 0 {
        return Err(ConfigError::new(format!(
            "{p}.native.preempt requires snapshot_host_bytes > 0"
        )));
    }
    let (ok, values) = if n.family == Family::Inkling2Loop {
        (
            n.speculative_draft_tokens == 0
                || INKLING_DRAFT_WINDOWS.contains(&n.speculative_draft_tokens),
            "0 or 2..8 with family inkling2_loop",
        )
    } else {
        (
            SPECULATIVE_DRAFT_TOKENS.contains(&n.speculative_draft_tokens),
            "0 or 4",
        )
    };
    if !ok {
        return Err(ConfigError::new(format!(
            "{p}.native.speculative_draft_tokens must be {values}"
        )));
    }
    // m23 design note PI-02: fail-closed against the lever registry (unknown lever or impl, a value the model's
    // drafts cannot run).
    if let Err(e) = n.lever_set() {
        return Err(ConfigError::new(format!("{p}.native.levers.{e}")));
    }
    // The hot swap (`cere_model_set_selection`) does not flush the prefix cache: a NO_SPECULATION request after a swap of
    // a numerics-changing lever could hit states computed under the previous levers.
    if n.lever_hot_swap && n.prefix_state_slots > 0 {
        return Err(ConfigError::new(format!(
            "{p}.native.lever_hot_swap with prefix_state_slots > 0: the hot swap leaves prefix states computed under \
             the previous levers; serve the hot swap cache-off"
        )));
    }
    if let Some(ab) = &n.ab {
        validate_ab(p, n, ab)?;
    }
    if let Some(lora) = &n.lora {
        validate_lora(p, n, lora)?;
    }
    if let Some(research) = &n.research {
        validate_research(p, n, research)?;
    }
    if let Some(guard) = &n.loop_guard {
        validate_loop_guard(p, guard)?;
    }
    if let Some(limit) = n.executor_memory_limit_bytes {
        if limit % (1 << 20) != 0 || limit < crate::readiness::NATIVE_MEMORY_BUDGET_BYTES {
            return Err(ConfigError::new(format!(
                "{p}.native.executor_memory_limit_bytes must be a whole MiB of at least 80 GiB"
            )));
        }
        if limit > crate::readiness::NATIVE_MEMORY_CEILING_BYTES {
            return Err(ConfigError::new(format!(
                "{p}.native.executor_memory_limit_bytes exceeds 86,432 MiB (GPU0 total - desktop baseline - the \
                 10,240 MiB free floor, tasks/m23-spec.md P8)"
            )));
        }
    }
    match n.image_digest.strip_prefix("sha256:") {
        Some(hex) if is_sha256_hex(hex) => Ok(()),
        _ => Err(ConfigError::new(format!(
            "{p}.native.image_digest must be sha256:<64 lowercase hex>"
        ))),
    }
}

/// Track LoRA: `[deployments.native.lora]`, fail-closed. The single form only: an adapter lives in one executor's
/// pool, and a second card would serve the name without it.
fn validate_lora(p: &str, n: &NativeConfig, lora: &LoraConfig) -> Result<(), ConfigError> {
    if n.replicas.is_some() {
        return Err(ConfigError::new(format!(
            "{p}.native.lora is refused with {p}.native.replicas"
        )));
    }
    range(
        &format!("{p}.native.lora.max_adapters"),
        u64::from(lora.max_adapters),
        1,
        u64::from(crate::native::ffi::LORA_MAX_ADAPTERS),
    )?;
    range(
        &format!("{p}.native.lora.max_rank"),
        u64::from(lora.max_rank),
        1,
        u64::from(crate::native::ffi::LORA_MAX_RANK),
    )?;
    let path = |c: u8| is_alnum(c) || b"._/-".contains(&c);
    pattern(
        &format!("{p}.native.lora.adapter_root"),
        &lora.adapter_root,
        1024,
        |c| c == b'/',
        path,
    )?;
    items(
        &format!("{p}.native.lora.preload"),
        lora.preload.len(),
        0,
        lora.max_adapters as usize,
    )?;
    let mut names = HashSet::new();
    for adapter in &lora.preload {
        id(&format!("{p}.native.lora.preload.name"), &adapter.name)?;
        pattern(
            &format!("{p}.native.lora.preload.path"),
            &adapter.path,
            1024,
            |c| c == b'/',
            path,
        )?;
        if !names.insert(adapter.name.as_str()) {
            return Err(ConfigError::new(format!(
                "duplicate {p}.native.lora.preload name {}",
                adapter.name
            )));
        }
    }
    Ok(())
}

/// WeightLab WL-01: `[deployments.native.research]`, fail-closed. The single form only: plans and captures live in
/// one executor (as LoRA's pool).
fn validate_research(p: &str, n: &NativeConfig, r: &ResearchConfig) -> Result<(), ConfigError> {
    if n.replicas.is_some() {
        return Err(ConfigError::new(format!(
            "{p}.native.research is refused with {p}.native.replicas"
        )));
    }
    let f = |key: &str| format!("{p}.native.research.{key}");
    range(
        &f("device_staging_bytes"),
        r.device_staging_bytes,
        1 << 20,
        1 << 30,
    )?;
    range(
        &f("host_buffer_bytes"),
        r.host_buffer_bytes,
        1 << 20,
        16 << 30,
    )?;
    range(
        &f("max_capture_bytes_per_request"),
        r.max_capture_bytes_per_request,
        4096,
        r.host_buffer_bytes,
    )?;
    range(
        &f("max_plans"),
        u64::from(r.max_plans),
        1,
        u64::from(crate::native::ffi::RESEARCH_MAX_PLANS),
    )?;
    range(&f("max_vectors"), u64::from(r.max_vectors), 1, 1024)
}

/// loop-guard design note LG-03: `[deployments.native.loop_guard]`, fail-closed (the schema's bounds plus the
/// cross-key rules `scripts/cere_contracts.py` mirrors): `min_period_bytes` and `max_period_bytes` 1..=4096,
/// `min_repeats` 0 or 2..=64, `min_span_bytes` at most [`LOOP_GUARD_MAX_WINDOW_BYTES`], `tool_repeat_run` 0 or
/// 2..=10,000, at most 16 `ignore_arg_keys` of 1..=64 characters; at least one detector on; with LG-D1 on,
/// `min_period_bytes <= max_period_bytes` and `min_repeats x max_period_bytes` at most the window bound.
fn validate_loop_guard(p: &str, g: &LoopGuardConfig) -> Result<(), ConfigError> {
    let field = format!("{p}.native.loop_guard");
    let zero_or = |key: &str, value: u32, min: u64, max: u64| {
        if value == 0 {
            Ok(())
        } else {
            range(&format!("{field}.{key}"), value.into(), min, max)
                .map_err(|_| ConfigError::new(format!("{field}.{key} must be 0 or {min}..{max}")))
        }
    };
    range(
        &format!("{field}.min_period_bytes"),
        g.min_period_bytes.into(),
        1,
        4096,
    )?;
    range(
        &format!("{field}.max_period_bytes"),
        g.max_period_bytes.into(),
        1,
        4096,
    )?;
    zero_or("min_repeats", g.min_repeats, 2, 64)?;
    range(
        &format!("{field}.min_span_bytes"),
        g.min_span_bytes.into(),
        0,
        LOOP_GUARD_MAX_WINDOW_BYTES,
    )?;
    zero_or("tool_repeat_run", g.tool_repeat_run, 2, 10_000)?;
    if g.ignore_arg_keys.len() > 16 {
        return Err(ConfigError::new(format!(
            "{field}.ignore_arg_keys holds at most 16 keys"
        )));
    }
    for key in &g.ignore_arg_keys {
        range(
            &format!("{field}.ignore_arg_keys[]"),
            key.chars().count() as u64,
            1,
            64,
        )?;
    }
    if g.min_repeats == 0 && g.tool_repeat_run == 0 {
        return Err(ConfigError::new(format!(
            "{field} turns both detectors off (min_repeats = 0, tool_repeat_run = 0); remove the table instead"
        )));
    }
    if g.min_repeats > 0 {
        if g.min_period_bytes > g.max_period_bytes {
            return Err(ConfigError::new(format!(
                "{field}.min_period_bytes must be at most max_period_bytes"
            )));
        }
        if g.window_bytes() > LOOP_GUARD_MAX_WINDOW_BYTES {
            return Err(ConfigError::new(format!(
                "{field}: min_repeats x max_period_bytes must be at most {LOOP_GUARD_MAX_WINDOW_BYTES}"
            )));
        }
    }
    Ok(())
}

/// ab-split design note AB-01/AB-02: `[deployments.native.ab]`, fail-closed: identifier salt and arm ids, 1..=16
/// arms with unique ids, weights 0..=1_000_000 summing above 0, and each arm's levers registered runtime-layer
/// levers at bit-identical implementations the model's drafts can run with the profile's levers.
fn validate_ab(p: &str, n: &NativeConfig, ab: &AbConfig) -> Result<(), ConfigError> {
    let field = format!("{p}.native.ab");
    id(&format!("{field}.salt"), &ab.salt)?;
    items(&format!("{field}.arms"), ab.arms.len(), 1, AB_MAX_ARMS)?;
    let mut seen = HashSet::new();
    let mut total = 0u64;
    for (i, arm) in ab.arms.iter().enumerate() {
        let at = format!("{field}.arms[{i}]");
        id(&format!("{at}.id"), &arm.id)?;
        if !seen.insert(arm.id.as_str()) {
            return Err(ConfigError::new(format!(
                "{at}.id {:?} is a duplicate arm id",
                arm.id
            )));
        }
        range(
            &format!("{at}.weight"),
            u64::from(arm.weight),
            0,
            AB_MAX_WEIGHT,
        )?;
        total += u64::from(arm.weight);
        crate::ab::ArmLevers::from_table(&arm.levers, &n.levers, n.speculative_draft_tokens)
            .map_err(|e| ConfigError::new(format!("{at}.levers.{e}")))?;
    }
    if total == 0 {
        return Err(ConfigError::new(format!(
            "{field}.arms weights must sum above 0"
        )));
    }
    Ok(())
}

/// `hip:<ordinal>` (decimal, no leading zero, at most 3 digits): an AMD GPU bound by `HIP_VISIBLE_DEVICES`
/// (NativeConfig::device_uuid). The ordinal as written, or `None` for any other string.
pub fn hip_ordinal(s: &str) -> Option<&str> {
    let ord = s.strip_prefix("hip:")?;
    let ok = !ord.is_empty()
        && ord.len() <= 3
        && ord.bytes().all(|c| c.is_ascii_digit())
        && (ord == "0" || !ord.starts_with('0'));
    ok.then_some(ord)
}

/// `vk:<ordinal>` (decimal, no leading zero, at most 3 digits): a Vulkan physical device (the Vulkan executor,
/// NativeConfig::device_uuid); the ordinal is the executor's device id. The ordinal as written, or `None` for any
/// other string.
pub fn vk_ordinal(s: &str) -> Option<&str> {
    let ord = s.strip_prefix("vk:")?;
    let ok = !ord.is_empty()
        && ord.len() <= 3
        && ord.bytes().all(|c| c.is_ascii_digit())
        && (ord == "0" || !ord.starts_with('0'));
    ok.then_some(ord)
}

/// `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` with lowercase hex digits (the `nvidia-smi` form).
fn is_gpu_uuid(s: &str) -> bool {
    let Some(uuid) = s.strip_prefix("GPU-") else {
        return false;
    };
    let groups: Vec<&str> = uuid.split('-').collect();
    groups.len() == 5
        && groups.iter().zip([8, 4, 4, 4, 12]).all(|(g, len)| {
            g.len() == len
                && g.bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        })
}

fn load_deployment(repo_root: &Path, prefix: &str) -> Result<Deployment, ConfigError> {
    let read = |part: &str| {
        let path = repo_root.join(format!("{prefix}.{part}.json"));
        std::fs::read(&path).map_err(|e| {
            ConfigError::new(format!(
                "deployment file {prefix}.{part}.json unreadable: {e}"
            ))
        })
    };
    fn parse<T: serde::de::DeserializeOwned>(part: &str, bytes: &[u8]) -> Result<T, ConfigError> {
        serde_json::from_slice(bytes)
            .map_err(|e| ConfigError::new(format!("deployment {part} invalid: {e}")))
    }
    let (recipe_bytes, lock_bytes, profile_bytes) =
        (read("recipe")?, read("lock")?, read("capabilities")?);
    let recipe: ModelRecipe = parse("recipe", &recipe_bytes)?;
    let lock: DeploymentLock = parse("lock", &lock_bytes)?;
    let profile: ExecutionCapabilities = parse("capabilities", &profile_bytes)?;
    check_recipe_schema(&recipe)?;
    check_lock_schema(&lock)?;
    check_capabilities_schema(&profile)?;
    profile
        .validate()
        .map_err(|e| ConfigError::new(format!("deployment capability profile: {e}")))?;
    if lock.recipe_sha256 != sha256_hex(&recipe_bytes) {
        return Err(ConfigError::new(
            "deployment lock does not pin the recipe bytes",
        ));
    }
    if recipe.backend_profile_id != lock.backend_profile_id
        || lock.backend_profile_id != profile.profile_id
    {
        return Err(ConfigError::new(
            "deployment recipe, lock and capability profile ids differ",
        ));
    }
    if profile.artifact_sha256 != lock.artifact_sha256 {
        return Err(ConfigError::new(
            "capability profile artifact differs from the lock",
        ));
    }
    if recipe.context_tokens != lock.effective_context_tokens
        || lock.effective_context_tokens != profile.max_context_tokens
    {
        return Err(ConfigError::new("deployment context lengths differ"));
    }
    if !recipe
        .required_operations
        .iter()
        .all(|op| lock.effective_operations.contains(op))
    {
        return Err(ConfigError::new(
            "lock does not provide every operation the recipe requires",
        ));
    }
    Ok(Deployment {
        recipe,
        lock,
        lock_sha256: sha256_hex(&lock_bytes),
        profile,
    })
}

/// Recipe/lock operation enum shared by `schemas/model_recipe.schema.json` and `deployment_lock`.
const DEPLOYMENT_OPERATIONS: &[&str] = &["chat", "infill", "embed", "rerank", "classify", "score"];
/// `schemas/execution_capabilities.schema.json` operation enum.
const CAPABILITY_OPERATIONS: &[&str] = &[
    "prefill", "decode", "verify", "snapshot", "restore", "cancel", "infill", "embed", "rerank",
    "classify", "score",
];
const MODALITIES: &[&str] = &["text", "image"];
/// M2.8 D10: `regex`, `ebnf` and `tool_calls_constrained` are grammar outputs beside `json_schema`.
const OUTPUTS: &[&str] = &[
    "text",
    "json_schema",
    "tool_calls",
    "regex",
    "ebnf",
    "tool_calls_constrained",
];
const NUMERICAL_MODES: &[&str] = &["reference", "optimized", "approximate"];
/// M2.12 D2 `token_controls` items.
pub const TOKEN_CONTROLS: &[&str] = &["stop_token_ids", "ignore_eos", "min_tokens"];
/// JSON-schema `maxItems` of every unbounded deployment array.
const MAX_ITEMS: usize = 10_000;

/// The constraints of `schemas/model_recipe.schema.json` beyond serde's shape and unknown-key checks.
fn check_recipe_schema(r: &ModelRecipe) -> Result<(), ConfigError> {
    if r.schema_version != "1" {
        return Err(ConfigError::new("unsupported recipe schema_version"));
    }
    id("recipe.recipe_id", &r.recipe_id)?;
    let reference = r
        .model_ref
        .strip_prefix("local://")
        .or_else(|| r.model_ref.strip_prefix("registry://"))
        .unwrap_or("");
    pattern("recipe.model_ref", reference, 1024, is_alnum, |c| {
        is_alnum(c) || b"._/@:-".contains(&c)
    })?;
    id("recipe.backend_profile_id", &r.backend_profile_id)?;
    range("recipe.context_tokens", r.context_tokens, 1, u64::MAX)?;
    enum_list(
        "recipe.required_operations",
        &r.required_operations,
        DEPLOYMENT_OPERATIONS,
    )?;
    items("recipe.adapters", r.adapters.len(), 0, MAX_ITEMS)?;
    for (i, a) in r.adapters.iter().enumerate() {
        chars("recipe.adapters.id", &a.id, 1, 256)?;
        if r.adapters[..i]
            .iter()
            .any(|b| b.id == a.id && b.scale == a.scale)
        {
            return Err(ConfigError::new("recipe.adapters items must be unique"));
        }
    }
    range("recipe.residency.priority", r.residency.priority, 0, 1000)?;
    range(
        "recipe.residency.keep_warm_seconds",
        r.residency.keep_warm_seconds,
        0,
        604_800,
    )?;
    let s = &r.sampling_defaults;
    if !(0.0..=10.0).contains(&s.temperature) {
        return Err(ConfigError::new(
            "recipe.sampling_defaults.temperature must be 0..10",
        ));
    }
    if !(s.top_p > 0.0 && s.top_p <= 1.0) {
        return Err(ConfigError::new(
            "recipe.sampling_defaults.top_p must be in (0, 1]",
        ));
    }
    if let Some(n) = s.max_new_tokens {
        range("recipe.sampling_defaults.max_new_tokens", n, 1, 131_072)?;
    }
    Ok(())
}

/// The constraints of `schemas/deployment_lock.schema.json` beyond serde's shape and unknown-key checks.
fn check_lock_schema(l: &DeploymentLock) -> Result<(), ConfigError> {
    if l.schema_version != "1" {
        return Err(ConfigError::new("unsupported lock schema_version"));
    }
    id("lock.lock_id", &l.lock_id)?;
    for (field, hash) in [
        ("lock.recipe_sha256", &l.recipe_sha256),
        ("lock.artifact_sha256", &l.artifact_sha256),
        ("lock.model_content_sha256", &l.model_content_sha256),
        ("lock.tokenizer_sha256", &l.tokenizer_sha256),
        ("lock.template_sha256", &l.template_sha256),
        ("lock.backend_build_sha256", &l.backend_build_sha256),
    ] {
        sha256_field(field, hash)?;
    }
    id("lock.backend_profile_id", &l.backend_profile_id)?;
    range(
        "lock.effective_context_tokens",
        l.effective_context_tokens,
        1,
        u64::MAX,
    )?;
    enum_list(
        "lock.effective_operations",
        &l.effective_operations,
        DEPLOYMENT_OPERATIONS,
    )?;
    unique("lock.adapter_sha256s", &l.adapter_sha256s, MAX_ITEMS)?;
    for hash in &l.adapter_sha256s {
        sha256_field("lock.adapter_sha256s", hash)?;
    }
    Ok(())
}

/// The constraints of `schemas/execution_capabilities.schema.json` that `ExecutionCapabilities::validate`
/// assumes already hold (it runs after schema validation).
fn check_capabilities_schema(p: &ExecutionCapabilities) -> Result<(), ConfigError> {
    id("capabilities.plugin", &p.plugin)?;
    id("capabilities.profile_id", &p.profile_id)?;
    sha256_field("capabilities.artifact_sha256", &p.artifact_sha256)?;
    one_of(
        "capabilities.execution_mode",
        &p.execution_mode,
        &["native", "backend_managed", "serial", "unsupported"],
    )?;
    one_of(
        "capabilities.evidence",
        &p.evidence,
        &["unknown", "declared", "measured", "synthetic"],
    )?;
    enum_list(
        "capabilities.operations",
        &p.operations,
        CAPABILITY_OPERATIONS,
    )?;
    range(
        "capabilities.decode_batch_min",
        p.decode_batch_min,
        1,
        u64::MAX,
    )?;
    range(
        "capabilities.decode_batch_max",
        p.decode_batch_max,
        1,
        u64::MAX,
    )?;
    range(
        "capabilities.max_context_tokens",
        p.max_context_tokens,
        1,
        u64::MAX,
    )?;
    enum_list(
        "capabilities.structured_outputs",
        &p.structured_outputs,
        OUTPUTS,
    )?;
    enum_list(
        "capabilities.numerical_modes",
        &p.numerical_modes,
        NUMERICAL_MODES,
    )?;
    one_of(
        "capabilities.checkpoint_mode",
        &p.checkpoint_mode,
        &["none", "exact", "replay"],
    )?;
    enum_list("capabilities.modalities", &p.modalities, MODALITIES)?;
    if let Some(top) = p.max_top_logprobs {
        range("capabilities.max_top_logprobs", u64::from(top), 0, 20)?;
    }
    if let Some(names) = &p.sampling_params {
        enum_list("capabilities.sampling_params", names, &SAMPLING_PARAMS)?;
    }
    if let Some(controls) = &p.token_controls {
        unique(
            "capabilities.token_controls",
            controls,
            TOKEN_CONTROLS.len(),
        )?;
        controls
            .iter()
            .try_for_each(|c| one_of("capabilities.token_controls", c, TOKEN_CONTROLS))?;
    }
    if let Some(vocab) = p.vocab_size {
        range(
            "capabilities.vocab_size",
            u64::from(vocab),
            1,
            u64::from(u32::MAX),
        )?;
    }
    if let Some(signature) = &p.prefill_signature {
        sha256_field("capabilities.prefill_signature", signature)?;
    }
    chars("capabilities.notes", &p.notes, 0, 4096)?;
    if let Some(dialect) = &p.json_schema_dialect {
        chars("capabilities.json_schema_dialect", dialect, 0, 128)?;
    }
    unique(
        "capabilities.json_schema_keywords",
        &p.json_schema_keywords,
        MAX_ITEMS,
    )?;
    for keyword in &p.json_schema_keywords {
        chars("capabilities.json_schema_keywords", keyword, 1, 128)?;
    }
    items(
        "capabilities.validated_combinations",
        p.validated_combinations.len(),
        0,
        256,
    )?;
    for c in &p.validated_combinations {
        let field = "capabilities.validated_combinations";
        one_of(field, &c.operation, CAPABILITY_OPERATIONS)?;
        one_of(field, &c.modality, MODALITIES)?;
        one_of(field, &c.output, OUTPUTS)?;
        one_of(field, &c.numerical_mode, NUMERICAL_MODES)?;
        one_of(
            field,
            &c.sampler_mode,
            &["greedy", "sampled", "not_applicable"],
        )?;
        items(field, c.row_counts.len(), 1, 32)?;
        for (i, &rows) in c.row_counts.iter().enumerate() {
            range(
                "capabilities.validated_combinations.row_counts",
                rows,
                1,
                65_536,
            )?;
            if c.row_counts[..i].contains(&rows) {
                return Err(ConfigError::new(
                    "capabilities.validated_combinations.row_counts items must be unique",
                ));
            }
        }
        range(field, c.max_context_tokens, 1, u64::MAX)?;
    }
    Ok(())
}

fn one_of(field: &str, value: &str, allowed: &[&str]) -> Result<(), ConfigError> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(ConfigError::new(format!(
            "{field} must be one of {allowed:?}"
        )))
    }
}

/// `uniqueItems` string array whose items come from `allowed`.
fn enum_list(field: &str, values: &[String], allowed: &[&str]) -> Result<(), ConfigError> {
    unique(field, values, MAX_ITEMS)?;
    values.iter().try_for_each(|v| one_of(field, v, allowed))
}

fn unique(field: &str, values: &[String], max_items: usize) -> Result<(), ConfigError> {
    items(field, values.len(), 0, max_items)?;
    let mut seen = HashSet::with_capacity(values.len());
    if values.iter().all(|v| seen.insert(v)) {
        Ok(())
    } else {
        Err(ConfigError::new(format!("{field} items must be unique")))
    }
}

fn items(field: &str, len: usize, min: usize, max: usize) -> Result<(), ConfigError> {
    if (min..=max).contains(&len) {
        Ok(())
    } else {
        Err(ConfigError::new(format!("{field} item count out of range")))
    }
}

/// JSON-schema `minLength`/`maxLength`, which count code points.
pub(crate) fn chars(field: &str, value: &str, min: usize, max: usize) -> Result<(), ConfigError> {
    if (min..=max).contains(&value.chars().count()) {
        Ok(())
    } else {
        Err(ConfigError::new(format!(
            "{field} length must be {min}..{max}"
        )))
    }
}

fn sha256_field(field: &str, value: &str) -> Result<(), ConfigError> {
    if is_sha256_hex(value) {
        Ok(())
    } else {
        Err(ConfigError::new(format!(
            "{field} must be 64 lowercase hex"
        )))
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
}

fn is_alnum(c: u8) -> bool {
    c.is_ascii_alphanumeric()
}

fn is_lower_alnum(c: u8) -> bool {
    c.is_ascii_lowercase() || c.is_ascii_digit()
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

/// `^[first][rest]{0,max_len-1}$` over ASCII bytes (the end anchor rejects a trailing newline too).
fn pattern(
    field: &str,
    value: &str,
    max_len: usize,
    first: impl Fn(u8) -> bool,
    rest: impl Fn(u8) -> bool,
) -> Result<(), ConfigError> {
    let bytes = value.as_bytes();
    match bytes.split_first() {
        Some((&head, tail))
            if bytes.len() <= max_len && first(head) && tail.iter().all(|&c| rest(c)) =>
        {
            Ok(())
        }
        _ => Err(ConfigError::new(format!(
            "{field} does not match its pattern"
        ))),
    }
}

/// The schema's identifier pattern `^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$`.
pub(crate) fn id(field: &str, value: &str) -> Result<(), ConfigError> {
    pattern(field, value, 128, is_alnum, |c| {
        is_alnum(c) || b"_.:-".contains(&c)
    })
}

fn id_list(field: &str, values: &[String], min_items: usize) -> Result<(), ConfigError> {
    if values.len() < min_items || values.len() > 10_000 {
        return Err(ConfigError::new(format!("{field} item count out of range")));
    }
    let mut seen = HashSet::new();
    for value in values {
        id(field, value)?;
        if !seen.insert(value) {
            return Err(ConfigError::new(format!("{field} items must be unique")));
        }
    }
    Ok(())
}

pub(crate) fn range(field: &str, value: u64, min: u64, max: u64) -> Result<(), ConfigError> {
    if (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(ConfigError::new(format!("{field} must be {min}..{max}")))
    }
}
