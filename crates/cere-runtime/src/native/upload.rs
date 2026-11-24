//! The upload sink (spec m22 R2, Q13): the model desc from an opened M2.1 [`Artifact`], then ALLOCATE
//! (`cere_model_create`), UPLOAD (one `Artifact::stage(All)` pass whose staging threads call `cere_model_put`
//! directly, m216 design note L2) and INSTANTIATE (`cere_model_seal`), with byte counters and stage timings.
//!
//! Placement comes from the artifact's families: `device` -> DEVICE, `host_pinned` -> HOST_PINNED, the MTP draft
//! family (role `draft`) -> DISCARD (counted, never read nor stored: m216 L1) unless the plan speculates
//! (`speculative_draft_tokens` 4, m29 design note D11: then DEVICE, read and stored, with the artifact's draft token
//! map in the desc; a window of 2..8 on family inkling2_loop, inkling design note D-I-S9B: DEVICE, with no map,
//! the drafter being full-vocab); `unused` tensors are not part of the desc and are not read either. The A4 config
//! struct is read from the artifact's `config.json` `text_config`; the stop tokens are the `eos_token_id` values of
//! `generation_config.json` and `text_config`. Both files are read by path and authenticated against the manifest's
//! sha256 of the opened artifact.
use super::driver::{
    thread_nice, AllocateThread, Background, Driver, DriverError, Model, ModelConfig, ModelSpec,
    Putter, TensorSpec,
};
use super::ffi::{
    self, ExecutionCapabilities, Inkling2Config, Qwen4ExpConfig, Qwen4ExpVisionConfig, SdarConfig,
};
use super::sampling::GenerationDefaults;
use cere_compiler::artifact::{
    self, Artifact, Class, ReadMode, Second, Sink, StageOptions, StageReport, TensorRef,
};
use cere_compiler::families::Role;
use cere_compiler::safetensors::Dtype;
use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Staging of the native upload (m216 design note L2, L7): one shared queue of buffer-sized reads across files,
/// [`DEFAULT_READERS`] reads in flight from [`DEFAULT_BUFFERS`] × 16 MiB registered buffers (768 MiB), the
/// buffers that are not being read held by the [`DEFAULT_CONSUMERS`] put threads of each model (the disk benches,
/// F1-F3 and L7: 32-64 outstanding reads of 2-16 MiB from one queue reach the disk's ceiling).
pub const DEFAULT_BUFFERS: usize = 48;
pub const DEFAULT_BUFFER_BYTES: usize = 16 << 20;
pub const DEFAULT_READERS: usize = 32;
pub const DEFAULT_CONSUMERS: usize = 16;

/// m32n design note D22 (3): a background load (a replica loading while the other card serves) stages through
/// this many buffers, the cap on copies in flight for the reloading card, with as many reader and put threads.
pub const BACKGROUND_COPIES_IN_FLIGHT: usize = 2;
/// D22 (3): the nice value of every thread of a background load's pass (readers, put threads, resident checks), the
/// lowest CPU priority, so the serving card's driver thread and its launches are never starved by the reload.
pub const BACKGROUND_NICE: i32 = 19;
/// D22 (3): the CPUs a background load's ALLOCATE runs on (the last ones the caller may use), and with it the
/// executor's threads started there: the host pin and an attached segment's digest check (~32 threads), which run on
/// into UPLOAD. At nice 0 on every CPU they took the serving card's decode down 25-45 % for 9-22 s after each
/// reload's ALLOCATE (M3.2N G-6 attempt 2, g6probe2).
pub const BACKGROUND_ALLOCATE_CPUS: usize = 2;

/// The model-desc limits that do not come from the artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanOptions {
    pub max_sessions: u64,
    pub max_context_tokens: u64,
    /// `cere_model_desc.snapshot_host_bytes` (M2.11 D5); 0 = no snapshot arena.
    pub snapshot_host_bytes: u64,
    /// `cere_model_desc.kv_pool_tokens` (m25 design note D4): 0 = the derived pool (max_sessions x
    /// round64(max_context_tokens)); the executor validates a non-zero pool at `cere_model_create`.
    pub kv_pool_tokens: u64,
    /// `cere_model_desc.prefix_state_slots` (M2.10 D6); 0 = prefix cache off.
    pub prefix_state_slots: u64,
    /// `cere_model_desc.speculative_draft_tokens` (m29 design note I1, D11): 0 = no draft (the MTP tensors are
    /// DISCARD), 4 = MTP speculation (DEVICE, plus the artifact's token map), 2..8 on Inkling = its MTP drafter's
    /// verify window (inkling design note D-I-S9B: DEVICE, no token map).
    pub speculative_draft_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UploadOptions {
    pub buffers: usize,
    pub buffer_bytes: usize,
    /// Reader threads (each one read in flight) and put threads per model; 0 = the engine's default.
    pub readers: usize,
    pub consumers: usize,
    /// load-speed design note LS-05: the default is [`ReadMode::Auto`] (a file less than half resident is read
    /// with `O_DIRECT`: cold loads reach the disk's ceiling at a fraction of the CPU; a warm one stays buffered).
    pub read: ReadMode,
    /// D22 (3): run the pass's threads at [`BACKGROUND_NICE`]. Linux keeps a nice value per thread and a new thread
    /// starts with its creator's, so the pass runs on a thread of its own that lowers itself first (the caller's
    /// priority never changes; an unprivileged thread could not raise it back).
    pub background: bool,
}

impl Default for UploadOptions {
    fn default() -> Self {
        UploadOptions {
            buffers: DEFAULT_BUFFERS,
            buffer_bytes: DEFAULT_BUFFER_BYTES,
            readers: DEFAULT_READERS,
            consumers: DEFAULT_CONSUMERS,
            read: ReadMode::Auto,
            background: false,
        }
    }
}

impl UploadOptions {
    /// D22 (3): a load while the other card serves: [`BACKGROUND_COPIES_IN_FLIGHT`] buffers, readers and put
    /// threads, at background priority.
    pub fn background() -> Self {
        UploadOptions {
            buffers: BACKGROUND_COPIES_IN_FLIGHT,
            readers: BACKGROUND_COPIES_IN_FLIGHT,
            consumers: BACKGROUND_COPIES_IN_FLIGHT,
            background: true,
            ..UploadOptions::default()
        }
    }
}

/// Why an upload failed.
#[derive(Debug)]
pub enum UploadError {
    /// The desc cannot be built from the artifact (class `config`, `dtype`, `tensor_table`, `ple_table`,
    /// `file_hash`, `io`).
    Plan {
        class: &'static str,
        message: String,
    },
    /// A host call on the driver thread refused (create, seal, capabilities).
    Driver(DriverError),
    /// Staging failed; a refused put is class `sink`.
    Stage(cere_compiler::Error),
}

impl UploadError {
    fn plan(class: &'static str, message: impl Into<String>) -> UploadError {
        UploadError::Plan {
            class,
            message: message.into(),
        }
    }

    /// A short error class for reports.
    pub fn class(&self) -> &'static str {
        match self {
            UploadError::Plan { class, .. } => class,
            UploadError::Driver(DriverError::Status { status, .. }) => status.name(),
            UploadError::Driver(DriverError::Argument(_)) => "argument",
            UploadError::Driver(DriverError::Missing(_)) => "missing_symbol",
            UploadError::Driver(DriverError::Stopped) => "driver_stopped",
            UploadError::Stage(e) => e.class,
        }
    }
}

impl fmt::Display for UploadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UploadError::Plan { class, message } => write!(f, "{class}: {message}"),
            UploadError::Driver(e) => write!(f, "{e}"),
            UploadError::Stage(e) => write!(f, "{}: {}", e.class, e.message),
        }
    }
}

impl std::error::Error for UploadError {}

impl From<DriverError> for UploadError {
    fn from(e: DriverError) -> Self {
        UploadError::Driver(e)
    }
}

/// `CERE_DTYPE_*` of a safetensors dtype; `None` for dtypes ABI v3 cannot express (U16, U32, U64).
pub fn abi_dtype(d: Dtype) -> Option<u32> {
    Some(match d {
        Dtype::Bf16 => ffi::DTYPE_BF16,
        Dtype::F16 => ffi::DTYPE_F16,
        Dtype::F32 => ffi::DTYPE_F32,
        Dtype::F64 => ffi::DTYPE_F64,
        Dtype::F8E4m3 => ffi::DTYPE_F8_E4M3,
        Dtype::F8E5m2 => ffi::DTYPE_F8_E5M2,
        Dtype::U8 => ffi::DTYPE_U8,
        Dtype::I8 => ffi::DTYPE_I8,
        Dtype::I16 => ffi::DTYPE_I16,
        Dtype::I32 => ffi::DTYPE_I32,
        Dtype::I64 => ffi::DTYPE_I64,
        Dtype::Bool => ffi::DTYPE_BOOL,
        Dtype::U16 | Dtype::U32 | Dtype::U64 => return None,
    })
}

/// `CERE_PLACEMENT_*` of a family placement (`device`, `host_pinned`, `unused`) and role under a plan's
/// `speculative_draft_tokens`; `None` for tensors outside the desc (`unused`). The one placement decision of the
/// native upload (m29 design note M29S2-15): only the draft role depends on the plan.
pub fn placement(placement: &str, role: Role, speculative_draft_tokens: u32) -> Option<u32> {
    match placement {
        "device" | "host_pinned" if role == Role::Draft => Some(if speculative_draft_tokens > 0 {
            ffi::PLACEMENT_DEVICE
        } else {
            ffi::PLACEMENT_DISCARD
        }),
        "device" => Some(ffi::PLACEMENT_DEVICE),
        "host_pinned" => Some(ffi::PLACEMENT_HOST_PINNED),
        _ => None,
    }
}

/// [`placement`] of a staged tensor.
pub fn abi_placement(t: &TensorRef<'_>, speculative_draft_tokens: u32) -> Option<u32> {
    placement(t.placement, t.role, speculative_draft_tokens)
}

/// The bytes of the artifact file with manifest role `role`, read from `dir` and checked against the manifest
/// sha256; `None` when the manifest lists no such file.
fn read_role(art: &Artifact, dir: &Path, role: &str) -> Result<Option<Vec<u8>>, UploadError> {
    let Some(f) = art.files.iter().find(|f| f.role == role) else {
        return Ok(None);
    };
    let raw = std::fs::read(dir.join(&f.path))
        .map_err(|e| UploadError::plan("io", format!("{}: {e}", f.path)))?;
    if cere_compiler::sha256_hex(&raw) != f.sha256 {
        return Err(UploadError::plan(
            "file_hash",
            format!("{}: sha256 differs from the manifest", f.path),
        ));
    }
    Ok(Some(raw))
}

fn parse_json(raw: &[u8], what: &str) -> Result<Value, UploadError> {
    serde_json::from_slice(raw).map_err(|e| UploadError::plan("config", format!("{what}: {e}")))
}

/// Typed reads of `text_config` keys (absent → the default).
struct Keys<'a>(&'a Map<String, Value>, &'a str);

impl Keys<'_> {
    fn err(&self, key: &str, want: &str) -> UploadError {
        UploadError::plan("config", format!("{}.{key} is not {want}", self.1))
    }

    fn u64(&self, key: &str) -> Result<u64, UploadError> {
        match self.0.get(key) {
            None => Ok(0),
            Some(v) => v
                .as_u64()
                .ok_or_else(|| self.err(key, "an unsigned integer")),
        }
    }

    fn u32(&self, key: &str) -> Result<u32, UploadError> {
        u32::try_from(self.u64(key)?).map_err(|_| self.err(key, "a 32-bit unsigned integer"))
    }

    fn f64(&self, key: &str) -> Result<f64, UploadError> {
        match self.0.get(key) {
            None => Ok(0.0),
            Some(v) => v.as_f64().ok_or_else(|| self.err(key, "a number")),
        }
    }

    fn flag(&self, key: &str, absent: u32) -> Result<u32, UploadError> {
        match self.0.get(key) {
            None => Ok(absent),
            Some(v) => v
                .as_bool()
                .map(u32::from)
                .ok_or_else(|| self.err(key, "a boolean")),
        }
    }

    fn str(&self, key: &str) -> Result<Option<&str>, UploadError> {
        match self.0.get(key) {
            None => Ok(None),
            Some(v) => v
                .as_str()
                .map(Some)
                .ok_or_else(|| self.err(key, "a string")),
        }
    }

    fn list(&self, key: &str) -> Result<&[Value], UploadError> {
        match self.0.get(key) {
            None => Ok(&[]),
            Some(v) => v
                .as_array()
                .map(Vec::as_slice)
                .ok_or_else(|| self.err(key, "a list")),
        }
    }
}

/// The `cere_qwen4_exp_config` of a `config.json` (header rules: absent key 0, `norm_topk_prob` absent 1, lists as
/// bitmasks, strings as enums, rope values from `rope_parameters`).
pub fn qwen4_exp_config(config: &Value) -> Result<Qwen4ExpConfig, UploadError> {
    let text = config
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or_else(|| UploadError::plan("config", "config.json has no text_config object"))?;
    let k = Keys(text, "text_config");
    let empty = Map::new();
    let rope = match text.get("rope_parameters") {
        None => &empty,
        Some(v) => v
            .as_object()
            .ok_or_else(|| k.err("rope_parameters", "an object"))?,
    };
    let r = Keys(rope, "text_config.rope_parameters");

    let mut full_attention_layers = 0u64;
    let layer_types = k.list("layer_types")?;
    if layer_types.len() > 64 {
        return Err(k.err("layer_types", "at most 64 entries"));
    }
    for (i, t) in layer_types.iter().enumerate() {
        match t.as_str() {
            Some("full_attention") => full_attention_layers |= 1 << i,
            Some(_) => {}
            None => return Err(k.err("layer_types", "a list of strings")),
        }
    }
    let mut ple_layer_ids = 0u64;
    for v in k.list("ple_layer_ids")? {
        match v.as_u64() {
            Some(n) if n < 64 => ple_layer_ids |= 1 << n,
            _ => return Err(k.err("ple_layer_ids", "a list of integers below 64")),
        }
    }
    let mut mrope_section = [0u32; 3];
    let section = r.list("mrope_section")?;
    if !section.is_empty() {
        if section.len() != 3 {
            return Err(r.err("mrope_section", "a list of 3 integers"));
        }
        for (slot, v) in mrope_section.iter_mut().zip(section) {
            *slot = v
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| r.err("mrope_section", "a list of 3 integers"))?;
        }
    }
    let output_gate_type = match k.str("output_gate_type")? {
        None => 0,
        Some("sigmoid") => ffi::OUTPUT_GATE_SIGMOID,
        Some(other) => {
            return Err(UploadError::plan(
                "config",
                format!("text_config.output_gate_type {other:?} has no ABI value"),
            ))
        }
    };
    let mamba_ssm_dtype = match k.str("mamba_ssm_dtype")? {
        None => 0,
        Some("float32") => ffi::DTYPE_F32,
        Some("bfloat16") => ffi::DTYPE_BF16,
        Some("float16") => ffi::DTYPE_F16,
        Some(other) => {
            return Err(UploadError::plan(
                "config",
                format!("text_config.mamba_ssm_dtype {other:?} has no ABI value"),
            ))
        }
    };
    Ok(Qwen4ExpConfig {
        struct_size: ffi::struct_size::<Qwen4ExpConfig>(),
        abi_version: ffi::ABI_VERSION,
        vocab_size: k.u32("vocab_size")?,
        hidden_size: k.u32("hidden_size")?,
        num_hidden_layers: k.u32("num_hidden_layers")?,
        full_attention_interval: k.u32("full_attention_interval")?,
        full_attention_layers,
        num_attention_heads: k.u32("num_attention_heads")?,
        num_key_value_heads: k.u32("num_key_value_heads")?,
        head_dim: k.u32("head_dim")?,
        output_gate_type,
        rope_theta: r.f64("rope_theta")?,
        partial_rotary_factor: r.f64("partial_rotary_factor")?,
        mrope_interleaved: r.flag("mrope_interleaved", 0)?,
        mrope_section,
        indexer_n_heads: k.u32("indexer_n_heads")?,
        indexer_head_dim: k.u32("indexer_head_dim")?,
        indexer_kv_heads: k.u32("indexer_kv_heads")?,
        indexer_compress_ratio: k.u32("indexer_compress_ratio")?,
        indexer_budget: k.u32("indexer_budget")?,
        linear_num_key_heads: k.u32("linear_num_key_heads")?,
        linear_num_value_heads: k.u32("linear_num_value_heads")?,
        linear_key_head_dim: k.u32("linear_key_head_dim")?,
        linear_value_head_dim: k.u32("linear_value_head_dim")?,
        linear_conv_kernel_dim: k.u32("linear_conv_kernel_dim")?,
        mamba_ssm_dtype,
        num_experts: k.u32("num_experts")?,
        num_experts_per_tok: k.u32("num_experts_per_tok")?,
        moe_intermediate_size: k.u32("moe_intermediate_size")?,
        shared_expert_intermediate_size: k.u32("shared_expert_intermediate_size")?,
        norm_topk_prob: k.flag("norm_topk_prob", 1)?,
        hc_count: k.u32("hc_count")?,
        hc_lowrank: k.u32("hc_lowrank")?,
        ple_layer_ids,
        ple_embed_dim: k.u32("ple_embed_dim")?,
        heads_per_ngram: k.u32("heads_per_ngram")?,
        ngram_size: k.u32("ngram_size")?,
        ple_conv_kernel_size: k.u32("ple_conv_kernel_size")?,
        ngram_vocab_size_base: k.u64("ngram_vocab_size_base")?,
        split_ngram_parts: k.u32("split_ngram_parts")?,
        make_ngram_vocab_size_divisible_by: k.u32("make_ngram_vocab_size_divisible_by")?,
        rms_norm_eps: k.f64("rms_norm_eps")?,
        tie_word_embeddings: k.flag("tie_word_embeddings", 0)?,
        mtp_num_hidden_layers: k.u32("mtp_num_hidden_layers")?,
        max_position_embeddings: k.u64("max_position_embeddings")?,
    })
}

/// The `cere_sdar_config` of a flat `config.json` (SelfTune ST4, DICE-8B: `model_type` `sdar`, no `text_config`;
/// header rules: absent key 0). The library refuses any value that differs from its compiled descriptor.
pub fn sdar_config(config: &Value) -> Result<SdarConfig, UploadError> {
    let top = config
        .as_object()
        .ok_or_else(|| UploadError::plan("config", "config.json is not an object"))?;
    let k = Keys(top, "config");
    Ok(SdarConfig {
        struct_size: ffi::struct_size::<SdarConfig>(),
        abi_version: ffi::ABI_VERSION,
        vocab_size: k.u32("vocab_size")?,
        hidden_size: k.u32("hidden_size")?,
        intermediate_size: k.u32("intermediate_size")?,
        num_hidden_layers: k.u32("num_hidden_layers")?,
        num_attention_heads: k.u32("num_attention_heads")?,
        num_key_value_heads: k.u32("num_key_value_heads")?,
        head_dim: k.u32("head_dim")?,
        block_size: k.u32("block_size")?,
        rope_theta: k.f64("rope_theta")?,
        rms_norm_eps: k.f64("rms_norm_eps")?,
        mask_token_id: k.u32("mask_token_id")?,
        tie_word_embeddings: k.flag("tie_word_embeddings", 0)?,
        max_position_embeddings: k.u64("max_position_embeddings")?,
    })
}

/// The `cere_inkling2_config` of Inkling 2 Mini's `config.json` (inkling design note S0/S5: `model_type`
/// `inkling2_loop`, `text_config` keys, `loop_config` sub-object, top-level `eos_token_id`; header rules: absent key
/// 0) and its `hf_quant_config.json`: `mxfp8_layer_mask` bit i is set when the parameter group holding
/// `model.llm.layers.<i>.mlp.experts.w13_weight` has `compute_dtype` `mxfp8`. The library refuses any value that
/// differs from its compiled descriptor.
pub fn inkling2_config(config: &Value, quant: &Value) -> Result<Inkling2Config, UploadError> {
    let top = config
        .as_object()
        .ok_or_else(|| UploadError::plan("config", "config.json is not an object"))?;
    let text = config
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or_else(|| UploadError::plan("config", "config.json has no text_config object"))?;
    let k = Keys(text, "text_config");
    let empty = Map::new();
    let lp = match text.get("loop_config") {
        None => &empty,
        Some(v) => v
            .as_object()
            .ok_or_else(|| k.err("loop_config", "an object"))?,
    };
    let l = Keys(lp, "text_config.loop_config");
    let mut local_layer_mask = 0u32;
    for v in k.list("local_layer_ids")? {
        match v.as_u64() {
            Some(n) if n < 32 => local_layer_mask |= 1 << n,
            _ => return Err(k.err("local_layer_ids", "a list of integers below 32")),
        }
    }
    let mut mxfp8_layer_mask = 0u32;
    let groups = quant
        .pointer("/model/parameter_groups")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            UploadError::plan(
                "config",
                "hf_quant_config.json has no model.parameter_groups list",
            )
        })?;
    for g in groups {
        if g.get("compute_dtype").and_then(Value::as_str) != Some("mxfp8") {
            continue;
        }
        for name in g
            .get("tensors")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let layer = name
                .as_str()
                .and_then(|n| n.strip_prefix("model.llm.layers."))
                .and_then(|n| n.strip_suffix(".mlp.experts.w13_weight"))
                .and_then(|i| i.parse::<u32>().ok());
            match layer {
                Some(i) if i < 32 => mxfp8_layer_mask |= 1 << i,
                Some(_) => {
                    return Err(UploadError::plan(
                        "config",
                        "hf_quant_config.json names an expert layer above 31",
                    ))
                }
                None => {}
            }
        }
    }
    Ok(Inkling2Config {
        struct_size: ffi::struct_size::<Inkling2Config>(),
        abi_version: ffi::ABI_VERSION,
        vocab_size: k.u32("vocab_size")?,
        unpadded_vocab_size: k.u32("unpadded_vocab_size")?,
        hidden_size: k.u32("hidden_size")?,
        num_hidden_layers: k.u32("num_hidden_layers")?,
        num_attention_heads: k.u32("num_attention_heads")?,
        num_key_value_heads: k.u32("num_key_value_heads")?,
        head_dim: k.u32("head_dim")?,
        moe_intermediate_size: k.u32("intermediate_size")?,
        num_experts: k.u32("n_routed_experts")?,
        num_experts_per_tok: k.u32("num_experts_per_tok")?,
        sliding_window_size: k.u32("sliding_window_size")?,
        rel_extent: k.u32("rel_extent")?,
        d_rel: k.u32("d_rel")?,
        loop_core_start: l.u32("core_start")?,
        loop_core_end: l.u32("core_end")?,
        loop_num_passes: l.u32("num_passes")?,
        loop_gate_rank: l.u32("gate_rank")?,
        n_layers_per_attn_res_block: k.u32("n_layers_per_attn_res_block")?,
        conv_kernel_size: k.u32("sconv_kernel_size")?,
        eos_token_id: Keys(top, "config").u32("eos_token_id")?,
        local_layer_mask,
        mxfp8_layer_mask,
        rms_norm_eps: k.f64("rms_norm_eps")?,
        final_logit_softcapping: k.f64("final_logit_softcapping")?,
        route_scale: k.f64("route_scale")?,
        max_position_embeddings: k.u64("model_max_length")?,
    })
}

/// The model config of a `config.json` by its top-level `model_type` (SelfTune ST4): `sdar` is the flat DICE-8B
/// config, `inkling2_loop` Inkling 2 Mini's (which also reads `quant`, the artifact's `hf_quant_config.json`, and
/// is refused without it), anything else the Qwen `text_config` form (refused without a `text_config` object).
pub fn model_config(config: &Value, quant: Option<&Value>) -> Result<ModelConfig, UploadError> {
    match config.get("model_type") {
        Some(Value::String(t)) if t == "sdar" => Ok(ModelConfig::Sdar(sdar_config(config)?)),
        Some(Value::String(t)) if t == "inkling2_loop" => {
            let quant = quant.ok_or_else(|| {
                UploadError::plan(
                    "config",
                    "an inkling2_loop model needs hf_quant_config.json",
                )
            })?;
            Ok(ModelConfig::Inkling2(inkling2_config(config, quant)?))
        }
        Some(v) if !v.is_string() => Err(UploadError::plan(
            "config",
            "config.model_type is not a string",
        )),
        _ => Ok(ModelConfig::Qwen4Exp(qwen4_exp_config(config)?)),
    }
}

/// The `cere_qwen4_exp_vision_config` of a `config.json` (M2.13 ledger N): `vision_config` keys, the top-level
/// image/video/vision start/end token IDs, `deepstack_count` = `len(deepstack_visual_indexes)`, `hidden_act` as
/// its enum and `layer_norm_eps` SGLang's fixed 1e-6 (`qwen3_vl.py:196`); `None` without `vision_config`.
pub fn qwen4_exp_vision_config(
    config: &Value,
) -> Result<Option<Qwen4ExpVisionConfig>, UploadError> {
    let Some(vision) = config.get("vision_config") else {
        return Ok(None);
    };
    let vision = vision
        .as_object()
        .ok_or_else(|| UploadError::plan("config", "config.json vision_config is not an object"))?;
    let v = Keys(vision, "vision_config");
    let top = config
        .as_object()
        .ok_or_else(|| UploadError::plan("config", "config.json is not an object"))?;
    let t = Keys(top, "config");
    let hidden_act = match v.str("hidden_act")? {
        Some("gelu_pytorch_tanh") => ffi::ACT_GELU_TANH,
        _ => return Err(v.err("hidden_act", "\"gelu_pytorch_tanh\"")),
    };
    let deepstack_count = u32::try_from(v.list("deepstack_visual_indexes")?.len())
        .map_err(|_| v.err("deepstack_visual_indexes", "a short list"))?;
    Ok(Some(Qwen4ExpVisionConfig {
        struct_size: ffi::struct_size::<Qwen4ExpVisionConfig>(),
        abi_version: ffi::ABI_VERSION,
        depth: v.u32("depth")?,
        hidden_size: v.u32("hidden_size")?,
        intermediate_size: v.u32("intermediate_size")?,
        num_heads: v.u32("num_heads")?,
        patch_size: v.u32("patch_size")?,
        temporal_patch_size: v.u32("temporal_patch_size")?,
        spatial_merge_size: v.u32("spatial_merge_size")?,
        in_channels: v.u32("in_channels")?,
        out_hidden_size: v.u32("out_hidden_size")?,
        num_position_embeddings: v.u32("num_position_embeddings")?,
        deepstack_count,
        hidden_act,
        image_token_id: t.u32("image_token_id")?,
        video_token_id: t.u32("video_token_id")?,
        vision_start_token_id: t.u32("vision_start_token_id")?,
        vision_end_token_id: t.u32("vision_end_token_id")?,
        layer_norm_eps: 1e-6,
    }))
}

/// Inkling 2 Mini's image frame in its chat template: `<|content_image|>` opens an image's placeholders and
/// `<|end_message|>` closes them (inkling design note 3.7, S6g).
const INKLING_IMAGE_START: u32 = 200_005;
const INKLING_IMAGE_END: u32 = 200_010;
/// `Inkling2Config.image_token_id`'s default (`<|unused_200054|>`), used when `config.json` is silent.
const INKLING_IMAGE_TOKEN: u32 = 200_054;

/// The `cere_qwen4_exp_vision_config` of Inkling 2 Mini's `config.json` (inkling design note S6g; the ABI carries
/// one vision struct): its `vision_config` (`inkling2_vision`; `Inkling2VisionConfig`'s default where a key is
/// absent, `native_patch_size` / `adapter_merge_size` its aliases of `patch_size` / `spatial_merge_size`) as depth
/// `num_hidden_layers`, `intermediate_size` (absent: `hidden_size * mlp_ratio`, an integer), heads
/// `num_attention_heads`, `in_channels` `num_channels`, `out_hidden_size` `text_hidden_size`,
/// `num_position_embeddings` `posemb_grid_size`², `hidden_act` "gelu" as the erf GELU, `layer_norm_eps` as given.
/// An image is one frame: `temporal_patch_size` 1 (the tower embeds 3 x patch x patch values per patch; the config's
/// value describes video, which Inkling serves never). Tokens: the top-level `image_token_id` (absent: 200054), no
/// video token, the template's image frame (200005, 200010), no deepstack. `None` without `vision_config`; the
/// library refuses any value that differs from its tower.
pub fn inkling_vision_config(config: &Value) -> Result<Option<Qwen4ExpVisionConfig>, UploadError> {
    let Some(vision) = config.get("vision_config") else {
        return Ok(None);
    };
    let vision = vision
        .as_object()
        .ok_or_else(|| UploadError::plan("config", "config.json vision_config is not an object"))?;
    let v = Keys(vision, "vision_config");
    let or = |key: &str, alias: Option<&str>, default: u32| -> Result<u32, UploadError> {
        match [Some(key), alias]
            .into_iter()
            .flatten()
            .find(|k| vision.contains_key(*k))
        {
            Some(k) => v.u32(k),
            None => Ok(default),
        }
    };
    let hidden_size = or("hidden_size", None, 1024)?;
    let intermediate_size = if vision.contains_key("intermediate_size") {
        v.u32("intermediate_size")?
    } else {
        let ratio = if vision.contains_key("mlp_ratio") {
            v.f64("mlp_ratio")?
        } else {
            4.0
        };
        let size = f64::from(hidden_size) * ratio;
        if !(size >= 1.0 && size <= f64::from(u32::MAX) && size.fract() == 0.0) {
            return Err(v.err("mlp_ratio", "a ratio giving an integer intermediate size"));
        }
        size as u32
    };
    let hidden_act = match v.str("hidden_act")? {
        None | Some("gelu") => ffi::ACT_GELU_ERF,
        Some(_) => return Err(v.err("hidden_act", "\"gelu\"")),
    };
    let grid = or("posemb_grid_size", None, 32)?;
    let num_position_embeddings = grid
        .checked_mul(grid)
        .ok_or_else(|| v.err("posemb_grid_size", "a grid side below 65,536"))?;
    let layer_norm_eps = if vision.contains_key("layer_norm_eps") {
        v.f64("layer_norm_eps")?
    } else {
        1e-5
    };
    let image_token_id = match config.get("image_token_id") {
        None => INKLING_IMAGE_TOKEN,
        Some(id) => id
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| {
                UploadError::plan("config", "config.image_token_id is not a token ID")
            })?,
    };
    Ok(Some(Qwen4ExpVisionConfig {
        struct_size: ffi::struct_size::<Qwen4ExpVisionConfig>(),
        abi_version: ffi::ABI_VERSION,
        depth: or("num_hidden_layers", None, 23)?,
        hidden_size,
        intermediate_size,
        num_heads: or("num_attention_heads", None, 16)?,
        patch_size: or("patch_size", Some("native_patch_size"), 20)?,
        temporal_patch_size: 1,
        spatial_merge_size: or("spatial_merge_size", Some("adapter_merge_size"), 2)?,
        in_channels: or("num_channels", None, 3)?,
        out_hidden_size: or("text_hidden_size", None, 4608)?,
        num_position_embeddings,
        deepstack_count: 0,
        hidden_act,
        image_token_id,
        video_token_id: 0,
        vision_start_token_id: INKLING_IMAGE_START,
        vision_end_token_id: INKLING_IMAGE_END,
        layer_norm_eps,
    }))
}

/// Appends the `eos_token_id` of `obj` (an integer or a list of integers) to `out`, skipping repeats.
fn push_eos(obj: &Value, what: &str, out: &mut Vec<u32>) -> Result<(), UploadError> {
    let bad = || {
        UploadError::plan(
            "config",
            format!("{what}.eos_token_id is not a token ID or a list of them"),
        )
    };
    let ids: Vec<&Value> = match obj.get("eos_token_id") {
        None | Some(Value::Null) => return Ok(()),
        Some(Value::Array(a)) => a.iter().collect(),
        Some(v) => vec![v],
    };
    for v in ids {
        let id = v
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(bad)?;
        if !out.contains(&id) {
            out.push(id);
        }
    }
    Ok(())
}

/// The sampling defaults of `generation_config.json` (m27 D10): `temperature` (>= 0), `top_k` (-1 or >= 1) and
/// `top_p` (in (0, 1]), each absent → `None`; a value outside its range refuses the plan.
fn generation_defaults(obj: &Value) -> Result<GenerationDefaults, UploadError> {
    let bad = |key: &str, want: &str| {
        UploadError::plan("config", format!("generation_config.{key} is not {want}"))
    };
    let number = |key: &str, ok: fn(f64) -> bool, want: &str| match obj.get(key) {
        None => Ok(None),
        Some(v) => v
            .as_f64()
            .filter(|x| ok(*x))
            .map(Some)
            .ok_or_else(|| bad(key, want)),
    };
    Ok(GenerationDefaults {
        temperature: number("temperature", |t| t >= 0.0, "a number >= 0")?,
        top_k: match obj.get("top_k") {
            None => None,
            Some(v) => Some(
                v.as_i64()
                    .filter(|k| *k == -1 || *k >= 1)
                    .ok_or_else(|| bad("top_k", "-1 or an integer >= 1"))?,
            ),
        },
        top_p: number("top_p", |p| p > 0.0 && p <= 1.0, "a number in (0, 1]")?,
    })
}

/// The artifact profile of the FP4-Local converted checkpoint (fp4local design note D8, D12, D14): NVFP4 PLE table
/// (U8 packed rows, per-shard E4M3 block scales, one F32 global), every other tensor in the shipped form.
pub const PLE_NVFP4_PROFILE_ID: &str = "rtxpro6000bw-plehost-nvfp4-text-v1";

/// The artifact profile of the S3 conversion (`cerec convert --format s3`, fp4local design note S3, FL-30): the
/// [`PLE_NVFP4_PROFILE_ID`] layout plus MXFP8 dense weights (F8_E4M3 `X.weight` + U8 UE8M0 `X.weight_scale`) and the
/// fused NVFP4 MTP experts (U8 packed, F8_E4M3 block scales, F32 globals). With [`S3_MXFP8_VISION_PROFILE_ID`] the
/// only profiles admitting either.
pub const S3_MXFP8_PROFILE_ID: &str = "rtxpro6000bw-plehost-nvfp4-mxfp8-text-v1";

/// [`S3_MXFP8_PROFILE_ID`] plus the BF16 vision tower on the device (m213l design note D-L1, profile `-v43`): the
/// same text tensor forms.
pub const S3_MXFP8_VISION_PROFILE_ID: &str = "rtxpro6000bw-plehost-nvfp4-mxfp8-vision-v1";

fn is_s3_profile(profile_id: &str) -> bool {
    profile_id == S3_MXFP8_PROFILE_ID || profile_id == S3_MXFP8_VISION_PROFILE_ID
}

/// Storage of the PLE table (the executor's `k4::TableFormat`, chosen by its binder from the table dtype).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PleTableFormat {
    /// E4M3 rows times one BF16 scale (the shipped checkpoint's).
    Fp8,
    /// Packed E2M1 rows, E4M3 block scales per 16 elements, one F32 global (K4's NVFP4 row decode, D8).
    Nvfp4,
}

/// The PLE table format of an artifact of profile `profile_id` whose `ple_table` tensors have `dtypes`
/// (fp4local design note D14: the K4 NVFP4 variant is reachable only from [`PLE_NVFP4_PROFILE_ID`] and the S3
/// profiles): every table shard U8 under those profiles, every shard F8_E4M3 under any other; anything
/// else is class `ple_table` (a U8 table under another profile, an E4M3 table under an NVFP4 profile, a mix, another
/// dtype).
pub fn ple_table_format(
    profile_id: &str,
    dtypes: impl IntoIterator<Item = Dtype>,
) -> Result<PleTableFormat, UploadError> {
    let (format, want) = if profile_id == PLE_NVFP4_PROFILE_ID || is_s3_profile(profile_id) {
        (PleTableFormat::Nvfp4, Dtype::U8)
    } else {
        (PleTableFormat::Fp8, Dtype::F8E4m3)
    };
    match dtypes.into_iter().find(|d| *d != want) {
        None => Ok(format),
        Some(d) => Err(UploadError::plan(
            "ple_table",
            format!(
                "artifact profile {profile_id}: a PLE table shard of dtype {}, expected {} \
                 (an NVFP4 table needs profile {PLE_NVFP4_PROFILE_ID}, {S3_MXFP8_PROFILE_ID} or \
                 {S3_MXFP8_VISION_PROFILE_ID})",
                d.name(),
                want.name()
            ),
        )),
    }
}

/// fp4local design note S3 (FL-30): the quantized dense and draft tensors of an artifact of profile `profile_id`
/// (`(name, role, dtype)`). A non-BF16 `dense` tensor is MXFP8 (class `mxfp8`: an F8_E4M3 `.weight` or a U8
/// `.weight_scale`), a non-BF16 `draft` tensor an NVFP4 MTP expert piece (class `mtp_nvfp4`: U8 packed, F8_E4M3
/// block scales or F32 globals of `mtp.*.mlp.experts.*`); both are admitted only under the S3 profiles, and
/// any other non-BF16 dense/draft tensor is refused under every profile.
pub fn s3_quantized_tensors<'a>(
    profile_id: &str,
    tensors: impl IntoIterator<Item = (&'a str, Role, Dtype)>,
) -> Result<(), UploadError> {
    let s3 = is_s3_profile(profile_id);
    for (name, role, dtype) in tensors {
        let (class, form_ok) = match role {
            Role::Dense => (
                "mxfp8",
                (dtype == Dtype::F8E4m3 && name.ends_with(".weight"))
                    || (dtype == Dtype::U8 && name.ends_with(".weight_scale")),
            ),
            Role::Draft => (
                "mtp_nvfp4",
                matches!(dtype, Dtype::U8 | Dtype::F8E4m3 | Dtype::F32)
                    && name.starts_with("mtp.")
                    && name.contains(".mlp.experts."),
            ),
            _ => continue,
        };
        if dtype == Dtype::Bf16 || (s3 && form_ok) {
            continue;
        }
        return Err(UploadError::plan(
            class,
            if form_ok {
                format!(
                    "artifact profile {profile_id}: {name} is {} ({class}), admitted only under profile \
                     {S3_MXFP8_PROFILE_ID} or {S3_MXFP8_VISION_PROFILE_ID}",
                    dtype.name()
                )
            } else {
                format!(
                    "artifact profile {profile_id}: {name} is {}, neither BF16 nor an S3 {class} tensor",
                    dtype.name()
                )
            },
        ));
    }
    Ok(())
}

/// The model desc of `art` (opened from `dir`): tensor table in [`Artifact::tensors`] order without `unused`
/// tensors, the config struct, the model stop tokens, the sampling defaults of `generation_config.json` and, when the
/// plan speculates, the artifact's draft token map.
pub fn model_spec(
    art: &Artifact,
    dir: &Path,
    opts: &PlanOptions,
) -> Result<ModelSpec, UploadError> {
    let raw = read_role(art, dir, "config")?
        .ok_or_else(|| UploadError::plan("config", "the artifact lists no config.json"))?;
    let config_json = parse_json(&raw, "config.json")?;
    // Only family inkling2_loop reads its quant config (the expert format per layer, `mxfp8_layer_mask`).
    let quant = match config_json.get("model_type").and_then(Value::as_str) {
        Some("inkling2_loop") => read_role(art, dir, "quant_config")?
            .map(|raw| parse_json(&raw, "hf_quant_config.json"))
            .transpose()?,
        _ => None,
    };
    let config = model_config(&config_json, quant.as_ref())?;
    let mut stop_token_ids = Vec::new();
    let mut defaults = GenerationDefaults::default();
    if let Some(raw) = read_role(art, dir, "generation_config")? {
        let generation = parse_json(&raw, "generation_config.json")?;
        push_eos(&generation, "generation_config", &mut stop_token_ids)?;
        defaults = generation_defaults(&generation)?;
    }
    match config_json.get("text_config") {
        Some(text) => push_eos(text, "text_config", &mut stop_token_ids)?,
        // The flat sdar config carries its `eos_token_id` at the top level.
        None if matches!(config, ModelConfig::Sdar(_)) => {
            push_eos(&config_json, "config", &mut stop_token_ids)?
        }
        None => {}
    }
    ple_table_format(
        &art.profile.profile_id,
        art.tensors()
            .filter(|t| t.role == Role::PleTable)
            .map(|t| t.dtype),
    )?;
    // Family inkling2_loop stores its NVFP4 / MXFP8 experts and FP32 router / depth-mix tensors as is; its
    // library checks every tensor's dtype by name (inkling design note S0), so the Qwen S3 rule does not apply.
    if !matches!(config, ModelConfig::Inkling2(_)) {
        s3_quantized_tensors(
            &art.profile.profile_id,
            art.tensors().map(|t| (t.name, t.role, t.dtype)),
        )?;
    }
    let mut tensors = Vec::new();
    let mut seen = HashMap::new();
    for t in art.tensors() {
        let Some(placement) = abi_placement(&t, opts.speculative_draft_tokens) else {
            continue;
        };
        let dtype = abi_dtype(t.dtype).ok_or_else(|| {
            UploadError::plan(
                "dtype",
                format!("{}: dtype {} has no ABI v3 value", t.name, t.dtype.name()),
            )
        })?;
        if seen.insert(t.name, t.path).is_some() {
            return Err(UploadError::plan(
                "tensor_table",
                format!("tensor {} appears in more than one shard", t.name),
            ));
        }
        tensors.push(TensorSpec {
            name: t.name.to_string(),
            dtype,
            placement,
            shape: t.shape.to_vec(),
            bytes: t.bytes,
        });
    }
    // The vision tower enters the desc only with its tensors (a text-only profile leaves `model.visual.*`
    // `unused`, so the desc stays text-only even though `config.json` describes the tower).
    let vision = if !tensors.iter().any(|t| t.name.starts_with("model.visual.")) {
        None
    } else if matches!(config, ModelConfig::Inkling2(_)) {
        inkling_vision_config(&config_json)?
    } else {
        qwen4_exp_vision_config(&config_json)?
    };
    // D11: the token map rides only with drafts on; the executor validates it (strictly increasing, < vocab). Inkling's
    // MTP drafter (inkling design note D-I-S9B) drafts over the full vocabulary: its desc carries none, and drafts
    // on need the artifact's drafter on the device (a profile that stores it `unused` cannot speculate).
    let inkling = matches!(config, ModelConfig::Inkling2(_));
    if inkling
        && opts.speculative_draft_tokens > 0
        && !art.tensors().any(|t| {
            t.role == Role::Draft && abi_placement(&t, opts.speculative_draft_tokens).is_some()
        })
    {
        return Err(UploadError::plan(
            "draft",
            format!(
                "speculative_draft_tokens > 0 needs the drafter placed by the artifact's profile ({} stores it unused)",
                art.profile.profile_id
            ),
        ));
    }
    let draft_token_map = if opts.speculative_draft_tokens > 0 && !inkling {
        let map = art
            .token_map()
            .map_err(|e| UploadError::plan("token_map", e.message))?;
        if map.is_empty() {
            return Err(UploadError::plan(
                "token_map",
                "speculative_draft_tokens > 0 needs the artifact's draft token map",
            ));
        }
        map.to_vec()
    } else {
        Vec::new()
    };
    Ok(ModelSpec {
        config,
        vision,
        tensors,
        max_sessions: opts.max_sessions,
        max_context_tokens: opts.max_context_tokens,
        kv_pool_tokens: opts.kv_pool_tokens,
        stop_token_ids,
        generation_defaults: defaults,
        snapshot_host_bytes: opts.snapshot_host_bytes,
        prefix_state_slots: opts.prefix_state_slots,
        speculative_draft_tokens: opts.speculative_draft_tokens,
        draft_token_map,
        // m23 design note PI-03: the library defaults; a profile's `[deployments.native.levers]` replaces them.
        levers: crate::levers::LeverSet::default(),
        // m210 design note D15⁗′ stage 2a: off; a profile's `prefix_end_state = true` turns it on.
        prefix_end_state: false,
        // m210 design note D17: off; a profile's `prefix_alt_edges = true` turns it on.
        prefix_alt_edges: false,
        // Track LoRA: no adapter pool; a profile's `[deployments.native.lora]` reserves one.
        lora: None,
        // WeightLab: no research reserve; a profile's `[deployments.native.research]` makes one.
        research: None,
    })
}

/// Bytes handed to `cere_model_put`, by ABI placement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct UploadCounters {
    pub uploaded_device_bytes: u64,
    pub uploaded_host_bytes: u64,
    pub discarded_bytes: u64,
    /// HOST_PINNED bytes covered from an attached shared segment without a read or a put (m216 L4).
    pub resident_host_bytes: u64,
}

/// The staging pass (`Artifact::stage_to` of class `all`, m216 design note L7).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct PassReport {
    pub tensors: u64,
    pub bytes: u64,
    pub high_water_bytes: u64,
    /// Reader threads, and put threads per model.
    pub threads: usize,
    pub consumers: usize,
    /// Tensor-data reads (a run of contiguous tensors is one, so is each buffer-sized piece of a larger tensor).
    pub reads: u64,
    pub read: ReadMode,
    /// Files read with `O_DIRECT`.
    pub direct_files: u64,
    pub ms: u64,
    /// `bytes` over `ms`, MB/s.
    pub mb_per_s: u64,
    /// Summed thread time inside the tensor-data reads (page cache or disk into the staging buffers).
    pub pread_ms: u64,
    /// Summed reader time waiting for a free buffer (the puts are behind), put-thread time waiting for a filled
    /// one (the reads are behind), and time inside the puts.
    pub wait_free_ms: u64,
    pub wait_fill_ms: u64,
    pub deliver_ms: u64,
    /// Reads in flight sampled every 50 ms: mean (hundredths) and most; and per second, with the bytes read in it.
    pub reads_in_flight_mean_x100: u64,
    pub reads_in_flight_max: u64,
    pub timeline: Vec<Second>,
}

impl PassReport {
    fn new(r: StageReport, read: ReadMode) -> Self {
        PassReport {
            tensors: r.tensors,
            bytes: r.bytes,
            high_water_bytes: r.high_water_bytes,
            threads: r.threads,
            consumers: r.consumers,
            reads: r.reads,
            read,
            direct_files: r.direct_files,
            ms: r.read_ms,
            mb_per_s: r.bytes.checked_div(r.read_ms).map_or(0, |b| b / 1000),
            pread_ms: r.pread_ms,
            wait_free_ms: r.wait_free_ms,
            wait_fill_ms: r.wait_fill_ms,
            deliver_ms: r.deliver_ms,
            reads_in_flight_mean_x100: r.reads_in_flight_mean_x100,
            reads_in_flight_max: r.reads_in_flight_max,
            timeline: r.timeline,
        }
    }
}

/// What [`stage_all`] did for one model: READ and UPLOAD overlap (readers fill free buffers while the put threads
/// drain filled ones), so the pass time covers both.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct StagingReport {
    pub counters: UploadCounters,
    pub pass: PassReport,
    /// Summed time inside `cere_model_put` over all put threads.
    pub put_ms: u64,
    /// The DEVICE puts' share of `put_ms` (registered staging: one DMA and its stream sync per put).
    pub device_put_ms: u64,
    /// Files checked from the attached segment's views instead of read (m216 L4), and the longest check.
    pub resident_files: u64,
    pub resident_ms: u64,
    /// Staging bytes registered as put staging (`cere_executor_register_staging`, m216 design note M216S-11) for
    /// the pass, as one range; 0 when the library predates put staging and every put took the bounce ring.
    pub registered_bytes: u64,
    /// The phases of [`stage_all`] around the pass: the desc index and all-HOST_PINNED files ([`TargetIndex`], built
    /// beside ALLOCATE by [`upload`]), the resident views, the staging allocation, its registration, the resident
    /// checks still running after the pass, and the unregistration.
    pub phases_ms: StagePhases,
    /// m32n design note D22 (3): the nice value the pass's threads ran at ([`BACKGROUND_NICE`] for a background
    /// upload, else the caller's).
    pub nice: i32,
    /// D22 (3): the most staging buffers' worth of bytes in flight (read and not yet put by every target): the
    /// pass's high water over the buffer size, rounded up, so at most the buffer count.
    pub copies_in_flight_max: u64,
}

/// [`StagingReport::phases_ms`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct StagePhases {
    /// [`TargetIndex::new`] for every target (LoadCrit LC-02: beside ALLOCATE in [`upload`], so not on the path).
    pub index: u64,
    pub prepare: u64,
    pub buffers: u64,
    pub register: u64,
    pub pass: u64,
    pub checks_after_pass: u64,
    pub unregister: u64,
}

/// [`upload`]'s result: the sealed model, its capabilities, the staging report and the stage times.
#[derive(Debug)]
pub struct Upload {
    pub model: Model,
    pub capabilities: ExecutionCapabilities,
    /// BC1 leftover (`cere_model_prefill_signature`, read right after the seal): the sealed model's prefill
    /// selection digest; `None` when the library predates the call or its device has no selection.
    pub prefill_signature: Option<[u8; 32]>,
    pub staging: StagingReport,
    /// ALLOCATE (`cere_model_create`).
    pub allocate_ms: u64,
    /// D22 (3): the nice value and CPU count of the thread ALLOCATE ran on: a background load's helper thread
    /// ([`BACKGROUND_NICE`] on [`BACKGROUND_ALLOCATE_CPUS`]), else the driver thread.
    pub allocate: AllocateThread,
    /// READ + UPLOAD (both passes).
    pub upload_ms: u64,
    /// INSTANTIATE (`cere_model_seal`).
    pub instantiate_ms: u64,
}

/// The sink: each staging thread puts its piece straight into the executor (Q13). It declines the DISCARD tensors
/// and the tensors outside the desc, so their bytes are never read (m216 L1).
struct PutSink<'a> {
    putter: Putter<'a>,
    index: HashMap<&'a str, (u64, u32)>,
    /// Tensors covered from an attached segment (m216 L4): never read, never put.
    resident: HashSet<&'a str>,
    device: AtomicU64,
    host: AtomicU64,
    put_ns: AtomicU64,
    device_put_ns: AtomicU64,
}

impl Sink for PutSink<'_> {
    fn wants(&self, tensor: &TensorRef<'_>) -> bool {
        !self.resident.contains(tensor.name)
            && self
                .index
                .get(tensor.name)
                .is_some_and(|&(_, placement)| placement != ffi::PLACEMENT_DISCARD)
    }

    /// load-soft design note LSF-01: HOST_PINNED tensors are read after the DEVICE ones, so the executor pins its host
    /// block (cudaHostRegister, which stalls the device puts) while only host copies remain.
    fn late(&self, tensor: &TensorRef<'_>) -> bool {
        self.index
            .get(tensor.name)
            .is_some_and(|&(_, placement)| placement == ffi::PLACEMENT_HOST_PINNED)
    }

    fn accept(
        &self,
        tensor: &TensorRef<'_>,
        offset_in_tensor: u64,
        data: &[u8],
        _is_last: bool,
    ) -> Result<(), String> {
        let &(index, placement) = self
            .index
            .get(tensor.name)
            .ok_or_else(|| "the tensor is not in the model desc".to_string())?;
        // A 0-byte tensor arrives as one empty piece; there is nothing to tile.
        if data.is_empty() {
            return Ok(());
        }
        let t = Instant::now();
        let put = self.putter.put(index, offset_in_tensor, data);
        let put_ns = t.elapsed().as_nanos() as u64;
        self.put_ns.fetch_add(put_ns, Ordering::Relaxed);
        put.map_err(|e| e.to_string())?;
        let counter = match placement {
            ffi::PLACEMENT_DEVICE => {
                self.device_put_ns.fetch_add(put_ns, Ordering::Relaxed);
                &self.device
            }
            ffi::PLACEMENT_HOST_PINNED => &self.host,
            _ => return Err(format!("placement {placement} is never staged")),
        };
        counter.fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(())
    }
}

/// A weights file path and its tensors' bytes in an attached shared segment, in file-offset order (m216 L4).
type ResidentFile<'a> = (&'a str, Vec<&'a [u8]>);

/// A target's put index (desc name -> index and placement) and its weights files whose tensors are all HOST_PINNED
/// (path order, tensors in file-offset order). Built from the artifact and the spec alone (~300K tensors), so
/// [`upload`] builds it beside ALLOCATE: built after it, beside the executor's populate of the host block, it took
/// ~1.6 s of the pass's start (load-crit design note LC-02, O158).
pub struct TargetIndex<'a> {
    index: HashMap<&'a str, (u64, u32)>,
    pinned: Vec<(&'a str, Vec<TensorRef<'a>>)>,
}

impl<'a> TargetIndex<'a> {
    pub fn new(art: &'a Artifact, spec: &'a ModelSpec) -> Self {
        let mut index = HashMap::with_capacity(spec.tensors.len());
        index.extend(
            spec.tensors
                .iter()
                .enumerate()
                .map(|(i, t)| (t.name.as_str(), (i as u64, t.placement))),
        );
        let mut all_pinned: BTreeMap<&'a str, bool> = BTreeMap::new();
        for t in art.tensors() {
            let p = index
                .get(t.name)
                .is_some_and(|&(_, p)| p == ffi::PLACEMENT_HOST_PINNED);
            *all_pinned.entry(t.path).or_insert(true) &= p;
        }
        let mut pinned: Vec<(&'a str, Vec<TensorRef<'a>>)> = all_pinned
            .into_iter()
            .filter(|&(_, p)| p)
            .map(|(path, _)| (path, Vec::new()))
            .collect();
        if !pinned.is_empty() {
            for t in art.tensors() {
                if let Ok(k) = pinned.binary_search_by(|(path, _)| (*path).cmp(t.path)) {
                    pinned[k].1.push(t);
                }
            }
        }
        TargetIndex { index, pinned }
    }
}

/// The all-HOST_PINNED files' tensors' bytes when the model's host region was attached from a published shared
/// segment (m216 L4): each tensor is covered by `cere_model_host_resident`, so it must not be put. Empty for a
/// private region or a segment this process creates.
fn resident_files<'a>(
    putter: &Putter<'a>,
    target: &TargetIndex<'a>,
) -> Result<Vec<ResidentFile<'a>>, UploadError> {
    let mut out = Vec::new();
    for (path, tensors) in &target.pinned {
        let mut views = Vec::with_capacity(tensors.len());
        for t in tensors {
            match putter.host_resident(target.index[t.name].0, t.bytes)? {
                Some(v) => views.push(v),
                // The region is private or created here: nothing was covered (a model's region is one kind).
                None if out.is_empty() && views.is_empty() => return Ok(Vec::new()),
                None => {
                    return Err(UploadError::plan(
                        "resident",
                        format!("{path}: {} is not resident in the attached segment", t.name),
                    ))
                }
            }
        }
        out.push((*path, views));
    }
    Ok(out)
}

/// The staging block registered as put staging with one model's executor for one pass (m216 design note
/// M216S-11), as ONE range: one `cudaHostRegister` instead of one per buffer, so the pass start does not queue a
/// registration per buffer behind the host region's background pin pieces. [`Registered::release`] unregisters
/// it; on an early return or a panic the drop unregisters it (best effort) before the block, which is declared
/// before it, is freed.
struct Registered<'p, 'a> {
    putter: &'p Putter<'a>,
    base: Option<*const u8>,
    bytes: u64,
}

impl<'p, 'a> Registered<'p, 'a> {
    /// Registers `block`; nothing (and `Ok`) when the library has no put staging.
    fn register(putter: &'p Putter<'a>, block: &[u8]) -> Result<Self, UploadError> {
        // SAFETY: the block outlives the registration (declared before it by the caller) and is neither freed nor
        // reallocated while registered; `release` or the drop unregisters it first.
        let registered = unsafe { putter.register_staging(block.as_ptr(), block.len()) }?;
        Ok(Registered {
            putter,
            base: registered.then_some(block.as_ptr()),
            bytes: if registered { block.len() as u64 } else { 0 },
        })
    }

    fn release(mut self) -> Result<(), UploadError> {
        match self.base.take() {
            Some(base) => self.putter.unregister_staging(base).map_err(Into::into),
            None => Ok(()),
        }
    }
}

impl Drop for Registered<'_, '_> {
    fn drop(&mut self) {
        if let Some(base) = self.base {
            let _ = self.putter.unregister_staging(base);
        }
    }
}

/// One model [`stage_all_to`] uploads into: unsealed, created from `spec` on `driver`.
#[derive(Clone, Copy)]
pub struct Target<'a> {
    pub driver: &'a Driver,
    pub model: &'a Model,
    pub spec: &'a ModelSpec,
}

fn ms_since(t: Instant) -> u64 {
    t.elapsed().as_millis() as u64
}

/// A scoped thread's result; its panic resumes on the caller.
fn joined<T>(h: std::thread::ScopedJoinHandle<'_, T>) -> T {
    h.join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

/// m32n design note D22 (3): lowers the calling thread, and so every thread it starts afterwards, to
/// [`BACKGROUND_NICE`]; the nice value it then has.
fn lower_priority() -> Result<i32, UploadError> {
    // SAFETY: changes only the calling thread's own nice value; lowering a priority needs no privilege.
    let lowered = unsafe {
        libc::setpriority(
            libc::PRIO_PROCESS,
            libc::gettid() as libc::id_t,
            BACKGROUND_NICE,
        )
    };
    if lowered != 0 {
        return Err(UploadError::Stage(cere_compiler::Error::reject(
            "priority",
            format!(
                "the background load's priority: {}",
                std::io::Error::last_os_error()
            ),
        )));
    }
    Ok(thread_nice())
}

/// UPLOAD into one unsealed `model` created from `spec`: [`stage_all_to`] with one target.
pub fn stage_all(
    driver: &Driver,
    model: &Model,
    art: &Artifact,
    spec: &ModelSpec,
    opts: &UploadOptions,
    block: Option<artifact::Buffer>,
) -> Result<StagingReport, UploadError> {
    let target = Target {
        driver,
        model,
        spec,
    };
    let mut reports = stage_all_to(art, &[target], opts, block)?;
    Ok(reports.remove(0))
}

/// The staging block's length for `opts`: `buffers` x `buffer_bytes` (at least one buffer).
fn staging_bytes(opts: &UploadOptions) -> Result<usize, UploadError> {
    if opts.buffers == 0 {
        return Err(UploadError::Stage(cere_compiler::Error::reject(
            "buffer",
            "at least one staging buffer is needed",
        )));
    }
    opts.buffers.checked_mul(opts.buffer_bytes).ok_or_else(|| {
        UploadError::Stage(cere_compiler::Error::reject(
            "buffer",
            "the staging block overflows",
        ))
    })
}

/// A staging block for `opts` ([`staging_bytes`]), zeroed (so its pages are faulted in), for [`stage_all_to`].
pub fn staging_block(opts: &UploadOptions) -> Result<artifact::Buffer, UploadError> {
    artifact::Buffer::new(staging_bytes(opts)?).map_err(UploadError::Stage)
}

/// UPLOAD into every target (m216 design note L7: one producer, N consumers): one `stage_to(All)` pass reads each
/// tensor some target wants (DEVICE and HOST_PINNED) once into the staging buffers, and each target's put threads
/// put every piece it wants; a buffer is read again only after every target's put of it has returned (so its DMA
/// is complete). A refused put aborts staging (class `sink`). The staging buffers are slices of one block
/// registered as put staging with each target's executor when its library has the two exports (m216 design note
/// M216S-11: DEVICE puts DMA straight from them), else that target's puts take its library's bounce ring.
/// `discarded_bytes` is the DISCARD tensors' bytes, which are skipped (the executor needs no put for them). With an
/// attached shared segment the all-HOST_PINNED files are not read for that target: each is checked against its
/// manifest sha256 from the segment's bytes on its own thread while the pass stages the rest (m216 L4); a
/// difference fails `file_hash`. Reports are in target order. `block` is the staging block when the caller made it
/// beforehand ([`staging_block`] of the same `opts`; the `buffers` phase then only checks it), else it is made here.
pub fn stage_all_to(
    art: &Artifact,
    targets: &[Target<'_>],
    opts: &UploadOptions,
    block: Option<artifact::Buffer>,
) -> Result<Vec<StagingReport>, UploadError> {
    let t = Instant::now();
    let indexes = targets
        .iter()
        .map(|target| TargetIndex::new(art, target.spec))
        .collect();
    stage_indexed(art, targets, indexes, ms_since(t), opts, block)
}

/// [`stage_all_to`] with each target's [`TargetIndex`] (target order) built by the caller in `index_ms`.
fn stage_indexed<'a>(
    art: &'a Artifact,
    targets: &[Target<'a>],
    indexes: Vec<TargetIndex<'a>>,
    index_ms: u64,
    opts: &UploadOptions,
    block: Option<artifact::Buffer>,
) -> Result<Vec<StagingReport>, UploadError> {
    let t = Instant::now();
    let mut sinks = Vec::with_capacity(targets.len());
    let mut files = Vec::with_capacity(targets.len());
    for (target, ix) in targets.iter().zip(indexes) {
        let putter = target.driver.putter(target.model);
        let resident_views = resident_files(&putter, &ix)?;
        let resident = resident_views
            .iter()
            .flat_map(|(path, _)| art.tensors().filter(move |t| t.path == *path))
            .map(|t| t.name)
            .collect();
        sinks.push(PutSink {
            putter,
            index: ix.index,
            resident,
            device: AtomicU64::new(0),
            host: AtomicU64::new(0),
            put_ns: AtomicU64::new(0),
            device_put_ns: AtomicU64::new(0),
        });
        files.push(resident_views);
    }
    let mut phases = StagePhases {
        index: index_ms,
        prepare: ms_since(t),
        ..StagePhases::default()
    };
    let t = Instant::now();
    let bytes = staging_bytes(opts)?;
    let mut block = match block {
        Some(b) if b.len() == bytes => b,
        Some(b) => {
            return Err(UploadError::Stage(cere_compiler::Error::reject(
                "buffer",
                format!(
                    "the staging block holds {} bytes, the options need {bytes}",
                    b.len()
                ),
            )))
        }
        None => artifact::Buffer::new(bytes).map_err(UploadError::Stage)?,
    };
    phases.buffers = ms_since(t);
    let t = Instant::now();
    let mut registered = Vec::with_capacity(sinks.len());
    for sink in &sinks {
        registered.push(Registered::register(&sink.putter, block.as_mut_slice())?);
    }
    phases.register = ms_since(t);
    let registered_bytes: Vec<u64> = registered.iter().map(|r| r.bytes).collect();
    let mut slices: Vec<&mut [u8]> = block.as_mut_slice().chunks_mut(opts.buffer_bytes).collect();
    let stage = StageOptions {
        read: opts.read,
        readers: opts.readers,
        consumers: opts.consumers,
        ..StageOptions::default()
    };
    let sink_refs: Vec<&PutSink<'_>> = sinks.iter().collect();
    let scoped = std::thread::scope(|scope| {
        let mut run = || -> Result<_, UploadError> {
            let nice = if opts.background {
                lower_priority()?
            } else {
                thread_nice()
            };
            let checks: Vec<Vec<_>> = files
                .iter()
                .map(|views| {
                    views
                        .iter()
                        .map(|(path, views)| {
                            scope.spawn(move || {
                                let t = Instant::now();
                                art.check_file_views(path, views).map(|()| ms_since(t))
                            })
                        })
                        .collect()
                })
                .collect();
            let t = Instant::now();
            let pass = art.stage_to(Class::All, &mut slices, &sink_refs, Some(&stage));
            phases.pass = ms_since(t);
            let t = Instant::now();
            let checks: Vec<Vec<_>> = checks
                .into_iter()
                .map(|hs| {
                    hs.into_iter()
                        .map(|h| h.join().expect("a resident check panicked"))
                        .collect()
                })
                .collect();
            phases.checks_after_pass = ms_since(t);
            Ok((nice, pass, checks))
        };
        if opts.background {
            // D22 (3): a thread of its own, so only it and the threads it starts run at the lowered priority.
            scope
                .spawn(run)
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        } else {
            run()
        }
    });
    let t = Instant::now();
    let released: Vec<_> = registered.into_iter().map(Registered::release).collect();
    phases.unregister = ms_since(t);
    let (nice, pass, checks) = scoped?;
    let mut resident_ms = vec![0; targets.len()];
    for (k, cs) in checks.into_iter().enumerate() {
        for c in cs {
            resident_ms[k] = resident_ms[k].max(c.map_err(UploadError::Stage)?);
        }
    }
    let pass = PassReport::new(pass.map_err(UploadError::Stage)?, opts.read);
    let copies_in_flight_max = pass.high_water_bytes.div_ceil(opts.buffer_bytes as u64);
    for r in released {
        r?;
    }
    Ok(sinks
        .into_iter()
        .zip(targets)
        .enumerate()
        .map(|(k, (sink, target))| StagingReport {
            counters: UploadCounters {
                uploaded_device_bytes: sink.device.into_inner(),
                uploaded_host_bytes: sink.host.into_inner(),
                discarded_bytes: target
                    .spec
                    .tensors
                    .iter()
                    .filter(|t| t.placement == ffi::PLACEMENT_DISCARD)
                    .map(|t| t.bytes)
                    .sum(),
                resident_host_bytes: files[k]
                    .iter()
                    .flat_map(|(_, views)| views.iter().map(|v| v.len() as u64))
                    .sum(),
            },
            pass: pass.clone(),
            put_ms: sink.put_ns.into_inner() / 1_000_000,
            device_put_ms: sink.device_put_ns.into_inner() / 1_000_000,
            resident_files: files[k].len() as u64,
            resident_ms: resident_ms[k],
            registered_bytes: registered_bytes[k],
            phases_ms: phases,
            nice,
            copies_in_flight_max,
        })
        .collect())
}

/// ALLOCATE, UPLOAD and INSTANTIATE `spec` from `art`. On any failure after ALLOCATE the unsealed model is released
/// (aborting the upload), so nothing is sealed or left behind.
pub fn upload(
    driver: &Driver,
    art: &Artifact,
    spec: &ModelSpec,
    opts: &UploadOptions,
) -> Result<Upload, UploadError> {
    let t = Instant::now();
    let background = opts.background.then_some(Background {
        nice: BACKGROUND_NICE,
        cpus: BACKGROUND_ALLOCATE_CPUS,
    });
    // load-soft design note LSF-03: a foreground load faults its staging block in beside ALLOCATE, before the
    // executor's background populate of the host block starts (faulting it beside the populate took ~0.8 s).
    // load-crit design note LC-02: every load builds its TargetIndex there too (beside the populate, ~1.6 s).
    let (created, block, (index, index_ms)) = std::thread::scope(|scope| {
        let block = (!opts.background).then(|| scope.spawn(|| staging_block(opts)));
        let index = scope.spawn(|| {
            let t = Instant::now();
            let index = TargetIndex::new(art, spec);
            (index, ms_since(t))
        });
        let created = driver.create_model_on(spec, background);
        let block = block.map(joined);
        (created, block, joined(index))
    });
    let (mut model, allocate) = created?;
    let block = match block.transpose() {
        Ok(b) => b,
        Err(e) => {
            let _ = driver.release_model(model);
            return Err(e);
        }
    };
    let allocate_ms = t.elapsed().as_millis() as u64;
    let t = Instant::now();
    let target = Target {
        driver,
        model: &model,
        spec,
    };
    let staged = stage_indexed(art, &[target], vec![index], index_ms, opts, block);
    let staging = match staged {
        Ok(mut s) => s.remove(0),
        Err(e) => {
            let _ = driver.release_model(model);
            return Err(e);
        }
    };
    let upload_ms = t.elapsed().as_millis() as u64;
    let t = Instant::now();
    let sealed = driver.seal(&mut model).and_then(|capabilities| {
        let prefill_signature = driver.prefill_signature(&model)?;
        Ok((capabilities, prefill_signature))
    });
    let (capabilities, prefill_signature) = match sealed {
        Ok(s) => s,
        Err(e) => {
            let _ = driver.release_model(model);
            return Err(e.into());
        }
    };
    let instantiate_ms = t.elapsed().as_millis() as u64;
    Ok(Upload {
        model,
        capabilities,
        prefill_signature,
        staging,
        allocate_ms,
        allocate,
        upload_ms,
        instantiate_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// D22 (3) rests on Linux's per-thread nice value: the lowered thread and every thread it starts run at
    /// [`BACKGROUND_NICE`], the thread that started it keeps its own.
    #[test]
    fn a_lowered_thread_and_the_threads_it_starts_run_at_the_background_nice() {
        let caller = thread_nice();
        let (lowered, child) = std::thread::spawn(|| {
            let lowered = lower_priority().unwrap();
            (lowered, std::thread::spawn(thread_nice).join().unwrap())
        })
        .join()
        .unwrap();
        assert_eq!((lowered, child), (BACKGROUND_NICE, BACKGROUND_NICE));
        assert_eq!(thread_nice(), caller);
    }
}
