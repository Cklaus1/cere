//! Track LoRA (`-v50`, docs/API.md "LoRA adapters"): PEFT LoRA adapters served on the base model.
//!
//! [`parse_adapter`] turns a PEFT directory (`adapter_config.json`, `adapter_model.safetensors`) into the
//! executor's `cere_model_lora_load` form ([`LoraWeights`]): every `lora_A`/`lora_B` pair of an attention
//! projection (q/k/v/o, full-attention layers only) or of the shared expert (gate/up/down), checked against the
//! model's dimensions and converted to BF16. Any other tensor, a missing half of a pair, a shape or rank mismatch or
//! a PEFT option the executor does not run (DoRA, biases, `modules_to_save`, per-module rank/alpha patterns,
//! `fan_in_fan_out`) refuses the adapter, naming the reason.
//!
//! [`Lora`] is the deployment's adapter registry: name -> slot of the executor's pool (`cere_model_lora_reserve`,
//! made at model create from `[deployments.native.lora]`). A request holds its adapter as a lease (an
//! `Arc<Adapter>` cloned under the registry lock) from acceptance until every session of it is released; an
//! unload while a lease is held is refused busy. Executor-side changes (load, unload, the reapply at a backend
//! load) are serialized, so the executor's pool always mirrors the registry.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::app::NativeBackend;
use crate::config::LoraConfig;
use crate::native::driver::{Driver, DriverError, LoraReserve, LoraWeights, Model};
use crate::native::ffi::{self, Qwen4ExpConfig, StatusError};

/// The PEFT adapter config file of an adapter directory.
pub const CONFIG_FILE: &str = "adapter_config.json";
/// The PEFT adapter weights file of an adapter directory.
pub const WEIGHTS_FILE: &str = "adapter_model.safetensors";
/// The largest `adapter_config.json` read.
const MAX_CONFIG_BYTES: u64 = 1 << 20;

/// The PEFT module paths (after `layers.<N>.`) and their executor module ids.
const MODULES: [(&str, u32); 7] = [
    ("self_attn.q_proj", ffi::LORA_Q_PROJ),
    ("self_attn.k_proj", ffi::LORA_K_PROJ),
    ("self_attn.v_proj", ffi::LORA_V_PROJ),
    ("self_attn.o_proj", ffi::LORA_O_PROJ),
    ("mlp.shared_expert.gate_proj", ffi::LORA_SHARED_GATE),
    ("mlp.shared_expert.up_proj", ffi::LORA_SHARED_UP),
    ("mlp.shared_expert.down_proj", ffi::LORA_SHARED_DOWN),
];

/// The base model's dimensions an adapter must fit (from the sealed model's `cere_qwen4_exp_config`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dims {
    pub hidden: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub shared: u64,
    pub layers: u32,
    /// Bit `i` set: layer `i` is a full-attention layer (the only ones with q/k/v/o targets).
    pub full_attention: u64,
}

impl Dims {
    pub fn of(c: &Qwen4ExpConfig) -> Dims {
        Dims {
            hidden: u64::from(c.hidden_size),
            heads: u64::from(c.num_attention_heads),
            kv_heads: u64::from(c.num_key_value_heads),
            head_dim: u64::from(c.head_dim),
            shared: u64::from(c.shared_expert_intermediate_size),
            layers: c.num_hidden_layers,
            full_attention: c.full_attention_layers,
        }
    }

    /// `(in, out)` of module `module` (q's out carries the output gate: heads x head_dim x 2).
    pub fn shape(&self, module: u32) -> (u64, u64) {
        match module {
            ffi::LORA_Q_PROJ => (self.hidden, self.heads * self.head_dim * 2),
            ffi::LORA_K_PROJ | ffi::LORA_V_PROJ => (self.hidden, self.kv_heads * self.head_dim),
            ffi::LORA_O_PROJ => (self.heads * self.head_dim, self.hidden),
            ffi::LORA_SHARED_GATE | ffi::LORA_SHARED_UP => (self.hidden, self.shared),
            _ => (self.shared, self.hidden),
        }
    }

    fn is_full_attention(&self, layer: u32) -> bool {
        layer < 64 && self.full_attention & (1 << layer) != 0
    }
}

/// A parsed adapter: the executor's form and the digest of its two files.
#[derive(Debug, Clone, PartialEq)]
pub struct Parsed {
    pub weights: LoraWeights,
    /// Lowercase hex SHA-256 over `len(config) u64 LE ‖ config ‖ safetensors`.
    pub sha256: String,
}

/// One target's (A, B) BF16 bytes while the safetensors file is read; either may still be missing.
type PendingPair = (Option<Vec<u8>>, Option<Vec<u8>>);

/// Parses an adapter's `adapter_config.json` and `adapter_model.safetensors` bytes against the model `dims` and the
/// pool's `max_rank`. The error names what is refused.
pub fn parse_adapter(
    config: &[u8],
    weights: &[u8],
    dims: &Dims,
    max_rank: u32,
) -> Result<Parsed, String> {
    let (rank, scale) = parse_config(config, max_rank)?;
    let header =
        cere_compiler::safetensors::parse_file_bytes(weights, &cere_compiler::Limits::default())
            .map_err(|e| format!("{WEIGHTS_FILE}: {e}"))?;
    // target -> (A, B) BF16 bytes.
    let mut pairs: BTreeMap<u32, PendingPair> = BTreeMap::new();
    for t in &header.tensors {
        let (layer, module, is_a) = tensor_target(&t.name)
            .ok_or_else(|| format!("{WEIGHTS_FILE}: unexpected tensor {}", t.name))?;
        if layer >= dims.layers {
            return Err(format!(
                "{WEIGHTS_FILE}: tensor {} targets layer {layer}; the model has {} layers",
                t.name, dims.layers
            ));
        }
        if module <= ffi::LORA_O_PROJ && !dims.is_full_attention(layer) {
            return Err(format!(
                "{WEIGHTS_FILE}: tensor {} targets layer {layer}, not a full_attention layer (q/k/v/o adapt only \
                 full_attention layers)",
                t.name
            ));
        }
        let (input, output) = dims.shape(module);
        let r = u64::from(rank);
        let expected = if is_a { [r, input] } else { [output, r] };
        if t.shape != expected {
            return Err(format!(
                "{WEIGHTS_FILE}: tensor {} has shape {:?}; expected {expected:?} (r = {rank})",
                t.name, t.shape
            ));
        }
        let start = usize::try_from(t.offset).map_err(|_| "tensor offset overflows".to_owned())?;
        let len = usize::try_from(t.bytes).map_err(|_| "tensor size overflows".to_owned())?;
        let raw = weights
            .get(start..start + len)
            .ok_or_else(|| format!("{WEIGHTS_FILE}: tensor {} lies outside the file", t.name))?;
        let bf16 =
            to_bf16(raw, t.dtype).map_err(|e| format!("{WEIGHTS_FILE}: tensor {}: {e}", t.name))?;
        let target = layer * ffi::LORA_MODULES + module;
        let pair = pairs.entry(target).or_default();
        let half = if is_a { &mut pair.0 } else { &mut pair.1 };
        if half.replace(bf16).is_some() {
            return Err(format!(
                "{WEIGHTS_FILE}: tensor {} repeats layer {layer}'s {}",
                t.name,
                module_name(module)
            ));
        }
    }
    if pairs.is_empty() {
        return Err(format!("{WEIGHTS_FILE}: no LoRA tensors"));
    }
    let mut out = LoraWeights {
        rank,
        scale,
        targets: Vec::with_capacity(pairs.len()),
        a: Vec::with_capacity(pairs.len()),
        b: Vec::with_capacity(pairs.len()),
    };
    for (target, pair) in pairs {
        let (layer, module) = (target / ffi::LORA_MODULES, target % ffi::LORA_MODULES);
        let (Some(a), Some(b)) = pair else {
            return Err(format!(
                "{WEIGHTS_FILE}: layer {layer}'s {} lacks its lora_A or lora_B",
                module_name(module)
            ));
        };
        out.targets.push(target);
        out.a.push(a);
        out.b.push(b);
    }
    let mut h = Sha256::new();
    h.update((config.len() as u64).to_le_bytes());
    h.update(config);
    h.update(weights);
    Ok(Parsed {
        weights: out,
        sha256: hex(&h.finalize()),
    })
}

/// `r` and the scale (`lora_alpha / r`, rsLoRA `lora_alpha / sqrt(r)`) of an `adapter_config.json`.
fn parse_config(bytes: &[u8], max_rank: u32) -> Result<(u32, f32), String> {
    let config: Value = serde_json::from_slice(bytes).map_err(|e| format!("{CONFIG_FILE}: {e}"))?;
    let object = config
        .as_object()
        .ok_or_else(|| format!("{CONFIG_FILE}: not a JSON object"))?;
    let field = |key: &str| object.get(key).filter(|v| !v.is_null());
    match field("peft_type").and_then(Value::as_str) {
        Some("LORA") => {}
        _ => return Err(format!("{CONFIG_FILE}: peft_type must be \"LORA\"")),
    }
    let rank = field("r")
        .and_then(Value::as_u64)
        .filter(|r| (1..=u64::from(max_rank)).contains(r))
        .ok_or_else(|| format!("{CONFIG_FILE}: r must be an integer in 1..={max_rank}"))?
        as u32;
    let alpha = field("lora_alpha")
        .and_then(Value::as_f64)
        .filter(|a| a.is_finite() && *a > 0.0)
        .ok_or_else(|| format!("{CONFIG_FILE}: lora_alpha must be a positive number"))?;
    let flag = |key: &str| -> Result<bool, String> {
        match field(key) {
            None => Ok(false),
            Some(Value::Bool(b)) => Ok(*b),
            Some(_) => Err(format!("{CONFIG_FILE}: {key} must be a boolean")),
        }
    };
    let rslora = flag("use_rslora")?;
    for key in ["use_dora", "fan_in_fan_out", "lora_bias"] {
        if flag(key)? {
            return Err(format!("{CONFIG_FILE}: {key} is not supported"));
        }
    }
    match field("bias") {
        None => {}
        Some(Value::String(s)) if s == "none" => {}
        Some(_) => return Err(format!("{CONFIG_FILE}: bias must be \"none\"")),
    }
    for key in ["modules_to_save", "rank_pattern", "alpha_pattern"] {
        let empty = match field(key) {
            None => true,
            Some(Value::Array(a)) => a.is_empty(),
            Some(Value::Object(o)) => o.is_empty(),
            Some(_) => false,
        };
        if !empty {
            return Err(format!("{CONFIG_FILE}: {key} is not supported"));
        }
    }
    let r = f64::from(rank);
    let scale = if rslora { alpha / r.sqrt() } else { alpha / r } as f32;
    if !scale.is_finite() {
        return Err(format!(
            "{CONFIG_FILE}: lora_alpha gives a non-finite scale"
        ));
    }
    Ok((rank, scale))
}

/// `(layer, module, is_a)` of a PEFT tensor name: any prefix, then `layers.<N>.<module>.lora_{A,B}.weight`.
fn tensor_target(name: &str) -> Option<(u32, u32, bool)> {
    let at = name.rfind("layers.")?;
    if at > 0 && !name[..at].ends_with('.') {
        return None;
    }
    let rest = &name[at + "layers.".len()..];
    let (layer, rest) = rest.split_once('.')?;
    if layer.is_empty() || !layer.bytes().all(|c| c.is_ascii_digit()) || layer.len() > 4 {
        return None;
    }
    let (path, is_a) = if let Some(p) = rest.strip_suffix(".lora_A.weight") {
        (p, true)
    } else {
        (rest.strip_suffix(".lora_B.weight")?, false)
    };
    let module = MODULES.iter().find(|(m, _)| *m == path)?.1;
    Some((layer.parse().ok()?, module, is_a))
}

fn module_name(module: u32) -> &'static str {
    MODULES
        .iter()
        .find(|(_, m)| *m == module)
        .map_or("?", |(name, _)| name)
}

/// Little-endian BF16 bytes of `raw` (BF16 copied, F16/F32 rounded to nearest even); a non-finite value refuses.
fn to_bf16(raw: &[u8], dtype: cere_compiler::safetensors::Dtype) -> Result<Vec<u8>, String> {
    use cere_compiler::safetensors::Dtype;
    let finite = |bits: u16| bits & 0x7f80 != 0x7f80;
    match dtype {
        Dtype::Bf16 => {
            if raw
                .chunks_exact(2)
                .any(|c| !finite(u16::from_le_bytes([c[0], c[1]])))
            {
                return Err("a non-finite value".to_owned());
            }
            Ok(raw.to_vec())
        }
        Dtype::F16 => raw
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .try_fold(Vec::with_capacity(raw.len()), push_bf16),
        Dtype::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .try_fold(Vec::with_capacity(raw.len() / 2), push_bf16),
        other => Err(format!("dtype {other:?}; expected BF16, F16 or F32")),
    }
}

fn push_bf16(mut out: Vec<u8>, v: f32) -> Result<Vec<u8>, String> {
    if !v.is_finite() {
        return Err("a non-finite value".to_owned());
    }
    out.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
    Ok(out)
}

/// Round-to-nearest-even of a finite `f32` to BF16 bits.
pub fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((f & 0x3ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::with_capacity(64), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// The canonical adapter directory of `path` (absolute, or relative to the root), which must resolve inside
/// `root` (both with symlinks resolved).
fn resolve(root: &Path, path: &str) -> Result<PathBuf, String> {
    if path.is_empty() {
        return Err("lora_path is empty".to_owned());
    }
    let root = root
        .canonicalize()
        .map_err(|e| format!("adapter_root {}: {e}", root.display()))?;
    let dir = root
        .join(path)
        .canonicalize()
        .map_err(|e| format!("lora_path {path}: {e}"))?;
    if !dir.starts_with(&root) {
        return Err(format!(
            "lora_path {path} is outside the adapter root {}",
            root.display()
        ));
    }
    Ok(dir)
}

fn read_file(path: &Path, max: Option<u64>) -> Result<Vec<u8>, String> {
    let len = std::fs::metadata(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    if max.is_some_and(|m| len > m) {
        return Err(format!(
            "{}: larger than {} bytes",
            path.display(),
            max.unwrap_or(0)
        ));
    }
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Reads and parses the adapter directory `path` (confined to `root`).
pub fn read_adapter(
    root: &Path,
    path: &str,
    dims: &Dims,
    max_rank: u32,
) -> Result<(PathBuf, Parsed), String> {
    let dir = resolve(root, path)?;
    let config = read_file(&dir.join(CONFIG_FILE), Some(MAX_CONFIG_BYTES))?;
    let weights = read_file(&dir.join(WEIGHTS_FILE), None)?;
    let parsed = parse_adapter(&config, &weights, dims, max_rank)?;
    Ok((dir, parsed))
}

/// A loaded adapter. Requests hold it (`Arc`) as their lease on its slot.
pub struct Adapter {
    pub name: String,
    /// The executor pool slot (`cere_model_lora_load`, `cere_session_set_lora`).
    pub slot: u32,
    /// [`Parsed::sha256`]: part of the request digest and of the prefix-cache namespace of its requests.
    pub sha256: String,
    pub rank: u32,
    /// The canonical adapter directory.
    pub path: String,
    /// Unix seconds of the load (`/v1/models` `created`).
    pub created: u64,
    /// Kept to load the adapter again into a reloaded model.
    weights: LoraWeights,
}

impl std::fmt::Debug for Adapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Adapter")
            .field("name", &self.name)
            .field("slot", &self.slot)
            .field("sha256", &self.sha256)
            .field("rank", &self.rank)
            .finish_non_exhaustive()
    }
}

/// Why a registry operation was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraError {
    /// 400: a bad name, path or adapter, a duplicate name, a full pool, or an executor refusal of the adapter.
    Invalid(String),
    /// 404: no adapter of that name.
    NotFound(String),
    /// 409: a request still uses the adapter.
    Busy(String),
    /// 503: the native backend is not loaded.
    NotReady,
    /// 500: the executor failed otherwise.
    Internal(String),
}

impl std::fmt::Display for LoraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoraError::Invalid(m)
            | LoraError::NotFound(m)
            | LoraError::Busy(m)
            | LoraError::Internal(m) => f.write_str(m),
            LoraError::NotReady => f.write_str("the native backend is not loaded"),
        }
    }
}

/// The deployment's adapter registry (see the module doc).
pub struct Lora {
    config: LoraConfig,
    /// Serializes every executor-side change: load, unload, and [`Lora::attach`].
    ops: Mutex<Ops>,
    adapters: Mutex<BTreeMap<String, Arc<Adapter>>>,
}

#[derive(Default)]
struct Ops {
    /// The sealed model's dimensions, from the first [`Lora::attach`].
    dims: Option<Dims>,
    /// The `preload` list ran (the first attach); later attaches load the registry again.
    preloaded: bool,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

impl Lora {
    pub fn new(config: LoraConfig) -> Lora {
        Lora {
            config,
            ops: Mutex::default(),
            adapters: Mutex::default(),
        }
    }

    /// The `cere_model_lora_reserve` of the model create.
    pub fn reserve(&self) -> LoraReserve {
        LoraReserve {
            max_adapters: self.config.max_adapters,
            max_rank: self.config.max_rank,
        }
    }

    pub fn max_adapters(&self) -> u32 {
        self.config.max_adapters
    }

    /// A lease on adapter `name`, if loaded.
    pub fn lease(&self, name: &str) -> Option<Arc<Adapter>> {
        lock(&self.adapters).get(name).cloned()
    }

    /// The loaded adapters in name order.
    pub fn list(&self) -> Vec<Arc<Adapter>> {
        lock(&self.adapters).values().cloned().collect()
    }

    /// At a backend load, after the model was sealed: the first time the `preload` list is parsed and loaded, later
    /// (a reloaded model) every registered adapter is loaded again into its slot. An error fails the load.
    pub fn attach(
        &self,
        driver: &Driver,
        model: &Model,
        config: &Qwen4ExpConfig,
    ) -> Result<(), String> {
        let mut ops = lock(&self.ops);
        let dims = Dims::of(config);
        ops.dims = Some(dims);
        if ops.preloaded {
            for a in self.list() {
                driver
                    .lora_load(model, a.slot, &a.weights)
                    .map_err(|e| format!("adapter {}: {e}", a.name))?;
            }
            return Ok(());
        }
        // A failed earlier preload (its model released) leaves nothing behind.
        lock(&self.adapters).clear();
        for p in &self.config.preload {
            let slot = self.free_slot().map_err(|e| e.to_string())?;
            let adapter = self
                .load_into(driver, model, &dims, slot, &p.name, &p.path)
                .map_err(|e| format!("preload adapter {}: {e}", p.name))?;
            lock(&self.adapters).insert(p.name.clone(), adapter);
        }
        ops.preloaded = true;
        Ok(())
    }

    /// `POST /v1/load_lora_adapter`: loads the adapter at `path` as `name` on the `current` backend. `reserved`
    /// are names an adapter may not take (the served models).
    pub fn load(
        &self,
        name: &str,
        path: &str,
        reserved: &[&str],
        current: impl FnOnce() -> Option<Arc<NativeBackend>>,
    ) -> Result<Arc<Adapter>, LoraError> {
        if crate::config::id("lora_name", name).is_err() {
            return Err(LoraError::Invalid(
                "lora_name must match ^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$".to_owned(),
            ));
        }
        if reserved.contains(&name) {
            return Err(LoraError::Invalid(format!(
                "lora_name {name} is a served model"
            )));
        }
        let ops = lock(&self.ops);
        if lock(&self.adapters).contains_key(name) {
            return Err(LoraError::Invalid(format!(
                "the adapter {name} is already loaded"
            )));
        }
        let (Some(dims), Some(backend)) = (ops.dims, current()) else {
            return Err(LoraError::NotReady);
        };
        let slot = self.free_slot()?;
        let adapter = self.load_into(&backend.driver, &backend.model, &dims, slot, name, path)?;
        lock(&self.adapters).insert(name.to_owned(), adapter.clone());
        drop(ops);
        Ok(adapter)
    }

    /// `POST /v1/unload_lora_adapter`: unloads adapter `name`; busy while a request holds it.
    pub fn unload(
        &self,
        name: &str,
        current: impl FnOnce() -> Option<Arc<NativeBackend>>,
    ) -> Result<(), LoraError> {
        let _ops = lock(&self.ops);
        let adapter = {
            let mut adapters = lock(&self.adapters);
            let Some(adapter) = adapters.get(name) else {
                return Err(LoraError::NotFound(format!(
                    "the adapter {name} is not loaded"
                )));
            };
            // Leases are cloned only under this lock: one count is the registry's own.
            if Arc::strong_count(adapter) > 1 {
                return Err(LoraError::Busy(format!(
                    "the adapter {name} is in use by a request"
                )));
            }
            adapters.remove(name).expect("present")
        };
        let reinsert = |e: LoraError| {
            lock(&self.adapters).insert(name.to_owned(), adapter.clone());
            Err(e)
        };
        let Some(backend) = current() else {
            return reinsert(LoraError::NotReady);
        };
        match backend.driver.lora_unload(&backend.model, adapter.slot) {
            Ok(()) => {
                eprintln!(
                    "cered lora: unloaded adapter {name} slot={} sha256={}",
                    adapter.slot, adapter.sha256
                );
                Ok(())
            }
            Err(e) if e.status() == Some(StatusError::Busy) => reinsert(LoraError::Busy(format!(
                "the adapter {name} is in use by a session"
            ))),
            Err(e) => reinsert(LoraError::Internal(format!("unloading {name}: {e}"))),
        }
    }

    /// The lowest slot no adapter holds (the caller holds `ops`).
    fn free_slot(&self) -> Result<u32, LoraError> {
        let adapters = lock(&self.adapters);
        (0..self.config.max_adapters)
            .find(|s| !adapters.values().any(|a| a.slot == *s))
            .ok_or_else(|| {
                LoraError::Invalid(format!(
                    "no free adapter slot: {} adapters are loaded (max_adapters)",
                    adapters.len()
                ))
            })
    }

    /// Reads, parses and loads one adapter into `slot` (the caller holds `ops`), timing each step.
    fn load_into(
        &self,
        driver: &Driver,
        model: &Model,
        dims: &Dims,
        slot: u32,
        name: &str,
        path: &str,
    ) -> Result<Arc<Adapter>, LoraError> {
        let t = Instant::now();
        let (dir, parsed) = read_adapter(
            Path::new(&self.config.adapter_root),
            path,
            dims,
            self.config.max_rank,
        )
        .map_err(LoraError::Invalid)?;
        let parse_ms = t.elapsed().as_millis();
        driver
            .lora_load(model, slot, &parsed.weights)
            .map_err(|e| match &e {
                DriverError::Status {
                    status: StatusError::Argument | StatusError::Unsupported | StatusError::Capacity,
                    ..
                }
                | DriverError::Argument(_)
                | DriverError::Missing(_) => {
                    LoraError::Invalid(format!("the executor refused the adapter: {e}"))
                }
                _ => LoraError::Internal(format!("loading {name}: {e}")),
            })?;
        let rank = parsed.weights.rank;
        eprintln!(
            "cered lora: loaded adapter {name} slot={slot} rank={rank} scale={} targets={} sha256={} parse_ms={parse_ms} \
             total_ms={}",
            parsed.weights.scale,
            parsed.weights.targets.len(),
            parsed.sha256,
            t.elapsed().as_millis()
        );
        Ok(Arc::new(Adapter {
            name: name.to_owned(),
            slot,
            sha256: parsed.sha256,
            rank,
            path: dir.to_string_lossy().into_owned(),
            created: unix_seconds(),
            weights: parsed.weights,
        }))
    }
}

/// The prefix-cache namespace of an adapter request: the base request's namespace mixed with the adapter digest, so
/// no cached state crosses between the base model and an adapter or between two adapters. Base requests keep theirs.
pub fn namespace(base: [u8; 16], adapter_sha256: &str) -> [u8; 16] {
    let mut h = Sha256::new();
    h.update(b"cere-lora-namespace-v1\0");
    h.update(base);
    h.update(adapter_sha256.as_bytes());
    let digest = h.finalize();
    let mut out = [0u8; 16];
    out.copy_from_slice(&digest[..16]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A safetensors file of `(name, dtype, shape, bytes)`.
    pub(crate) fn safetensors(tensors: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        for (name, dtype, shape, bytes) in tensors {
            let end = offset + bytes.len() as u64;
            header.insert(
                (*name).to_owned(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}),
            );
            offset = end;
        }
        let mut text = serde_json::to_vec(&Value::Object(header)).unwrap();
        while text.len() % 8 != 0 {
            text.push(b' ');
        }
        let mut out = (text.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&text);
        for (_, _, _, bytes) in tensors {
            out.extend_from_slice(bytes);
        }
        out
    }

    fn dims() -> Dims {
        Dims {
            hidden: 8,
            heads: 2,
            kv_heads: 1,
            head_dim: 2,
            shared: 4,
            layers: 2,
            full_attention: 0b10,
        }
    }

    fn f32s(n: u64, v: f32) -> Vec<u8> {
        (0..n).flat_map(|_| v.to_le_bytes()).collect()
    }

    fn config(r: u32, alpha: f64, extra: &str) -> Vec<u8> {
        format!(r#"{{"peft_type":"LORA","r":{r},"lora_alpha":{alpha},"bias":"none"{extra}}}"#)
            .into_bytes()
    }

    const P: &str = "base_model.model.model.layers";

    #[test]
    fn parses_attention_and_shared_expert_pairs() {
        let r = 4u64;
        let st = safetensors(&[
            (
                &format!("{P}.1.self_attn.q_proj.lora_A.weight"),
                "F32",
                vec![r, 8],
                f32s(r * 8, 1.0),
            ),
            (
                &format!("{P}.1.self_attn.q_proj.lora_B.weight"),
                "F32",
                vec![8, r],
                f32s(8 * r, 0.5),
            ),
            (
                &format!("{P}.0.mlp.shared_expert.down_proj.lora_A.weight"),
                "BF16",
                vec![r, 4],
                vec![0; (r * 4 * 2) as usize],
            ),
            (
                &format!("{P}.0.mlp.shared_expert.down_proj.lora_B.weight"),
                "F16",
                vec![8, r],
                vec![0; (8 * r * 2) as usize],
            ),
        ]);
        let cfg = config(4, 8.0, "");
        let p = parse_adapter(&cfg, &st, &dims(), 64).unwrap();
        assert_eq!(p.weights.rank, 4);
        assert_eq!(p.weights.scale, 2.0);
        // Sorted by target id: layer 0 down (6), layer 1 q (8).
        assert_eq!(
            p.weights.targets,
            vec![ffi::LORA_SHARED_DOWN, ffi::LORA_MODULES]
        );
        assert_eq!(p.weights.a[1].len(), (r * 8 * 2) as usize);
        assert_eq!(&p.weights.a[1][..2], &f32_to_bf16(1.0).to_le_bytes());
        assert_eq!(&p.weights.b[1][..2], &0x3f00u16.to_le_bytes());
        let again = parse_adapter(&cfg, &st, &dims(), 64).unwrap();
        assert_eq!(again.sha256, p.sha256);
        let other = parse_adapter(&config(4, 4.0, ""), &st, &dims(), 64).unwrap();
        assert_ne!(other.sha256, p.sha256);
        assert_eq!(other.weights.scale, 1.0);
    }

    #[test]
    fn rslora_scales_by_the_root_of_r() {
        let st = safetensors(&[
            (
                &format!("{P}.0.mlp.shared_expert.up_proj.lora_A.weight"),
                "F32",
                vec![16, 8],
                f32s(128, 0.0),
            ),
            (
                &format!("{P}.0.mlp.shared_expert.up_proj.lora_B.weight"),
                "F32",
                vec![4, 16],
                f32s(64, 0.0),
            ),
        ]);
        let p =
            parse_adapter(&config(16, 32.0, r#","use_rslora":true"#), &st, &dims(), 64).unwrap();
        assert_eq!(p.weights.scale, 8.0);
    }

    #[test]
    fn refusals_name_the_reason() {
        let pair = |layer: u32, module: &str, a: Vec<u64>, b: Vec<u64>| {
            let (na, nb) = (a.iter().product::<u64>(), b.iter().product::<u64>());
            safetensors(&[
                (
                    &format!("{P}.{layer}.{module}.lora_A.weight"),
                    "F32",
                    a,
                    f32s(na, 0.0),
                ),
                (
                    &format!("{P}.{layer}.{module}.lora_B.weight"),
                    "F32",
                    b,
                    f32s(nb, 0.0),
                ),
            ])
        };
        let good = pair(1, "self_attn.k_proj", vec![2, 8], vec![2, 2]);
        let cases: Vec<(Vec<u8>, Vec<u8>, &str)> = vec![
            (
                config(2, 4.0, r#","use_dora":true"#),
                good.clone(),
                "use_dora",
            ),
            (
                config(2, 4.0, r#","modules_to_save":["lm_head"]"#),
                good.clone(),
                "modules_to_save",
            ),
            (
                config(2, 4.0, r#","rank_pattern":{"q_proj":8}"#),
                good.clone(),
                "rank_pattern",
            ),
            (
                config(2, 4.0, r#","fan_in_fan_out":true"#),
                good.clone(),
                "fan_in_fan_out",
            ),
            (
                br#"{"peft_type":"LORA","r":2,"lora_alpha":4,"bias":"all"}"#.to_vec(),
                good.clone(),
                "bias",
            ),
            (
                br#"{"peft_type":"IA3","r":2,"lora_alpha":4}"#.to_vec(),
                good.clone(),
                "peft_type",
            ),
            (config(128, 4.0, ""), good.clone(), "r must be"),
            (config(4, 4.0, ""), good.clone(), "expected [4, 8]"),
            (
                config(2, 4.0, ""),
                pair(0, "self_attn.k_proj", vec![2, 8], vec![2, 2]),
                "not a full_attention",
            ),
            (
                config(2, 4.0, ""),
                pair(2, "mlp.shared_expert.up_proj", vec![2, 8], vec![4, 2]),
                "2 layers",
            ),
            (
                config(2, 4.0, ""),
                pair(1, "mlp.experts.0.up_proj", vec![2, 8], vec![4, 2]),
                "unexpected tensor",
            ),
            (
                config(2, 4.0, ""),
                safetensors(&[(
                    &format!("{P}.1.self_attn.v_proj.lora_A.weight"),
                    "F32",
                    vec![2, 8],
                    f32s(16, 0.0),
                )]),
                "lacks its lora_A or lora_B",
            ),
            (
                config(2, 4.0, ""),
                safetensors(&[
                    (
                        &format!("{P}.1.self_attn.v_proj.lora_A.weight"),
                        "F32",
                        vec![2, 8],
                        f32s(16, f32::NAN),
                    ),
                    (
                        &format!("{P}.1.self_attn.v_proj.lora_B.weight"),
                        "F32",
                        vec![2, 2],
                        f32s(4, 0.0),
                    ),
                ]),
                "non-finite",
            ),
            (config(2, 4.0, ""), safetensors(&[]), "no LoRA tensors"),
        ];
        for (cfg, st, want) in cases {
            let err = parse_adapter(&cfg, &st, &dims(), 64).unwrap_err();
            assert!(err.contains(want), "{want}: {err}");
        }
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        // 1 + 2^-8 is halfway between 1.0 and 1 + 2^-7: ties to even (1.0).
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0xc000), -2.0);
    }

    #[test]
    fn tensor_names_need_a_layer_segment() {
        assert_eq!(
            tensor_target("base_model.model.model.layers.3.self_attn.o_proj.lora_B.weight"),
            Some((3, ffi::LORA_O_PROJ, false))
        );
        assert_eq!(
            tensor_target("layers.0.mlp.shared_expert.gate_proj.lora_A.weight"),
            Some((0, ffi::LORA_SHARED_GATE, true))
        );
        assert_eq!(
            tensor_target("model.xlayers.0.self_attn.q_proj.lora_A.weight"),
            None
        );
        assert_eq!(
            tensor_target("model.layers.0.self_attn.q_proj.lora_A.default.weight"),
            None
        );
        assert_eq!(
            tensor_target("model.layers.0.self_attn.q_proj.weight"),
            None
        );
    }

    #[test]
    fn paths_stay_inside_the_root() {
        let base = std::env::temp_dir().join(format!("cere-lora-resolve-{}", std::process::id()));
        let root = base.join("root");
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::create_dir_all(base.join("outside")).unwrap();
        assert!(resolve(&root, "a").is_ok());
        assert!(resolve(&root, root.join("a").to_str().unwrap()).is_ok());
        assert!(resolve(&root, "../outside")
            .unwrap_err()
            .contains("outside the adapter root"));
        assert!(resolve(&root, base.join("outside").to_str().unwrap())
            .unwrap_err()
            .contains("outside the adapter root"));
        assert!(resolve(&root, "missing").is_err());
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn namespaces_differ_per_adapter_and_from_the_base() {
        let base = [7u8; 16];
        let a = namespace(base, "aa");
        assert_ne!(a, base);
        assert_ne!(a, namespace(base, "bb"));
        assert_ne!(a, namespace([0; 16], "aa"));
        assert_eq!(a, namespace(base, "aa"));
    }
}
