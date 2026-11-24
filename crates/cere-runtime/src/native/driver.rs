//! The driver thread (spec m22 R1, Q10): one thread owns the executor handle and makes every host call except
//! `cere_model_put`, which upload threads call directly through a [`Putter`] (Q13). Callers talk to the thread
//! through a command channel; while work is in flight it polls every ticket and sleeps [`IDLE_POLL`] when a round
//! made no progress and no command arrived. Every queued command goes before the next round (BcAdmit, O282), a
//! command waiting after a ticket's poll goes before the next ticket's poll (a poll runs at most one round: none when
//! a round ran since the ticket was last reported, whose results it reports first; S1 G3, beat-sglang design note),
//! and after a command or a ticket's end the thread waits [`COMMAND_LINGER`] for the follow-up command before the
//! next round (round-gap design note RG-L1).
//!
//! The ABI serializes host calls with one exception (`cere_executor.h`, "One executor serializes its host API
//! calls"): puts for different tensors may overlap each other. A read-write gate enforces exactly that: puts share
//! it, every driver-thread call holds it exclusively.
use super::ffi::{
    self, Api, Batch, Completion, CompletionTicket, ExecutionCapabilities, ExecutorOptions,
    Inkling2Config, Library, LogitBias, ModelDesc, PrefixCaptureStats, PrefixStats, Qwen4ExpConfig,
    Qwen4ExpVisionConfig, RawExecutor, RawModel, RawSession, RawSnapshot, SamplingParams,
    SdarConfig, SlotSelection, SnapshotInfo, StatusError, TensorDesc, TokenLogprobs, TokenScore,
    Work,
};
use crate::levers::{Family, LeverSet};
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::fmt;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};
use std::thread::JoinHandle;
use std::time::Duration;

/// Sleep of a poll round that found no progress and no command (Q10).
pub const IDLE_POLL: Duration = Duration::from_micros(200);

/// How long the driver waits, after a command or a ticket's end while tickets are in flight, for the next command
/// before it runs another round (round-gap design note RG-L1). A request's admission is a chain of commands, each
/// sent after the previous one's reply (create the session, set its cache namespace, submit the PREFILL, then submit
/// the DECODE after the PREFILL's terminal poll), and every round that runs between two links delays the request by
/// a whole round. A miss costs at most this long.
pub const COMMAND_LINGER: Duration = Duration::from_millis(1);

/// `cere_executor_options` in owned form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutorConfig {
    pub device_id: String,
    pub worker_epoch: u64,
    pub memory_budget_bytes: u64,
}

/// One row of the model's tensor table (`cere_tensor_desc`); its index is the `tensor_index` of puts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    /// `CERE_DTYPE_*`.
    pub dtype: u32,
    /// `CERE_PLACEMENT_*`.
    pub placement: u32,
    pub shape: Vec<u64>,
    pub bytes: u64,
}

/// `cere_model_desc.config` and the `family` / `model_kind` it implies (SelfTune ST4): a library is built for one
/// family and refuses the other's config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelConfig {
    /// `cere_qwen4_exp_config`, model kind `CERE_MODEL_QWEN4_EXP_NVFP4`.
    Qwen4Exp(Qwen4ExpConfig),
    /// `cere_sdar_config` (DICE-8B), model kind `CERE_MODEL_SDAR_BF16`.
    Sdar(SdarConfig),
    /// `cere_inkling2_config` (Inkling 2 Mini, inkling design note S5), model kind `CERE_MODEL_INKLING2_NVFP4`.
    Inkling2(Inkling2Config),
}

impl ModelConfig {
    /// The lever registry's family of this config (its `calibrate_windows`, SelfTune ST4 S3d D-n).
    pub fn lever_family(&self) -> Family {
        match self {
            ModelConfig::Qwen4Exp(_) => Family::Qwen4Exp,
            ModelConfig::Sdar(_) => Family::Sdar,
            ModelConfig::Inkling2(_) => Family::Inkling2Loop,
        }
    }

    pub fn family(&self) -> u32 {
        self.lever_family().ffi()
    }

    pub fn model_kind(&self) -> u32 {
        match self {
            ModelConfig::Qwen4Exp(_) => ffi::MODEL_QWEN4_EXP_NVFP4,
            ModelConfig::Sdar(_) => ffi::MODEL_SDAR_BF16,
            ModelConfig::Inkling2(_) => ffi::MODEL_INKLING2_NVFP4,
        }
    }

    pub fn vocab_size(&self) -> u32 {
        match self {
            ModelConfig::Qwen4Exp(c) => c.vocab_size,
            ModelConfig::Sdar(c) => c.vocab_size,
            ModelConfig::Inkling2(c) => c.vocab_size,
        }
    }

    /// The Qwen config, `None` on sdar and inkling2_loop (track LoRA's dimensions exist only for the Qwen family).
    pub fn qwen4_exp(&self) -> Option<&Qwen4ExpConfig> {
        match self {
            ModelConfig::Qwen4Exp(c) => Some(c),
            ModelConfig::Sdar(_) | ModelConfig::Inkling2(_) => None,
        }
    }

    /// The bytes of the config struct (`repr(C)`, no padding: both layouts are pinned against the C layout fixture
    /// by `tests/native_ffi.rs`, so every byte is initialized).
    pub fn bytes(&self) -> &[u8] {
        fn of<T>(c: &T) -> &[u8] {
            // SAFETY: `T` is one of the padding-free `repr(C)` config structs (above).
            unsafe {
                std::slice::from_raw_parts((c as *const T).cast::<u8>(), std::mem::size_of::<T>())
            }
        }
        match self {
            ModelConfig::Qwen4Exp(c) => of(c),
            ModelConfig::Sdar(c) => of(c),
            ModelConfig::Inkling2(c) => of(c),
        }
    }

    /// `cere_model_desc.config`: the header types it `const cere_qwen4_exp_config *`; under family sdar or
    /// inkling2_loop it points at that family's config (same leading `struct_size` / `abi_version`, checked by the
    /// library).
    fn desc_ptr(&self) -> *const Qwen4ExpConfig {
        match self {
            ModelConfig::Qwen4Exp(c) => c,
            ModelConfig::Sdar(c) => (c as *const SdarConfig).cast(),
            ModelConfig::Inkling2(c) => (c as *const Inkling2Config).cast(),
        }
    }
}

/// `cere_model_desc` in owned form.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    pub config: ModelConfig,
    /// `cere_model_desc.vision_config` (M2.13 D 112): `None` = text-only (no `model.visual.*` tensor in the desc).
    pub vision: Option<Qwen4ExpVisionConfig>,
    pub tensors: Vec<TensorSpec>,
    pub max_sessions: u64,
    pub max_context_tokens: u64,
    /// KV pool in tokens; 0 = the derived pool (m25 design note D4).
    pub kv_pool_tokens: u64,
    pub stop_token_ids: Vec<u32>,
    /// Not part of the desc: the model's sampling defaults (`generation_config.json`, m27 D10).
    pub generation_defaults: super::sampling::GenerationDefaults,
    /// The pinned host snapshot arena (`cere_model_desc.snapshot_host_bytes`, M2.11 D5); 0 = none.
    pub snapshot_host_bytes: u64,
    /// `cere_model_desc.prefix_state_slots` (M2.10 D6, ledger D 104): device state slots of the prefix cache; 0 =
    /// cache off.
    pub prefix_state_slots: u64,
    /// `cere_model_desc.speculative_draft_tokens` (m29 design note I1): 0 = no draft, 4 = MTP speculation.
    pub speculative_draft_tokens: u32,
    /// `cere_model_desc.draft_token_map` (I2): the artifact's draft token map (strictly increasing ids < vocab);
    /// non-empty exactly when `speculative_draft_tokens` > 0.
    pub draft_token_map: Vec<u32>,
    /// `cere_model_desc.selection` (m23 design note PI-03, ABI v5 slot block): the resolved lever set; the
    /// runtime-layer `drafter` is not a slot (it only sets `cere_model_desc.calibrate_windows`).
    pub levers: LeverSet,
    /// Not part of the desc: `cere_model_set_prefix_end_state(1)` right after `cere_model_create` (m210 design note
    /// D15⁗′ stage 2a, opt-in prefix profiles). Needs `prefix_state_slots > 0` and a library exporting the setter.
    pub prefix_end_state: bool,
    /// Not part of the desc: `cere_model_set_prefix_alt_edges(1)` right after `cere_model_create` (m210 design note
    /// D17, opt-in prefix profiles). Needs `prefix_state_slots > 0` and a library exporting the setter.
    pub prefix_alt_edges: bool,
    /// Not part of the desc: track LoRA's adapter pool, `cere_model_lora_reserve` right after `cere_model_create`
    /// (before seal); `None` = no reserve (no LoRA symbol needed). Needs a library exporting all four LoRA calls.
    pub lora: Option<LoraReserve>,
    /// Not part of the desc: WeightLab's research reserve (`cere_model_research_reserve` right after
    /// `cere_model_create`, before seal, weightlab design note WL-01); `None` = no reserve (no research symbol
    /// needed). Needs a library exporting all six research calls.
    pub research: Option<ResearchReserve>,
}

/// `cere_model_lora_reserve` arguments: slots and the largest rank a slot holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoraReserve {
    pub max_adapters: u32,
    pub max_rank: u32,
}

/// `cere_model_research_reserve` arguments (`[deployments.native.research]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResearchReserve {
    pub device_staging_bytes: u64,
    pub host_buffer_bytes: u64,
    pub max_plans: u32,
}

/// A finished capture as `cere_model_research_capture_read` filled it: pointers into the executor's host ring,
/// valid until `cere_model_research_capture_free` of `capture`. Read it with [`CaptureView::copy`] on the caller's
/// thread (no other call touches those bytes before the free), then free it.
#[derive(Debug)]
pub struct CaptureView(ffi::ResearchCapture);

// SAFETY: the pointers stay valid until the capture is freed, which only the holder of the view asks for.
unsafe impl Send for CaptureView {}

/// One captured tensor, copied out of the ring.
#[derive(Debug, Clone, PartialEq)]
pub struct CapturedTensor {
    pub name: String,
    pub op: u32,
    pub layer: u32,
    /// `CERE_RESEARCH_DTYPE_*`.
    pub dtype: u32,
    pub cols: u32,
    pub positions: Vec<u64>,
    /// Row-major little-endian `[positions.len(), cols]` elements.
    pub data: Vec<u8>,
}

/// A capture copied out of the ring ([`CaptureView::copy`]).
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureData {
    pub tensors: Vec<CapturedTensor>,
    pub bytes: u64,
    pub prefix_lookup_skipped: bool,
    pub plan_sha256: [u8; 32],
}

impl CaptureView {
    /// Copies every tensor out of the ring. A dtype outside `CERE_RESEARCH_DTYPE_*`, a null name or data
    /// pointer with rows, or a name that is not UTF-8 is an argument error (the library broke the contract).
    pub fn copy(&self) -> Result<CaptureData, DriverError> {
        let c = &self.0;
        let bad = |what: &str| DriverError::Argument(format!("capture_read: {what}"));
        let count = usize::try_from(c.tensor_count).map_err(|_| bad("tensor_count"))?;
        if count > 0 && c.tensors.is_null() {
            return Err(bad("null tensors"));
        }
        let mut tensors = Vec::with_capacity(count);
        for i in 0..count {
            // SAFETY: `tensor_count` tensors at `tensors`, valid until the capture is freed.
            let t = unsafe { &*c.tensors.add(i) };
            let elem = match t.dtype {
                ffi::RESEARCH_DTYPE_BF16 => 2usize,
                ffi::RESEARCH_DTYPE_I32 | ffi::RESEARCH_DTYPE_F32 => 4,
                d => return Err(bad(&format!("tensor {i} dtype {d}"))),
            };
            if t.name.is_null() {
                return Err(bad(&format!("tensor {i} has no name")));
            }
            // SAFETY: a NUL-terminated name valid until the free.
            let name = unsafe { std::ffi::CStr::from_ptr(t.name) }
                .to_str()
                .map_err(|_| bad(&format!("tensor {i} name is not UTF-8")))?
                .to_owned();
            let rows = usize::try_from(t.rows).map_err(|_| bad("rows"))?;
            let len = rows
                .checked_mul(t.cols as usize)
                .and_then(|n| n.checked_mul(elem))
                .ok_or_else(|| bad("tensor size"))?;
            if rows > 0 && (t.positions.is_null() || t.data.is_null()) {
                return Err(bad(&format!("tensor {name} has rows but no data")));
            }
            let (positions, data) = if rows == 0 {
                (Vec::new(), Vec::new())
            } else {
                // SAFETY: `rows` positions and `len` data bytes, valid until the free.
                unsafe {
                    (
                        std::slice::from_raw_parts(t.positions, rows).to_vec(),
                        std::slice::from_raw_parts(t.data.cast::<u8>(), len).to_vec(),
                    )
                }
            };
            tensors.push(CapturedTensor {
                name,
                op: t.op,
                layer: t.layer,
                dtype: t.dtype,
                cols: t.cols,
                positions,
                data,
            });
        }
        Ok(CaptureData {
            tensors,
            bytes: c.bytes,
            prefix_lookup_skipped: c.prefix_lookup_skipped != 0,
            plan_sha256: c.plan_sha256,
        })
    }
}

/// One adapter in `cere_model_lora_load` form: `targets[i]` = `layer * LORA_MODULES + module`, `a[i]` BF16
/// `[rank, in]` and `b[i]` BF16 `[out, rank]` row-major little-endian bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraWeights {
    pub rank: u32,
    pub scale: f32,
    pub targets: Vec<u32>,
    pub a: Vec<Vec<u8>>,
    pub b: Vec<Vec<u8>>,
}

/// One image span of a PREFILL prompt (`cere_image_input`, M2.13 ledger N) in owned form: the work item holds
/// the pixel bytes until `cere_executor_submit` returns (the library copies them).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInput {
    /// `[t, h, w]` in 16-px patches.
    pub grid_thw: [u32; 3],
    /// The index in the prompt of the image's first `<|image_pad|>` (`vision_start` is the row before it).
    pub token_offset: u64,
    /// The image's 64-bit digest (SGLang `pad_value`, M2.13 D16).
    pub pad_id: u32,
    /// BF16 `pixel_values` `[t·h·w, 1536]`, row-major little-endian, processor patch order.
    pub pixels: Vec<u8>,
    /// VisionLong C1: SHA-256 of exactly `pixels` (`cere_image_input.pixel_sha256`).
    pub pixel_sha256: [u8; 32],
}

impl ImageInput {
    /// An image span whose `pixel_sha256` is the SHA-256 of exactly `pixels` (VisionLong C1). Computed here, not
    /// taken from the sidecar, whose own digest is of the processor's f32 rows before the BF16 cast.
    pub fn new(grid_thw: [u32; 3], token_offset: u64, pad_id: u32, pixels: Vec<u8>) -> Self {
        use sha2::Digest;
        let pixel_sha256 = sha2::Sha256::digest(&pixels).into();
        Self {
            grid_thw,
            token_offset,
            pad_id,
            pixels,
            pixel_sha256,
        }
    }
}

/// Why a driver call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverError {
    /// Refused before any library call (malformed input on the Rust side).
    Argument(String),
    /// The library returned a non-OK status.
    Status {
        call: &'static str,
        status: StatusError,
    },
    /// The library does not export an optional ABI v4 append (it predates `call`).
    Missing(&'static str),
    /// The driver thread is gone (shut down or panicked).
    Stopped,
}

impl DriverError {
    /// The library status, if the library refused.
    pub fn status(&self) -> Option<StatusError> {
        match self {
            DriverError::Status { status, .. } => Some(*status),
            _ => None,
        }
    }
}

impl fmt::Display for DriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DriverError::Argument(m) => write!(f, "{m}"),
            DriverError::Status { call, status } => write!(f, "{call}: {status}"),
            DriverError::Missing(call) => write!(f, "the library predates {call}"),
            DriverError::Stopped => f.write_str("the executor driver thread has stopped"),
        }
    }
}

impl std::error::Error for DriverError {}

fn check(call: &'static str, code: ffi::Status) -> Result<(), DriverError> {
    StatusError::check(code).map_err(|status| DriverError::Status { call, status })
}

/// The thread that called `cere_model_create` (m32n design note D22 (3)): its nice value and how many CPUs it may
/// run on. Linux keeps both per thread and a new thread starts with its creator's, so every thread the executor
/// starts during ALLOCATE (the host pin, the attached segment's digest check, which run on into UPLOAD) has them too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocateThread {
    pub nice: i32,
    pub cpus: usize,
}

impl AllocateThread {
    /// The calling thread's.
    pub fn current() -> AllocateThread {
        AllocateThread {
            nice: thread_nice(),
            cpus: allowed_cpus().map_or(0, |c| c.len()),
        }
    }
}

/// D22 (3): how a background ALLOCATE runs ([`Driver::create_model_on`]): on a helper thread of the driver thread at
/// nice `nice`, confined to the last `cpus` of the driver thread's allowed CPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Background {
    pub nice: i32,
    pub cpus: usize,
}

/// The calling thread's nice value (Linux: a per-thread attribute).
pub(crate) fn thread_nice() -> i32 {
    // SAFETY: reads the calling thread's own nice value.
    unsafe { libc::getpriority(libc::PRIO_PROCESS, libc::gettid() as libc::id_t) }
}

/// The CPUs the calling thread may run on (Linux: a per-thread attribute), ascending.
pub fn allowed_cpus() -> std::io::Result<Vec<usize>> {
    // SAFETY: an all-zero cpu_set_t is the empty set; sched_getaffinity(0, ..) fills in the calling thread's mask.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    if unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) } != 0
    {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: every index is below CPU_SETSIZE.
    Ok((0..libc::CPU_SETSIZE as usize)
        .filter(|&c| unsafe { libc::CPU_ISSET(c, &set) })
        .collect())
}

/// D22 (3): lowers the calling thread to nice `nice` and confines it to the last `cpus` of its allowed CPUs (all of
/// them when it has fewer). Every thread it starts afterwards inherits both; no other thread changes, and an
/// unprivileged thread could not raise its priority back, so only a thread of its own calls this.
fn into_background(nice: i32, cpus: usize) -> std::io::Result<AllocateThread> {
    // SAFETY: changes only the calling thread's own nice value; lowering a priority needs no privilege.
    if unsafe { libc::setpriority(libc::PRIO_PROCESS, libc::gettid() as libc::id_t, nice) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let allowed = allowed_cpus()?;
    // SAFETY: as in allowed_cpus; every CPU set is one the thread may already run on.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &c in &allowed[allowed.len().saturating_sub(cpus.max(1))..] {
        unsafe { libc::CPU_SET(c, &mut set) };
    }
    // SAFETY: narrows only the calling thread's own mask to a subset of it.
    if unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(AllocateThread::current())
}

/// The terminal reason of a completion (`CERE_FINISH_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    Length,
    StopToken,
    Cancelled,
    Error,
    /// A value the header does not define (including `NONE` on a terminal completion).
    Other(u32),
}

impl Finish {
    fn from_raw(v: u32) -> Finish {
        match v {
            ffi::FINISH_LENGTH => Finish::Length,
            ffi::FINISH_STOP_TOKEN => Finish::StopToken,
            ffi::FINISH_CANCELLED => Finish::Cancelled,
            ffi::FINISH_ERROR => Finish::Error,
            other => Finish::Other(other),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Finish::Length => "length",
            Finish::StopToken => "stop_token",
            Finish::Cancelled => "cancelled",
            Finish::Error => "error",
            Finish::Other(_) => "other",
        }
    }
}

/// Per-item sampling (`cere_sampling_params`, spec m27 I11) in owned form; the ranges are the header's (the
/// library refuses anything else as `CERE_ERR_ARGUMENT`). `top_k` = the vocabulary size means no top-k.
#[derive(Debug, Clone, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: u32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
    pub logit_bias: Vec<LogitBias>,
}

/// One work item (`cere_work`): PREFILL with a prompt or DECODE from the session's last token.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkRequest {
    /// `CERE_OP_*`.
    pub operation: u32,
    pub tokens: Vec<u32>,
    pub max_new_tokens: u64,
    /// Lease an `out_scores` buffer (top-2 per token).
    pub scores: bool,
    pub stop_token_ids: Vec<u32>,
    /// `cere_work.sampling`: `None` is the greedy path (m27 D1).
    pub sampling: Option<Sampling>,
    /// Lease an `out_logprobs` buffer with this many `top` entries per token (`None`: no logprobs).
    pub top_logprobs: Option<u32>,
    /// `cere_work.flags` (`CERE_WORK_*`, ledger table F).
    pub flags: u32,
    /// `cere_work.vocab_mask` of position 0 (spec m28 A1): empty = unconstrained, else ⌈vocab/32⌉ words.
    pub vocab_mask: Vec<u32>,
    /// The prompt's image spans (PREFILL only, M2.13 W 152-160), in prompt order.
    pub images: Vec<ImageInput>,
    /// `cere_work.capture_position` (M2.10 D4 (c), W 132): PREFILL only; 0 = none.
    pub capture_position: u32,
    /// GrammarSpec GS-02: a constrained DECODE whose rounds may verify drafts under the grammar. Each report that
    /// leaves the job awaiting its next mask carries the round's frozen drafts ([`Committed::drafts`]), and the
    /// caller answers with [`Driver::constrain_drafts`]. Needs a non-empty `vocab_mask`.
    pub grammar_drafts: bool,
}

impl WorkRequest {
    pub fn prefill(prompt: Vec<u32>, max_new_tokens: u64) -> WorkRequest {
        WorkRequest {
            operation: ffi::OP_PREFILL,
            tokens: prompt,
            max_new_tokens,
            scores: false,
            stop_token_ids: Vec::new(),
            sampling: None,
            top_logprobs: None,
            flags: 0,
            vocab_mask: Vec::new(),
            images: Vec::new(),
            capture_position: 0,
            grammar_drafts: false,
        }
    }

    pub fn decode(max_new_tokens: u64) -> WorkRequest {
        WorkRequest {
            operation: ffi::OP_DECODE,
            tokens: Vec::new(),
            max_new_tokens,
            scores: false,
            stop_token_ids: Vec::new(),
            sampling: None,
            top_logprobs: None,
            flags: 0,
            vocab_mask: Vec::new(),
            images: Vec::new(),
            capture_position: 0,
            grammar_drafts: false,
        }
    }

    pub fn with_scores(mut self) -> WorkRequest {
        self.scores = true;
        self
    }

    pub fn with_stop_tokens(mut self, ids: Vec<u32>) -> WorkRequest {
        self.stop_token_ids = ids;
        self
    }

    pub fn with_sampling(mut self, sampling: Sampling) -> WorkRequest {
        self.sampling = Some(sampling);
        self
    }

    /// Logprobs of each committed token with `top` alternatives (0..=`CERE_MAX_TOP_LOGPROBS`).
    pub fn with_logprobs(mut self, top: u32) -> WorkRequest {
        self.top_logprobs = Some(top);
        self
    }

    pub fn with_images(mut self, images: Vec<ImageInput>) -> WorkRequest {
        self.images = images;
        self
    }

    pub fn with_flags(mut self, flags: u32) -> WorkRequest {
        self.flags = flags;
        self
    }

    /// A constrained item (spec m28 A7): `mask` is position 0's; it becomes the job's one mask buffer, which
    /// [`Job::mask`] reuses for every later position.
    pub fn with_vocab_mask(mut self, mask: &[u32]) -> WorkRequest {
        self.vocab_mask = mask.to_vec();
        self
    }

    /// A PREFILL checkpoint hint at `position` (M2.10 D4 (c)): a multiple of 64 inside the prompt.
    pub fn with_capture_position(mut self, position: u32) -> WorkRequest {
        self.capture_position = position;
        self
    }

    /// GrammarSpec GS-02: report each awaiting round's frozen drafts ([`WorkRequest::grammar_drafts`]).
    pub fn with_grammar_drafts(mut self) -> WorkRequest {
        self.grammar_drafts = true;
        self
    }
}

/// Receives a pending job's newly committed tokens, in order, as the driver thread polls them (spec m22 R5: SSE).
/// Tokens below `committed_tokens` are final (ABI). The sender is dropped at the terminal poll; the terminal
/// [`JobResult::tokens`] holds every committed token, including any not yet reported.
pub type Progress = tokio::sync::mpsc::UnboundedSender<Committed>;

/// One progress report: the tokens committed since the last one and, when the item leased `out_logprobs`, their
/// logprobs (`out_logprobs` is leased like `out_scores`: its entries below `committed_tokens` are final too).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Committed {
    pub tokens: Vec<u32>,
    /// One per token, or empty when logprobs were not asked for.
    pub logprobs: Vec<TokenLogprobs>,
    /// GrammarSpec GS-02: for a [`WorkRequest::grammar_drafts`] job this report leaves awaiting the mask of the
    /// position after its last token, `cere_executor_round_drafts` of that position (empty: the round drafts
    /// nothing); `None` otherwise.
    pub drafts: Option<Result<RoundDrafts, DriverError>>,
}

/// A round's frozen drafts (`cere_executor_round_drafts`, at most [`ffi::MAX_CONSTRAINED_DRAFTS`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RoundDrafts {
    ids: [u32; ffi::MAX_CONSTRAINED_DRAFTS as usize],
    count: u32,
}

impl RoundDrafts {
    /// The leading `drafts` of a round; `Argument` beyond [`ffi::MAX_CONSTRAINED_DRAFTS`].
    pub fn new(drafts: &[u32]) -> Result<RoundDrafts, DriverError> {
        let mut round = RoundDrafts::default();
        let Some(ids) = round.ids.get_mut(..drafts.len()) else {
            return Err(DriverError::Argument(format!(
                "{} drafts exceed CERE_MAX_CONSTRAINED_DRAFTS ({})",
                drafts.len(),
                ffi::MAX_CONSTRAINED_DRAFTS
            )));
        };
        ids.copy_from_slice(drafts);
        round.count = drafts.len() as u32;
        Ok(round)
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.ids[..self.count as usize]
    }
}

/// A terminal completion with the leased buffers returned.
#[derive(Debug, Clone, PartialEq)]
pub struct JobResult {
    pub ticket: CompletionTicket,
    /// `cere_completion.status`: `Ok` unless the finish is `Error`.
    pub status: Result<(), StatusError>,
    pub finish: Finish,
    pub committed_tokens: u64,
    pub session_revision: u64,
    pub released_bytes: u64,
    /// `cere_completion.cached_tokens` (M2.10 D10): prompt tokens a PREFILL took from the prefix cache.
    pub cached_tokens: u32,
    /// The committed tokens.
    pub tokens: Vec<u32>,
    /// The committed tokens' scores (empty unless asked for).
    pub scores: Vec<TokenScore>,
    /// The committed tokens' logprobs (empty unless asked for).
    pub logprobs: Vec<TokenLogprobs>,
    /// `cere_completion.verify_steps` (m29 design note I5): the ticket's speculate rounds; 0 without speculation.
    pub verify_steps: u64,
    /// `cere_completion.accepted_drafts` (I5): the drafts the ticket's rounds accepted.
    pub accepted_drafts: u64,
}

/// A submitted work item; [`Job::wait`] returns its terminal completion.
#[derive(Debug)]
pub struct Job {
    ticket: CompletionTicket,
    rx: Receiver<Result<JobResult, DriverError>>,
    /// A constrained job's mask buffer (spec m28 A7), shared with the driver thread.
    mask: Option<Arc<Mutex<Vec<u32>>>>,
}

impl Job {
    pub fn ticket(&self) -> CompletionTicket {
        self.ticket
    }

    /// A constrained job's mask buffer: write the next position's mask here, then [`Driver::constrain`]. `None`
    /// for an unconstrained job.
    pub fn mask(&self) -> Option<MutexGuard<'_, Vec<u32>>> {
        self.mask
            .as_ref()
            .map(|m| m.lock().unwrap_or_else(PoisonError::into_inner))
    }

    /// Blocks until the driver thread has polled the terminal completion.
    pub fn wait(self) -> Result<JobResult, DriverError> {
        self.rx.recv().unwrap_or(Err(DriverError::Stopped))
    }
}

/// A raw handle moved between threads; only the driver thread (and puts) dereference it in the library.
struct Raw<T>(NonNull<T>);

impl<T> Raw<T> {
    fn ptr(&self) -> *mut T {
        self.0.as_ptr()
    }
    fn key(&self) -> usize {
        self.0.as_ptr() as usize
    }
}

impl<T> Clone for Raw<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Raw<T> {}

impl<T> fmt::Debug for Raw<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}

// SAFETY: an opaque library handle; the library is told which thread may use it by the ABI's serialization rule,
// which the driver enforces (one driver thread + the put gate).
unsafe impl<T> Send for Raw<T> {}
// SAFETY: as above; `Raw` is only copied, never dereferenced in Rust.
unsafe impl<T> Sync for Raw<T> {}

/// A model created by [`Driver::create_model`]; released by [`Driver::release_model`] or at shutdown.
#[derive(Debug)]
pub struct Model {
    raw: Raw<RawModel>,
    /// The family its library is built for: [`Driver::set_levers`] lists the family's scoped slots (SdarOptIn).
    family: Family,
}

impl Model {
    /// The family its library is built for: it selects the output format (inkling design note S6).
    pub fn family(&self) -> Family {
        self.family
    }
}

/// A session created by [`Driver::create_session`]; released by [`Driver::release_session`] or at shutdown.
#[derive(Debug)]
pub struct Session {
    raw: Raw<RawSession>,
}

/// A snapshot in its model's host arena (M2.11 D1), made by [`Driver::snapshot`] or [`Driver::import_snapshot`];
/// released by [`Driver::release_snapshot`] or at shutdown, before any session and model.
#[derive(Debug)]
pub struct Snapshot {
    raw: Raw<RawSnapshot>,
}

type Reply<T> = SyncSender<Result<T, DriverError>>;

/// A caller's byte buffer lent to the driver thread for one round trip (the caller blocks until the reply).
struct Lent<P>(P, usize);

// SAFETY: the pointer is used only by the driver thread while the lending caller blocks on the reply, so the
// buffer outlives every use and nothing else touches it meanwhile.
unsafe impl<P> Send for Lent<P> {}

enum Command {
    /// ALLOCATE, on the driver thread (None) or on a background helper thread of it (D22 (3)).
    CreateModel(
        Box<ModelSpec>,
        Option<Background>,
        Reply<(Raw<RawModel>, AllocateThread)>,
    ),
    Seal(Raw<RawModel>, Reply<ExecutionCapabilities>),
    Capabilities(Raw<RawModel>, Reply<ExecutionCapabilities>),
    ReleaseModel(Raw<RawModel>, Reply<()>),
    CreateSession(Raw<RawModel>, u64, Reply<Raw<RawSession>>),
    ReleaseSession(Raw<RawSession>, Reply<()>),
    Submit(Raw<RawSession>, WorkRequest, Option<Progress>, Reply<Job>),
    Cancel(CompletionTicket, Reply<()>),
    /// Spec m28 A7: `cere_executor_constrain` of `job` at `position` with the mask in the job's buffer; with
    /// `drafts` (GrammarSpec GS-02), `cere_executor_constrain_drafts` with the buffer's `drafts + 1` masks.
    Constrain {
        job: CompletionTicket,
        position: u64,
        drafts: Option<RoundDrafts>,
        reply: Reply<()>,
    },
    Snapshot(Raw<RawSession>, Reply<Raw<RawSnapshot>>),
    Restore(Raw<RawSession>, Raw<RawSnapshot>, Reply<()>),
    SnapshotInfo(Raw<RawSnapshot>, Reply<SnapshotInfo>),
    ReadSnapshot(Raw<RawSnapshot>, u64, Lent<*mut u8>, Reply<()>),
    ImportSnapshot(Raw<RawModel>, Lent<*const u8>, Reply<Raw<RawSnapshot>>),
    ReleaseSnapshot(Raw<RawSnapshot>, Reply<()>),
    /// m23 design note PI-05: `cere_model_selection`.
    Levers(Raw<RawModel>, Reply<SlotSelection>),
    /// PI-04: `cere_model_set_selection` (`CERE_ERR_BUSY` while a ticket is in flight).
    SetLevers(Raw<RawModel>, Box<SlotSelection>, Reply<()>),
    /// O79 (m210 design note PF1): `cere_executor_set_decode_rounds`.
    SetDecodeRounds(u32, Reply<()>),
    /// MixedPD (mixed-pd design note): `cere_executor_set_prefill_slice`.
    SetPrefillSlice(u32, Reply<()>),
    /// S1 G4 (beat-sglang design note): `cere_executor_set_decode_share`.
    SetDecodeShare(u32, Reply<()>),
    /// BC4 (b) (batch-concurrency design note): `cere_executor_set_prefill_age`.
    SetPrefillAge(u32, Reply<()>),
    /// M2.10 D2: `cere_session_set_cache_namespace`.
    SetCacheNamespace(Raw<RawSession>, [u8; 16], Reply<()>),
    /// M2.10 D11: `cere_model_flush_prefix_cache`.
    FlushPrefixCache(Raw<RawModel>, Reply<()>),
    /// `cere_model_prefix_stats`: the prefix cache's cumulative counters.
    PrefixStats(Raw<RawModel>, Reply<PrefixStats>),
    /// `cere_model_prefix_capture_stats` (PT0a): the capture-skip sites and lost hit tokens.
    PrefixCaptureStats(Raw<RawModel>, Reply<PrefixCaptureStats>),
    /// BC1 leftover: `cere_model_prefill_signature`, the sealed model's prefill selection digest.
    PrefillSignature(Raw<RawModel>, Reply<Option<[u8; 32]>>),
    /// M2.10 D12: `cere_prefix_cache_export` of the lent tokens.
    ExportPrefix(
        Raw<RawModel>,
        [u8; 16],
        Lent<*const u32>,
        Reply<Raw<RawSnapshot>>,
    ),
    /// M2.10 D12: `cere_prefix_cache_import`.
    ImportPrefix(Raw<RawModel>, [u8; 16], Raw<RawSnapshot>, Reply<()>),
    /// Track LoRA: `cere_model_lora_load` of the lent weights into a slot.
    LoraLoad(Raw<RawModel>, u32, Lent<*const LoraWeights>, Reply<()>),
    /// Track LoRA: `cere_model_lora_unload` (BUSY while a live session uses the slot).
    LoraUnload(Raw<RawModel>, u32, Reply<()>),
    /// Track LoRA: `cere_session_set_lora`, before the session's first submit.
    SetLora(Raw<RawSession>, u32, Reply<()>),
    /// WeightLab: `cere_model_research_plan_load` of the lent ops (count) into a slot, with the plan's sha256.
    ResearchPlanLoad(
        Raw<RawModel>,
        u32,
        Lent<*const ffi::ResearchOp>,
        [u8; 32],
        Reply<()>,
    ),
    /// WeightLab: `cere_model_research_plan_unload` (BUSY while a live session uses the slot).
    ResearchPlanUnload(Raw<RawModel>, u32, Reply<()>),
    /// WeightLab: `cere_session_set_research` (slot, prompt_tokens, max_tokens), before the first submit; the
    /// capture id (0 = nothing captured).
    SetResearch(Raw<RawSession>, u32, u64, u64, Reply<u64>),
    /// WeightLab: `cere_model_research_capture_read` (BUSY while its session lives).
    ResearchCaptureRead(Raw<RawModel>, u64, Reply<CaptureView>),
    /// WeightLab: `cere_model_research_capture_free`.
    ResearchCaptureFree(Raw<RawModel>, u64, Reply<()>),
    Shutdown(Reply<()>),
}

impl Command {
    /// RG-L1: a command its caller (or a caller waiting on it) follows with another at once: a session's create or
    /// cache namespace precedes its PREFILL submit, and a session's release may admit a create waiting for its pages.
    /// The driver waits [`COMMAND_LINGER`] after one before the next round.
    fn has_follow_up(&self) -> bool {
        matches!(
            self,
            Command::CreateSession(..)
                | Command::SetCacheNamespace(..)
                | Command::SetLora(..)
                | Command::SetResearch(..)
                | Command::ReleaseSession(..)
        )
    }
}

/// The handle owner. Every method but [`Driver::putter`] is a round trip to the driver thread.
pub struct Driver {
    tx: Option<mpsc::Sender<Command>>,
    thread: Option<JoinHandle<()>>,
    lib: Arc<Library>,
    executor: Raw<RawExecutor>,
    gate: Arc<RwLock<()>>,
}

impl fmt::Debug for Driver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Driver")
            .field("library", &self.lib.path())
            .field("executor", &self.executor)
            .finish()
    }
}

impl Driver {
    /// Spawns the driver thread, which creates the executor (`cere_executor_create`).
    pub fn start(lib: Arc<Library>, config: &ExecutorConfig) -> Result<Driver, DriverError> {
        let device = CString::new(config.device_id.as_str())
            .map_err(|_| DriverError::Argument("device_id contains a NUL byte".to_string()))?;
        let (tx, rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let gate = Arc::new(RwLock::new(()));
        let thread_lib = Arc::clone(&lib);
        let thread_gate = Arc::clone(&gate);
        let epoch = config.worker_epoch;
        let options = ExecutorOptions {
            struct_size: ffi::struct_size::<ExecutorOptions>(),
            abi_version: ffi::ABI_VERSION,
            device_id: ptr::null(),
            worker_epoch: config.worker_epoch,
            memory_budget_bytes: config.memory_budget_bytes,
        };
        let thread = std::thread::Builder::new()
            .name("cere-executor-driver".to_string())
            .spawn(move || {
                let options = ExecutorOptions {
                    device_id: device.as_ptr(),
                    ..options
                };
                let mut out: *mut RawExecutor = ptr::null_mut();
                // SAFETY: valid options (device_id lives across the call) and out pointer.
                let rc = unsafe { (thread_lib.api().executor_create)(&mut out, &options) };
                let created = check("cere_executor_create", rc).and_then(|()| {
                    NonNull::new(out).map(Raw).ok_or(DriverError::Status {
                        call: "cere_executor_create",
                        status: StatusError::Internal,
                    })
                });
                let executor = match created {
                    Ok(e) => e,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok(executor));
                Thread {
                    api: *thread_lib.api(),
                    gate: thread_gate,
                    executor,
                    epoch,
                    models: HashMap::new(),
                    sessions: HashMap::new(),
                    snapshots: HashMap::new(),
                    jobs: Vec::new(),
                }
                .run(rx);
            })
            .map_err(|e| DriverError::Argument(format!("spawning the driver thread: {e}")))?;
        match ready_rx.recv() {
            Ok(Ok(executor)) => Ok(Driver {
                tx: Some(tx),
                thread: Some(thread),
                lib,
                executor,
                gate,
            }),
            Ok(Err(e)) => {
                let _ = thread.join();
                Err(e)
            }
            Err(_) => {
                let _ = thread.join();
                Err(DriverError::Stopped)
            }
        }
    }

    pub fn library(&self) -> &Arc<Library> {
        &self.lib
    }

    fn call<T>(&self, make: impl FnOnce(Reply<T>) -> Command) -> Result<T, DriverError> {
        let (reply, rx) = mpsc::sync_channel(1);
        self.tx
            .as_ref()
            .ok_or(DriverError::Stopped)?
            .send(make(reply))
            .map_err(|_| DriverError::Stopped)?;
        rx.recv().unwrap_or(Err(DriverError::Stopped))
    }

    /// ALLOCATE (`cere_model_create`).
    pub fn create_model(&self, spec: &ModelSpec) -> Result<Model, DriverError> {
        self.create_model_on(spec, None).map(|(model, _)| model)
    }

    /// ALLOCATE, and the thread it ran on. `None`: on the driver thread. m32n design note D22 (3), `Some`: the
    /// ALLOCATE of a background load (a replica loading while the other card serves); `cere_model_create` runs on a
    /// helper thread of the driver thread that first lowers itself to nice [`Background::nice`] and confines itself to
    /// the last [`Background::cpus`] of its allowed CPUs, so the threads the executor starts there inherit both. The
    /// driver thread waits for it (still the executor's one serialized caller) and keeps its own priority and CPUs for
    /// the requests it serves afterwards.
    pub fn create_model_on(
        &self,
        spec: &ModelSpec,
        background: Option<Background>,
    ) -> Result<(Model, AllocateThread), DriverError> {
        let (raw, thread) =
            self.call(|r| Command::CreateModel(Box::new(spec.clone()), background, r))?;
        Ok((
            Model {
                raw,
                family: spec.config.lever_family(),
            },
            thread,
        ))
    }

    /// A put handle for upload threads (Q13); its borrow of `model` ends before [`Driver::seal`] can run.
    pub fn putter<'a>(&'a self, model: &'a Model) -> Putter<'a> {
        Putter {
            api: self.lib.api(),
            gate: &self.gate,
            executor: self.executor,
            model: model.raw,
            _model: PhantomData,
        }
    }

    /// INSTANTIATE (`cere_model_seal`), then the model's capabilities.
    pub fn seal(&self, model: &mut Model) -> Result<ExecutionCapabilities, DriverError> {
        self.call(|r| Command::Seal(model.raw, r))
    }

    pub fn capabilities(&self, model: &Model) -> Result<ExecutionCapabilities, DriverError> {
        self.call(|r| Command::Capabilities(model.raw, r))
    }

    /// `cere_model_release`; on an error the model stays tracked and is released at shutdown.
    pub fn release_model(&self, model: Model) -> Result<(), DriverError> {
        self.call(|r| Command::ReleaseModel(model.raw, r))
    }

    pub fn create_session(
        &self,
        model: &Model,
        max_context_tokens: u64,
    ) -> Result<Session, DriverError> {
        let raw = self.call(|r| Command::CreateSession(model.raw, max_context_tokens, r))?;
        Ok(Session { raw })
    }

    /// `cere_session_release`; on an error the session stays tracked and is released at shutdown.
    pub fn release_session(&self, session: Session) -> Result<(), DriverError> {
        self.call(|r| Command::ReleaseSession(session.raw, r))
    }

    /// Submits one work item on `session` with the session's current revision; the driver thread polls it.
    pub fn submit(&self, session: &Session, work: WorkRequest) -> Result<Job, DriverError> {
        self.call(|r| Command::Submit(session.raw, work, None, r))
    }

    /// [`Driver::submit`] that also sends the committed tokens to `progress`, in order and each once: while the job
    /// is pending and, before the job's reply, those its terminal poll committed.
    pub fn submit_with_progress(
        &self,
        session: &Session,
        work: WorkRequest,
        progress: Progress,
    ) -> Result<Job, DriverError> {
        self.call(|r| Command::Submit(session.raw, work, Some(progress), r))
    }

    /// `cere_executor_cancel`: intent; the job still ends through its terminal poll. A job awaiting its mask is
    /// polled again, so it reaches `CERE_FINISH_CANCELLED` (spec m28 A7).
    pub fn cancel(&self, ticket: CompletionTicket) -> Result<(), DriverError> {
        self.call(|r| Command::Cancel(ticket, r))
    }

    /// `cere_executor_constrain` (spec m28 A2, A7): the mask of `job`'s `position` (its committed count) is the one
    /// in [`Job::mask`]; the driver polls the job again. A job that already ended needs no mask: `Ok`.
    pub fn constrain(&self, job: &Job, position: u64) -> Result<(), DriverError> {
        let ticket = job.ticket;
        self.call(|reply| Command::Constrain {
            job: ticket,
            position,
            drafts: None,
            reply,
        })
    }

    /// `cere_executor_constrain_drafts` (GrammarSpec GS-02): `drafts` is the leading part of the frozen drafts
    /// [`Committed::drafts`] reported for `job`'s `position` that the grammar accepts (possibly none: the round
    /// counts the rest as rejected); [`Job::mask`] holds `drafts.len() + 1` masks, that of `position` first and
    /// then the mask after each draft. A job that already ended needs no mask: `Ok`.
    pub fn constrain_drafts(
        &self,
        job: &Job,
        position: u64,
        drafts: &[u32],
    ) -> Result<(), DriverError> {
        let ticket = job.ticket;
        let drafts = Some(RoundDrafts::new(drafts)?);
        self.call(|reply| Command::Constrain {
            job: ticket,
            position,
            drafts,
            reply,
        })
    }

    /// `cere_session_snapshot` of an idle session at its current revision (M2.11 D1, D6): the session stays valid.
    pub fn snapshot(&self, session: &Session) -> Result<Snapshot, DriverError> {
        let raw = self.call(|r| Command::Snapshot(session.raw, r))?;
        Ok(Snapshot { raw })
    }

    /// `cere_session_restore` into a fresh session of the snapshot's model (D6); its revision stays 0.
    pub fn restore(&self, session: &Session, snapshot: &Snapshot) -> Result<(), DriverError> {
        self.call(|r| Command::Restore(session.raw, snapshot.raw, r))
    }

    /// `cere_snapshot_get_info`.
    pub fn snapshot_info(&self, snapshot: &Snapshot) -> Result<SnapshotInfo, DriverError> {
        self.call(|r| Command::SnapshotInfo(snapshot.raw, r))
    }

    /// `cere_snapshot_read`: `dst.len()` bytes of the snapshot image from `offset`.
    pub fn read_snapshot(
        &self,
        snapshot: &Snapshot,
        offset: u64,
        dst: &mut [u8],
    ) -> Result<(), DriverError> {
        let lent = Lent(dst.as_mut_ptr(), dst.len());
        self.call(|r| Command::ReadSnapshot(snapshot.raw, offset, lent, r))
    }

    /// `cere_snapshot_import`: a `CERESNP1` image into `model`'s arena (validated by the library, D14).
    pub fn import_snapshot(&self, model: &Model, bytes: &[u8]) -> Result<Snapshot, DriverError> {
        let lent = Lent(bytes.as_ptr(), bytes.len());
        let raw = self.call(|r| Command::ImportSnapshot(model.raw, lent, r))?;
        Ok(Snapshot { raw })
    }

    /// `cere_snapshot_release`; on an error the snapshot stays tracked and is released at shutdown.
    pub fn release_snapshot(&self, snapshot: Snapshot) -> Result<(), DriverError> {
        self.call(|r| Command::ReleaseSnapshot(snapshot.raw, r))
    }

    /// `cere_model_selection` (m23 design note PI-05): the model's active selection as the library holds it.
    pub fn levers(&self, model: &Model) -> Result<SlotSelection, DriverError> {
        self.call(|r| Command::Levers(model.raw, r))
    }

    /// `cere_model_set_selection` (PI-04): the executor-layer part of `levers` between requests. A ticket in flight
    /// is `DriverError::Status` with [`StatusError::Busy`]; on any error the active set is unchanged.
    pub fn set_levers(&self, model: &Model, levers: &LeverSet) -> Result<(), DriverError> {
        let raw = Box::new(levers.ffi(model.family));
        self.call(|r| Command::SetLevers(model.raw, raw, r))
    }

    /// `cere_executor_set_decode_rounds` (O79, m210 design note PF1): decode rounds per prefill chunk while
    /// decode rows are ready (1..=[`ffi::MAX_DECODE_ROUNDS_PER_CHUNK`], else `Status` ARGUMENT).
    /// [`DriverError::Missing`] when the library predates the call.
    pub fn set_decode_rounds(&self, rounds: u32) -> Result<(), DriverError> {
        self.call(|r| Command::SetDecodeRounds(rounds, r))
    }

    /// `cere_executor_set_prefill_slice` (MixedPD, mixed-pd design note): layer-sliced prefill, ms per segment
    /// (0 = off, ..=[`ffi::MAX_PREFILL_SLICE_MS`], else `Status` ARGUMENT); only models created after it slice.
    /// [`DriverError::Missing`] when the library predates the call.
    pub fn set_prefill_slice(&self, slice_ms: u32) -> Result<(), DriverError> {
        self.call(|r| Command::SetPrefillSlice(slice_ms, r))
    }

    /// `cere_executor_set_decode_share` (S1 G4, beat-sglang design note): while 2+ prompts wait they prefill back to
    /// back; with one left, decode rows get `share_pct` % of its prefill wall time (0 = off, keeps the decode rounds
    /// per chunk; ..=[`ffi::MAX_DECODE_SHARE_PCT`], else `Status` ARGUMENT); takes effect at the next round.
    /// [`DriverError::Missing`] when the library predates the call.
    pub fn set_decode_share(&self, share_pct: u32) -> Result<(), DriverError> {
        self.call(|r| Command::SetDecodeShare(share_pct, r))
    }

    /// `cere_executor_set_prefill_age` (BC4 (b), batch-concurrency design note): without a decode share, prefill
    /// rounds take the prompt with the fewest tokens left; one overtaken `calls` times by younger prompts' prefill
    /// calls runs next (1..=[`ffi::MAX_PREFILL_AGE_CALLS`], else `Status` ARGUMENT; default
    /// [`ffi::DEFAULT_PREFILL_AGE_CALLS`]); takes effect at the next prefill round.
    /// [`DriverError::Missing`] when the library predates the call.
    pub fn set_prefill_age(&self, calls: u32) -> Result<(), DriverError> {
        self.call(|r| Command::SetPrefillAge(calls, r))
    }

    /// `cere_session_set_cache_namespace` (M2.10 D2): before the session's first submit (after it the library
    /// refuses ARGUMENT); a model with `prefix_state_slots` 0 refuses UNSUPPORTED.
    pub fn set_cache_namespace(
        &self,
        session: &Session,
        namespace: [u8; 16],
    ) -> Result<(), DriverError> {
        self.call(|r| Command::SetCacheNamespace(session.raw, namespace, r))
    }

    /// `cere_model_flush_prefix_cache` (M2.10 D11): drops every cached prefix; BUSY while a ticket of the model is
    /// in flight or a session holds a hit (the caller retries), UNSUPPORTED with `prefix_state_slots` 0.
    pub fn flush_prefix_cache(&self, model: &Model) -> Result<(), DriverError> {
        self.call(|r| Command::FlushPrefixCache(model.raw, r))
    }

    /// `cere_model_prefix_stats`: the model's prefix cache counters (cumulative since the model's creation, a flush
    /// keeps them), read on the driver thread like every host call, so it is safe with tickets in flight.
    /// [`DriverError::Missing`] when the library predates the call, `Status` UNSUPPORTED with
    /// `prefix_state_slots` 0.
    pub fn prefix_stats(&self, model: &Model) -> Result<PrefixStats, DriverError> {
        self.call(|r| Command::PrefixStats(model.raw, r))
    }

    /// `cere_model_prefix_capture_stats` (prefix-token design note PT0a): `capture_skipped` by site and the
    /// drafting lookups' lost hit tokens, read like [`Self::prefix_stats`]. [`DriverError::Missing`] when the library
    /// predates the call, `Status` UNSUPPORTED with `prefix_state_slots` 0.
    pub fn prefix_capture_stats(&self, model: &Model) -> Result<PrefixCaptureStats, DriverError> {
        self.call(|r| Command::PrefixCaptureStats(model.raw, r))
    }

    /// `cere_model_prefill_signature` (BC1 leftover, batch-concurrency design note "Prefill selection
    /// signature"): the SHA-256 of the sealed model's prefill kernel selection. `None` when the library predates the
    /// call or its device has no selection (`Status` UNSUPPORTED, e.g. the toy without one); any other status is an
    /// error.
    pub fn prefill_signature(&self, model: &Model) -> Result<Option<[u8; 32]>, DriverError> {
        self.call(|r| Command::PrefillSignature(model.raw, r))
    }

    /// `cere_prefix_cache_export` (M2.10 D12): the deepest cached state of `tokens` under `namespace` as a
    /// snapshot in the model's arena (no cached prefix: ARGUMENT; in flight: BUSY).
    pub fn export_prefix(
        &self,
        model: &Model,
        namespace: [u8; 16],
        tokens: &[u32],
    ) -> Result<Snapshot, DriverError> {
        let lent = Lent(tokens.as_ptr(), tokens.len());
        let raw = self.call(|r| Command::ExportPrefix(model.raw, namespace, lent, r))?;
        Ok(Snapshot { raw })
    }

    /// `cere_prefix_cache_import` (M2.10 D12): inserts an exported snapshot under `namespace` (in flight: BUSY).
    pub fn import_prefix(
        &self,
        model: &Model,
        namespace: [u8; 16],
        snapshot: &Snapshot,
    ) -> Result<(), DriverError> {
        self.call(|r| Command::ImportPrefix(model.raw, namespace, snapshot.raw, r))
    }

    /// `cere_model_lora_load` (track LoRA): copies `weights` into the free `slot` of a sealed model with a reserve.
    /// [`DriverError::Missing`] when the library predates the call.
    pub fn lora_load(
        &self,
        model: &Model,
        slot: u32,
        weights: &LoraWeights,
    ) -> Result<(), DriverError> {
        let lent = Lent(weights as *const LoraWeights, 1);
        self.call(|r| Command::LoraLoad(model.raw, slot, lent, r))
    }

    /// `cere_model_lora_unload` (track LoRA): BUSY while a live session uses the slot.
    pub fn lora_unload(&self, model: &Model, slot: u32) -> Result<(), DriverError> {
        self.call(|r| Command::LoraUnload(model.raw, slot, r))
    }

    /// `cere_session_set_lora` (track LoRA): binds the session to a loaded slot before its first submit.
    pub fn set_lora(&self, session: &Session, slot: u32) -> Result<(), DriverError> {
        self.call(|r| Command::SetLora(session.raw, slot, r))
    }

    /// `cere_model_research_plan_load` (WeightLab): validates and copies `ops` (every pointer read before the
    /// return) into the free `slot` of a sealed model with a research reserve. [`DriverError::Missing`] when the
    /// library predates the call.
    pub fn research_plan_load(
        &self,
        model: &Model,
        slot: u32,
        ops: &[ffi::ResearchOp],
        sha256: [u8; 32],
    ) -> Result<(), DriverError> {
        let lent = Lent(ops.as_ptr(), ops.len());
        self.call(|r| Command::ResearchPlanLoad(model.raw, slot, lent, sha256, r))
    }

    /// `cere_model_research_plan_unload` (WeightLab): BUSY while a live session uses the slot.
    pub fn research_plan_unload(&self, model: &Model, slot: u32) -> Result<(), DriverError> {
        self.call(|r| Command::ResearchPlanUnload(model.raw, slot, r))
    }

    /// `cere_session_set_research` (WeightLab): binds the session to plan `slot` before its first submit; the
    /// capture id (> 0), or 0 when the plan captures nothing. CAPACITY when the host ring cannot hold the bound.
    pub fn set_research(
        &self,
        session: &Session,
        slot: u32,
        prompt_tokens: u64,
        max_tokens: u64,
    ) -> Result<u64, DriverError> {
        self.call(|r| Command::SetResearch(session.raw, slot, prompt_tokens, max_tokens, r))
    }

    /// `cere_model_research_capture_read` (WeightLab): the finished capture's view (its session released).
    pub fn research_capture_read(
        &self,
        model: &Model,
        capture: u64,
    ) -> Result<CaptureView, DriverError> {
        self.call(|r| Command::ResearchCaptureRead(model.raw, capture, r))
    }

    /// `cere_model_research_capture_free` (WeightLab): returns the capture's ring bytes; its view is dead after.
    pub fn research_capture_free(&self, model: &Model, capture: u64) -> Result<(), DriverError> {
        self.call(|r| Command::ResearchCaptureFree(model.raw, capture, r))
    }

    /// Cancels and drains in-flight work, releases every session and model, destroys the executor and joins the
    /// thread. Returns the first error of those calls.
    pub fn shutdown(mut self) -> Result<(), DriverError> {
        self.stop()
    }

    fn stop(&mut self) -> Result<(), DriverError> {
        let Some(tx) = self.tx.take() else {
            return Ok(());
        };
        let (reply, rx) = mpsc::sync_channel(1);
        let result = match tx.send(Command::Shutdown(reply)) {
            Ok(()) => rx.recv().unwrap_or(Err(DriverError::Stopped)),
            Err(_) => Err(DriverError::Stopped),
        };
        drop(tx);
        if let Some(t) = self.thread.take() {
            if t.join().is_err() {
                return Err(DriverError::Stopped);
            }
        }
        result
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Direct `cere_model_put` from any thread (Q13). Puts share the gate with each other and exclude driver-thread
/// calls; the borrow keeps the executor and the model alive and unsealed while it exists.
pub struct Putter<'a> {
    api: &'a Api,
    gate: &'a RwLock<()>,
    executor: Raw<RawExecutor>,
    model: Raw<RawModel>,
    _model: PhantomData<&'a Model>,
}

impl<'a> Putter<'a> {
    /// Copies `data` to tensor `tensor_index` at byte `offset` (returns after the copy).
    pub fn put(&self, tensor_index: u64, offset: u64, data: &[u8]) -> Result<(), DriverError> {
        let _shared = self.gate.read().unwrap_or_else(|e| e.into_inner());
        // SAFETY: live executor and model (borrowed from the driver); `data` is valid for `len` bytes during the
        // call and the library copies before return.
        let rc = unsafe {
            (self.api.model_put)(
                self.executor.ptr(),
                self.model.ptr(),
                tensor_index,
                offset,
                data.as_ptr().cast::<c_void>(),
                data.len() as u64,
            )
        };
        check("cere_model_put", rc)
    }

    /// `cere_model_host_resident` (m216 design note L4): covers HOST_PINNED tensor `tensor_index` (`bytes` long)
    /// without a put and returns its bytes in the attached shared segment; `None` when the model's host region is
    /// private or created by this process (`CERE_ERR_UNSUPPORTED`: put the bytes instead).
    pub fn host_resident(
        &self,
        tensor_index: u64,
        bytes: u64,
    ) -> Result<Option<&'a [u8]>, DriverError> {
        let _shared = self.gate.read().unwrap_or_else(|e| e.into_inner());
        let mut out: *const c_void = std::ptr::null();
        // SAFETY: live executor and model (borrowed from the driver); `out` is a valid out-pointer.
        let rc = unsafe {
            (self.api.model_host_resident)(
                self.executor.ptr(),
                self.model.ptr(),
                tensor_index,
                &mut out,
            )
        };
        match StatusError::check(rc) {
            Err(StatusError::Unsupported) => Ok(None),
            Err(status) => Err(DriverError::Status {
                call: "cere_model_host_resident",
                status,
            }),
            // SAFETY: on OK the library returns `bytes` read-only bytes of the tensor, mapped until the model is
            // released, which the `'a` borrow of the model outlives.
            Ok(()) if !out.is_null() => Ok(Some(unsafe {
                std::slice::from_raw_parts(out.cast::<u8>(), bytes as usize)
            })),
            Ok(()) => Err(DriverError::Status {
                call: "cere_model_host_resident",
                status: StatusError::Internal,
            }),
        }
    }

    /// `cere_executor_register_staging` (m216 design note M216S-11): the library pins `[base, base + len)` and
    /// copies DEVICE puts whose bytes lie inside it straight from it. `Ok(false)` when the library predates put
    /// staging or its device has no direct path (`CERE_ERR_UNSUPPORTED`, nothing kept): its puts keep the bounce
    /// path. Holds the gate exclusively, like a driver-thread call.
    ///
    /// # Safety
    /// The range stays allocated (and is not freed or remapped) until [`Putter::unregister_staging`] of `base`
    /// returns `Ok`.
    pub unsafe fn register_staging(
        &self,
        base: *const u8,
        len: usize,
    ) -> Result<bool, DriverError> {
        // Only with both symbols: a range registered without a way to unregister it would keep the executor busy.
        let (Some(register), Some(_)) = (
            self.api.executor_register_staging,
            self.api.executor_unregister_staging,
        ) else {
            return Ok(false);
        };
        let _excl = self.gate.write().unwrap_or_else(|e| e.into_inner());
        // SAFETY: a live executor; the caller keeps the range allocated while it is registered.
        let rc = unsafe { register(self.executor.ptr(), base.cast::<c_void>(), len as u64) };
        match StatusError::check(rc) {
            Ok(()) => Ok(true),
            Err(StatusError::Unsupported) => Ok(false),
            Err(status) => Err(DriverError::Status {
                call: "cere_executor_register_staging",
                status,
            }),
        }
    }

    /// `cere_executor_unregister_staging` of a range [`Putter::register_staging`] registered at `base`.
    pub fn unregister_staging(&self, base: *const u8) -> Result<(), DriverError> {
        let Some(unregister) = self.api.executor_unregister_staging else {
            return Ok(());
        };
        let _excl = self.gate.write().unwrap_or_else(|e| e.into_inner());
        // SAFETY: a live executor; the library only looks the pointer up among its registered ranges.
        let rc = unsafe { unregister(self.executor.ptr(), base.cast::<c_void>()) };
        check("cere_executor_unregister_staging", rc)
    }
}

struct ModelState {
    /// `profile_generation` once sealed.
    generation: Option<u64>,
}

struct SessionState {
    model: usize,
    /// The revision after the session's last terminal completion.
    revision: u64,
}

/// One [`Thread::poll_jobs`] pass: whether any job progressed or ended, whether one ended, and the command that ended
/// the pass early (RG-L1).
#[derive(Default)]
struct Pass {
    progressed: bool,
    ended: bool,
    command: Option<Command>,
}

/// A submitted item whose buffers are leased to the executor until its terminal poll.
struct InFlight {
    ticket: CompletionTicket,
    session: usize,
    out_tokens: Vec<u32>,
    out_scores: Vec<TokenScore>,
    out_logprobs: Vec<TokenLogprobs>,
    reply: SyncSender<Result<JobResult, DriverError>>,
    progress: Option<Progress>,
    /// Committed tokens already sent to `progress`.
    reported: usize,
    /// A constrained job's mask buffer (spec m28 A7), shared with its [`Job`].
    mask: Option<Arc<Mutex<Vec<u32>>>>,
    /// The words of one mask (the submit's `vocab_mask`).
    mask_words: usize,
    /// The positions below this one have a mask (position 0's came with the submit).
    masked: u64,
    /// The job committed every masked position and waits for the next mask: `poll_jobs` skips it until a
    /// `Constrain` or `Cancel` (spec m28 A7, D8).
    awaiting_mask: bool,
    /// [`WorkRequest::grammar_drafts`].
    grammar_drafts: bool,
}

impl InFlight {
    /// Sends the tokens (and logprobs) committed since the last report, with `drafts` ([`Committed::drafts`]), to
    /// `progress` (a closed receiver stops the reports).
    fn report(&mut self, committed: u64, drafts: Option<Result<RoundDrafts, DriverError>>) {
        let Some(progress) = &self.progress else {
            return;
        };
        let committed = (committed as usize).min(self.out_tokens.len());
        if committed <= self.reported {
            return;
        }
        // The library publishes `committed_tokens` with release ordering; entries below it are final.
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        // SAFETY: `out_tokens` and `out_logprobs` (when leased, as long as `out_tokens`) are leased to the executor,
        // which only writes entries at or above `committed`; the entries read here are final and never written
        // again, so the copies do not race.
        let tokens = unsafe {
            std::slice::from_raw_parts(
                self.out_tokens.as_ptr().add(self.reported),
                committed - self.reported,
            )
        }
        .to_vec();
        let logprobs = if self.out_logprobs.is_empty() {
            Vec::new()
        } else {
            // SAFETY: as above.
            unsafe {
                std::slice::from_raw_parts(
                    self.out_logprobs.as_ptr().add(self.reported),
                    committed - self.reported,
                )
            }
            .to_vec()
        };
        if progress
            .send(Committed {
                tokens,
                logprobs,
                drafts,
            })
            .is_err()
        {
            self.progress = None;
        }
        self.reported = committed;
    }
}

struct Thread {
    api: Api,
    gate: Arc<RwLock<()>>,
    executor: Raw<RawExecutor>,
    epoch: u64,
    models: HashMap<usize, ModelState>,
    sessions: HashMap<usize, SessionState>,
    /// Live snapshots, each with its model's key.
    snapshots: HashMap<usize, usize>,
    jobs: Vec<InFlight>,
}

fn completion_header() -> Completion {
    Completion {
        struct_size: ffi::struct_size::<Completion>(),
        abi_version: ffi::ABI_VERSION,
        ..Completion::default()
    }
}

fn capabilities_header() -> ExecutionCapabilities {
    ExecutionCapabilities {
        struct_size: ffi::struct_size::<ExecutionCapabilities>(),
        abi_version: ffi::ABI_VERSION,
        ..ExecutionCapabilities::default()
    }
}

/// The driver thread's in-flight tickets as [`next_command`] sees them; the tests stand a fake in for [`Thread`].
trait Tickets {
    /// No ticket is in flight.
    fn none(&self) -> bool;
    /// Every in-flight ticket (vacuously, none) awaits a mask, which only a command can bring.
    fn all_awaiting_mask(&self) -> bool;
    /// One [`Thread::poll_jobs`] pass (each poll runs at most one round) that a waiting command ends early.
    fn poll(&mut self, rx: &Receiver<Command>) -> Pass;
}

impl Tickets for Thread {
    fn none(&self) -> bool {
        self.jobs.is_empty()
    }

    fn all_awaiting_mask(&self) -> bool {
        self.jobs.iter().all(|j| j.awaiting_mask)
    }

    fn poll(&mut self, rx: &Receiver<Command>) -> Pass {
        self.poll_jobs(Some(rx))
    }
}

/// The driver thread's next command, running rounds until one comes; `None` once every sender is gone. `linger` is
/// set by the caller after a command with a follow-up and by a pass in which a ticket ended.
fn next_command(
    tickets: &mut impl Tickets,
    rx: &Receiver<Command>,
    linger: &mut bool,
) -> Option<Command> {
    loop {
        // RG-L1: after a command or a ticket's end, the next command (if it comes within COMMAND_LINGER) goes before
        // the next round.
        if std::mem::take(linger) && !tickets.none() {
            match rx.recv_timeout(COMMAND_LINGER) {
                Ok(cmd) => return Some(cmd),
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return None,
            }
        }
        // BcAdmit (O282, batch-concurrency design note): every command already queued goes before the next round,
        // whatever its kind. Before, only a follow-up command was waited for: after a Submit the next round ran first
        // (`poll_jobs` takes a command only after a poll), so a burst of N requests whose links queued together got
        // its sessions, namespaces and PREFILL submits one round (a whole prefill call or decode round) apart, and
        // the executor never held more than one prompt to prefill (BC6 r16 capture). The drain is bounded: each
        // caller blocks on its reply, and a chain's later links (a DECODE after its PREFILL, a constrain after a
        // report) wait for a round, so commands cannot keep the rounds from running.
        match rx.try_recv() {
            Ok(cmd) => return Some(cmd),
            Err(TryRecvError::Disconnected) => return None,
            Err(TryRecvError::Empty) => {}
        }
        let pass = tickets.poll(rx);
        *linger = pass.ended;
        match pass.command.map_or_else(|| rx.try_recv(), Ok) {
            Ok(cmd) => return Some(cmd),
            Err(TryRecvError::Disconnected) => return None,
            Err(TryRecvError::Empty) if pass.progressed => {}
            // Nothing to poll: every job (if any) awaits a mask, which only a command can bring.
            Err(TryRecvError::Empty) if tickets.all_awaiting_mask() => return rx.recv().ok(),
            Err(TryRecvError::Empty) => match rx.recv_timeout(IDLE_POLL) {
                Ok(cmd) => return Some(cmd),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return None,
            },
        }
    }
}

impl Thread {
    fn run(mut self, rx: Receiver<Command>) {
        let mut linger = false;
        loop {
            match next_command(&mut self, &rx, &mut linger) {
                Some(Command::Shutdown(reply)) => {
                    let _ = reply.send(self.shutdown());
                    return;
                }
                Some(cmd) => {
                    linger = cmd.has_follow_up();
                    self.handle(cmd);
                }
                None => {
                    let _ = self.shutdown();
                    return;
                }
            }
        }
    }

    /// Runs `f` holding the gate exclusively (no put overlaps a driver-thread call).
    fn exclusive<R>(&self, f: impl FnOnce(&Api) -> R) -> R {
        let _excl = self.gate.write().unwrap_or_else(|e| e.into_inner());
        f(&self.api)
    }

    fn handle(&mut self, cmd: Command) {
        match cmd {
            Command::CreateModel(spec, background, reply) => {
                let _ = reply.send(self.create_model(spec, background));
            }
            Command::Seal(model, reply) => {
                let _ = reply.send(self.seal(model));
            }
            Command::Capabilities(model, reply) => {
                let _ = reply.send(self.capabilities(model));
            }
            Command::ReleaseModel(model, reply) => {
                let _ = reply.send(self.release_model(model));
            }
            Command::CreateSession(model, max_context, reply) => {
                let _ = reply.send(self.create_session(model, max_context));
            }
            Command::ReleaseSession(session, reply) => {
                let _ = reply.send(self.release_session(session));
            }
            Command::Submit(session, work, progress, reply) => {
                let _ = reply.send(self.submit(session, work, progress));
            }
            Command::Cancel(ticket, reply) => {
                let _ = reply.send(self.cancel(ticket));
            }
            Command::Constrain {
                job,
                position,
                drafts,
                reply,
            } => {
                let _ = reply.send(self.constrain(job, position, drafts));
            }
            Command::Snapshot(session, reply) => {
                let _ = reply.send(self.snapshot(session));
            }
            Command::Restore(session, snapshot, reply) => {
                let _ = reply.send(self.restore(session, snapshot));
            }
            Command::SnapshotInfo(snapshot, reply) => {
                let _ = reply.send(self.snapshot_info(snapshot));
            }
            Command::ReadSnapshot(snapshot, offset, dst, reply) => {
                let _ = reply.send(self.read_snapshot(snapshot, offset, dst));
            }
            Command::ImportSnapshot(model, bytes, reply) => {
                let _ = reply.send(self.import_snapshot(model, bytes));
            }
            Command::ReleaseSnapshot(snapshot, reply) => {
                let _ = reply.send(self.release_snapshot(snapshot));
            }
            Command::Levers(model, reply) => {
                let _ = reply.send(self.levers(model));
            }
            Command::SetLevers(model, levers, reply) => {
                let _ = reply.send(self.set_levers(model, *levers));
            }
            Command::SetDecodeRounds(rounds, reply) => {
                let _ = reply.send(self.set_decode_rounds(rounds));
            }
            Command::SetPrefillSlice(slice_ms, reply) => {
                let _ = reply.send(self.set_prefill_slice(slice_ms));
            }
            Command::SetDecodeShare(share_pct, reply) => {
                let _ = reply.send(self.set_decode_share(share_pct));
            }
            Command::SetPrefillAge(calls, reply) => {
                let _ = reply.send(self.set_prefill_age(calls));
            }
            Command::SetCacheNamespace(session, namespace, reply) => {
                let _ = reply.send(self.set_cache_namespace(session, &namespace));
            }
            Command::FlushPrefixCache(model, reply) => {
                let _ = reply.send(self.flush_prefix_cache(model));
            }
            Command::PrefixStats(model, reply) => {
                let _ = reply.send(self.prefix_stats(model));
            }
            Command::PrefixCaptureStats(model, reply) => {
                let _ = reply.send(self.prefix_capture_stats(model));
            }
            Command::PrefillSignature(model, reply) => {
                let _ = reply.send(self.prefill_signature(model));
            }
            Command::ExportPrefix(model, namespace, tokens, reply) => {
                let _ = reply.send(self.export_prefix(model, &namespace, tokens));
            }
            Command::ImportPrefix(model, namespace, snapshot, reply) => {
                let _ = reply.send(self.import_prefix(model, &namespace, snapshot));
            }
            Command::LoraLoad(model, slot, weights, reply) => {
                let _ = reply.send(self.lora_load(model, slot, weights));
            }
            Command::LoraUnload(model, slot, reply) => {
                let _ = reply.send(self.lora_unload(model, slot));
            }
            Command::SetLora(session, slot, reply) => {
                let _ = reply.send(self.set_lora(session, slot));
            }
            Command::ResearchPlanLoad(model, slot, ops, sha256, reply) => {
                let _ = reply.send(self.research_plan_load(model, slot, ops, sha256));
            }
            Command::ResearchPlanUnload(model, slot, reply) => {
                let _ = reply.send(self.research_plan_unload(model, slot));
            }
            Command::SetResearch(session, slot, prompt_tokens, max_tokens, reply) => {
                let _ = reply.send(self.set_research(session, slot, prompt_tokens, max_tokens));
            }
            Command::ResearchCaptureRead(model, capture, reply) => {
                let _ = reply.send(self.research_capture_read(model, capture));
            }
            Command::ResearchCaptureFree(model, capture, reply) => {
                let _ = reply.send(self.research_capture_free(model, capture));
            }
            Command::Shutdown(_) => unreachable!("handled by run"),
        }
    }

    fn create_model(
        &mut self,
        spec: Box<ModelSpec>,
        background: Option<Background>,
    ) -> Result<(Raw<RawModel>, AllocateThread), DriverError> {
        let (api, executor, gate) = (self.api, self.executor, &*self.gate);
        let (spec, created) = match background {
            None => {
                let created = create_raw(&api, gate, executor, &spec);
                (spec, created.map(|raw| (raw, AllocateThread::current())))
            }
            Some(b) => std::thread::scope(|s| {
                let helper = std::thread::Builder::new()
                    .name("cere-allocate-bg".to_string())
                    .spawn_scoped(s, move || {
                        let created = into_background(b.nice, b.cpus)
                            .map_err(|e| {
                                DriverError::Argument(format!(
                                    "the background ALLOCATE thread: {e}"
                                ))
                            })
                            .and_then(|thread| {
                                create_raw(&api, gate, executor, &spec).map(|raw| (raw, thread))
                            });
                        (spec, created)
                    })
                    .map_err(|e| {
                        DriverError::Argument(format!(
                            "spawning the background ALLOCATE thread: {e}"
                        ))
                    })?;
                Ok(helper
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic)))
            })?,
        };
        let (raw, thread) = created?;
        self.models
            .insert(raw.key(), ModelState { generation: None });
        let setters = [
            (spec.prefix_end_state, Setter::EndState),
            (spec.prefix_alt_edges, Setter::AltEdges),
        ];
        for (on, setter) in setters {
            if !on {
                continue;
            }
            if let Err(e) = self.set_prefix_flag(raw, setter) {
                let _ = self.release_model(raw);
                return Err(e);
            }
        }
        if let Some(lora) = spec.lora {
            if let Err(e) = self.lora_reserve(raw, lora) {
                let _ = self.release_model(raw);
                return Err(e);
            }
        }
        if let Some(research) = spec.research {
            if let Err(e) = self.research_reserve(raw, research) {
                let _ = self.release_model(raw);
                return Err(e);
            }
        }
        Ok((raw, thread))
    }
}

/// `cere_model_create` of `spec` on `executor`, holding the put gate exclusively.
fn create_raw(
    api: &Api,
    gate: &RwLock<()>,
    executor: Raw<RawExecutor>,
    spec: &ModelSpec,
) -> Result<Raw<RawModel>, DriverError> {
    let names = spec
        .tensors
        .iter()
        .map(|t| {
            CString::new(t.name.as_str()).map_err(|_| {
                DriverError::Argument(format!("tensor name {:?} contains a NUL byte", t.name))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut tensors = Vec::with_capacity(spec.tensors.len());
    for (t, name) in spec.tensors.iter().zip(&names) {
        if t.shape.len() > ffi::TENSOR_MAX_DIMS {
            return Err(DriverError::Argument(format!(
                "tensor {} has {} dims (at most {})",
                t.name,
                t.shape.len(),
                ffi::TENSOR_MAX_DIMS
            )));
        }
        let mut shape = [0u64; ffi::TENSOR_MAX_DIMS];
        shape[..t.shape.len()].copy_from_slice(&t.shape);
        tensors.push(TensorDesc {
            struct_size: ffi::struct_size::<TensorDesc>(),
            abi_version: ffi::ABI_VERSION,
            name: name.as_ptr(),
            dtype: t.dtype,
            placement: t.placement,
            ndim: t.shape.len() as u32,
            reserved: 0,
            shape,
            bytes: t.bytes,
        });
    }
    let selection = spec.levers.ffi(spec.config.lever_family());
    let desc = ModelDesc {
        struct_size: ffi::struct_size::<ModelDesc>(),
        abi_version: ffi::ABI_VERSION,
        model_kind: spec.config.model_kind(),
        speculative_draft_tokens: spec.speculative_draft_tokens,
        config: spec.config.desc_ptr(),
        tensors: tensors.as_ptr(),
        tensor_count: tensors.len() as u64,
        max_sessions: spec.max_sessions,
        max_context_tokens: spec.max_context_tokens,
        stop_token_ids: spec.stop_token_ids.as_ptr(),
        stop_token_count: spec.stop_token_ids.len() as u64,
        kv_pool_tokens: spec.kv_pool_tokens,
        draft_token_map: if spec.draft_token_map.is_empty() {
            ptr::null()
        } else {
            spec.draft_token_map.as_ptr()
        },
        draft_token_map_count: spec.draft_token_map.len() as u64,
        snapshot_host_bytes: spec.snapshot_host_bytes,
        prefix_state_slots: spec.prefix_state_slots,
        vision_config: spec.vision.as_ref().map_or(ptr::null(), |v| v),
        selection: &selection,
        family: spec.config.family(),
        calibrate_windows: spec.levers.calibrate_windows(spec.config.lever_family()),
    };
    let mut out: *mut RawModel = ptr::null_mut();
    let rc = {
        let _excl = gate.write().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the desc and everything it points to (names, tensors, config, vision config, stop IDs, draft token
        // map) outlive the call, which copies them.
        unsafe { (api.model_create)(executor.ptr(), &desc, &mut out) }
    };
    check("cere_model_create", rc)?;
    NonNull::new(out).map(Raw).ok_or(DriverError::Status {
        call: "cere_model_create",
        status: StatusError::Internal,
    })
}

/// A prefix setting turned on once right after `cere_model_create`, before the model's first ticket.
#[derive(Clone, Copy)]
enum Setter {
    /// `cere_model_set_prefix_end_state` (m210 design note D15⁗′ stage 2a).
    EndState,
    /// `cere_model_set_prefix_alt_edges` (m210 design note D17).
    AltEdges,
}

impl Thread {
    /// `cere_model_set_prefix_end_state(1)` (D15⁗′ stage 2a) or `cere_model_set_prefix_alt_edges(1)` (D17) on a model
    /// without tickets.
    fn set_prefix_flag(&mut self, model: Raw<RawModel>, setter: Setter) -> Result<(), DriverError> {
        let (set, key, name) = match setter {
            Setter::EndState => (
                self.api.model_set_prefix_end_state,
                "prefix_end_state",
                "cere_model_set_prefix_end_state",
            ),
            Setter::AltEdges => (
                self.api.model_set_prefix_alt_edges,
                "prefix_alt_edges",
                "cere_model_set_prefix_alt_edges",
            ),
        };
        let Some(set) = set else {
            return Err(DriverError::Argument(format!(
                "{key} needs a library exporting {name}"
            )));
        };
        // SAFETY: a live model of this executor.
        let rc = self.exclusive(|_| unsafe { set(self.executor.ptr(), model.ptr(), 1) });
        check(name, rc)
    }

    /// `cere_model_lora_reserve` on a just-created model (track LoRA). A library without every LoRA call is refused:
    /// it could not serve what the config asks for.
    fn lora_reserve(&mut self, model: Raw<RawModel>, lora: LoraReserve) -> Result<(), DriverError> {
        let api = &self.api;
        let (Some(reserve), Some(_), Some(_), Some(_)) = (
            api.model_lora_reserve,
            api.model_lora_load,
            api.model_lora_unload,
            api.session_set_lora,
        ) else {
            return Err(DriverError::Argument(
                "lora needs a library exporting cere_model_lora_reserve, cere_model_lora_load, \
                 cere_model_lora_unload and cere_session_set_lora"
                    .to_string(),
            ));
        };
        // SAFETY: a live model of this executor; plain counts.
        let rc = self.exclusive(|_| unsafe {
            reserve(
                self.executor.ptr(),
                model.ptr(),
                lora.max_adapters,
                lora.max_rank,
            )
        });
        check("cere_model_lora_reserve", rc)
    }

    fn lora_load(
        &mut self,
        model: Raw<RawModel>,
        slot: u32,
        weights: Lent<*const LoraWeights>,
    ) -> Result<(), DriverError> {
        const CALL: &str = "cere_model_lora_load";
        let Some(load) = self.api.model_lora_load else {
            return Err(DriverError::Missing(CALL));
        };
        self.known_model(model)?;
        // SAFETY: the lending caller blocks on the reply, so the weights outlive this call (`Lent`).
        let w = unsafe { &*weights.0 };
        if w.a.len() != w.targets.len() || w.b.len() != w.targets.len() {
            return Err(DriverError::Argument(
                "lora weights need one A and one B per target".to_string(),
            ));
        }
        let a: Vec<*const c_void> = w.a.iter().map(|x| x.as_ptr().cast()).collect();
        let b: Vec<*const c_void> = w.b.iter().map(|x| x.as_ptr().cast()).collect();
        // SAFETY: a live model; `count` targets with an A and a B host pointer each, read before return.
        let rc = self.exclusive(|_| unsafe {
            load(
                self.executor.ptr(),
                model.ptr(),
                slot,
                w.rank,
                w.scale,
                w.targets.len() as u64,
                w.targets.as_ptr(),
                a.as_ptr(),
                b.as_ptr(),
            )
        });
        check(CALL, rc)
    }

    fn lora_unload(&mut self, model: Raw<RawModel>, slot: u32) -> Result<(), DriverError> {
        const CALL: &str = "cere_model_lora_unload";
        let Some(unload) = self.api.model_lora_unload else {
            return Err(DriverError::Missing(CALL));
        };
        self.known_model(model)?;
        // SAFETY: a live model of this executor.
        let rc = self.exclusive(|_| unsafe { unload(self.executor.ptr(), model.ptr(), slot) });
        check(CALL, rc)
    }

    fn set_lora(&mut self, session: Raw<RawSession>, slot: u32) -> Result<(), DriverError> {
        const CALL: &str = "cere_session_set_lora";
        let Some(set) = self.api.session_set_lora else {
            return Err(DriverError::Missing(CALL));
        };
        self.session_model(session)?;
        // SAFETY: a live session of this executor.
        let rc = self.exclusive(|_| unsafe { set(self.executor.ptr(), session.ptr(), slot) });
        check(CALL, rc)
    }

    /// `cere_model_research_reserve` on a just-created model (WeightLab WL-01). A library without every research
    /// call is refused: it could not serve what the config asks for.
    fn research_reserve(
        &mut self,
        model: Raw<RawModel>,
        research: ResearchReserve,
    ) -> Result<(), DriverError> {
        let api = &self.api;
        let (Some(reserve), Some(_), Some(_), Some(_), Some(_), Some(_)) = (
            api.model_research_reserve,
            api.model_research_plan_load,
            api.model_research_plan_unload,
            api.session_set_research,
            api.model_research_capture_read,
            api.model_research_capture_free,
        ) else {
            return Err(DriverError::Argument(
                "research needs a library exporting cere_model_research_reserve, \
                 cere_model_research_plan_load, cere_model_research_plan_unload, cere_session_set_research, \
                 cere_model_research_capture_read and cere_model_research_capture_free"
                    .to_string(),
            ));
        };
        // SAFETY: a live model of this executor; plain counts.
        let rc = self.exclusive(|_| unsafe {
            reserve(
                self.executor.ptr(),
                model.ptr(),
                research.device_staging_bytes,
                research.host_buffer_bytes,
                research.max_plans,
            )
        });
        check("cere_model_research_reserve", rc)
    }

    fn research_plan_load(
        &mut self,
        model: Raw<RawModel>,
        slot: u32,
        ops: Lent<*const ffi::ResearchOp>,
        sha256: [u8; 32],
    ) -> Result<(), DriverError> {
        const CALL: &str = "cere_model_research_plan_load";
        let Some(load) = self.api.model_research_plan_load else {
            return Err(DriverError::Missing(CALL));
        };
        self.known_model(model)?;
        let op_count = u32::try_from(ops.1)
            .map_err(|_| DriverError::Argument("too many research ops".to_string()))?;
        let plan = ffi::ResearchPlan {
            ops: ops.0,
            op_count,
            reserved: 0,
            sha256,
        };
        // SAFETY: the lending caller blocks on the reply, so the ops and every pointer in them outlive this call
        // (`Lent`); the library reads them before it returns.
        let rc = self.exclusive(|_| unsafe { load(self.executor.ptr(), model.ptr(), slot, &plan) });
        check(CALL, rc)
    }

    fn research_plan_unload(&mut self, model: Raw<RawModel>, slot: u32) -> Result<(), DriverError> {
        const CALL: &str = "cere_model_research_plan_unload";
        let Some(unload) = self.api.model_research_plan_unload else {
            return Err(DriverError::Missing(CALL));
        };
        self.known_model(model)?;
        // SAFETY: a live model of this executor.
        let rc = self.exclusive(|_| unsafe { unload(self.executor.ptr(), model.ptr(), slot) });
        check(CALL, rc)
    }

    fn set_research(
        &mut self,
        session: Raw<RawSession>,
        slot: u32,
        prompt_tokens: u64,
        max_tokens: u64,
    ) -> Result<u64, DriverError> {
        const CALL: &str = "cere_session_set_research";
        let Some(set) = self.api.session_set_research else {
            return Err(DriverError::Missing(CALL));
        };
        self.session_model(session)?;
        let mut capture = 0u64;
        // SAFETY: a live session of this executor and a valid out pointer.
        let rc = self.exclusive(|_| unsafe {
            set(
                self.executor.ptr(),
                session.ptr(),
                slot,
                prompt_tokens,
                max_tokens,
                &mut capture,
            )
        });
        check(CALL, rc).map(|()| capture)
    }

    fn research_capture_read(
        &mut self,
        model: Raw<RawModel>,
        capture: u64,
    ) -> Result<CaptureView, DriverError> {
        const CALL: &str = "cere_model_research_capture_read";
        let Some(read) = self.api.model_research_capture_read else {
            return Err(DriverError::Missing(CALL));
        };
        self.known_model(model)?;
        let mut out = ffi::ResearchCapture::default();
        // SAFETY: a live model of this executor and a valid out struct.
        let rc = self
            .exclusive(|_| unsafe { read(self.executor.ptr(), model.ptr(), capture, &mut out) });
        check(CALL, rc).map(|()| CaptureView(out))
    }

    fn research_capture_free(
        &mut self,
        model: Raw<RawModel>,
        capture: u64,
    ) -> Result<(), DriverError> {
        const CALL: &str = "cere_model_research_capture_free";
        let Some(free) = self.api.model_research_capture_free else {
            return Err(DriverError::Missing(CALL));
        };
        self.known_model(model)?;
        // SAFETY: a live model of this executor.
        let rc = self.exclusive(|_| unsafe { free(self.executor.ptr(), model.ptr(), capture) });
        check(CALL, rc)
    }

    fn known_model(&self, model: Raw<RawModel>) -> Result<(), DriverError> {
        if self.models.contains_key(&model.key()) {
            Ok(())
        } else {
            Err(DriverError::Argument(
                "the model is not live on this driver".to_string(),
            ))
        }
    }

    fn seal(&mut self, model: Raw<RawModel>) -> Result<ExecutionCapabilities, DriverError> {
        self.known_model(model)?;
        // SAFETY: a live model of this executor.
        let rc =
            self.exclusive(|api| unsafe { (api.model_seal)(self.executor.ptr(), model.ptr()) });
        check("cere_model_seal", rc)?;
        self.capabilities(model)
    }

    fn capabilities(&mut self, model: Raw<RawModel>) -> Result<ExecutionCapabilities, DriverError> {
        self.known_model(model)?;
        let mut caps = capabilities_header();
        // SAFETY: a live model and a caps struct with its header set.
        let rc = self.exclusive(|api| unsafe {
            (api.executor_capabilities)(self.executor.ptr(), model.ptr(), &mut caps)
        });
        check("cere_executor_capabilities", rc)?;
        if let Some(m) = self.models.get_mut(&model.key()) {
            m.generation = Some(caps.profile_generation);
        }
        Ok(caps)
    }

    fn release_model(&mut self, model: Raw<RawModel>) -> Result<(), DriverError> {
        self.known_model(model)?;
        // SAFETY: a live model of this executor.
        let rc =
            self.exclusive(|api| unsafe { (api.model_release)(self.executor.ptr(), model.ptr()) });
        check("cere_model_release", rc)?;
        self.models.remove(&model.key());
        Ok(())
    }

    fn levers(&mut self, model: Raw<RawModel>) -> Result<SlotSelection, DriverError> {
        self.known_model(model)?;
        let mut out = SlotSelection {
            struct_size: ffi::struct_size::<SlotSelection>(),
            abi_version: ffi::ABI_VERSION,
            ..SlotSelection::default()
        };
        // SAFETY: a live model and an out struct with its header set.
        let rc = self.exclusive(|api| unsafe {
            (api.model_selection)(self.executor.ptr(), model.ptr(), &mut out)
        });
        check("cere_model_selection", rc)?;
        Ok(out)
    }

    fn set_levers(
        &mut self,
        model: Raw<RawModel>,
        selection: SlotSelection,
    ) -> Result<(), DriverError> {
        self.known_model(model)?;
        // SAFETY: a live model and a selection block the call copies.
        let rc = self.exclusive(|api| unsafe {
            (api.model_set_selection)(self.executor.ptr(), model.ptr(), &selection)
        });
        check("cere_model_set_selection", rc)
    }

    fn set_decode_rounds(&mut self, rounds: u32) -> Result<(), DriverError> {
        const CALL: &str = "cere_executor_set_decode_rounds";
        let Some(set) = self.api.executor_set_decode_rounds else {
            return Err(DriverError::Missing(CALL));
        };
        // SAFETY: the live executor; the call takes a plain count.
        let rc = self.exclusive(|_| unsafe { set(self.executor.ptr(), rounds) });
        check(CALL, rc)
    }

    fn set_prefill_slice(&mut self, slice_ms: u32) -> Result<(), DriverError> {
        const CALL: &str = "cere_executor_set_prefill_slice";
        let Some(set) = self.api.executor_set_prefill_slice else {
            return Err(DriverError::Missing(CALL));
        };
        // SAFETY: the live executor; the call takes a plain duration.
        let rc = self.exclusive(|_| unsafe { set(self.executor.ptr(), slice_ms) });
        check(CALL, rc)
    }

    fn set_decode_share(&mut self, share_pct: u32) -> Result<(), DriverError> {
        const CALL: &str = "cere_executor_set_decode_share";
        let Some(set) = self.api.executor_set_decode_share else {
            return Err(DriverError::Missing(CALL));
        };
        // SAFETY: the live executor; the call takes a plain percentage.
        let rc = self.exclusive(|_| unsafe { set(self.executor.ptr(), share_pct) });
        check(CALL, rc)
    }

    fn set_prefill_age(&mut self, calls: u32) -> Result<(), DriverError> {
        const CALL: &str = "cere_executor_set_prefill_age";
        let Some(set) = self.api.executor_set_prefill_age else {
            return Err(DriverError::Missing(CALL));
        };
        // SAFETY: the live executor; the call takes a plain count.
        let rc = self.exclusive(|_| unsafe { set(self.executor.ptr(), calls) });
        check(CALL, rc)
    }

    /// `cere_executor_round_drafts` of `ticket`, awaiting the mask of `position` (GrammarSpec GS-02).
    fn round_drafts(
        &self,
        ticket: CompletionTicket,
        position: u64,
    ) -> Result<RoundDrafts, DriverError> {
        const CALL: &str = "cere_executor_round_drafts";
        let Some(round) = self.api.executor_round_drafts else {
            return Err(DriverError::Missing(CALL));
        };
        let mut drafts = RoundDrafts::default();
        // SAFETY: a ticket of this executor; `ids` holds the capacity passed, `count` is a plain out value.
        let rc = self.exclusive(|_| unsafe {
            round(
                self.executor.ptr(),
                ticket,
                position,
                drafts.ids.as_mut_ptr(),
                ffi::MAX_CONSTRAINED_DRAFTS,
                &mut drafts.count,
            )
        });
        check(CALL, rc)?;
        if drafts.count > ffi::MAX_CONSTRAINED_DRAFTS {
            return Err(DriverError::Argument(format!(
                "{CALL} reported {} drafts for a capacity of {}",
                drafts.count,
                ffi::MAX_CONSTRAINED_DRAFTS
            )));
        }
        Ok(drafts)
    }

    fn create_session(
        &mut self,
        model: Raw<RawModel>,
        max_context_tokens: u64,
    ) -> Result<Raw<RawSession>, DriverError> {
        self.known_model(model)?;
        let mut out: *mut RawSession = ptr::null_mut();
        // SAFETY: a live model and an out pointer.
        let rc = self.exclusive(|api| unsafe {
            (api.session_create)(
                self.executor.ptr(),
                model.ptr(),
                max_context_tokens,
                &mut out,
            )
        });
        check("cere_session_create", rc)?;
        let raw = NonNull::new(out).map(Raw).ok_or(DriverError::Status {
            call: "cere_session_create",
            status: StatusError::Internal,
        })?;
        self.sessions.insert(
            raw.key(),
            SessionState {
                model: model.key(),
                revision: 0,
            },
        );
        Ok(raw)
    }

    fn release_session(&mut self, session: Raw<RawSession>) -> Result<(), DriverError> {
        if !self.sessions.contains_key(&session.key()) {
            return Err(DriverError::Argument(
                "the session is not live on this driver".to_string(),
            ));
        }
        // SAFETY: a live session of this executor.
        let rc = self
            .exclusive(|api| unsafe { (api.session_release)(self.executor.ptr(), session.ptr()) });
        check("cere_session_release", rc)?;
        self.sessions.remove(&session.key());
        Ok(())
    }

    fn submit(
        &mut self,
        session: Raw<RawSession>,
        work: WorkRequest,
        progress: Option<Progress>,
    ) -> Result<Job, DriverError> {
        let state = self.sessions.get(&session.key()).ok_or_else(|| {
            DriverError::Argument("the session is not live on this driver".to_string())
        })?;
        let generation = self
            .models
            .get(&state.model)
            .and_then(|m| m.generation)
            .ok_or_else(|| {
                DriverError::Argument("the session's model is not sealed".to_string())
            })?;
        if work.grammar_drafts {
            if work.vocab_mask.is_empty() {
                return Err(DriverError::Argument(
                    "grammar drafts need a constrained item".to_string(),
                ));
            }
            if self.api.executor_round_drafts.is_none() {
                return Err(DriverError::Missing("cere_executor_round_drafts"));
            }
            if self.api.executor_constrain_drafts.is_none() {
                return Err(DriverError::Missing("cere_executor_constrain_drafts"));
            }
        }
        let capacity = usize::try_from(work.max_new_tokens).map_err(|_| {
            DriverError::Argument("max_new_tokens does not fit in memory".to_string())
        })?;
        let mut out_tokens = vec![0u32; capacity];
        let mut out_scores = if work.scores {
            vec![TokenScore::default(); capacity]
        } else {
            Vec::new()
        };
        let mut out_logprobs = if work.top_logprobs.is_some() {
            vec![TokenLogprobs::default(); capacity]
        } else {
            Vec::new()
        };
        // `cere_work.sampling` and its `logit_bias` list are copied at submit (`cere_executor.h`), so they only
        // need to outlive the call: `params` and `work` both live until the end of this function.
        let params = work.sampling.as_ref().map(|s| SamplingParams {
            struct_size: ffi::struct_size::<SamplingParams>(),
            abi_version: ffi::ABI_VERSION,
            temperature: s.temperature,
            top_p: s.top_p,
            min_p: s.min_p,
            top_k: s.top_k,
            frequency_penalty: s.frequency_penalty,
            presence_penalty: s.presence_penalty,
            repetition_penalty: s.repetition_penalty,
            flags: 0,
            seed: s.seed,
            logit_bias: if s.logit_bias.is_empty() {
                ptr::null()
            } else {
                s.logit_bias.as_ptr()
            },
            logit_bias_count: s.logit_bias.len() as u64,
            suppress_token_ids: ptr::null(),
            suppress_token_count: 0,
            mask_until_committed: 0,
        });
        // Borrows `work.images`' pixel buffers, which live until this function returns (the library copies them
        // before `cere_executor_submit` returns).
        let images: Vec<ffi::ImageInput> = work
            .images
            .iter()
            .map(|im| ffi::ImageInput {
                struct_size: ffi::struct_size::<ffi::ImageInput>(),
                abi_version: ffi::ABI_VERSION,
                grid_t: im.grid_thw[0],
                grid_h: im.grid_thw[1],
                grid_w: im.grid_thw[2],
                pad_id: im.pad_id,
                token_offset: im.token_offset,
                pixel_patches: im.pixels.as_ptr().cast(),
                pixel_bytes: im.pixels.len() as u64,
                pixel_sha256: im.pixel_sha256,
            })
            .collect();
        let item = Work {
            struct_size: ffi::struct_size::<Work>(),
            abi_version: ffi::ABI_VERSION,
            operation: work.operation,
            flags: work.flags,
            expected_worker_epoch: self.epoch,
            expected_session_revision: state.revision,
            session: session.ptr(),
            tokens: if work.tokens.is_empty() {
                ptr::null()
            } else {
                work.tokens.as_ptr()
            },
            token_count: work.tokens.len() as u64,
            max_new_tokens: work.max_new_tokens,
            reserved_peak_bytes: 0,
            out_tokens: out_tokens.as_mut_ptr(),
            out_capacity: work.max_new_tokens,
            out_scores: if work.scores {
                out_scores.as_mut_ptr()
            } else {
                ptr::null_mut()
            },
            stop_token_ids: if work.stop_token_ids.is_empty() {
                ptr::null()
            } else {
                work.stop_token_ids.as_ptr()
            },
            stop_token_count: work.stop_token_ids.len() as u64,
            sampling: params
                .as_ref()
                .map_or(ptr::null(), |p| p as *const SamplingParams),
            out_logprobs: if work.top_logprobs.is_some() {
                out_logprobs.as_mut_ptr()
            } else {
                ptr::null_mut()
            },
            top_logprobs: work.top_logprobs.unwrap_or(0),
            capture_position: work.capture_position,
            vocab_mask: if work.vocab_mask.is_empty() {
                ptr::null()
            } else {
                work.vocab_mask.as_ptr()
            },
            vocab_mask_words: work.vocab_mask.len() as u64,
            images: if images.is_empty() {
                ptr::null()
            } else {
                images.as_ptr()
            },
            image_count: images.len() as u64,
        };
        let batch = Batch {
            struct_size: ffi::struct_size::<Batch>(),
            abi_version: ffi::ABI_VERSION,
            expected_worker_epoch: self.epoch,
            capability_profile_generation: generation,
            items: &item,
            item_count: 1,
            reserved_peak_bytes: 0,
        };
        let mut ticket = CompletionTicket::default();
        // SAFETY: the batch, the item and its input lists (prompt, stop IDs, image descriptors and their pixel
        // bytes) outlive the call (copied before return); the output buffers move into `InFlight` (their heap
        // storage does not move) and stay there until the terminal poll.
        let rc = self.exclusive(|api| unsafe {
            (api.executor_submit)(self.executor.ptr(), &batch, &mut ticket)
        });
        check("cere_executor_submit", rc)?;
        let (reply, rx) = mpsc::sync_channel(1);
        // The submit copied position 0's mask; the same buffer carries every later position's (A7), and with grammar
        // drafts the masks after each draft too (room for them reserved here).
        let mask_words = work.vocab_mask.len();
        let mask = (mask_words > 0).then(|| {
            let mut buffer = work.vocab_mask;
            if work.grammar_drafts {
                buffer.reserve_exact(ffi::MAX_CONSTRAINED_DRAFTS as usize * mask_words);
            }
            Arc::new(Mutex::new(buffer))
        });
        self.jobs.push(InFlight {
            ticket,
            session: session.key(),
            out_tokens,
            out_scores,
            out_logprobs,
            reply,
            progress,
            reported: 0,
            mask: mask.clone(),
            mask_words,
            masked: 1,
            awaiting_mask: false,
            grammar_drafts: work.grammar_drafts,
        });
        Ok(Job { ticket, rx, mask })
    }

    fn cancel(&mut self, ticket: CompletionTicket) -> Result<(), DriverError> {
        // SAFETY: a ticket is plain data; the library validates it.
        let rc =
            self.exclusive(|api| unsafe { (api.executor_cancel)(self.executor.ptr(), ticket) });
        // A job awaiting its mask is polled again, so the cancel reaches its terminal poll (spec m28 A7).
        if let Some(job) = self.jobs.iter_mut().find(|j| j.ticket == ticket) {
            job.awaiting_mask = false;
        }
        check("cere_executor_cancel", rc)
    }

    /// `cere_executor_constrain` with the job's buffer (spec m28 A2, A7), or with `drafts`
    /// `cere_executor_constrain_drafts` with its `drafts + 1` masks (GrammarSpec GS-02). A job no longer in flight
    /// already ended (its terminal poll raced the mask): nothing to constrain.
    fn constrain(
        &mut self,
        ticket: CompletionTicket,
        position: u64,
        drafts: Option<RoundDrafts>,
    ) -> Result<(), DriverError> {
        let Some(index) = self.jobs.iter().position(|j| j.ticket == ticket) else {
            return Ok(());
        };
        let Some(mask) = self.jobs[index].mask.clone() else {
            return Err(DriverError::Argument(
                "the job is not constrained".to_string(),
            ));
        };
        let mask_words = self.jobs[index].mask_words;
        let words = mask.lock().unwrap_or_else(PoisonError::into_inner);
        let (call, rc) = match drafts {
            // SAFETY: the mask outlives the call (the library copies it before return).
            None => (
                "cere_executor_constrain",
                self.exclusive(|api| unsafe {
                    (api.executor_constrain)(
                        self.executor.ptr(),
                        ticket,
                        position,
                        words.as_ptr(),
                        words.len() as u64,
                    )
                }),
            ),
            Some(drafts) => {
                const CALL: &str = "cere_executor_constrain_drafts";
                let Some(constrain) = self.api.executor_constrain_drafts else {
                    return Err(DriverError::Missing(CALL));
                };
                let masks = drafts.count as usize + 1;
                if words.len() != masks * mask_words {
                    return Err(DriverError::Argument(format!(
                        "{CALL} needs {masks} masks of {mask_words} words, the job's buffer holds {} words",
                        words.len()
                    )));
                }
                // SAFETY: the drafts and the masks outlive the call (the library copies them before return).
                let rc = self.exclusive(|_| unsafe {
                    constrain(
                        self.executor.ptr(),
                        ticket,
                        position,
                        drafts.ids.as_ptr(),
                        drafts.count,
                        words.as_ptr(),
                        mask_words as u64,
                    )
                });
                (CALL, rc)
            }
        };
        drop(words);
        check(call, rc)?;
        let job = &mut self.jobs[index];
        job.masked = job.masked.max(position + 1);
        job.awaiting_mask = false;
        Ok(())
    }

    fn session_model(&self, session: Raw<RawSession>) -> Result<usize, DriverError> {
        self.sessions
            .get(&session.key())
            .map(|s| s.model)
            .ok_or_else(|| {
                DriverError::Argument("the session is not live on this driver".to_string())
            })
    }

    fn known_snapshot(&self, snapshot: Raw<RawSnapshot>) -> Result<(), DriverError> {
        if self.snapshots.contains_key(&snapshot.key()) {
            Ok(())
        } else {
            Err(DriverError::Argument(
                "the snapshot is not live on this driver".to_string(),
            ))
        }
    }

    fn track_snapshot(
        &mut self,
        call: &'static str,
        out: *mut RawSnapshot,
        model: usize,
    ) -> Result<Raw<RawSnapshot>, DriverError> {
        let raw = NonNull::new(out).map(Raw).ok_or(DriverError::Status {
            call,
            status: StatusError::Internal,
        })?;
        self.snapshots.insert(raw.key(), model);
        Ok(raw)
    }

    fn snapshot(&mut self, session: Raw<RawSession>) -> Result<Raw<RawSnapshot>, DriverError> {
        let model = self.session_model(session)?;
        let revision = self.sessions[&session.key()].revision;
        let mut out: *mut RawSnapshot = ptr::null_mut();
        // SAFETY: a live session of this executor and an out pointer.
        let rc = self.exclusive(|api| unsafe {
            (api.session_snapshot)(self.executor.ptr(), session.ptr(), revision, &mut out)
        });
        check("cere_session_snapshot", rc)?;
        self.track_snapshot("cere_session_snapshot", out, model)
    }

    fn restore(
        &mut self,
        session: Raw<RawSession>,
        snapshot: Raw<RawSnapshot>,
    ) -> Result<(), DriverError> {
        self.session_model(session)?;
        self.known_snapshot(snapshot)?;
        // SAFETY: a live session and a live snapshot of this executor.
        let rc = self.exclusive(|api| unsafe {
            (api.session_restore)(self.executor.ptr(), session.ptr(), snapshot.ptr())
        });
        check("cere_session_restore", rc)
    }

    fn snapshot_info(&mut self, snapshot: Raw<RawSnapshot>) -> Result<SnapshotInfo, DriverError> {
        self.known_snapshot(snapshot)?;
        let mut info = SnapshotInfo {
            struct_size: ffi::struct_size::<SnapshotInfo>(),
            abi_version: ffi::ABI_VERSION,
            ..SnapshotInfo::default()
        };
        // SAFETY: a live snapshot and an info struct with its header set.
        let rc = self.exclusive(|api| unsafe {
            (api.snapshot_get_info)(self.executor.ptr(), snapshot.ptr(), &mut info)
        });
        check("cere_snapshot_get_info", rc)?;
        Ok(info)
    }

    fn read_snapshot(
        &mut self,
        snapshot: Raw<RawSnapshot>,
        offset: u64,
        dst: Lent<*mut u8>,
    ) -> Result<(), DriverError> {
        self.known_snapshot(snapshot)?;
        // SAFETY: a live snapshot; `dst` is the caller's buffer of `dst.1` bytes, borrowed mutably while the
        // caller waits for this reply.
        let rc = self.exclusive(|api| unsafe {
            (api.snapshot_read)(
                self.executor.ptr(),
                snapshot.ptr(),
                offset,
                dst.0.cast::<c_void>(),
                dst.1 as u64,
            )
        });
        check("cere_snapshot_read", rc)
    }

    fn import_snapshot(
        &mut self,
        model: Raw<RawModel>,
        bytes: Lent<*const u8>,
    ) -> Result<Raw<RawSnapshot>, DriverError> {
        self.known_model(model)?;
        let mut out: *mut RawSnapshot = ptr::null_mut();
        // SAFETY: a live model; `bytes` is the caller's buffer of `bytes.1` bytes, borrowed while the caller waits
        // for this reply (the library copies it into the arena before return).
        let rc = self.exclusive(|api| unsafe {
            (api.snapshot_import)(
                self.executor.ptr(),
                model.ptr(),
                bytes.0.cast::<c_void>(),
                bytes.1 as u64,
                &mut out,
            )
        });
        check("cere_snapshot_import", rc)?;
        self.track_snapshot("cere_snapshot_import", out, model.key())
    }

    fn release_snapshot(&mut self, snapshot: Raw<RawSnapshot>) -> Result<(), DriverError> {
        self.known_snapshot(snapshot)?;
        // SAFETY: a live snapshot of this executor.
        let rc = self.exclusive(|api| unsafe {
            (api.snapshot_release)(self.executor.ptr(), snapshot.ptr())
        });
        check("cere_snapshot_release", rc)?;
        self.snapshots.remove(&snapshot.key());
        Ok(())
    }

    fn set_cache_namespace(
        &mut self,
        session: Raw<RawSession>,
        namespace: &[u8; 16],
    ) -> Result<(), DriverError> {
        self.session_model(session)?;
        // SAFETY: a live session of this executor; the library copies the 16 bytes before return.
        let rc = self.exclusive(|api| unsafe {
            (api.session_set_cache_namespace)(self.executor.ptr(), session.ptr(), namespace)
        });
        check("cere_session_set_cache_namespace", rc)
    }

    fn flush_prefix_cache(&mut self, model: Raw<RawModel>) -> Result<(), DriverError> {
        self.known_model(model)?;
        // SAFETY: a live model of this executor.
        let rc = self.exclusive(|api| unsafe {
            (api.model_flush_prefix_cache)(self.executor.ptr(), model.ptr())
        });
        check("cere_model_flush_prefix_cache", rc)
    }

    fn prefix_stats(&mut self, model: Raw<RawModel>) -> Result<PrefixStats, DriverError> {
        const CALL: &str = "cere_model_prefix_stats";
        self.known_model(model)?;
        let Some(read) = self.api.model_prefix_stats else {
            return Err(DriverError::Missing(CALL));
        };
        let mut out = PrefixStats {
            struct_size: ffi::struct_size::<PrefixStats>(),
            abi_version: ffi::ABI_VERSION,
            ..PrefixStats::default()
        };
        // SAFETY: a live model of this executor and an out struct with its header set.
        let rc = self.exclusive(|_| unsafe { read(self.executor.ptr(), model.ptr(), &mut out) });
        check(CALL, rc)?;
        Ok(out)
    }

    fn prefix_capture_stats(
        &mut self,
        model: Raw<RawModel>,
    ) -> Result<PrefixCaptureStats, DriverError> {
        const CALL: &str = "cere_model_prefix_capture_stats";
        self.known_model(model)?;
        let Some(read) = self.api.model_prefix_capture_stats else {
            return Err(DriverError::Missing(CALL));
        };
        let mut out = PrefixCaptureStats {
            struct_size: ffi::struct_size::<PrefixCaptureStats>(),
            abi_version: ffi::ABI_VERSION,
            ..PrefixCaptureStats::default()
        };
        // SAFETY: a live model of this executor and an out struct with its header set.
        let rc = self.exclusive(|_| unsafe { read(self.executor.ptr(), model.ptr(), &mut out) });
        check(CALL, rc)?;
        Ok(out)
    }

    fn prefill_signature(&mut self, model: Raw<RawModel>) -> Result<Option<[u8; 32]>, DriverError> {
        self.known_model(model)?;
        let Some(read) = self.api.model_prefill_signature else {
            return Ok(None);
        };
        let mut out = [0u8; 32];
        // SAFETY: a live model of this executor and a 32-byte out buffer.
        let rc =
            self.exclusive(|_| unsafe { read(self.executor.ptr(), model.ptr(), out.as_mut_ptr()) });
        match StatusError::check(rc) {
            Ok(()) => Ok(Some(out)),
            Err(StatusError::Unsupported) => Ok(None),
            Err(status) => Err(DriverError::Status {
                call: "cere_model_prefill_signature",
                status,
            }),
        }
    }

    fn export_prefix(
        &mut self,
        model: Raw<RawModel>,
        namespace: &[u8; 16],
        tokens: Lent<*const u32>,
    ) -> Result<Raw<RawSnapshot>, DriverError> {
        self.known_model(model)?;
        let mut out: *mut RawSnapshot = ptr::null_mut();
        // SAFETY: a live model; `tokens` is the caller's slice of `tokens.1` ids, borrowed while the caller waits
        // for this reply; the namespace is read before return.
        let rc = self.exclusive(|api| unsafe {
            (api.prefix_cache_export)(
                self.executor.ptr(),
                model.ptr(),
                namespace,
                tokens.0,
                tokens.1 as u64,
                &mut out,
            )
        });
        check("cere_prefix_cache_export", rc)?;
        self.track_snapshot("cere_prefix_cache_export", out, model.key())
    }

    fn import_prefix(
        &mut self,
        model: Raw<RawModel>,
        namespace: &[u8; 16],
        snapshot: Raw<RawSnapshot>,
    ) -> Result<(), DriverError> {
        self.known_model(model)?;
        self.known_snapshot(snapshot)?;
        // SAFETY: a live model and a live snapshot of this executor; the namespace is read before return.
        let rc = self.exclusive(|api| unsafe {
            (api.prefix_cache_import)(self.executor.ptr(), model.ptr(), namespace, snapshot.ptr())
        });
        check("cere_prefix_cache_import", rc)
    }

    /// Polls every in-flight ticket once, except jobs awaiting their mask (spec m28 A7). With `rx`, a command waiting
    /// after a poll (each one runs a round) ends the pass early (RG-L1), the unpolled jobs moved first for the next.
    fn poll_jobs(&mut self, rx: Option<&Receiver<Command>>) -> Pass {
        let mut pass = Pass::default();
        let mut i = 0;
        while i < self.jobs.len() {
            if self.jobs[i].awaiting_mask {
                i += 1;
                continue;
            }
            let ticket = self.jobs[i].ticket;
            let mut c = completion_header();
            // SAFETY: a ticket of this executor and a completion with its header set.
            let rc = self.exclusive(|api| unsafe {
                (api.executor_poll)(self.executor.ptr(), ticket, &mut c)
            });
            match rc {
                ffi::PENDING => {
                    pass.progressed |= c.committed_tokens > 0;
                    let job = &mut self.jobs[i];
                    // A constrained job that committed every masked position cannot step before the next mask
                    // (D9: the executor returns PENDING without stepping), so it is not polled until then. The
                    // flag is set before the report, so a `Constrain` answering it always finds it set.
                    job.awaiting_mask = job.mask.is_some()
                        && c.committed_tokens >= job.masked
                        && c.committed_tokens < job.out_tokens.len() as u64;
                    // GrammarSpec GS-02: the round awaiting that mask froze its drafts; they ride the report.
                    let drafts = (job.awaiting_mask && job.grammar_drafts)
                        .then(|| self.round_drafts(ticket, c.committed_tokens));
                    self.jobs[i].report(c.committed_tokens, drafts);
                    i += 1;
                }
                ffi::OK if c.terminal != 0 => {
                    pass.progressed = true;
                    pass.ended = true;
                    let mut job = self.jobs.swap_remove(i);
                    // The terminal poll may commit tokens too (a PREFILL's only token, a DECODE's last one):
                    // `progress` gets them before the reply, so it carries every committed token exactly once.
                    job.report(c.committed_tokens, None);
                    self.finish(job, &c);
                }
                other => {
                    // The ticket ended without a terminal completion: the executor may still reference the leased
                    // buffers, so they are leaked rather than freed.
                    pass.progressed = true;
                    pass.ended = true;
                    let job = self.jobs.swap_remove(i);
                    let status = StatusError::check(other)
                        .err()
                        .unwrap_or(StatusError::Internal);
                    let _ = job.reply.send(Err(DriverError::Status {
                        call: "cere_executor_poll",
                        status,
                    }));
                    std::mem::forget(job.out_tokens);
                    std::mem::forget(job.out_scores);
                    std::mem::forget(job.out_logprobs);
                }
            }
            // Jobs [i..] are this pass's unpolled ones (a swap_remove moved the last one to i).
            if let Some(cmd) = rx.and_then(|rx| rx.try_recv().ok()) {
                self.jobs.rotate_left(i);
                pass.command = Some(cmd);
                break;
            }
        }
        pass
    }

    fn finish(&mut self, job: InFlight, c: &Completion) {
        if let Some(s) = self.sessions.get_mut(&job.session) {
            s.revision = c.session_revision;
        }
        let n = (c.committed_tokens as usize).min(job.out_tokens.len());
        let mut tokens = job.out_tokens;
        tokens.truncate(n);
        let mut scores = job.out_scores;
        scores.truncate(n);
        let mut logprobs = job.out_logprobs;
        logprobs.truncate(n);
        let _ = job.reply.send(Ok(JobResult {
            ticket: job.ticket,
            status: StatusError::check(c.status),
            finish: Finish::from_raw(c.finish),
            committed_tokens: c.committed_tokens,
            session_revision: c.session_revision,
            released_bytes: c.released_bytes,
            cached_tokens: c.cached_tokens,
            tokens,
            scores,
            logprobs,
            verify_steps: c.verify_steps,
            accepted_drafts: c.accepted_drafts,
        }));
    }

    /// Cancel and drain every job, then release snapshots, sessions, models and the executor; the first error
    /// wins.
    fn shutdown(&mut self) -> Result<(), DriverError> {
        let mut first: Option<DriverError> = None;
        let mut note = |r: Result<(), DriverError>| {
            if let Err(e) = r {
                first.get_or_insert(e);
            }
        };
        let tickets: Vec<CompletionTicket> = self.jobs.iter().map(|j| j.ticket).collect();
        for t in tickets {
            let _ = self.cancel(t);
        }
        while !self.jobs.is_empty() {
            if !self.poll_jobs(None).progressed {
                std::thread::sleep(IDLE_POLL);
            }
        }
        let snapshots: Vec<usize> = self.snapshots.keys().copied().collect();
        for s in snapshots {
            let raw =
                Raw(NonNull::new(s as *mut RawSnapshot).expect("tracked snapshots are non-null"));
            note(self.release_snapshot(raw));
        }
        let sessions: Vec<usize> = self.sessions.keys().copied().collect();
        for s in sessions {
            let raw =
                Raw(NonNull::new(s as *mut RawSession).expect("tracked sessions are non-null"));
            note(self.release_session(raw));
        }
        let models: Vec<usize> = self.models.keys().copied().collect();
        for m in models {
            let raw = Raw(NonNull::new(m as *mut RawModel).expect("tracked models are non-null"));
            note(self.release_model(raw));
        }
        // SAFETY: the executor is idle (drained) and destroyed once.
        let rc = self.exclusive(|api| unsafe { (api.executor_destroy)(self.executor.ptr()) });
        note(check("cere_executor_destroy", rc));
        match first {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Stands in for the driver thread's tickets: one decoding job that every poll steps (one round), and the log
    /// of rounds and commands in the order the driver thread took them. Like [`Thread::poll_jobs`], a poll ends with
    /// the waiting command; with none waiting it ends the test (a `Shutdown`).
    struct Fake {
        log: Vec<&'static str>,
        started: Instant,
        first_round: Option<Duration>,
    }

    impl Fake {
        fn new() -> Self {
            Self {
                log: Vec::new(),
                started: Instant::now(),
                first_round: None,
            }
        }
    }

    impl Tickets for Fake {
        fn none(&self) -> bool {
            false
        }

        fn all_awaiting_mask(&self) -> bool {
            false
        }

        fn poll(&mut self, rx: &Receiver<Command>) -> Pass {
            self.first_round.get_or_insert(self.started.elapsed());
            self.log.push("round");
            Pass {
                progressed: true,
                ended: false,
                command: Some(rx.try_recv().unwrap_or_else(|_| Command::Shutdown(reply()))),
            }
        }
    }

    fn reply<T>() -> Reply<T> {
        mpsc::sync_channel(1).0
    }

    fn kind(cmd: &Command) -> &'static str {
        match cmd {
            Command::CreateSession(..) => "create",
            Command::SetCacheNamespace(..) => "namespace",
            Command::Submit(..) => "submit",
            Command::Shutdown(..) => "end",
            _ => "other",
        }
    }

    /// [`Thread::run`]'s loop over the fake, without handling the commands.
    fn drive(fake: &mut Fake, rx: &Receiver<Command>, mut linger: bool) {
        loop {
            let cmd = next_command(fake, rx, &mut linger).expect("the test holds a sender");
            fake.log.push(kind(&cmd));
            if matches!(cmd, Command::Shutdown(..)) {
                return;
            }
            linger = cmd.has_follow_up();
        }
    }

    /// BcAdmit (O282, batch-concurrency design note): commands that queued while a round ran all go before the
    /// next round, a Submit (no follow-up) included. Before, the round after each Submit split a burst's admission
    /// chains one round apart.
    #[test]
    fn queued_commands_all_go_before_the_next_round() {
        let (tx, rx) = mpsc::channel();
        let session = Raw(NonNull::<RawSession>::dangling());
        let model = Raw(NonNull::<RawModel>::dangling());
        let submit = || Command::Submit(session, WorkRequest::prefill(vec![1], 1), None, reply());
        for cmd in [
            submit(),
            Command::CreateSession(model, 64, reply()),
            Command::SetCacheNamespace(session, [0; 16], reply()),
            submit(),
        ] {
            tx.send(cmd).unwrap();
        }
        let mut fake = Fake::new();
        drive(&mut fake, &rx, false);
        assert_eq!(
            fake.log,
            ["submit", "create", "namespace", "submit", "round", "end"]
        );
    }

    /// RG-L1 kept: with nothing queued after a command with a follow-up, the next round waits `COMMAND_LINGER`.
    #[test]
    fn a_follow_up_still_lingers_before_the_next_round() {
        let (_tx, rx) = mpsc::channel::<Command>();
        let mut fake = Fake::new();
        drive(&mut fake, &rx, true);
        assert_eq!(fake.log, ["round", "end"]);
        assert!(fake.first_round.unwrap() >= COMMAND_LINGER);
    }
}
