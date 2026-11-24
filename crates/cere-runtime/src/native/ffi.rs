//! ABI v5 projection of `executor/include/cere_executor.h` and the executor library loader (spec m22 R1, Q10, Q19;
//! m23 design note PI-03 added the levers, selftune design note ST2 made them the `cere_slot_selection` block).
//!
//! Every struct is `#[repr(C)]` with the header's field order; `tests/native_ffi.rs` compares sizes, alignments and
//! offsets with `tests/fixtures/executor_abi_v5_layout.json`, which the C test generates, so the projection cannot
//! drift (`docs/EXECUTION_PLUGIN.md`). [`Library::open`] hashes the file through one descriptor, compares the
//! SHA-256 with the expected value *before* `dlopen`, loads that same inode through the descriptor's
//! `/proc/self/fd` name, checks `cere_executor_abi_version()` and resolves every symbol. The hash and the load are two
//! reads of one inode, not one read: see [`Library::open`] for what that does and does not exclude.
use sha2::{Digest, Sha256};
use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt;
use std::io::Read;
use std::marker::{PhantomData, PhantomPinned};
use std::os::fd::{AsRawFd, IntoRawFd};
use std::path::{Path, PathBuf};

pub const ABI_VERSION: u32 = 5;

pub type Status = i32;
pub const OK: Status = 0;
pub const PENDING: Status = 1;
pub const ERR_ARGUMENT: Status = -1;
pub const ERR_UNSUPPORTED: Status = -2;
pub const ERR_CAPACITY: Status = -3;
pub const ERR_STALE_EPOCH: Status = -4;
pub const ERR_BUSY: Status = -5;
pub const ERR_INTERNAL: Status = -6;

pub const OP_PREFILL: u32 = 1;
pub const OP_DECODE: u32 = 2;
pub const OP_VERIFY: u32 = 3;
pub const OPS_PREFILL: u64 = 0x2;
pub const OPS_DECODE: u64 = 0x4;
pub const OPS_VERIFY: u64 = 0x8;
pub const OPS_CANCEL: u64 = 0x10;
/// Non-work operations of M2.11 (ledger K): set iff the model has a snapshot arena.
pub const OPS_SNAPSHOT: u64 = 0x20;
pub const OPS_RESTORE: u64 = 0x40;

/// `cere_work.flags` (ledger table F).
pub const WORK_IGNORE_STOP_TOKENS: u32 = 0x1;
pub const WORK_NO_SPECULATION: u32 = 0x2;
/// M2.9L (m29l design note D5): suffix-lookup drafts may be verified; MTP proposals never are.
pub const WORK_LOOKUP_DRAFTS: u32 = 0x4;
pub const WORK_NO_MTP_DRAFTS: u32 = 0x8;
/// m29a design note A2: the session's policy picks each round's draft count, priced by the calibrated table.
pub const WORK_ADAPTIVE_DRAFTS: u32 = 0x10;
/// DraftAccept: a greedy adaptive lookup ticket may append the lookup tail past three agreeing MTP drafts.
pub const WORK_TAIL_DRAFTS: u32 = 0x20;
/// TreeSpec: a greedy adaptive lookup ticket may verify a long lookup proposal (up to 8 drafts) on its own.
pub const WORK_LONG_DRAFTS: u32 = 0x40;
/// Chains (O123): bits 5 and 6 may build chains of up to the model's max chain drafts instead of 8.
pub const WORK_CHAIN_DRAFTS: u32 = 0x80;
pub const WORK_KNOWN_FLAGS: u32 = 0xFF;

/// `cere_slot_choice.slot` (ABI v5, selftune design note ST2): one per registry executor lever.
pub const SLOT_K2_GEMM: u32 = 1;
pub const SLOT_PREFILL_ATTENTION: u32 = 2;
pub const SLOT_DECODE_STEP: u32 = 3;
pub const SLOT_MOE_GEMM: u32 = 4;
pub const SLOT_DRAFT_TOKENS: u32 = 5;
pub const SLOT_VERIFY_ATTENTION: u32 = 6;
pub const SLOT_K2_WEIGHTS: u32 = 7;
/// Slots 1..=SLOT_COUNT exist: the slots every family shares (the Qwen levers).
pub const SLOT_COUNT: u32 = 7;
/// SdarOptIn (O320): a family-scoped slot of family `sdar` (`sdar_gemm`); family-scoped slots start past
/// [`SLOT_COUNT`], which counts the shared slots only. A `qwen4_exp` library refuses it (`CERE_ERR_ARGUMENT`).
pub const SLOT_SDAR_GEMM: u32 = 8;
/// `cere_slot_selection.block_version`.
pub const SLOT_BLOCK_VERSION: u32 = 1;
/// Capacity of `cere_slot_selection.choices` and of `cere_slot_choice.params`.
pub const SLOT_MAX: usize = 16;
pub const SLOT_PARAMS: usize = 4;
/// `cere_model_desc.family` (SelfTune ST4): the struct `config` points at, `cere_qwen4_exp_config` ([`Qwen4ExpConfig`]),
/// `cere_sdar_config` ([`SdarConfig`]) or `cere_inkling2_config` ([`Inkling2Config`]), the last two cast by the caller.
/// A library is built for one family. 2 is build/qwen35-4b's `CERE_FAMILY_QWEN3_5`.
pub const FAMILY_QWEN4_EXP: u32 = 0;
pub const FAMILY_SDAR: u32 = 1;
pub const FAMILY_INKLING2: u32 = 3;

/// `cere_slot_choice.impl` values (m23 design note PI-03); `crate::levers` maps the registry's impl ids onto them.
pub const LEVER_K2_CUBLASLT: u32 = 0;
pub const LEVER_K2_SMALL_M: u32 = 1;
pub const LEVER_K2_SMALL_M_A8: u32 = 2;
/// VulkanAmd P4-B: the Vulkan K2 GEMM's tail split (Vulkan library only; any other refuses it as UNSUPPORTED).
pub const LEVER_K2_TAIL_SPLIT: u32 = 3;
pub const LEVER_PREFILL_CUDA_CORE: u32 = 0;
pub const LEVER_PREFILL_P4: u32 = 1;
pub const LEVER_DECODE_EAGER: u32 = 0;
/// P1's decode CUDA graph (eligible plain decode steps replay it, bit-identical to eager).
pub const LEVER_DECODE_GRAPH: u32 = 1;
/// `LEVER_DECODE_GRAPH` plus the speculation cycle (VERIFY, MTP draft-extend and draft steps) from per-shape graphs.
pub const LEVER_DECODE_SPEC_GRAPH: u32 = 2;
/// The MoE backend the library was compiled with; `cered` always names the backend explicitly.
pub const LEVER_MOE_BUILD: u32 = 0;
pub const LEVER_MOE_P3B: u32 = 1;
pub const LEVER_MOE_K7: u32 = 2;
/// The verify chain's full-attention cores: one one-token call per verify row, or one batched launch per layer
/// (bit-identical).
pub const LEVER_VERIFY_PER_ROW: u32 = 0;
pub const LEVER_VERIFY_BATCHED: u32 = 1;
/// The dense K2 weights' storage, fixed at `cere_model_create`: the checkpoint's BF16 weights, or E4M3 with FP32
/// per-row scales (numerics-changing; a `cere_model_set_selection` change is `CERE_ERR_UNSUPPORTED`).
pub const LEVER_K2_WEIGHTS_BF16: u32 = 0;
pub const LEVER_K2_WEIGHTS_FP8: u32 = 1;
/// `sdar_gemm`: the sdar library's bit-identical cuBLASLt pin table, or its split-K configs (numerics-changing,
/// opt-in; restart class).
pub const LEVER_SDAR_GEMM_IDENTICAL: u32 = 0;
pub const LEVER_SDAR_GEMM_CHANGING: u32 = 1;
/// The draft_tokens slot's one impl; its `params[0]` is the verify window cap (0 = the model's
/// `speculative_draft_tokens`).
pub const LEVER_DRAFT_WINDOW: u32 = 0;
/// LoadCrit LC-01 (`cere_model_desc.calibrate_windows`, not a slot): the speculation windows `cere_model_seal` times
/// for the adaptive cost tables, fixed at `cere_model_create`: every window the selection and the device's chains
/// allow, or the MTP windows 1..4 only (output-neutral: the entries 1..4 are measured as under ALL).
pub const CALIBRATE_ALL: u32 = 0;
pub const CALIBRATE_MTP: u32 = 1;

/// `cere_work.top_logprobs` upper bound (ledger table K).
pub const MAX_TOP_LOGPROBS: u32 = 20;
/// `cere_sampling_params.flags` (ledger table F2): no bits assigned.
pub const SAMPLING_KNOWN_FLAGS: u32 = 0;
/// `cere_model_desc.prefix_state_slots` upper bound (M2.10 D6, ledger D 104).
pub const MAX_PREFIX_STATE_SLOTS: u64 = 1024;
/// `cere_executor_set_decode_rounds` upper bound (O79, m210 design note PF1).
pub const MAX_DECODE_ROUNDS_PER_CHUNK: u32 = 64;
/// `cere_executor_set_prefill_slice` upper bound in ms (MixedPD, mixed-pd design note).
pub const MAX_PREFILL_SLICE_MS: u32 = 10_000;
/// `cere_executor_set_prefill_age` default and upper bound in overtaking prefill calls (BC4 (b),
/// batch-concurrency design note).
pub const DEFAULT_PREFILL_AGE_CALLS: u32 = 8;
pub const MAX_PREFILL_AGE_CALLS: u32 = 1024;
/// The most drafts `cere_executor_round_drafts` freezes for a constrained ticket's round (grammar-spec design note
/// GS-01): `vocab_masks` of `cere_executor_constrain_drafts` holds at most this many plus one masks.
pub const MAX_CONSTRAINED_DRAFTS: u32 = 3;
/// LoRA adapters (track LoRA, `cere_model_lora_reserve` / `_load` / `_unload`, `cere_session_set_lora`): the pool's
/// bounds and the target modules; a target id is `layer * LORA_MODULES + module`.
pub const LORA_MAX_ADAPTERS: u32 = 64;
pub const LORA_MAX_RANK: u32 = 256;
pub const LORA_MODULES: u32 = 8;
pub const LORA_Q_PROJ: u32 = 0;
pub const LORA_K_PROJ: u32 = 1;
pub const LORA_V_PROJ: u32 = 2;
pub const LORA_O_PROJ: u32 = 3;
pub const LORA_SHARED_GATE: u32 = 4;
pub const LORA_SHARED_UP: u32 = 5;
pub const LORA_SHARED_DOWN: u32 = 6;
/// `cere_executor_set_decode_share` upper bound in percent (S1 G4, beat-sglang design note).
pub const MAX_DECODE_SHARE_PCT: u32 = 1000;
/// WeightLab research (`cere_model_research_*`, `cere_session_set_research`, weightlab design note WL-02): bounds,
/// the final-layer bit, op kinds and capture dtypes.
pub const RESEARCH_MAX_PLANS: u32 = 64;
pub const RESEARCH_MAX_OPS: u32 = 128;
pub const RESEARCH_MAX_RANGES: u32 = 64;
pub const RESEARCH_MAX_TOPK: u32 = 64;
pub const RESEARCH_MAX_LOGIT_POSITIONS: u32 = 8;
pub const RESEARCH_MAX_PLAN_BYTES: u64 = 4_194_304;
pub const RESEARCH_FINAL_LAYER: u32 = 48;
pub const RESEARCH_CAPTURE: u32 = 0;
pub const RESEARCH_ADD: u32 = 1;
pub const RESEARCH_ABLATE_ZERO: u32 = 2;
pub const RESEARCH_ABLATE_MEAN: u32 = 3;
pub const RESEARCH_REPLACE: u32 = 4;
pub const RESEARCH_EXPERT_BIAS: u32 = 5;
pub const RESEARCH_EXPERT_MASK: u32 = 6;
pub const RESEARCH_EXPERT_FORCE: u32 = 7;
pub const RESEARCH_DTYPE_BF16: u32 = 0;
pub const RESEARCH_DTYPE_I32: u32 = 1;
pub const RESEARCH_DTYPE_F32: u32 = 2;

pub const EXECUTION_MODE_NATIVE: u32 = 1;
pub const EXECUTION_MODE_BACKEND_MANAGED: u32 = 2;
pub const EXECUTION_MODE_SERIAL: u32 = 3;
pub const EXECUTION_MODE_UNSUPPORTED: u32 = 4;

pub const MODEL_QWEN4_EXP_NVFP4: u32 = 1;
/// SelfTune ST4: DICE-8B's plain BF16 checkpoint (family sdar).
pub const MODEL_SDAR_BF16: u32 = 2;
/// Inkling 2 Mini (inkling design note): NVFP4 + MXFP8 experts as stored (family inkling2_loop).
pub const MODEL_INKLING2_NVFP4: u32 = 4;

pub const PLACEMENT_DEVICE: u32 = 1;
pub const PLACEMENT_HOST_PINNED: u32 = 2;
pub const PLACEMENT_DISCARD: u32 = 3;

pub const DTYPE_BF16: u32 = 1;
pub const DTYPE_F16: u32 = 2;
pub const DTYPE_F32: u32 = 3;
pub const DTYPE_F64: u32 = 4;
pub const DTYPE_F8_E4M3: u32 = 5;
pub const DTYPE_F8_E5M2: u32 = 6;
pub const DTYPE_U8: u32 = 7;
pub const DTYPE_I8: u32 = 8;
pub const DTYPE_I16: u32 = 9;
pub const DTYPE_I32: u32 = 10;
pub const DTYPE_I64: u32 = 11;
pub const DTYPE_BOOL: u32 = 12;

pub const OUTPUT_GATE_SIGMOID: u32 = 1;

/// `cere_qwen4_exp_vision_config.hidden_act` (ledger K): the tanh GELU (Qwen4Exp) and the erf GELU (Inkling).
pub const ACT_GELU_TANH: u32 = 1;
pub const ACT_GELU_ERF: u32 = 2;

pub const FINISH_NONE: u32 = 0;
pub const FINISH_LENGTH: u32 = 1;
pub const FINISH_STOP_TOKEN: u32 = 2;
pub const FINISH_CANCELLED: u32 = 3;
pub const FINISH_ERROR: u32 = 4;

pub const TENSOR_MAX_DIMS: usize = 8;

/// Every header constant by its C name (compared with the layout fixture).
pub const CONSTANTS: [(&str, i64); 131] = [
    ("CERE_EXECUTOR_ABI_VERSION", ABI_VERSION as i64),
    ("CERE_OK", OK as i64),
    ("CERE_PENDING", PENDING as i64),
    ("CERE_ERR_ARGUMENT", ERR_ARGUMENT as i64),
    ("CERE_ERR_UNSUPPORTED", ERR_UNSUPPORTED as i64),
    ("CERE_ERR_CAPACITY", ERR_CAPACITY as i64),
    ("CERE_ERR_STALE_EPOCH", ERR_STALE_EPOCH as i64),
    ("CERE_ERR_BUSY", ERR_BUSY as i64),
    ("CERE_ERR_INTERNAL", ERR_INTERNAL as i64),
    ("CERE_OP_PREFILL", OP_PREFILL as i64),
    ("CERE_OP_DECODE", OP_DECODE as i64),
    ("CERE_OP_VERIFY", OP_VERIFY as i64),
    ("CERE_OPS_PREFILL", OPS_PREFILL as i64),
    ("CERE_OPS_DECODE", OPS_DECODE as i64),
    ("CERE_OPS_VERIFY", OPS_VERIFY as i64),
    ("CERE_OPS_CANCEL", OPS_CANCEL as i64),
    ("CERE_OPS_SNAPSHOT", OPS_SNAPSHOT as i64),
    ("CERE_OPS_RESTORE", OPS_RESTORE as i64),
    (
        "CERE_WORK_IGNORE_STOP_TOKENS",
        WORK_IGNORE_STOP_TOKENS as i64,
    ),
    ("CERE_WORK_NO_SPECULATION", WORK_NO_SPECULATION as i64),
    ("CERE_WORK_LOOKUP_DRAFTS", WORK_LOOKUP_DRAFTS as i64),
    ("CERE_WORK_NO_MTP_DRAFTS", WORK_NO_MTP_DRAFTS as i64),
    ("CERE_WORK_ADAPTIVE_DRAFTS", WORK_ADAPTIVE_DRAFTS as i64),
    ("CERE_WORK_TAIL_DRAFTS", WORK_TAIL_DRAFTS as i64),
    ("CERE_WORK_LONG_DRAFTS", WORK_LONG_DRAFTS as i64),
    ("CERE_WORK_CHAIN_DRAFTS", WORK_CHAIN_DRAFTS as i64),
    ("CERE_WORK_KNOWN_FLAGS", WORK_KNOWN_FLAGS as i64),
    ("CERE_LEVER_K2_CUBLASLT", LEVER_K2_CUBLASLT as i64),
    ("CERE_LEVER_K2_SMALL_M", LEVER_K2_SMALL_M as i64),
    ("CERE_LEVER_K2_SMALL_M_A8", LEVER_K2_SMALL_M_A8 as i64),
    ("CERE_LEVER_K2_TAIL_SPLIT", LEVER_K2_TAIL_SPLIT as i64),
    (
        "CERE_LEVER_PREFILL_CUDA_CORE",
        LEVER_PREFILL_CUDA_CORE as i64,
    ),
    ("CERE_LEVER_PREFILL_P4", LEVER_PREFILL_P4 as i64),
    ("CERE_LEVER_DECODE_EAGER", LEVER_DECODE_EAGER as i64),
    ("CERE_LEVER_DECODE_GRAPH", LEVER_DECODE_GRAPH as i64),
    (
        "CERE_LEVER_DECODE_SPEC_GRAPH",
        LEVER_DECODE_SPEC_GRAPH as i64,
    ),
    ("CERE_LEVER_MOE_BUILD", LEVER_MOE_BUILD as i64),
    ("CERE_LEVER_MOE_P3B", LEVER_MOE_P3B as i64),
    ("CERE_LEVER_MOE_K7", LEVER_MOE_K7 as i64),
    ("CERE_LEVER_VERIFY_PER_ROW", LEVER_VERIFY_PER_ROW as i64),
    ("CERE_LEVER_VERIFY_BATCHED", LEVER_VERIFY_BATCHED as i64),
    ("CERE_LEVER_K2_WEIGHTS_BF16", LEVER_K2_WEIGHTS_BF16 as i64),
    ("CERE_LEVER_K2_WEIGHTS_FP8", LEVER_K2_WEIGHTS_FP8 as i64),
    (
        "CERE_LEVER_SDAR_GEMM_IDENTICAL",
        LEVER_SDAR_GEMM_IDENTICAL as i64,
    ),
    (
        "CERE_LEVER_SDAR_GEMM_CHANGING",
        LEVER_SDAR_GEMM_CHANGING as i64,
    ),
    ("CERE_LEVER_DRAFT_WINDOW", LEVER_DRAFT_WINDOW as i64),
    ("CERE_CALIBRATE_ALL", CALIBRATE_ALL as i64),
    ("CERE_CALIBRATE_MTP", CALIBRATE_MTP as i64),
    ("CERE_FAMILY_QWEN4_EXP", FAMILY_QWEN4_EXP as i64),
    ("CERE_FAMILY_SDAR", FAMILY_SDAR as i64),
    ("CERE_FAMILY_INKLING2", FAMILY_INKLING2 as i64),
    ("CERE_SLOT_K2_GEMM", SLOT_K2_GEMM as i64),
    ("CERE_SLOT_PREFILL_ATTENTION", SLOT_PREFILL_ATTENTION as i64),
    ("CERE_SLOT_DECODE_STEP", SLOT_DECODE_STEP as i64),
    ("CERE_SLOT_MOE_GEMM", SLOT_MOE_GEMM as i64),
    ("CERE_SLOT_DRAFT_TOKENS", SLOT_DRAFT_TOKENS as i64),
    ("CERE_SLOT_VERIFY_ATTENTION", SLOT_VERIFY_ATTENTION as i64),
    ("CERE_SLOT_K2_WEIGHTS", SLOT_K2_WEIGHTS as i64),
    ("CERE_SLOT_SDAR_GEMM", SLOT_SDAR_GEMM as i64),
    ("CERE_SLOT_COUNT", SLOT_COUNT as i64),
    ("CERE_SLOT_BLOCK_VERSION", SLOT_BLOCK_VERSION as i64),
    ("CERE_SLOT_MAX", SLOT_MAX as i64),
    ("CERE_SLOT_PARAMS", SLOT_PARAMS as i64),
    ("CERE_EXECUTION_MODE_NATIVE", EXECUTION_MODE_NATIVE as i64),
    (
        "CERE_EXECUTION_MODE_BACKEND_MANAGED",
        EXECUTION_MODE_BACKEND_MANAGED as i64,
    ),
    ("CERE_EXECUTION_MODE_SERIAL", EXECUTION_MODE_SERIAL as i64),
    (
        "CERE_EXECUTION_MODE_UNSUPPORTED",
        EXECUTION_MODE_UNSUPPORTED as i64,
    ),
    ("CERE_MODEL_QWEN4_EXP_NVFP4", MODEL_QWEN4_EXP_NVFP4 as i64),
    ("CERE_MODEL_SDAR_BF16", MODEL_SDAR_BF16 as i64),
    ("CERE_MODEL_INKLING2_NVFP4", MODEL_INKLING2_NVFP4 as i64),
    ("CERE_PLACEMENT_DEVICE", PLACEMENT_DEVICE as i64),
    ("CERE_PLACEMENT_HOST_PINNED", PLACEMENT_HOST_PINNED as i64),
    ("CERE_PLACEMENT_DISCARD", PLACEMENT_DISCARD as i64),
    ("CERE_DTYPE_BF16", DTYPE_BF16 as i64),
    ("CERE_DTYPE_F16", DTYPE_F16 as i64),
    ("CERE_DTYPE_F32", DTYPE_F32 as i64),
    ("CERE_DTYPE_F64", DTYPE_F64 as i64),
    ("CERE_DTYPE_F8_E4M3", DTYPE_F8_E4M3 as i64),
    ("CERE_DTYPE_F8_E5M2", DTYPE_F8_E5M2 as i64),
    ("CERE_DTYPE_U8", DTYPE_U8 as i64),
    ("CERE_DTYPE_I8", DTYPE_I8 as i64),
    ("CERE_DTYPE_I16", DTYPE_I16 as i64),
    ("CERE_DTYPE_I32", DTYPE_I32 as i64),
    ("CERE_DTYPE_I64", DTYPE_I64 as i64),
    ("CERE_DTYPE_BOOL", DTYPE_BOOL as i64),
    ("CERE_OUTPUT_GATE_SIGMOID", OUTPUT_GATE_SIGMOID as i64),
    ("CERE_ACT_GELU_TANH", ACT_GELU_TANH as i64),
    ("CERE_ACT_GELU_ERF", ACT_GELU_ERF as i64),
    ("CERE_FINISH_NONE", FINISH_NONE as i64),
    ("CERE_FINISH_LENGTH", FINISH_LENGTH as i64),
    ("CERE_FINISH_STOP_TOKEN", FINISH_STOP_TOKEN as i64),
    ("CERE_FINISH_CANCELLED", FINISH_CANCELLED as i64),
    ("CERE_FINISH_ERROR", FINISH_ERROR as i64),
    ("CERE_TENSOR_MAX_DIMS", TENSOR_MAX_DIMS as i64),
    ("CERE_MAX_TOP_LOGPROBS", MAX_TOP_LOGPROBS as i64),
    ("CERE_SAMPLING_KNOWN_FLAGS", SAMPLING_KNOWN_FLAGS as i64),
    ("CERE_MAX_PREFIX_STATE_SLOTS", MAX_PREFIX_STATE_SLOTS as i64),
    (
        "CERE_MAX_DECODE_ROUNDS_PER_CHUNK",
        MAX_DECODE_ROUNDS_PER_CHUNK as i64,
    ),
    ("CERE_MAX_PREFILL_SLICE_MS", MAX_PREFILL_SLICE_MS as i64),
    (
        "CERE_DEFAULT_PREFILL_AGE_CALLS",
        DEFAULT_PREFILL_AGE_CALLS as i64,
    ),
    ("CERE_MAX_PREFILL_AGE_CALLS", MAX_PREFILL_AGE_CALLS as i64),
    ("CERE_MAX_CONSTRAINED_DRAFTS", MAX_CONSTRAINED_DRAFTS as i64),
    ("CERE_LORA_MAX_ADAPTERS", LORA_MAX_ADAPTERS as i64),
    ("CERE_LORA_MAX_RANK", LORA_MAX_RANK as i64),
    ("CERE_LORA_MODULES", LORA_MODULES as i64),
    ("CERE_LORA_Q_PROJ", LORA_Q_PROJ as i64),
    ("CERE_LORA_K_PROJ", LORA_K_PROJ as i64),
    ("CERE_LORA_V_PROJ", LORA_V_PROJ as i64),
    ("CERE_LORA_O_PROJ", LORA_O_PROJ as i64),
    ("CERE_LORA_SHARED_GATE", LORA_SHARED_GATE as i64),
    ("CERE_LORA_SHARED_UP", LORA_SHARED_UP as i64),
    ("CERE_LORA_SHARED_DOWN", LORA_SHARED_DOWN as i64),
    ("CERE_MAX_DECODE_SHARE_PCT", MAX_DECODE_SHARE_PCT as i64),
    ("CERE_RESEARCH_MAX_PLANS", RESEARCH_MAX_PLANS as i64),
    ("CERE_RESEARCH_MAX_OPS", RESEARCH_MAX_OPS as i64),
    ("CERE_RESEARCH_MAX_RANGES", RESEARCH_MAX_RANGES as i64),
    ("CERE_RESEARCH_MAX_TOPK", RESEARCH_MAX_TOPK as i64),
    (
        "CERE_RESEARCH_MAX_LOGIT_POSITIONS",
        RESEARCH_MAX_LOGIT_POSITIONS as i64,
    ),
    (
        "CERE_RESEARCH_MAX_PLAN_BYTES",
        RESEARCH_MAX_PLAN_BYTES as i64,
    ),
    ("CERE_RESEARCH_FINAL_LAYER", RESEARCH_FINAL_LAYER as i64),
    ("CERE_RESEARCH_CAPTURE", RESEARCH_CAPTURE as i64),
    ("CERE_RESEARCH_ADD", RESEARCH_ADD as i64),
    ("CERE_RESEARCH_ABLATE_ZERO", RESEARCH_ABLATE_ZERO as i64),
    ("CERE_RESEARCH_ABLATE_MEAN", RESEARCH_ABLATE_MEAN as i64),
    ("CERE_RESEARCH_REPLACE", RESEARCH_REPLACE as i64),
    ("CERE_RESEARCH_EXPERT_BIAS", RESEARCH_EXPERT_BIAS as i64),
    ("CERE_RESEARCH_EXPERT_MASK", RESEARCH_EXPERT_MASK as i64),
    ("CERE_RESEARCH_EXPERT_FORCE", RESEARCH_EXPERT_FORCE as i64),
    ("CERE_RESEARCH_DTYPE_BF16", RESEARCH_DTYPE_BF16 as i64),
    ("CERE_RESEARCH_DTYPE_I32", RESEARCH_DTYPE_I32 as i64),
    ("CERE_RESEARCH_DTYPE_F32", RESEARCH_DTYPE_F32 as i64),
];

/// Opaque handles; only pointers to them cross the boundary.
#[repr(C)]
pub struct RawExecutor {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

#[repr(C)]
pub struct RawModel {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

#[repr(C)]
pub struct RawSession {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// `cere_logit_bias` (ledger table N, M2.7).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LogitBias {
    pub id: u32,
    pub bias: f32,
}

/// `cere_sampling_params` (ledger table N, M2.7; the M2.12 tail group is zero-required).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub struct_size: u32,
    pub abi_version: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: u32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub flags: u32,
    pub seed: u64,
    pub logit_bias: *const LogitBias,
    pub logit_bias_count: u64,
    pub suppress_token_ids: *const u32,
    pub suppress_token_count: u64,
    pub mask_until_committed: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LogprobEntry {
    pub id: u32,
    pub logprob: f32,
}

/// `cere_token_logprobs`: one committed token's logprob and its `top_count` alternatives.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TokenLogprobs {
    pub id: u32,
    pub logprob: f32,
    pub top_count: u32,
    pub reserved: u32,
    pub top: [LogprobEntry; MAX_TOP_LOGPROBS as usize],
}

/// `cere_snapshot` (M2.11, ledger N): a session snapshot in the model's pinned host arena.
#[repr(C)]
pub struct RawSnapshot {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// `struct cere_image_input` (M2.13, ledger N): one image span of a PREFILL prompt.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ImageInput {
    pub struct_size: u32,
    pub abi_version: u32,
    pub grid_t: u32,
    pub grid_h: u32,
    pub grid_w: u32,
    pub pad_id: u32,
    pub token_offset: u64,
    pub pixel_patches: *const c_void,
    pub pixel_bytes: u64,
    /// VisionLong C1: SHA-256 of exactly the `pixel_bytes` bytes at `pixel_patches` (all-zero is refused).
    pub pixel_sha256: [u8; 32],
}

/// `struct cere_qwen4_exp_vision_config` (M2.13, ledger N).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen4ExpVisionConfig {
    pub struct_size: u32,
    pub abi_version: u32,
    pub depth: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_heads: u32,
    pub patch_size: u32,
    pub temporal_patch_size: u32,
    pub spatial_merge_size: u32,
    pub in_channels: u32,
    pub out_hidden_size: u32,
    pub num_position_embeddings: u32,
    pub deepstack_count: u32,
    pub hidden_act: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub layer_norm_eps: f64,
}

/// `struct_size` of an ABI struct.
pub fn struct_size<T>() -> u32 {
    std::mem::size_of::<T>() as u32
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ExecutorOptions {
    pub struct_size: u32,
    pub abi_version: u32,
    pub device_id: *const c_char,
    pub worker_epoch: u64,
    pub memory_budget_bytes: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExecutionCapabilities {
    pub struct_size: u32,
    pub abi_version: u32,
    pub profile_generation: u64,
    pub worker_epoch: u64,
    pub decode_batch_min: u64,
    pub decode_batch_max: u64,
    pub max_context_tokens: u64,
    pub execution_mode: u32,
    pub native_decode_batch: u8,
    pub native_mixed_prefill_decode: u8,
    pub native_speculative_batch: u8,
    pub native_moe_grouping: u8,
    pub native_state_snapshot: u8,
    pub native_sampler: u8,
    pub native_token_mask: u8,
    pub speculative_draft_tokens: u8,
    /// M2.10 (ledger C 60): 1 when the model was created with `prefix_state_slots` > 0.
    pub native_prefix_cache: u8,
    pub native_image_input: u8,
    pub reserved: [u8; 2],
    pub operations: u64,
    pub kv_pool_tokens: u64,
    pub kv_page_tokens: u32,
    pub reserved2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CompletionTicket {
    pub worker_epoch: u64,
    pub model_generation: u64,
    pub submission_id: u64,
}

/// The model's `text_config` values, fixed width (header comment: absent key 0, `norm_topk_prob` absent 1, lists as
/// bitmasks, strings as enums).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Qwen4ExpConfig {
    pub struct_size: u32,
    pub abi_version: u32,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub full_attention_interval: u32,
    pub full_attention_layers: u64,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub output_gate_type: u32,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub mrope_interleaved: u32,
    pub mrope_section: [u32; 3],
    pub indexer_n_heads: u32,
    pub indexer_head_dim: u32,
    pub indexer_kv_heads: u32,
    pub indexer_compress_ratio: u32,
    pub indexer_budget: u32,
    pub linear_num_key_heads: u32,
    pub linear_num_value_heads: u32,
    pub linear_key_head_dim: u32,
    pub linear_value_head_dim: u32,
    pub linear_conv_kernel_dim: u32,
    pub mamba_ssm_dtype: u32,
    pub num_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: u32,
    pub shared_expert_intermediate_size: u32,
    pub norm_topk_prob: u32,
    pub hc_count: u32,
    pub hc_lowrank: u32,
    pub ple_layer_ids: u64,
    pub ple_embed_dim: u32,
    pub heads_per_ngram: u32,
    pub ngram_size: u32,
    pub ple_conv_kernel_size: u32,
    pub ngram_vocab_size_base: u64,
    pub split_ngram_parts: u32,
    pub make_ngram_vocab_size_divisible_by: u32,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: u32,
    pub mtp_num_hidden_layers: u32,
    pub max_position_embeddings: u64,
}

/// `cere_sdar_config` (SelfTune ST4, family sdar): DICE-8B's config.json values, fixed width; every field must equal
/// the library's compiled descriptor values.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SdarConfig {
    pub struct_size: u32,
    pub abi_version: u32,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub mask_token_id: u32,
    pub tie_word_embeddings: u32,
    pub max_position_embeddings: u64,
}

/// `cere_inkling2_config` (inkling design note S0, family inkling2_loop): Inkling 2 Mini's config.json values, fixed
/// width; every field must equal the library's compiled descriptor values. `local_layer_mask` bit i: layer i is
/// sliding-window attention; `mxfp8_layer_mask` bit i: layer i's experts are MXFP8 (else NVFP4).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Inkling2Config {
    pub struct_size: u32,
    pub abi_version: u32,
    pub vocab_size: u32,
    pub unpadded_vocab_size: u32,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub moe_intermediate_size: u32,
    pub num_experts: u32,
    pub num_experts_per_tok: u32,
    pub sliding_window_size: u32,
    pub rel_extent: u32,
    pub d_rel: u32,
    pub loop_core_start: u32,
    pub loop_core_end: u32,
    pub loop_num_passes: u32,
    pub loop_gate_rank: u32,
    pub n_layers_per_attn_res_block: u32,
    pub conv_kernel_size: u32,
    pub eos_token_id: u32,
    pub local_layer_mask: u32,
    pub mxfp8_layer_mask: u32,
    pub rms_norm_eps: f64,
    pub final_logit_softcapping: f64,
    pub route_scale: f64,
    pub max_position_embeddings: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorDesc {
    pub struct_size: u32,
    pub abi_version: u32,
    pub name: *const c_char,
    pub dtype: u32,
    pub placement: u32,
    pub ndim: u32,
    pub reserved: u32,
    pub shape: [u64; TENSOR_MAX_DIMS],
    pub bytes: u64,
}

/// `cere_slot_choice` (ABI v5): one slot's impl and its typed params (only the draft_tokens slot has one: the window).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlotChoice {
    pub slot: u32,
    pub r#impl: u32,
    pub params: [u32; SLOT_PARAMS],
}

/// `cere_slot_selection` (ABI v5, selftune design note ST2): which kernel/path runs over the same weights and
/// buffers, one choice per listed slot (an unlisted slot runs its default, impl 0).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlotSelection {
    pub struct_size: u32,
    pub abi_version: u32,
    pub block_version: u32,
    pub count: u32,
    pub choices: [SlotChoice; SLOT_MAX],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ModelDesc {
    pub struct_size: u32,
    pub abi_version: u32,
    pub model_kind: u32,
    pub speculative_draft_tokens: u32,
    /// The pointee follows `family`: a [`SdarConfig`] cast to this type when `family` is [`FAMILY_SDAR`].
    pub config: *const Qwen4ExpConfig,
    pub tensors: *const TensorDesc,
    pub tensor_count: u64,
    pub max_sessions: u64,
    pub max_context_tokens: u64,
    pub stop_token_ids: *const u32,
    pub stop_token_count: u64,
    pub kv_pool_tokens: u64,
    pub draft_token_map: *const u32,
    pub draft_token_map_count: u64,
    pub snapshot_host_bytes: u64,
    /// M2.10 D6 (ledger D 104): device state slots of the prefix cache; 0 = cache off (M2.5's eager page take),
    /// at most [`MAX_PREFIX_STATE_SLOTS`].
    pub prefix_state_slots: u64,
    pub vision_config: *const Qwen4ExpVisionConfig,
    /// Null = every slot at its default (impl 0); `cered` always passes its resolved set.
    pub selection: *const SlotSelection,
    /// `FAMILY_*`: the struct `config` points at.
    pub family: u32,
    /// `CALIBRATE_*`, fixed at create (derived from the `drafter` lever).
    pub calibrate_windows: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TokenScore {
    pub id: u32,
    pub runner_up: u32,
    pub logit: f32,
    pub runner_up_logit: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Work {
    pub struct_size: u32,
    pub abi_version: u32,
    pub operation: u32,
    pub flags: u32,
    pub expected_worker_epoch: u64,
    pub expected_session_revision: u64,
    pub session: *mut RawSession,
    pub tokens: *const u32,
    pub token_count: u64,
    pub max_new_tokens: u64,
    pub reserved_peak_bytes: u64,
    pub out_tokens: *mut u32,
    pub out_capacity: u64,
    pub out_scores: *mut TokenScore,
    pub stop_token_ids: *const u32,
    pub stop_token_count: u64,
    pub sampling: *const SamplingParams,
    pub out_logprobs: *mut TokenLogprobs,
    pub top_logprobs: u32,
    /// M2.10 D4 (c) (ledger W 132): PREFILL only; 0 = none, else a multiple of 64 inside `(0, token_count)` (else
    /// `CERE_ERR_ARGUMENT`, also non-zero on DECODE/VERIFY); a checkpoint hint, ignored without a namespace.
    pub capture_position: u32,
    pub vocab_mask: *const u32,
    pub vocab_mask_words: u64,
    pub images: *const ImageInput,
    pub image_count: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Batch {
    pub struct_size: u32,
    pub abi_version: u32,
    pub expected_worker_epoch: u64,
    pub capability_profile_generation: u64,
    pub items: *const Work,
    pub item_count: u64,
    pub reserved_peak_bytes: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Completion {
    pub struct_size: u32,
    pub abi_version: u32,
    pub status: Status,
    pub terminal: u32,
    pub committed_tokens: u64,
    pub session_revision: u64,
    pub released_bytes: u64,
    pub finish: u32,
    /// M2.10 D10 (ledger P 44): prompt tokens this PREFILL ticket took from the prefix cache; 0 otherwise.
    pub cached_tokens: u32,
    pub verify_steps: u64,
    pub accepted_drafts: u64,
}

/// `cere_snapshot_info` (M2.11, ledger N: 72 B, one size).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SnapshotInfo {
    pub struct_size: u32,
    pub abi_version: u32,
    pub bytes: u64,
    pub position: u64,
    pub token_count: u64,
    pub fed: u64,
    pub layout_id: [u8; 32],
}

/// `cere_prefix_stats` (appended to ABI v4, 104 B, one size): the prefix cache's cumulative counters of one model.
/// `struct_size`/`abi_version` are set by the caller; `reserved` reads 0.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixStats {
    pub struct_size: u32,
    pub abi_version: u32,
    /// 1 when `cere_model_set_prefix_alt_edges(1)` is in effect.
    pub alt_edges_on: u32,
    pub reserved: u32,
    pub hits: u64,
    pub captures: u64,
    pub capture_skipped: u64,
    pub capture_page_mismatch: u64,
    pub inserts: u64,
    pub state_evictions: u64,
    pub leaf_evictions: u64,
    pub host_spills: u64,
    pub host_loads: u64,
    pub host_drops: u64,
    pub alt_edges: u64,
}

/// `cere_prefix_capture_stats` (appended to ABI v4, 48 B, one size; prefix-token design note PT0a): `capture_skipped`
/// by site (the three sum to it) and the drafting lookups' lost hit tokens. `struct_size`/`abi_version` are set by
/// the caller.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCaptureStats {
    pub struct_size: u32,
    pub abi_version: u32,
    /// A prefill capture found no free or evictable slot.
    pub skipped_prefill_slot: u64,
    /// A session's decode track capture found no slot: once per ticket.
    pub skipped_decode_track: u64,
    /// An end-state capture found no slot pair.
    pub skipped_end_pair: u64,
    /// Per drafting lookup: tokens between its hit and the deepest prefill-chunk multiple its page match passed.
    pub hit_shortfall_tokens: u64,
    /// Drafting lookups with a nonzero shortfall.
    pub hit_shortfall_lookups: u64,
}

/// `cere_research_range` (WeightLab): absolute positions `[first, end)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResearchRange {
    pub first: u64,
    pub end: u64,
}

/// `cere_research_op` (WeightLab): one capture or intervention op of a plan; every pointer is read before
/// `cere_model_research_plan_load` returns.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ResearchOp {
    pub kind: u32,
    pub top_k: u32,
    pub point: *const c_char,
    pub layers: u64,
    pub prompt: *const ResearchRange,
    pub prompt_count: u32,
    pub generated: u32,
    pub generated_max: u64,
    pub alpha: f32,
    pub col_begin: u32,
    pub col_end: u32,
    pub vector_rows: u32,
    pub vector: *const c_void,
    pub vector_cols: u32,
    pub expert_count: u32,
    pub experts: *const i32,
    pub weights: *const f32,
}

/// `cere_research_plan` (WeightLab).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ResearchPlan {
    pub ops: *const ResearchOp,
    pub op_count: u32,
    pub reserved: u32,
    pub sha256: [u8; 32],
}

/// `cere_research_tensor` (WeightLab): one captured tensor; pointers valid until `cere_model_research_capture_free`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ResearchTensor {
    pub name: *const c_char,
    pub op: u32,
    pub layer: u32,
    pub dtype: u32,
    pub cols: u32,
    pub rows: u64,
    pub positions: *const u64,
    pub data: *const c_void,
}

/// `cere_research_capture` (WeightLab): a finished capture (`cere_model_research_capture_read`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ResearchCapture {
    pub tensors: *const ResearchTensor,
    pub tensor_count: u64,
    pub bytes: u64,
    pub prefix_lookup_skipped: u32,
    pub reserved: u32,
    pub plan_sha256: [u8; 32],
}

impl Default for ResearchCapture {
    fn default() -> ResearchCapture {
        ResearchCapture {
            tensors: std::ptr::null(),
            tensor_count: 0,
            bytes: 0,
            prefix_lookup_skipped: 0,
            reserved: 0,
            plan_sha256: [0; 32],
        }
    }
}

/// The library's entry points, resolved once by [`Library::open`].
#[derive(Debug, Clone, Copy)]
pub struct Api {
    pub abi_version: unsafe extern "C" fn() -> u32,
    pub executor_create:
        unsafe extern "C" fn(*mut *mut RawExecutor, *const ExecutorOptions) -> Status,
    pub executor_destroy: unsafe extern "C" fn(*mut RawExecutor) -> Status,
    pub executor_capabilities: unsafe extern "C" fn(
        *mut RawExecutor,
        *const RawModel,
        *mut ExecutionCapabilities,
    ) -> Status,
    pub model_create:
        unsafe extern "C" fn(*mut RawExecutor, *const ModelDesc, *mut *mut RawModel) -> Status,
    pub model_put: unsafe extern "C" fn(
        *mut RawExecutor,
        *mut RawModel,
        u64,
        u64,
        *const c_void,
        u64,
    ) -> Status,
    pub model_seal: unsafe extern "C" fn(*mut RawExecutor, *mut RawModel) -> Status,
    pub model_release: unsafe extern "C" fn(*mut RawExecutor, *mut RawModel) -> Status,
    pub session_create:
        unsafe extern "C" fn(*mut RawExecutor, *mut RawModel, u64, *mut *mut RawSession) -> Status,
    pub session_release: unsafe extern "C" fn(*mut RawExecutor, *mut RawSession) -> Status,
    pub executor_submit:
        unsafe extern "C" fn(*mut RawExecutor, *const Batch, *mut CompletionTicket) -> Status,
    pub executor_cancel: unsafe extern "C" fn(*mut RawExecutor, CompletionTicket) -> Status,
    pub executor_poll:
        unsafe extern "C" fn(*mut RawExecutor, CompletionTicket, *mut Completion) -> Status,
    /// M2.8 A2: the mask of a constrained ticket's next position (`cere_executor_constrain`).
    pub executor_constrain:
        unsafe extern "C" fn(*mut RawExecutor, CompletionTicket, u64, *const u32, u64) -> Status,
    pub session_snapshot: unsafe extern "C" fn(
        *mut RawExecutor,
        *mut RawSession,
        u64,
        *mut *mut RawSnapshot,
    ) -> Status,
    pub session_restore:
        unsafe extern "C" fn(*mut RawExecutor, *mut RawSession, *const RawSnapshot) -> Status,
    pub snapshot_get_info:
        unsafe extern "C" fn(*mut RawExecutor, *const RawSnapshot, *mut SnapshotInfo) -> Status,
    pub snapshot_read:
        unsafe extern "C" fn(*mut RawExecutor, *const RawSnapshot, u64, *mut c_void, u64) -> Status,
    pub snapshot_import: unsafe extern "C" fn(
        *mut RawExecutor,
        *mut RawModel,
        *const c_void,
        u64,
        *mut *mut RawSnapshot,
    ) -> Status,
    pub snapshot_release: unsafe extern "C" fn(*mut RawExecutor, *mut RawSnapshot) -> Status,
    /// M2.10 D2: the session's 16-byte cache namespace, before its first submit.
    pub session_set_cache_namespace:
        unsafe extern "C" fn(*mut RawExecutor, *mut RawSession, *const [u8; 16]) -> Status,
    /// M2.10 D11: drops every cached prefix of the model (BUSY while a ticket is in flight).
    pub model_flush_prefix_cache: unsafe extern "C" fn(*mut RawExecutor, *mut RawModel) -> Status,
    /// M2.10 D12: the deepest cached state of `tokens` under a namespace as a snapshot in the model's arena.
    pub prefix_cache_export: unsafe extern "C" fn(
        *mut RawExecutor,
        *mut RawModel,
        *const [u8; 16],
        *const u32,
        u64,
        *mut *mut RawSnapshot,
    ) -> Status,
    /// M2.10 D12: inserts an exported snapshot under a namespace.
    pub prefix_cache_import: unsafe extern "C" fn(
        *mut RawExecutor,
        *mut RawModel,
        *const [u8; 16],
        *const RawSnapshot,
    ) -> Status,
    /// M2.16 L4: the attached bytes of a HOST_PINNED tensor, covered without a put (`cere_model_host_resident`).
    pub model_host_resident:
        unsafe extern "C" fn(*mut RawExecutor, *mut RawModel, u64, *mut *const c_void) -> Status,
    /// PI-05: the model's active selection (`cere_model_selection`).
    pub model_selection:
        unsafe extern "C" fn(*mut RawExecutor, *mut RawModel, *mut SlotSelection) -> Status,
    /// PI-04: the dev hot swap between requests (`cere_model_set_selection`; `CERE_ERR_BUSY` while tickets fly).
    pub model_set_selection:
        unsafe extern "C" fn(*mut RawExecutor, *mut RawModel, *const SlotSelection) -> Status,
    /// m216 design note M216S-11: put staging (`cere_executor_register_staging` / `_unregister_staging`).
    /// Appended to ABI v4 without a version bump, so a library built before them lacks both: `None`, and the
    /// upload keeps every put on the library's bounce path.
    pub executor_register_staging:
        Option<unsafe extern "C" fn(*mut RawExecutor, *const c_void, u64) -> Status>,
    pub executor_unregister_staging:
        Option<unsafe extern "C" fn(*mut RawExecutor, *const c_void) -> Status>,
    /// O79 (m210 design note PF1): `cere_executor_set_decode_rounds`, decode rounds per prefill chunk. Appended to
    /// ABI v4 like the staging pair: `None` in a library built before it (the executor keeps alternating).
    pub executor_set_decode_rounds: Option<unsafe extern "C" fn(*mut RawExecutor, u32) -> Status>,
    /// m210 design note D15⁗′ stage 2a: prefix end states on/off (`cere_model_set_prefix_end_state`; `CERE_ERR_BUSY`
    /// once the model had a ticket). Appended to ABI v4 like the staging pair: `None` on a library before it.
    pub model_set_prefix_end_state:
        Option<unsafe extern "C" fn(*mut RawExecutor, *mut RawModel, u32) -> Status>,
    /// m210 design note D17: prefix alternative edges on/off (`cere_model_set_prefix_alt_edges`; `CERE_ERR_BUSY`
    /// once the model had a ticket). Appended to ABI v4 like the end-state setter: `None` on a library before it.
    pub model_set_prefix_alt_edges:
        Option<unsafe extern "C" fn(*mut RawExecutor, *mut RawModel, u32) -> Status>,
    /// The prefix cache's cumulative counters of a model (`cere_model_prefix_stats`; `CERE_ERR_UNSUPPORTED` on a
    /// model without a prefix cache). Appended to ABI v4 after the alt-edges setter: `None` on a library before it.
    pub model_prefix_stats:
        Option<unsafe extern "C" fn(*mut RawExecutor, *mut RawModel, *mut PrefixStats) -> Status>,
    /// MixedPD (mixed-pd design note): `cere_executor_set_prefill_slice`, layer-sliced prefill (ms per segment).
    /// Appended to ABI v4 like the round policy: `None` in a library built before it (every call runs whole).
    pub executor_set_prefill_slice: Option<unsafe extern "C" fn(*mut RawExecutor, u32) -> Status>,
    /// grammar-spec design note GS-01: a constrained ticket's round drafts (`cere_executor_round_drafts`: ticket,
    /// position, drafts, capacity, count) and that round's per-row masks (`cere_executor_constrain_drafts`: ticket,
    /// position, drafts, draft_count, masks, words per mask). Appended to ABI v4 like the staging pair: `None` on a
    /// library before them, and a profile with `grammar_drafts = "on"` then fails to load.
    pub executor_round_drafts: Option<
        unsafe extern "C" fn(
            *mut RawExecutor,
            CompletionTicket,
            u64,
            *mut u32,
            u32,
            *mut u32,
        ) -> Status,
    >,
    pub executor_constrain_drafts: Option<
        unsafe extern "C" fn(
            *mut RawExecutor,
            CompletionTicket,
            u64,
            *const u32,
            u32,
            *const u32,
            u64,
        ) -> Status,
    >,
    /// Track LoRA: the adapter pool (`cere_model_lora_reserve` before seal, `cere_model_lora_load` / `_unload` on the
    /// sealed model, `cere_session_set_lora` before a session's first submit). Appended to ABI v4 like the staging
    /// pair: `None` on a library before them, which serves no adapter (a lora config is refused at startup).
    pub model_lora_reserve:
        Option<unsafe extern "C" fn(*mut RawExecutor, *mut RawModel, u32, u32) -> Status>,
    pub model_lora_load: Option<
        unsafe extern "C" fn(
            *mut RawExecutor,
            *mut RawModel,
            u32,
            u32,
            f32,
            u64,
            *const u32,
            *const *const c_void,
            *const *const c_void,
        ) -> Status,
    >,
    pub model_lora_unload:
        Option<unsafe extern "C" fn(*mut RawExecutor, *mut RawModel, u32) -> Status>,
    pub session_set_lora:
        Option<unsafe extern "C" fn(*mut RawExecutor, *mut RawSession, u32) -> Status>,
    /// PT0a: the capture-skip sites and lost hit tokens (`cere_model_prefix_capture_stats`, same contract).
    /// Appended to ABI v4 after `cere_session_set_lora`: `None` on a library before it.
    pub model_prefix_capture_stats: Option<
        unsafe extern "C" fn(*mut RawExecutor, *mut RawModel, *mut PrefixCaptureStats) -> Status,
    >,
    /// S1 G4 (beat-sglang design note): `cere_executor_set_decode_share`, the decode rows' time share of the last
    /// waiting prompt's prefill (percent). Appended to ABI v4 like the round policy: `None` in a library built before
    /// it (the decode rounds per chunk stay).
    pub executor_set_decode_share: Option<unsafe extern "C" fn(*mut RawExecutor, u32) -> Status>,
    /// WeightLab (weightlab design note WL-02): the research reserve (before seal), plan load / unload, a session's
    /// plan (before its first submit) and a finished capture's read / free. Appended to ABI v4 after
    /// `cere_executor_set_decode_share`: `None` on a library before them (a research config is refused at load).
    pub model_research_reserve:
        Option<unsafe extern "C" fn(*mut RawExecutor, *mut RawModel, u64, u64, u32) -> Status>,
    pub model_research_plan_load: Option<
        unsafe extern "C" fn(*mut RawExecutor, *mut RawModel, u32, *const ResearchPlan) -> Status,
    >,
    pub model_research_plan_unload:
        Option<unsafe extern "C" fn(*mut RawExecutor, *mut RawModel, u32) -> Status>,
    pub session_set_research: Option<
        unsafe extern "C" fn(*mut RawExecutor, *mut RawSession, u32, u64, u64, *mut u64) -> Status,
    >,
    pub model_research_capture_read: Option<
        unsafe extern "C" fn(*mut RawExecutor, *mut RawModel, u64, *mut ResearchCapture) -> Status,
    >,
    pub model_research_capture_free:
        Option<unsafe extern "C" fn(*mut RawExecutor, *mut RawModel, u64) -> Status>,
    /// BC1 leftover (batch-concurrency design note "Prefill selection signature"): `cere_model_prefill_signature`,
    /// the SHA-256 of a sealed model's prefill kernel selection. Appended to ABI v4 after the research functions:
    /// `None` on a library before it (no signature recorded).
    pub model_prefill_signature:
        Option<unsafe extern "C" fn(*mut RawExecutor, *const RawModel, *mut u8) -> Status>,
    /// BC4 (b) (batch-concurrency design note): `cere_executor_set_prefill_age`, the aging cap of the
    /// fewest-tokens-left prefill order (overtaking prefill calls). Appended to ABI v4 after the prefill signature:
    /// `None` on a library before it (the default 8 stays).
    pub executor_set_prefill_age: Option<unsafe extern "C" fn(*mut RawExecutor, u32) -> Status>,
}

/// The exported functions, in header order.
pub const SYMBOLS: [&str; 50] = [
    "cere_executor_abi_version",
    "cere_executor_create",
    "cere_executor_destroy",
    "cere_executor_capabilities",
    "cere_model_create",
    "cere_model_put",
    "cere_model_seal",
    "cere_model_release",
    "cere_session_create",
    "cere_session_release",
    "cere_executor_submit",
    "cere_executor_cancel",
    "cere_executor_poll",
    "cere_executor_constrain",
    "cere_session_snapshot",
    "cere_session_restore",
    "cere_snapshot_get_info",
    "cere_snapshot_read",
    "cere_snapshot_import",
    "cere_snapshot_release",
    "cere_session_set_cache_namespace",
    "cere_model_flush_prefix_cache",
    "cere_prefix_cache_export",
    "cere_prefix_cache_import",
    "cere_model_host_resident",
    "cere_model_selection",
    "cere_model_set_selection",
    "cere_executor_register_staging",
    "cere_executor_unregister_staging",
    "cere_executor_set_decode_rounds",
    "cere_model_set_prefix_end_state",
    "cere_model_set_prefix_alt_edges",
    "cere_model_prefix_stats",
    "cere_executor_set_prefill_slice",
    "cere_executor_round_drafts",
    "cere_executor_constrain_drafts",
    "cere_model_lora_reserve",
    "cere_model_lora_load",
    "cere_model_lora_unload",
    "cere_session_set_lora",
    "cere_model_prefix_capture_stats",
    "cere_executor_set_decode_share",
    "cere_model_research_reserve",
    "cere_model_research_plan_load",
    "cere_model_research_plan_unload",
    "cere_session_set_research",
    "cere_model_research_capture_read",
    "cere_model_research_capture_free",
    "cere_model_prefill_signature",
    "cere_executor_set_prefill_age",
];

/// A non-OK status returned by the library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusError {
    /// `CERE_PENDING` where a call must complete.
    Pending,
    Argument,
    Unsupported,
    Capacity,
    StaleEpoch,
    Busy,
    Internal,
    /// A code the header does not define.
    Unknown(Status),
}

impl StatusError {
    /// `Ok` for `CERE_OK`, the typed error otherwise.
    pub fn check(code: Status) -> Result<(), StatusError> {
        Err(match code {
            OK => return Ok(()),
            PENDING => StatusError::Pending,
            ERR_ARGUMENT => StatusError::Argument,
            ERR_UNSUPPORTED => StatusError::Unsupported,
            ERR_CAPACITY => StatusError::Capacity,
            ERR_STALE_EPOCH => StatusError::StaleEpoch,
            ERR_BUSY => StatusError::Busy,
            ERR_INTERNAL => StatusError::Internal,
            other => StatusError::Unknown(other),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            StatusError::Pending => "pending",
            StatusError::Argument => "argument",
            StatusError::Unsupported => "unsupported",
            StatusError::Capacity => "capacity",
            StatusError::StaleEpoch => "stale_epoch",
            StatusError::Busy => "busy",
            StatusError::Internal => "internal",
            StatusError::Unknown(_) => "unknown",
        }
    }
}

impl fmt::Display for StatusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatusError::Unknown(code) => write!(f, "unknown status {code}"),
            other => f.write_str(other.name()),
        }
    }
}

impl std::error::Error for StatusError {}

/// Why [`Library::open`] refused a library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// The file could not be opened or read.
    Io {
        path: PathBuf,
        message: String,
    },
    /// The file's SHA-256 differs from the expected one; `dlopen` was not called.
    HashMismatch {
        expected: String,
        actual: String,
    },
    /// `dlopen` failed (`dlerror` text).
    Open(String),
    MissingSymbol(&'static str),
    /// `cere_executor_abi_version()` is not [`ABI_VERSION`].
    AbiVersion {
        found: u32,
    },
}

impl LoadError {
    /// The error class printed by the CLI.
    pub fn class(&self) -> &'static str {
        match self {
            LoadError::Io { .. } => "library_io",
            LoadError::HashMismatch { .. } => "library_hash",
            LoadError::Open(_) => "library_open",
            LoadError::MissingSymbol(_) => "library_symbol",
            LoadError::AbiVersion { .. } => "abi_version",
        }
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io { path, message } => write!(f, "{}: {message}", path.display()),
            LoadError::HashMismatch { expected, actual } => {
                write!(f, "library sha256 {actual} != expected {expected}")
            }
            LoadError::Open(m) => write!(f, "dlopen: {m}"),
            LoadError::MissingSymbol(s) => write!(f, "the library does not export {s}"),
            LoadError::AbiVersion { found } => {
                write!(
                    f,
                    "the library implements ABI v{found}, this build needs v{ABI_VERSION}"
                )
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// A loaded executor library with every symbol resolved and the ABI version checked.
#[derive(Debug)]
pub struct Library {
    _handle: Handle,
    api: Api,
    path: PathBuf,
    sha256: String,
    abi_version: u32,
}

// SAFETY: the handle and the function pointers are immutable after `open`; the ABI's own threading rules (one
// serialized caller per executor, concurrent `cere_model_put`) are enforced by the driver, not by `Library`.
unsafe impl Send for Library {}
// SAFETY: as above; `&Library` only exposes copies of immutable data.
unsafe impl Sync for Library {}

/// What a file looks like from its descriptor: inode, size, modification and change times. A write in place moves the
/// size or the times (the change time cannot be set from user space).
#[derive(Debug, PartialEq, Eq)]
struct FileStamp {
    dev: u64,
    ino: u64,
    size: u64,
    mtime: (i64, i64),
    ctime: (i64, i64),
}

impl FileStamp {
    fn of(file: &std::fs::File) -> std::io::Result<Self> {
        use std::os::unix::fs::MetadataExt;
        let meta = file.metadata()?;
        Ok(Self {
            dev: meta.dev(),
            ino: meta.ino(),
            size: meta.size(),
            mtime: (meta.mtime(), meta.mtime_nsec()),
            ctime: (meta.ctime(), meta.ctime_nsec()),
        })
    }

    /// `Ok` when `file` still has this stamp.
    fn still_holds(&self, file: &std::fs::File, path: &Path) -> Result<(), LoadError> {
        let io = |message: String| LoadError::Io {
            path: path.to_path_buf(),
            message,
        };
        let now = Self::of(file).map_err(|e| io(e.to_string()))?;
        if &now == self {
            Ok(())
        } else {
            Err(io(
                "the library file changed while it was hashed and loaded".to_owned(),
            ))
        }
    }
}

fn last_dl_error() -> String {
    // SAFETY: dlerror returns null or a NUL-terminated thread-local string.
    let e = unsafe { libc::dlerror() };
    if e.is_null() {
        "unknown dlopen error".to_string()
    } else {
        // SAFETY: non-null, NUL-terminated (above).
        unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
    }
}

impl Library {
    /// Hashes `path` through one descriptor, refuses a hash other than `expected_sha256` (lowercase hex) before any
    /// `dlopen`, then loads the inode that descriptor names (`/proc/self/fd/N`), checks the ABI version and resolves
    /// every symbol.
    ///
    /// This is not content addressing: the hash is one read of the file and `dlopen` (which opens
    /// `/proc/self/fd/N` afresh and maps the file) is another. The descriptor pins the *inode*, so replacing the
    /// path (a `rename` over it, as an installer does) cannot put other bytes between the two. Writing the file in
    /// place can, and the kernel gives no way to prevent that here. It is *detected*: the file's inode, size and
    /// modification and change times are compared before the hash, before `dlopen` and after it, and any difference
    /// is a [`LoadError::Io`]. A write that lands in the interval and restores every one of those (a clock set back
    /// by a privileged writer) is not detected. The library is meant to sit root-owned in a directory only the
    /// installer writes; the check guards against an installer racing a start, not against a hostile writer.
    ///
    /// The dynamic loader identifies a loaded object by the name it was opened with, and `RTLD_NODELETE` keeps it
    /// for the life of the process. So the descriptor of a loaded library stays open for the life of the process
    /// too: its number, and with it the name `/proc/self/fd/N`, is never reused for another file.
    pub fn open(path: &Path, expected_sha256: Option<&str>) -> Result<Library, LoadError> {
        let io = |e: std::io::Error| LoadError::Io {
            path: path.to_path_buf(),
            message: e.to_string(),
        };
        let mut file = std::fs::File::open(path).map_err(io)?;
        let stamp = FileStamp::of(&file).map_err(io)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = file.read(&mut buf).map_err(io)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let sha256 = cere_compiler::hex(&hasher.finalize());
        if let Some(expected) = expected_sha256.filter(|e| *e != sha256) {
            return Err(LoadError::HashMismatch {
                expected: expected.to_string(),
                actual: sha256,
            });
        }
        stamp.still_holds(&file, path)?;
        let fd_path = CString::new(format!("/proc/self/fd/{}", file.as_raw_fd()))
            .expect("no NUL in a /proc path");
        // SAFETY: a valid C string; RTLD_NOW resolves everything up front.
        let handle = unsafe {
            libc::dlopen(
                fd_path.as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL | libc::RTLD_NODELETE,
            )
        };
        if handle.is_null() {
            return Err(LoadError::Open(last_dl_error()));
        }
        let unchanged = stamp.still_holds(&file, path);
        // Deliberately never closed (see above), on the refusal below too: the library stays mapped.
        let _ = file.into_raw_fd();
        unchanged?;
        let handle = Handle(handle);
        // SAFETY: each symbol is declared in the header with exactly the signature it is read as.
        let abi_version: unsafe extern "C" fn() -> u32 = unsafe { handle.symbol(0)? };
        // SAFETY: a no-argument function of the library (above).
        let found = unsafe { abi_version() };
        if found != ABI_VERSION {
            return Err(LoadError::AbiVersion { found });
        }
        // SAFETY: as for `abi_version`.
        let api = unsafe {
            Api {
                abi_version,
                executor_create: handle.symbol(1)?,
                executor_destroy: handle.symbol(2)?,
                executor_capabilities: handle.symbol(3)?,
                model_create: handle.symbol(4)?,
                model_put: handle.symbol(5)?,
                model_seal: handle.symbol(6)?,
                model_release: handle.symbol(7)?,
                session_create: handle.symbol(8)?,
                session_release: handle.symbol(9)?,
                executor_submit: handle.symbol(10)?,
                executor_cancel: handle.symbol(11)?,
                executor_poll: handle.symbol(12)?,
                executor_constrain: handle.symbol(13)?,
                session_snapshot: handle.symbol(14)?,
                session_restore: handle.symbol(15)?,
                snapshot_get_info: handle.symbol(16)?,
                snapshot_read: handle.symbol(17)?,
                snapshot_import: handle.symbol(18)?,
                snapshot_release: handle.symbol(19)?,
                session_set_cache_namespace: handle.symbol(20)?,
                model_flush_prefix_cache: handle.symbol(21)?,
                prefix_cache_export: handle.symbol(22)?,
                prefix_cache_import: handle.symbol(23)?,
                model_host_resident: handle.symbol(24)?,
                model_selection: handle.symbol(25)?,
                model_set_selection: handle.symbol(26)?,
                executor_register_staging: handle.optional_symbol(27),
                executor_unregister_staging: handle.optional_symbol(28),
                executor_set_decode_rounds: handle.optional_symbol(29),
                model_set_prefix_end_state: handle.optional_symbol(30),
                model_set_prefix_alt_edges: handle.optional_symbol(31),
                model_prefix_stats: handle.optional_symbol(32),
                executor_set_prefill_slice: handle.optional_symbol(33),
                executor_round_drafts: handle.optional_symbol(34),
                executor_constrain_drafts: handle.optional_symbol(35),
                model_lora_reserve: handle.optional_symbol(36),
                model_lora_load: handle.optional_symbol(37),
                model_lora_unload: handle.optional_symbol(38),
                session_set_lora: handle.optional_symbol(39),
                model_prefix_capture_stats: handle.optional_symbol(40),
                executor_set_decode_share: handle.optional_symbol(41),
                model_research_reserve: handle.optional_symbol(42),
                model_research_plan_load: handle.optional_symbol(43),
                model_research_plan_unload: handle.optional_symbol(44),
                session_set_research: handle.optional_symbol(45),
                model_research_capture_read: handle.optional_symbol(46),
                model_research_capture_free: handle.optional_symbol(47),
                model_prefill_signature: handle.optional_symbol(48),
                executor_set_prefill_age: handle.optional_symbol(49),
            }
        };
        Ok(Library {
            _handle: handle,
            api,
            path: path.to_path_buf(),
            sha256,
            abi_version: found,
        })
    }

    pub fn api(&self) -> &Api {
        &self.api
    }

    /// The path the library was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Lowercase hex SHA-256 of the loaded file.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// `cere_executor_abi_version()` as reported at load (always [`ABI_VERSION`] for an opened library).
    pub fn abi_version(&self) -> u32 {
        self.abi_version
    }
}

/// A `dlopen` handle, closed on drop. Loaded with `RTLD_NODELETE`, so code its function pointers name stays mapped
/// for the life of the process.
#[derive(Debug)]
struct Handle(*mut c_void);

impl Handle {
    /// `SYMBOLS[i]` as the function pointer type `F`.
    ///
    /// # Safety
    /// `F` must be the header's signature of that symbol.
    unsafe fn symbol<F: Copy>(&self, i: usize) -> Result<F, LoadError> {
        // SAFETY: the caller's contract.
        unsafe { self.optional_symbol(i) }.ok_or(LoadError::MissingSymbol(SYMBOLS[i]))
    }

    /// `SYMBOLS[i]` as `F`, `None` when the library does not export it.
    ///
    /// # Safety
    /// As for [`Handle::symbol`].
    unsafe fn optional_symbol<F: Copy>(&self, i: usize) -> Option<F> {
        assert_eq!(size_of::<F>(), size_of::<*mut c_void>());
        let name = CString::new(SYMBOLS[i]).expect("symbol names have no NUL");
        // SAFETY: a live handle and a valid C string.
        let p = unsafe { libc::dlsym(self.0, name.as_ptr()) };
        // SAFETY: a non-null code address of the caller-asserted type, same size as a pointer.
        (!p.is_null()).then(|| unsafe { std::mem::transmute_copy::<*mut c_void, F>(&p) })
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: the handle came from dlopen and is closed once.
        unsafe { libc::dlclose(self.0) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A file written in place between the stamp and the check is refused; an untouched one holds.
    #[test]
    fn a_file_written_in_place_no_longer_matches_its_stamp() {
        let path = std::env::temp_dir().join(format!("cere-ffi-stamp-{}", std::process::id()));
        std::fs::write(&path, b"library bytes").unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let stamp = FileStamp::of(&file).unwrap();
        stamp.still_holds(&file, &path).unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"!")
            .unwrap();
        let err = stamp.still_holds(&file, &path).unwrap_err();
        std::fs::remove_file(&path).unwrap();
        assert!(
            matches!(&err, LoadError::Io { message, .. } if message.contains("changed")),
            "{err:?}"
        );
    }
}
