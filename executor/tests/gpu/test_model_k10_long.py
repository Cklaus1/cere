"""K10 (tasks/m22-spec.md, M2.2c Tier 1): chunked prefill of 4,096 tokens and the 73,728-token context on the real A4
model, GPU0, container only.

Suite `model` (`native_dev.py test --suite model`). The model is created, uploaded and sealed exactly as in
`test_model_k8_forward.py` (ABI v5 `cere_model_desc` from `/model` read-only, families placement, concurrent puts,
seal), here with `max_context_tokens` 73,728 and one session. The prompts are synthetic and deterministic: the chat
template (`enable_thinking` false) around a generated list of numbered records and one question, cut in token space
to the exact length (no randomness, so no seed is involved). Token IDs and text stay in memory.

Cases (K10 acceptance, tasks/m22c-spec.md D7; agreement with SGLang at long context is X17's job):
* `test_chunk_sizes`: the 9,000-token prompt prefilled at `prefill_chunk` 4,096 (served, 3 passes), 2,048 (5 passes)
  and 1,024 (9 passes) through `cere_ktest_forward_trace` (B8 names, last position only) on the one 73,728-ctx
  model; the pass counts reported by `cere_ktest_forward_trace_passes` are asserted first (a request above the
  workspace's 4,096 rows is clamped, so a "single 9,000-token prefill" does not exist on the served plan). Then the
  last-position hidden state (`model`, the final mixer's output) of 2,048 and 1,024 is within the tolerance rule's
  limit of the 4,096 trace, and greedy decoding after each prefill gives identical tokens for 32 steps. Every other
  B8 name's last-position rel-L2 is reported (diagnostic, not asserted).
* `test_plan_bytes` (C10's device half): `cere_ktest_model_plan_bytes` at 73,728; the workspace's excess over the
  same rows' dense-only plan (the K9 scratch joining the shared max() scratch) is within D5's K9 terms plus the K9
  path's row buffers.
* `test_device_use_at_73728`: a 73,727-token prompt plus one generated token fills the 73,728-token session; the
  executor's device bytes in use at that point (cudaMemGetInfo, reported by `cere_ktest_generate` before the
  session is released) are at most 80 GiB.
* `test_repeat_bit_identical`: the 9,000-token prompt generates 32 greedy steps bit-identically (token IDs and
  top-2 scores) over 2 repeats.
* `K10ServedPath.test_parity_trace_context_invariant` (D7, M22CS-09): parity_v1 p01 and p13 (chat template,
  `enable_thinking` false) `forward_trace`, every row and every B8 name, on a 2,048-context model (k6::full_attention)
  and on a 73,728-context model (the served path: indexer GEMM, compressed-key store, k6::prologue +
  k6::dense_attention, 4,096 workspace rows) of the same library are byte-equal, scores included (every row of both
  score buffers, pre-filled with -1, is first checked to be written). The two models are created and released one at
  a time (both do not fit on GPU0); only per-name SHA-256 digests and the scores of each model are kept.

Calibration (tolerance rule, explained): the rule's c is rel-L2(SGLang op, FP64 reference on the same inputs); a
48-layer forward over 9,000 tokens has no in-test FP64 reference (there is no FP64 model to run, and SGLang's
long-context arithmetic is compared by X17, not here). Both sides of this comparison are native, so the reference is
the served 4,096-chunk prefill; smaller chunks add more chunk boundaries (FP8 KV reads, BF16 GDN state, compressed
keys from the ring), which is exactly what is measured. c is the larger of (a) the reference's own spread, rel-L2
between two 4,096 runs (0 when deterministic, which K8 requires), and (b) the chunk-carry spread on the prompt's first
2,048 tokens (dense K6 range): one pass vs two passes of 1,024. Both are computed and written before the compared
runs are scored; the limit is max(1e-2, 2c), the rule's whole-block floor applied to the final hidden state.

Native entry points (shared with K8; the contract with the kernel track; `oracle.call` appends the stream; 0 = success, otherwise a
`cere_status`; "host" buffers are CPU memory, "device" buffers GPU0 memory; the ABI v5 types are those of
`executor/include/cere_executor.h`):

    int cere_ktest_model_create(const cere_model_desc *desc /*host*/, uint64_t memory_budget_bytes,
                                uint64_t *out_model /*host [1]*/, cudaStream_t stream);
        A ktest-owned executor on GPU0 (`cere_executor_create`, budget as given) and `cere_model_create(desc)`;
        writes an opaque handle for the pair. Status as `cere_model_create` (UNSUPPORTED, CAPACITY, ...).
    int cere_ktest_model_put(uint64_t model, uint64_t tensor_index, uint64_t offset, const void *bytes /*host*/,
                             uint64_t len, cudaStream_t stream);
        `cere_model_put`; callable from several threads at once for different tensor_index values.
    int cere_ktest_model_seal(uint64_t model, cudaStream_t stream);
        `cere_model_seal` (exact tiling of every non-discarded tensor, then INSTANTIATE).
    int cere_ktest_model_release(uint64_t model, cudaStream_t stream);
        `cere_model_release` then `cere_executor_destroy`; frees every device and pinned byte of the pair.
    int cere_ktest_generate(uint64_t model, const uint32_t *prompt /*host*/, uint64_t prompt_len,
                            uint64_t max_new_tokens, uint64_t prefill_chunk_tokens, uint64_t cancel_after,
                            uint32_t *out_tokens /*host [max_new_tokens]*/,
                            cere_token_score *out_scores /*host [max_new_tokens]*/,
                            uint64_t *out_result /*host [3]*/, cudaStream_t stream);
        One greedy request through the ABI v5 runtime: `cere_session_create` (the model's context limit), one
        PREFILL `cere_work` carrying the prompt, max_new_tokens and the leased out_tokens/out_scores, submit, poll to
        the terminal completion, `cere_session_release`. prefill_chunk_tokens 0 = the executor's own prefill policy
        (M2.2b: one prefill; M2.2c: chunks of 4,096, Q8), N > 0 = chunks of N tokens (N >= prompt_len: one
        prefill). cancel_after 0 = never; k > 0 = `cere_executor_cancel` is
        issued at the token boundary right after the k-th token is committed (a deterministic stand-in for a client
        cancel arriving between tokens k and k+1). out_result = {committed_tokens, finish (CERE_FINISH_*), device
        bytes in use (cudaMemGetInfo total - free) after the last committed token, before the session is
        released}. Returns 0 for LENGTH, STOP_TOKEN and CANCELLED completions, the completion status otherwise.
    int cere_ktest_forward_trace(uint64_t model, const uint32_t *tokens /*host*/, uint64_t token_count,
                                 uint64_t prefill_chunk_tokens, uint64_t last_rows,
                                 uint8_t *names /*host*/, uint64_t names_capacity,
                                 int64_t *shapes /*host [entry_capacity][3]*/, uint64_t entry_capacity,
                                 void *values /*device*/, uint64_t values_capacity_bytes,
                                 cere_token_score *scores /*host [rows]*/, uint64_t *out_counts /*host [3]*/,
                                 cudaStream_t stream);
        Teacher-forced forward of `tokens` in a fresh ktest-owned session (released before return), prefilled with
        the same prefill_chunk_tokens semantics as `cere_ktest_generate`. Emits one entry per B8 tensor: its
        NUL-terminated UTF-8 name appended to `names`, {ndim = 2, rows, cols} in `shapes`, and its BF16 row-major
        values appended back to back to `values`. rows = last_rows (the last positions only) or token_count when
        last_rows is 0. scores[r] is the top-2 of the logits at the r-th returned position (that position predicts
        the next token). out_counts = {entries, name bytes, value bytes}. Any capacity too small: CERE_ERR_CAPACITY.
    int cere_ktest_forward_trace_passes(uint64_t model, uint64_t *out_passes /*host [1]*/, cudaStream_t stream);
        The prefill passes (forward calls) of the last `cere_ktest_forward_trace` on `model`.
    int cere_ktest_model_plan_bytes(uint64_t model, uint64_t *out /*host [5]*/, cudaStream_t stream);
        The model's device plan: {workspace rows, workspace_bytes, workspace_bytes of the same rows at the dense
        limit (no K9 scratch), derived_bytes, session_bytes}.
"""
from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import re
import time
import unittest
from pathlib import Path
from typing import NamedTuple

import torch
from safetensors import safe_open
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

import oracle

ROW = "K10"
SRC = Path(__file__).resolve().parents[3]
FAMILIES_FILE = SRC / "configs/artifacts/qwen4_exp-nvfp4.families.json"
PARITY_FILE = SRC / "bench/prompts/parity_v1.jsonl"
PARITY_IDS = ("p01", "p13")  # M22CS-09
DENSE_CONTEXT = 2_048  # k6::kMaxContext: the model that never takes the K9 branch
MAX_CONTEXT = 73_728
PROMPT_TOKENS = 9_000
CHUNK = 4_096  # the served prefill chunk (D4)
CHUNKS = (CHUNK, 2_048, 1_024)  # D7: 3, 5 and 9 passes over PROMPT_TOKENS
CARRY_TOKENS = 2_048  # the chunk-carry calibration prompt (dense range): one pass vs two
# D5's K9 terms at 4,096 rows (logits tile, indices [4,096, 2,051] int32, index q [4,096, 4, 128] BF16) plus the K9
# path's row buffers (prologue q/k/gate, index qk [4,096, 640], ungated attention output), all BF16.
K9_SCRATCH_BOUND = (134_217_728 + 33_603_584 + 4_194_304
                    + 4_096 * (6_144 + 512 + 6_144) * 2 + 4_096 * 640 * 2 + 4_096 * 6_144 * 2)
STEPS = 32
REPEATS = 2
DEVICE_BOUND = 80 << 30
SENTINEL = "<<RECORDS>>"
QUESTION = "\n\nWhich record lists the heaviest animal? Answer with its record number only."
COLORS = ("red", "green", "blue", "amber", "violet", "silver", "black")
ANIMALS = ("otter", "heron", "lynx", "badger", "falcon", "marten", "ibex", "crane", "viper", "bison", "gecko")

# -- ABI v5 (executor/include/cere_executor.h), mirrored with ctypes ---------------------------------------------

ABI_VERSION = 5
MODEL_QWEN4_EXP_NVFP4 = 1
PLACEMENT_DEVICE = 1
PLACEMENT_HOST_PINNED = 2
PLACEMENT_DISCARD = 3
DTYPES = {"BF16": (1, 2), "F16": (2, 2), "F32": (3, 4), "F64": (4, 8), "F8_E4M3": (5, 1), "F8_E5M2": (6, 1),
          "U8": (7, 1), "I8": (8, 1), "I16": (9, 2), "I32": (10, 4), "I64": (11, 8), "BOOL": (12, 1)}
SSM_DTYPES = {"bfloat16": 1, "float16": 2, "float32": 3}
OUTPUT_GATES = {"sigmoid": 1}
FINISH_LENGTH = 1
FINISH_STOP_TOKEN = 2
FINISH_CANCELLED = 3
MAX_DIMS = 8
PUT_PIECE = 64 << 20

U32, U64, F64 = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_double


class Qwen4ExpConfig(ctypes.Structure):
    _fields_ = [("struct_size", U32), ("abi_version", U32), ("vocab_size", U32), ("hidden_size", U32),
                ("num_hidden_layers", U32), ("full_attention_interval", U32), ("full_attention_layers", U64),
                ("num_attention_heads", U32), ("num_key_value_heads", U32), ("head_dim", U32),
                ("output_gate_type", U32), ("rope_theta", F64), ("partial_rotary_factor", F64),
                ("mrope_interleaved", U32), ("mrope_section", U32 * 3), ("indexer_n_heads", U32),
                ("indexer_head_dim", U32), ("indexer_kv_heads", U32), ("indexer_compress_ratio", U32),
                ("indexer_budget", U32), ("linear_num_key_heads", U32), ("linear_num_value_heads", U32),
                ("linear_key_head_dim", U32), ("linear_value_head_dim", U32), ("linear_conv_kernel_dim", U32),
                ("mamba_ssm_dtype", U32), ("num_experts", U32), ("num_experts_per_tok", U32),
                ("moe_intermediate_size", U32), ("shared_expert_intermediate_size", U32), ("norm_topk_prob", U32),
                ("hc_count", U32), ("hc_lowrank", U32), ("ple_layer_ids", U64), ("ple_embed_dim", U32),
                ("heads_per_ngram", U32), ("ngram_size", U32), ("ple_conv_kernel_size", U32),
                ("ngram_vocab_size_base", U64), ("split_ngram_parts", U32),
                ("make_ngram_vocab_size_divisible_by", U32), ("rms_norm_eps", F64), ("tie_word_embeddings", U32),
                ("mtp_num_hidden_layers", U32), ("max_position_embeddings", U64)]


class TensorDesc(ctypes.Structure):
    _fields_ = [("struct_size", U32), ("abi_version", U32), ("name", ctypes.c_void_p), ("dtype", U32),
                ("placement", U32), ("ndim", U32), ("reserved", U32), ("shape", U64 * MAX_DIMS), ("bytes", U64)]


class ModelDesc(ctypes.Structure):
    _fields_ = [("struct_size", U32), ("abi_version", U32), ("model_kind", U32), ("speculative_draft_tokens", U32),
                ("config", ctypes.c_void_p), ("tensors", ctypes.c_void_p), ("tensor_count", U64),
                ("max_sessions", U64), ("max_context_tokens", U64), ("stop_token_ids", ctypes.c_void_p),
                ("stop_token_count", U64), ("kv_pool_tokens", U64), ("draft_token_map", ctypes.c_void_p),
                ("draft_token_map_count", U64), ("snapshot_host_bytes", U64), ("prefix_state_slots", U64),
                ("vision_config", ctypes.c_void_p), ("selection", ctypes.c_void_p), ("family", U32),
                ("calibrate_windows", U32)]


# Plain integer fields of cere_qwen4_exp_config, copied from text_config (an absent key is 0, header rule).
CONFIG_INT_KEYS = (
    "vocab_size", "hidden_size", "num_hidden_layers", "full_attention_interval", "num_attention_heads",
    "num_key_value_heads", "head_dim", "indexer_n_heads", "indexer_head_dim", "indexer_kv_heads",
    "indexer_compress_ratio", "indexer_budget", "linear_num_key_heads", "linear_num_value_heads",
    "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim", "num_experts", "num_experts_per_tok",
    "moe_intermediate_size", "shared_expert_intermediate_size", "hc_count", "hc_lowrank", "ple_embed_dim",
    "heads_per_ngram", "ngram_size", "ple_conv_kernel_size", "ngram_vocab_size_base", "split_ngram_parts",
    "make_ngram_vocab_size_divisible_by", "tie_word_embeddings", "mtp_num_hidden_layers", "max_position_embeddings",
)


def struct_buffer(ctype, count: int = 1):
    """A zeroed CPU tensor holding `count` ctypes `ctype` values, and the ctypes view over it (the tensor is what
    `oracle.call` passes, so the struct bytes and the pointer stay one object)."""
    buf = torch.zeros(ctypes.sizeof(ctype) * count, dtype=torch.uint8)
    view = (ctype * count).from_address(buf.data_ptr())
    return buf, view


def a4_config(tc: dict):
    """cere_qwen4_exp_config from text_config by the header's rules (lists as bitmasks, strings as enums)."""
    buf, (cfg,) = struct_buffer(Qwen4ExpConfig)
    cfg.struct_size = ctypes.sizeof(Qwen4ExpConfig)
    cfg.abi_version = ABI_VERSION
    for key in CONFIG_INT_KEYS:
        setattr(cfg, key, int(tc.get(key, 0)))
    cfg.full_attention_layers = sum(1 << i for i, kind in enumerate(tc["layer_types"]) if kind == "full_attention")
    gate = tc.get("output_gate_type")
    if gate is not None and gate not in OUTPUT_GATES:
        raise oracle.OracleError(f"output_gate_type {gate!r} has no CERE_OUTPUT_GATE_* value")
    cfg.output_gate_type = OUTPUT_GATES.get(gate, 0)
    rope = tc["rope_parameters"]
    cfg.rope_theta = float(rope["rope_theta"])
    cfg.partial_rotary_factor = float(rope["partial_rotary_factor"])
    cfg.mrope_interleaved = int(bool(rope.get("mrope_interleaved", False)))
    for i, value in enumerate(rope["mrope_section"]):
        cfg.mrope_section[i] = int(value)
    ssm = tc.get("mamba_ssm_dtype")
    if ssm is not None and ssm not in SSM_DTYPES:
        raise oracle.OracleError(f"mamba_ssm_dtype {ssm!r} has no CERE_DTYPE_* value")
    cfg.mamba_ssm_dtype = SSM_DTYPES.get(ssm, 0)
    cfg.norm_topk_prob = int(bool(tc.get("norm_topk_prob", True)))
    cfg.ple_layer_ids = sum(1 << int(n) for n in tc.get("ple_layer_ids", []))
    cfg.rms_norm_eps = float(tc["rms_norm_eps"])
    return buf


def tensor_placements() -> list:
    """(compiled pattern, placement) per rule of the artifact's family file, in rule order."""
    rules = json.loads(FAMILIES_FILE.read_text())["rules"]
    out = []
    for rule in rules:
        placement = PLACEMENT_DISCARD
        if rule["placement"] == "unused":
            # Out of the desc, as cered's model spec leaves it (a text-only desc carries no vision tensor, C7).
            placement = None
        elif rule["component"] == "target":
            placement = {"device": PLACEMENT_DEVICE, "host_pinned": PLACEMENT_HOST_PINNED}[rule["placement"]]
        out.append((re.compile(rule["pattern"]), placement))
    return out


class TableEntry(NamedTuple):
    name: str
    file: str
    placement: int
    nbytes: int


def a4_tensor_table():
    """Every checkpoint tensor in index (name) order but the family file's `unused` ones: table entries and the
    cere_tensor_desc array over them."""
    weight_map = json.loads((oracle.MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    rules = tensor_placements()
    by_file: dict[str, list[str]] = {}
    for name in weight_map:
        by_file.setdefault(weight_map[name], []).append(name)
    header: dict[str, tuple[str, list[int]]] = {}
    for file, names in by_file.items():
        with safe_open(str(oracle.MODEL_DIR / file), framework="pt", device="cpu") as f:
            for name in names:
                piece = f.get_slice(name)
                header[name] = (piece.get_dtype(), list(piece.get_shape()))
    placements = {}
    for name in sorted(weight_map):
        match = next(((p,) for pattern, p in rules if pattern.fullmatch(name)), None)
        if match is None:
            raise oracle.OracleError(f"{name} matches no rule of {FAMILIES_FILE.name}")
        if match[0] is not None:
            placements[name] = match[0]
    names = list(placements)
    blob = b"".join(n.encode() + b"\0" for n in names)
    name_buf = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
    desc_buf, descs = struct_buffer(TensorDesc, len(names))
    entries = []
    offset = 0
    for i, name in enumerate(names):
        placement = placements[name]
        dtype, shape = header[name]
        if dtype not in DTYPES:
            raise oracle.OracleError(f"{name}: dtype {dtype} has no CERE_DTYPE_* value")
        if len(shape) > MAX_DIMS:
            raise oracle.OracleError(f"{name}: rank {len(shape)} exceeds CERE_TENSOR_MAX_DIMS")
        code, size = DTYPES[dtype]
        nbytes = size
        for dim in shape:
            nbytes *= dim
        d = descs[i]
        d.struct_size = ctypes.sizeof(TensorDesc)
        d.abi_version = ABI_VERSION
        d.name = name_buf.data_ptr() + offset
        d.dtype = code
        d.placement = placement
        d.ndim = len(shape)
        for k, dim in enumerate(shape):
            d.shape[k] = dim
        d.bytes = nbytes
        offset += len(name.encode()) + 1
        entries.append(TableEntry(name, weight_map[name], placement, nbytes))
    return entries, desc_buf, name_buf


def stop_token_ids() -> list[int]:
    eos = json.loads((oracle.MODEL_DIR / "generation_config.json").read_text())["eos_token_id"]
    return [int(t) for t in (eos if isinstance(eos, list) else [eos])]


def upload_file(handle: int, file: str, items: list) -> None:
    """Puts every non-discarded tensor of one safetensors file in pieces of PUT_PIECE bytes."""
    with safe_open(str(oracle.MODEL_DIR / file), framework="pt", device="cpu") as f:
        for index, entry in items:
            flat = f.get_tensor(entry.name).contiguous().reshape(-1).view(torch.uint8)
            if flat.numel() != entry.nbytes:
                raise oracle.OracleError(f"{entry.name}: {flat.numel()} bytes read, header says {entry.nbytes}")
            for offset in range(0, entry.nbytes, PUT_PIECE):
                piece = flat[offset:offset + PUT_PIECE]
                oracle.call("cere_ktest_model_put", handle, index, offset, piece, piece.numel())


class A4Model(NamedTuple):
    handle: int
    keep: tuple  # the descriptor buffers (the executor copies them, they only need to outlive create)


def create_model(max_context_tokens: int) -> A4Model:
    """ALLOCATE (create), UPLOAD (concurrent puts), INSTANTIATE (seal) of the real A4 model on GPU0."""
    tc = oracle.text_config()
    config_buf = a4_config(tc)
    entries, desc_buf, name_buf = a4_tensor_table()
    stops = torch.tensor(stop_token_ids(), dtype=torch.int32)
    model_buf, (desc,) = struct_buffer(ModelDesc)
    desc.struct_size = ctypes.sizeof(ModelDesc)
    desc.abi_version = ABI_VERSION
    desc.model_kind = MODEL_QWEN4_EXP_NVFP4
    desc.config = config_buf.data_ptr()
    desc.tensors = desc_buf.data_ptr()
    desc.tensor_count = len(entries)
    desc.max_sessions = 1
    desc.max_context_tokens = max_context_tokens
    desc.stop_token_ids = stops.data_ptr()
    desc.stop_token_count = stops.numel()
    handle = oracle.allocate(model_buf)
    return A4Model(handle, (config_buf, desc_buf, name_buf, stops, model_buf, entries))


def upload_and_seal(model: A4Model) -> None:
    oracle.upload_and_seal(model.handle, model.keep[-1], upload_file)


def release_model(handle: int) -> None:
    oracle.release_model(handle)


# -- runtime entry points ----------------------------------------------------------------------------------------


class Generation(NamedTuple):
    tokens: torch.Tensor  # int32 [committed] (uint32 IDs)
    scores: torch.Tensor  # int32 [committed, 4]: cere_token_score bytes {id, runner_up, logit, runner_up_logit}
    committed: int
    finish: int
    device_bytes: int


def generate(handle: int, ids: list[int], max_new: int, *, chunk: int = 0, cancel_after: int = 0) -> Generation:
    prompt = torch.tensor(ids, dtype=torch.int32)
    tokens = torch.full((max_new,), -1, dtype=torch.int32)
    scores = torch.full((max_new, 4), -1, dtype=torch.int32)
    result = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_generate", handle, prompt, len(ids), max_new, chunk, cancel_after, tokens, scores,
                result)
    committed, finish, device_bytes = (int(v) for v in result.tolist())
    if not 0 <= committed <= max_new:
        raise oracle.OracleError(f"committed_tokens {committed} outside [0, {max_new}]")
    return Generation(tokens[:committed].clone(), scores[:committed].clone(), committed, finish, device_bytes)


class Trace(NamedTuple):
    values: dict  # B8 name -> BF16 [rows, cols] on GPU0
    scores: torch.Tensor  # int32 [rows, 4] (cere_token_score bytes)


def forward_trace(handle: int, ids: list[int], *, hidden: int, layers: int, chunk: int = 0,
                  last_rows: int = 0) -> Trace:
    rows = last_rows or len(ids)
    capacity = 4 * layers
    prompt = torch.tensor(ids, dtype=torch.int32)
    names = torch.zeros(capacity * 128, dtype=torch.uint8)
    shapes = torch.full((capacity, 3), -1, dtype=torch.int64)
    values = oracle.empty((capacity * rows * hidden,), torch.bfloat16)
    scores = torch.full((rows, 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_forward_trace", handle, prompt, len(ids), chunk, last_rows, names, names.numel(),
                shapes, capacity, values, values.numel() * values.element_size(), scores, counts)
    entries, name_bytes, value_bytes = (int(v) for v in counts.tolist())
    if not (0 <= entries <= capacity and 0 <= name_bytes <= names.numel()
            and 0 <= value_bytes <= values.numel() * values.element_size()):
        raise oracle.OracleError(f"forward_trace counts {entries}, {name_bytes}, {value_bytes} exceed capacities")
    emitted = bytes(names[:name_bytes].tolist()).split(b"\0")
    if len(emitted) != entries + 1 or emitted[-1] != b"":
        raise oracle.OracleError(f"forward_trace wrote {len(emitted) - 1} names for {entries} entries")
    out: dict = {}
    element = 0
    for i in range(entries):
        name = emitted[i].decode()
        ndim, r, c = (int(v) for v in shapes[i].tolist())
        if ndim != 2 or r < 0 or c < 0:
            raise oracle.OracleError(f"forward_trace entry {name}: shape record {(ndim, r, c)}")
        if name in out:
            raise oracle.OracleError(f"forward_trace emitted {name} twice")
        out[name] = values[element:element + r * c].view(r, c)
        element += r * c
    if element * values.element_size() != value_bytes:
        raise oracle.OracleError(f"forward_trace value bytes {value_bytes} != the entries' {element * 2}")
    return Trace(out, scores)


def b8_names(tc: dict) -> list[str]:
    """The B8 tensor names: per layer N the attention output projection, the routed experts and the shared expert's
    down projection, then the final `model` output."""
    names = []
    for layer, kind in enumerate(tc["layer_types"]):
        attn = "self_attn.o_proj" if kind == "full_attention" else "linear_attn.out_proj"
        names += [f"model.layers.{layer}.{attn}", f"model.layers.{layer}.mlp.experts",
                  f"model.layers.{layer}.mlp.shared_expert.down_proj"]
    names.append("model")
    return names


# -- prompts -----------------------------------------------------------------------------------------------------


@functools.cache
def tokenizer():
    return get_tokenizer(str(oracle.MODEL_DIR))


def synthetic_ids(total: int) -> list[int]:
    """Exactly `total` prompt IDs: the chat template around numbered records (cut in token space) and a question."""
    tok = tokenizer()
    text = tok.apply_chat_template([{"role": "user", "content": SENTINEL + QUESTION}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    if text.count(SENTINEL) != 1:
        raise oracle.OracleError("the rendered chat template does not hold the record sentinel exactly once")
    head, tail = text.split(SENTINEL)
    head_ids = list(tok.encode(head, add_special_tokens=False))
    tail_ids = list(tok.encode(tail, add_special_tokens=False))
    need = total - len(head_ids) - len(tail_ids)
    records = "\n".join(f"Record {i:05d}: the {COLORS[i % len(COLORS)]} {ANIMALS[i % len(ANIMALS)]} weighs "
                         f"{(i * 37) % 101 + 1} kg." for i in range(max(need // 8, 1)))
    body_ids = list(tok.encode(records, add_special_tokens=False))
    if need <= 0 or len(body_ids) < need:
        raise oracle.OracleError(f"synthetic prompt of {total} tokens: {len(body_ids)} record tokens for {need}")
    return head_ids + body_ids[:need] + tail_ids


def name_kind(name: str) -> tuple[str, int | None]:
    """(name pattern, layer) of a B8 name: `model.layers.N.rest` -> (rest, N); `model` -> (model, None)."""
    if name == "model":
        return name, None
    _, _, layer, rest = name.split(".", 3)
    return rest, int(layer)


class K10Long(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tc = oracle.text_config()
        cls.hidden = int(cls.tc["hidden_size"])
        cls.layers = int(cls.tc["num_hidden_layers"])
        cls.stops = set(stop_token_ids())
        cls.prompt = synthetic_ids(PROMPT_TOKENS)
        model = create_model(MAX_CONTEXT)
        cls.addClassCleanup(release_model, model.handle)
        upload_and_seal(model)
        cls.handle = model.handle

    def check_greedy(self, case: str, g: Generation, max_new: int) -> None:
        self.assertIn(g.finish, (FINISH_LENGTH, FINISH_STOP_TOKEN), f"{ROW} {case}: finish {g.finish}")
        self.assertTrue(g.committed >= 1, f"{ROW} {case}: no token committed")
        if g.finish == FINISH_LENGTH:
            self.assertEqual(g.committed, max_new, f"{ROW} {case}: LENGTH before max_new_tokens")
        else:
            self.assertTrue(int(g.tokens[-1]) in self.stops, f"{ROW} {case}: STOP_TOKEN without a stop token last")
        self.assertTrue(torch.equal(g.scores[:, 0], g.tokens), f"{ROW} {case}: score id != committed token")

    def trace_with_passes(self, ids: list[int], chunk: int) -> tuple[Trace, int]:
        trace = forward_trace(self.handle, ids, hidden=self.hidden, layers=self.layers, chunk=chunk, last_rows=1)
        passes = torch.zeros(1, dtype=torch.int64)
        oracle.call("cere_ktest_forward_trace_passes", self.handle, passes)
        missing = [n for n in b8_names(self.tc) if n not in trace.values]
        self.assertEqual(missing, [], f"{ROW} forward_trace (chunk {chunk}) lacks {missing}")
        return trace, int(passes.item())

    def test_chunk_sizes(self):
        # Pass counts first: the served 4,096 and the two smaller chunks really split the prompt differently.
        traces = {}
        for chunk in CHUNKS:
            trace, passes = self.trace_with_passes(self.prompt, chunk)
            oracle.check_count(self, ROW, f"prefill passes chunk {chunk}", passes,
                               expected=-(-PROMPT_TOKENS // chunk))
            traces[chunk] = trace
        self.assertEqual(len({-(-PROMPT_TOKENS // c) for c in CHUNKS}), len(CHUNKS), f"{ROW} pass counts not distinct")
        reference = traces[CHUNK].values["model"]
        self.assertEqual(tuple(reference.shape), (1, self.hidden), f"{ROW} last-position hidden shape")
        # c: the served chunk's repeat spread, and the chunk-carry spread of a dense-range prompt (one 2,048 pass vs
        # two 1,024 passes), whichever is larger; both exist before the long comparisons.
        again, _ = self.trace_with_passes(self.prompt, CHUNK)
        c_repeat = oracle.rel_l2(again.values["model"], reference)
        short = self.prompt[:CARRY_TOKENS]
        one, _ = self.trace_with_passes(short, CARRY_TOKENS)
        two, passes = self.trace_with_passes(short, CARRY_TOKENS // 2)
        oracle.check_count(self, ROW, "carry calibration passes", passes, expected=2)
        c_carry = oracle.rel_l2(two.values["model"], one.values["model"])
        oracle.report_row(ROW, "calibration parts", c_repeat=c_repeat, c_carry=c_carry)
        c = oracle.calibrate(ROW, "last hidden 9000", again.values["model"], reference, oracle.FLOOR_BLOCK)
        c = max(c, c_carry)
        limit = oracle.calibrated_limit(c, oracle.FLOOR_BLOCK)
        for chunk in CHUNKS[1:]:
            worst: dict = {}
            for name in b8_names(self.tc):
                spread = oracle.rel_l2(traces[chunk].values[name], traces[CHUNK].values[name])
                kind, layer = name_kind(name)
                if kind not in worst or spread > worst[kind][0]:
                    worst[kind] = (spread, layer)
            for kind, (spread, layer) in sorted(worst.items()):
                oracle.report_row(ROW, f"chunk {chunk} vs {CHUNK} worst {kind}", layer=layer, rel_l2=spread)
            measured = oracle.rel_l2(traces[chunk].values["model"], reference)
            oracle.check_within(self, ROW, f"last hidden 9000 chunk {chunk}", measured, c=c, limit=limit)
        greedy = {chunk: generate(self.handle, self.prompt, STEPS, chunk=chunk) for chunk in CHUNKS}
        for chunk, g in greedy.items():
            self.check_greedy(f"greedy chunk {chunk}", g, STEPS)
        for chunk in CHUNKS[1:]:
            oracle.check_count(self, ROW, f"greedy steps chunk {chunk} vs {CHUNK}", greedy[chunk].committed,
                               expected=greedy[CHUNK].committed)
            oracle.check_bit_exact(self, ROW, f"greedy tokens chunk {chunk} vs {CHUNK}", greedy[chunk].tokens,
                                   greedy[CHUNK].tokens)

    def test_plan_bytes(self):
        # C10's device half: the K9 scratch's excess over the dense plan's workspace stays within D5's K9 terms plus
        # the K9 path's own row buffers (prologue outputs, index qk, ungated attention output) at 4,096 rows.
        out = torch.zeros(5, dtype=torch.int64)
        oracle.call("cere_ktest_model_plan_bytes", self.handle, out)
        rows, workspace, dense_workspace, derived, session = (int(v) for v in out.tolist())
        excess = max(workspace - dense_workspace, 0)
        oracle.report_row(ROW, "plan 73728", workspace_rows=rows, workspace_bytes=workspace,
                          dense_workspace_bytes=dense_workspace, k9_excess_bytes=excess, bound=K9_SCRATCH_BOUND,
                          derived_bytes=derived, session_bytes=session)
        self.assertEqual(rows, CHUNK, f"{ROW} workspace rows")
        self.assertTrue(excess <= K9_SCRATCH_BOUND, f"{ROW} K9 scratch excess {excess} > {K9_SCRATCH_BOUND}")

    def test_device_use_at_73728(self):
        ids = synthetic_ids(MAX_CONTEXT - 1)
        started = time.perf_counter()
        g = generate(self.handle, ids, 1, chunk=CHUNK)
        # R1 (M22CS-12): the 73,727-token prefill plus one decode step, wall time of the whole request.
        oracle.report_row(ROW, "prefill wall 73728", prompt_tokens=len(ids), seconds=time.perf_counter() - started)
        self.assertIn(g.finish, (FINISH_LENGTH, FINISH_STOP_TOKEN), f"{ROW} 73728: finish {g.finish}")
        oracle.check_count(self, ROW, "73728 committed", g.committed, expected=1)
        oracle.report_row(ROW, "device use 73728", context=len(ids) + g.committed, device_bytes=g.device_bytes,
                          bound=DEVICE_BOUND, passed=0 < g.device_bytes <= DEVICE_BOUND)
        self.assertTrue(0 < g.device_bytes <= DEVICE_BOUND,
                        f"{ROW} device use {g.device_bytes} bytes at 73,728 tokens exceeds {DEVICE_BOUND}")

    def test_repeat_bit_identical(self):
        runs = [generate(self.handle, self.prompt, STEPS, chunk=CHUNK) for _ in range(REPEATS)]
        for k, run in enumerate(runs):
            self.check_greedy(f"9000 run {k}", run, STEPS)
        for k in range(1, REPEATS):
            oracle.check_count(self, ROW, f"9000 run {k} committed", runs[k].committed, expected=runs[0].committed)
            oracle.check_bit_exact(self, ROW, f"9000 run {k} tokens", runs[k].tokens, runs[0].tokens)
            oracle.check_bit_exact(self, ROW, f"9000 run {k} scores", runs[k].scores, runs[0].scores)


def parity_ids() -> dict[str, list[int]]:
    """parity_v1 prompt IDs as SGLang's chat path builds them (template, `enable_thinking` false, no extra specials)."""
    rows = {}
    for line in PARITY_FILE.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            rows[item["id"]] = item
    missing = [pid for pid in PARITY_IDS if pid not in rows]
    if missing:
        raise oracle.OracleError(f"{PARITY_FILE.name} lacks {missing}")
    tok = tokenizer()
    out = {}
    for pid in PARITY_IDS:
        text = tok.apply_chat_template(rows[pid]["messages"], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        out[pid] = list(tok.encode(text, add_special_tokens=False))
    return out


def trace_digests(trace: Trace) -> dict[str, str]:
    """SHA-256 of each traced tensor's bytes (BF16, row-major), so the device buffers can go with the model."""
    return {name: hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            for name, t in trace.values.items()}


class K10ServedPath(unittest.TestCase):
    def test_parity_trace_context_invariant(self):
        tc = oracle.text_config()
        hidden, layers = int(tc["hidden_size"]), int(tc["num_hidden_layers"])
        prompts = parity_ids()
        for pid, ids in prompts.items():
            self.assertTrue(0 < len(ids) <= DENSE_CONTEXT, f"{ROW} {pid}: {len(ids)} tokens")
        runs = {}
        for context in (DENSE_CONTEXT, MAX_CONTEXT):
            model = create_model(context)
            try:
                upload_and_seal(model)
                runs[context] = {}
                for pid, ids in prompts.items():
                    trace = forward_trace(model.handle, ids, hidden=hidden, layers=layers)
                    runs[context][pid] = (trace_digests(trace), trace.scores.clone())
                    del trace
            finally:
                release_model(model.handle)
                torch.cuda.empty_cache()
        for pid in prompts:
            dense, served = runs[DENSE_CONTEXT][pid], runs[MAX_CONTEXT][pid]
            self.assertEqual(sorted(served[0]), sorted(dense[0]), f"{ROW} {pid}: traced names differ")
            missing = [n for n in b8_names(tc) if n not in dense[0]]
            self.assertEqual(missing, [], f"{ROW} {pid}: forward_trace lacks {missing}")
            differing = sum(served[0][n] != dense[0][n] for n in dense[0])
            oracle.check_count(self, ROW, f"{pid} trace names differing 73728 vs 2048", differing, expected=0)
            for context, (_, scores) in ((DENSE_CONTEXT, dense), (MAX_CONTEXT, served)):
                written = int((scores[:, 0] >= 0).sum().item())
                oracle.check_count(self, ROW, f"{pid} scored rows ctx {context}", written,
                                   expected=len(prompts[pid]))
            oracle.check_bit_exact(self, ROW, f"{pid} scores 73728 vs 2048", served[1], dense[1])


if __name__ == "__main__":
    unittest.main()
