"""Container-only oracle helpers for the native kernel suites (tasks/m22-spec.md B2 and the tolerance rule).

The suites run only inside the pinned dev image on GPU0 (except the GPU1 kernel lab's `klab` and `k7t` suites, run by
`native_dev.py --gpu 1 test --suite klab|k7t` in `cerelab-g1-*`, tasks/m215-spec.md D1), through
`CERE_LIVE=1 python3 scripts/live/native_dev.py test --suite S`, which mounts the build at `/out` (read-only) and the
A4 model directory at `/model` (read-only) and runs `python3 -m unittest discover -s /src/executor/tests/gpu -p
'test_S_*.py' -v`. Any skip is a failure there, so nothing here or in a suite skips: a missing library or a missing
`cere_ktest_*` symbol raises `OracleError` (an `AssertionError`, i.e. a failed test).

Calling convention of every `cere_ktest_*` entry point (the contract with the kernel track):
    int cere_ktest_<name>(<args...>, cudaStream_t stream);
returns 0 on success and a nonzero `cere_status` otherwise; device buffers are passed as `void *` (a CUDA tensor's
`data_ptr()`), integers as `int64_t`, scalars as `float` (or `double` through `F64`), and the stream is always the
last argument (`call` appends the current torch stream). Each suite file states the C prototype of every symbol it
calls.

Tolerance rule: `calibrate(...)` computes `c` = rel-L2(SGLang op, in-test high-precision reference) and writes it to
the report BEFORE the native kernel is called; the limit is `calibrated_limit(c, floor)` = max(floor, 2c), floor 4e-3
for single ops and 1e-2 for a whole GDN layer or MoE block; `check_within` asserts against that limit. Bit-exact
items use `check_bit_exact`. A miss is a failed test, never a widened limit.

Port rows (tasks/m22n-spec.md D14): a row whose reference is the kernel the control actually dispatches may instead
use `port_limit(value, reason)`, an absolute rel-L2 limit written as literals in the suite (pre-registered, never
derived from a result), or `check_ulp(..., max_fraction=<literal>)`: at most that fraction of elements differ, each
by at most 1 ulp of the tensor's dtype. The static checker (tests/test_native_oracles_static.py) accepts exactly
these forms.

Report rows are printed as one `CERE_ORACLE_ROW <json>` line each on stdout (collected by native_dev.py); they hold
scalars only (no tensors, no token IDs, no text). Every real-model load that seals prints one `CERE_ORACLE_LOAD
attach=<0|1> ms=<ALLOCATE to sealed> mode=<fresh|create|attach> class=<module.Class|->` line on stderr (native_dev.py's
`oracle_loads`).

Dev loop (M2.16 L4d, dev lever 8; `native_dev.py test`): `dev_share.py` (the env contract, the classes that may
attach, the per-file check).
"""
from __future__ import annotations

import bisect
import ctypes
import functools
import json
import math
import os
import re
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import torch
from safetensors import safe_open

import dev_share

KTEST_LIB = Path(os.environ.get("CERE_KTEST_LIB", "/out/libcere_ktest.so"))
MODEL_DIR = Path(os.environ.get("CERE_MODEL_DIR", "/model"))
DEVICE = torch.device("cuda:0")
ROW_PREFIX = "CERE_ORACLE_ROW "
FLOOR_OP = 4e-3
FLOOR_BLOCK = 1e-2
SCALAR_TYPES = (str, int, float, bool, type(None))


class OracleError(AssertionError):
    """A precondition of an oracle test failed (library, symbol, weight, input). It fails the test; never a skip."""


class F64(float):
    """A scalar passed to a `cere_ktest_*` entry point as `double` instead of `float`."""


# -- native entry points ----------------------------------------------------------------------------------------


@functools.cache
def load_ktest() -> ctypes.CDLL:
    """`libcere_ktest.so` from the build mount; absence fails the calling test."""
    if not KTEST_LIB.is_file():
        raise OracleError(f"{KTEST_LIB} is absent (run native_dev.py build first)")
    return ctypes.CDLL(str(KTEST_LIB), mode=ctypes.RTLD_LOCAL)


# The build's own loader: a suite that swaps `oracle.load_ktest` for a merge-base library (`merge_base_library`) loads
# that library's models fresh (it predates the dev-loop attach, M2.16 L4d).
_BUILD_KTEST = load_ktest


def ktest(name: str):
    """The `cere_ktest_*` function `name`; an absent symbol raises OracleError (the test fails, it does not skip)."""
    if not name.startswith("cere_ktest_"):
        raise OracleError(f"{name} is not a cere_ktest_* entry point")
    lib = load_ktest()
    try:
        fn = getattr(lib, name)
    except AttributeError:
        raise OracleError(f"{name} is absent from {KTEST_LIB}") from None
    fn.restype = ctypes.c_int
    return fn


def _arg(value):
    if isinstance(value, torch.Tensor):
        if not value.is_contiguous():
            raise OracleError("tensor arguments must be contiguous")
        return ctypes.c_void_p(value.data_ptr())
    if isinstance(value, bool):
        return ctypes.c_int64(int(value))
    if isinstance(value, int):
        return ctypes.c_int64(value)
    if isinstance(value, F64):
        return ctypes.c_double(value)
    if isinstance(value, float):
        return ctypes.c_float(value)
    if value is None:
        return ctypes.c_void_p(0)
    raise OracleError(f"unsupported ktest argument type {type(value).__name__}")


def call_status(name: str, *args, synchronize: bool = True) -> int:
    """Calls `name(*args, stream)` on the current torch stream and synchronizes; the status, not raised.
    `synchronize=False` skips torch's synchronize (after a sticky device fault, which every later CUDA call reports:
    the `faults` suite, OH-08)."""
    fn = ktest(name)
    stream = torch.cuda.current_stream(DEVICE)
    rc = fn(*[_arg(a) for a in args], ctypes.c_void_p(stream.cuda_stream))
    if synchronize:
        stream.synchronize()
    return int(rc)


# Reported only, never a verdict: per entry point, [calls, wall seconds] of `call` (the call, its status check and the
# stream synchronize), e.g. K8's `digest wall` row.
CALL_WALL: dict[str, list] = {}


def _timed_status(name: str, *args) -> int:
    """`call_status` with its wall in CALL_WALL."""
    start = time.perf_counter()
    rc = call_status(name, *args)
    wall = CALL_WALL.setdefault(name, [0, 0.0])
    wall[0] += 1
    wall[1] += time.perf_counter() - start
    return rc


def call(name: str, *args) -> None:
    """Calls `name(*args, stream)` on the current torch stream, checks the status and synchronizes."""
    rc = _timed_status(name, *args)
    if rc != 0:
        raise OracleError(f"{name} returned status {rc}")


def empty(shape, dtype) -> torch.Tensor:
    """An output buffer on GPU0, filled with NaN (floating) or 0xFF bytes so unwritten elements are visible."""
    t = torch.empty(shape, dtype=dtype, device=DEVICE)
    if t.is_floating_point() and t.element_size() >= 2:
        t.fill_(float("nan"))
    else:
        t.view(torch.uint8).fill_(0xFF)
    return t


# -- real weights (read-only, safetensors) --------------------------------------------------------------------


@functools.cache
def _weight_map() -> dict:
    index = MODEL_DIR / "model.safetensors.index.json"
    if not index.is_file():
        raise OracleError(f"{index} is absent (the model directory must be mounted read-only at {MODEL_DIR})")
    return json.loads(index.read_text())["weight_map"]


def real_tensor(name: str, device: torch.device | str | None = DEVICE) -> torch.Tensor:
    """One checkpoint tensor by its safetensors name (e.g. `model.language_model.layers.0.mlp.gate.weight`)."""
    weight_map = _weight_map()
    if name not in weight_map:
        raise OracleError(f"tensor {name} is not in the checkpoint index")
    with safe_open(str(MODEL_DIR / weight_map[name]), framework="pt", device="cpu") as f:
        t = f.get_tensor(name)
    return t if device is None else t.to(device)


def has_tensor(name: str) -> bool:
    return name in _weight_map()


@functools.cache
def text_config() -> dict:
    """The model's `text_config` (config.json read-only)."""
    return json.loads((MODEL_DIR / "config.json").read_text())["text_config"]


def layer_prefix(layer: int) -> str:
    return f"model.language_model.layers.{layer}."


# The `linear_attn` tensors of one GDN layer, by their name after `layers.<id>.linear_attn.`.
GDN_LAYER_TENSORS = ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight",
                     "conv1d.weight", "A_log", "dt_bias", "norm.weight", "out_proj.weight")


def linear_attention_layers() -> tuple[int, ...]:
    """The ids whose `text_config.layer_types` entry is `linear_attention` (the GDN layers), ascending."""
    return tuple(i for i, kind in enumerate(text_config()["layer_types"]) if kind == "linear_attention")


def gdn_layer_weights(layer: int, device: torch.device | str | None = DEVICE) -> dict[str, torch.Tensor]:
    """One GDN layer's checkpoint tensors by layer id (tasks/m22d-spec.md D5): {name in GDN_LAYER_TENSORS: tensor},
    dtype and shape as stored. Refuses (OracleError) an id that is not an index of `layer_types` or whose entry is
    not `linear_attention`."""
    kinds = text_config()["layer_types"]
    if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < len(kinds):
        raise OracleError(f"layer {layer!r} is not a layer id (0..{len(kinds) - 1})")
    if kinds[layer] != "linear_attention":
        raise OracleError(f"layer {layer} is {kinds[layer]}, not linear_attention")
    prefix = layer_prefix(layer) + "linear_attn."
    return {name: real_tensor(prefix + name, device) for name in GDN_LAYER_TENSORS}


def generator(seed: int) -> torch.Generator:
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    return g


# -- the real A4 model on GPU0 (ABI v5, mirrored with ctypes; executor/include/cere_executor.h) ------------------
#
# Shared by the suites that need a whole-model forward (the K9 real-activation rows). Entry points (k8_ktest.cpp):
#     int cere_ktest_model_create(const cere_model_desc *desc /*host*/, uint64_t memory_budget_bytes,
#                                 uint64_t *out_model /*host [1]*/, cudaStream_t stream);
#     int cere_ktest_model_put(uint64_t model, uint64_t tensor_index, uint64_t offset, const void *bytes /*host*/,
#                              uint64_t len, cudaStream_t stream);
#     int cere_ktest_model_seal(uint64_t model, cudaStream_t stream);
#     int cere_ktest_model_release(uint64_t model, cudaStream_t stream);
#     int cere_ktest_model_host_resident(uint64_t model, uint64_t tensor_index, uint64_t *out_ptr /*host [1]*/,
#                                        cudaStream_t stream);
#         M2.16 L4d: `cere_model_host_resident`: out_ptr[0] = the read-only resident bytes of a HOST_PINNED tensor
#         of a model attached to a published shared segment (the tensor is covered, never put); -2 (UNSUPPORTED)
#         when the region is private or created by this process (put it).

SRC = Path(__file__).resolve().parents[3]
FAMILIES_FILE = SRC / "configs/artifacts/qwen4_exp-nvfp4.families.json"
ABI_VERSION = 5
MODEL_QWEN4_EXP_NVFP4 = 1
PLACEMENT_DEVICE = 1
PLACEMENT_HOST_PINNED = 2
PLACEMENT_DISCARD = 3
DTYPES = {"BF16": (1, 2), "F16": (2, 2), "F32": (3, 4), "F64": (4, 8), "F8_E4M3": (5, 1), "F8_E5M2": (6, 1),
          "U8": (7, 1), "I8": (8, 1), "I16": (9, 2), "I32": (10, 4), "I64": (11, 8), "BOOL": (12, 1)}
SSM_DTYPES = {"bfloat16": 1, "float16": 2, "float32": 3}
OUTPUT_GATES = {"sigmoid": 1}
MAX_DIMS = 8
PUT_PIECE = 64 << 20
UPLOAD_THREADS = 4
_U32, _U64, _F64 = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_double


class Qwen4ExpConfig(ctypes.Structure):
    _fields_ = [("struct_size", _U32), ("abi_version", _U32), ("vocab_size", _U32), ("hidden_size", _U32),
                ("num_hidden_layers", _U32), ("full_attention_interval", _U32), ("full_attention_layers", _U64),
                ("num_attention_heads", _U32), ("num_key_value_heads", _U32), ("head_dim", _U32),
                ("output_gate_type", _U32), ("rope_theta", _F64), ("partial_rotary_factor", _F64),
                ("mrope_interleaved", _U32), ("mrope_section", _U32 * 3), ("indexer_n_heads", _U32),
                ("indexer_head_dim", _U32), ("indexer_kv_heads", _U32), ("indexer_compress_ratio", _U32),
                ("indexer_budget", _U32), ("linear_num_key_heads", _U32), ("linear_num_value_heads", _U32),
                ("linear_key_head_dim", _U32), ("linear_value_head_dim", _U32), ("linear_conv_kernel_dim", _U32),
                ("mamba_ssm_dtype", _U32), ("num_experts", _U32), ("num_experts_per_tok", _U32),
                ("moe_intermediate_size", _U32), ("shared_expert_intermediate_size", _U32),
                ("norm_topk_prob", _U32), ("hc_count", _U32), ("hc_lowrank", _U32), ("ple_layer_ids", _U64),
                ("ple_embed_dim", _U32), ("heads_per_ngram", _U32), ("ngram_size", _U32),
                ("ple_conv_kernel_size", _U32), ("ngram_vocab_size_base", _U64), ("split_ngram_parts", _U32),
                ("make_ngram_vocab_size_divisible_by", _U32), ("rms_norm_eps", _F64),
                ("tie_word_embeddings", _U32), ("mtp_num_hidden_layers", _U32), ("max_position_embeddings", _U64)]


class TensorDesc(ctypes.Structure):
    _fields_ = [("struct_size", _U32), ("abi_version", _U32), ("name", ctypes.c_void_p), ("dtype", _U32),
                ("placement", _U32), ("ndim", _U32), ("reserved", _U32), ("shape", _U64 * MAX_DIMS),
                ("bytes", _U64)]


class ModelDesc(ctypes.Structure):
    _fields_ = [("struct_size", _U32), ("abi_version", _U32), ("model_kind", _U32),
                ("speculative_draft_tokens", _U32), ("config", ctypes.c_void_p), ("tensors", ctypes.c_void_p),
                ("tensor_count", _U64), ("max_sessions", _U64), ("max_context_tokens", _U64),
                ("stop_token_ids", ctypes.c_void_p), ("stop_token_count", _U64), ("kv_pool_tokens", _U64),
                ("draft_token_map", ctypes.c_void_p), ("draft_token_map_count", _U64),
                ("snapshot_host_bytes", _U64), ("prefix_state_slots", _U64), ("vision_config", ctypes.c_void_p),
                ("selection", ctypes.c_void_p), ("family", _U32), ("calibrate_windows", _U32)]


# ABI v5 slot selection (SelfTune ST2, crates/cere-runtime/src/levers.rs): CERE_SLOT_* by the registry lever it
# selects; `draft_tokens` carries its window in params[0] (impl CERE_LEVER_DRAFT_WINDOW = 0).
SLOTS = {"k2_gemm": 1, "prefill_attention": 2, "decode_step": 3, "moe_gemm": 4, "draft_tokens": 5,
         "verify_attention": 6, "k2_weights": 7}
SLOT_MAX = 16
SLOT_BLOCK_VERSION = 1


class SlotChoice(ctypes.Structure):
    """struct cere_slot_choice (ABI v5, 24 B): one slot's impl (a CERE_LEVER_* value) and its typed params."""
    _fields_ = [("slot", _U32), ("impl", _U32), ("params", _U32 * 4)]


class Levers(ctypes.Structure):
    """struct cere_slot_selection (ABI v5, 400 B): the model's slot choices (an unlisted slot is at its default)."""
    _fields_ = [("struct_size", _U32), ("abi_version", _U32), ("block_version", _U32), ("count", _U32),
                ("choices", SlotChoice * SLOT_MAX)]


def levers(**values: int) -> Levers:
    """A cere_slot_selection listing `values` (registry lever name -> CERE_LEVER_* value; `draft_tokens` -> its
    window), in slot order; every other slot at its default."""
    sel = Levers(struct_size=ctypes.sizeof(Levers), abi_version=ABI_VERSION, block_version=SLOT_BLOCK_VERSION)
    for name in sorted(values, key=lambda n: SLOTS[n]):
        c = sel.choices[sel.count]
        c.slot = SLOTS[name]
        if name == "draft_tokens":
            c.params[0] = values[name]
        else:
            c.impl = values[name]
        sel.count += 1
    return sel


class VisionConfig(ctypes.Structure):
    """struct cere_qwen4_exp_vision_config (80 B, ledger N)."""
    _fields_ = [(name, _U32) for name in (
        "struct_size", "abi_version", "depth", "hidden_size", "intermediate_size", "num_heads", "patch_size",
        "temporal_patch_size", "spatial_merge_size", "in_channels", "out_hidden_size", "num_position_embeddings",
        "deepstack_count", "hidden_act", "image_token_id", "video_token_id", "vision_start_token_id",
        "vision_end_token_id")] + [("layer_norm_eps", ctypes.c_double)]


ACT_GELU_TANH = 1  # CERE_ACT_GELU_TANH
VISION_HIDDEN_ACTS = {"gelu_pytorch_tanh": ACT_GELU_TANH}
VISION_LN_EPS = 1e-6  # SGLang's vision LayerNorm eps


def vision_config(**override) -> torch.Tensor:
    """cere_qwen4_exp_vision_config from $ART/config.json (`vision_config` and the top-level token ids), with
    `override` fields set last."""
    config = json.loads((MODEL_DIR / "config.json").read_text())
    vc = config["vision_config"]
    if vc["hidden_act"] not in VISION_HIDDEN_ACTS:
        raise OracleError(f"vision hidden_act {vc['hidden_act']!r} has no CERE_ACT_* value")
    if ctypes.sizeof(VisionConfig) != 80:
        raise OracleError(f"VisionConfig is {ctypes.sizeof(VisionConfig)} B, not 80")
    buf, (v,) = struct_buffer(VisionConfig)
    v.struct_size = ctypes.sizeof(VisionConfig)
    v.abi_version = ABI_VERSION
    for key in ("depth", "hidden_size", "intermediate_size", "num_heads", "patch_size", "temporal_patch_size",
                "spatial_merge_size", "in_channels", "out_hidden_size", "num_position_embeddings"):
        setattr(v, key, int(vc[key]))
    v.deepstack_count = len(vc.get("deepstack_visual_indexes", []))
    v.hidden_act = VISION_HIDDEN_ACTS[vc["hidden_act"]]
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        setattr(v, key, int(config[key]))
    v.layer_norm_eps = VISION_LN_EPS
    for key, value in override.items():
        setattr(v, key, value)
    return buf


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
    """A zeroed CPU tensor holding `count` ctypes `ctype` values and the ctypes view over it (the tensor is what
    `call` passes, so the struct bytes and the pointer stay one object)."""
    buf = torch.zeros(ctypes.sizeof(ctype) * count, dtype=torch.uint8)
    return buf, (ctype * count).from_address(buf.data_ptr())


def _a4_config(tc: dict) -> torch.Tensor:
    """cere_qwen4_exp_config from text_config by the header's rules (lists as bitmasks, strings as enums)."""
    buf, (cfg,) = struct_buffer(Qwen4ExpConfig)
    cfg.struct_size = ctypes.sizeof(Qwen4ExpConfig)
    cfg.abi_version = ABI_VERSION
    for key in CONFIG_INT_KEYS:
        setattr(cfg, key, int(tc.get(key, 0)))
    cfg.full_attention_layers = sum(1 << i for i, kind in enumerate(tc["layer_types"]) if kind == "full_attention")
    gate = tc.get("output_gate_type")
    if gate is not None and gate not in OUTPUT_GATES:
        raise OracleError(f"output_gate_type {gate!r} has no CERE_OUTPUT_GATE_* value")
    cfg.output_gate_type = OUTPUT_GATES.get(gate, 0)
    rope = tc["rope_parameters"]
    cfg.rope_theta = float(rope["rope_theta"])
    cfg.partial_rotary_factor = float(rope["partial_rotary_factor"])
    cfg.mrope_interleaved = int(bool(rope.get("mrope_interleaved", False)))
    for i, value in enumerate(rope["mrope_section"]):
        cfg.mrope_section[i] = int(value)
    ssm = tc.get("mamba_ssm_dtype")
    if ssm is not None and ssm not in SSM_DTYPES:
        raise OracleError(f"mamba_ssm_dtype {ssm!r} has no CERE_DTYPE_* value")
    cfg.mamba_ssm_dtype = SSM_DTYPES.get(ssm, 0)
    cfg.norm_topk_prob = int(bool(tc.get("norm_topk_prob", True)))
    cfg.ple_layer_ids = sum(1 << int(n) for n in tc.get("ple_layer_ids", []))
    cfg.rms_norm_eps = float(tc["rms_norm_eps"])
    return buf


class _TableEntry(NamedTuple):
    name: str
    file: str
    placement: int
    nbytes: int


def _a4_tensor_table(vision: bool = False, draft: bool = False):
    """Every checkpoint tensor in index (name) order but the family file's `unused` ones, placed by the artifact's
    family rules (first match): the entries and the cere_tensor_desc array over them (with its name buffer). With
    `vision` the vision family (`model.visual.*`, `unused` for a text-only desc) is DEVICE, as cered's vision model
    spec places it (tasks/m213-spec.md C7); with `draft` the draft family (`mtp.*`) is DEVICE, as cered places it
    for `speculative_draft_tokens` 4 (tasks/m29-spec.md D3), else DISCARD."""
    weight_map = _weight_map()
    rules = []
    for rule in json.loads(FAMILIES_FILE.read_text())["rules"]:
        placement = PLACEMENT_DISCARD
        if vision and rule["component"] == "vision":
            placement = PLACEMENT_DEVICE
        elif rule["placement"] == "unused":
            # Out of the desc, as cered's model spec leaves it (a text-only desc carries no vision tensor, C7).
            placement = None
        elif draft and rule["component"] == "draft":
            placement = PLACEMENT_DEVICE
        elif rule["component"] == "target":
            placement = {"device": PLACEMENT_DEVICE, "host_pinned": PLACEMENT_HOST_PINNED}[rule["placement"]]
        rules.append((re.compile(rule["pattern"]), placement))
    by_file: dict[str, list[str]] = {}
    for name in weight_map:
        by_file.setdefault(weight_map[name], []).append(name)
    header: dict[str, tuple[str, list[int]]] = {}
    for file, names in by_file.items():
        with safe_open(str(MODEL_DIR / file), framework="pt", device="cpu") as f:
            for name in names:
                piece = f.get_slice(name)
                header[name] = (piece.get_dtype(), list(piece.get_shape()))
    placements = {}
    for name in sorted(weight_map):
        match = next(((p,) for pattern, p in rules if pattern.fullmatch(name)), None)
        if match is None:
            raise OracleError(f"{name} matches no rule of {FAMILIES_FILE.name}")
        if match[0] is not None:
            placements[name] = match[0]
    names = list(placements)
    name_buf = torch.frombuffer(bytearray(b"".join(n.encode() + b"\0" for n in names)), dtype=torch.uint8)
    desc_buf, descs = struct_buffer(TensorDesc, len(names))
    entries = []
    offset = 0
    for i, name in enumerate(names):
        placement = placements[name]
        dtype, shape = header[name]
        if dtype not in DTYPES:
            raise OracleError(f"{name}: dtype {dtype} has no CERE_DTYPE_* value")
        if len(shape) > MAX_DIMS:
            raise OracleError(f"{name}: rank {len(shape)} exceeds CERE_TENSOR_MAX_DIMS")
        code, nbytes = DTYPES[dtype]
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
        entries.append(_TableEntry(name, weight_map[name], placement, nbytes))
    return entries, desc_buf, name_buf


def stop_token_ids() -> list[int]:
    eos = json.loads((MODEL_DIR / "generation_config.json").read_text())["eos_token_id"]
    return [int(t) for t in (eos if isinstance(eos, list) else [eos])]


def _upload_file(handle: int, file: str, items: list) -> None:
    """Puts every non-discarded tensor of one safetensors file in pieces of PUT_PIECE bytes."""
    with safe_open(str(MODEL_DIR / file), framework="pt", device="cpu") as f:
        for index, entry in items:
            flat = f.get_tensor(entry.name).contiguous().reshape(-1).view(torch.uint8)
            if flat.numel() != entry.nbytes:
                raise OracleError(f"{entry.name}: {flat.numel()} bytes read, header says {entry.nbytes}")
            for offset in range(0, entry.nbytes, PUT_PIECE):
                piece = flat[offset:offset + PUT_PIECE]
                call("cere_ktest_model_put", handle, index, offset, piece, piece.numel())


def a4_model_desc(max_context_tokens: int, *, max_sessions: int = 1, kv_pool_tokens: int = 0,
                  snapshot_host_bytes: int = 0, vision: bool = False, prefix_state_slots: int = 0,
                  draft_token_map: torch.Tensor | None = None) -> tuple:
    """The A4 `cere_model_desc` for `max_sessions` sessions of `max_context_tokens` over a paged KV pool of
    `kv_pool_tokens` (0 = the derived pool, tasks/m25-spec.md D4), a pinned snapshot arena of `snapshot_host_bytes`
    (M2.11; 0 = none) and `prefix_state_slots` (M2.10 D6′; 0 = cache off), with `vision` the A4 vision entry
    (tasks/m213-spec.md C7: `vision_config` and the DEVICE `model.visual.*` tensors): (the desc's CPU tensor, the
    tensor table entries, the buffers it points into, which the caller keeps alive while the desc is in use). With
    `draft_token_map` (int32 CPU, strictly increasing ids) the desc asks for 4 speculative draft tokens over that map
    and places the `mtp.*` tensors DEVICE (M2.9 D3)."""
    config_buf = _a4_config(text_config())
    entries, desc_buf, name_buf = _a4_tensor_table(vision, draft_token_map is not None)
    stops = torch.tensor(stop_token_ids(), dtype=torch.int32)
    vision_buf = vision_config() if vision else None
    model_buf, (desc,) = struct_buffer(ModelDesc)
    desc.struct_size = ctypes.sizeof(ModelDesc)
    desc.abi_version = ABI_VERSION
    desc.model_kind = MODEL_QWEN4_EXP_NVFP4
    desc.config = config_buf.data_ptr()
    desc.tensors = desc_buf.data_ptr()
    desc.tensor_count = len(entries)
    desc.max_sessions = max_sessions
    desc.max_context_tokens = max_context_tokens
    desc.kv_pool_tokens = kv_pool_tokens
    desc.stop_token_ids = stops.data_ptr()
    desc.snapshot_host_bytes = snapshot_host_bytes
    desc.prefix_state_slots = prefix_state_slots  # M2.10 D6′ (0 = cache off)
    desc.stop_token_count = stops.numel()
    desc.vision_config = None if vision_buf is None else vision_buf.data_ptr()
    if draft_token_map is not None:
        desc.speculative_draft_tokens = 4
        desc.draft_token_map = draft_token_map.data_ptr()
        desc.draft_token_map_count = draft_token_map.numel()
    return model_buf, entries, (config_buf, desc_buf, name_buf, stops, vision_buf, draft_token_map)


CERE_ERR_ARGUMENT = -1
CERE_ERR_UNSUPPORTED = -2
# handle -> (ALLOCATE start, the dev-loop share it was created on or None, its class), from create to seal or release.
_LOADS = dev_share.Loads()


@functools.cache
def _built_dev_share() -> dev_share.DevShare | None:
    share = dev_share.from_env(os.environ)
    if share is not None:
        dev_share.check_build(share, KTEST_LIB.parent / "libcere_executor.so", KTEST_LIB)
    return share


def dev_loop_share() -> dev_share.DevShare | None:
    """The dev-loop share native_dev.py set (M2.16 L4d, dev lever 8; dev_share.py), once the build's libraries in
    KTEST_LIB's directory hash to the build the share names (`dev_share.check_build`, once per process; a difference
    fails), while the share directory's stamp names the same artifact and build (`dev_share.stamped`, at every call:
    a GPU0 run of another build may replace the segment between two classes of a GPU1 run) and the build's own
    library is loaded (a merge-base library predates the attach). None otherwise: the load is fresh."""
    if load_ktest is not _BUILD_KTEST:
        return None
    try:
        share = _built_dev_share()
    except dev_share.DevShareError as e:
        raise OracleError(str(e)) from None
    return share if share is not None and dev_share.stamped(share) else None


def create_status(model_buf: torch.Tensor, budget: int, out: torch.Tensor, device_id: str | None = None) -> int:
    """The status (not raised) of one ALLOCATE of the desc in `model_buf` at device budget `budget`, the handle in
    out[0]: `cere_ktest_model_create`, or with `device_id` (O47: "0,host_share=<dir>,artifact=<sha256>")
    `cere_ktest_model_create_device` on that id. Without `device_id`, the dev-loop share's (`dev_loop_share`) when set
    and the creating unittest class (`dev_share.creating_class`) is one of dev_share.SHARE_CLASSES; every other class
    loads fresh. A created model's load clock starts here (`upload_and_seal` reports it with its class). This build's
    desc only (ABI v5): a merge-base library is one of the same ABI (SelfTune ST2 D-7 removed the older-desc ladder)."""
    cls = dev_share.creating_class(sys._getframe(1), unittest.TestCase)
    share = None
    if device_id is None:
        share = dev_share.share_for(dev_loop_share, cls)
        device_id = None if share is None else share.device_id()
    started = time.monotonic()

    def create() -> int:
        if device_id is None:
            return call_status("cere_ktest_model_create", model_buf, budget, out)
        id_buf = torch.tensor(list(device_id.encode()) + [0], dtype=torch.uint8)
        return call_status("cere_ktest_model_create_device", model_buf, budget, id_buf, out)

    rc = create()
    if rc == 0 and int(out[0]) != 0:
        _LOADS.begin(int(out[0]), started, share, cls)
    return rc


def allocate(model_buf: torch.Tensor, device_id: str | None = None) -> int:
    """`create_status` at budget = free device memory of the device's CUDA ordinal (`device_id`'s leading ordinal, M3.2N
    D4: "1,..." is GPU1 in a two-card container; GPU0 without one), raised: the ktest handle."""
    out = torch.zeros(1, dtype=torch.int64)
    ordinal = device_id.split(",", 1)[0] if device_id is not None else "0"
    if not ordinal.isdigit():
        raise OracleError(f"device id {device_id!r}: no leading CUDA ordinal")
    free, _ = torch.cuda.mem_get_info(torch.device("cuda", int(ordinal)))
    rc = create_status(model_buf, free, out, device_id)
    if rc != 0:
        raise OracleError(f"ALLOCATE (cere_ktest_model_create*) returned status {rc}")
    handle = int(out[0])
    if handle == 0:
        raise OracleError("cere_ktest_model_create returned a null handle")
    return handle


def allocate_model(max_context_tokens: int, *, max_sessions: int = 1, kv_pool_tokens: int = 0,
                   snapshot_host_bytes: int = 0) -> int:
    """ALLOCATE only: the model's device and pinned host memory taken, nothing uploaded. Returns the ktest handle;
    release with `release_model`."""
    model_buf, _, keep = a4_model_desc(max_context_tokens, max_sessions=max_sessions, kv_pool_tokens=kv_pool_tokens,
                                       snapshot_host_bytes=snapshot_host_bytes)
    return allocate(model_buf)


def _host_resident(handle: int, index: int, nbytes: int) -> memoryview | None:
    """`cere_ktest_model_host_resident` of tensor `index`: its `nbytes` resident bytes in place (no copy), or None
    (UNSUPPORTED: the region is private or created by this process)."""
    out = torch.zeros(1, dtype=torch.int64)
    rc = call_status("cere_ktest_model_host_resident", handle, index, out)
    if rc == CERE_ERR_UNSUPPORTED:
        return None
    if rc != 0:
        raise OracleError(f"cere_ktest_model_host_resident({index}) returned status {rc}")
    return memoryview((ctypes.c_ubyte * nbytes).from_address(int(out[0]))).cast("B")


def _check_file(file: str, digest: str, views: dict) -> None:
    try:
        dev_share.check_file(MODEL_DIR / file, digest, views)
    except dev_share.DevShareError as e:
        raise OracleError(str(e)) from None


def upload_and_seal(handle: int, entries: list, put_file=None) -> None:
    """`upload_and_seal_status`, a nonzero seal status raised (OracleError)."""
    rc = upload_and_seal_status(handle, entries, put_file)
    if rc != 0:
        raise OracleError(f"cere_ktest_model_seal returned status {rc}")


def upload_and_seal_status(handle: int, entries: list, put_file=None) -> int:
    """UPLOAD (concurrent puts of every non-discarded tensor in `entries`, the desc's table, by `put_file(handle, file,
    items)`, default `_upload_file`) and INSTANTIATE (seal) of an allocated model; the seal's status. On the dev-loop
    share (M2.16 L4d) an attached model's all-HOST_PINNED files with a digest are covered in place instead
    (dev_share.resident_views) and each is checked against its manifest digest on its own thread
    (dev_share.check_file) while the rest is put; any difference raises OracleError before the seal, and a segment
    content digest mismatch (verified by the executor during UPLOAD) fails the seal (the caller releases the model in
    both cases). Prints the load line after a successful seal."""
    started, share, cls = _LOADS.end(handle, time.monotonic())
    by_file: dict[str, list] = {}
    for index, entry in enumerate(entries):
        if entry.placement != PLACEMENT_DISCARD:
            by_file.setdefault(entry.file, []).append((index, entry))
    resident: dict[str, dict] = {}
    if share is not None:
        try:
            resident = dev_share.resident_views(by_file, share.file_sha256, functools.partial(_host_resident, handle),
                                                attach_only=share.attach_only)
        except dev_share.DevShareError as e:
            raise OracleError(str(e)) from None
    put = put_file or _upload_file
    with ThreadPoolExecutor(max_workers=max(1, len(resident))) as checks, \
            ThreadPoolExecutor(max_workers=UPLOAD_THREADS) as pool:
        futures = [checks.submit(_check_file, file, share.file_sha256[file], views)
                   for file, views in sorted(resident.items())]
        futures += [pool.submit(put, handle, file, items) for file, items in sorted(by_file.items())
                    if file not in resident]
        for future in futures:
            future.result()
    rc = _timed_status("cere_ktest_model_seal", handle)
    if rc != 0:
        return rc
    mode = dev_share.load_mode(share, bool(resident))
    print(dev_share.load_line(mode, (time.monotonic() - started) * 1000, cls), file=sys.stderr, flush=True)
    return 0


def create_model(max_context_tokens: int, *, max_sessions: int = 1, kv_pool_tokens: int = 0,
                 snapshot_host_bytes: int = 0, vision: bool = False, device_id: str | None = None,
                 prefix_state_slots: int = 0, draft_token_map: torch.Tensor | None = None,
                 model_levers: Levers | None = None) -> int:
    """The real A4 model on GPU0 with `max_sessions` sessions of `max_context_tokens` over a paged KV pool of
    `kv_pool_tokens` (0 = the derived pool, tasks/m25-spec.md D4), a pinned snapshot arena of `snapshot_host_bytes`
    (M2.11; 0 = none), `prefix_state_slots` (M2.10 D6′; 0 = cache off), with `vision`, the vision tower (M2.13 C7)
    and, with `draft_token_map`, the MTP draft layer (M2.9, `a4_model_desc`): ALLOCATE (create, budget = free device
    memory), UPLOAD (concurrent puts), INSTANTIATE (seal). `device_id`: `create_status`. `model_levers` (`levers`): the
    desc's cere_slot_selection (copied by the create; None = null, every slot at its default). Returns the ktest handle; on
    any failure the pair is released before the error propagates. Release with `release_model`."""
    model_buf, entries, keep = a4_model_desc(max_context_tokens, max_sessions=max_sessions,
                                             kv_pool_tokens=kv_pool_tokens, snapshot_host_bytes=snapshot_host_bytes,
                                             vision=vision, prefix_state_slots=prefix_state_slots,
                                             draft_token_map=draft_token_map)
    if model_levers is not None:
        ModelDesc.from_address(model_buf.data_ptr()).selection = ctypes.addressof(model_levers)
    handle = allocate(model_buf, device_id)
    try:
        upload_and_seal(handle, entries)
    except BaseException:
        release_model(handle)
        raise
    return handle


def release_model(handle: int) -> None:
    """`cere_ktest_model_release`: frees every device and pinned byte of the pair."""
    _LOADS.release(handle)
    call("cere_ktest_model_release", handle)


# -- K13: the BF16 MoE block of the MTP layer (m29 design note D3, K2) -----------------------------------------


def moe_bf16_reference(x: torch.Tensor, router: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor,
                       shared_gate: torch.Tensor, shared_up: torch.Tensor, shared_down: torch.Tensor,
                       shared_expert_gate: torch.Tensor, top_k: int,
                       router_logits: torch.Tensor | None = None) -> torch.Tensor:
    """The BF16 MoE block (`kernels/k13_moe_bf16.h`) in torch: out bfloat16 [m, H] from x bfloat16 [m, H].

    Weights in the checkpoint layout (bfloat16): router [E, H], gate_up [E, 2 I, H] (gate rows first), down
    [E, H, I], shared_gate/shared_up [S, H], shared_down [H, S], shared_expert_gate [H]. Every product of BF16
    operands is accumulated in FP64 (at least as exact as any FP32 order) and rounded to BF16 where the block
    rounds: router logits (or `router_logits` BF16 [m, E] when given: the caller's GEMM, `_mtp_router_logits`);
    softmax in FP32, slot k = the k-th largest logit (ties: lowest expert id), weights
    renormalized over the selected; per expert gate and up, SwiGLU silu(gate) * up in FP32, down; the weighted slot
    sum in FP32 in slot order; the shared expert's gate/up, SwiGLU and down; out = routed + sigmoid(x . g) * shared
    in FP32."""
    bf16 = torch.bfloat16

    def linear(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (a.to(torch.float64) @ w.to(torch.float64).t()).to(bf16)

    def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        g = gate.float()
        return ((g / (1.0 + torch.exp(-g))) * up.float()).to(bf16)

    m, hidden = x.shape
    inter = gate_up.shape[1] // 2
    logits = (linear(x, router) if router_logits is None else router_logits).float()
    probs = torch.softmax(logits, dim=-1)
    ids = torch.sort(-logits, dim=-1, stable=True).indices[:, :top_k]
    selected = probs.gather(1, ids)
    weights = selected / selected.sum(dim=-1, keepdim=True)
    y = torch.empty((m, top_k, hidden), dtype=bf16, device=x.device)
    for expert in ids.unique().tolist():
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        xe = x[rows]
        h = swiglu(linear(xe, gate_up[expert, :inter]), linear(xe, gate_up[expert, inter:]))
        y[rows, slots] = linear(h, down[expert])
    routed = torch.zeros((m, hidden), dtype=torch.float32, device=x.device)
    for k in range(top_k):
        routed = routed + weights[:, k:k + 1] * y[:, k].float()
    shared = linear(swiglu(linear(x, shared_gate), linear(x, shared_up)), shared_down).float()
    dot = (x.to(torch.float64) @ shared_expert_gate.to(torch.float64)).float()
    gate = 1.0 / (1.0 + torch.exp(-dot))
    return (routed.to(bf16).float() + gate[:, None] * shared).to(bf16)


# -- K14: the MTP draft layer (m29 design note D3, K3; sglang/srt/models/qwen4_exp_mtp.py:73-216) ----------------

MTP_PREFIX = "mtp."
MTP_LAYER = "mtp.layers.0."
DENSE_CONTEXT = 2048  # k6::kMaxContext: up to this many keys every causal position is attended (no QSA selection)


@functools.cache
def mtp_weights() -> dict[str, torch.Tensor]:
    """The 31 `mtp.*` checkpoint tensors (BF16, on DEVICE) by their name after `mtp.`, plus the target's shared
    `embed_tokens` and `lm_head` (the MTP layer has neither of its own: `mtp_use_dedicated_embeddings` is false and
    `Qwen4ExpForCausalLMMTP.lm_head` loads `lm_head.weight`)."""
    names = sorted(n for n in _weight_map() if n.startswith(MTP_PREFIX))
    w = {n[len(MTP_PREFIX):]: real_tensor(n).contiguous() for n in names}
    for name, t in w.items():
        if t.dtype != torch.bfloat16:
            raise OracleError(f"mtp.{name} is {t.dtype}, expected torch.bfloat16")
    w["embed"] = real_tensor("model.language_model.embed_tokens.weight").contiguous()
    w["lm_head"] = real_tensor("lm_head.weight").contiguous()
    attn = "layers.0.self_attn."
    w["qkv"] = torch.cat([w[attn + "q_proj.weight"], w[attn + "k_proj.weight"], w[attn + "v_proj.weight"]], dim=0)
    return w


@functools.cache
def mtp_rope():
    """SGLang's rotary embedding of the MTP layer, built as models/qwen3_5.py:935-956 from the text config (the MTP
    model is a deepcopy of it; `mtp.rope_theta` equals `rope_theta`), and the QSAIndexer fields its selection reads
    (qsa_indexer.py:46-86): (rope, axis map, indexer namespace). Needs the SGLang image and an installed
    `get_context().override_server_args(...)` (the caller's)."""
    import types

    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
    from sglang.srt.layers.rotary_embedding import get_rope
    from sglang.srt.utils.hf_transformers.common import get_rope_config

    tc = text_config()
    ns = types.SimpleNamespace(**tc)
    rope_theta, rope_scaling = get_rope_config(ns)
    if rope_scaling and not ("rope_type" in rope_scaling or "type" in rope_scaling):
        rope_scaling = None
    mtp_theta = tc.get("mtp", {}).get("rope_theta", rope_theta)
    if float(mtp_theta) != float(rope_theta):
        raise OracleError(f"mtp.rope_theta {mtp_theta} != rope_theta {rope_theta}")
    head_dim = int(tc["head_dim"])
    rope = get_rope(head_size=head_dim, rotary_dim=head_dim, max_position=int(tc["max_position_embeddings"]),
                    rope_scaling=rope_scaling, base=rope_theta,
                    partial_rotary_factor=float(getattr(ns, "partial_rotary_factor", 1.0)), is_neox_style=True,
                    dtype=torch.bfloat16).to(DEVICE)
    ratio = int(tc["indexer_compress_ratio"])
    indexer = types.SimpleNamespace(token_topk=int(tc["indexer_budget"]), compress_ratio=ratio,
                                    block_topk=int(tc["indexer_budget"]) // ratio, rotary_emb=rope,
                                    _rope_axis_map_cache=None)
    return rope, QSAIndexer._rope_axis_map(indexer, DEVICE), indexer


def _gemma_rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """GemmaRMSNorm over the last dim in FP32: x * rsqrt(mean(x^2) + eps) * (1 + w) (not rounded)."""
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + weight.float())


def _linear_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A BF16 nn.Linear (no bias): FP32 accumulation, one BF16 rounding of the output."""
    out = torch.empty((*x.shape[:-1], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    flat_x = x.reshape(-1, x.shape[-1])
    flat_o = out.view(-1, w.shape[0])
    wf = w.float()
    for a in range(0, flat_x.shape[0], 4096):
        flat_o[a:a + 4096] = (flat_x[a:a + 4096].float() @ wf.t()).to(torch.bfloat16)
    return out


def _linear_mm32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """`_linear_bf16` as a cuBLAS BF16-input GEMM with FP32 output (torch.mm out_dtype=float32), one BF16 rounding:
    the same math in another FP32 accumulation order (the K14 cross-implementation spread)."""
    out = torch.empty((*x.shape[:-1], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    flat_x = x.reshape(-1, x.shape[-1])
    flat_o = out.view(-1, w.shape[0])
    for a in range(0, flat_x.shape[0], 4096):
        flat_o[a:a + 4096] = torch.mm(flat_x[a:a + 4096], w.t(), out_dtype=torch.float32).to(torch.bfloat16)
    return out


def hc_mix_reference(hyper: torch.Tensor, prefix: str, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """GatedResidual.mix (sglang/srt/layers/hyperconnection.py:233-289): per 2,560-wide branch Gemma RMSNorm -> normed
    (BF16); t = silu(normed @ W_down^T / hc) rounded to BF16 (SGLang's mix kernels keep t in the activation dtype:
    the jit hc_mix's t_pad scratch, `run_splitk_dense_silu` "bf16(silu(scale * A @ B))"; K3 k3_hc.h); mixed = mean
    over the branches of sigmoid(t @ W_up^T) * normed (FP32, one BF16 rounding). Weights of `mtp_weights()` under
    `prefix` (e.g. `layers.0.attn_hyper_connection.`). Returns (normed, mixed)."""
    w = mtp_weights()
    tc = text_config()
    hc, hidden = int(tc["hc_count"]), int(tc["hidden_size"])
    m = hyper.shape[0]
    normed = _gemma_rms(hyper.view(m, hc, hidden), w[prefix + "hc_norm.weight"].view(hc, hidden), eps)
    normed = normed.to(torch.bfloat16).view(m, hc * hidden)
    t = torch.nn.functional.silu(normed.float() @ w[prefix + "input_mix_weight_down.weight"].float().t() / hc)
    t = t.to(torch.bfloat16).float()
    u = torch.sigmoid(t @ w[prefix + "input_mix_weight_up.weight"].float().t()).view(m, hc, hidden)
    mixed = (u * normed.float().view(m, hc, hidden)).mean(dim=-2).to(torch.bfloat16)
    return normed, mixed


def hc_mix_sglang(hyper: torch.Tensor, prefix: str, eps: float, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """GatedResidual.mix as SGLang dispatches it per call of `chunk` rows: the normed rows of `hc_mix_reference`, then
    at <= 16 rows the Triton `fused_hc_mix` (sglang/srt/layers/hc_mix_triton.py), whose down projection accumulates
    with device-scope atomics (two runs on the same rows can differ: the K14 control repeat floor); above, the eager
    form of the `_mix_compute` SGLang compiles (hyperconnection.py:195: BF16 F.linear outputs, silu, sigmoid, mean)."""
    from sglang.srt.layers.hc_mix_triton import fused_hc_mix

    w = mtp_weights()
    tc = text_config()
    hc, hidden = int(tc["hc_count"]), int(tc["hidden_size"])
    m = hyper.shape[0]
    normed = _gemma_rms(hyper.view(m, hc, hidden), w[prefix + "hc_norm.weight"].view(hc, hidden), eps)
    normed = normed.to(torch.bfloat16).view(m, hc * hidden)
    wd, wu = w[prefix + "input_mix_weight_down.weight"], w[prefix + "input_mix_weight_up.weight"]
    F = torch.nn.functional

    def one(xs: torch.Tensor) -> torch.Tensor:
        if chunk <= 16:
            return fused_hc_mix(xs.contiguous(), wd, wu, hc, hidden).to(torch.bfloat16)
        u = torch.sigmoid(F.linear(F.silu(F.linear(xs, wd) / hc), wu)).unflatten(-1, (hc, hidden))
        return (u * xs.unflatten(-1, (hc, hidden))).mean(dim=-2).to(torch.bfloat16)

    return normed, torch.cat([one(normed[a:a + chunk]) for a in range(0, m, chunk)])


def hc_combine_reference(block_out: torch.Tensor, hyper: torch.Tensor, normed: torch.Tensor, prefix: str):
    """GatedResidual.combine (hyperconnection.py:291-338): hyper + block_out (broadcast over the branches) *
    2 sigmoid(normed @ W_inject^T / hc), FP32, one BF16 rounding."""
    w = mtp_weights()
    tc = text_config()
    hc, hidden = int(tc["hc_count"]), int(tc["hidden_size"])
    m = hyper.shape[0]
    gate = 2.0 * torch.sigmoid(normed.float() @ w[prefix + "block_inject_weight.weight"].float().t() / hc)
    out = hyper.float().view(m, hc, hidden) + block_out.float().unsqueeze(-2) * gate.unsqueeze(-1)
    return out.to(torch.bfloat16).view(m, hc * hidden)


def _qsa_select_prefill(mixed: torch.Tensor, eps: float) -> torch.Tensor:
    """SGLang's QSA token selection of one prefill of T = mixed.shape[0] tokens at positions 0 .. T - 1 (the path
    test_kernels_k9_qsa.py checks): index qk = F.linear(mixed, index_qk_proj); `qsa_index_q_norm_rope_store` and
    `qsa_index_k_compress_store` (sglang/kernels/ops/attention/qsa_indexer.py) for the index q and compressed keys;
    `QSAIndexer.select_prefill_tokens` over `build_qsa_row_ranges`. Returns the selected logical positions [T, W]
    (int, -1 = none)."""
    from sglang.kernels.ops.attention.qsa_indexer import qsa_index_k_compress_store, qsa_index_q_norm_rope_store
    from sglang.srt.layers.attention.qsa.metadata import build_qsa_row_ranges
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    w = mtp_weights()
    tc = text_config()
    rope, axis_map, indexer = mtp_rope()
    heads, dim, ratio = int(tc["indexer_n_heads"]), int(tc["indexer_head_dim"]), int(tc["indexer_compress_ratio"])
    length = mixed.shape[0]
    attn = "layers.0.self_attn.indexer."
    qk = torch.nn.functional.linear(mixed, w[attn + "index_qk_proj.weight"]).contiguous()
    positions = torch.arange(length, device=DEVICE, dtype=torch.int64)
    state = torch.zeros((length, dim), dtype=torch.bfloat16, device=DEVICE)
    rope_positions = torch.zeros((length, 3), dtype=torch.int64, device=DEVICE)
    cos_sin = rope.cos_sin_cache.contiguous()
    q = qsa_index_q_norm_rope_store(qk, positions.unsqueeze(0).expand(3, -1).contiguous(), cos_sin, axis_map,
                                    w[attn + "q_layernorm.weight"], positions, state, rope_positions, heads,
                                    int(rope.rotary_dim), eps, True).contiguous()
    groups = length // ratio
    keys = torch.zeros((groups, dim), dtype=torch.bfloat16, device=DEVICE)
    group_locs = torch.arange(groups * ratio, device=DEVICE, dtype=torch.int32).view(groups, -1)
    write_locs = torch.arange(groups, device=DEVICE, dtype=torch.int32)
    qsa_index_k_compress_store(state, group_locs, rope_positions, cos_sin, axis_map, w[attn + "k_layernorm.weight"],
                               write_locs, keys, ratio, int(rope.rotary_dim), eps, True)
    row_starts, row_ends, _ = build_qsa_row_ranges(
        sequence_lengths=torch.tensor([length], dtype=torch.int32, device=DEVICE), query_positions=positions,
        query_sequence_ids=torch.zeros(length, dtype=torch.int64, device=DEVICE), compress_ratio=ratio)
    seq_lens = torch.full((length,), length, dtype=torch.int32, device=DEVICE)
    return QSAIndexer.select_prefill_tokens(indexer, q, keys.view(-1, 1, dim), row_starts, row_ends, positions,
                                            seq_lens)


def _mtp_attention(mixed: torch.Tensor, chunk: int, eps: float,
                   qkv: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3_5 attention of the MTP layer (Qwen4ExpAttentionDecoderLayer.self_attention, qwen4_exp.py:1500-1545) of
    rows at positions 0 .. T - 1 run in calls of `chunk` rows, as the native MTP layer stores and reads its KV:
    q_proj per head [q 256 | gate 256], k, v (BF16 F.linear); Gemma q/k norm (1 + w) and partial NeoX RoPE (64 of 256)
    by SGLang's own prep kernel `fused_qk_gemma_rmsnorm_rope_gate` (the path test_kernels_k6_attn.py holds K1 to, bit
    for bit); KV stored FP8 E4M3 (scale 1.0). A call with no prefix attends over its own BF16
    K/V, a call with a prefix over the FP8 cache (its own rows included; test_kernels_k6_attn.py). Above
    DENSE_CONTEXT keys (one prefix-0 call only) the keys are SGLang's QSA selection. Softmax(q k^T / 16) GQA 24/2 in
    FP32, BF16 output, sigmoid gate (FP32, BF16), o_proj. `qkv` (BF16 [T, 13,312]) replaces the q|k|v projection of
    `mixed` when given (the K14 stage bisection). Returns (o_proj input BF16 [T, heads x head_dim], block output BF16
    [T, hidden])."""
    w = mtp_weights()
    tc = text_config()
    n_q, n_kv, dim = int(tc["num_attention_heads"]), int(tc["num_key_value_heads"]), int(tc["head_dim"])
    group = n_q // n_kv
    total = mixed.shape[0]
    rope = mtp_rope()[0]
    rotary_dim = int(rope.rotary_dim)
    attn = "layers.0.self_attn."
    if qkv is None:
        qkv = _linear_bf16(mixed, w["qkv"])
    gate = qkv[:, :n_q * 2 * dim].view(total, n_q, 2 * dim)[..., dim:]
    v = qkv[:, n_q * 2 * dim + n_kv * dim:].view(total, n_kv, dim)
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import fused_qk_gemma_rmsnorm_rope_gate

    positions = torch.arange(total, device=mixed.device)
    q, k, _ = fused_qk_gemma_rmsnorm_rope_gate(
        qkv[:, :n_q * 2 * dim], qkv[:, n_q * 2 * dim:n_q * 2 * dim + n_kv * dim], w[attn + "q_norm.weight"],
        w[attn + "k_norm.weight"], rope.cos_sin_cache.contiguous(), positions, eps, n_q, n_kv, dim, rotary_dim,
        has_gate=True)
    q = q.view(total, n_q, dim).float()
    k = k.view(total, n_kv, dim)
    k_fp8 = k.to(torch.float8_e4m3fn).float()
    v_fp8 = v.to(torch.float8_e4m3fn).float()
    scale = dim ** -0.5
    selected = None
    if chunk == total and total > DENSE_CONTEXT:
        selected = _qsa_select_prefill(mixed, eps).to(torch.int64)
    elif chunk != total and total > DENSE_CONTEXT:
        raise OracleError("the MTP oracle runs QSA selection for one prefix-0 call only")
    out = torch.empty((total, n_q, dim), dtype=torch.float32, device=mixed.device)
    for p0 in range(0, total, chunk):
        p1 = min(p0 + chunk, total)
        kk, vv = (k.float(), v.float()) if p0 == 0 else (k_fp8, v_fp8)
        step = 64 if selected is not None else 256
        for a in range(p0, p1, step):
            b = min(a + step, p1)
            qq = q[a:b].view(b - a, n_kv, group, dim)
            if selected is None:
                keys = kk[:b].permute(1, 0, 2)  # [n_kv, b, dim]
                vals = vv[:b].permute(1, 0, 2)
                scores = torch.einsum("rgqd,gkd->rgqk", qq, keys) * scale
                causal = torch.arange(b, device=mixed.device)[None, :] <= torch.arange(a, b, device=mixed.device)[:, None]
                scores = scores.masked_fill(~causal[:, None, None, :], float("-inf"))
                probs = torch.softmax(scores, dim=-1)
                out[a:b] = torch.einsum("rgqk,gkd->rgqd", probs, vals).reshape(b - a, n_q, dim)
            else:
                sel = selected[a:b]
                valid = (sel >= 0) & (sel <= torch.arange(a, b, device=mixed.device)[:, None])
                idx = sel.clamp_min(0)
                ks = kk[idx]  # [r, W, n_kv, dim]
                vs = vv[idx]
                scores = torch.einsum("rgqd,rwgd->rgqw", qq, ks) * scale
                scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
                probs = torch.softmax(scores, dim=-1)
                out[a:b] = torch.einsum("rgqw,rwgd->rgqd", probs, vs).reshape(b - a, n_q, dim)
    gated = (out.to(torch.bfloat16).float() * torch.sigmoid(gate.float())).to(torch.bfloat16).reshape(total, n_q * dim)
    return gated, _linear_bf16(gated, w[attn + "o_proj.weight"])


MTP_STAGES = ("fusion.e", "fusion.hyper", "attn.normed", "attn.mixed", "attn.qkv", "attn.out", "attn.block_out",
              "attn.hyper", "mlp.normed", "mlp.mixed", "mlp.block_out", "mixer.y")
"""The MTP forward's named intermediates (executor/cuda/src/model/mtp.h StageSink, cere_ktest_mtp_forward `stages`)."""


def _mtp_fusion_h(hc_in: torch.Tensor, linear=_linear_bf16) -> torch.Tensor:
    """fc_hidden(GemmaRMSNorm_10240(hc_in) viewed [hc, hidden]) BF16 [T, hc, hidden] (_fuse_residual_linear_shared)."""
    w = mtp_weights()
    tc = text_config()
    hc, hidden = int(tc["hc_count"]), int(tc["hidden_size"])
    eps = float(tc["rms_norm_eps"])
    h = _gemma_rms(hc_in, w["pre_fc_norm_hidden.weight"], eps).to(torch.bfloat16).view(hc_in.shape[0], hc, hidden)
    return linear(h, w["fc_hidden.weight"])


def _mtp_fuse(e: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """x_s = h_s + e (BF16 add: FP32 sum, one rounding) viewed [T, hc x hidden]."""
    return (e.float().unsqueeze(-2) + h.float()).to(torch.bfloat16).view(e.shape[0], -1)


def _mtp_router_logits(mixed: torch.Tensor, chunk: int) -> torch.Tensor:
    """The MTP gate's BF16 router logits as the control dispatches them, one GEMM per call of `chunk` rows: the gate
    is a ReplicatedLinear (BF16, quant_config None; qwen2_moe.py:304, 536) whose UnquantizedLinearMethod.apply is
    F.linear on SM120 (unquant.py:332-393: bf16_gemm_backend auto picks cutedsl only on SM10x). The FP64 product
    differs from it in the top-10 sets of 3 of 4,096 extend rows (the BF16 logits tie exactly at the 10/11 boundary
    on 17-22% of the K14 rows), where native K13 routes as F.linear."""
    router = mtp_weights()["layers.0.mlp.gate.weight"]
    return torch.cat([torch.nn.functional.linear(mixed[a:a + chunk], router)
                      for a in range(0, mixed.shape[0], chunk)])


def _mtp_moe(mixed: torch.Tensor, chunk: int) -> torch.Tensor:
    """The MTP layer's BF16 MoE block (`moe_bf16_reference`, router logits `_mtp_router_logits` per call of `chunk`
    rows) of mixed BF16 [T, hidden], in 1,024-row pieces."""
    w = mtp_weights()
    tc = text_config()
    mlp = "layers.0.mlp."
    logits = _mtp_router_logits(mixed, chunk)
    moe = torch.empty_like(mixed)
    for a in range(0, mixed.shape[0], 1024):
        moe[a:a + 1024] = moe_bf16_reference(
            mixed[a:a + 1024], w[mlp + "gate.weight"], w[mlp + "experts.gate_up_proj"], w[mlp + "experts.down_proj"],
            w[mlp + "shared_expert.gate_proj.weight"], w[mlp + "shared_expert.up_proj.weight"],
            w[mlp + "shared_expert.down_proj.weight"], w[mlp + "shared_expert_gate.weight"].view(-1),
            int(tc["num_experts_per_tok"]), router_logits=logits[a:a + 1024])
    return moe


def _mtp_hot_logits(y: torch.Tensor, draft_map: torch.Tensor, linear=_linear_bf16) -> torch.Tensor:
    """Hot logits y @ lm_head[map]^T (BF16)."""
    hot = mtp_weights()["lm_head"][draft_map.to(device=y.device, dtype=torch.int64)]
    return linear(y, hot)


def mtp_reference(hc_in: torch.Tensor, next_ids: torch.Tensor, draft_map: torch.Tensor, chunk: int,
                  stage=None, *, seed: dict[str, torch.Tensor] | None = None, start: str | None = None,
                  gemm: str = "fp32", mix: str = "ref"):
    """The MTP draft layer of qwen4_exp_mtp.py:73-216 for rows at positions 0 .. T - 1 run in calls of `chunk` rows
    (the attention's KV source per call, `_mtp_attention`): hc_in BF16 [T, hc x hidden] (the target's pre-final-mixer
    hc rows), next_ids int [T] (the token at p + 1), draft_map int [count] (the hot ids). Steps:
    fusion `_fuse_residual_linear_shared` (e = fc_embedding(GemmaRMSNorm_2560(embed(next))); x_s =
    fc_hidden(GemmaRMSNorm_10240(hc_in) viewed [hc, hidden])_s + e); the one full-attention decoder layer
    (attn GatedResidual mix, attention, combine; mlp mix, the BF16 MoE `_mtp_moe`, combine); the
    hyper_connection_mixer mix; hot logits y @ lm_head[map]^T (BF16). Returns (hc_out BF16 [T, hc x hidden], hot
    logits BF16 [T, count]); `stage(name, tensor)`, when given, is called with each MTP_STAGES tensor of the chain.
    Variants (K14 evidence rows): `seed` (tensors by MTP_STAGES name) with `start` (an MTP_STAGES name) takes every
    stage up to and including `start` from `seed` and runs the rest of the chain on them; `gemm="mm32"` runs the
    fusion, qkv, o_proj and lm_head GEMMs as `_linear_mm32`; `mix="sglang"` runs every mix as `hc_mix_sglang`."""
    if (seed is None) != (start is None) or (start is not None and start not in MTP_STAGES):
        raise OracleError(f"seed and start go together and start is an MTP stage, not {start!r}")
    if gemm not in ("fp32", "mm32") or mix not in ("ref", "sglang"):
        raise OracleError(f"unknown MTP chain variant gemm={gemm!r} mix={mix!r}")
    linear = _linear_mm32 if gemm == "mm32" else _linear_bf16
    seeded = set(MTP_STAGES[:MTP_STAGES.index(start) + 1]) if start is not None else set()
    w = mtp_weights()
    tc = text_config()
    eps = float(tc["rms_norm_eps"])
    st = stage if stage is not None else (lambda name, t: None)

    def put(name: str, compute) -> torch.Tensor:
        t = seed[name] if name in seeded else compute()
        st(name, t)
        return t

    def mix_of(hyper: torch.Tensor, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        if mix == "sglang":
            return hc_mix_sglang(hyper, prefix, eps, chunk)
        return hc_mix_reference(hyper, prefix, eps)

    def mixes(hyper: torch.Tensor, prefix: str, names: tuple[str, str]) -> tuple[torch.Tensor, torch.Tensor]:
        pair = (None, None) if all(n in seeded for n in names) else mix_of(hyper, prefix)
        return put(names[0], lambda: pair[0]), put(names[1], lambda: pair[1])

    def embedding() -> torch.Tensor:
        e = _gemma_rms(w["embed"][next_ids.to(torch.int64)], w["pre_fc_norm_embedding.weight"], eps)
        return linear(e.to(torch.bfloat16), w["fc_embedding.weight"])

    layer = "layers.0."
    e = put("fusion.e", embedding)
    fused = put("fusion.hyper", lambda: _mtp_fuse(e, _mtp_fusion_h(hc_in, linear)))
    normed, mixed = mixes(fused, layer + "attn_hyper_connection.", ("attn.normed", "attn.mixed"))
    qkv = put("attn.qkv", lambda: linear(mixed, w["qkv"]))
    gated = put("attn.out", lambda: _mtp_attention(mixed, chunk, eps, qkv)[0])
    attn_out = put("attn.block_out", lambda: linear(gated, w[layer + "self_attn.o_proj.weight"]))
    hyper = put("attn.hyper", lambda: hc_combine_reference(attn_out, fused, normed, layer + "attn_hyper_connection."))
    normed, mixed = mixes(hyper, layer + "mlp_hyper_connection.", ("mlp.normed", "mlp.mixed"))
    moe = put("mlp.block_out", lambda: _mtp_moe(mixed, chunk))
    hc_out = hc_combine_reference(moe, hyper, normed, layer + "mlp_hyper_connection.")
    y = put("mixer.y", lambda: mix_of(hc_out, "hyper_connection_mixer.")[1])
    return hc_out, _mtp_hot_logits(y, draft_map, linear)


def mtp_local_stages(native: dict[str, torch.Tensor], hc_in: torch.Tensor, next_ids: torch.Tensor,
                     hc_out: torch.Tensor, draft_map: torch.Tensor, chunk: int):
    """The K14 stage bisection: yields (name, reference) for each MTP_STAGES step, then `hc_out` and `logits`, with
    the oracle step applied to the native inputs of that step (`native` holds the native MTP_STAGES tensors, `hc_out`
    the native pre-mixer rows), so rel-L2 against the native stage isolates the step. One reference is alive at a
    time besides the inputs."""
    w = mtp_weights()
    tc = text_config()
    eps = float(tc["rms_norm_eps"])
    layer = "layers.0."
    n = native
    e = _gemma_rms(w["embed"][next_ids.to(torch.int64)], w["pre_fc_norm_embedding.weight"], eps).to(torch.bfloat16)
    yield "fusion.e", _linear_bf16(e, w["fc_embedding.weight"])
    yield "fusion.hyper", _mtp_fuse(n["fusion.e"], _mtp_fusion_h(hc_in))
    normed, mixed = hc_mix_reference(n["fusion.hyper"], layer + "attn_hyper_connection.", eps)
    yield "attn.normed", normed
    yield "attn.mixed", mixed
    del normed, mixed
    yield "attn.qkv", _linear_bf16(n["attn.mixed"], w["qkv"])
    yield "attn.out", _mtp_attention(n["attn.mixed"], chunk, eps, n["attn.qkv"])[0]
    yield "attn.block_out", _linear_bf16(n["attn.out"], w[layer + "self_attn.o_proj.weight"])
    yield "attn.hyper", hc_combine_reference(n["attn.block_out"], n["fusion.hyper"], n["attn.normed"],
                                             layer + "attn_hyper_connection.")
    normed, mixed = hc_mix_reference(n["attn.hyper"], layer + "mlp_hyper_connection.", eps)
    yield "mlp.normed", normed
    yield "mlp.mixed", mixed
    del normed, mixed
    yield "mlp.block_out", _mtp_moe(n["mlp.mixed"], chunk)
    yield "hc_out", hc_combine_reference(n["mlp.block_out"], n["attn.hyper"], n["mlp.normed"],
                                         layer + "mlp_hyper_connection.")
    yield "mixer.y", hc_mix_reference(hc_out, "hyper_connection_mixer.", eps)[1]
    yield "logits", _mtp_hot_logits(n["mixer.y"], draft_map)


MTP_GATE_STAGES = (*MTP_STAGES, "hc_out", "logits")
"""The K14 per-stage gate rows: each MTP_STAGES step, the pre-mixer hc rows and the hot logits."""
MTP_STAGE_FLOORS = {name: FLOOR_BLOCK if name == "mlp.block_out" else FLOOR_OP for name in MTP_GATE_STAGES}
"""Predeclared floor of each K14 stage gate: FLOOR_OP per op (GEMM, norm, mix, attention, combine); FLOOR_BLOCK for the
MoE block (router GEMM, top-10/512 routing, experts and the shared expert)."""


def mtp_control_stage(name: str, native: dict[str, torch.Tensor], hc_in: torch.Tensor, next_ids: torch.Tensor,
                      hc_out: torch.Tensor, draft_map: torch.Tensor, chunk: int) -> torch.Tensor | None:
    """The control's dispatched op of one K14 stage on the same native inputs as `mtp_local_stages` (the calibration
    operand: c = rel-L2(this, the oracle step)), per call of `chunk` rows: the GEMMs as SGLang's BF16 F.linear
    (UnquantizedLinearMethod.apply on SM120), the mixes as `hc_mix_sglang`. None for the stages that have no separately
    callable dispatched op here (the norms, the attention, the combines; the MoE block, whose oracle step already
    routes as the control does): their c is 0 and the gate is the predeclared floor."""
    w = mtp_weights()
    eps = float(text_config()["rms_norm_eps"])
    layer = "layers.0."
    n = native

    def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.nn.functional.linear(x[a:a + chunk], weight) for a in range(0, x.shape[0], chunk)])

    if name == "fusion.e":
        e = _gemma_rms(w["embed"][next_ids.to(torch.int64)], w["pre_fc_norm_embedding.weight"], eps)
        return linear(e.to(torch.bfloat16), w["fc_embedding.weight"])
    if name == "fusion.hyper":
        return _mtp_fuse(n["fusion.e"], _mtp_fusion_h(hc_in, linear))
    if name == "attn.mixed":
        return hc_mix_sglang(n["fusion.hyper"], layer + "attn_hyper_connection.", eps, chunk)[1]
    if name == "attn.qkv":
        return linear(n["attn.mixed"], w["qkv"])
    if name == "attn.block_out":
        return linear(n["attn.out"], w[layer + "self_attn.o_proj.weight"])
    if name == "mlp.mixed":
        return hc_mix_sglang(n["attn.hyper"], layer + "mlp_hyper_connection.", eps, chunk)[1]
    if name == "mixer.y":
        return hc_mix_sglang(hc_out, "hyper_connection_mixer.", eps, chunk)[1]
    if name == "logits":
        return _mtp_hot_logits(n["mixer.y"], draft_map, linear)
    if name not in MTP_GATE_STAGES:
        raise OracleError(f"{name!r} is not a K14 stage")
    return None


# -- metrics and the tolerance rule ---------------------------------------------------------------------------


REL_L2_CHUNK = 1 << 24  # elements per FP64 piece of rel_l2 (128 MiB per FP64 operand)


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b||_2 / ||b||_2 in FP64 (b is the reference). The squared sums accumulate in FP64 over REL_L2_CHUNK-element
    pieces of the flattened operands, so a [4,096, 65,536] pair never holds whole FP64 copies."""
    if a.shape != b.shape:
        raise OracleError(f"shape {tuple(a.shape)} != reference shape {tuple(b.shape)}")
    fa = a.detach().reshape(-1)
    fb = b.detach().reshape(-1)
    num = torch.zeros((), dtype=torch.float64, device=fa.device)
    den = torch.zeros((), dtype=torch.float64, device=fa.device)
    finite = True
    for i in range(0, fa.numel(), REL_L2_CHUNK):
        a64 = fa[i:i + REL_L2_CHUNK].to(torch.float64)
        b64 = fb[i:i + REL_L2_CHUNK].to(device=fa.device, dtype=torch.float64)
        finite = finite and bool(torch.isfinite(a64).all())
        num += (a64 - b64).square().sum()
        den += b64.square().sum()
    if not finite:
        return math.inf
    denom = math.sqrt(den.item())
    if denom == 0.0 or not math.isfinite(denom):
        raise OracleError(f"reference norm is {denom}; rel-L2 is undefined")
    return math.sqrt(num.item()) / denom


def calibrated_limit(c: float, floor: float) -> float:
    """The predeclared limit max(floor, 2c) of the tolerance rule."""
    if not (isinstance(c, float) and math.isfinite(c) and c >= 0.0):
        raise OracleError(f"calibration c={c!r} is not a finite non-negative float")
    if floor not in (FLOOR_OP, FLOOR_BLOCK):
        raise OracleError(f"floor {floor!r} is not one of the predeclared floors {FLOOR_OP}, {FLOOR_BLOCK}")
    return max(floor, 2.0 * c)


def port_limit(value: float, reason: str) -> float:
    """A pre-registered absolute rel-L2 limit for a row whose reference is the control's dispatched kernel (D14)."""
    if not (isinstance(value, float) and math.isfinite(value) and 0.0 < value < 1.0):
        raise OracleError(f"port limit {value!r} is not a float in (0, 1)")
    if not (isinstance(reason, str) and reason.strip()):
        raise OracleError("a port limit needs a reason")
    return value


def calibrate(row: str, case: str, reference_op: torch.Tensor, high_precision: torch.Tensor, floor: float) -> float:
    """Calibration step: c = rel-L2(SGLang op, high-precision reference); written to the report before any native
    result exists. Returns c."""
    c = rel_l2(reference_op, high_precision)
    if not math.isfinite(c):
        raise OracleError(f"{row} {case}: the reference op produced non-finite values")
    report_row(row, case, phase="calibration", c=c, floor=floor, limit=calibrated_limit(c, floor))
    return c


def check_within(tc: unittest.TestCase, row: str, case: str, measured: float, *, c: float, limit: float) -> None:
    """Asserts `measured <= limit` (limit from calibrated_limit or port_limit) and reports the result row."""
    passed = math.isfinite(measured) and measured <= limit
    report_row(row, case, phase="result", c=c, limit=limit, measured=measured, passed=passed)
    tc.assertTrue(passed, f"{row} {case}: rel-L2 {measured:.6g} exceeds limit {limit:.6g} (c={c:.6g})")


def check_bit_exact(tc: unittest.TestCase, row: str, case: str, native: torch.Tensor, reference: torch.Tensor) -> None:
    """Bit-exact comparison (dtype, shape and bytes); reports the number of differing elements."""
    tc.assertEqual(native.dtype, reference.dtype, f"{row} {case}: dtype")
    tc.assertEqual(tuple(native.shape), tuple(reference.shape), f"{row} {case}: shape")
    a = native.detach().contiguous().view(torch.uint8).reshape(native.numel(), -1)
    b = reference.detach().contiguous().view(torch.uint8).reshape(reference.numel(), -1)
    differing = int((a != b).any(dim=-1).sum().item()) if native.numel() else 0
    report_row(row, case, phase="bit_exact", elements=native.numel(), differing=differing, passed=differing == 0)
    tc.assertEqual(differing, 0, f"{row} {case}: {differing} of {native.numel()} elements differ")


_ORDERED_BITS = {torch.bfloat16: (torch.int16, 0x7FFF), torch.float16: (torch.int16, 0x7FFF),
                 torch.float32: (torch.int32, 0x7FFFFFFF)}


def _ordered(t: torch.Tensor) -> torch.Tensor:
    """Bit patterns mapped to int64 so that adjacent representable values differ by 1 (+0 and -0 both map to 0)."""
    int_type, magnitude = _ORDERED_BITS[t.dtype]
    bits = t.detach().contiguous().view(int_type).to(torch.int64).reshape(-1)
    mag = bits & magnitude
    return torch.where(bits < 0, -mag, mag)


def check_ulp(tc: unittest.TestCase, row: str, case: str, native: torch.Tensor, reference: torch.Tensor, *,
              max_fraction: float) -> None:
    """At most `max_fraction` of the elements differ from the reference, and each differing element by at most 1 ulp
    of the dtype (bf16, fp16 or fp32); non-finite values must match exactly. Reports the counts."""
    tc.assertEqual(native.dtype, reference.dtype, f"{row} {case}: dtype")
    tc.assertEqual(tuple(native.shape), tuple(reference.shape), f"{row} {case}: shape")
    if native.dtype not in _ORDERED_BITS:
        raise OracleError(f"{row} {case}: check_ulp supports bf16, fp16 and fp32, not {native.dtype}")
    if not (isinstance(max_fraction, float) and 0.0 <= max_fraction < 1.0):
        raise OracleError(f"max_fraction {max_fraction!r} is not a float in [0, 1)")
    a, b = _ordered(native), _ordered(reference)
    finite = torch.isfinite(native.reshape(-1)) & torch.isfinite(reference.reshape(-1))
    distance = torch.where(finite, (a - b).abs(), (a != b).to(torch.int64) * (1 << 40))
    elements = native.numel()
    differing = int((distance != 0).sum().item()) if elements else 0
    max_ulp = int(distance.max().item()) if elements else 0
    fraction = differing / elements if elements else 0.0
    passed = max_ulp <= 1 and fraction <= max_fraction
    report_row(row, case, phase="ulp", elements=elements, differing=differing, fraction=fraction, max_ulp=max_ulp,
               max_fraction=max_fraction, passed=passed)
    tc.assertTrue(passed, f"{row} {case}: {differing} of {elements} elements differ (fraction {fraction:.3g}, "
                          f"limit {max_fraction:g}), max {max_ulp} ulp (limit 1)")


def check_count(tc: unittest.TestCase, row: str, case: str, count: int, *, expected: int) -> None:
    report_row(row, case, phase="count", count=count, expected=expected, passed=count == expected)
    tc.assertEqual(count, expected, f"{row} {case}")


def report_row(row: str, case: str, **fields) -> None:
    """One report line of scalars (no tensors, no token IDs, no text) on stdout."""
    for key, value in fields.items():
        if not isinstance(value, SCALAR_TYPES):
            raise OracleError(f"report field {key} has type {type(value).__name__}; rows hold scalars only")
        if isinstance(value, float) and not math.isfinite(value):
            fields[key] = repr(value)
    line = {"row": row, "case": case, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
    print(ROW_PREFIX + json.dumps(line, sort_keys=True), flush=True)


# -- D6: the host reference of the phase-S accept rule (m29 design note D6; ledger M8 counter; k12 draw's chunked
# fp64 inverse CDF). Shared by K13S (test_kernels_spec_accept.py) and K8Speculation's sampled chains. ------------

D6_VOCAB = 248_320
D6_CHUNKS = 1024  # k12::kChunks
_MASK32 = 0xFFFFFFFF


def d6_kept(words: torch.Tensor) -> torch.Tensor:
    """The bool keep set [D6_VOCAB] of a D3 keep mask (uint32 words as int32)."""
    bits = (words.long().unsqueeze(1) & _MASK32) >> torch.arange(32, device=words.device)
    return (bits & 1).bool().reshape(-1)[:D6_VOCAB]


def _mul32(a: torch.Tensor, c: int) -> torch.Tensor:
    """(a * c) mod 2^32 for a in [0, 2^32) without int64 overflow (16-bit halves of c)."""
    return (a * (c & 0xFFFF) + (((a * (c >> 16)) & 0xFFFF) << 16)) & _MASK32


def _mix(h: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    k = _mul32(k, 0xCC9E2D51)
    k = ((k << 15) | (k >> 17)) & _MASK32
    k = _mul32(k, 0x1B873593)
    h = h ^ k
    h = ((h << 13) | (h >> 19)) & _MASK32
    return (h * 5 + 0xE6546B64) & _MASK32


def _murmur(seeds: torch.Tensor, keys: torch.Tensor, c: int) -> torch.Tensor:
    """murmur_hash32(seed, key, c) for int64 seeds in [0, 2^63) and keys in [0, 2^32)."""
    h = _mix(torch.zeros_like(seeds), seeds & _MASK32)
    h = _mix(h, (seeds >> 32) & _MASK32)
    h = _mix(h, keys)
    h = _mix(h, torch.full_like(seeds, c))
    h = h ^ 16
    h = h ^ (h >> 16)
    h = _mul32(h, 0x85EBCA6B)
    h = h ^ (h >> 13)
    h = _mul32(h, 0xC2B2AE35)
    return h ^ (h >> 16)


def d6_uniforms(seeds: list[int], keys: list[int], ca: int, cb: int) -> list[float]:
    s = torch.tensor(seeds, dtype=torch.int64)
    k = torch.tensor(keys, dtype=torch.int64)
    hi, lo = _murmur(s, k, ca), _murmur(s, k, cb)
    # x = hi * 2^32 + lo; u = (x >> 11) * 2^-53 = (hi * 2^21 + (lo >> 11)) * 2^-53, exact in fp64.
    return [float(v) * 2.0 ** -53 for v in (hi * (1 << 21) + (lo >> 11)).tolist()]


class D6RowRef:
    """One row's fp64 sums as the kernel forms them: per chunk of ceil(vocab / 1024) ids the kept probabilities in
    id order (with the draft: `total`; without: the resample prefix), the chunks summed in chunk order; the running
    sums of each chunk restart at its prefix."""

    def __init__(self, probs: torch.Tensor, kept: torch.Tensor, draft: int | None):
        p = probs.cpu().tolist()
        kept_ids = torch.nonzero(kept).reshape(-1).cpu().tolist()
        self.length = (D6_VOCAB + D6_CHUNKS - 1) // D6_CHUNKS
        self.draft = draft
        self.p_draft = p[draft] if draft is not None and draft < D6_VOCAB else 0.0
        self.draft_kept = draft is not None and draft < D6_VOCAB and bool(kept[draft])
        with_d = [0.0] * D6_CHUNKS
        rest = [0.0] * D6_CHUNKS
        local: list[list[tuple[int, float]]] = [[] for _ in range(D6_CHUNKS)]
        for i in kept_ids:
            c = i // self.length
            with_d[c] += p[i]
            if i != draft:
                rest[c] += p[i]
                local[c].append((i, p[i]))
        self.total = 0.0
        self.prefix = [0.0] * (D6_CHUNKS + 1)
        acc = 0.0
        for c in range(D6_CHUNKS):
            self.total += with_d[c]
            acc += rest[c]
            self.prefix[c + 1] = acc
        self.ids: list[int] = []
        self.running: list[float] = []
        self.start = [0] * (D6_CHUNKS + 1)
        for c in range(D6_CHUNKS):
            self.start[c] = len(self.ids)
            run = self.prefix[c]
            for i, x in local[c]:
                run += x
                self.ids.append(i)
                self.running.append(run)
        self.start[D6_CHUNKS] = len(self.ids)
        self.last_kept = self.ids[-1] if self.ids else 0

    def accept(self, u: float) -> bool:
        return self.draft_kept and u < self.p_draft / self.total

    def draw(self, u: float) -> int:
        target = u * self.prefix[D6_CHUNKS]
        lo = bisect.bisect_right(self.prefix, target, 0, D6_CHUNKS) - 1
        a, b = self.start[lo], self.start[lo + 1]
        at = bisect.bisect_right(self.running, target, a, b)
        if at < b:
            return self.ids[at]
        return self.ids[b] if b < len(self.ids) else self.last_kept


def d6_ref_row(ref: D6RowRef, seeds: list[int], keys: list[int]) -> tuple[list[int], list[int]]:
    """(committed ids, accepted flags) of the D6 rule at each (seed, key)."""
    coins = d6_uniforms(seeds, keys, 4, 5)
    resample = d6_uniforms(seeds, keys, 6, 7)
    ids, flags = [], []
    for u, v in zip(coins, resample):
        ok = ref.accept(u)
        flags.append(int(ok))
        ids.append(ref.draft if ok else ref.draw(v))
    return ids, flags
