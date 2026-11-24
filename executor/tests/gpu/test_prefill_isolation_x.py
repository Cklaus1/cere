"""M2.2p D6 isolation at prefill M (tasks/m22p-spec.md D6, R3; rows X1-X7).

Suite `prefill_isolation` (`native_dev.py test --suite prefill_isolation`, inside the pinned SGLang image on GPU0):
`python3 -m unittest discover -s /src/executor/tests/gpu -p 'test_prefill_isolation_*.py' -v`.

Inputs: native's own D3(b) taps from the p12 and p13 prefill (the whole prompt, one PREFILL, every row), taken by
`cere_ktest_forward_trace_detail` with `detail` 1 (its prototype is declared in `test_model_k8_forward.py`) on the A4
model created, uploaded and sealed as `test_numerics_b8_layers.py` does; the prompts are rendered with the chat
template in memory (`test_model_k8_forward.chat_ids`), never written anywhere. The model is released once the taps are
held on GPU0. Layers: `prefill_isolation_layers.json` (`{"layers": [ints], "source": "<R1/R2 report name>"}`, the
D4 first-exceeding layer, L*-1, the top-3 P-single layers, 0 and 47) when present, else {0, 47}. Every row runs the
native `cere_ktest_*` entry point and the SGLang op the server dispatches at the same M, on the same input:

* X1 HC mix, per layer and site (`attn`, `mlp`): `cere_ktest_hc_mix` on `native.layers.L.<site>_mix.in` vs
  `GatedResidual.mix` as dispatched (M > 16 on SM120: the torch.compile'd `_mix_compute`, run twice for its repeat
  fraction), normed and mixed outputs; control: `fused_hc_mix` (the decode path) vs native on the <= 16-row slices.
  Bit-level.
* X2 every K2 GEMM shape of `executor/cuda/src/model/model.h` (attn_qkv, attn_o, gdn_qkv, gdn_z, gdn_ba, gdn_out,
  ple_key, ple_value, moe_router, moe_shared_gate_up, moe_shared_down, lm_head) plus the HC down (n 320, k 10,240) and
  up (n 10,240, k 320) shapes, at M = the prompt length (21 / 28) and M = 1 (the last row): `cere_ktest_gemm_bf16`
  vs torch `F.linear` with `allow_bf16_reduced_precision_reduction` True (torch's default, which SGLang never
  changes) and False, as bit-equal fraction and rel-L2. Where SGLang runs a fused weight the dispatched op is that
  fused `F.linear` sliced to the row's columns (`dispatched_fused`): GDN `finalize_fused_in_proj` stacks [qkv; z; b;
  a] (n 16,480) for M <= FUSED_IN_PROJ_MAX_M, above which in_proj_qkvz [qkv; z] and in_proj_ba [b; a] run as two
  GEMMs; the shared expert's gate_up is [gate; up] (n 1,280). Inputs are the native taps of the plan's input; attn_o
  reads SGLang's gated attention output of X6, the PLE plans read `native.layers.<first layer>.block.in` as a
  stand-in K = 2,560 activation (the PLE embedding is not a tap); the report names the source. Bit-level (native vs
  the dispatched op), except the GEMMs native never runs through a BF16-out K2 plan at that M (`x2_bit_level`: HC
  down/up run FP32-out inside K3, lm_head has max_m 1): those rows are `criterion: report`. Each row records the
  CUDA kernel names of the native call and of the dispatched `F.linear` (the cuBLAS/cuBLASLt algorithm as launched,
  from the torch profiler).
* X3 router: the native router logits (`model.layers.L.mlp.gate` tap) vs `F.linear(mlp.in, mlp.gate.weight)`
  (tolerance row), then on the native logits `cere_ktest_moe_router` vs `fused_topk` (`moe_fused_gate`): ids set-equal
  per row, weights within 1 FP32 ulp.
* X4 routed experts: `cere_ktest_moe_block` with the shared expert zeroed (its gated term is exactly 0) vs
  `cutlass_fused_moe` (the flashinfer_cutlass path, routing by SGLang's own router) at tactic -1 and at the bench's
  tactic for FlashInfer's MoE bucket of M from `prefill_isolation_tactics.json` (`prefill_tactics.read_tactic`: used
  only when >= 2 logged launches agree, else the row reports `bench_tactic: unavailable`). Each row reports whether
  native routing (its K2 router GEMM + `cere_ktest_moe_router`) and SGLang's (`F.linear` + `fused_topk`) agree; a
  failing row whose routing differs is named so (the router GEMM is X2's `moe_router`, not the experts). Bit-level.
  Each row also checks `routed_reference` (routed experts only loaded) bit-equal to the every-expert load; a
  difference fails the row.
* X5 shared expert + gated add: the block with every `g2_alphas` 0 (each routed term exactly 0) vs `Qwen2MoeMLP`'s
  ops + `fused_gate_sigmoid_mul_add` onto zeros; and the native `mlp.shared_expert.down_proj` tap vs SGLang's shared
  expert output. Tolerance rows.
* X6 attention (full-attention layers): `cere_ktest_attn_block` vs the QSA backend no-prefix extend
  (`QwenSparseAttnBackend.forward_extend` as test_kernels_k6_attn.py builds it) on `native.layers.L.block.in`.
  Tolerance rows (gated attention output, block output).
* X7 GDN prefill (linear-attention layers): `cere_ktest_gdn_layer_prefill` vs the FLA chunk extend path (the fused
  in_proj GEMM for M <= FUSED_IN_PROJ_MAX_M, then test_kernels_k5_gdn.py's extend core and mixer output, one partial
  chunk of 64) on `native.layers.L.block.in`. Tolerance rows.

Tolerance rows report `c` = rel-L2(SGLang op, FP64 reference), `limit` = max(floor, 2c), `measured` = rel-L2(native,
SGLang op), `within`, the bit-equal fraction; they assert nothing. A bit-level row (X1, X2, X4) has `passed` = native
bit-equal to the dispatched op, `sglang_repeat_bit_equal` = the dispatched op's run-to-run bit-equal fraction, and
`lever_pass` = D6's lever clause (bit-equal, or at least the dispatched op's own repeat fraction); each of those tests
reports all its rows first and then fails once, naming every failing row (spec R3: every X row reported, the failing
ones named). X1, X4, X6 and X7 rows also compare the native ktest output with the native forward's own tap of the
same value (`native_vs_forward_*`). No token id or text is reported.

`routed_reference(layer, x, tactic)` (X4's SGLang op, bounded: one layer's routed experts in reused buffers) and
`dispatched_fused(layer, plan)` (X2's fused weights) are shared with test_numerics_prefill.py's screens;
`release_moe()` frees every resident MoE weight and buffer.
"""
from __future__ import annotations

import contextlib
import functools
import importlib
import json
import os
from pathlib import Path
import unittest

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")  # before any flashinfer import, as test_kernels_k5_gdn.py

import torch
import torch.nn.functional as F
from flashinfer.autotuner import AutoTuner
from sglang.srt.layers.activation import silu_and_mul
from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices, qsa_fast_topk
from sglang.srt.layers.attention.qsa.metadata import build_qsa_row_ranges
from sglang.srt.layers.hc_mix_triton import fused_hc_mix, fused_hc_mix_supported
from sglang.srt.model_executor.forward_batch_info import ForwardMode

import oracle

pt = importlib.import_module("prefill_tactics")  # PHarness's stdlib tactics reader (static ALLOWED_TOP rule)
k5 = importlib.import_module("test_kernels_k5_gdn")
k3 = importlib.import_module("test_kernels_k3_hc")
k6 = importlib.import_module("test_kernels_k6_attn")
k7 = importlib.import_module("test_kernels_k7_moe")
k8 = importlib.import_module("test_model_k8_forward")

GPU_DIR = Path(__file__).resolve().parent
LAYERS_FILE = GPU_DIR / "prefill_isolation_layers.json"
TACTICS_FILE = GPU_DIR / "prefill_isolation_tactics.json"
ANCHOR_LAYERS = (0, 47)  # D6: always isolated, with the owner's D4/D5 layers added
PROMPTS = ("p12", "p13")
SITES = ("attn", "mlp")
FIXED_MIX_MAX_ROWS = 16  # SGLang's _FUSED_MIX_MAX_ROWS (hc_mix_triton.py:33): the decode-path slice height
DETAIL_ENTRIES_PER_LAYER = 32
DETAIL_COLS_PER_LAYER = 1 << 17  # value capacity per layer and row of a detail trace (every D3/B8 tap fits)
NAME_BYTES = 128
MOE_TACTIC_OPS = ("trtllm::fused_moe::gemm1", "trtllm::fused_moe::gemm2")
MOE_PROJS = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}  # k7._load_moe_layer's short names
MOE_SCALARS = ("weight_scale_2", "input_scale")  # per-expert F32 scalars, stacked to [num_experts]
QSA_SEED = 61_000
BF16 = torch.bfloat16
FUSED_IN_PROJ_MAX_M = 1024  # qwen3_5.py _forward_input_proj: the fused [in_proj_qkvz; in_proj_ba] GEMM up to M 1024
GDN_IN_PROJ = ("gdn_qkv", "gdn_z", "gdn_ba.b", "gdn_ba.a")  # finalize_fused_in_proj's row order, as X2 row names
GDN_IN_PROJ_TENSORS = tuple(f"linear_attn.in_proj_{p}.weight" for p in ("qkv", "z", "b", "a"))
SHARED_GATE_UP = ("moe_shared_gate_up.gate", "moe_shared_gate_up.up")
SHARED_GATE_UP_TENSORS = ("mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight")
GDN_PLANS = ("gdn_qkv", "gdn_z", "gdn_ba")
NATIVE_FP32_OUT_PLANS = ("hc_down", "hc_up")  # native runs them inside K3 with FP32 output, never as a BF16 K2 plan
NATIVE_LM_HEAD_MAX_M = 1  # model.h: lm_head plan max_m 1 (the last row only)
KERNEL_NAME_CHARS = 200


# -- inputs -------------------------------------------------------------------------------------------------------


def isolation_layers() -> tuple[tuple[int, ...], str]:
    """The D6 layers and their source: the owner's file's layers (D4 first-exceeding, L*-1, top-3 P-single) plus the
    anchors 0 and 47 when the file is present, else the anchors alone."""
    count = int(oracle.text_config()["num_hidden_layers"])
    if not LAYERS_FILE.is_file():
        return ANCHOR_LAYERS, "default {0, 47} (prefill_isolation_layers.json absent)"
    doc = json.loads(LAYERS_FILE.read_text())
    layers = doc.get("layers")
    if not isinstance(layers, list) or not layers or not all(
            isinstance(v, int) and not isinstance(v, bool) and 0 <= v < count for v in layers):
        raise oracle.OracleError(f"{LAYERS_FILE.name}: layers must be a non-empty list of layer ids 0..{count - 1}")
    return tuple(sorted(set(layers) | set(ANCHOR_LAYERS))), f"{doc.get('source', LAYERS_FILE.name)} + {{0, 47}}"


def prompt_ids() -> dict[str, list[int]]:
    """p12 and p13 rendered with the chat template (enable_thinking false) and encoded, in memory only."""
    rows = {}
    for line in k8.PARITY_FILE.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            rows[item["id"]] = item
    missing = [pid for pid in PROMPTS if pid not in rows]
    if missing:
        raise oracle.OracleError(f"{k8.PARITY_FILE.name} lacks {missing}")
    return {pid: k8.chat_ids(rows[pid]["messages"]) for pid in PROMPTS}


def bench_tactic(m: int) -> tuple[object, str]:
    """(tactic, status) of FlashInfer's MoE bucket of m rows from the bench's tactics log (`prefill_tactics`, the one
    bucket rule and parser shared with the writer and the P-moe screen)."""
    return pt.read_tactic(TACTICS_FILE, m)


# -- the native entry points ----------------------------------------------------------------------------------------


def _trace_detail(handle: int, ids: list[int], layers: int) -> dict[str, torch.Tensor]:
    """Native PREFILL of `ids` (chunk 0, every row) with the D3 detail taps: name -> BF16 [P, cols] on GPU0."""
    rows = len(ids)
    capacity = DETAIL_ENTRIES_PER_LAYER * layers + 8
    prompt = torch.tensor(ids, dtype=torch.int32)
    names = torch.zeros(capacity * NAME_BYTES, dtype=torch.uint8)
    shapes = torch.full((capacity, 3), -1, dtype=torch.int64)
    values = oracle.empty((rows * layers * DETAIL_COLS_PER_LAYER,), BF16)
    scores = torch.full((rows, 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_forward_trace_detail", handle, prompt, rows, 0, 0, 1, names, names.numel(), shapes,
                capacity, values, values.numel() * values.element_size(), scores, counts)
    entries, name_bytes, value_bytes = (int(v) for v in counts.tolist())
    if not (0 < entries <= capacity and 0 < name_bytes <= names.numel()
            and 0 < value_bytes <= values.numel() * values.element_size()):
        raise oracle.OracleError(f"forward_trace_detail counts {entries}, {name_bytes}, {value_bytes} exceed "
                                 "the capacities")
    emitted = bytes(names[:name_bytes].tolist()).split(b"\0")
    if len(emitted) != entries + 1 or emitted[-1] != b"":
        raise oracle.OracleError(f"forward_trace_detail wrote {len(emitted) - 1} names for {entries} entries")
    out: dict[str, torch.Tensor] = {}
    element = 0
    for i in range(entries):
        name = emitted[i].decode()
        ndim, r, c = (int(v) for v in shapes[i].tolist())
        if ndim != 2 or r != rows or c <= 0:
            raise oracle.OracleError(f"forward_trace_detail entry {name}: shape record {(ndim, r, c)}")
        if name in out:
            raise oracle.OracleError(f"forward_trace_detail emitted {name} twice")
        out[name] = values[element:element + r * c].view(r, c)
        element += r * c
    if element * values.element_size() != value_bytes:
        raise oracle.OracleError(f"forward_trace_detail value bytes {value_bytes} != the entries' {element * 2}")
    return out


def _native_hc_mix(hyper: torch.Tensor, module, hc: int, hidden: int, lowrank: int, eps: float):
    m = hyper.shape[0]
    normed = oracle.empty((m, hc * hidden), BF16)
    mixed = oracle.empty((m, hidden), BF16)
    oracle.call("cere_ktest_hc_mix", hyper.contiguous(), module.hc_norm.weight.detach(),
                module.input_mix_weight_down.weight.detach(), module.input_mix_weight_up.weight.detach(),
                m, hc, hidden, lowrank, float(eps), normed, mixed)
    return normed, mixed


def _native_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[0]
    if weight.shape[1] != k:
        raise oracle.OracleError(f"GEMM input k {k} != weight k {weight.shape[1]}")
    y = oracle.empty((m, n), BF16)
    oracle.call("cere_ktest_gemm_bf16", x.contiguous(), weight.contiguous(), m, n, k, y)
    return y


def _native_router(logits: torch.Tensor, top_k: int):
    m, num_experts = logits.shape
    ids = oracle.empty((m, top_k), torch.int32)
    weights = oracle.empty((m, top_k), torch.float32)
    oracle.call("cere_ktest_moe_router", logits.contiguous(), ids, weights, m, num_experts, top_k)
    return ids, weights


def _native_moe_block(x: torch.Tensor, w: dict, s: dict, top_k: int) -> torch.Tensor:
    m, hidden = x.shape
    num_experts, inter, _ = w["gate_weight"].shape
    out = oracle.empty((m, hidden), BF16)
    oracle.call("cere_ktest_moe_block", x.contiguous(), w["router"], w["gate_weight"], w["gate_weight_scale"],
                w["up_weight"], w["up_weight_scale"], w["down_weight"], w["down_weight_scale"], s["g1_alphas"],
                s["g2_alphas"], float(s["w13_input_scale_quant"].item()), float(s["w2_input_scale_quant"].item()),
                w["shared_gate"], w["shared_up"], w["shared_down"], w["shared_expert_gate"], out,
                m, hidden, num_experts, top_k, inter, w["shared_gate"].shape[0])
    return out


def _native_attn_block(suite: type, case, w: dict, k_cache: torch.Tensor, v_cache: torch.Tensor):
    """No-prefix extend of all `case.length` rows (kv_from_cache 0), as test_kernels_k6_attn.py `_native_step`."""
    rows = case.length
    attn_out = oracle.empty((rows, suite.n_q_heads * suite.head_dim), BF16)
    out = oracle.empty((rows, suite.hidden_size), BF16)
    oracle.call("cere_ktest_attn_block", case.hidden.contiguous(), w["w_q"], w["w_k"], w["w_v"], w["w_o"],
                w["q_norm"], w["k_norm"], suite.cos_sin, suite.cos_sin.shape[0], case.positions[:rows].contiguous(),
                case.slots[:rows].contiguous(), 0, rows, k_cache, v_cache, k_cache.shape[0], 0, suite.hidden_size,
                suite.n_q_heads, suite.n_kv_heads, suite.head_dim, suite.rotary_dim, suite.eps,
                float(suite.k_scale), suite.scaling, attn_out, out)
    return attn_out, out


def _native_gdn_prefill(w, hidden: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor) -> torch.Tensor:
    tokens = hidden.shape[0]
    out = oracle.empty((tokens, hidden.shape[1]), BF16)
    oracle.call("cere_ktest_gdn_layer_prefill", hidden.contiguous(), w.qkv, w.z, w.b, w.a, w.conv, w.A_log,
                w.dt_bias, w.norm, w.out, conv_state, ssm_state, out, tokens)
    return out


# -- the SGLang ops -------------------------------------------------------------------------------------------------


def _linear(x: torch.Tensor, weights: list[torch.Tensor], reduced: bool) -> torch.Tensor:
    """torch `F.linear` per weight (concatenated by columns) under the given BF16 reduced-precision flag."""
    flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced
    try:
        return torch.cat([F.linear(x, w) for w in weights], dim=-1)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag


def _tactic_pair(tactic) -> tuple[int, int]:
    if isinstance(tactic, int) and not isinstance(tactic, bool):
        if tactic != -1:
            raise oracle.OracleError(f"MoE tactic {tactic}: a logged tactic must be a (gemm1, gemm2) pair")
        return -1, -1
    pair = tuple(tactic)
    if len(pair) != 2 or not all(isinstance(t, int) and not isinstance(t, bool) for t in pair):
        raise oracle.OracleError(f"MoE tactic {tactic!r} is not -1 or a (gemm1, gemm2) pair")
    return pair


@contextlib.contextmanager
def _forced_moe_tactic(tactic):
    """`cutlass_fused_moe` runs `tactic` for its two GEMMs: AutoTuner.choose_one answers both custom ops (an untuned
    process answers -1, the fallback; a served process its autotuned pick). Yields the ops answered."""
    gemm1, gemm2 = _tactic_pair(tactic)
    chosen = dict(zip(MOE_TACTIC_OPS, (gemm1, gemm2)))
    tuner = AutoTuner.get()
    answered: list[str] = []

    def choose_one(custom_op, runners, tuning_config, inputs, **kwargs):
        if custom_op not in chosen:
            raise oracle.OracleError(f"unexpected autotuned op {custom_op} inside cutlass_fused_moe")
        answered.append(custom_op)
        return runners[0], chosen[custom_op]

    tuner.choose_one = choose_one
    try:
        yield answered
    finally:
        del tuner.choose_one


_MOE_FULL: dict = {}  # at most one layer: {layer: (weights, scales, cutlass operands)}
_MOE_SPARSE: dict = {}  # the reused cutlass operand buffers of routed_reference (every expert's shape)


def release_moe() -> None:
    """Drops every resident MoE layer, routed_reference's buffers and the cached routing weights; frees the
    allocator's cache."""
    _MOE_FULL.clear()
    _MOE_SPARSE.clear()
    _moe_routing.cache_clear()
    torch.cuda.empty_cache()


def _moe_layer(layer: int):
    """(weights, flashinfer_cutlass scales, cutlass operands) of one MoE layer on GPU0 (test_kernels_k7_moe.py),
    every expert loaded. One layer stays resident; the previous one is dropped before the next is built."""
    if layer not in _MOE_FULL:
        _MOE_FULL.clear()
        torch.cuda.empty_cache()
        w = k7._load_moe_layer(layer)
        s = k7._reference_scales(w)
        _MOE_FULL[layer] = (w, s, k7._cutlass_operands(w, s))
    return _MOE_FULL[layer]


@functools.lru_cache(maxsize=None)
def _moe_routing(layer: int) -> dict[str, torch.Tensor]:
    """What routing and the flashinfer_cutlass scale set need of one MoE layer, on GPU0: the router weight and every
    expert's F32 `weight_scale_2` / `input_scale` stacked to [num_experts] (k7._load_moe_layer's keys). No expert
    weight: about 2.6 MB per layer."""
    num_experts = int(oracle.text_config()["num_experts"])
    names = [k7._expert_name(layer, e, proj, field) for proj in MOE_PROJS.values() for field in MOE_SCALARS
             for e in range(num_experts)]
    loaded = k7._load_named(names)
    w = {}
    for short, proj in MOE_PROJS.items():
        for field in MOE_SCALARS:
            parts = [loaded.pop(k7._expert_name(layer, e, proj, field)).reshape(()).to(torch.float32)
                     for e in range(num_experts)]
            w[f"{short}_{field}"] = torch.stack(parts).to(oracle.DEVICE).contiguous()
    w["router"] = k7._router_weight(layer)
    return w


def _sparse_buffers(num_experts: int, inter: int, hidden: int) -> dict[str, torch.Tensor]:
    """The cutlass operand buffers for every expert ([up; gate] codes, down codes, both swizzled block scale sets,
    as bytes), allocated once and zeroed on every call."""
    key = (num_experts, inter, hidden)
    if _MOE_SPARSE.get("key") != key:
        _MOE_SPARSE.clear()
        torch.cuda.empty_cache()

        def pad(n, m):
            return (n + m - 1) // m * m

        def zeros(*shape):
            return torch.zeros(shape, dtype=torch.uint8, device=oracle.DEVICE)

        _MOE_SPARSE.update(key=key, w13=zeros(num_experts, 2 * inter, hidden // 2),
                           w2=zeros(num_experts, hidden, inter // 2),
                           sf13=zeros(num_experts, pad(2 * inter, 128), pad(hidden // 16, 4)),
                           sf2=zeros(num_experts, pad(hidden, 128), pad(inter // 16, 4)))
    buffers = {k: v for k, v in _MOE_SPARSE.items() if k != "key"}
    for buffer in buffers.values():
        buffer.zero_()
    return buffers


def _sparse_operands(layer: int, experts: list[int]) -> tuple[dict, dict]:
    """(scales, cutlass operands) of one MoE layer with only `experts` loaded from the checkpoint; every other expert
    is zero in the reused buffers, so the kernel's problem (512 experts, the same shapes) is unchanged. The scale set
    (w13_max, w2_max, alphas) comes from every expert's scalars, exactly as k7._reference_scales."""
    cfg = oracle.text_config()
    num_experts, inter, hidden = int(cfg["num_experts"]), int(cfg["moe_intermediate_size"]), int(cfg["hidden_size"])
    s = k7._reference_scales(_moe_routing(layer))
    b = _sparse_buffers(num_experts, inter, hidden)
    names = [k7._expert_name(layer, e, proj, field) for e in experts for proj in MOE_PROJS.values()
             for field in ("weight", "weight_scale")]
    loaded = k7._load_named(names)

    def get(e, short, field):
        return loaded.pop(k7._expert_name(layer, e, MOE_PROJS[short], field)).view(torch.uint8)

    sf13, sf2 = [], []
    for e in experts:
        b["w13"][e, :inter].copy_(get(e, "up", "weight"))
        b["w13"][e, inter:].copy_(get(e, "gate", "weight"))
        b["w2"][e].copy_(get(e, "down", "weight"))
        sf13.append(torch.cat([get(e, "up", "weight_scale"), get(e, "gate", "weight_scale")], dim=0))
        sf2.append(get(e, "down", "weight_scale"))
    index = torch.tensor(experts, dtype=torch.int64, device=oracle.DEVICE)
    fp8 = torch.float8_e4m3fn
    for key, parts in (("sf13", sf13), ("sf2", sf2)):
        swizzled = k7.swizzle_blockscale(torch.stack(parts).view(fp8)).view(torch.uint8)
        if tuple(swizzled.shape[1:]) != tuple(b[key].shape[1:]):
            raise oracle.OracleError(f"swizzled {key} {tuple(swizzled.shape)} does not fit {tuple(b[key].shape)}")
        b[key][index] = swizzled
    ops = {"w13_weight": b["w13"].view(torch.long), "w2_weight": b["w2"].view(torch.long),
           "quant_scales": [s["w13_input_scale_quant"], b["sf13"].view(torch.int32), s["g1_alphas"],
                            s["w2_input_scale_quant"], b["sf2"].view(torch.int32), s["g2_alphas"]]}
    return s, ops


def _sglang_routed(layer: int, x: torch.Tensor, tactic, ops: dict | None = None):
    """SGLang's routed experts at `tactic`: (output bf16 [M, H], topk_weights, topk_ids). `ops` None: the resident
    full layer (`_moe_layer`); else the given cutlass operands (with the layer's router)."""
    if ops is None:
        w, _, ops = _moe_layer(layer)
    else:
        w = {"router": _moe_routing(layer)["router"]}
    top_k = int(oracle.text_config()["num_experts_per_tok"])
    with _forced_moe_tactic(tactic) as answered:
        out, topk_weights, topk_ids = k7._sglang_routed(x.contiguous(), w, ops, top_k)
    if sorted(answered) != sorted(MOE_TACTIC_OPS):
        raise oracle.OracleError(f"cutlass_fused_moe asked the autotuner for {answered}, not {list(MOE_TACTIC_OPS)}")
    return out, topk_weights, topk_ids


def routed_reference(layer: int, x: torch.Tensor, tactic=-1) -> torch.Tensor:
    """X4's SGLang op: the routed experts' weighted sum (no shared expert) of MLP input `x` (bf16 [M, H] on GPU0)
    as the flashinfer_cutlass path computes it, router and `fused_topk` included, at `tactic` (-1, or a (gemm1,
    gemm2) pair of absolute autotuner ids); tune_max_num_tokens next_pow2(M), fused finalize per SGLang's env.
    Bounded: routes `x` first and loads only the routed experts into reused buffers (X4 checks it bit-equal to the
    every-expert load); no fallback, an allocation failure raises."""
    x = x.contiguous()
    top_k = int(oracle.text_config()["num_experts_per_tok"])
    logits = F.linear(x, _moe_routing(layer)["router"])
    experts = sorted(set(k7._sglang_route(x, logits, top_k)[1].reshape(-1).tolist()))
    _, ops = _sparse_operands(layer, experts)
    out, _, topk_ids = _sglang_routed(layer, x, tactic, ops)
    unloaded = set(topk_ids.reshape(-1).tolist()) - set(experts)
    if unloaded:
        raise oracle.OracleError(f"layer {layer}: cutlass_fused_moe routed to {len(unloaded)} expert(s) not loaded")
    return out


def _fused(layer: int, kind: str) -> tuple[torch.Tensor, dict[str, tuple[int, int]]]:
    """One fused weight SGLang dispatches on `layer` and its X2 row columns, stacked from the checkpoint's BF16
    tensors alone: `gdn` = [qkv; z; b; a] (finalize_fused_in_proj), `shared` = [gate; up] (the shared expert's merged
    gate_up_proj). Nothing is cached here: a caller holds the result for as long as it needs it."""
    names, tensors = (GDN_IN_PROJ, GDN_IN_PROJ_TENSORS) if kind == "gdn" else (SHARED_GATE_UP, SHARED_GATE_UP_TENSORS)
    prefix = oracle.layer_prefix(layer)
    parts = [oracle.real_tensor(prefix + t) for t in tensors]
    cols, start = {}, 0
    for name, tensor, part in zip(names, tensors, parts):
        if part.dtype != BF16 or len(part.shape) != 2:
            raise oracle.OracleError(f"{prefix}{tensor}: {part.dtype} {tuple(part.shape)}, expected a 2-D bfloat16")
        cols[name] = (start, start + part.shape[0])
        start += part.shape[0]
    return torch.cat(parts, dim=0).contiguous(), cols


def dispatched_fused(layer: int, plan: str) -> tuple[torch.Tensor, dict[str, tuple[int, int]]]:
    """(fused weight, {X2 row name: (first, end) output column}) of the fused GEMM SGLang dispatches for K2 `plan` on
    `layer`: the GDN in_proj plans (gdn_qkv, gdn_z, gdn_ba; linear-attention layers; whole for M <=
    FUSED_IN_PROJ_MAX_M, above which [qkv; z] and [b; a] run as row views of it) and moe_shared_gate_up. Read from
    the checkpoint on every call; callers cache it (per layer and kind) for one screen or one layer's X2 rows."""
    kinds = oracle.text_config()["layer_types"]
    if plan in GDN_PLANS:
        if kinds[layer] != "linear_attention":
            raise oracle.OracleError(f"layer {layer} is {kinds[layer]}: no fused GDN in_proj")
        weight, cols = _fused(layer, "gdn")
    elif plan == "moe_shared_gate_up":
        weight, cols = _fused(layer, "shared")
    else:
        raise oracle.OracleError(f"SGLang dispatches no fused weight for plan {plan}")
    return weight, dict(cols)


def _dispatched_view(fused: tuple[torch.Tensor, dict], plan: str, name: str,
                     m: int) -> tuple[torch.Tensor, int, int]:
    """(weight SGLang's GEMM runs for X2 row `name` at M = m, first and end output column in that GEMM), from the
    `dispatched_fused` result `fused` of the row's layer and plan."""
    weight, cols = fused
    a, b = cols[name]
    if plan in GDN_PLANS and m > FUSED_IN_PROJ_MAX_M:
        split = cols["gdn_ba.b"][0]
        lo, hi = (split, weight.shape[0]) if plan == "gdn_ba" else (0, split)
        weight, a, b = weight[lo:hi], a - lo, b - lo
    return weight, a, b


def _sglang_proj_fused(layer: int, w, hidden: torch.Tensor):
    """X7's projections as `_forward_input_proj` dispatches them: (mixed_qkv, z, b, a) from ONE F.linear on the fused
    [qkv; z; b; a] weight for M <= FUSED_IN_PROJ_MAX_M, else in_proj_qkvz and in_proj_ba (`k5._sglang_proj`)."""
    if hidden.shape[0] > FUSED_IN_PROJ_MAX_M:
        return k5._sglang_proj(w, hidden)
    weight, cols = dispatched_fused(layer, "gdn_qkv")
    out = F.linear(hidden, weight)
    return tuple(out[:, cols[name][0]:cols[name][1]].contiguous() for name in GDN_IN_PROJ)


def _qsa_prefill_topk(suite: type, positions: torch.Tensor, seed: int) -> tuple[torch.Tensor, int]:
    """SGLang's QSA token selection of a no-prefix extend (test_kernels_k6_attn.py `_qsa_topk`, prefill rows) and
    the rows whose selection is not exactly causal."""
    rows = positions.numel()
    seq_lens = torch.full((rows,), rows, dtype=torch.int32, device=oracle.DEVICE)
    row_starts, row_ends, _ = build_qsa_row_ranges(
        sequence_lengths=seq_lens[:1], query_positions=positions,
        query_sequence_ids=torch.zeros(rows, dtype=torch.int64, device=oracle.DEVICE),
        compress_ratio=suite.compress_ratio,
    )
    width = int(row_ends.max().item())
    if width == 0:
        blocks = torch.full((rows, suite.block_topk), -1, dtype=torch.int32, device=oracle.DEVICE)
    else:
        logits = torch.randn((rows, width), generator=oracle.generator(seed), device=oracle.DEVICE,
                             dtype=torch.float32)
        blocks = qsa_fast_topk(logits, row_starts, row_ends, topk=suite.block_topk)
    selected = expand_qsa_block_indices(blocks, positions, seq_lens, compress_ratio=suite.compress_ratio,
                                        token_topk=suite.token_topk)
    return selected, k6._causal_mismatch_rows(selected, positions)


# -- comparisons --------------------------------------------------------------------------------------------------


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of elements whose bits are identical (BF16 / FP32 / int32)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        raise oracle.OracleError(f"{tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}")
    return int((_bits(a) == _bits(b)).sum().item()) / a.numel()


def _pair(label: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    return {f"{label}_bit_equal": bit_equal(a, b), f"{label}_rel_l2": oracle.rel_l2(a, b)}


def tolerance_row(row: str, case: str, native: torch.Tensor, sglang_op: torch.Tensor, fp64_ref: torch.Tensor,
                  floor: float, **fields) -> None:
    """A report-only tolerance row: c = rel-L2(SGLang op, FP64), limit = max(floor, 2c), measured = rel-L2(native,
    SGLang op), the bit-equal fraction and the differing elements on identical inputs. Asserts nothing (D6)."""
    c = oracle.rel_l2(sglang_op, fp64_ref)
    limit = oracle.calibrated_limit(c, floor)
    measured = oracle.rel_l2(native, sglang_op)
    oracle.report_row(row, case, phase="isolate", criterion="tolerance", c=c, limit=limit, measured=measured,
                      within=measured <= limit, bit_equal=bit_equal(native, sglang_op), elements=native.numel(),
                      differing=int((_bits(native) != _bits(sglang_op)).sum().item()), **fields)


def _ulp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a - b| in FP32 ulps of finite same-signed values (the int32 bit patterns' distance)."""
    return (a.contiguous().view(torch.int32).long() - b.contiguous().view(torch.int32).long()).abs()


def lever_pass(native_vs_dispatched: float, dispatched_repeat: float) -> bool:
    """D6's lever clause: bit-equal to the dispatched op, or reaching the bit-equal fraction of the dispatched op's
    own run-to-run repeat."""
    return native_vs_dispatched >= min(1.0, dispatched_repeat)


def x2_bit_level(plan: str, m: int) -> bool:
    """Whether an X2 row judges a GEMM native runs as a BF16-out K2 plan at M = m (else the row is report-only)."""
    return plan not in NATIVE_FP32_OUT_PLANS and not (plan == "lm_head" and m > NATIVE_LM_HEAD_MAX_M)


def _kernel_names(fn) -> str:
    """The GEMM kernels `fn` launches (torch profiler, CUPTI: torch's and libcere_ktest's alike), sorted, ';'-joined,
    each cut to KERNEL_NAME_CHARS; torch's own elementwise kernels (the output fill of `oracle.empty`, slice copies,
    `torch.cat`) are left out. "not captured: <why>" when the profiler records none or fails."""
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize(oracle.DEVICE)
    except RuntimeError as e:
        return f"not captured: {type(e).__name__}"
    names = sorted({e.name[:KERNEL_NAME_CHARS] for e in prof.events()
                    if e.device_type == torch.autograd.DeviceType.CUDA
                    and not e.name.startswith(("Memcpy", "Memset")) and "at::native::" not in e.name})
    return ";".join(names) if names else "not captured: no CUDA kernel event"


def _routing(native_ids, native_weights, sg_ids, sg_weights) -> dict:
    """Native vs SGLang routing of the same rows: ids set-equal per row and, when they are, the bit-equal fraction
    of the weights matched by expert id."""
    a_ids, a_order = native_ids.to(torch.int64).sort(dim=-1)
    b_ids, b_order = sg_ids.to(torch.int64).sort(dim=-1)
    ids_equal = bool(torch.equal(a_ids, b_ids))
    weights = None
    if ids_equal:
        weights = bit_equal(native_weights.float().gather(-1, a_order).contiguous(),
                            sg_weights.float().gather(-1, b_order).contiguous())
    return dict(routing_ids_equal=ids_equal, routing_rows_ids_equal=int((a_ids == b_ids).all(dim=-1).sum().item()),
                routing_weights_bit_equal=weights)


def _fail_named(tc: unittest.TestCase, row: str, failed: list[str]) -> None:
    tc.assertFalse(failed, f"{row}: {len(failed)} bit-level row(s) not bit-equal to the dispatched op: "
                           + ", ".join(failed))


# -- the suite ----------------------------------------------------------------------------------------------------


class PrefillIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tc = oracle.text_config()
        cls.hc = int(cls.tc["hc_count"])
        cls.hidden = int(cls.tc["hidden_size"])
        cls.lowrank = int(cls.tc["hc_lowrank"])
        cls.eps = float(cls.tc["rms_norm_eps"])
        cls.top_k = int(cls.tc["num_experts_per_tok"])
        cls.kinds = cls.tc["layer_types"]
        cls.layers, cls.layer_source = isolation_layers()
        # The QSA backend, rotary table and pools read the published server config (fp8 KV, Q8): K6's test-scoped
        # publish, restored at class cleanup; K6's class attributes (rope, scales, head shapes) serve X6.
        k6.K6AttentionBlock.setUpClass()
        cls.addClassCleanup(k6.K6AttentionBlock.tearDownClass)
        cls.attn = k6.K6AttentionBlock()
        cls.attention: dict = {}
        ids = prompt_ids()
        layer_count = int(cls.tc["num_hidden_layers"])
        keep = tuple(f"{kind}.layers.{layer}." for layer in cls.layers for kind in ("model", "native"))
        cls.taps: dict[str, dict[str, torch.Tensor]] = {}
        model = k8.create_model(k8.MAX_CONTEXT)
        try:
            k8.upload_and_seal(model)
            for pid in PROMPTS:
                trace = _trace_detail(model.handle, ids[pid], layer_count)
                cls.taps[pid] = {name: t.clone() for name, t in trace.items()
                                 if name == "model" or name.startswith(keep)}
                del trace
        finally:
            k8.release_model(model.handle)
        torch.cuda.empty_cache()
        oracle.report_row("X", "inputs", phase="inputs", layers=",".join(str(v) for v in cls.layers),
                          layer_source=cls.layer_source, prompts=",".join(PROMPTS),
                          **{f"rows_{pid}": len(ids[pid]) for pid in PROMPTS})

    def tap(self, pid: str, name: str) -> torch.Tensor:
        taps = self.taps[pid]
        if name not in taps:
            raise oracle.OracleError(f"{pid}: the detail trace lacks {name}")
        return taps[name]

    def kind(self, layer: int) -> str:
        return self.kinds[layer]

    # -- X1 ---------------------------------------------------------------------------------------------------

    def _x1_site(self, failed: list[str], layer: int, site: str, pid: str) -> None:
        hc, hidden, lowrank, eps = self.hc, self.hidden, self.lowrank, self.eps
        module = k3._gated_residual(f"{oracle.layer_prefix(layer)}{site}_hyper_connection.", use_combine=False)
        w_down = module.input_mix_weight_down.weight.detach()
        w_up = module.input_mix_weight_up.weight.detach()
        hyper = self.tap(pid, f"native.layers.{layer}.{site}_mix.in")
        m = hyper.shape[0]
        with torch.no_grad():
            sg_mixed, (_, sg_normed) = module.mix(hyper)
            sg_mixed_again, (_, sg_normed_again) = module.mix(hyper)
        if module._jit_mix_ok and m <= 24:
            path = "cute_hc_mix"
        elif fused_hc_mix_supported(sg_normed, w_down, w_up):
            path = "triton_fused_hc_mix"
        else:
            path = "compiled_mix_compute"
        nat_normed, nat_mixed = _native_hc_mix(hyper, module, hc, hidden, lowrank, eps)
        again_normed, again_mixed = _native_hc_mix(hyper, module, hc, hidden, lowrank, eps)
        norm64 = k3._norm_fp64(hyper, module.hc_norm.weight.detach(), hc, hidden, eps)
        mix64 = k3._mix_fp64(norm64, w_down, w_up, hc, hidden)
        c = oracle.rel_l2(sg_mixed, mix64)
        forward_mixed = self.tap(pid, f"native.layers.{layer}.{'block' if site == 'attn' else 'mlp'}.in")
        forward_normed = self.tap(pid, f"model.layers.{layer}.{site}_hyper_connection.hc_norm")
        case = f"L{layer}/{site}_mix/{pid}/M{m}"
        native_vs_sg = min(bit_equal(nat_mixed, sg_mixed), bit_equal(nat_normed, sg_normed))
        sg_repeat = min(bit_equal(sg_mixed_again, sg_mixed), bit_equal(sg_normed_again, sg_normed))
        passed = native_vs_sg == 1.0
        oracle.report_row("X1", case, phase="isolate", criterion="bit", layer=layer, site=site, prompt=pid, m=m,
                          path=path, passed=passed, sglang_repeat_bit_equal=sg_repeat,
                          lever_pass=lever_pass(native_vs_sg, sg_repeat), **_pair("mixed", nat_mixed, sg_mixed),
                          **_pair("normed", nat_normed, sg_normed), c_mixed=c,
                          limit_mixed=oracle.calibrated_limit(c, oracle.FLOOR_OP),
                          native_repeat_bit_equal=min(bit_equal(again_mixed, nat_mixed),
                                                      bit_equal(again_normed, nat_normed)),
                          native_vs_forward_mixed_bit_equal=bit_equal(nat_mixed, forward_mixed),
                          native_vs_forward_normed_bit_equal=bit_equal(nat_normed, forward_normed))
        if not passed:
            failed.append(case)
        # Control: the decode path (fused_hc_mix, <= 16 rows) vs native on the same slices.
        for start in range(0, m, FIXED_MIX_MAX_ROWS):
            stop = min(start + FIXED_MIX_MAX_ROWS, m)
            normed_slice = sg_normed[start:stop].contiguous()
            fused = fused_hc_mix(normed_slice, w_down, w_up, hc, hidden).to(BF16)
            fused_again = fused_hc_mix(normed_slice, w_down, w_up, hc, hidden).to(BF16)
            _, nat_slice = _native_hc_mix(hyper[start:stop], module, hc, hidden, lowrank, eps)
            oracle.report_row("X1", f"{case}/control_rows_{start}_{stop}", phase="control", criterion="report",
                              layer=layer, site=site, prompt=pid, m=stop - start,
                              fused_supported=fused_hc_mix_supported(normed_slice, w_down, w_up),
                              **_pair("native_vs_fused", nat_slice, fused),
                              **_pair("fused_vs_dispatched", fused, sg_mixed[start:stop].contiguous()),
                              fused_repeat_bit_equal=bit_equal(fused_again, fused))

    def test_x1_hc_mix(self):
        failed: list[str] = []
        for layer in self.layers:
            for site in SITES:
                for pid in PROMPTS:
                    self._x1_site(failed, layer, site, pid)
        _fail_named(self, "X1", failed)

    # -- X6 (X2's attn_o input is its SGLang gated attention output) --------------------------------------------

    def _attention(self, layer: int, pid: str) -> dict:
        key = (layer, pid)
        if key in self.attention:
            return self.attention[key]
        suite = type(self.attn)
        w = k6._attn_weights(layer)
        hidden = self.tap(pid, f"native.layers.{layer}.block.in")
        length = hidden.shape[0]
        case = k6._Case(suite, length, seed=QSA_SEED + layer, layer=layer)
        case.hidden = hidden.contiguous()
        k_cache, v_cache = case.cache_bytes()
        q, k, v, gate = self.attn._sglang_prepare(case, 0, length, w=w)
        topk, noncausal = _qsa_prefill_topk(suite, case.positions[:length], QSA_SEED + 100 + layer)
        if noncausal:
            raise oracle.OracleError(f"L{layer} {pid}: QSA selection is not causal on {noncausal} row(s)")
        batch = suite._extend_batch(case, ForwardMode.EXTEND, 0, length)
        attn = case.backend.forward_extend(q, k, v, case.layer, batch, topk_indices=topk)
        sg_attn, sg_out = self.attn._sglang_finish(attn, gate, w)
        ref_attn, ref_out = self.attn._reference(case, length, 0, kv_fp8=False, w=w)
        nat_attn, nat_out = _native_attn_block(suite, case, w, k_cache, v_cache)
        result = {"w": w, "sg_attn": sg_attn.contiguous(), "sg_out": sg_out.contiguous(), "ref_attn": ref_attn,
                  "ref_out": ref_out, "nat_attn": nat_attn, "nat_out": nat_out, "m": length}
        self.attention[key] = result
        return result

    def test_x6_attention(self):
        layers = [layer for layer in self.layers if self.kind(layer) == "full_attention"]
        if not layers:
            oracle.report_row("X6", "attention", phase="isolate", criterion="tolerance",
                              status="no full_attention layer in the D6 layer set")
        for layer in layers:
            for pid in PROMPTS:
                r = self._attention(layer, pid)
                forward_out = self.tap(pid, f"model.layers.{layer}.self_attn.o_proj")
                case = f"L{layer}/attn/{pid}/M{r['m']}"
                fields = dict(layer=layer, prompt=pid, m=r["m"], control="QwenSparseAttnBackend.forward_extend "
                              "(no prefix) + fused_sigmoid_mul + F.linear(o_proj)")
                tolerance_row("X6", f"{case}/attn", r["nat_attn"], r["sg_attn"], r["ref_attn"], oracle.FLOOR_OP,
                              **fields)
                tolerance_row("X6", f"{case}/out", r["nat_out"], r["sg_out"], r["ref_out"], oracle.FLOOR_OP,
                              native_vs_forward_bit_equal=bit_equal(r["nat_out"], forward_out), **fields)

    # -- X2 ---------------------------------------------------------------------------------------------------

    def _x2_cases(self, layer: int, pid: str) -> list[dict]:
        """(plan, part, input, native weights, fused: the `dispatched_fused` result SGLang's GEMM runs for the row, or
        None when it dispatches the row's own weight, input source) of one layer. Each fused weight is read once."""
        p = oracle.layer_prefix(layer)
        cases = []

        def add(plan, part, x, weights, source, fused=None):
            cases.append(dict(plan=plan, part=part, x=x.contiguous(), weights=weights, source=source, fused=fused))

        block_in = self.tap(pid, f"native.layers.{layer}.block.in")
        mlp_in = self.tap(pid, f"native.layers.{layer}.mlp.in")
        if self.kind(layer) == "full_attention":
            w = k6._attn_weights(layer)
            add("attn_qkv", "", block_in, [w["w_qkv"]], "native block.in")
            add("attn_o", "", self._attention(layer, pid)["sg_attn"], [w["w_o"]], "SGLang gated attention (X6)")
        else:
            g = k5._weights(layer)
            gdn = dispatched_fused(layer, "gdn_qkv")
            add("gdn_qkv", "", block_in, [g.qkv], "native block.in", gdn)
            add("gdn_z", "", block_in, [g.z], "native block.in", gdn)
            add("gdn_ba", "b", block_in, [g.b], "native block.in", gdn)
            add("gdn_ba", "a", block_in, [g.a], "native block.in", gdn)
            add("gdn_out", "", self.tap(pid, f"model.layers.{layer}.linear_attn.norm"), [g.out],
                "native linear_attn.norm")
        shared = dispatched_fused(layer, "moe_shared_gate_up")
        shared_w = {part: oracle.real_tensor(f"{p}mlp.shared_expert.{part}_proj.weight").contiguous()
                    for part in ("gate", "up", "down")}
        add("moe_router", "", mlp_in, [_moe_routing(layer)["router"]], "native mlp.in")
        add("moe_shared_gate_up", "gate", mlp_in, [shared_w["gate"]], "native mlp.in", shared)
        add("moe_shared_gate_up", "up", mlp_in, [shared_w["up"]], "native mlp.in", shared)
        act = silu_and_mul(self.tap(pid, f"model.layers.{layer}.mlp.shared_expert.gate_up_proj").contiguous())
        add("moe_shared_down", "", act, [shared_w["down"]], "silu_and_mul(native shared_expert.gate_up_proj)")
        for site in SITES:
            prefix = f"{p}{site}_hyper_connection."
            w_down = oracle.real_tensor(prefix + "input_mix_weight_down.weight").contiguous()
            w_up = oracle.real_tensor(prefix + "input_mix_weight_up.weight").contiguous()
            normed = self.tap(pid, f"model.layers.{layer}.{site}_hyper_connection.hc_norm")
            add("hc_down", site, normed, [w_down], f"native {site}_hyper_connection.hc_norm")
            up_in = F.silu(F.linear(normed, w_down) / self.hc).to(BF16)
            add("hc_up", site, up_in, [w_up], f"silu(F.linear(native {site} hc_norm, down) / hc)")
        return cases

    def _global_weights(self) -> dict[str, torch.Tensor]:
        ple_ids = self.tc.get("ple_layer_ids")
        if not isinstance(ple_ids, list) or len(ple_ids) != 1:
            raise oracle.OracleError(f"text_config ple_layer_ids {ple_ids!r} is not one 1-based layer id")
        p = f"{oracle.layer_prefix(int(ple_ids[0]) - 1)}ple."
        return {"ple_key": oracle.real_tensor(p + "key_proj.weight").contiguous(),
                "ple_value": oracle.real_tensor(p + "value_proj.weight").contiguous(),
                "lm_head": oracle.real_tensor("lm_head.weight").contiguous()}

    def _x2_global_cases(self, pid: str, weights: dict[str, torch.Tensor]) -> list[dict]:
        stand_in = self.tap(pid, f"native.layers.{self.layers[0]}.block.in")
        source = f"stand-in: native layers.{self.layers[0]}.block.in"
        return [
            dict(plan="ple_key", part="", x=stand_in, weights=[weights["ple_key"]], source=source, fused=None),
            dict(plan="ple_value", part="", x=stand_in, weights=[weights["ple_value"]], source=source, fused=None),
            dict(plan="lm_head", part="", x=self.tap(pid, "model"), weights=[weights["lm_head"]],
                 source="native model (final mixer output)", fused=None),
        ]

    def _x2_case(self, failed: list[str], layer, pid: str, case: dict) -> None:
        name = case["plan"] + (f".{case['part']}" if case["part"] else "")
        weights = case["weights"]
        for x in (case["x"], case["x"][-1:].contiguous()):
            m, k = x.shape

            def run_native():
                return torch.cat([_native_gemm(x, w) for w in weights], dim=-1)

            if case["fused"] is not None:
                fused, a, b = _dispatched_view(case["fused"], case["plan"], name, m)

                def run_dispatched():
                    return _linear(x, [fused], True)[:, a:b].contiguous()

                dispatched_n = fused.shape[0]
            else:
                def run_dispatched():
                    return _linear(x, weights, True)

                dispatched_n = sum(w.shape[0] for w in weights)
            native, native_again = run_native(), run_native()
            dispatched, dispatched_again = run_dispatched(), run_dispatched()
            true = _linear(x, weights, True)
            true_again = _linear(x, weights, True)
            false = _linear(x, weights, False)
            row = f"{'' if layer is None else f'L{layer}/'}{name}/{pid}/M{m}"
            bit_level = x2_bit_level(case["plan"], m)
            native_vs_dispatched = bit_equal(native, dispatched)
            repeat = bit_equal(dispatched_again, dispatched)
            passed = native_vs_dispatched == 1.0 if bit_level else None
            native_kernels = _kernel_names(run_native)
            dispatched_kernels = _kernel_names(run_dispatched)
            oracle.report_row("X2", row, phase="isolate", criterion="bit" if bit_level else "report",
                              plan=case["plan"], part=case["part"], layer=-1 if layer is None else layer, prompt=pid,
                              m=m, n=native.shape[1], k=k, dispatched_n=dispatched_n, input=case["source"],
                              passed=passed, sglang_repeat_bit_equal=repeat,
                              lever_pass=lever_pass(native_vs_dispatched, repeat),
                              **_pair("native_vs_dispatched", native, dispatched),
                              **_pair("native_vs_true", native, true), **_pair("native_vs_false", native, false),
                              **_pair("true_vs_false", true, false),
                              torch_repeat_bit_equal=bit_equal(true_again, true),
                              native_repeat_bit_equal=bit_equal(native_again, native),
                              torch_blas=str(torch.backends.cuda.preferred_blas_library()),
                              native_kernels=native_kernels, dispatched_kernels=dispatched_kernels,
                              same_kernels=native_kernels == dispatched_kernels)
            if passed is False:
                failed.append(row)

    def test_x2_gemm_shapes(self):
        failed: list[str] = []
        weights = self._global_weights()
        for pid in PROMPTS:
            for layer in self.layers:
                for case in self._x2_cases(layer, pid):
                    self._x2_case(failed, layer, pid, case)
            for case in self._x2_global_cases(pid, weights):
                self._x2_case(failed, None, pid, case)
            torch.cuda.empty_cache()
        _fail_named(self, "X2", failed)

    # -- X3 ---------------------------------------------------------------------------------------------------

    def test_x3_router(self):
        for layer in self.layers:
            w = _moe_routing(layer)
            for pid in PROMPTS:
                x = self.tap(pid, f"native.layers.{layer}.mlp.in")
                logits = self.tap(pid, f"model.layers.{layer}.mlp.gate").contiguous()
                m, num_experts = logits.shape
                case = f"L{layer}/router/{pid}/M{m}"
                sg_logits = F.linear(x, w["router"])
                tolerance_row("X3", f"{case}/logits", logits, sg_logits, F.linear(x.double(), w["router"].double()),
                              oracle.FLOOR_OP, layer=layer, prompt=pid, m=m,
                              control="F.linear(mlp.in, mlp.gate.weight)",
                              native_gemm_vs_forward_bit_equal=bit_equal(_native_gemm(x, w["router"]), logits))
                ids, weights = _native_router(logits, self.top_k)
                ref_weights, ref_ids = k7._sglang_route(x, logits, self.top_k)
                ref_ids = ref_ids.to(torch.int32)
                ids_sorted, order = ids.sort(dim=-1)
                ref_sorted, ref_order = ref_ids.sort(dim=-1)
                set_equal = (ids_sorted == ref_sorted).all(dim=-1)
                ulp = _ulp(weights.gather(1, order), ref_weights.to(torch.float32).gather(1, ref_order))
                ulp_rows = ulp[set_equal]
                max_ulp = int(ulp_rows.max().item()) if ulp_rows.numel() else -1
                rows_set_equal = int(set_equal.sum().item())
                oracle.report_row("X3", f"{case}/route", phase="isolate", criterion="ids set-equal, weights <= 1 ulp",
                                  layer=layer, prompt=pid, m=m, control="fused_topk (moe_fused_gate) on the native logits",
                                  rows_set_equal=rows_set_equal, rows_order_equal=int((ids == ref_ids).all(dim=-1)
                                                                                      .sum().item()),
                                  weights_max_ulp=max_ulp,
                                  weights_over_1ulp=int((ulp_rows > 1).sum().item()),
                                  within=rows_set_equal == m and max_ulp <= 1)

    # -- X4 ---------------------------------------------------------------------------------------------------

    def test_x4_routed_experts(self):
        failed: list[str] = []
        self.addCleanup(release_moe)
        for layer in self.layers:
            w, s, _ = _moe_layer(layer)
            w_routed = dict(w, shared_gate=torch.zeros_like(w["shared_gate"]),
                            shared_up=torch.zeros_like(w["shared_up"]),
                            shared_down=torch.zeros_like(w["shared_down"]))
            for pid in PROMPTS:
                x = self.tap(pid, f"native.layers.{layer}.mlp.in").contiguous()
                m = x.shape[0]
                native = _native_moe_block(x, w_routed, s, self.top_k)
                native_ids, native_weights = _native_router(_native_gemm(x, w["router"]), self.top_k)
                forward = self.tap(pid, f"model.layers.{layer}.mlp.experts")
                logged, status = bench_tactic(m)
                for label, tactic in (("tactic_-1", -1), ("bench_tactic", logged)):
                    case = f"L{layer}/experts/{pid}/M{m}/{label}"
                    fields = dict(phase="isolate", criterion="bit", layer=layer, prompt=pid, m=m,
                                  bucket=pt.moe_bucket(m), native_vs_forward_bit_equal=bit_equal(native, forward))
                    if tactic is None:
                        oracle.report_row("X4", case, bench_tactic=status, passed=None, **fields)
                        continue
                    sg, topk_weights, topk_ids = _sglang_routed(layer, x, tactic)
                    sg_again = _sglang_routed(layer, x, tactic)[0]
                    routing = _routing(native_ids, native_weights, topk_ids, topk_weights)
                    routing_same = routing["routing_ids_equal"] and routing["routing_weights_bit_equal"] == 1.0
                    hp = k7._high_precision_block(x, w_routed, s, topk_weights, topk_ids)
                    c = oracle.rel_l2(sg, hp)
                    native_vs_sg = bit_equal(native, sg)
                    repeat = bit_equal(sg_again, sg)
                    bounded = bit_equal(routed_reference(layer, x, tactic), sg)
                    passed = native_vs_sg == 1.0
                    gemm1, gemm2 = _tactic_pair(tactic)
                    oracle.report_row("X4", case, bench_tactic=status if label == "bench_tactic" else "n/a",
                                      gemm1_tactic=gemm1, gemm2_tactic=gemm2, passed=passed,
                                      sglang_repeat_bit_equal=repeat, lever_pass=lever_pass(native_vs_sg, repeat),
                                      routed_reference_vs_full_load_bit_equal=bounded,
                                      routing="same" if routing_same else "routing differs (see X2 moe_router, X3)",
                                      **routing, **_pair("native_vs_cutlass", native, sg), c=c,
                                      limit=oracle.calibrated_limit(c, oracle.FLOOR_BLOCK), **fields)
                    if not passed:
                        failed.append(case if routing_same else f"{case} (routing differs)")
                    if bounded != 1.0:
                        failed.append(f"{case} (routed_reference's routed-only load differs from the full load)")
        _fail_named(self, "X4", failed)

    # -- X5 ---------------------------------------------------------------------------------------------------

    def test_x5_shared_expert(self):
        self.addCleanup(release_moe)
        for layer in self.layers:
            w, s, _ = _moe_layer(layer)
            s_shared = dict(s, g2_alphas=torch.zeros_like(s["g2_alphas"]))
            gate_up_w = torch.cat([w["shared_gate"], w["shared_up"]], dim=0)
            for pid in PROMPTS:
                x = self.tap(pid, f"native.layers.{layer}.mlp.in").contiguous()
                m = x.shape[0]
                case = f"L{layer}/shared/{pid}/M{m}"
                native = _native_moe_block(x, w, s_shared, self.top_k)
                sg = torch.zeros_like(x)
                k7._sglang_shared_add(x, w, sg)
                xd = x.double()
                act64 = F.silu(xd @ w["shared_gate"].double().T) * (xd @ w["shared_up"].double().T)
                down64 = act64 @ w["shared_down"].double().T
                gated64 = torch.sigmoid(xd @ w["shared_expert_gate"].double()).unsqueeze(-1) * down64
                fields = dict(layer=layer, prompt=pid, m=m)
                tolerance_row("X5", f"{case}/gated_add", native, sg, gated64, oracle.FLOOR_OP,
                              control="Qwen2MoeMLP ops + fused_gate_sigmoid_mul_add onto zeros; native g2_alphas 0",
                              **fields)
                sg_down = F.linear(silu_and_mul(F.linear(x, gate_up_w)), w["shared_down"])
                tolerance_row("X5", f"{case}/down_proj", self.tap(pid, f"model.layers.{layer}.mlp.shared_expert"
                                                                      ".down_proj"),
                              sg_down, down64, oracle.FLOOR_OP,
                              control="F.linear(gate_up) + silu_and_mul + F.linear(down) (Qwen2MoeMLP)", **fields)

    # -- X7 ---------------------------------------------------------------------------------------------------

    def test_x7_gdn_prefill(self):
        layers = [layer for layer in self.layers if self.kind(layer) == "linear_attention"]
        if not layers:
            oracle.report_row("X7", "gdn", phase="isolate", criterion="tolerance",
                              status="no linear_attention layer in the D6 layer set")
        for layer in layers:
            w = k5._weights(layer)
            for pid in PROMPTS:
                hidden = self.tap(pid, f"native.layers.{layer}.block.in").contiguous()
                m = hidden.shape[0]
                case = f"L{layer}/gdn/{pid}/M{m}"
                sg_conv, sg_ssm = k5._sglang_conv_state(), k5._sglang_ssm_state()
                mixed_qkv, z, b, a = _sglang_proj_fused(layer, w, hidden)
                sg_o = k5._sglang_extend_core(w, mixed_qkv, b, a, sg_conv, sg_ssm, False)
                sg_out = k5._sglang_mixer_out(w, sg_o, z)
                ref_out, ref_conv, ref_ssm = k5._ref_layer64(w, hidden, sg_conv[0].new_zeros(k5.CONV_DIM,
                                                                                             k5.CONV_K - 1),
                                                             sg_ssm[0].new_zeros(k5.HV, k5.DV, k5.DK))
                nat_conv = torch.zeros(k5.CONV_DIM, k5.CONV_K - 1, dtype=BF16, device=oracle.DEVICE)
                nat_ssm = torch.zeros(k5.HV, k5.DV, k5.DK, dtype=BF16, device=oracle.DEVICE)
                nat_out = _native_gdn_prefill(w, hidden, nat_conv, nat_ssm)
                fields = dict(layer=layer, prompt=pid, m=m, in_proj="fused n 16,480" if m <= FUSED_IN_PROJ_MAX_M
                              else "in_proj_qkvz + in_proj_ba",
                              control="fused in_proj F.linear + FLA chunk extend (chunk_gated_delta_rule, one "
                              "partial chunk of 64) + gated norm + F.linear(out_proj)")
                forward = self.tap(pid, f"model.layers.{layer}.linear_attn.out_proj")
                tolerance_row("X7", f"{case}/out", nat_out, sg_out, ref_out, oracle.FLOOR_BLOCK,
                              native_vs_forward_bit_equal=bit_equal(nat_out, forward), **fields)
                tolerance_row("X7", f"{case}/ssm_state", nat_ssm, sg_ssm[0], ref_ssm, oracle.FLOOR_BLOCK, **fields)
                tolerance_row("X7", f"{case}/conv_state", nat_conv, sg_conv[0], ref_conv, oracle.FLOOR_BLOCK,
                              **fields)


if __name__ == "__main__":
    unittest.main()
