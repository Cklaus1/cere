"""K7T: the tensor-core NVFP4 MoE grouped GEMM (tasks/m215-spec.md D4-D6), as kernel-test entry points only.

Suite `k7t` (GPU1 kernel lab, D1): `python3 scripts/live/native_dev.py --gpu 1 test --suite k7t --reports-dir <abs>`,
which runs `python3 -m unittest discover -s /src/executor/tests/gpu -p 'test_k7t_*.py' -v` in a `cerelab-g1-*`
container on GPU1 only. Nothing here is a G1 result; M2.3 judges speed on GPU0 under its own protocol.

Inputs (F8, the K7 suite's definitions, `test_kernels_k7_moe.py`): layer 0 the MoE input of the parity prompts'
tokens (input (i)), layers 23 and 47 seeded synthetic BF16 RMS-matched to it (input (ii)); the first M rows, M in
{1, 2, 4, 16, 64, 2048}; real expert weights and the flashinfer_cutlass scale set (`_reference_scales`). Every
weight and activation block-scale byte is asserted <= 0x7E (the domain D2 characterizes; 0x7F/0xFF are E4M3 NaN).

Rows (row key K7T):
  (i) asserted, `test_gemm_raw_accumulators_equal_chain_model`: the raw FP32 accumulators of
      `cere_ktest_k7t_gemm` (GEMM1 gate, GEMM1 up, GEMM2) are bit-equal to `omma_ref.chain_t`, the D3 model of the
      OMMA chain in ascending K. The sorted rows are the native routing of x (router logits by
      `cere_ktest_gemm_bf16`, the router plan's shape, then `cere_ktest_moe_router`), stable by expert; A of GEMM1
      is x quantized by `cere_ktest_fp4_quantize` (a1 scale), A of GEMM2 the SwiGLU of GEMM1's accumulators
      (alpha, BF16, SiLU(gate) * up, BF16) quantized with the a2 scale. Every element at M <= 16; at M in
      {64, 2048} a seeded sample of min(2^20, all) elements per GEMM (`k7t_compare.sample_indices`), each with
      its full chain;
  (ii) asserted, `test_block_port_bit_equal`: `cere_ktest_k7t_block` with gemm_mode 0 is bit-equal to
      `cere_ktest_moe_block` on the block output; reported: gemm_mode 1 against `cere_ktest_moe_block`, the
      bit-equal fraction and max BF16 ulp;
  (iii) reported, `test_block_against_cutlass_fused_moe`: gemm_mode 1 and K7 (`cere_ktest_moe_block`), each with
      the shared expert zeroed (final routed output only), against `cutlass_fused_moe` called as F8 calls it
      (`test_kernels_k7_moe._sglang_routed`: fused_topk routing, untuned tactics at tune_max_num_tokens =
      next_power_of_2(M), recorded per M);
  (iv) reported, same test: `cutlass_fused_moe` given the native routing (the token_selected_experts and
      token_final_scales `moe_route` gives the block on the router plan's logits) against gemm_mode 1 and K7.
      Headline field `k7t_exceeds_k7`: whether k7t's bit-equal fraction exceeds K7's;
  stage, reported, `test_block_stages`: where gemm_mode 0 and 1 part, from `cere_ktest_k7t_block_stages` of both
      modes on the same input, every layer and M: (a) h, the GEMM1 + SwiGLU output (BF16, before the a2
      quantization): bit-equal fraction and max BF16 ulp; (b) the a2 quantization's codes and scales: bit-equal
      byte fractions; (c) y, the GEMM2 output before combine (alpha applied, BF16 as K7 stores it): bit-equal
      fraction and max BF16 ulp. Asserted: both modes sort the pairs identically (perm), and the comparison sees a
      single flipped input bit (gemm_mode 0 on x and on x with the sign of one element of token 0 flipped: the
      smallest-|x| element with a nonzero a1 code whose flip leaves the native top-k ids unchanged; the ids and perm
      asserted bit-equal, then h, the codes and y each differ), so an all-equal stage row cannot come from comparing
      a buffer with itself, nor a differing one from a routing change [REVISED: M215R-06];
  D6 reported, `test_speed`: microseconds per layer call at every M and layer for k7t (gemm_mode 1), K7 and
      `cutlass_fused_moe` (routed experts only, native routing precomputed): the median of 20 CUDA-event spans
      after 5 warm-up calls (the whole entry point: the ktest's K2 plan setup and workspace allocation included),
      and the mean device time of the launched kernels over 20 more calls (torch profiler).
Comparisons (bit-equal fraction, ulp distance) are `k7t_compare`'s on the FP32 values of the BF16 outputs; a BF16
ulp is 2^16 FP32 ulps of such values, exactly.

Native entry points (C prototypes; every call returns 0 on success, nonzero `cere_status` otherwise). Also called:
`cere_ktest_moe_block` and `cere_ktest_moe_router` (test_kernels_k7_moe.py), `cere_ktest_fp4_quantize`
(test_layout_k0_nvfp4.py) and `cere_ktest_gemm_bf16` (test_kernels_k2_gemm.py), with their stated prototypes.

  int cere_ktest_k7t_gemm(const void *a_codes,        /* uint8 [rows, k/2] E2M1 codes, rows sorted by expert,
                                                         element 2j in the low nibble */
                          const void *a_scales,       /* FP8 E4M3 [rows, k/16], linear layout */
                          const void *expert_counts,  /* int32 [num_experts]: rows of each expert, in order */
                          const void *w0,             /* uint8 [num_experts, n, k/2]: GEMM1 gate / GEMM2 down */
                          const void *s0,             /* FP8 E4M3 [num_experts, n, k/16] */
                          const void *w1,             /* uint8 [num_experts, n, k/2]: GEMM1 up; null: GEMM2 */
                          const void *s1,             /* FP8 E4M3 [num_experts, n, k/16]; null with w1 */
                          void *out0,                 /* float32 [rows, n]: raw accumulators of w0 */
                          void *out1,                 /* float32 [rows, n]: raw accumulators of w1; null with w1 */
                          int64_t rows, int64_t n, int64_t k, int64_t num_experts,
                          cudaStream_t stream);
      One OMMA m16n8k64 block-scaled (kind::mxf4nvf4, scale_vec::4X, ue4m3) per 64-wide K step, chained in
      ascending K (no split-K), before any alpha or epilogue. k a multiple of 128, n of 64.

  int cere_ktest_k7t_block(const void *x, const void *router_weight, const void *gate_weight,
                           const void *gate_scale, const void *up_weight, const void *up_scale,
                           const void *down_weight, const void *down_scale, const void *g1_alphas,
                           const void *g2_alphas, float a1_global_scale, float a2_global_scale,
                           const void *shared_gate_weight, const void *shared_up_weight,
                           const void *shared_down_weight, const void *shared_expert_gate, void *out,
                           int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter,
                           int64_t shared_inter,
                           int64_t gemm_mode,          /* 0: K7's dot64 grouped GEMM (ported), 1: OMMA */
                           cudaStream_t stream);
      `cere_ktest_moe_block`'s parameters and plans plus gemm_mode; out bfloat16 [m, hidden].

  int cere_ktest_k7t_block_stages(const void *x, const void *router_weight, const void *gate_weight,
                                  const void *gate_scale, const void *up_weight, const void *up_scale,
                                  const void *down_weight, const void *down_scale, const void *g1_alphas,
                                  const void *g2_alphas, float a1_global_scale, float a2_global_scale,
                                  const void *shared_gate_weight, const void *shared_up_weight,
                                  const void *shared_down_weight, const void *shared_expert_gate, void *out,
                                  int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter,
                                  int64_t shared_inter, int64_t gemm_mode,
                                  void *perm,      /* int32 [m*top_k]: sorted row -> pair t*top_k + k */
                                  void *h,         /* bf16 [m*top_k, inter]: GEMM1 + SwiGLU, pre-a2-quantize */
                                  void *h_codes,   /* uint8 [m*top_k, inter/2]: the a2 quantization's codes */
                                  void *h_scales,  /* FP8 E4M3 [m*top_k, inter/16]: and its scales */
                                  void *y,         /* bf16 [m*top_k, hidden]: GEMM2 (alpha, BF16), pre-combine */
                                  cudaStream_t stream);
      `cere_ktest_k7t_block` plus the block's intermediates in sorted-row order, copied out after the block.
"""
from __future__ import annotations

import importlib
import statistics
import unittest

import torch
import torch.nn.functional as F
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType

import k7t_compare
import omma_ref
import oracle

k7 = importlib.import_module("test_kernels_k7_moe")  # F8's inputs, weights, scale set and cutlass call

ROW = "K7T"
LAYERS = (0, 23, 47)
M_CASES = (1, 2, 4, 16, 64, 2048)
FULL_CHECK_MAX_M = 16  # D5 (i): every element up to here, a sample above
SAMPLE = 1 << 20
SAMPLE_SEED = 15_000
MODEL_BATCH = 1 << 16  # elements per omma_ref.chain_t call
SCALE_MAX = 0x7E
BF16_ULP = 1 << 16  # FP32 ulps per BF16 ulp of a BF16 value
WARMUP = 5
TIMED = 20


# -- native calls -------------------------------------------------------------------------------------------------


def _k7t_gemm(a_codes: torch.Tensor, a_scales: torch.Tensor, counts: torch.Tensor, w0: torch.Tensor,
              s0: torch.Tensor, w1: torch.Tensor | None, s1: torch.Tensor | None):
    """Raw FP32 accumulators (out0, out1 or None) of the OMMA grouped GEMM over sorted rows."""
    rows = a_codes.shape[0]
    num_experts, n, half = w0.shape
    out0 = oracle.empty((rows, n), torch.float32)
    out1 = None if w1 is None else oracle.empty((rows, n), torch.float32)
    oracle.call("cere_ktest_k7t_gemm", a_codes, a_scales, counts, w0, s0, w1, s1, out0, out1,
                rows, n, 2 * half, num_experts)
    return out0, out1


def _layer_inputs(w: dict[str, torch.Tensor], s: dict[str, torch.Tensor], top_k: int) -> dict:
    """One layer's block arguments but x, out and m (host scalars resolved once, so a timed call never syncs)."""
    num_experts, inter, _ = w["gate_weight"].shape
    return dict(w, g1_alphas=s["g1_alphas"], g2_alphas=s["g2_alphas"], a1=float(s["w13_input_scale_quant"].item()),
                a2=float(s["w2_input_scale_quant"].item()), hidden=w["router"].shape[1], num_experts=num_experts,
                top_k=top_k, inter=inter, shared_inter=w["shared_gate"].shape[0])


def _k7_block(x: torch.Tensor, L: dict, out: torch.Tensor) -> None:
    oracle.call("cere_ktest_moe_block", x, L["router"], L["gate_weight"], L["gate_weight_scale"], L["up_weight"],
                L["up_weight_scale"], L["down_weight"], L["down_weight_scale"], L["g1_alphas"], L["g2_alphas"],
                L["a1"], L["a2"], L["shared_gate"], L["shared_up"], L["shared_down"], L["shared_expert_gate"], out,
                x.shape[0], L["hidden"], L["num_experts"], L["top_k"], L["inter"], L["shared_inter"])


def _k7t_block(x: torch.Tensor, L: dict, out: torch.Tensor, gemm_mode: int) -> None:
    oracle.call("cere_ktest_k7t_block", x, L["router"], L["gate_weight"], L["gate_weight_scale"], L["up_weight"],
                L["up_weight_scale"], L["down_weight"], L["down_weight_scale"], L["g1_alphas"], L["g2_alphas"],
                L["a1"], L["a2"], L["shared_gate"], L["shared_up"], L["shared_down"], L["shared_expert_gate"], out,
                x.shape[0], L["hidden"], L["num_experts"], L["top_k"], L["inter"], L["shared_inter"], gemm_mode)


def _block(x: torch.Tensor, L: dict, gemm_mode: int | None) -> torch.Tensor:
    """The block output: K7 (`gemm_mode` None) or k7t in `gemm_mode`."""
    out = oracle.empty(tuple(x.shape), torch.bfloat16)
    if gemm_mode is None:
        _k7_block(x, L, out)
    else:
        _k7t_block(x, L, out, gemm_mode)
    return out


def _stages(x: torch.Tensor, L: dict, gemm_mode: int) -> dict[str, torch.Tensor]:
    """The block's intermediates in `gemm_mode` (cere_ktest_k7t_block_stages), sorted-row order."""
    pairs = x.shape[0] * L["top_k"]
    out = oracle.empty(tuple(x.shape), torch.bfloat16)
    st = {"perm": oracle.empty((pairs,), torch.int32), "h": oracle.empty((pairs, L["inter"]), torch.bfloat16),
          "h_codes": oracle.empty((pairs, L["inter"] // 2), torch.uint8),
          "h_scales": oracle.empty((pairs, L["inter"] // 16), torch.uint8),
          "y": oracle.empty((pairs, L["hidden"]), torch.bfloat16)}
    oracle.call("cere_ktest_k7t_block_stages", x, L["router"], L["gate_weight"], L["gate_weight_scale"],
                L["up_weight"], L["up_weight_scale"], L["down_weight"], L["down_weight_scale"], L["g1_alphas"],
                L["g2_alphas"], L["a1"], L["a2"], L["shared_gate"], L["shared_up"], L["shared_down"],
                L["shared_expert_gate"], out, x.shape[0], L["hidden"], L["num_experts"], L["top_k"], L["inter"],
                L["shared_inter"], gemm_mode, st["perm"], st["h"], st["h_codes"], st["h_scales"], st["y"])
    return st


def _native_route(x: torch.Tensor, router: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The block's routing of x: logits by the router plan's shape (unpinned K2 plan of [E, H], max_m = m), then
    `moe_route`. int32 ids, float32 weights, [m, top_k]."""
    m, hidden = x.shape
    num_experts = router.shape[0]
    logits = oracle.empty((m, num_experts), torch.bfloat16)
    oracle.call("cere_ktest_gemm_bf16", x, router, m, num_experts, hidden, logits)
    ids = oracle.empty((m, top_k), torch.int32)
    weights = oracle.empty((m, top_k), torch.float32)
    oracle.call("cere_ktest_moe_router", logits, ids, weights, m, num_experts, top_k)
    return ids, weights


def _quantize(x: torch.Tensor, global_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """K0's activation quantizer (the one K7 calls): codes uint8 [rows, k/2], scale bytes uint8 [rows, k/16]."""
    rows, k = x.shape
    codes = oracle.empty((rows, k // 2), torch.uint8)
    scales = oracle.empty((rows, k // 16), torch.uint8)
    oracle.call("cere_ktest_fp4_quantize", x, global_scale, codes, scales, rows, k)
    return codes, scales


def _cutlass(x: torch.Tensor, ops: dict, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
    """`cutlass_fused_moe` exactly as F8 calls it (`test_kernels_k7_moe._sglang_routed`) on the given routing."""
    m = x.shape[0]
    output = torch.empty(m, x.shape[1], dtype=torch.bfloat16, device=x.device)
    return cutlass_fused_moe(
        output=output,
        input=x,
        token_selected_experts=topk_ids.to(torch.int),
        token_final_scales=topk_weights,
        fc1_expert_weights=ops["w13_weight"],
        fc2_expert_weights=ops["w2_weight"],
        output_dtype=torch.bfloat16,
        input_sf=None,
        quant_scales=ops["quant_scales"],
        ep_size=1,
        ep_rank=0,
        tp_size=1,
        tp_rank=0,
        tune_max_num_tokens=k7.next_power_of_2(m),
        activation_type=ActivationType(k7.get_activation_type("silu", is_gated=True)),
        enable_alltoall=False,
        use_fused_finalize=k7.envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
    )[0]


def _tactic(m: int) -> str:
    return f"untuned: flashinfer default tactics, tune_max_num_tokens={k7.next_power_of_2(m)}"


# -- the D3 chain model -------------------------------------------------------------------------------------------


def _unpack(codes: torch.Tensor) -> torch.Tensor:
    """uint8 [B, k/2] packed E2M1 (element 2j in the low nibble) -> uint8 [B, k], one code per byte."""
    return torch.stack([codes & 0x0F, codes >> 4], dim=-1).reshape(codes.shape[0], -1)


def _chain_model(a_codes: torch.Tensor, a_scales: torch.Tensor, row_expert: torch.Tensor, w: torch.Tensor,
                 s: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """omma_ref.chain_t of the GEMM elements at flat indices `idx` (sorted row * n + column): float32 [len(idx)]."""
    num_experts, n, half = w.shape
    w_rows = w.view(torch.uint8).reshape(num_experts * n, half)
    s_rows = s.view(torch.uint8).reshape(num_experts * n, -1)
    parts = []
    for start in range(0, idx.numel(), MODEL_BATCH):
        flat = idx[start:start + MODEL_BATCH]
        rows = flat // n
        w_row = row_expert[rows] * n + flat % n
        parts.append(omma_ref.chain_t(_unpack(a_codes[rows]), _unpack(w_rows[w_row]), a_scales[rows],
                                      s_rows[w_row]))
    return torch.cat(parts)


def _sorted_rows(ids: torch.Tensor, num_experts: int):
    """The (token, slot) pairs sorted stably by expert: token of each sorted row, its expert (int64) and the int32
    row count of each expert."""
    flat = ids.reshape(-1).long()
    order = torch.sort(flat, stable=True).indices
    counts = torch.bincount(flat, minlength=num_experts).to(torch.int32).contiguous()
    return order // ids.shape[1], flat[order], counts


def _swiglu(gate: torch.Tensor, up: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """GEMM2's activation from GEMM1's raw accumulators: alpha, BF16, SiLU(gate) * up in FP32, BF16."""
    g = (gate * alpha).to(torch.bfloat16).float()
    u = (up * alpha).to(torch.bfloat16).float()
    return (F.silu(g) * u).to(torch.bfloat16).contiguous()


# -- comparisons --------------------------------------------------------------------------------------------------


def _compare(native: torch.Tensor, reference: torch.Tensor) -> tuple[float, int]:
    """(bit-equal fraction, max BF16 ulp) of two BF16 tensors, by k7t_compare on their exact FP32 values."""
    xs = native.float().reshape(-1).tolist()
    ys = reference.float().reshape(-1).tolist()
    return k7t_compare.bit_equal_fraction(xs, ys), k7t_compare.max_ulp(xs, ys) // BF16_ULP


def _stage_compare(a: torch.Tensor, b: torch.Tensor) -> tuple[float, int]:
    """(bit-equal fraction, max BF16 ulp) of two BF16 tensors of any size: the differing positions by their bits,
    the ulp distance by k7t_compare over those positions only (equal positions are 0 ulps apart)."""
    differ = a.view(torch.int16) != b.view(torch.int16)
    count = int(differ.sum().item())
    ulp = 0 if count == 0 else k7t_compare.max_ulp(a[differ].float().tolist(), b[differ].float().tolist())
    return 1.0 - count / a.numel(), ulp // BF16_ULP


def _byte_fraction(a: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0 - int((a.view(torch.uint8) != b.view(torch.uint8)).sum().item()) / a.numel()


def _stage_fields(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict:
    """The stage row's fields, (a) h, (b) the a2 codes and scales, (c) y."""
    h_equal, h_ulp = _stage_compare(a["h"], b["h"])
    y_equal, y_ulp = _stage_compare(a["y"], b["y"])
    return {"h_bit_equal": h_equal, "h_max_ulp_bf16": h_ulp, "a2_codes_bit_equal": _byte_fraction(a["h_codes"],
            b["h_codes"]), "a2_scales_bit_equal": _byte_fraction(a["h_scales"], b["h_scales"]),
            "y_bit_equal": y_equal, "y_max_ulp_bf16": y_ulp, "h_elements": a["h"].numel(),
            "y_elements": a["y"].numel()}


def _scale_bytes_over(t: torch.Tensor) -> int:
    return int((t.view(torch.uint8) > SCALE_MAX).sum().item())


def _load(layer: int):
    w = k7._load_moe_layer(layer)
    return w, k7._reference_scales(w)


def _label(layer: int, m: int) -> str:
    return f"L{layer} ({k7._input_label(layer)}) M{m}"


def _timed(run) -> dict:
    """D6: median CUDA-event span of TIMED calls after WARMUP, and the mean kernel device time of TIMED more."""
    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize(oracle.DEVICE)
    spans = []
    for _ in range(TIMED):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        spans.append(start.elapsed_time(end) * 1000.0)
    fields = {"event_us_median": statistics.median(spans), "event_us_min": min(spans)}
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(TIMED):
                run()
            torch.cuda.synchronize(oracle.DEVICE)
    except RuntimeError as e:
        fields["kernel_us_mean"] = f"not captured: {type(e).__name__}"
        return fields
    device = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    fields["kernel_us_mean"] = (sum(e.time_range.elapsed_us() for e in device) / TIMED if device
                                else "not captured: no CUDA event")
    fields["kernels_per_call"] = len(device) / TIMED
    return fields


# -- tests --------------------------------------------------------------------------------------------------------


class K7tMoe(unittest.TestCase):
    def _gemm_case(self, layer: int, mi: int, m: int, w: dict[str, torch.Tensor], s: dict[str, torch.Tensor],
                   x: torch.Tensor, top_k: int) -> None:
        num_experts = w["gate_weight"].shape[0]
        label = _label(layer, m)
        ids, _ = _native_route(x, w["router"], top_k)
        tokens, row_expert, counts = _sorted_rows(ids, num_experts)
        x_codes, x_scales = _quantize(x, float(s["w13_input_scale_quant"].item()))
        oracle.check_count(self, ROW, f"gemm {label} x scale bytes over 0x7E", _scale_bytes_over(x_scales),
                           expected=0)
        a_codes, a_scales = x_codes[tokens].contiguous(), x_scales[tokens].contiguous()
        gate, up = _k7t_gemm(a_codes, a_scales, counts, w["gate_weight"], w["gate_weight_scale"],
                             w["up_weight"], w["up_weight_scale"])
        h = _swiglu(gate, up, s["g1_alphas"][row_expert].unsqueeze(-1))
        h_codes, h_scales = _quantize(h, float(s["w2_input_scale_quant"].item()))
        oracle.check_count(self, ROW, f"gemm {label} h scale bytes over 0x7E", _scale_bytes_over(h_scales),
                           expected=0)
        (down, _) = _k7t_gemm(h_codes, h_scales, counts, w["down_weight"], w["down_weight_scale"], None, None)
        gemms = (("gemm1_gate", gate, a_codes, a_scales, "gate"), ("gemm1_up", up, a_codes, a_scales, "up"),
                 ("gemm2_down", down, h_codes, h_scales, "down"))
        for g, (name, raw, codes, scales, proj) in enumerate(gemms):
            case = f"gemm {label} {name}"
            total = raw.numel()
            idx = torch.tensor(k7t_compare.sample_indices(SAMPLE_SEED + 100 * layer + 10 * mi + g, total,
                                                          total if m <= FULL_CHECK_MAX_M else SAMPLE),
                               dtype=torch.int64, device=oracle.DEVICE)
            native = raw.reshape(-1)[idx].contiguous()
            model = _chain_model(codes, scales, row_expert, w[f"{proj}_weight"], w[f"{proj}_weight_scale"], idx)
            differ = native.view(torch.int32) != model.view(torch.int32)
            finite = torch.isfinite(native) & torch.isfinite(model)
            fields = {"nonfinite": int((~finite).sum().item())}
            if bool((differ & finite).any()):
                fields["max_ulp"] = k7t_compare.max_ulp(native[differ & finite].tolist(),
                                                        model[differ & finite].tolist())
            oracle.report_row(ROW, case, phase="sample", elements=idx.numel(), population=total, m=m,
                              rows=int(raw.shape[0]), k=int(codes.shape[1]) * 2, **fields)
            oracle.check_bit_exact(self, ROW, case, native, model)

    def test_gemm_raw_accumulators_equal_chain_model(self) -> None:
        """D5 (i): k7t_gemm's raw accumulators are bit-equal to the D3 chain model at every M (asserted)."""
        top_k = oracle.text_config()["num_experts_per_tok"]
        for layer in LAYERS:
            w, s = _load(layer)
            for proj in ("gate", "up", "down"):
                with self.subTest(case=f"L{layer} {proj} weight scales"):
                    oracle.check_count(self, ROW, f"gemm L{layer} {proj} weight scale bytes over 0x7E",
                                       _scale_bytes_over(w[f"{proj}_weight_scale"]), expected=0)
            x_all = k7._moe_input(layer)
            for mi, m in enumerate(M_CASES):
                with self.subTest(case=f"gemm {_label(layer, m)}"):
                    self._gemm_case(layer, mi, m, w, s, x_all[:m].contiguous(), top_k)
            del w, s
            torch.cuda.empty_cache()

    def test_block_port_bit_equal(self) -> None:
        """D5 (ii): k7t_block gemm_mode 0 is bit-equal to cere_ktest_moe_block (asserted); gemm_mode 1 against it
        is reported."""
        top_k = oracle.text_config()["num_experts_per_tok"]
        for layer in LAYERS:
            w, s = _load(layer)
            L = _layer_inputs(w, s, top_k)
            x_all = k7._moe_input(layer)
            for m in M_CASES:
                label = _label(layer, m)
                with self.subTest(case=f"port {label}"):
                    x = x_all[:m].contiguous()
                    k7_out = _block(x, L, None)
                    fraction, ulp = _compare(_block(x, L, 1), k7_out)
                    oracle.report_row(ROW, f"port {label} gemm_mode 1 vs K7", phase="report", m=m, layer=layer,
                                      elements=k7_out.numel(), bit_equal=fraction, max_ulp_bf16=ulp)
                    oracle.check_bit_exact(self, ROW, f"port {label} gemm_mode 0 vs K7", _block(x, L, 0), k7_out)
            del w, s, L
            torch.cuda.empty_cache()

    def _stage_sanity(self, layer: int, L: dict, x_all: torch.Tensor) -> None:
        """The stage sanity [REVISED: M215R-06]: gemm_mode 0 on x (M16) and on x with one BF16 sign bit flipped,
        the flip kept only if it leaves the routing unchanged. Candidates are token 0's elements whose a1 E2M1 code
        is nonzero (GEMM1 sees the flip; the sign leaves every block amax, so every scale, unchanged), in ascending
        |x| (the smallest change of the router logits); the first whose native top-k ids equal x's is kept. Asserted:
        the ids and the pair permutation bit-equal x's, and h, the a2 codes or scales, and y each differ."""
        m = 16
        label = _label(layer, m)
        x = x_all[:m].contiguous()
        base = _stages(x, L, 0)
        ids, _ = _native_route(x, L["router"], L["top_k"])
        codes, _ = _quantize(x, L["a1"])
        order = x[0].float().abs().argsort()
        candidates = order[(_unpack(codes[:1])[0] & 0x7)[order] != 0].tolist()
        for tried, j in enumerate(candidates, start=1):
            flipped = x.clone()
            flipped[0, j] = -flipped[0, j]
            flipped_ids, _ = _native_route(flipped, L["router"], L["top_k"])
            if torch.equal(flipped_ids, ids):
                break
        other = _stages(flipped, L, 0)
        fields = _stage_fields(base, other)
        seen = [fields["h_bit_equal"] < 1.0, fields["a2_codes_bit_equal"] < 1.0 or
                fields["a2_scales_bit_equal"] < 1.0, fields["y_bit_equal"] < 1.0]
        # Row strings are LAB_REPORT_STRING (native_dev C5): no brackets.
        oracle.report_row(ROW, f"stage sanity {label} sign of x(0, {j}) flipped", phase="report", m=m, layer=layer,
                          candidates=len(candidates), tried=tried, **fields)
        oracle.check_bit_exact(self, ROW, f"stage sanity {label} top-k ids after the flip", flipped_ids, ids)
        oracle.check_bit_exact(self, ROW, f"stage sanity {label} perm after the flip", other["perm"], base["perm"])
        oracle.check_count(self, ROW, f"stage sanity {label} stages that see the flipped bit", sum(seen), expected=3)

    def test_block_stages(self) -> None:
        """Stage row, reported: gemm_mode 0 against gemm_mode 1 at h, the a2 codes and scales, and y. Asserted: both
        modes sort the pairs identically, and a single flipped input bit that leaves the routing unchanged is seen at
        every stage (so the taps follow the GEMMs, not a re-sort; `_stage_sanity`). [REVISED: M215R-06]"""
        top_k = oracle.text_config()["num_experts_per_tok"]
        for layer in LAYERS:
            w, s = _load(layer)
            L = _layer_inputs(w, s, top_k)
            x_all = k7._moe_input(layer)
            if layer == LAYERS[0]:
                with self.subTest(case="stage sanity"):
                    self._stage_sanity(layer, L, x_all)
            for m in M_CASES:
                label = _label(layer, m)
                with self.subTest(case=f"stage {label}"):
                    x = x_all[:m].contiguous()
                    dot64, omma = _stages(x, L, 0), _stages(x, L, 1)
                    oracle.check_bit_exact(self, ROW, f"stage {label} perm gemm_mode 1 vs 0", omma["perm"],
                                           dot64["perm"])
                    oracle.report_row(ROW, f"stage {label} gemm_mode 1 vs 0", phase="report", m=m, layer=layer,
                                      **_stage_fields(omma, dot64))
                    del dot64, omma
            del w, s, L
            torch.cuda.empty_cache()

    def test_block_against_cutlass_fused_moe(self) -> None:
        """D5 (iii) and (iv), reported: k7t (gemm_mode 1) and K7, shared expert zeroed, against cutlass_fused_moe on
        F8's routing (iii) and on the native routing (iv)."""
        top_k = oracle.text_config()["num_experts_per_tok"]
        for layer in LAYERS:
            w, s = _load(layer)
            ops = k7._cutlass_operands(w, s)
            routed = _layer_inputs(dict(w, shared_gate=torch.zeros_like(w["shared_gate"]),
                                        shared_up=torch.zeros_like(w["shared_up"]),
                                        shared_down=torch.zeros_like(w["shared_down"])), s, top_k)
            x_all = k7._moe_input(layer)
            for m in M_CASES:
                label = _label(layer, m)
                with self.subTest(case=f"cutlass {label}"):
                    x = x_all[:m].contiguous()
                    k7t_out = _block(x, routed, 1)
                    k7_out = _block(x, routed, None)
                    f8, _, _ = k7._sglang_routed(x, w, ops, top_k)
                    ids, weights = _native_route(x, w["router"], top_k)
                    controls = {"iii": ("cutlass_fused_moe on fused_topk routing (F8)", f8),
                                "iv": ("cutlass_fused_moe on the native routing (moe_route)",
                                       _cutlass(x, ops, ids, weights))}
                    for part, (control, reference) in controls.items():
                        k7t_fraction, k7t_ulp = _compare(k7t_out, reference)
                        k7_fraction, k7_ulp = _compare(k7_out, reference)
                        oracle.report_row(ROW, f"cutlass ({part}) {label}", phase="report", m=m, layer=layer,
                                          control=control, tactic=_tactic(m), elements=reference.numel(),
                                          k7t_bit_equal=k7t_fraction, k7t_max_ulp_bf16=k7t_ulp,
                                          k7_bit_equal=k7_fraction, k7_max_ulp_bf16=k7_ulp,
                                          k7t_exceeds_k7=k7t_fraction > k7_fraction)
            del w, s, ops, routed
            torch.cuda.empty_cache()

    def test_speed(self) -> None:
        """D6, reported: microseconds per layer call for k7t (gemm_mode 1), K7 and cutlass_fused_moe at every M."""
        top_k = oracle.text_config()["num_experts_per_tok"]
        for layer in LAYERS:
            w, s = _load(layer)
            L = _layer_inputs(w, s, top_k)
            ops = k7._cutlass_operands(w, s)
            x_all = k7._moe_input(layer)
            for m in M_CASES:
                label = _label(layer, m)
                with self.subTest(case=f"speed {label}"):
                    x = x_all[:m].contiguous()
                    ids, weights = _native_route(x, w["router"], top_k)
                    out = oracle.empty(tuple(x.shape), torch.bfloat16)
                    runs = {"k7t": ("block, gemm_mode 1", lambda: _k7t_block(x, L, out, 1)),
                            "K7": ("block", lambda: _k7_block(x, L, out)),
                            "cutlass_fused_moe": (f"routed experts and finalize, {_tactic(m)}",
                                                  lambda: _cutlass(x, ops, ids, weights))}
                    for impl, (scope, run) in runs.items():
                        oracle.report_row(ROW, f"speed {label} {impl}", phase="speed", m=m, layer=layer,
                                          impl=impl, scope=scope, warmup=WARMUP, timed=TIMED, **_timed(run))
            del w, s, L, ops
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
