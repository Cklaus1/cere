"""P3b: FlashInfer 0.6.17's SM120 CUTLASS fused MoE as the routed experts of the native MoE block (tasks/m23-spec.md
P3, M23-D1..D3), as kernel-test entry points only.

Suite `p3b` (GPU1 kernel lab): `python3 scripts/live/native_dev.py --gpu 1 test --suite p3b`, which runs
`python3 -m unittest discover -s /src/executor/tests/gpu -p 'test_p3b_*.py' -v` in a `cerelab-g1-*` container on GPU1
against a `--moe-backend p3b` build. Nothing here is a G1 result; M2.3 judges speed on GPU0 under its own protocol.

Operands (F8, the K7 suite's definitions, `test_kernels_k7_moe.py`): real expert weights, the flashinfer_cutlass scale
set (`_reference_scales`) and FlashInfer's operands (`_cutlass_operands`: w13 = [up; gate] per expert, swizzled block
scales). P3b's own w13 block scales and down block scales are the linear checkpoint bytes swizzled in place by
`cere_ktest_p3b_swizzle` (the INSTANTIATE swizzle). Inputs: K7's `_moe_input` (layer 0 the parity prompts' MoE input,
other layers seeded synthetic RMS-matched BF16, 2,048 rows); M 4,096 is those rows followed by the same rows reversed.

Rows (row key P3B):
  swizzle, asserted (`test_swizzle_equals_sglang`): the in-place swizzle is byte-equal to SGLang's
      `swizzle_blockscale` (the bytes `_cutlass_operands` hands FlashInfer), w13 and down, layers 0, 23, 47;
  pair, asserted (`test_pair_row`): the P3b routed rows (MoeTaps.routed: FlashInfer's finalized BF16 output of m's
      pinned tactic pair, p3b_tactic.h kPinnedBuckets, fused finalize off) are bit-equal to `cutlass_fused_moe`
      untuned (AutoTuner cache cleared, default tactics: GEMM1 index 0, GEMM2 the first of its range, i.e. (0, 20)),
      use_fused_finalize False, fed P3b's own routing (ids and weights of the block), for every m in [1, 16], 17 and
      the prefill buckets 32..4096, on every MoE layer (tasks/m23-spec.md P3 [REVISED: M23T-01]: the n <= 16 rule
      holds because every pin is bit-equal to (0, 20)); the pinned indices per m and the list sizes are reported
      (`cere_ktest_p3b_tactics`);
  block, asserted (`test_block_within_oracle`): the P3b block output within the K7 suite's calibrated tolerance
      (`oracle.calibrate` / `check_within` against SGLang's block and the FP64 `_high_precision_block`), exactly as
      `test_kernels_k7_moe._check_block`, layers 0, 23, 47, M 1, 64, 2048;
  determinism, asserted (`test_repeat_bit_equal`): two calls on the same input give bit-equal block output and routed
      rows, layers 0, 23, 47 at every pair M;
  row invariance, asserted (`test_row_invariance`): the shipped block (m's pinned pair, the executor's rows MoePlans
      at rows 16 as the batched decode and verify steps run it) on ROWINV_ROWS seeded synthetic rows gives every row
      of an m-row call, m in [1, 16] (across the pins' 6 -> 7 bucket switch), rows in order and reversed, the bits of
      that row alone (m = 1, single-row plans): block output, routed rows, top-k ids and weights; layers 0, 23, 47.
      Fail-first (`test_row_invariance_detects_perturbation`): the same comparison flags exactly the one batched row
      whose input is 1 ulp off in every element (ROWINV_PERTURB) and nothing in the unperturbed batch;
  finalize, asserted (`test_rows_equal_runmoe`, tasks/m23-spec.md P13): the block runs FlashInfer's launches only up
      to GEMM2 and K7's combine forms the routed row from the per-slot expert rows with finalizeMoeRoutingKernel's
      arithmetic (MoeTaps.routed_rows); its routed rows and block output are bit-equal to runMoe's finalized rows
      (`cere_ktest_p3b_routed`, same routing, m's pinned pair) fed through K7's routed_in path, every MoE layer, every
      FINALIZE_M. Fail-first (`test_rows_equal_runmoe_detects_slot_order`): the same reference with each token's
      slots reversed (finalize summing in the other order) is flagged in both fields, the unperturbed one in neither;
  arena, asserted (`test_arena_bytes_equal_k7`): DEVICE weight bytes (A4 tensor table, P3b order with w13 runs vs K7
      order, M23-D2: no second expert-weight copy) equal K7's, and the Workspace's shared scratch
      (model::scratch_bytes) exceeds K7's by exactly ARENA_SCRATCH_DELTA per context (ARENA_CONTEXTS): the larger
      P3b MoE workspace, recorded against the I7 floor (GPU0 free >= 10,240 MiB at the measured peak, M23-D2 / N7;
      tasks/review-m23.md), so any further growth fails here first;
  speed, reported (`test_speed`, T7): microseconds per layer call per M bucket for the P3b block (m's pinned pair),
      the P3b block on the reference pair (0, 20) (`P3b_reference`, the pre-[REVISED: M23T-01] pin), the K7 block
      (`cere_ktest_moe_block`) and the SGLang control (its whole MoE block: router `F.linear`, `fused_topk`,
      `cutlass_fused_moe` tuned as SGLang drives it (one `autotune(True)` call per M, then cached), the shared expert
      and `fused_gate_sigmoid_mul_add`), each the median of 20 CUDA-event spans after 5 warm-ups and the mean kernel
      device time (torch profiler); the native spans include the ktest's K2 plan setup and workspace allocation.
  shared fork, asserted (`test_shared_fork_bit_equal`, tasks/m23-spec.md P11R-23): the block that forks its shared
      expert onto the side stream (model::MoeFork), eager and as a replayed captured graph, gives the serial block's
      output, routed rows, ids and weights bit for bit, every m in [1, 16], single-row plans pinned at m and the rows
      MoePlans at 16, layers 0, 23, 47; fail-first (`test_shared_fork_detects_perturbation`): the comparison flags
      exactly the row whose input is 1 ulp off; `test_shared_fork_bound`: taken at m = 16, refused at 17;
  shared fork speed, reported (`test_shared_fork_speed`): serial vs forked block, one captured graph on the rows
      MoePlans, cold L2 (an l2_flush before each replay, ktest/l2_flush.h), median / min of FORK_TIMED replays.

Native entry points (C prototypes; 0 = success, else a cere_status):

  int cere_ktest_p3b_moe_block(const void *x, const void *router_weight, const void *w13_weight,
                               const void *w13_scale, const void *down_weight, const void *down_scale,
                               const void *g1_alphas, const void *g2_alphas, const void *act_scales,
                               const void *shared_gate_weight, const void *shared_up_weight,
                               const void *shared_down_weight, const void *shared_expert_gate, void *out,
                               void *routed, void *topk_ids, void *topk_weights, int64_t m, int64_t hidden,
                               int64_t num_experts, int64_t top_k, int64_t inter, int64_t shared_inter,
                               cudaStream_t stream);
      w13 uint8 [E, 2I, H/2], scales swizzled, act_scales FP32 {a1, a2} on the device; routed BF16 [m, H], topk_ids
      int32 / topk_weights FP32 [m, top_k], each nullable.
  int cere_ktest_p3b_moe_block_taps(... as cere_ktest_p3b_moe_block through topk_weights, const void *routed_in,
                                    const void *shared_in, void *shared, int64_t m, ... int64_t shared_inter,
                                    int64_t gemm1, int64_t gemm2 /* both -1: m's pinned pair */, cudaStream_t);
  int cere_ktest_p3b_moe_block_rows(... as cere_ktest_p3b_moe_block through shared_inter, int64_t gemm1,
                                    int64_t gemm2, int64_t rows /* 0: single-row plans; [1, 16]: the rows
                                    MoePlans, m <= rows */, cudaStream_t);
  int cere_ktest_p3b_moe_block_fork(... as cere_ktest_p3b_moe_block_rows through rows, int64_t fork /* 1: the
                                    plans carry a fork the block must take (m <= 16) */, int64_t warmup,
                                    int64_t timed /* 0: one call; else graph replays */, float *us /* host
                                    [timed] */, int64_t cold /* 1: an L2 flush before each timed replay */,
                                    cudaStream_t);
  int cere_ktest_p3b_swizzle(void *scales, int64_t batches, int64_t rows, int64_t cols, cudaStream_t stream);
  int cere_ktest_p3b_tactics(int64_t *out /* host [4]: n1, n2, m's pinned g1, g2 */, int64_t m, cudaStream_t stream);
  int cere_ktest_p3b_tactic_configs(int64_t *out /* host [2 * cap * 5] */, int64_t cap, cudaStream_t stream);
  int cere_ktest_p3b_arena_bytes(const cere_tensor_desc *tensors, uint64_t count,
                                 const cere_qwen4_exp_config *cfg, int64_t ctx,
                                 uint64_t *out /* host [6] */, cudaStream_t stream);
"""
from __future__ import annotations

import importlib
import statistics
import unittest

import torch
import torch.nn.functional as F
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType

import oracle

k7 = importlib.import_module("test_kernels_k7_moe")  # F8's inputs, weights, scale set and cutlass operands

ROW = "P3B"
LAYERS = (0, 23, 47)
# Every m in [1, 16] (the n <= 16 rule), the first m past it, and the prefill buckets of the pins (p3b_tactic.h).
PAIR_M = tuple(range(1, 17)) + (17, 32, 64, 128, 256, 512, 1024, 2048, 4096)
SPEED_M = (1, 2, 4, 8, 16, 64, 256, 2048, 4096)
ARENA_CONTEXTS = (2048, 4096, 32768, 262144)  # dense limit, kMaxChunkTokens rows, QSA contexts
# P3b - K7 shared scratch bytes per context (the MoE workspace at workspace_rows(ctx)): 133.0 MiB at 2048, 255.7 MiB
# from 4096 rows on since P13 (routed_rows' workspace, no finalize output; the 2026-10-11 lab run before it: 143.0 /
# 275.7 MiB). GPU0's measured peak left 17,934 MiB free with the larger delta (sampler_live-20261011T144358Z), so
# >= 17,658 MiB remain >= 10,240 (I7). Weight bytes stay equal.
ARENA_SCRATCH_DELTA = {2048: 139_447_552, 4096: 268_086_528, 32768: 268_086_528, 262144: 268_086_528}
WARMUP = 5
TIMED = 20
REFERENCE = (128, 128, 128, 0, 0)  # CtaShape128x128x128B, no swapAB, epilogue NONE: FlashInfer's untuned (0, 20)
TACTIC_CAP = 64
# Row invariance: every m of a rows call (GemmPlan::kMaxRows = 16, the batched decode and verify bound), which spans
# the pins' bucket switch at 6 -> 7 (p3b_tactic.h kPinnedBuckets); seeded synthetic rows, RMS-matched to K7's input.
ROWINV_ROWS = 16
ROWINV_M = tuple(range(1, ROWINV_ROWS + 1))
ROWINV_SEED = 46_000
ROWINV_PERTURB = (23, 8, 5)  # fail-first: (layer, m, row) whose batched input is 1 ulp off in every element
# P13 finalize equivalence: every decode / verify m (rows MoePlans bound 16), the first m past it and prefill sizes.
FINALIZE_M = tuple(range(1, 17)) + (17, 64, 256, 2048)
FINALIZE_PERTURB = (23, 16)  # fail-first: (layer, m) whose reference sums each token's slots in reverse order
# P11R-23 shared-expert fork: every m of the fork (k2s::kMaxM, the rows MoePlans bound), its fail-first perturbation
# (layer, m, row) and the cold-L2 speed rows (an l2_flush of 512 MiB, 4x the L2, before each replay).
FORK_ROWS = 16
FORK_PERTURB = (23, 4, 2)
FORK_SPEED_M = (1, 4, 8, 16)
FORK_TIMED = 50


def _x(layer: int, m: int) -> torch.Tensor:
    x_all = k7._moe_input(layer)
    if m > x_all.shape[0]:
        x_all = torch.cat([x_all, x_all.flip(0)])
    return x_all[:m].contiguous()


def _swizzle(linear: torch.Tensor) -> torch.Tensor:
    """P3b's in-place swizzle of a [B, rows, cols] uint8 block-scale tensor (a copy is swizzled)."""
    t = linear.contiguous().clone()
    b, rows, cols = t.shape
    oracle.call("cere_ktest_p3b_swizzle", t, b, rows, cols)
    return t


def _operands(w: dict, s: dict, top_k: int) -> dict:
    """P3b's layer operands in its arena layout (M23-D2)."""
    num_experts, inter, half = w["gate_weight"].shape
    w13 = torch.cat([w["up_weight"].view(torch.uint8), w["gate_weight"].view(torch.uint8)], dim=1).contiguous()
    w13_scale = torch.cat([w["up_weight_scale"].view(torch.uint8), w["gate_weight_scale"].view(torch.uint8)], dim=1)
    act = torch.tensor([float(s["w13_input_scale_quant"].item()), float(s["w2_input_scale_quant"].item())],
                       dtype=torch.float32, device=oracle.DEVICE)
    return {"router": w["router"], "w13": w13, "w13_scale": _swizzle(w13_scale),
            "down": w["down_weight"].view(torch.uint8).contiguous(),
            "down_scale": _swizzle(w["down_weight_scale"].view(torch.uint8)), "g1": s["g1_alphas"],
            "g2": s["g2_alphas"], "act": act, "shared_gate": w["shared_gate"], "shared_up": w["shared_up"],
            "shared_down": w["shared_down"], "shared_expert_gate": w["shared_expert_gate"],
            "hidden": 2 * half, "num_experts": num_experts, "top_k": top_k, "inter": inter,
            "shared_inter": w["shared_gate"].shape[0]}


def _p3b(x: torch.Tensor, P: dict, out: torch.Tensor, routed=None, ids=None, weights=None) -> None:
    oracle.call("cere_ktest_p3b_moe_block", x, P["router"], P["w13"], P["w13_scale"], P["down"], P["down_scale"],
                P["g1"], P["g2"], P["act"], P["shared_gate"], P["shared_up"], P["shared_down"],
                P["shared_expert_gate"], out, routed, ids, weights, x.shape[0], P["hidden"], P["num_experts"],
                P["top_k"], P["inter"], P["shared_inter"])


def _p3b_pair(x: torch.Tensor, P: dict, out: torch.Tensor, gemm1: int, gemm2: int) -> None:
    """The P3b block with the routed experts on the given tactic pair (indices into the runner's lists)."""
    oracle.call("cere_ktest_p3b_moe_block_taps", x, P["router"], P["w13"], P["w13_scale"], P["down"],
                P["down_scale"], P["g1"], P["g2"], P["act"], P["shared_gate"], P["shared_up"], P["shared_down"],
                P["shared_expert_gate"], out, None, None, None, None, None, None, x.shape[0], P["hidden"],
                P["num_experts"], P["top_k"], P["inter"], P["shared_inter"], gemm1, gemm2)


def tactic_configs() -> tuple[list[tuple], list[tuple]]:
    """The runner's GEMM1 and GEMM2 lists as (tile_m, tile_n, tile_k_bytes, swap_ab, finalize) tuples."""
    counts = torch.zeros(4, dtype=torch.int64)
    oracle.call("cere_ktest_p3b_tactics", counts, 1)
    n1, n2 = int(counts[0]), int(counts[1])
    if max(n1, n2) > TACTIC_CAP:
        raise oracle.OracleError(f"tactic lists {n1}/{n2} exceed {TACTIC_CAP}")
    out = torch.zeros(2 * TACTIC_CAP * 5, dtype=torch.int64)
    oracle.call("cere_ktest_p3b_tactic_configs", out, TACTIC_CAP)
    rows = [tuple(int(v) for v in out[i * 5:(i + 1) * 5].tolist()) for i in range(2 * TACTIC_CAP)]
    return rows[:n1], rows[TACTIC_CAP:TACTIC_CAP + n2]


def reference_indices(g1_list: list[tuple], g2_list: list[tuple]) -> tuple[int, int]:
    """The reference pair's indices, found by configuration (exactly one match per list, as INSTANTIATE)."""
    if g1_list.count(REFERENCE) != 1 or g2_list.count(REFERENCE) != 1:
        raise oracle.OracleError(f"the reference {REFERENCE} does not identify one tactic per list")
    return g1_list.index(REFERENCE), g2_list.index(REFERENCE)


def _p3b_block(x: torch.Tensor, P: dict) -> dict[str, torch.Tensor]:
    m = x.shape[0]
    r = {"out": oracle.empty((m, P["hidden"]), torch.bfloat16),
         "routed": oracle.empty((m, P["hidden"]), torch.bfloat16),
         "ids": oracle.empty((m, P["top_k"]), torch.int32), "weights": oracle.empty((m, P["top_k"]), torch.float32)}
    _p3b(x, P, r["out"], r["routed"], r["ids"], r["weights"])
    return r


def _rows_block(x: torch.Tensor, P: dict, rows: int) -> dict[str, torch.Tensor]:
    """`cere_ktest_p3b_moe_block_rows` on m's pinned pair: rows 0 the single-row plans, else the rows MoePlans."""
    m = x.shape[0]
    r = {"out": oracle.empty((m, P["hidden"]), torch.bfloat16),
         "routed": oracle.empty((m, P["hidden"]), torch.bfloat16),
         "ids": oracle.empty((m, P["top_k"]), torch.int32), "weights": oracle.empty((m, P["top_k"]), torch.float32)}
    oracle.call("cere_ktest_p3b_moe_block_rows", x, P["router"], P["w13"], P["w13_scale"], P["down"],
                P["down_scale"], P["g1"], P["g2"], P["act"], P["shared_gate"], P["shared_up"], P["shared_down"],
                P["shared_expert_gate"], r["out"], r["routed"], r["ids"], r["weights"], m, P["hidden"],
                P["num_experts"], P["top_k"], P["inter"], P["shared_inter"], -1, -1, rows)
    return r


def _rowinv_input(layer: int) -> torch.Tensor:
    """ROWINV_ROWS distinct seeded rows of `layer`, each scaled to the RMS of K7's input row of the same index."""
    z = torch.randn((ROWINV_ROWS, k7._moe_input(layer).shape[1]), generator=oracle.generator(ROWINV_SEED + layer),
                    device=oracle.DEVICE, dtype=torch.float32)
    z = z / k7._row_rms(z) * k7._row_rms(k7._moe_input(layer)[:ROWINV_ROWS])
    return z.to(torch.bfloat16).contiguous()


def _solo_rows(x: torch.Tensor, P: dict) -> list[dict[str, torch.Tensor]]:
    """Each row of x alone (m = 1, the single-row plans): the plain decode step's block."""
    return [_rows_block(x[r:r + 1].contiguous(), P, 0) for r in range(x.shape[0])]


def _differing_rows(batch: dict[str, torch.Tensor], solo: list[dict[str, torch.Tensor]],
                    order: list[int]) -> list[tuple[int, str]]:
    """(row, first differing field) for every batch position whose out, routed rows, ids or weights are not
    bit-equal to its row alone; `order[i]` = the solo row at batch position i."""
    differing = []
    for i, r in enumerate(order):
        for field in ("out", "routed", "ids", "weights"):
            a = batch[field][i].contiguous().view(torch.uint8)
            b = solo[r][field][0].contiguous().view(torch.uint8)
            if not torch.equal(a, b):
                differing.append((r, field))
                break
    return differing


def _shared_experts(solo: list[dict[str, torch.Tensor]], order: list[int]) -> int:
    """Experts chosen by two or more rows of the batch (their rows share a grouped expert GEMM)."""
    ids = torch.cat([solo[r]["ids"] for r in order]).view(-1)
    return int((torch.bincount(ids.long()) >= 2).sum().item())


def _runmoe_block(x: torch.Tensor, P: dict, ids: torch.Tensor, weights: torch.Tensor) -> dict[str, torch.Tensor]:
    """The pre-P13 P3b block on the routing (ids, weights): FlashInfer's whole runMoe (finalizeMoeRoutingKernel
    included) on m's pinned pair (`cere_ktest_p3b_routed`), its finalized rows through K7's routed_in path (the
    shared expert and combine)."""
    m = x.shape[0]
    routed = oracle.empty((m, P["hidden"]), torch.bfloat16)
    oracle.call("cere_ktest_p3b_routed", x, P["w13"], P["w13_scale"], P["down"], P["down_scale"], P["g1"], P["g2"],
                P["act"], ids, weights, routed, m, P["hidden"], P["num_experts"], P["top_k"], P["inter"], -1, -1, 0,
                0, None)
    out = oracle.empty((m, P["hidden"]), torch.bfloat16)
    oracle.call("cere_ktest_p3b_moe_block_taps", x, P["router"], P["w13"], P["w13_scale"], P["down"],
                P["down_scale"], P["g1"], P["g2"], P["act"], P["shared_gate"], P["shared_up"], P["shared_down"],
                P["shared_expert_gate"], out, None, None, None, routed, None, None, m, P["hidden"],
                P["num_experts"], P["top_k"], P["inter"], P["shared_inter"], -1, -1)
    return {"out": out, "routed": routed}


def _finalize_differs(block: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> list[str]:
    """The fields ("routed", "out") of `block` that are not bit-equal to `reference`."""
    return [f for f in ("routed", "out") if not _bit_equal(block[f], reference[f])]


def _cutlass(x: torch.Tensor, ops: dict, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """`cutlass_fused_moe` as SGLang calls it (`test_kernels_k7_moe._sglang_routed`), fused finalize off."""
    m = x.shape[0]
    output = torch.empty(m, x.shape[1], dtype=torch.bfloat16, device=x.device)
    return cutlass_fused_moe(
        output=output, input=x, token_selected_experts=ids.to(torch.int), token_final_scales=weights,
        fc1_expert_weights=ops["w13_weight"], fc2_expert_weights=ops["w2_weight"], output_dtype=torch.bfloat16,
        input_sf=None, quant_scales=ops["quant_scales"], ep_size=1, ep_rank=0, tp_size=1, tp_rank=0,
        tune_max_num_tokens=k7.next_power_of_2(m),
        activation_type=ActivationType(k7.get_activation_type("silu", is_gated=True)), enable_alltoall=False,
        use_fused_finalize=False,
    )[0]


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _load(layer: int):
    w = k7._load_moe_layer(layer)
    return w, k7._reference_scales(w)


def _fork_block(x: torch.Tensor, P: dict, rows: int, fork: int, timed: int = 0, cold: bool = False,
                status_only: bool = False):
    """`cere_ktest_p3b_moe_block_fork` on m's pinned pair (rows 0: single-row plans; else the rows MoePlans), fork 1
    the block must fork. timed 0: one eager call; else a captured graph replayed WARMUP then `timed` times (`us` the
    spans, the outputs the last replay's), `cold` an L2 flush before each. With `status_only` the call's status."""
    m = x.shape[0]
    r = {"out": oracle.empty((m, P["hidden"]), torch.bfloat16),
         "routed": oracle.empty((m, P["hidden"]), torch.bfloat16),
         "ids": oracle.empty((m, P["top_k"]), torch.int32), "weights": oracle.empty((m, P["top_k"]), torch.float32)}
    us = torch.zeros(max(timed, 1), dtype=torch.float32)
    args = (x, P["router"], P["w13"], P["w13_scale"], P["down"], P["down_scale"], P["g1"], P["g2"], P["act"],
            P["shared_gate"], P["shared_up"], P["shared_down"], P["shared_expert_gate"], r["out"], r["routed"],
            r["ids"], r["weights"], m, P["hidden"], P["num_experts"], P["top_k"], P["inter"], P["shared_inter"], -1,
            -1, rows, fork, WARMUP if timed else 0, timed, us if timed else None, int(cold))
    if status_only:
        return oracle.call_status("cere_ktest_p3b_moe_block_fork", *args)
    oracle.call("cere_ktest_p3b_moe_block_fork", *args)
    r["us"] = us.tolist()[:timed]
    return r


def _differing_fields(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> list[str]:
    """The fields of two _fork_block results that are not bit-equal (out, routed, ids, weights)."""
    return [f for f in ("out", "routed", "ids", "weights")
            if not torch.equal(a[f].view(torch.uint8), b[f].view(torch.uint8))]


def _timed(run) -> dict:
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


class P3bMoe(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.top_k = oracle.text_config()["num_experts_per_tok"]
        cls.layers = oracle.text_config()["num_hidden_layers"]

    def test_swizzle_equals_sglang(self) -> None:
        for layer in LAYERS:
            w, s = _load(layer)
            ops = k7._cutlass_operands(w, s)
            P = _operands(w, s, self.top_k)
            with self.subTest(layer=layer):
                sglang = [ops["quant_scales"][i].contiguous().view(torch.uint8).view(-1) for i in (1, 4)]
                w13_equal = torch.equal(P["w13_scale"].view(-1), sglang[0])
                down_equal = torch.equal(P["down_scale"].view(-1), sglang[1])
                oracle.report_row(ROW, f"swizzle L{layer}", phase="assert", layer=layer, w13_equal=w13_equal,
                                  down_equal=down_equal)
                self.assertTrue(w13_equal and down_equal, f"{ROW} swizzle L{layer}: w13 {w13_equal} down {down_equal}")
            del w, s, ops, P
            torch.cuda.empty_cache()

    def test_pair_row(self) -> None:
        for m in PAIR_M:
            tactics = torch.zeros(4, dtype=torch.int64)
            oracle.call("cere_ktest_p3b_tactics", tactics, m)
            n1, n2, g1, g2 = (int(v) for v in tactics.tolist())
            oracle.report_row(ROW, f"pair tactics M{m}", phase="report", m=m, gemm1_tactics=n1, gemm2_tactics=n2,
                              pinned_gemm1=g1, pinned_gemm2=g2, pinned_gemm2_absolute=n1 + g2)
        tuner = AutoTuner.get()
        failed = []
        for layer in range(self.layers):
            w, s = _load(layer)
            ops = k7._cutlass_operands(w, s)
            P = _operands(w, s, self.top_k)
            equal = 0
            for m in PAIR_M:
                tuner.clear_cache()
                x = _x(layer, m)
                r = _p3b_block(x, P)
                control = _cutlass(x, ops, r["ids"], r["weights"])
                if _bit_equal(r["routed"], control):
                    equal += 1
                else:
                    failed.append(f"L{layer} M{m}")
                del x, r, control
            oracle.report_row(ROW, f"pair L{layer}", phase="assert", layer=layer, cases=len(PAIR_M),
                              bit_equal_cases=equal)
            del w, s, ops, P
            torch.cuda.empty_cache()
        self.assertEqual(failed, [], f"{ROW} pair row: P3b routed rows differ from the untuned call at {failed}")

    def test_block_within_oracle(self) -> None:
        for layer in LAYERS:
            w, s = _load(layer)
            ops = k7._cutlass_operands(w, s)
            P = _operands(w, s, self.top_k)
            for m in k7.M_CASES:
                case = f"block L{layer} ({k7._input_label(layer)}) M{m}"
                with self.subTest(case=case):
                    x = _x(layer, m)
                    reference, topk_weights, topk_ids = k7._sglang_block(x, w, ops, self.top_k)
                    high_precision = k7._high_precision_block(x, w, s, topk_weights, topk_ids)
                    c = oracle.calibrate(ROW, case, reference, high_precision, oracle.FLOOR_BLOCK)
                    native = _p3b_block(x, P)["out"]
                    limit = oracle.calibrated_limit(c, oracle.FLOOR_BLOCK)
                    oracle.check_within(self, ROW, case, oracle.rel_l2(native, reference), c=c, limit=limit)
            del w, s, ops, P
            torch.cuda.empty_cache()

    def test_repeat_bit_equal(self) -> None:
        failed = []
        for layer in LAYERS:
            w, s = _load(layer)
            P = _operands(w, s, self.top_k)
            for m in PAIR_M:
                x = _x(layer, m)
                a = _p3b_block(x, P)
                b = _p3b_block(x, P)
                if not (_bit_equal(a["out"], b["out"]) and _bit_equal(a["routed"], b["routed"])):
                    failed.append(f"L{layer} M{m}")
            oracle.report_row(ROW, f"repeat L{layer}", phase="assert", layer=layer, cases=len(PAIR_M),
                              failed_cases=sum(f.startswith(f"L{layer} ") for f in failed))
            del w, s, P
            torch.cuda.empty_cache()
        self.assertEqual(failed, [], f"{ROW} determinism: repeat differs at {failed}")

    def test_row_invariance(self) -> None:
        """The shipped block (m's pinned pair, the rows MoePlans) gives every row of an m-row call the bits of that
        row alone, at every m in [1, 16], rows in order and reversed."""
        failed = []
        for layer in LAYERS:
            w, s = _load(layer)
            P = _operands(w, s, self.top_k)
            x = _rowinv_input(layer)
            solo = _solo_rows(x, P)
            for m in ROWINV_M:
                tactics = torch.zeros(4, dtype=torch.int64)
                oracle.call("cere_ktest_p3b_tactics", tactics, m)
                equal = checked = 0
                for name, order in (("forward", list(range(m))), ("reversed", list(range(m))[::-1])):
                    if name == "reversed" and m == 1:
                        continue
                    batch = _rows_block(x[order].contiguous(), P, ROWINV_ROWS)
                    differing = _differing_rows(batch, solo, order)
                    checked += m
                    equal += m - len(differing)
                    failed += [f"M{m} L{layer} {name} row {r} {field}" for r, field in differing]
                oracle.report_row(ROW, f"rowinv L{layer} M{m}", phase="assert", layer=layer, m=m,
                                  pinned_gemm1=int(tactics[2]), pinned_gemm2=int(tactics[3]),
                                  shared_experts=_shared_experts(solo, list(range(m))), rows_checked=checked,
                                  rows_equal=equal)
            del w, s, P, x, solo
            torch.cuda.empty_cache()
        oracle.report_row(ROW, "rowinv first differing", phase="assert", first=failed[0] if failed else "none",
                          differing=len(failed))
        self.assertEqual(failed, [], f"{ROW} row invariance: batched rows differ from the row alone at {failed}")

    def test_row_invariance_detects_perturbation(self) -> None:
        """Fail-first for test_row_invariance: one batched row 1 ulp off in every element is the only row it flags."""
        layer, m, row = ROWINV_PERTURB
        w, s = _load(layer)
        P = _operands(w, s, self.top_k)
        x = _rowinv_input(layer)
        solo = _solo_rows(x[:m].contiguous(), P)
        bad = x[:m].clone()
        bad[row].view(torch.int16).bitwise_xor_(1)
        differing = _differing_rows(_rows_block(bad, P, ROWINV_ROWS), solo, list(range(m)))
        clean = _differing_rows(_rows_block(x[:m].contiguous(), P, ROWINV_ROWS), solo, list(range(m)))
        oracle.report_row(ROW, f"rowinv perturbed L{layer} M{m} row {row}", phase="assert", layer=layer, m=m,
                          perturbed_row=row, flagged_rows=len(differing),
                          flagged_perturbed=[r for r, _ in differing] == [row], clean_flagged_rows=len(clean))
        self.assertEqual([r for r, _ in differing], [row], f"{ROW} rowinv perturbed: flagged {differing}")
        self.assertEqual(clean, [], f"{ROW} rowinv perturbed: the unperturbed batch flagged {clean}")

    def test_rows_equal_runmoe(self) -> None:
        """P13: the block (FlashInfer's launches up to GEMM2, K7's combine forming the routed row with
        finalizeMoeRoutingKernel's arithmetic) gives the routed rows and the block output of runMoe's finalize fed
        through K7's routed_in path, bit for bit, on its own routing, every MoE layer, every FINALIZE_M."""
        failed = []
        for layer in range(self.layers):
            w, s = _load(layer)
            P = _operands(w, s, self.top_k)
            equal = 0
            for m in FINALIZE_M:
                x = _x(layer, m)
                r = _p3b_block(x, P)
                differs = _finalize_differs(r, _runmoe_block(x, P, r["ids"], r["weights"]))
                if differs:
                    failed.append(f"L{layer} M{m} {differs[0]}")
                else:
                    equal += 1
                del x, r
            oracle.report_row(ROW, f"finalize L{layer}", phase="assert", layer=layer, cases=len(FINALIZE_M),
                              bit_equal_cases=equal)
            del w, s, P
            torch.cuda.empty_cache()
        self.assertEqual(failed, [], f"{ROW} finalize: the block differs from runMoe's finalize at {failed}")

    def test_rows_equal_runmoe_detects_slot_order(self) -> None:
        """Fail-first for test_rows_equal_runmoe: the reference fed the same routing with each token's slots reversed
        (the same experts and weights, finalize summing in the other order) is flagged in the routed row (the output,
        BF16 of routed + gate * shared, may round the 1-ulp routed differences away); the unperturbed one is not."""
        layer, m = FINALIZE_PERTURB
        w, s = _load(layer)
        P = _operands(w, s, self.top_k)
        x = _x(layer, m)
        r = _p3b_block(x, P)
        reversed_ = _finalize_differs(r, _runmoe_block(x, P, r["ids"].flip(1).contiguous(),
                                                       r["weights"].flip(1).contiguous()))
        clean = _finalize_differs(r, _runmoe_block(x, P, r["ids"], r["weights"]))
        oracle.report_row(ROW, f"finalize slot order L{layer} M{m}", phase="assert", layer=layer, m=m,
                          reversed_flagged_fields=len(reversed_), clean_flagged_fields=len(clean))
        self.assertIn("routed", reversed_, f"{ROW} finalize slot order: reversed flagged {reversed_}")
        self.assertEqual(clean, [], f"{ROW} finalize slot order: the unperturbed reference flagged {clean}")

    def test_arena_bytes_equal_k7(self) -> None:
        k8 = importlib.import_module("test_model_k8_forward")
        entries, desc_buf, _names = k8.a4_tensor_table()
        cfg_buf = k8.a4_config(oracle.text_config())
        weights, scratch = {}, {}
        for ctx in ARENA_CONTEXTS:
            out = torch.zeros(6, dtype=torch.int64)
            oracle.call("cere_ktest_p3b_arena_bytes", desc_buf, len(entries), cfg_buf, ctx, out)
            p3b, k7_bytes, ws_p3b, ws_k7, sc_p3b, sc_k7 = (int(v) for v in out.tolist())
            oracle.report_row(ROW, f"arena ctx{ctx}", phase="assert", context=ctx, weight_bytes_p3b=p3b,
                              weight_bytes_k7=k7_bytes, weight_delta=p3b - k7_bytes, moe_workspace_p3b=ws_p3b,
                              moe_workspace_k7=ws_k7, scratch_p3b=sc_p3b, scratch_k7=sc_k7,
                              arena_delta=(p3b - k7_bytes) + (sc_p3b - sc_k7))
            weights[ctx] = p3b - k7_bytes
            scratch[ctx] = sc_p3b - sc_k7
        self.assertEqual(weights, {ctx: 0 for ctx in ARENA_CONTEXTS}, f"{ROW} arena: P3b - K7 weight bytes")
        self.assertEqual(scratch, ARENA_SCRATCH_DELTA, f"{ROW} arena: P3b - K7 shared scratch bytes per context")

    def test_speed(self) -> None:
        ref = reference_indices(*tactic_configs())
        for layer in LAYERS:
            w, s = _load(layer)
            ops = k7._cutlass_operands(w, s)
            P = _operands(w, s, self.top_k)
            L = {"a1": float(s["w13_input_scale_quant"].item()), "a2": float(s["w2_input_scale_quant"].item())}
            for m in SPEED_M:
                x = _x(layer, m)
                out = oracle.empty(tuple(x.shape), torch.bfloat16)

                def k7_run() -> None:
                    oracle.call("cere_ktest_moe_block", x, w["router"], w["gate_weight"], w["gate_weight_scale"],
                                w["up_weight"], w["up_weight_scale"], w["down_weight"], w["down_weight_scale"],
                                s["g1_alphas"], s["g2_alphas"], L["a1"], L["a2"], w["shared_gate"], w["shared_up"],
                                w["shared_down"], w["shared_expert_gate"], out, m, P["hidden"], P["num_experts"],
                                self.top_k, P["inter"], P["shared_inter"])

                def sglang_run() -> None:
                    k7._sglang_block(x, w, ops, self.top_k)

                AutoTuner.get().clear_cache()
                with autotune(True):
                    sglang_run()
                torch.cuda.synchronize(oracle.DEVICE)
                runs = {"P3b": ("block, the pinned pair of m", lambda: _p3b(x, P, out)),
                        "P3b_reference": ("block, the reference pair (0, 20)", lambda: _p3b_pair(x, P, out, *ref)),
                        "K7": ("block", k7_run),
                        "sglang_flashinfer_cutlass": ("block: router, fused_topk, cutlass_fused_moe tuned "
                                                      "(autotune(True) once at this M), shared expert, "
                                                      "fused_gate_sigmoid_mul_add", sglang_run)}
                for impl, (scope, run) in runs.items():
                    oracle.report_row(ROW, f"speed L{layer} M{m} {impl}", phase="speed", m=m, layer=layer,
                                      impl=impl, scope=scope, warmup=WARMUP, timed=TIMED, **_timed(run))
            del w, s, ops, P
            AutoTuner.get().clear_cache()
            torch.cuda.empty_cache()

    # -- P11R-23 (m23 design note): the shared-expert fork -------------------------------------------------------

    def test_shared_fork_bit_equal(self) -> None:
        """The block that forks its shared expert onto the side stream (eager, and replayed as a captured graph with
        the fork one branch) gives the serial block's bits at every m in [1, FORK_ROWS], single-row plans pinned at m
        and the rows MoePlans at FORK_ROWS: block output, routed rows, top-k ids and weights."""
        failed = []
        cases = 0
        for layer in LAYERS:
            w, s = _load(layer)
            P = _operands(w, s, self.top_k)
            for rows in (0, FORK_ROWS):
                for m in range(1, FORK_ROWS + 1):
                    x = _x(layer, m)
                    serial = _fork_block(x, P, rows, 0)
                    for mode, timed in (("eager", 0), ("graph", 1)):
                        forked = _fork_block(x, P, rows, 1, timed)
                        cases += 1
                        failed += [f"L{layer} R{rows} M{m} {mode} {f}" for f in _differing_fields(forked, serial)]
            del w, s, P
            torch.cuda.empty_cache()
        oracle.report_row(ROW, "shared fork", phase="assert", cases=cases, differing=len(failed),
                          first=failed[0] if failed else "none")
        self.assertEqual(failed, [], f"{ROW} shared fork: the forked block differs from the serial one at {failed}")

    def test_shared_fork_detects_perturbation(self) -> None:
        """Fail-first for test_shared_fork_bit_equal: the forked block on an input whose row FORK_PERTURB[2] is 1 ulp
        off in every element differs from the serial block in that row's output (and no other row), and the same
        comparison on the unperturbed input flags nothing."""
        layer, m, row = FORK_PERTURB
        w, s = _load(layer)
        P = _operands(w, s, self.top_k)
        x = _x(layer, m)
        bad = x.clone()
        bad[row].view(torch.int16).bitwise_xor_(1)
        serial = _fork_block(x, P, FORK_ROWS, 0)["out"].view(torch.int16)
        flagged = (_fork_block(bad, P, FORK_ROWS, 1)["out"].view(torch.int16) != serial).any(dim=1)
        clean = (_fork_block(x, P, FORK_ROWS, 1)["out"].view(torch.int16) != serial).any(dim=1)
        flagged_rows = flagged.nonzero().flatten().tolist()
        clean_rows = clean.nonzero().flatten().tolist()
        oracle.report_row(ROW, f"shared fork perturbed L{layer} M{m} row {row}", phase="assert", layer=layer, m=m,
                          perturbed_row=row, flagged_rows=len(flagged_rows), clean_flagged_rows=len(clean_rows))
        self.assertEqual(flagged_rows, [row], f"{ROW} shared fork perturbed: flagged {flagged_rows}")
        self.assertEqual(clean_rows, [], f"{ROW} shared fork perturbed: the unperturbed input flagged {clean_rows}")

    def test_shared_fork_bound(self) -> None:
        """The fork is taken at m = FORK_ROWS (status 0) and refused one row past it (CERE_ERR_ARGUMENT:
        kernels::moe_shared_forks is false, so the block would run serially)."""
        w, s = _load(0)
        P = _operands(w, s, self.top_k)
        statuses = {m: _fork_block(_x(0, m), P, 0, 1, status_only=True) for m in (FORK_ROWS, FORK_ROWS + 1)}
        oracle.report_row(ROW, "shared fork bound", phase="assert", status_at_bound=statuses[FORK_ROWS],
                          status_past_bound=statuses[FORK_ROWS + 1])
        self.assertEqual(statuses, {FORK_ROWS: 0, FORK_ROWS + 1: oracle.CERE_ERR_ARGUMENT})

    def test_shared_fork_speed(self) -> None:
        """Reported: the serial and the forked block as one captured graph on the rows MoePlans, cold L2 (an
        l2_flush before every replay), the median and minimum of FORK_TIMED replay spans per m."""
        for layer in LAYERS:
            w, s = _load(layer)
            P = _operands(w, s, self.top_k)
            for m in FORK_SPEED_M:
                x = _x(layer, m)
                spans = {fork: _fork_block(x, P, FORK_ROWS, fork, FORK_TIMED, cold=True)["us"] for fork in (0, 1)}
                serial, forked = (statistics.median(spans[f]) for f in (0, 1))
                oracle.report_row(ROW, f"shared fork speed L{layer} M{m}", phase="speed", layer=layer, m=m,
                                  rows=FORK_ROWS, warmup=WARMUP, timed=FORK_TIMED, cold=1,
                                  serial_us_median=serial, forked_us_median=forked,
                                  serial_us_min=min(spans[0]), forked_us_min=min(spans[1]),
                                  saved_us_median=serial - forked)
            del w, s, P
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
