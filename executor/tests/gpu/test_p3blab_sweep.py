"""P3b tactic sweep (tasks/m23-spec.md P3 [REVISED: M23T-01]): offline and deterministic, never a run-time autotuner.

Suite `p3blab` (GPU1 kernel lab): `python3 scripts/live/native_dev.py --gpu 1 test --suite p3blab` against a
`--moe-backend p3b` build. Nothing here is a G1 result; it is the evidence behind the per-M-bucket pins of
`executor/cuda/src/p3b/p3b_tactic.h` (summarised by `bench/reports/p3b-tactic-sweep-20261012/make_report.py`).

Inputs as the `p3b` suite's (`test_p3b_moe.py`): real expert weights of layers 0, 23 and 47 in P3b's arena layout, K7's
`_moe_input` rows (layer 0 the parity prompts' MoE input, 23/47 seeded RMS-matched synthetic), M = every m in [1, 64]
(17..64 since tasks/batch-concurrency-spec.md BC0: the band row-invariant decode above 16 rows would run in) and the
prefill buckets 128..4096 (powers of two). The routing (ids and weights) is the P3b block's own
(`cere_ktest_p3b_moe_block`), so every candidate sees the executor's routing.

Candidates: every GEMM1 tactic x every GEMM2 tactic of the runner's own lists (fused finalize off, so no GEMM2
FINALIZE tactic is listed: F9's non-deterministic ones are never candidates), by index (`cere_ktest_p3b_routed`). The
reference is the untuned default pair (0, 20) = CtaShape128x128x128B, no swapAB, epilogue NONE, on both GEMMs, found
by configuration in the lists. Per (layer, M, pair):
  works: the call returns 0 (a tactic the runner rejects returns a status; the pair is reported as failing);
  bit_equal: the finalized BF16 routed rows equal the reference pair's bit for bit;
  repeatable: REPEATS eager calls and the CUDA-graph replay's output are bit-equal to the first call;
  graph time: one call captured in a CUDA graph, WARMUP replays then TIMED replays each between two CUDA events
      (median and min, microseconds; kernel time without host launch cost).

Rows (row key P3BSWEEP, counts and timings only):
  `tactic gemm<g> <i>`: the configuration of list entry i (tile_m, tile_n, tile_k_bytes, swap_ab, finalize);
  `pair M<m> g1 <i> g2 <j>`: per working pair, the layers where it is bit-equal and repeatable and its graph median
      per layer and summed over the layers;
  `failing M<m>`: the pairs that returned a status at that M, by count;
  `fastest M<m>`: the fastest pair (summed graph median) among those bit-equal and repeatable on every layer.
Asserted (`test_sweep`): the reference pair works and is repeatable at every (layer, M), and at every M at least one
pair works; the bit-equality of the other pairs is reported, not asserted (the pins' own row is the `p3b` suite's
pair row).

Native entry points (C prototypes; 0 = success, else a cere_status):

  int cere_ktest_p3b_tactic_configs(int64_t *out /* host [2 * cap * 5] */, int64_t cap, cudaStream_t stream);
  int cere_ktest_p3b_routed(const void *x, const void *w13_weight, const void *w13_scale, const void *down_weight,
                            const void *down_scale, const void *g1_alphas, const void *g2_alphas,
                            const void *act_scales, const void *topk_ids, const void *topk_weights, void *out,
                            int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter,
                            int64_t gemm1, int64_t gemm2 /* both -1: the pins */, int64_t warmup, int64_t timed,
                            float *us /* host [timed] */, cudaStream_t stream);
"""
from __future__ import annotations

import ctypes
import importlib
import statistics
import unittest

import torch

import oracle

p3b = importlib.import_module("test_p3b_moe")  # the p3b suite's operands, inputs and block call

ROW = "P3BSWEEP"
LAYERS = (0, 23, 47)
SWEEP_M = tuple(range(1, 65)) + (128, 256, 512, 1024, 2048, 4096)
REPEATS = 3
WARMUP = 5
TIMED = 20


CERE_ERR_INTERNAL = -6  # cere_executor.h; p3b_ktest.cpp: "A tactic the runner cannot run: CERE_ERR_INTERNAL."


def routed(x: torch.Tensor, P: dict, ids: torch.Tensor, weights: torch.Tensor, gemm1: int, gemm2: int,
           out: torch.Tensor, timed: int = 0, may_fail: bool = False) -> list[float] | None:
    """`cere_ktest_p3b_routed` into `out`; with `timed` > 0 the CUDA-graph replay spans (microseconds). With
    `may_fail` (a swept pair's first call) None when the runner cannot run the pair (CERE_ERR_INTERNAL, a failed pair
    of the sweep); any other non-zero status, or any failure of the reference or of a pair that ran once, fails."""
    us = torch.zeros(max(timed, 1), dtype=torch.float32)
    fn = oracle.ktest("cere_ktest_p3b_routed")
    args = (x, P["w13"], P["w13_scale"], P["down"], P["down_scale"], P["g1"], P["g2"], P["act"], ids, weights, out,
            x.shape[0], P["hidden"], P["num_experts"], P["top_k"], P["inter"], gemm1, gemm2,
            WARMUP if timed else 0, timed, us if timed else None)
    stream = torch.cuda.current_stream(oracle.DEVICE)
    status = fn(*[oracle._arg(a) for a in args], ctypes.c_void_p(stream.cuda_stream))
    stream.synchronize()
    if may_fail and status == CERE_ERR_INTERNAL:
        return None
    if status != 0:
        raise oracle.OracleError(f"cere_ktest_p3b_routed returned status {status}")
    return us.tolist()[:timed]


class P3bTacticSweep(unittest.TestCase):
    def test_sweep(self) -> None:
        top_k = oracle.text_config()["num_experts_per_tok"]
        g1_list, g2_list = p3b.tactic_configs()
        for gemm, tactics in ((1, g1_list), (2, g2_list)):
            for i, (tm, tn, tk, swap, fin) in enumerate(tactics):
                oracle.report_row(ROW, f"tactic gemm{gemm} {i}", phase="report", gemm=gemm, index=i, tile_m=tm,
                                  tile_n=tn, tile_k_bytes=tk, swap_ab=bool(swap), finalize=bool(fin))
        self.assertEqual([t for t in g2_list if t[4]], [], f"{ROW}: a FINALIZE GEMM2 tactic is listed")
        ref = p3b.reference_indices(g1_list, g2_list)
        pairs = [(i, j) for i in range(len(g1_list)) for j in range(len(g2_list))]
        # (m, i, j) -> layer -> (bit_equal, repeatable, graph median, graph min); None = the pair failed
        results: dict[tuple, dict] = {}
        problems = []
        for layer in LAYERS:
            w, s = p3b._load(layer)
            P = p3b._operands(w, s, top_k)
            for m in SWEEP_M:
                x = p3b._x(layer, m)
                block = p3b._p3b_block(x, P)
                ids, weights = block["ids"], block["weights"]
                reference = oracle.empty((m, P["hidden"]), torch.bfloat16)
                routed(x, P, ids, weights, *ref, reference)
                for i, j in pairs:
                    first = oracle.empty((m, P["hidden"]), torch.bfloat16)
                    if routed(x, P, ids, weights, i, j, first, may_fail=True) is None:
                        results.setdefault((m, i, j), {})[layer] = None
                        continue
                    repeatable = True
                    for _ in range(REPEATS - 1):
                        again = oracle.empty((m, P["hidden"]), torch.bfloat16)
                        routed(x, P, ids, weights, i, j, again)
                        repeatable = repeatable and p3b._bit_equal(again, first)
                    graph = oracle.empty((m, P["hidden"]), torch.bfloat16)
                    spans = routed(x, P, ids, weights, i, j, graph, timed=TIMED)
                    repeatable = repeatable and p3b._bit_equal(graph, first)
                    results.setdefault((m, i, j), {})[layer] = (p3b._bit_equal(first, reference), repeatable,
                                                                statistics.median(spans), min(spans))
                    if (i, j) == ref and not repeatable:
                        problems.append(f"reference not repeatable L{layer} M{m}")
                del x, block, reference
            del w, s, P
            torch.cuda.empty_cache()
        for m in SWEEP_M:
            failing = 0
            fastest = None
            for i, j in pairs:
                per = results[(m, i, j)]
                if any(per[layer] is None for layer in LAYERS):
                    failing += 1
                    if (i, j) == ref:
                        problems.append(f"reference fails at M{m}")
                    continue
                equal = sum(per[layer][0] for layer in LAYERS)
                repeat = sum(per[layer][1] for layer in LAYERS)
                total = sum(per[layer][2] for layer in LAYERS)
                fields = {f"graph_us_median_l{layer}": per[layer][2] for layer in LAYERS}
                oracle.report_row(ROW, f"pair M{m} g1 {i} g2 {j}", phase="sweep", m=m, gemm1=i, gemm2=j,
                                  layers=len(LAYERS), bit_equal_layers=equal, repeatable_layers=repeat,
                                  graph_us_median_sum=total,
                                  graph_us_min_sum=sum(per[layer][3] for layer in LAYERS), **fields)
                if equal == len(LAYERS) and repeat == len(LAYERS) and (fastest is None or total < fastest[0]):
                    fastest = (total, i, j)
            oracle.report_row(ROW, f"failing M{m}", phase="sweep", m=m, pairs=len(pairs), failing_pairs=failing)
            if failing == len(pairs):
                problems.append(f"no pair works at M{m}")
            if fastest is not None and all(results[(m, *ref)][layer] is not None for layer in LAYERS):
                reference_sum = sum(results[(m, *ref)][layer][2] for layer in LAYERS)
                oracle.report_row(ROW, f"fastest M{m}", phase="sweep", m=m, gemm1=fastest[1], gemm2=fastest[2],
                                  graph_us_median_sum=fastest[0], reference_us_median_sum=reference_sum,
                                  speedup=reference_sum / fastest[0])
        self.assertEqual(problems, [], f"{ROW}: {problems}")


if __name__ == "__main__":
    unittest.main()
