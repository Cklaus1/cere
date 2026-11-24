"""SelfTune ST6 S2 (tasks/selftune-spec.md ST6, D-ST6-9): cold-L2 timing of family sdar's kernels and GEMMs at
DICE-8B's decode shapes, and the cuBLASLt bar a GEMM candidate must beat. Report-only: no timing is a verdict.

Part of suite `sdar_kernels` (GPU0-only; `CERE_LIVE=1 python3 scripts/live/native_dev.py test --suite sdar_kernels`,
the dice-8b library). Synthetic inputs only. Every row is "SDAR_TIMING"; times are the median and minimum µs of
ITERS calls, each after an L2 flush (ktest/l2_flush.h cold_us).

- kernels: each of the nine kernels at a decode forward's shape (4 rows, one schedule block); attention at the S2
  context (180 keys) and the long-context workloads' (8K and 32K: scripts/tune/dice_longctx.py).
- GEMMs: each sdar_model.cpp plan (qo, kv, gate_up, down at 4 rows; lm_head at 4 rows, FP32 out) through
  cere::sdar::Gemm (the model's pick), then cuBLASLt's heuristic candidates with no workspace and with up to
  WORKSPACE bytes. `bar_us` = the fastest candidate's median: the time a GEMM candidate must beat (D-ST6-9
  condition 4); `heuristic_us` = candidate 0 without workspace, the algorithm the model runs.
- refusals: iters < 1, a refusal of the wrapped entry point, and a bad top_n are CERE_ERR_ARGUMENT.

Entry points (libcere_ktest.so, executor/cuda/src/ktest/sdar/sdar_timing_ktest.cpp; host `us` double [2] = median,
minimum µs; the stream last; 0 on success). Each `_cold` takes its sdar_ktest.cu entry point's arguments (attention
without `mask`: block-causal; top_tie without `fp32`), then `int64_t iters, void *us`:
    int cere_ktest_sdar_{embed,rmsnorm,rope,kv_store,attention,silu_mul,add,top_tie,unmask_select}_cold(...);
    int cere_ktest_sdar_gemm_cold(const void *x, const void *w, int64_t m, int64_t n, int64_t k, void *y,
                                  int64_t f32, int64_t iters, void *us, cudaStream_t stream);
    int cere_ktest_sdar_gemm_algos(const void *x, const void *w, int64_t m, int64_t n, int64_t k, void *y,
                                   int64_t f32, int64_t workspace_bytes, int64_t top_n, int64_t iters,
                                   void *us /*double [top_n, 2], -1 refused*/, void *found /*int64 [1]*/,
                                   void *ws /*int64 [top_n]*/, cudaStream_t stream);
"""
from __future__ import annotations

import unittest

import torch

import oracle

ROW = "SDAR_TIMING"
ITERS = 20
ROWS = 4  # one schedule block: a decode forward's rows
HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
Q_ROW, KV_ROW = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
HIDDEN, INTER, VOCAB = 4096, 12288, 151936
EPS, THETA = 1e-6, 1_000_000.0
CONTEXTS = {"s2": 180, "8k": 8192, "32k": 32768}  # keys a decode forward reads (past + ROWS)
GEMMS = {"qo": (HIDDEN, HIDDEN, 0), "kv": (KV_ROW, HIDDEN, 0), "gate_up": (2 * INTER, HIDDEN, 0),
         "down": (HIDDEN, INTER, 0), "lm_head": (VOCAB, HIDDEN, 1)}  # n, k, f32 (sdar_model.cpp plans)
TOP_N = 16
WORKSPACE = 32 << 20


def _randn(shape, seed: int, dtype=torch.bfloat16) -> torch.Tensor:
    g = oracle.generator(seed)
    return (torch.randn(shape, generator=g, device=oracle.DEVICE, dtype=torch.float32) * 0.5).to(dtype)


def cold(name: str, *args) -> tuple[float, float]:
    """(median, minimum) µs of `name` (an sdar `_cold` entry point) at ITERS; a non-zero status fails."""
    us = torch.zeros((2,), dtype=torch.float64)
    return timed(name, oracle.call_status(name, *args, ITERS, us), us)


def timed(name: str, rc: int, us: torch.Tensor) -> tuple[float, float]:
    """(median, minimum) µs a `_cold` entry point wrote into `us`; a non-zero status fails."""
    if rc != 0:
        raise oracle.OracleError(f"{name} returned status {rc}")
    return float(us[0]), float(us[1])


class SdarTiming(unittest.TestCase):
    def _report(self, case: str, us: tuple[float, float], **fields) -> None:
        self.assertTrue(0.0 < us[1] <= us[0], f"{case}: median {us[0]} µs, minimum {us[1]} µs")
        oracle.report_row(ROW, case, phase="time", us=us[0], min_us=us[1], iters=ITERS, **fields)

    def test_kernels_at_decode_shapes(self):
        x = _randn((ROWS, HIDDEN), 1)
        w = _randn((HIDDEN,), 2)
        y = oracle.empty((ROWS, HIDDEN), torch.bfloat16)
        table = _randn((VOCAB, HIDDEN), 3)
        ids = torch.arange(ROWS, dtype=torch.int32, device=oracle.DEVICE) * 7919 % VOCAB
        self._report("embed", cold("cere_ktest_sdar_embed_cold", table, ids, ROWS, HIDDEN, y), rows=ROWS)
        self._report("rmsnorm/hidden", cold("cere_ktest_sdar_rmsnorm_cold", x, HIDDEN, w, EPS, ROWS, HIDDEN, y,
                                            HIDDEN), rows=ROWS)
        q = _randn((ROWS, Q_ROW), 4)
        hw = _randn((HEAD_DIM,), 5)
        self._report("rmsnorm/q_heads", cold("cere_ktest_sdar_rmsnorm_cold", q, HEAD_DIM, hw, EPS, ROWS * HEADS,
                                             HEAD_DIM, q, HEAD_DIM), rows=ROWS * HEADS)
        self._report("rope/q", cold("cere_ktest_sdar_rope_cold", q, Q_ROW, ROWS, HEADS, 176, THETA), rows=ROWS)
        gate_up = _randn((ROWS, 2 * INTER), 6)
        act = oracle.empty((ROWS, INTER), torch.bfloat16)
        self._report("silu_mul", cold("cere_ktest_sdar_silu_mul_cold", gate_up, ROWS, INTER, act), rows=ROWS)
        self._report("add", cold("cere_ktest_sdar_add_cold", y, x, ROWS * HIDDEN), rows=ROWS)
        logits = _randn((ROWS, VOCAB), 7, torch.float32)
        top = torch.zeros((ROWS, 5), dtype=torch.int32, device=oracle.DEVICE)
        self._report("top_tie", cold("cere_ktest_sdar_top_tie_cold", logits, VOCAB, ROWS, VOCAB, top), rows=ROWS)
        oracle.call("cere_ktest_sdar_top_tie", logits, VOCAB, ROWS, VOCAB, top, 0)
        masked = torch.ones((ROWS,), dtype=torch.uint8, device=oracle.DEVICE)
        index = torch.zeros((1,), dtype=torch.int32, device=oracle.DEVICE)
        self._report("unmask_select", cold("cere_ktest_sdar_unmask_select_cold", top, masked, ROWS, index), rows=ROWS)

    def test_attention_and_kv_store_per_context(self):
        for workload, keys in CONTEXTS.items():
            with self.subTest(workload=workload):
                past = keys - ROWS
                k_cache = _randn((keys, KV_ROW), 10)
                v_cache = _randn((keys, KV_ROW), 11)
                q = _randn((ROWS, Q_ROW), 12)
                new = _randn((ROWS, KV_ROW), 13)
                out = oracle.empty((ROWS, Q_ROW), torch.bfloat16)
                us = torch.zeros((2,), dtype=torch.float64)
                rc = oracle.call_status("cere_ktest_sdar_kv_store_cold", new, KV_ROW, new, KV_ROW, ROWS, KV_HEADS,
                                        k_cache, v_cache, past, ITERS, us)
                self._report(f"kv_store/{workload}", timed("kv_store", rc, us), keys=keys)
                kv_bytes = 2 * keys * KV_ROW * 2
                us = torch.zeros((2,), dtype=torch.float64)
                rc = oracle.call_status("cere_ktest_sdar_attention_cold", q, Q_ROW, k_cache, v_cache, past, new,
                                        KV_ROW, new, KV_ROW, ROWS, HEADS, KV_HEADS, out, Q_ROW, ITERS, us)
                self._report(f"attention/{workload}", timed("attention", rc, us), keys=keys, kv_bytes=kv_bytes)

    def test_gemms_against_the_cublaslt_bar(self):
        for name, (n, k, f32) in GEMMS.items():
            with self.subTest(gemm=name):
                x = _randn((ROWS, k), 20)
                w = _randn((n, k), 21)
                y = oracle.empty((ROWS, n), torch.float32 if f32 else torch.bfloat16)
                plan = cold("cere_ktest_sdar_gemm_cold", x, w, ROWS, n, k, y, f32)
                candidates = []
                for limit in (0, WORKSPACE):
                    us = torch.zeros((TOP_N, 2), dtype=torch.float64)
                    found = torch.zeros((1,), dtype=torch.int64)
                    ws = torch.zeros((TOP_N,), dtype=torch.int64)
                    rc = oracle.call_status("cere_ktest_sdar_gemm_algos", x, w, ROWS, n, k, y, f32, limit, TOP_N,
                                            ITERS, us, found, ws)
                    self.assertEqual(rc, 0, f"{name} workspace {limit}")
                    self.assertTrue(int(found[0]) >= 1, f"{name}: cuBLASLt offers no algorithm")
                    candidates += [(float(us[i, 0]), float(us[i, 1]), limit, i, int(ws[i]))
                                   for i in range(int(found[0]))]
                timed = [c for c in candidates if c[0] > 0]
                self.assertTrue(timed, f"{name}: every candidate refused")
                heuristic = candidates[0]
                self.assertEqual(heuristic[2:4], (0, 0))
                bar = min(timed)
                self._report(f"gemm/{name}", plan, m=ROWS, n=n, k=k, f32=f32, weight_bytes=2 * n * k,
                             heuristic_us=heuristic[0], bar_us=bar[0], bar_min_us=bar[1],
                             bar_workspace_limit=bar[2], bar_candidate=bar[3], bar_workspace=bar[4],
                             candidates=len(candidates), refused=len(candidates) - len(timed))

    def test_refusals(self):
        x = _randn((ROWS, HIDDEN), 30)
        us = torch.zeros((2,), dtype=torch.float64)
        self.assertEqual(oracle.call_status("cere_ktest_sdar_add_cold", x, x, ROWS * HIDDEN, 0, us),
                         oracle.CERE_ERR_ARGUMENT)
        self.assertEqual(oracle.call_status("cere_ktest_sdar_rope_cold", x, HIDDEN, ROWS, -1, 0, THETA, 2, us),
                         oracle.CERE_ERR_ARGUMENT)
        w = _randn((HIDDEN, HIDDEN), 31)
        y = oracle.empty((ROWS, HIDDEN), torch.bfloat16)
        found = torch.zeros((1,), dtype=torch.int64)
        ws = torch.zeros((1,), dtype=torch.int64)
        self.assertEqual(oracle.call_status("cere_ktest_sdar_gemm_algos", x, w, ROWS, HIDDEN, HIDDEN, y, 0, 0, 0, 2,
                                            us, found, ws), oracle.CERE_ERR_ARGUMENT)
        self.assertEqual(oracle.call_status("cere_ktest_sdar_gemm_cold", x, w, 0, HIDDEN, HIDDEN, y, 0, 2, us),
                         oracle.CERE_ERR_ARGUMENT)


if __name__ == "__main__":
    unittest.main()
