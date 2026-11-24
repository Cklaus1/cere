"""K9R (track PrefillTtft, tasks/prefill-ttft-spec.md PT-02): K9's rows kernel, the sparse GQA of the prefill calls
(more than 16 rows: splits == 1), against K9's own attention_kernel on the same call.

Suite `kernels_k9r` (synthetic inputs, no checkpoint): `native_dev.py --gpu 1 test --suite kernels_k9r`.

- test_bit_identical: every heads-per-warp choice of the rows kernel (3, 6, 12) returns the bytes of attention_kernel
  (`rows_heads` 0) on BF16 K/V and on the FP8 cache through a slot permutation whose other rows hold the E4M3 NaN
  byte, for calls of 17 rows (the smallest splits == 1 call), 300 rows and 4,096 rows (one prefill chunk). Rows
  cover the index layouts of a selection: all 2,051 entries valid, a valid prefix ending off a 32-key tile, -1
  holes, one valid entry first or last, none valid; scores spread from flat to peaked. Calls of 16 rows (decode,
  verify: the split path) never run the rows kernel; on the FP8 cache they run split_attention_kernel
  (tasks/long-k9-spec.md LK9-02, row K9D), so the row pins that kernel's bytes too.
- test_cold_l2_speed: one 4,096-row call (2,051 valid entries per row: 512 blocks of 4 plus the 3-key tail, the
  selection's shape at 18K) over an 18,000-row FP8 cache, cold L2 (ktest/l2_flush.h, lesson O97): the default
  (k9::kRowsHeads) median is at most SPEEDUP_BOUND of attention_kernel's (pre-registered, PT-02); report rows give
  every choice's median and minimum us.

C prototype (libcere_ktest.so; executor/cuda/src/ktest/k9_ktest.cpp):

    int cere_ktest_qsa_attention_heads(const void* q, int64_t rows, const void* indices, const void* k,
        const void* v, const void* k_cache, const void* v_cache, const void* slots, float scale, void* out,
        int64_t rows_heads, int64_t iters, double* us, cudaStream_t stream);
"""
from __future__ import annotations

import unittest

import torch

import oracle

ROW = "K9R"
Q_HEADS, KV_HEADS, HEAD_DIM = 24, 2, 256
ROW_WIDTH = 2051  # k9::kRowWidth: 512 blocks x 4 + the 3-key tail
SCALE = HEAD_DIM ** -0.5
HEADS = (3, 6, 12)  # rows kernel choices; 0 = attention_kernel
DEFAULT_HEADS = 12  # k9::kRowsHeads
CALL_ROWS = (17, 300, 4096)
SMALL_ROWS = 16  # k9::kSplitMaxRows: the split path, never the rows kernel
CACHE_ROWS = 18_000
FP8_NAN = 0x7F
COLD_ITERS = 21  # 1 untimed + 20 cold-L2 calls
SPEEDUP_BOUND = 0.5  # PT-02: default median <= 0.5 x attention_kernel's
BF16 = torch.bfloat16


def _gen(seed: int) -> torch.Generator:
    return oracle.generator(seed)


def _q(rows: int, seed: int) -> torch.Tensor:
    """Query rows whose heads scale from flat (0.25) to peaked (3.0) scores."""
    g = _gen(seed)
    scale = torch.linspace(0.25, 3.0, Q_HEADS, device=oracle.DEVICE).repeat_interleave(HEAD_DIM)
    q = torch.randn((rows, Q_HEADS * HEAD_DIM), generator=g, device=oracle.DEVICE) * scale
    return q.to(BF16).contiguous()


def _kv(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = _gen(seed)
    k = torch.randn((n, KV_HEADS * HEAD_DIM), generator=g, device=oracle.DEVICE) * 1.5
    v = torch.randn((n, KV_HEADS * HEAD_DIM), generator=g, device=oracle.DEVICE)
    return k.to(BF16).contiguous(), v.to(BF16).contiguous()


def _blocks(rows: int, n: int, seed: int) -> torch.Tensor:
    """The selection's shape: per row 512 sorted blocks of 4 among n // 4, expanded, then 3 tail keys."""
    g = _gen(seed)
    nb = n // 4
    picks = torch.argsort(torch.rand((rows, nb), generator=g, device=oracle.DEVICE), dim=1)[:, :512]
    picks = picks.sort(dim=1).values * 4
    pos = (picks.unsqueeze(-1) + torch.arange(4, device=oracle.DEVICE)).reshape(rows, 2048)
    tail = torch.full((rows, 3), n - 1, dtype=pos.dtype, device=oracle.DEVICE) - torch.arange(
        3, device=oracle.DEVICE)
    return torch.cat([pos, tail], dim=1).to(torch.int32).contiguous()


def _indices(rows: int, n: int, seed: int) -> torch.Tensor:
    """Row r's layout by r % 6: all valid; a valid prefix ending off a tile; -1 holes; one valid first; one valid
    last; none valid."""
    g = _gen(seed)
    idx = _blocks(rows, n, seed + 1).clone()
    for r in range(rows):
        kind = r % 6
        if kind == 1:
            end = int(torch.randint(1, ROW_WIDTH, (1,), generator=g, device=oracle.DEVICE))
            end -= (end % 32 == 0)
            idx[r, end:] = -1
        elif kind == 2:
            holes = torch.rand((ROW_WIDTH,), generator=g, device=oracle.DEVICE) < 0.3
            idx[r][holes] = -1
        elif kind == 3:
            idx[r, 1:] = -1
        elif kind == 4:
            idx[r, :-1] = -1
        elif kind == 5:
            idx[r] = -1
    return idx.contiguous()


def _fp8_cache(k: torch.Tensor, v: torch.Tensor, seed: int):
    """(k_cache, v_cache, slots): the E4M3 rows at a permutation of 2x the rows, every other row the NaN byte."""
    n = k.shape[0]
    g = _gen(seed)
    slots = torch.randperm(2 * n, generator=g, device=oracle.DEVICE)[:n].to(torch.int64).contiguous()
    width = KV_HEADS * HEAD_DIM
    k_cache = torch.full((2 * n, width), FP8_NAN, dtype=torch.uint8, device=oracle.DEVICE)
    v_cache = torch.full_like(k_cache, FP8_NAN)
    k_cache[slots] = k.to(torch.float8_e4m3fn).view(torch.uint8)
    v_cache[slots] = v.to(torch.float8_e4m3fn).view(torch.uint8)
    return k_cache, v_cache, slots


def _call(q, idx, kv_args: tuple, heads: int, iters: int = 0) -> tuple[torch.Tensor, float, float]:
    rows = q.shape[0]
    k, v, k_cache, v_cache, slots = kv_args
    out = oracle.empty((rows, Q_HEADS * HEAD_DIM), BF16)
    us = torch.zeros((2,), dtype=torch.float64)
    oracle.call("cere_ktest_qsa_attention_heads", q, rows, idx, k, v, k_cache, v_cache, slots, SCALE, out, heads,
                iters, us if iters else None)
    return out, float(us[0]), float(us[1])


class K9RowsKernel(unittest.TestCase):
    def test_bit_identical(self) -> None:
        n = 4500
        k, v = _kv(n, seed=9900)
        k_cache, v_cache, slots = _fp8_cache(k, v, seed=9901)
        sources = (("bf16_kv", (k, v, None, None, None)), ("fp8_cache", (None, None, k_cache, v_cache, slots)))
        for rows in CALL_ROWS + (SMALL_ROWS,):
            q = _q(rows, seed=9910 + rows)
            idx = _indices(rows, n, seed=9920 + rows)
            for name, kv_args in sources:
                reference, _, _ = _call(q, idx, kv_args, 0)
                self.assertFalse(torch.isnan(reference.float()).any().item(), f"{name}/r{rows}: NaN in reference")
                for heads in HEADS:
                    case = f"identity/{name}/r{rows}/h{heads}"
                    with self.subTest(case=case):
                        out, _, _ = _call(q, idx, kv_args, heads)
                        oracle.check_bit_exact(self, ROW, case, out, reference)

    def test_cold_l2_speed(self) -> None:
        rows = 4096
        k, v = _kv(CACHE_ROWS, seed=9930)
        k_cache, v_cache, slots = _fp8_cache(k, v, seed=9931)
        q = _q(rows, seed=9932)
        idx = _blocks(rows, CACHE_ROWS, seed=9933)
        kv_args = (None, None, k_cache, v_cache, slots)
        flops = 4.0 * rows * Q_HEADS * ROW_WIDTH * HEAD_DIM  # QK and PV FMAs, 2 FLOP each
        times = {}
        for heads in (0,) + HEADS:
            _, med, low = _call(q, idx, kv_args, heads, iters=COLD_ITERS)
            times[heads] = med
            oracle.report_row(ROW, f"time/fp8_cache/r{rows}/h{heads}", phase="time", rows=rows, heads=heads,
                              median_us=med, min_us=low, tflops=flops / med / 1e6)
        ratio = times[DEFAULT_HEADS] / times[0]
        oracle.report_row(ROW, f"time/fp8_cache/r{rows}/ratio", phase="result", ratio=ratio, bound=SPEEDUP_BOUND,
                          passed=ratio <= SPEEDUP_BOUND)
        self.assertTrue(ratio <= SPEEDUP_BOUND, f"rows kernel median {times[DEFAULT_HEADS]:.0f} us vs "
                        f"attention_kernel {times[0]:.0f} us (ratio {ratio:.3f}, bound {SPEEDUP_BOUND})")


if __name__ == "__main__":
    unittest.main()
