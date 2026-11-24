"""K9D (track LongK9, tasks/long-k9-spec.md LK9-02): K9's decode-call kernels, the block logits and the sparse GQA of
the calls of up to 16 rows (decode, verify, the MTP draft), against the kernels they replace on the same call.

Suite `kernels_k9r` (synthetic inputs, no checkpoint): `native_dev.py --gpu 1 test --suite kernels_k9r`.

- test_logits_bit_identical: `logits_small_kernel` (qsa_logits impl 2) and qsa_select's choice (impl 0:
  `logits_prefill_kernel` above 16 rows) return the bytes of `logits_kernel` (impl 1) over the whole
  [rows, seq_len / 4] tile: calls of 1, 3, 4, 5, 16, 17 and 77 rows at the default positions (seq_len - rows + t:
  those of at most seq_len rows) and at scattered explicit positions (rows with no, one or 31 visible blocks, a
  128-key block partly visible), contexts from 7 to 70,001 tokens, with and without a slot permutation of the
  compressed keys.
- test_attention_bit_identical: `split_attention_kernel` (cere_ktest_qsa_attention_heads rows_heads 12, the default)
  returns the bytes of `attention_kernel` (rows_heads 0) on the FP8 cache, through a slot permutation whose other rows
  hold the E4M3 NaN byte and without slots, for calls of 1, 2, 3, 4, 7 and 16 rows. Rows cycle through valid-entry
  counts 2,051, 2,048, 2,049, 2,050, 1, 31, 32, 33, 160, 161, 1,000 and 0 (the splits' 5-, 4- and 1-tile shapes and
  empty splits), then the layouts of test_kernels_k9r_rows (-1 holes, one valid entry first or last).
- test_cold_l2_speed: 4-row calls (the window-4 verify) at 18K, 64K and 128K context, cold L2 (ktest/l2_flush.h,
  lesson O97): logits impl 1 vs impl 2 and attention rows_heads 0 vs 12 over that many FP8 cache rows; at 128K each
  new median is at most SPEEDUP_BOUND of the old one (pre-registered); report rows give every median and minimum.

C prototypes (libcere_ktest.so; executor/cuda/src/ktest/k9_ktest.cpp):

    int cere_ktest_qsa_logits(const void* q_idx, const void* keys, const void* slots, const void* positions_logical,
        int64_t rows, int64_t seq_len, int64_t impl, void* logits, int64_t iters, double* us, cudaStream_t stream);
    int cere_ktest_qsa_attention_heads(const void* q, int64_t rows, const void* indices, const void* k,
        const void* v, const void* k_cache, const void* v_cache, const void* slots, float scale, void* out,
        int64_t rows_heads, int64_t iters, double* us, cudaStream_t stream);
"""
from __future__ import annotations

import unittest

import torch

import oracle
from test_kernels_k9r_rows import _blocks, _call, _fp8_cache, _indices, _kv, _q

ROW = "K9D"
RATIO = 4  # k9::kRatio
INDEX_HEADS, INDEX_DIM = 4, 128
ROW_WIDTH = 2051  # k9::kRowWidth
LOGIT_CONTEXTS = (7, 131, 4_503, 18_003, 70_001)
LOGIT_ROWS = (1, 3, 4, 5, 16, 17, 77)  # 17, 77: the prefill kernel (77: three 32-row blocks, the last partial)
ATTN_ROWS = (1, 2, 3, 4, 7, 16)
VALID_COUNTS = (2051, 2048, 2049, 2050, 1, 31, 32, 33, 160, 161, 1000, 0)
ATTN_CACHE = 4500
DEFAULT_HEADS = 12  # k9::kRowsHeads: the split kernel on the FP8 cache
SPEED_CONTEXTS = (("18K", 18_000), ("64K", 64_000), ("128K", 128_000))
SPEED_ROWS = 4
COLD_ITERS = 21  # 1 untimed + 20 cold-L2 calls
SPEEDUP_BOUND = 0.5  # LK9-02: new median <= 0.5 x the old kernel's at 128K, logits and attention each
BF16 = torch.bfloat16


def _index_inputs(rows: int, seq_len: int, slotted: bool, seed: int):
    """(q_idx [rows, 4, 128], keys [pool, 128], slots [seq_len] or None): index q of mixed sign (some heads' dots
    negative: the ReLU), compressed keys at a permutation of a 2x pool when slotted (slots[4 b + i] = 4 perm[b] + i)."""
    g = oracle.generator(seed)
    q_idx = torch.randn((rows, INDEX_HEADS, INDEX_DIM), generator=g, device=oracle.DEVICE).to(BF16).contiguous()
    nb = seq_len // RATIO
    pool = 2 * nb + 1 if slotted else nb + 1
    keys = torch.randn((pool, INDEX_DIM), generator=g, device=oracle.DEVICE).to(BF16).contiguous()
    if not slotted:
        return q_idx, keys, None
    perm = torch.randperm(pool, generator=g, device=oracle.DEVICE)[: nb + 1].to(torch.int64)
    tok = torch.arange(seq_len, device=oracle.DEVICE, dtype=torch.int64)
    slots = (perm[torch.div(tok, RATIO, rounding_mode="floor")] * RATIO + tok % RATIO).contiguous()
    return q_idx, keys, slots


def _positions(rows: int, seq_len: int, seed: int) -> torch.Tensor:
    """Scattered positions in [0, seq_len): the first rows see no block, one block, 31 blocks and part of a 128-key
    block; the last row is at seq_len - 1."""
    g = oracle.generator(seed)
    pos = torch.randint(0, seq_len, (rows,), generator=g, device=oracle.DEVICE, dtype=torch.int64)
    for t, p in enumerate((2, 4, 4 * 31 + 1, 4 * 200 + 3)[: max(rows - 1, 0)]):
        pos[t] = min(p, seq_len - 1)
    pos[-1] = seq_len - 1
    return pos.contiguous()


def _logits(q_idx, keys, slots, positions, rows: int, seq_len: int, impl: int,
            iters: int = 0) -> tuple[torch.Tensor, float, float]:
    out = oracle.empty((rows, seq_len // RATIO), torch.float32)
    us = torch.zeros((2,), dtype=torch.float64)
    oracle.call("cere_ktest_qsa_logits", q_idx, keys, slots, positions, rows, seq_len, impl, out, iters,
                us if iters else None)
    return out, float(us[0]), float(us[1])


def _counted(rows: int, n: int, seed: int) -> torch.Tensor:
    """Row r: the selection's shape with its first VALID_COUNTS[r % 12] entries valid, then -1 (the valid prefix of a
    compacted selection); rows from 12 on take test_kernels_k9r_rows' layouts (holes, one valid first or last)."""
    idx = _blocks(rows, n, seed).clone()
    mixed = _indices(rows, n, seed + 1)
    for r in range(rows):
        if r < len(VALID_COUNTS):
            idx[r, VALID_COUNTS[r]:] = -1
        else:
            idx[r] = mixed[r]
    return idx.contiguous()


class K9DecodeKernels(unittest.TestCase):
    def test_logits_bit_identical(self) -> None:
        seed = 9500
        for seq_len in LOGIT_CONTEXTS:
            for slotted in (False, True):
                for rows in LOGIT_ROWS:
                    seed += 1
                    q_idx, keys, slots = _index_inputs(rows, seq_len, slotted, seed)
                    calls = [("default", None)] if rows <= seq_len else []
                    calls.append(("explicit", _positions(rows, seq_len, seed)))
                    for kind, positions in calls:
                        case = f"logits/n{seq_len}/r{rows}/{'slots' if slotted else 'direct'}/{kind}"
                        with self.subTest(case=case):
                            reference, _, _ = _logits(q_idx, keys, slots, positions, rows, seq_len, 1)
                            for impl in (2, 0):
                                out, _, _ = _logits(q_idx, keys, slots, positions, rows, seq_len, impl)
                                oracle.check_bit_exact(self, ROW, f"{case}/impl{impl}", out, reference)

    def test_attention_bit_identical(self) -> None:
        k, v = _kv(ATTN_CACHE, seed=9600)
        k_cache, v_cache, slots = _fp8_cache(k, v, seed=9601)
        direct = (None, None, k.to(torch.float8_e4m3fn).view(torch.uint8).contiguous(),
                  v.to(torch.float8_e4m3fn).view(torch.uint8).contiguous(), None)
        sources = (("fp8_slots", (None, None, k_cache, v_cache, slots)), ("fp8_direct", direct))
        for rows in ATTN_ROWS + (24,):
            call_rows = min(rows, 16)
            q = _q(call_rows, seed=9610 + rows)
            idx = _counted(rows, ATTN_CACHE, seed=9620 + rows)[-call_rows:].contiguous()
            for name, kv_args in sources:
                reference, _, _ = _call(q, idx, kv_args, 0)
                for heads in (DEFAULT_HEADS, 3):
                    case = f"attention/{name}/r{call_rows}{'/tail' if rows != call_rows else ''}/h{heads}"
                    with self.subTest(case=case):
                        out, _, _ = _call(q, idx, kv_args, heads)
                        oracle.check_bit_exact(self, ROW, case, out, reference)

    def test_cold_l2_speed(self) -> None:
        ratios = {}
        for label, seq_len in SPEED_CONTEXTS:
            q_idx, keys, slots = _index_inputs(SPEED_ROWS, seq_len, True, seed=9640)
            times = {}
            for impl in (1, 2):
                _, med, low = _logits(q_idx, keys, slots, None, SPEED_ROWS, seq_len, impl, iters=COLD_ITERS)
                times[impl] = med
                oracle.report_row(ROW, f"time/logits/{label}/r{SPEED_ROWS}/impl{impl}", phase="time", rows=SPEED_ROWS,
                                  seq_len=seq_len, impl=impl, median_us=med, min_us=low)
            ratios[("logits", label)] = times[2] / times[1]
            k, v = _kv(seq_len, seed=9650)
            k_cache, v_cache, cache_slots = _fp8_cache(k, v, seed=9651)
            del k, v
            q = _q(SPEED_ROWS, seed=9652)
            idx = _blocks(SPEED_ROWS, seq_len, seed=9653)
            kv_args = (None, None, k_cache, v_cache, cache_slots)
            for heads in (0, DEFAULT_HEADS):
                _, med, low = _call(q, idx, kv_args, heads, iters=COLD_ITERS)
                times[("attn", heads)] = med
                oracle.report_row(ROW, f"time/attention/{label}/r{SPEED_ROWS}/h{heads}", phase="time",
                                  rows=SPEED_ROWS, seq_len=seq_len, heads=heads, median_us=med, min_us=low)
            ratios[("attention", label)] = times[("attn", DEFAULT_HEADS)] / times[("attn", 0)]
            del k_cache, v_cache
        for kernel in ("logits", "attention"):
            ratio = ratios[(kernel, "128K")]
            passed = ratio <= SPEEDUP_BOUND
            oracle.report_row(ROW, f"time/{kernel}/128K/r{SPEED_ROWS}/ratio", phase="result", ratio=ratio,
                              bound=SPEEDUP_BOUND, passed=passed)
            with self.subTest(kernel=kernel):
                self.assertTrue(passed, f"{kernel}: new/old median ratio {ratio:.3f} at 128K (bound {SPEEDUP_BOUND})")


if __name__ == "__main__":
    unittest.main()
