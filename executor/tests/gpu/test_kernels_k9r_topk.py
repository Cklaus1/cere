"""K9T (track LongK9, tasks/long-k9-spec.md LK9-01): K9's cluster top-512 (`topk_cluster_kernel`: the selection of
the decode, verify and MTP draft calls of up to 16 rows on 8-block clusters per row) against the one-block-per-row
`topk_kernel` on the same logits tile.

Suite `kernels_k9r` (synthetic inputs, no checkpoint): `native_dev.py --gpu 1 test --suite kernels_k9r`.

- test_bit_identical: impl 2 (cluster) and impl 0 (qsa_select's choice) return the bytes of impl 1 (one block per
  row) on logits rows of five kinds: the indexer's shape (sums of 4 ReLUs over sqrt(128): non-negative, about 1 in 16
  exactly zero), heavy ties (3 distinct values: thousands of blocks share the 512th key), mostly zeros (the 512th
  key is 0 and ties fill the rest), all equal, and signed values with -0/+0, denormals, +-inf and both NaN signs.
  Visible blocks from 1 to 65,536 (the cluster's capacity, 262,144 tokens), calls of 1, 4, 16 and 17 rows at the
  default positions (seq_len - rows + t: every tail 0..3; calls of at most seq_len rows) and of 16 and 17 rows at
  scattered explicit positions (some rows with at most 512 visible blocks; 17 rows: impl 0 runs the prefill kernel
  `topk_prefill_kernel` up to 32,768 blocks, `topk_kernel` past them). Masked logits (blocks past a row's visible
  ones) are +inf, so a kernel that read them would differ. Above the capacity impl 2 refuses (CERE_ERR_ARGUMENT) and
  impl 0 keeps the one-block kernel; a 17-row call at 32,000 blocks runs the prefill kernel.
- test_speed: warm-L2 graph replays (in the model the logits kernel has just written the tile) of a 4-row call (the
  window-4 verify) at 18K, 64K and 128K context, impl 1 vs impl 2; at 128K the cluster median is at most
  SPEEDUP_BOUND of the one-block median (pre-registered); report rows give every median and minimum.

C prototype (libcere_ktest.so; executor/cuda/src/ktest/k9_ktest.cpp):

    int cere_ktest_qsa_topk(const void* logits, int64_t ld, const void* positions_logical, int64_t rows,
        int64_t seq_len, int64_t impl, void* indices, int64_t iters, double* us, cudaStream_t stream);
"""
from __future__ import annotations

import unittest

import torch

import oracle

ROW = "K9T"
RATIO = 4  # k9::kRatio
ROW_WIDTH = 2051  # k9::kRowWidth
CAPACITY = 65_536  # 8 blocks x 512 threads x 16 visible blocks per thread
SCORE_DIVISOR = 128 ** 0.5  # k9::kScoreDivisor
VISIBLE = (1, 300, 512, 513, 600, 2047, 4097, 4500, 16_000, 32_000, 32_768, 50_000, 65_535, CAPACITY)
KINDS = ("indexer", "ties", "zeros", "equal", "special")
CALL_ROWS = (1, 4, 16, 17)  # 17: qsa_select's prefill choice
SPEED_CONTEXTS = (("18K", 18_000), ("64K", 64_000), ("128K", 128_000))
SPEED_ROWS = 4
SPEED_ITERS = 50
SPEEDUP_BOUND = 0.5  # LK9-01: cluster median <= 0.5 x topk_kernel's at 128K


def _positions(rows: int, seq_len: int, explicit: bool, seed: int) -> torch.Tensor:
    """Row t's logical position: seq_len - rows + t (the null default), or scattered over [0, seq_len) with the
    last row at seq_len - 1."""
    if not explicit:
        return torch.arange(seq_len - rows, seq_len, dtype=torch.int64, device=oracle.DEVICE)
    g = oracle.generator(seed)
    pos = torch.randint(0, seq_len, (rows,), generator=g, device=oracle.DEVICE, dtype=torch.int64)
    pos[0] = min(seq_len - 1, 4 * 300 + 2)  # at most 512 visible blocks
    pos[-1] = seq_len - 1
    return pos.contiguous()


def _logits(kind: str, rows: int, ld: int, positions: torch.Tensor, seed: int) -> torch.Tensor:
    g = oracle.generator(seed)
    shape = (rows, ld)
    if kind == "indexer":
        x = torch.randn((rows, ld, 4), generator=g, device=oracle.DEVICE).clamp_min(0.0).sum(dim=-1) / SCORE_DIVISOR
    elif kind == "ties":
        x = torch.randint(0, 3, shape, generator=g, device=oracle.DEVICE).float()
    elif kind == "zeros":
        x = torch.zeros(shape, device=oracle.DEVICE)
        hot = torch.rand(shape, generator=g, device=oracle.DEVICE) < 0.004
        x[hot] = torch.rand(shape, generator=g, device=oracle.DEVICE)[hot] + 0.5
    elif kind == "equal":
        x = torch.ones(shape, device=oracle.DEVICE)
    else:
        x = torch.randn(shape, generator=g, device=oracle.DEVICE)
        pick = torch.randint(0, 64, shape, generator=g, device=oracle.DEVICE)
        bits = x.view(torch.int32)
        bits[pick == 0] = 0x00000000  # +0
        bits[pick == 1] = -0x80000000  # -0
        bits[pick == 2] = 0x7F800000  # +inf
        bits[pick == 3] = -0x00800000  # -inf (0xFF800000)
        bits[pick == 4] = 0x7FC00000  # +NaN
        bits[pick == 5] = -0x00400000  # -NaN (0xFFC00000)
        bits[pick == 6] = 0x00000001  # smallest denormal
        bits[pick == 7] = -0x7FFFFFFF  # -smallest denormal (0x80000001)
    visible = torch.div(positions + 1, RATIO, rounding_mode="floor")
    masked = torch.arange(ld, device=oracle.DEVICE).view(1, -1) >= visible.view(-1, 1)
    x = x.float().contiguous()
    x[masked] = float("inf")  # never read: a kernel that read them would select them
    return x


def _topk(logits: torch.Tensor, positions: torch.Tensor | None, rows: int, seq_len: int, impl: int,
          iters: int = 0) -> tuple[int, torch.Tensor, float, float]:
    out = oracle.empty((rows, ROW_WIDTH), torch.int32)
    us = torch.zeros((2,), dtype=torch.float64)
    rc = oracle.call_status("cere_ktest_qsa_topk", logits, logits.shape[1], positions, rows, seq_len, impl, out,
                            iters, us if iters else None)
    return rc, out, float(us[0]), float(us[1])


class K9ClusterTopk(unittest.TestCase):
    def _same(self, case: str, logits: torch.Tensor, positions: torch.Tensor | None, rows: int,
              seq_len: int) -> None:
        rc, reference, _, _ = _topk(logits, positions, rows, seq_len, 1)
        self.assertEqual(rc, 0, f"{ROW} {case}: one-block status {rc}")
        for impl in (2, 0):
            rc, out, _, _ = _topk(logits, positions, rows, seq_len, impl)
            self.assertEqual(rc, 0, f"{ROW} {case}/impl{impl}: status {rc}")
            oracle.check_bit_exact(self, ROW, f"{case}/impl{impl}", out, reference)

    def test_bit_identical(self) -> None:
        seed = 9700
        for nv in VISIBLE:
            seq_len = RATIO * nv + 3
            for kind in KINDS:
                # default positions need rows <= seq_len (qsa_select's argument check); explicit ones do not
                calls = [(r, False) for r in CALL_ROWS if r <= seq_len] + [(16, True), (17, True)]
                for rows, explicit in calls:
                    seed += 1
                    positions = _positions(rows, seq_len, explicit, seed)
                    logits = _logits(kind, rows, nv, positions, seed)
                    case = f"identity/{kind}/nv{nv}/r{rows}{'/explicit' if explicit else ''}"
                    with self.subTest(case=case):
                        self._same(case, logits, positions if explicit else None, rows, seq_len)

    def test_one_block_beyond_the_cluster(self) -> None:
        nv = CAPACITY + 1
        seq_len = RATIO * nv + 3
        positions = _positions(4, seq_len, False, 9690)
        logits = _logits("indexer", 4, nv, positions, 9691)
        rc, _, _, _ = _topk(logits, None, 4, seq_len, 2)
        oracle.check_count(self, ROW, f"capacity/nv{nv}/impl2_status", rc, expected=oracle.CERE_ERR_ARGUMENT)
        rc, reference, _, _ = _topk(logits, None, 4, seq_len, 1)
        self.assertEqual(rc, 0)
        rc, out, _, _ = _topk(logits, None, 4, seq_len, 0)
        self.assertEqual(rc, 0)
        oracle.check_bit_exact(self, ROW, f"capacity/nv{nv}/impl0", out, reference)
        seq_len = RATIO * 32_000 + 3
        positions = _positions(17, seq_len, False, 9692)
        logits = _logits("indexer", 17, 32_000, positions, 9693)
        self._same("prefill/nv32000/r17", logits, None, 17, seq_len)

    def test_position_past_the_row_is_held(self) -> None:
        """A device position whose block count exceeds the logits row (ld) must not read beyond the row: with
        +inf after the row (selectable if read) it returns the bytes of the row's own last position, on the
        one-block and the cluster kernels."""
        for nv in (2048, 600):
            seq_len = RATIO * nv + 3
            valid = torch.tensor([seq_len - 1], dtype=torch.int64, device=oracle.DEVICE)
            row = _logits("indexer", 1, nv, valid, 9710 + nv)
            padded = torch.full((nv + 40_000,), float("inf"), dtype=torch.float32, device=oracle.DEVICE)
            padded[:nv] = row.view(-1)
            logits = padded[:nv].view(1, nv)
            past = torch.tensor([RATIO * (nv + 30_000)], dtype=torch.int64, device=oracle.DEVICE)
            for impl in (1, 2):
                with self.subTest(nv=nv, impl=impl):
                    rc, reference, _, _ = _topk(logits, valid, 1, seq_len, impl)
                    self.assertEqual(rc, 0)
                    rc, held, _, _ = _topk(logits, past, 1, seq_len, impl)
                    self.assertEqual(rc, 0)
                    oracle.check_bit_exact(self, ROW, f"held/nv{nv}/impl{impl}", held, reference)

    def test_speed(self) -> None:
        times = {}
        for label, seq_len in SPEED_CONTEXTS:
            ld = seq_len // RATIO
            positions = _positions(SPEED_ROWS, seq_len, False, 9680)
            logits = _logits("indexer", SPEED_ROWS, ld, positions, 9681)
            for impl in (1, 2):
                rc, _, med, low = _topk(logits, None, SPEED_ROWS, seq_len, impl, iters=SPEED_ITERS)
                self.assertEqual(rc, 0, f"{ROW} time/{label}/impl{impl}: status {rc}")
                times[(label, impl)] = med
                oracle.report_row(ROW, f"time/{label}/r{SPEED_ROWS}/impl{impl}", phase="time", rows=SPEED_ROWS,
                                  seq_len=seq_len, impl=impl, median_us=med, min_us=low)
        ratio = times[("128K", 2)] / times[("128K", 1)]
        oracle.report_row(ROW, f"time/128K/r{SPEED_ROWS}/ratio", phase="result", ratio=ratio, bound=SPEEDUP_BOUND,
                          passed=ratio <= SPEEDUP_BOUND)
        self.assertTrue(ratio <= SPEEDUP_BOUND, f"cluster median {times[('128K', 2)]:.1f} us vs one-block "
                        f"{times[('128K', 1)]:.1f} us (ratio {ratio:.3f}, bound {SPEEDUP_BOUND})")


if __name__ == "__main__":
    unittest.main()
