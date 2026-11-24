"""Inkling 2 Mini S8 (tasks/inkling-spec.md 4.1, G4): the greedy scores of family inkling2_loop
(executor/cuda/src/model/inkling/logits.h): top2_rows (the device's step score: argmax, runner-up and their logits)
and argmax_rows (the forward ktest's greedy ids) over FP32 logit rows, one cluster of blocks per row.

Suite `inkling_kernels` (GPU0 only): `CERE_LIVE=1 python3 scripts/live/native_dev.py test --suite inkling_kernels`
on the inkling-2-mini library (build/native-inkling-2-mini). Every case reports row "INKLING_LOGITS".
- exactness (bit-exact vs a CPU reference of the order: NaN beats every number and lower-id NaNs beat higher ones,
  otherwise the greater value, ties to the lower id; +0 equals -0): random rows at the forward's vocab (200,058) and
  row stride (200,064, the padding holding values that would win if read) for 1, 4 and 64 rows; planted ties, signed
  zeros, NaNs, -inf rows, winners at the first and last ids and at the blocks' boundaries; small vocabularies. The
  argmax equals the top-2's first id.
- refusals: malformed arguments return a nonzero status and write nothing.
- timing (reported only): 1 and 4 rows at the forward's vocab.

Entry point (libcere_ktest.so, executor/cuda/src/ktest/inkling/logits_ktest.cu; int64 sizes, the stream last, 0 on
success):
    int cere_ktest_inkling_top2(const float *logits, int64_t rows, int64_t vocab, int64_t ld, void *out_scores,
                                int32_t *ids, cudaStream_t stream);
  out_scores: cere_token_score [rows] (id, runner_up, logit, runner_up_logit) or null; ids: int32 [rows] or null.
"""

from __future__ import annotations

import unittest

import torch

import oracle

ROW = "INKLING_LOGITS"
VOCAB = 200058
LD = 200064
SPAN = 8 * 1024  # entries per cluster step (logits.cu: kTopCluster x kTopThreads)
CPU = torch.device("cpu")
INF = float("inf")


def _best(row: torch.Tensor, excluded: int) -> int:
    """The id of the best entry of a CPU FP32 row by the order above, `excluded` (-1: none) left out."""
    nan = torch.isnan(row)
    if excluded >= 0:
        nan[excluded] = False
    nan_ids = torch.nonzero(nan).flatten()
    if nan_ids.numel():
        return int(nan_ids[0])
    keep = ~torch.isnan(row)
    if excluded >= 0:
        keep[excluded] = False
    top = row[keep].max()
    return int(torch.nonzero(keep & (row == top)).flatten()[0])


def _ref(logits: torch.Tensor, vocab: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(ids int32 [rows, 2], logit bits int32 [rows, 2]) of every row's best two over its first `vocab` entries."""
    rows = logits.detach().to(CPU)[:, :vocab].contiguous()
    ids = torch.zeros((rows.shape[0], 2), dtype=torch.int32)
    for r in range(rows.shape[0]):
        a = _best(rows[r], -1)
        ids[r, 0] = a
        ids[r, 1] = _best(rows[r], a) if vocab >= 2 else -1
    bits = rows.view(torch.int32).gather(1, ids.clamp(min=0).to(torch.int64))
    return ids, bits


def _scores(logits: torch.Tensor, vocab: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(top-2 records int32 [rows, 4] (id, runner_up, logit bits, runner_up bits), argmax int32 [rows]) on CPU."""
    rows = logits.shape[0]
    out = oracle.empty((rows, 4), torch.int32)
    ids = oracle.empty((rows,), torch.int32)
    rc = oracle.call_status("cere_ktest_inkling_top2", logits, rows, vocab, logits.stride(0), out, ids)
    if rc != 0:
        raise oracle.OracleError(f"cere_ktest_inkling_top2 returned status {rc}")
    return out.to(CPU), ids.to(CPU)


def _rows(n: int, seed: int, *, vocab: int = VOCAB, ld: int = LD) -> torch.Tensor:
    """n random rows (30 tanh(x / 30)-like spread) with every padding entry at +3e38 (it would win if read)."""
    x = torch.randn((n, ld), generator=oracle.generator(seed), device=oracle.DEVICE) * 8.0
    x[:, vocab:] = 3e38
    return x


class InklingLogits(unittest.TestCase):
    def _check(self, case: str, logits: torch.Tensor, vocab: int = VOCAB) -> None:
        got, arg = _scores(logits, vocab)
        ids, bits = _ref(logits, vocab)
        if vocab >= 2:
            oracle.check_bit_exact(self, ROW, f"{case} top2 ids", got[:, :2].contiguous(), ids)
            oracle.check_bit_exact(self, ROW, f"{case} top2 logits", got[:, 2:].contiguous(), bits)
        # clone, not contiguous(): a 1-row column keeps stride 2 under contiguous(), which view(uint8) refuses
        oracle.check_bit_exact(self, ROW, f"{case} argmax", arg, ids[:, 0].clone(memory_format=torch.contiguous_format))

    def test_random(self):
        for n, seed in ((1, 1), (4, 2), (64, 3)):
            self._check(f"random rows={n}", _rows(n, seed))

    def test_planted(self):
        x = _rows(12, 4)
        x[0, 5] = x[0, 100001] = 100.0  # tie for first: the lower id wins, the other is the runner-up
        x[1, 77] = 100.0
        x[1, 3] = x[1, 190000] = 90.0  # tie for second
        x[2, :VOCAB] = 0.0  # all equal: ids 0 and 1
        x[3, :VOCAB] = -0.0
        x[3, 777] = 0.0  # +0 equals -0: still ids 0 and 1, with their own bits
        x[4, 9000] = x[4, 150000] = float("nan")  # NaNs first, the lower id ahead
        x[5, VOCAB - 1] = float("nan")
        x[6, :VOCAB] = -INF
        x[6, 4242] = -1e30
        x[7, :VOCAB] = -INF  # all -inf: ids 0 and 1
        x[8, VOCAB - 1] = 200.0  # the last id
        x[8, 0] = 199.0  # and the first
        for k, c in enumerate((SPAN - 1, SPAN, 4 * SPAN - 1, 4 * SPAN)):  # block and unroll-step boundaries
            x[9 + k // 2, c] = 300.0 + k
        x[11, SPAN * 24:VOCAB] = 500.0  # the scalar tail past the unrolled steps: all tied
        self._check("planted", x)

    def test_small_vocab(self):
        for vocab, ld in ((2, 2), (3, 8), (1000, 1000), (SPAN + 3, SPAN + 8), (4 * SPAN + 1, 4 * SPAN + 1)):
            self._check(f"vocab={vocab}", _rows(3, vocab, vocab=vocab, ld=ld), vocab)
        x = _rows(2, 5, vocab=1, ld=4)
        _, arg = _scores_argmax_only(x, 1)
        oracle.check_bit_exact(self, ROW, "vocab=1 argmax", arg, torch.zeros(2, dtype=torch.int32))

    def test_refusals(self):
        x = _rows(2, 6, vocab=16, ld=16)
        big = torch.zeros((65536, 2), device=oracle.DEVICE)
        cases = (("rows 0", x, 0, 16, 16, True), ("rows 65536", big, 65536, 2, 2, True),
                 ("ld < vocab", x, 2, 16, 15, True), ("vocab 0", x, 2, 0, 16, True),
                 ("top2 vocab 1", x, 2, 1, 16, False), ("null logits", None, 2, 16, 16, True))
        for name, t, rows, vocab, ld, argmax in cases:
            out = oracle.empty((2, 4), torch.int32)
            ids = oracle.empty((2,), torch.int32)
            rc_top2 = oracle.call_status("cere_ktest_inkling_top2", t, rows, vocab, ld, out, None)
            rc_arg = oracle.call_status("cere_ktest_inkling_top2", t, rows, vocab, ld, None, ids) if argmax else -1
            wrote = int((out != -1).sum().item()) + int((ids != -1).sum().item())
            oracle.check_count(self, ROW, f"refusal {name}", int(rc_top2 != 0) + int(rc_arg != 0) + wrote,
                               expected=2)

    def test_time(self):
        stream = torch.cuda.current_stream(oracle.DEVICE)
        for n in (1, 4):
            x = _rows(n, 7)
            out = oracle.empty((n, 4), torch.int32)
            rc = oracle.call_status("cere_ktest_inkling_top2", x, n, VOCAB, LD, out, None)
            oracle.check_count(self, ROW, f"time rows={n} status", rc, expected=0)
            reps = 50
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record(stream)
            for _ in range(reps):
                oracle.call_status("cere_ktest_inkling_top2", x, n, VOCAB, LD, out, None, synchronize=False)
            t1.record(stream)
            t1.synchronize()
            us = t0.elapsed_time(t1) * 1000.0 / reps
            oracle.report_row(ROW, f"time top2 rows={n}", phase="timing", us=us, gb_s=n * VOCAB * 4 / us / 1e3)


def _scores_argmax_only(logits: torch.Tensor, vocab: int) -> tuple[None, torch.Tensor]:
    ids = oracle.empty((logits.shape[0],), torch.int32)
    rc = oracle.call_status("cere_ktest_inkling_top2", logits, logits.shape[0], vocab, logits.stride(0), None, ids)
    if rc != 0:
        raise oracle.OracleError(f"cere_ktest_inkling_top2 returned status {rc}")
    return None, ids.to(CPU)


if __name__ == "__main__":
    unittest.main()
