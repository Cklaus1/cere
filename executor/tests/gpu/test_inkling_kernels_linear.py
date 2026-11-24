"""Inkling 2 Mini S8 (tasks/inkling-spec.md 4.1, G4): the BF16 projections of family inkling2_loop
(executor/cuda/src/kernels/inkling/linear.h), y = x · w^T on the tensor cores in one fixed order for every M.

Suite `inkling_kernels` (GPU0 only): `CERE_LIVE=1 python3 scripts/live/native_dev.py test --suite inkling_kernels`
on the inkling-2-mini library (build/native-inkling-2-mini). Synthetic BF16 operands at the forward's projection
shapes: q / gate (3,072 rows), rel (384), sink (24), k / v (1,024), the dynamic-conv taps (32) at k = 2,304; o (2,304)
at k = 3,072; the unembed (200,058) at k = 2,304. Every case reports row "INKLING_LINEAR".
- batch invariance (bit-exact, GPU vs GPU; G4, CERE D9): each row of a batch equals the same row run alone (m = 1),
  inside batches of every narrow size class (m 1, 5, 9, 17, 33, 64), on the wide kernel, in reverse row order, at
  strided operands, and twice; a 4,096-row prefill equals its 512-row chunks and the narrow kernel on its last rows;
  several segments in one call equal each segment alone.
- accuracy: the FP64 product is the reference; c = rel-L2(torch's BF16 matmul, FP64) is reported first, the kernel
  must be within max(4e-3, 2c).
- refusals: malformed arguments return a nonzero status and write nothing.
- timing (reported only): the forward's projection calls at decode (m 1, 4, 64) and a 4,096-row prefill.

Entry point (libcere_ktest.so, executor/cuda/src/ktest/inkling/linear_ktest.cu; int64 sizes, the stream last, 0 on
success):
    int cere_ktest_inkling_linear(const int64_t *segs, int64_t count, int64_t m, int64_t k, int64_t path,
                                  cudaStream_t stream);
  segs: host int64 [count, 6] = (x, ldx, w, y, ldy, n) per segment, device pointers as integers; path 0 auto,
  1 narrow (m <= 64), 2 wide.
"""

from __future__ import annotations

import unittest

import torch

import oracle

ROW = "INKLING_LINEAR"
H = 2304
QROW = 3072
AUTO, NARROW, WIDE = 0, 1, 2
NARROW_ROWS = 64
SHAPES = (("q", QROW, H), ("rel", 384, H), ("sink", 24, H), ("kv", 1024, H), ("dyn", 32, H), ("o", H, QROW),
          ("unembed", 200058, H))
CPU = torch.device("cpu")


def _randn(shape, seed: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(shape, generator=oracle.generator(seed), device=oracle.DEVICE) * scale).to(torch.bfloat16)


def _table(segs) -> torch.Tensor:
    """segs: (x, ldx, w, y, ldy, n) tuples, tensors as device pointers."""
    rows = [[t.data_ptr() if isinstance(t, torch.Tensor) else int(t) for t in s] for s in segs]
    return torch.tensor(rows if rows else [[0] * 6], dtype=torch.int64, device=CPU)


def _status(segs, m: int, k: int, path: int = AUTO, count: int | None = None) -> int:
    count = len(segs) if count is None else count
    return oracle.call_status("cere_ktest_inkling_linear", _table(segs), count, m, k, path)


def _run(x: torch.Tensor, w: torch.Tensor, path: int = AUTO, *, ldy: int | None = None) -> torch.Tensor:
    """y = x · w^T; x may be a column slice of a wider tensor (ldx = its row stride); y's row stride defaults to n
    rounded up to 8 (linear.h: ldy % 8 == 0, the unembed's 200,064 for n = 200,058)."""
    m, k = x.shape
    n = w.shape[0]
    ldy = (n + 7) // 8 * 8 if ldy is None else ldy
    y = oracle.empty((m, ldy), torch.bfloat16)
    rc = _status([(x, x.stride(0), w, y, ldy, n)], m, k, path)
    if rc != 0:
        raise oracle.OracleError(f"cere_ktest_inkling_linear returned status {rc}")
    return y[:, :n]


def _narrow(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The narrow kernel over blocks of 64 rows."""
    return torch.cat([_run(x[i:i + NARROW_ROWS], w, NARROW) for i in range(0, x.shape[0], NARROW_ROWS)])


def _host(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(CPU).contiguous()


class InklingLinear(unittest.TestCase):

    def test_batch_invariance(self):
        for i, (name, n, k) in enumerate(SHAPES):
            rows = 24 if name == "unembed" else 80
            w = _randn((n, k), 100 + i, 0.02)
            x = _randn((rows, k), 200 + i)
            alone = torch.cat([_run(x[r:r + 1], w, NARROW) for r in range(rows)])
            oracle.check_bit_exact(self, ROW, f"{name} m=1 narrow vs auto", _host(alone),
                                   _host(torch.cat([_run(x[r:r + 1], w) for r in range(4)] + [alone[4:]])))
            for m in (1, 5, 9, 17, 33, 64):
                if m <= rows:
                    oracle.check_bit_exact(self, ROW, f"{name} narrow m={m} vs alone", _host(_run(x[:m], w, NARROW)),
                                           _host(alone[:m]))
            wide = _run(x, w, WIDE)
            oracle.check_bit_exact(self, ROW, f"{name} wide m={rows} vs alone", _host(wide), _host(alone))
            oracle.check_bit_exact(self, ROW, f"{name} wide m=1 vs alone", _host(_run(x[rows - 1:], w, WIDE)),
                                   _host(alone[rows - 1:]))
            rev = _run(x.flip(0).contiguous(), w).flip(0)
            oracle.check_bit_exact(self, ROW, f"{name} reversed rows", _host(rev), _host(alone))
            oracle.check_bit_exact(self, ROW, f"{name} twice", _host(_run(x, w, WIDE)), _host(wide))
            wide_x = _randn((rows, k + 64), 300 + i)
            wide_x[:, :k] = x
            pad = (n + 7) // 8 * 8 + 8
            strided = _run(wide_x[:, :k], w, ldy=pad)
            oracle.check_bit_exact(self, ROW, f"{name} strided x and y", _host(strided), _host(alone))

    def test_prefill_chunks(self):
        for i, (name, n, k) in enumerate(SHAPES[:-1]):
            if name not in ("q", "kv", "o"):
                continue
            w = _randn((n, k), 400 + i, 0.02)
            x = _randn((4096, k), 500 + i)
            whole = _run(x, w)
            chunks = torch.cat([_run(x[c:c + 512], w) for c in range(0, 4096, 512)])
            oracle.check_bit_exact(self, ROW, f"{name} m=4096 vs 512-row chunks", _host(chunks), _host(whole))
            oracle.check_bit_exact(self, ROW, f"{name} m=4096 last 100 rows vs narrow", _host(_narrow(x[-100:], w)),
                                   _host(whole[-100:]))

    def test_segments(self):
        m = 37
        xq, xk, xv = (_randn((m, H), 600 + j) for j in range(3))
        ws = {name: _randn((n, H), 700 + j, 0.02) for j, (name, n) in
              enumerate((("q", QROW), ("rel", 384), ("sink", 24), ("gate", QROW), ("k", 1024), ("v", 1024),
                         ("dynk", 32), ("dynv", 32)))}
        groups = {"q group": (("q", xq), ("rel", xq), ("sink", xq), ("gate", xq)),
                  "kv group": (("k", xk), ("v", xv), ("dynk", xk), ("dynv", xv)),
                  "all eight": (("q", xq), ("rel", xq), ("sink", xq), ("gate", xq), ("k", xk), ("v", xv),
                                ("dynk", xk), ("dynv", xv))}
        for gname, group in groups.items():
            for path in (NARROW, WIDE):
                ys = [oracle.empty((m, ws[s].shape[0]), torch.bfloat16) for s, _ in group]
                segs = [(x, H, ws[s], y, y.shape[1], y.shape[1]) for (s, x), y in zip(group, ys)]
                rc = _status(segs, m, H, path)
                oracle.check_count(self, ROW, f"{gname} path {path} status", rc, expected=0)
                for (s, x), y in zip(group, ys):
                    oracle.check_bit_exact(self, ROW, f"{gname} path {path} {s} vs alone", _host(y),
                                           _host(_run(x, ws[s], NARROW if m <= NARROW_ROWS else WIDE)))

    def test_accuracy(self):
        for i, (name, n, k) in enumerate(SHAPES):
            n_ref = min(n, 8192)
            w = _randn((n, k), 800 + i, 0.02)
            x = _randn((96, k), 900 + i)
            fp64 = x.double() @ w[:n_ref].double().T
            c = oracle.calibrate(ROW, f"{name} torch bf16 vs fp64", (x @ w[:n_ref].T).double(), fp64, oracle.FLOOR_OP)
            limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
            for path in (NARROW, WIDE):
                y = _narrow(x, w) if path == NARROW else _run(x, w, WIDE)
                oracle.check_within(self, ROW, f"{name} path {path} vs fp64",
                                    oracle.rel_l2(y[:, :n_ref].double(), fp64), c=c, limit=limit)
            edge = torch.zeros((3, k), dtype=torch.bfloat16, device=oracle.DEVICE)
            edge[1] = 1.0
            edge[2, 0] = -2.0
            got = _run(edge, w)
            want = edge.double() @ w.double().T
            oracle.check_within(self, ROW, f"{name} zero / ones / one-hot rows", oracle.rel_l2(
                got.double(), want.to(torch.bfloat16).double()), c=0.0,
                limit=oracle.calibrated_limit(0.0, oracle.FLOOR_OP))

    def test_refusals(self):
        n, k, m = 24, H, 4
        x = _randn((m, k + 8), 1)
        w = _randn((n, k), 2, 0.02)
        y = oracle.empty((NARROW_ROWS + 1, 32), torch.bfloat16)
        xs = _randn((NARROW_ROWS + 1, k), 3)
        ok = (x, k + 8, w, y, 32, n)
        cases = {
            "k 2,300": ([(x, k + 8, w, y, 32, n)], m, 2300, AUTO),
            "k 3,328": ([(x, k + 8, w, y, 32, n)], m, 3328, AUTO),
            "m 0": ([ok], 0, k, AUTO),
            "narrow m 65": ([(xs, k, w, y, 32, n)], NARROW_ROWS + 1, k, NARROW),
            "x misaligned": ([(x.data_ptr() + 2, k + 8, w, y, 32, n)], m, k, AUTO),
            "w misaligned": ([(x, k + 8, w.data_ptr() + 2, y, 32, n)], m, k, AUTO),
            "ldx < k": ([(x, k - 8, w, y, 32, n)], m, k, AUTO),
            "ldy not a multiple of 8": ([(x, k + 8, w, y, 25, n)], m, k, AUTO),
            "ldy < n": ([(x, k + 8, w, y, 16, n)], m, k, AUTO),
            "n 0": ([(x, k + 8, w, y, 32, 0)], m, k, AUTO),
            "null w": ([(x, k + 8, 0, y, 32, n)], m, k, AUTO),
            "count 0": ([], m, k, AUTO),
            "count 9": ([ok] * 9, m, k, AUTO),
        }
        for name, (segs, mm, kk, path) in cases.items():
            rc = _status(segs, mm, kk, path, count=len(segs))
            oracle.check_count(self, ROW, f"refusal {name}", int(rc != 0), expected=1)
        oracle.check_count(self, ROW, "refusals wrote nothing", int(y.isnan().all().item()), expected=1)

    def test_time(self):
        stream = torch.cuda.current_stream(oracle.DEVICE)
        calls = {"q group": (("q", QROW, H), ("rel", 384, H), ("sink", 24, H), ("gate", QROW, H)),
                 "kv group full": (("k", 1024, H), ("v", 1024, H), ("dynk", 32, H), ("dynv", 32, H)),
                 "o": (("o", H, QROW),), "unembed": (("unembed", 200058, H),)}
        for cname, segs in calls.items():
            k = segs[0][2]
            ws = [_randn((n, k), 1000 + j, 0.02) for j, (_, n, _) in enumerate(segs)]
            nbytes = sum(w.numel() * 2 for w in ws)
            for m in (1, 4, 64, 4096):
                if m == 4096 and cname == "unembed":
                    continue
                x = _randn((m, k), 1100)
                lds = [(n + 7) // 8 * 8 for _, n, _ in segs]
                ys = [oracle.empty((m, ld), torch.bfloat16) for ld in lds]
                table = _table([(x, k, w, y, ld, n) for w, y, ld, (_, n, _) in zip(ws, ys, lds, segs)])
                reps = 50 if m < 4096 else 10
                for _ in range(3):
                    rc = oracle.call_status("cere_ktest_inkling_linear", table, len(segs), m, k, AUTO)
                    oracle.check_count(self, ROW, f"time {cname} m={m} status", rc, expected=0)
                t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t0.record(stream)
                for _ in range(reps):
                    oracle.call_status("cere_ktest_inkling_linear", table, len(segs), m, k, AUTO, synchronize=False)
                t1.record(stream)
                t1.synchronize()
                us = t0.elapsed_time(t1) * 1000.0 / reps
                flops = 2.0 * m * k * sum(n for _, n, _ in segs)
                oracle.report_row(ROW, f"time {cname} m={m}", phase="timing", us=us, gb_s=nbytes / us / 1e3,
                                  tflops=flops / us / 1e6)


if __name__ == "__main__":
    unittest.main()
