"""KLORA: per-request LoRA adapters on the target's dense projections (track LoRA, tasks/lora-spec.md;
`kernels/lora.h`, GemmPlan::use_lora, kernels::gemm_segments).

Suite `kernels_lora` (GPU0 dev loop: `native_dev.py test --suite kernels_lora`):
`python3 -m unittest discover -s /src/executor/tests/gpu -p 'test_kernels_lora_*.py' -v`.

Contract: with an active table, a call on a registered weight returns, per row r with adapter slot s,
y[r, c] = bf16(float(base[r, c]) + scale_s * sum_j B_s[c, j] * (sum_i x[r, i] * A_s[j, i])), base the same plan's
call without a table (its bits), double-float accumulation rounded once; a zero product leaves base's bits. Rows (all
pre-registered in tasks/lora-spec.md before the first run):
- `<shape>/<path>/r<rank>`: within 1 bf16 ulp of the FP64 reference above (rounded once from FP64), at most 1 % of the
  elements differing (double-float vs FP64 sums; an FP32 sum failed the 1-ulp bound where the result cancels
  against base, tasks/lora-spec.md F8), on the target shapes (attention q|k|v packed, o_proj; the shared expert gate
  and up through gemm_segments; down) with init plans (small-M lever on: m 1, 4, 16; off: m 1, 16, 64, 512) and the
  executor's rows plans (init_rows(16), lever on and off, m 1, 3, 16), ranks 16, 32, 64 and 12 (padded to 16).
- `invariance/<shape>/<path>`: batch invariance, bit-exact: a rows-plan call of 16 rows mixing two adapters and rows
  without one equals, row by row, the m = 1 call of that row under its own slot; adapter-free rows equal the call
  without a table.
- `zero_b/<shape>` and `unloaded/<shape>`: an adapter with B = 0, and a slot never loaded, are bit-exact to the call
  without a table (prefill m 64 and decode m 1).
- `perturbed/<shape>`: one A element changed by one bf16 step changes the output (fail-first: the reference check
  sees the adapter). Against a library without the entry point every test fails (symbol absent).
- `refusals`: per-row slots for more than 16 rows are CERE_ERR_ARGUMENT.

Report-only rows `time/<shape>/m<m>/r<rank>`: cold-L2 (ktest/l2_flush.h) median and minimum us of the call with and
without the adapter (small-M lever on, init plan), and the overhead ratio. Gate nothing.

Native entry point (C prototype; returns 0 on success, nonzero `cere_status` otherwise):

  int cere_ktest_lora_plan(const void *x,             /* bfloat16 [m, k] */
                           int64_t m, int64_t k,
                           int64_t count,             /* 1..3 weights sharing x */
                           const int64_t *w_ptrs,     /* host int64 [count]: bfloat16 [ns[i], k] */
                           const int64_t *ns,         /* host int64 [count] */
                           const int64_t *y_ptrs,     /* host int64 [count]: bfloat16 [m, ns[i]] */
                           int64_t rows,              /* 0: init plan (max_m = m); [1, 16]: rows plan */
                           int64_t small_m,
                           int64_t packed,            /* 1 (count 1): the weight's columns are `segments` targets */
                           int64_t segments, const int64_t *seg_n,  /* host int64 [segments], sum = ns[0] */
                           int64_t table,             /* 0: no table (the base call) */
                           int64_t adapters, int64_t rank,  /* slots, max rank */
                           const int64_t *ranks,      /* host int64 [adapters]: 0 = empty slot */
                           const float *scales,       /* host float [adapters] */
                           const int64_t *a_ptrs,     /* host int64 [adapters x T]: HOST bf16 A [ranks[s], k] or 0 */
                           const int64_t *b_ptrs,     /* host int64 [adapters x T]: HOST bf16 B [n_t, ranks[s]] or 0 */
                           int64_t per_row,           /* 0: slots[0] for every row; 1: slots[r] (m <= 16) */
                           const int64_t *slots,      /* host int64: -1 = no adapter */
                           int64_t iters, void *us,   /* iters > 1: cold-L2 timing, us host float64 [2] */
                           cudaStream_t stream);
"""
from __future__ import annotations

import functools
import unittest

import torch

import oracle

ROW = "KLORA"
SEED = 3_300_000
ROWS = 16  # GemmPlan::kMaxRows, lora::kMaxRows
COLD_ITERS = 41
SCALE = 0.5
HIDDEN = 2560
# label: (k, weights (n per weight), packed segments or None): the production model's target projections.
SHAPES = {
    "attn_qkv": (HIDDEN, (13312,), (12288, 512, 512)),
    "attn_o": (6144, (HIDDEN,), None),
    "shared_gate_up": (HIDDEN, (640, 640), None),
    "shared_down": (640, (HIDDEN,), None),
}
INIT_PATHS = {"small_m": (True, (1, 4, 16)), "cublaslt": (False, (1, 16, 64, 512))}
ROWS_M = (1, 3, 16)
RANKS = (16, 32, 64, 12)
TIME_M = (1, 16, 512)
TIME_RANKS = (16, 64)


def _targets(label: str) -> tuple[int, ...]:
    """The n of every target of `label` (segments of the packed weight, else one per weight)."""
    _, ns, segs = SHAPES[label]
    return segs if segs is not None else ns


@functools.cache
def _weights(label: str) -> tuple[torch.Tensor, ...]:
    k, ns, _ = SHAPES[label]
    g = oracle.generator(SEED + list(SHAPES).index(label))
    return tuple((torch.randn((n, k), generator=g, device=oracle.DEVICE) / k ** 0.5).to(torch.bfloat16).contiguous()
                 for n in ns)


@functools.cache
def _x(m: int, k: int) -> torch.Tensor:
    g = oracle.generator(SEED + 7919 * m + k)
    return torch.randn((m, k), generator=g, device=oracle.DEVICE).to(torch.bfloat16).contiguous()


@functools.cache
def _adapter(label: str, rank: int, variant: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Per target: (A [rank, k], B [n, rank]) on the host, BF16; A ~ N(0, 1/k), B ~ N(0, 1/rank)."""
    k = SHAPES[label][0]
    g = torch.Generator()
    g.manual_seed(SEED + 1000 * variant + 31 * rank + list(SHAPES).index(label))
    out = []
    for n in _targets(label):
        a = (torch.randn((rank, k), generator=g) / k ** 0.5).to(torch.bfloat16).contiguous()
        b = (torch.randn((n, rank), generator=g) / rank ** 0.5).to(torch.bfloat16).contiguous()
        out.append((a, b))
    return tuple(out)


def _call(label: str, m: int, *, rows: int = 0, small_m: bool = True, adapters: list | None = None,
          slots: list[int] | None = None, per_row: bool = False, iters: int = 1,
          status: bool = False) -> tuple[torch.Tensor, float, float] | int:
    """One cere_ktest_lora_plan call on `label`'s weights: (y [m, sum n] with the weights' outputs side by side,
    median us, min us); `adapters` None: no table. Each adapter: (rank, scale, per-target (A, B) or None)."""
    k, ns, segs = SHAPES[label]
    ws = _weights(label)
    ys = [oracle.empty((m, n), torch.bfloat16) for n in ns]
    us = torch.zeros((2,), dtype=torch.float64)
    t = len(_targets(label))
    keep = []  # host tensors referenced by pointer
    if adapters is None:
        table, count, rank = 0, 0, 0
        ranks = scales = a_ptrs = b_ptrs = slot_t = None
    else:
        table, count = 1, len(adapters)
        rank = max(a[0] for a in adapters)
        ranks = torch.tensor([a[0] for a in adapters], dtype=torch.int64)
        scales = torch.tensor([a[1] for a in adapters], dtype=torch.float32)
        a_list, b_list = [], []
        for _, _, pairs in adapters:
            for i in range(t):
                pair = None if pairs is None else pairs[i]
                keep.append(pair)
                a_list.append(0 if pair is None else pair[0].data_ptr())
                b_list.append(0 if pair is None else pair[1].data_ptr())
        a_ptrs = torch.tensor(a_list, dtype=torch.int64)
        b_ptrs = torch.tensor(b_list, dtype=torch.int64)
        slot_t = torch.tensor(slots if slots is not None else [-1], dtype=torch.int64)
    args = (_x(m, k), m, k, len(ns), torch.tensor([w.data_ptr() for w in ws],
            dtype=torch.int64), torch.tensor(list(ns), dtype=torch.int64),
            torch.tensor([y.data_ptr() for y in ys], dtype=torch.int64), rows, int(small_m), int(segs is not None),
            len(segs) if segs is not None else 0, None if segs is None else torch.tensor(list(segs), dtype=torch.int64),
            table, count, max(rank, 1), ranks, scales, a_ptrs, b_ptrs, int(per_row), slot_t, iters, us)
    if status:
        return oracle.call_status("cere_ktest_lora_plan", *args)
    oracle.call("cere_ktest_lora_plan", *args)
    del keep
    return torch.cat(ys, dim=1), float(us[0]), float(us[1])


def _reference(label: str, m: int, base: torch.Tensor, adapter: tuple, slots_per_row: list[int]) -> torch.Tensor:
    """bf16(float(base) + scale * B (A x)) in FP64, per row's adapter (index into `adapter`, -1: base)."""
    k = SHAPES[label][0]
    x = _x(m, k).double()
    out = base.double().clone()
    col = 0
    for ti, n in enumerate(_targets(label)):
        for r in range(m):
            s = slots_per_row[r]
            if s < 0:
                continue
            _, scale, pairs = adapter[s]
            a, b = (p.to(oracle.DEVICE).double() for p in pairs[ti])
            d = scale * (b @ (a @ x[r]))
            out[r, col:col + n] = torch.where(d == 0, out[r, col:col + n], out[r, col:col + n] + d)
        col += n
    return out.to(torch.bfloat16)


def _differing(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum().item())


class LoraKernels(unittest.TestCase):
    def test_init_plans_match_the_reference(self) -> None:
        for label in SHAPES:
            for path, (small_m, ms) in INIT_PATHS.items():
                for rank in RANKS:
                    case = f"{label}/init/{path}/r{rank}"
                    with self.subTest(case=case):
                        adapter = [(rank, SCALE, _adapter(label, rank, 0))]
                        natives, refs = [], []
                        for m in ms:
                            base, _, _ = _call(label, m, small_m=small_m)
                            y, _, _ = _call(label, m, small_m=small_m, adapters=adapter, slots=[0])
                            natives.append(y.reshape(-1))
                            refs.append(_reference(label, m, base, adapter, [0] * m).reshape(-1))
                        oracle.check_ulp(self, ROW, case, torch.cat(natives), torch.cat(refs), max_fraction=0.01)

    def test_rows_plans_match_the_reference(self) -> None:
        for label in SHAPES:
            for small_m in (True, False):
                for rank in RANKS:
                    case = f"{label}/rows{ROWS}/{'small_m' if small_m else 'cublaslt'}/r{rank}"
                    with self.subTest(case=case):
                        adapter = [(rank, SCALE, _adapter(label, rank, 0))]
                        natives, refs = [], []
                        for m in ROWS_M:
                            base, _, _ = _call(label, m, rows=ROWS, small_m=small_m)
                            y, _, _ = _call(label, m, rows=ROWS, small_m=small_m, adapters=adapter, slots=[0])
                            natives.append(y.reshape(-1))
                            refs.append(_reference(label, m, base, adapter, [0] * m).reshape(-1))
                        oracle.check_ulp(self, ROW, case, torch.cat(natives), torch.cat(refs), max_fraction=0.01)

    def test_batch_invariance(self) -> None:
        mix = [0, -1, 1, 0, 1, -1, -1, 0, 1, 1, 0, -1, 0, 1, -1, 0]
        for label in SHAPES:
            k = SHAPES[label][0]
            adapters = [(16, SCALE, _adapter(label, 16, 1)), (64, 0.25, _adapter(label, 64, 2))]
            for small_m in (True, False):
                case = f"invariance/{label}/{'small_m' if small_m else 'cublaslt'}"
                with self.subTest(case=case):
                    y, _, _ = _call(label, ROWS, rows=ROWS, small_m=small_m, adapters=adapters, slots=mix,
                                    per_row=True)
                    singles = []
                    for r, s in enumerate(mix):
                        x_r = _x(ROWS, k)[r:r + 1].contiguous()
                        singles.append(_single(label, x_r, s, adapters, small_m))
                    oracle.check_bit_exact(self, ROW, case, y, torch.cat(singles))

    def test_zero_b_and_unloaded_slots_are_the_base_bits(self) -> None:
        for label in SHAPES:
            zero = tuple((a, torch.zeros_like(b)) for a, b in _adapter(label, 32, 3))
            for m in (1, 64):
                base, _, _ = _call(label, m)
                with self.subTest(case=f"zero_b/{label}/m{m}"):
                    y, _, _ = _call(label, m, adapters=[(32, SCALE, zero)], slots=[0])
                    oracle.check_bit_exact(self, ROW, f"zero_b/{label}/m{m}", y, base)
                with self.subTest(case=f"unloaded/{label}/m{m}"):
                    y, _, _ = _call(label, m, adapters=[(32, SCALE, _adapter(label, 32, 3)), (0, SCALE, None)],
                                    slots=[1])
                    oracle.check_bit_exact(self, ROW, f"unloaded/{label}/m{m}", y, base)

    def test_perturbed_adapter_is_seen(self) -> None:
        for label in SHAPES:
            with self.subTest(case=f"perturbed/{label}"):
                pairs = _adapter(label, 16, 4)
                a0 = pairs[0][0].clone()
                a0[3, 5] = (a0[3, 5].float() * 1.5 + 0.25).to(torch.bfloat16)
                perturbed = ((a0, pairs[0][1]),) + pairs[1:]
                y, _, _ = _call(label, 4, adapters=[(16, SCALE, pairs)], slots=[0])
                p, _, _ = _call(label, 4, adapters=[(16, SCALE, perturbed)], slots=[0])
                changed = _differing(y, p)
                oracle.report_row(ROW, f"perturbed/{label}", phase="detect", differing=changed, passed=changed > 0)
                self.assertNotEqual(changed, 0, f"perturbed/{label}: no output bit changed")

    def test_refusals(self) -> None:
        label = "shared_down"
        rc = _call(label, ROWS + 1, adapters=[(16, SCALE, _adapter(label, 16, 0))], slots=[0] * (ROWS + 1),
                   per_row=True, status=True)
        oracle.report_row(ROW, "refusals/per_row_17", phase="status", status=rc, passed=rc == oracle.CERE_ERR_ARGUMENT)
        self.assertEqual(rc, oracle.CERE_ERR_ARGUMENT)

    def test_time_report(self) -> None:
        for label in SHAPES:
            for m in TIME_M:
                base = _call(label, m, iters=COLD_ITERS)
                for rank in TIME_RANKS:
                    y = _call(label, m, adapters=[(rank, SCALE, _adapter(label, rank, 0))], slots=[0],
                              iters=COLD_ITERS)
                    oracle.report_row(ROW, f"time/{label}/m{m}/r{rank}", phase="report", base_us=base[1],
                                      base_min_us=base[2], lora_us=y[1], lora_min_us=y[2],
                                      ratio=y[1] / base[1] if base[1] > 0 else 0.0)
                    self.assertEqual(tuple(y[0].shape), (m, sum(SHAPES[label][1])))


def _single(label: str, x_r: torch.Tensor, slot: int, adapters: list, small_m: bool) -> torch.Tensor:
    """Row x_r alone through the rows plan (m 1) under its own slot (-1: no table)."""
    k, ns, segs = SHAPES[label]
    ws = _weights(label)
    ys = [oracle.empty((1, n), torch.bfloat16) for n in ns]
    t = len(_targets(label))
    a_list, b_list = [], []
    for _, _, pairs in adapters:
        for i in range(t):
            a_list.append(pairs[i][0].data_ptr())
            b_list.append(pairs[i][1].data_ptr())
    oracle.call("cere_ktest_lora_plan", x_r, 1, k, len(ns), torch.tensor([w.data_ptr() for w in ws],
                dtype=torch.int64), torch.tensor(list(ns), dtype=torch.int64),
                torch.tensor([y.data_ptr() for y in ys], dtype=torch.int64), ROWS, int(small_m),
                int(segs is not None), len(segs) if segs is not None else 0,
                None if segs is None else torch.tensor(list(segs), dtype=torch.int64), int(slot >= 0),
                len(adapters), max(a[0] for a in adapters), torch.tensor([a[0] for a in adapters], dtype=torch.int64),
                torch.tensor([a[1] for a in adapters], dtype=torch.float32), torch.tensor(a_list, dtype=torch.int64),
                torch.tensor(b_list, dtype=torch.int64), 0, torch.tensor([slot], dtype=torch.int64), 1, None)
    return torch.cat(ys, dim=1)


if __name__ == "__main__":
    unittest.main()
