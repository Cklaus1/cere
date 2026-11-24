"""K2F (track Fp8W): the FP8 weight path of the dense K2 GEMMs, the opt-in lever value `k2_weights = "fp8"`
(k2_weights slot impl CERE_LEVER_K2_WEIGHTS_FP8, numerics-changing; kernels/k2_fp8.h).

Every case reports row "K2F". Storage of a converted weight w [n, k] (the BF16 tensor's own 2nk bytes): q E4M3 [n, k]
at byte 0, then FP32 scales s [n] at byte n k. Quantization: s[j] = amax_j / 448 (FP32), q[j, i] = E4M3(w[j, i] /
s[j]), round to nearest even, saturated (a zero row: s 0, q 0). The kernel: y[r, j] = s[j] * sum_i x[r, i] q[j, i]
with BF16 tensor-core MMA and FP32 accumulation, BF16 or F32 output.

Cases, on the real checkpoint weights of every converted decode shape (attention qkv packed, o_proj and the QSA
indexer's index_qk_proj of layer 3; GDN in_proj_qkv / z / b / a and out_proj of layer 0; the shared expert gate / up /
down of layer 0; lm_head):
- `quantize_<shape>`: cere_ktest_k2f_quantize is bit-exact to the torch quantization above (torch.float8_e4m3fn).
- `convert_<shape>`: the in-place conversion (a scratch of the scales plus at most a few rows, so the chunk loop runs)
  is bit-exact to the quantization (q bytes and scales).
- `oracle_<shape>_M<m>` (m in 1, 4, 7, 16; the lever's registry contract test_fp8_oracle): the GemmPlan hook's output
  against the FP32 GEMM of the same x and dequantized weights. Tolerance rule: reference op = torch FP32 matmul of x
  and q (as FP32) times s, cast to the output dtype; c = rel-L2(reference op, the FP64 product); limit = max(4e-3,
  2c); measured = rel-L2(native, reference op). 10 repeats of each call are bit-identical. Report-only:
  `quant_error_<shape>_M<m>` = rel-L2(native, torch.matmul in BF16 of the checkpoint weight), the weight-only FP8
  error the D9Q screen judges end to end.
- `e4m3_exact`: every finite E4M3 byte (254 values, signed zeros included) through the kernel with scale 1 and a
  one-hot x equals the torch E4M3 value in F32, bit for bit (the in-register E4M3 -> BF16 widening is exact).
- `non_finite_rows` (audit KGM-4): a row holding a NaN, an Inf or both quantizes to scale NaN (0x7FC00000) and q 0,
  by quantize and by the in-place convert; so its m = 1 GEMM column and its dequantized row are NaN, while the finite
  rows (one saturating to +-448) equal the torch quantization and no q byte is an E4M3 NaN (0x7F / 0xFF, which the
  GEMM's widening reads as +-480). Before KGM-4 fmaxf dropped the NaN (a finite scale, a +-480 GEMM product) and an
  Inf gave scale Inf.
- `batch_invariant_<shape>`: for every m in 2..16 each row of the m-row call is bit-equal to the m = 1 call on that
  row (count 15).
- `segments_<group>_M<m>`: kernels::gemm_segments over converted weights (GDN in-projections, the shared gate/up)
  equals each weight's own plan call bit for bit.
- `prefill_<shape>_M<m>` (m in 17, 64): m > 16 dequantizes into the staging buffer and runs the BF16 plan: bit-equal
  to cere_ktest_gemm_path (cuBLASLt) on cere_ktest_k2f_dequantize's weights, which are bit-equal to BF16(q * s).
- `refusals`: k not a multiple of 64 and a staging buffer smaller than the weight at m > 16 are refused.
- Report-only bench `bench_<shape>_M<m>`: µs per call and effective TB/s (FP8 weight + scales + x + y bytes) of the
  FP8 kernel and of the BF16 small-M kernel (cere_ktest_gemm_path), cold (calls rotate over copies totalling >= 384
  MiB, 3x L2), and the FP8 kernel's fraction of the achievable DRAM read stream (scripts/bench/dram_peak.json).

Native entry points (oracle.py calling convention: device pointers `void *`, host int64 arrays `const int64_t *`,
integers `int64_t`, the stream last; 0 = success). All tensors are contiguous and row-major.

    int cere_ktest_k2f_quantize(const void *w, int64_t rows, int64_t k, void *q, void *scale, cudaStream_t stream);
        w bf16 [rows, k] -> q E4M3 bytes [rows, k], scale f32 [rows].

    int cere_ktest_k2f_convert(void *storage, int64_t n, int64_t k, void *scratch, int64_t scratch_bytes,
                               cudaStream_t stream);
        storage: bf16 [n, k] on entry, the converted storage (q, then the scales) on return.

    int cere_ktest_k2f_dequantize(const void *storage, int64_t n, int64_t k, void *w, cudaStream_t stream);
        w bf16 [n, k] = BF16(q * s) of a converted storage.

    int cere_ktest_k2f_gemm(const void *x, int64_t m, int64_t k, int64_t count, const int64_t *w_ptrs,
                            const int64_t *ns, const int64_t *y_ptrs, int64_t out_f32, int64_t iters, int64_t copies,
                            int64_t stride, void *us, cudaStream_t stream);
        One FP8 launch over `count` converted storages sharing x; `iters` launches rotating over `copies` storages
        `stride` bytes apart (count 1); us: host f64 [1], mean µs per launch after 5 warm-ups when iters > 5.

    int cere_ktest_k2f_plan(const void *x, const void *storage, int64_t m, int64_t n, int64_t k, int64_t out_f32,
                            int64_t small_m, void *staging, int64_t staging_bytes, void *y, cudaStream_t stream);
        y [m, n] = GemmPlan(n, k, max_m = m).run(x, storage) with the plan's fp8 table listing storage.

    int cere_ktest_k2f_segments(const void *x, int64_t m, int64_t k, int64_t count, const int64_t *w_ptrs,
                                const int64_t *ns, const int64_t *y_ptrs, int64_t out_f32, cudaStream_t stream);
        kernels::gemm_segments over `count` converted storages (one plan each, one fp8 table).

    int cere_ktest_gemm_path(const void *x, const void *w, int64_t m, int64_t n, int64_t k, int64_t out_f32,
                             int64_t small_m, int64_t iters, int64_t copies, void *y, void *us, cudaStream_t stream);
        As test_kernels_k2_gemm.py: the BF16 plan (small-M kernel on or off).
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

import oracle

ROW = "K2F"
ATTN_LAYER = 3
GDN_LAYER = 0
SHARED_LAYER = 0
ORACLE_M = (1, 4, 7, 16)
PREFILL_M = (17, 64)
BENCH_M = (1, 4, 16)
BENCH_ITERS = 55  # 5 warm-ups + 50 timed
COLD_BYTES = 384 << 20  # 3x the 128 MiB L2
# The achievable DRAM read stream (scripts/bench/dram_peak.json `peak_bps`, O249), in TB/s.
PEAK_TBPS = json.loads((Path(__file__).resolve().parents[3] / "scripts/bench/dram_peak.json").read_text())[
    "peak_bps"] / 1e12
REPEATS = 10
SEED = 900_000
E4M3_MAX = 448.0
STAGE_ROWS = 3  # convert_<shape>'s scratch holds the scales plus this many BF16 rows
REGION_ALIGN = 256
CERE_ERR_ARGUMENT = -1  # cere_executor.h
CERE_ERR_UNSUPPORTED = -2


def _weight(name: str) -> torch.Tensor:
    w = oracle.real_tensor(name)
    if w.dtype != torch.bfloat16 or w.dim() != 2:
        raise oracle.OracleError(f"{name} is {w.dtype} {tuple(w.shape)}; expected 2-D BF16")
    return w.contiguous()


def _shapes() -> list:
    """(label, loader, out_f32) of every weight the k2_weights lever converts (model/fp8_weights.cpp)."""
    attn = oracle.layer_prefix(ATTN_LAYER) + "self_attn."
    gdn = oracle.layer_prefix(GDN_LAYER) + "linear_attn."
    mlp = oracle.layer_prefix(SHARED_LAYER) + "mlp."
    return [
        ("attn_qkv", lambda: torch.cat([_weight(attn + p + ".weight") for p in ("q_proj", "k_proj", "v_proj")],
                                       dim=0).contiguous(), False),
        ("attn_o", lambda: _weight(attn + "o_proj.weight"), False),
        ("attn_index_qk", lambda: _weight(attn + "indexer.index_qk_proj.weight"), False),
        ("gdn_qkv", lambda: _weight(gdn + "in_proj_qkv.weight"), False),
        ("gdn_z", lambda: _weight(gdn + "in_proj_z.weight"), False),
        ("gdn_b", lambda: _weight(gdn + "in_proj_b.weight"), False),
        ("gdn_a", lambda: _weight(gdn + "in_proj_a.weight"), False),
        ("gdn_out", lambda: _weight(gdn + "out_proj.weight"), False),
        ("moe_shared_gate", lambda: _weight(mlp + "shared_expert.gate_proj.weight"), False),
        ("moe_shared_up", lambda: _weight(mlp + "shared_expert.up_proj.weight"), False),
        ("moe_shared_down", lambda: _weight(mlp + "shared_expert.down_proj.weight"), False),
        ("lm_head", lambda: _weight("lm_head.weight"), False),
    ]


def _activations(m: int, k: int, seed: int) -> torch.Tensor:
    g = oracle.generator(seed)
    return torch.randn((m, k), generator=g, device=oracle.DEVICE, dtype=torch.float32).to(torch.bfloat16)


def _torch_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(q uint8 [n, k], s f32 [n]) by the rule of the module docstring. Both divisions are the correctly rounded FP32
    quotient (FP64 division rounded once to FP32 equals it), as the kernel's __fdiv_rn."""
    qs, ss = [], []
    for w_rows in w.split(16_384):
        w64 = w_rows.double()
        s = (w64.abs().amax(dim=1) / E4M3_MAX).float()
        safe = torch.where(s > 0, s, torch.ones_like(s))
        q = (w64 / safe.double()[:, None]).float().clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        qs.append(torch.where((s > 0)[:, None], q.view(torch.uint8), torch.zeros_like(q.view(torch.uint8))))
        ss.append(s)
        del w64, q
    return torch.cat(qs).contiguous(), torch.cat(ss).contiguous()


def _native_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = w.shape
    q = oracle.empty((n, k), torch.uint8)
    s = oracle.empty((n,), torch.float32)
    oracle.call("cere_ktest_k2f_quantize", w, n, k, q, s)
    return q, s


def _storage(w: torch.Tensor, scratch_rows: int | None = None) -> torch.Tensor:
    """A converted copy of w (uint8 [2nk]) by cere_ktest_k2f_convert; scratch_rows None: room for every row."""
    n, k = w.shape
    storage = w.clone().view(torch.uint8).reshape(-1).contiguous()
    rows = n if scratch_rows is None else scratch_rows
    scale_bytes = -(-4 * n // REGION_ALIGN) * REGION_ALIGN
    scratch = torch.empty((scale_bytes + rows * 2 * k,), dtype=torch.uint8, device=oracle.DEVICE)
    oracle.call("cere_ktest_k2f_convert", storage, n, k, scratch, scratch.numel())
    del scratch
    return storage


def _split(storage: torch.Tensor, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(q uint8 [n, k], s f32 [n]) views of a converted storage."""
    q = storage[:n * k].view(n, k)
    s = storage[n * k:n * k + 4 * n].view(torch.float32)
    return q, s


def _dequant32(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return q.view(torch.float8_e4m3fn).float() * s[:, None]


def _plan(x: torch.Tensor, storage: torch.Tensor, n: int, out_f32: bool, small_m: bool = True,
          staging: torch.Tensor | None = None) -> torch.Tensor:
    m, k = x.shape
    y = oracle.empty((m, n), torch.float32 if out_f32 else torch.bfloat16)
    staging_bytes = 0 if staging is None else staging.numel()
    oracle.call("cere_ktest_k2f_plan", x.contiguous(), storage, m, n, k, int(out_f32), int(small_m), staging,
                staging_bytes, y)
    return y


def _gemm_path(x: torch.Tensor, w: torch.Tensor, out_f32: bool, small_m: bool, iters: int = 1, copies: int = 1):
    """`cere_ktest_gemm_path` (BF16 plan): (y, µs per call when iters > 5)."""
    m, k = x.shape
    n = w.shape[-2]
    y = oracle.empty((m, n), torch.float32 if out_f32 else torch.bfloat16)
    us = torch.zeros((1,), dtype=torch.float64)
    oracle.call("cere_ktest_gemm_path", x.contiguous(), w, m, n, k, int(out_f32), int(small_m), iters, copies, y, us)
    return y, float(us[0])


def _pointers(tensors) -> torch.Tensor:
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64)


class K2Fp8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")

    @classmethod
    def tearDownClass(cls):
        torch.set_float32_matmul_precision(cls.previous_precision)

    def test_fp8_quantize_and_convert(self):
        for label, load, _ in _shapes():
            w = load()
            n, k = w.shape
            q_ref, s_ref = _torch_quantize(w)
            q, s = _native_quantize(w)
            oracle.check_bit_exact(self, ROW, f"quantize_{label}_q", q, q_ref)
            oracle.check_bit_exact(self, ROW, f"quantize_{label}_scale", s, s_ref)
            storage = _storage(w, STAGE_ROWS)
            cq, cs = _split(storage, n, k)
            oracle.check_bit_exact(self, ROW, f"convert_{label}_q", cq, q_ref)
            oracle.check_bit_exact(self, ROW, f"convert_{label}_scale", cs, s_ref)
            del w, storage, q, s, q_ref, s_ref

    def test_fp8_oracle(self):
        for index, (label, load, out_f32) in enumerate(_shapes()):
            w = load()
            n, k = w.shape
            storage = _storage(w)
            q, s = _split(storage, n, k)
            deq = _dequant32(q, s)
            for m in ORACLE_M:
                with self.subTest(weight=label, m=m):
                    case = f"oracle_{label}_M{m}"
                    x = _activations(m, k, SEED + 1_000 * index + m)
                    exact = (x.double() @ q.view(torch.float8_e4m3fn).double().t()) * s.double()[None, :]
                    product = (x.float() @ q.view(torch.float8_e4m3fn).float().t()) * s[None, :]
                    reference_op = product if out_f32 else product.to(torch.bfloat16)
                    c = oracle.calibrate(ROW, case, reference_op, exact, oracle.FLOOR_OP)
                    limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
                    y = _plan(x, storage, n, out_f32)
                    oracle.check_within(self, ROW, case, oracle.rel_l2(y, reference_op), c=c, limit=limit)
                    same = 1
                    for _ in range(REPEATS - 1):
                        same += int(torch.equal(_plan(x, storage, n, out_f32).view(torch.uint8), y.view(torch.uint8)))
                    oracle.check_count(self, ROW, f"{case}_repeats_identical", same, expected=REPEATS)
                    bf16 = F.linear(x, w).float()
                    oracle.report_row(ROW, f"quant_error_{label}_M{m}", phase="report",
                                      measured=oracle.rel_l2(y.float(), bf16),
                                      dequant_vs_bf16=oracle.rel_l2((x.float() @ deq.t()), bf16))
                    del exact, product, reference_op, bf16
            del w, storage, deq

    def test_fp8_e4m3_exact(self):
        n, k = 256, 64
        codes = torch.arange(256, dtype=torch.int64, device=oracle.DEVICE)
        nan = (codes & 0x7F) == 0x7F
        codes = torch.where(nan, torch.zeros_like(codes), codes).to(torch.uint8)
        storage = torch.empty((n * k + 4 * n,), dtype=torch.uint8, device=oracle.DEVICE)
        storage[:n * k].view(n, k).copy_(codes[:, None].expand(n, k))
        storage[n * k:].view(torch.float32).fill_(1.0)
        x = torch.zeros((1, k), dtype=torch.bfloat16, device=oracle.DEVICE)
        x[0, 0] = 1.0
        y = oracle.empty((1, n), torch.float32)
        oracle.call("cere_ktest_k2f_gemm", x, 1, k, 1, _pointers([storage]), torch.tensor([n], dtype=torch.int64),
                    _pointers([y]), 1, 1, 1, 0, None)
        # + 0.0: the FP32 accumulator starts at +0, so code 0x80 (-0) sums to +0.
        expected = codes.view(torch.float8_e4m3fn).float()[None, :] + 0.0
        oracle.check_bit_exact(self, ROW, "e4m3_exact", y, expected)

    def test_fp8_non_finite_rows(self):
        n, k = 8, 128
        g = oracle.generator(SEED + 90_000)
        w = torch.randn((n, k), generator=g, device=oracle.DEVICE, dtype=torch.float32).to(torch.bfloat16)
        w[0, 5] = float("nan")
        w[2, 0] = float("inf")
        w[3, k - 1] = float("-inf")
        w[5, 7], w[5, 9] = float("nan"), float("inf")
        w[6].zero_()
        w[7, 3] = 3.0e38  # finite, far above every other element: the rest of the row rounds to 0
        bad = torch.zeros(n, dtype=torch.bool, device=oracle.DEVICE)
        bad[[0, 2, 3, 5]] = True
        q_ref, s_ref = _torch_quantize(torch.where(bad[:, None], torch.zeros_like(w), w))
        s_ref = torch.where(bad, torch.full_like(s_ref, float("nan")), s_ref)
        q, s = _native_quantize(w)
        storage = _storage(w, STAGE_ROWS)
        cq, cs = _split(storage, n, k)
        for name, (got_q, got_s) in (("quantize", (q, s)), ("convert", (cq, cs))):
            oracle.check_bit_exact(self, ROW, f"non_finite_rows_{name}_q", got_q, q_ref)
            oracle.check_bit_exact(self, ROW, f"non_finite_rows_{name}_scale", got_s, s_ref)
            oracle.check_count(self, ROW, f"non_finite_rows_{name}_nan_bytes", int(((got_q & 0x7F) == 0x7F).sum()),
                               expected=0)
        x = _activations(1, k, SEED + 90_001)
        y = _plan(x, storage, n, False)
        oracle.check_bit_exact(self, ROW, "non_finite_rows_gemm_nan_columns", torch.isnan(y[0]), bad)
        deq = oracle.empty((n, k), torch.bfloat16)
        oracle.call("cere_ktest_k2f_dequantize", storage, n, k, deq)
        oracle.check_count(self, ROW, "non_finite_rows_dequant_nan", int(torch.isnan(deq[bad]).sum()),
                           expected=4 * k)
        oracle.check_bit_exact(self, ROW, "non_finite_rows_dequant_finite", deq[~bad],
                               _dequant32(q_ref[~bad], s_ref[~bad]).to(torch.bfloat16))

    def test_fp8_batch_invariant(self):
        for index, (label, load, out_f32) in enumerate(_shapes()):
            w = load()
            n, k = w.shape
            storage = _storage(w)
            del w
            x = _activations(16, k, SEED + 50_000 + index)
            singles = [_plan(x[r:r + 1], storage, n, out_f32) for r in range(16)]
            invariant = 0
            for m in range(2, 17):
                y = _plan(x[:m], storage, n, out_f32)
                invariant += int(all(torch.equal(y[r:r + 1].view(torch.uint8), singles[r].view(torch.uint8))
                                     for r in range(m)))
            oracle.check_count(self, ROW, f"batch_invariant_{label}", invariant, expected=15)
            del storage

    def test_fp8_segments(self):
        shapes = {label: (load, out_f32) for label, load, out_f32 in _shapes()}
        groups = {"gdn_in": ("gdn_qkv", "gdn_z", "gdn_b", "gdn_a"), "shared_gate_up": ("moe_shared_gate",
                                                                                      "moe_shared_up")}
        for gi, (group, labels) in enumerate(groups.items()):
            weights = [shapes[label][0]() for label in labels]
            out_f32 = shapes[labels[0]][1]
            k = weights[0].shape[1]
            ns = [w.shape[0] for w in weights]
            storages = [_storage(w) for w in weights]
            del weights
            for m in (1, 4, 16):
                with self.subTest(group=group, m=m):
                    x = _activations(m, k, SEED + 60_000 + 100 * gi + m)
                    ys = [oracle.empty((m, n), torch.float32 if out_f32 else torch.bfloat16) for n in ns]
                    oracle.call("cere_ktest_k2f_segments", x, m, k, len(storages), _pointers(storages),
                                torch.tensor(ns, dtype=torch.int64), _pointers(ys), int(out_f32))
                    for i, label in enumerate(labels):
                        single = _plan(x, storages[i], ns[i], out_f32)
                        oracle.check_bit_exact(self, ROW, f"segments_{group}_{label}_M{m}", ys[i], single)
            del storages

    def test_fp8_prefill_dequant(self):
        for index, (label, load, out_f32) in enumerate(_shapes()):
            if label == "lm_head":
                continue  # lm_head runs m = 1 only (lm_head_last_row); the rows lm_head m <= 16
            w = load()
            n, k = w.shape
            storage = _storage(w)
            del w
            q, s = _split(storage, n, k)
            deq = oracle.empty((n, k), torch.bfloat16)
            oracle.call("cere_ktest_k2f_dequantize", storage, n, k, deq)
            oracle.check_bit_exact(self, ROW, f"dequantize_{label}", deq, _dequant32(q, s).to(torch.bfloat16))
            staging = torch.empty((2 * n * k,), dtype=torch.uint8, device=oracle.DEVICE)
            for m in PREFILL_M:
                with self.subTest(weight=label, m=m):
                    x = _activations(m, k, SEED + 70_000 + 100 * index + m)
                    y = _plan(x, storage, n, out_f32, small_m=True, staging=staging)
                    reference, _ = _gemm_path(x, deq, out_f32, False)
                    oracle.check_bit_exact(self, ROW, f"prefill_{label}_M{m}", y, reference)
            del storage, deq, staging

    def test_fp8_refusals(self):
        n, k = 64, 96
        x = _activations(1, k, SEED + 80_000)
        storage = torch.zeros((2 * n * k,), dtype=torch.uint8, device=oracle.DEVICE)
        y = oracle.empty((1, n), torch.bfloat16)
        rc = oracle.call_status("cere_ktest_k2f_gemm", x, 1, k, 1, _pointers([storage]),
                                torch.tensor([n], dtype=torch.int64), _pointers([y]), 0, 1, 1, 0, None)
        oracle.check_count(self, ROW, "refuse_k96", rc, expected=CERE_ERR_ARGUMENT)
        k = 128
        storage = torch.zeros((2 * n * k,), dtype=torch.uint8, device=oracle.DEVICE)
        x = _activations(17, k, SEED + 80_001)
        y = oracle.empty((17, n), torch.bfloat16)
        staging = torch.empty((2 * n * k - 2,), dtype=torch.uint8, device=oracle.DEVICE)
        rc = oracle.call_status("cere_ktest_k2f_plan", x, storage, 17, n, k, 0, 1, staging, staging.numel(), y)
        oracle.check_count(self, ROW, "refuse_short_staging", rc, expected=CERE_ERR_UNSUPPORTED)

    def test_fp8_bench(self):
        """Report-only: µs per call and effective TB/s of the FP8 kernel and of the BF16 small-M kernel."""
        for index, (label, load, out_f32) in enumerate(_shapes()):
            w = load()
            n, k = w.shape
            storage = _storage(w)
            stride = -(-(n * k + 4 * n) // REGION_ALIGN) * REGION_ALIGN
            copies = min(BENCH_ITERS, -(-COLD_BYTES // stride))
            pool = torch.empty((copies, stride), dtype=torch.uint8, device=oracle.DEVICE)
            pool[:, :n * k + 4 * n].copy_(storage[:n * k + 4 * n].expand(copies, -1))
            del storage
            bf16_copies = min(BENCH_ITERS, -(-COLD_BYTES // (2 * n * k)))
            ws = w.unsqueeze(0).repeat(bf16_copies, 1, 1).contiguous() if bf16_copies > 1 else w
            for m in BENCH_M:
                x = _activations(m, k, SEED + 90_000 + index)
                y = oracle.empty((m, n), torch.float32 if out_f32 else torch.bfloat16)
                us = torch.zeros((1,), dtype=torch.float64)
                oracle.call("cere_ktest_k2f_gemm", x, m, k, 1, _pointers([pool]), torch.tensor([n], dtype=torch.int64),
                            _pointers([y]), int(out_f32), BENCH_ITERS, copies, stride, us)
                fp8_us = float(us[0])
                _, bf16_us = _gemm_path(x, ws, out_f32, True, BENCH_ITERS, bf16_copies)
                out_bytes = (4 if out_f32 else 2) * m * n
                fp8_moved = n * k + 4 * n + 2 * m * k + out_bytes
                bf16_moved = 2 * n * k + 2 * m * k + out_bytes
                fp8_tbps = fp8_moved / fp8_us / 1e6
                oracle.report_row(ROW, f"bench_{label}_M{m}", phase="bench", n=n, k=k, m=m, fp8_us=fp8_us,
                                  bf16_small_m_us=bf16_us, fp8_tbps=fp8_tbps, bf16_tbps=bf16_moved / bf16_us / 1e6,
                                  fp8_peak_fraction=fp8_tbps / PEAK_TBPS, speedup=bf16_us / fp8_us)
            del w, ws, pool


if __name__ == "__main__":
    unittest.main()
