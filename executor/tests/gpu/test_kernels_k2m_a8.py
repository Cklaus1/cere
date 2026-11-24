"""K2A8: the W8A8 path of the k2_gemm lever value small_m_a8 (tasks/k2-prefill-spec.md P3, gate G1;
`kernels/k2_mxfp8.h` quantize_x / repack_scales, GemmPlan::init_a8 / use_a8).

Suite `kernels_k2m` (GPU1 lane: `native_dev.py --gpu 1 test --suite kernels_k2m`):
`python3 -m unittest discover -s /src/executor/tests/gpu -p 'test_kernels_k2m_*.py' -v`.

Contract (numerics-changing, opt-in `-v56`): an init() plan's call on an MXFP8 weight whose row choice (cls, else m)
exceeds 16 quantizes x to E4M3 with one UE8M0 scale per 32-element group of a row (e = the smallest exponent with
amax <= 448 * 2^e, clamped to [-127, 127]; amax 0 or an FP32 subnormal amax -> -127; q = RN-satfinite(x * 2^-e))
and runs cuBLASLt's MXFP8 GEMM on q and the weight's E4M3 bytes and scales (BF16 out). Every other call (m <= 16, rows
plans, BF16 weights) is small_m's, bit for bit. Weights and x are test_kernels_k2m_mxfp8.py's (same seeds, shapes).

Rows:
- `quantize/k<k>/m<m>` (every served k, m 17 / 64 / 1,600 / 4,096) and `quantize/edges` (all +-0 groups, BF16
  subnormal groups, amax at and one BF16 step above 448 * 2^e, the largest BF16, ties and underflows to E4M3 subnormal
  and zero): cere_ktest_k2m_quantize_x bit-exact against the torch rule above (the E4M3 bytes; the scale bytes read
  back through the blocked offset, every other byte of the buffer untouched).
- `repack/n<n>_k<k>`: cere_ktest_k2m_repack_scales bit-exact at the blocked offset, every other byte untouched.
- `a8/<shape>/m<m>` (m 17, 64, 512, 1,600, 4,096): the call takes the W8A8 path (info 5) on every shape, and its
  rel-L2 against the FP64 product of the two dequantized operands is within oracle.calibrated_limit(c, FLOOR_OP),
  c = the rel-L2 of torch's BF16 matmul of the same dequantized operands. Report-only `noise`: the rel-L2 of the a8
  call against the BF16 call on the unquantized x (the quantization noise the D9Q screen prices, gate G4).
- `a8/<shape>/cls`: the m 4 call with cls 64 equals rows 0..3 of the m 64 call on the same first rows (the batch
  invariance R3's hot == cold needs); `a8/<shape>/repeat`: two calls bit-equal.
- `same/<shape>/...`: under small_m_a8 every m <= 16 init call, every rows16 call and the BF16-weight calls (m 17,
  512) bit-equal to the same calls under small_m.
- `segments/<group>/m<m>/<mix>`: gemm_segments under small_m_a8 (m 64 and 1,600; every weight MXFP8, and a mix of
  MXFP8 and BF16) bit-equal to the per-plan calls.
- `refusal/attn_o`: a staging buffer one byte short of the W8A8 operands returns UNSUPPORTED; exactly enough runs.

Fail-first: `perturbed/*` gives one weight scale byte + 1 to the a8 call (plan and segments) and requires a difference
from the unperturbed call; against a library without these entry points every test fails (symbol absent).

Report-only rows `time/<shape>/m<m>` (m 512, 1,600, 4,096): cold-L2 median and minimum us of the a8 call, the
small_m call on the MXFP8 weight (dequantize to the staging buffer + BF16 cuBLASLt) and the BF16 cuBLASLt call.

Native entry points (besides test_kernels_k2m_mxfp8.py's; small_m 2 there is small_m_a8, info 5 the W8A8 path):

  int cere_ktest_k2m_quantize_x(const void *x,        /* bfloat16 [m, k], 16-byte aligned; k a multiple of 32 */
                                int64_t m, int64_t k,
                                void *q,              /* uint8 E4M3 [m, k], 16-byte aligned */
                                void *scales,         /* uint8, blocked_bytes(m, k / 32): the blocked layout */
                                cudaStream_t stream);

  int cere_ktest_k2m_repack_scales(const void *scales,  /* uint8 UE8M0 [n, k / 32], row-major */
                                   int64_t n, int64_t k,
                                   void *blocked,       /* uint8, blocked_bytes(n, k / 32) */
                                   cudaStream_t stream);

Blocked layout (cuBLASLt VEC32_UE8M0): blocked_bytes(rows, cols) = ceil(rows / 128) * 128 * ceil(cols / 4) * 4, and
scale (r, c) at ((r // 128) * ceil(cols / 4) + c // 4) * 512 + (r % 32) * 16 + (r % 128 // 32) * 4 + c % 4.
"""
from __future__ import annotations

import functools
import unittest

import torch

import oracle
import test_kernels_k2m_mxfp8 as k2m

ROW = "K2A8"
GROUP_K = k2m.GROUP_K
SMALL_M_A8 = 2  # cere_ktest_k2m_plan / _segments small_m: the lever value small_m_a8
SMALL_M = 1
A8_PATH = 5  # cere_ktest_k2m_plan info: the W8A8 path
QUANTIZE_M = (17, 64, 1600, 4096)
A8_M = (17, 64, 512, 1600, 4096)
SAME_LARGE_M = (17, 512)
SEGMENT_M = (64, 1600)
TIME_M = (512, 1600, 4096)
CLS_M, CLS = 4, 64
E4M3_MAX = 448.0
A8_ALIGN = 256  # k2_gemm.cpp kA8Align


def _blocked_bytes(rows: int, cols: int) -> int:
    return (rows + 127) // 128 * 128 * ((cols + 3) // 4 * 4)


@functools.cache
def _blocked_index(rows: int, cols: int) -> torch.Tensor:
    """int64 [rows, cols]: the byte offset of scale (r, c) in the blocked layout (module docstring)."""
    r = torch.arange(rows, device=oracle.DEVICE, dtype=torch.int64)[:, None]
    c = torch.arange(cols, device=oracle.DEVICE, dtype=torch.int64)[None, :]
    return ((r // 128) * ((cols + 3) // 4) + c // 4) * 512 + (r % 32) * 16 + (r % 128 // 32) * 4 + c % 4


def _read_blocked(buf: torch.Tensor, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(the [rows, cols] scales at their blocked offsets, the bytes at every other offset of `buf`)."""
    index = _blocked_index(rows, cols)
    mask = torch.ones(buf.numel(), dtype=torch.bool, device=oracle.DEVICE)
    mask[index.reshape(-1)] = False
    return buf[index], buf[mask]


def _quantize_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(q uint8 E4M3 [m, k], e int32 [m, k / 32]) of the module docstring's rule. 448 = 1.75 * 2^8: e = E - 8, + 1 when
    the FP32 mantissa field exceeds 0x600000; 2^-e is the normal FP32 of biased exponent 127 - e (e <= 120 for a
    finite BF16), so x * 2^-e is exact before torch's one RNE E4M3 rounding (|x * 2^-e| <= 448)."""
    m, k = x.shape
    xf = x.to(torch.float32).reshape(m, k // GROUP_K, GROUP_K)
    bits = xf.abs().amax(dim=-1).view(torch.int32)
    field = bits >> 23
    e = field - 135 + ((bits & 0x7FFFFF) > 0x600000).to(torch.int32)
    e = torch.where(field == 0, torch.full_like(e, -127), e.clamp(-127, 127))
    inv = ((127 - e) << 23).view(torch.float32)
    q = (xf * inv[..., None]).to(torch.float8_e4m3fn).view(torch.uint8).reshape(m, k)
    return q.contiguous(), e


def _dequantize_x(q: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """FP32 [m, k]: q * 2^e. Every x here has e >= -126 where q is nonzero (2^e then a normal FP32 from its bits)."""
    scale = ((e.clamp(min=-126) + 127) << 23).view(torch.float32).repeat_interleave(GROUP_K, dim=1)
    return q.view(torch.float8_e4m3fn).to(torch.float32) * scale


def _quantize_native(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(q, the whole blocked scale buffer) of one cere_ktest_k2m_quantize_x call (buffers 0xFF-filled first)."""
    m, k = x.shape
    q = oracle.empty((m, k), torch.uint8)
    scales = oracle.empty((_blocked_bytes(m, k // GROUP_K),), torch.uint8)
    oracle.call("cere_ktest_k2m_quantize_x", x, m, k, q, scales)
    return q, scales


def _edges() -> torch.Tensor:
    """BF16 [8, 256]: one 32-element group per pattern (module docstring's `quantize/edges`), each row's other groups
    seeded randn at shifted magnitudes."""
    g = oracle.generator(k2m.SEED + 11)
    x = torch.randn((8, 256), generator=g, device=oracle.DEVICE, dtype=torch.float32)
    x = x * torch.exp2(torch.arange(-12, 12, 3, device=oracle.DEVICE, dtype=torch.float32))[:, None]
    lanes = torch.arange(GROUP_K, device=oracle.DEVICE, dtype=torch.float32)
    patterns = [
        torch.zeros(GROUP_K, device=oracle.DEVICE),  # amax 0
        torch.full((GROUP_K,), -0.0, device=oracle.DEVICE),  # every lane -0
        (lanes + 1) * 2.0**-133,  # BF16 subnormals (amax FP32 exponent field 0)
        (lanes + 1) * 2.0**-125,  # normal amax below the clamp: e -127
        torch.where(lanes == 0, E4M3_MAX * 2.0**-3, lanes * 2.0**-9),  # amax exactly 448 * 2^-3
        torch.where(lanes == 0, 450.0 * 2.0**5, -lanes * 2.0**4),  # one BF16 step above 448 * 2^5: e + 1
        torch.where(lanes == 0, 3.3895313892515355e38, lanes * 2.0**110),  # the largest BF16
        torch.where(lanes == 0, 448.0, torch.tensor([1.0625, 2.0**-12, 2.0**-10, 3 * 2.0**-11, -1.0625,
                                                     416.0 + 16.0, -2.0**-10, 1.1875] * 4, device=oracle.DEVICE)),
    ]
    for r, p in enumerate(patterns):
        x[r, 32 * (r % 8):32 * (r % 8) + GROUP_K] = p
    return x.to(torch.bfloat16).contiguous()


def _call(label: str, x: torch.Tensor, *, mxfp8: bool, small_m: int, rows: int = 0, cls: int = 0,
          scales: torch.Tensor | None = None, staging: torch.Tensor | None = None,
          iters: int = 1) -> tuple[torch.Tensor, int, float, float]:
    """(y, info, median us, min us) of one cere_ktest_k2m_plan call of `label`'s weight on x (k2m._plan with x
    explicit)."""
    n, k = k2m.SHAPES[label]
    m = x.shape[0]
    q, s = k2m._mxfp8(label)
    w, sc = (q, s if scales is None else scales) if mxfp8 else (k2m._bf16(label), None)
    if staging is None and mxfp8:
        staging = k2m._staging()
    y = oracle.empty((m, n), torch.bfloat16)
    info = torch.zeros((1,), dtype=torch.int64)
    us = torch.zeros((2,), dtype=torch.float64)
    oracle.call("cere_ktest_k2m_plan", x, w, sc, m, n, k, rows, k2m.FUSED_ALL, k2m.DEFAULT_PATHS, 0, small_m, cls,
                staging, k2m._staging_bytes(staging), iters, y, info, us)
    return y, int(info[0]), float(us[0]), float(us[1])


def _segments(labels: tuple[str, ...], m: int, mxfp8: tuple[bool, ...],
              scales: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    """gemm_segments under small_m_a8 on k2m._x(m, k) (weight i MXFP8 when mxfp8[i]); the outputs flattened."""
    k = k2m.SHAPES[labels[0]][1]
    ws, ss, ys = [], [], []
    for label, mx in zip(labels, mxfp8):
        q, s = k2m._mxfp8(label)
        if scales is not None and label in scales:
            s = scales[label]
        ws.append(q if mx else k2m._bf16(label))
        ss.append(s.data_ptr() if mx else 0)
        ys.append(oracle.empty((m, k2m.SHAPES[label][0]), torch.bfloat16))
    staging = k2m._staging()
    oracle.call("cere_ktest_k2m_segments", k2m._x(m, k), m, k, len(labels),
                torch.tensor([w.data_ptr() for w in ws], dtype=torch.int64), torch.tensor(ss, dtype=torch.int64),
                torch.tensor([k2m.SHAPES[label][0] for label in labels], dtype=torch.int64),
                torch.tensor([y.data_ptr() for y in ys], dtype=torch.int64), 0, 0, SMALL_M_A8, 0, staging,
                k2m._staging_bytes(staging), 1, None)
    return torch.cat([y.reshape(-1) for y in ys])


def _a8_staging_bytes(m: int, n: int, k: int) -> int:
    """GemmPlan::a8_staging_bytes: x's E4M3 bytes, its blocked scales, the weight's blocked scales (each 256-aligned)."""
    def up(v: int) -> int:
        return (v + A8_ALIGN - 1) // A8_ALIGN * A8_ALIGN
    cols = k // GROUP_K
    return up(m * k) + up(_blocked_bytes(m, cols)) + _blocked_bytes(n, cols)


class K2mA8(unittest.TestCase):
    def test_quantize_x(self) -> None:
        cases = [(f"quantize/k{k}/m{m}", k2m._x(m, k)) for k in sorted({k for _, k in k2m.SHAPES.values()})
                 for m in QUANTIZE_M]
        cases.append(("quantize/edges", _edges()))
        for case, x in cases:
            with self.subTest(case=case):
                m, k = x.shape
                q, buf = _quantize_native(x)
                q_ref, e_ref = _quantize_ref(x)
                scales, rest = _read_blocked(buf, m, k // GROUP_K)
                oracle.check_bit_exact(self, ROW, f"{case}/q", q, q_ref)
                oracle.check_bit_exact(self, ROW, f"{case}/scales", scales, (e_ref + 127).to(torch.uint8))
                self.assertTrue(bool((rest == 0xFF).all()), f"{case}: a byte outside the blocked offsets was written")

    def test_repack_scales(self) -> None:
        for n, k in sorted(set(k2m.SHAPES.values())):
            case = f"repack/n{n}_k{k}"
            with self.subTest(case=case):
                s = k2m._mxfp8(next(label for label, shape in k2m.SHAPES.items() if shape == (n, k)))[1]
                buf = oracle.empty((_blocked_bytes(n, k // GROUP_K),), torch.uint8)
                oracle.call("cere_ktest_k2m_repack_scales", s, n, k, buf)
                scales, rest = _read_blocked(buf, n, k // GROUP_K)
                oracle.check_bit_exact(self, ROW, case, scales, s)
                self.assertTrue(bool((rest == 0xFF).all()), f"{case}: a byte outside the blocked offsets was written")

    def test_a8_oracle(self) -> None:
        """Every MXFP8 shape takes the W8A8 path above m 16, within the calibrated limit of the FP64 product of the
        dequantized operands (the lever registry's oracle test of small_m_a8)."""
        for label, (n, k) in k2m.SHAPES.items():
            w = k2m._bf16(label)
            for m in A8_M:
                case = f"a8/{label}/m{m}"
                with self.subTest(case=case):
                    x = k2m._x(m, k)
                    xd = _dequantize_x(*_quantize_ref(x))
                    high = (xd.to(torch.float64) @ w.to(torch.float64).T).to(torch.float32)
                    reference = torch.matmul(xd.to(torch.bfloat16), w.T)
                    c = oracle.calibrate(ROW, case, reference.to(torch.float32), high, oracle.FLOOR_OP)
                    y, info, _, _ = _call(label, x, mxfp8=True, small_m=SMALL_M_A8)
                    self.assertEqual(info, A8_PATH, f"{case}: the call does not take the W8A8 path")
                    noise = oracle.rel_l2(y.to(torch.float32), _call(label, x, mxfp8=False, small_m=SMALL_M)[0]
                                          .to(torch.float32))
                    oracle.report_row(ROW, case, phase="noise", n=n, k=k, m=m, noise=noise)
                    oracle.check_within(self, ROW, case, oracle.rel_l2(y.to(torch.float32), high), c=c,
                                        limit=oracle.calibrated_limit(c, oracle.FLOOR_OP))

    def test_a8_cls_and_repeat(self) -> None:
        for label, (_, k) in k2m.SHAPES.items():
            x = k2m._x(CLS, k)
            with self.subTest(case=f"a8/{label}/cls"):
                part, info, _, _ = _call(label, x[:CLS_M].contiguous(), mxfp8=True, small_m=SMALL_M_A8, cls=CLS)
                self.assertEqual(info, A8_PATH, f"{label}: the cls call does not take the W8A8 path")
                whole = _call(label, x, mxfp8=True, small_m=SMALL_M_A8)[0]
                oracle.check_bit_exact(self, ROW, f"a8/{label}/cls", part, whole[:CLS_M].contiguous())
            with self.subTest(case=f"a8/{label}/repeat"):
                first = _call(label, x, mxfp8=True, small_m=SMALL_M_A8)[0]
                oracle.check_bit_exact(self, ROW, f"a8/{label}/repeat", first,
                                       _call(label, x, mxfp8=True, small_m=SMALL_M_A8)[0])

    def test_other_calls_are_small_m(self) -> None:
        for label, (_, k) in k2m.SHAPES.items():
            calls = [(f"init/m{m}", k2m._x(m, k), True, 0) for m in k2m.SMALL_M]
            calls += [(f"rows{k2m.ROWS}/m{m}", k2m._x(m, k), True, k2m.ROWS) for m in k2m.SMALL_M]
            calls += [(f"bf16/m{m}", k2m._x(m, k), False, 0) for m in SAME_LARGE_M]
            for name, x, mxfp8, rows in calls:
                case = f"same/{label}/{name}"
                with self.subTest(case=case):
                    a8, info, _, _ = _call(label, x, mxfp8=mxfp8, small_m=SMALL_M_A8, rows=rows)
                    self.assertNotEqual(info, A8_PATH, f"{case}: the call takes the W8A8 path")
                    oracle.check_bit_exact(self, ROW, case, a8,
                                           _call(label, x, mxfp8=mxfp8, small_m=SMALL_M, rows=rows)[0])

    def test_segments(self) -> None:
        for group, labels in k2m.SEGMENTS.items():
            k = k2m.SHAPES[labels[0]][1]
            # gdn_in: qkv and a MXFP8 (one shared quantize_x), z and b BF16; shared_gate_up: gate alone MXFP8.
            mixed = tuple(i % 3 == 0 for i in range(len(labels)))
            for m in SEGMENT_M:
                for name, mx in (("mxfp8", (True,) * len(labels)), ("mixed", mixed)):
                    case = f"segments/{group}/m{m}/{name}"
                    with self.subTest(case=case):
                        x = k2m._x(m, k)
                        per_plan = [_call(label, x, mxfp8=f, small_m=SMALL_M_A8)[0].reshape(-1)
                                    for label, f in zip(labels, mx)]
                        oracle.check_bit_exact(self, ROW, case, _segments(labels, m, mx), torch.cat(per_plan))

    def test_staging_refusal(self) -> None:
        label, m = "attn_o", 64
        n, k = k2m.SHAPES[label]
        need = _a8_staging_bytes(m, n, k)
        x = k2m._x(m, k)
        q, s = k2m._mxfp8(label)
        for name, size, expected in (("short", need - 1, oracle.CERE_ERR_UNSUPPORTED), ("exact", need, 0)):
            with self.subTest(staging=name):
                staging = torch.empty(size, dtype=torch.uint8, device=oracle.DEVICE)
                y = oracle.empty((m, n), torch.bfloat16)
                status = oracle.call_status("cere_ktest_k2m_plan", x, q, s, m, n, k, 0, k2m.FUSED_ALL,
                                            k2m.DEFAULT_PATHS, 0, SMALL_M_A8, 0, staging, size, 1, y, None, None)
                oracle.report_row(ROW, f"refusal/{label}/{name}", phase="refusal", bytes=size, status=status)
                self.assertEqual(status, expected, f"staging {name} ({size} bytes)")

    def test_perturbed_scale_is_detected(self) -> None:
        """Fail-first: one weight scale byte + 1 changes the a8 call (plan and segments)."""
        for label in ("attn_o", "gdn_qkv", "shared_down"):
            scales, j, c = k2m._perturbed(label)
            k = k2m.SHAPES[label][1]
            for m in (64, 1600):
                case = f"perturbed/{label}/m{m}"
                with self.subTest(case=case):
                    x = k2m._x(m, k)
                    native = _call(label, x, mxfp8=True, small_m=SMALL_M_A8, scales=scales)[0]
                    differing = k2m._differing(native, _call(label, x, mxfp8=True, small_m=SMALL_M_A8)[0])
                    oracle.report_row(ROW, case, phase="perturbed", row_j=j, group=c, differing=differing,
                                      detected=differing > 0)
                    self.assertNotEqual(differing, 0, f"{case}: a perturbed scale byte went unnoticed")
        labels = k2m.SEGMENTS["gdn_in"]
        case = "perturbed/segments/gdn_in/m64"
        with self.subTest(case=case):
            native = _segments(labels, 64, (True,) * 4, scales={"gdn_z": k2m._perturbed("gdn_z")[0]})
            differing = k2m._differing(native, _segments(labels, 64, (True,) * 4))
            oracle.report_row(ROW, case, phase="perturbed", differing=differing, detected=differing > 0)
            self.assertNotEqual(differing, 0, f"{case}: a perturbed scale byte went unnoticed")

    def test_cold_l2_time(self) -> None:
        """Report-only (no verdict): cold-L2 us per shape and m (module docstring)."""
        for label, (n, k) in k2m.SHAPES.items():
            for m in TIME_M:
                with self.subTest(shape=label, m=m):
                    x = k2m._x(m, k)
                    _, _, a8_us, a8_min = _call(label, x, mxfp8=True, small_m=SMALL_M_A8, iters=k2m.COLD_ITERS)
                    _, _, st_us, st_min = _call(label, x, mxfp8=True, small_m=SMALL_M, iters=k2m.COLD_ITERS)
                    _, _, bf_us, bf_min = _call(label, x, mxfp8=False, small_m=SMALL_M, iters=k2m.COLD_ITERS)
                    oracle.report_row(ROW, f"time/{label}/m{m}", phase="time", n=n, k=k, m=m, a8_us=a8_us,
                                      a8_min_us=a8_min, staging_us=st_us, staging_min_us=st_min,
                                      bf16_cublaslt_us=bf_us, bf16_cublaslt_min_us=bf_min,
                                      a8_over_staging=a8_us / st_us)


if __name__ == "__main__":
    unittest.main()
