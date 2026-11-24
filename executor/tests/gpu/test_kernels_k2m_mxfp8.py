"""K2M: MXFP8 dense GEMMs of the S3 FP4-Local artifact (track Fp4S3, tasks/fp4local-spec.md "S3 perf plan", FL-30;
`kernels/k2_mxfp8.h`, `kernels/k2_small_m.h` gemm_mxfp8, GemmPlan::use_mxfp8).

Suite `kernels_k2m` (GPU1 lane: `native_dev.py --gpu 1 test --suite kernels_k2m`):
`python3 -m unittest discover -s /src/executor/tests/gpu -p 'test_kernels_k2m_*.py' -v`.

Contract: every GemmPlan call on an MXFP8 weight (q E4M3 [n, k], UE8M0 scales [n, k / 32], value q * 2^(s - 127))
returns the bits of the same call on the weight's exact BF16 upcast (the tensor `cerec convert`'s S2 path stores).
So every row here is bit-exact (raw bytes) against the BF16 call of the same plan on the torch upcast:
- `dequantize/<shape>` and `dequantize/codes`: cere_ktest_k2m_dequantize equals the torch upcast
  (torch.float8_e4m3fn * 2^(s - 127), exact in FP32 and BF16); `codes` is every non-NaN E4M3 byte under every scale
  byte 10..246.
- `<shape>/init/small_m[/f32]`, `<shape>/init/cublaslt[/f32]`: init plans of max_m = m, the small-M lever on (the
  MXFP8 operand variant of the small-M kernel) and off (cuBLASLt buckets and pins on the staging upcast), m 1..16.
- `<shape>/rows16/small_m`, `<shape>/rows16/cublaslt`: the executor's rows plans (init_rows(16)), m 1..16 (the
  BF16 call's path per m - gemv, strided-batched, per-row or small-M - is reported).
- `<shape>/large`: m 17, 64, 512, 4,096 (staging), and `<shape>/cls`: m 1 / 4 chosen for cls 17 / 64.
- `segments/<group>/...`: kernels::gemm_segments (GDN in_proj qkv, z, b, a and the shared expert gate, up) on init
  and rows plans, the small-M lever on (one gemm_mxfp8 launch) and off, all weights MXFP8 and a mix of MXFP8 and BF16.
Shapes are the S3 artifact's MXFP8 projections (attention q/k/v packed, o_proj, indexer.index_qk_proj
((indexer_n_heads + indexer_kv_heads) x indexer_head_dim = 640 rows, model.cpp); GDN in_proj_qkv / z / a / b and
out_proj; the shared expert gate / up / down) with synthetic seeded weights: E4M3 bytes uniform over the non-NaN codes
(subnormals and +-0 included, some groups all +-0), scale bytes 59..120 with groups at the extremes 10 and 246 (the
246 groups hold subnormal codes only, so no output overflows).

Fail-first: `perturbed/*` runs the same comparisons with one scale byte + 1 on the MXFP8 side (small-M, staging and
segments paths) and requires a difference; against a library without these entry points every test fails (symbol
absent).

Report-only rows `time/<shape>/m<m>` (m 1, 4, 16): cold-L2 (ktest/l2_flush.h) median and minimum us of the MXFP8
small-M call vs the BF16 small-M call, and of the MXFP8 staging call vs the BF16 cuBLASLt call, with GB/s of the bytes
each reads (weights + scales + x + y; staging: the dequantize's E4M3 + scales and the GEMM's BF16 staging read).

Native entry points (C prototypes; every call returns 0 on success, nonzero `cere_status` otherwise):

  int cere_ktest_k2m_dequantize(const void *q,        /* uint8 E4M3 [n, k], 16-byte aligned */
                                const void *scales,   /* uint8 UE8M0 [n, k / 32] */
                                int64_t n, int64_t k, /* k a multiple of 32 */
                                void *w,              /* bfloat16 [n, k] */
                                cudaStream_t stream);

  int cere_ktest_k2m_plan(const void *x,              /* bfloat16 [m, k] */
                          const void *w,              /* scales null: bfloat16 [n, k]; else uint8 E4M3 [n, k] */
                          const void *scales,         /* null or uint8 UE8M0 [n, k / 32] */
                          int64_t m, int64_t n, int64_t k,
                          int64_t rows,               /* 0: an init plan (max_m = max(m, cls)); [1, 16]: rows plan */
                          int64_t fused_allowed, int64_t paths,  /* init_rows arguments (paths -1: default) */
                          int64_t out_f32, int64_t small_m,  /* small_m 2: small_m_a8 (test_kernels_k2m_a8.py) */
                          int64_t cls,                /* 0 or >= m: the row count the plan chooses for */
                          void *staging, int64_t staging_bytes,  /* the MXFP8 table's staging buffer (nullable) */
                          int64_t iters,              /* 1: one call; > 1: 1 + (iters - 1) cold-L2 calls */
                          void *y,                    /* bfloat16 or float32 [m, n] */
                          void *info,                 /* host int64 [1] (nullable): the BF16 path, 0 cuBLASLt, 1
                                                         per-row, 2 gemv, 3 strided-batched, 4 small-M; 5 W8A8 */
                          void *us,                   /* host float64 [2]: median, min us (iters > 1) */
                          cudaStream_t stream);

  int cere_ktest_k2m_segments(const void *x, int64_t m, int64_t k, int64_t count,
                              const int64_t *w_ptrs,      /* host int64 [count] */
                              const int64_t *scale_ptrs,  /* host int64 [count]: 0 = BF16 weight */
                              const int64_t *ns,          /* host int64 [count] */
                              const int64_t *y_ptrs,      /* host int64 [count] */
                              int64_t rows, int64_t out_f32, int64_t small_m,
                              int64_t cls,                /* 0 or >= m (rows 0 only), as cere_ktest_k2m_plan */
                              void *staging, int64_t staging_bytes, int64_t iters, void *us, cudaStream_t stream);
"""
from __future__ import annotations

import functools
import unittest

import torch

import oracle

ROW = "K2M"
SEED = 3_100_000
GROUP_K = 32
MIN_SCALE = 10
MAX_SCALE = 246
ROWS = 16  # GemmPlan::kMaxRows
SMALL_M = tuple(range(1, ROWS + 1))
LARGE_M = (17, 64, 512, 4096)
TIME_M = (1, 4, 16)
COLD_ITERS = 41  # 1 untimed + 40 cold-L2 calls
FUSED_ALL = 0xFFFFFFFF  # init_rows' default fused_allowed
DEFAULT_PATHS = -1
SHAPES = {  # label: (n, k), the S3 artifact's MXFP8 weights
    "attn_qkv": (13312, 2560),
    "attn_o": (2560, 6144),
    "attn_index_qk": (640, 2560),
    "gdn_qkv": (10240, 2560),
    "gdn_z": (6144, 2560),
    "gdn_b": (48, 2560),
    "gdn_a": (48, 2560),
    "gdn_out": (2560, 6144),
    "shared_gate": (640, 2560),
    "shared_up": (640, 2560),
    "shared_down": (2560, 640),
}
SEGMENTS = {"gdn_in": ("gdn_qkv", "gdn_z", "gdn_b", "gdn_a"), "shared_gate_up": ("shared_gate", "shared_up")}
PATHS = {0: "cublaslt", 1: "per_row", 2: "gemv", 3: "batched", 4: "small_m"}


def _scales(n: int, groups: int, g: torch.Generator) -> torch.Tensor:
    """Scale bytes 59..120; rows j = 2 mod 7 hold groups of scale 10 (c = j mod 5), rows j = 3 mod 11 one group of
    scale 246 (c = j mod groups), so most rows stay in the common range and a 246 group dominates its row alone."""
    s = torch.randint(59, 121, (n, groups), generator=g, device=oracle.DEVICE, dtype=torch.int32)
    j = torch.arange(n, device=oracle.DEVICE, dtype=torch.int32)[:, None]
    c = torch.arange(groups, device=oracle.DEVICE, dtype=torch.int32)[None, :]
    s = torch.where((j % 7 == 2) & (c % 5 == j % 5), MIN_SCALE, s)
    return torch.where((j % 11 == 3) & (c == j % groups), MAX_SCALE, s)


@functools.cache
def _mxfp8(label: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(q uint8 [n, k], s uint8 [n, k / 32]) of the module docstring's synthetic weight."""
    n, k = SHAPES[label]
    g = oracle.generator(SEED + list(SHAPES).index(label))
    s = _scales(n, k // GROUP_K, g)
    q = torch.randint(0, 256, (n, k), generator=g, device=oracle.DEVICE, dtype=torch.int32)
    q = torch.where((q & 0x7F) == 0x7F, q & 0x80, q)  # no NaN byte: 0x7F / 0xFF -> +-0
    j = torch.arange(n, device=oracle.DEVICE, dtype=torch.int32)[:, None]
    c = torch.arange(k, device=oracle.DEVICE, dtype=torch.int32)[None, :] // GROUP_K
    q = torch.where(s.repeat_interleave(GROUP_K, dim=1) == MAX_SCALE, q & 0x87, q)  # subnormals and +-0 only
    q = torch.where((j + 5 * c) % 17 == 2, q & 0x80, q)  # groups of +-0
    return q.to(torch.uint8).contiguous(), s.to(torch.uint8).contiguous()


def _upcast(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """The exact BF16 upcast: q * 2^(s - 127) is a normal FP32 with at most 4 significant bits, so exact in BF16.
    2^(s - 127) is built from its bits (biased exponent s), not by a transcendental."""
    scale = (s.to(torch.int32) << 23).view(torch.float32).repeat_interleave(GROUP_K, dim=1)
    return (q.view(torch.float8_e4m3fn).to(torch.float32) * scale).to(torch.bfloat16).contiguous()


@functools.cache
def _bf16(label: str) -> torch.Tensor:
    return _upcast(*_mxfp8(label))


@functools.cache
def _x(m: int, k: int) -> torch.Tensor:
    g = oracle.generator(SEED + 7919 * m + k)
    return torch.randn((m, k), generator=g, device=oracle.DEVICE, dtype=torch.float32).to(torch.bfloat16)


@functools.cache
def _staging() -> torch.Tensor:
    return torch.empty(max(n * k for n, k in SHAPES.values()), dtype=torch.bfloat16, device=oracle.DEVICE)


def _staging_bytes(staging: torch.Tensor | None) -> int:
    return 0 if staging is None else staging.numel() * staging.element_size()


def _plan(label: str, m: int, *, mxfp8: bool, rows: int = 0, small_m: bool = True, out_f32: bool = False,
          cls: int = 0, scales: torch.Tensor | None = None, staging: torch.Tensor | None = None,
          iters: int = 1) -> tuple[torch.Tensor, int, float, float]:
    """(y, the BF16 path code, median us, min us) of one cere_ktest_k2m_plan call on `label`'s weight (MXFP8 with
    its scales, or `scales` instead, when mxfp8; else the BF16 upcast)."""
    n, k = SHAPES[label]
    q, s = _mxfp8(label)
    w, sc = (q, s if scales is None else scales) if mxfp8 else (_bf16(label), None)
    if staging is None and mxfp8:
        staging = _staging()
    y = oracle.empty((m, n), torch.float32 if out_f32 else torch.bfloat16)
    info = torch.zeros((1,), dtype=torch.int64)
    us = torch.zeros((2,), dtype=torch.float64)
    oracle.call("cere_ktest_k2m_plan", _x(m, k), w, sc, m, n, k, rows, FUSED_ALL, DEFAULT_PATHS, int(out_f32),
                int(small_m), cls, staging, _staging_bytes(staging), iters, y, info, us)
    return y, int(info[0]), float(us[0]), float(us[1])


def _plan_status(label: str, m: int, small_m: bool, staging: torch.Tensor | None, staging_bytes: int) -> int:
    n, k = SHAPES[label]
    q, s = _mxfp8(label)
    y = oracle.empty((m, n), torch.bfloat16)
    return oracle.call_status("cere_ktest_k2m_plan", _x(m, k), q, s, m, n, k, 0, FUSED_ALL, DEFAULT_PATHS, 0,
                              int(small_m), 0, staging, staging_bytes, 1, y, None, None)


def _segments(labels: tuple[str, ...], m: int, mxfp8: tuple[bool, ...], *, rows: int = 0, small_m: bool = True,
              scales: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    """gemm_segments over `labels` (weight i MXFP8 when mxfp8[i], else its BF16 upcast); the outputs flattened."""
    k = SHAPES[labels[0]][1]
    ws, ss, ys = [], [], []
    for label, mx in zip(labels, mxfp8):
        q, s = _mxfp8(label)
        if scales is not None and label in scales:
            s = scales[label]
        ws.append(q if mx else _bf16(label))
        ss.append(s.data_ptr() if mx else 0)
        ys.append(oracle.empty((m, SHAPES[label][0]), torch.bfloat16))
    staging = _staging()
    oracle.call("cere_ktest_k2m_segments", _x(m, k), m, k, len(labels),
                torch.tensor([w.data_ptr() for w in ws], dtype=torch.int64), torch.tensor(ss, dtype=torch.int64),
                torch.tensor([SHAPES[label][0] for label in labels], dtype=torch.int64),
                torch.tensor([y.data_ptr() for y in ys], dtype=torch.int64), rows, 0, int(small_m), 0, staging,
                _staging_bytes(staging), 1, None)
    return torch.cat([y.reshape(-1) for y in ys])


def _differing(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a.contiguous().view(torch.uint8).reshape(a.numel(), -1)
                != b.contiguous().view(torch.uint8).reshape(b.numel(), -1)).any(dim=-1).sum().item())


def _perturbed(label: str) -> tuple[torch.Tensor, int, int]:
    """`label`'s scales with one byte + 1: a group of row j holding a nonzero code whose scale is its row's largest
    (at most 120), so the change reaches y[., j] (scales, j, group)."""
    q, s = _mxfp8(label)
    nonzero = (q.to(torch.int32).view(-1, GROUP_K) & 0x7F).amax(dim=1).view(s.shape) != 0
    row_max = s.to(torch.int32).amax(dim=1, keepdim=True)
    candidates = ((s.to(torch.int32) == row_max) & (row_max <= 120) & nonzero).nonzero()
    j, c = (int(v) for v in candidates[len(candidates) // 3])
    p = s.clone()
    p[j, c] += 1
    return p, j, c


class Mxfp8Gemm(unittest.TestCase):
    def test_dequantize_matches_the_upcast(self) -> None:
        for label, (n, k) in SHAPES.items():
            with self.subTest(shape=label):
                q, s = _mxfp8(label)
                w = oracle.empty((n, k), torch.bfloat16)
                oracle.call("cere_ktest_k2m_dequantize", q, s, n, k, w)
                oracle.check_bit_exact(self, ROW, f"dequantize/{label}", w, _bf16(label))

    def test_dequantize_every_code_and_scale(self) -> None:
        codes = torch.arange(256, device=oracle.DEVICE, dtype=torch.int32)
        codes = torch.where((codes & 0x7F) == 0x7F, codes & 0x80, codes)
        n, k = MAX_SCALE - MIN_SCALE + 1, 256
        q = codes.to(torch.uint8).expand(n, k).contiguous()
        s = torch.arange(MIN_SCALE, MAX_SCALE + 1, device=oracle.DEVICE, dtype=torch.int32)
        s = s.to(torch.uint8)[:, None].expand(n, k // GROUP_K).contiguous()
        w = oracle.empty((n, k), torch.bfloat16)
        oracle.call("cere_ktest_k2m_dequantize", q, s, n, k, w)
        oracle.check_bit_exact(self, ROW, "dequantize/codes", w, _upcast(q, s))

    def test_init_plans(self) -> None:
        for label in SHAPES:
            for small_m in (True, False):
                for out_f32 in (False, True):
                    case = f"{label}/init/{'small_m' if small_m else 'cublaslt'}{'/f32' if out_f32 else ''}"
                    with self.subTest(case=case):
                        mx = [_plan(label, m, mxfp8=True, small_m=small_m, out_f32=out_f32)[0] for m in SMALL_M]
                        bf = [_plan(label, m, mxfp8=False, small_m=small_m, out_f32=out_f32)[0] for m in SMALL_M]
                        oracle.check_bit_exact(self, ROW, case, torch.cat([y.reshape(-1) for y in mx]),
                                               torch.cat([y.reshape(-1) for y in bf]))

    def test_rows_plans(self) -> None:
        for label in SHAPES:
            for small_m in (True, False):
                case = f"{label}/rows{ROWS}/{'small_m' if small_m else 'cublaslt'}"
                with self.subTest(case=case):
                    mx, bf, paths = [], [], []
                    for m in SMALL_M:
                        mx.append(_plan(label, m, mxfp8=True, rows=ROWS, small_m=small_m)[0].reshape(-1))
                        y, path, _, _ = _plan(label, m, mxfp8=False, rows=ROWS, small_m=small_m)
                        bf.append(y.reshape(-1))
                        paths.append(PATHS[path])
                    oracle.report_row(ROW, case, phase="paths", **{f"m{m}_path": p for m, p in zip(SMALL_M, paths)})
                    oracle.check_bit_exact(self, ROW, case, torch.cat(mx), torch.cat(bf))

    def test_large_m_and_cls_staging(self) -> None:
        for label in SHAPES:
            for m in LARGE_M:
                with self.subTest(shape=label, m=m):
                    oracle.check_bit_exact(self, ROW, f"{label}/large/m{m}", _plan(label, m, mxfp8=True)[0],
                                           _plan(label, m, mxfp8=False)[0])
            for m, cls in ((1, 17), (4, 64)):
                with self.subTest(shape=label, m=m, cls=cls):
                    mx = _plan(label, m, mxfp8=True, cls=cls)
                    bf = _plan(label, m, mxfp8=False, cls=cls)
                    self.assertEqual(bf[1], 0, f"{label} m{m} cls{cls}: the BF16 call is not cuBLASLt")
                    oracle.check_bit_exact(self, ROW, f"{label}/cls/m{m}_cls{cls}", mx[0], bf[0])

    def test_segments(self) -> None:
        for group, labels in SEGMENTS.items():
            mixed = tuple(i == 0 for i in range(len(labels)))
            for rows in (0, ROWS):
                for small_m in (True, False):
                    for name, mx in (("mxfp8", (True,) * len(labels)), ("mixed", mixed)):
                        case = f"segments/{group}/rows{rows}/{'small_m' if small_m else 'cublaslt'}/{name}"
                        with self.subTest(case=case):
                            native = [_segments(labels, m, mx, rows=rows, small_m=small_m) for m in SMALL_M]
                            reference = [_segments(labels, m, (False,) * len(labels), rows=rows, small_m=small_m)
                                         for m in SMALL_M]
                            oracle.check_bit_exact(self, ROW, case, torch.cat(native), torch.cat(reference))

    def test_staging_refusals(self) -> None:
        n, k = SHAPES["attn_o"]
        short = torch.empty(n * k - 8, dtype=torch.bfloat16, device=oracle.DEVICE)
        for name, staging, size in (("null", None, 0), ("short", short, _staging_bytes(short))):
            with self.subTest(staging=name):
                self.assertEqual(_plan_status("attn_o", 4, False, staging, size), oracle.CERE_ERR_UNSUPPORTED)
                self.assertEqual(_plan_status("attn_o", 4, True, staging, size), 0)

    def test_perturbed_scale_is_detected(self) -> None:
        """Fail-first: the comparisons above catch one scale byte + 1 on the MXFP8 side."""
        for label in ("attn_o", "gdn_qkv", "shared_down"):
            scales, j, c = _perturbed(label)
            for small_m, m in ((True, 4), (False, 4), (True, 1), (False, 64)):
                case = f"perturbed/{label}/{'small_m' if small_m else 'cublaslt'}/m{m}"
                with self.subTest(case=case):
                    native = _plan(label, m, mxfp8=True, small_m=small_m, scales=scales)[0]
                    differing = _differing(native, _plan(label, m, mxfp8=False, small_m=small_m)[0])
                    oracle.report_row(ROW, case, phase="perturbed", row_j=j, group=c, differing=differing,
                                      detected=differing > 0)
                    self.assertNotEqual(differing, 0, f"{case}: a perturbed scale byte went unnoticed")
        labels = SEGMENTS["gdn_in"]
        scales = {"gdn_z": _perturbed("gdn_z")[0]}
        for small_m in (True, False):
            case = f"perturbed/segments/gdn_in/{'small_m' if small_m else 'cublaslt'}/m4"
            with self.subTest(case=case):
                native = _segments(labels, 4, (True,) * 4, small_m=small_m, scales=scales)
                differing = _differing(native, _segments(labels, 4, (False,) * 4, small_m=small_m))
                oracle.report_row(ROW, case, phase="perturbed", differing=differing, detected=differing > 0)
                self.assertNotEqual(differing, 0, f"{case}: a perturbed scale byte went unnoticed")

    def test_cold_l2_time(self) -> None:
        """Report-only (no verdict): cold-L2 us and GB/s per shape and m (module docstring)."""
        for label, (n, k) in SHAPES.items():
            for m in TIME_M:
                with self.subTest(shape=label, m=m):
                    io = 2 * m * k + 2 * m * n
                    mx_bytes = n * k + n * k // GROUP_K + io
                    bf_bytes = 2 * n * k + io
                    staging_bytes = n * k + n * k // GROUP_K + 2 * n * k + io
                    _, _, mx_us, mx_min = _plan(label, m, mxfp8=True, iters=COLD_ITERS)
                    _, _, bf_us, bf_min = _plan(label, m, mxfp8=False, iters=COLD_ITERS)
                    _, _, st_us, st_min = _plan(label, m, mxfp8=True, small_m=False, iters=COLD_ITERS)
                    _, _, cb_us, cb_min = _plan(label, m, mxfp8=False, small_m=False, iters=COLD_ITERS)
                    oracle.report_row(ROW, f"time/{label}/m{m}", phase="time", n=n, k=k, m=m,
                                      mxfp8_small_m_us=mx_us, mxfp8_small_m_min_us=mx_min,
                                      mxfp8_small_m_gbps=mx_bytes / mx_us / 1e3,
                                      bf16_small_m_us=bf_us, bf16_small_m_min_us=bf_min,
                                      bf16_small_m_gbps=bf_bytes / bf_us / 1e3,
                                      mxfp8_staging_us=st_us, mxfp8_staging_min_us=st_min,
                                      mxfp8_staging_gbps=staging_bytes / st_us / 1e3,
                                      bf16_cublaslt_us=cb_us, bf16_cublaslt_min_us=cb_min,
                                      bf16_cublaslt_gbps=bf_bytes / cb_us / 1e3,
                                      small_m_speedup=bf_us / mx_us)


if __name__ == "__main__":
    unittest.main()
