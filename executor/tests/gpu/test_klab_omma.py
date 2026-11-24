"""KLAB (tasks/m215-spec.md D2): the GPU1 kernel-lab calibration of the NVFP4 tensor-core instruction
`mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` (SASS
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`) against the D3 instruction model `omma_ref`, plus the F5 measured rows.

Runs only in the pinned image on GPU1, through `native_dev.py --gpu 1 test --suite klab` (D1: `cerelab-g1-*`, the
attribution check lists exactly the GPU1 UUID before any test). Inputs are seeded synthetic codes; nothing is read
from the model directory.

Fragment layout (F2), lane t = 0..31, g = t / 4, q = t % 4, nibble i = 0..7 of a register (bits 4i..4i+3):
- A (16 x 64 e2m1): a0 = A[g][q*8+i], a1 = A[g+8][q*8+i], a2 = A[g][32+q*8+i], a3 = A[g+8][32+q*8+i].
- B (64 x 8 e2m1, k x col): b0 = B[q*8+i][g], b1 = B[32+q*8+i][g].
- C/D (16 x 8 FP32): c_i = C[g + 8*(i/2)][q*2 + i%2].
- Scales: byte j of a scale register scales K block j (k in [16j, 16j+16)); which lane's register serves a row of
  A or a column of B under the selectors is what row (d) finds.

C prototypes (libcere_ktest.so; executor/cuda/src/ktest/klab.cu):

    int cere_ktest_klab_omma(const uint32_t *a, const uint32_t *b, const uint32_t *scale_a, const uint32_t *scale_b,
                             const float *c, float *d, int64_t n, int64_t byte_id_a, int64_t thread_id_a,
                             int64_t byte_id_b, int64_t thread_id_b, cudaStream_t stream);
        n instructions, instruction i on warp i, every operand the raw per-lane register (device, lane-major):
        a [n][32][4], b [n][32][2], scale_a and scale_b [n][32], c and d [n][32][4]. The selectors are the PTX
        immediates `{byte-id-a, thread-id-a}` and `{byte-id-b, thread-id-b}`; only PTX ISA 9.7.14.3 Table 37's
        scale_vec::4X values run (byte-id 0, thread-id-a {0, 1}, thread-id-b {0, 1, 2, 3}). Any other selector,
        n <= 0 or a misaligned or null pointer: CERE_ERR_ARGUMENT before any launch, d untouched.
    int cere_ktest_klab_calib(int64_t row, int64_t size, double *out, cudaStream_t stream);
        One measured row (F5) with cudaEvent timing: one untimed warm-up, then 5 timed repetitions. out (host
        double [7]): out[0] the work of one repetition, out[1..5] milliseconds per repetition, out[6] the launch
        shape. row 0 OMMA NVFP4 loop / row 1 HMMA BF16 loop (FLOP; SMs * size blocks of 128 threads), row 2 HBM read
        of 8 GiB (bytes; SMs * size blocks of 512 threads), row 3 grid.sync (barriers; size 0 = one block per SM,
        1 = full occupancy), row 4 CUDA graph replay of `size` no-op nodes (nodes; out[6] = host milliseconds of
        cudaGraphInstantiate), row 5 the same `size` no-op kernels launched one by one (nodes). Other rows or sizes:
        CERE_ERR_ARGUMENT.

Rows (ROW "KLAB"):
- (a) layout: unit scales, exact sums (C = 0 and small integer C), bit-exact against the exact FP64 matmul.
- (b) OMMA equals the D3 model (`omma_ref.dot64_t`, which (e) ties to the stdlib core) on F3's domain: scale codes
  24-87 per K block, |C| <= 10^6 with exponent >= -11, 8,192 instructions = 1,048,576 elements, bits compared.
- (c) characterization against the stdlib core (`omma_ref.ROUNDING`), bits compared: every ue4m3 code 0x00-0x7E in
  every block position for both operands, e2m1 +-0 (with zero, subnormal and wide C), C from 2^-126 to 2^127
  relative to the products (clamped to the FP32 range, subnormals included), and the sign of every zero result
  (C = -(the model's C = 0 result)). On a miss the report carries the counts (1 ulp, more, toward/away from zero,
  zero sign) as D3 evidence.
- (d) scale-lane mapping (F4): for each of the 8 valid selector pairs a one-hot sweep (one lane's byte j at 2.0,
  every other byte 1.0) finds the lane serving each A row and B column; each row/column has exactly one lane, every
  byte j acts on K block j only, and OMMA with distinct random per-lane scales then equals the D3 model under the
  found mapping. The PTX ISA quad rule (A: lane / 4 = row % 8, lane % 4 in {2 thread-id-a, 2 thread-id-a + 1};
  B: lane / 4 = col, lane % 4 = thread-id-b) is reported, not asserted.
- (e) the D3 torch path equals the stdlib core on a seeded 2^16-step sample: 32,768 single-instruction elements
  (half F3's domain, half (c)'s wide domain) and 2,048 chains of 16 steps (K = 1,024; 32,768 steps), each chain run
  on OMMA step by step (D of one step is C of the next); OMMA equals the model on the same sample.
- refusals: selectors outside Table 37 return CERE_ERR_ARGUMENT, d untouched; a valid pair runs.
- measured (report_row, never gated): F5's set, median of the 5 repetitions.
"""
from __future__ import annotations

import ctypes
import math
import random
import statistics
import unittest
from typing import NamedTuple

import torch

import oracle
import omma_ref

ROW = "KLAB"
CERE_ERR_ARGUMENT = -1  # cere_executor.h
SEED = 21_500
ONE = 0x38  # ue4m3 1.0
TWO = 0x40  # ue4m3 2.0
MAX_SCALE = 0x7E  # the characterized scale domain is 0x00-0x7E (D2)
F3_SCALES = (24, 88)  # probe domain: codes 24-87
SELECTORS = tuple((ta, tb) for ta in (0, 1) for tb in range(4))
MODEL_CHUNK = 1 << 17
CALIB_REPS = 5
GRAPH_NODES = (100, 1_000, 5_000, 20_000, 100_000)


# -- fragment packing (F2) --------------------------------------------------------------------------------------


def _layout() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flat indices: A [32,4,8] into A[16*64], B [32,2,8] into B^T[8*64] (col-major k), C [32,4] into C[16*8]."""
    a_idx, b_idx, c_idx = [], [], []
    for t in range(32):
        g, q = t // 4, t % 4
        a_idx.append([[(g + 8 * (j & 1)) * 64 + 32 * (j >> 1) + q * 8 + i for i in range(8)] for j in range(4)])
        b_idx.append([[g * 64 + 32 * j + q * 8 + i for i in range(8)] for j in range(2)])
        c_idx.append([(g + 8 * (i >> 1)) * 8 + q * 2 + (i & 1) for i in range(4)])
    return tuple(torch.tensor(x, dtype=torch.int64, device=oracle.DEVICE) for x in (a_idx, b_idx, c_idx))


A_IDX, B_IDX, C_IDX = _layout()
NIBBLE_SHIFT = torch.arange(0, 32, 4, dtype=torch.int64, device=oracle.DEVICE)
BYTE_SHIFT = torch.arange(0, 32, 8, dtype=torch.int64, device=oracle.DEVICE)


def _u32(v: torch.Tensor) -> torch.Tensor:
    """int64 values in [0, 2^32) as int32 with the same bits (the uint32 registers)."""
    return (v - (v >= 1 << 31).to(torch.int64) * (1 << 32)).to(torch.int32).contiguous()


class Batch(NamedTuple):
    """n instructions: e2m1 codes a [n,16,64] and bt [n,8,64] (B transposed: column, k), per-lane scale bytes
    sa, sb [n,32,4], C [n,16,8] float32; all uint8 except C, on the device."""
    a: torch.Tensor
    bt: torch.Tensor
    sa: torch.Tensor
    sb: torch.Tensor
    c: torch.Tensor


def omma(batch: Batch, thread_id_a: int = 0, thread_id_b: int = 0) -> torch.Tensor:
    """D [n,16,8] float32 (CPU) of `cere_ktest_klab_omma` on the batch, packed per F2."""
    n = batch.a.shape[0]
    a = _u32((batch.a.reshape(n, -1).to(torch.int64)[:, A_IDX] << NIBBLE_SHIFT).sum(-1))
    b = _u32((batch.bt.reshape(n, -1).to(torch.int64)[:, B_IDX] << NIBBLE_SHIFT).sum(-1))
    sa = _u32((batch.sa.to(torch.int64) << BYTE_SHIFT).sum(-1))
    sb = _u32((batch.sb.to(torch.int64) << BYTE_SHIFT).sum(-1))
    c = batch.c.reshape(n, -1)[:, C_IDX].contiguous()
    d = oracle.empty((n, 32, 4), torch.float32)
    oracle.call("cere_ktest_klab_omma", a, b, sa, sb, c, d, n, 0, thread_id_a, 0, thread_id_b)
    out = torch.empty((n, 128), dtype=torch.float32, device=oracle.DEVICE)
    out[:, C_IDX.reshape(-1)] = d.reshape(n, 128)
    return out.reshape(n, 16, 8).cpu()


# -- model inputs and the two D3 backends -----------------------------------------------------------------------


def uniform_lanes(scales: torch.Tensor) -> torch.Tensor:
    """[n,4] scale bytes held identically by all 32 lanes -> [n,32,4]."""
    return scales[:, None, :].expand(-1, 32, -1).contiguous()


def elements(batch: Batch, row_lane=None, col_lane=None) -> tuple[torch.Tensor, ...]:
    """Per output element (n, row, col order): a [M,64], b [M,64], sa [M,4], sb [M,4], c [M], with A row r scaled by
    lane row_lane[r]'s register and B column j by lane col_lane[j]'s (default lane 0: uniform lanes)."""
    n = batch.a.shape[0]
    rl = torch.tensor(row_lane if row_lane is not None else [0] * 16, dtype=torch.int64, device=batch.a.device)
    cl = torch.tensor(col_lane if col_lane is not None else [0] * 8, dtype=torch.int64, device=batch.a.device)
    a = batch.a[:, :, None, :].expand(n, 16, 8, 64).reshape(-1, 64)
    b = batch.bt[:, None, :, :].expand(n, 16, 8, 64).reshape(-1, 64)
    sa = batch.sa[:, rl, :][:, :, None, :].expand(n, 16, 8, 4).reshape(-1, 4)
    sb = batch.sb[:, cl, :][:, None, :, :].expand(n, 16, 8, 4).reshape(-1, 4)
    return a, b, sa, sb, batch.c.reshape(-1)


def model_core(a, b, sa, sb, c) -> torch.Tensor:
    """The D3 stdlib core (`omma_ref.dot64`), one element at a time -> float32 [M] (CPU)."""
    rows = zip(a.tolist(), b.tolist(), sa.tolist(), sb.tolist(), c.tolist())
    return torch.tensor([omma_ref.dot64(*r) for r in rows], dtype=torch.float32)


def model_torch(a, b, sa, sb, c) -> torch.Tensor:
    """The D3 torch path (`omma_ref.dot64_t`) in chunks -> float32 [M] (CPU)."""
    return torch.cat([omma_ref.dot64_t(a[i:i + MODEL_CHUNK], b[i:i + MODEL_CHUNK], sa[i:i + MODEL_CHUNK],
                                       sb[i:i + MODEL_CHUNK], c[i:i + MODEL_CHUNK]).cpu()
                      for i in range(0, a.shape[0], MODEL_CHUNK)])


def codes(g: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randint(0, 16, shape, generator=g, device=oracle.DEVICE, dtype=torch.uint8)


def scale_bytes(g: torch.Generator, n: int, lo: int, hi: int) -> torch.Tensor:
    return torch.randint(lo, hi, (n, 4), generator=g, device=oracle.DEVICE, dtype=torch.uint8)


def f3_c(g: torch.Generator, n: int) -> torch.Tensor:
    """F3's C: an integer in [-10^6, 10^6] times 2^-(0..11) (exact in FP32)."""
    ints = torch.randint(-1_000_000, 1_000_001, (n, 16, 8), generator=g, device=oracle.DEVICE, dtype=torch.int64)
    exps = torch.randint(0, 12, (n, 16, 8), generator=g, device=oracle.DEVICE, dtype=torch.int64)
    return torch.ldexp(ints.to(torch.float64), -exps).to(torch.float32)


def wide(rng: random.Random, magnitude: float, shift: int) -> float:
    """A random FP32 value 2^shift times the binade of `magnitude` (or of 1 when it is 0), clamped to the FP32
    range; below 2^-126 it lies on the subnormal grid."""
    e = (math.frexp(magnitude)[1] - 1 if magnitude else 0) + shift
    e = max(-149, min(127, e))
    m, lsb = rng.getrandbits(23) | (1 << 23), e - 23
    if lsb < -149:
        m, lsb = m >> (-149 - lsb), -149
    v = math.ldexp(m, lsb)
    return -v if rng.getrandbits(1) else v


def disagreement(native: torch.Tensor, reference: torch.Tensor) -> dict:
    """Counts of how `native` differs from `reference` bit-wise (FP32): the D3 amendment evidence."""
    nb = native.reshape(-1).view(torch.int32).to(torch.int64)
    rb = reference.reshape(-1).view(torch.int32).to(torch.int64)
    no = torch.where(nb < 0, -(nb & 0x7FFFFFFF), nb & 0x7FFFFFFF)
    ro = torch.where(rb < 0, -(rb & 0x7FFFFFFF), rb & 0x7FFFFFFF)
    diff = nb != rb
    dist = (no - ro).abs()
    zero_sign = diff & (dist == 0)
    away = diff & (native.reshape(-1).abs() > reference.reshape(-1).abs())
    toward = diff & (native.reshape(-1).abs() < reference.reshape(-1).abs())
    return {"elements": native.numel(), "differing": int(diff.sum()), "one_ulp": int((dist == 1).sum()),
            "over_one_ulp": int((dist > 1).sum()), "zero_sign": int(zero_sign.sum()),
            "away_from_zero": int(away.sum()), "toward_zero": int(toward.sum()),
            "native_nonfinite": int((~torch.isfinite(native)).sum())}


# -- (d) sweep analysis -----------------------------------------------------------------------------------------

SWEEP_BLOCK_CODES = (0x1, 0x2, 0x4, 0x6)  # e2m1 0.5, 1, 2, 4 in K blocks 0..3: block sums 8, 16, 32, 64
SWEEP_BASE = 120.0


def sweep_batch() -> Batch:
    """256 instructions: operand A (0..127) then B (128..255), instruction 4 * lane + j doubles lane's byte j."""
    a = torch.tensor(SWEEP_BLOCK_CODES, dtype=torch.uint8, device=oracle.DEVICE).repeat_interleave(16)
    a = a[None, None, :].expand(256, 16, 64).contiguous()
    bt = torch.full((256, 8, 64), 0x2, dtype=torch.uint8, device=oracle.DEVICE)
    sa = torch.full((256, 32, 4), ONE, dtype=torch.uint8, device=oracle.DEVICE)
    sb = sa.clone()
    for lane in range(32):
        for j in range(4):
            sa[4 * lane + j, lane, j] = TWO
            sb[128 + 4 * lane + j, lane, j] = TWO
    return Batch(a, bt, sa, sb, torch.zeros((256, 16, 8), dtype=torch.float32, device=oracle.DEVICE))


def sweep_lanes(d: torch.Tensor, operand: int) -> tuple[list[int], int]:
    """From the sweep's D [256,16,8]: the lane serving each A row (operand 0) or B column (operand 1), -1 when not
    exactly one lane; plus the number of malformed one-hot results (a partial row/column, a delta other than byte
    j's block sum, or a lane whose bytes act on different rows/columns)."""
    size = 16 if operand == 0 else 8
    owners: list[list[int]] = [[] for _ in range(size)]
    malformed = 0
    for lane in range(32):
        hit_sets = []
        for j in range(4):
            delta = d[128 * operand + 4 * lane + j] - SWEEP_BASE
            lines = delta if operand == 0 else delta.t()
            want = 8.0 * 2 ** j
            hit = [i for i in range(size) if bool((lines[i] != 0).any())]
            malformed += sum(1 for i in hit if not bool((lines[i] == want).all()))
            hit_sets.append(hit)
        if any(h != hit_sets[0] for h in hit_sets):
            malformed += 1
        for i in hit_sets[0]:
            owners[i].append(lane)
    return [o[0] if len(o) == 1 else -1 for o in owners], malformed


class KlabOmma(unittest.TestCase):
    """D2 (a)-(e) and the selector refusals."""

    def test_a_layout_exact_sums_under_unit_scales(self):
        g = oracle.generator(SEED)
        n = 256
        a, bt = codes(g, n, 16, 64), codes(g, n, 8, 64)
        ints = torch.randint(-1000, 1001, (n, 16, 8), generator=g, device=oracle.DEVICE, dtype=torch.int64)
        c = torch.where(torch.arange(n, device=oracle.DEVICE)[:, None, None] < n // 2, torch.zeros_like(ints), ints)
        ones = uniform_lanes(torch.full((n, 4), ONE, dtype=torch.uint8, device=oracle.DEVICE))
        batch = Batch(a, bt, ones, ones, c.to(torch.float32))
        values = torch.tensor([omma_ref.e2m1(i) for i in range(16)], dtype=torch.float64, device=oracle.DEVICE)
        exact = torch.einsum("nrk,nck->nrc", values[a.long()], values[bt.long()]) + c.to(torch.float64)
        oracle.check_bit_exact(self, ROW, "D2a layout unit scales exact sums", omma(batch),
                               exact.to(torch.float32).cpu())

    def test_b_omma_equals_model_on_f3_domain(self):
        g = oracle.generator(SEED + 1)
        n = 8192
        zero_c = torch.arange(n, device=oracle.DEVICE)[:, None, None] < n // 8
        c = torch.where(zero_c, torch.zeros((), device=oracle.DEVICE), f3_c(g, n))
        batch = Batch(codes(g, n, 16, 64), codes(g, n, 8, 64), uniform_lanes(scale_bytes(g, n, *F3_SCALES)),
                      uniform_lanes(scale_bytes(g, n, *F3_SCALES)), c)
        native = omma(batch)
        reference = model_torch(*elements(batch)).reshape(n, 16, 8)
        oracle.report_row(ROW, "D2b F3 domain disagreement", **disagreement(native, reference))
        oracle.check_bit_exact(self, ROW, "D2b F3 domain", native, reference)

    def test_c_characterization(self):
        g = oracle.generator(SEED + 2)
        rng = random.Random(SEED + 2)
        cases = {}
        # Every scale code 0x00-0x7E at every byte position of both operands, C = 0 then wide C.
        i = torch.arange(127, dtype=torch.int64, device=oracle.DEVICE)[:, None]
        j = torch.arange(4, dtype=torch.int64, device=oracle.DEVICE)[None, :]
        sa = torch.cat([(i + 32 * j) % 127] * 2).to(torch.uint8)
        sb = torch.cat([(5 * i + 17 * j + 3) % 127] * 2).to(torch.uint8)
        cases["scale codes"] = (Batch(codes(g, 254, 16, 64), codes(g, 254, 8, 64), uniform_lanes(sa),
                                      uniform_lanes(sb), torch.zeros((254, 16, 8), device=oracle.DEVICE)), "wide_half")
        # e2m1 +-0: mostly zero codes (all of them in the first 16 instructions), C +-0, subnormal or wide.
        n = 64
        zeros = torch.randint(0, 2, (n, 24, 64), generator=g, device=oracle.DEVICE, dtype=torch.uint8) * 8
        keep = torch.rand((n, 24, 64), generator=g, device=oracle.DEVICE) < 0.8
        keep[:16] = True
        mixed = torch.where(keep, zeros, codes(g, n, 24, 64))
        cases["e2m1 zero"] = (Batch(mixed[:, :16].contiguous(), mixed[:, 16:].contiguous(),
                                    uniform_lanes(scale_bytes(g, n, 0, MAX_SCALE + 1)),
                                    uniform_lanes(scale_bytes(g, n, 0, MAX_SCALE + 1)),
                                    torch.zeros((n, 16, 8), device=oracle.DEVICE)), "zero_c")
        # C from 2^-126 to 2^127 relative to the products, every scale code allowed.
        n = 256
        cases["c range"] = (Batch(codes(g, n, 16, 64), codes(g, n, 8, 64),
                                  uniform_lanes(scale_bytes(g, n, 0, MAX_SCALE + 1)),
                                  uniform_lanes(scale_bytes(g, n, 0, MAX_SCALE + 1)),
                                  torch.zeros((n, 16, 8), device=oracle.DEVICE)), "wide")
        # Zero results: C = -(the model's C = 0 result), so every element whose product sum is an FP32 value cancels.
        n = 64
        cases["zero sign"] = (Batch(codes(g, n, 16, 64), codes(g, n, 8, 64),
                                    uniform_lanes(scale_bytes(g, n, 0, MAX_SCALE + 1)),
                                    uniform_lanes(scale_bytes(g, n, 0, MAX_SCALE + 1)),
                                    torch.zeros((n, 16, 8), device=oracle.DEVICE)), "cancel")
        totals = {}
        for case, (batch, c_mode) in cases.items():
            a, b, sa_e, sb_e, _ = elements(batch)
            al, bl, sal, sbl = a.tolist(), b.tolist(), sa_e.tolist(), sb_e.tolist()
            products = [omma_ref.dot64(*r, 0.0) for r in zip(al, bl, sal, sbl)]
            half = len(products) // 2
            if c_mode == "wide_half":
                cs = [0.0] * half + [wide(rng, p, rng.randint(-126, 127)) for p in products[half:]]
            elif c_mode == "wide":
                cs = [wide(rng, p, rng.randint(-126, 127)) for p in products]
            elif c_mode == "zero_c":
                pick = (lambda p: rng.choice((0.0, -0.0, wide(rng, 0.0, rng.randint(-149, -127)),
                                              wide(rng, p, rng.randint(-126, 127)))))
                cs = [pick(p) for p in products]
            else:
                cs = [-p for p in products]
            c = torch.tensor(cs, dtype=torch.float32)
            batch = batch._replace(c=c.reshape(batch.c.shape).to(oracle.DEVICE))
            native = omma(batch)
            reference = torch.tensor([omma_ref.dot64(*r) for r in zip(al, bl, sal, sbl, c.tolist())],
                                     dtype=torch.float32).reshape(native.shape)
            counts = disagreement(native, reference)
            zero_results = int((reference == 0).sum())
            neg_zero = int(((reference == 0) & torch.signbit(reference)).sum())
            native_neg_zero = int(((native == 0) & torch.signbit(native)).sum())
            oracle.report_row(ROW, f"D2c {case} disagreement", zero_results=zero_results,
                              negative_zero_results=neg_zero, native_negative_zero_results=native_neg_zero, **counts)
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
            totals["zero_results"] = totals.get("zero_results", 0) + zero_results
            with self.subTest(case=case):
                oracle.check_bit_exact(self, ROW, f"D2c {case}", native, reference)
        oracle.report_row(ROW, "D2c characterization total", **totals)

    def test_d_scale_lane_mapping(self):
        g = oracle.generator(SEED + 3)
        sweep = sweep_batch()
        for ta, tb in SELECTORS:
            with self.subTest(thread_id_a=ta, thread_id_b=tb):
                d = omma(sweep, ta, tb)
                row_lane, bad_a = sweep_lanes(d, 0)
                col_lane, bad_b = sweep_lanes(d, 1)
                ptx = (all(lane // 4 == r % 8 and lane % 4 in (2 * ta, 2 * ta + 1) for r, lane in enumerate(row_lane))
                       and all(lane // 4 == c and lane % 4 == tb for c, lane in enumerate(col_lane)))
                oracle.report_row(ROW, f"D2d mapping ta{ta} tb{tb}", ptx_quad_rule=ptx, malformed=bad_a + bad_b,
                                  **{f"a_row{r:02d}_lane": lane for r, lane in enumerate(row_lane)},
                                  **{f"b_col{c}_lane": lane for c, lane in enumerate(col_lane)})
                oracle.check_count(self, ROW, f"D2d ta{ta} tb{tb} malformed one-hot results", bad_a + bad_b,
                                   expected=0)
                oracle.check_count(self, ROW, f"D2d ta{ta} tb{tb} rows/columns without exactly one lane",
                                   (row_lane + col_lane).count(-1), expected=0)
                n = 32
                lanes = (n, 32, 4)
                batch = Batch(codes(g, n, 16, 64), codes(g, n, 8, 64),
                              torch.randint(0, MAX_SCALE + 1, lanes, generator=g, device=oracle.DEVICE,
                                            dtype=torch.uint8),
                              torch.randint(0, MAX_SCALE + 1, lanes, generator=g, device=oracle.DEVICE,
                                            dtype=torch.uint8),
                              f3_c(g, n))
                native = omma(batch, ta, tb)
                reference = model_core(*elements(batch, row_lane, col_lane)).reshape(native.shape)
                oracle.check_bit_exact(self, ROW, f"D2d ta{ta} tb{tb} per-lane scales under the found mapping",
                                       native, reference)

    def test_e_torch_path_equals_core(self):
        g = oracle.generator(SEED + 4)
        rng = random.Random(SEED + 4)
        # Singles: 128 instructions on F3's domain, 128 on the wide domain (every scale code, wide C).
        n = 128
        f3 = Batch(codes(g, n, 16, 64), codes(g, n, 8, 64), uniform_lanes(scale_bytes(g, n, *F3_SCALES)),
                   uniform_lanes(scale_bytes(g, n, *F3_SCALES)), f3_c(g, n))
        cs = [wide(rng, 1.0, rng.randint(-40, 40)) for _ in range(n * 128)]
        wide_batch = Batch(codes(g, n, 16, 64), codes(g, n, 8, 64),
                           uniform_lanes(scale_bytes(g, n, 0, MAX_SCALE + 1)),
                           uniform_lanes(scale_bytes(g, n, 0, MAX_SCALE + 1)),
                           torch.tensor(cs, dtype=torch.float32).reshape(n, 16, 8).to(oracle.DEVICE))
        for case, batch in (("singles F3", f3), ("singles wide", wide_batch)):
            inputs = elements(batch)
            core = model_core(*inputs).reshape(n, 16, 8)
            oracle.check_bit_exact(self, ROW, f"D2e {case} torch path", model_torch(*inputs).reshape(n, 16, 8), core)
            oracle.check_bit_exact(self, ROW, f"D2e {case} omma", omma(batch), core)
        # Chains: 16 instructions x 16 steps (K = 1,024), 2,048 chains; D of step s is C of step s + 1.
        n, steps = 16, 16
        batches = [Batch(codes(g, n, 16, 64), codes(g, n, 8, 64), uniform_lanes(scale_bytes(g, n, *F3_SCALES)),
                         uniform_lanes(scale_bytes(g, n, *F3_SCALES)), f3_c(g, n)) for _ in range(steps)]
        c0 = batches[0].c
        d = c0
        for batch in batches:
            d = omma(batch._replace(c=d.to(oracle.DEVICE)))
        parts = [elements(batch) for batch in batches]
        a, b, sa, sb = (torch.cat([p[i] for p in parts], dim=1) for i in range(4))
        rows = zip(a.tolist(), b.tolist(), sa.tolist(), sb.tolist(), c0.reshape(-1).tolist())
        core = torch.tensor([omma_ref.chain(*r) for r in rows], dtype=torch.float32).reshape(n, 16, 8)
        chained = omma_ref.chain_t(a, b, sa, sb, c0.reshape(-1)).cpu().reshape(n, 16, 8)
        oracle.check_bit_exact(self, ROW, "D2e chains K1024 torch path", chained, core)
        oracle.check_bit_exact(self, ROW, "D2e chains K1024 omma", d, core)

    def test_selectors_outside_table_37_are_refused(self):
        g = oracle.generator(SEED + 5)
        n = 4
        batch = Batch(codes(g, n, 16, 64), codes(g, n, 8, 64), uniform_lanes(scale_bytes(g, n, *F3_SCALES)),
                      uniform_lanes(scale_bytes(g, n, *F3_SCALES)), f3_c(g, n))
        a = torch.zeros((n, 32, 4), dtype=torch.int32, device=oracle.DEVICE)
        b = torch.zeros((n, 32, 2), dtype=torch.int32, device=oracle.DEVICE)
        sa = torch.full((n, 32), 0x38383838, dtype=torch.int32, device=oracle.DEVICE)
        sb = sa.clone()
        c = batch.c.reshape(n, -1)[:, C_IDX].contiguous()
        cases = (("byte-id-a 1", n, 1, 0, 0, 0), ("byte-id-a 2", n, 2, 0, 0, 0), ("byte-id-b 1", n, 0, 0, 1, 0),
                 ("byte-id-b 3", n, 0, 0, 3, 0), ("thread-id-a 2", n, 0, 2, 0, 0), ("thread-id-a -1", n, 0, -1, 0, 0),
                 ("thread-id-b 4", n, 0, 0, 0, 4), ("thread-id-b -1", n, 0, 0, 0, -1), ("n 0", 0, 0, 0, 0, 0))
        for case, count, ba, ta, bb, tb in cases:
            with self.subTest(case=case):
                d = oracle.empty((n, 32, 4), torch.float32)
                before = d.clone()
                status = omma_status(a, b, sa, sb, c, d, count, ba, ta, bb, tb)
                oracle.check_count(self, ROW, f"refusal {case} status", status, expected=CERE_ERR_ARGUMENT)
                oracle.check_bit_exact(self, ROW, f"refusal {case} d untouched", d, before)
        d = oracle.empty((n, 32, 4), torch.float32)
        status = omma_status(a, b, sa, sb, c, d, n, 0, 1, 0, 3)
        oracle.check_count(self, ROW, "valid selectors ta1 tb3 status", status, expected=0)
        oracle.check_count(self, ROW, "valid selectors ta1 tb3 outputs written", int(torch.isnan(d).sum()),
                           expected=0)


def omma_status(a, b, sa, sb, c, d, n, byte_id_a, thread_id_a, byte_id_b, thread_id_b) -> int:
    """The status of one `cere_ktest_klab_omma` call (a refusal is the expected result, so it is returned, not
    raised as `oracle.call` would)."""
    fn = oracle.ktest("cere_ktest_klab_omma")
    stream = torch.cuda.current_stream(oracle.DEVICE)
    status = fn(*[oracle._arg(x) for x in (a, b, sa, sb, c, d, n, byte_id_a, thread_id_a, byte_id_b, thread_id_b)],
                ctypes.c_void_p(stream.cuda_stream))
    stream.synchronize()
    return status


def calib(row: int, size: int) -> tuple[float, float, list[float], float]:
    """`cere_ktest_klab_calib`: (work, median ms, the 5 repetitions' ms, launch shape)."""
    out = torch.zeros(7, dtype=torch.float64)
    oracle.call("cere_ktest_klab_calib", row, size, out)
    reps = out[1:1 + CALIB_REPS].tolist()
    return out[0].item(), statistics.median(reps), reps, out[6].item()


class KlabMeasured(unittest.TestCase):
    """D2 measured rows (F5): reported with `report_row`, never gated."""

    def report(self, case: str, row: int, size: int, field: str, rate, extra: str = "blocks") -> None:
        """`extra` names out[6]: the launch's blocks, the host ms of cudaGraphInstantiate (row 4), unused (row 5)."""
        work, median, reps, shape = calib(row, size)
        oracle.report_row(ROW, f"measured {case}", phase="measured", **{field: rate(work, median), extra: shape},
                          work=work, median_ms=median, min_ms=min(reps), max_ms=max(reps), reps=len(reps))

    def test_measured_rows(self):
        def tflops(work, ms):
            return work / (ms * 1e-3) / 1e12

        def gbps(work, ms):
            return work / (ms * 1e-3) / 1e9

        def us_per(work, ms):
            return ms * 1e3 / work

        self.report("omma nvfp4 dense", 0, 4, "tflops", tflops)
        self.report("hmma bf16 dense", 1, 4, "tflops", tflops)
        for size in (1, 4, 16):
            self.report(f"hbm read grid sms x{size}", 2, size, "gbps", gbps)
        self.report("grid.sync one block per sm", 3, 0, "us_per_barrier", us_per)
        self.report("grid.sync full occupancy", 3, 1, "us_per_barrier", us_per)
        for nodes in GRAPH_NODES:
            self.report(f"cuda graph replay {nodes} nodes", 4, nodes, "us_per_node", us_per, "instantiate_ms")
            self.report(f"stream launch {nodes} kernels", 5, nodes, "us_per_launch", us_per, "unused")


if __name__ == "__main__":
    unittest.main()
