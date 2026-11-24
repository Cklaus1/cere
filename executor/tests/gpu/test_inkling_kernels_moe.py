"""Inkling 2 Mini S4 (tasks/inkling-spec.md 3.4, 5, G1): family inkling2_loop's routed-MoE kernels
(executor/cuda/src/kernels/inkling/moe.h) vs their CPU references (executor/ref/inkling/{router,quant,moe}_ref.h,
compiled into libcere_ktest.so's host entry points, ktest/inkling/moe_ref_ktest.cpp), bit for bit.

Suite `inkling_kernels` (GPU0 only): `CERE_LIVE=1 python3 scripts/live/native_dev.py test --suite inkling_kernels`
against build/native-inkling-2-mini, the checkpoint (Inkling-2-Mini@b85c3cfa) read-only at /model.

Every case reports row INKLING_MOE. G1 cases (bit-exact, `check_bit_exact`):
- cvt: the E4M3 / E2M1 software encoders of quant_ref.h vs the hardware cvt.rn.satfinite they stand for, on every
  FP32 bit pattern (device), and the host compile vs the device compile on every BF16 value plus rounding midpoints;
- router (layers 0 and 21's gate): ids, weights and logits on random rows, the zero row (top-8 of 0.5 + bias), an
  all-tied router (zero bias: lowest ids win), large and tiny magnitudes (NaN logits: any NaN equals any NaN), strided
  rows;
- quantizers: NVFP4 (codes, E4M3 scales, per-token amax; both fouroversix branches taken) and MXFP8 (values,
  UE8M0 exponents; the exponent clamp) on edge rows;
- experts (NVFP4 layer 0, MXFP8 layer 21): out, and the taps gate_up (BF16 GEMM1), h (capped SwiGLU x gamma) and
  expert_out (BF16 per routed row) at M = 1, 7, 64 (gamma 0 and near 8, rows hitting the SwiGLU caps, a zero row,
  strided in/out), NVFP4 M = 97 and 160 (the tensor-core GEMM of calls over 64 tokens; M = 160 with per-block
  magnitudes), M = 4,096 on sampled tokens;
- recorded rows (S1's tests/fixtures/inkling2_reference/ops, arm Q): router and experts GPU vs ref on the recorded
  MLP inputs.
Determinism (G4 / D9, bit-exact, GPU vs GPU): run twice, each token alone vs in the batch, reversed batch, chunk
splits, M = 4,096 vs 512-token chunks.
Ref vs REF (reported with the tolerance asserted): router logits vs FP64 torch (rel-L2), ids vs FP64 torch top-8;
quantizers' dequantized values vs run_golden.py's arm-Q torch emulation (bit agreement reported); recorded rows: the
CPU ref's router vs REF's recorded router outputs, the CPU ref's experts vs REF's recorded expert outputs (golden arm Q).
Fail-first: each MoeMutant (softmax routing, one global activation scale, gamma after the down projection, no
fouroversix, the tensor-core GEMM's block terms added in swapped pairs, the multi-row router's K chains descending)
differs from the ref (phase fail_first: a G1 check of the mutant would fail).
Timing (reported, never a verdict): one MoE visit (router + experts) at M = 1, 8, 64, NVFP4 and MXFP8.
"""
from __future__ import annotations

import json
import math
import types
import unittest
from pathlib import Path

import torch

import oracle

ROW = "INKLING_MOE"
H, I, E, TOPK = 2304, 1152, 128, 8
NVFP4_LAYER, MXFP8_LAYER = 0, 21
MUT_SOFTMAX, MUT_GLOBAL_SCALE, MUT_GAMMA_AFTER, MUT_ALWAYS_SIX, MUT_BLOCK_ORDER, MUT_ROUTER_ORDER = 1, 2, 3, 4, 5, 6
OPS_INDEX = Path("/src/tests/fixtures/inkling2_reference/ops/index.json")
FLT_MAX = 3.4028234663852886e38
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu").contiguous()


class Layer:
    """One layer's MoE tensors on GPU0 and their CPU copies (the ref's operands)."""

    def __init__(self, layer: int):
        p = f"model.llm.layers.{layer}.mlp."
        self.layer = layer
        self.nvfp4 = oracle.has_tensor(p + "experts.w13_weight_amax")
        self.gate_w = oracle.real_tensor(p + "gate.weight")
        self.gate_b = oracle.real_tensor(p + "gate.bias").float().contiguous()

        def u8(name):
            return oracle.real_tensor(p + name).contiguous().view(torch.uint8)

        self.w13, self.w13_scale = u8("experts.w13_weight"), u8("experts.w13_weight_scale")
        self.w2, self.w2_scale = u8("experts.w2_weight"), u8("experts.w2_weight_scale")
        if self.nvfp4:
            self.w13_amax = oracle.real_tensor(p + "experts.w13_weight_amax").float().contiguous()
            self.w2_amax = oracle.real_tensor(p + "experts.w2_weight_amax").float().contiguous()
            self.w13_den = oracle.real_tensor(p + "experts.w13_weight_outer_scale_denominator").float().contiguous()
            self.w2_den = oracle.real_tensor(p + "experts.w2_weight_outer_scale_denominator").float().contiguous()
        else:
            self.w13_amax = self.w2_amax = self.w13_den = self.w2_den = None
        self.cpu = {k: (None if v is None else _cpu(v)) for k, v in vars(self).items() if isinstance(v, torch.Tensor)
                    or v is None}

    def weight_args(self):
        return (int(self.nvfp4), E, self.w13, self.w13_scale, self.w2, self.w2_scale, self.w13_amax, self.w2_amax,
                self.w13_den, self.w2_den)

    def ref_weight_args(self):
        c = self.cpu
        den13 = float(c["w13_den"][0]) if self.nvfp4 else 0.0
        den2 = float(c["w2_den"][0]) if self.nvfp4 else 0.0
        return (int(self.nvfp4), E, c["w13"], c["w13_scale"], c["w2"], c["w2_scale"], c["w13_amax"], c["w2_amax"],
                den13, den2)


_LAYERS: dict[int, Layer] = {}


def layer(n: int) -> Layer:
    if n not in _LAYERS:
        if len(_LAYERS) >= 2:
            _LAYERS.clear()
            torch.cuda.empty_cache()
        _LAYERS[n] = Layer(n)
    return _LAYERS[n]


# -- native and reference calls ---------------------------------------------------------------------------------

def native_router(x, lay: Layer, mutant=0, x_stride=H, m=None):
    m = x.shape[0] if m is None else m
    ids, w, logits = oracle.empty((m, TOPK), torch.int32), oracle.empty((m, TOPK), torch.float32), \
        oracle.empty((m, E), torch.float32)
    oracle.call("cere_ktest_inkling_router", x, x_stride, m, lay.gate_w, lay.gate_b, ids, w, logits, mutant)
    return _cpu(ids), _cpu(w), _cpu(logits)


def ref_router(x_cpu, lay: Layer):
    m = x_cpu.shape[0]
    ids, w = torch.empty((m, TOPK), dtype=torch.int32), torch.empty((m, TOPK), dtype=torch.float32)
    logits = torch.empty((m, E), dtype=torch.float32)
    oracle.call("cere_ktest_inkling_router_ref", x_cpu, m, lay.cpu["gate_w"], lay.cpu["gate_b"], ids, w, logits)
    return ids, w, logits


def native_nvfp4(x, k, mutant=0, x_stride=None, rows=None):
    rows = x.shape[0] if rows is None else rows
    codes, scales, amax = oracle.empty((rows, k // 2), torch.uint8), oracle.empty((rows, k // 16), torch.uint8), \
        oracle.empty((rows,), torch.float32)
    oracle.call("cere_ktest_inkling_quant_nvfp4", x, k if x_stride is None else x_stride, rows, k, codes, scales, amax,
                mutant)
    return _cpu(codes), _cpu(scales), _cpu(amax)


def ref_nvfp4(x_cpu):
    rows, k = x_cpu.shape
    codes, scales = torch.empty((rows, k // 2), dtype=torch.uint8), torch.empty((rows, k // 16), dtype=torch.uint8)
    amax, chose4 = torch.empty((rows,), dtype=torch.float32), torch.empty((rows,), dtype=torch.int32)
    oracle.call("cere_ktest_inkling_quant_nvfp4_ref", x_cpu, rows, k, codes, scales, amax, chose4)
    return codes, scales, amax, chose4


def native_mxfp8(x, k, x_stride=None, rows=None):
    rows = x.shape[0] if rows is None else rows
    values, exps = oracle.empty((rows, k), torch.uint8), oracle.empty((rows, k // 32), torch.uint8)
    oracle.call("cere_ktest_inkling_quant_mxfp8", x, k if x_stride is None else x_stride, rows, k, values, exps)
    return _cpu(values), _cpu(exps)


def ref_mxfp8(x_cpu):
    rows, k = x_cpu.shape
    values, exps = torch.empty((rows, k), dtype=torch.uint8), torch.empty((rows, k // 32), dtype=torch.uint8)
    oracle.call("cere_ktest_inkling_quant_mxfp8_ref", x_cpu, rows, k, values, exps)
    return values, exps


def workspace_bytes(m: int, nvfp4: bool) -> int:
    out = torch.zeros((1,), dtype=torch.int64)
    oracle.call("cere_ktest_inkling_moe_workspace_bytes", m, int(nvfp4), out)
    return int(out.item())


def native_experts(x, ids, w, lay: Layer, mutant=0, taps=False, x_stride=H, out_stride=H, m=None):
    """x, ids, w on GPU0 -> CPU (out [m, H], taps (gate_up, h, expert_out) or None)."""
    m = x.shape[0] if m is None else m
    nbytes = workspace_bytes(m, lay.nvfp4)
    ws = torch.empty((nbytes,), dtype=torch.uint8, device=oracle.DEVICE)
    out = oracle.empty((m, out_stride), torch.bfloat16)
    gu = oracle.empty((m * TOPK, 2 * I), torch.bfloat16) if taps else None
    h = oracle.empty((m * TOPK, I), torch.bfloat16) if taps else None
    eo = oracle.empty((m * TOPK, H), torch.bfloat16) if taps else None
    oracle.call("cere_ktest_inkling_moe_experts", x, x_stride, m, ids, w, *lay.weight_args(), ws, nbytes, out,
                out_stride, gu, h, eo, mutant)
    return _cpu(out[:, :H]), (None if not taps else (_cpu(gu), _cpu(h), _cpu(eo)))


def ref_experts(x_cpu, ids_cpu, w_cpu, lay: Layer):
    m = x_cpu.shape[0]
    out = torch.empty((m, H), dtype=torch.bfloat16)
    gu = torch.empty((m * TOPK, 2 * I), dtype=torch.bfloat16)
    h = torch.empty((m * TOPK, I), dtype=torch.bfloat16)
    eo = torch.empty((m * TOPK, H), dtype=torch.bfloat16)
    oracle.call("cere_ktest_inkling_moe_ref", x_cpu, m, ids_cpu, w_cpu, *lay.ref_weight_args(), out, gu, h, eo)
    return out, (gu, h, eo)


def differing(a: torch.Tensor, b: torch.Tensor) -> int:
    a8 = a.contiguous().view(torch.uint8).reshape(a.numel(), -1)
    b8 = b.contiguous().view(torch.uint8).reshape(b.numel(), -1)
    return int((a8 != b8).any(dim=-1).sum().item())


# -- inputs -----------------------------------------------------------------------------------------------------

def edge_rows(k: int, gen: torch.Generator) -> torch.Tensor:
    """BF16 [12, k] (CPU): normal, zero, huge, tiny (BF16 subnormals), one spike, the BF16 maximum, a row of exact
    E2M1 rounding midpoints (ties), mixed per-block magnitudes, a constant row, a single nonzero, two normals."""
    rows = [torch.randn(k, generator=gen),
            torch.zeros(k),
            torch.randn(k, generator=gen) * 1e30,
            torch.randn(k, generator=gen) * 1e-39,
            torch.cat([torch.tensor([1e4]), torch.randn(k - 1, generator=gen) * 1e-3]),
            torch.full((k,), 3.3895313892515355e38) * torch.sign(torch.randn(k, generator=gen)),
            torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * (k // 8)) * 2.0 ** 7,
            torch.randn(k, generator=gen) * torch.exp2(torch.randint(-20, 20, (k // 16, 1), generator=gen)
                                                       .float()).repeat(1, 16).reshape(k),
            torch.full((k,), -0.375),
            torch.cat([torch.zeros(k - 1), torch.tensor([-7.0])]),
            torch.randn(k, generator=gen) * 3.0,
            torch.randn(k, generator=gen) * 0.02]
    return torch.stack(rows).to(torch.bfloat16)


def random_routing(m: int, gen: torch.Generator, pool: int = E):
    """ids int32 [m, 8] ascending distinct from experts [0, pool) (a small pool forces shared experts), weights FP32
    [m, 8] positive summing to about 8 (row 0's last slot weight 0, row 1's first slot near 8)."""
    ids = torch.stack([torch.randperm(pool, generator=gen)[:TOPK].sort().values for _ in range(m)]).to(torch.int32)
    w = torch.rand((m, TOPK), generator=gen) + 0.05
    w = (w / w.sum(-1, keepdim=True) * 8.0).float()
    if m > 1:
        w[0, -1] = 0.0
        w[1] = torch.tensor([7.93] + [0.01] * 7)
    return ids, w


def expert_inputs(m: int, gen: torch.Generator) -> torch.Tensor:
    """BF16 [m, H] (CPU) MLP inputs: unit-RMS rows with per-token scales; row 2 large (hits the SwiGLU caps), row 3
    zero, row 4 tiny."""
    x = torch.randn((m, H), generator=gen) * torch.exp2(torch.randint(-3, 4, (m, 1), generator=gen).float())
    if m > 2:
        x[2] *= 300.0
    if m > 3:
        x[3] = 0.0
    if m > 4:
        x[4] *= 1e-20
    return x.to(torch.bfloat16)


# -- REF-side models (torch) ------------------------------------------------------------------------------------


def golden_nvfp4(x: torch.Tensor):
    """scripts/native/inkling_reference/run_golden.py nvfp4_encode (arm Q emulation, golden v2: EXACT_ENCODER, the
    one-rounding 1536 / amax), FP32 torch on CPU: packed codes [n, k/2], E4M3 scale bytes [n, k/16], amax [n]."""
    def e2m1_rn(v):
        a = v.abs()
        r = torch.where(a <= 0.25, 0.0, torch.where(a < 0.75, 0.5, torch.where(a <= 1.25, 1.0, torch.where(
            a < 1.75, 1.5, torch.where(a <= 2.5, 2.0, torch.where(a < 3.5, 3.0, torch.where(a <= 5.0, 4.0, 6.0)))))))
        return torch.copysign(r, v)

    def pairwise16(e):
        while e.shape[-1] > 1:
            e = e.view(*e.shape[:-1], -1, 2)
            e = e[..., 0] + e[..., 1]
        return e[..., 0]

    n, k = x.shape
    amax = x.abs().amax(-1)
    encoder = torch.clamp(torch.full_like(amax, 1536.0) / amax, max=FLT_MAX)
    decoder = 1.0 / encoder
    outer = amax * float(1.0 / 1536.0)
    xb = x.view(n, k // 16, 16)
    bmax = xb.abs().amax(-1)
    b6 = torch.clamp(bmax * (encoder * float(1.0 / 6.0))[:, None], max=448.0).to(torch.float8_e4m3fn)
    b4 = torch.clamp(bmax * (encoder * 0.25)[:, None], max=448.0).to(torch.float8_e4m3fn)
    s6, s4 = b6.float(), b4.float()
    inv6 = torch.clamp(1.0 / (s6 * decoder[:, None]), max=FLT_MAX)
    inv4 = torch.clamp(1.0 / (s4 * decoder[:, None]), max=FLT_MAX)
    r6, r4 = e2m1_rn(xb * inv6[..., None]), e2m1_rn(xb * inv4[..., None])
    err6 = pairwise16(((r6 * s6[..., None]) * outer[:, None, None] - xb).abs())
    err4 = pairwise16(((r4 * s4[..., None]) * outer[:, None, None] - xb).abs())
    choose4 = err4 < err6
    r = torch.where(choose4[..., None], r4, r6)
    grid = torch.tensor(E2M1_VALUES[:8])
    c = (torch.searchsorted(grid, r.abs().contiguous()) | (torch.signbit(r).long() << 3)).view(n, k // 2, 2)
    codes = (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)
    scales = torch.where(choose4, b4.view(torch.uint8), b6.view(torch.uint8))
    return codes, scales, amax


def golden_mxfp8(x: torch.Tensor) -> torch.Tensor:
    """run_golden.py mxfp8_quant: dequantized FP32 [n, k]."""
    n, k = x.shape
    xb = x.view(n, k // 32, 32)
    raw = xb.abs().amax(-1) * float(1.0 / 448.0)
    exponent = ((raw.view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    min_scale = 2.0 ** -127
    exponent = torch.where(raw <= min_scale, torch.zeros_like(exponent), torch.clamp(exponent, max=254))
    scale = torch.clamp((exponent << 23).view(torch.float32), min=min_scale)
    inverse = torch.where(exponent == 0, torch.full_like(scale, 2.0 ** 127), 1.0 / scale)
    v = torch.clamp(xb * inverse[..., None], -448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (v * scale[..., None]).view(n, k)


def dequant_mxfp8(values: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    v = values.view(torch.float8_e4m3fn).float()
    e = exps.to(torch.int32)
    s = torch.where(e == 0, torch.full(e.shape, 2.0 ** -127), (e << 23).view(torch.float32))
    return (v.view(v.shape[0], -1, 32) * s[..., None]).view(v.shape[0], -1)


def torch_router(x_bf16: torch.Tensor, lay: Layer):
    """REF Inkling2LoopTopKRouter in FP64: logits, the top-8 id sets (ascending) and weights in that order."""
    logits = x_bf16.double() @ lay.cpu["gate_w"].double().T
    scores = torch.sigmoid(logits)
    ids = torch.topk(scores + lay.cpu["gate_b"].double(), TOPK, dim=-1).indices.sort(-1).values
    sel = scores.gather(1, ids)
    return logits, ids, sel / (sel.sum(-1, keepdim=True) + 1e-20) * 8.0


def bf16_ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def ordered(t):
        bits = t.contiguous().view(torch.int16).to(torch.int64)
        mag = bits & 0x7FFF
        return torch.where(bits < 0, -mag, mag)
    return (ordered(a) - ordered(b)).abs()


# -- recorded rows (S1) -----------------------------------------------------------------------------------------

def recorded_sets() -> list[tuple[str, dict]]:
    """(set name, {key: CPU tensor}) of the recorded MoE sets; absence fails the calling test."""
    if not OPS_INDEX.is_file():
        raise oracle.OracleError(f"{OPS_INDEX} is absent (S1's recorded per-op vectors, build/inkling-s1)")
    index = json.loads(OPS_INDEX.read_text())
    if index.get("schema") != "cere-inkling2-ops-v1":
        raise oracle.OracleError(f"{OPS_INDEX}: schema {index.get('schema')!r}")
    dtypes = {"bf16": torch.bfloat16, "f32": torch.float32, "i64": torch.int64, "i32": torch.int32}
    out = []
    for name, entry in index["sets"].items():
        tensors = entry.get("tensors", {})
        if "mlp.experts:in0" not in tensors:
            continue
        loaded = {}
        for key in ("mlp.gate:in0", "mlp.gate:out.0", "mlp.gate:out.1", "mlp.gate:out.2", "mlp.experts:in0",
                    "mlp.experts:in1", "mlp.experts:in2", "mlp.experts:out"):
            t = tensors[key]
            raw = (OPS_INDEX.parent / t["file"]).read_bytes()
            loaded[key] = torch.frombuffer(bytearray(raw), dtype=dtypes[t["dtype"]]).reshape(t["shape"]).clone()
        loaded["layer"] = entry["layer"]
        out.append((name, loaded))
    if not out:
        raise oracle.OracleError(f"{OPS_INDEX}: no set holds mlp.experts tensors")
    return out


def sort_routing(ids: torch.Tensor, w: torch.Tensor):
    """REF's (unsorted) top-8 ids/weights -> ascending ids (slot order) with their weights."""
    order = ids.argsort(-1)
    return ids.gather(1, order).to(torch.int32).contiguous(), w.gather(1, order).float().contiguous()


class TestInklingMoe(unittest.TestCase):

    def bit(self, case, native, reference):
        oracle.check_bit_exact(self, ROW, case, native, reference)

    def bit_float(self, case, native, reference):
        """Bit-exact FP32, except that any NaN equals any NaN: a NaN's sign and payload are not part of G1 (inf - inf
        is 0xffc00000 on the x86 host, 0x7fffffff on the GPU). Reports the NaN pairs and the first differences."""
        a, b = _cpu(native).float().reshape(-1), _cpu(reference).float().reshape(-1)
        self.assertEqual(tuple(native.shape), tuple(reference.shape), f"{case}: shape")
        both_nan = torch.isnan(a) & torch.isnan(b)
        bad = (a.view(torch.int32) != b.view(torch.int32)) & ~both_nan
        n = int(bad.sum().item())
        first = [(int(i), float(a[i]), float(b[i])) for i in bad.nonzero().view(-1)[:4]]
        oracle.report_row(ROW, case, phase="bit_exact", elements=a.numel(), differing=n,
                          nan_pairs=int(both_nan.sum().item()), first_index=first[0][0] if first else -1,
                          passed=n == 0)
        self.assertEqual(n, 0, f"{ROW} {case}: {n} of {a.numel()} elements differ (index, native, ref): {first}")

    def fail_first(self, case, native, reference):
        n = differing(native, reference)
        oracle.report_row(ROW, case, phase="fail_first", elements=native.numel(), differing=n, passed=n > 0)
        oracle.check_count(self, ROW, f"{case} detected (the mutant differs from the reference)", int(n > 0),
                           expected=1)

    # -- conversions --------------------------------------------------------------------------------------------

    def test_cvt_every_fp32(self):
        """quant_ref.h's E4M3 / E2M1 encoders (device compile) == cvt.rn.satfinite on all 2^32 FP32 patterns."""
        chunk = 1 << 27
        bad4 = bad2 = 0
        outs = [torch.empty((chunk,), dtype=torch.uint8, device=oracle.DEVICE) for _ in range(4)]
        for lo in range(0, 1 << 32, chunk):
            bits = torch.arange(lo, lo + chunk, dtype=torch.int64, device=oracle.DEVICE)
            bits = torch.where(bits >= (1 << 31), bits - (1 << 32), bits).to(torch.int32)
            x = bits.view(torch.float32)
            oracle.call("cere_ktest_inkling_cvt_sweep", x, chunk, *outs)
            not_nan = ~torch.isnan(x)  # NaN inputs never reach the quantizers (finite activations)
            bad4 += int(((outs[0] != outs[1]) & not_nan).sum().item())
            bad2 += int(((outs[2] != outs[3]) & not_nan).sum().item())
            del bits, x
        oracle.check_count(self, ROW, "cvt e4m3 emu vs hw, all non-NaN fp32", bad4, expected=0)
        oracle.check_count(self, ROW, "cvt e2m1 emu vs hw, all non-NaN fp32", bad2, expected=0)

    def test_cvt_host_vs_device(self):
        """The host compile of the encoders == the device compile (every BF16 value, the E4M3 / E2M1 grid midpoints
        and their neighbours, 1M random FP32)."""
        gen = torch.Generator().manual_seed(1)
        bf = torch.arange(0, 1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
        grid4 = torch.arange(0, 256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()
        grid4 = grid4[torch.isfinite(grid4)].unique().sort().values
        mids = torch.cat([(grid4[1:] + grid4[:-1]) / 2, torch.tensor(E2M1_VALUES[:8]).unique().diff() / 2
                          + torch.tensor(E2M1_VALUES[:8]).unique()[:-1], torch.tensor([448.0, 464.0, 480.0, 6.0, 7.0])])
        near = torch.cat([mids, torch.nextafter(mids, torch.tensor(math.inf)), torch.nextafter(mids, torch.tensor(0.0))])
        rnd = (torch.randint(-(1 << 31), (1 << 31) - 1, (1 << 20,), generator=gen, dtype=torch.int64)
               .to(torch.int32).view(torch.float32))
        x = torch.cat([bf, near, -near, rnd])
        x = x[~torch.isnan(x)].contiguous()
        n = x.numel()
        outs = [oracle.empty((n,), torch.uint8) for _ in range(4)]
        oracle.call("cere_ktest_inkling_cvt_sweep", x.to(oracle.DEVICE), n, *outs)
        e4, e2 = torch.empty((n,), dtype=torch.uint8), torch.empty((n,), dtype=torch.uint8)
        oracle.call("cere_ktest_inkling_cvt_ref", x, n, e4, e2)
        self.bit("cvt e4m3 host vs device", e4, _cpu(outs[1]))
        self.bit("cvt e2m1 host vs device", e2, _cpu(outs[3]))
        self.bit("cvt e4m3 host vs hw", e4, _cpu(outs[0]))
        self.bit("cvt e2m1 host vs hw", e2, _cpu(outs[2]))

    # -- router -------------------------------------------------------------------------------------------------

    def test_router(self):
        gen = torch.Generator().manual_seed(2)
        for n in (NVFP4_LAYER, MXFP8_LAYER):
            lay = layer(n)
            x = torch.cat([edge_rows(H, gen), expert_inputs(52, gen)]).contiguous()
            ids, w, logits = native_router(x.to(oracle.DEVICE), lay)
            rids, rw, rlogits = ref_router(x, lay)
            self.bit(f"router L{n} ids", ids, rids)
            self.bit(f"router L{n} weights", w, rw)
            self.bit_float(f"router L{n} logits", logits, rlogits)
            # the zero row: every score is sigmoid(0) = 0.5, so the top-8 follows 0.5 + bias (lowest id on ties)
            biased = torch.full((E,), 0.5, dtype=torch.float32) + lay.cpu["gate_b"]
            want = sorted(sorted(range(E), key=lambda e: (-float(biased[e]), e))[:TOPK])
            self.assertEqual(rids[1].tolist(), want, "the zero row's top-8 = top-8 of 0.5 + bias")
            # the tie rule: zero bias and the zero row tie all 128 experts -> ids 0..7, equal weights
            tie = types.SimpleNamespace(gate_w=lay.gate_w, gate_b=torch.zeros((E,), device=oracle.DEVICE),
                                        cpu={"gate_w": lay.cpu["gate_w"], "gate_b": torch.zeros((E,))})
            zero = torch.zeros((2, H), dtype=torch.bfloat16)
            tids_n, tw_n, _ = native_router(zero.to(oracle.DEVICE), tie)
            tids_r, tw_r, _ = ref_router(zero, tie)
            self.bit(f"router L{n} all-tied ids", tids_n, tids_r)
            self.bit(f"router L{n} all-tied weights", tw_n, tw_r)
            self.assertEqual(tids_r[0].tolist(), list(range(TOPK)), "all tied: the lowest ids win")
            # strided rows (x_stride 2,560 > 2,304)
            wide = torch.zeros((x.shape[0], 2560), dtype=torch.bfloat16)
            wide[:, :H] = x
            sids, sw, _ = native_router(wide.to(oracle.DEVICE), lay, x_stride=2560)
            self.bit(f"router L{n} strided ids", sids, rids)
            self.bit(f"router L{n} strided weights", sw, rw)
            # ref vs REF (FP64 torch) on the finite, unit-scale rows (12 onward)
            tl, tids, tw = torch_router(x[12:], lay)
            limit = oracle.port_limit(1e-5, "S4 router: FP32 reduce_dot<256> logits vs FP64 REF, 2,304 terms")
            oracle.check_within(self, ROW, f"router L{n} ref logits vs fp64 REF",
                                oracle.rel_l2(rlogits[12:].double(), tl), c=0.0, limit=limit)
            same = int((tids == rids[12:].long()).all(-1).sum().item())
            oracle.report_row(ROW, f"router L{n} ref ids vs fp64 REF", phase="agreement", tokens=tids.shape[0],
                              same_sets=same)
            oracle.check_count(self, ROW, f"router L{n} at most one top-8 set differs from FP64 REF",
                               int(tids.shape[0] - same <= 1), expected=1)
            # mutant: softmax routing
            _, mw, _ = native_router(x.to(oracle.DEVICE), lay, mutant=MUT_SOFTMAX)
            self.fail_first(f"router L{n} mutant softmax routing", mw, rw)
            # calls of more than 64 rows take the multi-row router (8 rows per cluster, D-I-CPF): 133 rows (a partial
            # last cluster of 5), contiguous and strided, the edge rows first; mutant: lane chains over K descending
            big = torch.cat([x, expert_inputs(133 - x.shape[0], gen)]).contiguous()
            bids, bw, blogits = native_router(big.to(oracle.DEVICE), lay)
            rbids, rbw, rblogits = ref_router(big, lay)
            self.bit(f"router L{n} M133 ids", bids, rbids)
            self.bit(f"router L{n} M133 weights", bw, rbw)
            self.bit_float(f"router L{n} M133 logits", blogits, rblogits)
            wide = torch.zeros((big.shape[0], 2560), dtype=torch.bfloat16)
            wide[:, :H] = big
            sids, sw, _ = native_router(wide.to(oracle.DEVICE), lay, x_stride=2560)
            self.bit(f"router L{n} M133 strided ids", sids, rbids)
            self.bit(f"router L{n} M133 strided weights", sw, rbw)
            _, _, ml = native_router(big.to(oracle.DEVICE), lay, mutant=MUT_ROUTER_ORDER)
            self.fail_first(f"router L{n} M133 mutant K descending", ml, rblogits)

    # -- quantizers ---------------------------------------------------------------------------------------------

    def test_quant_nvfp4(self):
        gen = torch.Generator().manual_seed(3)
        total4 = blocks = 0
        for k in (H, I):
            x = torch.cat([edge_rows(k, gen), expert_inputs(20, gen)[:, :k]]).contiguous()
            codes, scales, amax = native_nvfp4(x.to(oracle.DEVICE), k)
            rc, rs, ra, chose4 = ref_nvfp4(x)
            self.bit(f"nvfp4 k{k} codes", codes, rc)
            self.bit(f"nvfp4 k{k} scales", scales, rs)
            self.bit(f"nvfp4 k{k} amax", amax, ra)
            total4 += int(chose4.sum().item())
            blocks += x.shape[0] * (k // 16)
            # strided rows
            wide = torch.zeros((x.shape[0], k + 64), dtype=torch.bfloat16)
            wide[:, :k] = x
            sc, ss, sa = native_nvfp4(wide.to(oracle.DEVICE), k, x_stride=k + 64)
            self.bit(f"nvfp4 k{k} strided codes", sc, rc)
            self.bit(f"nvfp4 k{k} strided scales", ss, rs)
            # ref vs REF's quantizer: run_golden.py's arm-Q emulation (golden v2, bit-equal to routed_quant.cuh, S1)
            gc, gs, ga = golden_nvfp4(x.float())
            self.bit(f"nvfp4 k{k} ref codes vs golden arm-Q emulation", rc, gc)
            self.bit(f"nvfp4 k{k} ref scales vs golden arm-Q emulation", rs, gs)
            self.bit(f"nvfp4 k{k} ref amax vs golden arm-Q emulation", ra, ga)
            mc, ms, _ = native_nvfp4(x.to(oracle.DEVICE), k, mutant=MUT_ALWAYS_SIX)
            self.fail_first(f"nvfp4 k{k} mutant always /6 scales", ms, rs)
            mc, ms, _ = native_nvfp4(x.to(oracle.DEVICE), k, mutant=MUT_GLOBAL_SCALE)
            self.fail_first(f"nvfp4 k{k} mutant global activation scale", torch.cat([mc.view(-1), ms.view(-1)]),
                            torch.cat([rc.view(-1), rs.view(-1)]))
        oracle.report_row(ROW, "nvfp4 fouroversix branches", phase="count", blocks=blocks, chose4=total4,
                          passed=0 < total4 < blocks)
        self.assertTrue(0 < total4 < blocks, "both fouroversix branches must be exercised")

    def test_quant_mxfp8(self):
        gen = torch.Generator().manual_seed(4)
        for k in (H, I):
            x = torch.cat([edge_rows(k, gen), expert_inputs(20, gen)[:, :k]]).contiguous()
            values, exps = native_mxfp8(x.to(oracle.DEVICE), k)
            rv, re = ref_mxfp8(x)
            self.bit(f"mxfp8 k{k} values", values, rv)
            self.bit(f"mxfp8 k{k} exponents", exps, re)
            wide = torch.zeros((x.shape[0], k + 32), dtype=torch.bfloat16)
            wide[:, :k] = x
            sv, se = native_mxfp8(wide.to(oracle.DEVICE), k, x_stride=k + 32)
            self.bit(f"mxfp8 k{k} strided values", sv, rv)
            self.bit(f"mxfp8 k{k} strided exponents", se, re)
            gdeq = golden_mxfp8(x.float())
            rdeq = dequant_mxfp8(rv, re)
            agree = int(((gdeq == rdeq) | (torch.isnan(gdeq) & torch.isnan(rdeq))).sum().item())
            oracle.report_row(ROW, f"mxfp8 k{k} ref vs golden arm-Q emulation", phase="agreement",
                              elements=rdeq.numel(), equal=agree, passed=agree == rdeq.numel())
            self.assertEqual(agree, rdeq.numel(), "MXFP8 ref differs from the golden emulation")

    # -- experts ------------------------------------------------------------------------------------------------

    def _experts_case(self, lay: Layer, m: int, gen: torch.Generator, pool: int, strided: bool = False,
                      mixed: int = 0):
        x = expert_inputs(m, gen)
        if mixed:  # every `mixed` elements their own power-of-two magnitude: the K chains round (order shows)
            x = (x.float() * torch.exp2(torch.randint(-12, 13, (m, H // mixed), generator=gen).float())
                 .repeat_interleave(mixed, dim=1)).to(torch.bfloat16)
        ids, w = random_routing(m, gen, pool)
        name = f"experts L{lay.layer} M{m}" + (" strided" if strided else "")
        if strided:
            wide = torch.zeros((m, H + 128), dtype=torch.bfloat16)
            wide[:, :H] = x
            out, taps = native_experts(wide.to(oracle.DEVICE), ids.to(oracle.DEVICE), w.to(oracle.DEVICE), lay,
                                       taps=True, x_stride=H + 128, out_stride=H + 256)
        else:
            out, taps = native_experts(x.to(oracle.DEVICE), ids.to(oracle.DEVICE), w.to(oracle.DEVICE), lay,
                                       taps=True)
        rout, rtaps = ref_experts(x, ids, w, lay)
        self.bit(f"{name} gate_up", taps[0], rtaps[0])
        self.bit(f"{name} h", taps[1], rtaps[1])
        self.bit(f"{name} expert_out", taps[2], rtaps[2])
        self.bit(f"{name} out", out, rout)
        return x, ids, w, rout, rtaps

    def test_experts(self):
        gen = torch.Generator().manual_seed(5)
        for n in (NVFP4_LAYER, MXFP8_LAYER):
            lay = layer(n)
            self._experts_case(lay, 1, gen, E)
            self._experts_case(lay, 2, gen, 12)
            self._experts_case(lay, 3, gen, 10)
            self._experts_case(lay, 4, gen, 9, strided=True)
            self._experts_case(lay, 7, gen, 12, strided=True)
            x, ids, w, rout, _ = self._experts_case(lay, 64, gen, 12)
            dx, di, dw = x.to(oracle.DEVICE), ids.to(oracle.DEVICE), w.to(oracle.DEVICE)
            # mutants
            if lay.nvfp4:
                for mut, label in ((MUT_GLOBAL_SCALE, "global activation scale"), (MUT_ALWAYS_SIX, "always /6")):
                    mo, _ = native_experts(dx, di, dw, lay, mutant=mut)
                    self.fail_first(f"experts L{n} mutant {label}", mo, rout)
            mo, _ = native_experts(dx, di, dw, lay, mutant=MUT_GAMMA_AFTER)
            self.fail_first(f"experts L{n} mutant gamma after down", mo, rout)
        # NVFP4 calls of more than 64 tokens take the tensor-core GEMM (mma_gemm_kernel, D-I-CPF): experts with more
        # than one 64-row tile and partial tiles (M = 160 over 12 experts, strided, each 16-element block of its own
        # magnitude so the K chains round and an order change shows), most experts with a few rows (M = 97 over
        # all 128). The mutants are judged on the output and GEMM1's gate_up tap together: the order change rounds
        # GEMM1's FP32 sums, while the NVFP4 requantization of h and GEMM2's same-magnitude blocks hide it in the
        # output.
        lay = layer(NVFP4_LAYER)
        self._experts_case(lay, 97, gen, E)
        x, ids, w, rout, rtaps = self._experts_case(lay, 160, gen, 12, strided=True, mixed=16)
        dx, di, dw = x.to(oracle.DEVICE), ids.to(oracle.DEVICE), w.to(oracle.DEVICE)
        for mut, label in ((MUT_GAMMA_AFTER, "gamma after down"), (MUT_BLOCK_ORDER, "block terms in swapped pairs")):
            mo, mt = native_experts(dx, di, dw, lay, mutant=mut, taps=True)
            self.fail_first(f"experts L{NVFP4_LAYER} M160 mutant {label}", torch.cat([mo.flatten(), mt[0].flatten()]),
                            torch.cat([rout.flatten(), rtaps[0].flatten()]))

    def test_experts_determinism(self):
        """G4 / D9: run twice, alone vs batched, reversed, chunk splits (GPU vs GPU, bit-exact)."""
        gen = torch.Generator().manual_seed(6)
        for n in (NVFP4_LAYER, MXFP8_LAYER):
            lay = layer(n)
            m = 64
            x, (ids, w) = expert_inputs(m, gen), random_routing(m, gen, 10)
            dx, di, dw = x.to(oracle.DEVICE), ids.to(oracle.DEVICE), w.to(oracle.DEVICE)
            base, _ = native_experts(dx, di, dw, lay)
            again, _ = native_experts(dx, di, dw, lay)
            self.bit(f"determinism L{n} run twice", again, base)
            alone = torch.cat([native_experts(dx[t:t + 1].contiguous(), di[t:t + 1].contiguous(),
                                              dw[t:t + 1].contiguous(), lay)[0] for t in (0, 5, 31, 63)])
            self.bit(f"determinism L{n} alone vs batch", alone, base[[0, 5, 31, 63]])
            rev = torch.arange(m - 1, -1, -1)
            r, _ = native_experts(dx[rev].contiguous(), di[rev].contiguous(), dw[rev].contiguous(), lay)
            self.bit(f"determinism L{n} reversed batch", r[rev], base)
            parts = [native_experts(dx[a:b].contiguous(), di[a:b].contiguous(), dw[a:b].contiguous(), lay)[0]
                     for a, b in ((0, 23), (23, 24), (24, 26), (26, 29), (29, 33), (33, 64))]
            self.bit(f"determinism L{n} chunks 23/1/2/3/4/31", torch.cat(parts), base)

    def test_experts_prefill(self):
        """M = 4,096 (the largest prefill chunk): vs 512-token chunks (GPU) and vs the ref on sampled tokens."""
        gen = torch.Generator().manual_seed(7)
        lay = layer(NVFP4_LAYER)
        m = 4096
        x, (ids, w) = expert_inputs(m, gen), random_routing(m, gen, E)
        dx, di, dw = x.to(oracle.DEVICE), ids.to(oracle.DEVICE), w.to(oracle.DEVICE)
        full, _ = native_experts(dx, di, dw, lay)
        parts = [native_experts(dx[a:a + 512].contiguous(), di[a:a + 512].contiguous(), dw[a:a + 512].contiguous(),
                                lay)[0] for a in range(0, m, 512)]
        self.bit("prefill M4096 vs 512 chunks", torch.cat(parts), full)
        sample = torch.tensor([0, 1, 2, 3, 4, 1000, 2047, 3000, 4094, 4095])
        rout, _ = ref_experts(x[sample].contiguous(), ids[sample].contiguous(), w[sample].contiguous(), lay)
        self.bit("prefill M4096 sampled tokens vs ref", full[sample], rout)
        refused = oracle.call_status("cere_ktest_inkling_moe_experts", dx, H, m + 1, di, dw, *lay.weight_args(),
                                     torch.empty((256,), dtype=torch.uint8, device=oracle.DEVICE), 0, dx, H, None,
                                     None, None, 0)
        oracle.check_count(self, ROW, "refusal M4097", refused, expected=oracle.CERE_ERR_ARGUMENT)

    # -- recorded rows ------------------------------------------------------------------------------------------

    def test_recorded(self):
        for name, rec in recorded_sets():
            lay = layer(rec["layer"])
            xg = rec["mlp.gate:in0"].contiguous()
            ids, w, logits = native_router(xg.to(oracle.DEVICE), lay)
            rids, rw, rlogits = ref_router(xg, lay)
            self.bit(f"recorded {name} router ids", ids, rids)
            self.bit(f"recorded {name} router weights", w, rw)
            self.bit_float(f"recorded {name} router logits", logits, rlogits)
            # ref vs REF's recorded router (FP32 torch GEMM: another order)
            rec_ids, rec_w = sort_routing(rec["mlp.gate:out.2"], rec["mlp.gate:out.1"])
            same = int((rec_ids == rids).all(-1).sum().item())
            lrel = oracle.rel_l2(rlogits.double(), rec["mlp.gate:out.0"].double())
            oracle.report_row(ROW, f"recorded {name} ref router vs REF", phase="agreement", tokens=xg.shape[0],
                              same_sets=same, logits_rel_l2=lrel,
                              weights_max_abs=float((rw - rec_w).abs().max().item()) if same == xg.shape[0] else None)
            limit = oracle.port_limit(1e-5, "S4 router: FP32 reduce_dot<256> logits vs REF's FP32 torch GEMM")
            oracle.check_within(self, ROW, f"recorded {name} ref router logits vs REF", lrel, c=0.0, limit=limit)
            # experts on REF's routing (sorted to slot order)
            x = rec["mlp.experts:in0"].contiguous()
            eids, ew = sort_routing(rec["mlp.experts:in1"], rec["mlp.experts:in2"])
            out, taps = native_experts(x.to(oracle.DEVICE), eids.to(oracle.DEVICE), ew.to(oracle.DEVICE), lay,
                                       taps=True)
            rout, rtaps = ref_experts(x, eids, ew, lay)
            self.bit(f"recorded {name} experts h", taps[1], rtaps[1])
            self.bit(f"recorded {name} experts out", out, rout)
            gold = rec["mlp.experts:out"]
            rel = oracle.rel_l2(rout.double(), gold.double())
            ulps = bf16_ulps(rout, gold)
            oracle.report_row(ROW, f"recorded {name} ref experts vs REF golden arm Q", phase="agreement",
                              elements=gold.numel(), equal=int((ulps == 0).sum().item()),
                              within_1ulp=int((ulps <= 1).sum().item()), max_ulp=int(ulps.max().item()))
            limit = oracle.port_limit(1e-2, "S4 experts: CPU ref vs REF golden arm Q, same dequantized operands, "
                                            "FP32 GEMM sum order differs, BF16 roundings")
            oracle.check_within(self, ROW, f"recorded {name} ref experts vs REF golden arm Q", rel, c=0.0, limit=limit)

    # -- timing -------------------------------------------------------------------------------------------------

    def test_time(self):
        gen = torch.Generator().manual_seed(8)
        for n in (NVFP4_LAYER, MXFP8_LAYER):
            lay = layer(n)
            for m in (1, 8, 64):
                x = expert_inputs(m, gen).to(oracle.DEVICE)
                ids, w = oracle.empty((m, TOPK), torch.int32), oracle.empty((m, TOPK), torch.float32)
                nbytes = workspace_bytes(m, lay.nvfp4)
                ws = torch.empty((nbytes,), dtype=torch.uint8, device=oracle.DEVICE)
                out = oracle.empty((m, H), torch.bfloat16)
                us = torch.zeros((2,), dtype=torch.float32)
                oracle.call("cere_ktest_inkling_moe_time", x, m, lay.gate_w, lay.gate_b, ids, w, *lay.weight_args(),
                            ws, nbytes, out, 20, 200, us)
                oracle.report_row(ROW, f"time L{n} {'nvfp4' if lay.nvfp4 else 'mxfp8'} M{m}", phase="timing",
                                  router_us=float(us[0]), visit_us=float(us[1]))


if __name__ == "__main__":
    unittest.main()
