"""K13W: the MTP layer's MoE block with NVFP4 W4A16 routed experts (track Fp4S3, tasks/fp4local-spec.md "S3 perf
plan"; `kernels/k13_moe_bf16.h` MoeBf16Layer gate_up_q ...).

Suite `kernels_k13w` (GPU1 lane: `native_dev.py --gpu 1 test --suite kernels_k13w`):
`python3 -m unittest discover -s /src/executor/tests/gpu -p 'test_kernels_k13w_*.py' -v`.

Contract: moe_bf16_block on the packed experts (E2M1 codes, E4M3 block scales of 16, F32 globals) decodes each weight
in-kernel to BF16_RNE(RN(RN(e2m1 * scale) * global)), the BF16 tensor `cerec convert`'s S2 path stores for it
(convert.rs nvfp4_to_bf16), and then runs the BF16 kernel's accumulation; so its output is bit-equal to the
cere_ktest_moe_bf16 call on the S2-dequantized tensors. Every row is bit-exact (raw bytes) against that call, with
the S2 dequantization done here in torch (two separately rounded FP32 products, then BF16 round-to-nearest-even):
- `<case>/rows16/m1-16`: the executor's rows plans, m 1..16 (routed tails of at most 16 rows: the small-tail grouping
  and GEMVs, tasks/draft-span-spec.md DS-2); `<case>/init/m<m>`: plans pinned at max_m = m, m 1, 4, 16, 256 (m 256:
  the general grouped GEMVs);
- `<case>/rows<r>/m<m>/first<f>`: the MTP extend's `first` call (rows [first, m) routed; out compared whole, rows
  [0, first) keep the caller's bytes on both sides).
Synthetic seeded weights with the MTP layer's dims (H 2,560, 512 experts, top-10, I 640, S 640): the K13 suite's
router and planted routing (test_kernels_k13_moe_bf16.py, same seed), a shared expert N(0, 1/k); packed bytes uniform
(every E2M1 code at both nibbles, +-0 included), block scales E4M3 0.25..4 with blocks of subnormal and zero scales and
negative ones, globals uniform per expert and projection (gate, up, down), so the second product rounds.

Fail-first: `perturbed/*` changes one gate/up and one down block-scale byte of a routed expert on the W4A16 side and
requires a difference from the unperturbed BF16 call; against a library without these entry points every test fails
(symbol absent).

Report-only rows `time/m<m>` (m 1, 4, rows plans): cold-L2 (ktest/l2_flush.h) median and minimum us of one block call,
W4A16 vs BF16, with GB/s of the bytes each reads (router, shared expert, the routed experts' weights + scales +
globals, x and out).

Native entry points (C prototypes; every call returns 0 on success, nonzero `cere_status` otherwise):

  int cere_ktest_moe_bf16(const void *x, const void *router_weight, const void *gate_up_weight,
                          const void *down_weight, const void *shared_gate_weight, const void *shared_up_weight,
                          const void *shared_down_weight, const void *shared_expert_gate, void *out, int64_t m,
                          int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter, int64_t shared_inter,
                          int64_t rows, int64_t first, cudaStream_t stream);
      as declared by test_kernels_k13_moe_bf16.py.

  int cere_ktest_moe_bf16_cold(... the cere_ktest_moe_bf16 arguments up to `first` ...,
                               int64_t iterations,   /* >= 1 cold-L2 calls after one untimed call */
                               void *us,             /* host float64 [2]: median, min us of one call */
                               cudaStream_t stream);

  int cere_ktest_moe_w4a16(const void *x,               /* bfloat16 [m, hidden] */
                           const void *router_weight,   /* bfloat16 [num_experts, hidden] */
                           const void *gate_up_q,       /* uint8 [num_experts, 2 inter, hidden / 2] E2M1 pairs,
                                                           element 2j in the low nibble; gate rows first */
                           const void *gate_up_scale,   /* uint8 E4M3 [num_experts, 2 inter, hidden / 16] */
                           const void *gate_up_global,  /* float32 [num_experts, 2] (gate, up) */
                           const void *down_q,          /* uint8 [num_experts, hidden, inter / 2] */
                           const void *down_scale,      /* uint8 E4M3 [num_experts, hidden, inter / 16] */
                           const void *down_global,     /* float32 [num_experts] */
                           const void *shared_gate_weight, const void *shared_up_weight,
                           const void *shared_down_weight, const void *shared_expert_gate, void *out, int64_t m,
                           int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter, int64_t shared_inter,
                           int64_t rows, int64_t first,
                           int64_t iterations,          /* 0: one call; > 0: cold-L2 timed */
                           void *us,                    /* host float64 [2] (iterations > 0) */
                           cudaStream_t stream);
"""
from __future__ import annotations

import functools
import unittest

import torch

import oracle

ROW = "K13W"
HIDDEN = 2560
EXPERTS = 512
TOP_K = 10
INTER = 640
SHARED = 640
ROWS = 16  # GemmPlan::kMaxRows
SEED = 2913  # the K13 suite's router and routing
NVFP4_SEED = 3_130_000
BLOCK = 16
CASES = ("random", "same")
NOISE = 0.1
INIT_M = (1, 4, 16, 256)
TIME_M = (1, 4)
COLD_ITERS = 40
SLAB = 16  # experts per generation / dequantization slab
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _randn_bf16(shape: tuple[int, ...], scale: float, g: torch.Generator) -> torch.Tensor:
    out = torch.empty(shape, dtype=torch.bfloat16, device=oracle.DEVICE)
    flat = out.view(shape[0], -1)
    for i in range(0, shape[0], 32):
        slab = flat[i:i + 32]
        slab.copy_(torch.randn(slab.shape, generator=g, device=oracle.DEVICE) * scale)
    return out


@functools.cache
def _bf16_shared() -> dict[str, torch.Tensor]:
    """The K13 suite's router (same seed, first draw), then the shared expert N(0, 1/k) (no BF16 routed experts)."""
    g = oracle.generator(SEED)
    router = torch.randn((EXPERTS, HIDDEN), generator=g, device=oracle.DEVICE)
    return {
        "router": (router / router.norm(dim=1, keepdim=True)).to(torch.bfloat16),
        "shared_gate": _randn_bf16((SHARED, HIDDEN), HIDDEN ** -0.5, g),
        "shared_up": _randn_bf16((SHARED, HIDDEN), HIDDEN ** -0.5, g),
        "shared_down": _randn_bf16((HIDDEN, SHARED), SHARED ** -0.5, g),
        "shared_expert_gate": _randn_bf16((HIDDEN,), HIDDEN ** -0.5, g),
    }


def _block_scales(shape: tuple[int, int, int], first: int, g: torch.Generator) -> torch.Tensor:
    """E4M3 bytes 0x28..0x48 (0.25..4); blocks with (row + block + expert) = 0 mod 23 subnormal or zero (0x00..0x07),
    (row + 2 block) = 0 mod 29 negative (sign bit set)."""
    s = torch.randint(0x28, 0x49, shape, generator=g, device=oracle.DEVICE, dtype=torch.int32)
    e = torch.arange(first, first + shape[0], device=oracle.DEVICE, dtype=torch.int32)[:, None, None]
    r = torch.arange(shape[1], device=oracle.DEVICE, dtype=torch.int32)[None, :, None]
    b = torch.arange(shape[2], device=oracle.DEVICE, dtype=torch.int32)[None, None, :]
    s = torch.where((r + b + e) % 23 == 0, s & 0x07, s)
    return torch.where((r + 2 * b) % 29 == 0, s | 0x80, s).to(torch.uint8)


@functools.cache
def _nvfp4() -> dict[str, torch.Tensor]:
    """The packed experts (module docstring)."""
    g = oracle.generator(NVFP4_SEED)
    out = {
        "gate_up_q": torch.empty((EXPERTS, 2 * INTER, HIDDEN // 2), dtype=torch.uint8, device=oracle.DEVICE),
        "gate_up_scale": torch.empty((EXPERTS, 2 * INTER, HIDDEN // BLOCK), dtype=torch.uint8, device=oracle.DEVICE),
        "down_q": torch.empty((EXPERTS, HIDDEN, INTER // 2), dtype=torch.uint8, device=oracle.DEVICE),
        "down_scale": torch.empty((EXPERTS, HIDDEN, INTER // BLOCK), dtype=torch.uint8, device=oracle.DEVICE),
    }
    for e in range(0, EXPERTS, SLAB):
        for name in ("gate_up_q", "down_q"):
            t = out[name][e:e + SLAB]
            t.copy_(torch.randint(0, 256, t.shape, generator=g, device=oracle.DEVICE, dtype=torch.int32))
        for name in ("gate_up_scale", "down_scale"):
            t = out[name][e:e + SLAB]
            t.copy_(_block_scales(tuple(t.shape), e, g))
    out["gate_up_global"] = (0.002 + 0.004 * torch.rand((EXPERTS, 2), generator=g, device=oracle.DEVICE)).contiguous()
    out["down_global"] = (0.004 + 0.008 * torch.rand((EXPERTS,), generator=g, device=oracle.DEVICE)).contiguous()
    return out


def _s2_dequantize(q: torch.Tensor, scale: torch.Tensor, row_global: torch.Tensor) -> torch.Tensor:
    """BF16_RNE(RN(RN(e2m1 * scale) * global)) of packed q [e, R, K / 2], E4M3 scale [e, R, K / 16], row_global
    float32 [e, R] (convert.rs nvfp4_to_bf16; the first product is exact)."""
    table = torch.tensor(E2M1, dtype=torch.float32, device=oracle.DEVICE)
    codes = torch.stack((q & 0x0F, q >> 4), dim=-1).reshape(q.shape[0], q.shape[1], -1).long()
    product = table[codes] * scale.view(torch.float8_e4m3fn).to(torch.float32).repeat_interleave(BLOCK, dim=-1)
    return (product * row_global[:, :, None]).to(torch.bfloat16)


@functools.cache
def _s2_experts() -> dict[str, torch.Tensor]:
    """The S2 BF16 expert tensors of the packed experts (gate_up [E, 2 I, H], down [E, H, I])."""
    p = _nvfp4()
    gate_up = torch.empty((EXPERTS, 2 * INTER, HIDDEN), dtype=torch.bfloat16, device=oracle.DEVICE)
    down = torch.empty((EXPERTS, HIDDEN, INTER), dtype=torch.bfloat16, device=oracle.DEVICE)
    for e in range(0, EXPERTS, SLAB):
        sl = slice(e, e + SLAB)
        gate_up[sl] = _s2_dequantize(p["gate_up_q"][sl], p["gate_up_scale"][sl],
                                     p["gate_up_global"][sl].repeat_interleave(INTER, dim=1))
        down[sl] = _s2_dequantize(p["down_q"][sl], p["down_scale"][sl],
                                  p["down_global"][sl][:, None].expand(-1, HIDDEN))
    return {"gate_up": gate_up, "down": down}


@functools.cache
def _routed(case: str, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(x bfloat16 [m, H], the planted experts [m, TOP_K]): the K13 suite's planted routing."""
    g = oracle.generator(SEED + 1 + CASES.index(case) * 100_003 + m)
    router = _bf16_shared()["router"].float()
    if case == "same":
        experts = torch.randperm(EXPERTS, generator=g, device=oracle.DEVICE)[:TOP_K].expand(m, TOP_K).contiguous()
    else:
        experts = torch.rand((m, EXPERTS), generator=g, device=oracle.DEVICE).argsort(dim=1)[:, :TOP_K]
    strength = 9.0 - 0.5 * torch.arange(TOP_K, device=oracle.DEVICE, dtype=torch.float32)
    coefficients = torch.zeros((m, EXPERTS), device=oracle.DEVICE)
    coefficients.scatter_(1, experts, strength.expand(m, TOP_K).contiguous())
    noise = torch.randn((m, HIDDEN), generator=g, device=oracle.DEVICE)
    return (NOISE * noise + coefficients @ router).to(torch.bfloat16).contiguous(), experts


def _out(m: int, sentinel: bool) -> torch.Tensor:
    if sentinel:
        return torch.full((m, HIDDEN), 7.0, dtype=torch.bfloat16, device=oracle.DEVICE)
    return oracle.empty((m, HIDDEN), torch.bfloat16)


def _bf16_call(x: torch.Tensor, rows: int, first: int = 0, iterations: int = 0) -> tuple[torch.Tensor, float, float]:
    """(out, median us, min us): cere_ktest_moe_bf16 on the S2 tensors, or its cold-L2 timing (iterations > 0)."""
    s, w = _bf16_shared(), _s2_experts()
    m = x.shape[0]
    out = _out(m, first > 0)
    us = torch.zeros((2,), dtype=torch.float64)
    args = (x, s["router"], w["gate_up"], w["down"], s["shared_gate"], s["shared_up"], s["shared_down"],
            s["shared_expert_gate"], out, m, HIDDEN, EXPERTS, TOP_K, INTER, SHARED, rows, first)
    if iterations > 0:
        oracle.call("cere_ktest_moe_bf16_cold", *args, iterations, us)
    else:
        oracle.call("cere_ktest_moe_bf16", *args)
    return out, float(us[0]), float(us[1])


def _w4a16_call(x: torch.Tensor, rows: int, first: int = 0, iterations: int = 0,
                packed: dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, float, float]:
    """(out, median us, min us) of cere_ktest_moe_w4a16 on the packed experts (or `packed` instead)."""
    s, p = _bf16_shared(), packed if packed is not None else _nvfp4()
    m = x.shape[0]
    out = _out(m, first > 0)
    us = torch.zeros((2,), dtype=torch.float64)
    oracle.call("cere_ktest_moe_w4a16", x, s["router"], p["gate_up_q"], p["gate_up_scale"], p["gate_up_global"],
                p["down_q"], p["down_scale"], p["down_global"], s["shared_gate"], s["shared_up"], s["shared_down"],
                s["shared_expert_gate"], out, m, HIDDEN, EXPERTS, TOP_K, INTER, SHARED, rows, first, iterations,
                us if iterations > 0 else None)
    return out, float(us[0]), float(us[1])


def _differing(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum().item())


def _perturbed(name: str, expert: int) -> dict[str, torch.Tensor]:
    """The packed experts with one byte of `name` (a block scale) of `expert` changed: its first block whose codes are
    not all +-0 gets scale 4.0 (0x48), or 2.0 (0x40) if it had 4.0."""
    p = dict(_nvfp4())
    q = p[name.replace("_scale", "_q")][expert]
    nonzero = (q.to(torch.int32) & 0x77).view(q.shape[0], -1, BLOCK // 2).amax(dim=-1) != 0
    r, b = (int(v) for v in nonzero.nonzero()[0])
    scale = p[name].clone()
    scale[expert, r, b] = 0x40 if int(scale[expert, r, b]) == 0x48 else 0x48
    p[name] = scale
    return p


class MoeW4a16(unittest.TestCase):
    def test_rows_plans_bit_equal_to_bf16(self) -> None:
        for case in CASES:
            x, _ = _routed(case, ROWS)
            with self.subTest(case=case):
                native = torch.cat([_w4a16_call(x[:m], ROWS)[0].reshape(-1) for m in range(1, ROWS + 1)])
                reference = torch.cat([_bf16_call(x[:m], ROWS)[0].reshape(-1) for m in range(1, ROWS + 1)])
                oracle.check_bit_exact(self, ROW, f"{case}/rows{ROWS}/m1-{ROWS}", native, reference)

    def test_init_plans_bit_equal_to_bf16(self) -> None:
        for case in CASES:
            for m in INIT_M:
                with self.subTest(case=case, m=m):
                    x, _ = _routed(case, m)
                    oracle.check_bit_exact(self, ROW, f"{case}/init/m{m}", _w4a16_call(x, 0)[0], _bf16_call(x, 0)[0])

    def test_mtp_extend_first_bit_equal_to_bf16(self) -> None:
        for case, m, rows, firsts in (("random", ROWS, ROWS, (1, 8, 15)), ("same", 5, ROWS, (4,)),
                                      ("random", 256, 0, (255,))):
            x, _ = _routed(case, m)
            for first in firsts:
                with self.subTest(case=case, m=m, rows=rows, first=first):
                    oracle.check_bit_exact(self, ROW, f"{case}/rows{rows}/m{m}/first{first}",
                                           _w4a16_call(x, rows, first)[0], _bf16_call(x, rows, first)[0])

    def test_perturbed_scale_is_detected(self) -> None:
        """Fail-first: the comparison above catches one changed block-scale byte of a routed expert."""
        x, experts = _routed("same", 4)
        reference = _bf16_call(x, ROWS)[0]
        for name in ("gate_up_scale", "down_scale"):
            with self.subTest(scale=name):
                expert = int(experts[0, 0])
                differing = _differing(_w4a16_call(x, ROWS, packed=_perturbed(name, expert))[0], reference)
                oracle.report_row(ROW, f"perturbed/{name}", phase="perturbed", expert=expert, differing=differing,
                                  detected=differing > 0)
                self.assertNotEqual(differing, 0, f"a changed {name} byte of expert {expert} went unnoticed")

    def test_cold_l2_time(self) -> None:
        """Report-only (no verdict): cold-L2 us and GB/s of one block call on the rows plans (module docstring)."""
        for m in TIME_M:
            with self.subTest(m=m):
                x, experts = _routed("random", m)
                active = int(torch.unique(experts).numel())
                common = 2 * EXPERTS * HIDDEN + 2 * 3 * SHARED * HIDDEN + 2 * HIDDEN + 2 * 2 * m * HIDDEN
                w4_expert = (2 * INTER * HIDDEN // 2 + 2 * INTER * HIDDEN // BLOCK + 8
                             + HIDDEN * INTER // 2 + HIDDEN * INTER // BLOCK + 4)
                bf_expert = 2 * (2 * INTER * HIDDEN + HIDDEN * INTER)
                _, w4_us, w4_min = _w4a16_call(x, ROWS, iterations=COLD_ITERS)
                _, bf_us, bf_min = _bf16_call(x, ROWS, iterations=COLD_ITERS)
                oracle.report_row(ROW, f"time/m{m}", phase="time", m=m, active_experts=active,
                                  w4a16_us=w4_us, w4a16_min_us=w4_min,
                                  w4a16_gbps=(common + active * w4_expert) / w4_us / 1e3,
                                  bf16_us=bf_us, bf16_min_us=bf_min,
                                  bf16_gbps=(common + active * bf_expert) / bf_us / 1e3, speedup=bf_us / w4_us)


if __name__ == "__main__":
    unittest.main()
