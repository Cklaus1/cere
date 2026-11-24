"""K13: the BF16 MoE block of the MTP draft layer (tasks/m29-spec.md D3, Tier 1b K2; `kernels/k13_moe_bf16.h`).

Suite `kernels_k13` (GPU1 lane: `native_dev.py --gpu 1 test --suite kernels_k13`; also part of GPU0's `kernels`):
`python3 -m unittest discover -s /src/executor/tests/gpu -p 'test_kernels_k13_*.py' -v`.

Reference: `oracle.moe_bf16_reference` (BF16 weights, FP64 accumulation, BF16 rounding points of the block). There is
no SGLang control for the MTP layer's MoE (D3: draft numerics only change which tokens are proposed), so the rows
are oracle rows with a pre-registered port limit (c = 0).

Synthetic weights (seeded, the MTP layer's dims: H 2,560, 512 experts, top-10, I 640, S 640): router rows of unit
norm, expert and shared weights N(0, 1/k). Inputs plant the routing: x_t = 0.1 g_t + sum_j s_j router[e_tj] with
s_j = 9.0 - 0.5 j, so the logits of the 10 planted experts (about s_j) sit at least ~2 above every other expert's and
the selected set does not depend on BF16 rounding of the logits; case `random` draws e_t per row, case `same` routes
every row to the same 10 experts (each expert then holds all m rows).

Native entry points (C prototypes; every call returns 0 on success, nonzero `cere_status` otherwise):

  int cere_ktest_moe_bf16(const void *x,                   /* bfloat16 [m, hidden] */
                          const void *router_weight,       /* bfloat16 [num_experts, hidden] (mlp.gate.weight) */
                          const void *gate_up_weight,      /* bfloat16 [num_experts, 2 inter, hidden], gate rows
                                                              first (mlp.experts.gate_up_proj) */
                          const void *down_weight,         /* bfloat16 [num_experts, hidden, inter]
                                                              (mlp.experts.down_proj) */
                          const void *shared_gate_weight,  /* bfloat16 [shared_inter, hidden] */
                          const void *shared_up_weight,    /* bfloat16 [shared_inter, hidden] */
                          const void *shared_down_weight,  /* bfloat16 [hidden, shared_inter] */
                          const void *shared_expert_gate,  /* bfloat16 [hidden] */
                          void *out,                       /* bfloat16 [m, hidden] */
                          int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter,
                          int64_t shared_inter,
                          int64_t rows,                    /* 0: the K2 plans pinned at max_m = m; [1, 16] with
                                                              m <= rows: the executor's rows plans (init_rows) */
                          int64_t first,                   /* [0, m): the routed experts and the combine run on
                                                              rows [first, m) only; rows [0, first) of out are not
                                                              written (the MTP extend's last_only call) */
                          cudaStream_t stream);
      out = moe_bf16_block(x): router (K2 BF16 logits, K7 moe_route), grouped BF16 expert GEMMs with SwiGLU, K7's
      combine with the sigmoid-gated shared expert.

  int cere_ktest_moe_bf16_time(... the cere_ktest_moe_bf16 arguments up to `first` ...,
                               int64_t iterations,         /* >= 1 timed calls after one warm-up call */
                               void *elapsed_ms,           /* float32 [1]: mean milliseconds per call (CUDA
                                                              events; plans and workspace made outside) */
                               cudaStream_t stream);

Rows: `K13` block vs the oracle at m in {1, 4, 16, 256, 4,096} (cases random and same); row r of every m in [2, 16]
on the rows plans bit-equal to the m = 1 call on that row; run-to-run bit equality; a `first` call's rows [first, m)
bit-equal to the first = 0 call's and its rows [0, first) left as they were, first outside [0, m) refused;
report-only call times.
"""
from __future__ import annotations

import functools
import unittest

import torch

import oracle

ROW = "K13"
HIDDEN = 2560
EXPERTS = 512
TOP_K = 10
INTER = 640
SHARED = 640
ROWS = 16  # GemmPlan::kMaxRows, the rows plans of decode/verify/draft-extend (m29-spec D4/M29S2-05)
SEED = 2913
ORACLE_M = (1, 4, 16, 256, 4096)
CASES = ("random", "same")
NOISE = 0.1


def _generator(seed: int) -> torch.Generator:
    g = torch.Generator(device=oracle.DEVICE)
    g.manual_seed(seed)
    return g


def _randn_bf16(shape: tuple[int, ...], scale: float, g: torch.Generator) -> torch.Tensor:
    """N(0, scale^2) in bfloat16, drawn in slabs of the leading dimension (bounded FP32 staging)."""
    out = torch.empty(shape, dtype=torch.bfloat16, device=oracle.DEVICE)
    flat = out.view(shape[0], -1)
    for i in range(0, shape[0], 32):
        slab = flat[i:i + 32]
        slab.copy_(torch.randn(slab.shape, generator=g, device=oracle.DEVICE) * scale)
    return out


@functools.cache
def _weights() -> dict[str, torch.Tensor]:
    g = _generator(SEED)
    router = torch.randn((EXPERTS, HIDDEN), generator=g, device=oracle.DEVICE)
    return {
        "router": (router / router.norm(dim=1, keepdim=True)).to(torch.bfloat16),
        "gate_up": _randn_bf16((EXPERTS, 2 * INTER, HIDDEN), HIDDEN ** -0.5, g),
        "down": _randn_bf16((EXPERTS, HIDDEN, INTER), INTER ** -0.5, g),
        "shared_gate": _randn_bf16((SHARED, HIDDEN), HIDDEN ** -0.5, g),
        "shared_up": _randn_bf16((SHARED, HIDDEN), HIDDEN ** -0.5, g),
        "shared_down": _randn_bf16((HIDDEN, SHARED), SHARED ** -0.5, g),
        "shared_expert_gate": _randn_bf16((HIDDEN,), HIDDEN ** -0.5, g),
    }


@functools.cache
def _input(case: str, m: int) -> torch.Tensor:
    """x bfloat16 [m, H] whose top-10 experts are planted (module docstring)."""
    g = _generator(SEED + 1 + CASES.index(case) * 100_003 + m)
    router = _weights()["router"].float()
    if case == "same":
        experts = torch.randperm(EXPERTS, generator=g, device=oracle.DEVICE)[:TOP_K].expand(m, TOP_K).contiguous()
    else:
        experts = torch.rand((m, EXPERTS), generator=g, device=oracle.DEVICE).argsort(dim=1)[:, :TOP_K]
    strength = 9.0 - 0.5 * torch.arange(TOP_K, device=oracle.DEVICE, dtype=torch.float32)
    coefficients = torch.zeros((m, EXPERTS), device=oracle.DEVICE)
    coefficients.scatter_(1, experts, strength.expand(m, TOP_K).contiguous())
    noise = torch.randn((m, HIDDEN), generator=g, device=oracle.DEVICE)
    return (NOISE * noise + coefficients @ router).to(torch.bfloat16).contiguous()


def _native(x: torch.Tensor, rows: int = 0, first: int = 0, out: torch.Tensor | None = None) -> torch.Tensor:
    w = _weights()
    if out is None:
        out = oracle.empty((x.shape[0], HIDDEN), torch.bfloat16)
    oracle.call("cere_ktest_moe_bf16", x, w["router"], w["gate_up"], w["down"], w["shared_gate"], w["shared_up"],
                w["shared_down"], w["shared_expert_gate"], out, x.shape[0], HIDDEN, EXPERTS, TOP_K, INTER, SHARED,
                rows, first)
    return out


def _native_time(x: torch.Tensor, rows: int, iterations: int, first: int = 0) -> float:
    w = _weights()
    out = oracle.empty((x.shape[0], HIDDEN), torch.bfloat16)
    elapsed = oracle.empty((1,), torch.float32)
    oracle.call("cere_ktest_moe_bf16_time", x, w["router"], w["gate_up"], w["down"], w["shared_gate"],
                w["shared_up"], w["shared_down"], w["shared_expert_gate"], out, x.shape[0], HIDDEN, EXPERTS, TOP_K,
                INTER, SHARED, rows, first, iterations, elapsed)
    return float(elapsed.item())


def _reference(x: torch.Tensor) -> torch.Tensor:
    w = _weights()
    return oracle.moe_bf16_reference(x, w["router"], w["gate_up"], w["down"], w["shared_gate"], w["shared_up"],
                                     w["shared_down"], w["shared_expert_gate"], TOP_K)


class MoeBf16(unittest.TestCase):
    def test_block_vs_oracle(self) -> None:
        limit = oracle.port_limit(1e-2, "M29 K2: oracle row, BF16 weights with FP64-accumulated reference at "
                                        "the block's BF16 rounding points, FP32 sum order and SiLU ulps only")
        for case in CASES:
            for m in ORACLE_M:
                with self.subTest(case=case, m=m):
                    x = _input(case, m)
                    reference = _reference(x)
                    native = _native(x)
                    oracle.check_within(self, ROW, f"{case}/m{m}", oracle.rel_l2(native, reference), c=0.0,
                                        limit=limit)

    def test_rows_plans_vs_oracle(self) -> None:
        limit = oracle.port_limit(1e-2, "M29 K2: the same oracle row on the rows plans of decode and verify")
        for case in CASES:
            x = _input(case, ROWS)
            reference = _reference(x)
            for m in (1, 4, ROWS):
                with self.subTest(case=case, m=m):
                    native = _native(x[:m], rows=ROWS)
                    oracle.check_within(self, ROW, f"{case}/rows{ROWS}/m{m}",
                                        oracle.rel_l2(native, reference[:m]), c=0.0, limit=limit)

    def test_rows_bit_equal_to_single_row(self) -> None:
        for case in CASES:
            x = _input(case, ROWS)
            single = torch.cat([_native(x[r:r + 1], rows=ROWS) for r in range(ROWS)])
            for m in range(2, ROWS + 1):
                with self.subTest(case=case, m=m):
                    oracle.check_bit_exact(self, ROW, f"{case}/rows{ROWS}/m{m}_vs_m1", _native(x[:m], rows=ROWS),
                                           single[:m])

    def test_run_to_run_bit_reproducible(self) -> None:
        for case, m, rows in (("random", 4096, 0), ("same", 4096, 0), ("random", ROWS, ROWS), ("same", ROWS, ROWS)):
            with self.subTest(case=case, m=m, rows=rows):
                x = _input(case, m)
                oracle.check_bit_exact(self, ROW, f"{case}/rows{rows}/m{m}_rerun", _native(x, rows), _native(x, rows))

    def test_first_rows_bit_equal_and_head_untouched(self) -> None:
        """The MTP extend's last_only call (model/mtp.cu): rows [first, m) of a `first` call are bit-equal to the
        full call's (the router and the shared expert still run on all m rows, the routed experts and the combine
        on the tail alone) and rows [0, first) of `out` keep the caller's bytes."""
        for case, m, rows in (("random", 4096, 0), ("same", 4096, 0), ("random", 256, 0), ("random", ROWS, ROWS),
                              ("same", 5, ROWS)):
            x = _input(case, m)
            full = _native(x, rows)
            for first in sorted({1, m // 2, m - 1}):
                with self.subTest(case=case, m=m, rows=rows, first=first):
                    sentinel = torch.full((m, HIDDEN), 7.0, dtype=torch.bfloat16, device=oracle.DEVICE)
                    tail = _native(x, rows, first, out=sentinel.clone())
                    oracle.check_bit_exact(self, ROW, f"{case}/rows{rows}/m{m}/first{first}_tail", tail[first:],
                                           full[first:])
                    oracle.check_bit_exact(self, ROW, f"{case}/rows{rows}/m{m}/first{first}_head", tail[:first],
                                           sentinel[:first])

    def test_first_outside_rows_refused(self) -> None:
        w = _weights()
        x = _input("random", ROWS)
        out = oracle.empty((ROWS, HIDDEN), torch.bfloat16)
        for first in (-1, ROWS):
            with self.subTest(first=first):
                status = oracle.call_status("cere_ktest_moe_bf16", x, w["router"], w["gate_up"], w["down"],
                                            w["shared_gate"], w["shared_up"], w["shared_down"],
                                            w["shared_expert_gate"], out, ROWS, HIDDEN, EXPERTS, TOP_K, INTER, SHARED,
                                            ROWS, first)
                self.assertEqual(status, oracle.CERE_ERR_ARGUMENT)

    def test_call_time(self) -> None:
        """Report-only (no verdict): mean milliseconds per block call, decode m on the rows plans."""
        for m, rows, iterations, first in ((1, ROWS, 50, 0), (4, ROWS, 50, 0), (16, ROWS, 50, 0), (256, 0, 10, 0),
                                           (4096, 0, 5, 0), (4096, 0, 5, 4095)):
            with self.subTest(m=m, first=first):
                ms = _native_time(_input("random", m), rows, iterations, first)
                oracle.report_row(ROW, f"random/rows{rows}/m{m}/first{first}", phase="time", ms=ms,
                                  iterations=iterations)


if __name__ == "__main__":
    unittest.main()
