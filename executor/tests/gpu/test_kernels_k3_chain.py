"""K3Chain (tasks/k3chain-spec.md KC-02, KC-03): the hyper-connection chain combine_norm -> hc_act -> hc_gate with
early weights (`HcMixPath::kEarly`) is bit-identical to today's path (`kEpilogue`), and its cold-L2 cost per variant.

Part of suite `kernels_k3` (GPU0 or the GPU1 lane). Every case reports row "K3C". Real checkpoint weights of layers
0, 3 and 47 and the final mixer (no SGLang op: both sides are native and compared bit for bit). Seeded BF16 inputs: the
hyper state [m, 4 x 2560] with per-stream scales (1, 0.5, 2, 4) and the block output [m, 2560]; the combine's gate
input `normed` is the native per-stream norm of the hyper state (the mix that preceded the block).

Cases:
- `chain_<pair>_rows<R>_m<m>_{out,normed,mixed}`: `cere_ktest_hc_chain` (the forward's combine -> mix, `out` aliasing
  `hyper` as in the forward) by path 4 (kEarly) and path 1 (hc_mix's default) each equal path 3 (kEpilogue, main's
  default) bit for bit, for every m in 1..16 on the rows plans of 16 rows (batched rows, the verify m = 4 path) and
  on the plans made at max_m = m (rows 0; m = 1 is decode's instantiated plan), on the four combine -> mix pairs of
  the forward (attn combine -> mlp mix of a layer, mlp combine -> the next layer's attn mix or the final mixer).
- `mix_<site>_rows<R>_m<m>_{normed,mixed}`: `cere_ktest_hc_mix_small_m` (hc_mix: norm, then the mix; the forward's
  first mix and the final mixer) by paths 4 and 1 equal path 3 bit for bit, same rows and m.
- `early_opt_in_*` (audit KGM-4): the early launches' dynamic shared-memory opt-in is cached per kernel variant and
  context generation: a warm sweep of the L0 chain (m 1..16, rows 0 and 16) makes no `cudaFuncSetAttribute` call
  (`cere_ktest_hc_early_opt_ins` count 0; before KGM-4 every early launch made one), each of two context-generation
  advances makes the next sweep raise again the same 1..8 times, and every sweep's outputs equal the first's.
- `bench_rows<R>_m<m>` (report-only, phase `bench`; O97): `cere_ktest_hc_chain_us` cold (an l2_flush before each of
  BENCH_ITERS graph replays) of BENCH_CHAINS successive chains of the forward (layers 0..4, each chain on its own
  weights, as in the model), µs per chain (median and mean) for paths 3 and 4 and their ratio (means), at m 1 (rows 0,
  decode) and m 1, 4, 8, 16 (rows 16).

Native entry points (C prototypes; every call returns 0 on success, nonzero `cere_status` otherwise):

  int cere_ktest_hc_chain(const void *block_out,  /* bfloat16 [m, hidden] */
                          const void *hyper,      /* bfloat16 [m, hc * hidden]: the residual hyper state */
                          void *normed,           /* bfloat16 [m, hc * hidden]: in the combine gate, out the norm */
                          const void *w_inject,   /* bfloat16 [hc, hc * hidden] (block_inject_weight) */
                          const void *next_norm,  /* bfloat16 [hc * hidden] (the next mix's hc_norm.weight) */
                          const void *w_down,     /* bfloat16 [lowrank, hc * hidden] (the next mix's) */
                          const void *w_up,       /* bfloat16 [hc * hidden, lowrank] (the next mix's) */
                          int64_t m, int64_t hc, int64_t hidden, int64_t lowrank, float eps,
                          void *out,              /* bfloat16 [m, hc * hidden]: the next hyper state (may be hyper) */
                          void *mixed,            /* bfloat16 [m, hidden]: the next block input */
                          int64_t rows,           /* 0: plans at max_m = m; else the rows plans of `rows` rows */
                          int64_t path,           /* 1 default, 2 kUnfused, 3 kEpilogue, 4 kEarly */
                          cudaStream_t stream);
  int cere_ktest_hc_chain_us(<cere_ktest_hc_chain's arguments up to path, the four weights as stacks of `chains`
                             sets>, int64_t chains, int64_t timed, int64_t cold,
                             float *us /* host float32 [timed]: µs per replay of the chains */, cudaStream_t stream);
  int cere_ktest_hc_mix_small_m(hyper, norm_weight, w_down, w_up, m, hc, hidden, lowrank, eps, normed, mixed, rows,
                                path, stream);  /* as test_kernels_k3_hc.py, rows 0 and paths 3, 4 added */
  int cere_ktest_hc_early_opt_ins(int64_t next_generation, /* 1: advance the current ordinal's context generation */
                                  int64_t *out,            /* host int64 [1]: the early opt-in raises so far */
                                  cudaStream_t stream);
"""
from __future__ import annotations

import statistics
import unittest

import torch

import oracle

ROW = "K3C"
HC, HIDDEN, LOWRANK, EPS = 4, 2_560, 320, 1e-6
MIXER_PREFIX = "model.language_model.hyper_connection_mixer."
ATTN, MLP = "attn_hyper_connection", "mlp_hyper_connection"
# (name, combine site prefix, next mix prefix): the forward's combine -> mix pairs on layers 0, 3 and 47.
PAIRS = (
    ("L0_attn_L0_mlp", (0, ATTN), (0, MLP)),
    ("L0_mlp_L1_attn", (0, MLP), (1, ATTN)),
    ("L3_attn_L3_mlp", (3, ATTN), (3, MLP)),
    ("L47_mlp_mixer", (47, MLP), None),
)
MIX_SITES = (("L0_attn", (0, ATTN)), ("L47_mlp", (47, MLP)), ("mixer", None))
STREAM_SCALES = (1.0, 0.5, 2.0, 4.0)
MAX_ROWS = 16  # the small-M kernel's kMaxM, the rows plans of a verify step
ROWS_CASES = (0, MAX_ROWS)
DEFAULT_PATH, EPILOGUE_PATH, EARLY_PATH = 1, 3, 4
SEED = 170_000
BENCH_ITERS = 30
BENCH_CHAINS = 8
BENCH_CASES = ((0, 1), (MAX_ROWS, 1), (MAX_ROWS, 4), (MAX_ROWS, 8), (MAX_ROWS, 16))


def _prefix(site) -> str:
    return MIXER_PREFIX if site is None else f"{oracle.layer_prefix(site[0])}{site[1]}."


def _bf16(name: str) -> torch.Tensor:
    t = oracle.real_tensor(name)
    if t.dtype != torch.bfloat16:
        raise oracle.OracleError(f"{name} is {t.dtype}, expected torch.bfloat16")
    return t.contiguous()


def _mix_weights(site) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = _prefix(site)
    return (_bf16(p + "hc_norm.weight"), _bf16(p + "input_mix_weight_down.weight"),
            _bf16(p + "input_mix_weight_up.weight"))


def _hyper(m: int, seed: int) -> torch.Tensor:
    g = oracle.generator(seed)
    x = torch.randn((m, HC, HIDDEN), generator=g, device=oracle.DEVICE, dtype=torch.float32)
    scales = torch.tensor(STREAM_SCALES, dtype=torch.float32, device=oracle.DEVICE).view(1, HC, 1)
    return (x * scales).reshape(m, HC * HIDDEN).to(torch.bfloat16).contiguous()


def _block_out(m: int, seed: int) -> torch.Tensor:
    g = oracle.generator(seed)
    return torch.randn((m, HIDDEN), generator=g, device=oracle.DEVICE, dtype=torch.float32).to(torch.bfloat16)


def _mix(hyper, weights, rows: int, path: int):
    m = hyper.shape[0]
    normed = oracle.empty((m, HC * HIDDEN), torch.bfloat16)
    mixed = oracle.empty((m, HIDDEN), torch.bfloat16)
    oracle.call("cere_ktest_hc_mix_small_m", hyper, *weights, m, HC, HIDDEN, LOWRANK, float(EPS), normed, mixed,
                rows, path)
    return normed, mixed


class ChainCase:
    """One combine -> mix: inputs from seeds, the gate input `normed` from the combine site's own mix."""

    def __init__(self, combine_site, mix_site, m: int, seed: int):
        self.m = m
        self.w_inject = _bf16(_prefix(combine_site) + "block_inject_weight.weight")
        self.next_norm, self.w_down, self.w_up = _mix_weights(mix_site)
        self.hyper = _hyper(m, seed)
        self.block_out = _block_out(m, seed + 1)
        self.gate, _ = _mix(self.hyper, _mix_weights(combine_site), MAX_ROWS, EPILOGUE_PATH)

    def args(self, hyper, normed, mixed, rows: int, path: int) -> tuple:
        return (self.block_out, hyper, normed, self.w_inject, self.next_norm, self.w_down, self.w_up, self.m, HC,
                HIDDEN, LOWRANK, float(EPS), hyper, mixed, rows, path)


def _chain(case: ChainCase, rows: int, path: int):
    """The forward's chain: `out` aliases `hyper` (a fresh copy), `normed` starts as the gate input."""
    hyper = case.hyper.clone()
    normed = case.gate.clone()
    mixed = oracle.empty((case.m, HIDDEN), torch.bfloat16)
    oracle.call("cere_ktest_hc_chain", *case.args(hyper, normed, mixed, rows, path))
    return hyper, normed, mixed


def _bench_stacks() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The weights of BENCH_CHAINS successive combine -> mix pairs of the forward (layer L attn combine -> L mlp mix,
    L mlp combine -> L + 1 attn mix, from layer 0), stacked: w_inject, next_norm, w_down, w_up."""
    sites = [(layer, part) for layer in range(BENCH_CHAINS) for part in (ATTN, MLP)][:BENCH_CHAINS + 1]
    injects, norms, downs, ups = [], [], [], []
    for combine_site, mix_site in zip(sites, sites[1:]):
        injects.append(_bf16(_prefix(combine_site) + "block_inject_weight.weight"))
        norm, down, up = _mix_weights(mix_site)
        norms.append(norm)
        downs.append(down)
        ups.append(up)
    return tuple(torch.stack(t).contiguous() for t in (injects, norms, downs, ups))


def _chains_cold_us(stacks, m: int, rows: int, path: int) -> tuple[float, float]:
    """Cold-L2 µs per chain (span / BENCH_CHAINS) over BENCH_ITERS graph replays of BENCH_CHAINS chains: median,
    mean."""
    hyper = _hyper(m, SEED + 90_000 + m)
    block_out = _block_out(m, SEED + 91_000 + m)
    normed = oracle.empty((m, HC * HIDDEN), torch.bfloat16)
    normed.copy_(hyper)
    mixed = oracle.empty((m, HIDDEN), torch.bfloat16)
    w_inject, next_norm, w_down, w_up = stacks
    us = torch.zeros((BENCH_ITERS,), dtype=torch.float32)  # host: graph_spans writes the spans on the host
    oracle.call("cere_ktest_hc_chain_us", block_out, hyper, normed, w_inject, next_norm, w_down, w_up, m, HC, HIDDEN,
                LOWRANK, float(EPS), hyper, mixed, rows, path, BENCH_CHAINS, BENCH_ITERS, 1, us)
    spans = [v / BENCH_CHAINS for v in us.tolist()]
    return statistics.median(spans), statistics.fmean(spans)


def _early_opt_ins(next_generation: int) -> int:
    """`cere_ktest_hc_early_opt_ins`: the early launches' opt-in raises so far (after advancing the context
    generation when next_generation is 1)."""
    out = torch.zeros((1,), dtype=torch.int64)
    oracle.call("cere_ktest_hc_early_opt_ins", next_generation, out)
    return int(out[0])


class K3Chain(unittest.TestCase):
    def test_chain_bit_exact(self):
        """The chain by kEarly and by the default path equals kEpilogue bit for bit: out, normed and mixed."""
        for pair_index, (name, combine_site, mix_site) in enumerate(PAIRS):
            for rows in ROWS_CASES:
                for m in range(1, MAX_ROWS + 1):
                    with self.subTest(pair=name, rows=rows, m=m):
                        case = ChainCase(combine_site, mix_site, m, SEED + 1_000 * pair_index + 10 * m)
                        reference = _chain(case, rows, EPILOGUE_PATH)
                        for label, path in (("early", EARLY_PATH), ("default", DEFAULT_PATH)):
                            got = _chain(case, rows, path)
                            for part, a, b in zip(("out", "normed", "mixed"), got, reference):
                                oracle.check_bit_exact(self, ROW, f"chain_{name}_rows{rows}_m{m}_{label}_{part}",
                                                       a, b)

    def test_mix_bit_exact(self):
        """hc_mix (norm, then the mix) by kEarly and by the default path equals kEpilogue bit for bit."""
        for site_index, (name, site) in enumerate(MIX_SITES):
            weights = _mix_weights(site)
            for rows in ROWS_CASES:
                for m in range(1, MAX_ROWS + 1):
                    with self.subTest(site=name, rows=rows, m=m):
                        hyper = _hyper(m, SEED + 50_000 + 1_000 * site_index + m)
                        reference = _mix(hyper, weights, rows, EPILOGUE_PATH)
                        for label, path in (("early", EARLY_PATH), ("default", DEFAULT_PATH)):
                            got = _mix(hyper, weights, rows, path)
                            for part, a, b in zip(("normed", "mixed"), got, reference):
                                oracle.check_bit_exact(self, ROW, f"mix_{name}_rows{rows}_m{m}_{label}_{part}", a, b)

    def test_early_opt_in_cached(self):
        """KGM-4: the early launches raise a kernel variant's dynamic shared-memory limit once per context generation.
        A second sweep of the chain (every m, rows 0 and 16) raises nothing; after the context generation advances
        (a recorded primary-context reset) the next sweep raises again, the same number of times after each of two
        advances, at most once per variant of the two launch sites (8); every sweep's outputs equal the first's."""
        _, combine_site, mix_site = PAIRS[0]
        cases = [ChainCase(combine_site, mix_site, m, SEED + 80_000 + 10 * m) for m in range(1, MAX_ROWS + 1)]

        def sweep():
            return [_chain(case, rows, EARLY_PATH) for rows in ROWS_CASES for case in cases]

        first = sweep()  # raises every variant this process has not raised yet
        warm = _early_opt_ins(0)
        sweeps = {"warm": sweep()}
        oracle.check_count(self, ROW, "early_opt_in_warm_raises", _early_opt_ins(0) - warm, expected=0)
        raises = []
        for label in ("reset1", "reset2"):
            start = _early_opt_ins(1)
            sweeps[label] = sweep()
            raises.append(_early_opt_ins(0) - start)
        oracle.report_row(ROW, "early_opt_in_reset_raises", phase="count", reset1=raises[0], reset2=raises[1])
        oracle.check_count(self, ROW, "early_opt_in_reset_raises_equal", raises[1], expected=raises[0])
        oracle.check_count(self, ROW, "early_opt_in_reset_raises_in_1_8", int(1 <= raises[0] <= 8), expected=1)
        for label, got in sweeps.items():
            for i, (a, b) in enumerate(zip(got, first)):
                for part, x, y in zip(("out", "normed", "mixed"), a, b):
                    oracle.check_bit_exact(self, ROW, f"early_opt_in_{label}_{i}_{part}", x, y)

    def test_bench(self):
        """Report-only (O97): cold-L2 µs per chain over BENCH_CHAINS successive chains, kEpilogue (main) vs kEarly."""
        stacks = _bench_stacks()
        for rows, m in BENCH_CASES:
            fields = {}
            for label, path in (("epilogue", EPILOGUE_PATH), ("early", EARLY_PATH)):
                fields[f"{label}_us"], fields[f"{label}_mean_us"] = _chains_cold_us(stacks, m, rows, path)
            oracle.report_row(ROW, f"bench_rows{rows}_m{m}", phase="bench", m=m, rows=rows, iters=BENCH_ITERS,
                              chains=BENCH_CHAINS, early_speedup=fields["epilogue_mean_us"] / fields["early_mean_us"],
                              **fields)


if __name__ == "__main__":
    unittest.main()
