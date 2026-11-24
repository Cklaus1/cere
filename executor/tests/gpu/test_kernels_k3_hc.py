"""K3 (tasks/m22-spec.md): hyper-connection mix/combine and the final mix-only mixer vs SGLang on real weights.

Suite `kernels` (inside the pinned SGLang image on GPU0):
`python3 -m unittest discover -s /src/executor/tests/gpu -p 'test_kernels_*.py' -v`.

Every case reports row "K3". Tolerance rule: `c` = rel-L2(SGLang op vs an FP64 torch reference on the same inputs),
written to the report before the native call; limit = max(FLOOR_OP, 2c); measured = rel-L2(native vs SGLang op).
Every native case runs twice and the two results must be bit-identical.

SGLang op ($S181 = the reference image's site-packages): a `GatedResidual` (`sglang/srt/layers/hyperconnection.py:117`)
built exactly as the model builds it - `HyperConnectionConfig(hc_count=4, hidden_size=2560, params_dtype=bfloat16,
hc_lowrank=320, rms_norm_eps=1e-6, hc_per_branch_norm=True)` (`sglang/srt/models/qwen4_exp.py:1257-1273` for the
per-layer `attn_hyper_connection`/`mlp_hyper_connection`, `:1611-1619` for the final
`hyper_connection_mixer = GatedResidual(hc_config, use_combine=False)`), BF16 default dtype, on GPU0 - with the real
checkpoint weights of layers 0, 3 and 47 (`model.language_model.layers.L.{attn,mlp}_hyper_connection.*`) and of
`model.language_model.hyper_connection_mixer.*`. Inputs: seeded BF16 hyper states [M, 4 x 2560] with a different scale
per stream (1, 0.5, 2, 4), M in {1, 7, 517}; combine's block output is a seeded BF16 [M, 2560].

Cases per site and M:
- `norm`: native per-branch Gemma RMSNorm (the first half of `cere_ktest_hc_mix`) vs `GatedResidual.hc_norm`
  (`GroupedGemmaRMSNorm` with group 2,560, `hyperconnection.py:21-75`; on CUDA the JIT `grouped_gemma_rmsnorm`);
- `mix`: vs `GatedResidual.mix` (`:233-289`) - the path the server takes: on GPU0 (RTX PRO 6000 Blackwell, sm_120,
  spec "Environment facts" / Q11) `_jit_mix_ok` is false (`:157-167` requires compute capability 10), so rows <= 16
  (`_FUSED_MIX_MAX_ROWS`, `hc_mix_triton.py:33`) run the Triton `fused_hc_mix` (`hc_mix_triton.py:206`, gate
  `fused_hc_mix_supported` `:185`) and larger M the torch.compile'd `_mix_compute` (`:195-211`); the report records
  which one ran (`path`);
- `mix_torchref`: vs the torch reference `GatedResidual._mix_compute` (`:195`, torch.compile'd at `:230`) on the same
  normed input, for every M (the second SGLang reference named by the spec);
- `combine` (per-layer sites only): vs `GatedResidual.combine` (`:291-338`; on CUDA the JIT `hc_combine_split` for
  M <= 32 / `hc_combine`, `sglang/kernels/ops/elementwise/hc_combine.py:116,49`), with the SGLang normed residual
  as input to both sides;
- `combine_delta`: the same outputs minus the residual (the injection alone, which the residual would otherwise mask);
- `combine_torchref`: vs the torch reference `GatedResidual._combine_compute` (`:213`, compiled at `:231`).
- `isolate_L<layer>_<site>_{norm,mix,combine,combine_delta}_m1` (tasks/m22d-spec.md D6/C7, phase `isolate`): the
  same four ops at one decode token (M = 1) on every layer 0-47 and both per-layer sites (real weights, the server's
  dispatch, recorded in `control`), so the D6 isolation of any first-exceeding layer needs no code change.
  Report-only (nothing is asserted beyond running): `c` = rel-L2(SGLang, FP64), `limit` = max(FLOOR_OP, 2c),
  `measured` = rel-L2(native, SGLang) on identical inputs, `within`, and the differing elements. The final mixer at
  M = 1 is the existing `final_mixer_m1_*` rows.
- `p11r21_<site>_m<m>_fused_{normed,mixed}` (tasks/m23-spec.md P11R-21): `cere_ktest_hc_mix_small_m` (C prototype:
  `cere_ktest_hc_mix_rows`' arguments with `path` in place of `fused_allowed`) with both GEMMs on the small-M K2
  kernel on the rows plans of 16 rows: path 1, hc_mix's default (the activation fused into the down projection's
  epilogue, the gate into the up projection's), must equal path 2 (`HcMixPath::kUnfused`: GEMM, K3 activation
  kernel, GEMM, K3 gate kernel) bit for bit, for every m in [1, 16] on layers 0, 3, 47 (both sites) and the final
  mixer.

FP64 calibration references (the math of `_mix_compute`/`_combine_compute`, reimplemented here because those are
closures inside `GatedResidual.__init__`): normed = per 2,560-wide branch x * rsqrt(mean(x^2) + eps) * (1 + w);
mix = mean over the 4 branches of sigmoid(silu(normed @ W_down^T / hc) @ W_up^T) * normed;
combine = residual + block_output (broadcast over branches) * 2 * sigmoid(normed @ W_inject^T / hc).

Native entry points (C prototypes; every call returns 0 on success, nonzero `cere_status` otherwise):

  int cere_ktest_hc_mix(const void *hyper,        /* bfloat16 [m, hc * hidden]: the 4-stream hyper state */
                        const void *norm_weight,  /* bfloat16 [hc * hidden] (hc_norm.weight, Gemma 1 + w) */
                        const void *w_down,       /* bfloat16 [lowrank, hc * hidden] (input_mix_weight_down) */
                        const void *w_up,         /* bfloat16 [hc * hidden, lowrank] (input_mix_weight_up) */
                        int64_t m, int64_t hc, int64_t hidden, int64_t lowrank,
                        float eps,
                        void *normed,             /* bfloat16 [m, hc * hidden]: per-branch RMSNorm(hyper) */
                        void *mixed,              /* bfloat16 [m, hidden]: the block input */
                        cudaStream_t stream);
      normed[r, g, :] = hyper[r, g, :] * rsqrt(mean(hyper[r, g, :]^2) + eps) * (1 + norm_weight[g, :]);
      t = silu(normed[r] @ w_down^T / hc); u = sigmoid(t @ w_up^T) viewed [hc, hidden];
      mixed[r, :] = mean_g(u[g, :] * normed[r, g, :]).

  int cere_ktest_hc_combine(const void *block_out,  /* bfloat16 [m, hidden] */
                            const void *hyper,      /* bfloat16 [m, hc * hidden]: the residual hyper state */
                            const void *normed,     /* bfloat16 [m, hc * hidden]: `normed` of the matching mix */
                            const void *w_inject,   /* bfloat16 [hc, hc * hidden] (block_inject_weight) */
                            int64_t m, int64_t hc, int64_t hidden,
                            void *out,              /* bfloat16 [m, hc * hidden]: the next hyper state */
                            cudaStream_t stream);
      out[r, g, :] = hyper[r, g, :] + block_out[r, :] * 2 * sigmoid((normed[r] . w_inject[g]) / hc).
"""
from __future__ import annotations

import contextlib
import unittest

import torch
import torch.nn.functional as F

import oracle
from sglang.srt.layers.hc_mix_triton import fused_hc_mix_supported
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.srt.runtime_context import get_context

ROW = "K3"
LAYERS = (0, 3, 47)
SITES = ("attn_hyper_connection", "mlp_hyper_connection")
MIXER_PREFIX = "model.language_model.hyper_connection_mixer."
# One decode token, a small odd batch inside the fused-mix range, a prefill-sized odd batch outside it.
M_CASES = (1, 7, 517)
# hc_combine's launch bound (k3_hc.cu kCombineSplitMaxM): m <= 16 the split grid, above it one block per row.
COMBINE_SPLIT_MAX_M = 16
# Per-stream input scales: the per-branch norm must normalize each 2,560-wide stream on its own.
STREAM_SCALES = (1.0, 0.5, 2.0, 4.0)
SEED_BASE = 3_000
ISOLATE_SEED = 40_000
# m23 design note P11R-21 (cere_ktest_hc_mix_small_m): the rows plans of a verify step (m2.9 VERIFY_ROWS, the small-M
# kernel's kMaxM), hc_mix's default path (the fused hc epilogues) and HcMixPath::kUnfused.
FUSED_ROWS = 16
FUSED_PATH = 1
UNFUSED_PATH = 2
FUSED_SEED = 60_000


@contextlib.contextmanager
def _sglang_build():
    """Module construction as the model loader does it: BF16 default dtype, parameters on GPU0."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(oracle.DEVICE):
            yield
    finally:
        torch.set_default_dtype(previous)


def _bf16(name: str) -> torch.Tensor:
    t = oracle.real_tensor(name)
    if t.dtype != torch.bfloat16:
        raise oracle.OracleError(f"{name} is {t.dtype}, expected torch.bfloat16")
    return t.contiguous()


def _hc_config() -> HyperConnectionConfig:
    cfg = oracle.text_config()
    return HyperConnectionConfig(
        hc_count=int(cfg["hc_count"]),
        hidden_size=int(cfg["hidden_size"]),
        params_dtype=torch.bfloat16,
        hc_lowrank=int(cfg["hc_lowrank"]),
        rms_norm_eps=float(cfg["rms_norm_eps"]),
        hc_per_branch_norm=True,
    )


def _gated_residual(prefix: str, use_combine: bool) -> GatedResidual:
    """A GatedResidual as `qwen4_exp.py` builds it, with the checkpoint tensors under `prefix` copied in."""
    with _sglang_build():
        module = GatedResidual(_hc_config(), use_mix=True, use_combine=use_combine)
    params = dict(module.named_parameters())
    expected = {"hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight"}
    if use_combine:
        expected.add("block_inject_weight.weight")
    if set(params) != expected:
        raise oracle.OracleError(f"GatedResidual parameters {sorted(params)} != {sorted(expected)}")
    with torch.no_grad():
        for name, param in params.items():
            loaded = _bf16(prefix + name)
            if tuple(loaded.shape) != tuple(param.shape) or param.dtype != torch.bfloat16:
                raise oracle.OracleError(f"{prefix}{name}: checkpoint {tuple(loaded.shape)} vs module "
                                         f"{tuple(param.shape)} {param.dtype}")
            param.data.copy_(loaded)
    return module


def _hyper_input(m: int, hc: int, hidden: int, seed: int) -> torch.Tensor:
    g = oracle.generator(seed)
    x = torch.randn((m, hc, hidden), generator=g, device=oracle.DEVICE, dtype=torch.float32)
    scales = torch.tensor(STREAM_SCALES, dtype=torch.float32, device=oracle.DEVICE).view(1, hc, 1)
    return (x * scales).reshape(m, hc * hidden).to(torch.bfloat16).contiguous()


def _block_output(m: int, hidden: int, seed: int) -> torch.Tensor:
    g = oracle.generator(seed)
    return torch.randn((m, hidden), generator=g, device=oracle.DEVICE, dtype=torch.float32).to(torch.bfloat16)


# -- FP64 references (calibration) ------------------------------------------------------------------------------


def _norm_fp64(x: torch.Tensor, weight: torch.Tensor, hc: int, hidden: int, eps: float) -> torch.Tensor:
    x64 = x.to(torch.float64).reshape(x.shape[0], hc, hidden)
    rstd = torch.rsqrt(x64.pow(2).mean(dim=-1, keepdim=True) + eps)
    w64 = weight.to(torch.float64).reshape(1, hc, hidden)
    return (x64 * rstd * (1.0 + w64)).reshape(x.shape[0], hc * hidden)


def _mix_fp64(normed64: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor, hc: int, hidden: int):
    t = F.silu(normed64 @ w_down.to(torch.float64).T / hc)
    u = torch.sigmoid(t @ w_up.to(torch.float64).T).reshape(-1, hc, hidden)
    return (u * normed64.reshape(-1, hc, hidden)).mean(dim=-2)


def _combine_fp64(block_out: torch.Tensor, hyper: torch.Tensor, normed: torch.Tensor, w_inject: torch.Tensor,
                  hc: int, hidden: int) -> torch.Tensor:
    gate = 2.0 * torch.sigmoid(normed.to(torch.float64) @ w_inject.to(torch.float64).T / hc)
    injection = block_out.to(torch.float64).unsqueeze(-2) * gate.unsqueeze(-1)
    return (hyper.to(torch.float64).reshape(-1, hc, hidden) + injection).reshape(-1, hc * hidden)


# -- native entry points ----------------------------------------------------------------------------------------


def _native_mix(hyper: torch.Tensor, module: GatedResidual, hc: int, hidden: int, lowrank: int, eps: float):
    m = hyper.shape[0]
    normed = oracle.empty((m, hc * hidden), torch.bfloat16)
    mixed = oracle.empty((m, hidden), torch.bfloat16)
    oracle.call("cere_ktest_hc_mix", hyper, module.hc_norm.weight.detach(),
                module.input_mix_weight_down.weight.detach(), module.input_mix_weight_up.weight.detach(),
                m, hc, hidden, lowrank, float(eps), normed, mixed)
    return normed, mixed


def _native_combine(block_out: torch.Tensor, hyper: torch.Tensor, normed: torch.Tensor, module: GatedResidual,
                    hc: int, hidden: int) -> torch.Tensor:
    m = hyper.shape[0]
    out = oracle.empty((m, hc * hidden), torch.bfloat16)
    oracle.call("cere_ktest_hc_combine", block_out, hyper, normed, module.block_inject_weight.weight.detach(),
                m, hc, hidden, out)
    return out


def _native_mix_small_m(hyper: torch.Tensor, module: GatedResidual, hc: int, hidden: int, lowrank: int, eps: float,
                        path: int):
    """cere_ktest_hc_mix_small_m on the rows plans of FUSED_ROWS rows (every m fused): path FUSED_PATH or
    UNFUSED_PATH."""
    m = hyper.shape[0]
    normed = oracle.empty((m, hc * hidden), torch.bfloat16)
    mixed = oracle.empty((m, hidden), torch.bfloat16)
    oracle.call("cere_ktest_hc_mix_small_m", hyper, module.hc_norm.weight.detach(),
                module.input_mix_weight_down.weight.detach(), module.input_mix_weight_up.weight.detach(),
                m, hc, hidden, lowrank, float(eps), normed, mixed, FUSED_ROWS, path)
    return normed, mixed


def _isolation_row(layer: int, site: str, op: str, native: torch.Tensor, control: torch.Tensor,
                   high_precision: torch.Tensor, control_kernel: str) -> None:
    """One report-only D6 isolation row: `c` = rel-L2(control, FP64), `limit` = max(FLOOR_OP, 2c), `measured` =
    rel-L2(native, control) and the differing elements on identical inputs; `within` = measured <= limit. Asserts
    nothing (tasks/m22d-spec.md C7)."""
    c = oracle.rel_l2(control, high_precision)
    limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
    measured = oracle.rel_l2(native, control)
    differing = int((native != control).sum().item())
    oracle.report_row(ROW, f"isolate_L{layer}_{site}_{op}_m1", phase="isolate", layer=layer, op=f"{site}.{op}", m=1,
                      control=control_kernel, c=c, limit=limit, measured=measured, within=measured <= limit,
                      elements=native.numel(), differing=differing)


class K3HyperConnections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # SGLang modules read the published config at construction/dispatch; publish a dummy-boundary one
        # (`runtime_context.py:968`, the sanctioned test entry).
        cls.sglang_config = get_context().override_server_args()
        cls.sglang_config.install()
        cfg = oracle.text_config()
        cls.hc = int(cfg["hc_count"])
        cls.hidden = int(cfg["hidden_size"])
        cls.lowrank = int(cfg["hc_lowrank"])
        cls.eps = float(cfg["rms_norm_eps"])
        expected = {"hc_count": 4, "hidden_size": 2560, "hc_lowrank": 320, "rms_norm_eps": 1e-6}
        actual = {k: cfg[k] for k in expected}
        if actual != expected:
            raise oracle.OracleError(f"text_config HC fields {actual} != {expected}")

    @classmethod
    def tearDownClass(cls):
        cls.sglang_config.restore()

    # -- one site -----------------------------------------------------------------------------------------------

    def _check_mix(self, module: GatedResidual, site: str, m: int, seed: int):
        """Norm, mix (server dispatch) and mix (torch reference) cases; returns (hyper, SGLang normed)."""
        hc, hidden, lowrank, eps = self.hc, self.hidden, self.lowrank, self.eps
        hyper = _hyper_input(m, hc, hidden, seed)
        w_down = module.input_mix_weight_down.weight.detach()
        w_up = module.input_mix_weight_up.weight.detach()
        with torch.no_grad():
            sg_mixed, (_, sg_normed) = module.mix(hyper)
            sg_norm_direct = module.hc_norm(hyper)
            sg_torchref = module._mix_compute(sg_normed, w_down, w_up, hc, hidden).to(torch.bfloat16)
        # The branch GatedResidual.mix took (`hyperconnection.py:247-288`), recorded so the report states it.
        if module._jit_mix_ok and m <= 24:
            path = "cute_hc_mix"
        elif fused_hc_mix_supported(sg_normed, w_down, w_up):
            path = "triton_fused_hc_mix"
        else:
            path = "compiled_mix_compute"
        oracle.report_row(ROW, f"{site}_m{m}", phase="dispatch", path=path)
        # GatedResidual.mix normalizes with the same hc_norm module; both reads must agree bit for bit.
        oracle.check_bit_exact(self, ROW, f"{site}_m{m}_norm_consistent", sg_norm_direct, sg_normed)

        norm64 = _norm_fp64(hyper, module.hc_norm.weight.detach(), hc, hidden, eps)
        mix64 = _mix_fp64(norm64, w_down, w_up, hc, hidden)

        case = f"{site}_m{m}"
        c_norm = oracle.calibrate(ROW, f"{case}_norm", sg_normed, norm64, oracle.FLOOR_OP)
        limit_norm = oracle.calibrated_limit(c_norm, oracle.FLOOR_OP)
        c_mix = oracle.calibrate(ROW, f"{case}_mix", sg_mixed, mix64, oracle.FLOOR_OP)
        limit_mix = oracle.calibrated_limit(c_mix, oracle.FLOOR_OP)
        c_ref = oracle.calibrate(ROW, f"{case}_mix_torchref", sg_torchref, mix64, oracle.FLOOR_OP)
        limit_ref = oracle.calibrated_limit(c_ref, oracle.FLOOR_OP)

        nat_normed, nat_mixed = _native_mix(hyper, module, hc, hidden, lowrank, eps)
        oracle.check_within(self, ROW, f"{case}_norm", oracle.rel_l2(nat_normed, sg_normed), c=c_norm,
                            limit=limit_norm)
        oracle.check_within(self, ROW, f"{case}_mix", oracle.rel_l2(nat_mixed, sg_mixed), c=c_mix, limit=limit_mix)
        oracle.check_within(self, ROW, f"{case}_mix_torchref", oracle.rel_l2(nat_mixed, sg_torchref), c=c_ref,
                            limit=limit_ref)
        again_normed, again_mixed = _native_mix(hyper, module, hc, hidden, lowrank, eps)
        oracle.check_bit_exact(self, ROW, f"{case}_norm_repeat", again_normed, nat_normed)
        oracle.check_bit_exact(self, ROW, f"{case}_mix_repeat", again_mixed, nat_mixed)
        return hyper, sg_normed

    def _check_combine(self, module: GatedResidual, site: str, m: int, hyper: torch.Tensor,
                       sg_normed: torch.Tensor, seed: int):
        hc, hidden = self.hc, self.hidden
        block_out = _block_output(m, hidden, seed)
        w_inject = module.block_inject_weight.weight.detach()
        with torch.no_grad():
            sg_out = module.combine(block_out, (hyper, sg_normed))
            sg_torchref = module._combine_compute(block_out, hyper, sg_normed, w_inject, hc, hidden).to(
                torch.bfloat16)
        if sg_out.dtype != torch.bfloat16 or tuple(sg_out.shape) != (m, hc * hidden):
            raise oracle.OracleError(f"GatedResidual.combine returned {sg_out.dtype} {tuple(sg_out.shape)}")
        ref64 = _combine_fp64(block_out, hyper, sg_normed, w_inject, hc, hidden)
        hyper64 = hyper.to(torch.float64)

        case = f"{site}_m{m}"
        c_comb = oracle.calibrate(ROW, f"{case}_combine", sg_out, ref64, oracle.FLOOR_OP)
        limit_comb = oracle.calibrated_limit(c_comb, oracle.FLOOR_OP)
        c_delta = oracle.calibrate(ROW, f"{case}_combine_delta", sg_out.to(torch.float64) - hyper64,
                                   ref64 - hyper64, oracle.FLOOR_OP)
        limit_delta = oracle.calibrated_limit(c_delta, oracle.FLOOR_OP)
        c_ref = oracle.calibrate(ROW, f"{case}_combine_torchref", sg_torchref, ref64, oracle.FLOOR_OP)
        limit_ref = oracle.calibrated_limit(c_ref, oracle.FLOOR_OP)

        native = _native_combine(block_out, hyper, sg_normed, module, hc, hidden)
        oracle.check_within(self, ROW, f"{case}_combine", oracle.rel_l2(native, sg_out), c=c_comb,
                            limit=limit_comb)
        oracle.check_within(self, ROW, f"{case}_combine_delta",
                            oracle.rel_l2(native.to(torch.float64) - hyper64, sg_out.to(torch.float64) - hyper64),
                            c=c_delta, limit=limit_delta)
        oracle.check_within(self, ROW, f"{case}_combine_torchref", oracle.rel_l2(native, sg_torchref), c=c_ref,
                            limit=limit_ref)
        oracle.check_bit_exact(self, ROW, f"{case}_combine_repeat",
                               _native_combine(block_out, hyper, sg_normed, module, hc, hidden), native)
        if m > COMBINE_SPLIT_MAX_M:
            # m23 design note P11: m <= 16 runs the split combine grid, larger m the one-block-per-row kernel; the
            # first 16 rows of this call must equal the same rows combined alone (split grid), bit for bit.
            head = slice(0, COMBINE_SPLIT_MAX_M)
            split = _native_combine(block_out[head].contiguous(), hyper[head].contiguous(),
                                    sg_normed[head].contiguous(), module, hc, hidden)
            oracle.check_bit_exact(self, ROW, f"{case}_combine_split_equals_row", split, native[head])

    def _check_layer(self, layer: int):
        for site_index, site in enumerate(SITES):
            module = _gated_residual(f"{oracle.layer_prefix(layer)}{site}.", use_combine=True)
            for m_index, m in enumerate(M_CASES):
                seed = SEED_BASE + 100 * layer + 10 * site_index + m_index
                name = f"L{layer}_{site}"
                with self.subTest(layer=layer, site=site, m=m):
                    hyper, sg_normed = self._check_mix(module, name, m, seed)
                    self._check_combine(module, name, m, hyper, sg_normed, seed + 5_000)

    # -- tests --------------------------------------------------------------------------------------------------

    def test_layer_0(self):
        self._check_layer(0)

    def test_layer_3(self):
        self._check_layer(3)

    def test_layer_47(self):
        self._check_layer(47)

    def test_final_mixer(self):
        """`hyper_connection_mixer` (use_combine=False): the mix that replaces the final norm (qwen4_exp.py:1668)."""
        module = _gated_residual(MIXER_PREFIX, use_combine=False)
        self.assertFalse(hasattr(module, "block_inject_weight"), "the final mixer must be mix-only")
        for m_index, m in enumerate(M_CASES):
            with self.subTest(m=m):
                self._check_mix(module, "final_mixer", m, SEED_BASE + 9_000 + m_index)

    # -- m23 design note P11R-21: the fused hc epilogues of the small-M K2 kernel --------------------------------

    def test_p11r21_fused_mix_bit_exact(self):
        """hc_mix with both GEMMs on the small-M kernel (the k2_gemm lever's small_m): the default path (activation
        and gate fused into the GEMM epilogues) equals the unfused path (GEMM, K3 activation, GEMM, K3 gate) bit for
        bit, normed and mixed, for every m in [1, 16] on the rows plans of a verify step (rows 16), on layers 0, 3
        and 47 (both sites) and the final mixer."""
        hc, hidden, lowrank, eps = self.hc, self.hidden, self.lowrank, self.eps
        modules = [(f"L{layer}_{site}", f"{oracle.layer_prefix(layer)}{site}.", True)
                   for layer in LAYERS for site in SITES]
        modules.append(("final_mixer", MIXER_PREFIX, False))
        for module_index, (name, prefix, use_combine) in enumerate(modules):
            module = _gated_residual(prefix, use_combine=use_combine)
            for m in range(1, FUSED_ROWS + 1):
                with self.subTest(module=name, m=m):
                    hyper = _hyper_input(m, hc, hidden, FUSED_SEED + 100 * module_index + m)
                    fused = _native_mix_small_m(hyper, module, hc, hidden, lowrank, eps, FUSED_PATH)
                    unfused = _native_mix_small_m(hyper, module, hc, hidden, lowrank, eps, UNFUSED_PATH)
                    case = f"p11r21_{name}_m{m}"
                    oracle.check_bit_exact(self, ROW, f"{case}_fused_normed", fused[0], unfused[0])
                    oracle.check_bit_exact(self, ROW, f"{case}_fused_mixed", fused[1], unfused[1])

    # -- D6 isolation rows (m22d design note C7): decode M = 1 on every layer, report-only --------------------

    def _isolate_site(self, module: GatedResidual, layer: int, site: str, seed: int) -> None:
        hc, hidden, lowrank, eps = self.hc, self.hidden, self.lowrank, self.eps
        hyper = _hyper_input(1, hc, hidden, seed)
        w_down = module.input_mix_weight_down.weight.detach()
        w_up = module.input_mix_weight_up.weight.detach()
        with torch.no_grad():
            sg_mixed, (_, sg_normed) = module.mix(hyper)
        path = "triton_fused_hc_mix" if fused_hc_mix_supported(sg_normed, w_down, w_up) else "compiled_mix_compute"
        if module._jit_mix_ok:
            path = "cute_hc_mix"
        norm64 = _norm_fp64(hyper, module.hc_norm.weight.detach(), hc, hidden, eps)
        mix64 = _mix_fp64(norm64, w_down, w_up, hc, hidden)
        nat_normed, nat_mixed = _native_mix(hyper, module, hc, hidden, lowrank, eps)
        _isolation_row(layer, site, "norm", nat_normed, sg_normed, norm64, path)
        _isolation_row(layer, site, "mix", nat_mixed, sg_mixed, mix64, path)
        # combine on the same block output, residual and SGLang normed state for both sides.
        block_out = _block_output(1, hidden, seed + 5_000)
        w_inject = module.block_inject_weight.weight.detach()
        with torch.no_grad():
            sg_out = module.combine(block_out, (hyper, sg_normed))
        ref64 = _combine_fp64(block_out, hyper, sg_normed, w_inject, hc, hidden)
        native = _native_combine(block_out, hyper, sg_normed, module, hc, hidden)
        _isolation_row(layer, site, "combine", native, sg_out, ref64, "GatedResidual.combine")
        hyper64 = hyper.to(torch.float64)
        _isolation_row(layer, site, "combine_delta", native.to(torch.float64) - hyper64,
                       sg_out.to(torch.float64) - hyper64, ref64 - hyper64, "GatedResidual.combine")

    def test_isolate_decode_all_layers(self):
        """Norm, mix and combine of both per-layer sites at M = 1 on all 48 layers (any first-exceeding layer)."""
        for layer in range(int(oracle.text_config()["num_hidden_layers"])):
            for site_index, site in enumerate(SITES):
                with self.subTest(layer=layer, site=site):
                    module = _gated_residual(f"{oracle.layer_prefix(layer)}{site}.", use_combine=True)
                    self._isolate_site(module, layer, site, ISOLATE_SEED + 10 * layer + site_index)


if __name__ == "__main__":
    unittest.main()
