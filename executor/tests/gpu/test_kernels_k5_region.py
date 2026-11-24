"""K5R (tasks/selftune-spec.md phase STP stage a, D11, D12): the GDN decode-mixer region kernel against today's ops.

Container only (pinned SGLang image, `/out/libcere_ktest.so`, `/model` read-only); suite `kernels_k5`.

The region (`executor/cuda/src/ktest/k5r_region.{h,cu}`, libcere_ktest.so only) runs one GDN layer's decode mixer
(the four in-projections, conv, delta rule, gated RMSNorm, out_proj) as one persistent launch over a fixed instruction
list, prefetching out_proj (and the state) into L2 while the in-projections stream. The reference is today's sequence:
rows 1 `k5::layer_forward` DECODE (the single-session decode step); rows 2..4 the verify composition of model forward
(`kernels::gemm_segments`, `k5::chain_core`, the out_proj plan), both with small-M GEMM plans (the shipped
`k2_gemm = "small_m"` lever).

Cases, on each layer of LAYERS (first and last `linear_attention` layer) and each row count of ROWS: the state a
native PREFILL of PREFILL tokens leaves, then STEPS consecutive calls of `rows` rows, reference and region each
carrying their own state. Bit-exact (`oracle.check_bit_exact`): every step's out, the saves (state after each row
but the last), the final conv and SSM state. Fail-first, every run: the region's mutant variant (every GEMM's K
slices, in-projections and out_proj, summed in reverse order) must differ from the reference on out over STEPS carried
steps (`mutant_*` cases).

Speed (reported, not asserted; layer 0, rows 1 and 4): cold-L2 CUDA-graph replays (`ktest::graph_spans`, an
l2_flush of 512 MB before each timed replay), PAIRS alternating (reference, region) pairs of WARMUP + TIMED replays;
medians in microseconds over the PAIRS x TIMED replays of each side, and achieved GB/s of the bytes the call must move
(weights read once, the state read and written, the saves written).

Inputs are real activations: real embedding rows of seeded random token ids, RMS-normalised; token ids never leave the
process; weights are read in place from the checkpoint.

Native entry points (every buffer a contiguous row-major device pointer, the stream last, 0 or a nonzero
`cere_status`):

    int cere_ktest_k5r_region(const void *hidden, const void *w_qkv, const void *w_z, const void *w_b,
                              const void *w_a, const void *conv_weight, const void *A_log, const void *dt_bias,
                              const void *norm_weight, const void *w_out, void *conv_state, void *ssm_state,
                              void *conv_saves, void *ssm_saves, void *out, int64_t rows, int64_t variant,
                              int64_t warmup, int64_t timed, int64_t cold, void *us, cudaStream_t stream);
    int cere_ktest_k5r_reference(... the same without variant ...);
        hidden bf16 [rows, 2560]; weights as cere_ktest_gdn_layer_decode (test_kernels_k5_gdn.py); conv_state bf16
        [10240, 3] and ssm_state bf16 [48, 128, 128] in/out; conv_saves bf16 [rows - 1, 10240, 3] and ssm_saves bf16
        [rows - 1, 48, 128, 128] (null at rows 1): the state after row t; out bf16 [rows, 2560]; rows 1..4; variant
        0 exact, 1 mutant. timed 0: one call. timed > 0: warmup, then `timed` replays of a captured call (after an
        l2_flush each when cold != 0), each replay's span in microseconds into us (host F32 [timed]).
"""
from __future__ import annotations

import functools
import statistics
import types
import unittest

import torch

import oracle

torch.set_grad_enabled(False)

ROW = "K5R"
LAYERS = (0, 46)  # first and last linear_attention layer (m22d design note D5)
ROWS = (1, 2, 3, 4)  # decode (1) and the verify chain lengths (k5::kChainMaxRows = 4)
SPEED_ROWS = (1, 4)
HIDDEN = 2560
HK = 16
HV = 48
DK = 128
DV = 128
CONV_K = 4
KEY_DIM = HK * DK
VALUE_DIM = HV * DV
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
RMS_EPS = 1e-6
PREFILL = 27  # the native PREFILL that sets the starting state
STEPS = 8  # consecutive calls per bit-exact case
WARMUP = 10  # graph replays before each timed series (pre-stated)
TIMED = 50  # cold replays per series (pre-stated)
PAIRS = 3  # alternating (reference, region) series per row count (pre-stated)
BF16 = torch.bfloat16
EMBED = "model.language_model.embed_tokens.weight"


@functools.cache
def _checked_config() -> None:
    cfg = oracle.text_config()
    expected = {"hidden_size": HIDDEN, "linear_num_key_heads": HK, "linear_num_value_heads": HV,
                "linear_key_head_dim": DK, "linear_value_head_dim": DV, "linear_conv_kernel_dim": CONV_K,
                "rms_norm_eps": RMS_EPS}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise oracle.OracleError(f"text_config {key}={cfg.get(key)!r}, K5R is written for {value!r}")


def _expect(name: str, t: torch.Tensor, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    if tuple(t.shape) != shape or t.dtype != dtype:
        raise oracle.OracleError(f"{name}: {t.dtype} {tuple(t.shape)}, expected {dtype} {shape}")
    return t.contiguous()


@functools.lru_cache(maxsize=len(LAYERS))
def _weights(layer: int) -> types.SimpleNamespace:
    """One GDN layer's `linear_attn` weights on the device, as test_kernels_k5_gdn.py holds them."""
    _checked_config()
    raw = oracle.gdn_layer_weights(layer)
    p = oracle.layer_prefix(layer) + "linear_attn."

    def get(name, shape):
        return _expect(p + name, raw[name], shape, BF16)

    w = types.SimpleNamespace(layer=layer)
    w.qkv = get("in_proj_qkv.weight", (CONV_DIM, HIDDEN))
    w.z = get("in_proj_z.weight", (VALUE_DIM, HIDDEN))
    w.b = get("in_proj_b.weight", (HV, HIDDEN))
    w.a = get("in_proj_a.weight", (HV, HIDDEN))
    w.conv = get("conv1d.weight", (CONV_DIM, 1, CONV_K)).reshape(CONV_DIM, CONV_K).contiguous()
    w.A_log = get("A_log", (HV,)).float().contiguous()
    w.dt_bias = get("dt_bias", (HV,))
    w.norm = get("norm.weight", (DV,))
    w.out = get("out_proj.weight", (HIDDEN, VALUE_DIM))
    return w


def _bytes(w: types.SimpleNamespace, rows: int) -> int:
    """The bytes one call must move: every weight read once, the conv and SSM state read and written, the saves
    written, hidden read and out written."""
    weights = sum(t.numel() * t.element_size() for t in (w.qkv, w.z, w.b, w.a, w.conv, w.A_log, w.dt_bias, w.norm,
                                                         w.out))
    state = (CONV_DIM * (CONV_K - 1) + HV * DV * DK) * 2
    return weights + 2 * state + (rows - 1) * state + 2 * rows * HIDDEN * 2


@functools.cache
def _embedding_cpu() -> torch.Tensor:
    return oracle.real_tensor(EMBED, device=None)


def _hidden(seed: int, tokens: int) -> torch.Tensor:
    """bf16 [tokens, HIDDEN]: real embedding rows of seeded random token ids, RMS-normalised."""
    table = _embedding_cpu()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    ids = torch.randint(0, table.shape[0], (tokens,), generator=gen)
    x = table[ids].to(oracle.DEVICE, torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return x.to(BF16).contiguous()


def _state(w: types.SimpleNamespace, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The conv and SSM state a native PREFILL of PREFILL real rows leaves (`cere_ktest_gdn_layer_prefill`)."""
    conv = torch.zeros((CONV_DIM, CONV_K - 1), dtype=BF16, device=oracle.DEVICE)
    ssm = torch.zeros((HV, DV, DK), dtype=BF16, device=oracle.DEVICE)
    out = oracle.empty((PREFILL, HIDDEN), BF16)
    oracle.call("cere_ktest_gdn_layer_prefill", _hidden(seed, PREFILL), w.qkv, w.z, w.b, w.a, w.conv, w.A_log,
                w.dt_bias, w.norm, w.out, conv, ssm, out, PREFILL)
    return conv, ssm


class Side:
    """One side's state and buffers (reference or region) at `rows` rows."""

    def __init__(self, conv: torch.Tensor, ssm: torch.Tensor, rows: int):
        self.conv = conv.clone()
        self.ssm = ssm.clone()
        self.rows = rows
        self.conv_saves = oracle.empty((rows - 1, CONV_DIM, CONV_K - 1), BF16) if rows > 1 else None
        self.ssm_saves = oracle.empty((rows - 1, HV, DV, DK), BF16) if rows > 1 else None
        self.out = oracle.empty((rows, HIDDEN), BF16)


def _reference(w, hidden, s: Side, timed: int = 0, cold: bool = False, us: torch.Tensor | None = None) -> None:
    oracle.call("cere_ktest_k5r_reference", hidden, w.qkv, w.z, w.b, w.a, w.conv, w.A_log, w.dt_bias, w.norm,
                w.out, s.conv, s.ssm, s.conv_saves, s.ssm_saves, s.out, s.rows, WARMUP if timed else 0, timed,
                int(cold), us)


def _region(w, hidden, s: Side, variant: int = 0, timed: int = 0, cold: bool = False,
            us: torch.Tensor | None = None) -> None:
    oracle.call("cere_ktest_k5r_region", hidden, w.qkv, w.z, w.b, w.a, w.conv, w.A_log, w.dt_bias, w.norm, w.out,
                s.conv, s.ssm, s.conv_saves, s.ssm_saves, s.out, s.rows, variant, WARMUP if timed else 0, timed,
                int(cold), us)


class RegionK5(unittest.TestCase):
    def test_bit_exact(self):
        """Every step's out and saves and the final state of the region equal the reference's bit for bit."""
        for layer in LAYERS:
            w = _weights(layer)
            conv, ssm = _state(w, 7100 + layer)
            for rows in ROWS:
                with self.subTest(layer=layer, rows=rows):
                    ref, reg = Side(conv, ssm, rows), Side(conv, ssm, rows)
                    outs = ([], [])
                    saves = ([], [])
                    for step in range(STEPS):
                        hidden = _hidden(7200 + 100 * rows + step, rows)
                        _reference(w, hidden, ref)
                        _region(w, hidden, reg)
                        for i, s in enumerate((ref, reg)):
                            outs[i].append(s.out.clone())
                            if rows > 1:
                                saves[i].append(torch.cat([s.conv_saves.flatten(), s.ssm_saves.flatten()]))
                    case = f"layer{layer}/rows{rows}"
                    oracle.check_bit_exact(self, ROW, f"{case}/out", torch.stack(outs[1]), torch.stack(outs[0]))
                    oracle.check_bit_exact(self, ROW, f"{case}/conv_state", reg.conv, ref.conv)
                    oracle.check_bit_exact(self, ROW, f"{case}/ssm_state", reg.ssm, ref.ssm)
                    if rows > 1:
                        oracle.check_bit_exact(self, ROW, f"{case}/saves", torch.stack(saves[1]),
                                               torch.stack(saves[0]))

    def test_mutant_differs(self):
        """Fail-first: the mutant region (every GEMM's slices summed in reverse order) breaks the bit-exact out."""
        w = _weights(LAYERS[0])
        conv, ssm = _state(w, 7100 + LAYERS[0])
        for rows in SPEED_ROWS:
            with self.subTest(rows=rows):
                ref, reg = Side(conv, ssm, rows), Side(conv, ssm, rows)
                outs = ([], [])
                for step in range(STEPS):
                    hidden = _hidden(7300 + 100 * rows + step, rows)
                    _reference(w, hidden, ref)
                    _region(w, hidden, reg, variant=1)
                    outs[0].append(ref.out.clone())
                    outs[1].append(reg.out.clone())
                a = torch.stack(outs[1]).view(torch.int16)
                b = torch.stack(outs[0]).view(torch.int16)
                differing = int((a != b).sum().item())
                oracle.report_row(ROW, f"layer{LAYERS[0]}/rows{rows}/mutant_out", phase="fail_first",
                                  elements=b.numel(), differing=differing, passed=differing > 0)
                self.assertTrue(differing > 0, f"mutant out equals the reference at rows {rows}")

    def test_speed(self):
        """Reported: cold-L2 graph-replay medians of the reference sequence and the region, us and GB/s."""
        w = _weights(LAYERS[0])
        conv, ssm = _state(w, 7100 + LAYERS[0])
        for rows in SPEED_ROWS:
            with self.subTest(rows=rows):
                ref, reg = Side(conv, ssm, rows), Side(conv, ssm, rows)
                hidden = _hidden(7400 + rows, rows)
                spans = ([], [])
                for _ in range(PAIRS):
                    for i, run in enumerate((_reference, _region)):
                        us = torch.zeros(TIMED, dtype=torch.float32)
                        run(w, hidden, (ref, reg)[i], timed=TIMED, cold=True, us=us)
                        spans[i].extend(us.tolist())
                moved = _bytes(w, rows)
                ref_us = statistics.median(spans[0])
                reg_us = statistics.median(spans[1])
                oracle.report_row(ROW, f"layer{LAYERS[0]}/rows{rows}/speed_cold", phase="speed", rows=rows,
                                  warmup=WARMUP, timed=TIMED, pairs=PAIRS, bytes=moved,
                                  reference_us=round(ref_us, 2), region_us=round(reg_us, 2),
                                  reference_gbs=round(moved / ref_us / 1e3, 1),
                                  region_gbs=round(moved / reg_us / 1e3, 1),
                                  reference_min_us=round(min(spans[0]), 2), region_min_us=round(min(spans[1]), 2),
                                  reference_max_us=round(max(spans[0]), 2), region_max_us=round(max(spans[1]), 2),
                                  speedup=round(ref_us / reg_us, 4))
                self.assertEqual(len(spans[0]), PAIRS * TIMED)
                self.assertEqual(len(spans[1]), PAIRS * TIMED)


if __name__ == "__main__":
    unittest.main()
