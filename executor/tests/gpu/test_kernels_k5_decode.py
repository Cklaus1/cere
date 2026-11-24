"""K5 decode (tasks/m22n-spec.md C5, D5): the native single-token GDN delta rule against the FlashInfer kernel the
no-speculation control dispatches.

Container only (pinned SGLang image, GPU0, `/out/libcere_ktest.so`, `/model` read-only); suites `kernels`, `kernels_k5`.

Reference: `gated_delta_rule_decode_pretranspose` called in pool mode with B=1 exactly as SGLang's
`FlashInferGDNKernel.decode` (`gdn_flashinfer.py:239-251`) calls it: q, k bf16 [1, 1, 16, 128], v bf16 [1, 1, 48, 128],
a, b bf16 [1, 1, 48], `A_log.float()`, bf16 `dt_bias`, `state=None`, `use_qk_l2norm=True`, no scale (128^-0.5), the
BF16 state pool [1, 48, 128 (v), 128 (k)] with slot index 0, updated in place. At T=1, B*HV=48 this is FlashInfer's
BF16-state MTP-ILP4 kernel (`gdn_decode_bf16_state.py`).

Cases, 64 sequential decode steps each, native and reference each carrying their own state, on each GDN layer of
LAYERS (tasks/m22d-spec.md D5: first, second, middle and last `linear_attention` layer, each with its own weights; an
id whose `layer_types` entry is not `linear_attention` is refused by `oracle.gdn_layer_weights`), case names prefixed
`layer<id>/`:
- `delta_rule_decode_fi_random_state`: from a seeded random BF16 state scaled to the RMS of a real prefill state;
- `delta_rule_decode_fi_prefill_state`: from the state FLA `chunk_gated_delta_rule` (SGLang's extend path, gating by
  `fused_gdn_gating`) leaves after a fresh 27-token prefill.
Per case the 64 step outputs (stacked) and the final state are bit-exact (`oracle.check_bit_exact`), else within 1 BF16
ulp on at most 1e-3 of the elements (`oracle.check_ulp`); the bit-equal fractions are a report row. The layer-0
limits apply unchanged to every layer.

Inputs are real activations of the layer: q, k, v from SGLang's prefill conv (Triton `causal_conv1d_fn`, silu) over
the layer's in_proj_qkv projection, a and b from its in_proj_ba, of real embedding rows of seeded random token ids,
RMS-normalised (the pre-mixer norm without its weight; the same seeds on every layer); token ids never leave the
process.

Native entry point (every buffer a contiguous row-major device pointer, `float` a scalar, the stream last, the return
value 0 or a nonzero `cere_status`):

    int cere_ktest_gdn_decode_step(const void *q, const void *k, const void *v, const void *a, const void *b,
                                   const void *A_log, const void *dt_bias, void *state, void *o, float scale,
                                   cudaStream_t stream);
        One token: q, k bf16 [16, 128] (raw; L2-normalised in-kernel, eps 1e-6); v bf16 [48, 128]; a, b bf16 [48];
        A_log f32 [48]; dt_bias bf16 [48]; state bf16 [48, 128 (v), 128 (k)] in/out (SGLang's K-last pool slot);
        o bf16 [48, 128]. Value head h uses key head h / 3. g = -exp(A_log) * softplus(a + dt_bias) (softplus(x) = x
        for x > 20), beta = sigmoid(b); with S = state[h] as [v, k]: S *= exp(g); u = beta * (v - S k);
        S += u k^T; o = S (scale * q). The FlashInfer ILP4 arithmetic (tasks/m22n-spec.md D5).
"""
from __future__ import annotations

import functools
import os
import types
import unittest

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")  # as SGLang's gdn_flashinfer.py does before import

import torch
import torch.nn.functional as F
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.triton_gdn_fused_proj import fused_qkv_split_gdn_prefill
from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn

import oracle

torch.set_grad_enabled(False)

ROW = "K5"
LAYERS = (0, 1, 18, 46)  # D5 (m22d design note): first, second, middle and last linear_attention layer
HIDDEN = 2560
HK = 16  # key (and query) heads
HV = 48  # value heads
DK = 128
DV = 128
CONV_K = 4
KEY_DIM = HK * DK
VALUE_DIM = HV * DV
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
RMS_EPS = 1e-6  # text_config rms_norm_eps
SCALE = DK ** -0.5  # SGLang passes no scale: FlashInfer uses K ** -0.5
PREFILL = 27  # the prefill length of case (ii) (C5)
STEPS = 64  # sequential decode steps per case
BF16 = torch.bfloat16
EMBED = "model.language_model.embed_tokens.weight"


# -- model facts and weights (the subset of test_kernels_k5_gdn.py this suite needs) ------------------------------


@functools.cache
def _checked_config() -> None:
    cfg = oracle.text_config()
    expected = {"hidden_size": HIDDEN, "linear_num_key_heads": HK, "linear_num_value_heads": HV,
                "linear_key_head_dim": DK, "linear_value_head_dim": DV, "linear_conv_kernel_dim": CONV_K,
                "hidden_act": "silu", "rms_norm_eps": RMS_EPS}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise oracle.OracleError(f"text_config {key}={cfg.get(key)!r}, K5 decode is written for {value!r}")


def _expect(name: str, t: torch.Tensor, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    if tuple(t.shape) != shape or t.dtype != dtype:
        raise oracle.OracleError(f"{name}: {t.dtype} {tuple(t.shape)}, expected {dtype} {shape}")
    return t.contiguous()


@functools.lru_cache(maxsize=len(LAYERS))
def _weights(layer: int) -> types.SimpleNamespace:
    """One GDN layer's `linear_attn` weights on GPU0 in the dtypes SGLang's Qwen3_5GatedDeltaNet holds them
    (`oracle.gdn_layer_weights` refuses a non-`linear_attention` id)."""
    _checked_config()
    raw = oracle.gdn_layer_weights(layer)
    p = oracle.layer_prefix(layer) + "linear_attn."

    def get(name, shape):
        return _expect(p + name, raw[name], shape, BF16)

    w = types.SimpleNamespace(layer=layer)
    w.qkv = get("in_proj_qkv.weight", (CONV_DIM, HIDDEN))
    w.b = get("in_proj_b.weight", (HV, HIDDEN))
    w.a = get("in_proj_a.weight", (HV, HIDDEN))
    w.conv = get("conv1d.weight", (CONV_DIM, 1, CONV_K)).reshape(CONV_DIM, CONV_K).contiguous()
    w.A_log = get("A_log", (HV,)).float().contiguous()  # SGLang's A_log parameter is float32 (and decode passes .float())
    w.dt_bias = get("dt_bias", (HV,))  # SGLang's dt_bias parameter is the model dtype (bf16)
    w.ba = torch.cat([w.b, w.a], 0).contiguous()  # SGLang's fused in_proj_ba
    return w


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


def _slot() -> torch.Tensor:
    return torch.zeros(1, dtype=torch.int32, device=oracle.DEVICE)


def _cu(tokens: int) -> torch.Tensor:
    return torch.tensor([0, tokens], dtype=torch.int32, device=oracle.DEVICE)


def _activations(w: types.SimpleNamespace, seed: int, tokens: int) -> types.SimpleNamespace:
    """conv_out bf16 [tokens, CONV_DIM] (q | k | v after SGLang's prefill conv + silu from a fresh conv state) and
    a, b bf16 [tokens, HV] (in_proj_ba split), on the projections of layer `w`."""
    hidden = _hidden(seed, tokens)
    mixed_qkv = F.linear(hidden, w.qkv)
    ba = F.linear(hidden, w.ba)
    conv_out = causal_conv1d_fn(
        mixed_qkv.clone().transpose(0, 1),
        w.conv,
        None,
        activation="silu",
        conv_states=torch.zeros(1, CONV_DIM, CONV_K - 1, dtype=BF16, device=oracle.DEVICE),
        has_initial_state=torch.tensor([False], device=oracle.DEVICE),
        cache_indices=_slot(),
        query_start_loc=_cu(tokens),
        seq_lens_cpu=[tokens],
    )
    return types.SimpleNamespace(conv_out=conv_out.transpose(0, 1)[:tokens].contiguous(),
                                 b=ba[:, :HV].contiguous(), a=ba[:, HV:].contiguous())


def _fla_prefill_state(w: types.SimpleNamespace, act: types.SimpleNamespace, tokens: int) -> torch.Tensor:
    """The BF16 state [HV, DV, DK] FLA `chunk_gated_delta_rule` leaves after a fresh prefill of the first `tokens`
    activations (SGLang's extend path: fused split, `fused_gdn_gating`, in-kernel L2 norm)."""
    pool = torch.zeros(1, HV, DV, DK, dtype=BF16, device=oracle.DEVICE)
    q, k, v = fused_qkv_split_gdn_prefill(act.conv_out[:tokens], HK, HK, HV, DK, DK, DV)
    g, beta = fused_gdn_gating(w.A_log, act.a[:tokens], act.b[:tokens], w.dt_bias)
    chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=pool, initial_state_indices=_slot(),
                           cu_seqlens=_cu(tokens), head_first=False, use_qk_l2norm_in_kernel=True)
    return pool[0].clone()


# -- reference and native steps -----------------------------------------------------------------------------------


def _flashinfer_step(w: types.SimpleNamespace, row: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                     pool: torch.Tensor) -> torch.Tensor:
    """`FlashInferGDNKernel.decode` with the state pool (gdn_flashinfer.py:239-251), batch 1, slot 0; `pool`
    [1, HV, DV, DK] is updated in place. Returns o bf16 [HV, DV]."""
    q, k, v = torch.split(row.view(1, CONV_DIM), [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    o, _ = gated_delta_rule_decode_pretranspose(
        q=q.view(1, 1, HK, DK), k=k.view(1, 1, HK, DK), v=v.view(1, 1, HV, DV), state=None,
        A_log=w.A_log.detach().float(), a=a.view(1, 1, HV), dt_bias=w.dt_bias.detach(), b=b.view(1, 1, HV),
        use_qk_l2norm=True, initial_state=pool, initial_state_indices=_slot())
    return o.view(HV, DV)


def _native_step(w: types.SimpleNamespace, row: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                 state: torch.Tensor) -> torch.Tensor:
    """`cere_ktest_gdn_decode_step` on one token; `state` [HV, DV, DK] is updated in place. Returns o bf16 [HV, DV]."""
    q = row[:KEY_DIM].contiguous()
    k = row[KEY_DIM:2 * KEY_DIM].contiguous()
    v = row[2 * KEY_DIM:].contiguous()
    o = oracle.empty((HV, DV), BF16)
    oracle.call("cere_ktest_gdn_decode_step", q, k, v, a.contiguous(), b.contiguous(), w.A_log, w.dt_bias, state, o,
                SCALE)
    return o


def _bit_equal_fraction(native: torch.Tensor, reference: torch.Tensor) -> float:
    a = native.contiguous().view(torch.int16).reshape(-1)
    b = reference.contiguous().view(torch.int16).reshape(-1)
    return float((a == b).sum().item()) / a.numel()


def _check_exact_or_ulp(tc: unittest.TestCase, case: str, native: torch.Tensor, reference: torch.Tensor) -> None:
    """Bit-exact, else at most 1e-3 of the elements differ, each by at most 1 BF16 ulp (C5)."""
    if _bit_equal_fraction(native, reference) == 1.0:
        oracle.check_bit_exact(tc, ROW, case, native, reference)
    else:
        oracle.check_ulp(tc, ROW, case, native, reference, max_fraction=1e-3)


# -- cases --------------------------------------------------------------------------------------------------------


class DecodeFlashInfer(unittest.TestCase):
    """`delta_rule_decode` vs FlashInfer's BF16-state decode kernel in pool mode, 64 sequential steps, per layer."""

    def _decode_case(self, case: str, w: types.SimpleNamespace, act: types.SimpleNamespace, state0: torch.Tensor):
        case = f"layer{w.layer}/{case}"
        first = act.conv_out.shape[0] - STEPS
        pool = state0.clone().unsqueeze(0)
        nat_state = state0.clone()
        ref_o = torch.empty(STEPS, HV, DV, dtype=BF16, device=oracle.DEVICE)
        nat_o = torch.empty_like(ref_o)
        for t in range(STEPS):
            i = first + t
            ref_o[t] = _flashinfer_step(w, act.conv_out[i], act.a[i], act.b[i], pool)
            nat_o[t] = _native_step(w, act.conv_out[i], act.a[i], act.b[i], nat_state)
        steps_exact = sum(int(torch.equal(nat_o[t].view(torch.int16), ref_o[t].view(torch.int16)))
                          for t in range(STEPS))
        oracle.report_row(ROW, f"{case}/bit_equal", phase="bit_equal_fraction", steps=STEPS,
                          steps_o_bit_exact=steps_exact, o=_bit_equal_fraction(nat_o, ref_o),
                          ssm_state=_bit_equal_fraction(nat_state, pool[0]))
        if _bit_equal_fraction(nat_o, ref_o) == 1.0:
            oracle.check_bit_exact(self, ROW, f"{case}/o", nat_o, ref_o)
        else:  # the ulp budget applies to each step's o, not to the 64 steps stacked (review M22NI-06)
            for t in range(STEPS):
                _check_exact_or_ulp(self, f"{case}/o[{t}]", nat_o[t], ref_o[t])
        _check_exact_or_ulp(self, f"{case}/ssm_state", nat_state, pool[0])

    def test_decode_from_random_state(self):
        """(i) A seeded random BF16 state with the RMS of a real 27-token prefill state, then 64 steps."""
        for layer in LAYERS:
            with self.subTest(layer=layer):
                w = _weights(layer)
                act = _activations(w, 5501, PREFILL + STEPS)
                rms = _fla_prefill_state(w, act, PREFILL).float().pow(2).mean().sqrt()
                gen = torch.Generator(device="cpu")
                gen.manual_seed(5502)
                state0 = (torch.randn(HV, DV, DK, generator=gen).to(oracle.DEVICE) * rms).to(BF16).contiguous()
                self._decode_case("delta_rule_decode_fi_random_state", w, act, state0)

    def test_decode_from_prefill_state(self):
        """(ii) The state of a fresh FLA prefill of L=27, then 64 steps on the following tokens."""
        for layer in LAYERS:
            with self.subTest(layer=layer):
                w = _weights(layer)
                act = _activations(w, 5503, PREFILL + STEPS)
                self._decode_case("delta_rule_decode_fi_prefill_state", w, act, _fla_prefill_state(w, act, PREFILL))

    def test_decode_large_gate_input_stays_finite(self):
        """SS-3 (hardening): a gate input a + dt_bias past ~88 (log(1+exp(x)) = +inf) takes softplus = x (decay ->
        0), not a NaN: a finite, non-trivial o and a finite state. FlashInfer's own kernel returns NaN here
        (observed), so it is no reference for this input; below ~88 the two agree (the other tests)."""
        layer = LAYERS[0]
        w = _weights(layer)
        act = _activations(w, 5504, PREFILL + STEPS)
        state0 = _fla_prefill_state(w, act, PREFILL)
        i = act.conv_out.shape[0] - 1
        for gate in (60.0, 100.0, 200.0):
            with self.subTest(gate=gate):
                a = torch.full_like(act.a[i], gate)
                nat_state = state0.clone()
                native = _native_step(w, act.conv_out[i], a, act.b[i], nat_state)
                self.assertTrue(bool(torch.isfinite(native.float()).all()), "o")
                self.assertTrue(bool(torch.isfinite(nat_state.float()).all()), "state")
                self.assertTrue(bool((native.float() != 0).any()), "o is not all zero")


if __name__ == "__main__":
    unittest.main()
