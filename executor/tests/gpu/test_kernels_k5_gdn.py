"""K5 (tasks/m22-spec.md): the native GDN linear-attention kernels against SGLang's own ops on real GDN-layer weights.

Container only (pinned SGLang image, GPU0, `/out/libcere_ktest.so`, `/model` read-only); suites `kernels`,
`kernels_k5`.

Layers (tasks/m22d-spec.md D5): every case below except the C7 op rows runs on each GDN layer of LAYERS {0, 1, 18,
46} (first, second, middle and last `linear_attention` layer), each with that layer's own weights (conv, A_log,
dt_bias, norm, projections; `oracle.gdn_layer_weights` refuses an id whose `layer_types` entry is not
`linear_attention`) and the same seeded inputs; case names are prefixed `layer<id>/`. The layer-0 limits apply
unchanged to every layer.

What is compared (each case first writes its calibration `c` = rel-L2(SGLang op, FP64 torch reference written in this
file, same inputs) to the report, then calls the native entry point and checks it against the SGLang op):

- port rows (tasks/m22n-spec.md C3, C4, C4b; the reference is the kernel the no-speculation control dispatches, `c`
  is context only): causal conv1d bit-exact (y and conv state) vs `causal_conv1d_fn` routed to Triton exactly as
  `gdn_backend.forward_extend` calls it (prefill L in CONV_PREFILL_LENGTHS, fresh and carried) and vs Triton
  `causal_conv1d_update` (CONV_UPDATE_STEPS sequential single-token steps); L2 norm bit-exact vs FLA `l2norm_fwd`
  (BF16 out, T 64 and 4096); gating vs `fused_gdn_gating` (beta rounded to BF16) by `oracle.check_ulp` (max
  fraction 1e-3 of the elements 1 ulp apart; 0 differing = bit-exact). Each row reports the lowering variant the
  library was built with (`cere_ktest_gdn_variants`);
- pieces, floor `FLOOR_OP`, rel-L2 against max(floor, 2c): gated RMSNorm vs FLA `layernorm_fn` (sigmoid gate, norm
  before gate, the model's `output_gate_type`);
- layer-core port rows (tasks/m22n-spec.md C7, D7; `oracle.port_limit`, the reference is the control's served path):
  conv -> gating -> delta rule -> gated RMSNorm on SGLang's own projections (mixed_qkv, z, b, a of `in_proj_qkvz` /
  `in_proj_ba`), the native core called with the operation the control would run. PREFILL L in
  LAYER_CORE_PREFILL_LENGTHS (L=1 included: the control routes by forward mode, so a 1-token prompt is an extend)
  vs `forward_extend`: Triton `causal_conv1d_fn`, `fused_gdn_gating`, FLA `chunk_gated_delta_rule` with
  `use_qk_l2norm_in_kernel`, limit 5e-4 on y and the SSM state; a chain of a CHAIN_PREFILL-token PREFILL then
  CHAIN_STEPS single-token DECODE calls vs the extend path then `forward_decode` (Triton `causal_conv1d_update`,
  FlashInfer `gated_delta_rule_decode_pretranspose` in pool mode as `gdn_flashinfer.py:239-251`), limit 1e-3 on the
  decode steps' y and the final SSM state. The conv state is bit-exact in both (C3);
- the verify chain (tasks/m23-spec.md DG-02; `GdnLayerCore.test_chain_core_equals_per_row`, bit-exact):
  `cere_ktest_gdn_chain_core` on consecutive chains of CHAIN_CORE_ROWS (1..9) rows after a CHAIN_PREFILL-token
  PREFILL, each from the state the last one left, equals the same rows through `cere_ktest_gdn_layer_core` DECODE
  one token at a time: y, both final states and the state after every row but the last (the undo snapshots);
  R16-B (tasks/rows16-spec.md; `test_chains_core_equals_per_row`): `cere_ktest_gdn_chains_core`, one call over the
  chains of CHAINS_CORE_LAYOUTS, each on its own sequence's state, equals the per-row calls per chain;
- the whole linear-attention layer (in_proj -> conv -> gating -> delta rule -> gated RMSNorm -> out_proj), floor
  `FLOOR_BLOCK` for the layer output and both final states (calibrated limits; the K2 GEMMs vs torch's dominate):
  prefill L in {1, 64, 2048} vs SGLang's extend path (`chunk_gated_delta_rule`); decode after a 64-token prefill vs
  FlashInfer's GDN decode in pool mode (64 single-token steps; the primary decode oracle: it is the kernel the
  no-speculation control dispatches, tasks/m22n-spec.md D15) and, report-only (rel-L2 rows, no assertion), vs
  SGLang's target-verify path in 4-token steps (`causal_conv1d_update` with intermediate conv windows +
  `fused_sigmoid_gating_delta_rule_update` with intermediate states and `disable_state_update`, then the commit of
  the last accepted step, all 4 drafts accepted): the bench's NEXTN verify kernel is not a port target, so a correct
  port of the control's kernel may lie on either side of it. Both decode references and the native decode start
  from the identical BF16 state bytes SGLang's prefill left behind, so the decode cases measure decode only (prefill
  has its own cases). GDN state is BF16 at rest everywhere (Q8 override of `mamba_ssm_dtype`).

The native delta rule has two paths, each tested against its control kernel in its own file: the chunk-64 WY prefill
(`test_kernels_k5_chunk.py`, `cere_ktest_gdn_chunk_prefill`) and the FlashInfer-style decode
(`test_kernels_k5_decode.py`, `cere_ktest_gdn_decode_step`); here they run inside the layer.

Layer inputs are real embedding rows of seeded random token ids, RMS-normalised (the pre-mixer norm without its
weight; the same seeds on every layer); token ids never leave the process.

D5 whole-layer chain (`GdnLayer.test_chain_prefill_then_decode`): native `layer_forward` PREFILL of P tokens (P in
LAYER_CHAIN_PREFILLS {21, 27, 32}, native projections), then CHAIN_STEPS (64) native DECODE steps, against the
control's composition (extend: projections, Triton conv, `fused_gdn_gating`, FLA `chunk_gated_delta_rule`, gated
norm, out_proj; then FlashInfer decode steps), each side and the FP64 reference carrying its own state. Rows per
step: `layer<id>/layer_chain_P<P>_decode64/step<s>/out` and `.../ssm_state`, calibrated and checked as the other
`GdnLayer` rows (`FLOOR_BLOCK`, limit max(floor, 2c)).

C7 GDN-block decode ops (tasks/m22d-spec.md C7, D6; `GdnDecodeOps`, report-only, on every `linear_attention` layer):
the control runs a fresh ISO_PREFILL (27)-token extend, then ISO_STEPS (64) FlashInfer decode steps; at each step every
native op of the DECODE path is fed exactly the control's inputs of that step and compared with the kernel the
control dispatches for it: in_proj_qkv / in_proj_z / in_proj_b / in_proj_a at M=1 (`cere_ktest_gemm_bf16`, the
executor's GEMM plan, vs the matching columns of the control's fused `in_proj_qkvz` / `in_proj_ba` GEMMs), the conv
update (`cere_ktest_gdn_conv1d`, 1 token, vs Triton `causal_conv1d_update`; y and conv state), the delta-rule decode
step with its in-kernel gating and q/k L2 norm (`cere_ktest_gdn_decode_step` vs FlashInfer
`gated_delta_rule_decode_pretranspose`; o and ssm state), the gated RMSNorm of one token's 48 heads
(`cere_ktest_gdn_gated_rmsnorm` vs FLA `layernorm_fn`) and out_proj at M=1 (`cere_ktest_gemm_bf16` vs `F.linear`).
Rows `layer<id>/decode_op/<op>`: a calibration row (c vs FP64 on the same inputs), then rel-L2(native, control) over
the 64 steps, the differing element count and the first differing step. Neither decode path runs a standalone gating
or L2-norm kernel (both are inside the decode-step kernel on both sides), so they have no separate row; the HC mix
and the pre-mixer norm are K3's (`test_kernels_k3_hc.py`).

Native entry points (the contract with the kernel track; every buffer is a contiguous row-major device pointer, every
`int64_t` a count, `float` a scalar, the stream is last, the return value is 0 or a nonzero `cere_status`):

    int cere_ktest_gdn_conv1d(const void *x, const void *weight, void *conv_state, void *y,
                              int64_t tokens, int64_t dim, cudaStream_t stream);
        x bf16 [tokens, dim] (token-major, the in_proj_qkv output); weight bf16 [dim, 4] (tap 3 multiplies the
        current token); conv_state bf16 [dim, 3] in/out: the 3 previous raw inputs, oldest first (zeros = a fresh
        sequence), on return the last 3 raw inputs; y bf16 [tokens, dim] = silu(depthwise causal conv, no bias).
    int cere_ktest_gdn_gating(const void *a, const void *b, const void *A_log, const void *dt_bias,
                              void *g, void *beta, int64_t tokens, int64_t heads, cudaStream_t stream);
        a, b bf16 [tokens, heads]; A_log f32 [heads]; dt_bias bf16 [heads]; g, beta f32 [tokens, heads];
        g = -exp(A_log) * softplus(a + dt_bias) (softplus(x) = x for x > 20), beta = bf16(sigmoid(b)) as f32.
    int cere_ktest_gdn_l2norm(const void *x, void *y, int64_t rows, int64_t dim, float eps, cudaStream_t stream);
        x, y bf16 [rows, dim], dim 128; y = x / sqrt(sum(x^2) + eps) per row, FP32 arithmetic, BF16 out.
    int cere_ktest_gdn_variants(void *out, cudaStream_t stream);
        out int32 [3] = {kConvVariant, kGatingVariant, kL2normVariant} of k5_gdn.h.
    int cere_ktest_gdn_gated_rmsnorm(const void *x, const void *z, const void *weight, void *y,
                                     int64_t rows, int64_t dim, float eps, cudaStream_t stream);
        x, z, y bf16 [rows, dim]; weight bf16 [dim]; y = x * rsqrt(mean(x^2) + eps) * weight * sigmoid(z).
    int cere_ktest_gdn_layer_core(const void *mixed_qkv, const void *z, const void *b, const void *a,
                                  const void *conv_weight, const void *A_log, const void *dt_bias,
                                  const void *norm_weight, void *conv_state, void *ssm_state, void *y,
                                  int64_t tokens, int64_t operation, cudaStream_t stream);
        The mixer between the projections and out_proj: mixed_qkv bf16 [tokens, 10240] (q 16x128 | k 16x128 |
        v 48x128), z bf16 [tokens, 6144], b, a bf16 [tokens, 48]; weights and states as for the layer entry points
        below; y bf16 [tokens, 6144] = gated RMSNorm(o, z). operation CERE_OP_PREFILL (1): conv, gating
        (fused_gdn_gating), chunk-64 delta rule (FLA chunk_gated_delta_rule, in-kernel L2 norm), tokens 1..4096;
        CERE_OP_DECODE (2): conv update and the FlashInfer-style decode step (gating and L2 norm in-kernel),
        tokens 1. Scale 128^-0.5.
    int cere_ktest_gdn_chain_core(const void *mixed_qkv, const void *z, const void *b, const void *a,
                                  const void *conv_weight, const void *A_log, const void *dt_bias,
                                  const void *norm_weight, void *conv_state, void *ssm_state, void *conv_saves,
                                  void *ssm_saves, void *y, int64_t rows, cudaStream_t stream);
        The DECODE core over the rows of one verify chain (k5::chain_core): inputs as layer_core's with tokens =
        rows (1..16, else CERE_ERR_ARGUMENT); conv_saves bf16 [rows - 1, 10240, 3] and ssm_saves bf16
        [rows - 1, 48, 128, 128] (null when rows is 1, else CERE_ERR_ARGUMENT; aliasing the states is
        CERE_ERR_ARGUMENT) take the state after each row but the last.
    int cere_ktest_gdn_chains_core(<as cere_ktest_gdn_chain_core through y, with conv_states [C, 10240, 3] and
                                   ssm_states [C, 48, 128, 128] for the states and conv_saves / ssm_saves
                                   [rows, ...] row-indexed>, int64_t rows, const void *chain_rows, int64_t chains,
                                   cudaStream_t stream);
        k5::chain_core over `chains` chains back to back (host int64 chain_rows [chains], summing to rows), chain c
        on states[c]; the save of row r (every row but its chain's last) at index r.
    int cere_ktest_gdn_layer_prefill(const void *hidden, const void *w_qkv, const void *w_z, const void *w_b,
                                     const void *w_a, const void *conv_weight, const void *A_log,
                                     const void *dt_bias, const void *norm_weight, const void *w_out,
                                     void *conv_state, void *ssm_state, void *out, int64_t tokens,
                                     cudaStream_t stream);
        hidden bf16 [tokens, 2560]; w_qkv bf16 [10240, 2560] (rows: q 16x128 | k 16x128 | v 48x128); w_z bf16
        [6144, 2560]; w_b, w_a bf16 [48, 2560]; conv_weight bf16 [10240, 4]; A_log f32 [48]; dt_bias bf16 [48];
        norm_weight bf16 [128]; w_out bf16 [2560, 6144]; conv_state bf16 [10240, 3] and ssm_state bf16
        [48, 128 (v), 128 (k)] in/out (SGLang's K-last layout; zeros = fresh sequence); out bf16 [tokens, 2560].
        The whole linear-attention mixer as a PREFILL (tokens 1..4096): mixed = hidden w_qkv^T, z = hidden w_z^T,
        b = hidden w_b^T, a = hidden w_a^T, the PREFILL core above, out = y w_out^T.
    int cere_ktest_gdn_layer_decode(const void *hidden, const void *w_qkv, const void *w_z, const void *w_b,
                                    const void *w_a, const void *conv_weight, const void *A_log,
                                    const void *dt_bias, const void *norm_weight, const void *w_out,
                                    void *conv_state, void *ssm_state, void *out, cudaStream_t stream);
        The same mixer as one DECODE step: hidden and out bf16 [1, 2560]; states updated in place (BF16 at rest).
    int cere_ktest_gdn_decode_step(const void *q, const void *k, const void *v, const void *a, const void *b,
                                   const void *A_log, const void *dt_bias, void *state, void *o, float scale,
                                   cudaStream_t stream);
        The native decode's delta rule (test_kernels_k5_decode.py): one token's q, k bf16 [16, 128] (raw; L2 norm
        in-kernel), v bf16 [48, 128], a, b bf16 [48]; state bf16 [48, 128 (v), 128 (k)] in/out; o bf16 [48, 128].
    int cere_ktest_gemm_bf16(const void *x, const void *w, int64_t m, int64_t n, int64_t k, void *y,
                             cudaStream_t stream);
        The executor's BF16 GEMM plan (test_kernels_k2_gemm.py): y bf16 [m, n] = x bf16 [m, k] w bf16 [n, k]^T.
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
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.kernels.ops.attention.fla.layernorm_gated import layernorm_fn
from sglang.kernels.ops.attention.triton_gdn_fused_proj import fused_qkv_split_gdn_prefill
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
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
RMS_EPS = 1e-6  # text_config rms_norm_eps (the gated norm's eps in Qwen3_5GatedDeltaNet)
L2_EPS = 1e-6  # FLA l2norm_fwd default and the verify kernel's in-kernel L2 norm
SCALE = DK ** -0.5  # SGLang passes no scale: k.shape[-1] ** -0.5
DRAFT = 4  # target-verify step width (EAGLE 3/1/4, Q8)
PREFILL_BEFORE_DECODE = 64
DECODE_TOKENS = 64
CONV_PREFILL_LENGTHS = (1, 21, 27, 32, 64, 65, 4096)  # C3 (m22n design note)
CONV_UPDATE_STEPS = 64
LAYER_CORE_PREFILL_LENGTHS = (1, 21, 24, 27, 32, 64, 65, 128, 2048, 4096)  # C7 (m22n design note)
CHAIN_PREFILL = 27  # C7 chain: a PREFILL of this length, then CHAIN_STEPS DECODE steps
CHAIN_STEPS = 64
CHAIN_CORE_ROWS = (4, 1, 3, 2, 4, 9)  # DG-02: consecutive verify chains (the committed token and 0..8 drafts)
# R16-B (rows16 design note): the chain layouts of one chain_core call over several sequences (r4, r1, DraftAccept
# chains, decode-sized chains, one 16-row chain and 16 one-row chains).
CHAINS_CORE_LAYOUTS = ((4, 4, 4, 4), (4,), (9, 3, 1, 3), (2, 5, 1), (16,), (1,) * 16)
LAYER_CHAIN_PREFILLS = (21, 27, 32)  # D5 whole-layer chain: native PREFILL of P tokens, then CHAIN_STEPS DECODE
ISO_PREFILL = 27  # C7 isolation (m22d design note D6): the control's extend before the decode steps
ISO_STEPS = 64  # C7 isolation: decode steps per layer, every GDN-block op fed the control's inputs of the step
OP_PREFILL = 1  # CERE_OP_PREFILL (cere_executor.h)
OP_DECODE = 2  # CERE_OP_DECODE
BF16 = torch.bfloat16
EMBED = "model.language_model.embed_tokens.weight"


# -- model facts and weights ------------------------------------------------------------------------------------


@functools.cache
def _checked_config() -> None:
    cfg = oracle.text_config()
    expected = {"hidden_size": HIDDEN, "linear_num_key_heads": HK, "linear_num_value_heads": HV,
                "linear_key_head_dim": DK, "linear_value_head_dim": DV, "linear_conv_kernel_dim": CONV_K,
                "hidden_act": "silu", "output_gate_type": "sigmoid", "rms_norm_eps": RMS_EPS}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise oracle.OracleError(f"text_config {key}={cfg.get(key)!r}, K5 is written for {value!r}")


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
    w.z = get("in_proj_z.weight", (VALUE_DIM, HIDDEN))
    w.b = get("in_proj_b.weight", (HV, HIDDEN))
    w.a = get("in_proj_a.weight", (HV, HIDDEN))
    w.conv = get("conv1d.weight", (CONV_DIM, 1, CONV_K)).reshape(CONV_DIM, CONV_K).contiguous()
    w.A_log = get("A_log", (HV,)).float().contiguous()  # SGLang's A_log parameter is float32
    w.dt_bias = get("dt_bias", (HV,))  # SGLang's dt_bias parameter is the model dtype (bf16)
    w.norm = get("norm.weight", (DV,))
    w.out = get("out_proj.weight", (HIDDEN, VALUE_DIM))
    # SGLang loads in_proj_qkv/in_proj_z into one in_proj_qkvz GEMM and in_proj_b/in_proj_a into in_proj_ba
    # (qwen4_exp.py stacked_params_mapping); the reference ops below use the fused weights the same way.
    w.qkvz = torch.cat([w.qkv, w.z], 0).contiguous()
    w.ba = torch.cat([w.b, w.a], 0).contiguous()
    return w


@functools.lru_cache(maxsize=1)
def _weights64(layer: int) -> types.SimpleNamespace:
    """The FP64 copies of the weights the FP64 references read (not the fused qkvz / ba)."""
    w = _weights(layer)
    return types.SimpleNamespace(**{k: getattr(w, k).to(torch.float64)
                                    for k in ("qkv", "z", "b", "a", "conv", "A_log", "dt_bias", "norm", "out")})


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


def _sglang_conv_state() -> torch.Tensor:
    """One-slot conv pool [1, CONV_DIM, 3] (Mamba2StateShape conv: (dim, K-1), window axis last)."""
    return torch.zeros(1, CONV_DIM, CONV_K - 1, dtype=BF16, device=oracle.DEVICE)


def _sglang_ssm_state() -> torch.Tensor:
    """One-slot temporal pool [1, HV, DV, DK] (K-last), BF16 at rest (Q8)."""
    return torch.zeros(1, HV, DV, DK, dtype=BF16, device=oracle.DEVICE)


# -- FP64 references (calibration) ------------------------------------------------------------------------------


def _ref_conv64(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor):
    """x [T, D], weight [D, K], state [D, K-1] -> (silu(causal conv) [T, D], new state [D, K-1]), FP64."""
    x, weight, state = x.double(), weight.double(), state.double()
    tokens = x.shape[0]
    window = torch.cat([state.t(), x], 0)  # [K-1+T, D]
    y = sum(window[j:j + tokens] * weight[:, j] for j in range(CONV_K))
    return F.silu(y), window[-(CONV_K - 1):].t().contiguous()


def _ref_gating64(a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor):
    a, b = a.double(), b.double()
    g = -torch.exp(A_log.double()) * F.softplus(a + dt_bias.double(), beta=1.0, threshold=20.0)
    return g, torch.sigmoid(b)


def _ref_l2norm64(x: torch.Tensor) -> torch.Tensor:
    x = x.double()
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + L2_EPS)


def _ref_gated_rmsnorm64(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x, z = x.double(), z.double()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS) * weight.double() * torch.sigmoid(z)


def _ref_delta64(w64: types.SimpleNamespace, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: torch.Tensor,
                 a: torch.Tensor, ssm_state: torch.Tensor):
    """The token recurrence of the gated delta rule in FP64 on the conv outputs (q, k [T, KEY_DIM] raw, L2-normalised
    here; v [T, VALUE_DIM]; value head h uses key head h // (HV // HK)) with the gating of b, a [T, HV]. Returns
    (o [T, HV, DV], ssm [HV, DV, DK])."""
    tokens = q.shape[0]
    g, beta = _ref_gating64(a, b, w64.A_log, w64.dt_bias)
    rep = HV // HK
    q = _ref_l2norm64(q.reshape(tokens, HK, DK)).repeat_interleave(rep, dim=1) * SCALE
    k = _ref_l2norm64(k.reshape(tokens, HK, DK)).repeat_interleave(rep, dim=1)
    v = v.double().reshape(tokens, HV, DV)
    s = ssm_state.double().transpose(1, 2).contiguous()  # [HV, DK, DV]
    decay = torch.exp(g)
    o = torch.empty(tokens, HV, DV, dtype=torch.float64, device=q.device)
    for t in range(tokens):
        s.mul_(decay[t].view(HV, 1, 1))
        kv = torch.bmm(k[t].unsqueeze(1), s).squeeze(1)  # [HV, DV]
        u = (v[t] - kv) * beta[t].unsqueeze(-1)
        s.add_(k[t].unsqueeze(2) * u.unsqueeze(1))
        o[t] = torch.bmm(q[t].unsqueeze(1), s).squeeze(1)
    return o, s.transpose(1, 2).contiguous()


def _ref_core64(w: types.SimpleNamespace, mixed: torch.Tensor, z: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                conv_state: torch.Tensor, ssm_state: torch.Tensor):
    """The mixer core of layer `w` in FP64 from its inputs (mixed [T, CONV_DIM], z [T, VALUE_DIM], b, a [T, HV]):
    conv + SiLU, gating, the gated delta rule (`_ref_delta64`), gated RMSNorm. Returns (y [T, VALUE_DIM],
    conv_state [CONV_DIM, 3], ssm [HV, DV, DK])."""
    w64 = _weights64(w.layer)
    tokens = mixed.shape[0]
    y, conv_new = _ref_conv64(mixed, w64.conv, conv_state)
    q, k, v = y.split([KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    o, ssm_new = _ref_delta64(w64, q, k, v, b, a, ssm_state)
    normed = _ref_gated_rmsnorm64(o.reshape(-1, DV), z.reshape(-1, DV), w64.norm)
    return normed.reshape(tokens, VALUE_DIM), conv_new, ssm_new


def _ref_layer64(w: types.SimpleNamespace, hidden: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor):
    """The whole mixer of layer `w` in FP64 from bf16 inputs (states bf16 or FP64): (out [T, HIDDEN],
    conv_state [CONV_DIM, 3], ssm [HV, DV, DK]), all FP64."""
    w64 = _weights64(w.layer)
    x = hidden.double()
    y, conv_new, ssm_new = _ref_core64(w, F.linear(x, w64.qkv), F.linear(x, w64.z), F.linear(x, w64.b),
                                       F.linear(x, w64.a), conv_state, ssm_state)
    return F.linear(y, w64.out), conv_new, ssm_new


# -- SGLang reference ops (the code paths of gdn_backend.py / qwen3_5.py, one request in slot 0, layer `w`) -------


def _sglang_proj(w: types.SimpleNamespace, hidden: torch.Tensor):
    """in_proj_qkvz / in_proj_ba and the split of `fused_qkvzba_split_reshape_cat_contiguous`: mixed_qkv, z, b, a."""
    qkvz = F.linear(hidden, w.qkvz)
    ba = F.linear(hidden, w.ba)
    return (qkvz[:, :CONV_DIM].contiguous(), qkvz[:, CONV_DIM:].contiguous(),
            ba[:, :HV].contiguous(), ba[:, HV:].contiguous())


def _sglang_prefill_conv(w: types.SimpleNamespace, mixed_qkv: torch.Tensor, conv_state: torch.Tensor,
                         has_initial_state: bool):
    """`forward_extend`'s conv: x is the (dim, T) transpose with seq_lens_cpu, which routes to the Triton kernel."""
    tokens = mixed_qkv.shape[0]
    out = causal_conv1d_fn(
        mixed_qkv.clone().transpose(0, 1),
        w.conv,
        None,
        activation="silu",
        conv_states=conv_state,
        has_initial_state=torch.tensor([has_initial_state], device=oracle.DEVICE),
        cache_indices=_slot(),
        query_start_loc=_cu(tokens),
        seq_lens_cpu=[tokens],
    )
    return out.transpose(0, 1)[:tokens]


def _sglang_gated_norm(w: types.SimpleNamespace, o: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """RMSNormGated(head_v_dim, eps, group_size=None, norm_before_gate=True, activation=output_gate_type)."""
    return layernorm_fn(o.reshape(-1, DV), w.norm, None, z=z.reshape(-1, DV), eps=RMS_EPS,
                        group_size=None, norm_before_gate=True, is_rms_norm=True, activation="sigmoid")


def _sglang_mixer_out(w: types.SimpleNamespace, o: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    tokens = z.shape[0]
    return F.linear(_sglang_gated_norm(w, o, z).reshape(tokens, VALUE_DIM), w.out)


def _sglang_extend_core(w: types.SimpleNamespace, mixed_qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                        conv_state: torch.Tensor, ssm_state: torch.Tensor, has_initial_state: bool) -> torch.Tensor:
    """Extend path on the projections: conv, fused split, `fused_gdn_gating`, `chunk_gated_delta_rule` with the
    in-kernel L2 norm (TritonGDNKernel.extend). Returns o [1, T, HV, DV]; the pools are updated in place."""
    tokens = mixed_qkv.shape[0]
    conv_out = _sglang_prefill_conv(w, mixed_qkv, conv_state, has_initial_state)
    q, k, v = fused_qkv_split_gdn_prefill(conv_out, HK, HK, HV, DK, DK, DV)
    g, beta = fused_gdn_gating(w.A_log, a, b, w.dt_bias)
    o, _, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=ssm_state,
                                     initial_state_indices=_slot(), cu_seqlens=_cu(tokens), head_first=False,
                                     use_qk_l2norm_in_kernel=True)
    return o


def _sglang_prefill_core(w: types.SimpleNamespace, hidden: torch.Tensor, conv_state: torch.Tensor,
                         ssm_state: torch.Tensor, has_initial_state: bool):
    """Projections, then the extend path. Returns (o [1, T, HV, DV], z [T, VALUE_DIM]); the pools are updated in
    place."""
    mixed_qkv, z, b, a = _sglang_proj(w, hidden)
    return _sglang_extend_core(w, mixed_qkv, b, a, conv_state, ssm_state, has_initial_state), z


def _sglang_prefill(w: types.SimpleNamespace, hidden, conv_state, ssm_state, has_initial_state: bool) -> torch.Tensor:
    o, z = _sglang_prefill_core(w, hidden, conv_state, ssm_state, has_initial_state)
    return _sglang_mixer_out(w, o, z)


def _sglang_verify_step(w: types.SimpleNamespace, hidden: torch.Tensor, conv_state: torch.Tensor,
                        ssm_state: torch.Tensor) -> torch.Tensor:
    """One TARGET_VERIFY extend of DRAFT tokens (linear chain, no tree mask) and the commit of all DRAFT drafts
    (`update_mamba_state_after_mtp_verify` with last_correct_step = DRAFT-1: state <- intermediate[DRAFT-1])."""
    tokens = hidden.shape[0]
    mixed_qkv, z, b, a = _sglang_proj(w, hidden)
    window = torch.zeros(1, tokens, CONV_DIM, CONV_K - 1, dtype=BF16, device=oracle.DEVICE)  # dense layout
    inter = torch.zeros(1, tokens, HV, DV, DK, dtype=BF16, device=oracle.DEVICE)  # intermediate_ssm, ssm dtype
    conv_out = causal_conv1d_update(mixed_qkv.view(1, tokens, CONV_DIM).transpose(1, 2), conv_state, w.conv, None,
                                    "silu", conv_state_indices=_slot(), intermediate_conv_window=window,
                                    intermediate_state_indices=_slot())
    mixed = conv_out.transpose(1, 2).reshape(tokens, CONV_DIM)
    q, k, v = fused_qkv_split_gdn_prefill(mixed, HK, HK, HV, DK, DK, DV)
    o = fused_sigmoid_gating_delta_rule_update(
        A_log=w.A_log, a=a, dt_bias=w.dt_bias, softplus_beta=1.0, softplus_threshold=20.0, q=q, k=k, v=v, b=b,
        initial_state_source=ssm_state, initial_state_indices=_slot(), cu_seqlens=_cu(tokens),
        use_qk_l2norm_in_kernel=True, is_kda=False, disable_state_update=True, intermediate_states_buffer=inter,
        intermediate_state_indices=_slot(), cache_steps=tokens, retrieve_parent_token=None)
    ssm_state[0].copy_(inter[0, tokens - 1])
    conv_state[0].copy_(window[0, tokens - 1])
    return _sglang_mixer_out(w, o, z)


def _sglang_conv_update(w: types.SimpleNamespace, mixed_qkv: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
    """`forward_decode`'s conv: Triton `causal_conv1d_update` on one token (mixed_qkv [1, CONV_DIM]) with the conv
    pool [1, CONV_DIM, 3], slot 0, updated in place. Returns the q | k | v row [1, CONV_DIM]."""
    return causal_conv1d_update(mixed_qkv.clone(), conv_state, w.conv, None, "silu", conv_state_indices=_slot())


def _flashinfer_delta(w: types.SimpleNamespace, conv_out: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                      ssm_state: torch.Tensor) -> torch.Tensor:
    """`gated_delta_rule_decode_pretranspose` in pool mode on the BF16 state pool [1, HV, DV, DK], slot 0 (updated in
    place), exactly as FlashInferGDNKernel.decode calls it (gdn_flashinfer.py:239-251), on one token's q | k | v row
    [1, CONV_DIM] and b, a [1, HV] (gating and L2 norm in-kernel). Returns o [1, HV, DV]."""
    q, k, v = torch.split(conv_out, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    o, _ = gated_delta_rule_decode_pretranspose(
        q=q.view(1, 1, HK, DK), k=k.view(1, 1, HK, DK), v=v.view(1, 1, HV, DV), state=None,
        A_log=w.A_log.detach().float(), a=a.view(1, 1, HV), dt_bias=w.dt_bias.detach(), b=b.view(1, 1, HV),
        use_qk_l2norm=True, initial_state=ssm_state, initial_state_indices=_slot())
    return o.view(1, HV, DV).clone()  # the output buffer may be reused by the next call


def _flashinfer_decode_core(w: types.SimpleNamespace, mixed_qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                            conv_state: torch.Tensor, ssm_state: torch.Tensor) -> torch.Tensor:
    """`forward_decode` with the FlashInfer GDN kernel on one token's projections (mixed_qkv [1, CONV_DIM], b, a
    [1, HV]): `_sglang_conv_update`, then `_flashinfer_delta`. Returns o [1, HV, DV]; both pools are updated in
    place."""
    return _flashinfer_delta(w, _sglang_conv_update(w, mixed_qkv, conv_state), b, a, ssm_state)


def _flashinfer_decode_step(w: types.SimpleNamespace, hidden: torch.Tensor, conv_state: torch.Tensor,
                            ssm_state: torch.Tensor):
    """The whole mixer as one FlashInfer decode step (projections, `_flashinfer_decode_core`, gated norm, out_proj)."""
    mixed_qkv, z, b, a = _sglang_proj(w, hidden)
    return _sglang_mixer_out(w, _flashinfer_decode_core(w, mixed_qkv, b, a, conv_state, ssm_state), z)


# -- native entry points ----------------------------------------------------------------------------------------


def _native_conv1d(w: types.SimpleNamespace, x: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
    tokens = x.shape[0]
    y = oracle.empty((tokens, CONV_DIM), BF16)
    oracle.call("cere_ktest_gdn_conv1d", x, w.conv, conv_state, y, tokens, CONV_DIM)
    return y


def _native_gating(w: types.SimpleNamespace, a: torch.Tensor, b: torch.Tensor):
    tokens = a.shape[0]
    g = oracle.empty((tokens, HV), torch.float32)
    beta = oracle.empty((tokens, HV), torch.float32)
    oracle.call("cere_ktest_gdn_gating", a, b, w.A_log, w.dt_bias, g, beta, tokens, HV)
    return g, beta


def _native_l2norm(x: torch.Tensor) -> torch.Tensor:
    y = oracle.empty(tuple(x.shape), BF16)
    oracle.call("cere_ktest_gdn_l2norm", x, y, x.shape[0], x.shape[1], L2_EPS)
    return y


def _native_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`cere_ktest_gemm_bf16` (k2_ktest.cpp): y bf16 [m, n] = x [m, k] @ weight [n, k]^T, the executor's GEMM plan."""
    m, k = x.shape
    n = weight.shape[0]
    y = oracle.empty((m, n), BF16)
    oracle.call("cere_ktest_gemm_bf16", x.contiguous(), weight, m, n, k, y)
    return y


def _native_decode_delta(w: types.SimpleNamespace, row: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                         ssm_state: torch.Tensor) -> torch.Tensor:
    """`cere_ktest_gdn_decode_step` (the native decode's delta rule, gating and L2 norm in-kernel) on one token's
    q | k | v row [1, CONV_DIM] and a, b [1, HV]; `ssm_state` [HV, DV, DK] is updated in place. Returns o
    [1, HV, DV]."""
    row = row.reshape(-1)
    o = oracle.empty((HV, DV), BF16)
    oracle.call("cere_ktest_gdn_decode_step", row[:KEY_DIM].contiguous(), row[KEY_DIM:2 * KEY_DIM].contiguous(),
                row[2 * KEY_DIM:].contiguous(), a.contiguous(), b.contiguous(), w.A_log, w.dt_bias, ssm_state, o,
                SCALE)
    return o.view(1, HV, DV)


@functools.cache
def _native_variants() -> dict:
    """The lowering variants the library was built with (k5_gdn.h kConvVariant, kGatingVariant, kL2normVariant)."""
    out = oracle.empty((3,), torch.int32)
    oracle.call("cere_ktest_gdn_variants", out)
    conv, gating, l2norm = (int(v) for v in out.tolist())
    return {"conv": conv, "gating": gating, "l2norm": l2norm}


def _report_variant(case: str, piece: str) -> None:
    oracle.report_row(ROW, f"{case}/variant", phase="variant", piece=piece, variant=_native_variants()[piece])


def _native_gated_rmsnorm(w: types.SimpleNamespace, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    y = oracle.empty(tuple(x.shape), BF16)
    oracle.call("cere_ktest_gdn_gated_rmsnorm", x, z, w.norm, y, x.shape[0], x.shape[1], RMS_EPS)
    return y


def _native_layer_core(w: types.SimpleNamespace, mixed_qkv: torch.Tensor, z: torch.Tensor, b: torch.Tensor,
                       a: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor,
                       operation: int) -> torch.Tensor:
    """`cere_ktest_gdn_layer_core`: y bf16 [T, VALUE_DIM]; conv_state [CONV_DIM, 3] and ssm_state [HV, DV, DK] are
    updated in place."""
    tokens = mixed_qkv.shape[0]
    y = oracle.empty((tokens, VALUE_DIM), BF16)
    oracle.call("cere_ktest_gdn_layer_core", mixed_qkv, z, b, a, w.conv, w.A_log, w.dt_bias, w.norm, conv_state,
                ssm_state, y, tokens, operation)
    return y


def _native_chain_core(w: types.SimpleNamespace, mixed_qkv: torch.Tensor, z: torch.Tensor, b: torch.Tensor,
                       a: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor) -> tuple:
    """`cere_ktest_gdn_chain_core` on the rows of mixed_qkv: (y bf16 [rows, VALUE_DIM], conv saves [rows - 1,
    CONV_DIM, 3], ssm saves [rows - 1, HV, DV, DK]); the saves start NaN, the states are updated in place."""
    rows = mixed_qkv.shape[0]
    y = oracle.empty((rows, VALUE_DIM), BF16)
    conv_saves = oracle.empty((rows - 1, CONV_DIM, CONV_K - 1), BF16)
    ssm_saves = oracle.empty((rows - 1, HV, DV, DK), BF16)
    oracle.call("cere_ktest_gdn_chain_core", mixed_qkv, z, b, a, w.conv, w.A_log, w.dt_bias, w.norm, conv_state,
                ssm_state, conv_saves if rows > 1 else None, ssm_saves if rows > 1 else None, y, rows)
    return y, conv_saves, ssm_saves


def _native_layer_prefill(w: types.SimpleNamespace, hidden: torch.Tensor, conv_state: torch.Tensor,
                          ssm_state: torch.Tensor):
    """`cere_ktest_gdn_layer_prefill`: `layer_forward` PREFILL with the native projections."""
    tokens = hidden.shape[0]
    out = oracle.empty((tokens, HIDDEN), BF16)
    oracle.call("cere_ktest_gdn_layer_prefill", hidden, w.qkv, w.z, w.b, w.a, w.conv, w.A_log, w.dt_bias, w.norm,
                w.out, conv_state, ssm_state, out, tokens)
    return out


def _native_layer_decode(w: types.SimpleNamespace, hidden: torch.Tensor, conv_state: torch.Tensor,
                         ssm_state: torch.Tensor):
    """`cere_ktest_gdn_layer_decode`: `layer_forward` DECODE (one token) with the native projections."""
    out = oracle.empty((1, HIDDEN), BF16)
    oracle.call("cere_ktest_gdn_layer_decode", hidden, w.qkv, w.z, w.b, w.a, w.conv, w.A_log, w.dt_bias, w.norm,
                w.out, conv_state, ssm_state, out)
    return out


def _check_block(tc: unittest.TestCase, case: str, native: torch.Tensor, sglang_out: torch.Tensor, c: float):
    limit = oracle.calibrated_limit(c, oracle.FLOOR_BLOCK)
    oracle.check_within(tc, ROW, case, oracle.rel_l2(native, sglang_out), c=c, limit=limit)


def _report_block(case: str, native: torch.Tensor, sglang_out: torch.Tensor, c: float):
    """A report-only row (D15): rel-L2 against a non-target reference, no assertion."""
    oracle.report_row(ROW, case, phase="report_only", c=c, measured=oracle.rel_l2(native, sglang_out))


def _check_op(tc: unittest.TestCase, case: str, native: torch.Tensor, sglang_out: torch.Tensor, c: float):
    limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
    oracle.check_within(tc, ROW, case, oracle.rel_l2(native, sglang_out), c=c, limit=limit)


def _report_op_steps(case: str, native: list, control: list, reference: list) -> None:
    """A report-only C7 op row (tasks/m22d-spec.md D6) over per-step tensors (one entry per decode step, the op fed
    the control's inputs of that step): c = rel-L2(control, FP64) as a calibration row, then rel-L2(native, control)
    of the stacked steps, the elements that differ, and the first step with a differing element (-1 if none)."""
    nat, ctrl, ref = torch.stack(native, 0), torch.stack(control, 0), torch.stack(reference, 0)
    c = oracle.calibrate(ROW, case, ctrl, ref, oracle.FLOOR_OP)
    steps = nat.shape[0]
    a = nat.contiguous().view(torch.uint8).reshape(nat.numel(), -1)
    b = ctrl.contiguous().view(torch.uint8).reshape(ctrl.numel(), -1)
    per_step = (a != b).any(dim=-1).reshape(steps, -1).sum(dim=-1)
    differing_steps = torch.nonzero(per_step).reshape(-1)
    oracle.report_row(ROW, case, phase="report_only", c=c, measured=oracle.rel_l2(nat, ctrl), steps=steps,
                      elements=nat.numel(), differing=int(per_step.sum().item()),
                      first_differing_step=int(differing_steps[0].item()) if differing_steps.numel() else -1)


def _layer_cases(tc: unittest.TestCase, layers, run) -> None:
    """`run(w)` once per layer id, each in its own subTest (one failing layer does not hide the others)."""
    for layer in layers:
        with tc.subTest(layer=layer):
            run(_weights(layer))


# -- pieces -----------------------------------------------------------------------------------------------------


class GdnPieces(unittest.TestCase):
    """Each GDN piece vs its SGLang op on activations from real projections of each layer of LAYERS."""

    def _conv_prefill_case(self, w: types.SimpleNamespace, tokens: int, carried: bool, seed: int):
        """C3 (tasks/m22n-spec.md D3): native conv vs `causal_conv1d_fn` as forward_extend calls it, bit-exact."""
        case = f"layer{w.layer}/conv1d_prefill_L{tokens}_{'carried' if carried else 'fresh'}"
        mixed_qkv = F.linear(_hidden(seed, tokens), w.qkv)
        state0 = torch.zeros(CONV_DIM, CONV_K - 1, dtype=BF16, device=oracle.DEVICE)
        if carried:
            state0 = F.linear(_hidden(seed + 1, CONV_K - 1), w.qkv).t().contiguous()
        sg_state = state0.clone().unsqueeze(0)
        sg_y = _sglang_prefill_conv(w, mixed_qkv, sg_state, carried).contiguous()
        ref_y, _ = _ref_conv64(mixed_qkv, w.conv, state0)
        oracle.calibrate(ROW, f"{case}/y", sg_y, ref_y, oracle.FLOOR_OP)  # context only: the row is bit-exact
        nat_state = state0.clone()
        nat_y = _native_conv1d(w, mixed_qkv, nat_state)
        oracle.check_bit_exact(self, ROW, f"{case}/y", nat_y, sg_y)
        oracle.check_bit_exact(self, ROW, f"{case}/conv_state", nat_state, sg_state[0])
        _report_variant(case, "conv")

    def _conv_prefill_lengths(self, w: types.SimpleNamespace, carried: bool, seed0: int):
        for i, tokens in enumerate(CONV_PREFILL_LENGTHS):
            with self.subTest(L=tokens):
                self._conv_prefill_case(w, tokens, carried=carried, seed=seed0 + 2 * i)

    def test_conv1d_prefill_fresh(self):
        _layer_cases(self, LAYERS, lambda w: self._conv_prefill_lengths(w, False, 5121))

    def test_conv1d_prefill_carried_state(self):
        _layer_cases(self, LAYERS, lambda w: self._conv_prefill_lengths(w, True, 5141))

    def _conv_update_case(self, w: types.SimpleNamespace):
        case = f"layer{w.layer}/conv1d_update_{CONV_UPDATE_STEPS}x1"
        mixed_qkv = F.linear(_hidden(5105, CONV_UPDATE_STEPS), w.qkv)
        state0 = F.linear(_hidden(5106, CONV_K - 1), w.qkv).t().contiguous()
        sg_state = state0.clone().unsqueeze(0)
        nat_state = state0.clone()
        sg_steps, nat_steps = [], []
        for t in range(CONV_UPDATE_STEPS):
            x = mixed_qkv[t:t + 1].contiguous()
            sg_steps.append(_sglang_conv_update(w, x, sg_state))
            nat_steps.append(_native_conv1d(w, x, nat_state))
        sg_y, nat_y = torch.cat(sg_steps, 0), torch.cat(nat_steps, 0)
        ref_y, _ = _ref_conv64(mixed_qkv, w.conv, state0)
        oracle.calibrate(ROW, f"{case}/y", sg_y, ref_y, oracle.FLOOR_OP)  # context only: the row is bit-exact
        oracle.check_bit_exact(self, ROW, f"{case}/y", nat_y, sg_y)
        oracle.check_bit_exact(self, ROW, f"{case}/conv_state", nat_state, sg_state[0])
        _report_variant(case, "conv")

    def test_conv1d_update_sequential(self):
        """C3: CONV_UPDATE_STEPS single-token steps of Triton `causal_conv1d_update` (the control's decode conv) from
        a carried state; every step's y and the final state bit-exact."""
        _layer_cases(self, LAYERS, lambda w: self._conv_update_case(w))

    def _gating_case(self, w: types.SimpleNamespace, tokens: int, seed: int):
        """C4 (D4): native gating vs `fused_gdn_gating` (beta rounded to BF16), g and beta within 1 FP32 ulp."""
        case = f"layer{w.layer}/gating_prefill_T{tokens}"
        _, _, b, a = _sglang_proj(w, _hidden(seed, tokens))
        sg_g, sg_beta = fused_gdn_gating(w.A_log, a, b, w.dt_bias)
        sg_g, sg_beta = sg_g[0], sg_beta[0]  # (1, T, HV) -> (T, HV)
        ref_g, ref_beta = _ref_gating64(a, b, w.A_log, w.dt_bias)
        oracle.calibrate(ROW, f"{case}/g", sg_g, ref_g, oracle.FLOOR_OP)  # context only
        oracle.calibrate(ROW, f"{case}/beta", sg_beta, ref_beta, oracle.FLOOR_OP)  # context only
        nat_g, nat_beta = _native_gating(w, a, b)
        oracle.check_ulp(self, ROW, f"{case}/g", nat_g, sg_g, max_fraction=1e-3)
        oracle.check_ulp(self, ROW, f"{case}/beta", nat_beta, sg_beta, max_fraction=1e-3)
        _report_variant(case, "gating")

    def _gating_lengths(self, w: types.SimpleNamespace):
        for i, tokens in enumerate((64, 4096)):
            with self.subTest(T=tokens):
                self._gating_case(w, tokens, seed=5109 + 100 * i)

    def test_gating_prefill(self):
        _layer_cases(self, LAYERS, lambda w: self._gating_lengths(w))

    def _l2norm_case(self, w: types.SimpleNamespace, tokens: int, seed: int):
        """C4b (D6): native L2 norm vs FLA `l2norm_fwd` (BF16 out) on the q, k rows of a `tokens`-token prefill,
        bit-exact (a 1-FP32-ulp difference of a row's norm flips a BF16 output only every ~2^16 elements, so the
        4096-token case is the one that sees a reduction-order or lowering mismatch)."""
        case = f"layer{w.layer}/l2norm_qk_T{tokens}"
        mixed_qkv, _, _, _ = _sglang_proj(w, _hidden(seed, tokens))
        conv_out = _sglang_prefill_conv(w, mixed_qkv, _sglang_conv_state(), False)
        q, k, _ = fused_qkv_split_gdn_prefill(conv_out, HK, HK, HV, DK, DK, DV)
        x = torch.cat([q.reshape(-1, DK), k.reshape(-1, DK)], 0).contiguous()  # [2 * T * HK, DK]
        sg_y = l2norm_fwd(x, eps=L2_EPS)
        oracle.calibrate(ROW, case, sg_y, _ref_l2norm64(x), oracle.FLOOR_OP)  # context only
        nat_y = _native_l2norm(x)
        oracle.check_bit_exact(self, ROW, case, nat_y, sg_y)
        _report_variant(case, "l2norm")

    def _l2norm_lengths(self, w: types.SimpleNamespace):
        for i, tokens in enumerate((64, 4096)):
            with self.subTest(T=tokens):
                self._l2norm_case(w, tokens, seed=5111 + 100 * i)

    def test_l2norm(self):
        _layer_cases(self, LAYERS, lambda w: self._l2norm_lengths(w))

    def _gated_rmsnorm_case(self, w: types.SimpleNamespace):
        tokens = 64
        case = f"layer{w.layer}/gated_rmsnorm_T64"
        o, z = _sglang_prefill_core(w, _hidden(5113, tokens), _sglang_conv_state(), _sglang_ssm_state(), False)
        x = o.reshape(-1, DV).contiguous()  # [T * HV, DV], the delta-rule output as the mixer feeds it
        zr = z.reshape(-1, DV).contiguous()
        sg_y = _sglang_gated_norm(w, x, zr)
        c = oracle.calibrate(ROW, case, sg_y, _ref_gated_rmsnorm64(x, zr, w.norm), oracle.FLOOR_OP)
        nat_y = _native_gated_rmsnorm(w, x, zr)
        _check_op(self, case, nat_y, sg_y, c)

    def test_gated_rmsnorm(self):
        _layer_cases(self, LAYERS, lambda w: self._gated_rmsnorm_case(w))


# -- the mixer core on SGLang's projections (C7) ----------------------------------------------------------------


class GdnLayerCore(unittest.TestCase):
    """C7 (tasks/m22n-spec.md D7): conv -> gating -> delta rule -> gated RMSNorm, routed by operation, vs the served
    path of the no-speculation control on the same projections (port limits, D14), on each layer of LAYERS."""

    def _core_prefill_case(self, w: types.SimpleNamespace, tokens: int, seed: int):
        case = f"layer{w.layer}/layer_core_prefill_L{tokens}"
        mixed_qkv, z, b, a = _sglang_proj(w, _hidden(seed, tokens))
        sg_conv, sg_ssm = _sglang_conv_state(), _sglang_ssm_state()
        sg_y = _sglang_gated_norm(w, _sglang_extend_core(w, mixed_qkv, b, a, sg_conv, sg_ssm, False), z)
        sg_y = sg_y.reshape(tokens, VALUE_DIM)
        ref_y, _, ref_ssm = _ref_core64(w, mixed_qkv, z, b, a, sg_conv[0].new_zeros(CONV_DIM, CONV_K - 1),
                                        sg_ssm[0].new_zeros(HV, DV, DK))
        c_y = oracle.calibrate(ROW, f"{case}/y", sg_y, ref_y, oracle.FLOOR_BLOCK)  # context only
        c_ssm = oracle.calibrate(ROW, f"{case}/ssm_state", sg_ssm[0], ref_ssm, oracle.FLOOR_BLOCK)  # context only
        nat_conv = torch.zeros(CONV_DIM, CONV_K - 1, dtype=BF16, device=oracle.DEVICE)
        nat_ssm = torch.zeros(HV, DV, DK, dtype=BF16, device=oracle.DEVICE)
        nat_y = _native_layer_core(w, mixed_qkv, z, b, a, nat_conv, nat_ssm, OP_PREFILL)
        limit = oracle.port_limit(5e-4, "C7/D14: the extend path (Triton conv, fused_gdn_gating, FLA chunk-64 with "
                                        "in-kernel L2 norm) is the control's prefill; conv and gated norm exact, "
                                        "the delta rule within C6's multi-chunk limit")
        oracle.check_within(self, ROW, f"{case}/y", oracle.rel_l2(nat_y, sg_y), c=c_y, limit=limit)
        oracle.check_within(self, ROW, f"{case}/ssm_state", oracle.rel_l2(nat_ssm, sg_ssm[0]), c=c_ssm, limit=limit)
        oracle.check_bit_exact(self, ROW, f"{case}/conv_state", nat_conv, sg_conv[0])

    def _core_prefill_lengths(self, w: types.SimpleNamespace):
        for i, tokens in enumerate(LAYER_CORE_PREFILL_LENGTHS):
            with self.subTest(L=tokens):
                self._core_prefill_case(w, tokens, seed=5401 + 2 * i)

    def test_core_prefill(self):
        """PREFILL of every C7 length, fresh state; L=1 runs the chunk path, as the control's extend does."""
        _layer_cases(self, LAYERS, lambda w: self._core_prefill_lengths(w))

    def _core_chain_case(self, w: types.SimpleNamespace):
        case = f"layer{w.layer}/layer_core_chain_L{CHAIN_PREFILL}_decode{CHAIN_STEPS}"
        tokens = CHAIN_PREFILL + CHAIN_STEPS
        mixed_qkv, z, b, a = _sglang_proj(w, _hidden(5431, tokens))
        sg_conv, sg_ssm = _sglang_conv_state(), _sglang_ssm_state()
        sg_o = [_sglang_extend_core(w, mixed_qkv[:CHAIN_PREFILL], b[:CHAIN_PREFILL], a[:CHAIN_PREFILL], sg_conv,
                                    sg_ssm, False)[0]]
        for t in range(CHAIN_PREFILL, tokens):
            sg_o.append(_flashinfer_decode_core(w, mixed_qkv[t:t + 1], b[t:t + 1], a[t:t + 1], sg_conv, sg_ssm))
        sg_y = _sglang_gated_norm(w, torch.cat(sg_o, 0), z).reshape(tokens, VALUE_DIM)[CHAIN_PREFILL:]
        ref_y, _, ref_ssm = _ref_core64(w, mixed_qkv, z, b, a, sg_conv[0].new_zeros(CONV_DIM, CONV_K - 1),
                                        sg_ssm[0].new_zeros(HV, DV, DK))
        c_y = oracle.calibrate(ROW, f"{case}/y_decode", sg_y, ref_y[CHAIN_PREFILL:], oracle.FLOOR_BLOCK)
        c_ssm = oracle.calibrate(ROW, f"{case}/ssm_state", sg_ssm[0], ref_ssm, oracle.FLOOR_BLOCK)
        nat_conv = torch.zeros(CONV_DIM, CONV_K - 1, dtype=BF16, device=oracle.DEVICE)
        nat_ssm = torch.zeros(HV, DV, DK, dtype=BF16, device=oracle.DEVICE)
        _native_layer_core(w, mixed_qkv[:CHAIN_PREFILL], z[:CHAIN_PREFILL], b[:CHAIN_PREFILL], a[:CHAIN_PREFILL],
                           nat_conv, nat_ssm, OP_PREFILL)
        nat_y = torch.cat([_native_layer_core(w, mixed_qkv[t:t + 1], z[t:t + 1], b[t:t + 1], a[t:t + 1], nat_conv,
                                              nat_ssm, OP_DECODE) for t in range(CHAIN_PREFILL, tokens)], 0)
        limit = oracle.port_limit(1e-3, "C7/D14: FLA extend then FlashInfer pool-mode decode is the control's "
                                        "served path; one-chunk prefill state carried through 64 decode steps")
        oracle.check_within(self, ROW, f"{case}/y_decode", oracle.rel_l2(nat_y, sg_y), c=c_y, limit=limit)
        oracle.check_within(self, ROW, f"{case}/ssm_state", oracle.rel_l2(nat_ssm, sg_ssm[0]), c=c_ssm, limit=limit)
        oracle.check_bit_exact(self, ROW, f"{case}/conv_state", nat_conv, sg_conv[0])

    def test_core_chain_prefill_then_decode(self):
        """A CHAIN_PREFILL-token PREFILL (FLA extend) then CHAIN_STEPS single-token DECODE steps (FlashInfer pool
        mode), native and reference each carrying their own states; the decode steps' y and the final states."""
        _layer_cases(self, LAYERS, lambda w: self._core_chain_case(w))

    def _chain_core_case(self, w: types.SimpleNamespace):
        case = f"layer{w.layer}/chain_core_L{CHAIN_PREFILL}"
        tokens = CHAIN_PREFILL + sum(CHAIN_CORE_ROWS)
        mixed_qkv, z, b, a = _sglang_proj(w, _hidden(5443, tokens))
        conv = torch.zeros(CONV_DIM, CONV_K - 1, dtype=BF16, device=oracle.DEVICE)
        ssm = torch.zeros(HV, DV, DK, dtype=BF16, device=oracle.DEVICE)
        _native_layer_core(w, mixed_qkv[:CHAIN_PREFILL], z[:CHAIN_PREFILL], b[:CHAIN_PREFILL], a[:CHAIN_PREFILL],
                           conv, ssm, OP_PREFILL)
        ref_conv, ref_ssm = conv.clone(), ssm.clone()
        t0 = CHAIN_PREFILL
        for i, rows in enumerate(CHAIN_CORE_ROWS):
            chain = f"{case}/chain{i}_R{rows}"
            ref_y, ref_saves = [], []
            for t in range(t0, t0 + rows):
                ref_y.append(_native_layer_core(w, mixed_qkv[t:t + 1], z[t:t + 1], b[t:t + 1], a[t:t + 1], ref_conv,
                                                ref_ssm, OP_DECODE))
                if t + 1 < t0 + rows:
                    ref_saves.append((ref_conv.clone(), ref_ssm.clone()))
            rs = slice(t0, t0 + rows)
            y, conv_saves, ssm_saves = _native_chain_core(w, mixed_qkv[rs], z[rs], b[rs], a[rs], conv, ssm)
            oracle.check_bit_exact(self, ROW, f"{chain}/y", y, torch.cat(ref_y, 0))
            oracle.check_bit_exact(self, ROW, f"{chain}/conv_state", conv, ref_conv)
            oracle.check_bit_exact(self, ROW, f"{chain}/ssm_state", ssm, ref_ssm)
            for s, (save_conv, save_ssm) in enumerate(ref_saves):
                oracle.check_bit_exact(self, ROW, f"{chain}/save{s}/conv_state", conv_saves[s], save_conv)
                oracle.check_bit_exact(self, ROW, f"{chain}/save{s}/ssm_state", ssm_saves[s], save_ssm)
            t0 += rows

    def test_chain_core_equals_per_row(self):
        """DG-02: chain_core over consecutive verify chains of 1..9 rows equals layer_core DECODE once per row, in
        row order, byte for byte (y, the final states, the state after every row but the last)."""
        _layer_cases(self, LAYERS, lambda w: self._chain_core_case(w))

    def _chains_core_case(self, w: types.SimpleNamespace):
        for layout in CHAINS_CORE_LAYOUTS:
            chains, rows = len(layout), sum(layout)
            case = f"layer{w.layer}/chains_core_{'-'.join(map(str, layout))}"
            # Chain c's sequence: a CHAIN_PREFILL-token PREFILL of its own window, then its rows.
            mixed_qkv, z, b, a = _sglang_proj(w, _hidden(5449, CHAIN_PREFILL + chains + rows))
            conv = torch.zeros(chains, CONV_DIM, CONV_K - 1, dtype=BF16, device=oracle.DEVICE)
            ssm = torch.zeros(chains, HV, DV, DK, dtype=BF16, device=oracle.DEVICE)
            for c in range(chains):
                ps = slice(c, c + CHAIN_PREFILL)
                _native_layer_core(w, mixed_qkv[ps], z[ps], b[ps], a[ps], conv[c], ssm[c], OP_PREFILL)
            rs = slice(CHAIN_PREFILL + chains, CHAIN_PREFILL + chains + rows)
            mixed_qkv, z, b, a = (x[rs].contiguous() for x in (mixed_qkv, z, b, a))
            ref_conv, ref_ssm = conv.clone(), ssm.clone()
            ref_y, ref_saves = [], {}
            r = 0
            for c, count in enumerate(layout):
                for t in range(count):
                    ref_y.append(_native_layer_core(w, mixed_qkv[r:r + 1], z[r:r + 1], b[r:r + 1], a[r:r + 1],
                                                    ref_conv[c], ref_ssm[c], OP_DECODE))
                    if t + 1 < count:
                        ref_saves[r] = (ref_conv[c].clone(), ref_ssm[c].clone())
                    r += 1
            y = oracle.empty((rows, VALUE_DIM), BF16)
            conv_saves = oracle.empty((rows, CONV_DIM, CONV_K - 1), BF16)
            ssm_saves = oracle.empty((rows, HV, DV, DK), BF16)
            oracle.call("cere_ktest_gdn_chains_core", mixed_qkv, z, b, a, w.conv, w.A_log, w.dt_bias, w.norm, conv,
                        ssm, conv_saves, ssm_saves, y, rows, torch.tensor(layout, dtype=torch.long), chains)
            oracle.check_bit_exact(self, ROW, f"{case}/y", y, torch.cat(ref_y, 0))
            oracle.check_bit_exact(self, ROW, f"{case}/conv_states", conv, ref_conv)
            oracle.check_bit_exact(self, ROW, f"{case}/ssm_states", ssm, ref_ssm)
            for s, (save_conv, save_ssm) in ref_saves.items():
                oracle.check_bit_exact(self, ROW, f"{case}/save{s}/conv_state", conv_saves[s], save_conv)
                oracle.check_bit_exact(self, ROW, f"{case}/save{s}/ssm_state", ssm_saves[s], save_ssm)

    def test_chains_core_equals_per_row(self):
        """R16-B (tasks/rows16-spec.md): one chain_core call over several sequences' chains (each on its own state,
        CHAINS_CORE_LAYOUTS) equals layer_core DECODE once per row, per chain in row order, byte for byte (y, every
        chain's final states, the state after every row but its chain's last)."""
        _layer_cases(self, LAYERS[:2], lambda w: self._chains_core_case(w))

    def test_chain_core_arguments(self):
        """DG-02: rows outside 1..16, missing saves and saves aliasing the states are CERE_ERR_ARGUMENT before any
        launch (the states keep their bytes); R16-B: chains sharing a state, chain rows not summing to rows."""
        w = _weights(LAYERS[0])
        mixed_qkv, z, b, a = _sglang_proj(w, _hidden(5447, 17))
        conv = torch.zeros(CONV_DIM, CONV_K - 1, dtype=BF16, device=oracle.DEVICE)
        ssm = torch.zeros(HV, DV, DK, dtype=BF16, device=oracle.DEVICE)
        y = oracle.empty((17, VALUE_DIM), BF16)
        conv_saves = oracle.empty((16, CONV_DIM, CONV_K - 1), BF16)
        ssm_saves = oracle.empty((16, HV, DV, DK), BF16)
        for rows, saves in ((0, (conv_saves, ssm_saves)), (17, (conv_saves, ssm_saves)), (2, (None, None)),
                            (2, (conv, ssm))):
            with self.subTest(rows=rows, saves="none" if saves[0] is None else "state" if saves[0] is conv else "own"):
                status = oracle.call_status("cere_ktest_gdn_chain_core", mixed_qkv, z, b, a, w.conv, w.A_log,
                                            w.dt_bias, w.norm, conv, ssm, *saves, y, rows)
                self.assertEqual(status, oracle.CERE_ERR_ARGUMENT)
        # A save aliasing another chain's state (row 0's save is chain 1's state), and chain rows short of `rows`.
        convs = torch.zeros(2, CONV_DIM, CONV_K - 1, dtype=BF16, device=oracle.DEVICE)
        ssms = torch.zeros(2, HV, DV, DK, dtype=BF16, device=oracle.DEVICE)
        for case, saves, layout, rows in (("save_is_other_state", (convs[1], ssms[1]), (2, 1), 3),
                                          ("rows_short", (conv_saves, ssm_saves), (2,), 3)):
            with self.subTest(case=case):
                status = oracle.call_status("cere_ktest_gdn_chains_core", mixed_qkv, z, b, a, w.conv, w.A_log,
                                            w.dt_bias, w.norm, convs, ssms, *saves, y, rows,
                                            torch.tensor(layout, dtype=torch.long), len(layout))
                self.assertEqual(status, oracle.CERE_ERR_ARGUMENT)
        oracle.check_bit_exact(self, ROW, "chain_core_arguments/conv_state", conv, torch.zeros_like(conv))
        oracle.check_bit_exact(self, ROW, "chain_core_arguments/ssm_state", ssm, torch.zeros_like(ssm))
        oracle.check_bit_exact(self, ROW, "chain_core_arguments/conv_states", convs, torch.zeros_like(convs))
        oracle.check_bit_exact(self, ROW, "chain_core_arguments/ssm_states", ssms, torch.zeros_like(ssms))


# -- the whole mixer --------------------------------------------------------------------------------------------


class GdnLayer(unittest.TestCase):
    """The linear-attention mixer of each layer of LAYERS: prefill vs the extend path, decode vs FlashInfer
    (target-verify reported), and the D5 whole-layer chain (native PREFILL then DECODE vs extend then decode)."""

    def _prefill_case(self, w: types.SimpleNamespace, tokens: int, seed: int):
        case = f"layer{w.layer}/layer_prefill_L{tokens}"
        hidden = _hidden(seed, tokens)
        sg_conv, sg_ssm = _sglang_conv_state(), _sglang_ssm_state()
        sg_out = _sglang_prefill(w, hidden, sg_conv, sg_ssm, False)
        ref_out, ref_conv, ref_ssm = _ref_layer64(w, hidden, sg_conv[0].new_zeros(CONV_DIM, CONV_K - 1),
                                                  sg_ssm[0].new_zeros(HV, DV, DK))
        c_out = oracle.calibrate(ROW, f"{case}/out", sg_out, ref_out, oracle.FLOOR_BLOCK)
        c_ssm = oracle.calibrate(ROW, f"{case}/ssm_state", sg_ssm[0], ref_ssm, oracle.FLOOR_BLOCK)
        c_conv = oracle.calibrate(ROW, f"{case}/conv_state", sg_conv[0], ref_conv, oracle.FLOOR_BLOCK)
        nat_conv = torch.zeros(CONV_DIM, CONV_K - 1, dtype=BF16, device=oracle.DEVICE)
        nat_ssm = torch.zeros(HV, DV, DK, dtype=BF16, device=oracle.DEVICE)
        nat_out = _native_layer_prefill(w, hidden, nat_conv, nat_ssm)
        _check_block(self, f"{case}/out", nat_out, sg_out, c_out)
        _check_block(self, f"{case}/ssm_state", nat_ssm, sg_ssm[0], c_ssm)
        _check_block(self, f"{case}/conv_state", nat_conv, sg_conv[0], c_conv)

    def test_prefill_L1(self):
        _layer_cases(self, LAYERS, lambda w: self._prefill_case(w, 1, 5201))

    def test_prefill_L64(self):
        _layer_cases(self, LAYERS, lambda w: self._prefill_case(w, 64, 5203))

    def test_prefill_L2048(self):
        _layer_cases(self, LAYERS, lambda w: self._prefill_case(w, 2048, 5205))

    def _after_prefill(self, w: types.SimpleNamespace, seed: int):
        """SGLang's BF16 pools after a 64-token prefill (the common starting state of every decode path)."""
        conv_state, ssm_state = _sglang_conv_state(), _sglang_ssm_state()
        _sglang_prefill(w, _hidden(seed, PREFILL_BEFORE_DECODE), conv_state, ssm_state, False)
        return conv_state, ssm_state

    def _decode_case(self, w: types.SimpleNamespace, case: str, seed: int, step_tokens: int, reference_step, *,
                     report_only: bool = False):
        case = f"layer{w.layer}/{case}"
        conv0, ssm0 = self._after_prefill(w, seed)
        hidden = _hidden(seed + 1, DECODE_TOKENS)
        sg_conv, sg_ssm = conv0.clone(), ssm0.clone()
        sg_out = torch.cat([reference_step(w, hidden[t:t + step_tokens], sg_conv, sg_ssm)
                            for t in range(0, DECODE_TOKENS, step_tokens)], 0)
        ref_out, ref_conv, ref_ssm = _ref_layer64(w, hidden, conv0[0], ssm0[0])
        c_out = oracle.calibrate(ROW, f"{case}/out", sg_out, ref_out, oracle.FLOOR_BLOCK)
        c_ssm = oracle.calibrate(ROW, f"{case}/ssm_state", sg_ssm[0], ref_ssm, oracle.FLOOR_BLOCK)
        c_conv = oracle.calibrate(ROW, f"{case}/conv_state", sg_conv[0], ref_conv, oracle.FLOOR_BLOCK)
        nat_conv, nat_ssm = conv0[0].clone(), ssm0[0].clone()
        nat_out = torch.cat([_native_layer_decode(w, hidden[t:t + 1].contiguous(), nat_conv, nat_ssm)
                             for t in range(DECODE_TOKENS)], 0)
        for name, nat, sg, c in ((f"{case}/out", nat_out, sg_out, c_out),
                                 (f"{case}/ssm_state", nat_ssm, sg_ssm[0], c_ssm),
                                 (f"{case}/conv_state", nat_conv, sg_conv[0], c_conv)):
            if report_only:
                _report_block(name, nat, sg, c)
            else:
                _check_block(self, name, nat, sg, c)

    def test_decode_vs_flashinfer(self):
        """Primary decode oracle (D15): 64 single-token FlashInfer GDN decode steps after a 64-token prefill."""
        _layer_cases(self, LAYERS, lambda w: self._decode_case(w, "layer_decode_flashinfer_64", 5303, 1,
                                                                _flashinfer_decode_step))

    def test_decode_vs_target_verify(self):
        """Report-only (D15): 16 TARGET_VERIFY steps of 4 tokens after a 64-token prefill (the bench's verify kernel,
        not a port target)."""
        _layer_cases(self, LAYERS, lambda w: self._decode_case(w, "layer_decode_verify_4x16", 5301, DRAFT,
                                                                _sglang_verify_step, report_only=True))

    def _chain_case(self, w: types.SimpleNamespace, prefill: int, seed: int):
        """D5 (tasks/m22d-spec.md): native `layer_forward` PREFILL of `prefill` tokens (native projections), then
        CHAIN_STEPS native DECODE steps, against the control's extend then CHAIN_STEPS FlashInfer decode steps, each
        side (and the FP64 reference) carrying its own state. Per step: the step's out row and the ssm state after
        it, each calibrated against the FP64 reference and checked as the GdnLayer rows (max(FLOOR_BLOCK, 2c))."""
        case = f"layer{w.layer}/layer_chain_P{prefill}_decode{CHAIN_STEPS}"
        hidden = _hidden(seed, prefill + CHAIN_STEPS)
        sg_conv, sg_ssm = _sglang_conv_state(), _sglang_ssm_state()
        _sglang_prefill(w, hidden[:prefill], sg_conv, sg_ssm, False)
        _, ref_conv, ref_ssm = _ref_layer64(w, hidden[:prefill], sg_conv[0].new_zeros(CONV_DIM, CONV_K - 1),
                                            sg_ssm[0].new_zeros(HV, DV, DK))
        nat_conv = torch.zeros(CONV_DIM, CONV_K - 1, dtype=BF16, device=oracle.DEVICE)
        nat_ssm = torch.zeros(HV, DV, DK, dtype=BF16, device=oracle.DEVICE)
        _native_layer_prefill(w, hidden[:prefill].contiguous(), nat_conv, nat_ssm)
        for step in range(CHAIN_STEPS):
            x = hidden[prefill + step:prefill + step + 1].contiguous()
            sg_out = _flashinfer_decode_step(w, x, sg_conv, sg_ssm)
            ref_out, ref_conv, ref_ssm = _ref_layer64(w, x, ref_conv, ref_ssm)
            nat_out = _native_layer_decode(w, x, nat_conv, nat_ssm)
            with self.subTest(step=step):
                name = f"{case}/step{step}"
                c_out = oracle.calibrate(ROW, f"{name}/out", sg_out, ref_out, oracle.FLOOR_BLOCK)
                c_ssm = oracle.calibrate(ROW, f"{name}/ssm_state", sg_ssm[0], ref_ssm, oracle.FLOOR_BLOCK)
                _check_block(self, f"{name}/out", nat_out, sg_out, c_out)
                _check_block(self, f"{name}/ssm_state", nat_ssm, sg_ssm[0], c_ssm)

    def _chain_prefills(self, w: types.SimpleNamespace):
        for i, prefill in enumerate(LAYER_CHAIN_PREFILLS):
            with self.subTest(P=prefill):
                self._chain_case(w, prefill, seed=5441 + 2 * i)

    def test_chain_prefill_then_decode(self):
        """D5: the whole-layer chain on every layer of LAYERS and every P of LAYER_CHAIN_PREFILLS."""
        _layer_cases(self, LAYERS, lambda w: self._chain_prefills(w))


# -- the GDN block's decode ops one by one (m22d design note C7, D6 isolation) ---------------------------------


class GdnDecodeOps(unittest.TestCase):
    """C7 (tasks/m22d-spec.md D6): every GDN-block op of a native DECODE step (M=1) against the kernel the control
    dispatches for it, fed the control's inputs of the step, on every `linear_attention` layer. Report-only."""

    def _ops_case(self, w: types.SimpleNamespace):
        """The control runs a fresh ISO_PREFILL-token extend, then ISO_STEPS FlashInfer decode steps; at each step
        every native op gets exactly the control's inputs of that step (its hidden row, its projections, its conv
        and ssm state before the step, its delta-rule output, its gated-norm output)."""
        case = f"layer{w.layer}/decode_op"
        w64 = _weights64(w.layer)
        hidden = _hidden(5461, ISO_PREFILL + ISO_STEPS)
        sg_conv, sg_ssm = _sglang_conv_state(), _sglang_ssm_state()
        _sglang_prefill(w, hidden[:ISO_PREFILL], sg_conv, sg_ssm, False)
        ops = ("in_proj_qkv_M1", "in_proj_z_M1", "in_proj_b_M1", "in_proj_a_M1", "conv1d_update/y",
               "conv1d_update/conv_state", "delta_rule_decode/o", "delta_rule_decode/ssm_state", "gated_rmsnorm_M1",
               "out_proj_M1")
        rows = {op: ([], [], []) for op in ops}  # op -> (native, control, FP64) per step

        def add(op, native, control, reference):
            for dst, t in zip(rows[op], (native, control, reference)):
                dst.append(t)

        for step in range(ISO_STEPS):
            x = hidden[ISO_PREFILL + step:ISO_PREFILL + step + 1].contiguous()
            x64 = x.double()
            mixed_qkv, z, b, a = _sglang_proj(w, x)
            for op, ctrl, weight, ref_weight in (("in_proj_qkv_M1", mixed_qkv, w.qkv, w64.qkv),
                                                 ("in_proj_z_M1", z, w.z, w64.z), ("in_proj_b_M1", b, w.b, w64.b),
                                                 ("in_proj_a_M1", a, w.a, w64.a)):
                add(op, _native_gemm(x, weight), ctrl, F.linear(x64, ref_weight))
            conv0 = sg_conv[0].clone()
            nat_conv = conv0.clone()
            sg_mixed = _sglang_conv_update(w, mixed_qkv, sg_conv)
            ref_mixed, ref_conv = _ref_conv64(mixed_qkv, w.conv, conv0)
            add("conv1d_update/y", _native_conv1d(w, mixed_qkv, nat_conv), sg_mixed, ref_mixed)
            add("conv1d_update/conv_state", nat_conv, sg_conv[0].clone(), ref_conv)
            ssm0 = sg_ssm[0].clone()
            nat_ssm = ssm0.clone()
            sg_o = _flashinfer_delta(w, sg_mixed, b, a, sg_ssm)
            q, k, v = sg_mixed.split([KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
            ref_o, ref_ssm = _ref_delta64(w64, q, k, v, b, a, ssm0)
            add("delta_rule_decode/o", _native_decode_delta(w, sg_mixed, a, b, nat_ssm), sg_o, ref_o)
            add("delta_rule_decode/ssm_state", nat_ssm, sg_ssm[0].clone(), ref_ssm)
            o_rows, z_rows = sg_o.reshape(-1, DV).contiguous(), z.reshape(-1, DV).contiguous()
            sg_y = _sglang_gated_norm(w, o_rows, z_rows)
            add("gated_rmsnorm_M1", _native_gated_rmsnorm(w, o_rows, z_rows), sg_y,
                _ref_gated_rmsnorm64(o_rows, z_rows, w.norm))
            y = sg_y.reshape(1, VALUE_DIM).contiguous()
            add("out_proj_M1", _native_gemm(y, w.out), F.linear(y, w.out), F.linear(y.double(), w64.out))
        for op in ops:
            _report_op_steps(f"{case}/{op}", *rows[op])

    def test_decode_ops_every_gdn_layer(self):
        _layer_cases(self, oracle.linear_attention_layers(), lambda w: self._ops_case(w))


if __name__ == "__main__":
    unittest.main()
