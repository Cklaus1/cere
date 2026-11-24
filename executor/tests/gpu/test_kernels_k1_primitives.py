"""K1 (tasks/m22-spec.md): the primitive kernels of the A4 model vs the SGLang ops the reference server runs.

Every case reports row "K1". Tolerance cases follow the tolerance rule: `c` = rel-L2(SGLang op vs an FP64 torch
reference on the same inputs), written to the report before the native call; limit = max(4e-3, 2c); measured =
rel-L2(native vs SGLang op). Embedding, top-2 argmax and the FP8 KV store are bit-exact. Every native case runs twice
and the two results must be bit-identical.

Cases and their SGLang references ($S181 = the reference image's site-packages):
- embedding gather on the real `model.language_model.embed_tokens.weight` vs `UnquantizedEmbeddingMethod.embedding`
  (`F.embedding`, `sglang/srt/layers/quantization/unquant.py:299`), bit-exact;
- RMSNorm in the three weight conventions the model uses (eps = `rms_norm_eps`):
  * grouped Gemma `(1 + w)` with one variance per 2,560-wide branch over the 4 hyper-connection streams
    (`GroupedGemmaRMSNorm`, `sglang/srt/layers/hyperconnection.py:21`, built as in `GatedResidual` `:127-137` with
    `hc_per_branch_norm=True`, `qwen4_exp.py:1263`), real `attn_hyper_connection.hc_norm` (layer 0) and
    `hyper_connection_mixer.hc_norm` weights;
  * QK norm: per-head Gemma `(1 + w)` over head_dim 256 (`GemmaRMSNorm`, `sglang/srt/layers/layernorm.py:1004`, used
    as `q_norm`/`k_norm` by `qwen3_5.py:1038-1039,1085-1088`), real layer-3 `self_attn.q_norm`/`k_norm`;
  * plain `w` RMSNorm (`RMSNorm`, `layernorm.py:423`) over the GDN head_v_dim 128 with the real layer-0
    `linear_attn.norm.weight` (the ungated form of the GDN norm);
  * gated: the GDN output norm `RMSNormGated(head_v_dim, eps, group_size=None, norm_before_gate=True,
    activation=output_gate_type)` (`qwen3_5.py:366-378`, `sglang/kernels/ops/attention/fla/layernorm_gated.py:433`):
    y = x * rstd * w * sigmoid(z) (`layernorm_gated.py:176-177`, `output_gate_type` is "sigmoid");
- the rotary table (FP32 cos/sin rows) and partial interleaved mRoPE: 64 of 256 dims, `rope_theta` 1e7,
  `mrope_section` [11, 11, 10], `mrope_interleaved` true, NeoX rotate-half, vs `MRotaryEmbedding`
  (`sglang/srt/layers/rotary_embedding/mrope.py:54`, constructed with the arguments `get_rope` derives from the real
  `rope_parameters`, `factory.py:95-210`; `forward` -> `forward_triton`, `mrope.py:253`). Text positions are the
  same on all three mRoPE axes (SGLang receives positions [3, T] with equal rows), so the interleaved axis selection
  is the identity and the native kernel takes one int64 position per token;
- the attention output gate `x * sigmoid(gate)` vs `fused_sigmoid_mul` (`sglang/kernels/ops/elementwise/
  elementwise.py:409`, used by `qwen4_exp.py:1540`);
- device top-2 argmax over the vocabulary vs SGLang's greedy `torch.argmax(logits, -1)` (`sampler.py:137`) on the
  BF16 logits converted to FP32 (`logits_processor.py:741-743,831`): id 0 = first maximum, id 1 = first maximum of
  the row with id 0 excluded; a tie case asserts the first index wins; bit-exact;
- the FP8 E4M3 KV store with scale 1.0 vs SGLang's write: `cache_k.div_(k_scale)`, `.to(float8_e4m3fn)`, viewed as
  uint8 (`sglang/srt/mem_cache/memory_pool.py:2474-2486`), scattered by `_set_kv_buffer_impl` (`memory_pool.py:141`);
  bit-exact over the whole cache, including inputs outside the E4M3 range (the native cast must reproduce torch's
  conversion bit for bit).
- D6 isolation rows `isolate_L<layer>_{q,k}_norm_rope_pos<p>_n1` (tasks/m22d-spec.md D6/C7, phase `isolate`): QK
  norm + partial mRoPE of one decode token (positions 21, 59, 95, 159: the parity regime and one past the first
  128-key split) on every full-attention layer with its real `q_norm`/`k_norm`, vs the control's dispatched prep
  kernel `fused_qk_gemma_rmsnorm_rope_gate` (called as in the K6 suite's SGLang prep) on the same q|gate, k and
  SGLang `cos_sin_cache`.
  The native side is `cere_ktest_rmsnorm` then `cere_ktest_rope` (the K6 block runs both inside
  `k1::qk_norm_rope_gate`, which has no ktest; the K6 isolation rows measure that kernel inside the block).
  Report-only (nothing is asserted beyond running): `c` = rel-L2(control, FP64), `limit` = max(FLOOR_OP, 2c),
  `measured` = rel-L2(native, control), `within`, and the differing elements.

Native entry points (oracle.py calling convention: device pointers are `void *`, integers `int64_t`, `float`/`double`
scalars as stated, the stream last; 0 = success). All tensors are contiguous and row-major.

    int cere_ktest_embedding(const void *table, const void *ids, int64_t n_ids, int64_t vocab, int64_t hidden,
                             void *out, cudaStream_t stream);
        table bf16 [vocab, hidden] (checkpoint layout); ids int32 [n_ids]; out bf16 [n_ids, hidden]:
        out[i] = table[ids[i]].

    int cere_ktest_rmsnorm(const void *x, const void *weight, int64_t n_rows, int64_t row_dim, int64_t group_size,
                           float eps, float weight_offset, void *out, cudaStream_t stream);
        x bf16 [n_rows, row_dim]; weight bf16 [row_dim]; out bf16 [n_rows, row_dim]. row_dim % group_size == 0; each
        group of `group_size` consecutive elements of a row has its own variance:
        out = bf16(x * rsqrt(mean_group(x^2) + eps) * (weight_offset + w)), FP32 math.
        weight_offset 1.0 = Gemma (1 + w) (hc_norm, q_norm, k_norm); 0.0 = plain w.

    int cere_ktest_rmsnorm_gated(const void *x, const void *z, const void *weight, int64_t n_rows, int64_t row_dim,
                                 float eps, void *out, cudaStream_t stream);
        x, z bf16 [n_rows, row_dim]; weight bf16 [row_dim]; out bf16 [n_rows, row_dim]:
        out = bf16(x * rsqrt(mean(x^2) + eps) * w * sigmoid(z)), FP32 math (norm before gate, sigmoid gate).

    int cere_ktest_rope_table(void *cos_sin, int64_t n_pos, int64_t rotary_dim, double rope_theta,
                              cudaStream_t stream);
        cos_sin f32 [n_pos, rotary_dim]; row p = [cos(p * f_0) .. cos(p * f_{rd/2-1}) | sin(p * f_0) ..
        sin(p * f_{rd/2-1})] with f_i = rope_theta^(-2i / rotary_dim) (the layout of SGLang's `cos_sin_cache`).

    int cere_ktest_rope(const void *q, const void *k, const void *cos_sin, const void *positions, int64_t n_tokens,
                        int64_t n_q_heads, int64_t n_kv_heads, int64_t head_dim, int64_t rotary_dim, void *q_out,
                        void *k_out, cudaStream_t stream);
        q bf16 [n_tokens, n_q_heads * head_dim]; k bf16 [n_tokens, n_kv_heads * head_dim]; cos_sin f32 table as
        above; positions int64 [n_tokens]; q_out/k_out like q/k. Per head, NeoX rotate-half on dims [0, rotary_dim)
        with x1 = dims [0, rd/2), x2 = dims [rd/2, rd): (x1 cos - x2 sin, x2 cos + x1 sin); dims [rd, head_dim) copied.

    int cere_ktest_sigmoid_gate(const void *x, const void *gate, int64_t n_rows, int64_t row_dim, void *out,
                                cudaStream_t stream);
        x, gate, out bf16 [n_rows, row_dim]: out = bf16(x * sigmoid(gate)), FP32 math.

    int cere_ktest_top2(const void *logits, int64_t n_rows, int64_t vocab, void *ids, void *scores,
                        cudaStream_t stream);
        logits bf16 [n_rows, vocab]; ids int32 [n_rows, 2]; scores f32 [n_rows, 2] (the logit values at ids).
        ids[r][0] = first index of the row maximum; ids[r][1] = first index of the maximum over the row without
        ids[r][0] (equals ids[r][0]'s value on ties, at the next index).

    int cere_ktest_kv_store_fp8(const void *k, const void *v, const void *slots, int64_t n_tokens, int64_t row_dim,
                                void *k_cache, void *v_cache, int64_t n_slots, cudaStream_t stream);
        k, v bf16 [n_tokens, row_dim] (row_dim = n_kv_heads * head_dim); slots int64 [n_tokens] in [0, n_slots);
        k_cache, v_cache uint8 (E4M3 bytes) [n_slots, row_dim]: cache[slots[t]] = e4m3(x[t] / 1.0), torch's
        float -> float8_e4m3fn conversion; no other cache byte changes.

    int cere_ktest_copy_segments(const void *ranges, int64_t count, cudaStream_t stream);
        ranges host int64 [count, 6]: (dst, dst_pitch, src, src_pitch, width, rows) per segment, device addresses as
        integers, 1 <= count <= 128 (else CERE_ERR_ARGUMENT): k1::copy_segments, rows rows of width bytes each, the
        bytes of one cudaMemcpy2D D2D per segment (tasks/m23-spec.md P11R-20: the verify graph's undo copies; DG-03:
        the verify_commit restore, up to 98 segments in one launch).
"""
from __future__ import annotations

import contextlib
import unittest

import torch

import oracle
from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import fused_qk_gemma_rmsnorm_rope_gate
from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.srt.layers.hyperconnection import GroupedGemmaRMSNorm
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_impl
from sglang.srt.runtime_context import get_context

ROW = "K1"
# Token counts: one decode token, a small odd batch, a full 2,048-token prefill chunk (M2.2b context bound).
TOKEN_COUNTS = (1, 7, 2048)
# Longest native context (m25 design note D6, C5; M2.2c: 73,728): the rope table and positions cover [0, 200,000),
# the mRoPE decode and far-chunk cases end at 199,999.
MAX_CONTEXT = 200_000
ATTN_LAYER = 3
GDN_LAYER = 0
KV_SLOTS = 4096
# D6 isolation (m22d design note C7): decode positions of the parity regime (P 21-32 plus up to 64 decode steps)
# and one past the first 128-key split.
ISOLATE_POSITIONS = (21, 59, 95, 159)


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


def _bf16(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.dtype != torch.bfloat16:
        raise oracle.OracleError(f"{name} is {t.dtype}, expected torch.bfloat16")
    return t.contiguous()


def _randn(shape, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = oracle.generator(seed)
    return (torch.randn(shape, generator=g, device=oracle.DEVICE, dtype=torch.float32) * scale).to(torch.bfloat16)


# -- FP64 references (calibration) ------------------------------------------------------------------------------


def _rmsnorm_fp64(x: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float, offset: float) -> torch.Tensor:
    x64 = x.to(torch.float64)
    rows, row_dim = x64.shape
    grouped = x64.reshape(rows, row_dim // group_size, group_size)
    rstd = torch.rsqrt(grouped.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (grouped * rstd).reshape(rows, row_dim) * (weight.to(torch.float64) + offset)


def _inv_freq_fp64(rotary_dim: int, theta: float) -> torch.Tensor:
    exponent = torch.arange(0, rotary_dim, 2, dtype=torch.float64, device=oracle.DEVICE) / rotary_dim
    return torch.pow(torch.tensor(float(theta), dtype=torch.float64, device=oracle.DEVICE), -exponent)


def _rope_table_fp64(n_pos: int, rotary_dim: int, theta: float) -> torch.Tensor:
    angles = torch.arange(n_pos, dtype=torch.float64, device=oracle.DEVICE)[:, None] * _inv_freq_fp64(rotary_dim,
                                                                                                    theta)[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def _rope_fp64(x: torch.Tensor, positions: torch.Tensor, head_dim: int, rotary_dim: int, theta: float):
    tokens = x.shape[0]
    x64 = x.to(torch.float64).reshape(tokens, -1, head_dim)
    half = rotary_dim // 2
    angles = positions.to(torch.float64)[:, None] * _inv_freq_fp64(rotary_dim, theta)[None, :]
    cos = angles.cos()[:, None, :]
    sin = angles.sin()[:, None, :]
    x1 = x64[..., :half]
    x2 = x64[..., half:rotary_dim]
    out = x64.clone()
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:rotary_dim] = x2 * cos + x1 * sin
    return out.reshape(tokens, -1)


# -- native entry points ----------------------------------------------------------------------------------------


def _native_embedding(table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    vocab, hidden = table.shape
    out = oracle.empty((ids.numel(), hidden), torch.bfloat16)
    oracle.call("cere_ktest_embedding", table, ids, ids.numel(), vocab, hidden, out)
    return out


def _native_rmsnorm(x: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float, offset: float):
    rows, row_dim = x.shape
    out = oracle.empty((rows, row_dim), torch.bfloat16)
    oracle.call("cere_ktest_rmsnorm", x, weight, rows, row_dim, group_size, float(eps), float(offset), out)
    return out


def _native_rmsnorm_gated(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rows, row_dim = x.shape
    out = oracle.empty((rows, row_dim), torch.bfloat16)
    oracle.call("cere_ktest_rmsnorm_gated", x, z, weight, rows, row_dim, float(eps), out)
    return out


def _native_rope_table(n_pos: int, rotary_dim: int, theta: float) -> torch.Tensor:
    table = oracle.empty((n_pos, rotary_dim), torch.float32)
    oracle.call("cere_ktest_rope_table", table, n_pos, rotary_dim, oracle.F64(theta))
    return table


def _native_rope(q, k, table, positions, n_q_heads: int, n_kv_heads: int, head_dim: int, rotary_dim: int):
    q_out = oracle.empty(tuple(q.shape), torch.bfloat16)
    k_out = oracle.empty(tuple(k.shape), torch.bfloat16)
    oracle.call("cere_ktest_rope", q, k, table, positions, q.shape[0], n_q_heads, n_kv_heads, head_dim, rotary_dim,
                q_out, k_out)
    return q_out, k_out


def _native_sigmoid_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    rows, row_dim = x.shape
    out = oracle.empty((rows, row_dim), torch.bfloat16)
    oracle.call("cere_ktest_sigmoid_gate", x, gate, rows, row_dim, out)
    return out


def _native_top2(logits: torch.Tensor):
    rows, vocab = logits.shape
    ids = oracle.empty((rows, 2), torch.int32)
    scores = oracle.empty((rows, 2), torch.float32)
    oracle.call("cere_ktest_top2", logits, rows, vocab, ids, scores)
    return ids, scores


def _native_kv_store(k, v, slots, k_cache, v_cache) -> None:
    tokens, row_dim = k.shape
    oracle.call("cere_ktest_kv_store_fp8", k, v, slots, tokens, row_dim, k_cache, v_cache, k_cache.shape[0])


# -- shared checks ----------------------------------------------------------------------------------------------


def _check_tolerance(tc: unittest.TestCase, case: str, sglang_out: torch.Tensor, reference: torch.Tensor,
                     native_run) -> None:
    """Calibrate (report first), run native, check within the limit, and check a second run is bit-identical."""
    c = oracle.calibrate(ROW, case, sglang_out, reference, oracle.FLOOR_OP)
    limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
    native = native_run()
    oracle.check_within(tc, ROW, case, oracle.rel_l2(native, sglang_out), c=c, limit=limit)
    oracle.check_bit_exact(tc, ROW, f"{case}_repeat", native_run(), native)


# k1::rmsnorm / rmsnorm_gated_sigmoid take one warp per row above this many rows of dim <= 256 (the prefill path) and
# one block per row up to it (the decode path).
NORM_WARP_MIN_ROWS = 1024


def _check_row_split(tc: unittest.TestCase, case: str, rows: int, native_rows) -> None:
    """Above NORM_WARP_MIN_ROWS rows: the norm over all rows bit-identical to the same rows in calls of at most
    NORM_WARP_MIN_ROWS rows (one block per row), so the warp-per-row path reproduces the per-row arithmetic."""
    if rows <= NORM_WARP_MIN_ROWS:
        return
    whole = native_rows(0, rows)
    split = torch.cat([native_rows(a, min(a + NORM_WARP_MIN_ROWS, rows))
                       for a in range(0, rows, NORM_WARP_MIN_ROWS)])
    oracle.check_bit_exact(tc, ROW, f"{case}_row_split", whole, split)


def _isolation_row(layer: int, op: str, native: torch.Tensor, control: torch.Tensor,
                   high_precision: torch.Tensor) -> None:
    """One report-only D6 isolation row: `c` = rel-L2(control, FP64), `limit` = max(FLOOR_OP, 2c), `measured` =
    rel-L2(native, control) and the differing elements on identical inputs; `within` = measured <= limit. Asserts
    nothing (tasks/m22d-spec.md C7)."""
    c = oracle.rel_l2(control, high_precision)
    limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
    measured = oracle.rel_l2(native, control)
    differing = int((native.reshape(-1) != control.reshape(-1)).sum().item())
    oracle.report_row(ROW, f"isolate_L{layer}_{op}_n1", phase="isolate", layer=layer, op=op, m=1,
                      control="fused_qk_gemma_rmsnorm_rope_gate",
                      native="cere_ktest_rmsnorm + cere_ktest_rope",
                      c=c, limit=limit, measured=measured, within=measured <= limit, elements=native.numel(),
                      differing=differing)


class K1Primitives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # SGLang modules read the published config (get_exec) at construction; publish a dummy-boundary one.
        cls.sglang_config = get_context().override_server_args()
        cls.sglang_config.install()
        cls.cfg = oracle.text_config()
        cls.eps = float(cls.cfg["rms_norm_eps"])

    @classmethod
    def tearDownClass(cls):
        cls.sglang_config.restore()

    # -- embedding ----------------------------------------------------------------------------------------------

    def test_embedding_gather_bit_exact(self):
        table = _bf16(oracle.real_tensor("model.language_model.embed_tokens.weight"), "embed_tokens")
        vocab, hidden = table.shape
        self.assertEqual((vocab, hidden), (self.cfg["vocab_size"], self.cfg["hidden_size"]))
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(table, requires_grad=False)
        method = UnquantizedEmbeddingMethod()
        for seed, n in enumerate(TOKEN_COUNTS):
            with self.subTest(n=n):
                ids = torch.randint(0, vocab, (n,), generator=oracle.generator(100 + seed), device=oracle.DEVICE,
                                    dtype=torch.int32)
                ids[-1] = vocab - 1
                if n > 1:
                    ids[0] = 0
                reference = method.embedding(layer, ids.long())
                native = _native_embedding(table, ids)
                oracle.check_bit_exact(self, ROW, f"embedding_n{n}", native, reference)
                oracle.check_bit_exact(self, ROW, f"embedding_n{n}_repeat", _native_embedding(table, ids), native)

    # -- RMSNorm ------------------------------------------------------------------------------------------------

    def test_rmsnorm_grouped_gemma_hc_norm(self):
        hidden = self.cfg["hidden_size"]
        norm_dim = hidden * self.cfg["hc_count"]
        names = {
            "hc_norm_layer0": oracle.layer_prefix(0) + "attn_hyper_connection.hc_norm.weight",
            "hc_norm_mixer": "model.language_model.hyper_connection_mixer.hc_norm.weight",
        }
        for which, name in names.items():
            weight = _bf16(oracle.real_tensor(name), name)
            with _sglang_build():
                norm = GroupedGemmaRMSNorm(norm_dim, eps=self.eps, group_size=hidden)
            norm.weight.weight_loader(norm.weight, weight)
            for seed, n in enumerate(TOKEN_COUNTS):
                with self.subTest(norm=which, n=n):
                    x = _randn((n, norm_dim), 200 + seed)
                    sglang_out = norm(x)
                    reference = _rmsnorm_fp64(x, weight, hidden, self.eps, 1.0)
                    _check_tolerance(self, f"rmsnorm_{which}_n{n}", sglang_out, reference,
                                     lambda: _native_rmsnorm(x, weight, hidden, self.eps, 1.0))

    def test_rmsnorm_qk_norm(self):
        head_dim = self.cfg["head_dim"]
        heads = {"q_norm": self.cfg["num_attention_heads"], "k_norm": self.cfg["num_key_value_heads"]}
        for which, n_heads in heads.items():
            name = oracle.layer_prefix(ATTN_LAYER) + f"self_attn.{which}.weight"
            weight = _bf16(oracle.real_tensor(name), name)
            with _sglang_build():
                norm = GemmaRMSNorm(head_dim, eps=self.eps)
            norm.weight.weight_loader(norm.weight, weight)
            for seed, n in enumerate(TOKEN_COUNTS):
                with self.subTest(norm=which, n=n):
                    # The model normalizes q.reshape(-1, head_dim) (qwen3_5.py:1085-1088).
                    x = _randn((n * n_heads, head_dim), 300 + seed, scale=4.0)
                    sglang_out = norm(x)
                    reference = _rmsnorm_fp64(x, weight, head_dim, self.eps, 1.0)
                    _check_tolerance(self, f"rmsnorm_{which}_n{n}", sglang_out, reference,
                                     lambda: _native_rmsnorm(x, weight, head_dim, self.eps, 1.0))
                    _check_row_split(self, f"rmsnorm_{which}_n{n}", x.shape[0],
                                     lambda a, b: _native_rmsnorm(x[a:b], weight, head_dim, self.eps, 1.0))

    def test_rmsnorm_plain(self):
        head_v_dim = self.cfg["linear_value_head_dim"]
        n_heads = self.cfg["linear_num_value_heads"]
        name = oracle.layer_prefix(GDN_LAYER) + "linear_attn.norm.weight"
        weight = _bf16(oracle.real_tensor(name), name)
        with _sglang_build():
            norm = RMSNorm(head_v_dim, eps=self.eps)
        norm.weight.data.copy_(weight)
        for seed, n in enumerate(TOKEN_COUNTS):
            with self.subTest(n=n):
                x = _randn((n * n_heads, head_v_dim), 400 + seed, scale=0.5)
                sglang_out = norm(x)
                reference = _rmsnorm_fp64(x, weight, head_v_dim, self.eps, 0.0)
                _check_tolerance(self, f"rmsnorm_plain_n{n}", sglang_out, reference,
                                 lambda: _native_rmsnorm(x, weight, head_v_dim, self.eps, 0.0))
                _check_row_split(self, f"rmsnorm_plain_n{n}", x.shape[0],
                                 lambda a, b: _native_rmsnorm(x[a:b], weight, head_v_dim, self.eps, 0.0))

    def test_rmsnorm_gated_gdn(self):
        head_v_dim = self.cfg["linear_value_head_dim"]
        n_heads = self.cfg["linear_num_value_heads"]
        if self.cfg["output_gate_type"] != "sigmoid":
            raise oracle.OracleError(f"output_gate_type {self.cfg['output_gate_type']!r}; the kernel gates by sigmoid")
        name = oracle.layer_prefix(GDN_LAYER) + "linear_attn.norm.weight"
        weight = _bf16(oracle.real_tensor(name), name)
        with _sglang_build():
            norm = RMSNormGated(head_v_dim, eps=self.eps, group_size=None, norm_before_gate=True,
                                device=oracle.DEVICE, dtype=torch.bfloat16, activation=self.cfg["output_gate_type"])
        norm.weight.data.copy_(weight)
        for seed, n in enumerate(TOKEN_COUNTS):
            with self.subTest(n=n):
                # GDN normalizes core_attn_out.reshape(-1, head_v_dim) gated by z of the same shape.
                x = _randn((n * n_heads, head_v_dim), 500 + seed, scale=0.5)
                z = _randn((n * n_heads, head_v_dim), 550 + seed, scale=2.0)
                sglang_out = norm(x, z)
                x64 = _rmsnorm_fp64(x, weight, head_v_dim, self.eps, 0.0)
                reference = x64 * torch.sigmoid(z.to(torch.float64))
                _check_tolerance(self, f"rmsnorm_gated_n{n}", sglang_out, reference,
                                 lambda: _native_rmsnorm_gated(x, z, weight, self.eps))
                _check_row_split(self, f"rmsnorm_gated_n{n}", x.shape[0],
                                 lambda a, b: _native_rmsnorm_gated(x[a:b], z[a:b], weight, self.eps))

    # -- rotary -------------------------------------------------------------------------------------------------

    def _rotary(self):
        """A fresh SGLang MRotaryEmbedding with the arguments get_rope derives from the real rope_parameters
        (factory.py:95-210, qwen3_5.py:947-956), its FP32 table captured before any forward (forward casts the
        cached table to the query dtype, base.py:140-147)."""
        cfg = self.cfg
        params = cfg["rope_parameters"]
        head_dim = cfg["head_dim"]
        rotary_dim = int(head_dim * cfg["partial_rotary_factor"])
        if params.get("rope_type") != "default" or "mrope_section" not in params:
            raise oracle.OracleError(f"rope_parameters {params} are not the default mRoPE get_rope builds")
        with _sglang_build():
            rope = MRotaryEmbedding(head_dim, rotary_dim, cfg["max_position_embeddings"], params["rope_theta"], True,
                                    torch.bfloat16, mrope_section=params["mrope_section"],
                                    mrope_interleaved=params.get("mrope_interleaved", False),
                                    mrope_interleaved_glm=params.get("mrope_interleaved_glm", False))
        rope = rope.to(oracle.DEVICE)
        table = rope.cos_sin_cache.detach().clone()
        if table.dtype != torch.float32 or tuple(table.shape) != (cfg["max_position_embeddings"], rotary_dim):
            raise oracle.OracleError(f"SGLang cos_sin_cache is {table.dtype} {tuple(table.shape)}")
        if rotary_dim != 64 or list(rope.mrope_section) != [11, 11, 10] or not rope.mrope_interleaved:
            raise oracle.OracleError("the rotary module does not match the spec (64 dims, [11,11,10], interleaved)")
        return rope, table, rotary_dim, float(params["rope_theta"])

    def test_rope_table(self):
        _, table, rotary_dim, theta = self._rotary()
        sglang_table = table[:MAX_CONTEXT].contiguous()
        reference = _rope_table_fp64(MAX_CONTEXT, rotary_dim, theta)
        _check_tolerance(self, f"rope_table_{MAX_CONTEXT}", sglang_table, reference,
                         lambda: _native_rope_table(MAX_CONTEXT, rotary_dim, theta))

    def test_rope_partial_interleaved_mrope(self):
        rope, table, rotary_dim, theta = self._rotary()
        head_dim = self.cfg["head_dim"]
        n_q = self.cfg["num_attention_heads"]
        n_kv = self.cfg["num_key_value_heads"]
        native_table = table[:MAX_CONTEXT].contiguous()
        cases = {
            "prefill_2048": torch.arange(2048, dtype=torch.int64, device=oracle.DEVICE),
            "decode_1_last": torch.tensor([MAX_CONTEXT - 1], dtype=torch.int64, device=oracle.DEVICE),
            "scattered_7": torch.randint(0, MAX_CONTEXT, (7,), generator=oracle.generator(600), device=oracle.DEVICE,
                                         dtype=torch.int64),
            "chunk_far_2048": torch.arange(MAX_CONTEXT - 2048, MAX_CONTEXT, dtype=torch.int64, device=oracle.DEVICE),
        }
        for seed, (case, positions) in enumerate(cases.items()):
            with self.subTest(case=case):
                tokens = positions.numel()
                q = _randn((tokens, n_q * head_dim), 610 + seed, scale=2.0)
                k = _randn((tokens, n_kv * head_dim), 630 + seed, scale=2.0)
                # Text-only mRoPE positions: three equal rows (T, H, W).
                positions_3 = positions[None, :].expand(3, tokens).contiguous()
                q_sglang, k_sglang = rope(positions_3, q.clone(), k.clone())
                q_ref = _rope_fp64(q, positions, head_dim, rotary_dim, theta)
                k_ref = _rope_fp64(k, positions, head_dim, rotary_dim, theta)
                sglang_out = torch.cat((q_sglang.reshape(-1), k_sglang.reshape(-1)))
                reference = torch.cat((q_ref.reshape(-1), k_ref.reshape(-1)))

                def run():
                    q_out, k_out = _native_rope(q, k, native_table, positions, n_q, n_kv, head_dim, rotary_dim)
                    return torch.cat((q_out.reshape(-1), k_out.reshape(-1)))

                _check_tolerance(self, f"rope_{case}", sglang_out, reference, run)

    def test_isolate_decode_qk_norm_rope(self):
        """D6 isolation rows (tasks/m22d-spec.md C7), report-only: QK norm + partial mRoPE of one decode token at
        parity-regime positions on every full-attention layer vs the control's fused prep kernel."""
        cfg = self.cfg
        _, table, rotary_dim, theta = self._rotary()
        native_table = table[:MAX_CONTEXT].contiguous()
        head_dim = cfg["head_dim"]
        n_q = cfg["num_attention_heads"]
        n_kv = cfg["num_key_value_heads"]
        layers = [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
        for layer in layers:
            prefix = oracle.layer_prefix(layer) + "self_attn."
            q_norm = _bf16(oracle.real_tensor(prefix + "q_norm.weight"), "q_norm")
            k_norm = _bf16(oracle.real_tensor(prefix + "k_norm.weight"), "k_norm")
            for position in ISOLATE_POSITIONS:
                with self.subTest(layer=layer, position=position):
                    seed = 1_000 + 10 * layer + ISOLATE_POSITIONS.index(position)
                    # [q | gate] per head, as q_proj lays it out; k as k_proj.
                    q_gate = _randn((1, n_q * 2 * head_dim), seed, scale=4.0)
                    k = _randn((1, n_kv * head_dim), seed + 500, scale=4.0)
                    positions = torch.tensor([position], dtype=torch.int64, device=oracle.DEVICE)
                    sg_q, sg_k, _ = fused_qk_gemma_rmsnorm_rope_gate(
                        q_gate.clone(), k.clone(), q_norm, k_norm, table, positions, self.eps, n_q, n_kv, head_dim,
                        rotary_dim, has_gate=True,
                    )
                    q = q_gate.view(n_q, 2, head_dim)[:, 0].contiguous()
                    q_ref = _rope_fp64(_rmsnorm_fp64(q, q_norm, head_dim, self.eps, 1.0).reshape(1, -1), positions,
                                       head_dim, rotary_dim, theta)
                    k_ref = _rope_fp64(_rmsnorm_fp64(k.view(n_kv, head_dim), k_norm, head_dim, self.eps, 1.0)
                                       .reshape(1, -1), positions, head_dim, rotary_dim, theta)
                    q_normed = _native_rmsnorm(q, q_norm, head_dim, self.eps, 1.0).reshape(1, -1)
                    k_normed = _native_rmsnorm(k.view(n_kv, head_dim).contiguous(), k_norm, head_dim, self.eps,
                                               1.0).reshape(1, -1)
                    nat_q, nat_k = _native_rope(q_normed, k_normed, native_table, positions, n_q, n_kv, head_dim,
                                                rotary_dim)
                    _isolation_row(layer, f"q_norm_rope_pos{position}", nat_q, sg_q.reshape(1, -1), q_ref)
                    _isolation_row(layer, f"k_norm_rope_pos{position}", nat_k, sg_k.reshape(1, -1), k_ref)

    # -- attention output gate ----------------------------------------------------------------------------------

    def test_sigmoid_gate(self):
        row_dim = self.cfg["num_attention_heads"] * self.cfg["head_dim"]
        for seed, n in enumerate(TOKEN_COUNTS):
            with self.subTest(n=n):
                x = _randn((n, row_dim), 700 + seed)
                gate = _randn((n, row_dim), 750 + seed, scale=3.0)
                sglang_out = fused_sigmoid_mul(x, gate, inplace=False)
                reference = x.to(torch.float64) * torch.sigmoid(gate.to(torch.float64))
                _check_tolerance(self, f"sigmoid_gate_n{n}", sglang_out, reference,
                                 lambda: _native_sigmoid_gate(x, gate))

    # -- top-2 argmax -------------------------------------------------------------------------------------------

    def _check_top2(self, case: str, logits: torch.Tensor):
        # SGLang: BF16 logits -> .float() (logits_processor.py:831) -> torch.argmax (sampler.py:137).
        logits_f32 = logits.float()
        top1 = torch.argmax(logits_f32, -1)
        masked = logits_f32.clone()
        masked.scatter_(1, top1[:, None], float("-inf"))
        top2 = torch.argmax(masked, -1)
        ref_ids = torch.stack((top1, top2), dim=-1).to(torch.int32).contiguous()
        ref_scores = torch.gather(logits_f32, 1, ref_ids.long()).contiguous()
        ids, scores = _native_top2(logits)
        oracle.check_bit_exact(self, ROW, f"top2_{case}_ids", ids, ref_ids)
        oracle.check_bit_exact(self, ROW, f"top2_{case}_scores", scores, ref_scores)
        ids_again, scores_again = _native_top2(logits)
        oracle.check_bit_exact(self, ROW, f"top2_{case}_repeat_ids", ids_again, ids)
        oracle.check_bit_exact(self, ROW, f"top2_{case}_repeat_scores", scores_again, scores)
        return ids

    def test_top2_argmax(self):
        vocab = self.cfg["vocab_size"]
        for seed, n in enumerate((1, 7, 64)):
            with self.subTest(n=n):
                logits = _randn((n, vocab), 800 + seed, scale=3.0)
                self._check_top2(f"random_n{n}", logits)

    def test_top2_argmax_ties_first_index_wins(self):
        vocab = self.cfg["vocab_size"]
        logits = _randn((6, vocab), 850, scale=1.0)
        expected = []
        # Row 0: every logit equal -> (0, 1).
        logits[0].fill_(1.5)
        expected.append((0, 1))
        # Row 1: three tied maxima -> the first two of them.
        logits[1, [5, 100, vocab - 1]] = 32.0
        expected.append((5, 100))
        # Row 2: unique maximum, tied runner-up -> the first runner-up.
        logits[2, 7] = 40.0
        logits[2, [vocab - 2, 3, 9]] = 30.0
        expected.append((7, 3))
        # Row 3: maximum at the last index, runner-up tie at the end of the row.
        logits[3, vocab - 1] = 50.0
        logits[3, [vocab - 3, vocab - 2]] = 49.0
        expected.append((vocab - 1, vocab - 3))
        # Row 4: all negative, two tied maxima on either side of index 65,536.
        logits[4].fill_(-60.0)
        logits[4, [65_536, 65_535]] = -2.0
        expected.append((65_535, 65_536))
        # Row 5: BF16 rounding makes neighbours equal: 1 + 2^-9 and 1 + 2^-10 both round to 1.0 in BF16.
        logits[5].fill_(-1.0)
        logits[5, 1000] = 1.0 + 2.0 ** -9
        logits[5, 999] = 1.0 + 2.0 ** -10
        expected.append((999, 1000))
        ids = self._check_top2("ties", logits.contiguous())
        expected_ids = torch.tensor(expected, dtype=torch.int32, device=oracle.DEVICE)
        oracle.check_bit_exact(self, ROW, "top2_ties_first_index", ids, expected_ids)

    # -- FP8 E4M3 KV store --------------------------------------------------------------------------------------

    def test_kv_store_fp8_bit_exact(self):
        row_dim = self.cfg["num_key_value_heads"] * self.cfg["head_dim"]
        e4m3_max = float(torch.finfo(torch.float8_e4m3fn).max)
        for seed, n in enumerate(TOKEN_COUNTS):
            for kind in ("attention_scale", "full_range"):
                with self.subTest(n=n, kind=kind):
                    if kind == "attention_scale":
                        k = _randn((n, row_dim), 920 + seed, scale=3.0)
                        v = _randn((n, row_dim), 940 + seed, scale=1.0)
                    else:
                        g = oracle.generator(980 + seed)
                        # Every E4M3 binade incl. subnormals, zeros of both signs, rounding ties and values past
                        # the E4M3 maximum (torch's cast decides those bytes; the native cast must match them).
                        mag = torch.pow(2.0, torch.empty((2, n, row_dim), device=oracle.DEVICE).uniform_(
                            -12.0, 9.5, generator=g))
                        sign = torch.where(torch.rand((2, n, row_dim), generator=g, device=oracle.DEVICE) < 0.5,
                                           -1.0, 1.0)
                        both = (mag * sign).to(torch.bfloat16)
                        both[:, :, 0] = 0.0
                        both[:, :, 1] = -0.0
                        both[:, :, 2] = e4m3_max
                        both[:, :, 3] = -(e4m3_max + 16.0)
                        both[:, :, 4] = 1.0 + 2.0 ** -4  # halfway between two E4M3 values (ties to even)
                        k, v = both[0].contiguous(), both[1].contiguous()
                    # Slot 0 is the reference pool's reserved padding slot (store_cache skips it).
                    slots = (torch.randperm(KV_SLOTS - 1, generator=oracle.generator(960 + seed),
                                            device=oracle.DEVICE)[:n] + 1).to(torch.int64)
                    ref_k_cache = torch.full((KV_SLOTS, row_dim), 0xA5, dtype=torch.uint8, device=oracle.DEVICE)
                    ref_v_cache = ref_k_cache.clone()
                    # MHATokenToKVPool.set_kv_buffer (memory_pool.py:2474-2486) with scale 1.0, then the scatter.
                    cache_k = k.clone().div_(1.0).to(torch.float8_e4m3fn).view(torch.uint8)
                    cache_v = v.clone().div_(1.0).to(torch.float8_e4m3fn).view(torch.uint8)
                    _set_kv_buffer_impl(cache_k, cache_v, ref_k_cache, ref_v_cache, slots, row_dim=row_dim,
                                        store_dtype=torch.uint8, device_module=torch.cuda, size_limit=KV_SLOTS)
                    nat_k_cache = torch.full_like(ref_k_cache, 0xA5)
                    nat_v_cache = torch.full_like(ref_v_cache, 0xA5)
                    _native_kv_store(k, v, slots, nat_k_cache, nat_v_cache)
                    case = f"kv_store_fp8_{kind}_n{n}"
                    oracle.check_bit_exact(self, ROW, f"{case}_k", nat_k_cache, ref_k_cache)
                    oracle.check_bit_exact(self, ROW, f"{case}_v", nat_v_cache, ref_v_cache)
                    again_k = torch.full_like(ref_k_cache, 0xA5)
                    again_v = torch.full_like(ref_v_cache, 0xA5)
                    _native_kv_store(k, v, slots, again_k, again_v)
                    oracle.check_bit_exact(self, ROW, f"{case}_repeat_k", again_k, nat_k_cache)
                    oracle.check_bit_exact(self, ROW, f"{case}_repeat_v", again_v, nat_v_cache)

    # -- Device byte copies (m23 design note P11R-20) ------------------------------------------------------------

    def test_copy_segments_bit_exact(self):
        """k1::copy_segments equals torch's copies byte for byte: a 1.5 MiB flat range (a GDN SSM undo), a 60 KiB
        one (the conv state), a pitched 4-row range (one HC stream of the embedding) and an unaligned pitched range
        (byte path); 98 segments in one launch (DG-03: the verify_commit restore's count, the 16-byte and the byte
        paths mixed) and 128 (the bound); no other destination byte changes. 129 segments, 0, or a null pointer are
        CERE_ERR_ARGUMENT. Fails on a library without the kernel or without the 128-segment launch."""
        src = torch.randint(0, 256, (4 << 20,), dtype=torch.uint8, device=oracle.DEVICE,
                            generator=oracle.generator(1010))
        base = src.data_ptr()
        # (dst offset, dst pitch, src offset, src pitch, width, rows)
        cases = {
            "flat": [(0, 1572864, 0, 1572864, 1572864, 1), (1572864, 61440, 2 << 20, 61440, 61440, 1)],
            "pitched": [(4096, 20480, 3 << 20, 5120, 5120, 4)],
            "unaligned": [(3, 1003, 1, 1010, 1001, 3), (65536, 0, 77, 0, 0, 1)],
            "four": [(0, 256, 512, 256, 256, 1), (4096, 16, 8192, 16, 16, 1), (8192, 32, 16384, 64, 32, 3),
                     (65537, 9, 70001, 9, 9, 1)],
            "restore98": [(i * 40960, 0, (i * 37) % 4096 * 512, 0, 16 * (1 + i % 23) + (i % 3 == 2), 1)
                          for i in range(98)],
            "bound128": [(i * 32768, 4096, 3 << 20, 1024, 16 * (i % 64), 1 + i % 4) for i in range(128)],
        }
        for name, segments in cases.items():
            with self.subTest(case=name):
                dst = torch.full((4 << 20,), 0xA5, dtype=torch.uint8, device=oracle.DEVICE)
                expected = dst.clone()
                for d, dp, s, sp, width, rows in segments:
                    for r in range(rows):
                        expected[d + r * dp:d + r * dp + width] = src[s + r * sp:s + r * sp + width]
                ranges = torch.tensor([[dst.data_ptr() + d, dp, base + s, sp, width, rows]
                                       for d, dp, s, sp, width, rows in segments], dtype=torch.int64)
                oracle.call("cere_ktest_copy_segments", ranges, len(segments))
                oracle.check_bit_exact(self, ROW, f"copy_segments_{name}", dst, expected)
        sink = torch.zeros(1, dtype=torch.uint8, device=oracle.DEVICE)
        empty = torch.tensor([[sink.data_ptr(), 0, base, 0, 0, 1]] * 129, dtype=torch.int64)  # valid, 0 bytes each
        self.assertEqual(oracle.call_status("cere_ktest_copy_segments", empty, 128), 0)  # CERE_OK
        self.assertEqual(oracle.call_status("cere_ktest_copy_segments", empty, 129), oracle.CERE_ERR_ARGUMENT)
        self.assertEqual(oracle.call_status("cere_ktest_copy_segments", empty, 0), oracle.CERE_ERR_ARGUMENT)
        null = torch.zeros((5, 6), dtype=torch.int64)
        self.assertEqual(oracle.call_status("cere_ktest_copy_segments", null, 5), oracle.CERE_ERR_ARGUMENT)


if __name__ == "__main__":
    unittest.main()
