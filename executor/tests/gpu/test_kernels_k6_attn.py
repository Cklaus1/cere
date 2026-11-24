"""K6 (tasks/m22-spec.md, M2.2a): the native full-attention block vs SGLang's QSA backend on real layer-3 weights.

The block, per the kernel decision table ("Full attention (non-QSA part)" and "QSA sparse attention"):
hidden -> q_proj (24 heads x [q 256 | gate 256]) / k_proj / v_proj (BF16) -> Gemma QK RMSNorm (1 + w) -> partial
interleaved mRoPE (64 of 256 dims, NeoX rotate-half, text positions) -> FP8 E4M3 KV append (scale 1.0) -> dense
causal GQA (24 Q heads over 2 KV heads, softmax scale 256^-0.5) -> sigmoid output gate -> o_proj. K/V source as in
the reference: an extend without a cached prefix attends over the current tokens' unquantized BF16 K/V; an extend
with a prefix and a decode attend over the FP8 cache (dequantized with scale 1.0), including the entries the same
call just stored.

SGLang side (the reference op of the tolerance rule), on the real objects that serve this model:
- projections: BF16 `F.linear` (`UnquantizedLinearMethod.apply`, `$S181/sglang/srt/layers/quantization/unquant.py:393`;
  `*.self_attn.*` is in the checkpoint's quantization ignore list) over the q/k/v weights stacked as
  `QKVParallelLinear` does; prep is `fused_qk_gemma_rmsnorm_rope_gate`
  (`$S181/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py:131`, the CUDA path of
  `Qwen3_5AttentionDecoderLayer._prepare_qkv_gate`, `models/qwen3_5.py:1093-1120,1204-1208`) reading
  `MRotaryEmbedding.cos_sin_cache` built by `get_rope` exactly as `models/qwen3_5.py:935-956`; gate is
  `fused_sigmoid_mul(..., inplace=True)` and o_proj `F.linear` (`models/qwen4_exp.py:1535-1544`).
- attention: a real `QwenSparseAttnBackend`. Its `__init__(runner)` reads everything through `getattr`
  (`qwen_sparse_attn_backend.py:208-255`), so it is built from a namespace runner holding a real
  `MHATokenToKVPool` (FP8 E4M3, 2 heads x 256, the class `HybridLinearKVPool` uses as its full-attention pool,
  `mem_cache/memory_pool.py:3741`, here with `start_layer` = `end_layer` = 3) and a real `ReqToTokenPool`. A full
  `ModelRunner`/`ForwardBatch` is not built: the CUDA branches of `forward_extend`/`forward_decode` read only
  `forward_mode`, `out_cache_loc`, `extend_seq_lens[_cpu]`, `seq_lens_cpu`, `req_pool_indices`
  (`:1404-1506,1647-1665`), so a namespace batch carries exactly those fields. The layer is a namespace with the
  `RadixAttention` attributes the backend reads (`layer_id`, `tp_q_head_num`, `head_dim`, `scaling`). The paged
  path (`_forward_paged_attention`, `:1667-1759`, including `_forward_trtllm_sparse` `:1559-1645` when
  `_resolve_trtllm_sparse_decode` resolves) needs per-forward metadata; the test installs the
  `QwenSparseAttnMetadata` that `_metadata_from_forward_batch` derives for those modes (`sequence_lengths`,
  `token_to_batch_idx`, `token_slot_table`, `row_req_pool_indices`; non-graph) as `backend.forward_metadata`.
- top-k indices: SGLang's own QSA selection (`build_qsa_row_ranges`, `qsa_fast_topk` with block top-k 512,
  `expand_qsa_block_indices`; `qsa_indexer.py:455-533`) on seeded random block logits. At <= 2,048 tokens every
  causal position is selected whatever the logits (spec Q4); each case counts the rows whose selected set is not
  exactly the causal set (expected 0), resolving `[INFERENCE]` item 1.

Modes (spec K6 row, kernel decision table), each at the final sequence length L:
- `extend_no_prefix` (L in {1, 517, 2048}): one extend of L tokens, `forward_extend` `:1461-1470` (BF16 k/v).
- `extend_prefix_4` (L in {517, 2048}): prefix L-8 stored, then two 4-token extends through `_ck` (`:1472-1506`).
  L = 1 has no prefix, so the mode does not apply there.
- `extend_prefix_chunk` (prefix P + one extend of T tokens through `_ck`, the K10 later-prefill-chunk shape): the
  native kernel's FP8/slot-table paths at the boundaries of its launch choice (k6_attention.cu `split_count`,
  `kSplitMaxTokens` 16, `kKeysPerSplit` 128, 4-row query tiles): P 480 + T 37 and P 2031 + T 17 (T > 16, unsplit,
  partial last tile, pos0 > 0); P 64 + T 7 (T <= 16 but 71 keys, one split, partial tile); P 300 + T 7 (3 key
  splits, partial tile and partial last key tile); P 2032 + T 16 (16 splits at the context limit).
- `decode` (L in {1, 517, 2048}): prefix L-1 stored, then one decode token, `forward_decode` `:1647-1696`.
- `target_verify_4` (L in {517, 2048}): prefix L-4 stored, then the 4-row TARGET_VERIFY extend that the reference
  runs for every token after prefill (spec Q8; `forward_extend` `:1433-1437` -> `_forward_paged_attention`); the
  native side decodes the same 4 tokens one at a time (Q8: the verify path is the primary decode oracle).
- `isolate_L<layer>_decode_P<P>` (D6 isolation rows, tasks/m22d-spec.md D6/C7, phase `isolate`; report-only, nothing
  asserted beyond running and the causal-selection count): every full-attention layer (3, 7, ..., 47, real
  weights), P in {21, 95, 159, 516} (kv_len 22 in the parity regime; 96/160/517 = 1/2/5 native split-K key splits),
  prefix stored by SGLang's extend path and copied into the native cache, then one decode token. Rows, each
  `c` = rel-L2(control, FP64), `limit` = max(FLOOR_OP, 2c), `measured` = rel-L2(native, control), `within`,
  differing elements: `qkv_prep_fp8_attn_gate_same_projection` (the control's prep, FP8 store, paged decode and
  gate fed the native block's K2 projection bytes, vs the native `attn_out`: QK norm, RoPE, FP8 KV write/read and
  split-K attention on identical inputs); `o_proj_same_input` (`F.linear` on the native `attn_out` vs the native
  block output); `attn_gated_vs_served` and `block_out` (vs the control as served, its own projection); and
  `kv_new_token_bytes` (differing FP8 bytes of the new token's K/V row and of the prefix rows).
The prefix of a mode is stored through the pool's `set_kv_buffer` exactly as `forward_extend` does first
(`:1417-1420`); the native cache starts as a byte copy of SGLang's cache before the measured step(s).

KV scale: the backend stores with `token_to_kv_pool.set_kv_buffer(layer, loc, k, v)` (no scale argument,
`:1418-1419,1661-1662`); on the serving pool `QSATokenToKVPool` that resolves to `HybridLinearKVPool.set_kv_buffer`
whose `k_scale`/`v_scale` defaults are forwarded to the MHA pool's `cache_k.div_(k_scale)`
(`memory_pool.py:3947-3973,2472-2478`). The test reads those defaults from the signature, reports them and requires
1.0 (spec Q8); the direct MHA pool used here then produces the same bytes (division by 1.0 is exact).

Tolerance rule: c = rel-L2(SGLang op, FP64 reference on the same BF16 inputs); the FP64 reference applies the
specified arithmetic in FP64 (FP64 projections, Gemma RMSNorm, rotary angles from rope_theta, softmax, gate,
o_proj) and quantizes K/V to E4M3 with the scale (from FP64) in exactly the modes where SGLang reads the FP8 cache.
Limit max(FLOOR_OP, 2c); measured = rel-L2(native, SGLang). Two stages per case: `attn` (gated attention output,
the o_proj input) and `out` (block output), both on SGLang's default configuration.

KV store bit-exact (kernel table): the store is QK norm, rope and the FP8 quantization, so both sides must feed it
the same projection bytes. SGLang's stacked q/k/v `F.linear` does not give those bytes: torch's default
`allow_bf16_reduced_precision_reduction=True` lets cuBLAS pick split-K with BF16 partial sums at some M (on GPU0:
M 2-16, 256, 1024, 2048), and even with the flag False its FP32 accumulation order differs from the native K2 GEMM
at some M (on GPU0: M 1 with 1-3 K/V elements, M 17 with Q 311, K 14, V 19; all within 1 ulp). The bit-exact check
therefore runs SGLang's `fused_qk_gemma_rmsnorm_rope_gate` + `set_kv_buffer` for the same calls on the projection
bytes of `cere_ktest_gemm_bf16` (w_q, w_k, w_v: the native block's own GEMM calls) and compares all cache bytes.
Informational rows, not asserted, keep both projection effects visible per mode and M: `*/kv_[kv]_vs_default_flag`
(cache-byte differences against the default-flag SGLang run) and `*/projection_vs_flag_false` (differing elements
of the flag-False projection vs K2; the flag is toggled in a try/finally that restores it).

`cere_ktest_gemm_bf16` (K2, test_kernels_k2_gemm.py) is called here only to build that reference's input.

Native entry points
-------------------
    int cere_ktest_attn_block(
        const void *hidden,      /* bf16 [T, hidden_size] (tokens prefix_len .. prefix_len+T-1) */
        const void *w_q,         /* bf16 [n_q_heads*2*head_dim, hidden_size] q_proj, per head rows [q | gate] */
        const void *w_k,         /* bf16 [n_kv_heads*head_dim, hidden_size] */
        const void *w_v,         /* bf16 [n_kv_heads*head_dim, hidden_size] */
        const void *w_o,         /* bf16 [hidden_size, n_q_heads*head_dim] */
        const void *q_norm,      /* bf16 [head_dim] raw Gemma weight, applied as (1 + w) */
        const void *k_norm,      /* bf16 [head_dim] */
        const void *cos_sin,     /* f32 [n_pos, rotary_dim]; row p = [cos(p*inv_freq_0..rd/2-1) | sin(...)]
                                    (SGLang's cos_sin_cache layout, same as cere_ktest_rope in K1) */
        int64_t n_pos,
        const void *positions,   /* int64 [T] rotary positions of the T new tokens */
        const void *slots,       /* int64 [prefix_len + T] cache row of every logical position */
        int64_t prefix_len,      /* tokens already in the cache; new token i is at logical position prefix_len+i */
        int64_t T,
        void *k_cache,           /* uint8 [n_slots, n_kv_heads*head_dim] FP8 E4M3 bytes, NHD, in/out */
        void *v_cache,           /* uint8 [n_slots, n_kv_heads*head_dim] */
        int64_t n_slots,
        int64_t kv_from_cache,   /* 0: attend over the new tokens' BF16 K/V (requires prefix_len == 0);
                                    1: attend over the FP8 cache incl. the entries stored by this call */
        int64_t hidden_size, int64_t n_q_heads, int64_t n_kv_heads, int64_t head_dim, int64_t rotary_dim,
        float eps,               /* rms_norm_eps */
        float kv_scale,          /* FP8 store: e4m3(bf16 / kv_scale); load: e4m3 * kv_scale */
        float softmax_scale,     /* head_dim^-0.5 */
        void *attn_out,          /* bf16 [T, n_q_heads*head_dim] gated attention output (o_proj input) */
        void *out,               /* bf16 [T, hidden_size] block output */
        cudaStream_t stream);
    Query i attends the logical positions 0 .. prefix_len+i (dense causal GQA, Q head h uses KV head
    h / (n_q_heads / n_kv_heads)); the call stores the T new K/V rows at slots[prefix_len ..] before attending.
    cere_ktest_attn_block runs the default prefill_attention lever (the CUDA-core kernels for every call);
    cere_ktest_attn_block_p4 (same parameters)
    forces the P4 tensor-core kernel (tasks/m23-spec.md P4, kernels/p4_attention.h) for calls of T >= 17; the
    `*/p4` cases run it against the same calibrated limits (FP64 reference and SGLang op unchanged).

    int cere_ktest_attn_kv_strides(const void *q, const void *k, const void *v, int64_t k_stride, int64_t v_stride,
                                   int64_t tokens, void *out, cudaStream_t stream);
    `k6::attention` (1..16 tokens, pos0 0) over BF16 K/V rows at the given row strides (elements): a stride below
    512 or not a multiple of 8 returns non-zero before any launch (K6KvStrideContract).

    int cere_ktest_attn_p4_cls(const void *q, const void *k, const void *v, int64_t tokens, int64_t pos0,
                               int64_t cls, void *out, cudaStream_t stream);
    `k6::attention` with the P4 lever over BF16 K/V rows [pos0 + tokens] (key j at row j) for the rows pos0 ..
    pos0 + tokens - 1 pinned to class `cls` (0 = tokens; BeatSglang S7): K6P4PinnedTail checks that a hot tail of
    1 .. 100 rows (fewer than 17 included) pinned to its cold call's class equals that call's rows bit for bit.
"""
from __future__ import annotations

import inspect
import types
import unittest

import torch
import torch.nn.functional as F

import oracle
from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import fused_qk_gemma_rmsnorm_rope_gate
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices, qsa_fast_topk
from sglang.srt.layers.attention.qsa.metadata import build_qsa_row_ranges
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    QwenSparseAttnMetadata,
    _resolve_trtllm_sparse_decode,
)
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.utils.hf_transformers.common import get_rope_config

ROW = "K6"
LAYER = 3
MAX_CONTEXT = 2048  # M2.2b context limit (Q4); also the cache size of every case
REQ = 1  # request row in ReqToTokenPool (row 0 is its padding row)
# D6 isolation (m22d design note C7): decode after a prefix of P tokens; kv_len P + 1 = 22 (a parity-regime
# chunk), 96 and 160 (one and two 128-key splits of the native split-K decode), 517 (five splits).
ISOLATE_PREFIXES = (21, 95, 159, 516)
F64 = torch.float64
INT32_MAX = torch.iinfo(torch.int32).max


def _gemma_rms64(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma RMSNorm in FP64 over the last dim: x * rsqrt(mean(x^2) + eps) * (1 + w)."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + weight.to(F64))


def _rope64(x: torch.Tensor, positions: torch.Tensor, theta: float, rotary_dim: int) -> torch.Tensor:
    """NeoX rotate-half over dims [0, rotary_dim) with FP64 angles; for text positions the interleaved mRoPE
    sections all carry the same position, so this is the full mRoPE."""
    half = rotary_dim // 2
    inv_freq = theta ** (-torch.arange(0, rotary_dim, 2, dtype=F64, device=x.device) / rotary_dim)
    angles = positions.to(F64)[:, None] * inv_freq[None, :]
    cos = angles.cos()[:, None, :]
    sin = angles.sin()[:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:rotary_dim]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., rotary_dim:]], dim=-1)


def _fp8_round64(x: torch.Tensor, scale: float) -> torch.Tensor:
    """The FP8 E4M3 cache round trip with `scale`, applied to FP64 values."""
    return (x / scale).to(torch.float32).to(torch.float8_e4m3fn).to(F64) * scale


def _causal_mismatch_rows(selected: torch.Tensor, positions: torch.Tensor) -> int:
    """Rows whose valid selected logical indices are not exactly {0, ..., position} (no duplicates)."""
    sel = selected.to(torch.int64)
    keys = torch.sort(torch.where(sel >= 0, sel, torch.full_like(sel, INT32_MAX)), dim=1).values
    cols = torch.arange(sel.shape[1], device=sel.device, dtype=torch.int64)[None, :]
    expected = torch.where(cols <= positions.to(torch.int64)[:, None], cols, torch.full_like(cols, INT32_MAX))
    return int((keys != expected).any(dim=1).sum().item())


class _Case:
    """One sequence: seeded hidden states, SGLang's pools/backend, and the slot table of its logical positions."""

    def __init__(self, suite: type, length: int, seed: int, layer: int = LAYER):
        self.length = length
        g = oracle.generator(seed)
        x = torch.randn((length, suite.hidden_size), generator=g, device=oracle.DEVICE, dtype=torch.float32)
        # Rows at unit RMS: the attention block's input is the output of a norm.
        self.hidden = (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True))).to(torch.bfloat16).contiguous()
        self.positions = torch.arange(length, device=oracle.DEVICE, dtype=torch.int64)
        self.pool = MHATokenToKVPool(
            size=MAX_CONTEXT,
            page_size=1,
            dtype=torch.float8_e4m3fn,
            head_num=suite.n_kv_heads,
            head_dim=suite.head_dim,
            layer_num=1,
            device=str(oracle.DEVICE),
            enable_memory_saver=False,
            start_layer=layer,
            end_layer=layer,
        )
        self.req_pool = ReqToTokenPool(
            size=1, max_context_len=MAX_CONTEXT, device=str(oracle.DEVICE), enable_memory_saver=False
        )
        # Logical position -> cache row: a seeded permutation of rows 1..MAX_CONTEXT (row 0 is the pad row).
        self.slots = (torch.randperm(MAX_CONTEXT, generator=g, device=oracle.DEVICE) + 1).to(torch.int64)
        self.req_pool.req_to_token[REQ, :MAX_CONTEXT] = self.slots.to(torch.int32)
        runner = types.SimpleNamespace(
            token_to_kv_pool=self.pool,
            req_to_token_pool=self.req_pool,
            device=oracle.DEVICE,
            model_config=types.SimpleNamespace(hf_text_config=suite.config_ns, context_len=MAX_CONTEXT),
        )
        self.backend = QwenSparseAttnBackend(runner)
        self.layer = types.SimpleNamespace(
            layer_id=layer,
            tp_q_head_num=suite.n_q_heads,
            tp_k_head_num=suite.n_kv_heads,
            tp_v_head_num=suite.n_kv_heads,
            head_dim=suite.head_dim,
            qk_head_dim=suite.head_dim,
            v_head_dim=suite.head_dim,
            scaling=suite.scaling,
        )

    def cache_bytes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Byte copies of SGLang's FP8 K and V buffers as [rows, n_kv_heads*head_dim] uint8."""
        k = self.pool.k_buffer[0].view(torch.uint8)
        v = self.pool.v_buffer[0].view(torch.uint8)
        return k.reshape(k.shape[0], -1).clone(), v.reshape(v.shape[0], -1).clone()


def _attn_weights(layer: int) -> dict[str, torch.Tensor]:
    """The real self_attn weights of a full-attention layer, plus the QKVParallelLinear stacking w_qkv."""
    layer_type = oracle.text_config()["layer_types"][layer]
    if layer_type != "full_attention":
        raise oracle.OracleError(f"layer {layer} is {layer_type}, not full_attention")
    p = oracle.layer_prefix(layer) + "self_attn."
    w = {name: oracle.real_tensor(p + proj + ".weight").contiguous() for name, proj in
         (("w_q", "q_proj"), ("w_k", "k_proj"), ("w_v", "v_proj"), ("w_o", "o_proj"), ("q_norm", "q_norm"),
          ("k_norm", "k_norm"))}
    w["w_qkv"] = torch.cat([w["w_q"], w["w_k"], w["w_v"]], dim=0).contiguous()
    return w


def _isolation_row(layer: int, name: str, op: str, native: torch.Tensor, control: torch.Tensor,
                   high_precision: torch.Tensor, control_kernel: str) -> None:
    """One report-only D6 isolation row: `c` = rel-L2(control, FP64), `limit` = max(FLOOR_OP, 2c), `measured` =
    rel-L2(native, control) and the differing elements on identical inputs; `within` = measured <= limit. Asserts
    nothing (tasks/m22d-spec.md C7)."""
    c = oracle.rel_l2(control, high_precision)
    limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
    measured = oracle.rel_l2(native, control)
    differing = int((native != control).sum().item())
    oracle.report_row(ROW, f"{name}/{op}", phase="isolate", layer=layer, op=op, m=1, control=control_kernel, c=c,
                      limit=limit, measured=measured, within=measured <= limit, elements=native.numel(),
                      differing=differing)


class K6AttentionBlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Rotary and pool construction read the published config bags; this is the test-scoped publish that
        # runtime_context.py:968-988 sanctions, carrying the served KV dtype (Q8).
        cls.context_override = get_context().override_server_args(kv_cache_dtype="fp8_e4m3")
        cls.context_override.install()
        tc = oracle.text_config()
        cls.config_ns = types.SimpleNamespace(**tc)
        cls.hidden_size = int(tc["hidden_size"])
        cls.n_q_heads = int(tc["num_attention_heads"])
        cls.n_kv_heads = int(tc["num_key_value_heads"])
        cls.head_dim = int(tc["head_dim"])
        cls.eps = float(tc["rms_norm_eps"])
        cls.scaling = cls.head_dim ** -0.5
        cls.compress_ratio = int(tc["indexer_compress_ratio"])
        cls.token_topk = int(tc["indexer_budget"])
        cls.block_topk = cls.token_topk // cls.compress_ratio

        # Rotary table exactly as Qwen3_5AttentionDecoderLayer builds it (models/qwen3_5.py:935-956).
        rope_theta, rope_scaling = get_rope_config(cls.config_ns)
        if rope_scaling and not ("rope_type" in rope_scaling or "type" in rope_scaling):
            rope_scaling = None
        cls.rope_theta = float(rope_theta)
        cls.rope = get_rope(
            head_size=cls.head_dim,
            rotary_dim=cls.head_dim,
            max_position=int(tc["max_position_embeddings"]),
            rope_scaling=rope_scaling,
            base=rope_theta,
            partial_rotary_factor=float(getattr(cls.config_ns, "partial_rotary_factor", 1.0)),
            is_neox_style=True,
            dtype=torch.bfloat16,
        ).to(oracle.DEVICE)
        cls.rotary_dim = int(cls.rope.rotary_dim)
        cls.cos_sin = cls.rope.cos_sin_cache.contiguous()
        if cls.cos_sin.dtype != torch.float32 or cls.cos_sin.shape[1] != cls.rotary_dim:
            raise oracle.OracleError(f"cos_sin_cache is {cls.cos_sin.dtype} {tuple(cls.cos_sin.shape)}")

        cls.weights = _attn_weights(LAYER)

        # The scale SGLang's serving pool applies to every store from this backend (see module docstring).
        params = inspect.signature(QSATokenToKVPool.set_kv_buffer).parameters
        cls.k_scale = params["k_scale"].default
        cls.v_scale = params["v_scale"].default

    @classmethod
    def tearDownClass(cls):
        cls.context_override.restore()

    # -- SGLang side --------------------------------------------------------------------------------------------

    def _sglang_prepare(self, case: _Case, a: int, b: int, qkv: torch.Tensor | None = None,
                        w: dict[str, torch.Tensor] | None = None):
        """q [T, nq*hd], k/v [T, nkv, hd], gate [T, nq, hd] of tokens a..b-1 on SGLang's CUDA prep path, from
        SGLang's own projection or, for the KV-bytes check, from the given projection bytes `qkv`."""
        w = w or self.weights
        q_size = self.n_q_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        if qkv is None:
            qkv = F.linear(case.hidden[a:b], w["w_qkv"])
        q_gate, k, v = qkv.split([q_size * 2, kv_size, kv_size], dim=-1)
        q, k, gate = fused_qk_gemma_rmsnorm_rope_gate(
            q_gate, k, w["q_norm"], w["k_norm"], self.cos_sin, case.positions[a:b], self.eps,
            self.n_q_heads, self.n_kv_heads, self.head_dim, self.rotary_dim, has_gate=True,
        )
        # RadixAttention.forward views k/v per head before calling the backend.
        return q, k.view(-1, self.n_kv_heads, self.head_dim), v.view(-1, self.n_kv_heads, self.head_dim), gate

    def _k2_qkv(self, case: _Case, a: int, b: int, w: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """[T, q_gate | k | v] projection bytes from the K2 GEMM, the same three calls the native block makes."""
        w = w or self.weights
        parts = []
        for weight in (w["w_q"], w["w_k"], w["w_v"]):
            y = oracle.empty((b - a, weight.shape[0]), torch.bfloat16)
            oracle.call("cere_ktest_gemm_bf16", case.hidden[a:b].contiguous(), weight, b - a, weight.shape[0],
                        self.hidden_size, y)
            parts.append(y)
        return torch.cat(parts, dim=-1).contiguous()

    def _sglang_finish(self, attn: torch.Tensor, gate: torch.Tensor, w: dict[str, torch.Tensor] | None = None):
        w = w or self.weights
        gated = fused_sigmoid_mul(attn, gate, inplace=True)
        return gated, F.linear(gated, w["w_o"])

    def _sglang_store_prefix(self, case: _Case, n: int, w: dict[str, torch.Tensor] | None = None) -> None:
        """Prefix tokens 0..n-1 stored as forward_extend stores them (:1417-1420)."""
        if n == 0:
            return
        _, k, v, _ = self._sglang_prepare(case, 0, n, w=w)
        case.pool.set_kv_buffer(case.layer, case.slots[:n], k, v)

    def _qsa_topk(self, case: _Case, name: str, positions: torch.Tensor, seq_lens: torch.Tensor, decode: bool,
                  seed: int) -> torch.Tensor:
        """SGLang's QSA token selection for the query rows (prefill: qsa_indexer.py:455-491 with
        metadata.build_qsa_row_ranges; decode/verify rows: :523-533), checked against the causal set."""
        rows = positions.numel()
        if decode:
            row_starts = torch.zeros(rows, dtype=torch.int32, device=oracle.DEVICE)
            row_ends = torch.div(seq_lens.to(torch.int32), self.compress_ratio, rounding_mode="floor")
        else:
            row_starts, row_ends, _ = build_qsa_row_ranges(
                sequence_lengths=seq_lens[:1].to(torch.int32),
                query_positions=positions,
                query_sequence_ids=torch.zeros(rows, dtype=torch.int64, device=oracle.DEVICE),
                compress_ratio=self.compress_ratio,
            )
        width = int(row_ends.max().item())
        if width == 0:
            # No complete compressed block yet (select_prefill_tokens:461-467).
            blocks = torch.full((rows, self.block_topk), -1, dtype=torch.int32, device=oracle.DEVICE)
        else:
            logits = torch.randn((rows, width), generator=oracle.generator(seed), device=oracle.DEVICE,
                                 dtype=torch.float32)
            blocks = qsa_fast_topk(logits, row_starts, row_ends, topk=self.block_topk)
        selected = expand_qsa_block_indices(
            blocks, positions, seq_lens, compress_ratio=self.compress_ratio, token_topk=self.token_topk
        )
        oracle.check_count(self, ROW, f"{name}/selection_noncausal_rows",
                           _causal_mismatch_rows(selected, positions), expected=0)
        return selected

    @staticmethod
    def _extend_batch(case: _Case, mode: ForwardMode, a: int, b: int):
        """The ForwardBatch fields forward_extend reads, for one request extending a..b-1."""
        return types.SimpleNamespace(
            forward_mode=mode,
            out_cache_loc=case.slots[a:b],
            extend_seq_lens=torch.tensor([b - a], dtype=torch.int32, device=oracle.DEVICE),
            extend_seq_lens_cpu=[b - a],
            seq_lens=torch.tensor([b], dtype=torch.int64, device=oracle.DEVICE),
            seq_lens_cpu=torch.tensor([b], dtype=torch.int64),
            req_pool_indices=torch.tensor([REQ], dtype=torch.int64, device=oracle.DEVICE),
            positions=case.positions[a:b],
        )

    def _install_paged_metadata(self, case: _Case, positions: torch.Tensor) -> None:
        """QwenSparseAttnMetadata of a decode / target-verify forward (one request, rows at `positions`):
        sequence_lengths = position + 1 per row and the row's request (_metadata_from_forward_batch, non-graph)."""
        rows = positions.numel()
        sequence_lengths = (positions + 1).to(torch.int32)
        token_to_batch_idx = torch.zeros(rows, dtype=torch.int32, device=oracle.DEVICE)
        row_req = torch.full((rows,), REQ, dtype=torch.int64, device=oracle.DEVICE)
        max_length = int(sequence_lengths.max().item())
        case.backend.forward_metadata = QwenSparseAttnMetadata(
            sequence_lengths=sequence_lengths,
            token_to_batch_idx=token_to_batch_idx,
            token_slot_table=case.req_pool.req_to_token[row_req.long(), :max_length],
            indexer_metadata=None,
            row_req_pool_indices=row_req,
        )

    # -- FP64 reference -----------------------------------------------------------------------------------------

    def _reference(self, case: _Case, ctx: int, first_query: int, kv_fp8: bool,
                   w: dict[str, torch.Tensor] | None = None):
        """FP64 gated attention output and block output of the queries first_query..ctx-1 over keys 0..ctx-1."""
        w = w or self.weights
        nq, nkv, hd = self.n_q_heads, self.n_kv_heads, self.head_dim
        h = case.hidden[:ctx].to(F64)
        qg = (h @ w["w_q"].to(F64).T).view(ctx, nq, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:]
        k = (h @ w["w_k"].to(F64).T).view(ctx, nkv, hd)
        v = (h @ w["w_v"].to(F64).T).view(ctx, nkv, hd)
        pos = case.positions[:ctx]
        q = _rope64(_gemma_rms64(q, w["q_norm"], self.eps), pos, self.rope_theta, self.rotary_dim)
        k = _rope64(_gemma_rms64(k, w["k_norm"], self.eps), pos, self.rope_theta, self.rotary_dim)
        if kv_fp8:
            k = _fp8_round64(k, float(self.k_scale))
            v = _fp8_round64(v, float(self.v_scale))
        group = nq // nkv
        kh = k.repeat_interleave(group, dim=1)
        vh = v.repeat_interleave(group, dim=1)
        key_pos = torch.arange(ctx, device=oracle.DEVICE)
        outs = []
        for start in range(first_query, ctx, 256):
            stop = min(start + 256, ctx)
            scores = torch.einsum("rhd,nhd->hrn", q[start:stop], kh) * self.scaling
            future = key_pos[None, :] > torch.arange(start, stop, device=oracle.DEVICE)[:, None]
            probs = torch.softmax(scores.masked_fill(future[None], float("-inf")), dim=-1)
            outs.append(torch.einsum("hrn,nhd->rhd", probs, vh))
        attn = torch.cat(outs, dim=0) * torch.sigmoid(gate[first_query:ctx])
        attn = attn.reshape(ctx - first_query, nq * hd)
        return attn, attn @ w["w_o"].to(F64).T

    # -- native side and checks ---------------------------------------------------------------------------------

    def _native_step(self, case: _Case, a: int, b: int, kv_from_cache: int, k_cache: torch.Tensor,
                     v_cache: torch.Tensor, attn_out: torch.Tensor, out: torch.Tensor,
                     w: dict[str, torch.Tensor] | None = None, p4: bool = False) -> None:
        w = w or self.weights
        args = (
            case.hidden[a:b].contiguous(), w["w_q"], w["w_k"], w["w_v"], w["w_o"], w["q_norm"], w["k_norm"],
            self.cos_sin, self.cos_sin.shape[0], case.positions[a:b].contiguous(), case.slots[:b].contiguous(),
            a, b - a, k_cache, v_cache, k_cache.shape[0], kv_from_cache,
            self.hidden_size, self.n_q_heads, self.n_kv_heads, self.head_dim, self.rotary_dim,
            self.eps, float(self.k_scale), self.scaling, attn_out, out,
        )
        if p4:
            oracle.call("cere_ktest_attn_block_p4", *args)
        else:
            oracle.call("cere_ktest_attn_block", *args)

    def _compare(self, case: _Case, name: str, steps: list[tuple[int, int]], kv_from_cache: int,
                 sg_attn: torch.Tensor, sg_out: torch.Tensor, ref_attn: torch.Tensor, ref_out: torch.Tensor,
                 k_cache: torch.Tensor, v_cache: torch.Tensor, sg_calls: list[tuple[int, int]],
                 p4: bool = False) -> None:
        """Calibrates both stages, runs the native steps on the native cache copy, checks outputs and KV bytes."""
        c_attn = oracle.calibrate(ROW, f"{name}/attn", sg_attn, ref_attn, oracle.FLOOR_OP)
        c_out = oracle.calibrate(ROW, f"{name}/out", sg_out, ref_out, oracle.FLOOR_OP)
        rows = sg_out.shape[0]
        attn_out = oracle.empty((rows, self.n_q_heads * self.head_dim), torch.bfloat16)
        out = oracle.empty((rows, self.hidden_size), torch.bfloat16)
        first = steps[0][0]
        for a, b in steps:
            self._native_step(case, a, b, kv_from_cache, k_cache, v_cache,
                              attn_out[a - first:b - first], out[a - first:b - first], p4=p4)
        limit_attn = oracle.calibrated_limit(c_attn, oracle.FLOOR_OP)
        oracle.check_within(self, ROW, f"{name}/attn", oracle.rel_l2(attn_out, sg_attn), c=c_attn, limit=limit_attn)
        limit_out = oracle.calibrated_limit(c_out, oracle.FLOOR_OP)
        oracle.check_within(self, ROW, f"{name}/out", oracle.rel_l2(out, sg_out), c=c_out, limit=limit_out)
        # Informational (not asserted): the bytes of the run above, whose projections use torch's default
        # allow_bf16_reduced_precision_reduction=True (cuBLAS may pick split-K with BF16 partial sums for some M).
        sg_k, sg_v = case.cache_bytes()
        gemm_m = ",".join(str(b - a) for a, b in sg_calls)
        note = ("torch default allow_bf16_reduced_precision_reduction=True: cuBLAS may use split-K with BF16 "
                "partial sums at this M")
        oracle.report_row(ROW, f"{name}/kv_k_vs_default_flag", phase="info", sglang_gemm_m=gemm_m, note=note,
                          elements=sg_k.numel(), differing=int((k_cache != sg_k).sum().item()))
        oracle.report_row(ROW, f"{name}/kv_v_vs_default_flag", phase="info", sglang_gemm_m=gemm_m, note=note,
                          elements=sg_v.numel(), differing=int((v_cache != sg_v).sum().item()))
        # Informational (not asserted): with the flag False, SGLang's stacked projection still differs from the K2
        # GEMM in FP32 accumulation order at some M (M 1 and 17 on GPU0); record the differing BF16 elements.
        q_width = 2 * self.n_q_heads * self.head_dim
        kv_width = self.n_kv_heads * self.head_dim
        projections = [self._k2_qkv(case, a, b) for a, b in sg_calls]
        flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        try:
            strict = [F.linear(case.hidden[a:b], self.weights["w_qkv"]) for a, b in sg_calls]
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
        diff = [(p != s) for p, s in zip(projections, strict)]
        oracle.report_row(ROW, f"{name}/projection_vs_flag_false", phase="info", sglang_gemm_m=gemm_m,
                          note="torch F.linear(w_qkv), allow_bf16_reduced_precision_reduction=False, vs the K2 "
                               "GEMM: exact FP32 accumulation, order may differ",
                          q_gate_differing=sum(int(d[:, :q_width].sum().item()) for d in diff),
                          k_differing=sum(int(d[:, q_width:q_width + kv_width].sum().item()) for d in diff),
                          v_differing=sum(int(d[:, q_width + kv_width:].sum().item()) for d in diff),
                          q_gate_elements=q_width * rows, kv_elements=kv_width * rows)
        # The bit-exact check measures the store (QK norm, rope, FP8 quantization): SGLang's prep and
        # set_kv_buffer of the same calls on the K2 GEMM's projection bytes (the native block's inputs), overwriting
        # the new tokens' rows of SGLang's pool.
        for (a, b), qkv in zip(sg_calls, projections):
            _, k, v, _ = self._sglang_prepare(case, a, b, qkv)
            case.pool.set_kv_buffer(case.layer, case.slots[a:b], k, v)
        sg_k, sg_v = case.cache_bytes()
        oracle.check_bit_exact(self, ROW, f"{name}/kv_k", k_cache, sg_k)
        oracle.check_bit_exact(self, ROW, f"{name}/kv_v", v_cache, sg_v)

    # -- modes ---------------------------------------------------------------------------------------------------

    def _run_extend_no_prefix(self, length: int, p4: bool = False) -> None:
        name = f"extend_no_prefix/L{length}" + ("/p4" if p4 else "")
        case = _Case(type(self), length, seed=6100 + length)
        k_cache, v_cache = case.cache_bytes()
        q, k, v, gate = self._sglang_prepare(case, 0, length)
        seq_lens = torch.full((length,), length, dtype=torch.int32, device=oracle.DEVICE)
        topk = self._qsa_topk(case, name, case.positions[:length], seq_lens, decode=False, seed=6200 + length)
        batch = self._extend_batch(case, ForwardMode.EXTEND, 0, length)
        attn = case.backend.forward_extend(q, k, v, case.layer, batch, topk_indices=topk)
        sg_attn, sg_out = self._sglang_finish(attn, gate)
        ref_attn, ref_out = self._reference(case, length, 0, kv_fp8=False)
        self._compare(case, name, [(0, length)], 0, sg_attn, sg_out, ref_attn, ref_out, k_cache, v_cache,
                      [(0, length)], p4=p4)

    def _run_extend_prefix_4(self, length: int, n_steps: int) -> None:
        prefix = length - 4 * n_steps
        self._run_extend_prefix(f"extend_prefix_4/L{length}", length, prefix, 4, seed=6300 + length)

    def _run_extend_prefix(self, name: str, length: int, prefix: int, chunk: int, seed: int,
                           p4: bool = False) -> None:
        """Prefix stored, then extends of `chunk` tokens through `_ck` up to `length` (a later prefill chunk)."""
        case = _Case(type(self), length, seed=seed)
        self._sglang_store_prefix(case, prefix)
        k_cache, v_cache = case.cache_bytes()
        steps = [(a, min(a + chunk, length)) for a in range(prefix, length, chunk)]
        attn_parts, out_parts = [], []
        for i, (a, b) in enumerate(steps):
            q, k, v, gate = self._sglang_prepare(case, a, b)
            seq_lens = torch.full((b - a,), b, dtype=torch.int32, device=oracle.DEVICE)
            topk = self._qsa_topk(case, f"{name}/step{i}", case.positions[a:b], seq_lens, decode=False,
                                  seed=seed + 100 + i)
            batch = self._extend_batch(case, ForwardMode.EXTEND, a, b)
            attn = case.backend.forward_extend(q, k, v, case.layer, batch, topk_indices=topk)
            sg_attn, sg_out = self._sglang_finish(attn, gate)
            attn_parts.append(sg_attn)
            out_parts.append(sg_out)
        ref_attn, ref_out = self._reference(case, length, prefix, kv_fp8=True)
        self._compare(case, name, steps, 1, torch.cat(attn_parts), torch.cat(out_parts), ref_attn, ref_out,
                      k_cache, v_cache, steps, p4=p4)

    def _run_decode(self, length: int) -> None:
        name = f"decode/L{length}"
        case = _Case(type(self), length, seed=6500 + length)
        a, b = length - 1, length
        self._sglang_store_prefix(case, a)
        k_cache, v_cache = case.cache_bytes()
        q, k, v, gate = self._sglang_prepare(case, a, b)
        positions = case.positions[a:b]
        topk = self._qsa_topk(case, name, positions, (positions + 1).to(torch.int32), decode=True,
                              seed=6600 + length)
        self._install_paged_metadata(case, positions)
        batch = types.SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            out_cache_loc=case.slots[a:b],
            seq_lens=torch.tensor([b], dtype=torch.int64, device=oracle.DEVICE),
            seq_lens_cpu=torch.tensor([b], dtype=torch.int64),
            req_pool_indices=torch.tensor([REQ], dtype=torch.int64, device=oracle.DEVICE),
            positions=positions,
        )
        attn = case.backend.forward_decode(q, k, v, case.layer, batch, topk_indices=topk)
        oracle.report_row(ROW, name, sglang_paged_kernel="trtllm" if _resolve_trtllm_sparse_decode() else "fa2")
        sg_attn, sg_out = self._sglang_finish(attn, gate)
        ref_attn, ref_out = self._reference(case, length, a, kv_fp8=True)
        self._compare(case, name, [(a, b)], 1, sg_attn, sg_out, ref_attn, ref_out, k_cache, v_cache, [(a, b)])

    def _run_target_verify_4(self, length: int) -> None:
        name = f"target_verify_4/L{length}"
        case = _Case(type(self), length, seed=6700 + length)
        prefix = length - 4
        self._sglang_store_prefix(case, prefix)
        k_cache, v_cache = case.cache_bytes()
        q, k, v, gate = self._sglang_prepare(case, prefix, length)
        positions = case.positions[prefix:length]
        topk = self._qsa_topk(case, name, positions, (positions + 1).to(torch.int32), decode=True,
                              seed=6800 + length)
        self._install_paged_metadata(case, positions)
        batch = self._extend_batch(case, ForwardMode.TARGET_VERIFY, prefix, length)
        attn = case.backend.forward_extend(q, k, v, case.layer, batch, topk_indices=topk)
        oracle.report_row(ROW, name, sglang_paged_kernel="trtllm" if _resolve_trtllm_sparse_decode() else "fa2")
        sg_attn, sg_out = self._sglang_finish(attn, gate)
        ref_attn, ref_out = self._reference(case, length, prefix, kv_fp8=True)
        steps = [(t, t + 1) for t in range(prefix, length)]  # native decodes one token at a time
        self._compare(case, name, steps, 1, sg_attn, sg_out, ref_attn, ref_out, k_cache, v_cache,
                      [(prefix, length)])

    def _isolate_decode(self, layer: int, w: dict[str, torch.Tensor], prefix: int) -> None:
        """D6 isolation rows (tasks/m22d-spec.md C7), report-only: one decode token over a cache of `prefix` tokens
        stored by SGLang's extend path, on `layer`'s real weights; every row feeds both sides identical inputs."""
        name = f"isolate_L{layer}_decode_P{prefix}"
        length = prefix + 1
        case = _Case(type(self), length, seed=9_000 + 100 * layer + prefix, layer=layer)
        a, b = prefix, length
        self._sglang_store_prefix(case, prefix, w=w)
        k_cache, v_cache = case.cache_bytes()
        positions = case.positions[a:b]
        topk = self._qsa_topk(case, name, positions, (positions + 1).to(torch.int32), decode=True,
                              seed=9_500 + 100 * layer + prefix)
        self._install_paged_metadata(case, positions)
        batch = types.SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            out_cache_loc=case.slots[a:b],
            seq_lens=torch.tensor([b], dtype=torch.int64, device=oracle.DEVICE),
            seq_lens_cpu=torch.tensor([b], dtype=torch.int64),
            req_pool_indices=torch.tensor([REQ], dtype=torch.int64, device=oracle.DEVICE),
            positions=positions,
        )
        paged = "trtllm" if _resolve_trtllm_sparse_decode() else "fa2"
        # The control as served: its own BF16 projection, prep, FP8 store, paged decode, gate, o_proj.
        q, k, v, gate = self._sglang_prepare(case, a, b, w=w)
        sg_attn, sg_out = self._sglang_finish(case.backend.forward_decode(q, k, v, case.layer, batch,
                                                                          topk_indices=topk), gate, w=w)
        # The control fed the native block's projection bytes (K2 GEMM): prep, FP8 store of the new token (it
        # overwrites the row the call above stored), paged decode and gate on identical inputs.
        q, k, v, gate = self._sglang_prepare(case, a, b, qkv=self._k2_qkv(case, a, b, w=w), w=w)
        same_attn, _ = self._sglang_finish(case.backend.forward_decode(q, k, v, case.layer, batch,
                                                                       topk_indices=topk), gate, w=w)
        sg_k, sg_v = case.cache_bytes()
        attn_out = oracle.empty((1, self.n_q_heads * self.head_dim), torch.bfloat16)
        out = oracle.empty((1, self.hidden_size), torch.bfloat16)
        self._native_step(case, a, b, 1, k_cache, v_cache, attn_out, out, w=w)
        ref_attn, ref_out = self._reference(case, length, a, kv_fp8=True, w=w)
        control = f"QSA forward_decode ({paged}) over the FP8 pool, kv_len {length}"
        _isolation_row(layer, name, "qkv_prep_fp8_attn_gate_same_projection", attn_out, same_attn, ref_attn,
                       "fused_qk_gemma_rmsnorm_rope_gate + set_kv_buffer + " + control + " + fused_sigmoid_mul")
        o_control = F.linear(attn_out, w["w_o"])
        _isolation_row(layer, name, "o_proj_same_input", out, o_control, attn_out.to(F64) @ w["w_o"].to(F64).T,
                       "F.linear(attn_out, o_proj.weight)")
        _isolation_row(layer, name, "block_out", out, sg_out, ref_out, "the served block: F.linear(w_qkv) + " + control)
        _isolation_row(layer, name, "attn_gated_vs_served", attn_out, sg_attn, ref_attn,
                       "the served block up to the gate: F.linear(w_qkv) + " + control)
        new = case.slots[a].item()
        oracle.report_row(ROW, f"{name}/kv_new_token_bytes", phase="isolate", layer=layer, op="fp8_kv_store", m=1,
                          control="set_kv_buffer of the native projection bytes", elements=2 * k_cache.shape[1],
                          k_differing=int((k_cache[new] != sg_k[new]).sum().item()),
                          v_differing=int((v_cache[new] != sg_v[new]).sum().item()),
                          prefix_differing=int((k_cache != sg_k).sum().item() + (v_cache != sg_v).sum().item()
                                               - (k_cache[new] != sg_k[new]).sum().item()
                                               - (v_cache[new] != sg_v[new]).sum().item()))

    # -- tests ---------------------------------------------------------------------------------------------------

    def test_extend_no_prefix(self):
        oracle.report_row(ROW, "kv_scale", k_scale=float(self.k_scale), v_scale=float(self.v_scale),
                          rotary_dim=self.rotary_dim)
        self.assertEqual((self.k_scale, self.v_scale), (1.0, 1.0), "SGLang's FP8 KV scale is not 1.0 (spec Q8)")
        for length in (1, 517, 2048):
            with self.subTest(L=length):
                self._run_extend_no_prefix(length)

    def test_extend_with_prefix_4_token_steps(self):
        for length in (517, 2048):
            with self.subTest(L=length):
                self._run_extend_prefix_4(length, n_steps=2)

    def test_extend_with_prefix_chunk(self):
        for prefix, chunk in ((480, 37), (2031, 17), (64, 7), (300, 7), (2032, 16)):
            with self.subTest(P=prefix, T=chunk):
                self._run_extend_prefix(f"extend_prefix_chunk/P{prefix}_T{chunk}", prefix + chunk, prefix, chunk,
                                        seed=7100 + prefix + chunk)

    def test_p4_extend(self):
        """P4 (tasks/m23-spec.md): the tensor-core prefill kernel through cere_ktest_attn_block_p4, on the extend
        modes with T >= 17 (BF16 K/V without a prefix; FP8 cache after a prefix, chunks straddling the 16-row
        tiles), against the same calibrated limits and the same KV-byte checks as the default kernel."""
        for length in (517, 2048):
            with self.subTest(L=length):
                self._run_extend_no_prefix(length, p4=True)
        for prefix, chunk in ((480, 37), (2031, 17), (1024, 1024)):
            with self.subTest(P=prefix, T=chunk):
                self._run_extend_prefix(f"extend_prefix_chunk/P{prefix}_T{chunk}/p4", prefix + chunk, prefix,
                                        chunk, seed=7100 + prefix + chunk, p4=True)

    def test_decode(self):
        for length in (1, 517, 2048):
            with self.subTest(L=length):
                self._run_decode(length)

    def test_target_verify_4_rows(self):
        for length in (517, 2048):
            with self.subTest(L=length):
                self._run_target_verify_4(length)

    def test_isolate_decode_every_full_attention_layer(self):
        """D6 isolation (tasks/m22d-spec.md C7): report-only decode rows on every full-attention layer."""
        layers = [i for i, t in enumerate(oracle.text_config()["layer_types"]) if t == "full_attention"]
        for layer in layers:
            w = self.weights if layer == LAYER else _attn_weights(layer)
            for prefix in ISOLATE_PREFIXES:
                with self.subTest(layer=layer, P=prefix):
                    self._isolate_decode(layer, w, prefix)


class K6KvStrideContract(unittest.TestCase):
    """`k6::attention` over BF16 K/V rows: a row stride below the 2 x 256 row width (overlapping rows) or not a
    multiple of 8 is refused before any launch; a wider stride gives the bytes of the contiguous call."""

    TOKENS = 4
    WIDTH = 512

    def _call(self, q, k, v, k_stride: int, v_stride: int):
        out = oracle.empty((self.TOKENS, 24 * 256), torch.bfloat16)
        status = oracle.call_status("cere_ktest_attn_kv_strides", q, k, v, k_stride, v_stride, self.TOKENS, out)
        return status, out

    def test_kv_stride_contract(self):
        gen = oracle.generator(5601)
        q = torch.randn((self.TOKENS, 24 * 256), generator=gen, device=oracle.DEVICE).to(torch.bfloat16)
        k = torch.randn((self.TOKENS, self.WIDTH), generator=gen, device=oracle.DEVICE).to(torch.bfloat16)
        v = torch.randn((self.TOKENS, self.WIDTH), generator=gen, device=oracle.DEVICE).to(torch.bfloat16)
        status, base = self._call(q, k, v, self.WIDTH, self.WIDTH)
        self.assertEqual(status, 0, "contiguous K/V rows must run")
        wide = self.WIDTH + 8
        k_wide = torch.zeros((self.TOKENS, wide), dtype=torch.bfloat16, device=oracle.DEVICE)
        v_wide = torch.zeros((self.TOKENS, wide), dtype=torch.bfloat16, device=oracle.DEVICE)
        k_wide[:, :self.WIDTH] = k
        v_wide[:, :self.WIDTH] = v
        status, strided = self._call(q, k_wide, v_wide, wide, wide)
        self.assertEqual(status, 0, "a wider row stride must run")
        oracle.check_bit_exact(self, ROW, "kv_stride_wide_vs_contiguous", strided, base)
        for name, k_stride, v_stride in (("k_below_width", self.WIDTH - 8, self.WIDTH),
                                         ("v_below_width", self.WIDTH, self.WIDTH - 8),
                                         ("k_quarter", 128, self.WIDTH),
                                         ("k_unaligned", self.WIDTH + 1, self.WIDTH)):
            with self.subTest(stride=name):
                status, _ = self._call(q, k_wide, v_wide, k_stride, v_stride)
                oracle.report_row(ROW, f"kv_stride_{name}_refused", phase="status", status=status,
                                  passed=status != 0)
                self.assertNotEqual(status, 0, f"{name}: the call must be refused")



class K6P4PinnedTail(unittest.TestCase):
    """BeatSglang S7: P4 dense attention is row-partition invariant, so a hot tail (the last t rows of a cold call of
    `cls` rows at pos0 = cls - t, the chunk-0 end state's resume) pinned to cls equals the cold call's rows bit for bit,
    for t below 17 (P4 by the pin) and above."""

    CLASSES = (1_100, 1_200, 2_040)
    TAILS = (1, 5, 12, 16, 17, 48, 100)

    def test_pinned_tail_equals_cold(self):
        gen = oracle.generator(5602)
        for cls in self.CLASSES:
            q = torch.randn((cls, 24 * 256), generator=gen, device=oracle.DEVICE).to(torch.bfloat16)
            k = torch.randn((cls, 512), generator=gen, device=oracle.DEVICE).to(torch.bfloat16)
            v = torch.randn((cls, 512), generator=gen, device=oracle.DEVICE).to(torch.bfloat16)
            cold = oracle.empty((cls, 24 * 256), torch.bfloat16)
            oracle.call("cere_ktest_attn_p4_cls", q, k, v, cls, 0, 0, cold)
            for t in self.TAILS:
                with self.subTest(cls=cls, tail=t):
                    pos0 = cls - t
                    tail = oracle.empty((t, 24 * 256), torch.bfloat16)
                    oracle.call("cere_ktest_attn_p4_cls", q[pos0:].contiguous(), k, v, t, pos0, cls, tail)
                    oracle.check_bit_exact(self, ROW, f"p4_pinned_tail/cls{cls}/t{t}", tail, cold[pos0:])


if __name__ == "__main__":
    unittest.main()
