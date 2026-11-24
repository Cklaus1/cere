"""KV (tasks/m213-spec.md C9, D10, D11, D13): the native vision kernels KV1-KV8 vs the SGLang ops the reference server
runs for the Qwen3-VL vision tower (A4 vision_config: hidden 1152, 16 heads x 72, patch row 1536 = 3*2*16*16,
spatial_merge 2, 48x48 position grid, LayerNorm eps 1e-6) and the M10 three-axis text positions.

Every case reports row "KV" with a case name prefixed kv1_ .. kv8_. Tolerance cases follow the tolerance rule:
`c` = rel-L2(BF16 reference op (the SGLang function/module the contract names, else torch BF16 ops with SGLang's
rounding points) vs an FP32 torch reference on the same BF16 inputs), written to the report before the native call;
limit = max(4e-3, 2c) (`oracle.calibrated_limit(c, oracle.FLOOR_OP)`); measured = rel-L2(native vs the FP32 reference).
Every native case runs twice and the two results must be bit-identical. Inputs are random (oracle.generator seeds),
patch counts cover 1 .. 8,192.

Cases and their references ($S181 = the reference image's site-packages):
- KV1 patch embed, rows {1, 7, 1024, 8192}: SGLang `Conv3dLayer(3, 1152, kernel=stride=[2,16,16])` over the
  [P,3,2,16,16] view (`Qwen3VLVisionPatchEmbed`, `sglang/srt/models/qwen3_vl.py:147`, `sglang/srt/layers/conv.py:207`);
  FP32 reference pixels @ W^T + b.
- KV2 position embedding, grids (t,h,w) (1,2,2), (1,8,8), (1,32,32), (2,16,16), (1,64,128), x NULL and x given:
  align_corners=1 vs `Qwen3VLMoeVisionModel.fast_pos_embed_interpolate_from_list` (`qwen3_vl.py:547`) called unbound
  on a namespace with the BF16 table (BF16 op) and an FP32 copy (FP32 reference); align_corners=0 vs a torch port of
  the same bilinear gather with the indices of SGLang's `_torch_interp_indices` (`qwen3_vl.py:531`), BF16 corner
  products + sum (BF16 op) and FP32 throughout (reference). x given: BF16 op bf16(x + pos) (the vision forward's
  `x += pos_embeds`), reference x + pos in FP32. Table: the real `model.visual.pos_embed.weight` when the checkpoint
  has it, else random.
- KV3 LayerNorm(1152, eps 1e-6), rows {1, 7, 1024, 8192}: BF16 `nn.LayerNorm` vs FP32 `F.layer_norm`.
- KV4 2D vision RoPE, the KV2 grids: `Qwen3VLMoeVisionModel.rot_pos_emb` (`qwen3_vl.py:442`, positions by
  `RotaryPosMixin.rot_pos_ids`, `sglang/srt/models/utils.py:398`) unbound with `get_rope(72, 36, 8192, 10000, neox)`,
  then the vision attention's cat(cos, cos)/cat(sin, sin) and `apply_rotary_pos_emb`
  (`sglang/srt/layers/attention/vision.py:1398-1421`, `rotary_embedding/utils.py`) on q/k viewed [P,16,72];
  FP32 reference `apply_rotary_pos_emb_native_eager` on FP32 q/k.
- KV5 bidirectional attention per frame (seq_len = h*w), the KV2 grids, contiguous and q|k|v-strided rows (stride
  3456): BF16 op torch `scaled_dot_product_attention` per segment; FP32 reference per head and segment (bounded
  memory), scale 72^-0.5.
- KV6 bias + activation, rows {1, 7, 1024, 8192}: mode 0 (n 1152) with and without a residual (and the residual
  aliased as out, bit-equal to the separate output), mode 1 gelu_tanh and mode 2 gelu_erf (n 4304); BF16 op
  h = bf16(acc + b) then torch `gelu`/`+ residual` in BF16 (nn.GELU on the addmm output); FP32 reference without the
  intermediate rounding. A residual with mode 1 returns a nonzero (ARGUMENT) status.
- KV7 splice of image rows into a prefill chunk, bit-exact vs a torch slice copy: image inside the chunk, straddling
  the chunk start, straddling the chunk end, covering the whole chunk, and chunks with no image row (before/after).
- KV8 three-axis positions (rotary 64 of head_dim 256, mrope_section [11,11,10] interleaved, the text config's
  rope_theta): (a) three equal axes: `cere_ktest_rope3` bit-equal to `cere_ktest_rope` and
  `cere_ktest_attn_prologue3` bit-equal to `cere_ktest_attn_prologue` (outputs and FP8 cache rows) with positions =
  axis 0 (the prologue3 call passes a different `positions` buffer, which positions3 must override); (b) distinct
  axes (text, an image grid, text) vs SGLang `MRotaryEmbedding.forward_native` (`rotary_embedding/mrope.py:173`) on
  positions [3, T] with BF16 q/k (BF16 op) and FP32 q/k (FP32 reference: its FP32 `cos_sin_cache` with the
  interleaved section selection), tolerance rule.

Native entry points (oracle.py calling convention: device pointers `void *`, integers `int64_t`, `float` scalars,
the stream last; 0 = success). All tensors contiguous and row-major.

    int cere_ktest_vision_patch_embed(const void *pixels, const void *weight, const void *bias, int64_t P,
                                      void *out, cudaStream_t stream);
        pixels bf16 [P,1536]; weight bf16 [1152,1536]; bias bf16 [1152]; out bf16 [P,1152] = bf16(pixels W^T + b).

    int cere_ktest_vision_pos_embed(const void *table, int64_t num_position_embeddings, int64_t hidden,
                                    int64_t grid_t, int64_t grid_h, int64_t grid_w, int64_t align_corners,
                                    const void *x, void *out, cudaStream_t stream);
        table bf16 [2304,1152]; x bf16 [P,1152] or NULL; out bf16 [P,1152] (pos, or bf16(x + pos)).

    int cere_ktest_vision_layernorm(const void *x, const void *weight, const void *bias, int64_t rows, int64_t dim,
                                    float eps, void *out, cudaStream_t stream);

    int cere_ktest_vision_rope(const void *qkv, int64_t grid_t, int64_t grid_h, int64_t grid_w, void *q_out,
                               void *k_out, cudaStream_t stream);
        qkv bf16 [P,3456] (q | k | v); q_out, k_out bf16 [P,1152].

    int cere_ktest_vision_layernorm_strided(const void *x, int64_t x_stride, const void *weight, const void *bias,
                                            int64_t rows, int64_t dim, float eps, void *out, int64_t y_stride,
                                            cudaStream_t stream);
    int cere_ktest_vision_rope_rows(const void *qkv, int64_t grid_t, int64_t grid_h, int64_t grid_w,
                                    int64_t table_rows, void *q_out, void *k_out, cudaStream_t stream);
        contract checks: the layernorm with explicit row strides, the RoPE with a table of table_rows rows (a row
        stride below dim, or table_rows < max(grid_h, grid_w), must return non-zero without running).

    int cere_ktest_vision_attention(const void *q, const void *k, int64_t qk_stride, const void *v,
                                    int64_t v_stride, int64_t P, int64_t seq_len, void *out, cudaStream_t stream);
        out bf16 [P,1152]; segments of seq_len rows, P % seq_len == 0.

    int cere_ktest_vision_bias_act(const void *acc, const void *bias, const void *residual, int64_t rows,
                                   int64_t n, int64_t mode, void *out, cudaStream_t stream);
        acc f32 [rows,n]; bias bf16 [n]; residual bf16 [rows,n] or NULL (mode 0 only); out bf16 [rows,n].

    int cere_ktest_vision_splice(const void *embeds, int64_t image_rows, int64_t token_offset, int64_t chunk_start,
                                 int64_t n, int64_t hidden, void *mixed, cudaStream_t stream);

    int cere_ktest_rope3(const void *q, const void *k, const void *cos_sin, const void *positions3, int64_t n_tokens,
                         int64_t n_q_heads, int64_t n_kv_heads, int64_t head_dim, int64_t rotary_dim, void *q_out,
                         void *k_out, cudaStream_t stream);
        positions3 int64 [3, n_tokens] (axis a at positions3 + a*n_tokens).

    int cere_ktest_attn_prologue3(const void *q_gate, const void *k, const void *v, const void *q_norm,
                                  const void *k_norm, const void *cos_sin, int64_t n_pos, const void *positions,
                                  const void *positions3, const void *slots, int64_t prefix_len, int64_t T,
                                  void *k_cache, void *v_cache, int64_t n_slots, float eps, void *q_out, void *k_out,
                                  void *gate_out, cudaStream_t stream);

    int cere_ktest_model_create(const cere_model_desc *desc, uint64_t memory_budget_bytes, uint64_t *out_model,
                                cudaStream_t stream);
        (oracle.py) `cere_executor_create` on GPU0 + `cere_model_create`; the status is that of the create.

Device support (tasks/m213-spec.md D9, C7, ledger D 112): the CUDA device's support table holds the A4 model with its
vision tower. A desc whose `vision_config` is built from `$ART/config.json` (`vision_config` + the top-level vision
token ids, `hidden_act` gelu_pytorch_tanh -> CERE_ACT_GELU_TANH, SGLang's LayerNorm eps 1e-6) matches it: with one
`model.visual.*` tensor and an out-of-vocabulary stop token, `cere_model_create` passes the support match and refuses
the stop token (ARGUMENT, nothing allocated); a changed vision field and a visual tensor without `vision_config` are
UNSUPPORTED; the text-only desc is ARGUMENT at the same stop token, as before.
"""
from __future__ import annotations

import contextlib
import ctypes
import types
import unittest

import torch
import torch.nn.functional as F

import oracle
from sglang.srt.layers.conv import Conv3dLayer
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb, get_rope
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_pos_emb_native_eager
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
from sglang.srt.models.utils import RotaryPosMixin
from sglang.srt.runtime_context import get_context

ROW = "KV"
BF16 = torch.bfloat16
F32 = torch.float32
HIDDEN = 1152
HEADS = 16
HEAD_DIM = 72
INTERMEDIATE = 4304
PATCH_ROW = 1536  # in_channels 3 * temporal_patch 2 * patch 16 * 16
IN_CHANNELS, TEMPORAL_PATCH, PATCH = 3, 2, 16
MERGE = 2
GRID_SIDE = 48
NUM_POSITIONS = GRID_SIDE * GRID_SIDE
LN_EPS = 1e-6
OUT_HIDDEN = 2560
POS_EMBED_NAME = "model.visual.pos_embed.weight"
# Patch rows (KV1, KV3, KV6) and image grids (t, h, w) (KV2, KV4, KV5); (1, 64, 128) is 8,192 patches.
ROWS = (1, 7, 1024, 8192)
GRIDS = ((1, 2, 2), (1, 8, 8), (1, 32, 32), (2, 16, 16), (1, 64, 128))
# KV8: rotary table rows the native kernels read (positions stay below it).
ROPE_POSITIONS = 8192
TOKEN_COUNTS = (1, 7, 2048)


@contextlib.contextmanager
def _sglang_build():
    """Module construction as the model loader does it: BF16 default dtype, parameters on GPU0."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(BF16)
    try:
        with torch.device(oracle.DEVICE):
            yield
    finally:
        torch.set_default_dtype(previous)


def _randn(shape, seed: int, scale: float = 1.0, dtype=BF16) -> torch.Tensor:
    g = oracle.generator(seed)
    return (torch.randn(shape, generator=g, device=oracle.DEVICE, dtype=F32) * scale).to(dtype)


def _grid_name(grid) -> str:
    t, h, w = grid
    return f"t{t}h{h}w{w}"


def _check_tolerance(tc: unittest.TestCase, case: str, op_out: torch.Tensor, reference: torch.Tensor,
                     native_run) -> None:
    """Calibrate (report first), run native, check vs the FP32 reference within the limit, and check a second run
    is bit-identical."""
    c = oracle.calibrate(ROW, case, op_out, reference, oracle.FLOOR_OP)
    limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
    native = native_run()
    oracle.check_within(tc, ROW, case, oracle.rel_l2(native, reference), c=c, limit=limit)
    oracle.check_bit_exact(tc, ROW, f"{case}_repeat", native_run(), native)


# -- native entry points ----------------------------------------------------------------------------------------


def _native_patch_embed(pixels, weight, bias) -> torch.Tensor:
    rows = pixels.shape[0]
    out = oracle.empty((rows, HIDDEN), BF16)
    oracle.call("cere_ktest_vision_patch_embed", pixels, weight, bias, rows, out)
    return out


def _native_pos_embed(table, grid, align_corners: int, x) -> torch.Tensor:
    t, h, w = grid
    out = oracle.empty((t * h * w, HIDDEN), BF16)
    oracle.call("cere_ktest_vision_pos_embed", table, NUM_POSITIONS, HIDDEN, t, h, w, align_corners, x, out)
    return out


def _native_layernorm(x, weight, bias) -> torch.Tensor:
    rows, dim = x.shape
    out = oracle.empty((rows, dim), BF16)
    oracle.call("cere_ktest_vision_layernorm", x, weight, bias, rows, dim, LN_EPS, out)
    return out


def _native_vision_rope(qkv, grid):
    t, h, w = grid
    rows = qkv.shape[0]
    q_out = oracle.empty((rows, HIDDEN), BF16)
    k_out = oracle.empty((rows, HIDDEN), BF16)
    oracle.call("cere_ktest_vision_rope", qkv, t, h, w, q_out, k_out)
    return q_out, k_out


def _native_attention(q, k, qk_stride: int, v, v_stride: int, rows: int, seq_len: int) -> torch.Tensor:
    out = oracle.empty((rows, HIDDEN), BF16)
    oracle.call("cere_ktest_vision_attention", q, k, qk_stride, v, v_stride, rows, seq_len, out)
    return out


def _native_bias_act(acc, bias, residual, mode: int, out=None) -> torch.Tensor:
    rows, n = acc.shape
    if out is None:
        out = oracle.empty((rows, n), BF16)
    oracle.call("cere_ktest_vision_bias_act", acc, bias, residual, rows, n, mode, out)
    return out


def _bias_act_status(acc, bias, residual, mode: int) -> int:
    """The status of one `cere_ktest_vision_bias_act` call (a refusal is the expected result, so it is returned, not
    raised as `oracle.call` would)."""
    rows, n = acc.shape
    out = oracle.empty((rows, n), BF16)
    fn = oracle.ktest("cere_ktest_vision_bias_act")
    stream = torch.cuda.current_stream(oracle.DEVICE)
    status = fn(*[oracle._arg(a) for a in (acc, bias, residual, rows, n, mode, out)],
                ctypes.c_void_p(stream.cuda_stream))
    stream.synchronize()
    return int(status)


def _native_splice(embeds, token_offset: int, chunk_start: int, mixed) -> torch.Tensor:
    image_rows, hidden = embeds.shape
    out = mixed.clone()
    oracle.call("cere_ktest_vision_splice", embeds, image_rows, token_offset, chunk_start, out.shape[0], hidden, out)
    return out


def _native_rope_table(n_pos: int, rotary_dim: int, theta: float) -> torch.Tensor:
    table = oracle.empty((n_pos, rotary_dim), F32)
    oracle.call("cere_ktest_rope_table", table, n_pos, rotary_dim, oracle.F64(theta))
    return table


def _native_rope(q, k, table, positions, n_q: int, n_kv: int, head_dim: int, rotary_dim: int):
    q_out = oracle.empty(tuple(q.shape), BF16)
    k_out = oracle.empty(tuple(k.shape), BF16)
    oracle.call("cere_ktest_rope", q, k, table, positions, q.shape[0], n_q, n_kv, head_dim, rotary_dim, q_out, k_out)
    return torch.cat((q_out.reshape(-1), k_out.reshape(-1)))


def _native_rope3(q, k, table, positions3, n_q: int, n_kv: int, head_dim: int, rotary_dim: int):
    q_out = oracle.empty(tuple(q.shape), BF16)
    k_out = oracle.empty(tuple(k.shape), BF16)
    oracle.call("cere_ktest_rope3", q, k, table, positions3, q.shape[0], n_q, n_kv, head_dim, rotary_dim, q_out,
                k_out)
    return torch.cat((q_out.reshape(-1), k_out.reshape(-1)))


def _prologue_buffers(tokens: int, n_q: int, kv_width: int, head_dim: int):
    k_cache = oracle.empty((tokens, kv_width), torch.uint8)
    v_cache = oracle.empty((tokens, kv_width), torch.uint8)
    q_out = oracle.empty((tokens, n_q * head_dim), BF16)
    k_out = oracle.empty((tokens, kv_width), BF16)
    gate_out = oracle.empty((tokens, n_q * head_dim), BF16)
    return k_cache, v_cache, q_out, k_out, gate_out


def _native_prologue(inputs, table, positions, eps: float, n_q: int, head_dim: int):
    q_gate, k, v, q_norm, k_norm = inputs
    tokens, kv_width = k.shape
    k_cache, v_cache, q_out, k_out, gate_out = _prologue_buffers(tokens, n_q, kv_width, head_dim)
    oracle.call("cere_ktest_attn_prologue", q_gate, k, v, q_norm, k_norm, table, table.shape[0], positions, None, 0,
                tokens, k_cache, v_cache, tokens, eps, q_out, k_out, gate_out)
    return q_out, k_out, gate_out, k_cache, v_cache


def _native_prologue3(inputs, table, positions, positions3, eps: float, n_q: int, head_dim: int):
    q_gate, k, v, q_norm, k_norm = inputs
    tokens, kv_width = k.shape
    k_cache, v_cache, q_out, k_out, gate_out = _prologue_buffers(tokens, n_q, kv_width, head_dim)
    oracle.call("cere_ktest_attn_prologue3", q_gate, k, v, q_norm, k_norm, table, table.shape[0], positions,
                positions3, None, 0, tokens, k_cache, v_cache, tokens, eps, q_out, k_out, gate_out)
    return q_out, k_out, gate_out, k_cache, v_cache


# -- references -------------------------------------------------------------------------------------------------


def _pos_namespace(table: torch.Tensor, dtype: torch.dtype, align_corners: bool = True):
    """The attributes `fast_pos_embed_interpolate_from_list` / `_torch_interp_indices` read from the vision model."""
    embedding = torch.nn.Embedding(NUM_POSITIONS, HIDDEN, device=oracle.DEVICE, dtype=dtype)
    embedding.weight.data.copy_(table.to(dtype))
    return types.SimpleNamespace(num_grid_per_side=GRID_SIDE, spatial_merge_size=MERGE, pos_embed=embedding,
                                 device=oracle.DEVICE, dtype=dtype, align_corners=align_corners)


def _pos_embed_half_pixel(ns, grid) -> torch.Tensor:
    """align_corners=0: the from_list bilinear gather with SGLang's `_torch_interp_indices` half-pixel indices, in
    ns.dtype (BF16: BF16 weights and corner products, sum then round, as from_list; FP32: FP32 throughout)."""
    t, h, w = grid
    h_idx = Qwen3VLMoeVisionModel._torch_interp_indices(ns, h, oracle.DEVICE)
    w_idx = Qwen3VLMoeVisionModel._torch_interp_indices(ns, w, oracle.DEVICE)
    h_floor, w_floor = h_idx.to(torch.long), w_idx.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=GRID_SIDE - 1)
    w_ceil = torch.clamp(w_floor + 1, max=GRID_SIDE - 1)
    dh, dw = torch.meshgrid(h_idx - h_floor, w_idx - w_floor, indexing="ij")
    hf, wf = torch.meshgrid(h_floor, w_floor, indexing="ij")
    hc, wc = torch.meshgrid(h_ceil, w_ceil, indexing="ij")
    indices = torch.stack([hf * GRID_SIDE + wf, hf * GRID_SIDE + wc, hc * GRID_SIDE + wf, hc * GRID_SIDE + wc])
    weights = torch.stack([(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw]).to(ns.dtype)
    embeds = ns.pos_embed(indices.reshape(4, -1)) * weights.reshape(4, -1, 1)
    combined = embeds.sum(dim=0)
    combined = combined.reshape(h // MERGE, MERGE, w // MERGE, MERGE, HIDDEN).permute(0, 2, 1, 3, 4)
    return combined.reshape(1, -1, HIDDEN).expand(t, -1, -1).reshape(-1, HIDDEN)


def _attention_fp32(q, k, v, seq_len: int) -> torch.Tensor:
    """FP32 bidirectional attention within each segment of seq_len rows, one head and segment at a time."""
    rows = q.shape[0]
    out = torch.empty((rows, HEADS, HEAD_DIM), dtype=F32, device=oracle.DEVICE)
    q3, k3, v3 = (x.to(F32).reshape(rows, HEADS, HEAD_DIM) for x in (q, k, v))
    scale = HEAD_DIM ** -0.5
    for start in range(0, rows, seq_len):
        seg = slice(start, start + seq_len)
        for head in range(HEADS):
            scores = (q3[seg, head] @ k3[seg, head].T) * scale
            out[seg, head] = torch.softmax(scores, dim=-1) @ v3[seg, head]
    return out.reshape(rows, HIDDEN)


def _attention_bf16(q, k, v, seq_len: int) -> torch.Tensor:
    """BF16 op: torch SDPA per segment on [1, heads, seq, 72] BF16 tensors."""
    rows = q.shape[0]
    parts = []
    for start in range(0, rows, seq_len):
        seg = [x[start:start + seq_len].reshape(seq_len, HEADS, HEAD_DIM).transpose(0, 1)[None] for x in (q, k, v)]
        o = F.scaled_dot_product_attention(*seg, scale=HEAD_DIM ** -0.5)
        parts.append(o[0].transpose(0, 1).reshape(seq_len, HIDDEN))
    return torch.cat(parts)


def _mrope_positions(prefix: int, grid_h: int, grid_w: int, suffix: int, offset: int) -> torch.Tensor:
    """[3, T] Qwen-VL positions: `prefix` text tokens (equal axes), one image of grid_h x grid_w merged tokens (t
    constant, h and w over the grid), `suffix` text tokens continuing past the image's largest position."""
    text = torch.arange(offset, offset + prefix, dtype=torch.int64)
    base = offset + prefix
    hh, ww = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    image = torch.stack([torch.full((grid_h * grid_w,), base, dtype=torch.int64), base + hh.reshape(-1),
                         base + ww.reshape(-1)])
    after = base + max(grid_h, grid_w)
    tail = torch.arange(after, after + suffix, dtype=torch.int64)
    out = torch.cat([text[None].expand(3, -1), image, tail[None].expand(3, -1)], dim=1)
    return out.contiguous().to(oracle.DEVICE)


class KVVision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # SGLang modules read the published config (get_exec) at construction; publish a dummy-boundary one.
        cls.sglang_config = get_context().override_server_args()
        cls.sglang_config.install()

    @classmethod
    def tearDownClass(cls):
        cls.sglang_config.restore()

    # -- KV1 ----------------------------------------------------------------------------------------------------

    def test_kv1_patch_embed(self):
        kernel = [TEMPORAL_PATCH, PATCH, PATCH]
        with _sglang_build():
            proj = Conv3dLayer(IN_CHANNELS, HIDDEN, kernel_size=kernel, stride=kernel, bias=True)
        weight = _randn((HIDDEN, PATCH_ROW), 1000, scale=0.02)
        bias = _randn((HIDDEN,), 1001, scale=0.1)
        proj.weight.data.copy_(weight.reshape(HIDDEN, IN_CHANNELS, TEMPORAL_PATCH, PATCH, PATCH))
        proj.bias.data.copy_(bias)
        for seed, rows in enumerate(ROWS):
            with self.subTest(rows=rows):
                pixels = _randn((rows, PATCH_ROW), 1010 + seed)
                op_out = proj(pixels.view(-1, IN_CHANNELS, TEMPORAL_PATCH, PATCH, PATCH)).view(-1, HIDDEN)
                reference = pixels.to(F32) @ weight.to(F32).T + bias.to(F32)
                _check_tolerance(self, f"kv1_patch_embed_p{rows}", op_out, reference,
                                 lambda: _native_patch_embed(pixels, weight, bias))

    # -- KV2 ----------------------------------------------------------------------------------------------------

    def test_kv2_pos_embed(self):
        if oracle.has_tensor(POS_EMBED_NAME):
            table = oracle.real_tensor(POS_EMBED_NAME).to(BF16).contiguous()
        else:
            table = _randn((NUM_POSITIONS, HIDDEN), 2000, scale=0.5)
        self.assertEqual(tuple(table.shape), (NUM_POSITIONS, HIDDEN))
        ns_bf16 = _pos_namespace(table, BF16)
        ns_fp32 = _pos_namespace(table, F32)
        ns_half_bf16 = _pos_namespace(table, BF16, align_corners=False)
        ns_half_fp32 = _pos_namespace(table, F32, align_corners=False)
        for seed, grid in enumerate(GRIDS):
            t, h, w = grid
            rows = t * h * w
            x = _randn((rows, HIDDEN), 2010 + seed)
            with torch.no_grad():
                refs = {
                    1: (Qwen3VLMoeVisionModel.fast_pos_embed_interpolate_from_list(ns_bf16, [[t, h, w]]),
                        Qwen3VLMoeVisionModel.fast_pos_embed_interpolate_from_list(ns_fp32, [[t, h, w]])),
                    0: (_pos_embed_half_pixel(ns_half_bf16, grid), _pos_embed_half_pixel(ns_half_fp32, grid)),
                }
            for align, (pos_bf16, pos_fp32) in refs.items():
                for with_x in (False, True):
                    with self.subTest(grid=grid, align_corners=align, x=with_x):
                        case = f"kv2_pos_embed_{_grid_name(grid)}_ac{align}_{'x' if with_x else 'nox'}"
                        if with_x:
                            op_out = x + pos_bf16
                            reference = x.to(F32) + pos_fp32
                        else:
                            op_out = pos_bf16
                            reference = pos_fp32
                        x_arg = x if with_x else None
                        _check_tolerance(self, case, op_out.contiguous(), reference.contiguous(),
                                         lambda: _native_pos_embed(table, grid, align, x_arg))

    # -- KV3 ----------------------------------------------------------------------------------------------------

    def test_kv3_layernorm(self):
        with _sglang_build():
            norm = torch.nn.LayerNorm(HIDDEN, eps=LN_EPS)
        weight = (1.0 + _randn((HIDDEN,), 3000, scale=0.1, dtype=F32)).to(BF16)
        bias = _randn((HIDDEN,), 3001, scale=0.1)
        norm.weight.data.copy_(weight)
        norm.bias.data.copy_(bias)
        for seed, rows in enumerate(ROWS):
            with self.subTest(rows=rows):
                x = (_randn((rows, HIDDEN), 3010 + seed, scale=2.0, dtype=F32) + 0.5).to(BF16)
                with torch.no_grad():
                    op_out = norm(x)
                reference = F.layer_norm(x.to(F32), (HIDDEN,), weight.to(F32), bias.to(F32), LN_EPS)
                _check_tolerance(self, f"kv3_layernorm_r{rows}", op_out, reference,
                                 lambda: _native_layernorm(x, weight, bias))

    # -- KV4 ----------------------------------------------------------------------------------------------------

    def test_kv4_vision_rope(self):
        with _sglang_build():
            rotary = get_rope(head_size=HEAD_DIM, rotary_dim=HEAD_DIM // 2, max_position=8192, base=10000.0,
                              is_neox_style=True)
        rotary = rotary.to(oracle.DEVICE)
        ns = types.SimpleNamespace(rot_pos_ids=RotaryPosMixin.rot_pos_ids, spatial_merge_size=MERGE,
                                   device=oracle.DEVICE, rotary_pos_emb=rotary)
        for seed, grid in enumerate(GRIDS):
            t, h, w = grid
            rows = t * h * w
            with self.subTest(grid=grid):
                qkv = _randn((rows, 3 * HIDDEN), 4000 + seed, scale=2.0)
                cos, sin = Qwen3VLMoeVisionModel.rot_pos_emb(ns, [[t, h, w]])
                self.assertEqual(tuple(cos.shape), (rows, HEAD_DIM // 2))
                cos = torch.cat([cos, cos], dim=-1)
                sin = torch.cat([sin, sin], dim=-1)
                q = qkv[:, :HIDDEN].reshape(rows, HEADS, HEAD_DIM)
                k = qkv[:, HIDDEN:2 * HIDDEN].reshape(rows, HEADS, HEAD_DIM)
                q_op, k_op = apply_rotary_pos_emb(q, k, cos, sin)
                q_ref, k_ref = apply_rotary_pos_emb_native_eager(q.to(F32), k.to(F32), cos.to(F32), sin.to(F32))
                op_out = torch.cat((q_op.reshape(-1), k_op.reshape(-1)))
                reference = torch.cat((q_ref.reshape(-1), k_ref.reshape(-1)))

                def run():
                    q_out, k_out = _native_vision_rope(qkv, grid)
                    return torch.cat((q_out.reshape(-1), k_out.reshape(-1)))

                _check_tolerance(self, f"kv4_vision_rope_{_grid_name(grid)}", op_out, reference, run)

    def test_kv_layernorm_rope_contract_refusals(self):
        """A row stride below dim (overlapping rows) and a RoPE table with fewer than max(h, w) rows are refused."""
        weight = _randn((HIDDEN,), 4901)
        bias = _randn((HIDDEN,), 4902)
        for name, x_stride, y_stride, expect_ok in (("contiguous", HIDDEN, HIDDEN, True),
                                                    ("wide", HIDDEN + 8, HIDDEN + 8, True),
                                                    ("x_stride_lt_dim", HIDDEN - 8, HIDDEN, False),
                                                    ("y_stride_lt_dim", HIDDEN, HIDDEN - 8, False)):
            with self.subTest(layernorm=name):
                x = _randn((8, max(x_stride, HIDDEN)), 4903)
                out = oracle.empty((8, max(y_stride, HIDDEN)), BF16)
                status = oracle.call_status("cere_ktest_vision_layernorm_strided", x, x_stride, weight, bias, 8,
                                            HIDDEN, LN_EPS, out, y_stride)
                oracle.report_row(ROW, f"kv3_layernorm_stride_{name}", phase="status", status=status,
                                  passed=(status == 0) == expect_ok)
                self.assertEqual(status == 0, expect_ok, f"layernorm {name}: status {status}")
        grid = (1, 4, 6)
        qkv = _randn((24, 3 * HIDDEN), 4910, scale=2.0)
        q_out = oracle.empty((24, HIDDEN), BF16)
        k_out = oracle.empty((24, HIDDEN), BF16)
        for table_rows, expect_ok in ((6, True), (7, True), (5, False), (4, False), (1, False)):
            with self.subTest(rope_table_rows=table_rows):
                status = oracle.call_status("cere_ktest_vision_rope_rows", qkv, *grid, table_rows, q_out, k_out)
                oracle.report_row(ROW, f"kv4_rope_table_rows_{table_rows}", phase="status", status=status,
                                  passed=(status == 0) == expect_ok)
                self.assertEqual(status == 0, expect_ok, f"rope table rows {table_rows}: status {status}")

    # -- KV5 ----------------------------------------------------------------------------------------------------

    def test_kv5_vision_attention(self):
        for seed, grid in enumerate(GRIDS):
            t, h, w = grid
            rows = t * h * w
            seq_len = h * w
            qkv = _randn((rows, 3 * HIDDEN), 5000 + seed, scale=1.5)
            q, k, v = (qkv[:, i * HIDDEN:(i + 1) * HIDDEN].contiguous() for i in range(3))
            with torch.no_grad():
                op_out = _attention_bf16(q, k, v, seq_len)
            reference = _attention_fp32(q, k, v, seq_len)
            with self.subTest(grid=grid, layout="contiguous"):
                _check_tolerance(self, f"kv5_attention_{_grid_name(grid)}", op_out, reference,
                                 lambda: _native_attention(q, k, HIDDEN, v, HIDDEN, rows, seq_len))
            with self.subTest(grid=grid, layout="qkv_strided"):
                flat = qkv.reshape(-1)
                q_s, k_s, v_s = flat[:], flat[HIDDEN:], flat[2 * HIDDEN:]
                stride = 3 * HIDDEN
                _check_tolerance(self, f"kv5_attention_{_grid_name(grid)}_strided", op_out, reference,
                                 lambda: _native_attention(q_s, k_s, stride, v_s, stride, rows, seq_len))

    # -- KV6 ----------------------------------------------------------------------------------------------------

    def test_kv6_bias_act(self):
        modes = {"bias": (0, HIDDEN, False), "bias_residual": (0, HIDDEN, True), "gelu_tanh": (1, INTERMEDIATE, False),
                 "gelu_erf": (2, INTERMEDIATE, False)}
        for m_seed, (which, (mode, n, with_residual)) in enumerate(modes.items()):
            bias = _randn((n,), 6000 + m_seed, scale=0.5)
            for seed, rows in enumerate(ROWS):
                with self.subTest(mode=which, rows=rows):
                    acc = _randn((rows, n), 6100 + 10 * m_seed + seed, scale=2.0, dtype=F32)
                    residual = _randn((rows, n), 6300 + 10 * m_seed + seed) if with_residual else None
                    h_bf16 = (acc + bias.to(F32)).to(BF16)
                    h_fp32 = acc + bias.to(F32)
                    if mode == 0 and with_residual:
                        op_out = h_bf16 + residual
                        reference = h_fp32 + residual.to(F32)
                    elif mode == 0:
                        op_out = h_bf16
                        reference = h_fp32
                    else:
                        approximate = "tanh" if mode == 1 else "none"
                        op_out = F.gelu(h_bf16, approximate=approximate)
                        reference = F.gelu(h_fp32, approximate=approximate)
                    case = f"kv6_bias_act_{which}_r{rows}"
                    _check_tolerance(self, case, op_out, reference,
                                     lambda: _native_bias_act(acc, bias, residual, mode))
                    if with_residual:
                        aliased = residual.clone()
                        _native_bias_act(acc, bias, aliased, mode, out=aliased)
                        oracle.check_bit_exact(self, ROW, f"{case}_aliased",
                                               aliased, _native_bias_act(acc, bias, residual, mode))
        with self.subTest(mode="residual_gelu_tanh_refused"):
            acc = _randn((7, INTERMEDIATE), 6900, dtype=F32)
            bias = _randn((INTERMEDIATE,), 6901)
            residual = _randn((7, INTERMEDIATE), 6902)
            status = _bias_act_status(acc, bias, residual, 1)
            oracle.report_row(ROW, "kv6_bias_act_residual_mode1_refused", phase="status", status=status,
                              passed=status != 0)
            self.assertNotEqual(status, 0, "residual with mode 1 must be refused (ARGUMENT)")

    # -- KV7 ----------------------------------------------------------------------------------------------------

    def test_kv7_splice(self):
        # (image_rows, token_offset, chunk_start, n)
        cases = {
            "inside": (64, 100, 64, 256),
            "straddle_start": (300, 1900, 2048, 2048),
            "straddle_end": (300, 1900, 0, 2048),
            "covers_chunk": (4096, 1000, 2048, 2048),
            "chunk_before_image": (256, 4096, 0, 2048),
            "chunk_after_image": (256, 100, 2048, 2048),
            "one_row": (1, 5, 5, 1),
        }
        for seed, (which, (image_rows, token_offset, chunk_start, n)) in enumerate(cases.items()):
            with self.subTest(case=which):
                embeds = _randn((image_rows, OUT_HIDDEN), 7000 + seed)
                mixed = _randn((n, OUT_HIDDEN), 7100 + seed)
                reference = mixed.clone()
                lo = max(token_offset, chunk_start)
                hi = min(token_offset + image_rows, chunk_start + n)
                if lo < hi:
                    reference[lo - chunk_start:hi - chunk_start] = embeds[lo - token_offset:hi - token_offset]
                native = _native_splice(embeds, token_offset, chunk_start, mixed)
                oracle.report_row(ROW, f"kv7_splice_{which}", phase="rows", spliced=max(0, hi - lo))
                oracle.check_bit_exact(self, ROW, f"kv7_splice_{which}", native, reference)
                oracle.check_bit_exact(self, ROW, f"kv7_splice_{which}_repeat",
                                       _native_splice(embeds, token_offset, chunk_start, mixed), native)

    # -- KV8 ----------------------------------------------------------------------------------------------------

    def _text_rope(self):
        cfg = oracle.text_config()
        params = cfg["rope_parameters"]
        head_dim = int(cfg["head_dim"])
        rotary_dim = int(head_dim * cfg["partial_rotary_factor"])
        if rotary_dim != 64 or head_dim != 256 or list(params.get("mrope_section", [])) != [11, 11, 10]:
            raise oracle.OracleError(f"text rope ({head_dim}, {rotary_dim}, {params}) is not the spec's")
        return cfg, params, head_dim, rotary_dim

    def test_kv8_rope3_equal_axes(self):
        cfg, params, head_dim, rotary_dim = self._text_rope()
        n_q, n_kv = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"])
        eps = float(cfg["rms_norm_eps"])
        table = _native_rope_table(ROPE_POSITIONS, rotary_dim, float(params["rope_theta"]))
        for seed, tokens in enumerate(TOKEN_COUNTS):
            positions = torch.randint(0, ROPE_POSITIONS, (tokens,), generator=oracle.generator(8000 + seed),
                                      device=oracle.DEVICE, dtype=torch.int64)
            positions3 = positions[None, :].expand(3, tokens).contiguous()
            with self.subTest(op="rope3", tokens=tokens):
                q = _randn((tokens, n_q * head_dim), 8010 + seed, scale=2.0)
                k = _randn((tokens, n_kv * head_dim), 8020 + seed, scale=2.0)
                single = _native_rope(q, k, table, positions, n_q, n_kv, head_dim, rotary_dim)
                three = _native_rope3(q, k, table, positions3, n_q, n_kv, head_dim, rotary_dim)
                oracle.check_bit_exact(self, ROW, f"kv8_rope3_equal_axes_n{tokens}", three, single)
                oracle.check_bit_exact(self, ROW, f"kv8_rope3_equal_axes_n{tokens}_repeat",
                                       _native_rope3(q, k, table, positions3, n_q, n_kv, head_dim, rotary_dim), three)
            with self.subTest(op="attn_prologue3", tokens=tokens):
                inputs = (_randn((tokens, 2 * n_q * head_dim), 8030 + seed, scale=2.0),
                          _randn((tokens, n_kv * head_dim), 8040 + seed, scale=2.0),
                          _randn((tokens, n_kv * head_dim), 8050 + seed),
                          _randn((head_dim,), 8060 + seed, scale=0.1),
                          _randn((head_dim,), 8070 + seed, scale=0.1))
                # positions3 must win over `positions`: pass a different buffer there.
                decoy = torch.flip(positions, dims=(0,)).add_(1).remainder_(ROPE_POSITIONS).contiguous()
                single = _native_prologue(inputs, table, positions, eps, n_q, head_dim)
                three = _native_prologue3(inputs, table, decoy, positions3, eps, n_q, head_dim)
                again = _native_prologue3(inputs, table, decoy, positions3, eps, n_q, head_dim)
                for name, a, b, c in zip(("q", "k", "gate", "k_cache", "v_cache"), three, single, again):
                    oracle.check_bit_exact(self, ROW, f"kv8_prologue3_equal_axes_n{tokens}_{name}", a, b)
                    oracle.check_bit_exact(self, ROW, f"kv8_prologue3_equal_axes_n{tokens}_{name}_repeat", c, a)

    def test_kv8_rope3_distinct_axes(self):
        cfg, params, head_dim, rotary_dim = self._text_rope()
        n_q, n_kv = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"])
        with _sglang_build():
            rope = MRotaryEmbedding(head_dim, rotary_dim, cfg["max_position_embeddings"], params["rope_theta"], True,
                                    BF16, mrope_section=params["mrope_section"],
                                    mrope_interleaved=params.get("mrope_interleaved", False),
                                    mrope_interleaved_glm=params.get("mrope_interleaved_glm", False))
        rope = rope.to(oracle.DEVICE)
        if rope.cos_sin_cache.dtype != F32 or not rope.mrope_interleaved:
            raise oracle.OracleError("the rotary module does not match the spec (FP32 table, interleaved sections)")
        table = rope.cos_sin_cache[:ROPE_POSITIONS].contiguous()
        cases = {
            "image_4x6": _mrope_positions(5, 4, 6, 7, 0),
            "image_32x64": _mrope_positions(3, 32, 64, 5, 1000),
            "image_1x8_far": _mrope_positions(1, 1, 8, 2, ROPE_POSITIONS - 32),
        }
        for seed, (which, positions3) in enumerate(cases.items()):
            tokens = positions3.shape[1]
            with self.subTest(case=which):
                q = _randn((tokens, n_q * head_dim), 8100 + seed, scale=2.0)
                k = _randn((tokens, n_kv * head_dim), 8120 + seed, scale=2.0)
                with torch.no_grad():
                    q_op, k_op = rope.forward_native(positions3, q.clone(), k.clone())
                    q_ref, k_ref = rope.forward_native(positions3, q.to(F32), k.to(F32))
                op_out = torch.cat((q_op.reshape(-1), k_op.reshape(-1)))
                reference = torch.cat((q_ref.reshape(-1), k_ref.reshape(-1)))
                _check_tolerance(self, f"kv8_rope3_distinct_{which}", op_out, reference,
                                 lambda: _native_rope3(q, k, table, positions3, n_q, n_kv, head_dim, rotary_dim))


# -- device support (vision desc) -----------------------------------------------------------------------------------

CERE_ERR_ARGUMENT = -1  # cere_executor.h
CERE_ERR_UNSUPPORTED = -2
# The tensor the support-match descs carry (model.visual.merger.linear_fc2.bias, BF16 [2560]).
VISUAL_TENSOR = ("model.visual.merger.linear_fc2.bias", OUT_HIDDEN)
TEXT_TENSOR = ("model.language_model.norm.weight", OUT_HIDDEN)
# cere_qwen4_exp_vision_config from $ART/config.json (`vision_config` and the top-level token ids).
_vision_config = oracle.vision_config


def _create_status(tensor: tuple[str, int], vision: torch.Tensor | None) -> int:
    """`cere_model_create` status of an A4 desc with one BF16 [n] DEVICE tensor, max_context 2048, one session and a
    stop token equal to vocab_size (out of range: ARGUMENT once the desc matched a support entry, before ALLOCATE)."""
    tc = oracle.text_config()
    config_buf = oracle._a4_config(tc)
    name, n = tensor
    name_buf = torch.tensor(list(name.encode()) + [0], dtype=torch.uint8)
    table_buf, (t,) = oracle.struct_buffer(oracle.TensorDesc)
    t.struct_size = ctypes.sizeof(oracle.TensorDesc)
    t.abi_version = oracle.ABI_VERSION
    t.name = name_buf.data_ptr()
    t.dtype = oracle.DTYPES["BF16"][0]
    t.placement = oracle.PLACEMENT_DEVICE
    t.ndim = 1
    t.shape[0] = n
    t.bytes = n * oracle.DTYPES["BF16"][1]
    stops = torch.tensor([int(tc["vocab_size"])], dtype=torch.int32)
    desc_buf, (desc,) = oracle.struct_buffer(oracle.ModelDesc)
    desc.struct_size = ctypes.sizeof(oracle.ModelDesc)
    desc.abi_version = oracle.ABI_VERSION
    desc.model_kind = oracle.MODEL_QWEN4_EXP_NVFP4
    desc.config = config_buf.data_ptr()
    desc.tensors = table_buf.data_ptr()
    desc.tensor_count = 1
    desc.max_sessions = 1
    desc.max_context_tokens = 2048
    desc.stop_token_ids = stops.data_ptr()
    desc.stop_token_count = 1
    desc.vision_config = None if vision is None else vision.data_ptr()
    out = torch.zeros(1, dtype=torch.int64)
    fn = oracle.ktest("cere_ktest_model_create")
    stream = torch.cuda.current_stream(oracle.DEVICE)
    status = int(fn(*[oracle._arg(a) for a in (desc_buf, 1 << 30, out)], ctypes.c_void_p(stream.cuda_stream)))
    stream.synchronize()
    if int(out[0]) != 0:
        oracle.call("cere_ktest_model_release", int(out[0]))
        raise oracle.OracleError("cere_ktest_model_create accepted a desc with an out-of-vocabulary stop token")
    return status


class KVDeviceSupport(unittest.TestCase):
    def test_kv_device_accepts_the_vision_desc(self):
        cases = {
            "kv_support_vision_desc": (VISUAL_TENSOR, _vision_config(), CERE_ERR_ARGUMENT),
            "kv_support_vision_depth_changed": (VISUAL_TENSOR, _vision_config(depth=26), CERE_ERR_UNSUPPORTED),
            "kv_support_visual_tensor_without_vision_config": (VISUAL_TENSOR, None, CERE_ERR_UNSUPPORTED),
            "kv_support_text_desc": (TEXT_TENSOR, None, CERE_ERR_ARGUMENT),
        }
        for case, (tensor, vision, expected) in cases.items():
            with self.subTest(case=case):
                oracle.check_count(self, ROW, case, _create_status(tensor, vision), expected=expected)


if __name__ == "__main__":
    unittest.main()
