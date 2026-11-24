"""K9 (tasks/m22c-spec.md D3, D8a, D10, C1; tasks/m25-spec.md C5, D6, D8a): the native QSA sparse attention
(kernels/k9_qsa.h) vs SGLang's QSA indexer and sparse-attention reference, on identical inputs.

Inputs: the real BF16 weights of the full-attention layers 3, 23 and 47 (`self_attn.indexer.index_qk_proj`,
`q_layernorm`, `k_layernorm` and `self_attn.{q,k,v}_proj`, `q_norm`, `k_norm`; both are in the checkpoint's
quantization ignore list), the rotary table SGLang builds for the layer (`get_rope` as models/qwen3_5.py:935-956,
the table the indexer reuses), text positions 0 .. L - 1 (every mRoPE axis carries the same position), and the
cases of `_K9Rows._cases`: L in {2,049, 4,500, 18,000, 72,000} on layers 3, 23 and 47 (M2.2c), then each class's
LONG_LENGTHS on layers 3 and 47 (M2.5 D8a at the 200,000 context). Two sources of the layers' block inputs (hidden
states), one TestCase each:
- real (K9QsaReal; m22c D8a, m25 D8a): the native A4 model's own block inputs of the full-attention layers
  (`cere_ktest_forward_attn_inputs`: "model.layers.L.self_attn.input", the HC mix output) over one prefill of each
  REAL_PROMPTS prompt of parity_long_v2 at a 200,000-token context: l09 (72,000 tokens; the M2.2c lengths), l11
  (120,000; L 120,000) and l13 (190,000; L 190,000). The rows at length L are the first L rows of the shortest
  prompt that holds L (a causal forward, so they equal that prompt's first L tokens' own forward). parity_long_v2's
  longest prompt is l13, so no real prompt reaches 199,000: the real long lengths are REAL_LONG_LENGTHS and 199,000
  is a synthetic case only. The prompts arrive as native_dev.py's `cere-k9-prompt-v2` stdin frame (messages
  regenerated from the corpus and digest-checked on the host) and are rendered here with the model's chat template
  (enable_thinking false); each token count must equal the fixture's. Messages, text and ids stay in memory. The
  forwards run once when discovery imports this module (before any suite of the process holds GPU0 memory), one
  model per prompt, released before the next; each prompt's layer inputs are kept on the host until K9QsaReal runs.
- synthetic (K9QsaSynthetic): seeded hidden states at unit RMS (long lengths SYNTHETIC_LONG_LENGTHS), plus the D10
  prologue rows.

Rows (ROW "K9"; case names `<stage>/<source>/L<layer>/n<L>/...`, source `real` or `synthetic`):
- compress: index q prep and compressed keys. `cere_ktest_qsa_compress` vs SGLang's fused
  `qsa_index_q_norm_rope_store` + `qsa_index_k_compress_store` (sglang/kernels/ops/attention/qsa_indexer.py, the
  path QSAIndexer dispatches on CUDA) on the same qk projection, rotary table, mRoPE axis map and norm weights:
  q, compressed keys and the ring of the last incomplete group bit-exact. Twice: one call with identity slots, and
  uneven chunks (4,093 / 4,099 / 1 / 4,096 tokens, cycled) through a page-aligned non-identity slot permutation into
  a key buffer twice the needed size whose other rows are NaN; those rows must stay NaN (spec D3, M25S-01).
- compress_mrope (K9QsaSynthetic; contract C5, M2.13-L): `cere_ktest_qsa_compress_mrope` (indexer_qk_mrope +
  qsa_compress_store_mrope). Equal-axis positions3 (rope_delta 0 and -37) bit-equal to `cere_ktest_qsa_compress` at
  the same 1-D positions; image-like positions3 (MROPE_CASES: merged grid blocks inside text, rope_delta < 0,
  groups straddling image boundaries, chunks starting mid-group so p3_lead > 0) bit-exact vs SGLang's fused prep
  given the same 3-axis positions (q, compressed keys, ring), one call and chunked.
- select: token selection on SGLang's q and compressed keys. SGLang: `QSAIndexer.select_prefill_tokens` with
  `build_qsa_row_ranges` (prefill rows) and `QSAIndexer.select_decode_tokens` (one decode row through a paged
  compressed cache, 64-token pages permuted, the native call addressing the same rows through `slots`). Rows: every
  query row when L <= 4,500, else the last 512 and 512 seeded others. Gates: native
  logits within the tolerance rule of SGLang's `qsa_mqa_prefill` (c = rel-L2(SGLang, FP64) over the visible
  blocks); every row whose 512th and 513th largest FP64 block logits differ by more than the calibrated score
  limit (`calibrated_limit(c, FLOOR_OP)` times the RMS of the row's visible FP64 logits) has exactly SGLang's token
  set; tie-window rows are counted and reported, and their sets differ only in blocks whose FP64 logit lies within
  that limit of the row's 512/513 midpoint; tie rows stay within MAX_TIE_FRACTION of a case's rows or one row, and a
  case admitted only by the one-row floor (a decode row) has exactly SGLang's set on it (m22c Z2AC-05 as revised by
  the owner 2026-10-10, tasks/review-m25.md R1-K9); valid entries are contiguous (stable compaction). The same
  indices come out of a run with logits tiles of 64 rows and of a run through the slot permutation with every other
  compressed key row NaN (bit-exact).
- attention: sparse GQA over SGLang's selection vs `qsa_sparse_attention_reference` (kernel.py:322-345), q/k from
  SGLang's `fused_qk_gemma_rmsnorm_rope_gate` on the real projections: chunk 0's BF16 k/v (logical rows), and the
  FP8 E4M3 cache (scale 1.0, `set_kv_buffer`'s cast) through the slot permutation with every other row NaN (0x7F);
  the latter for the sampled rows, the last 16 rows and the last row (the split-K launches), and bit-exact equal to
  the same call over an identity-slot cache. Tolerance rule (c = rel-L2(SGLang reference, FP64)). Under P4
  (BeatSglang S7) the last 16 rows and the last row also run as a hot tail pinned to the sampled call's class
  (`cere_ktest_qsa_attention_p4_cls`, cls = the sampled rows): bit-exact equal to the sampled call's P4 rows.
- prologue (D10): `cere_ktest_attn_prologue` (K6's prologue, which the K9 path runs) for one token at positions
  2,047, 2,048, 4,096, 72,000 and 199,999 (the last of the 200,000 context, m25 C5) with the 200,000-row rotary
  table and a non-identity slot table: q and k within the
  tolerance rule of SGLang's `fused_qk_gemma_rmsnorm_rope_gate` (c = rel-L2(SGLang, FP64 Gemma norm + NeoX RoPE,
  the K1 reference)), the gate bit-exact, and the cache bit-exact: the token's rows equal the FP8 cast (scale 1.0)
  of the k the call returns and of v, every other row unchanged.

Every tolerance goes through `oracle.calibrated_limit` (tasks/m22-spec.md tolerance rule); everything else is
bit-exact or a count.

C prototypes (libcere_ktest.so; executor/cuda/src/ktest/k9_ktest.cpp, k6_ktest.cu, k8_ktest.cpp; the model
create/put/seal/release entry points are stated in oracle.py):

    int cere_ktest_qsa_compress(const void* qk, int64_t tokens, int64_t pos0, const void* positions,
        const void* slots, const void* q_norm, const void* k_norm, float eps, const void* cos_sin, void* ring,
        void* keys, void* q_idx, cudaStream_t stream);
    int cere_ktest_qsa_compress_mrope(const void* qk, int64_t tokens, int64_t pos0, const void* positions3,
        int64_t p3_stride, int64_t p3_lead, const void* slots, const void* q_norm, const void* k_norm, float eps,
        const void* cos_sin, void* ring, void* keys, void* q_idx, cudaStream_t stream);
    int cere_ktest_qsa_select(const void* q_idx, const void* keys, const void* slots,
        const void* positions_logical, int64_t rows, int64_t seq_len, int64_t scratch_bytes, void* indices,
        void* logits, cudaStream_t stream);
    int cere_ktest_qsa_attention(const void* q, int64_t rows, const void* indices, const void* k, const void* v,
        const void* k_cache, const void* v_cache, const void* slots, float scale, void* out, cudaStream_t stream);
    int cere_ktest_qsa_attention_p4(const void* q, int64_t rows, const void* indices, const void* k, const void* v,
        const void* k_cache, const void* v_cache, const void* slots, float scale, void* out, cudaStream_t stream);
    int cere_ktest_qsa_attention_p4_cls(const void* q, int64_t rows, const void* indices, const void* k,
        const void* v, const void* k_cache, const void* v_cache, const void* slots, float scale, int64_t cls,
        void* out, cudaStream_t stream);
    int cere_ktest_attn_prologue(const void* q_gate, const void* k, const void* v, const void* q_norm,
        const void* k_norm, const void* cos_sin, int64_t n_pos, const void* positions, const void* slots,
        int64_t prefix_len, int64_t T, void* k_cache, void* v_cache, int64_t n_slots, float eps, void* q_out,
        void* k_out, void* gate_out, cudaStream_t stream);
    int cere_ktest_forward_attn_inputs(uint64_t model, const uint32_t *tokens, uint64_t token_count,
        uint64_t prefill_chunk_tokens, uint8_t *names, uint64_t names_capacity, int64_t *shapes,
        uint64_t entry_capacity, void *values, uint64_t values_capacity_bytes, uint64_t *out_counts,
        cudaStream_t stream);
"""
from __future__ import annotations

import gc
import inspect
import json
import math
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

import oracle
from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import fused_qk_gemma_rmsnorm_rope_gate
from sglang.kernels.ops.attention.qsa_indexer import qsa_index_k_compress_store, qsa_index_q_norm_rope_store
from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention_reference
from sglang.srt.layers.attention.qsa.metadata import build_qsa_row_ranges
from sglang.srt.layers.attention.qsa.mqa import qsa_mqa_decode, qsa_mqa_prefill
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.runtime_context import get_context
from sglang.srt.utils.hf_transformers.common import get_rope_config
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

ROW = "K9"
LAYERS = (3, 23, 47)
LENGTHS = (2049, 4500, 18000, 72000)
LONG_LAYERS = (3, 47)  # m25 D8a: the long cases' layers
SYNTHETIC_LONG_LENGTHS = (120_000, 199_000)
REAL_LONG_LENGTHS = (120_000, 190_000)  # 190,000: parity_long_v2's longest prompt (l13), so not 199,000
CONTEXT = 200_000  # the session context bound (m25 design note D6; support.cpp max_context)
PROMPT_FRAME = "cere-k9-prompt-v2"  # native_dev.py's stdin frame of the real-activation rows
PROMPT_SET = "parity_long_v2"
# (id, fixture tokens, captured layers), ascending: a real length L reads the first entry with tokens >= L
REAL_PROMPTS = (("l09", 72000, LAYERS), ("l11", 120_000, LONG_LAYERS), ("l13", 190_000, LONG_LAYERS))
ATTN_INPUT = "model.layers.{}.self_attn.input"  # cere_ktest_forward_attn_inputs entry name per full-attention layer
PROLOGUE_POSITIONS = (2047, 2048, 4096, 72000, 199_999)
CHUNKS = (4093, 4099, 1, 4096)  # chunk sizes of the chunked prep run, cycled: groups and the ring cross chunks
DECODE_PAGE = 64  # token page size of the decode selection's paged compressed cache (SGLang's QSA page, facts F3)
ALL_ROWS_UP_TO = 4500
LAST_ROWS = 512
OTHER_ROWS = 512
ATTN_LAST = 128  # attention rows: the last 128 selection rows and 128 seeded others
ATTN_OTHER = 128
P4_MIN_ROWS = 17  # m23 design note P4: calls of more rows than 16 run the tensor-core kernel (p4_attention.h)
TILE_ROWS = 64  # rows per logits tile of the tiling run
FP8_NAN = 0x7F  # E4M3 NaN: poison of the cache rows a call must not read
F64 = torch.float64
BF16 = torch.bfloat16
INT32_MAX = torch.iinfo(torch.int32).max
HIDDEN_CHUNK = 8192
F32_TIE_ULPS = 4.0  # F32 ulps of the row's largest |logit| added to the selection score window
# Stated bound on the fraction of a case's rows whose 512/513 gap lies inside the window, never below one row: a
# one-row decode case would otherwise fail on any single tie, whatever its sets (M2.5 R1, 190,000-token decode row).
# A case over the fraction but within the one-row floor must have equal sets on its tie row (owner, 2026-10-10).
MAX_TIE_FRACTION = 0.05


def _hidden(length: int, width: int, seed: int) -> torch.Tensor:
    """Seeded rows at unit RMS (the attention block's input is the output of a norm), BF16."""
    g = oracle.generator(seed)
    out = torch.empty((length, width), dtype=BF16, device=oracle.DEVICE)
    for a in range(0, length, HIDDEN_CHUNK):
        b = min(a + HIDDEN_CHUNK, length)
        x = torch.randn((b - a, width), generator=g, device=oracle.DEVICE, dtype=torch.float32)
        out[a:b] = (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True))).to(BF16)
    return out


def _page_slots(length: int, seed: int, page: int = 4) -> tuple[torch.Tensor, int]:
    """Logical position -> cache row keeping each run of `page` positions in one aligned page (page * i + r ->
    page * perm[i] + r), perm a seeded injection of the ceil(length / page) runs into twice as many pages, so each
    group of 4 stays inside one aligned 4-row page. Returns (slots, pages)."""
    runs = (length + page - 1) // page
    pages = 2 * runs
    perm = torch.randperm(pages, generator=oracle.generator(seed), device=oracle.DEVICE)[:runs]
    offsets = torch.arange(page, device=oracle.DEVICE)
    slots = (perm[:, None] * page + offsets[None, :]).reshape(-1)[:length]
    return slots.to(torch.int64).contiguous(), pages


def _chunks(length: int) -> list[int]:
    out, total, i = [], 0, 0
    while total < length:
        n = min(CHUNKS[i % len(CHUNKS)], length - total)
        out.append(n)
        total += n
        i += 1
    return out


def _mrope_positions(segments: tuple[tuple, ...]) -> torch.Tensor:
    """Qwen-VL positions3 [3, L] int64 (axis-major t, h, w) of ("text", n) and ("image", h, w) segments (one temporal
    frame of h x w merged tokens): text advances all axes by one; an image at start s puts token (i, j) at
    (s, s + i, s + j) and the next text starts at max over the image + 1, so the final rope_delta is < 0."""
    cols, nxt = [], 0
    for seg in segments:
        if seg[0] == "text":
            cols += [(nxt + k,) * 3 for k in range(seg[1])]
            nxt += seg[1]
        else:
            _, h, w = seg
            cols += [(nxt, nxt + i, nxt + j) for i in range(h) for j in range(w)]
            nxt += max(h, w)
    return torch.tensor(cols, dtype=torch.int64, device=oracle.DEVICE).t().contiguous()


# (name, segments, chunks) of the mRoPE prep rows (contract C5): a 2x2-merged grid block inside text (rope_delta < 0),
# groups straddling the image boundaries, and chunks starting mid-group so a group's first member sits in the ring
# (p3_lead > 0).
MROPE_CASES = (
    ("image_small", (("text", 6), ("image", 8, 6), ("text", 30)), (6, 45, 33)),
    ("image_two", (("text", 1001), ("image", 32, 40), ("text", 515), ("image", 24, 24), ("text", 123)),
     (1003, 2001, 1, 490)),
)


def _prompt_ids() -> dict[str, list[int]]:
    """The token ids of the stdin frame's REAL_PROMPTS, each rendered with the model's chat template (memory only)."""
    raw = sys.stdin.read()
    if not raw:
        raise oracle.OracleError(f"no {PROMPT_FRAME} frame on stdin (run through native_dev.py test --suite kernels)")
    frame = json.loads(raw)
    head = (frame.get("frame"), frame.get("prompt_set"))
    prompts = frame.get("prompts")
    expected = [(pid, n) for pid, n, _ in REAL_PROMPTS]
    got = ([(p.get("id"), p.get("tokens")) if isinstance(p, dict) else None for p in prompts]
           if isinstance(prompts, list) else None)
    if (head != (PROMPT_FRAME, PROMPT_SET) or got != expected
            or not all(isinstance(p.get("messages"), list) for p in prompts)):
        raise oracle.OracleError(f"stdin frame is {head} with prompts {got}; expected {(PROMPT_FRAME, PROMPT_SET)} "
                                 f"with {expected} and messages")
    tokenizer = get_tokenizer(str(oracle.MODEL_DIR))
    out = {}
    for p in prompts:
        text = tokenizer.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != p["tokens"]:
            raise oracle.OracleError(f"{PROMPT_SET} {p['id']} renders to {len(ids)} tokens, the fixture says "
                                     f"{p['tokens']}")
        out[p["id"]] = ids
    return out


def _attn_inputs(pid: str, ids: list[int], layers: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """The native model's block input [len(ids), hidden] BF16 of each of `layers` over one prefill of `ids` (prompt
    `pid`), on the host (`cere_ktest_forward_attn_inputs` on a model created at CONTEXT and released before
    returning)."""
    tc = oracle.text_config()
    full = [i for i, kind in enumerate(tc["layer_types"]) if kind == "full_attention"]
    hidden, n = int(tc["hidden_size"]), len(ids)
    prompt = torch.tensor(ids, dtype=torch.int32)
    names = torch.zeros(len(full) * 64, dtype=torch.uint8)
    shapes = torch.full((len(full), 3), -1, dtype=torch.int64)
    counts = torch.zeros(3, dtype=torch.int64)
    # The model's plan is checked against the free device memory at create, so the trace buffer comes after it.
    gc.collect()
    torch.cuda.empty_cache()
    free_before, _ = torch.cuda.mem_get_info(oracle.DEVICE)
    handle = oracle.create_model(CONTEXT)
    try:
        free_after, _ = torch.cuda.mem_get_info(oracle.DEVICE)
        oracle.report_row(ROW, f"real/model_create/{pid}", phase="info", free_mib_before=free_before >> 20,
                          free_mib_after=free_after >> 20, trace_mib=(len(full) * n * hidden * 2) >> 20)
        values = oracle.empty((len(full) * n * hidden,), BF16)
        oracle.call("cere_ktest_forward_attn_inputs", handle, prompt, n, 0, names, names.numel(), shapes, len(full),
                    values, values.numel() * values.element_size(), counts)
        torch.cuda.synchronize(oracle.DEVICE)
    finally:
        oracle.release_model(handle)
    entries, name_bytes, value_bytes = (int(v) for v in counts.tolist())
    emitted = [s.decode() for s in bytes(names[:name_bytes].tolist()).split(b"\0")[:-1]]
    expected = [ATTN_INPUT.format(layer) for layer in full]
    if entries != len(full) or emitted != expected or value_bytes != len(full) * n * hidden * 2:
        raise oracle.OracleError(f"forward_attn_inputs wrote {entries} entries ({value_bytes} bytes), names "
                                 f"{emitted}; expected {expected}")
    for i, record in enumerate(shapes.tolist()):
        if record != [2, n, hidden]:
            raise oracle.OracleError(f"{emitted[i]}: shape record {record}, expected {[2, n, hidden]}")
    out = {layer: values[full.index(layer) * n * hidden:(full.index(layer) + 1) * n * hidden].view(n, hidden).cpu()
           for layer in layers}
    del values
    torch.cuda.empty_cache()
    return out


def _query_rows(length: int, seed: int) -> torch.Tensor:
    """Logical positions of the selection rows, ascending."""
    if length <= ALL_ROWS_UP_TO:
        return torch.arange(length, device=oracle.DEVICE, dtype=torch.int64)
    g = oracle.generator(seed)
    others = torch.randperm(length - LAST_ROWS, generator=g, device=oracle.DEVICE)[:OTHER_ROWS]
    last = torch.arange(length - LAST_ROWS, length, device=oracle.DEVICE)
    return torch.cat([torch.sort(others).values, last]).to(torch.int64).contiguous()


def _attention_rows(rows: int, seed: int) -> torch.Tensor:
    """Indices into the selection rows: ATTN_OTHER seeded rows then the last ATTN_LAST, ascending."""
    g = oracle.generator(seed)
    others = torch.randperm(rows - ATTN_LAST, generator=g, device=oracle.DEVICE)[:ATTN_OTHER]
    last = torch.arange(rows - ATTN_LAST, rows, device=oracle.DEVICE)
    return torch.cat([torch.sort(others).values, last]).contiguous()


def _canonical(indices: torch.Tensor) -> torch.Tensor:
    """Each row's selected positions as a sorted set (padding -1 sorts last as INT32_MAX)."""
    x = indices.to(torch.int64)
    return torch.sort(torch.where(x >= 0, x, torch.full_like(x, INT32_MAX)), dim=1).values


def _logits64(q_idx: torch.Tensor, keys: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """FP64 block logits relu(q . k) summed over the index heads / sqrt(head_dim); -inf beyond floor((p + 1) / 4)."""
    rows, _, dim = q_idx.shape
    blocks = keys.shape[0]
    kd = keys.to(F64)
    out = torch.empty((rows, blocks), dtype=F64, device=oracle.DEVICE)
    for a in range(0, rows, 128):
        b = min(a + 128, rows)
        out[a:b] = torch.einsum("mhd,nd->mnh", q_idx[a:b].to(F64), kd).clamp_min(0.0).sum(dim=-1) / math.sqrt(dim)
    visible = torch.div(positions + 1, 4, rounding_mode="floor")
    cols = torch.arange(blocks, device=oracle.DEVICE)
    return out.masked_fill(cols[None, :] >= visible[:, None], float("-inf"))


def _attention64(q: torch.Tensor, k_rows: torch.Tensor, v_rows: torch.Tensor, rows_sel: torch.Tensor,
                 scale: float) -> torch.Tensor:
    """FP64 sparse GQA: q [R, 24, 256]; k_rows/v_rows [N, 2, 256]; rows_sel [R, W] (cache row or -1)."""
    n_rows, n_q, dim = q.shape
    n_kv = k_rows.shape[1]
    group = n_q // n_kv
    out = torch.empty((n_rows, n_q, dim), dtype=F64, device=oracle.DEVICE)
    for a in range(0, n_rows, 16):
        b = min(a + 16, n_rows)
        sel = rows_sel[a:b].to(torch.int64)
        valid = sel >= 0
        idx = sel.clamp_min(0)
        zero = ~valid[:, :, None, None]
        kk = k_rows[idx].to(F64).masked_fill(zero, 0.0)
        vv = v_rows[idx].to(F64).masked_fill(zero, 0.0)
        qq = q[a:b].to(F64).view(b - a, n_kv, group, dim)
        scores = torch.einsum("rgqd,rwgd->rgqw", qq, kk) * scale
        probs = torch.softmax(scores.masked_fill(~valid[:, None, None, :], float("-inf")), dim=-1)
        out[a:b] = torch.einsum("rgqw,rwgd->rgqd", probs, vv).reshape(b - a, n_q, dim)
    return out


def _gemma_rms64(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma RMSNorm in FP64 over the last dim: x * rsqrt(mean(x^2) + eps) * (1 + w)."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + weight.to(F64))


def _rope64(x: torch.Tensor, positions: torch.Tensor, theta: float, rotary_dim: int) -> torch.Tensor:
    """NeoX rotate-half over dims [0, rotary_dim) with FP64 angles (text positions: the full mRoPE)."""
    half = rotary_dim // 2
    inv_freq = theta ** (-torch.arange(0, rotary_dim, 2, dtype=F64, device=x.device) / rotary_dim)
    angles = positions.to(F64)[:, None] * inv_freq[None, :]
    cos = angles.cos()[:, None, :]
    sin = angles.sin()[:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:rotary_dim]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., rotary_dim:]], dim=-1)


# The real block inputs are computed once, when unittest discovery imports this module: the model at CONTEXT needs
# most of GPU0, which is free only before the other kernel suites of the same process run and keep their class-level
# tensors. One model per REAL_PROMPTS prompt, created and released by _attn_inputs before the next (m22c measured the
# 72,000-token trace at 4,218 MiB; at 190,000 it is about 11.1 GiB [INFERENCE: R1 records free_mib_after]). The
# worker's result or exception is held by the future; K9QsaReal reads it (a failure there fails that class, not the
# module, so the synthetic rows still run). Result: {prompt id: {layer: [tokens, hidden] BF16 on the host}}.
def _real_inputs() -> dict[str, dict[int, torch.Tensor]]:
    ids = _prompt_ids()
    return {pid: _attn_inputs(pid, ids[pid], layers) for pid, _, layers in REAL_PROMPTS}


with ThreadPoolExecutor(max_workers=1) as _pool:
    REAL_INPUTS = _pool.submit(_real_inputs)


class _K9Rows:
    """The rows shared by both hidden-state sources; a subclass sets SOURCE and _hidden_source."""

    SOURCE = ""
    LONG_LENGTHS: tuple[int, ...] = ()  # a subclass's m25 D8a lengths, run on LONG_LAYERS

    def _cases(self) -> list[tuple[int, tuple[int, ...]]]:
        """(length, layers) ascending by length: the M2.2c lengths on LAYERS, then LONG_LENGTHS on LONG_LAYERS."""
        return [(length, LAYERS) for length in LENGTHS] + [(length, LONG_LAYERS) for length in self.LONG_LENGTHS]

    @classmethod
    def setUpClass(cls):
        # Rotary construction reads the published config bags (the test-scoped publish of runtime_context.py:968-988,
        # as the K6 suite), carrying the served KV dtype (Q8).
        cls.context_override = get_context().override_server_args(kv_cache_dtype="fp8_e4m3")
        cls.context_override.install()
        # A class cleanup, not tearDownClass: unittest skips tearDownClass when setUpClass fails after this point.
        cls.addClassCleanup(cls.context_override.restore)
        tc = oracle.text_config()
        cls.config_ns = types.SimpleNamespace(**tc)
        cls.hidden_size = int(tc["hidden_size"])
        cls.n_q_heads = int(tc["num_attention_heads"])
        cls.n_kv_heads = int(tc["num_key_value_heads"])
        cls.head_dim = int(tc["head_dim"])
        cls.eps = float(tc["rms_norm_eps"])
        cls.scaling = cls.head_dim ** -0.5
        cls.index_heads = int(tc["indexer_n_heads"])
        cls.index_dim = int(tc["indexer_head_dim"])
        cls.ratio = int(tc["indexer_compress_ratio"])
        cls.token_topk = int(tc["indexer_budget"])
        cls.block_topk = cls.token_topk // cls.ratio
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
            dtype=BF16,
        ).to(oracle.DEVICE)
        cls.rotary_dim = int(cls.rope.rotary_dim)
        cls.cos_sin = cls.rope.cos_sin_cache.contiguous()
        if cls.cos_sin.dtype != torch.float32 or cls.cos_sin.shape[1] != cls.rotary_dim:
            raise oracle.OracleError(f"cos_sin_cache is {cls.cos_sin.dtype} {tuple(cls.cos_sin.shape)}")
        if cls.cos_sin.shape[0] < CONTEXT + 1:
            raise oracle.OracleError(f"cos_sin_cache has {cls.cos_sin.shape[0]} rows, fewer than {CONTEXT + 1}")
        cls.cos_sin_ctx = cls.cos_sin[:CONTEXT].contiguous()
        # The QSAIndexer fields its selection and axis-map methods read (qsa_indexer.py:46-86, :135-158).
        cls.indexer = types.SimpleNamespace(token_topk=cls.token_topk, compress_ratio=cls.ratio,
                                            block_topk=cls.block_topk, rotary_emb=cls.rope, _rope_axis_map_cache=None)
        cls.axis_map = QSAIndexer._rope_axis_map(cls.indexer, oracle.DEVICE)
        params = inspect.signature(QSATokenToKVPool.set_kv_buffer).parameters
        cls.k_scale = params["k_scale"].default
        cls.v_scale = params["v_scale"].default
        if (cls.k_scale, cls.v_scale) not in ((None, None), (1.0, 1.0)):
            raise oracle.OracleError(f"set_kv_buffer default scales {cls.k_scale}, {cls.v_scale} are not 1.0")
        cls.weights = {}
        for layer in LAYERS:
            if tc["layer_types"][layer] != "full_attention":
                raise oracle.OracleError(f"layer {layer} is {tc['layer_types'][layer]}, not full_attention")
            p = oracle.layer_prefix(layer) + "self_attn."
            w = {name: oracle.real_tensor(p + tensor).contiguous() for name, tensor in (
                ("w_idx", "indexer.index_qk_proj.weight"), ("idx_q_norm", "indexer.q_layernorm.weight"),
                ("idx_k_norm", "indexer.k_layernorm.weight"), ("w_q", "q_proj.weight"), ("w_k", "k_proj.weight"),
                ("w_v", "v_proj.weight"), ("q_norm", "q_norm.weight"), ("k_norm", "k_norm.weight"))}
            w["w_qkv"] = torch.cat([w["w_q"], w["w_k"], w["w_v"]], dim=0).contiguous()
            cls.weights[layer] = w
        cls.hidden_cache = {}
        cls.prep_cache = {}

    @classmethod
    def tearDownClass(cls):
        cls.hidden_cache.clear()
        cls.prep_cache.clear()
        torch.cuda.empty_cache()

    # -- shared inputs ------------------------------------------------------------------------------------------

    def _hidden_source(self, length: int) -> dict[int, torch.Tensor]:
        raise NotImplementedError

    def _hidden_for(self, layer: int, length: int) -> torch.Tensor:
        """Block input [L, hidden] BF16 of `layer`; inputs of one length are kept at a time."""
        if length not in self.hidden_cache:
            self.hidden_cache.clear()
            self.prep_cache.clear()
            self.hidden_cache[length] = self._hidden_source(length)
        return self.hidden_cache[length][layer]

    def _sglang_prep(self, layer: int, length: int):
        """qk [L, 640] (F.linear of the real index_qk_proj), SGLang's fused index q [L, 4, 128] and compressed keys
        [floor(L / 4), 128] of one prefill of L tokens (identity state slots)."""
        key = (layer, length)
        if key in self.prep_cache:
            return self.prep_cache[key]
        w = self.weights[layer]
        qk = F.linear(self._hidden_for(layer, length), w["w_idx"]).contiguous()
        positions = torch.arange(length, device=oracle.DEVICE, dtype=torch.int64)
        state = torch.zeros((length, self.index_dim), dtype=BF16, device=oracle.DEVICE)
        rope_positions = torch.zeros((length, 3), dtype=torch.int64, device=oracle.DEVICE)
        q = qsa_index_q_norm_rope_store(
            qk, positions.unsqueeze(0).expand(3, -1).contiguous(), self.cos_sin, self.axis_map, w["idx_q_norm"],
            positions, state, rope_positions, self.index_heads, self.rotary_dim, self.eps, True,
        )
        groups = length // self.ratio
        keys = torch.zeros((groups, self.index_dim), dtype=BF16, device=oracle.DEVICE)
        group_locs = torch.arange(groups * self.ratio, device=oracle.DEVICE, dtype=torch.int32).view(groups, -1)
        write_locs = torch.arange(groups, device=oracle.DEVICE, dtype=torch.int32)
        qsa_index_k_compress_store(
            state, group_locs, rope_positions, self.cos_sin, self.axis_map, w["idx_k_norm"], write_locs, keys,
            self.ratio, self.rotary_dim, self.eps, True,
        )
        self.prep_cache[key] = (qk, q.contiguous(), keys)
        return self.prep_cache[key]

    # -- compress ----------------------------------------------------------------------------------------------

    def _native_compress(self, layer: int, qk: torch.Tensor, chunks: list[int], slots: torch.Tensor | None,
                         key_rows: int, with_positions: bool):
        w = self.weights[layer]
        length = qk.shape[0]
        q_idx = oracle.empty((length, self.index_heads, self.index_dim), BF16)
        keys = oracle.empty((key_rows, self.index_dim), BF16)
        ring = oracle.empty((self.ratio, self.index_dim), BF16)
        pos0 = 0
        for n in chunks:
            positions = (torch.arange(pos0, pos0 + n, device=oracle.DEVICE, dtype=torch.int64)
                         if with_positions else None)
            oracle.call("cere_ktest_qsa_compress", qk[pos0:pos0 + n].contiguous(), n, pos0, positions, slots,
                        w["idx_q_norm"], w["idx_k_norm"], self.eps, self.cos_sin, ring, keys, q_idx[pos0:pos0 + n])
            pos0 += n
        return q_idx, keys, ring

    def _check_ring(self, case: str, qk: torch.Tensor, ring: torch.Tensor) -> None:
        length = qk.shape[0]
        first = length // self.ratio * self.ratio
        if first == length:
            return
        members = torch.arange(first, length, device=oracle.DEVICE)
        k_raw = qk[:, self.index_heads * self.index_dim:]
        oracle.check_bit_exact(self, ROW, f"{case}/ring", ring[members % self.ratio], k_raw[members])

    def test_compress_prefill(self):
        for length, layers in self._cases():
            for layer in layers:
                with self.subTest(layer=layer, length=length):
                    case = f"compress/{self.SOURCE}/L{layer}/n{length}"
                    qk, sg_q, sg_keys = self._sglang_prep(layer, length)
                    groups = sg_keys.shape[0]
                    q_idx, keys, ring = self._native_compress(layer, qk, [length], None, groups, False)
                    oracle.check_bit_exact(self, ROW, f"{case}/one_call/q", q_idx, sg_q)
                    oracle.check_bit_exact(self, ROW, f"{case}/one_call/keys", keys, sg_keys)
                    self._check_ring(f"{case}/one_call", qk, ring)

                    slots, pages = _page_slots(length, seed=9200 + length + layer)
                    q_idx, keys, ring = self._native_compress(layer, qk, _chunks(length), slots, pages, True)
                    oracle.check_bit_exact(self, ROW, f"{case}/chunked_slots/q", q_idx, sg_q)
                    written = slots[torch.arange(groups, device=oracle.DEVICE) * self.ratio] // self.ratio
                    oracle.check_bit_exact(self, ROW, f"{case}/chunked_slots/keys", keys[written], sg_keys)
                    untouched = torch.ones(pages, dtype=torch.bool, device=oracle.DEVICE)
                    untouched[written] = False
                    oracle.check_count(self, ROW, f"{case}/chunked_slots/poison_rows_written",
                                       int((~torch.isnan(keys[untouched])).any(dim=1).sum().item()), expected=0)
                    self._check_ring(f"{case}/chunked_slots", qk, ring)

    # -- select ------------------------------------------------------------------------------------------------

    def _native_select(self, q_idx: torch.Tensor, keys: torch.Tensor, slots: torch.Tensor | None,
                       positions: torch.Tensor | None, rows: int, seq_len: int, scratch_bytes: int,
                       want_logits: bool):
        indices = oracle.empty((rows, self.token_topk + self.ratio - 1), torch.int32)
        logits = oracle.empty((rows, seq_len // self.ratio), torch.float32) if want_logits else None
        oracle.call("cere_ktest_qsa_select", q_idx, keys, slots, positions, rows, seq_len, scratch_bytes, indices,
                    logits)
        return indices, logits

    def _check_sets(self, case: str, native: torch.Tensor, sglang: torch.Tensor, sg_logits: torch.Tensor,
                    ref64: torch.Tensor, positions: torch.Tensor) -> None:
        """D8a's selection gates (module docstring)."""
        rows, blocks = ref64.shape
        nat, sg = _canonical(native), _canonical(sglang)
        differ = (nat != sg).any(dim=1)
        visible = torch.div(positions + 1, self.ratio, rounding_mode="floor")
        has_513 = visible > self.block_topk
        finite = torch.isfinite(ref64)
        # The score window: twice SGLang's own largest |logit - FP64| on the row plus F32_TIE_ULPS F32 ulps of the
        # row's largest |logit| (both sides accumulate in F32; the window does not depend on any native result).
        sg_err = torch.where(finite, (sg_logits[:, :blocks].to(F64) - ref64).abs(), 0.0).amax(dim=1)
        row_max = torch.where(finite, ref64.abs(), 0.0).amax(dim=1)
        window = 2.0 * sg_err + F32_TIE_ULPS * 2.0 ** -23 * row_max
        if blocks > self.block_topk:
            top = torch.topk(ref64, k=self.block_topk + 1, dim=1).values
            gap = top[:, self.block_topk - 1] - top[:, self.block_topk]
            mid = (top[:, self.block_topk - 1] + top[:, self.block_topk]) / 2
        else:
            gap = torch.full((rows,), float("inf"), dtype=F64, device=oracle.DEVICE)
            mid = torch.zeros((rows,), dtype=F64, device=oracle.DEVICE)
        tie = has_513 & (gap <= window)
        valid = native >= 0
        oracle.check_count(self, ROW, f"{case}/noncontiguous_rows",
                           int((valid[:, 1:] & ~valid[:, :-1]).any(dim=1).sum().item()), expected=0)
        # Every row selects min(visible, 512) whole blocks plus the p + 1 - 4 * visible tail positions.
        expected_valid = self.ratio * visible.clamp(max=self.block_topk) + (positions + 1 - self.ratio * visible)
        oracle.check_count(self, ROW, f"{case}/rows_with_wrong_selected_count",
                           int((valid.sum(dim=1) != expected_valid).sum().item()), expected=0)
        oracle.check_count(self, ROW, f"{case}/set_mismatch_outside_tie_window",
                           int((differ & ~tie).sum().item()), expected=0)
        violations = 0
        for r in torch.nonzero(differ & tie).flatten().tolist():
            a, b = nat[r], sg[r]
            only = torch.cat([a[(a < INT32_MAX) & ~torch.isin(a, b)], b[(b < INT32_MAX) & ~torch.isin(b, a)]])
            block = torch.div(only, self.ratio, rounding_mode="floor")
            inside = block < visible[r]
            logit = ref64[r, block.clamp(max=blocks - 1)]
            ok = bool(inside.all().item()) and bool(((logit - mid[r]).abs() <= window[r]).all().item())
            violations += 0 if ok else 1
        tie_rows = int(tie.sum().item())
        oracle.report_row(ROW, f"{case}/tie_window", phase="info", rows=rows, tie_rows=tie_rows,
                          tie_rows_differing=int((differ & tie).sum().item()),
                          max_window=float(window.max().item()), max_sglang_err=float(sg_err.max().item()),
                          max_tie_gap=float(gap[tie].max().item()) if tie_rows else None)
        oracle.check_count(self, ROW, f"{case}/tie_rows_differing_outside_window", violations, expected=0)
        oracle.check_count(self, ROW, f"{case}/tie_rows_above_{MAX_TIE_FRACTION}",
                           int(tie_rows > max(1.0, MAX_TIE_FRACTION * rows)), expected=0)
        oracle.check_count(self, ROW, f"{case}/tie_rows_differing_on_one_row_floor",
                           int(tie_rows > MAX_TIE_FRACTION * rows and bool((differ & tie).any().item())), expected=0)

    def _sglang_select_prefill(self, q_rows: torch.Tensor, keys: torch.Tensor, positions: torch.Tensor,
                               length: int):
        rows = positions.numel()
        row_starts, row_ends, _ = build_qsa_row_ranges(
            sequence_lengths=torch.tensor([length], dtype=torch.int32, device=oracle.DEVICE),
            query_positions=positions,
            query_sequence_ids=torch.zeros(rows, dtype=torch.int64, device=oracle.DEVICE),
            compress_ratio=self.ratio,
        )
        seq_lens = torch.full((rows,), length, dtype=torch.int32, device=oracle.DEVICE)
        keys3 = keys.view(-1, 1, self.index_dim)
        selected = QSAIndexer.select_prefill_tokens(self.indexer, q_rows, keys3, row_starts, row_ends, positions,
                                                    seq_lens)
        logits = qsa_mqa_prefill(q_rows, keys3, row_starts, row_ends)
        return selected, logits

    def _selection(self, layer: int, length: int):
        """The selection rows' positions, q rows and SGLang's prefill selection (shared by select and attention)."""
        key = (layer, length, "selection")
        if key in self.prep_cache:
            return self.prep_cache[key]
        _, sg_q, sg_keys = self._sglang_prep(layer, length)
        positions = _query_rows(length, seed=9300 + length + layer)
        q_rows = sg_q[positions].contiguous()
        selected, logits = self._sglang_select_prefill(q_rows, sg_keys, positions, length)
        self.prep_cache[key] = (positions, q_rows, selected, logits)
        return self.prep_cache[key]

    def test_select_prefill(self):
        for length, layers in self._cases():
            for layer in layers:
                with self.subTest(layer=layer, length=length):
                    case = f"select/{self.SOURCE}/L{layer}/n{length}/prefill"
                    _, _, sg_keys = self._sglang_prep(layer, length)
                    positions, q_rows, sg_sel, sg_logits = self._selection(layer, length)
                    rows = positions.numel()
                    ref64 = _logits64(q_rows, sg_keys, positions)
                    visible = torch.isfinite(ref64)
                    c = oracle.calibrate(ROW, f"{case}/logits", sg_logits[visible], ref64[visible], oracle.FLOOR_OP)
                    limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
                    native, nat_logits = self._native_select(q_rows, sg_keys, None, positions, rows, length, 0, True)
                    oracle.check_within(self, ROW, f"{case}/logits", oracle.rel_l2(nat_logits[visible],
                                        sg_logits[visible]), c=c, limit=limit)
                    oracle.check_count(self, ROW, f"{case}/logits_masked_not_neg_inf",
                                       int((nat_logits[~visible] != float("-inf")).sum().item()), expected=0)
                    self._check_sets(case, native, sg_sel, sg_logits, ref64, positions)

                    tile_bytes = TILE_ROWS * (length // self.ratio) * 4
                    tiled, _ = self._native_select(q_rows, sg_keys, None, positions, rows, length, tile_bytes, False)
                    oracle.check_bit_exact(self, ROW, f"{case}/tiles_of_{TILE_ROWS}_rows", tiled, native)

                    slots, pages = _page_slots(length, seed=9400 + length + layer)
                    keys_perm = oracle.empty((pages, self.index_dim), BF16)
                    groups = sg_keys.shape[0]
                    keys_perm[slots[torch.arange(groups, device=oracle.DEVICE) * self.ratio] // self.ratio] = sg_keys
                    permuted, _ = self._native_select(q_rows, keys_perm, slots, positions, rows, length, 0, False)
                    oracle.check_bit_exact(self, ROW, f"{case}/slots_poisoned", permuted, native)

    def test_select_decode(self):
        for length, layers in self._cases():
            for layer in layers:
                with self.subTest(layer=layer, length=length):
                    case = f"select/{self.SOURCE}/L{layer}/n{length}/decode"
                    _, sg_q, sg_keys = self._sglang_prep(layer, length)
                    groups = sg_keys.shape[0]
                    # SGLang's paged compressed cache: token pages of DECODE_PAGE, compressed pages of
                    # DECODE_PAGE / 4 rows (TileLang's decode MQA takes 8/16/32/64-row pages).
                    cpage = DECODE_PAGE // self.ratio
                    slots, pages = _page_slots(length, seed=9500 + length + layer, page=DECODE_PAGE)
                    row_of_block = slots[torch.arange(groups, device=oracle.DEVICE) * self.ratio] // self.ratio
                    used = torch.unique(row_of_block // cpage)
                    keys_perm = oracle.empty((pages * cpage, self.index_dim), BF16)
                    keys_perm.view(pages, cpage, -1)[used] = 0  # the used pages' rows past the last block
                    keys_perm[row_of_block] = sg_keys
                    cache = keys_perm.view(pages, cpage, 1, self.index_dim)
                    page_table = (row_of_block[::cpage] // cpage).to(torch.int32).view(1, -1).contiguous()
                    lengths = torch.tensor([groups], dtype=torch.int32, device=oracle.DEVICE)
                    q_row = sg_q[length - 1:length].contiguous()
                    position = torch.tensor([length - 1], dtype=torch.int64, device=oracle.DEVICE)
                    sg_sel = QSAIndexer.select_decode_tokens(
                        self.indexer, q_row, cache, page_table, lengths, groups, position,
                        torch.tensor([length], dtype=torch.int32, device=oracle.DEVICE),
                    )
                    ref64 = _logits64(q_row, sg_keys, position)
                    sg_logits = qsa_mqa_decode(q_row, cache, page_table, lengths, groups)
                    # Reported calibration of SGLang's decode logits; the selection window is per row (_check_sets).
                    oracle.calibrate(ROW, f"{case}/logits", sg_logits.reshape(-1), ref64.reshape(-1), oracle.FLOOR_OP)
                    native, _ = self._native_select(q_row, keys_perm, slots, None, 1, length, 0, False)
                    self._check_sets(case, native, sg_sel, sg_logits, ref64, position)

    # -- attention ---------------------------------------------------------------------------------------------

    def _attention_inputs(self, layer: int, length: int, positions: torch.Tensor):
        """k (normed, rotated) and v [L, 2 * 256] of every token and q [R, 24 * 256] of the query rows, from SGLang's
        fused prep on the real projections."""
        w = self.weights[layer]
        hidden = self._hidden_for(layer, length)
        q_size = self.n_q_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        all_positions = torch.arange(length, device=oracle.DEVICE, dtype=torch.int64)
        k_all = torch.empty((length, kv_size), dtype=BF16, device=oracle.DEVICE)
        v_all = torch.empty((length, kv_size), dtype=BF16, device=oracle.DEVICE)
        for a in range(0, length, HIDDEN_CHUNK):
            b = min(a + HIDDEN_CHUNK, length)
            q_gate, k, v = F.linear(hidden[a:b], w["w_qkv"]).split([2 * q_size, kv_size, kv_size], dim=-1)
            _, k_rot, _ = fused_qk_gemma_rmsnorm_rope_gate(
                q_gate, k, w["q_norm"], w["k_norm"], self.cos_sin, all_positions[a:b], self.eps, self.n_q_heads,
                self.n_kv_heads, self.head_dim, self.rotary_dim, has_gate=True,
            )
            k_all[a:b] = k_rot.reshape(b - a, kv_size)
            v_all[a:b] = v
        q_gate, k, _ = F.linear(hidden[positions], w["w_qkv"]).split([2 * q_size, kv_size, kv_size], dim=-1)
        q, _, _ = fused_qk_gemma_rmsnorm_rope_gate(
            q_gate, k, w["q_norm"], w["k_norm"], self.cos_sin, positions, self.eps, self.n_q_heads, self.n_kv_heads,
            self.head_dim, self.rotary_dim, has_gate=True,
        )
        return q.reshape(positions.numel(), q_size).contiguous(), k_all, v_all

    def _attention_case(self, case: str, q: torch.Tensor, sel: torch.Tensor, k_rows: torch.Tensor,
                        v_rows: torch.Tensor, ref_sel: torch.Tensor, args: tuple) -> torch.Tensor:
        """Calibrates SGLang's reference over cache rows k_rows/v_rows [N, 2, 256] (ref_sel: cache row or -1), runs
        the native call with `args` (k, v, k_cache, v_cache, slots) and checks it; a call of >= 17 rows also runs
        the P4 tensor-core entry (tasks/m23-spec.md P4) against the same limit (case suffix /p4)."""
        rows = q.shape[0]
        q3 = q.view(rows, self.n_q_heads, self.head_dim)
        sg = qsa_sparse_attention_reference(q3, k_rows, v_rows, ref_sel, self.scaling).reshape(rows, -1)
        ref = _attention64(q3, k_rows, v_rows, ref_sel, self.scaling).reshape(rows, -1)
        c = oracle.calibrate(ROW, case, sg, ref, oracle.FLOOR_OP)
        out = oracle.empty((rows, self.n_q_heads * self.head_dim), BF16)
        oracle.call("cere_ktest_qsa_attention", q, rows, sel, *args, self.scaling, out)
        limit = oracle.calibrated_limit(c, oracle.FLOOR_OP)
        oracle.check_within(self, ROW, case, oracle.rel_l2(out, sg), c=c, limit=limit)
        if rows >= P4_MIN_ROWS:
            p4 = oracle.empty((rows, self.n_q_heads * self.head_dim), BF16)
            oracle.call("cere_ktest_qsa_attention_p4", q, rows, sel, *args, self.scaling, p4)
            oracle.check_within(self, ROW, f"{case}/p4", oracle.rel_l2(p4, sg), c=c, limit=limit)
            again = oracle.empty((rows, self.n_q_heads * self.head_dim), BF16)
            oracle.call("cere_ktest_qsa_attention_p4", q, rows, sel, *args, self.scaling, again)
            oracle.check_bit_exact(self, ROW, f"{case}/p4/deterministic", again, p4)
        return out

    def test_attention(self):
        kv = self.n_kv_heads * self.head_dim
        for length, layers in self._cases():
            for layer in layers:
                with self.subTest(layer=layer, length=length):
                    case = f"attention/{self.SOURCE}/L{layer}/n{length}"
                    positions, _, sg_sel, _ = self._selection(layer, length)
                    pick = _attention_rows(positions.numel(), seed=9600 + length + layer)
                    positions = positions[pick].contiguous()
                    sel = sg_sel[pick].contiguous()
                    q, k_all, v_all = self._attention_inputs(layer, length, positions)
                    k3 = k_all.view(length, self.n_kv_heads, self.head_dim)
                    v3 = v_all.view(length, self.n_kv_heads, self.head_dim)
                    self._attention_case(f"{case}/bf16_kv", q, sel, k3, v3, sel, (k_all, v_all, None, None, None))

                    # FP8 cache (set_kv_buffer: k / 1.0 -> E4M3) at permuted rows, every other row NaN.
                    slots, pages = _page_slots(length, seed=9700 + length + layer)
                    k8 = k_all.to(torch.float8_e4m3fn).view(torch.uint8)
                    v8 = v_all.to(torch.float8_e4m3fn).view(torch.uint8)
                    k_cache = torch.full((pages * self.ratio, kv), FP8_NAN, dtype=torch.uint8, device=oracle.DEVICE)
                    v_cache = torch.full_like(k_cache, FP8_NAN)
                    k_cache[slots] = k8
                    v_cache[slots] = v8
                    k_deq = k_cache.view(torch.float8_e4m3fn).to(BF16).view(-1, self.n_kv_heads, self.head_dim)
                    v_deq = v_cache.view(torch.float8_e4m3fn).to(BF16).view(-1, self.n_kv_heads, self.head_dim)
                    sel64 = sel.to(torch.int64)
                    rows_sel = torch.where(sel64 >= 0, slots[sel64.clamp_min(0)], -1).to(torch.int32)
                    full_p4 = None  # the sampled call's P4 rows (its class pins the tails below)
                    for name, part in (("fp8_cache", slice(None)), ("fp8_cache_last16", slice(-16, None)),
                                       ("fp8_cache_last1", slice(-1, None))):
                        q_p, sel_p = q[part].contiguous(), sel[part].contiguous()
                        out = self._attention_case(f"{case}/{name}", q_p, sel_p, k_deq, v_deq,
                                                   rows_sel[part].contiguous(), (None, None, k_cache, v_cache, slots))
                        identity = oracle.empty(tuple(out.shape), BF16)
                        oracle.call("cere_ktest_qsa_attention", q_p, q_p.shape[0], sel_p, None, None,
                                    k8.contiguous(), v8.contiguous(), None, self.scaling, identity)
                        oracle.check_bit_exact(self, ROW, f"{case}/{name}/slots_poisoned_vs_identity", out, identity)
                        if q_p.shape[0] >= P4_MIN_ROWS:
                            p4 = oracle.empty(tuple(out.shape), BF16)
                            oracle.call("cere_ktest_qsa_attention_p4", q_p, q_p.shape[0], sel_p, None, None, k_cache,
                                        v_cache, slots, self.scaling, p4)
                            p4_identity = oracle.empty(tuple(out.shape), BF16)
                            oracle.call("cere_ktest_qsa_attention_p4", q_p, q_p.shape[0], sel_p, None, None,
                                        k8.contiguous(), v8.contiguous(), None, self.scaling, p4_identity)
                            oracle.check_bit_exact(self, ROW, f"{case}/{name}/p4/slots_poisoned_vs_identity", p4,
                                                   p4_identity)
                            if full_p4 is None:
                                full_p4 = p4
                        elif full_p4 is not None:
                            # BeatSglang S7: a hot tail of fewer than 17 rows pinned to the cold call's class runs P4
                            # and returns that call's rows bit for bit (it was CERE_ERR_UNSUPPORTED before).
                            tail = oracle.empty(tuple(out.shape), BF16)
                            oracle.call("cere_ktest_qsa_attention_p4_cls", q_p, q_p.shape[0], sel_p, None, None,
                                        k_cache, v_cache, slots, self.scaling, q.shape[0], tail)
                            oracle.check_bit_exact(self, ROW, f"{case}/{name}/p4/pinned_tail_equals_cold", tail,
                                                   full_p4[part].contiguous())


class K9QsaReal(_K9Rows, unittest.TestCase):
    SOURCE = "real"
    LONG_LENGTHS = REAL_LONG_LENGTHS

    @classmethod
    def setUpClass(cls):
        cls.inputs = REAL_INPUTS.result()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.inputs.clear()
        super().tearDownClass()

    def _hidden_source(self, length: int) -> dict[int, torch.Tensor]:
        # the first L rows of the shortest REAL_PROMPTS prompt that holds L, on its captured layers
        pid = next(pid for pid, tokens, _ in REAL_PROMPTS if tokens >= length)
        return {layer: t[:length].to(oracle.DEVICE) for layer, t in self.inputs[pid].items()}


class K9QsaSynthetic(_K9Rows, unittest.TestCase):
    SOURCE = "synthetic"
    LONG_LENGTHS = SYNTHETIC_LONG_LENGTHS

    def _hidden_source(self, length: int) -> dict[int, torch.Tensor]:
        hidden = _hidden(length, self.hidden_size, seed=9100 + length)
        return {layer: hidden for layer in LAYERS}

    # -- mRoPE prep (contract C5: image sessions' prompt rows) -------------------------------------------------

    def _native_compress_mrope(self, layer: int, qk: torch.Tensor, p3: torch.Tensor, chunks) -> tuple:
        """cere_ktest_qsa_compress_mrope over `chunks` (identity slots): each call gets the positions3 columns of
        logical positions pos0 - pos0 % 4 .. pos0 + n - 1 (p3_lead = pos0 % 4)."""
        w = self.weights[layer]
        length = qk.shape[0]
        q_idx = oracle.empty((length, self.index_heads, self.index_dim), BF16)
        keys = oracle.empty((length // self.ratio, self.index_dim), BF16)
        ring = oracle.empty((self.ratio, self.index_dim), BF16)
        pos0 = 0
        for n in chunks:
            lead = pos0 % self.ratio
            cols = p3[:, pos0 - lead:pos0 + n].contiguous()
            oracle.call("cere_ktest_qsa_compress_mrope", qk[pos0:pos0 + n].contiguous(), n, pos0, cols, lead + n,
                        lead, None, w["idx_q_norm"], w["idx_k_norm"], self.eps, self.cos_sin, ring, keys,
                        q_idx[pos0:pos0 + n])
            pos0 += n
        return q_idx, keys, ring

    def test_compress_mrope_equal_axes(self):
        """(a) equal-axis positions3 (offset by a negative rope_delta) give cere_ktest_qsa_compress's bytes."""
        layer, length = LAYERS[0], 4500
        qk = F.linear(self._hidden_for(layer, length), self.weights[layer]["w_idx"]).contiguous()
        for delta in (0, -37):
            with self.subTest(delta=delta):
                case = f"compress_mrope/{self.SOURCE}/L{layer}/n{length}/equal_axes_d{-delta}"
                pos = torch.arange(length, device=oracle.DEVICE, dtype=torch.int64) + 40 + delta
                p3 = pos.unsqueeze(0).expand(3, -1).contiguous()
                q_m, keys_m, ring_m = self._native_compress_mrope(layer, qk, p3, _chunks(length))
                w = self.weights[layer]
                q_idx = oracle.empty((length, self.index_heads, self.index_dim), BF16)
                keys = oracle.empty((length // self.ratio, self.index_dim), BF16)
                ring = oracle.empty((self.ratio, self.index_dim), BF16)
                pos0 = 0
                for n in _chunks(length):
                    oracle.call("cere_ktest_qsa_compress", qk[pos0:pos0 + n].contiguous(), n, pos0,
                                pos[pos0:pos0 + n].contiguous(), None, w["idx_q_norm"], w["idx_k_norm"], self.eps,
                                self.cos_sin, ring, keys, q_idx[pos0:pos0 + n])
                    pos0 += n
                oracle.check_bit_exact(self, ROW, f"{case}/q", q_m, q_idx)
                oracle.check_bit_exact(self, ROW, f"{case}/keys", keys_m, keys)
                oracle.check_bit_exact(self, ROW, f"{case}/ring", ring_m, ring)

    def test_compress_mrope_image(self):
        """(b) image-like positions3 vs SGLang's fused prep with the same mRoPE positions (rope_positions)."""
        for layer in LONG_LAYERS:
            for name, segments, chunks in MROPE_CASES:
                with self.subTest(layer=layer, case=name):
                    p3 = _mrope_positions(segments)
                    length = p3.shape[1]
                    case = f"compress_mrope/{self.SOURCE}/L{layer}/n{length}/{name}"
                    w = self.weights[layer]
                    qk = F.linear(_hidden(length, self.hidden_size, seed=9300 + length + layer),
                                  w["w_idx"]).contiguous()
                    locs = torch.arange(length, device=oracle.DEVICE, dtype=torch.int64)
                    state = torch.zeros((length, self.index_dim), dtype=BF16, device=oracle.DEVICE)
                    rope_positions = torch.zeros((length, 3), dtype=torch.int64, device=oracle.DEVICE)
                    sg_q = qsa_index_q_norm_rope_store(
                        qk, p3, self.cos_sin, self.axis_map, w["idx_q_norm"], locs, state, rope_positions,
                        self.index_heads, self.rotary_dim, self.eps, True,
                    ).contiguous()
                    groups = length // self.ratio
                    sg_keys = torch.zeros((groups, self.index_dim), dtype=BF16, device=oracle.DEVICE)
                    group_locs = torch.arange(groups * self.ratio, device=oracle.DEVICE,
                                              dtype=torch.int32).view(groups, -1)
                    write_locs = torch.arange(groups, device=oracle.DEVICE, dtype=torch.int32)
                    qsa_index_k_compress_store(
                        state, group_locs, rope_positions, self.cos_sin, self.axis_map, w["idx_k_norm"],
                        write_locs, sg_keys, self.ratio, self.rotary_dim, self.eps, True,
                    )
                    for label, ch in (("one_call", [length]), ("chunked", list(chunks))):
                        q_idx, keys, ring = self._native_compress_mrope(layer, qk, p3, ch)
                        oracle.check_bit_exact(self, ROW, f"{case}/{label}/q", q_idx, sg_q)
                        oracle.check_bit_exact(self, ROW, f"{case}/{label}/keys", keys, sg_keys)
                        self._check_ring(f"{case}/{label}", qk, ring)

    # -- D10 prologue (one synthetic token per position) -------------------------------------------------------

    def test_prologue_positions(self):
        q_size = self.n_q_heads * self.head_dim
        kv = self.n_kv_heads * self.head_dim
        for layer in LAYERS:
            w = self.weights[layer]
            for position in PROLOGUE_POSITIONS:
                with self.subTest(layer=layer, position=position):
                    case = f"prologue/{self.SOURCE}/L{layer}/pos{position}"
                    hidden = _hidden(1, self.hidden_size, seed=9800 + position + layer)
                    q_gate, k, v = (t.contiguous() for t in F.linear(hidden, w["w_qkv"]).split(
                        [2 * q_size, kv, kv], dim=-1))
                    positions = torch.tensor([position], dtype=torch.int64, device=oracle.DEVICE)
                    sg_q, sg_k, sg_gate = fused_qk_gemma_rmsnorm_rope_gate(
                        q_gate.clone(), k.clone(), w["q_norm"], w["k_norm"], self.cos_sin, positions, self.eps,
                        self.n_q_heads, self.n_kv_heads, self.head_dim, self.rotary_dim, has_gate=True,
                    )
                    sg_q, sg_k, sg_gate = sg_q.reshape(1, -1), sg_k.reshape(1, -1), sg_gate.reshape(1, -1)
                    qg = q_gate.to(F64).view(1, self.n_q_heads, 2 * self.head_dim)
                    q64 = _rope64(_gemma_rms64(qg[..., :self.head_dim], w["q_norm"], self.eps), positions,
                                  self.rope_theta, self.rotary_dim).reshape(1, -1)
                    k64 = _rope64(_gemma_rms64(k.to(F64).view(1, self.n_kv_heads, self.head_dim), w["k_norm"],
                                               self.eps), positions, self.rope_theta, self.rotary_dim).reshape(1, -1)
                    c_q = oracle.calibrate(ROW, f"{case}/q", sg_q, q64, oracle.FLOOR_OP)
                    c_k = oracle.calibrate(ROW, f"{case}/k", sg_k, k64, oracle.FLOOR_OP)

                    g = oracle.generator(9900 + position + layer)
                    slots = torch.randperm(CONTEXT, generator=g, device=oracle.DEVICE)[:position + 1].contiguous()
                    k_cache = torch.randint(0, 256, (CONTEXT, kv), generator=g, device=oracle.DEVICE,
                                            dtype=torch.int32).to(torch.uint8)
                    v_cache = torch.randint(0, 256, (CONTEXT, kv), generator=g, device=oracle.DEVICE,
                                            dtype=torch.int32).to(torch.uint8)
                    k_expected, v_expected = k_cache.clone(), v_cache.clone()
                    q_out = oracle.empty((1, q_size), BF16)
                    k_out = oracle.empty((1, kv), BF16)
                    gate_out = oracle.empty((1, q_size), BF16)
                    oracle.call("cere_ktest_attn_prologue", q_gate, k, v, w["q_norm"], w["k_norm"], self.cos_sin_ctx,
                                CONTEXT, positions, slots, position, 1, k_cache, v_cache, CONTEXT, self.eps, q_out,
                                k_out, gate_out)
                    limit_q = oracle.calibrated_limit(c_q, oracle.FLOOR_OP)
                    oracle.check_within(self, ROW, f"{case}/q", oracle.rel_l2(q_out, sg_q), c=c_q, limit=limit_q)
                    limit_k = oracle.calibrated_limit(c_k, oracle.FLOOR_OP)
                    oracle.check_within(self, ROW, f"{case}/k", oracle.rel_l2(k_out, sg_k), c=c_k, limit=limit_k)
                    oracle.check_bit_exact(self, ROW, f"{case}/gate", gate_out, sg_gate)
                    # The appended rows are the FP8 cast (scale 1.0) of the k the call returns and of v.
                    k_expected[slots[position]] = k_out.reshape(-1).to(torch.float8_e4m3fn).view(torch.uint8)
                    v_expected[slots[position]] = v.reshape(-1).to(torch.float8_e4m3fn).view(torch.uint8)
                    oracle.check_bit_exact(self, ROW, f"{case}/k_cache", k_cache, k_expected)
                    oracle.check_bit_exact(self, ROW, f"{case}/v_cache", v_cache, v_expected)


if __name__ == "__main__":
    unittest.main()
