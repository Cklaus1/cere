// P4 (m23 design note P4): tensor-core prefill attention for calls of more than 16 query rows, the dense causal
// GQA of K6 (k6_attention.h) and the sparse GQA of K9 (k9_qsa.h) on BF16 mma.sync (m16n8k16, F32 accumulation),
// sm_120a. The path SGLang's control runs for these calls is its Triton sparse GQA (`qsa/sparse_attn.py`
// _sparse_gqa_prefill / _sparse_gqa_chunk_prefill: the 12 query heads of a KV head padded to one 16-row MMA tile,
// tl.dot with F32 accumulation, P rounded to the value dtype before P.V); FlashInfer has no per-row index prefill
// (m23 design note P4 [REVISED], N3). Unpinned calls of <= 16 rows (decode, verify) never come here: K6/K9 keep
// their split-K CUDA-core kernels (P4 leaves the n <= 16 path unchanged); only K9's class-pinned hot tail does (below).
//
// Arithmetic per (query row, query head), over the key set of the row:
//   s_j = F32 mma(q_bf16, k_j) * (scale * log2 e)            (K FP8 E4M3 -> BF16 exactly, scale 1.0)
//   online softmax in base 2 per 32-key tile: m, alpha = 2^(m_old - m_new), p_j = 2^(s_j - m_new) (F32),
//   l = l * alpha + sum_j p_j (F32, unrounded p), o = o * alpha + mma(bf16(p), v) (F32 accumulation)
//   out = bf16(o / l); a row without valid keys is 0.
// Deterministic: one block per (row tile, KV head), a fixed key order and fixed in-block reduction orders.
//
// The model selects P4 per model at run time by the prefill_attention slot (cere_slot_selection impl
// CERE_LEVER_PREFILL_P4 sets model::Workspace::prefill_p4, the use_p4 argument of K6/K9; the default
// CERE_LEVER_PREFILL_CUDA_CORE keeps the CUDA-core kernels). Hot-safe: P4 takes no workspace (it reads q, the K/V
// source and indices and writes out; K6/K9's scratch is carved for every call whatever the lever), so the lever only
// flips that bool between requests.
//
// math_reference: sglang
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cere_model_constants.h"

namespace cere::p4 {

using bf16 = __nv_bfloat16;

// The compiled support table: the A4 model's full-attention shape (as k6/k9).
constexpr int kQHeads = gen::kNumAttentionHeads;
constexpr int kKvHeads = gen::kNumKeyValueHeads;
constexpr int kHeadDim = gen::kHeadDim;
constexpr int64_t kMinRows = 17;  // calls of <= 16 rows stay on the K6/K9 split-K kernels

// Dense causal GQA (k6::attention's contract for its P4 choice): out[t, h] = softmax_j(q[t, h] . K[j] * scale) V[j]
// over keys j = 0 .. pos0 + t. q [tokens, 24 * 256] BF16 (contiguous), out likewise. K/V: BF16 rows k/v (row strides
// in elements, key j at row j) or the FP8 cache k_cache/v_cache [slots, 2 * 256] bytes (key j at row slots[j], j when
// slots is null); exactly one pair non-null. Row-partition invariant: a row's key tiles start at key 0 for every call,
// and the tiles a block runs past the row's own last key are fully masked (alpha exp2(0) = 1, p 0, o and l unchanged),
// so the rows [a, b) of a call equal the call of those rows alone at pos0 + a bit for bit. That is why k6::attention's
// class pin may choose it for fewer than kMinRows tokens (a hot tail; BeatSglang S7). cudaErrorInvalidValue for
// tokens < 1 or any other bad argument.
cudaError_t dense_attention(const bf16* q, int64_t tokens, int64_t pos0, const bf16* k, int64_t k_stride,
                            const bf16* v, int64_t v_stride, const uint8_t* k_cache, const uint8_t* v_cache,
                            const int64_t* slots, float scale, bf16* out, cudaStream_t stream);

// Sparse GQA (k9::qsa_attention's contract for its P4 choice): out[t, h] = softmax_j(q[t, h] . K[indices[t, j]] *
// scale) V[indices[t, j]] over the valid (>= 0) entries of indices [rows, row_width] int32 (logical positions), zeros
// for a row without one. q / out [rows, 24 * 256] BF16 with row strides q_stride / out_stride. K/V source as above
// (BF16 rows: position j at row j; FP8 cache: row slots[j], j when slots is null). One block per (query token, KV
// head), so a row's bits do not depend on the call's other rows: K9 runs it for any rows >= 1 of a call whose class
// pin chose P4 (a hot tail of fewer than kMinRows rows of a longer cold call, BeatSglang S7). cudaErrorInvalidValue
// for rows < 1 or any other bad argument.
cudaError_t sparse_attention(const bf16* q, int64_t q_stride, const int32_t* indices, int64_t row_width, int64_t rows,
                             const bf16* k, int64_t k_stride, const bf16* v, int64_t v_stride, const uint8_t* k_cache,
                             const uint8_t* v_cache, const int64_t* slots, float scale, bf16* out, int64_t out_stride,
                             cudaStream_t stream);

}  // namespace cere::p4
