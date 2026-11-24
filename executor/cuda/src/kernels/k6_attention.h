// K6 full attention (m22 design note K6, split by m22c design note D10): the prologue (QK norm + mRoPE + gate
// (K1), FP8 KV append (K1)) and the dense causal GQA with the sigmoid output gate (K1). The model runs the prologue
// for every full-attention call and then dense_attention (prefix_len + tokens <= kMaxContext) or K9 QSA (above,
// m22c D2). The o_proj GEMM is the caller's (K2).
//
// K/V source (spec "Model-specific kernel decisions", review M22-03), chosen by the caller with kv_from_cache: an
// extend without a cached prefix attends over the unquantized BF16 K/V of its own tokens (kv_from_cache false); an
// extend with a prefix (a later prefill chunk) and every decode step attend over the FP8 cache, dequantized with
// scale 1.0, including the entries this call has just stored (kv_from_cache true).
//
// math_reference: sglang
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cere_model_constants.h"
#include "kernels/k1_primitives.h"

namespace cere::k6 {

using bf16 = __nv_bfloat16;

// The compiled support table of this kernel: the A4 model's full-attention shape.
constexpr int kQHeads = gen::kNumAttentionHeads;
constexpr int kKvHeads = gen::kNumKeyValueHeads;
constexpr int kHeadDim = gen::kHeadDim;
constexpr int kRotaryDim = gen::kRotaryDim;
constexpr int kMaxContext = 2048;  // dense causal GQA limit (Q4); longer contexts are QSA (K9)
constexpr float kSoftmaxScale = 0.0625f;  // head_dim^-0.5
static_assert(kSoftmaxScale * kSoftmaxScale * kHeadDim == 1.0f, "kSoftmaxScale is head_dim^-0.5");

// Bytes of the prologue's outputs for max_tokens tokens: q [T, 24*256], k [T, 2*256], gate [T, 24*256] BF16, each
// 256-byte aligned, in that order (full_attention carves them from its workspace).
size_t prologue_bytes(int64_t max_tokens);
// Bytes of the workspace dense_attention needs for up to max_tokens tokens per call.
size_t dense_attention_workspace_bytes(int64_t max_tokens);
// Bytes of the workspace full_attention needs: prologue_bytes + dense_attention_workspace_bytes.
size_t full_attention_workspace_bytes(int64_t max_tokens);
// Blocks per (token, key split) of the FP8-cache attention launch of fewer than 4 tokens (AttnSplit): a decode step,
// an MTP extend or an eager call of 1-3 tokens (chain false, rows 1) or a verify chain of `rows` rows (chain true).
// Each block runs a group of a KV head's query heads over the split's keys, with the arithmetic and the split
// boundaries of the per-KV-head tile kernel (bit-equal outputs).
int split_blocks(int64_t rows, bool chain);

// Dense causal GQA: out[t, h*D:(h+1)*D] = softmax(q[t,h] . K[0..pos0+t]^T * scale) V for the 12 query heads of each
// KV head. q [tokens, 24*256] BF16. KV is either BF16 rows (k/v with row strides in elements, key j at row j) or the
// FP8 cache (k_cache/v_cache [slots, 2*256] bytes, key j at row slots[j], or row j when slots is null); exactly one
// pair is non-null. Keys 0 .. pos0 + tokens - 1 must be present. FP32 scores, softmax and accumulation; BF16 output.
// use_p4 (the model passes Workspace::prefill_p4, its prefill_attention lever): calls of >= p4::kMinRows tokens (by
// `cls` below) run the P4 tensor-core kernel instead (p4_attention.h); calls of fewer run the CUDA-core kernels.
// `cls` (m210 D15⁗′ class pin; 0 = tokens): the row count of the call these rows were cut from, at pos0 - (cls -
// tokens) .. ; every row-count choice (key splits, rows per block, P4) is made for it, so with pos0 a multiple of 4 a
// row's output equals that call's row bit for bit (P4 at any pos0: it is row-partition invariant, p4_attention.h).
// cls < tokens is cudaErrorInvalidValue.
cudaError_t attention(const bf16* q, int64_t tokens, int64_t pos0, const bf16* k, int64_t k_stride, const bf16* v,
                      int64_t v_stride, const uint8_t* k_cache, const uint8_t* v_cache, const int64_t* slots,
                      float scale, void* workspace, size_t workspace_bytes, bf16* out, cudaStream_t stream,
                      bool use_p4, int64_t cls = 0);

// The prologue of one full-attention call (m22c D10), shared by the dense (K6) and the sparse (K9) path:
//   q_gate [tokens, 24 * 2 * 256] (per head q then gate), k/v [tokens, 2 * 256] (row strides in elements),
//   q_norm_weight/k_norm_weight [256] (Gemma, applied as 1 + w), cos_sin the K1 rope_table with `context` rows
//   (rotary_dim 64); new token t has logical position prefix_len + t and rotary position positions[t] (int64,
//   device, each < context; null: prefix_len + t), or with positions3 (M10, m213 design note D11: int64 device
//   [3][tokens], axes t, h, w, each < context; non-null wins over positions) the interleaved mRoPE of k1_primitives.h;
//   slots [prefix_len + tokens] (int64, device) maps logical positions
//   to cache rows (null: identity); k_cache/v_cache [rows, 2 * 256] FP8 E4M3 bytes (scale 1.0) receive the new
//   tokens' rotated K and raw V. Writes q_out [tokens, 24 * 256] (normed, rotated), k_out [tokens, 2 * 256] (normed,
//   rotated) and gate_out [tokens, 24 * 256], contiguous BF16. Bound: prefix_len + tokens <= context (the session
//   context; no kMaxContext cap). cudaErrorInvalidValue before any launch on a null pointer or a bound violation.
cudaError_t prologue(const bf16* q_gate, int64_t q_gate_stride, const bf16* k, int64_t k_stride, const bf16* v,
                     int64_t v_stride, const bf16* q_norm_weight, const bf16* k_norm_weight, float eps,
                     const float* cos_sin, int64_t context, const int64_t* positions, const int64_t* positions3,
                     const int64_t* slots, int64_t tokens, int64_t prefix_len, uint8_t* k_cache, uint8_t* v_cache,
                     bf16* q_out, bf16* k_out, bf16* gate_out, cudaStream_t stream);

// Dense causal GQA after the prologue, then the output gate: out [tokens, 24 * 256] (row stride out_stride) =
// attention(q) * sigmoid(gate). q, gate: the prologue's q_out / gate_out. K/V source: kv_from_cache false reads the
// BF16 rows [0, prefix_len + tokens) of k (stride 2 * 256; prefix_len 0: the prologue's k_out) and v (row stride
// v_stride); true reads the FP8 cache through slots (null: identity), including the rows the prologue just stored.
// prefix_len + tokens <= kMaxContext. workspace: dense_attention_workspace_bytes(tokens), 16-byte aligned. use_p4,
// cls: as attention.
cudaError_t dense_attention(const bf16* q, const bf16* k, const bf16* v, int64_t v_stride, const bf16* gate,
                            int64_t tokens, int64_t prefix_len, bool kv_from_cache, const uint8_t* k_cache,
                            const uint8_t* v_cache, const int64_t* slots, void* workspace, size_t workspace_bytes,
                            bf16* out, int64_t out_stride, cudaStream_t stream, bool use_p4, int64_t cls = 0);

// prologue + dense_attention in one call (the K6 oracle's block): the arguments as above; q/k/gate are carved from
// the workspace (full_attention_workspace_bytes(tokens)). prefix_len + tokens <= kMaxContext; kv_from_cache false
// requires prefix_len 0. Returns cudaErrorInvalidValue on any shape outside the support table. use_p4: as attention.
cudaError_t full_attention(const bf16* q_gate, int64_t q_gate_stride, const bf16* k, int64_t k_stride, const bf16* v,
                           int64_t v_stride, const bf16* q_norm_weight, const bf16* k_norm_weight, float eps,
                           const float* cos_sin, const int64_t* positions, const int64_t* positions3,
                           const int64_t* slots, int64_t tokens, int64_t prefix_len, bool kv_from_cache,
                           uint8_t* k_cache, uint8_t* v_cache, void* workspace, size_t workspace_bytes, bf16* out,
                           int64_t out_stride, cudaStream_t stream, bool use_p4);

// A VERIFY chain's rows (verify_attention = batched, m23 design note P11R-19): `rows` tokens at the logical positions
// pos0 .. pos0 + rows - 1 whose K/V the prologue already stored in the FP8 cache. Output row t is bit-equal to
// dense_attention(q + t row, ..., gate + t row, tokens 1, prefix_len pos0 + t, kv_from_cache true): one launch runs every
// row's one-token key splits (split_count / keys per split of kv_len pos0 + t + 1, causal at pos0 + t), one launch
// combines each row's splits and one gates all rows. q, gate: the prologue's q_out / gate_out [rows, 24 * 256]; slots
// as dense_attention. 1 <= rows <= kSplitMaxTokens, pos0 + rows <= kMaxContext; workspace
// dense_attention_workspace_bytes(rows), 16-byte aligned; cudaErrorInvalidValue before any launch otherwise.
cudaError_t dense_attention_rows(const bf16* q, const bf16* gate, int64_t rows, int64_t pos0, const uint8_t* k_cache,
                                 const uint8_t* v_cache, const int64_t* slots, void* workspace, size_t workspace_bytes,
                                 bf16* out, int64_t out_stride, cudaStream_t stream);

// The decode graph's step (m23 design note P1, M23P1-03/04): one token at the logical position *position (int64,
// device; the caller checks 1 <= *position < the context before the launch). Each is the call above at tokens 1,
// prefix_len = *position, positions and positions3 null and kv_from_cache true, launch for launch and bit for bit,
// with every position-dependent quantity read on the device (the rope row, the KV slot slots[*position], the split
// count and keys per split; the attention grid is fixed at the most splits of a kMaxContext key range and the extra
// splits exit), so no launch carries a host scalar of the position and a captured graph replays at any position.
// slots non-null. dense_attention_step / full_attention_step: *position + 1 <= kMaxContext; workspace
// dense_attention_workspace_bytes(1) / full_attention_workspace_bytes(1), 16-byte aligned.
cudaError_t prologue_step(const bf16* q_gate, int64_t q_gate_stride, const bf16* k, int64_t k_stride, const bf16* v,
                          int64_t v_stride, const bf16* q_norm_weight, const bf16* k_norm_weight, float eps,
                          const float* cos_sin, const int64_t* slots, const int64_t* position, uint8_t* k_cache,
                          uint8_t* v_cache, bf16* q_out, bf16* k_out, bf16* gate_out, cudaStream_t stream);
cudaError_t dense_attention_step(const bf16* q, const bf16* gate, const uint8_t* k_cache, const uint8_t* v_cache,
                                 const int64_t* slots, const int64_t* position, void* workspace,
                                 size_t workspace_bytes, bf16* out, int64_t out_stride, cudaStream_t stream);
cudaError_t full_attention_step(const bf16* q_gate, int64_t q_gate_stride, const bf16* k, int64_t k_stride,
                                const bf16* v, int64_t v_stride, const bf16* q_norm_weight, const bf16* k_norm_weight,
                                float eps, const float* cos_sin, const int64_t* slots, const int64_t* position,
                                uint8_t* k_cache, uint8_t* v_cache, void* workspace, size_t workspace_bytes, bf16* out,
                                int64_t out_stride, cudaStream_t stream);

// The verify graph's chain (decode_step = spec_graph, m29 design note M29S2-18b, m23 design note P11R-20): the
// calls above for `rows` (1 .. kSplitMaxTokens) consecutive tokens at the device positions positions[0 .. rows)
// (int64, device, positions[t] = positions[0] + t; the caller checks positions[0] >= 1 and positions[0] + rows <=
// kMaxContext before the capture). prologue_rows_at is prologue(..., positions, nullptr, slots, rows, prefix_len =
// positions[0], ...) with the KV slots read on the device; dense_attention_rows_at is dense_attention_rows at pos0 =
// positions[0] with every row's split count and keys per split computed on the device and the split grid fixed at
// the most splits of a kMaxContext key range (the extra splits exit), so a captured chain replays at any position.
// Launch for launch and bit for bit the eager calls. slots non-null; workspace dense_attention_workspace_bytes(rows),
// 16-byte aligned; cudaErrorInvalidValue before any launch otherwise.
cudaError_t prologue_rows_at(const bf16* q_gate, int64_t q_gate_stride, const bf16* k, int64_t k_stride, const bf16* v,
                             int64_t v_stride, const bf16* q_norm_weight, const bf16* k_norm_weight, float eps,
                             const float* cos_sin, const int64_t* slots, const int64_t* positions, int64_t rows,
                             uint8_t* k_cache, uint8_t* v_cache, bf16* q_out, bf16* k_out, bf16* gate_out,
                             cudaStream_t stream);
cudaError_t dense_attention_rows_at(const bf16* q, const bf16* gate, int64_t rows, const int64_t* positions,
                                    const uint8_t* k_cache, const uint8_t* v_cache, const int64_t* slots,
                                    void* workspace, size_t workspace_bytes, bf16* out, int64_t out_stride,
                                    cudaStream_t stream);

// The verify graph's chains of several sequences in one call each (rows16 design note R16-C): `rows` (1 ..
// kSplitMaxTokens) rows, row t at the device position positions[t] (int64 [rows], device; the caller checks every
// position >= 1 and < kMaxContext before the capture) over its own sequence's caches and slot map kv.*[t]
// (k1::KvRows; rows of one sequence consecutive, in position order). prologue_chains_at is prologue_rows_at over each
// sequence's rows (the per-token QK norm/RoPE/gate over all rows, then every row's K/V store,
// k1::kv_store_fp8_rows); dense_attention_chains_at is dense_attention_rows_at over each sequence's rows, row t
// attending to its sequence's keys 0 .. positions[t] (every row's K/V stored before the call). Row for row and bit
// for bit the per-sequence calls. Workspace dense_attention_workspace_bytes(rows), 16-byte aligned; caches 16-byte
// aligned; cudaErrorInvalidValue before any launch otherwise.
cudaError_t prologue_chains_at(const bf16* q_gate, int64_t q_gate_stride, const bf16* k, int64_t k_stride,
                               const bf16* v, int64_t v_stride, const bf16* q_norm_weight, const bf16* k_norm_weight,
                               float eps, const float* cos_sin, const k1::KvRows& kv, const int64_t* positions,
                               int64_t rows, bf16* q_out, bf16* k_out, bf16* gate_out, cudaStream_t stream);
cudaError_t dense_attention_chains_at(const bf16* q, const bf16* gate, int64_t rows, const int64_t* positions,
                                      const k1::KvRows& kv, void* workspace, size_t workspace_bytes, bf16* out,
                                      int64_t out_stride, cudaStream_t stream);

// The spec_graph draft graph's MTP extend (m29 design note M29S2-18b, m23 design note P11R-20): dense_attention(q,
// ..., gate, tokens, prefix_len = positions[0], kv_from_cache true) with pos0 = positions[0] read on the device (int64,
// device; the caller checks positions[0] >= 1 and positions[0] + tokens <= kMaxContext before the capture), the split
// count and keys per split computed on the device and the split grid fixed at the most splits of a kMaxContext key
// range (the extra splits exit): launch for launch and bit for bit the eager call, so a captured extend replays at any
// position. dense_attention_step is this at tokens 1. 1 <= tokens <= kSplitMaxTokens; slots non-null; workspace
// dense_attention_workspace_bytes(tokens), 16-byte aligned; cudaErrorInvalidValue before any launch otherwise.
cudaError_t dense_attention_at(const bf16* q, const bf16* gate, int64_t tokens, const int64_t* positions,
                               const uint8_t* k_cache, const uint8_t* v_cache, const int64_t* slots, void* workspace,
                               size_t workspace_bytes, bf16* out, int64_t out_stride, cudaStream_t stream);

}  // namespace cere::k6
