// K1 primitives (m22 design note K1): embedding gather, RMSNorm (Gemma/plain, grouped weights), gated RMSNorm
// with a sigmoid gate, the attention input preparation (per-head Gemma QK RMSNorm + partial NeoX RoPE + gate
// deinterleave), standalone partial RoPE, sigmoid output gate, device top-2 argmax and the FP8 E4M3 KV store.
// All functions enqueue on `stream`, never synchronize, and return the launch status. Tensors are row-major with row
// strides in elements; BF16 in and out, FP32 arithmetic. The decode-path kernels (embedding gather, both RMSNorms,
// the QK norm/RoPE/gate, sigmoid_mul, the KV stores and copy_segments) are programmatic dependent launches at decode
// grids (<= 1,024 blocks; m23 design note P5R, SmallFuse): each lets the stream's next kernel launch, then waits for
// the previous grid before its first memory access, so stream order and every output bit are those of a plain launch.
//
// math_reference: sglang
// math_reference: flash-linear-attention
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cere_executor.h"

namespace cere::k1 {

using bf16 = __nv_bfloat16;

// out[t, :] = table[ids[t], :] for t < n (bit-exact copy). ids must lie in [0, vocab); the caller validates them.
cudaError_t embedding_gather(const bf16* table, int64_t vocab, int hidden, const int32_t* ids, int64_t n, bf16* out,
                             cudaStream_t stream);

// y[r, :] = bf16(x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w_r) with w_r = 1 + weight (gemma) or weight (plain), and
// weight row (r % weight_groups) of a [weight_groups, dim] weight. A grouped norm over x [T, G * dim] is rows = T * G,
// x_stride = dim, weight_groups = G; a per-head norm is rows = T * heads with x_stride the head stride.
// y may alias x with the same stride (in-place). Requires dim % 8 == 0, dim <= 8192, strides % 8 == 0 and 16-byte
// aligned pointers.
cudaError_t rmsnorm(const bf16* x, int64_t x_stride, const bf16* weight, int weight_groups, bool gemma, float eps,
                    int64_t rows, int dim, bf16* y, int64_t y_stride, cudaStream_t stream);

// y = bf16(x * rsqrt(mean(x^2) + eps) * weight * sigmoid(z)) per row (FLA RMSNormGated, norm before gate, plain
// weight, activation sigmoid). Same shape requirements as rmsnorm; y may alias x.
cudaError_t rmsnorm_gated_sigmoid(const bf16* x, int64_t x_stride, const bf16* z, int64_t z_stride, const bf16* weight,
                                  float eps, int64_t rows, int dim, bf16* y, int64_t y_stride, cudaStream_t stream);

// The rotary table of SGLang's RotaryEmbedding._compute_cos_sin_cache, computed on the device in FP32 as torch does
// on CUDA: inv_freq[i] = 1 / powf(theta, float(2i) / rotary_dim), angle = float(p) * inv_freq[i];
// cos_sin [n_pos, rotary_dim] F32, row p = [cos(angle_0 .. angle_{rd/2-1}) | sin(angle_0 .. angle_{rd/2-1})].
// Requires rotary_dim even, 2 <= rotary_dim <= 1024.
cudaError_t rope_table(float* cos_sin, int64_t n_pos, int rotary_dim, float theta, cudaStream_t stream);

// Attention input preparation, as SGLang's fused_qk_gemma_rmsnorm_rope_gate:
//   q_gate [T, q_heads * 2 * head_dim] (per head: q then gate), k [T, kv_heads * head_dim];
//   q_out [T, q_heads * head_dim] = rope(bf16(gemma_rmsnorm(q, q_weight))), k_out likewise with k_weight,
//   gate_out [T, q_heads * head_dim] = gate (copied). Outputs are contiguous.
// RoPE: NeoX halves over the first rotary_dim dims with cos/sin from the rope_table row of the token's position:
// positions[t] (int64, device) or, when positions is null, pos0 + t. Positions must lie in [0, n_pos) of the table.
// Text positions: the three interleaved mRoPE axes carry the same position, so the mRoPE section map selects the
// same angle for every frequency.
// positions3 (M10, m213 design note D11; int64 device [3][tokens], axes t, h, w; null = the text path above, and
// positions / pos0 are ignored when it is set): frequency i takes its angle from the row of the position on its axis
// under the interleaved mrope_section [11, 11, 10] (MRotaryEmbedding.apply_interleaved_rope); three equal axes give
// the text result bit for bit. Requires rotary_dim 64 (32 frequencies = 11 + 11 + 10).
// Requires head_dim % 8 == 0, head_dim <= 1024, rotary_dim even and <= head_dim, strides % 8 == 0.
cudaError_t qk_norm_rope_gate(const bf16* q_gate, int64_t q_gate_stride, const bf16* k, int64_t k_stride,
                              const bf16* q_weight, const bf16* k_weight, const float* cos_sin,
                              const int64_t* positions, const int64_t* positions3, int64_t pos0, int64_t tokens,
                              int q_heads, int kv_heads, int head_dim, int rotary_dim, float eps, bf16* q_out,
                              bf16* k_out, bf16* gate_out, cudaStream_t stream);

// Partial NeoX RoPE (SGLang MRotaryEmbedding): y[t, h, :] = rope(x[t, h, :]) for x, y [T, heads, head_dim] (row
// strides in elements), the table, positions and positions3 as in qk_norm_rope_gate; dims [rotary_dim, head_dim)
// are copied. y may alias x with the same stride. Requires head_dim <= 1024, rotary_dim even and <= head_dim.
cudaError_t rope(const bf16* x, int64_t x_stride, bf16* y, int64_t y_stride, const float* cos_sin,
                 const int64_t* positions, const int64_t* positions3, int64_t pos0, int64_t tokens, int heads,
                 int head_dim, int rotary_dim, cudaStream_t stream);

// out[t, i] = bf16(float(x[t, i]) * sigmoid(float(gate[t, i]))) for i < width; out may alias x.
cudaError_t sigmoid_mul(const bf16* x, int64_t x_stride, const bf16* gate, int64_t gate_stride, int64_t rows, int width,
                        bf16* out, int64_t out_stride, cudaStream_t stream);

// Per row of logits [rows, vocab]: the first index of the maximum (torch.argmax: NaN is the maximum, the lowest index
// wins ties) and the runner-up (the same rule over the remaining indices), with their values as float. One
// 8-CTA cluster per row; rows <= 65,535, else cudaErrorInvalidValue.
cudaError_t top2(const bf16* logits, int64_t stride, int64_t rows, int64_t vocab, cere_token_score* out,
                 cudaStream_t stream);

// top2 into separate arrays: ids int32 [rows, 2] = (id, runner_up), scores f32 [rows, 2] = their logits.
cudaError_t top2_split(const bf16* logits, int64_t stride, int64_t rows, int64_t vocab, int32_t* ids, float* scores,
                       cudaStream_t stream);

// FP8 E4M3 KV store with scale 1.0, as torch 2.13's `.to(torch.float8_e4m3fn)`: round to nearest even; a value that
// rounds above 448 (|x| > 464) and Inf saturate to +-448 (0x7E | sign), NaN stays NaN (0x7F | sign).
// k_cache[slot(t), :] = fp8(k[t, :]) and likewise v, for t < tokens, width = kv_heads * head_dim bytes per slot,
// slot(t) = slots[t] (int64, device) or, when slots is null, slot0 + t. Slots must lie inside the caches. Requires
// width % 8 == 0, strides % 8 == 0.
cudaError_t kv_store_fp8(const bf16* k, int64_t k_stride, const bf16* v, int64_t v_stride, int64_t tokens, int width,
                         uint8_t* k_cache, uint8_t* v_cache, const int64_t* slots, int64_t slot0,
                         cudaStream_t stream);

// kv_store_fp8 at the device slot-map offset *at (m23 design note P1, M23P1-03: the decode graph's step):
// slot(t) = slots[*at + t] (slots non-null), the stores of kv_store_fp8(slots + *at, ...) bit for bit.
cudaError_t kv_store_fp8_at(const bf16* k, int64_t k_stride, const bf16* v, int64_t v_stride, int64_t tokens,
                            int width, uint8_t* k_cache, uint8_t* v_cache, const int64_t* slots, const int64_t* at,
                            cudaStream_t stream);

// The KV targets of a store over the rows of several sequences (rows16 design note R16-C: the verify graph's chains
// of every sequence at once): row t's caches and slot map.
constexpr int kKvRowsMax = 16;
struct KvRows {
    uint8_t* k_cache[kKvRowsMax];
    uint8_t* v_cache[kKvRowsMax];
    const int64_t* slots[kKvRowsMax];
};
// kv_store_fp8 with row t into rows.k_cache[t] / rows.v_cache[t] at slot rows.slots[t][at[t]] (at int64 [tokens],
// device): row t's store of kv_store_fp8_at(.., rows.slots[t], at + t) bit for bit (equal when at[t] = *at + t).
// 0 <= tokens <= kKvRowsMax; every row's pointers non-null, caches 8-byte aligned.
cudaError_t kv_store_fp8_rows(const bf16* k, int64_t k_stride, const bf16* v, int64_t v_stride, int64_t tokens,
                              int width, const KvRows& rows, const int64_t* at, cudaStream_t stream);

// One device byte range of copy_segments: `rows` rows of `width` bytes, row r from src + r * src_pitch to
// dst + r * dst_pitch (rows 1: a flat copy of `width` bytes). Ranges must not overlap.
struct CopySegment {
    void* dst;
    uint64_t dst_pitch;
    const void* src;
    uint64_t src_pitch;
    uint64_t width;
    uint64_t rows;
};
constexpr int kMaxCopySegments = 128;

// The bytes of `count` (1 .. kMaxCopySegments) segments in one kernel launch, the result of one
// cudaMemcpy(2D)Async D2D per segment byte for byte (m23 design note P11R-20: inside a CUDA graph a D2D memcpy
// node costs a copy-engine switch, several us of idle each, a kernel node about one; DG-03: the verify_commit
// restore, up to 98 pieces, is one launch). 16-byte accesses when every pointer, pitch and width of a segment is
// 16-byte aligned, else byte accesses. Up to 4 segments pass a 200-byte parameter block, more the 6,272-byte one (a
// kernel parameter above 4 KiB: CUDA 12.1). cudaErrorInvalidValue on a null pointer, rows > 65,535 or a count outside
// the range; a segment of 0 bytes is a no-op.
cudaError_t copy_segments(const CopySegment* segments, int count, cudaStream_t stream);

}  // namespace cere::k1
