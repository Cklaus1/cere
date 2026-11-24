// SelfTune ST4 S3c (selftune design note ST4): the kernels of family sdar (DICE-8B: SDAR block diffusion over
// Qwen3-8B dense layers; hidden 4096, 32 query / 8 KV heads of head_dim 128, intermediate 12288, BF16 weights and KV).
// Compiled only into a library built with CERE_MODEL naming an sdar descriptor (executor/CMakeLists.txt); no file
// here includes cere_model_constants.h, so the shapes are arguments. Every function enqueues on `stream`, never
// synchronizes, and returns the launch status (cudaErrorInvalidValue for a malformed argument, nothing launched).
// Tensors are row-major and contiguous unless a stride is given; BF16 in and out, FP32 arithmetic, with the rounding
// points of the model's reference forward (DICE modeling_sdar.py at the S2 pin) noted per function.
//
// math_reference: DICE-8B
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cere::sdar {

using bf16 = __nv_bfloat16;

// The block of block diffusion: query row at absolute position p sees key j iff j / kBlock <= p / kBlock.
constexpr int64_t kBlock = 4;
constexpr int kHeadDim = 128;

// out[t, :] = table[ids[t], :] for t < n (bit-exact copy). ids must lie in [0, vocab); the caller validates them.
cudaError_t embed(const bf16 *table, const uint32_t *ids, int64_t n, int hidden, bf16 *out, cudaStream_t stream);

// RMSNorm per row of `dim` elements: y = x * rsqrt(mean(x^2) + eps) * w, FP32 math, one rounding to BF16 (the
// reference's flash_attn rms_norm_fn). rows rows at strides x_stride / y_stride (elements); y may alias x with the
// same stride. Per-head q_norm / k_norm: rows = tokens * heads, dim = kHeadDim, stride kHeadDim.
cudaError_t rmsnorm(const bf16 *x, int64_t x_stride, const bf16 *w, float eps, int64_t rows, int dim, bf16 *y,
                    int64_t y_stride, cudaStream_t stream);

// In-place NeoX RoPE (rotate_half over the full head_dim 128) on rows*heads heads of kHeadDim at row stride
// `stride` (elements): row t is at absolute position pos0 + t, inv_freq[i] = theta^(-2i/128) in FP32, cos/sin
// rounded to BF16 and q*cos + rotate_half(q)*sin with the reference's BF16 rounding of each product and the sum.
cudaError_t rope(bf16 *x, int64_t stride, int64_t rows, int heads, int64_t pos0, float theta, cudaStream_t stream);

// KV store: copies n rows of K and V (kv_heads * kHeadDim elements each, row strides k_stride / v_stride) into one
// layer's cache rows [pos, pos + n) (cache rows are kv_heads * kHeadDim contiguous elements). Bit-exact copy.
cudaError_t kv_store(const bf16 *k, int64_t k_stride, const bf16 *v, int64_t v_stride, int64_t n, int kv_heads,
                     bf16 *k_cache, bf16 *v_cache, int64_t pos, cudaStream_t stream);

// The attention mask; kBlockCausal is the model's. kCausal (query p sees j <= p) exists only for the kernel tests'
// fail-first mutant.
enum class Mask : int { kBlockCausal = 0, kCausal = 1 };

// Block-causal GQA attention of n query rows at absolute positions [past, past + n) (past % kBlock == 0):
//   q:   [n, heads, kHeadDim] at row stride q_stride (after q_norm and RoPE);
//   keys [0, past): one layer's cache rows (k_cache / v_cache, kv_heads * kHeadDim each);
//   keys [past, past + n): new_k / new_v rows ([n, kv_heads, kHeadDim] at strides new_k_stride / new_v_stride),
//   either the cache's own rows at `past` (the tail slots, written by kv_store first) or a separate scratch buffer:
//   the output is the same bit for bit, since only these pointers are read for those keys.
// Query head h reads KV head h / (heads / kv_heads). Scores scaled by 1/sqrt(kHeadDim), FP32 softmax and accumulate,
// out [n, heads * kHeadDim] at row stride out_stride, one rounding to BF16.
cudaError_t attention(const bf16 *q, int64_t q_stride, const bf16 *k_cache, const bf16 *v_cache, int64_t past,
                      const bf16 *new_k, int64_t new_k_stride, const bf16 *new_v, int64_t new_v_stride, int64_t n,
                      int heads, int kv_heads, bf16 *out, int64_t out_stride, Mask mask, cudaStream_t stream);

// out[t, i] = silu(gate[t, i]) * up[t, i] for i < inter (gate at gate_up[t, 0..inter), up at gate_up[t,
// inter..2*inter), row stride 2*inter), with the reference's rounding: silu rounded to BF16, then the product.
cudaError_t silu_mul(const bf16 *gate_up, int64_t rows, int inter, bf16 *out, cudaStream_t stream);

// Residual add in place: x[i] = bf16(float(x[i]) + float(y[i])) for i < n, one round-to-nearest-even (the reference's
// `hidden_states = residual + hidden_states` on BF16 tensors; both residual adds of a layer). y may equal x.
cudaError_t add(bf16 *x, const bf16 *y, int64_t n, cudaStream_t stream);

// Per row of FP32 logits (rows at stride `stride`, vocab entries): every logit rounded to BF16 (round to nearest
// even, torch's cast), the maximum, the lowest id holding it (`id`), the number of ids holding it (`ties`, >= 1),
// and the runner-up (the second-lowest id holding the maximum when ties > 1, else the lowest id holding the largest
// value below it) with their BF16 values as floats.
struct TopTie {
    uint32_t id;
    uint32_t ties;
    uint32_t runner_up;
    float logit;
    float runner_up_logit;
};
cudaError_t top_tie(const float *logits, int64_t stride, int64_t rows, int64_t vocab, TopTie *out,
                    cudaStream_t stream);

// The unmask choice of one denoise step (low_confidence_static, top_k 1: confidence = 1 / ties): among the rows
// r < rows with masked[r] != 0, the one with the fewest ties, the lowest r among equals; -1 when none is masked.
// One thread; `*index` is device memory.
cudaError_t unmask_select(const TopTie *top, const uint8_t *masked, int rows, int32_t *index, cudaStream_t stream);

}  // namespace cere::sdar
