// VulkanAmd P2: the K1 primitives on Vulkan (shaders/k1_*.comp). Same contracts as executor/cuda/src/kernels/k1_primitives.h
// (the HIP build is the reference): BF16 in and out, FP32 arithmetic, tensors row-major with strides in elements.
// Addresses are buffer device addresses; each tensor lies inside one <= 4 GiB buffer. Functions record into the
// context's open command buffer (Context::begin .. submit_wait); no allocation, no hidden fallback. CERE_ERR_ARGUMENT
// outside the contract; CERE_ERR_UNSUPPORTED for the declared-but-not-implemented parts (positions3 / mRoPE).
//
// Table buffers: the CUDA kernels take KvRows / CopySegment[] by value as kernel parameters. A Vulkan push constant block is 128 bytes,
// so kv_store_fp8_rows and copy_segments copy the descriptors into a caller-provided host-visible vk::Buffer (`table`, written at record
// time through its mapping) that the shader reads by device address. One table per call in flight: do not reuse it before the
// command buffer has executed. mRoPE (positions3) is CERE_ERR_UNSUPPORTED everywhere.
#pragma once

#include <cstdint>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vkk1 {

// Every pipeline the functions below can create, for Context::warm at instantiate: the shape-independent ones plus
// the row-norm pipelines for each `dim` the model normalizes (rmsnorm / rmsnorm_gated_sigmoid) and the qk-norm pipeline
// for each head_dim (qk_norm_rope_gate); their workgroup size is a specialization constant of dim / head_dim.
std::vector<vk::PipelineSpec> pipeline_specs(const std::vector<int> &norm_dims, const std::vector<int> &qk_head_dims);

// The pipeline rmsnorm (gated = false) / rmsnorm_gated_sigmoid (true) request for a row width `dim`: the single selection
// used by the dispatch and by pipeline_specs (and by the families that call the norms: K3, K4, K5, MTP lists).
vk::PipelineSpec rmsnorm_spec(bool gated, int dim);

// out[t, :] = table[ids[t], :] for t < n, bit-exact. hidden % 8 == 0; ids in [0, vocab) (caller validates).
cere_status embedding_gather(vk::Context &ctx, uint64_t table, int64_t vocab, int hidden, uint64_t ids, int64_t n, uint64_t out);

// y[r, :] = bf16(x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w_r), w_r = 1 + weight (gemma) or weight, weight row r % weight_groups.
// dim % 8 == 0, dim <= 8192, strides % 8 == 0, 16-byte aligned pointers. y may alias x with the same stride.
cere_status rmsnorm(vk::Context &ctx, uint64_t x, int64_t x_stride, uint64_t weight, int weight_groups, bool gemma, float eps,
                    int64_t rows, int dim, uint64_t y, int64_t y_stride);

// y = bf16(x * rsqrt(mean(x^2) + eps) * weight * sigmoid(z)). Same requirements.
cere_status rmsnorm_gated_sigmoid(vk::Context &ctx, uint64_t x, int64_t x_stride, uint64_t z, int64_t z_stride, uint64_t weight,
                                  float eps, int64_t rows, int dim, uint64_t y, int64_t y_stride);

// cos_sin [n_pos, rotary_dim] F32 (row p = [cos(angle_i) | sin(angle_i)]). rotary_dim even, 2 <= rotary_dim <= 1024.
cere_status rope_table(vk::Context &ctx, uint64_t cos_sin, int64_t n_pos, int rotary_dim, float theta);

// Partial NeoX RoPE on x, y [T, heads, head_dim] (row strides in elements); dims [rotary_dim, head_dim) copied. positions
// (int64, device) or null = pos0 + t. positions3 != 0 (mRoPE) is CERE_ERR_UNSUPPORTED. y may alias x.
cere_status rope(vk::Context &ctx, uint64_t x, int64_t x_stride, uint64_t y, int64_t y_stride, uint64_t cos_sin, uint64_t positions,
                 uint64_t positions3, int64_t pos0, int64_t tokens, int heads, int head_dim, int rotary_dim);

// out[t, i] = bf16(x * sigmoid(gate)), i < width; out may alias x. width % 8 == 0.
cere_status sigmoid_mul(vk::Context &ctx, uint64_t x, int64_t x_stride, uint64_t gate, int64_t gate_stride, int64_t rows, int width,
                        uint64_t out, int64_t out_stride);

// top2 into ids int32 [rows, 2] and scores f32 [rows, 2]: torch.argmax rules (NaN is the maximum, lowest index wins ties).
// rows <= 65,535, vocab < 2^32.
cere_status top2_split(vk::Context &ctx, uint64_t logits, int64_t stride, int64_t rows, int64_t vocab, uint64_t ids, uint64_t scores);

// FP8 E4M3 KV store, scale 1.0 (RNE, > 448 saturates to 0x7E, NaN 0x7F). slots (int64 device) or null = slot0 + t.
// width % 8 == 0, strides % 8 == 0.
cere_status kv_store_fp8(vk::Context &ctx, uint64_t k, int64_t k_stride, uint64_t v, int64_t v_stride, int64_t tokens, int width,
                         uint64_t k_cache, uint64_t v_cache, uint64_t slots, int64_t slot0);

// qk_norm_rope_gate: q_gate [T, q_heads * 2 * head_dim] (per head q then gate), k [T, kv_heads * head_dim]; q_out, k_out, gate_out contiguous.
// out = rope(bf16(gemma_rmsnorm(x, weight))) per head, gate copied. head_dim % 8 == 0, <= 1024, rotary_dim even <= head_dim, strides % 8,
// 16-byte aligned pointers. positions (int64 device) or null = pos0 + t; positions3 != 0 -> CERE_ERR_UNSUPPORTED.
cere_status qk_norm_rope_gate(vk::Context &ctx, uint64_t q_gate, int64_t q_gate_stride, uint64_t k, int64_t k_stride, uint64_t q_weight,
                              uint64_t k_weight, uint64_t cos_sin, uint64_t positions, uint64_t positions3, int64_t pos0, int64_t tokens,
                              int q_heads, int kv_heads, int head_dim, int rotary_dim, float eps, uint64_t q_out, uint64_t k_out,
                              uint64_t gate_out);

// kv_store_fp8 at the device slot-map offset *at: slot(t) = slots[*at + t] (slots, at non-null).
cere_status kv_store_fp8_at(vk::Context &ctx, uint64_t k, int64_t k_stride, uint64_t v, int64_t v_stride, int64_t tokens, int width,
                            uint64_t k_cache, uint64_t v_cache, uint64_t slots, uint64_t at);

constexpr int kKvRowsMax = 16;
struct KvRows {
    uint64_t k_cache[kKvRowsMax];
    uint64_t v_cache[kKvRowsMax];
    uint64_t slots[kKvRowsMax];  // int64 device slot maps
};
// Row t into rows.k_cache[t] / v_cache[t] at slot rows.slots[t][at[t]] (at int64 [tokens], device). 0 <= tokens <= 16, every used row's
// pointers non-null. `table` >= 384 bytes (see above).
cere_status kv_store_fp8_rows(vk::Context &ctx, uint64_t k, int64_t k_stride, uint64_t v, int64_t v_stride, int64_t tokens, int width,
                              const KvRows &rows, uint64_t at, vk::Buffer &table);

struct CopySegment {
    uint64_t dst, dst_pitch, src, src_pitch, width, rows;  // device addresses (48 bytes, the shader's table entry)
};
constexpr int kMaxCopySegments = 128;
// The bytes of count (1 .. 128) segments in one dispatch, byte for byte a D2D copy per segment (16-byte units when a segment is 16-byte
// aligned throughout, else bytes). ARGUMENT on a null pointer (of a segment with width and rows > 0), rows > 65,535, count outside the
// range. A segment of 0 bytes is a no-op. Ranges must not overlap. `table` >= 48 * count bytes.
cere_status copy_segments(vk::Context &ctx, const CopySegment *segments, int count, vk::Buffer &table);

// top2 into cere_token_score[rows] (16 bytes: id, runner_up, logit, runner_up_logit): out is its device address.
cere_status top2(vk::Context &ctx, uint64_t logits, int64_t stride, int64_t rows, int64_t vocab, uint64_t out);

}  // namespace cere::vkk1
