// VulkanAmd P3 (T3): the K6 attention prologue on Vulkan, eager form only (k6::prologue / prologue_bytes of
// executor/cuda/src/kernels/k6_attention.h:36-38, 64-79; definition k6_attention.cu:1153-1159, 1203-1216).
//
// The CUDA prologue has no arithmetic of its own: it validates, then launches k1::qk_norm_rope_gate (QK Gemma norm +
// NeoX RoPE + gate split) and k1::kv_store_fp8 (FP8 E4M3 store of the rotated k_out and the raw v, scale 1.0) with
// slots + prefix_len (or slot0 = prefix_len when slots is null). This is the same composition of the Vulkan K1
// primitives (k1_primitives_vk.h), recorded into the context's open command buffer with a barrier between the two
// dispatches (the store reads k_out) and none after the last: the caller records a barrier before reading q_out,
// k_out, gate_out or the cache. Its outputs are therefore bit-equal to the unfused vkk1 sequence by construction (the
// k6_prologue_test row asserts it).
//
// Shape: the A4 full-attention support table (24 query heads, 2 KV heads, head dim 256, rotary dim 64). Addresses are
// buffer device addresses; every tensor lies in one <= 4 GiB buffer. Differences from the CUDA header: positions3
// (mRoPE) is CERE_ERR_UNSUPPORTED (as every Vulkan K1 rope); tokens <= 65,535 per call (the K1 dispatch bound); the
// alignment rules of the K1 primitives apply (16-byte tensors, strides % 8 elements, 8-byte caches and slot maps).
// Nothing is recorded when a call is refused. Not provided (P3 design 2.2: not on the eager path): prologue_step,
// prologue_rows_at, prologue_chains_at.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vkk6 {

namespace prologue_shape {
constexpr int kQHeads = 24;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 256;
constexpr int kRotaryDim = 64;
constexpr int kQWidth = kQHeads * kHeadDim;    // 6,144
constexpr int kKvWidth = kKvHeads * kHeadDim;  // 512
constexpr int64_t kMaxTokens = 65535;          // tokens of one call (K1 dispatch bound)
}  // namespace prologue_shape

// Every pipeline prologue requests: K1's qk_norm_rope_gate for head_dim 256 and K1's kv_store (the union of the two K1
// calls' pipelines, taken from vkk1::pipeline_specs). No parameters: the shapes are prologue_shape's constants.
std::vector<vk::PipelineSpec> pipeline_specs();

// Bytes of the prologue's outputs for max_tokens tokens: q [T, 6144], k [T, 512], gate [T, 6144] BF16, each 256-byte
// aligned, in that order (k6_attention.cu:1153-1159). 0 for max_tokens <= 0.
size_t prologue_bytes(int64_t max_tokens);

// k6::prologue: q_gate [tokens, 24 * 2 * 256] (per head q then gate, row stride q_gate_stride elements), k / v
// [tokens, 512] (row strides k_stride / v_stride), q_norm_weight / k_norm_weight [256] BF16 (Gemma: 1 + w), cos_sin the
// K1 rope table [context, 64] F32. New token t has logical position prefix_len + t and rotary position positions[t]
// (int64 device, each < context; 0: prefix_len + t). slots (int64 device [prefix_len + tokens]) maps logical positions
// to cache rows (0: identity); k_cache / v_cache [rows, 512] FP8 E4M3 bytes receive the rotated k_out row and the raw v
// row of each new token. Writes q_out [tokens, 6144], k_out [tokens, 512], gate_out [tokens, 6144] contiguous BF16.
// CERE_ERR_ARGUMENT (nothing recorded) on a null pointer, prefix_len + tokens > context, context <= 0, tokens or
// prefix_len < 0, tokens > 65,535 or an alignment / stride rule; CERE_ERR_UNSUPPORTED for positions3 != 0.
cere_status prologue(vk::Context &ctx, uint64_t q_gate, int64_t q_gate_stride, uint64_t k, int64_t k_stride, uint64_t v,
                     int64_t v_stride, uint64_t q_norm_weight, uint64_t k_norm_weight, float eps, uint64_t cos_sin,
                     int64_t context, uint64_t positions, uint64_t positions3, uint64_t slots, int64_t tokens,
                     int64_t prefix_len, uint64_t k_cache, uint64_t v_cache, uint64_t q_out, uint64_t k_out,
                     uint64_t gate_out);

}  // namespace cere::vkk6
