// VulkanAmd P2: K5 GDN decode and pieces on Vulkan (shaders/k5_gdn_decode.comp, k5_gdn_conv.comp, k5_gdn_gating.comp,
// k5_gdn_l2norm.comp). Not the chunked prefill (k5_gdn_chunk) and not the layer composition.
//
// Same contracts as executor/cuda/src/kernels/k5_gdn_decode.h and k5_gdn.h (decode, decode_rows, conv1d_silu,
// conv1d_state_at, conv1d_silu_rows, gating, l2norm), with buffer device addresses instead of pointers and
// CERE_ERR_ARGUMENT instead of cudaErrorInvalidValue. Shapes are the Flash-Next GDN layer's: 16 key heads, 48 value
// heads, head dim 128, conv width 4. Every tensor lies inside one <= 4 GiB buffer (byte offsets are 32-bit; calls whose
// extents do not fit are refused). The functions record into the context's open command buffer (Context::begin ..
// submit_wait); the caller orders calls that depend on each other with Context::barrier (conv1d_silu records its own
// barrier between its two dispatches). Arithmetic and order are documented in the shaders; the contract is per device
// (accuracy vs FP64 within the HIP ktests' bounds, repeats and chains bit-equal), not bit-identity with CUDA/HIP.
//
// The *_rows calls read their chain table (states, saves, first, rows) from a host-visible buffer the call writes at
// `table_off` (16-byte aligned, kChainTableBytes); that region must not be rewritten until the submit completes.
#pragma once

#include <cstdint>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vkk5 {

constexpr int kKeyHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kHeadDim = 128;
constexpr int kConvWidth = 4;
constexpr int kConvState = kConvWidth - 1;
constexpr int kMaxRows = 16;  // rows (all chains) of one *_rows call: a verify step's rows
constexpr uint32_t kChainTableBytes = 384;

// --- decode (k5_gdn_decode.h) ---------------------------------------------------------------------------------------
// q, k: bf16 [16, 128]; v: bf16 [48, 128]; a, b: bf16 [48]; A_log: f32 [48]; dt_bias: bf16 [48];
// state: bf16 [48, 128, 128] in/out; o: bf16 [48, 128]. q, k, v, o, state 8-byte aligned.
// Pipeline groups of this file's functions, for pipeline_specs (a bit set).
constexpr uint32_t kPieceDecode = 1;      // delta_rule_decode
constexpr uint32_t kPieceRows = 2;        // delta_rule_decode_rows, conv1d_silu_rows
constexpr uint32_t kPieceConvGating = 4;  // conv1d_silu, conv1d_state_at, gating
constexpr uint32_t kPieceL2Norm = 8;      // l2norm
constexpr uint32_t kPiecesAll = 15;
// The pipelines of the selected groups (no shape parameter: none of them depends on a shape; each is a single fixed
// pipeline). The composite families (chunk, layer, chain) take the subsets they call.
std::vector<vk::PipelineSpec> piece_pipeline_specs(uint32_t pieces = kPiecesAll);

struct DecodeArgs {
    uint64_t q, k, v, a, b, A_log, dt_bias, state, o;
    float scale;
};
cere_status delta_rule_decode(vk::Context &ctx, const DecodeArgs &args);

// The verify chains of one step (k5_gdn_decode.h DecodeRowsArgs): strides in elements; step.state is not read.
struct DecodeRowsArgs {
    DecodeArgs step;
    int chains;
    int first[kMaxRows];
    int rows[kMaxRows];
    uint64_t states[kMaxRows];
    uint64_t saves[kMaxRows];
    int64_t qkv_stride, ab_stride, o_stride;
};
cere_status delta_rule_decode_rows(vk::Context &ctx, const DecodeRowsArgs &args, const vk::Buffer &table,
                                   uint64_t table_off);

// --- pieces (k5_gdn.h) ----------------------------------------------------------------------------------------------
// Depthwise causal conv1d + SiLU over `tokens` rows of `dim` channels; state bf16 [dim, 3] in/out. Two dispatches with a
// barrier between them (the conv reads the state, the second writes the new one).
cere_status conv1d_silu(vk::Context &ctx, uint64_t x, int64_t x_stride, uint64_t weight, uint64_t state, uint64_t y,
                        int64_t y_stride, int64_t tokens, int64_t dim);
// The state `tokens` (>= 3) rows into x into `state`.
cere_status conv1d_state_at(vk::Context &ctx, uint64_t x, int64_t x_stride, uint64_t state, int64_t tokens, int64_t dim);

struct ConvChains {
    int chains;
    int first[kMaxRows];
    int rows[kMaxRows];
    uint64_t states[kMaxRows];
    uint64_t saves[kMaxRows];
};
cere_status conv1d_silu_rows(vk::Context &ctx, uint64_t x, int64_t x_stride, uint64_t weight, const ConvChains &chains,
                             uint64_t y, int64_t y_stride, int64_t dim, const vk::Buffer &table, uint64_t table_off);

// a, b bf16 [tokens, heads] with row strides; A_log f32 [heads]; dt_bias bf16 [heads]; g, beta f32 [tokens, heads].
cere_status gating(vk::Context &ctx, uint64_t a, int64_t a_stride, uint64_t b, int64_t b_stride, uint64_t A_log,
                   uint64_t dt_bias, uint64_t g, uint64_t beta, int64_t tokens, int64_t heads);

// x, y bf16 [rows, 128] contiguous, 16-byte aligned (may alias); dim must be 128.
cere_status l2norm(vk::Context &ctx, uint64_t x, uint64_t y, int64_t rows, int64_t dim, float eps);

}  // namespace cere::vkk5
