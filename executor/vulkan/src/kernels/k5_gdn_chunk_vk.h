// VulkanAmd P2: K5 chunked prefill on Vulkan (shaders/k5_gdn_chunk.comp). Same contract as
// executor/cuda/src/kernels/k5_gdn_chunk.h (the HIP build, executor/hip/src/k5_gdn_chunk_hip.cpp, is the reference), with
// buffer device addresses instead of pointers and CERE_ERR_ARGUMENT instead of cudaErrorInvalidValue:
//
//   q, k  bf16 [tokens, 16, 128] raw (L2-normalised inside with vkk5::l2norm, eps 1e-6, to BF16 first);
//   v     bf16 [tokens, 48, 128];  g, beta f32 [tokens, 48];  state bf16 [48, 128 (v), 128 (k)] in/out;
//   o     bf16 [tokens, 48, 128];  1 <= tokens <= 4096; chunks of 64 rows from the call's first row.
//   state_in32 / capture32 (+ _hi split: heads 24..47 there, the base holds heads 0..23) as the header:
//   capture_chunk in 0..chunks, the FP32 h at the start of that chunk (chunks = the final h before its BF16 rounding).
//
// Arithmetic (shader header): FP32 FMA chains in a fixed order per output, exact FP32 forward-substitution inverse,
// FLA's BF16 rounding points; no atomics, no split reductions; a chunk's bits depend only on its rows and the incoming
// FP32 state, so a call from a 64-aligned row resumed from a capture reproduces the longer call bit for bit.
//
// Records into the context's open command buffer (Context::begin .. submit_wait), with barriers between its own
// dispatches and none after the last: the caller records a barrier before reading o / state / capture32. Every tensor
// lies inside one <= 4 GiB buffer; q, k, v, o 16-byte aligned; workspace 256-byte aligned, chunk_workspace_bytes(tokens)
// bytes inside one buffer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vkk5 {

constexpr int kChunkTokens = 64;
constexpr int64_t kChunkMaxTokens = 4096;

struct ChunkArgs {
    uint64_t q = 0, k = 0, v = 0, g = 0, beta = 0, state = 0, o = 0;
    int64_t tokens = 0;
    float scale = 0.f;
    uint64_t state_in32 = 0;
    uint64_t capture32 = 0;
    int64_t capture_chunk = -1;
    uint64_t state_in32_hi = 0;
    uint64_t capture32_hi = 0;
};

// Every pipeline delta_rule_chunk requests: its six chunk kernels and K5's l2norm (k5_gdn_vk.h piece_pipeline_specs).
std::vector<vk::PipelineSpec> chunk_pipeline_specs();

// Workspace bytes for `tokens` rows (head-major q, k, v; G, B; u, w, v_new; P; BF16 h at every chunk start; o); 0 when
// tokens is outside 1..4096.
size_t chunk_workspace_bytes(int64_t tokens);

cere_status delta_rule_chunk(vk::Context &ctx, const ChunkArgs &args, uint64_t workspace);

}  // namespace cere::vkk5
