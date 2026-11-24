// VulkanAmd P2: the K12 device sampler on Vulkan (shaders/k12_*.comp). Same function set and semantics as
// executor/cuda/src/kernels/k12_sampler.h (host/sampler_ref is the reference); the arithmetic is made equal to the host's by
// construction (correctly rounded fp32 / fp64 division, fixed-order fp64 sums, integer murmur counter, bit-pattern ranking), see the
// report an internal benchmark report Addresses are buffer device addresses (0 = null where the CUDA
// pointer is nullable); each tensor lies inside one <= 4 GiB buffer; functions record into the context's open command buffer
// (Context::begin .. submit_wait), allocate nothing and return CERE_ERR_ARGUMENT outside their contract. The caller puts
// Context::barrier() between dependent calls (as with K1). Needs a context with shaderFloat64 (the ktest build compiles a copy of
// vk_launch.cpp with that feature enabled; the shared launch layer does not request it yet) and subgroup size 32.
//
// Differences from the CUDA header, all deliberate: logits are BF16 read as uint16; `vocab` is in [2, 2^30] (every row must fit one
// 4 GiB buffer: 2^30 fp32 values), not [2, 2^31); scratch is a device address of scratch_bytes() bytes, 8-byte aligned, shared by
// probabilities, keep_mask and the draws exactly as in CUDA (nothing in it outlives the call that wrote it). probabilities
// stores z = x / T in `probs` between its launches (probs is the output either way).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vkk12 {

// The draw's fp64 scan runs over kChunks contiguous id ranges.
constexpr int kChunks = 1024;

// D2 without the temperature (see k12_sampler.h): out[i] = fp32(bf16 logits[i]); counts[i] > 0: x - (f * c + p) (f * c one fp32
// multiply, no contraction), then x < 0 ? x * r : x / r (division correctly rounded); ids whose bit is clear in mask_dev become -inf; then
// out[bias_ids[j * stride]] += bias_values[j * stride] for j < bias_count. counts / mask_dev / bias_*: 0 = none.
// Every pipeline the functions below can request: process (+ bias), top2, the three probability passes, the keep-mask passes
// (the masked pass only when top_k can exceed 1024: pass the largest top_k the server accepts; the model's vocabulary is
// 248,320), the draw kernels (prefix, seeded, caller-uniform), finalize, logprobs and count_commit. No other shape
// parameter: no K12 pipeline depends on vocab or n.
std::vector<vk::PipelineSpec> pipeline_specs(int64_t max_top_k);
// The draw_u / draw_seeded subset (the prefix pass + the seeded and / or caller-uniform pass), for K13S's bonus draw.
std::vector<vk::PipelineSpec> draw_pipeline_specs(bool seeded, bool u);

cere_status process(vk::Context &ctx, uint64_t logits, int64_t vocab, uint64_t counts, float frequency_penalty, float presence_penalty,
                    float repetition_penalty, uint64_t mask_dev, uint64_t bias_ids, uint64_t bias_values, int64_t bias_stride,
                    int64_t bias_count, uint64_t out);

// ids u32 [2] (first maximum, first maximum without it: ties to the lower id, NaN above +inf), scores f32 [2]. Exact.
cere_status top2_f32(vk::Context &ctx, uint64_t values, int64_t vocab, uint64_t ids, uint64_t scores);

// z = x / T (fp32), lse = max + log(sum exp(z - max)) in fp64, probs[i] = fp32(exp(z - lse)); -inf -> 0; *lse (fp64, device).
// Non-finite z: +inf counts as the largest finite float, NaN carries no mass, an all-massless row puts p = 1 on its lowest NaN id (0).
cere_status probabilities(vk::Context &ctx, uint64_t processed, int64_t vocab, float temperature, uint64_t probs, uint64_t lse,
                          uint64_t scratch);

// D3 keep mask (sampler_ref::keep_mask semantics, exact on the fp32 probabilities); mask: ceil(vocab / 32) words.
cere_status keep_mask(vk::Context &ctx, uint64_t probs, int64_t vocab, int64_t top_k, float top_p, float min_p, uint64_t mask, uint64_t scratch);

size_t scratch_bytes();

// D4 inverse CDF over the kept ids (sampler_ref::draw, exact): draw_u reads us[n] (fp64 [0, 1)); draw_seeded uses
// u = uniform(first_seed + j, key, 0, 1) for draw j. ids u32 [n].
cere_status draw_u(vk::Context &ctx, uint64_t probs, uint64_t mask, int64_t vocab, uint64_t us, int64_t n, uint64_t ids, uint64_t scratch);
cere_status draw_seeded(vk::Context &ctx, uint64_t probs, uint64_t mask, int64_t vocab, uint64_t first_seed, int64_t n, uint32_t key,
                        uint64_t ids, uint64_t scratch);

// cere_token_score at `out` (16 bytes): id = *drawn (0: top2_ids[0]), runner_up as in k12_sampler.h, logits from `processed`.
cere_status finalize(vk::Context &ctx, uint64_t processed, uint64_t top2_ids, uint64_t drawn, uint64_t out);

// D7 (see k12_sampler.h): committed_dev (device u32, 0 = use committed_host); top n <= CERE_MAX_TOP_LOGPROBS to
// top_ids[j * stride] / top_logprobs[j * stride] (stride in 4-byte words).
cere_status top_logprobs(vk::Context &ctx, uint64_t processed, int64_t vocab, float temperature, uint64_t committed_dev,
                         uint32_t committed_host, int64_t n, uint64_t committed_logprob, uint64_t top_ids, uint64_t top_lps, int64_t stride);

// D5: counts[ids[i]] += 1 for i < n.
cere_status count_commit(vk::Context &ctx, uint64_t counts, uint64_t ids, int64_t n);

}  // namespace cere::vkk12
