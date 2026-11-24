// VulkanAmd P2: K13S phase-S accept / resample on Vulkan (shaders/k13s_accept.comp), the rule and contracts of
// executor/cuda/src/kernels/k13s_spec_accept.h: SGLang's target-only rule with thresholds 1.0 on one speculative chain, evaluated on device
// buffers with no host round trip; row j's inputs are K12's `probabilities` and `keep_mask` (k12_sampler_vk.h). Addresses are buffer
// device addresses; the same conventions as k12_sampler_vk.h (record into the open command buffer, caller barriers between calls,
// shaderFloat64 context, subgroup 32, CERE_ERR_ARGUMENT outside the contract, vocab in [2, 2^30]; probs of all rows within one buffer).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vkk13s {

constexpr int64_t kMaxDrafts = 15;

// Bytes of the device scratch of spec_accept for chains of up to max_drafts drafts (8-byte aligned address; holds K12's draw scratch).
// Every pipeline spec_accept / row_trials can request: the three accept stages and K12's seeded-draw pipelines (the prefix
// pass and the seeded draw: the bonus id's). No parameters: no pipeline depends on vocab, n or the draft count.
std::vector<vk::PipelineSpec> pipeline_specs();

size_t scratch_bytes(int64_t max_drafts);

// One chain: probs f32 [n + 1, vocab], masks u32 [n + 1, ceil(vocab / 32)], drafts u32 [n] (an id >= vocab is never accepted), seed, key0 (the
// absolute position row 0 chooses). committed u32 [n + 1] (committed[0 .. count) are the committed ids), count u32 [1] in [1, n + 1].
cere_status spec_accept(vk::Context &ctx, uint64_t probs, uint64_t masks, int64_t vocab, uint64_t drafts, int64_t n, uint64_t seed, uint32_t key0,
                        uint64_t committed, uint64_t count, uint64_t scratch);

size_t trial_scratch_bytes();

// spec_accept's row decision for `trials` independent coins on one row with draft `draft` (>= vocab: never accepted): trial t uses seed
// first_seed + t * seed_step and key key0 + t * key_step (uint32 wrap). ids u32 [trials], accepted u8 [trials].
cere_status row_trials(vk::Context &ctx, uint64_t probs, uint64_t mask, int64_t vocab, uint32_t draft, uint64_t first_seed, uint64_t seed_step,
                       uint32_t key0, uint32_t key_step, int64_t trials, uint64_t ids, uint64_t accepted, uint64_t scratch);

}  // namespace cere::vkk13s
