// VulkanAmd P3 T2: the INSTANTIATE pipeline warm-up list (vulkan_warm.cpp).
#pragma once

#include "cere_executor.h"
#include "launch/vk_launch.h"

#include <string>
#include <vector>

namespace cere::vkdev {

// The union of the kernel families' pipeline lists for the model's shapes; *used names each list taken (with its
// size) and the families whose list was absent at build time ("missing:..."). `mx`: some K5 / K7 dense weight of the
// bound model is MXFP8 (the S3 artifact: every GDN projection and the shared expert), so their MXFP8 pipelines are
// created too (T5; otherwise they would be built at first use). K6 / K9 list both K/V formats (a prompt's first chunk
// reads its own BF16 K/V, later chunks the FP8 cache). `coopmat` (P4-E: the seal-time probe passed): also the
// cooperative-matrix prefill attention pipelines (Prefill::kCoopmat: coopmat kernels, their FP32 fallback, the K9
// route kernel), so either prefill_attention selection runs without a pipeline compiled after the seal.
std::vector<vk::PipelineSpec> vk_warm_specs(const cere_qwen4_exp_config &config, bool drafts, std::string *used,
                                            bool mx = false, bool coopmat = false);

}  // namespace cere::vkdev
