// VulkanAmd P3 (T3): the three kernels local to the MTP draft layer (executor/cuda/src/model/mtp.cu:43-115) on Vulkan.
// Call sites: norm_hidden mtp.cu:322 (kEmbedTokens fusion), add_embedding mtp.cu:334-335, argmax_map mtp.cu:246
// (hot_lm_head, per chunk of rows). Addresses are buffer device addresses; every tensor lies in one <= 4 GiB buffer.
// Each function records one dispatch into the context's open command buffer and no barrier: the caller records a barrier
// before reading the output. CERE_ERR_ARGUMENT (nothing recorded) outside the contract.
//
// norm_hidden (mtp.cu:43-71, pre_fc_norm_hidden): y[r] = bf16(x[r] * rsqrt(mean(x[r]^2) + eps) * (1 + w)) over the whole
//   10,240-wide row, FP32 arithmetic. This is vkk1::rmsnorm's Gemma row (weight_groups 1, dim 10,240) term for term:
//   the per-element expression is the same, CUDA differs only in the order of the sum of squares (one element per
//   thread stride 256 vs K1's 8-element chunks), which is a per-device choice (spec "Contracts": bit-identity with
//   CUDA/HIP is a non-goal). vkk1::rmsnorm itself refuses dim > 8,192, so norm_hidden records the K1 shader
//   (k1_rmsnorm_g0.spv, the same SPIR-V and push layout) at dim 10,240; the ktest proves the dispatch equal to
//   vkk1::rmsnorm bit for bit at the dims vkk1::rmsnorm accepts and checks the MTP shape against FP64.
// add_embedding (mtp.cu:73-80, the fusion's broadcast add): hyper[r, s, :] = bf16(float(hyper[r, s, :]) + float(e[r, :]))
//   for the 4 streams s, FP32 add (never contracted), BF16 round to nearest even, NaN 0x7FFF; denormals preserved.
// argmax_map (mtp.cu:82-115): draft[r] = map[argmax(logits[r, :count])]: the lowest index among equal maxima
//   (torch.argmax), a NaN logit counts as -inf (so an all-NaN / all -inf row gives index 0), -0 equals +0. Exact
//   (integer keys), independent of the reduction order.
#pragma once

#include <cstdint>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vkmtp {

constexpr int kHidden = 2560;            // gen::kHiddenSize of the A4 model
constexpr int kHc = 4;                   // gen::kHcCount
constexpr int kHcHidden = kHc * kHidden;  // 10,240: the HC state width
constexpr int64_t kMaxRows = 65535;      // rows of one call (the portable workgroup-count bound)

// x, y bf16 [rows, 10240] contiguous (y may alias x), w bf16 [10240]; 16-byte aligned; 1 <= rows <= 65,535.
// Every pipeline norm_hidden / add_embedding / argmax_map (and detail::gemma_rows at the extra dims) request: the add and
// argmax kernels and K1's ungated rmsnorm (gemma) with the workgroup size of width kHcHidden (norm_hidden) and of each
// extra_norm_dims width (detail::gemma_rows callers; the model's MTP layer uses none beyond kHcHidden).
std::vector<vk::PipelineSpec> pipeline_specs(const std::vector<int> &extra_norm_dims = {});

cere_status norm_hidden(vk::Context &ctx, uint64_t x, uint64_t w, float eps, int64_t rows, uint64_t y);

// hyper bf16 [rows, 4, 2560] in place, e bf16 [rows, 2560]; 16-byte aligned; 1 <= rows, rows * 10240 * 2 < 4 GiB.
cere_status add_embedding(vk::Context &ctx, uint64_t hyper, uint64_t e, int64_t rows);

// logits bf16 [rows, count] contiguous (2-byte aligned; the 16-byte vector path when the base is 16-byte aligned and
// count % 8 == 0), map u32 [count], draft u32 [rows]; 1 <= count < 2^31, count * 2 < 4 GiB, 1 <= rows <= 65,535.
cere_status argmax_map(vk::Context &ctx, uint64_t logits, uint32_t count, uint64_t map, int64_t rows, uint64_t draft);

namespace detail {
// norm_hidden's dispatch at any dim (8 <= dim <= 16,384, dim % 8 == 0): the K1 Gemma row with weight_groups 1. Exposed
// for the ktest's bit-equality row against vkk1::rmsnorm (which accepts dim <= 8,192).
cere_status gemma_rows(vk::Context &ctx, uint64_t x, uint64_t w, float eps, int64_t rows, int dim, uint64_t y);
}  // namespace detail

}  // namespace cere::vkmtp
