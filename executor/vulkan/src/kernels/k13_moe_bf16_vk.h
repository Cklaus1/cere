// VulkanAmd P2: K13, the BF16 MoE block of the MTP draft layer (executor/cuda/src/kernels/k13_moe_bf16.h), on Vulkan.
//
//   out = sum_k w_k * down_e(silu(gate_e(x)) * up_e(x)) + sigmoid(x . shared_expert_gate) * shared(x)
//
// K7's block on BF16 expert weights (or NVFP4 W4A16 ones, read as their BF16 dequantization). Reused from K7
// (vkk7 / shaders/k7_*.comp, unmodified): the router (k7_route: FP32 softmax, top-k, renormalize), the stable grouping
// (fused one-workgroup form for tail * K <= 256, else router + counts + placement), the sigmoid gate, the shared
// SwiGLU and the combine (k7_elem); the BF16 router / shared GEMMs are K2 (vkk2::run), routed by the row class as HIP's
// GemmPlan::run (k13_moe_bf16.h `plans` = K7's MoePlans: its cls; MoeBf16Options::cls here, 0 = m): class <= 16 the
// small-M GEMV (k2_mgemv_vk, row-invariant), class >= 17 the large-m GEMM (k2_gemm_vk, shift and row-count invariant).
// New: the grouped expert GEMV (shaders/k13_gemv.comp): lane l of 32 accumulates elements {8 l + 256 i + j}
// with FP32 FMAs (i, then j ascending), then a fixed xor butterfly, CUDA's order; GEMM1 rounds gate and up to BF16 and
// applies SwiGLU in FP32 (rounded to BF16), GEMM2 rounds down to BF16. Every output element's arithmetic depends on
// k alone, so a routed row is bit-equal whatever m, `first` or the batch, and NVFP4 experts give the bits of the BF16
// call on BF16(RN(RN(e2m1 * scale) * global)) tensors. The router / shared rows are bit-equal at the same row class
// (not across the 16 / 17 class boundary: k2_gemm_vk.h, as on HIP).
//
// Contract differences from CUDA (per device; the HIP test's rows hold, k13_test): the combine and the gates use K7's
// Vulkan order (FMA forms, k7_elem.comp); the router / shared GEMM orders are K2's (k2_mgemv / k2_gemm). Plans do not
// exist: their row class is MoeBf16Options::cls; rows plans are class <= 16 (the GEMV). Row classes >= 17 need
// vkk2::gemm_probe on the context, cooperative matrices and H, S multiples of 64 (CERE_ERR_UNSUPPORTED otherwise: no
// chunked-GEMV fallback). Not implemented: CUDA's programmatic-dependent small-tail kernels (the same arithmetic; here
// one GEMV kernel serves every tail).
//
// Addresses are buffer device addresses (16-byte aligned; NVFP4 scales 1-byte, globals 4-byte; workspace 256-byte),
// every tensor in one <= 4 GiB buffer. Records into the context's open command buffer, no trailing barrier.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cere_executor.h"
#include "kernels/k7_moe_vk.h"
#include "launch/vk_launch.h"

namespace cere::vkk13 {

using MoeDims = vkk7::MoeDims;

struct MoeBf16Layer {
    uint64_t router = 0;              // BF16 [E, H]
    uint64_t gate_up = 0;             // BF16 [E, 2 I, H]: gate rows 0..I-1, up rows I..2I-1
    uint64_t down = 0;                // BF16 [E, H, I]
    uint64_t shared_gate = 0, shared_up = 0;  // BF16 [S, H]
    uint64_t shared_down = 0;         // BF16 [H, S]
    uint64_t shared_expert_gate = 0;  // BF16 [H]
    // NVFP4 W4A16 routed experts (k13_moe_bf16.h): gate_up_q [E, 2 I, H / 2], gate_up_scale E4M3 [E, 2 I, H / 16],
    // gate_up_global F32 [E, 2] (gate, up), down_q [E, H, I / 2], down_scale [E, H, I / 16], down_global [E]. Set
    // gate_up_q (and the rest) with gate_up / down 0, or none of them.
    uint64_t gate_up_q = 0, gate_up_scale = 0, gate_up_global = 0;
    uint64_t down_q = 0, down_scale = 0, down_global = 0;
    // Optional K2 row info of router / shared_gate / shared_up / shared_down (moe_bf16_weight_info_bytes bytes, written
    // by moe_bf16_prepare_weights once at load; 0: each K2 GEMM scans its weight in the call).
    uint64_t bf16_info = 0;
};

struct MoeBf16Options {
    vkk7::GroupPath group = vkk7::GroupPath::kAuto;  // forcing changes no bit (k13_test)
    int32_t profile_stamp0 = -1;  // >= 0: timestamps profile_stamp0 .. + kProfileStamps - 1 at the phase boundaries
    uint32_t cls = 0;  // row class of the router / shared K2 products (MoePlans::cls): 0 = m, else m <= cls
};
// start | router + shared gate/up + gates | route(+group) + SwiGLU | split grouping | GEMM1 + shared down | GEMM2 |
// combine
constexpr uint32_t kProfileStamps = 7;

// Includes the four K2 GEMM workspaces (vkk2::gemm_workspace_bytes(max_m, n), n = E, S, S, H).
// Every pipeline moe_bf16_block / moe_bf16_prepare_weights can request for `dims`: the element-wise kernels and the four
// K7 route modes, the routed GEMV (glu 1 at k = hidden, glu 0 at k = inter; the lanes-per-step variant follows k < 512 / >= 512;
// one-row and 8-row capacity) for each expert format asked (bf16_experts: BF16 gate_up / down; nvfp4_experts: W4A16), K2's BF16
// GEMV for k = hidden, shared_inter and, when `gemm`, K2's GEMM + probe (row classes >= 17). Model: MoeDims{2560, 512, 10, 640,
// 640} (k 2560 and 640: both the 16-element variant), the S3 artifact is NVFP4: nvfp4_experts true (bf16 for the tests), gemm true.
std::vector<vk::PipelineSpec> pipeline_specs(const MoeDims &dims, bool bf16_experts, bool nvfp4_experts, bool gemm);

size_t moe_bf16_workspace_bytes(const MoeDims &dims, uint32_t max_m);

// Bytes of MoeBf16Layer::bf16_info (router E, shared gate S, shared up S, shared down H words, 16-byte aligned parts:
// vkk7::moe_prepare_weights's layout); 0 for invalid dims.
size_t moe_bf16_weight_info_bytes(const MoeDims &dims);
// Records the K2 weight scans of layer.router / shared_gate / shared_up / shared_down into layer.bf16_info. A barrier
// before the first moe_bf16_block. CERE_ERR_ARGUMENT for invalid dims / pointers, H or S not a multiple of 64.
cere_status moe_bf16_prepare_weights(vk::Context &ctx, const MoeDims &dims, const MoeBf16Layer &layer);

// out BF16 [m, H] from x BF16 [m, H]: the router and the shared expert on all m rows, routing, experts and combine on
// rows [first, m); rows [0, first) of out are not written. CERE_ERR_ARGUMENT for invalid dims / pointers / layer
// (mixed BF16 and NVFP4 experts), first >= m, 0 < cls < m or a short workspace.
cere_status moe_bf16_block(vk::Context &ctx, const MoeDims &dims, const MoeBf16Layer &layer, uint64_t x, uint32_t m,
                           uint32_t first, uint64_t out, uint64_t workspace, size_t workspace_bytes,
                           const MoeBf16Options *opts = nullptr);

}  // namespace cere::vkk13
