// VulkanAmd P2: K7, the NVFP4 MoE block (W4A4), on Vulkan (shaders/k7_*.comp + K2 for the BF16 parts: vkk2::run, the
// small-M GEMV for row classes <= 16 and the cooperative-matrix GEMM (k2_gemm_vk.h) for row classes >= 17).
//
// Same contract as executor/cuda/src/kernels/k7_moe.h, per device (vulkan-amd design note "Contracts"):
//   out = sum_k w_k * down_e(q2(silu(gate_e(q1(x))) * up_e(q1(x)))) + sigmoid(x . shared_expert_gate) * shared(x)
// router: BF16 logits x . router^T (K2), FP32 softmax, top-k by logit (ties: lowest id), renormalized; q1 / q2:
// K0's activation quantizer (one global scale per GEMM); W4A4 grouped GEMM with exact block integer dots, exact
// FP8 x FP8 block-scale products, FP32 accumulation in block order, alpha, BF16; SwiGLU in FP32 to BF16; combine in
// slot order + the sigmoid-gated BF16 shared expert (K2 gate / up / down, SwiGLU between).
//
// Every output element has a fixed operation sequence that depends on neither m, the batch composition, the routing
// of other tokens nor the routed-GEMM path: the GEMV (k7_gemv.comp, m <= 16 by default) and the cooperative-matrix GEMM
// (k7_gemm.comp) give the same bits. The four BF16 K2 products (router, shared gate / up / down) are routed by the row
// class as HIP's GemmPlan::run (k7_moe.h MoePlans::cls; MoeOptions::cls here, 0 = m): class <= 16 runs the small-M GEMV
// (every row independent of the call), class >= 17 the K2 GEMM (shift and row-count invariant). A row's bits are
// therefore a function of the row and its class: equal to that row alone at the same class (HIP k7_test "cls 40"
// rows), not across classes 16 / 17 (k2_gemm_vk.h, as on HIP). The summation orders are Vulkan's own (bit-identity
// with CUDA / HIP is a non-goal); the HIP test's rows and bounds are re-proven by ktest/k7_test.cpp.
// Row classes >= 17 need vkk2::gemm_probe on the context, cooperative matrices, and H, S multiples of 64 (refused
// CERE_ERR_UNSUPPORTED otherwise: there is no chunked-GEMV fallback).
//
// Not implemented (refused, CERE_ERR_UNSUPPORTED / ARGUMENT, never emulated): MoeTaps (hook points, routed_in /
// shared_in / routed_rows, the post-route hook), the shared-expert fork (no second queue here), P3b.
//
// Addresses are buffer device addresses (16-byte aligned; workspace 256-byte aligned), every tensor inside one
// <= 4 GiB buffer. The functions record into the context's open command buffer (Context::begin .. submit_wait) and
// end without a trailing barrier (the caller orders later reads, as k2_mgemv_vk does); no allocation.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vkk7 {

struct MoeDims {
    uint32_t hidden = 0;        // H, a multiple of 128
    uint32_t num_experts = 0;   // E, 1..1024
    uint32_t top_k = 0;         // K, 1..min(E, 32)
    uint32_t inter = 0;         // I, a multiple of 128
    uint32_t shared_inter = 0;  // S, a multiple of 16 (K2's GEMV k rule; the CUDA header allows any S > 0)
};

// Device addresses in the checkpoint layout (k7_moe.h MoeLayer).
struct MoeLayer {
    uint64_t router = 0;       // BF16 [E, H]
    uint64_t gate_weight = 0;  // [E, I, H/2] E2M1, element 2j in the low nibble
    uint64_t gate_scale = 0;   // [E, I, H/16] E4M3
    uint64_t up_weight = 0, up_scale = 0;
    uint64_t down_weight = 0;  // [E, H, I/2]
    uint64_t down_scale = 0;   // [E, H, I/16]
    uint64_t g1_alphas = 0, g2_alphas = 0;  // FP32 [E]
    float a1_global_scale = 0.0f, a2_global_scale = 0.0f;
    uint64_t shared_gate = 0, shared_up = 0;  // BF16 [S, H]
    uint64_t shared_down = 0;                 // BF16 [H, S]
    uint64_t shared_expert_gate = 0;          // BF16 [H]
    // Optional (0: each K2 GEMM scans its weight in the call): the four BF16 weights' K2 row info, moe_weight_info_bytes
    // bytes written by moe_prepare_weights (once per layer, at load). Read only by row classes >= 17.
    uint64_t bf16_info = 0;
    // MXFP8 shared expert (the S3 artifact's: executor/cuda/src/model/mxfp8_plan.h kShared): a nonzero `*_scales` makes
    // that projection MXFP8, its weight field then holding E4M3 bytes [n, k] and `*_scales` UE8M0 [n, k / 32]
    // (k2_mgemv_vk.h; 0 = BF16). Each is independent; MXFP8 gives its BF16 upcast's bits (vkk2's contract). shared_down
    // (k = S) as MXFP8 needs S % 32. The router stays BF16 (not an MXFP8 group in mxfp8_plan.h). bf16_info keeps its name
    // and layout: the K2 row info is per weight whatever its format.
    uint64_t shared_gate_scales = 0, shared_up_scales = 0, shared_down_scales = 0;
};

// Which kernels run. kAuto: the GEMV (k7_gemv.comp) for m <= kGemvAutoMaxM (decode and the verify widths) when
// H, I <= kGemvMaxK, else the cooperative-matrix GEMM (k7_gemm.comp); the one-workgroup router + grouping for
// m * top_k <= kFusedMaxPairs, else router, counts and placement. The GEMV's workgroup tiles (kAuto: kRow): kExpert =
// an active expert's rows (m <= kGemvMaxExpertRows), kRow = one sorted row each (any m; an expert's weights are read
// once per row routed to it: faster at m 5 .. 64 on random routing, k7_test --explore). Forcing a path changes no
// output bit (k7_test checks); a forced path outside its limits is CERE_ERR_ARGUMENT.
// profile_stamp0 >= 0: moe_block also writes Context timestamps profile_stamp0 .. + kProfileStamps - 1 at its phase
// boundaries (start, after router / shared gate-up / q1 / gate, after route + SwiGLU, after grouping, after GEMM1 +
// shared down, after q2, after GEMM2, after the combine); timing only, no output bit changes.
enum class GemmPath { kAuto, kGemv, kCoopmat };
enum class GroupPath { kAuto, kFused, kSplit };
enum class GemvTiles { kAuto, kExpert, kRow };
struct MoeOptions {
    GemmPath gemm = GemmPath::kAuto;
    GroupPath group = GroupPath::kAuto;
    GemvTiles tiles = GemvTiles::kAuto;
    int32_t profile_stamp0 = -1;
    // The row class of the BF16 K2 products (k7_moe.h MoePlans::cls; HIP GemmPlan::run): 0 = m; else m <= cls (a call
    // of m rows standing for a class of cls rows: a D15 resume part, a split, a pinned bucket). CERE_ERR_ARGUMENT if
    // 0 < cls < m. Changes no routed-expert bit (those are class-independent).
    uint32_t cls = 0;
#ifdef CERE_VK_TUNE
    // VulkanAutotune knobs (vulkan-autotune design note H3 / H4): only in the ktest build (-DCERE_VK_TUNE: k7_test and
    // its k7_moe_vk_tune.o), never in the library or any configuration; 0 = the default. Speed only (int8 block sums are
    // exact, any tiling sums each output element's blocks in block order). CERE_ERR_ARGUMENT outside the lists.
    uint32_t gemv_lanes_long = 0;   // routed-expert GEMV lanes per column at k > 1024 (default 16): 4 | 8 | 16
    uint32_t gemv_lanes_short = 0;  // ... at k <= 1024 (default 8): 4 | 8 | 16
    uint32_t gemm_bn = 0;           // output columns per workgroup of the k7_gemm SPIR-V in use (its BN, default 64): 64 | 128
#endif
};
constexpr uint32_t kProfileStamps = 8;
constexpr uint32_t kGemvAutoMaxM = 16;
constexpr uint32_t kGemvMaxExpertRows = 8;
constexpr uint32_t kGemvMaxK = 2560;
constexpr uint32_t kFusedMaxPairs = 256;

// Runs k7_probe.comp (its own submit: call outside recording) and records whether the device's cooperative-matrix
// accumulator layout is the one k7_gemm.comp assumes. moe_block refuses the cooperative-matrix path
// (CERE_ERR_UNSUPPORTED) on a context without a passing probe. Returns CERE_OK when the layout matches.
// Every pipeline moe_block / moe_route / probe / moe_prepare_weights can request for `dims`, per the paths enabled:
//   gemv: the decode / verify path (row classes <= 16): K2 BF16 GEMV (k = hidden, k = shared_inter), the routed-expert
//         GEMV (row-tile variants);
//   gemm: the prefill path (rows > 16, or a K2 class >= 17): the coopmat routed GEMM + its probe, K2's GEMM + probe,
//         and the split-routing passes;
//   option_variants: also the pipelines only a non-default MoeOptions (GemvTiles::kExpert, GroupPath::kSplit, GemmPath
//         forcing) can request.
// The element-wise kernels, the quantizer and the fused / one-thread-per-row route are always listed. Model
// (qwen38-flash-next): MoeDims{2560, 512, 10, 640, 640}; deployment: gemv true, gemm true, option_variants false.
//   mx: also the MXFP8 K2 pipelines of an MXFP8 shared expert (GEMV k = H and S; the MXFP8 GEMM and weight scan): the
//       S3 artifact: true; a BF16 artifact: false.
std::vector<vk::PipelineSpec> pipeline_specs(const MoeDims &dims, bool gemv, bool gemm, bool option_variants, bool mx = false);

cere_status probe(vk::Context &ctx);

// Bytes of the workspace moe_block needs for up to max_m tokens (0 for invalid dims). Includes the four K2 GEMM
// workspaces (vkk2::gemm_workspace_bytes(max_m, n) for n = E, S, S, H: they run without barriers between them).
size_t moe_workspace_bytes(const MoeDims &dims, uint32_t max_m);

// Bytes of MoeLayer::bf16_info (the router's, shared gate's, shared up's and shared down's K2 row info, 16-byte
// aligned parts); 0 for invalid dims.
size_t moe_weight_info_bytes(const MoeDims &dims);
// Records the K2 weight scans (vkk2::gemm_prepare_weight) of layer.router / shared_gate / shared_up / shared_down into
// layer.bf16_info (16-byte aligned). The caller puts a barrier after it before a moe_block reads the info.
// Each weight is scanned in its own format (the `*_scales` of MoeLayer). CERE_ERR_ARGUMENT for invalid dims / pointers,
// H or S not a multiple of 64.
cere_status moe_prepare_weights(vk::Context &ctx, const MoeDims &dims, const MoeLayer &layer);

// Router on BF16 logits [m, E]: ids int32 [m, K], weights FP32 [m, K] (k7_moe.h moe_route's contract).
cere_status moe_route(vk::Context &ctx, uint64_t logits, uint64_t topk_ids, uint64_t topk_weights, uint32_t m,
                      uint32_t num_experts, uint32_t top_k);

// out BF16 [m, H] from x BF16 [m, H]. CERE_ERR_ARGUMENT before anything is recorded for an MXFP8 shared_down with
// S % 32 != 0.
cere_status moe_block(vk::Context &ctx, const MoeDims &dims, const MoeLayer &layer, uint64_t x, uint32_t m,
                      uint64_t out, uint64_t workspace, size_t workspace_bytes, const MoeOptions *opts = nullptr);

}  // namespace cere::vkk7
