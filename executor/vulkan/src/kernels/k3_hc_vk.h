// VulkanAmd P2: K3 hyper-connections on Vulkan (shaders/k3_hc.comp; K1's gemma rmsnorm and K2's small-M GEMV are called, not
// re-implemented). Same contract as executor/cuda/src/kernels/k3_hc.h (the HIP build is the reference), in the Vulkan layer's
// conventions: addresses are buffer device addresses (BF16 tensors 16-byte aligned, each inside one <= 4 GiB buffer),
// functions record into the context's open command buffer (Context::begin .. submit_wait), the caller puts a barrier after a
// call before reading its outputs, nothing is allocated.
//
// Execution mode (reported, not hidden): hc_mix is always the "unfused" path of HIP (norm, down GEMM, act, up GEMM, gate: five
// stages) and hc_combine_norm is always two launches (combine, then the norm). The down / up products are K2 (vkk2::run) routed
// by the row class as HIP's GemmPlan::run (k3_hc.h: "`cls` as GemmPlan::run (0 = m): ... both GEMMs dispatch for cls rows"):
// class <= 16 the small-M GEMV, class >= 17 the large-m K2 GEMM (k2_gemm_vk.h; needs vkk2::gemm_probe on the context and
// cooperative matrices, else CERE_ERR_UNSUPPORTED: no chunked-GEMV fallback). A rows plan (init_rows; HIP: K2 rows plans,
// always the GEMV) runs the GEMV in calls of <= 16 rows whatever cls. The fused paths of the CUDA header (HcMixPath::kEpilogue
// / kEarly, the cluster combine+norm) are not implemented: a request for them is CERE_ERR_UNSUPPORTED; kAuto and kUnfused run
// the unfused path. Bits: a row of hc_mix is a function of the row and its class: equal to that row at m = 1 with the same
// cls (and to that row of any call of the same class, any first row: the GEMM is shift invariant), not across the 16 / 17
// class boundary (k2_gemm_vk.h, as on HIP); the norm, act, gate, combine are row-local at every m.
// Differences from the CUDA signatures: ctx first, `workspace` and `hc_mix` take device addresses, hc_combine needs no workspace.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vkk3 {

struct HcShape {
    uint32_t hc;       // streams (4: the only compiled count)
    uint32_t hidden;   // per-stream width (multiple of 8, <= 8192: the norm's limit)
    uint32_t lowrank;  // mix rank (multiple of 16: the GEMV's k for the up projection)
    float eps;         // RMSNorm epsilon
};

// Every pipeline hc_mix / hc_mix_normed / hc_combine / hc_combine_norm can request for `shape`: k3_act / k3_gate /
// k3_combine, K1's gemma rmsnorm of width hidden, K2's GEMV for k = hc * hidden and k = lowrank (m 1..16), and, when
// `gemm`, K2's GEMM (BF16) + its probe for row classes >= 17 (prefill). Model: HcShape{4, 2560, 320, 1e-6f}, gemm true.
std::vector<vk::PipelineSpec> pipeline_specs(const HcShape &shape, bool gemm);

constexpr uint32_t kMaxRows = 16383;  // the norm dispatches m * hc workgroups (<= 65,535)

struct HcPlans {
    HcShape shape{};
    uint32_t max_m = 0;
    bool rows = false;  // an init_rows plan (behaviour identical, see above)

    // CERE_ERR_UNSUPPORTED for hc != 4, hidden not a multiple of 8 or above 8192, lowrank not a multiple of 16;
    // CERE_ERR_ARGUMENT for max_m 0 or above kMaxRows.
    cere_status init(const HcShape &shape, uint32_t max_m);
    cere_status init_rows(const HcPlans &instantiated, uint32_t rows);
};

enum class HcMixPath { kAuto, kEpilogue, kEarly, kUnfused };

// Bytes of the mix workspace for up to max_m rows (FP32 t_raw, BF16 t, FP32 up projection, the K2 GEMM workspace
// vkk2::gemm_workspace_bytes(max_m, hc * hidden) shared by the down and up products (a barrier between them); each
// 256-byte aligned).
size_t hc_mix_workspace_bytes(const HcShape &shape, uint32_t max_m);

// Always false: no fused GEMM-epilogue path exists on Vulkan.
bool hc_mix_fuses(const HcPlans &plans, uint32_t m);
const char *hc_mix_mode();  // "unfused: rmsnorm + K2(down) + act + K2(up) + gate; K2 = GEMV for row classes <= 16, GEMM above"

cere_status hc_mix(vk::Context &ctx, const HcPlans &plans, uint64_t hyper, uint64_t norm_weight, uint64_t w_down, uint64_t w_up,
                   uint32_t m, uint64_t normed, uint64_t mixed, uint64_t workspace, HcMixPath path = HcMixPath::kAuto,
                   uint32_t cls = 0);
cere_status hc_mix_normed(vk::Context &ctx, const HcPlans &plans, uint64_t normed, uint64_t w_down, uint64_t w_up, uint32_t m,
                          uint64_t mixed, uint64_t workspace, HcMixPath path = HcMixPath::kAuto, uint32_t cls = 0);
// out may alias hyper.
cere_status hc_combine(vk::Context &ctx, const HcShape &shape, uint64_t block_out, uint64_t hyper, uint64_t normed,
                       uint64_t w_inject, uint32_t m, uint64_t out);
// hc_combine, then the next mix's per-stream norm of `out` into `normed` (two launches, a barrier between). out may alias hyper.
cere_status hc_combine_norm(vk::Context &ctx, const HcShape &shape, uint64_t block_out, uint64_t hyper, uint64_t normed,
                            uint64_t w_inject, uint64_t next_norm_weight, uint32_t m, uint64_t out);

}  // namespace cere::vkk3
