// K3 (m22 design note): hyper-connections (GatedResidual with hc_per_branch_norm) - mix, combine, and the final
// mix-only mixer `hyper_connection_mixer` (qwen4_exp.py:1611-1619, 1668), which uses the same mix.
// math_reference: sglang
// (sglang/srt/layers/hyperconnection.py: GroupedGemmaRMSNorm :21-75, _mix_compute :195, _combine_compute :213,
// GatedResidual.mix/combine :233-338; sglang/srt/layers/hc_mix_triton.py fused_hc_mix for the FP32 intermediates.)
//
// The hyper state is BF16 [m, hc * hidden] (hc streams of `hidden`). With W_down [lowrank, hc * hidden],
// W_up [hc * hidden, lowrank], W_inject [hc, hc * hidden] and norm weight [hc * hidden]:
//   normed[r, g, :] = bf16(x * rsqrt(mean_g(x^2) + eps) * (1 + w_norm[g, :]))     (per stream; K1 k1::rmsnorm)
//   t[r, :]         = bf16(silu((normed[r] · W_down^T) / hc))                       (FP32 GEMM, as fused_hc_mix)
//   mixed[r, j]     = bf16(mean_g(sigmoid((t[r] · W_up^T)[g, j]) * normed[r, g, j]))  (FP32)
//   a[r, g]         = 2 * sigmoid((normed[r] · W_inject[g]) / hc)                    (FP32)
//   out[r, g, j]    = bf16(hyper[r, g, j] + a[r, g] * block_out[r, j])               (FP32)
// Every reduction has a fixed order (no atomics), so repeated calls are bit-identical.
#pragma once

#include "kernels/k2_gemm.h"

#include <cstddef>
#include <cstdint>

namespace cere::kernels {

struct HcShape {
    uint32_t hc;       // streams (4)
    uint32_t hidden;   // per-stream width (2,560)
    uint32_t lowrank;  // mix rank (320)
    float eps;         // RMSNorm epsilon (1e-6)
};

// The two GEMM plans of the mix (shared by every GatedResidual with this shape): down (n = lowrank, k = hc * hidden)
// and up (n = hc * hidden, k = lowrank), both with FP32 output. Created at INSTANTIATE.
struct HcPlans {
    HcShape shape{};
    GemmPlan down;
    GemmPlan up;

    // CERE_ERR_UNSUPPORTED for hc other than 4 (the only compiled stream count) or a hidden size not a multiple of 8.
    cere_status init(cublasLtHandle_t handle, const HcShape &shape, uint32_t max_m);
    // Rows plans of `instantiated` (GemmPlan::init_rows on down and up; m26 design note D6, I6) for hc_mix at up to
    // `rows` rows; K1 test_rows_plan_invariant checks each row against `instantiated` at m = 1 bit for bit.
    // `fused_allowed`: GemmPlan::init_rows's kernel-test hook.
    cere_status init_rows(cublasLtHandle_t handle, const HcPlans &instantiated, uint32_t rows,
                          uint32_t fused_allowed = ~uint32_t{0});
};

// Bytes of the mix workspace for up to max_m rows (FP32 t, BF16 t, FP32 up-projection; each 256-byte aligned).
size_t hc_mix_workspace_bytes(const HcShape &shape, uint32_t max_m);

// How hc_mix runs its GEMMs and element-wise steps (all bit-identical; kernel tests compare them):
// - kEpilogue: when hc_mix_fuses(plans, m) the activation and the gate run in the GEMMs' epilogues (k2s::gemm_hc_act,
//   k2s::gemm_hc_gate; m23 design note P11R-21): 2 launches after the norm instead of 4; else as kUnfused.
// - kEarly (K3Chain, k3chain design note): kEpilogue with both GEMMs' `early` weights (k2_small_m.h): every warp
//   copies its weight slice into shared memory before its programmatic dependency wait, so the two 6.55 MB weights of
//   a mix stream from DRAM while the kernels before them (the combine and norm, the down projection) run. Same
//   launches, same operations in the same order.
// - kUnfused: GEMM, K3 activation, GEMM, K3 gate.
// - kAuto: kHcMixDefault.
enum class HcMixPath { kAuto, kEpilogue, kEarly, kUnfused };

// The library's kAuto path (the K3Chain flag; k3chain design note KC-04).
constexpr HcMixPath kHcMixDefault = HcMixPath::kEarly;

// True when hc_mix's kEpilogue path fuses at m: both plans run one small-M launch at m (k2_gemm lever small_m, m <=
// k2s::kMaxM, not per row) and k2s::hc_gate_supported(lowrank).
bool hc_mix_fuses(const HcPlans &plans, uint32_t m);

// GatedResidual.mix: writes `normed` [m, hc * hidden] (kept for combine) and `mixed` [m, hidden] (the block input).
// `workspace` holds hc_mix_workspace_bytes(shape, plans' max_m) device bytes, 256-byte aligned. `cls` as
// GemmPlan::run (0 = m): the fusion choice and both GEMMs dispatch for cls rows.
cere_status hc_mix(const HcPlans &plans, const __nv_bfloat16 *hyper, const __nv_bfloat16 *norm_weight,
                   const __nv_bfloat16 *w_down, const __nv_bfloat16 *w_up, uint32_t m, __nv_bfloat16 *normed,
                   __nv_bfloat16 *mixed, void *workspace, cudaStream_t stream, HcMixPath path = HcMixPath::kAuto,
                   uint32_t cls = 0);
// hc_mix after its per-stream norm: `normed` already holds it (hc_combine_norm wrote it), so the GEMMs and gate run
// alone (by `path`, as hc_mix). hc_mix == the K1 norm into `normed`, then hc_mix_normed.
cere_status hc_mix_normed(const HcPlans &plans, const __nv_bfloat16 *normed, const __nv_bfloat16 *w_down,
                          const __nv_bfloat16 *w_up, uint32_t m, __nv_bfloat16 *mixed, void *workspace,
                          cudaStream_t stream, HcMixPath path = HcMixPath::kAuto, uint32_t cls = 0);

// GatedResidual.combine: out = hyper + a * block_out per stream. `out` may alias `hyper` (in-place update).
cere_status hc_combine(const HcShape &shape, const __nv_bfloat16 *block_out, const __nv_bfloat16 *hyper,
                       const __nv_bfloat16 *normed, const __nv_bfloat16 *w_inject, uint32_t m, __nv_bfloat16 *out,
                       cudaStream_t stream);

// hc_combine, then the next mix's per-stream norm (weight next_norm_weight [hc * hidden]) of `out` into `normed`
// (m23 design note P5R): bit-identical to hc_combine followed by hc_mix's K1 norm, in one launch at m <= 16 (a
// cluster of the hc blocks of each row; hidden 1,800..2,944, 16-byte aligned), else as those two launches. `out` may
// alias `hyper`; `normed` is read (the combine gate) and then overwritten. The one launch is a programmatic dependent
// launch that reads w_inject and next_norm_weight before its dependency wait, so no kernel right before it on
// `stream` may write them (weights).
cere_status hc_combine_norm(const HcShape &shape, const __nv_bfloat16 *block_out, const __nv_bfloat16 *hyper,
                            __nv_bfloat16 *normed, const __nv_bfloat16 *w_inject,
                            const __nv_bfloat16 *next_norm_weight, uint32_t m, __nv_bfloat16 *out,
                            cudaStream_t stream);

}  // namespace cere::kernels
