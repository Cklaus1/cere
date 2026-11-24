// K3 (k3_hc.h): the element-wise arithmetic of the hyper-connection mix (sglang/srt/layers/hyperconnection.py
// _mix_compute :195; hc_mix_triton.py fused_hc_mix), shared by K3's own kernels and the small-M K2 kernel's fused hc
// epilogues (k2_small_m.h, m23 design note P11R-21), so both compile the same operations in the same order and
// their outputs are bit-identical (CUDA sources only).
// math_reference: sglang
#pragma once

#include <cuda_bf16.h>

#include <cstdint>

#include "cere_executor.h"
#include "cere_model_constants.h"

namespace cere::kernels::hc {

#if CERE_GEN_FAMILY == CERE_FAMILY_QWEN4_EXP
constexpr uint32_t kHc = gen::kHcCount;  // the compiled stream count (config hc_count)
#else
// Inkling 2 Mini (inkling design note S5/S7): a family without hyper-connections links K2's GemmPlan, whose small-M
// kernel carries the fused hc epilogues; they compile for one stream and are never launched (no hc layer).
constexpr uint32_t kHc = 1;
#endif

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// t = bf16(silu(raw / hc)): raw is one FP32 element of normed · W_down^T.
__device__ __forceinline__ __nv_bfloat16 act(float raw) {
    const float a = raw * (1.0f / static_cast<float>(kHc));
    return __float2bfloat16_rn(a * sigmoid(a));
}

// mixed = bf16(sum_g sigmoid(u[g]) * normed[g] / hc), summed in stream order g = 0 .. hc - 1: u[g] is the FP32
// element (g, j) of t · W_up^T and normed[g] the element (g, j) of the normalized streams.
__device__ __forceinline__ __nv_bfloat16 gate(const float (&u)[kHc], const __nv_bfloat16 (&normed)[kHc]) {
    float acc = 0.0f;
#pragma unroll
    for (uint32_t g = 0; g < kHc; ++g) {
        acc += sigmoid(u[g]) * __bfloat162float(normed[g]);
    }
    return __float2bfloat16_rn(acc * (1.0f / static_cast<float>(kHc)));
}

}  // namespace cere::kernels::hc
