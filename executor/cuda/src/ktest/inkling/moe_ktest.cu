// Inkling 2 Mini S4: kernel-test entry points of the routed-MoE kernels (kernels/inkling/moe.h) for
// executor/tests/gpu/test_inkling_kernels_moe.py. Device pointers, int64 sizes, float scalars, the stream last; 0 =
// CERE_OK, CERE_ERR_ARGUMENT for arguments outside a kernel's contract, CERE_ERR_INTERNAL for a failed launch.
// `mutant` = cere::inkling::MoeMutant (0 = the model's arithmetic). The CPU references' entry points (host pointers)
// are in moe_ref_ktest.cpp.
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cere_executor.h"
#include "kernels/inkling/moe.h"
#include "inkling/quant_ref.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace {

using cere::inkling::bf16;
using cere::inkling::MoeMutant;

int status_of(cudaError_t err) {
    if (err == cudaSuccess) return CERE_OK;
    if (err == cudaErrorInvalidValue) return CERE_ERR_ARGUMENT;
    return CERE_ERR_INTERNAL;
}

bool fits_int(int64_t v) { return v >= 0 && v <= INT32_MAX; }

cere::inkling::MoeWeights weights_of(int64_t nvfp4, int64_t num_experts, const void *w13, const void *w13_scale,
                                     const void *w2, const void *w2_scale, const void *w13_amax, const void *w2_amax,
                                     const void *w13_den, const void *w2_den) {
    cere::inkling::MoeWeights w;
    w.nvfp4 = nvfp4 != 0;
    w.num_experts = static_cast<int32_t>(num_experts);
    w.w13 = static_cast<const uint8_t *>(w13);
    w.w13_scale = static_cast<const uint8_t *>(w13_scale);
    w.w2 = static_cast<const uint8_t *>(w2);
    w.w2_scale = static_cast<const uint8_t *>(w2_scale);
    w.w13_amax = static_cast<const float *>(w13_amax);
    w.w2_amax = static_cast<const float *>(w2_amax);
    w.w13_den = static_cast<const float *>(w13_den);
    w.w2_den = static_cast<const float *>(w2_den);
    return w;
}

// The hardware conversions the quantizers' software emulation (quant_ref.h) stands for (routed_quant.cuh's PTX).
__global__ void cvt_sweep_kernel(const float *x, int64_t n, uint8_t *e4m3_hw, uint8_t *e4m3_emu, uint8_t *e2m1_hw,
                                 uint8_t *e2m1_emu) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    uint16_t pair;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(0.0f), "f"(v));
    e4m3_hw[i] = static_cast<uint8_t>(pair & 0xFFu);
    e4m3_emu[i] = ref::encode_e4m3_satfinite(v);
    uint32_t packed;
    asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %1, %2; cvt.u32.u8 %0, b; }" : "=r"(packed) : "f"(0.0f), "f"(v));
    e2m1_hw[i] = static_cast<uint8_t>(packed & 0xFu);
    e2m1_emu[i] = ref::encode_e2m1_satfinite(v);
}

}  // namespace

// x: BF16 [m, 2304] at x_stride; w: BF16 [128, 2304]; bias: FP32 [128]; ids int32 [m, 8]; weights FP32 [m, 8];
// logits FP32 [m, 128] (nullable).
CERE_KTEST int cere_ktest_inkling_router(const void *x, int64_t x_stride, int64_t m, const void *w, const void *bias,
                                         void *ids, void *weights, void *logits, int64_t mutant, cudaStream_t stream) {
    if (!fits_int(m)) return CERE_ERR_ARGUMENT;
    return status_of(cere::inkling::router(static_cast<const bf16 *>(x), x_stride, static_cast<int>(m),
                                           static_cast<const bf16 *>(w), static_cast<const float *>(bias),
                                           static_cast<int32_t *>(ids), static_cast<float *>(weights),
                                           static_cast<float *>(logits), stream, static_cast<MoeMutant>(mutant)));
}

// codes uint8 [rows, k/2], scales [rows, k/16], amax FP32 [rows].
CERE_KTEST int cere_ktest_inkling_quant_nvfp4(const void *x, int64_t x_stride, int64_t rows, int64_t k, void *codes,
                                              void *scales, void *amax, int64_t mutant, cudaStream_t stream) {
    if (!fits_int(rows) || !fits_int(k)) return CERE_ERR_ARGUMENT;
    return status_of(cere::inkling::quantize_nvfp4(static_cast<const bf16 *>(x), x_stride, static_cast<int>(rows),
                                                   static_cast<int>(k), static_cast<uint8_t *>(codes),
                                                   static_cast<uint8_t *>(scales), static_cast<float *>(amax), stream,
                                                   static_cast<MoeMutant>(mutant)));
}

// values uint8 [rows, k], exponents [rows, k/32].
CERE_KTEST int cere_ktest_inkling_quant_mxfp8(const void *x, int64_t x_stride, int64_t rows, int64_t k, void *values,
                                              void *exponents, cudaStream_t stream) {
    if (!fits_int(rows) || !fits_int(k)) return CERE_ERR_ARGUMENT;
    return status_of(cere::inkling::quantize_mxfp8(static_cast<const bf16 *>(x), x_stride, static_cast<int>(rows),
                                                   static_cast<int>(k), static_cast<uint8_t *>(values),
                                                   static_cast<uint8_t *>(exponents), stream));
}

// out_bytes: host uint64 [1].
CERE_KTEST int cere_ktest_inkling_moe_workspace_bytes(int64_t m, int64_t nvfp4, void *out_bytes, cudaStream_t) {
    if (!fits_int(m) || out_bytes == nullptr) return CERE_ERR_ARGUMENT;
    *static_cast<uint64_t *>(out_bytes) = cere::inkling::moe_workspace_bytes(static_cast<int>(m), nvfp4 != 0);
    return CERE_OK;
}

// moe_experts with taps (each nullable): gate_up BF16 [m*8, 2304], h BF16 [m*8, 1152], expert_out BF16 [m*8, 2304].
CERE_KTEST int cere_ktest_inkling_moe_experts(const void *x, int64_t x_stride, int64_t m, const void *ids,
                                              const void *weights, int64_t nvfp4, int64_t num_experts, const void *w13,
                                              const void *w13_scale, const void *w2, const void *w2_scale,
                                              const void *w13_amax, const void *w2_amax, const void *w13_den,
                                              const void *w2_den, void *workspace, int64_t workspace_bytes, void *out,
                                              int64_t out_stride, void *gate_up_tap, void *h_tap, void *expert_out_tap,
                                              int64_t mutant, cudaStream_t stream) {
    if (!fits_int(m) || !fits_int(num_experts) || workspace_bytes < 0) return CERE_ERR_ARGUMENT;
    const cere::inkling::MoeWeights w =
        weights_of(nvfp4, num_experts, w13, w13_scale, w2, w2_scale, w13_amax, w2_amax, w13_den, w2_den);
    cere::inkling::MoeTaps taps;
    taps.gate_up = static_cast<bf16 *>(gate_up_tap);
    taps.h = static_cast<bf16 *>(h_tap);
    taps.expert_out = static_cast<bf16 *>(expert_out_tap);
    return status_of(cere::inkling::moe_experts(static_cast<const bf16 *>(x), x_stride, static_cast<int>(m),
                                                static_cast<const int32_t *>(ids), static_cast<const float *>(weights),
                                                w, workspace, static_cast<size_t>(workspace_bytes),
                                                static_cast<bf16 *>(out), out_stride, stream, &taps,
                                                static_cast<MoeMutant>(mutant)));
}

// One MoE visit (router + experts) timed: `warmup` untimed then `iters` timed back-to-back visits on `stream`
// (CUDA events); out_us host float [2] = {mean router us, mean router + experts us}. ids/weights/logits scratch as
// cere_ktest_inkling_router's.
CERE_KTEST int cere_ktest_inkling_moe_time(const void *x, int64_t m, const void *gate_w, const void *gate_bias,
                                           void *ids, void *weights, int64_t nvfp4, int64_t num_experts,
                                           const void *w13, const void *w13_scale, const void *w2, const void *w2_scale,
                                           const void *w13_amax, const void *w2_amax, const void *w13_den,
                                           const void *w2_den, void *workspace, int64_t workspace_bytes, void *out,
                                           int64_t warmup, int64_t iters, void *out_us, cudaStream_t stream) {
    if (!fits_int(m) || !fits_int(num_experts) || workspace_bytes < 0 || iters < 1 || warmup < 0 ||
        out_us == nullptr)
        return CERE_ERR_ARGUMENT;
    const cere::inkling::MoeWeights w =
        weights_of(nvfp4, num_experts, w13, w13_scale, w2, w2_scale, w13_amax, w2_amax, w13_den, w2_den);
    const int mi = static_cast<int>(m);
    auto router = [&]() {
        return cere::inkling::router(static_cast<const bf16 *>(x), cere::inkling::kMoeHidden, mi,
                                     static_cast<const bf16 *>(gate_w), static_cast<const float *>(gate_bias),
                                     static_cast<int32_t *>(ids), static_cast<float *>(weights), nullptr, stream);
    };
    auto experts = [&]() {
        return cere::inkling::moe_experts(static_cast<const bf16 *>(x), cere::inkling::kMoeHidden, mi,
                                          static_cast<const int32_t *>(ids), static_cast<const float *>(weights), w,
                                          workspace, static_cast<size_t>(workspace_bytes), static_cast<bf16 *>(out),
                                          cere::inkling::kMoeHidden, stream);
    };
    cudaError_t err = cudaSuccess;
    for (int64_t i = 0; i < warmup && err == cudaSuccess; ++i) {
        err = router();
        if (err == cudaSuccess) err = experts();
    }
    if (err != cudaSuccess) return status_of(err);
    cudaEvent_t ev[3];
    for (auto &e : ev) cudaEventCreate(&e);
    float total[2] = {0.0f, 0.0f};
    for (int64_t i = 0; i < iters && err == cudaSuccess; ++i) {
        cudaEventRecord(ev[0], stream);
        err = router();
        cudaEventRecord(ev[1], stream);
        if (err == cudaSuccess) err = experts();
        cudaEventRecord(ev[2], stream);
        if (err == cudaSuccess) err = cudaEventSynchronize(ev[2]);
        float a = 0.0f, b = 0.0f;
        cudaEventElapsedTime(&a, ev[0], ev[1]);
        cudaEventElapsedTime(&b, ev[0], ev[2]);
        total[0] += a;
        total[1] += b;
    }
    for (auto &e : ev) cudaEventDestroy(e);
    if (err != cudaSuccess) return status_of(err);
    float *us = static_cast<float *>(out_us);
    us[0] = total[0] * 1000.0f / static_cast<float>(iters);
    us[1] = total[1] * 1000.0f / static_cast<float>(iters);
    return CERE_OK;
}

// x FP32 [n] (device) -> per value: the E4M3 byte of cvt.rn.satfinite.e4m3x2.f32 and of ref::encode_e4m3_satfinite,
// the E2M1 code of cvt.rn.satfinite.e2m1x2.f32 and of ref::encode_e2m1_satfinite (uint8 [n] each).
CERE_KTEST int cere_ktest_inkling_cvt_sweep(const void *x, int64_t n, void *e4m3_hw, void *e4m3_emu, void *e2m1_hw,
                                            void *e2m1_emu, cudaStream_t stream) {
    if (n < 1 || n > (int64_t(1) << 31)) return CERE_ERR_ARGUMENT;
    const int threads = 256;
    cvt_sweep_kernel<<<static_cast<unsigned>((n + threads - 1) / threads), threads, 0, stream>>>(
        static_cast<const float *>(x), n, static_cast<uint8_t *>(e4m3_hw), static_cast<uint8_t *>(e4m3_emu),
        static_cast<uint8_t *>(e2m1_hw), static_cast<uint8_t *>(e2m1_emu));
    return status_of(cudaGetLastError());
}
