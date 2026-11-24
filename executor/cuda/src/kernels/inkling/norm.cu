// Inkling 2 Mini (inkling design note 3.2, G1): RMSNorm and the normed embedding; contracts in norm.h, the exact
// order in executor/ref/inkling/norm_ref.h. Compiled with -fmad=false: every fused multiply-add is ref::fma.
#include "kernels/inkling/norm.h"

#include <climits>

#include "kernels/inkling/mix_common.cuh"
#include "kernels/inkling/pdl.cuh"

namespace cere::inkling {

namespace {

using mix_detail::kPerThread;
using mix_detail::kRowThreads;

__global__ void __launch_bounds__(kRowThreads)
    rms_norm_kernel(const void *x, bool x_f32, int64_t ldx, const uint16_t *w, float eps, void *y, bool y_f32,
                    int64_t ldy) {
    pdl::trigger();
    __shared__ float red[kRowThreads];
    const int64_t row = blockIdx.x;
    const int t = threadIdx.x;
    uint16_t wv[kPerThread];  // the weights, loaded before the PDL wait (they do not depend on the row)
    if (w != nullptr) {
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) wv[j] = w[t + kRowThreads * j];
    }
    pdl::wait();
    float v[kPerThread];
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) v[j] = mix_detail::load_any(x, x_f32, row * ldx + t + kRowThreads * j);
    const float r = mix_detail::row_rms_inv(v, eps, red);
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int i = t + kRowThreads * j;
        float o = ref::mul(v[j], r);
        if (w != nullptr) o = ref::mul(ref::decode_bf16(wv[j]), o);
        if (y_f32) static_cast<float *>(y)[row * ldy + i] = o;
        else static_cast<uint16_t *>(y)[row * ldy + i] = ref::encode_bf16(o);
    }
}

__global__ void __launch_bounds__(kRowThreads)
    embed_norm_kernel(const uint16_t *table, const uint32_t *ids, const uint16_t *w, float eps, uint16_t *out,
                      int64_t ldo) {
    pdl::enter();
    __shared__ float red[kRowThreads];
    const int64_t row = blockIdx.x;
    const int t = threadIdx.x;
    const uint16_t *src = table + int64_t(ids[row]) * ref::kHidden;
    float v[kPerThread];
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) v[j] = ref::decode_bf16(src[t + kRowThreads * j]);
    const float r = mix_detail::row_rms_inv(v, eps, red);
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int i = t + kRowThreads * j;
        out[row * ldo + i] = ref::encode_bf16(ref::mul(ref::decode_bf16(w[i]), ref::mul(v[j], r)));
    }
}

bool rows_ok(int64_t rows) { return rows >= 1 && rows <= INT_MAX; }

}  // namespace

cudaError_t rms_norm(const void *x, bool x_f32, int64_t ldx, const bf16 *w, float eps, int64_t rows, void *y,
                     bool y_f32, int64_t ldy, cudaStream_t stream) {
    if (x == nullptr || y == nullptr || !rows_ok(rows) || ldx < ref::kHidden || ldy < ref::kHidden ||
        !(eps >= 0.0f))
        return cudaErrorInvalidValue;
    if (x == y && (x_f32 != y_f32 || ldx != ldy)) return cudaErrorInvalidValue;
    return pdl::launch(rms_norm_kernel, dim3(unsigned(rows)), dim3(kRowThreads), 0, stream, x, x_f32, ldx,
                       reinterpret_cast<const uint16_t *>(w), eps, y, y_f32, ldy);
}

cudaError_t embed_norm(const bf16 *table, const uint32_t *ids, int64_t rows, const bf16 *w, float eps, bf16 *out,
                       int64_t ldo, cudaStream_t stream) {
    if (table == nullptr || ids == nullptr || w == nullptr || out == nullptr || !rows_ok(rows) ||
        ldo < ref::kHidden || !(eps >= 0.0f))
        return cudaErrorInvalidValue;
    return pdl::launch(embed_norm_kernel, dim3(unsigned(rows)), dim3(kRowThreads), 0, stream,
                       reinterpret_cast<const uint16_t *>(table), ids, reinterpret_cast<const uint16_t *>(w), eps,
                       reinterpret_cast<uint16_t *>(out), ldo);
}

}  // namespace cere::inkling
