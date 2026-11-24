// AmdAmd PAR-6 step 14: k2_mgemv.h's small-M GEMM and MXFP8 upcast.
#include "k2_mgemv.h"

namespace cere::hipk2 {
namespace {

constexpr int kWaves = 8;  // columns per block
constexpr int kStep = 512; // k elements per wave step (32 lanes x 16)

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

// 4 E4M3 bytes -> their exact values times 2^(s - 127) (sc = that power, a normal float). (b & 0x7f) << 20 is the
// E4M3 value times 2^-120 as a float (normal and subnormal alike); the products are exact (the contract keeps every
// weight a normal BF16).
__device__ __forceinline__ void e4m3x4(uint32_t q, float sc, float (&o)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t b = (q >> (8 * i)) & 0xFFu;
        const float v = __uint_as_float((b & 0x7Fu) << 20) * 0x1p120f * sc;
        o[i] = __uint_as_float(__float_as_uint(v) | ((b & 0x80u) << 24));
    }
}

template <bool kMx, int kMR>
__global__ void __launch_bounds__(kWaves * 32) gemv_kernel(const __hip_bfloat16 *__restrict__ x,
                                                            const void *__restrict__ w,
                                                            const uint8_t *__restrict__ scales, void *__restrict__ y,
                                                            uint32_t m, uint32_t n, uint32_t k, bool out_f32) {
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t col = blockIdx.x * kWaves + threadIdx.x / 32;
    if (col >= n) return;  // wave-uniform
    float acc[kMR];
#pragma unroll
    for (int r = 0; r < kMR; ++r) acc[r] = 0.0f;
    const uint16_t *xs = reinterpret_cast<const uint16_t *>(x);
    for (uint32_t i0 = lane * 16; i0 < k; i0 += kStep) {
        uint32_t wp[8];  // the 16 weights as BF16 pairs (an MXFP8 value is exact in BF16: the FP32 decode's high half)
        if constexpr (kMx) {
            const uint4 q = *reinterpret_cast<const uint4 *>(static_cast<const uint8_t *>(w) + size_t(col) * k + i0);
            const float sc = __uint_as_float(uint32_t(scales[size_t(col) * (k / 32) + i0 / 32]) << 23);
            const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float t[4];
                e4m3x4(qw[j], sc, t);
                wp[2 * j] = __builtin_amdgcn_perm(__float_as_uint(t[1]), __float_as_uint(t[0]), 0x07060302u);
                wp[2 * j + 1] = __builtin_amdgcn_perm(__float_as_uint(t[3]), __float_as_uint(t[2]), 0x07060302u);
            }
        } else {
            const uint4 *p = reinterpret_cast<const uint4 *>(static_cast<const uint16_t *>(w) + size_t(col) * k + i0);
            const uint4 a = p[0], b = p[1];
            wp[0] = a.x; wp[1] = a.y; wp[2] = a.z; wp[3] = a.w; wp[4] = b.x; wp[5] = b.y; wp[6] = b.z; wp[7] = b.w;
        }
#pragma unroll
        for (int r = 0; r < kMR; ++r) {
            if (uint32_t(r) >= m) continue;  // uniform
            const uint4 *px = reinterpret_cast<const uint4 *>(xs + size_t(r) * k + i0);
            const uint4 a = px[0], b = px[1];
            const uint32_t u[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            float s = acc[r];
#pragma unroll
            for (int j = 0; j < 8; ++j)  // v_dot2_f32_bf16: two BF16 products and the add in one instruction
                s = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, wp[j]), __builtin_bit_cast(bf16x2, u[j]), s, false);
            acc[r] = s;
        }
    }
#pragma unroll
    for (int r = 0; r < kMR; ++r) {
        float v = acc[r];
#pragma unroll
        for (int o = 16; o >= 1; o /= 2) v += __shfl_xor(v, o, 32);  // every lane ends with the same bits
        acc[r] = v;
    }
#pragma unroll
    for (int r = 0; r < kMR; ++r) {
        if (uint32_t(r) < m && lane == uint32_t(r)) {
            if (out_f32) static_cast<float *>(y)[size_t(r) * n + col] = acc[r];
            else static_cast<__hip_bfloat16 *>(y)[size_t(r) * n + col] = __float2bfloat16(acc[r]);
        }
    }
}

__global__ void dequant_kernel(const uint8_t *__restrict__ q, const uint8_t *__restrict__ scales, size_t groups16,
                               uint32_t k, __hip_bfloat16 *__restrict__ w) {
    const size_t g = size_t(blockIdx.x) * blockDim.x + threadIdx.x;  // 16 elements of one row
    if (g >= groups16) return;
    const size_t e0 = g * 16;
    const size_t row = e0 / k, i0 = e0 % k;
    const uint4 v = *reinterpret_cast<const uint4 *>(q + e0);
    const float sc = __uint_as_float(uint32_t(scales[row * (k / 32) + i0 / 32]) << 23);
    const uint32_t u[4] = {v.x, v.y, v.z, v.w};
    uint32_t o[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float t[4];
        e4m3x4(u[j], sc, t);  // exact BF16 values: truncation is exact
        o[2 * j] = (__float_as_uint(t[0]) >> 16) | (__float_as_uint(t[1]) & 0xFFFF0000u);
        o[2 * j + 1] = (__float_as_uint(t[2]) >> 16) | (__float_as_uint(t[3]) & 0xFFFF0000u);
    }
    uint4 *dst = reinterpret_cast<uint4 *>(w + e0);
    dst[0] = uint4{o[0], o[1], o[2], o[3]};
    dst[1] = uint4{o[4], o[5], o[6], o[7]};
}

template <bool kMx>
cere_status launch(const __hip_bfloat16 *x, const void *w, const uint8_t *s, void *y, uint32_t m, uint32_t n,
                   uint32_t k, bool f32, hipStream_t st) {
    const dim3 grid((n + kWaves - 1) / kWaves), block(kWaves * 32);
    if (m == 1) gemv_kernel<kMx, 1><<<grid, block, 0, st>>>(x, w, s, y, m, n, k, f32);
    else if (m == 2) gemv_kernel<kMx, 2><<<grid, block, 0, st>>>(x, w, s, y, m, n, k, f32);
    else if (m <= 4) gemv_kernel<kMx, 4><<<grid, block, 0, st>>>(x, w, s, y, m, n, k, f32);
    else if (m <= 8) gemv_kernel<kMx, 8><<<grid, block, 0, st>>>(x, w, s, y, m, n, k, f32);
    else gemv_kernel<kMx, 16><<<grid, block, 0, st>>>(x, w, s, y, m, n, k, f32);
    return hipGetLastError() == hipSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

bool aligned16(const void *p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

}  // namespace

cere_status gemv(const __hip_bfloat16 *x, const void *w, const uint8_t *scales, void *y, uint32_t m, uint32_t n,
                 uint32_t k, bool out_f32, hipStream_t stream) {
    if (!x || !w || !y || m < 1 || m > kGemvMaxM || n < 1 || k < 1 || k % (scales ? 32 : 16) != 0 || !aligned16(x) ||
        !aligned16(w) || !aligned16(y))
        return CERE_ERR_ARGUMENT;
    return scales ? launch<true>(x, w, scales, y, m, n, k, out_f32, stream)
                  : launch<false>(x, w, nullptr, y, m, n, k, out_f32, stream);
}

hipError_t mxfp8_dequantize(const uint8_t *q, const uint8_t *scales, uint32_t n, uint32_t k, __hip_bfloat16 *w,
                            hipStream_t stream) {
    if (!q || !scales || !w || n == 0 || k == 0 || k % 32 != 0 || !aligned16(q) || !aligned16(w))
        return hipErrorInvalidValue;
    const size_t groups = size_t(n) * k / 16;
    dequant_kernel<<<dim3((groups + 255) / 256), dim3(256), 0, stream>>>(q, scales, groups, k, w);
    return hipGetLastError();
}

}  // namespace cere::hipk2
