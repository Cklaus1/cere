// AmdAmd PAR-6 K2-SHIFT: see k2_wmma.h.
//
// Block tile 128 (x rows) x 128 (w rows), k step 64, 256 threads = 8 waves as 2 (m) x 4 (n), each wave 64 x 32 =
// 4 x 2 WMMA tiles. x and w rows are both k-contiguous, so they are WMMA's A rows and B columns as they lie: staged
// into LDS row-major (rows padded to 40 BF16 = 80 B), the next k step's global loads in registers while this one
// computes. Per output: acc = 0; for k0 = 0, 32, ...: acc = wmma(k0 .. k0+15), ... , acc = wmma(k0+48 .. k0+63) - the
// same instruction sequence for every element, whatever its tile, wave or lane.
#include "k2_wmma.h"

namespace cere::hipk2 {
namespace {

// k step 64 (measured, m 4,096: 24-28.5 -> 30-37 TFLOP/s against 32: half the barriers and round trips per element)
constexpr int BM = 128, BN = 128, BK = 64, LDK = BK + 8, kThreads = 256;
constexpr int kQ = BM * BK / 8 / kThreads;  // 16-byte loads per thread per matrix and k step
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
using u16 = uint16_t;
struct U16x16 { uint4 lo, hi; };

__device__ inline bf16x16 ld16(const u16 *p) {
    return __builtin_bit_cast(bf16x16, U16x16{*reinterpret_cast<const uint4 *>(p), *reinterpret_cast<const uint4 *>(p + 8)});
}

template <WmmaOut kOut>
__global__ void __launch_bounds__(kThreads) gemm_kernel(const u16 *__restrict__ x, const u16 *__restrict__ w,
                                                        void *__restrict__ y, uint32_t m, uint32_t n, uint32_t k,
                                                        uint32_t kGroupM) {
    __shared__ __attribute__((aligned(16))) u16 XS[BM * LDK];
    __shared__ __attribute__((aligned(16))) u16 WS[BN * LDK];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, r = lane % 16;
    const int wm = wave / 4, wn = wave % 4;  // wave tile: rows wm*64.., cols wn*32..
    // block order: kGroupM x-row tiles run together across the n tiles. With kGroupM 1 the resident blocks share x
    // rows and sweep every w row per k step; past ~8K w rows (one 4 KB page each at k >= 2,048) that thrashes the
    // GPU's page translation, so wide n uses groups of 32 (measured, m 4,096: n 10,240 11 -> 24 TFLOP/s; n <= 6,144
    // is fastest at 1)
    const uint32_t nbn = (n + BN - 1) / BN, nbm = (m + BM - 1) / BM, lin = blockIdx.x, per = kGroupM * nbn;
    const uint32_t g0 = (lin / per) * kGroupM, gm = nbm - g0 < kGroupM ? nbm - g0 : kGroupM;
    const int64_t m0 = int64_t(g0 + (lin % per) % gm) * BM, n0 = int64_t((lin % per) / gm) * BN;
    // staging: e = tid + q * 256 (q < kQ) -> tile row e / (BK / 8), 8 k at (e % (BK / 8)) * 8
    uint4 rx[kQ], rw[kQ];
    auto fetch = [&](uint32_t k0) {
#pragma unroll
        for (int q = 0; q < kQ; ++q) {
            const int e = tid + q * kThreads, row = e / (BK / 8), c8 = (e % (BK / 8)) * 8;
            const int64_t gx = m0 + row, gw = n0 + row;
            rx[q] = gx < m ? *reinterpret_cast<const uint4 *>(x + gx * k + k0 + c8) : uint4{0, 0, 0, 0};
            rw[q] = gw < n ? *reinterpret_cast<const uint4 *>(w + gw * k + k0 + c8) : uint4{0, 0, 0, 0};
        }
    };
    f32x8 acc[4][2] = {};
    fetch(0);
    for (uint32_t k0 = 0; k0 < k; k0 += BK) {
        __syncthreads();  // the previous step's LDS reads are done
#pragma unroll
        for (int q = 0; q < kQ; ++q) {
            const int e = tid + q * kThreads, row = e / (BK / 8), c8 = (e % (BK / 8)) * 8;
            *reinterpret_cast<uint4 *>(XS + row * LDK + c8) = rx[q];
            *reinterpret_cast<uint4 *>(WS + row * LDK + c8) = rw[q];
        }
        __syncthreads();
        if (k0 + BK < k) fetch(k0 + BK);  // in flight during this step's WMMAs
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            bf16x16 a[4], b[2];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) a[mt] = ld16(XS + (wm * 64 + mt * 16 + r) * LDK + kk);
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) b[nt] = ld16(WS + (wn * 32 + nt * 16 + r) * LDK + kk);
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[mt], b[nt], acc[mt][nt]);
        }
    }
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int64_t row = m0 + wm * 64 + mt * 16 + 2 * i + lane / 16, col = n0 + wn * 32 + nt * 16 + r;
                if (row < m && col < n) {
                    if constexpr (kOut == WmmaOut::kBF16)
                        static_cast<__hip_bfloat16 *>(y)[row * n + col] = __float2bfloat16(acc[mt][nt][i]);
                    else
                        static_cast<float *>(y)[row * n + col] = acc[mt][nt][i];
                }
            }
}

}  // namespace

cere_status gemm_wmma(const __hip_bfloat16 *x, const __hip_bfloat16 *w, void *y, uint32_t m, uint32_t n, uint32_t k,
                      WmmaOut out, hipStream_t stream) {
    auto misaligned = [](const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15) != 0; };
    if (!x || !w || !y || m == 0 || n == 0 || k == 0 || k % BK != 0 || misaligned(x) || misaligned(w) || misaligned(y))
        return CERE_ERR_ARGUMENT;
    const dim3 grid(((n + BN - 1) / BN) * ((m + BM - 1) / BM));
    const uint32_t group = n > 8192 ? 32 : 1;  // block order only: no effect on any output bit
    const auto *xs = reinterpret_cast<const u16 *>(x);
    const auto *ws = reinterpret_cast<const u16 *>(w);
    if (out == WmmaOut::kBF16)
        gemm_kernel<WmmaOut::kBF16><<<grid, kThreads, 0, stream>>>(xs, ws, y, m, n, k, group);
    else
        gemm_kernel<WmmaOut::kF32><<<grid, kThreads, 0, stream>>>(xs, ws, y, m, n, k, group);
    return hipGetLastError() == hipSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

}  // namespace cere::hipk2
