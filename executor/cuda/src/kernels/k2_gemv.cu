// K2 gemvx-order rows kernel: contract and order in k2_gemv.h (m23 design note P11R-21).
//
// Mapping: G = T / L threads cooperate on one output; thread g holds the L consecutive leaves g * L .. g * L + L - 1,
// so per chunk it reads E = L * S consecutive elements of the weight row (one vector load, or one BF16 when E = 1)
// and the same elements of every x row it serves; element e of that vector feeds leaf g * L + e / S, in increasing e
// (each leaf's chain stays in increasing index). The halving tree's levels h >= L pair thread g with thread g + h / L
// (warp shuffles within the G-lane group, the same pairings), the levels h < L run in registers. A thread group
// serves J outputs (P = B / G apart within the CTA's tile of P * J) and a CTA serves MR rows (rows past m read
// nothing and are not stored); m > MR runs ceil(m / MR) row groups as adjacent CTAs of one output tile, so the
// weight tile is still hot in L2 when the next group reads it. None of B, L, U, MR, J or the L2 prefetch size of the
// weight loads changes an output bit (every leaf's chain and the tree are fixed by T and S): they are chosen per
// shape and m by speed only (dispatch below).
#include "kernels/k2_gemv.h"

namespace cere::kernels::k2v {
namespace {

constexpr size_t kCublasLt = 130101;  // the libcublasLt whose gemvx orders were identified (k2_gemm.cpp kPinsCublasLt)
constexpr int kGemvxAlgo = 13;

// Read-only global loads of 16, 8, 4 or 2 bytes; PF 256 adds the L2::256B prefetch hint (the weights stream once),
// PF 0 is a plain __ldg.
template <int PF>
__device__ __forceinline__ uint4 ld16(const void *p) {
    if constexpr (PF == 256) {
        uint4 r;
        asm("ld.global.nc.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
        return r;
    } else {
        return __ldg(static_cast<const uint4 *>(p));
    }
}

template <int PF>
__device__ __forceinline__ uint2 ld8(const void *p) {
    if constexpr (PF == 256) {
        uint2 r;
        asm("ld.global.nc.L2::256B.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
        return r;
    } else {
        return __ldg(static_cast<const uint2 *>(p));
    }
}

template <int PF>
__device__ __forceinline__ uint32_t ld4(const void *p) {
    if constexpr (PF == 256) {
        uint32_t r;
        asm("ld.global.nc.L2::256B.u32 %0, [%1];" : "=r"(r) : "l"(p));
        return r;
    } else {
        return __ldg(static_cast<const unsigned int *>(p));
    }
}

template <int PF>
__device__ __forceinline__ uint32_t ld2(const void *p) {
    if constexpr (PF == 256) {
        unsigned short r;
        asm("ld.global.nc.L2::256B.u16 %0, [%1];" : "=h"(r) : "l"(p));
        return r;
    } else {
        return __ldg(static_cast<const unsigned short *>(p));
    }
}

// E BF16 elements in 32-bit words (E = 1: one element in the low half of one word).
template <int E>
struct Vec {
    uint32_t v[E == 1 ? 1 : E / 2];
};

// E consecutive elements from an address aligned to their size (E = 1 or even), in the widest vector loads that
// divide them.
template <int E, int PF>
__device__ __forceinline__ void load(Vec<E> &d, const __nv_bfloat16 *p) {
    if constexpr (E == 1) {
        d.v[0] = ld2<PF>(p);
    } else if constexpr ((E / 2) % 4 == 0) {
#pragma unroll
        for (int i = 0; i < E / 2; i += 4) {
            const uint4 q = ld16<PF>(reinterpret_cast<const uint4 *>(p) + i / 4);
            d.v[i] = q.x;
            d.v[i + 1] = q.y;
            d.v[i + 2] = q.z;
            d.v[i + 3] = q.w;
        }
    } else if constexpr ((E / 2) % 2 == 0) {
#pragma unroll
        for (int i = 0; i < E / 2; i += 2) {
            const uint2 q = ld8<PF>(reinterpret_cast<const uint2 *>(p) + i / 2);
            d.v[i] = q.x;
            d.v[i + 1] = q.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < E / 2; ++i) {
            d.v[i] = ld4<PF>(reinterpret_cast<const unsigned int *>(p) + i);
        }
    }
}

// Element e of a loaded vector as FP32 (exact): the lower address is the low half of the word.
template <int E>
__device__ __forceinline__ float element(const Vec<E> &d, int e) {
    if constexpr (E == 1) {
        return __uint_as_float(d.v[0] << 16);
    } else {
        const uint32_t word = d.v[e / 2];
        return __uint_as_float((e & 1) == 0 ? word << 16 : word & 0xffff0000u);
    }
}

template <int B, int T, int S, int L, int U, int MR, int J, int PF, bool F32>
__global__ void __launch_bounds__(B)
    gemv_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ w, void *__restrict__ y,
                uint32_t m, uint32_t n, uint32_t k, uint32_t groups) {
    constexpr int G = T / L;
    constexpr int E = L * S;
    constexpr int C = T * S;
    constexpr int P = B / G;  // outputs per CTA per q
    static_assert(T % L == 0 && G <= 32 && B % G == 0 && (E == 1 || E % 2 == 0), "a thread holds whole leaves");
    const uint32_t rg = blockIdx.x % groups;
    const uint32_t tile = blockIdx.x / groups;
    const uint32_t g = threadIdx.x % G;
    const uint32_t jj = threadIdx.x / G;
    const uint32_t r0 = rg * MR;
    const uint32_t rows = m - r0;  // >= 1: live rows of this group (only r < rows is read or stored)
    // Outputs past n run a clamped row so the whole warp joins the shuffles; they store nothing.
    const __nv_bfloat16 *wp[J];
#pragma unroll
    for (int q = 0; q < J; ++q) {
        const uint32_t j = tile * (P * J) + q * P + jj;
        wp[q] = w + size_t(j < n ? j : n - 1) * k + g * E;
    }
    const __nv_bfloat16 *xp = x + size_t(r0) * k + g * E;
    float acc[J][MR][L];
#pragma unroll
    for (int q = 0; q < J; ++q) {
#pragma unroll
        for (int r = 0; r < MR; ++r) {
#pragma unroll
            for (int l = 0; l < L; ++l) {
                acc[q][r][l] = 0.0f;
            }
        }
    }
    // One chunk of every live row: acc[q][r][e / S] = fma(w_qe, x_re, acc[q][r][e / S]) in increasing e.
    auto step = [&](const Vec<E>(&wq)[J], uint32_t c) {
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            if (static_cast<uint32_t>(r) < rows) {
                Vec<E> xv;
                load<E, 0>(xv, xp + size_t(r) * k + size_t(c) * C);
#pragma unroll
                for (int q = 0; q < J; ++q) {
#pragma unroll
                    for (int e = 0; e < E; ++e) {
                        acc[q][r][e / S] = __fmaf_rn(element<E>(wq[q], e), element<E>(xv, e), acc[q][r][e / S]);
                    }
                }
            }
        }
    };
    const uint32_t chunks = k / C;
    uint32_t c = 0;
    for (; c + U <= chunks; c += U) {
        Vec<E> wv[U][J];
#pragma unroll
        for (int u = 0; u < U; ++u) {
#pragma unroll
            for (int q = 0; q < J; ++q) {
                load<E, PF>(wv[u][q], wp[q] + size_t(c + u) * C);
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            step(wv[u], c + u);
        }
    }
    for (; c < chunks; ++c) {
        Vec<E> wv[J];
#pragma unroll
        for (int q = 0; q < J; ++q) {
            load<E, PF>(wv[q], wp[q] + size_t(c) * C);
        }
        step(wv, c);
    }
    // Halving tree: levels h >= L across the group's threads, then h < L in registers.
#pragma unroll
    for (int h = T / 2; h >= L; h /= 2) {
#pragma unroll
        for (int q = 0; q < J; ++q) {
#pragma unroll
            for (int r = 0; r < MR; ++r) {
#pragma unroll
                for (int l = 0; l < L; ++l) {
                    acc[q][r][l] = __fadd_rn(acc[q][r][l], __shfl_down_sync(0xffffffffu, acc[q][r][l], h / L, G));
                }
            }
        }
    }
#pragma unroll
    for (int h = L / 2; h >= 1; h /= 2) {
#pragma unroll
        for (int q = 0; q < J; ++q) {
#pragma unroll
            for (int r = 0; r < MR; ++r) {
#pragma unroll
                for (int l = 0; l < h; ++l) {
                    acc[q][r][l] = __fadd_rn(acc[q][r][l], acc[q][r][l + h]);
                }
            }
        }
    }
    if (g != 0) {
        return;
    }
#pragma unroll
    for (int q = 0; q < J; ++q) {
        const uint32_t j = tile * (P * J) + q * P + jj;
        if (j >= n) {
            continue;
        }
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            if (static_cast<uint32_t>(r) < rows) {
                const float v = __fadd_rn(acc[q][r][0], 0.0f);  // the epilogue's + 0: -0 -> +0
                const size_t at = size_t(r0 + r) * n + j;
                if constexpr (F32) {
                    static_cast<float *>(y)[at] = v;
                } else {
                    static_cast<__nv_bfloat16 *>(y)[at] = __float2bfloat16_rn(v);
                }
            }
        }
    }
}

template <int B, int T, int S, int L, int U, int MR, int J, int PF, bool F32>
cudaError_t launch(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const __nv_bfloat16 *w, uint32_t n, void *y,
                   cudaStream_t stream) {
    constexpr uint32_t per_tile = B / (T / L) * J;
    const uint32_t groups = (m + MR - 1) / MR;
    const uint32_t tiles = (n + per_tile - 1) / per_tile;
    gemv_kernel<B, T, S, L, U, MR, J, PF, F32><<<tiles * groups, B, 0, stream>>>(x, w, y, m, n, k, groups);
    return cudaGetLastError();
}

// The fastest measured configuration per order, shape and m (bit-equal by construction; cold weights, GPU0, 325 W,
// every m in 1..16 against the previous fixed configuration: m23 design note P11R-21 S4c).
template <bool F32>
cudaError_t dispatch(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const __nv_bfloat16 *w, uint32_t n, void *y,
                     Order o, cudaStream_t stream) {
    if (o.lanes == 8) {
        // lm_head, MTP hot lm_head: 4 threads per output (2 leaves each), 4 outputs per thread group, 16 chunks in
        // flight.
        return launch<256, 8, 1, 2, 16, 4, 4, 256, F32>(x, m, k, w, n, y, stream);
    }
    if (o.chunk == 20) {
        // The blocked order (customOption 82 at k 640: one chunk of 20 elements per leaf): one leaf per thread.
        return launch<64, 32, 20, 1, 1, 4, 1, 0, F32>(x, m, k, w, n, y, stream);
    }
    if (n < 2048) {
        // Narrow n (router, shared gate/up, GDN b/a, hc down): one 32-thread CTA per output and row group, one leaf
        // (one BF16) per thread, many chunks in flight; more rows per CTA only where m is large enough to pay.
        if (n < 128 || m <= 6) {
            return launch<32, 32, 1, 1, 64, 1, 1, 0, F32>(x, m, k, w, n, y, stream);
        }
        if (m <= 12) {
            return launch<32, 32, 1, 1, 64, 2, 1, 0, F32>(x, m, k, w, n, y, stream);
        }
        if (k >= 8192) {
            return launch<32, 32, 1, 2, 32, 2, 1, 0, F32>(x, m, k, w, n, y, stream);
        }
        return launch<32, 32, 1, 2, 16, 4, 1, 0, F32>(x, m, k, w, n, y, stream);
    }
    // Wide n (attention / GDN out_proj, GDN z, MTP fc halves).
    if (m <= 4) {
        return launch<32, 32, 1, 1, 32, 4, 1, 0, F32>(x, m, k, w, n, y, stream);
    }
    if (n >= 4096) {
        return launch<64, 32, 1, 2, 16, 4, 2, 256, F32>(x, m, k, w, n, y, stream);
    }
    if (m <= 12) {
        return launch<32, 32, 1, 2, 32, 4, 1, 256, F32>(x, m, k, w, n, y, stream);
    }
    if (k >= 4096) {
        return launch<64, 32, 1, 1, 32, 8, 2, 0, F32>(x, m, k, w, n, y, stream);
    }
    return launch<64, 32, 1, 2, 32, 4, 1, 256, F32>(x, m, k, w, n, y, stream);
}

bool aligned(const void *p, uintptr_t a) { return (reinterpret_cast<uintptr_t>(p) & (a - 1)) == 0; }

}  // namespace

Order order_of(size_t cublaslt_version, int algo_id, uint32_t custom_option, uint32_t k) {
    if (cublaslt_version != kCublasLt || algo_id != kGemvxAlgo) {
        return {};
    }
    if ((custom_option == 37 || custom_option == 75) && (k == 2560 || k == 6144 || k == 10240)) {
        return {32, 1};
    }
    if ((custom_option == 89 || custom_option == 71) && k == 2560) {
        return {8, 1};
    }
    if (custom_option == 82 && k == 640) {
        return {32, 20};  // the blocked variant: S = k / T, one chunk (verified at k 640 only)
    }
    return {};
}

bool supported(Order o, uint32_t k) {
    const bool known = ((o.lanes == 8 || o.lanes == 32) && o.chunk == 1) || (o.lanes == 32 && o.chunk == 20);
    return known && k > 0 && k % (o.lanes * o.chunk) == 0 && k % 8 == 0;
}

cudaError_t gemv(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const __nv_bfloat16 *w, uint32_t n, void *y, bool f32,
                 Order o, cudaStream_t stream) {
    if (x == nullptr || w == nullptr || y == nullptr || m == 0 || m > kMaxM || n == 0 || !supported(o, k) ||
        !aligned(x, 16) || !aligned(w, 16) || !aligned(y, f32 ? 4 : 2)) {
        return cudaErrorInvalidValue;
    }
    return f32 ? dispatch<true>(x, m, k, w, n, y, o, stream) : dispatch<false>(x, m, k, w, n, y, o, stream);
}

}  // namespace cere::kernels::k2v
