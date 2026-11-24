// Inkling 2 Mini (inkling design note 3.1-3.2, G1): the depth mixes and the residual helpers; contracts in depth.h,
// the exact order in executor/ref/inkling/depth_ref.h. Compiled with -fmad=false.
#include "kernels/inkling/depth.h"

#include <climits>

#include "kernels/inkling/mix_common.cuh"
#include "kernels/inkling/mix_mutants.h"
#include "kernels/inkling/pdl.cuh"
#include "inkling/depth_ref.h"

namespace cere::inkling {

namespace {

using mix_detail::kPerThread;
using mix_detail::kRowThreads;

static_assert(kDepthMaxStates == ref::kMaxDepthStates, "depth.h and depth_ref.h agree on the stack depth");

// One block per row. The S key norms, the nq x S scores and (kNorm) the nq output norms each run as one
// block_tree256_all pass, so a call synchronizes the block for two (three) trees whatever S and nq; every tree is
// ref::reduce_tree_<256> and every value goes through the operations of depth_ref.h (then norm_ref.h) in their order.
// The query rows and norm weights do not depend on the row: they are loaded first, before the PDL wait (weights).
// mutant_w != nullptr: the fail-first mutant (mix_mutants.h), a weighted key norm.
template <bool kNorm>
__global__ void __launch_bounds__(kRowThreads)
    depth_mix_kernel(DepthStates states, const float *queries, int nq, float *out, int64_t ldo, int64_t out_qstride,
                     DepthNorm norm, const uint16_t *mutant_w) {
    pdl::trigger();
    constexpr int kS = kDepthMaxStates, kQ = kDepthMaxQueries, kTrees = kQ * kS;
    __shared__ float red[kTrees][kRowThreads];
    __shared__ float bc[kTrees];
    const int64_t row = blockIdx.x;
    const int t = threadIdx.x;
    const int S = states.count;
    float qr[kQ][kPerThread];
    [[maybe_unused]] uint16_t nw[kQ][kPerThread];
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        if (q < nq) {
#pragma unroll
            for (int j = 0; j < kPerThread; ++j) {
                qr[q][j] = queries[int64_t(q) * ref::kHidden + t + kRowThreads * j];
                if constexpr (kNorm) nw[q][j] = reinterpret_cast<const uint16_t *>(norm.w[q])[t + kRowThreads * j];
            }
        }
    }
    pdl::wait();
    // 1. k_s = the unweighted RMSNorm of state s.
    float k[kS][kPerThread];
    float ksq[kS] = {}, kss[kS];
#pragma unroll
    for (int s = 0; s < kS; ++s) {
        if (s < S) {
            const DepthState st = states.s[s];
#pragma unroll
            for (int j = 0; j < kPerThread; ++j)
                k[s][j] = mix_detail::load_any(st.ptr, st.f32 != 0, row * st.ld + t + kRowThreads * j);
#pragma unroll
            for (int j = 0; j < kPerThread; ++j) ksq[s] = ref::fma(k[s][j], k[s][j], ksq[s]);
        }
    }
    mix_detail::block_tree256_all<kS>(ksq, (1u << S) - 1u, red, bc, kss);
#pragma unroll
    for (int s = 0; s < kS; ++s) {
        if (s < S) {
            const float r = ref::rsqrt(ref::add(ref::div(kss[s], float(ref::kHidden)), ref::kRmsEps));
#pragma unroll
            for (int j = 0; j < kPerThread; ++j) {
                k[s][j] = ref::mul(k[s][j], r);
                if (mutant_w != nullptr) k[s][j] = ref::mul(ref::decode_bf16(mutant_w[t + kRowThreads * j]), k[s][j]);
            }
        }
    }
    // 2. scores, tree q * kS + s.
    float dot[kTrees] = {}, sc[kTrees];
    uint32_t score_mask = 0;
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
#pragma unroll
        for (int s = 0; s < kS; ++s) {
            if (q < nq && s < S) {
#pragma unroll
                for (int j = 0; j < kPerThread; ++j) dot[q * kS + s] = ref::fma(qr[q][j], k[s][j], dot[q * kS + s]);
                score_mask |= 1u << (q * kS + s);
            }
        }
    }
    mix_detail::block_tree256_all<kTrees>(dot, score_mask, red, bc, sc);
    // 3. per query: the softmax over the states, the mix (4. with kNorm: its sum of squares).
    [[maybe_unused]] float o[kQ][kPerThread];
    [[maybe_unused]] float osq[kQ] = {}, oss[kQ];
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
        if (q < nq) {
            float score[kS], p[kS];
            float m = -INFINITY;
#pragma unroll
            for (int s = 0; s < kS; ++s) {
                if (s < S) {
                    score[s] = ref::mul(sc[q * kS + s], ref::kDepthScale);
                    m = ref::max(m, score[s]);
                }
            }
            float sum = 0.0f;
#pragma unroll
            for (int s = 0; s < kS; ++s) {
                if (s < S) {
                    p[s] = ref::exp(ref::sub(score[s], m));
                    sum = ref::add(sum, p[s]);
                }
            }
#pragma unroll
            for (int s = 0; s < kS; ++s)
                if (s < S) p[s] = ref::div(p[s], sum);
#pragma unroll
            for (int j = 0; j < kPerThread; ++j) {
                float acc = 0.0f;
#pragma unroll
                for (int s = 0; s < kS; ++s)
                    if (s < S) acc = ref::fma(p[s], k[s][j], acc);
                if constexpr (kNorm) {
                    o[q][j] = acc;
                    osq[q] = ref::fma(acc, acc, osq[q]);
                } else {
                    out[int64_t(q) * out_qstride + row * ldo + t + kRowThreads * j] = acc;
                }
            }
        }
    }
    if constexpr (kNorm) {
        // 4. y_q = bf16(w_q x (o_q x rms_inv(o_q))), norm.h's rms_norm of the FP32 row.
        mix_detail::block_tree256_all<kQ>(osq, (1u << nq) - 1u, red, bc, oss);
#pragma unroll
        for (int q = 0; q < kQ; ++q) {
            if (q < nq) {
                const float r = ref::rsqrt(ref::add(ref::div(oss[q], float(ref::kHidden)), norm.eps));
                uint16_t *y = reinterpret_cast<uint16_t *>(norm.y[q]) + row * norm.ldy;
#pragma unroll
                for (int j = 0; j < kPerThread; ++j)
                    y[t + kRowThreads * j] =
                        ref::encode_bf16(ref::mul(ref::decode_bf16(nw[q][j]), ref::mul(o[q][j], r)));
            }
        }
    }
}

constexpr int kElemThreads = 256;
constexpr int kElemBlocksPerRow = (ref::kHidden + kElemThreads - 1) / kElemThreads;  // 9
constexpr int64_t kMaxGridY = 65535;

// h is nullptr at a boundary (not read).
__global__ void __launch_bounds__(kElemThreads)
    attn_residual_kernel(bool boundary, const uint16_t *h, int64_t ldh, const uint16_t *o, int64_t ldo, float *r,
                         int64_t ldr) {
    pdl::enter();
    const int64_t row = blockIdx.y;
    const int i = blockIdx.x * kElemThreads + threadIdx.x;
    const uint16_t hv = boundary ? uint16_t(0) : h[row * ldh + i];
    r[row * ldr + i] = ref::attn_residual(boundary, hv, o[row * ldo + i]);
}

__global__ void __launch_bounds__(kElemThreads)
    layer_out_kernel(const float *r, int64_t ldr, const uint16_t *d, int64_t ldd, uint16_t *h, int64_t ldh) {
    pdl::enter();
    const int64_t row = blockIdx.y;
    const int i = blockIdx.x * kElemThreads + threadIdx.x;
    h[row * ldh + i] = ref::layer_out(r[row * ldr + i], d[row * ldd + i]);
}

bool rows_ok(int64_t rows) { return rows >= 1 && rows <= INT_MAX; }

bool mix_ok(const DepthStates &states, int64_t rows, const float *queries, int nq) {
    if (states.count < 1 || states.count > kDepthMaxStates || !rows_ok(rows) || queries == nullptr || nq < 1 ||
        nq > kDepthMaxQueries)
        return false;
    for (int s = 0; s < states.count; ++s)
        if (states.s[s].ptr == nullptr || states.s[s].ld < ref::kHidden) return false;
    return true;
}

cudaError_t depth_launch(const DepthStates &states, int64_t rows, const float *queries, int nq, float *out,
                         int64_t ldo, int64_t out_qstride, const uint16_t *mutant_w, cudaStream_t stream) {
    if (!mix_ok(states, rows, queries, nq) || out == nullptr || ldo < ref::kHidden || (nq > 1 && out_qstride == 0))
        return cudaErrorInvalidValue;
    return pdl::launch(depth_mix_kernel<false>, dim3(unsigned(rows)), dim3(kRowThreads), 0, stream, states, queries, nq,
                       out, ldo, out_qstride, DepthNorm{}, mutant_w);
}

}  // namespace

cudaError_t depth_mix(const DepthStates &states, int64_t rows, const float *queries, int nq, float *out, int64_t ldo,
                      int64_t out_qstride, cudaStream_t stream) {
    return depth_launch(states, rows, queries, nq, out, ldo, out_qstride, nullptr, stream);
}

cudaError_t depth_mix_norm(const DepthStates &states, int64_t rows, const float *queries, int nq,
                           const DepthNorm &norm, cudaStream_t stream) {
    if (!mix_ok(states, rows, queries, nq) || norm.ldy < ref::kHidden || !(norm.eps >= 0.0f))
        return cudaErrorInvalidValue;
    for (int q = 0; q < nq; ++q)
        if (norm.w[q] == nullptr || norm.y[q] == nullptr) return cudaErrorInvalidValue;
    return pdl::launch(depth_mix_kernel<true>, dim3(unsigned(rows)), dim3(kRowThreads), 0, stream, states, queries, nq,
                       static_cast<float *>(nullptr), int64_t(0), int64_t(0), norm,
                       static_cast<const uint16_t *>(nullptr));
}

cudaError_t attn_residual(bool boundary, const bf16 *h, int64_t ldh, const bf16 *o, int64_t ldo, int64_t rows,
                          float *r, int64_t ldr, cudaStream_t stream) {
    if ((!boundary && (h == nullptr || ldh < ref::kHidden)) || o == nullptr || r == nullptr || ldo < ref::kHidden ||
        ldr < ref::kHidden || !rows_ok(rows))
        return cudaErrorInvalidValue;
    for (int64_t r0 = 0; r0 < rows; r0 += kMaxGridY) {
        const int64_t n = rows - r0 < kMaxGridY ? rows - r0 : kMaxGridY;
        const cudaError_t err =
            pdl::launch(attn_residual_kernel, dim3(kElemBlocksPerRow, unsigned(n)), dim3(kElemThreads), 0, stream,
                        boundary, boundary ? nullptr : reinterpret_cast<const uint16_t *>(h) + r0 * ldh, ldh,
                        reinterpret_cast<const uint16_t *>(o) + r0 * ldo, ldo, r + r0 * ldr, ldr);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

cudaError_t layer_out(const float *r, int64_t ldr, const bf16 *d, int64_t ldd, int64_t rows, bf16 *h_next,
                      int64_t ldh, cudaStream_t stream) {
    if (r == nullptr || d == nullptr || h_next == nullptr || ldr < ref::kHidden || ldd < ref::kHidden ||
        ldh < ref::kHidden || static_cast<const void *>(h_next) == static_cast<const void *>(r) ||
        (h_next == d && ldh != ldd) || !rows_ok(rows))
        return cudaErrorInvalidValue;
    for (int64_t r0 = 0; r0 < rows; r0 += kMaxGridY) {
        const int64_t n = rows - r0 < kMaxGridY ? rows - r0 : kMaxGridY;
        const cudaError_t err =
            pdl::launch(layer_out_kernel, dim3(kElemBlocksPerRow, unsigned(n)), dim3(kElemThreads), 0, stream,
                        r + r0 * ldr, ldr, reinterpret_cast<const uint16_t *>(d) + r0 * ldd, ldd,
                        reinterpret_cast<uint16_t *>(h_next) + r0 * ldh, ldh);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

namespace mutant {

cudaError_t depth_mix_weighted_norm(const DepthStates &states, int64_t rows, const float *queries, int nq,
                                    float *out, int64_t ldo, int64_t out_qstride, const bf16 *w,
                                    cudaStream_t stream) {
    if (w == nullptr) return cudaErrorInvalidValue;
    return depth_launch(states, rows, queries, nq, out, ldo, out_qstride, reinterpret_cast<const uint16_t *>(w),
                        stream);
}

}  // namespace mutant

}  // namespace cere::inkling
