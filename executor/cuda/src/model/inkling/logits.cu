// Inkling 2 Mini S5: logits.h.
#include "model/inkling/logits.h"

#include <cooperative_groups.h>

#include "kernels/inkling/pdl.cuh"

namespace cere::inkling {
namespace {

constexpr int kThreads = 256;

__global__ void widen_kernel(const __nv_bfloat16 *in, int64_t ldi, int64_t vocab, float *out, int64_t ldo) {
    pdl::enter();
    const int64_t r = blockIdx.y;
    for (int64_t c = int64_t(blockIdx.x) * kThreads + threadIdx.x; c < vocab; c += int64_t(gridDim.x) * kThreads) {
        out[r * ldo + c] = __bfloat162float(in[r * ldi + c]);
    }
}

// (value, id) a beats b: NaN beats everything but an earlier NaN; otherwise greater, ties to the lower id. A strict
// total order over distinct ids, so the best two of a row do not depend on the order entries are merged in.
__device__ bool beats(float va, int32_t ia, float vb, int32_t ib) {
    const bool na = va != va, nb = vb != vb;
    if (na || nb) return na && (!nb || ia < ib);
    return va > vb || (va == vb && ia < ib);
}

// The best two (value, id) entries of a set, [0] beating [1]; id -1 = empty.
struct Top2 {
    float v[2];
    int32_t i[2];
};

__device__ void top2_push(Top2 &t, float v, int32_t i) {
    if (i < 0) return;
    if (t.i[0] < 0 || beats(v, i, t.v[0], t.i[0])) {
        t.v[1] = t.v[0];
        t.i[1] = t.i[0];
        t.v[0] = v;
        t.i[0] = i;
    } else if (t.i[1] < 0 || beats(v, i, t.v[1], t.i[1])) {
        t.v[1] = v;
        t.i[1] = i;
    }
}

// t = the best two of the warp's sets (xor butterfly; the lanes' sets are disjoint).
__device__ void top2_warp(Top2 &t) {
#pragma unroll
    for (int off = 16; off >= 1; off /= 2) {
        const float v0 = __shfl_xor_sync(0xffffffffu, t.v[0], off), v1 = __shfl_xor_sync(0xffffffffu, t.v[1], off);
        const int32_t i0 = __shfl_xor_sync(0xffffffffu, t.i[0], off), i1 = __shfl_xor_sync(0xffffffffu, t.i[1], off);
        top2_push(t, v0, i0);
        top2_push(t, v1, i1);
    }
}

// One cluster of kTopCluster blocks per row: entry c goes to thread c % (kTopCluster * kTopThreads) of the cluster
// (kTopUnroll loads in flight per step), then the warp butterflies, the block's 32 warps, and rank 0 over the
// cluster's blocks (distributed shared memory). Writes the row's score (out) or its argmax (ids).
constexpr int kTopCluster = 8;
constexpr int kTopThreads = 1024;
constexpr int kTopUnroll = 4;

__global__ void __cluster_dims__(kTopCluster, 1, 1) __launch_bounds__(kTopThreads)
    top2_kernel(const float *logits, int64_t vocab, int64_t ld, cere_token_score *out, int32_t *ids) {
    pdl::enter();
    __shared__ Top2 warp_best[kTopThreads / 32];
    __shared__ Top2 block_best;
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = int(cluster.block_rank());
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float *row = logits + int64_t(blockIdx.y) * ld;
    const int64_t stride = int64_t(kTopCluster) * kTopThreads;
    Top2 t{{0.0f, 0.0f}, {-1, -1}};
    int64_t c = int64_t(rank) * kTopThreads + threadIdx.x;
    for (; c + (kTopUnroll - 1) * stride < vocab; c += kTopUnroll * stride) {
        float v[kTopUnroll];
#pragma unroll
        for (int u = 0; u < kTopUnroll; ++u) v[u] = row[c + u * stride];
#pragma unroll
        for (int u = 0; u < kTopUnroll; ++u) top2_push(t, v[u], int32_t(c + u * stride));
    }
    for (; c < vocab; c += stride) top2_push(t, row[c], int32_t(c));
    top2_warp(t);
    if (lane == 0) warp_best[warp] = t;
    __syncthreads();
    if (warp == 0) {
        t = warp_best[lane];
        top2_warp(t);
        if (lane == 0) block_best = t;
    }
    cluster.sync();
    if (rank == 0 && warp == 0) {
        t = lane < kTopCluster ? *cluster.map_shared_rank(&block_best, lane) : Top2{{0.0f, 0.0f}, {-1, -1}};
        top2_warp(t);
        if (lane == 0) {
            if (out != nullptr) {
                out[blockIdx.y] = cere_token_score{uint32_t(t.i[0]), uint32_t(t.i[1]), t.v[0], t.v[1]};
            } else {
                ids[blockIdx.y] = t.i[0];
            }
        }
    }
    cluster.sync();  // every block's block_best stays mapped until rank 0 has read it
}

// k12::process's D2 (penalties, then bias) on FP32 logits: the same fp32 operations in the same order.
__global__ void process_kernel(const float *__restrict__ logits, int64_t vocab, const uint32_t *__restrict__ counts,
                               float f, float p, float r, float *__restrict__ out) {
    const int64_t i = int64_t(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= vocab) return;
    float x = logits[i];
    if (counts != nullptr) {
        const uint32_t c = counts[i];
        if (c > 0) {
            x = __fsub_rn(x, __fadd_rn(__fmul_rn(f, __uint2float_rn(c)), p));
            x = x < 0.0f ? __fmul_rn(x, r) : __fdiv_rn(x, r);
        }
    }
    out[i] = x;
}

__global__ void bias_kernel(const cere_logit_bias *__restrict__ bias, int64_t count, float *__restrict__ out) {
    const int64_t j = int64_t(blockIdx.x) * kThreads + threadIdx.x;
    if (j >= count) return;
    out[bias[j].id] = __fadd_rn(out[bias[j].id], bias[j].bias);
}

}  // namespace

cudaError_t widen_logits(const __nv_bfloat16 *in, int64_t ldi, int64_t rows, int64_t vocab, float *out, int64_t ldo,
                         cudaStream_t stream) {
    if (in == nullptr || out == nullptr || rows < 1 || rows > 65535 || vocab < 1 || ldi < vocab || ldo < vocab) {
        return cudaErrorInvalidValue;
    }
    const int64_t blocks = (vocab + kThreads - 1) / kThreads;
    const unsigned bx = unsigned(blocks < 1024 ? blocks : 1024);
    return pdl::launch(widen_kernel, dim3(bx, unsigned(rows)), dim3(kThreads), 0, stream, in, ldi, vocab, out, ldo);
}

cudaError_t argmax_rows(const float *logits, int64_t rows, int64_t vocab, int64_t ld, int32_t *ids,
                        cudaStream_t stream) {
    if (logits == nullptr || ids == nullptr || rows < 1 || rows > 65535 || vocab < 1 ||
        vocab > (int64_t(1) << 31) - 1 || ld < vocab) {
        return cudaErrorInvalidValue;
    }
    return pdl::launch(top2_kernel, dim3(kTopCluster, unsigned(rows)), dim3(kTopThreads), 0, stream, logits, vocab, ld,
                       static_cast<cere_token_score *>(nullptr), ids);
}

cudaError_t top2_rows(const float *logits, int64_t rows, int64_t vocab, int64_t ld, cere_token_score *out,
                      cudaStream_t stream) {
    if (logits == nullptr || out == nullptr || rows < 1 || rows > 65535 || vocab < 2 ||
        vocab > (int64_t(1) << 31) - 1 || ld < vocab) {
        return cudaErrorInvalidValue;
    }
    return pdl::launch(top2_kernel, dim3(kTopCluster, unsigned(rows)), dim3(kTopThreads), 0, stream, logits, vocab, ld,
                       out, static_cast<int32_t *>(nullptr));
}

cudaError_t process_logits(const float *logits, int64_t vocab, const uint32_t *counts, float frequency_penalty,
                           float presence_penalty, float repetition_penalty, const cere_logit_bias *bias,
                           int64_t bias_count, float *out, cudaStream_t stream) {
    if (logits == nullptr || out == nullptr || vocab < 2 || vocab > (int64_t(1) << 31) - 1 || bias_count < 0 ||
        bias_count > vocab || (bias_count > 0 && bias == nullptr)) {
        return cudaErrorInvalidValue;
    }
    process_kernel<<<unsigned((vocab + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        logits, vocab, counts, frequency_penalty, presence_penalty, repetition_penalty, out);
    if (bias_count > 0) {
        bias_kernel<<<unsigned((bias_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(bias, bias_count, out);
    }
    return cudaGetLastError();
}

}  // namespace cere::inkling
