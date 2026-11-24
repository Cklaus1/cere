// K7 reference router (executor/tests/gpu/test_kernels_k7_moe.py `test_router_equals_block_router`, BeatSglang S4
// D7): K7's 128-thread block-argmax router as it stood before the sorted-key router (k7_moe.cu router_kernel)
// replaced it, kept verbatim so the new router is checked bit for bit against it (ids and weights). Built with the
// same CUDA flags as k7_moe.cu (no fast math), so expf and the divisions are the same instructions.
// `variant` 1 (fail-first only): the block sum adds its 4 warp partials in reverse order.
//
//   int cere_ktest_moe_router_block(const void *logits,       /* bfloat16 [m, num_experts] */
//                                   void *topk_ids,           /* int32 [m, top_k] */
//                                   void *topk_weights,       /* float32 [m, top_k] */
//                                   int64_t m, int64_t num_experts, int64_t top_k,
//                                   int64_t variant,          /* 0 verbatim, 1 reversed warp sum */
//                                   cudaStream_t stream);
#include "cere_executor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

namespace cere::ktest {
namespace {

constexpr uint32_t kMaxExperts = 1024;
constexpr uint32_t kMaxTopK = 32;
constexpr int kRouterThreads = 128;
constexpr int kRouterPerThread = kMaxExperts / kRouterThreads;
constexpr int kWarps = kRouterThreads / 32;

__device__ __forceinline__ float block_max(float v, float *scratch) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    __syncthreads();
    if ((threadIdx.x & 31) == 0) {
        scratch[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    float r = scratch[0];
    for (int i = 1; i < kWarps; ++i) {
        r = fmaxf(r, scratch[i]);
    }
    return r;
}

__device__ __forceinline__ float block_sum(float v, float *scratch, bool reversed) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    __syncthreads();
    if ((threadIdx.x & 31) == 0) {
        scratch[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    if (reversed) {
        float r = scratch[kWarps - 1];
        for (int i = kWarps - 2; i >= 0; --i) {
            r += scratch[i];
        }
        return r;
    }
    float r = scratch[0];
    for (int i = 1; i < kWarps; ++i) {
        r += scratch[i];
    }
    return r;
}

// (value, id) with the larger value first and the lower id on ties.
__device__ __forceinline__ bool better(float v, int id, float bv, int bid) {
    return v > bv || (v == bv && id < bid);
}

__global__ void __launch_bounds__(kRouterThreads)
    block_router_kernel(const __nv_bfloat16 *__restrict__ logits, int32_t *__restrict__ ids,
                        float *__restrict__ weights, int num_experts, int top_k, int variant) {
    __shared__ float scratch[kWarps];
    __shared__ float arg_v[kWarps];
    __shared__ int arg_id[kWarps];
    __shared__ float sel_p[kMaxTopK];
    __shared__ int sel_id[kMaxTopK];
    const int64_t t = blockIdx.x;
    const __nv_bfloat16 *row = logits + t * num_experts;
    float v[kRouterPerThread];
    float p[kRouterPerThread];
    float local_max = -INFINITY;
#pragma unroll
    for (int i = 0; i < kRouterPerThread; ++i) {
        const int e = threadIdx.x + i * kRouterThreads;
        v[i] = e < num_experts ? __bfloat162float(row[e]) : -INFINITY;
        local_max = fmaxf(local_max, v[i]);
    }
    const float row_max = block_max(local_max, scratch);
    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kRouterPerThread; ++i) {
        const int e = threadIdx.x + i * kRouterThreads;
        p[i] = e < num_experts ? expf(v[i] - row_max) : 0.0f;
        local_sum += p[i];
    }
    const float row_sum = block_sum(local_sum, scratch, variant == 1);
#pragma unroll
    for (int i = 0; i < kRouterPerThread; ++i) {
        p[i] = p[i] / row_sum;
        v[i] = v[i] == v[i] ? v[i] : -INFINITY;  // NaN ranks last
    }
    uint32_t taken = 0;  // bit i: this thread's i-th logit is already selected
    for (int k = 0; k < top_k; ++k) {
        float bv = -INFINITY;
        int bid = INT_MAX;
#pragma unroll
        for (int i = 0; i < kRouterPerThread; ++i) {
            const int e = threadIdx.x + i * kRouterThreads;
            if (e < num_experts && !((taken >> i) & 1u) && better(v[i], e, bv, bid)) {
                bv = v[i];
                bid = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oid = __shfl_xor_sync(0xffffffffu, bid, o);
            if (better(ov, oid, bv, bid)) {
                bv = ov;
                bid = oid;
            }
        }
        __syncthreads();
        if ((threadIdx.x & 31) == 0) {
            arg_v[threadIdx.x >> 5] = bv;
            arg_id[threadIdx.x >> 5] = bid;
        }
        __syncthreads();
        bv = arg_v[0];
        bid = arg_id[0];
        for (int w = 1; w < kWarps; ++w) {
            if (better(arg_v[w], arg_id[w], bv, bid)) {
                bv = arg_v[w];
                bid = arg_id[w];
            }
        }
        // The owner records the winner's probability and removes it from the candidates.
#pragma unroll
        for (int i = 0; i < kRouterPerThread; ++i) {
            if (threadIdx.x + i * kRouterThreads == bid) {
                sel_p[k] = p[i];
                sel_id[k] = bid;
                taken |= 1u << i;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            sum += sel_p[k];
        }
        const bool finite_row = row_sum > 0.0f && row_sum < INFINITY;
        const float norm = sum > 0.0f ? sum : 1.0f;
        for (int k = 0; k < top_k; ++k) {
            ids[t * top_k + k] = sel_id[k];
            weights[t * top_k + k] = finite_row ? sel_p[k] / norm : 0.0f;
        }
    }
}

}  // namespace
}  // namespace cere::ktest

extern "C" __attribute__((visibility("default"))) int cere_ktest_moe_router_block(const void *logits, void *topk_ids,
                                                                                  void *topk_weights, int64_t m,
                                                                                  int64_t num_experts, int64_t top_k,
                                                                                  int64_t variant,
                                                                                  cudaStream_t stream) {
    if (logits == nullptr || topk_ids == nullptr || topk_weights == nullptr || m <= 0 || m > INT32_MAX ||
        num_experts <= 0 || num_experts > cere::ktest::kMaxExperts || top_k <= 0 ||
        top_k > cere::ktest::kMaxTopK || top_k > num_experts || variant < 0 || variant > 1) {
        return CERE_ERR_ARGUMENT;
    }
    cere::ktest::block_router_kernel<<<static_cast<unsigned>(m), cere::ktest::kRouterThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(logits), static_cast<int32_t *>(topk_ids),
        static_cast<float *>(topk_weights), static_cast<int>(num_experts), static_cast<int>(top_k),
        static_cast<int>(variant));
    return cudaGetLastError() == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}
