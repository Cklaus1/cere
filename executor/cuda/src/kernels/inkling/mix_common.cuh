// Inkling 2 Mini (inkling design note G1): device helpers shared by the residual-stream kernels (norm, depth, msg):
// one 2304-wide row per 256-thread block, thread t holding elements i = t + 256 j (j < 9), so a lane-strided
// accumulation per thread followed by block_tree256 is ref::reduce_*<256> (common_ref.h item 3) bit for bit.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inkling/norm_ref.h"

namespace cere::inkling::mix_detail {

constexpr int kRowThreads = ref::kReduceHidden;              // 256
constexpr int kPerThread = ref::kHidden / ref::kReduceHidden;  // 9
static_assert(kPerThread * kRowThreads == ref::kHidden, "2304 = 9 x 256");

// ref::reduce_tree_<256> over red[0..255] in one warp: lane i holds red[i + 32 m] (m < 8) and adds the tree's pairs
// in registers (level 128: m with m + 4, level 64: m with m + 2, level 32: 0 with 1), then levels 16..1 by xor
// shuffles (lane i < off adds lane i + off: the tree's pairs, so lane 0 ends with the tree's result and, every add
// commuting, so does every lane). The same adds as the tree: bit-equal.
__device__ inline float warp_tree256(const float *red, int lane) {
    float a[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) a[m] = red[lane + 32 * m];
#pragma unroll
    for (int m = 0; m < 4; ++m) a[m] = ref::add(a[m], a[m + 4]);
#pragma unroll
    for (int m = 0; m < 2; ++m) a[m] = ref::add(a[m], a[m + 2]);
    float x = ref::add(a[0], a[1]);
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) x = ref::add(x, __shfl_xor_sync(0xffffffffu, x, off));
    return x;
}

// One tree over one value per thread (red: shared [256]); every warp runs it (warp_tree256), so every thread gets
// the result. Callable by all 256 threads only (it synchronizes the block); red may be reused at once.
__device__ inline float block_tree256(float v, float *red) {
    red[threadIdx.x] = v;
    __syncthreads();
    const float r = warp_tree256(red, int(threadIdx.x) % 32);
    __syncthreads();  // red is free for the next call
    return r;
}

// kB trees at once (v[k] one value per thread; red: shared [kB][256]); warp k % 8 runs tree k (warp_tree256) and the
// results land in thread 0's out[k] (other threads' out is left as is). Callable by all 256 threads only (it
// synchronizes the block); red may be reused at once.
template <int kB>
__device__ inline void block_tree256_batch(const float (&v)[kB], float (*red)[kRowThreads], float (&out)[kB]) {
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
#pragma unroll
    for (int k = 0; k < kB; ++k) red[k][t] = v[k];
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kB; ++k) {
        if (k % (kRowThreads / 32) == warp) {
            const float x = warp_tree256(red[k], lane);
            if (lane == 0) red[k][0] = x;  // tree k's values are read by this warp only, all before the shuffles
        }
    }
    __syncthreads();
    if (t == 0)
#pragma unroll
        for (int k = 0; k < kB; ++k) out[k] = red[k][0];
    __syncthreads();  // red is free for the next call
}

// The trees k with bit k of `mask` set, of ref::reduce_tree_<256> at once (v[k] one value per thread; red: shared
// [kMax][256], bc: shared [kMax]); every thread gets tree k's result in out[k] (out[k] of the other k left as is).
// Warp k % 8 runs tree k (warp_tree256). Callable by all 256 threads only (it synchronizes the block); red and bc
// may be reused by the next call at once.
template <int kMax>
__device__ inline void block_tree256_all(const float (&v)[kMax], uint32_t mask, float (*red)[kRowThreads], float *bc,
                                         float (&out)[kMax]) {
    const int t = threadIdx.x;
#pragma unroll
    for (int k = 0; k < kMax; ++k)
        if ((mask >> k) & 1u) red[k][t] = v[k];
    __syncthreads();
    const int warp = t / 32, lane = t % 32;
#pragma unroll
    for (int k = 0; k < kMax; ++k) {
        if (k % (kRowThreads / 32) == warp && ((mask >> k) & 1u)) {
            const float x = warp_tree256(red[k], lane);
            if (lane == 0) bc[k] = x;
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kMax; ++k)
        if ((mask >> k) & 1u) out[k] = bc[k];
}

// rms_inv of norm_ref.h over the row held as v[j] = x[t + 256 j].
__device__ inline float row_rms_inv(const float (&v)[kPerThread], float eps, float *red) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) acc = ref::fma(v[j], v[j], acc);
    const float ss = block_tree256(acc, red);
    return ref::rsqrt(ref::add(ref::div(ss, float(ref::kHidden)), eps));
}

// Element `idx` of a BF16 (f32 = false) or FP32 (f32 = true) buffer, widened exactly.
__device__ inline float load_any(const void *p, bool f32, int64_t idx) {
    return f32 ? static_cast<const float *>(p)[idx] : ref::decode_bf16(static_cast<const uint16_t *>(p)[idx]);
}

}  // namespace cere::inkling::mix_detail
