// K3: hyper-connection mix/combine kernels. Contract and math in k3_hc.h.
// math_reference: sglang
#include "kernels/k3_hc.h"

#include <cooperative_groups.h>

#include "kernels/k1_primitives.h"
#include "kernels/k2_small_m.h"
#include "kernels/k3_hc_math.cuh"

namespace cere::kernels {
namespace {

namespace cg = cooperative_groups;

constexpr uint32_t kHc = hc::kHc;  // the compiled stream count (config hc_count)
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr size_t kAlign = 256;

size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

// Fixed-order block sum (shuffle tree per warp, then warp 0 over the warp partials); every thread gets the result.
__device__ float block_sum(float v, float *scratch) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (lane == 0) {
        scratch[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        float w = lane < kWarps ? scratch[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            w += __shfl_xor_sync(0xffffffffu, w, offset);
        }
        if (lane == 0) {
            scratch[kWarps] = w;
        }
    }
    __syncthreads();
    const float total = scratch[kWarps];
    __syncthreads();  // scratch is reused by the next call
    return total;
}

// t = bf16(silu(t_raw / hc)) over [m, lowrank] (hc::act).
__global__ void hc_act_kernel(const float *__restrict__ t_raw, __nv_bfloat16 *__restrict__ t, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        t[i] = hc::act(t_raw[i]);
    }
}

// mixed[r, j] = bf16(sum_g sigmoid(u[r, g, j]) * normed[r, g, j] / hc) (hc::gate).
__global__ void hc_gate_kernel(const float *__restrict__ u, const __nv_bfloat16 *__restrict__ normed,
                               __nv_bfloat16 *__restrict__ mixed, uint32_t m, uint32_t hidden) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(m) * hidden) {
        return;
    }
    const size_t r = i / hidden;
    const size_t j = i % hidden;
    const size_t row = r * kHc * hidden + j;
    float uv[kHc];
    __nv_bfloat16 nv[kHc];
#pragma unroll
    for (uint32_t g = 0; g < kHc; ++g) {
        uv[g] = u[row + g * hidden];
        nv[g] = normed[row + g * hidden];
    }
    mixed[i] = hc::gate(uv, nv);
}

// Two launch shapes of the same arithmetic. Both compute a[g] = 2 * sigmoid(normed[r] · w_inject[g] / hc) with the
// same per-thread element order and block sum, then out = hyper + a[g] * block_out, so their output is bit-identical.
// m > kCombineSplitMaxM (prefill): one block per row computes all kHc gates in one pass over normed[r].
// m <= kCombineSplitMaxM (decode, verify; m23 design note P11): one block per row was latency-bound (m = 1: one
// block read 100 KB and rewrote 20 KB), so block (r, g, part) computes a[g] alone and writes its `part` of stream g.
// The split re-reads normed[r] kHc * kCombineParts times, which cost +126 ms of K3 on the 18K prefill (D12), hence
// the row kernel above that bound.
constexpr uint32_t kCombineParts = 4;
constexpr uint32_t kCombineSplitMaxM = 16;

struct alignas(16) Bf16x8 {
    __nv_bfloat162 h[4];
};

// block_sum of kHc values with two barriers instead of 3 * kHc: value g takes block_sum's exact tree (the per-warp
// shuffle tree, then warp 0's shuffle tree over the warp partials, zero past kWarps); the values never mix.
__device__ void block_sum_hc(float (&v)[kHc], float (*scratch)[kWarps + 1]) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (uint32_t g = 0; g < kHc; ++g) {
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            v[g] += __shfl_xor_sync(0xffffffffu, v[g], offset);
        }
        if (lane == 0) {
            scratch[g][warp] = v[g];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (uint32_t g = 0; g < kHc; ++g) {
            float w = lane < kWarps ? scratch[g][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                w += __shfl_xor_sync(0xffffffffu, w, offset);
            }
            if (lane == 0) {
                scratch[g][kWarps] = w;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (uint32_t g = 0; g < kHc; ++g) {
        v[g] = scratch[g][kWarps];
    }
}

// The row kernel's movement (prefill; the arithmetic is unchanged). It ran at 63 % of DRAM peak: its combine loop made
// one round trip per element (40 per thread at hidden 2560; a `hyper` load cannot pass the previous `out` store, which
// may alias it) after a 64-bit divide. Now the gate sum loads kCombineBatch of its elements (normed and the kHc
// W_inject values) before it accumulates them in the same order, and the combine moves 16-byte chunks, a batch's loads
// before its stores, when block_out, hyper and out are 16-byte aligned: out[e] is still
// bf16(hyper[e] + a[g] * block_out[j]) on its own, so the chunking cannot change it.
constexpr uint32_t kCombineBatch = 4;

__global__ void __launch_bounds__(kThreads) hc_combine_row_kernel(const __nv_bfloat16 *__restrict__ block_out,
                                                                  const __nv_bfloat16 *hyper,
                                                                  const __nv_bfloat16 *__restrict__ normed,
                                                                  const __nv_bfloat16 *__restrict__ w_inject,
                                                                  __nv_bfloat16 *out, uint32_t hidden, bool vec) {
    __shared__ float scratch[kHc][kWarps + 1];
    const uint32_t width = kHc * hidden;
    const size_t base = static_cast<size_t>(blockIdx.x) * width;
    const __nv_bfloat16 *n = normed + base;
    float acc[kHc];
#pragma unroll
    for (uint32_t g = 0; g < kHc; ++g) {
        acc[g] = 0.0f;
    }
    // Thread tid still adds elements tid, tid + kThreads, ... in order (the split kernel's combine_gate order).
    for (uint32_t e0 = threadIdx.x; e0 < width; e0 += kCombineBatch * kThreads) {
        __nv_bfloat16 nv[kCombineBatch];
        __nv_bfloat16 wv[kCombineBatch][kHc];
#pragma unroll
        for (uint32_t b = 0; b < kCombineBatch; ++b) {
            const uint32_t e = e0 + b * kThreads;
            if (e < width) {
                nv[b] = n[e];
#pragma unroll
                for (uint32_t g = 0; g < kHc; ++g) {
                    wv[b][g] = w_inject[static_cast<size_t>(g) * width + e];
                }
            }
        }
#pragma unroll
        for (uint32_t b = 0; b < kCombineBatch; ++b) {
            if (e0 + b * kThreads < width) {
                const float v = __bfloat162float(nv[b]);
#pragma unroll
                for (uint32_t g = 0; g < kHc; ++g) {
                    acc[g] += v * __bfloat162float(wv[b][g]);
                }
            }
        }
    }
    block_sum_hc(acc, scratch);
    float a[kHc];
#pragma unroll
    for (uint32_t g = 0; g < kHc; ++g) {
        a[g] = 2.0f / (1.0f + expf(-acc[g] / static_cast<float>(kHc)));
    }
    const __nv_bfloat16 *y = block_out + static_cast<size_t>(blockIdx.x) * hidden;
    if (vec) {
        const uint32_t per = hidden / 8;  // 16-byte chunks per stream (hidden % 8 == 0)
        for (uint32_t c0 = threadIdx.x; c0 < kHc * per; c0 += kCombineBatch * kThreads) {
            Bf16x8 hv[kCombineBatch], yv[kCombineBatch];
#pragma unroll
            for (uint32_t b = 0; b < kCombineBatch; ++b) {
                const uint32_t c = c0 + b * kThreads;
                if (c < kHc * per) {
                    const uint32_t g = c / per;
                    const uint32_t j = (c - g * per) * 8;
                    hv[b] = *reinterpret_cast<const Bf16x8 *>(hyper + base + static_cast<size_t>(g) * hidden + j);
                    yv[b] = *reinterpret_cast<const Bf16x8 *>(y + j);
                }
            }
#pragma unroll
            for (uint32_t b = 0; b < kCombineBatch; ++b) {
                const uint32_t c = c0 + b * kThreads;
                if (c < kHc * per) {
                    const uint32_t g = c / per;
                    const uint32_t j = (c - g * per) * 8;
                    float gate = a[0];
#pragma unroll
                    for (uint32_t s = 1; s < kHc; ++s) {
                        gate = g == s ? a[s] : gate;
                    }
                    Bf16x8 o;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float2 h = __bfloat1622float2(hv[b].h[i]);
                        const float2 x = __bfloat1622float2(yv[b].h[i]);
                        o.h[i] = __halves2bfloat162(__float2bfloat16_rn(h.x + gate * x.x),
                                                    __float2bfloat16_rn(h.y + gate * x.y));
                    }
                    *reinterpret_cast<Bf16x8 *>(out + base + static_cast<size_t>(g) * hidden + j) = o;
                }
            }
        }
        return;
    }
    for (size_t e = threadIdx.x; e < width; e += kThreads) {
        const uint32_t g = static_cast<uint32_t>(e / hidden);
        const size_t j = e % hidden;
        float gate = a[0];
#pragma unroll
        for (uint32_t s = 1; s < kHc; ++s) {
            gate = g == s ? a[s] : gate;
        }
        out[base + e] = __float2bfloat16_rn(__bfloat162float(hyper[base + e]) + gate * __bfloat162float(y[j]));
    }
}

// The gate a[g] = 2 * sigmoid(normed[r] · w_inject[g] / hc) of hc_combine_kernel's block: thread tid sums elements tid,
// tid + kThreads, ... in order, then block_sum. Every thread gets it.
__device__ float combine_gate(const __nv_bfloat16 *__restrict__ n, const __nv_bfloat16 *__restrict__ wg, size_t width,
                              float *scratch) {
    float acc = 0.0f;
#pragma unroll 8
    for (size_t e = threadIdx.x; e < width; e += kThreads) {
        const float v = __bfloat162float(n[e]);
        acc += v * __bfloat162float(wg[e]);
    }
    const float total = block_sum(acc, scratch);
    return 2.0f / (1.0f + expf(-total / static_cast<float>(kHc)));
}

__global__ void __launch_bounds__(kThreads) hc_combine_kernel(const __nv_bfloat16 *__restrict__ block_out,
                                                              const __nv_bfloat16 *hyper,
                                                              const __nv_bfloat16 *__restrict__ normed,
                                                              const __nv_bfloat16 *__restrict__ w_inject,
                                                              __nv_bfloat16 *out, uint32_t hidden) {
    __shared__ float scratch[kWarps + 1];
    const uint32_t g = blockIdx.y;
    const size_t width = static_cast<size_t>(kHc) * hidden;
    const size_t base = static_cast<size_t>(blockIdx.x) * width;
    const float gate = combine_gate(normed + base, w_inject + g * width, width, scratch);
    const __nv_bfloat16 *y = block_out + static_cast<size_t>(blockIdx.x) * hidden;
    const size_t stream = base + static_cast<size_t>(g) * hidden;
    const uint32_t part = (hidden + kCombineParts - 1) / kCombineParts;
    const uint32_t end = min(hidden, (blockIdx.z + 1) * part);
    for (uint32_t j = blockIdx.z * part + threadIdx.x; j < end; j += kThreads) {
        out[stream + j] = __float2bfloat16_rn(__bfloat162float(hyper[stream + j]) + gate * __bfloat162float(y[j]));
    }
}

// hc_combine_norm (m23 design note P5R): hc_combine_kernel's combine, then the next mix's K1 per-stream Gemma
// RMSNorm of the combined stream, reproducing k1::rmsnorm's block (kThreads = norm_threads(hidden) threads, thread
// tid's 8-element chunks tid, tid + kThreads in order, K1's block sum, v * rstd * (1 + w)), so `normed` equals what
// hc_mix's k1::rmsnorm writes, bit for bit. Block (r, g) of a cluster of the kHc blocks of row r: every block reads all
// of normed[r] for its gate, so the cluster barrier comes before any block overwrites its stream of normed[r].
// [REVISED: SmallFuse: one round trip per operand instead of the gate loop's dependent strided loads: the block
// stages W_inject[g] and normed[r] in shared memory with 16-byte loads and its stream's hyper, block_out and norm
// weight chunks in registers, then runs the same per-thread sums from shared memory; the weights (never written by a
// kernel of a step) load before the programmatic dependency wait and the cluster barrier is split (arrive once
// normed[r] is staged, wait before the normed stores)]
constexpr int kNormVec = 8;     // BF16 per 16-byte chunk (K1 kVec)
constexpr int kNormChunks = 2;  // chunks per thread: hidden <= kNormChunks * kNormVec * kThreads

__device__ __forceinline__ void unpack8(const Bf16x8 &v, float (&out)[kNormVec]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(v.h[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

__device__ __forceinline__ void store8(__nv_bfloat16 *p, const float (&in)[kNormVec]) {
    Bf16x8 v;
#pragma unroll
    for (int i = 0; i < 4; ++i) v.h[i] = __floats2bfloat162_rn(in[2 * i], in[2 * i + 1]);
    *reinterpret_cast<Bf16x8 *>(p) = v;
}

// k1_primitives.cu block_sum's order: shuffle tree per warp, then every thread adds the warp partials in order.
__device__ float k1_block_sum(float v, float *partial) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < kWarps; ++w) total += partial[w];
    return total;
}

// Programmatic dependent launch (as k2_small_m's dependency_wait): the next kernel of the stream may launch (it waits
// for this grid in its own dependency wait); then wait for the previous kernel's grid and its writes.
__device__ __forceinline__ void launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void dependency_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// Shared memory of hc_combine_norm_kernel's staged operands: W_inject[g] then normed[r], kHc * hidden BF16 each.
size_t combine_norm_stage_bytes(uint32_t hidden) { return size_t{2} * kHc * hidden * sizeof(__nv_bfloat16); }
constexpr size_t kCombineNormStageMax = 46 * 1024;  // dynamic shared memory without the opt-in attribute (48 KiB)
// 16-byte words of one staged operand per thread (kCombineNormStageMax / 2 bytes over kThreads): all of a thread's
// loads are issued before any of its shared-memory stores, so the copy costs one memory latency, not one per word.
constexpr int kStageWords = static_cast<int>((kCombineNormStageMax / 2 / sizeof(uint4) + kThreads - 1) / kThreads);

__device__ __forceinline__ void load_stage(const uint4 *src, uint32_t vecs, uint4 (&t)[kStageWords]) {
#pragma unroll
    for (int k = 0; k < kStageWords; ++k) {
        const uint32_t i = threadIdx.x + static_cast<uint32_t>(k) * kThreads;
        if (i < vecs) t[k] = src[i];
    }
}

__device__ __forceinline__ void store_stage(uint4 *dst, uint32_t vecs, const uint4 (&t)[kStageWords]) {
#pragma unroll
    for (int k = 0; k < kStageWords; ++k) {
        const uint32_t i = threadIdx.x + static_cast<uint32_t>(k) * kThreads;
        if (i < vecs) dst[i] = t[k];
    }
}

__global__ void __cluster_dims__(1, kHc, 1) __launch_bounds__(kThreads)
    hc_combine_norm_kernel(const __nv_bfloat16 *__restrict__ block_out, const __nv_bfloat16 *hyper,
                           __nv_bfloat16 *normed, const __nv_bfloat16 *__restrict__ w_inject,
                           const __nv_bfloat16 *__restrict__ next_norm, float eps, __nv_bfloat16 *out,
                           uint32_t hidden) {
    extern __shared__ uint4 stage[];  // [vecs] W_inject[g], then [vecs] normed[r]
    __shared__ float scratch[kWarps + 1];
    __shared__ float partial[kWarps];
    const uint32_t g = blockIdx.y;
    const uint32_t width = kHc * hidden;
    const uint32_t vecs = width / kNormVec;
    const size_t base = static_cast<size_t>(blockIdx.x) * width;
    const size_t stream = base + static_cast<size_t>(g) * hidden;
    const uint4 *wg = reinterpret_cast<const uint4 *>(w_inject + static_cast<size_t>(g) * width);
    uint4 t[kStageWords];
    load_stage(wg, vecs, t);
    Bf16x8 wn[kNormChunks], hv[kNormChunks], yv[kNormChunks];
    const __nv_bfloat16 *wr = next_norm + static_cast<size_t>(g) * hidden;
#pragma unroll
    for (int k = 0; k < kNormChunks; ++k) {
        const uint32_t c = (threadIdx.x + k * kThreads) * kNormVec;
        if (c < hidden) wn[k] = *reinterpret_cast<const Bf16x8 *>(wr + c);
    }
    store_stage(stage, vecs, t);
    launch_dependents();
    dependency_wait();
    load_stage(reinterpret_cast<const uint4 *>(normed + base), vecs, t);
    const __nv_bfloat16 *y = block_out + static_cast<size_t>(blockIdx.x) * hidden;
#pragma unroll
    for (int k = 0; k < kNormChunks; ++k) {
        const uint32_t c = (threadIdx.x + k * kThreads) * kNormVec;
        if (c < hidden) {
            hv[k] = *reinterpret_cast<const Bf16x8 *>(hyper + stream + c);
            yv[k] = *reinterpret_cast<const Bf16x8 *>(y + c);
        }
    }
    store_stage(stage + vecs, vecs, t);
    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");  // this block has read normed[r]
    // combine_gate's sum (thread tid: elements tid, tid + kThreads, ... in order, then block_sum) over the stage.
    const __nv_bfloat16 *ws = reinterpret_cast<const __nv_bfloat16 *>(stage);
    const __nv_bfloat16 *n = ws + width;
    float acc = 0.0f;
#pragma unroll 8
    for (uint32_t e = threadIdx.x; e < width; e += kThreads) {
        const float v = __bfloat162float(n[e]);
        acc += v * __bfloat162float(ws[e]);
    }
    const float total = block_sum(acc, scratch);
    const float gate = 2.0f / (1.0f + expf(-total / static_cast<float>(kHc)));
    float v[kNormChunks][kNormVec];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < kNormChunks; ++k) {
        const uint32_t c = (threadIdx.x + k * kThreads) * kNormVec;
        if (c < hidden) {
            Bf16x8 o;
#pragma unroll
            for (int i = 0; i < kNormVec; i += 2) {
                const float2 h = __bfloat1622float2(hv[k].h[i / 2]);
                const float2 b = __bfloat1622float2(yv[k].h[i / 2]);
                const __nv_bfloat16 lo = __float2bfloat16_rn(h.x + gate * b.x);
                const __nv_bfloat16 hi = __float2bfloat16_rn(h.y + gate * b.y);
                o.h[i / 2] = __halves2bfloat162(lo, hi);
                v[k][i] = __bfloat162float(lo);
                v[k][i + 1] = __bfloat162float(hi);
            }
            *reinterpret_cast<Bf16x8 *>(out + stream + c) = o;
#pragma unroll
            for (int i = 0; i < kNormVec; ++i) ss += v[k][i] * v[k][i];
        }
    }
    const float rstd = rsqrtf(k1_block_sum(ss, partial) / static_cast<float>(hidden) + eps);
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");  // the row's kHc blocks have read normed[r]
#pragma unroll
    for (int k = 0; k < kNormChunks; ++k) {
        const uint32_t c = (threadIdx.x + k * kThreads) * kNormVec;
        if (c < hidden) {
            float w[kNormVec];
            unpack8(wn[k], w);
#pragma unroll
            for (int i = 0; i < kNormVec; ++i) v[k][i] = v[k][i] * rstd * (1.0f + w[i]);
            store8(normed + stream + c, v[k]);
        }
    }
}

unsigned int blocks_for(size_t count) { return static_cast<unsigned int>((count + kThreads - 1) / kThreads); }

bool streams_ok(const HcShape &s) { return s.hc == kHc && s.hidden > 0 && s.hidden % 8 == 0; }
bool shape_ok(const HcShape &s) { return streams_ok(s) && s.lowrank > 0; }
bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// The per-stream Gemma RMSNorm of hc_mix: rows = m * hc streams of `hidden`, stream g uses weight row g (K1 grouped
// rmsnorm).
cere_status stream_norm(const HcShape &s, const __nv_bfloat16 *hyper, const __nv_bfloat16 *norm_weight, uint32_t m,
                        __nv_bfloat16 *normed, cudaStream_t stream) {
    const cudaError_t norm = k1::rmsnorm(hyper, s.hidden, norm_weight, static_cast<int>(kHc), true, s.eps,
                                         static_cast<int64_t>(m) * kHc, static_cast<int>(s.hidden), normed, s.hidden,
                                         stream);
    if (norm != cudaSuccess) {
        return norm == cudaErrorInvalidValue ? CERE_ERR_ARGUMENT : CERE_ERR_INTERNAL;
    }
    return CERE_OK;
}

}  // namespace

cere_status HcPlans::init(cublasLtHandle_t handle, const HcShape &s, uint32_t max_m) {
    if (!shape_ok(s)) {
        return CERE_ERR_UNSUPPORTED;
    }
    shape = s;
    const uint32_t width = s.hc * s.hidden;
    // gemm_pins_for's pins (the bucket re-pins, k2_gemm.h): split-K-free, so they take no workspace.
    cere_status st = down.init(handle, s.lowrank, width, max_m, GemmOut::kF32,
                               gemm_pins_for(s.lowrank, width, GemmOut::kF32));
    if (st == CERE_OK) {
        st = up.init(handle, width, s.lowrank, max_m, GemmOut::kF32, gemm_pins_for(width, s.lowrank, GemmOut::kF32));
    }
    return st;
}

cere_status HcPlans::init_rows(cublasLtHandle_t handle, const HcPlans &instantiated, uint32_t rows,
                               uint32_t fused_allowed) {
    if (!shape_ok(instantiated.shape)) {
        return CERE_ERR_ARGUMENT;
    }
    shape = instantiated.shape;
    cere_status st = down.init_rows(handle, instantiated.down, rows, fused_allowed);
    if (st == CERE_OK) {
        st = up.init_rows(handle, instantiated.up, rows, fused_allowed);
    }
    return st;
}

size_t hc_mix_workspace_bytes(const HcShape &s, uint32_t max_m) {
    const size_t rows = max_m;
    return align_up(rows * s.lowrank * sizeof(float)) + align_up(rows * s.lowrank * sizeof(__nv_bfloat16)) +
           align_up(rows * s.hc * s.hidden * sizeof(float));
}

bool hc_mix_fuses(const HcPlans &plans, uint32_t m) {
    return m >= 1 && m <= k2s::kMaxM && plans.down.small_m() && plans.up.small_m() && !plans.down.per_row(m) &&
           !plans.up.per_row(m) && k2s::hc_gate_supported(plans.shape.lowrank);
}

cere_status hc_mix(const HcPlans &plans, const __nv_bfloat16 *hyper, const __nv_bfloat16 *norm_weight,
                   const __nv_bfloat16 *w_down, const __nv_bfloat16 *w_up, uint32_t m, __nv_bfloat16 *normed,
                   __nv_bfloat16 *mixed, void *workspace, cudaStream_t stream, HcMixPath path, uint32_t cls) {
    if (hyper == nullptr || norm_weight == nullptr || !shape_ok(plans.shape) || m == 0 || m > plans.down.max_m() ||
        normed == nullptr || mixed == nullptr || workspace == nullptr ||
        reinterpret_cast<uintptr_t>(workspace) % kAlign != 0) {
        return CERE_ERR_ARGUMENT;
    }
    const cere_status st = stream_norm(plans.shape, hyper, norm_weight, m, normed, stream);
    if (st != CERE_OK) {
        return st;
    }
    return hc_mix_normed(plans, normed, w_down, w_up, m, mixed, workspace, stream, path, cls);
}

cere_status hc_mix_normed(const HcPlans &plans, const __nv_bfloat16 *normed, const __nv_bfloat16 *w_down,
                          const __nv_bfloat16 *w_up, uint32_t m, __nv_bfloat16 *mixed, void *workspace,
                          cudaStream_t stream, HcMixPath path, uint32_t cls) {
    const HcShape &s = plans.shape;
    const uint32_t max_m = plans.down.max_m();
    if (!shape_ok(s) || m == 0 || m > max_m || normed == nullptr || mixed == nullptr || workspace == nullptr ||
        reinterpret_cast<uintptr_t>(workspace) % kAlign != 0) {
        return CERE_ERR_ARGUMENT;
    }
    char *ws = static_cast<char *>(workspace);
    float *t_raw = reinterpret_cast<float *>(ws);
    __nv_bfloat16 *t = reinterpret_cast<__nv_bfloat16 *>(ws + align_up(size_t{max_m} * s.lowrank * sizeof(float)));
    float *u = reinterpret_cast<float *>(ws + align_up(size_t{max_m} * s.lowrank * sizeof(float)) +
                                         align_up(size_t{max_m} * s.lowrank * sizeof(__nv_bfloat16)));

    const HcMixPath chosen = path == HcMixPath::kAuto ? kHcMixDefault : path;
    if (chosen != HcMixPath::kUnfused && hc_mix_fuses(plans, cls != 0 ? cls : m)) {
        if (w_down == nullptr || w_up == nullptr) {
            return CERE_ERR_ARGUMENT;
        }
        const bool early = chosen == HcMixPath::kEarly;
        cudaError_t e = k2s::gemm_hc_act(normed, m, plans.down.k(), w_down, plans.down.n(), t, stream, early);
        if (e == cudaSuccess) {
            e = k2s::gemm_hc_gate(t, m, plans.up.k(), w_up, s.hidden, normed, mixed, stream, early);
        }
        return e == cudaSuccess ? CERE_OK : (e == cudaErrorInvalidValue ? CERE_ERR_ARGUMENT : CERE_ERR_INTERNAL);
    }
    cere_status st = plans.down.run(normed, w_down, m, t_raw, stream, cls);
    if (st != CERE_OK) {
        return st;
    }
    const size_t t_count = size_t{m} * s.lowrank;
    hc_act_kernel<<<blocks_for(t_count), kThreads, 0, stream>>>(t_raw, t, t_count);
    if (cudaGetLastError() != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    st = plans.up.run(t, w_up, m, u, stream, cls);
    if (st != CERE_OK) {
        return st;
    }
    hc_gate_kernel<<<blocks_for(size_t{m} * s.hidden), kThreads, 0, stream>>>(u, normed, mixed, m, s.hidden);
    return cudaGetLastError() == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

cere_status hc_combine(const HcShape &s, const __nv_bfloat16 *block_out, const __nv_bfloat16 *hyper,
                       const __nv_bfloat16 *normed, const __nv_bfloat16 *w_inject, uint32_t m, __nv_bfloat16 *out,
                       cudaStream_t stream) {
    if (!streams_ok(s)) {
        return CERE_ERR_UNSUPPORTED;
    }
    if (m == 0 || block_out == nullptr || hyper == nullptr || normed == nullptr || w_inject == nullptr ||
        out == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    if (m <= kCombineSplitMaxM) {
        hc_combine_kernel<<<dim3(m, kHc, kCombineParts), kThreads, 0, stream>>>(block_out, hyper, normed, w_inject,
                                                                                  out, s.hidden);
    } else {
        const bool vec = aligned16(block_out) && aligned16(hyper) && aligned16(out);
        hc_combine_row_kernel<<<m, kThreads, 0, stream>>>(block_out, hyper, normed, w_inject, out, s.hidden, vec);
    }
    return cudaGetLastError() == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

cere_status hc_combine_norm(const HcShape &s, const __nv_bfloat16 *block_out, const __nv_bfloat16 *hyper,
                            __nv_bfloat16 *normed, const __nv_bfloat16 *w_inject,
                            const __nv_bfloat16 *next_norm_weight, uint32_t m, __nv_bfloat16 *out,
                            cudaStream_t stream) {
    if (!streams_ok(s)) {
        return CERE_ERR_UNSUPPORTED;
    }
    if (m == 0 || block_out == nullptr || hyper == nullptr || normed == nullptr || w_inject == nullptr ||
        next_norm_weight == nullptr || out == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    // The fused block is k1::rmsnorm's only when that runs kThreads threads (norm_threads(hidden): more than 224
    // chunks) of at most kNormChunks chunks each, on 16-byte aligned rows; its stage fits the default shared memory.
    const uint32_t chunks = s.hidden / kNormVec;
    const size_t stage = combine_norm_stage_bytes(s.hidden);
    const bool fused = m <= kCombineSplitMaxM && chunks > static_cast<uint32_t>(kThreads - 32) &&
                       chunks <= static_cast<uint32_t>(kNormChunks * kThreads) && stage <= kCombineNormStageMax &&
                       aligned16(block_out) && aligned16(hyper) && aligned16(normed) && aligned16(w_inject) &&
                       aligned16(next_norm_weight) && aligned16(out);
    if (!fused) {
        const cere_status st = hc_combine(s, block_out, hyper, normed, w_inject, m, out, stream);
        return st == CERE_OK ? stream_norm(s, out, next_norm_weight, m, normed, stream) : st;
    }
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(m, kHc);
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = stage;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;
    const cudaError_t e = cudaLaunchKernelEx(&config, hc_combine_norm_kernel, block_out, hyper, normed, w_inject,
                                             next_norm_weight, s.eps, out, s.hidden);
    return e == cudaSuccess && cudaGetLastError() == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

}  // namespace cere::kernels
