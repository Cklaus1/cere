// K7 (m22 design note K7, Q14): the NVFP4 MoE block. Contract in k7_moe.h.
//
// math_reference: sglang
// SGLang d4689328 python/sglang/srt/models/qwen2_moe.py (`Qwen2MoeSparseMoeBlock.forward`: routed experts, shared
// expert, `fused_gate_sigmoid_mul_add`), python/sglang/kernels/ops/moe/moe_fused_gate.py (softmax router: FP32 softmax,
// top-k by logit with the lowest id on ties, renormalize), python/sglang/kernels/ops/elementwise/elementwise.py
// (`_fused_gate_sigmoid_mul_add_kernel`), python/sglang/srt/layers/quantization/modelopt_quant.py (`g1_alphas`,
// `g2_alphas`, the per-layer `1 / max input_scale` activation scales).
// math_reference: flashinfer
// FlashInfer 0.6.17 csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh (`doActivationKernel` with
// `GLUAdaptor<SiLu>`: BF16 GEMM1 output, silu(gate) * linear in FP32, BF16 before the FP4 re-quantization; the
// finalize step's FP32 weighted sum over the slots).
#include "kernels/k7_moe.h"

#include "fault/fault_inject.h"

#include "kernels/k0_nvfp4.cuh"
#include "kernels/k0_nvfp4.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>

namespace cere::kernels {
namespace {

// -- workspace layout ---------------------------------------------------------------------------------------------

constexpr size_t kAlign = 256;

size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

struct Workspace {
    __nv_bfloat16 *logits;   // [m, E]
    int32_t *topk_ids;       // [m, K]
    float *topk_weights;     // [m, K]
    uint8_t *x_codes;        // [m, H/2]
    uint8_t *x_scales;       // [m, H/16]
    int32_t *counts;         // [E]
    int32_t *expert_offset;  // [E + 1] first sorted row of each expert
    int32_t *tile_offset;    // [E + 1] first GEMM row tile of each expert
    int32_t *perm;           // [R] sorted row -> pair index t * K + k
    int32_t *pos;            // [R] pair index -> sorted row
    __nv_bfloat16 *h;        // [R, I] SwiGLU output
    uint8_t *h_codes;        // [R, I/2]
    uint8_t *h_scales;       // [R, I/16]
    __nv_bfloat16 *y;        // [R, H] expert outputs (alpha applied, BF16)
    __nv_bfloat16 *s_gate;   // [m, S]
    __nv_bfloat16 *s_up;     // [m, S]
    __nv_bfloat16 *s_h;      // [m, S]
    __nv_bfloat16 *s_out;    // [m, H]
    size_t bytes;
};

// Carves the workspace (base may be null to only measure).
Workspace carve(const MoeDims &d, uint64_t m, uint8_t *base) {
    Workspace w{};
    size_t off = 0;
    auto take = [&](size_t bytes) {
        uint8_t *p = base == nullptr ? nullptr : base + off;
        off += align_up(bytes);
        return p;
    };
    const uint64_t r = m * d.top_k;
    w.logits = reinterpret_cast<__nv_bfloat16 *>(take(m * d.num_experts * 2));
    w.topk_ids = reinterpret_cast<int32_t *>(take(r * 4));
    w.topk_weights = reinterpret_cast<float *>(take(r * 4));
    w.x_codes = take(m * d.hidden / 2);
    w.x_scales = take(m * d.hidden / 16);
    w.counts = reinterpret_cast<int32_t *>(take(uint64_t{d.num_experts} * 4));
    w.expert_offset = reinterpret_cast<int32_t *>(take((uint64_t{d.num_experts} + 1) * 4));
    w.tile_offset = reinterpret_cast<int32_t *>(take((uint64_t{d.num_experts} + 1) * 4));
    w.perm = reinterpret_cast<int32_t *>(take(r * 4));
    w.pos = reinterpret_cast<int32_t *>(take(r * 4));
    w.h = reinterpret_cast<__nv_bfloat16 *>(take(r * d.inter * 2));
    w.h_codes = take(r * d.inter / 2);
    w.h_scales = take(r * d.inter / 16);
    w.y = reinterpret_cast<__nv_bfloat16 *>(take(r * d.hidden * 2));
    w.s_gate = reinterpret_cast<__nv_bfloat16 *>(take(m * d.shared_inter * 2));
    w.s_up = reinterpret_cast<__nv_bfloat16 *>(take(m * d.shared_inter * 2));
    w.s_h = reinterpret_cast<__nv_bfloat16 *>(take(m * d.shared_inter * 2));
    w.s_out = reinterpret_cast<__nv_bfloat16 *>(take(m * d.hidden * 2));
    w.bytes = off;
    return w;
}

constexpr uint32_t kMaxExperts = 1024;
constexpr uint32_t kMaxTopK = 32;

bool dims_ok(const MoeDims &d) {
    return d.hidden > 0 && d.hidden % 128 == 0 && d.inter > 0 && d.inter % 128 == 0 && d.shared_inter > 0 &&
           d.num_experts > 0 && d.num_experts <= kMaxExperts && d.top_k > 0 && d.top_k <= kMaxTopK &&
           d.top_k <= d.num_experts;
}

// -- block reductions ---------------------------------------------------------------------------------------------

constexpr int kRouterThreads = 128;
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

__device__ __forceinline__ float block_sum(float v, float *scratch) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    __syncthreads();
    if ((threadIdx.x & 31) == 0) {
        scratch[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    float r = scratch[0];
    for (int i = 1; i < kWarps; ++i) {
        r += scratch[i];
    }
    return r;
}

// Programmatic dependent launch (k1_primitives.cu pdl_enter, launch_pdl): the stream's next kernel may launch now
// (the K3 hc_combine_norm after a decode MoE stages its weights meanwhile), then this grid waits for the previous
// kernel's grid and its writes. Called first, before any global memory access, so results are a plain launch's.
__device__ __forceinline__ void pdl_enter() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

// The largest grid launched as a programmatic dependent launch (k1_primitives.cu kPdlMaxBlocks: decode/verify rows);
// a larger (prefill) grid launches plainly and its pdl_enter wait is a no-op.
constexpr unsigned long long kPdlMaxBlocks = 1024;

template <class... Params, class... Args>
cudaError_t launch_pdl(void (*kernel)(Params...), dim3 grid, dim3 block, cudaStream_t stream, Args... args) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1ull * grid.x * grid.y * grid.z <= kPdlMaxBlocks ? 1 : 0;
    return cudaLaunchKernelEx(&config, kernel, args...);
}

// The router's top-k order on BF16 logits (a larger value first, then the lower id; a NaN ranks with -inf, -0 ties
// +0) as one 32-bit key per expert: order-preserving logit bits in 31..16, 65,535 - id in 15..0. It is a strict total
// order, so the K winners and their order are the same however the argmax is reduced.
__device__ __forceinline__ uint32_t route_key(__nv_bfloat16 x, int e) {
    uint32_t b = __bfloat16_as_ushort(x);
    if ((b & 0x7FFFu) > 0x7F80u) {
        b = 0xFF80u;  // NaN ranks with -inf
    } else if (b == 0x8000u) {
        b = 0;  // -0 ties +0
    }
    const uint32_t ord = (b & 0x8000u) != 0 ? (~b & 0xFFFFu) : (b | 0x8000u);
    return ord << 16 | (0xFFFFu - static_cast<uint32_t>(e));
}

// -- router -------------------------------------------------------------------------------------------------------

// One block per token: FP32 softmax over the E logits, then the K largest logits (ties: lowest id), then the selected
// probabilities divided by their slot-order sum. Non-finite rows (a NaN or Inf in the token's hidden row makes every
// logit NaN or Inf) keep the output well formed: a NaN logit ranks below every number (-inf and NaN tie, lowest id
// first), the K ids of a token are always K distinct experts in [0, E) whatever the row holds, and a row whose softmax
// denominator is not a finite positive number (it is >= 1 for every row of finite logits) gets weights 0, never NaN.
//
// Thread x holds experts x + 128 i (i < I, E <= 128 I); slots past E are -inf logits and 0 probabilities, so I = 4
// at E <= 512 gives the same row_max (fmaxf with -inf) and row_sum (+= +0 on a sum that is never -0) as the
// kMaxExperts slots. Each probability is p / row_sum (IEEE division, this file has no fast math). Warp 0 then picks
// the top K: lane l sorts the keys (route_key) of experts l + 32 j, and each of the K rounds takes the warp's largest
// list head (__reduce_max_sync) and pops it from its owner's list. The K slots' probabilities are summed in slot
// order from 0.0f, as before: ids and weights are the bits of the 128-thread argmax router this replaced
// (ktest/k7_router_ktest.cu keeps it; test_kernels_k7_moe.py test_router_equals_block_router).
template <int I>
__global__ void __launch_bounds__(kRouterThreads)
    router_kernel(const __nv_bfloat16 *__restrict__ logits, int32_t *__restrict__ ids, float *__restrict__ weights,
                  int num_experts, int top_k) {
    constexpr int kLane = kRouterThreads / 32 * I;  // keys per lane of warp 0
    __shared__ float scratch[kWarps];
    __shared__ uint32_t key_s[kRouterThreads * I];
    __shared__ float p_s[kRouterThreads * I];
    pdl_enter();
    const int64_t t = blockIdx.x;
    const __nv_bfloat16 *row = logits + t * num_experts;
    __nv_bfloat16 x[I];
    float v[I];
    float p[I];
    float local_max = -INFINITY;
#pragma unroll
    for (int i = 0; i < I; ++i) {
        const int e = static_cast<int>(threadIdx.x) + i * kRouterThreads;
        x[i] = e < num_experts ? row[e] : __ushort_as_bfloat16(0xFF80u);
        v[i] = __bfloat162float(x[i]);
        local_max = fmaxf(local_max, v[i]);
    }
    const float row_max = block_max(local_max, scratch);
    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < I; ++i) {
        const int e = static_cast<int>(threadIdx.x) + i * kRouterThreads;
        p[i] = e < num_experts ? expf(v[i] - row_max) : 0.0f;
        local_sum += p[i];
    }
    const float row_sum = block_sum(local_sum, scratch);
#pragma unroll
    for (int i = 0; i < I; ++i) {
        const int e = static_cast<int>(threadIdx.x) + i * kRouterThreads;
        key_s[e] = e < num_experts ? route_key(x[i], e) : 0u;  // 0: below every expert's key
        p_s[e] = p[i];
    }
    __syncthreads();
    if (threadIdx.x >= 32) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    uint32_t key[kLane];
#pragma unroll
    for (int j = 0; j < kLane; ++j) {
        key[j] = key_s[lane + 32 * j];
    }
    // Bitonic sort of the lane's keys, descending.
#pragma unroll
    for (int size = 2; size <= kLane; size *= 2) {
#pragma unroll
        for (int stride = size / 2; stride > 0; stride /= 2) {
#pragma unroll
            for (int j = 0; j < kLane; ++j) {
                const int o = j ^ stride;
                if (o > j) {
                    const bool desc = (j & size) == 0;
                    const uint32_t a = key[j];
                    const uint32_t b = key[o];
                    key[j] = desc ? max(a, b) : min(a, b);
                    key[o] = desc ? min(a, b) : max(a, b);
                }
            }
        }
    }
    int sel_id = 0;  // lane k < top_k: the k-th winner
    for (int k = 0; k < top_k; ++k) {
        const uint32_t win = __reduce_max_sync(0xffffffffu, key[0]);
        const bool own = key[0] == win;
#pragma unroll
        for (int j = 0; j + 1 < kLane; ++j) {
            key[j] = own ? key[j + 1] : key[j];
        }
        key[kLane - 1] = own ? 0u : key[kLane - 1];
        sel_id = lane == k ? static_cast<int>(0xFFFFu - (win & 0xFFFFu)) : sel_id;
    }
    const float sel_p = p_s[sel_id] / row_sum;
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < static_cast<int>(kMaxTopK); ++k) {
        const float s = __shfl_sync(0xffffffffu, sel_p, k);
        sum = k < top_k ? sum + s : sum;
    }
    const bool finite_row = row_sum > 0.0f && row_sum < INFINITY;
    const float norm = sum > 0.0f ? sum : 1.0f;
    if (lane < top_k) {
        ids[t * top_k + lane] = sel_id;
        weights[t * top_k + lane] = finite_row ? sel_p / norm : 0.0f;
    }
}

// -- stable counting sort of the (token, slot) pairs by expert ----------------------------------------------------

constexpr int kSortThreads = 256;
constexpr int kTileRows = 32;  // GEMM row tile

// A pair whose id is outside [0, num_experts) (moe_route never writes one) is not counted and not placed: its pos
// is row 0, so the combine's read of the expert rows stays inside them.
__global__ void count_kernel(const int32_t *__restrict__ ids, int32_t *__restrict__ counts, int32_t *__restrict__ pos,
                             int64_t pairs, int num_experts) {
    const int64_t p = static_cast<int64_t>(blockIdx.x) * kSortThreads + threadIdx.x;
    if (p < pairs) {
        const int32_t id = ids[p];
        if (static_cast<uint32_t>(id) < static_cast<uint32_t>(num_experts)) {
            atomicAdd(&counts[id], 1);
        } else {
            pos[p] = 0;
        }
    }
}

// One block: exclusive prefix sums of the expert counts and of their row tiles.
__global__ void offsets_kernel(const int32_t *__restrict__ counts, int32_t *__restrict__ expert_offset,
                               int32_t *__restrict__ tile_offset, int num_experts) {
    __shared__ int32_t c[kMaxExperts];
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        c[e] = counts[e];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t rows = 0;
        int32_t tiles = 0;
        for (int e = 0; e < num_experts; ++e) {
            expert_offset[e] = rows;
            tile_offset[e] = tiles;
            rows += c[e];
            tiles += (c[e] + kTileRows - 1) / kTileRows;
        }
        expert_offset[num_experts] = rows;
        tile_offset[num_experts] = tiles;
    }
}

// One block per expert: its pairs in ascending pair order (stable), written to perm and pos.
__global__ void __launch_bounds__(kSortThreads)
    place_kernel(const int32_t *__restrict__ ids, const int32_t *__restrict__ expert_offset,
                 int32_t *__restrict__ perm, int32_t *__restrict__ pos, int64_t pairs) {
    __shared__ int32_t warp_counts[kSortThreads / 32];
    const int e = blockIdx.x;
    const int32_t first = expert_offset[e];
    if (expert_offset[e + 1] == first) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int32_t base = first;
    for (int64_t p0 = 0; p0 < pairs; p0 += kSortThreads) {
        const int64_t p = p0 + threadIdx.x;
        const bool hit = p < pairs && ids[p] == e;
        const uint32_t ballot = __ballot_sync(0xffffffffu, hit);
        if (lane == 0) {
            warp_counts[warp] = __popc(ballot);
        }
        __syncthreads();
        int32_t before = 0;
        int32_t total = 0;
        for (int w = 0; w < kSortThreads / 32; ++w) {
            before += w < warp ? warp_counts[w] : 0;
            total += warp_counts[w];
        }
        if (hit) {
            const int32_t row = base + before + __popc(ballot & ((1u << lane) - 1u));
            perm[row] = static_cast<int32_t>(p);
            pos[p] = row;
        }
        base += total;
        __syncthreads();
    }
}

// -- W4A4 grouped GEMM --------------------------------------------------------------------------------------------

constexpr int kGemmThreads = 128;
constexpr int kTileCols = 64;
constexpr int kChunk = 128;  // k elements staged per step
constexpr int kRowsPerThread = kTileRows / (kGemmThreads / kTileCols);

struct GemmArgs {
    const uint8_t *a_codes;   // [*, k/2]
    const uint8_t *a_scales;  // [*, k/16]
    const int32_t *a_perm;    // sorted row r -> A row perm[r] / a_div; null: A row = r
    int32_t a_div;
    const uint8_t *w0;  // [E, n, k/2] (GEMM1: gate, GEMM2: down)
    const uint8_t *s0;  // [E, n, k/16]
    const uint8_t *w1;  // GEMM1: up
    const uint8_t *s1;
    const float *alphas;  // [E]
    const int32_t *expert_offset;
    const int32_t *tile_offset;
    int32_t num_experts;
    int32_t n;
    int32_t k;
    __nv_bfloat16 *out;  // [R, n]
};

struct WeightChunk {
    uint32_t q[16];  // 64 int8 values 2 * e2m1
    float s[4];      // 4 block scales
};

__device__ __forceinline__ void load_weights(const uint8_t *w, const uint8_t *s, int64_t row, int k, int k0,
                                             WeightChunk &c) {
    const uint4 *src = reinterpret_cast<const uint4 *>(w + row * (k / 2) + k0 / 2);
    const uint4 a = __ldg(src);
    const uint4 b = __ldg(src + 1);
    const uint32_t words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        c.q[2 * i] = nvfp4::e2m1x4_to_s8x4(words[i] & 0xFFFFu);
        c.q[2 * i + 1] = nvfp4::e2m1x4_to_s8x4(words[i] >> 16);
    }
    const uint32_t sw = __ldg(reinterpret_cast<const uint32_t *>(s + row * (k / 16) + k0 / 16));
#pragma unroll
    for (int b4 = 0; b4 < 4; ++b4) {
        c.s[b4] = nvfp4::e4m3_to_float(static_cast<uint8_t>(sw >> (8 * b4)));
    }
}

// Exact per-block dot products (4 x dp4a over 16 int8 pairs), scaled by the exact product of the two FP8 block
// scales, accumulated in FP32 in block order.
__device__ __forceinline__ float dot64(const WeightChunk &c, const uint32_t *a, const float *as, float acc) {
#pragma unroll
    for (int b4 = 0; b4 < 4; ++b4) {
        int isum = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            isum = __dp4a(static_cast<int>(c.q[4 * b4 + j]), static_cast<int>(a[4 * b4 + j]), isum);
        }
        acc += static_cast<float>(isum) * (as[b4] * c.s[b4]);
    }
    return acc;
}

template <bool kGlu>
__global__ void __launch_bounds__(kGemmThreads) grouped_gemm_kernel(GemmArgs g) {
    __shared__ __align__(16) uint32_t a_s8[kTileRows][kChunk / 4];
    __shared__ float a_sf[kTileRows][kChunk / 16];
    __shared__ int64_t a_row[kTileRows];

    const int tile = blockIdx.y;
    if (tile >= g.tile_offset[g.num_experts]) {
        return;
    }
    int lo = 0;
    int hi = g.num_experts;  // tile_offset[lo] <= tile < tile_offset[hi]
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (g.tile_offset[mid] <= tile) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const int e = lo;
    const int32_t row0 = g.expert_offset[e] + (tile - g.tile_offset[e]) * kTileRows;
    const int rows = min(kTileRows, g.expert_offset[e + 1] - row0);

    if (threadIdx.x < kTileRows) {
        const int r = threadIdx.x;
        int64_t src = -1;
        if (r < rows) {
            src = g.a_perm != nullptr ? g.a_perm[row0 + r] / g.a_div : row0 + r;
        }
        a_row[r] = src;
    }

    const int col = blockIdx.x * kTileCols + (threadIdx.x % kTileCols);
    const bool col_ok = col < g.n;
    const int r_first = (threadIdx.x / kTileCols) * kRowsPerThread;
    const int64_t w_row = static_cast<int64_t>(e) * g.n + (col_ok ? col : 0);

    float acc0[kRowsPerThread];
    float acc1[kRowsPerThread];
#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
        acc0[i] = 0.0f;
        acc1[i] = 0.0f;
    }

    for (int k0 = 0; k0 < g.k; k0 += kChunk) {
        __syncthreads();
        // Stage the A chunk as int8 2 * e2m1 and its block scales as float.
        for (int idx = threadIdx.x; idx < kTileRows * (kChunk / 8); idx += kGemmThreads) {
            const int r = idx / (kChunk / 8);
            const int wd = idx % (kChunk / 8);
            const int64_t src = a_row[r];
            uint32_t v = 0;
            if (src >= 0) {
                v = *reinterpret_cast<const uint32_t *>(g.a_codes + src * (g.k / 2) + k0 / 2 + wd * 4);
            }
            a_s8[r][2 * wd] = nvfp4::e2m1x4_to_s8x4(v & 0xFFFFu);
            a_s8[r][2 * wd + 1] = nvfp4::e2m1x4_to_s8x4(v >> 16);
        }
        for (int idx = threadIdx.x; idx < kTileRows * (kChunk / 16); idx += kGemmThreads) {
            const int r = idx / (kChunk / 16);
            const int b = idx % (kChunk / 16);
            const int64_t src = a_row[r];
            a_sf[r][b] = src >= 0 ? nvfp4::e4m3_to_float(g.a_scales[src * (g.k / 16) + k0 / 16 + b]) : 0.0f;
        }
        __syncthreads();
        if (!col_ok) {
            continue;
        }
#pragma unroll
        for (int sub = 0; sub < kChunk / 64; ++sub) {
            WeightChunk c0;
            load_weights(g.w0, g.s0, w_row, g.k, k0 + sub * 64, c0);
            WeightChunk c1;
            if constexpr (kGlu) {
                load_weights(g.w1, g.s1, w_row, g.k, k0 + sub * 64, c1);
            }
#pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                const int r = r_first + i;
                if (r < rows) {
                    uint32_t a[16];
                    const uint4 *ap = reinterpret_cast<const uint4 *>(&a_s8[r][sub * 16]);
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const uint4 t4 = ap[q];
                        a[4 * q] = t4.x;
                        a[4 * q + 1] = t4.y;
                        a[4 * q + 2] = t4.z;
                        a[4 * q + 3] = t4.w;
                    }
                    const float *as = &a_sf[r][sub * 4];
                    acc0[i] = dot64(c0, a, as, acc0[i]);
                    if constexpr (kGlu) {
                        acc1[i] = dot64(c1, a, as, acc1[i]);
                    }
                }
            }
        }
    }

    if (!col_ok) {
        return;
    }
    const float alpha = g.alphas[e];
#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
        const int r = r_first + i;
        if (r >= rows) {
            continue;
        }
        // The int8 operands are 2 * e2m1 each: 0.25 (exact) restores the scale, then the epilogue's alpha * acc.
        const float g0 = __bfloat162float(__float2bfloat16((acc0[i] * 0.25f) * alpha));
        float result;
        if constexpr (kGlu) {
            const float up = __bfloat162float(__float2bfloat16((acc1[i] * 0.25f) * alpha));
            result = (g0 / (1.0f + expf(-g0))) * up;  // SiLU(gate) * linear in FP32
        } else {
            result = g0;
        }
        g.out[static_cast<int64_t>(row0 + r) * g.n + col] = __float2bfloat16(result);
    }
}

// -- shared expert and combine ------------------------------------------------------------------------------------

__global__ void silu_mul_kernel(const __nv_bfloat16 *__restrict__ gate, const __nv_bfloat16 *__restrict__ up,
                                __nv_bfloat16 *__restrict__ out, int64_t count) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float g = __bfloat162float(gate[i]);
        out[i] = __float2bfloat16((g / (1.0f + expf(-g))) * __bfloat162float(up[i]));
    }
}

constexpr int kCombineThreads = 128;

// How combine_kernel forms the BF16 routed row of token t (MoeTaps.routed_rows uses rows_combine_kernel).
enum class Routed : int {
    kSum = 0,    // K7's grouped rows: BF16(sum_k w_k * y_k) in slot order, y_k = y[pos[t * top_k + k]]
    kGiven = 1,  // `y` is the BF16 routed row itself [m, H] (MoeTaps.routed_in; `pos`, `topk_weights` unused)
};

// FP32 fma, round-to-nearest, denormal inputs and result flushed to zero: the FFMA.FTZ FlashInfer's
// finalizeMoeRoutingKernel compiles to in the P3b build (`-use_fast_math`, m23 design note P13).
__device__ __forceinline__ float fma_rn_ftz(float a, float b, float c) {
    float d;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}

// The sigmoid gate of token t, sigmoid(x . g), for a block of kCombineThreads threads: 128-thread strided partial
// sums, warp butterfly, warp-order sum. Every block of a token computes the same FP32 value. Ends past a
// __syncthreads.
__device__ __forceinline__ float combine_gate(const __nv_bfloat16 *__restrict__ x,
                                              const __nv_bfloat16 *__restrict__ gate_weight, int64_t t, int hidden,
                                              float *scratch) {
    float part = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += kCombineThreads) {
        part += __bfloat162float(x[t * hidden + j]) * __bfloat162float(gate_weight[j]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        part += __shfl_xor_sync(0xffffffffu, part, o);
    }
    if ((threadIdx.x & 31) == 0) {
        scratch[threadIdx.x >> 5] = part;
    }
    __syncthreads();
    float dot = 0.0f;
    for (int w = 0; w < kCombineThreads / 32; ++w) {
        dot += scratch[w];
    }
    return 1.0f / (1.0f + expf(-dot));
}

// One block per token: the BF16 routed row (Routed), then out = routed + sigmoid(x . g) * shared. `routed_out`
// (nullable) also receives the BF16 routed row.
template <Routed kMode>
__global__ void __launch_bounds__(kCombineThreads)
    combine_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ gate_weight,
                   const __nv_bfloat16 *__restrict__ y, const int32_t *__restrict__ pos,
                   const float *__restrict__ topk_weights, const __nv_bfloat16 *__restrict__ shared,
                   __nv_bfloat16 *__restrict__ out, __nv_bfloat16 *__restrict__ routed_out, int hidden, int top_k) {
    __shared__ float scratch[kCombineThreads / 32];
    const int64_t t = blockIdx.x;
    const float gate = combine_gate(x, gate_weight, t, hidden, scratch);
    for (int j = threadIdx.x; j < hidden; j += kCombineThreads) {
        __nv_bfloat16 routed_bf16;
        if constexpr (kMode == Routed::kGiven) {
            routed_bf16 = y[t * hidden + j];
        } else {
            float acc = 0.0f;
            for (int k = 0; k < top_k; ++k) {
                const int64_t r = pos[t * top_k + k];
                acc += topk_weights[t * top_k + k] * __bfloat162float(y[r * hidden + j]);
            }
            routed_bf16 = __float2bfloat16(acc);
        }
        if (routed_out != nullptr) {
            routed_out[t * hidden + j] = routed_bf16;
        }
        const float routed = __bfloat162float(routed_bf16);
        out[t * hidden + j] = __float2bfloat16(routed + gate * __bfloat162float(shared[t * hidden + j]));
    }
}

bool launched() { return cudaGetLastError() == cudaSuccess; }

constexpr int kRowsVec = 8;        // BF16 elements per thread on 16-byte aligned operands
constexpr int kRowsSlotBatch = 4;  // expert-row loads issued before their fma chain

template <int kVec>
struct alignas(sizeof(__nv_bfloat16) * kVec) BfPack {
    __nv_bfloat16 v[kVec];
};

// MoeTaps.routed_rows: block (t, c) forms elements [c, c + 1) * kCombineThreads * kVec of token t's routed row as
// FlashInfer's finalizeMoeRoutingKernel does (no bias, one EP rank): acc = 0, then acc = fma.rn.ftz(w_k, y_k, acc)
// in slot order (y_k = rows[pos[k * rows + t]], w_k = topk_weights[t * top_k + k]) and BF16 round-to-nearest-even;
// then out = routed + sigmoid(x . g) * shared with combine_gate (every block of the token recomputes the same gate)
// and combine_kernel's per-element expression. Per-element arithmetic only, so the chunking and the vector width
// change no bit.
template <int kVec>
__global__ void __launch_bounds__(kCombineThreads)
    rows_combine_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ gate_weight,
                        const __nv_bfloat16 *__restrict__ y, const int32_t *__restrict__ pos,
                        const float *__restrict__ topk_weights, const __nv_bfloat16 *__restrict__ shared,
                        __nv_bfloat16 *__restrict__ out, __nv_bfloat16 *__restrict__ routed_out, int hidden,
                        int top_k, int rows) {
    __shared__ float scratch[kCombineThreads / 32];
    __shared__ int64_t s_row[kMaxTopK];
    __shared__ float s_weight[kMaxTopK];
    const int64_t t = blockIdx.x;
    pdl_enter();
    if (static_cast<int>(threadIdx.x) < top_k) {
        s_row[threadIdx.x] = pos[int64_t{threadIdx.x} * rows + t];
        s_weight[threadIdx.x] = topk_weights[t * top_k + threadIdx.x];
    }
    const float gate = combine_gate(x, gate_weight, t, hidden, scratch);
    const int j0 = (static_cast<int>(blockIdx.y) * kCombineThreads + static_cast<int>(threadIdx.x)) * kVec;
    if (j0 >= hidden) {
        return;
    }
    using Pack = BfPack<kVec>;
    float acc[kVec];
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
        acc[e] = 0.0f;
    }
    for (int k0 = 0; k0 < top_k; k0 += kRowsSlotBatch) {
        Pack in[kRowsSlotBatch];
#pragma unroll
        for (int b = 0; b < kRowsSlotBatch; ++b) {
            if (k0 + b < top_k) {
                in[b] = *reinterpret_cast<const Pack *>(y + s_row[k0 + b] * hidden + j0);
            }
        }
#pragma unroll
        for (int b = 0; b < kRowsSlotBatch; ++b) {
            if (k0 + b < top_k) {
                const float w = s_weight[k0 + b];
#pragma unroll
                for (int e = 0; e < kVec; ++e) {
                    acc[e] = fma_rn_ftz(w, __bfloat162float(in[b].v[e]), acc[e]);
                }
            }
        }
    }
    const Pack sh = *reinterpret_cast<const Pack *>(shared + t * hidden + j0);
    Pack routed, res;
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
        routed.v[e] = __float2bfloat16_rn(acc[e]);
        const float r = __bfloat162float(routed.v[e]);
        res.v[e] = __float2bfloat16(r + gate * __bfloat162float(sh.v[e]));
    }
    if (routed_out != nullptr) {
        *reinterpret_cast<Pack *>(routed_out + t * hidden + j0) = routed;
    }
    *reinterpret_cast<Pack *>(out + t * hidden + j0) = res;
}

bool aligned16(const void *p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

// The shared expert without hook points into w.s_out on `stream`: gate and up share x, one small-M launch at
// m <= 16 (each output bit-equal to its own call), then SwiGLU and the down projection.
cere_status shared_expert(const MoeDims &d, const MoeLayer &L, const MoePlans &plans, const Workspace &w,
                          const __nv_bfloat16 *x, uint32_t m, cudaStream_t stream) {
    const GemmPlan *gate_up[2] = {plans.shared_gate_up, plans.shared_gate_up};
    const __nv_bfloat16 *weight[2] = {L.shared_gate, L.shared_up};
    void *const half[2] = {w.s_gate, w.s_up};
    cere_status s = gemm_segments(gate_up, weight, half, 2, x, m, stream, plans.cls);
    if (s != CERE_OK) {
        return s;
    }
    const int64_t shared_count = int64_t{m} * d.shared_inter;
    silu_mul_kernel<<<static_cast<unsigned>((shared_count + 255) / 256), 256, 0, stream>>>(w.s_gate, w.s_up, w.s_h,
                                                                                          shared_count);
    if (!launched()) {
        return CERE_ERR_INTERNAL;
    }
    return plans.shared_down->run(w.s_h, L.shared_down, m, w.s_out, stream, plans.cls);
}

}  // namespace

size_t moe_workspace_bytes(const MoeDims &dims, uint32_t max_m) {
    if (!dims_ok(dims) || max_m == 0) {
        return 0;
    }
    return carve(dims, max_m, nullptr).bytes;
}

cere_status moe_route(const __nv_bfloat16 *logits, int32_t *topk_ids, float *topk_weights, uint32_t m,
                      uint32_t num_experts, uint32_t top_k, cudaStream_t stream) {
    if (logits == nullptr || topk_ids == nullptr || topk_weights == nullptr || m == 0 || num_experts == 0 ||
        num_experts > kMaxExperts || top_k == 0 || top_k > kMaxTopK || top_k > num_experts) {
        return CERE_ERR_ARGUMENT;
    }
    const auto kernel = num_experts <= 4u * kRouterThreads ? router_kernel<4> : router_kernel<8>;
    return launch_pdl(kernel, dim3(m), dim3(kRouterThreads), stream, logits, topk_ids, topk_weights,
                      static_cast<int>(num_experts), static_cast<int>(top_k)) == cudaSuccess
               ? CERE_OK
               : CERE_ERR_INTERNAL;
}

bool moe_shared_forks(const MoePlans &plans, uint32_t m) {
    const uint32_t rows = plans.cls != 0 ? plans.cls : m;
    return plans.side != nullptr && plans.fork != nullptr && plans.join != nullptr && m >= 1 && rows <= k2s::kMaxM;
}

cere_status moe_shared_fork(const MoeDims &d, const MoeLayer &L, const MoePlans &plans, const __nv_bfloat16 *x,
                            uint32_t m, void *workspace, size_t workspace_bytes, cudaStream_t stream,
                            const __nv_bfloat16 **shared) {
    if (!dims_ok(d) || !moe_shared_forks(plans, m) || x == nullptr || shared == nullptr || workspace == nullptr ||
        reinterpret_cast<uintptr_t>(workspace) % kAlign != 0 || plans.shared_gate_up == nullptr ||
        plans.shared_down == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    for (const void *p : {static_cast<const void *>(L.shared_gate), static_cast<const void *>(L.shared_up),
                          static_cast<const void *>(L.shared_down)}) {
        if (p == nullptr || !aligned16(p)) {
            return CERE_ERR_ARGUMENT;
        }
    }
    const Workspace w = carve(d, m, static_cast<uint8_t *>(workspace));
    if (w.bytes > workspace_bytes) {
        return CERE_ERR_ARGUMENT;
    }
    if (cudaEventRecord(plans.fork, stream) != cudaSuccess ||
        cudaStreamWaitEvent(plans.side, plans.fork, 0) != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    cere_status s = shared_expert(d, L, plans, w, x, m, plans.side);
    if (s == CERE_OK && cudaEventRecord(plans.join, plans.side) != cudaSuccess) {
        s = CERE_ERR_INTERNAL;
    }
    if (s != CERE_OK) {
        // Whatever the side stream got is joined into `stream` (best effort: the failure being reported is the one
        // that matters), so no launched work is left unjoined (a stream capture would end unjoined).
        (void)cudaEventRecord(plans.join, plans.side);
        (void)cudaStreamWaitEvent(stream, plans.join, 0);
        return s;
    }
    *shared = w.s_out;
    return CERE_OK;
}

cere_status moe_shared_join(const MoePlans &plans, cudaStream_t stream) {
    if (plans.side == nullptr || plans.join == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    return cudaStreamWaitEvent(stream, plans.join, 0) == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

cere_status moe_block(const MoeDims &d, const MoeLayer &L, const MoePlans &plans, const __nv_bfloat16 *x,
                      uint32_t m, __nv_bfloat16 *out, void *workspace, size_t workspace_bytes, cudaStream_t stream,
                      const MoeTaps *taps) {
    if (!dims_ok(d) || m == 0 || x == nullptr || out == nullptr || workspace == nullptr ||
        reinterpret_cast<uintptr_t>(workspace) % kAlign != 0 || plans.router == nullptr ||
        plans.shared_gate_up == nullptr || plans.shared_down == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    const void *weights[] = {L.router, L.gate_weight, L.gate_scale, L.up_weight, L.up_scale, L.down_weight,
                             L.down_scale, L.g1_alphas, L.g2_alphas, L.shared_gate, L.shared_up, L.shared_down,
                             L.shared_expert_gate};
    for (const void *p : weights) {
        if (p == nullptr || !aligned16(p)) {
            return CERE_ERR_ARGUMENT;
        }
    }
    const Workspace w = carve(d, m, static_cast<uint8_t *>(workspace));
    if (w.bytes > workspace_bytes) {
        return CERE_ERR_ARGUMENT;
    }
    const int64_t pairs = int64_t{m} * d.top_k;
    if (pairs > INT32_MAX / 2) {
        return CERE_ERR_ARGUMENT;
    }
    cere_status s;
    const bool routed_given = taps != nullptr && taps->routed_in != nullptr;
    const bool routed_rows = taps != nullptr && taps->routed_rows != nullptr;
    if (routed_rows && (routed_given || taps->routed_pos == nullptr || taps->routed_scales == nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    // Row tiles: sum_e ceil(count_e / kTileRows) <= pairs / kTileRows + min(E, pairs); blocks past the actual
    // count (tile_offset[E], known on the device only) return at once. Checked before the shared expert forks.
    const int64_t tiles = pairs / kTileRows + std::min<int64_t>(d.num_experts, pairs);
    if (!routed_given && !routed_rows && tiles > 65535) {
        return CERE_ERR_ARGUMENT;
    }
    const bool shared_given = taps != nullptr && taps->shared_in != nullptr;
    const bool hooked = taps != nullptr && taps->point != nullptr;
    const bool detail = hooked && taps->detail;
    // A hook point of `cols` columns over `values` [m, cols] (only "*.out" points may be written by the hook).
    const auto point = [&](const char *name, const __nv_bfloat16 *values, uint32_t cols) {
        return taps->point(taps->ctx, name, const_cast<__nv_bfloat16 *>(values), m, cols, stream);
    };

    // The shared-expert fork (k7_moe.h MoePlans) when the routed experts run here (P3b forks before its router).
    const __nv_bfloat16 *shared = shared_given ? taps->shared_in : w.s_out;
    const bool forked =
        !shared_given && !hooked && !routed_given && !routed_rows && moe_shared_forks(plans, m);
    SharedForkScope fork_scope(plans, stream);  // joins the fork on every error return below
    if (forked) {
        if ((s = moe_shared_fork(d, L, plans, x, m, workspace, workspace_bytes, stream, &shared)) != CERE_OK) {
            return s;
        }
        fork_scope.arm();
        CERE_FAULT_POINT(moe_fork, return CERE_ERR_INTERNAL);  // OH-08: an error after the fork (KGM-2)
    }
    if (!routed_given && !routed_rows) {
        // Router.
        if (hooked && (s = point("gemm.moe_router.in", x, d.hidden)) != CERE_OK) {
            return s;
        }
        if ((s = plans.router->run(x, L.router, m, w.logits, stream, plans.cls)) != CERE_OK) {
            return s;
        }
        if (hooked && (s = point("gemm.moe_router.out", w.logits, d.num_experts)) != CERE_OK) {
            return s;
        }
        if (detail && (s = point("mlp.gate", w.logits, d.num_experts)) != CERE_OK) {
            return s;
        }
        if ((s = moe_route(w.logits, w.topk_ids, w.topk_weights, m, d.num_experts, d.top_k, stream)) != CERE_OK) {
            return s;
        }
        if (taps != nullptr && taps->route != nullptr &&
            (s = taps->route(taps->ctx, w.topk_ids, w.topk_weights, m, d.top_k, stream)) != CERE_OK) {
            return s;
        }
        // Expert grouping.
        if (cudaMemsetAsync(w.counts, 0, sizeof(int32_t) * d.num_experts, stream) != cudaSuccess) {
            return CERE_ERR_INTERNAL;
        }
        count_kernel<<<static_cast<unsigned>((pairs + kSortThreads - 1) / kSortThreads), kSortThreads, 0, stream>>>(
            w.topk_ids, w.counts, w.pos, pairs, static_cast<int>(d.num_experts));
        offsets_kernel<<<1, kSortThreads, 0, stream>>>(w.counts, w.expert_offset, w.tile_offset,
                                                       static_cast<int>(d.num_experts));
        place_kernel<<<d.num_experts, kSortThreads, 0, stream>>>(w.topk_ids, w.expert_offset, w.perm, w.pos, pairs);
        if (!launched()) {
            return CERE_ERR_INTERNAL;
        }
        // GEMM1 (gate and up with g1_alphas) + SwiGLU, then the a2 quantization.
        if ((s = fp4_quantize(x, L.a1_global_scale, w.x_codes, w.x_scales, m, d.hidden, stream)) != CERE_OK) {
            return s;
        }
        GemmArgs g1{w.x_codes,     w.x_scales,    w.perm,        static_cast<int32_t>(d.top_k),
                    L.gate_weight, L.gate_scale,  L.up_weight,   L.up_scale,
                    L.g1_alphas,   w.expert_offset, w.tile_offset, static_cast<int32_t>(d.num_experts),
                    static_cast<int32_t>(d.inter), static_cast<int32_t>(d.hidden), w.h};
        grouped_gemm_kernel<true><<<dim3((d.inter + kTileCols - 1) / kTileCols, static_cast<unsigned>(tiles)), kGemmThreads, 0, stream>>>(g1);
        if (!launched()) {
            return CERE_ERR_INTERNAL;
        }
        if ((s = fp4_quantize(w.h, L.a2_global_scale, w.h_codes, w.h_scales, pairs, d.inter, stream)) != CERE_OK) {
            return s;
        }
        // GEMM2 (down with g2_alphas).
        GemmArgs g2{w.h_codes,     w.h_scales,    nullptr,       1,
                    L.down_weight, L.down_scale,  nullptr,       nullptr,
                    L.g2_alphas,   w.expert_offset, w.tile_offset, static_cast<int32_t>(d.num_experts),
                    static_cast<int32_t>(d.hidden), static_cast<int32_t>(d.inter), w.y};
        grouped_gemm_kernel<false><<<dim3((d.hidden + kTileCols - 1) / kTileCols, static_cast<unsigned>(tiles)), kGemmThreads, 0, stream>>>(g2);
        if (!launched()) {
            return CERE_ERR_INTERNAL;
        }
    }

    if (forked) {
        if ((s = fork_scope.join()) != CERE_OK) {
            return s;
        }
    } else if (!shared_given && !hooked) {
        // Shared expert (BF16 K2 GEMMs).
        if ((s = shared_expert(d, L, plans, w, x, m, stream)) != CERE_OK) {
            return s;
        }
    } else if (!shared_given) {
        // Shared expert (BF16 K2 GEMMs), one call per weight between its hook points.
        const __nv_bfloat16 *weight[2] = {L.shared_gate, L.shared_up};
        __nv_bfloat16 *half[2] = {w.s_gate, w.s_up};
        for (int i = 0; i < 2; ++i) {
            if ((s = point("gemm.moe_shared_gate_up.in", x, d.hidden)) != CERE_OK ||
                (s = plans.shared_gate_up->run(x, weight[i], m, half[i], stream, plans.cls)) != CERE_OK ||
                (s = point("gemm.moe_shared_gate_up.out", half[i], d.shared_inter)) != CERE_OK) {
                return s;
            }
        }
        if (detail) {
            // gate | up [m, 2 S] in the SwiGLU region (dead after the a2 quantization, unused when routed_given).
            const size_t half_bytes = sizeof(__nv_bfloat16) * d.shared_inter;
            for (int i = 0; i < 2; ++i) {
                if (cudaMemcpy2DAsync(w.h + i * d.shared_inter, 2 * half_bytes, half[i], half_bytes, half_bytes, m,
                                      cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                    return CERE_ERR_INTERNAL;
                }
            }
            if ((s = point("mlp.shared_expert.gate_up_proj", w.h, 2 * d.shared_inter)) != CERE_OK) {
                return s;
            }
        }
        const int64_t shared_count = int64_t{m} * d.shared_inter;
        silu_mul_kernel<<<static_cast<unsigned>((shared_count + 255) / 256), 256, 0, stream>>>(w.s_gate, w.s_up,
                                                                                              w.s_h, shared_count);
        if (!launched()) {
            return CERE_ERR_INTERNAL;
        }
        if ((s = point("gemm.moe_shared_down.in", w.s_h, d.shared_inter)) != CERE_OK) {
            return s;
        }
        if ((s = plans.shared_down->run(w.s_h, L.shared_down, m, w.s_out, stream, plans.cls)) != CERE_OK) {
            return s;
        }
        if ((s = point("gemm.moe_shared_down.out", w.s_out, d.hidden)) != CERE_OK) {
            return s;
        }
    }
    if (taps != nullptr && taps->shared != nullptr && taps->shared != shared &&
        cudaMemcpyAsync(taps->shared, shared, sizeof(__nv_bfloat16) * m * d.hidden, cudaMemcpyDeviceToDevice,
                        stream) != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    // Combine.
    const int hidden = static_cast<int>(d.hidden), top_k = static_cast<int>(d.top_k), rows = static_cast<int>(m);
    if (routed_given) {
        __nv_bfloat16 *routed_out = taps->routed == taps->routed_in ? nullptr : taps->routed;
        combine_kernel<Routed::kGiven><<<m, kCombineThreads, 0, stream>>>(
            x, L.shared_expert_gate, taps->routed_in, nullptr, nullptr, shared, out, routed_out, hidden, top_k);
    } else if (routed_rows) {
        // One block per (token, kCombineThreads * vector chunk of the hidden row); 16-byte vectors when every
        // operand row allows them (hidden % 128 == 0, so only the base pointers decide).
        const bool vec = aligned16(taps->routed_rows) && aligned16(shared) && aligned16(out) &&
                         (taps->routed == nullptr || aligned16(taps->routed));
        const int per_block = kCombineThreads * (vec ? kRowsVec : 1);
        const dim3 grid(static_cast<unsigned>(m), static_cast<unsigned>((hidden + per_block - 1) / per_block));
        const cudaError_t e =
            vec ? launch_pdl(rows_combine_kernel<kRowsVec>, grid, dim3(kCombineThreads), stream, x,
                             L.shared_expert_gate, taps->routed_rows, taps->routed_pos, taps->routed_scales, shared,
                             out, taps->routed, hidden, top_k, rows)
                : launch_pdl(rows_combine_kernel<1>, grid, dim3(kCombineThreads), stream, x, L.shared_expert_gate,
                             taps->routed_rows, taps->routed_pos, taps->routed_scales, shared, out, taps->routed,
                             hidden, top_k, rows);
        if (e != cudaSuccess) {
            return CERE_ERR_INTERNAL;
        }
    } else {
        combine_kernel<Routed::kSum><<<m, kCombineThreads, 0, stream>>>(
            x, L.shared_expert_gate, w.y, w.pos, w.topk_weights, shared, out,
            taps != nullptr ? taps->routed : nullptr, hidden, top_k);
    }
    return launched() ? CERE_OK : CERE_ERR_INTERNAL;
}

}  // namespace cere::kernels
