// K7T (m215 design note D4): the tensor-core NVFP4 MoE grouped GEMM, as kernel-test entry points only
// (executor/tests/gpu/test_k7t_moe.py, suite `k7t`). Device pointers, int64 sizes, the stream last; 0 = success, else
// a cere_status.
//
//   int cere_ktest_k7t_gemm(const void *a_codes,        /* uint8 [rows, k/2] E2M1 codes, rows sorted by expert */
//                           const void *a_scales,       /* FP8 E4M3 [rows, k/16], linear layout */
//                           const void *expert_counts,  /* int32 [num_experts]: rows of each expert, in order;
//                                                          precondition: sum == rows (not checked on the device) */
//                           const void *w0,             /* uint8 [num_experts, n, k/2]: GEMM1 gate / GEMM2 down */
//                           const void *s0,             /* FP8 E4M3 [num_experts, n, k/16] */
//                           const void *w1,             /* uint8 [num_experts, n, k/2]: GEMM1 up; null: GEMM2 */
//                           const void *s1,             /* FP8 E4M3 [num_experts, n, k/16]; null with w1 */
//                           void *out0,                 /* float32 [rows, n]: raw accumulators of w0 */
//                           void *out1,                 /* float32 [rows, n]: raw accumulators of w1; null with w1 */
//                           int64_t rows, int64_t n, int64_t k, int64_t num_experts,
//                           cudaStream_t stream);
//   int cere_ktest_k7t_block(const void *x, const void *router_weight, const void *gate_weight,
//                            const void *gate_scale, const void *up_weight, const void *up_scale,
//                            const void *down_weight, const void *down_scale, const void *g1_alphas,
//                            const void *g2_alphas, float a1_global_scale, float a2_global_scale,
//                            const void *shared_gate_weight, const void *shared_up_weight,
//                            const void *shared_down_weight, const void *shared_expert_gate, void *out,
//                            int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter,
//                            int64_t shared_inter,
//                            int64_t gemm_mode,          /* 0: K7's dot64 GEMM (ported), 1: OMMA */
//                            cudaStream_t stream);
//       cere_ktest_moe_block's parameters (k7_ktest.cpp, test_kernels_k7_moe.py) plus gemm_mode; out bf16 [m, hidden].
//   int cere_ktest_k7t_block_stages(<cere_ktest_k7t_block's parameters up to gemm_mode>,
//                                   void *perm,      /* int32 [m*top_k]: sorted row -> pair t*top_k + k */
//                                   void *h,         /* bf16 [m*top_k, inter]: GEMM1 + SwiGLU, pre-a2-quantize */
//                                   void *h_codes,   /* uint8 [m*top_k, inter/2]: the a2 quantization's codes */
//                                   void *h_scales,  /* FP8 E4M3 [m*top_k, inter/16]: and its scales */
//                                   void *y,         /* bf16 [m*top_k, hidden]: GEMM2 (alpha, BF16), pre-combine */
//                                   cudaStream_t stream);
//       cere_ktest_k7t_block plus the block's intermediates (sorted-row order), copied out after the block.
//
// cere_ktest_k7t_gemm returns the raw FP32 accumulators of the OMMA grouped GEMM, before any alpha or epilogue: per
// element, one `mma.sync.m16n8k64 ... kind::mxf4nvf4.block_scale.scale_vec::4X ... ue4m3` per 64-wide K step, chained
// in ascending K order (each step's C is the previous step's D), no split-K. k a multiple of 128, n of 64.
//
// cere_ktest_k7t_block is K7's MoE block (kernels::moe_block, k7_moe.cu) rebuilt from test-only ports of its
// anonymous-namespace pieces (workspace carve, sort/place, grouped GEMM and epilogue, silu_mul, combine) and its
// orchestration, with identical launch geometry; the router GEMM and shared-expert plans are built as
// cere_ktest_moe_block builds them, and the public kernels::moe_route and kernels::fp4_quantize run unchanged.
// gemm_mode 0 runs the ported dot64 grouped GEMM (bit-equal to cere_ktest_moe_block), 1 the OMMA grouped GEMM;
// nothing else differs. One tile configuration (32 rows x 64 columns, 4 warps) serves every m.
//
// OMMA operands (PTX ISA 9.7.14.3, m16n8k64 e2m1; probe an internal benchmark report): lane =
// 4 g + q. A (rows = sorted token rows): a0 = row g, k 8q..8q+7; a1 = row g+8; a2 = row g, k 32+8q..; a3 = row g+8,
// k 32+8q.. (element i in nibble i, the checkpoint byte order). B (columns = weight rows): b0 = column g, k 8q..8q+7;
// b1 = k 32+8q... D: d0, d1 = row g, columns 2q, 2q+1; d2, d3 = row g+8. Scales, selectors {0, 0} (PTX Table 37):
// lanes q = 0, 1 of each quad supply SF_A, lane q = 0 supplies SF_B of column g (4 bytes = the 4 K blocks of the
// step, block j in byte j). Within the SF_A pair, q = 0 is taken to hold row g and q = 1 row g+8 (CUTLASS's SM120
// SFA layout; klab D2 (d) asserts the mapping and D5 (i) fails on any other): lanes with q even hold row g's
// scales, q odd row g+8's, and every lane its column's.
#include "kernels/k0_nvfp4.cuh"
#include "kernels/k0_nvfp4.h"
#include "kernels/k7_moe.h"
#include "ktest/k2_ktest.h"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace cere::ktest::k7t {
namespace {

using kernels::MoeDims;
using kernels::MoeLayer;
using kernels::MoePlans;
namespace nvfp4 = kernels::nvfp4;

// -- workspace layout (port of k7_moe.cu) ------------------------------------------------------------------------

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

// -- stable counting sort of the (token, slot) pairs by expert (port of k7_moe.cu) --------------------------------

constexpr int kSortThreads = 256;
constexpr int kTileRows = 32;  // GEMM row tile

__global__ void count_kernel(const int32_t *__restrict__ ids, int32_t *__restrict__ counts, int64_t pairs) {
    const int64_t p = static_cast<int64_t>(blockIdx.x) * kSortThreads + threadIdx.x;
    if (p < pairs) {
        atomicAdd(&counts[ids[p]], 1);
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

// -- W4A4 grouped GEMM: geometry, tile lookup and epilogue shared by both modes -----------------------------------

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

// Where a GEMM writes: `out` through the epilogue, or (raw0, raw1 non-null) the FP32 accumulators themselves.
struct RawOut {
    float *raw0 = nullptr;  // [R, n] accumulators of w0
    float *raw1 = nullptr;  // [R, n] accumulators of w1 (GLU only)
};

// The row tile of block `tile`: its expert, first sorted row and row count; false past the device's tile count.
__device__ __forceinline__ bool tile_rows(const GemmArgs &g, int tile, int &e, int32_t &row0, int &rows) {
    if (tile >= g.tile_offset[g.num_experts]) {
        return false;
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
    e = lo;
    row0 = g.expert_offset[e] + (tile - g.tile_offset[e]) * kTileRows;
    rows = min(kTileRows, g.expert_offset[e + 1] - row0);
    return true;
}

// The epilogue of one output element from its value-scaled accumulators v0 (w0) and v1 (w1, GLU): alpha, BF16, and
// for GLU SiLU(gate) * up in FP32; BF16 out.
template <bool kGlu>
__device__ __forceinline__ void epilogue(const GemmArgs &g, int64_t idx, float alpha, float v0, float v1) {
    const float g0 = __bfloat162float(__float2bfloat16(v0 * alpha));
    float result;
    if constexpr (kGlu) {
        const float up = __bfloat162float(__float2bfloat16(v1 * alpha));
        result = (g0 / (1.0f + expf(-g0))) * up;  // SiLU(gate) * linear in FP32
    } else {
        result = g0;
    }
    g.out[idx] = __float2bfloat16(result);
}

// -- gemm_mode 0: K7's dot64 grouped GEMM (port of k7_moe.cu) -----------------------------------------------------

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

    int e;
    int32_t row0;
    int rows;
    if (!tile_rows(g, blockIdx.y, e, row0, rows)) {
        return;
    }

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
        epilogue<kGlu>(g, static_cast<int64_t>(row0 + r) * g.n + col, alpha, acc0[i] * 0.25f, acc1[i] * 0.25f);
    }
}

// -- gemm_mode 1: the OMMA grouped GEMM ---------------------------------------------------------------------------

constexpr int kWarpCols = kTileCols / (kGemmThreads / 32);  // 16 columns per warp: two n8 tiles
constexpr int kNTiles = kWarpCols / 8;
constexpr int kMTiles = kTileRows / 16;
constexpr int kAWords = kChunk / 8 + 4;  // e2m1 words per staged row, padded: conflict-free fragment reads

// D (+)= A (16 x 64 e2m1, row) * B (64 x 8 e2m1, col), block scales ue4m3 per 16 K, FP32 accumulators.
__device__ __forceinline__ void omma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                     uint32_t sb) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {0, 0}, %11, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

// One weight row's operands for one 64-wide K step: the B fragment words and the 4 block scales.
__device__ __forceinline__ void load_b(const uint8_t *w, const uint8_t *s, int64_t row, int k, int kk, int q,
                                       uint32_t &b0, uint32_t &b1, uint32_t &sb) {
    const uint8_t *p = w + row * (k / 2) + kk / 2 + q * 4;
    b0 = __ldg(reinterpret_cast<const uint32_t *>(p));
    b1 = __ldg(reinterpret_cast<const uint32_t *>(p + 16));
    sb = __ldg(reinterpret_cast<const uint32_t *>(s + row * (k / 16) + kk / 16));
}

// Same tiles and grid as grouped_gemm_kernel. Warp w covers the tile's columns 16 w .. 16 w + 15 (two n8 tiles) and
// its 32 rows (two m16 tiles; the second is skipped when the tile has at most 16 rows). Every output element is one
// chain of OMMA steps in ascending K. kRaw: the accumulators are written to RawOut, else through the epilogue.
template <bool kGlu, bool kRaw>
__global__ void __launch_bounds__(kGemmThreads) grouped_gemm_omma_kernel(GemmArgs g, RawOut raw) {
    __shared__ uint32_t a_q[kTileRows][kAWords];        // e2m1 codes as stored (8 per word)
    __shared__ uint32_t a_sc[kTileRows][kChunk / 64];   // ue4m3 scale bytes, 4 per 64-wide K step
    __shared__ int64_t a_row[kTileRows];

    int e;
    int32_t row0;
    int rows;
    if (!tile_rows(g, blockIdx.y, e, row0, rows)) {
        return;
    }
    if (threadIdx.x < kTileRows) {
        const int r = threadIdx.x;
        int64_t src = -1;
        if (r < rows) {
            src = g.a_perm != nullptr ? g.a_perm[row0 + r] / g.a_div : row0 + r;
        }
        a_row[r] = src;
    }

    const int lane = threadIdx.x & 31;
    const int gq = lane >> 2;  // fragment row / column group
    const int q = lane & 3;
    const int col_base = blockIdx.x * kTileCols + (threadIdx.x >> 5) * kWarpCols;
    const int m_tiles = rows > 16 ? 2 : 1;  // block-uniform

    float acc0[kMTiles][kNTiles][4];
    float acc1[kMTiles][kNTiles][4];
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc0[mt][nt][i] = 0.0f;
                acc1[mt][nt][i] = 0.0f;
            }
        }
    }
    int64_t w_row[kNTiles];
#pragma unroll
    for (int nt = 0; nt < kNTiles; ++nt) {
        w_row[nt] = static_cast<int64_t>(e) * g.n + col_base + nt * 8 + gq;
    }

    for (int k0 = 0; k0 < g.k; k0 += kChunk) {
        __syncthreads();
        for (int idx = threadIdx.x; idx < kTileRows * (kChunk / 8); idx += kGemmThreads) {
            const int r = idx / (kChunk / 8);
            const int wd = idx % (kChunk / 8);
            const int64_t src = a_row[r];
            a_q[r][wd] = src >= 0 ? *reinterpret_cast<const uint32_t *>(g.a_codes + src * (g.k / 2) + k0 / 2 + wd * 4)
                                  : 0u;
        }
        if (threadIdx.x < kTileRows * (kChunk / 64)) {
            const int r = threadIdx.x / (kChunk / 64);
            const int j = threadIdx.x % (kChunk / 64);
            const int64_t src = a_row[r];
            a_sc[r][j] = src >= 0
                             ? *reinterpret_cast<const uint32_t *>(g.a_scales + src * (g.k / 16) + k0 / 16 + j * 4)
                             : 0u;
        }
        __syncthreads();
#pragma unroll
        for (int sub = 0; sub < kChunk / 64; ++sub) {
            const int kk = k0 + sub * 64;
            uint32_t b0[kNTiles], b1[kNTiles], sb0[kNTiles];
            uint32_t c0[kNTiles], c1[kNTiles], sb1[kNTiles];
#pragma unroll
            for (int nt = 0; nt < kNTiles; ++nt) {
                load_b(g.w0, g.s0, w_row[nt], g.k, kk, q, b0[nt], b1[nt], sb0[nt]);
                if constexpr (kGlu) {
                    load_b(g.w1, g.s1, w_row[nt], g.k, kk, q, c0[nt], c1[nt], sb1[nt]);
                }
            }
#pragma unroll
            for (int mt = 0; mt < kMTiles; ++mt) {
                if (mt >= m_tiles) {
                    break;
                }
                const int r = mt * 16 + gq;
                const uint32_t a[4] = {a_q[r][sub * 8 + q], a_q[r + 8][sub * 8 + q], a_q[r][sub * 8 + 4 + q],
                                       a_q[r + 8][sub * 8 + 4 + q]};
                const uint32_t sa = a_sc[r + ((q & 1) ? 8 : 0)][sub];
#pragma unroll
                for (int nt = 0; nt < kNTiles; ++nt) {
                    omma(acc0[mt][nt], a, b0[nt], b1[nt], sa, sb0[nt]);
                    if constexpr (kGlu) {
                        omma(acc1[mt][nt], a, c0[nt], c1[nt], sa, sb1[nt]);
                    }
                }
            }
        }
    }

    const float alpha = kRaw ? 0.0f : g.alphas[e];
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = mt * 16 + gq + (i >= 2 ? 8 : 0);
                if (r >= rows) {
                    continue;
                }
                const int64_t idx = static_cast<int64_t>(row0 + r) * g.n + col_base + nt * 8 + 2 * q + (i & 1);
                if constexpr (kRaw) {
                    raw.raw0[idx] = acc0[mt][nt][i];
                    if constexpr (kGlu) {
                        raw.raw1[idx] = acc1[mt][nt][i];
                    }
                } else {
                    // The accumulators hold e2m1 values (no int8 factor): the epilogue takes them as they are.
                    epilogue<kGlu>(g, idx, alpha, acc0[mt][nt][i], acc1[mt][nt][i]);
                }
            }
        }
    }
}

// -- shared expert and combine (port of k7_moe.cu) -----------------------------------------------------------------

__global__ void silu_mul_kernel(const __nv_bfloat16 *__restrict__ gate, const __nv_bfloat16 *__restrict__ up,
                                __nv_bfloat16 *__restrict__ out, int64_t count) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float g = __bfloat162float(gate[i]);
        out[i] = __float2bfloat16((g / (1.0f + expf(-g))) * __bfloat162float(up[i]));
    }
}

constexpr int kCombineThreads = 128;

// One block per token: routed = BF16(sum_k w_k * y_k) in slot order, then out = routed + sigmoid(x . g) * shared.
__global__ void __launch_bounds__(kCombineThreads)
    combine_kernel(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ gate_weight,
                   const __nv_bfloat16 *__restrict__ y, const int32_t *__restrict__ pos,
                   const float *__restrict__ topk_weights, const __nv_bfloat16 *__restrict__ shared,
                   __nv_bfloat16 *__restrict__ out, int hidden, int top_k) {
    __shared__ float scratch[kCombineThreads / 32];
    const int64_t t = blockIdx.x;
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
    const float gate = 1.0f / (1.0f + expf(-dot));
    for (int j = threadIdx.x; j < hidden; j += kCombineThreads) {
        float acc = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            const int64_t r = pos[t * top_k + k];
            acc += topk_weights[t * top_k + k] * __bfloat162float(y[r * hidden + j]);
        }
        const float routed = __bfloat162float(__float2bfloat16(acc));
        out[t * hidden + j] = __float2bfloat16(routed + gate * __bfloat162float(shared[t * hidden + j]));
    }
}

bool launched() { return cudaGetLastError() == cudaSuccess; }

bool aligned16(const void *p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

bool u32(int64_t v) { return v > 0 && v <= std::numeric_limits<int32_t>::max(); }

// Row tiles of `pairs` sorted rows over E experts: sum_e ceil(count_e / kTileRows) <= pairs / kTileRows +
// min(E, pairs); blocks past the actual count (tile_offset[E], known on the device only) return at once.
int64_t row_tiles(int64_t pairs, uint32_t num_experts) {
    return pairs / kTileRows + std::min<int64_t>(num_experts, pairs);
}

// One grouped GEMM launch of `gemm_mode` (0 dot64, 1 OMMA) over `tiles` row tiles with the epilogue.
template <bool kGlu>
bool launch_gemm(int gemm_mode, const GemmArgs &g, int64_t tiles, cudaStream_t stream) {
    const dim3 grid((g.n + kTileCols - 1) / kTileCols, static_cast<unsigned>(tiles));
    if (gemm_mode == 0) {
        grouped_gemm_kernel<kGlu><<<grid, kGemmThreads, 0, stream>>>(g);
    } else {
        grouped_gemm_omma_kernel<kGlu, false><<<grid, kGemmThreads, 0, stream>>>(g, RawOut{});
    }
    return launched();
}

// The block's per-pair intermediates, copied out after the block (cere_ktest_k7t_block_stages), all in sorted-row
// order: perm int32 [R] (sorted row -> pair t * K + k), h bf16 [R, I] (GEMM1 + SwiGLU, before the a2 quantization),
// h_codes uint8 [R, I/2] and h_scales uint8 [R, I/16] (the a2 quantization), y bf16 [R, H] (GEMM2, before combine).
struct Stages {
    void *perm;
    void *h;
    void *h_codes;
    void *h_scales;
    void *y;
};

// kernels::moe_block (k7_moe.cu) without taps, its grouped GEMMs run in `gemm_mode`; `stages` (nullable) receives
// the intermediates.
cere_status block(const MoeDims &d, const MoeLayer &L, const MoePlans &plans, const __nv_bfloat16 *x, uint32_t m,
                  __nv_bfloat16 *out, void *workspace, size_t workspace_bytes, int gemm_mode, const Stages *stages,
                  cudaStream_t stream) {
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
    // Router.
    if ((s = plans.router->run(x, L.router, m, w.logits, stream)) != CERE_OK) {
        return s;
    }
    if ((s = kernels::moe_route(w.logits, w.topk_ids, w.topk_weights, m, d.num_experts, d.top_k, stream)) !=
        CERE_OK) {
        return s;
    }
    // Expert grouping.
    if (cudaMemsetAsync(w.counts, 0, sizeof(int32_t) * d.num_experts, stream) != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    count_kernel<<<static_cast<unsigned>((pairs + kSortThreads - 1) / kSortThreads), kSortThreads, 0, stream>>>(
        w.topk_ids, w.counts, pairs);
    offsets_kernel<<<1, kSortThreads, 0, stream>>>(w.counts, w.expert_offset, w.tile_offset,
                                                   static_cast<int>(d.num_experts));
    place_kernel<<<d.num_experts, kSortThreads, 0, stream>>>(w.topk_ids, w.expert_offset, w.perm, w.pos, pairs);
    if (!launched()) {
        return CERE_ERR_INTERNAL;
    }
    // GEMM1 (gate and up with g1_alphas) + SwiGLU, then the a2 quantization.
    if ((s = kernels::fp4_quantize(x, L.a1_global_scale, w.x_codes, w.x_scales, m, d.hidden, stream)) != CERE_OK) {
        return s;
    }
    const int64_t tiles = row_tiles(pairs, d.num_experts);
    if (tiles > 65535) {
        return CERE_ERR_ARGUMENT;
    }
    const GemmArgs g1{w.x_codes,     w.x_scales,      w.perm,        static_cast<int32_t>(d.top_k),
                      L.gate_weight, L.gate_scale,    L.up_weight,   L.up_scale,
                      L.g1_alphas,   w.expert_offset, w.tile_offset, static_cast<int32_t>(d.num_experts),
                      static_cast<int32_t>(d.inter), static_cast<int32_t>(d.hidden), w.h};
    if (!launch_gemm<true>(gemm_mode, g1, tiles, stream)) {
        return CERE_ERR_INTERNAL;
    }
    if ((s = kernels::fp4_quantize(w.h, L.a2_global_scale, w.h_codes, w.h_scales, pairs, d.inter, stream)) !=
        CERE_OK) {
        return s;
    }
    // GEMM2 (down with g2_alphas).
    const GemmArgs g2{w.h_codes,     w.h_scales,      nullptr,       1,
                      L.down_weight, L.down_scale,    nullptr,       nullptr,
                      L.g2_alphas,   w.expert_offset, w.tile_offset, static_cast<int32_t>(d.num_experts),
                      static_cast<int32_t>(d.hidden), static_cast<int32_t>(d.inter), w.y};
    if (!launch_gemm<false>(gemm_mode, g2, tiles, stream)) {
        return CERE_ERR_INTERNAL;
    }
    // Shared expert (BF16 K2 GEMMs).
    if ((s = plans.shared_gate_up->run(x, L.shared_gate, m, w.s_gate, stream)) != CERE_OK ||
        (s = plans.shared_gate_up->run(x, L.shared_up, m, w.s_up, stream)) != CERE_OK) {
        return s;
    }
    const int64_t shared_count = int64_t{m} * d.shared_inter;
    silu_mul_kernel<<<static_cast<unsigned>((shared_count + 255) / 256), 256, 0, stream>>>(w.s_gate, w.s_up, w.s_h,
                                                                                          shared_count);
    if (!launched()) {
        return CERE_ERR_INTERNAL;
    }
    if ((s = plans.shared_down->run(w.s_h, L.shared_down, m, w.s_out, stream)) != CERE_OK) {
        return s;
    }
    // Combine.
    combine_kernel<<<m, kCombineThreads, 0, stream>>>(x, L.shared_expert_gate, w.y, w.pos, w.topk_weights, w.s_out,
                                                      out, static_cast<int>(d.hidden), static_cast<int>(d.top_k));
    if (!launched()) {
        return CERE_ERR_INTERNAL;
    }
    if (stages != nullptr) {
        const struct {
            void *dst;
            const void *src;
            size_t bytes;
        } copies[] = {{stages->perm, w.perm, sizeof(int32_t) * pairs},
                      {stages->h, w.h, sizeof(__nv_bfloat16) * pairs * d.inter},
                      {stages->h_codes, w.h_codes, size_t(pairs) * d.inter / 2},
                      {stages->h_scales, w.h_scales, size_t(pairs) * d.inter / 16},
                      {stages->y, w.y, sizeof(__nv_bfloat16) * pairs * d.hidden}};
        for (const auto &c : copies) {
            if (cudaMemcpyAsync(c.dst, c.src, c.bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                return CERE_ERR_INTERNAL;
            }
        }
    }
    return CERE_OK;
}

// cere::ktest::moe_block (k7_ktest.cpp) with `gemm_mode`: the same K2 plans, split-K workspace and block workspace.
cere_status block_entry(const MoeDims &dims, const MoeLayer &layer, const void *x, int64_t m, void *out,
                        int gemm_mode, const Stages *stages, cudaStream_t stream) {
    cublasLtHandle_t handle = gemm_handle();
    if (handle == nullptr) {
        return CERE_ERR_INTERNAL;
    }
    const uint32_t mm = static_cast<uint32_t>(m);
    kernels::GemmPlan router;
    kernels::GemmPlan shared_gate_up;
    kernels::GemmPlan shared_down;
    void *gemm_workspace = nullptr;
    cere_status s;
    if ((s = router.init(handle, dims.num_experts, dims.hidden, mm, kernels::GemmOut::kBF16)) != CERE_OK ||
        (s = plan_init(&shared_gate_up, handle, dims.shared_inter, dims.hidden, mm, kernels::GemmOut::kBF16,
                       &gemm_workspace, stream)) != CERE_OK) {
        return s;
    }
    const size_t bytes = dims_ok(dims) ? carve(dims, mm, nullptr).bytes : 0;
    void *workspace = nullptr;
    s = shared_down.init(handle, dims.hidden, dims.shared_inter, mm, kernels::GemmOut::kBF16);
    if (s == CERE_OK && bytes == 0) {
        s = CERE_ERR_ARGUMENT;
    }
    if (s == CERE_OK && cudaMallocAsync(&workspace, bytes, stream) != cudaSuccess) {
        (void)cudaGetLastError();  // clear it, so the next call's launched() does not report this failure
        s = CERE_ERR_CAPACITY;
    }
    if (s == CERE_OK) {
        const MoePlans plans{&router, &shared_gate_up, &shared_down};
        s = block(dims, layer, plans, static_cast<const __nv_bfloat16 *>(x), mm, static_cast<__nv_bfloat16 *>(out),
                  workspace, bytes, gemm_mode, stages, stream);
        if (cudaFreeAsync(workspace, stream) != cudaSuccess && s == CERE_OK) {
            s = CERE_ERR_INTERNAL;
        }
    }
    if (gemm_workspace != nullptr && cudaFreeAsync(gemm_workspace, stream) != cudaSuccess && s == CERE_OK) {
        s = CERE_ERR_INTERNAL;
    }
    return s;
}

// The OMMA grouped GEMM over caller-sorted rows, raw accumulators out.
cere_status gemm_entry(const void *a_codes, const void *a_scales, const void *expert_counts, const void *w0,
                       const void *s0, const void *w1, const void *s1, void *out0, void *out1, int64_t rows,
                       int64_t n, int64_t k, int64_t num_experts, cudaStream_t stream) {
    const bool glu = w1 != nullptr;
    if (!u32(rows) || !u32(n) || !u32(k) || !u32(num_experts) || num_experts > kMaxExperts || k % kChunk != 0 ||
        n % kTileCols != 0 || rows > INT32_MAX / 2 || (s1 != nullptr) != glu || (out1 != nullptr) != glu) {
        return CERE_ERR_ARGUMENT;
    }
    const void *ptrs[] = {a_codes, a_scales, expert_counts, w0, s0, out0, glu ? w1 : w0, glu ? s1 : s0,
                          glu ? out1 : out0};
    for (const void *p : ptrs) {
        if (p == nullptr || !aligned16(p)) {
            return CERE_ERR_ARGUMENT;
        }
    }
    const int64_t tiles = row_tiles(rows, static_cast<uint32_t>(num_experts));
    if (tiles > 65535) {
        return CERE_ERR_ARGUMENT;
    }
    int32_t *offsets = nullptr;  // expert_offset [E + 1] | tile_offset [E + 1]
    if (cudaMallocAsync(reinterpret_cast<void **>(&offsets), sizeof(int32_t) * 2 * (num_experts + 1), stream) !=
        cudaSuccess) {
        (void)cudaGetLastError();
        return CERE_ERR_CAPACITY;
    }
    const int e = static_cast<int>(num_experts);
    offsets_kernel<<<1, kSortThreads, 0, stream>>>(static_cast<const int32_t *>(expert_counts), offsets,
                                                   offsets + e + 1, e);
    if (!launched()) {  // never enqueue the GEMM over unwritten offsets [REVISED: M215R-07]
        (void)cudaFreeAsync(offsets, stream);
        return CERE_ERR_INTERNAL;
    }
    const GemmArgs g{static_cast<const uint8_t *>(a_codes), static_cast<const uint8_t *>(a_scales), nullptr, 1,
                     static_cast<const uint8_t *>(w0), static_cast<const uint8_t *>(s0),
                     static_cast<const uint8_t *>(w1), static_cast<const uint8_t *>(s1), nullptr, offsets,
                     offsets + e + 1, e, static_cast<int32_t>(n), static_cast<int32_t>(k), nullptr};
    const RawOut raw{static_cast<float *>(out0), static_cast<float *>(out1)};
    const dim3 grid(static_cast<unsigned>(n / kTileCols), static_cast<unsigned>(tiles));
    if (glu) {
        grouped_gemm_omma_kernel<true, true><<<grid, kGemmThreads, 0, stream>>>(g, raw);
    } else {
        grouped_gemm_omma_kernel<false, true><<<grid, kGemmThreads, 0, stream>>>(g, raw);
    }
    cere_status s = launched() ? CERE_OK : CERE_ERR_INTERNAL;
    if (cudaFreeAsync(offsets, stream) != cudaSuccess && s == CERE_OK) {
        s = CERE_ERR_INTERNAL;
    }
    return s;
}

// The extern "C" block entries' common body: argument checks and the K7 dims/layer.
cere_status block_c(const void *x, const void *router_weight, const void *gate_weight, const void *gate_scale,
                    const void *up_weight, const void *up_scale, const void *down_weight, const void *down_scale,
                    const void *g1_alphas, const void *g2_alphas, float a1_global_scale, float a2_global_scale,
                    const void *shared_gate_weight, const void *shared_up_weight, const void *shared_down_weight,
                    const void *shared_expert_gate, void *out, int64_t m, int64_t hidden, int64_t num_experts,
                    int64_t top_k, int64_t inter, int64_t shared_inter, int64_t gemm_mode, const Stages *stages,
                    cudaStream_t stream) {
    if (!u32(m) || !u32(hidden) || !u32(num_experts) || !u32(top_k) || !u32(inter) || !u32(shared_inter) ||
        (gemm_mode != 0 && gemm_mode != 1)) {
        return CERE_ERR_ARGUMENT;
    }
    if (stages != nullptr) {
        for (const void *p : {stages->perm, stages->h, stages->h_codes, stages->h_scales, stages->y}) {
            if (p == nullptr) {
                return CERE_ERR_ARGUMENT;
            }
        }
    }
    kernels::MoeDims dims;
    dims.hidden = static_cast<uint32_t>(hidden);
    dims.num_experts = static_cast<uint32_t>(num_experts);
    dims.top_k = static_cast<uint32_t>(top_k);
    dims.inter = static_cast<uint32_t>(inter);
    dims.shared_inter = static_cast<uint32_t>(shared_inter);
    kernels::MoeLayer layer;
    layer.router = static_cast<const __nv_bfloat16 *>(router_weight);
    layer.gate_weight = static_cast<const uint8_t *>(gate_weight);
    layer.gate_scale = static_cast<const uint8_t *>(gate_scale);
    layer.up_weight = static_cast<const uint8_t *>(up_weight);
    layer.up_scale = static_cast<const uint8_t *>(up_scale);
    layer.down_weight = static_cast<const uint8_t *>(down_weight);
    layer.down_scale = static_cast<const uint8_t *>(down_scale);
    layer.g1_alphas = static_cast<const float *>(g1_alphas);
    layer.g2_alphas = static_cast<const float *>(g2_alphas);
    layer.a1_global_scale = a1_global_scale;
    layer.a2_global_scale = a2_global_scale;
    layer.shared_gate = static_cast<const __nv_bfloat16 *>(shared_gate_weight);
    layer.shared_up = static_cast<const __nv_bfloat16 *>(shared_up_weight);
    layer.shared_down = static_cast<const __nv_bfloat16 *>(shared_down_weight);
    layer.shared_expert_gate = static_cast<const __nv_bfloat16 *>(shared_expert_gate);
    return block_entry(dims, layer, x, m, out, static_cast<int>(gemm_mode), stages, stream);
}

}  // namespace
}  // namespace cere::ktest::k7t

extern "C" __attribute__((visibility("default"))) int cere_ktest_k7t_gemm(
    const void *a_codes, const void *a_scales, const void *expert_counts, const void *w0, const void *s0,
    const void *w1, const void *s1, void *out0, void *out1, int64_t rows, int64_t n, int64_t k, int64_t num_experts,
    cudaStream_t stream) {
    return cere::ktest::k7t::gemm_entry(a_codes, a_scales, expert_counts, w0, s0, w1, s1, out0, out1, rows, n, k,
                                        num_experts, stream);
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_k7t_block(
    const void *x, const void *router_weight, const void *gate_weight, const void *gate_scale, const void *up_weight,
    const void *up_scale, const void *down_weight, const void *down_scale, const void *g1_alphas,
    const void *g2_alphas, float a1_global_scale, float a2_global_scale, const void *shared_gate_weight,
    const void *shared_up_weight, const void *shared_down_weight, const void *shared_expert_gate, void *out,
    int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter, int64_t shared_inter,
    int64_t gemm_mode, cudaStream_t stream) {
    return cere::ktest::k7t::block_c(x, router_weight, gate_weight, gate_scale, up_weight, up_scale, down_weight,
                                     down_scale, g1_alphas, g2_alphas, a1_global_scale, a2_global_scale,
                                     shared_gate_weight, shared_up_weight, shared_down_weight, shared_expert_gate,
                                     out, m, hidden, num_experts, top_k, inter, shared_inter, gemm_mode, nullptr,
                                     stream);
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_k7t_block_stages(
    const void *x, const void *router_weight, const void *gate_weight, const void *gate_scale, const void *up_weight,
    const void *up_scale, const void *down_weight, const void *down_scale, const void *g1_alphas,
    const void *g2_alphas, float a1_global_scale, float a2_global_scale, const void *shared_gate_weight,
    const void *shared_up_weight, const void *shared_down_weight, const void *shared_expert_gate, void *out,
    int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter, int64_t shared_inter,
    int64_t gemm_mode, void *perm, void *h, void *h_codes, void *h_scales, void *y, cudaStream_t stream) {
    const cere::ktest::k7t::Stages stages{perm, h, h_codes, h_scales, y};
    return cere::ktest::k7t::block_c(x, router_weight, gate_weight, gate_scale, up_weight, up_scale, down_weight,
                                     down_scale, g1_alphas, g2_alphas, a1_global_scale, a2_global_scale,
                                     shared_gate_weight, shared_up_weight, shared_down_weight, shared_expert_gate,
                                     out, m, hidden, num_experts, top_k, inter, shared_inter, gemm_mode, &stages,
                                     stream);
}
