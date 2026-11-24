// K13 (m29 design note D3, K2): the BF16 MoE block of the MTP draft layer. Contract in k13_moe_bf16.h.
//
// math_reference: sglang
// SGLang qwen4_exp_mtp.py (the MTP decoder layer's `mlp`, an unquantized `Qwen2MoeSparseMoeBlock`): the same router,
// shared expert and `fused_gate_sigmoid_mul_add` combine as K7 (kernels/k7_moe.cu cites them), with BF16 expert
// weights (`mlp.experts.gate_up_proj` [E, 2 I, H], gate rows first, and `mlp.experts.down_proj` [E, H, I]).
//
// Pieces: the router is K7's (BF16 logits from the K2 plan, then moe_route); the (token, slot) pairs are grouped by
// expert with K7's stable counting sort; the expert GEMMs are grouped BF16 GEMVs on CUDA cores, one warp per group of
// output columns and up to kTileRows rows of one expert: lane l accumulates its k elements {8 l + 256 i + j} in FP32
// (fmaf, i then j ascending), then a fixed xor-butterfly sums the 32 lanes. That order depends on neither the other
// rows of the tile nor m, so every output element is bit-equal whatever the batch composition. GEMM1 runs gate and
// up of a column in one warp and applies the epilogue BF16(silu(BF16(gate)) * BF16(up)) (FP32 SiLU); GEMM2 rounds
// the down projection to BF16. The combine and the shared expert are K7's.
#include "kernels/k13_moe_bf16.h"

#include <algorithm>
#include <cstdint>

#include <cuda_fp16.h>

#include "kernels/k4_nvfp4.h"

namespace cere::kernels {
namespace {

// -- workspace layout ---------------------------------------------------------------------------------------------

constexpr size_t kAlign = 256;
constexpr int kTileRows = 16;  // rows of one expert per GEMV block

size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

// Row tiles of r = m * K pairs: sum_e ceil(count_e / kTileRows) <= r / kTileRows + min(E, r).
uint64_t tiles_bound(uint64_t r, uint64_t num_experts) { return r / kTileRows + std::min(num_experts, r); }

struct Workspace {
    __nv_bfloat16 *logits;   // [m, E]
    int32_t *topk_ids;       // [m, K]
    float *topk_weights;     // [m, K]
    int32_t *counts;         // [E]
    int32_t *expert_offset;  // [E + 1] first sorted row of each expert
    int32_t *tile_offset;    // [E + 1] first row tile of each expert
    int4 *tile_info;         // [tiles_bound] row tile -> {expert, first sorted row, rows, 0}
    int32_t *perm;           // [R] sorted row -> pair index t * K + k
    int32_t *pos;            // [R] pair index -> sorted row
    __nv_bfloat16 *h;        // [R, I] SwiGLU output
    __nv_bfloat16 *y;        // [R, H] expert outputs (BF16)
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
    w.counts = reinterpret_cast<int32_t *>(take(uint64_t{d.num_experts} * 4));
    w.expert_offset = reinterpret_cast<int32_t *>(take((uint64_t{d.num_experts} + 1) * 4));
    w.tile_offset = reinterpret_cast<int32_t *>(take((uint64_t{d.num_experts} + 1) * 4));
    w.tile_info = reinterpret_cast<int4 *>(take(tiles_bound(r, d.num_experts) * sizeof(int4)));
    w.perm = reinterpret_cast<int32_t *>(take(r * 4));
    w.pos = reinterpret_cast<int32_t *>(take(r * 4));
    w.h = reinterpret_cast<__nv_bfloat16 *>(take(r * d.inter * 2));
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

// -- stable counting sort of the (token, slot) pairs by expert (K7's) ---------------------------------------------

constexpr int kSortThreads = 256;

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

// One block of kSortThreads: exclusive prefix sums of the expert counts and of their row tiles. Thread t owns the
// experts [t * per, (t + 1) * per); a warp shuffle scan and the warps' totals give each run's offsets (integer sums:
// the same offsets as a serial loop).
constexpr int kOffsetsPer = kMaxExperts / kSortThreads;

__global__ void __launch_bounds__(kSortThreads)
    offsets_kernel(const int32_t *__restrict__ counts, int32_t *__restrict__ expert_offset,
                   int32_t *__restrict__ tile_offset, int num_experts) {
    __shared__ int32_t warp_rows[kSortThreads / 32];
    __shared__ int32_t warp_tiles[kSortThreads / 32];
    const int per = (num_experts + kSortThreads - 1) / kSortThreads;
    const int e0 = threadIdx.x * per;
    int32_t c[kOffsetsPer];
    int32_t rows = 0;
    int32_t tiles = 0;
#pragma unroll
    for (int i = 0; i < kOffsetsPer; ++i) {
        c[i] = i < per && e0 + i < num_experts ? counts[e0 + i] : 0;
        rows += c[i];
        tiles += (c[i] + kTileRows - 1) / kTileRows;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int32_t rows_in = rows;  // inclusive scans over the warp's lanes
    int32_t tiles_in = tiles;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int32_t r = __shfl_up_sync(0xffffffffu, rows_in, o);
        const int32_t t = __shfl_up_sync(0xffffffffu, tiles_in, o);
        if (lane >= o) {
            rows_in += r;
            tiles_in += t;
        }
    }
    if (lane == 31) {
        warp_rows[warp] = rows_in;
        warp_tiles[warp] = tiles_in;
    }
    __syncthreads();
    int32_t row = rows_in - rows;
    int32_t tile = tiles_in - tiles;
    int32_t rows_total = 0;
    int32_t tiles_total = 0;
    for (int w = 0; w < kSortThreads / 32; ++w) {
        row += w < warp ? warp_rows[w] : 0;
        tile += w < warp ? warp_tiles[w] : 0;
        rows_total += warp_rows[w];
        tiles_total += warp_tiles[w];
    }
#pragma unroll
    for (int i = 0; i < kOffsetsPer; ++i) {
        if (i < per && e0 + i < num_experts) {
            expert_offset[e0 + i] = row;
            tile_offset[e0 + i] = tile;
            row += c[i];
            tile += (c[i] + kTileRows - 1) / kTileRows;
        }
    }
    if (threadIdx.x == 0) {
        expert_offset[num_experts] = rows_total;
        tile_offset[num_experts] = tiles_total;
    }
}

// One block per expert: its pairs in ascending pair order (stable), written to perm and pos, and its row tiles to
// tile_info (the GEMVs' tile -> expert and rows map).
__global__ void __launch_bounds__(kSortThreads)
    place_kernel(const int32_t *__restrict__ ids, const int32_t *__restrict__ expert_offset,
                 const int32_t *__restrict__ tile_offset, int32_t *__restrict__ perm, int32_t *__restrict__ pos,
                 int4 *__restrict__ tile_info, int64_t pairs) {
    __shared__ int32_t warp_counts[kSortThreads / 32];
    const int e = blockIdx.x;
    const int32_t first = expert_offset[e];
    const int32_t count = expert_offset[e + 1] - first;
    if (count == 0) {
        return;
    }
    const int32_t tile0 = tile_offset[e];
    for (int32_t t = threadIdx.x; t * kTileRows < count; t += kSortThreads) {
        tile_info[tile0 + t] = make_int4(e, first + t * kTileRows, min(kTileRows, count - t * kTileRows), 0);
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

// The small tail's grouping (pairs <= kTileRows * K: moe_route's top-k experts of a token are distinct, so an expert
// has at most kTileRows rows) in one block, thread t on expert t and pair t: the outputs the small GEMVs and the
// combine read value for value as count_kernel, offsets_kernel and place_kernel write them (perm, pos, tile_info and
// the tile count tile_offset[E]; rows of an expert in ascending pair order). Lets the GEMM1 launch go first
// (griddepcontrol.launch_dependents; GEMM1 waits for the whole grid before it reads the map).
constexpr int kGroupThreads = kMaxExperts;

__global__ void __launch_bounds__(kGroupThreads)
    small_group_kernel(const int32_t *__restrict__ ids, int32_t *__restrict__ tile_offset, int32_t *__restrict__ perm,
                       int32_t *__restrict__ pos, int4 *__restrict__ tile_info, int pairs, int num_experts) {
    __shared__ int32_t s_ids[kTileRows * kMaxTopK];
    __shared__ int32_t s_count[kMaxExperts];
    __shared__ int32_t s_warp[2][kGroupThreads / 32];
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int t = threadIdx.x;
    s_count[t] = 0;
    if (t < pairs) {
        const int32_t id = ids[t];  // an id outside [0, num_experts) (never written by moe_route) is skipped: -1
        s_ids[t] = static_cast<uint32_t>(id) < static_cast<uint32_t>(num_experts) ? id : -1;
    }
    __syncthreads();
    if (t < pairs && s_ids[t] >= 0) {
        atomicAdd(&s_count[s_ids[t]], 1);
    }
    __syncthreads();
    // Inclusive scans of the expert rows and row tiles (offsets_kernel's sums).
    const int32_t count = t < num_experts ? s_count[t] : 0;
    const int32_t own_tiles = (count + kTileRows - 1) / kTileRows;
    int32_t rows = count;
    int32_t tiles = own_tiles;
    const int lane = t & 31;
    const int warp = t >> 5;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int32_t r = __shfl_up_sync(0xffffffffu, rows, o);
        const int32_t q = __shfl_up_sync(0xffffffffu, tiles, o);
        if (lane >= o) {
            rows += r;
            tiles += q;
        }
    }
    if (lane == 31) {
        s_warp[0][warp] = rows;
        s_warp[1][warp] = tiles;
    }
    __syncthreads();
    if (warp == 0) {
        int32_t r = s_warp[0][lane];
        int32_t q = s_warp[1][lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int32_t r2 = __shfl_up_sync(0xffffffffu, r, o);
            const int32_t q2 = __shfl_up_sync(0xffffffffu, q, o);
            if (lane >= o) {
                r += r2;
                q += q2;
            }
        }
        s_warp[0][lane] = r;
        s_warp[1][lane] = q;
    }
    __syncthreads();
    if (warp > 0) {
        rows += s_warp[0][warp - 1];
        tiles += s_warp[1][warp - 1];
    }
    const int32_t first = rows - count;
    const int32_t tile0 = tiles - own_tiles;
    for (int32_t i = 0; i < own_tiles; ++i) {
        tile_info[tile0 + i] = make_int4(t, first + i * kTileRows, min(kTileRows, count - i * kTileRows), 0);
    }
    if (t == num_experts - 1) {
        tile_offset[num_experts] = tiles;
    }
    s_count[t] = first;  // every read of the counts is above the last barrier
    __syncthreads();
    if (t < pairs && s_ids[t] < 0) {
        pos[t] = 0;  // its row 0: the combine's read stays inside the expert rows
    } else if (t < pairs) {
        const int32_t e = s_ids[t];
        int32_t rank = 0;
        for (int q = 0; q < t; ++q) {
            rank += s_ids[q] == e ? 1 : 0;
        }
        const int32_t row = s_count[e] + rank;
        perm[row] = t;
        pos[t] = row;
    }
}

// -- BF16 grouped GEMV --------------------------------------------------------------------------------------------

constexpr int kGemvWarps = 4;
constexpr int kGemvThreads = 32 * kGemvWarps;
constexpr int kLaneElems = 8;                // one 16-byte load of BF16
constexpr int kWarpElems = 32 * kLaneElems;  // k elements per warp step

struct GemvArgs {
    const __nv_bfloat16 *a;   // activations [*, k]
    const int32_t *a_perm;    // sorted row r -> A row a_perm[r] / a_div; null: A row = r
    int32_t a_div;
    const __nv_bfloat16 *w;   // [E, n, k], or [E, 2 n, k] for GLU (gate rows 0..n-1, up rows n..2n-1)
    const int32_t *tile_offset;  // [E + 1]: tile_offset[E] row tiles
    const int4 *tile_info;       // place_kernel's tile map
    int32_t num_experts;
    int32_t n;  // output columns
    int32_t k;
    __nv_bfloat16 *out;  // [R, n]
};

// NVFP4 W4A16 expert weights (MoeBf16Layer gate_up_q ..., track Fp4S3): packed E2M1 [E, rows, k / 2], E4M3 block
// scales [E, rows, k / 16], F32 globals [E, kGlu ? 2 : 1] (GLU: gate, up).
struct Nvfp4Weights {
    const uint8_t *q;
    const uint8_t *scale;
    const float *global;
};

// Eight E2M1 codes of one 16-block (byte j of `codes`: element 2j in the low nibble, 2j + 1 in the high) -> the BF16
// values cerec convert's S2 path stores for them, as F32: BF16_RNE(RN(RN(e2m1 * scale) * global)) (convert.rs
// nvfp4_to_bf16; the first product is exact). cvt e2m1x2 -> f16x2 is exact; __fmul_rn keeps both products unfused.
__device__ __forceinline__ void nvfp4x8_to_float(uint32_t codes, float scale, float global,
                                                 float (&f)[kLaneElems]) {
    uint32_t h[4];
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(codes));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float lo = __half2float(__ushort_as_half(static_cast<unsigned short>(h[i] & 0xFFFFu)));
        const float hi = __half2float(__ushort_as_half(static_cast<unsigned short>(h[i] >> 16)));
        f[2 * i] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(lo, scale), global)));
        f[2 * i + 1] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(hi, scale), global)));
    }
}

__device__ __forceinline__ void bf16x8_to_float(const uint4 v, float (&f)[kLaneElems]) {
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i] = __uint_as_float(words[i] << 16);              // element 2i: the low half
        f[2 * i + 1] = __uint_as_float(words[i] & 0xFFFF0000u);  // element 2i + 1: the high half
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

// Block (column block x, row tile y): warp w computes kCols output columns starting at
// (x * kGemvWarps + w) * kCols for the tile's rows (all of one expert). kGlu: each column needs its gate and up
// weight rows; the output is BF16(silu(BF16(gate)) * BF16(up)) with SiLU in FP32. kFp4: the weights are `f` (g.w
// unused), each lane's 8 elements decoded to the BF16 values the BF16 kernel reads (nvfp4x8_to_float), then the same
// accumulation.
template <bool kGlu, int kCols, bool kFp4>
__device__ __forceinline__ void grouped_gemv(const GemvArgs &g, const Nvfp4Weights &f) {
    __shared__ int64_t a_row[kTileRows];

    const int tile = blockIdx.y;
    const int32_t tiles = g.tile_offset[g.num_experts];
    const int4 info = g.tile_info[tile];  // read before the bound check (one round trip); used only below it
    if (tile >= tiles) {
        return;
    }
    const int e = info.x;
    const int32_t row0 = info.y;
    const int rows = info.z;
    if (threadIdx.x < kTileRows) {
        const int r = threadIdx.x;
        int64_t src = -1;
        if (r < rows) {
            src = g.a_perm != nullptr ? g.a_perm[row0 + r] / g.a_div : row0 + r;
        }
        a_row[r] = src;
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int col0 = (blockIdx.x * kGemvWarps + (threadIdx.x >> 5)) * kCols;
    if (col0 >= g.n) {
        return;
    }
    constexpr int kW = kGlu ? 2 * kCols : kCols;
    const int64_t expert_rows = kGlu ? 2 * int64_t{g.n} : int64_t{g.n};
    const __nv_bfloat16 *w_row[kW];
    const uint8_t *q_row[kW];
    const uint8_t *s_row[kW];
    float global[kW];
#pragma unroll
    for (int c = 0; c < kW; ++c) {
        // Weight row of slot c: gate (or down) row col0 + c, GLU up rows n + col0 + c - kCols.
        const int64_t row = c < kCols ? col0 + c : g.n + col0 + c - kCols;
        if constexpr (kFp4) {
            const int64_t at = static_cast<int64_t>(e) * expert_rows + row;
            q_row[c] = f.q + at * (g.k / 2);
            s_row[c] = f.scale + at * (g.k / 16);
            global[c] = kGlu ? f.global[2 * e + (c < kCols ? 0 : 1)] : f.global[e];
        } else {
            w_row[c] = g.w + (static_cast<int64_t>(e) * expert_rows + row) * g.k;
        }
    }

    float acc[kTileRows][kW];
#pragma unroll
    for (int r = 0; r < kTileRows; ++r) {
#pragma unroll
        for (int c = 0; c < kW; ++c) {
            acc[r][c] = 0.0f;
        }
    }
    for (int k0 = lane * kLaneElems; k0 < g.k; k0 += kWarpElems) {
        float wf[kW][kLaneElems];
#pragma unroll
        for (int c = 0; c < kW; ++c) {
            if constexpr (kFp4) {
                nvfp4x8_to_float(__ldg(reinterpret_cast<const uint32_t *>(q_row[c] + k0 / 2)),
                                 ::cere::k4::nvfp4::e4m3(__ldg(s_row[c] + k0 / 16)), global[c], wf[c]);
            } else {
                bf16x8_to_float(__ldg(reinterpret_cast<const uint4 *>(w_row[c] + k0)), wf[c]);
            }
        }
#pragma unroll
        for (int r = 0; r < kTileRows; ++r) {
            if (r < rows) {
                float xf[kLaneElems];
                bf16x8_to_float(__ldg(reinterpret_cast<const uint4 *>(g.a + a_row[r] * g.k + k0)), xf);
#pragma unroll
                for (int c = 0; c < kW; ++c) {
#pragma unroll
                    for (int j = 0; j < kLaneElems; ++j) {
                        acc[r][c] = fmaf(xf[j], wf[c][j], acc[r][c]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < kTileRows; ++r) {
        if (r < rows) {
#pragma unroll
            for (int c = 0; c < kW; ++c) {
                acc[r][c] = warp_sum(acc[r][c]);
            }
        }
    }
    // Every lane holds every sum; lane (r * kCols + c) % 32 stores output (r, c).
#pragma unroll
    for (int r = 0; r < kTileRows; ++r) {
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            if (r < rows && lane == (r * kCols + c) % 32) {
                float result;
                if constexpr (kGlu) {
                    const float gate = __bfloat162float(__float2bfloat16(acc[r][c]));
                    const float up = __bfloat162float(__float2bfloat16(acc[r][kCols + c]));
                    result = (gate / (1.0f + expf(-gate))) * up;  // SiLU(gate) * up in FP32
                } else {
                    result = acc[r][c];
                }
                g.out[static_cast<int64_t>(row0 + r) * g.n + col0 + c] = __float2bfloat16(result);
            }
        }
    }
}

// The kernels: one name, overloaded on the weights (BF16 g.w, or NVFP4 `f`).
template <bool kGlu, int kCols>
__global__ void __launch_bounds__(kGemvThreads) grouped_gemv_kernel(const GemvArgs g) {
    grouped_gemv<kGlu, kCols, false>(g, Nvfp4Weights{});
}

template <bool kGlu, int kCols>
__global__ void __launch_bounds__(kGemvThreads) grouped_gemv_kernel(const GemvArgs g, const Nvfp4Weights f) {
    grouped_gemv<kGlu, kCols, true>(g, f);
}

// -- small-tail NVFP4 grouped GEMV (tail <= kTileRows tokens: the MTP draft's calls) --------------------------------
//
// grouped_gemv's arithmetic with the loads reordered. With at most kTileRows tokens every expert has at most
// kTileRows rows, so each expert is one tile. A warp first issues every load of its weight rows: the packed codes for
// the whole k (kIters 32-bit loads per row and lane, held in registers) and the scale row (32-bit words, lane l holding
// words l and l + 32, handed to the lanes that need them by shuffles), so all of a block's DRAM reads are in flight
// at once instead of one k step at a time. Then, per group of kGroup rows (the codes stay in registers across
// groups): lane l accumulates its elements {8 l + 256 i + j} with fmaf, i then j ascending, per row and column,
// warp_sum, and grouped_gemv's epilogue, so every output element is bit-equal to grouped_gemv's.
//
// Launch: both GEMMs are programmatic dependent launches (griddepcontrol). kMapFirst (GEMM1, dependent of
// small_group_kernel): the CTA waits for the grouping (griddepcontrol.wait) before it reads the tile map, then lets
// GEMM2 launch (griddepcontrol.launch_dependents). Else (GEMM2, dependent of GEMM1): the CTA lets the stream's next
// kernel launch, reads its tile and issues its weight loads (the grouping finished before any GEMM1 CTA let GEMM2
// launch; the weights are constants), then waits for GEMM1 before it reads the activations or writes. A launch
// without the attribute makes the wait a no-op.
constexpr int kSmallGroup = 1;
constexpr int kSmallGluIters = 10;  // GEMM1: k = hidden <= 2,560
constexpr int kSmallDownIters = 3;  // GEMM2: k = inter <= 768
constexpr int kSmallDownCols = 8;   // GEMM2: 8 columns per warp

template <bool kGlu, int kCols, int kIters, int kGroup, bool kMapFirst>
__global__ void __launch_bounds__(kGemvThreads) small_gemv_kernel(const GemvArgs g, const Nvfp4Weights f) {
    static_assert(kIters * kWarpElems <= 64 * 64, "scale rows of at most 64 words");
    if constexpr (kMapFirst) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int tile = blockIdx.y;
    const int32_t tiles = g.tile_offset[g.num_experts];
    const int4 info = g.tile_info[tile];  // read before the bound check (one round trip); used only below it
    if (tile >= tiles) {
        return;
    }
    const int e = info.x;
    const int32_t row0 = info.y;
    const int rows = info.z;
    const int lane = threadIdx.x & 31;
    const int col0 = (blockIdx.x * kGemvWarps + (threadIdx.x >> 5)) * kCols;
    if (col0 >= g.n) {
        return;
    }
    constexpr int kW = kGlu ? 2 * kCols : kCols;
    const int64_t expert_rows = kGlu ? 2 * int64_t{g.n} : int64_t{g.n};
    const int words = g.k / 64;  // 32-bit words of one scale row (k / 16 bytes)
    uint32_t q[kW][kIters];
    uint32_t sw[kW][2];
    float global[kW];
#pragma unroll
    for (int c = 0; c < kW; ++c) {
        // Weight row of slot c: gate (or down) row col0 + c, GLU up rows n + col0 + c - kCols.
        const int64_t row = c < kCols ? col0 + c : g.n + col0 + c - kCols;
        const int64_t at = static_cast<int64_t>(e) * expert_rows + row;
        const uint32_t *q_row = reinterpret_cast<const uint32_t *>(f.q + at * (g.k / 2));
        const uint32_t *s_row = reinterpret_cast<const uint32_t *>(f.scale + at * (g.k / 16));
#pragma unroll
        for (int i = 0; i < kIters; ++i) {
            q[c][i] = lane * kLaneElems + i * kWarpElems < g.k ? __ldg(q_row + lane + 32 * i) : 0u;
        }
        sw[c][0] = lane < words ? __ldg(s_row + lane) : 0u;
        sw[c][1] = lane + 32 < words ? __ldg(s_row + lane + 32) : 0u;
        global[c] = kGlu ? f.global[2 * e + (c < kCols ? 0 : 1)] : f.global[e];
    }
    int32_t src = 0;  // lane r < rows: the A row of tile row r
    if (lane < rows) {
        src = g.a_perm != nullptr ? g.a_perm[row0 + lane] / g.a_div : row0 + lane;
    }
    if constexpr (!kMapFirst) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    for (int r0 = 0; r0 < rows; r0 += kGroup) {
        int64_t a_row[kGroup];
#pragma unroll
        for (int r = 0; r < kGroup; ++r) {
            a_row[r] = __shfl_sync(0xffffffffu, src, r0 + r);
        }
        float acc[kGroup][kW];
#pragma unroll
        for (int r = 0; r < kGroup; ++r) {
#pragma unroll
            for (int c = 0; c < kW; ++c) {
                acc[r][c] = 0.0f;
            }
        }
#pragma unroll
        for (int i = 0; i < kIters; ++i) {
            const int k0 = lane * kLaneElems + i * kWarpElems;
            // Scale byte k0 / 16 = lane / 2 + 16 i of the row: word lane / 8 + 4 i (held by lane (lane / 8 + 4 i) % 32,
            // in sw[.][1] from word 32), byte (lane / 2) % 4.
            uint32_t scale_word[kW];
#pragma unroll
            for (int c = 0; c < kW; ++c) {
                scale_word[c] = __shfl_sync(0xffffffffu, sw[c][i >= 8 ? 1 : 0], (lane / 8 + 4 * i) % 32);
            }
            if (k0 < g.k) {
                uint4 xv[kGroup];
#pragma unroll
                for (int r = 0; r < kGroup; ++r) {
                    if (r0 + r < rows) {
                        xv[r] = __ldg(reinterpret_cast<const uint4 *>(g.a + a_row[r] * g.k + k0));
                    }
                }
#pragma unroll
                for (int c = 0; c < kW; ++c) {
                    float wf[kLaneElems];
                    const uint32_t scale_byte = (scale_word[c] >> (8 * ((lane / 2) % 4))) & 0xFFu;
                    nvfp4x8_to_float(q[c][i], ::cere::k4::nvfp4::e4m3(scale_byte), global[c], wf);
#pragma unroll
                    for (int r = 0; r < kGroup; ++r) {
                        if (r0 + r < rows) {
                            float xf[kLaneElems];
                            bf16x8_to_float(xv[r], xf);
#pragma unroll
                            for (int j = 0; j < kLaneElems; ++j) {
                                acc[r][c] = fmaf(xf[j], wf[j], acc[r][c]);
                            }
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < kGroup; ++r) {
            if (r0 + r < rows) {
#pragma unroll
                for (int c = 0; c < kW; ++c) {
                    acc[r][c] = warp_sum(acc[r][c]);
                }
            }
        }
        // Every lane holds every sum; lane (r * kCols + c) % 32 stores output (r, c).
#pragma unroll
        for (int r = 0; r < kGroup; ++r) {
#pragma unroll
            for (int c = 0; c < kCols; ++c) {
                if (r0 + r < rows && lane == (r * kCols + c) % 32) {
                    float result;
                    if constexpr (kGlu) {
                        const float gate = __bfloat162float(__float2bfloat16(acc[r][c]));
                        const float up = __bfloat162float(__float2bfloat16(acc[r][kCols + c]));
                        result = (gate / (1.0f + expf(-gate))) * up;  // SiLU(gate) * up in FP32
                    } else {
                        result = acc[r][c];
                    }
                    g.out[static_cast<int64_t>(row0 + r0 + r) * g.n + col0 + c] = __float2bfloat16(result);
                }
            }
        }
    }
}

constexpr int kGluCols = 2;   // GEMM1: 2 columns (4 weight rows) per warp
constexpr int kDownCols = 4;  // GEMM2: 4 columns per warp

// -- shared expert and combine (K7's) -----------------------------------------------------------------------------

__global__ void silu_mul_kernel(const __nv_bfloat16 *__restrict__ gate, const __nv_bfloat16 *__restrict__ up,
                                __nv_bfloat16 *__restrict__ out, int64_t count) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float g = __bfloat162float(gate[i]);
        out[i] = __float2bfloat16((g / (1.0f + expf(-g))) * __bfloat162float(up[i]));
    }
}

constexpr int kCombineThreads = 128;

// Block (t, c): token t's columns c * kCombineThreads + threadIdx.x (stride gridDim.y * kCombineThreads): routed =
// BF16(sum_k w_k * y_k) in slot order, then out = routed + sigmoid(x . g) * shared. Every block of the token computes
// the same x . g (one fixed reduction order), so the split changes no bit.
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
    for (int j = blockIdx.y * kCombineThreads + threadIdx.x; j < hidden; j += gridDim.y * kCombineThreads) {
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

bool aligned4(const void *p) { return reinterpret_cast<uintptr_t>(p) % 4 == 0; }

cudaError_t launch_dependent(const cudaLaunchConfig_t &base, void (*kernel)(GemvArgs, Nvfp4Weights), const GemvArgs &g,
                             const Nvfp4Weights &f) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config = base;
    config.attrs = &attr;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel, g, f);
}

// The small tail's routed experts on the pairs of w.topk_ids: small_group_kernel, then GEMM1 (gate and up) + SwiGLU
// into h, then GEMM2 (down) into y, each GEMM a programmatic dependent launch of the kernel before it, on `tiles`
// row tiles (CTAs past the tile count exit).
template <int kG1Cols, int kG2Cols, int kGroup>
cere_status small_routed(const MoeDims &d, const MoeBf16Layer &L, const Workspace &w, const __nv_bfloat16 *x_tail,
                         int64_t pairs, int64_t tiles, cudaStream_t stream) {
    small_group_kernel<<<1, kGroupThreads, 0, stream>>>(w.topk_ids, w.tile_offset, w.perm, w.pos, w.tile_info,
                                                        static_cast<int>(pairs), static_cast<int>(d.num_experts));
    const GemvArgs g1{x_tail,    w.perm, static_cast<int32_t>(d.top_k), L.gate_up, w.tile_offset, w.tile_info,
                      static_cast<int32_t>(d.num_experts), static_cast<int32_t>(d.inter),
                      static_cast<int32_t>(d.hidden), w.h};
    const GemvArgs g2{w.h,       nullptr, 1, L.down, w.tile_offset, w.tile_info,
                      static_cast<int32_t>(d.num_experts), static_cast<int32_t>(d.hidden),
                      static_cast<int32_t>(d.inter), w.y};
    constexpr unsigned kG1BlockCols = kGemvWarps * kG1Cols;
    constexpr unsigned kG2BlockCols = kGemvWarps * kG2Cols;
    cudaLaunchConfig_t config{};
    config.blockDim = dim3(kGemvThreads);
    config.stream = stream;
    config.gridDim = dim3((d.inter + kG1BlockCols - 1) / kG1BlockCols, static_cast<unsigned>(tiles));
    if (launch_dependent(config, small_gemv_kernel<true, kG1Cols, kSmallGluIters, kGroup, true>, g1,
                         Nvfp4Weights{L.gate_up_q, L.gate_up_scale, L.gate_up_global}) != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    config.gridDim = dim3((d.hidden + kG2BlockCols - 1) / kG2BlockCols, static_cast<unsigned>(tiles));
    if (launch_dependent(config, small_gemv_kernel<false, kG2Cols, kSmallDownIters, kGroup, false>, g2,
                         Nvfp4Weights{L.down_q, L.down_scale, L.down_global}) != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    return launched() ? CERE_OK : CERE_ERR_INTERNAL;
}

// The routed experts' grouping and GEMVs on the pairs of w.topk_ids (`pairs` = tail * K; `tiles` its row tiles
// bound): h, then y. small_routed serves tail <= kTileRows tokens of NVFP4 experts within its k bounds;
// grouped_gemv everything else.
cere_status routed_gemvs(const MoeDims &d, const MoeBf16Layer &L, const Workspace &w, const __nv_bfloat16 *x_tail,
                         int64_t pairs, int64_t tiles, cudaStream_t stream) {
    const bool fp4 = L.gate_up_q != nullptr;
    if (fp4 && pairs <= int64_t{kTileRows} * d.top_k && d.hidden <= kSmallGluIters * kWarpElems &&
        d.inter <= kSmallDownIters * kWarpElems && aligned4(L.gate_up_scale) && aligned4(L.down_scale)) {
        return small_routed<kGluCols, kSmallDownCols, kSmallGroup>(d, L, w, x_tail, pairs, tiles, stream);
    }
    if (cudaMemsetAsync(w.counts, 0, sizeof(int32_t) * d.num_experts, stream) != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    count_kernel<<<static_cast<unsigned>((pairs + kSortThreads - 1) / kSortThreads), kSortThreads, 0, stream>>>(
        w.topk_ids, w.counts, w.pos, pairs, static_cast<int>(d.num_experts));
    offsets_kernel<<<1, kSortThreads, 0, stream>>>(w.counts, w.expert_offset, w.tile_offset,
                                                   static_cast<int>(d.num_experts));
    place_kernel<<<d.num_experts, kSortThreads, 0, stream>>>(w.topk_ids, w.expert_offset, w.tile_offset, w.perm,
                                                             w.pos, w.tile_info, pairs);
    if (!launched()) {
        return CERE_ERR_INTERNAL;
    }
    const GemvArgs g1{x_tail,    w.perm, static_cast<int32_t>(d.top_k), L.gate_up, w.tile_offset, w.tile_info,
                      static_cast<int32_t>(d.num_experts), static_cast<int32_t>(d.inter),
                      static_cast<int32_t>(d.hidden), w.h};
    const GemvArgs g2{w.h,       nullptr, 1, L.down, w.tile_offset, w.tile_info,
                      static_cast<int32_t>(d.num_experts), static_cast<int32_t>(d.hidden),
                      static_cast<int32_t>(d.inter), w.y};
    // GEMM1 (gate and up) + SwiGLU.
    constexpr unsigned kGluBlockCols = kGemvWarps * kGluCols;
    const dim3 grid1((d.inter + kGluBlockCols - 1) / kGluBlockCols, static_cast<unsigned>(tiles));
    if (fp4) {
        grouped_gemv_kernel<true, kGluCols><<<grid1, kGemvThreads, 0, stream>>>(
            g1, Nvfp4Weights{L.gate_up_q, L.gate_up_scale, L.gate_up_global});
    } else {
        grouped_gemv_kernel<true, kGluCols><<<grid1, kGemvThreads, 0, stream>>>(g1);
    }
    // GEMM2 (down).
    constexpr unsigned kDownBlockCols = kGemvWarps * kDownCols;
    const dim3 grid2((d.hidden + kDownBlockCols - 1) / kDownBlockCols, static_cast<unsigned>(tiles));
    if (fp4) {
        grouped_gemv_kernel<false, kDownCols><<<grid2, kGemvThreads, 0, stream>>>(
            g2, Nvfp4Weights{L.down_q, L.down_scale, L.down_global});
    } else {
        grouped_gemv_kernel<false, kDownCols><<<grid2, kGemvThreads, 0, stream>>>(g2);
    }
    return launched() ? CERE_OK : CERE_ERR_INTERNAL;
}

}  // namespace

size_t moe_bf16_workspace_bytes(const MoeDims &dims, uint32_t max_m) {
    if (!dims_ok(dims) || max_m == 0) {
        return 0;
    }
    return carve(dims, max_m, nullptr).bytes;
}

cere_status moe_bf16_block(const MoeDims &d, const MoeBf16Layer &L, const MoePlans &plans, const __nv_bfloat16 *x,
                           uint32_t m, uint32_t first, __nv_bfloat16 *out, void *workspace, size_t workspace_bytes,
                           cudaStream_t stream) {
    if (!dims_ok(d) || m == 0 || first >= m || x == nullptr || out == nullptr || workspace == nullptr ||
        reinterpret_cast<uintptr_t>(workspace) % kAlign != 0 || plans.router == nullptr ||
        plans.shared_gate_up == nullptr || plans.shared_down == nullptr || !aligned16(x)) {
        return CERE_ERR_ARGUMENT;
    }
    // Routed experts: BF16 gate_up/down, or (gate_up_q set) the NVFP4 W4A16 set with gate_up/down null. Packed rows
    // are read 4 bytes per lane (H / 2, I / 2 bytes per row: dims_ok's multiples of 128 keep them 4-byte aligned).
    const bool fp4 = L.gate_up_q != nullptr;
    const void *weights[] = {L.router,        L.shared_gate, L.shared_up, L.shared_down, L.shared_expert_gate,
                             fp4 ? L.gate_up_q : static_cast<const void *>(L.gate_up),
                             fp4 ? L.down_q : static_cast<const void *>(L.down)};
    for (const void *p : weights) {
        if (p == nullptr || !aligned16(p)) {
            return CERE_ERR_ARGUMENT;
        }
    }
    if (fp4 && (L.gate_up != nullptr || L.down != nullptr || L.gate_up_scale == nullptr || L.down_scale == nullptr ||
                L.gate_up_global == nullptr || L.down_global == nullptr ||
                reinterpret_cast<uintptr_t>(L.gate_up_global) % alignof(float) != 0 ||
                reinterpret_cast<uintptr_t>(L.down_global) % alignof(float) != 0)) {
        return CERE_ERR_ARGUMENT;
    }
    if (!fp4 && (L.gate_up_scale != nullptr || L.gate_up_global != nullptr || L.down_q != nullptr ||
                 L.down_scale != nullptr || L.down_global != nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    const Workspace w = carve(d, m, static_cast<uint8_t *>(workspace));
    if (w.bytes > workspace_bytes) {
        return CERE_ERR_ARGUMENT;
    }
    // The routed rows [first, m): routing, grouping, expert GEMVs and the combine see the tail alone (every one of
    // them is per-row bit-equal at any row count); the router and the shared expert GEMMs keep all m rows.
    const uint32_t tail = m - first;
    const __nv_bfloat16 *x_tail = x + uint64_t{first} * d.hidden;
    const int64_t pairs = int64_t{tail} * d.top_k;
    // Row tiles: blocks past the actual count (tile_offset[E], known on the device only) return at once.
    const int64_t tiles = static_cast<int64_t>(tiles_bound(static_cast<uint64_t>(pairs), d.num_experts));
    if (pairs > INT32_MAX / 2 || tiles > 65535) {
        return CERE_ERR_ARGUMENT;
    }
    cere_status s;

    // Router.
    if ((s = plans.router->run(x, L.router, m, w.logits, stream, plans.cls)) != CERE_OK) {
        return s;
    }
    if ((s = moe_route(w.logits + uint64_t{first} * d.num_experts, w.topk_ids, w.topk_weights, tail, d.num_experts,
                       d.top_k, stream)) != CERE_OK) {
        return s;
    }
    // Expert grouping, GEMM1 (gate and up) + SwiGLU, GEMM2 (down).
    if ((s = routed_gemvs(d, L, w, x_tail, pairs, tiles, stream)) != CERE_OK) {
        return s;
    }

    // Shared expert (BF16 K2 GEMMs).
    if ((s = plans.shared_gate_up->run(x, L.shared_gate, m, w.s_gate, stream, plans.cls)) != CERE_OK ||
        (s = plans.shared_gate_up->run(x, L.shared_up, m, w.s_up, stream, plans.cls)) != CERE_OK) {
        return s;
    }
    const int64_t shared_count = int64_t{m} * d.shared_inter;
    silu_mul_kernel<<<static_cast<unsigned>((shared_count + 255) / 256), 256, 0, stream>>>(w.s_gate, w.s_up, w.s_h,
                                                                                          shared_count);
    if (!launched()) {
        return CERE_ERR_INTERNAL;
    }
    if ((s = plans.shared_down->run(w.s_h, L.shared_down, m, w.s_out, stream, plans.cls)) != CERE_OK) {
        return s;
    }
    // Combine: one block per token and kCombineThreads columns (dims_ok: hidden a multiple of 128).
    const dim3 combine_grid(tail, static_cast<unsigned>(d.hidden / kCombineThreads));
    combine_kernel<<<combine_grid, kCombineThreads, 0, stream>>>(
        x_tail, L.shared_expert_gate, w.y, w.pos, w.topk_weights, w.s_out + uint64_t{first} * d.hidden,
        out + uint64_t{first} * d.hidden, static_cast<int>(d.hidden), static_cast<int>(d.top_k));
    return launched() ? CERE_OK : CERE_ERR_INTERNAL;
}

}  // namespace cere::kernels
