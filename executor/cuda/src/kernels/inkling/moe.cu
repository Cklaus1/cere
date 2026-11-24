// Inkling 2 Mini S4: the routed-MoE kernels (contracts in moe.h; arithmetic = executor/ref/inkling/{router,quant,
// moe}_ref.h, G1). Every float operation is a ref:: primitive (IEEE RN, fused only where written ref::fma) or an
// exact operation (integer dot products of 2 * e2m1 codes, exact conversions); the GEMMs compute each output element
// in one thread in block order, so no output depends on the launch shape or on other tokens.
#include "kernels/inkling/moe.h"
#include "kernels/inkling/pdl.cuh"

#include <cooperative_groups.h>
#include <cuda_fp16.h>

#include "inkling/moe_ref.h"
#include "inkling/quant_ref.h"
#include "inkling/router_ref.h"

namespace cere::inkling {
namespace {

namespace cg = cooperative_groups;

constexpr int H = kMoeHidden;
constexpr int I = kMoeInter;

__device__ __forceinline__ float bf(const bf16 v) { return ref::decode_bf16(__bfloat16_as_ushort(v)); }
__device__ __forceinline__ bf16 to_bf(float v) { return __ushort_as_bfloat16(ref::encode_bf16(v)); }

// Exact E4M3 pair -> two floats (cvt to f16x2 is exact for every non-NaN E4M3 value; f16 -> f32 exact).
__device__ __forceinline__ void e4m3x2_to_float(uint16_t pair, float &lo, float &hi) {
    uint32_t halves;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(halves) : "h"(pair));
    lo = __half2float(__ushort_as_half(static_cast<unsigned short>(halves & 0xFFFFu)));
    hi = __half2float(__ushort_as_half(static_cast<unsigned short>(halves >> 16)));
}

// Four E2M1 codes (bits 4i..4i+3 of the low 16 bits) -> four int8 2 * e2m1 (byte i), exact (k0_nvfp4.cuh's map).
__device__ __forceinline__ uint32_t e2m1x4_to_s8x4(uint32_t v) {
    const uint32_t index = v & 0x7777u;
    const uint32_t positive = __byte_perm(0x03020100u, 0x0C080604u, index);
    const uint32_t negative = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, index);
    const uint32_t spread = __byte_perm(v & 0x0F0Fu, (v >> 4) & 0x0F0Fu, 0x5140u);
    const uint32_t sign = ((spread >> 3) & 0x01010101u) * 0xFFu;
    return (positive & ~sign) | (negative & sign);
}

// The magnitudes 2 * |e2m1| (int8, byte i) of the codes in selector nibbles 0..3 (bits 0-2 magnitude, bit 3 sign)
// that are positive; a negative code gives 0 (prmt.b32's default mode: a selector nibble with its msb set replicates
// the selected byte's sign, and every table byte is positive). With the sign bits flipped (v ^ 0x8888) it gives the
// negative codes' magnitudes, so a dp4a of each against the same activations differ by the signed sum.
__device__ __forceinline__ uint32_t e2m1x4_mag(uint32_t sel) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(0x03020100u), "r"(0x0C080604u), "r"(sel));
    return d;
}

bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// -- router --------------------------------------------------------------------------------------------------------
// One thread block cluster of kRouterCluster blocks per row, one warp per expert; rank 0's warp 0 gathers the 128
// logits over distributed shared memory and selects.

constexpr int kRouterCluster = 8;
constexpr int kRouterWarps = kMoeExperts / kRouterCluster;  // 16 experts per block
constexpr int kRouterThreads = 32 * kRouterWarps;

__global__ void __cluster_dims__(kRouterCluster, 1, 1) __launch_bounds__(kRouterThreads)
    router_kernel(const bf16 *__restrict__ x, int64_t x_stride, const bf16 *__restrict__ w,
                  const float *__restrict__ bias, int32_t *ids, float *weights, float *logits_out, int mutant) {
    pdl::trigger();
    __shared__ float xs[H];
    __shared__ float part[kRouterWarps];  // this block's experts' logits
    __shared__ float logit[kMoeExperts];  // rank 0: all of them
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = int(cluster.block_rank());
    const int t = blockIdx.y;
    pdl::wait();
    for (int i = threadIdx.x; i < H; i += kRouterThreads) xs[i] = bf(x[int64_t(t) * x_stride + i]);
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    // ref::reduce_dot<256>: lane l of the 256 accumulates elements l, l + 256, ..; warp lane `lane` holds lanes
    // lane + 32 s (s = 0..7); halving levels 128, 64, 32 in registers, 16..1 by xor shuffles.
    {
        const int e = rank * kRouterWarps + warp;
        const bf16 *we = w + int64_t(e) * H;
        float acc[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            float a = 0.0f;
#pragma unroll
            for (int q = 0; q < H / 256; ++q) {
                const int i = lane + 32 * s + 256 * q;
                a = ref::fma(xs[i], bf(we[i]), a);
            }
            acc[s] = a;
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) acc[s] = ref::add(acc[s], acc[s + 4]);
#pragma unroll
        for (int s = 0; s < 2; ++s) acc[s] = ref::add(acc[s], acc[s + 2]);
        float v = ref::add(acc[0], acc[1]);
#pragma unroll
        for (int off = 16; off >= 1; off /= 2) v = ref::add(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
        if (lane == 0) part[warp] = v;
    }
    cluster.sync();
    if (rank == 0 && warp == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int e = lane + 32 * j;
            logit[e] = cluster.map_shared_rank(part, e / kRouterWarps)[e % kRouterWarps];
        }
    }
    cluster.sync();  // every block's part stays mapped until rank 0 has read it
    if (rank != 0 || warp != 0) return;
    __syncwarp();
    float score[4], biased[4];
    bool taken[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int e = lane + 32 * j;
        score[j] = ref::sigmoid(logit[e]);
        biased[j] = ref::add(score[j], bias[e]);
        taken[j] = false;
        if (logits_out != nullptr) logits_out[int64_t(t) * kMoeExperts + e] = logit[e];
    }
    __shared__ float sscore[kMoeExperts];
#pragma unroll
    for (int j = 0; j < 4; ++j) sscore[lane + 32 * j] = score[j];
    int32_t pick[kMoeTopK];
    for (int k = 0; k < kMoeTopK; ++k) {
        float bv = 0.0f;
        int bi = -1;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int e = lane + 32 * j;
            if (!taken[j] && (bi < 0 || ref::router_before(biased[j], e, bv, bi))) bv = biased[j], bi = e;
        }
#pragma unroll
        for (int off = 16; off >= 1; off /= 2) {
            const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, off);
            const int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, off);
            if (oi >= 0 && (bi < 0 || ref::router_before(ov, oi, bv, bi))) bv = ov, bi = oi;
        }
        pick[k] = bi;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (bi == lane + 32 * j) taken[j] = true;
    }
    __syncwarp();
    if (lane != 0) return;
    for (int i = 1; i < kMoeTopK; ++i) {
        const int32_t v = pick[i];
        int j = i - 1;
        while (j >= 0 && pick[j] > v) {
            pick[j + 1] = pick[j];
            --j;
        }
        pick[j + 1] = v;
    }
    float wk[kMoeTopK];
    if (mutant == static_cast<int>(MoeMutant::kSoftmaxRouting)) {
        float mx = logit[pick[0]];
        for (int k = 1; k < kMoeTopK; ++k) mx = fmaxf(mx, logit[pick[k]]);
        float ex[kMoeTopK], sum = 0.0f;
        for (int k = 0; k < kMoeTopK; ++k) ex[k] = ref::exp(ref::sub(logit[pick[k]], mx)), sum = ref::add(sum, ex[k]);
        for (int k = 0; k < kMoeTopK; ++k) wk[k] = ref::mul(ref::div(ex[k], sum), ref::kRouteScale);
    } else {
        ref::router_weights(sscore, pick, wk);
    }
    for (int k = 0; k < kMoeTopK; ++k) {
        ids[int64_t(t) * kMoeTopK + k] = pick[k];
        weights[int64_t(t) * kMoeTopK + k] = wk[k];
    }
}

// -- row quantizers (optionally fused with the capped SwiGLU x gamma) ---------------------------------------------
// One CTA per row, kQuantEpt consecutive values per thread: a quantization block (16 NVFP4 / 32 MXFP8 values) is
// kBlockThreads aligned lanes of one warp. The row amax (NVFP4) and each block's max |x| are order-free fmaxf
// reductions (warp shuffles, then one shared pass); ref::nvfp4_block's pairwise error sums run levels 1-2 in the
// thread (elements 4j .. 4j + 3) and levels 3-4 by shuffles across the block's lanes (IEEE add commutes, so each lane
// of a pair forms the same sum).

constexpr int kQuantEpt = 4;
constexpr int kMaxQuantThreads = 4096 / kQuantEpt;

struct QuantArgs {
    const bf16 *x;          // plain: [rows, k] at x_stride; swiglu: gate/up [rows, 2k] (interleaved)
    int64_t x_stride;
    int k;
    const int32_t *row_id;  // swiglu: sorted position p -> routed row id (gamma index, h tap row)
    const float *gamma;     // swiglu: [m * 8] by routed row id
    bf16 *h_tap;            // swiglu, nullable: [m * 8, k] by routed row id
    uint8_t *codes;         // NVFP4 [rows, k/2] / MXFP8 values [rows, k]
    uint8_t *scales;        // NVFP4 [rows, k/16] / MXFP8 exponents [rows, k/32]
    float *amax;            // NVFP4 [rows]
    int mutant;
};

template <bool kNvfp4, bool kSwiglu>
__global__ void __launch_bounds__(kMaxQuantThreads) quant_kernel(QuantArgs a) {
    constexpr int B = kNvfp4 ? 16 : 32;
    constexpr int kBlockThreads = B / kQuantEpt;
    pdl::enter();
    static_assert(kQuantEpt == 4 && 32 % kBlockThreads == 0, "levels 1-2 in-thread; a block in one warp");
    __shared__ float red[kMaxQuantThreads / 32];
    const int row = blockIdx.x;
    const int t = threadIdx.x, e0 = kQuantEpt * t;
    const bool active = e0 < a.k;  // a.k is a multiple of B: a block's lanes are all active or all idle
    float v[kQuantEpt];
    float gamma = 1.0f;
    int64_t id = row;
    if constexpr (kSwiglu) {
        id = a.row_id[row];
        if (a.mutant != static_cast<int>(MoeMutant::kGammaAfterDown)) gamma = a.gamma[id];
    }
    float block_max = 0.0f;
#pragma unroll
    for (int i = 0; i < kQuantEpt; ++i) {
        float x = 0.0f;
        if (active) {
            if constexpr (kSwiglu) {
                const bf16 *gu = a.x + int64_t(row) * a.x_stride + 2 * (e0 + i);
                x = ref::bf16_round(ref::capped_swiglu(bf(gu[0]), bf(gu[1]), gamma));
            } else {
                x = bf(a.x[int64_t(row) * a.x_stride + e0 + i]);
            }
        }
        v[i] = x;
        block_max = fmaxf(block_max, fabsf(x));
    }
    if constexpr (kSwiglu) {
        if (active && a.h_tap != nullptr) {
#pragma unroll
            for (int i = 0; i < kQuantEpt; ++i) a.h_tap[id * a.k + e0 + i] = to_bf(v[i]);
        }
    }
#pragma unroll
    for (int off = 1; off < kBlockThreads; off *= 2)
        block_max = fmaxf(block_max, __shfl_xor_sync(0xFFFFFFFFu, block_max, off));
    const bool block_lead = t % kBlockThreads == 0;
    if constexpr (kNvfp4) {
        float m = block_max;
#pragma unroll
        for (int off = 16; off >= kBlockThreads; off /= 2) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, off));
        if ((t & 31) == 0) red[t >> 5] = m;
        __syncthreads();
        float amax = red[0];
        for (int wi = 1; wi < static_cast<int>(blockDim.x >> 5); ++wi) amax = fmaxf(amax, red[wi]);
        if (a.mutant == static_cast<int>(MoeMutant::kGlobalActScale)) amax = kMutantGlobalAmax;
        // ref::nvfp4_block, this thread's 4 values of the block.
        const float encoder = fminf(ref::div(1536.0f, amax), ref::kFltMax);
        const float decoder = ref::div(1.0f, encoder);
        const float outer = ref::mul(amax, float(1.0 / 1536.0));
        const uint8_t b6 =
            ref::encode_e4m3_satfinite(fminf(ref::mul(block_max, ref::mul(encoder, float(1.0 / 6.0))), 448.0f));
        const uint8_t b4 = ref::encode_e4m3_satfinite(fminf(ref::mul(block_max, ref::mul(encoder, 0.25f)), 448.0f));
        const float s6 = ref::decode_e4m3(b6), s4 = ref::decode_e4m3(b4);
        const float inv6 = fminf(ref::div(1.0f, ref::mul(s6, decoder)), ref::kFltMax);
        const float inv4 = fminf(ref::div(1.0f, ref::mul(s4, decoder)), ref::kFltMax);
        uint8_t c6[kQuantEpt], c4[kQuantEpt];
        float e6[kQuantEpt], e4[kQuantEpt];
#pragma unroll
        for (int i = 0; i < kQuantEpt; ++i) {
            c6[i] = ref::encode_e2m1_satfinite(ref::mul(v[i], inv6));
            e6[i] = fabsf(ref::sub(ref::mul(ref::mul(ref::decode_e2m1(c6[i]), s6), outer), v[i]));
            c4[i] = ref::encode_e2m1_satfinite(ref::mul(v[i], inv4));
            e4[i] = fabsf(ref::sub(ref::mul(ref::mul(ref::decode_e2m1(c4[i]), s4), outer), v[i]));
        }
        float err6 = ref::add(ref::add(e6[0], e6[1]), ref::add(e6[2], e6[3]));
        float err4 = ref::add(ref::add(e4[0], e4[1]), ref::add(e4[2], e4[3]));
#pragma unroll
        for (int off = 1; off < kBlockThreads; off *= 2) {
            err6 = ref::add(err6, __shfl_xor_sync(0xFFFFFFFFu, err6, off));
            err4 = ref::add(err4, __shfl_xor_sync(0xFFFFFFFFu, err4, off));
        }
        if (!active) return;
        if (t == 0) a.amax[row] = amax;
        const bool choose4 = a.mutant != static_cast<int>(MoeMutant::kAlwaysSix) && err4 < err6;
        const uint8_t *c = choose4 ? c4 : c6;
        *reinterpret_cast<uint16_t *>(a.codes + int64_t(row) * (a.k / 2) + e0 / 2) =
            uint16_t(c[0] | c[1] << 4 | c[2] << 8 | c[3] << 12);
        if (block_lead) a.scales[int64_t(row) * (a.k / 16) + t / kBlockThreads] = choose4 ? b4 : b6;
    } else {
        if (!active) return;
        // ref::mxfp8_block with the block's amax = block_max, this thread's 4 values.
        const float min_scale = ref::bits_float(0x00400000u);
        const float raw = ref::mul(block_max, float(1.0 / 448.0));
        int exponent = int(((ref::float_bits(raw) + 0x007fffffu) & 0x7f800000u) >> 23);
        exponent = raw <= min_scale ? 0 : (exponent < 254 ? exponent : 254);
        if (raw != raw) exponent = 255;
        const float scale = fmaxf(ref::bits_float(uint32_t(exponent) << 23), min_scale);
        float inverse = exponent == 0 ? ref::bits_float(0x7f000000u) : ref::div(1.0f, scale);
        if (exponent == 255) inverse = 0.0f;
        uint32_t word = 0;
#pragma unroll
        for (int i = 0; i < kQuantEpt; ++i)
            word |= uint32_t(ref::encode_e4m3_satfinite(fminf(fmaxf(ref::mul(v[i], inverse), -448.0f), 448.0f)))
                    << (8 * i);
        *reinterpret_cast<uint32_t *>(a.codes + int64_t(row) * a.k + e0) = word;
        if (block_lead) a.scales[int64_t(row) * (a.k / 32) + t / kBlockThreads] = uint8_t(exponent);
    }
}

template <bool kNvfp4, bool kSwiglu>
cudaError_t launch_quant(const QuantArgs &a, int rows, cudaStream_t stream) {
    const int threads = (a.k / kQuantEpt + 31) / 32 * 32;
    return pdl::launch(quant_kernel<kNvfp4, kSwiglu>, dim3(rows), dim3(threads), 0, stream, a);
}

// -- dispatch: stable expert-sorted routed rows --------------------------------------------------------------------
// One CTA of 1024 threads: thread (e = tid % 128, seg = tid / 128) scans segment seg of each 1024-token chunk for
// expert e. Routed row ids t * 8 + k are placed per expert in ascending token order (k = the rank of e among the
// token's ascending ids). Outputs offsets [E + 1], tile_offset [E + 1] (row tiles of kTileRows), rows [m * 8]. The
// expert offsets are exclusive scans over threads 0..127 (integers: any order is exact).

constexpr int kDispatchThreads = 1024;
constexpr int kSegs = kDispatchThreads / kMoeExperts;
constexpr int kSegTokens = kDispatchThreads / kSegs;
constexpr int kTileRows = 16;
static_assert(kMoeExperts % 32 == 0 && kMoeExperts <= kDispatchThreads, "experts in whole warps");

__global__ void __launch_bounds__(kDispatchThreads)
    dispatch_kernel(const int32_t *__restrict__ ids, int m, int num_experts, int32_t *offsets, int32_t *tile_offset,
                    int32_t *rows) {
    pdl::enter();
    __shared__ uint32_t mask[kDispatchThreads][4];
    __shared__ int cnt[kSegs][kMoeExperts];
    __shared__ int run[kMoeExperts];
    __shared__ int warp_c[kMoeExperts / 32], warp_t[kMoeExperts / 32];
    const int e = threadIdx.x % kMoeExperts, seg = threadIdx.x / kMoeExperts;
    auto build = [&](int c0) {
        const int t = c0 + threadIdx.x;
        uint32_t mk[4] = {0u, 0u, 0u, 0u};
        if (t < m) {
            for (int k = 0; k < kMoeTopK; ++k) {
                const int id = ids[int64_t(t) * kMoeTopK + k];
                if (id >= 0 && id < num_experts) mk[id >> 5] |= 1u << (id & 31);
            }
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) mask[threadIdx.x][q] = mk[q];
    };
    const uint32_t bit = 1u << (e & 31);
    const int word = e >> 5;
    // tokens of this thread's segment in chunk c0 (the masks of tokens >= m are empty)
    auto seg_tokens = [&](int c0) { return max(0, min(kSegTokens, m - c0 - seg * kSegTokens)); };
    auto count = [&](int c0) {
        int c = 0;
        const int n = seg_tokens(c0);
        for (int j = 0; j < n; ++j) c += (mask[seg * kSegTokens + j][word] & bit) ? 1 : 0;
        return c;
    };
    int total = 0;
    for (int c0 = 0; c0 < m; c0 += kDispatchThreads) {
        __syncthreads();
        build(c0);
        __syncthreads();
        total += count(c0);
    }
    cnt[seg][e] = total;
    __syncthreads();
    const int x = threadIdx.x, lane = x & 31, wid = x >> 5;
    int c = 0, tl = 0, ci = 0, ti = 0;  // expert x: rows, row tiles, their inclusive scans within the warp
    if (x < kMoeExperts) {
        for (int s = 0; s < kSegs; ++s) c += cnt[s][x];
        tl = (c + kTileRows - 1) / kTileRows;
        ci = c, ti = tl;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int cu = __shfl_up_sync(0xFFFFFFFFu, ci, d), tu = __shfl_up_sync(0xFFFFFFFFu, ti, d);
            if (lane >= d) ci += cu, ti += tu;
        }
        if (lane == 31) warp_c[wid] = ci, warp_t[wid] = ti;
    }
    __syncthreads();
    if (x < kMoeExperts) {
        int off = ci - c, tiles = ti - tl;
        for (int w = 0; w < wid; ++w) off += warp_c[w], tiles += warp_t[w];
        if (x <= num_experts) offsets[x] = off, tile_offset[x] = tiles;
        run[x] = off;
        if (x == kMoeExperts - 1 && num_experts == kMoeExperts)
            offsets[kMoeExperts] = off + c, tile_offset[kMoeExperts] = tiles + tl;
    }
    // one chunk (m <= 1024): its masks and counts are still in place
    const bool rebuild = m > kDispatchThreads;
    for (int c0 = 0; c0 < m; c0 += kDispatchThreads) {
        __syncthreads();
        if (rebuild) {
            build(c0);
            __syncthreads();
            cnt[seg][e] = count(c0);
            __syncthreads();
        }
        int pos = run[e];
        for (int s = 0; s < seg; ++s) pos += cnt[s][e];
        const int n = seg_tokens(c0);
        for (int j = 0; j < n; ++j) {
            const uint32_t *mk = mask[seg * kSegTokens + j];
            if (mk[word] & bit) {
                int k = __popc(mk[word] & (bit - 1u));
                for (int q = 0; q < word; ++q) k += __popc(mk[q]);
                rows[pos++] = (c0 + seg * kSegTokens + j) * kMoeTopK + k;
            }
        }
        __syncthreads();
        if (seg == 0) {
            int sum = 0;
            for (int s = 0; s < kSegs; ++s) sum += cnt[s][e];
            run[e] += sum;
        }
    }
}

// -- grouped GEMM --------------------------------------------------------------------------------------------------
// CTA = 128 weight rows (one per thread) x one row tile (<= kTileRows routed rows of one expert); per thread the
// kTileRows accumulators are chained over the K blocks in order (ref::nvfp4_dot / ref::mxfp8_dot).

constexpr int kGemmThreads = 128;
constexpr int kChunk = 128;  // K elements staged per step

struct GemmArgs {
    const uint8_t *a;        // NVFP4 codes [*, k/2] / MXFP8 values [*, k]
    const uint8_t *a_scale;  // [*, k/16] / [*, k/32]
    const float *a_amax;     // NVFP4 [*]
    bool a_by_token;         // GEMM1: A row = rows[p] / 8; GEMM2: A row = p
    const int32_t *rows;     // sorted position p -> routed row id
    const int32_t *offsets;
    const int32_t *tile_offset;
    int32_t num_experts;
    const uint8_t *w;        // [E, n, k/2] / [E, n, k]
    const uint8_t *w_scale;  // [E, n, k/16] / [E, n, k/32]
    const float *w_amax;     // NVFP4 [E]
    const float *w_den;      // NVFP4 [1]
    int32_t n;
    int32_t k;
    bf16 *out;               // GEMM1: [R, n] by p; GEMM2: [R, n] by routed row id
    bf16 *tap;               // GEMM1, nullable: [R, n] by routed row id
    const float *gamma;      // GEMM2 kGammaAfterDown: [R] by routed row id
    int mutant;
};

template <bool kNvfp4, bool kDown>
__global__ void __launch_bounds__(kGemmThreads) gemm_kernel(GemmArgs g) {
    constexpr int kBlk = kNvfp4 ? 16 : 32;
    constexpr int kChunkBlocks = kChunk / kBlk;
    pdl::enter();
    __shared__ __align__(16) uint32_t a_s8[kNvfp4 ? kTileRows : 1][kChunk / 4];  // NVFP4: int8 2 * e2m1
    __shared__ __align__(16) float a_f[kNvfp4 ? 1 : kTileRows][kChunk];         // MXFP8: decoded E4M3
    __shared__ float a_sf[kTileRows][kChunkBlocks];
    __shared__ int64_t a_row[kTileRows];
    __shared__ int32_t p_row[kTileRows];

    const int tile = blockIdx.y;
    if (tile >= g.tile_offset[g.num_experts]) return;
    int lo = 0, hi = g.num_experts;  // tile_offset[lo] <= tile < tile_offset[hi]
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (g.tile_offset[mid] <= tile) lo = mid;
        else hi = mid;
    }
    const int e = lo;
    const int p0 = g.offsets[e] + (tile - g.tile_offset[e]) * kTileRows;
    const int nrows = min(kTileRows, g.offsets[e + 1] - p0);
    if (threadIdx.x < kTileRows) {
        const int r = threadIdx.x;
        int64_t src = -1;
        int32_t id = -1;
        if (r < nrows) {
            id = g.rows[p0 + r];
            src = g.a_by_token ? id / kMoeTopK : p0 + r;
        }
        a_row[r] = src;
        p_row[r] = id;
    }
    const int col = blockIdx.x * kGemmThreads + threadIdx.x;
    const bool col_ok = col < g.n;
    const int64_t w_row = int64_t(e) * g.n + (col_ok ? col : 0);
    float acc[kTileRows];
#pragma unroll
    for (int r = 0; r < kTileRows; ++r) acc[r] = 0.0f;

    for (int k0 = 0; k0 < g.k; k0 += kChunk) {
        __syncthreads();
        if constexpr (kNvfp4) {
            for (int idx = threadIdx.x; idx < kTileRows * (kChunk / 8); idx += kGemmThreads) {
                const int r = idx / (kChunk / 8), wd = idx % (kChunk / 8);
                const int64_t src = a_row[r];
                uint32_t v = 0;
                if (src >= 0) v = *reinterpret_cast<const uint32_t *>(g.a + src * (g.k / 2) + k0 / 2 + wd * 4);
                a_s8[r][2 * wd] = e2m1x4_to_s8x4(v & 0xFFFFu);
                a_s8[r][2 * wd + 1] = e2m1x4_to_s8x4(v >> 16);
            }
        } else {
            for (int idx = threadIdx.x; idx < kTileRows * (kChunk / 4); idx += kGemmThreads) {
                const int r = idx / (kChunk / 4), wd = idx % (kChunk / 4);
                const int64_t src = a_row[r];
                uint32_t v = 0;
                if (src >= 0) v = *reinterpret_cast<const uint32_t *>(g.a + src * g.k + k0 + wd * 4);
                e4m3x2_to_float(uint16_t(v & 0xFFFFu), a_f[r][4 * wd], a_f[r][4 * wd + 1]);
                e4m3x2_to_float(uint16_t(v >> 16), a_f[r][4 * wd + 2], a_f[r][4 * wd + 3]);
            }
        }
        for (int idx = threadIdx.x; idx < kTileRows * kChunkBlocks; idx += kGemmThreads) {
            const int r = idx / kChunkBlocks, j = idx % kChunkBlocks;
            const int64_t src = a_row[r];
            const uint8_t s = src >= 0 ? g.a_scale[src * (g.k / kBlk) + k0 / kBlk + j] : uint8_t(0);
            a_sf[r][j] = kNvfp4 ? ref::decode_e4m3(s) : ref::decode_ue8m0(s);
        }
        __syncthreads();
        if (!col_ok) continue;
        if constexpr (kNvfp4) {
            const uint4 *src = reinterpret_cast<const uint4 *>(g.w + w_row * (g.k / 2) + k0 / 2);
            const uint8_t *ssrc = g.w_scale + w_row * (g.k / 16) + k0 / 16;
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {  // 64 K elements (4 blocks) per half
                const uint4 q0 = __ldg(src + 2 * h2), q1 = __ldg(src + 2 * h2 + 1);
                const uint32_t words[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
                uint32_t wq[16];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    wq[2 * i] = e2m1x4_to_s8x4(words[i] & 0xFFFFu);
                    wq[2 * i + 1] = e2m1x4_to_s8x4(words[i] >> 16);
                }
                float ws[4];
#pragma unroll
                for (int b4 = 0; b4 < 4; ++b4) ws[b4] = ref::decode_e4m3(__ldg(ssrc + 4 * h2 + b4));
#pragma unroll
                for (int r = 0; r < kTileRows; ++r) {
                    if (r < nrows) {
#pragma unroll
                        for (int b4 = 0; b4 < 4; ++b4) {
                            int isum = 0;
#pragma unroll
                            for (int j = 0; j < 4; ++j)
                                isum = __dp4a(static_cast<int>(wq[4 * b4 + j]),
                                              static_cast<int>(a_s8[r][16 * h2 + 4 * b4 + j]), isum);
                            // T_j = S_j * (sx * sw), S_j = isum / 4: every step exact (ref::nvfp4_block_term).
                            const float t = ref::mul(ref::mul(static_cast<float>(isum),
                                                              ref::mul(a_sf[r][4 * h2 + b4], ws[b4])),
                                                     0.25f);
                            acc[r] = ref::add(acc[r], t);
                        }
                    }
                }
            }
        } else {
            const uint4 *src = reinterpret_cast<const uint4 *>(g.w + w_row * g.k + k0);
            const uint8_t *ssrc = g.w_scale + w_row * (g.k / 32) + k0 / 32;
#pragma unroll
            for (int b = 0; b < kChunkBlocks; ++b) {
                const uint4 q0 = __ldg(src + 2 * b), q1 = __ldg(src + 2 * b + 1);
                const uint32_t words[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
                float wf[32];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    e4m3x2_to_float(uint16_t(words[i] & 0xFFFFu), wf[4 * i], wf[4 * i + 1]);
                    e4m3x2_to_float(uint16_t(words[i] >> 16), wf[4 * i + 2], wf[4 * i + 3]);
                }
                const float ws = ref::decode_ue8m0(__ldg(ssrc + b));
#pragma unroll
                for (int r = 0; r < kTileRows; ++r) {
                    if (r < nrows) {
                        float blk = 0.0f;
#pragma unroll
                        for (int i = 0; i < 32; ++i) blk = ref::fma(a_f[r][32 * b + i], wf[i], blk);
                        acc[r] = ref::add(acc[r], ref::mul(blk, ref::mul(a_sf[r][b], ws)));
                    }
                }
            }
        }
    }
    if (!col_ok) return;
    float rcp = 0.0f, w_amax = 0.0f;
    if constexpr (kNvfp4) {
        rcp = ref::nvfp4_outer_rcp(g.w_den[0]);
        w_amax = g.w_amax[e];
    }
#pragma unroll
    for (int r = 0; r < kTileRows; ++r) {
        if (r < nrows) {
            float y = acc[r];
            if constexpr (kNvfp4) y = ref::mul(y, ref::nvfp4_outer_scale(g.a_amax[a_row[r]], w_amax, rcp));
            const int32_t id = p_row[r];
            if constexpr (kDown) {
                if (g.mutant == static_cast<int>(MoeMutant::kGammaAfterDown)) y = ref::mul(y, g.gamma[id]);
                g.out[int64_t(id) * g.n + col] = to_bf(y);
            } else {
                const bf16 v = to_bf(y);
                g.out[int64_t(p0 + r) * g.n + col] = v;
                if (g.tap != nullptr) g.tap[int64_t(id) * g.n + col] = v;
            }
        }
    }
}

// -- grouped GEMV, decode calls of at most kWarpMaxM tokens (NVFP4 and MXFP8) ----------------------------------------
// gemm_kernel's arithmetic element for element: the same block terms (each a pure function of its block), chained
// over the K blocks in order from 0, and the same outer factor. The tile's activation rows are staged in shared
// memory once per block (NVFP4 codes widened to int8, MXFP8 bytes as stored, scales decoded). kRp (a power of two
// >= m) bounds the tile rows: a token routes to an expert at most once, so a tile holds at most m rows.

constexpr int kWarpMaxM = 4;

// Tile blockIdx.y of a grouped GEMV call: its expert, first sorted position and row count (at most kRp), from the
// call's tile_offset / offsets staged in toff / offs (shared). nrows = 0 past the call's tiles (block-uniform).
struct GemvTile {
    int e, p0, nrows;
};

template <int kRp>
__device__ __forceinline__ GemvTile gemv_tile(const GemmArgs &g, int32_t *toff, int32_t *offs) {
    for (int i = threadIdx.x; i <= g.num_experts; i += blockDim.x) {
        toff[i] = g.tile_offset[i];
        offs[i] = g.offsets[i];
    }
    __syncthreads();
    const int tile = blockIdx.y;
    if (tile >= toff[g.num_experts]) return {0, 0, 0};
    int lo = 0, hi = g.num_experts;  // toff[lo] <= tile < toff[hi]
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (toff[mid] <= tile) lo = mid;
        else hi = mid;
    }
    const int p0 = offs[lo] + (tile - toff[lo]) * kTileRows;
    return {lo, p0, min(kRp, offs[lo + 1] - p0)};
}

// Threads r < kRp: tile row r's activation row and routed id (-1 past nrows).
template <int kRp>
__device__ __forceinline__ void gemv_rows(const GemmArgs &g, const GemvTile &t, int64_t *a_row, int32_t *p_row) {
    if (threadIdx.x < kRp) {
        const int r = threadIdx.x;
        int64_t a = -1;
        int32_t id = -1;
        if (r < t.nrows) {
            id = g.rows[t.p0 + r];
            a = g.a_by_token ? id / kMoeTopK : t.p0 + r;
        }
        a_row[r] = a;
        p_row[r] = id;
    }
}

// The output of tile row r (sorted position p0 + r, routed id), weight row col: GEMM1 stores by sorted position (and
// the tap by routed id), GEMM2 by routed id (gamma after the down projection in the fail-first mutant).
template <bool kDown>
__device__ __forceinline__ void gemv_store(const GemmArgs &g, int p, int32_t id, int col, float y) {
    if constexpr (kDown) {
        if (g.mutant == static_cast<int>(MoeMutant::kGammaAfterDown)) y = ref::mul(y, g.gamma[id]);
        g.out[int64_t(id) * g.n + col] = to_bf(y);
    } else {
        const bf16 v = to_bf(y);
        g.out[int64_t(p) * g.n + col] = v;
        if (g.tap != nullptr) g.tap[int64_t(id) * g.n + col] = v;
    }
}

// NVFP4: a quad of lanes per weight row. Lane q of the quad loads chunks q, q + 4, q + 8, .. of the row (16 bytes =
// 2 blocks each; the quad streams 64 contiguous bytes per step), all in flight at once, and computes its chunks'
// block terms for every tile row. At step s the quad's eight terms (blocks 8s .. 8s + 7, lane j holding 8s + 2j and
// 8s + 2j + 1) go round the quad by shuffles and every lane of the quad adds them to its running sums in block order;
// lane q stores tile row q.
template <bool kDown, int kRp>
struct QuadGemv {
    static constexpr int kK = kDown ? I : H;  // the call's k
    static constexpr int kKb = kK / 16;        // blocks per row
    static constexpr int kSteps = kKb / 8;     // chunks per lane
    static constexpr int kRowBytes = kK / 2;
    static constexpr int kWarps = 4;
    static constexpr int kCols = kWarps * 8;  // weight rows per block
    static_assert(kKb % 8 == 0 && kRp <= 4, "whole quad steps; one quad lane per tile row");
};

template <bool kDown, int kRp>
__global__ void __launch_bounds__(32 * QuadGemv<kDown, kRp>::kWarps) quad_gemv_kernel(GemmArgs g) {
    using P = QuadGemv<kDown, kRp>;
    constexpr int kKb = P::kKb, kSteps = P::kSteps, kThreads = 32 * P::kWarps;
    pdl::enter();
    __shared__ __align__(16) uint32_t a_w[kRp][P::kK / 4];  // int8 2 * e2m1, 4 per word
    __shared__ float a_sf[kRp][kKb];
    __shared__ int32_t toff[kMoeExperts + 1], offs[kMoeExperts + 1];
    __shared__ int64_t a_row[kRp];
    __shared__ int32_t p_row[kRp];

    const GemvTile t = gemv_tile<kRp>(g, toff, offs);
    if (t.nrows == 0) return;
    const int lane = threadIdx.x % 32, q = lane % 4;
    const int col = (blockIdx.x * P::kWarps + threadIdx.x / 32) * 8 + lane / 4;  // the quad's weight row
    const bool col_ok = col < g.n;
    const int64_t w_row = int64_t(t.e) * g.n + (col_ok ? col : 0);
    // The quad's weights first (in flight while the activations are staged).
    const uint4 *wsrc = reinterpret_cast<const uint4 *>(g.w + w_row * P::kRowBytes) + q;
    const uint16_t *ssrc = reinterpret_cast<const uint16_t *>(g.w_scale + w_row * kKb) + q;
    uint4 wv[kSteps];
    uint32_t wsb[kSteps];  // the chunk's two scale bytes
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
        wv[s] = __ldg(wsrc + 4 * s);
        wsb[s] = __ldg(ssrc + 4 * s);
    }
    gemv_rows<kRp>(g, t, a_row, p_row);
    __syncthreads();
    constexpr int kAw = P::kRowBytes / 4;  // 32-bit words of an activation row
    for (int idx = threadIdx.x; idx < t.nrows * kAw; idx += kThreads) {
        const int r = idx / kAw, wd = idx % kAw;
        const uint32_t v = __ldg(reinterpret_cast<const uint32_t *>(g.a + a_row[r] * P::kRowBytes) + wd);
        a_w[r][2 * wd] = e2m1x4_to_s8x4(v & 0xFFFFu);
        a_w[r][2 * wd + 1] = e2m1x4_to_s8x4(v >> 16);
    }
    for (int idx = threadIdx.x; idx < t.nrows * kKb; idx += kThreads) {
        const int r = idx / kKb, j = idx % kKb;
        a_sf[r][j] = ref::decode_e4m3(g.a_scale[a_row[r] * kKb + j]);
    }
    __syncthreads();
    float y[kRp];
#pragma unroll
    for (int r = 0; r < kRp; ++r) y[r] = 0.0f;
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
        const uint32_t ww[4] = {wv[s].x, wv[s].y, wv[s].z, wv[s].w};
        float ws[2];  // the chunk's two block scales
        e4m3x2_to_float(uint16_t(wsb[s]), ws[0], ws[1]);
        float tm[2][kRp];
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int blk = 2 * (4 * s + q) + b;
            // Word h of the block (codes 4h .. 4h + 3) as the magnitudes of its positive codes (pos) and of its
            // negative ones (neg): isum = dot(pos) - dot(neg), the same exact integer as e2m1x4_to_s8x4's.
            uint32_t pos[4], neg[4];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t u = ww[2 * b + h], un = u ^ 0x88888888u;
                pos[2 * h] = e2m1x4_mag(u);
                pos[2 * h + 1] = e2m1x4_mag(u >> 16);
                neg[2 * h] = e2m1x4_mag(un);
                neg[2 * h + 1] = e2m1x4_mag(un >> 16);
            }
#pragma unroll
            for (int r = 0; r < kRp; ++r) {
                tm[b][r] = 0.0f;
                if (r < t.nrows) {
                    const uint4 a4 = *reinterpret_cast<const uint4 *>(&a_w[r][4 * blk]);
                    const int av[4] = {static_cast<int>(a4.x), static_cast<int>(a4.y), static_cast<int>(a4.z),
                                       static_cast<int>(a4.w)};
                    int ip = 0, in = 0;
#pragma unroll
                    for (int h = 0; h < 4; ++h) {
                        ip = __dp4a(static_cast<int>(pos[h]), av[h], ip);
                        in = __dp4a(static_cast<int>(neg[h]), av[h], in);
                    }
                    // T_j = S_j * (sx * sw), S_j = isum / 4: every step exact (ref::nvfp4_block_term).
                    tm[b][r] = ref::mul(ref::mul(static_cast<float>(ip - in), ref::mul(a_sf[r][blk], ws[b])), 0.25f);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int b = 0; b < 2; ++b)
#pragma unroll
                for (int r = 0; r < kRp; ++r)
                    y[r] = ref::add(y[r], __shfl_sync(0xFFFFFFFFu, tm[b][r], (lane & ~3) | j));
    }
    if (!col_ok) return;
#pragma unroll
    for (int r = 0; r < kRp; ++r) {
        if (r != q || r >= t.nrows) continue;
        const float v =
            ref::mul(y[r], ref::nvfp4_outer_scale(g.a_amax[a_row[r]], g.w_amax[t.e], ref::nvfp4_outer_rcp(g.w_den[0])));
        gemv_store<kDown>(g, t.p0 + r, p_row[r], col, v);
    }
}

// MXFP8: one warp per kWr consecutive weight rows of an expert, one contiguous run of kWr * kKb blocks (32 bytes
// each): lane l loads blocks l, l + 32, ... of the run, all in flight at once, computes their block terms for every
// tile row into shared memory, then the lane owning (weight row lane / kRp, tile row lane % kRp) adds its row's
// terms in block order.
template <bool kDown, int kRp>
struct WarpGemv {
    static constexpr int kK = kDown ? I : H;              // the call's k
    static constexpr int kKb = kK / 32;                   // blocks per row
    static constexpr int kWr = kDown ? 4 : 2;             // weight rows per warp
    static constexpr int kItems = (kWr * kKb + 31) / 32;  // blocks per lane
    static constexpr int kWarps = kRp <= 2 ? 4 : 2;       // warps per block (shared memory < 48 KiB)
    static constexpr int kCols = kWarps * kWr;            // weight rows per block
    static_assert(kWr * kRp <= 32, "one lane per (weight row, tile row)");
};

template <bool kDown, int kRp>
__global__ void __launch_bounds__(32 * WarpGemv<kDown, kRp>::kWarps) warp_gemv_kernel(GemmArgs g) {
    using P = WarpGemv<kDown, kRp>;
    constexpr int kKb = P::kKb, kWr = P::kWr, kThreads = 32 * P::kWarps;
    pdl::enter();
    __shared__ __align__(16) uint32_t a_w[kRp][P::kK / 4];  // E4M3 bytes
    __shared__ float a_sf[kRp][kKb];
    __shared__ float terms[P::kWarps][kWr * kRp][kKb + 1];  // [warp][weight row * kRp + tile row][block]
    __shared__ int32_t toff[kMoeExperts + 1], offs[kMoeExperts + 1];
    __shared__ int64_t a_row[kRp];
    __shared__ int32_t p_row[kRp];

    const GemvTile tl = gemv_tile<kRp>(g, toff, offs);
    if (tl.nrows == 0) return;
    const int nrows = tl.nrows;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int col0 = (blockIdx.x * P::kWarps + warp) * kWr;
    const int span = max(0, min(kWr, g.n - col0)) * kKb;  // blocks of the warp's weight rows
    const int64_t row0 = int64_t(tl.e) * g.n + col0;      // the warp's first weight row
    // The warp's weights first (in flight while the activations are staged).
    const uint4 *wsrc = reinterpret_cast<const uint4 *>(g.w + row0 * P::kK);
    uint4 wv[P::kItems][2];
    uint32_t wsb[P::kItems];  // the block's scale byte
#pragma unroll
    for (int j = 0; j < P::kItems; ++j) {
        const int i = lane + 32 * j;
        if (i < span) {
            wv[j][0] = __ldg(wsrc + 2 * i);
            wv[j][1] = __ldg(wsrc + 2 * i + 1);
            wsb[j] = __ldg(g.w_scale + row0 * kKb + i);
        }
    }
    gemv_rows<kRp>(g, tl, a_row, p_row);
    __syncthreads();
    constexpr int kAw = P::kK / 4;  // 32-bit words of an activation row
    for (int idx = threadIdx.x; idx < nrows * kAw; idx += kThreads) {
        const int r = idx / kAw, wd = idx % kAw;
        a_w[r][wd] = __ldg(reinterpret_cast<const uint32_t *>(g.a + a_row[r] * P::kK) + wd);
    }
    for (int idx = threadIdx.x; idx < nrows * kKb; idx += kThreads) {
        const int r = idx / kKb, j = idx % kKb;
        a_sf[r][j] = ref::decode_ue8m0(g.a_scale[a_row[r] * kKb + j]);
    }
    __syncthreads();
    float (*t)[kKb + 1] = terms[warp];
#pragma unroll
    for (int j = 0; j < P::kItems; ++j) {
        const int i = lane + 32 * j;
        if (i >= span) continue;
        const int w = i / kKb, c = i - w * kKb;  // weight row of the warp, block of the row
        const uint32_t ww[8] = {wv[j][0].x, wv[j][0].y, wv[j][0].z, wv[j][0].w,
                                wv[j][1].x, wv[j][1].y, wv[j][1].z, wv[j][1].w};
        float wf[32];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            e4m3x2_to_float(uint16_t(ww[u] & 0xFFFFu), wf[4 * u], wf[4 * u + 1]);
            e4m3x2_to_float(uint16_t(ww[u] >> 16), wf[4 * u + 2], wf[4 * u + 3]);
        }
        const float ws = ref::decode_ue8m0(uint8_t(wsb[j]));
#pragma unroll
        for (int r = 0; r < kRp; ++r) {
            if (r < nrows) {
                float blk = 0.0f;
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const uint32_t av = a_w[r][8 * c + u];
                    float a0, a1, a2, a3;
                    e4m3x2_to_float(uint16_t(av & 0xFFFFu), a0, a1);
                    e4m3x2_to_float(uint16_t(av >> 16), a2, a3);
                    blk = ref::fma(a0, wf[4 * u], blk);
                    blk = ref::fma(a1, wf[4 * u + 1], blk);
                    blk = ref::fma(a2, wf[4 * u + 2], blk);
                    blk = ref::fma(a3, wf[4 * u + 3], blk);
                }
                t[w * kRp + r][c] = ref::mul(blk, ref::mul(a_sf[r][c], ws));
            }
        }
    }
    __syncwarp();
    const int w = lane / kRp, r = lane % kRp, col = col0 + w;
    if (lane >= kWr * kRp || r >= nrows || col >= g.n) return;
    float y = 0.0f;
    for (int blk = 0; blk < kKb; ++blk) y = ref::add(y, t[lane][blk]);
    gemv_store<kDown>(g, tl.p0 + r, p_row[r], col, y);
}

template <bool kDown, int kRp>
cudaError_t launch_quad_gemv(const GemmArgs &g, int max_tiles, cudaStream_t stream) {
    using P = QuadGemv<kDown, kRp>;
    const dim3 grid((g.n + P::kCols - 1) / P::kCols, max_tiles);
    return pdl::launch(quad_gemv_kernel<kDown, kRp>, grid, dim3(32 * P::kWarps), 0, stream, g);
}

template <bool kDown, int kRp>
cudaError_t launch_warp_gemv(const GemmArgs &g, int max_tiles, cudaStream_t stream) {
    using P = WarpGemv<kDown, kRp>;
    const dim3 grid((g.n + P::kCols - 1) / P::kCols, max_tiles);
    return pdl::launch(warp_gemv_kernel<kDown, kRp>, grid, dim3(32 * P::kWarps), 0, stream, g);
}

// A call of m tokens: quad_gemv_kernel (NVFP4) / warp_gemv_kernel (MXFP8) when m <= kWarpMaxM and k is the model's
// (NVFP4: weight scales 2-byte aligned), else gemm_kernel. All of them compute the same bits.
template <bool kNvfp4, bool kDown>
cudaError_t launch_gemm(const GemmArgs &g, int m, int max_tiles, cudaStream_t stream) {
    const uintptr_t sa = reinterpret_cast<uintptr_t>(g.w_scale);
    if (m <= kWarpMaxM && g.k == (kDown ? I : H) && (!kNvfp4 || (sa & 1u) == 0)) {
        if constexpr (kNvfp4) {
            if (m == 1) return launch_quad_gemv<kDown, 1>(g, max_tiles, stream);
            if (m == 2) return launch_quad_gemv<kDown, 2>(g, max_tiles, stream);
            return launch_quad_gemv<kDown, 4>(g, max_tiles, stream);
        } else {
            if (m == 1) return launch_warp_gemv<kDown, 1>(g, max_tiles, stream);
            if (m == 2) return launch_warp_gemv<kDown, 2>(g, max_tiles, stream);
            return launch_warp_gemv<kDown, 4>(g, max_tiles, stream);
        }
    }
    return pdl::launch(gemm_kernel<kNvfp4, kDown>, dim3((g.n + kGemmThreads - 1) / kGemmThreads, max_tiles),
                       dim3(kGemmThreads), 0, stream, g);
}

// -- combine -------------------------------------------------------------------------------------------------------

constexpr int kCombineThreads = 256;

__global__ void __launch_bounds__(kCombineThreads)
    combine_kernel(const bf16 *__restrict__ o, bf16 *out, int64_t out_stride) {
    pdl::enter();
    const int t = blockIdx.x;
    for (int n = threadIdx.x; n < H; n += kCombineThreads) {
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < kMoeTopK; ++k) s = ref::add(s, bf(o[(int64_t(t) * kMoeTopK + k) * H + n]));
        out[int64_t(t) * out_stride + n] = to_bf(s);
    }
}

// -- workspace -----------------------------------------------------------------------------------------------------

constexpr size_t align256(size_t v) { return (v + 255) / 256 * 256; }

struct Workspace {
    int32_t *offsets, *tile_offset, *rows;
    uint8_t *xq, *xs;
    float *xa;
    bf16 *gu;
    uint8_t *hq, *hs;
    float *ha;
    bf16 *o;
    size_t bytes;
};

Workspace carve(void *base, int m, bool nvfp4) {
    const size_t r = size_t(m) * kMoeTopK;
    Workspace w{};
    size_t off = 0;
    auto take = [&](size_t bytes) {
        char *p = static_cast<char *>(base) + off;
        off += align256(bytes);
        return p;
    };
    w.offsets = reinterpret_cast<int32_t *>(take(sizeof(int32_t) * (kMoeExperts + 1)));
    w.tile_offset = reinterpret_cast<int32_t *>(take(sizeof(int32_t) * (kMoeExperts + 1)));
    w.rows = reinterpret_cast<int32_t *>(take(sizeof(int32_t) * r));
    w.xq = reinterpret_cast<uint8_t *>(take(size_t(m) * (nvfp4 ? H / 2 : H)));
    w.xs = reinterpret_cast<uint8_t *>(take(size_t(m) * (nvfp4 ? H / 16 : H / 32)));
    w.xa = reinterpret_cast<float *>(take(sizeof(float) * m));
    w.gu = reinterpret_cast<bf16 *>(take(sizeof(bf16) * r * 2 * I));
    w.hq = reinterpret_cast<uint8_t *>(take(r * (nvfp4 ? I / 2 : I)));
    w.hs = reinterpret_cast<uint8_t *>(take(r * (nvfp4 ? I / 16 : I / 32)));
    w.ha = reinterpret_cast<float *>(take(sizeof(float) * r));
    w.o = reinterpret_cast<bf16 *>(take(sizeof(bf16) * r * H));
    w.bytes = off;
    return w;
}

bool valid_mutant(MoeMutant mutant) {
    const int v = static_cast<int>(mutant);
    return v >= 0 && v <= static_cast<int>(MoeMutant::kAlwaysSix);
}

}  // namespace

cudaError_t router(const bf16 *x, int64_t x_stride, int m, const bf16 *w, const float *bias, int32_t *ids,
                   float *weights, float *logits, cudaStream_t stream, MoeMutant mutant) {
    if (x == nullptr || w == nullptr || bias == nullptr || ids == nullptr || weights == nullptr || m < 1 ||
        m > 65535 || x_stride < H || !valid_mutant(mutant))
        return cudaErrorInvalidValue;
    return pdl::launch(router_kernel, dim3(kRouterCluster, unsigned(m)), dim3(kRouterThreads), 0, stream, x, x_stride,
                       w, bias, ids, weights, logits, static_cast<int>(mutant));
}

cudaError_t quantize_nvfp4(const bf16 *x, int64_t x_stride, int rows, int k, uint8_t *codes, uint8_t *scales,
                           float *amax, cudaStream_t stream, MoeMutant mutant) {
    if (x == nullptr || codes == nullptr || scales == nullptr || amax == nullptr || rows < 1 || k < 64 ||
        k % 64 != 0 || k > 4096 || x_stride < k || !valid_mutant(mutant) || (reinterpret_cast<uintptr_t>(codes) & 7u))
        return cudaErrorInvalidValue;
    QuantArgs a{x, x_stride, k, nullptr, nullptr, nullptr, codes, scales, amax, static_cast<int>(mutant)};
    return launch_quant<true, false>(a, rows, stream);
}

cudaError_t quantize_mxfp8(const bf16 *x, int64_t x_stride, int rows, int k, uint8_t *values, uint8_t *exponents,
                           cudaStream_t stream) {
    if (x == nullptr || values == nullptr || exponents == nullptr || rows < 1 || k < 32 || k % 32 != 0 ||
        k > 4096 || x_stride < k || !aligned16(values))
        return cudaErrorInvalidValue;
    QuantArgs a{x, x_stride, k, nullptr, nullptr, nullptr, values, exponents, nullptr, 0};
    return launch_quant<false, false>(a, rows, stream);
}

size_t moe_workspace_bytes(int m, bool nvfp4) { return carve(nullptr, m, nvfp4).bytes; }

cudaError_t moe_experts(const bf16 *x, int64_t x_stride, int m, const int32_t *ids, const float *weights,
                        const MoeWeights &w, void *workspace, size_t workspace_bytes, bf16 *out, int64_t out_stride,
                        cudaStream_t stream, const MoeTaps *taps, MoeMutant mutant) {
    if (x == nullptr || ids == nullptr || weights == nullptr || out == nullptr || workspace == nullptr || m < 1 ||
        m > kMoeMaxTokens || x_stride < H || out_stride < H || !valid_mutant(mutant) ||
        (reinterpret_cast<uintptr_t>(workspace) & 255u) || w.num_experts < kMoeTopK ||
        w.num_experts > kMoeExperts || w.w13 == nullptr || w.w13_scale == nullptr || w.w2 == nullptr ||
        w.w2_scale == nullptr || !aligned16(w.w13) || !aligned16(w.w2) ||
        (w.nvfp4 && (w.w13_amax == nullptr || w.w2_amax == nullptr || w.w13_den == nullptr || w.w2_den == nullptr)))
        return cudaErrorInvalidValue;
    const Workspace ws = carve(workspace, m, w.nvfp4);
    if (workspace_bytes < ws.bytes) return cudaErrorInvalidValue;
    const int mi = static_cast<int>(mutant);
    const int r = m * kMoeTopK;
    const int max_tiles = (r < w.num_experts ? r : w.num_experts) + r / kTileRows;
    bf16 *gu_tap = taps != nullptr ? taps->gate_up : nullptr;
    bf16 *h_tap = taps != nullptr ? taps->h : nullptr;

    cudaError_t err;
    if (w.nvfp4) err = quantize_nvfp4(x, x_stride, m, H, ws.xq, ws.xs, ws.xa, stream, mutant);
    else err = quantize_mxfp8(x, x_stride, m, H, ws.xq, ws.xs, stream);
    if (err != cudaSuccess) return err;
    err = pdl::launch(dispatch_kernel, dim3(1), dim3(kDispatchThreads), 0, stream, ids, m, w.num_experts, ws.offsets,
                      ws.tile_offset, ws.rows);
    if (err != cudaSuccess) return err;

    GemmArgs g1{ws.xq, ws.xs, ws.xa, true, ws.rows, ws.offsets, ws.tile_offset, w.num_experts, w.w13, w.w13_scale,
                w.w13_amax, w.w13_den, 2 * I, H, ws.gu, gu_tap, weights, mi};
    err = w.nvfp4 ? launch_gemm<true, false>(g1, m, max_tiles, stream)
                  : launch_gemm<false, false>(g1, m, max_tiles, stream);
    if (err != cudaSuccess) return err;

    QuantArgs qa{ws.gu, 2 * I, I, ws.rows, weights, h_tap, ws.hq, ws.hs, ws.ha, mi};
    err = w.nvfp4 ? launch_quant<true, true>(qa, r, stream) : launch_quant<false, true>(qa, r, stream);
    if (err != cudaSuccess) return err;

    GemmArgs g2{ws.hq, ws.hs, ws.ha, false, ws.rows, ws.offsets, ws.tile_offset, w.num_experts, w.w2, w.w2_scale,
                w.w2_amax, w.w2_den, H, I, ws.o, nullptr, weights, mi};
    err = w.nvfp4 ? launch_gemm<true, true>(g2, m, max_tiles, stream)
                  : launch_gemm<false, true>(g2, m, max_tiles, stream);
    if (err != cudaSuccess) return err;

    err = pdl::launch(combine_kernel, dim3(m), dim3(kCombineThreads), 0, stream, ws.o, out, out_stride);
    if (err != cudaSuccess) return err;
    if (taps != nullptr && taps->expert_out != nullptr)
        err = cudaMemcpyAsync(taps->expert_out, ws.o, sizeof(bf16) * size_t(r) * H, cudaMemcpyDeviceToDevice, stream);
    return err;
}

}  // namespace cere::inkling
