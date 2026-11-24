// K5R region kernel (see k5r_region.h): one persistent launch interpreting a fixed instruction list over
// device-function copies of today's GDN decode kernels. Each copy names the function whose arithmetic it repeats; a
// change there must be repeated here (the K5R bit-exact rows fail otherwise).
#include "ktest/k5r_region.h"

#include "kernels/k2_small_m.h"
#include "kernels/k5_gdn.h"

#include <cstdint>

namespace cere::ktest::k5r {
namespace {

using bf16 = __nv_bfloat16;
using k5::kConvDim;
using k5::kConvState;
using k5::kConvWidth;
using k5::kHeadDim;
using k5::kKeyDim;
using k5::kValueDim;
using k5::kValueHeads;

constexpr uint32_t kHidden = 2560;
constexpr uint32_t kWarps = 16;  // k2s::split_warps(2560) == split_warps(6144) == 16 (forward() checks)
constexpr uint32_t kThreads = kWarps * 32;
constexpr uint32_t kUnroll = 5;    // k2_small_m.cu's load schedule (no bit depends on it)
constexpr uint32_t kTileMax = 16;  // columns per GEMM tile (16 or 8; no bit depends on it)
constexpr uint32_t kMaxSegs = 4;
constexpr uint32_t kMaxPrefetch = 4;
constexpr uint32_t kLine = 128;
// A CORE item: 16 warps x 4 value rows (the decode kernel's ILP4 rows per warp) = 64 rows, half a value head.
constexpr uint32_t kRowsPerWarp = 4;
constexpr uint32_t kLaneElems = 4;
constexpr uint32_t kHalves = kHeadDim / (kWarps * kRowsPerWarp);
constexpr uint32_t kCoreItems = kValueHeads * kHalves;
constexpr uint32_t kQkChannels = 2 * kKeyDim;  // the conv channels whose state wave 3 writes
constexpr uint32_t kConvQkItems = kQkChannels / kThreads;
static_assert(kHalves == 2 && kQkChannels % kThreads == 0, "region item shapes");

constexpr float kScale = 0.08838834764831845f;  // 128^-0.5 (k5_gdn_layer.cu)
constexpr float kLog2e = 1.4426950408889634f;   // 0x3FB8AA3B (k5_gdn.cu, k5_gdn_decode.cu)
constexpr float kLn2 = 0x1.62e430p-1f;          // k5_gdn_decode.cu
constexpr float kL2Eps = 1e-6f;
constexpr float kSoftplusThreshold = 20.0f;

// ---- instruction list -------------------------------------------------------------------------------------------

enum Op : uint32_t {
    kOpGemm16 = 0,  // GemmIns `arg`, 16-column tiles
    kOpGemm8 = 1,   // GemmIns `arg`, 8-column tiles
    kOpCore = 2,    // conv + delta-rule decode + gated norm, kCoreItems items
    kOpConvQk = 3,  // the q / k channels' conv state, kConvQkItems items
    kOpBarrier = 4  // grid barrier: the next wave reads what this one wrote
};

struct Instr {
    uint32_t op;
    uint32_t arg;
    uint32_t items;
};

struct Seg {
    const uint4 *w;  // [n, k]
    bf16 *y;         // [rows, n]
    uint32_t n;
    uint32_t tile_begin;
};

struct GemmIns {
    Seg seg[kMaxSegs];
    uint32_t count;
    const uint4 *x;  // [rows, k]
    uint32_t chunks;  // k / 8
    uint32_t coherent_x;  // x is written earlier in this launch: read it through L2, never the non-coherent path
    uint32_t evict_first;
    uint32_t reverse_sum;  // Variant::kMutantSliceSum
    // Cross-op prefetch: the lines of these ranges, split evenly over the instruction's items.
    const char *pf[kMaxPrefetch];
    uint64_t pf_lines[kMaxPrefetch];
    uint32_t pf_count;
    uint64_t pf_total;
};

struct Core {
    const bf16 *mixed;  // [rows, 10240], wave 1
    const bf16 *z;      // [rows, 6144]
    const bf16 *b;      // [rows, 48]
    const bf16 *a;      // [rows, 48]
    const bf16 *conv_w;  // [10240, 4]
    const float *A_log;
    const bf16 *dt_bias;
    const bf16 *norm;
    bf16 *conv_state;
    bf16 *ssm_state;
    bf16 *conv_saves[kMaxRows - 1];
    bf16 *ssm_saves[kMaxRows - 1];
    bf16 *o;  // [rows, 6144]
    bf16 *y;  // [rows, 6144]
    uint32_t *heads;  // [48] arrival counters, zero between launches
    int norm_dim;     // 128, a runtime value as rmsnorm_kernel's `dim` (so `/ dim` stays a division)
};

constexpr uint32_t kMaxInstrs = 8;

struct Program {
    Instr ins[kMaxInstrs];
    uint32_t count;
    GemmIns gemm[2];
    Core core;
    uint32_t *sync;  // [2]: barrier arrivals, generation
    int rows;
};

// ---- memory helpers ---------------------------------------------------------------------------------------------

__device__ __forceinline__ uint64_t policy(bool evict_first) {
    uint64_t p;
    if (evict_first) {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    } else {
        asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
    }
    return p;
}

// Weights: no L1 allocation, L2 256-byte prefetch (k2_small_m.cu's load_stream), with an L2 eviction policy.
__device__ __forceinline__ uint4 load_weight(const uint4 *p, uint64_t pol) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p), "l"(pol));
    return v;
}

__device__ __forceinline__ void prefetch_l2(const void *p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

__device__ __forceinline__ bf16 ld_bf16(const bf16 *p) {
    return __ushort_as_bfloat16(__ldcg(reinterpret_cast<const unsigned short *>(p)));
}

__device__ __forceinline__ uint32_t ld_acquire(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// Every CTA arrives; the last resets the count and advances the generation the others wait on. Each thread's fence
// first, so every write of the wave is visible to every CTA after the barrier (intermediates are read through L2).
__device__ void grid_barrier(uint32_t *sync) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t gen = ld_acquire(sync + 1);
        if (atomicAdd(sync, 1u) == gridDim.x - 1) {
            atomicExch(sync, 0u);
            __threadfence();
            atomicAdd(sync + 1, 1u);
        } else {
            while (ld_acquire(sync + 1) == gen) {
            }
        }
        __threadfence();
    }
    __syncthreads();
}

// ---- GEMM: k2_small_m.cu small_m_kernel (cluster 1, Hi false) --------------------------------------------------

// k2_small_m.cu mma.
__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One tile of TN = 8 Q columns on the CTA: warp v runs slice v (steps v, v + 16, ... in increasing order, k2_small_m's
// slice_mma and step with rows 8..15 zero), the slices summed in slice order (reverse_sum: the mutant's reverse
// order), rounded to BF16.
template <uint32_t Q>
__device__ void gemm_tile(const GemmIns &g, uint32_t tile, uint32_t items, int rows,
                          float (&partial)[kWarps][kMaxRows][kTileMax]) {
    constexpr uint32_t TN = 8 * Q;
    const uint32_t tid = threadIdx.x;
    if (g.pf_total != 0) {
        const uint64_t per = (g.pf_total + items - 1) / items;
        const uint64_t end = (tile + 1) * per < g.pf_total ? (tile + 1) * per : g.pf_total;
        for (uint64_t line = tile * per + tid; line < end; line += kThreads) {
            uint64_t at = line;
            for (uint32_t r = 0; r < g.pf_count; ++r) {
                if (at < g.pf_lines[r]) {
                    prefetch_l2(g.pf[r] + at * kLine);
                    break;
                }
                at -= g.pf_lines[r];
            }
        }
    }
    const uint4 *w = g.seg[0].w;
    bf16 *y = g.seg[0].y;
    uint32_t n = g.seg[0].n;
    uint32_t begin = 0;
#pragma unroll
    for (uint32_t i = 1; i < kMaxSegs; ++i) {
        if (i < g.count && tile >= g.seg[i].tile_begin) {
            w = g.seg[i].w;
            y = g.seg[i].y;
            n = g.seg[i].n;
            begin = g.seg[i].tile_begin;
        }
    }
    const uint32_t col0 = (tile - begin) * TN;
    const uint32_t warp = tid / 32;
    const uint32_t lane = tid % 32;
    const uint32_t gq = lane / 4;
    const uint32_t t = lane % 4;
    const uint32_t chunks = g.chunks;
    const uint32_t steps = chunks / 4;
    const uint64_t pol = policy(g.evict_first != 0);
    bool col[Q];
    const uint4 *wc[Q];
#pragma unroll
    for (uint32_t q = 0; q < Q; ++q) {
        const uint32_t j = col0 + 8 * q + gq;
        col[q] = j < n;
        wc[q] = w + static_cast<size_t>(col[q] ? j : 0) * chunks + t;
    }
    const bool row0 = gq < static_cast<uint32_t>(rows);
    const uint4 *xr0 = g.x + static_cast<size_t>(row0 ? gq : 0) * chunks + t;
    const uint4 zero = make_uint4(0, 0, 0, 0);
    float acc[Q][4] = {};
    for (uint32_t s0 = warp; s0 < steps; s0 += kUnroll * kWarps) {
        uint4 xa[kUnroll], wv[kUnroll][Q];
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
            const uint32_t s = s0 + u * kWarps;
            const bool in = s < steps;
            const size_t off = static_cast<size_t>(s) * 4;
            xa[u] = in && row0 ? (g.coherent_x ? __ldcg(xr0 + off) : __ldg(xr0 + off)) : zero;
#pragma unroll
            for (uint32_t q = 0; q < Q; ++q) {
                wv[u][q] = in && col[q] ? load_weight(wc[q] + off, pol) : zero;
            }
        }
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
            if (s0 + u * kWarps < steps) {
#pragma unroll
                for (uint32_t q = 0; q < Q; ++q) {
                    mma(acc[q], xa[u].x, 0u, xa[u].y, 0u, wv[u][q].x, wv[u][q].y);
                    mma(acc[q], xa[u].z, 0u, xa[u].w, 0u, wv[u][q].z, wv[u][q].w);
                }
            }
        }
    }
    if (gq < kMaxRows) {
#pragma unroll
        for (uint32_t q = 0; q < Q; ++q) {
            partial[warp][gq][8 * q + 2 * t] = acc[q][0];
            partial[warp][gq][8 * q + 2 * t + 1] = acc[q][1];
        }
    }
    __syncthreads();
    for (uint32_t i = tid; i < static_cast<uint32_t>(rows) * TN; i += kThreads) {
        const uint32_t r = i / TN;
        const uint32_t j = i % TN;
        if (col0 + j < n) {
            float sum;
            if (g.reverse_sum) {
                sum = partial[kWarps - 1][r][j];
                for (uint32_t v = kWarps - 1; v-- > 0;) {
                    sum += partial[v][r][j];
                }
            } else {
                sum = partial[0][r][j];
                for (uint32_t v = 1; v < kWarps; ++v) {
                    sum += partial[v][r][j];
                }
            }
            y[static_cast<size_t>(r) * n + col0 + j] = __float2bfloat16_rn(sum);
        }
    }
    __syncthreads();  // partial is reused by the CTA's next tile
}

// ---- conv: k5_gdn.cu (kConvVariant 1) ---------------------------------------------------------------------------

__device__ __forceinline__ float ex2_conv(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float div_full(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float triton_exp(float x) { return ex2_conv(__fmul_rn(x, kLog2e)); }

__device__ __forceinline__ float conv_tap(float acc, bf16 x, bf16 w) {
    return __fadd_rn(acc, __bfloat162float(__hmul(x, w)));
}

__device__ __forceinline__ float silu_triton(float acc) {
    return div_full(acc, __fadd_rn(triton_exp(__fsub_rn(0.0f, acc)), 1.0f));
}

__device__ __forceinline__ bf16 conv_silu_token(bf16 (&win)[kConvState], bf16 xt, bf16 w0, bf16 w1, bf16 w2,
                                                bf16 w3) {
    float acc = 0.0f;
    acc = conv_tap(acc, win[0], w0);
    acc = conv_tap(acc, win[1], w1);
    acc = conv_tap(acc, win[2], w2);
    acc = conv_tap(acc, xt, w3);
    win[0] = win[1];
    win[1] = win[2];
    win[2] = xt;
    return __float2bfloat16_rn(silu_triton(acc));
}

// Channel c over the rows (conv1d_silu_rows_kernel): y[t] as float and the window after row t in win_after[t] (the
// value conv1d_silu_rows stores to conv_saves[t], or to the state after the last row).
__device__ __forceinline__ void conv_channel(const Core &k, int rows, int c, float (&y)[kMaxRows],
                                             bf16 (&win_after)[kMaxRows][kConvState]) {
    const bf16 w0 = k.conv_w[c * kConvWidth + 0];
    const bf16 w1 = k.conv_w[c * kConvWidth + 1];
    const bf16 w2 = k.conv_w[c * kConvWidth + 2];
    const bf16 w3 = k.conv_w[c * kConvWidth + 3];
    bf16 win[kConvState];
#pragma unroll
    for (int j = 0; j < kConvState; ++j) {
        win[j] = ld_bf16(k.conv_state + c * kConvState + j);
    }
#pragma unroll
    for (int t = 0; t < kMaxRows; ++t) {
        if (t < rows) {
            y[t] = __bfloat162float(conv_silu_token(win, ld_bf16(k.mixed + t * kConvDim + c), w0, w1, w2, w3));
#pragma unroll
            for (int j = 0; j < kConvState; ++j) {
                win_after[t][j] = win[j];
            }
        }
    }
}

// The q / k channels' state (wave 3, after every CORE item read the old one): the windows conv_channel would store.
__device__ void conv_qk_item(const Core &k, int rows, uint32_t item) {
    const int c = static_cast<int>(item * kThreads + threadIdx.x);
    bf16 win[kConvState];
#pragma unroll
    for (int j = 0; j < kConvState; ++j) {
        win[j] = ld_bf16(k.conv_state + c * kConvState + j);
    }
    for (int t = 0; t < rows; ++t) {
        win[0] = win[1];
        win[1] = win[2];
        win[2] = ld_bf16(k.mixed + t * kConvDim + c);
        bf16 *dst = t + 1 == rows ? k.conv_state : k.conv_saves[t];
#pragma unroll
        for (int j = 0; j < kConvState; ++j) {
            dst[c * kConvState + j] = win[j];
        }
    }
}

// ---- delta rule: k5_gdn_decode.cu (kDecodeVariant 1) ------------------------------------------------------------

__device__ __forceinline__ float ex2_approx(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float lg2_approx(float x) {
    float y;
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float rsqrt_approx(float x) {
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float exp_fast(float x) { return ex2_approx(__fmul_rn(x, kLog2e)); }

__device__ __forceinline__ float butterfly_sum(float s) {
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, offset));
    }
    return s;
}

__device__ __forceinline__ void load4(const bf16 *p, float out[4]) {
    const uint2 raw = __ldcg(reinterpret_cast<const uint2 *>(p));
    const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&raw.x));
    const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&raw.y));
    out[0] = lo.x;
    out[1] = lo.y;
    out[2] = hi.x;
    out[3] = hi.y;
}

__device__ __forceinline__ void store4(bf16 *p, const float in[4]) {
    uint2 raw;
    *reinterpret_cast<__nv_bfloat162 *>(&raw.x) = __floats2bfloat162_rn(in[0], in[1]);
    *reinterpret_cast<__nv_bfloat162 *>(&raw.y) = __floats2bfloat162_rn(in[2], in[3]);
    *reinterpret_cast<uint2 *>(p) = raw;
}

__device__ __forceinline__ float inv_l2(const float x[4]) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < kLaneElems; ++i) {
        s = __fmaf_rn(x[i], x[i], s);
    }
    return rsqrt_approx(__fadd_rn(butterfly_sum(s), kL2Eps));
}

__device__ __forceinline__ float row_dot(const float h[4], const float x[4]) {
    float even = __fmaf_rn(h[0], x[0], 0.0f);
    float odd = __fmaf_rn(h[1], x[1], 0.0f);
    even = __fmaf_rn(h[2], x[2], even);
    odd = __fmaf_rn(h[3], x[3], odd);
    return butterfly_sum(__fadd_rn(even, odd));
}

// decode_step with the lane's q and k slices and the warp's 4 v entries in registers (the BF16 conv outputs as float,
// the values load4 gives the original).
__device__ __forceinline__ void decode_step(float (&h)[kRowsPerWarp][kLaneElems], const float (&q_in)[kLaneElems],
                                            const float (&k_in)[kLaneElems], const float (&v)[kRowsPerWarp], bf16 a,
                                            bf16 b, float A_log, bf16 dt_bias, float scale,
                                            float (&o)[kRowsPerWarp]) {
    float q[kLaneElems];
    float k[kLaneElems];
#pragma unroll
    for (int i = 0; i < kLaneElems; ++i) {
        q[i] = q_in[i];
        k[i] = k_in[i];
    }
    const float inv_q = __fmul_rn(inv_l2(q), scale);
    const float inv_k = inv_l2(k);
#pragma unroll
    for (int i = 0; i < kLaneElems; ++i) {
        q[i] = __fmul_rn(inv_q, q[i]);
        k[i] = __fmul_rn(inv_k, k[i]);
    }
    const float x = __fadd_rn(__bfloat162float(a), __bfloat162float(dt_bias));
    const float log1p_exp = __fmul_rn(lg2_approx(__fadd_rn(exp_fast(x), 1.0f)), kLn2);
    const float use = x <= kSoftplusThreshold ? 1.0f : 0.0f;
    const float softplus = __fmaf_rn(log1p_exp, use, __fmul_rn(x, __fsub_rn(1.0f, use)));
    const float decay = exp_fast(-__fmul_rn(softplus, exp_fast(A_log)));
    const float beta = __frcp_rn(__fadd_rn(exp_fast(-__bfloat162float(b)), 1.0f));
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
        for (int i = 0; i < kLaneElems; ++i) {
            h[r][i] = __fmul_rn(h[r][i], decay);
        }
        const float vn = __fmul_rn(beta, __fsub_rn(v[r], row_dot(h[r], k)));
#pragma unroll
        for (int i = 0; i < kLaneElems; ++i) {
            h[r][i] = __fmaf_rn(k[i], vn, h[r][i]);
        }
        o[r] = row_dot(h[r], q);
    }
}

// ---- gated norm: k1_primitives.cu rmsnorm_kernel<true> at dim 128 (norm_threads(128) = one warp per row) ----------

constexpr int kVec = 8;

__device__ __forceinline__ void load8(const bf16 *p, float (&out)[kVec]) {
    const uint4 raw = __ldcg(reinterpret_cast<const uint4 *>(p));
    const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(h[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

__device__ __forceinline__ void store8(bf16 *p, const float (&in)[kVec]) {
    uint4 raw;
    __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(in[2 * i], in[2 * i + 1]);
    *reinterpret_cast<uint4 *>(p) = raw;
}

// Row (t, hv) of the gated norm on one warp (block_sum of a one-warp block: the shuffle tree, then 0 + partial[0]).
__device__ void norm_row(const Core &k, int t, int hv) {
    const int lane = static_cast<int>(threadIdx.x % 32);
    const int dim = k.norm_dim;
    const int64_t at = static_cast<int64_t>(t) * kValueDim + static_cast<int64_t>(hv) * kHeadDim;
    const bf16 *xr = k.o + at;
    float ss = 0.0f;
    for (int c = lane * kVec; c < dim; c += 32 * kVec) {
        float v[kVec];
        load8(xr + c, v);
#pragma unroll
        for (int i = 0; i < kVec; ++i) ss += v[i] * v[i];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    float total = 0.0f;
    total += ss;
    const float rstd = rsqrtf(total / static_cast<float>(dim) + k5::kNormEps);
    bf16 *yr = k.y + at;
    for (int c = lane * kVec; c < dim; c += 32 * kVec) {
        float v[kVec], w[kVec], g[kVec];
        load8(xr + c, v);
        load8(k.norm + c, w);
        load8(k.z + at + c, g);
#pragma unroll
        for (int i = 0; i < kVec; ++i) v[i] = v[i] * rstd * w[i] * (1.0f / (1.0f + expf(-g[i])));
        store8(yr + c, v);
    }
}

// ---- CORE item: value head item / 2, rows 64 (item % 2) .. + 63 ------------------------------------------------

__device__ void core_item(const Core &k, int rows, uint32_t item, uint32_t &last) {
    const int warp = static_cast<int>(threadIdx.x / 32);
    const int lane = static_cast<int>(threadIdx.x % 32);
    const int hv = static_cast<int>(item / kHalves);
    const int hk = hv / (kValueHeads / k5::kKeyHeads);
    const int v0 = static_cast<int>(item % kHalves) * kWarps * kRowsPerWarp + warp * kRowsPerWarp;
    const int k0 = lane * kLaneElems;
    float q[kMaxRows][kLaneElems];
    float kk[kMaxRows][kLaneElems];
    float v[kMaxRows][kRowsPerWarp];
    bf16 v_win[kLaneElems][kMaxRows][kConvState];
#pragma unroll
    for (int i = 0; i < kLaneElems; ++i) {
        float y[kMaxRows];
        bf16 unused[kMaxRows][kConvState];
        conv_channel(k, rows, hk * kHeadDim + k0 + i, y, unused);
#pragma unroll
        for (int t = 0; t < kMaxRows; ++t) q[t][i] = y[t];
        conv_channel(k, rows, kKeyDim + hk * kHeadDim + k0 + i, y, unused);
#pragma unroll
        for (int t = 0; t < kMaxRows; ++t) kk[t][i] = y[t];
        conv_channel(k, rows, 2 * kKeyDim + hv * kHeadDim + v0 + i, y, v_win[i]);
#pragma unroll
        for (int t = 0; t < kMaxRows; ++t) v[t][i] = y[t];
    }
    // The warp's v channels (v0 .. v0 + 3, owned by this item): every lane read their old state above; lane 0 then
    // stores the windows as conv1d_silu_rows does (the q / k channels' state waits for wave 3).
    __syncwarp();
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < kLaneElems; ++i) {
            const int c = 2 * kKeyDim + hv * kHeadDim + v0 + i;
#pragma unroll
            for (int t = 0; t < kMaxRows; ++t) {
                if (t < rows) {
                    bf16 *dst = t + 1 == rows ? k.conv_state : k.conv_saves[t];
#pragma unroll
                    for (int j = 0; j < kConvState; ++j) {
                        dst[c * kConvState + j] = v_win[i][t][j];
                    }
                }
            }
        }
    }
    const int64_t head = static_cast<int64_t>(hv) * kHeadDim * kHeadDim;
    float h[kRowsPerWarp][kLaneElems];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        load4(k.ssm_state + head + static_cast<int64_t>(v0 + r) * kHeadDim + k0, h[r]);
    }
    const float A_log = k.A_log[hv];
    const bf16 dt_bias = k.dt_bias[hv];
#pragma unroll
    for (int t = 0; t < kMaxRows; ++t) {
        if (t < rows) {
            float o[kRowsPerWarp];
            decode_step(h, q[t], kk[t], v[t], ld_bf16(k.a + t * kValueHeads + hv), ld_bf16(k.b + t * kValueHeads + hv),
                        A_log, dt_bias, kScale, o);
            if (lane == 0) {
                store4(k.o + static_cast<int64_t>(t) * kValueDim + hv * kHeadDim + v0, o);
            }
            bf16 *dst = (t + 1 == rows ? k.ssm_state : k.ssm_saves[t]) + head;
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                store4(dst + static_cast<int64_t>(v0 + r) * kHeadDim + k0, h[r]);
#pragma unroll
                for (int i = 0; i < kLaneElems; ++i) {
                    h[r][i] = __bfloat162float(__float2bfloat16_rn(h[r][i]));
                }
            }
        }
    }
    // The head's second arriving CTA (both halves' o written) runs its gated norm, a warp per row.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(k.heads + hv, 1u) == kHalves - 1 ? 1u : 0u;
        if (last != 0) {
            atomicExch(k.heads + hv, 0u);
            __threadfence();
        }
    }
    __syncthreads();
    if (last != 0 && warp < rows) {
        norm_row(k, warp, hv);
    }
    __syncthreads();  // `last` is rewritten by the CTA's next item
}

__global__ void __launch_bounds__(kThreads, 1) region_kernel(const __grid_constant__ Program p) {
    __shared__ float partial[kWarps][kMaxRows][kTileMax];
    __shared__ uint32_t last;
    const uint32_t grid = gridDim.x;
    uint32_t base = 0;  // the wave's item number of the instruction's first item
    for (uint32_t i = 0; i < p.count; ++i) {
        const Instr in = p.ins[i];
        if (in.op == kOpBarrier) {
            grid_barrier(p.sync);
            base = 0;
            continue;
        }
        // Items are numbered across the wave's instructions; CTA b takes the numbers congruent to b.
        for (uint32_t j = base + (blockIdx.x + grid - base % grid) % grid; j < base + in.items; j += grid) {
            const uint32_t item = j - base;
            switch (in.op) {
            case kOpGemm16:
                gemm_tile<2>(p.gemm[in.arg], item, in.items, p.rows, partial);
                break;
            case kOpGemm8:
                gemm_tile<1>(p.gemm[in.arg], item, in.items, p.rows, partial);
                break;
            case kOpCore:
                core_item(p.core, p.rows, item, last);
                break;
            default:
                conv_qk_item(p.core, p.rows, item);
                break;
            }
        }
        base += in.items;
    }
}

// ---- host -------------------------------------------------------------------------------------------------------

constexpr size_t kAlign = 256;

size_t round_up(size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

struct Workspace {
    uint32_t *sync;   // [2]
    uint32_t *heads;  // [48]
    bf16 *mixed;
    bf16 *z;
    bf16 *b;
    bf16 *a;
    bf16 *o;
    bf16 *y;
    size_t bytes;
};

Workspace carve(void *base, int rows) {
    const size_t r = static_cast<size_t>(rows);
    auto *at = static_cast<char *>(base);
    size_t off = 0;
    const auto take = [&](size_t bytes) {
        char *p = at == nullptr ? nullptr : at + off;
        off += round_up(bytes);
        return p;
    };
    Workspace ws{};
    ws.sync = reinterpret_cast<uint32_t *>(take((2 + kValueHeads) * sizeof(uint32_t)));
    ws.heads = ws.sync == nullptr ? nullptr : ws.sync + 2;
    ws.mixed = reinterpret_cast<bf16 *>(take(r * kConvDim * sizeof(bf16)));
    ws.z = reinterpret_cast<bf16 *>(take(r * kValueDim * sizeof(bf16)));
    ws.b = reinterpret_cast<bf16 *>(take(r * kValueHeads * sizeof(bf16)));
    ws.a = reinterpret_cast<bf16 *>(take(r * kValueHeads * sizeof(bf16)));
    ws.o = reinterpret_cast<bf16 *>(take(r * kValueDim * sizeof(bf16)));
    ws.y = reinterpret_cast<bf16 *>(take(r * kValueDim * sizeof(bf16)));
    ws.bytes = off;
    return ws;
}

bool aligned16(const void *p) { return p != nullptr && reinterpret_cast<uintptr_t>(p) % 16 == 0; }

uint64_t lines(size_t bytes) { return (bytes + kLine - 1) / kLine; }

}  // namespace

size_t workspace_bytes(int rows) { return rows >= 1 && rows <= kMaxRows ? carve(nullptr, rows).bytes : 0; }

cudaError_t workspace_init(void *workspace, size_t bytes, cudaStream_t stream) {
    if (workspace == nullptr || bytes == 0) return cudaErrorInvalidValue;
    return cudaMemsetAsync(workspace, 0, bytes, stream);
}

cere_status forward(const Args &args, void *workspace, size_t workspace_bytes, Variant variant, cudaStream_t stream) {
    const int rows = args.rows;
    const k5::LayerWeights &w = args.w;
    bool ok = rows >= 1 && rows <= kMaxRows && aligned16(args.hidden) && aligned16(w.in_proj_qkv) &&
              aligned16(w.in_proj_z) && aligned16(w.in_proj_b) && aligned16(w.in_proj_a) && aligned16(w.conv1d) &&
              w.A_log != nullptr && w.dt_bias != nullptr && aligned16(w.norm) && aligned16(w.out_proj) &&
              aligned16(args.conv_state) && aligned16(args.ssm_state) && aligned16(args.out) && workspace != nullptr &&
              reinterpret_cast<uintptr_t>(workspace) % kAlign == 0 &&
              (variant == Variant::kExact || variant == Variant::kMutantSliceSum);
    for (int t = 0; ok && t + 1 < rows; ++t) {
        ok = aligned16(args.conv_saves[t]) && aligned16(args.ssm_saves[t]);
    }
    if (!ok) return CERE_ERR_ARGUMENT;
    const Workspace ws = carve(workspace, rows);
    if (ws.bytes > workspace_bytes) return CERE_ERR_ARGUMENT;
    if (!kernels::k2s::supported(kHidden) || !kernels::k2s::supported(kValueDim) ||
        kernels::k2s::split_warps(kHidden) != kWarps || kernels::k2s::split_warps(kValueDim) != kWarps) {
        return CERE_ERR_UNSUPPORTED;
    }

    Program p{};
    p.rows = rows;
    p.sync = ws.sync;

    GemmIns &in = p.gemm[0];
    const bf16 *in_w[kMaxSegs] = {w.in_proj_qkv, w.in_proj_z, w.in_proj_b, w.in_proj_a};
    bf16 *in_y[kMaxSegs] = {ws.mixed, ws.z, ws.b, ws.a};
    const uint32_t in_n[kMaxSegs] = {kConvDim, kValueDim, kValueHeads, kValueHeads};
    uint32_t tiles = 0;
    for (uint32_t s = 0; s < kMaxSegs; ++s) {
        in.seg[s] = Seg{reinterpret_cast<const uint4 *>(in_w[s]), in_y[s], in_n[s], tiles};
        tiles += (in_n[s] + 15) / 16;
    }
    in.count = kMaxSegs;
    in.x = reinterpret_cast<const uint4 *>(args.hidden);
    in.chunks = kHidden / 8;
    in.evict_first = 1;
    in.reverse_sum = variant == Variant::kMutantSliceSum ? 1 : 0;
    const void *pf[kMaxPrefetch] = {w.out_proj, args.ssm_state, args.conv_state, w.conv1d};
    const size_t pf_bytes[kMaxPrefetch] = {size_t{kHidden} * kValueDim * sizeof(bf16),
                                           size_t{kValueHeads} * kHeadDim * kHeadDim * sizeof(bf16),
                                           size_t{kConvDim} * kConvState * sizeof(bf16),
                                           size_t{kConvDim} * kConvWidth * sizeof(bf16)};
    for (uint32_t r = 0; r < kMaxPrefetch; ++r) {
        in.pf[r] = static_cast<const char *>(pf[r]);
        in.pf_lines[r] = lines(pf_bytes[r]);
        in.pf_total += in.pf_lines[r];
    }
    in.pf_count = kMaxPrefetch;
    const uint32_t in_tiles = tiles;

    GemmIns &out = p.gemm[1];
    out.seg[0] = Seg{reinterpret_cast<const uint4 *>(w.out_proj), args.out, kHidden, 0};
    out.count = 1;
    out.x = reinterpret_cast<const uint4 *>(ws.y);
    out.chunks = kValueDim / 8;
    out.coherent_x = 1;
    out.reverse_sum = in.reverse_sum;
    const uint32_t out_tiles = (kHidden + 7) / 8;

    Core &c = p.core;
    c.mixed = ws.mixed;
    c.z = ws.z;
    c.b = ws.b;
    c.a = ws.a;
    c.conv_w = w.conv1d;
    c.A_log = w.A_log;
    c.dt_bias = w.dt_bias;
    c.norm = w.norm;
    c.conv_state = args.conv_state;
    c.ssm_state = args.ssm_state;
    for (int t = 0; t + 1 < rows; ++t) {
        c.conv_saves[t] = args.conv_saves[t];
        c.ssm_saves[t] = args.ssm_saves[t];
    }
    c.o = ws.o;
    c.y = ws.y;
    c.heads = ws.heads;
    c.norm_dim = kHeadDim;

    const Instr list[] = {{kOpGemm16, 0, in_tiles}, {kOpBarrier, 0, 0}, {kOpCore, 0, kCoreItems},
                          {kOpBarrier, 0, 0},       {kOpGemm8, 1, out_tiles}, {kOpConvQk, 0, kConvQkItems}};
    static_assert(sizeof(list) / sizeof(list[0]) <= kMaxInstrs, "instruction list");
    p.count = sizeof(list) / sizeof(list[0]);
    for (uint32_t i = 0; i < p.count; ++i) p.ins[i] = list[i];

    int device = 0;
    int sms = 0;
    int per_sm = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, region_kernel, kThreads, 0) != cudaSuccess ||
        sms <= 0 || per_sm <= 0) {
        return CERE_ERR_INTERNAL;
    }
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeCooperative;
    attr.val.cooperative = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(static_cast<unsigned>(sms * per_sm));
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;
    const cudaError_t e = cudaLaunchKernelEx(&config, region_kernel, p);
    const cudaError_t last = cudaGetLastError();
    return e == cudaSuccess && last == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

}  // namespace cere::ktest::k5r
