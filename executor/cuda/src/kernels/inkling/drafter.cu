// Inkling 2 Mini S9 phase B: the drafter's kernels (drafter.h). Compiled with -fmad=false: every multiply, add and
// fma is an explicit ref:: call (round to nearest even) and every reduction keeps the lane structure of
// executor/ref/inkling/drafter_ref.h / norm_ref.h; the comments name the reference step each piece reproduces.
#include "kernels/inkling/drafter.h"

#include <climits>

#include "inkling/drafter_ref.h"
#include "kernels/inkling/mix_common.cuh"
#include "kernels/inkling/pdl.cuh"

namespace cere::inkling::draft {
namespace {

namespace D = ref::drafter;
using mix_detail::kPerThread;
using mix_detail::kRowThreads;

constexpr unsigned kFullMask = 0xffffffffu;
constexpr int H = ref::kHidden;
constexpr int kTile = D::kTile;                     // 64 keys
constexpr int kSpan = D::kSpan;                     // 256 keys
constexpr int kSpansPerGroup = D::kGroupKeys / kSpan;  // 16
constexpr int kQueries = kBlock * (kHeads / kKvHeads);  // 24 query rows per (block, KV head)
constexpr int kMaxLocalSpans = 6;                   // [L - 1024, L + 8) touches at most 6 spans
constexpr int kAttnScratchSpans = 256;              // blocks x spans-per-pass partial records
constexpr int kRnnRow = 2 * kRank + 2 * H;          // 5120
constexpr int kMaxBlockDrafts = D::kGamma;          // 7
constexpr int64_t kRingElems = int64_t(kWindow) * kKvRow;

static_assert(kHeadDim == 4 * 32, "a head row is 4 elements per lane");
static_assert(kTile == 64 && kSpan == 4 * kTile, "S2's tile and span");

__device__ __forceinline__ float bfv(uint16_t x) { return ref::decode_bf16(x); }
__host__ __device__ __forceinline__ const uint16_t *u16(const bf16 *p) { return reinterpret_cast<const uint16_t *>(p); }
__host__ __device__ __forceinline__ uint16_t *u16(bf16 *p) { return reinterpret_cast<uint16_t *>(p); }
// 16 bytes global -> shared by cp.async (L2 only), zero-filled when !valid (src must still be a mapped address).
__device__ __forceinline__ void cp_async16(void *dst, const void *src, bool valid) {
    const uint32_t d = uint32_t(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
// All but the most recent cp.async group of this thread are complete.
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }

// ref::reduce_tree_<32> over one value per lane (xor butterfly: lanes i and i + off add the tree's pair, commuting,
// so every lane ends with the tree's result).
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) x = ref::add(x, __shfl_xor_sync(kFullMask, x, off));
    return x;
}
__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) x = ref::max(x, __shfl_xor_sync(kFullMask, x, off));
    return x;
}

// ---- mix_rmsnorm -----------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(kRowThreads) mix_kernel(MixArgs a) {
    pdl::enter();
    __shared__ float red[kRowThreads];
    const int64_t i = blockIdx.x;
    const int p = blockIdx.y, t = threadIdx.x;
    const int64_t r = a.src != nullptr ? int64_t(a.src[i]) : i;
    const uint16_t *x = u16(a.taps) + r * a.tap_ld;
    float al[D::kTaps];
#pragma unroll
    for (int k = 0; k < D::kTaps; ++k) al[k] = a.alpha[(a.a0 + p) * D::kTaps + k];
    float v[kPerThread];
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int idx = t + kRowThreads * j;
        float m = ref::mul(bfv(x[idx]), al[0]);
#pragma unroll
        for (int k = 1; k < D::kTaps; ++k) m = ref::fma(bfv(x[k * H + idx]), al[k], m);
        v[j] = m;
    }
    const float rr = mix_detail::row_rms_inv(v, D::kEps, red);
    const uint16_t *w = u16(a.w[p]);
    uint16_t *y = a.dst != nullptr ? u16(a.dst[i]) : u16(a.y) + p * a.plane_stride + i * a.ldy;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int idx = t + kRowThreads * j;
        y[idx] = ref::encode_bf16(ref::mul(bfv(w[idx]), ref::mul(v[j], rr)));
    }
}

// ---- residual add + RMSNorm --------------------------------------------------------------------------------------
__global__ void __launch_bounds__(kRowThreads) add_norm_kernel(AddNormArgs a) {
    pdl::enter();
    __shared__ float red[kRowThreads];
    const int64_t i = blockIdx.x;
    const int t = threadIdx.x;
    const int64_t r = (i / a.take) * a.group + a.skip + i % a.take;
    const uint16_t *res = u16(a.res_in) + r * H;
    float v[kPerThread];
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int idx = t + kRowThreads * j;
        float x = bfv(res[idx]);
        if (a.add_in != nullptr) x = D::bf16r(ref::add(bfv(u16(a.add_in)[r * H + idx]), x));
        v[j] = x;
    }
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) u16(a.res_out)[r * H + t + kRowThreads * j] = ref::encode_bf16(v[j]);
    const float rr = mix_detail::row_rms_inv(v, D::kEps, red);
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int idx = t + kRowThreads * j;
        u16(a.y)[i * H + idx] = ref::encode_bf16(ref::mul(bfv(u16(a.w)[idx]), ref::mul(v[j], rr)));
    }
}

__global__ void silu_mul_kernel(const uint16_t *g, const uint16_t *u, uint16_t *act, int64_t count) {
    pdl::enter();
    for (int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; e < count; e += int64_t(gridDim.x) * blockDim.x) {
        const float x = bfv(g[e]);
        const float s = ref::div(x, ref::add(1.0f, ref::exp(-x)));
        act[e] = ref::encode_bf16(ref::mul(s, bfv(u[e])));
    }
}

// ---- head norm + RoPE: one warp per head, lane l holding elements l + 32 m (m < 4) ------------------------------
// drafter_ref.h head_norm_rope: r over reduce_dot<32>, n = w * (x * r) (ctx: rounded to BF16), then RoPE with the
// staged row (element i < 64 pairs with i + 64: m = 0 with 2, m = 1 with 3, both in the same lane).
__device__ __forceinline__ void head_norm_rope_warp(const uint16_t *x, const uint16_t *w, const float *cs, bool ctx,
                                                    uint16_t *out, int lane) {
    float v[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) v[m] = bfv(x[lane + 32 * m]);
    float acc = 0.0f;
#pragma unroll
    for (int m = 0; m < 4; ++m) acc = ref::fma(v[m], v[m], acc);
    const float r = ref::rsqrt(ref::add(ref::div(warp_sum(acc), float(kHeadDim)), D::kEps));
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        v[m] = ref::mul(bfv(w[lane + 32 * m]), ref::mul(v[m], r));
        if (ctx) v[m] = D::bf16r(v[m]);
    }
    float y[4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int i = lane + 32 * h;
        const float c = cs[i], s = cs[D::kHalf + i], a = v[h], b = v[h + 2];
        y[h] = ref::sub(ref::mul(a, c), ref::mul(b, s));
        y[h + 2] = ref::add(ref::mul(b, c), ref::mul(a, s));
    }
#pragma unroll
    for (int m = 0; m < 4; ++m) out[lane + 32 * m] = ref::encode_bf16(y[m]);
}

constexpr int kNormHeads = kHeads + kKvHeads;  // 32 heads of a block row: 24 q then 8 k

__global__ void __launch_bounds__(256) qk_norm_rope_kernel(uint16_t *q, int64_t ldq, const uint16_t *q_norm,
                                                           uint16_t *k, int64_t ldk, const uint16_t *k_norm,
                                                           const float *cs, int64_t rows) {
    pdl::enter();
    const int64_t wi = int64_t(blockIdx.x) * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (wi >= rows * kNormHeads) return;
    const int64_t row = wi / kNormHeads;
    const int h = int(wi % kNormHeads);
    uint16_t *x = h < kHeads ? q + row * ldq + h * kHeadDim : k + row * ldk + (h - kHeads) * kHeadDim;
    head_norm_rope_warp(x, h < kHeads ? q_norm : k_norm, cs + row * kRopeRow, false, x, lane);
}

struct KNorms {
    const uint16_t *w[3];
};

// One warp per (row, layer, KV head): the ctx k head and the v head into the session's cache.
__global__ void __launch_bounds__(256) ctx_store_kernel(const uint16_t *k, const uint16_t *v, int64_t layer_stride,
                                                        KNorms kn, const float *cs, const CtxRow *tab, int64_t rows) {
    pdl::enter();
    const int64_t wi = int64_t(blockIdx.x) * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (wi >= rows * 3 * kKvHeads) return;
    const int64_t i = wi / (3 * kKvHeads);
    const int l = int(wi / kKvHeads % 3), g = int(wi % kKvHeads);
    const CtxRow c = tab[i];
    uint16_t *kd, *vd;
    if (l == 1) {
        kd = u16(c.full) + c.pos * kKvRow;
        vd = u16(c.full) + (c.max_ctx + c.pos) * kKvRow;
    } else {
        if (c.ring_write == 0) return;
        kd = u16(c.ring) + (l == 0 ? 0 : 2) * kRingElems + (c.pos % kWindow) * kKvRow;
        vd = kd + kRingElems;
    }
    const int64_t src = l * layer_stride + i * kKvRow + g * kHeadDim;
    head_norm_rope_warp(k + src, kn.w[l], cs + i * kRopeRow, true, kd + g * kHeadDim, lane);
#pragma unroll
    for (int m = 0; m < 4; ++m) vd[g * kHeadDim + lane + 32 * m] = v[src + lane + 32 * m];
}

// ---- block attention -------------------------------------------------------------------------------------------
// Per (block, KV head) the 24 query rows (row r, head 3 g + hh as query r * 3 + hh) see the same keys. Pass p of
// P spans: attn_span_kernel computes the partial of each span (attend's span: the tiles in order, tile_step), split
// over kQSplit CTAs of 8 queries each, then attn_fold_kernel folds the pass's spans in order into the group partial
// and, at a group change, the group into the total (attend's two folds), keeping (group, total) in the workspace
// between passes. The last pass divides.
struct Rec {
    float m, l, empty, pad;
    float acc[kHeadDim];
};
struct FoldState {
    float cur, gm, gl, ge, tm, tl, te, pad;
    float ga[kHeadDim];
    float ta[kHeadDim];
};

__host__ __device__ inline int spans_per_pass(int64_t blocks) {
    return int(kAttnScratchSpans / blocks) < 1 ? 1 : int(kAttnScratchSpans / blocks);
}

__device__ __forceinline__ void block_range(const BlockSeq &s, bool local, int64_t &lo, int64_t &hi) {
    lo = D::key_lo(local ? 0 : 1, s.L);
    hi = s.L + kBlock;
}

// The query's own key range under a mutant (the plain kernel: [lo, hi)).
__device__ __forceinline__ void query_range(const BlockSeq &s, bool local, AttnMutant mut, int r, int64_t lo,
                                            int64_t hi, int64_t &qlo, int64_t &qhi) {
    qlo = lo;
    qhi = hi;
    if (mut == AttnMutant::kCausal) qhi = s.L + r + 1;
    if (mut == AttnMutant::kRowWindow && local) {
        const int64_t w = s.L + r - (kWindow - 1);
        qlo = w > 0 ? w : 0;
    }
}

constexpr int kQSplit = 3;                       // CTAs per (span, block, KV head)
constexpr int kCtaQueries = kQueries / kQSplit;  // 8: one warp each in the softmax step
constexpr int kSpanThreads = 32 * kCtaQueries;   // 256 = 64 element pairs x 4 query pairs in the PV step
constexpr int kKsStride = kHeadDim + 8;          // padded K/V row: the 16-byte loads of 8 keys hit distinct banks
static_assert(kQueries % kQSplit == 0 && kCtaQueries % 2 == 0, "query pairs per CTA");
static_assert(kTile == 64, "lane scores keys lane and lane + 32 of a tile");

// Two keys' scores, entirely in this thread (one q load serves both): ref::reduce_dot<32>(q, k, 128) (chain t holds
// elements t + 32 c in ascending c, then reduce_tree_<32>) times kScale. q FP32 [128], k0 / k1 BF16 [128] (16-byte
// aligned).
__device__ __forceinline__ void key_scores(const float *q, const uint16_t *k0, const uint16_t *k1, float &s0,
                                           float &s1) {
    float c0[32], c1[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        c0[i] = 0.0f;
        c1[i] = 0.0f;
    }
#pragma unroll
    for (int c = 0; c < kHeadDim / 8; ++c) {
        const uint4 ka = *reinterpret_cast<const uint4 *>(k0 + c * 8);
        const uint4 kb = *reinterpret_cast<const uint4 *>(k1 + c * 8);
        const float4 qa = *reinterpret_cast<const float4 *>(q + c * 8);
        const float4 qb = *reinterpret_cast<const float4 *>(q + c * 8 + 4);
        const uint32_t wa[4] = {ka.x, ka.y, ka.z, ka.w}, wb[4] = {kb.x, kb.y, kb.z, kb.w};
        const float qq[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float va = __uint_as_float((e & 1) ? (wa[e / 2] & 0xffff0000u) : (wa[e / 2] << 16));
            const float vb = __uint_as_float((e & 1) ? (wb[e / 2] & 0xffff0000u) : (wb[e / 2] << 16));
            const int d = c * 8 + e;
            c0[d % 32] = ref::fma(qq[e], va, c0[d % 32]);
            c1[d % 32] = ref::fma(qq[e], vb, c1[d % 32]);
        }
    }
#pragma unroll
    for (int off = 16; off >= 1; off /= 2)
#pragma unroll
        for (int i = 0; i < off; ++i) {
            c0[i] = ref::add(c0[i], c0[i + off]);
            c1[i] = ref::add(c1[i], c1[i + off]);
        }
    s0 = ref::mul(c0[0], D::kScale);
    s1 = ref::mul(c1[0], D::kScale);
}

// Grid (P, kKvHeads * kQSplit, blocks), kSpanThreads threads. Per tile: warp w runs tile_step's softmax for query w
// of this CTA (lane t scores keys t and t + 32 itself, so no shuffle per key); then thread (pair dp, query pair qp)
// runs the PV chains of elements 2 dp, 2 dp + 1 of queries 2 qp, 2 qp + 1, keys in order (one fused multiply-add
// per kept key, as tile_step). MinCtas: 1 for local layers (at most 6 spans: registers over occupancy), 2 on the full
// layer (many spans: two CTAs per SM). Same arithmetic either way.
template <int MinCtas>
__global__ void __launch_bounds__(kSpanThreads, MinCtas) attn_span_kernel(BlockAttnArgs a, int64_t pass, int P) {
    pdl::enter();
    __shared__ __align__(16) uint16_t ks[kTile * kKsStride];
    __shared__ __align__(16) uint16_t vs[kTile * kHeadDim];
    __shared__ __align__(16) float qs[kCtaQueries][kHeadDim];
    __shared__ __align__(16) float ps[kCtaQueries][kTile];
    __shared__ float fs[kCtaQueries];
    __shared__ int modes[kCtaQueries];  // 0 tile skipped, 1 first tile (acc = 0), 2 rescale by fs
    __shared__ unsigned long long oks[kCtaQueries];
    const int b = blockIdx.z, g = blockIdx.y / kQSplit, part = blockIdx.y % kQSplit, idx = blockIdx.x;
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    const BlockSeq s = a.seqs[b];
    int64_t lo, hi;
    block_range(s, a.local, lo, hi);
    const int64_t sp = lo / kSpan + pass * P + idx;
    Rec *rec = reinterpret_cast<Rec *>(a.ws) + ((int64_t(b) * kKvHeads + g) * P + idx) * kQueries + part * kCtaQueries;
    if (sp > (hi - 1) / kSpan) {
        if (t < kCtaQueries) rec[t].empty = 1.0f;
        return;
    }
    // Softmax role: this warp's query.
    const int qi = part * kCtaQueries + warp, r = qi / 3, h = g * 3 + qi % 3;
    {
        const uint16_t *q = u16(a.q) + (int64_t(b) * kBlock + r) * kQRow + h * kHeadDim;
#pragma unroll
        for (int mm = 0; mm < 4; ++mm) qs[warp][lane + 32 * mm] = bfv(q[lane + 32 * mm]);
    }
    int64_t qlo, qhi;
    query_range(s, a.local, a.mutant, r, lo, hi, qlo, qhi);
    float m = 0.0f, l = 0.0f;
    bool empty = true;
    // PV role.
    const int dp = t % (kHeadDim / 2), qp = t / (kHeadDim / 2);
    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    // Live tiles [tb, te) in steps of kTile (attend skips the tiles below lo). K and V of a tile land by cp.async
    // (zero-filled outside [lo, hi)), one commit group each, in the order K(t), V(t), K(t + 1), ...: the next K
    // streams in during this tile's PV, the next V during the next tile's scores.
    const int64_t s0 = sp * kSpan;
    const int64_t te = s0 + kSpan < hi ? s0 + kSpan : hi;
    const int64_t tb = lo > s0 ? s0 + (lo - s0) / kTile * kTile : s0;
    const auto issue = [&](uint16_t *dst, int stride, bool is_v, int64_t t0) {
        if (t0 < te) {
            for (int e = t; e < kTile * kHeadDim / 8; e += kSpanThreads) {
                const int key = e / (kHeadDim / 8), c = e % (kHeadDim / 8);
                const int64_t j = t0 + key;
                const bool valid = j >= lo && j < hi;
                const uint16_t *src = u16(a.k);
                if (valid) {
                    const bool own = j >= s.L;
                    const int64_t row = own ? int64_t(b) * kBlock + (j - s.L) : (s.ring ? j % kWindow : j);
                    const bf16 *base = own ? (is_v ? a.v : a.k) : (is_v ? s.ctx_v : s.ctx_k);
                    src = u16(base) + row * kKvRow + g * kHeadDim + c * 8;
                }
                cp_async16(dst + key * stride + c * 8, src, valid);
            }
        }
        cp_async_commit();
    };
    issue(ks, kKsStride, false, tb);
    issue(vs, kHeadDim, true, tb);
    for (int64_t t0 = tb; t0 < te; t0 += kTile) {
        cp_async_wait1();  // K(t0) landed (V(t0) may be in flight)
        __syncthreads();
        {
            // drafter_ref.h tile_step, the softmax half, for query qi.
            const int64_t j0 = t0 + lane, j1 = t0 + lane + 32;
            const bool ok0 = j0 >= qlo && j0 < qhi, ok1 = j1 >= qlo && j1 < qhi;
            const unsigned long long okm =
                uint64_t(__ballot_sync(kFullMask, ok0)) | (uint64_t(__ballot_sync(kFullMask, ok1)) << 32);
            int mode = 0;
            float f = 0.0f;
            if (okm != 0) {
                float sc0, sc1;
                key_scores(qs[warp], ks + lane * kKsStride, ks + (lane + 32) * kKsStride, sc0, sc1);
                float x = -INFINITY;
                if (ok0) x = ref::max(x, sc0);
                if (ok1) x = ref::max(x, sc1);
                const float mt = warp_max(x);
                const float mn = empty ? mt : ref::max(m, mt);
                const float pr0 = ok0 ? ref::exp(ref::sub(sc0, mn)) : 0.0f;
                const float pr1 = ok1 ? ref::exp(ref::sub(sc1, mn)) : 0.0f;
                x = 0.0f;
                if (ok0) x = ref::add(x, pr0);
                if (ok1) x = ref::add(x, pr1);
                const float ts = warp_sum(x);
                if (empty) {
                    empty = false;
                    l = ts;
                    mode = 1;
                } else {
                    f = ref::exp(ref::sub(m, mn));
                    l = ref::add(ref::mul(l, f), ts);
                    mode = 2;
                }
                m = mn;
                ps[warp][lane] = pr0;
                ps[warp][lane + 32] = pr1;
            }
            if (lane == 0) {
                modes[warp] = mode;
                fs[warp] = f;
                oks[warp] = okm;
            }
        }
        __syncthreads();  // K(t0) consumed, the softmax outputs visible
        issue(ks, kKsStride, false, t0 + kTile);
        cp_async_wait1();  // V(t0) landed (K(t0 + 1) may be in flight)
        __syncthreads();
        // PV: tile_step's rescale (or the first tile's zero), then the kept keys in order.
        const uint32_t *vw = reinterpret_cast<const uint32_t *>(vs) + dp;  // key i's pair at vw[64 i]
        int mode[2];
        unsigned long long okq[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int pq = 2 * qp + k;
            mode[k] = modes[pq];
            okq[k] = oks[pq];
            if (mode[k] == 1) {
                acc[k][0] = 0.0f;
                acc[k][1] = 0.0f;
            } else if (mode[k] == 2) {
                acc[k][0] = ref::mul(acc[k][0], fs[pq]);
                acc[k][1] = ref::mul(acc[k][1], fs[pq]);
            }
        }
        if (mode[0] != 0 && mode[1] != 0 && okq[0] == ~0ull && okq[1] == ~0ull) {
            // every key kept by both queries: four independent chains
            const float *p0 = ps[2 * qp], *p1 = ps[2 * qp + 1];
#pragma unroll 4
            for (int i = 0; i < kTile; i += 4) {
                const float4 pa = *reinterpret_cast<const float4 *>(p0 + i);
                const float4 pb = *reinterpret_cast<const float4 *>(p1 + i);
                const float av[4] = {pa.x, pa.y, pa.z, pa.w}, bv[4] = {pb.x, pb.y, pb.z, pb.w};
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const uint32_t w = vw[(i + u) * (kHeadDim / 2)];
                    const float v0 = __uint_as_float(w << 16), v1 = __uint_as_float(w & 0xffff0000u);
                    acc[0][0] = ref::fma(av[u], v0, acc[0][0]);
                    acc[0][1] = ref::fma(av[u], v1, acc[0][1]);
                    acc[1][0] = ref::fma(bv[u], v0, acc[1][0]);
                    acc[1][1] = ref::fma(bv[u], v1, acc[1][1]);
                }
            }
        } else {
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                if (mode[k] == 0) continue;
                const float *p = ps[2 * qp + k];
                for (int i = 0; i < kTile; ++i) {
                    if (!((okq[k] >> i) & 1u)) continue;
                    const uint32_t w = vw[i * (kHeadDim / 2)];
                    acc[k][0] = ref::fma(p[i], __uint_as_float(w << 16), acc[k][0]);
                    acc[k][1] = ref::fma(p[i], __uint_as_float(w & 0xffff0000u), acc[k][1]);
                }
            }
        }
        __syncthreads();  // V(t0) and the softmax outputs consumed
        issue(vs, kHeadDim, true, t0 + kTile);
    }
#pragma unroll
    for (int k = 0; k < 2; ++k)
        *reinterpret_cast<float2 *>(rec[2 * qp + k].acc + 2 * dp) = make_float2(acc[k][0], acc[k][1]);
    if (lane == 0) {
        rec[warp].m = m;
        rec[warp].l = l;
        rec[warp].empty = empty ? 1.0f : 0.0f;
    }
}

// drafter_ref.h fold(a, b) for one element d of acc (m, l and the empty flags are the same in every thread).
__device__ __forceinline__ void fold_into(float &am, float &al, float &ae, float &aa, float bm, float bl, float be,
                                          float ba) {
    if (be != 0.0f) return;
    if (ae != 0.0f) {
        am = bm;
        al = bl;
        ae = 0.0f;
        aa = ba;
        return;
    }
    const float mn = ref::max(am, bm);
    const float ca = ref::exp(ref::sub(am, mn)), cb = ref::exp(ref::sub(bm, mn));
    al = ref::add(ref::mul(al, ca), ref::mul(bl, cb));
    aa = ref::add(ref::mul(aa, ca), ref::mul(ba, cb));
    am = mn;
}

__global__ void __launch_bounds__(kHeadDim) attn_fold_kernel(BlockAttnArgs a, int64_t pass, int P, bool last) {
    pdl::enter();
    const int qi = blockIdx.x, g = blockIdx.y, b = blockIdx.z, d = threadIdx.x;
    FoldState *st = reinterpret_cast<FoldState *>(static_cast<uint8_t *>(a.ws) +
                                                  size_t(a.blocks) * kKvHeads * P * kQueries * sizeof(Rec)) +
                    (int64_t(b) * kKvHeads + g) * kQueries + qi;
    float cur, gm, gl, ge, tm, tl, te, ga, ta;
    if (pass == 0) {
        cur = -1.0f;
        gm = gl = tm = tl = 0.0f;
        ge = te = 1.0f;
        ga = ta = 0.0f;
    } else {
        cur = st->cur;
        gm = st->gm;
        gl = st->gl;
        ge = st->ge;
        tm = st->tm;
        tl = st->tl;
        te = st->te;
        ga = st->ga[d];
        ta = st->ta[d];
    }
    const BlockSeq s = a.seqs[b];
    int64_t lo, hi;
    block_range(s, a.local, lo, hi);
    const Rec *rec = reinterpret_cast<const Rec *>(a.ws) + (int64_t(b) * kKvHeads + g) * P * kQueries + qi;
    // The pass's live spans [0, n), their records loaded kFoldBatch at a time ahead of the (ordered) folds.
    const int64_t live = (hi - 1) / kSpan - (lo / kSpan + pass * P) + 1;
    const int n = live < P ? int(live) : P;
    constexpr int kFoldBatch = 8;
    for (int i0 = 0; i0 < n; i0 += kFoldBatch) {
        float bm[kFoldBatch], bl[kFoldBatch], be[kFoldBatch], ba[kFoldBatch];
#pragma unroll
        for (int u = 0; u < kFoldBatch; ++u) {
            if (i0 + u >= n) break;
            const Rec &r = rec[int64_t(i0 + u) * kQueries];
            bm[u] = r.m;
            bl[u] = r.l;
            be[u] = r.empty;
            ba[u] = r.acc[d];
        }
#pragma unroll
        for (int u = 0; u < kFoldBatch; ++u) {
            if (i0 + u >= n) break;
            const int64_t sp = lo / kSpan + pass * P + i0 + u;
            const float grp = float(sp / kSpansPerGroup);
            if (grp != cur) {
                fold_into(tm, tl, te, ta, gm, gl, ge, ga);
                ge = 1.0f;
                cur = grp;
            }
            fold_into(gm, gl, ge, ga, bm[u], bl[u], be[u], ba[u]);
        }
    }
    if (last) {
        fold_into(tm, tl, te, ta, gm, gl, ge, ga);
        const int row = qi / 3, h = g * 3 + qi % 3;
        u16(a.o)[(int64_t(b) * kBlock + row) * kQRow + h * kHeadDim + d] = ref::encode_bf16(ref::div(ta, tl));
        return;
    }
    __syncthreads();  // every thread has read the shared scalars before thread 0 rewrites them
    st->ga[d] = ga;
    st->ta[d] = ta;
    if (d == 0) {
        st->cur = cur;
        st->gm = gm;
        st->gl = gl;
        st->ge = ge;
        st->tm = tm;
        st->tl = tl;
        st->te = te;
    }
}

// ---- Markov head -----------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) markov_input_kernel(const uint16_t *state, const uint16_t *w1,
                                                           const uint16_t *hid, int hid_rows, int j,
                                                           const bf16 *const *f_rnn, const uint32_t *anchors,
                                                           unsigned long long *keys, uint32_t *ids, uint16_t *inp,
                                                           bool assemble) {
    pdl::enter();
    const int b = blockIdx.x, t = threadIdx.x;
    const uint32_t prev = j == 0 ? anchors[b] : key_id(keys[b]);
    __syncthreads();  // every thread has read keys[b]
    if (j > 0 && t == 0) {
        ids[b * kMaxBlockDrafts + j - 1] = prev;
        keys[b] = 0;
    }
    if (!assemble) return;
    uint16_t *row = inp + int64_t(b) * kRnnRow;
    row[t] = state[b * kRank + t];
    row[kRank + t] = w1[int64_t(prev) * kRank + t];
    const uint16_t *h = hid + (int64_t(b) * hid_rows + j) * H;
    const uint16_t *f = u16(f_rnn[b]);
    for (int e = t; e < H; e += 256) {
        row[2 * kRank + e] = h[e];
        row[2 * kRank + H + e] = f[e];
    }
}

__global__ void __launch_bounds__(kRank) markov_gates_kernel(const float *acc, const uint16_t *bias, uint16_t *state,
                                                             uint16_t *latent) {
    pdl::enter();
    const int b = blockIdx.x, i = threadIdx.x;
    float p[3];
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float x = acc[int64_t(b) * kProj + c * kRank + i];
        p[c] = D::bf16r(bias != nullptr ? ref::add(x, bfv(bias[c * kRank + i])) : x);
    }
    // drafter_ref.h markov_gates.
    const float g = D::bf16r(ref::sigmoid(p[0]));
    const float kept = D::bf16r(ref::mul(g, bfv(state[b * kRank + i])));
    const float blend = D::bf16r(ref::mul(D::bf16r(ref::sub(1.0f, g)), D::bf16r(ref::tanh(p[1]))));
    state[b * kRank + i] = ref::encode_bf16(ref::add(kept, blend));
    latent[b * kRank + i] = ref::encode_bf16(ref::tanh(p[2]));
}

constexpr int kVocabWarps = 8;
constexpr int kVocabChunk = 8;  // ids per warp chunk
constexpr int kVocabChunks = (kStepVocab + kVocabChunk - 1) / kVocabChunk;
constexpr int kVocabGrid = (kVocabChunks + kVocabWarps - 1) / kVocabWarps;  // one chunk per warp

__device__ __forceinline__ unsigned long long make_key(float x, uint32_t v) {
    if (!(x > -INFINITY)) return 0ull;  // NaN (and -inf) never win, as markov_step's strict >
    if (x == 0.0f) x = 0.0f;           // -0 ties +0
    const uint32_t u = ref::float_bits(x);
    const uint32_t o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (static_cast<unsigned long long>(o) << 32) | (~v);
}

__device__ __forceinline__ unsigned long long umax64(unsigned long long a, unsigned long long b) { return a > b ? a : b; }

// Warp w takes one chunk of 8 ids; lane l loads elements l + 32 mm (mm < 8) of the chunk's fold rows straight from
// global memory (64 contiguous bytes per warp load). For block b and id v: the reduce_dot<64> of drafter_ref.h
// step_logit (lane l: chains l (elements l + 64 c) and l + 32 (elements l + 32 + 64 c), level 32 in registers, then
// the 32-lane tree), kept by lane v - v0; lanes < 8 run the epilogue (bias, base, softcap) for their own ids and the
// warp's max key goes to block b's slot in lane b % 32; one atomicMax per (CTA, block).
__global__ void __launch_bounds__(kVocabWarps * 32) markov_vocab_kernel(const uint16_t *latent,
                                                                        const uint16_t *__restrict__ fold_w,
                                                                        const uint16_t *__restrict__ fold_b,
                                                                        const uint16_t *base, int64_t ld_base,
                                                                        int blocks, unsigned long long *keys,
                                                                        uint16_t *logits, int64_t ld_logits) {
    extern __shared__ uint16_t lat[];  // [blocks][256]
    __shared__ unsigned long long best_sm[kVocabWarps][64];
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    pdl::enter();
    for (int e = t; e < blocks * kRank; e += blockDim.x) lat[e] = latent[e];
    __syncthreads();
    unsigned long long bk0 = 0, bk1 = 0;
    const int64_t ch = int64_t(blockIdx.x) * kVocabWarps + warp;
    if (ch < kVocabChunks) {
        const int64_t v0 = ch * kVocabChunk;
        const int nv = int(kStepVocab - v0 < kVocabChunk ? kStepVocab - v0 : kVocabChunk);
        uint16_t fr[kVocabChunk][8];
#pragma unroll
        for (int i = 0; i < kVocabChunk; ++i) {
#pragma unroll
            for (int mm = 0; mm < 8; ++mm) fr[i][mm] = i < nv ? fold_w[(v0 + i) * kRank + lane + 32 * mm] : uint16_t(0);
        }
        const int64_t v = v0 + lane;
        const float fb = lane < nv ? bfv(fold_b[v]) : 0.0f;
        for (int b = 0; b < blocks; ++b) {
            float a[8];
#pragma unroll
            for (int mm = 0; mm < 8; ++mm) a[mm] = bfv(lat[b * kRank + lane + 32 * mm]);
            float mine = 0.0f;
#pragma unroll
            for (int i = 0; i < kVocabChunk; ++i) {
                if (i >= nv) break;
                float c0 = 0.0f, c1 = 0.0f;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    c0 = ref::fma(a[2 * c], bfv(fr[i][2 * c]), c0);
                    c1 = ref::fma(a[2 * c + 1], bfv(fr[i][2 * c + 1]), c1);
                }
                const float d = warp_sum(ref::add(c0, c1));
                if (lane == i) mine = d;
            }
            unsigned long long key = 0;
            if (lane < nv) {
                const float bias = D::bf16r(ref::add(mine, fb));
                const float x = D::bf16r(ref::add(bfv(base[b * ld_base + v]), bias));
                const uint16_t lg = ref::encode_bf16(ref::mul(ref::tanh(ref::div(x, D::kSoftcap)), D::kSoftcap));
                if (logits != nullptr) logits[b * ld_logits + v] = lg;
                key = make_key(bfv(lg), uint32_t(v));
            }
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) key = umax64(key, __shfl_xor_sync(kFullMask, key, off));
            if (lane == (b & 31)) {
                if (b < 32) bk0 = umax64(bk0, key);
                else bk1 = umax64(bk1, key);
            }
        }
    }
    best_sm[warp][lane] = bk0;
    best_sm[warp][lane + 32] = bk1;
    __syncthreads();
    if (t < blocks) {
        unsigned long long k = 0;
        for (int w = 0; w < kVocabWarps; ++w) k = umax64(k, best_sm[w][t]);
        if (k != 0) atomicMax(keys + t, k);
    }
}

// ---- fold ------------------------------------------------------------------------------------------------------
constexpr int kFoldRows = 8;

__global__ void __launch_bounds__(kRank) fold_kernel(const uint16_t *lm, int64_t rows, const uint16_t *hw,
                                                     const uint16_t *hb, uint16_t *fold_w, uint16_t *fold_b) {
    pdl::enter();
    __shared__ uint16_t lms[kFoldRows][H];
    const int t = threadIdx.x;
    const int64_t r0 = int64_t(blockIdx.x) * kFoldRows;
    const int nr = int(rows - r0 < kFoldRows ? rows - r0 : kFoldRows);
    for (int e = t; e < kFoldRows * H; e += kRank) {
        const int r = e / H;
        lms[r][e % H] = r < nr ? lm[(r0 + r) * H + e % H] : uint16_t(0);
    }
    __syncthreads();
    double acc[kFoldRows];
#pragma unroll
    for (int r = 0; r < kFoldRows; ++r) acc[r] = 0.0;
    for (int k = 0; k < H; ++k) {
        const double w = double(bfv(hw[k * kRank + t]));
#pragma unroll
        for (int r = 0; r < kFoldRows; ++r) acc[r] = __dadd_rn(acc[r], __dmul_rn(double(bfv(lms[r][k])), w));
    }
    for (int r = 0; r < nr; ++r) fold_w[(r0 + r) * kRank + t] = ref::encode_bf16(__double2float_rn(acc[r]));
    if (t < nr) {
        double s = 0.0;
        for (int k = 0; k < H; ++k) s = __dadd_rn(s, __dmul_rn(double(bfv(lms[t][k])), double(bfv(hb[k]))));
        fold_b[r0 + t] = ref::encode_bf16(__double2float_rn(s));
    }
}

bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace

cudaError_t mix_rmsnorm(const MixArgs &a, cudaStream_t stream) {
    if (a.taps == nullptr || a.alpha == nullptr || a.rows < 1 || a.rows > INT_MAX || a.planes < 1 || a.planes > 3 ||
        a.a0 < 0 || a.a0 + a.planes > D::kAlphaRows || a.tap_ld < D::kTaps * H)
        return cudaErrorInvalidValue;
    for (int p = 0; p < a.planes; ++p)
        if (a.w[p] == nullptr) return cudaErrorInvalidValue;
    if (a.dst == nullptr ? (a.y == nullptr || a.ldy < H || (a.planes > 1 && a.plane_stride < a.rows * a.ldy))
                         : a.planes != 1)
        return cudaErrorInvalidValue;
    return pdl::launch(mix_kernel, dim3(unsigned(a.rows), unsigned(a.planes)), dim3(kRowThreads), 0, stream, a);
}

cudaError_t add_rmsnorm(const AddNormArgs &a, cudaStream_t stream) {
    if (a.res_in == nullptr || a.res_out == nullptr || a.w == nullptr || a.y == nullptr || a.rows < 1 ||
        a.rows > INT_MAX || a.take < 1 || a.group < a.take + a.skip || a.skip < 0 || a.add_in == a.res_out)
        return cudaErrorInvalidValue;
    return pdl::launch(add_norm_kernel, dim3(unsigned(a.rows)), dim3(kRowThreads), 0, stream, a);
}

cudaError_t silu_mul(const bf16 *g, const bf16 *u, bf16 *act, int64_t rows, int64_t n, cudaStream_t stream) {
    if (g == nullptr || u == nullptr || act == nullptr || rows < 1 || n < 1) return cudaErrorInvalidValue;
    const int64_t count = rows * n;
    const int64_t blocks = (count + 255) / 256;
    return pdl::launch(silu_mul_kernel, dim3(unsigned(blocks < 4096 ? blocks : 4096)), dim3(256), 0, stream,
                       u16(g), u16(u), u16(act), count);
}

cudaError_t qk_norm_rope(bf16 *q, int64_t ldq, const bf16 *q_norm, bf16 *k, int64_t ldk, const bf16 *k_norm,
                         const float *cs, int64_t rows, cudaStream_t stream) {
    if (q == nullptr || k == nullptr || q_norm == nullptr || k_norm == nullptr || cs == nullptr || rows < 1 ||
        rows > (INT_MAX / kNormHeads) || ldq < kQRow || ldk < kKvRow)
        return cudaErrorInvalidValue;
    const unsigned grid = unsigned((rows * kNormHeads + 7) / 8);
    return pdl::launch(qk_norm_rope_kernel, dim3(grid), dim3(256), 0, stream, u16(q), ldq, u16(q_norm), u16(k), ldk,
                       u16(k_norm), cs, rows);
}

cudaError_t ctx_store(const bf16 *k, const bf16 *v, int64_t layer_stride, const bf16 *const k_norm[3],
                      const float *cs, const CtxRow *rows_tab, int64_t rows, cudaStream_t stream) {
    if (k == nullptr || v == nullptr || cs == nullptr || rows_tab == nullptr || rows < 1 ||
        rows > INT_MAX / (3 * kKvHeads) || layer_stride < rows * kKvRow)
        return cudaErrorInvalidValue;
    KNorms kn{};
    for (int l = 0; l < 3; ++l) {
        if (k_norm[l] == nullptr) return cudaErrorInvalidValue;
        kn.w[l] = u16(k_norm[l]);
    }
    const unsigned grid = unsigned((rows * 3 * kKvHeads + 7) / 8);
    return pdl::launch(ctx_store_kernel, dim3(grid), dim3(256), 0, stream, u16(k), u16(v), layer_stride, kn, cs,
                       rows_tab, rows);
}

size_t block_attention_workspace_bytes(int64_t blocks) {
    if (blocks < 1) return 0;
    const size_t recs = size_t(blocks) * kKvHeads * size_t(spans_per_pass(blocks)) * kQueries * sizeof(Rec);
    const size_t states = size_t(blocks) * kKvHeads * kQueries * sizeof(FoldState);
    return (recs + states + 255) / 256 * 256;
}

cudaError_t block_attention(const BlockAttnArgs &a, cudaStream_t stream) {
    if (a.seqs == nullptr || a.blocks < 1 || a.blocks > 64 || a.q == nullptr || a.k == nullptr || a.v == nullptr ||
        a.o == nullptr || a.bound < kBlock || a.ws == nullptr || a.ws_bytes < block_attention_workspace_bytes(a.blocks) ||
        (reinterpret_cast<uintptr_t>(a.ws) & 255u) != 0 || !aligned16(a.k) || !aligned16(a.v))
        return cudaErrorInvalidValue;
    // Spans of one block: [lo / 256, (L + 7) / 256], at most (bound - 1) / 256 + 1 (and 6 on a local layer).
    int64_t spans = (a.bound - 1) / kSpan + 1;
    if (a.local && spans > kMaxLocalSpans) spans = kMaxLocalSpans;
    const int P = int(spans < spans_per_pass(a.blocks) ? spans : spans_per_pass(a.blocks));
    const int64_t passes = (spans + P - 1) / P;
    for (int64_t p = 0; p < passes; ++p) {
        cudaError_t e = a.local ? pdl::launch(attn_span_kernel<1>, dim3(unsigned(P), kKvHeads * kQSplit,
                                              unsigned(a.blocks)), dim3(kSpanThreads), 0, stream, a, p, P)
                                : pdl::launch(attn_span_kernel<2>, dim3(unsigned(P), kKvHeads * kQSplit,
                                              unsigned(a.blocks)), dim3(kSpanThreads), 0, stream, a, p, P);
        if (e != cudaSuccess) return e;
        e = pdl::launch(attn_fold_kernel, dim3(kQueries, kKvHeads, unsigned(a.blocks)), dim3(kHeadDim), 0, stream, a,
                        p, P, p + 1 == passes);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

cudaError_t markov_input(const bf16 *state, const bf16 *w1, const bf16 *hid, int hid_rows, int j,
                         const bf16 *const *f_rnn, const uint32_t *anchors, unsigned long long *keys, uint32_t *ids,
                         bf16 *inp, int64_t blocks, bool assemble, cudaStream_t stream) {
    if (blocks < 1 || blocks > 64 || j < 0 || j > kMaxBlockDrafts || (j == 0 && (anchors == nullptr || !assemble)) ||
        (j > 0 && (keys == nullptr || ids == nullptr)) ||
        (assemble && (state == nullptr || w1 == nullptr || hid == nullptr || hid_rows <= j || f_rnn == nullptr ||
                      inp == nullptr)))
        return cudaErrorInvalidValue;
    return pdl::launch(markov_input_kernel, dim3(unsigned(blocks)), dim3(256), 0, stream, u16(state), u16(w1), u16(hid),
                       hid_rows, j, f_rnn, anchors, keys, ids, u16(inp), assemble);
}

cudaError_t markov_gates(const float *acc, const bf16 *bias, bf16 *state, bf16 *latent, int64_t blocks,
                         cudaStream_t stream) {
    if (acc == nullptr || state == nullptr || latent == nullptr || blocks < 1 || blocks > INT_MAX)
        return cudaErrorInvalidValue;
    return pdl::launch(markov_gates_kernel, dim3(unsigned(blocks)), dim3(kRank), 0, stream, acc, u16(bias), u16(state),
                       u16(latent));
}

cudaError_t markov_vocab(const bf16 *latent, const bf16 *fold_w, const bf16 *fold_b, const bf16 *base,
                         int64_t ld_base, int64_t blocks, unsigned long long *keys, bf16 *logits, int64_t ld_logits,
                         cudaStream_t stream) {
    if (latent == nullptr || fold_w == nullptr || fold_b == nullptr || base == nullptr || keys == nullptr ||
        blocks < 1 || blocks > 64 || ld_base < kStepVocab || (logits != nullptr && ld_logits < kStepVocab) ||
        !aligned16(fold_w))
        return cudaErrorInvalidValue;
    const size_t smem = size_t(blocks) * kRank * sizeof(uint16_t);
    return pdl::launch(markov_vocab_kernel, dim3(kVocabGrid), dim3(kVocabWarps * 32), smem, stream, u16(latent),
                       u16(fold_w), u16(fold_b), u16(base), ld_base, int(blocks), keys, u16(logits), ld_logits);
}

cudaError_t fold(const bf16 *lm, int64_t rows, const bf16 *hw, const bf16 *hb, bf16 *fold_w, bf16 *fold_b,
                 cudaStream_t stream) {
    if (lm == nullptr || hw == nullptr || hb == nullptr || fold_w == nullptr || fold_b == nullptr || rows < 1 ||
        (rows + kFoldRows - 1) / kFoldRows > INT_MAX)
        return cudaErrorInvalidValue;
    return pdl::launch(fold_kernel, dim3(unsigned((rows + kFoldRows - 1) / kFoldRows)), dim3(kRank), 0, stream, u16(lm),
                       rows, u16(hw), u16(hb), u16(fold_w), u16(fold_b));
}

}  // namespace cere::inkling::draft
