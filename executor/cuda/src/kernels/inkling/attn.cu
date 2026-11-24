// Inkling 2 Mini attention kernels (attn.h). The order of every operation is executor/ref/inkling/attn_ref.h's;
// the comments name the reference step each piece reproduces. Compiled with -fmad=false; every multiply, add and
// fma is an explicit ref:: call (round to nearest even), every reduction keeps the reference's lane structure.
#include "kernels/inkling/attn.h"
#include "kernels/inkling/pdl.cuh"

#include <cstring>

#include <cooperative_groups.h>

namespace cere::inkling::attn {
namespace {

namespace R = ref::attn;
namespace cg = cooperative_groups;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kTile = R::kTile;
constexpr int kSpan = R::kSpan;
constexpr int kSpansPerGroup = R::kSpansPerGroup;
constexpr int kGroup = R::kGroup;
constexpr int kDRel = R::kDRel;
constexpr int kRelExtent = R::kRelExtent;
constexpr int kWarps = 8;                 // warps of a post block
constexpr int kThreads = kWarps * 32;
constexpr int kPartialFloats = 132;       // m, l, acc[128], 2 pad
constexpr int kDecodeSlots = kDecodeSpansPerPass + 1;  // span partials of a pass + the running total

__device__ __forceinline__ float lo_bf16(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hi_bf16(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ float bf(const bf16 &x) { return ref::decode_bf16(*reinterpret_cast<const uint16_t *>(&x)); }
__device__ __forceinline__ bf16 to_bf(float f) {
    const uint16_t u = ref::encode_bf16(f);
    bf16 r;
    memcpy(&r, &u, 2);
    return r;
}

// 16 bytes global -> shared without a register round trip (cp.async.cg: L2 only); wait_all before the barrier that
// publishes them.
__device__ __forceinline__ void cp_async16(void *dst, const void *src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                 "l"(src)
                 : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

// Halving tree over the 32 lanes (xor butterfly; every lane ends with lane 0's value, add and max commutative).
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off >= 1; off /= 2) v = ref::add(v, __shfl_xor_sync(kFullMask, v, off));
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off >= 1; off /= 2) v = ref::max(v, __shfl_xor_sync(kFullMask, v, off));
    return v;
}

// ---- 1. post ---------------------------------------------------------------------------------------------------
// A warp per 128-wide head: lane t holds elements t, t+32, t+64, t+96 (reduce_dot<32>'s lane t); the sum of squares
// is the lane's fma chain and the butterfly. scale: the lane's 4 scale values (nullptr: none).
__device__ void clip128(const bf16 *in, const float *scale, float threshold, float t, bf16 *out, int lane) {
    float x[4];
    float acc = 0.0f;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        x[m] = bf(in[lane + 32 * m]);
        if (scale) x[m] = ref::mul(x[m], scale[m]);
    }
#pragma unroll
    for (int m = 0; m < 4; ++m) acc = ref::fma(x[m], x[m], acc);
    const float c = R::clip_factor(warp_sum(acc), threshold);
    __syncwarp();  // every lane has read before any writes (in place)
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        float y = ref::decode_bf16(ref::encode_bf16(ref::mul(x[m], c)));
        if (t != 0.0f) y = ref::mul(y, t);
        out[lane + 32 * m] = to_bf(y);
    }
}

__device__ void rms128(const bf16 *in, bf16 *out, int lane) {
    float x[4];
    float acc = 0.0f;
#pragma unroll
    for (int m = 0; m < 4; ++m) x[m] = bf(in[lane + 32 * m]);
#pragma unroll
    for (int m = 0; m < 4; ++m) acc = ref::fma(x[m], x[m], acc);
    const float r = ref::rsqrt(ref::add(ref::div(warp_sum(acc), float(kHeadDim)), R::kVEps));
    __syncwarp();
#pragma unroll
    for (int m = 0; m < 4; ++m) out[lane + 32 * m] = to_bf(ref::mul(x[m], r));
}

// Two 16-wide heads per warp: lanes 0-15 head 2i, lanes 16-31 head 2i+1 (reduce_dot<16>: one element per lane,
// butterfly 8 .. 1 within the half warp). scale: the lane's scale value.
__device__ void clip16x2(const bf16 *in, float scale, float t, bf16 *out, int lane) {
    float x = ref::mul(bf(in[lane]), scale);
    float acc = ref::fma(x, x, 0.0f);
#pragma unroll
    for (int off = 8; off >= 1; off /= 2) acc = ref::add(acc, __shfl_xor_sync(kFullMask, acc, off));
    const float c = R::clip_factor(acc, R::kRelClip);
    __syncwarp();
    float y = ref::decode_bf16(ref::encode_bf16(ref::mul(x, c)));
    if (t != 0.0f) y = ref::mul(y, t);
    out[lane] = to_bf(y);
}

// One warp per head task (kPostTasks: the q heads, the rel head pairs, the k heads, the v heads), kPostBlocks blocks
// per row so all of a row's tasks run at once; a task's scale weights load before the PDL wait.
constexpr int kPostTasks = kHeads + kHeads / 2 + 2 * kKvHeads;
constexpr int kPostBlocks = (kPostTasks + kWarps - 1) / kWarps;

__global__ void __launch_bounds__(kThreads) post_kernel(PostArgs a) {
    pdl::trigger();
    const int64_t row = blockIdx.x;
    const int task = int(blockIdx.y) * kWarps + int(threadIdx.x) / 32, lane = threadIdx.x % 32;
    constexpr int kQ0 = 0, kR0 = kHeads, kK0 = kR0 + kHeads / 2, kV0 = kK0 + kKvHeads;
    const bool q = task < kR0, r = task >= kR0 && task < kK0, k = task >= kK0 && task < kV0;
    const bool v = task >= kV0 && task < kPostTasks;
    if ((q && !a.q_in) || (r && !a.r_in) || (k && !a.k_in) || (v && !a.v_in) || task >= kPostTasks) return;
    float sc[4];
    if (q) {
#pragma unroll
        for (int m = 0; m < 4; ++m) sc[m] = bf(a.q_scale[(task - kQ0) * kHeadDim + lane + 32 * m]);
    }
    if (r) sc[0] = bf(a.rel_scale[(task - kR0) * 32 + lane]);
    pdl::wait();
    float t = 0.0f;
    if ((q || r) && a.kind == kFull && a.mutant != R::kNoLogScale) t = R::tau(a.positions[row]);
    if (q) {
        const int h = task - kQ0;
        clip128(a.q_in + row * a.q_in_stride + h * kHeadDim, sc, R::kQkClip, t,
                a.q_out + row * a.q_out_stride + h * kHeadDim, lane);
    } else if (r) {
        const int h2 = task - kR0;
        clip16x2(a.r_in + row * a.r_in_stride + h2 * 32, sc[0], t, a.r_out + row * a.r_out_stride + h2 * 32, lane);
    } else if (k) {
        const int h = task - kK0;
        clip128(a.k_in + row * a.k_in_stride + h * kHeadDim, nullptr, R::kQkClip, 0.0f,
                a.k_out + row * a.k_out_stride + h * kHeadDim, lane);
    } else {
        const int h = task - kV0;
        rms128(a.v_in + row * a.v_in_stride + h * kHeadDim, a.v_out + row * a.v_out_stride + h * kHeadDim, lane);
    }
}

// ---- 2. attention ----------------------------------------------------------------------------------------------
// A partial of H heads (a lane's 4 dims each): the decode fold's levels (H = 1).
template <int H> struct Level {
    bool empty;
    float m[H], l[H], acc[H][4];
};

template <int H> __device__ __forceinline__ void level_reset(Level<H> &x) { x.empty = true; }

// fold(dst, src) (attn_ref.h fold).
template <int H> __device__ __forceinline__ void level_fold(Level<H> &dst, const Level<H> &src) {
    if (src.empty) return;
    if (dst.empty) {
        dst = src;
        return;
    }
#pragma unroll
    for (int h = 0; h < H; ++h) {
        float ca, cb;
        R::fold_coeffs(dst.m[h], dst.l[h], src.m[h], src.l[h], &dst.m[h], &dst.l[h], &ca, &cb);
#pragma unroll
        for (int u = 0; u < 4; ++u) dst.acc[h][u] = R::fold_acc(dst.acc[h][u], ca, src.acc[h][u], cb);
    }
}

// The epilogue (attn_ref.h epilogue) of query head qh of `row` from its total (m, l, the lane's 4 acc dims).
__device__ void epilogue_head(float m, float l, const float *acc, const AttnArgs &a, int64_t row, int qh, int lane) {
    float al = 1.0f, L = l;
    if (a.mutant != R::kNoSink) {
        const float z = bf(a.sink[row * a.sink_stride + qh]);
        const float mn = ref::max(m, z);
        al = ref::exp(ref::sub(m, mn));
        const float e = ref::exp(ref::sub(z, mn));
        L = ref::add(ref::mul(l, al), e);
    }
    const bf16 *gate = a.gate + row * a.gate_stride + qh * kHeadDim + 4 * lane;
    alignas(8) bf16 y[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const float x = a.mutant == R::kNoSink ? acc[u] : ref::mul(acc[u], al);
        const float o = ref::decode_bf16(ref::encode_bf16(ref::div(x, L)));
        y[u] = to_bf(ref::mul(o, ref::sigmoid(bf(gate[u]))));
    }
    *reinterpret_cast<uint2 *>(a.out + row * a.out_stride + qh * kHeadDim + 4 * lane) =
        *reinterpret_cast<const uint2 *>(y);
}

// rel_bias (attn_ref.h): reduce_dot<16> of the head's rel state r (16) with proj column P, BF16-rounded.
__device__ __forceinline__ float bias16(const float *r, const float *P) {
    float a[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) a[i] = ref::fma(r[i], P[i], 0.0f);
#pragma unroll
    for (int off = 8; off >= 1; off /= 2)
#pragma unroll
        for (int i = 0; i < off; ++i) a[i] = ref::add(a[i], a[i + off]);
    return ref::decode_bf16(ref::encode_bf16(a[0]));
}
__device__ __forceinline__ float bias16(const bf16 *r, const float *P) {
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = bf(r[i]);
    return bias16(x, P);
}

__device__ __forceinline__ void load_proj(const bf16 *proj, int64_t d, bool valid, int mutant, float *P, bool &has) {
    if (mutant == R::kBiasOffByOne) d += 1;
    has = valid && d >= 0 && d < kRelExtent;
#pragma unroll
    for (int i = 0; i < 16; ++i) P[i] = has ? bf(proj[i * kRelExtent + d]) : 0.0f;
}

// The K/V row of KV head g at position j (Keys::at in attn_ref.h).
__device__ __forceinline__ const bf16 *kv_row(const AttnArgs &a, const Seq &s, bool is_k, int64_t j, int g) {
    if (j >= s.pos0) {
        const int64_t r = s.row + (j - s.pos0);
        return (is_k ? a.k + r * a.k_stride : a.v + r * a.v_stride) + g * kHeadDim;
    }
    const int64_t slot = a.kind == kSliding ? j % kRingSlots : (s.slots ? s.slots[j] : j);
    return (is_k ? s.k_cache : s.v_cache) + slot * kKvRow + g * kHeadDim;
}

__device__ __forceinline__ int64_t row_lo(const AttnArgs &a, int64_t p) {
    return a.mutant == R::kNoWindow ? 0 : R::key_lo(a.kind, p);
}

// ---- prefill (PA) ----------------------------------------------------------------------------------------------
// Grid (row blocks of kPaRows rows, longest first; KV heads; sequences), kPaThreads threads, dynamic smem PaSmem.
// A block walks every key of its rows in attn_ref.h's 64-key tiles, ascending, and keeps each query head's span,
// group and total partials (attend's levels). Warp w owns rows 2w, 2w + 1 (query heads 6w .. 6w + 5) in every
// phase, so only the tile's K, V and proj columns are shared: one barrier after they are stored (K FP32, V BF16),
// one before they are overwritten, and in between each warp runs its three phases on its own (syncwarp):
//   1. scores: lane t scores the warp's 6 heads against keys t, t + 32. Each q.k is reduce_dot<32> exactly: lane
//      chain l = fma over elements l, l+32, l+64, l+96 from 0, then the halving tree, the 12 pairs' trees evaluated
//      depth first (qk_node); q and the K tile sit in smem as FP32 in lane-chain order (one float4 = one lane's 4
//      elements; the q reads are warp broadcasts, the K reads 32 distinct keys). Then mul by kScale and add the rel
//      bias (bias16's order; r and the tile's proj columns as FP32 in smem) or +0;
//   2. softmax: the warp's 6 heads, tile_step's max and sum with reduce_*<32>'s lanes (keys t, t+32), the
//      probabilities in place of the scores, the levels' (m, l, empty) in smem and the head's actions for the tile:
//      fold span -> group and group -> total (a copy into an empty level, or fold_coeffs), then first or rescale;
//   3. PV: lane t owns dims 4t .. 4t+3 of the warp's 6 heads in all three levels (registers), applies the actions
//      (fold_acc, mul) and runs acc[d] = fma(p_j, v_j[d], acc[d]) over each row's valid keys ascending.
// So every value is attn_ref.h's, in its order. The next tile's K and proj columns load into registers after the
// scores, its V after the softmax (its cache slots a tile earlier); PV widens V's BF16 exactly.
constexpr int kPaRows = 16;
constexpr int kPaThreads = 256;
constexpr int kPaWarps = kPaThreads / 32;
constexpr int kPaWarpRows = kPaRows / kPaWarps;  // rows of a warp
constexpr int kPaHeads = kPaRows * kGroup;       // query heads of a block
constexpr int kPaHeadsPerWarp = kPaHeads / kPaWarps;
constexpr int kPaKeys = kTile / 32;                    // keys per lane in the scores
constexpr int kPaPairs = kPaHeadsPerWarp * kPaKeys;    // (head, key) pairs per lane
// q, p and r are read as warp broadcasts (no padding); K (FP32) and V (BF16, converted in PV: the values are exact)
// pad 16 bytes per key, so 8 consecutive keys' 16-byte stores land on distinct banks.
constexpr int kPaQStride = kGroup * kHeadDim;
constexpr int kPaKStride = kHeadDim + 4;
constexpr int kPaVStride = kHeadDim + 8;
constexpr int kPaPStride = kTile;
constexpr int kPaRStride = kGroup * kDRel;
constexpr int kPaCols = kPaRows + kTile - 1;  // backward distances of a tile: column c = row - key + kTile - 1
constexpr int kPaColStride = kDRel + 4;
constexpr int kPaColLoads = (kPaCols * kDRel + kPaThreads - 1) / kPaThreads;
static_assert(kPaWarpRows * kPaWarps == kPaRows && kPaHeadsPerWarp == kPaWarpRows * kGroup && kPaKeys == 2,
              "a warp: 2 rows x 3 heads, a lane: keys t, t + 32");
static_assert(kHeadDim == 4 * 32, "PV: one lane per 4 dims");
static_assert(kTile * 4 == kPaThreads, "loads: 4 threads per key");

struct PaHead {
    float m[3], l[3];  // levels: span, group, total
    int empty[3];
    int64_t cur_span;
    // the tile's actions: folds 0 none, 1 copy, 2 coefficients (a, b); tile 0 none, 1 first, 2 rescale
    int sg, gt, tile;
    float sg_a, sg_b, gt_a, gt_b, alpha;
};

struct __align__(16) PaSmem {
    float q[kPaRows][kPaQStride];  // q[row][h * 128 + 4 l + m] = element 32 m + l of head h
    float k[kTile][kPaKStride];    // the same order per key
    bf16 v[kTile][kPaVStride];
    float p[kPaHeads][kPaPStride];  // scores, then probabilities
    float r[kPaRows][kPaRStride];
    float col[kPaCols][kPaColStride];  // col[c][i] = proj[i][cbase + c]
    PaHead head[kPaHeads];
};
static_assert(sizeof(PaSmem) <= 99 * 1024, "PaSmem fits sm_120's 99 KiB per-block shared memory opt-in");

__device__ __forceinline__ uint4 ldg16(const bf16 *p) {
    uint4 r;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    return r;
}

__device__ __forceinline__ float elem(const uint4 &u, int c) {
    const uint32_t w = c / 2 == 0 ? u.x : c / 2 == 1 ? u.y : c / 2 == 2 ? u.z : u.w;
    return c % 2 ? hi_bf16(w) : lo_bf16(w);
}

// Elements 32 m + 8 part + c (x[m], c < 8) of a 128-wide row to lane-chain order: dst = row + 32 part.
__device__ __forceinline__ void store_lanes(float *dst, const uint4 (&x)[4]) {
#pragma unroll
    for (int c = 0; c < 8; ++c)
        *reinterpret_cast<float4 *>(dst + 4 * c) =
            make_float4(elem(x[0], c), elem(x[1], c), elem(x[2], c), elem(x[3], c));
}

// The cache slot of tile t0's key t0 + tid % 64 (kv_row's; 0 when the key is outside [lo, hi] or in the call),
// read a tile ahead so the row loads below do not wait on it.
__device__ __forceinline__ int64_t pa_slot(const AttnArgs &a, const Seq &s, int64_t t0, int64_t lo, int64_t hi) {
    const int64_t j = t0 + threadIdx.x % kTile;
    if (j < lo || j > hi || j >= s.pos0) return 0;
    return a.kind == kSliding ? j % kRingSlots : (s.slots ? s.slots[j] : j);
}

// The K/V row of tile t0's key t0 + tid % 64 given its pa_slot (kv_row).
__device__ __forceinline__ const bf16 *pa_row(const AttnArgs &a, const Seq &s, bool is_k, int64_t t0, int64_t slot,
                                              int g) {
    const int64_t j = t0 + threadIdx.x % kTile;
    if (j >= s.pos0) {
        const int64_t r = s.row + (j - s.pos0);
        return (is_k ? a.k + r * a.k_stride : a.v + r * a.v_stride) + g * kHeadDim;
    }
    return (is_k ? s.k_cache : s.v_cache) + slot * kKvRow + g * kHeadDim;
}

// Tile t0's K rows (keys outside [lo, hi] skipped): thread (key tid % 64, part tid / 64) loads elements 32 m + 8 part
// .. + 7, m < 4 (8 consecutive lanes: 8 keys, so the smem stores are conflict free).
__device__ __forceinline__ void pa_load_k(uint4 (&x)[4], const AttnArgs &a, const Seq &s, int g, int64_t t0,
                                          int64_t slot, int64_t lo, int64_t hi) {
    const int64_t j = t0 + threadIdx.x % kTile;
    if (j < lo || j > hi) return;
    const bf16 *k = pa_row(a, s, true, t0, slot, g) + 8 * (threadIdx.x / kTile);
#pragma unroll
    for (int m = 0; m < 4; ++m) x[m] = ldg16(k + 32 * m);
}

// Tile t0's V rows: thread (key tid % 64, part tid / 64) loads dims 32 part + 8 u .. + 7, u < 4.
__device__ __forceinline__ void pa_load_v(uint4 (&x)[4], const AttnArgs &a, const Seq &s, int g, int64_t t0,
                                          int64_t slot, int64_t lo, int64_t hi) {
    const int64_t j = t0 + threadIdx.x % kTile;
    if (j < lo || j > hi) return;
    const bf16 *v = pa_row(a, s, false, t0, slot, g) + 32 * (threadIdx.x / kTile);
#pragma unroll
    for (int u = 0; u < 4; ++u) x[u] = ldg16(v + 8 * u);
}

__device__ __forceinline__ void pa_store_v(PaSmem &sm, const uint4 (&x)[4]) {
    bf16 *v = sm.v[threadIdx.x % kTile] + 32 * (threadIdx.x / kTile);
#pragma unroll
    for (int u = 0; u < 4; ++u) *reinterpret_cast<uint4 *>(v + 8 * u) = x[u];
}

// The proj columns d = cbase + c (c < kPaCols; 0 outside [0, 1024)): load index x = tid + u * kPaThreads is
// (i = x / kPaCols, c = x % kPaCols).
__device__ __forceinline__ void pa_load_cols(float (&x)[kPaColLoads], const bf16 *proj, int64_t cbase) {
#pragma unroll
    for (int u = 0; u < kPaColLoads; ++u) {
        const int idx = threadIdx.x + u * kPaThreads, i = idx / kPaCols;
        const int64_t d = cbase + idx % kPaCols;
        x[u] = idx < kPaCols * kDRel && d >= 0 && d < kRelExtent ? bf(proj[i * kRelExtent + d]) : 0.0f;
    }
}

__device__ __forceinline__ void pa_store_cols(PaSmem &sm, const float (&x)[kPaColLoads]) {
#pragma unroll
    for (int u = 0; u < kPaColLoads; ++u) {
        const int idx = threadIdx.x + u * kPaThreads;
        if (idx < kPaCols * kDRel) sm.col[idx % kPaCols][idx / kPaCols] = x[u];
    }
}

__device__ __forceinline__ float chain4(const float4 &q, const float4 &k) {
    float c = ref::fma(q.x, k.x, 0.0f);
    c = ref::fma(q.y, k.y, c);
    c = ref::fma(q.z, k.z, c);
    return ref::fma(q.w, k.w, c);
}

// reduce_dot<32>'s halving tree, depth first: node (T, S) = add(node(T, 2S), node(T + S, 2S)), node(T, 32) = lane
// chain T; o[x * kPaKeys + i] = node(0, 1) of the warp's head x (row x / 3 from q, head x % 3) and key k + 32 i.
template <int T, int S>
__device__ __forceinline__ void qk_node(const float *q, const float *k, float (&o)[kPaPairs]) {
    if constexpr (S == 32) {
        float4 kv[kPaKeys];
#pragma unroll
        for (int i = 0; i < kPaKeys; ++i) kv[i] = *reinterpret_cast<const float4 *>(k + i * 32 * kPaKStride + 4 * T);
#pragma unroll
        for (int x = 0; x < kPaHeadsPerWarp; ++x) {
            const float4 qv =
                *reinterpret_cast<const float4 *>(q + (x / kGroup) * kPaQStride + (x % kGroup) * kHeadDim + 4 * T);
#pragma unroll
            for (int i = 0; i < kPaKeys; ++i) o[x * kPaKeys + i] = chain4(qv, kv[i]);
        }
    } else {
        float b[kPaPairs];
        qk_node<T, 2 * S>(q, k, o);
        qk_node<T + S, 2 * S>(q, k, b);
#pragma unroll
        for (int x = 0; x < kPaPairs; ++x) o[x] = ref::add(o[x], b[x]);
    }
}

// kMutantLaneSum (ktest only): the 32 lane chains added in lane order instead of the tree.
__device__ __noinline__ void qk_lanes_in_order(const float *q, const float *k, float (&o)[kPaPairs]) {
    for (int x = 0; x < kPaPairs; ++x) {
        const int hx = x / kPaKeys;
        const float *qh = q + (hx / kGroup) * kPaQStride + (hx % kGroup) * kHeadDim;
        const float *kk = k + (x % kPaKeys) * 32 * kPaKStride;
        float acc = 0.0f;
        for (int t = 0; t < 32; ++t) {
            const float c = chain4(*reinterpret_cast<const float4 *>(qh + 4 * t),
                                   *reinterpret_cast<const float4 *>(kk + 4 * t));
            acc = t == 0 ? c : ref::add(acc, c);
        }
        o[x] = acc;
    }
}

// fold(dst, src) of the levels' scalars (src non-empty): action 1 (copy) into an empty dst, else 2 with fold_coeffs.
__device__ __forceinline__ void pa_fold(float &dm, float &dl, int &de, float sm_, float sl, int &act, float &ca,
                                        float &cb) {
    if (de) {
        dm = sm_;
        dl = sl;
        de = 0;
        act = 1;
    } else {
        R::fold_coeffs(dm, dl, sm_, sl, &dm, &dl, &ca, &cb);
        act = 2;
    }
}

// ref::exp without branches: the same operations in the same order, the special cases as selects. Every x takes
// the two-factor scale y * 2^h * 2^(n - h) (h = n / 2), which ref::exp uses outside n in [-126, 127]: inside it,
// y * 2^h is exact (h, n - h in [-75, 64] for x >= -103.97) and the second product rounds once, as y * 2^n does.
__device__ __forceinline__ float pa_exp(float x) {
    const float shifter = 12582912.0f;  // 1.5 * 2^23
    const float k = ref::sub(ref::fma(x, 1.44269504088896341f, shifter), shifter);
    float r = ref::fma(k, -0.693359375f, x);
    r = ref::fma(k, 2.12194440e-4f, r);
    float p = 1.9875691500e-4f;
    p = ref::fma(p, r, 1.3981999507e-3f);
    p = ref::fma(p, r, 8.3334519073e-3f);
    p = ref::fma(p, r, 4.1665795894e-2f);
    p = ref::fma(p, r, 1.6666665459e-1f);
    p = ref::fma(p, r, 5.0000001201e-1f);
    const float y = ref::add(ref::fma(ref::mul(r, r), p, r), 1.0f);
    const int n = int(k), h = n / 2;
    const float e = ref::mul(ref::mul(y, __uint_as_float(uint32_t(h + 127) << 23)),
                             __uint_as_float(uint32_t(n - h + 127) << 23));
    const float big = x > 88.72283905f ? INFINITY : e;
    return x != x ? x : x < -103.972084f ? 0.0f : big;
}

// Phase 2 for tile t0 (finish: the closing folds of attend instead, no tile). Lane k < kPaHeadsPerWarp keeps the
// books of the warp's head k (levels, the span -> group -> total folds, the tile's key range, the closing m and l);
// then the heads' tile_step max and sum run interleaved over all lanes (their butterflies and exps overlap), each
// head's arithmetic tile_step's, in its order.
__device__ __forceinline__ void pa_softmax(PaSmem &sm, const AttnArgs &a, const Seq &s, int64_t i0, int64_t t0,
                                           bool finish, int w, int lane) {
    constexpr int H = kPaHeadsPerWarp;
    const int64_t sp = t0 / kSpan;
    const bool mine = lane < H;
    const int qh = w * H + (mine ? lane : 0);
    PaHead &x = sm.head[qh];
    const int64_t i = i0 + qh / kGroup, p = s.pos0 + i, lo = row_lo(a, p);
    const bool active = mine && i < s.rows && (finish || (t0 + kTile - 1 >= lo && t0 <= p));
    float m0 = x.m[0], l0 = x.l[0];
    int ez = x.empty[0], info = 0;  // info: bit 0 active, bit 1 level 0 empty, ja - t0 << 8, jb - t0 << 16
    {
        float m1 = x.m[1], l1 = x.l[1], m2 = x.m[2], l2 = x.l[2];
        int e1 = x.empty[1], e2 = x.empty[2], sg = 0, gt = 0;
        float sga = 0.0f, sgb = 0.0f, gta = 0.0f, gtb = 0.0f;
        if (active) {
            const int64_t cur = x.cur_span;
            if (!ez && (finish || sp != cur)) {
                pa_fold(m1, l1, e1, m0, l0, sg, sga, sgb);
                ez = 1;
                if (finish || sp / kSpansPerGroup != cur / kSpansPerGroup) {
                    pa_fold(m2, l2, e2, m1, l1, gt, gta, gtb);
                    e1 = 1;
                }
            } else if (finish && !e1) {
                pa_fold(m2, l2, e2, m1, l1, gt, gta, gtb);
                e1 = 1;
            }
            x.m[1] = m1, x.l[1] = l1, x.m[2] = m2, x.l[2] = l2;
            x.empty[0] = ez, x.empty[1] = e1, x.empty[2] = e2;
            const int64_t ja = lo > t0 ? lo : t0, jb = p < t0 + kTile - 1 ? p : t0 + kTile - 1;
            info = 1 | ez << 1 | int(ja - t0) << 8 | int(jb - t0) << 16;
        }
        if (mine) {
            x.sg = sg, x.gt = gt, x.tile = 0;
            x.sg_a = sga, x.sg_b = sgb, x.gt_a = gta, x.gt_b = gtb, x.alpha = 0.0f;
        }
    }
    if (finish) {
        __syncwarp();
        return;
    }
    // tile_step of each head: lane t takes keys t, t + 32; reduce_max<32>'s and reduce_sum<32>'s halving trees
    // (warp_max / warp_sum), the heads interleaved.
    float s0[H], s1[H], mx[H], ts[H];
    bool v0[H], v1[H];
#pragma unroll
    for (int k = 0; k < H; ++k) {
        const int ik = __shfl_sync(kFullMask, info, k);
        const int ja = ik >> 8 & 255, jb = ik >> 16 & 255;
        v0[k] = (ik & 1) && lane >= ja && lane <= jb;
        v1[k] = (ik & 1) && lane + 32 >= ja && lane + 32 <= jb;
        s0[k] = sm.p[w * H + k][lane], s1[k] = sm.p[w * H + k][lane + 32];
        mx[k] = -INFINITY;
        if (v0[k]) mx[k] = ref::max(mx[k], s0[k]);
        if (v1[k]) mx[k] = ref::max(mx[k], s1[k]);
    }
#pragma unroll
    for (int off = 16; off >= 1; off /= 2)
#pragma unroll
        for (int k = 0; k < H; ++k) mx[k] = ref::max(mx[k], __shfl_xor_sync(kFullMask, mx[k], off));
#pragma unroll
    for (int k = 0; k < H; ++k) {
        const int ik = __shfl_sync(kFullMask, info, k);
        const float m0k = __shfl_sync(kFullMask, m0, k);
        const float mn = (ik & 2) ? mx[k] : ref::max(m0k, mx[k]);
        const float p0 = v0[k] ? pa_exp(ref::sub(s0[k], mn)) : 0.0f;
        const float p1 = v1[k] ? pa_exp(ref::sub(s1[k], mn)) : 0.0f;
        ts[k] = 0.0f;
        if (v0[k]) ts[k] = ref::add(ts[k], p0);
        if (v1[k]) ts[k] = ref::add(ts[k], p1);
        mx[k] = mn;
        if (ik & 1) sm.p[w * H + k][lane] = p0, sm.p[w * H + k][lane + 32] = p1;
    }
#pragma unroll
    for (int off = 16; off >= 1; off /= 2)
#pragma unroll
        for (int k = 0; k < H; ++k) ts[k] = ref::add(ts[k], __shfl_xor_sync(kFullMask, ts[k], off));
    // lane k closes head k's tile: first (l = sum) or rescale (alpha = exp(m - mn), l = l alpha + sum).
    float mn = mx[0], sum = ts[0];
#pragma unroll
    for (int k = 1; k < H; ++k)
        if (lane == k) mn = mx[k], sum = ts[k];
    if (active) {
        const float al = ez ? 0.0f : pa_exp(ref::sub(m0, mn));
        x.m[0] = mn, x.l[0] = ez ? sum : ref::add(ref::mul(l0, al), sum), x.empty[0] = 0, x.cur_span = sp;
        x.tile = ez ? 1 : 2, x.alpha = al;
    }
    __syncwarp();
}

// Phase 3's actions on one head's 4 dims of the three levels.
__device__ __forceinline__ void pa_apply(const PaHead &x, float (&sa)[4], float (&ga)[4], float (&ta)[4]) {
    const int sg = x.sg, gt = x.gt, tile = x.tile;
    if (sg == 1) {
#pragma unroll
        for (int u = 0; u < 4; ++u) ga[u] = sa[u];
    } else if (sg == 2) {
        const float ca = x.sg_a, cb = x.sg_b;
#pragma unroll
        for (int u = 0; u < 4; ++u) ga[u] = R::fold_acc(ga[u], ca, sa[u], cb);
    }
    if (gt == 1) {
#pragma unroll
        for (int u = 0; u < 4; ++u) ta[u] = ga[u];
    } else if (gt == 2) {
        const float ca = x.gt_a, cb = x.gt_b;
#pragma unroll
        for (int u = 0; u < 4; ++u) ta[u] = R::fold_acc(ta[u], ca, ga[u], cb);
    }
    if (tile == 1) {
#pragma unroll
        for (int u = 0; u < 4; ++u) sa[u] = 0.0f;
    } else if (tile == 2) {
        const float al = x.alpha;
#pragma unroll
        for (int u = 0; u < 4; ++u) sa[u] = ref::mul(sa[u], al);
    }
}

__global__ void __launch_bounds__(kPaThreads, 1) prefill_kernel(AttnArgs a) {
    pdl::enter();
    extern __shared__ __align__(16) unsigned char pa_raw[];
    PaSmem &sm = *reinterpret_cast<PaSmem *>(pa_raw);
    const Seq s = a.seqs[blockIdx.z];
    const int64_t i0 = int64_t(gridDim.x - 1 - blockIdx.x) * kPaRows;  // the rows with the most keys first
    if (i0 >= s.rows) return;
    const int g = blockIdx.y, tid = threadIdx.x, w = tid / 32, lane = tid % 32;
    const int64_t P0 = s.pos0 + i0, blo = row_lo(a, P0);
    const int64_t bhi = s.pos0 + (i0 + kPaRows < s.rows ? i0 + kPaRows : s.rows) - 1;
    const int shift = a.mutant == R::kBiasOffByOne ? 1 : 0;
    // The warp's rows r0 + r: position, first key, in the call.
    const int r0 = kPaWarpRows * w;
    int64_t pr[kPaWarpRows], lor[kPaWarpRows];
    bool ract[kPaWarpRows];
#pragma unroll
    for (int r = 0; r < kPaWarpRows; ++r) {
        pr[r] = P0 + r0 + r;
        lor[r] = row_lo(a, pr[r]);
        ract[r] = i0 + r0 + r < s.rows;
    }

    // q (lane-chain order) and r of the block's rows, FP32; task (row, head, part) takes elements 32 m + 8 part.
    for (int t = tid; t < kPaRows * kGroup * 4; t += kPaThreads) {
        const int rr = t / (kGroup * 4), h = t / 4 % kGroup, part = t % 4;
        if (i0 + rr >= s.rows) continue;
        const bf16 *src = a.q + (s.row + i0 + rr) * a.q_stride + (g * kGroup + h) * kHeadDim + 8 * part;
        uint4 x[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) x[m] = *reinterpret_cast<const uint4 *>(src + 32 * m);
        store_lanes(&sm.q[rr][h * kHeadDim + 32 * part], x);
    }
    for (int t = tid; t < kPaRows * kGroup * kDRel; t += kPaThreads) {
        const int rr = t / (kGroup * kDRel), c = t % (kGroup * kDRel);
        if (i0 + rr < s.rows) sm.r[rr][c] = bf(a.r[(s.row + i0 + rr) * a.r_stride + g * kGroup * kDRel + c]);
    }
    if (tid < kPaHeads) {
        PaHead &x = sm.head[tid];
        x.empty[0] = x.empty[1] = x.empty[2] = 1;
        x.cur_span = -1;
    }

    float sa[kPaHeadsPerWarp][4] = {}, ga[kPaHeadsPerWarp][4] = {}, ta[kPaHeadsPerWarp][4] = {};
    uint4 kr[4] = {}, vr[4] = {};
    float cr[kPaColLoads] = {};
    const int64_t first = blo / kTile * kTile;
    {
        const int64_t slot = pa_slot(a, s, first, blo, bhi);
        pa_load_k(kr, a, s, g, first, slot, blo, bhi);
        pa_load_v(vr, a, s, g, first, slot, blo, bhi);
    }
    if (P0 - first - (kTile - 1) + shift < kRelExtent) pa_load_cols(cr, a.rel_proj, P0 - first - (kTile - 1) + shift);
    for (int64_t t0 = first; t0 <= bhi; t0 += kTile) {
        const int64_t cbase = P0 - t0 - (kTile - 1) + shift;
        const bool bias = cbase < kRelExtent;
        const int64_t nslot = pa_slot(a, s, t0 + kTile, blo, bhi);  // the next tile's, a tile ahead
        store_lanes(sm.k[tid % kTile] + 32 * (tid / kTile), kr);
        pa_store_v(sm, vr);
        if (bias) pa_store_cols(sm, cr);
        __syncthreads();

        // 1. scores
        float sc[kPaPairs];
        if (a.mutant == kMutantLaneSum) {
            float t[kPaPairs];  // the out-of-line mutant writes local memory; sc itself stays in registers
            qk_lanes_in_order(sm.q[r0], sm.k[lane], t);
#pragma unroll
            for (int x = 0; x < kPaPairs; ++x) sc[x] = t[x];
        } else {
            qk_node<0, 1>(sm.q[r0], sm.k[lane], sc);
        }
#pragma unroll
        for (int x = 0; x < kPaPairs; ++x) sc[x] = ref::mul(sc[x], R::kScale);
        if (bias) {
#pragma unroll
            for (int x = 0; x < kPaHeadsPerWarp; ++x) {
                const int r = x / kGroup, h = x % kGroup;
                float rh[kDRel];
#pragma unroll
                for (int c = 0; c < kDRel; c += 4) {
                    const float4 f = *reinterpret_cast<const float4 *>(&sm.r[r0 + r][h * kDRel + c]);
                    rh[c] = f.x, rh[c + 1] = f.y, rh[c + 2] = f.z, rh[c + 3] = f.w;
                }
#pragma unroll
                for (int i = 0; i < kPaKeys; ++i) {
                    const int kk = lane + 32 * i;
                    const int64_t j = t0 + kk, d = pr[r] - j + shift;
                    const bool has = ract[r] && j >= lor[r] && j <= pr[r] && d >= 0 && d < kRelExtent;
                    sc[x * kPaKeys + i] =
                        ref::add(sc[x * kPaKeys + i], has ? bias16(rh, sm.col[kTile - 1 + r0 + r - kk]) : 0.0f);
                }
            }
        } else {
#pragma unroll
            for (int x = 0; x < kPaPairs; ++x) sc[x] = ref::add(sc[x], 0.0f);
        }
#pragma unroll
        for (int x = 0; x < kPaHeadsPerWarp; ++x)
#pragma unroll
            for (int i = 0; i < kPaKeys; ++i) sm.p[w * kPaHeadsPerWarp + x][lane + 32 * i] = sc[x * kPaKeys + i];
        // The next tile's K and its proj columns now, its V after the softmax (spreads the loads' issue).
        const bool more = t0 + kTile <= bhi;
        if (more) {
            pa_load_k(kr, a, s, g, t0 + kTile, nslot, blo, bhi);
            if (cbase - kTile < kRelExtent) pa_load_cols(cr, a.rel_proj, cbase - kTile);
        }
        __syncwarp();

        // 2. softmax
        pa_softmax(sm, a, s, i0, t0, false, w, lane);
        if (more) pa_load_v(vr, a, s, g, t0 + kTile, nslot, blo, bhi);

        // 3. PV: each row's keys ja .. jb of the tile (none: ja > jb).
#pragma unroll
        for (int x = 0; x < kPaHeadsPerWarp; ++x) pa_apply(sm.head[w * kPaHeadsPerWarp + x], sa[x], ga[x], ta[x]);
        int ja[kPaWarpRows], jb[kPaWarpRows];
        bool whole = true;
#pragma unroll
        for (int r = 0; r < kPaWarpRows; ++r) {
            const bool on = ract[r] && t0 + kTile - 1 >= lor[r] && t0 <= pr[r];
            ja[r] = on ? int((lor[r] > t0 ? lor[r] : t0) - t0) : kTile;
            jb[r] = on ? int((pr[r] < t0 + kTile - 1 ? pr[r] : t0 + kTile - 1) - t0) : -1;
            whole = whole && ja[r] == 0 && jb[r] == kTile - 1;
        }
        const float *pw = sm.p[w * kPaHeadsPerWarp];
        const bf16 *vl = sm.v[0] + 4 * lane;  // dims 4 lane .. + 3 (8 bytes; a warp reads one 256-byte row)
        if (whole) {
#pragma unroll 2
            for (int j4 = 0; j4 < kTile; j4 += 4) {
                float4 pv[kPaHeadsPerWarp];
#pragma unroll
                for (int x = 0; x < kPaHeadsPerWarp; ++x)
                    pv[x] = *reinterpret_cast<const float4 *>(pw + x * kPaPStride + j4);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const uint2 v = *reinterpret_cast<const uint2 *>(vl + (j4 + e) * kPaVStride);
                    const float vf[4] = {lo_bf16(v.x), hi_bf16(v.x), lo_bf16(v.y), hi_bf16(v.y)};
#pragma unroll
                    for (int x = 0; x < kPaHeadsPerWarp; ++x) {
                        const float pj = e == 0 ? pv[x].x : e == 1 ? pv[x].y : e == 2 ? pv[x].z : pv[x].w;
#pragma unroll
                        for (int u = 0; u < 4; ++u) sa[x][u] = ref::fma(pj, vf[u], sa[x][u]);
                    }
                }
            }
        } else {
            const int jlo = ja[0] < ja[1] ? ja[0] : ja[1], jhi = jb[0] > jb[1] ? jb[0] : jb[1];
            for (int j = jlo; j <= jhi; ++j) {
                const uint2 v = *reinterpret_cast<const uint2 *>(vl + j * kPaVStride);
                const float vf[4] = {lo_bf16(v.x), hi_bf16(v.x), lo_bf16(v.y), hi_bf16(v.y)};
#pragma unroll
                for (int x = 0; x < kPaHeadsPerWarp; ++x) {
                    const int r = x / kGroup;
                    const bool on = j >= ja[r] && j <= jb[r];
                    const float pj = pw[x * kPaPStride + j];
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const float f = ref::fma(pj, vf[u], sa[x][u]);
                        sa[x][u] = on ? f : sa[x][u];
                    }
                }
            }
        }
        __syncthreads();
    }

    // attend's closing folds (the warp's own heads), then the epilogue of the lane's 4 dims of the 6 heads.
    pa_softmax(sm, a, s, i0, 0, true, w, lane);
#pragma unroll
    for (int x = 0; x < kPaHeadsPerWarp; ++x) {
        const int r = x / kGroup, h = x % kGroup;
        if (!ract[r]) continue;
        const PaHead &hx = sm.head[w * kPaHeadsPerWarp + x];
        pa_apply(hx, sa[x], ga[x], ta[x]);
        epilogue_head(hx.m[2], hx.l[2], ta[x], a, s.row + i0 + r0 + r, g * kGroup + h, lane);
    }
}

// Decode pass, kernel 1: grid (span slots of the pass x kSplit, KV heads, sequences x max_rows); a cluster of kSplit
// blocks of kSpanThreads per row, span and KV head. The span is base + pass_first + blockIdx.x / kSplit (base: 0
// full; the first key span of the sequence's first row, sliding); the cluster writes the row's span partial of its 3
// query heads to slot blockIdx.x / kSplit when the row has a key in the span. Block c of the cluster streams keys
// c * 32 .. c * 32 + 31 (K) and dims c * 16 .. c * 16 + 15 (V) of the span, so the span's bytes spread over kSplit
// SMs. warp_tile's arithmetic element for element:
//   1. thread t < 32 scores key c * 32 + t for the 3 heads (q, r in smem, the K row in registers) while the others
//      stage the block's V columns in smem; every block then gathers the cluster's scores (distributed shared
//      memory);
//   2. warp k takes tile k of each head: the tile max (reduce_max<32>'s lane t: keys t, t + 32), then mn from the
//      tiles before it, the probabilities and their sum (reduce_sum<32>'s lanes), the same values in every block;
//   3. thread (h, d) of the block's dims runs head h's l and acc[d] over the tiles in order (times the tile factor
//      between tiles, fma over the tile's valid keys ascending).
__device__ __forceinline__ int64_t seq_base_span(const AttnArgs &a, const Seq &s) {
    return a.kind == kFull ? 0 : row_lo(a, s.pos0) / kSpan;
}

constexpr int kTilesPerSpan = R::kTilesPerSpan;
constexpr int kSplit = 8;                         // blocks (one cluster) per row, span and KV head
constexpr int kSplitKeys = kSpan / kSplit;        // phase 1: one thread per key
constexpr int kSplitDims = kHeadDim / kSplit;     // phase 3: one thread per (head, dim)
constexpr int kSpanThreads = kTilesPerSpan * 32;  // phase 2: one warp per tile
static_assert(kSplitKeys == 32 && kSplitDims % 8 == 0, "a block: one warp of keys, whole 16-byte V pieces");
static_assert(kGroup * kSplitDims <= kSpanThreads && kGroup * kHeadDim / 8 + kGroup * kDRel <= kSpanThreads,
              "phase 3 and the q / r staging fit the block");

struct __align__(16) SpanSmem {
    bf16 v[kSpan][kSplitDims];  // the block's dims of the span's V rows
    bf16 q[kGroup][kHeadDim];
    bf16 r[kGroup][kDRel];
    float own[kGroup][kSplitKeys];  // the block's scores, gathered by the cluster
    float p[kGroup][kSpan];         // the span's scores, then probabilities
    float mt[kGroup][kTilesPerSpan];
    float mn[kGroup][kTilesPerSpan];
    float ts[kGroup][kTilesPerSpan];
};

// reduce_dot<32> of q (smem, 128) with the K row k (registers, 16 x 8 BF16): lane l = fma chain over elements l,
// l + 32, l + 64, l + 96 from 0, then the halving tree.
__device__ __forceinline__ float dot_row(const bf16 *q, const uint4 (&k)[16]) {
    float a[32];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint4 qv = *reinterpret_cast<const uint4 *>(q + 32 * m + 8 * c);
            const uint4 kv = k[4 * m + c];
            const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w}, kw[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int l = 8 * c + 2 * e;
                a[l] = ref::fma(lo_bf16(qw[e]), lo_bf16(kw[e]), m == 0 ? 0.0f : a[l]);
                a[l + 1] = ref::fma(hi_bf16(qw[e]), hi_bf16(kw[e]), m == 0 ? 0.0f : a[l + 1]);
            }
        }
    }
#pragma unroll
    for (int off = 16; off >= 1; off /= 2)
#pragma unroll
        for (int t = 0; t < off; ++t) a[t] = ref::add(a[t], a[t + off]);
    return a[0];
}

// dot_row with the K row read from shared memory (the same operands in the same order; no K registers, so a block
// of kMultiThreads threads keeps several blocks per SM).
__device__ __forceinline__ float dot_row_smem(const bf16 *q, const bf16 *k) {
    float a[32];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint4 qv = *reinterpret_cast<const uint4 *>(q + 32 * m + 8 * c);
            const uint4 kv = *reinterpret_cast<const uint4 *>(k + 32 * m + 8 * c);
            const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w}, kw[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int l = 8 * c + 2 * e;
                a[l] = ref::fma(lo_bf16(qw[e]), lo_bf16(kw[e]), m == 0 ? 0.0f : a[l]);
                a[l + 1] = ref::fma(hi_bf16(qw[e]), hi_bf16(kw[e]), m == 0 ? 0.0f : a[l + 1]);
            }
        }
    }
#pragma unroll
    for (int off = 16; off >= 1; off /= 2)
#pragma unroll
        for (int t = 0; t < off; ++t) a[t] = ref::add(a[t], a[t + off]);
    return a[0];
}

__global__ void __cluster_dims__(kSplit, 1, 1) __launch_bounds__(kSpanThreads)
    decode_span_kernel(AttnArgs a, int64_t pass_first, float *ws) {
    pdl::enter();
    __shared__ SpanSmem sm;
    cg::cluster_group cluster = cg::this_cluster();
    // Every exit below depends on (span slot, KV head, row) only, the same for the cluster's blocks.
    const Seq s = a.seqs[blockIdx.z / unsigned(a.max_rows)];
    const int64_t i = blockIdx.z % unsigned(a.max_rows);
    if (i >= s.rows) return;
    const int g = blockIdx.y, c = int(cluster.block_rank());
    const int64_t slot = blockIdx.x / kSplit;
    const int64_t row = s.row + i, p = s.pos0 + i, k0 = (seq_base_span(a, s) + pass_first + slot) * kSpan;
    const int64_t lo = row_lo(a, p);
    const int ka = int((lo > k0 ? lo : k0) - k0), kb = int((p < k0 + kSpan - 1 ? p : k0 + kSpan - 1) - k0);
    if (ka > kb) return;  // no key of the row in the span
    const int tid = threadIdx.x, w = tid / 32, lane = tid % 32;
    const int tfirst = ka / kTile, tlast = kb / kTile;
    // Loads first, all in flight together: thread t < 32 its key's K row (registers), the others the block's V
    // columns of the span (cp.async into smem, waited for before the phase-2 barrier).
    const int key = c * kSplitKeys + tid;
    const bool key_on = tid < kSplitKeys && key >= ka && key <= kb;
    uint4 kr[16];
    if (key_on) {
        const bf16 *kp = kv_row(a, s, true, k0 + key, g);
#pragma unroll
        for (int u = 0; u < 16; ++u) kr[u] = *reinterpret_cast<const uint4 *>(kp + 8 * u);
    } else if (tid >= kSplitKeys) {
        constexpr int kParts = kSplitDims / 8;
        for (int x = tid - kSplitKeys; x < (kb - ka + 1) * kParts; x += kSpanThreads - kSplitKeys) {
            const int kk = ka + x / kParts, part = x % kParts;
            cp_async16(&sm.v[kk][part * 8], kv_row(a, s, false, k0 + kk, g) + c * kSplitDims + part * 8);
        }
    }
    if (tid < kGroup * kHeadDim / 8) {
        const int h = tid / (kHeadDim / 8), part = tid % (kHeadDim / 8);
        *reinterpret_cast<uint4 *>(&sm.q[h][part * 8]) =
            *reinterpret_cast<const uint4 *>(a.q + row * a.q_stride + (g * kGroup + h) * kHeadDim + part * 8);
    } else if (tid < kGroup * kHeadDim / 8 + kGroup * kDRel) {
        const int x = tid - kGroup * kHeadDim / 8;
        sm.r[x / kDRel][x % kDRel] = a.r[row * a.r_stride + g * kGroup * kDRel + x];
    }
    __syncthreads();
    if (key_on) {
        const int64_t j = k0 + key;
        float P[16];
        bool has;
        load_proj(a.rel_proj, p - j, true, a.mutant, P, has);
#pragma unroll
        for (int h = 0; h < kGroup; ++h)
            sm.own[h][tid] = ref::add(ref::mul(dot_row(sm.q[h], kr), R::kScale), has ? bias16(sm.r[h], P) : 0.0f);
    }
    cluster.sync();  // every block's scores are written
    for (int x = tid; x < kGroup * kSpan; x += kSpanThreads) {
        const int h = x / kSpan, k = x % kSpan;
        if (k >= ka && k <= kb) sm.p[h][k] = *cluster.map_shared_rank(&sm.own[h][k % kSplitKeys], k / kSplitKeys);
    }
    cp_async_wait_all();
    cluster.sync();  // the gathers are done (no block's scores are read after this) and sm.p / sm.v are complete
    const int pt = w;  // phase 2: this warp's tile, for each head
    const int i0 = pt * kTile + lane, i1 = i0 + 32;
    const bool v0 = i0 >= ka && i0 <= kb, v1 = i1 >= ka && i1 <= kb;
    const bool tile_on = pt >= tfirst && pt <= tlast;
    if (tile_on) {
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            float mx = -INFINITY;
            if (v0) mx = ref::max(mx, sm.p[h][i0]);
            if (v1) mx = ref::max(mx, sm.p[h][i1]);
            const float mt = warp_max(mx);
            if (lane == 0) sm.mt[h][pt] = mt;
        }
    }
    __syncthreads();
    if (tile_on) {
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            float mn = sm.mt[h][tfirst];
            for (int u = tfirst + 1; u <= pt; ++u) mn = ref::max(mn, sm.mt[h][u]);
            const float p0 = v0 ? ref::exp(ref::sub(sm.p[h][i0], mn)) : 0.0f;
            const float p1 = v1 ? ref::exp(ref::sub(sm.p[h][i1], mn)) : 0.0f;
            float ts = 0.0f;
            if (v0) ts = ref::add(ts, p0);
            if (v1) ts = ref::add(ts, p1);
            ts = warp_sum(ts);
            if (v0) sm.p[h][i0] = p0;
            if (v1) sm.p[h][i1] = p1;
            if (lane == 0) {
                sm.mn[h][pt] = mn;
                sm.ts[h][pt] = ts;
            }
        }
    }
    __syncthreads();
    if (tid >= kGroup * kSplitDims) return;
    const int h = tid / kSplitDims, dl = tid % kSplitDims, d = c * kSplitDims + dl;
    float m = 0.0f, l = 0.0f, acc = 0.0f;
    for (int t = tfirst; t <= tlast; ++t) {
        const float mn = sm.mn[h][t], ts = sm.ts[h][t];
        if (t == tfirst) {
            l = ts;
        } else {
            const float al = ref::exp(ref::sub(m, mn));
            l = ref::add(ref::mul(l, al), ts);
            acc = ref::mul(acc, al);
        }
        m = mn;
        const int ia = ka > t * kTile ? ka : t * kTile, ib = kb < t * kTile + kTile - 1 ? kb : t * kTile + kTile - 1;
#pragma unroll 8
        for (int ii = ia; ii <= ib; ++ii) acc = ref::fma(sm.p[h][ii], bf(sm.v[ii][dl]), acc);
    }
    float *dst = ws + ((row * kHeads + g * kGroup + h) * kDecodeSlots + slot) * kPartialFloats;
    if (d == 0) {
        dst[0] = m;
        dst[1] = l;
    }
    dst[4 + d] = acc;
}

// Decode pass, kernel 1 for sequences of several rows (S9 phase B verify): decode_span_kernel's arithmetic element
// for element, kRowGroup rows of one sequence per cluster. Grid (span slots x kSplit, KV heads, sequences x row
// groups); group q of the block (threads q * kSpanThreads ..) is row i0 + q, exactly decode_span_kernel's block of
// that row, except that the block stages the span's K rows (smem, read back into the same registers) and V columns
// once for the union of its rows' keys. A row's scores, probabilities, folds and partial are its own, so its bits
// are decode_span_kernel's (and the M = 1 decode's).
constexpr int kRowGroup = 4;
constexpr int kMultiThreads = kRowGroup * kSpanThreads;

struct __align__(16) MultiSmem {
    bf16 k[kSplitKeys][kHeadDim];  // the block's keys of the span
    bf16 v[kSpan][kSplitDims];     // the block's dims of the span's V rows
    struct Row {
        bf16 q[kGroup][kHeadDim];
        bf16 r[kGroup][kDRel];
        float own[kGroup][kSplitKeys];
        float p[kGroup][kSpan];
        float mt[kGroup][kTilesPerSpan];
        float mn[kGroup][kTilesPerSpan];
        float ts[kGroup][kTilesPerSpan];
    } row[kRowGroup];
};

__global__ void __cluster_dims__(kSplit, 1, 1) __launch_bounds__(kMultiThreads, 2)
    decode_span_multi_kernel(AttnArgs a, int64_t pass_first, float *ws, int64_t groups) {
    pdl::enter();
    __shared__ MultiSmem sm;
    cg::cluster_group cluster = cg::this_cluster();
    // Every exit and branch below that guards a barrier depends on (span slot, KV head, sequence, row group) only.
    const Seq s = a.seqs[blockIdx.z / unsigned(groups)];
    const int64_t i0 = int64_t(blockIdx.z % unsigned(groups)) * kRowGroup;
    if (i0 >= s.rows) return;
    const int g = blockIdx.y, c = int(cluster.block_rank());
    const int64_t slot = blockIdx.x / kSplit;
    const int64_t k0 = (seq_base_span(a, s) + pass_first + slot) * kSpan;
    const int tid = threadIdx.x, q = tid / kSpanThreads, t = tid % kSpanThreads, w = t / 32, lane = t % 32;
    const int64_t i = i0 + q, row = s.row + i, p = s.pos0 + i;
    // The row's keys in the span (decode_span_kernel's ka, kb); rows past the sequence have none.
    int ka = 1, kb = 0;
    if (i < s.rows) {
        const int64_t lo = row_lo(a, p);
        ka = int((lo > k0 ? lo : k0) - k0);
        kb = int((p < k0 + kSpan - 1 ? p : k0 + kSpan - 1) - k0);
    }
    // The union over the group's rows (every thread computes it the same way).
    int ua = kSpan, ub = -1;
    for (int u = 0; u < kRowGroup && i0 + u < s.rows; ++u) {
        const int64_t pu = s.pos0 + i0 + u, lu = row_lo(a, pu);
        const int au = int((lu > k0 ? lu : k0) - k0), bu = int((pu < k0 + kSpan - 1 ? pu : k0 + kSpan - 1) - k0);
        if (au > bu) continue;
        ua = au < ua ? au : ua;
        ub = bu > ub ? bu : ub;
    }
    if (ua > ub) return;  // no key of any row in the span
    const bool on = ka <= kb;
    // Loads: the block's K rows and V columns of the union (cp.async), each row's q and r (its group).
    for (int x = tid; x < kSplitKeys * (kHeadDim / 8); x += kMultiThreads) {
        const int kk = c * kSplitKeys + x / (kHeadDim / 8), part = x % (kHeadDim / 8);
        if (kk >= ua && kk <= ub)
            cp_async16(&sm.k[kk - c * kSplitKeys][part * 8], kv_row(a, s, true, k0 + kk, g) + part * 8);
    }
    {
        constexpr int kParts = kSplitDims / 8;
        for (int x = tid; x < (ub - ua + 1) * kParts; x += kMultiThreads) {
            const int kk = ua + x / kParts, part = x % kParts;
            cp_async16(&sm.v[kk][part * 8], kv_row(a, s, false, k0 + kk, g) + c * kSplitDims + part * 8);
        }
    }
    MultiSmem::Row &rs = sm.row[q];
    if (on) {
        if (t < kGroup * kHeadDim / 8) {
            const int h = t / (kHeadDim / 8), part = t % (kHeadDim / 8);
            *reinterpret_cast<uint4 *>(&rs.q[h][part * 8]) =
                *reinterpret_cast<const uint4 *>(a.q + row * a.q_stride + (g * kGroup + h) * kHeadDim + part * 8);
        } else if (t < kGroup * kHeadDim / 8 + kGroup * kDRel) {
            const int x = t - kGroup * kHeadDim / 8;
            rs.r[x / kDRel][x % kDRel] = a.r[row * a.r_stride + g * kGroup * kDRel + x];
        }
    }
    cp_async_wait_all();
    __syncthreads();
    const int key = c * kSplitKeys + t;
    if (on && t < kSplitKeys && key >= ka && key <= kb) {
        const int64_t j = k0 + key;
        float P[16];
        bool has;
        load_proj(a.rel_proj, p - j, true, a.mutant, P, has);
#pragma unroll 1
        for (int h = 0; h < kGroup; ++h)
            rs.own[h][t] =
                ref::add(ref::mul(dot_row_smem(rs.q[h], sm.k[t]), R::kScale), has ? bias16(rs.r[h], P) : 0.0f);
    }
    cluster.sync();  // every block's scores are written
    if (on) {
        for (int x = t; x < kGroup * kSpan; x += kSpanThreads) {
            const int h = x / kSpan, k = x % kSpan;
            if (k >= ka && k <= kb) rs.p[h][k] = *cluster.map_shared_rank(&rs.own[h][k % kSplitKeys], k / kSplitKeys);
        }
    }
    cluster.sync();  // the gathers are done
    const int tfirst = ka / kTile, tlast = kb / kTile;
    const int pt = w;
    const int j0 = pt * kTile + lane, j1 = j0 + 32;
    const bool v0 = j0 >= ka && j0 <= kb, v1 = j1 >= ka && j1 <= kb;
    const bool tile_on = on && pt >= tfirst && pt <= tlast;
    if (tile_on) {
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            float mx = -INFINITY;
            if (v0) mx = ref::max(mx, rs.p[h][j0]);
            if (v1) mx = ref::max(mx, rs.p[h][j1]);
            const float mt = warp_max(mx);
            if (lane == 0) rs.mt[h][pt] = mt;
        }
    }
    __syncthreads();
    if (tile_on) {
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            float mn = rs.mt[h][tfirst];
            for (int u = tfirst + 1; u <= pt; ++u) mn = ref::max(mn, rs.mt[h][u]);
            const float p0 = v0 ? ref::exp(ref::sub(rs.p[h][j0], mn)) : 0.0f;
            const float p1 = v1 ? ref::exp(ref::sub(rs.p[h][j1], mn)) : 0.0f;
            float ts = 0.0f;
            if (v0) ts = ref::add(ts, p0);
            if (v1) ts = ref::add(ts, p1);
            ts = warp_sum(ts);
            if (v0) rs.p[h][j0] = p0;
            if (v1) rs.p[h][j1] = p1;
            if (lane == 0) {
                rs.mn[h][pt] = mn;
                rs.ts[h][pt] = ts;
            }
        }
    }
    __syncthreads();
    if (!on || t >= kGroup * kSplitDims) return;
    const int h = t / kSplitDims, dl = t % kSplitDims, d = c * kSplitDims + dl;
    float m = 0.0f, l = 0.0f, acc = 0.0f;
    for (int u = tfirst; u <= tlast; ++u) {
        const float mn = rs.mn[h][u], ts = rs.ts[h][u];
        if (u == tfirst) {
            l = ts;
        } else {
            const float al = ref::exp(ref::sub(m, mn));
            l = ref::add(ref::mul(l, al), ts);
            acc = ref::mul(acc, al);
        }
        m = mn;
        const int ia = ka > u * kTile ? ka : u * kTile, ib = kb < u * kTile + kTile - 1 ? kb : u * kTile + kTile - 1;
#pragma unroll 8
        for (int ii = ia; ii <= ib; ++ii) acc = ref::fma(rs.p[h][ii], bf(sm.v[ii][dl]), acc);
    }
    float *dst = ws + ((row * kHeads + g * kGroup + h) * kDecodeSlots + slot) * kPartialFloats;
    if (d == 0) {
        dst[0] = m;
        dst[1] = l;
    }
    dst[4 + d] = acc;
}

// Decode pass, kernel 2: grid (rows of a sequence (max_rows), sequences), kFoldThreads threads; warp h folds head h
// of the row: spans of the pass in ascending order into their group, groups into the running total (slot
// kDecodeSpansPerPass between passes); the row's last pass runs the epilogue. The span partials are read
// kFoldBatch at a time ahead of their folds. A group never straddles two passes: full layers start at span 0 and a
// pass is a whole number of groups; a sliding call needs at most (1023 + 64 + 255) / 256 + 1 = 6 spans, one pass.
static_assert(kDecodeSpansPerPass % kSpansPerGroup == 0, "a decode pass holds whole span groups");
static_assert((int64_t(R::kWindow) - 1 + kMaxDecodeRows + kSpan - 1) / kSpan + 1 <= kDecodeSpansPerPass,
              "a sliding decode call is one pass");
constexpr int kFoldThreads = kHeads * 32;
constexpr int kFoldBatch = 8;
__global__ void __launch_bounds__(kFoldThreads) decode_fold_kernel(AttnArgs a, int64_t pass_first, float *ws) {
    pdl::enter();
    const Seq s = a.seqs[blockIdx.y];
    const int64_t i = blockIdx.x;
    if (i >= s.rows) return;
    const int h = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int64_t row = s.row + i, p = s.pos0 + i;
    const int64_t base = seq_base_span(a, s);
    const int64_t first = row_lo(a, p) / kSpan - base, last = p / kSpan - base;  // the row's spans, pass-relative
    const int64_t pass_last = pass_first + kDecodeSpansPerPass - 1;
    if (last < pass_first || first > pass_last) return;
    const int64_t x0 = first > pass_first ? first : pass_first, x1 = last < pass_last ? last : pass_last;
    {
        float *head = ws + (row * kHeads + h) * kDecodeSlots * kPartialFloats;
        float *state = head + kDecodeSpansPerPass * kPartialFloats;
        Level<1> total, group, span;
        level_reset(total);
        level_reset(group);
        if (first < pass_first) {
            total.empty = false;
            total.m[0] = state[0];
            total.l[0] = state[1];
            const float4 v = *reinterpret_cast<const float4 *>(state + 4 + 4 * lane);
            total.acc[0][0] = v.x, total.acc[0][1] = v.y, total.acc[0][2] = v.z, total.acc[0][3] = v.w;
        }
        for (int64_t xb = x0; xb <= x1; xb += kFoldBatch) {
            float pm[kFoldBatch], pl[kFoldBatch];
            float4 pv[kFoldBatch];
#pragma unroll
            for (int u = 0; u < kFoldBatch; ++u) {
                if (xb + u > x1) break;
                const float *src = head + (xb + u - pass_first) * kPartialFloats;
                pm[u] = src[0];
                pl[u] = src[1];
                pv[u] = *reinterpret_cast<const float4 *>(src + 4 + 4 * lane);
            }
#pragma unroll
            for (int u = 0; u < kFoldBatch; ++u) {
                const int64_t x = xb + u;
                if (x > x1) break;
                span.empty = false;
                span.m[0] = pm[u];
                span.l[0] = pl[u];
                span.acc[0][0] = pv[u].x, span.acc[0][1] = pv[u].y, span.acc[0][2] = pv[u].z, span.acc[0][3] = pv[u].w;
                if (x > x0 && (base + x) / kSpansPerGroup != (base + x - 1) / kSpansPerGroup) {
                    level_fold(total, group);
                    level_reset(group);
                }
                level_fold(group, span);
            }
        }
        level_fold(total, group);
        if (last <= pass_last) {
            epilogue_head(total.m[0], total.l[0], total.acc[0], a, row, h, lane);
        } else {
            if (lane == 0) {
                state[0] = total.m[0];
                state[1] = total.l[0];
            }
            *reinterpret_cast<float4 *>(state + 4 + 4 * lane) =
                make_float4(total.acc[0][0], total.acc[0][1], total.acc[0][2], total.acc[0][3]);
        }
    }
}

// ---- 3. KV store -----------------------------------------------------------------------------------------------
__global__ void kv_store_kernel(int kind, const Seq *seqs, const bf16 *k, int64_t k_stride, const bf16 *v,
                                int64_t v_stride) {
    pdl::enter();
    const Seq s = seqs[blockIdx.y];
    const int64_t i = blockIdx.x;
    if (i >= s.rows) return;
    if (kind == kSliding && i < s.rows - kRingSlots) return;
    const int64_t pos = s.pos0 + i, row = s.row + i;
    const int64_t slot = kind == kSliding ? pos % kRingSlots : (s.slots ? s.slots[pos] : pos);
    for (int c = threadIdx.x; c < 2 * kKvRow / 8; c += blockDim.x) {
        const bool is_k = c < kKvRow / 8;
        const int part = is_k ? c : c - kKvRow / 8;
        const bf16 *src = (is_k ? k + row * k_stride : v + row * v_stride) + part * 8;
        bf16 *dst = (is_k ? s.k_cache : s.v_cache) + slot * kKvRow + part * 8;
        *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
    }
}

bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }
bool stride8(int64_t s, int64_t min) { return s >= min && s % 8 == 0; }

bool attn_args_ok(const AttnArgs &a, int64_t max_rows_cap) {
    if (a.kind != kSliding && a.kind != kFull) return false;
    if (!a.seqs || a.n_seqs < 1 || a.n_seqs > 65535 || a.max_rows < 1 || a.max_rows > max_rows_cap) return false;
    if (!a.q || !a.r || !a.rel_proj || !a.k || !a.v || !a.sink || !a.gate || !a.out) return false;
    if (!aligned16(a.q) || !aligned16(a.k) || !aligned16(a.v) || !aligned16(a.out)) return false;
    if (!stride8(a.q_stride, kQRow) || !stride8(a.k_stride, kKvRow) || !stride8(a.v_stride, kKvRow) ||
        !stride8(a.out_stride, kQRow) || !stride8(a.gate_stride, kQRow))
        return false;
    if (reinterpret_cast<uintptr_t>(a.gate) % 8 || a.r_stride < kRelRow || a.sink_stride < kHeads) return false;
    return a.mutant >= R::kNone && a.mutant <= R::kBiasOffByOne;
}

}  // namespace

cudaError_t post(const PostArgs &a, cudaStream_t stream) {
    if (a.rows < 0 || a.rows > INT32_MAX || (a.kind != kSliding && a.kind != kFull)) return cudaErrorInvalidValue;
    if (a.mutant != R::kNone && a.mutant != R::kNoLogScale) return cudaErrorInvalidValue;
    if ((a.q_in && (!a.q_scale || !a.q_out || a.q_in_stride < kQRow || a.q_out_stride < kQRow)) ||
        (a.r_in && (!a.rel_scale || !a.r_out || a.r_in_stride < kRelRow || a.r_out_stride < kRelRow)) ||
        (a.k_in && (!a.k_out || a.k_in_stride < kKvRow || a.k_out_stride < kKvRow)) ||
        (a.v_in && (!a.v_out || a.v_in_stride < kKvRow || a.v_out_stride < kKvRow)))
        return cudaErrorInvalidValue;
    if (a.kind == kFull && (a.q_in || a.r_in) && !a.positions) return cudaErrorInvalidValue;
    if (a.rows == 0) return cudaSuccess;
    return pdl::launch(post_kernel, dim3(unsigned(a.rows), kPostBlocks), dim3(kThreads), 0, stream, a);
}

cudaError_t prefill(const AttnArgs &a, cudaStream_t stream) {
    AttnArgs b = a;
    if (b.mutant == kMutantLaneSum) b.mutant = R::kNone;  // a prefill-only mutant
    if (!attn_args_ok(b, kMaxPrefillRows)) return cudaErrorInvalidValue;
    constexpr size_t bytes = sizeof(PaSmem);
    const cudaError_t e =
        cudaFuncSetAttribute(prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if (e != cudaSuccess) return e;
    const dim3 grid(unsigned((a.max_rows + kPaRows - 1) / kPaRows), kKvHeads, unsigned(a.n_seqs));
    return pdl::launch(prefill_kernel, grid, dim3(kPaThreads), bytes, stream, a);
}

size_t decode_workspace_bytes(int64_t total_rows) {
    if (total_rows < 1) return 0;
    return size_t(total_rows) * kHeads * kDecodeSlots * kPartialFloats * sizeof(float);
}

cudaError_t decode(const AttnArgs &a, int64_t total_rows, void *workspace, size_t workspace_bytes,
                   cudaStream_t stream) {
    if (!attn_args_ok(a, kMaxDecodeRows) || total_rows < 1 || total_rows > kMaxDecodeRows) return cudaErrorInvalidValue;
    if (!workspace || (reinterpret_cast<uintptr_t>(workspace) & 255u) ||
        workspace_bytes < decode_workspace_bytes(total_rows))
        return cudaErrorInvalidValue;
    if (a.max_pos < 1) return cudaErrorInvalidValue;
    if (int64_t(a.n_seqs) * a.max_rows > 65535) return cudaErrorInvalidValue;
    // Spans to cover: full: [0, ceil(max_pos / 256)); sliding: the window of the first row plus the rows, from the
    // sequence's base span (at most (1023 + max_rows) / 256 + 2 spans).
    const int64_t spans = a.kind == kFull ? (a.max_pos + kSpan - 1) / kSpan
                                          : (int64_t(R::kWindow) - 1 + a.max_rows + kSpan - 1) / kSpan + 1;
    if (a.kind == kSliding && a.mutant == R::kNoWindow) return cudaErrorInvalidValue;  // prefill-only mutant
    float *ws = static_cast<float *>(workspace);
    // Sequences of several rows (a verify) take the row-group kernel: the same bits per row, the span's K/V staged
    // once per kRowGroup rows instead of once per row.
    const int64_t groups = (a.max_rows + kRowGroup - 1) / kRowGroup;
    for (int64_t first = 0; first < spans; first += kDecodeSpansPerPass) {
        const int64_t n = spans - first < kDecodeSpansPerPass ? spans - first : kDecodeSpansPerPass;
        cudaError_t e;
        if (a.max_rows > 1) {
            const dim3 grid(unsigned(n * kSplit), kKvHeads, unsigned(a.n_seqs * groups));
            e = pdl::launch(decode_span_multi_kernel, grid, dim3(kMultiThreads), 0, stream, a, first, ws, groups);
        } else {
            const dim3 span_grid(unsigned(n * kSplit), kKvHeads, unsigned(a.n_seqs * a.max_rows));
            e = pdl::launch(decode_span_kernel, span_grid, dim3(kSpanThreads), 0, stream, a, first, ws);
        }
        if (e == cudaSuccess)
            e = pdl::launch(decode_fold_kernel, dim3(unsigned(a.max_rows), unsigned(a.n_seqs)), dim3(kFoldThreads), 0,
                            stream, a, first, ws);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

cudaError_t kv_store(int kind, const Seq *seqs, int n_seqs, int64_t max_rows, const bf16 *k, int64_t k_stride,
                     const bf16 *v, int64_t v_stride, cudaStream_t stream) {
    if ((kind != kSliding && kind != kFull) || !seqs || n_seqs < 1 || n_seqs > 65535 || max_rows < 1 ||
        max_rows > INT32_MAX || !k || !v || !aligned16(k) || !aligned16(v) || !stride8(k_stride, kKvRow) ||
        !stride8(v_stride, kKvRow))
        return cudaErrorInvalidValue;
    return pdl::launch(kv_store_kernel, dim3(unsigned(max_rows), unsigned(n_seqs)), dim3(128), 0, stream, kind, seqs, k,
                       k_stride, v, v_stride);
}

}  // namespace cere::inkling::attn
