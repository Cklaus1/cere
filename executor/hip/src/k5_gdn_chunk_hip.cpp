// AmdAmd PAR-6 step 3: K5 chunked prefill (the contract of executor/cuda/src/kernels/k5_gdn_chunk.h) on HIP for
// gfx1151, written from that header's math with FLA's rounding points. The CUDA kernel's tensor-core MMAs, cp.async
// staging and TF32 block merge are NVIDIA-only. Here every matrix product has BF16 operands by the contract (k, q, A,
// BF16(v beta), BF16(k beta exp G), h_c, w, P, v_new, BF16(d exp(G_last - G))) and runs on gfx11's WMMA
// (v_wmma_f32_16x16x16_bf16, FP32 accumulate, fixed k order: no atomics, no split-K, identical calls give identical
// bytes); the triangular inverse, whose operand is FP32, is exact forward substitution in FP32 FMAs (FLA's is a TF32
// block merge, which the CUDA side reproduces; this one differs from FLA by FP32 rounding only and is held to an FP64
// reference).
//
// Per call: q, k L2-normalised to BF16 (k5::l2norm), G = per-chunk inclusive cumsum of g (FP32). Per chunk of up to
// 64 rows and value head h (key head h / 3):
//   A = (I + tril(diag(beta) (k k^T * exp(G_i - G_j)), -1))^-1 -> BF16
//   u = BF16(A BF16(v * beta)),  w = BF16(A BF16(k * beta * exp(G)))
//   h_c = BF16(h);  d = u - w h_c^T;  v_new = BF16(d)
//   o = BF16((q h_c^T) * exp(G) * scale + (BF16(tril(q k^T * exp(G_i - G_j))) v_new) * scale)
//   h = h * exp(G_last) + k^T BF16(d * exp(G_last - G))
// final state = BF16(h); state_in32 / capture32 / *_hi as the header.
//
// Operand layout: every WMMA operand is k-contiguous in LDS (B stored transposed where the math has it k-major), rows
// padded by 8 BF16 (16 B) so the 16 lanes' 16-byte reads of one k step hit distinct banks.
//
// FLA's three phases: prep (A, u, w, P; every chunk in parallel), the recurrence (h resident in LDS, one launch, the
// only sequential part; tiled over 16 value rows of h, which are independent), output (o; every chunk in parallel
// from the h_c the recurrence stores).
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/k5_gdn.h"
#include "kernels/k5_gdn_chunk.h"

namespace cere::k5 {
namespace {

constexpr int KH = 16, VH = 48, D = kHeadDim, BT = kChunkTokens, kThreads = 256, kWaves = kThreads / 32;
constexpr int VT = 32, RW = 4;      // recurrence: value rows of h per block, waves per block
constexpr int OT = 64;              // output: value columns per block
constexpr int LD = D + 8, LT = BT + 8;  // padded LDS row strides (BF16 elements) for 128- and 64-wide rows
using u16 = uint16_t;               // BF16 bits
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

__device__ inline u16 tob(float x) { return __builtin_bit_cast(u16, __float2bfloat16(x)); }
__device__ inline float fb(u16 x) { return __uint_as_float(uint32_t(x) << 16); }

struct U16x16 { uint4 lo, hi; };
__device__ inline bf16x16 ld16(const u16 *p) {
    return __builtin_bit_cast(bf16x16, U16x16{*reinterpret_cast<const uint4 *>(p), *reinterpret_cast<const uint4 *>(p + 8)});
}
// c (16x16) += A B over k < K (a multiple of 16): A row-major at a (row stride lda), B stored transposed at b ([col][k],
// stride ldb). Lane l feeds A row l % 16 and B column l % 16; element i of c is row 2 i + (l % 32) / 16, column l % 16.
__device__ inline f32x8 mma(f32x8 c, int K, const u16 *a, int lda, const u16 *b, int ldb) {
    const int r = threadIdx.x % 16;
    a += r * lda; b += r * ldb;
    for (int k0 = 0; k0 < K; k0 += 16) c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ld16(a + k0), ld16(b + k0), c);
    return c;
}
__device__ inline int crow(int i) { return 2 * i + (threadIdx.x % 32) / 16; }
__device__ inline int ccol() { return threadIdx.x % 16; }

// Staging global -> LDS in two passes, all of a thread's 16-byte loads first, then the LDS stores, so one wave keeps
// every load in flight (a load-wait-store loop serialises the memory latency, ~1 us on this APU, per iteration).
template <int N> struct Regs { uint4 r[N]; };
template <int N, class F> __device__ inline Regs<N> gather(F ld) {  // ld(e) for e = tid + q * kThreads
    Regs<N> x;
#pragma unroll
    for (int q = 0; q < N; ++q) x.r[q] = ld(int(threadIdx.x) + q * kThreads);
    return x;
}
template <int N, class F> __device__ inline void scatter(const Regs<N> &x, F st) {
#pragma unroll
    for (int q = 0; q < N; ++q) st(int(threadIdx.x) + q * kThreads, x.r[q]);
}
constexpr uint4 kZero4{0, 0, 0, 0};
constexpr int kRowIters = BT * (D / 8) / kThreads;  // 64 rows x 128 BF16 as uint4 per thread: 4
__device__ inline const uint4 &as4(const u16 *p) { return *reinterpret_cast<const uint4 *>(p); }
__device__ inline uint4 &as4(u16 *p) { return *reinterpret_cast<uint4 *>(p); }
// e -> (row t = e / 16, 8 columns at c8): rows >= n zero; row-major store into dst (stride LD)
template <class F> __device__ inline Regs<kRowIters> gather_rows(int n, F src) {
    return gather<kRowIters>([&](int e) { const int t = e / (D / 8), c8 = (e % (D / 8)) * 8; return t < n ? as4(src(t) + c8) : kZero4; });
}
__device__ inline void scatter_rows(const Regs<kRowIters> &x, u16 *dst) {
    scatter(x, [&](int e, uint4 v) { as4(dst + (e / (D / 8)) * LD + (e % (D / 8)) * 8) = v; });
}
// e -> (row t = e % 64, 8 columns at c8 = (e / 64) * 8): transposed store dst[c][t] (stride LT), optionally mapped
template <class F> __device__ inline Regs<kRowIters> gather_cols(int n, F src) {
    return gather<kRowIters>([&](int e) { const int t = e % BT, c8 = (e / BT) * 8; return t < n ? as4(src(t) + c8) : kZero4; });
}
template <class M> __device__ inline void scatter_cols(const Regs<kRowIters> &x, u16 *dst, M map) {
    scatter(x, [&](int e, uint4 v) {
        const int t = e % BT, c8 = (e / BT) * 8;
        const u16 *h = reinterpret_cast<const u16 *>(&v);
#pragma unroll
        for (int q = 0; q < 8; ++q) dst[(c8 + q) * LT + t] = map(t, h[q]);
    });
}

// Head-major throughout (measured on gfx1151: a block reading 64 rows of 128-256 B at a 4-12 KB stride, the
// token-major [T, heads, 128] pattern, streams at 14-29 GB/s against 206 GB/s contiguous: the cost is per 4 KB page
// touched), so each (chunk, head) block's operands are one contiguous 4-16 KB run; the inputs are permuted to head-major
// once per call and o back at the end (row moves of 256 B, every page fully used).
struct Workspace {
    u16 *qn, *kn;     // [16, T, 128] normalised
    u16 *vt;          // [48, T, 128] v
    float *G, *B;     // [48, T] cumsum g, beta
    u16 *u, *w, *vn;  // [48, T, 128]
    u16 *P;           // [48, T, 64]: row i of its chunk's BF16 intra-chunk attention (j <= i, else 0)
    u16 *hc;          // [48, chunks, 128 (v), 128 (k)]: BF16(h) at the start of each chunk
    u16 *kT;          // [16, chunks, 128 (k), 64 (t)]: each chunk's normalised k transposed (rows >= n zero)
    u16 *ot;          // [48, T, 128] o; before that, the token-major normalised q and k ([T, 16, 128] each)
};

size_t align256(size_t b) { return (b + 255) & ~size_t(255); }

size_t carve(void *base, int64_t T, Workspace *ws) {
    uint8_t *p = static_cast<uint8_t *>(base), *p0 = p;
    const int64_t chunks = (T + BT - 1) / BT;
    auto take = [&](size_t bytes) { uint8_t *r = p; p += align256(bytes); return r; };
    Workspace w{};
    w.qn = reinterpret_cast<u16 *>(take(size_t(T) * KH * D * 2));
    w.kn = reinterpret_cast<u16 *>(take(size_t(T) * KH * D * 2));
    w.vt = reinterpret_cast<u16 *>(take(size_t(T) * VH * D * 2));
    w.G = reinterpret_cast<float *>(take(size_t(T) * VH * 4));
    w.B = reinterpret_cast<float *>(take(size_t(T) * VH * 4));
    w.u = reinterpret_cast<u16 *>(take(size_t(T) * VH * D * 2));
    w.w = reinterpret_cast<u16 *>(take((size_t(T) * VH + BT) * D * 2));  // + 64 rows: the last partial chunk's A reads
    w.vn = reinterpret_cast<u16 *>(take(size_t(T) * VH * D * 2));
    w.P = reinterpret_cast<u16 *>(take(size_t(T) * VH * BT * 2));
    w.hc = reinterpret_cast<u16 *>(take(size_t(chunks) * VH * D * D * 2));
    w.kT = reinterpret_cast<u16 *>(take(size_t(chunks) * KH * D * BT * 2));
    w.ot = reinterpret_cast<u16 *>(take(size_t(T) * VH * D * 2));  // >= 2 x [T, 16, 128]
    if (ws) *ws = w;
    return size_t(p - p0);
}

// G[h, t] = inclusive cumsum of g within t's chunk (FP32, in row order), B[h, t] = beta[t, h]. One thread per
// (chunk, head).
__global__ void cumsum_kernel(const float *g, const float *beta, float *G, float *B, int64_t T) {
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x, c = idx / VH;
    const int h = int(idx % VH);
    if (c * BT >= T) return;
    float acc = 0.f;
    for (int64_t t = c * BT; t < T && t < (c + 1) * BT; ++t) { acc += g[t * VH + h]; G[h * T + t] = acc; B[h * T + t] = beta[t * VH + h]; }
}

// Row permute of 128-BF16 rows: token-major [T, H] <-> head-major [H, T]. Block: 16 tokens x all H heads, 8 heads'
// loads in flight per pass.
template <bool kToHeadMajor>
__global__ void __launch_bounds__(kThreads) permute_kernel(const u16 *src, u16 *dst, int64_t T, int H) {
    const int64_t t0 = int64_t(blockIdx.x) * 16;
    const int tid = threadIdx.x, tok = tid / 16, c8 = (tid % 16) * 8;
    const int64_t t = t0 + tok;
    if (t >= T) return;
    for (int h0 = 0; h0 < H; h0 += 8) {
        uint4 r[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int64_t row = kToHeadMajor ? t * H + h0 + q : (h0 + q) * T + t;
            r[q] = as4(src + row * D + c8);
        }
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int64_t row = kToHeadMajor ? (h0 + q) * T + t : t * H + h0 + q;
            as4(dst + row * D + c8) = r[q];
        }
    }
}

// Prep, block (chunk, value head): A, u, w, P. LDS regions: X = K [64][LD] -> BF16(v beta)^T [128][LT];
// Y = Q [64][LD] -> A FP32 [64][64] -> A BF16 [64][LT]; Z = L FP32 [64][65] -> BF16(k beta exp G)^T [128][LT].
__global__ void __launch_bounds__(kThreads) prep_kernel(Workspace ws, int64_t T) {
    const int c = blockIdx.x, vh = blockIdx.y, kh = vh / 3, tid = threadIdx.x, wave = tid / 32;
    const int64_t r0 = int64_t(c) * BT;
    const int n = int(T - r0 < BT ? T - r0 : BT);
    __shared__ __attribute__((aligned(16))) u16 X[D * LT];
    __shared__ __attribute__((aligned(16))) u16 Y[BT * LD];
    __shared__ __attribute__((aligned(16))) u16 Z[D * LT];
    __shared__ float Gs[BT], Bs[BT], Eg[BT], part[4][BT];
    u16 *KS = X, *VBT = X, *QS = Y, *AB = Y, *KBT = Z;
    float (*A)[BT] = reinterpret_cast<float (*)[BT]>(Y);
    float (*L)[BT + 1] = reinterpret_cast<float (*)[BT + 1]>(Z);
    {
        const Regs<kRowIters> rk = gather_rows(n, [&](int t) { return ws.kn + (kh * T + r0 + t) * D; });
        const Regs<kRowIters> rq = gather_rows(n, [&](int t) { return ws.qn + (kh * T + r0 + t) * D; });
        scatter_rows(rk, KS); scatter_rows(rq, QS);
    }
    const Regs<kRowIters> rv = gather_cols(n, [&](int t) { return ws.vt + (vh * T + r0 + t) * D; });  // used after the inverse
    for (int i = tid; i < BT; i += kThreads) {
        const bool in = i < n;
        Gs[i] = in ? ws.G[vh * T + r0 + i] : 0.f; Bs[i] = in ? ws.B[vh * T + r0 + i] : 0.f; Eg[i] = expf(Gs[i]);
    }
    __syncthreads();
    if (vh % 3 == 0) {  // k^T of the chunk for the recurrence (rows >= n are zero in KS)
        const int64_t chunks = (T + BT - 1) / BT;
        u16 *dst = ws.kT + (size_t(kh) * chunks + c) * D * BT;
        for (int e = tid; e < D * BT; e += kThreads) dst[e] = KS[(e % BT) * LD + e / BT];
    }
    // k k^T -> L[i][j] = beta_i s exp(G_i - G_j) (j < i), q k^T -> P = BF16(s exp(G_i - G_j)) (j <= i): 16 tiles each
    for (int job = wave; job < 32; job += kWaves) {
        const int qk = job / 16, ti = (job % 16) / 4, tj = job % 4;
        f32x8 acc = {};
        if (tj <= ti) acc = mma(acc, D, (qk ? QS : KS) + ti * 16 * LD, LD, KS + tj * 16 * LD, LD);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int i = ti * 16 + crow(e), j = tj * 16 + ccol();
            if (!qk) L[i][j] = (i < n && j < i) ? Bs[i] * acc[e] * expf(Gs[i] - Gs[j]) : 0.f;
            else if (i < n) ws.P[(vh * T + r0 + i) * BT + j] = j <= i ? tob(acc[e] * expf(Gs[i] - Gs[j])) : u16(0);
        }
    }
    __syncthreads();
    for (int e = tid; e < BT * BT; e += kThreads) A[e / BT][e % BT] = (e / BT == e % BT) ? 1.f : 0.f;  // Q is dead
    __syncthreads();
    // A = (I + L)^-1, row by row: A[i][j] = -sum_{m=j}^{i-1} L[i][m] A[m][j]; thread (j, p) sums m = j + p (mod 4)
    {
        const int j = tid % BT, p = tid / BT;
        for (int i = 1; i < n; ++i) {
            float s = 0.f;
            if (j < i) for (int m = j + p; m < i; m += 4) s = fmaf(L[i][m], A[m][j], s);
            part[p][j] = s;
            __syncthreads();
            if (p == 0 && j < i) A[i][j] = -(((part[0][j] + part[1][j]) + part[2][j]) + part[3][j]);
            __syncthreads();
        }
    }
    // A -> BF16 [64][LT] (in place, through registers); L is dead: KB^T[c][t] = BF16(k beta exp G) from K in X
    float areg[BT * BT / kThreads];
#pragma unroll
    for (int q = 0; q < BT * BT / kThreads; ++q) { const int e = tid + q * kThreads; areg[q] = A[e / BT][e % BT]; }
    for (int e = tid; e < BT * D; e += kThreads) { const int t = e % BT, cc = e / BT; KBT[cc * LT + t] = tob(fb(KS[t * LD + cc]) * Bs[t] * Eg[t]); }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < BT * BT / kThreads; ++q) { const int e = tid + q * kThreads; AB[(e / BT) * LT + e % BT] = tob(areg[q]); }
    // K is dead: VB^T[c][t] = BF16(v beta)
    scatter_cols(rv, VBT, [&](int t, u16 x) { return t < n ? tob(fb(x) * Bs[t]) : u16(0); });
    __syncthreads();
    // u = BF16(A VB), w = BF16(A KB): 4 x 8 tiles each, k over the 64 chunk rows
    for (int job = wave; job < 64; job += kWaves) {
        const int isw = job / 32, ti = (job % 32) / 8, tj = job % 8;
        f32x8 acc = {};
        acc = mma(acc, BT, AB + ti * 16 * LT, LT, (isw ? KBT : VBT) + tj * 16 * LT, LT);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int i = ti * 16 + crow(e), cc = tj * 16 + ccol();
            if (i < n) (isw ? ws.w : ws.u)[(vh * T + r0 + i) * D + cc] = tob(acc[e]);
        }
    }
}

// Recurrence, block (value head, 32-row v tile), 4 waves: every chunk in order. The FP32 h tile lives in the waves'
// WMMA accumulators (wave w: v row tiles 0-1 x k column tiles 2w, 2w+1); w and k^T feed WMMA straight from global
// (contiguous rows); LDS holds only BF16(h), the d tile, u and G (9 KB), so all 192 blocks are resident at once.
// Each block reads and writes only its own rows of `state`, so state in == state out is safe.
__global__ void __launch_bounds__(RW * 32) recur_kernel(Workspace ws, int64_t T, const u16 *state, const float *in32,
                                                        const float *in32_hi, u16 *state_out, float *cap,
                                                        float *cap_hi, int64_t cap_chunk) {
    const int vh = blockIdx.x, v0 = blockIdx.y * VT, kh = vh / 3, tid = threadIdx.x, wave = tid / 32;
    const int64_t chunks = (T + BT - 1) / BT;
    __shared__ __attribute__((aligned(16))) u16 HC[VT * LD];   // BF16(h) [v][k]
    __shared__ __attribute__((aligned(16))) u16 DST[VT * LT];  // BF16(d exp(G_last - G))^T [v][t]
    __shared__ __attribute__((aligned(16))) u16 US[BT * VT];   // u [t][v]
    __shared__ float Gs[BT];
    const float *src32 = in32 == nullptr ? nullptr : (in32_hi != nullptr && vh >= 24 ? in32_hi + size_t(vh - 24) * D * D : in32 + size_t(vh) * D * D);
    float *dst32 = cap == nullptr ? nullptr : (cap_hi != nullptr && vh >= 24 ? cap_hi + size_t(vh - 24) * D * D : cap + size_t(vh) * D * D);
    f32x8 H[2][2];  // [v row tile][k column tile 2 wave + ct]
    auto hidx = [&](int rt, int ct, int e) { return size_t(v0 + rt * 16 + crow(e)) * D + (2 * wave + ct) * 16 + ccol(); };
#pragma unroll
    for (int rt = 0; rt < 2; ++rt)
#pragma unroll
        for (int ct = 0; ct < 2; ++ct)
#pragma unroll
            for (int e = 0; e < 8; ++e) H[rt][ct][e] = src32 ? src32[hidx(rt, ct, e)] : fb(state[size_t(vh) * D * D + hidx(rt, ct, e)]);
    for (int64_t c = 0; c <= chunks; ++c) {
        if (dst32 != nullptr && c == cap_chunk)
#pragma unroll
            for (int rt = 0; rt < 2; ++rt)
#pragma unroll
                for (int ct = 0; ct < 2; ++ct)
#pragma unroll
                    for (int e = 0; e < 8; ++e) dst32[hidx(rt, ct, e)] = H[rt][ct][e];
        if (c == chunks) break;
        const int64_t r0 = c * BT;
        const int n = int(T - r0 < BT ? T - r0 : BT);
        // BF16(h) -> LDS and hc; u tile and G -> LDS
        u16 *hcg = ws.hc + (size_t(vh) * chunks + c) * D * D;
#pragma unroll
        for (int rt = 0; rt < 2; ++rt)
#pragma unroll
            for (int ct = 0; ct < 2; ++ct)
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int vr = rt * 16 + crow(e), k = (2 * wave + ct) * 16 + ccol();
                    const u16 x = tob(H[rt][ct][e]);
                    HC[vr * LD + k] = x;
                    hcg[size_t(v0 + vr) * D + k] = x;
                }
        for (int e = tid; e < BT * (VT / 8); e += RW * 32) {
            const int t = e / (VT / 8), c8 = (e % (VT / 8)) * 8;
            as4(US + t * VT + c8) = t < n ? as4(ws.u + (vh * T + r0 + t) * D + v0 + c8) : kZero4;
        }
        if (tid < BT) Gs[tid] = tid < n ? ws.G[vh * T + r0 + tid] : 0.f;
        __syncthreads();
        const float gl = Gs[n - 1];
        // d = u - w h_c^T: wave = row tile, both 16-wide v tiles; v_new = BF16(d), DS = BF16(d exp(G_last - G_i))
        {
            const int ti = wave;
            f32x8 acc[2] = {};
            const u16 *wrow = ws.w + (vh * T + r0 + ti * 16 + ccol()) * D;
            for (int k0 = 0; k0 < D; k0 += 16) {
                const bf16x16 a = ld16(wrow + k0);
#pragma unroll
                for (int tv = 0; tv < 2; ++tv) acc[tv] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, ld16(HC + (tv * 16 + ccol()) * LD + k0), acc[tv]);
            }
#pragma unroll
            for (int tv = 0; tv < 2; ++tv)
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int i = ti * 16 + crow(e), cv = tv * 16 + ccol();
                    u16 ds = 0;
                    if (i < n) {
                        const float d = fb(US[i * VT + cv]) - acc[tv][e];
                        ws.vn[(vh * T + r0 + i) * D + v0 + cv] = tob(d);
                        ds = tob(d * expf(gl - Gs[i]));
                    }
                    DST[cv * LT + i] = ds;
                }
        }
        __syncthreads();
        // h = h exp(G_last) + DS^T k: A = DST rows (v), B = k^T rows (k) from global, k over the chunk rows
        const float eg = expf(gl);
        const u16 *kt = ws.kT + (size_t(kh) * chunks + c) * D * BT;
#pragma unroll
        for (int ct = 0; ct < 2; ++ct) {
            const u16 *brow = kt + size_t((2 * wave + ct) * 16 + ccol()) * BT;
            f32x8 acc[2] = {};
            for (int t0 = 0; t0 < BT; t0 += 16) {
                const bf16x16 b = ld16(brow + t0);
#pragma unroll
                for (int rt = 0; rt < 2; ++rt) acc[rt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ld16(DST + (rt * 16 + ccol()) * LT + t0), b, acc[rt]);
            }
#pragma unroll
            for (int rt = 0; rt < 2; ++rt)
#pragma unroll
                for (int e = 0; e < 8; ++e) H[rt][ct][e] = fmaf(H[rt][ct][e], eg, acc[rt][e]);
        }
    }
#pragma unroll
    for (int rt = 0; rt < 2; ++rt)
#pragma unroll
        for (int ct = 0; ct < 2; ++ct)
#pragma unroll
            for (int e = 0; e < 8; ++e) state_out[size_t(vh) * D * D + hidx(rt, ct, e)] = tob(H[rt][ct][e]);
}

// Output, block (chunk, value head, 64-column v tile): o = BF16((q h_c^T) exp(G) scale + (P v_new) scale).
__global__ void __launch_bounds__(kThreads) out_kernel(Workspace ws, int64_t T, float scale) {
    const int c = blockIdx.x, vh = blockIdx.y, v0 = blockIdx.z * OT, kh = vh / 3, tid = threadIdx.x, wave = tid / 32;
    const int64_t r0 = int64_t(c) * BT, chunks = (T + BT - 1) / BT;
    const int n = int(T - r0 < BT ? T - r0 : BT);
    __shared__ __attribute__((aligned(16))) u16 QS[BT * LD];   // q [t][k]
    __shared__ __attribute__((aligned(16))) u16 HS[OT * LD];   // h_c [v][k]
    __shared__ __attribute__((aligned(16))) u16 PS[BT * LT];   // P [i][m]
    __shared__ __attribute__((aligned(16))) u16 VST[OT * LT];  // v_new^T [v][m]
    __shared__ float Eg[BT];
    {
        const Regs<kRowIters> rq = gather_rows(n, [&](int t) { return ws.qn + (kh * T + r0 + t) * D; });
        const Regs<OT * (D / 8) / kThreads> rh = gather<OT * (D / 8) / kThreads>([&](int e) {
            return as4(ws.hc + ((size_t(vh) * chunks + c) * D + v0 + e / (D / 8)) * D + (e % (D / 8)) * 8); });
        const Regs<BT * (BT / 8) / kThreads> rp = gather<BT * (BT / 8) / kThreads>([&](int e) {
            const int t = e / (BT / 8); return t < n ? as4(ws.P + (vh * T + r0 + t) * BT + (e % (BT / 8)) * 8) : kZero4; });
        const Regs<BT * (OT / 8) / kThreads> rv = gather<BT * (OT / 8) / kThreads>([&](int e) {
            const int t = e % BT; return t < n ? as4(ws.vn + (vh * T + r0 + t) * D + v0 + (e / BT) * 8) : kZero4; });
        scatter_rows(rq, QS);
        scatter(rh, [&](int e, uint4 x) { as4(HS + (e / (D / 8)) * LD + (e % (D / 8)) * 8) = x; });
        scatter(rp, [&](int e, uint4 x) { as4(PS + (e / (BT / 8)) * LT + (e % (BT / 8)) * 8) = x; });
        scatter(rv, [&](int e, uint4 x) {
            const int t = e % BT, c8 = (e / BT) * 8;
            const u16 *h = reinterpret_cast<const u16 *>(&x);
#pragma unroll
            for (int q = 0; q < 8; ++q) VST[(c8 + q) * LT + t] = h[q];
        });
    }
    for (int i = tid; i < BT; i += kThreads) Eg[i] = i < n ? expf(ws.G[vh * T + r0 + i]) : 0.f;
    __syncthreads();
    for (int job = wave; job < (BT / 16) * (OT / 16); job += kWaves) {
        const int ti = job / (OT / 16), tj = job % (OT / 16);
        f32x8 sq = {}, sp = {};
        sq = mma(sq, D, QS + ti * 16 * LD, LD, HS + tj * 16 * LD, LD);
        sp = mma(sp, BT, PS + ti * 16 * LT, LT, VST + tj * 16 * LT, LT);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int i = ti * 16 + crow(e), cv = tj * 16 + ccol();
            if (i < n) ws.ot[(vh * T + r0 + i) * D + v0 + cv] = tob(sq[e] * Eg[i] * scale + sp[e] * scale);
        }
    }
}

}  // namespace

size_t chunk_workspace_bytes(int64_t tokens) {
    if (tokens < 1 || tokens > kChunkMaxTokens) return 0;
    return carve(nullptr, tokens, nullptr);
}

hipError_t delta_rule_chunk(const ChunkArgs &a, void *workspace, hipStream_t stream) {
    auto misaligned = [](const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15) != 0; };
    if (a.tokens < 1 || a.tokens > kChunkMaxTokens || !a.q || !a.k || !a.v || !a.g || !a.beta || !a.state || !a.o ||
        !workspace || (reinterpret_cast<uintptr_t>(workspace) & 255) || misaligned(a.v) || misaligned(a.o))  // v, o: 16-byte row moves
        return hipErrorInvalidValue;
    const int64_t T = a.tokens, chunks = (T + BT - 1) / BT;
    if (a.capture32 != nullptr && (a.capture_chunk < 0 || a.capture_chunk > chunks)) return hipErrorInvalidValue;
    Workspace ws{};
    carve(workspace, T, &ws);
    u16 *qtok = ws.ot, *ktok = ws.ot + size_t(T) * KH * D;  // token-major normalised q, k (ot is free until out_kernel)
    if (hipError_t e = l2norm(a.q, reinterpret_cast<__hip_bfloat16 *>(qtok), T * KH, D, 1e-6f, stream); e != hipSuccess) return e;
    if (hipError_t e = l2norm(a.k, reinterpret_cast<__hip_bfloat16 *>(ktok), T * KH, D, 1e-6f, stream); e != hipSuccess) return e;
    const unsigned pblocks = unsigned((T + 15) / 16);
    permute_kernel<true><<<pblocks, kThreads, 0, stream>>>(qtok, ws.qn, T, KH);
    permute_kernel<true><<<pblocks, kThreads, 0, stream>>>(ktok, ws.kn, T, KH);
    permute_kernel<true><<<pblocks, kThreads, 0, stream>>>(reinterpret_cast<const u16 *>(a.v), ws.vt, T, VH);
    auto *state = reinterpret_cast<u16 *>(a.state);
    cumsum_kernel<<<unsigned((chunks * VH + 255) / 256), 256, 0, stream>>>(a.g, a.beta, ws.G, ws.B, T);
    prep_kernel<<<dim3(unsigned(chunks), VH), kThreads, 0, stream>>>(ws, T);
    recur_kernel<<<dim3(VH, D / VT), RW * 32, 0, stream>>>(ws, T, state, a.state_in32, a.state_in32_hi, state,
                                                           a.capture32, a.capture32_hi, a.capture_chunk);
    out_kernel<<<dim3(unsigned(chunks), VH, D / OT), kThreads, 0, stream>>>(ws, T, a.scale);
    permute_kernel<false><<<pblocks, kThreads, 0, stream>>>(ws.ot, reinterpret_cast<u16 *>(a.o), T, VH);
    return hipGetLastError();
}

}  // namespace cere::k5
