// AmdAmd PAR-6 step 4b: P4 dense causal GQA (the contract of executor/cuda/src/kernels/p4_attention.h) on gfx1151's
// WMMA (v_wmma_f32_16x16x16_bf16), written from that header's arithmetic (the CUDA kernel is mma.sync m16n8k16):
//   s_j = F32 mma(q_bf16, k_j) * (scale * log2 e)   (K FP8 E4M3 -> BF16 exactly)
//   online softmax in base 2 per 32-key tile: m_new = max(m, tile max), alpha = 2^(m - m_new), p_j = 2^(s_j - m_new),
//   l = l * alpha + sum_j p_j (F32, unrounded p), o = o * alpha + mma(bf16(p), v); out = bf16(o / l), 0 without keys.
// Layout: the 12 query heads of a KV head are one 16-row WMMA tile (rows 12-15 zero). A wave pair owns one (token, KV
// head): each wave computes the partial q . k over its 128 of the 256 dims, the pair swaps partials through LDS and
// both add them in one order (lo + hi), so both hold the same scores and run the same softmax; each wave then
// accumulates o for its 128 output dims. Block: 4 tokens x 2 halves = 8 waves over 32-key tiles staged in LDS (K
// row-major, V transposed: both WMMA B operands k-contiguous).
// Row-partition invariance (the header's promise): a wave runs only the key tiles that start at or before its own
// token's last key and masks keys past it inside the last one. The contract's fully masked tiles change nothing
// (alpha = 2^0 = 1, p = 0: o * 1 + 0 = o, l * 1 + 0 = l), so skipping them is bit-identical, and a row's bits are a
// function of its position and the keys only: rows [a, b) of a call equal the call of those rows alone.
// Deterministic: fixed key order, fixed reduction orders, no atomics.
// sparse_attention (K9's P4 path): sparse_kernel below.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>

#include <cmath>

#include "kernels/p4_attention.h"

namespace cere::p4 {
namespace {

constexpr int kGroup = kQHeads / kKvHeads;  // 12
constexpr int kKvWidth = kKvHeads * kHeadDim, kQWidth = kQHeads * kHeadDim;
constexpr int kTok = 4, kThreads = kTok * 2 * 32, kKeys = 32, kHalf = kHeadDim / 2;
constexpr int LK = kHeadDim + 8, LV = kKeys + 8;  // padded LDS rows (BF16)
constexpr float kLog2e = 1.4426950408889634f;
using u16 = uint16_t;
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
struct U16x16 { uint4 lo, hi; };

__device__ inline bf16x16 ld16(const u16 *p) {
    return __builtin_bit_cast(bf16x16, U16x16{*reinterpret_cast<const uint4 *>(p), *reinterpret_cast<const uint4 *>(p + 8)});
}
__device__ inline u16 tob(float x) { return __builtin_bit_cast(u16, __float2bfloat16(x)); }
__device__ inline float e4m3(uint32_t c) {  // OCP E4M3 -> float, exact (and exact in BF16)
    // OCP E4M3 -> float, exact and branch-free: the 7 magnitude bits at FP32 bit 20 are 2^(e-127)(1 + m/8)
    // (or the subnormal m 2^-129), times 2^120 is 2^(e-7)(1 + m/8) (m 2^-9); 0x7f is NaN; the sign is OR-ed in.
    const uint32_t a = c & 0x7fu;
    const float v = a == 0x7fu ? __uint_as_float(0x7fc00000u) : __uint_as_float(a << 20) * 0x1p120f;
    return __uint_as_float(__float_as_uint(v) | ((c & 0x80u) << 24));
}
// E4M3 bytes -> BF16 bits: every E4M3 value is exact in BF16, so the FP32 decode's high half is the BF16 (no rounding)
__device__ inline uint32_t e4m3_f32bits(uint32_t c) { return __float_as_uint(e4m3(c)); }
__device__ inline uint4 e4m3x8_bf16(uint2 b) {  // 8 bytes -> 8 BF16 (packed pairs, low element first)
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t w = i < 2 ? b.x : b.y, sh = 16 * (i % 2);
        o[i] = (e4m3_f32bits((w >> sh) & 0xffu) >> 16) | (e4m3_f32bits((w >> (sh + 8)) & 0xffu) & 0xffff0000u);
    }
    return uint4{o[0], o[1], o[2], o[3]};
}
__device__ inline float hmax16(float x) {  // max over the 16 lanes of a half wave
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor(x, o, 32));
    return x;
}
__device__ inline float hsum16(float x) {
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) x += __shfl_xor(x, o, 32);
    return x;
}

// key j's 8 elements at dims c8.. as BF16 bits
template <bool kFp8>
__device__ inline uint4 load8(const void *base, int64_t row, int64_t stride, int col) {
    if constexpr (kFp8) {
        return e4m3x8_bf16(*reinterpret_cast<const uint2 *>(static_cast<const uint8_t *>(base) + row * stride + col));
    } else {
        return *reinterpret_cast<const uint4 *>(static_cast<const u16 *>(base) + row * stride + col);
    }
}

template <bool kFp8>
__global__ void __launch_bounds__(kThreads) dense_kernel(const u16 *__restrict__ q, int64_t tokens, int64_t pos0,
                                                         const void *__restrict__ k, int64_t k_stride,
                                                         const void *__restrict__ v, int64_t v_stride,
                                                         const int64_t *__restrict__ slots, float score_scale,
                                                         u16 *__restrict__ out) {
    __shared__ __attribute__((aligned(16))) u16 KS[kKeys * LK];        // K [key][dim]
    __shared__ __attribute__((aligned(16))) u16 VT[kHeadDim * LV];     // V^T [dim][key]
    __shared__ __attribute__((aligned(16))) float SX[kTok][2][16 * kKeys];  // each half's partial scores
    __shared__ __attribute__((aligned(16))) u16 PS[kTok][2][16 * LV];  // bf16(p) [row][key], one copy per wave
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, r = lane % 16, hl = lane / 16;
    const int tw = wave / 2, half = wave % 2, kvh = blockIdx.y;
    const int64_t nblk = (tokens + kTok - 1) / kTok, blk = nblk - 1 - blockIdx.x;  // longest causal rows first
    const int64_t t = blk * kTok + tw;
    const bool live = t < tokens;
    const int64_t last = pos0 + (live ? t : tokens - 1);  // this wave's last key
    const int64_t blk_last = pos0 + (blk * kTok + kTok - 1 < tokens ? blk * kTok + kTok - 1 : tokens - 1);
    // q fragments: row r = query head kvh * 12 + r (rows 12..15 zero), dims half * 128 + 16 s ..
    bf16x16 qf[kHalf / 16];
    {
        const u16 *qrow = q + (live ? t : 0) * kQWidth + int64_t(kvh * kGroup + (r < kGroup ? r : 0)) * kHeadDim + half * kHalf;
#pragma unroll
        for (int s = 0; s < kHalf / 16; ++s) {
            qf[s] = ld16(qrow + 16 * s);
            if (r >= kGroup || !live) qf[s] = bf16x16{};
        }
    }
    f32x8 o[kHalf / 16] = {};
    float m[8], l[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { m[i] = -INFINITY; l[i] = 0.f; }
    for (int64_t kb = 0; kb <= blk_last; kb += kKeys) {
        __syncthreads();  // the previous tile's LDS reads are done
        // stage keys kb .. kb + 31 (zero past blk_last): thread -> (key tid / 8, 32 dims at (tid % 8) * 32)
        {
            const int key = tid / 8, d0 = (tid % 8) * 32;
            const int64_t j = kb + key;
            const bool ok = j <= blk_last;
            const int64_t row = ok ? (slots ? slots[j] : j) : 0;
            uint4 kr[4], vr[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                kr[c] = ok ? load8<kFp8>(k, row, k_stride, kvh * kHeadDim + d0 + 8 * c) : uint4{0, 0, 0, 0};
                vr[c] = ok ? load8<kFp8>(v, row, v_stride, kvh * kHeadDim + d0 + 8 * c) : uint4{0, 0, 0, 0};
            }
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                *reinterpret_cast<uint4 *>(KS + key * LK + d0 + 8 * c) = kr[c];
                const u16 *h = reinterpret_cast<const u16 *>(&vr[c]);
#pragma unroll
                for (int e = 0; e < 8; ++e) VT[(d0 + 8 * c + e) * LV + key] = h[e];
            }
        }
        __syncthreads();
        const bool run = live && kb <= last;  // tiles past this row's last key: skipped (= the masked no-op)
        f32x8 s[2] = {};
        if (run) {
#pragma unroll
            for (int st = 0; st < 2; ++st)
#pragma unroll
                for (int ks = 0; ks < kHalf / 16; ++ks)
                    s[st] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf[ks], ld16(KS + (st * 16 + r) * LK + half * kHalf + 16 * ks), s[st]);
#pragma unroll
            for (int st = 0; st < 2; ++st)
#pragma unroll
                for (int i = 0; i < 8; ++i) SX[tw][half][(2 * i + hl) * kKeys + st * 16 + r] = s[st][i];
        }
        __syncthreads();
        if (run) {
            // full scores = lo + hi (one order for both waves), scaled, masked past `last`
#pragma unroll
            for (int st = 0; st < 2; ++st)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int idx = (2 * i + hl) * kKeys + st * 16 + r;
                    const float full = (SX[tw][0][idx] + SX[tw][1][idx]) * score_scale;
                    s[st][i] = kb + st * 16 + r <= last ? full : -INFINITY;
                }
            float alpha[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float mx = fmaxf(hmax16(s[0][i]), hmax16(s[1][i]));
                const float mn = fmaxf(m[i], mx);
                alpha[i] = exp2f(m[i] - mn);  // m = -inf (first tile): 0
                const float p0 = exp2f(s[0][i] - mn), p1 = exp2f(s[1][i] - mn);
                l[i] = fmaf(l[i], alpha[i], hsum16(p0) + hsum16(p1));
                m[i] = mn;
                PS[tw][half][(2 * i + hl) * LV + r] = tob(p0);
                PS[tw][half][(2 * i + hl) * LV + 16 + r] = tob(p1);
            }
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
            const bf16x16 pa0 = ld16(PS[tw][half] + r * LV), pa1 = ld16(PS[tw][half] + r * LV + 16);
#pragma unroll
            for (int dt = 0; dt < kHalf / 16; ++dt) {
#pragma unroll
                for (int i = 0; i < 8; ++i) o[dt][i] *= alpha[i];
                const u16 *vb = VT + (half * kHalf + dt * 16 + r) * LV;
                o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pa0, ld16(vb), o[dt]);
                o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pa1, ld16(vb + 16), o[dt]);
            }
        }
    }
    if (!live) return;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = 2 * i + hl;
        if (row >= kGroup) continue;
        u16 *orow = out + t * kQWidth + int64_t(kvh * kGroup + row) * kHeadDim + half * kHalf;
#pragma unroll
        for (int dt = 0; dt < kHalf / 16; ++dt) orow[dt * 16 + r] = l[i] > 0.f ? tob(o[dt][i] / l[i]) : u16(0);
    }
}

bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

// 16 elements of key `row` at dims c.. as a WMMA operand (FP8 -> BF16 exactly); zeros for row < 0
template <bool kFp8>
__device__ inline bf16x16 key16(const void *base, int64_t row, int64_t stride, int c) {
    if (row < 0) return bf16x16{};
    if constexpr (kFp8) {
        const uint4 b = *reinterpret_cast<const uint4 *>(static_cast<const uint8_t *>(base) + row * stride + c);
        return __builtin_bit_cast(bf16x16, U16x16{e4m3x8_bf16(uint2{b.x, b.y}), e4m3x8_bf16(uint2{b.z, b.w})});
    } else {
        return ld16(static_cast<const u16 *>(base) + row * stride + c);
    }
}

// Sparse GQA (P4's sparse_attention): the dense kernel's wave pairs and arithmetic over each row's own index list,
// 32 entries per tile in entry order (entries < 0 masked wherever they sit). No K/V tile is shared between rows, so K
// feeds WMMA straight from global (FP8 converted in registers) and each wave transposes its V slice through a small
// LDS buffer, 32 keys x 32 dims at a time. A tile with no valid entry for a row is that row's no-op and is skipped.
template <bool kFp8>
__global__ void __launch_bounds__(kThreads) sparse_kernel(const u16 *__restrict__ q, int64_t q_stride,
                                                          const int32_t *__restrict__ indices, int64_t row_width,
                                                          int64_t rows, const void *__restrict__ k, int64_t k_stride,
                                                          const void *__restrict__ v, int64_t v_stride,
                                                          const int64_t *__restrict__ slots, float score_scale,
                                                          u16 *__restrict__ out, int64_t out_stride,
                                                          const u16 *__restrict__ kbf, const u16 *__restrict__ vbf,
                                                          int64_t cap) {
    constexpr int kChunk = 32, LC = kKeys + 8;
    __shared__ __attribute__((aligned(16))) float SX[kTok][2][16 * kKeys];
    __shared__ __attribute__((aligned(16))) u16 PS[kTok][2][16 * LV];
    __shared__ __attribute__((aligned(16))) u16 VC[kTok * 2][kChunk * LC];  // per wave: V^T chunk [dim][key]
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, r = lane % 16, hl = lane / 16;
    const int tw = wave / 2, half = wave % 2, kvh = blockIdx.y;
    const int64_t t = int64_t(blockIdx.x) * kTok + tw;
    const bool live = t < rows;
    const int32_t *irow = indices + (live ? t : 0) * row_width;
    bf16x16 qf[kHalf / 16];
    {
        const u16 *qrow = q + (live ? t : 0) * q_stride + int64_t(kvh * kGroup + (r < kGroup ? r : 0)) * kHeadDim + half * kHalf;
#pragma unroll
        for (int s = 0; s < kHalf / 16; ++s) {
            qf[s] = ld16(qrow + 16 * s);
            if (r >= kGroup || !live) qf[s] = bf16x16{};
        }
    }
    f32x8 o[kHalf / 16] = {};
    float m[8], l[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { m[i] = -INFINITY; l[i] = 0.f; }
    u16 *vc = VC[wave];
    for (int64_t kb = 0; kb < row_width; kb += kKeys) {
        // this lane's entry (lane = key of the tile) and its cache row
        const int64_t e = kb + lane;
        const int32_t pj = live && e < row_width ? irow[e] : -1;
        // positions < cap read the call's BF16 copy (decode_kv_kernel: the same bits key16<true> decodes)
        const int64_t brow = kFp8 && pj >= 0 && pj < cap ? int64_t(pj) : -1;
        const int64_t myrow = pj < 0 ? -1 : brow >= 0 ? 0 : (slots ? slots[pj] : int64_t(pj));
        const bool run = __ballot(myrow >= 0) != 0;  // wave-uniform: a tile without valid entries is skipped
        f32x8 s[2] = {};
        if (run) {
#pragma unroll
            for (int st = 0; st < 2; ++st) {
                const int64_t krow = __shfl(myrow, st * 16 + r, 32), kbrow = __shfl(brow, st * 16 + r, 32);
#pragma unroll
                for (int ks = 0; ks < kHalf / 16; ++ks) {
                    const int c = kvh * kHeadDim + half * kHalf + 16 * ks;
                    const bf16x16 kf = kbrow >= 0 ? key16<false>(kbf, kbrow, kKvWidth, c) : key16<kFp8>(k, krow, k_stride, c);
                    s[st] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf[ks], kf, s[st]);
                }
            }
#pragma unroll
            for (int st = 0; st < 2; ++st)
#pragma unroll
                for (int i = 0; i < 8; ++i) SX[tw][half][(2 * i + hl) * kKeys + st * 16 + r] = s[st][i];
        }
        __syncthreads();  // both halves' partial scores (the pair runs the same `run`: same token, same entries)
        if (run) {
            const bool valid0 = __shfl(myrow, r, 32) >= 0, valid1 = __shfl(myrow, 16 + r, 32) >= 0;
#pragma unroll
            for (int st = 0; st < 2; ++st)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int idx = (2 * i + hl) * kKeys + st * 16 + r;
                    const float full = (SX[tw][0][idx] + SX[tw][1][idx]) * score_scale;
                    s[st][i] = (st ? valid1 : valid0) ? full : -INFINITY;
                }
            float alpha[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float mx = fmaxf(hmax16(s[0][i]), hmax16(s[1][i]));
                const float mn = fmaxf(m[i], mx);
                alpha[i] = exp2f(m[i] - mn);
                const float p0 = exp2f(s[0][i] - mn), p1 = exp2f(s[1][i] - mn);
                l[i] = fmaf(l[i], alpha[i], hsum16(p0) + hsum16(p1));
                m[i] = mn;
                PS[tw][half][(2 * i + hl) * LV + r] = tob(p0);
                PS[tw][half][(2 * i + hl) * LV + 16 + r] = tob(p1);
            }
#pragma unroll
            for (int dt = 0; dt < kHalf / 16; ++dt)
#pragma unroll
                for (int i = 0; i < 8; ++i) o[dt][i] *= alpha[i];
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            const bf16x16 pa0 = ld16(PS[tw][half] + r * LV), pa1 = ld16(PS[tw][half] + r * LV + 16);
            for (int c0 = 0; c0 < kHalf; c0 += kChunk) {
                // V^T chunk: lane = key, its 32 dims half * 128 + c0 .. -> vc[dim][key]
                const int cc = kvh * kHeadDim + half * kHalf + c0;
                const bf16x16 lo = brow >= 0 ? key16<false>(vbf, brow, kKvWidth, cc) : key16<kFp8>(v, myrow, v_stride, cc);
                const bf16x16 hi = brow >= 0 ? key16<false>(vbf, brow, kKvWidth, cc + 16) : key16<kFp8>(v, myrow, v_stride, cc + 16);
                const U16x16 lo16 = __builtin_bit_cast(U16x16, lo), hi16 = __builtin_bit_cast(U16x16, hi);
                const u16 *a = reinterpret_cast<const u16 *>(&lo16), *b = reinterpret_cast<const u16 *>(&hi16);
#pragma unroll
                for (int d = 0; d < 16; ++d) { vc[d * LC + lane] = a[d]; vc[(16 + d) * LC + lane] = b[d]; }
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int dt = (c0 + 16 * j) / 16;
                    const u16 *vb = vc + (16 * j + r) * LC;
                    o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pa0, ld16(vb), o[dt]);
                    o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pa1, ld16(vb + 16), o[dt]);
                }
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();  // the chunk's reads are done before the next overwrites it
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            }
        }
        __syncthreads();  // SX is rewritten by the next tile
    }
    if (!live) return;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = 2 * i + hl;
        if (row >= kGroup) continue;
        u16 *orow = out + t * out_stride + int64_t(kvh * kGroup + row) * kHeadDim + half * kHalf;
#pragma unroll
        for (int dt = 0; dt < kHalf / 16; ++dt) orow[dt * 16 + r] = l[i] > 0.f ? tob(o[dt][i] / l[i]) : u16(0);
    }
}

}  // namespace

hipError_t dense_attention(const bf16 *q, int64_t tokens, int64_t pos0, const bf16 *k, int64_t k_stride, const bf16 *v,
                           int64_t v_stride, const uint8_t *k_cache, const uint8_t *v_cache, const int64_t *slots,
                           float scale, bf16 *out, hipStream_t stream) {
    const bool bf16_kv = k && v && !k_cache && !v_cache, fp8_kv = !k && !v && k_cache && v_cache;
    if (!q || !out || tokens < 1 || pos0 < 0 || !(bf16_kv || fp8_kv) || !aligned16(q) || !aligned16(out) ||
        (bf16_kv && (!aligned16(k) || !aligned16(v) || k_stride % 8 || v_stride % 8 || k_stride < kKvWidth || v_stride < kKvWidth)) ||
        (fp8_kv && (!aligned16(k_cache) || !aligned16(v_cache))))
        return hipErrorInvalidValue;
    const dim3 grid(unsigned((tokens + kTok - 1) / kTok), kKvHeads);
    const float ss = scale * kLog2e;
    const auto *qq = reinterpret_cast<const u16 *>(q);
    auto *oo = reinterpret_cast<u16 *>(out);
    if (bf16_kv)
        dense_kernel<false><<<grid, kThreads, 0, stream>>>(qq, tokens, pos0, k, k_stride, v, v_stride, nullptr, ss, oo);
    else
        dense_kernel<true><<<grid, kThreads, 0, stream>>>(qq, tokens, pos0, k_cache, kKvWidth, v_cache, kKvWidth, slots, ss, oo);
    return hipGetLastError();
}

namespace {
// AmdAmd PAR-6 step 16: on the FP8 cache every query row decoded its own 2,048 selected keys (E4M3 -> BF16 is ~8 VALU
// ops an element: the kernel ran 67 ms per 4,096 rows vs 31 on BF16 rows). With scratch (K9's select logits tile,
// dead by attention: qsa_attention's `scratch`), the call's positions 0 .. n - 1 (n = its largest index + 1) are
// decoded once into BF16 K / V copies [position, 512] and the kernel reads those; positions past the copy's capacity
// keep the per-row decode. The copy holds key16<true>'s exact values, so every output bit is unchanged.
constexpr int kDecThreads = 256;
__global__ void init_count_kernel(int32_t *__restrict__ n) { *n = 0; }
// n = 1 + the largest index (entries are positions or -1), by block maxima and an atomic max (integers: any order)
__global__ void __launch_bounds__(256) max_index_kernel(const int32_t *__restrict__ idx, int64_t count,
                                                        int32_t *__restrict__ n) {
    __shared__ int32_t red[8];
    int32_t mx = -1;
    for (int64_t i = int64_t(blockIdx.x) * 256 + threadIdx.x; i < count; i += int64_t(gridDim.x) * 256) mx = max(mx, idx[i]);
    for (int o = 16; o > 0; o >>= 1) mx = max(mx, __shfl_xor(mx, o, 32));
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; ++w) mx = max(mx, red[w]);
        if (mx >= 0) atomicMax(n, mx + 1);
    }
}
__global__ void __launch_bounds__(kDecThreads) decode_kv_kernel(const uint8_t *__restrict__ kc, const uint8_t *__restrict__ vc,
                                                                const int64_t *__restrict__ slots,
                                                                const int32_t *__restrict__ n, int64_t cap,
                                                                u16 *__restrict__ kb, u16 *__restrict__ vb) {
    constexpr int kPer = kKvWidth / 16;  // 16-byte chunks per position
    const int64_t i = int64_t(blockIdx.x) * kDecThreads + threadIdx.x, j = i / kPer;
    if (j >= cap || j >= *n) return;
    const int c = int(i % kPer) * 16;
    const int64_t src = slots ? slots[j] : j;
    const bf16x16 kv = key16<true>(kc, src, kKvWidth, c), vv = key16<true>(vc, src, kKvWidth, c);
    *reinterpret_cast<bf16x16 *>(kb + j * kKvWidth + c) = kv;
    *reinterpret_cast<bf16x16 *>(vb + j * kKvWidth + c) = vv;
}
}  // namespace

hipError_t sparse_attention(const bf16 *q, int64_t q_stride, const int32_t *indices, int64_t row_width, int64_t rows,
                            const bf16 *k, int64_t k_stride, const bf16 *v, int64_t v_stride, const uint8_t *k_cache,
                            const uint8_t *v_cache, const int64_t *slots, float scale, bf16 *out, int64_t out_stride,
                            hipStream_t stream, void *scratch, size_t scratch_bytes) {
    const bool bf16_kv = k && v && !k_cache && !v_cache, fp8_kv = !k && !v && k_cache && v_cache;
    if (!q || !out || !indices || rows < 1 || row_width < 1 || !(bf16_kv || fp8_kv) || !aligned16(q) || !aligned16(out) ||
        q_stride % 8 || out_stride % 8 || q_stride < kQWidth || out_stride < kQWidth ||
        (bf16_kv && (!aligned16(k) || !aligned16(v) || k_stride % 8 || v_stride % 8 || k_stride < kKvWidth || v_stride < kKvWidth)) ||
        (fp8_kv && (!aligned16(k_cache) || !aligned16(v_cache))))
        return hipErrorInvalidValue;
    const dim3 grid(unsigned((rows + kTok - 1) / kTok), kKvHeads);
    const float ss = scale * kLog2e;
    const auto *qq = reinterpret_cast<const u16 *>(q);
    auto *oo = reinterpret_cast<u16 *>(out);
    if (bf16_kv)
        sparse_kernel<false><<<grid, kThreads, 0, stream>>>(qq, q_stride, indices, row_width, rows, k, k_stride, v, v_stride, nullptr, ss, oo, out_stride, nullptr, nullptr, 0);
    else {
        // scratch: [0, 256) the position count, then K and V copies of `cap` positions each
        const size_t per = size_t(kKvWidth) * sizeof(u16);
        const int64_t cap = scratch && aligned16(scratch) && scratch_bytes > 256 + 2 * per ? int64_t((scratch_bytes - 256) / (2 * per)) : 0;
        u16 *kb = nullptr, *vb = nullptr;
        if (cap > 0) {
            auto *base = static_cast<uint8_t *>(scratch);
            auto *n = reinterpret_cast<int32_t *>(base);
            kb = reinterpret_cast<u16 *>(base + 256);
            vb = kb + size_t(cap) * kKvWidth;
            init_count_kernel<<<1, 1, 0, stream>>>(n);
            const int64_t count = rows * row_width;
            max_index_kernel<<<unsigned(std::min<int64_t>(512, (count + 255) / 256)), 256, 0, stream>>>(indices, count, n);
            const int64_t chunks = cap * (kKvWidth / 16);
            decode_kv_kernel<<<unsigned((chunks + kDecThreads - 1) / kDecThreads), kDecThreads, 0, stream>>>(k_cache, v_cache, slots, n, cap, kb, vb);
        }
        sparse_kernel<true><<<grid, kThreads, 0, stream>>>(qq, q_stride, indices, row_width, rows, k_cache, kKvWidth, v_cache, kKvWidth, slots, ss, oo, out_stride, kb, vb, cap);
    }
    return hipGetLastError();
}

}  // namespace cere::p4
