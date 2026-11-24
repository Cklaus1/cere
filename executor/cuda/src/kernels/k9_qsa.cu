// K9 QSA sparse attention (m22c design note D3); contracts in k9_qsa.h. The prep arithmetic follows SGLang's fused
// indexer kernels (`sglang/kernels/jit/csrc/attention/qsa_indexer.cuh`) step by step: the Gemma norm's F32 sum of
// squares in flashinfer's partial layout (lane l < 16 sums elements [8l, 8l + 8), xor butterfly), rsqrtf, one rounding
// of x * nf * (1 + w); RoPE with cos/sin and every product rounded to BF16; the group mean accumulated in F32 in member
// order and rounded to BF16 before the norm. Block logits use BF16 tensor-core MMA with F32 accumulation (as SGLang's
// TileLang MQA kernel); top-512 is an exact radix select on fast_topk's order-preserving uint32 key.
//
// math_reference: sglang
#include "kernels/k9_qsa.h"

#include "cere_model_constants.h"

#include <cmath>

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <mma.h>

#include "kernels/p4_attention.h"

namespace cere::k9 {
namespace {

namespace cg = cooperative_groups;

constexpr int kWarp = 32;
constexpr int kHalfRotary = kRotaryDim / 2;
constexpr int kGroup = kQHeads / kKvHeads;  // query heads per KV head (12)
constexpr int kKvWidth = kKvHeads * kHeadDim;
constexpr int kQWidth = kQHeads * kHeadDim;

// Logits: 8 warps, 32 query rows (128 index-q vectors) per block, 256 keys per block in sub-tiles of 32.
constexpr int kLogitThreads = 256;
constexpr int kLogitRows = 32;
constexpr int kLogitVectors = kLogitRows * kIndexHeads;  // 128
constexpr int kLogitKeysPerBlock = 256;
constexpr int kLogitKeyTile = 32;
constexpr int kLdsQ = kIndexHeadDim + 8;  // padded BF16 rows (272 B: every 16-row fragment stays 32-byte aligned)
// Small-call logits (logits_small_kernel, track LongK9 LK9-02): 4 warps, 4 query rows (16 vectors: one warp fragment
// of logits_kernel) per block, 128 keys per block, one 32-key sub-tile per warp.
constexpr int kLogitSmallThreads = 128;
constexpr int kLogitSmallRows = 4;
constexpr int kLogitSmallVectors = kLogitSmallRows * kIndexHeads;  // 16
constexpr int kLogitSmallWarps = kLogitSmallThreads / kWarp;
constexpr int kLogitSmallKeysPerBlock = kLogitSmallWarps * kLogitKeyTile;  // 128

// Top-k: one block per query row.
constexpr int kTopkThreads = 512;
constexpr int kTopkWarps = kTopkThreads / kWarp;
// Cluster top-k (topk_cluster_kernel): kTkCluster blocks of kTopkThreads per row, at most kTkPerThread visible blocks
// per thread, so calls whose logits rows hold at most kTkMaxBlocks blocks (262,144 tokens).
constexpr int kTkCluster = 8;
constexpr int kTkPerThread = 16;
constexpr int64_t kTkMaxBlocks = int64_t{kTkCluster} * kTopkThreads * kTkPerThread;
// Prefill top-k (topk_prefill_kernel<kPer>): one block of kTopkThreads per row, at most kPer (<= kTpMaxPer) visible
// blocks per thread in registers, so logits rows of at most kTpMaxBlocks blocks (131,072 tokens).
constexpr int kTpMaxPer = 64;
constexpr int64_t kTpMaxBlocks = int64_t{kTopkThreads} * kTpMaxPer;

// Sparse attention: one block per (row, KV head, split), 128 threads, 32 keys per tile.
constexpr int kAttnThreads = 128;
constexpr int kAttnKeyTile = 32;
constexpr int kAttnKStride = kHeadDim + 2;  // BF16 elements: 129 words per key row, the 32 keys hit 32 banks
constexpr int kSplits = 16;
// Split attention (split_attention_kernel, track LongK9 LK9-02): attention_kernel's splits > 1 call on the FP8 cache,
// 12 warps, every key of the split staged at once (kSaKeys covers per_split for every end <= kRowWidth).
constexpr int kSaThreads = 384;
constexpr int kSaWarps = kSaThreads / kWarp;
constexpr int kSaTiles = 5;
constexpr int kSaKeys = kSaTiles * kAttnKeyTile;  // 160
constexpr int kSaKStride = kHeadDim + 16;  // FP8 bytes per staged K row: 8 lanes' 16-byte reads hit 32 banks
constexpr int kSaScoreHeads = 6;          // score phase: thread = (key, 6 heads), warps 0 .. 9
constexpr int kSaPvHeads = 4;             // P.V phase: thread = (2 dims, 4 heads)

static_assert(kIndexHeadDim == 128 && kIndexHeads == 4, "the prep and logits kernels assume 4 heads x 128 dims");
static_assert(kGroup == 12 && kHeadDim == 256, "the attention kernel assumes 12 query heads per KV head, dim 256");
static_assert(((kRowWidth + kSplits - 1) / kSplits + kAttnKeyTile - 1) / kAttnKeyTile * kAttnKeyTile <= kSaKeys,
              "split_attention_kernel stages at most kSaKeys keys per split");
static_assert(kSaWarps == kGroup && kSaKeys * (kGroup / kSaScoreHeads) <= kSaThreads &&
                  kHeadDim / 2 * (kGroup / kSaPvHeads) == kSaThreads,
              "split_attention_kernel: one warp per head (softmax), (key, 6 heads) scores, (2 dims, 4 heads) P.V");

size_t align256(size_t n) { return (n + 255) & ~static_cast<size_t>(255); }

bool aligned(const void* p, uintptr_t a) { return (reinterpret_cast<uintptr_t>(p) & (a - 1)) == 0; }

cere_status status_of(cudaError_t err) {
    if (err == cudaSuccess) return CERE_OK;
    if (err == cudaErrorInvalidValue) return CERE_ERR_ARGUMENT;
    return CERE_ERR_INTERNAL;
}

__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// FP8 E4M3 byte -> float, exact (scale 1.0); as k6_attention.cu.
__device__ __forceinline__ float fp8_to_float(uint32_t byte) {
    const uint32_t sign = (byte & 0x80u) << 24;
    const uint32_t e = (byte >> 3) & 0xfu;
    const uint32_t m = byte & 0x7u;
    float f;
    if (e == 0xfu && m == 0x7u) {
        f = __uint_as_float(0x7fc00000u);
    } else if (e == 0) {
        f = static_cast<float>(m) * 0.001953125f;  // m * 2^-9
    } else {
        f = __uint_as_float(((e + 120u) << 23) | (m << 20));
    }
    return __uint_as_float(__float_as_uint(f) | sign);
}

// fast_topk.cuh convert_to_uint32: order-preserving key of a float.
__device__ __forceinline__ uint32_t order_key(float x) {
    const uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// -- index q / compressed-key prep (one warp per 128-dim row) ----------------------------------------------------

// Gemma RMSNorm of the F32 values x[4 lane .. 4 lane + 3] (BF16-exact) of one row whose BF16 copy is in `row`
// (shared); writes the normed BF16 row back to `row`. As qsa_gemma_norm_row / the compress kernel's norm.
__device__ __forceinline__ void gemma_norm_row(const float (&x)[4], bf16* row, const bf16* __restrict__ weight,
                                               float eps, int lane) {
    float ss = 0.0f;
    if (lane < kIndexHeadDim / 8) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float f = __bfloat162float(row[lane * 8 + j]);
            ss = __fmaf_rn(f, f, ss);
        }
    }
#pragma unroll
    for (int mask = kWarp / 2; mask >= 1; mask >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, mask, kWarp));
    const float nf = rsqrtf(__fadd_rn(__fdiv_rn(ss, static_cast<float>(kIndexHeadDim)), eps));
    __syncwarp();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane * 4 + i;
        const float wf = __bfloat162float(weight[d]);
        row[d] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x[i], nf), __fadd_rn(1.0f, wf)));
    }
    __syncwarp();
}

// NeoX RoPE over the first 64 dims of the normed row (shared), as qsa_mrope_apply (text positions: every mRoPE
// axis carries the same position). Writes the 4 dims of this lane to out.
__device__ __forceinline__ void rope_row(const bf16* row, const float* __restrict__ cs, bf16* __restrict__ out,
                                         int lane) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane * 4 + i;
        bf16 o;
        if (d < kHalfRotary) {
            const float c = bf16_round(cs[d]);
            const float s = bf16_round(cs[kHalfRotary + d]);
            const float nd = __bfloat162float(row[d]);
            const float np = __bfloat162float(row[d + kHalfRotary]);
            o = __float2bfloat16_rn(__fsub_rn(bf16_round(__fmul_rn(nd, c)), bf16_round(__fmul_rn(np, s))));
        } else if (d < kRotaryDim) {
            const int p = d - kHalfRotary;
            const float c = bf16_round(cs[p]);
            const float s = bf16_round(cs[kHalfRotary + p]);
            const float nd = __bfloat162float(row[d]);
            const float np = __bfloat162float(row[p]);
            o = __float2bfloat16_rn(__fadd_rn(bf16_round(__fmul_rn(nd, c)), bf16_round(__fmul_rn(np, s))));
        } else {
            o = row[d];
        }
        out[d] = o;
    }
}

// Bounds' clamp (k9_qsa.h, audit KernelsAttention F3): i inside [lo, hi) is returned as is; any other value gives the
// nearest end and stores kBoundFault to *fault (null: none). hi > lo (the entry points check it).
__device__ __forceinline__ int64_t bounded(int64_t i, int64_t lo, int64_t hi, uint32_t* fault) {
    const uint64_t span = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo);
    if (static_cast<uint64_t>(i) - static_cast<uint64_t>(lo) < span) return i;
    if (fault != nullptr) *reinterpret_cast<volatile uint32_t*>(fault) = kBoundFault;
    return i < lo ? lo : hi - 1;
}

// The rope_table row of rotary position `pos`, clamped into the call's bounds.
__device__ __forceinline__ const float* rope_at(const float* __restrict__ cos_sin, const Bounds& b, int64_t pos) {
    return cos_sin + bounded(pos, b.rope_first, b.rope_end, b.fault) * kRotaryDim;
}

// Block = one token, warp h = index head h.
__global__ void __launch_bounds__(kIndexHeads * kWarp) index_q_kernel(const bf16* __restrict__ qk, int64_t qk_stride,
                                                                    const bf16* __restrict__ weight, float eps,
                                                                    const float* __restrict__ cos_sin, Bounds b,
                                                                    const int64_t* __restrict__ positions,
                                                                    int64_t pos0, bf16* __restrict__ q_idx) {
    __shared__ bf16 rows[kIndexHeads][kIndexHeadDim];
    const int64_t t = blockIdx.x;
    const int h = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    const bf16* x_row = qk + t * qk_stride + h * kIndexHeadDim;
    float x[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bf16 v = x_row[lane * 4 + i];
        rows[h][lane * 4 + i] = v;
        x[i] = __bfloat162float(v);
    }
    __syncwarp();
    gemma_norm_row(x, rows[h], weight, eps, lane);
    const int64_t pos = positions ? positions[t] : pos0 + t;
    rope_row(rows[h], rope_at(cos_sin, b, pos), q_idx + (t * kIndexHeads + h) * kIndexHeadDim, lane);
}

// Group g (complete after this call) by one warp (lane < 32) into `row` (shared, kIndexHeadDim BF16).
__device__ __forceinline__ void compress_group(const bf16* __restrict__ k_idx, int64_t k_stride,
                                               const bf16* __restrict__ weight, float eps,
                                               const float* __restrict__ cos_sin, const Bounds& b,
                                               const int64_t* __restrict__ positions,
                                               const int64_t* __restrict__ slots, int64_t pos0, int64_t g,
                                               const bf16* __restrict__ ring, bf16* __restrict__ keys, bf16* row,
                                               int lane) {
    float acc[4];
    for (int r = 0; r < kRatio; ++r) {
        const int64_t p = g * kRatio + r;
        const bf16* src = p < pos0 ? ring + (p % kRatio) * kIndexHeadDim : k_idx + (p - pos0) * k_stride;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float f = __bfloat162float(src[lane * 4 + i]);
            acc[i] = r == 0 ? f : __fadd_rn(acc[i], f);
        }
    }
    const float inv_ratio = 1.0f / static_cast<float>(kRatio);
    float m[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bf16 v = __float2bfloat16_rn(__fmul_rn(acc[i], inv_ratio));
        m[i] = __bfloat162float(v);
        row[lane * 4 + i] = v;
    }
    __syncwarp();
    gemma_norm_row(m, row, weight, eps, lane);
    const int64_t first = g * kRatio;
    int64_t pos;
    if (first >= pos0) {
        pos = positions ? positions[first - pos0] : first;
    } else {
        pos = positions ? positions[0] - (pos0 - first) : first;  // text positions are consecutive
    }
    const int64_t slot = slots ? slots[first] / kRatio : g;
    rope_row(row, rope_at(cos_sin, b, pos), keys + slot * kIndexHeadDim, lane);
}

// Warp w of block b = group g_lo + 4 b + w (complete after this call).
__global__ void __launch_bounds__(4 * kWarp) compress_kernel(const bf16* __restrict__ k_idx, int64_t k_stride,
                                                             const bf16* __restrict__ weight, float eps,
                                                             const float* __restrict__ cos_sin, Bounds b,
                                                             const int64_t* __restrict__ positions,
                                                             const int64_t* __restrict__ slots, int64_t pos0,
                                                             int64_t g_lo, int64_t groups,
                                                             const bf16* __restrict__ ring, bf16* __restrict__ keys) {
    __shared__ bf16 rows[4][kIndexHeadDim];
    const int w = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    const int64_t gi = static_cast<int64_t>(blockIdx.x) * 4 + w;
    if (gi >= groups) return;
    compress_group(k_idx, k_stride, weight, eps, cos_sin, b, positions, slots, pos0, g_lo + gi, ring, keys, rows[w],
                   lane);
}

// -- mRoPE variants (contract C5, M2.13-L): image sessions' prompt rows ---------------------------------------------

// The mRoPE axis (0 t, 1 h, 2 w) of rotary frequency i under Qwen3.8's interleaved mrope_section [11, 11, 10], as
// K1's mrope_axis (k1_primitives.cu:153-162) and SGLang's QSAIndexer._rope_axis_map (qsa_indexer.py:147-153).
// The sections as scalars: a host-side constexpr array is not usable in device code, its elements as constants are.
constexpr int kMropeSectionH9 = static_cast<int>(gen::kMropeSection[1]);
constexpr int kMropeSectionW9 = static_cast<int>(gen::kMropeSection[2]);
__device__ __forceinline__ int mrope_axis_k9(int i) {
    if (i % 3 == 1 && i < 3 * kMropeSectionH9) return 1;
    if (i % 3 == 2 && i < 3 * kMropeSectionW9) return 2;
    return 0;
}

// rope_row with frequency i's cos/sin from the rope_table row of axis mrope_axis_k9(i) (cs3[a] = row of the axis-a
// position), as qsa_mrope_apply (qsa_indexer.cuh:106-125). Same BF16 rounding sequence as rope_row, so three equal
// rows give rope_row's bytes.
__device__ __forceinline__ void rope_row_mrope(const bf16* row, const float* __restrict__ cs_t,
                                               const float* __restrict__ cs_h, const float* __restrict__ cs_w,
                                               bf16* __restrict__ out, int lane) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane * 4 + i;
        bf16 o;
        if (d < kRotaryDim) {
            const int f = d < kHalfRotary ? d : d - kHalfRotary;
            const int axis = mrope_axis_k9(f);
            const float* cs = axis == 0 ? cs_t : (axis == 1 ? cs_h : cs_w);
            const float c = bf16_round(cs[f]);
            const float s = bf16_round(cs[kHalfRotary + f]);
            const float nd = __bfloat162float(row[d]);
            if (d < kHalfRotary) {
                const float np = __bfloat162float(row[d + kHalfRotary]);
                o = __float2bfloat16_rn(__fsub_rn(bf16_round(__fmul_rn(nd, c)), bf16_round(__fmul_rn(np, s))));
            } else {
                const float np = __bfloat162float(row[f]);
                o = __float2bfloat16_rn(__fadd_rn(bf16_round(__fmul_rn(nd, c)), bf16_round(__fmul_rn(np, s))));
            }
        } else {
            o = row[d];
        }
        out[d] = o;
    }
}

// rope_row_mrope at column c of positions3 ([3][p3_stride], axis-major), each axis' row clamped into the bounds.
__device__ __forceinline__ void rope_row_p3(const bf16* row, const float* __restrict__ cos_sin, const Bounds& b,
                                            const int64_t* __restrict__ positions3, int64_t p3_stride, int64_t c,
                                            bf16* __restrict__ out, int lane) {
    rope_row_mrope(row, rope_at(cos_sin, b, positions3[c]), rope_at(cos_sin, b, positions3[p3_stride + c]),
                   rope_at(cos_sin, b, positions3[2 * p3_stride + c]), out, lane);
}

// index_q_kernel with token t rotated at positions3 column p3_lead + t.
__global__ void __launch_bounds__(kIndexHeads * kWarp) index_q_mrope_kernel(
    const bf16* __restrict__ qk, int64_t qk_stride, const bf16* __restrict__ weight, float eps,
    const float* __restrict__ cos_sin, Bounds b, const int64_t* __restrict__ positions3, int64_t p3_stride,
    int64_t p3_lead, bf16* __restrict__ q_idx) {
    __shared__ bf16 rows[kIndexHeads][kIndexHeadDim];
    const int64_t t = blockIdx.x;
    const int h = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    const bf16* x_row = qk + t * qk_stride + h * kIndexHeadDim;
    float x[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bf16 v = x_row[lane * 4 + i];
        rows[h][lane * 4 + i] = v;
        x[i] = __bfloat162float(v);
    }
    __syncwarp();
    gemma_norm_row(x, rows[h], weight, eps, lane);
    rope_row_p3(rows[h], cos_sin, b, positions3, p3_stride, p3_lead + t,
                q_idx + (t * kIndexHeads + h) * kIndexHeadDim, lane);
}

// compress_kernel with group g rotated at its first member's (logical 4g) positions3 column 4g - (pos0 - p3_lead),
// as qsa_index_k_compress_store (loc0 = locs[0], qsa_indexer.cuh:326,378-381). Mean and norm as compress_group.
__global__ void __launch_bounds__(4 * kWarp) compress_mrope_kernel(
    const bf16* __restrict__ k_idx, int64_t k_stride, const bf16* __restrict__ weight, float eps,
    const float* __restrict__ cos_sin, Bounds b, const int64_t* __restrict__ positions3, int64_t p3_stride,
    int64_t p3_lead, const int64_t* __restrict__ slots, int64_t pos0, int64_t g_lo, int64_t groups,
    const bf16* __restrict__ ring,
    bf16* __restrict__ keys) {
    __shared__ bf16 rows[4][kIndexHeadDim];
    const int w = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    const int64_t gi = static_cast<int64_t>(blockIdx.x) * 4 + w;
    if (gi >= groups) return;
    const int64_t g = g_lo + gi;
    bf16* row = rows[w];
    float acc[4];
    for (int r = 0; r < kRatio; ++r) {
        const int64_t p = g * kRatio + r;
        const bf16* src = p < pos0 ? ring + (p % kRatio) * kIndexHeadDim : k_idx + (p - pos0) * k_stride;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float f = __bfloat162float(src[lane * 4 + i]);
            acc[i] = r == 0 ? f : __fadd_rn(acc[i], f);
        }
    }
    const float inv_ratio = 1.0f / static_cast<float>(kRatio);
    float m[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bf16 v = __float2bfloat16_rn(__fmul_rn(acc[i], inv_ratio));
        m[i] = __bfloat162float(v);
        row[lane * 4 + i] = v;
    }
    __syncwarp();
    gemma_norm_row(m, row, weight, eps, lane);
    const int64_t first = g * kRatio;
    const int64_t slot = slots ? slots[first] / kRatio : g;
    rope_row_p3(row, cos_sin, b, positions3, p3_stride, first - (pos0 - p3_lead), keys + slot * kIndexHeadDim, lane);
}

// Raw index keys of the new members of the last incomplete group -> ring rows p % 4.
__global__ void ring_kernel(const bf16* __restrict__ k_idx, int64_t k_stride, int64_t pos0, int64_t first,
                            int64_t end, bf16* __restrict__ ring) {
    for (int64_t p = first; p < end; ++p)
        ring[(p % kRatio) * kIndexHeadDim + threadIdx.x] = k_idx[(p - pos0) * k_stride + threadIdx.x];
}

// qsa_compress_store of one token at position p (kIndexHeadDim threads; `row` shared; p inside the bounds' context):
// the token completes group p / 4 (compress_kernel's warp 0 with pos0 = p, groups 1) or else goes to ring row p % 4
// (ring_kernel).
__device__ __forceinline__ void compress_step_row(const bf16* __restrict__ k_idx, const bf16* __restrict__ weight,
                                                  float eps, const float* __restrict__ cos_sin, const Bounds& b,
                                                  const int64_t* __restrict__ slots, int64_t p, bf16* __restrict__ ring,
                                                  bf16* __restrict__ keys, bf16* row) {
    if ((p + 1) % kRatio == 0) {
        if (threadIdx.x < kWarp) {
            compress_group(k_idx, 0, weight, eps, cos_sin, b, nullptr, slots, p, p / kRatio, ring, keys, row,
                           static_cast<int>(threadIdx.x));
        }
    } else {
        ring[(p % kRatio) * kIndexHeadDim + threadIdx.x] = k_idx[threadIdx.x];
    }
}

// qsa_compress_store of one token at the device position p = *pos (clamped to the context).
__global__ void __launch_bounds__(kIndexHeadDim) compress_step_kernel(const bf16* __restrict__ k_idx,
                                                                      const bf16* __restrict__ weight, float eps,
                                                                      const float* __restrict__ cos_sin, Bounds b,
                                                                      const int64_t* __restrict__ slots,
                                                                      const int64_t* __restrict__ pos,
                                                                      bf16* __restrict__ ring, bf16* __restrict__ keys) {
    __shared__ bf16 row[kIndexHeadDim];
    compress_step_row(k_idx, weight, eps, cos_sin, b, slots, bounded(*pos, 0, b.context, b.fault), ring, keys, row);
}

// The device position pos0 = *pos of a call of `tokens` tokens, clamped to 0 .. context - tokens.
__device__ __forceinline__ int64_t call_start(const int64_t* __restrict__ pos, int64_t tokens, const Bounds& b) {
    return bounded(*pos, 0, b.context - tokens + 1, b.fault);
}

// qsa_compress_store of `tokens` tokens at the device position pos0 = *pos: compress_kernel's warps over the groups
// [pos0 / 4, (pos0 + tokens) / 4) (grid at the most groups of `tokens`, the extra warps exit), then, in a second
// launch (stream order: the compress may still read ring rows it overwrites), ring_kernel's pending members.
__global__ void __launch_bounds__(4 * kWarp) compress_at_kernel(const bf16* __restrict__ k_idx, int64_t k_stride,
                                                                const bf16* __restrict__ weight, float eps,
                                                                const float* __restrict__ cos_sin, Bounds b,
                                                                const int64_t* __restrict__ slots,
                                                                const int64_t* __restrict__ pos, int64_t tokens,
                                                                const bf16* __restrict__ ring, bf16* __restrict__ keys) {
    __shared__ bf16 rows[4][kIndexHeadDim];
    const int64_t pos0 = call_start(pos, tokens, b);
    const int64_t g_lo = pos0 / kRatio;
    const int64_t groups = (pos0 + tokens) / kRatio - g_lo;
    const int w = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    const int64_t gi = static_cast<int64_t>(blockIdx.x) * 4 + w;
    if (gi >= groups) return;
    compress_group(k_idx, k_stride, weight, eps, cos_sin, b, nullptr, slots, pos0, g_lo + gi, ring, keys, rows[w],
                   lane);
}

__global__ void ring_at_kernel(const bf16* __restrict__ k_idx, int64_t k_stride, Bounds b,
                               const int64_t* __restrict__ pos, int64_t tokens, bf16* __restrict__ ring) {
    const int64_t pos0 = call_start(pos, tokens, b);
    const int64_t end = pos0 + tokens;
    const int64_t g_hi = end / kRatio;
    const int64_t first = g_hi * kRatio > pos0 ? g_hi * kRatio : pos0;
    for (int64_t p = first; p < end; ++p)
        ring[(p % kRatio) * kIndexHeadDim + threadIdx.x] = k_idx[(p - pos0) * k_stride + threadIdx.x];
}

// -- block logits ------------------------------------------------------------------------------------------------

// Row `row`'s logical position: positions_logical[row] (device) clamped to 0 .. seq_len - 1 (Bounds' rule; fault:
// the decode step's word, else null), so (p + 1) / kRatio <= floor(seq_len / 4) <= ld and no read passes the logits
// row; or the default seq_len - rows_total + row.
__device__ __forceinline__ int64_t row_position(const int64_t* positions_logical, int64_t row, int64_t rows_total,
                                                int64_t seq_len, uint32_t* fault) {
    return positions_logical ? bounded(positions_logical[row], 0, seq_len, fault) : seq_len - rows_total + row;
}

struct LogitSmem {
    union {
        bf16 q[kLogitVectors][kLdsQ];              // staged index q (fragments are then kept in registers)
        float acc[kLogitThreads / kWarp][16][kLogitKeyTile];  // per-warp accumulator tile
    } u;
    bf16 k[kLogitKeyTile][kLdsQ];
    int64_t max_visible;
};

// Rows row0 .. row0 + tile_rows - 1 of the call (q_idx and logits already offset to row0); logits[r, b] for
// b < ld. Grid (ceil(tile_rows / 32), ceil(ld / 256)). skip_masked: a block whose keys are all past every row's
// visible blocks exits before any store (the decode step's grid at the context's largest ld, qsa_select_step).
__global__ void __launch_bounds__(kLogitThreads) logits_kernel(const bf16* __restrict__ q_idx,
                                                               const bf16* __restrict__ keys,
                                                               const int64_t* __restrict__ slots,
                                                               const int64_t* __restrict__ positions_logical,
                                                               int64_t row0, int64_t tile_rows, int64_t rows_total,
                                                               int64_t seq_len, uint32_t* fault,
                                                               float* __restrict__ logits, int64_t ld,
                                                               bool skip_masked) {
    using namespace nvcuda;
    __shared__ __align__(128) LogitSmem sm;
    const int tid = threadIdx.x;
    const int warp = tid / kWarp;
    const int lane = tid % kWarp;
    const int64_t r0 = static_cast<int64_t>(blockIdx.x) * kLogitRows;  // first row of this block within the tile
    const int64_t b_begin = static_cast<int64_t>(blockIdx.y) * kLogitKeysPerBlock;
    const int64_t b_end = b_begin + kLogitKeysPerBlock < ld ? b_begin + kLogitKeysPerBlock : ld;

    // Visible blocks of each of the 32 rows; the block maximum bounds the keys that are read.
    if (warp == 0) {
        const int64_t r = r0 + lane;
        int64_t nv = 0;
        if (r < tile_rows) nv = (row_position(positions_logical, row0 + r, rows_total, seq_len, fault) + 1) / kRatio;
#pragma unroll
        for (int mask = kWarp / 2; mask >= 1; mask >>= 1) {
            const int64_t o = __shfl_xor_sync(0xffffffffu, nv, mask, kWarp);
            nv = o > nv ? o : nv;
        }
        if (lane == 0) sm.max_visible = nv;
    }
    // Stage q: vector v = row (r0 + v / 4), head v % 4.
    for (int e = tid; e < kLogitVectors * (kIndexHeadDim / 8); e += kLogitThreads) {
        const int v = e / (kIndexHeadDim / 8);
        const int c = (e % (kIndexHeadDim / 8)) * 8;
        const int64_t r = r0 + v / kIndexHeads;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (r < tile_rows)
            val = *reinterpret_cast<const uint4*>(q_idx + (r * kIndexHeads + v % kIndexHeads) * kIndexHeadDim + c);
        uint32_t* dst = reinterpret_cast<uint32_t*>(&sm.u.q[v][c]);
        dst[0] = val.x;
        dst[1] = val.y;
        dst[2] = val.z;
        dst[3] = val.w;
    }
    __syncthreads();
    const int64_t max_visible = sm.max_visible;
    if (skip_masked && b_begin >= max_visible) return;  // uniform per block, before any store
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[kIndexHeadDim / 16];
#pragma unroll
    for (int k = 0; k < kIndexHeadDim / 16; ++k) wmma::load_matrix_sync(a[k], &sm.u.q[warp * 16][k * 16], kLdsQ);
    __syncthreads();  // the q staging area becomes the accumulator area

    // This warp's 16 vectors are rows r0 + 4 warp .. r0 + 4 warp + 3; lane = key within the sub-tile.
    int64_t nv[4];
    bool row_ok[4];
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        const int64_t r = r0 + warp * 4 + rr;
        row_ok[rr] = r < tile_rows;
        nv[rr] = row_ok[rr] ? (row_position(positions_logical, row0 + r, rows_total, seq_len, fault) + 1) / kRatio
                            : 0;
    }
    const float neg_inf = -INFINITY;
    for (int64_t kb = b_begin; kb < b_end; kb += kLogitKeyTile) {
        const int64_t b = kb + lane;
        if (kb >= max_visible) {
            // Every row of the block is causally masked here.
#pragma unroll
            for (int rr = 0; rr < 4; ++rr)
                if (row_ok[rr] && b < b_end) logits[(r0 + warp * 4 + rr) * ld + b] = neg_inf;
            continue;
        }
        // Stage keys kb .. kb + 31 (blocks >= max_visible are not read: zeros).
        for (int e = tid; e < kLogitKeyTile * (kIndexHeadDim / 8); e += kLogitThreads) {
            const int j = e / (kIndexHeadDim / 8);
            const int c = (e % (kIndexHeadDim / 8)) * 8;
            const int64_t bj = kb + j;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (bj < max_visible) {
                const int64_t slot = slots ? slots[bj * kRatio] / kRatio : bj;
                val = *reinterpret_cast<const uint4*>(keys + slot * kIndexHeadDim + c);
            }
            uint32_t* dst = reinterpret_cast<uint32_t*>(&sm.k[j][c]);
            dst[0] = val.x;
            dst[1] = val.y;
            dst[2] = val.z;
            dst[3] = val.w;
        }
        __syncthreads();
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kLogitKeyTile / 16];
#pragma unroll
        for (int n = 0; n < kLogitKeyTile / 16; ++n) {
            wmma::fill_fragment(acc[n], 0.0f);
#pragma unroll
            for (int k = 0; k < kIndexHeadDim / 16; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bfrag;
                wmma::load_matrix_sync(bfrag, &sm.k[n * 16][k * 16], kLdsQ);
                wmma::mma_sync(acc[n], a[k], bfrag, acc[n]);
            }
            wmma::store_matrix_sync(&sm.u.acc[warp][0][n * 16], acc[n], kLogitKeyTile, wmma::mem_row_major);
        }
        __syncwarp();
        if (b < b_end) {
#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                if (!row_ok[rr]) continue;
                float s = fmaxf(sm.u.acc[warp][rr * 4][lane], 0.0f);
#pragma unroll
                for (int h = 1; h < kIndexHeads; ++h) s = __fadd_rn(s, fmaxf(sm.u.acc[warp][rr * 4 + h][lane], 0.0f));
                logits[(r0 + warp * 4 + rr) * ld + b] = b < nv[rr] ? __fdiv_rn(s, kScoreDivisor) : neg_inf;
            }
        }
        __syncthreads();  // sm.k and the accumulator tile are reused
    }
}

struct LogitSmallSmem {
    bf16 q[kLogitSmallVectors][kLdsQ];
    union {
        bf16 k[kLogitKeyTile][kLdsQ];                  // this warp's staged keys
        float acc[kLogitSmallVectors][kLogitKeyTile];  // then its accumulator tile
    } w[kLogitSmallWarps];
};

// logits_kernel's values for the calls of up to kSplitMaxRows rows (track LongK9, long-k9 design note LK9-02).
// logits_kernel gives such a call one busy warp per 4 rows, which scores its block's 8 sub-tiles of 32 keys one after
// another; here a block holds 4 rows and 128 keys and each warp scores one sub-tile, so the sub-tiles run side by
// side. Per element the arithmetic is logits_kernel's: the 16 vectors (row r0 + rr, head h at fragment row 4 rr + h)
// and the keys (key b at fragment column b % 16) sit at the same fragment positions beside the same neighbours, the
// same 8 MMAs per 16 keys run from a zero accumulator, then the same ReLU sum over the heads in order and the same
// rounded division. A sub-tile at or past every visible block of the 4 rows stores -inf unread; in a scored sub-tile
// keys past them stage zeros and store -inf (logits_kernel's 32-row block maximum reads some of those keys, whose
// logits are -inf for these rows either way). skip_masked: a block whose keys are all past its rows' visible blocks
// exits before any store. Grid (ceil(ld / 128), ceil(tile_rows / 4)).
__global__ void __launch_bounds__(kLogitSmallThreads) logits_small_kernel(const bf16* __restrict__ q_idx,
                                                                          const bf16* __restrict__ keys,
                                                                          const int64_t* __restrict__ slots,
                                                                          const int64_t* __restrict__ positions_logical,
                                                                          int64_t row0, int64_t tile_rows,
                                                                          int64_t rows_total, int64_t seq_len,
                                                                          uint32_t* fault, float* __restrict__ logits,
                                                                          int64_t ld, bool skip_masked) {
    using namespace nvcuda;
    __shared__ __align__(128) LogitSmallSmem sm;
    const int tid = threadIdx.x;
    const int warp = tid / kWarp;
    const int lane = tid % kWarp;
    const int64_t r0 = static_cast<int64_t>(blockIdx.y) * kLogitSmallRows;  // first row of this block in the tile
    const int64_t b_begin = static_cast<int64_t>(blockIdx.x) * kLogitSmallKeysPerBlock;
    int64_t nv[kLogitSmallRows];
    bool row_ok[kLogitSmallRows];
    int64_t max_visible = 0;
#pragma unroll
    for (int rr = 0; rr < kLogitSmallRows; ++rr) {
        const int64_t r = r0 + rr;
        row_ok[rr] = r < tile_rows;
        nv[rr] = row_ok[rr] ? (row_position(positions_logical, row0 + r, rows_total, seq_len, fault) + 1) / kRatio
                            : 0;
        max_visible = nv[rr] > max_visible ? nv[rr] : max_visible;
    }
    if (skip_masked && b_begin >= max_visible) return;  // uniform per block, before any store
    // Stage q: vector v = row (r0 + v / 4), head v % 4.
    for (int e = tid; e < kLogitSmallVectors * (kIndexHeadDim / 8); e += kLogitSmallThreads) {
        const int v = e / (kIndexHeadDim / 8);
        const int c = (e % (kIndexHeadDim / 8)) * 8;
        const int64_t r = r0 + v / kIndexHeads;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (r < tile_rows)
            val = *reinterpret_cast<const uint4*>(q_idx + (r * kIndexHeads + v % kIndexHeads) * kIndexHeadDim + c);
        *reinterpret_cast<uint4*>(&sm.q[v][c]) = val;
    }
    __syncthreads();  // the last block barrier: from here each warp works alone
    const int64_t kb = b_begin + static_cast<int64_t>(warp) * kLogitKeyTile;
    if (kb >= ld) return;
    const int64_t b = kb + lane;
    if (kb >= max_visible) {
        // Every row of the block is causally masked here.
#pragma unroll
        for (int rr = 0; rr < kLogitSmallRows; ++rr)
            if (row_ok[rr] && b < ld) logits[(r0 + rr) * ld + b] = -INFINITY;
        return;
    }
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[kIndexHeadDim / 16];
#pragma unroll
    for (int k = 0; k < kIndexHeadDim / 16; ++k) wmma::load_matrix_sync(a[k], &sm.q[0][k * 16], kLdsQ);
    // Stage keys kb .. kb + 31 (blocks >= max_visible are not read: zeros): lane j looks up key j's slot, then each
    // step moves 16 bytes of each of two keys per lane.
    int64_t slot = 0;
    if (b < max_visible) slot = slots ? slots[b * kRatio] / kRatio : b;
    auto& w = sm.w[warp];
#pragma unroll
    for (int i = 0; i < kLogitKeyTile * (kIndexHeadDim / 8) / kWarp; ++i) {
        const int e = i * kWarp + lane;
        const int j = e / (kIndexHeadDim / 8);
        const int c = (e % (kIndexHeadDim / 8)) * 8;
        const int64_t sj = __shfl_sync(0xffffffffu, slot, j, kWarp);
        uint4 val = make_uint4(0, 0, 0, 0);
        if (kb + j < max_visible) val = *reinterpret_cast<const uint4*>(keys + sj * kIndexHeadDim + c);
        *reinterpret_cast<uint4*>(&w.k[j][c]) = val;
    }
    __syncwarp();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kLogitKeyTile / 16];
#pragma unroll
    for (int n = 0; n < kLogitKeyTile / 16; ++n) {
        wmma::fill_fragment(acc[n], 0.0f);
#pragma unroll
        for (int k = 0; k < kIndexHeadDim / 16; ++k) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bfrag;
            wmma::load_matrix_sync(bfrag, &w.k[n * 16][k * 16], kLdsQ);
            wmma::mma_sync(acc[n], a[k], bfrag, acc[n]);
        }
    }
    __syncwarp();  // every lane's key fragments are loaded before the accumulator tile overwrites them
#pragma unroll
    for (int n = 0; n < kLogitKeyTile / 16; ++n)
        wmma::store_matrix_sync(&w.acc[0][n * 16], acc[n], kLogitKeyTile, wmma::mem_row_major);
    __syncwarp();
    if (b < ld) {
#pragma unroll
        for (int rr = 0; rr < kLogitSmallRows; ++rr) {
            if (!row_ok[rr]) continue;
            float s = fmaxf(w.acc[rr * 4][lane], 0.0f);
#pragma unroll
            for (int h = 1; h < kIndexHeads; ++h) s = __fadd_rn(s, fmaxf(w.acc[rr * 4 + h][lane], 0.0f));
            logits[(r0 + rr) * ld + b] = b < nv[rr] ? __fdiv_rn(s, kScoreDivisor) : -INFINITY;
        }
    }
}

// 16 bytes global -> shared by cp.async, zero-filled when !valid (src must still be a mapped address).
__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
// Every cp.async group of this thread is complete.
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

constexpr int kLogitWarps = kLogitThreads / kWarp;
constexpr int kLdsAcc = kLogitKeyTile + 8;  // F32 accumulator rows (160 B): a fragment store's rows hit distinct banks
static_assert(kLogitThreads == kLogitKeysPerBlock && kLogitWarps * 16 == kLogitVectors,
              "logits_prefill_kernel: one key offset per thread, one 16-vector fragment per warp");

struct LogitPrefillSmem {
    union {
        bf16 q[kLogitVectors][kIndexHeadDim];  // staged index q (read once into fragments, so unpadded)
        struct {
            bf16 k[2][kLogitKeyTile][kLdsQ];                // key sub-tiles, double-buffered
            float acc[kLogitWarps][16][kLdsAcc];            // per-warp accumulator tile
        } t;
    } u;
    int64_t key_off[kLogitKeysPerBlock];  // element offset (slot * 128) of the block's keys below max_visible
};

// logits_kernel's values for the prefill calls (more than kSplitMaxRows rows; qsa_logits impl 3). logits_kernel
// stages each 32-key sub-tile after a dependent slots -> keys trip to memory between two block barriers, so its warps
// wait on memory between MMA bursts; here the slots of the block's 256 keys are read once up front, sub-tile t + 1
// streams in by cp.async while sub-tile t is scored (one barrier per sub-tile), and the accumulator rows are padded.
// Blocks, grid and per element the arithmetic are logits_kernel's: the same 32 rows (so the same max_visible and the
// same zero-staged keys at or past it), vector v = (row r0 + v / 4, head v % 4) at row v % 16 of warp v / 16's A
// fragments, key b at column (b - kb) % 16 of sub-tile kb's B fragment (b - kb) / 16, the same 8 wmma MMAs per 16
// keys in k order from a zero accumulator, then the same ReLU sum over the heads in order and the same rounded
// division; sub-tiles at or past max_visible store -inf unread. Only shared-memory strides and copy timing differ.
__global__ void __launch_bounds__(kLogitThreads) logits_prefill_kernel(const bf16* __restrict__ q_idx,
                                                                       const bf16* __restrict__ keys,
                                                                       const int64_t* __restrict__ slots,
                                                                       const int64_t* __restrict__ positions_logical,
                                                                       int64_t row0, int64_t tile_rows,
                                                                       int64_t rows_total, int64_t seq_len,
                                                                       uint32_t* fault, float* __restrict__ logits,
                                                                       int64_t ld, bool skip_masked) {
    using namespace nvcuda;
    __shared__ __align__(128) LogitPrefillSmem sm;
    const int tid = threadIdx.x;
    const int warp = tid / kWarp;
    const int lane = tid % kWarp;
    const int64_t r0 = static_cast<int64_t>(blockIdx.x) * kLogitRows;  // first row of this block within the tile
    const int64_t b_begin = static_cast<int64_t>(blockIdx.y) * kLogitKeysPerBlock;
    const int64_t b_end = b_begin + kLogitKeysPerBlock < ld ? b_begin + kLogitKeysPerBlock : ld;

    // Every warp takes the 32 rows' visible blocks (lane l: row r0 + l) and their maximum itself; its own 4 rows'
    // counts are shuffled out of them.
    int64_t nv_lane = 0;
    if (r0 + lane < tile_rows)
        nv_lane = (row_position(positions_logical, row0 + r0 + lane, rows_total, seq_len, fault) + 1) / kRatio;
    int64_t max_visible = nv_lane;
#pragma unroll
    for (int mask = kWarp / 2; mask >= 1; mask >>= 1) {
        const int64_t o = __shfl_xor_sync(0xffffffffu, max_visible, mask, kWarp);
        max_visible = o > max_visible ? o : max_visible;
    }
    if (skip_masked && b_begin >= max_visible) return;  // uniform per block, before any store
    int64_t nv[4];
    bool row_ok[4];
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        row_ok[rr] = r0 + warp * 4 + rr < tile_rows;
        nv[rr] = __shfl_sync(0xffffffffu, nv_lane, warp * 4 + rr, kWarp);
    }
    // Key offsets (slots read for blocks below max_visible only, as logits_kernel's staging) and q (vector v = row
    // (r0 + v / 4), head v % 4; zeros past the tile).
    {
        const int64_t bj = b_begin + tid;
        sm.key_off[tid] = bj < max_visible ? (slots ? slots[bj * kRatio] / kRatio : bj) * kIndexHeadDim : 0;
    }
    for (int e = tid; e < kLogitVectors * (kIndexHeadDim / 8); e += kLogitThreads) {
        const int v = e / (kIndexHeadDim / 8);
        const int c = (e % (kIndexHeadDim / 8)) * 8;
        const int64_t r = r0 + v / kIndexHeads;
        const bool ok = r < tile_rows;
        cp16(&sm.u.q[v][c], ok ? q_idx + (r * kIndexHeads + v % kIndexHeads) * kIndexHeadDim + c : q_idx, ok);
    }
    cp_commit();
    cp_wait0();
    __syncthreads();  // q, key_off
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[kIndexHeadDim / 16];
#pragma unroll
    for (int k = 0; k < kIndexHeadDim / 16; ++k)
        wmma::load_matrix_sync(a[k], &sm.u.q[warp * 16][k * 16], kIndexHeadDim);
    __syncthreads();  // the q staging area becomes the key tiles and the accumulators

    // Sub-tiles kb = b_begin + 32 t below max_visible are scored (a prefix), the rest store -inf.
    const int tiles = static_cast<int>((b_end - b_begin + kLogitKeyTile - 1) / kLogitKeyTile);
    int scored = 0;
    if (max_visible > b_begin) {
        const int64_t vis = max_visible - b_begin;
        scored = vis >= b_end - b_begin ? tiles : static_cast<int>((vis + kLogitKeyTile - 1) / kLogitKeyTile);
    }
    // Stage keys kb .. kb + 31 into buffer `buf` (blocks >= max_visible are not read: zeros).
    const auto stage = [&](int buf, int64_t kb) {
#pragma unroll
        for (int e = tid; e < kLogitKeyTile * (kIndexHeadDim / 8); e += kLogitThreads) {
            const int j = e / (kIndexHeadDim / 8);
            const int c = (e % (kIndexHeadDim / 8)) * 8;
            const int64_t bj = kb + j;
            const bool ok = bj < max_visible;
            cp16(&sm.u.t.k[buf][j][c], ok ? keys + sm.key_off[bj - b_begin] + c : keys, ok);
        }
        cp_commit();
    };
    if (scored > 0) stage(0, b_begin);
    const float neg_inf = -INFINITY;
    for (int t = 0; t < scored; ++t) {
        const int64_t kb = b_begin + static_cast<int64_t>(t) * kLogitKeyTile;
        const int buf = t & 1;
        cp_wait0();
        __syncthreads();  // sub-tile t landed for every thread; every warp is done with buffer t + 1 and its tile
        if (t + 1 < scored) stage(buf ^ 1, kb + kLogitKeyTile);
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kLogitKeyTile / 16];
#pragma unroll
        for (int n = 0; n < kLogitKeyTile / 16; ++n) {
            wmma::fill_fragment(acc[n], 0.0f);
#pragma unroll
            for (int k = 0; k < kIndexHeadDim / 16; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bfrag;
                wmma::load_matrix_sync(bfrag, &sm.u.t.k[buf][n * 16][k * 16], kLdsQ);
                wmma::mma_sync(acc[n], a[k], bfrag, acc[n]);
            }
            wmma::store_matrix_sync(&sm.u.t.acc[warp][0][n * 16], acc[n], kLdsAcc, wmma::mem_row_major);
        }
        __syncwarp();
        const int64_t b = kb + lane;
        if (b < b_end) {
#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                if (!row_ok[rr]) continue;
                float s = fmaxf(sm.u.t.acc[warp][rr * 4][lane], 0.0f);
#pragma unroll
                for (int h = 1; h < kIndexHeads; ++h) s = __fadd_rn(s, fmaxf(sm.u.t.acc[warp][rr * 4 + h][lane], 0.0f));
                logits[(r0 + warp * 4 + rr) * ld + b] = b < nv[rr] ? __fdiv_rn(s, kScoreDivisor) : neg_inf;
            }
        }
    }
    // Every row of the block is causally masked in the remaining sub-tiles.
    for (int64_t kb = b_begin + static_cast<int64_t>(scored) * kLogitKeyTile; kb < b_end; kb += kLogitKeyTile) {
        const int64_t b = kb + lane;
#pragma unroll
        for (int rr = 0; rr < 4; ++rr)
            if (row_ok[rr] && b < b_end) logits[(r0 + warp * 4 + rr) * ld + b] = neg_inf;
    }
}

// The block logits kernels; every one writes the same bytes.
enum class LogitsKernel {
    block,    // logits_kernel (the reference of the others)
    small,    // logits_small_kernel: the calls of up to kSplitMaxRows rows
    prefill,  // logits_prefill_kernel: the larger calls
};

cudaError_t launch_logits(const bf16* q_idx, const bf16* keys, const int64_t* slots, const int64_t* positions_logical,
                          int64_t row0, int64_t tile_rows, int64_t rows_total, int64_t seq_len, uint32_t* fault,
                          float* logits, int64_t ld, bool skip_masked, LogitsKernel kernel, cudaStream_t stream) {
    if (tile_rows == 0 || ld == 0) return cudaSuccess;
    const bf16* q = q_idx + row0 * kIndexHeads * kIndexHeadDim;
    if (kernel == LogitsKernel::small) {
        const dim3 grid(static_cast<unsigned>((ld + kLogitSmallKeysPerBlock - 1) / kLogitSmallKeysPerBlock),
                        static_cast<unsigned>((tile_rows + kLogitSmallRows - 1) / kLogitSmallRows));
        logits_small_kernel<<<grid, kLogitSmallThreads, 0, stream>>>(q, keys, slots, positions_logical, row0,
                                                                     tile_rows, rows_total, seq_len, fault, logits, ld,
                                                                     skip_masked);
        return cudaGetLastError();
    }
    const dim3 grid(static_cast<unsigned>((tile_rows + kLogitRows - 1) / kLogitRows),
                    static_cast<unsigned>((ld + kLogitKeysPerBlock - 1) / kLogitKeysPerBlock));
    if (kernel == LogitsKernel::prefill) {
        logits_prefill_kernel<<<grid, kLogitThreads, 0, stream>>>(q, keys, slots, positions_logical, row0, tile_rows,
                                                                  rows_total, seq_len, fault, logits, ld, skip_masked);
    } else {
        logits_kernel<<<grid, kLogitThreads, 0, stream>>>(q, keys, slots, positions_logical, row0, tile_rows,
                                                          rows_total, seq_len, fault, logits, ld, skip_masked);
    }
    return cudaGetLastError();
}

// -- top-512 + expansion -------------------------------------------------------------------------------------------

// Block-wide exclusive prefix sum of `flag` (topk_kernel: 0/1; topk_cluster_kernel: packed counts that never carry)
// over kTopkThreads threads; *total = the block sum.
__device__ __forceinline__ int block_exclusive_scan(int flag, int* warp_sums, int* total) {
    const int tid = threadIdx.x;
    const int lane = tid % kWarp;
    const int warp = tid / kWarp;
    int incl = flag;
#pragma unroll
    for (int o = 1; o < kWarp; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o, kWarp);
        if (lane >= o) incl += v;
    }
    if (lane == kWarp - 1) warp_sums[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        int s = lane < kTopkWarps ? warp_sums[lane] : 0;
#pragma unroll
        for (int o = 1; o < kWarp; o <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, s, o, kWarp);
            if (lane >= o) s += v;
        }
        if (lane < kTopkWarps) warp_sums[lane] = s;  // inclusive warp prefix
    }
    __syncthreads();
    const int base = warp == 0 ? 0 : warp_sums[warp - 1];
    *total = warp_sums[kTopkWarps - 1];
    const int excl = base + incl - flag;
    __syncthreads();  // warp_sums is reused by the next scan
    return excl;
}

// Row r of the tile: logits row (ld floats), output row out (kRowWidth int32).
__global__ void __launch_bounds__(kTopkThreads) topk_kernel(const float* __restrict__ logits, int64_t ld,
                                                            const int64_t* __restrict__ positions_logical,
                                                            int64_t row0, int64_t rows_total, int64_t seq_len,
                                                            uint32_t* fault, int32_t* __restrict__ out) {
    __shared__ int hist[256];
    __shared__ int warp_sums[kTopkWarps];
    __shared__ int s_bin;
    __shared__ int s_k;
    const int tid = threadIdx.x;
    const int64_t r = blockIdx.x;
    const int64_t p = row_position(positions_logical, row0 + r, rows_total, seq_len, fault);
    const int64_t nv = (p + 1) / kRatio;
    const float* x = logits + r * ld;
    int32_t* o = out + (row0 + r) * kRowWidth;
    int64_t selected;
    if (nv <= kBlockTopk) {
        for (int64_t e = tid; e < nv * kRatio; e += kTopkThreads) o[e] = static_cast<int32_t>(e);
        selected = nv;
    } else {
        uint32_t prefix = 0;
        uint32_t mask = 0;
        int k = kBlockTopk;
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            if (tid < 256) hist[tid] = 0;
            __syncthreads();
            for (int64_t i = tid; i < nv; i += kTopkThreads) {
                const uint32_t key = order_key(x[i]);
                if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 0xffu], 1);
            }
            __syncthreads();
            if (tid == 0) {
                int cum = 0;
                int bin = 0;
                for (int bb = 255; bb >= 0; --bb) {
                    if (cum + hist[bb] >= k) {
                        bin = bb;
                        break;
                    }
                    cum += hist[bb];
                }
                s_bin = bin;
                s_k = k - cum;
            }
            __syncthreads();
            prefix |= static_cast<uint32_t>(s_bin) << shift;
            mask |= 0xffu << shift;
            k = s_k;
            __syncthreads();
        }
        // Keys > prefix are all taken; of the keys == prefix, the first k by block index.
        const int need = k;
        int eq_base = 0;
        int sel_base = 0;
        for (int64_t base = 0; base < nv; base += kTopkThreads) {
            const int64_t i = base + tid;
            const uint32_t key = i < nv ? order_key(x[i]) : 0u;
            const bool gt = i < nv && key > prefix;
            const bool eq = i < nv && key == prefix;
            int eq_total;
            const int eq_rank = block_exclusive_scan(eq ? 1 : 0, warp_sums, &eq_total);
            const bool take = gt || (eq && eq_base + eq_rank < need);
            int sel_total;
            const int sel_rank = block_exclusive_scan(take ? 1 : 0, warp_sums, &sel_total);
            if (take) {
                const int64_t pos = static_cast<int64_t>(sel_base + sel_rank) * kRatio;
#pragma unroll
                for (int q = 0; q < kRatio; ++q) o[pos + q] = static_cast<int32_t>(i * kRatio + q);
            }
            eq_base += eq_total;
            sel_base += sel_total;
        }
        selected = kBlockTopk;
    }
    // Tail floor((p + 1) / 4) * 4 .. p, then -1 padding.
    const int64_t tail = p + 1 - nv * kRatio;
    for (int64_t e = selected * kRatio + tid; e < kRowWidth; e += kTopkThreads) {
        const int64_t j = e - selected * kRatio;
        o[e] = j < tail ? static_cast<int32_t>(nv * kRatio + j) : -1;
    }
}

// topk_kernel's selection on a cluster of kTkCluster blocks per row (track LongK9, long-k9 design note LK9-01):
// thread t of block rank c holds the contiguous run of S = ceil(nv / (kTkCluster * kTopkThreads)) visible blocks from
// (c * kTopkThreads + t) * S in registers (S <= kTkPerThread: nv <= ld <= kTkMaxBlocks, the caller's check). The
// same 4 radix passes over the same key (order_key): per pass every block counts its keys into its own histogram
// (warp-aggregated shared atomics), the cluster barrier publishes them, and every block sums the kTkCluster
// histograms (DSMEM, integer counts in rank order) and picks the same bin as one parallel suffix scan. Then one block
// scan: a run takes its keys > prefix and those of its keys == prefix whose rank among the row's keys == prefix (the
// lower blocks' totals plus the scan) is below `need`, and writes them in block order after the taken keys of the
// lower runs. Bytes equal topk_kernel's for every row: all keys > prefix, the first `need` keys == prefix by block
// index, blocks ascending, then the tail and -1 padding. Grid (kTkCluster, rows).
__global__ void __cluster_dims__(kTkCluster, 1, 1) __launch_bounds__(kTopkThreads)
    topk_cluster_kernel(const float* __restrict__ logits, int64_t ld, const int64_t* __restrict__ positions_logical,
                        int64_t row0, int64_t rows_total, int64_t seq_len, uint32_t* fault,
                        int32_t* __restrict__ out) {
    __shared__ int hist[4][256];  // one histogram per pass: a block never rewrites what another may still read
    __shared__ int warp_sums[kTopkWarps];
    __shared__ int s_bin;
    __shared__ int s_k;
    __shared__ int s_total[2];   // this block's keys > prefix, == prefix
    __shared__ int s_before[2];  // the lower blocks' totals
    cg::cluster_group cl = cg::this_cluster();
    const int rank = static_cast<int>(cl.block_rank());
    const int tid = threadIdx.x;
    const int lane = tid % kWarp;
    const int64_t r = blockIdx.y;
    const int64_t p = row_position(positions_logical, row0 + r, rows_total, seq_len, fault);
    const int64_t nv = (p + 1) / kRatio;
    int32_t* o = out + (row0 + r) * kRowWidth;
    if (nv <= kBlockTopk) {  // every visible block and the tail (positions 0 .. p), then -1; uniform per cluster
        for (int64_t e = static_cast<int64_t>(rank) * kTopkThreads + tid; e < kRowWidth;
             e += int64_t{kTkCluster} * kTopkThreads)
            o[e] = e <= p ? static_cast<int32_t>(e) : -1;
        return;
    }
    const int64_t span = (nv + int64_t{kTkCluster} * kTopkThreads - 1) / (int64_t{kTkCluster} * kTopkThreads);
    const int64_t first = (static_cast<int64_t>(rank) * kTopkThreads + tid) * span;
    const int count = nv - first <= 0 ? 0 : static_cast<int>(nv - first < span ? nv - first : span);
    const float* x = logits + r * ld + first;
    uint32_t key[kTkPerThread];
#pragma unroll
    for (int j = 0; j < kTkPerThread; ++j) key[j] = j < count ? order_key(x[j]) : 0u;
    for (int e = tid; e < 4 * 256; e += kTopkThreads) hist[e / 256][e % 256] = 0;
    __syncthreads();
    uint32_t prefix = 0;
    uint32_t mask = 0;
    int k = kBlockTopk;
#pragma unroll 1
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
#pragma unroll
        for (int j = 0; j < kTkPerThread; ++j) {
            if (j >= span) break;  // uniform per cluster
            const bool part = j < count && (key[j] & mask) == prefix;
            if (__ballot_sync(0xffffffffu, part) == 0u) continue;
            const uint32_t d = part ? (key[j] >> shift) & 0xffu : 0x100u;
            const uint32_t peers = __match_any_sync(0xffffffffu, d);
            if (part && lane == __ffs(peers) - 1) atomicAdd(&hist[pass][d], __popc(peers));
        }
        cl.sync();  // every block's histogram of this pass is complete and visible to the cluster
        // Thread t < 256 owns bin 255 - t: the inclusive scan over t counts the row's keys in bins >= 255 - t.
        int c = 0;
        int incl = 0;
        if (tid < 256) {
#pragma unroll
            for (int q = 0; q < kTkCluster; ++q) c += cl.map_shared_rank(&hist[pass][0], q)[255 - tid];
            incl = c;
#pragma unroll
            for (int off = 1; off < kWarp; off <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, incl, off, kWarp);
                if (lane >= off) incl += v;
            }
            if (lane == kWarp - 1) warp_sums[tid / kWarp] = incl;
        }
        __syncthreads();
        if (tid < 256) {
            int hi = incl;
            for (int w = 0; w < tid / kWarp; ++w) hi += warp_sums[w];
            const int lo = hi - c;  // keys in bins above this one
            if (lo < k && hi >= k) {
                s_bin = 255 - tid;
                s_k = k - lo;
            }
        }
        __syncthreads();
        prefix |= static_cast<uint32_t>(s_bin) << shift;
        mask |= 0xffu << shift;
        k = s_k;
    }
    // Keys > prefix are all taken; of the keys == prefix, the first `need` by block index.
    const int need = k;
    int gt = 0;
    int eq = 0;
#pragma unroll
    for (int j = 0; j < kTkPerThread; ++j) {
        if (j < count) {
            gt += key[j] > prefix ? 1 : 0;
            eq += key[j] == prefix ? 1 : 0;
        }
    }
    // Packed (eq << 16 | gt): a block holds < 512 keys > prefix and at most 8,192 keys, so no field carries.
    int total;
    const int before = block_exclusive_scan(gt | (eq << 16), warp_sums, &total);
    if (tid == 0) {
        s_total[0] = total & 0xffff;
        s_total[1] = total >> 16;
    }
    cl.sync();  // every block's totals are published
    if (tid < 2) {
        int v = 0;
        for (int q = 0; q < rank; ++q) v += cl.map_shared_rank(&s_total[0], q)[tid];
        s_before[tid] = v;
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");  // this block's remote reads are done
    int eq_rank = s_before[1] + (before >> 16);
    int64_t at = s_before[0] + (before & 0xffff) + (eq_rank < need ? eq_rank : need);
#pragma unroll
    for (int j = 0; j < kTkPerThread; ++j) {
        if (j < count) {
            bool take = key[j] > prefix;
            if (key[j] == prefix) take = eq_rank++ < need;
            if (take) {
                const int64_t i = first + j;
#pragma unroll
                for (int q = 0; q < kRatio; ++q) o[at * kRatio + q] = static_cast<int32_t>(i * kRatio + q);
                ++at;
            }
        }
    }
    if (rank == 0) {  // tail floor((p + 1) / 4) * 4 .. p after the 512 blocks, then -1 padding
        const int64_t tail = p + 1 - nv * kRatio;
        for (int64_t e = kTokenTopk + tid; e < kRowWidth; e += kTopkThreads) {
            const int64_t j = e - kTokenTopk;
            o[e] = j < tail ? static_cast<int32_t>(nv * kRatio + j) : -1;
        }
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");  // no block exits while another reads it
}

// topk_kernel's selection for the prefill calls (more than kSplitMaxRows rows, ld <= kPer * kTopkThreads).
// topk_kernel reads its row from memory five times (4 radix passes and the placing scan), counts every key with its
// own shared atomic (pass 0's keys mostly share one or two bins), picks each bin on one thread and runs two block
// scans per 512 keys; here the row is read once, coalesced, into registers (thread t holds blocks j * kTopkThreads +
// t, j < S = ceil(nv / kTopkThreads) <= kPer), every pass counts warp-aggregated into its own histogram and picks the
// bin by topk_cluster_kernel's parallel suffix scan, and one scan over the (round, warp) counts places every taken
// block. Bytes equal topk_kernel's for every row: the same 4 passes over the same key (order_key) give the same
// integer histograms, hence the same bin (the highest one whose suffix count reaches k), prefix and need; a block is
// taken iff its key > prefix, or == prefix with fewer than `need` keys == prefix at lower block indices (the tie
// rule: equal keys go lowest block index first), and it lands at (lower blocks > prefix) + min(lower blocks ==
// prefix, need), topk_kernel's count of taken lower blocks (blocks ascending); then the same tail and -1 padding.
// Grid (rows).
template <int kPer>
__global__ void __launch_bounds__(kTopkThreads) topk_prefill_kernel(const float* __restrict__ logits, int64_t ld,
                                                                    const int64_t* __restrict__ positions_logical,
                                                                    int64_t row0, int64_t rows_total, int64_t seq_len,
                                                                    uint32_t* fault, int32_t* __restrict__ out) {
    static_assert(kPer <= kTpMaxPer && kPer * kTopkWarps <= 2 * kTopkThreads, "two (round, warp) counts per thread");
    __shared__ int hist[4][256];  // one histogram per pass: zeroed once
    __shared__ int warp_sums[kTopkWarps];
    __shared__ int s_bin;
    __shared__ int s_k;
    // Round j, warp w: packed (keys > prefix) << 16 | (keys == prefix), then the exclusive prefix in block order.
    // The > field stays below 512 and the == field at most nv <= 32,768, so no field carries.
    __shared__ int run[kPer * kTopkWarps];
    const int tid = threadIdx.x;
    const int lane = tid % kWarp;
    const int warp = tid / kWarp;
    const int64_t r = blockIdx.x;
    const int64_t p = row_position(positions_logical, row0 + r, rows_total, seq_len, fault);
    const int64_t nv = (p + 1) / kRatio;
    const float* x = logits + r * ld;
    int32_t* o = out + (row0 + r) * kRowWidth;
    int64_t selected;
    if (nv <= kBlockTopk) {
        for (int64_t e = tid; e < nv * kRatio; e += kTopkThreads) o[e] = static_cast<int32_t>(e);
        selected = nv;
    } else {
        const int n = static_cast<int>(nv);  // nv <= ld <= kPer * kTopkThreads (the launcher's choice)
        const int rounds = (n + kTopkThreads - 1) / kTopkThreads;
        uint32_t key[kPer];
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            const int i = j * kTopkThreads + tid;
            key[j] = i < n ? order_key(x[i]) : 0u;
        }
        for (int e = tid; e < 4 * 256; e += kTopkThreads) hist[e / 256][e % 256] = 0;
        __syncthreads();
        uint32_t prefix = 0;
        uint32_t mask = 0;
        int k = kBlockTopk;
#pragma unroll 1
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
#pragma unroll
            for (int j = 0; j < kPer; ++j) {
                if (j >= rounds) break;  // uniform per block
                const bool part = j * kTopkThreads + tid < n && (key[j] & mask) == prefix;
                if (__ballot_sync(0xffffffffu, part) == 0u) continue;
                const uint32_t d = part ? (key[j] >> shift) & 0xffu : 0x100u;
                const uint32_t peers = __match_any_sync(0xffffffffu, d);
                if (part && lane == __ffs(peers) - 1) atomicAdd(&hist[pass][d], __popc(peers));
            }
            __syncthreads();  // this pass's histogram is complete
            // Thread t < 256 owns bin 255 - t: the inclusive scan over t counts the row's keys in bins >= 255 - t.
            int c = 0;
            int incl = 0;
            if (tid < 256) {
                c = hist[pass][255 - tid];
                incl = c;
#pragma unroll
                for (int off = 1; off < kWarp; off <<= 1) {
                    const int v = __shfl_up_sync(0xffffffffu, incl, off, kWarp);
                    if (lane >= off) incl += v;
                }
                if (lane == kWarp - 1) warp_sums[tid / kWarp] = incl;
            }
            __syncthreads();
            if (tid < 256) {
                int hi = incl;
                for (int w = 0; w < tid / kWarp; ++w) hi += warp_sums[w];
                const int lo = hi - c;  // keys in bins above this one
                if (lo < k && hi >= k) {
                    s_bin = 255 - tid;
                    s_k = k - lo;
                }
            }
            __syncthreads();
            prefix |= static_cast<uint32_t>(s_bin) << shift;
            mask |= 0xffu << shift;
            k = s_k;
        }
        // Keys > prefix are all taken; of the keys == prefix, the first `need` by block index.
        const int need = k;
        const uint32_t below = (1u << lane) - 1u;  // the lanes of lower block indices in a round
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            if (j >= rounds) break;
            const bool valid = j * kTopkThreads + tid < n;
            const uint32_t gt = __ballot_sync(0xffffffffu, valid && key[j] > prefix);
            const uint32_t eq = __ballot_sync(0xffffffffu, valid && key[j] == prefix);
            if (lane == 0) run[j * kTopkWarps + warp] = (__popc(gt) << 16) | __popc(eq);
        }
        __syncthreads();
        {  // exclusive prefix of the rounds * kTopkWarps counts (round-major = block order): entries 2t, 2t + 1
            const int entries = rounds * kTopkWarps;
            const int e = 2 * tid;
            const int c0 = e < entries ? run[e] : 0;
            const int c1 = e + 1 < entries ? run[e + 1] : 0;
            int total;
            const int before = block_exclusive_scan(c0 + c1, warp_sums, &total);  // its barriers order reads/writes
            if (e < entries) run[e] = before;
            if (e + 1 < entries) run[e + 1] = before + c0;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            if (j >= rounds) break;
            const int i = j * kTopkThreads + tid;
            const bool valid = i < n;
            const bool above = valid && key[j] > prefix;
            const bool tie = valid && key[j] == prefix;
            const uint32_t gt = __ballot_sync(0xffffffffu, above);
            const uint32_t eq = __ballot_sync(0xffffffffu, tie);
            const int base = run[j * kTopkWarps + warp];
            const int eq_before = (base & 0xffff) + __popc(eq & below);
            if (above || (tie && eq_before < need)) {
                const int64_t at = (base >> 16) + __popc(gt & below) + (eq_before < need ? eq_before : need);
#pragma unroll
                for (int q = 0; q < kRatio; ++q) o[at * kRatio + q] = static_cast<int32_t>(i * kRatio + q);
            }
        }
        selected = kBlockTopk;
    }
    // Tail floor((p + 1) / 4) * 4 .. p, then -1 padding.
    const int64_t tail = p + 1 - nv * kRatio;
    for (int64_t e = selected * kRatio + tid; e < kRowWidth; e += kTopkThreads) {
        const int64_t j = e - selected * kRatio;
        o[e] = j < tail ? static_cast<int32_t>(nv * kRatio + j) : -1;
    }
}

// The top-512 kernels; every one writes the same bytes.
enum class TopkKernel {
    block,    // topk_kernel (the reference of the others)
    cluster,  // topk_cluster_kernel: the calls of up to kSplitMaxRows rows, ld <= kTkMaxBlocks
    prefill,  // topk_prefill_kernel: the larger calls, ld <= kTpMaxBlocks
};

// qsa_select's choice for a call of `rows` rows over logits rows of ld blocks.
TopkKernel topk_choice(int64_t rows, int64_t ld) {
    if (rows <= kSplitMaxRows) return ld <= kTkMaxBlocks ? TopkKernel::cluster : TopkKernel::block;
    return ld <= kTpMaxBlocks ? TopkKernel::prefill : TopkKernel::block;
}

// The top-512 + expansion of rows row0 .. row0 + n - 1 of a call (logits rows at logits + (r - row0) * ld) on
// `kernel` (the caller checked its ld capacity); the prefill kernel holds the fewest registers that cover ld.
cudaError_t launch_topk(const float* logits, int64_t ld, const int64_t* positions_logical, int64_t row0, int64_t n,
                        int64_t rows_total, int64_t seq_len, uint32_t* fault, int32_t* out, TopkKernel kernel,
                        cudaStream_t stream) {
    const unsigned rows = static_cast<unsigned>(n);
    if (kernel == TopkKernel::cluster) {
        topk_cluster_kernel<<<dim3(kTkCluster, rows), kTopkThreads, 0, stream>>>(logits, ld, positions_logical, row0,
                                                                                 rows_total, seq_len, fault, out);
    } else if (kernel == TopkKernel::prefill && ld <= int64_t{kTopkThreads} * 16) {
        topk_prefill_kernel<16><<<rows, kTopkThreads, 0, stream>>>(logits, ld, positions_logical, row0, rows_total,
                                                                   seq_len, fault, out);
    } else if (kernel == TopkKernel::prefill && ld <= int64_t{kTopkThreads} * 32) {
        topk_prefill_kernel<32><<<rows, kTopkThreads, 0, stream>>>(logits, ld, positions_logical, row0, rows_total,
                                                                   seq_len, fault, out);
    } else if (kernel == TopkKernel::prefill) {
        topk_prefill_kernel<kTpMaxPer><<<rows, kTopkThreads, 0, stream>>>(logits, ld, positions_logical, row0,
                                                                          rows_total, seq_len, fault, out);
    } else {
        topk_kernel<<<rows, kTopkThreads, 0, stream>>>(logits, ld, positions_logical, row0, rows_total, seq_len,
                                                       fault, out);
    }
    return cudaGetLastError();
}

// -- sparse attention ----------------------------------------------------------------------------------------------

struct AttnSmem {
    float q[kGroup][kHeadDim];
    bf16 k[kAttnKeyTile][kAttnKStride];
    bf16 v[kAttnKeyTile][kHeadDim];
    float s[kGroup][kAttnKeyTile];
    float alpha[kGroup];
    float m[kGroup];
    float l[kGroup];
    int64_t row[kAttnKeyTile];
    int end;
};

__device__ __forceinline__ void store_pair(bf16* dst, float a, float b) {
    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(a, b);
}

// Grid (rows, kKvHeads, splits). splits == 1: normalized BF16 output; else F32 partials (o, m, l) for the combine.
template <bool kFp8>
__global__ void __launch_bounds__(kAttnThreads) attention_kernel(
    const bf16* __restrict__ q, int64_t q_stride, const int32_t* __restrict__ indices, const void* __restrict__ k_src,
    int64_t k_stride, const void* __restrict__ v_src, int64_t v_stride, const int64_t* __restrict__ slots, float scale,
    int splits, int64_t rows, bf16* __restrict__ out, int64_t out_stride, float* __restrict__ part_o,
    float* __restrict__ part_m, float* __restrict__ part_l) {
    __shared__ __align__(16) AttnSmem sm;
    const int tid = threadIdx.x;
    const int warp = tid / kWarp;
    const int lane = tid % kWarp;
    const int64_t t = blockIdx.x;
    const int g = blockIdx.y;
    const int split = blockIdx.z;
    const int32_t* idx = indices + t * kRowWidth;

    // End of the valid entries (last index >= 0, plus 1).
    if (tid == 0) sm.end = 0;
    for (int h = tid; h < kGroup; h += kAttnThreads) {
        sm.m[h] = -INFINITY;
        sm.l[h] = 0.0f;
    }
    __syncthreads();
    int last = 0;
    for (int j = tid; j < kRowWidth; j += kAttnThreads)
        if (idx[j] >= 0) last = j + 1;
#pragma unroll
    for (int mask = kWarp / 2; mask >= 1; mask >>= 1) {
        const int o = __shfl_xor_sync(0xffffffffu, last, mask, kWarp);
        last = o > last ? o : last;
    }
    if (lane == 0) atomicMax(&sm.end, last);
    for (int e = tid; e < kGroup * kHeadDim; e += kAttnThreads) {
        const int h = e / kHeadDim;
        const int d = e % kHeadDim;
        sm.q[h][d] = __bfloat162float(q[t * q_stride + (g * kGroup + h) * kHeadDim + d]);
    }
    __syncthreads();
    const int end = sm.end;
    int per_split = (end + splits - 1) / splits;
    per_split = (per_split + kAttnKeyTile - 1) / kAttnKeyTile * kAttnKeyTile;
    const int lo = split * per_split;
    const int hi = lo + per_split < end ? lo + per_split : end;

    float acc[kGroup][2];
#pragma unroll
    for (int h = 0; h < kGroup; ++h) acc[h][0] = acc[h][1] = 0.0f;

    for (int j0 = lo; j0 < hi; j0 += kAttnKeyTile) {
        if (tid < kAttnKeyTile) {
            const int j = j0 + tid;
            int64_t row = -1;
            if (j < hi) {
                const int64_t li = idx[j];
                if (li >= 0) row = (kFp8 && slots) ? slots[li] : li;
            }
            sm.row[tid] = row;
        }
        __syncthreads();
        // Stage K and V rows of this KV head as BF16 (FP8 E4M3 -> BF16 is exact); invalid entries are zeros.
        for (int e = tid; e < kAttnKeyTile * (kHeadDim / 16); e += kAttnThreads) {
            const int j = e / (kHeadDim / 16);
            const int c = (e % (kHeadDim / 16)) * 16;
            const int64_t row = sm.row[j];
            float kf[16];
            float vf[16];
            if (row < 0) {
#pragma unroll
                for (int i = 0; i < 16; ++i) kf[i] = vf[i] = 0.0f;
            } else if constexpr (kFp8) {
                const uint8_t* kp = static_cast<const uint8_t*>(k_src) + row * k_stride + g * kHeadDim + c;
                const uint8_t* vp = static_cast<const uint8_t*>(v_src) + row * v_stride + g * kHeadDim + c;
                const uint4 kr = *reinterpret_cast<const uint4*>(kp);
                const uint4 vr = *reinterpret_cast<const uint4*>(vp);
                const uint8_t* kb = reinterpret_cast<const uint8_t*>(&kr);
                const uint8_t* vb = reinterpret_cast<const uint8_t*>(&vr);
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    kf[i] = fp8_to_float(kb[i]);
                    vf[i] = fp8_to_float(vb[i]);
                }
            } else {
                const bf16* kp = static_cast<const bf16*>(k_src) + row * k_stride + g * kHeadDim + c;
                const bf16* vp = static_cast<const bf16*>(v_src) + row * v_stride + g * kHeadDim + c;
#pragma unroll
                for (int h8 = 0; h8 < 2; ++h8) {
                    const uint4 kr = *reinterpret_cast<const uint4*>(kp + 8 * h8);
                    const uint4 vr = *reinterpret_cast<const uint4*>(vp + 8 * h8);
                    const __nv_bfloat162* k2 = reinterpret_cast<const __nv_bfloat162*>(&kr);
                    const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&vr);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float2 kk = __bfloat1622float2(k2[i]);
                        const float2 vv = __bfloat1622float2(v2[i]);
                        kf[8 * h8 + 2 * i] = kk.x;
                        kf[8 * h8 + 2 * i + 1] = kk.y;
                        vf[8 * h8 + 2 * i] = vv.x;
                        vf[8 * h8 + 2 * i + 1] = vv.y;
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 16; i += 2) {
                store_pair(&sm.k[j][c + i], kf[i], kf[i + 1]);
                store_pair(&sm.v[j][c + i], vf[i], vf[i + 1]);
            }
        }
        __syncthreads();
        // Scores: thread = (key lane, head group warp) -> heads 3 warp .. 3 warp + 2.
        {
            float dot[3] = {0.0f, 0.0f, 0.0f};
            const int h0 = warp * 3;
            for (int d = 0; d < kHeadDim; d += 2) {
                const float2 kk = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sm.k[lane][d]));
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    dot[i] = fmaf(sm.q[h0 + i][d], kk.x, dot[i]);
                    dot[i] = fmaf(sm.q[h0 + i][d + 1], kk.y, dot[i]);
                }
            }
            const bool valid = sm.row[lane] >= 0;
#pragma unroll
            for (int i = 0; i < 3; ++i) sm.s[h0 + i][lane] = valid ? dot[i] * scale : -INFINITY;
        }
        __syncthreads();
        // Online softmax per head: warp w handles heads w, w + 4, w + 8.
        for (int h = warp; h < kGroup; h += kAttnThreads / kWarp) {
            const float s = sm.s[h][lane];
            float mx = s;
#pragma unroll
            for (int mask = kWarp / 2; mask >= 1; mask >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, mask, kWarp));
            const float m_old = sm.m[h];
            const float m_new = fmaxf(m_old, mx);
            float pr = 0.0f;
            float alpha = 1.0f;
            if (m_new != -INFINITY) {
                pr = s == -INFINITY ? 0.0f : expf(s - m_new);
                alpha = m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
            }
            float sum = pr;
#pragma unroll
            for (int mask = kWarp / 2; mask >= 1; mask >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, mask, kWarp);
            sm.s[h][lane] = pr;
            __syncwarp();
            if (lane == 0) {
                sm.alpha[h] = alpha;
                sm.m[h] = m_new;
                sm.l[h] = sm.l[h] * alpha + sum;
            }
        }
        __syncthreads();
        // P.V: thread owns dims tid and tid + 128 of every head.
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            acc[h][0] *= sm.alpha[h];
            acc[h][1] *= sm.alpha[h];
        }
        for (int j = 0; j < kAttnKeyTile; ++j) {
            const float v0 = __bfloat162float(sm.v[j][tid]);
            const float v1 = __bfloat162float(sm.v[j][tid + kAttnThreads]);
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                const float pr = sm.s[h][j];
                acc[h][0] = fmaf(pr, v0, acc[h][0]);
                acc[h][1] = fmaf(pr, v1, acc[h][1]);
            }
        }
        __syncthreads();
    }

    if (splits == 1) {
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            const float l = sm.l[h];
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
            bf16* orow = out + t * out_stride + (g * kGroup + h) * kHeadDim;
            orow[tid] = __float2bfloat16_rn(acc[h][0] * inv);
            orow[tid + kAttnThreads] = __float2bfloat16_rn(acc[h][1] * inv);
        }
    } else {
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            const int64_t pr = (static_cast<int64_t>(split) * rows + t) * kQHeads + g * kGroup + h;
            part_o[pr * kHeadDim + tid] = acc[h][0];
            part_o[pr * kHeadDim + tid + kAttnThreads] = acc[h][1];
            if (tid == 0) {
                part_m[pr] = sm.m[h];
                part_l[pr] = sm.l[h];
            }
        }
    }
}

// Grid (rows, kQHeads), kHeadDim threads.
__global__ void combine_kernel(int splits, int64_t rows, const float* __restrict__ part_o,
                               const float* __restrict__ part_m, const float* __restrict__ part_l,
                               bf16* __restrict__ out, int64_t out_stride) {
    const int64_t t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, part_m[(s * rows + t) * kQHeads + h]);
    float l = 0.0f;
    float o = 0.0f;
    if (m != -INFINITY) {
        for (int s = 0; s < splits; ++s) {
            const int64_t pr = (s * rows + t) * kQHeads + h;
            const float ms = part_m[pr];
            if (ms == -INFINITY) continue;
            const float w = expf(ms - m);
            l += part_l[pr] * w;
            o += part_o[pr * kHeadDim + d] * w;
        }
    }
    out[t * out_stride + h * kHeadDim + d] = __float2bfloat16_rn(l > 0.0f ? o / l : 0.0f);
}

// -- rows attention (track PrefillTtft, prefill-ttft design note PT-02) ------------------------------------------
//
// rows_attention_kernel is attention_kernel's splits == 1 call (calls of more than kSplitMaxRows rows: the prefill)
// with the same F32 arithmetic for every output element, laid out for throughput. A warp owns kHeads query heads of
// one (row, KV head) and takes the keys two 32-key tiles at a time. Scores: lane = key of each tile; the lane reads
// its two K rows straight from the source and each dot is attention_kernel's FMA chain over d = 0 .. 255 (q as F32
// in shared memory, one broadcast load feeding both keys), then one rounded multiply by the scale (__fmul_rn: never
// contracted into the softmax's subtraction, as attention_kernel's trip through shared memory). Then each tile in
// order (the second only when attention_kernel has it, j0 + 32 < end): online softmax per head in registers with
// the same xor butterflies (max, then the sum of the probabilities), m and l held by every lane, l updated as
// l * alpha + sum; P.V: lane = 8 dims, acc *= alpha, then the FMAs over j = 0 .. 31 in tile order with the
// probabilities broadcast from shared memory and the V slices read straight from the source. Loads run ahead of
// their FMAs (the next 16 K elements per key during a chunk's dots; V in groups of kRowsVGroup keys, the first during
// the softmax). Invalid and past-the-end keys read row 0 and keep a zero V slice and a -inf score, so every FMA of
// attention_kernel happens here too. Output: acc * (1 / l) (0 for l == 0) rounded to BF16. FP8 E4M3 -> F16 -> F32 is
// exact for every byte (attention_kernel's fp8_to_float; NaN stays NaN). Bit-identity to attention_kernel:
// cere_ktest_qsa_attention_heads (test_kernels_k9r_rows.py).
constexpr int kRowsDims = kHeadDim / kWarp;  // P.V dims per lane (8)
constexpr int kRowsTiles = 2;                 // 32-key tiles per score pass

__device__ __forceinline__ void fp8x2_to_float(uint32_t two, float& a, float& b) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two & 0xffffu), __NV_E4M3);
    const float2 f = __half22float2(__half2(h));
    a = f.x;
    b = f.y;
}

// 4 FP8 bytes (one word) -> 4 floats.
__device__ __forceinline__ void fp8x4_to_float(uint32_t w, float* f) {
    fp8x2_to_float(w, f[0], f[1]);
    fp8x2_to_float(w >> 16, f[2], f[3]);
}

// 8 BF16 -> 8 floats.
__device__ __forceinline__ void bf16x8_to_float(const uint4& raw, float* f) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 x = __bfloat1622float2(p[i]);
        f[2 * i] = x.x;
        f[2 * i + 1] = x.y;
    }
}

// The raw bytes of 16 K elements of one key (FP8: one 16-byte word; BF16: two), loaded one chunk ahead of their use.
template <bool kFp8>
struct K16Raw {
    uint4 w[kFp8 ? 1 : 2];
};

// K elements 16c .. 16c + 15 of the key at kp.
template <bool kFp8>
__device__ __forceinline__ K16Raw<kFp8> k16_load(const void* kp, int c) {
    K16Raw<kFp8> r;
#pragma unroll
    for (int i = 0; i < (kFp8 ? 1 : 2); ++i) r.w[i] = __ldg(static_cast<const uint4*>(kp) + (kFp8 ? 1 : 2) * c + i);
    return r;
}

template <bool kFp8>
__device__ __forceinline__ void k16_to_float(const K16Raw<kFp8>& r, float* f) {
    if constexpr (kFp8) {
        fp8x4_to_float(r.w[0].x, f);
        fp8x4_to_float(r.w[0].y, f + 4);
        fp8x4_to_float(r.w[0].z, f + 8);
        fp8x4_to_float(r.w[0].w, f + 12);
    } else {
        bf16x8_to_float(r.w[0], f);
        bf16x8_to_float(r.w[1], f + 8);
    }
}

// One lane's V slice (kRowsDims elements at d0) of one key: FP8 8 bytes, BF16 16 bytes.
template <bool kFp8>
struct VRaw {
    uint32_t w[kFp8 ? 2 : 4];
};

// The V slice of source row r (row 0, in bounds, for an invalid key r < 0: its value is replaced by zeros).
template <bool kFp8>
__device__ __forceinline__ VRaw<kFp8> v_load(const uint8_t* vb, int64_t r, int64_t v_stride, int d0) {
    const int64_t rr = r >= 0 ? r : 0;
    VRaw<kFp8> out;
    if constexpr (kFp8) {
        const uint2 raw = __ldg(reinterpret_cast<const uint2*>(vb + rr * v_stride + d0));
        out.w[0] = raw.x;
        out.w[1] = raw.y;
    } else {
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(vb + (rr * v_stride + d0) * 2));
        out.w[0] = raw.x;
        out.w[1] = raw.y;
        out.w[2] = raw.z;
        out.w[3] = raw.w;
    }
    return out;
}

template <bool kFp8>
__device__ __forceinline__ void v_to_float(const VRaw<kFp8>& r, bool valid, float* f) {
    if constexpr (kFp8) {
        fp8x4_to_float(r.w[0], f);
        fp8x4_to_float(r.w[1], f + 4);
    } else {
        bf16x8_to_float(make_uint4(r.w[0], r.w[1], r.w[2], r.w[3]), f);
    }
#pragma unroll
    for (int d = 0; d < kRowsDims; ++d) f[d] = valid ? f[d] : 0.0f;
}

constexpr int kRowsVGroup = 4;  // V slices loaded ahead of their FMAs (one group while the previous one computes)
// Blocks per SM the register allocation is made for (__launch_bounds__): no spills at 3, 6 or 12 heads per warp.
constexpr int rows_min_blocks(int heads) { return heads == 3 ? 4 : heads == 6 ? 5 : 1; }

// dot[t][h] = fmaf(q[h][d], k[t][d], dot[t][h]) for the 4 consecutive d at kf[t] + i (one q load per head).
template <int kHeads>
__device__ __forceinline__ void dot4x2(const float (*sq)[kHeadDim], int h0, int d, const float (*kf)[16], int i,
                                       float (*dot)[kHeads]) {
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
        const float4 qv = *reinterpret_cast<const float4*>(&sq[h0 + h][d]);
#pragma unroll
        for (int t = 0; t < kRowsTiles; ++t) {
            dot[t][h] = fmaf(qv.x, kf[t][i], dot[t][h]);
            dot[t][h] = fmaf(qv.y, kf[t][i + 1], dot[t][h]);
            dot[t][h] = fmaf(qv.z, kf[t][i + 2], dot[t][h]);
            dot[t][h] = fmaf(qv.w, kf[t][i + 3], dot[t][h]);
        }
    }
}

// Order-preserving signed-int key of a float (NaN -> the key of -inf, as fmaxf ignores NaN), and back.
__device__ __forceinline__ int max_key(float x) {
    const int b = x != x ? static_cast<int>(0xff800000u) : __float_as_int(x);
    return b ^ ((b >> 31) & 0x7fffffff);
}
__device__ __forceinline__ float key_float(int k) { return __int_as_float(k ^ ((k >> 31) & 0x7fffffff)); }

// O-PTC2 (FP8, one warp of all kGroup heads: the prefill's rows_attention_kernel<true, 12>), bit-identical to the
// general form: the tile max of every head is one redux.sync over the int keys (the max of the butterflies' fmaxf),
// and lane h sums head h's probabilities from shared memory in the butterflies' pairing (x[i] + x[i + 16], then + 8,
// 4, 2, 1); l lives in lane h. The next score pass's source rows are read during this pass.
// Grid (rows, kKvHeads), kGroup / kHeads warps.
template <bool kFp8, int kHeads>
__global__ void __launch_bounds__(kGroup / kHeads * kWarp, rows_min_blocks(kHeads)) rows_attention_kernel(
    const bf16* __restrict__ q, int64_t q_stride, const int32_t* __restrict__ indices, const void* __restrict__ k_src,
    int64_t k_stride, const void* __restrict__ v_src, int64_t v_stride, const int64_t* __restrict__ slots, float scale,
    bf16* __restrict__ out, int64_t out_stride) {
    constexpr int kWarps = kGroup / kHeads;
    constexpr int kThreads = kWarps * kWarp;
    constexpr int kElem = kFp8 ? 1 : 2;  // K/V bytes per element
    __shared__ __align__(16) float sq[kGroup][kHeadDim];
    constexpr bool kFast = kFp8 && kHeads == kGroup;  // O-PTC2 (above)
    constexpr int kSpW = kFast ? kAttnKeyTile + 4 : kAttnKeyTile;  // probabilities row stride (floats)
    __shared__ __align__(16) float sp[kWarps][kHeads][kSpW];
    __shared__ float sa[kFast ? kGroup : 1];  // O-PTC2: alpha of head h for lane h's l
    __shared__ int s_end;
    const int tid = threadIdx.x;
    const int warp = tid / kWarp;
    const int lane = tid % kWarp;
    const int64_t t = blockIdx.x;
    const int g = blockIdx.y;
    const int h0 = warp * kHeads;
    const int32_t* idx = indices + t * kRowWidth;
    const uint8_t* kb = static_cast<const uint8_t*>(k_src) + int64_t(g) * kHeadDim * kElem;
    const uint8_t* vb = static_cast<const uint8_t*>(v_src) + int64_t(g) * kHeadDim * kElem;

    if (tid == 0) s_end = 0;
    for (int e = tid; e < kGroup * kHeadDim / 8; e += kThreads) {
        const uint4 raw = *reinterpret_cast<const uint4*>(q + t * q_stride + int64_t(g) * kGroup * kHeadDim + 8 * e);
        float f[8];
        bf16x8_to_float(raw, f);
        float4* dst = reinterpret_cast<float4*>(&sq[0][0] + 8 * e);
        dst[0] = make_float4(f[0], f[1], f[2], f[3]);
        dst[1] = make_float4(f[4], f[5], f[6], f[7]);
    }
    __syncthreads();
    int last = 0;
    for (int j = tid; j < kRowWidth; j += kThreads)
        if (idx[j] >= 0) last = j + 1;
#pragma unroll
    for (int mask = kWarp / 2; mask >= 1; mask >>= 1) {
        const int o = __shfl_xor_sync(0xffffffffu, last, mask, kWarp);
        last = o > last ? o : last;
    }
    if (lane == 0) atomicMax(&s_end, last);
    __syncthreads();
    const int end = s_end;

    const auto row_of = [&](int j) -> int64_t {
        if (j >= end) return -1;
        const int64_t li = idx[j];
        if (li < 0) return -1;
        return (kFp8 && slots) ? slots[li] : li;
    };
    const int d0 = lane * kRowsDims;
    float m[kHeads];
    float l[kFast ? 1 : kHeads];  // O-PTC2: lane h holds head h's l (lanes >= kGroup unused)
    float acc[kHeads][kRowsDims];
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
        m[h] = -INFINITY;
#pragma unroll
        for (int d = 0; d < kRowsDims; ++d) acc[h][d] = 0.0f;
    }
#pragma unroll
    for (int h = 0; h < (kFast ? 1 : kHeads); ++h) l[h] = 0.0f;
    int64_t next_rows[kRowsTiles];
    if constexpr (kFast) {
#pragma unroll
        for (int tt = 0; tt < kRowsTiles; ++tt) next_rows[tt] = row_of(tt * kAttnKeyTile + lane);
    }

    for (int j0 = 0; j0 < end; j0 += kRowsTiles * kAttnKeyTile) {
        // Scores of keys j0 + 32 tt + lane.
        int64_t rows[kRowsTiles];
        const uint8_t* kp[kRowsTiles];
        float dot[kRowsTiles][kHeads];
        K16Raw<kFp8> kr[kRowsTiles];
#pragma unroll
        for (int tt = 0; tt < kRowsTiles; ++tt) {
            rows[tt] = kFast ? next_rows[tt] : row_of(j0 + tt * kAttnKeyTile + lane);
            // An invalid key reads row 0 of the source (in bounds) and its score is never used.
            kp[tt] = kb + (rows[tt] >= 0 ? rows[tt] : 0) * k_stride * kElem;
            kr[tt] = k16_load<kFp8>(kp[tt], 0);
#pragma unroll
            for (int h = 0; h < kHeads; ++h) dot[tt][h] = 0.0f;
        }
        if constexpr (kFast) {
#pragma unroll
            for (int tt = 0; tt < kRowsTiles; ++tt) next_rows[tt] = row_of(j0 + (kRowsTiles + tt) * kAttnKeyTile + lane);
        }
#pragma unroll 2
        for (int c = 0; c < kHeadDim / 16; ++c) {
            // The next chunk's K words are in flight while this chunk's FMAs run (the last chunk reloads itself).
            K16Raw<kFp8> kn[kRowsTiles];
#pragma unroll
            for (int tt = 0; tt < kRowsTiles; ++tt) kn[tt] = k16_load<kFp8>(kp[tt], c + 1 < kHeadDim / 16 ? c + 1 : c);
            float kf[kRowsTiles][16];
#pragma unroll
            for (int tt = 0; tt < kRowsTiles; ++tt) k16_to_float<kFp8>(kr[tt], kf[tt]);
#pragma unroll
            for (int i = 0; i < 16; i += 4) dot4x2<kHeads>(sq, h0, c * 16 + i, kf, i, dot);
#pragma unroll
            for (int tt = 0; tt < kRowsTiles; ++tt) kr[tt] = kn[tt];
        }
        // Each tile in order: online softmax per head (attention_kernel's butterflies; every lane holds m and l),
        // then P.V over the tile's keys in order.
#pragma unroll
        for (int tt = 0; tt < kRowsTiles; ++tt) {
            if (j0 + tt * kAttnKeyTile >= end) break;
            const bool valid = rows[tt] >= 0;
            // The first V group of the tile is in flight during the softmax (key j's row from lane j).
            VRaw<kFp8> vr[kRowsVGroup];
            bool vv[kRowsVGroup];
#pragma unroll
            for (int jj = 0; jj < kRowsVGroup; ++jj) {
                const int64_t r = __shfl_sync(0xffffffffu, rows[tt], jj, kWarp);
                vv[jj] = r >= 0;
                vr[jj] = v_load<kFp8>(vb, r, v_stride, d0);
            }
            float alpha[kHeads];
            if constexpr (kFast) {
                float s[kHeads];
                int mk[kHeads];
#pragma unroll
                for (int h = 0; h < kHeads; ++h) {
                    s[h] = valid ? __fmul_rn(dot[tt][h], scale) : -INFINITY;
                    mk[h] = __reduce_max_sync(0xffffffffu, max_key(s[h]));
                }
#pragma unroll
                for (int h = 0; h < kHeads; ++h) {
                    const float m_new = fmaxf(m[h], key_float(mk[h]));
                    const bool live = m_new != -INFINITY;
                    const float pr = live && s[h] != -INFINITY ? expf(s[h] - m_new) : 0.0f;
                    const float a = !live ? 1.0f : m[h] == -INFINITY ? 0.0f : expf(m[h] - m_new);
                    sp[warp][h][lane] = pr;
                    if (lane == h) sa[h] = a;
                    m[h] = m_new;
                    alpha[h] = a;
                }
                __syncwarp();
                if (lane < kHeads) {
                    const float* row = sp[warp][lane];
                    float x[16];
#pragma unroll
                    for (int i = 0; i < 16; i += 4) {
                        const float4 lo = *reinterpret_cast<const float4*>(row + i);
                        const float4 hi = *reinterpret_cast<const float4*>(row + 16 + i);
                        x[i] = lo.x + hi.x;
                        x[i + 1] = lo.y + hi.y;
                        x[i + 2] = lo.z + hi.z;
                        x[i + 3] = lo.w + hi.w;
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) x[i] = x[i] + x[i + 8];
#pragma unroll
                    for (int i = 0; i < 4; ++i) x[i] = x[i] + x[i + 4];
#pragma unroll
                    for (int i = 0; i < 2; ++i) x[i] = x[i] + x[i + 2];
                    l[0] = l[0] * sa[lane] + (x[0] + x[1]);
                }
            } else {
#pragma unroll
                for (int h = 0; h < kHeads; ++h) {
                    const float s = valid ? __fmul_rn(dot[tt][h], scale) : -INFINITY;
                    float mx = s;
#pragma unroll
                    for (int mask = kWarp / 2; mask >= 1; mask >>= 1)
                        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, mask, kWarp));
                    const float m_new = fmaxf(m[h], mx);
                    float pr = 0.0f;
                    float a = 1.0f;
                    if (m_new != -INFINITY) {
                        pr = s == -INFINITY ? 0.0f : expf(s - m_new);
                        a = m[h] == -INFINITY ? 0.0f : expf(m[h] - m_new);
                    }
                    float sum = pr;
#pragma unroll
                    for (int mask = kWarp / 2; mask >= 1; mask >>= 1)
                        sum += __shfl_xor_sync(0xffffffffu, sum, mask, kWarp);
                    sp[warp][h][lane] = pr;
                    l[kFast ? 0 : h] = l[kFast ? 0 : h] * a + sum;
                    m[h] = m_new;
                    alpha[h] = a;
                }
                __syncwarp();
            }
#pragma unroll
            for (int h = 0; h < kHeads; ++h) {
#pragma unroll
                for (int d = 0; d < kRowsDims; ++d) acc[h][d] *= alpha[h];
            }
#pragma unroll 1
            for (int jg = 0; jg < kAttnKeyTile; jg += kRowsVGroup) {
                VRaw<kFp8> cur[kRowsVGroup];
                bool cv[kRowsVGroup];
#pragma unroll
                for (int jj = 0; jj < kRowsVGroup; ++jj) {
                    cur[jj] = vr[jj];
                    cv[jj] = vv[jj];
                }
                // The next group's V slices are in flight while this group's FMAs run (the last group reloads the
                // tile's first, unused).
#pragma unroll
                for (int jj = 0; jj < kRowsVGroup; ++jj) {
                    const int64_t r = __shfl_sync(0xffffffffu, rows[tt], (jg + kRowsVGroup + jj) % kAttnKeyTile, kWarp);
                    vv[jj] = r >= 0;
                    vr[jj] = v_load<kFp8>(vb, r, v_stride, d0);
                }
#pragma unroll
                for (int j4 = 0; j4 < kRowsVGroup; j4 += 4) {
                    float pr4[kHeads][4];
#pragma unroll
                    for (int h = 0; h < kHeads; ++h)
                        *reinterpret_cast<float4*>(pr4[h]) = *reinterpret_cast<const float4*>(&sp[warp][h][jg + j4]);
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        float vf[kRowsDims];
                        v_to_float<kFp8>(cur[j4 + jj], cv[j4 + jj], vf);
#pragma unroll
                        for (int h = 0; h < kHeads; ++h) {
#pragma unroll
                            for (int d = 0; d < kRowsDims; ++d) acc[h][d] = fmaf(pr4[h][jj], vf[d], acc[h][d]);
                        }
                    }
                }
            }
            __syncwarp();
        }
    }

#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
        const float lh = kFast ? __shfl_sync(0xffffffffu, l[0], h, kWarp) : l[kFast ? 0 : h];
        const float inv = lh > 0.0f ? 1.0f / lh : 0.0f;
        uint4 packed;
        __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
        for (int i = 0; i < kRowsDims / 2; ++i)
            p[i] = __halves2bfloat162(__float2bfloat16_rn(acc[h][2 * i] * inv), __float2bfloat16_rn(acc[h][2 * i + 1] * inv));
        *reinterpret_cast<uint4*>(out + t * out_stride + int64_t(g * kGroup + h0 + h) * kHeadDim + d0) = packed;
    }
}

// qsa_key_save_step's rule at the position `at` (kIndexHeadDim threads, one element each: a bit copy; at inside the
// context, so q - 3 <= at reads inside the slot map).
__device__ __forceinline__ void key_save_row(const bf16* __restrict__ keys, const int64_t* __restrict__ slots,
                                             int64_t at, int64_t j, int64_t n, bf16* __restrict__ saved) {
    const int64_t p = at - j;
    const int64_t q = at + (kRatio - 1 - at % kRatio) % kRatio;  // first q >= p + j with q % 4 == 3
    if (q > p + n) return;
    const int64_t row = slots[q - (kRatio - 1)] / kRatio;
    saved[threadIdx.x] = keys[row * kIndexHeadDim + threadIdx.x];
}

// qsa_key_save_step: one block of kIndexHeadDim threads.
__global__ void key_save_kernel(const bf16* __restrict__ keys, const int64_t* __restrict__ slots, Bounds b,
                                const int64_t* __restrict__ position, int64_t j, int64_t n, bf16* __restrict__ saved) {
    key_save_row(keys, slots, bounded(*position, 0, b.context, b.fault), j, n, saved);
}

// qsa_chains_step: block c = chain c, kIndexHeadDim threads, its rows in row order; per row the undo ring copy (a bit
// copy), key_save_row, a barrier, compress_step_row, a barrier (the next row's undo reads what this row wrote).
__global__ void __launch_bounds__(kIndexHeadDim) chains_step_kernel(ChainsStep a, const bf16* __restrict__ weight,
                                                                    float eps, const float* __restrict__ cos_sin,
                                                                    Bounds b, const int64_t* __restrict__ positions) {
    __shared__ bf16 row[kIndexHeadDim];
    const int first = a.chain_first[blockIdx.x];
    const int end = first + a.chain_rows[blockIdx.x];
    for (int r = first; r < end; ++r) {
        const int64_t at = bounded(positions[r], 0, a.context[r], b.fault);
        if (a.undo_ring[r] != nullptr) {
            for (int i = static_cast<int>(threadIdx.x); i < kRatio * kIndexHeadDim; i += kIndexHeadDim) {
                a.undo_ring[r][i] = a.ring[r][i];
            }
            if (a.undo_key[r] != nullptr) key_save_row(a.keys[r], a.slots[r], at, a.j[r], a.n[r], a.undo_key[r]);
        }
        if (a.k_idx[r] == nullptr) continue;
        __syncthreads();
        compress_step_row(a.k_idx[r], weight, eps, cos_sin, b, a.slots[r], at, a.ring[r], a.keys[r], row);
        __syncthreads();
    }
}

// -- split attention (track LongK9, long-k9 design note LK9-02) ---------------------------------------------------
//
// split_attention_kernel is attention_kernel's splits > 1 call on the FP8 cache (calls of up to kSplitMaxRows rows:
// decode, verify, the MTP draft) with the same F32 arithmetic for every partial. attention_kernel walks its split's
// keys in 32-key tiles, each tile a dependent trip to memory (rows, stage, scores, softmax, P.V) for 128 threads;
// here 384 threads stage every key of the split (at most kSaKeys) at once by cp.async (zero-filled for invalid and
// past-the-end keys, as attention_kernel's zero tiles), K and V as separate groups so the scores run while V lands:
//   scores:  warp 5 i + tile, thread = (key 32 tile + lane, heads 6 i .. 6 i + 5): dot = fmaf chain over d ascending
//            from 0 of q (BF16 -> F32) and K (FP8 -> F16 -> F32, exact: attention_kernel's values), s = dot * scale,
//            -inf for an invalid or past-the-end key;
//   softmax: warp h = head h walks the tiles in order with attention_kernel's per-tile update (xor butterflies for
//            the max and the sum, expf, l = l * alpha + sum), m and l held by every lane;
//   P.V:     thread = (dims 2 dp, 2 dp + 1, heads 4 i .. 4 i + 3): per tile acc *= alpha, then the FMAs over the
//            tile's 32 keys in order (invalid keys: zero V and probability, as attention_kernel).
// Partials (o, m, l) as attention_kernel's for combine_kernel; an empty split writes (0, -inf, 0) as attention_kernel
// does without a tile. Bit-identity to attention_kernel: cere_ktest_qsa_attention_heads (test_kernels_k9r_decode.py,
// test_kernels_k9r_rows.py). Grid (rows, kKvHeads, kSplits); dynamic shared memory sizeof(SplitSmem).
struct SplitSmem {
    float q[kGroup][kHeadDim];
    union {
        uint8_t k[kSaKeys][kSaKStride];  // K rows of this KV head
        float p[kSaKeys][kGroup];        // then the scores, then the probabilities (4 heads per 16-byte load)
    } kp;
    uint8_t v[kSaKeys][kHeadDim];
    int64_t row[kSaKeys];
    float alpha[kSaTiles][kGroup];
    int last[kSaWarps];
};
static_assert(sizeof(SplitSmem) <= 99 * 1024, "split_attention_kernel: sm_120's dynamic shared memory per block");

__global__ void __launch_bounds__(kSaThreads) split_attention_kernel(
    const bf16* __restrict__ q, int64_t q_stride, const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ k_cache, const uint8_t* __restrict__ v_cache, const int64_t* __restrict__ slots,
    float scale, int64_t rows, float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l) {
    extern __shared__ __align__(16) unsigned char split_smem[];
    SplitSmem& sm = *reinterpret_cast<SplitSmem*>(split_smem);
    const int tid = threadIdx.x;
    const int warp = tid / kWarp;
    const int lane = tid % kWarp;
    const int64_t t = blockIdx.x;
    const int g = blockIdx.y;
    const int split = blockIdx.z;
    const int32_t* idx = indices + t * kRowWidth;

    // q of the KV head's 12 query heads (8 per thread), and the end of the valid entries (last index >= 0, plus 1).
    {
        const uint4 raw = *reinterpret_cast<const uint4*>(q + t * q_stride + g * kGroup * kHeadDim + tid * 8);
        bf16x8_to_float(raw, &sm.q[0][0] + tid * 8);
    }
    int last = 0;
    for (int j = tid; j < kRowWidth; j += kSaThreads)
        if (idx[j] >= 0) last = j + 1;
#pragma unroll
    for (int mask = kWarp / 2; mask >= 1; mask >>= 1) {
        const int o = __shfl_xor_sync(0xffffffffu, last, mask, kWarp);
        last = o > last ? o : last;
    }
    if (lane == 0) sm.last[warp] = last;
    __syncthreads();
    int end = 0;
#pragma unroll
    for (int w = 0; w < kSaWarps; ++w) end = sm.last[w] > end ? sm.last[w] : end;
    int per_split = (end + kSplits - 1) / kSplits;
    per_split = (per_split + kAttnKeyTile - 1) / kAttnKeyTile * kAttnKeyTile;
    const int lo = split * per_split;
    const int hi = lo + per_split < end ? lo + per_split : end;
    const int64_t pr0 = (static_cast<int64_t>(split) * rows + t) * kQHeads + g * kGroup;  // partial row of head 0
    if (lo >= hi) {
        for (int e = tid; e < kGroup * kHeadDim; e += kSaThreads) part_o[pr0 * kHeadDim + e] = 0.0f;
        if (tid < kGroup) {
            part_m[pr0 + tid] = -INFINITY;
            part_l[pr0 + tid] = 0.0f;
        }
        return;
    }
    const int tiles = (hi - lo + kAttnKeyTile - 1) / kAttnKeyTile;
    const int keys = tiles * kAttnKeyTile;
    if (tid < keys) {
        int64_t row = -1;
        if (lo + tid < hi) {
            const int64_t li = idx[lo + tid];
            if (li >= 0) row = slots ? slots[li] : li;
        }
        sm.row[tid] = row;
    }
    __syncthreads();
    constexpr int kChunks = kHeadDim / 16;  // 16-byte chunks of one key's KV-head row
    const uint32_t k_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sm.kp.k[0][0]));
    const uint32_t v_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sm.v[0][0]));
    for (int e = tid; e < keys * kChunks; e += kSaThreads) {
        const int j = e / kChunks;
        const int c = (e % kChunks) * 16;
        const int64_t row = sm.row[j];
        const uint8_t* src = k_cache + (row >= 0 ? row * kKvWidth + g * kHeadDim + c : 0);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(k_addr + j * kSaKStride + c), "l"(src),
                     "r"(row >= 0 ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;\n" ::);
    for (int e = tid; e < keys * kChunks; e += kSaThreads) {
        const int j = e / kChunks;
        const int c = (e % kChunks) * 16;
        const int64_t row = sm.row[j];
        const uint8_t* src = v_cache + (row >= 0 ? row * kKvWidth + g * kHeadDim + c : 0);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(v_addr + j * kHeadDim + c), "l"(src),
                     "r"(row >= 0 ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();  // K

    // Scores.
    const int tile = warp % kSaTiles;
    const int h0 = warp / kSaTiles * kSaScoreHeads;
    const bool scorer = warp < kSaTiles * (kGroup / kSaScoreHeads) && tile < tiles;
    const int js = tile * kAttnKeyTile + lane;
    float s[kSaScoreHeads] = {};
    if (scorer) {
        float dot[kSaScoreHeads];
#pragma unroll
        for (int i = 0; i < kSaScoreHeads; ++i) dot[i] = 0.0f;
        const uint8_t* krow = &sm.kp.k[js][0];
#pragma unroll 2
        for (int c = 0; c < kHeadDim; c += 16) {
            const uint4 kr = *reinterpret_cast<const uint4*>(krow + c);
            float kf[16];
            fp8x4_to_float(kr.x, kf);
            fp8x4_to_float(kr.y, kf + 4);
            fp8x4_to_float(kr.z, kf + 8);
            fp8x4_to_float(kr.w, kf + 12);
#pragma unroll
            for (int i = 0; i < 16; i += 4) {
#pragma unroll
                for (int hh = 0; hh < kSaScoreHeads; ++hh) {
                    const float4 qv = *reinterpret_cast<const float4*>(&sm.q[h0 + hh][c + i]);
                    dot[hh] = fmaf(qv.x, kf[i], dot[hh]);
                    dot[hh] = fmaf(qv.y, kf[i + 1], dot[hh]);
                    dot[hh] = fmaf(qv.z, kf[i + 2], dot[hh]);
                    dot[hh] = fmaf(qv.w, kf[i + 3], dot[hh]);
                }
            }
        }
        const bool valid = sm.row[js] >= 0;
#pragma unroll
        for (int i = 0; i < kSaScoreHeads; ++i) s[i] = valid ? dot[i] * scale : -INFINITY;
    }
    __syncthreads();  // every K read before the scores overwrite it
    if (scorer) {
#pragma unroll
        for (int i = 0; i < kSaScoreHeads; ++i) sm.kp.p[js][h0 + i] = s[i];
    }
    __syncthreads();

    // Online softmax, warp = head, tiles in order.
    {
        const int h = warp;
        float m = -INFINITY;
        float l = 0.0f;
        for (int tt = 0; tt < tiles; ++tt) {
            float* ps = &sm.kp.p[tt * kAttnKeyTile + lane][h];
            const float sc = *ps;
            float mx = sc;
#pragma unroll
            for (int mask = kWarp / 2; mask >= 1; mask >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, mask, kWarp));
            const float m_new = fmaxf(m, mx);
            float pr = 0.0f;
            float a = 1.0f;
            if (m_new != -INFINITY) {
                pr = sc == -INFINITY ? 0.0f : expf(sc - m_new);
                a = m == -INFINITY ? 0.0f : expf(m - m_new);
            }
            float sum = pr;
#pragma unroll
            for (int mask = kWarp / 2; mask >= 1; mask >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, mask, kWarp);
            *ps = pr;
            if (lane == 0) sm.alpha[tt][h] = a;
            l = l * a + sum;
            m = m_new;
        }
        if (lane == 0) {
            part_m[pr0 + h] = m;
            part_l[pr0 + h] = l;
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();  // probabilities, alphas, V

    // P.V.
    const int dp = tid % (kHeadDim / 2);
    const int hp = tid / (kHeadDim / 2) * kSaPvHeads;
    float acc[kSaPvHeads][2];
#pragma unroll
    for (int i = 0; i < kSaPvHeads; ++i) acc[i][0] = acc[i][1] = 0.0f;
    for (int tt = 0; tt < tiles; ++tt) {
#pragma unroll
        for (int i = 0; i < kSaPvHeads; ++i) {
            acc[i][0] *= sm.alpha[tt][hp + i];
            acc[i][1] *= sm.alpha[tt][hp + i];
        }
#pragma unroll 8
        for (int jj = 0; jj < kAttnKeyTile; ++jj) {
            const int j = tt * kAttnKeyTile + jj;
            float v0;
            float v1;
            fp8x2_to_float(*reinterpret_cast<const uint16_t*>(&sm.v[j][2 * dp]), v0, v1);
            const float4 p = *reinterpret_cast<const float4*>(&sm.kp.p[j][hp]);
            const float pv[kSaPvHeads] = {p.x, p.y, p.z, p.w};
#pragma unroll
            for (int i = 0; i < kSaPvHeads; ++i) {
                acc[i][0] = fmaf(pv[i], v0, acc[i][0]);
                acc[i][1] = fmaf(pv[i], v1, acc[i][1]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kSaPvHeads; ++i)
        *reinterpret_cast<float2*>(&part_o[(pr0 + hp + i) * kHeadDim + 2 * dp]) = make_float2(acc[i][0], acc[i][1]);
}

cudaError_t launch_split(const bf16* q, int64_t q_stride, const int32_t* indices, int64_t rows,
                         const uint8_t* k_cache, const uint8_t* v_cache, const int64_t* slots, float scale,
                         float* part_o, float* part_m, float* part_l, cudaStream_t stream) {
    constexpr int bytes = static_cast<int>(sizeof(SplitSmem));
    const cudaError_t err =
        cudaFuncSetAttribute(split_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (err != cudaSuccess) return err;
    const dim3 grid(static_cast<unsigned>(rows), kKvHeads, kSplits);
    split_attention_kernel<<<grid, kSaThreads, bytes, stream>>>(q, q_stride, indices, k_cache, v_cache, slots, scale,
                                                                rows, part_o, part_m, part_l);
    return cudaGetLastError();
}

size_t partial_rows(int64_t rows) { return static_cast<size_t>(kSplits) * static_cast<size_t>(rows) * kQHeads; }

template <int kHeads>
void launch_rows(const bf16* q, int64_t q_stride, const int32_t* indices, int64_t rows, const void* k,
                 int64_t k_stride, const void* v, int64_t v_stride, const int64_t* slots, bool fp8, float scale,
                 bf16* out, int64_t out_stride, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(rows), kKvHeads);
    constexpr int threads = kGroup / kHeads * kWarp;
    if (fp8) {
        rows_attention_kernel<true, kHeads><<<grid, threads, 0, stream>>>(q, q_stride, indices, k, k_stride, v, v_stride,
                                                                         slots, scale, out, out_stride);
    } else {
        rows_attention_kernel<false, kHeads><<<grid, threads, 0, stream>>>(q, q_stride, indices, k, k_stride, v,
                                                                          v_stride, nullptr, scale, out, out_stride);
    }
}

}  // namespace

int64_t logits_tile_rows(int64_t rows, int64_t ctx) {
    const int64_t keys = ctx / kRatio;
    if (rows <= 0 || keys <= 0) return rows > 1 ? rows : 1;
    const int64_t block_q = 128 / kIndexHeads;
    const int64_t bytes_per_row = keys * static_cast<int64_t>(sizeof(float));
    int64_t max_padded = static_cast<int64_t>(kLogitsTileBytes) / bytes_per_row;
    if (max_padded < block_q) max_padded = block_q;
    max_padded = max_padded / block_q * block_q;
    if (max_padded < block_q) max_padded = block_q;
    return rows < max_padded ? rows : max_padded;
}

size_t select_scratch_bytes(int64_t rows, int64_t ctx) {
    if (rows <= 0 || ctx < kRatio) return 0;
    return align256(static_cast<size_t>(logits_tile_rows(rows, ctx)) * static_cast<size_t>(ctx / kRatio) *
                    sizeof(float));
}

size_t attention_scratch_bytes(int64_t rows) {
    if (rows <= 0 || rows > kSplitMaxRows) return 0;
    const size_t pr = partial_rows(rows);
    return align256(pr * kHeadDim * sizeof(float)) + 2 * align256(pr * sizeof(float));
}

size_t scratch_bytes(int64_t rows, int64_t ctx) {
    if (rows <= 0) return 0;
    const size_t r = static_cast<size_t>(rows);
    const size_t sel = select_scratch_bytes(rows, ctx);
    const size_t att = attention_scratch_bytes(rows);
    return align256(r * kIndexHeads * kIndexHeadDim * sizeof(bf16)) + align256(r * kRowWidth * sizeof(int32_t)) +
           (sel > att ? sel : att);
}

cere_status carve_scratch(void* base, size_t bytes, int64_t rows, int64_t ctx, Scratch* out) {
    if (!out || !base || rows <= 0 || ctx <= 0 || !aligned(base, 256) || bytes < scratch_bytes(rows, ctx))
        return CERE_ERR_ARGUMENT;
    const size_t r = static_cast<size_t>(rows);
    unsigned char* p = static_cast<unsigned char*>(base);
    out->q_idx = reinterpret_cast<bf16*>(p);
    p += align256(r * kIndexHeads * kIndexHeadDim * sizeof(bf16));
    out->indices = reinterpret_cast<int32_t*>(p);
    p += align256(r * kRowWidth * sizeof(int32_t));
    out->work = p;
    out->work_bytes = bytes - static_cast<size_t>(p - static_cast<unsigned char*>(base));
    return CERE_OK;
}

namespace {

// The entry points' Bounds checks (k9_qsa.h): a non-empty rotary range; rotary rows lo .. pos0 + tokens - 1 inside
// it; logical positions pos0 .. pos0 + tokens - 1 inside a context of at least one position (no int64 overflow).
bool rope_ok(const Bounds& b) { return b.rope_first < b.rope_end; }
bool rope_rows_ok(const Bounds& b, int64_t lo, int64_t pos0, int64_t tokens) {
    return rope_ok(b) && lo >= b.rope_first && pos0 <= b.rope_end && tokens <= b.rope_end - pos0;
}
bool context_ok(const Bounds& b, int64_t pos0, int64_t tokens) {
    return b.context >= 1 && pos0 <= b.context && tokens <= b.context - pos0;
}

}  // namespace

cere_status indexer_qk(const bf16* qk, int64_t qk_stride, const bf16* q_norm_weight, float eps, const float* cos_sin,
                       const Bounds& bounds, const int64_t* positions, int64_t pos0, int64_t tokens, bf16* q_idx,
                       cudaStream_t stream) {
    if (!qk || !q_norm_weight || !cos_sin || !q_idx || tokens < 0 || pos0 < 0 || qk_stride < kQkWidth ||
        !(eps > 0.0f) || tokens > INT32_MAX || !rope_ok(bounds))
        return CERE_ERR_ARGUMENT;
    if (tokens == 0) return CERE_OK;
    if (!positions && !rope_rows_ok(bounds, pos0, pos0, tokens)) return CERE_ERR_ARGUMENT;
    index_q_kernel<<<static_cast<unsigned>(tokens), kIndexHeads * kWarp, 0, stream>>>(qk, qk_stride, q_norm_weight,
                                                                                      eps, cos_sin, bounds, positions,
                                                                                      pos0, q_idx);
    return status_of(cudaGetLastError());
}

cere_status qsa_compress_store(const bf16* k_idx, int64_t k_stride, const bf16* k_norm_weight, float eps,
                               const float* cos_sin, const Bounds& bounds, const int64_t* positions,
                               const int64_t* slots, int64_t pos0, int64_t tokens, bf16* ring, bf16* keys,
                               cudaStream_t stream) {
    if (!k_idx || !k_norm_weight || !cos_sin || !ring || !keys || tokens < 0 || pos0 < 0 ||
        k_stride < kIndexHeadDim || !(eps > 0.0f) || !rope_ok(bounds))
        return CERE_ERR_ARGUMENT;
    if (tokens == 0) return CERE_OK;
    if (!context_ok(bounds, pos0, tokens) ||
        (!positions && !rope_rows_ok(bounds, pos0 / kRatio * kRatio, pos0, tokens)))
        return CERE_ERR_ARGUMENT;
    const int64_t end = pos0 + tokens;
    const int64_t g_lo = pos0 / kRatio;
    const int64_t g_hi = end / kRatio;
    if (g_hi > g_lo) {
        const int64_t groups = g_hi - g_lo;
        compress_kernel<<<static_cast<unsigned>((groups + 3) / 4), 4 * kWarp, 0, stream>>>(
            k_idx, k_stride, k_norm_weight, eps, cos_sin, bounds, positions, slots, pos0, g_lo, groups, ring, keys);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return status_of(err);
    }
    // After the compress kernel (stream order): it may still read ring rows this overwrites.
    const int64_t pending = g_hi * kRatio > pos0 ? g_hi * kRatio : pos0;
    if (pending < end) {
        ring_kernel<<<1, kIndexHeadDim, 0, stream>>>(k_idx, k_stride, pos0, pending, end, ring);
        return status_of(cudaGetLastError());
    }
    return CERE_OK;
}

cere_status indexer_qk_mrope(const bf16* qk, int64_t qk_stride, const bf16* q_norm_weight, float eps,
                             const float* cos_sin, const Bounds& bounds, const int64_t* positions3, int64_t p3_stride,
                             int64_t p3_lead, int64_t tokens, bf16* q_idx, cudaStream_t stream) {
    if (!qk || !q_norm_weight || !cos_sin || !positions3 || !q_idx || tokens < 0 || qk_stride < kQkWidth ||
        !(eps > 0.0f) || tokens > INT32_MAX || p3_lead < 0 || p3_lead >= kRatio || p3_stride < p3_lead + tokens ||
        !rope_ok(bounds))
        return CERE_ERR_ARGUMENT;
    if (tokens == 0) return CERE_OK;
    index_q_mrope_kernel<<<static_cast<unsigned>(tokens), kIndexHeads * kWarp, 0, stream>>>(
        qk, qk_stride, q_norm_weight, eps, cos_sin, bounds, positions3, p3_stride, p3_lead, q_idx);
    return status_of(cudaGetLastError());
}

cere_status qsa_compress_store_mrope(const bf16* k_idx, int64_t k_stride, const bf16* k_norm_weight, float eps,
                                     const float* cos_sin, const Bounds& bounds, const int64_t* positions3,
                                     int64_t p3_stride, int64_t p3_lead, const int64_t* slots, int64_t pos0,
                                     int64_t tokens, bf16* ring, bf16* keys, cudaStream_t stream) {
    if (!k_idx || !k_norm_weight || !cos_sin || !positions3 || !ring || !keys || tokens < 0 || pos0 < 0 ||
        k_stride < kIndexHeadDim || !(eps > 0.0f) || p3_lead != pos0 % kRatio || p3_stride < p3_lead + tokens ||
        !rope_ok(bounds))
        return CERE_ERR_ARGUMENT;
    if (tokens == 0) return CERE_OK;
    if (!context_ok(bounds, pos0, tokens)) return CERE_ERR_ARGUMENT;
    const int64_t end = pos0 + tokens;
    const int64_t g_lo = pos0 / kRatio;
    const int64_t g_hi = end / kRatio;
    if (g_hi > g_lo) {
        const int64_t groups = g_hi - g_lo;
        compress_mrope_kernel<<<static_cast<unsigned>((groups + 3) / 4), 4 * kWarp, 0, stream>>>(
            k_idx, k_stride, k_norm_weight, eps, cos_sin, bounds, positions3, p3_stride, p3_lead, slots, pos0, g_lo,
            groups, ring, keys);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return status_of(err);
    }
    // After the compress kernel (stream order): it may still read ring rows this overwrites.
    const int64_t pending = g_hi * kRatio > pos0 ? g_hi * kRatio : pos0;
    if (pending < end) {
        ring_kernel<<<1, kIndexHeadDim, 0, stream>>>(k_idx, k_stride, pos0, pending, end, ring);
        return status_of(cudaGetLastError());
    }
    return CERE_OK;
}

cere_status qsa_compress_store_step(const bf16* k_idx, const bf16* k_norm_weight, float eps, const float* cos_sin,
                                    const Bounds& bounds, const int64_t* slots, const int64_t* position, bf16* ring,
                                    bf16* keys, cudaStream_t stream) {
    if (!k_idx || !k_norm_weight || !cos_sin || !position || !ring || !keys || !(eps > 0.0f) || !rope_ok(bounds) ||
        bounds.context < 1)
        return CERE_ERR_ARGUMENT;
    compress_step_kernel<<<1, kIndexHeadDim, 0, stream>>>(k_idx, k_norm_weight, eps, cos_sin, bounds, slots, position,
                                                          ring, keys);
    return status_of(cudaGetLastError());
}

cere_status qsa_compress_store_at(const bf16* k_idx, int64_t k_stride, const bf16* k_norm_weight, float eps,
                                  const float* cos_sin, const Bounds& bounds, const int64_t* slots,
                                  const int64_t* position, int64_t tokens, bf16* ring, bf16* keys,
                                  cudaStream_t stream) {
    if (!k_idx || !k_norm_weight || !cos_sin || !position || !ring || !keys || tokens < 1 || tokens > INT32_MAX ||
        k_stride < kIndexHeadDim || !(eps > 0.0f) || !rope_ok(bounds) || bounds.context < tokens)
        return CERE_ERR_ARGUMENT;
    const int64_t groups = (kRatio - 1 + tokens) / kRatio;  // the most groups `tokens` tokens complete
    compress_at_kernel<<<static_cast<unsigned>((groups + 3) / 4), 4 * kWarp, 0, stream>>>(
        k_idx, k_stride, k_norm_weight, eps, cos_sin, bounds, slots, position, tokens, ring, keys);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return status_of(err);
    ring_at_kernel<<<1, kIndexHeadDim, 0, stream>>>(k_idx, k_stride, bounds, position, tokens, ring);
    return status_of(cudaGetLastError());
}

cere_status qsa_key_save_step(const bf16* keys, const int64_t* slots, const Bounds& bounds, const int64_t* position,
                              int64_t j, int64_t n, bf16* saved, cudaStream_t stream) {
    if (!keys || !slots || !position || !saved || j < 1 || n < j || n > 4 * kRatio || bounds.context < 1)
        return CERE_ERR_ARGUMENT;
    key_save_kernel<<<1, kIndexHeadDim, 0, stream>>>(keys, slots, bounds, position, j, n, saved);
    return status_of(cudaGetLastError());
}

cere_status qsa_chains_step(const ChainsStep& rows, int chains, const bf16* k_norm_weight, float eps,
                            const float* cos_sin, const Bounds& bounds, const int64_t* positions,
                            cudaStream_t stream) {
    if (!positions || chains < 1 || chains > kChainsStepRows) return CERE_ERR_ARGUMENT;
    int next = 0;
    for (int c = 0; c < chains; ++c) {
        const int first = rows.chain_first[c];
        const int count = rows.chain_rows[c];
        if (first != next || count < 1 || count > kChainsStepRows - first) return CERE_ERR_ARGUMENT;
        next = first + count;
        for (int r = first; r < next; ++r) {
            if (rows.context[r] < 1) return CERE_ERR_ARGUMENT;
            if ((rows.undo_ring[r] || rows.k_idx[r]) && !rows.ring[r]) return CERE_ERR_ARGUMENT;
            if (rows.undo_key[r] && (!rows.undo_ring[r] || !rows.keys[r] || !rows.slots[r] || rows.j[r] < 1 ||
                                     rows.n[r] < rows.j[r] || rows.n[r] > 4 * kRatio))
                return CERE_ERR_ARGUMENT;
            if (rows.k_idx[r] && (!k_norm_weight || !cos_sin || !rows.keys[r] || !(eps > 0.0f) || !rope_ok(bounds)))
                return CERE_ERR_ARGUMENT;
        }
    }
    chains_step_kernel<<<static_cast<unsigned>(chains), kIndexHeadDim, 0, stream>>>(rows, k_norm_weight, eps, cos_sin,
                                                                                   bounds, positions);
    return status_of(cudaGetLastError());
}

cere_status qsa_logits(const bf16* q_idx, const bf16* keys, const int64_t* slots, const int64_t* positions_logical,
                       int64_t rows, int64_t seq_len, float* logits, int64_t logits_ld, int impl,
                       cudaStream_t stream) {
    if (!q_idx || !keys || !logits || rows < 0 || seq_len <= 0 || logits_ld < seq_len / kRatio ||
        (!positions_logical && rows > seq_len) || !aligned(q_idx, 16) || !aligned(keys, 16) ||
        rows > INT32_MAX / kLogitRows || impl < 0 || impl > 2)
        return CERE_ERR_ARGUMENT;
    const LogitsKernel kernel = impl == 1   ? LogitsKernel::block
                                : impl == 2 ? LogitsKernel::small
                                : rows <= kSplitMaxRows ? LogitsKernel::small
                                                        : LogitsKernel::prefill;
    return status_of(launch_logits(q_idx, keys, slots, positions_logical, 0, rows, rows, seq_len, nullptr, logits,
                                   logits_ld, false, kernel, stream));
}

cere_status qsa_select(const bf16* q_idx, const bf16* keys, const int64_t* slots, const int64_t* positions_logical,
                       int64_t rows, int64_t seq_len, void* scratch, size_t scratch_bytes, int32_t* out_indices,
                       cudaStream_t stream) {
    if (!q_idx || !keys || !out_indices || rows < 0 || seq_len <= 0 || (!positions_logical && rows > seq_len) ||
        !aligned(q_idx, 16) || !aligned(keys, 16) || rows > INT32_MAX)
        return CERE_ERR_ARGUMENT;
    if (rows == 0) return CERE_OK;
    const int64_t ld = seq_len / kRatio;
    int64_t tile = rows;
    if (ld > 0) {
        if (!scratch || !aligned(scratch, 16)) return CERE_ERR_ARGUMENT;
        tile = logits_tile_rows(rows, seq_len);
        const int64_t fit = static_cast<int64_t>(scratch_bytes / (static_cast<size_t>(ld) * sizeof(float)));
        if (fit < tile) tile = fit >= kLogitRows ? fit / kLogitRows * kLogitRows : fit;
        if (tile <= 0) return CERE_ERR_ARGUMENT;
    }
    float* logits = static_cast<float*>(scratch);
    // Calls of up to kSplitMaxRows rows (decode, verify, the MTP draft) score on logits_small_kernel (LK9-02) and
    // select on clusters (LK9-01); prefill tiles score on logits_prefill_kernel and select on topk_prefill_kernel
    // (topk_kernel past kTpMaxBlocks); every choice writes logits_kernel's and topk_kernel's bytes.
    const LogitsKernel scorer = rows <= kSplitMaxRows ? LogitsKernel::small : LogitsKernel::prefill;
    const TopkKernel selector = topk_choice(rows, ld);
    for (int64_t row0 = 0; row0 < rows; row0 += tile) {
        const int64_t n = rows - row0 < tile ? rows - row0 : tile;
        cudaError_t err = launch_logits(q_idx, keys, slots, positions_logical, row0, n, rows, seq_len, nullptr,
                                        logits, ld, false, scorer, stream);
        if (err != cudaSuccess) return status_of(err);
        err = launch_topk(logits, ld, positions_logical, row0, n, rows, seq_len, nullptr, out_indices, selector,
                          stream);
        if (err != cudaSuccess) return status_of(err);
    }
    return CERE_OK;
}

cere_status qsa_select_step(const bf16* q_idx, const bf16* keys, const int64_t* slots, const Bounds& bounds,
                            const int64_t* position, int64_t ctx, void* scratch, size_t scratch_bytes,
                            int32_t* out_indices, cudaStream_t stream) {
    const int64_t ld = ctx / kRatio;
    if (!q_idx || !keys || !position || !out_indices || ld <= 0 || !scratch || !aligned(scratch, 16) ||
        scratch_bytes < static_cast<size_t>(ld) * sizeof(float) || !aligned(q_idx, 16) || !aligned(keys, 16) ||
        bounds.context < 1 || bounds.context > ctx)
        return CERE_ERR_ARGUMENT;
    float* logits = static_cast<float*>(scratch);
    // The row's position, clamped to the session's context (<= ctx), keeps (p + 1) / 4 <= ld.
    cudaError_t err = launch_logits(q_idx, keys, slots, position, 0, 1, 1, bounds.context, bounds.fault, logits, ld,
                                    true, LogitsKernel::small, stream);
    if (err != cudaSuccess) return status_of(err);
    return status_of(launch_topk(logits, ld, position, 0, 1, 1, bounds.context, bounds.fault, out_indices,
                                 topk_choice(1, ld), stream));
}

cere_status qsa_topk(const float* logits, int64_t ld, const int64_t* positions_logical, int64_t rows,
                     int64_t seq_len, int impl, int32_t* out_indices, cudaStream_t stream) {
    if (!logits || !out_indices || rows <= 0 || rows > 65535 || seq_len <= 0 || ld < seq_len / kRatio ||
        (!positions_logical && rows > seq_len) || impl < 0 || impl > 2 || (impl == 2 && ld > kTkMaxBlocks))
        return CERE_ERR_ARGUMENT;
    const TopkKernel kernel = impl == 1 ? TopkKernel::block : impl == 2 ? TopkKernel::cluster : topk_choice(rows, ld);
    return status_of(
        launch_topk(logits, ld, positions_logical, 0, rows, rows, seq_len, nullptr, out_indices, kernel, stream));
}

cere_status qsa_attention(const bf16* q, int64_t q_stride, const int32_t* indices, int64_t rows, const bf16* k,
                          int64_t k_stride, const bf16* v, int64_t v_stride, const uint8_t* k_cache,
                          const uint8_t* v_cache, const int64_t* slots, float scale, void* scratch,
                          size_t scratch_bytes, bf16* out, int64_t out_stride, cudaStream_t stream, bool use_p4,
                          int64_t cls) {
    return qsa_attention_heads(q, q_stride, indices, rows, k, k_stride, v, v_stride, k_cache, v_cache, slots, scale,
                               scratch, scratch_bytes, out, out_stride, stream, use_p4, cls, kRowsHeads);
}

cere_status qsa_attention_heads(const bf16* q, int64_t q_stride, const int32_t* indices, int64_t rows, const bf16* k,
                                int64_t k_stride, const bf16* v, int64_t v_stride, const uint8_t* k_cache,
                                const uint8_t* v_cache, const int64_t* slots, float scale, void* scratch,
                                size_t scratch_bytes, bf16* out, int64_t out_stride, cudaStream_t stream, bool use_p4,
                                int64_t cls, int rows_heads) {
    if (rows_heads != 0 && rows_heads != 3 && rows_heads != 6 && rows_heads != 12) return CERE_ERR_ARGUMENT;
    const bool bf16_kv = k && v && !k_cache && !v_cache;
    const bool fp8_kv = !k && !v && k_cache && v_cache;
    if (!q || !indices || !out || rows < 0 || rows > INT32_MAX || !(bf16_kv || fp8_kv) || q_stride < kQWidth ||
        out_stride < kQWidth || !std::isfinite(scale))
        return CERE_ERR_ARGUMENT;
    if (bf16_kv && (k_stride < kKvWidth || v_stride < kKvWidth || k_stride % 8 || v_stride % 8 || !aligned(k, 16) ||
                    !aligned(v, 16)))
        return CERE_ERR_ARGUMENT;
    if (fp8_kv && (!aligned(k_cache, 16) || !aligned(v_cache, 16))) return CERE_ERR_ARGUMENT;
    if (rows == 0) return CERE_OK;
    if (cls != 0 && (cls < rows || cls > INT32_MAX)) return CERE_ERR_ARGUMENT;
    // The rows the kernel choice below is made for (m210 D15⁗′): the call's unless it stands for a longer one.
    const int64_t sel = cls != 0 ? cls : rows;
    // P4: calls of more than 16 rows run the tensor-core sparse GQA; so does a pinned call of fewer rows whose class
    // chose it (its rows are bit for bit those of the cold call: one block per query token, p4_attention.h).
    if (use_p4 && sel >= p4::kMinRows && q_stride % 8 == 0 && out_stride % 8 == 0 && aligned(q, 16) &&
        aligned(out, 16))
        return status_of(p4::sparse_attention(q, q_stride, indices, kRowWidth, rows, k, k_stride, v, v_stride, k_cache,
                                              v_cache, slots, scale, out, out_stride, stream));
    const int splits = sel <= kSplitMaxRows ? kSplits : 1;
    float* part_o = nullptr;
    float* part_m = nullptr;
    float* part_l = nullptr;
    if (splits > 1) {
        if (!scratch || !aligned(scratch, 256) || scratch_bytes < attention_scratch_bytes(rows)) return CERE_ERR_ARGUMENT;
        const size_t pr = partial_rows(rows);
        unsigned char* p = static_cast<unsigned char*>(scratch);
        part_o = reinterpret_cast<float*>(p);
        p += align256(pr * kHeadDim * sizeof(float));
        part_m = reinterpret_cast<float*>(p);
        p += align256(pr * sizeof(float));
        part_l = reinterpret_cast<float*>(p);
    }
    if (splits == 1 && rows_heads != 0 && q_stride % 8 == 0 && out_stride % 8 == 0 && aligned(q, 16) &&
        aligned(out, 16)) {
        const void* ks = fp8_kv ? static_cast<const void*>(k_cache) : k;
        const void* vs = fp8_kv ? static_cast<const void*>(v_cache) : v;
        const int64_t kst = fp8_kv ? kKvWidth : k_stride;
        const int64_t vst = fp8_kv ? kKvWidth : v_stride;
        if (rows_heads == 3) {
            launch_rows<3>(q, q_stride, indices, rows, ks, kst, vs, vst, slots, fp8_kv, scale, out, out_stride, stream);
        } else if (rows_heads == 6) {
            launch_rows<6>(q, q_stride, indices, rows, ks, kst, vs, vst, slots, fp8_kv, scale, out, out_stride, stream);
        } else {
            launch_rows<12>(q, q_stride, indices, rows, ks, kst, vs, vst, slots, fp8_kv, scale, out, out_stride, stream);
        }
        return status_of(cudaGetLastError());
    }
    cudaError_t err = cudaSuccess;
    if (splits > 1 && fp8_kv && rows_heads != 0 && q_stride % 8 == 0 && aligned(q, 16)) {
        // Decode, verify and the MTP draft on the FP8 cache (track LongK9 LK9-02): the same partials.
        err = launch_split(q, q_stride, indices, rows, k_cache, v_cache, slots, scale, part_o, part_m, part_l, stream);
    } else {
        const dim3 grid(static_cast<unsigned>(rows), kKvHeads, static_cast<unsigned>(splits));
        if (fp8_kv) {
            attention_kernel<true><<<grid, kAttnThreads, 0, stream>>>(q, q_stride, indices, k_cache, kKvWidth,
                                                                      v_cache, kKvWidth, slots, scale, splits, rows,
                                                                      out, out_stride, part_o, part_m, part_l);
        } else {
            attention_kernel<false><<<grid, kAttnThreads, 0, stream>>>(q, q_stride, indices, k, k_stride, v,
                                                                       v_stride, nullptr, scale, splits, rows, out,
                                                                       out_stride, part_o, part_m, part_l);
        }
        err = cudaGetLastError();
    }
    if (err != cudaSuccess || splits == 1) return status_of(err);
    combine_kernel<<<dim3(static_cast<unsigned>(rows), kQHeads), kHeadDim, 0, stream>>>(splits, rows, part_o, part_m,
                                                                                      part_l, out, out_stride);
    return status_of(cudaGetLastError());
}

}  // namespace cere::k9
