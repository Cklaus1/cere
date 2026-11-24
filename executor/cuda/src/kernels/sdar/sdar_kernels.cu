// SelfTune ST4 S3c: the kernels of family sdar (DICE-8B); contracts in sdar_kernels.h, the kernel tests' mutants in
// sdar_mutants.h. Each kernel notes the rounding points of the reference forward it follows (modeling_sdar.py at the
// S2 pin: flash_attn's triton rms_norm_fn, the HF rotary embedding with cos/sin cast to BF16, the MLP's silu path
// with liger unavailable, scaled_dot_product_attention with the additive block-causal mask). Correctness first: the
// attention is one CTA per (32 query rows, query head) with an online FP32 softmax over 32-key shared tiles, no
// tensor cores (tuning is S4).
//
// math_reference: DICE-8B
#include "kernels/sdar/sdar_kernels.h"
#include "kernels/sdar/sdar_mutants.h"

#include <cmath>
#include <cstdint>

namespace cere::sdar {
namespace {

constexpr int kThreads = 128;
constexpr int64_t kMaxGridX = 2147483647;
constexpr int64_t kMaxGridY = 65535;
constexpr int kHalf = kHeadDim / 2;
constexpr uint32_t kNone = 0xFFFFFFFFu;

__device__ __forceinline__ float bf2f(bf16 x) { return __bfloat162float(x); }

// torch's float -> bfloat16 cast: round to nearest even.
__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    return v;
}

bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// count BF16 elements src -> dst by the threads of one block, bit for bit. kVec: both 16-byte aligned, count % 8 == 0.
template <bool kVec>
__device__ __forceinline__ void copy_row(const bf16 *src, bf16 *dst, int64_t count) {
    if constexpr (kVec) {
        const uint4 *s = reinterpret_cast<const uint4 *>(src);
        uint4 *d = reinterpret_cast<uint4 *>(dst);
        for (int64_t i = threadIdx.x; i < count / 8; i += blockDim.x) d[i] = s[i];
    } else {
        const uint16_t *s = reinterpret_cast<const uint16_t *>(src);
        uint16_t *d = reinterpret_cast<uint16_t *>(dst);
        for (int64_t i = threadIdx.x; i < count; i += blockDim.x) d[i] = s[i];
    }
}

// ---- embed ---------------------------------------------------------------------------------------------------------

template <bool kVec>
__global__ void __launch_bounds__(kThreads) embed_kernel(const bf16 *__restrict__ table,
                                                         const uint32_t *__restrict__ ids, int hidden,
                                                         bf16 *__restrict__ out) {
    const int64_t t = blockIdx.x;
    copy_row<kVec>(table + int64_t(ids[t]) * hidden, out + t * hidden, hidden);
}

// ---- RMSNorm -------------------------------------------------------------------------------------------------------

// Up to this row width one warp normalizes a row (q_norm / k_norm: 128), above it one block (the 4,096-wide norms).
constexpr int kWarpRowMaxDim = 1024;
constexpr int kNormBlock = 256;

// flash_attn.ops.triton.layer_norm rms_norm_fn (SDARRMSNorm.forward): x and w loaded as FP32, var = sum(x * x) / N,
// rstd = 1 / sqrt(var + eps), y = (x * rstd) * w stored once in the input dtype (BF16). The triton kernel's sqrt and
// division may be the approximate PTX forms [INFERENCE]; here both are IEEE (sqrtf, /), so rstd can differ by an ulp.
// x and y may alias (same stride): every element is read and written by one thread, after the reduction.
__device__ __forceinline__ float norm_rstd(float sum_sq, int dim, float eps) {
    return 1.0f / sqrtf(sum_sq / float(dim) + eps);
}

__global__ void __launch_bounds__(kThreads) rmsnorm_warp_kernel(const bf16 *x, int64_t x_stride,
                                                                const bf16 *__restrict__ w, float eps, int64_t rows,
                                                                int dim, bf16 *y, int64_t y_stride) {
    const int64_t row = int64_t(blockIdx.x) * (kThreads / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x % 32;
    const bf16 *xr = x + row * x_stride;
    bf16 *yr = y + row * y_stride;
    float ss = 0.0f;
    for (int i = lane; i < dim; i += 32) {
        const float v = bf2f(xr[i]);
        ss = fmaf(v, v, ss);
    }
    const float rstd = norm_rstd(warp_sum(ss), dim, eps);
    for (int i = lane; i < dim; i += 32) yr[i] = __float2bfloat16_rn(bf2f(xr[i]) * rstd * bf2f(w[i]));
}

__global__ void __launch_bounds__(kNormBlock) rmsnorm_block_kernel(const bf16 *x, int64_t x_stride,
                                                                  const bf16 *__restrict__ w, float eps, int dim,
                                                                  bf16 *y, int64_t y_stride) {
    __shared__ float partial[kNormBlock / 32];
    const int64_t row = blockIdx.x;
    const bf16 *xr = x + row * x_stride;
    bf16 *yr = y + row * y_stride;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += kNormBlock) {
        const float v = bf2f(xr[i]);
        ss = fmaf(v, v, ss);
    }
    ss = warp_sum(ss);
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = ss;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < kNormBlock / 32; ++i) total += partial[i];
    const float rstd = norm_rstd(total, dim, eps);
    for (int i = threadIdx.x; i < dim; i += kNormBlock) yr[i] = __float2bfloat16_rn(bf2f(xr[i]) * rstd * bf2f(w[i]));
}

// ---- RoPE ----------------------------------------------------------------------------------------------------------

// One block per row: the 64 frequencies' cos/sin once, then every head's 64 rotation pairs.
__global__ void __launch_bounds__(kThreads) rope_kernel(bf16 *x, int64_t stride, int heads, int64_t pos0,
                                                        float theta) {
    __shared__ float cos_s[kHalf];
    __shared__ float sin_s[kHalf];
    const int64_t row = blockIdx.x;
    if (threadIdx.x < kHalf) {
        const int i = threadIdx.x;
        // HF _compute_default_rope_parameters: inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim)), on the
        // CPU at load in FP32: the power rounded to FP32 (here the double pow rounded once, i.e. a correctly rounded
        // powf; torch's CPU pow is assumed correctly rounded on these 64 values [INFERENCE]), then the FP32
        // reciprocal rounded again. The exponent 2i/128 is exact.
        const float base_pow = float(pow(double(theta), double(2 * i) / double(kHeadDim)));
        const float inv_freq = 1.0f / base_pow;
        // SDARRotaryEmbedding.forward: freqs = inv_freq @ position_ids.float() (K = 1: one FP32 product), emb =
        // cat(freqs, freqs), cos/sin in FP32 (attention_scaling 1.0 for rope_type default), cast to BF16.
        const float angle = inv_freq * float(pos0 + row);
        cos_s[i] = round_bf16(cosf(angle));
        sin_s[i] = round_bf16(sinf(angle));
    }
    __syncthreads();
    bf16 *base = x + row * stride;
    for (int idx = threadIdx.x; idx < heads * kHalf; idx += blockDim.x) {
        bf16 *p = base + int64_t(idx / kHalf) * kHeadDim;
        const int i = idx % kHalf;
        const float x1 = bf2f(p[i]);
        const float x2 = bf2f(p[i + kHalf]);
        const float c = cos_s[i];
        const float s = sin_s[i];
        // apply_rotary_pos_emb on BF16 tensors: q * cos + rotate_half(q) * sin with rotate_half(q) = cat(-x2, x1);
        // torch rounds each product and the sum to BF16 (each binary op computes in FP32 and stores BF16).
        const float lo = round_bf16(x1 * c) + round_bf16(-x2 * s);
        const float hi = round_bf16(x2 * c) + round_bf16(x1 * s);
        p[i] = __float2bfloat16_rn(lo);
        p[i + kHalf] = __float2bfloat16_rn(hi);
    }
}

// ---- KV store ------------------------------------------------------------------------------------------------------

template <bool kVec>
__global__ void __launch_bounds__(kThreads) kv_store_kernel(const bf16 *__restrict__ k, int64_t k_stride,
                                                            const bf16 *__restrict__ v, int64_t v_stride,
                                                            int64_t row_elems, bf16 *__restrict__ k_cache,
                                                            bf16 *__restrict__ v_cache, int64_t pos) {
    const int64_t t = blockIdx.x;
    copy_row<kVec>(k + t * k_stride, k_cache + (pos + t) * row_elems, row_elems);
    copy_row<kVec>(v + t * v_stride, v_cache + (pos + t) * row_elems, row_elems);
}

// ---- attention -----------------------------------------------------------------------------------------------------

constexpr int kAttnWarps = 4;
constexpr int kAttnRowsPerWarp = 8;
constexpr int kAttnRows = kAttnWarps * kAttnRowsPerWarp;  // query rows per CTA (one query head)
constexpr int kAttnKeys = 32;                             // keys per shared tile: one per lane in the score pass
constexpr int kKPitch = kHeadDim + 2;  // K tile row pitch: 65 words, so the 32 lanes' rows hit 32 banks

struct AttnArgs {
    const bf16 *q;
    int64_t q_stride;
    const bf16 *k_cache;
    const bf16 *v_cache;
    int64_t past;
    const bf16 *new_k;
    int64_t new_k_stride;
    const bf16 *new_v;
    int64_t new_v_stride;
    int64_t n;
    int group;       // query heads per KV head
    int64_t kv_row;  // cache row elements (kv_heads * kHeadDim)
    bf16 *out;
    int64_t out_stride;
    int mask;
    float scale;
};

// Keys [0, limit) are visible to the query at absolute position p; no key exists at or past end = past + n.
__device__ __forceinline__ int64_t key_limit(int64_t p, int64_t end, int mask) {
    const int64_t limit = mask == int(Mask::kBlockCausal) ? (p / kBlock + 1) * kBlock : p + 1;
    return limit < end ? limit : end;
}

__device__ __forceinline__ uint4 load8_raw(const bf16 *p) {
    const uint16_t *s = reinterpret_cast<const uint16_t *>(p);
    return make_uint4(uint32_t(s[0]) | (uint32_t(s[1]) << 16), uint32_t(s[2]) | (uint32_t(s[3]) << 16),
                      uint32_t(s[4]) | (uint32_t(s[5]) << 16), uint32_t(s[6]) | (uint32_t(s[7]) << 16));
}

// K and V of KV head g for keys [j0, j0 + kAttnKeys) into shared memory, zeros at or past kend. Keys below past come
// from the cache rows, the others from new_k / new_v only (so the cache tail is never read).
template <bool kVec>
__device__ __forceinline__ void load_kv_tile(const AttnArgs &a, int g, int64_t j0, int64_t kend,
                                             bf16 (*sk)[kKPitch], bf16 (*sv)[kHeadDim]) {
    constexpr int kChunks = kHeadDim / 8;  // 16-byte chunks per row
    for (int c = threadIdx.x; c < kAttnKeys * kChunks; c += blockDim.x) {
        const int r = c / kChunks;
        const int d = (c % kChunks) * 8;
        const int64_t j = j0 + r;
        uint4 kw = make_uint4(0, 0, 0, 0);
        uint4 vw = kw;
        if (j < kend) {
            const bf16 *kp;
            const bf16 *vp;
            if (j < a.past) {
                kp = a.k_cache + j * a.kv_row + int64_t(g) * kHeadDim + d;
                vp = a.v_cache + j * a.kv_row + int64_t(g) * kHeadDim + d;
            } else {
                kp = a.new_k + (j - a.past) * a.new_k_stride + int64_t(g) * kHeadDim + d;
                vp = a.new_v + (j - a.past) * a.new_v_stride + int64_t(g) * kHeadDim + d;
            }
            if constexpr (kVec) {
                kw = *reinterpret_cast<const uint4 *>(kp);
                vw = *reinterpret_cast<const uint4 *>(vp);
            } else {
                kw = load8_raw(kp);
                vw = load8_raw(vp);
            }
        }
        uint32_t *kd = reinterpret_cast<uint32_t *>(&sk[r][d]);  // 4-byte aligned (pitch 260 B, d even)
        kd[0] = kw.x;
        kd[1] = kw.y;
        kd[2] = kw.z;
        kd[3] = kw.w;
        *reinterpret_cast<uint4 *>(&sv[r][d]) = vw;  // 16-byte aligned (pitch 256 B)
    }
}

// The reference (SDARAttention.forward, q_len > 1 or no cache) runs F.scaled_dot_product_attention on BF16 q/k/v
// with an additive BF16 mask (0 / finfo.min): scores and softmax in FP32 inside the backend, output rounded to BF16
// once. Here: FP32 dot products of the BF16 inputs, scaled by 1/sqrt(128), masked keys excluded (exp of finfo.min
// minus the row max underflows to 0 in FP32, so excluding them is the same softmax), an online FP32 softmax over the
// key tiles and FP32 accumulation of P V, then out = acc / sum rounded to BF16 once. The summation order is not the
// backend's, so this matches to rounding, not bit for bit.
template <bool kVec>
__global__ void __launch_bounds__(kAttnWarps * 32) attention_kernel(const AttnArgs a) {
    __shared__ float sq[kAttnRows][kHeadDim];
    __shared__ __align__(16) bf16 sk[kAttnKeys][kKPitch];
    __shared__ __align__(16) bf16 sv[kAttnKeys][kHeadDim];
    __shared__ float sp[kAttnWarps][kAttnRowsPerWarp][kAttnKeys];

    const int h = blockIdx.y;
    const int g = h / a.group;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int64_t t0 = int64_t(blockIdx.x) * kAttnRows;
    const int64_t end = a.past + a.n;

    for (int c = threadIdx.x; c < kAttnRows * kHeadDim; c += blockDim.x) {
        const int r = c / kHeadDim;
        const int d = c % kHeadDim;
        const int64_t t = t0 + r;
        sq[r][d] = t < a.n ? bf2f(a.q[t * a.q_stride + int64_t(h) * kHeadDim + d]) : 0.0f;
    }

    int64_t limit[kAttnRowsPerWarp];
    int64_t warp_end = 0;
#pragma unroll
    for (int r = 0; r < kAttnRowsPerWarp; ++r) {
        const int64_t t = t0 + warp * kAttnRowsPerWarp + r;
        limit[r] = t < a.n ? key_limit(a.past + t, end, a.mask) : 0;
        warp_end = limit[r] > warp_end ? limit[r] : warp_end;
    }
    // The CTA's last row sees the most keys (limits grow with the position).
    const int64_t t_last = (t0 + kAttnRows < a.n ? t0 + kAttnRows : a.n) - 1;
    const int64_t kend = key_limit(a.past + t_last, end, a.mask);

    float m[kAttnRowsPerWarp];
    float lsum[kAttnRowsPerWarp];  // this lane's share of the softmax denominator
    float o[kAttnRowsPerWarp][4];  // dims 4 * lane .. 4 * lane + 3
#pragma unroll
    for (int r = 0; r < kAttnRowsPerWarp; ++r) {
        m[r] = -INFINITY;
        lsum[r] = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; ++e) o[r][e] = 0.0f;
    }

    for (int64_t j0 = 0; j0 < kend; j0 += kAttnKeys) {
        __syncthreads();  // sq written (first tile); the previous tile's readers done
        load_kv_tile<kVec>(a, g, j0, kend, sk, sv);
        __syncthreads();
        if (j0 >= warp_end) continue;  // warp-uniform: no row of this warp sees these keys

        const int64_t j = j0 + lane;
        float s[kAttnRowsPerWarp];
#pragma unroll
        for (int r = 0; r < kAttnRowsPerWarp; ++r) s[r] = 0.0f;
        const __nv_bfloat162 *krow = reinterpret_cast<const __nv_bfloat162 *>(&sk[lane][0]);
#pragma unroll 4
        for (int w = 0; w < kHalf; ++w) {
            const float2 kf = __bfloat1622float2(krow[w]);
#pragma unroll
            for (int r = 0; r < kAttnRowsPerWarp; ++r) {
                const float2 qf = *reinterpret_cast<const float2 *>(&sq[warp * kAttnRowsPerWarp + r][2 * w]);
                s[r] = fmaf(qf.x, kf.x, s[r]);
                s[r] = fmaf(qf.y, kf.y, s[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < kAttnRowsPerWarp; ++r) {
            const float x = j < limit[r] ? s[r] * a.scale : -INFINITY;
            const float m_new = fmaxf(m[r], warp_max(x));
            float p = 0.0f;
            float corr = 1.0f;
            if (m_new != -INFINITY) {  // a row with no visible key so far keeps m = -inf, p = 0
                p = expf(x - m_new);
                corr = expf(m[r] - m_new);
            }
            m[r] = m_new;
            lsum[r] = lsum[r] * corr + p;
#pragma unroll
            for (int e = 0; e < 4; ++e) o[r][e] *= corr;
            sp[warp][r][lane] = p;
        }
        __syncwarp();
#pragma unroll 4
        for (int jj = 0; jj < kAttnKeys; ++jj) {
            const uint2 raw = *reinterpret_cast<const uint2 *>(&sv[jj][4 * lane]);
            const float2 v01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&raw.x));
            const float2 v23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&raw.y));
#pragma unroll
            for (int r = 0; r < kAttnRowsPerWarp; ++r) {
                const float p = sp[warp][r][jj];
                o[r][0] = fmaf(p, v01.x, o[r][0]);
                o[r][1] = fmaf(p, v01.y, o[r][1]);
                o[r][2] = fmaf(p, v23.x, o[r][2]);
                o[r][3] = fmaf(p, v23.y, o[r][3]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < kAttnRowsPerWarp; ++r) {
        const float l = warp_sum(lsum[r]);
        const int64_t t = t0 + warp * kAttnRowsPerWarp + r;
        if (t < a.n) {
            bf16 *dst = a.out + t * a.out_stride + int64_t(h) * kHeadDim + 4 * lane;
#pragma unroll
            for (int e = 0; e < 4; ++e) dst[e] = __float2bfloat16_rn(o[r][e] / l);
        }
    }
}

// ---- SiLU * up -----------------------------------------------------------------------------------------------------

// SDARMLP.forward without liger: act_fn(gate_proj(x)) * up_proj(x) on BF16 tensors. ACT2FN["silu"] is nn.SiLU, whose
// CUDA kernel computes x / (1 + exp(-x)) in FP32 and stores BF16; the product of the two BF16 tensors is one FP32
// multiply rounded to BF16.
__global__ void __launch_bounds__(256) silu_mul_kernel(const bf16 *__restrict__ gate_up, int64_t rows, int inter,
                                                       bf16 *__restrict__ out) {
    const int64_t total = rows * inter;
    for (int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += int64_t(gridDim.x) * blockDim.x) {
        const int64_t t = idx / inter;
        const int64_t i = idx - t * inter;
        const bf16 *row = gate_up + t * 2 * inter;
        const float gate = bf2f(row[i]);
        const float up = bf2f(row[inter + i]);
        const float act = round_bf16(gate / (1.0f + expf(-gate)));
        out[idx] = __float2bfloat16_rn(act * up);
    }
}

// ---- residual add --------------------------------------------------------------------------------------------------

// torch's BF16 add (`residual + hidden_states`): both operands widened to FP32, one FP32 add, one RNE cast to BF16.
__global__ void __launch_bounds__(256) add_kernel(bf16 *x, const bf16 *y, int64_t n) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        x[i] = __float2bfloat16_rn(bf2f(x[i]) + bf2f(y[i]));
    }
}

// ---- top_tie -------------------------------------------------------------------------------------------------------

// A partial reduction of one row: the maximum m1, the lowest id holding it (i1), how many ids hold it (c1, 0 = empty
// state), the second-lowest id holding it (i1b, kNone if none) and the best value below m1 (m2) with its lowest id
// (i2, kNone if none). Merging is exact, so the result does not depend on the reduction order.
struct TieState {
    float m1;
    uint32_t i1;
    uint32_t c1;
    uint32_t i1b;
    float m2;
    uint32_t i2;
};

// (m, i) beats (m', i') when m > m', or m == m' with the lower id (kNone loses every equal value).
__device__ __forceinline__ bool beats(float m, uint32_t i, float m_other, uint32_t i_other) {
    return m > m_other || (m == m_other && i < i_other);
}

__device__ __forceinline__ TieState merge(const TieState &a, const TieState &b) {
    if (b.c1 == 0) return a;
    if (a.c1 == 0) return b;
    TieState r;
    if (a.m1 == b.m1) {
        r.m1 = a.m1;
        r.c1 = a.c1 + b.c1;
        if (a.i1 < b.i1) {
            r.i1 = a.i1;
            r.i1b = a.i1b < b.i1 ? a.i1b : b.i1;
        } else {
            r.i1 = b.i1;
            r.i1b = b.i1b < a.i1 ? b.i1b : a.i1;
        }
        const bool a_wins = beats(a.m2, a.i2, b.m2, b.i2);
        r.m2 = a_wins ? a.m2 : b.m2;
        r.i2 = a_wins ? a.i2 : b.i2;
    } else {
        const TieState &hi = a.m1 > b.m1 ? a : b;
        const TieState &lo = a.m1 > b.m1 ? b : a;
        r = hi;
        if (beats(lo.m1, lo.i1, hi.m2, hi.i2)) {
            r.m2 = lo.m1;
            r.i2 = lo.i1;
        }
    }
    return r;
}

__device__ __forceinline__ TieState shfl_state(const TieState &s, int lane_mask) {
    TieState r;
    r.m1 = __shfl_xor_sync(0xFFFFFFFFu, s.m1, lane_mask);
    r.i1 = __shfl_xor_sync(0xFFFFFFFFu, s.i1, lane_mask);
    r.c1 = __shfl_xor_sync(0xFFFFFFFFu, s.c1, lane_mask);
    r.i1b = __shfl_xor_sync(0xFFFFFFFFu, s.i1b, lane_mask);
    r.m2 = __shfl_xor_sync(0xFFFFFFFFu, s.m2, lane_mask);
    r.i2 = __shfl_xor_sync(0xFFFFFFFFu, s.i2, lane_mask);
    return r;
}

constexpr int kTieThreads = 512;

// One block per row. kRound: every logit rounded to BF16 before any comparison (the contract); without it the FP32
// values are compared (the condition-1 mutant, sdar_mutants.h).
template <bool kRound>
__global__ void __launch_bounds__(kTieThreads) top_tie_kernel(const float *__restrict__ logits, int64_t stride,
                                                              int64_t vocab, TopTie *__restrict__ out) {
    __shared__ TieState warps[kTieThreads / 32];
    const int64_t row = blockIdx.x;
    const float *x = logits + row * stride;
    TieState s{-INFINITY, kNone, 0, kNone, -INFINITY, kNone};
    // Ids ascend per thread, so a new maximum demotes the old one to the runner-up and a tie's first repeat is the
    // second-lowest id.
    for (int64_t id = threadIdx.x; id < vocab; id += kTieThreads) {
        const float v = kRound ? round_bf16(x[id]) : x[id];
        const uint32_t i = uint32_t(id);
        if (s.c1 == 0 || v > s.m1) {
            if (s.c1 != 0) {
                s.m2 = s.m1;
                s.i2 = s.i1;
            }
            s.m1 = v;
            s.i1 = i;
            s.c1 = 1;
            s.i1b = kNone;
        } else if (v == s.m1) {
            ++s.c1;
            if (s.i1b == kNone) s.i1b = i;
        } else if (s.i2 == kNone || v > s.m2) {
            s.m2 = v;
            s.i2 = i;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s = merge(s, shfl_state(s, o));
    if (threadIdx.x % 32 == 0) warps[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x != 0) return;
    TieState r = warps[0];
    for (int w = 1; w < kTieThreads / 32; ++w) r = merge(r, warps[w]);
    TopTie t;
    t.id = r.i1;
    t.ties = r.c1;
    t.logit = r.m1;
    if (r.c1 > 1) {
        t.runner_up = r.i1b;
        t.runner_up_logit = r.m1;
    } else {  // kNone / -inf when the row has a single entry
        t.runner_up = r.i2;
        t.runner_up_logit = r.m2;
    }
    out[row] = t;
}

template <bool kRound>
cudaError_t launch_top_tie(const float *logits, int64_t stride, int64_t rows, int64_t vocab, TopTie *out,
                           cudaStream_t stream) {
    if (rows < 0 || vocab <= 0 || vocab > int64_t(kNone) || stride < vocab || rows > kMaxGridX) {
        return cudaErrorInvalidValue;
    }
    if (rows == 0) return cudaSuccess;
    if (!logits || !out) return cudaErrorInvalidValue;
    top_tie_kernel<kRound><<<unsigned(rows), kTieThreads, 0, stream>>>(logits, stride, vocab, out);
    return cudaGetLastError();
}

// ---- unmask_select -------------------------------------------------------------------------------------------------

// low_confidence_static with top_k 1: confidence = 1 / ties, the most confident masked row first, torch.topk's
// lowest index among equals.
__global__ void unmask_select_kernel(const TopTie *__restrict__ top, const uint8_t *__restrict__ masked, int rows,
                                     int32_t *__restrict__ index) {
    int32_t best = -1;
    uint32_t best_ties = 0;
    for (int r = 0; r < rows; ++r) {
        if (masked[r] != 0 && (best < 0 || top[r].ties < best_ties)) {
            best = r;
            best_ties = top[r].ties;
        }
    }
    *index = best;
}

}  // namespace

cudaError_t embed(const bf16 *table, const uint32_t *ids, int64_t n, int hidden, bf16 *out, cudaStream_t stream) {
    if (n < 0 || n > kMaxGridX || hidden <= 0) return cudaErrorInvalidValue;
    if (n == 0) return cudaSuccess;
    if (!table || !ids || !out) return cudaErrorInvalidValue;
    if (hidden % 8 == 0 && aligned16(table) && aligned16(out)) {
        embed_kernel<true><<<unsigned(n), kThreads, 0, stream>>>(table, ids, hidden, out);
    } else {
        embed_kernel<false><<<unsigned(n), kThreads, 0, stream>>>(table, ids, hidden, out);
    }
    return cudaGetLastError();
}

cudaError_t rmsnorm(const bf16 *x, int64_t x_stride, const bf16 *w, float eps, int64_t rows, int dim, bf16 *y,
                    int64_t y_stride, cudaStream_t stream) {
    if (rows < 0 || dim <= 0 || x_stride < dim || y_stride < dim || !(eps >= 0.0f)) return cudaErrorInvalidValue;
    if (x == y && x_stride != y_stride) return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    if (!x || !w || !y) return cudaErrorInvalidValue;
    if (dim <= kWarpRowMaxDim) {
        constexpr int64_t kRowsPerBlock = kThreads / 32;
        const int64_t blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
        if (blocks > kMaxGridX) return cudaErrorInvalidValue;
        rmsnorm_warp_kernel<<<unsigned(blocks), kThreads, 0, stream>>>(x, x_stride, w, eps, rows, dim, y, y_stride);
    } else {
        if (rows > kMaxGridX) return cudaErrorInvalidValue;
        rmsnorm_block_kernel<<<unsigned(rows), kNormBlock, 0, stream>>>(x, x_stride, w, eps, dim, y, y_stride);
    }
    return cudaGetLastError();
}

cudaError_t rope(bf16 *x, int64_t stride, int64_t rows, int heads, int64_t pos0, float theta, cudaStream_t stream) {
    if (rows < 0 || rows > kMaxGridX || heads <= 0 || stride < int64_t(heads) * kHeadDim || pos0 < 0 ||
        !(theta > 0.0f)) {
        return cudaErrorInvalidValue;
    }
    if (rows == 0) return cudaSuccess;
    if (!x) return cudaErrorInvalidValue;
    rope_kernel<<<unsigned(rows), kThreads, 0, stream>>>(x, stride, heads, pos0, theta);
    return cudaGetLastError();
}

cudaError_t kv_store(const bf16 *k, int64_t k_stride, const bf16 *v, int64_t v_stride, int64_t n, int kv_heads,
                     bf16 *k_cache, bf16 *v_cache, int64_t pos, cudaStream_t stream) {
    const int64_t row = int64_t(kv_heads) * kHeadDim;
    if (n < 0 || n > kMaxGridX || kv_heads <= 0 || pos < 0 || k_stride < row || v_stride < row) {
        return cudaErrorInvalidValue;
    }
    if (n == 0) return cudaSuccess;
    if (!k || !v || !k_cache || !v_cache) return cudaErrorInvalidValue;
    if (aligned16(k) && aligned16(v) && aligned16(k_cache) && aligned16(v_cache) && k_stride % 8 == 0 &&
        v_stride % 8 == 0) {
        kv_store_kernel<true><<<unsigned(n), kThreads, 0, stream>>>(k, k_stride, v, v_stride, row, k_cache, v_cache,
                                                                    pos);
    } else {
        kv_store_kernel<false><<<unsigned(n), kThreads, 0, stream>>>(k, k_stride, v, v_stride, row, k_cache, v_cache,
                                                                     pos);
    }
    return cudaGetLastError();
}

cudaError_t attention(const bf16 *q, int64_t q_stride, const bf16 *k_cache, const bf16 *v_cache, int64_t past,
                      const bf16 *new_k, int64_t new_k_stride, const bf16 *new_v, int64_t new_v_stride, int64_t n,
                      int heads, int kv_heads, bf16 *out, int64_t out_stride, Mask mask, cudaStream_t stream) {
    if (n < 0 || past < 0 || past % kBlock != 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 ||
        heads > kMaxGridY || (mask != Mask::kBlockCausal && mask != Mask::kCausal)) {
        return cudaErrorInvalidValue;
    }
    const int64_t q_row = int64_t(heads) * kHeadDim;
    const int64_t kv_row = int64_t(kv_heads) * kHeadDim;
    if (q_stride < q_row || out_stride < q_row || new_k_stride < kv_row || new_v_stride < kv_row) {
        return cudaErrorInvalidValue;
    }
    const int64_t blocks = (n + kAttnRows - 1) / kAttnRows;
    if (blocks > kMaxGridX) return cudaErrorInvalidValue;
    if (n == 0) return cudaSuccess;
    if (!q || !new_k || !new_v || !out || (past > 0 && (!k_cache || !v_cache))) return cudaErrorInvalidValue;
    const AttnArgs a{q,     q_stride,     k_cache,          v_cache, past,       new_k,     new_k_stride, new_v,
                     new_v_stride, n, heads / kv_heads, kv_row,  out, out_stride, int(mask),
                     float(1.0 / std::sqrt(double(kHeadDim)))};
    const bool vec = aligned16(new_k) && aligned16(new_v) && new_k_stride % 8 == 0 && new_v_stride % 8 == 0 &&
                     (past == 0 || (aligned16(k_cache) && aligned16(v_cache)));
    const dim3 grid{unsigned(blocks), unsigned(heads)};
    if (vec) {
        attention_kernel<true><<<grid, kAttnWarps * 32, 0, stream>>>(a);
    } else {
        attention_kernel<false><<<grid, kAttnWarps * 32, 0, stream>>>(a);
    }
    return cudaGetLastError();
}

cudaError_t silu_mul(const bf16 *gate_up, int64_t rows, int inter, bf16 *out, cudaStream_t stream) {
    if (rows < 0 || inter <= 0) return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    if (!gate_up || !out) return cudaErrorInvalidValue;
    constexpr int kBlock256 = 256;
    constexpr int64_t kMaxBlocks = 8192;
    const int64_t total = rows * int64_t(inter);
    const int64_t want = (total + kBlock256 - 1) / kBlock256;
    silu_mul_kernel<<<unsigned(want < kMaxBlocks ? want : kMaxBlocks), kBlock256, 0, stream>>>(gate_up, rows, inter,
                                                                                               out);
    return cudaGetLastError();
}

cudaError_t add(bf16 *x, const bf16 *y, int64_t n, cudaStream_t stream) {
    if (n < 0) return cudaErrorInvalidValue;
    if (n == 0) return cudaSuccess;
    if (!x || !y) return cudaErrorInvalidValue;
    constexpr int kBlock256 = 256;
    constexpr int64_t kMaxBlocks = 8192;
    const int64_t want = (n + kBlock256 - 1) / kBlock256;
    add_kernel<<<unsigned(want < kMaxBlocks ? want : kMaxBlocks), kBlock256, 0, stream>>>(x, y, n);
    return cudaGetLastError();
}

cudaError_t top_tie(const float *logits, int64_t stride, int64_t rows, int64_t vocab, TopTie *out,
                    cudaStream_t stream) {
    return launch_top_tie<true>(logits, stride, rows, vocab, out, stream);
}

cudaError_t unmask_select(const TopTie *top, const uint8_t *masked, int rows, int32_t *index, cudaStream_t stream) {
    if (rows < 0 || !index || (rows > 0 && (!top || !masked))) return cudaErrorInvalidValue;
    unmask_select_kernel<<<1, 1, 0, stream>>>(top, masked, rows, index);
    return cudaGetLastError();
}

namespace mutant {

cudaError_t top_tie_fp32(const float *logits, int64_t stride, int64_t rows, int64_t vocab, TopTie *out,
                         cudaStream_t stream) {
    return launch_top_tie<false>(logits, stride, rows, vocab, out, stream);
}

}  // namespace mutant

}  // namespace cere::sdar
