// K4 (m22 design note): PLE n-gram hash, mapped pinned gather (FP8, or NVFP4 per fp4local design note D8),
// gate/value, short conv with state.
//
// math_reference: sglang
//   python/sglang/srt/models/qwen4_exp.py :603-683 (hash), :718-870 (pinned gather + scale), :955-1074 (short conv),
//   :1162-1218 (PLE layer forward); python/sglang/kernels/ops/qwen4_ple.py (fused hash, gate/value, conv state).
#include "kernels/k4_ple.h"

#include "kernels/k1_primitives.h"
#include "kernels/k2_gemm.h"
#include "kernels/k4_nvfp4.h"

#include <cuda_fp8.h>

#include <cmath>

namespace cere::k4 {

namespace {

constexpr int kRowChunks = kHeadDim / 16;                  // 10 x 16 FP8 bytes per table row
constexpr int kGatherThreads = kNgramHeads * kRowChunks;   // 160: one token's 16 rows per block
constexpr int kGroupThreads = kHidden / 8;                 // 320: 8 BF16 elements per thread per 2,560 group
constexpr int kConvThreads = 128;
constexpr int kConvVec = 4;    // conv_tile_kernel: channels per thread (one 8-byte access)
constexpr int kConvTile = 16;  // conv_tile_kernel: tokens per thread; forward takes it for t > kConvTile

static_assert(kHeadDim % 16 == 0, "table rows are read in 16-byte chunks");
static_assert(nvfp4::kRowElements == kHeadDim && nvfp4::kBlock == 16 && kRowChunks == nvfp4::kRowScales,
              "an NVFP4 chunk is one 16-element block: 8 packed bytes and one E4M3 scale");
static_assert(kConvStateLen == 9 && kConvKernel == 4 && kConvDilation == 3, "conv window layout");
static_assert(kChannels % (kConvVec * kConvThreads) == 0 && kConvTile >= kConvStateLen,
              "conv_tile_kernel covers the channels exactly; its t > kConvTile steps end with a full gn window");

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// `_hash_contexts` for the last position of a window (t0, t1, t2) (qwen4_exp.py:603-656): the shift-1 source is t1
// (EOS stays EOS), the shift-2 source is EOS when t0 or t1 is EOS (`_shift_right_ignore_eos` :658-683 on a 3-wide
// window; qwen4_ple.py `_qwen4_ngram_hash_kernel`). Products wrap as int64; floor-mod as torch.remainder.
__device__ __forceinline__ int64_t ngram_id(int64_t t0, int64_t t1, int64_t t2, int head, const int64_t *mult,
                                            const int64_t *vocab, const int64_t *offs, int64_t eos) {
    const int64_t prev1 = t1;
    const int64_t prev2 = (t0 == eos || t1 == eos) ? eos : t0;
    uint64_t mix = uint64_t(t2) * uint64_t(mult[0]) ^ uint64_t(prev1) * uint64_t(mult[1]);
    if (head >= kHeadsPerNgram) mix ^= uint64_t(prev2) * uint64_t(mult[2]);
    const int64_t a = int64_t(mix);
    const int64_t b = vocab[head];
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r + offs[head];
}

// One 16-byte chunk of a table row: 16 FP8 E4M3 -> 16 BF16 (exact), optionally times the BF16 scale (one rounding,
// as `embeddings * weight_scale` in BF16). An out-of-range id reads zero bytes, i.e. a +0 row that is then scaled like
// SGLang's `tl.where(in_range, values, 0.0)` row (`:740-745`).
__device__ __forceinline__ void gather_chunk(const uint8_t *table, int64_t table_rows, int64_t id, int chunk,
                                             const __nv_bfloat16 *scale, __nv_bfloat16 *dst) {
    uint4 raw = make_uint4(0, 0, 0, 0);
    if (id >= 0 && id < table_rows) raw = *reinterpret_cast<const uint4 *>(table + id * kHeadDim + chunk * 16);
    const float s = scale != nullptr ? __bfloat162float(*scale) : 1.0f;
    const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
    __align__(16) __nv_bfloat162 out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t word = words[i / 2];
        const __nv_fp8x2_storage_t pair = __nv_fp8x2_storage_t((i % 2 == 0) ? (word & 0xFFFFu) : (word >> 16));
        const float2 f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3)));
        float lo = __bfloat162float(__float2bfloat16_rn(f.x));
        float hi = __bfloat162float(__float2bfloat16_rn(f.y));
        if (scale != nullptr) {
            lo *= s;
            hi *= s;
        }
        out[i] = __floats2bfloat162_rn(lo, hi);
    }
    const uint4 *src = reinterpret_cast<const uint4 *>(out);
    uint4 *d = reinterpret_cast<uint4 *>(dst);
    d[0] = src[0];
    d[1] = src[1];
}

// The NVFP4 counterpart (D8): one 16-element block of a packed row (8 bytes, one E4M3 scale) decoded as k4_nvfp4.h,
// one BF16 RNE per element. An out-of-range id decodes zero bytes, as gather_chunk does.
__device__ __forceinline__ void gather_chunk_nvfp4(const uint8_t *table, const uint8_t *block_scales,
                                                   const float *global_scale, int64_t table_rows, int64_t id,
                                                   int chunk, __nv_bfloat16 *dst) {
    uint2 raw = make_uint2(0, 0);
    uint32_t scale_byte = 0;
    if (id >= 0 && id < table_rows) {
        raw = *reinterpret_cast<const uint2 *>(table + id * nvfp4::kRowBytes + chunk * nvfp4::kBlockBytes);
        scale_byte = block_scales[id * nvfp4::kRowScales + chunk];
    }
    const float scale = nvfp4::e4m3(scale_byte);
    const float global = *global_scale;
    __align__(16) __nv_bfloat162 out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t byte = ((i < 4 ? raw.x : raw.y) >> (8 * (i % 4))) & 0xFFu;
        out[i] = __floats2bfloat162_rn(nvfp4::scaled(byte & 0xFu, scale, global),
                                       nvfp4::scaled(byte >> 4, scale, global));
    }
    const uint4 *src = reinterpret_cast<const uint4 *>(out);
    uint4 *d = reinterpret_cast<uint4 *>(dst);
    d[0] = src[0];
    d[1] = src[1];
}

// One table chunk of `w`'s format.
template <TableFormat F>
__device__ __forceinline__ void table_chunk(const Weights &w, int64_t id, int chunk, __nv_bfloat16 *dst) {
    if constexpr (F == TableFormat::kNvfp4) {
        gather_chunk_nvfp4(w.table, w.block_scales, w.global_scale, w.table_rows, id, chunk, dst);
    } else {
        gather_chunk(w.table, w.table_rows, id, chunk, w.table_scale, dst);
    }
}

__global__ void hash_windows_kernel(const int64_t *windows, int64_t n, const int64_t *mult, const int64_t *vocab,
                                    const int64_t *offs, int64_t eos, int64_t *ids) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n * kNgramHeads) return;
    const int64_t w = i / kNgramHeads;
    const int head = int(i % kNgramHeads);
    ids[i] = ngram_id(windows[w * 3], windows[w * 3 + 1], windows[w * 3 + 2], head, mult, vocab, offs, eos);
}

__global__ void gather_kernel(const uint8_t *table, int64_t table_rows, const int64_t *ids, int64_t n,
                              const __nv_bfloat16 *scale, __nv_bfloat16 *out) {
    const int64_t row = int64_t(blockIdx.x) * kNgramHeads + threadIdx.x / kRowChunks;
    const int chunk = threadIdx.x % kRowChunks;
    if (row >= n) return;
    gather_chunk(table, table_rows, ids[row], chunk, scale, out + row * kHeadDim + chunk * 16);
}

__global__ void gather_nvfp4_kernel(const uint8_t *table, const uint8_t *block_scales, const float *global_scale,
                                    int64_t table_rows, const int64_t *ids, int64_t n, __nv_bfloat16 *out) {
    const int64_t row = int64_t(blockIdx.x) * kNgramHeads + threadIdx.x / kRowChunks;
    const int chunk = threadIdx.x % kRowChunks;
    if (row >= n) return;
    gather_chunk_nvfp4(table, block_scales, global_scale, table_rows, ids[row], chunk,
                       out + row * kHeadDim + chunk * 16);
}

// The 16 ids of the window (c0, c1, token) into `ids` (shared), then its 16 scaled rows into `emb` [2560]: one
// token per block of kGatherThreads.
template <TableFormat F>
__device__ __forceinline__ void embed_window(int64_t c0, int64_t c1, int64_t token, const Weights &w, int64_t *ids,
                                             __nv_bfloat16 *emb) {
    if (threadIdx.x < kNgramHeads) {
        ids[threadIdx.x] =
            ngram_id(c0, c1, token, int(threadIdx.x), w.multipliers, w.head_vocab_sizes, w.head_offsets, w.eos_token_id);
    }
    __syncthreads();
    const int row = threadIdx.x / kRowChunks;
    const int chunk = threadIdx.x % kRowChunks;
    table_chunk<F>(w, ids[row], chunk, emb + row * kHeadDim + chunk * 16);
}

// One block per token: the token's 16 ids from its window over history ++ tokens, then its 16 scaled rows.
template <TableFormat F>
__global__ void embed_kernel(const int64_t *history, const int64_t *tokens, Weights w, __nv_bfloat16 *emb) {
    __shared__ int64_t ids[kNgramHeads];
    const int64_t t = blockIdx.x;
    // ctx = history[0], history[1], tokens[0], ...; the window of token t is ctx[t .. t + 2].
    const int64_t c0 = t >= 2 ? tokens[t - 2] : history[t];
    const int64_t c1 = t >= 1 ? tokens[t - 1] : history[1];
    embed_window<F>(c0, c1, tokens[t], w, ids, emb + t * kEmbedDim);
}

// The next history = the last two entries of history ++ tokens (`_commit_ple_batch`, qwen4_exp.py:276-322).
__global__ void commit_history_kernel(int64_t *history, const int64_t *tokens, int64_t t) {
    const int64_t h0 = t >= 2 ? tokens[t - 2] : history[1];
    const int64_t h1 = tokens[t - 1];
    history[0] = h0;
    history[1] = h1;
}

// forward_rows' call with each row's sequence and index in it (kernel parameter, by value).
struct RowsArgs {
    RowsCall call;
    uint8_t seq[kMaxRows];
    uint8_t j[kMaxRows];
    uint32_t m;
};

// One block per row r of a rows call: embed_kernel's block for row j of its sequence (tokens of the sequence from
// first[s], its history before the call).
template <TableFormat F>
__global__ void embed_rows_kernel(RowsArgs a, const int64_t *tokens, Weights w, __nv_bfloat16 *emb) {
    __shared__ int64_t ids[kNgramHeads];
    const uint32_t r = blockIdx.x;
    const uint32_t s = a.seq[r];
    const int64_t j = a.j[r];
    const int64_t *history = a.call.states[s].history;
    const int64_t *seq_tokens = tokens + a.call.first[s];
    const int64_t c0 = j >= 2 ? seq_tokens[j - 2] : history[j];
    const int64_t c1 = j >= 1 ? seq_tokens[j - 1] : history[1];
    embed_window<F>(c0, c1, seq_tokens[j], w, ids, emb + int64_t(r) * kEmbedDim);
}

// One block of kMaxRows threads: row r's history save (the history after row r: commit_history_kernel at t = j + 1),
// then each sequence's commit (after the saves, which read the history before the call).
__global__ void commit_rows_kernel(RowsArgs a, const int64_t *tokens) {
    const uint32_t i = threadIdx.x;
    if (i < a.m && a.call.saves[i].history != nullptr) {
        const uint32_t s = a.seq[i];
        const int64_t j = a.j[i];
        const int64_t *seq_tokens = tokens + a.call.first[s];
        a.call.saves[i].history[0] = j >= 1 ? seq_tokens[j - 1] : a.call.states[s].history[1];
        a.call.saves[i].history[1] = seq_tokens[j];
    }
    __syncthreads();
    if (i < a.call.sequences) {
        int64_t *history = a.call.states[i].history;
        const int64_t *seq_tokens = tokens + a.call.first[i];
        const int64_t t = a.call.count[i];
        const int64_t h0 = t >= 2 ? seq_tokens[t - 2] : history[1];
        const int64_t h1 = seq_tokens[t - 1];
        history[0] = h0;
        history[1] = h1;
    }
}

__device__ __forceinline__ float block_sum(float v, float *smem) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    const int warps = blockDim.x / 32;
    float total = lane < warps ? smem[lane] : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) total += __shfl_xor_sync(0xFFFFFFFFu, total, o);
        if (lane == 0) smem[0] = total;
    }
    __syncthreads();
    total = smem[0];
    __syncthreads();
    return total;
}

// One block per (token, stream g): gate = sum(kn * qn) (BF16 products, FP32 sum, BF16 result) / sqrt(2560) in BF16,
// then the signed sqrt, sigmoid and value broadcast with every BF16 rounding of the eager graph
// (`Qwen4ExpPLELayer.forward` :1184-1205; `_qwen4_gate_value_kernel`).
__global__ void __launch_bounds__(kGroupThreads)
    gate_kernel(const __nv_bfloat16 *kn, const __nv_bfloat16 *qn, const __nv_bfloat16 *value, float inv_sqrt_hidden,
                __nv_bfloat16 *gated) {
    __shared__ float smem[32];
    const int64_t row = blockIdx.x;  // token * 4 + g
    const int64_t token = row / kHcCount;
    const int e = threadIdx.x * 8;
    const uint4 kraw = *reinterpret_cast<const uint4 *>(kn + row * kHidden + e);
    const uint4 qraw = *reinterpret_cast<const uint4 *>(qn + row * kHidden + e);
    const uint4 vraw = *reinterpret_cast<const uint4 *>(value + token * kHidden + e);
    const __nv_bfloat162 *k2 = reinterpret_cast<const __nv_bfloat162 *>(&kraw);
    const __nv_bfloat162 *q2 = reinterpret_cast<const __nv_bfloat162 *>(&qraw);
    const __nv_bfloat162 *v2 = reinterpret_cast<const __nv_bfloat162 *>(&vraw);
    float partial = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 k = __bfloat1622float2(k2[i]);
        const float2 q = __bfloat1622float2(q2[i]);
        partial += bf16r(k.x * q.x) + bf16r(k.y * q.y);
    }
    const float dot = bf16r(block_sum(partial, smem));
    const float gate = bf16r(dot * inv_sqrt_hidden);
    // `gate.abs().clamp_min(1e-6)`: torch casts the scalar bound to the BF16 tensor dtype.
    const float root = bf16r(sqrtf(fmaxf(fabsf(gate), bf16r(1.0e-6f))));
    const float sign = gate > 0.0f ? 1.0f : (gate < 0.0f ? -1.0f : 0.0f);
    const float transformed = bf16r(root * sign);
    const float activated = bf16r(1.0f / (1.0f + expf(-transformed)));
    __align__(16) __nv_bfloat162 out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 v = __bfloat1622float2(v2[i]);
        out[i] = __floats2bfloat162_rn(activated * v.x, activated * v.y);
    }
    *reinterpret_cast<uint4 *>(gated + row * kHidden + e) = *reinterpret_cast<const uint4 *>(out);
}

// One token of one channel of the short conv: the depthwise conv (kernel 4, dilation 3) over [window (9) | x], FP32
// accumulation, BF16 output, SiLU in BF16, delta = gated + conv (+ query when add_query); the window then takes x.
__device__ __forceinline__ __nv_bfloat16 conv_token(const float (&w)[kConvKernel],
                                                   __nv_bfloat16 (&window)[kConvStateLen], __nv_bfloat16 x,
                                                   __nv_bfloat16 gated, const __nv_bfloat16 *query, bool add_query) {
    float acc = w[0] * __bfloat162float(window[0]);
    acc = fmaf(w[1], __bfloat162float(window[3]), acc);
    acc = fmaf(w[2], __bfloat162float(window[6]), acc);
    acc = fmaf(w[3], __bfloat162float(x), acc);
    const float conv = bf16r(acc);
    const float act = bf16r(conv / (1.0f + expf(-conv)));
    float delta = bf16r(__bfloat162float(gated) + act);
    if (add_query) delta = __bfloat162float(*query) + delta;
#pragma unroll
    for (int j = 0; j + 1 < kConvStateLen; ++j) window[j] = window[j + 1];
    window[kConvStateLen - 1] = x;
    return __float2bfloat16_rn(delta);
}

// One thread per channel, sequential over the step's tokens: conv_token over [state (9) | gn]; the state keeps the
// last 9 inputs (`_short_conv` :955-1074, `_qwen4_short_conv_state_kernel`). forward's steps of t <= kConvTile rows
// (decode); longer steps take conv_tile_kernel.
__global__ void __launch_bounds__(kConvThreads)
    conv_kernel(const __nv_bfloat16 *gn, const __nv_bfloat16 *gated, const __nv_bfloat16 *conv_weight,
                __nv_bfloat16 *conv_state, int64_t t, const __nv_bfloat16 *query, bool add_query,
                __nv_bfloat16 *out) {
    const int c = blockIdx.x * kConvThreads + threadIdx.x;
    if (c >= kChannels) return;
    float w[kConvKernel];
#pragma unroll
    for (int k = 0; k < kConvKernel; ++k) w[k] = __bfloat162float(conv_weight[c * kConvKernel + k]);
    __nv_bfloat16 window[kConvStateLen];
#pragma unroll
    for (int j = 0; j < kConvStateLen; ++j) window[j] = conv_state[c * kConvStateLen + j];
    for (int64_t i = 0; i < t; ++i) {
        const int64_t at = i * kChannels + c;
        out[at] = conv_token(w, window, gn[at], gated[at], query + at, add_query);
    }
#pragma unroll
    for (int j = 0; j < kConvStateLen; ++j) conv_state[c * kConvStateLen + j] = window[j];
}

// conv_kernel per sequence of a rows call (grid.y = sequence): its rows in order from its own conv state; after row r
// the window goes to saves[r].conv_state where set.
__global__ void __launch_bounds__(kConvThreads)
    conv_rows_kernel(const __nv_bfloat16 *gn, const __nv_bfloat16 *gated, const __nv_bfloat16 *conv_weight,
                     RowsArgs a, const __nv_bfloat16 *query, bool add_query, __nv_bfloat16 *out) {
    const int c = blockIdx.x * kConvThreads + threadIdx.x;
    if (c >= kChannels) return;
    const uint32_t s = blockIdx.y;
    __nv_bfloat16 *conv_state = a.call.states[s].conv_state;
    float w[kConvKernel];
#pragma unroll
    for (int k = 0; k < kConvKernel; ++k) w[k] = __bfloat162float(conv_weight[c * kConvKernel + k]);
    __nv_bfloat16 window[kConvStateLen];
#pragma unroll
    for (int j = 0; j < kConvStateLen; ++j) window[j] = conv_state[c * kConvStateLen + j];
    const uint32_t first = a.call.first[s];
    for (uint32_t r = first; r < first + a.call.count[s]; ++r) {
        const int64_t at = int64_t(r) * kChannels + c;
        out[at] = conv_token(w, window, gn[at], gated[at], query + at, add_query);
        __nv_bfloat16 *save = a.call.saves[r].conv_state;
        if (save != nullptr) {
#pragma unroll
            for (int j = 0; j < kConvStateLen; ++j) save[c * kConvStateLen + j] = window[j];
        }
    }
#pragma unroll
    for (int j = 0; j < kConvStateLen; ++j) conv_state[c * kConvStateLen + j] = window[j];
}

// conv_kernel's outputs for prefill steps, token-parallel: conv_token's output for token i reads only its window, the
// 9 inputs before it in [state (9) | gn] (no output feeds a later one). One thread per kConvVec channels and
// kConvTile tokens (grid (tiles, channel blocks)): it loads the window before its first token (from gn, or from the
// old conv state before token 0), then runs conv_token over its tokens exactly as conv_kernel does, so every output is
// the same arithmetic on the same values. conv_state is only read here; forward writes the new state afterwards
// (conv_state_at_kernel at row t). out may alias query (each element is read and written by the same thread only);
// it may not alias gn or gated, which other threads read. out_vec: out is 8-byte aligned (else scalar stores).
__global__ void __launch_bounds__(kConvThreads)
    conv_tile_kernel(const __nv_bfloat16 *gn, const __nv_bfloat16 *gated, const __nv_bfloat16 *conv_weight,
                     const __nv_bfloat16 *conv_state, int64_t t, const __nv_bfloat16 *query, bool add_query,
                     __nv_bfloat16 *out, bool out_vec) {
    const int c0 = (blockIdx.y * kConvThreads + threadIdx.x) * kConvVec;
    const int64_t i0 = int64_t(blockIdx.x) * kConvTile;
    const int64_t i1 = i0 + kConvTile < t ? i0 + kConvTile : t;
    float w[kConvVec][kConvKernel];
#pragma unroll
    for (int v = 0; v < kConvVec; ++v) {
#pragma unroll
        for (int k = 0; k < kConvKernel; ++k) w[v][k] = __bfloat162float(conv_weight[(c0 + v) * kConvKernel + k]);
    }
    // window[v][j] = position i0 - 9 + j of [state | gn]: state entry 9 + p for p < 0.
    __nv_bfloat16 window[kConvVec][kConvStateLen];
#pragma unroll
    for (int j = 0; j < kConvStateLen; ++j) {
        const int64_t p = i0 - kConvStateLen + j;
        if (p >= 0) {
            __align__(8) __nv_bfloat16 x[kConvVec];
            *reinterpret_cast<uint2 *>(x) = *reinterpret_cast<const uint2 *>(gn + p * kChannels + c0);
#pragma unroll
            for (int v = 0; v < kConvVec; ++v) window[v][j] = x[v];
        } else {
#pragma unroll
            for (int v = 0; v < kConvVec; ++v) window[v][j] = conv_state[(c0 + v) * kConvStateLen + kConvStateLen + p];
        }
    }
    for (int64_t i = i0; i < i1; ++i) {
        const int64_t at = i * kChannels + c0;
        __align__(8) __nv_bfloat16 x[kConvVec], g[kConvVec], q[kConvVec], o[kConvVec];
        *reinterpret_cast<uint2 *>(x) = *reinterpret_cast<const uint2 *>(gn + at);
        *reinterpret_cast<uint2 *>(g) = *reinterpret_cast<const uint2 *>(gated + at);
        if (add_query) *reinterpret_cast<uint2 *>(q) = *reinterpret_cast<const uint2 *>(query + at);
#pragma unroll
        for (int v = 0; v < kConvVec; ++v) o[v] = conv_token(w[v], window[v], x[v], g[v], q + v, add_query);
        if (out_vec) {
            *reinterpret_cast<uint2 *>(out + at) = *reinterpret_cast<const uint2 *>(o);
        } else {
#pragma unroll
            for (int v = 0; v < kConvVec; ++v) out[at + v] = o[v];
        }
    }
}

// The conv state `row` tokens into the step (row >= 9): the 9 gn rows before it, as conv_kernel leaves it after them.
__global__ void __launch_bounds__(kConvThreads)
    conv_state_at_kernel(const __nv_bfloat16 *gn, int64_t row, __nv_bfloat16 *conv_state) {
    const int c = blockIdx.x * kConvThreads + threadIdx.x;
    if (c >= kChannels) return;
#pragma unroll
    for (int j = 0; j < kConvStateLen; ++j) {
        conv_state[c * kConvStateLen + j] = gn[(row - kConvStateLen + j) * kChannels + c];
    }
}

constexpr size_t align256(size_t bytes) { return (bytes + 255) & ~size_t(255); }

struct Layout {
    __nv_bfloat16 *emb;    // [t, 2560]
    __nv_bfloat16 *key;    // [t, 10240]: key, then kn (in place), then gn
    __nv_bfloat16 *value;  // [t, 2560]
    __nv_bfloat16 *qn;     // [t, 10240]
    __nv_bfloat16 *gated;  // [t, 10240]
    size_t bytes;
};

Layout layout(void *base, int64_t t) {
    const size_t n = size_t(t);
    const size_t emb = align256(n * kEmbedDim * sizeof(__nv_bfloat16));
    const size_t wide = align256(n * kChannels * sizeof(__nv_bfloat16));
    const size_t narrow = align256(n * kHidden * sizeof(__nv_bfloat16));
    char *p = static_cast<char *>(base);
    Layout l{};
    l.emb = reinterpret_cast<__nv_bfloat16 *>(p);
    l.key = reinterpret_cast<__nv_bfloat16 *>(p + emb);
    l.value = reinterpret_cast<__nv_bfloat16 *>(p + emb + wide);
    l.qn = reinterpret_cast<__nv_bfloat16 *>(p + emb + wide + narrow);
    l.gated = reinterpret_cast<__nv_bfloat16 *>(p + emb + 2 * wide + narrow);
    l.bytes = emb + 3 * wide + narrow;
    return l;
}

cere_status launched() { return cudaGetLastError() == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL; }

// The embedding kernel of `w`'s format (check_table has accepted `w`).
cere_status launch_embed(const Weights &w, const int64_t *history, const int64_t *tokens, int64_t t,
                         __nv_bfloat16 *emb, cudaStream_t stream) {
    if (w.format == TableFormat::kNvfp4) {
        embed_kernel<TableFormat::kNvfp4><<<unsigned(t), kGatherThreads, 0, stream>>>(history, tokens, w, emb);
    } else {
        embed_kernel<TableFormat::kFp8><<<unsigned(t), kGatherThreads, 0, stream>>>(history, tokens, w, emb);
    }
    return launched();
}

// forward's refusals that do not depend on the state (shared with forward_rows): t rows, the plans, the workspace, the
// pointers and the 16-byte alignment K2 (`GemmPlan::run`) and K1 (`rmsnorm`) require, checked before any launch so
// that they cannot reject after the history has advanced.
cere_status check_step(const Weights &w, const Plans &plans, int64_t t, const int64_t *tokens,
                       const __nv_bfloat16 *query, const __nv_bfloat16 *out, const void *workspace,
                       size_t workspace_size) {
    if (t <= 0 || t > int64_t(INT32_MAX) / kHcCount) return CERE_ERR_ARGUMENT;
    if (plans.key == nullptr || plans.value == nullptr || plans.key->n() != uint32_t(kChannels) ||
        plans.value->n() != uint32_t(kHidden) || plans.key->k() != uint32_t(kEmbedDim) ||
        plans.value->k() != uint32_t(kEmbedDim) || plans.key->out() != kernels::GemmOut::kBF16 ||
        plans.value->out() != kernels::GemmOut::kBF16 || t > int64_t(plans.key->max_m()) ||
        t > int64_t(plans.value->max_m()))
        return CERE_ERR_ARGUMENT;
    if (workspace == nullptr || reinterpret_cast<uintptr_t>(workspace) % 256 != 0 ||
        workspace_size < workspace_bytes(t))
        return CERE_ERR_ARGUMENT;
    if (tokens == nullptr || query == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    if (check_table(w) != CERE_OK) return CERE_ERR_ARGUMENT;
    const auto a16 = [](const void *p) { return p != nullptr && reinterpret_cast<uintptr_t>(p) % 16 == 0; };
    if (!a16(w.key_proj) || !a16(w.value_proj) || !a16(w.norm_key) || !a16(w.norm_query) || !a16(w.norm_conv) ||
        !a16(query))
        return CERE_ERR_ARGUMENT;
    return CERE_OK;
}

// forward's row-wise stages between the embedding and the conv over t rows (row r reads row r only): the key /
// value GEMMs, the grouped norms, the gate and the conv input norm into ws.key (gn).
cere_status gate_rows(const Weights &w, const Plans &plans, const Layout &ws, int64_t t, const __nv_bfloat16 *query,
                      cudaStream_t stream) {
    const int64_t rows = t * kHcCount;
    cere_status s;
    // key_proj / value_proj (ReplicatedLinear, no bias): emb [t, 2560] @ W^T.
    if ((s = plans.key->run(ws.emb, w.key_proj, uint32_t(t), ws.key, stream, plans.cls)) != CERE_OK) return s;
    if ((s = plans.value->run(ws.emb, w.value_proj, uint32_t(t), ws.value, stream, plans.cls)) != CERE_OK) return s;

    // Grouped Gemma RMSNorm (Qwen4ExpPLEGroupedNorm, group 2,560): rows of 2,560, weight group = row % 4.
    if (k1::rmsnorm(ws.key, kHidden, w.norm_key, kHcCount, true, w.eps, rows, kHidden, ws.key, kHidden, stream) !=
        cudaSuccess)
        return CERE_ERR_INTERNAL;
    if (k1::rmsnorm(query, kHidden, w.norm_query, kHcCount, true, w.eps, rows, kHidden, ws.qn, kHidden, stream) !=
        cudaSuccess)
        return CERE_ERR_INTERNAL;

    // `gate / math.sqrt(hidden_size)`: a BF16 CUDA tensor divided by a CPU scalar is a multiply by the FP32
    // reciprocal of the scalar (PyTorch div_true_kernel_cuda).
    const float inv_sqrt_hidden = 1.0f / float(std::sqrt(double(kHidden)));
    gate_kernel<<<unsigned(rows), kGroupThreads, 0, stream>>>(ws.key, ws.qn, ws.value, inv_sqrt_hidden, ws.gated);
    if ((s = launched()) != CERE_OK) return s;

    if (k1::rmsnorm(ws.gated, kHidden, w.norm_conv, kHcCount, true, w.eps, rows, kHidden, ws.key, kHidden, stream) !=
        cudaSuccess)
        return CERE_ERR_INTERNAL;
    return CERE_OK;
}

}  // namespace

cere_status ngram_hash_windows(const int64_t *windows, int64_t n, const int64_t *multipliers,
                               const int64_t *head_vocab_sizes, const int64_t *head_offsets, int64_t eos_token_id,
                               int64_t *ids, cudaStream_t stream) {
    if (n < 0 || n > int64_t(INT32_MAX)) return CERE_ERR_ARGUMENT;
    if (n == 0) return CERE_OK;
    const int64_t outputs = n * kNgramHeads;
    const int threads = 256;
    hash_windows_kernel<<<unsigned((outputs + threads - 1) / threads), threads, 0, stream>>>(
        windows, n, multipliers, head_vocab_sizes, head_offsets, eos_token_id, ids);
    return launched();
}

cere_status gather(const uint8_t *table, int64_t table_rows, const int64_t *ids, int64_t n,
                   const __nv_bfloat16 *scale, __nv_bfloat16 *out, cudaStream_t stream) {
    if (n < 0 || n > int64_t(INT32_MAX) || table_rows < 0) return CERE_ERR_ARGUMENT;
    if (n == 0) return CERE_OK;
    gather_kernel<<<unsigned((n + kNgramHeads - 1) / kNgramHeads), kGatherThreads, 0, stream>>>(table, table_rows,
                                                                                                ids, n, scale, out);
    return launched();
}

cere_status check_table(const Weights &w) {
    if (w.table == nullptr || w.table_rows < 0) return CERE_ERR_ARGUMENT;
    switch (w.format) {
        case TableFormat::kFp8:
            return w.block_scales == nullptr && w.global_scale == nullptr ? CERE_OK : CERE_ERR_ARGUMENT;
        case TableFormat::kNvfp4:
            return w.table_scale == nullptr && w.block_scales != nullptr && w.global_scale != nullptr &&
                           reinterpret_cast<uintptr_t>(w.table) % 8 == 0
                       ? CERE_OK
                       : CERE_ERR_ARGUMENT;
    }
    return CERE_ERR_ARGUMENT;
}

cere_status gather_nvfp4(const uint8_t *table, const uint8_t *block_scales, const float *global_scale,
                         int64_t table_rows, const int64_t *ids, int64_t n, __nv_bfloat16 *out, cudaStream_t stream) {
    if (n < 0 || n > int64_t(INT32_MAX) || table_rows < 0 || table == nullptr || block_scales == nullptr ||
        global_scale == nullptr || reinterpret_cast<uintptr_t>(table) % 8 != 0)
        return CERE_ERR_ARGUMENT;
    if (n == 0) return CERE_OK;
    gather_nvfp4_kernel<<<unsigned((n + kNgramHeads - 1) / kNgramHeads), kGatherThreads, 0, stream>>>(
        table, block_scales, global_scale, table_rows, ids, n, out);
    return launched();
}

cere_status embed(const Weights &w, const int64_t *history, const int64_t *tokens, int64_t t, __nv_bfloat16 *emb,
                  cudaStream_t stream) {
    if (t <= 0 || t > int64_t(INT32_MAX) || check_table(w) != CERE_OK || history == nullptr || tokens == nullptr ||
        emb == nullptr || w.multipliers == nullptr || w.head_vocab_sizes == nullptr || w.head_offsets == nullptr)
        return CERE_ERR_ARGUMENT;
    return launch_embed(w, history, tokens, t, emb, stream);
}

size_t workspace_bytes(int64_t max_tokens) { return max_tokens <= 0 ? 0 : layout(nullptr, max_tokens).bytes; }

cere_status forward(const Weights &w, const Plans &plans, const State &state, const int64_t *tokens, int64_t t,
                    const __nv_bfloat16 *query, bool add_query, __nv_bfloat16 *out, void *workspace,
                    size_t workspace_size, cudaStream_t stream, const Capture *capture) {
    // Every check precedes the first launch: a rejected step leaves history, conv state and out untouched.
    cere_status s = check_step(w, plans, t, tokens, query, out, workspace, workspace_size);
    if (s != CERE_OK) return s;
    if (state.history == nullptr || state.conv_state == nullptr) return CERE_ERR_ARGUMENT;
    if (capture != nullptr && (capture->row < kConvStateLen || capture->row >= t || capture->history == nullptr ||
                               capture->conv_state == nullptr))
        return CERE_ERR_ARGUMENT;
    const Layout ws = layout(workspace, t);

    if ((s = launch_embed(w, state.history, tokens, t, ws.emb, stream)) != CERE_OK) return s;
    commit_history_kernel<<<1, 1, 0, stream>>>(state.history, tokens, t);
    if ((s = launched()) != CERE_OK) return s;
    if ((s = gate_rows(w, plans, ws, t, query, stream)) != CERE_OK) return s;
    if (capture != nullptr) {
        // The state as of row capture->row: history = its two previous tokens, conv state = the 9 gn rows before it.
        if (cudaMemcpyAsync(capture->history, tokens + capture->row - 2, 2 * sizeof(int64_t),
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            return CERE_ERR_INTERNAL;
        conv_state_at_kernel<<<(kChannels + kConvThreads - 1) / kConvThreads, kConvThreads, 0, stream>>>(
            ws.key, capture->row, capture->conv_state);
        if ((s = launched()) != CERE_OK) return s;
    }

    if (t > kConvTile) {
        // Prefill: the token-parallel outputs from the old state, then the new state = the last 9 gn rows (t >= 9),
        // as conv_kernel leaves it; stream order puts the state write after every read of the old state.
        const dim3 grid(unsigned((t + kConvTile - 1) / kConvTile), unsigned(kChannels / (kConvVec * kConvThreads)));
        conv_tile_kernel<<<grid, kConvThreads, 0, stream>>>(ws.key, ws.gated, w.conv_weight, state.conv_state, t,
                                                           query, add_query, out,
                                                           reinterpret_cast<uintptr_t>(out) % 8 == 0);
        if ((s = launched()) != CERE_OK) return s;
        conv_state_at_kernel<<<(kChannels + kConvThreads - 1) / kConvThreads, kConvThreads, 0, stream>>>(
            ws.key, t, state.conv_state);
        return launched();
    }
    conv_kernel<<<(kChannels + kConvThreads - 1) / kConvThreads, kConvThreads, 0, stream>>>(
        ws.key, ws.gated, w.conv_weight, state.conv_state, t, query, add_query, out);
    return launched();
}

cere_status forward_rows(const Weights &w, const Plans &plans, const RowsCall &call, const int64_t *tokens, int64_t m,
                         const __nv_bfloat16 *query, bool add_query, __nv_bfloat16 *out, void *workspace,
                         size_t workspace_size, cudaStream_t stream) {
    // Every check precedes the first launch, as forward's.
    if (m <= 0 || m > int64_t(kMaxRows)) return CERE_ERR_ARGUMENT;
    cere_status s = check_step(w, plans, m, tokens, query, out, workspace, workspace_size);
    if (s != CERE_OK) return s;
    if (call.sequences == 0 || call.sequences > uint32_t(m)) return CERE_ERR_ARGUMENT;
    RowsArgs a{};
    a.call = call;
    a.m = uint32_t(m);
    uint32_t next = 0;
    for (uint32_t q = 0; q < call.sequences; ++q) {
        const State &st = call.states[q];
        if (call.first[q] != next || call.count[q] == 0 || call.count[q] > uint32_t(m) - next ||
            st.history == nullptr || st.conv_state == nullptr)
            return CERE_ERR_ARGUMENT;
        for (uint32_t j = 0; j < call.count[q]; ++j) {
            a.seq[next + j] = uint8_t(q);
            a.j[next + j] = uint8_t(j);
        }
        next += call.count[q];
    }
    if (next != uint32_t(m)) return CERE_ERR_ARGUMENT;
    for (uint32_t r = 0; r < uint32_t(m); ++r) {
        if ((call.saves[r].history == nullptr) != (call.saves[r].conv_state == nullptr)) return CERE_ERR_ARGUMENT;
    }
    const Layout ws = layout(workspace, m);

    if (w.format == TableFormat::kNvfp4) {
        embed_rows_kernel<TableFormat::kNvfp4><<<unsigned(m), kGatherThreads, 0, stream>>>(a, tokens, w, ws.emb);
    } else {
        embed_rows_kernel<TableFormat::kFp8><<<unsigned(m), kGatherThreads, 0, stream>>>(a, tokens, w, ws.emb);
    }
    if ((s = launched()) != CERE_OK) return s;
    commit_rows_kernel<<<1, kMaxRows, 0, stream>>>(a, tokens);
    if ((s = launched()) != CERE_OK) return s;
    if ((s = gate_rows(w, plans, ws, m, query, stream)) != CERE_OK) return s;
    const dim3 grid((kChannels + kConvThreads - 1) / kConvThreads, call.sequences);
    conv_rows_kernel<<<grid, kConvThreads, 0, stream>>>(ws.key, ws.gated, w.conv_weight, a, query, add_query, out);
    return launched();
}

}  // namespace cere::k4
