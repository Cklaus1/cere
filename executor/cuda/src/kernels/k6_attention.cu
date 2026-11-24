// K6 full attention (m22 design note K6); contracts in k6_attention.h. The layer sequence and the K/V source per
// mode follow SGLang's Qwen3_5AttentionDecoderLayer / Qwen4ExpAttentionDecoderLayer.self_attention and
// QwenSparseAttnBackend.forward_extend / forward_decode (dense causal token set at context <= 2,048, spec Q4).
//
// math_reference: sglang
#include "kernels/k6_attention.h"

#include <cmath>
#include <type_traits>

#include <cuda_fp16.h>

#include "kernels/k1_primitives.h"
#include "kernels/p4_attention.h"

namespace cere::k6 {
namespace {

constexpr int kGroup = kQHeads / kKvHeads;  // query heads per KV head (12)
constexpr int kKvWidth = kKvHeads * kHeadDim;  // elements (or FP8 bytes) per token of K or V
constexpr int kQWidth = kQHeads * kHeadDim;
constexpr int kThreads = kHeadDim;  // one thread per output dimension in the P.V phase
constexpr int kKeyTile = 16;
// Padded float K rows: 16-byte aligned (1,040 bytes), and the 16 keys' 16-byte reads of a score step cover all 32
// banks in two wavefronts.
constexpr int kKStride = kHeadDim + 4;
constexpr int kSplitMaxTokens = 16;  // calls this short split the key range across blocks (decode, verify)
constexpr int kMaxSplits = 32;
constexpr int kKeysPerSplit = 128;
constexpr float kLog2e = 1.4426950408889634f;

static_assert(kThreads == 256 && kHeadDim % 16 == 0, "the tile loaders assume 256 threads and 16-dim chunks");

size_t align256(size_t n) { return (n + 255) & ~static_cast<size_t>(255); }

__host__ __device__ constexpr int split_count(int64_t tokens, int64_t kv_len) {
    if (tokens > kSplitMaxTokens) return 1;
    const int64_t s = (kv_len + kKeysPerSplit - 1) / kKeysPerSplit;
    return static_cast<int>(s < 1 ? 1 : (s > kMaxSplits ? kMaxSplits : s));
}

// Keys of each split: kv_len for one split, else ceil(kv_len / splits) rounded up to a key tile.
__host__ __device__ constexpr int64_t split_keys(int64_t kv_len, int splits) {
    return splits > 1 ? ((kv_len + splits - 1) / splits + kKeyTile - 1) / kKeyTile * kKeyTile : kv_len;
}

// The decode step's grid (P1, M23P1-04): the most splits of one token at a K6 kv_len.
constexpr int kStepSplits = split_count(1, kMaxContext);

size_t partial_bytes(int64_t tokens) {
    if (tokens > kSplitMaxTokens) return 0;
    const size_t rows = static_cast<size_t>(kMaxSplits) * static_cast<size_t>(tokens) * kQHeads;
    return align256(rows * kHeadDim * sizeof(float)) + 2 * align256(rows * sizeof(float));
}

template <int kTokens>
struct Smem {
    static constexpr int kRows = kTokens * kGroup;
    static constexpr size_t kQ = static_cast<size_t>(kRows) * kHeadDim * sizeof(float);
    static constexpr size_t kK = static_cast<size_t>(kKeyTile) * kKStride * sizeof(float);
    static constexpr size_t kV = static_cast<size_t>(kKeyTile) * kHeadDim * sizeof(float);
    static constexpr size_t kP = static_cast<size_t>(kKeyTile) * kRows * sizeof(float);
    static constexpr size_t kStats = 3 * static_cast<size_t>(kRows) * sizeof(float);
    static constexpr size_t kBytes = kQ + kK + kV + kP + kStats;
};

// 16 consecutive K or V elements of one key as loaded: two 16-byte words of BF16 or one of FP8 bytes. Loading and
// converting are separate so a tile's loads can be issued a tile ahead of its conversion (attention_kernel).
template <typename KvT>
struct Chunk {
    static constexpr int kWords = static_cast<int>(16 * sizeof(KvT) / sizeof(uint4));
    uint4 w[kWords];
};

template <typename KvT>
__device__ __forceinline__ Chunk<KvT> load_raw(const KvT* src) {
    Chunk<KvT> c;
#pragma unroll
    for (int i = 0; i < Chunk<KvT>::kWords; ++i) c.w[i] = reinterpret_cast<const uint4*>(src)[i];
    return c;
}

__device__ __forceinline__ void to_float(const Chunk<bf16>& c, float (&out)[16]) {
    const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&c.w[0]);
    const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&c.w[1]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f0 = __bfloat1622float2(h0[i]);
        const float2 f1 = __bfloat1622float2(h1[i]);
        out[2 * i] = f0.x;
        out[2 * i + 1] = f0.y;
        out[8 + 2 * i] = f1.x;
        out[8 + 2 * i + 1] = f1.y;
    }
}

// FP8 E4M3 -> float, exact (scale 1.0): NaN (magnitude 0x7f) is 0x7fc00000 with the sign; otherwise the 7 magnitude
// bits placed at float bit 20 are e (4 exponent bits) and m (3 mantissa bits) of the float 2^(e - 127) (1 + m/8), or
// for e 0 the denormal m 2^-129, and the exact product by 2^120 is the E4M3 value 2^(e - 7) (1 + m/8) or m 2^-9
// (this file compiles without flush-to-zero).
__device__ __forceinline__ void to_float(const Chunk<uint8_t>& c, float (&out)[16]) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&c.w[0]);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint32_t sign = (b[i] & 0x80u) << 24;
        const uint32_t mag = b[i] & 0x7fu;
        const float f = mag == 0x7fu ? __uint_as_float(0x7fc00000u) : __uint_as_float(mag << 20) * 0x1p120f;
        out[i] = __uint_as_float(__float_as_uint(f) | sign);
    }
}

// Block (query tile, KV head, key split). Rows r = tt * kGroup + hr: token t0 + tt, query head g * kGroup + hr.
// kStep (the decode graph's step, P1 M23P1-04): pos0 = *step_pos, and splits / keys_per_split come from it on the
// device exactly as attention_impl computes them on the host; the grid has kStepSplits splits, the blocks of splits
// past the step's count exit, and one split writes `out` directly as the eager launch without partials does.
// The one-token FP8-cache calls (decode step, MTP extend, eager calls of 1-3 tokens, verify chains) run
// split_rows_kernel, the same arithmetic per (row, key) and (row, output dimension). Calls of more than
// kSplitMaxTokens rows run prefill_kernel, this kernel's kTokens 4 arithmetic re-tiled.
template <int kTokens, typename KvT, bool kStep = false>
__global__ void __launch_bounds__(kThreads) attention_kernel(const bf16* __restrict__ q, int64_t tokens, int64_t pos0,
                                                             const KvT* __restrict__ k, int64_t k_stride,
                                                             const KvT* __restrict__ v, int64_t v_stride,
                                                             const int64_t* __restrict__ slots, float scale,
                                                             int64_t keys_per_split, bf16* __restrict__ out,
                                                             float* __restrict__ part_o, float* __restrict__ part_m,
                                                             float* __restrict__ part_l,
                                                             const int64_t* __restrict__ step_pos) {
    if constexpr (kStep) {
        pos0 = *step_pos;
        const int splits = split_count(tokens, pos0 + tokens);
        if (static_cast<int>(blockIdx.z) >= splits) return;
        keys_per_split = split_keys(pos0 + tokens, splits);
        if (splits == 1) part_o = nullptr;
    }
    using S = Smem<kTokens>;
    constexpr int kRows = S::kRows;
    static_assert(kRows % 2 == 0, "the softmax half-warps of one warp take the same branch");
    constexpr int kRowsPerThread = (kRows + 15) / 16;
    extern __shared__ __align__(16) unsigned char smem[];
    float* qs = reinterpret_cast<float*>(smem);
    float* ks = reinterpret_cast<float*>(smem + S::kQ);
    float* vs = reinterpret_cast<float*>(smem + S::kQ + S::kK);
    float* ps = reinterpret_cast<float*>(smem + S::kQ + S::kK + S::kV);
    float* row_m = reinterpret_cast<float*>(smem + S::kQ + S::kK + S::kV + S::kP);
    float* row_l = row_m + kRows;
    float* row_alpha = row_l + kRows;

    const int tid = threadIdx.x;
    const int64_t t0 = static_cast<int64_t>(blockIdx.x) * kTokens;
    const int g = blockIdx.y;
    const int split = blockIdx.z;
    const int64_t last_token = (t0 + kTokens <= tokens ? t0 + kTokens : tokens) - 1;
    const int64_t max_pos = pos0 + last_token;
    const int64_t key_begin = static_cast<int64_t>(split) * keys_per_split;
    int64_t key_end = key_begin + keys_per_split;
    if (key_end > max_pos + 1) key_end = max_pos + 1;

    // K/V loads run one key tile ahead of the tile computed and the slot lookups two, so a tile's DRAM latency
    // overlaps the previous tile's math; the arithmetic per element is that of a load-then-compute loop. This thread
    // loads 16 dims of key `lkey` of each tile; a key at or past key_end holds zeros (its probabilities are 0).
    const int load_key = tid >> 4;
    const int load_dim = (tid & 15) * 16;
    const int64_t kv_dim = static_cast<int64_t>(g) * kHeadDim + load_dim;
    int64_t lkey = key_begin + load_key;
    Chunk<KvT> k_cur{}, v_cur{};
    if (lkey < key_end) {
        const int64_t row = slots ? slots[lkey] : lkey;
        k_cur = load_raw(k + row * k_stride + kv_dim);
        v_cur = load_raw(v + row * v_stride + kv_dim);
    }
    int64_t next_row = 0;
    if (lkey + kKeyTile < key_end) next_row = slots ? slots[lkey + kKeyTile] : lkey + kKeyTile;

    // Q tile as float (exact BF16 widening, once per block): rows beyond the last token repeat it (finite, never
    // stored).
    for (int idx = tid; idx < kRows * (kHeadDim / 8); idx += kThreads) {
        const int r = idx / (kHeadDim / 8);
        const int c = (idx % (kHeadDim / 8)) * 8;
        int64_t t = t0 + r / kGroup;
        if (t > last_token) t = last_token;
        const int h = g * kGroup + r % kGroup;
        const uint4 raw = *reinterpret_cast<const uint4*>(q + t * kQWidth + static_cast<int64_t>(h) * kHeadDim + c);
        const __nv_bfloat162* qh = reinterpret_cast<const __nv_bfloat162*>(&raw);
        const float2 f0 = __bfloat1622float2(qh[0]), f1 = __bfloat1622float2(qh[1]);
        const float2 f2 = __bfloat1622float2(qh[2]), f3 = __bfloat1622float2(qh[3]);
        *reinterpret_cast<float4*>(qs + r * kHeadDim + c) = make_float4(f0.x, f0.y, f1.x, f1.y);
        *reinterpret_cast<float4*>(qs + r * kHeadDim + c + 4) = make_float4(f2.x, f2.y, f3.x, f3.y);
    }
    for (int r = tid; r < kRows; r += kThreads) {
        row_m[r] = -INFINITY;
        row_l[r] = 0.0f;
    }

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.0f;

    const float score_scale = scale * kLog2e;
    const int lj = tid & (kKeyTile - 1);  // key of this thread in the score phase
    const int lrg = tid >> 4;             // row group of this thread in the score phase
    __syncthreads();

    for (int64_t key0 = key_begin; key0 < key_end; key0 += kKeyTile, lkey += kKeyTile) {
        Chunk<KvT> k_next{}, v_next{};
        if (lkey + kKeyTile < key_end) {
            k_next = load_raw(k + next_row * k_stride + kv_dim);
            v_next = load_raw(v + next_row * v_stride + kv_dim);
        }
        if (lkey + 2 * kKeyTile < key_end) next_row = slots ? slots[lkey + 2 * kKeyTile] : lkey + 2 * kKeyTile;
        {  // K/V tile, converted to float
            float kv[16];
            to_float(k_cur, kv);
#pragma unroll
            for (int i = 0; i < 16; i += 4)
                *reinterpret_cast<float4*>(ks + load_key * kKStride + load_dim + i) =
                    make_float4(kv[i], kv[i + 1], kv[i + 2], kv[i + 3]);
            to_float(v_cur, kv);
#pragma unroll
            for (int i = 0; i < 16; i += 4)
                *reinterpret_cast<float4*>(vs + load_key * kHeadDim + load_dim + i) =
                    make_float4(kv[i], kv[i + 1], kv[i + 2], kv[i + 3]);
        }
        k_cur = k_next;
        v_cur = v_next;
        __syncthreads();

        {  // scores (log2 domain), causal mask
            float dot[kRowsPerThread];
#pragma unroll
            for (int m = 0; m < kRowsPerThread; ++m) dot[m] = 0.0f;
            const float* kr = ks + lj * kKStride;
            // Row lrg + 16 m of this thread exists for m = 0 iff any does. Each batch reads 16 key dims and 16 query
            // dims per row before its fmafs, which run d ascending per row as a d-at-a-time loop would.
            if (lrg < kRows) {
#pragma unroll 2
                for (int d0 = 0; d0 < kHeadDim; d0 += 16) {
                    float4 kb[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) kb[i] = *reinterpret_cast<const float4*>(kr + d0 + 4 * i);
#pragma unroll
                    for (int m = 0; m < kRowsPerThread; ++m) {
                        const int r = lrg + 16 * m;
                        if (r < kRows) {
                            const float* qr = qs + r * kHeadDim + d0;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const float4 qv = *reinterpret_cast<const float4*>(qr + 4 * i);
                                dot[m] = fmaf(qv.x, kb[i].x, dot[m]);
                                dot[m] = fmaf(qv.y, kb[i].y, dot[m]);
                                dot[m] = fmaf(qv.z, kb[i].z, dot[m]);
                                dot[m] = fmaf(qv.w, kb[i].w, dot[m]);
                            }
                        }
                    }
                }
            }
            // Online softmax of row r = lrg + 16 m over this tile's 16 keys, in the 16 lanes (one per key j = lj)
            // that hold the row's scores (kRows is even, so a warp's two half-warps run the same m). The arithmetic is
            // that of one thread per row: s_j = dot * scale (log2 domain) or -inf when masked; tile max (fmaxf is
            // exact, so the lane tree equals the j-ascending fold up to the sign of a zero, which no later value
            // depends on: m only enters as exp2f(x - m)); p_j = exp2f(s_j - m_new); sum = p_0 + ... + p_15 in j order.
            const int64_t key = key0 + lj;
#pragma unroll
            for (int m = 0; m < kRowsPerThread; ++m) {
                const int r = lrg + 16 * m;
                if (r < kRows) {
                    int64_t t = t0 + r / kGroup;
                    if (t > last_token) t = last_token;
                    const bool visible = key < key_end && key <= pos0 + t;
                    const float s = visible ? __fmul_rn(dot[m], score_scale) : -INFINITY;
                    float tile_max = fmaxf(-INFINITY, s);
#pragma unroll
                    for (int o = 8; o > 0; o >>= 1)
                        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, o));
                    const float m_old = row_m[r];
                    const float m_new = fmaxf(m_old, tile_max);
                    const bool empty = m_new == -INFINITY;
                    const float p = empty ? 0.0f : exp2f(__fsub_rn(s, m_new));
                    ps[lj * kRows + r] = p;
                    float sum = 0.0f;
#pragma unroll
                    for (int j = 0; j < kKeyTile; ++j) sum += __shfl_sync(0xffffffffu, p, (tid & 16) + j);
                    if (lj == 0) {
                        const float alpha = empty ? 1.0f : exp2f(m_old - m_new);
                        row_m[r] = m_new;
                        row_l[r] = fmaf(row_l[r], alpha, empty ? 0.0f : sum);  // the one-thread form's contraction
                        row_alpha[r] = alpha;
                    }
                }
            }
        }
        __syncthreads();

        {  // acc = acc * alpha + P . V for this thread's output dimension
            const int d = tid;
#pragma unroll
            for (int r = 0; r < kRows; r += 4) {
                const float4 a = *reinterpret_cast<const float4*>(row_alpha + r);
                acc[r] *= a.x;
                acc[r + 1] *= a.y;
                acc[r + 2] *= a.z;
                acc[r + 3] *= a.w;
            }
#pragma unroll 4
            for (int j = 0; j < kKeyTile; ++j) {
                const float vv = vs[j * kHeadDim + d];
                const float* pj = ps + j * kRows;
#pragma unroll
                for (int r = 0; r < kRows; r += 4) {
                    const float4 p = *reinterpret_cast<const float4*>(pj + r);
                    acc[r] = fmaf(p.x, vv, acc[r]);
                    acc[r + 1] = fmaf(p.y, vv, acc[r + 1]);
                    acc[r + 2] = fmaf(p.z, vv, acc[r + 2]);
                    acc[r + 3] = fmaf(p.w, vv, acc[r + 3]);
                }
            }
        }
        __syncthreads();
    }

    const int d = tid;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const int64_t t = t0 + r / kGroup;
        if (t > last_token) continue;
        const int h = g * kGroup + r % kGroup;
        if (part_o == nullptr) {
            out[t * kQWidth + static_cast<int64_t>(h) * kHeadDim + d] = __float2bfloat16_rn(acc[r] / row_l[r]);
        } else {
            const int64_t row = (static_cast<int64_t>(split) * tokens + t) * kQHeads + h;
            part_o[row * kHeadDim + d] = acc[r];
            if (d == 0) {
                part_m[row] = row_m[r];
                part_l[row] = row_l[r];
            }
        }
    }
}

// The prefill kernel (calls of more than kSplitMaxTokens rows: one split, keys 0 .. max_pos of a 4-token tile): the
// arithmetic of attention_kernel<4> per (row, key) and per (row, output dimension) with its 4-token query tiles
// (key_end, the causal mask and the zero K/V past key_end unchanged) and its 16-key softmax tiles, the work spread
// for throughput:
//   - block (KV head x, tile gridDim.y - 1 - y): the longest causal tiles are dispatched first (LPT, no tail wave);
//   - a step stages 32 keys, two softmax tiles: thread (lrg, lj) runs the fmaf chains (d ascending from 0) of rows
//     lrg + 16 m with keys lj and 16 + lj, then tile 0's online softmax and, when tile 1 starts before key_end,
//     tile 1's, each attention_kernel's softmax block verbatim (tile 1 reads the m / l tile 0's lane 0 wrote, as the
//     next attention_kernel iteration does); a tile at or past key_end is not run, as attention_kernel's loop ends;
//   - K then V of a step share one float buffer, each loaded a step ahead in registers (slots two steps ahead);
//   - P.V thread (12 rows, 4 output dims): per tile acc *= alpha, then acc = fmaf(p_j, v_j, acc), j ascending.
// Q / K / V rows padded to kPadStride floats: a score load's two rows (K: eight keys) fall in distinct banks.
constexpr int kPrefillTokens = 4;  // attention_kernel<4>'s query tile
constexpr int kPrefillRows = kPrefillTokens * kGroup;
constexpr int kStepKeys = 2 * kKeyTile;
constexpr int kPadStride = kHeadDim + 4;
constexpr int kPStride = kPrefillRows + 4;  // probabilities of one key over the rows: 2-way store conflicts at most
constexpr int kPvRows = 12;                 // P.V rows per thread
static_assert(kPrefillRows == 3 * 16 && kThreads == (kPrefillRows / kPvRows) * (kHeadDim / 4) && kPvRows % 4 == 0,
              "score thread = (row group of 3 rows, key); P.V thread = (12 rows, 4 dims)");

struct PrefillSmem {
    static constexpr size_t kQ = static_cast<size_t>(kPrefillRows) * kPadStride * sizeof(float);
    static constexpr size_t kKV = static_cast<size_t>(kStepKeys) * kPadStride * sizeof(float);
    static constexpr size_t kP = static_cast<size_t>(kStepKeys) * kPStride * sizeof(float);
    static constexpr size_t kStats = 4 * static_cast<size_t>(kPrefillRows) * sizeof(float);  // m, l, alpha x 2 tiles
    static constexpr size_t kBytes = kQ + kKV + kP + kStats;
};
static_assert(PrefillSmem::kQ % 16 == 0 && PrefillSmem::kKV % 16 == 0 && PrefillSmem::kP % 16 == 0 &&
                  PrefillSmem::kBytes <= 99 * 1024,
              "16-byte aligned regions within one sm_120 block's shared memory");

// 8 K or V elements of one key at `src` and 8 at src + kHeadDim / 2, as one Chunk: to_float gives out[0..7] and
// out[8..15] (each element's float is the exact widening to_float makes of it anywhere).
template <typename KvT>
__device__ __forceinline__ Chunk<KvT> load_halves(const KvT* src) {
    Chunk<KvT> c;
    if constexpr (std::is_same_v<KvT, bf16>) {
        c.w[0] = *reinterpret_cast<const uint4*>(src);
        c.w[1] = *reinterpret_cast<const uint4*>(src + kHeadDim / 2);
    } else {
        const uint2 lo = *reinterpret_cast<const uint2*>(src);
        const uint2 hi = *reinterpret_cast<const uint2*>(src + kHeadDim / 2);
        c.w[0] = make_uint4(lo.x, lo.y, hi.x, hi.y);
    }
    return c;
}

template <typename KvT>
__device__ __forceinline__ void store_halves(const Chunk<KvT>& c, float* dst) {
    float f[16];
    to_float(c, f);
    *reinterpret_cast<float4*>(dst) = make_float4(f[0], f[1], f[2], f[3]);
    *reinterpret_cast<float4*>(dst + 4) = make_float4(f[4], f[5], f[6], f[7]);
    *reinterpret_cast<float4*>(dst + kHeadDim / 2) = make_float4(f[8], f[9], f[10], f[11]);
    *reinterpret_cast<float4*>(dst + kHeadDim / 2 + 4) = make_float4(f[12], f[13], f[14], f[15]);
}

template <typename KvT>
__global__ void __launch_bounds__(kThreads) prefill_kernel(const bf16* __restrict__ q, int64_t tokens, int64_t pos0,
                                                           const KvT* __restrict__ k, int64_t k_stride,
                                                           const KvT* __restrict__ v, int64_t v_stride,
                                                           const int64_t* __restrict__ slots, float scale,
                                                           bf16* __restrict__ out) {
    using S = PrefillSmem;
    constexpr int kRows = kPrefillRows;
    extern __shared__ __align__(16) unsigned char smem[];
    float* qs = reinterpret_cast<float*>(smem);
    float* kvs = reinterpret_cast<float*>(smem + S::kQ);
    float* ps = reinterpret_cast<float*>(smem + S::kQ + S::kKV);
    float* row_m = reinterpret_cast<float*>(smem + S::kQ + S::kKV + S::kP);
    float* row_l = row_m + kRows;
    float* row_alpha = row_l + kRows;  // [2][kRows]: the step's tiles 0 and 1

    const int tid = threadIdx.x;
    const int g = blockIdx.x;
    const int64_t t0 = static_cast<int64_t>(gridDim.y - 1 - blockIdx.y) * kPrefillTokens;
    const int64_t last_token = (t0 + kPrefillTokens <= tokens ? t0 + kPrefillTokens : tokens) - 1;
    const int64_t key_end = pos0 + last_token + 1;  // attention_kernel's at one split (keys_per_split = kv_len)

    // This thread loads dims load_dim .. +7 and +128 .. +135 of keys load_key and 16 + load_key of each step.
    const int load_key = tid >> 4;
    const int load_dim = (tid & 15) * 8;
    const int64_t kv_dim = static_cast<int64_t>(g) * kHeadDim + load_dim;
    Chunk<KvT> kc[2]{}, vc[2]{};
    int64_t row1[2] = {0, 0};  // cache rows of this thread's keys of the next step
    int64_t row2[2] = {0, 0};  // and of the step after
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int64_t key = load_key + i * kKeyTile;
        if (key < key_end) {
            const int64_t row = slots ? slots[key] : key;
            kc[i] = load_halves(k + row * k_stride + kv_dim);
            vc[i] = load_halves(v + row * v_stride + kv_dim);
        }
        if (key + kStepKeys < key_end) row1[i] = slots ? slots[key + kStepKeys] : key + kStepKeys;
    }

    // Q tile as float, as attention_kernel's (rows beyond the last token repeat it; never stored).
    for (int idx = tid; idx < kRows * (kHeadDim / 8); idx += kThreads) {
        const int r = idx / (kHeadDim / 8);
        const int c = (idx % (kHeadDim / 8)) * 8;
        int64_t t = t0 + r / kGroup;
        if (t > last_token) t = last_token;
        const int h = g * kGroup + r % kGroup;
        const uint4 raw = *reinterpret_cast<const uint4*>(q + t * kQWidth + static_cast<int64_t>(h) * kHeadDim + c);
        const __nv_bfloat162* qh = reinterpret_cast<const __nv_bfloat162*>(&raw);
        const float2 f0 = __bfloat1622float2(qh[0]), f1 = __bfloat1622float2(qh[1]);
        const float2 f2 = __bfloat1622float2(qh[2]), f3 = __bfloat1622float2(qh[3]);
        *reinterpret_cast<float4*>(qs + r * kPadStride + c) = make_float4(f0.x, f0.y, f1.x, f1.y);
        *reinterpret_cast<float4*>(qs + r * kPadStride + c + 4) = make_float4(f2.x, f2.y, f3.x, f3.y);
    }
    for (int r = tid; r < kRows; r += kThreads) {
        row_m[r] = -INFINITY;
        row_l[r] = 0.0f;
    }

    float acc[kPvRows][4];
#pragma unroll
    for (int r = 0; r < kPvRows; ++r) acc[r][0] = acc[r][1] = acc[r][2] = acc[r][3] = 0.0f;

    const float score_scale = scale * kLog2e;
    const int lj = tid & (kKeyTile - 1);  // key of this thread in each score tile
    const int lrg = tid >> 4;             // rows lrg + 16 m of this thread in the score phase
    const int pv_r0 = (tid >> 6) * kPvRows;  // P.V rows pv_r0 .. +11 (warp-uniform)
    const int pv_d = (tid & 63) * 4;          // P.V output dims pv_d .. +3

    for (int64_t key0 = 0; key0 < key_end; key0 += kStepKeys) {
        const bool two = key0 + kKeyTile < key_end;  // tile 1 of the step runs
        // K of the step; the next step's K loads and the slots of the step after.
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            store_halves(kc[i], kvs + (load_key + i * kKeyTile) * kPadStride + load_dim);
            const int64_t key = key0 + kStepKeys + load_key + i * kKeyTile;
            kc[i] = Chunk<KvT>{};
            if (key < key_end) kc[i] = load_halves(k + row1[i] * k_stride + kv_dim);
            if (key + kStepKeys < key_end) row2[i] = slots ? slots[key + kStepKeys] : key + kStepKeys;
        }
        __syncthreads();

        {  // scores (log2 domain), causal mask, online softmax of tile 0 then tile 1
            float dot[2][3];
#pragma unroll
            for (int i = 0; i < 2; ++i) dot[i][0] = dot[i][1] = dot[i][2] = 0.0f;
            const float* kr0 = kvs + lj * kPadStride;
            const float* kr1 = kvs + (kKeyTile + lj) * kPadStride;
#pragma unroll 4
            for (int d0 = 0; d0 < kHeadDim; d0 += 8) {
                float4 ka[2], kb[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    ka[i] = *reinterpret_cast<const float4*>(kr0 + d0 + 4 * i);
                    kb[i] = *reinterpret_cast<const float4*>(kr1 + d0 + 4 * i);
                }
#pragma unroll
                for (int m = 0; m < 3; ++m) {
                    const float* qr = qs + (lrg + 16 * m) * kPadStride + d0;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const float4 qv = *reinterpret_cast<const float4*>(qr + 4 * i);
                        dot[0][m] = fmaf(qv.x, ka[i].x, dot[0][m]);
                        dot[0][m] = fmaf(qv.y, ka[i].y, dot[0][m]);
                        dot[0][m] = fmaf(qv.z, ka[i].z, dot[0][m]);
                        dot[0][m] = fmaf(qv.w, ka[i].w, dot[0][m]);
                        dot[1][m] = fmaf(qv.x, kb[i].x, dot[1][m]);
                        dot[1][m] = fmaf(qv.y, kb[i].y, dot[1][m]);
                        dot[1][m] = fmaf(qv.z, kb[i].z, dot[1][m]);
                        dot[1][m] = fmaf(qv.w, kb[i].w, dot[1][m]);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                if (i == 0 || two) {
                    const int64_t key = key0 + i * kKeyTile + lj;
#pragma unroll
                    for (int m = 0; m < 3; ++m) {
                        const int r = lrg + 16 * m;
                        int64_t t = t0 + r / kGroup;
                        if (t > last_token) t = last_token;
                        const bool visible = key < key_end && key <= pos0 + t;
                        const float s = visible ? __fmul_rn(dot[i][m], score_scale) : -INFINITY;
                        float tile_max = fmaxf(-INFINITY, s);
#pragma unroll
                        for (int o = 8; o > 0; o >>= 1)
                            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, o));
                        const float m_old = row_m[r];
                        const float m_new = fmaxf(m_old, tile_max);
                        const bool empty = m_new == -INFINITY;
                        const float p = empty ? 0.0f : exp2f(__fsub_rn(s, m_new));
                        ps[(i * kKeyTile + lj) * kPStride + r] = p;
                        float sum = 0.0f;
#pragma unroll
                        for (int j = 0; j < kKeyTile; ++j) sum += __shfl_sync(0xffffffffu, p, (tid & 16) + j);
                        if (lj == 0) {
                            const float alpha = empty ? 1.0f : exp2f(m_old - m_new);
                            row_m[r] = m_new;
                            row_l[r] = fmaf(row_l[r], alpha, empty ? 0.0f : sum);
                            row_alpha[i * kRows + r] = alpha;
                        }
                    }
                    __syncwarp();  // tile 1's row_m reads see tile 0's lane-0 writes (rows of this half-warp)
                }
            }
        }
        __syncthreads();

        // V of the step (over K); the next step's V loads.
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            store_halves(vc[i], kvs + (load_key + i * kKeyTile) * kPadStride + load_dim);
            const int64_t key = key0 + kStepKeys + load_key + i * kKeyTile;
            vc[i] = Chunk<KvT>{};
            if (key < key_end) vc[i] = load_halves(v + row1[i] * v_stride + kv_dim);
            row1[i] = row2[i];
        }
        __syncthreads();

        // acc = acc * alpha + P . V per tile, for rows pv_r0 .. +11 and dims pv_d .. +3
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            if (i == 0 || two) {
#pragma unroll
                for (int r = 0; r < kPvRows; r += 4) {
                    const float4 a = *reinterpret_cast<const float4*>(row_alpha + i * kRows + pv_r0 + r);
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        acc[r][c] *= a.x;
                        acc[r + 1][c] *= a.y;
                        acc[r + 2][c] *= a.z;
                        acc[r + 3][c] *= a.w;
                    }
                }
#pragma unroll 4
                for (int j = 0; j < kKeyTile; ++j) {
                    const float4 vv = *reinterpret_cast<const float4*>(kvs + (i * kKeyTile + j) * kPadStride + pv_d);
                    const float* pj = ps + (i * kKeyTile + j) * kPStride + pv_r0;
#pragma unroll
                    for (int r = 0; r < kPvRows; r += 4) {
                        const float4 p = *reinterpret_cast<const float4*>(pj + r);
                        const float pr[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            acc[r + u][0] = fmaf(pr[u], vv.x, acc[r + u][0]);
                            acc[r + u][1] = fmaf(pr[u], vv.y, acc[r + u][1]);
                            acc[r + u][2] = fmaf(pr[u], vv.z, acc[r + u][2]);
                            acc[r + u][3] = fmaf(pr[u], vv.w, acc[r + u][3]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int u = 0; u < kPvRows; ++u) {
        const int r = pv_r0 + u;
        const int64_t t = t0 + r / kGroup;
        if (t > last_token) continue;
        const int h = g * kGroup + r % kGroup;
        const float l = row_l[r];
        bf16* o = out + t * kQWidth + static_cast<int64_t>(h) * kHeadDim + pv_d;
#pragma unroll
        for (int c = 0; c < 4; ++c) o[c] = __float2bfloat16_rn(acc[u][c] / l);
    }
}

// The one-token split block for the FP8 cache (AttnSplit): the arithmetic per (row, key) and per (row, output
// dimension) of attention_kernel's one-token tile loop (the merge-base kernel for these calls), with the work of a
// split spread across threads instead of a tile-serial loop; split boundaries, partials and the combine unchanged.
// Block (token t, KV head g x row group, split): kR of the KV head's kGroup query heads.
// kStep: as attention_kernel's. kChain (verify_attention = batched, m23 design note P11R-19): block x = t runs the
// one-token call at pos0 + t, its splits and keys per split computed on the device exactly as attention_impl computes
// them for tokens 1 at kv_len pos0 + t + 1; the grid has at least the last row's (the most) splits, a row's blocks
// past its count exit, and a row of one split writes `out` directly. The partials index (split * tokens + t) as the
// multi-token call does. A non-null step_pos (the verify graph's chain, P11R-20) gives pos0 = *step_pos.
// Per block of up to kBlockKeys keys (one block whenever keys_per_split <= 128, every call of fewer than 4 tokens):
//   1. thread (key, row half) loads its key's slot; the key's V half row (and with kSmemK its K half row) goes to
//      shared memory by cp.async, zero-filled at or past key_end as attention_kernel's tiles (without kSmemK the
//      thread loads its key's whole K row into registers, zeros past key_end);
//   2. scores: dot = fmaf chain over d ascending from 0, s = dot * score_scale or -inf past key_end (visible is
//      key < key_end for one token: key_end <= pos0 + t + 1);
//   3. tile t's max over its 16 keys (the same half-warp xor tree), the prefix m_t = fmaxf(m_{t-1}, max_t) from the
//      carried m, p = exp2f(s - m_t) (0 when m_t is -inf), the tile sum p_0 + ... + p_15 in key order (the same
//      shuffle loop);
//   4. one thread per row runs the alpha / l recurrence over the tiles in order;
//   5. thread (output dimension pair, row half): per tile acc *= alpha_t, then acc = fmaf(p_j, v_j, acc), j ascending.
// FP8 E4M3 -> half (cvt.rn.f16x2.e4m3x2) -> float is exact for every finite E4M3 value (E4M3 has no infinity); a NaN
// byte gives a NaN as to_float does, and every K / V value enters an fmaf whose NaN result is the canonical NaN.
constexpr int kBlockKeys = 128;
constexpr int kTilesPerBlock = kBlockKeys / kKeyTile;
// Query heads per block (a divisor of kGroup, even): kKvHeads * kGroup / kR blocks per (token, split). Measured (12
// layer graphs, L2 flushed, 1.2K-2K keys): a decode step / MTP extend / eager call of 1-3 tokens runs
// kStepRows with K in shared memory (2 KV heads x 3 groups x <= 16 splits, one wave); a verify chain of up to
// kChainFewMax rows (the speculation window) kChainRowsFew and a longer chain kChainRowsMany, both with K in registers
// (rows x 4 or 2 groups x splits blocks exceed a wave, and the smaller block fits twice on an SM).
constexpr int kStepRows = 4;
constexpr int kChainRowsFew = 6;
constexpr int kChainRowsMany = 12;
constexpr int kChainFewMax = 4;
static_assert(kThreads == 2 * kBlockKeys && kHeadDim == 2 * kBlockKeys, "thread = (key, row half) = (dim pair, row half)");
// K rows in shared memory (kSmemK): 256 FP8 bytes padded to 272, so the 16-byte reads of 8 consecutive keys hit
// distinct banks. Without kSmemK each thread loads its key's K row into registers (half the shared memory, so two
// blocks fit on an SM: the verify chain's grid exceeds one wave).
constexpr int kKRowBytes = kHeadDim + 16;

template <int kR, bool kSmemK>
struct RowsSmem {
    static constexpr size_t kQ = static_cast<size_t>(kR) * kHeadDim * sizeof(float);
    static constexpr size_t kK = kSmemK ? static_cast<size_t>(kBlockKeys) * kKRowBytes : 0;  // FP8, padded rows
    static constexpr size_t kV = static_cast<size_t>(kBlockKeys) * kHeadDim;                 // FP8 bytes
    static constexpr size_t kP = static_cast<size_t>(kR) * kBlockKeys * sizeof(float);
    static constexpr size_t kTile = static_cast<size_t>(kR) * kTilesPerBlock * sizeof(float);
    static constexpr size_t kBytes = kQ + kK + kV + kP + 4 * kTile + 2 * static_cast<size_t>(kR) * sizeof(float);
};

__device__ __forceinline__ float2 fp8x2_to_float2(uint32_t two) {
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(static_cast<unsigned short>(two)));
    return __half22float2(*reinterpret_cast<const __half2*>(&h2));
}

__device__ __forceinline__ void fp8x16_to_float(const uint4& w, float (&out)[16]) {
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 lo = fp8x2_to_float2(ws[i] & 0xffffu);
        const float2 hi = fp8x2_to_float2(ws[i] >> 16);
        out[4 * i] = lo.x;
        out[4 * i + 1] = lo.y;
        out[4 * i + 2] = hi.x;
        out[4 * i + 3] = hi.y;
    }
}

// kRowKv (rows16 design note R16-C, with kChain and step_pos): block row t's caches and slot map are rkv.*[t] and
// its position step_pos[t] (pos0 = step_pos[t] - t below, so pos0 + t is the row's position): the chains of several
// sequences in one launch, each row's block the one of its own sequence's chain launch.
struct NoRows {};
template <bool kRowKv>
using RowKvArg = std::conditional_t<kRowKv, k1::KvRows, NoRows>;

template <int kR, bool kSmemK, bool kStep, bool kChain, bool kRowKv = false>
__global__ void __launch_bounds__(kThreads) split_rows_kernel(const bf16* __restrict__ q, int64_t tokens, int64_t pos0,
                                                              const uint8_t* __restrict__ k,
                                                              const uint8_t* __restrict__ v,
                                                              const int64_t* __restrict__ slots, float scale,
                                                              int64_t keys_per_split, bf16* __restrict__ out,
                                                              float* __restrict__ part_o, float* __restrict__ part_m,
                                                              float* __restrict__ part_l,
                                                              const int64_t* __restrict__ step_pos,
                                                              const RowKvArg<kRowKv> rkv) {
    static_assert(kGroup % kR == 0 && kR % 2 == 0, "row groups split the KV head's query heads into row halves");
    static_assert(!kRowKv || kChain, "per-row caches are the multi-sequence chain's");
    constexpr int kRG = kGroup / kR;
    constexpr int kRT = kR / 2;  // rows per thread
    if constexpr (kStep) {
        pos0 = *step_pos;
        const int splits = split_count(tokens, pos0 + tokens);
        if (static_cast<int>(blockIdx.z) >= splits) return;
        keys_per_split = split_keys(pos0 + tokens, splits);
        if (splits == 1) part_o = nullptr;
    }
    if constexpr (kRowKv) {
        k = rkv.k_cache[blockIdx.x];
        v = rkv.v_cache[blockIdx.x];
        slots = rkv.slots[blockIdx.x];
    }
    if constexpr (kChain) {
        if constexpr (kRowKv) {
            pos0 = step_pos[blockIdx.x] - static_cast<int64_t>(blockIdx.x);
        } else if (step_pos != nullptr) {
            pos0 = *step_pos;
        }
        const int64_t kv_len = pos0 + static_cast<int64_t>(blockIdx.x) + 1;
        const int splits = split_count(1, kv_len);
        if (static_cast<int>(blockIdx.z) >= splits) return;
        keys_per_split = split_keys(kv_len, splits);
        if (splits == 1) part_o = nullptr;
    }
    using S = RowsSmem<kR, kSmemK>;
    extern __shared__ __align__(16) unsigned char smem[];
    float* qs = reinterpret_cast<float*>(smem);
    uint8_t* ks = smem + S::kQ;
    uint8_t* vs = ks + S::kK;
    float* ps = reinterpret_cast<float*>(vs + S::kV);
    float* tile_max = ps + kR * kBlockKeys;
    float* tile_m = tile_max + kR * kTilesPerBlock;
    float* tile_sum = tile_m + kR * kTilesPerBlock;
    float* tile_alpha = tile_sum + kR * kTilesPerBlock;
    float* row_m = tile_alpha + kR * kTilesPerBlock;
    float* row_l = row_m + kR;

    const int tid = threadIdx.x;
    const int lk = tid & (kBlockKeys - 1);  // key of this thread (scores) / output dimension pair (P.V)
    const int half = tid >> 7;              // row half: rows half * kRT .. + kRT - 1
    const int64_t t = blockIdx.x;
    const int g = static_cast<int>(blockIdx.y) / kRG;
    const int h0 = g * kGroup + (static_cast<int>(blockIdx.y) % kRG) * kR;  // query head of row 0
    const int split = blockIdx.z;
    const int64_t key_begin = static_cast<int64_t>(split) * keys_per_split;
    int64_t key_end = key_begin + keys_per_split;
    if (key_end > pos0 + t + 1) key_end = pos0 + t + 1;

    const int64_t kv_off = static_cast<int64_t>(g) * kHeadDim;
    // The first block's slot, before the Q loads.
    int64_t row = 0;
    if (key_begin + lk < key_end) row = slots ? slots[key_begin + lk] : key_begin + lk;

    // Q rows as float (exact BF16 widening).
    for (int idx = tid; idx < kR * (kHeadDim / 8); idx += kThreads) {
        const int r = idx / (kHeadDim / 8);
        const int c = (idx % (kHeadDim / 8)) * 8;
        const uint4 raw =
            *reinterpret_cast<const uint4*>(q + t * kQWidth + static_cast<int64_t>(h0 + r) * kHeadDim + c);
        const __nv_bfloat162* qh = reinterpret_cast<const __nv_bfloat162*>(&raw);
        const float2 f0 = __bfloat1622float2(qh[0]), f1 = __bfloat1622float2(qh[1]);
        const float2 f2 = __bfloat1622float2(qh[2]), f3 = __bfloat1622float2(qh[3]);
        *reinterpret_cast<float4*>(qs + r * kHeadDim + c) = make_float4(f0.x, f0.y, f1.x, f1.y);
        *reinterpret_cast<float4*>(qs + r * kHeadDim + c + 4) = make_float4(f2.x, f2.y, f3.x, f3.y);
    }
    if (tid < kR) {
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.0f;
    }

    float acc[kRT][2];
#pragma unroll
    for (int m = 0; m < kRT; ++m) acc[m][0] = acc[m][1] = 0.0f;
    const float score_scale = scale * kLog2e;
    const uint32_t ks_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ks));
    const uint32_t vs_addr = static_cast<uint32_t>(__cvta_generic_to_shared(vs));

    for (int64_t kb = key_begin; kb < key_end; kb += kBlockKeys) {
        const int64_t rem = key_end - kb;
        const int tiles = rem >= kBlockKeys ? kTilesPerBlock : static_cast<int>((rem + kKeyTile - 1) / kKeyTile);
        const bool valid = kb + lk < key_end;
        {  // this key's V (and with kSmemK K) half row -> shared memory (zero past key_end)
            const int64_t off = row * kKvWidth + kv_off + half * (kHeadDim / 2);
            const int vsz = valid ? 16 : 0;
            if constexpr (kSmemK) {
                const uint32_t kdst = ks_addr + static_cast<uint32_t>(lk * kKRowBytes + half * (kHeadDim / 2));
#pragma unroll
                for (int i = 0; i < kHeadDim / 2 / 16; ++i)
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(kdst + 16 * i),
                                 "l"(k + off + 16 * i), "r"(vsz));
                asm volatile("cp.async.commit_group;\n" ::);
            }
            const uint32_t vdst = vs_addr + static_cast<uint32_t>(lk * kHeadDim + half * (kHeadDim / 2));
#pragma unroll
            for (int i = 0; i < kHeadDim / 2 / 16; ++i)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(vdst + 16 * i),
                             "l"(v + off + 16 * i), "r"(vsz));
            asm volatile("cp.async.commit_group;\n" ::);
        }
        uint4 kraw[kSmemK ? 1 : kHeadDim / 16];  // the key's whole K row (without kSmemK)
        if constexpr (!kSmemK) {
            const uint4* src = reinterpret_cast<const uint4*>(k + row * kKvWidth + kv_off);
#pragma unroll
            for (int i = 0; i < kHeadDim / 16; ++i) kraw[i] = valid ? src[i] : make_uint4(0, 0, 0, 0);
        }
        if (kb + kBlockKeys < key_end) {  // the next block's slot
            const int64_t next = kb + kBlockKeys + lk;
            row = next < key_end ? (slots ? slots[next] : next) : 0;
        }
        if constexpr (kSmemK) asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();  // K (kSmemK), qs, row_m / row_l (first block)

        float s[kRT];
        {
            float dot[kRT];
#pragma unroll
            for (int m = 0; m < kRT; ++m) dot[m] = 0.0f;
            const uint4* krow = reinterpret_cast<const uint4*>(ks + lk * kKRowBytes);
#pragma unroll
            for (int c = 0; c < kHeadDim / 16; ++c) {
                float kf[16];
                if constexpr (kSmemK) {
                    fp8x16_to_float(krow[c], kf);
                } else {
                    fp8x16_to_float(kraw[c], kf);
                }
#pragma unroll
                for (int m = 0; m < kRT; ++m) {
                    const float* qr = qs + (half * kRT + m) * kHeadDim + c * 16;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float4 qv = *reinterpret_cast<const float4*>(qr + 4 * i);
                        dot[m] = fmaf(qv.x, kf[4 * i], dot[m]);
                        dot[m] = fmaf(qv.y, kf[4 * i + 1], dot[m]);
                        dot[m] = fmaf(qv.z, kf[4 * i + 2], dot[m]);
                        dot[m] = fmaf(qv.w, kf[4 * i + 3], dot[m]);
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < kRT; ++m) s[m] = valid ? __fmul_rn(dot[m], score_scale) : -INFINITY;
        }
        const int tile = lk / kKeyTile;
        const int lj = lk & (kKeyTile - 1);
#pragma unroll
        for (int m = 0; m < kRT; ++m) {
            float mx = fmaxf(-INFINITY, s[m]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            if (lj == 0) tile_max[(half * kRT + m) * kTilesPerBlock + tile] = mx;
        }
        __syncthreads();

#pragma unroll
        for (int m = 0; m < kRT; ++m) {
            const int r = half * kRT + m;
            float m_new = row_m[r];
            for (int i = 0; i <= tile; ++i) m_new = fmaxf(m_new, tile_max[r * kTilesPerBlock + i]);
            const bool empty = m_new == -INFINITY;
            const float p = empty ? 0.0f : exp2f(__fsub_rn(s[m], m_new));
            ps[r * kBlockKeys + lk] = p;
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < kKeyTile; ++j) sum += __shfl_sync(0xffffffffu, p, (tid & 16) + j);
            if (lj == 0) {
                tile_m[r * kTilesPerBlock + tile] = m_new;
                tile_sum[r * kTilesPerBlock + tile] = sum;
            }
        }
        __syncthreads();

        if (tid < kR) {  // the alpha / l recurrence of row tid over the tiles in order
            float m_old = row_m[tid];
            float l = row_l[tid];
            for (int i = 0; i < tiles; ++i) {
                const float m_new = tile_m[tid * kTilesPerBlock + i];
                const bool empty = m_new == -INFINITY;
                const float alpha = empty ? 1.0f : exp2f(m_old - m_new);
                l = fmaf(l, alpha, empty ? 0.0f : tile_sum[tid * kTilesPerBlock + i]);
                tile_alpha[tid * kTilesPerBlock + i] = alpha;
                m_old = m_new;
            }
            row_m[tid] = m_old;
            row_l[tid] = l;
        }
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        // acc = acc * alpha_t + P . V for dims 2 lk, 2 lk + 1 of rows half * kRT + m.
        const uint8_t* vcol = vs + 2 * lk;
        for (int i = 0; i < tiles; ++i) {
#pragma unroll
            for (int m = 0; m < kRT; ++m) {
                const float a = tile_alpha[(half * kRT + m) * kTilesPerBlock + i];
                acc[m][0] *= a;
                acc[m][1] *= a;
            }
#pragma unroll
            for (int j4 = 0; j4 < kKeyTile; j4 += 4) {
                float4 p4[kRT];
#pragma unroll
                for (int m = 0; m < kRT; ++m)
                    p4[m] = *reinterpret_cast<const float4*>(ps + (half * kRT + m) * kBlockKeys + i * kKeyTile + j4);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const int j = i * kKeyTile + j4 + jj;
                    const float2 vv = fp8x2_to_float2(*reinterpret_cast<const unsigned short*>(vcol + j * kHeadDim));
#pragma unroll
                    for (int m = 0; m < kRT; ++m) {
                        const float p = jj == 0 ? p4[m].x : jj == 1 ? p4[m].y : jj == 2 ? p4[m].z : p4[m].w;
                        acc[m][0] = fmaf(p, vv.x, acc[m][0]);
                        acc[m][1] = fmaf(p, vv.y, acc[m][1]);
                    }
                }
            }
        }
        __syncthreads();  // this block's P.V is done with vs / ps
    }
    __syncthreads();  // row_m / row_l of a call with no keys

#pragma unroll
    for (int m = 0; m < kRT; ++m) {
        const int r = half * kRT + m;
        const int h = h0 + r;
        const int d = 2 * lk;
        if (part_o == nullptr) {
            const float l = row_l[r];
            *reinterpret_cast<__nv_bfloat162*>(out + t * kQWidth + static_cast<int64_t>(h) * kHeadDim + d) =
                __halves2bfloat162(__float2bfloat16_rn(acc[m][0] / l), __float2bfloat16_rn(acc[m][1] / l));
        } else {
            const int64_t prow = (static_cast<int64_t>(split) * tokens + t) * kQHeads + h;
            *reinterpret_cast<float2*>(part_o + prow * kHeadDim + d) = make_float2(acc[m][0], acc[m][1]);
            if (lk == 0) {
                part_m[prow] = row_m[r];
                part_l[prow] = row_l[r];
            }
        }
    }
}

// Combine the key splits of each (token, head): out = sum_s 2^(m_s - M) o_s / sum_s 2^(m_s - M) l_s. kStep: splits
// from *step_pos as the kernels' step (one split: the kernel wrote `out`, nothing to combine). kChain: token t's
// splits are its one-token call's at pos0 + t (pos0 = *step_pos when step_pos is non-null), as split_rows_kernel's
// chain (one split: nothing to combine); kRowPos: at step_pos[t], as split_rows_kernel's kRowKv.
template <bool kStep = false, bool kChain = false, bool kRowPos = false>
__global__ void combine_kernel(int splits, int64_t tokens, const float* __restrict__ part_o,
                               const float* __restrict__ part_m, const float* __restrict__ part_l, bf16* __restrict__ out,
                               const int64_t* __restrict__ step_pos, int64_t pos0) {
    static_assert(!kRowPos || kChain, "row positions are the multi-sequence chain's");
    if constexpr (kStep) {
        splits = split_count(tokens, *step_pos + tokens);
        if (splits == 1) return;
    }
    if constexpr (kChain) {
        if constexpr (kRowPos) {
            pos0 = step_pos[blockIdx.x] - static_cast<int64_t>(blockIdx.x);
        } else if (step_pos != nullptr) {
            pos0 = *step_pos;
        }
        splits = split_count(1, pos0 + static_cast<int64_t>(blockIdx.x) + 1);
        if (splits == 1) return;
    }
    const int64_t t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    float m_max = -INFINITY;
    for (int s = 0; s < splits; ++s) m_max = fmaxf(m_max, part_m[(s * tokens + t) * kQHeads + h]);
    float l = 0.0f;
    float o = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const int64_t row = (s * tokens + t) * kQHeads + h;
        const float m = part_m[row];
        if (m == -INFINITY) continue;
        const float w = exp2f(m - m_max);
        l = fmaf(part_l[row], w, l);
        o = fmaf(part_o[row * kHeadDim + d], w, o);
    }
    out[t * kQWidth + static_cast<int64_t>(h) * kHeadDim + d] = __float2bfloat16_rn(o / l);
}

template <int kR, bool kSmemK, bool kStep, bool kChain, bool kRowKv = false>
cudaError_t launch_rows(const bf16* q, int64_t tokens, int64_t pos0, const uint8_t* k, const uint8_t* v,
                        const int64_t* slots, float scale, int splits, int64_t keys_per_split, bf16* out,
                        float* part_o, float* part_m, float* part_l, const int64_t* step_pos, cudaStream_t stream,
                        const RowKvArg<kRowKv>& rkv = {}) {
    constexpr size_t bytes = RowsSmem<kR, kSmemK>::kBytes;
    auto kernel = split_rows_kernel<kR, kSmemK, kStep, kChain, kRowKv>;
    cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if (err != cudaSuccess) return err;
    const dim3 grid(static_cast<unsigned>(tokens), kKvHeads * (kGroup / kR), static_cast<unsigned>(splits));
    kernel<<<grid, kThreads, bytes, stream>>>(q, tokens, pos0, k, v, slots, scale, keys_per_split, out, part_o, part_m,
                                              part_l, step_pos, rkv);
    return cudaGetLastError();
}

template <int kTokens, typename KvT, bool kStep = false, bool kChain = false>
cudaError_t launch(const bf16* q, int64_t tokens, int64_t pos0, const KvT* k, int64_t k_stride, const KvT* v,
                   int64_t v_stride, const int64_t* slots, float scale, int splits, int64_t keys_per_split, bf16* out,
                   float* part_o, float* part_m, float* part_l, const int64_t* step_pos, cudaStream_t stream) {
    if constexpr (kTokens == 1 && std::is_same_v<KvT, uint8_t>) {
        // FP8 cache rows are kKvWidth bytes (every caller); the split blocks index them so.
        if (k_stride != kKvWidth || v_stride != kKvWidth) return cudaErrorInvalidValue;
        if constexpr (kChain) {
            if (tokens <= kChainFewMax)
                return launch_rows<kChainRowsFew, false, kStep, kChain>(q, tokens, pos0, k, v, slots, scale, splits,
                                                                        keys_per_split, out, part_o, part_m, part_l,
                                                                        step_pos, stream);
            return launch_rows<kChainRowsMany, false, kStep, kChain>(q, tokens, pos0, k, v, slots, scale, splits,
                                                                     keys_per_split, out, part_o, part_m, part_l,
                                                                     step_pos, stream);
        } else {
            return launch_rows<kStepRows, true, kStep, kChain>(q, tokens, pos0, k, v, slots, scale, splits,
                                                               keys_per_split, out, part_o, part_m, part_l, step_pos,
                                                               stream);
        }
    } else {
        static_assert(!kChain, "a verify chain runs one-token FP8 blocks");
        constexpr size_t bytes = Smem<kTokens>::kBytes;
        auto kernel = attention_kernel<kTokens, KvT, kStep>;
        cudaError_t err =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
        if (err != cudaSuccess) return err;
        const dim3 grid(static_cast<unsigned>((tokens + kTokens - 1) / kTokens), kKvHeads,
                        static_cast<unsigned>(splits));
        kernel<<<grid, kThreads, bytes, stream>>>(q, tokens, pos0, k, k_stride, v, v_stride, slots, scale,
                                                  keys_per_split, out, part_o, part_m, part_l, step_pos);
        return cudaGetLastError();
    }
}

// The partials of `tokens` rows at the start of `workspace` (partial_bytes(tokens)).
void carve_partials(void* workspace, int64_t tokens, float** part_o, float** part_m, float** part_l) {
    const size_t rows = static_cast<size_t>(kMaxSplits) * static_cast<size_t>(tokens) * kQHeads;
    *part_o = static_cast<float*>(workspace);
    *part_m = reinterpret_cast<float*>(static_cast<unsigned char*>(workspace) + align256(rows * kHeadDim * sizeof(float)));
    *part_l = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(*part_m) + align256(rows * sizeof(float)));
}

template <typename KvT>
cudaError_t launch_prefill(const bf16* q, int64_t tokens, int64_t pos0, const KvT* k, int64_t k_stride, const KvT* v,
                           int64_t v_stride, const int64_t* slots, float scale, bf16* out, cudaStream_t stream) {
    constexpr size_t bytes = PrefillSmem::kBytes;
    auto kernel = prefill_kernel<KvT>;
    cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if (err != cudaSuccess) return err;
    const dim3 grid(kKvHeads, static_cast<unsigned>((tokens + kPrefillTokens - 1) / kPrefillTokens));
    kernel<<<grid, kThreads, bytes, stream>>>(q, tokens, pos0, k, k_stride, v, v_stride, slots, scale, out);
    return cudaGetLastError();
}

// `sel`: the row count every choice is made for (attention's cls pin, else tokens).
template <typename KvT>
cudaError_t attention_impl(const bf16* q, int64_t tokens, int64_t pos0, const KvT* k, int64_t k_stride, const KvT* v,
                           int64_t v_stride, const int64_t* slots, float scale, void* workspace,
                           size_t workspace_bytes, bf16* out, cudaStream_t stream, int64_t sel) {
    // Prefill shapes (one split, 4-token tiles): prefill_kernel, attention_kernel<4>'s arithmetic row for row.
    if (sel > kSplitMaxTokens)
        return launch_prefill(q, tokens, pos0, k, k_stride, v, v_stride, slots, scale, out, stream);
    const int64_t kv_len = pos0 + tokens;
    const int splits = split_count(sel, kv_len);
    float* part_o = nullptr;
    float* part_m = nullptr;
    float* part_l = nullptr;
    int64_t keys_per_split = kv_len;
    if (splits > 1) {
        if (!workspace || workspace_bytes < partial_bytes(tokens)) return cudaErrorInvalidValue;
        carve_partials(workspace, tokens, &part_o, &part_m, &part_l);
        keys_per_split = split_keys(kv_len, splits);
    }
    cudaError_t err = sel >= 4 ? launch<4>(q, tokens, pos0, k, k_stride, v, v_stride, slots, scale, splits,
                                           keys_per_split, out, part_o, part_m, part_l, nullptr, stream)
                               : launch<1>(q, tokens, pos0, k, k_stride, v, v_stride, slots, scale, splits,
                                           keys_per_split, out, part_o, part_m, part_l, nullptr, stream);
    if (err != cudaSuccess || splits == 1) return err;
    combine_kernel<false><<<dim3(static_cast<unsigned>(tokens), kQHeads), kHeadDim, 0, stream>>>(
        splits, tokens, part_o, part_m, part_l, out, nullptr, 0);
    return cudaGetLastError();
}

// `tokens` (1 .. kSplitMaxTokens) tokens from the FP8 cache at the device positions *pos .. *pos + tokens - 1 (the
// decode graph's step at tokens 1; the spec_graph draft graph's MTP extend): the launches of attention_impl, pos0 =
// *pos, with the split grid fixed at kStepSplits (the extra splits exit) and the combine always launched (a call of
// one split returns from it).
cudaError_t attention_step_impl(const bf16* q, int64_t tokens, const uint8_t* k_cache, const uint8_t* v_cache,
                                const int64_t* slots, const int64_t* pos, float scale, void* workspace,
                                size_t workspace_bytes, bf16* out, cudaStream_t stream) {
    if (!workspace || workspace_bytes < partial_bytes(tokens)) return cudaErrorInvalidValue;
    float* part_o = nullptr;
    float* part_m = nullptr;
    float* part_l = nullptr;
    carve_partials(workspace, tokens, &part_o, &part_m, &part_l);
    const cudaError_t err =
        tokens >= 4 ? launch<4, uint8_t, true>(q, tokens, 0, k_cache, int64_t{kKvWidth}, v_cache, int64_t{kKvWidth},
                                               slots, scale, kStepSplits, 0, out, part_o, part_m, part_l, pos, stream)
                    : launch<1, uint8_t, true>(q, tokens, 0, k_cache, int64_t{kKvWidth}, v_cache, int64_t{kKvWidth},
                                               slots, scale, kStepSplits, 0, out, part_o, part_m, part_l, pos, stream);
    if (err != cudaSuccess) return err;
    combine_kernel<true><<<dim3(static_cast<unsigned>(tokens), kQHeads), kHeadDim, 0, stream>>>(
        kStepSplits, tokens, part_o, part_m, part_l, out, pos, 0);
    return cudaGetLastError();
}

// `tokens` one-token calls from the FP8 cache at pos0 .. pos0 + tokens - 1 (verify_attention = batched): the launches
// of attention_impl for tokens 1 at each position, as one chain launch (split_rows_kernel's kChain) and one chain
// combine. The grid's splits are the last row's, the most of any row (split_count grows with kv_len). With `pos`
// (device; the verify graph's chain) pos0 = *pos on the device and the grid has kStepSplits splits, the combine always
// launched (its rows of one split exit).
cudaError_t attention_chain_impl(const bf16* q, int64_t tokens, int64_t pos0, const int64_t* pos,
                                 const uint8_t* k_cache, const uint8_t* v_cache, const int64_t* slots, float scale,
                                 void* workspace, size_t workspace_bytes, bf16* out, cudaStream_t stream) {
    const int splits = pos != nullptr ? kStepSplits : split_count(1, pos0 + tokens);
    float* part_o = nullptr;
    float* part_m = nullptr;
    float* part_l = nullptr;
    if (splits > 1) {
        if (!workspace || workspace_bytes < partial_bytes(tokens)) return cudaErrorInvalidValue;
        carve_partials(workspace, tokens, &part_o, &part_m, &part_l);
    }
    const cudaError_t err = launch<1, uint8_t, false, true>(q, tokens, pos0, k_cache, int64_t{kKvWidth}, v_cache,
                                                            int64_t{kKvWidth}, slots, scale, splits, 0, out, part_o,
                                                            part_m, part_l, pos, stream);
    if (err != cudaSuccess || splits == 1) return err;
    combine_kernel<false, true><<<dim3(static_cast<unsigned>(tokens), kQHeads), kHeadDim, 0, stream>>>(
        splits, tokens, part_o, part_m, part_l, out, pos, pos0);
    return cudaGetLastError();
}

// attention_chain_impl with pos over several sequences' chains at once (dense_attention_chains_at, R16-C): one kRowKv
// chain launch (block row t on kv.*[t] at positions[t]) and one kRowPos chain combine, the grid at kStepSplits as the
// device-position chain's (a row's extra splits exit; a row of one split writes `out` and its combine exits).
cudaError_t attention_chains_impl(const bf16* q, int64_t tokens, const int64_t* positions, const k1::KvRows& kv,
                                  float scale, void* workspace, size_t workspace_bytes, bf16* out,
                                  cudaStream_t stream) {
    if (!workspace || workspace_bytes < partial_bytes(tokens)) return cudaErrorInvalidValue;
    float* part_o = nullptr;
    float* part_m = nullptr;
    float* part_l = nullptr;
    carve_partials(workspace, tokens, &part_o, &part_m, &part_l);
    const cudaError_t err =
        tokens <= kChainFewMax
            ? launch_rows<kChainRowsFew, false, false, true, true>(q, tokens, 0, nullptr, nullptr, nullptr, scale,
                                                                   kStepSplits, 0, out, part_o, part_m, part_l,
                                                                   positions, stream, kv)
            : launch_rows<kChainRowsMany, false, false, true, true>(q, tokens, 0, nullptr, nullptr, nullptr, scale,
                                                                    kStepSplits, 0, out, part_o, part_m, part_l,
                                                                    positions, stream, kv);
    if (err != cudaSuccess) return err;
    combine_kernel<false, true, true><<<dim3(static_cast<unsigned>(tokens), kQHeads), kHeadDim, 0, stream>>>(
        kStepSplits, tokens, part_o, part_m, part_l, out, positions, 0);
    return cudaGetLastError();
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace

size_t prologue_bytes(int64_t max_tokens) {
    if (max_tokens <= 0) return 0;
    const size_t t = static_cast<size_t>(max_tokens);
    return align256(t * kQWidth * sizeof(bf16))       // q (normed, rotated)
           + align256(t * kKvWidth * sizeof(bf16))    // k (normed, rotated)
           + align256(t * kQWidth * sizeof(bf16));    // gate
}

size_t dense_attention_workspace_bytes(int64_t max_tokens) {
    if (max_tokens <= 0) return 0;
    const size_t t = static_cast<size_t>(max_tokens);
    return align256(t * kQWidth * sizeof(bf16))     // attention output before the gate
           + partial_bytes(max_tokens < kSplitMaxTokens ? max_tokens : kSplitMaxTokens);
}

size_t full_attention_workspace_bytes(int64_t max_tokens) {
    return prologue_bytes(max_tokens) + dense_attention_workspace_bytes(max_tokens);
}

int split_blocks(int64_t rows, bool chain) {
    const int r = !chain ? kStepRows : rows <= kChainFewMax ? kChainRowsFew : kChainRowsMany;
    return kKvHeads * (kGroup / r);
}

cudaError_t attention(const bf16* q, int64_t tokens, int64_t pos0, const bf16* k, int64_t k_stride, const bf16* v,
                      int64_t v_stride, const uint8_t* k_cache, const uint8_t* v_cache, const int64_t* slots,
                      float scale, void* workspace, size_t workspace_bytes, bf16* out, cudaStream_t stream,
                      bool use_p4, int64_t cls) {
    const bool bf16_kv = k && v && !k_cache && !v_cache;
    const bool fp8_kv = !k && !v && k_cache && v_cache;
    if (!q || !out || tokens < 0 || pos0 < 0 || pos0 + tokens > kMaxContext || !(bf16_kv || fp8_kv) ||
        !aligned16(q) || cls < 0 || (cls != 0 && cls < tokens) ||
        (bf16_kv && (!aligned16(k) || !aligned16(v) || k_stride % 8 || v_stride % 8 || k_stride < kKvWidth ||
                     v_stride < kKvWidth)) ||
        (fp8_kv && (!aligned16(k_cache) || !aligned16(v_cache))))
        return cudaErrorInvalidValue;
    if (tokens == 0) return cudaSuccess;
    const int64_t sel = cls != 0 ? cls : tokens;
    // P4: calls of more than 16 tokens run the tensor-core dense GQA, and so does a pinned call of fewer whose class
    // chose it (P4 is row-partition invariant, p4_attention.h).
    if (use_p4 && sel >= p4::kMinRows && aligned16(out))
        return p4::dense_attention(q, tokens, pos0, k, k_stride, v, v_stride, k_cache, v_cache, slots, scale, out,
                                   stream);
    if (bf16_kv)
        return attention_impl(q, tokens, pos0, k, k_stride, v, v_stride, nullptr, scale, workspace, workspace_bytes,
                              out, stream, sel);
    return attention_impl(q, tokens, pos0, k_cache, int64_t{kKvWidth}, v_cache, int64_t{kKvWidth}, slots, scale,
                          workspace, workspace_bytes, out, stream, sel);
}

cudaError_t prologue(const bf16* q_gate, int64_t q_gate_stride, const bf16* k, int64_t k_stride, const bf16* v,
                     int64_t v_stride, const bf16* q_norm_weight, const bf16* k_norm_weight, float eps,
                     const float* cos_sin, int64_t context, const int64_t* positions, const int64_t* positions3,
                     const int64_t* slots, int64_t tokens, int64_t prefix_len, uint8_t* k_cache, uint8_t* v_cache,
                     bf16* q_out, bf16* k_out, bf16* gate_out, cudaStream_t stream) {
    if (!q_gate || !k || !v || !q_norm_weight || !k_norm_weight || !cos_sin || !k_cache || !v_cache || !q_out ||
        !k_out || !gate_out || tokens < 0 || prefix_len < 0 || context <= 0 || prefix_len + tokens > context)
        return cudaErrorInvalidValue;
    if (tokens == 0) return cudaSuccess;
    cudaError_t err = k1::qk_norm_rope_gate(q_gate, q_gate_stride, k, k_stride, q_norm_weight, k_norm_weight, cos_sin,
                                            positions, positions3, prefix_len, tokens, kQHeads, kKvHeads, kHeadDim,
                                            kRotaryDim, eps, q_out, k_out, gate_out, stream);
    if (err != cudaSuccess) return err;
    return k1::kv_store_fp8(k_out, kKvWidth, v, v_stride, tokens, kKvWidth, k_cache, v_cache,
                            slots ? slots + prefix_len : nullptr, prefix_len, stream);
}

cudaError_t dense_attention(const bf16* q, const bf16* k, const bf16* v, int64_t v_stride, const bf16* gate,
                            int64_t tokens, int64_t prefix_len, bool kv_from_cache, const uint8_t* k_cache,
                            const uint8_t* v_cache, const int64_t* slots, void* workspace, size_t workspace_bytes,
                            bf16* out, int64_t out_stride, cudaStream_t stream, bool use_p4, int64_t cls) {
    if (!q || !k || !v || !gate || !k_cache || !v_cache || !workspace || !out || tokens < 0 || prefix_len < 0 ||
        prefix_len + tokens > kMaxContext || workspace_bytes < dense_attention_workspace_bytes(tokens) ||
        !aligned16(workspace))
        return cudaErrorInvalidValue;
    if (tokens == 0) return cudaSuccess;
    unsigned char* ws = static_cast<unsigned char*>(workspace);
    bf16* attn = reinterpret_cast<bf16*>(ws);
    ws += align256(static_cast<size_t>(tokens) * kQWidth * sizeof(bf16));
    const size_t rest = workspace_bytes - static_cast<size_t>(ws - static_cast<unsigned char*>(workspace));
    cudaError_t err;
    if (kv_from_cache) {
        err = attention(q, tokens, prefix_len, nullptr, 0, nullptr, 0, k_cache, v_cache, slots, kSoftmaxScale, ws,
                        rest, attn, stream, use_p4, cls);
    } else {
        err = attention(q, tokens, prefix_len, k, kKvWidth, v, v_stride, nullptr, nullptr, nullptr, kSoftmaxScale,
                        ws, rest, attn, stream, use_p4, cls);
    }
    if (err != cudaSuccess) return err;
    return k1::sigmoid_mul(attn, kQWidth, gate, kQWidth, tokens, kQWidth, out, out_stride, stream);
}

cudaError_t full_attention(const bf16* q_gate, int64_t q_gate_stride, const bf16* k, int64_t k_stride, const bf16* v,
                           int64_t v_stride, const bf16* q_norm_weight, const bf16* k_norm_weight, float eps,
                           const float* cos_sin, const int64_t* positions, const int64_t* positions3,
                           const int64_t* slots, int64_t tokens, int64_t prefix_len, bool kv_from_cache,
                           uint8_t* k_cache, uint8_t* v_cache, void* workspace, size_t workspace_bytes, bf16* out,
                           int64_t out_stride, cudaStream_t stream, bool use_p4) {
    if (!q_gate || !k || !v || !q_norm_weight || !k_norm_weight || !cos_sin || !k_cache || !v_cache || !workspace ||
        !out || tokens < 0 || prefix_len < 0 || prefix_len + tokens > kMaxContext || (!kv_from_cache && prefix_len) ||
        workspace_bytes < full_attention_workspace_bytes(tokens) || !aligned16(workspace))
        return cudaErrorInvalidValue;
    if (tokens == 0) return cudaSuccess;
    const size_t t = static_cast<size_t>(tokens);
    unsigned char* ws = static_cast<unsigned char*>(workspace);
    bf16* q = reinterpret_cast<bf16*>(ws);
    ws += align256(t * kQWidth * sizeof(bf16));
    bf16* k_rot = reinterpret_cast<bf16*>(ws);
    ws += align256(t * kKvWidth * sizeof(bf16));
    bf16* gate = reinterpret_cast<bf16*>(ws);
    ws += align256(t * kQWidth * sizeof(bf16));
    const size_t rest = workspace_bytes - static_cast<size_t>(ws - static_cast<unsigned char*>(workspace));
    cudaError_t err = prologue(q_gate, q_gate_stride, k, k_stride, v, v_stride, q_norm_weight, k_norm_weight, eps,
                               cos_sin, kMaxContext, positions, positions3, slots, tokens, prefix_len, k_cache,
                               v_cache, q, k_rot, gate, stream);
    if (err != cudaSuccess) return err;
    return dense_attention(q, k_rot, v, v_stride, gate, tokens, prefix_len, kv_from_cache, k_cache, v_cache, slots,
                           ws, rest, out, out_stride, stream, use_p4);
}

namespace {

// dense_attention_rows (pos == nullptr) and dense_attention_rows_at (pos0 = *pos on the device).
cudaError_t chain_rows(const bf16* q, const bf16* gate, int64_t rows, int64_t pos0, const int64_t* pos,
                       const uint8_t* k_cache, const uint8_t* v_cache, const int64_t* slots, void* workspace,
                       size_t workspace_bytes, bf16* out, int64_t out_stride, cudaStream_t stream) {
    if (!q || !gate || !k_cache || !v_cache || !workspace || !out || rows < 1 || rows > kSplitMaxTokens || pos0 < 0 ||
        pos0 + rows > kMaxContext || workspace_bytes < dense_attention_workspace_bytes(rows) ||
        !aligned16(workspace) || !aligned16(q) || !aligned16(k_cache) || !aligned16(v_cache))
        return cudaErrorInvalidValue;
    unsigned char* ws = static_cast<unsigned char*>(workspace);
    bf16* attn = reinterpret_cast<bf16*>(ws);
    ws += align256(static_cast<size_t>(rows) * kQWidth * sizeof(bf16));
    const size_t rest = workspace_bytes - static_cast<size_t>(ws - static_cast<unsigned char*>(workspace));
    const cudaError_t err = attention_chain_impl(q, rows, pos0, pos, k_cache, v_cache, slots, kSoftmaxScale, ws, rest,
                                                 attn, stream);
    if (err != cudaSuccess) return err;
    return k1::sigmoid_mul(attn, kQWidth, gate, kQWidth, rows, kQWidth, out, out_stride, stream);
}

}  // namespace

cudaError_t dense_attention_rows(const bf16* q, const bf16* gate, int64_t rows, int64_t pos0, const uint8_t* k_cache,
                                 const uint8_t* v_cache, const int64_t* slots, void* workspace, size_t workspace_bytes,
                                 bf16* out, int64_t out_stride, cudaStream_t stream) {
    return chain_rows(q, gate, rows, pos0, nullptr, k_cache, v_cache, slots, workspace, workspace_bytes, out,
                      out_stride, stream);
}

cudaError_t dense_attention_rows_at(const bf16* q, const bf16* gate, int64_t rows, const int64_t* positions,
                                    const uint8_t* k_cache, const uint8_t* v_cache, const int64_t* slots,
                                    void* workspace, size_t workspace_bytes, bf16* out, int64_t out_stride,
                                    cudaStream_t stream) {
    if (!positions || !slots) return cudaErrorInvalidValue;
    return chain_rows(q, gate, rows, 0, positions, k_cache, v_cache, slots, workspace, workspace_bytes, out,
                      out_stride, stream);
}

cudaError_t prologue_rows_at(const bf16* q_gate, int64_t q_gate_stride, const bf16* k, int64_t k_stride, const bf16* v,
                             int64_t v_stride, const bf16* q_norm_weight, const bf16* k_norm_weight, float eps,
                             const float* cos_sin, const int64_t* slots, const int64_t* positions, int64_t rows,
                             uint8_t* k_cache, uint8_t* v_cache, bf16* q_out, bf16* k_out, bf16* gate_out,
                             cudaStream_t stream) {
    if (!q_gate || !k || !v || !q_norm_weight || !k_norm_weight || !cos_sin || !slots || !positions || !k_cache ||
        !v_cache || !q_out || !k_out || !gate_out || rows < 1 || rows > kSplitMaxTokens)
        return cudaErrorInvalidValue;
    cudaError_t err = k1::qk_norm_rope_gate(q_gate, q_gate_stride, k, k_stride, q_norm_weight, k_norm_weight, cos_sin,
                                            positions, nullptr, 0, rows, kQHeads, kKvHeads, kHeadDim, kRotaryDim, eps,
                                            q_out, k_out, gate_out, stream);
    if (err != cudaSuccess) return err;
    return k1::kv_store_fp8_at(k_out, kKvWidth, v, v_stride, rows, kKvWidth, k_cache, v_cache, slots, positions,
                               stream);
}

static_assert(kSplitMaxTokens == k1::kKvRowsMax, "a multi-sequence chain call's rows are a split chain's");

namespace {
bool kv_rows_ok(const k1::KvRows& kv, int64_t rows) {
    for (int64_t t = 0; t < rows; ++t) {
        if (!kv.k_cache[t] || !kv.v_cache[t] || !kv.slots[t] || !aligned16(kv.k_cache[t]) || !aligned16(kv.v_cache[t]))
            return false;
    }
    return true;
}
}  // namespace

cudaError_t prologue_chains_at(const bf16* q_gate, int64_t q_gate_stride, const bf16* k, int64_t k_stride,
                               const bf16* v, int64_t v_stride, const bf16* q_norm_weight, const bf16* k_norm_weight,
                               float eps, const float* cos_sin, const k1::KvRows& kv, const int64_t* positions,
                               int64_t rows, bf16* q_out, bf16* k_out, bf16* gate_out, cudaStream_t stream) {
    if (!q_gate || !k || !v || !q_norm_weight || !k_norm_weight || !cos_sin || !positions || !q_out || !k_out ||
        !gate_out || rows < 1 || rows > kSplitMaxTokens || !kv_rows_ok(kv, rows))
        return cudaErrorInvalidValue;
    cudaError_t err = k1::qk_norm_rope_gate(q_gate, q_gate_stride, k, k_stride, q_norm_weight, k_norm_weight, cos_sin,
                                            positions, nullptr, 0, rows, kQHeads, kKvHeads, kHeadDim, kRotaryDim, eps,
                                            q_out, k_out, gate_out, stream);
    if (err != cudaSuccess) return err;
    return k1::kv_store_fp8_rows(k_out, kKvWidth, v, v_stride, rows, kKvWidth, kv, positions, stream);
}

cudaError_t dense_attention_chains_at(const bf16* q, const bf16* gate, int64_t rows, const int64_t* positions,
                                      const k1::KvRows& kv, void* workspace, size_t workspace_bytes, bf16* out,
                                      int64_t out_stride, cudaStream_t stream) {
    if (!q || !gate || !positions || !workspace || !out || rows < 1 || rows > kSplitMaxTokens ||
        workspace_bytes < dense_attention_workspace_bytes(rows) || !aligned16(workspace) || !aligned16(q) ||
        !kv_rows_ok(kv, rows))
        return cudaErrorInvalidValue;
    unsigned char* ws = static_cast<unsigned char*>(workspace);
    bf16* attn = reinterpret_cast<bf16*>(ws);
    ws += align256(static_cast<size_t>(rows) * kQWidth * sizeof(bf16));
    const size_t rest = workspace_bytes - static_cast<size_t>(ws - static_cast<unsigned char*>(workspace));
    const cudaError_t err =
        attention_chains_impl(q, rows, positions, kv, kSoftmaxScale, ws, rest, attn, stream);
    if (err != cudaSuccess) return err;
    return k1::sigmoid_mul(attn, kQWidth, gate, kQWidth, rows, kQWidth, out, out_stride, stream);
}

cudaError_t prologue_step(const bf16* q_gate, int64_t q_gate_stride, const bf16* k, int64_t k_stride, const bf16* v,
                          int64_t v_stride, const bf16* q_norm_weight, const bf16* k_norm_weight, float eps,
                          const float* cos_sin, const int64_t* slots, const int64_t* position, uint8_t* k_cache,
                          uint8_t* v_cache, bf16* q_out, bf16* k_out, bf16* gate_out, cudaStream_t stream) {
    return prologue_rows_at(q_gate, q_gate_stride, k, k_stride, v, v_stride, q_norm_weight, k_norm_weight, eps,
                            cos_sin, slots, position, 1, k_cache, v_cache, q_out, k_out, gate_out, stream);
}

cudaError_t dense_attention_at(const bf16* q, const bf16* gate, int64_t tokens, const int64_t* positions,
                               const uint8_t* k_cache, const uint8_t* v_cache, const int64_t* slots, void* workspace,
                               size_t workspace_bytes, bf16* out, int64_t out_stride, cudaStream_t stream) {
    if (!q || !gate || !k_cache || !v_cache || !slots || !positions || !workspace || !out || tokens < 1 ||
        tokens > kSplitMaxTokens || workspace_bytes < dense_attention_workspace_bytes(tokens) ||
        !aligned16(workspace) || !aligned16(q) || !aligned16(k_cache) || !aligned16(v_cache))
        return cudaErrorInvalidValue;
    unsigned char* ws = static_cast<unsigned char*>(workspace);
    bf16* attn = reinterpret_cast<bf16*>(ws);
    ws += align256(static_cast<size_t>(tokens) * kQWidth * sizeof(bf16));
    const size_t rest = workspace_bytes - static_cast<size_t>(ws - static_cast<unsigned char*>(workspace));
    const cudaError_t err = attention_step_impl(q, tokens, k_cache, v_cache, slots, positions, kSoftmaxScale, ws, rest,
                                                attn, stream);
    if (err != cudaSuccess) return err;
    return k1::sigmoid_mul(attn, kQWidth, gate, kQWidth, tokens, kQWidth, out, out_stride, stream);
}

cudaError_t dense_attention_step(const bf16* q, const bf16* gate, const uint8_t* k_cache, const uint8_t* v_cache,
                                 const int64_t* slots, const int64_t* position, void* workspace,
                                 size_t workspace_bytes, bf16* out, int64_t out_stride, cudaStream_t stream) {
    return dense_attention_at(q, gate, 1, position, k_cache, v_cache, slots, workspace, workspace_bytes, out,
                              out_stride, stream);
}

cudaError_t full_attention_step(const bf16* q_gate, int64_t q_gate_stride, const bf16* k, int64_t k_stride,
                                const bf16* v, int64_t v_stride, const bf16* q_norm_weight, const bf16* k_norm_weight,
                                float eps, const float* cos_sin, const int64_t* slots, const int64_t* position,
                                uint8_t* k_cache, uint8_t* v_cache, void* workspace, size_t workspace_bytes, bf16* out,
                                int64_t out_stride, cudaStream_t stream) {
    if (!workspace || workspace_bytes < full_attention_workspace_bytes(1) || !aligned16(workspace))
        return cudaErrorInvalidValue;
    unsigned char* ws = static_cast<unsigned char*>(workspace);
    bf16* q = reinterpret_cast<bf16*>(ws);
    ws += align256(kQWidth * sizeof(bf16));
    bf16* k_rot = reinterpret_cast<bf16*>(ws);
    ws += align256(kKvWidth * sizeof(bf16));
    bf16* gate = reinterpret_cast<bf16*>(ws);
    ws += align256(kQWidth * sizeof(bf16));
    const size_t rest = workspace_bytes - static_cast<size_t>(ws - static_cast<unsigned char*>(workspace));
    cudaError_t err = prologue_step(q_gate, q_gate_stride, k, k_stride, v, v_stride, q_norm_weight, k_norm_weight, eps,
                                    cos_sin, slots, position, k_cache, v_cache, q, k_rot, gate, stream);
    if (err != cudaSuccess) return err;
    return dense_attention_step(q, gate, k_cache, v_cache, slots, position, ws, rest, out, out_stride, stream);
}

}  // namespace cere::k6
