// K1 primitives (m22 design note K1); contracts in k1_primitives.h.
// Math follows SGLang's GemmaRMSNorm, fused_qk_gemma_rmsnorm_rope_gate (Q/K norm rounded to BF16 before the FP32
// RoPE), MRotaryEmbedding's cos/sin cache, fused_sigmoid_mul, torch.argmax for greedy sampling and the
// MHATokenToKVPool FP8 store; the gated norm follows FLA's layernorm_gated (RMSNorm, norm before gate).
//
// math_reference: sglang
// math_reference: flash-linear-attention
#include "kernels/k1_primitives.h"

#include "cere_model_constants.h"

#include <cstddef>
#include <cmath>

#include <cooperative_groups.h>

namespace cere::k1 {

namespace cg = cooperative_groups;
namespace {

constexpr int kVec = 8;  // BF16 elements per 16-byte access

struct alignas(16) Bf16x8 {
    __nv_bfloat162 h[4];
};

__device__ __forceinline__ void load8(const bf16* p, float (&out)[kVec]) {
    const Bf16x8 v = *reinterpret_cast<const Bf16x8*>(p);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(v.h[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

__device__ __forceinline__ void store8(bf16* p, const float (&in)[kVec]) {
    Bf16x8 v;
#pragma unroll
    for (int i = 0; i < 4; ++i) v.h[i] = __floats2bfloat162_rn(in[2 * i], in[2 * i + 1]);
    *reinterpret_cast<Bf16x8*>(p) = v;
}

// Deterministic block sum (fixed shuffle tree, then warp partials in order). blockDim.x is a multiple of 32, <= 1024.
__device__ float block_sum(float v) {
    __shared__ float partial[32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    __syncthreads();  // partial[] may still be read by a previous call
    if (lane == 0) partial[warp] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < warps; ++w) total += partial[w];
    return total;
}

// block_sum of a one-warp block for the calling warp (all 32 lanes): the same xor tree, then total = 0.0f + lane 0's
// value (block_sum's partial[0]).
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    float total = 0.0f;
    total += __shfl_sync(0xffffffffu, v, 0);
    return total;
}

int norm_threads(int dim) {
    const int chunks = dim / kVec;
    int threads = ((chunks + 31) / 32) * 32;
    return threads > 256 ? 256 : threads;
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// Programmatic dependent launch (m23 design note P5R, SmallFuse; as k2_small_m's dependency_wait): the stream's
// next kernel may launch now (a launch_pdl kernel waits for this grid in its own pdl_enter), then this grid waits for
// the previous kernel's grid and its writes. Every launch_pdl kernel calls it first, before any global memory access,
// so its results and the order of its arithmetic are those of a plain launch.
__device__ __forceinline__ void pdl_enter() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

// The largest grid launched as a programmatic dependent launch: the decode/verify calls (m <= 16 rows: at most
// 16 x 48 GDN heads = 768 gated-norm blocks). A larger (prefill) grid launches plainly: its blocks, resident early
// and waiting, would hold SM slots the predecessor's later waves need (SmallFuse S2: ttft_cold_18K +1-2 %). Its
// pdl_enter wait is then a no-op and the stream orders it as before.
constexpr unsigned long long kPdlMaxBlocks = 1024;

template <typename... Params, typename... Args>
cudaError_t launch_pdl(void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                       Args... args) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1ull * grid.x * grid.y * grid.z <= kPdlMaxBlocks ? 1 : 0;
    const cudaError_t e = cudaLaunchKernelEx(&config, kernel, args...);
    const cudaError_t last = cudaGetLastError();
    return e != cudaSuccess ? e : last;
}

// One row of rmsnorm over `threads` threads (this thread is `tid`); `sum` reduces the per-thread sums of squares.
// Pass 1 sums squares, pass 2 re-reads its own chunks and writes y (so y may alias x).
template <bool kGated, typename Sum>
__device__ __forceinline__ void rmsnorm_row(const bf16* __restrict__ x, int64_t x_stride, const bf16* z,
                                            int64_t z_stride, const bf16* __restrict__ weight, int weight_groups,
                                            bool gemma, float eps, int dim, bf16* y, int64_t y_stride, int64_t row,
                                            int tid, int threads, Sum sum) {
    const bf16* xr = x + row * x_stride;
    const bf16* wr = weight + static_cast<int64_t>(row % weight_groups) * dim;
    float ss = 0.0f;
    for (int c = tid * kVec; c < dim; c += threads * kVec) {
        float v[kVec];
        load8(xr + c, v);
#pragma unroll
        for (int i = 0; i < kVec; ++i) ss += v[i] * v[i];
    }
    const float rstd = rsqrtf(sum(ss) / static_cast<float>(dim) + eps);
    bf16* yr = y + row * y_stride;
    for (int c = tid * kVec; c < dim; c += threads * kVec) {
        float v[kVec], w[kVec];
        load8(xr + c, v);
        load8(wr + c, w);
        if constexpr (kGated) {
            float g[kVec];
            load8(z + row * z_stride + c, g);
#pragma unroll
            for (int i = 0; i < kVec; ++i) v[i] = v[i] * rstd * w[i] * (1.0f / (1.0f + expf(-g[i])));
        } else {
#pragma unroll
            for (int i = 0; i < kVec; ++i) v[i] = v[i] * rstd * (gemma ? 1.0f + w[i] : w[i]);
        }
        store8(yr + c, v);
    }
}

// One block of norm_threads(dim) threads per row.
template <bool kGated>
__global__ void rmsnorm_kernel(const bf16* __restrict__ x, int64_t x_stride, const bf16* z, int64_t z_stride,
                               const bf16* __restrict__ weight, int weight_groups, bool gemma, float eps, int dim, bf16* y,
                               int64_t y_stride) {
    pdl_enter();
    rmsnorm_row<kGated>(x, x_stride, z, z_stride, weight, weight_groups, gemma, eps, dim, y, y_stride,
                        static_cast<int64_t>(blockIdx.x), static_cast<int>(threadIdx.x), static_cast<int>(blockDim.x),
                        [](float v) { return block_sum(v); });
}

// rmsnorm_kernel for dim <= 256, where its block is one warp: one warp per row, kNormWarpRows rows per block. The sum
// is warp_sum (block_sum with one warp), so each row's arithmetic is rmsnorm_kernel's; lanes past dim / 8 hold 0.0f
// in the tree as there.
constexpr int kNormWarpRows = 8;

template <bool kGated>
__global__ void __launch_bounds__(32 * kNormWarpRows)
    rmsnorm_warp_kernel(const bf16* __restrict__ x, int64_t x_stride, const bf16* z, int64_t z_stride,
                        const bf16* __restrict__ weight, int weight_groups, bool gemma, float eps, int dim, bf16* y,
                        int64_t y_stride, int64_t rows) {
    pdl_enter();
    const int64_t row = static_cast<int64_t>(blockIdx.x) * kNormWarpRows + (threadIdx.x >> 5);
    if (row >= rows) return;  // the whole warp: warp_sum's shuffles see every lane
    rmsnorm_row<kGated>(x, x_stride, z, z_stride, weight, weight_groups, gemma, eps, dim, y, y_stride, row,
                        static_cast<int>(threadIdx.x & 31), 32, [](float v) { return warp_sum(v); });
}

// Rows above which a dim <= 256 norm takes rmsnorm_warp_kernel: the decode/verify calls (kPdlMaxBlocks rows or
// fewer) keep rmsnorm_kernel.
constexpr int64_t kNormWarpMinRows = static_cast<int64_t>(kPdlMaxBlocks);

template <bool kGated>
cudaError_t launch_rmsnorm(const bf16* x, int64_t x_stride, const bf16* z, int64_t z_stride, const bf16* weight,
                           int weight_groups, bool gemma, float eps, int64_t rows, int dim, bf16* y, int64_t y_stride,
                           cudaStream_t stream) {
    const int threads = norm_threads(dim);
    if (threads == 32 && rows > kNormWarpMinRows) {
        const unsigned blocks = static_cast<unsigned>((rows + kNormWarpRows - 1) / kNormWarpRows);
        return launch_pdl(rmsnorm_warp_kernel<kGated>, dim3(blocks), dim3(32 * kNormWarpRows), 0, stream, x, x_stride,
                          z, z_stride, weight, weight_groups, gemma, eps, dim, y, y_stride, rows);
    }
    return launch_pdl(rmsnorm_kernel<kGated>, dim3(static_cast<unsigned>(rows)), dim3(threads), 0, stream, x, x_stride,
                      z, z_stride, weight, weight_groups, gemma, eps, dim, y, y_stride);
}

__global__ void rope_table_kernel(float* __restrict__ cos_sin, int64_t n_pos, int rotary_dim, float theta) {
    const int half = rotary_dim / 2;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_pos * half) return;
    const int64_t p = idx / half;
    const int i = static_cast<int>(idx % half);
    // torch on CUDA: arange(0, rd, 2, float) / rd, theta ** that, 1.0 / that, t * inv_freq (einsum), cos, sin.
    const float exponent = static_cast<float>(2 * i) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0f / powf(theta, exponent);
    const float angle = static_cast<float>(p) * inv_freq;
    cos_sin[p * rotary_dim + i] = cosf(angle);
    cos_sin[p * rotary_dim + half + i] = sinf(angle);
}

__device__ __forceinline__ int64_t position_of(const int64_t* positions, int64_t pos0, int64_t t) {
    return positions ? positions[t] : pos0 + t;
}

// M10 (m213 design note D11): the mRoPE axis of frequency i under the interleaved mrope_section [11, 11, 10]
// (MRotaryEmbedding apply_interleaved_rope: x_t[1:33:3] from axis h, x_t[2:30:3] from axis w, the rest from t).
constexpr int kMropeSectionH = gen::kMropeSection[1];
constexpr int kMropeSectionW = gen::kMropeSection[2];
constexpr int kMropeRotaryDim = 2 * (gen::kMropeSection[0] + kMropeSectionH + kMropeSectionW);  // 64
__device__ __forceinline__ int mrope_axis(int i) {
    if (i % 3 == 1 && i < 3 * kMropeSectionH) return 1;
    if (i % 3 == 2 && i < 3 * kMropeSectionW) return 2;
    return 0;
}

// The cos / sin of frequency i of token t: from the row of its position, or with positions3 ([3][tokens]) from the
// row of the position on frequency i's axis.
__device__ __forceinline__ float2 rope_cs(const float* cos_sin, const int64_t* positions, const int64_t* positions3,
                                          int64_t pos0, int64_t t, int64_t tokens, int rotary_dim, int i) {
    const int64_t p = positions3 ? positions3[mrope_axis(i) * tokens + t] : position_of(positions, pos0, t);
    const float* row = cos_sin + p * rotary_dim;
    return make_float2(row[i], row[rotary_dim / 2 + i]);
}

// Grid (tokens, q_heads + kv_heads); one block of ceil32(head_dim / 8) threads per (token, head).
__global__ void qk_norm_rope_gate_kernel(const bf16* __restrict__ q_gate, int64_t q_gate_stride,
                                         const bf16* __restrict__ k, int64_t k_stride, const bf16* __restrict__ q_weight,
                                         const bf16* __restrict__ k_weight, const float* __restrict__ cos_sin,
                                         const int64_t* __restrict__ positions, const int64_t* __restrict__ positions3,
                                         int64_t pos0, int q_heads, int head_dim, int rotary_dim, float eps,
                                         bf16* __restrict__ q_out, bf16* __restrict__ k_out, bf16* __restrict__ gate_out) {
    extern __shared__ float normed[];  // [head_dim], BF16-rounded normalized values
    pdl_enter();
    const int64_t t = blockIdx.x;
    const int head = blockIdx.y;
    const bool is_k = head >= q_heads;
    const int h = is_k ? head - q_heads : head;
    const bf16* in = is_k ? k + t * k_stride + static_cast<int64_t>(h) * head_dim
                          : q_gate + t * q_gate_stride + static_cast<int64_t>(h) * 2 * head_dim;
    const bf16* w = is_k ? k_weight : q_weight;
    bf16* out = is_k ? k_out + t * static_cast<int64_t>(gridDim.y - q_heads) * head_dim + static_cast<int64_t>(h) * head_dim
                     : q_out + t * static_cast<int64_t>(q_heads) * head_dim + static_cast<int64_t>(h) * head_dim;
    const int c = threadIdx.x * kVec;
    const bool active = c < head_dim;
    float v[kVec] = {};
    float ss = 0.0f;
    if (active) {
        load8(in + c, v);
#pragma unroll
        for (int i = 0; i < kVec; ++i) ss += v[i] * v[i];
    }
    const float rstd = rsqrtf(block_sum(ss) / static_cast<float>(head_dim) + eps);
    if (active) {
        float wv[kVec];
        load8(w + c, wv);
#pragma unroll
        for (int i = 0; i < kVec; ++i) normed[c + i] = __bfloat162float(__float2bfloat16_rn(v[i] * rstd * (wv[i] + 1.0f)));
        if (!is_k) {
            const bf16* g = in + head_dim + c;
            *reinterpret_cast<Bf16x8*>(gate_out + t * static_cast<int64_t>(q_heads) * head_dim +
                                       static_cast<int64_t>(h) * head_dim + c) = *reinterpret_cast<const Bf16x8*>(g);
        }
    }
    __syncthreads();
    const int half = rotary_dim / 2;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float r;
        if (d < rotary_dim) {
            const int i = d < half ? d : d - half;
            const float2 c = rope_cs(cos_sin, positions, positions3, pos0, t, gridDim.x, rotary_dim, i);
            const float cs = c.x;
            const float sn = c.y;
            const float x1 = normed[i];
            const float x2 = normed[i + half];
            r = d < half ? x1 * cs - x2 * sn : x2 * cs + x1 * sn;
        } else {
            r = normed[d];
        }
        out[d] = __float2bfloat16_rn(r);
    }
}

// Grid (tokens, heads); each block reads its head row into shared memory before writing (y may alias x).
__global__ void rope_kernel(const bf16* x, int64_t x_stride, bf16* y, int64_t y_stride, const float* __restrict__ cos_sin,
                            const int64_t* __restrict__ positions, const int64_t* __restrict__ positions3, int64_t pos0,
                            int head_dim, int rotary_dim) {
    extern __shared__ float vals[];  // [head_dim]
    const int64_t t = blockIdx.x;
    const bf16* in = x + t * x_stride + static_cast<int64_t>(blockIdx.y) * head_dim;
    bf16* out = y + t * y_stride + static_cast<int64_t>(blockIdx.y) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) vals[d] = __bfloat162float(in[d]);
    __syncthreads();
    const int half = rotary_dim / 2;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float r;
        if (d < rotary_dim) {
            const int i = d < half ? d : d - half;
            const float2 c = rope_cs(cos_sin, positions, positions3, pos0, t, gridDim.x, rotary_dim, i);
            const float cs = c.x;
            const float sn = c.y;
            const float x1 = vals[i];
            const float x2 = vals[i + half];
            r = d < half ? x1 * cs - x2 * sn : x2 * cs + x1 * sn;
        } else {
            r = vals[d];
        }
        out[d] = __float2bfloat16_rn(r);
    }
}

__global__ void embedding_kernel(const bf16* __restrict__ table, int hidden, const int32_t* __restrict__ ids,
                                 bf16* __restrict__ out) {
    pdl_enter();
    const int64_t t = blockIdx.y;
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    if (c >= hidden) return;
    const int64_t id = ids[t];
    *reinterpret_cast<Bf16x8*>(out + t * hidden + c) = *reinterpret_cast<const Bf16x8*>(table + id * hidden + c);
}

__global__ void sigmoid_mul_kernel(const bf16* x, int64_t x_stride, const bf16* gate, int64_t gate_stride, int width,
                                   bf16* out, int64_t out_stride) {
    pdl_enter();
    const int64_t r = blockIdx.y;
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    if (c >= width) return;
    float a[kVec], g[kVec];
    load8(x + r * x_stride + c, a);
    load8(gate + r * gate_stride + c, g);
#pragma unroll
    for (int i = 0; i < kVec; ++i) a[i] = a[i] * (1.0f / (1.0f + expf(-g[i])));
    store8(out + r * out_stride + c, a);
}

// torch.argmax order: NaN beats everything (first NaN wins), then larger value, then lower index. A candidate is one
// 64-bit key whose unsigned order is that order: the BF16 logit's order-preserving 16-bit key (sign-magnitude to
// offset binary; -0 keyed as +0, as they compare equal; every NaN keyed 0xffff, above +Inf) << 32 | ~index, so equal
// logits order by lower index. Every key of a real logit is >= 0x7f << 32 (-Inf), so 0 is "no candidate" (kNone).
constexpr uint32_t kNone = 0xffffffffu;

__device__ __forceinline__ uint64_t top2_key(uint16_t bits, uint32_t i) {
    uint32_t k;
    if ((bits & 0x7fffu) > 0x7f80u) {
        k = 0xffffu;
    } else if (bits == 0x8000u) {
        k = 0x8000u;
    } else {
        k = (bits & 0x8000u) ? (~bits & 0xffffu) : (bits | 0x8000u);
    }
    return (static_cast<uint64_t>(k) << 32) | static_cast<uint32_t>(~i);
}

// Keys are distinct (distinct indices), so (best, second) is the top-2 of the keys pushed, in any order.
__device__ __forceinline__ void top2_push(uint64_t& best, uint64_t& second, uint64_t p) {
    if (p > second) {
        if (p > best) {
            second = best;
            best = p;
        } else {
            second = p;
        }
    }
}

// The top-2 of two top-2 pairs: the larger best, then the larger of the smaller best and the larger second.
__device__ __forceinline__ void top2_merge(uint64_t& best, uint64_t& second, uint64_t ob, uint64_t os) {
    const uint64_t lo = best < ob ? best : ob;
    const uint64_t hs = second > os ? second : os;
    best = best > ob ? best : ob;
    second = lo > hs ? lo : hs;
}

__device__ __forceinline__ void top2_warp(uint64_t& best, uint64_t& second) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        top2_merge(best, second, __shfl_xor_sync(0xffffffffu, best, o), __shfl_xor_sync(0xffffffffu, second, o));
    }
}

constexpr int kTop2Threads = 1024;
constexpr int kTop2Cluster = 8;  // CTAs per row: one portable thread block cluster
constexpr int kTop2Batch = 4;    // 16-byte words in flight per thread

// Row blockIdx.y of logits is split over the kTop2Cluster CTAs of one cluster (thread g = rank * blockDim + tid of
// the cluster's kTop2Cluster * blockDim). Row r writes ids[r * out_stride + {0, 1}] and scores[r * out_stride +
// {0, 1}] (4-byte words): the token-score struct is ids = &id, scores = &logit, out_stride 4; separate [rows, 2]
// arrays use out_stride 2. The top-2 of the keys is a function of the candidate set, so the scan order (16-byte
// words of 8 logits when the row is 16-byte aligned, then the tail), the split and the merge tree (lanes, warps,
// then warp 0 of rank 0 over the ranks' shared memory) do not change the result. A score is the winner's logit read
// back (its own bits: -0 stays -0, a NaN keeps its payload); a missing second (vocab 1) is (kNone, -Inf).
__global__ void __cluster_dims__(kTop2Cluster, 1, 1) __launch_bounds__(kTop2Threads)
    top2_kernel(const bf16* __restrict__ logits, int64_t stride, int64_t vocab, uint32_t* ids, float* scores,
                int out_stride) {
    __shared__ uint64_t sb[kTop2Threads / 32], ss[kTop2Threads / 32];
    const int64_t r = blockIdx.y;
    const bf16* row = logits + r * stride;
    const uint16_t* bits = reinterpret_cast<const uint16_t*>(row);
    const int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    uint64_t best = 0, second = 0;
    int64_t scalar_from = 0;
    if ((reinterpret_cast<uintptr_t>(row) & 15u) == 0) {
        const int64_t words = vocab / 8;
        const uint4* w = reinterpret_cast<const uint4*>(row);
        for (int64_t c0 = g; c0 < words; c0 += step * kTop2Batch) {
            uint4 raw[kTop2Batch];
#pragma unroll
            for (int b = 0; b < kTop2Batch; ++b) {
                const int64_t c = c0 + b * step;
                if (c < words) raw[b] = __ldg(w + c);
            }
#pragma unroll
            for (int b = 0; b < kTop2Batch; ++b) {
                const int64_t c = c0 + b * step;
                if (c >= words) break;
                const uint32_t* h = reinterpret_cast<const uint32_t*>(&raw[b]);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const uint32_t i = static_cast<uint32_t>(c * 8 + 2 * e);
                    top2_push(best, second, top2_key(static_cast<uint16_t>(h[e] & 0xffffu), i));
                    top2_push(best, second, top2_key(static_cast<uint16_t>(h[e] >> 16), i + 1));
                }
            }
        }
        scalar_from = words * 8;
    }
    for (int64_t j = scalar_from + g; j < vocab; j += step)
        top2_push(best, second, top2_key(bits[j], static_cast<uint32_t>(j)));
    top2_warp(best, second);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) {
        sb[warp] = best;
        ss[warp] = second;
    }
    __syncthreads();
    if (warp == 0) {
        const bool live = lane < static_cast<int>(blockDim.x >> 5);
        best = live ? sb[lane] : 0;
        second = live ? ss[lane] : 0;
        top2_warp(best, second);  // every lane has read its slot before lane 0 overwrites slot 0
        if (lane == 0) {
            sb[0] = best;
            ss[0] = second;
        }
    }
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();  // every rank's top-2 is in its sb[0] / ss[0]
    if (cl.block_rank() == 0 && warp == 0) {
        const bool live = lane < static_cast<int>(cl.num_blocks());
        best = live ? *cl.map_shared_rank(&sb[0], lane) : 0;
        second = live ? *cl.map_shared_rank(&ss[0], lane) : 0;
        top2_warp(best, second);
        if (lane < 2) {
            const uint64_t p = lane == 0 ? best : second;
            const uint32_t i = p == 0 ? kNone : ~static_cast<uint32_t>(p);
            const int64_t o = r * out_stride + lane;
            ids[o] = i;
            scores[o] = p == 0 ? -INFINITY : __bfloat162float(row[i]);
        }
    }
    cl.sync();  // no rank exits while rank 0 reads its shared memory
}

// torch's float -> float8_e4m3fn conversion (torch 2.13 fp8e4m3fn_from_fp32_value, used by `.to(float8_e4m3fn)`):
// round to nearest even; finite overflow (|x| >= 480 or a rounding carry past 448) and Inf saturate to +-448 (0x7E),
// NaN gives NaN (0x7F) with the input's sign.
__device__ __forceinline__ uint8_t fp8_e4m3(float x) {
    const uint32_t bits = __float_as_uint(x);
    const uint32_t sign = (bits >> 24) & 0x80u;
    const uint32_t a = bits & 0x7fffffffu;
    uint32_t r;
    if (a > 0x7f800000u) {  // NaN
        r = 0x7fu;
    } else if (a >= 0x43f00000u) {  // |x| >= 480 or Inf
        r = 0x7eu;
    } else if (a < 0x3c800000u) {  // |x| < 2^-6: subnormal, units of 2^-9 (exact scaling, RNE to integer)
        r = static_cast<uint32_t>(__float2int_rn(__uint_as_float(a) * 512.0f));
    } else {  // normal: keep 3 mantissa bits with round to nearest even, rebias 127 -> 7
        const uint32_t rounded = a + 0x7ffffu + ((a >> 20) & 1u);
        r = (rounded >> 20) - ((127u - 7u) << 3);
        if (r == 0x7fu) r = 0x7eu;
    }
    return static_cast<uint8_t>(r | sign);
}

__device__ __forceinline__ uint2 pack_fp8(const float (&a)[kVec]) {
    uint2 r;
    r.x = fp8_e4m3(a[0]) | (fp8_e4m3(a[1]) << 8) | (fp8_e4m3(a[2]) << 16) | (static_cast<uint32_t>(fp8_e4m3(a[3])) << 24);
    r.y = fp8_e4m3(a[4]) | (fp8_e4m3(a[5]) << 8) | (fp8_e4m3(a[6]) << 16) | (static_cast<uint32_t>(fp8_e4m3(a[7])) << 24);
    return r;
}

__global__ void kv_store_fp8_kernel(const bf16* __restrict__ k, int64_t k_stride, const bf16* __restrict__ v,
                                    int64_t v_stride, int width, uint8_t* __restrict__ k_cache,
                                    uint8_t* __restrict__ v_cache, const int64_t* __restrict__ slots, int64_t slot0,
                                    const int64_t* __restrict__ at) {
    pdl_enter();
    const int64_t t = blockIdx.y;
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    if (c >= width) return;
    const int64_t dst = (at ? slots[*at + t] : (slots ? slots[t] : slot0 + t)) * width + c;
    float a[kVec];
    load8(k + t * k_stride + c, a);
    *reinterpret_cast<uint2*>(k_cache + dst) = pack_fp8(a);
    load8(v + t * v_stride + c, a);
    *reinterpret_cast<uint2*>(v_cache + dst) = pack_fp8(a);
}

// kv_store_fp8_kernel with row t's caches and slot from `rows` (kv_store_fp8_rows).
__global__ void kv_store_fp8_rows_kernel(const bf16* __restrict__ k, int64_t k_stride, const bf16* __restrict__ v,
                                         int64_t v_stride, int width, const KvRows rows,
                                         const int64_t* __restrict__ at) {
    pdl_enter();
    const int64_t t = blockIdx.y;
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;
    if (c >= width) return;
    const int64_t dst = rows.slots[t][at[t]] * width + c;
    float a[kVec];
    load8(k + t * k_stride + c, a);
    *reinterpret_cast<uint2*>(rows.k_cache[t] + dst) = pack_fp8(a);
    load8(v + t * v_stride + c, a);
    *reinterpret_cast<uint2*>(rows.v_cache[t] + dst) = pack_fp8(a);
}

dim3 vec_grid(int width, int64_t rows, int threads) {
    const int vecs = width / kVec;
    return dim3(static_cast<unsigned>((vecs + threads - 1) / threads), static_cast<unsigned>(rows));
}

constexpr int64_t kMaxGridY = 65535;

}  // namespace

cudaError_t embedding_gather(const bf16* table, int64_t vocab, int hidden, const int32_t* ids, int64_t n, bf16* out,
                             cudaStream_t stream) {
    if (!table || !ids || !out || vocab <= 0 || hidden <= 0 || hidden % kVec || n < 0 || n > kMaxGridY ||
        !aligned16(table) || !aligned16(out))
        return cudaErrorInvalidValue;
    if (n == 0) return cudaSuccess;
    return launch_pdl(embedding_kernel, vec_grid(hidden, n, 128), dim3(128), 0, stream, table, hidden, ids, out);
}

namespace {
bool norm_args_ok(const bf16* x, int64_t x_stride, const bf16* weight, int64_t rows, int dim, const bf16* y,
                  int64_t y_stride) {
    return x && weight && y && rows >= 0 && dim > 0 && dim % kVec == 0 && dim <= 8192 && x_stride % kVec == 0 &&
           y_stride % kVec == 0 && rows <= 0x7fffffff && aligned16(x) && aligned16(y) && aligned16(weight);
}
}  // namespace

cudaError_t rmsnorm(const bf16* x, int64_t x_stride, const bf16* weight, int weight_groups, bool gemma, float eps,
                    int64_t rows, int dim, bf16* y, int64_t y_stride, cudaStream_t stream) {
    if (!norm_args_ok(x, x_stride, weight, rows, dim, y, y_stride) || weight_groups <= 0) return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    return launch_rmsnorm<false>(x, x_stride, static_cast<const bf16*>(nullptr), int64_t{0}, weight, weight_groups, gemma,
                                 eps, rows, dim, y, y_stride, stream);
}

cudaError_t rmsnorm_gated_sigmoid(const bf16* x, int64_t x_stride, const bf16* z, int64_t z_stride, const bf16* weight,
                                  float eps, int64_t rows, int dim, bf16* y, int64_t y_stride, cudaStream_t stream) {
    if (!norm_args_ok(x, x_stride, weight, rows, dim, y, y_stride) || !z || z_stride % kVec || !aligned16(z))
        return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    return launch_rmsnorm<true>(x, x_stride, z, z_stride, weight, 1, false, eps, rows, dim, y, y_stride, stream);
}

cudaError_t rope_table(float* cos_sin, int64_t n_pos, int rotary_dim, float theta, cudaStream_t stream) {
    if (!cos_sin || n_pos < 0 || rotary_dim < 2 || rotary_dim % 2 || rotary_dim > 1024 ||
        n_pos > (int64_t{1} << 40) / rotary_dim)
        return cudaErrorInvalidValue;
    if (n_pos == 0) return cudaSuccess;
    const int64_t items = n_pos * (rotary_dim / 2);
    rope_table_kernel<<<static_cast<unsigned>((items + 255) / 256), 256, 0, stream>>>(cos_sin, n_pos, rotary_dim, theta);
    return cudaGetLastError();
}

cudaError_t qk_norm_rope_gate(const bf16* q_gate, int64_t q_gate_stride, const bf16* k, int64_t k_stride,
                              const bf16* q_weight, const bf16* k_weight, const float* cos_sin,
                              const int64_t* positions, const int64_t* positions3, int64_t pos0, int64_t tokens,
                              int q_heads, int kv_heads, int head_dim, int rotary_dim, float eps, bf16* q_out,
                              bf16* k_out, bf16* gate_out, cudaStream_t stream) {
    if (!q_gate || !k || !q_weight || !k_weight || !cos_sin || !q_out || !k_out || !gate_out || tokens < 0 ||
        tokens > 0x7fffffff || pos0 < 0 || q_heads <= 0 || kv_heads <= 0 || q_heads + kv_heads > kMaxGridY ||
        head_dim <= 0 || head_dim % kVec || head_dim > 1024 || rotary_dim < 0 || rotary_dim % 2 ||
        rotary_dim > head_dim || (positions3 != nullptr && rotary_dim != kMropeRotaryDim) || q_gate_stride % kVec ||
        k_stride % kVec || !aligned16(q_gate) || !aligned16(k) || !aligned16(q_weight) || !aligned16(k_weight) ||
        !aligned16(q_out) || !aligned16(k_out) || !aligned16(gate_out))
        return cudaErrorInvalidValue;
    if (tokens == 0) return cudaSuccess;
    const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(q_heads + kv_heads));
    return launch_pdl(qk_norm_rope_gate_kernel, grid, dim3(norm_threads(head_dim)), head_dim * sizeof(float), stream,
                      q_gate, q_gate_stride, k, k_stride, q_weight, k_weight, cos_sin, positions, positions3, pos0,
                      q_heads, head_dim, rotary_dim, eps, q_out, k_out, gate_out);
}

cudaError_t rope(const bf16* x, int64_t x_stride, bf16* y, int64_t y_stride, const float* cos_sin,
                 const int64_t* positions, const int64_t* positions3, int64_t pos0, int64_t tokens, int heads,
                 int head_dim, int rotary_dim, cudaStream_t stream) {
    if (!x || !y || !cos_sin || tokens < 0 || tokens > 0x7fffffff || pos0 < 0 || heads <= 0 || heads > kMaxGridY ||
        head_dim <= 0 || head_dim > 1024 || rotary_dim < 0 || rotary_dim % 2 || rotary_dim > head_dim ||
        (positions3 != nullptr && rotary_dim != kMropeRotaryDim))
        return cudaErrorInvalidValue;
    if (tokens == 0) return cudaSuccess;
    const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(heads));
    rope_kernel<<<grid, 64, head_dim * sizeof(float), stream>>>(x, x_stride, y, y_stride, cos_sin, positions, positions3,
                                                                 pos0, head_dim, rotary_dim);
    return cudaGetLastError();
}

cudaError_t sigmoid_mul(const bf16* x, int64_t x_stride, const bf16* gate, int64_t gate_stride, int64_t rows, int width,
                        bf16* out, int64_t out_stride, cudaStream_t stream) {
    if (!x || !gate || !out || rows < 0 || rows > kMaxGridY || width <= 0 || width % kVec || x_stride % kVec ||
        gate_stride % kVec || out_stride % kVec || !aligned16(x) || !aligned16(gate) || !aligned16(out))
        return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    return launch_pdl(sigmoid_mul_kernel, vec_grid(width, rows, 128), dim3(128), 0, stream, x, x_stride, gate,
                      gate_stride, width, out, out_stride);
}

namespace {
bool top2_args_ok(const bf16* logits, int64_t stride, int64_t rows, int64_t vocab) {
    return logits && rows >= 0 && rows <= kMaxGridY && vocab > 0 && vocab < static_cast<int64_t>(kNone) &&
           stride >= vocab;
}

dim3 top2_grid(int64_t rows) { return dim3(kTop2Cluster, static_cast<unsigned>(rows)); }
}  // namespace

cudaError_t top2(const bf16* logits, int64_t stride, int64_t rows, int64_t vocab, cere_token_score* out,
                 cudaStream_t stream) {
    if (!top2_args_ok(logits, stride, rows, vocab) || !out) return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    static_assert(sizeof(cere_token_score) == 16 && offsetof(cere_token_score, logit) == 8, "token score layout");
    top2_kernel<<<top2_grid(rows), kTop2Threads, 0, stream>>>(logits, stride, vocab, &out->id, &out->logit,
                                                              static_cast<int>(sizeof(cere_token_score) / 4));
    return cudaGetLastError();
}

cudaError_t top2_split(const bf16* logits, int64_t stride, int64_t rows, int64_t vocab, int32_t* ids, float* scores,
                       cudaStream_t stream) {
    if (!top2_args_ok(logits, stride, rows, vocab) || !ids || !scores) return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    top2_kernel<<<top2_grid(rows), kTop2Threads, 0, stream>>>(logits, stride, vocab, reinterpret_cast<uint32_t*>(ids),
                                                              scores, 2);
    return cudaGetLastError();
}

cudaError_t kv_store_fp8(const bf16* k, int64_t k_stride, const bf16* v, int64_t v_stride, int64_t tokens, int width,
                         uint8_t* k_cache, uint8_t* v_cache, const int64_t* slots, int64_t slot0,
                         cudaStream_t stream) {
    if (!k || !v || !k_cache || !v_cache || tokens < 0 || tokens > kMaxGridY || slot0 < 0 || width <= 0 ||
        width % kVec || k_stride % kVec || v_stride % kVec || !aligned16(k) || !aligned16(v) ||
        (reinterpret_cast<uintptr_t>(k_cache) & 7u) || (reinterpret_cast<uintptr_t>(v_cache) & 7u))
        return cudaErrorInvalidValue;
    if (tokens == 0) return cudaSuccess;
    return launch_pdl(kv_store_fp8_kernel, vec_grid(width, tokens, 64), dim3(64), 0, stream, k, k_stride, v, v_stride,
                      width, k_cache, v_cache, slots, slot0, static_cast<const int64_t*>(nullptr));
}

cudaError_t kv_store_fp8_at(const bf16* k, int64_t k_stride, const bf16* v, int64_t v_stride, int64_t tokens,
                            int width, uint8_t* k_cache, uint8_t* v_cache, const int64_t* slots, const int64_t* at,
                            cudaStream_t stream) {
    if (!k || !v || !k_cache || !v_cache || !slots || !at || tokens < 0 || tokens > kMaxGridY || width <= 0 ||
        width % kVec || k_stride % kVec || v_stride % kVec || !aligned16(k) || !aligned16(v) ||
        (reinterpret_cast<uintptr_t>(k_cache) & 7u) || (reinterpret_cast<uintptr_t>(v_cache) & 7u))
        return cudaErrorInvalidValue;
    if (tokens == 0) return cudaSuccess;
    return launch_pdl(kv_store_fp8_kernel, vec_grid(width, tokens, 64), dim3(64), 0, stream, k, k_stride, v, v_stride,
                      width, k_cache, v_cache, slots, int64_t{0}, at);
}

cudaError_t kv_store_fp8_rows(const bf16* k, int64_t k_stride, const bf16* v, int64_t v_stride, int64_t tokens,
                              int width, const KvRows& rows, const int64_t* at, cudaStream_t stream) {
    if (!k || !v || !at || tokens < 0 || tokens > kKvRowsMax || width <= 0 || width % kVec || k_stride % kVec ||
        v_stride % kVec || !aligned16(k) || !aligned16(v))
        return cudaErrorInvalidValue;
    for (int64_t t = 0; t < tokens; ++t) {
        if (!rows.k_cache[t] || !rows.v_cache[t] || !rows.slots[t] ||
            (reinterpret_cast<uintptr_t>(rows.k_cache[t]) & 7u) || (reinterpret_cast<uintptr_t>(rows.v_cache[t]) & 7u))
            return cudaErrorInvalidValue;
    }
    if (tokens == 0) return cudaSuccess;
    return launch_pdl(kv_store_fp8_rows_kernel, vec_grid(width, tokens, 64), dim3(64), 0, stream, k, k_stride, v,
                      v_stride, width, rows, at);
}

namespace {

constexpr int kCopyThreads = 256;
constexpr unsigned kCopyMaxBlocks = 256;  // per segment; larger segments stride

// Segments of one launch; N is 4 (the verify graph's undo copies, a small parameter block) or kMaxCopySegments.
template <int N>
struct CopyArgs {
    CopySegment s[N];
    bool vec[N];  // 16-byte units, else bytes
};
constexpr int kNarrowCopySegments = 4;
static_assert(sizeof(CopyArgs<kNarrowCopySegments>) == 200, "the narrow parameter block");
static_assert(sizeof(CopyArgs<kMaxCopySegments>) <= 32764, "a kernel parameter block (CUDA 12.1)");

bool copy_vec16(const CopySegment& s) {
    return aligned16(s.dst) && aligned16(s.src) && ((s.dst_pitch | s.src_pitch | s.width) & 15u) == 0;
}

// blockIdx.y = segment; the blocks of a segment stride over its units (16 bytes or 1 byte), row-major.
template <int N>
__global__ void __launch_bounds__(kCopyThreads) copy_segments_kernel(CopyArgs<N> a) {
    pdl_enter();
    const CopySegment& s = a.s[blockIdx.y];
    const bool vec = a.vec[blockIdx.y];
    const uint64_t per_row = vec ? s.width / 16 : s.width;
    const uint64_t units = per_row * s.rows;
    for (uint64_t u = uint64_t(blockIdx.x) * kCopyThreads + threadIdx.x; u < units;
         u += uint64_t(gridDim.x) * kCopyThreads) {
        const uint64_t r = u / per_row;
        const uint64_t c = u - r * per_row;
        if (vec) {
            const uint4* src = reinterpret_cast<const uint4*>(static_cast<const uint8_t*>(s.src) + r * s.src_pitch);
            reinterpret_cast<uint4*>(static_cast<uint8_t*>(s.dst) + r * s.dst_pitch)[c] = src[c];
        } else {
            static_cast<uint8_t*>(s.dst)[r * s.dst_pitch + c] = static_cast<const uint8_t*>(s.src)[r * s.src_pitch + c];
        }
    }
}

template <int N>
cudaError_t launch_copy_segments(const CopySegment* segments, int count, cudaStream_t stream) {
    CopyArgs<N> a{};
    uint64_t most = 0;
    for (int i = 0; i < count; ++i) {
        const CopySegment& s = segments[i];
        if (!s.dst || !s.src || s.rows > uint64_t(kMaxGridY)) return cudaErrorInvalidValue;
        a.s[i] = s;
        a.vec[i] = copy_vec16(s);
        const uint64_t units = (a.vec[i] ? s.width / 16 : s.width) * s.rows;
        most = units > most ? units : most;
    }
    if (most == 0) return cudaSuccess;
    const uint64_t blocks = (most + kCopyThreads - 1) / kCopyThreads;
    const unsigned grid_x = blocks < kCopyMaxBlocks ? unsigned(blocks) : kCopyMaxBlocks;
    return launch_pdl(copy_segments_kernel<N>, dim3(grid_x, unsigned(count)), dim3(kCopyThreads), 0, stream, a);
}

}  // namespace

cudaError_t copy_segments(const CopySegment* segments, int count, cudaStream_t stream) {
    if (!segments || count < 1 || count > kMaxCopySegments) return cudaErrorInvalidValue;
    return count <= kNarrowCopySegments ? launch_copy_segments<kNarrowCopySegments>(segments, count, stream)
                                        : launch_copy_segments<kMaxCopySegments>(segments, count, stream);
}

}  // namespace cere::k1
