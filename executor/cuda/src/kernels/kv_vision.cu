// KV vision kernels (m213 design note D10, D11); contracts in kv_vision.h. Math follows SGLang's
// Qwen3VLMoeVisionModel (sglang/srt/models/qwen3_vl.py): pos_embed interpolation `fast_pos_embed_interpolate_from_list`
// (:547-606), 2D RoPE `rot_pos_emb` (:442-459) + `apply_rotary_pos_emb_native` (layers/rotary_embedding/utils.py),
// LayerNorm (nn.LayerNorm, eps 1e-6), GELU (nn.GELU tanh / erf) on the BF16 addmm output.
//
// math_reference: sglang
#include "kernels/kv_vision.h"

#include "kernels/k1_primitives.h"

#include <cmath>
#include <cstdint>

namespace cere::kv {
namespace {

constexpr int kVec = 8;  // BF16 elements per 16-byte access
constexpr int64_t kMaxGrid = 0x7fffffff;

struct alignas(16) Bf16x8 {
    __nv_bfloat162 h[4];
};

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

__device__ __forceinline__ void load8(const bf16* p, float (&out)[kVec]) {
    const Bf16x8 v = *reinterpret_cast<const Bf16x8*>(p);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(v.h[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

__device__ __forceinline__ void store8(bf16* p, const float (&v)[kVec]) {
    Bf16x8 o;
#pragma unroll
    for (int i = 0; i < 4; ++i) o.h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
    *reinterpret_cast<Bf16x8*>(p) = o;
}

__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

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

// ---- KV6 ----------------------------------------------------------------------------------------------------------

__device__ __forceinline__ float gelu_tanh(float x) {
    // PyTorch GeluCUDAKernelImpl, approximate="tanh".
    constexpr float kBeta = 0.7978845608028654f;  // M_SQRT2 * M_2_SQRTPI * 0.5
    constexpr float kKappa = 0.044715f;
    const float inner = kBeta * (x + kKappa * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float gelu_erf(float x) {
    constexpr float kAlpha = 0.70710678118654752f;  // M_SQRT1_2
    return x * 0.5f * (1.0f + erff(x * kAlpha));
}

__global__ void bias_act_kernel(const float* __restrict__ acc, const bf16* __restrict__ bias, const bf16* residual,
                                int64_t rows, int n, int act, bf16* y, int64_t y_stride) {
    const int64_t chunks = n / kVec;
    const int64_t total = rows * chunks;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t r = idx / chunks;
        const int c = static_cast<int>(idx % chunks) * kVec;
        const float4 a0 = *reinterpret_cast<const float4*>(acc + r * n + c);
        const float4 a1 = *reinterpret_cast<const float4*>(acc + r * n + c + 4);
        const float a[kVec] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        float b[kVec];
        load8(bias + c, b);
        float out[kVec];
#pragma unroll
        for (int i = 0; i < kVec; ++i) out[i] = round_bf16(a[i] + b[i]);
        if (act == static_cast<int>(Act::kGeluTanh)) {
#pragma unroll
            for (int i = 0; i < kVec; ++i) out[i] = gelu_tanh(out[i]);
        } else if (act == static_cast<int>(Act::kGeluErf)) {
#pragma unroll
            for (int i = 0; i < kVec; ++i) out[i] = gelu_erf(out[i]);
        } else if (residual != nullptr) {
            float res[kVec];
            load8(residual + r * y_stride + c, res);
#pragma unroll
            for (int i = 0; i < kVec; ++i) out[i] += res[i];
        }
        store8(y + r * y_stride + c, out);
    }
}

unsigned elementwise_blocks(int64_t items) {
    const int64_t blocks = (items + 255) / 256;
    return static_cast<unsigned>(blocks < 1 ? 1 : (blocks > 65535 * 16 ? 65535 * 16 : blocks));
}

// ---- KV3 ----------------------------------------------------------------------------------------------------------

int norm_threads(int dim) {
    const int chunks = dim / kVec;
    int threads = ((chunks + 31) / 32) * 32;
    return threads > 256 ? 256 : threads;
}

__global__ void layernorm_kernel(const bf16* __restrict__ x, int64_t x_stride, const bf16* __restrict__ weight,
                                 const bf16* __restrict__ bias, float eps, int dim, bf16* __restrict__ y,
                                 int64_t y_stride) {
    const int64_t row = blockIdx.x;
    const bf16* xr = x + row * x_stride;
    float sum = 0.0f;
    for (int c = threadIdx.x * kVec; c < dim; c += blockDim.x * kVec) {
        float v[kVec];
        load8(xr + c, v);
#pragma unroll
        for (int i = 0; i < kVec; ++i) sum += v[i];
    }
    const float mean = block_sum(sum) / static_cast<float>(dim);
    float ss = 0.0f;
    for (int c = threadIdx.x * kVec; c < dim; c += blockDim.x * kVec) {
        float v[kVec];
        load8(xr + c, v);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            const float d = v[i] - mean;
            ss += d * d;
        }
    }
    const float rstd = rsqrtf(block_sum(ss) / static_cast<float>(dim) + eps);
    bf16* yr = y + row * y_stride;
    for (int c = threadIdx.x * kVec; c < dim; c += blockDim.x * kVec) {
        float v[kVec], w[kVec], b[kVec], o[kVec];
        load8(xr + c, v);
        load8(weight + c, w);
        load8(bias + c, b);
#pragma unroll
        for (int i = 0; i < kVec; ++i) o[i] = (v[i] - mean) * rstd * w[i] + b[i];
        store8(yr + c, o);
    }
}

// ---- merge-window order (KV2, KV4) ---------------------------------------------------------------------------------

// Row r of an image of grid (t, h, w) in SGLang's merge-window order: frame-major, then (h / 2, w / 2) windows
// row-major, then the 2x2 patches of the window row-major.
__device__ __forceinline__ void merge_window_hw(int64_t r, int64_t h, int64_t w, int64_t* hi, int64_t* wi) {
    const int64_t rr = r % (h * w);
    const int64_t unit = rr % (kMerge * kMerge);
    const int64_t window = rr / (kMerge * kMerge);
    const int64_t wm = w / kMerge;
    *hi = (window / wm) * kMerge + unit / kMerge;
    *wi = (window % wm) * kMerge + unit % kMerge;
}

// ---- KV2 ----------------------------------------------------------------------------------------------------------

// torch.linspace(0, end, steps) on CUDA (float): the first half counts up from start, the second down from end.
__device__ __forceinline__ float linspace_at(float end, int64_t steps, int64_t i) {
    if (steps == 1) return 0.0f;
    const float step = end / static_cast<float>(steps - 1);
    return i < steps / 2 ? step * static_cast<float>(i) : end - step * static_cast<float>(steps - i - 1);
}

__device__ __forceinline__ float interp_index(int side, int64_t n, int64_t i, bool align_corners, float scale) {
    const float end = static_cast<float>(side - 1);
    if (align_corners) return linspace_at(end, n, i);
    const float idx = (static_cast<float>(i) + 0.5f) * scale - 0.5f;
    return fminf(fmaxf(idx, 0.0f), end);
}

__global__ void pos_embed_kernel(const bf16* __restrict__ table, int side, int64_t h, int64_t w, bool align_corners,
                                 float h_scale, float w_scale, const bf16* x, bf16* out) {
    const int64_t r = blockIdx.x;
    int64_t hi, wi;
    merge_window_hw(r, h, w, &hi, &wi);
    const float hf = interp_index(side, h, hi, align_corners, h_scale);
    const float wf = interp_index(side, w, wi, align_corners, w_scale);
    const int64_t h0 = static_cast<int64_t>(hf);  // .to(torch.long) truncates; indices are >= 0
    const int64_t w0 = static_cast<int64_t>(wf);
    const int64_t h1 = h0 + 1 < side ? h0 + 1 : side - 1;
    const int64_t w1 = w0 + 1 < side ? w0 + 1 : side - 1;
    const float dh = hf - static_cast<float>(h0);
    const float dw = wf - static_cast<float>(w0);
    // from_list reuses w11: w10 = dh - w11, w01 = dw - w11, w00 = 1 - dh - w01; then weights.to(bf16).
    const float w11 = dh * dw;
    const float w10 = dh - w11;
    const float w01 = dw - w11;
    const float w00 = 1.0f - dh - w01;
    const float wt[4] = {round_bf16(w00), round_bf16(w01), round_bf16(w10), round_bf16(w11)};
    const bf16* rows[4] = {table + (h0 * side + w0) * kHidden, table + (h0 * side + w1) * kHidden,
                           table + (h1 * side + w0) * kHidden, table + (h1 * side + w1) * kHidden};
    for (int c = threadIdx.x * kVec; c < kHidden; c += blockDim.x * kVec) {
        float sum[kVec] = {};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float e[kVec];
            load8(rows[k] + c, e);
#pragma unroll
            for (int i = 0; i < kVec; ++i) sum[i] += round_bf16(e[i] * wt[k]);
        }
        float o[kVec];
#pragma unroll
        for (int i = 0; i < kVec; ++i) o[i] = round_bf16(sum[i]);
        if (x != nullptr) {
            float xv[kVec];
            load8(x + r * kHidden + c, xv);
#pragma unroll
            for (int i = 0; i < kVec; ++i) o[i] += xv[i];
        }
        store8(out + r * kHidden + c, o);
    }
}

// ---- KV4 ----------------------------------------------------------------------------------------------------------

__global__ void rope2d_kernel(const bf16* __restrict__ qkv, int64_t qkv_stride, const float* __restrict__ cos_sin,
                              int64_t h, int64_t w, bf16* __restrict__ q_out, bf16* __restrict__ k_out) {
    const int64_t r = blockIdx.x;
    int64_t hi, wi;
    merge_window_hw(r, h, w, &hi, &wi);
    constexpr int kTableWidth = 2 * kRotaryHalf;  // 18 cos, 18 sin
    constexpr int kAngles = 2 * kRotaryHalf;      // 36: h then w frequencies
    constexpr int kPair = kHeadDim / 2;           // 36: NeoX pairs (i, i + 36)
    const float* h_row = cos_sin + hi * kTableWidth;
    const float* w_row = cos_sin + wi * kTableWidth;
    const bf16* in = qkv + r * qkv_stride;
    for (int e = threadIdx.x; e < 2 * kHidden; e += blockDim.x) {
        const bool is_k = e >= kHidden;
        const int col = is_k ? e - kHidden : e;
        const int d = col % kHeadDim;
        const int head_base = col - d;
        const int j = d % kAngles;  // cat(cos, cos): dim d uses angle d mod 36
        const float* row = j < kRotaryHalf ? h_row : w_row;
        const int f = j < kRotaryHalf ? j : j - kRotaryHalf;
        const float cs = row[f];
        const float sn = row[kRotaryHalf + f];
        const bf16* x = in + (is_k ? kHidden : 0) + head_base;
        const float xd = __bfloat162float(x[d]);
        // q * cos + rotate_half(q) * sin, rotate_half = cat(-x2, x1).
        const float r_val = d < kPair ? xd * cs + (-__bfloat162float(x[d + kPair])) * sn
                                      : xd * cs + __bfloat162float(x[d - kPair]) * sn;
        (is_k ? k_out : q_out)[r * kHidden + col] = __float2bfloat16_rn(r_val);
    }
}

// ---- KV5 ----------------------------------------------------------------------------------------------------------

constexpr int kAttnRows = 32;    // query rows per block
constexpr int kAttnLanes = 8;    // threads per query row
constexpr int kAttnKeys = 64;    // keys per tile
constexpr int kAttnThreads = kAttnRows * kAttnLanes;
constexpr int kKeysPerLane = kAttnKeys / kAttnLanes;
constexpr int kDimsPerLane = kHeadDim / kAttnLanes;  // 9
constexpr int kKsStride = kHeadDim + 1;
constexpr int kPsStride = kAttnKeys + 1;
static_assert(kHeadDim % kAttnLanes == 0 && kAttnKeys % kAttnLanes == 0, "KV5 tiling");

// Grid (row tiles over all segments, heads). Each block: kAttnRows query rows of one segment and one head; each row
// is owned by kAttnLanes consecutive threads (scores of keys lane + 8m, output dims lane + 8i), online softmax in
// base 2 over key tiles of the segment.
__global__ void __launch_bounds__(kAttnThreads) vision_attention_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k, int64_t qk_stride, const bf16* __restrict__ v,
    int64_t v_stride, int64_t seq_len, int64_t tiles_per_segment, float scale_log2, bf16* __restrict__ out) {
    __shared__ float ks[kAttnKeys * kKsStride];
    __shared__ float vs[kAttnKeys * kHeadDim];
    __shared__ float ps[kAttnRows * kPsStride];
    const int64_t segment = blockIdx.x / tiles_per_segment;
    const int64_t row0 = (blockIdx.x % tiles_per_segment) * kAttnRows;
    const int64_t base = segment * seq_len;
    const int head = blockIdx.y;
    const int row = threadIdx.x / kAttnLanes;
    const int lane = threadIdx.x % kAttnLanes;
    const int64_t qrow = row0 + row;
    const bool valid = qrow < seq_len;

    float qv[kHeadDim];
    {
        const bf16* qp = q + (base + (valid ? qrow : 0)) * qk_stride + head * kHeadDim;
#pragma unroll
        for (int d = 0; d < kHeadDim; ++d) qv[d] = __bfloat162float(qp[d]) * scale_log2;
    }
    float acc[kDimsPerLane] = {};
    float m = -INFINITY;
    float l = 0.0f;
    for (int64_t key0 = 0; key0 < seq_len; key0 += kAttnKeys) {
        __syncthreads();  // the previous tile's ks / vs / ps are no longer read
        for (int e = threadIdx.x; e < kAttnKeys * kHeadDim; e += kAttnThreads) {
            const int kk = e / kHeadDim;
            const int d = e % kHeadDim;
            const int64_t key = key0 + kk;
            float kval = 0.0f, vval = 0.0f;
            if (key < seq_len) {
                kval = __bfloat162float(k[(base + key) * qk_stride + head * kHeadDim + d]);
                vval = __bfloat162float(v[(base + key) * v_stride + head * kHeadDim + d]);
            }
            ks[kk * kKsStride + d] = kval;
            vs[kk * kHeadDim + d] = vval;
        }
        __syncthreads();
        float s[kKeysPerLane];
#pragma unroll
        for (int j = 0; j < kKeysPerLane; ++j) s[j] = 0.0f;
#pragma unroll 8
        for (int d = 0; d < kHeadDim; ++d) {
#pragma unroll
            for (int j = 0; j < kKeysPerLane; ++j) s[j] = fmaf(qv[d], ks[(lane + kAttnLanes * j) * kKsStride + d], s[j]);
        }
        float tile_max = -INFINITY;
#pragma unroll
        for (int j = 0; j < kKeysPerLane; ++j) {
            if (key0 + lane + kAttnLanes * j >= seq_len) s[j] = -INFINITY;
            tile_max = fmaxf(tile_max, s[j]);
        }
#pragma unroll
        for (int o = 1; o < kAttnLanes; o <<= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, o));
        const float m_new = fmaxf(m, tile_max);  // finite: every tile holds at least one key
        const float alpha = exp2f(m - m_new);
        float psum = 0.0f;
#pragma unroll
        for (int j = 0; j < kKeysPerLane; ++j) {
            const float p = exp2f(s[j] - m_new);
            psum += p;
            ps[row * kPsStride + lane + kAttnLanes * j] = p;
        }
#pragma unroll
        for (int o = 1; o < kAttnLanes; o <<= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
        l = l * alpha + psum;
        m = m_new;
        __syncwarp();
#pragma unroll
        for (int i = 0; i < kDimsPerLane; ++i) acc[i] *= alpha;
        const int keys = seq_len - key0 < kAttnKeys ? static_cast<int>(seq_len - key0) : kAttnKeys;
        for (int kk = 0; kk < keys; ++kk) {
            const float p = ps[row * kPsStride + kk];
#pragma unroll
            for (int i = 0; i < kDimsPerLane; ++i) acc[i] = fmaf(p, vs[kk * kHeadDim + lane + kAttnLanes * i], acc[i]);
        }
    }
    if (!valid) return;
    const float inv = 1.0f / l;
    bf16* o = out + (base + qrow) * kHidden + head * kHeadDim;
#pragma unroll
    for (int i = 0; i < kDimsPerLane; ++i) o[lane + kAttnLanes * i] = __float2bfloat16_rn(acc[i] * inv);
}

}  // namespace

cudaError_t bias_act(const float* acc, const bf16* bias, const bf16* residual, int64_t rows, int n, Act act, bf16* y,
                     int64_t y_stride, cudaStream_t stream) {
    const int mode = static_cast<int>(act);
    if (!acc || !bias || !y || rows < 0 || n <= 0 || n % kVec || y_stride < n || y_stride % kVec || mode < 0 ||
        mode > 2 || (residual != nullptr && act != Act::kBias) || !aligned16(acc) || !aligned16(bias) ||
        !aligned16(y) || (residual != nullptr && !aligned16(residual)))
        return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    bias_act_kernel<<<elementwise_blocks(rows * (n / kVec)), 256, 0, stream>>>(acc, bias, residual, rows, n, mode, y,
                                                                               y_stride);
    return cudaGetLastError();
}

cudaError_t layernorm(const bf16* x, int64_t x_stride, const bf16* weight, const bf16* bias, float eps, int64_t rows,
                      int dim, bf16* y, int64_t y_stride, cudaStream_t stream) {
    if (!x || !weight || !bias || !y || rows < 0 || rows > kMaxGrid || dim <= 0 || dim % kVec || dim > 8192 ||
        x_stride % kVec || y_stride % kVec || x_stride < dim || y_stride < dim || !aligned16(x) || !aligned16(y) ||
        !aligned16(weight) || !aligned16(bias))
        return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    layernorm_kernel<<<static_cast<unsigned>(rows), norm_threads(dim), 0, stream>>>(x, x_stride, weight, bias, eps, dim,
                                                                                    y, y_stride);
    return cudaGetLastError();
}

cudaError_t pos_embed(const bf16* table, int side, int64_t grid_t, int64_t grid_h, int64_t grid_w, bool align_corners,
                      const bf16* x, bf16* out, cudaStream_t stream) {
    if (!table || !out || side <= 0 || grid_t <= 0 || grid_h <= 0 || grid_w <= 0 || grid_h % kMerge ||
        grid_w % kMerge || grid_h > kMaxGrid / grid_w || grid_t > kMaxGrid / (grid_h * grid_w) || !aligned16(table) ||
        !aligned16(out) || (x != nullptr && !aligned16(x)))
        return cudaErrorInvalidValue;
    const int64_t rows = grid_t * grid_h * grid_w;
    // `side / dim_size` of _torch_interp_indices: a Python float, applied as an FP32 scalar.
    const float h_scale = static_cast<float>(static_cast<double>(side) / static_cast<double>(grid_h));
    const float w_scale = static_cast<float>(static_cast<double>(side) / static_cast<double>(grid_w));
    pos_embed_kernel<<<static_cast<unsigned>(rows), 160, 0, stream>>>(table, side, grid_h, grid_w, align_corners,
                                                                      h_scale, w_scale, x, out);
    return cudaGetLastError();
}

cudaError_t rope_table(float* cos_sin, int64_t n_pos, cudaStream_t stream) {
    return k1::rope_table(cos_sin, n_pos, 2 * kRotaryHalf, kRopeTheta, stream);
}

cudaError_t rope2d(const bf16* qkv, int64_t qkv_stride, const float* cos_sin, int64_t n_pos, int64_t grid_t,
                   int64_t grid_h, int64_t grid_w, bf16* q_out, bf16* k_out, cudaStream_t stream) {
    if (!qkv || !cos_sin || !q_out || !k_out || qkv_stride < 2 * kHidden || grid_t <= 0 || grid_h <= 0 ||
        grid_w <= 0 || grid_h % kMerge || grid_w % kMerge || grid_h > kMaxGrid / grid_w ||
        grid_t > kMaxGrid / (grid_h * grid_w) || n_pos < grid_h || n_pos < grid_w)
        return cudaErrorInvalidValue;
    const int64_t rows = grid_t * grid_h * grid_w;
    rope2d_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(qkv, qkv_stride, cos_sin, grid_h, grid_w, q_out,
                                                                   k_out);
    return cudaGetLastError();
}

cudaError_t attention(const bf16* q, const bf16* k, int64_t qk_stride, const bf16* v, int64_t v_stride, int64_t rows,
                      int64_t seq_len, bf16* out, cudaStream_t stream) {
    if (!q || !k || !v || !out || rows < 0 || seq_len <= 0 || rows % seq_len || qk_stride < kHidden ||
        v_stride < kHidden)
        return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    const int64_t tiles = (seq_len + kAttnRows - 1) / kAttnRows;
    const int64_t blocks = tiles * (rows / seq_len);
    if (blocks > kMaxGrid) return cudaErrorInvalidValue;
    const float scale_log2 = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHeadDim)) * 1.4426950408889634);
    vision_attention_kernel<<<dim3(static_cast<unsigned>(blocks), kHeads), kAttnThreads, 0, stream>>>(
        q, k, qk_stride, v, v_stride, seq_len, tiles, scale_log2, out);
    return cudaGetLastError();
}

cudaError_t splice(const bf16* embeds, int64_t image_rows, int64_t token_offset, int64_t chunk_start, int64_t n,
                   int64_t hidden, bf16* mixed, cudaStream_t stream) {
    if (!embeds || !mixed || image_rows < 0 || token_offset < 0 || chunk_start < 0 || n < 0 || hidden <= 0)
        return cudaErrorInvalidValue;
    const int64_t first = token_offset > chunk_start ? token_offset : chunk_start;
    const int64_t end_image = token_offset + image_rows;
    const int64_t end_chunk = chunk_start + n;
    const int64_t last = end_image < end_chunk ? end_image : end_chunk;
    if (first >= last) return cudaSuccess;
    const size_t row_bytes = static_cast<size_t>(hidden) * sizeof(bf16);
    return cudaMemcpyAsync(mixed + (first - chunk_start) * hidden, embeds + (first - token_offset) * hidden,
                           static_cast<size_t>(last - first) * row_bytes, cudaMemcpyDeviceToDevice, stream);
}

}  // namespace cere::kv
