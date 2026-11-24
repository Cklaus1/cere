// Inkling 2 Mini vision tower kernels (inkling design note 3.6, S7); contracts in vision.h. Compiled with
// -fmad=false like every inkling kernel: a multiply-add is fused only where written as fmaf (the attention dots).
//
// math_reference: transformers
#include "kernels/inkling/vision.h"

#include <cmath>
#include <cstdint>

namespace cere::inkling::vision {
namespace {

constexpr int kVec = 8;  // BF16 elements per 16-byte access

struct alignas(16) Bf16x8 {
    __nv_bfloat162 h[4];
};

bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

__device__ __forceinline__ void load8(const bf16 *p, float (&out)[kVec]) {
    const Bf16x8 v = *reinterpret_cast<const Bf16x8 *>(p);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(v.h[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

__device__ __forceinline__ void load8f(const float *p, float (&out)[kVec]) {
    const float4 a = *reinterpret_cast<const float4 *>(p);
    const float4 b = *reinterpret_cast<const float4 *>(p + 4);
    out[0] = a.x, out[1] = a.y, out[2] = a.z, out[3] = a.w, out[4] = b.x, out[5] = b.y, out[6] = b.z, out[7] = b.w;
}

__device__ __forceinline__ void store8(bf16 *p, const float (&v)[kVec]) {
    Bf16x8 o;
#pragma unroll
    for (int i = 0; i < 4; ++i) o.h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
    *reinterpret_cast<Bf16x8 *>(p) = o;
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

unsigned elementwise_blocks(int64_t items) {
    const int64_t blocks = (items + 255) / 256;
    return static_cast<unsigned>(blocks < 1 ? 1 : (blocks > 65535 * 16 ? 65535 * 16 : blocks));
}

bool images_ok(const Images &images) {
    if (images.count <= 0 || images.count > kMaxImages || images.max_seq <= 0) return false;
    for (int i = 0; i < images.count; ++i) {
        const Image &im = images.image[i];
        if (im.h < kMerge || im.w < kMerge || im.h % kMerge || im.w % kMerge || im.seq_start < 0 ||
            im.patch_start < 0 || (im.has_class != 0 && im.has_class != 1) ||
            static_cast<int64_t>(im.h) * im.w + im.has_class > images.max_seq)
            return false;
    }
    return true;
}

// Patch p of an image of patch grid (h, w) in merge-block order -> (row, col).
__device__ __forceinline__ void block_order(int p, int w, int *row, int *col) {
    const int blocks_w = w / kMerge;
    const int in_col = p % kMerge;
    const int in_row = (p / kMerge) % kMerge;
    const int block_col = (p / (kMerge * kMerge)) % blocks_w;
    const int block_row = p / (kMerge * kMerge * blocks_w);
    *row = block_row * kMerge + in_row;
    *col = block_col * kMerge + in_col;
}

// ---- bias / activation / residual -----------------------------------------------------------------------------------

__device__ __forceinline__ float gelu_erf(float x) {
    constexpr float kAlpha = 0.70710678118654752f;  // M_SQRT1_2 (PyTorch GeluCUDAKernelImpl, approximate none)
    return x * 0.5f * (1.0f + erff(x * kAlpha));
}

__global__ void bias_act_kernel(const float *__restrict__ acc, const bf16 *__restrict__ bias, int64_t rows, int n,
                                int act, bf16 *__restrict__ y, int64_t ldy) {
    const int64_t chunks = n / kVec;
    const int64_t total = rows * chunks;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t r = idx / chunks;
        const int c = static_cast<int>(idx % chunks) * kVec;
        float a[kVec], b[kVec] = {};
        load8f(acc + r * n + c, a);
        if (bias != nullptr) load8(bias + c, b);
        float out[kVec];
#pragma unroll
        for (int i = 0; i < kVec; ++i) out[i] = round_bf16(a[i] + b[i]);
        if (act == static_cast<int>(Act::kGeluErf)) {
#pragma unroll
            for (int i = 0; i < kVec; ++i) out[i] = gelu_erf(out[i]);
        }
        store8(y + r * ldy + c, out);
    }
}

__global__ void scaled_residual_kernel(const float *__restrict__ acc, const bf16 *__restrict__ bias,
                                       const bf16 *__restrict__ gamma, int64_t rows, bf16 *__restrict__ x) {
    constexpr int64_t chunks = kHidden / kVec;
    const int64_t total = rows * chunks;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t r = idx / chunks;
        const int c = static_cast<int>(idx % chunks) * kVec;
        float a[kVec], b[kVec], g[kVec], res[kVec], out[kVec];
        load8f(acc + r * kHidden + c, a);
        load8(bias + c, b);
        load8(gamma + c, g);
        load8(x + r * kHidden + c, res);
#pragma unroll
        for (int i = 0; i < kVec; ++i) out[i] = res[i] + round_bf16(g[i] * round_bf16(a[i] + b[i]));
        store8(x + r * kHidden + c, out);
    }
}

// ---- LayerNorm ------------------------------------------------------------------------------------------------------

int norm_threads(int dim) {
    const int chunks = dim / kVec;
    int threads = ((chunks + 31) / 32) * 32;
    return threads > 256 ? 256 : threads;
}

__global__ void layernorm_kernel(const bf16 *__restrict__ x, int64_t ldx, const bf16 *__restrict__ weight,
                                 const bf16 *__restrict__ bias, float eps, int dim, bf16 *__restrict__ y,
                                 int64_t ldy) {
    const int64_t row = blockIdx.x;
    const bf16 *xr = x + row * ldx;
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
    bf16 *yr = y + row * ldy;
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

// ---- embeddings -----------------------------------------------------------------------------------------------------

// _interpolation_axis_taps_weights (bilinear, align_corners false, border): src = (index + 0.5) * side / size - 0.5
// in FP32, taps floor and floor + 1 clamped to [0, side - 1], weights max(1 - |src - tap_unclamped|, 0).
__device__ __forceinline__ void axis_taps(int index, int size, int taps[2], float weights[2]) {
    const float src = (static_cast<float>(index) + 0.5f) * static_cast<float>(kPosSide) / static_cast<float>(size) -
                      0.5f;
    const float fl = floorf(src);
    const int base = static_cast<int>(fl);
#pragma unroll
    for (int o = 0; o < 2; ++o) {
        const int t = base + o;
        taps[o] = t < 0 ? 0 : (t > kPosSide - 1 ? kPosSide - 1 : t);
        const float distance = fabsf(src - fl - static_cast<float>(o));
        weights[o] = fmaxf(1.0f - distance, 0.0f);
    }
}

// Grid (max_seq, images), 128 threads x 8 columns.
__global__ void embed_kernel(const float *__restrict__ acc, const bf16 *__restrict__ table,
                             const bf16 *__restrict__ class_embedding, const bf16 *__restrict__ cls_position,
                             Images images, bf16 *__restrict__ seq) {
    const Image im = images.image[blockIdx.y];
    const int r = blockIdx.x;
    if (r >= im.has_class + im.h * im.w) return;
    bf16 *out = seq + (static_cast<int64_t>(im.seq_start) + r) * kHidden;
    const int c = threadIdx.x * kVec;
    float o[kVec];
    if (r < im.has_class) {
        float a[kVec], b[kVec];
        load8(class_embedding + c, a);
        load8(cls_position + c, b);
#pragma unroll
        for (int i = 0; i < kVec; ++i) o[i] = a[i] + b[i];
        store8(out + c, o);
        return;
    }
    const int p = r - im.has_class;
    int row, col;
    block_order(p, im.w, &row, &col);
    int ht[2], wt[2];
    float hw[2], ww[2];
    axis_taps(row, im.h, ht, hw);
    axis_taps(col, im.w, wt, ww);
    float pos[kVec] = {};
#pragma unroll
    for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const float weight = hw[a] * ww[b];
            float e[kVec];
            load8(table + static_cast<int64_t>(ht[a] * kPosSide + wt[b]) * kHidden + c, e);
#pragma unroll
            for (int i = 0; i < kVec; ++i) pos[i] += e[i] * weight;
        }
    }
    float x[kVec];
    load8f(acc + (static_cast<int64_t>(im.patch_start) + p) * kHidden + c, x);
#pragma unroll
    for (int i = 0; i < kVec; ++i) o[i] = round_bf16(x[i]) + round_bf16(pos[i]);
    store8(out + c, o);
}

// ---- q / k / v and the axial RoPE -----------------------------------------------------------------------------------

struct InvFreq {
    float f[kRopeFreqs];
};

// Grid (max_seq, images), 256 threads: thread t owns the interleaved pairs t, t + 256 of the row's 512 q/k pairs
// (q: pairs 0..511 over the 16 heads, the same for k) and v's columns.
__global__ void qkv_rope_kernel(const float *__restrict__ acc_q, const float *__restrict__ acc_k,
                                const float *__restrict__ acc_v, const bf16 *__restrict__ bias_q,
                                const bf16 *__restrict__ bias_k, const bf16 *__restrict__ bias_v, Images images,
                                InvFreq inv, int swap_axes, bf16 *__restrict__ q, bf16 *__restrict__ k,
                                bf16 *__restrict__ v) {
    const Image im = images.image[blockIdx.y];
    const int r = blockIdx.x;
    if (r >= im.has_class + im.h * im.w) return;
    int hpos = 0, wpos = 0;
    if (r >= im.has_class) {
        int row, col;
        block_order(r - im.has_class, im.w, &row, &col);
        hpos = row + 1;
        wpos = col + 1;
    }
    if (swap_axes) {
        const int t = hpos;
        hpos = wpos;
        wpos = t;
    }
    const int64_t base = (static_cast<int64_t>(im.seq_start) + r) * kHidden;
    for (int pair = threadIdx.x; pair < kHidden / 2; pair += blockDim.x) {
        const int col = 2 * pair;
        const int j = pair % (kHeadDim / 2);  // the pair within its head
        const float angle = j < kRopeFreqs ? static_cast<float>(wpos) * inv.f[j]
                                           : static_cast<float>(hpos) * inv.f[j - kRopeFreqs];
        const float cs = cosf(angle);
        const float sn = sinf(angle);
#pragma unroll
        for (int which = 0; which < 2; ++which) {
            const float *acc = which ? acc_k : acc_q;
            const bf16 *bias = which ? bias_k : bias_q;
            const float x1 = round_bf16(acc[base + col] + __bfloat162float(bias[col]));
            const float x2 = round_bf16(acc[base + col + 1] + __bfloat162float(bias[col + 1]));
            // q * cos + rotate_pairwise(q) * sin, rotate_pairwise(x1, x2) = (-x2, x1)
            const float y1 = x1 * cs + (-x2) * sn;
            const float y2 = x2 * cs + x1 * sn;
            *reinterpret_cast<__nv_bfloat162 *>((which ? k : q) + base + col) = __floats2bfloat162_rn(y1, y2);
        }
        const float v1 = acc_v[base + col] + __bfloat162float(bias_v[col]);
        const float v2 = acc_v[base + col + 1] + __bfloat162float(bias_v[col + 1]);
        *reinterpret_cast<__nv_bfloat162 *>(v + base + col) = __floats2bfloat162_rn(v1, v2);
    }
}

// ---- attention ------------------------------------------------------------------------------------------------------

constexpr int kAttnRows = 32;   // query rows per block
constexpr int kAttnLanes = 8;   // threads per query row
constexpr int kAttnKeys = 64;   // keys per tile
constexpr int kAttnThreads = kAttnRows * kAttnLanes;
constexpr int kKeysPerLane = kAttnKeys / kAttnLanes;
constexpr int kDimsPerLane = kHeadDim / kAttnLanes;  // 8
constexpr int kKsStride = kHeadDim + 1;
constexpr int kPsStride = kAttnKeys + 1;
static_assert(kHeadDim % kAttnLanes == 0 && kAttnKeys % kAttnLanes == 0, "attention tiling");

// Grid (row tiles of the longest segment, heads, images). Each block: kAttnRows query rows of one image segment and
// one head; each row is owned by kAttnLanes consecutive threads (scores of keys lane + 8m, output dims lane + 8i),
// online softmax in base 2 over key tiles of the segment.
__global__ void __launch_bounds__(kAttnThreads) attention_kernel(const bf16 *__restrict__ q,
                                                                 const bf16 *__restrict__ k,
                                                                 const bf16 *__restrict__ v, Images images,
                                                                 float scale_log2, bf16 *__restrict__ out) {
    __shared__ float ks[kAttnKeys * kKsStride];
    __shared__ float vs[kAttnKeys * kHeadDim];
    __shared__ float ps[kAttnRows * kPsStride];
    const Image im = images.image[blockIdx.z];
    const int64_t seq_len = im.has_class + static_cast<int64_t>(im.h) * im.w;
    const int64_t row0 = static_cast<int64_t>(blockIdx.x) * kAttnRows;
    if (row0 >= seq_len) return;  // uniform over the block
    const int64_t base = im.seq_start;
    const int head = blockIdx.y;
    const int row = threadIdx.x / kAttnLanes;
    const int lane = threadIdx.x % kAttnLanes;
    const int64_t qrow = row0 + row;
    const bool valid = qrow < seq_len;

    float qv[kHeadDim];
    {
        const bf16 *qp = q + (base + (valid ? qrow : 0)) * kHidden + head * kHeadDim;
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
                kval = __bfloat162float(k[(base + key) * kHidden + head * kHeadDim + d]);
                vval = __bfloat162float(v[(base + key) * kHidden + head * kHeadDim + d]);
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
    bf16 *o = out + (base + qrow) * kHidden + head * kHeadDim;
#pragma unroll
    for (int i = 0; i < kDimsPerLane; ++i) o[lane + kAttnLanes * i] = __float2bfloat16_rn(acc[i] * inv);
}

}  // namespace

cudaError_t bias_act(const float *acc, const bf16 *bias, int64_t rows, int n, Act act, bf16 *y, int64_t ldy,
                     cudaStream_t stream) {
    const int mode = static_cast<int>(act);
    if (!acc || !y || rows < 0 || n <= 0 || n % kVec || ldy < n || ldy % kVec || mode < 0 || mode > 1 ||
        !aligned16(acc) || !aligned16(y) || (bias != nullptr && !aligned16(bias)))
        return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    bias_act_kernel<<<elementwise_blocks(rows * (n / kVec)), 256, 0, stream>>>(acc, bias, rows, n, mode, y, ldy);
    return cudaGetLastError();
}

cudaError_t scaled_residual(const float *acc, const bf16 *bias, const bf16 *gamma, int64_t rows, bf16 *x,
                            cudaStream_t stream) {
    if (!acc || !bias || !gamma || !x || rows < 0 || !aligned16(acc) || !aligned16(bias) || !aligned16(gamma) ||
        !aligned16(x))
        return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    scaled_residual_kernel<<<elementwise_blocks(rows * (kHidden / kVec)), 256, 0, stream>>>(acc, bias, gamma, rows,
                                                                                            x);
    return cudaGetLastError();
}

cudaError_t layernorm(const bf16 *x, int64_t ldx, const bf16 *weight, const bf16 *bias, float eps, int64_t rows,
                      int dim, bf16 *y, int64_t ldy, cudaStream_t stream) {
    if (!x || !weight || !bias || !y || rows < 0 || rows > 0x7fffffff || dim <= 0 || dim % kVec || dim > 8192 ||
        ldx % kVec || ldy % kVec || ldx < dim || ldy < dim || !aligned16(x) || !aligned16(y) || !aligned16(weight) ||
        !aligned16(bias))
        return cudaErrorInvalidValue;
    if (rows == 0) return cudaSuccess;
    layernorm_kernel<<<static_cast<unsigned>(rows), norm_threads(dim), 0, stream>>>(x, ldx, weight, bias, eps, dim, y,
                                                                                    ldy);
    return cudaGetLastError();
}

cudaError_t embed(const float *acc, const bf16 *table, const bf16 *class_embedding, const bf16 *cls_position,
                  const Images &images, bf16 *seq, cudaStream_t stream) {
    if (!acc || !table || !class_embedding || !cls_position || !seq || !images_ok(images) || !aligned16(acc) ||
        !aligned16(table) || !aligned16(class_embedding) || !aligned16(cls_position) || !aligned16(seq))
        return cudaErrorInvalidValue;
    embed_kernel<<<dim3(static_cast<unsigned>(images.max_seq), static_cast<unsigned>(images.count)), kHidden / kVec, 0,
                   stream>>>(acc, table, class_embedding, cls_position, images, seq);
    return cudaGetLastError();
}

cudaError_t qkv_rope(const float *acc_q, const float *acc_k, const float *acc_v, const bf16 *bias_q,
                     const bf16 *bias_k, const bf16 *bias_v, const Images &images, bool swap_axes, bf16 *q, bf16 *k,
                     bf16 *v, cudaStream_t stream) {
    if (!acc_q || !acc_k || !acc_v || !bias_q || !bias_k || !bias_v || !q || !k || !v || !images_ok(images))
        return cudaErrorInvalidValue;
    // compute_axial_rope_parameters: inv_freq = 1 / theta^(arange(0, 32, 2) / 32) in FP32.
    InvFreq inv;
    for (int f = 0; f < kRopeFreqs; ++f)
        inv.f[f] = 1.0f / std::pow(kRopeTheta, static_cast<float>(2 * f) / static_cast<float>(kHeadDim / 2));
    qkv_rope_kernel<<<dim3(static_cast<unsigned>(images.max_seq), static_cast<unsigned>(images.count)), 256, 0,
                      stream>>>(acc_q, acc_k, acc_v, bias_q, bias_k, bias_v, images, inv, swap_axes ? 1 : 0, q, k, v);
    return cudaGetLastError();
}

cudaError_t attention(const bf16 *q, const bf16 *k, const bf16 *v, const Images &images, bf16 *out,
                      cudaStream_t stream) {
    if (!q || !k || !v || !out || !images_ok(images)) return cudaErrorInvalidValue;
    const unsigned tiles = static_cast<unsigned>((images.max_seq + kAttnRows - 1) / kAttnRows);
    const float scale_log2 = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHeadDim)) * 1.4426950408889634);
    attention_kernel<<<dim3(tiles, kHeads, static_cast<unsigned>(images.count)), kAttnThreads, 0, stream>>>(
        q, k, v, images, scale_log2, out);
    return cudaGetLastError();
}

}  // namespace cere::inkling::vision
