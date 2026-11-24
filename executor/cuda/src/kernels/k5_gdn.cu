// K5 (m22 design note, m22n design note): GDN pieces (conv1d + state, gating, L2 norm); the delta rule lives in
// k5_gdn_chunk.cu (prefill) and k5_gdn_decode.cu (decode).
//
// math_reference: flash-linear-attention
// math_reference: sglang
//
// Written from the math of SGLang's GDN path (causal_conv1d_triton.py, fused_gdn_gating.py, l2norm.py); see
// k5_gdn.h. Nothing is vendored. Conv, gating and L2 norm follow the rounding points and FP32 lowerings of the Triton
// kernels the no-speculation control dispatches (m22n design note D3, D4, D6; the variants are named in k5_gdn.h).
#include "kernels/k5_gdn.h"

#include <cuda_bf16.h>

namespace cere::k5 {
namespace {

constexpr int kConvThreads = 256;
constexpr int kConvTokensPerThread = 16;
constexpr int kMaxGridY = 65535;
constexpr float kLog2e = 1.4426950408889634f;  // 0x3FB8AA3B, the constant Triton multiplies by before ex2

// Triton's FP32 lowerings on sm_120 (PTX of the control's kernels): tl.exp(x) = ex2.approx.f32(x * log2e), `/` =
// div.full.f32, tl.sqrt = sqrt.approx.ftz.f32. Explicit PTX keeps nvcc from substituting its own lowering.
__device__ __forceinline__ float ex2_approx(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float div_full(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float sqrt_approx_ftz(float x) {
    float y;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float triton_exp(float x) { return ex2_approx(__fmul_rn(x, kLog2e)); }

// One conv tap: Triton multiplies the two BF16 operands in BF16 (one rounding) and adds the product to the FP32 acc.
__device__ __forceinline__ float conv_tap(float acc, __nv_bfloat16 x, __nv_bfloat16 w) {
    return __fadd_rn(acc, __bfloat162float(__hmul(x, w)));
}

// acc / (1 + exp(-acc)) with -acc formed as 0 - acc, as Triton emits it.
__device__ __forceinline__ float silu_triton(float acc) {
    return div_full(acc, __fadd_rn(triton_exp(__fsub_rn(0.0f, acc)), 1.0f));
}

// y of one token from the window (the 3 raw inputs before it, oldest first) and its raw input xt; then the window
// shifts in xt.
__device__ __forceinline__ __nv_bfloat16 conv_silu_token(__nv_bfloat16 (&win)[kConvState], __nv_bfloat16 xt,
                                                         __nv_bfloat16 w0, __nv_bfloat16 w1, __nv_bfloat16 w2,
                                                         __nv_bfloat16 w3) {
    float acc = 0.0f;
    acc = conv_tap(acc, win[0], w0);
    acc = conv_tap(acc, win[1], w1);
    acc = conv_tap(acc, win[2], w2);
    acc = conv_tap(acc, xt, w3);
    win[0] = win[1];
    win[1] = win[2];
    win[2] = xt;
    return __float2bfloat16_rn(silu_triton(acc));
}

__global__ void conv1d_silu_kernel(const __nv_bfloat16 *__restrict__ x, int64_t x_stride,
                                   const __nv_bfloat16 *__restrict__ weight, const __nv_bfloat16 *__restrict__ state,
                                   __nv_bfloat16 *__restrict__ y, int64_t y_stride, int64_t tokens, int64_t dim) {
    const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= dim) {
        return;
    }
    const int64_t t0 = static_cast<int64_t>(blockIdx.y) * kConvTokensPerThread;
    const __nv_bfloat16 w0 = weight[c * kConvWidth + 0];
    const __nv_bfloat16 w1 = weight[c * kConvWidth + 1];
    const __nv_bfloat16 w2 = weight[c * kConvWidth + 2];
    const __nv_bfloat16 w3 = weight[c * kConvWidth + 3];
    __nv_bfloat16 win[kConvState];
#pragma unroll
    for (int j = 0; j < kConvState; ++j) {
        const int64_t tp = t0 - kConvState + j;
        win[j] = tp >= 0 ? x[tp * x_stride + c] : state[c * kConvState + (tp + kConvState)];
    }
    const int64_t t_end = t0 + kConvTokensPerThread < tokens ? t0 + kConvTokensPerThread : tokens;
    for (int64_t t = t0; t < t_end; ++t) {
        const __nv_bfloat16 xt = x[t * x_stride + c];
        y[t * y_stride + c] = conv_silu_token(win, xt, w0, w1, w2, w3);
    }
}

// A chain's rows on one thread per channel (grid.y = chain): the window starts as the chain's state, each token
// shifts it (as conv1d_silu_kernel does), and after row r it is the state a one-token call leaves: saved to saves[r]
// but after the chain's last row, which writes its state. The state is read before any write, so the update is in
// place.
struct ConvRows {
    const __nv_bfloat16 *x;
    int64_t x_stride;
    const __nv_bfloat16 *weight;
    ConvChains chains;
    __nv_bfloat16 *y;
    int64_t y_stride;
    int64_t dim;
};

__global__ void conv1d_silu_rows_kernel(ConvRows p) {
    const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= p.dim) {
        return;
    }
    const int chain = blockIdx.y;
    __nv_bfloat16 *const state = p.chains.states[chain];
    const int first = p.chains.first[chain];
    const int end = first + p.chains.rows[chain];
    const __nv_bfloat16 w0 = p.weight[c * kConvWidth + 0];
    const __nv_bfloat16 w1 = p.weight[c * kConvWidth + 1];
    const __nv_bfloat16 w2 = p.weight[c * kConvWidth + 2];
    const __nv_bfloat16 w3 = p.weight[c * kConvWidth + 3];
    __nv_bfloat16 win[kConvState];
#pragma unroll
    for (int j = 0; j < kConvState; ++j) {
        win[j] = state[c * kConvState + j];
    }
    for (int r = first; r < end; ++r) {
        const __nv_bfloat16 xt = p.x[r * p.x_stride + c];
        p.y[r * p.y_stride + c] = conv_silu_token(win, xt, w0, w1, w2, w3);
        __nv_bfloat16 *dst = r + 1 == end ? state : p.chains.saves[r];
#pragma unroll
        for (int j = 0; j < kConvState; ++j) {
            dst[c * kConvState + j] = win[j];
        }
    }
}

// Runs after conv1d_silu_kernel on the same stream: the new state is the last 3 raw inputs (old state entries when
// tokens < 3). Each thread reads its 3 entries before writing them, so the update is in place.
__global__ void conv1d_state_kernel(const __nv_bfloat16 *__restrict__ x, int64_t x_stride,
                                    __nv_bfloat16 *__restrict__ state, int64_t tokens, int64_t dim) {
    const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= dim) {
        return;
    }
    __nv_bfloat16 next[kConvState];
#pragma unroll
    for (int j = 0; j < kConvState; ++j) {
        const int64_t tp = tokens - kConvState + j;
        next[j] = tp >= 0 ? x[tp * x_stride + c] : state[c * kConvState + (tp + kConvState)];
    }
#pragma unroll
    for (int j = 0; j < kConvState; ++j) {
        state[c * kConvState + j] = next[j];
    }
}

// fused_gdn_gating_kernel's arithmetic; its tl.where evaluates both softplus branches and selects, the same value.
__global__ void gating_kernel(const __nv_bfloat16 *__restrict__ a, int64_t a_stride,
                              const __nv_bfloat16 *__restrict__ b, int64_t b_stride, const float *__restrict__ A_log,
                              const __nv_bfloat16 *__restrict__ dt_bias, float *__restrict__ g,
                              float *__restrict__ beta, int64_t tokens, int64_t heads) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= tokens * heads) {
        return;
    }
    const int64_t t = i / heads;
    const int64_t h = i - t * heads;
    const float x = __fadd_rn(__bfloat162float(a[t * a_stride + h]), __bfloat162float(dt_bias[h]));
    const float softplus = x <= 20.0f ? logf(__fadd_rn(triton_exp(x), 1.0f)) : x;
    g[i] = __fmul_rn(__fsub_rn(0.0f, triton_exp(A_log[h])), softplus);
    const float nb = __fsub_rn(0.0f, __bfloat162float(b[t * b_stride + h]));
    beta[i] = __bfloat162float(__float2bfloat16_rn(div_full(1.0f, __fadd_rn(triton_exp(nb), 1.0f))));
}

// l2norm_fwd_kernel (BT 16, 8 warps) as Triton specializes it for 16-byte aligned x and y (every torch allocation):
// thread layout sizePerThread [1, 8], threadsPerWarp [2, 16], so a half-warp owns a row of kHeadDim = 16 x 8 columns
// and lane l of it holds columns [8l, 8l + 8). The lane sums its squares as fma(x0, x0, x1^2), then fma(x_j, x_j, s)
// for j = 2..7 (the mixed-precision fma.rn.f32.bf16 of the BF16 element, issued as that instruction), the half-warp
// butterfly adds at offsets 8, 4, 2, 1, and every lane divides its own elements by sqrt(sum + eps).
static_assert(kHeadDim == 16 * 8, "l2norm: a half-warp of 8-column lanes per row");
__global__ void l2norm_kernel(const __nv_bfloat16 *x, __nv_bfloat16 *y, int64_t rows, float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 16) + threadIdx.x / 16;
    const int col = 8 * (threadIdx.x & 15);
    const bool valid = row < rows;  // no early return: the butterfly shuffles over the whole warp
    uint4 raw = make_uint4(0u, 0u, 0u, 0u);
    if (valid) {
        raw = *reinterpret_cast<const uint4 *>(x + row * kHeadDim + col);
    }
    const unsigned short *h = reinterpret_cast<const unsigned short *>(&raw);
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = __bfloat162float(__ushort_as_bfloat16(h[j]));
    }
    float s;
    asm("fma.rn.f32.bf16 %0, %1, %1, %2;" : "=f"(s) : "h"(h[0]), "f"(__fmul_rn(v[1], v[1])));
#pragma unroll
    for (int j = 2; j < 8; ++j) {
        asm("fma.rn.f32.bf16 %0, %1, %1, %2;" : "=f"(s) : "h"(h[j]), "f"(s));
    }
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, offset));
    }
    const float norm = sqrt_approx_ftz(__fadd_rn(s, eps));
    if (!valid) {
        return;
    }
    // Each lane reads only the elements it writes, so y may alias x.
    uint4 out;
    unsigned short *o = reinterpret_cast<unsigned short *>(&out);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        o[j] = __bfloat16_as_ushort(__float2bfloat16_rn(div_full(v[j], norm)));
    }
    *reinterpret_cast<uint4 *>(y + row * kHeadDim + col) = out;
}

}  // namespace

cudaError_t conv1d_silu(const __nv_bfloat16 *x, int64_t x_stride, const __nv_bfloat16 *weight, __nv_bfloat16 *state,
                        __nv_bfloat16 *y, int64_t y_stride, int64_t tokens, int64_t dim, cudaStream_t stream) {
    if (x == nullptr || weight == nullptr || state == nullptr || y == nullptr || tokens < 1 || dim < 1 ||
        x_stride < dim || y_stride < dim) {
        return cudaErrorInvalidValue;
    }
    const int64_t chunks = (tokens + kConvTokensPerThread - 1) / kConvTokensPerThread;
    const int64_t blocks = (dim + kConvThreads - 1) / kConvThreads;
    if (chunks > kMaxGridY || blocks > INT32_MAX) {
        return cudaErrorInvalidValue;
    }
    conv1d_silu_kernel<<<dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(chunks)), kConvThreads, 0,
                         stream>>>(x, x_stride, weight, state, y, y_stride, tokens, dim);
    conv1d_state_kernel<<<static_cast<unsigned>(blocks), kConvThreads, 0, stream>>>(x, x_stride, state, tokens, dim);
    return cudaGetLastError();
}

cudaError_t conv1d_state_at(const __nv_bfloat16 *x, int64_t x_stride, __nv_bfloat16 *state, int64_t tokens,
                            int64_t dim, cudaStream_t stream) {
    if (x == nullptr || state == nullptr || tokens < kConvState || dim < 1 || x_stride < dim) {
        return cudaErrorInvalidValue;
    }
    const int64_t blocks = (dim + kConvThreads - 1) / kConvThreads;
    if (blocks > INT32_MAX) {
        return cudaErrorInvalidValue;
    }
    conv1d_state_kernel<<<static_cast<unsigned>(blocks), kConvThreads, 0, stream>>>(x, x_stride, state, tokens, dim);
    return cudaGetLastError();
}

cudaError_t conv1d_silu_rows(const __nv_bfloat16 *x, int64_t x_stride, const __nv_bfloat16 *weight,
                             const ConvChains &chains, __nv_bfloat16 *y, int64_t y_stride, int64_t dim,
                             cudaStream_t stream) {
    if (x == nullptr || weight == nullptr || y == nullptr || dim < 1 || x_stride < dim || y_stride < dim ||
        chains.chains < 1 || chains.chains > kConvMaxRows) {
        return cudaErrorInvalidValue;
    }
    int next = 0;
    for (int c = 0; c < chains.chains; ++c) {
        if (chains.states[c] == nullptr || chains.first[c] != next || chains.rows[c] < 1 ||
            chains.rows[c] > kConvMaxRows - next) {
            return cudaErrorInvalidValue;
        }
        next += chains.rows[c];
        for (int r = chains.first[c]; r + 1 < next; ++r) {
            if (chains.saves[r] == nullptr) {
                return cudaErrorInvalidValue;
            }
            for (int s = 0; s < chains.chains; ++s) {
                if (chains.saves[r] == chains.states[s]) {
                    return cudaErrorInvalidValue;
                }
            }
        }
    }
    const int64_t blocks = (dim + kConvThreads - 1) / kConvThreads;
    if (blocks > INT32_MAX) {
        return cudaErrorInvalidValue;
    }
    const ConvRows p{x, x_stride, weight, chains, y, y_stride, dim};
    conv1d_silu_rows_kernel<<<dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(chains.chains)), kConvThreads,
                              0, stream>>>(p);
    return cudaGetLastError();
}

cudaError_t gating(const __nv_bfloat16 *a, int64_t a_stride, const __nv_bfloat16 *b, int64_t b_stride,
                   const float *A_log, const __nv_bfloat16 *dt_bias, float *g, float *beta, int64_t tokens,
                   int64_t heads, cudaStream_t stream) {
    if (a == nullptr || b == nullptr || A_log == nullptr || dt_bias == nullptr || g == nullptr || beta == nullptr ||
        tokens < 1 || heads < 1 || a_stride < heads || b_stride < heads) {
        return cudaErrorInvalidValue;
    }
    constexpr int threads = 256;
    const int64_t blocks = (tokens * heads + threads - 1) / threads;
    if (blocks > INT32_MAX) {
        return cudaErrorInvalidValue;
    }
    gating_kernel<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(a, a_stride, b, b_stride, A_log, dt_bias, g,
                                                                         beta, tokens, heads);
    return cudaGetLastError();
}

cudaError_t l2norm(const __nv_bfloat16 *x, __nv_bfloat16 *y, int64_t rows, int64_t dim, float eps,
                   cudaStream_t stream) {
    if (x == nullptr || y == nullptr || rows < 1 || dim != kHeadDim || reinterpret_cast<uintptr_t>(x) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(y) % 16 != 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int threads = 256;  // 16 rows per block, like l2norm_fwd's BT 16 x 8 warps
    constexpr int rows_per_block = threads / 16;
    const int64_t blocks = (rows + rows_per_block - 1) / rows_per_block;
    if (blocks > INT32_MAX) {
        return cudaErrorInvalidValue;
    }
    l2norm_kernel<<<static_cast<unsigned>(blocks), threads, 0, stream>>>(x, y, rows, eps);
    return cudaGetLastError();
}

}  // namespace cere::k5
