// K5 (m22 design note, m22n design note): GDN (gated delta net) linear-attention kernels of Qwen3.8-Flash-Next
// layers whose layer_types entry is linear_attention. Three paths, as SGLang's GDN backend routes them:
//   - the pieces in this header: causal conv1d + SiLU with its state, gating and L2 norm;
//   - PREFILL: k5_gdn_chunk.h delta_rule_chunk, FLA's chunk-64 WY form of the gated delta rule (D2: the chunk
//     algorithm inside one prefill call is M2.2; scheduling a prompt over several calls stays M2.2c / M2.3);
//   - DECODE: k5_gdn_decode.h delta_rule_decode, FlashInfer's BF16-state single-token step (gating and L2 norm
//     in-kernel).
// BF16 activations and BF16 state at rest (Q8). k5_gdn_layer.h composes them per operation.
//
// math_reference: flash-linear-attention
// math_reference: sglang
//
// The math follows SGLang's GDN path (srt/layers/attention/linear/gdn_backend.py forward_extend/forward_decode,
// srt/models/qwen3_5.py Qwen3_5GatedDeltaNet.forward): causal_conv1d (kernels/ops/mamba/causal_conv1d_triton.py),
// fused_gdn_gating.py and l2norm.py. Nothing is vendored: these are plain CUDA kernels written from that math.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "cere_model_constants.h"

namespace cere::k5 {

constexpr int kConvWidth = gen::kLinearConvKernelDim;  // linear_conv_kernel_dim
constexpr int kConvState = kConvWidth - 1;
static_assert(gen::kLinearKeyHeadDim == gen::kLinearValueHeadDim, "one GDN head dim");
constexpr int kHeadDim = gen::kLinearKeyHeadDim;       // linear_key_head_dim == linear_value_head_dim

// Lowering variants chosen per m22n design note D3, D4 and D6 (the first one bit-exact against the control's Triton
// kernel; each agrees with the PTX Triton 3.7.1 emits for that kernel on sm_120):
//   kConvVariant 1: SiLU's exp as ex2.approx.f32(-acc * log2e) (non-ftz) and its `/` as div.full.f32.
//   kGatingVariant 4: tl.exp as ex2.approx.f32(x * log2e) (non-ftz), tl.log as the precise logf (libdevice
//     __nv_logf), tl.sigmoid as div.full.f32(1, 1 + ex2.approx.f32(-b * log2e)). D4's variants (1)-(3) use lg2.approx
//     or precise exp; the PTX of fused_gdn_gating_kernel uses neither, so this fourth variant is the one it lowers to.
//   kL2normVariant 2: l2norm_fwd_kernel as Triton specializes it for 16-byte aligned x and y (layout sizePerThread
//     [1, 8], threadsPerWarp [2, 16]): per lane fma(x0, x0, x1^2) then fma(x_j, x_j, s) for j = 2..7
//     (fma.rn.f32.bf16), a half-warp butterfly at offsets 8, 4, 2, 1, tl.sqrt as sqrt.approx.ftz.f32 and `/` as
//     div.full.f32. Variant 1 followed the unaligned specialization's order and differed in 68 of 16.8M at T 4096.
constexpr int kConvVariant = 1;
constexpr int kGatingVariant = 4;
constexpr int kL2normVariant = 2;

// Depthwise causal conv1d + SiLU over `tokens` rows of `dim` channels, no bias, in the arithmetic of SGLang's Triton
// conv (causal_conv1d_triton.py, prefill and update alike):
//   acc = 0; acc += bf16(x[t - 3 + j, c] * weight[c, j]) for j = 0..3 (FP32 adds, each product rounded to BF16;
//   tap 3 multiplies the current token);  y[t, c] = bf16(acc / (1 + exp(-acc)))  (kConvVariant lowering),
// where x[t'] for t' < 0 is state[c, t' + 3]. `state` is bf16 [dim, 3] (oldest first; zeros = fresh sequence) and on
// return holds the last 3 raw inputs. x and y are bf16 with row strides in elements (y must not alias x).
cudaError_t conv1d_silu(const __nv_bfloat16 *x, int64_t x_stride, const __nv_bfloat16 *weight, __nv_bfloat16 *state,
                        __nv_bfloat16 *y, int64_t y_stride, int64_t tokens, int64_t dim, cudaStream_t stream);
// The conv state `tokens` rows into x (the last 3 raw inputs before row `tokens`, tokens >= 3) into `state` bf16
// [dim, 3], as conv1d_silu over those rows leaves it (m210 D15⁗′ end capture inside a call). x is not written.
cudaError_t conv1d_state_at(const __nv_bfloat16 *x, int64_t x_stride, __nv_bfloat16 *state, int64_t tokens,
                            int64_t dim, cudaStream_t stream);

// conv1d_silu over the rows of the speculation chains of a verify step (m23 design note DG-02; one launch for all
// chains, rows16 design note R16-B): chain c is rows first[c] .. first[c] + rows[c] - 1 of x and y, on its own
// state states[c] (bf16 [dim, 3], distinct per chain), each chain's rows in order as conv1d_silu leaves y and the
// state (each token's arithmetic is conv1d_silu's); after row r, unless it is its chain's last row, the state a
// one-token call would leave (the last 3 raw inputs) goes to saves[r], bf16 [dim, 3] not aliasing any state (the undo
// snapshot right before row r + 1); after the last, to states[c].
constexpr int kConvMaxRows = 16;  // rows (all chains) of one launch: a verify step's rows
struct ConvChains {
    int chains;                           // 1 .. kConvMaxRows
    int first[kConvMaxRows];              // back to back: first[0] = 0, first[c + 1] = first[c] + rows[c]
    int rows[kConvMaxRows];               // >= 1 each, sum <= kConvMaxRows
    __nv_bfloat16 *states[kConvMaxRows];  // chain c's state
    __nv_bfloat16 *saves[kConvMaxRows];   // row r's save (rows that are not their chain's last)
};
cudaError_t conv1d_silu_rows(const __nv_bfloat16 *x, int64_t x_stride, const __nv_bfloat16 *weight,
                             const ConvChains &chains, __nv_bfloat16 *y, int64_t y_stride, int64_t dim,
                             cudaStream_t stream);

// fused_gdn_gating's arithmetic (kGatingVariant lowering): x = a + dt_bias (FP32), g = -exp(A_log) * softplus(x)
// (softplus(x) = log(1 + exp(x)) for x <= 20, else x), beta = bf16(sigmoid(b)) read back as FP32. a, b bf16
// [tokens, heads] with row strides; A_log f32 [heads]; dt_bias bf16 [heads]; g, beta f32 [tokens, heads] contiguous.
cudaError_t gating(const __nv_bfloat16 *a, int64_t a_stride, const __nv_bfloat16 *b, int64_t b_stride,
                   const float *A_log, const __nv_bfloat16 *dt_bias, float *g, float *beta, int64_t tokens,
                   int64_t heads, cudaStream_t stream);

// FLA l2norm_fwd's arithmetic (kL2normVariant): y = bf16(x / sqrt(sum(x^2) + eps)) per row; x, y bf16 [rows, dim]
// contiguous and 16-byte aligned (may alias); dim must be kHeadDim (the only width the GDN path normalises).
cudaError_t l2norm(const __nv_bfloat16 *x, __nv_bfloat16 *y, int64_t rows, int64_t dim, float eps,
                   cudaStream_t stream);

}  // namespace cere::k5
