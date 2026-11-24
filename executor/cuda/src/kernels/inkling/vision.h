// Inkling 2 Mini (inkling design note 3.6, slice S7): the vision tower's elementwise, norm, position and attention
// kernels (transformers 5.24.0 `Inkling2VisionModel`, modeling_inkling2.py:983-1266, 1533-1603). The linears are K2
// GEMMs with F32 output (model/inkling/vision.h) followed by `bias_act` / `scaled_residual`, so every linear rounds to
// BF16 once as an addmm does. Gate: statistical vs the reference tower (executor/tests/gpu/
// test_inkling_kernels_vision.py), not bit-exact; the math still follows the reference's rounding points.
// All functions enqueue on `stream`, never synchronize, and return the launch status (cudaErrorInvalidValue before any
// launch on a bad argument). Tensors are row-major BF16 unless noted; strides in elements.
//
// math_reference: transformers
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cere::inkling::vision {

using bf16 = __nv_bfloat16;

// vision_config of the checkpoint (config.json; Inkling2VisionConfig defaults where it is silent).
constexpr int kHidden = 1024;
constexpr int kHeads = 16;
constexpr int kHeadDim = kHidden / kHeads;  // 64
constexpr int kInter = 4096;                // hidden_size * mlp_ratio 4.0
constexpr int kPatch = 20;                  // native_patch_size
constexpr int kPatchRow = 3 * kPatch * kPatch;  // 1,200: one patch, (y, x, channel)
constexpr int kMerge = 2;                   // adapter_merge_size: one image token = 2 x 2 patches
constexpr int kMergeRow = kMerge * kMerge * kHidden;  // 4,096
constexpr int kProjHidden = 4096;           // projection_hidden_dim
constexpr int kOutHidden = 2304;            // text_hidden_size
constexpr int kPosSide = 32;                // posemb_grid_size: a 32 x 32 learned table
constexpr int kLayers = 23;
constexpr float kEps = 1e-5f;               // layer_norm_eps
constexpr float kRopeTheta = 10000.0f;      // rope_parameters {rope_type axial, rope_theta 1e4}
constexpr int kRopeFreqs = kHeadDim / 4;    // 16 frequencies per axis
constexpr int kMaxImages = 32;              // per call (the tokenizer sidecar's MAX_IMAGES)

// One image of a call. Its sequence segment is rows [seq_start, seq_start + has_class + h * w) of the encoder's
// [S, 1,024] state: the class token first (when has_class), then its h * w patches in merge-block order (2 x 2 blocks
// row-major, the patches of a block row-major: the processor's order). patch_start: its first patch row in the patch
// embedding output. h, w: the grid in patches (even, >= 2).
struct Image {
    int32_t seq_start;
    int32_t patch_start;
    int32_t h;
    int32_t w;
    int32_t has_class;
};

// The images of one call, passed by value (no upload).
struct Images {
    int32_t count;
    int32_t max_seq;  // the longest segment
    Image image[kMaxImages];
};

enum class Act : int { kNone = 0, kGeluErf = 1 };

// y[r, c] = bf16(act(h)), h = bf16(acc[r, c] + bias[c]) (bias null: h = bf16(acc)); kGeluErf: nn.GELU() (erf, FP32
// on the BF16 h). acc F32 [rows, n] contiguous; n % 8 == 0; y row stride ldy.
cudaError_t bias_act(const float *acc, const bf16 *bias, int64_t rows, int n, Act act, bf16 *y, int64_t ldy,
                     cudaStream_t stream);

// The layer-scaled residual of Inkling2VisionEncoderLayer: x[r, c] = bf16(x[r, c] + bf16(gamma[c] * h)),
// h = bf16(acc[r, c] + bias[c]). acc F32 [rows, kHidden] contiguous, x [rows, kHidden] contiguous.
cudaError_t scaled_residual(const float *acc, const bf16 *bias, const bf16 *gamma, int64_t rows, bf16 *x,
                            cudaStream_t stream);

// y = bf16(LayerNorm(x) * weight + bias), FP32 mean and biased variance (nn.LayerNorm on BF16). dim % 8 == 0,
// dim <= 8192, strides % 8 == 0 and >= dim.
cudaError_t layernorm(const bf16 *x, int64_t ldx, const bf16 *weight, const bf16 *bias, float eps, int64_t rows,
                      int dim, bf16 *y, int64_t ldy, cudaStream_t stream);

// Inkling2VisionEmbeddings.forward: per image, the class row = bf16(class_embedding + cls_position_embedding) (when
// has_class) and patch row p = bf16(bf16(acc[patch_start + p]) + bf16(pos(p))), pos(p) = FP32 sum over the 4
// bilinear taps (align_corners false, border clamp; get_vision_interpolation_indices_and_weights) of
// table[tap] * weight (FP32 products, taps (h0, w0), (h0, w1), (h1, w0), (h1, w1)). acc F32 [P, kHidden] (the patch
// embedding, no bias); table [kPosSide^2, kHidden]; seq [S, kHidden] contiguous.
cudaError_t embed(const float *acc, const bf16 *table, const bf16 *class_embedding, const bf16 *cls_position,
                  const Images &images, bf16 *seq, cudaStream_t stream);

// The attention inputs: q, k, v [S, kHidden] = bf16(acc_{q,k,v} + bias) (acc F32 [S, kHidden] each, contiguous), then
// q and k rotated in place by the axial 2D RoPE (Inkling2VisionRotaryEmbedding + apply_rotary_pos_emb_vision): a
// class row sits at (0, 0), patch (row, col) at (row + 1, col + 1); per head, interleaved channel pair j (2j, 2j + 1)
// turns by angle w_pos * inv_freq[j] for j < 16 and h_pos * inv_freq[j - 16] for 16 <= j < 32, inv_freq[f] =
// 1 / theta^(2f / 32); out = x * cos + rotate_pairwise(x) * sin in FP32 on the BF16 x, one rounding.
// swap_axes (fail-first mutant only): the h and w positions trade places.
cudaError_t qkv_rope(const float *acc_q, const float *acc_k, const float *acc_v, const bf16 *bias_q,
                     const bf16 *bias_k, const bf16 *bias_v, const Images &images, bool swap_axes, bf16 *q, bf16 *k,
                     bf16 *v, cudaStream_t stream);

// Bidirectional attention within each image's segment (cu_seqlens), per head softmax(q k^T / 8) v with FP32 scores,
// softmax and accumulation; q, k, v, out [S, kHidden] contiguous.
cudaError_t attention(const bf16 *q, const bf16 *k, const bf16 *v, const Images &images, bf16 *out,
                      cudaStream_t stream);

}  // namespace cere::inkling::vision
