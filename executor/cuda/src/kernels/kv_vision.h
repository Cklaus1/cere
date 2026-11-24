// KV (m213 design note D10, C9): the vision encoder kernels of the A4 model's tower (SGLang
// `Qwen3VLMoeVisionModel`, sglang/srt/models/qwen3_vl.py) and the splice of its output into the prompt rows (D11).
// The encoder's linears are K2 GEMMs with F32 output followed by `bias_act` (KV6), so every linear rounds to BF16
// once as an addmm does: patch embed 1536->1152 (KV1, Conv3d with stride = kernel over the [P, 3, 2, 16, 16] view),
// qkv 1152->3456, proj 1152->1152, fc1 1152->4304, fc2 4304->1152, merger fc1 4608->4608, merger fc2 4608->2560.
// All functions enqueue on `stream`, never synchronize, and return the launch status (cudaErrorInvalidValue before
// any launch on a bad argument). Tensors are row-major BF16 unless noted; strides in elements.
//
// math_reference: sglang
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cere_model_constants.h"

namespace cere::kv {

using bf16 = __nv_bfloat16;

// The vision tower shape this file is compiled for (A4 vision_config; host/support.cpp make_a4_vision_config).
constexpr int kHidden = gen::kVisionHiddenSize;
constexpr int kHeads = gen::kVisionNumHeads;
constexpr int kHeadDim = kHidden / kHeads;  // 72
static_assert(kHeadDim * kHeads == kHidden, "vision heads split hidden_size evenly");
constexpr int kQkvWidth = 3 * kHidden;
constexpr int kRotaryHalf = kHeadDim / 4;  // 18 frequencies per axis (rotary_dim 36 = h and w halves)
constexpr int kMerge = gen::kVisionSpatialMergeSize;  // spatial_merge_size
constexpr float kRopeTheta = 10000.0f;

enum class Act : int { kBias = 0, kGeluTanh = 1, kGeluErf = 2 };

// KV6: h = bf16(acc[r, c] + bias[c]); kBias: y = h, or bf16(h + residual[r, c]) when residual is non-null;
// kGeluTanh / kGeluErf: y = bf16(gelu(h)) in FP32 (nn.GELU(approximate="tanh") / nn.GELU() on the addmm output).
// acc: F32 [rows, n] contiguous; y [rows, n] with row stride y_stride; residual [rows, n] with stride y_stride and
// may alias y. A residual with a GELU mode is invalid. n % 8 == 0.
cudaError_t bias_act(const float* acc, const bf16* bias, const bf16* residual, int64_t rows, int n, Act act, bf16* y,
                     int64_t y_stride, cudaStream_t stream);

// KV3: y = bf16(LayerNorm(x) * weight + bias) with FP32 mean and biased variance (nn.LayerNorm). dim % 8 == 0,
// dim <= 8192, strides % 8 == 0 and >= dim (rows do not overlap).
cudaError_t layernorm(const bf16* x, int64_t x_stride, const bf16* weight, const bf16* bias, float eps, int64_t rows,
                      int dim, bf16* y, int64_t y_stride, cudaStream_t stream);

// KV2: the bilinearly interpolated position embedding of one image of grid (t, h, w) patches (SGLang
// `fast_pos_embed_interpolate_from_list`, qwen3_vl.py:547-606), rows in merge-window order and repeated over t:
// table [side * side, kHidden] (side = 48); align_corners: linspace(0, side - 1, n) indices (true; what from_list
// uses) or the half-pixel form of `_torch_interp_indices` (false). Corner products are rounded to BF16 (BF16
// weights times BF16 rows), the four-term sum is FP32 then rounded. x null: out = pos; else out = bf16(x + pos) (x
// may alias out). h, w even and non-zero.
cudaError_t pos_embed(const bf16* table, int side, int64_t grid_t, int64_t grid_h, int64_t grid_w, bool align_corners,
                      const bf16* x, bf16* out, cudaStream_t stream);

// The 2D vision RoPE table: cos_sin [n_pos, 36] F32 = K1's rope_table with rotary_dim 36, theta 1e4 (SGLang's
// get_rope(72, 36, 8192, 1e4) cache: per row 18 cos then 18 sin).
cudaError_t rope_table(float* cos_sin, int64_t n_pos, cudaStream_t stream);

// KV4: q_out / k_out [P, kHidden] = 2D RoPE of the q and k parts of qkv [P, qkv_stride] (q at column 0, k at
// kHidden) for one image of grid (t, h, w): row r has (h_id, w_id) in merge-window order (RotaryPosMixin.rot_pos_ids,
// repeated t times); per head, angle[j] = h_id * inv_freq[j] for j < 18 and w_id * inv_freq[j - 18] for 18 <= j <
// 36, NeoX rotate-half over the full 72 dims (pairs (i, i + 36), angle[i % 36]); FP32 math, one rounding.
// cos_sin: rope_table with n_pos rows; max(h, w) <= n_pos is required (the rows read are the grid's h and w ids).
cudaError_t rope2d(const bf16* qkv, int64_t qkv_stride, const float* cos_sin, int64_t n_pos, int64_t grid_t,
                   int64_t grid_h, int64_t grid_w, bf16* q_out, bf16* k_out, cudaStream_t stream);

// KV5: out [P, kHidden] = per head h, softmax(q_h k_h^T * 72^-0.5) v_h, bidirectional within each segment of
// seq_len consecutive rows (one frame of an image, SGLang's cu_seqlens); P % seq_len == 0. q, k row stride
// qk_stride, v row stride v_stride, out contiguous. FP32 scores, softmax and accumulation; BF16 output.
cudaError_t attention(const bf16* q, const bf16* k, int64_t qk_stride, const bf16* v, int64_t v_stride, int64_t rows,
                      int64_t seq_len, bf16* out, cudaStream_t stream);

// KV7: copies the rows r of embeds [image_rows, hidden] with token_offset + r in [chunk_start, chunk_start + n) over
// mixed[token_offset + r - chunk_start] (mixed [n, hidden] contiguous); bit-exact, other rows untouched.
cudaError_t splice(const bf16* embeds, int64_t image_rows, int64_t token_offset, int64_t chunk_start, int64_t n,
                   int64_t hidden, bf16* mixed, cudaStream_t stream);

}  // namespace cere::kv
