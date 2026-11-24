// Inkling 2 Mini (inkling design note 3.2, G1, slice S3): RMSNorm over the 2304-wide hidden row, weighted and
// unweighted, and the normed embedding. Bit-equal to executor/ref/inkling/norm_ref.h and embed_ref.h (the exact
// operation order is documented there). Forward-graph contract for S5:
// - Every function enqueues on `stream`, never synchronizes, allocates nothing and needs no workspace; it returns
//   the launch status (cudaErrorInvalidValue for a malformed argument, nothing launched).
// - Rows are row-major, 2304 elements each, at the given row strides (elements). rows in [1, 2^31 - 1] (prefill
//   chunks up to 4,096 rows, decode M 1..64 and anything else alike).
// - Determinism: one 256-thread block per row, fixed reduction order; a row's output depends only on that row (not on
//   `rows`, the other rows or how a batch is split into launches: G4 / D9), and two runs are bit-identical.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cere::inkling {

using bf16 = __nv_bfloat16;

// y[r, :] = RMSNorm(x[r, :]) for r < rows: x BF16 (x_f32 = false) or FP32 (x_f32 = true) at row stride ldx; w BF16
// [2304] (weighted: input_layernorm, q/k/v_input_layernorm, post_attention_layernorm, the final norm) or nullptr
// (unweighted); eps = config rms_norm_eps (1e-6); y BF16 (y_f32 = false, one RNE rounding) or FP32 at row stride ldy.
// y may alias x only with x_f32 == y_f32 and ldx == ldy (each thread reads its elements before writing them).
cudaError_t rms_norm(const void *x, bool x_f32, int64_t ldx, const bf16 *w, float eps, int64_t rows, void *y,
                     bool y_f32, int64_t ldy, cudaStream_t stream);

// h_0 = the normed embedding: out[r, :] = bf16(RMSNorm_w(float(table[ids[r], :]))) for r < rows; table BF16
// [vocab, 2304] (model.llm.embed.weight), ids uint32 [rows] (the caller validates ids < vocab), w BF16 [2304]
// (model.llm.embed_norm.weight), out BF16 at row stride ldo.
cudaError_t embed_norm(const bf16 *table, const uint32_t *ids, int64_t rows, const bf16 *w, float eps, bf16 *out,
                       int64_t ldo, cudaStream_t stream);

}  // namespace cere::inkling
