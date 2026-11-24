// Inkling 2 Mini S5: the logit epilogue around the unembed GEMM (forward.h): widen the BF16 GEMM output to FP32
// (REF :1534 `.float()`, exact) before S3's softcap, the greedy argmax of softcapped rows, and the sampler's first
// stage on those FP32 rows (kernels/k12_sampler.h runs the rest). The argmax and the top two are exact under a strict
// total order (below), and the sampler's first stage is elementwise, so a row's result depends only on that row,
// whatever blocks share the scan (G4). Enqueue on `stream`, never synchronize; cudaErrorInvalidValue for a malformed
// argument (nothing launched).
#pragma once

#include <cstdint>

#include "cere_executor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cere::inkling {

// out[r, c] = float(in[r, c]) for r < rows, c < vocab; in at row stride ldi, out at ldo.
cudaError_t widen_logits(const __nv_bfloat16 *in, int64_t ldi, int64_t rows, int64_t vocab, float *out, int64_t ldo,
                         cudaStream_t stream);
// ids[r] = argmax over c < vocab of logits[r, c] (row stride ld): NaN is the maximum, ties go to the lowest id.
// ids is device memory; rows <= 65535.
cudaError_t argmax_rows(const float *logits, int64_t rows, int64_t vocab, int64_t ld, int32_t *ids,
                        cudaStream_t stream);
// The greedy score of every row (the device's step and step_rows, host/device.h): out[r].id = the argmax as
// argmax_rows, runner_up the best other id by the same order, logit / runner_up_logit their values. out is device
// memory; vocab >= 2, rows <= 65535.
cudaError_t top2_rows(const float *logits, int64_t rows, int64_t vocab, int64_t ld, cere_token_score *out,
                      cudaStream_t stream);
// k12::process (kernels/k12_sampler.h, D2) on one softcapped FP32 row instead of BF16 logits, without a vocab mask:
// out[i] = logits[i]; for counts[i] > 0, x - (f * c + p) then x < 0 ? x * r : x / r (fp32, no contraction); then
// out[bias[j].id] += bias[j].bias for j < bias_count (device array; ids unique and < vocab, the caller checked).
// counts null = no output tokens yet.
cudaError_t process_logits(const float *logits, int64_t vocab, const uint32_t *counts, float frequency_penalty,
                           float presence_penalty, float repetition_penalty, const cere_logit_bias *bias,
                           int64_t bias_count, float *out, cudaStream_t stream);

}  // namespace cere::inkling
