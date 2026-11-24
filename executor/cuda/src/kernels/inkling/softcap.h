// Inkling 2 Mini (inkling design note 3.1, G1, slice S3): the final logit softcap y = 30 tanh(x / 30) in place on the
// FP32 logits (lm_head output widened to FP32, REF :1535). Bit-equal to executor/ref/inkling/softcap_ref.h. Contract
// for S5: logits FP32 [rows, vocab] at row stride ld (elements, >= vocab), rows in [1, 65,535], vocab in
// [1, 2^31 - 1]; elementwise, so any batch split gives the same bits; enqueues on `stream`, never synchronizes, no
// workspace, returns the launch status (cudaErrorInvalidValue for a malformed argument, nothing launched). A NaN logit
// stays NaN (its payload is not part of the contract: CUDA arithmetic returns the canonical NaN).
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace cere::inkling {

cudaError_t softcap(float *logits, int64_t rows, int64_t vocab, int64_t ld, cudaStream_t stream);

}  // namespace cere::inkling
