// SelfTune ST4 S3c: the fail-first mutants of family sdar's kernels, for the kernel tests only (ktest/sdar); the
// executor never calls them. Contracts as sdar_kernels.h, except where stated.
//
// math_reference: DICE-8B
#pragma once

#include "kernels/sdar/sdar_kernels.h"

namespace cere::sdar::mutant {

// top_tie comparing the FP32 logits as given (no BF16 rounding; the reported logits are the FP32 values): two logits
// that differ in FP32 but round to one BF16 value are not a tie here (the condition-1 mutant).
cudaError_t top_tie_fp32(const float *logits, int64_t stride, int64_t rows, int64_t vocab, TopTie *out,
                         cudaStream_t stream);

}  // namespace cere::sdar::mutant
