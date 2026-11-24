// Inkling 2 Mini (inkling design note G1, slice S3): the fail-first mutants of the residual-stream kernels, reached
// only by the kernel tests (executor/cuda/src/ktest/inkling/mix_ktest.cu): each is its kernel with one deliberate
// defect, and the suite (executor/tests/gpu/test_inkling_kernels_mix.py) shows the G1 comparison refuses it.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "kernels/inkling/depth.h"

namespace cere::inkling::mutant {

// The convs without the residual and the scale: y = bf16(acc).
cudaError_t conv_static_prefill_no_residual(const bf16 *x, int64_t ldx, int64_t n, int C, const bf16 *w,
                                            float *state, bf16 *y, int64_t ldy, cudaStream_t stream);
cudaError_t conv_dynamic_prefill_no_residual(const bf16 *x, int64_t ldx, int64_t n, int C, const bf16 *static_w,
                                             const bf16 *dyn, int64_t ld_dyn, float *state, bf16 *y, int64_t ldy,
                                             cudaStream_t stream);

// The depth mix with a weighted key norm (k_s = w * RMSNorm(state s), w BF16 [2304]) instead of the unweighted one.
cudaError_t depth_mix_weighted_norm(const DepthStates &states, int64_t rows, const float *queries, int nq,
                                    float *out, int64_t ldo, int64_t out_qstride, const bf16 *w,
                                    cudaStream_t stream);

// The message gate with the forget and message gates swapped: (1 - tanh m) * h + tanh f * msg.
cudaError_t msg_gate_swapped(const float *h, int64_t ldh, const float *msg, int64_t ldm, int64_t rows,
                             const bf16 *down, const bf16 *forget, const bf16 *gain, float eps, float *out,
                             int64_t ldo, float *ws, cudaStream_t stream);

}  // namespace cere::inkling::mutant
