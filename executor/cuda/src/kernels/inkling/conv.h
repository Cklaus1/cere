// Inkling 2 Mini (inkling design note 3.2, G1, slice S3): the short causal depthwise convolutions (kernel 4), static
// and dynamic, with their FP32 state. Bit-equal to executor/ref/inkling/conv_ref.h (exact order there). Forward-graph
// contract for S5:
// - Four convs per visit per session, each with its own state: K and V (C = 1024; static on sliding layers, dynamic
//   on the full layers 5, 11, 17, 23), post-attention and post-MLP (C = 2304, static). Loop layers keep one state
//   set per pass (the REF cache_offset copies): 48 visits x 4 states per session.
// - State: FP32 [3, C] contiguous (state[j * C + c], row 0 the oldest), zero for a fresh sequence; 3 x C x 4 bytes.
// - x: the BF16 linear / o_proj / MoE output [n, C] at row stride ldx; y: BF16 [n, C] at row stride ldy; y must not
//   alias x. Static weights w: BF16 [C, 4] = the checkpoint's `[C, 1, 4]` (`*_sconv.weight`, tap j multiplies
//   x[t - 3 + j]). Dynamic: static_w BF16 [4, C] (`*_sconv.static_w`, row = lag), dyn BF16 [n, (C / 128) * 4] at row
//   stride ld_dyn = the `*_sconv.weight_proj` output (a [n, 2304] x [2304, 32] BF16 GEMM of the k/v projection's
//   input, which S5 runs: dyn[t, h * 4 + lag] is the per-token tap of KV head h = c / 128 at lag `lag`).
// - Prefill (conv_*_prefill): one sequence of n rows (a prefill chunk, or a spec-verify run of one session),
//   n in [1, 2,097,120] (65,535 tiles of 32 rows), state in/out (two launches: the outputs, then the state
//   advance). Splitting a sequence at any row with the state carried gives the same outputs and final state bit for
//   bit (G4 / D9).
// - Decode (conv_*_decode): M independent rows, row r from its own session (states[r] -> that session's [3, C]
//   state, a device array of M pointers; all distinct), M in [1, 65,535]; each row is the n = 1 prefill of its
//   state. A row's output depends only on its row and its state.
// - Every function enqueues on `stream`, never synchronizes, needs no workspace and returns the launch status
//   (cudaErrorInvalidValue for a malformed argument, nothing launched). Two runs are bit-identical.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cere::inkling {

using bf16 = __nv_bfloat16;

// C >= 1.
cudaError_t conv_static_prefill(const bf16 *x, int64_t ldx, int64_t n, int C, const bf16 *w, float *state, bf16 *y,
                                int64_t ldy, cudaStream_t stream);
cudaError_t conv_static_decode(const bf16 *x, int64_t ldx, int64_t M, int C, const bf16 *w, float *const *states,
                               bf16 *y, int64_t ldy, cudaStream_t stream);

// C a multiple of 128 (8 KV heads x 128 = 1024 in the model).
cudaError_t conv_dynamic_prefill(const bf16 *x, int64_t ldx, int64_t n, int C, const bf16 *static_w, const bf16 *dyn,
                                 int64_t ld_dyn, float *state, bf16 *y, int64_t ldy, cudaStream_t stream);
cudaError_t conv_dynamic_decode(const bf16 *x, int64_t ldx, int64_t M, int C, const bf16 *static_w, const bf16 *dyn,
                                int64_t ld_dyn, float *const *states, bf16 *y, int64_t ldy, cudaStream_t stream);

}  // namespace cere::inkling
