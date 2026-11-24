// K5 decode (m22n design note D5): the single-token gated delta rule of a Qwen3.8-Flash-Next linear-attention
// layer, with the arithmetic of the kernel the no-speculation SGLang control dispatches for decode: FlashInfer's
// BF16-state MTP-ILP4 GDN decode kernel (T=1).
//
// math_reference: flashinfer
// math_reference: sglang
//
// Written from the math of flashinfer/gdn_kernels/gdn_decode_bf16_state.py (gdn_decode_bf16state_mtp_ilp4_kernel,
// reached from gated_delta_rule_decode_pretranspose in pool mode, as SGLang's gdn_flashinfer.py decode calls it) and
// the PTX the image's CuTe DSL generates for it. Nothing is vendored.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace cere::k5 {

// The D5 lowering variant kept: (1) ex2/lg2/rsqrt.approx.ftz, the reciprocal of beta correctly rounded, FMA
// contraction of the L2 sums on (the image's PTX: rcp.rn.f32 for 1/(1+exp(-b)), fma.rn.f32.bf16 for sum += x*x).
constexpr int kDecodeVariant = 1;

// One token (batch 1), 16 key heads x 128, 48 value heads x 128; value head h uses key head h / 3.
//   q, k: bf16 [16, 128] raw (L2-normalised in-kernel: x * rsqrt(sum x^2 + 1e-6), scale folded into q);
//   v: bf16 [48, 128]; a, b: bf16 [48]; A_log: f32 [48]; dt_bias: bf16 [48];
//   state: bf16 [48, 128 (v), 128 (k)] in/out (K-last, SGLang's pool slot), FP32 in registers, BF16 at rest;
//   o: bf16 [48, 128].
// Per value head, with S = state[h] as [v, k]: decay = exp(-exp(A_log) * softplus(a + dt_bias)) (softplus(x) = x
// for x > 20), beta = 1 / (1 + exp(-b)); S *= decay; u = beta * (v - S k); S += u k^T; o = S q.
// All buffers contiguous; o must not alias the inputs.
struct DecodeArgs {
    const __nv_bfloat16 *q;
    const __nv_bfloat16 *k;
    const __nv_bfloat16 *v;
    const __nv_bfloat16 *a;
    const __nv_bfloat16 *b;
    const float *A_log;
    const __nv_bfloat16 *dt_bias;
    __nv_bfloat16 *state;
    __nv_bfloat16 *o;
    float scale;
};

// Enqueues the step on `stream`; cudaErrorInvalidValue for null or misaligned pointers, else the launch status.
cudaError_t delta_rule_decode(const DecodeArgs &args, cudaStream_t stream);

// The most rows (all chains) of one delta_rule_decode_rows launch: a verify step's rows (model forward_verify_rows).
constexpr int kDecodeMaxRows = 16;

// The rows of a verify step's speculation chains (m23 design note DG-02, the verify chains of a GDN layer; one
// launch for all chains, rows16 design note R16-B): chain c is rows first[c] .. first[c] + rows[c] - 1 on its own
// state states[c] (step.state is not read); row r reads q, k, v at step.q / step.k / step.v + r * qkv_stride, a and
// b at step.a / step.b + r * ab_stride, writes o at step.o + r * o_stride. Per chain bit-equal to rows[c]
// delta_rule_decode launches in row order on its state: each row runs the one-token kernel's arithmetic on the state
// as it rests between launches (BF16); the state after row r is stored to saves[r] (the undo snapshot right before
// row r + 1) unless r is its chain's last row, which stores it to states[c].
struct DecodeRowsArgs {
    DecodeArgs step;
    int chains;                           // 1 .. kDecodeMaxRows
    int first[kDecodeMaxRows];            // back to back: first[0] = 0, first[c + 1] = first[c] + rows[c]
    int rows[kDecodeMaxRows];             // >= 1 each, sum <= kDecodeMaxRows
    __nv_bfloat16 *states[kDecodeMaxRows];
    __nv_bfloat16 *saves[kDecodeMaxRows];
    int64_t qkv_stride;
    int64_t ab_stride;
    int64_t o_stride;
};

// Enqueues the chains on `stream`; cudaErrorInvalidValue for chains that are not back to back from row 0 or exceed
// kDecodeMaxRows rows, strides that break the 8-byte alignment or overlap a row's own row, a null or misaligned
// pointer (states[c], saves[r] for rows that are not their chain's last), else the launch status.
cudaError_t delta_rule_decode_rows(const DecodeRowsArgs &args, cudaStream_t stream);

}  // namespace cere::k5
