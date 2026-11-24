// Inkling 2 Mini (inkling design note 3.1-3.2, G1, slice S3): the depth mixes (block attention residual and message
// attention residual) and the residual-stream bookkeeping. Bit-equal to executor/ref/inkling/depth_ref.h, which also
// documents which buffer holds what across a visit, the loop passes and the final mix, and the checkpoint query
// mapping. Forward-graph contract for S5:
// - Every function enqueues on `stream`, never synchronizes, needs no workspace and returns the launch status
//   (cudaErrorInvalidValue for a malformed argument, nothing launched). rows in [1, 2^31 - 1].
// - Determinism: one 256-thread block per token row (depth_mix) or one thread per element (the helpers); a row's
//   output depends only on that row's inputs (not on `rows`, other rows or the launch split), two runs bit-identical.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cere::inkling {

using bf16 = __nv_bfloat16;

constexpr int kDepthMaxStates = 5;   // 4 block states + the partial
constexpr int kDepthMaxQueries = 4;  // message mix: q, k, v, mlp

// One state of the stack: rows of 2304 elements at row stride ld, BF16 (f32 = 0: the block states h_0, h_6, h_12,
// h_18, h_L, the previous pass's h_12 / h_18) or FP32 (f32 = 1: the post-attention residual r of the MLP input).
struct DepthState {
    const void *ptr;
    int64_t ld;
    int f32;
};
// states[0 .. count) in stack order: the block states oldest first, the partial last. count in [1, 5].
struct DepthStates {
    DepthState s[kDepthMaxStates];
    int count;
};

// out[q][r, :] = depth_mix(stack row r, query q) for r < rows, q < nq (nq in [1, 4]): queries FP32 [nq, 2304]
// contiguous (attention_input 1 query, 3 on full layers = the q, k, v inputs; mlp_input 1; final_input 1; the
// message mix 4 = q, k, v, mlp messages); out FP32, query q's row r at out + q * out_qstride + r * ldo (out_qstride
// = 0 is allowed only for nq = 1). The message mix is this call with states [previous pass's h_12, previous pass's
// h_18] and message_input.query[i].
cudaError_t depth_mix(const DepthStates &states, int64_t rows, const float *queries, int nq, float *out, int64_t ldo,
                      int64_t out_qstride, cudaStream_t stream);

// The weighted RMSNorm after a depth mix, fused: y[q] = bf16(RMSNorm_w(out[q])) for q < nq, w[q] BF16 [2304], y[q]
// BF16 rows at stride ldy; bit-equal to depth_mix then norm.h's rms_norm(out[q], x_f32 = true, w[q], eps, y[q],
// y_f32 = false) per query, without the FP32 out (the attention inputs of a visit without messages, its MLP input,
// the final norm). y[q] may not overlap a state.
struct DepthNorm {
    const bf16 *w[kDepthMaxQueries];
    bf16 *y[kDepthMaxQueries];
    int64_t ldy;
    float eps;
};
cudaError_t depth_mix_norm(const DepthStates &states, int64_t rows, const float *queries, int nq,
                           const DepthNorm &norm, cudaStream_t stream);

// The post-attention residual (REF :901-905): r[t, i] = boundary ? float(o[t, i]) : float(h[t, i]) + float(o[t, i]),
// FP32; h = the visit's input h_L BF16, o = the post-attention conv output BF16. At a boundary layer (L % 6 == 0)
// h_L joins the block-state list (the caller keeps the pointer) and the residual restarts from o; h may then be
// nullptr (not read).
cudaError_t attn_residual(bool boundary, const bf16 *h, int64_t ldh, const bf16 *o, int64_t ldo, int64_t rows,
                          float *r, int64_t ldr, cudaStream_t stream);

// The layer output (REF :910): h_next[t, i] = bf16(r[t, i] + float(d[t, i])), d = the post-MLP conv output BF16.
// h_next may not alias r; it may alias d (same row stride).
cudaError_t layer_out(const float *r, int64_t ldr, const bf16 *d, int64_t ldd, int64_t rows, bf16 *h_next,
                      int64_t ldh, cudaStream_t stream);

}  // namespace cere::inkling
