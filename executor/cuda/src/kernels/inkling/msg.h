// Inkling 2 Mini (inkling design note 3.1, G1, slice S3): the loop message gate in two kernels (the two norms and the
// BF16 rank-64 down projection into a workspace; then the FP32 forget / message gate projections, tanh and the mix;
// no cuBLAS). Bit-equal to executor/ref/inkling/msg_ref.h (exact order there). Forward-graph contract for S5:
// - Loop layers 6..17 in passes 2 and 3: q, k, v gates on the attention input (each followed by input_layernorm),
//   the mlp gate on the MLP input (followed by post_attention_layernorm). Weights per gate (controllers.{attn,k,v,mlp}
//   of the layer; `attn` is the q gate): down BF16 [64, 4608] (down_weight), forget BF16 [2304, 64] (forget_weight),
//   gain BF16 [2304, 64] (gain_weight, the message gate); forget and gain 16-byte aligned.
// - h: FP32 [rows, 2304] at row stride ldh (the depth mix output: attention_input / mlp_input); msg: FP32 at row stride
//   ldm (the message mix output for this gate); out FP32 at row stride ldo. out may alias h (ldo == ldh) or msg
//   (ldo == ldm): the second kernel reads each element before the same thread writes it. ws: FP32 [rows, 64]
//   workspace (the down projection), aliasing none of h, msg, out.
// - Rows in [1, 2^31 - 1]; the grid shape depends on rows (decode-sized calls spread a row over more blocks), every
//   element's arithmetic does not; a row's output depends only on that row; two runs bit-identical. Enqueues on
//   `stream`, never synchronizes, returns the launch status (cudaErrorInvalidValue for a malformed argument, nothing
//   launched). eps = config rms_norm_eps (1e-6).
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cere::inkling {

using bf16 = __nv_bfloat16;

cudaError_t msg_gate(const float *h, int64_t ldh, const float *msg, int64_t ldm, int64_t rows, const bf16 *down,
                     const bf16 *forget, const bf16 *gain, float eps, float *out, int64_t ldo, float *ws,
                     cudaStream_t stream);

// The workspace msg_gate needs for `rows` rows.
constexpr uint64_t msg_gate_workspace_floats(int64_t rows) { return uint64_t(rows) * 64; }

// The forward's gate-then-norm in two launches for up to kMsgMaxGates gates at once (S8): per gate i,
// y_i = bf16(rms_norm(msg_gate(h_i, msg_i), norm_w_i)) (norm_ref.h weighted, eps for both the gate's norms and this
// one), bit-equal to msg_gate followed by rms_norm (norm.h) to a BF16 output, with no FP32 gate output. The q, k, v
// gates of a visit run in one call (h may be shared: nothing writes it). h, msg at row strides ldh / ldm, y BF16 at
// ldy (aliasing no h or msg); norm_w BF16 [2304]; ws FP32 [msg_gate_norm_workspace_floats(rows)] aliasing no operand.
// Rows in [1, 2^28 - 1] (2^31 / 8); the rest as msg_gate.
constexpr int kMsgMaxGates = 3;

struct MsgGateNorm {
    const float *h = nullptr;
    const float *msg = nullptr;
    const bf16 *down = nullptr, *forget = nullptr, *gain = nullptr;
    const bf16 *norm_w = nullptr;
    bf16 *y = nullptr;
};

cudaError_t msg_gate_norm(const MsgGateNorm *gates, int count, int64_t ldh, int64_t ldm, int64_t ldy, int64_t rows,
                          float eps, float *ws, cudaStream_t stream);

constexpr uint64_t msg_gate_norm_workspace_floats(int64_t rows) { return uint64_t(rows) * 64 * kMsgMaxGates; }

}  // namespace cere::inkling
