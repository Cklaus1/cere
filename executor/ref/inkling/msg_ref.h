#pragma once
// Inkling 2 Mini (inkling design note 3.1, G1): the loop message gate, CPU reference (kernel:
// executor/cuda/src/kernels/inkling/msg.h). REF Inkling2LoopMessageGate (modeling_inkling2_loop.py:79-96), used by the
// loop layers in passes 2 and 3 (q, k, v gates on the attention input, the mlp gate on the MLP input, REF :961-992 and
// :1115-1144). Checkpoint tensors (controllers.{attn,k,v,mlp}): down_weight BF16 [64, 4608] (down_proj, a BF16
// linear), forget_weight / gain_weight BF16 [2304, 64] (forget_gate_proj / message_gate_proj: FP32 linears, the BF16
// values widened exactly).
//
// Exact order, one token row: h, msg FP32 [2304] (the depth / message mix outputs):
//   1. nh = unweighted RMSNorm(h), nm = unweighted RMSNorm(msg) (norm_ref.h, L = 256, eps kRmsEps)
//   2. g[i] = float(encode_bf16(nh[i])), g[2304 + i] = float(encode_bf16(nm[i]))          (the concat cast to BF16)
//   3. d[r] = float(encode_bf16(reduce_dot<256>(down[r, 0..4608), g, 4608))), r < 64       lane t: i = t, t+256, ..,
//      t+4352 (18 terms, the h half then the msg half), acc = fma(down[r, i], g[i], acc); the halving tree
//   4. per channel c: f = tanh(reduce_dot<64>(forget[c, 0..64), d, 64)), m = tanh(reduce_dot<64>(gain[c, 0..64), d,
//      64)) (lane t = the one term fma(w[t], d[t], 0), then the tree over 64 lanes)
//   5. out[c] = add(mul(sub(1, f), h[c]), mul(m, msg[c]))                                  FP32 output
#include "norm_ref.h"

namespace ref {

constexpr int kGateRank = 64;
constexpr int kGateIn = 2 * kHidden;  // 4608

// out[0..2304) for one row; out may alias h or msg.
inline void msg_gate_row(const float *h, const float *msg, const uint16_t *down, const uint16_t *forget,
                         const uint16_t *gain, float eps, float *out) {
    float g[kGateIn], w[kGateIn], d[kGateRank], wr[kGateRank];
    rms_norm<kReduceHidden>(h, kHidden, eps, nullptr, g);
    rms_norm<kReduceHidden>(msg, kHidden, eps, nullptr, g + kHidden);
    for (int i = 0; i < kGateIn; ++i) g[i] = decode_bf16(encode_bf16(g[i]));
    for (int r = 0; r < kGateRank; ++r) {
        for (int i = 0; i < kGateIn; ++i) w[i] = decode_bf16(down[int64_t(r) * kGateIn + i]);
        d[r] = decode_bf16(encode_bf16(reduce_dot<kReduceHidden>(w, g, kGateIn)));
    }
    for (int c = 0; c < kHidden; ++c) {
        for (int j = 0; j < kGateRank; ++j) wr[j] = decode_bf16(forget[int64_t(c) * kGateRank + j]);
        const float f = ref::tanh(reduce_dot<kReduceRank>(wr, d, kGateRank));
        for (int j = 0; j < kGateRank; ++j) wr[j] = decode_bf16(gain[int64_t(c) * kGateRank + j]);
        const float m = ref::tanh(reduce_dot<kReduceRank>(wr, d, kGateRank));
        out[c] = add(mul(sub(1.0f, f), h[c]), mul(m, msg[c]));
    }
}

}  // namespace ref
