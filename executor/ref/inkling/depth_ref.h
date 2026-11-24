#pragma once
// Inkling 2 Mini (inkling design note 3.1-3.2, G1): the depth mixes (block attention residual and message attention
// residual) and the residual-stream helpers, CPU reference (kernels: executor/cuda/src/kernels/inkling/depth.h).
// REF modeling_inkling2_loop.py: Inkling2LoopBlockAttentionResidual :828-853 (attention_input: 1 query, 3 on full
// layers; mlp_input: 1 query; final_input :1442: 1 query), Inkling2LoopMessageAttentionResidual :145-173 (4 queries
// per loop layer: q, k, v, mlp), eager_attention_forward :111-142 (no mask, no sink), the decoder layers :856-1149 and
// the model loop :1359-1443. All FP32 (_keep_in_fp32_modules_strict).
//
// Depth mix, one token row: S states (S <= kMaxDepthStates; FP32 values: BF16 block states widened exactly, the
// partial FP32 or BF16 widened) and nq FP32 queries [nq, 2304]:
//   1. k_s = unweighted RMSNorm of state s (norm_ref.h: reduce_dot<256>, div by 2304, eps kRmsEps), FP32.
//   2. score[q][s] = mul(reduce_dot<256>(query_q, k_s, 2304), kDepthScale)    lane t: acc = fma(query[i], k[i], acc)
//      kDepthScale = fp32(2304^-0.5) = fp32(1/48) (0x3caaaaab)
//   3. per query, sequential in s = 0 .. S-1: m = -inf, m = max(m, score[s]); e_s = exp(sub(score[s], m));
//      sum = 0, sum = add(sum, e_s); p_s = div(e_s, sum)
//   4. out[q][i] = acc: acc = 0, acc = fma(p_s, k_s[i], acc) for s = 0 .. S-1     FP32 output (keys = values)
//
// Residual stream (who holds what; REF :886-910 and the model loop :1359-1443). h_L = the BF16 [M, 2304] input of
// visit L (h_0 = the normed embedding). B = the block-state list (BF16 rows, at most 4): empty at layer 0; after a
// boundary layer (L % 6 == 0: 0, 6, 12, 18) h_L is appended. Passes 2 and 3 of the loop restart from h_6 and from
// B = [h_0] (the pre-loop list); the post-loop layers continue from pass 3's list [h_0, h_6, h_12(pass 3)].
//   attention input  a = depth_mix(B + [h_L], attention query)       (layer 0: a = h_0, no mix; full layers: 3 queries
//                                                                     -> the q, k, v inputs)
//   post-attn conv   o = conv(attention output) BF16
//   residual         r = attn_residual(boundary, h_L, o): boundary ? float(o) : add(float(h_L), float(o))   FP32
//   MLP input        u = depth_mix(B' + [r], mlp query), B' = B + [h_L] at a boundary layer, else B
//   post-MLP conv    d = conv(MoE output) BF16
//   layer output     h_{L+1} = layer_out(r, d) = encode_bf16(add(r, float(d)))
//   final            f = depth_mix([h_0, h_6, h_12, h_18] + [h_24], output_query) -> final RMSNorm -> unembed
//   messages (passes 2, 3; per loop layer i = L - 6): depth_mix([h_12 of the previous pass] + [the previous pass's
//                    output h_18], message_input.query[i]: 4 queries q, k, v, mlp) (REF :1400 block_states[npre+1:])
// Checkpoint queries (FP32): attention_input of layer L = attn_res.attn_queries.{L-1} ([2304], or [3, 2304] = q, k, v
// on full layers), mlp_input = attn_res.mlp_queries.L, final = attn_res.output_query; message queries of loop layer i:
// [q, k, v, mlp] = [middle_cycle_cross_pass_queries[2i], middle_cycle_cross_pass_kv_queries[2i],
// middle_cycle_cross_pass_kv_queries[2i + 1], middle_cycle_cross_pass_queries[2i + 1]] (core_model_loading.py:843-849).
#include <vector>

#include "norm_ref.h"

namespace ref {

constexpr int kMaxDepthStates = 5;  // 4 block states + the partial (mlp_input of layer 18, final_input)
constexpr float kDepthScale = 0.0208333339542150497f;  // fp32(2304^-0.5) = 0x3caaaaab

// out[q * ldo + i] for q < nq, i < 2304. states[s]: FP32 [2304] rows (S of them), queries[q * ldq + i].
inline void depth_mix_row(const float *const *states, int S, const float *queries, int64_t ldq, int nq, float *out,
                          int64_t ldo) {
    std::vector<float> k(size_t(S) * kHidden);
    for (int s = 0; s < S; ++s) rms_norm<kReduceHidden>(states[s], kHidden, kRmsEps, nullptr, &k[size_t(s) * kHidden]);
    for (int q = 0; q < nq; ++q) {
        const float *qv = queries + int64_t(q) * ldq;
        float score[kMaxDepthStates], e[kMaxDepthStates];
        float m = -INFINITY;
        for (int s = 0; s < S; ++s) {
            score[s] = mul(reduce_dot<kReduceHidden>(qv, &k[size_t(s) * kHidden], kHidden), kDepthScale);
            m = max(m, score[s]);
        }
        float sum = 0.0f;
        for (int s = 0; s < S; ++s) {
            e[s] = ref::exp(sub(score[s], m));
            sum = add(sum, e[s]);
        }
        float p[kMaxDepthStates];
        for (int s = 0; s < S; ++s) p[s] = div(e[s], sum);
        for (int i = 0; i < kHidden; ++i) {
            float acc = 0.0f;
            for (int s = 0; s < S; ++s) acc = fma(p[s], k[size_t(s) * kHidden + i], acc);
            out[int64_t(q) * ldo + i] = acc;
        }
    }
}

// r = boundary ? float(o) : add(float(h), float(o)); h, o BF16, r FP32.
CERE_HD inline float attn_residual(bool boundary, uint16_t h, uint16_t o) {
    return boundary ? decode_bf16(o) : add(decode_bf16(h), decode_bf16(o));
}

// h_next = encode_bf16(add(r, float(d))); r FP32, d BF16.
CERE_HD inline uint16_t layer_out(float r, uint16_t d) { return encode_bf16(add(r, decode_bf16(d))); }

}  // namespace ref
