#pragma once
// Inkling 2 Mini (inkling design note 3.4, G1): the CPU reference of the MoE router (REF modeling_inkling2_loop.py
// Inkling2LoopTopKRouter, :1152-1172). Header-only C++17 on common_ref.h; every operation one IEEE binary32 operation,
// round to nearest even, fused only where written ref::fma.
//
// Per token (x = the MLP input row, hidden 2304, BF16 widened exactly; W = mlp.gate.weight [128, 2304], stored BF16
// in the checkpoint and widened exactly, as REF's FP32 parameter holds it; bias = mlp.gate.bias, REF's
// e_score_correction_bias, FP32 [128]):
//   logit_e  = ref::reduce_dot<256>(x, W_e, 2304)        (lane t: fma over elements t, t+256, ..; halving tree)
//   score_e  = ref::sigmoid(logit_e)
//   biased_e = add(score_e, bias_e)
//   top-8 by biased value: e ranks before f iff biased_e > biased_f, or they are equal (+0 == -0) and e < f (the
//     lowest expert id wins a tie; torch.topk(sorted=False) leaves ties unspecified). A NaN biased value ranks below
//     every number (and NaNs among themselves by id). A NaN's sign and payload are not part of G1 (an overflowing
//     dot's inf - inf is 0xffc00000 on x86, 0x7fffffff on the GPU): logits compare with any NaN equal to any NaN.
//   slots: the 8 selected ids sorted ascending by expert id (slot k = the k-th smallest selected id); every consumer
//     (the expert combine) works in slot order = expert-id order, as REF's index_add_ over ascending expert ids.
//   sum = 0.0f; for k in 0..7: sum = add(sum, score_{id_k})
//   weight_k = mul(div(score_{id_k}, add(sum, 1e-20f)), 8.0f)   (route_scale 8.0)
#include <cstdint>

#include "common_ref.h"

namespace ref {

constexpr int kRouterHidden = 2304;
constexpr int kRouterExperts = 128;
constexpr int kRouterTopK = 8;
constexpr float kRouteScale = 8.0f;

// True when expert a (value va) ranks before expert b (value vb).
CERE_HD inline bool router_before(float va, int a, float vb, int b) {
    const bool na = va != va, nb = vb != vb;
    if (na != nb) return nb;
    if (!na && va != vb) return va > vb;
    return a < b;
}

// Top-8 of biased [experts] (experts >= 8), ascending ids.
CERE_HD inline void router_select(const float *biased, int experts, int32_t *ids) {
    int32_t pick[kRouterTopK];
    for (int k = 0; k < kRouterTopK; ++k) {
        int best = -1;
        for (int e = 0; e < experts; ++e) {
            bool taken = false;
            for (int j = 0; j < k; ++j) taken = taken || pick[j] == e;
            if (taken) continue;
            if (best < 0 || router_before(biased[e], e, biased[best], best)) best = e;
        }
        pick[k] = best;
    }
    for (int i = 1; i < kRouterTopK; ++i) {  // insertion sort ascending
        const int32_t v = pick[i];
        int j = i - 1;
        while (j >= 0 && pick[j] > v) {
            pick[j + 1] = pick[j];
            --j;
        }
        pick[j + 1] = v;
    }
    for (int k = 0; k < kRouterTopK; ++k) ids[k] = pick[k];
}

// Weights of the selected ids (slot order) from the sigmoid scores [experts].
CERE_HD inline void router_weights(const float *scores, const int32_t *ids, float *weights) {
    float sum = 0.0f;
    for (int k = 0; k < kRouterTopK; ++k) sum = add(sum, scores[ids[k]]);
    const float denom = add(sum, 1e-20f);
    for (int k = 0; k < kRouterTopK; ++k) weights[k] = mul(div(scores[ids[k]], denom), kRouteScale);
}

// One token: x [hidden] and w [experts * hidden] widened to float; logits/scores (nullable) [experts].
inline void route_row(const float *x, const float *w, const float *bias, int32_t *ids, float *weights,
                      float *logits_out = nullptr) {
    float score[kRouterExperts], biased[kRouterExperts];
    for (int e = 0; e < kRouterExperts; ++e) {
        const float logit = reduce_dot<kReduceHidden>(x, w + size_t(e) * kRouterHidden, kRouterHidden);
        if (logits_out != nullptr) logits_out[e] = logit;
        score[e] = sigmoid(logit);
        biased[e] = add(score[e], bias[e]);
    }
    router_select(biased, kRouterExperts, ids);
    router_weights(score, ids, weights);
}

}  // namespace ref
