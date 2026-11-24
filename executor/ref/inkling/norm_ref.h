#pragma once
// Inkling 2 Mini (inkling design note 3.2, G1): RMSNorm, weighted and unweighted. CPU reference shared by the CUDA
// kernels (executor/cuda/src/kernels/inkling/norm.h) and the Vulkan shaders; numeric contract in common_ref.h.
// REF (transformers modeling_inkling2_loop.py): Inkling2LoopRMSNorm :731-745 (weighted: input_layernorm,
// q/k/v_input_layernorm, post_attention_layernorm, the final norm, embed_norm), Inkling2LoopUnweightedRMSNorm :67-76
// (the depth / message mixes' key norm, the message gates' input norm, the value norm).
//
// Exact order, one row of n elements x (FP32 values; a BF16 input is widened exactly first):
//   ss   = reduce_dot<L>(x, x, n)        lane t: acc = 0, acc = fma(x[i], x[i], acc) for i = t, t+L, t+2L, ...;
//                                        then the halving tree add(lane[i], lane[i+off]), off = L/2 .. 1
//   mean = div(ss, float(n))
//   r    = rsqrt(add(mean, eps))         = div(1, sqrt(add(mean, eps)))
//   unweighted: y[i] = mul(x[i], r)
//   weighted:   y[i] = mul(float(w[i]), mul(x[i], r))      (w BF16, widened exactly)
// then the call site's output: FP32 as is, or BF16 by encode_bf16 (RNE). L = kReduceHidden (256) for the hidden row
// (n = 2304); eps = kRmsEps (config rms_norm_eps 1e-6, also the unweighted norm's default eps).
// An overflowing sum of squares gives ss = +inf, r = 0 and y = +-0 (finite x); an all-zero row gives y = 0.
#include "common_ref.h"

namespace ref {

constexpr int kHidden = 2304;
constexpr float kRmsEps = 1e-6f;

template <int L> CERE_HD inline float rms_inv(const float *x, int n, float eps) {
    const float ss = reduce_dot<L>(x, x, n);
    return rsqrt(add(div(ss, float(n)), eps));
}

// y[0..n) = RMSNorm(x[0..n)) in FP32; w: BF16 [n] or nullptr (unweighted). y may alias x.
template <int L> CERE_HD inline void rms_norm(const float *x, int n, float eps, const uint16_t *w, float *y) {
    const float r = rms_inv<L>(x, n, eps);
    for (int i = 0; i < n; ++i) {
        const float v = mul(x[i], r);
        y[i] = w ? mul(decode_bf16(w[i]), v) : v;
    }
}

}  // namespace ref
