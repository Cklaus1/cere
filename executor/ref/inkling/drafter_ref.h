#pragma once
// Inkling 2 Mini block drafter (inkling design note 3.5, S9, G1): CPU reference of the drafter's new kernels,
// shared by the CUDA kernels (phase B) and the Vulkan shaders (S12); numeric contract in common_ref.h. The algorithm
// is SGLang 61915e54's Inkling2ForConditionalGenerationMTP (srt/models/inkling2_mtp.py, dflash.py, dspark.py,
// dspark_components/dspark_draft.py, kernels/ops/speculative/inkling2_mtp.py, dspark/markov_sample.py); the torch
// reference is scripts/native/inkling_reference/drafter_ref.py and the write-up an internal note
// math_reference: sglang-upstream
// The dense BF16 projections (q/k/v/o, MLP, k_proj_ctx/v_proj_ctx, rnn_proj, the base-logit unembed) are the
// executor's existing GEMM plans and are not part of this contract (G2/G3 cover them, as for the target).
//
// ---- Exact order, per op ---------------------------------------------------------------------------------------
// fuse alpha (once, host): alpha[r][t] = bf16(div(e_t, s)), e_t = exp(sub(w[r][t], m)), m = max over t ascending
//   (from -inf), s = add(add(add(e_0, e_1), e_2), e_3); w = fuse_weight BF16 [4][4]. Row r < 3: layer r's context,
//   row 3: the Markov head's f_rnn.
// mix_rmsnorm (taps BF16 [4][2304] in tap order 5, 23, 41, 48; a = alpha[r] as float): mixed[i] = fma(x_3[i], a_3,
//   fma(x_2[i], a_2, fma(x_1[i], a_1, mul(x_0[i], a_0)))); y[i] = bf16(mul(w[i], mul(mixed[i], r))) with
//   r = rsqrt(add(div(reduce_dot<256>(mixed, mixed, 2304), 2304f), 1e-6f)) (norm_ref.h's order).
// RoPE (neox halves, theta 1e6, head 128): the HOST stages per position p cos[i] = float(cos(p * f_i)),
//   sin[i] = float(sin(p * f_i)), f_i = 1e6^(-2i/128) evaluated in double (rope_cos_sin); kernels read the staged
//   rows, so both backends see the same bits. y[i] = sub(mul(x[i], c[i]), mul(x[i+64], s[i])),
//   y[i+64] = add(mul(x[i+64], c[i]), mul(x[i], s[i])), i < 64.
// head norm + rope (per head of 128, w BF16 [128], eps 1e-6): n[i] = mul(w[i], mul(x[i], r)), r over reduce_dot<32>;
//   block q/k (SGLang's fused qk-norm-rope): out = bf16(rope(n)); context k (k_norm_ctx then apply_k_rope, two
//   modules): out = bf16(rope(float(bf16(n)))).
// block attention (8 query rows at L..L+7, 24 heads, GQA 3; NON-causal): keys of every row are the absolute
//   positions [lo, L + 8), lo = max(0, L - 1024) on layers 0 and 2 (window anchored at the block start), 0 on layer
//   1; j < L read the context cache, j >= L the block's own k/v. score s_j = mul(reduce_dot<32>(q, k_j, 128), kScale).
//   The online softmax visits FIXED absolute blocks exactly as attn_ref.h (S2): tiles of 64 keys, spans of 4 tiles,
//   groups of 16 spans, the same tile step and fold; no bias, no sink, no gate: o[d] = bf16(div(acc[d], l)).
// Markov gates (proj BF16 [768] = gate | cand | out, state BF16 [256]): g = bf16(sigmoid(gate)), kept =
//   bf16(mul(g, state)), blend = bf16(mul(bf16(sub(1, g)), bf16(tanh(cand)))), state' = bf16(add(kept, blend)),
//   latent = bf16(tanh(out)). state_0 = 0 for every block.
// Markov vocab step (id v < 200058; fold_w BF16 [vocab][256], fold_b BF16 [vocab], base BF16 [vocab] the block
//   row's unembed): bias = bf16(add(reduce_dot<64>(latent, fold_w[v], 256), fold_b[v])), x = bf16(add(base, bias)),
//   logit = bf16(mul(tanh(div(x, 30)), 30)). The draft is the first id of the maximal logit (ids >= 200058 never
//   win; a NaN logit loses).
// fold (once, host, SGLang refresh_fold): fold_w[v][j] = bf16(float(sum_k lm[v][k] * W[k][j])), fold_b[v] =
//   bf16(float(sum_k lm[v][k] * b[k])), sums in double, k ascending; the FP64 result reaches the BF16 buffer through
//   float (torch's copy_ into BF16 constructs c10::BFloat16 from float), so two roundings, as here. lm = the unembed
//   BF16 [vocab][2304], W/b = markov_head.hidden_out (BF16 [2304][256], [2304]). SGLang's FP64 GEMM sums in cuBLAS
//   order, not k ascending; the two can differ in BF16 only when the FP64 sums straddle a rounding boundary
//   [INFERENCE: rare, not checked element-wise]. The executor computes the fold on the host with this routine.
#include <cmath>
#include <cstdint>

#include "common_ref.h"

namespace ref {
namespace drafter {

constexpr int kHidden = 2304;
constexpr int kTaps = 4;
constexpr int kAlphaRows = 4;  // 3 layers + the Markov head
constexpr int kLayers = 3;
constexpr int kBlock = 8;
constexpr int kGamma = kBlock - 1;
constexpr int kHeads = 24;
constexpr int kKvHeads = 8;
constexpr int kGroup = kHeads / kKvHeads;
constexpr int kHeadDim = 128;
constexpr int kHalf = kHeadDim / 2;
constexpr int kQRow = kHeads * kHeadDim;     // 3072
constexpr int kKvRow = kKvHeads * kHeadDim;  // 1024
constexpr int kWindow = 1024;
constexpr int kRank = 256;
constexpr int kProj = 3 * kRank;  // 768
constexpr int kVocab = 200058;    // unpadded vocab: the step vocab limit
constexpr int kReduceRank256 = 64;
constexpr int kTile = 64;
constexpr int kSpan = 256;
constexpr int kGroupKeys = 4096;
constexpr float kEps = 1e-6f;
constexpr float kSoftcap = 30.0f;
constexpr float kScale = 0.08838834764831845f;  // float(128^-0.5)
constexpr double kTheta = 1e6;

CERE_HD inline float bf16r(float x) { return decode_bf16(encode_bf16(x)); }
CERE_HD inline bool is_local(int layer) { return layer == 0 || layer == 2; }
// First context key of a block starting at L on `layer`.
CERE_HD inline int64_t key_lo(int layer, int64_t L) { return is_local(layer) && L > kWindow ? L - kWindow : 0; }

// alpha [4][4] (float holding BF16 values) from fuse_weight BF16 [4][4].
inline void fuse_alpha(const uint16_t *w, float *alpha) {
    for (int r = 0; r < kAlphaRows; ++r) {
        float m = -INFINITY, e[kTaps];
        for (int t = 0; t < kTaps; ++t) m = ref::max(m, decode_bf16(w[r * kTaps + t]));
        float s = 0.0f;
        for (int t = 0; t < kTaps; ++t) {
            e[t] = ref::exp(sub(decode_bf16(w[r * kTaps + t]), m));
            s = t ? add(s, e[t]) : e[t];
        }
        for (int t = 0; t < kTaps; ++t) alpha[r * kTaps + t] = bf16r(div(e[t], s));
    }
}

// One position: taps BF16 [4][2304] (tap t at t * tap_stride), a = alpha row [4], w BF16 [2304] -> y BF16 [2304].
inline void mix_rmsnorm(const uint16_t *taps, int64_t tap_stride, const float *a, const uint16_t *w, uint16_t *y) {
    static thread_local float mixed[kHidden];
    for (int i = 0; i < kHidden; ++i) {
        float m = mul(decode_bf16(taps[i]), a[0]);
        for (int t = 1; t < kTaps; ++t) m = ref::fma(decode_bf16(taps[t * tap_stride + i]), a[t], m);
        mixed[i] = m;
    }
    const float r = rsqrt(add(div(reduce_dot<kReduceHidden>(mixed, mixed, kHidden), float(kHidden)), kEps));
    for (int i = 0; i < kHidden; ++i) y[i] = encode_bf16(mul(decode_bf16(w[i]), mul(mixed[i], r)));
}

// The staged rotary row of position p (host only: double cos/sin, rounded once).
inline void rope_cos_sin(int64_t p, float *c, float *s) {
    for (int i = 0; i < kHalf; ++i) {
        const double f = std::pow(kTheta, -2.0 * i / kHeadDim);
        const double ang = double(p) * f;
        c[i] = float(std::cos(ang));
        s[i] = float(std::sin(ang));
    }
}

CERE_HD inline void rope(float *x, const float *c, const float *s) {
    for (int i = 0; i < kHalf; ++i) {
        const float a = x[i], b = x[i + kHalf];
        x[i] = sub(mul(a, c[i]), mul(b, s[i]));
        x[i + kHalf] = add(mul(b, c[i]), mul(a, s[i]));
    }
}

// One head of 128: RMSNorm (w BF16 [128]) then RoPE; `ctx`: round to BF16 between them (k_norm_ctx, apply_k_rope).
inline void head_norm_rope(const uint16_t *x, const uint16_t *w, const float *c, const float *s, bool ctx,
                           uint16_t *out) {
    float v[kHeadDim];
    for (int i = 0; i < kHeadDim; ++i) v[i] = decode_bf16(x[i]);
    const float r = rsqrt(add(div(reduce_dot<kReduceHead>(v, v, kHeadDim), float(kHeadDim)), kEps));
    for (int i = 0; i < kHeadDim; ++i) {
        v[i] = mul(decode_bf16(w[i]), mul(v[i], r));
        if (ctx) v[i] = bf16r(v[i]);
    }
    rope(v, c, s);
    for (int i = 0; i < kHeadDim; ++i) out[i] = encode_bf16(v[i]);
}

// The keys of one block: positions [lo, L) from the context cache rows ctx_k/ctx_v [positions][1024] (position j
// at row j; a ring caller maps j % slots), [L, L + 8) from the block's own rows blk_k/blk_v [8][1024].
struct BlockKeys {
    int64_t L;
    const uint16_t *ctx_k, *ctx_v, *blk_k, *blk_v;
    int64_t ring;  // 0: ctx row j; else ctx row j % ring
    const uint16_t *at(const uint16_t *ctx, const uint16_t *blk, int64_t j, int g) const {
        if (j >= L) return blk + (j - L) * kKvRow + g * kHeadDim;
        return ctx + (ring ? j % ring : j) * kKvRow + g * kHeadDim;
    }
};

struct Partial {
    bool empty = true;
    float m = 0.0f, l = 0.0f;
    float acc[kHeadDim] = {};
};

// attn_ref.h's tile step without the relative bias: keys [lo, hi) of tile [t0, t0 + 64).
inline void tile_step(Partial &s, const BlockKeys &keys, const uint16_t *q, int g, int64_t lo, int64_t hi,
                      int64_t t0) {
    float sc[kTile], pr[kTile], a[kHeadDim], b[kHeadDim];
    bool ok[kTile];
    for (int d = 0; d < kHeadDim; ++d) a[d] = decode_bf16(q[d]);
    for (int i = 0; i < kTile; ++i) {
        const int64_t j = t0 + i;
        ok[i] = j >= lo && j < hi;
        if (!ok[i]) continue;
        const uint16_t *k = keys.at(keys.ctx_k, keys.blk_k, j, g);
        for (int d = 0; d < kHeadDim; ++d) b[d] = decode_bf16(k[d]);
        sc[i] = mul(reduce_dot<kReduceHead>(a, b, kHeadDim), kScale);
    }
    float lane[32];
    for (int t = 0; t < 32; ++t) {
        float x = -INFINITY;
        if (ok[t]) x = ref::max(x, sc[t]);
        if (ok[t + 32]) x = ref::max(x, sc[t + 32]);
        lane[t] = x;
    }
    for (int off = 16; off >= 1; off /= 2)
        for (int t = 0; t < off; ++t) lane[t] = ref::max(lane[t], lane[t + off]);
    const float mt = lane[0];
    const float mn = s.empty ? mt : ref::max(s.m, mt);
    for (int i = 0; i < kTile; ++i)
        if (ok[i]) pr[i] = ref::exp(sub(sc[i], mn));
    for (int t = 0; t < 32; ++t) {
        float x = 0.0f;
        if (ok[t]) x = add(x, pr[t]);
        if (ok[t + 32]) x = add(x, pr[t + 32]);
        lane[t] = x;
    }
    const float ts = reduce_tree_<32>(lane);
    if (s.empty) {
        s.empty = false;
        s.l = ts;
        for (int d = 0; d < kHeadDim; ++d) s.acc[d] = 0.0f;
    } else {
        const float f = ref::exp(sub(s.m, mn));
        s.l = add(mul(s.l, f), ts);
        for (int d = 0; d < kHeadDim; ++d) s.acc[d] = mul(s.acc[d], f);
    }
    s.m = mn;
    for (int i = 0; i < kTile; ++i) {
        if (!ok[i]) continue;
        const uint16_t *v = keys.at(keys.ctx_v, keys.blk_v, t0 + i, g);
        for (int d = 0; d < kHeadDim; ++d) s.acc[d] = ref::fma(pr[i], decode_bf16(v[d]), s.acc[d]);
    }
}

inline void fold(Partial &a, const Partial &b) {
    if (b.empty) return;
    if (a.empty) {
        a = b;
        return;
    }
    const float mn = ref::max(a.m, b.m);
    const float ca = ref::exp(sub(a.m, mn)), cb = ref::exp(sub(b.m, mn));
    a.l = add(mul(a.l, ca), mul(b.l, cb));
    for (int d = 0; d < kHeadDim; ++d) a.acc[d] = add(mul(a.acc[d], ca), mul(b.acc[d], cb));
    a.m = mn;
}

// One block row's head h: q BF16 [128] -> out BF16 [128]. Every row of the block sees the same keys [lo, L + 8).
inline void attend(const BlockKeys &keys, int layer, const uint16_t *q, int h, uint16_t *out) {
    const int g = h / kGroup;
    const int64_t lo = key_lo(layer, keys.L), hi = keys.L + kBlock;
    Partial total;
    for (int64_t g0 = lo / kGroupKeys * kGroupKeys; g0 < hi; g0 += kGroupKeys) {
        Partial group;
        for (int64_t s0 = g0; s0 < g0 + kGroupKeys && s0 < hi; s0 += kSpan) {
            if (s0 + kSpan <= lo) continue;
            Partial span;
            for (int64_t t0 = s0; t0 < s0 + kSpan && t0 < hi; t0 += kTile)
                if (t0 + kTile > lo) tile_step(span, keys, q, g, lo, hi, t0);
            fold(group, span);
        }
        fold(total, group);
    }
    for (int d = 0; d < kHeadDim; ++d) out[d] = encode_bf16(div(total.acc[d], total.l));
}

// One Markov step's gates: proj BF16 [768], state BF16 [256] updated in place, latent BF16 [256].
inline void markov_gates(const uint16_t *proj, uint16_t *state, uint16_t *latent) {
    for (int i = 0; i < kRank; ++i) {
        const float g = bf16r(ref::sigmoid(decode_bf16(proj[i])));
        const float kept = bf16r(mul(g, decode_bf16(state[i])));
        const float blend = bf16r(mul(bf16r(sub(1.0f, g)), bf16r(ref::tanh(decode_bf16(proj[kRank + i])))));
        state[i] = encode_bf16(add(kept, blend));
        latent[i] = encode_bf16(ref::tanh(decode_bf16(proj[2 * kRank + i])));
    }
}

// The step logit of id v (as BF16 bits).
inline uint16_t step_logit(const uint16_t *latent, const uint16_t *fold_w_row, uint16_t fold_b, uint16_t base) {
    float a[kRank], b[kRank];
    for (int j = 0; j < kRank; ++j) {
        a[j] = decode_bf16(latent[j]);
        b[j] = decode_bf16(fold_w_row[j]);
    }
    const float bias = bf16r(add(reduce_dot<kReduceRank256>(a, b, kRank), decode_bf16(fold_b)));
    const float x = bf16r(add(decode_bf16(base), bias));
    return encode_bf16(mul(ref::tanh(div(x, kSoftcap)), kSoftcap));
}

// One Markov step over the vocab: logits BF16 [kVocab] (optional), returns the draft id.
inline uint32_t markov_step(const uint16_t *latent, const uint16_t *fold_w, const uint16_t *fold_b,
                            const uint16_t *base, uint16_t *logits) {
    uint32_t best = 0;
    float best_v = -INFINITY;
    for (uint32_t v = 0; v < uint32_t(kVocab); ++v) {
        const uint16_t l = step_logit(latent, fold_w + uint64_t(v) * kRank, fold_b[v], base[v]);
        if (logits) logits[v] = l;
        const float x = decode_bf16(l);
        if (x > best_v) {
            best_v = x;
            best = v;
        }
    }
    return best;
}

// Host fold of one vocab row: lm BF16 [2304], hidden_out W BF16 [2304][256] and b BF16 [2304].
inline void fold_row(const uint16_t *lm, const uint16_t *w, const uint16_t *b, uint16_t *fold_w, uint16_t *fold_b) {
    for (int j = 0; j < kRank; ++j) {
        double acc = 0.0;
        for (int k = 0; k < kHidden; ++k) acc += double(decode_bf16(lm[k])) * double(decode_bf16(w[k * kRank + j]));
        fold_w[j] = encode_bf16(float(acc));
    }
    double acc = 0.0;
    for (int k = 0; k < kHidden; ++k) acc += double(decode_bf16(lm[k])) * double(decode_bf16(b[k]));
    *fold_b = encode_bf16(float(acc));
}

}  // namespace drafter
}  // namespace ref
