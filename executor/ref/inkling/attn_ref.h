#pragma once
// Inkling 2 Mini (inkling design note 3.3, G1): the attention ops between the BF16 projections (q, k, v, gate, rel,
// sink) and o_proj, for both layer kinds. CPU reference shared by the CUDA kernels
// (executor/cuda/src/kernels/inkling/attn.h) and the Vulkan shaders; numeric contract in common_ref.h (every op one
// IEEE binary32 rounding, explicit fma only, ref::exp/log/sigmoid, ref::reduce_*<L> lane structure, bf16 RNE).
// REF = transformers modeling_inkling2_loop.py @046bdb8c: L2 clips :176-236, relative bias :239-270, q/k/v :458-529,
// gated output :531-543, the layers :546-727, eager attention with the sink :111-142.
//
// Shapes (CFG text_config): 24 query / 8 KV heads (group 3: query head h reads KV head h / 3), head_dim 128,
// d_rel 16, rel_extent 1,024, window 1,024 (sliding layers: query p sees keys p-1023 .. p; full: 0 .. p).
// Row layouts (BF16, row-major): q [24 * 128], k / v [8 * 128], rel [24 * 16], sink [24], gate [24 * 128].
//
// ---- Exact order, per op ---------------------------------------------------------------------------------------
// tau(pos) (full layers, REF:215-220) = add(1, mul(0.1f, log(c))), c = div(float(pos + 1), 8192f), c < 1 -> 1.
//
// q post, per row and head h (REF:233-236 full, :198-202 sliding), x = 128 values:
//   x[i] = mul(bf16 q[i], bf16 q_scale[h][i]); n = sqrt(reduce_dot<32>(x, x, 128));
//   c = div(22.627416997969522f, add(n, 1e-6f)), c > 1 -> 1 (NaN stays NaN); y[i] = bf16(mul(x[i], c));
//   full layers then y[i] = bf16(mul(float(y[i]), tau(pos)))  (two BF16 roundings, as REF).
// rel post (REF:263-264): the same with 16 values, rel_scale [24][16], threshold 8.0f, reduce_dot<16>.
// k post (REF:183-187): the same with 128 values, no scale, threshold 22.627416997969522f, no tau.
// v post (REF:72-76, eps 1e-6): x = 128 values, r = rsqrt(add(div(reduce_dot<32>(x, x, 128), 128f), 1e-6f))
//   (rsqrt = div(1, sqrt)), y[i] = bf16(mul(x[i], r)).
//
// rel bias (REF:266-270) for query state r_h (16 BF16, after rel post) and backward distance d = p - j:
//   0 <= d < 1024: bias = float(bf16(reduce_dot<16>(r_h, proj[:, d], 16))), proj BF16 [16][1024] (proj[i][d] at
//   i * 1024 + d); otherwise bias = +0.0f.
// score of query p (head h) and key j (KV head h / 3), REF:126-128 in FP32 (CERE keeps the score in FP32; REF rounds
//   q k^T, the bias and their sum to BF16): s = add(mul(reduce_dot<32>(q_h, k_j, 128), kScale), bias),
//   kScale = float(128^-0.5).
//
// softmax and PV (REF:132-139; CERE order, FP32 probabilities, no BF16 cast before PV). The keys of query p are
// j in [lo, p] (lo = max(0, p - 1023) sliding, 0 full), visited in FIXED absolute blocks: tiles of 64 keys
// [64t, 64t+64), spans of 4 tiles (256 keys), groups of 16 spans (4,096 keys). A partial is (m, l, acc[128]), or
// empty. Within a span, tiles in ascending order, keys of a tile outside [lo, p] skipped (never read):
//   mt = max over the tile's keys, lane structure of reduce_max<32>: lane t takes keys 64t0+t, 64t0+t+32 (ascending,
//        valid only), then the halving tree;
//   mn = mt (span empty) or max(m, mt); p_j = exp(sub(s_j, mn));
//   ts = the tile's sum of p_j, lane structure of reduce_sum<32> (lane t: add(add(0, p_t), p_{t+32}), valid only);
//   span empty: l = ts, acc[d] = 0;  else a = exp(sub(m, mn)), l = add(mul(l, a), ts), acc[d] = mul(acc[d], a);
//   then acc[d] = fma(p_j, float(v_j[d]), acc[d]) for the tile's valid keys j ascending; m = mn.
// fold(A, B) (B empty: A; A empty: B): mn = max(mA, mB), a = exp(sub(mA, mn)), b = exp(sub(mB, mn)),
//   l = add(mul(lA, a), mul(lB, b)), acc[d] = add(mul(accA[d], a), mul(accB[d], b)), m = mn.
// Spans fold into their group in ascending order (group = fold(fold(span0, span1), ...)), groups into the total in
// ascending order; only spans / groups holding a valid key take part.
// epilogue (sink REF:132-135 with zero value, gate REF:540-542): z = float(sink[h]); mn = max(m, z);
//   a = exp(sub(m, mn)); e = exp(sub(z, mn)); L = add(mul(l, a), e); o[d] = bf16(div(mul(acc[d], a), L));
//   out[d] = bf16(mul(float(o[d]), sigmoid(float(gate[h][d])))).
// So a row's output depends only on its own q/rel/sink/gate row, its position and the key/value bits at its key
// positions: not on the chunking, the batch, the other rows or the prefill/decode kernel (G4, CERE D9).
//
// KV store: sliding rings [1024 slots][8 * 128] BF16, key at position j in slot j % 1024 (D-I3); a call of n rows
// at pos0 stores rows i >= n - 1024 (the earlier ones would be overwritten). Full: pool rows [tokens][8 * 128], key
// j at row slots[j] (slots null: row j). Bit copies.
#include <cstdint>

#include "common_ref.h"

namespace ref {
namespace attn {

constexpr int kHeads = 24;
constexpr int kKvHeads = 8;
constexpr int kGroup = kHeads / kKvHeads;  // query heads per KV head
constexpr int kHeadDim = 128;
constexpr int kQRow = kHeads * kHeadDim;     // 3072
constexpr int kKvRow = kKvHeads * kHeadDim;  // 1024
constexpr int kDRel = 16;
constexpr int kRelRow = kHeads * kDRel;  // 384
constexpr int kRelExtent = 1024;
constexpr int kWindow = 1024;
constexpr int kRingSlots = 1024;
constexpr int kTile = 64;
constexpr int kSpan = 256;
constexpr int kGroupKeys = 4096;
constexpr int kTilesPerSpan = kSpan / kTile;
constexpr int kSpansPerGroup = kGroupKeys / kSpan;
constexpr float kQkClip = 22.627416997969522f;
constexpr float kRelClip = 8.0f;
constexpr float kClipEps = 1e-6f;
constexpr float kVEps = 1e-6f;
constexpr float kLogAlpha = 0.1f;
constexpr float kLogFloor = 8192.0f;
constexpr float kScale = 0.08838834764831845f;  // float(128^-0.5)

enum Kind : int { kSliding = 0, kFull = 1 };

// First key of the query at position p.
CERE_HD inline int64_t key_lo(int kind, int64_t p) {
    return kind == kSliding ? (p >= kWindow - 1 ? p - (kWindow - 1) : 0) : 0;
}

CERE_HD inline float clamp_max1(float c) { return c > 1.0f ? 1.0f : c; }

CERE_HD inline float tau(int64_t pos) {
    float c = div(float(pos + 1), kLogFloor);
    if (c < 1.0f) c = 1.0f;
    return add(1.0f, mul(kLogAlpha, ref::log(c)));
}

// The clip factor of a vector whose sum of squares (reduce_dot order) is ss.
CERE_HD inline float clip_factor(float ss, float threshold) {
    return clamp_max1(div(threshold, add(ref::sqrt(ss), kClipEps)));
}

// One head vector of N values: (scale ?) L2 clip at `threshold`, BF16 out; then, with t != 0, bf16(mul(y, t)).
template <int N, int L>
inline void l2_clip(const uint16_t *in, const uint16_t *scale, float threshold, float t, uint16_t *out) {
    float x[N];
    for (int i = 0; i < N; ++i) {
        x[i] = decode_bf16(in[i]);
        if (scale) x[i] = mul(x[i], decode_bf16(scale[i]));
    }
    const float c = clip_factor(reduce_dot<L>(x, x, N), threshold);
    for (int i = 0; i < N; ++i) {
        out[i] = encode_bf16(mul(x[i], c));
        if (t != 0.0f) out[i] = encode_bf16(mul(decode_bf16(out[i]), t));
    }
}

// q post of one row: q_in / q_out [24 * 128], q_scale [24][128]; tau on full layers.
inline void q_post_row(int kind, const uint16_t *q_in, const uint16_t *q_scale, int64_t pos, uint16_t *q_out) {
    const float t = kind == kFull ? tau(pos) : 0.0f;
    for (int h = 0; h < kHeads; ++h)
        l2_clip<kHeadDim, kReduceHead>(q_in + h * kHeadDim, q_scale + h * kHeadDim, kQkClip, t, q_out + h * kHeadDim);
}

// rel post of one row: r_in / r_out [24 * 16], rel_scale [24][16].
inline void rel_post_row(int kind, const uint16_t *r_in, const uint16_t *rel_scale, int64_t pos, uint16_t *r_out) {
    const float t = kind == kFull ? tau(pos) : 0.0f;
    for (int h = 0; h < kHeads; ++h)
        l2_clip<kDRel, kReduceRel>(r_in + h * kDRel, rel_scale + h * kDRel, kRelClip, t, r_out + h * kDRel);
}

// k post of one row [8 * 128].
inline void k_post_row(const uint16_t *k_in, uint16_t *k_out) {
    for (int g = 0; g < kKvHeads; ++g)
        l2_clip<kHeadDim, kReduceHead>(k_in + g * kHeadDim, nullptr, kQkClip, 0.0f, k_out + g * kHeadDim);
}

// v post of one row [8 * 128]: unweighted RMSNorm per head.
inline void v_post_row(const uint16_t *v_in, uint16_t *v_out) {
    for (int g = 0; g < kKvHeads; ++g) {
        float x[kHeadDim];
        for (int i = 0; i < kHeadDim; ++i) x[i] = decode_bf16(v_in[g * kHeadDim + i]);
        const float r = rsqrt(add(div(reduce_dot<kReduceHead>(x, x, kHeadDim), float(kHeadDim)), kVEps));
        for (int i = 0; i < kHeadDim; ++i) v_out[g * kHeadDim + i] = encode_bf16(mul(x[i], r));
    }
}

// The bias of backward distance d for one head's rel state r (16 BF16 after rel post); proj BF16 [16][1024].
inline float rel_bias(const uint16_t *r, const uint16_t *proj, int64_t d) {
    if (d < 0 || d >= kRelExtent) return 0.0f;
    float a[kDRel], b[kDRel];
    for (int i = 0; i < kDRel; ++i) {
        a[i] = decode_bf16(r[i]);
        b[i] = decode_bf16(proj[i * kRelExtent + d]);
    }
    return decode_bf16(encode_bf16(reduce_dot<kReduceRel>(a, b, kDRel)));
}

// The score of one query head (q 128 BF16) and one key (128 BF16) plus its bias.
inline float score(const uint16_t *q, const uint16_t *k, float bias) {
    float a[kHeadDim], b[kHeadDim];
    for (int i = 0; i < kHeadDim; ++i) {
        a[i] = decode_bf16(q[i]);
        b[i] = decode_bf16(k[i]);
    }
    return add(mul(reduce_dot<kReduceHead>(a, b, kHeadDim), kScale), bias);
}

// fold's scalar part: B into A (both non-empty): returns the new m, l and the factors of A's and B's acc.
CERE_HD inline void fold_coeffs(float ma, float la, float mb, float lb, float *m, float *l, float *ca, float *cb) {
    const float mn = ref::max(ma, mb);
    *ca = ref::exp(sub(ma, mn));
    *cb = ref::exp(sub(mb, mn));
    *l = add(mul(la, *ca), mul(lb, *cb));
    *m = mn;
}
CERE_HD inline float fold_acc(float acc_a, float ca, float acc_b, float cb) {
    return add(mul(acc_a, ca), mul(acc_b, cb));
}

struct Partial {
    bool empty = true;
    float m = 0.0f, l = 0.0f;
    float acc[kHeadDim] = {};
};

inline void fold(Partial &a, const Partial &b) {
    if (b.empty) return;
    if (a.empty) {
        a = b;
        return;
    }
    float ca, cb;
    fold_coeffs(a.m, a.l, b.m, b.l, &a.m, &a.l, &ca, &cb);
    for (int d = 0; d < kHeadDim; ++d) a.acc[d] = fold_acc(a.acc[d], ca, b.acc[d], cb);
}

// Fail-first mutants (ktest only; the reference itself is always kNone).
enum Mutant : int { kNone = 0, kNoWindow = 1, kNoSink = 2, kBiasOffByOne = 3, kNoLogScale = 4 };

// The keys and values of one attention call as the kernels see them: key position j < pos0 comes from the cache
// (sliding ring slot j % 1024; full pool row slots[j], or j when slots is null), j >= pos0 from the call's new rows.
struct Keys {
    int kind;
    int64_t row, pos0;              // the sequence's first row in k/v (new rows) and its position
    const uint16_t *k, *v;          // new rows, row strides k_stride / v_stride (elements)
    int64_t k_stride, v_stride;
    const uint16_t *k_cache, *v_cache;  // [slots][8 * 128]
    const int64_t *slots;
    const uint16_t *key(int64_t j, int g) const { return at(k, k_stride, k_cache, j, g); }
    const uint16_t *value(int64_t j, int g) const { return at(v, v_stride, v_cache, j, g); }
    const uint16_t *at(const uint16_t *fresh, int64_t stride, const uint16_t *cache, int64_t j, int g) const {
        if (j >= pos0) return fresh + (row + (j - pos0)) * stride + g * kHeadDim;
        const int64_t slot = kind == kSliding ? j % kRingSlots : (slots ? slots[j] : j);
        return cache + slot * kKvRow + g * kHeadDim;
    }
};

// One tile step of a span partial for the query at p, head h, rel state r, keys [lo, p] of tile [t0, t0 + 64).
inline void tile_step(Partial &s, const Keys &keys, const uint16_t *q, const uint16_t *r, const uint16_t *proj,
                      int g, int64_t p, int64_t lo, int64_t t0) {
    float sc[kTile], pr[kTile];
    bool ok[kTile];
    for (int i = 0; i < kTile; ++i) {
        const int64_t j = t0 + i;
        ok[i] = j >= lo && j <= p;
        if (ok[i]) sc[i] = score(q, keys.key(j, g), rel_bias(r, proj, p - j));
    }
    float lane[32];
    for (int t = 0; t < 32; ++t) {
        float a = -INFINITY;
        if (ok[t]) a = ref::max(a, sc[t]);
        if (ok[t + 32]) a = ref::max(a, sc[t + 32]);
        lane[t] = a;
    }
    for (int off = 16; off >= 1; off /= 2)
        for (int t = 0; t < off; ++t) lane[t] = ref::max(lane[t], lane[t + off]);
    const float mt = lane[0];
    const float mn = s.empty ? mt : ref::max(s.m, mt);
    for (int i = 0; i < kTile; ++i)
        if (ok[i]) pr[i] = ref::exp(sub(sc[i], mn));
    for (int t = 0; t < 32; ++t) {
        float a = 0.0f;
        if (ok[t]) a = add(a, pr[t]);
        if (ok[t + 32]) a = add(a, pr[t + 32]);
        lane[t] = a;
    }
    const float ts = reduce_tree_<32>(lane);
    if (s.empty) {
        s.empty = false;
        s.l = ts;
        for (int d = 0; d < kHeadDim; ++d) s.acc[d] = 0.0f;
    } else {
        const float a = ref::exp(sub(s.m, mn));
        s.l = add(mul(s.l, a), ts);
        for (int d = 0; d < kHeadDim; ++d) s.acc[d] = mul(s.acc[d], a);
    }
    s.m = mn;
    for (int i = 0; i < kTile; ++i) {
        if (!ok[i]) continue;
        const uint16_t *v = keys.value(t0 + i, g);
        for (int d = 0; d < kHeadDim; ++d) s.acc[d] = fma(pr[i], decode_bf16(v[d]), s.acc[d]);
    }
}

// The epilogue: sink, normalization, BF16, gate, BF16. out / gate: the head's 128 values.
inline void epilogue(const Partial &t, uint16_t sink, const uint16_t *gate, uint16_t *out) {
    const float z = decode_bf16(sink);
    const float mn = ref::max(t.m, z);
    const float a = ref::exp(sub(t.m, mn));
    const float e = ref::exp(sub(z, mn));
    const float L = add(mul(t.l, a), e);
    for (int d = 0; d < kHeadDim; ++d) {
        const float o = decode_bf16(encode_bf16(div(mul(t.acc[d], a), L)));
        out[d] = encode_bf16(mul(o, sigmoid(decode_bf16(gate[d]))));
    }
}

// The total partial of query p, head h (groups -> spans -> tiles, ascending).
inline Partial attend(const Keys &keys, const uint16_t *q, const uint16_t *r, const uint16_t *proj, int h,
                      int64_t p) {
    const int g = h / kGroup;
    const int64_t lo = key_lo(keys.kind, p);
    Partial total;
    for (int64_t g0 = lo / kGroupKeys * kGroupKeys; g0 <= p; g0 += kGroupKeys) {
        Partial group;
        for (int64_t s0 = g0; s0 < g0 + kGroupKeys && s0 <= p; s0 += kSpan) {
            if (s0 + kSpan <= lo) continue;
            Partial span;
            for (int64_t t0 = s0; t0 < s0 + kSpan && t0 <= p; t0 += kTile) {
                if (t0 + kTile <= lo) continue;
                tile_step(span, keys, q, r, proj, g, p, lo, t0);
            }
            fold(group, span);
        }
        fold(total, group);
    }
    return total;
}

// Attention of one query row (all 24 heads): q [24 * 128], r [24 * 16] (after post), sink [24], gate [24 * 128],
// out [24 * 128]; p = the row's position.
inline void attention_row(const Keys &keys, const uint16_t *q, const uint16_t *r, const uint16_t *proj,
                          const uint16_t *sink, const uint16_t *gate, int64_t p, uint16_t *out) {
    for (int h = 0; h < kHeads; ++h) {
        const Partial t = attend(keys, q + h * kHeadDim, r + h * kDRel, proj, h, p);
        epilogue(t, sink[h], gate + h * kHeadDim, out + h * kHeadDim);
    }
}

}  // namespace attn
}  // namespace ref
