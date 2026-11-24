#pragma once
// Inkling 2 Mini (inkling design note 3.4, 5, G1): the CPU reference of the routed experts (arm Q, D-I1: the
// checkpoint's declared activation quantization as SGLang routed_quant.cuh computes it, with CERE's exact-block
// accumulation). Header-only C++17 on common_ref.h, quant_ref.h, router_ref.h; every operation one IEEE binary32
// operation rounded to nearest even, fused only where written ref::fma.
//
// Shapes: hidden H = 2304, expert intermediate I = 1152, 128 experts, top-8. Weights exactly as the checkpoint stores
// them (spec 5, never requantized):
//   NVFP4 (layers 0-19): w13 codes U8 [E, 2I, H/2], w13 scales E4M3 [E, 2I, H/16], w13 amax F32 [E], w13 outer
//     denominator F32 [1]; w2 codes [E, H, I/2], scales [E, H, I/16], amax [E], denominator [1]. Element 2j of a row
//     is the low nibble of byte j.
//   MXFP8 (layers 20-23): w13 E4M3 [E, 2I, H], w13 exponents UE8M0 [E, 2I, H/32]; w2 [E, H, I], [E, H, I/32].
//   w13 rows are gate/up interleaved as stored: row 2c = gate column c, row 2c + 1 = up column c (REF's conversion
//   mapping de-interleaves with Interleave(dim=1); SGLang's `gate_up_interleaved` layout; golden acc[:, 0::2/1::2]).
//
// GEMM element (one routed row r, one weight row n, K = H or I):
//   NVFP4: T_j = mul(S_j, mul(sx_j, sw_j)), S_j = sum of the 16 products e2m1(a_i) * e2m1(w_i) of block j (exact: a
//     multiple of 0.25 below 2^10, any order) and sx_j, sw_j the E4M3 block scales; T_j is exact in FP32.
//     acc = 0.0f; for j = 0 .. K/16 - 1: acc = add(acc, T_j)        (plain block order)
//     y = mul(acc, scale), scale = mul(mul(amax_x, amax_w[e]), rcp), rcp = div(1, mul(1536, denominator))
//   MXFP8: B_j = 0.0f; for i in block j (32, index order): B_j = fma(e4m3(a_i), e4m3(w_i), B_j)
//     T_j = mul(B_j, mul(ue8m0(ex_j), ue8m0(ew_j)))
//     acc = 0.0f; for j = 0 .. K/32 - 1: acc = add(acc, T_j); y = acc (no outer factor)
// One routed row (token t, slot k, expert e = id_k, gamma = weight_k from router_ref.h):
//   x_q = quantize(x_t) (NVFP4 per-token or MXFP8, quant_ref.h; the token's row, once for all its experts)
//   for c in 0..I-1: g = bf16(y(w13 row 2c)), u = bf16(y(w13 row 2c + 1))
//     h_c = bf16(capped_swiglu(g, u, gamma))      gamma inside, before the down quantization (routed_quant.cuh:249-313)
//   h_q = quantize(h) (the row's own amax for NVFP4)
//   for n in 0..H-1: o_{t,k,n} = bf16(y(w2 row n))
// capped_swiglu(g, u, gamma) = mul(mul(mul(mul(4, tanh(div(g, 4))), sigmoid(g)), mul(25, tanh(div(u, 25)))), gamma)
//   with ref::tanh / ref::sigmoid (never tanh.approx / ex2.approx).
// Combine (token t): out_n = bf16(s), s = 0.0f; for k = 0..7 (slot order = ascending expert id): s = add(s, o_{t,k,n})
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common_ref.h"
#include "quant_ref.h"
#include "router_ref.h"

namespace ref {

constexpr int kMoeHidden = 2304;
constexpr int kMoeInter = 1152;

CERE_HD inline float bf16_round(float v) { return decode_bf16(encode_bf16(v)); }

CERE_HD inline float capped_swiglu(float g, float u, float gamma) {
    const float gc = mul(4.0f, ref::tanh(div(g, 4.0f)));
    const float uc = mul(25.0f, ref::tanh(div(u, 25.0f)));
    return mul(mul(mul(gc, ref::sigmoid(g)), uc), gamma);
}

// NVFP4 block term T_j: a, w = 8 code bytes each, sa, sw the E4M3 scale bytes.
CERE_HD inline float nvfp4_block_term(const uint8_t *a, uint8_t sa, const uint8_t *w, uint8_t sw) {
    float s = 0.0f;
    for (int i = 0; i < 8; ++i) {
        s = add(s, mul(decode_e2m1(uint8_t(a[i] & 15u)), decode_e2m1(uint8_t(w[i] & 15u))));
        s = add(s, mul(decode_e2m1(uint8_t(a[i] >> 4)), decode_e2m1(uint8_t(w[i] >> 4))));
    }
    return mul(s, mul(decode_e4m3(sa), decode_e4m3(sw)));
}

// acc of one NVFP4 GEMM element over k (a multiple of 16): codes [k/2], scales [k/16] of the row and the weight row.
CERE_HD inline float nvfp4_dot(const uint8_t *a, const uint8_t *sa, const uint8_t *w, const uint8_t *sw, int k) {
    float acc = 0.0f;
    for (int j = 0; j < k / 16; ++j) acc = add(acc, nvfp4_block_term(a + 8 * j, sa[j], w + 8 * j, sw[j]));
    return acc;
}

CERE_HD inline float mxfp8_block_term(const uint8_t *a, uint8_t ea, const uint8_t *w, uint8_t ew) {
    float b = 0.0f;
    for (int i = 0; i < 32; ++i) b = fma(decode_e4m3(a[i]), decode_e4m3(w[i]), b);
    return mul(b, mul(decode_ue8m0(ea), decode_ue8m0(ew)));
}

CERE_HD inline float mxfp8_dot(const uint8_t *a, const uint8_t *ea, const uint8_t *w, const uint8_t *ew, int k) {
    float acc = 0.0f;
    for (int j = 0; j < k / 32; ++j) acc = add(acc, mxfp8_block_term(a + 32 * j, ea[j], w + 32 * j, ew[j]));
    return acc;
}

// rcp of an NVFP4 projection's outer factor.
CERE_HD inline float nvfp4_outer_rcp(float denominator) { return div(1.0f, mul(1536.0f, denominator)); }
CERE_HD inline float nvfp4_outer_scale(float amax_x, float amax_w, float rcp) { return mul(mul(amax_x, amax_w), rcp); }

// One layer's expert weights (host pointers, checkpoint layout, all 128 experts). Exactly one format is set.
struct MoeRefLayer {
    bool nvfp4 = true;
    const uint8_t *w13 = nullptr, *w13_scale = nullptr, *w2 = nullptr, *w2_scale = nullptr;
    const float *w13_amax = nullptr, *w2_amax = nullptr;  // NVFP4 only, [E]
    float w13_den = 1.0f, w2_den = 1.0f;                  // NVFP4 only
};

// One routed row: x BF16 [H] (the token's row), expert e, gamma; o BF16 [H]. Taps (nullable): gu BF16 [2I] (the
// interleaved gate/up, bf16 of each GEMM1 element), h BF16 [I] (capped SwiGLU x gamma).
inline void moe_expert_row(const MoeRefLayer &L, const uint16_t *x, int e, float gamma, uint16_t *o,
                           uint16_t *gu_tap = nullptr, uint16_t *h_tap = nullptr) {
    constexpr int H = kMoeHidden, I = kMoeInter;
    std::vector<float> xf(H), h(I);
    for (int i = 0; i < H; ++i) xf[i] = decode_bf16(x[i]);
    float g = 0.0f, u = 0.0f;
    if (L.nvfp4) {
        std::vector<uint8_t> xc(H / 2), xs(H / 16), hc(I / 2), hs(I / 16);
        const float ax = nvfp4_quantize_row(xf.data(), H, xc.data(), xs.data());
        const float s13 = nvfp4_outer_scale(ax, L.w13_amax[e], nvfp4_outer_rcp(L.w13_den));
        const uint8_t *w = L.w13 + size_t(e) * 2 * I * (H / 2);
        const uint8_t *ws = L.w13_scale + size_t(e) * 2 * I * (H / 16);
        for (int c = 0; c < I; ++c) {
            const size_t rg = size_t(2 * c), ru = rg + 1;
            g = bf16_round(mul(nvfp4_dot(xc.data(), xs.data(), w + rg * (H / 2), ws + rg * (H / 16), H), s13));
            u = bf16_round(mul(nvfp4_dot(xc.data(), xs.data(), w + ru * (H / 2), ws + ru * (H / 16), H), s13));
            h[c] = bf16_round(capped_swiglu(g, u, gamma));
            if (gu_tap != nullptr) gu_tap[2 * c] = encode_bf16(g), gu_tap[2 * c + 1] = encode_bf16(u);
        }
        const float ah = nvfp4_quantize_row(h.data(), I, hc.data(), hs.data());
        const float s2 = nvfp4_outer_scale(ah, L.w2_amax[e], nvfp4_outer_rcp(L.w2_den));
        const uint8_t *d = L.w2 + size_t(e) * H * (I / 2);
        const uint8_t *ds = L.w2_scale + size_t(e) * H * (I / 16);
        for (int n = 0; n < H; ++n)
            o[n] = encode_bf16(
                mul(nvfp4_dot(hc.data(), hs.data(), d + size_t(n) * (I / 2), ds + size_t(n) * (I / 16), I), s2));
    } else {
        std::vector<uint8_t> xv(H), xe(H / 32), hv(I), he(I / 32);
        mxfp8_quantize_row(xf.data(), H, xv.data(), xe.data());
        const uint8_t *w = L.w13 + size_t(e) * 2 * I * H;
        const uint8_t *ws = L.w13_scale + size_t(e) * 2 * I * (H / 32);
        for (int c = 0; c < I; ++c) {
            const size_t rg = size_t(2 * c), ru = rg + 1;
            g = bf16_round(mxfp8_dot(xv.data(), xe.data(), w + rg * H, ws + rg * (H / 32), H));
            u = bf16_round(mxfp8_dot(xv.data(), xe.data(), w + ru * H, ws + ru * (H / 32), H));
            h[c] = bf16_round(capped_swiglu(g, u, gamma));
            if (gu_tap != nullptr) gu_tap[2 * c] = encode_bf16(g), gu_tap[2 * c + 1] = encode_bf16(u);
        }
        mxfp8_quantize_row(h.data(), I, hv.data(), he.data());
        const uint8_t *d = L.w2 + size_t(e) * H * I;
        const uint8_t *ds = L.w2_scale + size_t(e) * H * (I / 32);
        for (int n = 0; n < H; ++n)
            o[n] = encode_bf16(mxfp8_dot(hv.data(), he.data(), d + size_t(n) * I, ds + size_t(n) * (I / 32), I));
    }
    if (h_tap != nullptr)
        for (int c = 0; c < I; ++c) h_tap[c] = encode_bf16(h[c]);
}

// Combine of one token: rows [8][H] BF16 in slot order -> out BF16 [H].
inline void moe_combine_row(const uint16_t *const *rows, uint16_t *out) {
    for (int n = 0; n < kMoeHidden; ++n) {
        float s = 0.0f;
        for (int k = 0; k < kRouterTopK; ++k) s = add(s, decode_bf16(rows[k][n]));
        out[n] = encode_bf16(s);
    }
}

// The experts and combine of one token with given routing (ids ascending, weights from router_ref.h).
inline void moe_experts_token(const MoeRefLayer &L, const uint16_t *x, const int32_t *ids, const float *weights,
                              uint16_t *out) {
    std::vector<uint16_t> o(size_t(kRouterTopK) * kMoeHidden);
    const uint16_t *rows[kRouterTopK];
    for (int k = 0; k < kRouterTopK; ++k) {
        moe_expert_row(L, x, ids[k], weights[k], o.data() + size_t(k) * kMoeHidden);
        rows[k] = o.data() + size_t(k) * kMoeHidden;
    }
    moe_combine_row(rows, out);
}

}  // namespace ref
