// Inkling 2 Mini S4 (inkling design note 3.4, G1): the routed-MoE CPU references executor/ref/inkling/{quant,
// router,moe}_ref.h. Exhaustive E4M3/E2M1 round trips and ties (the cvt.rn.satfinite emulation), saturation, the
// NVFP4 fouroversix choice and its zero row, the MXFP8 exponent rule, the router's tie rule and weights, the exact
// block terms and the capped SwiGLU against double precision. The GPU kernels are compared with these references bit
// for bit by executor/tests/gpu/test_inkling_kernels_moe.py.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../../ref/inkling/moe_ref.h"
#include "../../ref/inkling/quant_ref.h"
#include "../../ref/inkling/router_ref.h"
#include "check.h"

namespace {

uint32_t rng_state = 0x9e3779b9u;
uint32_t next_u32() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
float uniform() { return float(next_u32() >> 8) * (1.0f / 16777216.0f); }
float gaussian() {
    const float u = std::max(uniform(), 1e-7f), v = uniform();
    return std::sqrt(-2.0f * std::log(u)) * std::cos(6.2831853f * v);
}

void test_e4m3() {
    // Round trip of every finite code, both signs.
    for (uint32_t c = 0; c < 0x7f; ++c) {
        CHECK_EQ(ref::encode_e4m3_satfinite(ref::decode_e4m3(uint8_t(c))), c);
        if (c != 0) CHECK_EQ(ref::encode_e4m3_satfinite(ref::decode_e4m3(uint8_t(c | 0x80u))), c | 0x80u);
    }
    // Midpoints between neighbours round to the even code; just off the midpoint rounds to the nearer one.
    for (uint32_t c = 0; c + 1 < 0x7f; ++c) {
        const float lo = ref::decode_e4m3(uint8_t(c)), hi = ref::decode_e4m3(uint8_t(c + 1));
        const float mid = (lo + hi) * 0.5f;
        CHECK_EQ(ref::encode_e4m3_satfinite(mid), (c & 1u) ? c + 1 : c);
        CHECK_EQ(ref::encode_e4m3_satfinite(std::nextafter(mid, 0.0f)), c);
        CHECK_EQ(ref::encode_e4m3_satfinite(std::nextafter(mid, 1e30f)), c + 1);
    }
    CHECK_EQ(ref::encode_e4m3_satfinite(448.0f), 0x7eu);
    CHECK_EQ(ref::encode_e4m3_satfinite(470.0f), 0x7eu);
    CHECK_EQ(ref::encode_e4m3_satfinite(1e30f), 0x7eu);
    CHECK_EQ(ref::encode_e4m3_satfinite(INFINITY), 0x7eu);
    CHECK_EQ(ref::encode_e4m3_satfinite(-500.0f), 0xfeu);
    CHECK_EQ(ref::encode_e4m3_satfinite(-0.0f), 0x80u);
    CHECK_EQ(ref::encode_e4m3_satfinite(1e-30f), 0x00u);
}

void test_e2m1() {
    for (uint32_t c = 0; c < 16; ++c) {
        if (c == 8) continue;
        CHECK_EQ(ref::encode_e2m1_satfinite(ref::decode_e2m1(uint8_t(c))), c);
    }
    const float ties[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    const uint32_t want[7] = {0, 2, 2, 4, 4, 6, 6};
    for (int i = 0; i < 7; ++i) {
        CHECK_EQ(ref::encode_e2m1_satfinite(ties[i]), want[i]);
        CHECK_EQ(ref::encode_e2m1_satfinite(-ties[i]), want[i] | 8u);
    }
    CHECK_EQ(ref::encode_e2m1_satfinite(7.0f), 7u);
    CHECK_EQ(ref::encode_e2m1_satfinite(INFINITY), 7u);
    CHECK_EQ(ref::encode_e2m1_satfinite(-0.0f), 8u);
    CHECK_EQ(ref::encode_e2m1_satfinite(std::nextafter(0.25f, 1.0f)), 1u);
}

// Dequantized NVFP4 row (codes * scale * outer).
std::vector<float> dequant_nvfp4(const std::vector<uint8_t> &codes, const std::vector<uint8_t> &scales, float amax) {
    const size_t k = codes.size() * 2;
    std::vector<float> y(k);
    const float outer = amax * float(1.0 / 1536.0);
    for (size_t i = 0; i < k; ++i) {
        const uint8_t c = (i & 1) ? uint8_t(codes[i / 2] >> 4) : uint8_t(codes[i / 2] & 15u);
        y[i] = ref::decode_e2m1(c) * ref::decode_e4m3(scales[i / 16]) * outer;
    }
    return y;
}

void test_nvfp4() {
    constexpr int K = 2304;
    std::vector<float> x(K, 0.0f);
    std::vector<uint8_t> codes(K / 2), scales(K / 16);
    int n4 = -1;
    // Zero row: no NaN, every code and scale 0, the /6 candidate (equal errors).
    CHECK(ref::nvfp4_quantize_row(x.data(), K, codes.data(), scales.data(), &n4) == 0.0f);
    CHECK_EQ(n4, 0);
    bool zero = true;
    for (uint8_t c : codes) zero = zero && c == 0;
    for (uint8_t s : scales) zero = zero && s == 0;
    CHECK(zero);
    // Gaussian rows: both candidates occur, and the per-block choice never has a larger error than always-/6.
    int total4 = 0;
    for (int row = 0; row < 16; ++row) {
        for (float &v : x) v = ref::decode_bf16(ref::encode_bf16(gaussian() * (row == 0 ? 1e-3f : 1.0f)));
        const float amax = ref::nvfp4_quantize_row(x.data(), K, codes.data(), scales.data(), &n4);
        total4 += n4;
        float bmax[K / 16];
        for (int b = 0; b < K / 16; ++b) {
            bmax[b] = 0.0f;
            for (int i = 0; i < 16; ++i) bmax[b] = std::fmax(bmax[b], std::fabs(x[16 * b + i]));
        }
        std::vector<uint8_t> c6(K / 2), s6(K / 16);
        for (int b = 0; b < K / 16; ++b)
            CHECK(!ref::nvfp4_block(x.data() + 16 * b, amax, bmax[b], c6.data() + 8 * b, s6.data() + b, false));
        const std::vector<float> y = dequant_nvfp4(codes, scales, amax), y6 = dequant_nvfp4(c6, s6, amax);
        double e = 0.0, e6 = 0.0;
        for (int i = 0; i < K; ++i) e += std::fabs(double(y[i]) - x[i]), e6 += std::fabs(double(y6[i]) - x[i]);
        CHECK(e <= e6);
    }
    CHECK(total4 > 0 && total4 < 16 * (K / 16));
    // Saturated scale: one huge value with a tiny rest keeps the scale finite.
    for (float &v : x) v = 1e-30f;
    x[5] = 3e38f;
    ref::nvfp4_quantize_row(x.data(), K, codes.data(), scales.data());
    for (uint8_t s : scales) CHECK(s != 0x7f && s != 0xff);
}

void test_mxfp8() {
    float x[32];
    uint8_t v[32];
    for (float &f : x) f = 448.0f;
    CHECK_EQ(ref::mxfp8_block(x, v), 127u);
    for (uint8_t b : v) CHECK_EQ(b, 0x7eu);
    for (float &f : x) f = 0.0f;
    CHECK_EQ(ref::mxfp8_block(x, v), 0u);
    for (uint8_t b : v) CHECK_EQ(b, 0u);
    for (float &f : x) f = 0.5f;
    x[0] = -1.0f;  // raw = 1/448 rounds up to 2^-8: values x * 256
    CHECK_EQ(ref::mxfp8_block(x, v), 119u);
    CHECK_EQ(v[0], ref::encode_e4m3_satfinite(-256.0f));
    CHECK_EQ(v[1], ref::encode_e4m3_satfinite(128.0f));
    for (float &f : x) f = 1e-39f;  // raw below 2^-127: exponent 0, inverse 2^127
    CHECK_EQ(ref::mxfp8_block(x, v), 0u);
    CHECK_EQ(v[0], ref::encode_e4m3_satfinite(1e-39f * 1.7014118e38f));
    for (float &f : x) f = 3e38f;  // raw ~2^119.2 rounds up to 2^120
    CHECK_EQ(ref::mxfp8_block(x, v), 247u);
    x[0] = INFINITY;  // the exponent clamps at 254; inf * 2^-127 clamps to 448
    CHECK_EQ(ref::mxfp8_block(x, v), 254u);
    CHECK_EQ(v[0], 0x7eu);
}

void test_router() {
    float biased[ref::kRouterExperts], score[ref::kRouterExperts];
    int32_t ids[8];
    float w[8];
    for (int e = 0; e < ref::kRouterExperts; ++e) biased[e] = score[e] = 0.5f;
    ref::router_select(biased, ref::kRouterExperts, ids);
    for (int k = 0; k < 8; ++k) CHECK_EQ(ids[k], k);  // all tied: the lowest ids
    ref::router_weights(score, ids, w);
    for (int k = 0; k < 8; ++k) CHECK(w[k] == 1.0f);
    // Selection by biased value, slots ascending by id; -0 ties +0; NaN ranks last.
    for (int e = 0; e < ref::kRouterExperts; ++e) biased[e] = -1.0f;
    const int pick[8] = {100, 3, 77, 64, 5, 127, 40, 9};
    for (int k = 0; k < 8; ++k) biased[pick[k]] = 1.0f + float(k);
    biased[0] = NAN;
    ref::router_select(biased, ref::kRouterExperts, ids);
    const int sorted[8] = {3, 5, 9, 40, 64, 77, 100, 127};
    for (int k = 0; k < 8; ++k) CHECK_EQ(ids[k], sorted[k]);
    for (int e = 0; e < ref::kRouterExperts; ++e) biased[e] = (e % 2) ? -0.0f : 0.0f;
    biased[20] = 2.0f;
    ref::router_select(biased, ref::kRouterExperts, ids);
    const int tie[8] = {0, 1, 2, 3, 4, 5, 6, 20};
    for (int k = 0; k < 8; ++k) CHECK_EQ(ids[k], tie[k]);
    // Weights: score / (sum + 1e-20) * 8 with the sum in slot order.
    for (int e = 0; e < ref::kRouterExperts; ++e) score[e] = 0.01f * float(e + 1);
    const int32_t some[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    ref::router_weights(score, some, w);
    float sum = 0.0f;
    for (int k = 0; k < 8; ++k) sum = sum + score[some[k]];
    for (int k = 0; k < 8; ++k) CHECK(w[k] == score[some[k]] / (sum + 1e-20f) * 8.0f);
}

void test_gemm_terms() {
    // Block terms are exact: equal to the double-precision product sum for random codes and scales.
    for (int trial = 0; trial < 2000; ++trial) {
        uint8_t a[8], w[8];
        for (int i = 0; i < 8; ++i) a[i] = uint8_t(next_u32()), w[i] = uint8_t(next_u32());
        uint8_t sa = uint8_t(next_u32() % 0x7f), sw = uint8_t(next_u32() % 0x7f);
        double s = 0.0;
        for (int i = 0; i < 16; ++i) {
            const uint8_t ca = (i & 1) ? uint8_t(a[i / 2] >> 4) : uint8_t(a[i / 2] & 15u);
            const uint8_t cw = (i & 1) ? uint8_t(w[i / 2] >> 4) : uint8_t(w[i / 2] & 15u);
            s += double(ref::decode_e2m1(ca)) * double(ref::decode_e2m1(cw));
        }
        const double want = s * double(ref::decode_e4m3(sa)) * double(ref::decode_e4m3(sw));
        CHECK(double(ref::nvfp4_block_term(a, sa, w, sw)) == want);
    }
    // Capped SwiGLU within 4 ulp-ish of double; gamma inside.
    for (int trial = 0; trial < 2000; ++trial) {
        const float g = gaussian() * 8.0f, u = gaussian() * 40.0f, gamma = uniform() * 2.0f;
        const double want = 4.0 * std::tanh(g / 4.0) * (1.0 / (1.0 + std::exp(-double(g)))) * 25.0 *
                            std::tanh(u / 25.0) * gamma;
        const float got = ref::capped_swiglu(g, u, gamma);
        CHECK(std::fabs(double(got) - want) <= 1e-5 * std::fabs(want) + 1e-30);
    }
}

}  // namespace

int main() {
    test_e4m3();
    test_e2m1();
    test_nvfp4();
    test_mxfp8();
    test_router();
    test_gemm_terms();
    return cere::test::finish("test_inkling_moe_ref");
}
