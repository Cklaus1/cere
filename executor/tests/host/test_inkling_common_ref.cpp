// Inkling 2 Mini (inkling design note, G1): the shared numeric contract executor/ref/inkling/common_ref.h.
// Exhaustive format decodes, the reduction order (lane-strided sequential accumulation, then the halving tree, equal
// to lane 0 of a xor butterfly), and the transcendentals within 2 ulp of the double-precision value on a sweep.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../../ref/inkling/common_ref.h"
#include "check.h"

namespace {

// Distance in units in the last place between a float and the exactly rounded double value.
double ulp_error(float got, double want) {
    if (std::isnan(want)) return std::isnan(got) ? 0.0 : 1e30;
    if (std::isinf(want)) return got == float(want) ? 0.0 : 1e30;
    const float w = float(want);
    if (std::isinf(w)) return std::isinf(got) && (got > 0) == (w > 0) ? 0.0 : 1e30;
    const double ulp = std::ldexp(1.0, std::max(std::ilogb(w == 0.0f ? 1e-45f : w), -126) - 23);
    return std::fabs(double(got) - want) / ulp;
}

uint32_t rng_state = 0x12345678u;
uint32_t next_u32() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
float next_float(float lo, float hi) { return lo + (hi - lo) * float(next_u32() >> 8) * (1.0f / 16777216.0f); }

template <typename F, typename D> double max_ulp(F f, D d, float lo, float hi, int samples) {
    double worst = 0.0;
    for (int i = 0; i <= samples; ++i) {
        const float x = lo + (hi - lo) * float(i) / float(samples);
        const double e = ulp_error(f(x), d(double(x)));
        if (e > worst) worst = e;
    }
    for (int i = 0; i < samples; ++i) {
        const float x = next_float(lo, hi);
        const double e = ulp_error(f(x), d(double(x)));
        if (e > worst) worst = e;
    }
    return worst;
}

void test_formats() {
    // bf16: exact widen of every pattern; encode inverts decode on every non-NaN pattern; RNE ties.
    for (uint32_t v = 0; v < 65536; ++v) {
        const float f = ref::decode_bf16(uint16_t(v));
        uint32_t u;
        std::memcpy(&u, &f, 4);
        CHECK_EQ(u, v << 16);
        if (!std::isnan(f)) CHECK_EQ(ref::encode_bf16(f), v);
        else CHECK(std::isnan(ref::decode_bf16(ref::encode_bf16(f))));
    }
    CHECK_EQ(ref::encode_bf16(ref::bits_float(0x3f808000u)), 0x3f80);  // tie, even below
    CHECK_EQ(ref::encode_bf16(ref::bits_float(0x3f818000u)), 0x3f82);  // tie, even above
    CHECK_EQ(ref::encode_bf16(ref::bits_float(0x3f808001u)), 0x3f81);

    const float e2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (int v = 0; v < 16; ++v) {
        const float want = (v & 8) ? -e2m1[v & 7] : e2m1[v & 7];
        CHECK(ref::decode_e2m1(uint8_t(v)) == want);
        CHECK(std::signbit(ref::decode_e2m1(uint8_t(v))) == bool(v & 8));
    }

    for (int v = 0; v < 256; ++v) {
        const int e = (v >> 3) & 15, m = v & 7;
        const float got = ref::decode_e4m3(uint8_t(v));
        if (e == 15 && m == 7) {
            CHECK(std::isnan(got));
            continue;
        }
        const double mag = e == 0 ? std::ldexp(double(m), -9) : std::ldexp(1.0 + m / 8.0, e - 7);
        CHECK(double(got) == ((v & 0x80) ? -mag : mag));
        CHECK(std::signbit(got) == bool(v & 0x80));
    }
    CHECK(ref::decode_e4m3(0x7e) == 448.0f);

    for (int v = 0; v < 256; ++v) {
        const float got = ref::decode_ue8m0(uint8_t(v));
        if (v == 255) CHECK(std::isnan(got));
        else CHECK(double(got) == std::ldexp(1.0, v - 127));
    }
}

// The butterfly a warp or subgroup runs: every lane i takes op(lane i, lane i ^ off) for off = L/2 .. 1.
template <int L> float butterfly_lane0(std::vector<float> lane, bool is_max) {
    for (int off = L / 2; off >= 1; off /= 2) {
        std::vector<float> next(L);
        for (int i = 0; i < L; ++i)
            next[i] = is_max ? ref::max(lane[i], lane[i ^ off]) : ref::add(lane[i], lane[i ^ off]);
        lane = next;
    }
    return lane[0];
}

template <int L> void check_reduction_width(int n) {
    std::vector<float> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        a[i] = next_float(-4.0f, 4.0f) * std::ldexp(1.0f, int(next_u32() % 24) - 12);
        b[i] = next_float(-2.0f, 2.0f);
    }
    std::vector<float> s(L), d(L), m(L);
    for (int t = 0; t < L; ++t) {
        float as = 0.0f, ad = 0.0f, am = -INFINITY;
        for (int i = t; i < n; i += L) {
            as = as + a[i];
            ad = std::fma(a[i], b[i], ad);
            am = std::fmax(am, a[i]);
        }
        s[t] = as;
        d[t] = ad;
        m[t] = am;
    }
    const float sum = ref::reduce_sum<L>(a.data(), n);
    const float dot = ref::reduce_dot<L>(a.data(), b.data(), n);
    const float mx = ref::reduce_max<L>(a.data(), n);
    CHECK(ref::float_bits(sum) == ref::float_bits(butterfly_lane0<L>(s, false)));
    CHECK(ref::float_bits(dot) == ref::float_bits(butterfly_lane0<L>(d, false)));
    CHECK(ref::float_bits(mx) == ref::float_bits(butterfly_lane0<L>(m, true)));
    float want_max = -INFINITY;
    for (float x : a) want_max = std::fmax(want_max, x);
    CHECK(mx == want_max);
}

void test_reductions() {
    // A hand-worked L = 4 case: lanes {1+1e8, 2, 3, 4} -> tree (l0+l2)+(l1+l3), not a left-to-right sum.
    const float x[5] = {1.0f, 2.0f, 3.0f, 4.0f, 1e8f};
    const float lane0 = 1.0f + 1e8f;
    const float want = (lane0 + 3.0f) + (2.0f + 4.0f);
    CHECK(ref::reduce_sum<4>(x, 5) == want);
    CHECK(ref::reduce_max<4>(x, 5) == 1e8f);
    CHECK(ref::reduce_sum<4>(x, 0) == 0.0f);
    CHECK(ref::reduce_max<4>(x, 0) == -INFINITY);
    // The dot fuses each product into its lane accumulator.
    const float a[2] = {1.0f + 0x1p-12f, 3.0f};
    const float b[2] = {1.0f - 0x1p-12f, 0.0f};
    CHECK(ref::reduce_dot<1>(a, b, 2) == std::fma(3.0f, 0.0f, std::fma(a[0], b[0], 0.0f)));
    CHECK(ref::reduce_dot<1>(a, b, 1) == 1.0f - 0x1p-24f);

    check_reduction_width<ref::kReduceHidden>(2304);
    check_reduction_width<ref::kReduceHead>(128);
    check_reduction_width<ref::kReduceRel>(16);
    check_reduction_width<ref::kReduceRank>(64);
    check_reduction_width<32>(37);
}

void test_transcendentals() {
    const int n = 200000;
    const double e_exp = max_ulp(ref::exp, [](double v) { return std::exp(v); }, -103.9f, 88.7f, n);
    const double e_exp_small = max_ulp(ref::exp, [](double v) { return std::exp(v); }, -1.0f, 1.0f, n);
    const double e_log = max_ulp(ref::log, [](double v) { return std::log(v); }, 1e-30f, 1e30f, n);
    const double e_log1 = max_ulp(ref::log, [](double v) { return std::log(v); }, 0.5f, 2.0f, n);
    const double e_tanh = max_ulp(ref::tanh, [](double v) { return std::tanh(v); }, -10.0f, 10.0f, n);
    const double e_tanh_small = max_ulp(ref::tanh, [](double v) { return std::tanh(v); }, -1.0f, 1.0f, n);
    const double e_sig = max_ulp(ref::sigmoid, [](double v) { return 1.0 / (1.0 + std::exp(-v)); }, -80.0f,
                                 80.0f, n);
    std::printf("max ulp: exp %.3f (|x|<1 %.3f) log %.3f ([0.5,2] %.3f) tanh %.3f (|x|<1 %.3f) sigmoid %.3f\n",
                e_exp, e_exp_small, e_log, e_log1, e_tanh, e_tanh_small, e_sig);
    CHECK(e_exp <= 2.0);
    CHECK(e_exp_small <= 2.0);
    CHECK(e_log <= 2.0);
    CHECK(e_log1 <= 2.0);
    CHECK(e_tanh <= 2.0);
    CHECK(e_tanh_small <= 2.0);
    CHECK(e_sig <= 2.0);
    // Log over the subnormal range and the exact points.
    CHECK(max_ulp(ref::log, [](double v) { return std::log(v); }, 1e-45f, 1e-38f, 20000) <= 2.0);
    CHECK(ref::exp(0.0f) == 1.0f);
    CHECK(ref::log(1.0f) == 0.0f);
    CHECK(ref::tanh(0.0f) == 0.0f);
    CHECK(ref::sigmoid(0.0f) == 0.5f);
    CHECK(std::isinf(ref::exp(89.0f)));
    CHECK(ref::exp(-104.0f) == 0.0f);
    CHECK(ref::exp(-103.0f) > 0.0f);
    CHECK(ref::log(0.0f) == -INFINITY);
    CHECK(std::isnan(ref::log(-1.0f)));
    CHECK(ref::log(INFINITY) == INFINITY);
    CHECK(ref::tanh(20.0f) == 1.0f);
    CHECK(ref::tanh(-20.0f) == -1.0f);
    CHECK(ref::sigmoid(-200.0f) == 0.0f);
    CHECK(ref::sigmoid(200.0f) == 1.0f);
    CHECK(std::isnan(ref::exp(NAN)) && std::isnan(ref::tanh(NAN)) && std::isnan(ref::sigmoid(NAN)));
    CHECK(ref::rsqrt(4.0f) == 0.5f);
    CHECK(ref::rsqrt(2.0f) == 1.0f / std::sqrt(2.0f));
}

}  // namespace

int main() {
    test_formats();
    test_reductions();
    test_transcendentals();
    return cere::test::finish("test_inkling_common_ref");
}
