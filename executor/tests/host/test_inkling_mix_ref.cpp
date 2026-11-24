// Inkling 2 Mini (inkling design note G1, slice S3): the CPU references of the residual-stream ops
// (executor/ref/inkling/{norm,embed,conv,depth,msg,softcap}_ref.h) against FP64 evaluations of the REF formulas, plus
// their structural properties: the constants' bit patterns, conv chunk/decode invariance with the state carried, the
// depth mix's exact cases (one state, one-hot softmax), the message gate's identity case and the softcap's limits.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../../ref/inkling/conv_ref.h"
#include "../../ref/inkling/depth_ref.h"
#include "../../ref/inkling/embed_ref.h"
#include "../../ref/inkling/msg_ref.h"
#include "../../ref/inkling/norm_ref.h"
#include "../../ref/inkling/softcap_ref.h"
#include "check.h"

namespace {

constexpr int H = ref::kHidden;

uint32_t rng_state = 0x9e3779b9u;
uint32_t next_u32() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
float uniform(float lo, float hi) { return lo + (hi - lo) * float(next_u32() >> 8) * (1.0f / 16777216.0f); }
uint16_t bf16(float x) { return ref::encode_bf16(x); }
float f(uint16_t v) { return ref::decode_bf16(v); }
std::vector<uint16_t> rand_bf16(size_t n, float lo, float hi) {
    std::vector<uint16_t> v(n);
    for (auto &e : v) e = bf16(uniform(lo, hi));
    return v;
}
std::vector<float> rand_f32(size_t n, float lo, float hi) {
    std::vector<float> v(n);
    for (auto &e : v) e = uniform(lo, hi);
    return v;
}

// |got - want| relative to the largest |want| of the row: max over the row.
double max_rel(const std::vector<double> &want, const float *got) {
    double scale = 0.0, err = 0.0;
    for (double w : want) scale = std::fmax(scale, std::fabs(w));
    for (size_t i = 0; i < want.size(); ++i) err = std::fmax(err, std::fabs(double(got[i]) - want[i]));
    return scale > 0.0 ? err / scale : err;
}

void check_le(double value, double limit, const char *what) {
    if (!(value <= limit)) {
        std::fprintf(stderr, "%s: %.3g > %.3g\n", what, value, limit);
        CHECK(value <= limit);
    }
}

std::vector<double> rms64(const float *x, int n, const uint16_t *w) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
    const double r = 1.0 / std::sqrt(ss / n + double(ref::kRmsEps));
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = double(x[i]) * r * (w ? double(f(w[i])) : 1.0);
    return y;
}

void test_constants() {
    CHECK_EQ(ref::float_bits(ref::kConvStaticScale), 0x400f1bbdu);
    CHECK(ref::kConvStaticScale == float(std::sqrt(5.0)));
    CHECK_EQ(ref::float_bits(ref::kDepthScale), 0x3caaaaabu);
    CHECK(ref::kDepthScale == float(std::pow(2304.0, -0.5)));
    CHECK(ref::kRmsEps == float(1e-6));
}

void test_norm() {
    for (int trial = 0; trial < 8; ++trial) {
        const float mag = trial < 4 ? 1.0f : 1e4f;
        auto x = rand_f32(H, -mag, mag);
        auto w = rand_bf16(H, -2.0f, 2.0f);
        std::vector<float> y(H), yu(H);
        ref::rms_norm<256>(x.data(), H, ref::kRmsEps, w.data(), y.data());
        ref::rms_norm<256>(x.data(), H, ref::kRmsEps, nullptr, yu.data());
        check_le(max_rel(rms64(x.data(), H, w.data()), y.data()), 4e-7, "weighted rmsnorm vs fp64");
        check_le(max_rel(rms64(x.data(), H, nullptr), yu.data()), 4e-7, "unweighted rmsnorm vs fp64");
        // weight 1 = unweighted
        std::vector<uint16_t> ones(H, bf16(1.0f));
        std::vector<float> y1(H);
        ref::rms_norm<256>(x.data(), H, ref::kRmsEps, ones.data(), y1.data());
        CHECK(std::memcmp(y1.data(), yu.data(), sizeof(float) * H) == 0);
    }
    // all zero -> 0; an overflowing sum of squares -> +-0
    std::vector<float> z(H, 0.0f), y(H, 1.0f);
    ref::rms_norm<256>(z.data(), H, ref::kRmsEps, nullptr, y.data());
    for (float v : y) CHECK(v == 0.0f);
    std::vector<float> big(H, 3e19f);
    ref::rms_norm<256>(big.data(), H, ref::kRmsEps, nullptr, y.data());
    for (float v : y) CHECK(v == 0.0f);
    // embedding: the gather + weighted norm, one BF16 rounding
    const int vocab = 5;
    auto table = rand_bf16(size_t(vocab) * H, -0.05f, 0.05f);
    auto w = rand_bf16(H, 0.5f, 1.5f);
    std::vector<uint16_t> out(H);
    ref::embed_norm_row(table.data(), 3, w.data(), ref::kRmsEps, out.data());
    std::vector<float> x(H), want(H);
    for (int i = 0; i < H; ++i) x[i] = f(table[3 * H + i]);
    ref::rms_norm<256>(x.data(), H, ref::kRmsEps, w.data(), want.data());
    for (int i = 0; i < H; ++i) CHECK_EQ(out[i], bf16(want[i]));
}

// FP64 conv of one sequence from a zero state (the REF formulas), rows [0, n).
std::vector<double> conv64(const std::vector<uint16_t> &x, int n, int C, const uint16_t *w, const uint16_t *sw,
                           const uint16_t *dyn) {
    std::vector<double> y(size_t(n) * C);
    for (int t = 0; t < n; ++t)
        for (int c = 0; c < C; ++c) {
            double acc = 0.0;
            for (int lag = 0; lag < 4; ++lag) {
                const double xv = t - lag >= 0 ? double(f(x[size_t(t - lag) * C + c])) : 0.0;
                const double tap = w ? double(f(w[c * 4 + 3 - lag]))
                                     : double(f(sw[lag * C + c])) + double(f(dyn[size_t(t) * 32 + (c / 128) * 4 + lag]));
                acc += tap * xv;
            }
            y[size_t(t) * C + c] = (acc + double(f(x[size_t(t) * C + c]))) / std::sqrt(w ? 5.0 : 9.0);
        }
    return y;
}

void test_conv(bool dynamic) {
    const int C = dynamic ? 1024 : 2304, n = 11;
    auto x = rand_bf16(size_t(n) * C, -3.0f, 3.0f);
    auto w = rand_bf16(size_t(C) * 4, -1.0f, 1.0f);
    auto sw = rand_bf16(size_t(4) * C, -1.0f, 1.0f);
    auto dyn = rand_bf16(size_t(n) * 32, -0.5f, 0.5f);
    auto run = [&](int t0, int len, std::vector<float> &state, std::vector<uint16_t> &y) {
        if (dynamic) {
            ref::conv_dynamic_seq(&x[size_t(t0) * C], C, len, C, sw.data(), &dyn[size_t(t0) * 32], 32, state.data(),
                                  &y[size_t(t0) * C], C);
        } else {
            ref::conv_static_seq(&x[size_t(t0) * C], C, len, C, w.data(), state.data(), &y[size_t(t0) * C], C);
        }
    };
    std::vector<float> state_whole(size_t(3) * C, 0.0f);
    std::vector<uint16_t> whole(size_t(n) * C);
    run(0, n, state_whole, whole);
    // the state after n rows is x[n-3 .. n-1]
    for (int j = 0; j < 3; ++j)
        for (int c = 0; c < C; ++c) CHECK(state_whole[size_t(j) * C + c] == f(x[size_t(n - 3 + j) * C + c]));
    // vs FP64: each output within one BF16 rounding (plus FP32 noise) of the exact value
    auto want = conv64(x, n, C, dynamic ? nullptr : w.data(), sw.data(), dyn.data());
    double worst = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double tol = std::fabs(want[i]) * (1.0 / 256.0) + 1e-6;
        worst = std::fmax(worst, std::fabs(double(f(whole[i])) - want[i]) / tol);
    }
    check_le(worst, 1.0, dynamic ? "dynamic conv vs fp64 (bf16 ulps)" : "static conv vs fp64 (bf16 ulps)");
    // every split into chunks of 1, 2, 3 and 5 rows (the state carried) and every two-chunk split == whole
    for (int chunk : {1, 2, 3, 5}) {
        std::vector<float> state(size_t(3) * C, 0.0f);
        std::vector<uint16_t> y(size_t(n) * C);
        for (int t0 = 0; t0 < n; t0 += chunk) run(t0, std::min(chunk, n - t0), state, y);
        CHECK(y == whole);
        CHECK(std::memcmp(state.data(), state_whole.data(), sizeof(float) * state.size()) == 0);
    }
    for (int cut = 0; cut <= n; ++cut) {
        std::vector<float> state(size_t(3) * C, 0.0f);
        std::vector<uint16_t> y(size_t(n) * C);
        run(0, cut, state, y);
        run(cut, n - cut, state, y);
        CHECK(y == whole);
        CHECK(std::memcmp(state.data(), state_whole.data(), sizeof(float) * state.size()) == 0);
    }
}

void test_depth() {
    for (int S = 1; S <= ref::kMaxDepthStates; ++S) {
        std::vector<std::vector<float>> st(S);
        for (auto &s : st) s = rand_f32(H, -2.0f, 2.0f);
        std::vector<const float *> ptr(S);
        for (int s = 0; s < S; ++s) ptr[s] = st[s].data();
        const int nq = 3;
        auto q = rand_f32(size_t(nq) * H, -1.0f, 1.0f);
        std::vector<float> out(size_t(nq) * H);
        ref::depth_mix_row(ptr.data(), S, q.data(), H, nq, out.data(), H);
        // FP64
        std::vector<std::vector<double>> k(S);
        for (int s = 0; s < S; ++s) k[s] = rms64(st[s].data(), H, nullptr);
        for (int qi = 0; qi < nq; ++qi) {
            std::vector<double> sc(S), want(H, 0.0);
            double m = -INFINITY, sum = 0.0;
            for (int s = 0; s < S; ++s) {
                double dot = 0.0;
                for (int i = 0; i < H; ++i) dot += double(q[size_t(qi) * H + i]) * k[s][i];
                sc[s] = dot / 48.0;
                m = std::fmax(m, sc[s]);
            }
            for (int s = 0; s < S; ++s) sum += std::exp(sc[s] - m);
            for (int s = 0; s < S; ++s)
                for (int i = 0; i < H; ++i) want[i] += std::exp(sc[s] - m) / sum * k[s][i];
            check_le(max_rel(want, &out[size_t(qi) * H]), 1e-5, "depth mix vs fp64");
        }
        if (S == 1) {  // p = 1 exactly: the output is the normed state
            std::vector<float> k0(H);
            ref::rms_norm<256>(st[0].data(), H, ref::kRmsEps, nullptr, k0.data());
            CHECK(std::memcmp(out.data(), k0.data(), sizeof(float) * H) == 0);
        }
    }
    // one-hot: a query aligned with state 1 at a large scale selects it exactly
    std::vector<std::vector<float>> st(3);
    for (auto &s : st) s = rand_f32(H, -1.0f, 1.0f);
    std::vector<float> k1(H), q(H), out(H);
    ref::rms_norm<256>(st[1].data(), H, ref::kRmsEps, nullptr, k1.data());
    for (int i = 0; i < H; ++i) q[i] = 1e3f * k1[i];
    const float *ptr[3] = {st[0].data(), st[1].data(), st[2].data()};
    ref::depth_mix_row(ptr, 3, q.data(), H, 1, out.data(), H);
    CHECK(std::memcmp(out.data(), k1.data(), sizeof(float) * H) == 0);
    // residual helpers
    CHECK(ref::attn_residual(true, bf16(1.0f), bf16(0.5f)) == 0.5f);
    CHECK(ref::attn_residual(false, bf16(1.0f), bf16(0.5f)) == 1.5f);
    CHECK_EQ(ref::layer_out(1.0f, bf16(1.0f / 512.0f)), bf16(1.0f));         // tie to even
    CHECK_EQ(ref::layer_out(1.0f, bf16(3.0f / 512.0f)), bf16(1.0f + 1.0f / 128.0f));
}

void test_msg() {
    auto h = rand_f32(H, -3.0f, 3.0f), msg = rand_f32(H, -3.0f, 3.0f);
    auto down = rand_bf16(size_t(64) * 4608, -0.05f, 0.05f);
    auto forget = rand_bf16(size_t(H) * 64, -0.3f, 0.3f), gain = rand_bf16(size_t(H) * 64, -0.3f, 0.3f);
    std::vector<float> out(H);
    ref::msg_gate_row(h.data(), msg.data(), down.data(), forget.data(), gain.data(), ref::kRmsEps, out.data());
    // FP64 with the REF's two BF16 roundings (the concat and the down projection's output)
    auto nh = rms64(h.data(), H, nullptr), nm = rms64(msg.data(), H, nullptr);
    std::vector<double> g(4608), d(64), want(H);
    for (int i = 0; i < H; ++i) {
        g[i] = f(bf16(float(nh[i])));
        g[H + i] = f(bf16(float(nm[i])));
    }
    for (int r = 0; r < 64; ++r) {
        double acc = 0.0;
        for (int i = 0; i < 4608; ++i) acc += double(f(down[size_t(r) * 4608 + i])) * g[i];
        d[r] = f(bf16(float(acc)));
    }
    for (int c = 0; c < H; ++c) {
        double fa = 0.0, ma = 0.0;
        for (int j = 0; j < 64; ++j) {
            fa += double(f(forget[size_t(c) * 64 + j])) * d[j];
            ma += double(f(gain[size_t(c) * 64 + j])) * d[j];
        }
        want[c] = (1.0 - std::tanh(fa)) * h[c] + std::tanh(ma) * msg[c];
    }
    // the BF16 roundings of g and d are data-dependent (a near-tie flips one BF16 ulp), so the bound is loose
    check_le(max_rel(want, out.data()), 2e-3, "message gate vs fp64");
    // zero gate weights: f = m = 0, out = h
    std::vector<uint16_t> zero(size_t(H) * 64, 0);
    ref::msg_gate_row(h.data(), msg.data(), down.data(), zero.data(), zero.data(), ref::kRmsEps, out.data());
    CHECK(std::memcmp(out.data(), h.data(), sizeof(float) * H) == 0);
}

void test_softcap() {
    double worst = 0.0;
    for (int i = 0; i < 200000; ++i) {
        const float x = uniform(-400.0f, 400.0f);
        const double want = 30.0 * std::tanh(double(x) / 30.0);
        const float got = ref::softcap(x);
        const double ulp = std::ldexp(1.0, std::ilogb(std::fmax(std::fabs(want), 1e-30)) - 23);
        worst = std::fmax(worst, std::fabs(double(got) - want) / ulp);
    }
    check_le(worst, 4.0, "softcap vs fp64 (ulps)");
    CHECK(ref::softcap(INFINITY) == 30.0f);
    CHECK(ref::softcap(-INFINITY) == -30.0f);
    CHECK(ref::softcap(1e30f) == 30.0f);
    CHECK(ref::softcap(-0.0f) == 0.0f);  // ref::tanh(-0) is +0 (fma(-0 * 0, P, -0) with P < 0)
    CHECK(ref::softcap(0.0f) == 0.0f);
    CHECK(std::isnan(ref::softcap(NAN)));
}

}  // namespace

int main() {
    test_constants();
    test_norm();
    test_conv(false);
    test_conv(true);
    test_depth();
    test_msg();
    test_softcap();
    return cere::test::finish("test_inkling_mix_ref");
}
