// Inkling 2 Mini block drafter (inkling design note S9, G1): executor/ref/inkling/drafter_ref.h against
// independent double-precision formulas (each output within 1 BF16 ulp, attention within 2), plus the structural
// rules the kernels must keep: the block-anchored window on the local layers, every block row seeing the same keys,
// the span/group folds beyond 4,096 keys, the first-index argmax and the host fold.
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../../ref/inkling/drafter_ref.h"
#include "check.h"

namespace {

namespace d = ref::drafter;

uint32_t rng_state = 0x9e3779b9u;
uint32_t next_u32() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
float uniform(float lo, float hi) { return lo + (hi - lo) * float(next_u32() >> 8) * (1.0f / 16777216.0f); }
uint16_t rbf(float lo, float hi) { return ref::encode_bf16(uniform(lo, hi)); }
std::vector<uint16_t> rbf_vec(size_t n, float lo, float hi) {
    std::vector<uint16_t> v(n);
    for (auto &x : v) x = rbf(lo, hi);
    return v;
}
double f(uint16_t v) { return double(ref::decode_bf16(v)); }

// |got - want| in BF16 ulps of want (ulp floor at 2^-20 so values near zero compare absolutely).
double bf16_ulps(uint16_t got, double want) {
    const double ulp = std::ldexp(1.0, std::max(std::ilogb(std::fabs(want) > 0 ? want : 1.0), -20) - 7);
    return std::fabs(f(got) - want) / ulp;
}

void test_fuse_alpha() {
    uint16_t w[16];
    float a[16];
    for (auto &x : w) x = ref::encode_bf16(0.0f);
    d::fuse_alpha(w, a);
    for (float x : a) CHECK(x == 0.25f);
    for (auto &x : w) x = rbf(-3.0f, 3.0f);
    d::fuse_alpha(w, a);
    for (int r = 0; r < 4; ++r) {
        double m = -1e30, s = 0.0;
        for (int t = 0; t < 4; ++t) m = std::max(m, f(w[r * 4 + t]));
        for (int t = 0; t < 4; ++t) s += std::exp(f(w[r * 4 + t]) - m);
        for (int t = 0; t < 4; ++t) {
            const double want = std::exp(f(w[r * 4 + t]) - m) / s;
            CHECK(bf16_ulps(ref::encode_bf16(a[r * 4 + t]), want) <= 1.0);
            CHECK(a[r * 4 + t] == d::bf16r(a[r * 4 + t]));  // BF16 values (SGLang keeps alpha in BF16)
        }
    }
}

void test_mix_rmsnorm() {
    const int64_t stride = d::kHidden + 7;
    auto taps = rbf_vec(size_t(4 * stride), -4.0f, 4.0f);
    auto w = rbf_vec(d::kHidden, 0.5f, 1.5f);
    const float a[4] = {0.125f, 0.25f, 0.5f, 0.125f};
    std::vector<uint16_t> y(d::kHidden);
    d::mix_rmsnorm(taps.data(), stride, a, w.data(), y.data());
    std::vector<double> m(d::kHidden);
    double ss = 0.0;
    for (int i = 0; i < d::kHidden; ++i) {
        for (int t = 0; t < 4; ++t) m[i] += f(taps[t * stride + i]) * a[t];
        ss += m[i] * m[i];
    }
    const double r = 1.0 / std::sqrt(ss / d::kHidden + 1e-6);
    double worst = 0.0;
    for (int i = 0; i < d::kHidden; ++i) worst = std::max(worst, bf16_ulps(y[i], f(w[i]) * m[i] * r));
    CHECK(worst <= 1.0);
}

void test_rope() {
    float c[d::kHalf], s[d::kHalf];
    d::rope_cos_sin(0, c, s);
    for (int i = 0; i < d::kHalf; ++i) CHECK(c[i] == 1.0f && s[i] == 0.0f);
    const int64_t p = 123457;
    d::rope_cos_sin(p, c, s);
    CHECK(c[0] == float(std::cos(double(p))));  // frequency 0 is theta^0 = 1
    auto x = rbf_vec(d::kHeadDim, -2.0f, 2.0f);
    auto wn = rbf_vec(d::kHeadDim, 0.5f, 1.5f);
    uint16_t out[d::kHeadDim], out_ctx[d::kHeadDim];
    d::head_norm_rope(x.data(), wn.data(), c, s, false, out);
    d::head_norm_rope(x.data(), wn.data(), c, s, true, out_ctx);
    double ss = 0.0;
    for (int i = 0; i < d::kHeadDim; ++i) ss += f(x[i]) * f(x[i]);
    const double r = 1.0 / std::sqrt(ss / d::kHeadDim + 1e-6);
    double worst = 0.0;
    for (int i = 0; i < d::kHalf; ++i) {
        const double fr = std::pow(1e6, -2.0 * i / d::kHeadDim), ang = double(p) * fr;
        const double a = f(wn[i]) * f(x[i]) * r, b = f(wn[i + d::kHalf]) * f(x[i + d::kHalf]) * r;
        const double lo = a * std::cos(ang) - b * std::sin(ang), hi = b * std::cos(ang) + a * std::sin(ang);
        // rope's cancellation: compare against the vector's scale, not the element.
        const double scale = std::max(std::fabs(a), std::fabs(b));
        worst = std::max(worst, std::fabs(f(out[i]) - lo) / (scale * 0x1p-7));
        worst = std::max(worst, std::fabs(f(out[i + d::kHalf]) - hi) / (scale * 0x1p-7));
    }
    CHECK(worst <= 1.0);
    int differ = 0;
    for (int i = 0; i < d::kHeadDim; ++i) differ += out[i] != out_ctx[i];
    CHECK(differ > 0);  // the context path's extra BF16 rounding is observable
}

// Double attention of one row/head over keys [lo, L + 8).
void attend_double(const d::BlockKeys &k, int64_t lo, const uint16_t *q, int h, double *out) {
    const int g = h / d::kGroup;
    const int64_t hi = k.L + d::kBlock;
    std::vector<double> sc(size_t(hi - lo));
    double m = -1e300;
    for (int64_t j = lo; j < hi; ++j) {
        const uint16_t *kr = k.at(k.ctx_k, k.blk_k, j, g);
        double dot = 0.0;
        for (int e = 0; e < d::kHeadDim; ++e) dot += f(q[e]) * f(kr[e]);
        sc[size_t(j - lo)] = dot * double(d::kScale);
        m = std::max(m, sc[size_t(j - lo)]);
    }
    double l = 0.0;
    for (int e = 0; e < d::kHeadDim; ++e) out[e] = 0.0;
    for (int64_t j = lo; j < hi; ++j) {
        const double p = std::exp(sc[size_t(j - lo)] - m);
        l += p;
        const uint16_t *vr = k.at(k.ctx_v, k.blk_v, j, g);
        for (int e = 0; e < d::kHeadDim; ++e) out[e] += p * f(vr[e]);
    }
    for (int e = 0; e < d::kHeadDim; ++e) out[e] /= l;
}

void test_attention() {
    const int64_t cases[] = {0, 1, 57, 1016, 1024, 1025, 3000, 4100, 9000};
    for (int64_t L : cases) {
        auto ctx_k = rbf_vec(size_t((L + 1) * d::kKvRow), -1.0f, 1.0f);
        auto ctx_v = rbf_vec(size_t((L + 1) * d::kKvRow), -1.0f, 1.0f);
        auto blk_k = rbf_vec(d::kBlock * d::kKvRow, -1.0f, 1.0f);
        auto blk_v = rbf_vec(d::kBlock * d::kKvRow, -1.0f, 1.0f);
        auto q = rbf_vec(d::kQRow, -2.0f, 2.0f);
        const d::BlockKeys keys{L, ctx_k.data(), ctx_v.data(), blk_k.data(), blk_v.data(), 0};
        for (int layer = 0; layer < 2; ++layer) {
            const int64_t lo = d::key_lo(layer, L);
            CHECK_EQ(lo, layer == 0 && L > 1024 ? L - 1024 : 0);
            for (int h : {0, 5, 23}) {
                uint16_t out[d::kHeadDim];
                double want[d::kHeadDim];
                d::attend(keys, layer, q.data() + h * d::kHeadDim, h, out);
                attend_double(keys, lo, q.data() + h * d::kHeadDim, h, want);
                double worst = 0.0;
                for (int e = 0; e < d::kHeadDim; ++e) worst = std::max(worst, bf16_ulps(out[e], want[e]));
                CHECK(worst <= 2.0);
            }
        }
        if (L > 1024) {
            // Local layer: the key just below the window is ignored; the full layer reads it.
            uint16_t a[d::kHeadDim], b[d::kHeadDim], c[d::kHeadDim], e2[d::kHeadDim];
            d::attend(keys, 2, q.data(), 0, a);
            d::attend(keys, 1, q.data(), 0, c);
            const int64_t j = L - 1025;
            for (int e = 0; e < d::kHeadDim; ++e) ctx_v[size_t(j * d::kKvRow + e)] = ref::encode_bf16(64.0f);
            d::attend(keys, 2, q.data(), 0, b);
            d::attend(keys, 1, q.data(), 0, e2);
            int same_local = 0, same_full = 0;
            for (int e = 0; e < d::kHeadDim; ++e) {
                same_local += a[e] == b[e];
                same_full += c[e] == e2[e];
            }
            CHECK_EQ(same_local, d::kHeadDim);
            CHECK(same_full < d::kHeadDim);
            // A ring of 1,024 slots holding the same rows gives the same local-layer output.
            std::vector<uint16_t> rk(size_t(1024 * d::kKvRow)), rv(rk.size());
            for (int64_t p = L - 1024; p < L; ++p)
                for (int e = 0; e < d::kKvRow; ++e) {
                    rk[size_t((p % 1024) * d::kKvRow + e)] = ctx_k[size_t(p * d::kKvRow + e)];
                    rv[size_t((p % 1024) * d::kKvRow + e)] = ctx_v[size_t(p * d::kKvRow + e)];
                }
            const d::BlockKeys ring{L, rk.data(), rv.data(), blk_k.data(), blk_v.data(), 1024};
            uint16_t r[d::kHeadDim];
            d::attend(ring, 2, q.data(), 0, r);
            int same = 0;
            for (int e = 0; e < d::kHeadDim; ++e) same += r[e] == b[e];
            CHECK_EQ(same, d::kHeadDim);
        }
    }
}

void test_markov() {
    auto proj = rbf_vec(d::kProj, -4.0f, 4.0f);
    auto state = rbf_vec(d::kRank, -1.0f, 1.0f);
    const auto state0 = state;
    std::vector<uint16_t> latent(d::kRank);
    d::markov_gates(proj.data(), state.data(), latent.data());
    double worst = 0.0;
    for (int i = 0; i < d::kRank; ++i) {
        const double g = 1.0 / (1.0 + std::exp(-f(proj[i])));
        const double c = std::tanh(f(proj[d::kRank + i]));
        const double kept = g * f(state0[i]), blend = (1.0 - g) * c;
        // Six BF16 roundings (SGLang's BF16 elementwise ops) on two terms that may cancel; bf16(g) also enters
        // 1 - g, an absolute error of |c| ulp(g) when g is near 1: 3 ulps of the largest of the three.
        const double scale = std::max(std::max(std::fabs(kept), std::fabs(blend)), std::fabs(c)) + 1e-30;
        worst = std::max(worst, std::fabs(f(state[i]) - (kept + blend)) / (scale * 0x1p-7));
        CHECK(bf16_ulps(latent[i], std::tanh(f(proj[2 * d::kRank + i]))) <= 1.0);
    }
    CHECK(worst <= 3.0);
}

void test_step_and_fold() {
    // Fold of one row against a direct double sum.
    auto lm = rbf_vec(d::kHidden, -0.1f, 0.1f);
    auto w = rbf_vec(size_t(d::kHidden) * d::kRank, -0.1f, 0.1f);
    auto b = rbf_vec(d::kHidden, -0.1f, 0.1f);
    uint16_t fw[d::kRank], fb;
    d::fold_row(lm.data(), w.data(), b.data(), fw, &fb);
    for (int j : {0, 77, 255}) {
        double s = 0.0;
        for (int k = 0; k < d::kHidden; ++k) s += f(lm[k]) * f(w[size_t(k) * d::kRank + j]);
        CHECK(bf16_ulps(fw[j], s) <= 0.5 + 1e-9);
    }

    // Step logits near softcap(base + bias); argmax takes the first of equal maxima.
    const size_t V = size_t(d::kVocab);
    std::vector<uint16_t> fold_w(V * d::kRank, ref::encode_bf16(0.0f)), fold_b(V), base(V), logits(V);
    auto latent = rbf_vec(d::kRank, -1.0f, 1.0f);
    for (size_t v = 0; v < V; ++v) {
        fold_b[v] = rbf(-1.0f, 1.0f);
        base[v] = rbf(-20.0f, 5.0f);
    }
    for (size_t v = 0; v < 64; ++v)
        for (int j = 0; j < d::kRank; ++j) fold_w[v * d::kRank + size_t(j)] = rbf(-0.5f, 0.5f);
    double worst = 0.0;
    for (size_t v = 0; v < 64; ++v) {
        double dot = 0.0;
        for (int j = 0; j < d::kRank; ++j) dot += f(latent[size_t(j)]) * f(fold_w[v * d::kRank + size_t(j)]);
        const double x = f(base[v]) + dot + f(fold_b[v]);
        const uint16_t got = d::step_logit(latent.data(), fold_w.data() + v * d::kRank, fold_b[v], base[v]);
        // bias and base + bias are BF16-rounded before the softcap (SGLang's BF16 step dtype), the sum may cancel:
        // 2 ulps of the largest operand (tanh is 1-Lipschitz) plus the output rounding.
        const double scale = std::max({std::fabs(f(base[v])), std::fabs(dot + f(fold_b[v])), std::fabs(x)});
        worst = std::max(worst, std::fabs(f(got) - 30.0 * std::tanh(x / 30.0)) / (scale * 0x1p-7));
    }
    CHECK(worst <= 2.0);
    for (size_t v : {size_t(1000), size_t(150000), V - 1}) base[v] = ref::encode_bf16(40.0f);
    fold_b[1000] = fold_b[150000] = fold_b[V - 1] = ref::encode_bf16(0.0f);
    const uint32_t best = d::markov_step(latent.data(), fold_w.data(), fold_b.data(), base.data(), logits.data());
    CHECK_EQ(best, 1000);
    CHECK(logits[1000] == logits[150000] && logits[1000] == logits[V - 1]);
    base[V - 1] = ref::encode_bf16(80.0f);
    CHECK_EQ(d::markov_step(latent.data(), fold_w.data(), fold_b.data(), base.data(), nullptr), uint32_t(V - 1));
}

}  // namespace

int main() {
    test_fuse_alpha();
    test_mix_rmsnorm();
    test_rope();
    test_attention();
    test_markov();
    test_step_and_fold();
    return cere::test::finish("test_inkling_drafter_ref");
}
