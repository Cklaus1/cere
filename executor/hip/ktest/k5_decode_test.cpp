// AmdAmd PAR-6 step 2d: K5 GDN decode (executor/cuda/src/kernels/k5_gdn_decode.cu, hipified) on gfx1151, against
// its header contract with an FP64 reference that rests the state in BF16 between tokens, as the kernel does.
//  1. delta_rule_decode over 8 consecutive tokens: o and state rel-L2 vs the reference (limit 8e-3: BF16 o and
//     BF16 state at rest; the CUDA side's approx ex2/lg2/rsqrt become gfx11's hardware approximations here).
//  2. delta_rule_decode_rows, 3 chains of 2/3/1 rows in one launch: every chain's outputs, final state and undo
//     snapshots bit-equal to the same rows as sequential single-token launches (the header's promise).
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k5_gdn_decode.h"

using bf16 = __hip_bfloat16;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T))); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }

constexpr int KH = 16, VH = 48, D = 128;
struct Tok { std::vector<uint16_t> q, k, v, a, b; };

int main() {
    std::mt19937 g(5); std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    std::vector<float> A_log(VH); for (auto &x : A_log) x = std::log(0.5f + std::fabs(nd(g)));
    auto dt_bias = rnd(VH, 0.5f);
    auto state0 = rnd(size_t(VH) * D * D, 0.05f);
    const float scale = 1.f / std::sqrt(float(D));
    const int T = 8; std::vector<Tok> toks(T);
    for (auto &t : toks) { t.q = rnd(KH * D, 1.f); t.k = rnd(KH * D, 1.f); t.v = rnd(VH * D, 1.f); t.a = rnd(VH, 1.f); t.b = rnd(VH, 1.f); }

    // --- 1. sequential decode vs FP64 reference (state at rest in BF16)
    auto *dA = dev(A_log); auto *ddt = dev(dt_bias); auto *dstate = dev(state0);
    uint16_t *dq, *dk, *dv, *da, *db, *dout; HCK(hipMalloc(&dq, KH * D * 2)); HCK(hipMalloc(&dk, KH * D * 2)); HCK(hipMalloc(&dv, VH * D * 2));
    HCK(hipMalloc(&da, VH * 2)); HCK(hipMalloc(&db, VH * 2)); HCK(hipMalloc(&dout, VH * D * 2));
    std::vector<uint16_t> ref_state = state0;
    double worst_o = 0, worst_s = 0;
    for (int t = 0; t < T; ++t) {
        auto &tk = toks[t];
        HCK(hipMemcpy(dq, tk.q.data(), KH * D * 2, hipMemcpyHostToDevice)); HCK(hipMemcpy(dk, tk.k.data(), KH * D * 2, hipMemcpyHostToDevice));
        HCK(hipMemcpy(dv, tk.v.data(), VH * D * 2, hipMemcpyHostToDevice)); HCK(hipMemcpy(da, tk.a.data(), VH * 2, hipMemcpyHostToDevice)); HCK(hipMemcpy(db, tk.b.data(), VH * 2, hipMemcpyHostToDevice));
        cere::k5::DecodeArgs args{reinterpret_cast<bf16 *>(dq), reinterpret_cast<bf16 *>(dk), reinterpret_cast<bf16 *>(dv), reinterpret_cast<bf16 *>(da), reinterpret_cast<bf16 *>(db),
                                  dA, reinterpret_cast<bf16 *>(ddt), reinterpret_cast<bf16 *>(dstate), reinterpret_cast<bf16 *>(dout), scale};
        HCK(cere::k5::delta_rule_decode(args, nullptr));
        auto o = host(dout, VH * D); auto st = host(dstate, size_t(VH) * D * D);
        // reference
        std::vector<double> qn(KH * D), kn(KH * D);
        for (int h = 0; h < KH; ++h) {
            double sq = 0, sk = 0; for (int i = 0; i < D; ++i) { sq += double(bf2f(tk.q[h * D + i])) * bf2f(tk.q[h * D + i]); sk += double(bf2f(tk.k[h * D + i])) * bf2f(tk.k[h * D + i]); }
            for (int i = 0; i < D; ++i) { qn[h * D + i] = bf2f(tk.q[h * D + i]) / std::sqrt(sq + 1e-6) * scale; kn[h * D + i] = bf2f(tk.k[h * D + i]) / std::sqrt(sk + 1e-6); }
        }
        double no = 0, dn = 0, ns = 0, ds = 0;
        for (int h = 0; h < VH; ++h) {
            const int kh = h / 3; const double x = double(bf2f(tk.a[h])) + bf2f(dt_bias[h]);
            const double sp = x > 20 ? x : std::log1p(std::exp(x));
            const double decay = std::exp(-std::exp(double(A_log[h])) * sp), beta = 1 / (1 + std::exp(-double(bf2f(tk.b[h]))));
            std::vector<double> S(D * D);
            for (int i = 0; i < D * D; ++i) S[i] = bf2f(ref_state[size_t(h) * D * D + i]) * decay;
            for (int vi = 0; vi < D; ++vi) {
                double sk = 0; for (int ki = 0; ki < D; ++ki) sk += S[vi * D + ki] * kn[kh * D + ki];
                const double u = beta * (bf2f(tk.v[h * D + vi]) - sk);
                for (int ki = 0; ki < D; ++ki) S[vi * D + ki] += u * kn[kh * D + ki];
            }
            for (int vi = 0; vi < D; ++vi) {
                double ov = 0; for (int ki = 0; ki < D; ++ki) ov += S[vi * D + ki] * qn[kh * D + ki];
                const double got = bf2f(o[h * D + vi]); no += (got - ov) * (got - ov); dn += ov * ov;
            }
            for (int i = 0; i < D * D; ++i) {
                const double got = bf2f(st[size_t(h) * D * D + i]); ns += (got - S[i]) * (got - S[i]); ds += S[i] * S[i];
                ref_state[size_t(h) * D * D + i] = f2bf(float(S[i]));  // at rest in BF16
            }
        }
        worst_o = std::fmax(worst_o, std::sqrt(no / dn)); worst_s = std::fmax(worst_s, std::sqrt(ns / ds));
        ref_state = st;  // follow the kernel's own BF16 state, so each step measures one step's error
    }
    { char d[120]; std::snprintf(d, sizeof d, "8 tokens: worst o rel-L2 %.2e, state rel-L2 %.2e (limit 8e-3)", worst_o, worst_s);
      const bool ok = worst_o <= 8e-3 && worst_s <= 8e-3; if (!ok) ++g_fail; std::printf("%-30s %s  %s\n", "delta_rule_decode", ok ? "PASS" : "FAIL", d); }

    // --- 2. rows: 3 chains (2, 3, 1 rows) in one launch == sequential single-token launches
    const int chains = 3, rows_of[3] = {2, 3, 1}, R = 6;
    std::vector<uint16_t> Q, K, V, Aa, Bb;
    for (int r = 0; r < R; ++r) { auto &tk = toks[r]; Q.insert(Q.end(), tk.q.begin(), tk.q.end()); K.insert(K.end(), tk.k.begin(), tk.k.end());
        V.insert(V.end(), tk.v.begin(), tk.v.end()); Aa.insert(Aa.end(), tk.a.begin(), tk.a.end()); Bb.insert(Bb.end(), tk.b.begin(), tk.b.end()); }
    // q, k, v share one row stride in the API (qkv_stride): pack each row as [q | k | v]
    const int64_t qkv_stride = KH * D * 2 + VH * D; std::vector<uint16_t> QKV(size_t(R) * qkv_stride);
    for (int r = 0; r < R; ++r) { std::memcpy(&QKV[r * qkv_stride], &Q[r * KH * D], KH * D * 2); std::memcpy(&QKV[r * qkv_stride + KH * D], &K[r * KH * D], KH * D * 2); std::memcpy(&QKV[r * qkv_stride + 2 * KH * D], &V[r * VH * D], VH * D * 2); }
    auto *dqkv = dev(QKV); auto *dAa = dev(Aa); auto *dBb = dev(Bb); uint16_t *dO; HCK(hipMalloc(&dO, size_t(R) * VH * D * 2));
    std::vector<uint16_t *> st(chains), saves(R, nullptr);
    for (int c = 0; c < chains; ++c) st[c] = dev(state0);
    for (int r = 0; r < R; ++r) HCK(hipMalloc(&saves[r], size_t(VH) * D * D * 2));
    cere::k5::DecodeRowsArgs ra{};
    ra.step = {reinterpret_cast<bf16 *>(dqkv), reinterpret_cast<bf16 *>(dqkv + KH * D), reinterpret_cast<bf16 *>(dqkv + 2 * KH * D), reinterpret_cast<bf16 *>(dAa), reinterpret_cast<bf16 *>(dBb),
               dA, reinterpret_cast<bf16 *>(ddt), nullptr, reinterpret_cast<bf16 *>(dO), scale};
    ra.chains = chains; ra.qkv_stride = qkv_stride; ra.ab_stride = VH; ra.o_stride = VH * D;
    for (int c = 0, f = 0; c < chains; ++c) { ra.first[c] = f; ra.rows[c] = rows_of[c]; ra.states[c] = reinterpret_cast<bf16 *>(st[c]); f += rows_of[c]; }
    for (int r = 0; r < R; ++r) ra.saves[r] = reinterpret_cast<bf16 *>(saves[r]);
    HCK(cere::k5::delta_rule_decode_rows(ra, nullptr));
    auto O = host(dO, size_t(R) * VH * D);
    bool equal = true;
    for (int c = 0, f = 0; c < chains; ++c) {
        auto *s1 = dev(state0);
        for (int i = 0; i < rows_of[c]; ++i) {
            const int r = f + i;
            cere::k5::DecodeArgs one{reinterpret_cast<bf16 *>(dqkv + r * qkv_stride), reinterpret_cast<bf16 *>(dqkv + r * qkv_stride + KH * D), reinterpret_cast<bf16 *>(dqkv + r * qkv_stride + 2 * KH * D),
                                     reinterpret_cast<bf16 *>(dAa + r * VH), reinterpret_cast<bf16 *>(dBb + r * VH), dA, reinterpret_cast<bf16 *>(ddt), reinterpret_cast<bf16 *>(s1), reinterpret_cast<bf16 *>(dout), scale};
            HCK(cere::k5::delta_rule_decode(one, nullptr));
            auto o1 = host(dout, VH * D);
            equal = equal && std::memcmp(o1.data(), &O[size_t(r) * VH * D], VH * D * 2) == 0;
            auto s_now = host(s1, size_t(VH) * D * D);
            auto s_kernel = host(i + 1 < rows_of[c] ? saves[r] : st[c], size_t(VH) * D * D);
            equal = equal && s_now == s_kernel;
        }
        f += rows_of[c];
    }
    if (!equal) ++g_fail;
    std::printf("%-30s %s  3 chains (2/3/1 rows): outputs, undo saves and final states %s\n", "delta_rule_decode_rows", equal ? "PASS" : "FAIL", equal ? "bit-equal to sequential launches" : "DIFFER");
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
