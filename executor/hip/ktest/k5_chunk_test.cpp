// AmdAmd PAR-6 step 3a: K5 chunked prefill (executor/hip/src/k5_gdn_chunk_hip.cpp, the k5_gdn_chunk.h contract) on
// gfx1151. Inputs: q, k, v N(0,1) BF16, g in [-0.3, -0.005], beta in (0, 1), a random BF16 start state, 200 tokens
// (3 full chunks + a partial 8-row chunk). Checks:
//  1. vs an FP64 mirror of the header's chunk math with its BF16 rounding points (from the kernel's own normalised
//     q, k): o and final state rel-L2 (limit 1e-3: the FP32 vs FP64 accumulation flips some BF16 roundings);
//  2. vs the exact FP64 gated delta rule token by token, no rounding at all (limit 2e-2: the chunk form's BF16 points);
//  3. determinism: the same call twice, bit-equal o and state;
//  4. continuity: capture32 at chunk 2 of the 200-row call, then a 72-row call from row 128 with state_in32 = that
//     capture: its o and final state bit-equal to the longer call's rows 128.. and final state; the split (_hi) form
//     of state_in32 / capture32 gives the same bytes.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k5_gdn.h"
#include "kernels/k5_gdn_chunk.h"

namespace k5 = cere::k5;
using bf16 = __hip_bfloat16;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static double rb(double x) { return bf2f(f2bf(float(x))); }
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
static double rel(const std::vector<double> &ref, const std::vector<uint16_t> &got) {
    double n = 0, d = 0; for (size_t i = 0; i < ref.size(); ++i) { const double e = bf2f(got[i]) - ref[i]; n += e * e; d += ref[i] * ref[i]; }
    return std::sqrt(n / d);
}
static void check(const char *name, bool ok, const char *fmt, double a = 0, double b = 0) {
    std::printf("%-26s %s  ", name, ok ? "PASS" : "FAIL"); std::printf(fmt, a, b); std::printf("\n"); if (!ok) ++g_fail;
}

constexpr int KH = 16, VH = 48, D = 128, BT = 64;

int main() {
    std::mt19937 g(7); std::normal_distribution<float> nd(0.f, 1.f); std::uniform_real_distribution<float> ud(0.f, 1.f);
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    const int64_t T = 200; const float scale = 1.f / std::sqrt(float(D));
    auto q = rnd(T * KH * D, 1.f), k = rnd(T * KH * D, 1.f), v = rnd(T * VH * D, 1.f), state0 = rnd(size_t(VH) * D * D, 0.05f);
    std::vector<float> gg(T * VH), beta(T * VH);
    for (auto &x : gg) x = -0.005f - 0.295f * ud(g);
    for (auto &x : beta) x = 0.02f + 0.96f * ud(g);
    auto *dq = dev(q), *dk = dev(k), *dv = dev(v); auto *dg = dev(gg), *db = dev(beta);
    uint16_t *dst, *dout; HCK(hipMalloc(&dst, state0.size() * 2)); HCK(hipMalloc(&dout, size_t(T) * VH * D * 2));
    void *ws; HCK(hipMalloc(&ws, k5::chunk_workspace_bytes(T)));
    auto call = [&](int64_t r0, int64_t n, const float *in32, float *cap, int64_t cap_chunk, const float *in32_hi = nullptr, float *cap_hi = nullptr) {
        k5::ChunkArgs a{};
        a.q = reinterpret_cast<const bf16 *>(dq) + r0 * KH * D; a.k = reinterpret_cast<const bf16 *>(dk) + r0 * KH * D;
        a.v = reinterpret_cast<const bf16 *>(dv) + r0 * VH * D; a.g = dg + r0 * VH; a.beta = db + r0 * VH;
        a.state = reinterpret_cast<bf16 *>(dst); a.o = reinterpret_cast<bf16 *>(dout) + r0 * VH * D; a.tokens = n; a.scale = scale;
        a.state_in32 = in32; a.capture32 = cap; a.capture_chunk = cap_chunk; a.state_in32_hi = in32_hi; a.capture32_hi = cap_hi;
        if (hipError_t e = k5::delta_rule_chunk(a, ws, nullptr); e != hipSuccess) { std::printf("chunk failed: %s\n", hipGetErrorString(e)); std::exit(2); }
    };
    const size_t SS = size_t(VH) * D * D; float *cap; HCK(hipMalloc(&cap, SS * 4));
    HCK(hipMemcpy(dst, state0.data(), SS * 2, hipMemcpyHostToDevice));
    call(0, T, nullptr, cap, 2);
    auto o1 = host(dout, size_t(T) * VH * D), s1 = host(dst, SS);
    // the kernel's own normalised q, k (k5::l2norm, as the chunk path calls it)
    uint16_t *dqn, *dkn; HCK(hipMalloc(&dqn, q.size() * 2)); HCK(hipMalloc(&dkn, k.size() * 2));
    HCK(k5::l2norm(reinterpret_cast<const bf16 *>(dq), reinterpret_cast<bf16 *>(dqn), T * KH, D, 1e-6f, nullptr));
    HCK(k5::l2norm(reinterpret_cast<const bf16 *>(dk), reinterpret_cast<bf16 *>(dkn), T * KH, D, 1e-6f, nullptr));
    auto qn = host(dqn, q.size()), kn = host(dkn, k.size());

    // --- 1. FP64 mirror of the chunk math with the header's rounding points
    std::vector<double> om(size_t(T) * VH * D), sm(SS), oe(size_t(T) * VH * D), se(SS);
    for (int vh = 0; vh < VH; ++vh) {
        const int kh = vh / 3;
        auto Q = [&](int64_t t, int c) { return double(bf2f(qn[(t * KH + kh) * D + c])); };
        auto K = [&](int64_t t, int c) { return double(bf2f(kn[(t * KH + kh) * D + c])); };
        auto V = [&](int64_t t, int c) { return double(bf2f(v[(t * VH + vh) * D + c])); };
        std::vector<double> h(D * D); for (int i = 0; i < D * D; ++i) h[i] = bf2f(state0[size_t(vh) * D * D + i]);
        for (int64_t r0 = 0; r0 < T; r0 += BT) {
            const int n = int(std::min<int64_t>(BT, T - r0));
            std::vector<double> G(n), B(n); double acc = 0;
            for (int i = 0; i < n; ++i) { acc += gg[(r0 + i) * VH + vh]; G[i] = acc; B[i] = beta[(r0 + i) * VH + vh]; }
            std::vector<double> L(n * n, 0), A(n * n, 0);
            for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) { double s = 0; for (int c = 0; c < D; ++c) s += K(r0 + i, c) * K(r0 + j, c); L[i * n + j] = B[i] * s * std::exp(G[i] - G[j]); }
            for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) { double x = i == j; if (i > j) for (int m = j; m < i; ++m) x -= L[i * n + m] * A[m * n + j]; A[i * n + j] = x; }
            for (auto &x : A) x = rb(x);
            std::vector<double> u(n * D), w(n * D), d(n * D), vn(n * D), hc(D * D);
            for (int i = 0; i < D * D; ++i) hc[i] = rb(h[i]);
            for (int i = 0; i < n; ++i) for (int c = 0; c < D; ++c) {
                double su = 0, sw = 0;
                for (int m = 0; m <= i; ++m) { su += A[i * n + m] * rb(V(r0 + m, c) * B[m]); sw += A[i * n + m] * rb(K(r0 + m, c) * B[m] * std::exp(G[m])); }
                u[i * D + c] = rb(su); w[i * D + c] = rb(sw);
            }
            for (int i = 0; i < n; ++i) for (int cv = 0; cv < D; ++cv) { double s = 0; for (int ck = 0; ck < D; ++ck) s += w[i * D + ck] * hc[cv * D + ck]; d[i * D + cv] = u[i * D + cv] - s; vn[i * D + cv] = rb(d[i * D + cv]); }
            std::vector<double> P(n * n, 0);
            for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) { double s = 0; for (int c = 0; c < D; ++c) s += Q(r0 + i, c) * K(r0 + j, c); P[i * n + j] = rb(s * std::exp(G[i] - G[j])); }
            for (int i = 0; i < n; ++i) for (int cv = 0; cv < D; ++cv) {
                double sq = 0, sp = 0;
                for (int ck = 0; ck < D; ++ck) sq += Q(r0 + i, ck) * hc[cv * D + ck];
                for (int m = 0; m <= i; ++m) sp += P[i * n + m] * vn[m * D + cv];
                om[((r0 + i) * VH + vh) * D + cv] = rb(sq * std::exp(G[i]) * scale + sp * scale);
            }
            const double gl = G[n - 1];
            for (int cv = 0; cv < D; ++cv) for (int ck = 0; ck < D; ++ck) {
                double s = 0; for (int t = 0; t < n; ++t) s += K(r0 + t, ck) * rb(d[t * D + cv] * std::exp(gl - G[t]));
                h[cv * D + ck] = h[cv * D + ck] * std::exp(gl) + s;
            }
        }
        for (int i = 0; i < D * D; ++i) sm[size_t(vh) * D * D + i] = rb(h[i]);
        // --- 2. exact FP64 gated delta rule, token by token (q, k normalised in FP64 from the raw inputs)
        std::vector<double> S(D * D); for (int i = 0; i < D * D; ++i) S[i] = bf2f(state0[size_t(vh) * D * D + i]);
        for (int64_t t = 0; t < T; ++t) {
            double qv[D], kv[D], nq = 0, nk = 0;
            for (int c = 0; c < D; ++c) { qv[c] = bf2f(q[(t * KH + kh) * D + c]); kv[c] = bf2f(k[(t * KH + kh) * D + c]); nq += qv[c] * qv[c]; nk += kv[c] * kv[c]; }
            for (int c = 0; c < D; ++c) { qv[c] /= std::sqrt(nq + 1e-6); kv[c] /= std::sqrt(nk + 1e-6); }
            const double decay = std::exp(double(gg[t * VH + vh])), bt = beta[t * VH + vh];
            for (int cv = 0; cv < D; ++cv) {
                double sk = 0; for (int ck = 0; ck < D; ++ck) { S[cv * D + ck] *= decay; sk += S[cv * D + ck] * kv[ck]; }
                const double dl = (V(t, cv) - sk) * bt; double so = 0;
                for (int ck = 0; ck < D; ++ck) { S[cv * D + ck] += dl * kv[ck]; so += S[cv * D + ck] * qv[ck]; }
                oe[(t * VH + vh) * D + cv] = so * scale;
            }
        }
        for (int i = 0; i < D * D; ++i) se[size_t(vh) * D * D + i] = S[i];
    }
    const double eo = rel(om, o1), es = rel(sm, s1), xo = rel(oe, o1), xs = rel(se, s1);
    check("vs FP64 chunk mirror", eo <= 1e-3 && es <= 1e-3, "o rel-L2 %.2e, final state %.2e (limit 1e-3)", eo, es);
    check("vs exact delta rule", xo <= 2e-2 && xs <= 2e-2, "o rel-L2 %.2e, final state %.2e (limit 2e-2)", xo, xs);
    // --- 3. determinism
    HCK(hipMemcpy(dst, state0.data(), SS * 2, hipMemcpyHostToDevice)); call(0, T, nullptr, nullptr, -1);
    const bool det = host(dout, o1.size()) == o1 && host(dst, SS) == s1;
    check("determinism", det, det ? "two identical calls: o and state bit-equal" : "two identical calls DIFFER");
    // --- 4. continuity from the FP32 capture at chunk 2 (row 128), base and split (_hi) forms
    HCK(hipMemset(dout, 0, o1.size() * 2)); call(128, T - 128, cap, nullptr, -1);
    auto o2 = host(dout, o1.size()), s2 = host(dst, SS);
    bool cont = s2 == s1; for (size_t i = 128 * VH * D; i < o1.size(); ++i) cont &= o2[i] == o1[i];
    check("continuity (state_in32)", cont, cont ? "rows 128..199 from the chunk-2 capture == the 200-row call, bit for bit" : "rows 128.. DIFFER");
    float *lo, *hi; HCK(hipMalloc(&lo, SS * 2)); HCK(hipMalloc(&hi, SS * 2));  // heads 0..23 / 24..47
    HCK(hipMemcpy(dst, state0.data(), SS * 2, hipMemcpyHostToDevice)); call(0, T, nullptr, lo, 2, nullptr, hi);
    auto lo_h = host(lo, SS / 2), hi_h = host(hi, SS / 2), cap_h = host(cap, SS);
    bool split = std::memcmp(lo_h.data(), cap_h.data(), SS / 2 * 4) == 0 && std::memcmp(hi_h.data(), cap_h.data() + SS / 2, SS / 2 * 4) == 0;
    HCK(hipMemset(dout, 0, o1.size() * 2)); call(128, T - 128, lo, nullptr, -1, hi, nullptr);
    auto o3 = host(dout, o1.size()); split &= host(dst, SS) == s1; for (size_t i = 128 * VH * D; i < o1.size(); ++i) split &= o3[i] == o1[i];
    check("continuity (_hi split)", split, split ? "capture32_hi / state_in32_hi: same capture, same rows and state" : "split form DIFFERS");
    {   // timing: one 4,096-row call (the largest the contract allows), mean of 5 after a warm-up
        const int64_t TB = 4096; std::vector<uint16_t> big(TB * VH * D, f2bf(0.5f)); std::vector<float> gb(TB * VH, -0.05f), bb(TB * VH, 0.5f);
        k5::ChunkArgs a{}; a.q = reinterpret_cast<const bf16 *>(dev(big)); a.k = a.q; a.v = a.q; a.g = dev(gb); a.beta = dev(bb);
        a.state = reinterpret_cast<bf16 *>(dst); a.tokens = TB; a.scale = scale; uint16_t *bo; HCK(hipMalloc(&bo, big.size() * 2)); a.o = reinterpret_cast<bf16 *>(bo);
        void *bws; HCK(hipMalloc(&bws, k5::chunk_workspace_bytes(TB))); hipEvent_t e0, e1; HCK(hipEventCreate(&e0)); HCK(hipEventCreate(&e1));
        HCK(k5::delta_rule_chunk(a, bws, nullptr)); HCK(hipEventRecord(e0, nullptr));
        for (int i = 0; i < 5; ++i) HCK(k5::delta_rule_chunk(a, bws, nullptr));
        HCK(hipEventRecord(e1, nullptr)); HCK(hipEventSynchronize(e1)); float ms; HCK(hipEventElapsedTime(&ms, e0, e1));
        std::printf("%-26s      4,096 rows x 48 heads: %.2f ms per call (one GDN layer's chunked prefill)\n", "timing", ms / 5);
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
