// VulkanAmd P2: executor/hip/ktest/k5_chunk_test.cpp's contract rows on the Vulkan K5 chunked prefill
// (shaders/k5_gdn_chunk.comp, src/kernels/k5_gdn_chunk_vk.*), same seed, data and bounds as the HIP test.
// Inputs: q, k, v N(0,1) BF16, g in [-0.3, -0.005], beta in (0, 1), a random BF16 start state, 200 tokens (3 full chunks
// + a partial 8-row chunk). Rows:
//  1. vs an FP64 mirror of the header's chunk math with its BF16 rounding points (from the kernel's own normalised q, k:
//     vkk5::l2norm, as the chunk path calls it): o and final state rel-L2 <= 1e-3;
//  2. vs the exact FP64 gated delta rule token by token, no rounding at all: <= 2e-2;
//  3. determinism: the same call twice, bit-equal o and state;
//  4. continuity: capture32 at chunk 2 of the 200-row call, then a 72-row call from row 128 with state_in32 = that
//     capture: o rows 128.. and final state bit-equal to the longer call's; the split (_hi) form gives the same bytes.
// Extra rows (not in the HIP test): argument refusals. Then timing (GPU timestamps): one call of 4,096 rows (the HIP
// test's data: 0.5 everywhere, g -0.05, beta 0.5), median of 7 after a warm-up, and of 1,205 and 512 rows; also the HIP test's
// method (mean of 5 back-to-back calls). The first call's o, final state and capture are printed as FNV-1a digests
// ("bits" line), so shader variants can be compared for bit identity across runs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "launch/vk_launch_testing.h"
#include "kernels/k5_gdn_chunk_vk.h"
#include "kernels/k5_gdn_vk.h"
#include "ktest_util.h"

using kt::bf2f;
using kt::f2bf;
namespace k5 = cere::vkk5;
static int g_fail = 0;
static double rb(double x) { return bf2f(f2bf(float(x))); }
static double rel(const std::vector<double> &ref, const std::vector<uint16_t> &got) {
    double n = 0, d = 0; for (size_t i = 0; i < ref.size(); ++i) { const double e = bf2f(got[i]) - ref[i]; n += e * e; d += ref[i] * ref[i]; }
    return std::sqrt(n / d);
}
static void check(const char *name, bool ok, const char *fmt, double a = 0, double b = 0) {
    std::printf("%-26s %s  ", name, ok ? "PASS" : "FAIL"); std::printf(fmt, a, b); std::printf("\n"); if (!ok) ++g_fail;
}

constexpr int KH = 16, VH = 48, D = 128, BT = 64;

int main(int argc, char **argv) {
    // VulkanAutotune hooks (ktest_util.h): [--identity DIR] (the 200-row call's o / state, each timed shape's o)
    // [--samples] (every timed shape's samples, >= 15, and also 256 rows); no --tune knob (NV / RIF are SPIR-V defines).
    const char *spv = "build/vulkan/spv";
    for (int i = 1; i < argc; ++i) { if (!kt::hook_arg(argc, argv, i)) spv = argv[i]; }
    kt::tune_keys({});
    auto ctx_owner = cere::vk::testing::create_or_die(spv);
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str());
    std::mt19937 g(7); std::normal_distribution<float> nd(0.f, 1.f); std::uniform_real_distribution<float> ud(0.f, 1.f);
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    const int64_t T = 200; const float scale = 1.f / std::sqrt(float(D));
    auto q = rnd(T * KH * D, 1.f), k = rnd(T * KH * D, 1.f), v = rnd(T * VH * D, 1.f), state0 = rnd(size_t(VH) * D * D, 0.05f);
    std::vector<float> gg(T * VH), beta(T * VH);
    for (auto &x : gg) x = -0.005f - 0.295f * ud(g);
    for (auto &x : beta) x = 0.02f + 0.96f * ud(g);
    auto dq = kt::upload(ctx, q), dk = kt::upload(ctx, k), dv = kt::upload(ctx, v), dg = kt::upload(ctx, gg), db = kt::upload(ctx, beta);
    const size_t SS = size_t(VH) * D * D;
    auto dst = kt::make(ctx, SS * 2), dout = kt::make(ctx, size_t(T) * VH * D * 2);
    auto ws = kt::make(ctx, k5::chunk_workspace_bytes(T));
    auto cap = kt::make(ctx, SS * 4);
    auto run = [&](int64_t r0, int64_t n, uint64_t in32, uint64_t capa, int64_t cap_chunk, uint64_t in32_hi = 0, uint64_t cap_hi = 0) {
        k5::ChunkArgs a{};
        a.q = dq.addr + uint64_t(r0) * KH * D * 2; a.k = dk.addr + uint64_t(r0) * KH * D * 2; a.v = dv.addr + uint64_t(r0) * VH * D * 2;
        a.g = dg.addr + uint64_t(r0) * VH * 4; a.beta = db.addr + uint64_t(r0) * VH * 4;
        a.state = dst.addr; a.o = dout.addr + uint64_t(r0) * VH * D * 2; a.tokens = n; a.scale = scale;
        a.state_in32 = in32; a.capture32 = capa; a.capture_chunk = cap_chunk; a.state_in32_hi = in32_hi; a.capture32_hi = cap_hi;
        ctx.begin();
        kt::must(k5::delta_rule_chunk(ctx, a, ws.addr), "delta_rule_chunk");
        ctx.barrier();
        ctx.submit_wait();
    };
    std::memcpy(dst.map, state0.data(), SS * 2);
    run(0, T, 0, cap.addr, 2);
    auto o1 = kt::host<uint16_t>(dout, size_t(T) * VH * D), s1 = kt::host<uint16_t>(dst, SS);
    kt::identity("chunk200.o", o1); kt::identity("chunk200.state", s1);
    {
        auto fnv = [](const void *p, size_t n) {
            uint64_t h = 1469598103934665603ull; const auto *b = static_cast<const unsigned char *>(p);
            for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
            return (unsigned long long)h;
        };
        const auto c1 = kt::host<uint32_t>(cap, SS);
        std::printf("bits: o %016llx state %016llx capture %016llx\n", fnv(o1.data(), o1.size() * 2), fnv(s1.data(), s1.size() * 2),
                    fnv(c1.data(), c1.size() * 4));
    }
    // the kernel's own normalised q, k (vkk5::l2norm, as the chunk path calls it)
    auto dqn = kt::make(ctx, q.size() * 2), dkn = kt::make(ctx, k.size() * 2);
    ctx.begin();
    kt::must(k5::l2norm(ctx, dq.addr, dqn.addr, T * KH, D, 1e-6f), "l2norm");
    kt::must(k5::l2norm(ctx, dk.addr, dkn.addr, T * KH, D, 1e-6f), "l2norm");
    ctx.barrier();
    ctx.submit_wait();
    auto qn = kt::host<uint16_t>(dqn, q.size()), kn = kt::host<uint16_t>(dkn, k.size());

    // --- 1. FP64 mirror of the chunk math with the header's rounding points (the HIP test's code)
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
    std::memcpy(dst.map, state0.data(), SS * 2); run(0, T, 0, 0, -1);
    const bool det = kt::host<uint16_t>(dout, o1.size()) == o1 && kt::host<uint16_t>(dst, SS) == s1;
    check("determinism", det, det ? "two identical calls: o and state bit-equal" : "two identical calls DIFFER");
    // --- 4. continuity from the FP32 capture at chunk 2 (row 128), base and split (_hi) forms
    std::memset(dout.map, 0, o1.size() * 2); run(128, T - 128, cap.addr, 0, -1);
    auto o2 = kt::host<uint16_t>(dout, o1.size()), s2 = kt::host<uint16_t>(dst, SS);
    bool cont = s2 == s1; for (size_t i = 128 * VH * D; i < o1.size(); ++i) cont &= o2[i] == o1[i];
    check("continuity (state_in32)", cont, cont ? "rows 128..199 from the chunk-2 capture == the 200-row call, bit for bit" : "rows 128.. DIFFER");
    auto lo = kt::make(ctx, SS * 2), hi = kt::make(ctx, SS * 2);  // heads 0..23 / 24..47
    std::memcpy(dst.map, state0.data(), SS * 2); run(0, T, 0, lo.addr, 2, 0, hi.addr);
    auto lo_h = kt::host<float>(lo, SS / 2), hi_h = kt::host<float>(hi, SS / 2), cap_h = kt::host<float>(cap, SS);
    bool split = std::memcmp(lo_h.data(), cap_h.data(), SS / 2 * 4) == 0 && std::memcmp(hi_h.data(), cap_h.data() + SS / 2, SS / 2 * 4) == 0;
    std::memset(dout.map, 0, o1.size() * 2); run(128, T - 128, lo.addr, 0, -1, hi.addr, 0);
    auto o3 = kt::host<uint16_t>(dout, o1.size()); split &= kt::host<uint16_t>(dst, SS) == s1; for (size_t i = 128 * VH * D; i < o1.size(); ++i) split &= o3[i] == o1[i];
    check("continuity (_hi split)", split, split ? "capture32_hi / state_in32_hi: same capture, same rows and state" : "split form DIFFERS");
    {   // extra: refusals (nothing recorded)
        k5::ChunkArgs a{}; a.q = dq.addr; a.k = dk.addr; a.v = dv.addr; a.g = dg.addr; a.beta = db.addr; a.state = dst.addr; a.o = dout.addr; a.scale = scale;
        bool ok = true;
        ctx.begin();
        a.tokens = 0; ok &= k5::delta_rule_chunk(ctx, a, ws.addr) == CERE_ERR_ARGUMENT;
        a.tokens = 4097; ok &= k5::delta_rule_chunk(ctx, a, ws.addr) == CERE_ERR_ARGUMENT;
        a.tokens = T; ok &= k5::delta_rule_chunk(ctx, a, ws.addr + 16) == CERE_ERR_ARGUMENT;
        a.capture32 = cap.addr; a.capture_chunk = 5; ok &= k5::delta_rule_chunk(ctx, a, ws.addr) == CERE_ERR_ARGUMENT;
        a.capture_chunk = 2; a.capture32_hi = hi.addr; a.capture32 = 0; ok &= k5::delta_rule_chunk(ctx, a, ws.addr) == CERE_ERR_ARGUMENT;
        a.capture32_hi = 0; a.v = dv.addr + 8; ok &= k5::delta_rule_chunk(ctx, a, ws.addr) == CERE_ERR_ARGUMENT;
        ok &= k5::chunk_workspace_bytes(0) == 0 && k5::chunk_workspace_bytes(4097) == 0;
        ctx.submit_wait();
        check("refusals (extra)", ok, ok ? "tokens 0 / 4097, misaligned workspace / v, capture_chunk > chunks, _hi without base: ARGUMENT" : "a bad call was accepted");
    }
    {   // timing
        const int64_t TB = 4096; std::vector<uint16_t> big(TB * VH * D, f2bf(0.5f)); std::vector<float> gb(TB * VH, -0.05f), bb(TB * VH, 0.5f);
        auto dbig = kt::upload(ctx, big), dgb = kt::upload(ctx, gb), dbb = kt::upload(ctx, bb), dbo = kt::make(ctx, big.size() * 2);
        auto bws = kt::make(ctx, k5::chunk_workspace_bytes(TB));
        std::vector<int64_t> rows_list = {TB, int64_t(1205), int64_t(512)};
        if (kt::hooks().samples) rows_list.push_back(256);
        for (int64_t rows : rows_list) {
            k5::ChunkArgs a{}; a.q = dbig.addr; a.k = dbig.addr; a.v = dbig.addr; a.g = dgb.addr; a.beta = dbb.addr;
            a.state = dst.addr; a.o = dbo.addr; a.tokens = rows; a.scale = scale;
            const int reps = kt::reps(7);
            ctx.begin();
            kt::must(k5::delta_rule_chunk(ctx, a, bws.addr), "chunk"); ctx.barrier();
            for (int i = 0; i < reps; ++i) { ctx.stamp(2 * i); kt::must(k5::delta_rule_chunk(ctx, a, bws.addr), "chunk"); ctx.stamp(2 * i + 1); ctx.barrier(); }
            ctx.stamp(2 * reps);
            for (int i = 0; i < 5; ++i) { kt::must(k5::delta_rule_chunk(ctx, a, bws.addr), "chunk"); ctx.barrier(); }
            ctx.stamp(2 * reps + 1);
            ctx.submit_wait();
            std::vector<double> t; for (int i = 0; i < reps; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
            kt::samples("rows" + std::to_string(rows), "warm", t);
            kt::identity("rows" + std::to_string(rows) + ".o", kt::host<uint16_t>(dbo, size_t(rows) * VH * D));
            std::sort(t.begin(), t.end());
            std::printf("%-26s      %4ld rows x 48 heads: %.3f ms per call (median of %d; min %.3f, max %.3f); HIP method (mean of 5 back to back) %.3f ms\n",
                        "timing", long(rows), t[reps / 2], reps, t.front(), t.back(), ctx.stamp_ms(2 * reps, 2 * reps + 1) / 5);
        }
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
