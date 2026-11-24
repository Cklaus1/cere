// VulkanAmd P1: executor/hip/ktest/k2mx_test.cpp's contract rows on the Vulkan small-M GEMV and MXFP8 upcast.
//  1. dequantize == the CPU upcast w = q * 2^(s - 127) bit for bit (random valid E4M3 bytes, scales over the whole
//     contract range [10, 246]);
//  2. gemv on the MXFP8 weight == gemv on its BF16 upcast, bit for bit, m = 1 .. 16, BF16 and FP32 outputs;
//  3. vs FP64: FP32 output |err| / sum|products| <= 1e-5; BF16 output within 1 ulp of the FP64 value;
//  4. row independence: each row of an m-row call (m = 2 .. 16, from rows 0 and 5) == its m = 1 call; repeats equal;
//  5. timing (GPU timestamps, median of 15 calls): the lm_head-sized stream (248,320 x 2,560) BF16 and MXFP8 at
//     m = 1, 4, 5, 8, 16, and the short-K shapes (n 10,240, k 320 / 640) with rotating weight copies (>= 1 GB, cold)
//     and one copy (hot). hipBLASLt (the HIP test's third column) has no Vulkan counterpart and is not timed.
// Same shapes, seeds and data generation as the HIP test for rows 1-4. Usage: k2mx_test [spv dir]
// [--no-timing|--tune|--stability R] (R rounds of the short-K cold rows m 4 / 16, one call and 20 calls per window)
// [--space long|short --ms M[,M..] [--mapping G,E,WG,U,J] [--identity DIR] [--samples]] (VulkanAutotune H1, space_mode).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "launch/vk_launch_testing.h"
#include "ktest_util.h"

using kt::bf2f;
using kt::f2bf;
using cere::vkk2::kGemvMaxM;
using cere::vkk2::Mapping;
static int g_fail = 0;

static double e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const double v = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
    return (b & 0x80) ? -v : v;
}

struct Mx { std::vector<uint8_t> q, s; std::vector<uint16_t> up; };
// realistic weights: scales around 2^-8 .. 2^-4 with a few at the contract's ends (lo / hi: 0 = whole range)
static Mx make_mx(std::mt19937 &g, uint32_t n, uint32_t k, int lo, int hi) {
    Mx w; w.q.resize(size_t(n) * k); w.s.resize(size_t(n) * k / 32); w.up.resize(w.q.size());
    std::uniform_int_distribution<int> qb(0, 255), sb(lo ? lo : 10, hi ? hi : 246);
    for (auto &b : w.q) { do b = uint8_t(qb(g)); while ((b & 0x7f) == 0x7f); }
    for (auto &b : w.s) b = uint8_t(sb(g));
    for (size_t i = 0; i < w.q.size(); ++i) {
        const double v = std::ldexp(e4m3(w.q[i]), int(w.s[i / 32]) - 127);
        w.up[i] = f2bf(float(v));
        if (double(bf2f(w.up[i])) != v) { std::printf("CPU upcast not exact\n"); std::exit(2); }
    }
    return w;
}

// Fast fill for the timing buffers (values do not change the speed): valid E4M3 bytes, scales 112 .. 122, normal BF16.
static uint64_t g_xs = 0x9E3779B97F4A7C15ull;
static uint64_t xs() { g_xs ^= g_xs << 13; g_xs ^= g_xs >> 7; g_xs ^= g_xs << 17; return g_xs; }
static void fill_q(uint8_t *p, size_t n) { for (size_t i = 0; i < n; ++i) { uint8_t b = uint8_t(xs()); if ((b & 0x7f) == 0x7f) b ^= 1; p[i] = b; } }
static void fill_s(uint8_t *p, size_t n) { for (size_t i = 0; i < n; ++i) p[i] = uint8_t(112 + xs() % 11); }
static void fill_bf(uint16_t *p, size_t n) { for (size_t i = 0; i < n; ++i) p[i] = uint16_t(0x3C00 + (xs() & 0x03FF) + ((xs() & 1) << 15)); }

static void contract_rows(cere::vk::Context &ctx) {
    std::mt19937 g(7); std::normal_distribution<float> nd;
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    {   // 1. dequantize, whole scale range
        const uint32_t n = 300, k = 1024;
        Mx w = make_mx(g, n, k, 0, 0);
        auto dq = kt::upload(ctx, w.q), ds = kt::upload(ctx, w.s), dw = kt::make(ctx, w.up.size() * 2);
        ctx.begin(); kt::must(cere::vkk2::mxfp8_dequantize(ctx, dq.addr, ds.addr, n, k, dw.addr), "dequantize"); ctx.barrier(); ctx.submit_wait();
        const bool ok = kt::host<uint16_t>(dw, w.up.size()) == w.up;
        std::printf("%-22s %s  300 x 1024, scales 10 .. 246\n", "dequantize exact", ok ? "PASS" : "FAIL");
        if (!ok) ++g_fail;
        ctx.free(dq); ctx.free(ds); ctx.free(dw);
    }
    {   // 2-4
        const uint32_t n = 333, k = 2560, M = 21;
        Mx w = make_mx(g, n, k, 112, 122);
        auto x = rnd(size_t(M) * k, 1.f);
        auto dq = kt::upload(ctx, w.q), ds = kt::upload(ctx, w.s), dwb = kt::upload(ctx, w.up), dx = kt::upload(ctx, x);
        auto yb = kt::make(ctx, M * n * 2), yb2 = kt::make(ctx, M * n * 2), yf = kt::make(ctx, M * n * 4), yf2 = kt::make(ctx, M * n * 4);
        long diff = 0;
        for (uint32_t m = 1; m <= kGemvMaxM; ++m) {
            for (bool f32 : {false, true}) {
                auto &a = f32 ? yf : yb, &b = f32 ? yf2 : yb2;
                ctx.begin();
                kt::must(cere::vkk2::gemv(ctx, dx.addr, dq.addr, ds.addr, a.addr, m, n, k, f32), "gemv mx");
                kt::must(cere::vkk2::gemv(ctx, dx.addr, dwb.addr, 0, b.addr, m, n, k, f32), "gemv bf16");
                ctx.barrier(); ctx.submit_wait();
                diff += kt::host<uint8_t>(a, size_t(m) * n * (f32 ? 4 : 2)) != kt::host<uint8_t>(b, size_t(m) * n * (f32 ? 4 : 2));
            }
        }
        std::printf("%-22s %s  333 x 2560, m 1 .. 16, BF16 + FP32 out: %ld calls differ\n", "MXFP8 == BF16 upcast", diff ? "FAIL" : "PASS", diff);
        if (diff) ++g_fail;
        ctx.begin();
        kt::must(cere::vkk2::gemv(ctx, dx.addr, dq.addr, ds.addr, yf.addr, kGemvMaxM, n, k, true), "gemv");
        kt::must(cere::vkk2::gemv(ctx, dx.addr, dq.addr, ds.addr, yb.addr, kGemvMaxM, n, k, false), "gemv");
        ctx.barrier(); ctx.submit_wait();
        auto hf = kt::host<float>(yf, kGemvMaxM * n); auto hb = kt::host<uint16_t>(yb, kGemvMaxM * n);
        double worst = 0; long bad = 0;
        for (uint32_t r = 0; r < kGemvMaxM; ++r) for (uint32_t j = 0; j < n; ++j) {
            double s = 0, sa = 0;
            for (uint32_t i = 0; i < k; ++i) { const double p = double(bf2f(x[r * k + i])) * bf2f(w.up[size_t(j) * k + i]); s += p; sa += std::fabs(p); }
            worst = std::max(worst, std::fabs(hf[r * n + j] - s) / (sa + 1e-30));
            bad += std::abs(int(hb[r * n + j]) - int(f2bf(float(s)))) > 1;
        }
        const bool ok3 = worst <= 1e-5 && bad == 0;
        std::printf("%-22s %s  FP32 worst |err| / sum|p| %.1e; BF16 > 1 ulp: %ld\n", "vs FP64", ok3 ? "PASS" : "FAIL", worst, bad);
        if (!ok3) ++g_fail;
        long badr = 0;
        for (uint32_t s0 : {0u, 5u}) {
            std::vector<std::vector<uint16_t>> one(kGemvMaxM);
            for (uint32_t r = 0; r < kGemvMaxM; ++r) {
                kt::gemv_sync(ctx, dx.addr + uint64_t(s0 + r) * k * 2, dq.addr, ds.addr, yb2.addr, 1, n, k, false);
                one[r] = kt::host<uint16_t>(yb2, n);
            }
            for (uint32_t m = 2; m <= kGemvMaxM; ++m) {
                kt::gemv_sync(ctx, dx.addr + uint64_t(s0) * k * 2, dq.addr, ds.addr, yb.addr, m, n, k, false);
                auto h = kt::host<uint16_t>(yb, m * n);
                for (uint32_t r = 0; r < m; ++r) badr += std::memcmp(h.data() + r * n, one[r].data(), n * 2) != 0;
            }
        }
        kt::gemv_sync(ctx, dx.addr, dq.addr, ds.addr, yb2.addr, kGemvMaxM, n, k, false);
        const bool rep = kt::host<uint16_t>(yb2, kGemvMaxM * n) == hb;
        std::printf("%-22s %s  rows of m 2 .. 16 calls (from 0, 5) vs m = 1: %ld differ; repeat %s\n", "row independence", badr == 0 && rep ? "PASS" : "FAIL", badr, rep ? "bit-equal" : "DIFFERS");
        if (badr || !rep) ++g_fail;
        {   // negative control: the bit comparisons above must be able to see an order change. A forced different order
            // (G 16 instead of 32) at m 16 must differ from the default order somewhere; equal would mean the
            // comparison is blind (FAIL).
            const Mapping other{16, 16, 256, 1, 1};
            ctx.begin(); kt::must(cere::vkk2::gemv(ctx, dx.addr, dq.addr, ds.addr, yf2.addr, kGemvMaxM, n, k, true, &other), "gemv"); ctx.barrier(); ctx.submit_wait();
            auto ho = kt::host<float>(yf2, kGemvMaxM * n);
            long nd = 0; for (size_t i = 0; i < ho.size(); ++i) nd += std::memcmp(&ho[i], &hf[i], 4) != 0;
            std::printf("%-22s %s  forced order (G16,E16) vs default (G32,E16), m 16 FP32: %ld of %zu outputs differ (must be > 0)\n", "negative control", nd ? "PASS" : "FAIL", nd, ho.size());
            if (!nd) ++g_fail;
        }
        for (auto *b : {&dq, &ds, &dwb, &dx, &yb, &yb2, &yf, &yf2}) ctx.free(*b);
    }
    {   // 4b. the same invariance and MXFP8 identity on the short-K mapping (k 320 and 640 take a different mapping)
        for (uint32_t k : {320u, 640u}) {
            const uint32_t n = 1000, M = 21;
            Mx w = make_mx(g, n, k, 112, 122);
            auto x = rnd(size_t(M) * k, 1.f);
            auto dq = kt::upload(ctx, w.q), ds = kt::upload(ctx, w.s), dwb = kt::upload(ctx, w.up), dx = kt::upload(ctx, x);
            auto ya = kt::make(ctx, M * n * 4), yb = kt::make(ctx, M * n * 4);
            long diff = 0, badr = 0;
            for (uint32_t m = 1; m <= kGemvMaxM; ++m) for (bool f32 : {false, true}) {
                ctx.begin();
                kt::must(cere::vkk2::gemv(ctx, dx.addr, dq.addr, ds.addr, ya.addr, m, n, k, f32), "gemv mx");
                kt::must(cere::vkk2::gemv(ctx, dx.addr, dwb.addr, 0, yb.addr, m, n, k, f32), "gemv bf16");
                ctx.barrier(); ctx.submit_wait();
                diff += kt::host<uint8_t>(ya, size_t(m) * n * (f32 ? 4 : 2)) != kt::host<uint8_t>(yb, size_t(m) * n * (f32 ? 4 : 2));
            }
            kt::gemv_sync(ctx, dx.addr + uint64_t(5) * k * 2, dwb.addr, 0, ya.addr, kGemvMaxM, n, k, true);
            auto all = kt::host<float>(ya, kGemvMaxM * n);
            double worst = 0;
            for (uint32_t r = 0; r < kGemvMaxM; ++r) {
                kt::gemv_sync(ctx, dx.addr + uint64_t(5 + r) * k * 2, dwb.addr, 0, yb.addr, 1, n, k, true);
                badr += std::memcmp(kt::host<float>(yb, n).data(), all.data() + r * n, n * 4) != 0;
                for (uint32_t j = 0; j < n; ++j) {
                    double s = 0, sa = 0;
                    for (uint32_t i = 0; i < k; ++i) { const double p = double(bf2f(x[(5 + r) * k + i])) * bf2f(w.up[size_t(j) * k + i]); s += p; sa += std::fabs(p); }
                    worst = std::max(worst, std::fabs(all[r * n + j] - s) / (sa + 1e-30));
                }
            }
            const auto mp = cere::vkk2::mapping_for(k, 1);
            const bool ok = diff == 0 && badr == 0 && worst <= 1e-5;
            std::printf("%-22s %s  1000 x %u (G%u,E%u): MXFP8 vs upcast %ld calls differ; m 16 (from 5) rows vs m 1: %ld differ; FP32 worst %.1e\n",
                        k == 320 ? "short-K k 320" : "short-K k 640", ok ? "PASS" : "FAIL", k, mp.G, mp.E, diff, badr, worst);
            if (!ok) ++g_fail;
            for (auto *b : {&dq, &ds, &dwb, &dx, &ya, &yb}) ctx.free(*b);
        }
    }
}

struct Shape { const char *name; uint32_t n, k; uint64_t cold_bytes; };

// Times BF16 and MXFP8 at the given m list. copies: enough weight copies for >= cold_bytes of distinct weights.
// per: calls per timestamp window (time_gemv); the line says `xN` when it is not 1.
static void time_shape(cere::vk::Context &ctx, const Shape &sh, const std::vector<uint32_t> &ms, const Mapping *force, bool hot_too,
                       uint32_t per = 1) {
    const uint64_t wb = uint64_t(sh.n) * sh.k * 2, qb = uint64_t(sh.n) * sh.k, sb = uint64_t(sh.n) * sh.k / 32;
    const uint32_t copies = uint32_t(std::max<uint64_t>(1, (sh.cold_bytes + qb - 1) / qb));
    auto dwb = kt::make(ctx, wb * copies), dq = kt::make(ctx, qb * copies), ds = kt::make(ctx, sb * copies);
    auto dx = kt::make(ctx, uint64_t(16) * sh.k * 2), dy = kt::make(ctx, uint64_t(16) * sh.n * 4);
    fill_bf(static_cast<uint16_t *>(dwb.map), wb * copies / 2); fill_q(static_cast<uint8_t *>(dq.map), qb * copies);
    fill_s(static_cast<uint8_t *>(ds.map), sb * copies); fill_bf(static_cast<uint16_t *>(dx.map), uint64_t(16) * sh.k);
    for (uint32_t m : ms) {
        const auto mp = force ? *force : cere::vkk2::mapping_for(sh.k, m);
        for (int hot = 0; hot <= (hot_too ? 1 : 0); ++hot) {
            const uint32_t c = hot ? 1 : copies;
            const auto tb = kt::time_gemv(ctx, dx.addr, dwb.addr, wb, 0, 0, c, dy.addr, m, sh.n, sh.k, force, 15, per);
            const auto tm = kt::time_gemv(ctx, dx.addr, dq.addr, qb, ds.addr, sb, c, dy.addr, m, sh.n, sh.k, force, 15, per);
            std::string tag = hot ? "hot " : (copies > 1 ? "cold" : "    ");
            if (per != 1) tag += " x" + std::to_string(per);
            std::printf("%-8s %-24s (G%u,E%u,WG%u,U%u,J%u) %s m %2u: bf16 %.3f ms (%.0f GB/s) [min %.3f max %.3f], mxfp8 %.3f ms (%.0f GB/s) [min %.3f max %.3f]\n",
                        "timing", sh.name, mp.G, mp.E, mp.WG, mp.U, mp.J, tag.c_str(), m,
                        tb.ms_median, wb / (tb.ms_median * 1e-3) / 1e9, tb.ms_min, tb.ms_max,
                        tm.ms_median, qb / (tm.ms_median * 1e-3) / 1e9, tm.ms_min, tm.ms_max);
            std::fflush(stdout);
        }
    }
    for (auto *b : {&dwb, &dq, &ds, &dx, &dy}) ctx.free(*b);
}

// VulkanAutotune H1 (--mapping G,E,WG,U,J --space long|short --ms M[,M..]): one candidate mapping on its space's
// production shapes (scripts/tune/vulkan_autotune.py LONG_K / SHORT_K), forced at the space's row counts, the default
// mapping at every other m. Contract rows per (shape, m): accuracy (FP32 output vs FP64, |err| / sum|products|
// <= 1e-5, 64 sampled columns of 4 rows), determinism (repeat bit-equal), row_invariance (each row of the m-row call
// == its m 1 call, whose mapping is the candidate's when the space holds m 1, else the default: G / E are the order),
// mxfp8_eq_bf16 (MXFP8 shapes: the MXFP8 call == the call on its exact BF16 upcast). --identity writes every output;
// timing: the production format, weight copies >= 1 GB (cold), >= 15 samples (TIMING lines under --samples).
struct SpaceShape { const char *name; uint32_t n, k; bool mx; };
static int space_mode(cere::vk::Context &ctx, const Mapping &cand, const std::string &space, const std::vector<uint32_t> &ms, bool timing) {
    static const SpaceShape long_k[] = {{"gdn_qkv", 10240, 2560, true}, {"gdn_z", 6144, 2560, true}, {"attn_qkv", 13312, 2560, true},
                                        {"out_k6144", 2560, 6144, true}, {"hc_down", 320, 10240, false}, {"lm_head", 248320, 2560, false}};
    static const SpaceShape short_k[] = {{"hc_up", 10240, 320, false}};
    std::vector<SpaceShape> shapes;
    if (space == "long") shapes.assign(std::begin(long_k), std::end(long_k));
    else if (space == "short") shapes.assign(std::begin(short_k), std::end(short_k));
    else { std::printf("FATAL --space %s: long | short\n", space.c_str()); return 2; }
    auto in_space = [&](uint32_t m) { return std::find(ms.begin(), ms.end(), m) != ms.end(); };
    std::mt19937 g(29);
    for (const auto &sh : shapes) {
        const uint32_t n = sh.n, k = sh.k, M = kGemvMaxM;
        auto force = [&](uint32_t m) { return in_space(m) ? &cand : nullptr; };
        Mx w;
        std::vector<uint16_t> wbf;
        if (sh.mx) w = make_mx(g, n, k, 112, 122);
        else { wbf.resize(size_t(n) * k); fill_bf(wbf.data(), wbf.size()); }
        const std::vector<uint16_t> &wup = sh.mx ? w.up : wbf;
        std::vector<uint16_t> x(size_t(M) * k); fill_bf(x.data(), x.size());
        auto dx = kt::upload(ctx, x), dwb = kt::upload(ctx, wup), dy = kt::make(ctx, size_t(M) * n * 4), dz = kt::make(ctx, size_t(M) * n * 4);
        cere::vk::Buffer dq, ds;
        if (sh.mx) { dq = kt::upload(ctx, w.q); ds = kt::upload(ctx, w.s); }
        for (uint32_t m : ms) {
            const std::string key = std::string(sh.name) + ".m" + std::to_string(m);
            bool acc_ok = true, det_ok = true, inv_ok = true, mx_ok = true;
            for (bool f32 : {false, true}) {
                const size_t es = f32 ? 4 : 2;
                auto run = [&](uint64_t xa, uint32_t mm, const cere::vk::Buffer &out, bool use_mx) {
                    kt::gemv_sync(ctx, xa, use_mx ? dq.addr : dwb.addr, use_mx ? ds.addr : 0, out.addr, mm, n, k, f32, force(mm));
                    return kt::host<uint8_t>(out, size_t(mm) * n * es);
                };
                const auto y = run(dx.addr, m, dy, sh.mx);
                det_ok &= run(dx.addr, m, dy, sh.mx) == y;
                if (sh.mx) mx_ok &= run(dx.addr, m, dz, false) == y;
                kt::identity(key + (f32 ? ".f32" : ".bf16"), y);
                for (uint32_t r = 0; r < m && m > 1; ++r)
                    inv_ok &= std::memcmp(run(dx.addr + uint64_t(r) * k * 2, 1, dz, sh.mx).data(), y.data() + size_t(r) * n * es, n * es) == 0;
                if (f32) {
                    std::uniform_int_distribution<uint32_t> col(0, n - 1);
                    for (uint32_t r = 0; r < std::min(m, 4u); ++r) for (int t = 0; t < 64; ++t) {
                        const uint32_t j = col(g); double sum = 0, sa = 0;
                        for (uint32_t i = 0; i < k; ++i) { const double p = double(bf2f(x[size_t(r) * k + i])) * bf2f(wup[size_t(j) * k + i]); sum += p; sa += std::fabs(p); }
                        float got; std::memcpy(&got, &y[(size_t(r) * n + j) * 4], 4);
                        acc_ok &= std::fabs(got - sum) <= 1e-5 * (sa + 1e-30);
                    }
                }
            }
            auto row = [&](const char *name, bool ok) { std::printf("CONTRACT %s %s %s\n", key.c_str(), name, ok ? "PASS" : "FAIL"); if (!ok) ++g_fail; };
            row("accuracy", acc_ok); row("determinism", det_ok); row("row_invariance", inv_ok);
            if (sh.mx) row("mxfp8_eq_bf16", mx_ok);
        }
        for (auto *b : {&dx, &dwb, &dy, &dz}) ctx.free(*b);
        if (sh.mx) { ctx.free(dq); ctx.free(ds); }
        if (!timing || g_fail) continue;
        // timing in the production format, weight copies >= 1 GB (cold)
        const uint64_t wb = uint64_t(n) * k * (sh.mx ? 1 : 2), sb = sh.mx ? uint64_t(n) * k / 32 : 0;
        const uint32_t copies = uint32_t(std::max<uint64_t>(1, (1000000000ull + wb - 1) / wb));
        auto tw = kt::make(ctx, wb * copies), tx = kt::make(ctx, uint64_t(16) * k * 2), ty = kt::make(ctx, uint64_t(16) * n * 4);
        cere::vk::Buffer ts;
        if (sh.mx) { fill_q(static_cast<uint8_t *>(tw.map), wb * copies); ts = kt::make(ctx, sb * copies); fill_s(static_cast<uint8_t *>(ts.map), sb * copies); }
        else fill_bf(static_cast<uint16_t *>(tw.map), wb * copies / 2);
        fill_bf(static_cast<uint16_t *>(tx.map), uint64_t(16) * k);
        for (uint32_t m : ms) {
            const auto t = kt::time_gemv(ctx, tx.addr, tw.addr, wb, sh.mx ? ts.addr : 0, sb, copies, ty.addr, m, n, k, &cand, kt::reps(15));
            kt::samples(std::string(sh.name) + ".m" + std::to_string(m), "cold", t.ms);
            std::printf("timing %s m %u (G%u,E%u,WG%u,U%u,J%u) %s: median %.4f ms [min %.4f max %.4f]\n", sh.name, m, cand.G, cand.E,
                        cand.WG, cand.U, cand.J, sh.mx ? "mxfp8" : "bf16", t.ms_median, t.ms_min, t.ms_max);
        }
        for (auto *b : {&tw, &tx, &ty}) ctx.free(*b);
        if (sh.mx) ctx.free(ts);
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

int main(int argc, char **argv) {
    auto ctx_owner = cere::vk::testing::create_or_die(argc > 1 && argv[1][0] != '-' ? argv[1] : "build/vulkan/spv");
    cere::vk::Context &ctx = *ctx_owner;
    bool timing = true, tune = false;
    int stability = 0;  // --stability R: R rounds of the short-K cold rows (fresh buffers each round), x1 and x20
    std::string space;
    Mapping cand;
    std::vector<uint32_t> space_ms;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (kt::hook_arg(argc, argv, i)) continue;
        if (a == "--no-timing") timing = false;
        if (a == "--tune") tune = true;
        if (a == "--stability" && i + 1 < argc) stability = std::atoi(argv[++i]);
        if (a == "--space" && i + 1 < argc) space = argv[++i];
        if (a == "--mapping" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%u,%u,%u,%u,%u", &cand.G, &cand.E, &cand.WG, &cand.U, &cand.J) != 5) { std::printf("FATAL --mapping G,E,WG,U,J\n"); return 2; }
        }
        if (a == "--ms" && i + 1 < argc) {
            for (const char *p = argv[++i]; *p;) { space_ms.push_back(uint32_t(std::strtoul(p, const_cast<char **>(&p), 10))); if (*p == ',') ++p; else if (*p) { std::printf("FATAL --ms\n"); return 2; } }
        }
    }
    kt::tune_keys({});
    if (!space.empty()) {
        if (space_ms.empty()) { std::printf("FATAL --space needs --ms\n"); return 2; }
        return space_mode(ctx, cand, space, space_ms, timing);
    }
    std::printf("device: %s, driver %s %s\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str());
    if (!tune && stability == 0) contract_rows(ctx);
    const Shape lm{"lm_head 248320 x 2560", 248320, 2560, 0};
    const Shape s320{"short-K n 10240 k 320", 10240, 320, 1000000000ull}, s640{"short-K n 10240 k 640", 10240, 640, 1000000000ull};
    if (stability > 0) {
        for (int r = 0; r < stability; ++r) {
            std::printf("round %d\n", r);
            for (uint32_t per : {1u, 20u}) { time_shape(ctx, s320, {4, 16}, nullptr, false, per); time_shape(ctx, s640, {4, 16}, nullptr, false, per); }
        }
        return 0;
    }
    if (timing && !tune) {
        time_shape(ctx, lm, {1, 4, 5, 8, 16}, nullptr, false);
        time_shape(ctx, s320, {1, 4, 16}, nullptr, true);
        time_shape(ctx, s640, {1, 4, 16}, nullptr, true);
        time_shape(ctx, s320, {1, 4, 16}, nullptr, false, 20);
        time_shape(ctx, s640, {1, 4, 16}, nullptr, false, 20);
    }
    if (tune) {  // speed of candidate mappings (different G/E = different order: tuning only, never a contract row)
        for (uint32_t wg : {256u, 128u, 64u}) for (uint32_t u : {1u, 2u}) for (uint32_t j : {1u, 2u, 4u}) {
            if ((wg == 64 && j == 1) || (u == 2 && j == 4)) continue;
            const Mapping mp{32, 16, wg, u, j};
            time_shape(ctx, lm, {1, 4, 5, 8, 16}, &mp, false);
        }
        for (uint32_t ge : {0x0808u, 0x0810u, 0x1008u, 0x2010u}) for (uint32_t wj : {0x10001u, 0x8002u, 0x10002u}) {
            const Mapping mp{ge >> 8, ge & 0xFF, wj >> 8, 1, wj & 0xFF};
            time_shape(ctx, s320, {1, 4, 16}, &mp, false); time_shape(ctx, s640, {1, 4, 16}, &mp, false);
        }
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
