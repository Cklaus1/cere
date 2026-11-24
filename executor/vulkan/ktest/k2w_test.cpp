// VulkanAmd P2: executor/hip/ktest/k2w_test.cpp's contract rows on the Vulkan K2 large-m GEMM (k2_gemm_vk.h), plus
// the rows the BF16-as-FP16 design needs (spec "Contracts"; HIP's bounds unchanged):
//  0. accumulator-layout probe (the epilogue's assumption) PASS, else every GEMM row is refused;
//  1. vs FP64 (m 200, n 300, k 256: partial tiles both ways), BF16 and MXFP8 weights: every BF16 output within 1 ulp
//     of the FP64 product rounded to BF16, FP32 output |err| / sum|products| <= 1e-5. The MXFP8 row's weights are the
//     BF16 row's N(0, 0.1) weights MXFP8-quantized. On k2mx_test's uniform random E4M3 bytes ("MX random bytes") the
//     row is the FP32 bound only and the BF16 1-ulp count is INFO: there FP64 results cancel to ~4e-6 of sum|p|, and
//     the exact FP32 FMA route and HIP's own MXFP8 large-m path (dequantize + k2_wmma, hip_k2_gemm_timing's first
//     line) miss 1 ulp on the same outputs (HIP: the same 2 outputs, the same bits as this GEMM);
//  2. shift invariance: a call on rows s.. equals rows s.. of the full call bit for bit (s = 1, 17, 64, 100); row-count
//     invariance: m = 1, 37, 128 at row 0; repeats bit-identical (512 x 10240 x 2048), BF16 and MXFP8 weights;
//  3. MXFP8 == BF16 upcast, bit for bit, m 17 / 64 / 200, BF16 and FP32 out, with flagged (FP32-route) rows in both;
//  4. the exact fallback: rows that FP16 * 2^e cannot carry are classified by the scan (P4-B): a few tail values far
//     below the row max (2^-45 next to 1.0, a tiny weight) -> SPLIT, an Inf or a 2^110 row -> FB (FP32 route); split x
//     rows' outputs are exact (2^-45 comes back as 2^-45: matrix unit for the row without its tail + exact FP32 chain
//     for the tail), FB rows' outputs and split / FB w columns' outputs equal the FP32 route's bits, non-finite outputs
//     classify as in FP64, and shift / row-count invariance holds with mixed routes; FP16-subnormal operands (2^-36
//     next to 1.0: scaled to 2^-21) stay on the matrix path and come back exact. Four variants (A: with a split w row,
//     handled as FB; B: x rows only, the FP32 route's sparse path; C: 43 FB x rows, its full-tile path through x rows
//     alone; D: 43 split x rows in one tile, two 32-row tail chunks);
//  4b. split limits: a row with SPLIT_MAX (64) tail values is SPLIT, with 65 FB; tails down to 2^-133 (BF16
//     subnormal) and 2^-100 come back exact; vs FP64 on both;
//  4c. split pool overflow: more tail values in a tile than the GEMM's shared pool holds (the "late" rows' second
//     read): vs FP64, and shifted / shorter calls (other pool / late assignments) bit-equal;
//  4d. zero sensitivity (P4-E; P4-A's probe2 fact: a coopMatMulAdd result depends on B values that meet zeros of A):
//     the outputs of rows with many +0 / -0 zeros and split (zeroed) tails do not change when OTHER x rows or OTHER w
//     rows change at those positions, nor at m edges (clamped padding); a split row == the same row with its tails as
//     +0 on the P2 loop + the FP32 fma chain of the tails; INFO: the sign of zero, x values under zero weights;
//  5. the FP32 route alone (GemmRoute::kFp32) vs FP64 and its shift invariance;
//  6. the 16 / 17 boundary: run() routes by the row class like HIP's GemmPlan::run: m 7 with cls 40 (rows 0.. and
//     33..) == those rows of the 40-row call; rows of an m 16 call == m 1 calls (GEMV class); refusals (cls < m,
//     k % 64 != 0, a context without the probe). INFO (not a contract row; HIP's kernels differ the same way): how
//     many outputs differ between the GEMV (m 16) and the GEMM (m 17) on the same rows;
//  7. fallback frequency: rows the scan marks FB / SPLIT on synthetic activation sets (m 4,096, k 2,560) and the scan's time;
//  8. --timing: GPU-timestamp medians on k2w_test's four prefill shapes at m 17, 64, 512, 4,096: BF16 (weight info
//     prepared / scanned in the call), MXFP8, the FP32 route, and the auto route with one flagged x row.
// Usage: k2w_test [spv dir] [--timing]. Exit 1 on any FAIL.
#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "launch/vk_launch_testing.h"
#include "kernels/k2_gemm_vk.h"
#include "ktest_util.h"

using kt::bf2f;
using kt::f2bf;
namespace vkk2 = cere::vkk2;
using cere::vk::Buffer;
using cere::vk::Context;
static int g_fail = 0;

static void row(const char *name, bool ok, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
static void row(const char *name, bool ok, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::printf("%-26s %s  %s\n", name, ok ? "PASS" : "FAIL", buf);
    std::fflush(stdout);
    if (!ok) ++g_fail;
}

static double e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const double v = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
    return (b & 0x80) ? -v : v;
}
struct Mx { std::vector<uint8_t> q, s; std::vector<uint16_t> up; };
// k2mx_test's generator: valid E4M3 bytes, block scales in [lo, hi]; up = the exact BF16 upcast
static Mx make_mx(std::mt19937 &g, uint32_t n, uint32_t k, int lo, int hi) {
    Mx w; w.q.resize(size_t(n) * k); w.s.resize(size_t(n) * k / 32); w.up.resize(w.q.size());
    std::uniform_int_distribution<int> qb(0, 255), sb(lo, hi);
    for (auto &b : w.q) { do b = uint8_t(qb(g)); while ((b & 0x7f) == 0x7f); }
    for (auto &b : w.s) b = uint8_t(sb(g));
    return w;
}
static void upcast(Mx &w, uint32_t k) {
    for (size_t i = 0; i < w.q.size(); ++i) {
        const double v = std::ldexp(e4m3(w.q[i]), int(w.s[i / k * (k / 32) + (i % k) / 32]) - 127);
        w.up[i] = f2bf(float(v));
        if (double(bf2f(w.up[i])) != v) { std::printf("CPU upcast not exact\n"); std::exit(2); }
    }
}

// OCP MX quantization of BF16 values: per 32-block scale 2^(floor(log2 amax) - 8), E4M3 round to nearest even
// (ties to the even code), saturating at 448; up = the exact BF16 upcast of the result.
static Mx quantize_mx(const std::vector<uint16_t> &v, uint32_t n, uint32_t k) {
    Mx w; w.q.resize(size_t(n) * k); w.s.resize(size_t(n) * k / 32); w.up.resize(w.q.size());
    for (size_t b = 0; b < w.s.size(); ++b) {
        double amax = 0; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(double(bf2f(v[b * 32 + i]))));
        int e = 0; if (amax > 0) { std::frexp(amax, &e); e -= 1; }
        const int se = std::clamp(e - 8, -117, 119);
        w.s[b] = uint8_t(se + 127);
        for (int i = 0; i < 32; ++i) {
            const double t = std::fabs(double(bf2f(v[b * 32 + i]))) / std::ldexp(1.0, se);
            int best = 0; double bd = 1e300;
            for (int c = 0; c < 127; ++c) {
                const double d = std::fabs(e4m3(uint8_t(c)) - t);
                if (d < bd || (d == bd && (c & 1) == 0 && (best & 1) == 1)) { bd = d; best = c; }
            }
            w.q[b * 32 + i] = uint8_t(best | ((v[b * 32 + i] & 0x8000) ? 0x80 : 0));
        }
    }
    upcast(w, k);
    return w;
}

struct Run {
    Context &ctx;
    void gemm(uint64_t x, uint64_t w, uint64_t s, uint64_t y, uint32_t m, uint32_t n, uint32_t k, bool f32, uint64_t ws,
              uint64_t winfo = 0, vkk2::GemmRoute route = vkk2::GemmRoute::kAuto) {
        vkk2::GemmOptions o; o.route = route;
        ctx.begin();
        kt::must(vkk2::gemm(ctx, x, w, s, y, m, n, k, f32, ws, winfo, &o), "gemm");
        ctx.barrier();
        ctx.submit_wait();
    }
    void run(uint64_t x, uint64_t w, uint64_t s, uint64_t y, uint32_t m, uint32_t n, uint32_t k, bool f32, uint32_t cls, uint64_t ws) {
        ctx.begin();
        kt::must(vkk2::run(ctx, x, w, s, y, m, n, k, f32, cls, ws), "run");
        ctx.barrier();
        ctx.submit_wait();
    }
};

// FP64 reference of one output and sum |products|
static void ref(const uint16_t *xr, const uint16_t *wr, uint32_t k, double &s, double &sa) {
    s = 0; sa = 0;
    for (uint32_t t = 0; t < k; ++t) { const double p = double(bf2f(xr[t])) * bf2f(wr[t]); s += p; sa += std::fabs(p); }
}
static float f32at(const Buffer &b, size_t i) { float f; std::memcpy(&f, static_cast<const uint8_t *>(b.map) + 4 * i, 4); return f; }
static uint16_t bfat(const Buffer &b, size_t i) { uint16_t h; std::memcpy(&h, static_cast<const uint8_t *>(b.map) + 2 * i, 2); return h; }
static bool same(const Buffer &a, size_t ao, const Buffer &b, size_t bo, size_t bytes) {
    return std::memcmp(static_cast<const uint8_t *>(a.map) + ao, static_cast<const uint8_t *>(b.map) + bo, bytes) == 0;
}
static long ndiff16(const Buffer &a, size_t ao, const Buffer &b, size_t bo, size_t count) {
    long d = 0;
    for (size_t i = 0; i < count; ++i) d += bfat(a, ao / 2 + i) != bfat(b, bo / 2 + i);
    return d;
}

// 1: vs FP64 on (x, w-values) with weight buffer w / scales s
// ulp_info: the BF16 1-ulp count is printed, not part of PASS (only for data on which HIP's own kernel misses it too)
static void vs_fp64(Run &R, const char *name, const std::vector<uint16_t> &x, const std::vector<uint16_t> &wv, uint64_t w,
                    uint64_t s, uint32_t m, uint32_t n, uint32_t k, vkk2::GemmRoute route = vkk2::GemmRoute::kAuto,
                    bool ulp_info = false) {
    auto dx = kt::upload(R.ctx, x), y = kt::make(R.ctx, size_t(m) * n * 2), yf = kt::make(R.ctx, size_t(m) * n * 4),
         ws = kt::make(R.ctx, vkk2::gemm_workspace_bytes(m, n));
    R.gemm(dx.addr, w, s, y.addr, m, n, k, false, ws.addr, 0, route);
    R.gemm(dx.addr, w, s, yf.addr, m, n, k, true, ws.addr, 0, route);
    long bad = 0; double worst = 0;
    for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) {
        double r, sa; ref(&x[size_t(i) * k], &wv[size_t(j) * k], k, r, sa);
        const bool b1 = std::abs(int(bfat(y, size_t(i) * n + j)) - int(f2bf(float(r)))) > 1;
        bad += b1;
        worst = std::max(worst, std::fabs(f32at(yf, size_t(i) * n + j) - r) / (sa + 1e-30));
        if (b1 && bad <= 4)
            std::printf("    > 1 ulp at (%u, %u): FP64 %.9g (bf16 0x%04x), got bf16 0x%04x, FP32 %.9g, |FP64| / sum|p| %.1e\n", i, j, r,
                        f2bf(float(r)), bfat(y, size_t(i) * n + j), f32at(yf, size_t(i) * n + j), std::fabs(r) / sa);
    }
    row(name, (ulp_info || bad == 0) && worst <= 1e-5, "%u x %u x %u: %ld BF16 outputs > 1 ulp from FP64%s; FP32 worst |err| / sum|products| %.1e (limit 1e-5)",
        m, n, k, bad, ulp_info ? " (INFO, see header)" : "", worst);
    for (auto *b : {&dx, &y, &yf, &ws}) R.ctx.free(*b);
}

// 2: shift / row-count invariance and repeat on (x, w buffer)
static void invariance(Run &R, const char *name, uint64_t dx, uint64_t w, uint64_t s, uint32_t M, uint32_t n, uint32_t k,
                       vkk2::GemmRoute route = vkk2::GemmRoute::kAuto) {
    auto y = kt::make(R.ctx, size_t(M) * n * 2), y2 = kt::make(R.ctx, size_t(M) * n * 2), ws = kt::make(R.ctx, vkk2::gemm_workspace_bytes(M, n));
    R.gemm(dx, w, s, y.addr, M, n, k, false, ws.addr, 0, route);
    long bad = 0, badm = 0;
    for (uint32_t s0 : {1u, 17u, 64u, 100u}) {
        R.gemm(dx + uint64_t(s0) * k * 2, w, s, y2.addr, M - s0, n, k, false, ws.addr, 0, route);
        bad += ndiff16(y2, 0, y, size_t(s0) * n * 2, size_t(M - s0) * n);
    }
    for (uint32_t m : {1u, 37u, 128u}) {
        R.gemm(dx, w, s, y2.addr, m, n, k, false, ws.addr, 0, route);
        badm += ndiff16(y2, 0, y, 0, size_t(m) * n);
    }
    R.gemm(dx, w, s, y2.addr, M, n, k, false, ws.addr, 0, route);
    const bool rep = same(y, 0, y2, 0, size_t(M) * n * 2);
    row(name, bad == 0 && badm == 0 && rep, "%u x %u x %u: shifted calls (1, 17, 64, 100) %ld differ; m 1/37/128 %ld differ; repeat %s",
        M, n, k, bad, badm, rep ? "bit-equal" : "DIFFERS");
    for (auto *b : {&y, &y2, &ws}) R.ctx.free(*b);
}

struct Timing { double med, mn, mx; };
static Timing time_calls(Context &ctx, const std::function<void()> &call, int reps) {
    ctx.begin();
    for (int i = 0; i < 2; ++i) { call(); ctx.barrier(); }
    for (int i = 0; i < reps; ++i) { ctx.stamp(2 * i); call(); ctx.stamp(2 * i + 1); ctx.barrier(); }
    ctx.submit_wait();
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
    std::sort(t.begin(), t.end());
    return {t[t.size() / 2], t.front(), t.back()};
}

static uint64_t g_xs = 0x9E3779B97F4A7C15ull;
static uint64_t xs() { g_xs ^= g_xs << 13; g_xs ^= g_xs >> 7; g_xs ^= g_xs << 17; return g_xs; }

int main(int argc, char **argv) {
    bool timing = false;
    std::string spv = "build/vulkan/spv";
    for (int i = 1; i < argc; ++i) {
        if (kt::hook_arg(argc, argv, i)) continue;  // VulkanAutotune: --identity DIR, --samples (ktest_util.h)
        if (std::string(argv[i]) == "--timing") timing = true; else spv = argv[i];
    }
    kt::tune_keys({});
    auto ctx_owner = cere::vk::testing::create_or_die(spv);
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s, coopmat %d\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(),
                ctx.info().driver_info.c_str(), int(ctx.info().coopmat));
    {
        const cere_status st = vkk2::gemm_probe(ctx);
        row("accumulator probe", st == CERE_OK, "FP32 16x16 accumulator: element e of lane l = (2 e + l / 16, l %% 16): %s", st == CERE_OK ? "yes" : "NO (GEMM refused)");
        if (st != CERE_OK) { std::printf("FAIL (%d failing)\n", g_fail); return 1; }
    }
    Run R{ctx};
    // P4-B rows below exercise the split route: k2_gemm tail_split on (the library default is off, row 4e).
    ctx.set_k2_tail_split(true);
    std::mt19937 g(23); std::normal_distribution<float> nd;
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };

    {   // 1. vs FP64, BF16 and MXFP8 weights
        const uint32_t m = 200, n = 300, k = 256;
        auto x = rnd(size_t(m) * k, 1.f), w = rnd(size_t(n) * k, 0.1f);
        auto dw = kt::upload(ctx, w);
        vs_fp64(R, "vs FP64 (BF16 w)", x, w, dw.addr, 0, m, n, k);
        Mx mq = quantize_mx(w, n, k);  // the same N(0, 0.1) weights, MXFP8-quantized
        auto dq = kt::upload(ctx, mq.q), ds = kt::upload(ctx, mq.s);
        vs_fp64(R, "vs FP64 (MXFP8 w)", x, mq.up, dq.addr, ds.addr, m, n, k);
        Mx mw = make_mx(g, n, k, 112, 122); upcast(mw, k);  // k2mx_test's uniform random bytes
        auto dr = kt::upload(ctx, mw.q), dsr = kt::upload(ctx, mw.s);
        vs_fp64(R, "vs FP64 (MX random bytes)", x, mw.up, dr.addr, dsr.addr, m, n, k, vkk2::GemmRoute::kAuto, true);
        vs_fp64(R, "FP32 route vs FP64", x, w, dw.addr, 0, m, n, k, vkk2::GemmRoute::kFp32);
        vs_fp64(R, "FP32 route vs FP64 (MX)", x, mq.up, dq.addr, ds.addr, m, n, k, vkk2::GemmRoute::kFp32);
        vs_fp64(R, "FP32 route (MX random)", x, mw.up, dr.addr, dsr.addr, m, n, k, vkk2::GemmRoute::kFp32, true);
        for (auto *b : {&dw, &dq, &ds, &dr, &dsr}) ctx.free(*b);
    }
    {   // 2. shift / row-count invariance, repeats
        const uint32_t M = 512, n = 10240, k = 2048;
        auto x = rnd(size_t(M) * k, 1.f), w = rnd(size_t(n) * k, 0.05f);
        auto dx = kt::upload(ctx, x), dw = kt::upload(ctx, w);
        invariance(R, "shift invariance (BF16)", dx.addr, dw.addr, 0, M, n, k);
        Mx mw = make_mx(g, n, k, 112, 122);
        auto dq = kt::upload(ctx, mw.q), ds = kt::upload(ctx, mw.s);
        invariance(R, "shift invariance (MXFP8)", dx.addr, dq.addr, ds.addr, M, n, k);
        invariance(R, "shift inv. (FP32 route)", dx.addr, dw.addr, 0, M, 2048, k, vkk2::GemmRoute::kFp32);
        for (auto *b : {&dx, &dw, &dq, &ds}) ctx.free(*b);
    }
    {   // 3. MXFP8 == BF16 upcast (with flagged rows in both)
        const uint32_t M = 200, n = 300, k = 2560;
        Mx mw = make_mx(g, n, k, 112, 122);
        for (uint32_t b = 0; b < k / 32; ++b) mw.s[size_t(7) * (k / 32) + b] = uint8_t(b % 2 ? 187 : 67);  // w row 7: 2^120 spread -> flagged
        upcast(mw, k);
        auto x = rnd(size_t(M) * k, 1.f);
        x[size_t(5) * k + 9] = f2bf(std::ldexp(1.f, -50));  // x row 5: split (one tail value)
        x[size_t(6) * k + 3] = f2bf(std::ldexp(1.f, 110));  // x row 6: FB (E > 100)
        auto dx = kt::upload(ctx, x), dq = kt::upload(ctx, mw.q), ds = kt::upload(ctx, mw.s), dwb = kt::upload(ctx, mw.up);
        auto ya = kt::make(ctx, size_t(M) * n * 4), yb = kt::make(ctx, size_t(M) * n * 4), ws = kt::make(ctx, vkk2::gemm_workspace_bytes(M, n));
        long diff = 0;
        for (uint32_t m : {17u, 64u, 200u})
            for (bool f32 : {false, true}) {
                R.gemm(dx.addr, dq.addr, ds.addr, ya.addr, m, n, k, f32, ws.addr);
                R.gemm(dx.addr, dwb.addr, 0, yb.addr, m, n, k, f32, ws.addr);
                diff += !same(ya, 0, yb, 0, size_t(m) * n * (f32 ? 4 : 2));
            }
        R.gemm(dx.addr, dq.addr, ds.addr, ya.addr, M, n, k, true, ws.addr);
        const uint32_t fx = kt::host<uint32_t>(ws, 1, 5 * 4)[0] >> 30, fx6 = kt::host<uint32_t>(ws, 1, 6 * 4)[0] >> 30,
                       fw = kt::host<uint32_t>(ws, 1, vkk2::gemm_workspace_bytes(M, 0) + 7 * 4)[0] >> 30;
        row("MXFP8 == BF16 upcast", diff == 0 && fx == 1 && fx6 == 2 && fw == 2,
            "%u x %u, m 17 / 64 / 200, BF16 + FP32 out: %ld calls differ; rows in the data: x row 5 %s, x row 6 %s, w row 7 %s",
            n, k, diff, fx == 1 ? "split" : "NOT split", fx6 == 2 ? "FB" : "NOT FB", fw == 2 ? "FB" : "NOT FB");
        for (auto *b : {&dx, &dq, &ds, &dwb, &ya, &yb, &ws}) ctx.free(*b);
    }
    // 4. exact fallback. Variants: A = split / FB x rows and a split w row (FB on the w side: the FP32 route's
    // full-tile path), B = the same x rows and no flagged w row (its sparse path: <= 32 FB x rows per tile), C = 43 FB
    // x rows in the tile and no flagged w row (full-tile path through x rows alone), D = 43 split x rows (two chunks).
    for (int v = 0; v < 4; ++v) {
        const uint32_t m = 64, n = 96, k = 256;
        const char V = char('A' + v);
        auto nm = [&](const char *what) { static char b[64]; std::snprintf(b, sizeof b, "fallback %c: %s", V, what); return b; };
        auto x = rnd(size_t(m) * k, 1.f), w = rnd(size_t(n) * k, 0.1f);
        auto X = [&](uint32_t r, uint32_t i) -> uint16_t & { return x[size_t(r) * k + i]; };
        auto W = [&](uint32_t r, uint32_t i) -> uint16_t & { return w[size_t(r) * k + i]; };
        std::vector<uint32_t> want_x = {30, 40}, want_s = {3, 11}, want_w;
        X(3, 1) = f2bf(std::ldexp(1.f, -45));                                       // 2^-45 among N(0,1): split
        for (uint32_t i = 0; i < k; ++i) { X(10, i) = 0; X(11, i) = 0; }
        X(10, 0) = f2bf(1.f); X(10, 1) = f2bf(std::ldexp(1.f, -36));               // exact: 2^-36 -> FP16 subnormal 2^-21
        X(11, 0) = f2bf(1.f); X(11, 1) = f2bf(std::ldexp(1.f, -45));               // split: 2^-45 -> 2^-30 not in FP16
        W(20, 0) = 0; W(20, 1) = f2bf(1.f);                                        // y[10][20] = 2^-36, y[11][20] = 2^-45
        X(30, 2) = 0x7F80;                                                         // +Inf: flagged
        X(40, 0) = f2bf(std::ldexp(1.f, 110));                                     // E = 110 > 100: flagged
        if (v == 0) { W(50, 3) = f2bf(std::ldexp(1.f, -60)); want_w = {50}; }      // tiny weight: split -> FB on the w side
        if (v >= 2) {
            // C: 2^110 in 41 more rows -> FB; D: 2^-50 -> split
            const uint16_t val = f2bf(std::ldexp(1.f, v == 2 ? 110 : -50));
            for (uint32_t r = 12; r < 30; ++r) X(r, 5) = val;
            for (uint32_t r = 41; r < 64; ++r) X(r, 5) = val;
            auto &l = v == 2 ? want_x : want_s;
            l.clear();
            for (uint32_t r = 0; r < m; ++r)
                if ((r >= 12 && r < 30) || r >= 41 || (v == 2 ? (r == 30 || r == 40) : (r == 3 || r == 11))) l.push_back(r);
        }
        auto dx = kt::upload(ctx, x), dw = kt::upload(ctx, w), y = kt::make(ctx, size_t(m) * n * 4), yb = kt::make(ctx, size_t(m) * n * 2),
             yr = kt::make(ctx, size_t(m) * n * 4), ws = kt::make(ctx, vkk2::gemm_workspace_bytes(m, n));
        R.gemm(dx.addr, dw.addr, 0, y.addr, m, n, k, true, ws.addr);
        auto info = kt::host<uint32_t>(ws, m + n);
        std::vector<uint32_t> fxs, sxs, fws;
        for (uint32_t r = 0; r < m; ++r) { if (info[r] >> 31) fxs.push_back(r); if ((info[r] >> 30) == 1) sxs.push_back(r); }
        auto winf = kt::host<uint32_t>(ws, n, vkk2::gemm_workspace_bytes(m, 0));
        for (uint32_t j = 0; j < n; ++j) if (winf[j] >> 30) fws.push_back(j);
        auto list = [](const std::vector<uint32_t> &l) { std::string s; for (auto r : l) s += std::to_string(r) + " "; return s; };
        row(nm("scan flags"), fxs == want_x && sxs == want_s && fws == want_w,
            "x rows FB {%s}, split {%s}, w rows FB/split {%s} (expected {%s}, {%s}, {%s}; x row 10 exact)", list(fxs).c_str(),
            list(sxs).c_str(), list(fws).c_str(), list(want_x).c_str(), list(want_s).c_str(), list(want_w).c_str());
        const float v10 = f32at(y, size_t(10) * n + 20), v11 = f32at(y, size_t(11) * n + 20);
        row(nm("fp16 subnormal"), v10 == std::ldexp(1.f, -36), "row 10 (1.0, 2^-36; scaled 2^-21, FP16 subnormal) on the matrix path: y = %a (exact 0x1p-36)", v10);
        row(nm("exact value"), v11 == std::ldexp(1.f, -45), "row 11 (1.0, 2^-45; scaled 2^-30, below FP16) on the split route%s: y = %a (exact 0x1p-45)",
            v == 0 ? " (w column 20 exact)" : "", v11);
        R.gemm(dx.addr, dw.addr, 0, yb.addr, m, n, k, false, ws.addr);
        long badb = 0, cls = 0; double worst = 0;
        for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) {
            double r, sa; ref(&x[size_t(i) * k], &w[size_t(j) * k], k, r, sa);
            const float got = f32at(y, size_t(i) * n + j);
            if (!std::isfinite(r)) { cls += !(std::isnan(r) ? std::isnan(got) : got == float(r)); continue; }
            worst = std::max(worst, std::fabs(got - r) / (sa + 1e-30));
            badb += std::abs(int(bfat(yb, size_t(i) * n + j)) - int(f2bf(float(r)))) > 1;
        }
        row(nm("vs FP64"), worst <= 1e-5 && badb == 0 && cls == 0, "all %u x %u outputs: FP32 worst %.1e (limit 1e-5), BF16 > 1 ulp %ld, non-finite outputs misclassified %ld (row 30 has +Inf)", m, n, worst, badb, cls);
        R.gemm(dx.addr, dw.addr, 0, yr.addr, m, n, k, true, ws.addr, 0, vkk2::GemmRoute::kFp32);
        long fd = 0, fcount = 0;
        for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) {
            if (!(info[i] >> 31) && !(winf[j] >> 30)) continue;
            ++fcount;
            fd += !same(y, (size_t(i) * n + j) * 4, yr, (size_t(i) * n + j) * 4, 4);
        }
        row(nm("== FP32 route"), fd == 0 && fcount > 0, "%ld outputs on FB x rows / FB or split w columns: %ld differ from the GemmRoute::kFp32 call (full-tile path)", fcount, fd);
        // invariance with mixed routes: rows 3.. and the first 12 rows, FP32 out
        auto y2 = kt::make(ctx, size_t(m) * n * 4);
        R.gemm(dx.addr + 3 * k * 2, dw.addr, 0, y2.addr, m - 3, n, k, true, ws.addr);
        const bool sh = same(y2, 0, y, size_t(3) * n * 4, size_t(m - 3) * n * 4);
        R.gemm(dx.addr, dw.addr, 0, y2.addr, 12, n, k, true, ws.addr);
        const bool rc = same(y2, 0, y, 0, size_t(12) * n * 4);
        R.gemm(dx.addr, dw.addr, 0, y2.addr, m, n, k, true, ws.addr);
        const bool rep = same(y2, 0, y, 0, size_t(m) * n * 4);
        row(nm("invariance"), sh && rc && rep, "mixed routes: rows 3.. call %s, m 12 call %s, repeat %s", sh ? "bit-equal" : "DIFFERS",
            rc ? "bit-equal" : "DIFFERS", rep ? "bit-equal" : "DIFFERS");
        for (auto *b : {&dx, &dw, &y, &yb, &yr, &ws, &y2}) ctx.free(*b);
    }
    {   // 4e. k2_gemm tail_split off (D7: the library default, P2's routing): a row the split route takes is FB and its
        // outputs are the FP32 route's bits; the w-side words do not depend on the lever.
        const uint32_t m = 64, n = 96, k = 256;
        auto x = rnd(size_t(m) * k, 1.f), w = rnd(size_t(n) * k, 0.1f);
        x[size_t(3) * k + 1] = f2bf(std::ldexp(1.f, -45));   // split when on
        x[size_t(11) * k + 0] = f2bf(1.f); x[size_t(11) * k + 1] = f2bf(std::ldexp(1.f, -45));
        w[size_t(50) * k + 3] = f2bf(std::ldexp(1.f, -60));  // tiny weight: FB on the w side either way
        auto dx = kt::upload(ctx, x), dw = kt::upload(ctx, w), y = kt::make(ctx, size_t(m) * n * 4),
             yr = kt::make(ctx, size_t(m) * n * 4), ws = kt::make(ctx, vkk2::gemm_workspace_bytes(m, n));
        R.gemm(dx.addr, dw.addr, 0, y.addr, m, n, k, true, ws.addr);
        const auto on_x = kt::host<uint32_t>(ws, m), on_w = kt::host<uint32_t>(ws, n, vkk2::gemm_workspace_bytes(m, 0));
        ctx.set_k2_tail_split(false);
        R.gemm(dx.addr, dw.addr, 0, y.addr, m, n, k, true, ws.addr);
        const auto off_x = kt::host<uint32_t>(ws, m), off_w = kt::host<uint32_t>(ws, n, vkk2::gemm_workspace_bytes(m, 0));
        R.gemm(dx.addr, dw.addr, 0, yr.addr, m, n, k, true, ws.addr, 0, vkk2::GemmRoute::kFp32);
        ctx.set_k2_tail_split(true);
        long split_off = 0, fd = 0;
        for (uint32_t r = 0; r < m; ++r) split_off += (off_x[r] >> 30) == 1;
        for (uint32_t r : {3u, 11u})
            for (uint32_t j = 0; j < n; ++j) fd += !same(y, (size_t(r) * n + j) * 4, yr, (size_t(r) * n + j) * 4, 4);
        const bool was_split = (on_x[3] >> 30) == 1 && (on_x[11] >> 30) == 1;
        const bool now_fb = (off_x[3] >> 30) == 2 && (off_x[11] >> 30) == 2;
        row("tail_split off: FB rows", was_split && now_fb && split_off == 0 && on_w == off_w && fd == 0,
            "rows 3 / 11: on %s, off %s; %ld split rows off; w words %s; %ld of %u outputs on rows 3 / 11 differ from the FP32 route",
            was_split ? "split" : "NOT split", now_fb ? "FB" : "NOT FB", split_off, on_w == off_w ? "equal" : "DIFFER", fd, 2 * n);
        for (auto *b : {&dx, &dw, &y, &yr, &ws}) ctx.free(*b);
    }
    {   // 4b. split limits: SPLIT_MAX tail values, extreme tails
        const uint32_t m = 40, n = 64, k = 512;
        auto x = rnd(size_t(m) * k, 1.f), w = rnd(size_t(n) * k, 0.1f);
        auto X = [&](uint32_t r, uint32_t i) -> uint16_t & { return x[size_t(r) * k + i]; };
        for (uint32_t r = 0; r < 3; ++r) for (uint32_t i = 0; i < k; ++i) X(r, i) = 0;
        X(0, 0) = f2bf(1.f); for (uint32_t i = 1; i <= 64; ++i) X(0, i * 7) = f2bf(std::ldexp(i % 2 ? 1.f : -1.5f, -50));  // 64 tail values: split
        X(1, 0) = f2bf(1.f); for (uint32_t i = 1; i <= 65; ++i) X(1, i * 7) = f2bf(std::ldexp(i % 2 ? 1.f : -1.5f, -50));  // 65: FB
        X(2, 0) = f2bf(1.f); X(2, 1) = f2bf(std::ldexp(1.f, -100)); X(2, 2) = 0x0001;  // 2^-100 and the BF16 subnormal 2^-133
        for (uint32_t t = 0; t < 10; ++t) X(3, 37 * t + 5) = f2bf(std::ldexp(float(nd(g)), -40));  // N(0,1) row + 10 tails
        for (uint32_t i = 0; i < k; ++i) { w[size_t(7) * k + i] = 0; w[size_t(8) * k + i] = 0; }
        w[size_t(7) * k + 1] = f2bf(1.f);                // y[2][7] = 2^-100
        w[size_t(8) * k + 2] = f2bf(std::ldexp(1.f, 20));  // y[2][8] = 2^-113 (product of an FP32 subnormal operand)
        auto dx = kt::upload(ctx, x), dw = kt::upload(ctx, w), y = kt::make(ctx, size_t(m) * n * 4), yb = kt::make(ctx, size_t(m) * n * 2),
             yr = kt::make(ctx, size_t(m) * n * 4), ws = kt::make(ctx, vkk2::gemm_workspace_bytes(m, n));
        R.gemm(dx.addr, dw.addr, 0, y.addr, m, n, k, true, ws.addr);
        auto info = kt::host<uint32_t>(ws, m);
        const uint32_t c0 = info[0] >> 30, c1 = info[1] >> 30, c2 = info[2] >> 30, c3 = info[3] >> 30;
        row("split: tail cap", c0 == 1 && c1 == 2 && c2 == 1 && c3 == 1, "64 tail values -> %s, 65 -> %s; 2^-100 + 2^-133 row -> %s; N(0,1) + 10 tails -> %s",
            c0 == 1 ? "split" : "NOT split", c1 == 2 ? "FB" : "NOT FB", c2 == 1 ? "split" : "NOT split", c3 == 1 ? "split" : "NOT split");
        R.gemm(dx.addr, dw.addr, 0, yr.addr, m, n, k, true, ws.addr, 0, vkk2::GemmRoute::kFp32);
        const float v7 = f32at(y, size_t(2) * n + 7), v8 = f32at(y, size_t(2) * n + 8), r8 = f32at(yr, size_t(2) * n + 8);
        row("split: extreme tails", v7 == std::ldexp(1.f, -100) && std::memcmp(&v8, &r8, 4) == 0,
            "2^-100 tail: y = %a (exact 0x1p-100); 2^-133 (BF16 subnormal) x 2^20: y = %a, FP32 route %a (same bits required; exact would be 0x1p-113)", v7, v8, r8);
        R.gemm(dx.addr, dw.addr, 0, yb.addr, m, n, k, false, ws.addr);
        long badb = 0; double worst = 0;
        for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) {
            if (i == 2 && j == 8) continue;  // the FP32-subnormal operand output above
            double r, sa; ref(&x[size_t(i) * k], &w[size_t(j) * k], k, r, sa);
            const double eij = std::fabs(f32at(y, size_t(i) * n + j) - r) / (sa + 1e-300);
            worst = std::max(worst, eij);
            badb += std::abs(int(bfat(yb, size_t(i) * n + j)) - int(f2bf(float(r)))) > 1;
        }
        row("split: vs FP64", worst <= 1e-5 && badb == 0, "%u x %u x %u: FP32 worst |err| / sum|p| %.1e (limit 1e-5), BF16 > 1 ulp %ld", m, n, k, worst, badb);
        for (auto *b : {&dx, &dw, &y, &yb, &yr, &ws}) ctx.free(*b);
    }
    {   // 4c. pool overflow: 40 split rows x 60 tails = 2,400 tail values in one tile (pool 640): later rows are listed
        // after the k loop ("late"); which rows are late depends on the batch, the bits must not
        const uint32_t m = 64, n = 192, k = 1024;
        auto x = rnd(size_t(m) * k, 1.f), w = rnd(size_t(n) * k, 0.1f);
        for (uint32_t r = 0; r < 40; ++r)
            for (uint32_t t = 0; t < 60; ++t) x[size_t(r) * k + (t * 17 + r) % k] = f2bf(std::ldexp(float(nd(g)), -45 - int(t % 30)));
        auto dx = kt::upload(ctx, x), dw = kt::upload(ctx, w), y = kt::make(ctx, size_t(m) * n * 4), y2 = kt::make(ctx, size_t(m) * n * 4),
             ws = kt::make(ctx, vkk2::gemm_workspace_bytes(m, n));
        R.gemm(dx.addr, dw.addr, 0, y.addr, m, n, k, true, ws.addr);
        auto info = kt::host<uint32_t>(ws, m);
        long sp = 0; for (uint32_t r = 0; r < 40; ++r) sp += (info[r] >> 30) == 1;
        double worst = 0;
        for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) {
            double r, sa; ref(&x[size_t(i) * k], &w[size_t(j) * k], k, r, sa);
            worst = std::max(worst, std::fabs(f32at(y, size_t(i) * n + j) - r) / (sa + 1e-300));
        }
        long bad = 0;
        for (uint32_t s0 : {1u, 13u, 30u}) {  // other pool / late assignments of the same rows
            R.gemm(dx.addr + uint64_t(s0) * k * 2, dw.addr, 0, y2.addr, m - s0, n, k, true, ws.addr);
            bad += !same(y2, 0, y, size_t(s0) * n * 4, size_t(m - s0) * n * 4);
        }
        for (uint32_t mm : {5u, 12u, 20u}) {
            R.gemm(dx.addr, dw.addr, 0, y2.addr, mm, n, k, true, ws.addr);
            bad += !same(y2, 0, y, 0, size_t(mm) * n * 4);
        }
        row("split: pool overflow", sp == 40 && worst <= 1e-5 && bad == 0,
            "40 rows x 60 tails (2,400 > pool 640): %ld split; FP32 worst |err| / sum|p| %.1e (limit 1e-5); shifted (1, 13, 30) and m 5 / 12 / 20 calls: %ld of 6 differ",
            sp, worst, bad);
        for (auto *b : {&dx, &dw, &y, &y2, &ws}) ctx.free(*b);
    }
    {   // 4d. P4-E: P4-A's device fact (k6_test "p4 probe2": on this RADV a coopMatMulAdd result depends on B values that
        // meet zeros of A; an all-zero A row is an exact no-op, a partly zero one is not). The GEMM's A is x (tile rows =
        // x rows), its B is w (tile columns = w rows), so the B values that meet a zero of x row r are w values (fixed per
        // call), and a row's bits stay a function of its own row and the w rows as long as no output depends on OTHER x
        // rows or OTHER w rows. These rows look for exactly that, at the zeros: genuine +0 / -0 values and the split
        // route's zeroed tails (k2_gemm.comp STAGE_X_SPLIT), in full and partial (m edge, clamped) tiles. k % 64 == 0 is
        // the GEMM's contract (k2_gemm_vk.cpp), so there is no partial k tile.
        const uint32_t m = 128, n = 160, k = 320;  // one m tile, two n tiles (the second partial), K3 hc-up's k
        auto x = rnd(size_t(m) * k, 1.f), w = rnd(size_t(n) * k, 0.1f);
        const std::vector<uint32_t> T = {0, 7, 37, 127};  // split targets: ~40 % zeros (+0 / -0) and 3 tails 2^-60..
        const uint32_t CL = 64;                           // a clean (exact, not split) target with the same zeros
        std::vector<char> zpos(k, 0);
        auto zero_at = [](uint32_t r, uint32_t i) { return (i * 7 + r * 3) % 5 < 2; };
        auto is_tail = [](uint16_t v) { return (v & 0x7FFF) != 0 && std::fabs(bf2f(v)) < 1e-15f; };
        for (uint32_t r : T) {
            uint32_t tails = 0;
            for (uint32_t i = 0; i < k; ++i) {
                if (!zero_at(r, i)) continue;
                zpos[i] = 1;
                x[size_t(r) * k + i] = i % 3 == 0 ? 0x8000 : 0;
                if (tails < 3 && i > 40 * (tails + 1)) {
                    x[size_t(r) * k + i] = f2bf(std::ldexp(tails % 2 ? -1.f : 1.f, -60 - int(tails)));
                    ++tails;
                }
            }
        }
        for (uint32_t i = 0; i < k; ++i) if (zero_at(CL, i)) { zpos[i] = 1; x[size_t(CL) * k + i] = i % 3 == 0 ? 0x8000 : 0; }
        auto target = [&](uint32_t r) { return r == CL || std::find(T.begin(), T.end(), r) != T.end(); };
        // x2: every other row's values at the targets' zero / tail positions replaced (large and small); x3: zeroed
        auto x2 = x, x3 = x;
        for (uint32_t r = 0; r < m; ++r) {
            if (target(r)) continue;
            for (uint32_t i = 0; i < k; ++i)
                if (zpos[i]) { x2[size_t(r) * k + i] = f2bf(nd(g) * std::ldexp(1.f, int(xs() % 13) - 6)); x3[size_t(r) * k + i] = 0; }
        }
        // x4: the targets' tails as +0 (no split row in the tile: the P2 k loop)
        auto x4 = x;
        for (uint32_t r : T) for (uint32_t i = 0; i < k; ++i) if (is_tail(x4[size_t(r) * k + i])) x4[size_t(r) * k + i] = 0;
        // w2: every w row other than the target columns J changed at the targets' zero positions
        const std::vector<uint32_t> J = {0, 5, 127, 130, 159};
        auto w2 = w;
        for (uint32_t j = 0; j < n; ++j) {
            if (std::find(J.begin(), J.end(), j) != J.end()) continue;
            for (uint32_t i = 0; i < k; ++i) if (zpos[i]) w2[size_t(j) * k + i] = f2bf(nd(g) * std::ldexp(1.f, int(xs() % 9) - 6));
        }
        auto dx = kt::upload(ctx, x), dx2 = kt::upload(ctx, x2), dx3 = kt::upload(ctx, x3), dx4 = kt::upload(ctx, x4),
             dw = kt::upload(ctx, w), dw2 = kt::upload(ctx, w2);
        auto y = kt::make(ctx, size_t(m) * n * 4), y2 = kt::make(ctx, size_t(m) * n * 4), yb = kt::make(ctx, size_t(m) * n * 2),
             yb2 = kt::make(ctx, size_t(m) * n * 2), ws = kt::make(ctx, vkk2::gemm_workspace_bytes(m, n));
        auto out_ne = [](const Buffer &a, const Buffer &b, size_t idx, size_t es) {
            return std::memcmp(static_cast<const uint8_t *>(a.map) + idx * es, static_cast<const uint8_t *>(b.map) + idx * es, es) != 0;
        };
        R.gemm(dx.addr, dw.addr, 0, y.addr, m, n, k, true, ws.addr);
        auto info = kt::host<uint32_t>(ws, m);
        long nsplit = 0, other_flag = 0;
        for (uint32_t r = 0; r < m; ++r) {
            if (std::find(T.begin(), T.end(), r) != T.end()) nsplit += (info[r] >> 30) == 1;
            else other_flag += (info[r] >> 30) != 0;
        }
        R.gemm(dx.addr, dw.addr, 0, yb.addr, m, n, k, false, ws.addr);
        long dn = 0;
        for (const Buffer *xb : {&dx2, &dx3}) {
            R.gemm(xb->addr, dw.addr, 0, y2.addr, m, n, k, true, ws.addr);
            R.gemm(xb->addr, dw.addr, 0, yb2.addr, m, n, k, false, ws.addr);
            for (uint32_t r = 0; r < m; ++r) {
                if (!target(r)) continue;
                dn += !same(y, size_t(r) * n * 4, y2, size_t(r) * n * 4, size_t(n) * 4);
                dn += !same(yb, size_t(r) * n * 2, yb2, size_t(r) * n * 2, size_t(n) * 2);
            }
        }
        row("zero-A: other x rows", nsplit == 4 && other_flag == 0 && dn == 0,
            "%u x %u x %u, %ld split targets (3 tails each) + 1 clean target, ~40 %% (+0 / -0) zeros; the other 123 rows' values "
            "at those positions replaced (x 2^-6..2^6) or zeroed: %ld of 20 target rows (2 variants, FP32 + BF16 out) differ",
            m, n, k, nsplit, dn);
        // the clean target in a tile without split rows (x4: the P2 k loop) == in the split tile (x); each split target ==
        // the same row with its tails as +0 on the P2 loop, plus the host's FP32 fma chain of the tails (exact scaling:
        // fl(y_clean + tail))
        R.gemm(dx4.addr, dw.addr, 0, y2.addr, m, n, k, true, ws.addr);
        auto info4 = kt::host<uint32_t>(ws, m);
        long flag4 = 0; for (uint32_t r = 0; r < m; ++r) flag4 += (info4[r] >> 30) != 0;
        const bool clean_same = same(y, size_t(CL) * n * 4, y2, size_t(CL) * n * 4, size_t(n) * 4);
        long dt = 0;
        for (uint32_t r : T) for (uint32_t j = 0; j < n; ++j) {
            float tl = 0.f;
            for (uint32_t i = 0; i < k; ++i) {
                const uint16_t v = x[size_t(r) * k + i];
                if (is_tail(v)) tl = std::fmaf(bf2f(v), bf2f(w[size_t(j) * k + i]), tl);
            }
            const float want = f32at(y2, size_t(r) * n + j) + tl, got = f32at(y, size_t(r) * n + j);
            dt += std::memcmp(&want, &got, 4) != 0;
        }
        row("zero-A: split == zeros", flag4 == 0 && clean_same && dt == 0,
            "clean target in a tile with split rows vs without (P2 k loop): %s; 4 split targets vs the same rows with the tails "
            "as +0 (P2 loop) + host fma chain of the tails: %ld of %u FP32 outputs differ", clean_same ? "bit-equal" : "DIFFERS", dt, 4 * n);
        // other w rows at the zeros: the target columns J of every row
        R.gemm(dx.addr, dw2.addr, 0, y2.addr, m, n, k, true, ws.addr);
        R.gemm(dx.addr, dw2.addr, 0, yb2.addr, m, n, k, false, ws.addr);
        long dc = 0;
        for (uint32_t r = 0; r < m; ++r) for (uint32_t j : J) dc += out_ne(y, y2, size_t(r) * n + j, 4) + out_ne(yb, yb2, size_t(r) * n + j, 2);
        row("zero-A: other w rows", dc == 0, "w rows other than columns 0 / 5 / 127 / 130 / 159 changed at the targets' zero "
            "positions: %ld of %u outputs of those columns (all 128 rows, FP32 + BF16) differ", dc, 2 * m * uint32_t(J.size()));
        // partial m tiles: a target as the last row (the padding rows are clamped copies of it) and at other positions
        long de = 0;
        const uint32_t edges[][2] = {{0, 38}, {37, 1}, {37, 91}, {30, 8}, {7, 30}, {120, 8}, {127, 1}, {100, 28}};
        for (const auto &c : edges) {
            R.gemm(dx.addr + uint64_t(c[0]) * k * 2, dw.addr, 0, y2.addr, c[1], n, k, true, ws.addr);
            for (uint32_t r = c[0]; r < c[0] + c[1]; ++r)
                if (target(r)) de += !same(y2, size_t(r - c[0]) * n * 4, y, size_t(r) * n * 4, size_t(n) * 4);
        }
        row("zero-A: m edges", de == 0, "targets as the last row of m 38 / 1 / 8 / 28 calls (padding clamped to them) and the "
            "first row of m 91 / 30 / 8 calls: %ld target rows differ from the m 128 call", de);
        // INFO 1: the sign of an exact zero in x (+0 vs -0 at the clean target's zeros, the P2 loop)
        auto x5 = x4;
        for (uint32_t i = 0; i < k; ++i) if (zero_at(CL, i)) x5[size_t(CL) * k + i] = 0x8000;
        auto dx5 = kt::upload(ctx, x5);
        R.gemm(dx4.addr, dw.addr, 0, y.addr, m, n, k, true, ws.addr);
        R.gemm(dx5.addr, dw.addr, 0, y2.addr, m, n, k, true, ws.addr);
        long ds = 0; for (uint32_t j = 0; j < n; ++j) ds += out_ne(y, y2, size_t(CL) * n + j, 4);
        // INFO 2: the fact with the roles swapped: a row's x values under zeros of w row 9 changed (mathematically no-ops)
        auto w3 = w;
        for (uint32_t i = 1; i < k; i += 2) w3[size_t(9) * k + i] = 0;
        auto x7 = x4;
        for (uint32_t r = 0; r < m; ++r) if (!target(r)) for (uint32_t i = 1; i < k; i += 2) x7[size_t(r) * k + i] = f2bf(nd(g) * 8.f);
        auto dw3 = kt::upload(ctx, w3), dx7 = kt::upload(ctx, x7);
        R.gemm(dx4.addr, dw3.addr, 0, y.addr, m, n, k, true, ws.addr);
        R.gemm(dx7.addr, dw3.addr, 0, y2.addr, m, n, k, true, ws.addr);
        long dsw = 0, nsw = 0;
        for (uint32_t r = 0; r < m; ++r) if (!target(r)) { ++nsw; dsw += out_ne(y, y2, size_t(r) * n + 9, 4); }
        std::printf("%-26s INFO  +0 vs -0 at the clean target's zeros: %ld of %u FP32 outputs differ; each row's x values under "
                    "the zeros of w row 9 changed: %ld of %ld outputs y[r][9] differ (a function of the row and the w row: no "
                    "invariance claim covers it)\n", "zero-A: facts", ds, n, dsw, nsw);
        for (auto *b : {&dx, &dx2, &dx3, &dx4, &dx5, &dx7, &dw, &dw2, &dw3, &y, &y2, &yb, &yb2, &ws}) ctx.free(*b);
    }
    {   // 6. the 16 / 17 boundary and run()'s routing
        const uint32_t M = 40, n = 512, k = 2560;
        auto x = rnd(size_t(M) * k, 1.f), w = rnd(size_t(n) * k, 0.05f);
        auto dx = kt::upload(ctx, x), dw = kt::upload(ctx, w), y = kt::make(ctx, size_t(M) * n * 4), y2 = kt::make(ctx, size_t(M) * n * 4),
             ws = kt::make(ctx, vkk2::gemm_workspace_bytes(M, n));
        long d = 0;
        for (bool f32 : {false, true}) {
            const size_t es = f32 ? 4 : 2;
            R.run(dx.addr, dw.addr, 0, y.addr, M, n, k, f32, 0, ws.addr);
            R.run(dx.addr, dw.addr, 0, y2.addr, 7, n, k, f32, M, ws.addr);
            d += !same(y2, 0, y, 0, size_t(7) * n * es);
            R.run(dx.addr + 33 * k * 2, dw.addr, 0, y2.addr, 7, n, k, f32, M, ws.addr);
            d += !same(y2, 0, y, size_t(33) * n * es, size_t(7) * n * es);
        }
        row("cls routing (class 40)", d == 0, "m 7 calls with cls 40 (rows 0.. and 33..) vs the 40-row call, BF16 + FP32 out: %ld of 4 differ", d);
        long dg = 0;
        R.run(dx.addr, dw.addr, 0, y.addr, 16, n, k, true, 0, ws.addr);
        for (uint32_t r = 0; r < 16; ++r) {
            R.run(dx.addr + uint64_t(r) * k * 2, dw.addr, 0, y2.addr, 1, n, k, true, 0, ws.addr);
            dg += !same(y2, 0, y, size_t(r) * n * 4, size_t(n) * 4);
        }
        row("class <= 16 rows (GEMV)", dg == 0, "rows of the m 16 call vs m 1 calls: %ld of 16 differ", dg);
        // INFO: the GEMV (class 16) and GEMM (class 17) on the same 16 rows
        R.run(dx.addr, dw.addr, 0, y2.addr, 17, n, k, true, 0, ws.addr);
        long di = 0; double dmax = 0;
        for (size_t i = 0; i < size_t(16) * n; ++i) {
            const float a = f32at(y, i), b = f32at(y2, i);
            di += a != b;
            double r, sa; ref(&x[(i / n) * k], &w[(i % n) * k], k, r, sa);
            dmax = std::max(dmax, std::fabs(double(a) - b) / (sa + 1e-30));
        }
        std::printf("%-26s INFO  GEMV (m 16) vs GEMM (m 17) on rows 0..15, FP32 out: %ld of %u outputs differ, max |diff| / sum|p| %.1e (different summation orders, as on HIP)\n",
                    "16/17 boundary", di, 16 * n, dmax);
        // refusals
        const cere_status a = vkk2::run(ctx, dx.addr, dw.addr, 0, y.addr, 7, n, k, false, 5, ws.addr);
        const cere_status b = vkk2::run(ctx, dx.addr, dw.addr, 0, y.addr, 20, n, 2576, false, 0, ws.addr);
        const cere_status c = vkk2::gemm(ctx, dx.addr, dw.addr, 0, y.addr, 20, n, 2576, false, ws.addr);
        const cere_status e = vkk2::gemm(ctx, dx.addr + 8, dw.addr, 0, y.addr, 20, n, k, false, ws.addr);
        cere_status f;
        { auto ctx2_owner = cere::vk::testing::create_or_die(spv); cere::vk::Context &ctx2 = *ctx2_owner; f = vkk2::gemm(ctx2, dx.addr, dw.addr, 0, y.addr, 20, n, k, false, ws.addr); }
        row("refusals", a == CERE_ERR_ARGUMENT && b == CERE_ERR_UNSUPPORTED && c == CERE_ERR_ARGUMENT && e == CERE_ERR_ARGUMENT && f == CERE_ERR_UNSUPPORTED,
            "cls < m -> %d, run k %% 64 != 0 -> %d, gemm k %% 64 != 0 -> %d, misaligned x -> %d, unprobed context -> %d", int(a), int(b), int(c), int(e), int(f));
        for (auto *bb : {&dx, &dw, &y, &y2, &ws}) ctx.free(*bb);
    }
    {   // 7. fallback frequency on synthetic activations (m 4,096, k 2,560): rows the scan flags
        const uint32_t m = 4096, k = 2560;
        std::mt19937 gg(5); std::normal_distribution<double> N;
        std::uniform_real_distribution<double> U(0.5, 1.5);
        auto silu = [](double v) { return v / (1 + std::exp(-v)); };
        struct D { const char *name; std::function<double(uint32_t)> f; };
        std::vector<double> gamma(k); for (auto &v : gamma) v = U(gg);
        for (uint32_t c : {7u, 300u, 1555u, 2048u}) gamma[c] = 60;
        const std::vector<D> ds = {
            {"gaussian N(0,1)", [&](uint32_t) { return N(gg); }},
            {"rmsnorm out + 4 outlier ch", [&](uint32_t c) { return N(gg) * gamma[c]; }},
            {"student-t nu=2", [&](uint32_t) { double a = N(gg), b = N(gg), c2 = N(gg); return a / std::sqrt((b * b + c2 * c2) / 2); }},
            {"swiglu, gate sd 1", [&](uint32_t) { return silu(N(gg)) * N(gg); }},
            {"swiglu, gate sd 4", [&](uint32_t) { return silu(4 * N(gg)) * N(gg); }},
            {"swiglu, gate sd 8", [&](uint32_t) { return silu(8 * N(gg)) * N(gg); }},
        };
        auto info = kt::make(ctx, size_t(m) * 4);
        for (const auto &dd : ds) {
            std::vector<uint16_t> x(size_t(m) * k);
            for (uint32_t r = 0; r < m; ++r) for (uint32_t c = 0; c < k; ++c) x[size_t(r) * k + c] = f2bf(float(dd.f(c)));
            auto dx = kt::upload(ctx, x);
            ctx.begin(); kt::must(vkk2::gemm_scan_rows(ctx, dx.addr, m, k, info.addr), "scan"); ctx.barrier(); ctx.submit_wait();
            auto inf = kt::host<uint32_t>(info, m);
            long fl = 0, sp = 0; for (auto v : inf) { fl += v >> 31; sp += (v >> 30) == 1; }
            const Timing t = time_calls(ctx, [&] { kt::must(vkk2::gemm_scan_rows(ctx, dx.addr, m, k, info.addr), "scan"); }, 9);
            std::printf("%-26s INFO  %-28s %ld of %u rows FB (%.3f %%), %ld split (%.3f %%); scan of 4096 x 2560 %.3f ms (%.0f GB/s)\n", "fallback frequency",
                        dd.name, fl, m, 100.0 * fl / m, sp, 100.0 * sp / m, t.med, double(m) * k * 2 / (t.med * 1e-3) / 1e9);
            ctx.free(dx);
        }
        ctx.free(info);
    }

    if (kt::hooks().samples) {   // 9. VulkanAutotune k2_gemm space: the S3 prefill shapes (vulkan_autotune.py PREFILL_GEMM)
        // at m 256 and 4,096, each in its production format (MXFP8 or BF16, prepared weight info), >= 15 samples each;
        // --identity: each shape's output bytes.
        struct S { uint32_t n, k; bool mx; const char *name; } shapes[] = {
            {10240, 2560, true, "gdn_qkv"}, {6144, 2560, true, "gdn_z"}, {13312, 2560, true, "attn_qkv"},
            {2560, 6144, true, "out_k6144"}, {320, 10240, false, "hc_down"}, {10240, 320, false, "hc_up"}};
        const uint32_t MM = 4096;
        for (auto sh : shapes) {
            std::vector<uint16_t> hx(size_t(MM) * sh.k), hw(sh.mx ? 0 : size_t(sh.n) * sh.k);
            std::vector<uint8_t> hq(sh.mx ? size_t(sh.n) * sh.k : 0), hs(sh.mx ? size_t(sh.n) * sh.k / 32 : 0);
            for (auto &v : hx) v = uint16_t(0x3C00 + (xs() & 0x03FF) + ((xs() & 1) << 15));
            for (auto &v : hw) v = uint16_t(0x3C00 + (xs() & 0x03FF) + ((xs() & 1) << 15));
            for (auto &b : hq) { b = uint8_t(xs()); if ((b & 0x7f) == 0x7f) b ^= 1; }
            for (auto &b : hs) b = uint8_t(112 + xs() % 11);
            auto dx = kt::upload(ctx, hx);
            auto dw = sh.mx ? kt::upload(ctx, hq) : kt::upload(ctx, hw);
            auto ds = sh.mx ? kt::upload(ctx, hs) : kt::make(ctx, 16);
            auto y = kt::make(ctx, size_t(MM) * sh.n * 2), ws = kt::make(ctx, vkk2::gemm_workspace_bytes(MM, sh.n)),
                 wi = kt::make(ctx, size_t(sh.n) * 4);
            const uint64_t sa = sh.mx ? ds.addr : 0;
            ctx.begin();
            kt::must(vkk2::gemm_prepare_weight(ctx, dw.addr, sa, sh.n, sh.k, wi.addr), "prep");
            ctx.barrier(); ctx.submit_wait();
            for (uint32_t m : {256u, 4096u}) {
                auto call = [&] { kt::must(vkk2::gemm(ctx, dx.addr, dw.addr, sa, y.addr, m, sh.n, sh.k, false, ws.addr, wi.addr), "g"); };
                const int reps = 15;
                ctx.begin();
                for (int i = 0; i < 2; ++i) { call(); ctx.barrier(); }
                for (int i = 0; i < reps; ++i) { ctx.stamp(2 * i); call(); ctx.stamp(2 * i + 1); ctx.barrier(); }
                ctx.submit_wait();
                std::vector<double> t;
                for (int i = 0; i < reps; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
                const std::string key = std::string(sh.name) + ".m" + std::to_string(m);
                kt::samples(key, "warm", t);
                kt::identity(key, kt::host<uint8_t>(y, size_t(m) * sh.n * 2));
            }
            for (auto *b : {&dx, &dw, &ds, &y, &ws, &wi}) ctx.free(*b);
        }
    }

    if (timing) {   // 8. timing
        struct S { uint32_t n, k; const char *name; } shapes[] = {{10240, 2048, "in_proj_qkv"}, {6144, 2048, "in_proj_z"}, {2048, 6144, "out_proj"}, {2048, 2048, "2048 sq"}};
        const uint32_t MM = 4096;
        for (auto sh : shapes) {
            std::vector<uint16_t> hx(size_t(MM) * sh.k), hw(size_t(sh.n) * sh.k);
            std::vector<uint8_t> hq(size_t(sh.n) * sh.k), hs(size_t(sh.n) * sh.k / 32);
            for (auto &v : hx) v = uint16_t(0x3C00 + (xs() & 0x03FF) + ((xs() & 1) << 15));
            for (auto &v : hw) v = uint16_t(0x3C00 + (xs() & 0x03FF) + ((xs() & 1) << 15));
            for (auto &b : hq) { b = uint8_t(xs()); if ((b & 0x7f) == 0x7f) b ^= 1; }
            for (auto &b : hs) b = uint8_t(112 + xs() % 11);
            auto hx1 = hx; hx1[size_t(300) * sh.k + 5] = f2bf(std::ldexp(1.f, -60));  // one flagged x row
            auto dx = kt::upload(ctx, hx), dx1 = kt::upload(ctx, hx1), dw = kt::upload(ctx, hw), dq = kt::upload(ctx, hq), ds = kt::upload(ctx, hs);
            auto y = kt::make(ctx, size_t(MM) * sh.n * 2), ws = kt::make(ctx, vkk2::gemm_workspace_bytes(MM, sh.n)),
                 wib = kt::make(ctx, size_t(sh.n) * 4), wim = kt::make(ctx, size_t(sh.n) * 4);
            ctx.begin();
            kt::must(vkk2::gemm_prepare_weight(ctx, dw.addr, 0, sh.n, sh.k, wib.addr), "prep");
            kt::must(vkk2::gemm_prepare_weight(ctx, dq.addr, ds.addr, sh.n, sh.k, wim.addr), "prep");
            ctx.barrier(); ctx.submit_wait();
            for (uint32_t m : {17u, 64u, 512u, 4096u}) {
                const int reps = m >= 4096 ? 7 : 15;
                const double fl = 2.0 * m * sh.n * sh.k / 1e9;
                vkk2::GemmOptions f32o; f32o.route = vkk2::GemmRoute::kFp32;
                const Timing tb = time_calls(ctx, [&] { kt::must(vkk2::gemm(ctx, dx.addr, dw.addr, 0, y.addr, m, sh.n, sh.k, false, ws.addr, wib.addr), "g"); }, reps);
                const Timing tbs = time_calls(ctx, [&] { kt::must(vkk2::gemm(ctx, dx.addr, dw.addr, 0, y.addr, m, sh.n, sh.k, false, ws.addr, 0), "g"); }, reps);
                const Timing tm = time_calls(ctx, [&] { kt::must(vkk2::gemm(ctx, dx.addr, dq.addr, ds.addr, y.addr, m, sh.n, sh.k, false, ws.addr, wim.addr), "g"); }, reps);
                const Timing tf = time_calls(ctx, [&] { kt::must(vkk2::gemm(ctx, dx.addr, dw.addr, 0, y.addr, m, sh.n, sh.k, false, ws.addr, wib.addr, &f32o), "g"); }, m >= 4096 ? 5 : reps);
                std::printf("timing  %-11s m %4u n %5u k %4u: bf16 %.3f ms %.1f TFLOP/s [%.3f..%.3f] | bf16 w-scan in call %.3f ms %.1f | mxfp8 %.3f ms %.1f [%.3f..%.3f] | FP32 route %.3f ms %.1f",
                            sh.name, m, sh.n, sh.k, tb.med, fl / tb.med, tb.mn, tb.mx, tbs.med, fl / tbs.med, tm.med, fl / tm.med, tm.mn, tm.mx, tf.med, fl / tf.med);
                if (m >= 512) {
                    const Timing t1 = time_calls(ctx, [&] { kt::must(vkk2::gemm(ctx, dx1.addr, dw.addr, 0, y.addr, m, sh.n, sh.k, false, ws.addr, wib.addr), "g"); }, reps);
                    std::printf(" | 1 flagged x row %.3f ms %.1f", t1.med, fl / t1.med);
                }
                std::printf("\n");
                std::fflush(stdout);
            }
            for (auto *b : {&dx, &dx1, &dw, &dq, &ds, &y, &ws, &wib, &wim}) ctx.free(*b);
        }
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
