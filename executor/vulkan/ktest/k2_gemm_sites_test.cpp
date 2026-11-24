// VulkanAmd P4-B: the K2 large-m GEMM on the value distributions of the two prefill sites where the exact FP32
// fallback fired on the real model (an internal benchmark report), and their timing:
//   K3 hc_mix up projection: x = bf16(silu(raw / 4)) [m, 320] (k3_hc.comp MODE 0), w [10240, 320], FP32 out;
//   K5 out_proj:             x = bf16(rmsnorm(o) * g * sigmoid(z)) [m, 6144] (k1 gated norm), w [2560, 6144], BF16 out.
// Data: synthetic generators fitted to real-model captures (an internal benchmark report),
// or real captured rows (--k3 FILE / --k5 FILE: raw BF16 [rows, k], from a CERE_VK_K2_CAPTURE test build). Weights
// N(0, 0.02) BF16 (no weight row is flagged on the real artifact, P2 report). Rows, per site:
//   classification  INFO: rows the scan marks exact / SPLIT / FB (the FB share is the fallback rate);
//   vs FP64         256 rows (up to 128 split / FB rows first) x 512 columns: FP32 |err| / sum|p| <= 1e-5 (k2w_test's
//                   bound), BF16 out == RNE of the FP32 out bit for bit, BF16 rel-L2 vs FP64 <= 4e-3 (k2_test's); the
//                   BF16 1-ulp count is INFO here (these sums cancel: the P2 baseline misses it on the same outputs);
//   invariance      512 rows, all columns, BF16 and FP32 out: shifted calls (1, 17, 64, 100) and m 1 / 37 / 128 equal
//                   the full call bit for bit, repeat bit-equal;
//   negative        the same rows with a +Inf and a 2^110 row and a row of 2 x SPLIT_MAX tail values inserted: those
//                   rows are FB, their outputs equal the FP32 route's bits, the others are unchanged;
//   --timing        GPU-timestamp medians at m (default 3,000) of the auto route on the site data, the same rows with
//                   their tail values zeroed (no row split or FB: "as if there were no fallback"), and the FP32 route.
// The split route runs with k2_gemm tail_split on (Context::set_k2_tail_split, D7); --no-split runs the library's
// default routing (P2's: the SPLIT counts are 0 and the split rows show as FB). Usage: k2_gemm_sites_test [spv dir]
// [--timing | --only-timing] [--m M] [--k3 FILE] [--k5 FILE] [--real-dir DIR] [--no-split]. Exit 1 on any FAIL
// (--only-timing: the classification and timing lines only; --real-dir: only the per-site totals over a capture
// directory, see time_real_dir).
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <dirent.h>

#include "kernels/k2_gemm_vk.h"
#include "ktest_util.h"
#include "launch/vk_launch_testing.h"

using kt::bf2f;
using kt::f2bf;
namespace vkk2 = cere::vkk2;
using cere::vk::Buffer;
using cere::vk::Context;
static int g_fail = 0;

static void row(const char *name, bool ok, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
static void row(const char *name, bool ok, const char *fmt, ...) {
    char buf[600];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::printf("%-30s %s  %s\n", name, ok ? "PASS" : "FAIL", buf);
    std::fflush(stdout);
    if (!ok) ++g_fail;
}

static float f32at(const Buffer &b, size_t i) { float f; std::memcpy(&f, static_cast<const uint8_t *>(b.map) + 4 * i, 4); return f; }
static uint16_t bfat(const Buffer &b, size_t i) { uint16_t h; std::memcpy(&h, static_cast<const uint8_t *>(b.map) + 2 * i, 2); return h; }
static bool same(const Buffer &a, size_t ao, const Buffer &b, size_t bo, size_t bytes) {
    return std::memcmp(static_cast<const uint8_t *>(a.map) + ao, static_cast<const uint8_t *>(b.map) + bo, bytes) == 0;
}

static void gemm(Context &ctx, uint64_t x, uint64_t w, uint64_t y, uint32_t m, uint32_t n, uint32_t k, bool f32, uint64_t ws,
                 uint64_t winfo = 0, vkk2::GemmRoute route = vkk2::GemmRoute::kAuto) {
    vkk2::GemmOptions o; o.route = route;
    ctx.begin();
    kt::must(vkk2::gemm(ctx, x, w, 0, y, m, n, k, f32, ws, winfo, &o), "gemm");
    ctx.barrier();
    ctx.submit_wait();
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

// --- data -------------------------------------------------------------------------------------------------------
// Row statistics as k2_gemm_scan.comp computes them: E = max exponent, L = min lowest-set-bit exponent; tail count =
// nonzero values with exponent < E - 32 (k2_gemm_common.comp split_threshold).
struct RowStat { int E, L; uint32_t tail; bool nonfinite; };
static RowStat row_stat(const uint16_t *r, uint32_t k) {
    RowStat s{-100000, 100000, 0, false};
    for (uint32_t i = 0; i < k; ++i) {
        const uint32_t a = r[i] & 0x7FFF;
        if (!a) continue;
        const uint32_t ef = a >> 7, mt = a & 0x7F;
        if (ef == 0xFF) s.nonfinite = true;
        const int e = ef ? int(ef) - 127 : (31 - __builtin_clz(mt)) - 133;
        const int l = ef ? e - 7 + __builtin_ctz(mt | 0x80) : __builtin_ctz(mt) - 133;
        s.E = std::max(s.E, e); s.L = std::min(s.L, l);
    }
    if (s.E > -100000) for (uint32_t i = 0; i < k; ++i) {
        const uint32_t a = r[i] & 0x7FFF;
        if (a && a < (uint32_t(s.E + 95) << 7)) ++s.tail;
    }
    return s;
}

struct Site {
    const char *name;
    uint32_t n, k;
    bool f32;
    std::vector<uint16_t> x;  // [rows, k]
    uint32_t rows = 0;
};

static double silu(double v) { return v / (1 + std::exp(-v)); }
static double sigm(double v) { return 1 / (1 + std::exp(-v)); }

// K3 up: t = bf16(silu(a)), a = raw / 4 (the down projection's FP32 output / hc). Fitted to the S3 capture (report):
// row maxima 2^1 .. 2^5; 25 % of rows hold tail values, all negative (silu's left tail: a <= -24 gives |t| < 2^-29),
// 1 .. 64 per row (mean 2.9; 47 % one, 24 % two), with exponents from E - 33 down to E - 124 (a down to about -88).
static void gen_k3(Site &s, uint32_t rows, std::mt19937 &g) {
    std::normal_distribution<double> N;
    std::uniform_real_distribution<double> U(0, 1);
    std::exponential_distribution<double> X(1.0 / 15);
    std::geometric_distribution<int> G(1.0 / 2.9);
    s.rows = rows; s.x.resize(size_t(rows) * s.k);
    for (uint32_t r = 0; r < rows; ++r) {
        uint16_t *x = &s.x[size_t(r) * s.k];
        const double sd = 1.0 + 2.5 * U(g);
        for (uint32_t i = 0; i < s.k; ++i) x[i] = f2bf(float(silu(std::max(-20.0, N(g) * sd - 0.3))));
        if (U(g) < 0.25) {  // a tailed row
            int t = 1 + G(g);
            if (U(g) < 0.02) t = 17 + int(U(g) * 48);  // the rare 17 .. 64
            t = std::min(t, 64);
            for (int q = 0; q < t; ++q) x[uint32_t(U(g) * s.k) % s.k] = f2bf(float(silu(-(24.0 + std::min(X(g), 64.0)))));
        }
    }
}
// K5 out_proj: y = bf16(rmsnorm(o) * g * sigmoid(z)) per 128-wide head. Fitted to the S3 capture (report): row maxima
// 2^-4 .. 2^3; 5.75 % of rows hold tail values, 1 or 3 per row at fixed channels (whole layers: GDN layers 1 and 2
// have them in every row), at 2^-85 .. 2^-118 (sigmoid(z) of z about -60 .. -80), half of them negative.
static void gen_k5(Site &s, uint32_t rows, std::mt19937 &g) {
    std::normal_distribution<double> N;
    std::uniform_real_distribution<double> U(0, 1);
    s.rows = rows; s.x.resize(size_t(rows) * s.k);
    std::vector<double> gamma(128);
    for (auto &v : gamma) v = 0.5 + U(g);
    const uint32_t ch[3] = {1037, 2950, 4801};
    for (uint32_t r = 0; r < rows; ++r) {
        uint16_t *x = &s.x[size_t(r) * s.k];
        for (uint32_t i = 0; i < s.k; ++i) x[i] = f2bf(float(N(g) * gamma[i % 128] * sigm(std::max(-10.0, -1.0 + 2.5 * N(g)))));
        if (U(g) < 0.06) {
            const int t = U(g) < 0.5 ? 1 : 3;
            for (int q = 0; q < t; ++q) x[ch[q]] = f2bf(float(N(g) * gamma[ch[q] % 128] * sigm(-60.0 - 20.0 * U(g))));
        }
    }
}
static bool load(Site &s, const std::string &file, uint32_t rows) {
    FILE *f = std::fopen(file.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    const uint32_t have = uint32_t(bytes / (long(s.k) * 2));
    s.rows = std::min(have, rows);
    s.x.resize(size_t(s.rows) * s.k);
    const bool ok = std::fread(s.x.data(), 2, s.x.size(), f) == s.x.size();
    std::fclose(f);
    return ok && s.rows > 0;
}

static void ref(const uint16_t *xr, const uint16_t *wr, uint32_t k, double &s, double &sa) {
    s = 0; sa = 0;
    for (uint32_t t = 0; t < k; ++t) { const double p = double(bf2f(xr[t])) * bf2f(wr[t]); s += p; sa += std::fabs(p); }
}

static bool g_only_timing = false;
static void run_site(Context &ctx, Site &s, bool timing, uint32_t tm) {
    std::mt19937 g(77);
    std::normal_distribution<float> nd;
    const uint32_t n = s.n, k = s.k, M = s.rows;
    std::vector<uint16_t> w(size_t(n) * k);
    for (auto &v : w) v = f2bf(nd(g) * 0.02f);
    auto nm = [&](const char *what) { static char b[96]; std::snprintf(b, sizeof b, "%s: %s", s.name, what); return b; };

    // classification (host statistics and the scan's words)
    std::vector<RowStat> st(M);
    for (uint32_t r = 0; r < M; ++r) st[r] = row_stat(&s.x[size_t(r) * k], k);
    auto dx = kt::upload(ctx, s.x), dw = kt::upload(ctx, w), info = kt::make(ctx, size_t(M) * 4), winfo = kt::make(ctx, size_t(n) * 4);
    ctx.begin();
    kt::must(vkk2::gemm_scan_rows(ctx, dx.addr, M, k, info.addr), "scan x");
    kt::must(vkk2::gemm_prepare_weight(ctx, dw.addr, 0, n, k, winfo.addr), "scan w");
    ctx.barrier(); ctx.submit_wait();
    auto xi = kt::host<uint32_t>(info, M);
    auto wi = kt::host<uint32_t>(winfo, n);
    long fb = 0, sp = 0, wfl = 0, host_inexact = 0, tail_sum = 0;
    for (uint32_t r = 0; r < M; ++r) {
        fb += xi[r] >> 31; sp += (xi[r] >> 30) == 1;
        const bool inexact = st[r].E > -100000 && st[r].L < st[r].E - 39;
        host_inexact += inexact;
        if (inexact) tail_sum += st[r].tail;
    }
    for (uint32_t j = 0; j < n; ++j) wfl += (wi[j] >> 30) != 0;
    std::printf("%-30s INFO  %u rows, k %u: not exact as FP16 * 2^e %ld (%.2f %%); scan: FB %ld (%.2f %%), split %ld (%.2f %%); "
                "mean tail values per inexact row %.2f; weight rows flagged %ld\n",
                nm("classification"), M, k, host_inexact, 100.0 * host_inexact / M, fb, 100.0 * fb / M, sp, 100.0 * sp / M,
                host_inexact ? double(tail_sum) / host_inexact : 0.0, wfl);

    // vs FP64: the first 256 rows that are split / FB first, then others
    if (!g_only_timing) {
        std::vector<uint32_t> pick;
        for (uint32_t r = 0; r < M && pick.size() < 128; ++r) if (xi[r] >> 30) pick.push_back(r);
        for (uint32_t r = 0; r < M && pick.size() < 256; ++r) if (!(xi[r] >> 30)) pick.push_back(r);
        const uint32_t m = uint32_t(pick.size()), nn = std::min<uint32_t>(512, n);
        std::vector<uint16_t> xs(size_t(m) * k);
        for (uint32_t i = 0; i < m; ++i) std::memcpy(&xs[size_t(i) * k], &s.x[size_t(pick[i]) * k], size_t(k) * 2);
        auto ds = kt::upload(ctx, xs), yf = kt::make(ctx, size_t(m) * nn * 4), yb = kt::make(ctx, size_t(m) * nn * 2),
             ws = kt::make(ctx, vkk2::gemm_workspace_bytes(m, nn));
        gemm(ctx, ds.addr, dw.addr, yf.addr, m, nn, k, true, ws.addr);
        gemm(ctx, ds.addr, dw.addr, yb.addr, m, nn, k, false, ws.addr);
        long bad = 0, nsp = 0, rne = 0; double worst = 0, worst_sp = 0, e2 = 0, r2 = 0, bad_rel = 0;
        for (uint32_t i = 0; i < m; ++i) {
            const bool spl = (xi[pick[i]] >> 30) != 0;
            nsp += spl;
            for (uint32_t j = 0; j < nn; ++j) {
                double r, sa; ref(&xs[size_t(i) * k], &w[size_t(j) * k], k, r, sa);
                const double e = std::fabs(f32at(yf, size_t(i) * nn + j) - r) / (sa + 1e-300);
                worst = std::max(worst, e);
                if (spl) worst_sp = std::max(worst_sp, e);
                const uint16_t b = bfat(yb, size_t(i) * nn + j);
                if (std::abs(int(b) - int(f2bf(float(r)))) > 1) { ++bad; bad_rel = std::max(bad_rel, std::fabs(r) / (sa + 1e-300)); }
                rne += b != f2bf(f32at(yf, size_t(i) * nn + j));
                e2 += (double(bf2f(b)) - r) * (double(bf2f(b)) - r); r2 += r * r;
            }
        }
        const double rel = std::sqrt(e2 / (r2 + 1e-300));
        row(nm("vs FP64"), worst <= 1e-5 && rne == 0 && rel <= 4e-3,
            "%u rows (%ld split / FB) x %u x %u: FP32 worst |err| / sum|p| %.1e (split / FB rows %.1e; limit 1e-5); BF16 out == RNE(FP32 out) "
            "except %ld; BF16 rel-L2 vs FP64 %.1e (k2_test's limit 4e-3); INFO: BF16 > 1 ulp from FP64 %ld (largest |FP64| / sum|p| among them %.1e: cancellation)",
            m, nsp, nn, k, worst, worst_sp, rne, rel, bad, bad_rel);
        for (auto *b : {&ds, &yf, &yb, &ws}) ctx.free(*b);
    }

    // invariance: 512 rows, BF16 and FP32 out
    if (!g_only_timing) {
        const uint32_t MM = std::min<uint32_t>(512, M);
        long bad = 0, badm = 0; bool rep = true;
        for (bool f32 : {false, true}) {
            const size_t es = f32 ? 4 : 2;
            auto y = kt::make(ctx, size_t(MM) * n * es), y2 = kt::make(ctx, size_t(MM) * n * es), ws = kt::make(ctx, vkk2::gemm_workspace_bytes(MM, n));
            gemm(ctx, dx.addr, dw.addr, y.addr, MM, n, k, f32, ws.addr);
            for (uint32_t s0 : {1u, 17u, 64u, 100u}) {
                gemm(ctx, dx.addr + uint64_t(s0) * k * 2, dw.addr, y2.addr, MM - s0, n, k, f32, ws.addr);
                bad += !same(y2, 0, y, size_t(s0) * n * es, size_t(MM - s0) * n * es);
            }
            for (uint32_t m : {1u, 37u, 128u}) {
                gemm(ctx, dx.addr, dw.addr, y2.addr, m, n, k, f32, ws.addr);
                badm += !same(y2, 0, y, 0, size_t(m) * n * es);
            }
            gemm(ctx, dx.addr, dw.addr, y2.addr, MM, n, k, f32, ws.addr);
            rep = rep && same(y, 0, y2, 0, size_t(MM) * n * es);
            for (auto *b : {&y, &y2, &ws}) ctx.free(*b);
        }
        long spl = 0; for (uint32_t r = 0; r < MM; ++r) spl += (xi[r] >> 30) != 0;
        row(nm("invariance"), bad == 0 && badm == 0 && rep, "%u x %u x %u (%ld split / FB rows), BF16 + FP32 out: shifted calls (1, 17, 64, 100) %ld of 8 differ; m 1/37/128 %ld of 6 differ; repeat %s",
            MM, n, k, spl, bad, badm, rep ? "bit-equal" : "DIFFERS");
    }

    // negative: rows 5 (+Inf), 6 (2^110), 7 (2 x SPLIT_MAX tails) -> FB, == the FP32 route; other rows unchanged
    if (!g_only_timing) {
        const uint32_t MM = std::min<uint32_t>(256, M);
        std::vector<uint16_t> xn(s.x.begin(), s.x.begin() + size_t(MM) * k);
        xn[size_t(5) * k + 3] = 0x7F80;
        xn[size_t(6) * k + 9] = f2bf(std::ldexp(1.f, 110));
        for (uint32_t i = 0; i < k; ++i) xn[size_t(7) * k + i] = (i % 2) ? f2bf(std::ldexp(1.f, -60)) : f2bf(i % 4 ? 1.f : -1.f);
        auto dn = kt::upload(ctx, xn), y = kt::make(ctx, size_t(MM) * n * 4), yr = kt::make(ctx, size_t(MM) * n * 4), y0 = kt::make(ctx, size_t(MM) * n * 4),
             ws = kt::make(ctx, vkk2::gemm_workspace_bytes(MM, n));
        gemm(ctx, dn.addr, dw.addr, y.addr, MM, n, k, true, ws.addr);
        auto inf = kt::host<uint32_t>(ws, MM);
        gemm(ctx, dn.addr, dw.addr, yr.addr, MM, n, k, true, ws.addr, 0, vkk2::GemmRoute::kFp32);
        gemm(ctx, dx.addr, dw.addr, y0.addr, MM, n, k, true, ws.addr);
        const bool flags = (inf[5] >> 31) && (inf[6] >> 31) && (inf[7] >> 31);
        long dfb = 0, dother = 0;
        for (uint32_t r = 5; r < 8; ++r) dfb += !same(y, size_t(r) * n * 4, yr, size_t(r) * n * 4, size_t(n) * 4);
        for (uint32_t r = 0; r < MM; ++r) if (r < 5 || r > 7) dother += !same(y, size_t(r) * n * 4, y0, size_t(r) * n * 4, size_t(n) * 4);
        row(nm("negative rows"), flags && dfb == 0 && dother == 0, "+Inf / 2^110 / %u-tail rows: %s; their outputs vs the FP32 route: %ld of 3 rows differ; other %u rows vs the call without them: %ld differ",
            k / 2, flags ? "all FB" : "NOT all FB", dfb, MM - 3, dother);
        for (auto *b : {&dn, &y, &yr, &y0, &ws}) ctx.free(*b);
    }

    if (timing) {
        const uint32_t m = std::min(tm, M);
        std::vector<uint16_t> xc(s.x.begin(), s.x.begin() + size_t(m) * k);  // tails zeroed: every row exact
        long zeroed = 0;
        for (uint32_t r = 0; r < m; ++r) {
            const RowStat q = row_stat(&xc[size_t(r) * k], k);
            if (q.E > -100000 && q.L < q.E - 39)
                for (uint32_t i = 0; i < k; ++i) {
                    uint16_t &v = xc[size_t(r) * k + i];
                    if ((v & 0x7FFF) && (v & 0x7FFF) < (uint32_t(q.E + 95) << 7)) { v = 0; ++zeroed; }
                }
        }
        auto dc = kt::upload(ctx, xc), y = kt::make(ctx, size_t(m) * n * (s.f32 ? 4 : 2)), ws = kt::make(ctx, vkk2::gemm_workspace_bytes(m, n));
        long fbm = 0; for (uint32_t r = 0; r < m; ++r) fbm += xi[r] >> 31;
        long spm = 0; for (uint32_t r = 0; r < m; ++r) spm += (xi[r] >> 30) == 1;
        const double fl = 2.0 * m * n * k / 1e9;
        vkk2::GemmOptions f32o; f32o.route = vkk2::GemmRoute::kFp32;
        const Timing ta = time_calls(ctx, [&] { kt::must(vkk2::gemm(ctx, dx.addr, dw.addr, 0, y.addr, m, n, k, s.f32, ws.addr, winfo.addr), "g"); }, 9);
        const Timing tc = time_calls(ctx, [&] { kt::must(vkk2::gemm(ctx, dc.addr, dw.addr, 0, y.addr, m, n, k, s.f32, ws.addr, winfo.addr), "g"); }, 9);
        const Timing tf = time_calls(ctx, [&] { kt::must(vkk2::gemm(ctx, dx.addr, dw.addr, 0, y.addr, m, n, k, s.f32, ws.addr, winfo.addr, &f32o), "g"); }, 7);
        std::printf("timing  %-14s m %u n %u k %u %s out: site data (FB %ld = %.2f %%, split %ld = %.2f %%) %.3f ms %.1f TFLOP/s [%.3f..%.3f] | "
                    "tails zeroed (%ld values; no FB / split) %.3f ms %.1f [%.3f..%.3f] | FP32 route %.3f ms %.1f\n",
                    s.name, m, n, k, s.f32 ? "FP32" : "BF16", fbm, 100.0 * fbm / m, spm, 100.0 * spm / m, ta.med, fl / ta.med, ta.mn, ta.mx,
                    zeroed, tc.med, fl / tc.med, tc.mn, tc.mx, tf.med, fl / tf.med);
        std::fflush(stdout);
        for (auto *b : {&dc, &y, &ws}) ctx.free(*b);
    }
    for (auto *b : {&dx, &dw, &info, &winfo}) ctx.free(*b);
}

// --real-dir DIR: every capture file of a CERE_VK_K2_CAPTURE run (<seq>_n<n>_k<k>_m<m>.bf16, in call order): K3 up =
// k 320; n 2560 / k 6144 calls alternate GDN out_proj and attention o_proj by layer (layer % 4 == 3: attention, not
// this site). Per site: FB / split rows and the summed GPU time of the auto route and of the same rows with tails
// zeroed (median of 5 calls each), i.e. the prefill's total for that site.
static void time_real_dir(Context &ctx, const std::string &dir) {
    std::vector<std::string> files;
    if (DIR *d = opendir(dir.c_str())) {
        while (dirent *e = readdir(d)) if (std::strstr(e->d_name, ".bf16")) files.push_back(e->d_name);
        closedir(d);
    }
    std::sort(files.begin(), files.end());
    struct Acc { long calls = 0, rows = 0, fb = 0, sp = 0; double ms = 0, ms0 = 0; } k3a, k5a;
    std::mt19937 g(77);
    std::normal_distribution<float> nd;
    struct Wt { uint32_t n, k; Buffer w, info; };
    std::vector<Wt> wts;
    auto weight = [&](uint32_t n, uint32_t k) -> Wt & {
        for (auto &t : wts) if (t.n == n && t.k == k) return t;
        std::vector<uint16_t> w(size_t(n) * k);
        for (auto &v : w) v = f2bf(nd(g) * 0.02f);
        Wt t{n, k, kt::upload(ctx, w), kt::make(ctx, size_t(n) * 4)};
        ctx.begin(); kt::must(vkk2::gemm_prepare_weight(ctx, t.w.addr, 0, n, k, t.info.addr), "scan w"); ctx.barrier(); ctx.submit_wait();
        wts.push_back(t);
        return wts.back();
    };
    int k6144 = 0;
    for (const auto &f : files) {
        unsigned n = 0, k = 0, m = 0;
        const char *u = std::strchr(f.c_str(), '_');
        if (!u || std::sscanf(u + 1, "n%u_k%u_m%u", &n, &k, &m) != 3) continue;
        Acc *a = nullptr;
        if (k == 320 && n == 10240) a = &k3a;
        else if (k == 6144 && n == 2560) { if (k6144++ % 4 != 3) a = &k5a; }
        if (!a) continue;
        Site st{"", n, k, k == 320, {}};
        if (!load(st, dir + "/" + f, m) || st.rows != m) { std::printf("FAIL: cannot read %s\n", f.c_str()); ++g_fail; return; }
        std::vector<uint16_t> xc = st.x;
        for (uint32_t r = 0; r < m; ++r) {
            const RowStat q = row_stat(&xc[size_t(r) * k], k);
            if (q.E > -100000 && q.L < q.E - 39)
                for (uint32_t i = 0; i < k; ++i) {
                    uint16_t &v = xc[size_t(r) * k + i];
                    if ((v & 0x7FFF) && (v & 0x7FFF) < (uint32_t(q.E + 95) << 7)) v = 0;
                }
        }
        Wt &w = weight(n, k);
        auto dx = kt::upload(ctx, st.x), dc = kt::upload(ctx, xc), y = kt::make(ctx, size_t(m) * n * (st.f32 ? 4 : 2)),
             ws = kt::make(ctx, vkk2::gemm_workspace_bytes(m, n));
        ctx.begin(); kt::must(vkk2::gemm_scan_rows(ctx, dx.addr, m, k, ws.addr), "scan x"); ctx.barrier(); ctx.submit_wait();
        auto xi = kt::host<uint32_t>(ws, m);
        for (auto v : xi) { a->fb += v >> 31; a->sp += (v >> 30) == 1; }
        const Timing ta = time_calls(ctx, [&] { kt::must(vkk2::gemm(ctx, dx.addr, w.w.addr, 0, y.addr, m, n, k, st.f32, ws.addr, w.info.addr), "g"); }, 5);
        const Timing tc = time_calls(ctx, [&] { kt::must(vkk2::gemm(ctx, dc.addr, w.w.addr, 0, y.addr, m, n, k, st.f32, ws.addr, w.info.addr), "g"); }, 5);
        a->calls++; a->rows += m; a->ms += ta.med; a->ms0 += tc.med;
        for (auto *b : {&dx, &dc, &y, &ws}) ctx.free(*b);
    }
    for (auto *pa : {&k3a, &k5a}) {
        const Acc &a = *pa;
        std::printf("real    %-12s %ld calls, %ld rows: FB %ld (%.2f %%), split %ld (%.2f %%); auto route total %.3f ms (%.3f per call) | tails zeroed total %.3f ms (%.3f per call)\n",
                    pa == &k3a ? "K3 hc up" : "K5 out_proj", a.calls, a.rows, a.fb, 100.0 * a.fb / std::max(1L, a.rows), a.sp,
                    100.0 * a.sp / std::max(1L, a.rows), a.ms, a.ms / std::max(1L, a.calls), a.ms0, a.ms0 / std::max(1L, a.calls));
    }
    std::fflush(stdout);
    for (auto &t : wts) { ctx.free(t.w); ctx.free(t.info); }
}

int main(int argc, char **argv) {
    bool timing = false, split = true;
    uint32_t tm = 3000;
    std::string spv = "build/vulkan/spv", k3f, k5f, real_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--timing") timing = true;
        else if (a == "--only-timing") timing = g_only_timing = true;
        else if (a == "--m" && i + 1 < argc) tm = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else if (a == "--k3" && i + 1 < argc) k3f = argv[++i];
        else if (a == "--k5" && i + 1 < argc) k5f = argv[++i];
        else if (a == "--real-dir" && i + 1 < argc) real_dir = argv[++i];
        else if (a == "--no-split") split = false;
        else spv = a;
    }
    auto ctx_owner = cere::vk::testing::create_or_die(spv);
    Context &ctx = *ctx_owner;
    ctx.set_k2_tail_split(split);
    std::printf("device: %s, driver %s %s\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str());
    if (vkk2::gemm_probe(ctx) != CERE_OK) { std::printf("FAIL: accumulator probe\n"); return 1; }
    if (!real_dir.empty()) {
        time_real_dir(ctx, real_dir);
        std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
        return g_fail ? 1 : 0;
    }
    std::mt19937 g(2026);
    const uint32_t rows = std::max<uint32_t>(tm, 512);
    Site k3{"K3 hc up", 10240, 320, true, {}};
    Site k5{"K5 out_proj", 2560, 6144, false, {}};
    if (!k3f.empty()) { if (!load(k3, k3f, rows)) { std::printf("FAIL: cannot read %s\n", k3f.c_str()); return 1; } k3.name = "K3 hc up (real)"; }
    else gen_k3(k3, rows, g);
    if (!k5f.empty()) { if (!load(k5, k5f, rows)) { std::printf("FAIL: cannot read %s\n", k5f.c_str()); return 1; } k5.name = "K5 out_proj (real)"; }
    else gen_k5(k5, rows, g);
    run_site(ctx, k3, timing, tm);
    run_site(ctx, k5, timing, tm);
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
