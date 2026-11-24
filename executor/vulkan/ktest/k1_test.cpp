// VulkanAmd P2 K1: executor/hip/ktest/k1_test.cpp's contract rows on the Vulkan K1 primitives (shaders/k1_*.comp), the same
// shapes, seeds, data, references and bounds: tolerance rows rel-L2 <= 4e-3 vs FP64 (rope_table: max |err| <= 1e-4), embedding /
// top-2 / FP8 KV store bit-exact (against host references written from the kernel definitions), every kernel run twice and bit-identical.
// Then negative rows (the contract's refusals) and a decode-shape timing table (GPU timestamps, median of 15).
// Built with -DK1_HIP_TIMING under hipcc it is instead the same-method HIP timing of the same primitives (reference).
// Exit 0 when every row passes.
#ifdef K1_HIP_TIMING
#include <hip/hip_runtime.h>
#include "kernels/k1_primitives.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <tuple>
#include <random>
#include <vector>

#ifndef K1_HIP_TIMING
#include "kernels/k1_primitives_vk.h"
#include "launch/vk_launch.h"
#include "launch/vk_launch_testing.h"
#endif

static int g_fail = 0;
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) {  // round to nearest even, NaN kept quiet (the HIP test's)
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static std::vector<uint16_t> randbf(size_t n, std::mt19937 &g, float scale = 1.f) {
    std::normal_distribution<float> d(0.f, scale); std::vector<uint16_t> v(n);
    for (auto &x : v) x = f2bf(d(g));
    return v;
}

// The decode-path shapes timed on both sides (the decode step of the S3 model at m = 1; the HIP test's data shapes).
struct Shape { const char *name; };
static const char *kShapes[] = {"embedding_gather n=1 hidden 2560", "rmsnorm gemma grouped 4 rows x 2560 (hc_norm)", "rmsnorm gemma 24 rows x 256 (qk_norm)",
                                "rmsnorm plain 48 rows x 128 (gdn)", "rmsnorm_gated_sigmoid 48 rows x 128", "rope 1 tok, 8 heads, 64/256", "sigmoid_mul 1 x 6144",
                                "top2_split 1 row x 248320", "kv_store_fp8 1 tok x 512",
                                "qk_norm_rope_gate 1 tok, 24 q + 2 kv heads x 256", "kv_store_fp8_at 1 tok x 512", "kv_store_fp8_rows 16 rows x 512",
                                "copy_segments 6 x 5120 B flat", "top2 (token_score) 1 row x 248320", "top2 (token_score) 5 rows x 248320"};

static const int kBatch = 100;
static double median(std::vector<double> t) { std::sort(t.begin(), t.end()); return t[t.size() / 2]; }

#ifndef K1_HIP_TIMING
using namespace cere;
static void report(const char *name, bool ok, const char *detail) {
    std::printf("%-34s %s  %s\n", name, ok ? "PASS" : "FAIL", detail);
    if (!ok) ++g_fail;
}
static vk::Buffer make(vk::Context &ctx, uint64_t bytes) {
    vk::Buffer b;
    if (!ctx.alloc(bytes + 256, b)) std::exit(2);
    std::memset(b.map, 0, bytes + 256);
    return b;
}
template <class T> static vk::Buffer up(vk::Context &ctx, const std::vector<T> &h) {
    vk::Buffer b = make(ctx, h.size() * sizeof(T)); std::memcpy(b.map, h.data(), h.size() * sizeof(T)); return b;
}
template <class T> static std::vector<T> host(const vk::Buffer &b, size_t n) {
    std::vector<T> h(n); std::memcpy(h.data(), b.map, n * sizeof(T)); return h;
}
static void must(cere_status s, const char *what) { if (s != CERE_OK) { std::printf("FATAL %s returned %d\n", what, int(s)); std::exit(2); } }
// One recorded call, submitted and waited for.
static void sync(vk::Context &ctx, const std::function<cere_status()> &f, const char *what) { ctx.begin(); must(f(), what); ctx.barrier(); ctx.submit_wait(); }

static double rel_l2(const std::vector<uint16_t> &got, const std::vector<double> &ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) { const double d = bf2f(got[i]) - ref[i]; num += d * d; den += ref[i] * ref[i]; }
    return std::sqrt(num / (den > 0 ? den : 1));
}
static void tol(const char *name, const std::vector<uint16_t> &got, const std::vector<uint16_t> &again, const std::vector<double> &ref) {
    const double e = rel_l2(got, ref); char d[96];
    const bool det = got == again;
    std::snprintf(d, sizeof d, "rel-L2 %.2e (limit 4e-3), repeat bit-identical %s", e, det ? "yes" : "NO");
    report(name, e <= 4e-3 && det, d);
}

static void test_rmsnorm(vk::Context &ctx, std::mt19937 &g, const char *name, int64_t rows, int dim, int groups, bool gemma) {
    auto x = randbf(rows * dim, g, 2.f), w = randbf(size_t(groups) * dim, g, 0.3f);
    auto dx = up(ctx, x), dw = up(ctx, w), dy = make(ctx, rows * dim * 2);
    const float eps = 1e-6f;
    auto run = [&] { sync(ctx, [&] { return vkk1::rmsnorm(ctx, dx.addr, dim, dw.addr, groups, gemma, eps, rows, dim, dy.addr, dim); }, "rmsnorm"); return host<uint16_t>(dy, rows * dim); };
    auto y = run(), y2 = run();
    std::vector<double> ref(rows * dim);
    for (int64_t r = 0; r < rows; ++r) {
        double ss = 0; for (int i = 0; i < dim; ++i) { const double v = bf2f(x[r * dim + i]); ss += v * v; }
        const double rs = 1.0 / std::sqrt(ss / dim + eps);
        for (int i = 0; i < dim; ++i) { const double wv = bf2f(w[(r % groups) * dim + i]); ref[r * dim + i] = bf2f(x[r * dim + i]) * rs * (gemma ? 1 + wv : wv); }
    }
    tol(name, y, y2, ref);
    ctx.free(dx); ctx.free(dw); ctx.free(dy);
}

static uint8_t fp8_ref(float x) {  // the HIP test's independent reference: nearest representable E4M3, ties to even mantissa
    uint32_t u; std::memcpy(&u, &x, 4); const uint8_t s = uint8_t((u >> 24) & 0x80);
    if (std::isnan(x)) return s | 0x7f;
    const float a = std::fabs(x);
    if (a > 464.f) return s | 0x7e;
    uint8_t best = 0; double bd = INFINITY;
    for (uint8_t c = 0; c <= 0x7e; ++c) {
        const int e = c >> 3, m = c & 7; const double v = e == 0 ? m / 8.0 * std::ldexp(1.0, -6) : (1 + m / 8.0) * std::ldexp(1.0, e - 7);
        const double d = std::fabs(v - a);
        if (d < bd || (d == bd && (c & 1) == 0)) { bd = d; best = c; }
    }
    return s | best;
}

int main(int argc, char **argv) {
    auto ctx_owner = cere::vk::testing::create_or_die(argc > 1 ? argv[1] : "build/vulkan-k1/spv");
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s, subgroup %u..%u\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str(),
                ctx.info().subgroup_min, ctx.info().subgroup_max);
    std::mt19937 g(1234);

    {   // embedding gather, bit-exact
        const int64_t vocab = 1000; const int hidden = 256; const int64_t n = 37;
        auto table = randbf(vocab * hidden, g); std::vector<int32_t> ids(n); for (auto &i : ids) i = int32_t(g() % vocab);
        auto dt = up(ctx, table), di = up(ctx, ids), dout = make(ctx, n * hidden * 2);
        auto run = [&] { sync(ctx, [&] { return vkk1::embedding_gather(ctx, dt.addr, vocab, hidden, di.addr, n, dout.addr); }, "embedding"); return host<uint16_t>(dout, n * hidden); };
        auto out = run(), out2 = run(); bool ok = out == out2;
        for (int64_t t = 0; t < n && ok; ++t) ok = std::memcmp(&out[t * hidden], &table[ids[t] * hidden], hidden * 2) == 0;
        report("embedding_gather", ok, "bit-exact");
        ctx.free(dt); ctx.free(di); ctx.free(dout);
    }
    test_rmsnorm(ctx, g, "rmsnorm gemma grouped (hc_norm)", 7 * 4, 2560, 4, true);
    test_rmsnorm(ctx, g, "rmsnorm gemma per-head (qk_norm)", 5 * 24, 256, 1, true);
    test_rmsnorm(ctx, g, "rmsnorm plain (gdn 128)", 5 * 48, 128, 1, false);
    {   // gated RMSNorm (sigmoid)
        const int64_t rows = 5 * 48; const int dim = 128; const float eps = 1e-6f;
        auto x = randbf(rows * dim, g, 2.f), z = randbf(rows * dim, g, 2.f), w = randbf(dim, g, 0.5f);
        auto dx = up(ctx, x), dz = up(ctx, z), dw = up(ctx, w), dy = make(ctx, rows * dim * 2);
        auto run = [&] { sync(ctx, [&] { return vkk1::rmsnorm_gated_sigmoid(ctx, dx.addr, dim, dz.addr, dim, dw.addr, eps, rows, dim, dy.addr, dim); }, "gated"); return host<uint16_t>(dy, rows * dim); };
        auto y = run(), y2 = run(); std::vector<double> ref(rows * dim);
        for (int64_t r = 0; r < rows; ++r) {
            double ss = 0; for (int i = 0; i < dim; ++i) { const double v = bf2f(x[r * dim + i]); ss += v * v; }
            const double rs = 1.0 / std::sqrt(ss / dim + eps);
            for (int i = 0; i < dim; ++i) { const double zz = bf2f(z[r * dim + i]); ref[r * dim + i] = bf2f(x[r * dim + i]) * rs * bf2f(w[i]) / (1 + std::exp(-zz)); }
        }
        tol("rmsnorm_gated_sigmoid (gdn)", y, y2, ref);
        // in-place (y aliases x) must give the same bits
        sync(ctx, [&] { return vkk1::rmsnorm_gated_sigmoid(ctx, dx.addr, dim, dz.addr, dim, dw.addr, eps, rows, dim, dx.addr, dim); }, "gated inplace");
        report("rmsnorm_gated in-place == out-of-place", host<uint16_t>(dx, rows * dim) == y, "bit-identical (y aliases x)");
        ctx.free(dx); ctx.free(dz); ctx.free(dw); ctx.free(dy);
    }
    {   // rope table + partial NeoX rope on text positions
        const int64_t n_pos = 512; const int rd = 64; const float theta = 1e7f;
        auto dtab = make(ctx, n_pos * rd * 4);
        sync(ctx, [&] { return vkk1::rope_table(ctx, dtab.addr, n_pos, rd, theta); }, "rope_table"); auto tab = host<float>(dtab, n_pos * rd);
        double worst = 0;
        for (int64_t p = 0; p < n_pos; ++p) for (int i = 0; i < rd / 2; ++i) {
            const double a = double(p) * (1.0 / std::pow(double(theta), double(2 * i) / rd));
            worst = std::fmax(worst, std::fabs(tab[p * rd + i] - std::cos(a))); worst = std::fmax(worst, std::fabs(tab[p * rd + rd / 2 + i] - std::sin(a)));
        }
        char d[64]; std::snprintf(d, sizeof d, "max |err| %.2e vs FP64 (limit 1e-4)", worst); report("rope_table (theta 1e7, rd 64)", worst <= 1e-4, d);
        const int64_t T = 5; const int heads = 8, hd = 256;
        auto x = randbf(T * heads * hd, g); std::vector<int64_t> pos = {0, 21, 59, 95, 511};
        auto dx = up(ctx, x), dp = up(ctx, pos), dy = make(ctx, T * heads * hd * 2);
        auto run = [&] { sync(ctx, [&] { return vkk1::rope(ctx, dx.addr, heads * hd, dy.addr, heads * hd, dtab.addr, dp.addr, 0, 0, T, heads, hd, rd); }, "rope"); return host<uint16_t>(dy, T * heads * hd); };
        auto y = run(), y2 = run(); std::vector<double> ref(T * heads * hd);
        for (int64_t t = 0; t < T; ++t) for (int h = 0; h < heads; ++h) {
            const size_t b = (t * heads + h) * hd;
            for (int i = 0; i < hd; ++i) ref[b + i] = bf2f(x[b + i]);
            for (int i = 0; i < rd / 2; ++i) {
                const double a = double(pos[t]) * (1.0 / std::pow(double(theta), double(2 * i) / rd));
                const double x1 = bf2f(x[b + i]), x2 = bf2f(x[b + i + rd / 2]);
                ref[b + i] = x1 * std::cos(a) - x2 * std::sin(a); ref[b + i + rd / 2] = x2 * std::cos(a) + x1 * std::sin(a);
            }
        }
        tol("rope partial NeoX (64 of 256)", y, y2, ref);
        // extra rows: pos0 + t addressing equals the explicit positions; in-place equals out-of-place; dims >= rd copied bit-exact
        std::vector<int64_t> seq = {37, 38, 39, 40, 41};
        auto ds = up(ctx, seq);
        sync(ctx, [&] { return vkk1::rope(ctx, dx.addr, heads * hd, dy.addr, heads * hd, dtab.addr, ds.addr, 0, 0, T, heads, hd, rd); }, "rope seq");
        auto a = host<uint16_t>(dy, T * heads * hd);
        sync(ctx, [&] { return vkk1::rope(ctx, dx.addr, heads * hd, dy.addr, heads * hd, dtab.addr, 0, 0, 37, T, heads, hd, rd); }, "rope pos0");
        report("rope pos0 + t == positions[t]", host<uint16_t>(dy, T * heads * hd) == a, "bit-identical");
        sync(ctx, [&] { return vkk1::rope(ctx, dx.addr, heads * hd, dx.addr, heads * hd, dtab.addr, dp.addr, 0, 0, T, heads, hd, rd); }, "rope inplace");
        report("rope in-place == out-of-place", host<uint16_t>(dx, T * heads * hd) == y, "bit-identical (y aliases x)");
        ctx.free(dtab); ctx.free(dx); ctx.free(dp); ctx.free(dy); ctx.free(ds);
    }
    {   // sigmoid gate
        const int64_t rows = 9; const int width = 6144;
        auto x = randbf(rows * width, g), gt = randbf(rows * width, g, 2.f);
        auto dx = up(ctx, x), dg = up(ctx, gt), dy = make(ctx, rows * width * 2);
        auto run = [&] { sync(ctx, [&] { return vkk1::sigmoid_mul(ctx, dx.addr, width, dg.addr, width, rows, width, dy.addr, width); }, "sigmoid_mul"); return host<uint16_t>(dy, rows * width); };
        auto y = run(), y2 = run(); std::vector<double> ref(rows * width);
        for (size_t i = 0; i < ref.size(); ++i) ref[i] = bf2f(x[i]) / (1 + std::exp(-double(bf2f(gt[i]))));
        tol("sigmoid_mul (attn gate)", y, y2, ref);
        ctx.free(dx); ctx.free(dg); ctx.free(dy);
    }
    {   // top-2: torch.argmax rules (NaN is the maximum, lowest index wins ties), bit-exact
        const int64_t vocab = 248320, rows = 4;
        auto l = randbf(rows * vocab, g, 3.f);
        l[0 * vocab + 1000] = f2bf(50.f); l[0 * vocab + 200000] = f2bf(49.f);
        l[1 * vocab + 7] = f2bf(60.f); l[1 * vocab + 3] = f2bf(60.f);
        l[2 * vocab + 123456] = 0x7fc0;
        auto dl = up(ctx, l), di = make(ctx, rows * 8), ds = make(ctx, rows * 8);
        auto run = [&] { sync(ctx, [&] { return vkk1::top2_split(ctx, dl.addr, vocab, rows, vocab, di.addr, ds.addr); }, "top2"); return std::make_pair(host<int32_t>(di, rows * 2), host<float>(ds, rows * 2)); };
        auto a = run(), b = run(); bool ok = a.first == b.first && std::memcmp(a.second.data(), b.second.data(), rows * 8) == 0;
        for (int64_t r = 0; r < rows && ok; ++r) {
            auto key = [&](int64_t i) { const float v = bf2f(l[r * vocab + i]); return std::isnan(v) ? INFINITY : v; };
            auto better = [&](int64_t i, int64_t j) { const bool ni = std::isnan(bf2f(l[r * vocab + i])), nj = std::isnan(bf2f(l[r * vocab + j])); if (ni != nj) return ni; return key(i) > key(j) || (key(i) == key(j) && i < j); };
            int64_t best = 0; for (int64_t i = 1; i < vocab; ++i) if (better(i, best)) best = i;
            int64_t sec = best == 0 ? 1 : 0; for (int64_t i = 0; i < vocab; ++i) if (i != best && better(i, sec)) sec = i;
            uint32_t sb0, sb1, rb0, rb1; float r0 = bf2f(l[r * vocab + best]), r1 = bf2f(l[r * vocab + sec]);
            std::memcpy(&sb0, &a.second[r * 2], 4); std::memcpy(&sb1, &a.second[r * 2 + 1], 4); std::memcpy(&rb0, &r0, 4); std::memcpy(&rb1, &r1, 4);
            ok = a.first[r * 2] == best && a.first[r * 2 + 1] == sec && sb0 == rb0 && sb1 == rb1;
            if (!ok) std::printf("  row %ld: got (%d,%d) want (%ld,%ld)\n", (long)r, a.first[r * 2], a.first[r * 2 + 1], (long)best, (long)sec);
        }
        report("top2_split (248320, ties, NaN)", ok, "bit-exact ids and scores, single block per row");
        // extra row: an unaligned base and a stride that is not a multiple of 8 take the scalar path, same answers
        const int64_t v2 = 1001, rows2 = 3; std::vector<uint16_t> l2 = randbf(rows2 * v2 + 1, g, 3.f);
        l2[1 + 0 * v2 + 500] = f2bf(70.f); l2[1 + 1 * v2 + 999] = f2bf(70.f); l2[1 + 1 * v2 + 1000] = f2bf(70.f); l2[1 + 2 * v2] = f2bf(-80.f);
        auto dl2 = up(ctx, l2); auto di2 = make(ctx, rows2 * 8), ds2 = make(ctx, rows2 * 8);
        sync(ctx, [&] { return vkk1::top2_split(ctx, dl2.addr + 2, v2, rows2, v2, di2.addr, ds2.addr); }, "top2 scalar");
        auto ii = host<int32_t>(di2, rows2 * 2); bool ok2 = ii[0] == 500 && ii[2] == 999 && ii[3] == 1000;
        for (int64_t r = 0; r < rows2; ++r) { int64_t b = 0, s = -1; for (int64_t i = 1; i < v2; ++i) if (bf2f(l2[1 + r * v2 + i]) > bf2f(l2[1 + r * v2 + b])) b = i;
            for (int64_t i = 0; i < v2; ++i) if (i != b && (s < 0 || bf2f(l2[1 + r * v2 + i]) > bf2f(l2[1 + r * v2 + s]))) s = i;
            ok2 = ok2 && ii[r * 2] == b && ii[r * 2 + 1] == s; }
        report("top2_split unaligned/scalar path", ok2, "ids equal the host scan (stride 1001, base +2 bytes)");
        ctx.free(dl); ctx.free(di); ctx.free(ds); ctx.free(dl2); ctx.free(di2); ctx.free(ds2);
    }
    {   // FP8 E4M3 KV store, bit-exact vs the documented rule (independent CPU implementation)
        const int64_t tokens = 6; const int width = 512; const int64_t slots_n = 16;
        auto k = randbf(tokens * width, g, 60.f), v = randbf(tokens * width, g, 0.01f);
        k[0] = f2bf(500.f); k[1] = f2bf(-470.f); k[2] = f2bf(460.f); k[3] = 0x7f80; k[4] = 0x7fc0; k[5] = f2bf(1e-9f);
        std::vector<int64_t> slots = {9, 2, 15, 0, 7, 11};
        auto dk = up(ctx, k), dv = up(ctx, v), ds = up(ctx, slots), kc = make(ctx, slots_n * width), vc = make(ctx, slots_n * width);
        auto run = [&] { std::memset(kc.map, 0xAB, slots_n * width); std::memset(vc.map, 0xAB, slots_n * width);
            sync(ctx, [&] { return vkk1::kv_store_fp8(ctx, dk.addr, width, dv.addr, width, tokens, width, kc.addr, vc.addr, ds.addr, 0); }, "kv_store");
            return std::make_pair(host<uint8_t>(kc, slots_n * width), host<uint8_t>(vc, slots_n * width)); };
        auto a = run(), b = run(); bool ok = a == b; int64_t bad = 0;
        for (int64_t s = 0; s < slots_n; ++s) {
            int64_t t = -1; for (int64_t i = 0; i < tokens; ++i) if (slots[i] == s) t = i;
            for (int i = 0; i < width; ++i) {
                const uint8_t wk = t < 0 ? 0xAB : fp8_ref(bf2f(k[t * width + i])), wv = t < 0 ? 0xAB : fp8_ref(bf2f(v[t * width + i]));
                bad += (a.first[s * width + i] != wk) + (a.second[s * width + i] != wv);
            }
        }
        char d[96]; std::snprintf(d, sizeof d, "%ld of %ld bytes differ (incl. 500, -470, 460, Inf, NaN, untouched slots)", (long)bad, (long)(2 * slots_n * width));
        report("kv_store_fp8 (E4M3, scale 1)", ok && bad == 0, d);
        // extra row (beyond the HIP test): every BF16 bit pattern through the store against the reference, slot0 + t addressing
        std::vector<uint16_t> all(65536); for (uint32_t i = 0; i < 65536; ++i) all[i] = uint16_t(i);
        auto da = up(ctx, all), kc2 = make(ctx, 65536), vc2 = make(ctx, 65536);
        sync(ctx, [&] { return vkk1::kv_store_fp8(ctx, da.addr, 1024, da.addr, 1024, 64, 1024, kc2.addr, vc2.addr, 0, 0); }, "kv_store all");
        auto o = host<uint8_t>(kc2, 65536), o2 = host<uint8_t>(vc2, 65536); int64_t bad2 = 0;
        for (uint32_t i = 0; i < 65536; ++i) { const uint8_t w = fp8_ref(bf2f(uint16_t(i))); bad2 += (o[i] != w) + (o2[i] != w); }
        std::snprintf(d, sizeof d, "%ld of 131072 bytes differ (all 65,536 BF16 patterns, slot0 + t)", (long)bad2);
        report("kv_store_fp8 all BF16 patterns", bad2 == 0, d);
        ctx.free(dk); ctx.free(dv); ctx.free(ds); ctx.free(kc); ctx.free(vc); ctx.free(da); ctx.free(kc2); ctx.free(vc2);
    }
    {   // negative rows: contract refusals, no hidden fallback
        auto b = make(ctx, 1 << 16);
        ctx.begin();
        bool ok = vkk1::rope(ctx, b.addr, 512, b.addr, 512, b.addr, 0, b.addr, 0, 1, 2, 256, 64) == CERE_ERR_UNSUPPORTED;       // mRoPE
        ok = ok && vkk1::rmsnorm(ctx, b.addr, 100, b.addr, 1, true, 1e-6f, 1, 100, b.addr, 100) == CERE_ERR_ARGUMENT;          // dim % 8
        ok = ok && vkk1::rmsnorm(ctx, b.addr, 128, b.addr, 1, true, 1e-6f, 1, 16384, b.addr, 128) == CERE_ERR_ARGUMENT;        // dim > 8192
        ok = ok && vkk1::rmsnorm(ctx, b.addr + 2, 128, b.addr, 1, true, 1e-6f, 1, 128, b.addr, 128) == CERE_ERR_ARGUMENT;      // misaligned
        ok = ok && vkk1::kv_store_fp8(ctx, b.addr, 12, b.addr, 12, 1, 12, b.addr, b.addr, 0, 0) == CERE_ERR_ARGUMENT;          // width % 8
        ok = ok && vkk1::top2_split(ctx, b.addr, 8, 70000, 8, b.addr, b.addr) == CERE_ERR_ARGUMENT;                            // rows > 65,535
        ok = ok && vkk1::embedding_gather(ctx, b.addr, 10, 12, b.addr, 1, b.addr) == CERE_ERR_ARGUMENT;                        // hidden % 8
        ok = ok && vkk1::rope_table(ctx, b.addr, 4, 7, 1e4f) == CERE_ERR_ARGUMENT;                                              // odd rotary_dim
        ctx.submit_wait();
        report("negative: refusals", ok, "mRoPE UNSUPPORTED; bad dim / alignment / rows / width / rotary_dim ARGUMENT");
        ctx.free(b);
    }

    // ---- the primitives the HIP K1 test has no rows for (the model forward calls them): contract from k1_primitives.h and the .cu
    //      definitions; row sets modelled on executor/tests/gpu/test_kernels_k1_primitives.py (kv_store bit-exact, copy_segments, top-2 ties).
    {   // qk_norm_rope_gate (k1_primitives.h:61-77; k1_primitives.cu qk_norm_rope_gate_kernel)
        // The CUDA kernel computes, per head, exactly rmsnorm_row's gemma norm (same thread count ceil32(hd/8), same sum tree), rounds the
        // result to BF16 and applies rope_kernel's FP32 rotation, so it equals the unfused rmsnorm -> rope sequence bit for bit
        // (the CUDA suite compares it to SGLang's fused op, not to an unfused sequence: test_isolate_decode_qk_norm_rope). Asserted here:
        // bit-equal to this device's unfused sequence, gate bit-exact, FP64 rel-L2 <= 4e-3, repeat bit-identical.
        const int64_t T = 5; const int qh = 24, kvh = 2, hd = 256, rd = 64; const float eps = 1e-6f, theta = 1e7f;
        auto qg = randbf(T * qh * 2 * hd, g, 4.f), k = randbf(T * kvh * hd, g, 4.f), qw = randbf(hd, g, 0.3f), kw = randbf(hd, g, 0.3f);
        std::vector<int64_t> pos = {0, 21, 59, 95, 511};
        auto dqg = up(ctx, qg), dk = up(ctx, k), dqw = up(ctx, qw), dkw = up(ctx, kw), dp = up(ctx, pos), dtab = make(ctx, 512 * rd * 4);
        auto qo = make(ctx, T * qh * hd * 2), ko = make(ctx, T * kvh * hd * 2), go = make(ctx, T * qh * hd * 2);
        sync(ctx, [&] { return vkk1::rope_table(ctx, dtab.addr, 512, rd, theta); }, "tab");
        auto fused = [&](uint64_t positions, int64_t pos0) {
            sync(ctx, [&] { return vkk1::qk_norm_rope_gate(ctx, dqg.addr, qh * 2 * hd, dk.addr, kvh * hd, dqw.addr, dkw.addr, dtab.addr, positions, 0, pos0, T,
                                                           qh, kvh, hd, rd, eps, qo.addr, ko.addr, go.addr); }, "qk_norm_rope_gate");
            return std::make_tuple(host<uint16_t>(qo, T * qh * hd), host<uint16_t>(ko, T * kvh * hd), host<uint16_t>(go, T * qh * hd));
        };
        auto a = fused(dp.addr, 0), b = fused(dp.addr, 0);
        // unfused: gemma rmsnorm of the q rows (x_stride 2 * hd skips the gate halves) and k rows, then rope
        auto tq = make(ctx, T * qh * hd * 2), tk = make(ctx, T * kvh * hd * 2), uq = make(ctx, T * qh * hd * 2), uk = make(ctx, T * kvh * hd * 2);
        sync(ctx, [&] {
            must(vkk1::rmsnorm(ctx, dqg.addr, 2 * hd, dqw.addr, 1, true, eps, T * qh, hd, tq.addr, hd), "rmsnorm q");
            must(vkk1::rmsnorm(ctx, dk.addr, hd, dkw.addr, 1, true, eps, T * kvh, hd, tk.addr, hd), "rmsnorm k");
            ctx.barrier();
            must(vkk1::rope(ctx, tq.addr, qh * hd, uq.addr, qh * hd, dtab.addr, dp.addr, 0, 0, T, qh, hd, rd), "rope q");
            return vkk1::rope(ctx, tk.addr, kvh * hd, uk.addr, kvh * hd, dtab.addr, dp.addr, 0, 0, T, kvh, hd, rd); }, "unfused");
        const bool eq_q = std::get<0>(a) == host<uint16_t>(uq, T * qh * hd), eq_k = std::get<1>(a) == host<uint16_t>(uk, T * kvh * hd);
        bool gate_ok = true;
        for (int64_t t = 0; t < T; ++t) for (int h = 0; h < qh; ++h)
            gate_ok = gate_ok && std::memcmp(&std::get<2>(a)[(t * qh + h) * hd], &qg[(t * qh * 2 + 2 * h + 1) * hd], hd * 2) == 0;
        std::vector<double> rq(T * qh * hd), rk(T * kvh * hd);
        auto ref_head = [&](const uint16_t *x, const std::vector<uint16_t> &w, int64_t p, double *out) {
            double ss = 0; for (int i = 0; i < hd; ++i) { const double v = bf2f(x[i]); ss += v * v; }
            const double rs = 1.0 / std::sqrt(ss / hd + eps); std::vector<double> n(hd);
            for (int i = 0; i < hd; ++i) n[i] = bf2f(x[i]) * rs * (1.0 + bf2f(w[i]));
            for (int i = 0; i < hd; ++i) out[i] = n[i];
            for (int i = 0; i < rd / 2; ++i) {
                const double an = double(p) * (1.0 / std::pow(double(theta), double(2 * i) / rd));
                out[i] = n[i] * std::cos(an) - n[i + rd / 2] * std::sin(an); out[i + rd / 2] = n[i + rd / 2] * std::cos(an) + n[i] * std::sin(an);
            }
        };
        for (int64_t t = 0; t < T; ++t) {
            for (int h = 0; h < qh; ++h) ref_head(&qg[(t * qh * 2 + 2 * h) * hd], qw, pos[t], &rq[(t * qh + h) * hd]);
            for (int h = 0; h < kvh; ++h) ref_head(&k[(t * kvh + h) * hd], kw, pos[t], &rk[(t * kvh + h) * hd]);
        }
        const double eqr = rel_l2(std::get<0>(a), rq), ekr = rel_l2(std::get<1>(a), rk);
        char d[160];
        std::snprintf(d, sizeof d, "q rel-L2 %.2e, k rel-L2 %.2e (limit 4e-3), repeat bit-identical %s", eqr, ekr, a == b ? "yes" : "NO");
        report("qk_norm_rope_gate vs FP64 (24q+2kv x 256)", eqr <= 4e-3 && ekr <= 4e-3 && a == b, d);
        report("qk_norm_rope_gate == rmsnorm -> rope", eq_q && eq_k, "q and k bit-identical to this device's unfused sequence");
        report("qk_norm_rope_gate gate copied", gate_ok, "gate halves bit-exact");
        // pos0 + t addressing equals explicit consecutive positions
        std::vector<int64_t> seq = {37, 38, 39, 40, 41}; auto ds = up(ctx, seq);
        auto c1 = fused(ds.addr, 0), c2 = fused(0, 37);
        report("qk_norm_rope_gate pos0 + t", c1 == c2, "bit-identical to positions[t] = pos0 + t");
        // mRoPE refused
        ctx.begin();
        const bool refused = vkk1::qk_norm_rope_gate(ctx, dqg.addr, qh * 2 * hd, dk.addr, kvh * hd, dqw.addr, dkw.addr, dtab.addr, dp.addr, dp.addr, 0, T, qh, kvh, hd, rd, eps,
                                                     qo.addr, ko.addr, go.addr) == CERE_ERR_UNSUPPORTED &&
                             vkk1::qk_norm_rope_gate(ctx, dqg.addr, qh * 2 * hd, dk.addr, kvh * hd, dqw.addr, dkw.addr, dtab.addr, dp.addr, 0, 0, T, qh, kvh, 100, rd, eps,
                                                     qo.addr, ko.addr, go.addr) == CERE_ERR_ARGUMENT;
        ctx.submit_wait();
        report("negative: qk_norm_rope_gate refusals", refused, "positions3 UNSUPPORTED; head_dim % 8 ARGUMENT");
        for (auto *b2 : {&dqg, &dk, &dqw, &dkw, &dp, &dtab, &qo, &ko, &go, &tq, &tk, &uq, &uk, &ds}) ctx.free(*b2);
    }
    {   // kv_store_fp8_at / kv_store_fp8_rows (k1_primitives.h:96-114): bytes equal the host reference; rows == per-row _at, bit for bit
        const int width = 512; const int64_t slots_n = 24;
        auto mk = [&](size_t n, float sc, int seed) { std::mt19937 gg(seed); auto v = randbf(n, gg, sc); return v; };
        const int64_t T = 6;
        auto k = mk(T * width, 60.f, 91), v = mk(T * width, 0.01f, 92);
        k[0] = f2bf(500.f); k[1] = f2bf(-470.f); k[2] = f2bf(460.f); k[3] = 0x7f80; k[4] = 0x7fc0; k[5] = f2bf(1e-9f);
        std::vector<int64_t> slotmap(32); for (size_t i = 0; i < slotmap.size(); ++i) slotmap[i] = int64_t((i * 7 + 3) % slots_n);  // distinct (7 coprime to 24)
        const int64_t at0 = 5; std::vector<int64_t> atv = {at0};
        auto dk = up(ctx, k), dv = up(ctx, v), dsl = up(ctx, slotmap), dat = up(ctx, atv), kc = make(ctx, slots_n * width), vc = make(ctx, slots_n * width);
        auto fill = [&](vk::Buffer &b) { std::memset(b.map, 0xAB, slots_n * width); };
        fill(kc); fill(vc);
        sync(ctx, [&] { return vkk1::kv_store_fp8_at(ctx, dk.addr, width, dv.addr, width, T, width, kc.addr, vc.addr, dsl.addr, dat.addr); }, "kv_at");
        auto ak = host<uint8_t>(kc, slots_n * width), av = host<uint8_t>(vc, slots_n * width);
        std::vector<uint8_t> ek(slots_n * width, 0xAB), ev(slots_n * width, 0xAB);
        for (int64_t t = 0; t < T; ++t) for (int i = 0; i < width; ++i) { const int64_t s = slotmap[at0 + t]; ek[s * width + i] = fp8_ref(bf2f(k[t * width + i])); ev[s * width + i] = fp8_ref(bf2f(v[t * width + i])); }
        int64_t bad = 0; for (size_t i = 0; i < ek.size(); ++i) bad += (ak[i] != ek[i]) + (av[i] != ev[i]);
        // == kv_store_fp8 on the shifted slot map (the contract: "the stores of kv_store_fp8(slots + *at, ...) bit for bit")
        fill(kc); fill(vc);
        sync(ctx, [&] { return vkk1::kv_store_fp8(ctx, dk.addr, width, dv.addr, width, T, width, kc.addr, vc.addr, dsl.addr + 8 * at0, 0); }, "kv_plain");
        const bool same = host<uint8_t>(kc, slots_n * width) == ak && host<uint8_t>(vc, slots_n * width) == av;
        char d[120]; std::snprintf(d, sizeof d, "%ld of %ld bytes differ from the host reference; == kv_store_fp8(slots + at) %s", (long)bad, (long)(2 * ek.size()), same ? "bit-identical" : "DIFFERS");
        report("kv_store_fp8_at", bad == 0 && same, d);
        // rows: 16 rows, each its own caches and slot map, its own at[t]
        const int R = 16; std::vector<vk::Buffer> rk(R), rv(R), rs(R); vkk1::KvRows rows{};
        std::vector<std::vector<int64_t>> maps(R); std::vector<int64_t> atr(R);
        auto kr = mk(R * width, 60.f, 93), vr = mk(R * width, 0.5f, 94); kr[0] = f2bf(500.f); kr[1] = f2bf(-470.f); kr[width] = 0x7fc0; kr[2 * width] = 0x7f80;
        for (int t = 0; t < R; ++t) {
            rk[t] = make(ctx, slots_n * width); rv[t] = make(ctx, slots_n * width); std::memset(rk[t].map, 0xAB, slots_n * width); std::memset(rv[t].map, 0xAB, slots_n * width);
            maps[t].resize(8); for (int i = 0; i < 8; ++i) maps[t][i] = (i * 5 + t) % slots_n;
            rs[t] = up(ctx, maps[t]); atr[t] = (t * 3) % 8;
            rows.k_cache[t] = rk[t].addr; rows.v_cache[t] = rv[t].addr; rows.slots[t] = rs[t].addr;
        }
        auto dkr = up(ctx, kr), dvr = up(ctx, vr), dar = up(ctx, atr); auto table = make(ctx, 24 * 16 + 64);
        sync(ctx, [&] { return vkk1::kv_store_fp8_rows(ctx, dkr.addr, width, dvr.addr, width, R, width, rows, dar.addr, table); }, "kv_rows");
        std::vector<std::vector<uint8_t>> rowk(R), rowv(R); bad = 0;
        for (int t = 0; t < R; ++t) {
            rowk[t] = host<uint8_t>(rk[t], slots_n * width); rowv[t] = host<uint8_t>(rv[t], slots_n * width);
            std::vector<uint8_t> a2(slots_n * width, 0xAB), b2(slots_n * width, 0xAB); const int64_t s = maps[t][atr[t]];
            for (int i = 0; i < width; ++i) { a2[s * width + i] = fp8_ref(bf2f(kr[t * width + i])); b2[s * width + i] = fp8_ref(bf2f(vr[t * width + i])); }
            for (size_t i = 0; i < a2.size(); ++i) bad += (rowk[t][i] != a2[i]) + (rowv[t][i] != b2[i]);
        }
        // row t == kv_store_fp8_at(.., row t's caches and map, at + t) (tokens = 1 on row t's inputs)
        bool eq = true;
        for (int t = 0; t < R; ++t) {
            std::memset(rk[t].map, 0xAB, slots_n * width); std::memset(rv[t].map, 0xAB, slots_n * width);
            sync(ctx, [&] { return vkk1::kv_store_fp8_at(ctx, dkr.addr + uint64_t(t) * width * 2, width, dvr.addr + uint64_t(t) * width * 2, width, 1, width, rk[t].addr, rv[t].addr, rs[t].addr, dar.addr + 8u * t); }, "kv_at row");
            eq = eq && host<uint8_t>(rk[t], slots_n * width) == rowk[t] && host<uint8_t>(rv[t], slots_n * width) == rowv[t];
        }
        std::snprintf(d, sizeof d, "%ld of %d bytes differ from the host reference; row t == kv_store_fp8_at(.., at + t) %s", (long)bad, R * 2 * int(slots_n) * width, eq ? "bit-identical" : "DIFFERS");
        report("kv_store_fp8_rows (16 rows)", bad == 0 && eq, d);
        for (int t = 0; t < R; ++t) { ctx.free(rk[t]); ctx.free(rv[t]); ctx.free(rs[t]); }
        for (auto *b2 : {&dk, &dv, &dsl, &dat, &kc, &vc, &dkr, &dvr, &dar, &table}) ctx.free(*b2);
    }
    {   // copy_segments (k1_primitives.h:116-131), the CUDA suite's cases (test_copy_segments_bit_exact)
        const uint64_t N = 4 << 20; std::mt19937 gs(1010);
        std::vector<uint8_t> src(N); for (auto &b : src) b = uint8_t(gs());
        auto dsrc = up(ctx, src), ddst = make(ctx, N), table = make(ctx, 48 * 128 + 64);
        struct C { uint64_t d, dp, s, sp, w, r; };
        std::vector<std::pair<const char *, std::vector<C>>> cases;
        cases.push_back({"flat", {{0, 1572864, 0, 1572864, 1572864, 1}, {1572864, 61440, 2 << 20, 61440, 61440, 1}}});
        cases.push_back({"pitched", {{4096, 20480, 3 << 20, 5120, 5120, 4}}});
        cases.push_back({"unaligned", {{3, 1003, 1, 1010, 1001, 3}, {65536, 0, 77, 0, 0, 1}}});
        cases.push_back({"four", {{0, 256, 512, 256, 256, 1}, {4096, 16, 8192, 16, 16, 1}, {8192, 32, 16384, 64, 32, 3}, {65537, 9, 70001, 9, 9, 1}}});
        std::vector<C> r98, b128;
        for (uint64_t i = 0; i < 98; ++i) r98.push_back({i * 40960, 0, (i * 37) % 4096 * 512, 0, 16 * (1 + i % 23) + (i % 3 == 2), 1});
        for (uint64_t i = 0; i < 128; ++i) b128.push_back({i * 32768, 4096, 3u << 20, 1024, 16 * (i % 64), 1 + i % 4});
        cases.push_back({"restore98", r98}); cases.push_back({"bound128", b128});
        bool all = true; char d[160]; int64_t units = 0;
        for (auto &cs : cases) {
            std::memset(ddst.map, 0xA5, N); std::vector<uint8_t> exp(N, 0xA5); std::vector<vkk1::CopySegment> segs;
            for (auto &c : cs.second) {
                for (uint64_t r = 0; r < c.r; ++r) std::memcpy(&exp[c.d + r * c.dp], &src[c.s + r * c.sp], c.w);
                segs.push_back({ddst.addr + c.d, c.dp, dsrc.addr + c.s, c.sp, c.w, c.r}); units += int64_t(c.w * c.r);
            }
            sync(ctx, [&] { return vkk1::copy_segments(ctx, segs.data(), int(segs.size()), table); }, "copy_segments");
            const bool ok = host<uint8_t>(ddst, N) == exp; all = all && ok;
            std::snprintf(d, sizeof d, "%zu segments: destination bytes %s the host copy (no other byte changed)", segs.size(), ok ? "==" : "DIFFER from");
            report((std::string("copy_segments ") + cs.first).c_str(), ok, d);
        }
        // refusals as the CUDA row: 129 segments, 0, a null pointer are ARGUMENT; 128 zero-byte segments are OK (a no-op)
        std::vector<vkk1::CopySegment> zero(129, vkk1::CopySegment{ddst.addr, 0, dsrc.addr, 0, 0, 1}), nul(5, vkk1::CopySegment{});
        auto big = make(ctx, 48 * 130);
        ctx.begin();
        const bool neg = vkk1::copy_segments(ctx, zero.data(), 128, big) == CERE_OK && vkk1::copy_segments(ctx, zero.data(), 129, big) == CERE_ERR_ARGUMENT &&
                         vkk1::copy_segments(ctx, zero.data(), 0, big) == CERE_ERR_ARGUMENT && vkk1::copy_segments(ctx, nul.data(), 5, big) == CERE_ERR_ARGUMENT;
        ctx.submit_wait();
        report("negative: copy_segments refusals", neg, "129 / 0 segments and null pointers ARGUMENT; 128 zero-byte segments OK");
        for (auto *b2 : {&dsrc, &ddst, &table, &big}) ctx.free(*b2);
    }
    {   // top2 into cere_token_score (k1_primitives.h:86-90): 16-byte {id, runner_up, logit, runner_up_logit}; same answers as top2_split
        const int64_t vocab = 248320, rows = 4, stride = vocab + 16;
        auto l = randbf(rows * stride, g, 3.f);
        l[0 * stride + 1000] = f2bf(50.f); l[0 * stride + 200000] = f2bf(49.f);
        l[1 * stride + 7] = f2bf(60.f); l[1 * stride + 3] = f2bf(60.f);
        l[2 * stride + 123456] = 0x7fc0; l[3 * stride + 9] = f2bf(-INFINITY); 
        auto dl = up(ctx, l), out = make(ctx, rows * 16), oi = make(ctx, rows * 8), os = make(ctx, rows * 8);
        sync(ctx, [&] { return vkk1::top2(ctx, dl.addr, stride, rows, vocab, out.addr); }, "top2 struct");
        sync(ctx, [&] { return vkk1::top2_split(ctx, dl.addr, stride, rows, vocab, oi.addr, os.addr); }, "top2 split");
        struct S { uint32_t id, ru; float lg, rlg; };
        auto st = host<S>(out, rows); auto ii = host<int32_t>(oi, rows * 2); auto ss = host<float>(os, rows * 2);
        bool ok = true;
        for (int64_t r = 0; r < rows; ++r) {
            auto isnan_ = [&](int64_t i) { return std::isnan(bf2f(l[r * stride + i])); };
            auto key = [&](int64_t i) { return isnan_(i) ? INFINITY : bf2f(l[r * stride + i]); };
            auto better = [&](int64_t i, int64_t j) { if (isnan_(i) != isnan_(j)) return isnan_(i); return key(i) > key(j) || (key(i) == key(j) && i < j); };
            int64_t best = 0; for (int64_t i = 1; i < vocab; ++i) if (better(i, best)) best = i;
            int64_t sec = best == 0 ? 1 : 0; for (int64_t i = 0; i < vocab; ++i) if (i != best && better(i, sec)) sec = i;
            float b0 = bf2f(l[r * stride + best]), b1 = bf2f(l[r * stride + sec]);
            const bool row = st[r].id == uint32_t(best) && st[r].ru == uint32_t(sec) && std::memcmp(&st[r].lg, &b0, 4) == 0 && std::memcmp(&st[r].rlg, &b1, 4) == 0 &&
                             ii[2 * r] == int32_t(best) && ii[2 * r + 1] == int32_t(sec) && std::memcmp(&ss[2 * r], &b0, 4) == 0 && std::memcmp(&ss[2 * r + 1], &b1, 4) == 0;
            if (!row) std::printf("  row %ld: got (%u,%u) want (%ld,%ld)\n", (long)r, st[r].id, st[r].ru, (long)best, (long)sec);
            ok = ok && row;
        }
        sync(ctx, [&] { return vkk1::top2(ctx, dl.addr, stride, rows, vocab, out.addr); }, "top2 struct again");
        const bool rep = std::memcmp(host<S>(out, rows).data(), st.data(), rows * 16) == 0;
        report("top2 token_score (248320, ties, NaN, -Inf)", ok && rep, "bit-exact {id, runner_up, logit, runner_up_logit} == host scan == top2_split; stride vocab + 16; repeat identical");
        for (auto *b2 : {&dl, &out, &oi, &os}) ctx.free(*b2);
    }
    std::printf("contract rows: %s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);

    {   // decode-shape timing (GPU timestamps between each call's own pair, barrier after it; 3 warm-ups, median of 15)
        std::mt19937 gt(7);
        const int64_t V = 248320;
        auto tab = up(ctx, randbf(1000 * 2560, gt)); std::vector<int32_t> id1 = {123}; auto ids = up(ctx, id1);
        auto x = up(ctx, randbf(4 * 2560, gt, 2.f)), w = up(ctx, randbf(4 * 2560, gt, 0.3f)), y = make(ctx, 4 * 2560 * 2);
        auto z = up(ctx, randbf(48 * 128, gt, 2.f)); auto wd = up(ctx, randbf(128, gt)); auto q = up(ctx, randbf(24 * 256, gt, 2.f));
        auto wq = up(ctx, randbf(256, gt)); auto yq = make(ctx, 48 * 256 * 2);
        auto ropx = up(ctx, randbf(8 * 256, gt)); auto ropy = make(ctx, 8 * 256 * 2); auto cs = make(ctx, 512 * 64 * 4);
        sync(ctx, [&] { return vkk1::rope_table(ctx, cs.addr, 512, 64, 1e7f); }, "tab"); std::vector<int64_t> p1 = {95}; auto dp = up(ctx, p1);
        auto sx = up(ctx, randbf(6144, gt)), sg = up(ctx, randbf(6144, gt, 2.f)), so = make(ctx, 6144 * 2);
        auto lg = up(ctx, randbf(V, gt, 3.f)); auto ti = make(ctx, 8), ts = make(ctx, 8);
        auto kk = up(ctx, randbf(512, gt)), vv = up(ctx, randbf(512, gt)); auto kc = make(ctx, 16 * 512), vc = make(ctx, 16 * 512); std::vector<int64_t> s1 = {5}; auto sl = up(ctx, s1);
        auto qgb = up(ctx, randbf(24 * 512, gt, 4.f)), kb = up(ctx, randbf(2 * 256, gt, 4.f)), qwb = up(ctx, randbf(256, gt)), kwb = up(ctx, randbf(256, gt));
        auto qob = make(ctx, 24 * 256 * 2), kob = make(ctx, 2 * 256 * 2), gob = make(ctx, 24 * 256 * 2);
        std::vector<int64_t> smap(64); for (size_t i = 0; i < smap.size(); ++i) smap[i] = int64_t(i % 16); auto smb = up(ctx, smap); std::vector<int64_t> at1 = {3}; auto atb = up(ctx, at1);
        std::vector<int64_t> atr16(16); for (int i = 0; i < 16; ++i) atr16[i] = i; auto at16 = up(ctx, atr16);
        auto kv16 = up(ctx, randbf(16 * 512, gt)), vv16 = up(ctx, randbf(16 * 512, gt)); vkk1::KvRows rws{}; auto rtab = make(ctx, 24 * 16 + 64);
        for (int i = 0; i < 16; ++i) { rws.k_cache[i] = kc.addr; rws.v_cache[i] = vc.addr; rws.slots[i] = smb.addr; }
        auto csrc = up(ctx, randbf(6 * 2560, gt)), cdst = make(ctx, 6 * 5120), ctab = make(ctx, 48 * 8); std::vector<vkk1::CopySegment> segs6;
        for (int i = 0; i < 6; ++i) segs6.push_back({cdst.addr + 5120u * i, 5120, csrc.addr + 5120u * i, 5120, 5120, 1});
        auto lg5 = up(ctx, randbf(5 * V, gt, 3.f)); auto tsc = make(ctx, 5 * 16);
        std::vector<std::function<cere_status()>> calls = {
            [&] { return vkk1::embedding_gather(ctx, tab.addr, 1000, 2560, ids.addr, 1, y.addr); },
            [&] { return vkk1::rmsnorm(ctx, x.addr, 2560, w.addr, 4, true, 1e-6f, 4, 2560, y.addr, 2560); },
            [&] { return vkk1::rmsnorm(ctx, q.addr, 256, wq.addr, 1, true, 1e-6f, 24, 256, yq.addr, 256); },
            [&] { return vkk1::rmsnorm(ctx, z.addr, 128, wd.addr, 1, false, 1e-6f, 48, 128, yq.addr, 128); },
            [&] { return vkk1::rmsnorm_gated_sigmoid(ctx, z.addr, 128, z.addr, 128, wd.addr, 1e-6f, 48, 128, yq.addr, 128); },
            [&] { return vkk1::rope(ctx, ropx.addr, 2048, ropy.addr, 2048, cs.addr, dp.addr, 0, 0, 1, 8, 256, 64); },
            [&] { return vkk1::sigmoid_mul(ctx, sx.addr, 6144, sg.addr, 6144, 1, 6144, so.addr, 6144); },
            [&] { return vkk1::top2_split(ctx, lg.addr, V, 1, V, ti.addr, ts.addr); },
            [&] { return vkk1::kv_store_fp8(ctx, kk.addr, 512, vv.addr, 512, 1, 512, kc.addr, vc.addr, sl.addr, 0); },
            [&] { return vkk1::qk_norm_rope_gate(ctx, qgb.addr, 24 * 512, kb.addr, 512, qwb.addr, kwb.addr, cs.addr, dp.addr, 0, 0, 1, 24, 2, 256, 64, 1e-6f, qob.addr, kob.addr, gob.addr); },
            [&] { return vkk1::kv_store_fp8_at(ctx, kk.addr, 512, vv.addr, 512, 1, 512, kc.addr, vc.addr, smb.addr, atb.addr); },
            [&] { return vkk1::kv_store_fp8_rows(ctx, kv16.addr, 512, vv16.addr, 512, 16, 512, rws, at16.addr, rtab); },
            [&] { return vkk1::copy_segments(ctx, segs6.data(), 6, ctab); },
            [&] { return vkk1::top2(ctx, lg.addr, V, 1, V, tsc.addr); },
            [&] { return vkk1::top2(ctx, lg5.addr, V, 5, V, tsc.addr); }};
        std::printf("\ndecode-shape timing: us per call = (GPU time between two timestamps around %d dependent calls, a barrier after each) / %d,\n"
                    "median of 7 such batches [min..max]; 3 untimed warm-up batches first\n", kBatch, kBatch);
        for (size_t i = 0; i < calls.size(); ++i) {
            const int reps = 10;
            ctx.begin();
            for (int c = 0; c < reps; ++c) {
                ctx.stamp(2 * c);
                for (int k = 0; k < kBatch; ++k) { must(calls[i](), "call"); ctx.barrier(); }
                ctx.stamp(2 * c + 1);
            }
            ctx.submit_wait();
            std::vector<double> t; for (int c = 3; c < reps; ++c) t.push_back(ctx.stamp_ms(2 * c, 2 * c + 1) * 1e3 / kBatch);
            std::printf("TIME vulkan | %-46s | %8.2f us [%.2f..%.2f]\n", kShapes[i], median(t), *std::min_element(t.begin(), t.end()), *std::max_element(t.begin(), t.end()));
        }
    }
    return g_fail ? 1 : 0;
}

#else  // K1_HIP_TIMING: the same shapes through the HIP K1 kernels, hipEvents around each call
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
using bf16 = __hip_bfloat16;
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
static bf16 *B(uint16_t *p) { return reinterpret_cast<bf16 *>(p); }
int main() {
    hipDeviceProp_t prop; HCK(hipGetDeviceProperties(&prop, 0)); std::printf("device %s\n", prop.gcnArchName);
    std::mt19937 gt(7); const int64_t V = 248320;
    uint16_t *tab = dev(randbf(1000 * 2560, gt)); int32_t *ids = dev(std::vector<int32_t>{123});
    uint16_t *x = dev(randbf(4 * 2560, gt, 2.f)), *w = dev(randbf(4 * 2560, gt, 0.3f)), *y; HCK(hipMalloc(&y, 4 * 2560 * 2 + 256));
    uint16_t *z = dev(randbf(48 * 128, gt, 2.f)), *wd = dev(randbf(128, gt)), *q = dev(randbf(24 * 256, gt, 2.f)), *wq = dev(randbf(256, gt)), *yq; HCK(hipMalloc(&yq, 48 * 256 * 2 + 256));
    uint16_t *ropx = dev(randbf(8 * 256, gt)), *ropy; HCK(hipMalloc(&ropy, 8 * 256 * 2 + 256)); float *cs; HCK(hipMalloc(&cs, 512 * 64 * 4));
    HCK(cere::k1::rope_table(cs, 512, 64, 1e7f, nullptr)); int64_t *dp = dev(std::vector<int64_t>{95});
    uint16_t *sx = dev(randbf(6144, gt)), *sg = dev(randbf(6144, gt, 2.f)), *so; HCK(hipMalloc(&so, 6144 * 2 + 256));
    uint16_t *lg = dev(randbf(V, gt, 3.f)); int32_t *ti; float *ts; HCK(hipMalloc(&ti, 64)); HCK(hipMalloc(&ts, 64));
    uint16_t *kk = dev(randbf(512, gt)), *vv = dev(randbf(512, gt)); uint8_t *kc, *vc; HCK(hipMalloc(&kc, 16 * 512)); HCK(hipMalloc(&vc, 16 * 512)); int64_t *sl = dev(std::vector<int64_t>{5});
    uint16_t *qgb = dev(randbf(24 * 512, gt, 4.f)), *kb = dev(randbf(2 * 256, gt, 4.f)), *qwb = dev(randbf(256, gt)), *kwb = dev(randbf(256, gt)), *qob, *kob, *gob;
    HCK(hipMalloc(&qob, 24 * 256 * 2 + 256)); HCK(hipMalloc(&kob, 2 * 256 * 2 + 256)); HCK(hipMalloc(&gob, 24 * 256 * 2 + 256));
    std::vector<int64_t> smap(64); for (size_t i = 0; i < smap.size(); ++i) smap[i] = int64_t(i % 16); int64_t *smb = dev(smap), *atb = dev(std::vector<int64_t>{3});
    std::vector<int64_t> atr16(16); for (int i = 0; i < 16; ++i) atr16[i] = i; int64_t *at16 = dev(atr16);
    uint16_t *kv16 = dev(randbf(16 * 512, gt)), *vv16 = dev(randbf(16 * 512, gt)); cere::k1::KvRows rws{};
    for (int i = 0; i < 16; ++i) { rws.k_cache[i] = kc; rws.v_cache[i] = vc; rws.slots[i] = smb; }
    uint16_t *csrc = dev(randbf(6 * 2560, gt)), *cdst; HCK(hipMalloc(&cdst, 6 * 5120 + 256)); cere::k1::CopySegment segs6[6];
    for (int i = 0; i < 6; ++i) segs6[i] = {reinterpret_cast<uint8_t *>(cdst) + 5120 * i, 5120, reinterpret_cast<uint8_t *>(csrc) + 5120 * i, 5120, 5120, 1};
    uint16_t *lg5 = dev(randbf(5 * V, gt, 3.f)); cere_token_score *tsc; HCK(hipMalloc(&tsc, 5 * 16));
    HCK(hipDeviceSynchronize());
    std::vector<std::function<hipError_t(hipStream_t)>> calls = {
        [&](hipStream_t s) { return cere::k1::embedding_gather(B(tab), 1000, 2560, ids, 1, B(y), s); },
        [&](hipStream_t s) { return cere::k1::rmsnorm(B(x), 2560, B(w), 4, true, 1e-6f, 4, 2560, B(y), 2560, s); },
        [&](hipStream_t s) { return cere::k1::rmsnorm(B(q), 256, B(wq), 1, true, 1e-6f, 24, 256, B(yq), 256, s); },
        [&](hipStream_t s) { return cere::k1::rmsnorm(B(z), 128, B(wd), 1, false, 1e-6f, 48, 128, B(yq), 128, s); },
        [&](hipStream_t s) { return cere::k1::rmsnorm_gated_sigmoid(B(z), 128, B(z), 128, B(wd), 1e-6f, 48, 128, B(yq), 128, s); },
        [&](hipStream_t s) { return cere::k1::rope(B(ropx), 2048, B(ropy), 2048, cs, dp, nullptr, 0, 1, 8, 256, 64, s); },
        [&](hipStream_t s) { return cere::k1::sigmoid_mul(B(sx), 6144, B(sg), 6144, 1, 6144, B(so), 6144, s); },
        [&](hipStream_t s) { return cere::k1::top2_split(B(lg), V, 1, V, ti, ts, s); },
        [&](hipStream_t s) { return cere::k1::kv_store_fp8(B(kk), 512, B(vv), 512, 1, 512, kc, vc, sl, 0, s); },
        [&](hipStream_t s) { return cere::k1::qk_norm_rope_gate(B(qgb), 24 * 512, B(kb), 512, B(qwb), B(kwb), cs, dp, nullptr, 0, 1, 24, 2, 256, 64, 1e-6f, B(qob), B(kob), B(gob), s); },
        [&](hipStream_t s) { return cere::k1::kv_store_fp8_at(B(kk), 512, B(vv), 512, 1, 512, kc, vc, smb, atb, s); },
        [&](hipStream_t s) { return cere::k1::kv_store_fp8_rows(B(kv16), 512, B(vv16), 512, 16, 512, rws, at16, s); },
        [&](hipStream_t s) { return cere::k1::copy_segments(segs6, 6, s); },
        [&](hipStream_t s) { return cere::k1::top2(B(lg), V, 1, V, tsc, s); },
        [&](hipStream_t s) { return cere::k1::top2(B(lg5), V, 5, V, tsc, s); }};
    std::printf("decode-shape timing: us per call = hipEvent time of %d back-to-back calls on one stream / %d, median of 7 batches [min..max]; "
                "'stream' = plain launches, 'graph' = the same %d calls captured into one hipGraph (the executor's decode path replays graphs)\n", kBatch, kBatch, kBatch);
    hipStream_t st; HCK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    for (size_t i = 0; i < calls.size(); ++i) {
        for (int mode = 0; mode < 2; ++mode) {
            hipGraph_t gr; hipGraphExec_t ge = nullptr;
            if (mode) {
                HCK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal));
                for (int k = 0; k < kBatch; ++k) HCK(calls[i](st));
                HCK(hipStreamEndCapture(st, &gr)); HCK(hipGraphInstantiate(&ge, gr, nullptr, nullptr, 0));
            }
            const int reps = 10; std::vector<hipEvent_t> ev(2 * reps); for (auto &e : ev) HCK(hipEventCreate(&e));
            for (int c = 0; c < reps; ++c) {
                HCK(hipEventRecord(ev[2 * c], st));
                if (mode) HCK(hipGraphLaunch(ge, st)); else for (int k = 0; k < kBatch; ++k) HCK(calls[i](st));
                HCK(hipEventRecord(ev[2 * c + 1], st));
            }
            HCK(hipStreamSynchronize(st));
            std::vector<double> t; for (int c = 3; c < reps; ++c) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * c], ev[2 * c + 1])); t.push_back(ms * 1e3 / kBatch); }
            std::printf("TIME hip-%-6s | %-46s | %8.2f us [%.2f..%.2f]\n", mode ? "graph" : "stream", kShapes[i], median(t), *std::min_element(t.begin(), t.end()), *std::max_element(t.begin(), t.end()));
        }
    }
    return 0;
}
#endif
