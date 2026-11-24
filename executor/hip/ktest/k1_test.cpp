// AmdAmd PAR-6 step 2a: the K1 primitives (executor/cuda/src/kernels/k1_primitives.cu, hipified by gen_hip.py) on
// gfx1151 against their header contract, with FP64 CPU references (the CUDA suite compares to SGLang ops in the
// pinned image, which cannot run here). Tolerance cases use the CUDA suite's floor, rel-L2 <= 4e-3 vs FP64;
// embedding, top-2 and the FP8 KV store are bit-exact; every kernel runs twice and must be bit-identical.
// Exit 0 when every case passes.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k1_primitives.h"

using bf16 = __hip_bfloat16;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)

static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) {  // round to nearest even, NaN kept quiet
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static std::vector<uint16_t> randbf(size_t n, std::mt19937 &g, float scale = 1.f) {
    std::normal_distribution<float> d(0.f, scale); std::vector<uint16_t> v(n);
    for (auto &x : v) x = f2bf(d(g)); return v;
}
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 16)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
static double rel_l2(const std::vector<uint16_t> &got, const std::vector<double> &ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) { const double d = bf2f(got[i]) - ref[i]; num += d * d; den += ref[i] * ref[i]; }
    return std::sqrt(num / (den > 0 ? den : 1));
}
static void report(const char *name, bool ok, const char *detail) {
    std::printf("%-34s %s  %s\n", name, ok ? "PASS" : "FAIL", detail);
    if (!ok) ++g_fail;
}
static void tol(const char *name, const std::vector<uint16_t> &got, const std::vector<uint16_t> &again, const std::vector<double> &ref) {
    const double e = rel_l2(got, ref); char d[96];
    const bool det = got == again;
    std::snprintf(d, sizeof d, "rel-L2 %.2e (limit 4e-3), repeat bit-identical %s", e, det ? "yes" : "NO");
    report(name, e <= 4e-3 && det, d);
}

static void test_rmsnorm(std::mt19937 &g, const char *name, int64_t rows, int dim, int groups, bool gemma) {
    auto x = randbf(rows * dim, g, 2.f), w = randbf(size_t(groups) * dim, g, 0.3f);
    auto *dx = dev(x); auto *dw = dev(w); uint16_t *dy; HCK(hipMalloc(&dy, rows * dim * 2));
    const float eps = 1e-6f;
    auto run = [&] { HCK(cere::k1::rmsnorm(reinterpret_cast<bf16 *>(dx), dim, reinterpret_cast<bf16 *>(dw), groups, gemma, eps, rows, dim, reinterpret_cast<bf16 *>(dy), dim, nullptr)); return host(dy, rows * dim); };
    auto y = run(), y2 = run();
    std::vector<double> ref(rows * dim);
    for (int64_t r = 0; r < rows; ++r) {
        double ss = 0; for (int i = 0; i < dim; ++i) { const double v = bf2f(x[r * dim + i]); ss += v * v; }
        const double rs = 1.0 / std::sqrt(ss / dim + eps);
        for (int i = 0; i < dim; ++i) { const double wv = bf2f(w[(r % groups) * dim + i]); ref[r * dim + i] = bf2f(x[r * dim + i]) * rs * (gemma ? 1 + wv : wv); }
    }
    tol(name, y, y2, ref);
}

int main() {
    hipDeviceProp_t prop; HCK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s\n", prop.gcnArchName);
    std::mt19937 g(1234);

    {   // embedding gather, bit-exact
        const int64_t vocab = 1000; const int hidden = 256; const int64_t n = 37;
        auto table = randbf(vocab * hidden, g); std::vector<int32_t> ids(n); for (auto &i : ids) i = int32_t(g() % vocab);
        auto *dt = dev(table); auto *di = dev(ids); uint16_t *dout; HCK(hipMalloc(&dout, n * hidden * 2));
        auto run = [&] { HCK(cere::k1::embedding_gather(reinterpret_cast<bf16 *>(dt), vocab, hidden, di, n, reinterpret_cast<bf16 *>(dout), nullptr)); return host(dout, n * hidden); };
        auto out = run(), out2 = run(); bool ok = out == out2;
        for (int64_t t = 0; t < n && ok; ++t) ok = std::memcmp(&out[t * hidden], &table[ids[t] * hidden], hidden * 2) == 0;
        report("embedding_gather", ok, "bit-exact");
    }
    test_rmsnorm(g, "rmsnorm gemma grouped (hc_norm)", 7 * 4, 2560, 4, true);
    test_rmsnorm(g, "rmsnorm gemma per-head (qk_norm)", 5 * 24, 256, 1, true);
    test_rmsnorm(g, "rmsnorm plain (gdn 128)", 5 * 48, 128, 1, false);
    {   // gated RMSNorm (sigmoid)
        const int64_t rows = 5 * 48; const int dim = 128; const float eps = 1e-6f;
        auto x = randbf(rows * dim, g, 2.f), z = randbf(rows * dim, g, 2.f), w = randbf(dim, g, 0.5f);
        auto *dx = dev(x); auto *dz = dev(z); auto *dw = dev(w); uint16_t *dy; HCK(hipMalloc(&dy, rows * dim * 2));
        auto run = [&] { HCK(cere::k1::rmsnorm_gated_sigmoid(reinterpret_cast<bf16 *>(dx), dim, reinterpret_cast<bf16 *>(dz), dim, reinterpret_cast<bf16 *>(dw), eps, rows, dim, reinterpret_cast<bf16 *>(dy), dim, nullptr)); return host(dy, rows * dim); };
        auto y = run(), y2 = run(); std::vector<double> ref(rows * dim);
        for (int64_t r = 0; r < rows; ++r) {
            double ss = 0; for (int i = 0; i < dim; ++i) { const double v = bf2f(x[r * dim + i]); ss += v * v; }
            const double rs = 1.0 / std::sqrt(ss / dim + eps);
            for (int i = 0; i < dim; ++i) { const double zz = bf2f(z[r * dim + i]); ref[r * dim + i] = bf2f(x[r * dim + i]) * rs * bf2f(w[i]) / (1 + std::exp(-zz)); }
        }
        tol("rmsnorm_gated_sigmoid (gdn)", y, y2, ref);
    }
    {   // rope table + partial NeoX rope on text positions
        const int64_t n_pos = 512; const int rd = 64; const float theta = 1e7f;
        float *dtab; HCK(hipMalloc(&dtab, n_pos * rd * 4));
        HCK(cere::k1::rope_table(dtab, n_pos, rd, theta, nullptr)); auto tab = host(dtab, n_pos * rd);
        double worst = 0;
        for (int64_t p = 0; p < n_pos; ++p) for (int i = 0; i < rd / 2; ++i) {
            const double a = double(p) * (1.0 / std::pow(double(theta), double(2 * i) / rd));
            worst = std::fmax(worst, std::fabs(tab[p * rd + i] - std::cos(a))); worst = std::fmax(worst, std::fabs(tab[p * rd + rd / 2 + i] - std::sin(a)));
        }
        char d[64]; std::snprintf(d, sizeof d, "max |err| %.2e vs FP64 (limit 1e-4)", worst); report("rope_table (theta 1e7, rd 64)", worst <= 1e-4, d);
        const int64_t T = 5; const int heads = 8, hd = 256;
        auto x = randbf(T * heads * hd, g); std::vector<int64_t> pos = {0, 21, 59, 95, 511};
        auto *dx = dev(x); auto *dp = dev(pos); uint16_t *dy; HCK(hipMalloc(&dy, T * heads * hd * 2));
        auto run = [&] { HCK(cere::k1::rope(reinterpret_cast<bf16 *>(dx), heads * hd, reinterpret_cast<bf16 *>(dy), heads * hd, dtab, dp, nullptr, 0, T, heads, hd, rd, nullptr)); return host(dy, T * heads * hd); };
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
    }
    {   // sigmoid gate
        const int64_t rows = 9; const int width = 6144;
        auto x = randbf(rows * width, g), gt = randbf(rows * width, g, 2.f);
        auto *dx = dev(x); auto *dg = dev(gt); uint16_t *dy; HCK(hipMalloc(&dy, rows * width * 2));
        auto run = [&] { HCK(cere::k1::sigmoid_mul(reinterpret_cast<bf16 *>(dx), width, reinterpret_cast<bf16 *>(dg), width, rows, width, reinterpret_cast<bf16 *>(dy), width, nullptr)); return host(dy, rows * width); };
        auto y = run(), y2 = run(); std::vector<double> ref(rows * width);
        for (size_t i = 0; i < ref.size(); ++i) ref[i] = bf2f(x[i]) / (1 + std::exp(-double(bf2f(gt[i]))));
        tol("sigmoid_mul (attn gate)", y, y2, ref);
    }
    {   // top-2: torch.argmax rules (NaN is the maximum, lowest index wins ties), bit-exact
        const int64_t vocab = 248320, rows = 4;
        auto l = randbf(rows * vocab, g, 3.f);
        l[0 * vocab + 1000] = f2bf(50.f); l[0 * vocab + 200000] = f2bf(49.f);   // row 0: clear winners
        l[1 * vocab + 7] = f2bf(60.f); l[1 * vocab + 3] = f2bf(60.f);          // row 1: tie -> 3 then 7
        l[2 * vocab + 123456] = 0x7fc0;                                         // row 2: NaN is the maximum
        auto *dl = dev(l); int32_t *di; float *ds; HCK(hipMalloc(&di, rows * 8)); HCK(hipMalloc(&ds, rows * 8));
        auto run = [&] { HCK(cere::k1::top2_split(reinterpret_cast<bf16 *>(dl), vocab, rows, vocab, di, ds, nullptr)); return std::make_pair(host(di, rows * 2), host(ds, rows * 2)); };
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
    }
    {   // FP8 E4M3 KV store, bit-exact vs the documented rule (independent CPU implementation)
        auto fp8 = [](float x) -> uint8_t {
            uint32_t u; std::memcpy(&u, &x, 4); const uint8_t s = uint8_t((u >> 24) & 0x80);
            if (std::isnan(x)) return s | 0x7f;
            const float a = std::fabs(x);
            if (a > 464.f) return s | 0x7e;  // rounds above 448 (and Inf) saturate
            uint8_t best = 0; double bd = INFINITY;
            for (uint8_t c = 0; c <= 0x7e; ++c) {  // nearest representable, ties to even mantissa
                const int e = c >> 3, m = c & 7; const double v = e == 0 ? m / 8.0 * std::ldexp(1.0, -6) : (1 + m / 8.0) * std::ldexp(1.0, e - 7);
                const double d = std::fabs(v - a);
                if (d < bd || (d == bd && (c & 1) == 0)) { bd = d; best = c; }
            }
            return s | best;
        };
        const int64_t tokens = 6; const int width = 512; const int64_t slots_n = 16;
        auto k = randbf(tokens * width, g, 60.f), v = randbf(tokens * width, g, 0.01f);
        k[0] = f2bf(500.f); k[1] = f2bf(-470.f); k[2] = f2bf(460.f); k[3] = 0x7f80; k[4] = 0x7fc0; k[5] = f2bf(1e-9f);
        std::vector<int64_t> slots = {9, 2, 15, 0, 7, 11};
        auto *dk = dev(k); auto *dv = dev(v); auto *ds = dev(slots); uint8_t *kc, *vc;
        HCK(hipMalloc(&kc, slots_n * width)); HCK(hipMalloc(&vc, slots_n * width));
        auto run = [&] { HCK(hipMemset(kc, 0xAB, slots_n * width)); HCK(hipMemset(vc, 0xAB, slots_n * width));
            HCK(cere::k1::kv_store_fp8(reinterpret_cast<bf16 *>(dk), width, reinterpret_cast<bf16 *>(dv), width, tokens, width, kc, vc, ds, 0, nullptr));
            return std::make_pair(host(kc, slots_n * width), host(vc, slots_n * width)); };
        auto a = run(), b = run(); bool ok = a == b; int64_t bad = 0;
        for (int64_t s = 0; s < slots_n; ++s) {
            int64_t t = -1; for (int64_t i = 0; i < tokens; ++i) if (slots[i] == s) t = i;
            for (int i = 0; i < width; ++i) {
                const uint8_t wk = t < 0 ? 0xAB : fp8(bf2f(k[t * width + i])), wv = t < 0 ? 0xAB : fp8(bf2f(v[t * width + i]));
                bad += (a.first[s * width + i] != wk) + (a.second[s * width + i] != wv);
            }
        }
        char d[80]; std::snprintf(d, sizeof d, "%ld of %ld bytes differ (incl. 500, -470, 460, Inf, NaN, untouched slots)", (long)bad, (long)(2 * slots_n * width));
        report("kv_store_fp8 (E4M3, scale 1)", ok && bad == 0, d);
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
