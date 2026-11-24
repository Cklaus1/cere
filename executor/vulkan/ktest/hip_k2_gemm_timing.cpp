// VulkanAmd P2: same-session HIP reference for the Vulkan K2 large-m GEMM (k2w_test --timing), with its method: 2
// untimed warm-up calls, then `reps` calls each between its own pair of hipEvents (GPU time, calls serialized on one
// stream), median / min / max. Shapes: k2w_test's four prefill shapes at m 17, 64, 512, 4,096. Columns: k2_wmma BF16
// weights (BF16 out: what HIP's GemmPlan::run does for a class > 16), and the HIP adapter's MXFP8 path for a class > 16
// (k2m dequantize into a staging copy, then k2_wmma). Also the INFO row k2w_test prints for Vulkan: the HIP GEMV (m 16)
// vs k2_wmma (m 17) on the same rows, FP32 out, how many outputs differ. Built by build.d/k2_gemm.sh when hipcc
// exists (HIP is the reference here, not part of the Vulkan executor).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "k2_mgemv.h"
#include "k2_wmma.h"

#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
using namespace cere::hipk2;

static uint64_t g_xs = 0x9E3779B97F4A7C15ull;
static uint64_t xs() { g_xs ^= g_xs << 13; g_xs ^= g_xs >> 7; g_xs ^= g_xs << 17; return g_xs; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

struct T { double med, mn, mx; };
static T time_calls(const std::function<void()> &call, int reps) {
    std::vector<hipEvent_t> ev(2 * reps);
    for (auto &e : ev) HCK(hipEventCreate(&e));
    for (int i = 0; i < 2; ++i) call();
    for (int i = 0; i < reps; ++i) { HCK(hipEventRecord(ev[2 * i], nullptr)); call(); HCK(hipEventRecord(ev[2 * i + 1], nullptr)); }
    HCK(hipDeviceSynchronize());
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1])); t.push_back(ms); }
    for (auto &e : ev) HCK(hipEventDestroy(e));
    std::sort(t.begin(), t.end());
    return {t[t.size() / 2], t.front(), t.back()};
}

static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static double e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const double v = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
    return (b & 0x80) ? -v : v;
}

int main(int argc, char **argv) {  // --p4b: only the P4-B site shapes (k2_gemm_sites_test)
    using B = const __hip_bfloat16 *;
    {   // k2w_test's "vs FP64 (MX random bytes)" data (same generator and seed): HIP's MXFP8 large-m path
        // (dequantize + k2_wmma) and HIP's GEMV against FP64: how many BF16 outputs miss 1 ulp, FP32 worst
        std::mt19937 g(23); std::normal_distribution<float> nd;
        auto rnd = [&](size_t n, float sc) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * sc); return v; };
        const uint32_t m = 200, n = 300, k = 256;
        auto x = rnd(size_t(m) * k, 1.f), w = rnd(size_t(n) * k, 0.1f);
        std::vector<uint8_t> q(size_t(n) * k), s(size_t(n) * k / 32); std::vector<uint16_t> up(q.size());
        std::uniform_int_distribution<int> qb(0, 255), sb(112, 122);
        for (auto &b : q) { do b = uint8_t(qb(g)); while ((b & 0x7f) == 0x7f); }
        for (auto &b : s) b = uint8_t(sb(g));
        for (size_t i = 0; i < q.size(); ++i) up[i] = f2bf(float(std::ldexp(e4m3(q[i]), int(s[i / 32]) - 127)));
        void *dx, *dw, *y; uint8_t *dq, *ds;
        HCK(hipMalloc(&dx, x.size() * 2)); HCK(hipMalloc(&dq, q.size())); HCK(hipMalloc(&ds, s.size())); HCK(hipMalloc(&dw, up.size() * 2)); HCK(hipMalloc(&y, size_t(m) * n * 4));
        HCK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice)); HCK(hipMemcpy(dq, q.data(), q.size(), hipMemcpyHostToDevice));
        HCK(hipMemcpy(ds, s.data(), s.size(), hipMemcpyHostToDevice));
        HCK(mxfp8_dequantize(dq, ds, n, k, static_cast<__hip_bfloat16 *>(dw), nullptr));
        for (int path = 0; path < 2; ++path) {
            std::vector<uint16_t> yb(size_t(m) * n); std::vector<float> yf(size_t(m) * n);
            for (int f32 = 0; f32 < 2; ++f32) {
                if (path == 0) {
                    if (gemm_wmma(B(dx), B(dw), y, m, n, k, f32 ? WmmaOut::kF32 : WmmaOut::kBF16, nullptr) != CERE_OK) return 2;
                } else {
                    for (uint32_t r0 = 0; r0 < m; r0 += kGemvMaxM)
                        if (gemv(B(dx) + size_t(r0) * k, dq, ds, static_cast<char *>(y) + size_t(r0) * n * (f32 ? 4 : 2), std::min(kGemvMaxM, m - r0), n, k, f32, nullptr) != CERE_OK) return 2;
                }
                HCK(hipDeviceSynchronize());
                if (f32) HCK(hipMemcpy(yf.data(), y, yf.size() * 4, hipMemcpyDeviceToHost)); else HCK(hipMemcpy(yb.data(), y, yb.size() * 2, hipMemcpyDeviceToHost));
            }
            long bad = 0; double worst = 0; std::string where;
            for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) {
                double r = 0, sa = 0;
                for (uint32_t t = 0; t < k; ++t) { const double p = double(bf2f(x[size_t(i) * k + t])) * bf2f(up[size_t(j) * k + t]); r += p; sa += std::fabs(p); }
                if (std::abs(int(yb[size_t(i) * n + j]) - int(f2bf(float(r)))) > 1) {
                    ++bad; char b[96]; std::snprintf(b, sizeof b, " (%u, %u) got 0x%04x", i, j, yb[size_t(i) * n + j]); where += b;
                }
                worst = std::max(worst, std::fabs(yf[size_t(i) * n + j] - r) / (sa + 1e-30));
            }
            std::printf("hip  MX random bytes vs FP64  %-22s 200 x 300 x 256: %ld BF16 outputs > 1 ulp%s; FP32 worst %.1e\n",
                        path == 0 ? "dequantize + k2_wmma:" : "GEMV (16-row calls):", bad, where.c_str(), worst);
        }
        HCK(hipFree(dx)); HCK(hipFree(dq)); HCK(hipFree(ds)); HCK(hipFree(dw)); HCK(hipFree(y));
    }
    {   // INFO: GEMV (m 16) vs k2_wmma (m 17), rows 0..15, FP32 out
        const uint32_t n = 512, k = 2560, M = 17;
        std::vector<uint16_t> x(size_t(M) * k), w(size_t(n) * k);
        for (auto &v : x) v = f2bf(float(int64_t(xs() % 2001) - 1000) / 400.f);
        for (auto &v : w) v = f2bf(float(int64_t(xs() % 2001) - 1000) / 8000.f);
        void *dx, *dw; float *y1, *y2;
        HCK(hipMalloc(&dx, x.size() * 2)); HCK(hipMalloc(&dw, w.size() * 2)); HCK(hipMalloc(&y1, size_t(M) * n * 4)); HCK(hipMalloc(&y2, size_t(M) * n * 4));
        HCK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice)); HCK(hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice));
        if (gemv(B(dx), dw, nullptr, y1, 16, n, k, true, nullptr) != CERE_OK) return 2;
        if (gemm_wmma(B(dx), B(dw), y2, 17, n, k, WmmaOut::kF32, nullptr) != CERE_OK) return 2;
        std::vector<float> a(size_t(16) * n), b(size_t(16) * n);
        HCK(hipMemcpy(a.data(), y1, a.size() * 4, hipMemcpyDeviceToHost)); HCK(hipMemcpy(b.data(), y2, b.size() * 4, hipMemcpyDeviceToHost));
        long diff = 0; for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
        std::printf("hip  16/17 boundary INFO  GEMV (m 16) vs k2_wmma (m 17) on rows 0..15, FP32 out: %ld of %zu outputs differ\n", diff, a.size());
        HCK(hipFree(dx)); HCK(hipFree(dw)); HCK(hipFree(y1)); HCK(hipFree(y2));
    }
    if (argc > 1 && std::string(argv[1]) == "--p4b") {
        // P4-B: k2_gemm_sites_test's two prefill sites at m 3,000 (K3 hc up: FP32 out; K5 out_proj: BF16 out); k2_wmma
        // has no data-dependent route, so the fast fill stands for any data
        struct P { uint32_t n, k; bool f32; const char *name; } ps[] = {{10240, 320, true, "K3 hc up"}, {2560, 6144, false, "K5 out_proj"}};
        const uint32_t m = 3000;
        for (auto p : ps) {
            std::vector<uint16_t> hx(size_t(m) * p.k), hw(size_t(p.n) * p.k);
            for (auto &v : hx) v = uint16_t(0x3C00 + (xs() & 0x03FF) + ((xs() & 1) << 15));
            for (auto &v : hw) v = uint16_t(0x3C00 + (xs() & 0x03FF) + ((xs() & 1) << 15));
            void *dx, *dw, *dy;
            HCK(hipMalloc(&dx, hx.size() * 2)); HCK(hipMalloc(&dw, hw.size() * 2)); HCK(hipMalloc(&dy, size_t(m) * p.n * 4));
            HCK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); HCK(hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
            const double fl = 2.0 * m * p.n * p.k / 1e9;
            const T tb = time_calls([&] { if (gemm_wmma(B(dx), B(dw), dy, m, p.n, p.k, p.f32 ? WmmaOut::kF32 : WmmaOut::kBF16, nullptr) != CERE_OK) std::exit(2); }, 9);
            std::printf("hip timing  %-11s m %u n %5u k %4u %s out: k2_wmma %.3f ms %.1f TFLOP/s [%.3f..%.3f]\n", p.name, m, p.n, p.k,
                        p.f32 ? "FP32" : "BF16", tb.med, fl / tb.med, tb.mn, tb.mx);
            std::fflush(stdout);
            HCK(hipFree(dx)); HCK(hipFree(dw)); HCK(hipFree(dy));
        }
        return 0;
    }
    struct S { uint32_t n, k; const char *name; } shapes[] = {{10240, 2048, "in_proj_qkv"}, {6144, 2048, "in_proj_z"}, {2048, 6144, "out_proj"}, {2048, 2048, "2048 sq"}};
    const uint32_t MM = 4096;
    for (auto sh : shapes) {
        std::vector<uint16_t> hx(size_t(MM) * sh.k), hw(size_t(sh.n) * sh.k);
        std::vector<uint8_t> hq(size_t(sh.n) * sh.k), hs(size_t(sh.n) * sh.k / 32);
        for (auto &v : hx) v = uint16_t(0x3C00 + (xs() & 0x03FF) + ((xs() & 1) << 15));
        for (auto &v : hw) v = uint16_t(0x3C00 + (xs() & 0x03FF) + ((xs() & 1) << 15));
        for (auto &b : hq) { b = uint8_t(xs()); if ((b & 0x7f) == 0x7f) b ^= 1; }
        for (auto &b : hs) b = uint8_t(112 + xs() % 11);
        void *dx, *dw, *dy, *stg; uint8_t *dq, *ds;
        HCK(hipMalloc(&dx, hx.size() * 2)); HCK(hipMalloc(&dw, hw.size() * 2)); HCK(hipMalloc(&dy, size_t(MM) * sh.n * 2));
        HCK(hipMalloc(&stg, hw.size() * 2)); HCK(hipMalloc(&dq, hq.size())); HCK(hipMalloc(&ds, hs.size()));
        HCK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); HCK(hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
        HCK(hipMemcpy(dq, hq.data(), hq.size(), hipMemcpyHostToDevice)); HCK(hipMemcpy(ds, hs.data(), hs.size(), hipMemcpyHostToDevice));
        for (uint32_t m : {17u, 64u, 512u, 4096u}) {
            const int reps = m >= 4096 ? 7 : 15;
            const double fl = 2.0 * m * sh.n * sh.k / 1e9;
            const T tb = time_calls([&] { if (gemm_wmma(B(dx), B(dw), dy, m, sh.n, sh.k, WmmaOut::kBF16, nullptr) != CERE_OK) std::exit(2); }, reps);
            const T tm = time_calls([&] {
                HCK(mxfp8_dequantize(dq, ds, sh.n, sh.k, static_cast<__hip_bfloat16 *>(stg), nullptr));
                if (gemm_wmma(B(dx), B(stg), dy, m, sh.n, sh.k, WmmaOut::kBF16, nullptr) != CERE_OK) std::exit(2);
            }, reps);
            std::printf("hip timing  %-11s m %4u n %5u k %4u: bf16 k2_wmma %.3f ms %.1f TFLOP/s [%.3f..%.3f] | mxfp8 dequant+k2_wmma %.3f ms %.1f [%.3f..%.3f]\n",
                        sh.name, m, sh.n, sh.k, tb.med, fl / tb.med, tb.mn, tb.mx, tm.med, fl / tm.med, tm.mn, tm.mx);
            std::fflush(stdout);
        }
        HCK(hipFree(dx)); HCK(hipFree(dw)); HCK(hipFree(dy)); HCK(hipFree(stg)); HCK(hipFree(dq)); HCK(hipFree(ds));
    }
    return 0;
}
