// AmdAmd PAR-6 step 14: the small-M GEMV and the MXFP8 upcast (executor/hip/src/k2_mgemv.cpp) on gfx1151.
//  1. dequantize == the CPU upcast w = q * 2^(s - 127) bit for bit (random valid E4M3 bytes, scales over the whole
//     contract range [10, 246]);
//  2. gemv on the MXFP8 weight == gemv on its BF16 upcast, bit for bit, m = 1 .. 16, BF16 and FP32 outputs;
//  3. vs FP64: FP32 output |err| / sum|products| <= 1e-5; BF16 output within 1 ulp of the FP64 value;
//  4. row independence: each row of an m-row call (m = 2 .. 16, from rows 0 and 5) == its m = 1 call; repeats equal;
//  5. timing at decode shapes: GEMV BF16 / MXFP8 vs the pinned hipBLASLt bucket (GemmPlan::run_blas), m = 1, 4, 16.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "k2_gemm_hip.h"
#include "k2_mgemv.h"

using namespace cere::hipk2;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static double e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const double v = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
    return (b & 0x80) ? -v : v;
}
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
static const __hip_bfloat16 *B(const uint16_t *p) { return reinterpret_cast<const __hip_bfloat16 *>(p); }

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

int main() {
    std::mt19937 g(7); std::normal_distribution<float> nd;
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    {   // 1. dequantize, whole scale range
        const uint32_t n = 300, k = 1024;
        Mx w = make_mx(g, n, k, 0, 0);
        uint8_t *dq = dev(w.q), *ds = dev(w.s); uint16_t *dw; HCK(hipMalloc(&dw, w.up.size() * 2));
        HCK(mxfp8_dequantize(dq, ds, n, k, reinterpret_cast<__hip_bfloat16 *>(dw), nullptr));
        const bool ok = host(dw, w.up.size()) == w.up;
        std::printf("%-22s %s  300 x 1024, scales 10 .. 246\n", "dequantize exact", ok ? "PASS" : "FAIL");
        if (!ok) ++g_fail;
    }
    {   // 2-4
        const uint32_t n = 333, k = 2560, M = 21;
        Mx w = make_mx(g, n, k, 112, 122);
        auto x = rnd(size_t(M) * k, 1.f);
        uint8_t *dq = dev(w.q), *ds = dev(w.s); uint16_t *dwb = dev(w.up), *dx = dev(x);
        uint16_t *yb, *yb2; float *yf, *yf2;
        HCK(hipMalloc(&yb, M * n * 2)); HCK(hipMalloc(&yb2, M * n * 2)); HCK(hipMalloc(&yf, M * n * 4)); HCK(hipMalloc(&yf2, M * n * 4));
        long diff = 0;
        for (uint32_t m = 1; m <= kGemvMaxM; ++m) {
            for (bool f32 : {false, true}) {
                void *a = f32 ? (void *)yf : (void *)yb, *b = f32 ? (void *)yf2 : (void *)yb2;
                if (gemv(B(dx), dq, ds, a, m, n, k, f32, nullptr) != CERE_OK || gemv(B(dx), dwb, nullptr, b, m, n, k, f32, nullptr) != CERE_OK) return 2;
                if (f32) diff += host(yf, m * n) != host(yf2, m * n);
                else diff += host(yb, m * n) != host(yb2, m * n);
            }
        }
        std::printf("%-22s %s  333 x 2560, m 1 .. 16, BF16 + FP32 out: %ld calls differ\n", "MXFP8 == BF16 upcast", diff ? "FAIL" : "PASS", diff);
        if (diff) ++g_fail;
        if (gemv(B(dx), dq, ds, yf, kGemvMaxM, n, k, true, nullptr) != CERE_OK || gemv(B(dx), dq, ds, yb, kGemvMaxM, n, k, false, nullptr) != CERE_OK) return 2;
        auto hf = host(yf, kGemvMaxM * n); auto hb = host(yb, kGemvMaxM * n);
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
                if (gemv(B(dx + size_t(s0 + r) * k), dq, ds, yb2, 1, n, k, false, nullptr) != CERE_OK) return 2;
                one[r] = host(yb2, n);
            }
            for (uint32_t m = 2; m <= kGemvMaxM; ++m) {
                if (gemv(B(dx + size_t(s0) * k), dq, ds, yb, m, n, k, false, nullptr) != CERE_OK) return 2;
                auto h = host(yb, m * n);
                for (uint32_t r = 0; r < m; ++r) badr += std::memcmp(h.data() + r * n, one[r].data(), n * 2) != 0;
            }
        }
        if (gemv(B(dx), dq, ds, yb2, kGemvMaxM, n, k, false, nullptr) != CERE_OK) return 2;
        const bool rep = host(yb2, kGemvMaxM * n) == hb;
        std::printf("%-22s %s  rows of m 2 .. 16 calls (from 0, 5) vs m = 1: %ld differ; repeat %s\n", "row independence", badr == 0 && rep ? "PASS" : "FAIL", badr, rep ? "bit-equal" : "DIFFERS");
        if (badr || !rep) ++g_fail;
    }
    {   // 5. timing
        hipblasLtHandle_t lt; hipblasLtCreate(&lt);
        struct S { uint32_t n, k; const char *name; } shapes[] = {{8192, 2560, "n 8192 k 2560"}, {2560, 4096, "n 2560 k 4096"}, {2560, 640, "n 2560 k 640"}, {248320, 2560, "lm_head"}};
        for (auto sh : shapes) {
            Mx w = make_mx(g, sh.n, sh.k, 112, 122);
            auto x = rnd(size_t(16) * sh.k, 1.f);
            uint8_t *dq = dev(w.q), *ds = dev(w.s); uint16_t *dwb = dev(w.up), *dx = dev(x), *dy;
            HCK(hipMalloc(&dy, size_t(16) * sh.n * 2));
            GemmPlan p; if (p.init(lt, sh.n, sh.k, GemmOut::kBF16, 16) != CERE_OK) return 2;
            hipEvent_t e0, e1; HCK(hipEventCreate(&e0)); HCK(hipEventCreate(&e1));
            for (uint32_t m : {1u, 4u, 16u}) {
                auto time = [&](auto fn) { fn(); HCK(hipEventRecord(e0, nullptr)); for (int i = 0; i < 20; ++i) fn(); HCK(hipEventRecord(e1, nullptr)); HCK(hipEventSynchronize(e1)); float ms; HCK(hipEventElapsedTime(&ms, e0, e1)); return ms / 20; };
                const float tb = time([&] { gemv(B(dx), dwb, nullptr, dy, m, sh.n, sh.k, false, nullptr); });
                const float tm = time([&] { gemv(B(dx), dq, ds, dy, m, sh.n, sh.k, false, nullptr); });
                const float tl = time([&] { p.run_blas(dx, dwb, dy, m, nullptr); });
                const double gb = double(sh.n) * sh.k / 1e6;  // MB per byte-per-element
                std::printf("%-22s       %-14s m %2u: GEMV bf16 %.3f ms (%.0f GB/s), mxfp8 %.3f ms (%.0f GB/s), hipBLASLt %.3f ms (%.0f GB/s)\n",
                            "timing", sh.name, m, tb, 2 * gb / tb, tm, gb / tm, tl, 2 * gb / tl);
            }
            HCK(hipFree(dq)); HCK(hipFree(ds)); HCK(hipFree(dwb)); HCK(hipFree(dx)); HCK(hipFree(dy));
        }
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
