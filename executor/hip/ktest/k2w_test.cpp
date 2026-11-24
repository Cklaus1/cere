// AmdAmd PAR-6 K2-SHIFT: the shift-invariant WMMA GEMM (executor/hip/src/k2_wmma.cpp) on gfx1151.
//  1. vs FP64 (m 200, n 300, k 256: partial tiles both ways): every BF16 output within 1 ulp of the FP64 product
//     rounded to BF16, FP32 output rel error <= 1e-5 per element (FP32 accumulation of exact BF16 products);
//  2. shift invariance: a call on rows s.. of x equals rows s.. of the full call bit for bit (s = 1, 17, 64, 100);
//     row-count invariance: m = 1, 37, 128 at row 0 equal the full call's first rows; repeat calls bit-identical;
//  3. timing at the model's prefill shapes, m = 4,096, direct and through the HIP plan (k2_gemm_hip: m > 16 routes to
//     this GEMM, so both columns are k2_wmma; the plan adds its dispatch).
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "k2_gemm_hip.h"
#include "k2_wmma.h"

using namespace cere::hipk2;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
static const __hip_bfloat16 *B(const uint16_t *p) { return reinterpret_cast<const __hip_bfloat16 *>(p); }

int main() {
    std::mt19937 g(23); std::normal_distribution<float> nd;
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    {   // 1. vs FP64
        const uint32_t m = 200, n = 300, k = 256;
        auto x = rnd(size_t(m) * k, 1.f), w = rnd(size_t(n) * k, 0.1f);
        auto *dx = dev(x), *dw = dev(w); uint16_t *dy; float *dyf; HCK(hipMalloc(&dy, m * n * 2)); HCK(hipMalloc(&dyf, m * n * 4));
        if (gemm_wmma(B(dx), B(dw), dy, m, n, k, WmmaOut::kBF16, nullptr) != CERE_OK || gemm_wmma(B(dx), B(dw), dyf, m, n, k, WmmaOut::kF32, nullptr) != CERE_OK) return 2;
        auto y = host(dy, m * n); auto yf = host(dyf, m * n);
        long bad = 0; double worst = 0;
        for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) {
            double s = 0, sa = 0; for (uint32_t t = 0; t < k; ++t) { const double p = double(bf2f(x[i * k + t])) * bf2f(w[j * k + t]); s += p; sa += std::fabs(p); }
            const int ulp = std::abs(int(y[i * n + j]) - int(f2bf(float(s))));
            bad += ulp > 1;
            worst = std::max(worst, std::fabs(yf[i * n + j] - s) / (sa + 1e-30));
        }
        const bool ok = bad == 0 && worst <= 1e-5;
        std::printf("%-22s %s  200 x 300 x 256: %ld BF16 outputs > 1 ulp from FP64; FP32 worst |err| / sum|products| %.1e\n", "vs FP64", ok ? "PASS" : "FAIL", bad, worst);
        if (!ok) ++g_fail;
    }
    {   // 2. shift / row-count invariance, determinism
        const uint32_t M = 512, n = 10240, k = 2048;
        auto x = rnd(size_t(M) * k, 1.f), w = rnd(size_t(n) * k, 0.05f);
        auto *dx = dev(x), *dw = dev(w); uint16_t *dy, *dy2; HCK(hipMalloc(&dy, size_t(M) * n * 2)); HCK(hipMalloc(&dy2, size_t(M) * n * 2));
        if (gemm_wmma(B(dx), B(dw), dy, M, n, k, WmmaOut::kBF16, nullptr) != CERE_OK) return 2;
        auto full = host(dy, size_t(M) * n);
        long bad = 0;
        for (uint32_t s : {1u, 17u, 64u, 100u}) {
            if (gemm_wmma(B(dx + size_t(s) * k), B(dw), dy2, M - s, n, k, WmmaOut::kBF16, nullptr) != CERE_OK) return 2;
            auto part = host(dy2, size_t(M - s) * n); for (size_t e = 0; e < part.size(); ++e) bad += part[e] != full[size_t(s) * n + e];
        }
        long badm = 0;
        for (uint32_t m : {1u, 37u, 128u}) {
            if (gemm_wmma(B(dx), B(dw), dy2, m, n, k, WmmaOut::kBF16, nullptr) != CERE_OK) return 2;
            auto part = host(dy2, size_t(m) * n); for (size_t e = 0; e < part.size(); ++e) badm += part[e] != full[e];
        }
        if (gemm_wmma(B(dx), B(dw), dy2, M, n, k, WmmaOut::kBF16, nullptr) != CERE_OK) return 2;
        const bool rep = host(dy2, size_t(M) * n) == full;
        const bool ok = bad == 0 && badm == 0 && rep;
        std::printf("%-22s %s  512 x 10240 x 2048: shifted calls (1, 17, 64, 100) %ld differ; m 1/37/128 %ld differ; repeat %s\n", "shift invariance", ok ? "PASS" : "FAIL", bad, badm, rep ? "bit-equal" : "DIFFERS");
        if (!ok) ++g_fail;
    }
    {   // 3. timing
        hipblasLtHandle_t lt; hipblasLtCreate(&lt);
        struct S { uint32_t n, k; const char *name; } shapes[] = {{10240, 2048, "in_proj_qkv"}, {6144, 2048, "in_proj_z"}, {2048, 6144, "out_proj"}, {2048, 2048, "2048 sq"}};
        const uint32_t m = 4096;
        for (auto sh : shapes) {
            auto x = rnd(size_t(m) * sh.k, 1.f), w = rnd(size_t(sh.n) * sh.k, 0.05f);
            auto *dx = dev(x), *dw = dev(w); uint16_t *dy; HCK(hipMalloc(&dy, size_t(m) * sh.n * 2));
            hipEvent_t e0, e1; HCK(hipEventCreate(&e0)); HCK(hipEventCreate(&e1)); float ms_w = 0, ms_b = 0;
            gemm_wmma(B(dx), B(dw), dy, m, sh.n, sh.k, WmmaOut::kBF16, nullptr); HCK(hipEventRecord(e0, nullptr));
            for (int i = 0; i < 5; ++i) gemm_wmma(B(dx), B(dw), dy, m, sh.n, sh.k, WmmaOut::kBF16, nullptr);
            HCK(hipEventRecord(e1, nullptr)); HCK(hipEventSynchronize(e1)); HCK(hipEventElapsedTime(&ms_w, e0, e1)); ms_w /= 5;
            GemmPlan p; if (p.init(lt, sh.n, sh.k, GemmOut::kBF16, m) != CERE_OK) return 2;
            p.run(dx, dw, dy, m, nullptr); HCK(hipEventRecord(e0, nullptr));
            for (int i = 0; i < 5; ++i) p.run(dx, dw, dy, m, nullptr);
            HCK(hipEventRecord(e1, nullptr)); HCK(hipEventSynchronize(e1)); HCK(hipEventElapsedTime(&ms_b, e0, e1)); ms_b /= 5;
            const double fl = 2.0 * m * sh.n * sh.k / 1e9;
            std::printf("%-22s       m 4096 n %5u k %4u (%s): wmma %.3f ms %.1f TFLOP/s, via the HIP plan %.3f ms %.1f TFLOP/s\n", "timing", sh.n, sh.k, sh.name, ms_w, fl / ms_w, ms_b, fl / ms_b);
            HCK(hipFree(dx)); HCK(hipFree(dw)); HCK(hipFree(dy));
        }
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
