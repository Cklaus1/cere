// AmdAmd PAR-6 step 2b: the HIP K2 GEMM (executor/hip/src/k2_gemm_hip.cpp) on Flash-Next's dense shapes.
// Per (shape, out type, m): rel-L2 vs an FP64 CPU reference on 48 sampled columns (limit 4e-3 for BF16 output,
// 1e-4 for F32), and two runs bit-identical; then the pinned kernels per bucket and the m = 2048 throughput.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "k2_gemm_hip.h"

using cere::hipk2::GemmOut;
using cere::hipk2::GemmPlan;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)

static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

struct Shape { const char *name; uint32_t n, k; std::vector<uint32_t> ms; };

int main() {
    hipblasLtHandle_t h; if (hipblasLtCreate(&h) != HIPBLAS_STATUS_SUCCESS) { std::printf("hipblasLtCreate failed\n"); return 2; }
    std::mt19937 g(7); std::normal_distribution<float> d(0.f, 1.f);
    const std::vector<Shape> shapes = {
        {"q|gate proj 2560->12288", 12288, 2560, {1, 3, 4, 7, 16, 100, 512, 2048}},
        {"kv proj 2560->512", 512, 2560, {1, 4, 16, 512, 2048}},
        {"o_proj 6144->2560", 2560, 6144, {1, 2, 16, 512, 2048}},
        {"lm_head 2560->248320", 248320, 2560, {1, 4}},
    };
    for (const auto &s : shapes) {
        const uint32_t max_m = s.ms.back();
        std::vector<uint16_t> x(size_t(max_m) * s.k), w(size_t(s.n) * s.k);
        for (auto &v : x) v = f2bf(d(g)); for (auto &v : w) v = f2bf(d(g) * 0.05f);
        void *dx, *dw, *dy; HCK(hipMalloc(&dx, x.size() * 2)); HCK(hipMalloc(&dw, w.size() * 2)); HCK(hipMalloc(&dy, size_t(max_m) * s.n * 4));
        HCK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice)); HCK(hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice));
        std::vector<uint32_t> cols(48); for (auto &c : cols) c = g() % s.n;
        for (GemmOut out : {GemmOut::kBF16, GemmOut::kF32}) {
            GemmPlan plan;
            if (plan.init(h, s.n, s.k, out, max_m) != CERE_OK) { std::printf("%-26s init FAILED\n", s.name); ++g_fail; continue; }
            const size_t es = out == GemmOut::kBF16 ? 2 : 4;
            for (uint32_t m : s.ms) {
                auto run = [&] { if (plan.run(dx, dw, dy, m, nullptr) != CERE_OK) { std::printf("run failed\n"); std::exit(2); } HCK(hipDeviceSynchronize());
                                 std::vector<uint8_t> r(size_t(m) * s.n * es); HCK(hipMemcpy(r.data(), dy, r.size(), hipMemcpyDeviceToHost)); return r; };
                auto y1 = run(), y2 = run();
                double num = 0, den = 0;
                for (uint32_t r = 0; r < m; ++r) for (uint32_t c : cols) {
                    double ref = 0; for (uint32_t i = 0; i < s.k; ++i) ref += double(bf2f(x[size_t(r) * s.k + i])) * bf2f(w[size_t(c) * s.k + i]);
                    float got; if (es == 2) { uint16_t b; std::memcpy(&b, &y1[(size_t(r) * s.n + c) * 2], 2); got = bf2f(b); } else std::memcpy(&got, &y1[(size_t(r) * s.n + c) * 4], 4);
                    num += (got - ref) * (got - ref); den += ref * ref;
                }
                const double e = std::sqrt(num / den), lim = out == GemmOut::kBF16 ? 4e-3 : 1e-4;
                const bool ok = e <= lim && y1 == y2;
                if (!ok) ++g_fail;
                std::printf("%-26s %s m=%-5u rel-L2 %.2e (limit %.0e) repeat %s  %s\n", s.name, out == GemmOut::kBF16 ? "bf16" : "f32 ", m, e, lim, y1 == y2 ? "bit-identical" : "DIFFERS", ok ? "PASS" : "FAIL");
            }
            if (out == GemmOut::kBF16) {
                for (const auto &p : plan.pins()) std::printf("    bucket %u-%u: solution %d %s\n", p.m_lo, p.m_hi, p.solution_index, p.kernel.substr(0, 60).c_str());
                if (max_m >= 2048) {
                    for (int i = 0; i < 3; ++i) plan.run(dx, dw, dy, 2048, nullptr);
                    HCK(hipDeviceSynchronize()); auto t0 = std::chrono::steady_clock::now();
                    const int reps = 20; for (int i = 0; i < reps; ++i) plan.run(dx, dw, dy, 2048, nullptr);
                    HCK(hipDeviceSynchronize()); const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
                    std::printf("    m=2048 throughput: %.3f ms, %.1f TFLOP/s\n", sec * 1e3, 2.0 * 2048 * s.n * s.k / sec / 1e12);
                }
            }
        }
        HCK(hipFree(dx)); HCK(hipFree(dw)); HCK(hipFree(dy));
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
