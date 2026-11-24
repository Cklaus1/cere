// AmdAmd PAR-6 step 2c: K2 gemvx-order rows kernel (executor/cuda/src/kernels/k2_gemv.cu, hipified) on gfx1151.
// Contract (k2_gemv.h): y[m, n] = x[m, k] · w[n, k]^T for m in [1, 16] in a fixed per-output FMA order that does not
// depend on m or on the other rows. Checks per shape/order: rel-L2 vs FP64 (limit 4e-3 BF16 out, 1e-4 F32), every
// row of an m = 16 launch bit-equal to the m = 1 launch on that row (batch invariance), repeats bit-identical;
// then m = 1 weight-streaming bandwidth (the decode cost).
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k2_gemv.h"

namespace k2v = cere::kernels::k2v;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

struct Case { const char *name; uint32_t n, k; k2v::Order o; };

int main() {
    std::mt19937 g(11); std::normal_distribution<float> d(0.f, 1.f);
    const std::vector<Case> cases = {
        {"q|gate 2560->12288 (T32,S1)", 12288, 2560, {32, 1}},
        {"o_proj 6144->2560 (T32,S1)", 2560, 6144, {32, 1}},
        {"lm_head 2560->248320 (T8,S1)", 248320, 2560, {8, 1}},
        {"shared down 640->2560 (T32,S20)", 2560, 640, {32, 20}},
    };
    for (const auto &c : cases) {
        if (!k2v::supported(c.o, c.k)) { std::printf("%-34s order unsupported at k=%u\n", c.name, c.k); ++g_fail; continue; }
        const uint32_t M = k2v::kMaxM;
        std::vector<uint16_t> x(size_t(M) * c.k), w(size_t(c.n) * c.k);
        for (auto &v : x) v = f2bf(d(g)); for (auto &v : w) v = f2bf(d(g) * 0.05f);
        void *dx, *dw, *dy; HCK(hipMalloc(&dx, x.size() * 2)); HCK(hipMalloc(&dw, w.size() * 2)); HCK(hipMalloc(&dy, size_t(M) * c.n * 4));
        HCK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice)); HCK(hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice));
        for (bool f32 : {false, true}) {
            const size_t es = f32 ? 4 : 2;
            auto run = [&](const void *xp, uint32_t m) {
                HCK(k2v::gemv(static_cast<const __hip_bfloat16 *>(xp), m, c.k, static_cast<const __hip_bfloat16 *>(dw), c.n, dy, f32, c.o, nullptr));
                HCK(hipDeviceSynchronize()); std::vector<uint8_t> r(size_t(m) * c.n * es); HCK(hipMemcpy(r.data(), dy, r.size(), hipMemcpyDeviceToHost)); return r;
            };
            auto all = run(dx, M), again = run(dx, M);
            bool invariant = true;
            for (uint32_t r = 0; r < M && invariant; ++r) {
                auto one = run(static_cast<const uint8_t *>(dx) + size_t(r) * c.k * 2, 1);
                invariant = std::memcmp(one.data(), all.data() + size_t(r) * c.n * es, c.n * es) == 0;
            }
            double num = 0, den = 0; std::uniform_int_distribution<uint32_t> col(0, c.n - 1);
            for (uint32_t r = 0; r < M; ++r) for (int t = 0; t < 48; ++t) {
                const uint32_t j = col(g); double ref = 0;
                for (uint32_t i = 0; i < c.k; ++i) ref += double(bf2f(x[size_t(r) * c.k + i])) * bf2f(w[size_t(j) * c.k + i]);
                float got; if (es == 2) { uint16_t b; std::memcpy(&b, &all[(size_t(r) * c.n + j) * 2], 2); got = bf2f(b); } else std::memcpy(&got, &all[(size_t(r) * c.n + j) * 4], 4);
                num += (got - ref) * (got - ref); den += ref * ref;
            }
            const double e = std::sqrt(num / den), lim = f32 ? 1e-4 : 4e-3;
            const bool ok = e <= lim && invariant && all == again;
            if (!ok) ++g_fail;
            std::printf("%-34s %s rel-L2 %.2e (limit %.0e)  m=1 vs m=16 rows %s  repeat %s  %s\n", c.name, f32 ? "f32 " : "bf16", e, lim,
                        invariant ? "bit-equal" : "DIFFER", all == again ? "bit-identical" : "DIFFERS", ok ? "PASS" : "FAIL");
        }
        for (int i = 0; i < 3; ++i) k2v::gemv(static_cast<const __hip_bfloat16 *>(dx), 1, c.k, static_cast<const __hip_bfloat16 *>(dw), c.n, dy, false, c.o, nullptr);
        HCK(hipDeviceSynchronize()); auto t0 = std::chrono::steady_clock::now(); const int reps = 30;
        for (int i = 0; i < reps; ++i) k2v::gemv(static_cast<const __hip_bfloat16 *>(dx), 1, c.k, static_cast<const __hip_bfloat16 *>(dw), c.n, dy, false, c.o, nullptr);
        HCK(hipDeviceSynchronize()); const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("    m=1: %.1f us, weights %.0f GB/s\n", sec * 1e6, double(c.n) * c.k * 2 / sec / 1e9);
        HCK(hipFree(dx)); HCK(hipFree(dw)); HCK(hipFree(dy));
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
