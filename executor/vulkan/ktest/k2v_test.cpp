// VulkanAmd P1: executor/hip/ktest/k2v_test.cpp's contract rows on the Vulkan small-M GEMV (shaders/k2_mgemv.comp).
// Contract (per device, vulkan-amd design note "Contracts"): y[m, n] = x[m, k] · w[n, k]^T for m in [1, 16] in a
// fixed per-output FMA order that does not depend on m or on the other rows. Checks per shape: rel-L2 vs FP64 (limit
// 4e-3 BF16 out, 1e-4 F32, the HIP test's), every row of an m = 16 call bit-equal to the m = 1 call on that row (batch
// invariance), repeats bit-identical; then m = 1 weight-streaming time (GPU timestamps, median of 15).
// Same shapes, seeds and data generation as the HIP test. The HIP test runs the gemvx (cuBLASLt-order) kernel with an
// order (T, S) per shape; the Vulkan kernel's order is its own fixed mapping (G lanes, E elements per lane per step,
// chosen from k): the contract is per device, not bit-equality with CUDA/HIP (spec Non-goals).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "launch/vk_launch_testing.h"
#include "ktest_util.h"

using kt::bf2f;
using kt::f2bf;
static int g_fail = 0;

struct Case { const char *name; uint32_t n, k; };

int main(int argc, char **argv) {
    auto ctx_owner = cere::vk::testing::create_or_die(argc > 1 ? argv[1] : "build/vulkan/spv");
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s, subgroup %u..%u\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(),
                ctx.info().driver_info.c_str(), ctx.info().subgroup_min, ctx.info().subgroup_max);
    std::mt19937 g(11); std::normal_distribution<float> d(0.f, 1.f);
    const std::vector<Case> cases = {
        {"q|gate 2560->12288", 12288, 2560},
        {"o_proj 6144->2560", 2560, 6144},
        {"lm_head 2560->248320", 248320, 2560},
        {"shared down 640->2560", 2560, 640},
    };
    for (const auto &c : cases) {
        const uint32_t M = cere::vkk2::kGemvMaxM;
        const auto mp = cere::vkk2::mapping_for(c.k, 1);
        std::vector<uint16_t> x(size_t(M) * c.k), w(size_t(c.n) * c.k);
        for (auto &v : x) v = f2bf(d(g));
        for (auto &v : w) v = f2bf(d(g) * 0.05f);
        auto dx = kt::upload(ctx, x), dw = kt::upload(ctx, w), dy = kt::make(ctx, size_t(M) * c.n * 4);
        for (bool f32 : {false, true}) {
            const size_t es = f32 ? 4 : 2;
            auto run = [&](uint64_t xa, uint32_t m) {
                kt::gemv_sync(ctx, xa, dw.addr, 0, dy.addr, m, c.n, c.k, f32);
                return kt::host<uint8_t>(dy, size_t(m) * c.n * es);
            };
            auto all = run(dx.addr, M), again = run(dx.addr, M);
            bool invariant = true;
            for (uint32_t r = 0; r < M && invariant; ++r) {
                auto one = run(dx.addr + uint64_t(r) * c.k * 2, 1);
                invariant = std::memcmp(one.data(), all.data() + size_t(r) * c.n * es, c.n * es) == 0;
            }
            double num = 0, den = 0; std::uniform_int_distribution<uint32_t> col(0, c.n - 1);
            for (uint32_t r = 0; r < M; ++r) for (int t = 0; t < 48; ++t) {
                const uint32_t j = col(g); double ref = 0;
                for (uint32_t i = 0; i < c.k; ++i) ref += double(bf2f(x[size_t(r) * c.k + i])) * bf2f(w[size_t(j) * c.k + i]);
                float got;
                if (es == 2) { uint16_t b; std::memcpy(&b, &all[(size_t(r) * c.n + j) * 2], 2); got = bf2f(b); }
                else std::memcpy(&got, &all[(size_t(r) * c.n + j) * 4], 4);
                num += (got - ref) * (got - ref); den += ref * ref;
            }
            const double e = std::sqrt(num / den), lim = f32 ? 1e-4 : 4e-3;
            const bool ok = e <= lim && invariant && all == again;
            if (!ok) ++g_fail;
            std::printf("%-24s (G%u,E%u) %s rel-L2 %.2e (limit %.0e)  m=1 vs m=16 rows %s  repeat %s  %s\n", c.name, mp.G, mp.E,
                        f32 ? "f32 " : "bf16", e, lim, invariant ? "bit-equal" : "DIFFER", all == again ? "bit-identical" : "DIFFERS",
                        ok ? "PASS" : "FAIL");
        }
        const auto t = kt::time_gemv(ctx, dx.addr, dw.addr, 0, 0, 0, 1, dy.addr, 1, c.n, c.k);
        std::printf("    m=1: %.1f us (min %.1f, max %.1f), weights %.0f GB/s%s\n", t.ms_median * 1e3, t.ms_min * 1e3, t.ms_max * 1e3,
                    double(c.n) * c.k * 2 / (t.ms_median * 1e-3) / 1e9, double(c.n) * c.k * 2 < 32e6 ? " (fits the 32 MB MALL: not the measure)" : "");
        ctx.free(dx); ctx.free(dw); ctx.free(dy);
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
