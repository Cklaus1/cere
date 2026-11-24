// VulkanAmd P2: executor/hip/ktest/k2_test.cpp's rows on the Vulkan K2 routing (k2_gemm_vk.h run(): row class <= 16
// -> the small-M GEMV, > 16 -> the large-m GEMM), on Flash-Next's dense shapes. Per (shape, weight format, out type,
// m): rel-L2 vs an FP64 CPU reference on 48 sampled columns (limit 4e-3 for BF16 output, 1e-4 for FP32: the HIP
// test's), and two runs bit-identical. The HIP test's rows are the BF16-weight rows (same shapes, m lists, seed and
// data generation); the MXFP8-weight rows repeat them on an MXFP8 weight (k2mx_test's generator, block scales
// 112 .. 122) against the FP64 product with its exact BF16 upcast (the S3 artifact stores the dense projections
// MXFP8, which the HIP adapter runs as GEMV (<= 16) or dequantize + k2_wmma). HIP prints its pinned hipBLASLt
// buckets; the Vulkan routing has none and prints the route instead. Then the m = 2,048 throughput of run().
// Usage: k2_test [spv dir]. Exit 1 on any FAIL.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "launch/vk_launch_testing.h"
#include "kernels/k2_gemm_vk.h"
#include "ktest_util.h"

using kt::bf2f;
using kt::f2bf;
namespace vkk2 = cere::vkk2;
static int g_fail = 0;

struct Shape { const char *name; uint32_t n, k; std::vector<uint32_t> ms; };

static uint64_t g_xs = 0x9E3779B97F4A7C15ull;
static uint64_t xs() { g_xs ^= g_xs << 13; g_xs ^= g_xs >> 7; g_xs ^= g_xs << 17; return g_xs; }
static double e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const double v = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
    return (b & 0x80) ? -v : v;
}

int main(int argc, char **argv) {
    auto ctx_owner = cere::vk::testing::create_or_die(argc > 1 ? argv[1] : "build/vulkan/spv");
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str());
    if (vkk2::gemm_probe(ctx) != CERE_OK) { std::printf("gemm_probe refused: FAIL\n"); return 1; }
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
        for (auto &v : x) v = f2bf(d(g));
        for (auto &v : w) v = f2bf(d(g) * 0.05f);
        std::vector<uint32_t> cols(48); for (auto &c : cols) c = g() % s.n;
        // MXFP8 weight (fast fill: valid E4M3 bytes, scales 112 .. 122); its upcast only on the sampled columns
        std::vector<uint8_t> q(size_t(s.n) * s.k), sc(size_t(s.n) * s.k / 32);
        for (auto &b : q) { b = uint8_t(xs()); if ((b & 0x7f) == 0x7f) b ^= 1; }
        for (auto &b : sc) b = uint8_t(112 + xs() % 11);
        auto dx = kt::upload(ctx, x), dw = kt::upload(ctx, w), dq = kt::upload(ctx, q), ds = kt::upload(ctx, sc);
        auto dy = kt::make(ctx, size_t(max_m) * s.n * 4), dy2 = kt::make(ctx, size_t(max_m) * s.n * 4);
        auto ws = kt::make(ctx, vkk2::gemm_workspace_bytes(max_m, s.n));
        for (int fmt = 0; fmt < 2; ++fmt) {
            const bool mx = fmt == 1;
            // FP64 reference for every row and sampled column
            std::vector<double> ref(size_t(max_m) * cols.size());
            std::vector<double> wcol(s.k);
            for (size_t ci = 0; ci < cols.size(); ++ci) {
                const uint32_t c = cols[ci];
                for (uint32_t i = 0; i < s.k; ++i)
                    wcol[i] = mx ? std::ldexp(e4m3(q[size_t(c) * s.k + i]), int(sc[(size_t(c) * s.k + i) / 32]) - 127) : double(bf2f(w[size_t(c) * s.k + i]));
                for (uint32_t r = 0; r < max_m; ++r) {
                    double a = 0; for (uint32_t i = 0; i < s.k; ++i) a += double(bf2f(x[size_t(r) * s.k + i])) * wcol[i];
                    ref[size_t(r) * cols.size() + ci] = a;
                }
            }
            const uint64_t wa = mx ? dq.addr : dw.addr, sa = mx ? ds.addr : 0;
            for (bool f32 : {false, true}) {
                const size_t es = f32 ? 4 : 2;
                for (uint32_t m : s.ms) {
                    for (auto *b : {&dy, &dy2}) {
                        ctx.begin();
                        kt::must(vkk2::run(ctx, dx.addr, wa, sa, b->addr, m, s.n, s.k, f32, 0, ws.addr), "run");
                        ctx.barrier();
                        ctx.submit_wait();
                    }
                    double num = 0, den = 0;
                    for (uint32_t r = 0; r < m; ++r) for (size_t ci = 0; ci < cols.size(); ++ci) {
                        const size_t o = size_t(r) * s.n + cols[ci];
                        float got;
                        if (es == 2) got = bf2f(kt::host<uint16_t>(dy, 1, o * 2)[0]); else got = kt::host<float>(dy, 1, o * 4)[0];
                        const double rf = ref[size_t(r) * cols.size() + ci];
                        num += (got - rf) * (got - rf); den += rf * rf;
                    }
                    const bool rep = std::memcmp(dy.map, dy2.map, size_t(m) * s.n * es) == 0;
                    const double e = std::sqrt(num / den), lim = f32 ? 1e-4 : 4e-3;
                    const bool ok = e <= lim && rep;
                    if (!ok) ++g_fail;
                    std::printf("%-26s %s %s m=%-5u rel-L2 %.2e (limit %.0e) repeat %s  [%s]  %s\n", s.name, mx ? "mxfp8" : "bf16 ", f32 ? "f32 " : "bf16",
                                m, e, lim, rep ? "bit-identical" : "DIFFERS", m < vkk2::kGemmMinM ? "GEMV" : "GEMM", ok ? "PASS" : "FAIL");
                    std::fflush(stdout);
                }
            }
        }
        if (max_m >= 2048) {
            for (int fmt = 0; fmt < 2; ++fmt) {
                const uint64_t wa = fmt ? dq.addr : dw.addr, sa = fmt ? ds.addr : 0;
                ctx.begin();
                for (int i = 0; i < 2; ++i) { kt::must(vkk2::run(ctx, dx.addr, wa, sa, dy.addr, 2048, s.n, s.k, false, 0, ws.addr), "run"); ctx.barrier(); }
                const int reps = 20;
                ctx.stamp(0);
                for (int i = 0; i < reps; ++i) { kt::must(vkk2::run(ctx, dx.addr, wa, sa, dy.addr, 2048, s.n, s.k, false, 0, ws.addr), "run"); ctx.barrier(); }
                ctx.stamp(1);
                ctx.submit_wait();
                const double ms = ctx.stamp_ms(0, 1) / reps;
                std::printf("    %s m=2048 throughput (run(), weight scanned in the call): %.3f ms, %.1f TFLOP/s\n", fmt ? "mxfp8" : "bf16 ", ms, 2.0 * 2048 * s.n * s.k / (ms * 1e-3) / 1e12);
            }
        }
        for (auto *b : {&dx, &dw, &dq, &ds, &dy, &dy2, &ws}) ctx.free(*b);
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
