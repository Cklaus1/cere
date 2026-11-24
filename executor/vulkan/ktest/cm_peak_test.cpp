// VulkanAutotune H7: FP16 and int8 cooperative-matrix peak on this device (shaders/cm_peak.comp), the roof table's
// coopmat_fp16 / int8_coopmat peaks (scripts/tune/vulkan_autotune.py peaks reads the `PEAK` lines). GPU timestamps;
// 3 warm-ups, then 15 samples of one dispatch each (grid: 8 workgroups per CU-ish, 2,560 workgroups x 4 subgroups),
// sized so one dispatch is ~20-60 ms (well under the 1 s submit budget). Prints every sample (H6 form) and the best
// and median rate. Usage: cm_peak_test [spv dir]
#include <algorithm>
#include <cstdio>
#include <vector>

#include "launch/vk_launch_testing.h"
#include "ktest_util.h"

int main(int argc, char **argv) {
    auto ctx_owner = cere::vk::testing::create_or_die(argc > 1 && argv[1][0] != '-' ? argv[1] : "build/vulkan/spv-tune");
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s, coopmat %d\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(),
                ctx.info().driver_info.c_str(), int(ctx.info().coopmat));
    if (!ctx.info().coopmat) { std::printf("FAIL no VK_KHR_cooperative_matrix\n"); return 1; }
    const uint32_t groups = 2560, nsg = 4, nacc = 8;
    auto src = kt::make(ctx, 1024), dst = kt::make(ctx, uint64_t(groups) * nsg * 256 * 4);
    std::vector<uint8_t> init(1024);
    for (size_t i = 0; i < init.size(); ++i) init[i] = (i & 1) ? 0x00 : uint8_t(1 + i % 3);  // small finite values
    std::memcpy(src.map, init.data(), init.size());
    int fail = 0;
    for (int int8 : {0, 1}) {
        const char *name = int8 ? "cm_peak_int8.spv" : "cm_peak_fp16.spv";
        struct { uint64_t src, dst; uint32_t iters, pad; } pc{src.addr, dst.addr, 0, 0};
        const cere::vk::Pipeline &p = ctx.pipeline(name, sizeof(pc), 32);
        // iteration count: grow until one dispatch takes >= 20 ms
        double ms = 0;
        for (pc.iters = 256; pc.iters <= (1u << 20); pc.iters *= 2) {
            ctx.begin(); ctx.stamp(0); kt::must(ctx.dispatch(p, &pc, groups), "dispatch"); ctx.stamp(1);
            kt::must(ctx.submit_wait(), "submit");
            ms = ctx.stamp_ms(0, 1);
            if (ms >= 20.0) break;
        }
        std::vector<double> t;
        for (int i = 0; i < 18; ++i) {
            ctx.begin(); ctx.stamp(0); kt::must(ctx.dispatch(p, &pc, groups), "dispatch"); ctx.stamp(1);
            kt::must(ctx.submit_wait(), "submit");
            if (i >= 3) t.push_back(ctx.stamp_ms(0, 1));
        }
        const double flop = double(groups) * nsg * pc.iters * nacc * 2.0 * 16 * 16 * 16;
        std::printf("TIMING cm_peak.%s warm n %zu ms", int8 ? "int8" : "fp16", t.size());
        for (double v : t) std::printf(" %.5f", v);
        std::printf("\n");
        std::sort(t.begin(), t.end());
        const double best = flop / (t.front() * 1e-3), med = flop / (t[t.size() / 2] * 1e-3);
        std::printf("PEAK %s ops_per_s %.4e median %.4e (iters %u, %.1f GFLOP per dispatch)\n",
                    int8 ? "int8_coopmat" : "coopmat_fp16", best, med, pc.iters, flop * 1e-9);
        if (!(best > 0)) ++fail;
    }
    std::printf("%s (%d failing)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}
