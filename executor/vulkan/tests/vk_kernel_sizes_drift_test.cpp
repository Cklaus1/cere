// VulkanAmd P3 T2: drift check of executor/vulkan/src/device/vk_kernel_sizes.h (the memory plan's restated workspace
// table) against the Vulkan kernels' own sizers. Links the kernel objects and libvulkan, but creates no Vulkan context,
// so it runs without a GPU. A FAIL names the entry whose restatement is stale: update vk_kernel_sizes.h.
#include "device/vk_kernel_sizes.h"

#include "kernels/k12_sampler_vk.h"
#include "kernels/k13_moe_bf16_vk.h"
#include "kernels/k13s_spec_accept_vk.h"
#include "kernels/k2_gemm_vk.h"
#include "kernels/k3_hc_vk.h"
#include "kernels/k4_ple_vk.h"
#include "kernels/k5_gdn_chunk_vk.h"
#include "kernels/k5_gdn_layer_vk.h"
#include "kernels/k6_attention_vk.h"
#include "kernels/k7_moe_vk.h"
#include "kernels/k9_qsa_vk.h"

#include <cstdio>
#include <string>

using namespace cere;
namespace sz = cere::vkdev::sizes;

static_assert(sz::kK4MaxBlocks == vkk4::kMaxBlocks, "K4 kMaxBlocks");
static_assert(sz::kK4BlockTableBytes == vkk4::kBlockTableBytes, "K4 block table");
static_assert(sz::kK4CallTableBytes == vkk4::kCallTableBytes, "K4 call table");
static_assert(sz::kK4Nvfp4RowBytes == vkk4::kNvfp4RowBytes && sz::kK4Nvfp4RowScales == vkk4::kNvfp4RowScales, "K4 rows");
static_assert(sz::kK5MaxTokens == vkk5::kChunkMaxTokens && sz::kK5ChunkTokens == vkk5::kChunkTokens, "K5 chunk");
static_assert(sz::kK6MaxContext == vkk6::kMaxContext && sz::kK6SplitMaxTokens == vkk6::kSplitMaxTokens, "K6");
static_assert(sz::kK9Ratio == vkk9::kRatio && sz::kK9RowWidth == vkk9::kRowWidth && sz::kK9Tiles == vkk9::kTiles &&
                  sz::kK9LogitsTileBytes == vkk9::kLogitsTileBytes && sz::kK9SplitMaxRows == vkk9::kSplitMaxRows,
              "K9");
static_assert(sz::kK3MaxRows == vkk3::kMaxRows, "K3");

namespace {
int fails = 0, passes = 0;
void cmp(const std::string &what, uint64_t restated, uint64_t real) {
    const bool ok = restated == real;
    std::printf("%-52s %s  restated %llu real %llu\n", what.c_str(), ok ? "PASS" : "FAIL (stale)",
                (unsigned long long)restated, (unsigned long long)real);
    (ok ? passes : fails)++;
}
}  // namespace

int main() {
    const uint64_t rows[] = {1, 4, 16, 17, 64, 511, 2048, 4096};
    const int64_t ctxs[] = {2049, 8192, 32768};
    // A4 / Flash-Next dims (host/support.cpp make_a4_config).
    const vkk7::MoeDims md{2560, 512, 10, 640, 640};
    const sz::MoeDims sd{2560, 512, 10, 640, 640};
    for (uint64_t m : rows) {
        const std::string at = " m=" + std::to_string(m);
        cmp("K2 gemm_workspace_bytes(m, 13312)" + at, sz::k2_gemm_workspace(m, 13312),
            vkk2::gemm_workspace_bytes(uint32_t(m), 13312));
        cmp("K3 hc_mix_workspace_bytes" + at, sz::k3_hc_mix_workspace(4, 2560, 320, m),
            vkk3::hc_mix_workspace_bytes(vkk3::HcShape{4, 2560, 320, 1e-6f}, uint32_t(m)));
        cmp("K4 workspace_bytes" + at, sz::k4_workspace(2560, 10240, 2560, int64_t(m)), vkk4::workspace_bytes(int64_t(m)));
        cmp("K5 chunk_workspace_bytes" + at, sz::k5_chunk_workspace(16, 48, 128, int64_t(m)),
            vkk5::chunk_workspace_bytes(int64_t(m)));
        cmp("K5 layer_workspace_bytes" + at, sz::k5_layer_workspace(16, 48, 128, int64_t(m)),
            vkk5::layer_workspace_bytes(int64_t(m)));
        cmp("K6 workspace_bytes" + at, sz::k6_workspace(24, 256, int64_t(m)), vkk6::workspace_bytes(int64_t(m)));
        // P4-E: the coopmat prefill attention's sizes (the memory plan charges these)
        cmp("K6 workspace_bytes kCoopmat" + at, sz::k6_workspace(24, 256, int64_t(m), true),
            vkk6::workspace_bytes(int64_t(m), vkk6::Options{64, 64, vkk6::Prefill::kCoopmat}));
        cmp("K7 moe_workspace_bytes" + at, sz::k7_moe_workspace(sd, m), vkk7::moe_workspace_bytes(md, uint32_t(m)));
        cmp("K13 moe_bf16_workspace_bytes" + at, sz::k13_moe_workspace(sd, m),
            vkk13::moe_bf16_workspace_bytes(md, uint32_t(m)));
        for (int64_t ctx : ctxs) {
            cmp("K9 scratch_bytes" + at + " ctx=" + std::to_string(ctx), sz::k9_scratch(24, 256, int64_t(m), ctx),
                vkk9::scratch_bytes(int64_t(m), ctx));
            // P4-E: the forward puts the coopmat flags and block routes at the start of carve_scratch's work region
            // (k9s.work, the select logits tile, dead by then): it must hold coopmat_scratch_bytes(rows) when rows > 16
            if (m > 16) {
                const size_t bytes = vkk9::scratch_bytes(int64_t(m), ctx);
                vkk9::Scratch sc{};
                const cere_status st = vkk9::carve_scratch(uint64_t(1) << 20, bytes, int64_t(m), ctx, &sc);
                const bool ok = st == CERE_OK && sc.work_bytes >= vkk9::coopmat_scratch_bytes(int64_t(m));
                std::printf("%-52s %s  work %llu coopmat %llu\n", ("K9 work >= coopmat_scratch_bytes" + at + " ctx=" +
                            std::to_string(ctx)).c_str(), ok ? "PASS" : "FAIL", (unsigned long long)sc.work_bytes,
                            (unsigned long long)vkk9::coopmat_scratch_bytes(int64_t(m)));
                (ok ? passes : fails)++;
            }
        }
    }
    cmp("K7 moe_weight_info_bytes", sz::k7_weight_info(sd), vkk7::moe_weight_info_bytes(md));
    cmp("K13 moe_bf16_weight_info_bytes", sz::k7_weight_info(sd), vkk13::moe_bf16_weight_info_bytes(md));
    cmp("K4 proj_info_bytes", sz::k4_proj_info(10240, 2560), vkk4::proj_info_bytes());
    cmp("K5 weight_info_bytes(2560)", sz::k5_weight_info(16, 48, 128, 2560), vkk5::weight_info_bytes(2560));
    cmp("K12 scratch_bytes", sz::kK12Scratch, vkk12::scratch_bytes());
    for (int64_t n : {0, 1, 3, 15, 16}) {
        cmp("K13S scratch_bytes n=" + std::to_string(n), sz::k13s_scratch(n), vkk13s::scratch_bytes(n));
    }
    std::printf("vk_kernel_sizes_drift_test: %d PASS, %d FAIL\n", passes, fails);
    return fails == 0 ? 0 : 1;
}
