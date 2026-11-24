// VulkanAmd P1: see k2_mgemv_vk.h.
#include "kernels/k2_mgemv_vk.h"
#if defined(CERE_VK_DISPATCH_PROFILE)
#include <cstdio>
#include <cstdlib>
#endif

namespace cere::vkk2 {
namespace {

struct GemvPush {
    uint64_t x, w, s, y;
    uint32_t m, n, k, out_f32;
};
static_assert(sizeof(GemvPush) == 48, "push layout matches k2_mgemv.comp");

struct DequantPush {
    uint64_t q, s, w;
    uint32_t groups16, k;
};
static_assert(sizeof(DequantPush) == 32, "push layout matches k2_mxfp8_dequant.comp");

bool aligned16(uint64_t a) { return a % 16 == 0; }

uint32_t row_capacity(uint32_t m) { return m == 1 ? 1 : m == 2 ? 2 : m <= 4 ? 4 : m <= 8 ? 8 : 16; }

#if defined(CERE_VK_DISPATCH_PROFILE)
}  // namespace
}  // namespace cere::vkk2
extern "C" uint32_t cere_vk_dprof_step();  // vk_launch.cpp: DECODE calls so far (the forward_s3_dprof test build only)
namespace cere::vkk2 {
namespace {
// TEST BUILD ONLY: env CERE_VK_GEMV_ROTATE = "k=WG,U,J|WG,U,J|...;k=...": the m = 1 GEMV of that k uses alternative
// (decode call index % count) for WG / U / J; G and E (the order) stay mapping_for's, so the bits do not change.
Mapping dprof_rotate(uint32_t k, Mapping mp) {
    static const std::vector<std::pair<uint32_t, std::vector<Mapping>>> table = [] {
        std::vector<std::pair<uint32_t, std::vector<Mapping>>> t;
        const char *env = std::getenv("CERE_VK_GEMV_ROTATE");
        std::string s = env ? env : "";
        size_t p = 0;
        while (p < s.size()) {
            size_t e = s.find(';', p);
            if (e == std::string::npos) e = s.size();
            const std::string item = s.substr(p, e - p);
            p = e + 1;
            const size_t eq = item.find('=');
            if (eq == std::string::npos) continue;
            std::vector<Mapping> alts;
            size_t q = eq + 1;
            while (q < item.size()) {
                size_t b = item.find('|', q);
                if (b == std::string::npos) b = item.size();
                Mapping m;
                if (std::sscanf(item.substr(q, b - q).c_str(), "%u,%u,%u", &m.WG, &m.U, &m.J) == 3) alts.push_back(m);
                q = b + 1;
            }
            if (!alts.empty()) t.emplace_back(uint32_t(std::strtoul(item.c_str(), nullptr, 10)), alts);
        }
        return t;
    }();
    for (const auto &kv : table) {
        if (kv.first != k) continue;
        const Mapping &a = kv.second[cere_vk_dprof_step() % kv.second.size()];
        mp.WG = a.WG;
        mp.U = a.U;
        mp.J = a.J;
    }
    return mp;
}
#endif

}  // namespace

Mapping mapping_for(uint32_t k, uint32_t m) {
    // The order (G, E) from k only. Long k: the HIP kernel's shape (a wave32 per column, 16 elements per lane per
    // step). Short k (<= 1024: hyper-connection K 320, expert down K 640): 16 lanes per column, 8 elements (16 B) per
    // lane, so every lane loads at K 320 (P0-4). WG / U / J by speed per m (k2mx_test --tune, 2026-11-15, AMD):
    // m >= 2 shares each decoded activation across 4 (long) or 2 (short) columns, which keeps MXFP8 at m 4 .. 16 off
    // the VALU limit; m = 1 streams best with one column per group. Long k at m = 1 keeps 4 steps in flight (U 4: in
    // the decode step on AMD, VulkanAutotune stage 2 2026-11-22, hc_down 42.5 -> 29.6 us, out k6144 92.5 -> 75.6 us,
    // lm_head 5480 -> 5342 us; U does not change the order). Short k at m = 1: flat in the step, unchanged.
    if (k <= 1024) return m == 1 ? Mapping{16, 8, 256, 1, 1} : Mapping{16, 8, 128, 1, 2};
    return m == 1 ? Mapping{32, 16, 256, 4, 1} : Mapping{32, 16, 64, 1, 4};
}

std::string spv_name(bool mx, uint32_t mr, const Mapping &mp) {
    return std::string("k2_mgemv_") + (mx ? "mx" : "bf16") + "_g" + std::to_string(mp.G) + "_e" + std::to_string(mp.E) +
           "_wg" + std::to_string(mp.WG) + "_u" + std::to_string(mp.U) + "_j" + std::to_string(mp.J) + "_mr" + std::to_string(mr) + ".spv";
}

vk::PipelineSpec gemv_spec(bool mx, uint32_t m, const Mapping &mp) {
    return {spv_name(mx, row_capacity(m), mp), sizeof(GemvPush), 32, {}};
}

std::vector<vk::PipelineSpec> pipeline_specs(const std::vector<uint32_t> &bf16_ks, const std::vector<uint32_t> &mx_ks, bool dequant) {
    std::vector<vk::PipelineSpec> out;
    const auto add = [&](const vk::PipelineSpec &s) { append_unique(out, {s}); };
    for (int mx = 0; mx < 2; ++mx)
        for (uint32_t k : mx ? mx_ks : bf16_ks)
            for (uint32_t m = 1; m <= kGemvMaxM; ++m) add(gemv_spec(mx != 0, m, mapping_for(k, m)));
    if (dequant) add({"k2_mxfp8_dequant.spv", sizeof(DequantPush), 0, {}});
    return out;
}

cere_status gemv(vk::Context &ctx, uint64_t x, uint64_t w, uint64_t scales, uint64_t y, uint32_t m, uint32_t n,
                 uint32_t k, bool out_f32, const Mapping *force) {
    const bool mx = scales != 0;
    if (!x || !w || !y || m < 1 || m > kGemvMaxM || n < 1 || k < 1 || k % (mx ? 32 : 16) != 0 || !aligned16(x) ||
        !aligned16(w) || !aligned16(y) || uint64_t(n) * k * (mx ? 1 : 2) >= (uint64_t(1) << 32))
        return CERE_ERR_ARGUMENT;
    Mapping mp = force ? *force : mapping_for(k, m);
#if defined(CERE_VK_DISPATCH_PROFILE)
    if (!force && m == 1) mp = dprof_rotate(k, mp);
#endif
    const vk::PipelineSpec ps = gemv_spec(mx, m, mp);  // the one selection, shared with pipeline_specs
    const vk::Pipeline &p = ctx.pipeline(ps.spv, ps.push_bytes, ps.required_subgroup, ps.spec);
    const GemvPush pc{x, w, scales, y, m, n, k, out_f32 ? 1u : 0u};
    const uint32_t cols = mp.WG / mp.G * mp.J;
    ctx.dispatch(p, &pc, (n + cols - 1) / cols);
    return CERE_OK;
}

cere_status mxfp8_dequantize(vk::Context &ctx, uint64_t q, uint64_t scales, uint32_t n, uint32_t k, uint64_t w) {
    if (!q || !scales || !w || n == 0 || k == 0 || k % 32 != 0 || !aligned16(q) || !aligned16(w) ||
        uint64_t(n) * k >= (uint64_t(1) << 32))
        return CERE_ERR_ARGUMENT;
    const vk::Pipeline &p = ctx.pipeline("k2_mxfp8_dequant.spv", sizeof(DequantPush));
    const uint32_t groups = uint32_t(uint64_t(n) * k / 16);
    const DequantPush pc{q, scales, w, groups, k};
    ctx.dispatch(p, &pc, (groups + 255) / 256);
    return CERE_OK;
}

}  // namespace cere::vkk2
