// VulkanAmd P2: see k7_moe_vk.h.
#include "kernels/k7_moe_vk.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

#include "kernels/k2_gemm_vk.h"
#include "kernels/k2_mgemv_vk.h"

namespace cere::vkk7 {
namespace {

constexpr uint64_t kAlign = 256;
uint64_t align_up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

struct Workspace {
    uint64_t logits, ids, wts, x_codes, x_scales, gates, grp, perm, pos, h, h_codes, h_scales, y, s_gate, s_up, s_h,
        s_out, g_router, g_gate, g_up, g_down, bytes;
};

Workspace carve(const MoeDims &d, uint64_t m, uint64_t base) {
    Workspace w{};
    uint64_t off = 0;
    auto take = [&](uint64_t bytes) {
        const uint64_t p = base + off;
        off += align_up(bytes);
        return p;
    };
    const uint64_t r = m * d.top_k;
    w.logits = take(m * d.num_experts * 2);
    w.ids = take(r * 4);
    w.wts = take(r * 4);
    w.x_codes = take(m * d.hidden / 2);
    w.x_scales = take(m * d.hidden / 16);
    w.gates = take(m * 4);
    w.grp = take((2ull * d.num_experts + 2) * 4);
    w.perm = take(r * 4);
    w.pos = take(r * 4);
    w.h = take(r * d.inter * 2);
    w.h_codes = take(r * d.inter / 2);
    w.h_scales = take(r * d.inter / 16);
    w.y = take(r * d.hidden * 2);
    w.s_gate = take(m * d.shared_inter * 2);
    w.s_up = take(m * d.shared_inter * 2);
    w.s_h = take(m * d.shared_inter * 2);
    w.s_out = take(m * d.hidden * 2);
    // K2 GEMM workspaces: one per product recorded without a barrier before the next (k2_gemm_vk.h's sharing rule)
    w.g_router = take(vkk2::gemm_workspace_bytes(uint32_t(m), d.num_experts));
    w.g_gate = take(vkk2::gemm_workspace_bytes(uint32_t(m), d.shared_inter));
    w.g_up = take(vkk2::gemm_workspace_bytes(uint32_t(m), d.shared_inter));
    w.g_down = take(vkk2::gemm_workspace_bytes(uint32_t(m), d.hidden));
    w.bytes = off;
    return w;
}

constexpr uint32_t kMaxExperts = 1024;
constexpr uint32_t kMaxTopK = 32;

bool dims_ok(const MoeDims &d) {
    return d.hidden > 0 && d.hidden % 128 == 0 && d.inter > 0 && d.inter % 128 == 0 && d.shared_inter > 0 &&
           d.shared_inter % 16 == 0 && d.num_experts > 0 && d.num_experts <= kMaxExperts && d.top_k > 0 &&
           d.top_k <= kMaxTopK && d.top_k <= d.num_experts &&
           // every expert weight tensor inside one <= 4 GiB buffer (32-bit byte offsets in the shaders)
           uint64_t{d.num_experts} * d.inter * d.hidden / 2 < (uint64_t{1} << 32);
}

bool a16(uint64_t a) { return a != 0 && a % 16 == 0; }

struct GemmPush {
    uint64_t a_codes, a_scales, perm, w0, s0, w1, s1, alphas, grp, out, ids;
    uint32_t a_div, n, k, e, row_tiles, pad;
};
static_assert(sizeof(GemmPush) == 112, "k7_gemv.comp / k7_gemm.comp push layout");
struct RoutePush {
    uint64_t logits, ids, wts, grp, perm, pos;
    uint32_t m, e, k, pad;
};
static_assert(sizeof(RoutePush) == 64, "k7_route.comp push layout");
struct QuantPush {
    uint64_t x, codes, scales;
    float gs;
    uint32_t blocks;
};
static_assert(sizeof(QuantPush) == 32, "k7_quant.comp push layout");
struct GatePush {
    uint64_t x, g, gates;
    uint32_t m, h;
};
struct SiluPush {
    uint64_t gate, up, out;
    uint32_t count8, pad;
};
struct CombinePush {
    uint64_t gates, y, pos, wts, shared, out;
    uint32_t h, k;
};


uint32_t gemv_rows(uint32_t m) { return m <= 1 ? 1 : m <= 2 ? 2 : m <= 4 ? 4 : 8; }
uint32_t gemv_lanes(uint32_t k) { return k > 1024 ? 16 : 8; }  // VulkanAutotune stage 1 winner (was 8 / 4); speed only (k7_gemv.comp: the order is the block order)
// VulkanAutotune knobs (MoeOptions' CERE_VK_TUNE fields): compiled only into k7_test's own object (build.d/k7_moe.sh
// k7_moe_vk_tune.o); the library's object has the default lanes / tile only, so its bytes do not change.
#ifdef CERE_VK_TUNE
uint32_t gemv_lanes(uint32_t k, const MoeOptions &o) {
    const uint32_t g = k > 1024 ? o.gemv_lanes_long : o.gemv_lanes_short;
    return g != 0 ? g : gemv_lanes(k);
}
uint32_t gemm_bn(const MoeOptions &o) { return o.gemm_bn != 0 ? o.gemm_bn : 64; }
bool tune_ok(const MoeDims &d, const MoeOptions &o) {
    auto lanes = [](uint32_t g) { return g == 0 || g == 4 || g == 8 || g == 16; };
    return lanes(o.gemv_lanes_long) && lanes(o.gemv_lanes_short) &&
           (o.gemm_bn == 0 || ((o.gemm_bn == 64 || o.gemm_bn == 128) && d.inter % o.gemm_bn == 0 && d.hidden % o.gemm_bn == 0));
}
#endif
uint32_t ceil_div(uint64_t a, uint64_t b) { return static_cast<uint32_t>((a + b - 1) / b); }

// The route kernels' selection (modes 0: one thread per row, 1: fused route + group, 2 / 3: the split path's grouping
// passes; experts-per-lane variant for modes 0 / 1), and the GEMV's (glu, row capacity, lanes from k): one place for the
// dispatches and for pipeline_specs.
std::string route_spv(int mode, uint32_t num_experts) {
    std::string name = "k7_route_m" + std::to_string(mode);
    if (mode <= 1) name += num_experts <= 512 ? "_epl16" : "_epl32";
    return name + ".spv";
}
#ifdef CERE_VK_TUNE
std::string gemv_spv_g(bool glu, bool row_tiles, uint32_t m, uint32_t g) {
    return "k7_gemv_glu" + std::to_string(glu ? 1 : 0) + "_mr" + std::to_string(row_tiles ? 1 : gemv_rows(m)) + "_g" +
           std::to_string(g) + ".spv";
}
#endif
std::string gemv_spv(bool glu, bool row_tiles, uint32_t m, uint32_t k) {
    return "k7_gemv_glu" + std::to_string(glu ? 1 : 0) + "_mr" + std::to_string(row_tiles ? 1 : gemv_rows(m)) + "_g" +
           std::to_string(gemv_lanes(k)) + ".spv";
}
const vk::Pipeline &route_pipe(vk::Context &ctx, int mode, uint32_t num_experts) {
    return ctx.pipeline(route_spv(mode, num_experts), sizeof(RoutePush), 32);
}

// The four BF16 weights' K2 row info inside MoeLayer::bf16_info (16-byte aligned parts): router, shared gate, up, down.
struct Info { uint64_t router, gate, up, down, bytes; };
uint64_t round16(uint64_t b) { return (b + 15) / 16 * 16; }
Info info_parts(const MoeDims &d, uint64_t base) {
    Info i{};
    i.router = base;
    i.gate = i.router + round16(uint64_t{d.num_experts} * 4);
    i.up = i.gate + round16(uint64_t{d.shared_inter} * 4);
    i.down = i.up + round16(uint64_t{d.shared_inter} * 4);
    i.bytes = i.down + round16(uint64_t{d.hidden} * 4) - base;
    return i;
}

}  // namespace

std::vector<vk::PipelineSpec> pipeline_specs(const MoeDims &d, bool gemv, bool gemm, bool option_variants, bool mx) {
    std::vector<vk::PipelineSpec> s;
    if (!dims_ok(d)) return s;
    const uint32_t H = d.hidden, I = d.inter, S = d.shared_inter, K = d.top_k;
    s.push_back({"k7_quant.spv", sizeof(QuantPush), 0, {}});
    s.push_back({"k7_elem_gate.spv", sizeof(GatePush), 32, {}});
    s.push_back({"k7_elem_silu.spv", sizeof(SiluPush), 0, {}});
    s.push_back({"k7_elem_combine.spv", sizeof(CombinePush), 0, {}});
    // Routing: moe_route (mode 0) and the fused path (mode 1: pairs = m * K <= kFusedMaxPairs, always for m = 1) are always
    // possible; the split path's passes (modes 2 / 3) when pairs can exceed kFusedMaxPairs (coopmat rows > 16 / K, GemvAutoMaxM
    // * K > kFusedMaxPairs) or a caller forces GroupPath::kSplit.
    s.push_back({route_spv(0, d.num_experts), sizeof(RoutePush), 32, {}});
    s.push_back({route_spv(1, d.num_experts), sizeof(RoutePush), 32, {}});
    if (gemm || option_variants || (gemv && uint64_t{kGemvAutoMaxM} * K > kFusedMaxPairs)) {
        s.push_back({route_spv(2, d.num_experts), sizeof(RoutePush), 32, {}});
        s.push_back({route_spv(3, d.num_experts), sizeof(RoutePush), 32, {}});
    }
    // The BF16 K2 products (router / shared gate / up: k = H; shared down: k = S): the GEMV at row classes <= 16.
    // MXFP8 shared expert (mx): gate / up k = H, down k = S on the MXFP8 GEMV (S % 32 for the down projection).
    if (gemv) vkk2::append_unique(s, vkk2::pipeline_specs({H, S}, mx ? std::vector<uint32_t>{H, S} : std::vector<uint32_t>{}, false));
    // Routed experts, GEMV path (needs H and I <= kGemvMaxK): GEMM1 (glu, k = H) and GEMM2 (down, k = I). Auto tiles use
    // row tiles (mr 1); GemvTiles::kExpert (an option) uses the row capacity of m <= kGemvMaxExpertRows.
    if ((gemv || option_variants) && H <= kGemvMaxK && I <= kGemvMaxK) {
        for (int glu = 1; glu >= 0; --glu) {
            const uint32_t k = glu ? H : I;
            s.push_back({gemv_spv(glu != 0, true, 1, k), sizeof(GemmPush), 32, {}});
            if (option_variants)
                for (uint32_t m = 1; m <= kGemvMaxExpertRows; ++m) vkk2::append_unique(s, {{gemv_spv(glu != 0, false, m, k), sizeof(GemmPush), 32, {}}});
        }
    }
    // Coopmat path (rows > 16) and the K2 GEMM at row classes >= 17.
    if (gemm || option_variants) {
        s.push_back({"k7_gemm_glu1.spv", sizeof(GemmPush), 32, {}});
        s.push_back({"k7_gemm_glu0.spv", sizeof(GemmPush), 32, {}});
        s.push_back({"k7_probe.spv", 16, 32, {}});  // probe()'s push block: { idx, out }
        vkk2::append_unique(s, vkk2::pipeline_specs(true, mx));
    }
    return s;
}

cere_status probe(vk::Context &ctx) {
    vk::Buffer idx, out;
    cere_status st = ctx.allocate(256 * 4, idx);
    if (st == CERE_OK) st = ctx.allocate((32 * 16 + 32) * 4, out);
    if (st != CERE_OK) {
        ctx.free(idx);
        return st;
    }
    for (int i = 0; i < 256; ++i) static_cast<int32_t *>(idx.map)[i] = i;
    std::memset(out.map, 0xFF, (32 * 16 + 32) * 4);
    struct { uint64_t idx, out; } pc{idx.addr, out.addr};
    const vk::Pipeline *p = nullptr;
    st = ctx.pipeline(vk::PipelineSpec{"k7_probe.spv", sizeof(pc), 32, {}}, &p);
    if (st == CERE_OK) st = ctx.begin();
    if (st == CERE_OK) {
        ctx.dispatch(*p, &pc, 1);
        ctx.barrier();
        st = ctx.submit_wait();  // a latched recording error comes back here
    }
    bool ok = st == CERE_OK;
    const int32_t *o = static_cast<const int32_t *>(out.map);
    for (int l = 0; l < 32 && ok; ++l) {
        ok = ok && o[32 * 16 + l] == 8;
        for (int i = 0; i < 8 && ok; ++i) ok = o[l * 16 + i] == (2 * i + l / 16) * 16 + l % 16;
    }
    ctx.free(idx);
    ctx.free(out);
    if (st != CERE_OK) return st;  // not probed (launch failure): the state stays as it was
    ctx.set_probe(vk::Probe::kK7MoeLayout, ok ? vk::ProbeState::kPass : vk::ProbeState::kFail);
    return ok ? CERE_OK : CERE_ERR_UNSUPPORTED;
}

size_t moe_workspace_bytes(const MoeDims &dims, uint32_t max_m) {
    if (!dims_ok(dims) || max_m == 0) return 0;
    return carve(dims, max_m, 0).bytes;
}

size_t moe_weight_info_bytes(const MoeDims &dims) { return dims_ok(dims) ? size_t(info_parts(dims, 0).bytes) : 0; }

cere_status moe_prepare_weights(vk::Context &ctx, const MoeDims &d, const MoeLayer &L) {
    if (!dims_ok(d) || !a16(L.bf16_info) || !a16(L.router) || !a16(L.shared_gate) || !a16(L.shared_up) || !a16(L.shared_down) ||
        d.hidden % 64 != 0 || d.shared_inter % 64 != 0)
        return CERE_ERR_ARGUMENT;
    const Info i = info_parts(d, L.bf16_info);
    cere_status s;
    if ((s = vkk2::gemm_prepare_weight(ctx, L.router, 0, d.num_experts, d.hidden, i.router)) != CERE_OK) return s;
    if ((s = vkk2::gemm_prepare_weight(ctx, L.shared_gate, L.shared_gate_scales, d.shared_inter, d.hidden, i.gate)) != CERE_OK) return s;
    if ((s = vkk2::gemm_prepare_weight(ctx, L.shared_up, L.shared_up_scales, d.shared_inter, d.hidden, i.up)) != CERE_OK) return s;
    return vkk2::gemm_prepare_weight(ctx, L.shared_down, L.shared_down_scales, d.hidden, d.shared_inter, i.down);
}

cere_status moe_route(vk::Context &ctx, uint64_t logits, uint64_t topk_ids, uint64_t topk_weights, uint32_t m,
                      uint32_t num_experts, uint32_t top_k) {
    if (!logits || !topk_ids || !topk_weights || m == 0 || num_experts == 0 || num_experts > kMaxExperts ||
        top_k == 0 || top_k > kMaxTopK || top_k > num_experts || uint64_t{m} * num_experts * 2 >= (uint64_t{1} << 32))
        return CERE_ERR_ARGUMENT;
    const RoutePush pc{logits, topk_ids, topk_weights, 0, 0, 0, m, num_experts, top_k, 0};
    ctx.dispatch(route_pipe(ctx, 0, num_experts), &pc, ceil_div(m, 8));
    return CERE_OK;
}

cere_status moe_block(vk::Context &ctx, const MoeDims &d, const MoeLayer &L, uint64_t x, uint32_t m, uint64_t out,
                      uint64_t workspace, size_t workspace_bytes, const MoeOptions *opts) {
    const MoeOptions o = opts ? *opts : MoeOptions{};
    if (o.profile_stamp0 >= 0 && uint32_t(o.profile_stamp0) + kProfileStamps > vk::Context::kStamps) return CERE_ERR_ARGUMENT;
    if (!dims_ok(d) || m == 0 || !a16(x) || !a16(out) || workspace == 0 || workspace % kAlign != 0) return CERE_ERR_ARGUMENT;
    if (o.cls != 0 && o.cls < m) return CERE_ERR_ARGUMENT;
#ifdef CERE_VK_TUNE
    if (!tune_ok(d, o)) return CERE_ERR_ARGUMENT;
#endif
    if (L.bf16_info % 16 != 0) return CERE_ERR_ARGUMENT;
    if (L.shared_down_scales && d.shared_inter % 32 != 0) return CERE_ERR_ARGUMENT;  // MXFP8 k % 32 (K2's GEMV rule)
    for (uint64_t p : {L.router, L.gate_weight, L.gate_scale, L.up_weight, L.up_scale, L.down_weight, L.down_scale,
                       L.g1_alphas, L.g2_alphas, L.shared_gate, L.shared_up, L.shared_down, L.shared_expert_gate})
        if (!a16(p)) return CERE_ERR_ARGUMENT;
    const Workspace w = carve(d, m, workspace);
    if (w.bytes > workspace_bytes || w.bytes >= (uint64_t{1} << 32)) return CERE_ERR_ARGUMENT;
    const uint64_t pairs = uint64_t{m} * d.top_k;
    const uint32_t H = d.hidden, E = d.num_experts, K = d.top_k, I = d.inter, S = d.shared_inter;

    // Paths.
    const bool row_tiles = o.tiles != GemvTiles::kExpert;
    const bool gemv_fits = (row_tiles || m <= kGemvMaxExpertRows) && H <= kGemvMaxK && I <= kGemvMaxK;
    const bool use_gemv = o.gemm == GemmPath::kGemv || (o.gemm == GemmPath::kAuto && gemv_fits && m <= kGemvAutoMaxM);
    if (o.gemm == GemmPath::kGemv && !gemv_fits) return CERE_ERR_ARGUMENT;
    if (!use_gemv && (!ctx.info().coopmat || ctx.probe(vk::Probe::kK7MoeLayout) != vk::ProbeState::kPass))
        return CERE_ERR_UNSUPPORTED;
    const bool fused_fits = pairs <= kFusedMaxPairs;
    const bool fused = o.group == GroupPath::kFused || (o.group == GroupPath::kAuto && fused_fits);
    if (o.group == GroupPath::kFused && !fused_fits) return CERE_ERR_ARGUMENT;
    const uint32_t tiles = static_cast<uint32_t>(std::min<uint64_t>(E, pairs));
    // The BF16 K2 products: row class cls (0 = m) as HIP GemmPlan::run; >= 17 is the K2 GEMM (k % 64, refused before
    // anything is recorded otherwise).
    const uint32_t cls = o.cls;
    if ((cls != 0 ? cls : m) >= vkk2::kGemmMinM && (H % 64 != 0 || S % 64 != 0 || !vkk2::gemm_ready(ctx)))
        return CERE_ERR_UNSUPPORTED;  // the K2 GEMM's refusals (k % 64, unprobed context), before anything is recorded
    const Info inf = info_parts(d, L.bf16_info);
    auto k2 = [&](uint64_t a, uint64_t wt, uint64_t sc, uint64_t dst, uint32_t n, uint32_t k, uint64_t gws, uint64_t winfo) {
        return vkk2::run(ctx, a, wt, sc, dst, m, n, k, false, cls, gws, L.bf16_info ? winfo : 0);
    };

    uint32_t ps = o.profile_stamp0 >= 0 ? uint32_t(o.profile_stamp0) : 0;
    auto stamp = [&] { if (o.profile_stamp0 >= 0) ctx.stamp(ps++); };
    stamp();
    // Phase A: router logits, shared gate / up, q1(x), the sigmoid gates (all read x only).
    cere_status s;
    if ((s = k2(x, L.router, 0, w.logits, E, H, w.g_router, inf.router)) != CERE_OK) return s;  // BF16 always
    if ((s = k2(x, L.shared_gate, L.shared_gate_scales, w.s_gate, S, H, w.g_gate, inf.gate)) != CERE_OK) return s;
    if ((s = k2(x, L.shared_up, L.shared_up_scales, w.s_up, S, H, w.g_up, inf.up)) != CERE_OK) return s;
    {
        const QuantPush pc{x, w.x_codes, w.x_scales, L.a1_global_scale, static_cast<uint32_t>(uint64_t{m} * H / 16)};
        ctx.dispatch(ctx.pipeline("k7_quant.spv", sizeof(pc)), &pc, ceil_div(pc.blocks, 256));
        const GatePush gp{x, L.shared_expert_gate, w.gates, m, H};
        ctx.dispatch(ctx.pipeline("k7_elem_gate.spv", sizeof(gp), 32), &gp, ceil_div(m, 8));
    }
    ctx.barrier();
    stamp();
    // Phase B: router (+ grouping), shared SwiGLU.
    const RoutePush rp{w.logits, w.ids, w.wts, w.grp, w.perm, w.pos, m, E, K, 0};
    if (fused) {
        ctx.dispatch(route_pipe(ctx, 1, E), &rp, 1);
    } else {
        ctx.dispatch(route_pipe(ctx, 0, E), &rp, ceil_div(m, 8));
    }
    {
        const SiluPush sp{w.s_gate, w.s_up, w.s_h, static_cast<uint32_t>(uint64_t{m} * S / 8), 0};
        ctx.dispatch(ctx.pipeline("k7_elem_silu.spv", sizeof(sp)), &sp, ceil_div(sp.count8, 256));
    }
    ctx.barrier();
    stamp();
    if (!fused) {
        ctx.dispatch(route_pipe(ctx, 2, E), &rp, 1);
        ctx.barrier();
        ctx.dispatch(route_pipe(ctx, 3, E), &rp, E);
        ctx.barrier();
    }
    stamp();
    // Phase C: GEMM1 (gate + up, SwiGLU) -> h; the shared down projection.
    auto gemm = [&](bool glu, uint64_t a_codes, uint64_t a_scales, uint32_t a_div, uint64_t w0, uint64_t s0, uint64_t w1,
                    uint64_t s1, uint64_t alphas, uint32_t n, uint32_t k, uint64_t dst) {
        const GemmPush pc{a_codes, a_scales, w.perm, w0, s0, w1, s1, alphas, w.grp, dst, w.ids, a_div, n, k, E,
                          use_gemv && row_tiles ? 1u : 0u, 0};
        if (use_gemv) {
#ifdef CERE_VK_TUNE
            const uint32_t g = gemv_lanes(k, o), cols = 256 / g;
            const std::string name = gemv_spv_g(glu, row_tiles, m, g);
#else
            const uint32_t g = gemv_lanes(k), cols = 256 / g;
            const std::string name = gemv_spv(glu, row_tiles, m, k);
#endif
            ctx.dispatch(ctx.pipeline(name, sizeof(pc), 32), &pc, n / cols, row_tiles ? static_cast<uint32_t>(pairs) : tiles);
        } else {
#ifdef CERE_VK_TUNE
            ctx.dispatch(ctx.pipeline(glu ? "k7_gemm_glu1.spv" : "k7_gemm_glu0.spv", sizeof(pc), 32), &pc, n / gemm_bn(o), tiles);
#else
            ctx.dispatch(ctx.pipeline(glu ? "k7_gemm_glu1.spv" : "k7_gemm_glu0.spv", sizeof(pc), 32), &pc, n / 64, tiles);
#endif
        }
    };
    gemm(true, w.x_codes, w.x_scales, K, L.gate_weight, L.gate_scale, L.up_weight, L.up_scale, L.g1_alphas, I, H, w.h);
    if ((s = k2(w.s_h, L.shared_down, L.shared_down_scales, w.s_out, H, S, w.g_down, inf.down)) != CERE_OK) return s;
    ctx.barrier();
    stamp();
    // Phase D: q2(h) over the R sorted rows.
    {
        const QuantPush pc{w.h, w.h_codes, w.h_scales, L.a2_global_scale, static_cast<uint32_t>(pairs * I / 16)};
        ctx.dispatch(ctx.pipeline("k7_quant.spv", sizeof(pc)), &pc, ceil_div(pc.blocks, 256));
    }
    ctx.barrier();
    stamp();
    // Phase E: GEMM2 (down) -> y.
    gemm(false, w.h_codes, w.h_scales, 0, L.down_weight, L.down_scale, 0, 0, L.g2_alphas, H, I, w.y);
    ctx.barrier();
    stamp();
    // Phase F: combine.
    {
        const CombinePush pc{w.gates, w.y, w.pos, w.wts, w.s_out, out, H, K};
        ctx.dispatch(ctx.pipeline("k7_elem_combine.spv", sizeof(pc)), &pc, m, ceil_div(H, 2048));
    }
    if (o.profile_stamp0 >= 0) {
        ctx.barrier();
        stamp();
    }
    return CERE_OK;
}

}  // namespace cere::vkk7
