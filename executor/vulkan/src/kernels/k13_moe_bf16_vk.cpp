// VulkanAmd P2: see k13_moe_bf16_vk.h.
#include "kernels/k13_moe_bf16_vk.h"

#include <algorithm>
#include <string>

#include "kernels/k2_gemm_vk.h"
#include "kernels/k2_mgemv_vk.h"

namespace cere::vkk13 {
namespace {

constexpr uint64_t kAlign = 256;
uint64_t align_up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }
uint32_t ceil_div(uint64_t a, uint64_t b) { return static_cast<uint32_t>((a + b - 1) / b); }

struct Workspace {
    uint64_t logits, ids, wts, gates, grp, perm, pos, h, y, s_gate, s_up, s_h, s_out, g_router, g_gate, g_up, g_down, bytes;
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
    w.gates = take(m * 4);
    w.grp = take((2ull * d.num_experts + 2) * 4);
    w.perm = take(r * 4);
    w.pos = take(r * 4);
    w.h = take(r * d.inter * 2);
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

bool dims_ok(const MoeDims &d) {
    return d.hidden > 0 && d.hidden % 128 == 0 && d.inter > 0 && d.inter % 128 == 0 && d.shared_inter > 0 &&
           d.shared_inter % 16 == 0 && d.num_experts > 0 && d.num_experts <= 1024 && d.top_k > 0 && d.top_k <= 32 &&
           d.top_k <= d.num_experts &&
           // the BF16 gate_up tensor inside one <= 4 GiB buffer (32-bit byte offsets in k13_gemv.comp)
           uint64_t{d.num_experts} * 2 * d.inter * d.hidden * 2 < (uint64_t{1} << 32);
}

bool a16(uint64_t a) { return a != 0 && a % 16 == 0; }

// Push layouts of the K7 shaders reused here (k7_route.comp, k7_elem.comp; the same structs as k7_moe_vk.cpp).
struct RoutePush {
    uint64_t logits, ids, wts, grp, perm, pos;
    uint32_t m, e, k, pad;
};
static_assert(sizeof(RoutePush) == 64, "k7_route.comp push layout");
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
struct GemvPush {
    uint64_t a, perm, w, scale, glob, grp, out;
    uint32_t a_div, n, k, e;
};
static_assert(sizeof(GemvPush) == 72, "k13_gemv.comp push layout");

// One selection for the dispatches and pipeline_specs: the K7 route kernel (same naming as k7_moe_vk.cpp) and this file's GEMV.
std::string route_spv(int mode, uint32_t num_experts) {
    std::string name = "k7_route_m" + std::to_string(mode);
    if (mode <= 1) name += num_experts <= 512 ? "_epl16" : "_epl32";
    return name + ".spv";
}
const vk::Pipeline &route_pipe(vk::Context &ctx, int mode, uint32_t num_experts) {
    return ctx.pipeline(route_spv(mode, num_experts), sizeof(RoutePush), 32);
}

// The four BF16 weights' K2 row info inside MoeBf16Layer::bf16_info: router, shared gate, up, down (16-byte aligned
// parts; vkk7::moe_prepare_weights's layout).
struct Info { uint64_t router, gate, up, down; };
uint64_t round16(uint64_t b) { return (b + 15) / 16 * 16; }
Info info_parts(const MoeDims &d, uint64_t base) {
    Info i{};
    i.router = base;
    i.gate = i.router + round16(uint64_t{d.num_experts} * 4);
    i.up = i.gate + round16(uint64_t{d.shared_inter} * 4);
    i.down = i.up + round16(uint64_t{d.shared_inter} * 4);
    return i;
}

// Elements per lane per step of k13_gemv.comp, from k alone (part of the order: never from m or the format).
uint32_t lane_elems(uint32_t k) { return k >= 512 ? 16 : 8; }

std::string gemv_spv(bool glu, bool fp4, bool one_row, uint32_t k) {
    return std::string("k13_gemv_glu") + (glu ? "1" : "0") + "_fp4" + (fp4 ? "1" : "0") + "_mr" + (one_row ? "1" : "8") + "_le" +
           std::to_string(lane_elems(k)) + ".spv";
}

}  // namespace

std::vector<vk::PipelineSpec> pipeline_specs(const MoeDims &d, bool bf16_experts, bool nvfp4_experts, bool gemm) {
    std::vector<vk::PipelineSpec> s;
    if (!dims_ok(d)) return s;
    s.push_back({"k7_elem_gate.spv", sizeof(GatePush), 32, {}});
    s.push_back({"k7_elem_silu.spv", sizeof(SiluPush), 0, {}});
    s.push_back({"k7_elem_combine.spv", sizeof(CombinePush), 0, {}});
    for (int mode = 0; mode < 4; ++mode) s.push_back({route_spv(mode, d.num_experts), sizeof(RoutePush), 32, {}});
    for (int fp4 = 0; fp4 < 2; ++fp4) {
        if (!(fp4 ? nvfp4_experts : bf16_experts)) continue;
        for (int one_row = 0; one_row < 2; ++one_row) {   // tail == 1 (decode) / > 1 (verify, prefill)
            s.push_back({gemv_spv(true, fp4 != 0, one_row != 0, d.hidden), sizeof(GemvPush), 32, {}});
            s.push_back({gemv_spv(false, fp4 != 0, one_row != 0, d.inter), sizeof(GemvPush), 32, {}});
        }
    }
    // The router / shared products: K2 BF16 GEMV (k = hidden, shared_inter) at row classes <= 16, the GEMM above.
    vkk2::append_unique(s, vkk2::pipeline_specs({d.hidden, d.shared_inter}, {}, false));
    if (gemm) vkk2::append_unique(s, vkk2::pipeline_specs(true, false));
    return s;
}

size_t moe_bf16_workspace_bytes(const MoeDims &dims, uint32_t max_m) {
    if (!dims_ok(dims) || max_m == 0) return 0;
    return carve(dims, max_m, 0).bytes;
}

size_t moe_bf16_weight_info_bytes(const MoeDims &d) {
    if (!dims_ok(d)) return 0;
    const Info i = info_parts(d, 0);
    return size_t(i.down + round16(uint64_t{d.hidden} * 4));
}

cere_status moe_bf16_prepare_weights(vk::Context &ctx, const MoeDims &d, const MoeBf16Layer &L) {
    if (!dims_ok(d) || !a16(L.bf16_info) || !a16(L.router) || !a16(L.shared_gate) || !a16(L.shared_up) || !a16(L.shared_down) ||
        d.hidden % 64 != 0 || d.shared_inter % 64 != 0)
        return CERE_ERR_ARGUMENT;
    const Info i = info_parts(d, L.bf16_info);
    cere_status s;
    if ((s = vkk2::gemm_prepare_weight(ctx, L.router, 0, d.num_experts, d.hidden, i.router)) != CERE_OK) return s;
    if ((s = vkk2::gemm_prepare_weight(ctx, L.shared_gate, 0, d.shared_inter, d.hidden, i.gate)) != CERE_OK) return s;
    if ((s = vkk2::gemm_prepare_weight(ctx, L.shared_up, 0, d.shared_inter, d.hidden, i.up)) != CERE_OK) return s;
    return vkk2::gemm_prepare_weight(ctx, L.shared_down, 0, d.hidden, d.shared_inter, i.down);
}

cere_status moe_bf16_block(vk::Context &ctx, const MoeDims &d, const MoeBf16Layer &L, uint64_t x, uint32_t m,
                           uint32_t first, uint64_t out, uint64_t workspace, size_t workspace_bytes,
                           const MoeBf16Options *opts) {
    const MoeBf16Options o = opts ? *opts : MoeBf16Options{};
    if (o.profile_stamp0 >= 0 && uint32_t(o.profile_stamp0) + kProfileStamps > vk::Context::kStamps) return CERE_ERR_ARGUMENT;
    if (!dims_ok(d) || m == 0 || first >= m || !a16(x) || !a16(out) || workspace == 0 || workspace % kAlign != 0 ||
        (o.cls != 0 && o.cls < m) || L.bf16_info % 16 != 0)
        return CERE_ERR_ARGUMENT;
    const bool fp4 = L.gate_up_q != 0;
    for (uint64_t p : {L.router, L.shared_gate, L.shared_up, L.shared_down, L.shared_expert_gate, fp4 ? L.gate_up_q : L.gate_up,
                       fp4 ? L.down_q : L.down})
        if (!a16(p)) return CERE_ERR_ARGUMENT;
    if (fp4 && (L.gate_up || L.down || !L.gate_up_scale || !L.down_scale || !L.gate_up_global || !L.down_global ||
                L.gate_up_global % 4 || L.down_global % 4))
        return CERE_ERR_ARGUMENT;
    if (!fp4 && (L.gate_up_scale || L.gate_up_global || L.down_q || L.down_scale || L.down_global)) return CERE_ERR_ARGUMENT;
    const Workspace w = carve(d, m, workspace);
    if (w.bytes > workspace_bytes || w.bytes >= (uint64_t{1} << 32)) return CERE_ERR_ARGUMENT;
    const uint32_t H = d.hidden, E = d.num_experts, K = d.top_k, I = d.inter, S = d.shared_inter;
    // The router / shared K2 products on all m rows at row class cls (0 = m), HIP GemmPlan::run's routing; class >= 17
    // is the K2 GEMM (k % 64: refused before anything is recorded otherwise).
    if ((o.cls != 0 ? o.cls : m) >= vkk2::kGemmMinM && (H % 64 != 0 || S % 64 != 0 || !vkk2::gemm_ready(ctx)))
        return CERE_ERR_UNSUPPORTED;
    const Info inf = info_parts(d, L.bf16_info);
    auto k2 = [&](uint64_t a, uint64_t wt, uint64_t dst, uint32_t n, uint32_t k, uint64_t gws, uint64_t winfo) {
        return vkk2::run(ctx, a, wt, 0, dst, m, n, k, false, o.cls, gws, L.bf16_info ? winfo : 0);
    };
    const uint32_t tail = m - first;
    const uint64_t pairs = uint64_t{tail} * K;
    const uint64_t x_tail = x + uint64_t{first} * H * 2;
    const bool fused_fits = pairs <= vkk7::kFusedMaxPairs;
    if (o.group == vkk7::GroupPath::kFused && !fused_fits) return CERE_ERR_ARGUMENT;
    const bool fused = o.group == vkk7::GroupPath::kFused || (o.group == vkk7::GroupPath::kAuto && fused_fits);
    uint32_t ps = o.profile_stamp0 >= 0 ? uint32_t(o.profile_stamp0) : 0;
    auto stamp = [&] { if (o.profile_stamp0 >= 0) ctx.stamp(ps++); };
    cere_status s;

    stamp();
    // Phase A: router logits and shared gate / up on all m rows (K2 at class cls), the sigmoid gates of the tail rows.
    if ((s = k2(x, L.router, w.logits, E, H, w.g_router, inf.router)) != CERE_OK) return s;
    if ((s = k2(x, L.shared_gate, w.s_gate, S, H, w.g_gate, inf.gate)) != CERE_OK) return s;
    if ((s = k2(x, L.shared_up, w.s_up, S, H, w.g_up, inf.up)) != CERE_OK) return s;
    {
        const GatePush gp{x_tail, L.shared_expert_gate, w.gates, tail, H};
        ctx.dispatch(ctx.pipeline("k7_elem_gate.spv", sizeof(gp), 32), &gp, ceil_div(tail, 8));
    }
    ctx.barrier();
    stamp();
    // Phase B: route the tail rows' logits (+ grouping), shared SwiGLU on all m rows.
    const RoutePush rp{w.logits + uint64_t{first} * E * 2, w.ids, w.wts, w.grp, w.perm, w.pos, tail, E, K, 0};
    ctx.dispatch(route_pipe(ctx, fused ? 1 : 0, E), &rp, fused ? 1 : ceil_div(tail, 8));
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
    // Phase C: GEMM1 (gate + up, SwiGLU) -> h; the shared down projection on all m rows (K2 at class cls).
    const uint32_t tiles = static_cast<uint32_t>(std::min<uint64_t>(E, pairs));
    auto gemv = [&](bool glu, uint64_t a, uint32_t a_div, uint64_t wt, uint64_t sc, uint64_t gl, uint32_t n, uint32_t k,
                    uint64_t dst) {
        const GemvPush pc{a, w.perm, wt, sc, gl, w.grp, dst, a_div, n, k, E};
        const std::string name = gemv_spv(glu, fp4, tail == 1, k);
        const uint32_t cols = 8 * (glu ? 2 : 4);
        ctx.dispatch(ctx.pipeline(name, sizeof(pc), 32), &pc, ceil_div(n, cols), tiles);
    };
    gemv(true, x_tail, K, fp4 ? L.gate_up_q : L.gate_up, L.gate_up_scale, L.gate_up_global, I, H, w.h);
    if ((s = k2(w.s_h, L.shared_down, w.s_out, H, S, w.g_down, inf.down)) != CERE_OK) return s;
    ctx.barrier();
    stamp();
    // Phase D: GEMM2 (down) -> y.
    gemv(false, w.h, 0, fp4 ? L.down_q : L.down, L.down_scale, L.down_global, H, I, w.y);
    ctx.barrier();
    stamp();
    // Phase E: combine on the tail rows.
    {
        const CombinePush pc{w.gates, w.y, w.pos, w.wts, w.s_out + uint64_t{first} * H * 2, out + uint64_t{first} * H * 2, H, K};
        ctx.dispatch(ctx.pipeline("k7_elem_combine.spv", sizeof(pc)), &pc, tail, ceil_div(H, 2048));
    }
    if (o.profile_stamp0 >= 0) {
        ctx.barrier();
        stamp();
    }
    return CERE_OK;
}

}  // namespace cere::vkk13
