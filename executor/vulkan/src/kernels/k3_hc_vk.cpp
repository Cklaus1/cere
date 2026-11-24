// VulkanAmd P2: see k3_hc_vk.h.
#include "kernels/k3_hc_vk.h"

#include <algorithm>

#include "kernels/k1_primitives_vk.h"
#include "kernels/k2_gemm_vk.h"
#include "kernels/k2_mgemv_vk.h"

namespace cere::vkk3 {
namespace {

constexpr uint64_t kAlign = 256;
constexpr uint32_t kHc = 4;
constexpr uint32_t kChunk = vkk2::kGemvMaxM;

struct ActPush { uint64_t raw, t; uint32_t count, pad; };
struct GatePush { uint64_t u, normed, mixed; uint32_t m, hidden; };
struct CombinePush { uint64_t block_out, hyper, normed, w_inject, out; uint32_t hidden, span, m; };
static_assert(sizeof(ActPush) == 24 && sizeof(GatePush) == 32 && sizeof(CombinePush) == 56, "push layouts match k3_hc.comp");

uint64_t align_up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }
bool streams_ok(const HcShape &s) { return s.hc == kHc && s.hidden > 0 && s.hidden % 8 == 0 && s.hidden <= 8192; }
bool shape_ok(const HcShape &s) { return streams_ok(s) && s.lowrank > 0 && s.lowrank % 16 == 0; }
bool al16(uint64_t a) { return a % 16 == 0; }
uint32_t groups(uint64_t n, uint32_t per) { return uint32_t((n + per - 1) / per); }

}  // namespace

std::vector<vk::PipelineSpec> pipeline_specs(const HcShape &shape, bool gemm) {
    std::vector<vk::PipelineSpec> s = {{"k3_act.spv", sizeof(ActPush), 32, {}},
                                       {"k3_gate.spv", sizeof(GatePush), 32, {}},
                                       {"k3_combine.spv", sizeof(CombinePush), 32, {}}};
    // hc_mix / hc_combine_norm: K1's gemma rmsnorm over the per-stream width.
    vkk2::append_unique(s, {vkk1::rmsnorm_spec(false, int(shape.hidden))});
    // The down (k = hc * hidden) and up (k = lowrank) products are K2 BF16 at every m class <= 16 (a rows plan: always).
    vkk2::append_unique(s, vkk2::pipeline_specs({kHc * shape.hidden, shape.lowrank}, {}, false));
    if (gemm) vkk2::append_unique(s, vkk2::pipeline_specs(true, false));  // row classes >= 17
    return s;
}

cere_status HcPlans::init(const HcShape &s, uint32_t m) {
    if (!shape_ok(s)) return CERE_ERR_UNSUPPORTED;
    if (m == 0 || m > kMaxRows) return CERE_ERR_ARGUMENT;
    shape = s;
    max_m = m;
    rows = false;
    return CERE_OK;
}

cere_status HcPlans::init_rows(const HcPlans &inst, uint32_t r) {
    if (!shape_ok(inst.shape) || inst.max_m == 0) return CERE_ERR_ARGUMENT;
    if (r == 0 || r > kMaxRows) return CERE_ERR_ARGUMENT;
    shape = inst.shape;
    max_m = r;
    rows = true;
    return CERE_OK;
}

size_t hc_mix_workspace_bytes(const HcShape &s, uint32_t max_m) {
    return size_t(align_up(uint64_t(max_m) * s.lowrank * 4) + align_up(uint64_t(max_m) * s.lowrank * 2) +
                  align_up(uint64_t(max_m) * s.hc * s.hidden * 4) +
                  align_up(vkk2::gemm_workspace_bytes(max_m, std::max(s.lowrank, s.hc * s.hidden))));
}

bool hc_mix_fuses(const HcPlans &, uint32_t) { return false; }
const char *hc_mix_mode() { return "unfused: rmsnorm + K2(down) + act + K2(up) + gate; K2 = GEMV for row classes <= 16, GEMM above"; }

static cere_status stream_norm(vk::Context &ctx, const HcShape &s, uint64_t hyper, uint64_t w, uint32_t m, uint64_t normed) {
    return vkk1::rmsnorm(ctx, hyper, s.hidden, w, int(kHc), true, s.eps, int64_t(m) * kHc, int(s.hidden), normed, s.hidden);
}

cere_status hc_mix(vk::Context &ctx, const HcPlans &plans, uint64_t hyper, uint64_t norm_weight, uint64_t w_down, uint64_t w_up,
                   uint32_t m, uint64_t normed, uint64_t mixed, uint64_t workspace, HcMixPath path, uint32_t cls) {
    if (!hyper || !norm_weight || !normed || !mixed || !workspace || !shape_ok(plans.shape) || m == 0 || m > plans.max_m ||
        !al16(hyper) || !al16(norm_weight) || !al16(normed) || workspace % kAlign != 0)
        return CERE_ERR_ARGUMENT;
    if (path == HcMixPath::kEpilogue || path == HcMixPath::kEarly) return CERE_ERR_UNSUPPORTED;
    if (cls != 0 && cls < m) return CERE_ERR_ARGUMENT;
    if (!plans.rows && (cls != 0 ? cls : m) >= vkk2::kGemmMinM &&
        ((kHc * plans.shape.hidden) % 64 != 0 || plans.shape.lowrank % 64 != 0 || !vkk2::gemm_ready(ctx)))
        return CERE_ERR_UNSUPPORTED;  // hc_mix_normed's K2 GEMM refusals, before the norm is recorded
    const cere_status st = stream_norm(ctx, plans.shape, hyper, norm_weight, m, normed);
    if (st != CERE_OK) return st;
    ctx.barrier();
    return hc_mix_normed(ctx, plans, normed, w_down, w_up, m, mixed, workspace, path, cls);
}

cere_status hc_mix_normed(vk::Context &ctx, const HcPlans &plans, uint64_t normed, uint64_t w_down, uint64_t w_up, uint32_t m,
                          uint64_t mixed, uint64_t workspace, HcMixPath path, uint32_t cls) {
    const HcShape &s = plans.shape;
    if (!shape_ok(s) || m == 0 || m > plans.max_m || !normed || !mixed || !w_down || !w_up || !workspace || !al16(normed) ||
        !al16(mixed) || !al16(w_down) || !al16(w_up) || workspace % kAlign != 0)
        return CERE_ERR_ARGUMENT;
    if (path == HcMixPath::kEpilogue || path == HcMixPath::kEarly) return CERE_ERR_UNSUPPORTED;
    if (cls != 0 && cls < m) return CERE_ERR_ARGUMENT;
    const uint32_t W = kHc * s.hidden, LR = s.lowrank;
    // Row class of both products (k3_hc.h cls, 0 = m). A rows plan is HIP's K2 rows plan: the GEMV, whatever cls.
    const uint32_t choice = cls != 0 ? cls : m;
    const bool gemm = !plans.rows && choice >= vkk2::kGemmMinM;
    if (gemm && (W % 64 != 0 || LR % 64 != 0)) return CERE_ERR_UNSUPPORTED;  // K2 GEMM: k % 64 (refused before recording)
    if (gemm && !vkk2::gemm_ready(ctx)) return CERE_ERR_UNSUPPORTED;  // no coopmat or an unprobed context
    const uint64_t t_raw = workspace;
    const uint64_t t = workspace + align_up(uint64_t(plans.max_m) * LR * 4);
    const uint64_t u = t + align_up(uint64_t(plans.max_m) * LR * 2);
    const uint64_t gws = u + align_up(uint64_t(plans.max_m) * W * 4);  // K2 GEMM workspace (down, then up: barrier between)
    // y[0, m) = x[0, m) w^T (FP32 out): one K2 call at the class, or (rows plan) GEMV calls of <= 16 rows.
    auto product = [&](uint64_t x, uint32_t kx, uint64_t w, uint64_t y, uint32_t n) -> cere_status {
        if (!plans.rows) return vkk2::run(ctx, x, w, 0, y, m, n, kx, true, cls, gws);
        for (uint32_t off = 0; off < m; off += kChunk) {  // independent chunks: no barrier between
            const uint32_t r = std::min(kChunk, m - off);
            const cere_status st = vkk2::gemv(ctx, x + uint64_t(off) * kx * 2, w, 0, y + uint64_t(off) * n * 4, r, n, kx, true);
            if (st != CERE_OK) return st;
        }
        return CERE_OK;
    };
    if (const cere_status st = product(normed, W, w_down, t_raw, LR); st != CERE_OK) return st;
    ctx.barrier();
    const ActPush ap{t_raw, t, m * LR, 0};
    ctx.dispatch(ctx.pipeline("k3_act.spv", sizeof ap, 32), &ap, groups(uint64_t(m) * LR / 2, 256));
    ctx.barrier();
    if (const cere_status st = product(t, LR, w_up, u, W); st != CERE_OK) return st;
    ctx.barrier();
    const GatePush gp{u, normed, mixed, m, s.hidden};
    ctx.dispatch(ctx.pipeline("k3_gate.spv", sizeof gp, 32), &gp, groups(uint64_t(m) * s.hidden / 2, 256));
    return CERE_OK;
}

cere_status hc_combine(vk::Context &ctx, const HcShape &s, uint64_t block_out, uint64_t hyper, uint64_t normed, uint64_t w_inject,
                       uint32_t m, uint64_t out) {
    if (!streams_ok(s)) return CERE_ERR_UNSUPPORTED;
    if (m == 0 || m > kMaxRows || !block_out || !hyper || !normed || !w_inject || !out || !al16(block_out) || !al16(hyper) ||
        !al16(normed) || !al16(w_inject) || !al16(out))
        return CERE_ERR_ARGUMENT;
    const uint32_t vecs = kHc * s.hidden / 8;
    // m <= 16: 256 chunks per workgroup so the row spreads over many workgroups; larger m: one workgroup per row.
    const uint32_t span = m <= kChunk ? 256 : vecs;
    const CombinePush pc{block_out, hyper, normed, w_inject, out, s.hidden, span, m};
    ctx.dispatch(ctx.pipeline("k3_combine.spv", sizeof pc, 32), &pc, m, groups(vecs, span));
    return CERE_OK;
}

cere_status hc_combine_norm(vk::Context &ctx, const HcShape &s, uint64_t block_out, uint64_t hyper, uint64_t normed,
                            uint64_t w_inject, uint64_t next_norm_weight, uint32_t m, uint64_t out) {
    if (!streams_ok(s)) return CERE_ERR_UNSUPPORTED;
    if (!next_norm_weight || !al16(next_norm_weight)) return CERE_ERR_ARGUMENT;
    const cere_status st = hc_combine(ctx, s, block_out, hyper, normed, w_inject, m, out);
    if (st != CERE_OK) return st;
    ctx.barrier();
    return stream_norm(ctx, s, out, next_norm_weight, m, normed);
}

}  // namespace cere::vkk3
