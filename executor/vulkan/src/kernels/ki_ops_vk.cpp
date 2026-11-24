// Inkling S11: see ki_ops_vk.h.
#include "kernels/ki_ops_vk.h"

#include <algorithm>
#include <cstring>
#include <string>

namespace cere::vki {
namespace {

struct NormPush {
    uint64_t x, w, o, tau;
    uint32_t n, rows, heads, flags;
    float eps, thr;
};
struct ConvPush {
    uint64_t x, w, dyn, o, st_in, st_out;
    uint32_t T, C, heads, flags;
};
struct QuantPush {
    uint64_t x, q, s, amax, c4;
    uint32_t K, rows, flags, pad;
};
struct RouterPush {
    uint64_t x, w, bias, ids, wts, lg;
    uint32_t m, E, H, pad;
    float scaling;
};
struct MoePush {
    uint64_t aq, as, aamax, w, ws, wamax, ids, wts, o, tab;
    uint32_t N, K, pairs, tiles;
    float den, pad2;
};
static_assert(sizeof(MoePush) == 104, "ki_moe.comp push layout");
struct MixPush {
    uint64_t a[8];
    uint32_t n0, n1, n2, n3;
    float f0, f1;
};
static_assert(sizeof(MixPush) == 88, "ki_mix.comp push layout");
struct AttnPush {
    uint64_t q, r, proj, sink, gate, o, k, v, kc, vc, slots;
    uint32_t kind, pos0, n, qs, rs, ss, gs, ks, vs, r0;
};
static_assert(sizeof(AttnPush) == 128, "ki_attn.comp push layout");
constexpr uint32_t kWgAttn = 64;
struct DensePush {
    uint64_t x, w, y;
    uint32_t m, n, k, flags;
};
static_assert(sizeof(DensePush) == 40, "ki_dense.comp push layout");
constexpr uint32_t kWgDense = 64, kDenseRows = 8;

constexpr uint32_t kWg = 256;
uint32_t bf(Dt d, uint32_t bit) { return d == Dt::kBF16 ? bit : 0u; }

// The refs' fixed reduce length of a row of width n (0: unsupported width).
uint32_t lanes_for(uint32_t n) { return n == 2304 ? 256u : n == 128 ? 32u : n == 16 ? 16u : 0u; }
std::string norm_spv(uint32_t mode, uint32_t lanes) {
    return "ki_rownorm_m" + std::to_string(mode) + "_l" + std::to_string(lanes) + ".spv";
}

cere_status norm(vk::Context &ctx, uint32_t mode, const NormPush &p) {
    const uint32_t lanes = lanes_for(p.n);
    if (lanes == 0) return CERE_ERR_UNSUPPORTED;
    if (p.rows == 0) return CERE_OK;
    const uint32_t per_wg = kWg / lanes;
    return ctx.dispatch(ctx.pipeline(norm_spv(mode, lanes), sizeof p), &p, (p.rows + per_wg - 1) / per_wg);
}

}  // namespace

std::vector<vk::PipelineSpec> pipeline_specs(bool coopmat) {
    std::vector<vk::PipelineSpec> s;
    for (uint32_t mode : {0u, 1u})
        for (uint32_t lanes : {256u, 32u, 16u}) s.push_back({norm_spv(mode, lanes), sizeof(NormPush), 0, {}});
    s.push_back({"ki_conv_m0.spv", sizeof(ConvPush), 0, {}});
    s.push_back({"ki_conv_m1.spv", sizeof(ConvPush), 0, {}});
    s.push_back({"ki_quant_mx.spv", sizeof(QuantPush), 0, {}});
    s.push_back({"ki_quant_nv.spv", sizeof(QuantPush), 0, {}});
    s.push_back({"ki_router_s0.spv", sizeof(RouterPush), 0, {}});
    s.push_back({"ki_router_s1.spv", sizeof(RouterPush), 0, {}});
    for (const char *n : {"ki_moe_f0_s0.spv", "ki_moe_f0_s1.spv", "ki_moe_f1_s0.spv", "ki_moe_f1_s1.spv", "ki_moe_f0_s3.spv",
                          "ki_moe_f0_s4.spv", "ki_moe_f1_s3.spv", "ki_moe_f1_s4.spv", "ki_moe_comb.spv", "ki_moe_bucket.spv"})
        s.push_back({n, sizeof(MoePush), 0, {}});
    if (coopmat) {
        s.push_back({"ki_moe_f0_s3c.spv", sizeof(MoePush), 32, {}});
        s.push_back({"ki_moe_f0_s4c.spv", sizeof(MoePush), 32, {}});
        s.push_back({"ki_moe_probe.spv", sizeof(MoePush), 32, {}});
    }
    for (int m = 0; m < 9; ++m) s.push_back({"ki_mix_m" + std::to_string(m) + ".spv", sizeof(MixPush), 0, {}});
    s.push_back({"ki_attn_m0.spv", sizeof(AttnPush), 0, {}});
    s.push_back({"ki_attn_m1.spv", sizeof(AttnPush), 0, {}});
    s.push_back({"ki_attn_m2.spv", sizeof(AttnPush), 0, {}});
    s.push_back({"ki_attn_m3.spv", sizeof(AttnPush), 0, {}});
    s.push_back({"ki_attn_m4.spv", sizeof(AttnPush), 0, {}});
    s.push_back({"ki_dense.spv", sizeof(DensePush), 0, {}});
    return s;
}

cere_status rms_norm(vk::Context &ctx, Tensor x, Tensor w, Tensor out, uint32_t n, uint32_t rows, float eps) {
    if (n != 2304 && n != 128) return CERE_ERR_UNSUPPORTED;
    const NormPush p{x.addr, w.addr, out.addr, 0, n, rows, 1, bf(x.dt, 1) | bf(out.dt, 2) | bf(w.dt, 4), eps, 0.f};
    return norm(ctx, 0, p);
}

cere_status l2_clip(vk::Context &ctx, Tensor x, Tensor scale, uint64_t tau, Tensor out, uint32_t n, uint32_t rows, uint32_t heads,
                    float thr) {
    if (n != 128 && n != 16) return CERE_ERR_UNSUPPORTED;
    if (heads == 0) return CERE_ERR_ARGUMENT;
    const NormPush p{x.addr, scale.addr, out.addr, tau, n, rows, heads, bf(x.dt, 1) | bf(out.dt, 2) | bf(scale.dt, 4), 0.f, thr};
    return norm(ctx, 1, p);
}

static cere_status conv(vk::Context &ctx, uint32_t mode, const ConvPush &p) {
    if (p.st_in == p.st_out) return CERE_ERR_ARGUMENT;
    if (p.C == 0) return CERE_OK;
    const uint32_t gy = p.T > 3 ? p.T : 3;  // rows 0..2 also write the next state
    return ctx.dispatch(ctx.pipeline(mode == 0 ? "ki_conv_m0.spv" : "ki_conv_m1.spv", sizeof p), &p, (p.C + kWg - 1) / kWg, gy);
}

cere_status conv_static(vk::Context &ctx, Tensor x, Tensor w, Tensor out, uint64_t st_in, uint64_t st_out, uint32_t T, uint32_t C) {
    if (x.dt != out.dt) return CERE_ERR_ARGUMENT;
    const ConvPush p{x.addr, w.addr, 0, out.addr, st_in, st_out, T, C, 1, bf(x.dt, 1) | bf(out.dt, 2) | bf(w.dt, 4)};
    return conv(ctx, 0, p);
}

cere_status conv_dynamic(vk::Context &ctx, Tensor x, Tensor static_w, Tensor dyn, Tensor out, uint64_t st_in, uint64_t st_out,
                         uint32_t T, uint32_t C, uint32_t heads) {
    if (x.dt != out.dt) return CERE_ERR_ARGUMENT;
    if (heads == 0 || C % heads != 0) return CERE_ERR_ARGUMENT;
    const ConvPush p{x.addr, static_w.addr, dyn.addr, out.addr, st_in, st_out, T, C, heads,
                     bf(x.dt, 1) | bf(out.dt, 2) | bf(static_w.dt, 4) | bf(dyn.dt, 8)};
    return conv(ctx, 1, p);
}

cere_status quant_mxfp8(vk::Context &ctx, Tensor x, uint64_t q, uint64_t s, uint32_t K, uint32_t rows) {
    if (K % 32 != 0 || K / 32 > kWg) return CERE_ERR_UNSUPPORTED;
    if (rows == 0) return CERE_OK;
    const QuantPush p{x.addr, q, s, 0, 0, K, rows, bf(x.dt, 1), 0};
    return ctx.dispatch(ctx.pipeline("ki_quant_mx.spv", sizeof p), &p, rows);
}

cere_status quant_nvfp4(vk::Context &ctx, Tensor x, uint64_t q, uint64_t s, uint64_t amax, uint64_t c4, uint32_t K, uint32_t rows,
                        Mutant mutant) {
    if (K % 16 != 0 || K / 16 > kWg) return CERE_ERR_UNSUPPORTED;
    if (rows == 0) return CERE_OK;
    const uint32_t flags = bf(x.dt, 1) | (mutant == Mutant::kGlobalActScale ? 2u : 0u);
    const QuantPush p{x.addr, q, s, amax, c4, K, rows, flags, 0};
    return ctx.dispatch(ctx.pipeline("ki_quant_nv.spv", sizeof p), &p, rows);
}

cere_status router(vk::Context &ctx, uint64_t x, uint64_t w, uint64_t bias, uint64_t lg, uint64_t ids, uint64_t wts, uint32_t m,
                   uint32_t E, uint32_t H, float scaling, Mutant mutant) {
    if (H % kWg != 0 || E < 8 || E > 128) return CERE_ERR_UNSUPPORTED;
    if (m == 0) return CERE_OK;
    const RouterPush p{x, w, bias, ids, wts, lg, m, E, H, mutant == Mutant::kSoftmaxRouting ? 1u : 0u, scaling};
    if (const cere_status st = ctx.dispatch(ctx.pipeline("ki_router_s0.spv", sizeof p), &p, (E + 3) / 4, m); st != CERE_OK)
        return st;
    if (const cere_status st = ctx.barrier(); st != CERE_OK) return st;
    return ctx.dispatch(ctx.pipeline("ki_router_s1.spv", sizeof p), &p, m);
}

static constexpr uint32_t kMoeWg = 64;
static constexpr uint32_t kMoeRows = 4;   // ki_moe.comp ROWS: weight rows per workgroup (stages 0 / 1)
static constexpr uint32_t kMoeGRows = 64;  // ki_moe.comp GROWS: weight rows per grouped tile (stages 3 / 4)
static constexpr uint32_t kMoeNbMax = 144;  // ki_moe.comp NBMAX: scale blocks per row it holds
static bool moe_shape_ok(uint32_t fmt, uint32_t K) {
    const uint32_t blk = fmt == 0 ? 16u : 32u;
    return fmt <= 1 && K > 0 && K % blk == 0 && K / blk <= kMoeNbMax;
}

static constexpr uint32_t kMoePt = 32;        // ki_moe.comp PT: pairs per grouped tile
static constexpr uint32_t kMoeExpertMax = 128;
static uint32_t moe_tiles(uint32_t pairs) { return (pairs + kMoePt - 1) / kMoePt + kMoeExpertMax; }

uint64_t moe_tab_bytes(uint32_t pairs) { return uint64_t(moe_tiles(pairs)) * 16 + uint64_t(pairs) * 4; }

cere_status moe_bucket(vk::Context &ctx, uint64_t ids, uint64_t tab, uint32_t pairs, uint32_t E) {
    if (E == 0 || E > kMoeExpertMax) return CERE_ERR_UNSUPPORTED;
    if (pairs == 0) return CERE_OK;
    const MoePush p{0, 0, 0, 0, 0, 0, ids, 0, 0, tab, E, 0, pairs, moe_tiles(pairs), 1.0f, 0.0f};
    return ctx.dispatch(ctx.pipeline("ki_moe_bucket.spv", sizeof p), &p, 1);
}

// tab == 0: one workgroup per (4 rows, pair) (stages 0 / 1); else per (64 rows, tile) of moe_bucket's tiles (3 / 4),
// NVFP4 on the cooperative-matrix tiles (suffix c, mutant m) when the route allows it.
static cere_status moe_product(vk::Context &ctx, uint32_t fmt, uint32_t stage, const MoePush &p, uint64_t tab, MoeRoute route,
                               Mutant mutant) {
    const bool cm = tab != 0 && fmt == 0 && route == MoeRoute::kAuto && moe_coopmat(ctx);
    const bool mut = mutant == Mutant::kMoeBlockOrder;
    if (mut && !cm) return CERE_ERR_UNSUPPORTED;
    const uint32_t rows = tab != 0 ? kMoeGRows : kMoeRows, gx = (p.N + rows - 1) / rows;
    std::string spv = std::string("ki_moe_f") + char('0' + fmt) + "_s" + char('0' + stage + (tab != 0 ? 3 : 0));
    if (cm) spv += mut ? "m" : "c";
    return ctx.dispatch(ctx.pipeline(spv + ".spv", sizeof p, cm ? 32 : 0), &p, gx, tab != 0 ? p.tiles : p.pairs);
}

cere_status moe_gate_up(vk::Context &ctx, uint32_t fmt, ActQ x, ExpertW w13, uint64_t ids, uint64_t wts, uint64_t v, uint32_t N,
                        uint32_t K, uint32_t pairs, uint64_t tab, MoeRoute route, Mutant mutant) {
    if (!moe_shape_ok(fmt, K) || N % 2 != 0) return CERE_ERR_UNSUPPORTED;
    if (pairs == 0) return CERE_OK;
    const MoePush p{x.q, x.s, x.amax, w13.w, w13.ws, w13.wamax, ids, wts, v, tab, N, K, pairs, moe_tiles(pairs), w13.den, 0.0f};
    return moe_product(ctx, fmt, 0, p, tab, route, mutant);
}

cere_status moe_down(vk::Context &ctx, uint32_t fmt, ActQ v, ExpertW w2, uint64_t ids, uint64_t y, uint32_t N, uint32_t K,
                     uint32_t pairs, uint64_t tab, MoeRoute route, Mutant mutant) {
    if (!moe_shape_ok(fmt, K)) return CERE_ERR_UNSUPPORTED;
    if (pairs == 0) return CERE_OK;
    const MoePush p{v.q, v.s, v.amax, w2.w, w2.ws, w2.wamax, ids, 0, y, tab, N, K, pairs, moe_tiles(pairs), w2.den, 0.0f};
    return moe_product(ctx, fmt, 1, p, tab, route, mutant);
}

bool moe_cm_device(const vk::Context &ctx) {
    const vk::DeviceInfo &i = ctx.info();
    return i.coopmat && i.subgroup_min <= 32 && i.subgroup_max >= 32;
}

bool moe_coopmat(const vk::Context &ctx) {
    return moe_cm_device(ctx) && ctx.probe(vk::Probe::kK7MoeLayout) == vk::ProbeState::kPass;
}

cere_status moe_probe(vk::Context &ctx) {
    if (!moe_cm_device(ctx)) return CERE_ERR_UNSUPPORTED;
    if (ctx.probe(vk::Probe::kK7MoeLayout) != vk::ProbeState::kUnprobed) return moe_coopmat(ctx) ? CERE_OK : CERE_ERR_UNSUPPORTED;
    vk::Buffer idx, out;
    cere_status st = ctx.allocate(256 * 4, idx);
    if (st == CERE_OK) st = ctx.allocate((32 * 16 + 32) * 4, out);
    if (st != CERE_OK) {
        ctx.free(idx);
        return st;
    }
    for (int i = 0; i < 256; ++i) static_cast<int32_t *>(idx.map)[i] = i;
    std::memset(out.map, 0xFF, (32 * 16 + 32) * 4);
    const MoePush p{idx.addr, 0, 0, 0, 0, 0, 0, 0, out.addr, 0, 0, 0, 0, 0, 1.0f, 0.0f};
    const vk::Pipeline *pl = nullptr;
    st = ctx.pipeline(vk::PipelineSpec{"ki_moe_probe.spv", sizeof p, 32, {}}, &pl);
    if (st == CERE_OK) st = ctx.begin();
    if (st == CERE_OK) {
        ctx.dispatch(*pl, &p, 1);
        ctx.barrier();
        st = ctx.submit_wait();  // a latched recording error comes back here
    }
    bool ok = st == CERE_OK;
    const int32_t *o = static_cast<const int32_t *>(out.map);
    for (int l = 0; l < 32 && ok; ++l) {
        ok = o[32 * 16 + l] == 8;
        for (int i = 0; i < 8 && ok; ++i) ok = o[l * 16 + i] == (2 * i + l / 16) * 16 + l % 16;
    }
    ctx.free(idx);
    ctx.free(out);
    if (st != CERE_OK) return st;  // not probed (launch failure): the state stays as it was
    ctx.set_probe(vk::Probe::kK7MoeLayout, ok ? vk::ProbeState::kPass : vk::ProbeState::kFail);
    return ok ? CERE_OK : CERE_ERR_UNSUPPORTED;
}

cere_status moe_combine(vk::Context &ctx, uint64_t y, uint64_t out, uint32_t N, uint32_t tokens) {
    if (tokens == 0) return CERE_OK;
    const MoePush p{y, 0, 0, 0, 0, 0, 0, 0, out, 0, N, 0, tokens * 8, 0, 1.0f, 0.0f};
    return ctx.dispatch(ctx.pipeline("ki_moe_comb.spv", sizeof p), &p, (N + kMoeWg - 1) / kMoeWg, tokens);
}

static cere_status mix(vk::Context &ctx, int mode, const MixPush &p, uint32_t gx, uint32_t gy = 1) {
    return ctx.dispatch(ctx.pipeline("ki_mix_m" + std::to_string(mode) + ".spv", sizeof p), &p, gx, gy);
}
// Elementwise grid: n / 256 groups folded into x <= 32768, y.
static cere_status mix_elem(vk::Context &ctx, int mode, const MixPush &p) {
    if (p.n0 == 0) return CERE_OK;
    const uint32_t groups = (p.n0 + kWg - 1) / kWg, gx = std::min<uint32_t>(groups, 32768);
    return mix(ctx, mode, p, gx, (groups + gx - 1) / gx);
}

cere_status depth_mix(vk::Context &ctx, const Tensor *states, uint32_t S, uint64_t queries, uint32_t nq, uint64_t out, uint32_t M,
                      float eps) {
    if (S == 0 || S > 5 || nq == 0) return CERE_ERR_ARGUMENT;
    if (M == 0) return CERE_OK;
    MixPush p{};
    for (uint32_t s = 0; s < S; ++s) {
        p.a[s] = states[s].addr;
        p.n3 |= bf(states[s].dt, 1u << s);
    }
    p.a[5] = queries;
    p.a[6] = out;
    p.n0 = S; p.n1 = nq; p.n2 = M; p.f0 = eps;
    return mix(ctx, 0, p, M);
}

cere_status msg_gate(vk::Context &ctx, uint64_t h, uint64_t msg, uint64_t down, uint64_t forget, uint64_t gain, uint64_t d,
                     uint64_t out, uint32_t M, float eps, MsgRoute route, Mutant mutant) {
    const bool mut = mutant == Mutant::kMsgGateOrder;
    if (mut && route != MsgRoute::kTiled) return CERE_ERR_UNSUPPORTED;
    const bool aligned = (forget | gain) % 16 == 0;
    if (route == MsgRoute::kTiled && !aligned) return CERE_ERR_ARGUMENT;
    if (M == 0) return CERE_OK;
    MixPush p{};
    p.a[0] = h; p.a[1] = msg; p.a[2] = down; p.a[3] = forget; p.a[4] = gain; p.a[5] = out; p.a[6] = d;
    p.n2 = M; p.f0 = eps;
    if (route == MsgRoute::kTiled || (route == MsgRoute::kAuto && aligned && M >= kMsgTiledRows)) {
        const uint32_t g7 = (M + kMsgTileRows7 - 1) / kMsgTileRows7;
        if (const cere_status st = ctx.dispatch(ctx.pipeline(mut ? "ki_mix_m7m.spv" : "ki_mix_m7.spv", sizeof p), &p, g7);
            st != CERE_OK)
            return st;
        if (const cere_status st = ctx.barrier(); st != CERE_OK) return st;
        return mix(ctx, 8, p, 2304 / 256, (M + kMsgTileRows8 - 1) / kMsgTileRows8);
    }
    if (const cere_status st = mix(ctx, 1, p, 64 / 4, M); st != CERE_OK) return st;  // d rows, 4 per workgroup
    if (const cere_status st = ctx.barrier(); st != CERE_OK) return st;
    return mix(ctx, 6, p, 2304 / 256, M);
}

cere_status softcap(vk::Context &ctx, Tensor x, uint64_t out, uint32_t n) {
    MixPush p{};
    p.a[0] = x.addr; p.a[1] = out; p.n0 = n; p.n3 = bf(x.dt, 1);
    return mix_elem(ctx, 2, p);
}

cere_status embed_norm(vk::Context &ctx, uint64_t table, uint64_t ids, uint64_t w, uint64_t out, uint32_t M, float eps) {
    if (M == 0) return CERE_OK;
    MixPush p{};
    p.a[0] = table; p.a[1] = ids; p.a[2] = w; p.a[3] = out; p.n2 = M; p.f0 = eps;
    return mix(ctx, 3, p, M);
}

cere_status attn_residual(vk::Context &ctx, uint64_t h, uint64_t o, uint64_t out, uint32_t n, bool boundary) {
    MixPush p{};
    p.a[0] = h; p.a[1] = o; p.a[2] = out; p.n0 = n; p.n3 = boundary ? 1u : 0u;
    return mix_elem(ctx, 4, p);
}

cere_status layer_out(vk::Context &ctx, uint64_t r, uint64_t d, uint64_t out, uint32_t n) {
    MixPush p{};
    p.a[0] = r; p.a[1] = d; p.a[2] = out; p.n0 = n;
    return mix_elem(ctx, 5, p);
}

static AttnPush attn_push(const AttnSeq &s) {
    AttnPush p{};
    p.k = s.k; p.v = s.v; p.kc = s.kc; p.vc = s.vc; p.slots = s.slots;
    p.kind = s.kind; p.pos0 = s.pos0; p.n = s.n; p.ks = s.ks; p.vs = s.vs;
    return p;
}

static bool attn_seq_ok(const AttnSeq &s) {
    return s.kind <= 1 && s.ks >= 1024 && s.vs >= 1024 && s.ks % 8 == 0 &&
           (s.k | s.kc) % 16 == 0 && uint64_t(s.pos0) + s.n <= (uint64_t(1) << 31);
}

cere_status attention(vk::Context &ctx, const AttnSeq &s, const AttnRows &q, uint32_t r0, uint32_t rows, AttnRoute route,
                      Mutant mutant) {
    const bool tiled_mutant = mutant == Mutant::kAttnKeyOrder;
    if (tiled_mutant && route != AttnRoute::kTiled) return CERE_ERR_UNSUPPORTED;
    if (!attn_seq_ok(s) || q.qs < 3072 || q.rs < 384 || q.ss < 24 || q.gs < 3072) return CERE_ERR_ARGUMENT;
    if (r0 >= s.n) return CERE_OK;
    rows = std::min(rows, s.n - r0);
    AttnPush p = attn_push(s);
    p.q = q.q; p.r = q.r; p.proj = q.proj; p.sink = q.sink; p.gate = q.gate; p.o = q.out;
    p.qs = q.qs; p.rs = q.rs; p.ss = q.ss; p.gs = q.gs; p.r0 = r0;
    // Split (MODE 2 + 3) when the span partials of the rows fit in q.part: zs = the most spans of any row. Not monotone
    // in the row: a sliding row at pos % 256 == 255 has 4 spans where its neighbours have 5, so the last row (a
    // 512-row prefill chunk's end) undercounts; the maximum is over the last row and the one before it.
    const auto spans = [&](uint32_t pos) {
        const uint32_t lo = s.kind == 0 && pos >= 1023 ? pos - 1023 : 0;
        return pos / 256 - lo / 256 + 1;
    };
    const uint32_t last = s.pos0 + r0 + rows - 1;
    const uint32_t zs = rows > 1 ? std::max(spans(last), spans(last - 1)) : spans(last);
    if (q.part == 0 || uint64_t(rows) * 24 * zs * kAttnPartBytes > q.part_bytes) {
        if (route == AttnRoute::kAuto) return ctx.dispatch(ctx.pipeline("ki_attn_m0.spv", sizeof p), &p, rows, 24);
        return CERE_ERR_ARGUMENT;
    }
    AttnPush p2 = p, p3 = p;  // the partials in o (MODE 2 / 4) / q (MODE 3), zs in n (ki_attn.comp)
    p2.o = q.part;
    p3.q = q.part;
    p2.n = p3.n = zs;
    cere_status st = CERE_OK;
    if (route == AttnRoute::kTiled || (route == AttnRoute::kAuto && rows >= kAttnTiledRows)) {
        // MODE 4: kAttnTileRows rows x one KV head per workgroup, spans counted from the block's first row (one more
        // than any row has: a block's rows may straddle a span boundary of lo); MODE 4 reads the call's row end in ss.
        p2.ss = r0 + rows;
        st = ctx.dispatch(ctx.pipeline(tiled_mutant ? "ki_attn_m4m.spv" : "ki_attn_m4.spv", sizeof p2), &p2,
                          (rows + kAttnTileRows - 1) / kAttnTileRows, 8, zs + 1);
    } else {
        st = ctx.dispatch(ctx.pipeline("ki_attn_m2.spv", sizeof p2), &p2, rows, 24, zs);
    }
    if (st != CERE_OK) return st;
    if (const cere_status st2 = ctx.barrier(); st2 != CERE_OK) return st2;
    return ctx.dispatch(ctx.pipeline("ki_attn_m3.spv", sizeof p3), &p3, rows, 24);
}

cere_status kv_store(vk::Context &ctx, const AttnSeq &s) {
    if (!attn_seq_ok(s)) return CERE_ERR_ARGUMENT;
    const uint32_t rows = s.kind == 0 && s.n > 1024 ? 1024 : s.n;
    if (rows == 0) return CERE_OK;
    const AttnPush p = attn_push(s);
    return ctx.dispatch(ctx.pipeline("ki_attn_m1.spv", sizeof p), &p, 1024 / kWgAttn, rows);
}

cere_status dense(vk::Context &ctx, uint64_t x, uint64_t w, uint64_t y, uint32_t m, uint32_t n, uint32_t k, bool out_f32) {
    if (k == 0 || k % 2 != 0 || (uint64_t(n) * k) >> 31 != 0 || (uint64_t(m) * k) >> 31 != 0) return CERE_ERR_UNSUPPORTED;
    if (m == 0 || n == 0) return CERE_OK;
    const DensePush p{x, w, y, m, n, k, out_f32 ? 1u : 0u};
    return ctx.dispatch(ctx.pipeline("ki_dense.spv", sizeof p), &p, (n + kWgDense - 1) / kWgDense,
                        (m + kDenseRows - 1) / kDenseRows);
}

}  // namespace cere::vki
