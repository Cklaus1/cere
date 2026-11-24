// VulkanAmd P2: executor/hip/ktest/k7_test.cpp's contract rows on the Vulkan K7 NVFP4 MoE block (k7_moe_vk.h), same
// shapes (H 256, E 16, top-4, I 128, S 128), seed, data generation, host mirror and bounds:
//  1. moe_route vs the host (FP32 softmax over all experts, k-th largest with ties to the lowest id, renormalized):
//     ids equal, weights within 2.4e-7 relative;
//  2. moe_block vs the host mirror of the header's arithmetic (K0's quantizer with exact reciprocals, the W4A4 block
//     dots, alphas, BF16 rounding of gate / up / SwiGLU / down, the slot-order combine, the sigmoid-gated BF16 shared
//     expert): out rel-L2 <= 1e-2 at m 7 and 40; repeat bit-identical;
//  3. "GEMV == GEMM kernel": the m 7 call with cls 40 (routed experts on the GEMV, k7_gemv.comp, default at m <= 16) ==
//     rows 0..6 of the m 40 call (the cooperative-matrix GEMM, k7_gemm.comp), bit for bit;
//  4. "WMMA == GEMV kernel": rows 0, 8, 31 of the m 40 call == each row alone (m 1, GEMV, cls 40), bit for bit;
//  5. batch invariance: each row of a 5-row call == that row alone, bit for bit (both class <= 16: the K2 GEMV, the
//     HIP test's rows plans).
// As in the HIP test (executor/hip/ktest/k7_test.cpp:153-175), `cls 40` (MoeOptions::cls, k7_moe.h MoePlans::cls) pins
// the router's and shared expert's K2 products to the 40-row class (the K2 GEMM, k2_gemm_vk.h), so only the routed
// GEMM's kernel differs between the compared calls; without it the m 7 / m 1 calls would take the K2 GEMV, which is
// not bit-equal to the GEMM (on HIP neither).
// Added rows (not in the HIP test): the cooperative-matrix layout probes the GEMM paths depend on (K7's, K2's); forced
// kernel paths (coopmat at m 7, split grouping at m 7 / 40) bit-equal to the automatic ones; prepared K2 weight info
// (moe_prepare_weights) == the weight scanned in the call; cls < m refused; and, at the model's shape (H 2,560, E 512,
// top-10, I 640, S 640), an m 4,096 prefill call (split grouping + coopmat GEMM + K2 GEMM) whose rows equal the m 1
// calls of the same class (cls 4,096) and the host mirror, an m 16 verify call whose rows equal the m 1 calls (class
// <= 16: the K2 GEMV both). Then timing (GPU timestamps, median) at m 1, 5, 8, 16, 4,096.
// MXFP8 shared expert (the S3 artifact's format: executor/cuda/src/model/mxfp8_plan.h kShared; the router stays BF16):
// the test's shared gate / up / down MXFP8-quantized (k2w_test's rule); the block with MXFP8 shared weights == the block
// with their exact BF16 upcast, bit for bit, at m 1, 7 (K2 GEMV), m 7 cls 40 and m 40 (K2 GEMM), and at the model shape
// m 1 / 16 / 4,096; batch invariance and prepared info on MXFP8; MXFP8 shared_down with S % 32 != 0 refused. MXFP8
// timing at m 1 and 4,096 next to BF16.
// Usage: k7_test [spv_dir] [--no-timing] [--no-model] [--explore (kernel-path timing at m 1 .. 64)]
//                [--tune gemv_lanes_long=G | gemv_lanes_short=G | gemm_bn=BN] [--identity DIR] [--samples]
// (VulkanAutotune hooks, ktest_util.h: --tune sets MoeOptions' tune knobs on every call; --identity writes the m 7 / 40
// block outputs and the model shape's m 1 / 5 / 16 / 4,096 outputs; --samples prints every timed shape's samples, the
// model-shape timing then also at m 256, >= 15 samples each, rot = cold, hot = warm.)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "launch/vk_launch_testing.h"
#include "kernels/k2_gemm_vk.h"
#include "kernels/k7_moe_vk.h"
#include "ktest_util.h"

namespace kv = cere::vkk7;
using cere::vk::Buffer;
using cere::vk::Context;
using kt::bf2f;
using kt::f2bf;
static int g_fail = 0;
static void check(const char *name, bool ok, const char *msg) {
    std::printf("%-22s %s  %s\n", name, ok ? "PASS" : "FAIL", msg);
    std::fflush(stdout);
    if (!ok) ++g_fail;
}
static float rb(double x) { return bf2f(f2bf(float(x))); }
static float e4m3f(uint8_t c) { const int s = c >> 7, e = (c >> 3) & 15, m = c & 7; const float v = e == 0 ? std::ldexp(m / 8.f, -6) : std::ldexp(1.f + m / 8.f, e - 7); return s ? -v : v; }
static uint8_t to_e4m3(float x) {  // satfinite, round to nearest even (the HIP test's host reference)
    if (!(x == x)) return 0x7f;
    uint8_t best = 0; float bd = 1e30f; const float ax = std::min(std::fabs(x), 448.f);
    for (int c = 0; c < 0x7f; ++c) { const float d = std::fabs(e4m3f(uint8_t(c)) - ax); if (d < bd || (d == bd && (c & 1) == 0)) { bd = d; best = uint8_t(c); } }
    return uint8_t(best | (x < 0 ? 0x80 : 0));
}
static const float kE2[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
static int e2m1(float x) {
    const float a = std::fabs(x); int c;
    if (a <= .25f) c = 0; else if (a < .75f) c = 1; else if (a <= 1.25f) c = 2; else if (a < 1.75f) c = 3; else if (a <= 2.5f) c = 4; else if (a < 3.5f) c = 5; else if (a <= 5.f) c = 6; else c = 7;
    return c | (x < 0 ? 8 : 0);
}
static float e2f(int c) { return (c & 8) ? -kE2[c & 7] : kE2[c & 7]; }
static void q_row(const float *x, uint32_t n, float gs, std::vector<int> &codes, std::vector<uint8_t> &sfs) {
    codes.resize(n); sfs.resize(n / 16);
    for (uint32_t b = 0; b < n / 16; ++b) {
        float vmax = 0; for (int i = 0; i < 16; ++i) vmax = std::max(vmax, std::fabs(x[b * 16 + i]));
        const uint8_t sf = to_e4m3(gs * (vmax * (1.f / 6.f))); sfs[b] = sf;
        const float os = vmax != 0 ? 1.f / (e4m3f(sf) * (1.f / gs)) : 0.f;
        for (int i = 0; i < 16; ++i) codes[b * 16 + i] = e2m1(x[b * 16 + i] * os);
    }
}
static double w4a4(const std::vector<int> &ac, const std::vector<uint8_t> &as, const uint8_t *wc, const uint8_t *ws, uint32_t n) {
    double acc = 0;
    for (uint32_t b = 0; b < n / 16; ++b) {
        double d = 0; for (int i = 0; i < 16; ++i) { const uint32_t j = b * 16 + i; const int c = (wc[j / 2] >> (4 * (j % 2))) & 15; d += double(e2f(ac[j])) * e2f(c); }
        acc += d * e4m3f(as[b]) * e4m3f(ws[b]);
    }
    return acc;
}

// One layer: host copies (for the mirror), device buffers, the MoeLayer of their addresses.
struct Model {
    kv::MoeDims d;
    std::vector<uint16_t> router, sg, su, sd, seg;
    std::vector<uint8_t> gw, uw, dw, gs, us, ds;
    std::vector<float> g1, g2;
    float a1 = 2688.f / 4.f, a2 = 2688.f / 2.f;
    std::vector<Buffer> bufs;
    kv::MoeLayer L;
    template <class T> uint64_t up(Context &ctx, const std::vector<T> &h) { bufs.push_back(kt::upload(ctx, h)); return bufs.back().addr; }
    void upload(Context &ctx) {
        L.router = up(ctx, router); L.gate_weight = up(ctx, gw); L.gate_scale = up(ctx, gs); L.up_weight = up(ctx, uw); L.up_scale = up(ctx, us);
        L.down_weight = up(ctx, dw); L.down_scale = up(ctx, ds); L.g1_alphas = up(ctx, g1); L.g2_alphas = up(ctx, g2);
        L.a1_global_scale = a1; L.a2_global_scale = a2;
        L.shared_gate = up(ctx, sg); L.shared_up = up(ctx, su); L.shared_down = up(ctx, sd); L.shared_expert_gate = up(ctx, seg);
    }
    void release(Context &ctx) { for (auto &b : bufs) ctx.free(b); bufs.clear(); }
    // The HIP test's host block for one BF16 row xr16 [H] -> out[H] (double before the final rounding).
    std::vector<double> mirror(const uint16_t *xr16) const {
        const uint32_t H = d.hidden, E = d.num_experts, K = d.top_k, I = d.inter, S = d.shared_inter;
        std::vector<double> out(H);
        std::vector<float> xr(H); for (uint32_t i = 0; i < H; ++i) xr[i] = bf2f(xr16[i]);
        std::vector<float> lg(E); for (uint32_t e = 0; e < E; ++e) { double s = 0; for (uint32_t i = 0; i < H; ++i) s += double(xr[i]) * bf2f(router[size_t(e) * H + i]); lg[e] = rb(s); }
        float mx = *std::max_element(lg.begin(), lg.end()); std::vector<float> p(E); float sum = 0; for (uint32_t e = 0; e < E; ++e) { p[e] = std::exp(lg[e] - mx); sum += p[e]; }
        std::vector<uint32_t> idx(E); for (uint32_t e = 0; e < E; ++e) idx[e] = e;
        std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return lg[a] != lg[b] ? lg[a] > lg[b] : a < b; });
        float ks = 0; for (uint32_t k = 0; k < K; ++k) ks += p[idx[k]] / sum;
        std::vector<int> ac; std::vector<uint8_t> as; q_row(xr.data(), H, a1, ac, as);
        std::vector<double> f(H, 0.0);
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t e = idx[k]; const float wk = (p[e] / sum) / ks;
            std::vector<float> hmid(I);
            for (uint32_t j = 0; j < I; ++j) {
                const size_t row = size_t(e) * I + j;
                const float gt = rb(w4a4(ac, as, &gw[row * H / 2], &gs[row * H / 16], H) * g1[e]);
                const float upv = rb(w4a4(ac, as, &uw[row * H / 2], &us[row * H / 16], H) * g1[e]);
                hmid[j] = rb(double(gt) / (1 + std::exp(-double(gt))) * upv);
            }
            std::vector<int> hc; std::vector<uint8_t> hs; q_row(hmid.data(), I, a2, hc, hs);
            for (uint32_t o = 0; o < H; ++o) { const size_t row = size_t(e) * H + o; f[o] += double(wk) * rb(w4a4(hc, hs, &dw[row * I / 2], &ds[row * I / 16], I) * g2[e]); }
        }
        std::vector<float> sh(S);
        for (uint32_t j = 0; j < S; ++j) { double a = 0, b = 0; for (uint32_t i = 0; i < H; ++i) { a += double(xr[i]) * bf2f(sg[size_t(j) * H + i]); b += double(xr[i]) * bf2f(su[size_t(j) * H + i]); }
            const float ga = rb(a), ub = rb(b); sh[j] = rb(double(ga) / (1 + std::exp(-double(ga))) * ub); }
        double gd = 0; for (uint32_t i = 0; i < H; ++i) gd += double(xr[i]) * bf2f(seg[i]);
        const double sig = 1 / (1 + std::exp(-gd));
        for (uint32_t o = 0; o < H; ++o) { double dd = 0; for (uint32_t j = 0; j < S; ++j) dd += double(sh[j]) * bf2f(sd[size_t(o) * S + j]); out[o] = double(rb(f[o])) + sig * rb(dd); }
        return out;
    }
};

// The HIP test's generators, in its order (rnd: BF16 normal * s; codes: random bytes; scales: E4M3 0x30 .. 0x47).
struct Gen {
    std::mt19937 g; std::normal_distribution<float> nd;
    explicit Gen(uint32_t seed) : g(seed) {}
    std::vector<uint16_t> rnd(size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; }
    std::vector<uint8_t> codes(size_t n) { std::vector<uint8_t> v(n); for (auto &b : v) b = uint8_t(g()); return v; }
    std::vector<uint8_t> scales(size_t n) { std::vector<uint8_t> v(n); for (auto &b : v) b = uint8_t(0x30 + g() % 0x18); return v; }
};
static void make_model(Model &M, Gen &G, const kv::MoeDims &d, bool model_shape) {
    const uint32_t H = d.hidden, E = d.num_experts, I = d.inter, S = d.shared_inter;
    M.d = d;
    M.router = G.rnd(size_t(E) * H, 0.1f); M.sg = G.rnd(size_t(S) * H, 1.f / 16); M.su = G.rnd(size_t(S) * H, 1.f / 16);
    M.sd = G.rnd(size_t(H) * S, 1.f / 11); M.seg = G.rnd(H, 0.06f);
    M.gw = G.codes(size_t(E) * I * H / 2); M.uw = G.codes(size_t(E) * I * H / 2); M.dw = G.codes(size_t(E) * H * I / 2);
    M.gs = G.scales(size_t(E) * I * H / 16); M.us = G.scales(size_t(E) * I * H / 16); M.ds = G.scales(size_t(E) * H * I / 16);
    M.g1.resize(E); M.g2.resize(E);
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t ee = model_shape ? e % 16 : e;  // the HIP test's alphas; cycled at E 512 to keep their range
        M.g1[e] = (1.f / M.a1) * 0.004f * (1 + 0.1f * ee); M.g2[e] = (1.f / M.a2) * 0.004f * (1 + 0.05f * ee);
    }
}

// MXFP8 weights for the MXFP8 rows (k2w_test's quantize_mx rule): per 32-element block a UE8M0 scale
// 2^(floor(log2 amax) - 8), each element E4M3 round-to-nearest (ties to the even code), saturating at 448; `up` is the
// exact BF16 upcast q * 2^(s - 127) (checked exact), the tensor the MXFP8 call must reproduce bit for bit.
struct MxW { std::vector<uint8_t> q, s; std::vector<uint16_t> up; };
static double mx_e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const double v = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
    return (b & 0x80) ? -v : v;
}
static MxW mx_quantize(const std::vector<uint16_t> &v, uint32_t n, uint32_t k) {
    static double pos[127];  // the positive E4M3 values, codes 0 .. 126 (increasing)
    for (int c = 0; c < 127; ++c) pos[c] = mx_e4m3(uint8_t(c));
    MxW w; w.q.resize(size_t(n) * k); w.s.resize(size_t(n) * k / 32); w.up.resize(w.q.size());
    for (size_t b = 0; b < w.s.size(); ++b) {
        double amax = 0; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(double(bf2f(v[b * 32 + i]))));
        int e = 0; if (amax > 0) { std::frexp(amax, &e); e -= 1; }
        const int se = std::clamp(e - 8, -117, 119);
        w.s[b] = uint8_t(se + 127);
        for (int i = 0; i < 32; ++i) {
            const double t = std::fabs(double(bf2f(v[b * 32 + i]))) / std::ldexp(1.0, se);
            int c = int(std::lower_bound(pos, pos + 127, t) - pos);  // first code >= t
            if (c >= 127) c = 126;                                    // saturate at 448
            else if (c > 0) { const double dl = t - pos[c - 1], dh = pos[c] - t; if (dl < dh || (dl == dh && ((c - 1) & 1) == 0)) c = c - 1; }
            w.q[b * 32 + i] = uint8_t(c | ((v[b * 32 + i] & 0x8000) ? 0x80 : 0));
        }
    }
    for (size_t i = 0; i < w.q.size(); ++i) {
        const double x = std::ldexp(mx_e4m3(w.q[i]), int(w.s[i / 32]) - 127);
        w.up[i] = f2bf(float(x));
        if (double(bf2f(w.up[i])) != x) { std::printf("MXFP8 upcast not exact\n"); std::exit(2); }
    }
    return w;
}

// The layer `L` with its shared expert's weights replaced by MXFP8 (bufs keep the uploads) or by their BF16 upcast.
struct MxShared {
    std::vector<Buffer> bufs;
    kv::MoeLayer mx, up;
    MxShared(Context &ctx, const kv::MoeLayer &L, const std::vector<uint16_t> &sg, const std::vector<uint16_t> &su, const std::vector<uint16_t> &sd,
             uint32_t H, uint32_t S) : mx(L), up(L) {
        const MxW g = mx_quantize(sg, S, H), u = mx_quantize(su, S, H), d = mx_quantize(sd, H, S);
        auto put = [&](const auto &v) { bufs.push_back(kt::upload(ctx, v)); return bufs.back().addr; };
        mx.shared_gate = put(g.q); mx.shared_gate_scales = put(g.s); mx.shared_up = put(u.q); mx.shared_up_scales = put(u.s);
        mx.shared_down = put(d.q); mx.shared_down_scales = put(d.s);
        up.shared_gate = put(g.up); up.shared_up = put(u.up); up.shared_down = put(d.up);
    }
    void release(Context &ctx) { for (auto &b : bufs) ctx.free(b); bufs.clear(); }
};

// The --tune knobs (ktest_util.h hooks), applied to every moe_block call of the test.
static kv::MoeOptions g_tune;
static cere_status tuned_block(Context &ctx, const kv::MoeDims &d, const kv::MoeLayer &L, uint64_t x, uint32_t m, uint64_t out,
                               uint64_t ws, size_t ws_bytes, const kv::MoeOptions *o = nullptr) {
    kv::MoeOptions t = o ? *o : kv::MoeOptions{};
    t.gemv_lanes_long = g_tune.gemv_lanes_long; t.gemv_lanes_short = g_tune.gemv_lanes_short; t.gemm_bn = g_tune.gemm_bn;
    return kv::moe_block(ctx, d, L, x, m, out, ws, ws_bytes, &t);
}
static std::vector<uint16_t> run_block(Context &ctx, const Model &M, uint64_t x, uint32_t m, Buffer &out, Buffer &ws,
                                       const kv::MoeOptions *o = nullptr) {
    ctx.begin();
    kt::must(tuned_block(ctx, M.d, M.L, x, m, out.addr, ws.addr, ws.size, o), "moe_block");
    ctx.barrier();
    ctx.submit_wait();
    return kt::host<uint16_t>(out, size_t(m) * M.d.hidden);
}
static kv::MoeOptions with_cls(uint32_t cls) { kv::MoeOptions o; o.cls = cls; return o; }
static long diff_rows(const std::vector<uint16_t> &a, size_t a_row, const std::vector<uint16_t> &b, size_t b_row, size_t rows, uint32_t H) {
    long n = 0;
    for (size_t i = 0; i < rows * H; ++i) n += a[a_row * H + i] != b[b_row * H + i];
    return n;
}
static double rel_l2(const Model &M, const std::vector<uint16_t> &x, const std::vector<uint16_t> &out, const std::vector<uint32_t> &rows, size_t out_row0 = 0) {
    const uint32_t H = M.d.hidden; double num = 0, den = 0;
    for (uint32_t r : rows) {
        const auto ref = M.mirror(&x[size_t(r) * H]);
        for (uint32_t o = 0; o < H; ++o) { const double e = bf2f(out[(size_t(r) - out_row0) * H + o]) - ref[o]; num += e * e; den += ref[o] * ref[o]; }
    }
    return std::sqrt(num / den);
}

int main(int argc, char **argv) {
    std::string spv = "build/vulkan-k7/spv";
    bool timing = true, model = true, explore = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (kt::hook_arg(argc, argv, i)) continue;
        if (a == "--no-timing") timing = false;
        else if (a == "--no-model") model = false;
        else if (a == "--explore") explore = true;
        else spv = a;
    }
    kt::tune_keys({"gemv_lanes_long", "gemv_lanes_short", "gemm_bn"});
    g_tune.gemv_lanes_long = kt::tune_u32("gemv_lanes_long", 0);
    g_tune.gemv_lanes_short = kt::tune_u32("gemv_lanes_short", 0);
    g_tune.gemm_bn = kt::tune_u32("gemm_bn", 0);
    auto ctx_owner = cere::vk::testing::create_or_die(spv);
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s, subgroup %u..%u\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(),
                ctx.info().driver_info.c_str(), ctx.info().subgroup_min, ctx.info().subgroup_max);
    {
        const cere_status s = kv::probe(ctx);
        check("coopmat layout", s == CERE_OK, "k7_probe: accumulator element i of lane l == (2 i + l / 16, l % 16), length 8 (k7_gemm.comp's map)");
        const cere_status s2 = cere::vkk2::gemm_probe(ctx);
        check("K2 GEMM probe", s2 == CERE_OK, "k2_gemm_probe: the K2 GEMM's accumulator map (row classes >= 17 of the router / shared expert)");
    }

    // ---- the HIP test's shape and data
    const uint32_t H = 256, E = 16, K = 4, I = 128, S = 128, MAXM = 64;
    Gen G(43);
    Model M;
    make_model(M, G, kv::MoeDims{H, E, K, I, S}, false);
    M.upload(ctx);
    const size_t wsb = kv::moe_workspace_bytes(M.d, MAXM);
    Buffer ws = kt::make(ctx, wsb), dout = kt::make(ctx, size_t(MAXM) * H * 2);
    auto x = G.rnd(size_t(MAXM) * H, 1.f);
    Buffer dx = kt::upload(ctx, x);
    {   // 1. moe_route
        const uint32_t m = 40; std::vector<uint16_t> lg = G.rnd(size_t(m) * E, 2.f);
        for (uint32_t r = 0; r < m; r += 5) lg[r * E + 3] = lg[r * E + 9];  // ties
        Buffer dl = kt::upload(ctx, lg), dids = kt::make(ctx, m * K * 4), dwts = kt::make(ctx, m * K * 4);
        ctx.begin(); kt::must(kv::moe_route(ctx, dl.addr, dids.addr, dwts.addr, m, E, K), "moe_route"); ctx.barrier(); ctx.submit_wait();
        auto ids = kt::host<int32_t>(dids, m * K); auto wts = kt::host<float>(dwts, m * K);
        long idbad = 0; double wbad = 0;
        for (uint32_t r = 0; r < m; ++r) {
            std::vector<float> p(E); float mx = -1e30f; for (uint32_t e = 0; e < E; ++e) mx = std::max(mx, bf2f(lg[r * E + e]));
            float sum = 0; for (uint32_t e = 0; e < E; ++e) { p[e] = std::exp(bf2f(lg[r * E + e]) - mx); sum += p[e]; }
            std::vector<uint32_t> idx(E); for (uint32_t e = 0; e < E; ++e) idx[e] = e;
            std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return lg[r * E + a] != lg[r * E + b] ? bf2f(lg[r * E + a]) > bf2f(lg[r * E + b]) : a < b; });
            double ks = 0; for (uint32_t k = 0; k < K; ++k) ks += p[idx[k]] / sum;
            for (uint32_t k = 0; k < K; ++k) { idbad += uint32_t(ids[r * K + k]) != idx[k]; wbad = std::max(wbad, std::fabs(wts[r * K + k] - (p[idx[k]] / sum) / ks) / ((p[idx[k]] / sum) / ks)); }
        }
        char msg[160]; std::snprintf(msg, sizeof msg, "40 rows: %ld ids differ (ties included); weights worst rel %.1e (2.4e-7)", idbad, wbad);
        check("moe_route", idbad == 0 && wbad <= 2.4e-7, msg);
        ctx.free(dl); ctx.free(dids); ctx.free(dwts);
    }
    {   // added: non-finite logits (k7_moe.h moe_route): K distinct ids in [0, E), weights 0 for a row whose softmax
        // denominator is not finite positive, finite rows unaffected. Rows: 0 a NaN logit, 1 a +Inf, 2 all -Inf, 3 all NaN, 4 finite.
        const uint32_t m = 5; std::vector<uint16_t> lg = G.rnd(size_t(m) * E, 2.f);
        const std::vector<uint16_t> ref4(lg.begin() + 4 * E, lg.end());
        lg[0 * E + 5] = 0x7FC1; lg[1 * E + 7] = 0x7F80;
        for (uint32_t e = 0; e < E; ++e) { lg[2 * E + e] = 0xFF80; lg[3 * E + e] = 0xFFC0; }
        Buffer dl = kt::upload(ctx, lg), dids = kt::make(ctx, m * K * 4), dwts = kt::make(ctx, m * K * 4);
        ctx.begin(); kt::must(kv::moe_route(ctx, dl.addr, dids.addr, dwts.addr, m, E, K), "moe_route"); ctx.barrier(); ctx.submit_wait();
        auto ids = kt::host<int32_t>(dids, m * K); auto wts = kt::host<float>(dwts, m * K);
        bool ok = true;
        for (uint32_t r = 0; r < m; ++r) {
            std::vector<int> seen(E, 0);
            for (uint32_t k = 0; k < K; ++k) { const int32_t id = ids[r * K + k]; ok &= id >= 0 && uint32_t(id) < E && !seen[id]; if (id >= 0 && uint32_t(id) < E) seen[id] = 1; ok &= std::isfinite(wts[r * K + k]); }
            if (r < 4) for (uint32_t k = 0; k < K; ++k) ok &= wts[r * K + k] == 0.f;
        }
        ok &= ids[1 * K] == 7;  // the +Inf logit is the top pick
        double s4 = 0; for (uint32_t k = 0; k < K; ++k) s4 += wts[4 * K + k];
        ok &= std::fabs(s4 - 1.0) < 1e-6;
        check("route non-finite", ok, "NaN / +Inf / all -Inf / all NaN rows: K distinct ids in [0, E), weights 0; a finite row sums to 1");
        ctx.free(dl); ctx.free(dids); ctx.free(dwts);
    }
    for (uint32_t m : {7u, 40u}) {  // 2. moe_block
        auto out = run_block(ctx, M, dx.addr, m, dout, ws);
        std::vector<uint32_t> rows(m); for (uint32_t r = 0; r < m; ++r) rows[r] = r;
        kt::identity("block_m" + std::to_string(m), out);
        const double err = rel_l2(M, x, out, rows);
        const bool rep = run_block(ctx, M, dx.addr, m, dout, ws) == out;
        char msg[160]; std::snprintf(msg, sizeof msg, "m %2u: out rel-L2 %.2e vs the host mirror (1e-2); repeat %s", m, err, rep ? "bit-equal" : "DIFFERS");
        check("moe_block", err <= 1e-2 && rep, msg);
    }
    const auto big = run_block(ctx, M, dx.addr, 40, dout, ws);  // m 40: cooperative-matrix GEMM, fused grouping
    const kv::MoeOptions c40 = with_cls(40);  // the HIP test's p40 (k7_test.cpp:157, :167)
    {   // 3. GEMV == GEMM kernel
        auto small = run_block(ctx, M, dx.addr, 7, dout, ws, &c40);
        char msg[160]; std::snprintf(msg, sizeof msg, "m 7 (GEMV kernel, cls 40) vs rows 0..6 of m 40 (coopmat GEMM): %ld of %zu differ", diff_rows(small, 0, big, 0, 7, H), small.size());
        check("GEMV == GEMM kernel", diff_rows(small, 0, big, 0, 7, H) == 0, msg);
    }
    {   // 4. WMMA == GEMV kernel
        long diff = 0;
        for (uint32_t r0 : {0u, 8u, 31u}) { auto one = run_block(ctx, M, dx.addr + uint64_t(r0) * H * 2, 1, dout, ws, &c40); diff += diff_rows(one, 0, big, r0, 1, H); }
        char msg[200]; std::snprintf(msg, sizeof msg, "rows 0, 8, 31 of m 40 (coopmat GEMM) vs each row alone (GEMV, cls 40): %ld of %u differ", diff, 3 * H);
        check("WMMA == GEMV kernel", diff == 0, msg);
    }
    {   // 5. batch invariance
        auto all = run_block(ctx, M, dx.addr, 5, dout, ws); long diff = 0;
        for (uint32_t r = 0; r < 5; ++r) { auto one = run_block(ctx, M, dx.addr + uint64_t(r) * H * 2, 1, dout, ws); diff += diff_rows(one, 0, all, r, 1, H); }
        char msg[160]; std::snprintf(msg, sizeof msg, "each row of a 5-row block == that row alone, bit for bit: %ld of %u differ", diff, 5 * H);
        check("batch invariance", diff == 0, msg);
    }
    {   // added: forced paths change no bit
        kv::MoeOptions cm; cm.gemm = kv::GemmPath::kCoopmat;
        kv::MoeOptions sp; sp.group = kv::GroupPath::kSplit;
        const auto s7 = run_block(ctx, M, dx.addr, 7, dout, ws);
        const long d1 = diff_rows(run_block(ctx, M, dx.addr, 7, dout, ws, &cm), 0, s7, 0, 7, H);
        const long d2 = diff_rows(run_block(ctx, M, dx.addr, 7, dout, ws, &sp), 0, s7, 0, 7, H);
        const long d3 = diff_rows(run_block(ctx, M, dx.addr, 40, dout, ws, &sp), 0, big, 0, 40, H);
        char msg[220]; std::snprintf(msg, sizeof msg, "m 7 coopmat vs GEMV %ld, m 7 split vs fused grouping %ld, m 40 split vs fused %ld differ", d1, d2, d3);
        check("forced paths", d1 == 0 && d2 == 0 && d3 == 0, msg);
    }
    {   // added: refusals (a forced path outside its limits)
        kv::MoeOptions gv; gv.gemm = kv::GemmPath::kGemv; gv.tiles = kv::GemvTiles::kExpert;
        kv::MoeOptions fu; fu.group = kv::GroupPath::kFused;
        ctx.begin();
        const cere_status r1 = tuned_block(ctx, M.d, M.L, dx.addr, 9, dout.addr, ws.addr, ws.size, &gv);
        const cere_status r2 = tuned_block(ctx, M.d, M.L, dx.addr, 65, dout.addr, ws.addr, ws.size, &fu);  // also > workspace
        const cere_status r3 = tuned_block(ctx, kv::MoeDims{H, E, K, 100, S}, M.L, dx.addr, 1, dout.addr, ws.addr, ws.size);
        const kv::MoeOptions c5 = with_cls(5);
        const cere_status r4 = tuned_block(ctx, M.d, M.L, dx.addr, 7, dout.addr, ws.addr, ws.size, &c5);
        ctx.submit_wait();
        char msg[200]; std::snprintf(msg, sizeof msg, "GEMV expert tiles at m 9: %d, m 65 > workspace: %d, I 100: %d, cls 5 < m 7: %d (CERE_ERR_ARGUMENT %d)", int(r1), int(r2), int(r3), int(r4), int(CERE_ERR_ARGUMENT));
        check("refusals", r1 == CERE_ERR_ARGUMENT && r2 == CERE_ERR_ARGUMENT && r3 == CERE_ERR_ARGUMENT && r4 == CERE_ERR_ARGUMENT, msg);
    }
    {   // added: prepared K2 weight info (moe_prepare_weights) == the weights scanned in the call, m 40 (class 40) and m 7
        Buffer info = kt::make(ctx, kv::moe_weight_info_bytes(M.d));
        kv::MoeLayer Lp = M.L; Lp.bf16_info = info.addr;
        ctx.begin(); kt::must(kv::moe_prepare_weights(ctx, M.d, Lp), "moe_prepare_weights"); ctx.barrier(); ctx.submit_wait();
        kv::MoeLayer keep = M.L; M.L = Lp;
        const long d40 = diff_rows(run_block(ctx, M, dx.addr, 40, dout, ws), 0, big, 0, 40, H);
        const auto s7p = run_block(ctx, M, dx.addr, 7, dout, ws, &c40);
        M.L = keep;
        const long d7 = diff_rows(s7p, 0, run_block(ctx, M, dx.addr, 7, dout, ws, &c40), 0, 7, H);
        char msg[160]; std::snprintf(msg, sizeof msg, "m 40: %ld, m 7 cls 40: %ld outputs differ from the in-call scan", d40, d7);
        check("prepared weight info", d40 == 0 && d7 == 0, msg);
        ctx.free(info);
    }
    {   // added: MXFP8 shared expert (S3) == its BF16 upcast, bit for bit
        MxShared X(ctx, M.L, M.sg, M.su, M.sd, H, S);
        const kv::MoeLayer keep = M.L;
        auto with = [&](const kv::MoeLayer &L, uint64_t x, uint32_t m, const kv::MoeOptions *o = nullptr) { M.L = L; auto r = run_block(ctx, M, x, m, dout, ws, o); M.L = keep; return r; };
        long dv = 0, dg = 0;
        for (uint32_t m : {1u, 7u}) dv += diff_rows(with(X.mx, dx.addr, m), 0, with(X.up, dx.addr, m), 0, m, H);
        dg += diff_rows(with(X.mx, dx.addr, 40), 0, with(X.up, dx.addr, 40), 0, 40, H);
        dg += diff_rows(with(X.mx, dx.addr, 7, &c40), 0, with(X.up, dx.addr, 7, &c40), 0, 7, H);
        const bool differs = with(X.mx, dx.addr, 40) != big;  // the quantized weights are a different layer (sanity)
        char msg[220]; std::snprintf(msg, sizeof msg, "m 1 + 7 (K2 GEMV): %ld, m 40 + m 7 cls 40 (K2 GEMM): %ld outputs differ from the BF16-upcast call; != the BF16 layer: %s",
                                     dv, dg, differs ? "yes" : "NO");
        check("MXFP8 == upcast", dv == 0 && dg == 0 && differs, msg);
        // invariance on MXFP8: rows of a 5-row block == alone (class <= 16); m 7 cls 40 == rows 0..6 of m 40
        const auto all5 = with(X.mx, dx.addr, 5), all40 = with(X.mx, dx.addr, 40);
        long di = 0;
        for (uint32_t r = 0; r < 5; ++r) di += diff_rows(with(X.mx, dx.addr + uint64_t(r) * H * 2, 1), 0, all5, r, 1, H);
        const long dc = diff_rows(with(X.mx, dx.addr, 7, &c40), 0, all40, 0, 7, H);
        std::snprintf(msg, sizeof msg, "MXFP8: 5-row block rows vs alone %ld, m 7 cls 40 vs rows 0..6 of m 40 %ld differ", di, dc);
        check("MXFP8 invariance", di == 0 && dc == 0, msg);
        // prepared info scanned in the MXFP8 format == the in-call scan (m 40)
        Buffer info = kt::make(ctx, kv::moe_weight_info_bytes(M.d));
        kv::MoeLayer Lp = X.mx; Lp.bf16_info = info.addr;
        ctx.begin(); kt::must(kv::moe_prepare_weights(ctx, M.d, Lp), "moe_prepare_weights (MXFP8)"); ctx.barrier(); ctx.submit_wait();
        const long dp = diff_rows(with(Lp, dx.addr, 40), 0, all40, 0, 40, H);
        std::snprintf(msg, sizeof msg, "m 40 with the MXFP8 weights' prepared info: %ld outputs differ from the in-call scan", dp);
        check("MXFP8 prepared info", dp == 0, msg);
        ctx.free(info);
        // refusal: MXFP8 shared_down (k = S) with S % 32 != 0 (K2's MXFP8 GEMV rule), before anything is recorded
        const size_t pre = ctx.pipeline_count();
        ctx.begin();
        const cere_status r1 = tuned_block(ctx, kv::MoeDims{H, E, K, I, 48}, X.mx, dx.addr, 1, dout.addr, ws.addr, ws.size);
        ctx.submit_wait();
        std::snprintf(msg, sizeof msg, "MXFP8 shared_down at S 48 (k %% 32 != 0): %d (ARGUMENT %d); nothing recorded: %s", int(r1), int(CERE_ERR_ARGUMENT),
                      ctx.pipeline_count() == pre ? "yes" : "NO");
        check("MXFP8 refusals", r1 == CERE_ERR_ARGUMENT && ctx.pipeline_count() == pre, msg);
        X.release(ctx);
    }
    M.release(ctx); ctx.free(ws); ctx.free(dout); ctx.free(dx);

    if (model || timing || explore) {
        // ---- the model's shape (amd-amd design note step 11): H 2,560, E 512, top-10, I 640, S 640
        const uint32_t MH = 2560, ME = 512, MK = 10, MI = 640, MS = 640, PM = 4096, POOL = 4096;
        const auto t0 = std::chrono::steady_clock::now();
        Gen G2(7);
        Model R;
        make_model(R, G2, kv::MoeDims{MH, ME, MK, MI, MS}, true);
        R.upload(ctx);
        Buffer rinfo = kt::make(ctx, kv::moe_weight_info_bytes(R.d));  // the K2 weight info, prepared once (at load)
        R.L.bf16_info = rinfo.addr;
        ctx.begin(); kt::must(kv::moe_prepare_weights(ctx, R.d, R.L), "moe_prepare_weights"); ctx.barrier(); ctx.submit_wait();
        auto xp = G2.rnd(size_t(POOL) * MH, 1.f);
        Buffer px = kt::upload(ctx, xp);
        MxShared RX(ctx, R.L, R.sg, R.su, R.sd, MH, MS);  // the MXFP8 shared expert (S3 format) and its upcast
        Buffer rinfo_mx = kt::make(ctx, kv::moe_weight_info_bytes(R.d));
        RX.mx.bf16_info = rinfo_mx.addr;
        ctx.begin(); kt::must(kv::moe_prepare_weights(ctx, R.d, RX.mx), "moe_prepare_weights (MXFP8)"); ctx.barrier(); ctx.submit_wait();
        const size_t rwsb = kv::moe_workspace_bytes(R.d, PM);
        Buffer rws = kt::make(ctx, rwsb), rout = kt::make(ctx, size_t(PM) * MH * 2);
        auto csum = [](const auto &v, uint64_t c) { for (auto e : v) c = c * 1000003u + uint64_t(e); return c; };
        const uint64_t cs = csum(xp, csum(R.ds, csum(R.gw, csum(R.router, 1))));  // hip_k7_timing prints the same
        std::printf("model shape: data checksum %016llx\n", (unsigned long long)cs);
        std::printf("model shape: weights %.2f GB, workspace %.0f MB, set up in %.1f s\n",
                    double(R.gw.size() + R.uw.size() + R.dw.size() + R.gs.size() + R.us.size() + R.ds.size()) / 1e9, rwsb / 1e6,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        if (model) {
            const auto pre = run_block(ctx, R, px.addr, PM, rout, rws);
            kt::identity("model_m4096", pre);
            for (uint32_t mi : {1u, 5u}) kt::identity("model_m" + std::to_string(mi), run_block(ctx, R, px.addr, mi, rout, rws));
            const bool rep = run_block(ctx, R, px.addr, PM, rout, rws) == pre;
            long diff = 0;
            const std::vector<uint32_t> rows = {0, 1777, 4095};
            const kv::MoeOptions c4096 = with_cls(PM);  // the prefill's class (HIP k7_test.cpp:157's cls pin)
            for (uint32_t r : rows) diff += diff_rows(run_block(ctx, R, px.addr + uint64_t(r) * MH * 2, 1, rout, rws, &c4096), 0, pre, r, 1, MH);
            const double err = rel_l2(R, xp, pre, rows);
            char msg[220]; std::snprintf(msg, sizeof msg, "m 4096: rows 0, 1777, 4095 vs alone (cls 4096) %ld of %u differ; rel-L2 %.2e vs the mirror (1e-2); repeat %s",
                                         diff, 3 * MH, err, rep ? "bit-equal" : "DIFFERS");
            check("model prefill", diff == 0 && err <= 1e-2 && rep, msg);
            const auto v16 = run_block(ctx, R, px.addr, 16, rout, rws);
            kt::identity("model_m16", v16);
            long d16 = 0;
            for (uint32_t r : {0u, 9u, 15u}) d16 += diff_rows(run_block(ctx, R, px.addr + uint64_t(r) * MH * 2, 1, rout, rws), 0, v16, r, 1, MH);
            const double e16 = rel_l2(R, xp, v16, {0, 9, 15});
            kv::MoeOptions cm; cm.gemm = kv::GemmPath::kCoopmat;
            const long dcm = diff_rows(run_block(ctx, R, px.addr, 16, rout, rws, &cm), 0, v16, 0, 16, MH);
            std::snprintf(msg, sizeof msg, "m 16 (GEMV row tiles, fused grouping): rows 0, 9, 15 vs alone %ld differ; rel-L2 %.2e (1e-2); coopmat m 16 vs GEMV %ld of %u differ",
                          d16, e16, dcm, 16 * MH);
            check("model verify", d16 == 0 && e16 <= 1e-2 && dcm == 0, msg);
            {   // MXFP8 shared expert at the model shape == its BF16 upcast (m 1, 16: GEMV; m 4,096: GEMM)
                const kv::MoeLayer keep = R.L;
                long dm = 0;
                for (uint32_t m : {1u, 16u, 4096u}) {
                    R.L = RX.mx; const auto a = run_block(ctx, R, px.addr, m, rout, rws);
                    R.L = RX.up; R.L.bf16_info = 0; const auto b = run_block(ctx, R, px.addr, m, rout, rws);
                    dm += diff_rows(a, 0, b, 0, m, MH);
                }
                R.L = keep;
                std::snprintf(msg, sizeof msg, "MXFP8 shared expert (prepared info) vs the BF16-upcast call at m 1 / 16 / 4,096: %ld outputs differ", dm);
                check("model MXFP8 == upcast", dm == 0, msg);
            }
        }
        if (explore) {  // kernel-path comparison at decode / verify widths (not contract rows)
            struct V { const char *name; kv::MoeOptions o; };
            std::vector<V> vs(3);
            vs[0].name = "gemv expert tiles"; vs[0].o.gemm = kv::GemmPath::kGemv; vs[0].o.tiles = kv::GemvTiles::kExpert;
            vs[1].name = "gemv row tiles"; vs[1].o.gemm = kv::GemmPath::kGemv; vs[1].o.tiles = kv::GemvTiles::kRow;
            vs[2].name = "coopmat"; vs[2].o.gemm = kv::GemmPath::kCoopmat;
            for (uint32_t m : {1u, 2u, 4u, 5u, 8u, 12u, 16u, 24u, 32u, 64u}) {
                for (auto &v : vs) {
                    if (v.o.tiles == kv::GemvTiles::kExpert && m > kv::kGemvMaxExpertRows) continue;
                    ctx.begin();
                    for (int i = 0; i < 3; ++i) { kt::must(tuned_block(ctx, R.d, R.L, px.addr + uint64_t(i * 61 * m % (POOL - m)) * MH * 2, m, rout.addr, rws.addr, rws.size, &v.o), "moe_block"); ctx.barrier(); }
                    for (int i = 0; i < 11; ++i) {
                        ctx.stamp(2 * i);
                        kt::must(tuned_block(ctx, R.d, R.L, px.addr + uint64_t((i + 3) * 61 * m % (POOL - m)) * MH * 2, m, rout.addr, rws.addr, rws.size, &v.o), "moe_block");
                        ctx.stamp(2 * i + 1);
                        ctx.barrier();
                    }
                    ctx.submit_wait();
                    std::vector<double> t; for (int i = 0; i < 11; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
                    std::sort(t.begin(), t.end());
                    std::printf("  explore m %3u %-18s rot median %.3f ms\n", m, v.name, t[5]);
                }
            }
        }
        if (explore) {  // phase breakdown (one call between timestamps at each phase boundary; median of 7)
            const char *names[kv::kProfileStamps - 1] = {"router+shared g/u+q1+gate", "route(+group)+silu", "split grouping", "GEMM1+shared down", "q2", "GEMM2", "combine"};
            for (uint32_t m : {1u, 16u, 1205u, 4096u}) {
                const int reps = 7;
                std::vector<std::vector<double>> ph(kv::kProfileStamps - 1);
                ctx.begin();
                for (int i = 0; i < reps + 1; ++i) {
                    kv::MoeOptions po; po.profile_stamp0 = int32_t(i * kv::kProfileStamps);
                    kt::must(tuned_block(ctx, R.d, R.L, px.addr + uint64_t(m < POOL ? (i * 61 * m) % (POOL - m) : 0) * MH * 2, m, rout.addr, rws.addr, rws.size, &po), "moe_block");
                    ctx.barrier();
                }
                ctx.submit_wait();
                for (int i = 1; i <= reps; ++i)
                    for (uint32_t p = 0; p + 1 < kv::kProfileStamps; ++p) ph[p].push_back(ctx.stamp_ms(i * kv::kProfileStamps + p, i * kv::kProfileStamps + p + 1));
                std::printf("  phases m %4u:", m);
                for (uint32_t p = 0; p + 1 < kv::kProfileStamps; ++p) { std::sort(ph[p].begin(), ph[p].end()); std::printf(" %s %.3f |", names[p], ph[p][reps / 2]); }
                std::printf("\n");
            }
        }
        if (timing) {
            // GPU time of one moe_block call: 3 untimed calls, then `reps` calls each between its own pair of timestamps
            // with a barrier after it. hot: every call on rows 0 .. m-1 (the same experts: partly cached in the 32 MB
            // MALL, the HIP step-15 harness's method); rot: call i on rows starting at (i * 61 * m) % (POOL - m), so each
            // call routes to other experts (closer to a real decode token's ~37 MB uncached).
            std::printf("timing (model shape, GPU timestamps):\n");
            std::vector<uint32_t> ms = {1u, 5u, 8u, 16u, 1205u, 4096u};
            if (kt::hooks().samples) ms.insert(ms.begin() + 4, 256u);
            for (uint32_t m : ms) {
                for (int rot = 0; rot <= (m < POOL ? 1 : 0); ++rot) {
                    const int reps = kt::reps(m >= 1024 ? 7 : 15);
                    auto xa = [&](int i) { return px.addr + uint64_t(rot ? (uint64_t(i) * 61 * m) % (POOL - m) : 0) * MH * 2; };
                    ctx.begin();
                    int c = 0;
                    for (int i = 0; i < 3; ++i, ++c) { kt::must(tuned_block(ctx, R.d, R.L, xa(c), m, rout.addr, rws.addr, rws.size), "moe_block"); ctx.barrier(); }
                    for (int i = 0; i < reps; ++i, ++c) {
                        ctx.stamp(2 * i);
                        kt::must(tuned_block(ctx, R.d, R.L, xa(c), m, rout.addr, rws.addr, rws.size), "moe_block");
                        ctx.stamp(2 * i + 1);
                        ctx.barrier();
                    }
                    ctx.submit_wait();
                    std::vector<double> t;
                    for (int i = 0; i < reps; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
                    kt::samples("moe_block.m" + std::to_string(m), rot ? "cold" : "warm", t);
                    std::sort(t.begin(), t.end());
                    std::printf("  vulkan moe_block m %4u %s: median %.3f ms (min %.3f, max %.3f, n %d)\n", m, rot ? "rot" : "hot", t[t.size() / 2], t.front(), t.back(), reps);
                    std::fflush(stdout);
                }
            }
        }
        if (timing) {  // MXFP8 shared expert (S3 format, prepared info) next to BF16
            for (uint32_t m : {1u, 4096u}) {
                for (int fmt = 0; fmt < 2; ++fmt) {
                    const kv::MoeLayer &L = fmt ? RX.mx : R.L;
                    const int reps = m >= 1024 ? 7 : 15;
                    ctx.begin();
                    for (int i = 0; i < 3; ++i) { kt::must(tuned_block(ctx, R.d, L, px.addr, m, rout.addr, rws.addr, rws.size), "moe_block"); ctx.barrier(); }
                    for (int i = 0; i < reps; ++i) {
                        ctx.stamp(2 * i);
                        kt::must(tuned_block(ctx, R.d, L, px.addr, m, rout.addr, rws.addr, rws.size), "moe_block");
                        ctx.stamp(2 * i + 1);
                        ctx.barrier();
                    }
                    ctx.submit_wait();
                    std::vector<double> t;
                    for (int i = 0; i < reps; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
                    std::sort(t.begin(), t.end());
                    std::printf("  vulkan moe_block m %4u hot, shared expert %s: median %.3f ms (min %.3f, max %.3f, n %d)\n", m, fmt ? "MXFP8" : "BF16 ", t[t.size() / 2],
                                t.front(), t.back(), reps);
                }
            }
        }
        RX.release(ctx); ctx.free(rinfo_mx);
        R.release(ctx); ctx.free(px); ctx.free(rws); ctx.free(rout); ctx.free(rinfo);
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
