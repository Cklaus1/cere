// VulkanAmd P2: executor/hip/ktest/k5_layer_test.cpp's contract rows on the Vulkan K5 GDN mixer layer
// (src/kernels/k5_gdn_layer_vk.*: K2 projections routed by row class, conv1d + SiLU, gating, the Vulkan chunk rule for
// PREFILL, the Vulkan decode rule for DECODE, K1 gated norm, out_proj), at hidden 256 with the model's GDN dims; same
// seed, data, bounds and row class as the HIP test: every layer_forward and projection call passes cls 128 as the HIP
// test does (k5_layer_test.cpp:10-11, :73, :145-146), so all of them run the shift-invariant K2 GEMM. Rows:
//  1. layer_forward PREFILL, 100 tokens from a fresh state, vs an FP64 reference of Qwen3_5GatedDeltaNet.forward from
//     the same BF16 inputs (projections rounded to BF16; conv1d + SiLU; gating; the exact gated delta rule on
//     L2-normalised q, k; gated RMSNorm; out_proj): out and ssm state rel-L2 <= 2e-2;
//  2. prefill 99 then DECODE 1 vs prefill 100: last row out rel-L2 <= 2e-2, ssm state <= 1e-2, conv state bit-equal;
//  3. EndState at row 64 (FP32 ssm + conv state captured, then a 36-row PREFILL from them):
//     a. layer_core: the capture leaves the call unchanged; rows 64..99 of y, the final conv and ssm bit-equal;
//     b. layer_forward (with the projections): the same, out rows 64..99 bit-equal.
// Extra rows (not in the HIP test): K2 GEMM probe; prepared K2 weight info (prepare_weights) == the in-call scan;
// refusals (cls < tokens too). Then timing (GPU timestamps, median of 5 after a warm-up) at the model's hidden 2,560,
// cls 0 (the HIP timing harness's), prepared weight info: layer_forward PREFILL of 4,096 and 512 rows (and the 4,096
// split: projections / mixer core / out_proj), and the DECODE handoff step (one token, the K2 GEMV).
// MXFP8 rows (the S3 artifact stores the five projections as MXFP8: executor/cuda/src/model/mxfp8_plan.h kGdn): the test's
// BF16 weights MXFP8-quantized (k2w_test's rule); the layer with MXFP8 projections == the layer with their exact BF16 upcast,
// bit for bit (out, ssm, conv state) at a prefill class (cls 128: K2 GEMM), at the GEMV class (PREFILL 10 + DECODE 1, cls 0),
// with mixed formats, with prepared weight info, and the EndState capture / resume on MXFP8; an MXFP8 in-projection at
// hidden_size % 32 != 0 refused. MXFP8 timing (decode, PREFILL 512 / 4,096) next to the BF16 numbers.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "launch/vk_launch_testing.h"
#include "kernels/k2_gemm_vk.h"
#include "kernels/k5_gdn_layer_vk.h"
#include "kernels/k5_gdn_vk.h"
#include "ktest_util.h"

using kt::bf2f;
using kt::f2bf;
namespace k5 = cere::vkk5;
static int g_fail = 0;
static double rb(double x) { return bf2f(f2bf(float(x))); }
static double rel(const double *ref, const uint16_t *got, size_t n) {
    double a = 0, d = 0; for (size_t i = 0; i < n; ++i) { const double e = bf2f(got[i]) - ref[i]; a += e * e; d += ref[i] * ref[i]; }
    return std::sqrt(a / d);
}
static void check(const char *name, bool ok, const char *fmt, double a = 0, double b = 0, double c = 0) {
    std::printf("%-24s %s  ", name, ok ? "PASS" : "FAIL"); std::printf(fmt, a, b, c); std::printf("\n"); if (!ok) ++g_fail;
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

constexpr int HID = 256, CD = k5::kConvDim, KD = k5::kKeyDim, VD = k5::kValueDim, VH = k5::kValueHeads, D = 128;

int main(int argc, char **argv) {
    auto ctx_owner = cere::vk::testing::create_or_die(argc > 1 ? argv[1] : "build/vulkan/spv");
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str());
    check("K2 GEMM probe", cere::vkk2::gemm_probe(ctx) == CERE_OK, "k2_gemm_probe: the K2 GEMM's accumulator map (projection row classes >= 17)");
    const uint32_t kCls = 128;  // the HIP test's row class for every call (k5_layer_test.cpp:73, :145-146)
    std::mt19937 g(17); std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd = [&](size_t n, float s, float mean = 0.f) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(mean + nd(g) * s); return v; };
    const float ws = 1.f / std::sqrt(float(HID));
    auto Wqkv = rnd(size_t(CD) * HID, ws), Wz = rnd(size_t(VD) * HID, ws), Wb = rnd(size_t(VH) * HID, ws), Wa = rnd(size_t(VH) * HID, ws);
    auto Wout = rnd(size_t(HID) * VD, 1.f / std::sqrt(float(VD))), conv_w = rnd(size_t(CD) * 4, 0.5f);
    auto dt_bias = rnd(VH, 0.5f), norm_w = rnd(D, 0.1f, 1.f);
    std::vector<float> A_log(VH); for (auto &x : A_log) x = std::log(0.5f + std::fabs(nd(g)));
    const int64_t T = 100;
    auto hidden = rnd(size_t(T) * HID, 1.f);

    auto bWqkv = kt::upload(ctx, Wqkv), bWz = kt::upload(ctx, Wz), bWb = kt::upload(ctx, Wb), bWa = kt::upload(ctx, Wa), bconv = kt::upload(ctx, conv_w);
    auto bAlog = kt::upload(ctx, A_log), bdt = kt::upload(ctx, dt_bias), bnorm = kt::upload(ctx, norm_w), bWout = kt::upload(ctx, Wout);
    const k5::LayerWeights w{bWqkv.addr, bWz.addr, bWb.addr, bWa.addr, bconv.addr, bAlog.addr, bdt.addr, bnorm.addr, bWout.addr};
    const size_t wsb = k5::layer_workspace_bytes(T); auto dws = kt::make(ctx, wsb);
    auto dh = kt::upload(ctx, hidden); auto dout = kt::make(ctx, size_t(T) * HID * 2);
    const std::vector<uint16_t> conv0(size_t(CD) * 3, 0), ssm0(size_t(VH) * D * D, 0);
    struct St { cere::vk::Buffer conv, ssm; };
    auto fresh = [&]() { St s{kt::upload(ctx, conv0), kt::upload(ctx, ssm0)}; return s; };
    auto fwd = [&](const St &s, uint32_t op, int64_t r0, int64_t n, const k5::EndState *end = nullptr) {
        ctx.begin();
        const cere_status st = k5::layer_forward(ctx, w, k5::LayerState{s.conv.addr, s.ssm.addr}, op, dh.addr + uint64_t(r0) * HID * 2, n, HID,
                                                 dout.addr + uint64_t(r0) * HID * 2, dws.addr, wsb, kCls, end);
        if (st != CERE_OK) { std::printf("layer_forward failed: %d\n", st); std::exit(2); }
        ctx.barrier();
        ctx.submit_wait();
    };
    auto hostv = [&](const cere::vk::Buffer &b, size_t n) { return kt::host<uint16_t>(b, n); };

    // --- 1. prefill 100 vs FP64
    St s1 = fresh(); fwd(s1, CERE_OP_PREFILL, 0, T);
    auto out1 = hostv(dout, size_t(T) * HID); auto conv1 = hostv(s1.conv, conv0.size()); auto ssm1 = hostv(s1.ssm, ssm0.size());
    std::vector<double> ref(size_t(T) * HID), Sref(size_t(VH) * D * D, 0.0);
    {
        std::vector<double> mixed(size_t(T) * CD), z(size_t(T) * VD), bb(T * VH), aa(T * VH);
        auto proj = [&](const std::vector<uint16_t> &W, int n, std::vector<double> &y) {
            for (int64_t t = 0; t < T; ++t) for (int o = 0; o < n; ++o) {
                double s = 0; for (int i = 0; i < HID; ++i) s += double(bf2f(hidden[t * HID + i])) * bf2f(W[size_t(o) * HID + i]);
                y[t * n + o] = rb(s);
            }
        };
        proj(Wqkv, CD, mixed); proj(Wz, VD, z); proj(Wb, VH, bb); proj(Wa, VH, aa);
        std::vector<double> cv(size_t(T) * CD);
        for (int64_t t = 0; t < T; ++t) for (int c = 0; c < CD; ++c) {
            double s = 0; for (int j = 0; j < 4; ++j) { const int64_t tt = t - 3 + j; if (tt >= 0) s += double(bf2f(conv_w[c * 4 + j])) * mixed[tt * CD + c]; }
            cv[t * CD + c] = rb(s / (1 + std::exp(-s)));
        }
        std::vector<double> o(size_t(T) * VD);
        for (int vh = 0; vh < VH; ++vh) {
            const int kh = vh / 3; double *S = &Sref[size_t(vh) * D * D];
            for (int64_t t = 0; t < T; ++t) {
                double q[D], k[D], nq = 0, nk = 0;
                for (int c = 0; c < D; ++c) { q[c] = cv[t * CD + kh * D + c]; k[c] = cv[t * CD + KD + kh * D + c]; nq += q[c] * q[c]; nk += k[c] * k[c]; }
                for (int c = 0; c < D; ++c) { q[c] /= std::sqrt(nq + 1e-6); k[c] /= std::sqrt(nk + 1e-6); }
                const double x = aa[t * VH + vh] + bf2f(dt_bias[vh]);
                const double gg = -std::exp(double(A_log[vh])) * (x <= 20 ? std::log1p(std::exp(x)) : x);
                const double beta = rb(1 / (1 + std::exp(-bb[t * VH + vh]))), decay = std::exp(gg);
                for (int cv_ = 0; cv_ < D; ++cv_) {
                    double sk = 0; for (int ck = 0; ck < D; ++ck) { S[cv_ * D + ck] *= decay; sk += S[cv_ * D + ck] * k[ck]; }
                    const double dl = (cv[t * CD + 2 * KD + vh * D + cv_] - sk) * beta; double so = 0;
                    for (int ck = 0; ck < D; ++ck) { S[cv_ * D + ck] += dl * k[ck]; so += S[cv_ * D + ck] * q[ck]; }
                    o[t * VD + vh * D + cv_] = rb(so / std::sqrt(double(D)));
                }
            }
        }
        std::vector<double> y(size_t(T) * VD);
        for (int64_t t = 0; t < T; ++t) for (int vh = 0; vh < VH; ++vh) {
            double ss = 0; for (int c = 0; c < D; ++c) ss += o[t * VD + vh * D + c] * o[t * VD + vh * D + c];
            const double r = 1 / std::sqrt(ss / D + 1e-6);
            for (int c = 0; c < D; ++c) { const size_t i = t * VD + vh * D + c; y[i] = rb(o[i] * r * bf2f(norm_w[c]) / (1 + std::exp(-z[i]))); }
        }
        for (int64_t t = 0; t < T; ++t) for (int hh = 0; hh < HID; ++hh) {
            double s = 0; for (int i = 0; i < VD; ++i) s += y[t * VD + i] * bf2f(Wout[size_t(hh) * VD + i]);
            ref[t * HID + hh] = s;
        }
    }
    const double e1 = rel(ref.data(), out1.data(), ref.size()), es = rel(Sref.data(), ssm1.data(), Sref.size());
    check("prefill vs FP64", e1 <= 2e-2 && es <= 2e-2, "100 tokens: out rel-L2 %.2e, ssm state %.2e (limit 2e-2)", e1, es);

    // --- 2. prefill 99 + decode 1 vs prefill 100
    St s2 = fresh(); fwd(s2, CERE_OP_PREFILL, 0, T - 1); fwd(s2, CERE_OP_DECODE, T - 1, 1);
    auto out2 = hostv(dout, size_t(T) * HID); auto conv2 = hostv(s2.conv, conv0.size()); auto ssm2 = hostv(s2.ssm, ssm0.size());
    std::vector<double> last1(HID), s1d(ssm1.size()); for (int i = 0; i < HID; ++i) last1[i] = bf2f(out1[(T - 1) * HID + i]);
    for (size_t i = 0; i < ssm1.size(); ++i) s1d[i] = bf2f(ssm1[i]);
    const double e2 = rel(last1.data(), out2.data() + (T - 1) * HID, HID), e2s = rel(s1d.data(), ssm2.data(), s1d.size());
    int64_t cdiff = 0;
    for (size_t i = 0; i < conv1.size(); ++i) if (conv2[i] != conv1[i]) ++cdiff;
    check("prefill 99 + decode 1", e2 <= 2e-2 && e2s <= 1e-2 && cdiff == 0, "last row out rel-L2 %.2e (2e-2), ssm %.2e (1e-2); conv state: %.0f entries differ",
          e2, e2s, double(cdiff));

    // --- 3a. EndState at row 64 on layer_core: the projections computed once, the mixer captured and resumed
    auto cap32 = kt::make(ctx, ssm0.size() * 4), capconv = kt::make(ctx, conv0.size() * 2);
    {
        auto mx = kt::make(ctx, size_t(T) * CD * 2), zz = kt::make(ctx, size_t(T) * VD * 2), bq = kt::make(ctx, T * VH * 2), aq = kt::make(ctx, T * VH * 2);
        auto y1 = kt::make(ctx, size_t(T) * VD * 2), y2 = kt::make(ctx, size_t(T) * VD * 2);
        // the projections as the HIP test runs them (pq / pz / pba .run(..., T, ..., 128): K2 at class 128, the GEMM)
        auto gws = kt::make(ctx, cere::vkk2::gemm_workspace_bytes(T, CD));
        const uint64_t x = dh.addr;
        auto proj = [&](uint64_t wt, uint64_t y, uint32_t n, const char *what) {
            ctx.begin();
            kt::must(cere::vkk2::run(ctx, x, wt, 0, y, T, n, HID, false, kCls, gws.addr), what);
            ctx.barrier();
            ctx.submit_wait();
        };
        proj(w.in_proj_qkv, mx.addr, CD, "run qkv"); proj(w.in_proj_z, zz.addr, VD, "run z");
        proj(w.in_proj_b, bq.addr, VH, "run b"); proj(w.in_proj_a, aq.addr, VH, "run a");
        ctx.free(gws);
        const size_t cws = k5::core_workspace_bytes(T); auto dcw = kt::make(ctx, cws);
        auto core = [&](const St &s, int64_t r0, int64_t n, uint64_t y, const k5::EndState *e) {
            ctx.begin();
            const cere_status st = k5::layer_core(ctx, w, k5::LayerState{s.conv.addr, s.ssm.addr}, CERE_OP_PREFILL, mx.addr + uint64_t(r0) * CD * 2,
                                                  zz.addr + uint64_t(r0) * VD * 2, bq.addr + uint64_t(r0) * VH * 2, aq.addr + uint64_t(r0) * VH * 2, n,
                                                  y + uint64_t(r0) * VD * 2, dcw.addr, cws, e);
            if (st != CERE_OK) { std::printf("layer_core failed: %d\n", st); std::exit(2); }
            ctx.barrier();
            ctx.submit_wait();
        };
        St c1 = fresh(); core(c1, 0, T, y1.addr, nullptr);
        auto yr = hostv(y1, size_t(T) * VD); auto cc1 = hostv(c1.conv, conv0.size()); auto cs1 = hostv(c1.ssm, ssm0.size());
        k5::EndState end{}; end.ssm_out32 = cap32.addr; end.conv_out = capconv.addr; end.row = 64;
        St c2 = fresh(); core(c2, 0, T, y2.addr, &end);
        const bool neutral = hostv(y2, size_t(T) * VD) == yr && hostv(c2.conv, conv0.size()) == cc1 && hostv(c2.ssm, ssm0.size()) == cs1;
        std::memset(y2.map, 0, size_t(T) * VD * 2);
        St c3{kt::upload(ctx, hostv(capconv, conv0.size())), kt::upload(ctx, ssm0)}; k5::EndState res{}; res.ssm_in32 = cap32.addr;
        core(c3, 64, T - 64, y2.addr, &res);
        auto yb = hostv(y2, size_t(T) * VD);
        bool cont = hostv(c3.conv, conv0.size()) == cc1 && hostv(c3.ssm, ssm0.size()) == cs1;
        for (size_t i = 64 * VD; i < yr.size(); ++i) cont &= yb[i] == yr[i];
        check("EndState (layer_core)", neutral && cont, "capture leaves the call unchanged (%.0f); rows 64..99, conv and ssm resumed from it bit-equal (%.0f)",
              neutral ? 1.0 : 0.0, cont ? 1.0 : 0.0);
    }
    // --- 3b. the same through layer_forward
    {
        k5::EndState end{}; end.ssm_out32 = cap32.addr; end.conv_out = capconv.addr; end.row = 64;
        St s3 = fresh(); fwd(s3, CERE_OP_PREFILL, 0, T, &end);
        const bool neutral = hostv(dout, size_t(T) * HID) == out1 && hostv(s3.ssm, ssm0.size()) == ssm1 && hostv(s3.conv, conv0.size()) == conv1;
        std::memset(dout.map, 0, size_t(T) * HID * 2);
        St s4{kt::upload(ctx, hostv(capconv, conv0.size())), kt::upload(ctx, ssm0)}; k5::EndState res{}; res.ssm_in32 = cap32.addr;
        fwd(s4, CERE_OP_PREFILL, 64, T - 64, &res);
        auto out4 = hostv(dout, size_t(T) * HID);
        int64_t diff = 0; for (size_t i = 64 * HID; i < out1.size(); ++i) diff += out4[i] != out1[i];
        std::vector<double> r64(out1.size() - 64 * HID); for (size_t i = 0; i < r64.size(); ++i) r64[i] = bf2f(out1[64 * HID + i]);
        bool cont = diff == 0 && hostv(s4.ssm, ssm0.size()) == ssm1 && hostv(s4.conv, conv0.size()) == conv1;
        std::printf("%-24s %s  capture neutral: %s; resumed rows 64..99: %ld of %zu outputs differ (rel-L2 %.2e), conv + ssm %s\n", "EndState (layer_forward)",
                    neutral && cont ? "PASS" : "FAIL", neutral ? "yes" : "NO", (long)diff, r64.size(), rel(r64.data(), out4.data() + 64 * HID, r64.size()),
                    cont ? "bit-equal" : "DIFFER");
        if (!neutral || !cont) ++g_fail;
    }
    {   // extra: prepared K2 weight info == the weights scanned in the call (PREFILL 100 at cls 128)
        auto info = kt::make(ctx, k5::weight_info_bytes(HID));
        k5::LayerWeights wp = w; wp.info = info.addr;
        ctx.begin(); kt::must(k5::prepare_weights(ctx, wp, HID), "prepare_weights"); ctx.barrier(); ctx.submit_wait();
        St sp = fresh();
        ctx.begin();
        kt::must(k5::layer_forward(ctx, wp, k5::LayerState{sp.conv.addr, sp.ssm.addr}, CERE_OP_PREFILL, dh.addr, T, HID, dout.addr, dws.addr, wsb, kCls),
                 "layer_forward");
        ctx.barrier(); ctx.submit_wait();
        const bool eq = hostv(dout, size_t(T) * HID) == out1 && hostv(sp.ssm, ssm0.size()) == ssm1 && hostv(sp.conv, conv0.size()) == conv1;
        check("prepared weight info", eq, eq ? "PREFILL 100 with LayerWeights::info: out, ssm and conv state bit-equal to the in-call scan" : "DIFFER");
        ctx.free(info);
    }
    {   // extra: refusals (nothing recorded)
        St s = fresh(); const k5::LayerState ls{s.conv.addr, s.ssm.addr};
        bool ok = true;
        ctx.begin();
        ok &= k5::layer_forward(ctx, w, ls, CERE_OP_VERIFY, dh.addr, 1, HID, dout.addr, dws.addr, wsb) == CERE_ERR_UNSUPPORTED;
        ok &= k5::layer_forward(ctx, w, ls, CERE_OP_DECODE, dh.addr, 2, HID, dout.addr, dws.addr, wsb) == CERE_ERR_ARGUMENT;
        ok &= k5::layer_forward(ctx, w, ls, CERE_OP_PREFILL, dh.addr, 0, HID, dout.addr, dws.addr, wsb) == CERE_ERR_ARGUMENT;
        ok &= k5::layer_forward(ctx, w, ls, CERE_OP_PREFILL, dh.addr, T, HID, dout.addr, dws.addr, wsb - 256) == CERE_ERR_ARGUMENT;
        ok &= k5::layer_forward(ctx, w, ls, CERE_OP_PREFILL, dh.addr, T, HID + 8, dout.addr, dws.addr, wsb) == CERE_ERR_ARGUMENT;
        k5::EndState bad{}; bad.ssm_out32 = cap32.addr; bad.conv_out = capconv.addr; bad.row = 32;  // off the 64 grid
        ok &= k5::layer_forward(ctx, w, ls, CERE_OP_PREFILL, dh.addr, T, HID, dout.addr, dws.addr, wsb, kCls, &bad) == CERE_ERR_ARGUMENT;
        bad.row = 128;  // not inside the call
        ok &= k5::layer_forward(ctx, w, ls, CERE_OP_PREFILL, dh.addr, T, HID, dout.addr, dws.addr, wsb, kCls, &bad) == CERE_ERR_ARGUMENT;
        k5::EndState dres{}; dres.ssm_in32 = cap32.addr;  // resume on a DECODE
        ok &= k5::layer_forward(ctx, w, ls, CERE_OP_DECODE, dh.addr, 1, HID, dout.addr, dws.addr, wsb, kCls, &dres) == CERE_ERR_ARGUMENT;
        ok &= k5::layer_forward(ctx, w, ls, CERE_OP_PREFILL, dh.addr, T, HID, dout.addr, dws.addr, wsb, uint32_t(T - 1)) == CERE_ERR_ARGUMENT;  // cls < tokens
        ok &= k5::layer_workspace_bytes(0) == 0 && k5::layer_workspace_bytes(4097) == 0;
        ctx.submit_wait();
        check("refusals (extra)", ok, ok ? "VERIFY: UNSUPPORTED; DECODE of 2, 0 tokens, small workspace, hidden %% 16, capture off-grid / outside, DECODE resume, cls < tokens: ARGUMENT"
                                          : "a bad call was accepted");
    }
    {   // MXFP8 projections (S3): == the BF16 upcast, bit for bit
        const MxW mq = mx_quantize(Wqkv, CD, HID), mz = mx_quantize(Wz, VD, HID), mb = mx_quantize(Wb, VH, HID), ma = mx_quantize(Wa, VH, HID),
                  mo = mx_quantize(Wout, HID, VD);
        auto uq = kt::upload(ctx, mq.q), usq = kt::upload(ctx, mq.s), uz = kt::upload(ctx, mz.q), usz = kt::upload(ctx, mz.s), ub = kt::upload(ctx, mb.q),
             usb = kt::upload(ctx, mb.s), ua = kt::upload(ctx, ma.q), usa = kt::upload(ctx, ma.s), uo = kt::upload(ctx, mo.q), uso = kt::upload(ctx, mo.s);
        auto pq = kt::upload(ctx, mq.up), pz = kt::upload(ctx, mz.up), pb = kt::upload(ctx, mb.up), pa = kt::upload(ctx, ma.up), po = kt::upload(ctx, mo.up);
        k5::LayerWeights wm = w;  // MXFP8: E4M3 bytes + scales
        wm.in_proj_qkv = uq.addr; wm.in_proj_qkv_scales = usq.addr; wm.in_proj_z = uz.addr; wm.in_proj_z_scales = usz.addr;
        wm.in_proj_b = ub.addr; wm.in_proj_b_scales = usb.addr; wm.in_proj_a = ua.addr; wm.in_proj_a_scales = usa.addr;
        wm.out_proj = uo.addr; wm.out_proj_scales = uso.addr;
        k5::LayerWeights wu = w;  // their exact BF16 upcast
        wu.in_proj_qkv = pq.addr; wu.in_proj_z = pz.addr; wu.in_proj_b = pb.addr; wu.in_proj_a = pa.addr; wu.out_proj = po.addr;
        k5::LayerWeights wx = wu;  // mixed: qkv and out_proj MXFP8, z / b / a BF16
        wx.in_proj_qkv = uq.addr; wx.in_proj_qkv_scales = usq.addr; wx.out_proj = uo.addr; wx.out_proj_scales = uso.addr;
        struct Res { std::vector<uint16_t> out, conv, ssm; };
        // calls: {op, first row, rows, cls} in one sequence from a fresh state; the result: out rows [0, T), final state
        auto seq = [&](const k5::LayerWeights &lw, std::initializer_list<std::array<int64_t, 4>> calls, const k5::EndState *end = nullptr,
                       const St *from = nullptr) {
            St s = from ? *from : fresh();
            std::memset(dout.map, 0, size_t(T) * HID * 2);
            for (const auto &c : calls) {
                ctx.begin();
                const cere_status st = k5::layer_forward(ctx, lw, k5::LayerState{s.conv.addr, s.ssm.addr}, uint32_t(c[0]), dh.addr + uint64_t(c[1]) * HID * 2, c[2],
                                                         HID, dout.addr + uint64_t(c[1]) * HID * 2, dws.addr, wsb, uint32_t(c[3]), end);
                if (st != CERE_OK) { std::printf("layer_forward (MXFP8 rows) failed: %d\n", st); std::exit(2); }
                ctx.barrier(); ctx.submit_wait();
            }
            return Res{hostv(dout, size_t(T) * HID), hostv(s.conv, conv0.size()), hostv(s.ssm, ssm0.size())};
        };
        auto same = [](const Res &a, const Res &b) { return a.out == b.out && a.conv == b.conv && a.ssm == b.ssm; };
        const Res gm = seq(wm, {{CERE_OP_PREFILL, 0, T, kCls}}), gu = seq(wu, {{CERE_OP_PREFILL, 0, T, kCls}});
        check("MXFP8 == upcast (GEMM)", same(gm, gu), "PREFILL 100, cls 128 (K2 GEMM, weights scanned in the call): out, conv, ssm bit-equal: %.0f", same(gm, gu) ? 1.0 : 0.0);
        const Res vm = seq(wm, {{CERE_OP_PREFILL, 0, 10, 0}, {CERE_OP_DECODE, 10, 1, 0}}), vu = seq(wu, {{CERE_OP_PREFILL, 0, 10, 0}, {CERE_OP_DECODE, 10, 1, 0}});
        check("MXFP8 == upcast (GEMV)", same(vm, vu), "PREFILL 10 + DECODE 1, cls 0 (K2 GEMV): out, conv, ssm bit-equal: %.0f", same(vm, vu) ? 1.0 : 0.0);
        const Res xm = seq(wx, {{CERE_OP_PREFILL, 0, T, kCls}}), xd = seq(wx, {{CERE_OP_PREFILL, 0, 10, 0}, {CERE_OP_DECODE, 10, 1, 0}});
        const bool mixed_ok = same(xm, gu) && same(xd, vu);
        check("MXFP8 mixed formats", mixed_ok, "qkv + out_proj MXFP8, z / b / a BF16 == all upcast: cls 128 PREFILL 100 and cls 0 PREFILL 10 + DECODE 1 bit-equal: %.0f",
              mixed_ok ? 1.0 : 0.0);
        {   // prepared weight info (scanned in the MXFP8 format) == the in-call scan
            auto info = kt::make(ctx, k5::weight_info_bytes(HID));
            k5::LayerWeights wp = wm; wp.info = info.addr;
            ctx.begin(); kt::must(k5::prepare_weights(ctx, wp, HID), "prepare_weights (MXFP8)"); ctx.barrier(); ctx.submit_wait();
            const Res pm = seq(wp, {{CERE_OP_PREFILL, 0, T, kCls}});
            check("MXFP8 prepared info", same(pm, gm), "PREFILL 100, cls 128 with the MXFP8 weights' prepared info == the in-call scan: %.0f", same(pm, gm) ? 1.0 : 0.0);
            ctx.free(info);
        }
        {   // EndState on MXFP8: capture at 64 neutral; resume rows 64..99 == the single call
            k5::EndState cap{}; cap.ssm_out32 = cap32.addr; cap.conv_out = capconv.addr; cap.row = 64;
            const Res cm = seq(wm, {{CERE_OP_PREFILL, 0, T, kCls}}, &cap);
            St r{kt::upload(ctx, hostv(capconv, conv0.size())), kt::upload(ctx, ssm0)};
            k5::EndState res{}; res.ssm_in32 = cap32.addr;
            const Res rm = seq(wm, {{CERE_OP_PREFILL, 64, T - 64, kCls}}, &res, &r);
            const bool cont = std::equal(rm.out.begin() + 64 * HID, rm.out.end(), gm.out.begin() + 64 * HID) && rm.conv == gm.conv && rm.ssm == gm.ssm;
            check("MXFP8 EndState", same(cm, gm) && cont, "capture at 64 neutral (%.0f); resumed rows 64..99, conv, ssm == the single call (%.0f)", same(cm, gm) ? 1.0 : 0.0,
                  cont ? 1.0 : 0.0);
        }
        {   // refusals: an MXFP8 in-projection needs hidden_size % 32 (K2's MXFP8 GEMV rule); prepare_weights k % 64
            St s = fresh();
            ctx.begin();
            const cere_status r1 = k5::layer_forward(ctx, wm, k5::LayerState{s.conv.addr, s.ssm.addr}, CERE_OP_DECODE, dh.addr, 1, HID + 16, dout.addr, dws.addr, wsb);
            const size_t pre = ctx.pipeline_count();
            auto info = kt::make(ctx, k5::weight_info_bytes(HID + 16));
            k5::LayerWeights wp = wm; wp.info = info.addr;
            const cere_status r2 = k5::prepare_weights(ctx, wp, HID + 16);
            ctx.submit_wait();
            ctx.free(info);
            const bool ok = r1 == CERE_ERR_ARGUMENT && r2 == CERE_ERR_ARGUMENT && ctx.pipeline_count() == pre;
            check("MXFP8 refusals", ok, "MXFP8 in-projections at hidden 272 (k %% 32 != 0): layer_forward %.0f; prepare_weights %.0f (ARGUMENT -1)", double(r1), double(r2));
        }
        for (auto *b : {&uq, &usq, &uz, &usz, &ub, &usb, &ua, &usa, &uo, &uso, &pq, &pz, &pb, &pa, &po}) ctx.free(*b);
    }
    {   // timing at the model's hidden size
        const int64_t H2 = 2560, TB = 4096;
        std::vector<std::vector<uint16_t>> hw;  // host copies of the projections (qkv, z, b, a, out) for the MXFP8 timing
        auto big = [&](size_t n, float s, bool keep = true) {
            std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s);
            auto b = kt::upload(ctx, v); if (keep) hw.push_back(std::move(v)); return b;
        };
        const float s2 = 1.f / std::sqrt(float(H2));
        auto tq = big(size_t(CD) * H2, s2), tz = big(size_t(VD) * H2, s2), tb = big(size_t(VH) * H2, s2), ta = big(size_t(VH) * H2, s2);
        auto to = big(size_t(H2) * VD, 1.f / std::sqrt(float(VD))), th = big(size_t(TB) * H2, 1.f, false);
        k5::LayerWeights w2{tq.addr, tz.addr, tb.addr, ta.addr, bconv.addr, bAlog.addr, bdt.addr, bnorm.addr, to.addr};
        auto winfo2 = kt::make(ctx, k5::weight_info_bytes(H2));  // the projections' K2 row info, prepared once (at load)
        w2.info = winfo2.addr;
        ctx.begin(); kt::must(k5::prepare_weights(ctx, w2, H2), "prepare_weights"); ctx.barrier(); ctx.submit_wait();
        const size_t wb2 = k5::layer_workspace_bytes(TB); auto dws2 = kt::make(ctx, wb2); auto tout = kt::make(ctx, size_t(TB) * H2 * 2);
        St st = fresh(); const k5::LayerState ls{st.conv.addr, st.ssm.addr};
        auto med = [&](std::vector<double> t) { std::sort(t.begin(), t.end()); return t[t.size() / 2]; };
        const int reps = 5;
        for (int64_t rows : {TB, int64_t(512)}) {
            ctx.begin();
            kt::must(k5::layer_forward(ctx, w2, ls, CERE_OP_PREFILL, th.addr, rows, H2, tout.addr, dws2.addr, wb2), "layer_forward"); ctx.barrier();
            for (int i = 0; i < reps; ++i) {
                ctx.stamp(2 * i);
                kt::must(k5::layer_forward(ctx, w2, ls, CERE_OP_PREFILL, th.addr, rows, H2, tout.addr, dws2.addr, wb2), "layer_forward");
                ctx.stamp(2 * i + 1); ctx.barrier();
            }
            ctx.submit_wait();
            std::vector<double> t; for (int i = 0; i < reps; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
            std::printf("%-24s       layer_forward PREFILL %4ld rows, hidden %ld: %.2f ms (median of %d)\n", "timing", long(rows), long(H2), med(t), reps);
        }
        {   // the 4,096-row split: in-projections (K2 GEMM, as layer_forward: prepared info, one workspace each) / core / out_proj
            const size_t cws = k5::core_workspace_bytes(TB); auto dcw = kt::make(ctx, cws);
            auto mx = kt::make(ctx, size_t(TB) * CD * 2), zz = kt::make(ctx, size_t(TB) * VD * 2), bq = kt::make(ctx, TB * VH * 2), aq = kt::make(ctx, TB * VH * 2);
            std::vector<cere::vk::Buffer> gw;
            for (int i = 0; i < 4; ++i) gw.push_back(kt::make(ctx, cere::vkk2::gemm_workspace_bytes(TB, CD)));
            // LayerWeights::info's parts (k5_gdn_layer_vk.cpp info_parts): qkv, z, b, a, out, 16-byte aligned
            const uint64_t iq = w2.info, iz = iq + CD * 4, ib = iz + VD * 4, ia = ib + 192, io = ia + 192;
            ctx.begin();
            for (int i = 0; i < 2; ++i) {
                ctx.stamp(6 * i);
                kt::must(cere::vkk2::run(ctx, th.addr, w2.in_proj_qkv, 0, mx.addr, TB, CD, H2, false, 0, gw[0].addr, iq), "run");
                kt::must(cere::vkk2::run(ctx, th.addr, w2.in_proj_z, 0, zz.addr, TB, VD, H2, false, 0, gw[1].addr, iz), "run");
                kt::must(cere::vkk2::run(ctx, th.addr, w2.in_proj_b, 0, bq.addr, TB, VH, H2, false, 0, gw[2].addr, ib), "run");
                kt::must(cere::vkk2::run(ctx, th.addr, w2.in_proj_a, 0, aq.addr, TB, VH, H2, false, 0, gw[3].addr, ia), "run");
                ctx.barrier();
                ctx.stamp(6 * i + 1);
                ctx.stamp(6 * i + 2);
                kt::must(k5::layer_core(ctx, w2, ls, CERE_OP_PREFILL, mx.addr, zz.addr, bq.addr, aq.addr, TB, mx.addr, dcw.addr, cws), "layer_core");
                ctx.barrier();
                ctx.stamp(6 * i + 3);
                ctx.stamp(6 * i + 4);
                kt::must(cere::vkk2::run(ctx, mx.addr, w2.out_proj, 0, tout.addr, TB, uint32_t(H2), VD, false, 0, gw[0].addr, io), "run");
                ctx.barrier();
                ctx.stamp(6 * i + 5);
            }
            ctx.submit_wait();
            std::printf("%-24s       4,096-row split (2nd of 2): in-projections %.2f ms, mixer core %.2f ms, out_proj %.2f ms\n", "timing",
                        ctx.stamp_ms(6, 7), ctx.stamp_ms(8, 9), ctx.stamp_ms(10, 11));
        }
        {   // the decode handoff: one DECODE token after a prefill
            ctx.begin();
            kt::must(k5::layer_forward(ctx, w2, ls, CERE_OP_PREFILL, th.addr, 99, H2, tout.addr, dws2.addr, wb2), "prefill"); ctx.barrier();
            for (int i = 0; i < 3; ++i) { kt::must(k5::layer_forward(ctx, w2, ls, CERE_OP_DECODE, th.addr, 1, H2, tout.addr, dws2.addr, wb2), "decode"); ctx.barrier(); }
            const int dr = 15;
            for (int i = 0; i < dr; ++i) {
                ctx.stamp(2 * i);
                kt::must(k5::layer_forward(ctx, w2, ls, CERE_OP_DECODE, th.addr, 1, H2, tout.addr, dws2.addr, wb2), "decode");
                ctx.stamp(2 * i + 1); ctx.barrier();
            }
            ctx.submit_wait();
            std::vector<double> t; for (int i = 0; i < dr; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
            std::printf("%-24s       layer_forward DECODE 1 token after a prefill, hidden %ld: %.3f ms (median of %d)\n", "timing", long(H2), med(t), dr);
        }
        {   // MXFP8 projections (the S3 format): the same calls on the same weights MXFP8-quantized, prepared info
            const MxW q0 = mx_quantize(hw[0], CD, H2), q1 = mx_quantize(hw[1], VD, H2), q2 = mx_quantize(hw[2], VH, H2), q3 = mx_quantize(hw[3], VH, H2),
                      q4 = mx_quantize(hw[4], uint32_t(H2), VD);
            std::vector<cere::vk::Buffer> mb;
            auto upm = [&](const MxW &m, uint64_t &wq, uint64_t &ws_) { mb.push_back(kt::upload(ctx, m.q)); wq = mb.back().addr; mb.push_back(kt::upload(ctx, m.s)); ws_ = mb.back().addr; };
            k5::LayerWeights wm2 = w2;
            upm(q0, wm2.in_proj_qkv, wm2.in_proj_qkv_scales); upm(q1, wm2.in_proj_z, wm2.in_proj_z_scales); upm(q2, wm2.in_proj_b, wm2.in_proj_b_scales);
            upm(q3, wm2.in_proj_a, wm2.in_proj_a_scales); upm(q4, wm2.out_proj, wm2.out_proj_scales);
            auto minfo = kt::make(ctx, k5::weight_info_bytes(H2));
            wm2.info = minfo.addr;
            ctx.begin(); kt::must(k5::prepare_weights(ctx, wm2, H2), "prepare_weights (MXFP8)"); ctx.barrier(); ctx.submit_wait();
            for (int64_t rows : {TB, int64_t(512)}) {
                ctx.begin();
                kt::must(k5::layer_forward(ctx, wm2, ls, CERE_OP_PREFILL, th.addr, rows, H2, tout.addr, dws2.addr, wb2), "layer_forward"); ctx.barrier();
                for (int i = 0; i < reps; ++i) {
                    ctx.stamp(2 * i);
                    kt::must(k5::layer_forward(ctx, wm2, ls, CERE_OP_PREFILL, th.addr, rows, H2, tout.addr, dws2.addr, wb2), "layer_forward");
                    ctx.stamp(2 * i + 1); ctx.barrier();
                }
                ctx.submit_wait();
                std::vector<double> t; for (int i = 0; i < reps; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
                std::printf("%-24s       MXFP8 layer_forward PREFILL %4ld rows, hidden %ld: %.2f ms (median of %d)\n", "timing", long(rows), long(H2), med(t), reps);
            }
            ctx.begin();
            kt::must(k5::layer_forward(ctx, wm2, ls, CERE_OP_PREFILL, th.addr, 99, H2, tout.addr, dws2.addr, wb2), "prefill"); ctx.barrier();
            for (int i = 0; i < 3; ++i) { kt::must(k5::layer_forward(ctx, wm2, ls, CERE_OP_DECODE, th.addr, 1, H2, tout.addr, dws2.addr, wb2), "decode"); ctx.barrier(); }
            const int dr = 15;
            for (int i = 0; i < dr; ++i) {
                ctx.stamp(2 * i);
                kt::must(k5::layer_forward(ctx, wm2, ls, CERE_OP_DECODE, th.addr, 1, H2, tout.addr, dws2.addr, wb2), "decode");
                ctx.stamp(2 * i + 1); ctx.barrier();
            }
            ctx.submit_wait();
            std::vector<double> t; for (int i = 0; i < dr; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
            std::printf("%-24s       MXFP8 layer_forward DECODE 1 token after a prefill, hidden %ld: %.3f ms (median of %d)\n", "timing", long(H2), med(t), dr);
            for (auto &b : mb) ctx.free(b);
            ctx.free(minfo);
        }
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
