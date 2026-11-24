// Inkling S11 G1 on Vulkan: the Inkling kernels (shaders/ki_*.comp, src/kernels/ki_*_vk.cpp) bit-equal to their CPU
// references (executor/ref/inkling/*_ref.h) on random inputs. Usage: ki_test SPV_DIR [--rows NAME,..] [--quick].
// Every row runs twice and must be bit-identical run to run. Exit 0 when every row passes, 1 on a failed row.
// On a development host without the GPU: CERE_VK_ALLOW_SOFTWARE=1 (lavapipe); the gate is RADV on amd.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "ktest_util.h"
#include "launch/vk_launch.h"
#include "launch/vk_launch_testing.h"
#include "kernels/ki_ops_vk.h"
#include "ref/inkling/common_ref.h"
#include "ref/inkling/attn_ref.h"
#include "ref/inkling/conv_ref.h"
#include "ref/inkling/depth_ref.h"
#include "ref/inkling/embed_ref.h"
#include "ref/inkling/moe_ref.h"
#include "ref/inkling/msg_ref.h"
#include "ref/inkling/norm_ref.h"
#include "ref/inkling/quant_ref.h"
#include "ref/inkling/router_ref.h"
#include "ref/inkling/softcap_ref.h"

using namespace cere;

static int g_fail = 0;
static std::string g_only;
static bool g_quick = false;

static void report(const std::string &name, bool ok, const std::string &detail) {
    std::printf("%-40s %s  %s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.c_str());
    std::fflush(stdout);
    if (!ok) ++g_fail;
}
static bool selected(const char *row) {
    if (g_only.empty()) return true;
    return ("," + g_only + ",").find("," + std::string(row) + ",") != std::string::npos;
}
static uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float bitsf(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
// Bit equality of two float vectors (any NaN equals any NaN: payloads are not part of the contract, the GPU returns
// the canonical NaN); *first = the first differing index.
static size_t ndiff(const std::vector<float> &a, const std::vector<float> &b, size_t *first) {
    size_t n = 0;
    *first = SIZE_MAX;
    for (size_t i = 0; i < a.size(); ++i)
        if (fbits(a[i]) != fbits(b[i]) && !(std::isnan(a[i]) && std::isnan(b[i]))) { if (n++ == 0) *first = i; }
    return n;
}

// ---- row "common": ki_common.comp's exact helpers and canonical reductions ----
static float ref_bf_round(float f) {
    uint32_t u = fbits(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return bitsf(((u >> 16) | 0x40u) << 16);
    return bitsf(((u + 0x7fffu + ((u >> 16) & 1u)) >> 16) << 16);
}
// ref::reduce_*<L> (common_ref.h contract): lane t accumulates t, t+L, ...; halving tree; lane 0.
static void ref_reduce(const float *x, const float *y, uint32_t n, uint32_t L, float *s, float *d, float *m) {
    std::vector<float> ls(L, 0.f), ld(L, 0.f), lm(L, -INFINITY);
    for (uint32_t t = 0; t < L; ++t)
        for (uint32_t i = t; i < n; i += L) {
            ls[t] = ls[t] + x[i];
            ld[t] = std::fma(x[i], y[i], ld[t]);
            lm[t] = std::max(lm[t], x[i]);
        }
    for (uint32_t o = L / 2; o > 0; o /= 2)
        for (uint32_t t = 0; t < o; ++t) {
            ls[t] = ls[t] + ls[t + o];
            ld[t] = ld[t] + ld[t + o];
            lm[t] = std::max(lm[t], lm[t + o]);
        }
    *s = ls[0], *d = ld[0], *m = lm[0];
}

static void check_twice(vk::Context &ctx, const std::string &name, vk::Buffer &bo, size_t n, bool bf, const std::vector<float> &want,
                        const std::function<void()> &launch);
static void row_common(vk::Context &ctx) {
    struct Push { uint64_t x, y, out; uint32_t n, rows; };
    std::mt19937 g(1234);
    // Elementwise: random bit patterns (all classes incl. denormals) plus normal-range values.
    const uint32_t n = g_quick ? (1u << 16) : (1u << 22);
    std::vector<float> x(n), y(n);
    std::uniform_int_distribution<uint32_t> bits;
    std::normal_distribution<float> nd(0.f, 3.f);
    for (uint32_t i = 0; i < n; ++i) {
        if (i % 4 == 0) { x[i] = bitsf(bits(g) & 0x7fffffffu); y[i] = bitsf(bits(g)); }
        else if (i % 4 == 1) { x[i] = bitsf(bits(g) & 0x007fffffu); y[i] = bitsf(bits(g) & 0x807fffffu); }  // denormals
        else { x[i] = std::fabs(nd(g)) * std::exp2(float(int(g() % 40) - 20)); y[i] = nd(g); }
    }
    vk::Buffer bx = kt::upload(ctx, x), by = kt::upload(ctx, y), bo = kt::make(ctx, uint64_t(n) * 16);
    std::vector<float> out[2];
    for (int run = 0; run < 2; ++run) {
        std::memset(bo.map, 0xff, uint64_t(n) * 16);
        ctx.begin();
        const Push p{bx.addr, by.addr, bo.addr, n, 0};
        ctx.dispatch(ctx.pipeline("ki_selftest_e.spv", sizeof p, 0), &p, (n + 255) / 256);
        ctx.barrier();
        ctx.submit_wait();
        out[run] = kt::host<float>(bo, size_t(n) * 4);
    }
    size_t bad[4] = {0, 0, 0, 0}, first[4] = {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
    for (uint32_t i = 0; i < n; ++i) {
        const float sq = std::sqrt(x[i]);
        const float want[4] = {sq, x[i] / y[i], 1.0f / sq, ref_bf_round(y[i])};
        for (int k = 0; k < 4; ++k) {
            const float got = out[0][size_t(i) * 4 + k];
            const bool nan_ok = std::isnan(want[k]) && std::isnan(got);
            if (!nan_ok && fbits(got) != fbits(want[k])) { if (bad[k]++ == 0) first[k] = i; }
        }
    }
    size_t f;
    const bool same = ndiff(out[0], out[1], &f) == 0;
    const char *names[4] = {"common.sqrt_rn", "common.div_rn", "common.rsqrt", "common.bf16_rne"};
    for (int k = 0; k < 4; ++k) {
        char d[256];
        if (bad[k] == 0) std::snprintf(d, sizeof d, "%u values bit-equal, run-to-run %s", n, same ? "identical" : "DIFFERS");
        else {
            const uint32_t i = uint32_t(first[k]);
            std::snprintf(d, sizeof d, "%zu / %u differ; first i %u x %08x y %08x got %08x", bad[k], n, i, fbits(x[i]), fbits(y[i]),
                          fbits(out[0][size_t(i) * 4 + k]));
        }
        report(names[k], bad[k] == 0 && same, d);
    }
    ctx.free(bx); ctx.free(by); ctx.free(bo);

    // fma: mixed signs, near-cancelling addends (c ~ -a*b), exact-product ties, random bit patterns.
    {
        std::vector<float> ab(size_t(n) * 2), c(n);
        std::uniform_real_distribution<float> ud(-1.f, 1.f);
        for (uint32_t i = 0; i < n; ++i) {
            float &a = ab[2 * size_t(i)], &b = ab[2 * size_t(i) + 1];
            switch (i % 5) {
            case 0: a = bitsf(bits(g)); b = bitsf(bits(g)); c[i] = bitsf(bits(g)); break;
            case 1: a = nd(g); b = nd(g); c[i] = -a * b * (1.f + ud(g) * 1e-6f); break;
            case 2: a = float(int(g() % 31) - 15) * 0.125f; b = ref::decode_e4m3(uint8_t(g() % 0x7f)); c[i] = nd(g); break;
            case 3: a = nd(g); b = nd(g) * 1e-8f; c[i] = nd(g) * 1e3f; break;
            default: a = nd(g) * std::exp2(float(int(g() % 60) - 30)); b = nd(g); c[i] = nd(g) * std::exp2(float(int(g() % 60) - 30));
            }
        }
        vk::Buffer bab = kt::upload(ctx, ab), bc = kt::upload(ctx, c), bf = kt::make(ctx, uint64_t(n) * 4);
        std::vector<float> want(n);
        for (uint32_t i = 0; i < n; ++i) want[i] = std::fma(ab[2 * size_t(i)], ab[2 * size_t(i) + 1], c[i]);
        check_twice(ctx, "common.fma_rn", bf, n, false, want, [&] {
            const Push p{bab.addr, bc.addr, bf.addr, n, 0};
            ctx.dispatch(ctx.pipeline("ki_selftest_f.spv", sizeof p, 0), &p, (n + 255) / 256);
        });
        ctx.free(bab); ctx.free(bc); ctx.free(bf);
    }

    // Reductions: rows of several lengths, L = 256.
    for (uint32_t len : {1u, 255u, 256u, 2304u, 4608u, 9999u}) {
        const uint32_t rows = 64;
        std::vector<float> a(size_t(rows) * len), b(size_t(rows) * len);
        for (auto &v : a) v = nd(g) * std::exp2(float(int(g() % 16) - 8));
        for (auto &v : b) v = nd(g);
        vk::Buffer ba = kt::upload(ctx, a), bb = kt::upload(ctx, b), br = kt::make(ctx, uint64_t(rows) * 12);
        ctx.begin();
        const Push p{ba.addr, bb.addr, br.addr, len, rows};
        ctx.dispatch(ctx.pipeline("ki_selftest_r.spv", sizeof p, 0), &p, rows);
        ctx.barrier();
        ctx.submit_wait();
        const std::vector<float> got = kt::host<float>(br, size_t(rows) * 3);
        std::vector<float> want(size_t(rows) * 3);
        for (uint32_t r = 0; r < rows; ++r)
            ref_reduce(&a[size_t(r) * len], &b[size_t(r) * len], len, 256, &want[r * 3], &want[r * 3 + 1], &want[r * 3 + 2]);
        const size_t nd_ = ndiff(got, want, &f);
        size_t per[3] = {0, 0, 0};
        for (size_t i = 0; i < got.size(); ++i) per[i % 3] += fbits(got[i]) != fbits(want[i]);
        char d[200];
        std::snprintf(d, sizeof d, "%u rows x %u: sum/dot/max %s (differ %zu/%zu/%zu)", rows, len, nd_ == 0 ? "bit-equal" : "DIFFER",
                      per[0], per[1], per[2]);
        report("common.reduce n=" + std::to_string(len), nd_ == 0, d);
        ctx.free(ba); ctx.free(bb); ctx.free(br);
    }
}

// ---- row "trans": ki_exp / ki_log / ki_tanh / ki_sigmoid bit-equal to common_ref.h ref::exp / log / tanh / sigmoid ----
static void row_trans(vk::Context &ctx) {
    struct Push { uint64_t x, y, out; uint32_t n, rows; };
    std::mt19937 g(4321);
    const uint32_t n = g_quick ? (1u << 16) : (1u << 22);
    std::vector<float> x(n);
    std::uniform_int_distribution<uint32_t> bits;
    std::uniform_real_distribution<float> ud(-110.f, 110.f);
    std::normal_distribution<float> nd(0.f, 2.f);
    for (uint32_t i = 0; i < n; ++i) {
        if (i % 4 == 0) x[i] = bitsf(bits(g));            // every class: NaN, inf, denormal, huge
        else if (i % 4 == 1) x[i] = ud(g);                 // exp's whole range incl. the subnormal tail
        else x[i] = nd(g) * std::exp2(float(int(g() % 12) - 6));
    }
    vk::Buffer bx = kt::upload(ctx, x), bo = kt::make(ctx, uint64_t(n) * 16);
    std::vector<float> out[2];
    for (int run = 0; run < 2; ++run) {
        std::memset(bo.map, 0xff, uint64_t(n) * 16);
        ctx.begin();
        const Push p{bx.addr, bx.addr, bo.addr, n, 0};
        ctx.dispatch(ctx.pipeline("ki_selftest_t.spv", sizeof p, 0), &p, (n + 255) / 256);
        ctx.barrier();
        ctx.submit_wait();
        out[run] = kt::host<float>(bo, size_t(n) * 4);
    }
    size_t bad[4] = {0, 0, 0, 0}, first[4] = {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
    for (uint32_t i = 0; i < n; ++i) {
        const float want[4] = {ref::exp(x[i]), ref::log(x[i]), ref::tanh(x[i]), ref::sigmoid(x[i])};
        for (int k = 0; k < 4; ++k) {
            const float got = out[0][size_t(i) * 4 + k];
            const bool nan_ok = std::isnan(want[k]) && std::isnan(got);
            if (!nan_ok && fbits(got) != fbits(want[k])) { if (bad[k]++ == 0) first[k] = i; }
        }
    }
    size_t f;
    const bool same = ndiff(out[0], out[1], &f) == 0;
    const char *names[4] = {"trans.exp", "trans.log", "trans.tanh", "trans.sigmoid"};
    for (int k = 0; k < 4; ++k) {
        char d[256];
        if (bad[k] == 0) std::snprintf(d, sizeof d, "%u values bit-equal, run-to-run %s", n, same ? "identical" : "DIFFERS");
        else {
            const uint32_t i = uint32_t(first[k]);
            std::snprintf(d, sizeof d, "%zu / %u differ; first i %u x %08x got %08x want %08x", bad[k], n, i, fbits(x[i]),
                          fbits(out[0][size_t(i) * 4 + k]), fbits(k == 0 ? ref::exp(x[i]) : k == 1 ? ref::log(x[i]) : k == 2 ? ref::tanh(x[i]) : ref::sigmoid(x[i])));
        }
        report(names[k], bad[k] == 0 && same, d);
    }
    ctx.free(bx); ctx.free(bo);
}

// ---- shared row helpers ----
// Values stored as BF16 when bf (the vector then holds the BF16-rounded values, so references see what the GPU reads).
static vk::Buffer upload_as(vk::Context &ctx, std::vector<float> &v, bool bf) {
    if (!bf) return kt::upload(ctx, v);
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) { h[i] = uint16_t(fbits(ref_bf_round(v[i])) >> 16); v[i] = ref_bf_round(v[i]); }
    return kt::upload(ctx, h);
}
static std::vector<float> read_as(const vk::Buffer &b, size_t n, bool bf) {
    if (!bf) return kt::host<float>(b, n);
    const std::vector<uint16_t> h = kt::host<uint16_t>(b, n);
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) f[i] = bitsf(uint32_t(h[i]) << 16);
    return f;
}
// Dispatch `launch` twice into a fresh output, compare with `want` and run to run, report.
static void check_twice(vk::Context &ctx, const std::string &name, vk::Buffer &bo, size_t n, bool bf, const std::vector<float> &want,
                        const std::function<void()> &launch) {
    std::vector<float> got[2];
    for (int run = 0; run < 2; ++run) {
        std::memset(bo.map, 0xa5, n * (bf ? 2 : 4));
        ctx.begin();
        launch();
        ctx.barrier();
        ctx.submit_wait();
        got[run] = read_as(bo, n, bf);
    }
    size_t f = 0, fr = 0;
    const size_t bad = ndiff(got[0], want, &f), rr = ndiff(got[0], got[1], &fr);
    char d[256];
    if (bad == 0) std::snprintf(d, sizeof d, "%zu values bit-equal, run-to-run %s", n, rr == 0 ? "identical" : "DIFFERS");
    else std::snprintf(d, sizeof d, "%zu / %zu differ; first i %zu got %08x want %08x", bad, n, f, fbits(got[0][f]), fbits(want[f]));
    report(name, bad == 0 && rr == 0, d);
}
static float cast_out(float v, bool bf) { return bf ? ref_bf_round(v) : v; }

// ---- row "norm": ki_rownorm.comp. RMSNorm against executor/ref/inkling/norm_ref.h; the L2 clips (S2's attention refs,
// not landed yet) against an interim reference restating the agreed contract. ref::reduce_dot<L>(y, y) of one row:
static float ref_dot_self(const std::vector<float> &y, uint32_t L) {
    std::vector<float> l(L, 0.f);
    for (uint32_t t = 0; t < L; ++t)
        for (size_t i = t; i < y.size(); i += L) l[t] = std::fma(y[i], y[i], l[t]);
    for (uint32_t o = L / 2; o > 0; o /= 2)
        for (uint32_t t = 0; t < o; ++t) l[t] = l[t] + l[t + o];
    return l[0];
}
struct NormCase {
    const char *name;
    uint32_t mode, n, L, rows, heads;
    bool x_bf, o_bf, w_bf, has_w, has_tau;
    float eps, thr, xscale;
};
static void row_norm(vk::Context &ctx) {
    std::mt19937 g(77);
    std::normal_distribution<float> nd(0.f, 1.f);
    const NormCase cases[] = {
        {"norm.rms_w_2304_bf16", 0, 2304, 256, 37, 1, true, true, true, true, false, 1e-6f, 0, 3.f},
        {"norm.rms_unw_2304_f32", 0, 2304, 256, 37, 1, false, false, false, false, false, 1e-6f, 0, 0.01f},
        {"norm.rms_w_2304_f32in_bf16out", 0, 2304, 256, 5, 1, false, true, true, true, false, 1e-6f, 0, 40.f},
        {"norm.v_rms_128", 0, 128, 32, 8 * 37, 8, true, true, false, false, false, 1e-6f, 0, 2.f},
        {"norm.k_l2clip_128", 1, 128, 32, 8 * 37, 8, true, true, false, false, false, 0, 22.627416997969522f, 2.5f},
        {"norm.q_l2clip_scaled_128", 1, 128, 32, 24 * 37, 24, true, true, false, true, false, 0, 22.627416997969522f, 2.5f},
        {"norm.q_l2clip_logscaled_128", 1, 128, 32, 24 * 37, 24, true, true, false, true, true, 0, 22.627416997969522f, 2.5f},
        {"norm.rel_l2clip_16", 1, 16, 16, 24 * 37, 24, true, true, true, true, true, 0, 8.0f, 2.5f},
    };
    for (const NormCase &c : cases) {
        const size_t n = size_t(c.rows) * c.n;
        std::vector<float> x(n), w(c.mode == 0 ? c.n : size_t(c.heads) * c.n), tau(c.rows / c.heads);
        for (auto &v : x) v = nd(g) * c.xscale;
        for (auto &v : w) v = 1.f + 0.3f * nd(g);
        for (size_t i = 0; i < tau.size(); ++i) tau[i] = 1.f + 0.1f * std::log(std::max(float(i * 997 + 1) / 8192.f, 1.f));
        vk::Buffer bx = upload_as(ctx, x, c.x_bf), bw = upload_as(ctx, w, c.w_bf), bt = kt::upload(ctx, tau);
        vk::Buffer bo = kt::make(ctx, n * 4);
        std::vector<float> want(n);
        for (uint32_t r = 0; r < c.rows; ++r) {
            std::vector<float> y(x.begin() + size_t(r) * c.n, x.begin() + size_t(r + 1) * c.n);
            const float *wr = c.mode == 0 ? w.data() : &w[size_t(r % c.heads) * c.n];
            if (c.mode == 0) {
                std::vector<uint16_t> wb(c.n);
                for (uint32_t i = 0; i < c.n; ++i) wb[i] = uint16_t(fbits(wr[i]) >> 16);  // w_bf: already BF16-exact
                const uint16_t *wp = c.has_w ? wb.data() : nullptr;
                if (c.L == 256) ref::rms_norm<256>(y.data(), int(c.n), c.eps, wp, y.data());
                else ref::rms_norm<32>(y.data(), int(c.n), c.eps, wp, y.data());
                for (uint32_t i = 0; i < c.n; ++i) want[size_t(r) * c.n + i] = cast_out(y[i], c.o_bf);
            } else {
                if (c.has_w) for (uint32_t i = 0; i < c.n; ++i) y[i] = y[i] * wr[i];
                const float den = std::sqrt(ref_dot_self(y, c.L)) + 1.0e-6f;
                const float q = c.thr / den;
                const float f = q > 1.f ? 1.f : q;
                for (uint32_t i = 0; i < c.n; ++i) {
                    float v = y[i] * f;
                    if (c.has_tau) v = ref_bf_round(v) * tau[r / c.heads];
                    want[size_t(r) * c.n + i] = cast_out(v, c.o_bf);
                }
            }
        }
        const vki::Tensor tx{bx.addr, c.x_bf ? vki::Dt::kBF16 : vki::Dt::kF32}, to{bo.addr, c.o_bf ? vki::Dt::kBF16 : vki::Dt::kF32};
        const vki::Tensor tw{c.has_w ? bw.addr : 0, c.w_bf ? vki::Dt::kBF16 : vki::Dt::kF32};
        check_twice(ctx, c.name, bo, n, c.o_bf, want, [&] {
            const cere_status st = c.mode == 0 ? vki::rms_norm(ctx, tx, tw, to, c.n, c.rows, c.eps)
                                               : vki::l2_clip(ctx, tx, tw, c.has_tau ? bt.addr : 0, to, c.n, c.rows, c.heads, c.thr);
            if (st != CERE_OK) report(c.name, false, "launch refused");
        });
        ctx.free(bx); ctx.free(bw); ctx.free(bt); ctx.free(bo);
    }
}

// ---- row "conv": ki_conv.comp static and dynamic short convolutions against executor/ref/inkling/conv_ref.h ----
static void row_conv(vk::Context &ctx) {
    std::mt19937 g(99);
    std::normal_distribution<float> nd(0.f, 1.f);
    struct Case { const char *name; uint32_t mode, T, C, heads; };
    const Case cases[] = {
        {"conv.static_2304_decode", 0, 1, 2304, 1},    {"conv.static_2304_T2", 0, 2, 2304, 1},
        {"conv.static_1024_prefill", 0, 77, 1024, 1}, {"conv.dynamic_1024_decode", 1, 1, 1024, 8},
        {"conv.dynamic_1024_prefill", 1, 61, 1024, 8},
    };
    auto bf_bits = [](const std::vector<float> &v) {
        std::vector<uint16_t> h(v.size());
        for (size_t i = 0; i < v.size(); ++i) h[i] = uint16_t(fbits(v[i]) >> 16);
        return h;
    };
    for (const Case &c : cases) {
        const size_t n = size_t(c.T) * c.C;
        std::vector<float> x(n), w(size_t(c.C) * 4), dyn(size_t(c.T) * c.heads * 4), st(size_t(c.C) * 3);
        for (auto &v : x) v = nd(g);
        for (auto &v : w) v = 0.5f * nd(g);
        for (auto &v : dyn) v = 0.3f * nd(g);
        for (auto &v : st) v = nd(g);
        vk::Buffer bx = upload_as(ctx, x, true), bw = upload_as(ctx, w, true), bd = upload_as(ctx, dyn, true);  // BF16-rounds
        vk::Buffer bs = kt::upload(ctx, st), bs2 = kt::make(ctx, st.size() * 4), bo = kt::make(ctx, n * 4);
        std::vector<float> want_st = st, want(n);
        std::vector<uint16_t> y(n);
        const std::vector<uint16_t> xh = bf_bits(x), wh = bf_bits(w), dh = bf_bits(dyn);
        if (c.mode == 0) ref::conv_static_seq(xh.data(), c.C, int(c.T), int(c.C), wh.data(), want_st.data(), y.data(), c.C);
        else ref::conv_dynamic_seq(xh.data(), c.C, int(c.T), int(c.C), wh.data(), dh.data(), c.heads * 4, want_st.data(), y.data(), c.C);
        for (size_t i = 0; i < n; ++i) want[i] = ref::decode_bf16(y[i]);
        const vki::Dt xd = vki::Dt::kBF16;
        check_twice(ctx, c.name, bo, n, true, want, [&] {
            const cere_status st =
                c.mode == 0 ? vki::conv_static(ctx, {bx.addr, xd}, {bw.addr, xd}, {bo.addr, xd}, bs.addr, bs2.addr, c.T, c.C)
                            : vki::conv_dynamic(ctx, {bx.addr, xd}, {bw.addr, xd}, {bd.addr, xd}, {bo.addr, xd}, bs.addr, bs2.addr, c.T,
                                                c.C, c.heads);
            if (st != CERE_OK) report(c.name, false, "launch refused");
        });
        size_t f = 0;
        const size_t bad = ndiff(kt::host<float>(bs2, st.size()), want_st, &f);
        report(std::string(c.name) + ".state", bad == 0, bad == 0 ? "next state bit-equal" : "next state DIFFERS");
        ctx.free(bx); ctx.free(bw); ctx.free(bd); ctx.free(bs); ctx.free(bs2); ctx.free(bo);
    }
}

// ---- row "quant": ki_quant.comp MXFP8 / NVFP4 fouroversix_mae activation quantizers against quant_ref.h (InkS4) ----
static void row_quant(vk::Context &ctx) {
    std::mt19937 g(777);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (uint32_t K : {2304u, 1152u})
        for (int bfin : {1, 0}) {
            const uint32_t rows = g_quick ? 16 : 256;
            std::vector<float> x(size_t(rows) * K);
            for (uint32_t r = 0; r < rows; ++r) {
                const float sc = r == 0 ? 0.f : r == 1 ? 1e-30f : std::exp2(float(int(g() % 30) - 15));
                for (uint32_t i = 0; i < K; ++i) {
                    float v = nd(g) * sc;
                    if (g() % 97 == 0) v *= 40.f;  // outliers: blocks with a large dynamic range
                    if (r == 2 && i % 7 == 0) v = 0.f;
                    x[size_t(r) * K + i] = v;
                }
            }
            vk::Buffer bx = upload_as(ctx, x, bfin != 0);
            const vki::Tensor tx{bx.addr, bfin ? vki::Dt::kBF16 : vki::Dt::kF32};
            const std::string tag = std::to_string(K) + (bfin ? "_bf16" : "_f32");
            {  // MXFP8
                std::vector<uint8_t> wq(size_t(rows) * K), ws(size_t(rows) * K / 32);
                for (uint32_t r = 0; r < rows; ++r)
                    ref::mxfp8_quantize_row(&x[size_t(r) * K], int(K), &wq[size_t(r) * K], &ws[size_t(r) * K / 32]);
                vk::Buffer bq = kt::make(ctx, wq.size()), bs = kt::make(ctx, ws.size());
                std::vector<uint8_t> gq[2], gs[2];
                for (int run = 0; run < 2; ++run) {
                    std::memset(bq.map, 0xcd, wq.size());
                    std::memset(bs.map, 0xcd, ws.size());
                    ctx.begin();
                    kt::must(vki::quant_mxfp8(ctx, tx, bq.addr, bs.addr, K, rows), "quant_mxfp8");
                    ctx.barrier();
                    ctx.submit_wait();
                    gq[run] = kt::host<uint8_t>(bq, wq.size());
                    gs[run] = kt::host<uint8_t>(bs, ws.size());
                }
                size_t bad = 0, first = SIZE_MAX, bads = 0;
                for (size_t i = 0; i < wq.size(); ++i) if (gq[0][i] != wq[i] && bad++ == 0) first = i;
                for (size_t i = 0; i < ws.size(); ++i) bads += gs[0][i] != ws[i];
                const bool same = gq[0] == gq[1] && gs[0] == gs[1];
                char d[200];
                std::snprintf(d, sizeof d, "%u rows: values differ %zu (first %lld), scales differ %zu, run-to-run %s", rows, bad,
                              first == SIZE_MAX ? -1LL : (long long)first, bads, same ? "identical" : "DIFFERS");
                report("quant.mxfp8_" + tag, bad == 0 && bads == 0 && same, d);
                ctx.free(bq); ctx.free(bs);
            }
            {  // NVFP4
                std::vector<uint8_t> wq(size_t(rows) * K / 2), ws(size_t(rows) * K / 16);
                std::vector<float> wa(rows);
                std::vector<uint32_t> wc(rows);
                for (uint32_t r = 0; r < rows; ++r) {
                    int n4 = 0;
                    wa[r] = ref::nvfp4_quantize_row(&x[size_t(r) * K], int(K), &wq[size_t(r) * K / 2], &ws[size_t(r) * K / 16], &n4);
                    wc[r] = uint32_t(n4);
                }
                vk::Buffer bq = kt::make(ctx, wq.size()), bs = kt::make(ctx, ws.size()), ba = kt::make(ctx, rows * 4ull),
                           bc = kt::make(ctx, rows * 4ull);
                std::vector<uint8_t> gq[2], gs[2];
                std::vector<float> ga[2];
                std::vector<uint32_t> gc[2];
                for (int run = 0; run < 2; ++run) {
                    std::memset(bq.map, 0xcd, wq.size());
                    std::memset(bs.map, 0xcd, ws.size());
                    std::memset(bc.map, 0, rows * 4ull);
                    ctx.begin();
                    kt::must(vki::quant_nvfp4(ctx, tx, bq.addr, bs.addr, ba.addr, bc.addr, K, rows), "quant_nvfp4");
                    ctx.barrier();
                    ctx.submit_wait();
                    gq[run] = kt::host<uint8_t>(bq, wq.size());
                    gs[run] = kt::host<uint8_t>(bs, ws.size());
                    ga[run] = kt::host<float>(ba, rows);
                    gc[run] = kt::host<uint32_t>(bc, rows);
                }
                size_t bad = 0, first = SIZE_MAX, bads = 0, bada = 0, badc = 0, c4 = 0;
                for (size_t i = 0; i < wq.size(); ++i) if (gq[0][i] != wq[i] && bad++ == 0) first = i;
                for (size_t i = 0; i < ws.size(); ++i) bads += gs[0][i] != ws[i];
                for (uint32_t r = 0; r < rows; ++r) {
                    bada += fbits(ga[0][r]) != fbits(wa[r]);
                    badc += gc[0][r] != wc[r];
                    c4 += wc[r];
                }
                const bool same = gq[0] == gq[1] && gs[0] == gs[1] && ga[0] == ga[1] && gc[0] == gc[1];
                char d[240];
                std::snprintf(d, sizeof d, "%u rows: codes differ %zu (first %lld), scales %zu, amax %zu, choose4 %zu (%zu / %u blocks /4), run-to-run %s",
                              rows, bad, first == SIZE_MAX ? -1LL : (long long)first, bads, bada, badc, c4, rows * K / 16,
                              same ? "identical" : "DIFFERS");
                report("quant.nvfp4_" + tag, bad == 0 && bads == 0 && bada == 0 && badc == 0 && same, d);
                ctx.free(bq); ctx.free(bs); ctx.free(ba); ctx.free(bc);
            }
            ctx.free(bx);
        }
}

// ---- row "router": ki_router.comp against router_ref.h (InkS4) ----
static void row_router(vk::Context &ctx) {
    std::mt19937 g(99);
    std::normal_distribution<float> nd(0.f, 1.f);
    const uint32_t E = 128, H = 2304, m = g_quick ? 8 : 64;
    std::vector<float> x(size_t(m) * H), w(size_t(E) * H), bias(E);
    for (auto &v : x) v = nd(g);
    for (auto &v : w) v = nd(g) * 0.02f;
    for (uint32_t e = 0; e < E; ++e) bias[e] = nd(g) * 0.01f;
    bias[20] = std::numeric_limits<float>::quiet_NaN();  // a NaN key ranks below every number (never selected here)
    for (uint32_t i = 0; i < H; ++i) w[size_t(7) * H + i] = w[size_t(3) * H + i];  // experts 3 / 7 tie (same weights, bias)
    for (uint32_t e = 10; e < 17; ++e) bias[e] = 0.5f;
    bias[3] = bias[7] = 0.25f;
    for (uint32_t i = 0; i < H; ++i) x[i] = 0.f;  // token 0: every logit 0 -> keys = 0.5 + bias
    vk::Buffer bx = upload_as(ctx, x, true), bw = upload_as(ctx, w, true), bb = kt::upload(ctx, bias);
    vk::Buffer bi = kt::make(ctx, m * 32ull), bt = kt::make(ctx, m * 32ull), bl = kt::make(ctx, uint64_t(m) * E * 4);
    std::vector<int32_t> wi(m * 8);
    std::vector<float> wt(m * 8);
    for (uint32_t t = 0; t < m; ++t) ref::route_row(&x[size_t(t) * H], w.data(), bias.data(), &wi[t * 8], &wt[t * 8]);
    std::vector<int32_t> gi[2];
    std::vector<float> gt[2];
    for (int run = 0; run < 2; ++run) {
        std::memset(bi.map, 0xff, m * 32ull);
        std::memset(bt.map, 0xff, m * 32ull);
        ctx.begin();
        kt::must(vki::router(ctx, bx.addr, bw.addr, bb.addr, bl.addr, bi.addr, bt.addr, m, E, H, 8.0f), "router");
        ctx.barrier();
        ctx.submit_wait();
        gi[run] = kt::host<int32_t>(bi, m * 8);
        gt[run] = kt::host<float>(bt, m * 8);
    }
    size_t badi = 0, badw = 0, f;
    for (size_t i = 0; i < wi.size(); ++i) badi += gi[0][i] != wi[i];
    badw = ndiff(gt[0], wt, &f);
    size_t tie = 0;  // tokens whose winners hold exactly one of the tied pair 3 / 7 (the tie rule decided)
    for (uint32_t t = 0; t < m; ++t) {
        bool h3 = false, h7 = false;  // token 0: experts 10..16 lead, 3 and 7 tie for the 8th slot (3 must win)
        for (int k = 0; k < 8; ++k) { h3 |= wi[t * 8 + k] == 3; h7 |= wi[t * 8 + k] == 7; }
        tie += h3 != h7;
    }
    const bool same = gi[0] == gi[1] && ndiff(gt[0], gt[1], &f) == 0;
    char d[200];
    std::snprintf(d, sizeof d, "%u tokens: ids differ %zu, weights differ %zu, tie-decided tokens %zu, run-to-run %s", m, badi, badw,
                  tie, same ? "identical" : "DIFFERS");
    report("router.e128_h2304", badi == 0 && badw == 0 && same && tie >= 1, d);
    ctx.free(bx); ctx.free(bw); ctx.free(bb); ctx.free(bi); ctx.free(bt);
}

// ---- row "moe": ki_moe.comp routed-expert products against moe_ref.h (InkS4) ----
struct MoeData {
    std::vector<uint8_t> aq, as, w, ws;
    std::vector<float> aamax, wamax;
    float den = 1.f;
};
static float ref_moe_product(uint32_t fmt, const MoeData &d, uint32_t K, uint32_t ar, uint32_t wr, uint32_t e) {
    if (fmt == 0) {
        const float acc = ref::nvfp4_dot(&d.aq[size_t(ar) * K / 2], &d.as[size_t(ar) * K / 16], &d.w[size_t(wr) * K / 2],
                                         &d.ws[size_t(wr) * K / 16], int(K));
        return ref::mul(acc, ref::nvfp4_outer_scale(d.aamax[ar], d.wamax[e], ref::nvfp4_outer_rcp(d.den)));
    }
    return ref::mxfp8_dot(&d.aq[size_t(ar) * K], &d.as[size_t(ar) * K / 32], &d.w[size_t(wr) * K], &d.ws[size_t(wr) * K / 32], int(K));
}

// Random operands: E experts of N x K weights, `rows` activation rows; e4m3 bytes avoid NaN (0x7f / 0xff).
static MoeData moe_data(std::mt19937 &g, uint32_t fmt, uint32_t E, uint32_t N, uint32_t K, uint32_t rows) {
    MoeData d;
    std::uniform_int_distribution<uint32_t> byte(0, 255), sc4(0x20, 0x48), sc8(118, 134);
    auto e4 = [&] { uint32_t v = byte(g); return uint8_t((v & 0x7f) == 0x7f ? v - 1 : v); };
    const uint32_t per = fmt == 0 ? 2 : 1, bs = fmt == 0 ? 16 : 32;
    d.aq.resize(size_t(rows) * K / per);
    d.w.resize(size_t(E) * N * K / per);
    for (auto &v : d.aq) v = fmt == 0 ? uint8_t(byte(g)) : e4();
    for (auto &v : d.w) v = fmt == 0 ? uint8_t(byte(g)) : e4();
    d.as.resize(size_t(rows) * K / bs);
    d.ws.resize(size_t(E) * N * K / bs);
    for (auto &v : d.as) v = uint8_t(fmt == 0 ? sc4(g) : sc8(g));
    for (auto &v : d.ws) v = uint8_t(fmt == 0 ? sc4(g) : sc8(g));
    std::uniform_real_distribution<float> ud(0.5f, 8.f);
    d.aamax.resize(rows);
    d.wamax.resize(E);
    for (auto &v : d.aamax) v = ud(g);
    for (auto &v : d.wamax) v = ud(g);
    d.den = ud(g);
    return d;
}

static void row_moe(vk::Context &ctx) {
    std::mt19937 g(1234);
    // Full: 40 tokens over 9 experts, so most experts hold more pairs than one grouped tile (PT = 32) and the grouped
    // path's multi-tile lists run; quick: 2 tokens over 16.
    const uint32_t E = 16, Epick = g_quick ? 16 : 9, H = 2304, I = 1152, tokens = g_quick ? 2 : 40, pairs = tokens * 8;
    std::vector<int32_t> ids(pairs);
    std::vector<float> wts(pairs);
    std::uniform_real_distribution<float> uw(0.1f, 3.f);
    for (uint32_t t = 0; t < tokens; ++t) {  // 8 distinct experts per token, ascending
        std::vector<int32_t> all(Epick);
        for (uint32_t e = 0; e < Epick; ++e) all[e] = int32_t(e);
        std::shuffle(all.begin(), all.end(), g);
        std::sort(all.begin(), all.begin() + 8);
        for (uint32_t k = 0; k < 8; ++k) { ids[t * 8 + k] = all[k]; wts[t * 8 + k] = uw(g); }
    }
    vk::Buffer bids = kt::upload(ctx, ids), bwts = kt::upload(ctx, wts), btab = kt::make(ctx, vki::moe_tab_bytes(pairs));
    // D-I-EPB (a1): the NVFP4 grouped tiles' int8 cooperative-matrix route ("_cm"), which an AMD cooperative-matrix
    // device must take (its layout check passes). Exactness: a block's integer sum adds 16 products of staged codes
    // 2 * e2m1 in [-12, 12], so |S_j| <= 16 * 144 = 2304 < 2^24: the int32 sum and float(S_j) are exact whatever the
    // matrix unit's internal order, and the epilogue is the vector route's. The "_bound" data (every code +-6, each
    // block's sum +-2304) reaches the bound; the mutant (the chunk's block terms added in reverse) must differ.
    static_assert(16 * 12 * 12 == 2304 && 2304 < (1 << 24), "NVFP4 block sum bound");
    const bool cm = vki::moe_probe(ctx) == CERE_OK, need_cm = vki::moe_cm_device(ctx) && ctx.info().vendor_id == 0x1002;
    report("moe.cm_probe", cm || !need_cm,
           cm ? "int8 cooperative-matrix route on" : need_cm ? "layout check FAILED on an AMD device" : "vector route only");
    for (uint32_t v = 0; v < 4; ++v) {  // nvfp4, nvfp4 bound, nvfp4 order, mxfp8
        const uint32_t fmt = v == 3 ? 1 : 0, bound = v == 1 ? 1 : 0, order = v == 2 ? 1 : 0;
        const std::string tag = std::string(fmt == 0 ? "nvfp4" : "mxfp8") + (bound ? "_bound" : order ? "_order" : "");
        for (uint32_t stage = 0; stage < 2; ++stage) {
            const uint32_t N = stage == 0 ? 2 * I : H, K = stage == 0 ? H : I, arows = stage == 0 ? tokens : pairs;
            MoeData d = moe_data(g, fmt, E, N, K, arows);
            if (bound) {  // activation codes all +6; a weight block's codes all +6 or all -6
                for (auto &v : d.aq) v = 0x77;
                for (size_t i = 0; i < d.w.size(); ++i) d.w[i] = (i / 8) % 3 == 1 ? 0xFF : 0x77;
            }
            if (order) {  // every code +-1; block j's terms by j % 16: +B, -B, then fourteen s
                // B = 16 * 448 * 448 (ulp 2^-2), s = 16 * 2^-9 * 2^-9 = 2^-14 < ulp(B) / 64. The +-B pair opens a
                // chunk for any chunk size C in {4, 8, 16} (16-aligned). In order each output ends at 14 s (the s
                // after the last pair; a carried sum < ulp(B) / 2 vanishes into +B); reversed within a chunk, the
                // pair's chunk-mates are wiped (s ... -B + B = 0), so it ends at (16 - C) s: every output differs.
                for (auto &v : d.aq) v = 0x22;
                for (size_t i = 0; i < d.w.size(); ++i) d.w[i] = (i / 8) % (K / 16) % 16 == 1 ? 0xAA : 0x22;
                for (size_t i = 0; i < d.as.size(); ++i) d.as[i] = i % (K / 16) % 16 < 2 ? 0x7E : 0x01;
                for (size_t i = 0; i < d.ws.size(); ++i) d.ws[i] = i % (K / 16) % 16 < 2 ? 0x7E : 0x01;
            }
            vk::Buffer baq = kt::upload(ctx, d.aq), bas = kt::upload(ctx, d.as), bw = kt::upload(ctx, d.w), bws = kt::upload(ctx, d.ws);
            vk::Buffer baa = kt::upload(ctx, d.aamax), bwa = kt::upload(ctx, d.wamax);
            const uint32_t ncol = stage == 0 ? I : H;
            std::vector<float> want(size_t(pairs) * ncol);
            for (uint32_t p = 0; p < pairs; ++p) {
                const uint32_t e = uint32_t(ids[p]);
                for (uint32_t c = 0; c < ncol; ++c) {
                    if (stage == 1) { want[size_t(p) * ncol + c] = ref::bf16_round(ref_moe_product(fmt, d, K, p, e * N + c, e)); continue; }
                    const float gt = ref::bf16_round(ref_moe_product(fmt, d, K, p / 8, e * N + 2 * c, e));
                    const float up = ref::bf16_round(ref_moe_product(fmt, d, K, p / 8, e * N + 2 * c + 1, e));
                    want[size_t(p) * ncol + c] = ref::bf16_round(ref::capped_swiglu(gt, up, wts[p]));
                }
            }
            vk::Buffer bo = kt::make(ctx, uint64_t(pairs) * ncol * 2);
            const vki::ActQ a{baq.addr, bas.addr, baa.addr};
            const vki::ExpertW w{bw.addr, bws.addr, bwa.addr, d.den};
            auto launch = [&](uint64_t tab, vki::MoeRoute route, vki::Mutant mut) {
                if (tab) {
                    kt::must(vki::moe_bucket(ctx, bids.addr, tab, pairs, E), "moe_bucket");
                    ctx.barrier();
                }
                kt::must(stage == 0 ? vki::moe_gate_up(ctx, fmt, a, w, bids.addr, bwts.addr, bo.addr, N, K, pairs, tab, route, mut)
                                    : vki::moe_down(ctx, fmt, a, w, bids.addr, bo.addr, N, K, pairs, tab, route, mut),
                         "moe");
            };
            const std::string name = "moe." + tag + (stage == 0 ? "_gate_up" : "_down");
            check_twice(ctx, name, bo, want.size(), true, want, [&] { launch(0, vki::MoeRoute::kAuto, vki::Mutant::kNone); });
            check_twice(ctx, name + "_grouped", bo, want.size(), true, want,
                        [&] { launch(btab.addr, vki::MoeRoute::kVector, vki::Mutant::kNone); });
            if (fmt == 0 && cm) {
                check_twice(ctx, name + "_grouped_cm", bo, want.size(), true, want,
                            [&] { launch(btab.addr, vki::MoeRoute::kAuto, vki::Mutant::kNone); });
                if (order) {  // fail-first: every output of the mutant tile must differ from the reference
                    std::memset(bo.map, 0xa5, want.size() * 2);
                    ctx.begin();
                    launch(btab.addr, vki::MoeRoute::kAuto, vki::Mutant::kMoeBlockOrder);
                    ctx.barrier();
                    ctx.submit_wait();
                    size_t f = 0;
                    const size_t bad = ndiff(read_as(bo, want.size(), true), want, &f);
                    char det[128];
                    std::snprintf(det, sizeof det, "mutant blockorder: %zu / %zu differ (must be all)", bad, want.size());
                    report(name + "_grouped_cm_mutant", bad == want.size(), det);
                }
            }
            ctx.free(baq); ctx.free(bas); ctx.free(bw); ctx.free(bws); ctx.free(baa); ctx.free(bwa); ctx.free(bo);
        }
    }
    {  // combine
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> y(size_t(pairs) * H), want(size_t(tokens) * H);
        for (auto &v : y) v = ref_bf_round(nd(g));
        for (uint32_t t = 0; t < tokens; ++t) {
            std::vector<uint16_t> yb(size_t(8) * H), ob(H);
            const uint16_t *rows[8];
            for (uint32_t k = 0; k < 8; ++k) {
                for (uint32_t c = 0; c < H; ++c) yb[size_t(k) * H + c] = ref::encode_bf16(y[size_t(t * 8 + k) * H + c]);
                rows[k] = &yb[size_t(k) * H];
            }
            ref::moe_combine_row(rows, ob.data());
            for (uint32_t c = 0; c < H; ++c) want[size_t(t) * H + c] = ref::decode_bf16(ob[c]);
        }
        vk::Buffer by = upload_as(ctx, y, true), bo = kt::make(ctx, uint64_t(tokens) * H * 2);
        check_twice(ctx, "moe.combine", bo, want.size(), true, want,
                    [&] { kt::must(vki::moe_combine(ctx, by.addr, bo.addr, H, tokens), "moe_combine"); });
        ctx.free(by); ctx.free(bo);
    }
    ctx.free(bids); ctx.free(bwts); ctx.free(btab);
}

// ---- row "mix": ki_mix.comp residual-stream ops against executor/ref/inkling depth_ref.h, msg_ref.h, embed_ref.h,
// softcap_ref.h (InkS3) ----
static std::vector<uint16_t> bf_bits(std::vector<float> &v) {  // v becomes its BF16-rounded values
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) { h[i] = ref::encode_bf16(v[i]); v[i] = ref::decode_bf16(h[i]); }
    return h;
}
static void row_mix(vk::Context &ctx) {
    constexpr uint32_t H = 2304;
    std::mt19937 g(4242);
    std::normal_distribution<float> nd(0.f, 1.f);
    const uint32_t M = g_quick ? 3 : 17;
    // depth mix: S = 1 .. 5 states, mixed BF16 / FP32, nq = 1 and 3 (one state scaled huge, one tiny).
    for (uint32_t S : {1u, 2u, 4u, 5u})
        for (uint32_t nq : {1u, 3u}) {
            std::vector<std::vector<float>> st(S);
            std::vector<vk::Buffer> bs(S);
            std::vector<vki::Tensor> ts(S);
            for (uint32_t s = 0; s < S; ++s) {
                const bool bf = s + 1 < S || S == 5;
                const float sc = s == 1 ? 300.f : s == 2 ? 1e-3f : 1.f;
                st[s].resize(size_t(M) * H);
                for (auto &v : st[s]) v = nd(g) * sc;
                bs[s] = upload_as(ctx, st[s], bf);
                ts[s] = {bs[s].addr, bf ? vki::Dt::kBF16 : vki::Dt::kF32};
            }
            std::vector<float> qv(size_t(nq) * H);
            for (auto &v : qv) v = nd(g) * 0.5f;
            vk::Buffer bq = kt::upload(ctx, qv), bo = kt::make(ctx, uint64_t(nq) * M * H * 4);
            std::vector<float> want(size_t(nq) * M * H);
            for (uint32_t m = 0; m < M; ++m) {
                const float *rows[5];
                for (uint32_t s = 0; s < S; ++s) rows[s] = &st[s][size_t(m) * H];
                ref::depth_mix_row(rows, int(S), qv.data(), H, int(nq), &want[size_t(m) * H], int64_t(M) * H);
            }
            check_twice(ctx, "mix.depth_S" + std::to_string(S) + "_q" + std::to_string(nq), bo, want.size(), false, want, [&] {
                kt::must(vki::depth_mix(ctx, ts.data(), S, bq.addr, nq, bo.addr, M, ref::kRmsEps), "depth_mix");
            });
            for (auto &b : bs) ctx.free(b);
            ctx.free(bq); ctx.free(bo);
        }
    {  // message gate; Mg tokens: several of both token tiles plus tails (D-I-EPB (a2))
        const uint32_t Mg = M + 20;
        std::vector<float> h(size_t(Mg) * H), ms(size_t(Mg) * H), down(size_t(64) * 2 * H), fg(size_t(H) * 64), gn(size_t(H) * 64);
        for (auto &v : h) v = nd(g) * 2.f;
        for (auto &v : ms) v = nd(g);
        for (auto &v : down) v = nd(g) * 0.03f;
        for (auto &v : fg) v = nd(g) * 0.2f;
        for (auto &v : gn) v = nd(g) * 0.2f;
        // Order-sensitive rows: in every even d row, lane t's first two chain terms (h columns t and t + 256, equal
        // values) are +2^16 g and -2^16 g. In order they cancel exactly before the small terms; any other order rounds
        // the small partial sum against 2^16 g (ulp ~ 2^-7), so the mutant must change d even after its BF16 rounding.
        for (uint32_t m = 0; m < Mg; ++m)
            for (uint32_t c = 0; c < 256; ++c) h[size_t(m) * H + c + 256] = h[size_t(m) * H + c];
        for (uint32_t r = 0; r < 64; r += 2)
            for (uint32_t c = 0; c < 256; ++c) {
                down[size_t(r) * 2 * H + c] = 65536.f;
                down[size_t(r) * 2 * H + c + 256] = -65536.f;
            }
        const std::vector<uint16_t> dh = bf_bits(down), fh = bf_bits(fg), gh = bf_bits(gn);
        vk::Buffer bh = kt::upload(ctx, h), bm = kt::upload(ctx, ms), bd = kt::upload(ctx, dh), bf = kt::upload(ctx, fh),
                   bgn = kt::upload(ctx, gh), bo = kt::make(ctx, uint64_t(Mg) * H * 4),
                   bdd = kt::make(ctx, uint64_t(Mg) * 64 * 4);
        std::vector<float> want(size_t(Mg) * H);
        for (uint32_t m = 0; m < Mg; ++m)
            ref::msg_gate_row(&h[size_t(m) * H], &ms[size_t(m) * H], dh.data(), fh.data(), gh.data(), ref::kRmsEps,
                              &want[size_t(m) * H]);
        const auto gate = [&](vki::MsgRoute route, vki::Mutant m) {
            kt::must(vki::msg_gate(ctx, bh.addr, bm.addr, bd.addr, bf.addr, bgn.addr, bdd.addr, bo.addr, Mg, ref::kRmsEps, route, m),
                     "msg_gate");
        };
        check_twice(ctx, "mix.msg_gate", bo, want.size(), false, want, [&] { gate(vki::MsgRoute::kRow, vki::Mutant::kNone); });
        check_twice(ctx, "mix.msg_gate_tiled", bo, want.size(), false, want,
                    [&] { gate(vki::MsgRoute::kTiled, vki::Mutant::kNone); });
        // The d scratch word for word against the row route, and the fail-first mutant (MODE 7's d chains backwards)
        // must change it.
        const auto run_d = [&](vki::MsgRoute route, vki::Mutant m) {
            std::memset(bdd.map, 0x5a, size_t(Mg) * 64 * 4);
            ctx.begin();
            gate(route, m);
            ctx.barrier();
            ctx.submit_wait();
            return kt::host<uint32_t>(bdd, size_t(Mg) * 64);
        };
        const std::vector<uint32_t> d_row = run_d(vki::MsgRoute::kRow, vki::Mutant::kNone);
        const auto ddiff = [&](const std::vector<uint32_t> &x) {
            size_t n = 0;
            for (size_t i = 0; i < x.size(); ++i) n += x[i] != d_row[i];
            return n;
        };
        const size_t dt = ddiff(run_d(vki::MsgRoute::kTiled, vki::Mutant::kNone));
        const size_t dm = ddiff(run_d(vki::MsgRoute::kTiled, vki::Mutant::kMsgGateOrder));
        char det[128];
        std::snprintf(det, sizeof det, "%zu d words, %zu differ from the row route", d_row.size(), dt);
        report("mix.msg_gate_tiled_d", dt == 0, det);
        std::snprintf(det, sizeof det, "mutant msgorder: %zu d words differ (must be > 0)", dm);
        report("mix.msg_gate_tiled_mutant", dm > 0, det);
        ctx.free(bh); ctx.free(bm); ctx.free(bd); ctx.free(bf); ctx.free(bgn); ctx.free(bo); ctx.free(bdd);
    }
    {  // normed embedding
        const uint32_t V = 301;
        std::vector<float> tab(size_t(V) * H), w(H);
        for (auto &v : tab) v = nd(g) * 0.05f;
        for (auto &v : w) v = 1.f + nd(g) * 0.1f;
        const std::vector<uint16_t> th = bf_bits(tab), wh = bf_bits(w);
        std::vector<int32_t> ids(M);
        for (auto &id : ids) id = int32_t(g() % V);
        ids[0] = V - 1;
        vk::Buffer bt = kt::upload(ctx, th), bw = kt::upload(ctx, wh), bi = kt::upload(ctx, ids), bo = kt::make(ctx, uint64_t(M) * H * 2);
        std::vector<float> want(size_t(M) * H);
        std::vector<uint16_t> row(H);
        for (uint32_t m = 0; m < M; ++m) {
            ref::embed_norm_row(th.data(), uint32_t(ids[m]), wh.data(), ref::kRmsEps, row.data());
            for (uint32_t i = 0; i < H; ++i) want[size_t(m) * H + i] = ref::decode_bf16(row[i]);
        }
        check_twice(ctx, "mix.embed_norm", bo, want.size(), true, want, [&] {
            kt::must(vki::embed_norm(ctx, bt.addr, bi.addr, bw.addr, bo.addr, M, ref::kRmsEps), "embed_norm");
        });
        ctx.free(bt); ctx.free(bw); ctx.free(bi); ctx.free(bo);
    }
    {  // softcap (BF16 and FP32 logits incl. specials), attention residual, layer output
        const uint32_t n = g_quick ? 70001 : 201024 * 3 + 5;
        std::vector<float> x(n), x2(n);
        for (uint32_t i = 0; i < n; ++i) x[i] = i % 7 == 0 ? bitsf(uint32_t(g())) : nd(g) * 40.f;
        x2 = x;
        for (const bool xbf : {true, false}) {
            std::vector<float> &xv = xbf ? x : x2;
            vk::Buffer bx = upload_as(ctx, xv, xbf), bo = kt::make(ctx, uint64_t(n) * 4);
            std::vector<float> want(n);
            for (uint32_t i = 0; i < n; ++i) want[i] = ref::softcap(xv[i]);
            check_twice(ctx, std::string("mix.softcap_") + (xbf ? "bf16" : "f32"), bo, n, false, want, [&] {
                kt::must(vki::softcap(ctx, {bx.addr, xbf ? vki::Dt::kBF16 : vki::Dt::kF32}, bo.addr, n), "softcap");
            });
            ctx.free(bx); ctx.free(bo);
        }
        std::vector<float> hv(n), ov(n), rv(n);
        for (uint32_t i = 0; i < n; ++i) { hv[i] = nd(g) * 4.f; ov[i] = nd(g); rv[i] = nd(g) * 4.f; }
        const std::vector<uint16_t> hh = bf_bits(hv), oh = bf_bits(ov);
        vk::Buffer bh = kt::upload(ctx, hh), bo2 = kt::upload(ctx, oh), br = kt::upload(ctx, rv), bo = kt::make(ctx, uint64_t(n) * 4);
        for (const bool boundary : {false, true}) {
            std::vector<float> want(n);
            for (uint32_t i = 0; i < n; ++i) want[i] = ref::attn_residual(boundary, hh[i], oh[i]);
            check_twice(ctx, std::string("mix.attn_residual") + (boundary ? "_boundary" : ""), bo, n, false, want, [&] {
                kt::must(vki::attn_residual(ctx, bh.addr, bo2.addr, bo.addr, n, boundary), "attn_residual");
            });
        }
        std::vector<float> want(n);
        for (uint32_t i = 0; i < n; ++i) want[i] = ref::decode_bf16(ref::layer_out(rv[i], oh[i]));
        check_twice(ctx, "mix.layer_out", bo, n, true, want, [&] {
            kt::must(vki::layer_out(ctx, br.addr, bo2.addr, bo.addr, n), "layer_out");
        });
        ctx.free(bh); ctx.free(bo2); ctx.free(br); ctx.free(bo);
    }
}

// ---- row "attn": ki_attn.comp (attention core, KV store) and the q / rel / k / v posts (ki_rownorm.comp) against
// executor/ref/inkling/attn_ref.h (InkS2) ----
namespace ra = ref::attn;
struct AttnCase {
    const char *name;
    int kind;
    uint32_t pos0, n;
    bool slots;      // full: a permuted pool
    bool attend;     // false: KV store only
};
static void row_attn(vk::Context &ctx) {
    std::mt19937 g(4242);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd_bf = [&](size_t n, float sc) {
        std::vector<uint16_t> h(n);
        for (auto &x : h) x = ref::encode_bf16(nd(g) * sc);
        return h;
    };
    const AttnCase cases[] = {
        {"attn.sliding_ring_1500x40", ra::kSliding, 1500, g_quick ? 8u : 40u, false, true},
        {"attn.sliding_prefill_0x150", ra::kSliding, 0, g_quick ? 70u : 150u, false, true},
        {"attn.full_pool_4090x12", ra::kFull, 4090, g_quick ? 8u : 12u, true, true},
        {"attn.full_ident_0x70", ra::kFull, 0, 70, false, true},
        {"attn.full_ident_300x20", ra::kFull, 300, 20, false, true},
        // Past one 4,096-key group (p07's 4,833-token prompt): the split path's span / group fold order.
        {"attn.full_pool_4790x40", ra::kFull, 4790, g_quick ? 8u : 40u, true, true},
        {"attn.sliding_ring_4790x40", ra::kSliding, 4790, g_quick ? 8u : 40u, false, true},
        // Last row at pos % 256 == 255 (a 512-row prefill chunk's end): 4 sliding spans, the earlier rows 5.
        {"attn.sliding_ring_1500x36", ra::kSliding, 1500, 36, false, true},
        {"attn.sliding_store_100x1100", ra::kSliding, 100, 1100, false, false},
    };
    for (const AttnCase &c : cases) {
        const uint32_t n = c.n, P = c.pos0 + n;
        // Raw projections, then the posts on the host (the ref's inputs) and on the device (the post rows).
        const std::vector<uint16_t> q_raw = rnd_bf(size_t(n) * ra::kQRow, 1.5f), r_raw = rnd_bf(size_t(n) * ra::kRelRow, 1.5f),
                                    k_raw = rnd_bf(size_t(n) * ra::kKvRow, 1.5f), v_raw = rnd_bf(size_t(n) * ra::kKvRow, 1.f),
                                    q_sc = rnd_bf(ra::kQRow, 1.f), r_sc = rnd_bf(ra::kRelRow, 1.f),
                                    proj = rnd_bf(size_t(ra::kDRel) * ra::kRelExtent, 1.f), sink = rnd_bf(size_t(n) * 24, 2.f),
                                    gate = rnd_bf(size_t(n) * ra::kQRow, 2.f);
        std::vector<uint16_t> q(q_raw.size()), r(r_raw.size()), k(k_raw.size()), v(v_raw.size());
        for (uint32_t i = 0; i < n; ++i) {
            ra::q_post_row(c.kind, &q_raw[size_t(i) * ra::kQRow], q_sc.data(), c.pos0 + i, &q[size_t(i) * ra::kQRow]);
            ra::rel_post_row(c.kind, &r_raw[size_t(i) * ra::kRelRow], r_sc.data(), c.pos0 + i, &r[size_t(i) * ra::kRelRow]);
            ra::k_post_row(&k_raw[size_t(i) * ra::kKvRow], &k[size_t(i) * ra::kKvRow]);
            ra::v_post_row(&v_raw[size_t(i) * ra::kKvRow], &v[size_t(i) * ra::kKvRow]);
        }
        // The cache: sliding ring of 1024 slots (key j < pos0 at j % 1024), or a pool of P + 64 rows (key j at slots[j]).
        const uint32_t cslots = c.kind == ra::kSliding ? ra::kRingSlots : P + 64;
        std::vector<uint16_t> kc(size_t(cslots) * ra::kKvRow), vc(kc.size());
        for (auto &x : kc) x = ref::encode_bf16(std::fma(nd(g), 0.4f, 0.f));
        for (auto &x : vc) x = ref::encode_bf16(nd(g));
        std::vector<int64_t> sl64;
        std::vector<int32_t> sl32;
        if (c.slots) {
            std::vector<int32_t> perm(cslots);
            for (uint32_t i = 0; i < cslots; ++i) perm[i] = int32_t(i);
            std::shuffle(perm.begin(), perm.end(), g);
            sl32.assign(perm.begin(), perm.begin() + P);
            sl64.assign(sl32.begin(), sl32.end());
        }
        vk::Buffer bq = kt::upload(ctx, q), br = kt::upload(ctx, r), bk = kt::upload(ctx, k), bv = kt::upload(ctx, v),
                   bp = kt::upload(ctx, proj), bs = kt::upload(ctx, sink), bg = kt::upload(ctx, gate), bkc = kt::upload(ctx, kc),
                   bvc = kt::upload(ctx, vc), bo = kt::make(ctx, uint64_t(n) * ra::kQRow * 2),
                   bsl = c.slots ? kt::upload(ctx, sl32) : kt::make(ctx, 4);
        vki::AttnSeq s;
        s.kind = uint32_t(c.kind); s.pos0 = c.pos0; s.n = n;
        s.k = bk.addr; s.v = bv.addr; s.kc = bkc.addr; s.vc = bvc.addr; s.slots = c.slots ? bsl.addr : 0;
        if (c.attend) {
            vki::AttnRows a;
            a.q = bq.addr; a.r = br.addr; a.proj = bp.addr; a.sink = bs.addr; a.gate = bg.addr; a.out = bo.addr;
            const ra::Keys keys{c.kind, 0, int64_t(c.pos0), k.data(), v.data(), ra::kKvRow, ra::kKvRow, kc.data(), vc.data(),
                                c.slots ? sl64.data() : nullptr};
            std::vector<uint16_t> o(size_t(n) * ra::kQRow);
            for (uint32_t i = 0; i < n; ++i)
                ra::attention_row(keys, &q[size_t(i) * ra::kQRow], &r[size_t(i) * ra::kRelRow], proj.data(), &sink[size_t(i) * 24],
                                  &gate[size_t(i) * ra::kQRow], c.pos0 + i, &o[size_t(i) * ra::kQRow]);
            std::vector<float> want(o.size());
            for (size_t i = 0; i < o.size(); ++i) want[i] = ref::decode_bf16(o[i]);
            // Two row blocks (r0 = 0 and n / 2): the model's per-submit split; the same bits as one dispatch.
            check_twice(ctx, c.name, bo, want.size(), true, want, [&] {
                kt::must(vki::attention(ctx, s, a, 0, n / 2), "attention");
                kt::must(vki::attention(ctx, s, a, n / 2, n - n / 2), "attention");
            });
            // The split path (span partials in parallel, then their fold): the same bits.
            const uint64_t pb = uint64_t(n) * 24 * (P / 256 + 1) * vki::kAttnPartBytes;
            vk::Buffer bpart = kt::make(ctx, pb);
            a.part = bpart.addr;
            a.part_bytes = pb;
            check_twice(ctx, std::string(c.name) + ".split", bo, want.size(), true, want, [&] {
                kt::must(vki::attention(ctx, s, a, 0, n / 2, vki::AttnRoute::kSplit), "attention");
                kt::must(vki::attention(ctx, s, a, n / 2, n - n / 2, vki::AttnRoute::kSplit), "attention");
            });
            // D-I-EPB (a2): the tiled spans (MODE 4, kAttnTileRows rows x one KV head per workgroup) on every row count,
            // the blocks' ends included (n / 2 and n - n / 2 not multiples of the tile rows): the same bits.
            const auto tiled = [&](vki::Mutant m) {
                kt::must(vki::attention(ctx, s, a, 0, n / 2, vki::AttnRoute::kTiled, m), "attention tiled");
                kt::must(vki::attention(ctx, s, a, n / 2, n - n / 2, vki::AttnRoute::kTiled, m), "attention tiled");
            };
            check_twice(ctx, std::string(c.name) + ".tiled", bo, want.size(), true, want, [&] { tiled(vki::Mutant::kNone); });
            // The FP32 span partials themselves (before the fold and BF16): MODE 4's must be MODE 2's, word for word, and
            // the fail-first mutant (each tile's P x V chain in reverse key order) must change them.
            const auto parts = [&](vki::AttnRoute route, vki::Mutant m) {
                std::memset(bpart.map, 0x5a, pb);
                ctx.begin();
                kt::must(vki::attention(ctx, s, a, 0, n, route, m), "attention parts");
                ctx.barrier();
                ctx.submit_wait();
                return kt::host<uint32_t>(bpart, pb / 4);
            };
            const std::vector<uint32_t> p_split = parts(vki::AttnRoute::kSplit, vki::Mutant::kNone);
            const auto pdiff = [&](const std::vector<uint32_t> &x) {
                size_t d = 0;
                for (size_t i = 0; i < x.size(); ++i) d += x[i] != p_split[i];
                return d;
            };
            const size_t dt = pdiff(parts(vki::AttnRoute::kTiled, vki::Mutant::kNone));
            const size_t dm = pdiff(parts(vki::AttnRoute::kTiled, vki::Mutant::kAttnKeyOrder));
            char det[160];
            std::snprintf(det, sizeof det, "%zu FP32 partial words, %zu differ from MODE 2", p_split.size(), dt);
            report(std::string(c.name) + ".tiled_parts", dt == 0, det);
            std::snprintf(det, sizeof det, "mutant keyorder: %zu partial words differ (must be > 0)", dm);
            report(std::string(c.name) + ".tiled_mutant", dm > 0, det);
            ctx.free(bpart);
        }
        {  // KV store (after the attention): the expected cache
            std::vector<uint16_t> wk = kc, wv = vc;
            const uint32_t first = c.kind == ra::kSliding && n > 1024 ? n - 1024 : 0;
            for (uint32_t i = first; i < n; ++i) {
                const uint32_t j = c.pos0 + i;
                const size_t slot = c.kind == ra::kSliding ? j % ra::kRingSlots : (c.slots ? size_t(sl32[j]) : j);
                std::memcpy(&wk[slot * ra::kKvRow], &k[size_t(i) * ra::kKvRow], ra::kKvRow * 2);
                std::memcpy(&wv[slot * ra::kKvRow], &v[size_t(i) * ra::kKvRow], ra::kKvRow * 2);
            }
            ctx.begin();
            kt::must(vki::kv_store(ctx, s), "kv_store");
            ctx.barrier();
            ctx.submit_wait();
            const bool ok = kt::host<uint16_t>(bkc, wk.size()) == wk && kt::host<uint16_t>(bvc, wv.size()) == wv;
            report(std::string(c.name) + ".store", ok, ok ? "cache bit-equal" : "cache DIFFERS");
        }
        // The posts on the device: q (scaled, logscaled on full), rel, k, v.
        {
            std::vector<float> tau(n);
            for (uint32_t i = 0; i < n; ++i) tau[i] = ra::tau(c.pos0 + i);
            vk::Buffer bqr = kt::upload(ctx, q_raw), brr = kt::upload(ctx, r_raw), bkr = kt::upload(ctx, k_raw),
                       bvr = kt::upload(ctx, v_raw), bqs = kt::upload(ctx, q_sc), brs = kt::upload(ctx, r_sc),
                       btau = kt::upload(ctx, tau), bpo = kt::make(ctx, uint64_t(n) * ra::kQRow * 2);
            const uint64_t tau_addr = c.kind == ra::kFull ? btau.addr : 0;
            const vki::Tensor out{bpo.addr, vki::Dt::kBF16};
            auto post = [&](const char *what, const std::vector<uint16_t> &w16, const std::function<void()> &launch) {
                std::vector<float> want(w16.size());
                for (size_t i = 0; i < w16.size(); ++i) want[i] = ref::decode_bf16(w16[i]);
                check_twice(ctx, std::string(c.name) + ".post_" + what, bpo, want.size(), true, want, launch);
            };
            post("q", q, [&] {
                kt::must(vki::l2_clip(ctx, {bqr.addr, vki::Dt::kBF16}, {bqs.addr, vki::Dt::kBF16}, tau_addr, out, 128, n * 24, 24,
                                      ra::kQkClip), "l2_clip q");
            });
            post("rel", r, [&] {
                kt::must(vki::l2_clip(ctx, {brr.addr, vki::Dt::kBF16}, {brs.addr, vki::Dt::kBF16}, tau_addr, out, 16, n * 24, 24,
                                      ra::kRelClip), "l2_clip rel");
            });
            post("k", k, [&] {
                kt::must(vki::l2_clip(ctx, {bkr.addr, vki::Dt::kBF16}, {}, 0, out, 128, n * 8, 8, ra::kQkClip), "l2_clip k");
            });
            post("v", v, [&] {
                kt::must(vki::rms_norm(ctx, {bvr.addr, vki::Dt::kBF16}, {}, out, 128, n * 8, ra::kVEps), "rms v");
            });
            ctx.free(bqr); ctx.free(brr); ctx.free(bkr); ctx.free(bvr); ctx.free(bqs); ctx.free(brs); ctx.free(btau); ctx.free(bpo);
        }
        ctx.free(bq); ctx.free(br); ctx.free(bk); ctx.free(bv); ctx.free(bp); ctx.free(bs); ctx.free(bg); ctx.free(bkc);
        ctx.free(bvc); ctx.free(bo); ctx.free(bsl);
    }
}

// ---- row "dense": ki_dense.comp against the fma chain y[r][j] = fma(x[r][i], w[j][i], acc) in ascending i ----
static void row_dense(vk::Context &ctx) {
    std::mt19937 g(5150);
    std::normal_distribution<float> nd(0.f, 1.f);
    struct Case { uint32_t m, n, k; bool f32; };
    const Case cases[] = {{1, 2304, 2304, true}, {17, 384, 2304, false}, {9, 2304, 3072, false}, {g_quick ? 3u : 40u, 130, 1024, true}};
    for (const Case &c : cases) {
        std::vector<float> x(size_t(c.m) * c.k), w(size_t(c.n) * c.k);
        for (auto &v : x) v = nd(g);
        for (auto &v : w) v = nd(g) * 0.03f;
        const auto hx = bf_bits(x), hw = bf_bits(w);
        std::vector<float> want(size_t(c.m) * c.n);
        for (uint32_t r = 0; r < c.m; ++r)
            for (uint32_t j = 0; j < c.n; ++j) {
                float acc = 0.f;
                for (uint32_t i = 0; i < c.k; ++i) acc = std::fma(x[size_t(r) * c.k + i], w[size_t(j) * c.k + i], acc);
                want[size_t(r) * c.n + j] = c.f32 ? acc : ref::decode_bf16(ref::encode_bf16(acc));
            }
        vk::Buffer bx = kt::upload(ctx, hx), bw = kt::upload(ctx, hw), bo = kt::make(ctx, want.size() * 4);
        const std::string name = "dense." + std::to_string(c.m) + "x" + std::to_string(c.n) + "x" + std::to_string(c.k);
        check_twice(ctx, name, bo, want.size(), !c.f32, want, [&] {
            kt::must(vki::dense(ctx, bx.addr, bw.addr, bo.addr, c.m, c.n, c.k, c.f32), "dense");
        });
        ctx.free(bx); ctx.free(bw); ctx.free(bo);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { std::printf("usage: ki_test SPV_DIR [--rows a,b] [--quick]\n"); return 2; }
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc) g_only = argv[++i];
        else if (a == "--quick") g_quick = true;
        else { std::printf("unknown argument %s\n", a.c_str()); return 2; }
    }
    auto ctx = vk::testing::create_or_die(argv[1]);
    std::printf("device: %s\n", ctx->info().name.c_str());
    struct Row { const char *name; std::function<void(vk::Context &)> fn; };
    const Row rows[] = {{"common", row_common}, {"trans", row_trans}, {"norm", row_norm}, {"conv", row_conv}, {"quant", row_quant},
                        {"router", row_router}, {"moe", row_moe}, {"mix", row_mix}, {"attn", row_attn}, {"dense", row_dense}};
    for (const Row &r : rows)
        if (selected(r.name)) r.fn(*ctx);
    std::printf("%s (%d failed)\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}
