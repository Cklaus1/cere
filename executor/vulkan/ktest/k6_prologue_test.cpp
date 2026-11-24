// VulkanAmd P3 (T3): contract rows of vkk6::prologue (src/kernels/k6_prologue_vk.*), the eager K6 attention prologue
// (executor/cuda/src/kernels/k6_attention.h:64-79, k6_attention.cu:1203-1216: k1::qk_norm_rope_gate then k1::kv_store_fp8
// with slots + prefix_len). Rows:
//  1. the CUDA suite's prologue rows (executor/tests/gpu/test_kernels_k9_qsa.py test_prologue_positions, D10): one token at
//     positions 2,047, 2,048, 4,096, 72,000 and 199,999 with a 200,000-row rotary table (vkk1::rope_table, theta 1e7) and a
//     non-identity slot table (a seeded permutation): q and k within the tolerance rule max(4e-3, 2c) against the FP64
//     Gemma norm + NeoX RoPE (exact angles), c = rel-L2 of a host FP32-table emulation (the role SGLang's fused op plays in
//     the CUDA row) against the same FP64; the gate bit-exact; the cache bit-exact: the token's rows equal the FP8 E4M3
//     cast (scale 1.0, the HIP test's independent reference) of the k the call returns and of v, every other row unchanged;
//  2. composition: the prologue == vkk1::qk_norm_rope_gate then vkk1::kv_store_fp8 (separate submits), bit for bit (q,
//     k, gate, both caches), at a decode shape (1 token), a verify shape (4 tokens) and an extend (37 tokens after a
//     prefix of 5), each with a slot map and with slots null (identity: slot0 = prefix_len);
//  3. repeats bit-identical; positions null == explicit positions prefix_len + t;
//  4. negative: positions3 UNSUPPORTED, bound / null / stride / alignment ARGUMENT with nothing recorded (outputs and
//     caches untouched after the submit), tokens 0 a no-op; prologue_bytes == the CUDA formula.
// Then decode-shape timing (the K1 batch method: us per call = GPU time of 100 dependent calls with a barrier after each
// / 100, median of 7 batches after 3 untimed). Built with -DT3_HIP_TIMING under hipcc it is instead the same-method HIP
// timing of cere::k6::prologue (reference). Exit 0 when every row passes.
#ifdef T3_HIP_TIMING
#include <hip/hip_runtime.h>
#include "kernels/k6_attention.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#ifndef T3_HIP_TIMING
#include "kernels/k1_primitives_vk.h"
#include "kernels/k6_prologue_vk.h"
#include "launch/vk_launch.h"
#include "launch/vk_launch_testing.h"
#endif

static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) {  // round to nearest even, NaN 0x7FFF
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fff;
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static std::vector<uint16_t> randbf(size_t n, std::mt19937 &g, float scale = 1.f) {
    std::normal_distribution<float> d(0.f, scale); std::vector<uint16_t> v(n);
    for (auto &x : v) x = f2bf(d(g));
    return v;
}
static double median(std::vector<double> t) { std::sort(t.begin(), t.end()); return t[t.size() / 2]; }

constexpr int QH = 24, KVH = 2, HD = 256, RD = 64, QW = QH * HD, KW = KVH * HD, QGW = QH * 2 * HD;
constexpr float kEps = 1e-6f, kTheta = 1e7f;
constexpr int kBatch = 100;
// Timed shapes: (tokens, prefix_len) at a 32,768-row table, slot map = identity permutation of 32,768 rows.
struct TShape { const char *name; int64_t tokens, prefix; };
static const TShape kShapes[] = {{"prologue decode 1 tok @1000", 1, 1000}, {"prologue verify 4 tok @1000", 4, 1000},
                                 {"prologue extend 512 tok @0", 512, 0}};

#ifndef T3_HIP_TIMING
using namespace cere;
static int g_fail = 0;
static void report(const char *name, bool ok, const char *detail) {
    std::printf("%-44s %s  %s\n", name, ok ? "PASS" : "FAIL", detail);
    if (!ok) ++g_fail;
}
static vk::Buffer make(vk::Context &ctx, uint64_t bytes) {
    vk::Buffer b;
    if (!ctx.alloc(bytes + 256, b)) std::exit(2);
    std::memset(b.map, 0, bytes + 256);
    return b;
}
template <class T> static vk::Buffer up(vk::Context &ctx, const std::vector<T> &h) {
    vk::Buffer b = make(ctx, h.size() * sizeof(T)); std::memcpy(b.map, h.data(), h.size() * sizeof(T)); return b;
}
template <class T> static std::vector<T> host(const vk::Buffer &b, size_t n) {
    std::vector<T> h(n); std::memcpy(h.data(), b.map, n * sizeof(T)); return h;
}
static void must(cere_status s, const char *what) { if (s != CERE_OK) { std::printf("FATAL %s returned %d\n", what, int(s)); std::exit(2); } }
static void sync(vk::Context &ctx, const std::function<cere_status()> &f, const char *what) { ctx.begin(); must(f(), what); ctx.barrier(); ctx.submit_wait(); }

static double rel_l2(const std::vector<uint16_t> &got, const std::vector<double> &ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) { const double d = bf2f(got[i]) - ref[i]; num += d * d; den += ref[i] * ref[i]; }
    return std::sqrt(num / (den > 0 ? den : 1));
}
static double rel_l2d(const std::vector<double> &got, const std::vector<double> &ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) { const double d = got[i] - ref[i]; num += d * d; den += ref[i] * ref[i]; }
    return std::sqrt(num / (den > 0 ? den : 1));
}
static uint8_t fp8_ref(float x) {  // the HIP k1 test's independent reference: nearest representable E4M3, ties to even mantissa
    uint32_t u; std::memcpy(&u, &x, 4); const uint8_t s = uint8_t((u >> 24) & 0x80);
    if (std::isnan(x)) return s | 0x7f;
    const float a = std::fabs(x);
    if (a > 464.f) return s | 0x7e;
    uint8_t best = 0; double bd = INFINITY;
    for (uint8_t c = 0; c <= 0x7e; ++c) {
        const int e = c >> 3, m = c & 7; const double v = e == 0 ? m / 8.0 * std::ldexp(1.0, -6) : (1 + m / 8.0) * std::ldexp(1.0, e - 7);
        const double d = std::fabs(v - a);
        if (d < bd || (d == bd && (c & 1) == 0)) { bd = d; best = c; }
    }
    return s | best;
}

// One head: FP64 Gemma norm (1 + w) then NeoX RoPE with exact angles (out64), and the FP32-table emulation (out32: the
// norm rounded to BF16 as the kernel does, the table's float angle with libm cosf/sinf, FP32 rotation, BF16 out).
static void ref_head(const uint16_t *x, const std::vector<uint16_t> &w, int64_t p, double *out64, double *out32) {
    double ss = 0; for (int i = 0; i < HD; ++i) { const double v = bf2f(x[i]); ss += v * v; }
    const double rs = 1.0 / std::sqrt(ss / HD + kEps);
    std::vector<double> n(HD); std::vector<float> nb(HD);
    for (int i = 0; i < HD; ++i) { n[i] = bf2f(x[i]) * rs * (1.0 + bf2f(w[i])); nb[i] = bf2f(f2bf(float(n[i]))); }
    for (int i = 0; i < HD; ++i) { out64[i] = n[i]; out32[i] = nb[i]; }
    for (int i = 0; i < RD / 2; ++i) {
        const double an = double(p) * (1.0 / std::pow(double(kTheta), double(2 * i) / RD));
        out64[i] = n[i] * std::cos(an) - n[i + RD / 2] * std::sin(an);
        out64[i + RD / 2] = n[i + RD / 2] * std::cos(an) + n[i] * std::sin(an);
        const float inv = 1.0f / std::pow(kTheta, float(2 * i) / float(RD)), af = float(p) * inv, c = std::cos(af), s = std::sin(af);
        out32[i] = bf2f(f2bf(nb[i] * c - nb[i + RD / 2] * s));
        out32[i + RD / 2] = bf2f(f2bf(nb[i + RD / 2] * c + nb[i] * s));
    }
}

int main(int argc, char **argv) {
    auto ctx_owner = cere::vk::testing::create_or_die(argc > 1 ? argv[1] : "build/vulkan-t3/spv");
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str());
    std::mt19937 g(29);
    using vkk6::prologue;

    // ---- 1. the CUDA suite's position rows (200,000-row table, one token, permuted slots)
    {
        const int64_t CTX = 200000;
        auto tab = make(ctx, uint64_t(CTX) * RD * 4);
        sync(ctx, [&] { return vkk1::rope_table(ctx, tab.addr, CTX, RD, kTheta); }, "rope_table");
        auto qw = randbf(HD, g, 0.3f), kw = randbf(HD, g, 0.3f);
        auto dqw = up(ctx, qw), dkw = up(ctx, kw);
        auto kc = make(ctx, uint64_t(CTX) * KW), vc = make(ctx, uint64_t(CTX) * KW);
        auto qo = make(ctx, QW * 2), ko = make(ctx, KW * 2), go = make(ctx, QW * 2);
        for (int64_t position : {int64_t(2047), int64_t(2048), int64_t(4096), int64_t(72000), int64_t(199999)}) {
            auto qg = randbf(QGW, g, 4.f), k = randbf(KW, g, 4.f), v = randbf(KW, g, 1.f);
            std::vector<int64_t> perm(CTX); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), g);
            std::vector<int64_t> slots(perm.begin(), perm.begin() + position + 1);
            std::vector<uint8_t> kc0(size_t(CTX) * KW), vc0(size_t(CTX) * KW);
            for (size_t i = 0; i < kc0.size(); ++i) { kc0[i] = uint8_t(g()); vc0[i] = uint8_t(g()); }
            std::memcpy(kc.map, kc0.data(), kc0.size()); std::memcpy(vc.map, vc0.data(), vc0.size());
            std::vector<int64_t> pos1 = {position};
            auto dqg = up(ctx, qg), dk = up(ctx, k), dv = up(ctx, v), ds = up(ctx, slots), dp = up(ctx, pos1);
            sync(ctx, [&] { return prologue(ctx, dqg.addr, QGW, dk.addr, KW, dv.addr, KW, dqw.addr, dkw.addr, kEps, tab.addr, CTX, dp.addr, 0, ds.addr,
                                            1, position, kc.addr, vc.addr, qo.addr, ko.addr, go.addr); }, "prologue");
            auto q = host<uint16_t>(qo, QW), kk = host<uint16_t>(ko, KW), gt = host<uint16_t>(go, QW);
            std::vector<double> q64(QW), q32(QW), k64(KW), k32(KW);
            for (int h = 0; h < QH; ++h) ref_head(&qg[size_t(2 * h) * HD], qw, position, &q64[h * HD], &q32[h * HD]);
            for (int h = 0; h < KVH; ++h) ref_head(&k[size_t(h) * HD], kw, position, &k64[h * HD], &k32[h * HD]);
            const double cq = rel_l2d(q32, q64), ck = rel_l2d(k32, k64), lq = std::max(4e-3, 2 * cq), lk = std::max(4e-3, 2 * ck);
            const double eq = rel_l2(q, q64), ek = rel_l2(kk, k64);
            bool gate_ok = true;
            for (int h = 0; h < QH; ++h) gate_ok = gate_ok && std::memcmp(&gt[h * HD], &qg[size_t(2 * h + 1) * HD], HD * 2) == 0;
            // cache: the token's row = fp8(k_out) / fp8(v) at slots[position]; every other row unchanged
            auto kca = host<uint8_t>(kc, kc0.size()), vca = host<uint8_t>(vc, vc0.size());
            const int64_t row = slots[position];
            for (int i = 0; i < KW; ++i) { kc0[size_t(row) * KW + i] = fp8_ref(bf2f(kk[i])); vc0[size_t(row) * KW + i] = fp8_ref(bf2f(v[i])); }
            int64_t kbad = 0, vbad = 0;
            for (size_t i = 0; i < kc0.size(); ++i) { kbad += kca[i] != kc0[i]; vbad += vca[i] != vc0[i]; }
            char name[64], d[220];
            std::snprintf(name, sizeof name, "prologue pos %ld (200,000 ctx)", long(position));
            std::snprintf(d, sizeof d, "q rel-L2 %.2e (limit %.2e, c %.2e), k %.2e (limit %.2e, c %.2e); gate %s; cache rows differ k %ld v %ld of %ld",
                          eq, lq, cq, ek, lk, ck, gate_ok ? "bit-exact" : "DIFFERS", long(kbad), long(vbad), long(CTX) * KW);
            report(name, eq <= lq && ek <= lk && gate_ok && kbad == 0 && vbad == 0, d);
            for (auto *b : {&dqg, &dk, &dv, &ds, &dp}) ctx.free(*b);
        }
        for (auto *b : {&tab, &dqw, &dkw, &kc, &vc, &qo, &ko, &go}) ctx.free(*b);
    }

    // ---- 2. + 3. composition == the unfused vkk1 sequence; repeats; positions null == explicit
    {
        const int64_t CTX = 4096, CROWS = 512;
        auto tab = make(ctx, uint64_t(CTX) * RD * 4);
        sync(ctx, [&] { return vkk1::rope_table(ctx, tab.addr, CTX, RD, kTheta); }, "rope_table");
        auto dqw = up(ctx, randbf(HD, g, 0.3f)), dkw = up(ctx, randbf(HD, g, 0.3f));
        std::vector<int64_t> perm(CROWS); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), g);
        auto ds = up(ctx, perm);
        std::vector<uint8_t> init(size_t(CROWS) * KW); for (auto &b : init) b = uint8_t(g());
        const int64_t TMAX = 37;
        auto qo = make(ctx, TMAX * QW * 2), ko = make(ctx, TMAX * KW * 2), go = make(ctx, TMAX * QW * 2), kc = make(ctx, CROWS * KW), vc = make(ctx, CROWS * KW);
        auto qo2 = make(ctx, TMAX * QW * 2), ko2 = make(ctx, TMAX * KW * 2), go2 = make(ctx, TMAX * QW * 2), kc2 = make(ctx, CROWS * KW), vc2 = make(ctx, CROWS * KW);
        struct Case { const char *name; int64_t tokens, prefix; bool slots; };
        const Case cases[] = {{"decode 1 tok @300, slots", 1, 300, true}, {"decode 1 tok @300, identity", 1, 300, false},
                              {"verify 4 tok @123, slots", 4, 123, true}, {"verify 4 tok @123, identity", 4, 123, false},
                              {"extend 37 tok @5, slots", 37, 5, true}, {"extend 37 tok @5, identity", 37, 5, false}};
        for (const Case &cs : cases) {
            const int64_t T = cs.tokens;
            // strided inputs as the forward hands them (q_gate / k / v slices of one qkv row of 6,144 * 2 + 512 * 2)
            const int64_t S = QGW + 2 * KW;
            auto qkv = randbf(size_t(T) * S, g, 4.f);
            for (int64_t t = 0; t < T; ++t) for (int i = 0; i < KW; ++i) qkv[size_t(t) * S + QGW + KW + i] = f2bf(bf2f(qkv[size_t(t) * S + QGW + KW + i]) * 30.f);  // v up to ~+-500: saturation
            auto dqkv = up(ctx, qkv);
            const uint64_t qg = dqkv.addr, kk = dqkv.addr + uint64_t(QGW) * 2, vv = dqkv.addr + uint64_t(QGW + KW) * 2;
            const uint64_t sl = cs.slots ? ds.addr : 0;
            auto reset = [&] { std::memcpy(kc.map, init.data(), init.size()); std::memcpy(vc.map, init.data(), init.size());
                               std::memcpy(kc2.map, init.data(), init.size()); std::memcpy(vc2.map, init.data(), init.size()); };
            reset();
            ctx.begin();
            must(prologue(ctx, qg, S, kk, S, vv, S, dqw.addr, dkw.addr, kEps, tab.addr, CTX, 0, 0, sl, T, cs.prefix, kc.addr, vc.addr, qo.addr, ko.addr, go.addr), "prologue");
            ctx.barrier(); ctx.submit_wait();
            sync(ctx, [&] { return vkk1::qk_norm_rope_gate(ctx, qg, S, kk, S, dqw.addr, dkw.addr, tab.addr, 0, 0, cs.prefix, T, QH, KVH, HD, RD, kEps,
                                                           qo2.addr, ko2.addr, go2.addr); }, "qk_norm_rope_gate");
            sync(ctx, [&] { return vkk1::kv_store_fp8(ctx, ko2.addr, KW, vv, S, T, KW, kc2.addr, vc2.addr, sl ? sl + uint64_t(cs.prefix) * 8 : 0, cs.prefix); }, "kv_store");
            const bool eq = host<uint16_t>(qo, T * QW) == host<uint16_t>(qo2, T * QW) && host<uint16_t>(ko, T * KW) == host<uint16_t>(ko2, T * KW) &&
                            host<uint16_t>(go, T * QW) == host<uint16_t>(go2, T * QW) && host<uint8_t>(kc, CROWS * KW) == host<uint8_t>(kc2, CROWS * KW) &&
                            host<uint8_t>(vc, CROWS * KW) == host<uint8_t>(vc2, CROWS * KW);
            const bool touched = host<uint8_t>(kc, CROWS * KW) != init;
            // repeat (same cache start) and positions explicit == null
            auto q1 = host<uint16_t>(qo, T * QW); auto c1 = host<uint8_t>(kc, CROWS * KW); auto v1 = host<uint8_t>(vc, CROWS * KW);
            std::vector<int64_t> pos(T); for (int64_t t = 0; t < T; ++t) pos[t] = cs.prefix + t; auto dp = up(ctx, pos);
            reset();
            sync(ctx, [&] { return prologue(ctx, qg, S, kk, S, vv, S, dqw.addr, dkw.addr, kEps, tab.addr, CTX, dp.addr, 0, sl, T, cs.prefix, kc.addr, vc.addr,
                                            qo.addr, ko.addr, go.addr); }, "prologue positions");
            const bool same = host<uint16_t>(qo, T * QW) == q1 && host<uint8_t>(kc, CROWS * KW) == c1 && host<uint8_t>(vc, CROWS * KW) == v1;
            char name[64], d[200];
            std::snprintf(name, sizeof name, "== unfused vkk1: %s", cs.name);
            std::snprintf(d, sizeof d, "q, k, gate, k_cache, v_cache bit-equal to qk_norm_rope_gate -> kv_store_fp8: %s (cache written: %s); repeat with positions[t] = prefix + t identical: %s",
                          eq ? "yes" : "NO", touched ? "yes" : "NO", same ? "yes" : "NO");
            report(name, eq && touched && same, d);
            ctx.free(dqkv); ctx.free(dp);
        }
        // ---- 4. negative rows
        {
            std::vector<uint16_t> qkv1 = randbf(QGW + 2 * KW, g); auto d1 = up(ctx, qkv1);
            const int64_t S = QGW + 2 * KW;
            const uint64_t qg = d1.addr, kk = d1.addr + uint64_t(QGW) * 2, vv = d1.addr + uint64_t(QGW + KW) * 2;
            std::memcpy(kc.map, init.data(), init.size()); std::memset(qo.map, 0x5A, QW * 2);
            auto qo_before = host<uint16_t>(qo, QW);
            int tried = 0, ok = 0;
            auto expect = [&](cere_status got, cere_status want) { ++tried; ok += got == want; };
            ctx.begin();
            auto P = [&](uint64_t q_gate, int64_t qgs, uint64_t k, uint64_t v, int64_t vs, uint64_t positions3, uint64_t slots, int64_t tokens, int64_t prefix,
                         int64_t context, uint64_t kcache) {
                return prologue(ctx, q_gate, qgs, k, S, v, vs, dqw.addr, dkw.addr, kEps, tab.addr, context, 0, positions3, slots, tokens, prefix, kcache, vc.addr,
                                qo.addr, ko.addr, go.addr);
            };
            expect(P(qg, S, kk, vv, S, ds.addr, 0, 1, 10, CTX, kc.addr), CERE_ERR_UNSUPPORTED);        // mRoPE
            expect(P(qg, S, kk, vv, S, 0, 0, 2, CTX - 1, CTX, kc.addr), CERE_ERR_ARGUMENT);              // prefix + tokens > context
            expect(P(qg, S, kk, vv, S, 0, 0, 1, 0, 0, kc.addr), CERE_ERR_ARGUMENT);                      // context 0
            expect(P(qg, S, kk, vv, S, 0, 0, -1, 0, CTX, kc.addr), CERE_ERR_ARGUMENT);                   // tokens < 0
            expect(P(qg, S, kk, vv, S, 0, 0, 1, -1, CTX, kc.addr), CERE_ERR_ARGUMENT);                   // prefix < 0
            expect(P(0, S, kk, vv, S, 0, 0, 1, 0, CTX, kc.addr), CERE_ERR_ARGUMENT);                     // null q_gate
            expect(P(qg, S, kk, vv, S, 0, 0, 1, 0, CTX, 0), CERE_ERR_ARGUMENT);                          // null k_cache
            expect(P(qg, S, kk, vv, S - 4, 0, 0, 1, 0, CTX, kc.addr), CERE_ERR_ARGUMENT);                // v stride % 8 (the store's rule)
            expect(P(qg, S, kk, vv + 2, S, 0, 0, 1, 0, CTX, kc.addr), CERE_ERR_ARGUMENT);                // v misaligned (the store's rule)
            expect(P(qg, S, kk, vv, S, 0, ds.addr + 4, 1, 0, CTX, kc.addr), CERE_ERR_ARGUMENT);          // slots misaligned
            expect(P(qg, QGW - 8, kk, vv, S, 0, 0, 1, 0, CTX, kc.addr), CERE_ERR_ARGUMENT);              // q_gate stride < 12,288
            expect(P(qg, S, kk, vv, S, 0, 0, 65536, 0, 1 << 20, kc.addr), CERE_ERR_ARGUMENT);            // tokens > 65,535
            expect(P(qg, S, kk, vv, S, 0, 0, 0, 0, CTX, kc.addr), CERE_OK);                              // tokens 0: no-op
            ctx.barrier(); ctx.submit_wait();
            const bool untouched = host<uint16_t>(qo, QW) == qo_before && host<uint8_t>(kc, CROWS * KW) == init;
            char d[200];
            std::snprintf(d, sizeof d, "%d of %d calls returned the expected status (positions3 UNSUPPORTED, 11 ARGUMENT, tokens 0 OK); outputs and cache untouched: %s",
                          ok, tried, untouched ? "yes" : "NO");
            report("negative: refusals record nothing", ok == tried && untouched, d);
            const bool bytes_ok = vkk6::prologue_bytes(1) == 12288 + 1024 + 12288 && vkk6::prologue_bytes(5) == 61440 + 5120 + 61440 &&
                                  vkk6::prologue_bytes(4096) == size_t(4096) * (12288 + 1024 + 12288) && vkk6::prologue_bytes(0) == 0 &&
                                  vkk6::prologue_bytes(3) == 36864 + 3072 + 36864;
            report("prologue_bytes == CUDA formula", bytes_ok, "align256(T*6144*2) + align256(T*512*2) + align256(T*6144*2); 0 for T <= 0");
            ctx.free(d1);
        }
        for (auto *b : {&tab, &dqw, &dkw, &ds, &qo, &ko, &go, &kc, &vc, &qo2, &ko2, &go2, &kc2, &vc2}) ctx.free(*b);
    }
    std::printf("contract rows: %s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);

    {   // decode-shape timing (K1 batch method)
        std::mt19937 gt(7);
        const int64_t CTX = 32768, TM = 512;
        auto tab = make(ctx, uint64_t(CTX) * RD * 4);
        sync(ctx, [&] { return vkk1::rope_table(ctx, tab.addr, CTX, RD, kTheta); }, "rope_table");
        std::vector<int64_t> ident(CTX); std::iota(ident.begin(), ident.end(), 0); auto ds = up(ctx, ident);
        auto qg = up(ctx, randbf(TM * QGW, gt, 4.f)), k = up(ctx, randbf(TM * KW, gt, 4.f)), v = up(ctx, randbf(TM * KW, gt));
        auto qw = up(ctx, randbf(HD, gt, 0.3f)), kw = up(ctx, randbf(HD, gt, 0.3f));
        auto kc = make(ctx, uint64_t(CTX) * KW), vc = make(ctx, uint64_t(CTX) * KW);
        auto qo = make(ctx, TM * QW * 2), ko = make(ctx, TM * KW * 2), go = make(ctx, TM * QW * 2);
        std::printf("\ndecode-shape timing: us per call = (GPU time between two timestamps around %d dependent calls, a barrier after each) / %d,\n"
                    "median of 7 such batches [min..max]; 3 untimed warm-up batches first\n", kBatch, kBatch);
        for (const TShape &s : kShapes) {
            auto call = [&] { return prologue(ctx, qg.addr, QGW, k.addr, KW, v.addr, KW, qw.addr, kw.addr, kEps, tab.addr, CTX, 0, 0, ds.addr, s.tokens, s.prefix,
                                              kc.addr, vc.addr, qo.addr, ko.addr, go.addr); };
            const int reps = 10;
            ctx.begin();
            for (int c = 0; c < reps; ++c) {
                ctx.stamp(2 * c);
                for (int i = 0; i < kBatch; ++i) { must(call(), "prologue"); ctx.barrier(); }
                ctx.stamp(2 * c + 1);
            }
            ctx.submit_wait();
            std::vector<double> t; for (int c = 3; c < reps; ++c) t.push_back(ctx.stamp_ms(2 * c, 2 * c + 1) * 1e3 / kBatch);
            std::printf("TIME vulkan | %-34s | %9.2f us [%.2f..%.2f]\n", s.name, median(t), *std::min_element(t.begin(), t.end()), *std::max_element(t.begin(), t.end()));
        }
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

#else  // T3_HIP_TIMING: the same shapes and data through cere::k6::prologue (hipified), hipEvents around each batch
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
using bf16 = __hip_bfloat16;
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
static bf16 *B(uint16_t *p) { return reinterpret_cast<bf16 *>(p); }
int main() {
    hipDeviceProp_t prop; HCK(hipGetDeviceProperties(&prop, 0)); std::printf("device %s\n", prop.gcnArchName);
    std::mt19937 gt(7);
    const int64_t CTX = 32768, TM = 512;
    float *tab; HCK(hipMalloc(&tab, size_t(CTX) * RD * 4)); HCK(cere::k1::rope_table(tab, CTX, RD, kTheta, nullptr));
    std::vector<int64_t> ident(CTX); std::iota(ident.begin(), ident.end(), 0); int64_t *ds = dev(ident);
    uint16_t *qg = dev(randbf(TM * QGW, gt, 4.f)), *k = dev(randbf(TM * KW, gt, 4.f)), *v = dev(randbf(TM * KW, gt));
    uint16_t *qw = dev(randbf(HD, gt, 0.3f)), *kw = dev(randbf(HD, gt, 0.3f));
    uint8_t *kc, *vc; HCK(hipMalloc(&kc, size_t(CTX) * KW)); HCK(hipMalloc(&vc, size_t(CTX) * KW));
    uint16_t *qo, *ko, *go; HCK(hipMalloc(&qo, TM * QW * 2)); HCK(hipMalloc(&ko, TM * KW * 2)); HCK(hipMalloc(&go, TM * QW * 2));
    HCK(hipDeviceSynchronize());
    std::printf("decode-shape timing: us per call = hipEvent time of %d back-to-back calls on one stream / %d, median of 7 batches [min..max]\n", kBatch, kBatch);
    hipStream_t st; HCK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    for (const TShape &s : kShapes) {
        auto call = [&] { return cere::k6::prologue(B(qg), QGW, B(k), KW, B(v), KW, B(qw), B(kw), kEps, tab, CTX, nullptr, nullptr, ds, s.tokens, s.prefix,
                                                    kc, vc, B(qo), B(ko), B(go), st); };
        const int reps = 10; std::vector<hipEvent_t> ev(2 * reps); for (auto &e : ev) HCK(hipEventCreate(&e));
        for (int c = 0; c < reps; ++c) {
            HCK(hipEventRecord(ev[2 * c], st));
            for (int i = 0; i < kBatch; ++i) HCK(call());
            HCK(hipEventRecord(ev[2 * c + 1], st));
        }
        HCK(hipStreamSynchronize(st));
        std::vector<double> t; for (int c = 3; c < reps; ++c) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * c], ev[2 * c + 1])); t.push_back(ms * 1e3 / kBatch); }
        std::printf("TIME hip-stream | %-34s | %9.2f us [%.2f..%.2f]\n", s.name, median(t), *std::min_element(t.begin(), t.end()), *std::max_element(t.begin(), t.end()));
    }
    return 0;
}
#endif
