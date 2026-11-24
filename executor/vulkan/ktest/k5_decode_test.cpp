// VulkanAmd P2: executor/hip/ktest/k5_decode_test.cpp's contract rows on the Vulkan K5 GDN decode
// (shaders/k5_gdn_decode.comp), same seeds, data and bounds as the HIP test:
//  1. delta_rule_decode over 8 consecutive tokens: o and state rel-L2 vs an FP64 reference that rests the state in BF16
//     between tokens (limit 8e-3, the HIP test's; HIP measured 1.69e-3 / 1.66e-3).
//  2. delta_rule_decode_rows, 3 chains of 2/3/1 rows in one call: every chain's outputs, final state and undo snapshots
//     bit-equal to the same rows as sequential single-token calls.
// Extra rows (the spec's generic contract, not in the HIP test): the rows call repeated from the same inputs is
// bit-identical; malformed chains / misaligned addresses are refused with CERE_ERR_ARGUMENT.
// Then timing at decode shapes (GPU timestamps): one token, and one 8-row verify chain; "call" = one dispatch between
// its own timestamps (median of 15), "stream" = 50 back-to-back calls with a barrier between them / 50 (median of 9).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "launch/vk_launch_testing.h"
#include "kernels/k5_gdn_vk.h"
#include "ktest_util.h"

using kt::bf2f;
using kt::f2bf;
namespace k5 = cere::vkk5;
static int g_fail = 0;

constexpr int KH = 16, VH = 48, D = 128;
struct Tok { std::vector<uint16_t> q, k, v, a, b; };

static void report(const char *n, bool ok, const char *d) { std::printf("%-30s %s  %s\n", n, ok ? "PASS" : "FAIL", d); if (!ok) ++g_fail; }

// Median / min / max of `reps` timed single calls, and the median per-call time of `samples` x `burst` back-to-back calls.
struct Times { double call_med, call_min, call_max, stream_med; };
static Times time_it(cere::vk::Context &ctx, const std::function<void()> &rec, int reps = 15, int samples = 9, int burst = 50) {
    ctx.begin();
    for (int i = 0; i < 50; ++i) { rec(); ctx.barrier(); }
    for (int i = 0; i < reps; ++i) { ctx.stamp(2 * i); rec(); ctx.stamp(2 * i + 1); ctx.barrier(); }
    ctx.submit_wait();
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
    std::sort(t.begin(), t.end());
    ctx.begin();
    for (int s = 0; s < samples; ++s) {
        ctx.stamp(2 * s);
        for (int i = 0; i < burst; ++i) { rec(); ctx.barrier(); }
        ctx.stamp(2 * s + 1);
    }
    ctx.submit_wait();
    std::vector<double> u;
    for (int s = 0; s < samples; ++s) u.push_back(ctx.stamp_ms(2 * s, 2 * s + 1) / burst);
    std::sort(u.begin(), u.end());
    return {t[t.size() / 2], t.front(), t.back(), u[u.size() / 2]};
}
static void print_times(const char *n, const Times &t) {
    std::printf("    %-34s call %7.2f us (min %.2f, max %.2f)  stream %7.2f us/call\n", n, t.call_med * 1e3, t.call_min * 1e3,
                t.call_max * 1e3, t.stream_med * 1e3);
}

int main(int argc, char **argv) {
    auto ctx_owner = cere::vk::testing::create_or_die(argc > 1 ? argv[1] : "build/vulkan/spv");
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str());
    std::mt19937 g(5); std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    std::vector<float> A_log(VH); for (auto &x : A_log) x = std::log(0.5f + std::fabs(nd(g)));
    auto dt_bias = rnd(VH, 0.5f);
    auto state0 = rnd(size_t(VH) * D * D, 0.05f);
    const float scale = 1.f / std::sqrt(float(D));
    const int T = 8; std::vector<Tok> toks(T);
    for (auto &t : toks) { t.q = rnd(KH * D, 1.f); t.k = rnd(KH * D, 1.f); t.v = rnd(VH * D, 1.f); t.a = rnd(VH, 1.f); t.b = rnd(VH, 1.f); }
    const size_t SB = size_t(VH) * D * D * 2;  // state bytes

    auto dA = kt::upload(ctx, A_log), ddt = kt::upload(ctx, dt_bias), dstate = kt::upload(ctx, state0);
    auto dq = kt::make(ctx, KH * D * 2), dk = kt::make(ctx, KH * D * 2), dv = kt::make(ctx, VH * D * 2);
    auto da = kt::make(ctx, VH * 2), db = kt::make(ctx, VH * 2), dout = kt::make(ctx, VH * D * 2);
    auto one_sync = [&](const k5::DecodeArgs &a) { ctx.begin(); kt::must(k5::delta_rule_decode(ctx, a), "delta_rule_decode"); ctx.barrier(); ctx.submit_wait(); };

    // --- 1. sequential decode vs FP64 reference (state at rest in BF16)
    std::vector<uint16_t> ref_state = state0;
    double worst_o = 0, worst_s = 0;
    for (int t = 0; t < T; ++t) {
        auto &tk = toks[t];
        std::memcpy(dq.map, tk.q.data(), KH * D * 2); std::memcpy(dk.map, tk.k.data(), KH * D * 2);
        std::memcpy(dv.map, tk.v.data(), VH * D * 2); std::memcpy(da.map, tk.a.data(), VH * 2); std::memcpy(db.map, tk.b.data(), VH * 2);
        one_sync({dq.addr, dk.addr, dv.addr, da.addr, db.addr, dA.addr, ddt.addr, dstate.addr, dout.addr, scale});
        auto o = kt::host<uint16_t>(dout, VH * D); auto st = kt::host<uint16_t>(dstate, size_t(VH) * D * D);
        std::vector<double> qn(KH * D), kn(KH * D);
        for (int h = 0; h < KH; ++h) {
            double sq = 0, sk = 0; for (int i = 0; i < D; ++i) { sq += double(bf2f(tk.q[h * D + i])) * bf2f(tk.q[h * D + i]); sk += double(bf2f(tk.k[h * D + i])) * bf2f(tk.k[h * D + i]); }
            for (int i = 0; i < D; ++i) { qn[h * D + i] = bf2f(tk.q[h * D + i]) / std::sqrt(sq + 1e-6) * scale; kn[h * D + i] = bf2f(tk.k[h * D + i]) / std::sqrt(sk + 1e-6); }
        }
        double no = 0, dn = 0, ns = 0, ds = 0;
        for (int h = 0; h < VH; ++h) {
            const int kh = h / 3; const double x = double(bf2f(tk.a[h])) + bf2f(dt_bias[h]);
            const double sp = x > 20 ? x : std::log1p(std::exp(x));
            const double decay = std::exp(-std::exp(double(A_log[h])) * sp), beta = 1 / (1 + std::exp(-double(bf2f(tk.b[h]))));
            std::vector<double> S(D * D);
            for (int i = 0; i < D * D; ++i) S[i] = bf2f(ref_state[size_t(h) * D * D + i]) * decay;
            for (int vi = 0; vi < D; ++vi) {
                double sk = 0; for (int ki = 0; ki < D; ++ki) sk += S[vi * D + ki] * kn[kh * D + ki];
                const double u = beta * (bf2f(tk.v[h * D + vi]) - sk);
                for (int ki = 0; ki < D; ++ki) S[vi * D + ki] += u * kn[kh * D + ki];
            }
            for (int vi = 0; vi < D; ++vi) {
                double ov = 0; for (int ki = 0; ki < D; ++ki) ov += S[vi * D + ki] * qn[kh * D + ki];
                const double got = bf2f(o[h * D + vi]); no += (got - ov) * (got - ov); dn += ov * ov;
            }
            for (int i = 0; i < D * D; ++i) {
                const double got = bf2f(st[size_t(h) * D * D + i]); ns += (got - S[i]) * (got - S[i]); ds += S[i] * S[i];
            }
        }
        worst_o = std::fmax(worst_o, std::sqrt(no / dn)); worst_s = std::fmax(worst_s, std::sqrt(ns / ds));
        ref_state = st;  // follow the kernel's own BF16 state, so each step measures one step's error
    }
    { char d[120]; std::snprintf(d, sizeof d, "8 tokens: worst o rel-L2 %.2e, state rel-L2 %.2e (limit 8e-3)", worst_o, worst_s);
      report("delta_rule_decode", worst_o <= 8e-3 && worst_s <= 8e-3, d); }

    // --- 2. rows: 3 chains (2, 3, 1 rows) in one call == sequential single-token calls
    const int chains = 3, rows_of[3] = {2, 3, 1}, R = 6;
    const int64_t qkv_stride = KH * D * 2 + VH * D;  // q, k, v share one row stride: each row packed [q | k | v]
    std::vector<uint16_t> QKV(size_t(T) * qkv_stride), Aa, Bb;
    for (int r = 0; r < T; ++r) {
        auto &tk = toks[r];
        std::memcpy(&QKV[r * qkv_stride], tk.q.data(), KH * D * 2); std::memcpy(&QKV[r * qkv_stride + KH * D], tk.k.data(), KH * D * 2);
        std::memcpy(&QKV[r * qkv_stride + 2 * KH * D], tk.v.data(), VH * D * 2);
        Aa.insert(Aa.end(), tk.a.begin(), tk.a.end()); Bb.insert(Bb.end(), tk.b.begin(), tk.b.end());
    }
    auto dqkv = kt::upload(ctx, QKV), dAa = kt::upload(ctx, Aa), dBb = kt::upload(ctx, Bb);
    auto dO = kt::make(ctx, size_t(T) * VH * D * 2), table = kt::make(ctx, 4096);
    std::vector<cere::vk::Buffer> st(T), saves(T);
    for (int c = 0; c < T; ++c) st[c] = kt::upload(ctx, state0);
    for (int r = 0; r < T; ++r) saves[r] = kt::make(ctx, SB);
    auto rows_args = [&](int nchains, const int *rows) {
        k5::DecodeRowsArgs ra{};
        ra.step = {dqkv.addr, dqkv.addr + KH * D * 2, dqkv.addr + 2 * KH * D * 2, dAa.addr, dBb.addr, dA.addr, ddt.addr, 0, dO.addr, scale};
        ra.chains = nchains; ra.qkv_stride = qkv_stride; ra.ab_stride = VH; ra.o_stride = VH * D;
        for (int c = 0, f = 0; c < nchains; ++c) { ra.first[c] = f; ra.rows[c] = rows[c]; ra.states[c] = st[c].addr; f += rows[c]; }
        for (int r = 0; r < k5::kMaxRows; ++r) ra.saves[r] = r < T ? saves[r].addr : 0;
        return ra;
    };
    auto rows_sync = [&](const k5::DecodeRowsArgs &ra) {
        for (int c = 0; c < ra.chains; ++c) std::memcpy(st[c].map, state0.data(), SB);
        ctx.begin(); kt::must(k5::delta_rule_decode_rows(ctx, ra, table, 0), "delta_rule_decode_rows"); ctx.barrier(); ctx.submit_wait();
    };
    const auto ra = rows_args(chains, rows_of);
    rows_sync(ra);
    auto O = kt::host<uint16_t>(dO, size_t(R) * VH * D);
    std::vector<std::vector<uint16_t>> snap;  // every state and save the call wrote, for the repeat row
    for (int c = 0; c < chains; ++c) snap.push_back(kt::host<uint16_t>(st[c], SB / 2));
    for (int r = 0; r < R; ++r) snap.push_back(kt::host<uint16_t>(saves[r], SB / 2));
    bool equal = true;
    auto s1 = kt::make(ctx, SB);
    for (int c = 0, f = 0; c < chains; ++c) {
        std::memcpy(s1.map, state0.data(), SB);
        for (int i = 0; i < rows_of[c]; ++i) {
            const int r = f + i;
            const uint64_t row = dqkv.addr + uint64_t(r) * qkv_stride * 2;
            one_sync({row, row + KH * D * 2, row + 2 * KH * D * 2, dAa.addr + r * VH * 2, dBb.addr + r * VH * 2, dA.addr, ddt.addr, s1.addr, dout.addr, scale});
            equal = equal && std::memcmp(dout.map, &O[size_t(r) * VH * D], VH * D * 2) == 0;
            equal = equal && std::memcmp(s1.map, (i + 1 < rows_of[c] ? saves[r] : st[c]).map, SB) == 0;
        }
        f += rows_of[c];
    }
    report("delta_rule_decode_rows", equal, equal ? "3 chains (2/3/1 rows): outputs, undo saves and final states bit-equal to sequential launches"
                                                   : "3 chains (2/3/1 rows): outputs, undo saves and final states DIFFER from sequential launches");

    // --- extra: repeat bit-identical; argument refusals
    {
        for (int r = 0; r < R; ++r) std::memset(saves[r].map, 0, SB);
        std::memset(dO.map, 0, size_t(R) * VH * D * 2);
        rows_sync(ra);
        bool same = std::memcmp(dO.map, O.data(), size_t(R) * VH * D * 2) == 0;
        for (int c = 0; c < chains; ++c) same = same && std::memcmp(st[c].map, snap[c].data(), SB) == 0;
        for (int r = 0; r < R; ++r) same = same && std::memcmp(saves[r].map, snap[chains + r].data(), SB) == 0;
        report("delta_rule_decode_rows repeat", same, same ? "(extra) second call bit-identical (outputs, states, saves)" : "(extra) second call DIFFERS");
    }
    {
        ctx.begin();
        int refused = 0, tried = 0;
        auto bad = [&](k5::DecodeRowsArgs a) { ++tried; refused += k5::delta_rule_decode_rows(ctx, a, table, 0) == CERE_ERR_ARGUMENT; };
        auto a1 = ra; a1.first[1] = 3; bad(a1);                       // not back to back
        auto a2 = ra; a2.chains = 0; bad(a2);                         // no chain
        auto a3 = ra; a3.rows[2] = 15; bad(a3);                       // > 16 rows in all
        auto a4 = ra; a4.saves[0] = 0; bad(a4);                       // a row that needs a save has none
        auto a5 = ra; a5.qkv_stride = 100; bad(a5);                   // stride overlaps the row
        auto a6 = ra; a6.step.q += 2; bad(a6);                        // misaligned
        auto a7 = ra; ++tried; refused += k5::delta_rule_decode_rows(ctx, a7, table, 8) == CERE_ERR_ARGUMENT;  // table offset
        k5::DecodeArgs one{dq.addr, dk.addr, dv.addr, da.addr, db.addr, dA.addr, ddt.addr, dstate.addr + 4, dout.addr, scale};
        ++tried; refused += k5::delta_rule_decode(ctx, one) == CERE_ERR_ARGUMENT;
        ctx.submit_wait();
        char d[100]; std::snprintf(d, sizeof d, "(extra) %d of %d malformed calls refused with CERE_ERR_ARGUMENT", refused, tried);
        report("argument refusals", refused == tried, d);
    }

    // --- timing at decode shapes
    std::printf("timing (GPU timestamps; state %.1f MB read + written per token):\n", SB / 1e6);
    {
        std::memcpy(dstate.map, state0.data(), SB);
        const k5::DecodeArgs a{dq.addr, dk.addr, dv.addr, da.addr, db.addr, dA.addr, ddt.addr, dstate.addr, dout.addr, scale};
        print_times("delta_rule_decode (1 token)", time_it(ctx, [&] { kt::must(k5::delta_rule_decode(ctx, a), "decode"); }));
        const int eight[1] = {8};
        const auto r8 = rows_args(1, eight);
        print_times("delta_rule_decode_rows (1 chain x 8)", time_it(ctx, [&] { kt::must(k5::delta_rule_decode_rows(ctx, r8, table, 0), "rows"); }));
        // Also the sequential alternative: 8 one-token calls (what the rows call replaces).
        std::vector<k5::DecodeArgs> seq;
        for (int r = 0; r < 8; ++r) {
            const uint64_t row = dqkv.addr + uint64_t(r) * qkv_stride * 2;
            seq.push_back({row, row + KH * D * 2, row + 2 * KH * D * 2, dAa.addr + r * VH * 2, dBb.addr + r * VH * 2, dA.addr, ddt.addr, dstate.addr, dout.addr, scale});
        }
        print_times("8 x delta_rule_decode (sequential)", time_it(ctx, [&] {
            for (int r = 0; r < 8; ++r) { kt::must(k5::delta_rule_decode(ctx, seq[r]), "decode"); if (r < 7) ctx.barrier(); }
        }));
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
