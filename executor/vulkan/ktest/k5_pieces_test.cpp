// VulkanAmd P2: executor/hip/ktest/k5_pieces_test.cpp's contract rows on the Vulkan K5 GDN pieces (shaders/k5_gdn_conv.comp,
// k5_gdn_gating.comp, k5_gdn_l2norm.comp), same seeds, data and bounds as the HIP test: conv1d_silu vs a reference that
// rounds each tap product to BF16 and sums in FP32 (SiLU in FP64), state continuity (two calls == one call, bit for bit;
// conv1d_state_at == the state conv1d_silu leaves), the verify chains (conv1d_silu_rows == sequential one-token calls,
// bit for bit, saves included); gating and l2norm vs FP64. Limits: 4e-3 rel-L2 for BF16 outputs, 1e-5 relative for the
// F32 gating outputs. Extra rows (not in the HIP test): repeats bit-identical, argument refusals.
// Then timing at decode shapes (GPU timestamps; "call" = one call between its own timestamps, median of 15; "stream" =
// 50 back-to-back calls with a barrier between them / 50, median of 9).
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
static void report(const char *n, bool ok, const char *d) { std::printf("%-30s %s  %s\n", n, ok ? "PASS" : "FAIL", d); if (!ok) ++g_fail; }

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
    std::mt19937 g(9); std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    const int64_t T = 37, dim = 8192;  // Flash-Next's conv width: q|k|v of the GDN layer (2*16*128 + 48*128)
    auto x = rnd(T * dim, 1.f), w = rnd(dim * k5::kConvWidth, 0.5f), st0 = rnd(dim * k5::kConvState, 1.f);
    auto dx = kt::upload(ctx, x), dw = kt::upload(ctx, w);
    auto dy = kt::make(ctx, T * dim * 2), dst = kt::make(ctx, dim * k5::kConvState * 2), table = kt::make(ctx, 4096);
    const size_t SB = size_t(dim) * k5::kConvState * 2;
    // conv1d_silu calls recorded in one command buffer, each after a barrier (the state carries from one to the next).
    auto conv = [&](int64_t t0, int64_t n, uint64_t state) {
        kt::must(k5::conv1d_silu(ctx, dx.addr + uint64_t(t0) * dim * 2, dim, dw.addr, state, dy.addr + uint64_t(t0) * dim * 2, dim, n, dim), "conv1d_silu");
        ctx.barrier();
    };
    {   // one call vs reference
        std::memcpy(dst.map, st0.data(), SB);
        ctx.begin(); conv(0, T, dst.addr); ctx.submit_wait();
        auto y = kt::host<uint16_t>(dy, T * dim); auto st = kt::host<uint16_t>(dst, dim * k5::kConvState);
        double num = 0, den = 0; bool state_ok = true;
        for (int64_t t = 0; t < T; ++t) for (int64_t c = 0; c < dim; ++c) {
            float acc = 0;
            for (int j = 0; j < k5::kConvWidth; ++j) {
                const int64_t tt = t - 3 + j; const float xv = tt < 0 ? bf2f(st0[c * 3 + (tt + 3)]) : bf2f(x[tt * dim + c]);
                acc += bf2f(f2bf(xv * bf2f(w[c * k5::kConvWidth + j])));
            }
            const double ref = double(acc) / (1 + std::exp(-double(acc))), got = bf2f(y[t * dim + c]);
            num += (got - ref) * (got - ref); den += ref * ref;
        }
        for (int64_t c = 0; c < dim; ++c) for (int j = 0; j < 3; ++j) state_ok = state_ok && st[c * 3 + j] == x[(T - 3 + j) * dim + c];
        char d[100]; std::snprintf(d, sizeof d, "T=%ld dim=%ld rel-L2 %.2e (limit 4e-3), state = last 3 inputs %s", (long)T, (long)dim, std::sqrt(num / den), state_ok ? "yes" : "NO");
        report("conv1d_silu", std::sqrt(num / den) <= 4e-3 && state_ok, d);
        // split 20 + 17 == one call
        auto y_one = y;
        std::memcpy(dst.map, st0.data(), SB); std::memset(dy.map, 0, T * dim * 2);
        ctx.begin(); conv(0, 20, dst.addr); conv(20, T - 20, dst.addr); ctx.submit_wait();
        auto y_split = kt::host<uint16_t>(dy, T * dim); auto st_split = kt::host<uint16_t>(dst, dim * k5::kConvState);
        report("conv1d_silu split 20+17", y_split == y_one && st_split == st, y_split == y_one && st_split == st ? "outputs and state bit-equal to one call" : "outputs or state DIFFER from one call");
        // repeat (extra)
        std::memcpy(dst.map, st0.data(), SB); std::memset(dy.map, 0, T * dim * 2);
        ctx.begin(); conv(0, T, dst.addr); ctx.submit_wait();
        const bool rep = kt::host<uint16_t>(dy, T * dim) == y_one && kt::host<uint16_t>(dst, dim * 3) == st;
        report("conv1d_silu repeat", rep, rep ? "(extra) second call bit-identical" : "(extra) second call DIFFERS");
        // state_at
        auto dsa = kt::make(ctx, SB);
        ctx.begin(); kt::must(k5::conv1d_state_at(ctx, dx.addr, dim, dsa.addr, T, dim), "conv1d_state_at"); ctx.barrier(); ctx.submit_wait();
        report("conv1d_state_at", kt::host<uint16_t>(dsa, dim * 3) == st, "== the state conv1d_silu leaves");
    }
    {   // verify chains: rows 3/1/2 on distinct states == sequential one-token calls
        const int chains = 3, rows_of[3] = {3, 1, 2}, R = 6;
        k5::ConvChains cc{}; cc.chains = chains;
        std::vector<cere::vk::Buffer> states(chains), saves(R);
        for (int c = 0, f = 0; c < chains; ++c) { cc.first[c] = f; cc.rows[c] = rows_of[c]; states[c] = kt::upload(ctx, st0); cc.states[c] = states[c].addr; f += rows_of[c]; }
        for (int r = 0; r < R; ++r) { saves[r] = kt::make(ctx, SB); cc.saves[r] = saves[r].addr; }
        auto dyr = kt::make(ctx, R * dim * 2);
        ctx.begin(); kt::must(k5::conv1d_silu_rows(ctx, dx.addr, dim, dw.addr, cc, dyr.addr, dim, dim, table, 0), "conv1d_silu_rows"); ctx.barrier(); ctx.submit_wait();
        auto Y = kt::host<uint16_t>(dyr, R * dim); bool eq = true;
        auto s1 = kt::make(ctx, SB);
        for (int c = 0, f = 0; c < chains; ++c) {
            std::memcpy(s1.map, st0.data(), SB);
            for (int i = 0; i < rows_of[c]; ++i) {
                const int r = f + i;
                ctx.begin(); kt::must(k5::conv1d_silu(ctx, dx.addr + uint64_t(r) * dim * 2, dim, dw.addr, s1.addr, dy.addr, dim, 1, dim), "conv1d_silu"); ctx.barrier(); ctx.submit_wait();
                eq = eq && std::memcmp(dy.map, &Y[r * dim], dim * 2) == 0;
                eq = eq && std::memcmp(s1.map, (i + 1 < rows_of[c] ? saves[r] : states[c]).map, SB) == 0;
            }
            f += rows_of[c];
        }
        report("conv1d_silu_rows", eq, eq ? "3 chains (3/1/2): outputs, saves, states bit-equal to sequential" : "3 chains (3/1/2): outputs, saves or states DIFFER from sequential");
        // refusals (extra)
        ctx.begin();
        int refused = 0, tried = 0;
        auto bad = [&](const k5::ConvChains &b) { ++tried; refused += k5::conv1d_silu_rows(ctx, dx.addr, dim, dw.addr, b, dyr.addr, dim, dim, table, 0) == CERE_ERR_ARGUMENT; };
        auto b1 = cc; b1.first[2] = 5; bad(b1);                 // not back to back
        auto b2 = cc; b2.saves[0] = cc.states[1]; bad(b2);      // a save aliases a state
        auto b3 = cc; b3.saves[1] = 0; bad(b3);                 // a needed save missing
        auto b4 = cc; b4.chains = 17; bad(b4);                  // too many chains
        ++tried; refused += k5::conv1d_silu(ctx, dx.addr, dim - 1, dw.addr, dst.addr, dy.addr, dim, 1, dim) == CERE_ERR_ARGUMENT;  // stride < dim
        ++tried; refused += k5::conv1d_state_at(ctx, dx.addr, dim, dst.addr, 2, dim) == CERE_ERR_ARGUMENT;  // tokens < 3
        ++tried; refused += k5::l2norm(ctx, dx.addr + 2, dy.addr, 4, 128, 1e-6f) == CERE_ERR_ARGUMENT;      // misaligned
        ++tried; refused += k5::l2norm(ctx, dx.addr, dy.addr, 4, 64, 1e-6f) == CERE_ERR_ARGUMENT;          // dim != 128
        ctx.submit_wait();
        char d[100]; std::snprintf(d, sizeof d, "(extra) %d of %d malformed calls refused with CERE_ERR_ARGUMENT", refused, tried);
        report("argument refusals", refused == tried, d);
    }
    {   // gating vs FP64
        const int64_t tokens = 9, heads = 48;
        auto a = rnd(tokens * heads, 3.f), b = rnd(tokens * heads, 3.f), dt = rnd(heads, 1.f);
        std::vector<float> A_log(heads); for (auto &v : A_log) v = std::log(0.1f + std::fabs(nd(g)));
        a[0] = f2bf(30.f);  // softplus linear branch (x > 20)
        auto da = kt::upload(ctx, a), db = kt::upload(ctx, b), ddt = kt::upload(ctx, dt), dA = kt::upload(ctx, A_log);
        auto dg = kt::make(ctx, tokens * heads * 4), dbeta = kt::make(ctx, tokens * heads * 4);
        ctx.begin(); kt::must(k5::gating(ctx, da.addr, heads, db.addr, heads, dA.addr, ddt.addr, dg.addr, dbeta.addr, tokens, heads), "gating"); ctx.barrier(); ctx.submit_wait();
        auto gg = kt::host<float>(dg, tokens * heads), bb = kt::host<float>(dbeta, tokens * heads); double worst = 0; bool beta_ok = true;
        for (int64_t i = 0; i < tokens * heads; ++i) {
            const int h = int(i % heads); const double xx = double(bf2f(a[i])) + bf2f(dt[h]);
            // the contract's softplus: log(1 + exp(x)) with 1 + exp(x) rounded to FP32 (as the HIP test's reference)
            const double sp = xx > 20 ? xx : std::log(double(1.0f + float(std::exp(xx)))), ref = -std::exp(double(A_log[h])) * sp;
            worst = std::fmax(worst, std::fabs(gg[i] - ref) / std::fmax(std::fabs(ref), 1e-30));
            const float sig = float(1 / (1 + std::exp(-double(bf2f(b[i])))));
            beta_ok = beta_ok && std::fabs(bb[i] - bf2f(f2bf(sig))) <= 1e-2f * std::fabs(sig) + 1e-6f;  // bf16(sigmoid)
        }
        char d[100]; std::snprintf(d, sizeof d, "g worst rel err %.2e (limit 1e-5), beta = bf16(sigmoid(b)) %s", worst, beta_ok ? "yes" : "NO");
        report("gating", worst <= 1e-5 && beta_ok, d);
    }
    {   // l2norm vs FP64
        const int64_t rows = 7 * 16; const int64_t d128 = k5::kHeadDim; const float eps = 1e-6f;
        auto xx = rnd(rows * d128, 2.f); auto dxx = kt::upload(ctx, xx); auto dyy = kt::make(ctx, rows * d128 * 2);
        ctx.begin(); kt::must(k5::l2norm(ctx, dxx.addr, dyy.addr, rows, d128, eps), "l2norm"); ctx.barrier(); ctx.submit_wait();
        auto yy = kt::host<uint16_t>(dyy, rows * d128);
        double num = 0, den = 0;
        for (int64_t r = 0; r < rows; ++r) { double ss = 0; for (int i = 0; i < d128; ++i) ss += double(bf2f(xx[r * d128 + i])) * bf2f(xx[r * d128 + i]);
            for (int i = 0; i < d128; ++i) { const double ref = bf2f(xx[r * d128 + i]) / std::sqrt(ss + eps), got = bf2f(yy[r * d128 + i]); num += (got - ref) * (got - ref); den += ref * ref; } }
        char d[64]; std::snprintf(d, sizeof d, "rel-L2 %.2e (limit 4e-3)", std::sqrt(num / den)); report("l2norm", std::sqrt(num / den) <= 4e-3, d);
        // in place (y aliases x, allowed by the contract) gives the same bits (extra)
        ctx.begin(); kt::must(k5::l2norm(ctx, dxx.addr, dxx.addr, rows, d128, eps), "l2norm"); ctx.barrier(); ctx.submit_wait();
        const bool same = kt::host<uint16_t>(dxx, rows * d128) == yy;
        report("l2norm in place", same, same ? "(extra) y aliasing x bit-equal to out of place" : "(extra) in place DIFFERS");
    }

    // --- timing at decode shapes (one token: conv over the 8,192 q|k|v channels, gating over 48 heads; an 8-row verify
    // chain; l2norm over 32 rows = the q and k heads of one token)
    std::printf("timing (GPU timestamps):\n");
    {
        std::memcpy(dst.map, st0.data(), SB);
        print_times("conv1d_silu (1 token, dim 8192)", time_it(ctx, [&] { kt::must(k5::conv1d_silu(ctx, dx.addr, dim, dw.addr, dst.addr, dy.addr, dim, 1, dim), "conv"); }));
        k5::ConvChains c8{}; c8.chains = 1; c8.first[0] = 0; c8.rows[0] = 8; c8.states[0] = dst.addr;
        std::vector<cere::vk::Buffer> sv(7);
        for (int r = 0; r < 7; ++r) { sv[r] = kt::make(ctx, SB); c8.saves[r] = sv[r].addr; }
        print_times("conv1d_silu_rows (1 chain x 8)", time_it(ctx, [&] { kt::must(k5::conv1d_silu_rows(ctx, dx.addr, dim, dw.addr, c8, dy.addr, dim, dim, table, 0), "rows"); }));
        const int64_t heads = 48;
        auto a = rnd(8 * heads, 3.f), b = rnd(8 * heads, 3.f), dt = rnd(heads, 1.f);
        std::vector<float> A_log(heads, -0.5f);
        auto da = kt::upload(ctx, a), db = kt::upload(ctx, b), ddt = kt::upload(ctx, dt), dA = kt::upload(ctx, A_log);
        auto dg = kt::make(ctx, 8 * heads * 4), dbeta = kt::make(ctx, 8 * heads * 4);
        print_times("gating (1 token, 48 heads)", time_it(ctx, [&] { kt::must(k5::gating(ctx, da.addr, heads, db.addr, heads, dA.addr, ddt.addr, dg.addr, dbeta.addr, 1, heads), "gating"); }));
        print_times("gating (8 tokens, 48 heads)", time_it(ctx, [&] { kt::must(k5::gating(ctx, da.addr, heads, db.addr, heads, dA.addr, ddt.addr, dg.addr, dbeta.addr, 8, heads), "gating"); }));
        print_times("l2norm (32 rows)", time_it(ctx, [&] { kt::must(k5::l2norm(ctx, dx.addr, dy.addr, 32, 128, 1e-6f), "l2norm"); }));
        print_times("l2norm (256 rows = 8 tokens)", time_it(ctx, [&] { kt::must(k5::l2norm(ctx, dx.addr, dy.addr, 256, 128, 1e-6f), "l2norm"); }));
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
