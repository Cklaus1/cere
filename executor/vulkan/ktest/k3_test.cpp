// VulkanAmd P2 K3: hyper-connections on Vulkan (k3_hc_vk.h) against the rows and bounds of executor/hip/ktest/k3_test.cpp
// (hc 4, hidden 2,560, lowrank 320, the same seed and data order), plus rows the Vulkan design adds:
//  1. hc_mix at m 7 and 300: normed within 1 BF16 ulp of a host mirror of the Gemma RMSNorm, mixed vs FP64 from the kernel's own
//     normed (t kept at its BF16 rounding point), rel-L2 <= 1e-2; repeat bit-identical;
//  2. hc_combine vs FP64: out rel-L2 <= 4e-3 (m 7; also m 300, the one-workgroup-per-row path);
//  3. hc_combine_norm == hc_combine then hc_mix's per-stream norm, bit for bit (m 7, 300; in place too);
//  4. rows plan (init_rows, 5 rows): hc_mix of 5 rows == each row at m 1 on the instantiated plan, bit for bit; plus m 300 (the K2
//     GEMM) rows 0,1,15,16,17,100,299 == that row at m 1 with the call's class (cls 300: k3_hc.h "`cls` as GemmPlan::run ...
//     both GEMMs dispatch for cls rows"; the HIP k7_test.cpp:157 / :167 class pin), and shift invariance (a 40-row call
//     starting at row 37, class >= 17 both: the shift-invariant K2 GEMM);
//  5. combine: m 7 (split rows) == the first 7 rows of m 300 (row per workgroup) == m 1, bit for bit;
//  6. canonical NaN: a NaN in block_out / hyper gives 0x7FFF where NaN is due, other rows unchanged; negative control; refusals.
// Usage: k3_test <spv dir> [time]
#include <functional>
#include <random>
#include <string>

#include "launch/vk_launch_testing.h"
#include "kernels/k2_gemm_vk.h"
#include "kernels/k3_hc_vk.h"
#include "ktest_util.h"

namespace k3 = cere::vkk3;
using cere::vk::Buffer;
using cere::vk::Context;
using kt::bf2f;
using kt::f2bf;
static int g_fail = 0;
static void check(const char *name, bool ok, const char *msg) { std::printf("%-22s %s  %s\n", name, ok ? "PASS" : "FAIL", msg); if (!ok) ++g_fail; }

constexpr uint32_t HC = 4, H = 2560, LR = 320, W = HC * H;

static void sync(Context &ctx, const std::function<void()> &rec) { ctx.begin(); rec(); ctx.barrier(); ctx.submit_wait(); }
static cere_status last;
#define RUN(call) do { sync(ctx, [&] { last = (call); }); kt::must(last, #call); } while (0)

struct Times { double med, min, max, stream; };
static Times time_it(Context &ctx, const std::function<void(int)> &rec, int warm = 20, int reps = 15, int samples = 5, int burst = 20) {
    int c = 0;
    ctx.begin();
    for (int i = 0; i < warm; ++i) { rec(c++); ctx.barrier(); }
    for (int i = 0; i < reps; ++i) { ctx.stamp(2 * i); rec(c++); ctx.stamp(2 * i + 1); ctx.barrier(); }
    ctx.submit_wait();
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
    std::sort(t.begin(), t.end());
    ctx.begin();
    for (int s = 0; s < samples; ++s) {
        ctx.stamp(2 * s);
        for (int i = 0; i < burst; ++i) { rec(c++); ctx.barrier(); }
        ctx.stamp(2 * s + 1);
    }
    ctx.submit_wait();
    std::vector<double> u;
    for (int s = 0; s < samples; ++s) u.push_back(ctx.stamp_ms(2 * s, 2 * s + 1) / burst);
    std::sort(u.begin(), u.end());
    return {t[t.size() / 2], t.front(), t.back(), u[u.size() / 2]};
}

static void timing(Context &ctx) {
    std::printf("timing (GPU timestamps; call = one call between its own stamps, median of 15 after 20 warm-up; stream = 20 back-to-back / 20, median of 5;\n"
                "         \"rot8\" = call i reads weight set i %% 8 (8 x 13.2 MB > the 32 MB MALL: DRAM-cold weights), \"hot\" = one set):\n");
    std::mt19937 g(5); std::normal_distribution<float> nd;
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    const uint32_t MAXM = 4096; const int SETS = 8;
    const k3::HcShape shape{HC, H, LR, 1e-6f};
    k3::HcPlans plans; kt::must(plans.init(shape, MAXM), "init");
    auto wd = rnd(size_t(LR) * W, 0.01f), wu = rnd(size_t(W) * LR, 0.05f), wi = rnd(size_t(HC) * W, 0.01f), wn = rnd(W, 0.1f);
    Buffer bwd = kt::make(ctx, size_t(SETS) * wd.size() * 2), bwu = kt::make(ctx, size_t(SETS) * wu.size() * 2), bwi = kt::make(ctx, size_t(SETS) * wi.size() * 2), bwn = kt::make(ctx, size_t(SETS) * wn.size() * 2);
    for (int s = 0; s < SETS; ++s) {
        std::memcpy(static_cast<uint8_t *>(bwd.map) + size_t(s) * wd.size() * 2, wd.data(), wd.size() * 2);
        std::memcpy(static_cast<uint8_t *>(bwu.map) + size_t(s) * wu.size() * 2, wu.data(), wu.size() * 2);
        std::memcpy(static_cast<uint8_t *>(bwi.map) + size_t(s) * wi.size() * 2, wi.data(), wi.size() * 2);
        std::memcpy(static_cast<uint8_t *>(bwn.map) + size_t(s) * wn.size() * 2, wn.data(), wn.size() * 2);
    }
    Buffer hyper = kt::upload(ctx, rnd(size_t(MAXM) * W, 1.f)), block = kt::upload(ctx, rnd(size_t(MAXM) * H, 1.f));
    Buffer normed = kt::make(ctx, size_t(MAXM) * W * 2), mixed = kt::make(ctx, size_t(MAXM) * H * 2), out = kt::make(ctx, size_t(MAXM) * W * 2);
    Buffer ws = kt::make(ctx, k3::hc_mix_workspace_bytes(shape, MAXM) + 256);
    sync(ctx, [&] {});
    // warm every weight set once so first touch is not timed
    auto setaddr = [&](const Buffer &b, size_t one, int i, int sets) { return b.addr + size_t(i % sets) * one * 2; };
    std::printf("  %-44s %10s %10s   (m: 1, 5, 8, 16 decode / verify (K2 GEMV); 1205, 4096 prefill (K2 GEMM))\n", "", "call us", "stream us");
    for (int sets : {1, SETS}) {
        for (uint32_t m : {1u, 5u, 8u, 16u, 1205u, 4096u}) {
            auto row = [&](const char *name, const std::function<void(int)> &f) {
                const Times t = time_it(ctx, f, 20, 15, 5, m >= 1024 ? 5 : 20);
                std::printf("  %-4s m %4u %-30s call %10.2f us (min %.2f, max %.2f)  stream %10.2f us/call\n", sets == 1 ? "hot" : "rot8", m, name, t.med * 1e3, t.min * 1e3, t.max * 1e3, t.stream * 1e3);
            };
            row("hc_mix (norm+down+act+up+gate)", [&](int i) {
                kt::must(k3::hc_mix(ctx, plans, hyper.addr, setaddr(bwn, W, i, sets), setaddr(bwd, wd.size(), i, sets), setaddr(bwu, wu.size(), i, sets), m, normed.addr, mixed.addr, ws.addr), "hc_mix"); });
            row("hc_combine", [&](int i) {
                kt::must(k3::hc_combine(ctx, shape, block.addr, hyper.addr, normed.addr, setaddr(bwi, wi.size(), i, sets), m, out.addr), "hc_combine"); });
            row("hc_combine_norm", [&](int i) {
                kt::must(k3::hc_combine_norm(ctx, shape, block.addr, hyper.addr, normed.addr, setaddr(bwi, wi.size(), i, sets), setaddr(bwn, W, i, sets), m, out.addr), "hc_combine_norm"); });
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { std::printf("usage: k3_test <spv dir> [time]\n"); return 2; }
    auto ctx_owner = cere::vk::testing::create_or_die(argv[1]);
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s, subgroup %u..%u\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str(), ctx.info().subgroup_min, ctx.info().subgroup_max);
    std::printf("hc_mix mode: %s\n", k3::hc_mix_mode());
    check("K2 GEMM probe", cere::vkk2::gemm_probe(ctx) == CERE_OK, "k2_gemm_probe: the K2 GEMM's accumulator map (hc_mix row classes >= 17)");
    std::mt19937 g(41); std::normal_distribution<float> nd;
    auto rnd = [&](size_t n, float s, float mean = 0.f) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(mean + nd(g) * s); return v; };
    const k3::HcShape shape{HC, H, LR, 1e-6f};
    const uint32_t MAXM = 300;
    k3::HcPlans plans; kt::must(plans.init(shape, MAXM), "plans.init");
    auto wnorm = rnd(W, 0.1f), wdown = rnd(size_t(LR) * W, 1.f / std::sqrt(float(W))), wup = rnd(size_t(W) * LR, 1.f / std::sqrt(float(LR))), winj = rnd(size_t(HC) * W, 1.f / std::sqrt(float(W)));
    Buffer dwn = kt::upload(ctx, wnorm), dwd = kt::upload(ctx, wdown), dwu = kt::upload(ctx, wup), dwi = kt::upload(ctx, winj);
    Buffer dws = kt::make(ctx, k3::hc_mix_workspace_bytes(shape, MAXM));
    auto hyper = rnd(size_t(MAXM) * W, 1.f), block = rnd(size_t(MAXM) * H, 1.f);
    Buffer dhy = kt::upload(ctx, hyper), dbo = kt::upload(ctx, block);
    Buffer dnormed = kt::make(ctx, size_t(MAXM) * W * 2), dmixed = kt::make(ctx, size_t(MAXM) * H * 2), dout = kt::make(ctx, size_t(MAXM) * W * 2);
    auto mix = [&](const k3::HcPlans &p, uint64_t hy, uint32_t m, uint64_t nm, uint64_t mx, uint32_t cls = 0) {
        RUN(k3::hc_mix(ctx, p, hy, dwn.addr, dwd.addr, dwu.addr, m, nm, mx, dws.addr, k3::HcMixPath::kAuto, cls));
    };

    for (uint32_t m : {7u, 300u}) {   // 1. hc_mix
        mix(plans, dhy.addr, m, dnormed.addr, dmixed.addr);
        auto normed = kt::host<uint16_t>(dnormed, size_t(m) * W), mixed = kt::host<uint16_t>(dmixed, size_t(m) * H);
        int64_t nbad = 0;
        for (uint32_t r = 0; r < m; ++r) for (uint32_t s = 0; s < HC; ++s) {
            double ss = 0; for (uint32_t j = 0; j < H; ++j) { const double x = bf2f(hyper[size_t(r) * W + s * H + j]); ss += x * x; }
            const double rs = 1 / std::sqrt(ss / H + 1e-6);
            for (uint32_t j = 0; j < H; ++j) { const size_t i = size_t(r) * W + s * H + j;
                const uint16_t want = f2bf(float(bf2f(hyper[i]) * rs * (1 + double(bf2f(wnorm[s * H + j])))));
                nbad += std::abs(int(normed[i]) - int(want)) > 1; }
        }
        double num = 0, den = 0;
        for (uint32_t r = 0; r < m; r += (m > 16 ? 13 : 1)) {
            std::vector<double> t(LR);
            for (uint32_t o = 0; o < LR; ++o) { double a = 0; for (uint32_t i = 0; i < W; ++i) a += double(bf2f(normed[size_t(r) * W + i])) * bf2f(wdown[size_t(o) * W + i]);
                a /= HC; t[o] = bf2f(f2bf(float(a / (1 + std::exp(-a))))); }
            for (uint32_t j = 0; j < H; ++j) {
                double acc = 0;
                for (uint32_t s = 0; s < HC; ++s) { double u = 0; for (uint32_t o = 0; o < LR; ++o) u += t[o] * bf2f(wup[size_t(s * H + j) * LR + o]);
                    acc += 1 / (1 + std::exp(-u)) * bf2f(normed[size_t(r) * W + s * H + j]); }
                const double ref = acc / HC, e = bf2f(mixed[size_t(r) * H + j]) - ref; num += e * e; den += ref * ref;
            }
        }
        const double err = std::sqrt(num / den);
        mix(plans, dhy.addr, m, dnormed.addr, dmixed.addr);
        const bool rep = kt::host<uint16_t>(dmixed, size_t(m) * H) == mixed && kt::host<uint16_t>(dnormed, size_t(m) * W) == normed;
        char msg[200]; std::snprintf(msg, sizeof msg, "m %3u: normed %ld outputs > 1 ulp; mixed rel-L2 %.2e vs FP64 (1e-2); repeat %s", m, (long)nbad, err, rep ? "bit-equal" : "DIFFERS");
        check("hc_mix", nbad == 0 && err <= 1e-2 && rep, msg);
    }
    std::vector<uint16_t> mixed300 = kt::host<uint16_t>(dmixed, size_t(300) * H), normed300 = kt::host<uint16_t>(dnormed, size_t(300) * W);
    {   // 2. hc_combine, 3. hc_combine_norm
        for (uint32_t m : {7u, 300u}) {
            mix(plans, dhy.addr, m, dnormed.addr, dmixed.addr);
            auto normed = kt::host<uint16_t>(dnormed, size_t(m) * W);
            RUN(k3::hc_combine(ctx, shape, dbo.addr, dhy.addr, dnormed.addr, dwi.addr, m, dout.addr));
            auto out = kt::host<uint16_t>(dout, size_t(m) * W);
            double num = 0, den = 0;
            for (uint32_t r = 0; r < m; ++r) for (uint32_t s = 0; s < HC; ++s) {
                double d = 0; for (uint32_t i = 0; i < W; ++i) d += double(bf2f(normed[size_t(r) * W + i])) * bf2f(winj[size_t(s) * W + i]);
                const double a = 2 / (1 + std::exp(-d / HC));
                for (uint32_t j = 0; j < H; ++j) { const size_t i = size_t(r) * W + s * H + j; const double ref = bf2f(hyper[i]) + a * bf2f(block[size_t(r) * H + j]);
                    const double e = bf2f(out[i]) - ref; num += e * e; den += ref * ref; }
            }
            const double err = std::sqrt(num / den);
            char msg[160]; std::snprintf(msg, sizeof msg, "m %3u: out rel-L2 %.2e vs FP64 (4e-3)", m, err);
            check("hc_combine", err <= 4e-3, msg);
            // combine_norm vs combine, then hc_mix's norm of `out` (next norm weight)
            auto wnext = rnd(W, 0.1f); Buffer dwx = kt::upload(ctx, wnext);
            Buffer dn2 = kt::make(ctx, size_t(m) * W * 2), do2 = kt::make(ctx, size_t(m) * W * 2), dn3 = kt::make(ctx, size_t(m) * W * 2), dn4 = kt::make(ctx, size_t(m) * W * 2), dhc = kt::make(ctx, size_t(m) * W * 2);
            std::memcpy(dn2.map, dnormed.map, size_t(m) * W * 2);
            RUN(k3::hc_combine_norm(ctx, shape, dbo.addr, dhy.addr, dn2.addr, dwi.addr, dwx.addr, m, do2.addr));
            RUN(k3::hc_mix(ctx, plans, dout.addr, dwx.addr, dwd.addr, dwu.addr, m, dn3.addr, dmixed.addr, dws.addr));
            const bool eq = kt::host<uint16_t>(do2, size_t(m) * W) == out && kt::host<uint16_t>(dn2, size_t(m) * W) == kt::host<uint16_t>(dn3, size_t(m) * W);
            std::snprintf(msg, sizeof msg, "m %3u: == hc_combine, then hc_mix's per-stream norm of its output: out and normed bit-equal", m);
            check("hc_combine_norm", eq, msg);
            // in place: out aliases hyper
            std::memcpy(dhc.map, dhy.map, size_t(m) * W * 2);
            std::memcpy(dn4.map, dnormed.map, size_t(m) * W * 2);
            RUN(k3::hc_combine_norm(ctx, shape, dbo.addr, dhc.addr, dn4.addr, dwi.addr, dwx.addr, m, dhc.addr));
            const bool ip = kt::host<uint16_t>(dhc, size_t(m) * W) == out && kt::host<uint16_t>(dn4, size_t(m) * W) == kt::host<uint16_t>(dn3, size_t(m) * W);
            std::snprintf(msg, sizeof msg, "m %3u: out aliasing hyper: out and normed bit-equal to the out-of-place call", m);
            check("combine in place", ip, msg);
        }
    }
    {   // 4. rows plans (HIP row), then m 300 vs m 1 and shift invariance
        k3::HcPlans rows; kt::must(rows.init_rows(plans, 5), "init_rows");
        mix(rows, dhy.addr, 5, dnormed.addr, dmixed.addr);
        auto mr = kt::host<uint16_t>(dmixed, 5 * H);
        bool eq = true;
        for (uint32_t r = 0; r < 5; ++r) {
            mix(plans, dhy.addr + size_t(r) * W * 2, 1, dnormed.addr, dmixed.addr);
            auto one = kt::host<uint16_t>(dmixed, H); eq &= std::equal(one.begin(), one.end(), mr.begin() + size_t(r) * H);
        }
        check("rows plans", eq, "hc_mix of 5 rows on init_rows plans == each row at m 1 on the instantiated plans, bit for bit");
        bool e2 = true; std::string which;
        for (uint32_t r : {0u, 1u, 15u, 16u, 17u, 100u, 299u}) {
            mix(plans, dhy.addr + size_t(r) * W * 2, 1, dnormed.addr, dmixed.addr, 300);  // the m 300 call's class
            auto one = kt::host<uint16_t>(dmixed, H), on = kt::host<uint16_t>(dnormed, W);
            const bool ok = std::equal(one.begin(), one.end(), mixed300.begin() + size_t(r) * H) && std::equal(on.begin(), on.end(), normed300.begin() + size_t(r) * W);
            if (!ok) which += " " + std::to_string(r);
            e2 &= ok;
        }
        check("rows m300 vs m1", e2, e2 ? "rows 0,1,15,16,17,100,299 of the m 300 call (mixed and normed) == that row at m 1 (cls 300), bit for bit" : ("rows differ:" + which).c_str());
        mix(plans, dhy.addr + size_t(37) * W * 2, 40, dnormed.addr, dmixed.addr);
        auto sh = kt::host<uint16_t>(dmixed, size_t(40) * H);
        const bool e3 = std::equal(sh.begin(), sh.end(), mixed300.begin() + size_t(37) * H);
        check("shift invariance", e3, "a 40-row call starting at row 37 == rows 37..76 of the m 300 call (K2 GEMM row tiles move), bit for bit");
    }
    {   // 5. combine: m 7 (split rows) == first rows of m 300 (row per workgroup) == m 1
        mix(plans, dhy.addr, 300, dnormed.addr, dmixed.addr);
        RUN(k3::hc_combine(ctx, shape, dbo.addr, dhy.addr, dnormed.addr, dwi.addr, 300, dout.addr));
        auto o300 = kt::host<uint16_t>(dout, size_t(300) * W);
        RUN(k3::hc_combine(ctx, shape, dbo.addr, dhy.addr, dnormed.addr, dwi.addr, 7, dout.addr));
        auto o7 = kt::host<uint16_t>(dout, size_t(7) * W);
        bool eq = std::equal(o7.begin(), o7.end(), o300.begin());
        for (uint32_t r : {0u, 16u, 17u, 299u}) {
            RUN(k3::hc_combine(ctx, shape, dbo.addr + size_t(r) * H * 2, dhy.addr + size_t(r) * W * 2, dnormed.addr + size_t(r) * W * 2, dwi.addr, 1, dout.addr));
            auto one = kt::host<uint16_t>(dout, W); eq &= std::equal(one.begin(), one.end(), o300.begin() + size_t(r) * W);
        }
        check("combine rows", eq, "m 7 (rows split over workgroups) == rows of m 300 (workgroup per row) == rows 0,16,17,299 at m 1, bit for bit");
    }
    {   // 6. NaN, negative control, refusals
        mix(plans, dhy.addr, 6, dnormed.addr, dmixed.addr);
        auto clean_mixed = kt::host<uint16_t>(dmixed, size_t(6) * H);
        RUN(k3::hc_combine(ctx, shape, dbo.addr, dhy.addr, dnormed.addr, dwi.addr, 6, dout.addr));
        auto clean_out = kt::host<uint16_t>(dout, size_t(6) * W);
        auto hnan = hyper; hnan.resize(size_t(6) * W); hnan[size_t(2) * W + 3 * H + 17] = 0x7FC1;   // row 2, stream 3, j 17: a NaN with a payload
        auto bnan = block; bnan.resize(size_t(6) * H); bnan[size_t(4) * H + 100] = 0xFFC2;           // row 4, j 100
        Buffer hn = kt::upload(ctx, hnan), bn = kt::upload(ctx, bnan);
        mix(plans, hn.addr, 6, dnormed.addr, dmixed.addr);
        auto nm = kt::host<uint16_t>(dmixed, size_t(6) * H);
        bool rows_ok = true, nan_ok = true;
        for (uint32_t r = 0; r < 6; ++r) for (uint32_t j = 0; j < H; ++j) {
            const uint16_t v = nm[size_t(r) * H + j];
            if (r == 2) nan_ok &= (v & 0x7FFF) == 0x7FFF && (v == 0x7FFF || v == 0xFFFF);     // NaN: encoded 0x7FFF (sign never set by the RNE encode)
            else rows_ok &= v == clean_mixed[size_t(r) * H + j];
        }
        for (uint32_t j = 0; j < H; ++j) nan_ok &= nm[size_t(2) * H + j] == 0x7FFF;
        RUN(k3::hc_combine(ctx, shape, bn.addr, dhy.addr, dnormed.addr, dwi.addr, 6, dout.addr));  // dnormed: from the NaN mix, so row 2's gate is NaN
        auto no = kt::host<uint16_t>(dout, size_t(6) * W);
        // row 4: block_out NaN at j 100 -> out[4, g, 100] = 0x7FFF for every g (whatever the gate; a gate is finite for row 4); other elements of row 4 unchanged
        bool nan2 = true;
        for (uint32_t g2 = 0; g2 < HC; ++g2) nan2 &= no[size_t(4) * W + g2 * H + 100] == 0x7FFF;
        for (uint32_t i = 0; i < W; ++i) if (i % H != 100) nan2 &= no[size_t(4) * W + i] == clean_out[size_t(4) * W + i];
        check("canonical NaN", nan_ok && rows_ok && nan2, "NaN in hyper: that row's mixed is all 0x7FFF, other rows bit-equal to clean; NaN in block_out: that element 0x7FFF in every stream, row otherwise unchanged");
        // negative control: the bit comparison sees a one-bit change of w_down
        uint16_t save = wdown[12345]; wdown[12345] ^= 0x0040;
        Buffer dwd2 = kt::upload(ctx, wdown);
        mix(plans, dhy.addr, 6, dnormed.addr, dmixed.addr);
        auto a = kt::host<uint16_t>(dmixed, size_t(6) * H);
        RUN(k3::hc_mix(ctx, plans, dhy.addr, dwn.addr, dwd2.addr, dwu.addr, 6, dnormed.addr, dmixed.addr, dws.addr));
        auto b = kt::host<uint16_t>(dmixed, size_t(6) * H);
        wdown[12345] = save;
        check("negative control", a != b && a == clean_mixed, "a one-bit change of one w_down element changes mixed (the bit comparison sees it)");
        // refusals
        bool ref = true;
        k3::HcPlans bad; 
        ref &= bad.init({3, H, LR, 1e-6f}, 4) == CERE_ERR_UNSUPPORTED;
        ref &= bad.init({4, 2564, LR, 1e-6f}, 4) == CERE_ERR_UNSUPPORTED;
        ref &= bad.init({4, 8200, LR, 1e-6f}, 4) == CERE_ERR_UNSUPPORTED;
        ref &= bad.init({4, H, 328, 1e-6f}, 4) == CERE_ERR_UNSUPPORTED;
        ref &= bad.init(shape, 0) == CERE_ERR_ARGUMENT;
        ctx.begin();
        ref &= k3::hc_mix(ctx, plans, dhy.addr, dwn.addr, dwd.addr, dwu.addr, 1, dnormed.addr, dmixed.addr, dws.addr, k3::HcMixPath::kEpilogue) == CERE_ERR_UNSUPPORTED;
        ref &= k3::hc_mix(ctx, plans, dhy.addr, dwn.addr, dwd.addr, dwu.addr, 1, dnormed.addr, dmixed.addr, dws.addr, k3::HcMixPath::kEarly) == CERE_ERR_UNSUPPORTED;
        ref &= k3::hc_mix(ctx, plans, dhy.addr, dwn.addr, dwd.addr, dwu.addr, 0, dnormed.addr, dmixed.addr, dws.addr) == CERE_ERR_ARGUMENT;
        ref &= k3::hc_mix(ctx, plans, dhy.addr, dwn.addr, dwd.addr, dwu.addr, MAXM + 1, dnormed.addr, dmixed.addr, dws.addr) == CERE_ERR_ARGUMENT;
        ref &= k3::hc_mix(ctx, plans, 0, dwn.addr, dwd.addr, dwu.addr, 1, dnormed.addr, dmixed.addr, dws.addr) == CERE_ERR_ARGUMENT;
        ref &= k3::hc_mix(ctx, plans, dhy.addr + 2, dwn.addr, dwd.addr, dwu.addr, 1, dnormed.addr, dmixed.addr, dws.addr) == CERE_ERR_ARGUMENT;
        ref &= k3::hc_mix(ctx, plans, dhy.addr, dwn.addr, dwd.addr, dwu.addr, 1, dnormed.addr, dmixed.addr, dws.addr + 16) == CERE_ERR_ARGUMENT;
        ref &= k3::hc_mix(ctx, plans, dhy.addr, dwn.addr, dwd.addr, dwu.addr, 7, dnormed.addr, dmixed.addr, dws.addr, k3::HcMixPath::kAuto, 5) == CERE_ERR_ARGUMENT;
        ref &= k3::hc_mix_normed(ctx, plans, dnormed.addr, dwd.addr, dwu.addr, 7, dmixed.addr, dws.addr, k3::HcMixPath::kAuto, 5) == CERE_ERR_ARGUMENT;
        ref &= k3::hc_combine(ctx, {3, H, LR, 1e-6f}, dbo.addr, dhy.addr, dnormed.addr, dwi.addr, 1, dout.addr) == CERE_ERR_UNSUPPORTED;
        ref &= k3::hc_combine(ctx, shape, dbo.addr, dhy.addr, dnormed.addr, dwi.addr, 0, dout.addr) == CERE_ERR_ARGUMENT;
        ref &= k3::hc_combine_norm(ctx, shape, dbo.addr, dhy.addr, dnormed.addr, dwi.addr, 0, 1, dout.addr) == CERE_ERR_ARGUMENT;
        ref &= !k3::hc_mix_fuses(plans, 1);
        ctx.barrier(); ctx.submit_wait();
        check("refusals", ref, "hc 3, hidden 2564 / 8200, lowrank 328, kEpilogue / kEarly: UNSUPPORTED; m 0, m > max, null, misaligned, null norm weight, cls < m: ARGUMENT");
    }
    if (argc > 2 && std::string(argv[2]) == "time") timing(ctx);
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
