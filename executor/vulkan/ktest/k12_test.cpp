// VulkanAmd P2 K12: executor/hip/ktest/k12_test.cpp's contract rows on the Vulkan device sampler (shaders/k12_*.comp), the same row
// shapes, seeds, data, references (host/sampler_ref) and bounds, on a 248,320-id row: process within the D2 bound 2^-22 max(1,|x|) +
// count|f| 2^-23 with the mask's -inf ids exact; top2_f32 exact (ties, NaN); probabilities (fp32 probs within 1e-6 relative, lse within
// 1e-12; the non-finite hardening rows); keep_mask exact outside sampler_ref's 1e-6 boundary band; draw_u and draw_seeded exact;
// top_logprobs ids exact, logprobs within 1e-6; count_commit; every kernel bit-identical on repeat. Rows added over the HIP test (evidence for
// the exactness argument, never a loosened bound): process and the fp32 division bit-exact against the host, the arithmetic helpers
// (fdiv fp32 / fp64, exp, log, uniform) against the host's libm over millions of operands, finalize, top_logprobs at T = 0 and with the
// cere_logprob_entry stride, more keep_mask shapes, and the refusals. Then a decode-shape timing table (GPU timestamps, median of 15).
// Built with -DK12_HIP_TIMING under hipcc it is instead the same-method HIP timing of the same stages (reference). Exit 0 when every row passes.
#ifdef K12_HIP_TIMING
#include <hip/hip_runtime.h>
#include "kernels/k12_sampler.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#ifndef K12_HIP_TIMING
#include "host/sampler_ref.h"
#include "kernels/k12_sampler_vk.h"
#include "launch/vk_launch.h"
#include "launch/vk_launch_testing.h"
#endif

static int g_fail = 0;
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static double median(std::vector<double> t) { std::sort(t.begin(), t.end()); return t[t.size() / 2]; }
static const int kReps = 15;
static const int kBatch = 10;  // calls per timed sample (the sample is divided by kBatch): amortizes the per-launch / per-event overhead of microsecond kernels on both sides

// The decode-shape data both sides use (the HIP test's: seed 17, N(0,2) BF16 logits, 300 counted ids, 2000 masked ids, 4 biases).
struct Data {
    int64_t V = 248320; size_t W = (V + 31) / 32;
    std::vector<uint16_t> logits; std::vector<uint32_t> counts, mask, bias_ids = {5, 777, 123456, 248319};
    std::vector<float> bias_v = {3.5f, -100.f, 7.25f, 1.f};
    std::mt19937 g{17};
    Data() {
        std::normal_distribution<float> nd(0.f, 2.f);
        logits.resize(V); for (auto &x : logits) x = f2bf(nd(g));
        counts.assign(V, 0); for (int i = 0; i < 300; ++i) counts[g() % V] += 1 + g() % 3;
        mask.assign(W, ~0u); for (int i = 0; i < 2000; ++i) { const uint32_t id = g() % V; mask[id / 32] &= ~(1u << (id % 32)); }
    }
};

#ifndef K12_HIP_TIMING
using namespace cere;
namespace k12 = cere::vkk12;
namespace ref = cere::sampler_ref;
static vk::Context *g_ctx;
static void report(const char *name, bool ok, const char *detail) {
    std::printf("%-34s %s  %s\n", name, ok ? "PASS" : "FAIL", detail);
    if (!ok) ++g_fail;
}
static vk::Buffer make(uint64_t bytes) {
    vk::Buffer b;
    if (!g_ctx->alloc(bytes + 256, b)) std::exit(2);
    std::memset(b.map, 0, bytes + 256);
    return b;
}
template <class T> static vk::Buffer up(const std::vector<T> &h) { vk::Buffer b = make(h.size() * sizeof(T)); std::memcpy(b.map, h.data(), h.size() * sizeof(T)); return b; }
template <class T> static std::vector<T> host(const vk::Buffer &b, size_t n) { std::vector<T> h(n); std::memcpy(h.data(), b.map, n * sizeof(T)); return h; }
template <class T> static void put(const vk::Buffer &b, const std::vector<T> &h) { std::memcpy(b.map, h.data(), h.size() * sizeof(T)); }
static void must(cere_status s, const char *what) { if (s != CERE_OK) { std::printf("FATAL %s returned %d\n", what, int(s)); std::exit(2); } }
static void sync(const std::function<void()> &f) { g_ctx->begin(); f(); g_ctx->barrier(); g_ctx->submit_wait(); }

// ---- arithmetic helpers against the host (k12_selftest.comp) ----
struct Self { vk::Buffer a, b, o; };
static void selftest_run(Self &s, uint32_t n, uint32_t op) {
    struct { uint64_t a, b, o; uint32_t n, op; } pc{s.a.addr, s.b.addr, s.o.addr, n, op};
    sync([&] { g_ctx->dispatch(g_ctx->pipeline("k12_selftest.spv", sizeof pc, 32), &pc, (n + 63) / 64); });
}
static float f_from_bits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static uint32_t bits_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static uint64_t bits_of(double f) { uint64_t u; std::memcpy(&u, &f, 8); return u; }

static void arithmetic_rows() {
    const uint32_t N = 1u << 23;  // 8.4M operands per row
    Self s{make(N * 8), make(N * 8), make(N * 8)};
    std::mt19937_64 g(99);
    // fp32 division: full 24-bit mantissas, exponents spread over the normal range where the quotient stays normal, plus the fast-path band.
    for (int variant = 0; variant < 3; ++variant) {
        std::vector<float> a(N), b(N);
        for (uint32_t i = 0; i < N; ++i) {
            const uint32_t ma = uint32_t(g()) & 0x7fffffu, mb = uint32_t(g()) & 0x7fffffu;
            uint32_t ea, eb;
            if (variant == 0) { ea = 100 + g() % 55; eb = 100 + g() % 55; }          // the fast-path band
            else if (variant == 1) { ea = 30 + g() % 190; eb = 30 + g() % 190; }    // wide normal range
            else { ea = 127 + g() % 8; eb = 127 + g() % 3; }                          // logits / temperature-like magnitudes
            a[i] = f_from_bits(((uint32_t(g()) & 1u) << 31) | (ea << 23) | ma);
            b[i] = f_from_bits(((uint32_t(g()) & 1u) << 31) | (eb << 23) | mb);
        }
        put(s.a, a); put(s.b, b);
        selftest_run(s, N, 0);
        const auto got = host<float>(s.o, N);
        selftest_run(s, N, 4);
        const auto hw = host<float>(s.o, N);
        uint64_t bad = 0, hw_bad = 0, shown = 0;
        for (uint32_t i = 0; i < N; ++i) {
            const float want = a[i] / b[i];
            bad += bits_of(got[i]) != bits_of(want);
            hw_bad += bits_of(hw[i]) != bits_of(want);
            if (bits_of(got[i]) != bits_of(want) && shown++ < 3) std::printf("    fdiv_rn32 %a / %a: got %a want %a\n", a[i], b[i], got[i], want);
        }
        char d[160]; std::snprintf(d, sizeof d, "%u operand pairs (exponent variant %d): %llu differ from the host; plain GLSL a / b differs in %llu",
                                   N, variant, (unsigned long long)bad, (unsigned long long)hw_bad);
        report("fdiv_rn32 == host fp32 division", bad == 0, d);
    }
    {   // fp64 division
        std::vector<double> a(N), b(N);
        for (uint32_t i = 0; i < N; ++i) {
            const uint64_t ma = g() & 0xfffffffffffffull, mb = g() & 0xfffffffffffffull;
            a[i] = [&] { const uint64_t u = (uint64_t(900 + g() % 250) << 52) | ma; double d; std::memcpy(&d, &u, 8); return d; }();
            b[i] = [&] { const uint64_t u = (uint64_t(900 + g() % 250) << 52) | mb; double d; std::memcpy(&d, &u, 8); return d; }();
        }
        put(s.a, a); put(s.b, b);
        selftest_run(s, N, 1);
        const auto got = host<double>(s.o, N);
        selftest_run(s, N, 5);
        const auto hw = host<double>(s.o, N);
        uint64_t bad = 0, hw_bad = 0;
        for (uint32_t i = 0; i < N; ++i) { const double want = a[i] / b[i]; bad += bits_of(got[i]) != bits_of(want); hw_bad += bits_of(hw[i]) != bits_of(want); }
        char d[160]; std::snprintf(d, sizeof d, "%u operand pairs: %llu differ from the host; plain GLSL a / b differs in %llu", N, (unsigned long long)bad, (unsigned long long)hw_bad);
        report("fdiv_rn64 == host fp64 division", bad == 0, d);
    }
    {   // exp / log: accuracy against the host's libm (the contract rows need lse within 1e-12 and probs within 1e-6 relative)
        std::vector<double> x(N);
        for (uint32_t i = 0; i < N; ++i) x[i] = i % 4 == 0 ? -708.0 * (double(g() >> 11) / 9007199254740992.0) : -60.0 * (double(g() >> 11) / 9007199254740992.0);
        put(s.a, x);
        selftest_run(s, N, 2);
        const auto got = host<double>(s.o, N);
        double worst = 0; uint64_t exact = 0;
        for (uint32_t i = 0; i < N; ++i) { const double want = std::exp(x[i]); exact += got[i] == want; if (want > 0) worst = std::fmax(worst, std::fabs(got[i] - want) / want); }
        char d[160]; std::snprintf(d, sizeof d, "exp64 on [-708, 0]: worst relative error vs libm %.2e (limit 4e-16), %.2f %% bit-equal", worst, 100.0 * double(exact) / N);
        report("exp64 vs libm", worst <= 4e-16, d);
        for (uint32_t i = 0; i < N; ++i) x[i] = -708.0 - 37.5 * (double(g() >> 11) / 9007199254740992.0);  // subnormal results
        put(s.a, x); selftest_run(s, N, 2);
        const auto sub = host<double>(s.o, N); double wabs = 0;
        for (uint32_t i = 0; i < N; ++i) wabs = std::fmax(wabs, std::fabs(sub[i] - std::exp(x[i])));
        std::snprintf(d, sizeof d, "exp64 on [-745.5, -708] (subnormal results): worst absolute error vs libm %.2e (limit 2 subnormal ulps = 9.9e-324)", wabs);
        report("exp64 subnormal tail", wabs <= 2 * 4.9406564584124654e-324, d);
        std::vector<double> y(N);
        for (uint32_t i = 0; i < N; ++i) y[i] = i % 2 == 0 ? 1.0 + 3e5 * (double(g() >> 11) / 9007199254740992.0) : std::ldexp(1.0 + double(g() >> 11) / 9007199254740992.0, int(g() % 100) - 60);
        y[0] = 1.0; y[1] = 0.5; y[2] = 248320.0;
        put(s.a, y);
        selftest_run(s, N, 3);
        const auto lg = host<double>(s.o, N);
        double lw = 0; uint64_t lexact = 0;
        for (uint32_t i = 0; i < N; ++i) { const double want = std::log(y[i]); lexact += lg[i] == want; lw = std::fmax(lw, std::fabs(lg[i] - want) / std::fmax(std::fabs(want), 1e-300)); }
        std::snprintf(d, sizeof d, "log64: worst relative error vs libm %.2e (limit 1e-15 relative, log(1) == 0 %s), %.2f %% bit-equal", lw, lg[0] == 0.0 ? "yes" : "NO", 100.0 * double(lexact) / N);
        report("log64 vs libm", lw <= 1e-15 && lg[0] == 0.0, d);
    }
    {   // the murmur counter uniform: bit-exact
        const uint32_t n = 1u << 20; std::vector<uint64_t> seeds(n); std::vector<uint32_t> keys(n);
        for (uint32_t i = 0; i < n; ++i) { seeds[i] = g(); keys[i] = uint32_t(g()); }
        put(s.a, seeds); put(s.b, keys);
        selftest_run(s, n, 6);
        const auto got = host<double>(s.o, n);
        uint64_t bad = 0;
        for (uint32_t i = 0; i < n; ++i) bad += bits_of(got[i]) != bits_of(ref::uniform(seeds[i], keys[i], 0, 1));
        char d[100]; std::snprintf(d, sizeof d, "%u (seed, key) pairs: %llu differ from sampler_ref::uniform", n, (unsigned long long)bad);
        report("uniform (murmur counter) exact", bad == 0, d);
    }
}

int main(int argc, char **argv) {
    const std::string spv = argc > 1 ? argv[1] : "build/vulkan-k12/spv";
    auto ctx_owner = cere::vk::testing::create_or_die(spv); cere::vk::Context &ctx = *ctx_owner; g_ctx = &ctx;
    std::printf("device %s (%s %s), Vulkan %u.%u.%u, subgroup %u..%u\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str(),
                (ctx.info().api_version >> 22), (ctx.info().api_version >> 12) & 0x3ff, ctx.info().api_version & 0xfff, ctx.info().subgroup_min, ctx.info().subgroup_max);
    Data D; const int64_t V = D.V; const size_t W = D.W; std::mt19937 &g = D.g;
    auto bl = up(D.logits), bc = up(D.counts), bm = up(D.mask), bbi = up(D.bias_ids), bbv = up(D.bias_v);
    auto bproc = make(V * 4);
    const float fp = 0.3f, pp = 0.2f, rp = 1.15f;
    auto run_process = [&] { sync([&] { must(k12::process(ctx, bl.addr, V, bc.addr, fp, pp, rp, bm.addr, bbi.addr, bbv.addr, 1, 4, bproc.addr), "process"); }); return host<float>(bproc, V); };
    const auto proc = run_process(), proc2 = run_process();
    std::vector<ref::Bias> rb; for (size_t j = 0; j < D.bias_ids.size(); ++j) rb.push_back({D.bias_ids[j], D.bias_v[j]});
    std::vector<float> rproc(V); ref::process(D.logits.data(), V, D.counts.data(), fp, pp, rp, D.mask.data(), rb.data(), rb.size(), rproc.data());
    {   int64_t bad = 0, inf_bad = 0, exact_bad = 0;
        for (int64_t i = 0; i < V; ++i) {
            if (bits_of(proc[i]) != bits_of(rproc[i])) ++exact_bad;
            if (std::isinf(rproc[i]) || std::isinf(proc[i])) { inf_bad += !(std::isinf(rproc[i]) && std::isinf(proc[i]) && std::signbit(rproc[i]) == std::signbit(proc[i])); continue; }
            const double bound = std::ldexp(1.0, -22) * std::fmax(1.0, std::fabs(rproc[i])) + D.counts[i] * std::fabs(fp) * std::ldexp(1.0, -23);
            bad += std::fabs(double(proc[i]) - rproc[i]) > bound;
        }
        char d[130]; std::snprintf(d, sizeof d, "%ld outside the D2 bound, %ld -inf mismatches; repeat %s", (long)bad, (long)inf_bad, proc == proc2 ? "bit-identical" : "DIFFERS");
        report("process", bad == 0 && inf_bad == 0 && proc == proc2, d);
        std::snprintf(d, sizeof d, "%ld of %ld outputs differ in any bit from sampler_ref::process", (long)exact_bad, (long)V);
        report("process bit-exact (added)", exact_bad == 0, d);
    }
    {   // the penalties' division on every id: counts all positive, f = p = 0, r in {1.15, 0.7, 3.3333}: x < 0 ? x * r : x / r, bit-exact
        std::vector<uint32_t> ones(V, 1); auto bo = up(ones); int64_t bad = 0;
        for (float r : {1.15f, 0.7f, 3.3333f}) {
            sync([&] { must(k12::process(ctx, bl.addr, V, bo.addr, 0.f, 0.f, r, 0, 0, 0, 1, 0, bproc.addr), "process"); });
            const auto got = host<float>(bproc, V); std::vector<float> want(V);
            ref::process(D.logits.data(), V, ones.data(), 0.f, 0.f, r, nullptr, nullptr, 0, want.data());
            for (int64_t i = 0; i < V; ++i) bad += bits_of(got[i]) != bits_of(want[i]);
        }
        char d[100]; std::snprintf(d, sizeof d, "3 repetition penalties x %ld ids: %ld differ from the host", (long)V, (long)bad);
        report("process division (added)", bad == 0, d);
    }
    // the reference's processed row drives every later stage, so each stage is measured on its own
    auto bp = up(rproc);
    auto btop = make(8), bsc = make(8);
    {   std::vector<float> t2 = rproc; t2[1000] = 99.f; t2[2000] = 99.f; t2[3000] = NAN;  // ties and a NaN
        auto bt2 = up(t2);
        sync([&] { must(k12::top2_f32(ctx, bt2.addr, V, btop.addr, bsc.addr), "top2"); });
        const auto hi = host<uint32_t>(btop, 2);
        const ref::Top2 want{3000, 1000};
        sync([&] { must(k12::top2_f32(ctx, bp.addr, V, btop.addr, bsc.addr), "top2"); });
        const auto hi2 = host<uint32_t>(btop, 2), sc2 = host<uint32_t>(bsc, 2); const auto want2 = ref::top2(rproc.data(), V);
        const bool scores_ok = sc2[0] == bits_of(rproc[want2.id]) && sc2[1] == bits_of(rproc[want2.runner_up]);
        char d[150]; std::snprintf(d, sizeof d, "NaN+ties row (%u,%u) want (%u,%u); plain row (%u,%u) want (%u,%u); scores %s", hi[0], hi[1], want.id, want.runner_up, hi2[0], hi2[1], want2.id, want2.runner_up, scores_ok ? "exact" : "WRONG");
        report("top2_f32", hi[0] == want.id && hi[1] == want.runner_up && hi2[0] == want2.id && hi2[1] == want2.runner_up && scores_ok, d);
        // +-0 ties, all-NaN, -inf row
        std::vector<float> z(V, 0.f); z[7] = -0.f; z[9] = 0.f; auto bz = up(z);
        sync([&] { must(k12::top2_f32(ctx, bz.addr, V, btop.addr, bsc.addr), "top2"); }); const auto hz = host<uint32_t>(btop, 2);
        std::vector<float> nn(V, NAN); auto bn = up(nn);
        sync([&] { must(k12::top2_f32(ctx, bn.addr, V, btop.addr, bsc.addr), "top2"); }); const auto hn = host<uint32_t>(btop, 2);
        std::vector<float> ni(V, -INFINITY); auto bi = up(ni);
        sync([&] { must(k12::top2_f32(ctx, bi.addr, V, btop.addr, bsc.addr), "top2"); }); const auto hi3 = host<uint32_t>(btop, 2);
        std::snprintf(d, sizeof d, "+-0 row (%u,%u) want (0,1); all-NaN (%u,%u) want (0,1); all -inf (%u,%u) want (0,1)", hz[0], hz[1], hn[0], hn[1], hi3[0], hi3[1]);
        report("top2_f32 edge rows (added)", hz[0] == 0 && hz[1] == 1 && hn[0] == 0 && hn[1] == 1 && hi3[0] == 0 && hi3[1] == 1, d);
    }
    auto bscratch = make(k12::scratch_bytes());
    {   // non-finite hardening (test_non_finite_logits_hardening): NaN carries no mass; +inf takes it all; two +inf split it
        const int64_t v = 4096; std::vector<float> row(v); std::normal_distribution<float> n1(0.f, 1.f); for (auto &x : row) x = n1(g);
        auto br = make(v * 4), bpp = make(v * 4), bl2 = make(8);
        auto probs_of = [&](std::vector<float> r) { put(br, r); sync([&] { must(k12::probabilities(ctx, br.addr, v, 0.7f, bpp.addr, bl2.addr, bscratch.addr), "probabilities"); }); return host<float>(bpp, v); };
        auto a = row; a[10] = NAN; auto b = row; b[10] = -INFINITY;
        const auto pa = probs_of(a), pb = probs_of(b);
        auto c = row; c[77] = INFINITY; const auto pc = probs_of(c);
        auto e = row; e[77] = INFINITY; e[90] = INFINITY; const auto pe = probs_of(e);
        int64_t nz = 0; for (float x : pc) nz += x != 0;
        const bool ok = pa == pb && pa[10] == 0.f && pc[77] == 1.f && nz == 1 && pe[77] == pe[90] && std::fabs(pe[77] - 0.5f) <= 1e-6f;
        report("probabilities non-finite", ok, "NaN == -inf (no mass, bit-exact); +inf takes all; two +inf split half");
        // a row with no mass at all: p = 1 on the lowest NaN id (id 0 without one)
        std::vector<float> allnan(v, NAN); allnan[5] = NAN; auto q1 = probs_of(allnan);
        std::vector<float> allninf(v, -INFINITY); auto q2 = probs_of(allninf);
        std::vector<float> mix(v, -INFINITY); mix[40] = NAN; mix[33] = NAN; auto q3 = probs_of(mix);
        auto one_at = [&](const std::vector<float> &p, int64_t id) { int64_t n = 0; for (int64_t i = 0; i < v; ++i) n += p[i] != 0.f; return n == 1 && p[id] == 1.f; };
        report("probabilities no-mass rows (added)", one_at(q1, 0) && one_at(q2, 0) && one_at(q3, 33), "all-NaN -> p=1 on id 0; all -inf -> id 0; NaN at 33, 40 amid -inf -> id 33");
    }
    auto bprobs = make(V * 4), blse = make(8);
    const float T = 0.8f;
    auto run_probs = [&] { sync([&] { must(k12::probabilities(ctx, bp.addr, V, T, bprobs.addr, blse.addr, bscratch.addr), "probabilities"); }); return host<float>(bprobs, V); };
    const auto probs = run_probs(), probs2 = run_probs(); const double lse = host<double>(blse, 1)[0];
    std::vector<float> rprobs(V); const double rlse = ref::probabilities(rproc.data(), V, T, rprobs.data());
    {   double worst = 0; int64_t bit_equal = 0;
        for (int64_t i = 0; i < V; ++i) { if (rprobs[i] > 1e-30f) worst = std::fmax(worst, std::fabs(probs[i] - rprobs[i]) / rprobs[i]); bit_equal += bits_of(probs[i]) == bits_of(rprobs[i]); }
        char d[150]; std::snprintf(d, sizeof d, "probs worst rel %.2e (limit 1e-6), lse |diff| %.1e (limit 1e-12); repeat %s; %.3f %% of probs bit-equal", worst, std::fabs(lse - rlse), probs == probs2 ? "bit-identical" : "DIFFERS", 100.0 * double(bit_equal) / V);
        report("probabilities", worst <= 1e-6 && std::fabs(lse - rlse) <= 1e-12 && probs == probs2, d);
    }
    auto bpr = up(rprobs);  // the reference's probabilities feed the mask and draws
    auto bmask = make(W * 4);
    struct KM { int64_t k; float p, mp; };
    std::vector<KM> kms = {{50, 0.9f, 0.f}, {248320, 0.95f, 0.f}, {1, 1.f, 0.f}, {40, 0.8f, 0.05f}, {2000, 0.99f, 0.f}};
    const size_t hip_rows = kms.size();
    for (KM x : std::vector<KM>{{1024, 0.999f, 0.f}, {1025, 0.5f, 0.1f}, {5, 1.f, 0.f}, {300, 0.97f, 0.02f}, {3000, 0.999f, 0.2f}, {248320, 1.f, 0.f}, {7, 0.001f, 0.f}}) kms.push_back(x);
    for (size_t ki = 0; ki < kms.size(); ++ki) {
        const auto &km = kms[ki];
        sync([&] { must(k12::keep_mask(ctx, bpr.addr, V, km.k, km.p, km.mp, bmask.addr, bscratch.addr), "keep_mask"); }); auto got = host<uint32_t>(bmask, W);
        sync([&] { must(k12::keep_mask(ctx, bpr.addr, V, km.k, km.p, km.mp, bmask.addr, bscratch.addr), "keep_mask"); }); const auto got2 = host<uint32_t>(bmask, W);
        std::vector<uint32_t> boundary; const auto want = ref::keep_mask(rprobs.data(), V, km.k, km.p, km.mp, &boundary);
        std::vector<char> band(V, 0); for (auto b : boundary) band[b] = 1;
        int64_t diff = 0, kept = 0, in_band = 0, any_diff = 0; for (int64_t i = 0; i < V; ++i) { const bool a = ref::kept(got, i), b = ref::kept(want, i); kept += b; diff += (a != b) && !band[i]; any_diff += a != b; in_band += band[i]; }
        char d[170]; std::snprintf(d, sizeof d, "top_k %ld top_p %.3f min_p %.2f: %ld kept, %ld differ outside the band (%ld in band, %ld differ anywhere); repeat %s", (long)km.k, km.p, km.mp, (long)kept, (long)diff, (long)in_band, (long)any_diff, got == got2 ? "bit-identical" : "DIFFERS");
        report(ki < hip_rows ? "keep_mask" : "keep_mask (added shape)", diff == 0 && got == got2, d);
        // draws on this mask
        const int n = 64; std::vector<double> us(n); std::uniform_real_distribution<double> ud(0, 1); for (auto &u : us) u = ud(g);
        auto bus = up(us); auto bids = make(n * 4); auto bwm = up(want);
        sync([&] { must(k12::draw_u(ctx, bpr.addr, bwm.addr, V, bus.addr, n, bids.addr, bscratch.addr), "draw_u"); }); auto ids = host<uint32_t>(bids, n);
        sync([&] { must(k12::draw_u(ctx, bpr.addr, bwm.addr, V, bus.addr, n, bids.addr, bscratch.addr), "draw_u"); }); const auto ids2 = host<uint32_t>(bids, n);
        int bad = 0; for (int j = 0; j < n; ++j) bad += ids[j] != ref::draw(rprobs.data(), V, want, us[j]);
        const uint64_t seed = 0x1234abcdULL; const uint32_t key = 7;
        sync([&] { must(k12::draw_seeded(ctx, bpr.addr, bwm.addr, V, seed, n, key, bids.addr, bscratch.addr), "draw_seeded"); }); auto sids = host<uint32_t>(bids, n);
        int sbad = 0; for (int j = 0; j < n; ++j) sbad += sids[j] != ref::draw(rprobs.data(), V, want, ref::uniform(seed + j, key, 0, 1));
        char d2[110]; std::snprintf(d2, sizeof d2, "%d/%d draw_u and %d/%d draw_seeded differ; repeat %s", bad, n, sbad, n, ids == ids2 ? "bit-identical" : "DIFFERS");
        report("  draws on that mask", bad == 0 && sbad == 0 && ids == ids2, d2);
    }
    {   // draws also on the device's own probabilities and mask (the real pipeline), against the reference run on those same arrays
        sync([&] { must(k12::keep_mask(ctx, bprobs.addr, V, 50, 0.9f, 0.f, bmask.addr, bscratch.addr), "keep_mask"); });
        const auto dm = host<uint32_t>(bmask, W); const int n = 256; auto bids = make(n * 4);
        sync([&] { must(k12::draw_seeded(ctx, bprobs.addr, bmask.addr, V, 4242, n, 11, bids.addr, bscratch.addr), "draw_seeded"); });
        const auto ids = host<uint32_t>(bids, n); int bad = 0;
        for (int j = 0; j < n; ++j) bad += ids[j] != ref::draw(probs.data(), V, dm, ref::uniform(4242 + j, 11, 0, 1));
        char d[100]; std::snprintf(d, sizeof d, "%d/%d draws differ from sampler_ref::draw on the device's probabilities and mask", bad, n);
        report("draws on device pipeline (added)", bad == 0, d);
    }
    {   const uint32_t committed = ref::top2(rproc.data(), V).id; const int64_t n = 20;
        auto btids = make(n * 4), btlp = make(n * 4), bclp = make(4);
        auto run = [&](float temp, uint32_t cid, int64_t nn) { sync([&] { must(k12::top_logprobs(ctx, bp.addr, V, temp, 0, cid, nn, bclp.addr, btids.addr, btlp.addr, 1), "top_logprobs"); }); };
        run(T, committed, n);
        auto ti = host<uint32_t>(btids, n); auto tl = host<float>(btlp, n); const float cl = host<float>(bclp, 1)[0];
        run(T, committed, n); const auto ti2 = host<uint32_t>(btids, n); const auto tl2 = host<float>(btlp, n);
        const auto want = ref::logprobs(rproc.data(), V, T, committed, uint32_t(n));
        bool ok = std::fabs(cl - want.committed) <= 1e-6f; double worst = std::fabs(cl - want.committed);
        for (int64_t j = 0; j < n; ++j) { ok = ok && ti[j] == want.top[j].id; worst = std::fmax(worst, std::fabs(tl[j] - want.top[j].logprob)); }
        ok = ok && worst <= 1e-6 && ti == ti2 && tl == tl2;
        char d[110]; std::snprintf(d, sizeof d, "top 20 ids exact, worst |logprob diff| %.1e (limit 1e-6); repeat %s", worst, ti == ti2 && tl == tl2 ? "bit-identical" : "DIFFERS");
        report("top_logprobs", ok, d);
        // T = 0 (greedy: log_softmax of the processed logits), a different committed id, n = 5
        run(0.f, 4242, 5); ti = host<uint32_t>(btids, 5); tl = host<float>(btlp, 5); const float cl0 = host<float>(bclp, 1)[0];
        const auto w0 = ref::logprobs(rproc.data(), V, 0.f, 4242, 5);
        bool ok0 = std::fabs(cl0 - w0.committed) <= 1e-6f; double worst0 = std::fabs(cl0 - w0.committed);
        for (int j = 0; j < 5; ++j) { ok0 = ok0 && ti[j] == w0.top[j].id; worst0 = std::fmax(worst0, std::fabs(tl[j] - w0.top[j].logprob)); }
        std::snprintf(d, sizeof d, "T = 0, committed 4242, top 5 ids exact, worst |diff| %.1e", worst0);
        report("top_logprobs T=0 (added)", ok0 && worst0 <= 1e-6, d);
        // the cere_logprob_entry stride (2) and the committed id read from device memory; n = 0 writes no entry
        auto bent = make(CERE_MAX_TOP_LOGPROBS * 8), bcid = up(std::vector<uint32_t>{committed});
        std::memset(bent.map, 0xAB, CERE_MAX_TOP_LOGPROBS * 8);
        sync([&] { must(k12::top_logprobs(ctx, bp.addr, V, T, bcid.addr, 0, 3, bclp.addr, bent.addr, bent.addr + 4, 2), "top_logprobs"); });
        const auto ent = host<uint32_t>(bent, CERE_MAX_TOP_LOGPROBS * 2); bool sok = true;
        for (int j = 0; j < 3; ++j) { sok = sok && ent[2 * j] == want.top[j].id; float lp; std::memcpy(&lp, &ent[2 * j + 1], 4); sok = sok && std::fabs(lp - want.top[j].logprob) <= 1e-6f; }
        sok = sok && ent[6] == 0xABABABABu && ent[7] == 0xABABABABu;
        sync([&] { must(k12::top_logprobs(ctx, bp.addr, V, T, 0, committed, 0, bclp.addr, 0, 0, 1), "top_logprobs"); });
        report("top_logprobs stride/device id (added)", sok, "entries at stride 2 (id, logprob), committed id from device memory, entry 3 untouched, n = 0 accepted");
    }
    {   // finalize: id = drawn or top2[0]; runner_up the other of the two
        auto bt2 = up(std::vector<uint32_t>{12, 34}), bdr = up(std::vector<uint32_t>{34, 0}), bout = make(32);
        sync([&] { must(k12::finalize(ctx, bp.addr, bt2.addr, 0, bout.addr), "finalize"); });
        auto o = host<uint32_t>(bout, 4); float l0, l1; std::memcpy(&l0, &o[2], 4); std::memcpy(&l1, &o[3], 4);
        const bool a = o[0] == 12 && o[1] == 34 && l0 == rproc[12] && l1 == rproc[34];
        sync([&] { must(k12::finalize(ctx, bp.addr, bt2.addr, bdr.addr, bout.addr + 16), "finalize"); });
        o = host<uint32_t>(bout, 8); std::memcpy(&l0, &o[6], 4); std::memcpy(&l1, &o[7], 4);
        const bool b = o[4] == 34 && o[5] == 12 && l0 == rproc[34] && l1 == rproc[12];
        auto bdr2 = up(std::vector<uint32_t>{777});
        sync([&] { must(k12::finalize(ctx, bp.addr, bt2.addr, bdr2.addr, bout.addr), "finalize"); });
        o = host<uint32_t>(bout, 4); const bool c = o[0] == 777 && o[1] == 12;
        report("finalize (added)", a && b && c, "greedy (id = top2[0], runner_up = top2[1]); drawn = top2[1] -> runner_up top2[0]; drawn elsewhere -> runner_up top2[0]");
    }
    {   std::vector<uint32_t> c0(V, 0); auto bcc = up(c0); std::vector<uint32_t> com = {5, 9, 5, 248319}; auto bcom = up(com);
        sync([&] { must(k12::count_commit(ctx, bcc.addr, bcom.addr, com.size()), "count_commit"); }); auto cc = host<uint32_t>(bcc, V);
        report("count_commit", cc[5] == 2 && cc[9] == 1 && cc[248319] == 1 && cc[0] == 0, "counts[ids[i]] += 1");
    }
    arithmetic_rows();
    {   // refusals: arguments outside the contract are CERE_ERR_ARGUMENT, nothing is recorded
        ctx.begin(); int bad = 0;
        auto refused = [&](cere_status s) { bad += s != CERE_ERR_ARGUMENT; };
        refused(k12::process(ctx, bl.addr, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, bproc.addr));
        refused(k12::process(ctx, 0, V, 0, 0, 0, 1, 0, 0, 0, 1, 0, bproc.addr));
        refused(k12::process(ctx, bl.addr, V, 0, 0, 0, 1, 0, 0, 0, 1, 2, bproc.addr));  // bias count without arrays
        refused(k12::top2_f32(ctx, bp.addr, V, 0, bsc.addr));
        refused(k12::probabilities(ctx, bp.addr, V, 0.f, bprobs.addr, blse.addr, bscratch.addr));
        refused(k12::probabilities(ctx, bp.addr, V, -1.f, bprobs.addr, blse.addr, bscratch.addr));
        refused(k12::probabilities(ctx, bp.addr, V, 0.8f, bprobs.addr, blse.addr, 0));
        refused(k12::probabilities(ctx, bp.addr, V, 0.8f, bprobs.addr, blse.addr, bscratch.addr + 4));  // scratch not 8-byte aligned
        refused(k12::keep_mask(ctx, bpr.addr, V, 0, 0.9f, 0.f, bmask.addr, bscratch.addr));
        refused(k12::keep_mask(ctx, bpr.addr, V, V + 1, 0.9f, 0.f, bmask.addr, bscratch.addr));
        refused(k12::keep_mask(ctx, bpr.addr, V, 10, 0.f, 0.f, bmask.addr, bscratch.addr));
        refused(k12::keep_mask(ctx, bpr.addr, V, 10, 1.5f, 0.f, bmask.addr, bscratch.addr));
        refused(k12::keep_mask(ctx, bpr.addr, V, 10, 0.9f, 1.5f, bmask.addr, bscratch.addr));
        refused(k12::draw_u(ctx, bpr.addr, bmask.addr, V, 0, 4, bprobs.addr, bscratch.addr));
        refused(k12::draw_seeded(ctx, bpr.addr, 0, V, 1, 4, 0, bprobs.addr, bscratch.addr));
        refused(k12::top_logprobs(ctx, bp.addr, V, 0.8f, 0, 5, 21, blse.addr, bprobs.addr, bprobs.addr, 1));  // n > CERE_MAX_TOP_LOGPROBS
        refused(k12::top_logprobs(ctx, bp.addr, V, 0.8f, 0, uint32_t(V), 0, blse.addr, 0, 0, 1));            // committed id out of range
        refused(k12::top_logprobs(ctx, bp.addr, V, -0.5f, 0, 5, 1, blse.addr, bprobs.addr, bprobs.addr, 1));
        refused(k12::count_commit(ctx, 0, bprobs.addr, 3));
        refused(k12::finalize(ctx, bp.addr, 0, 0, bprobs.addr));
        ctx.barrier(); ctx.submit_wait();
        char d[80]; std::snprintf(d, sizeof d, "20 calls outside the contract; %d not refused", bad);
        report("refusals (added)", bad == 0, d);
    }

    // ---- decode-shape timing: GPU timestamps, median of kReps, one row of V = 248,320 (the HIP binary times the same stages with hipEvents) ----
    std::printf("\nK12 timing, V = %ld, median of %d samples of %d back-to-back calls / %d (GPU timestamps), ms per call\n", (long)V, kReps, kBatch, kBatch);
    auto timeit = [&](const char *name, const std::function<void()> &rec) {
        ctx.begin();
        for (int i = 0; i < 3; ++i) { rec(); ctx.barrier(); }
        for (int i = 0; i < kReps; ++i) { ctx.stamp(2 * i); for (int b = 0; b < kBatch; ++b) { rec(); ctx.barrier(); } ctx.stamp(2 * i + 1); }
        ctx.submit_wait();
        std::vector<double> t; for (int i = 0; i < kReps; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1) / kBatch);
        std::printf("TIME %-44s %8.4f\n", name, median(t));
    };
    auto bscore = make(16), bdraw = make(16), bcnt = up(D.counts);
    auto bkeep = make(W * 4), btl = make(20 * 8), bcl = make(4), bt2i = make(8), bt2s = make(8);
    sync([&] { must(k12::top2_f32(ctx, bp.addr, V, bt2i.addr, bt2s.addr), "top2"); });
    auto stage_process = [&] { must(k12::process(ctx, bl.addr, V, bc.addr, fp, pp, rp, bm.addr, bbi.addr, bbv.addr, 1, 4, bproc.addr), "process"); };
    auto stage_top2 = [&] { must(k12::top2_f32(ctx, bp.addr, V, bt2i.addr, bt2s.addr), "top2"); };
    auto stage_probs = [&] { must(k12::probabilities(ctx, bp.addr, V, T, bprobs.addr, blse.addr, bscratch.addr), "probs"); };
    auto stage_keep = [&](int64_t k, float tp, float mp) { return [&, k, tp, mp] { must(k12::keep_mask(ctx, bpr.addr, V, k, tp, mp, bkeep.addr, bscratch.addr), "keep"); }; };
    auto stage_draw = [&] { must(k12::draw_seeded(ctx, bpr.addr, bkeep.addr, V, 1234, 1, 99, bdraw.addr, bscratch.addr), "draw"); };
    auto stage_final = [&] { must(k12::finalize(ctx, bp.addr, bt2i.addr, bdraw.addr, bscore.addr), "finalize"); };
    auto stage_lp = [&](int64_t n) { return [&, n] { must(k12::top_logprobs(ctx, bp.addr, V, T, bdraw.addr, 0, n, bcl.addr, btl.addr, btl.addr + 4 * 20, 1), "lp"); }; };
    auto stage_count = [&] { must(k12::count_commit(ctx, bcnt.addr, bdraw.addr, 1), "count"); };
    sync([&] { stage_keep(50, 0.9f, 0.f)(); });
    timeit("process (counts+mask+bias)", stage_process);
    timeit("top2_f32", stage_top2);
    timeit("probabilities", stage_probs);
    timeit("keep_mask k50 p0.9", stage_keep(50, 0.9f, 0.f));
    timeit("keep_mask k40 p0.8 minp0.05", stage_keep(40, 0.8f, 0.05f));
    timeit("keep_mask k2000 p0.99 (whole-row path)", stage_keep(2000, 0.99f, 0.f));
    timeit("keep_mask k248320 p0.95 (whole-row path)", stage_keep(248320, 0.95f, 0.f));
    sync([&] { stage_keep(50, 0.9f, 0.f)(); });  // the draws run on the decode mask (50 kept), not the dense one of the whole-row rows above
    timeit("draw_seeded n=1", stage_draw);
    timeit("finalize", stage_final);
    timeit("top_logprobs n=5", stage_lp(5));
    timeit("top_logprobs n=20", stage_lp(20));
    timeit("count_commit n=1", stage_count);
    auto chain = [&] { stage_process(); ctx.barrier(); stage_top2(); ctx.barrier(); stage_probs(); ctx.barrier(); stage_keep(50, 0.9f, 0.f)(); ctx.barrier();
                       stage_draw(); ctx.barrier(); stage_final(); ctx.barrier(); stage_count(); };
    timeit("sampled step x1 (process..count_commit)", chain);
    timeit("sampled step x4 (4 verify rows, no accept)", [&] { for (int r = 0; r < 4; ++r) { chain(); ctx.barrier(); } });
    auto chain_lp = [&] { chain(); ctx.barrier(); stage_lp(5)(); };
    timeit("sampled step x1 + top_logprobs n=5", chain_lp);
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

#else  // K12_HIP_TIMING: the same stages through the HIP K12 kernels, hipEvents around each call
namespace k12 = cere::k12;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
// Host-side probabilities/processed row for the timing: computed with the reference formulas is unnecessary, so the HIP binary derives the
// same inputs from the device (process -> probabilities) exactly as the Vulkan binary's reference-fed stages would see: only the data
// shape matters for timing, and both sides use the HIP test's seed-17 BF16 logits.
int main() {
    hipDeviceProp_t prop; HCK(hipGetDeviceProperties(&prop, 0)); std::printf("device %s\n", prop.gcnArchName);
    Data D; const int64_t V = D.V; const size_t W = D.W; const float fp = 0.3f, pp = 0.2f, rp = 1.15f, T = 0.8f;
    auto *dl = dev(D.logits); auto *dc = dev(D.counts); auto *dm = dev(D.mask); auto *dbi = dev(D.bias_ids); auto *dbv = dev(D.bias_v);
    float *dproc, *dprobs, *dtl, *dcl; HCK(hipMalloc(&dproc, V * 4)); HCK(hipMalloc(&dprobs, V * 4)); HCK(hipMalloc(&dtl, 20 * 8)); HCK(hipMalloc(&dcl, 4));
    double *dlse; HCK(hipMalloc(&dlse, 8)); void *scratch; HCK(hipMalloc(&scratch, k12::scratch_bytes()));
    uint32_t *dmask, *dids, *dt2i, *dcnt = dev(D.counts); float *dt2s; cere_token_score *dscore;
    HCK(hipMalloc(&dmask, W * 4 + 256)); HCK(hipMalloc(&dids, 16)); HCK(hipMalloc(&dt2i, 8)); HCK(hipMalloc(&dt2s, 8)); HCK(hipMalloc(&dscore, 16));
    HCK(k12::process(reinterpret_cast<const __hip_bfloat16 *>(dl), V, dc, fp, pp, rp, dm, dbi, dbv, 1, 4, dproc, nullptr));
    HCK(k12::top2_f32(dproc, V, dt2i, dt2s, nullptr)); HCK(k12::probabilities(dproc, V, T, dprobs, dlse, scratch, nullptr));
    HCK(k12::keep_mask(dprobs, V, 50, 0.9f, 0.f, dmask, scratch, nullptr)); HCK(k12::draw_seeded(dprobs, dmask, V, 1234, 1, 99, dids, scratch, nullptr)); HCK(hipDeviceSynchronize());
    hipEvent_t e0, e1; HCK(hipEventCreate(&e0)); HCK(hipEventCreate(&e1));
    std::printf("\nK12 timing, V = %ld, median of %d samples of %d back-to-back calls / %d (hipEvents), ms per call\n", (long)V, kReps, kBatch, kBatch);
    auto timeit = [&](const char *name, const std::function<void()> &rec) {
        for (int i = 0; i < 3; ++i) rec();
        HCK(hipDeviceSynchronize()); std::vector<double> t;
        for (int i = 0; i < kReps; ++i) { HCK(hipEventRecord(e0)); for (int b = 0; b < kBatch; ++b) rec(); HCK(hipEventRecord(e1)); HCK(hipEventSynchronize(e1)); float ms; HCK(hipEventElapsedTime(&ms, e0, e1)); t.push_back(ms / kBatch); }
        std::printf("TIME %-44s %8.4f\n", name, median(t));
    };
    auto stage_process = [&] { HCK(k12::process(reinterpret_cast<const __hip_bfloat16 *>(dl), V, dc, fp, pp, rp, dm, dbi, dbv, 1, 4, dproc, nullptr)); };
    auto stage_top2 = [&] { HCK(k12::top2_f32(dproc, V, dt2i, dt2s, nullptr)); };
    auto stage_probs = [&] { HCK(k12::probabilities(dproc, V, T, dprobs, dlse, scratch, nullptr)); };
    auto stage_keep = [&](int64_t k, float tp, float mp) { return [&, k, tp, mp] { HCK(k12::keep_mask(dprobs, V, k, tp, mp, dmask, scratch, nullptr)); }; };
    auto stage_draw = [&] { HCK(k12::draw_seeded(dprobs, dmask, V, 1234, 1, 99, dids, scratch, nullptr)); };
    auto stage_final = [&] { HCK(k12::finalize(dproc, dt2i, dids, dscore, nullptr)); };
    auto stage_lp = [&](int64_t n) { return [&, n] { HCK(k12::top_logprobs(dproc, V, T, dids, 0, n, dcl, reinterpret_cast<uint32_t *>(dtl), dtl + 20, 1, nullptr)); }; };
    auto stage_count = [&] { HCK(k12::count_commit(dcnt, dids, 1, nullptr)); };
    HCK(k12::keep_mask(dprobs, V, 50, 0.9f, 0.f, dmask, scratch, nullptr));
    timeit("process (counts+mask+bias)", stage_process);
    timeit("top2_f32", stage_top2);
    timeit("probabilities", stage_probs);
    timeit("keep_mask k50 p0.9", stage_keep(50, 0.9f, 0.f));
    timeit("keep_mask k40 p0.8 minp0.05", stage_keep(40, 0.8f, 0.05f));
    timeit("keep_mask k2000 p0.99 (whole-row path)", stage_keep(2000, 0.99f, 0.f));
    timeit("keep_mask k248320 p0.95 (whole-row path)", stage_keep(248320, 0.95f, 0.f));
    HCK(k12::keep_mask(dprobs, V, 50, 0.9f, 0.f, dmask, scratch, nullptr));  // the draws run on the decode mask (50 kept)
    timeit("draw_seeded n=1", stage_draw);
    timeit("finalize", stage_final);
    timeit("top_logprobs n=5", stage_lp(5));
    timeit("top_logprobs n=20", stage_lp(20));
    timeit("count_commit n=1", stage_count);
    auto chain = [&] { stage_process(); stage_top2(); stage_probs(); stage_keep(50, 0.9f, 0.f)(); stage_draw(); stage_final(); stage_count(); };
    timeit("sampled step x1 (process..count_commit)", chain);
    timeit("sampled step x4 (4 verify rows, no accept)", [&] { for (int r = 0; r < 4; ++r) chain(); });
    timeit("sampled step x1 + top_logprobs n=5", [&] { chain(); stage_lp(5)(); });
    return 0;
}
#endif
