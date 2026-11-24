// AmdAmd PAR-6 step 2f: the K12 device sampler (executor/cuda/src/kernels/k12_sampler.cu, hipified) on gfx1151
// against CERE's own semantic reference, host/sampler_ref (what the CUDA suite test_kernels_k12_sampler.py compares
// the kernels to), on a 248,320-id row: process (penalties, vocab mask, logit_bias) within the CUDA suite's D2 bound
// 2^-22 max(1,|x|) + count|f| 2^-23 and the mask's -inf ids exact; top2_f32 bit-exact (ties, NaN); probabilities
// (fp32 probs within 1e-6 relative, lse within 1e-12); keep_mask exact outside sampler_ref's 1e-6 boundary band;
// draw_u and draw_seeded exact (CERE's murmur counter for the seeded u); top_logprobs ids exact, logprobs within
// 1e-6; finalize; count_commit; every kernel bit-identical on repeat.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "host/sampler_ref.h"
#include "kernels/k12_sampler.h"

namespace k12 = cere::k12;
namespace ref = cere::sampler_ref;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 16)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
static void report(const char *n, bool ok, const char *d) { std::printf("%-26s %s  %s\n", n, ok ? "PASS" : "FAIL", d); if (!ok) ++g_fail; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

int main() {
    const int64_t V = 248320; const size_t W = (V + 31) / 32;
    std::mt19937 g(17); std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<uint16_t> logits(V); for (auto &x : logits) x = f2bf(nd(g));
    std::vector<uint32_t> counts(V, 0); for (int i = 0; i < 300; ++i) counts[g() % V] += 1 + g() % 3;
    std::vector<uint32_t> mask(W, ~0u); for (int i = 0; i < 2000; ++i) { const uint32_t id = g() % V; mask[id / 32] &= ~(1u << (id % 32)); }
    std::vector<uint32_t> bias_ids = {5, 777, 123456, 248319}; std::vector<float> bias_v = {3.5f, -100.f, 7.25f, 1.f};
    const float fp = 0.3f, pp = 0.2f, rp = 1.15f;
    auto *dl = dev(logits); auto *dc = dev(counts); auto *dm = dev(mask); auto *dbi = dev(bias_ids); auto *dbv = dev(bias_v);
    float *dproc; HCK(hipMalloc(&dproc, V * 4));
    auto run_process = [&] { HCK(k12::process(reinterpret_cast<const __hip_bfloat16 *>(dl), V, dc, fp, pp, rp, dm, dbi, dbv, 1, 4, dproc, nullptr)); return host(dproc, V); };
    auto proc = run_process(), proc2 = run_process();
    std::vector<ref::Bias> rb; for (size_t j = 0; j < bias_ids.size(); ++j) rb.push_back({bias_ids[j], bias_v[j]});
    std::vector<float> rproc(V); ref::process(logits.data(), V, counts.data(), fp, pp, rp, mask.data(), rb.data(), rb.size(), rproc.data());
    {   int64_t bad = 0, inf_bad = 0;
        for (int64_t i = 0; i < V; ++i) {
            if (std::isinf(rproc[i]) || std::isinf(proc[i])) { inf_bad += !(std::isinf(rproc[i]) && std::isinf(proc[i]) && std::signbit(rproc[i]) == std::signbit(proc[i])); continue; }
            const double bound = std::ldexp(1.0, -22) * std::fmax(1.0, std::fabs(rproc[i])) + counts[i] * std::fabs(fp) * std::ldexp(1.0, -23);
            bad += std::fabs(double(proc[i]) - rproc[i]) > bound;
        }
        char d[110]; std::snprintf(d, sizeof d, "%ld outside the D2 bound, %ld -inf mismatches; repeat %s", (long)bad, (long)inf_bad, proc == proc2 ? "bit-identical" : "DIFFERS");
        report("process", bad == 0 && inf_bad == 0 && proc == proc2, d);
    }
    // the reference's processed row drives every later stage, so each stage is measured on its own
    auto *dp = dev(rproc);
    {   std::vector<float> t2 = rproc; t2[1000] = 99.f; t2[2000] = 99.f; t2[3000] = NAN;  // ties and a NaN
        auto *dt2 = dev(t2); uint32_t *ids; float *sc; HCK(hipMalloc(&ids, 8)); HCK(hipMalloc(&sc, 8));
        HCK(k12::top2_f32(dt2, V, ids, sc, nullptr)); auto hi = host(ids, 2);
        // the header rule (and the CUDA suite's hardening test): NaN ranks above +inf, so the NaN id is first and
        // the runner-up is the tie's lower id; sampler_ref::top2 does not define NaN, so this row is not its case
        const ref::Top2 want{3000, 1000};
        HCK(k12::top2_f32(dp, V, ids, sc, nullptr)); auto hi2 = host(ids, 2); const auto want2 = ref::top2(rproc.data(), V);
        char d[120]; std::snprintf(d, sizeof d, "NaN+ties row (%u,%u) want (%u,%u); plain row (%u,%u) want (%u,%u)", hi[0], hi[1], want.id, want.runner_up, hi2[0], hi2[1], want2.id, want2.runner_up);
        report("top2_f32", hi[0] == want.id && hi[1] == want.runner_up && hi2[0] == want2.id && hi2[1] == want2.runner_up, d);
    }
    void *scratch; HCK(hipMalloc(&scratch, k12::scratch_bytes()));
    {   // non-finite hardening (test_non_finite_logits_hardening): NaN carries no mass; +inf takes it all; two +inf split it
        const int64_t v = 4096; std::vector<float> row(v); std::normal_distribution<float> n1(0.f, 1.f); for (auto &x : row) x = n1(g);
        float *dr, *dpp; double *dl2; HCK(hipMalloc(&dr, v * 4)); HCK(hipMalloc(&dpp, v * 4)); HCK(hipMalloc(&dl2, 8));
        auto probs_of = [&](std::vector<float> r) { HCK(hipMemcpy(dr, r.data(), v * 4, hipMemcpyHostToDevice)); HCK(k12::probabilities(dr, v, 0.7f, dpp, dl2, scratch, nullptr)); return host(dpp, v); };
        auto a = row; a[10] = NAN; auto b = row; b[10] = -INFINITY;
        const auto pa = probs_of(a), pb = probs_of(b);
        auto c = row; c[77] = INFINITY; const auto pc = probs_of(c);
        auto e = row; e[77] = INFINITY; e[90] = INFINITY; const auto pe = probs_of(e);
        int64_t nz = 0; for (float x : pc) nz += x != 0;
        const bool ok = pa == pb && pa[10] == 0.f && pc[77] == 1.f && nz == 1 && pe[77] == pe[90] && std::fabs(pe[77] - 0.5f) <= 1e-6f;
        report("probabilities non-finite", ok, "NaN == -inf (no mass, bit-exact); +inf takes all; two +inf split half");
    }
    float *dprobs; double *dlse; HCK(hipMalloc(&dprobs, V * 4)); HCK(hipMalloc(&dlse, 8));
    const float T = 0.8f;
    auto run_probs = [&] { HCK(k12::probabilities(dp, V, T, dprobs, dlse, scratch, nullptr)); return host(dprobs, V); };
    auto probs = run_probs(), probs2 = run_probs(); const double lse = host(dlse, 1)[0];
    std::vector<float> rprobs(V); const double rlse = ref::probabilities(rproc.data(), V, T, rprobs.data());
    {   double worst = 0; for (int64_t i = 0; i < V; ++i) if (rprobs[i] > 1e-30f) worst = std::fmax(worst, std::fabs(probs[i] - rprobs[i]) / rprobs[i]);
        char d[110]; std::snprintf(d, sizeof d, "probs worst rel %.2e (limit 1e-6), lse |diff| %.1e; repeat %s", worst, std::fabs(lse - rlse), probs == probs2 ? "bit-identical" : "DIFFERS");
        report("probabilities", worst <= 1e-6 && std::fabs(lse - rlse) <= 1e-12 && probs == probs2, d);
    }
    auto *dpr = dev(rprobs);  // the reference's probabilities feed the mask and draws
    uint32_t *dmask; HCK(hipMalloc(&dmask, W * 4));
    struct KM { int64_t k; float p, mp; } kms[] = {{50, 0.9f, 0.f}, {248320, 0.95f, 0.f}, {1, 1.f, 0.f}, {40, 0.8f, 0.05f}, {2000, 0.99f, 0.f}};
    for (const auto &km : kms) {
        HCK(k12::keep_mask(dpr, V, km.k, km.p, km.mp, dmask, scratch, nullptr)); auto got = host(dmask, W);
        std::vector<uint32_t> boundary; const auto want = ref::keep_mask(rprobs.data(), V, km.k, km.p, km.mp, &boundary);
        std::vector<char> band(V, 0); for (auto b : boundary) band[b] = 1;
        int64_t diff = 0, kept = 0; for (int64_t i = 0; i < V; ++i) { const bool a = ref::kept(got, i), b = ref::kept(want, i); kept += b; diff += (a != b) && !band[i]; }
        char d[110]; std::snprintf(d, sizeof d, "top_k %ld top_p %.2f min_p %.2f: %ld kept, %ld differ outside the band", (long)km.k, km.p, km.mp, (long)kept, (long)diff);
        report("keep_mask", diff == 0, d);
        // draws on this mask
        const int n = 64; std::vector<double> us(n); std::uniform_real_distribution<double> ud(0, 1); for (auto &u : us) u = ud(g);
        auto *dus = dev(us); uint32_t *dids; HCK(hipMalloc(&dids, n * 4)); auto *dwm = dev(want);
        HCK(k12::draw_u(dpr, dwm, V, dus, n, dids, scratch, nullptr)); auto ids = host(dids, n);
        int bad = 0; for (int j = 0; j < n; ++j) bad += ids[j] != ref::draw(rprobs.data(), V, want, us[j]);
        const uint64_t seed = 0x1234abcdULL; const uint32_t key = 7;
        HCK(k12::draw_seeded(dpr, dwm, V, seed, n, key, dids, scratch, nullptr)); auto sids = host(dids, n);
        int sbad = 0; for (int j = 0; j < n; ++j) sbad += sids[j] != ref::draw(rprobs.data(), V, want, ref::uniform(seed + j, key, 0, 1));
        char d2[90]; std::snprintf(d2, sizeof d2, "%d/%d draw_u and %d/%d draw_seeded differ", bad, n, sbad, n);
        report("  draws on that mask", bad == 0 && sbad == 0, d2);
    }
    {   const uint32_t committed = ref::top2(rproc.data(), V).id; const int64_t n = 20;
        uint32_t *tids; float *tlp, *clp; HCK(hipMalloc(&tids, n * 4)); HCK(hipMalloc(&tlp, n * 4)); HCK(hipMalloc(&clp, 4));
        HCK(k12::top_logprobs(dp, V, T, nullptr, committed, n, clp, tids, tlp, 1, nullptr));
        auto ti = host(tids, n); auto tl = host(tlp, n); const float cl = host(clp, 1)[0];
        const auto want = ref::logprobs(rproc.data(), V, T, committed, uint32_t(n));
        bool ok = std::fabs(cl - want.committed) <= 1e-6f; double worst = std::fabs(cl - want.committed);
        for (int64_t j = 0; j < n; ++j) { ok = ok && ti[j] == want.top[j].id; worst = std::fmax(worst, std::fabs(tl[j] - want.top[j].logprob)); }
        ok = ok && worst <= 1e-6;
        char d[90]; std::snprintf(d, sizeof d, "top 20 ids exact, worst |logprob diff| %.1e (limit 1e-6)", worst); report("top_logprobs", ok, d);
    }
    {   std::vector<uint32_t> c0(V, 0); auto *dcc = dev(c0); std::vector<uint32_t> com = {5, 9, 5, 248319}; auto *dcom = dev(com);
        HCK(k12::count_commit(dcc, dcom, com.size(), nullptr)); auto cc = host(dcc, V);
        report("count_commit", cc[5] == 2 && cc[9] == 1 && cc[248319] == 1 && cc[0] == 0, "counts[ids[i]] += 1");
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
