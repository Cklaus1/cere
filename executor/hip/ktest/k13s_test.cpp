// AmdAmd PAR-6 step 2g: K13S speculative accept/resample (executor/cuda/src/kernels/k13s_spec_accept.cu, hipified) on
// gfx1151 against a host implementation of the header's rule (counter RNG from CERE's host/sampler_ref; fp64 sums
// over k12::kChunks = 1024 chunks of ceil(vocab / 1024) ids, then chunk order; resample restarted at each chunk's
// prefix; bonus = sampler_ref::draw with u = uniform(seed, key0 + n, 0, 1), which the K12 test shows equals
// draw_seeded). Bit-exact: spec_accept over 200 seeds of a 4-draft chain (committed ids and count), row_trials over
// 20,000 trials (ids and accept flags), and the accept rate vs p(d) / S.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <utility>
#include <random>
#include <vector>

#include "host/sampler_ref.h"
#include "kernels/k13s_spec_accept.h"

namespace ref = cere::sampler_ref;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 16)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }

constexpr int64_t kChunks = 1024;
// Per (row, skipped id): the chunk prefixes of the fp64 kept mass in K12's order (chunk sums in id order, then chunk
// order) and the kept ids per chunk. They depend on the row and the skipped id only, so they are computed once.
struct Prefix { std::vector<double> prefix; std::vector<std::vector<int64_t>> ids; };
static const Prefix &prefixes(const float *p, const uint32_t *m, int64_t V, int64_t skip) {
    static std::map<std::pair<const float *, int64_t>, Prefix> memo;
    auto it = memo.find({p, skip});
    if (it != memo.end()) return it->second;
    const int64_t len = (V + kChunks - 1) / kChunks; Prefix pr; pr.prefix.assign(kChunks + 1, 0.0); pr.ids.resize(kChunks);
    for (int64_t c = 0; c < kChunks; ++c) {
        double cs = 0;
        for (int64_t i = c * len; i < std::min(V, (c + 1) * len); ++i) if (i != skip && ((m[i / 32] >> (i % 32)) & 1u)) { cs += p[i]; pr.ids[c].push_back(i); }
        pr.prefix[c + 1] = pr.prefix[c] + cs;
    }
    return memo.emplace(std::make_pair(p, skip), std::move(pr)).first->second;
}
static double kept_total(const float *p, const uint32_t *m, int64_t V, int64_t skip) { return prefixes(p, m, V, skip).prefix[kChunks]; }
// inverse CDF over kept ids other than `skip`, running sum restarted at each chunk's prefix
static uint32_t resample(const float *p, const uint32_t *m, int64_t V, int64_t skip, double u) {
    const Prefix &pr = prefixes(p, m, V, skip); const double target = u * pr.prefix[kChunks]; int64_t last = -1;
    for (int64_t c = 0; c < kChunks; ++c) {
        if (!pr.ids[c].empty()) last = pr.ids[c].back();
        if (!(pr.prefix[c + 1] > target)) continue;
        double acc = pr.prefix[c];
        for (int64_t i : pr.ids[c]) { acc += p[i]; if (acc > target) return uint32_t(i); }
    }
    return uint32_t(last);
}
struct Decision { bool accept; uint32_t id; };
static Decision decide(const float *p, const uint32_t *m, int64_t V, uint32_t d, uint64_t seed, uint32_t key) {
    const double u = ref::uniform(seed, key, 4, 5);
    const bool kept = d < V && ((m[d / 32] >> (d % 32)) & 1u);
    if (kept && u < double(p[d]) / kept_total(p, m, V, -1)) return {true, d};
    return {false, resample(p, m, V, d < V ? int64_t(d) : -1, ref::uniform(seed, key, 6, 7))};
}

int main() {
    const int64_t V = 248320, W = (V + 31) / 32, n = 4;
    std::mt19937 g(3); std::normal_distribution<float> nd(0.f, 2.5f);
    std::vector<float> P((n + 1) * V); std::vector<uint32_t> M((n + 1) * W); std::vector<uint32_t> drafts(n);
    for (int64_t j = 0; j <= n; ++j) {
        std::vector<float> logits(V); for (auto &x : logits) x = nd(g);
        ref::probabilities(logits.data(), V, 0.9f, &P[j * V]);
        auto mk = ref::keep_mask(&P[j * V], V, 40, 0.95f, 0.f); std::copy(mk.begin(), mk.end(), &M[j * W]);
        if (j < n) drafts[j] = ref::top2(&P[j * V], V).id;  // the row's most likely id: accepted often, not always
    }
    drafts[2] = ref::top2(&P[2 * V], V).runner_up;
    auto *dP = dev(P); auto *dM = dev(M); auto *dD = dev(drafts);
    uint32_t *dcom, *dcnt; HCK(hipMalloc(&dcom, (n + 1) * 4)); HCK(hipMalloc(&dcnt, 4));
    void *scratch; HCK(hipMalloc(&scratch, std::max(cere::k13s::scratch_bytes(n), cere::k13s::trial_scratch_bytes())));
    int mismatch = 0; int hist[n + 2] = {0};
    for (uint64_t s = 0; s < 200; ++s) {
        const uint64_t seed = 0x9e3779b97f4a7c15ULL * (s + 1); const uint32_t key0 = 1000 + uint32_t(s);
        HCK(cere::k13s::spec_accept(dP, dM, V, dD, n, seed, key0, dcom, dcnt, scratch, nullptr));
        const auto com = host(dcom, n + 1); const uint32_t cnt = host(dcnt, 1)[0];
        std::vector<uint32_t> want; int64_t j = 0;
        for (; j < n; ++j) { const auto dd = decide(&P[j * V], &M[j * W], V, drafts[j], seed, key0 + uint32_t(j)); want.push_back(dd.id); if (!dd.accept) break; }
        if (j == n) { std::vector<uint32_t> mk(&M[n * W], &M[n * W] + W); want.push_back(ref::draw(&P[n * V], V, mk, ref::uniform(seed, key0 + uint32_t(n), 0, 1))); }
        bool eq = cnt == want.size(); for (size_t i = 0; eq && i < want.size(); ++i) eq = com[i] == want[i];
        mismatch += !eq; hist[cnt <= n + 1 ? cnt : 0]++;
    }
    std::printf("%-22s %s  200 seeds, 4 drafts: %d mismatches; committed counts 1:%d 2:%d 3:%d 4:%d 5:%d\n", "spec_accept",
                mismatch ? "FAIL" : "PASS", mismatch, hist[1], hist[2], hist[3], hist[4], hist[5]);
    if (mismatch) ++g_fail;
    {   // peaked rows: each draft carries ~97 % of its row, so whole-chain accepts (and the bonus draw) occur
        std::vector<float> PP((n + 1) * V); std::vector<uint32_t> MM((n + 1) * W);
        for (int64_t j = 0; j <= n; ++j) {
            std::vector<float> logits(V); for (auto &x : logits) x = nd(g);
            logits[drafts[std::min<int64_t>(j, n - 1)]] = 30.f;  // dominant id (the bonus row too)
            ref::probabilities(logits.data(), V, 0.9f, &PP[j * V]);
            auto mk = ref::keep_mask(&PP[j * V], V, 40, 0.999f, 0.f); std::copy(mk.begin(), mk.end(), &MM[j * W]);
        }
        auto *dPP = dev(PP); auto *dMM = dev(MM); int mm = 0; int hh[n + 2] = {0};
        for (uint64_t s2 = 0; s2 < 200; ++s2) {
            const uint64_t seed = 0x51ed27ULL * (s2 + 7); const uint32_t key0 = 50000 + uint32_t(s2);
            HCK(cere::k13s::spec_accept(dPP, dMM, V, dD, n, seed, key0, dcom, dcnt, scratch, nullptr));
            const auto com = host(dcom, n + 1); const uint32_t cnt = host(dcnt, 1)[0];
            std::vector<uint32_t> want; int64_t j = 0;
            for (; j < n; ++j) { const auto dd = decide(&PP[j * V], &MM[j * W], V, drafts[j], seed, key0 + uint32_t(j)); want.push_back(dd.id); if (!dd.accept) break; }
            if (j == n) { std::vector<uint32_t> mk(&MM[n * W], &MM[n * W] + W); want.push_back(ref::draw(&PP[n * V], V, mk, ref::uniform(seed, key0 + uint32_t(n), 0, 1))); }
            bool eq = cnt == want.size(); for (size_t i = 0; eq && i < want.size(); ++i) eq = com[i] == want[i];
            mm += !eq; hh[cnt <= n + 1 ? cnt : 0]++;
        }
        const bool ok = mm == 0 && hh[n + 1] > 0;
        std::printf("%-22s %s  peaked rows, 200 seeds: %d mismatches; counts 1:%d 2:%d 3:%d 4:%d 5(all + bonus):%d\n", "spec_accept",
                    ok ? "PASS" : "FAIL", mm, hh[1], hh[2], hh[3], hh[4], hh[5]);
        if (!ok) ++g_fail;
    }
    {   const int64_t trials = 20000; const uint32_t d = drafts[0];
        uint32_t *tid; uint8_t *tacc; HCK(hipMalloc(&tid, trials * 4)); HCK(hipMalloc(&tacc, trials));
        HCK(cere::k13s::row_trials(dP, dM, V, d, 77, 13, 5, 3, trials, tid, tacc, scratch, nullptr));
        const auto ids = host(tid, trials); const auto acc = host(tacc, trials);
        int64_t bad = 0, accepted = 0;
        for (int64_t t = 0; t < trials; ++t) { const auto dd = decide(P.data(), M.data(), V, d, 77 + uint64_t(t) * 13, 5 + uint32_t(t) * 3);
            bad += ids[t] != dd.id || bool(acc[t]) != dd.accept; accepted += acc[t]; }
        const double want_rate = double(P[d]) / kept_total(P.data(), M.data(), V, -1), rate = double(accepted) / trials;
        const double sigma = std::sqrt(want_rate * (1 - want_rate) / trials);
        const bool ok = bad == 0 && std::fabs(rate - want_rate) <= 5 * sigma;
        std::printf("%-22s %s  20,000 trials: %ld differ from the host rule; accept rate %.4f vs p(d)/S %.4f (5 sigma %.4f)\n", "row_trials",
                    ok ? "PASS" : "FAIL", (long)bad, rate, want_rate, 5 * sigma);
        if (!ok) ++g_fail;
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
