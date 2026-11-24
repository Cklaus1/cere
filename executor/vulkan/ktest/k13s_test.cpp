// VulkanAmd P2 K13S: executor/hip/ktest/k13s_test.cpp's contract rows on the Vulkan speculative accept / resample (shaders/k13s_accept.comp +
// the K12 draw), the same data and host implementation of the header's rule (counter RNG from CERE's host/sampler_ref; fp64 sums over
// 1024 chunks of ceil(vocab / 1024) ids, then chunk order; resample restarted at each chunk's prefix; bonus = sampler_ref::draw with
// u = uniform(seed, key0 + n, 0, 1)). Bit-exact: spec_accept over 200 seeds of a 4-draft chain (committed ids and count), the peaked rows
// (whole-chain accepts and the bonus draw), row_trials over 20,000 trials (ids and accept flags) and the accept rate vs p(d) / S.
// Rows added over the HIP test: repeat bit-identity, n = 0, an out-of-vocabulary and a masked-out draft, chains of 1..15 drafts, the refusals.
// Then a timing table of spec_accept at the verify widths (GPU timestamps, median of 15). Built with -DK12_HIP_TIMING under hipcc it is instead
// the same-method HIP timing of the HIP kernels. Exit 0 when every row passes.
#ifdef K12_HIP_TIMING
#include <hip/hip_runtime.h>
#include "kernels/k12_sampler.h"
#include "kernels/k13s_spec_accept.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#ifndef K12_HIP_TIMING
#include "host/sampler_ref.h"
#include "kernels/k12_sampler_vk.h"
#include "kernels/k13s_spec_accept_vk.h"
#include "launch/vk_launch.h"
#include "launch/vk_launch_testing.h"
#endif

static int g_fail = 0;
static double median(std::vector<double> t) { std::sort(t.begin(), t.end()); return t[t.size() / 2]; }
static const int kReps = 15;
static const int kBatch = 10;  // calls per timed sample (divided by kBatch), on both sides

#ifndef K12_HIP_TIMING
using namespace cere;
namespace ref = cere::sampler_ref;
static vk::Context *g_ctx;
static vk::Buffer make(uint64_t bytes) { vk::Buffer b; if (!g_ctx->alloc(bytes + 256, b)) std::exit(2); std::memset(b.map, 0, bytes + 256); return b; }
template <class T> static vk::Buffer up(const std::vector<T> &h) { vk::Buffer b = make(h.size() * sizeof(T)); std::memcpy(b.map, h.data(), h.size() * sizeof(T)); return b; }
template <class T> static std::vector<T> host(const vk::Buffer &b, size_t n) { std::vector<T> h(n); std::memcpy(h.data(), b.map, n * sizeof(T)); return h; }
static void must(cere_status s, const char *what) { if (s != CERE_OK) { std::printf("FATAL %s returned %d\n", what, int(s)); std::exit(2); } }
static void sync(const std::function<void()> &f) { g_ctx->begin(); f(); g_ctx->barrier(); g_ctx->submit_wait(); }
static void report(const char *name, bool ok, const char *detail) { std::printf("%-26s %s  %s\n", name, ok ? "PASS" : "FAIL", detail); if (!ok) ++g_fail; }

constexpr int64_t kChunks = 1024;
// Per (row, skipped id): the chunk prefixes of the fp64 kept mass in K12's order (chunk sums in id order, then chunk order) and the kept ids
// per chunk. They depend on the row and the skipped id only, so they are computed once.
struct Prefix { std::vector<double> prefix; std::vector<std::vector<int64_t>> ids; };
static std::map<std::pair<const float *, int64_t>, Prefix> g_memo;  // keyed by the row's address: reset whenever the rows are reallocated
static const Prefix &prefixes(const float *p, const uint32_t *m, int64_t V, int64_t skip) {
    auto &memo = g_memo;
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
// The host chain: the accepted drafts, then the resampled id or (every draft accepted) the bonus draw.
static std::vector<uint32_t> host_chain(const std::vector<float> &P, const std::vector<uint32_t> &M, int64_t V, int64_t W, int64_t n,
                                        const std::vector<uint32_t> &drafts, uint64_t seed, uint32_t key0) {
    std::vector<uint32_t> want; int64_t j = 0;
    for (; j < n; ++j) { const auto dd = decide(&P[j * V], &M[j * W], V, drafts[j], seed, key0 + uint32_t(j)); want.push_back(dd.id); if (!dd.accept) break; }
    if (j == n) { std::vector<uint32_t> mk(&M[n * W], &M[n * W] + W); want.push_back(ref::draw(&P[n * V], V, mk, ref::uniform(seed, key0 + uint32_t(n), 0, 1))); }
    return want;
}

int main(int argc, char **argv) {
    const std::string spv = argc > 1 ? argv[1] : "build/vulkan-k12/spv";
    auto ctx_owner = cere::vk::testing::create_or_die(spv); cere::vk::Context &ctx = *ctx_owner; g_ctx = &ctx;
    std::printf("device %s (%s %s)\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str());
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
    auto bP = up(P), bM = up(M), bD = up(drafts);
    auto bcom = make((n + 1) * 4), bcnt = make(4);
    auto bscratch = make(std::max(vkk13s::scratch_bytes(n), vkk13s::trial_scratch_bytes()));
    int mismatch = 0; int hist[n + 2] = {0}; int nondet = 0;
    for (uint64_t s = 0; s < 200; ++s) {
        const uint64_t seed = 0x9e3779b97f4a7c15ULL * (s + 1); const uint32_t key0 = 1000 + uint32_t(s);
        sync([&] { must(vkk13s::spec_accept(ctx, bP.addr, bM.addr, V, bD.addr, n, seed, key0, bcom.addr, bcnt.addr, bscratch.addr), "spec_accept"); });
        const auto com = host<uint32_t>(bcom, n + 1); const uint32_t cnt = host<uint32_t>(bcnt, 1)[0];
        if (s % 20 == 0) {  // repeat bit-identity on a sample of the seeds
            sync([&] { must(vkk13s::spec_accept(ctx, bP.addr, bM.addr, V, bD.addr, n, seed, key0, bcom.addr, bcnt.addr, bscratch.addr), "spec_accept"); });
            const auto com2 = host<uint32_t>(bcom, n + 1); nondet += com2 != com || host<uint32_t>(bcnt, 1)[0] != cnt;
        }
        const auto want = host_chain(P, M, V, W, n, drafts, seed, key0);
        bool eq = cnt == want.size(); for (size_t i = 0; eq && i < want.size(); ++i) eq = com[i] == want[i];
        mismatch += !eq; hist[cnt <= n + 1 ? cnt : 0]++;
    }
    {   char d[200]; std::snprintf(d, sizeof d, "200 seeds, 4 drafts: %d mismatches, repeat %s; committed counts 1:%d 2:%d 3:%d 4:%d 5:%d", mismatch, nondet ? "DIFFERS" : "bit-identical", hist[1], hist[2], hist[3], hist[4], hist[5]);
        report("spec_accept", mismatch == 0 && nondet == 0, d); }
    {   // peaked rows: each draft carries ~97 % of its row, so whole-chain accepts (and the bonus draw) occur
        std::vector<float> PP((n + 1) * V); std::vector<uint32_t> MM((n + 1) * W);
        for (int64_t j = 0; j <= n; ++j) {
            std::vector<float> logits(V); for (auto &x : logits) x = nd(g);
            logits[drafts[std::min<int64_t>(j, n - 1)]] = 30.f;  // dominant id (the bonus row too)
            ref::probabilities(logits.data(), V, 0.9f, &PP[j * V]);
            auto mk = ref::keep_mask(&PP[j * V], V, 40, 0.999f, 0.f); std::copy(mk.begin(), mk.end(), &MM[j * W]);
        }
        g_memo.clear();
        auto bPP = up(PP), bMM = up(MM); int mm = 0; int hh[n + 2] = {0};
        for (uint64_t s2 = 0; s2 < 200; ++s2) {
            const uint64_t seed = 0x51ed27ULL * (s2 + 7); const uint32_t key0 = 50000 + uint32_t(s2);
            sync([&] { must(vkk13s::spec_accept(ctx, bPP.addr, bMM.addr, V, bD.addr, n, seed, key0, bcom.addr, bcnt.addr, bscratch.addr), "spec_accept"); });
            const auto com = host<uint32_t>(bcom, n + 1); const uint32_t cnt = host<uint32_t>(bcnt, 1)[0];
            const auto want = host_chain(PP, MM, V, W, n, drafts, seed, key0);
            bool eq = cnt == want.size(); for (size_t i = 0; eq && i < want.size(); ++i) eq = com[i] == want[i];
            mm += !eq; hh[cnt <= n + 1 ? cnt : 0]++;
        }
        const bool ok = mm == 0 && hh[n + 1] > 0;
        char d[200]; std::snprintf(d, sizeof d, "peaked rows, 200 seeds: %d mismatches; counts 1:%d 2:%d 3:%d 4:%d 5(all + bonus):%d", mm, hh[1], hh[2], hh[3], hh[4], hh[5]);
        report("spec_accept", ok, d);
    }
    g_memo.clear();  // the peaked rows' (freed) addresses may be reused
    {   const int64_t trials = 20000; const uint32_t d = drafts[0];
        auto btid = make(trials * 4), btacc = make(trials);
        sync([&] { must(vkk13s::row_trials(ctx, bP.addr, bM.addr, V, d, 77, 13, 5, 3, trials, btid.addr, btacc.addr, bscratch.addr), "row_trials"); });
        const auto ids = host<uint32_t>(btid, trials); const auto acc = host<uint8_t>(btacc, trials);
        sync([&] { must(vkk13s::row_trials(ctx, bP.addr, bM.addr, V, d, 77, 13, 5, 3, trials, btid.addr, btacc.addr, bscratch.addr), "row_trials"); });
        const bool det = host<uint32_t>(btid, trials) == ids && host<uint8_t>(btacc, trials) == acc;
        int64_t bad = 0, accepted = 0;
        for (int64_t t = 0; t < trials; ++t) { const auto dd = decide(P.data(), M.data(), V, d, 77 + uint64_t(t) * 13, 5 + uint32_t(t) * 3);
            bad += ids[t] != dd.id || bool(acc[t]) != dd.accept; accepted += acc[t]; }
        const double want_rate = double(P[d]) / kept_total(P.data(), M.data(), V, -1), rate = double(accepted) / trials;
        const double sigma = std::sqrt(want_rate * (1 - want_rate) / trials);
        const bool ok = bad == 0 && std::fabs(rate - want_rate) <= 5 * sigma && det;
        char dd[220]; std::snprintf(dd, sizeof dd, "20,000 trials: %ld differ from the host rule; accept rate %.4f vs p(d)/S %.4f (5 sigma %.4f); repeat %s", (long)bad, rate, want_rate, 5 * sigma, det ? "bit-identical" : "DIFFERS");
        report("row_trials", ok, dd);
        // a draft outside the vocabulary and a draft whose bit is clear in the mask are never accepted (ids still resampled as the host does)
        int64_t bad2 = 0, acc2 = 0;
        uint32_t unkept = 0; while ((M[unkept / 32] >> (unkept % 32)) & 1u) ++unkept;
        for (uint32_t dr : {uint32_t(V), uint32_t(V + 12345), unkept}) {
            sync([&] { must(vkk13s::row_trials(ctx, bP.addr, bM.addr, V, dr, 5, 1, 9, 1, 2000, btid.addr, btacc.addr, bscratch.addr), "row_trials"); });
            const auto i2 = host<uint32_t>(btid, 2000); const auto a2 = host<uint8_t>(btacc, 2000);
            for (int64_t t = 0; t < 2000; ++t) { const auto dd2 = decide(P.data(), M.data(), V, dr, 5 + uint64_t(t), 9 + uint32_t(t)); bad2 += i2[t] != dd2.id || bool(a2[t]) != dd2.accept; acc2 += a2[t]; }
        }
        char d3[120]; std::snprintf(d3, sizeof d3, "3 drafts (id = vocab, id >> vocab, a masked-out id) x 2000 trials: %ld differ, %ld accepted (must be 0)", (long)bad2, (long)acc2);
        report("row_trials never-accepted", bad2 == 0 && acc2 == 0, d3);
    }
    {   // chains of 0 .. 15 drafts on peaked-ish rows reuse rows modulo 5: the host chain with the same rows
        int bad = 0; int total = 0;
        for (int64_t nn : {0, 1, 2, 3, 7, 15}) {
            std::vector<float> PX((nn + 1) * V); std::vector<uint32_t> MX((nn + 1) * W), dr(std::max<int64_t>(nn, 1));
            for (int64_t j = 0; j <= nn; ++j) { std::copy(&P[(j % 5) * V], &P[(j % 5) * V] + V, &PX[j * V]); std::copy(&M[(j % 5) * W], &M[(j % 5) * W] + W, &MX[j * W]); if (j < nn) dr[j] = ref::top2(&P[(j % 5) * V], V).id; }
            g_memo.clear();
            auto bpx = up(PX), bmx = up(MX), bdr = up(dr), bc2 = make((nn + 1) * 4), bn2 = make(4), bsc = make(vkk13s::scratch_bytes(nn));
            for (uint64_t s = 0; s < 20; ++s) {
                const uint64_t seed = 77 * (s + 1) + nn; const uint32_t key0 = 9000 + uint32_t(s);
                sync([&] { must(vkk13s::spec_accept(ctx, bpx.addr, bmx.addr, V, nn ? bdr.addr : 0, nn, seed, key0, bc2.addr, bn2.addr, bsc.addr), "spec_accept"); });
                const auto com = host<uint32_t>(bc2, nn + 1); const uint32_t cnt = host<uint32_t>(bn2, 1)[0];
                const auto want = host_chain(PX, MX, V, W, nn, dr, seed, key0);
                bool eq = cnt == want.size(); for (size_t i = 0; eq && i < want.size(); ++i) eq = com[i] == want[i];
                bad += !eq; ++total;
            }
        }
        char d[100]; std::snprintf(d, sizeof d, "n in {0,1,2,3,7,15} x 20 seeds: %d of %d chains differ from the host", bad, total);
        report("spec_accept n sweep (added)", bad == 0, d);
    }
    {   ctx.begin(); int bad = 0;
        auto refused = [&](cere_status s) { bad += s != CERE_ERR_ARGUMENT; };
        refused(vkk13s::spec_accept(ctx, bP.addr, bM.addr, V, bD.addr, 16, 1, 1, bcom.addr, bcnt.addr, bscratch.addr));  // n > kMaxDrafts
        refused(vkk13s::spec_accept(ctx, bP.addr, bM.addr, V, 0, 4, 1, 1, bcom.addr, bcnt.addr, bscratch.addr));        // drafts missing
        refused(vkk13s::spec_accept(ctx, bP.addr, bM.addr, 1, bD.addr, 4, 1, 1, bcom.addr, bcnt.addr, bscratch.addr));  // vocab 1
        refused(vkk13s::spec_accept(ctx, bP.addr, bM.addr, V, bD.addr, 4, 1, 1, 0, bcnt.addr, bscratch.addr));          // committed null
        refused(vkk13s::spec_accept(ctx, bP.addr, bM.addr, V, bD.addr, 4, 1, 1, bcom.addr, bcnt.addr, 0));              // scratch null
        refused(vkk13s::spec_accept(ctx, bP.addr, bM.addr, V, bD.addr, -1, 1, 1, bcom.addr, bcnt.addr, bscratch.addr));
        refused(vkk13s::row_trials(ctx, bP.addr, bM.addr, V, 1, 1, 1, 1, 1, 10, 0, bcnt.addr, bscratch.addr));
        ctx.barrier(); ctx.submit_wait();
        char d[80]; std::snprintf(d, sizeof d, "7 calls outside the contract; %d not refused", bad);
        report("refusals (added)", bad == 0, d);
    }

    // ---- timing: spec_accept at the verify widths (n drafts, n + 1 rows), GPU timestamps, median of kReps ----
    std::printf("\nK13S timing, V = %ld, median of %d samples of %d back-to-back calls / %d (GPU timestamps), ms per call\n", (long)V, kReps, kBatch, kBatch);
    for (int64_t nn : {1, 2, 3, 4}) {
        ctx.begin();
        for (int i = 0; i < 3; ++i) { must(vkk13s::spec_accept(ctx, bP.addr, bM.addr, V, bD.addr, nn, 5, 7, bcom.addr, bcnt.addr, bscratch.addr), "sa"); ctx.barrier(); }
        for (int i = 0; i < kReps; ++i) { ctx.stamp(2 * i); for (int b = 0; b < kBatch; ++b) { must(vkk13s::spec_accept(ctx, bP.addr, bM.addr, V, bD.addr, nn, 5 + i + b, 7, bcom.addr, bcnt.addr, bscratch.addr), "sa"); ctx.barrier(); } ctx.stamp(2 * i + 1); }
        ctx.submit_wait();
        std::vector<double> t; for (int i = 0; i < kReps; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1) / kBatch);
        std::printf("TIME spec_accept n=%ld (%ld rows)%*s %8.4f\n", (long)nn, (long)(nn + 1), 22, "", median(t));
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

#else  // K12_HIP_TIMING: the same calls through the HIP K13S kernels, hipEvents around each call; rows built on the device by K12 (same distribution)
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 16)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
int main() {
    hipDeviceProp_t prop; HCK(hipGetDeviceProperties(&prop, 0)); std::printf("device %s\n", prop.gcnArchName);
    const int64_t V = 248320, W = (V + 31) / 32, n = 4;
    std::mt19937 g(3); std::normal_distribution<float> nd(0.f, 2.5f);
    std::vector<float> logits((n + 1) * V); for (auto &x : logits) x = nd(g);
    float *dlog = dev(logits), *dP; uint32_t *dM; HCK(hipMalloc(&dP, (n + 1) * V * 4)); HCK(hipMalloc(&dM, (n + 1) * W * 4)); double *dlse; HCK(hipMalloc(&dlse, 8));
    void *k12s; HCK(hipMalloc(&k12s, cere::k12::scratch_bytes()));
    std::vector<uint32_t> top(1);
    for (int64_t j = 0; j <= n; ++j) { HCK(cere::k12::probabilities(dlog + j * V, V, 0.9f, dP + j * V, dlse, k12s, nullptr)); HCK(cere::k12::keep_mask(dP + j * V, V, 40, 0.95f, 0.f, dM + j * W, k12s, nullptr)); }
    uint32_t *dt2i; float *dt2s; HCK(hipMalloc(&dt2i, 8)); HCK(hipMalloc(&dt2s, 8)); std::vector<uint32_t> drafts(n);
    for (int64_t j = 0; j < n; ++j) { HCK(cere::k12::top2_f32(dlog + j * V, V, dt2i, dt2s, nullptr)); HCK(hipMemcpy(&drafts[j], dt2i, 4, hipMemcpyDeviceToHost)); }
    uint32_t *dD = dev(drafts), *dcom, *dcnt; HCK(hipMalloc(&dcom, (n + 1) * 4)); HCK(hipMalloc(&dcnt, 4));
    void *scratch; HCK(hipMalloc(&scratch, cere::k13s::scratch_bytes(n))); HCK(hipDeviceSynchronize());
    hipEvent_t e0, e1; HCK(hipEventCreate(&e0)); HCK(hipEventCreate(&e1));
    std::printf("\nK13S timing, V = %ld, median of %d samples of %d back-to-back calls / %d (hipEvents), ms per call\n", (long)V, kReps, kBatch, kBatch);
    for (int64_t nn : {1, 2, 3, 4}) {
        for (int i = 0; i < 3; ++i) HCK(cere::k13s::spec_accept(dP, dM, V, dD, nn, 5, 7, dcom, dcnt, scratch, nullptr));
        HCK(hipDeviceSynchronize()); std::vector<double> t;
        for (int i = 0; i < kReps; ++i) { HCK(hipEventRecord(e0)); for (int b = 0; b < kBatch; ++b) HCK(cere::k13s::spec_accept(dP, dM, V, dD, nn, 5 + i + b, 7, dcom, dcnt, scratch, nullptr)); HCK(hipEventRecord(e1)); HCK(hipEventSynchronize(e1)); float ms; HCK(hipEventElapsedTime(&ms, e0, e1)); t.push_back(ms / kBatch); }
        std::printf("TIME spec_accept n=%ld (%ld rows)%*s %8.4f\n", (long)nn, (long)(nn + 1), 22, "", median(t));
    }
    return 0;
}
#endif
