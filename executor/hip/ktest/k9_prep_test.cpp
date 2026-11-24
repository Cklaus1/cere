// AmdAmd PAR-6 step 5b: K9's index prep (executor/cuda/src/kernels/k9_qsa.cu, hipified) on gfx1151:
//  1. indexer_qk vs a host mirror of its arithmetic (Gemma RMSNorm with the kernel's F32 sum order: 16 lanes x 8
//     sequential FMAs then a 32-lane xor tree; BF16 normed row; NeoX RoPE over 64 dims with BF16 cos / sin and every
//     product rounded to BF16): every output within 1 BF16 ulp (the device rsqrtf), the ulp count reported;
//  2. qsa_compress_store: the compressed keys of 23 new tokens at pos0 2 (ring rows 0-1 from before) vs the host mirror
//     (F32 sequential mean of the 4 members -> BF16 -> norm -> RoPE at the first member's position), within 1 ulp;
//     the same tokens as calls of 5 + 7 + 11 (ring continuity), as 23 qsa_compress_store_step calls, and as one
//     qsa_compress_store_at (device pos0): keys and the ring rows the contract defines afterwards (the incomplete
//     last group's members; the other rows hold whatever the last call left) bit-equal to the single call.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k1_primitives.h"
#include "kernels/k9_qsa.h"

namespace k9 = cere::k9;
using bf16 = __hip_bfloat16;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
#define CCK(x) do { cere_status s_ = (x); if (s_ != CERE_OK) { std::printf("cere %s:%d status %d\n", __FILE__, __LINE__, int(s_)); std::exit(2); } } while (0)
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float rb(float x) { return bf2f(f2bf(x)); }
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
static void check(const char *name, bool ok, const char *msg) { std::printf("%-24s %s  %s\n", name, ok ? "PASS" : "FAIL", msg); if (!ok) ++g_fail; }

constexpr int DI = k9::kIndexHeadDim, HR = k9::kRotaryDim / 2;

// norm + rope of one 128 row (BF16-exact inputs x) -> BF16 bits
static void ref_row(const float *x, const std::vector<uint16_t> &w, const float *cs, float eps, uint16_t *out) {
    float lane[32] = {};
    for (int l = 0; l < 16; ++l) for (int j = 0; j < 8; ++j) lane[l] = std::fmaf(x[l * 8 + j], x[l * 8 + j], lane[l]);
    for (int mask = 16; mask >= 1; mask >>= 1) { float t[32]; for (int l = 0; l < 32; ++l) t[l] = lane[l] + lane[l ^ mask]; std::memcpy(lane, t, sizeof t); }
    const float nf = 1.f / std::sqrt(lane[0] / float(DI) + eps);
    float row[DI]; for (int d = 0; d < DI; ++d) row[d] = rb((x[d] * nf) * (1.f + bf2f(w[d])));
    for (int d = 0; d < DI; ++d) {
        float o;
        if (d < HR) o = rb(rb(row[d] * rb(cs[d])) - rb(row[d + HR] * rb(cs[HR + d])));
        else if (d < 2 * HR) o = rb(rb(row[d] * rb(cs[d - HR])) + rb(row[d - HR] * rb(cs[HR + d - HR])));
        else o = row[d];
        out[d] = f2bf(o);
    }
}

int main() {
    std::mt19937 g(37); std::normal_distribution<float> nd;
    const int64_t NPOS = 4096; float *dcs; HCK(hipMalloc(&dcs, NPOS * k9::kRotaryDim * 4));
    HCK(cere::k1::rope_table(dcs, NPOS, k9::kRotaryDim, 1e7f, nullptr)); auto cs = host(dcs, size_t(NPOS) * k9::kRotaryDim);
    const float eps = 1e-6f;
    std::vector<uint16_t> wq(DI), wk(DI); for (auto &x : wq) x = f2bf(nd(g) * 0.1f); for (auto &x : wk) x = f2bf(nd(g) * 0.1f);
    auto *dwq = dev(wq), *dwk = dev(wk);
    k9::Bounds bnd{0, NPOS, NPOS, nullptr};
    {   // 1. indexer_qk
        const int64_t T = 37, pos0 = 1000;
        std::vector<uint16_t> qk(T * k9::kQkWidth); for (auto &x : qk) x = f2bf(nd(g) * 2.f);
        auto *dqk = dev(qk); uint16_t *dqi; HCK(hipMalloc(&dqi, T * 4 * DI * 2));
        CCK(k9::indexer_qk(reinterpret_cast<const bf16 *>(dqk), k9::kQkWidth, reinterpret_cast<const bf16 *>(dwq), eps, dcs, bnd, nullptr, pos0, T, reinterpret_cast<bf16 *>(dqi), nullptr));
        auto got = host(dqi, size_t(T) * 4 * DI);
        int64_t one = 0, more = 0;
        for (int64_t t = 0; t < T; ++t) for (int h = 0; h < 4; ++h) {
            float x[DI]; for (int d = 0; d < DI; ++d) x[d] = bf2f(qk[t * k9::kQkWidth + h * DI + d]);
            uint16_t want[DI]; ref_row(x, wq, &cs[(pos0 + t) * k9::kRotaryDim], eps, want);
            for (int d = 0; d < DI; ++d) { const int u = std::abs(int(got[(t * 4 + h) * DI + d]) - int(want[d])); one += u == 1; more += u > 1; }
        }
        char msg[160]; std::snprintf(msg, sizeof msg, "37 tokens x 4 heads: %ld outputs at 1 ulp, %ld beyond (rsqrtf)", (long)one, (long)more);
        check("indexer_qk", more == 0, msg);
    }
    {   // 2. compressed keys
        const int64_t T = 23, pos0 = 2, NB = 64;
        std::vector<uint16_t> kidx(T * DI), ring0(4 * DI, 0); for (auto &x : kidx) x = f2bf(nd(g)); for (int i = 0; i < 2 * DI; ++i) ring0[i] = f2bf(nd(g));
        auto *dk = dev(kidx);
        const std::vector<uint16_t> keys0(NB * DI, 0);
        auto fresh = [&](uint16_t **ring, uint16_t **keys) { *ring = dev(ring0); *keys = dev(keys0); };
        uint16_t *r1, *k1s; fresh(&r1, &k1s);
        CCK(k9::qsa_compress_store(reinterpret_cast<const bf16 *>(dk), DI, reinterpret_cast<const bf16 *>(dwk), eps, dcs, bnd, nullptr, nullptr, pos0, T, reinterpret_cast<bf16 *>(r1), reinterpret_cast<bf16 *>(k1s), nullptr));
        auto keys1 = host(k1s, keys0.size()), ring1 = host(r1, ring0.size());
        int64_t one = 0, more = 0, untouched_bad = 0;
        const int64_t gend = (pos0 + T) / 4;  // groups 0 .. gend-1 complete
        for (int64_t gg = 0; gg < NB; ++gg) {
            if (gg >= gend) { for (int d = 0; d < DI; ++d) untouched_bad += keys1[gg * DI + d] != 0; continue; }
            float acc[DI];
            for (int r = 0; r < 4; ++r) { const int64_t p = gg * 4 + r;
                for (int d = 0; d < DI; ++d) { const float f = p < pos0 ? bf2f(ring0[(p % 4) * DI + d]) : bf2f(kidx[(p - pos0) * DI + d]); acc[d] = r == 0 ? f : acc[d] + f; } }
            float x[DI]; for (int d = 0; d < DI; ++d) x[d] = rb(acc[d] * 0.25f);
            uint16_t want[DI]; ref_row(x, wk, &cs[(gg * 4) * k9::kRotaryDim], eps, want);
            for (int d = 0; d < DI; ++d) { const int u = std::abs(int(keys1[gg * DI + d]) - int(want[d])); one += u == 1; more += u > 1; }
        }
        // the ring rows the contract defines after the call: the incomplete last group's members (rows p % 4)
        auto ring_eq = [&](const std::vector<uint16_t> &r) {
            bool ok = true; for (int64_t p = gend * 4; p < pos0 + T; ++p) for (int d = 0; d < DI; ++d) ok &= r[(p % 4) * DI + d] == ring1[(p % 4) * DI + d];
            return ok; };
        bool ring_ok = true;  // positions 24 = group 6 incomplete: members 24 at ring row 0 (pos0 + T = 25)
        for (int64_t p = gend * 4; p < pos0 + T; ++p) for (int d = 0; d < DI; ++d) ring_ok &= ring1[(p % 4) * DI + d] == kidx[(p - pos0) * DI + d];
        char msg[200]; std::snprintf(msg, sizeof msg, "%ld groups: %ld outputs at 1 ulp, %ld beyond; groups not completed untouched %s; ring holds the tail %s", (long)gend, (long)one, (long)more, untouched_bad ? "NO" : "yes", ring_ok ? "yes" : "NO");
        check("qsa_compress_store", more == 0 && !untouched_bad && ring_ok, msg);
        // splits 5 + 7 + 11
        uint16_t *r2, *k2; fresh(&r2, &k2);
        int64_t at = 0; for (int64_t n : {5, 7, 11}) {
            CCK(k9::qsa_compress_store(reinterpret_cast<const bf16 *>(dk + at * DI), DI, reinterpret_cast<const bf16 *>(dwk), eps, dcs, bnd, nullptr, nullptr, pos0 + at, n, reinterpret_cast<bf16 *>(r2), reinterpret_cast<bf16 *>(k2), nullptr));
            at += n; }
        const bool split_ok = host(k2, keys0.size()) == keys1 && ring_eq(host(r2, ring0.size()));
        // step by step (device position)
        uint16_t *r3, *k3; fresh(&r3, &k3); int64_t *dpos; HCK(hipMalloc(&dpos, 8));
        for (int64_t t = 0; t < T; ++t) {
            const int64_t p = pos0 + t; HCK(hipMemcpy(dpos, &p, 8, hipMemcpyHostToDevice));
            CCK(k9::qsa_compress_store_step(reinterpret_cast<const bf16 *>(dk + t * DI), reinterpret_cast<const bf16 *>(dwk), eps, dcs, bnd, nullptr, dpos, reinterpret_cast<bf16 *>(r3), reinterpret_cast<bf16 *>(k3), nullptr));
        }
        const bool step_ok = host(k3, keys0.size()) == keys1 && ring_eq(host(r3, ring0.size()));
        uint16_t *r4, *k4; fresh(&r4, &k4); const int64_t p0 = pos0; HCK(hipMemcpy(dpos, &p0, 8, hipMemcpyHostToDevice));
        CCK(k9::qsa_compress_store_at(reinterpret_cast<const bf16 *>(dk), DI, reinterpret_cast<const bf16 *>(dwk), eps, dcs, bnd, nullptr, dpos, T, reinterpret_cast<bf16 *>(r4), reinterpret_cast<bf16 *>(k4), nullptr));
        const bool at_ok = host(k4, keys0.size()) == keys1 && ring_eq(host(r4, ring0.size()));
        std::snprintf(msg, sizeof msg, "keys + the defined ring rows bit-equal to the 23-token call: 5+7+11 %s, 23 x step %s, _at %s", split_ok ? "yes" : "NO", step_ok ? "yes" : "NO", at_ok ? "yes" : "NO");
        check("compress continuity", split_ok && step_ok && at_ok, msg);
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
