// VulkanAmd P2: same-session HIP reference timing of K9 (the hipified executor/cuda/src/kernels/k9_qsa.cu + HIP's
// executor/hip/src/p4_attention_hip.cpp, as executor/hip/build.sh builds k9_test) with the method and data of the
// Vulkan k9_test --time section: the same generator (seed 53, the same order; the data checksum is printed by both),
// 3 untimed calls, then 15 calls each between its own pair of hipEvents (GPU time, stream-serialized), median / min /
// max. Shapes: one-token decode (qsa_attention alone, and the K9 step: indexer_qk + qsa_compress_store_step +
// qsa_select_step + qsa_attention) on BF16 and FP8 KV at a 2K (position 2,047) and a 16K (position 16,383) context; an
// 8-row verify chain (qsa_select + qsa_attention, FP8); a 4,096-row prefill attention (FP8, BF16) and its qsa_select;
// P4-A: the S3 3,000-token prompt's call (k9_test's shape: 3,000 rows at positions 0..2,999, BF16 rows and FP8 cache).
// qsa_attention gets the model's scratch (max of the select and attention scratch: on HIP the FP8 prefill decodes
// its BF16 K / V copy there, amd-amd step 16). HIP is the reference here, not part of the Vulkan executor.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k9_qsa.h"

namespace k9 = cere::k9;
using bf16 = __hip_bfloat16;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
#define CCK(x) do { cere_status s_ = (x); if (s_ != CERE_OK) { std::printf("cere %s:%d status %d\n", __FILE__, __LINE__, int(s_)); std::exit(2); } } while (0)
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
static const bf16 *B(const uint16_t *p) { return reinterpret_cast<const bf16 *>(p); }

constexpr int QH = k9::kQHeads, D = k9::kHeadDim, QW = QH * D, KW = k9::kKvHeads * D, RW = k9::kRowWidth;

static void timing_row(std::mt19937 &g, int64_t p, int32_t *out) {  // k9_test.cpp's
    const int64_t vis = (p + 1) / 4;
    std::vector<int64_t> blocks;
    if (vis <= 512) { for (int64_t b = 0; b < vis; ++b) blocks.push_back(b); }
    else {
        std::vector<int64_t> all(vis); for (int64_t b = 0; b < vis; ++b) all[b] = b;
        for (int64_t i = 0; i < 512; ++i) { const int64_t j = i + int64_t(g() % uint32_t(vis - i)); std::swap(all[i], all[j]); }
        blocks.assign(all.begin(), all.begin() + 512); std::sort(blocks.begin(), blocks.end());
    }
    int n = 0;
    for (int64_t b : blocks) for (int i = 0; i < 4; ++i) out[n++] = int32_t(4 * b + i);
    for (int64_t q = vis * 4; q <= p; ++q) out[n++] = int32_t(q);
    while (n < RW) out[n++] = -1;
}

int main() {
    std::mt19937 g(53); std::normal_distribution<float> nd;
    const int64_t CTX = 16384, NB = CTX / 4, PR = 4096;
    std::vector<uint8_t> kc(CTX * KW), vc(CTX * KW);
    for (auto &b : kc) { b = uint8_t(g() % 0x78); if (g() & 1) b |= 0x80; }
    for (auto &b : vc) { b = uint8_t(g() % 0x78); if (g() & 1) b |= 0x80; }
    std::vector<uint16_t> kb(CTX * KW), vb(CTX * KW);
    for (auto &x : kb) { x = f2bf(nd(g)); }
    for (auto &x : vb) { x = f2bf(nd(g)); }
    std::vector<uint16_t> q(PR * QW);
    for (auto &x : q) { x = f2bf(nd(g) * 0.25f); }
    std::vector<int32_t> idx(PR * RW), idx2k(RW), idx16(RW);
    for (int64_t t = 0; t < PR; ++t) timing_row(g, CTX - PR + t, &idx[t * RW]);
    timing_row(g, 2047, idx2k.data()); timing_row(g, CTX - 1, idx16.data());
    std::vector<uint16_t> qk(16 * k9::kQkWidth), keys(NB * 128), w(128), qi(PR * 4 * 128);
    for (auto &x : qk) { x = f2bf(nd(g)); }
    for (auto &x : keys) { x = f2bf(nd(g)); }
    for (auto &x : w) { x = f2bf(nd(g) * 0.1f); }
    for (auto &x : qi) { x = f2bf(nd(g)); }
    std::vector<float> cs(size_t(CTX) * 64);
    for (int64_t p = 0; p < CTX; ++p) for (int i = 0; i < 32; ++i) { const double a = double(p) * std::pow(1e7, -2.0 * i / 64); cs[p * 64 + i] = float(std::cos(a)); cs[p * 64 + 32 + i] = float(std::sin(a)); }
    uint64_t sum = 1; for (auto x : idx) sum = sum * 1000003u + uint32_t(x); for (auto x : q) sum = sum * 1000003u + x; for (auto x : kc) sum = sum * 1000003u + x;
    std::printf("timing data checksum %016llx\n", (unsigned long long)sum);
    auto *dkc = dev(kc), *dvc = dev(vc); auto *dkb = dev(kb), *dvb = dev(vb), *dq = dev(q), *dqk = dev(qk), *dkeys = dev(keys), *dw = dev(w), *dqi = dev(qi);
    auto *didx = dev(idx), *di2 = dev(idx2k), *di16 = dev(idx16); auto *dcs = dev(cs);
    uint16_t *dout, *dring, *dqidx; int32_t *dsel;
    HCK(hipMalloc(&dout, size_t(PR) * QW * 2)); HCK(hipMalloc(&dring, 4 * 128 * 2)); HCK(hipMalloc(&dqidx, 16 * 4 * 128 * 2)); HCK(hipMalloc(&dsel, size_t(PR) * RW * 4));
    HCK(hipMemset(dring, 0, 4 * 128 * 2));
    const size_t sb = std::max({k9::select_scratch_bytes(PR, CTX), k9::select_scratch_bytes(16, CTX), k9::attention_scratch_bytes(16)});
    void *ds; HCK(hipMalloc(&ds, sb + 256));
    std::vector<int64_t> p2k{2047}, p16{CTX - 1}, pv8(8); for (int i = 0; i < 8; ++i) pv8[i] = CTX - 8 + i;
    auto *dp2k = dev(p2k), *dp16 = dev(p16), *dpv8 = dev(pv8);
    hipEvent_t ev[32]; for (auto &e : ev) HCK(hipEventCreate(&e));
    auto timeit = [&](auto &&rec) {
        const int reps = 15;
        for (int i = 0; i < 3; ++i) rec();
        std::vector<double> t;
        for (int i = 0; i < reps; ++i) { HCK(hipEventRecord(ev[2 * i], nullptr)); rec(); HCK(hipEventRecord(ev[2 * i + 1], nullptr)); }
        HCK(hipDeviceSynchronize());
        for (int i = 0; i < reps; ++i) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1])); t.push_back(ms); }
        std::sort(t.begin(), t.end());
        return std::vector<double>{t[t.size() / 2], t.front(), t.back()};
    };
    auto attn = [&](bool fp8, const int32_t *ii, int64_t rows) {
        CCK(fp8 ? k9::qsa_attention(B(dq), QW, ii, rows, nullptr, 0, nullptr, 0, dkc, dvc, nullptr, k9::kSoftmaxScale, ds, sb, reinterpret_cast<bf16 *>(dout), QW, nullptr, true)
                : k9::qsa_attention(B(dq), QW, ii, rows, B(dkb), KW, B(dvb), KW, nullptr, nullptr, nullptr, k9::kSoftmaxScale, ds, sb, reinterpret_cast<bf16 *>(dout), QW, nullptr, true));
    };
    auto step = [&](bool fp8, const int64_t *dpos, int64_t p) {
        const k9::Bounds b{0, CTX, p + 1, nullptr};
        CCK(k9::indexer_qk(B(dqk), k9::kQkWidth, B(dw), 1e-6f, dcs, b, nullptr, p, 1, reinterpret_cast<bf16 *>(dqidx), nullptr));
        CCK(k9::qsa_compress_store_step(B(dqk) + 512, B(dw), 1e-6f, dcs, b, nullptr, dpos, reinterpret_cast<bf16 *>(dring), reinterpret_cast<bf16 *>(dkeys), nullptr));
        CCK(k9::qsa_select_step(B(dqidx), B(dkeys), nullptr, b, dpos, CTX, ds, sb, dsel, nullptr));
        attn(fp8, dsel, 1);
    };
    auto pr = [&](const char *name, std::vector<double> s) { std::printf("HIPTIME %-44s median %8.4f ms  (min %.4f, max %.4f)\n", name, s[0], s[1], s[2]); };
    for (bool fp8 : {false, true}) {
        const char *f = fp8 ? "FP8" : "BF16";
        char n[96];
        std::snprintf(n, sizeof n, "attention 1 row, 2K ctx, %s", f); pr(n, timeit([&] { attn(fp8, di2, 1); }));
        std::snprintf(n, sizeof n, "attention 1 row, 16K ctx, %s", f); pr(n, timeit([&] { attn(fp8, di16, 1); }));
        std::snprintf(n, sizeof n, "K9 decode step, 2K ctx, %s", f); pr(n, timeit([&] { step(fp8, dp2k, 2047); }));
        std::snprintf(n, sizeof n, "K9 decode step, 16K ctx, %s", f); pr(n, timeit([&] { step(fp8, dp16, CTX - 1); }));
    }
    pr("select 1 row (select_step), 16K ctx", timeit([&] { const k9::Bounds b{0, CTX, CTX, nullptr}; CCK(k9::qsa_select_step(B(dqi), B(dkeys), nullptr, b, dp16, CTX, ds, sb, dsel, nullptr)); }));
    pr("verify 8 rows: select + attention, 16K, FP8", timeit([&] { CCK(k9::qsa_select(B(dqi), B(dkeys), nullptr, dpv8, 8, CTX, ds, sb, dsel, nullptr)); attn(true, dsel, 8); }));
    pr("verify 8 rows: attention only, 16K, FP8", timeit([&] { attn(true, didx + (PR - 8) * RW, 8); }));
    pr("prefill 4,096 rows attention, 16K, FP8", timeit([&] { attn(true, didx, PR); }));
    pr("prefill 4,096 rows attention, 16K, BF16", timeit([&] { attn(false, didx, PR); }));
    pr("prefill 512 rows attention, 16K, FP8", timeit([&] { attn(true, didx, 512); }));
    pr("prefill 4,096 rows select, 16K", timeit([&] { CCK(k9::qsa_select(B(dqi), B(dkeys), nullptr, nullptr, PR, CTX, ds, sb, dsel, nullptr)); }));
    {   // P4-A: the S3 3,000-token prompt's sparse attention call (k9_test.cpp's shape: the same generator, after the above)
        const int64_t S3 = 3000;
        std::vector<int32_t> is3(S3 * RW);
        for (int64_t t = 0; t < S3; ++t) timing_row(g, t, &is3[t * RW]);
        auto *dis3 = dev(is3);
        pr("S3 3,000-token prompt, BF16 rows", timeit([&] { attn(false, dis3, S3); }));
        pr("S3 3,000-token prompt as FP8 cache", timeit([&] { attn(true, dis3, S3); }));
    }
    return 0;
}
