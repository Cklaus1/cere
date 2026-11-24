// AmdAmd PAR-6 step 4: K6 dense causal GQA (executor/cuda/src/kernels/k6_attention.cu, hipified; FP8 convert, cp.async
// and the 64 KB LDS fit patched in gen_hip.py) on gfx1151: calls of <= 16 rows on its split CUDA-core kernels, longer
// ones on the HIP P4 (executor/hip/src/p4_attention_hip.cpp, WMMA), against k6::attention's contract:
// out[t, h] = softmax_j(q[t, h] . K[j] * scale) V[j], j = 0 .. pos0 + t, KV head h / 12, FP64 reference.
//  1. BF16 K/V rows and the FP8 E4M3 cache (scale 1.0; reference from the decoded bytes) at decode (1 token at pos0
//     100 and 1,500), a short extend (7 at 50), and prefill-size calls (64 at 0, 300 at 0, 2,048 at 0): rel-L2 vs FP64
//     (limit 8e-3: BF16 q, k, v, BF16 out);
//  2. repeat bit-identical;
//  3. the class pin: rows 128..299 of a 300-row call (pos0 0) == a 172-row call at pos0 128 with cls 300, bit for bit;
//  4. timing (reported): prefill 2,048 / 512 rows, decode 1 row at 2,047 keys, a 4-row extend at 2,040.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k6_attention.h"

namespace k6 = cere::k6;
using bf16 = __hip_bfloat16;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float e4m3(uint8_t c) {
    const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    const float v = e == 0 ? std::ldexp(m / 8.f, -6) : std::ldexp(1.f + m / 8.f, e - 7);
    return s ? -v : v;
}
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }

constexpr int QH = k6::kQHeads, KVH = k6::kKvHeads, D = k6::kHeadDim, QW = QH * D, KW = KVH * D;

int main() {
    std::mt19937 g(29); std::normal_distribution<float> nd;
    const int64_t NK = k6::kMaxContext;  // keys available
    std::vector<uint16_t> kb(NK * KW), vb(NK * KW);
    for (auto &x : kb) x = f2bf(nd(g)); for (auto &x : vb) x = f2bf(nd(g));
    std::vector<uint8_t> kc(NK * KW), vc(NK * KW);
    for (auto &b : kc) { b = uint8_t(g() % 0x78); if (g() & 1) b |= 0x80; }  // finite E4M3 up to 240
    for (auto &b : vc) { b = uint8_t(g() % 0x78); if (g() & 1) b |= 0x80; }
    auto *dk = dev(kb), *dv = dev(vb); auto *dkc = dev(kc), *dvc = dev(vc);
    const int64_t TMAX = 2048;
    std::vector<uint16_t> q(TMAX * QW); for (auto &x : q) x = f2bf(nd(g) * 0.25f);
    auto *dq = dev(q); uint16_t *dout; HCK(hipMalloc(&dout, TMAX * QW * 2));
    const size_t wsb = std::max(k6::dense_attention_workspace_bytes(TMAX), k6::dense_attention_workspace_bytes(16)); void *dws; HCK(hipMalloc(&dws, wsb + 256));
    const float scale = k6::kSoftmaxScale;
    auto run = [&](bool fp8, int64_t T, int64_t pos0, int64_t qrow0, int64_t cls = 0) {
        const bf16 *qq = reinterpret_cast<const bf16 *>(dq + qrow0 * QW);
        hipError_t e = fp8 ? k6::attention(qq, T, pos0, nullptr, 0, nullptr, 0, dkc, dvc, nullptr, scale, dws, wsb, reinterpret_cast<bf16 *>(dout), nullptr, false, cls)
                           : k6::attention(qq, T, pos0, reinterpret_cast<const bf16 *>(dk), KW, reinterpret_cast<const bf16 *>(dv), KW, nullptr, nullptr, nullptr, scale, dws, wsb, reinterpret_cast<bf16 *>(dout), nullptr, false, cls);
        if (e != hipSuccess) { std::printf("attention failed: %s (T %ld pos0 %ld)\n", hipGetErrorString(e), (long)T, (long)pos0); std::exit(2); }
        return host(dout, size_t(T) * QW);
    };
    auto kval = [&](bool fp8, int64_t j, int c) { return fp8 ? double(e4m3(kc[j * KW + c])) : double(bf2f(kb[j * KW + c])); };
    auto vval = [&](bool fp8, int64_t j, int c) { return fp8 ? double(e4m3(vc[j * KW + c])) : double(bf2f(vb[j * KW + c])); };
    struct C { int64_t T, pos0; } cases[] = {{1, 100}, {1, 1500}, {7, 50}, {64, 0}, {300, 0}, {2048, 0}};
    for (bool fp8 : {false, true}) for (auto cs : cases) {
        auto out = run(fp8, cs.T, cs.pos0, 0);
        double num = 0, den = 0;
        const int64_t step = cs.T > 256 ? 37 : 1;  // sample rows of the big calls (FP64 reference cost)
        for (int64_t t = 0; t < cs.T; t += step) for (int h = 0; h < QH; ++h) {
            const int kvh = h / (QH / KVH); const int64_t nk = cs.pos0 + t + 1;
            std::vector<double> s(nk); double mx = -1e300;
            for (int64_t j = 0; j < nk; ++j) { double d = 0; for (int c = 0; c < D; ++c) d += double(bf2f(q[t * QW + h * D + c])) * kval(fp8, j, kvh * D + c); s[j] = d * scale; mx = std::max(mx, s[j]); }
            double l = 0; for (auto &x : s) { x = std::exp(x - mx); l += x; }
            for (int c = 0; c < D; ++c) {
                double o = 0; for (int64_t j = 0; j < nk; ++j) o += s[j] * vval(fp8, j, kvh * D + c);
                o /= l; const double e = bf2f(out[t * QW + h * D + c]) - o; num += e * e; den += o * o;
            }
        }
        const double err = std::sqrt(num / den);
        auto again = run(fp8, cs.T, cs.pos0, 0);
        const bool ok = err <= 8e-3 && again == out;
        std::printf("%-6s T %4ld pos0 %4ld   %s  rel-L2 %.2e vs FP64 (limit 8e-3), repeat %s\n", fp8 ? "fp8" : "bf16", (long)cs.T, (long)cs.pos0, ok ? "PASS" : "FAIL", err, again == out ? "bit-equal" : "DIFFERS");
        if (!ok) ++g_fail;
    }
    for (bool fp8 : {false, true}) {  // class pin
        auto full = run(fp8, 300, 0, 0);
        auto part = run(fp8, 172, 128, 128, 300);
        int64_t diff = 0; for (size_t i = 0; i < part.size(); ++i) diff += part[i] != full[128 * QW + i];
        std::printf("%-6s class pin   %s  rows 128..299 at pos0 128 (cls 300) vs the 300-row call: %ld of %zu differ\n", fp8 ? "fp8" : "bf16", diff ? "FAIL" : "PASS", (long)diff, part.size());
        if (diff) ++g_fail;
    }
    {   // the verify chain kernel vs one-token calls (the contract: row t == dense_attention at prefix pos0 + t)
        const int64_t R = 5, pos0 = 1700; const size_t g_sz = size_t(R) * QW;
        std::vector<uint16_t> gate(g_sz); for (auto &x : gate) x = f2bf(nd(g)); auto *dgate = dev(gate);
        uint16_t *dch; HCK(hipMalloc(&dch, g_sz * 2));
        HCK(k6::dense_attention_rows(reinterpret_cast<const bf16 *>(dq), reinterpret_cast<const bf16 *>(dgate), R, pos0, dkc, dvc, nullptr, dws, wsb, reinterpret_cast<bf16 *>(dch), QW, nullptr));
        auto chain = host(dch, g_sz); int64_t diff = 0;
        for (int64_t t = 0; t < R; ++t) {
            HCK(k6::dense_attention(reinterpret_cast<const bf16 *>(dq + t * QW), reinterpret_cast<const bf16 *>(dk), reinterpret_cast<const bf16 *>(dv), KW, reinterpret_cast<const bf16 *>(dgate + t * QW), 1, pos0 + t, true, dkc, dvc, nullptr, dws, wsb, reinterpret_cast<bf16 *>(dout), QW, nullptr, false));
            auto one = host(dout, QW); for (int c = 0; c < QW; ++c) diff += one[c] != chain[t * QW + c];
        }
        std::printf("%-6s chain rows   %s  dense_attention_rows (5 rows at 1,700) vs one-token dense_attention: %ld of %zu differ\n", "fp8", diff ? "FAIL" : "PASS", (long)diff, g_sz);
        if (diff) ++g_fail;
    }
    for (bool fp8 : {false, true}) {  // timing (mean of 5 after a warm-up)
        struct TC { int64_t T, pos0; } tc[] = {{2048, 0}, {512, 0}, {1, 2047}, {4, 2040}};
        for (auto c : tc) {
            auto launch = [&]() {
                const bf16 *qq = reinterpret_cast<const bf16 *>(dq);
                return fp8 ? k6::attention(qq, c.T, c.pos0, nullptr, 0, nullptr, 0, dkc, dvc, nullptr, scale, dws, wsb, reinterpret_cast<bf16 *>(dout), nullptr, false)
                           : k6::attention(qq, c.T, c.pos0, reinterpret_cast<const bf16 *>(dk), KW, reinterpret_cast<const bf16 *>(dv), KW, nullptr, nullptr, nullptr, scale, dws, wsb, reinterpret_cast<bf16 *>(dout), nullptr, false);
            };
            HCK(launch()); hipEvent_t e0, e1; HCK(hipEventCreate(&e0)); HCK(hipEventCreate(&e1)); HCK(hipEventRecord(e0, nullptr));
            for (int i = 0; i < 5; ++i) HCK(launch());
            HCK(hipEventRecord(e1, nullptr)); HCK(hipEventSynchronize(e1)); float ms; HCK(hipEventElapsedTime(&ms, e0, e1)); ms /= 5;
            double pairs = 0; for (int64_t t = 0; t < c.T; ++t) pairs += double(c.pos0 + t + 1);
            const double tf = 4.0 * QH * D * pairs / (ms * 1e9);
            std::printf("%-6s timing       T %4ld pos0 %4ld: %.3f ms (%s, %.1f TFLOP/s causal-useful)\n", fp8 ? "fp8" : "bf16", (long)c.T, (long)c.pos0, ms,
                        c.T > 16 ? "P4 WMMA" : "split CUDA-core", tf);
        }
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
