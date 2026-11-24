// VulkanAmd P2 K6 / P4-A (not part of the executor; a same-method HIP reference, built against the amd-amd checkout's
// generated HIP K6 sources): times cere::k6 calls on HIP with the Vulkan k6_test method: 3 untimed warm-up calls, then
// 15 calls each between its own pair of hipEvents (calls serialized by the stream); median / min / max. Same data
// generation as k6_test (seed 29). Shapes: attention BF16 / FP8 at T 2,048 / 512 / 1,024 at 0, 1,000 at 1,048 (P4-A: a
// second prefill chunk), 1 at 2,047, 4 at 2,040;
// dense_attention_rows (the verify chain, FP8, gated) 4 at 2,040, 8 at 2,040, 16 at 2,032; the gated one-token step.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k6_attention.h"

namespace k6 = cere::k6;
using bf16 = __hip_bfloat16;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
constexpr int QH = k6::kQHeads, KVH = k6::kKvHeads, D = k6::kHeadDim, QW = QH * D, KW = KVH * D;

template <class F> static void time_it(const char *label, F &&f) {
    const int reps = 15;
    for (int i = 0; i < 3; ++i) HCK(f());
    std::vector<hipEvent_t> ev(2 * reps);
    for (auto &e : ev) HCK(hipEventCreate(&e));
    for (int i = 0; i < reps; ++i) { HCK(hipEventRecord(ev[2 * i], nullptr)); HCK(f()); HCK(hipEventRecord(ev[2 * i + 1], nullptr)); }
    HCK(hipDeviceSynchronize());
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1])); t.push_back(ms); }
    std::sort(t.begin(), t.end());
    std::printf("HIP %-44s %.3f ms median of 15 (min %.3f, max %.3f)\n", label, t[7], t.front(), t.back());
    for (auto &e : ev) HCK(hipEventDestroy(e));
}

int main() {
    std::mt19937 g(29); std::normal_distribution<float> nd;
    const int64_t NK = k6::kMaxContext;
    std::vector<uint16_t> kb(NK * KW), vb(NK * KW);
    for (auto &x : kb) x = f2bf(nd(g)); for (auto &x : vb) x = f2bf(nd(g));
    std::vector<uint8_t> kc(NK * KW), vc(NK * KW);
    for (auto &b : kc) { b = uint8_t(g() % 0x78); if (g() & 1) b |= 0x80; }
    for (auto &b : vc) { b = uint8_t(g() % 0x78); if (g() & 1) b |= 0x80; }
    auto *dk = dev(kb), *dv = dev(vb); auto *dkc = dev(kc), *dvc = dev(vc);
    const int64_t TMAX = 2048;
    std::vector<uint16_t> q(TMAX * QW); for (auto &x : q) x = f2bf(nd(g) * 0.25f);
    std::vector<uint16_t> gate(16 * QW); for (auto &x : gate) x = f2bf(nd(g));
    auto *dq = dev(q); auto *dgate = dev(gate); uint16_t *dout; HCK(hipMalloc(&dout, TMAX * QW * 2));
    const size_t wsb = std::max(k6::dense_attention_workspace_bytes(TMAX), k6::dense_attention_workspace_bytes(16)); void *dws; HCK(hipMalloc(&dws, wsb + 256));
    const auto *qq = reinterpret_cast<const bf16 *>(dq);
    auto *oo = reinterpret_cast<bf16 *>(dout);
    for (bool fp8 : {false, true}) {
        struct TC { int64_t T, pos0; } tc[] = {{2048, 0}, {512, 0}, {1024, 0}, {1000, 1048}, {1, 2047}, {4, 2040}};
        for (auto c : tc) {
            char label[96]; std::snprintf(label, sizeof label, "%s attention T %ld pos0 %ld", fp8 ? "fp8 " : "bf16", (long)c.T, (long)c.pos0);
            time_it(label, [&] {
                return fp8 ? k6::attention(qq, c.T, c.pos0, nullptr, 0, nullptr, 0, dkc, dvc, nullptr, k6::kSoftmaxScale, dws, wsb, oo, nullptr, false)
                           : k6::attention(qq, c.T, c.pos0, reinterpret_cast<const bf16 *>(dk), KW, reinterpret_cast<const bf16 *>(dv), KW, nullptr, nullptr, nullptr, k6::kSoftmaxScale, dws, wsb, oo, nullptr, false);
            });
        }
    }
    struct CC { int64_t R, pos0; } cc[] = {{4, 2040}, {8, 2040}, {16, 2032}};
    for (auto c : cc) {
        char label[96]; std::snprintf(label, sizeof label, "fp8  chain (dense_attention_rows) R %ld pos0 %ld", (long)c.R, (long)c.pos0);
        time_it(label, [&] { return k6::dense_attention_rows(qq, reinterpret_cast<const bf16 *>(dgate), c.R, c.pos0, dkc, dvc, nullptr, dws, wsb, oo, QW, nullptr); });
    }
    time_it("fp8  step (dense_attention, gated) T 1 pos0 2047", [&] {
        return k6::dense_attention(qq, reinterpret_cast<const bf16 *>(dk), reinterpret_cast<const bf16 *>(dv), KW, reinterpret_cast<const bf16 *>(dgate), 1, 2047, true, dkc, dvc, nullptr, dws, wsb, oo, QW, nullptr, false);
    });
    return 0;
}
