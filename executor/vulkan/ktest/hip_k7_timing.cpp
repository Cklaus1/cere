// VulkanAmd P2: same-session HIP reference timing of K7's moe_block (the hipified executor/cuda/src/kernels/k7_moe.cu
// with executor/hip/src/k7_*.inc, as executor/hip/build.sh builds k7_test) with the method of the Vulkan k7_test
// timing section: the model's shape (H 2,560, E 512, top-10, I 640, S 640), the same generated data (k7_test's Gen,
// seed 7, the same order; a checksum of the data is printed by both so equality is visible), 3 untimed calls, then
// `reps` calls each between its own pair of hipEvents (GPU time), median / min / max; hot = every call on rows
// 0 .. m-1, rot = call i on rows starting at (i * 61 * m) % (4096 - m). Router / shared GEMMs on hipBLASLt GemmPlans
// (max_m 4,096), as k7_test's plans. HIP is the reference here, not part of the Vulkan executor.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k7_moe.h"

namespace kn = cere::kernels;
using bf16 = __hip_bfloat16;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
#define CCK(x) do { cere_status s_ = (x); if (s_ != CERE_OK) { std::printf("cere %s:%d status %d\n", __FILE__, __LINE__, int(s_)); std::exit(2); } } while (0)
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
struct Gen {  // k7_test.cpp's
    std::mt19937 g; std::normal_distribution<float> nd;
    explicit Gen(uint32_t seed) : g(seed) {}
    std::vector<uint16_t> rnd(size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; }
    std::vector<uint8_t> codes(size_t n) { std::vector<uint8_t> v(n); for (auto &b : v) b = uint8_t(g()); return v; }
    std::vector<uint8_t> scales(size_t n) { std::vector<uint8_t> v(n); for (auto &b : v) b = uint8_t(0x30 + g() % 0x18); return v; }
};
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static uint64_t sum(const std::vector<T> &v, uint64_t s) { for (auto x : v) s = s * 1000003u + uint64_t(x); return s; }
static const bf16 *B(const uint16_t *p) { return reinterpret_cast<const bf16 *>(p); }

int main() {
    const uint32_t H = 2560, E = 512, K = 10, I = 640, S = 640, POOL = 4096, MAXM = 4096;
    Gen G(7);
    auto router = G.rnd(size_t(E) * H, 0.1f), sg = G.rnd(size_t(S) * H, 1.f / 16), su = G.rnd(size_t(S) * H, 1.f / 16),
         sd = G.rnd(size_t(H) * S, 1.f / 11), seg = G.rnd(H, 0.06f);
    auto gw = G.codes(size_t(E) * I * H / 2), uw = G.codes(size_t(E) * I * H / 2), dw = G.codes(size_t(E) * H * I / 2);
    auto gs = G.scales(size_t(E) * I * H / 16), us = G.scales(size_t(E) * I * H / 16), ds = G.scales(size_t(E) * H * I / 16);
    const float a1 = 2688.f / 4.f, a2 = 2688.f / 2.f;
    std::vector<float> g1(E), g2(E);
    for (uint32_t e = 0; e < E; ++e) { g1[e] = (1.f / a1) * 0.004f * (1 + 0.1f * (e % 16)); g2[e] = (1.f / a2) * 0.004f * (1 + 0.05f * (e % 16)); }
    auto xp = G.rnd(size_t(POOL) * H, 1.f);
    uint64_t cs = sum(router, 1); cs = sum(gw, cs); cs = sum(ds, cs); cs = sum(xp, cs);
    std::printf("data checksum %016llx\n", (unsigned long long)cs);
    kn::MoeDims dims{H, E, K, I, S};
    kn::MoeLayer L{};
    L.router = B(dev(router)); L.gate_weight = dev(gw); L.gate_scale = dev(gs); L.up_weight = dev(uw); L.up_scale = dev(us);
    L.down_weight = dev(dw); L.down_scale = dev(ds); L.g1_alphas = dev(g1); L.g2_alphas = dev(g2); L.a1_global_scale = a1; L.a2_global_scale = a2;
    L.shared_gate = B(dev(sg)); L.shared_up = B(dev(su)); L.shared_down = B(dev(sd)); L.shared_expert_gate = B(dev(seg));
    hipblasLtHandle_t lt; if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) return 2;
    kn::GemmPlan pr, pgu, pd;
    CCK(pr.init(lt, E, H, MAXM, kn::GemmOut::kBF16)); CCK(pgu.init(lt, S, H, MAXM, kn::GemmOut::kBF16)); CCK(pd.init(lt, H, S, MAXM, kn::GemmOut::kBF16));
    kn::MoePlans plans{&pr, &pgu, &pd};
    const size_t wsb = kn::moe_workspace_bytes(dims, MAXM); void *dws; HCK(hipMalloc(&dws, wsb + 256));
    uint16_t *dx = dev(xp); uint16_t *dout; HCK(hipMalloc(&dout, size_t(MAXM) * H * 2));
    std::printf("timing (model shape, hipEvents):\n");
    for (uint32_t m : {1u, 5u, 8u, 16u, 1205u, 4096u}) {
        for (int rot = 0; rot <= (m < POOL ? 1 : 0); ++rot) {
            const int reps = m >= 1024 ? 7 : 15;
            auto xa = [&](int i) { return dx + size_t(rot ? (uint64_t(i) * 61 * m) % (POOL - m) : 0) * H; };
            std::vector<hipEvent_t> ev(2 * reps);
            for (auto &e : ev) HCK(hipEventCreate(&e));
            int c = 0;
            for (int i = 0; i < 3; ++i, ++c) CCK(kn::moe_block(dims, L, plans, B(xa(c)), m, reinterpret_cast<bf16 *>(dout), dws, wsb, nullptr));
            for (int i = 0; i < reps; ++i, ++c) {
                HCK(hipEventRecord(ev[2 * i], nullptr));
                CCK(kn::moe_block(dims, L, plans, B(xa(c)), m, reinterpret_cast<bf16 *>(dout), dws, wsb, nullptr));
                HCK(hipEventRecord(ev[2 * i + 1], nullptr));
            }
            HCK(hipDeviceSynchronize());
            std::vector<double> t;
            for (int i = 0; i < reps; ++i) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1])); t.push_back(ms); }
            for (auto &e : ev) HCK(hipEventDestroy(e));
            std::sort(t.begin(), t.end());
            std::printf("  hip    moe_block m %4u %s: median %.3f ms (min %.3f, max %.3f, n %d)\n", m, rot ? "rot" : "hot", t[t.size() / 2], t.front(), t.back(), reps);
            std::fflush(stdout);
        }
    }
    return 0;
}
