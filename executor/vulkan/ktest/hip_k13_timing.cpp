// VulkanAmd P2: same-session HIP reference timing of K13's moe_bf16_block (the hipified executor/cuda/src/kernels/
// k13_moe_bf16.cu, as executor/hip/build.sh builds k13_test) with the method of the Vulkan k13_test timing section:
// the MTP layer's shape (H 2,560, E 512, top-10, I 640, S 640), NVFP4 W4A16 experts (and, with --bf16, their BF16
// dequantization), k13_test's generated data (seed 9, the same order; the checksum is printed by both), 3 untimed
// calls, then `reps` calls each between its own pair of hipEvents, median / min / max; call i on rows starting at
// (i * 61 * m) % (4,096 - m). Router / shared GEMMs on hipBLASLt GemmPlans (max_m 4,096) as k13_test's plans.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels/k13_moe_bf16.h"

namespace kn = cere::kernels;
using bf16 = __hip_bfloat16;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
#define CCK(x) do { cere_status s_ = (x); if (s_ != CERE_OK) { std::printf("cere %s:%d status %d\n", __FILE__, __LINE__, int(s_)); std::exit(2); } } while (0)
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float e4m3f(uint8_t c) { const int s = c >> 7, e = (c >> 3) & 15, m = c & 7; const float v = e == 0 ? std::ldexp(m / 8.f, -6) : std::ldexp(1.f + m / 8.f, e - 7); return s ? -v : v; }
static const float kE2[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
static uint16_t deq1(const std::vector<uint8_t> &q, const std::vector<uint8_t> &sc, size_t idx, float glob) {
    const int code = (q[idx / 2] >> (4 * (idx % 2))) & 15; const float v = (code & 8) ? -kE2[code & 7] : kE2[code & 7];
    return f2bf((v * e4m3f(sc[idx / 16])) * glob);
}
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static uint64_t sum(const std::vector<T> &v, uint64_t s) { for (auto x : v) s = s * 1000003u + uint64_t(x); return s; }
static const bf16 *B(const uint16_t *p) { return reinterpret_cast<const bf16 *>(p); }

int main(int argc, char **argv) {
    const bool bf16_too = argc > 1 && std::string(argv[1]) == "--bf16";
    const uint32_t H = 2560, E = 512, K = 10, I = 640, S = 640, POOL = 4096, MAXM = 4096;
    std::mt19937 g(9); std::normal_distribution<float> nd;
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    auto router = rnd(size_t(E) * H, 0.1f), sg = rnd(size_t(S) * H, 1.f / 16), su = rnd(size_t(S) * H, 1.f / 16), sd = rnd(size_t(H) * S, 1.f / 11), seg = rnd(H, 0.06f);
    std::vector<uint8_t> guq(size_t(E) * 2 * I * H / 2), gus(size_t(E) * 2 * I * H / 16), dq(size_t(E) * H * I / 2), dsc(size_t(E) * H * I / 16);
    for (auto &b : guq) b = uint8_t(g());
    for (auto &b : dq) b = uint8_t(g());
    for (auto &b : gus) b = uint8_t(0x30 + g() % 0x18);
    for (auto &b : dsc) b = uint8_t(0x30 + g() % 0x18);
    std::vector<float> gug(E * 2), dg(E);
    for (auto &x : gug) x = 0.01f * (1 + 0.1f * float(g() % 8));
    for (auto &x : dg) x = 0.008f * (1 + 0.1f * float(g() % 8));
    auto xp = rnd(size_t(POOL) * H, 1.f);
    std::printf("data checksum %016llx\n", (unsigned long long)sum(xp, sum(dsc, sum(guq, sum(router, 1)))));
    kn::MoeDims dims{H, E, K, I, S};
    kn::MoeBf16Layer Lq{};
    Lq.router = B(dev(router)); Lq.shared_gate = B(dev(sg)); Lq.shared_up = B(dev(su)); Lq.shared_down = B(dev(sd)); Lq.shared_expert_gate = B(dev(seg));
    kn::MoeBf16Layer Lb = Lq;
    Lq.gate_up_q = dev(guq); Lq.gate_up_scale = dev(gus); Lq.gate_up_global = dev(gug); Lq.down_q = dev(dq); Lq.down_scale = dev(dsc); Lq.down_global = dev(dg);
    if (bf16_too) {
        std::vector<uint16_t> gu(size_t(E) * 2 * I * H), dn(size_t(E) * H * I);
        for (size_t r = 0; r < size_t(E) * 2 * I; ++r) { const size_t e = r / (2 * I); const float gl = gug[e * 2 + ((r % (2 * I)) >= I ? 1 : 0)]; for (uint32_t c = 0; c < H; ++c) gu[r * H + c] = deq1(guq, gus, r * H + c, gl); }
        for (size_t r = 0; r < size_t(E) * H; ++r) { const float gl = dg[r / H]; for (uint32_t c = 0; c < I; ++c) dn[r * I + c] = deq1(dq, dsc, r * I + c, gl); }
        Lb.gate_up = B(dev(gu)); Lb.down = B(dev(dn));
    }
    hipblasLtHandle_t lt; if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) return 2;
    kn::GemmPlan pr, pgu, pd;
    CCK(pr.init(lt, E, H, MAXM, kn::GemmOut::kBF16)); CCK(pgu.init(lt, S, H, MAXM, kn::GemmOut::kBF16)); CCK(pd.init(lt, H, S, MAXM, kn::GemmOut::kBF16));
    kn::MoePlans plans{&pr, &pgu, &pd};
    const size_t wsb = kn::moe_bf16_workspace_bytes(dims, MAXM); void *dws; HCK(hipMalloc(&dws, wsb + 256));
    uint16_t *dx = dev(xp); uint16_t *dout; HCK(hipMalloc(&dout, size_t(MAXM) * H * 2));
    struct Shape { uint32_t m, first; const char *what; };
    const Shape shapes[] = {{1, 0, "draft step"}, {2, 1, "extend last_only"}, {5, 4, "extend last_only"},
                            {5, 0, "extend staged"}, {16, 0, "16 rows"}, {4096, 4095, "prompt extend last_only"}};
    std::printf("timing (model shape, hipEvents, rot):\n");
    for (int fmt = 0; fmt <= (bf16_too ? 1 : 0); ++fmt) {
        const kn::MoeBf16Layer &L = fmt ? Lb : Lq;
        for (const auto &sh : shapes) {
            const uint32_t m = sh.m; const int reps = m >= 1024 ? 7 : 15;
            auto xa = [&](int i) { return dx + size_t(m < POOL ? (uint64_t(i) * 61 * m) % (POOL - m) : 0) * H; };
            std::vector<hipEvent_t> ev(2 * reps);
            for (auto &e : ev) HCK(hipEventCreate(&e));
            int c = 0;
            for (int i = 0; i < 3; ++i, ++c) CCK(kn::moe_bf16_block(dims, L, plans, B(xa(c)), m, sh.first, reinterpret_cast<bf16 *>(dout), dws, wsb, nullptr));
            for (int i = 0; i < reps; ++i, ++c) {
                HCK(hipEventRecord(ev[2 * i], nullptr));
                CCK(kn::moe_bf16_block(dims, L, plans, B(xa(c)), m, sh.first, reinterpret_cast<bf16 *>(dout), dws, wsb, nullptr));
                HCK(hipEventRecord(ev[2 * i + 1], nullptr));
            }
            HCK(hipDeviceSynchronize());
            std::vector<double> t;
            for (int i = 0; i < reps; ++i) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1])); t.push_back(ms); }
            for (auto &e : ev) HCK(hipEventDestroy(e));
            std::sort(t.begin(), t.end());
            std::printf("  hip    %s m %4u first %4u (%s): median %.3f ms (min %.3f, max %.3f, n %d)\n", fmt ? "bf16 " : "nvfp4", m, sh.first, sh.what,
                        t[t.size() / 2], t.front(), t.back(), reps);
            std::fflush(stdout);
        }
    }
    return 0;
}
