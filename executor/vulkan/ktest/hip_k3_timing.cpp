// VulkanAmd P2: same-session HIP reference timing of the hipified K3 (gen_hip.py's kernels/k3_hc.hip: kUnfused mix with
// hipBLASLt / k2_wmma GEMMs, two-launch combine + norm) at the shapes k3_test's "time" section times on Vulkan, with the same method:
// 20 warm-up calls, "call" = 15 calls each between its own pair of hipEvents (median, min, max), "stream" = 5 samples of 20
// back-to-back calls (5 at m >= 1,024) / burst (median); "hot" reads one weight set, "rot8" call i reads weight set i % 8. Same random
// data distributions and shapes (hc 4, hidden 2,560, lowrank 320). HIP is the reference here, not part of the Vulkan executor.
// Not a contract test. Built by build.d/k3_hc.sh only when hipcc and a hipified tree exist.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "kernels/k3_hc.h"

namespace kn = cere::kernels;
using bf16 = __hip_bfloat16;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
#define CCK(x) do { cere_status s_ = (x); if (s_ != CERE_OK) { std::printf("cere %s:%d status %d\n", __FILE__, __LINE__, int(s_)); std::exit(2); } } while (0)
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static bf16 *dev(size_t bytes) { void *p; HCK(hipMalloc(&p, bytes + 256)); HCK(hipMemset(p, 0, bytes + 256)); return static_cast<bf16 *>(p); }
static bf16 *up(const std::vector<uint16_t> &h, int sets = 1) { bf16 *p = dev(h.size() * 2 * sets); for (int s = 0; s < sets; ++s) HCK(hipMemcpy(reinterpret_cast<char *>(p) + size_t(s) * h.size() * 2, h.data(), h.size() * 2, hipMemcpyHostToDevice)); return p; }

constexpr uint32_t HC = 4, H = 2560, LR = 320, W = HC * H;

static void time_it(const char *sets_name, uint32_t m, const char *name, const std::function<void(int)> &call, int burst) {
    int c = 0; const int warm = 20, reps = 15, samples = 5;
    for (int i = 0; i < warm; ++i) call(c++);
    std::vector<hipEvent_t> ev(2 * std::max(reps, samples));
    for (auto &e : ev) HCK(hipEventCreate(&e));
    for (int i = 0; i < reps; ++i) { HCK(hipEventRecord(ev[2 * i], nullptr)); call(c++); HCK(hipEventRecord(ev[2 * i + 1], nullptr)); }
    HCK(hipDeviceSynchronize());
    std::vector<double> t, u;
    for (int i = 0; i < reps; ++i) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1])); t.push_back(ms); }
    for (int s = 0; s < samples; ++s) {
        HCK(hipEventRecord(ev[2 * s], nullptr));
        for (int i = 0; i < burst; ++i) call(c++);
        HCK(hipEventRecord(ev[2 * s + 1], nullptr));
    }
    HCK(hipDeviceSynchronize());
    for (int s = 0; s < samples; ++s) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * s], ev[2 * s + 1])); u.push_back(ms / burst); }
    for (auto &e : ev) HCK(hipEventDestroy(e));
    std::sort(t.begin(), t.end()); std::sort(u.begin(), u.end());
    std::printf("  %-4s m %4u %-30s call %10.2f us (min %.2f, max %.2f)  stream %10.2f us/call\n", sets_name, m, name, t[t.size() / 2] * 1e3, t.front() * 1e3, t.back() * 1e3, u[u.size() / 2] * 1e3);
}

int main() {
    std::printf("hip k3 timing (hipEvents; call = one call between its own events, median of 15 after 20 warm-up; stream = back-to-back burst, median of 5)\n");
    std::mt19937 g(5); std::normal_distribution<float> nd;
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    const uint32_t MAXM = 4096; const int SETS = 8;
    const kn::HcShape shape{HC, H, LR, 1e-6f};
    hipblasLtHandle_t lt; if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) return 2;
    kn::HcPlans plans; CCK(plans.init(lt, shape, MAXM));
    auto wd = rnd(size_t(LR) * W, 0.01f), wu = rnd(size_t(W) * LR, 0.05f), wi = rnd(size_t(HC) * W, 0.01f), wn = rnd(W, 0.1f);
    bf16 *bwd = up(wd, SETS), *bwu = up(wu, SETS), *bwi = up(wi, SETS), *bwn = up(wn, SETS);
    bf16 *hyper = up(rnd(size_t(MAXM) * W, 1.f)), *block = up(rnd(size_t(MAXM) * H, 1.f));
    bf16 *normed = dev(size_t(MAXM) * W * 2), *mixed = dev(size_t(MAXM) * H * 2), *out = dev(size_t(MAXM) * W * 2);
    void *ws; HCK(hipMalloc(&ws, kn::hc_mix_workspace_bytes(shape, MAXM) + 256));
    for (int sets : {1, SETS}) for (uint32_t m : {1u, 5u, 8u, 16u, 1205u, 4096u}) {
        const char *sn = sets == 1 ? "hot" : "rot8"; const int burst = m >= 1024 ? 5 : 20;
        auto at = [&](bf16 *b, size_t one, int i) { return b + size_t(i % sets) * one; };
        time_it(sn, m, "hc_mix (norm+down+act+up+gate)", [&](int i) {
            CCK(kn::hc_mix(plans, hyper, at(bwn, W, i), at(bwd, size_t(LR) * W, i), at(bwu, size_t(W) * LR, i), m, normed, mixed, ws, nullptr)); }, burst);
        time_it(sn, m, "hc_combine", [&](int i) { CCK(kn::hc_combine(shape, block, hyper, normed, at(bwi, size_t(HC) * W, i), m, out, nullptr)); }, burst);
        time_it(sn, m, "hc_combine_norm", [&](int i) { CCK(kn::hc_combine_norm(shape, block, hyper, normed, at(bwi, size_t(HC) * W, i), at(bwn, W, i), m, out, nullptr)); }, burst);
    }
    return 0;
}
