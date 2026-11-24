// VulkanAmd P2: same-session HIP reference timing of the hipified K5 GDN decode and pieces (gen_hip.py's
// kernels/k5_gdn_decode.hip, kernels/k5_gdn.hip) at the decode shapes k5_decode_test / k5_pieces_test time on Vulkan, with
// the same method: 50 untimed warm-up calls, then "call" = 15 calls each between its own pair of hipEvents (median, min,
// max) and "stream" = 9 samples of 50 back-to-back calls on one stream / 50 (median). HIP is the reference here, not part
// of the Vulkan executor; built by build.d/k5_gdn.sh only when hipcc and a hipified tree exist. Not a contract test.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "kernels/k5_gdn.h"
#include "kernels/k5_gdn_decode.h"

#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
using bf16 = __hip_bfloat16;
namespace k5 = cere::k5;

static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
static bf16 *B(uint16_t *p) { return reinterpret_cast<bf16 *>(p); }

static void time_it(const char *name, const std::function<void()> &call, int reps = 15, int samples = 9, int burst = 50) {
    for (int i = 0; i < 50; ++i) call();
    std::vector<hipEvent_t> ev(2 * std::max(reps, samples));
    for (auto &e : ev) HCK(hipEventCreate(&e));
    for (int i = 0; i < reps; ++i) { HCK(hipEventRecord(ev[2 * i], nullptr)); call(); HCK(hipEventRecord(ev[2 * i + 1], nullptr)); }
    HCK(hipDeviceSynchronize());
    std::vector<double> t, u;
    for (int i = 0; i < reps; ++i) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1])); t.push_back(ms); }
    for (int s = 0; s < samples; ++s) {
        HCK(hipEventRecord(ev[2 * s], nullptr));
        for (int i = 0; i < burst; ++i) call();
        HCK(hipEventRecord(ev[2 * s + 1], nullptr));
    }
    HCK(hipDeviceSynchronize());
    for (int s = 0; s < samples; ++s) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * s], ev[2 * s + 1])); u.push_back(ms / burst); }
    for (auto &e : ev) HCK(hipEventDestroy(e));
    std::sort(t.begin(), t.end()); std::sort(u.begin(), u.end());
    std::printf("    %-34s call %7.2f us (min %.2f, max %.2f)  stream %7.2f us/call\n", name, t[t.size() / 2] * 1e3, t.front() * 1e3,
                t.back() * 1e3, u[u.size() / 2] * 1e3);
}

int main() {
    hipDeviceProp_t prop; HCK(hipGetDeviceProperties(&prop, 0));
    std::printf("HIP device: %s (%s)\n", prop.name, prop.gcnArchName);
    std::mt19937 g(5); std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    constexpr int KH = 16, VH = 48, D = 128;
    const float scale = 1.f / std::sqrt(float(D));
    std::printf("timing decode (hipEvents):\n");
    {
        std::vector<float> A_log(VH); for (auto &x : A_log) x = std::log(0.5f + std::fabs(nd(g)));
        auto dt = rnd(VH, 0.5f), state0 = rnd(size_t(VH) * D * D, 0.05f);
        const int64_t qkv_stride = KH * D * 2 + VH * D;
        auto QKV = rnd(8 * qkv_stride, 1.f), Aa = rnd(8 * VH, 1.f), Bb = rnd(8 * VH, 1.f);
        auto *dA = dev(A_log); auto *ddt = dev(dt); auto *dstate = dev(state0); auto *dqkv = dev(QKV); auto *dAa = dev(Aa); auto *dBb = dev(Bb);
        uint16_t *dO; HCK(hipMalloc(&dO, 8 * VH * D * 2));
        std::vector<uint16_t *> saves(8);
        for (auto &s : saves) HCK(hipMalloc(&s, size_t(VH) * D * D * 2));
        const k5::DecodeArgs one{B(dqkv), B(dqkv + KH * D), B(dqkv + 2 * KH * D), B(dAa), B(dBb), dA, B(ddt), B(dstate), B(dO), scale};
        time_it("delta_rule_decode (1 token)", [&] { HCK(k5::delta_rule_decode(one, nullptr)); });
        k5::DecodeRowsArgs ra{};
        ra.step = one; ra.step.state = nullptr; ra.chains = 1; ra.first[0] = 0; ra.rows[0] = 8; ra.states[0] = B(dstate);
        for (int r = 0; r < 8; ++r) ra.saves[r] = B(saves[r]);
        ra.qkv_stride = qkv_stride; ra.ab_stride = VH; ra.o_stride = VH * D;
        time_it("delta_rule_decode_rows (1 chain x 8)", [&] { HCK(k5::delta_rule_decode_rows(ra, nullptr)); });
        std::vector<k5::DecodeArgs> seq;
        for (int r = 0; r < 8; ++r)
            seq.push_back({B(dqkv + r * qkv_stride), B(dqkv + r * qkv_stride + KH * D), B(dqkv + r * qkv_stride + 2 * KH * D), B(dAa + r * VH),
                           B(dBb + r * VH), dA, B(ddt), B(dstate), B(dO), scale});
        time_it("8 x delta_rule_decode (sequential)", [&] { for (auto &a : seq) HCK(k5::delta_rule_decode(a, nullptr)); });
    }
    std::printf("timing pieces (hipEvents):\n");
    {
        const int64_t dim = 8192, heads = 48;
        auto x = rnd(37 * dim, 1.f), w = rnd(dim * 4, 0.5f), st0 = rnd(dim * 3, 1.f);
        auto *dx = dev(x); auto *dw = dev(w); auto *dst = dev(st0);
        uint16_t *dy; HCK(hipMalloc(&dy, 37 * dim * 2));
        time_it("conv1d_silu (1 token, dim 8192)", [&] { HCK(k5::conv1d_silu(B(dx), dim, B(dw), B(dst), B(dy), dim, 1, dim, nullptr)); });
        k5::ConvChains c8{}; c8.chains = 1; c8.first[0] = 0; c8.rows[0] = 8; c8.states[0] = B(dst);
        std::vector<uint16_t *> sv(7);
        for (int r = 0; r < 7; ++r) { HCK(hipMalloc(&sv[r], dim * 3 * 2)); c8.saves[r] = B(sv[r]); }
        time_it("conv1d_silu_rows (1 chain x 8)", [&] { HCK(k5::conv1d_silu_rows(B(dx), dim, B(dw), c8, B(dy), dim, dim, nullptr)); });
        auto a = rnd(8 * heads, 3.f), b = rnd(8 * heads, 3.f), dt = rnd(heads, 1.f);
        std::vector<float> A_log(heads, -0.5f);
        auto *da = dev(a); auto *db = dev(b); auto *ddt = dev(dt); auto *dA = dev(A_log);
        float *dg, *dbeta; HCK(hipMalloc(&dg, 8 * heads * 4)); HCK(hipMalloc(&dbeta, 8 * heads * 4));
        time_it("gating (1 token, 48 heads)", [&] { HCK(k5::gating(B(da), heads, B(db), heads, dA, B(ddt), dg, dbeta, 1, heads, nullptr)); });
        time_it("gating (8 tokens, 48 heads)", [&] { HCK(k5::gating(B(da), heads, B(db), heads, dA, B(ddt), dg, dbeta, 8, heads, nullptr)); });
        time_it("l2norm (32 rows)", [&] { HCK(k5::l2norm(B(dx), B(dy), 32, 128, 1e-6f, nullptr)); });
        time_it("l2norm (256 rows = 8 tokens)", [&] { HCK(k5::l2norm(B(dx), B(dy), 256, 128, 1e-6f, nullptr)); });
    }
    return 0;
}
