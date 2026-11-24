// VulkanAmd P2: the same-session HIP reference timing for k5_chunk_test / k5_layer_test (Vulkan), same data and method:
// delta_rule_chunk (executor/hip/src/k5_gdn_chunk_hip.cpp) at 4,096, 1,205 and 512 rows (0.5 everywhere, g -0.05, beta 0.5; one
// call between its own events, median of 7 after a warm-up; and k5_chunk_test's mean of 5 back to back), then
// layer_forward (the hipified k5_gdn_layer: hipBLASLt / k2_wmma projections, cls 0) at hidden 2,560: PREFILL 4,096 and 512
// rows (median of 5) and a DECODE token after a 99-row prefill (median of 15). Built by build.d/k5_gdn_chunk.sh against a
// HIP build tree (K5C_HIPROOT); timing only, no contract rows (those are the HIP ktests).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k5_gdn_chunk.h"
#include "kernels/k5_gdn_layer.h"

namespace k5 = cere::k5;
using bf16 = __hip_bfloat16;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
static const bf16 *B(const uint16_t *p) { return reinterpret_cast<const bf16 *>(p); }
static bf16 *B(uint16_t *p) { return reinterpret_cast<bf16 *>(p); }

template <class F> static double median_ms(int reps, F call) {
    hipEvent_t e0, e1; HCK(hipEventCreate(&e0)); HCK(hipEventCreate(&e1));
    call(); HCK(hipDeviceSynchronize());
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        HCK(hipEventRecord(e0, nullptr)); call(); HCK(hipEventRecord(e1, nullptr)); HCK(hipEventSynchronize(e1));
        float ms; HCK(hipEventElapsedTime(&ms, e0, e1)); t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main() {
    constexpr int KH = 16, VH = 48, D = 128;
    const float scale = 1.f / std::sqrt(float(D));
    {
        const int64_t TB = 4096; std::vector<uint16_t> big(TB * VH * D, f2bf(0.5f)); std::vector<float> gb(TB * VH, -0.05f), bb(TB * VH, 0.5f);
        uint16_t *dbig = dev(big), *dst, *bo; HCK(hipMalloc(&dst, size_t(VH) * D * D * 2)); HCK(hipMalloc(&bo, big.size() * 2));
        float *dgb = dev(gb), *dbb = dev(bb);
        void *bws; HCK(hipMalloc(&bws, k5::chunk_workspace_bytes(TB)));
        for (int64_t rows : {TB, int64_t(1205), int64_t(512)}) {
            k5::ChunkArgs a{}; a.q = B(dbig); a.k = a.q; a.v = a.q; a.g = dgb; a.beta = dbb; a.state = B(dst); a.o = B(bo); a.tokens = rows; a.scale = scale;
            const double med = median_ms(7, [&] { HCK(k5::delta_rule_chunk(a, bws, nullptr)); });
            hipEvent_t e0, e1; HCK(hipEventCreate(&e0)); HCK(hipEventCreate(&e1));
            HCK(hipEventRecord(e0, nullptr)); for (int i = 0; i < 5; ++i) HCK(k5::delta_rule_chunk(a, bws, nullptr));
            HCK(hipEventRecord(e1, nullptr)); HCK(hipEventSynchronize(e1)); float ms; HCK(hipEventElapsedTime(&ms, e0, e1));
            std::printf("HIP chunk   %4ld rows x 48 heads: %.3f ms per call (median of 7); mean of 5 back to back %.3f ms\n", long(rows), med, ms / 5);
        }
        (void)KH;
    }
    {
        constexpr int H2 = 2560, CD = k5::kConvDim, VD = k5::kValueDim; const int64_t TB = 4096;
        std::mt19937 g(5); std::normal_distribution<float> nd(0.f, 1.f);
        auto rnd = [&](size_t n, float s, float mean = 0.f) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(mean + nd(g) * s); return v; };
        const float s2 = 1.f / std::sqrt(float(H2));
        std::vector<float> A_log(VH); for (auto &x : A_log) x = std::log(0.5f + std::fabs(nd(g)));
        k5::LayerWeights w{B(dev(rnd(size_t(CD) * H2, s2))), B(dev(rnd(size_t(VD) * H2, s2))), B(dev(rnd(size_t(VH) * H2, s2))),
                           B(dev(rnd(size_t(VH) * H2, s2))), B(dev(rnd(size_t(CD) * 4, 0.5f))), dev(A_log), B(dev(rnd(VH, 0.5f))),
                           B(dev(rnd(D, 0.1f, 1.f))), B(dev(rnd(size_t(H2) * VD, 1.f / std::sqrt(float(VD)))))};
        hipblasLtHandle_t lt; if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) return 2;
        cere::kernels::GemmPlan pq, pz, pba, po;
        if (pq.init(lt, CD, H2, TB, cere::kernels::GemmOut::kBF16) != CERE_OK || pz.init(lt, VD, H2, TB, cere::kernels::GemmOut::kBF16) != CERE_OK ||
            pba.init(lt, VH, H2, TB, cere::kernels::GemmOut::kBF16) != CERE_OK || po.init(lt, H2, VD, TB, cere::kernels::GemmOut::kBF16) != CERE_OK) { std::printf("plan init failed\n"); return 2; }
        k5::LayerGemms gm{&pq, &pz, &pba, &po};
        const size_t wsb = k5::layer_workspace_bytes(TB); void *dws; HCK(hipMalloc(&dws, wsb));
        uint16_t *dh = dev(rnd(size_t(TB) * H2, 1.f)), *dout; HCK(hipMalloc(&dout, size_t(TB) * H2 * 2));
        uint16_t *conv = dev(std::vector<uint16_t>(size_t(CD) * 3, 0)), *ssm = dev(std::vector<uint16_t>(size_t(VH) * D * D, 0));
        auto fwd = [&](uint32_t op, int64_t n) {
            const cere_status st = k5::layer_forward(gm, w, k5::LayerState{B(conv), B(ssm)}, op, B(dh), n, H2, B(dout), dws, wsb, nullptr, 0, nullptr);
            if (st != CERE_OK) { std::printf("layer_forward failed: %d\n", st); std::exit(2); }
        };
        for (int64_t rows : {TB, int64_t(512)})
            std::printf("HIP layer_forward PREFILL %4ld rows, hidden %d: %.2f ms (median of 5)\n", long(rows), H2, median_ms(5, [&] { fwd(CERE_OP_PREFILL, rows); }));
        fwd(CERE_OP_PREFILL, 99); HCK(hipDeviceSynchronize());
        std::printf("HIP layer_forward DECODE 1 token after a prefill, hidden %d: %.3f ms (median of 15)\n", H2, median_ms(15, [&] { fwd(CERE_OP_DECODE, 1); }));
    }
    return 0;
}
