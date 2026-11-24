// Inkling 2 Mini S4 (inkling design note G1): CPU entry points of the routed-MoE references
// (executor/ref/inkling/{router,quant,moe}_ref.h) for executor/tests/gpu/test_inkling_kernels_moe.py, which compares
// the GPU kernels (moe_ktest.cu) with them bit for bit. Host pointers only (CPU tensors), the trailing stream ignored;
// 0 = success, -1 for an argument refused (null pointer, shape outside the reference's contract).
#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

#include "inkling/moe_ref.h"
#include "inkling/quant_ref.h"
#include "inkling/router_ref.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace {

// fn(i) for i in [0, n) over up to 32 threads (static contiguous split; each i independent).
template <typename F> void parallel_for(int64_t n, F fn) {
    const int64_t threads = std::min<int64_t>(std::max(1u, std::thread::hardware_concurrency()), std::min<int64_t>(32, n));
    std::vector<std::thread> pool;
    for (int64_t t = 0; t < threads; ++t) {
        pool.emplace_back([=]() {
            for (int64_t i = n * t / threads; i < n * (t + 1) / threads; ++i) fn(i);
        });
    }
    for (auto &th : pool) th.join();
}

std::vector<float> widen(const uint16_t *v, size_t n) {
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) f[i] = ref::decode_bf16(v[i]);
    return f;
}

}  // namespace

// x BF16 [m, 2304] (contiguous), w BF16 [128, 2304], bias FP32 [128] -> ids int32 [m, 8], weights FP32 [m, 8],
// logits FP32 [m, 128] (nullable).
CERE_KTEST int cere_ktest_inkling_router_ref(const uint16_t *x, int64_t m, const uint16_t *w, const float *bias,
                                             int32_t *ids, float *weights, float *logits, void *) {
    if (x == nullptr || w == nullptr || bias == nullptr || ids == nullptr || weights == nullptr || m < 1) return -1;
    const std::vector<float> wf = widen(w, size_t(ref::kRouterExperts) * ref::kRouterHidden);
    parallel_for(m, [&](int64_t t) {
        const std::vector<float> xf = widen(x + t * ref::kRouterHidden, ref::kRouterHidden);
        ref::route_row(xf.data(), wf.data(), bias, ids + t * ref::kRouterTopK, weights + t * ref::kRouterTopK,
                       logits != nullptr ? logits + t * ref::kRouterExperts : nullptr);
    });
    return 0;
}

// x BF16 [rows, k] -> codes [rows, k/2], scales [rows, k/16], amax FP32 [rows]; chose4 int32 [rows] (nullable): the
// number of blocks of each row that took the /4 scale.
CERE_KTEST int cere_ktest_inkling_quant_nvfp4_ref(const uint16_t *x, int64_t rows, int64_t k, uint8_t *codes,
                                                  uint8_t *scales, float *amax, int32_t *chose4, void *) {
    if (x == nullptr || codes == nullptr || scales == nullptr || amax == nullptr || rows < 1 || k < 16 || k % 16)
        return -1;
    parallel_for(rows, [&](int64_t r) {
        const std::vector<float> xf = widen(x + r * k, size_t(k));
        int n4 = 0;
        amax[r] = ref::nvfp4_quantize_row(xf.data(), int(k), codes + r * (k / 2), scales + r * (k / 16), &n4);
        if (chose4 != nullptr) chose4[r] = n4;
    });
    return 0;
}

// x BF16 [rows, k] -> values [rows, k], exponents [rows, k/32].
CERE_KTEST int cere_ktest_inkling_quant_mxfp8_ref(const uint16_t *x, int64_t rows, int64_t k, uint8_t *values,
                                                  uint8_t *exponents, void *) {
    if (x == nullptr || values == nullptr || exponents == nullptr || rows < 1 || k < 32 || k % 32) return -1;
    parallel_for(rows, [&](int64_t r) {
        const std::vector<float> xf = widen(x + r * k, size_t(k));
        ref::mxfp8_quantize_row(xf.data(), int(k), values + r * k, exponents + r * (k / 32));
    });
    return 0;
}

// The experts + combine of m tokens (ref::moe_expert_row per routed row, ref::moe_combine_row per token): x BF16
// [m, 2304], ids int32 [m, 8] (ascending), weights FP32 [m, 8], expert weights in the checkpoint layout (moe.h
// MoeWeights; the denominators as scalars) -> out BF16 [m, 2304]; taps (nullable) by routed row id t*8+k: gate_up
// BF16 [m*8, 2304], h BF16 [m*8, 1152], expert_out BF16 [m*8, 2304].
CERE_KTEST int cere_ktest_inkling_moe_ref(const uint16_t *x, int64_t m, const int32_t *ids, const float *weights,
                                          int64_t nvfp4, int64_t num_experts, const uint8_t *w13,
                                          const uint8_t *w13_scale, const uint8_t *w2, const uint8_t *w2_scale,
                                          const float *w13_amax, const float *w2_amax, float w13_den, float w2_den,
                                          uint16_t *out, uint16_t *gate_up_tap, uint16_t *h_tap,
                                          uint16_t *expert_out_tap, void *) {
    constexpr int H = ref::kMoeHidden, I = ref::kMoeInter, K = ref::kRouterTopK;
    if (x == nullptr || ids == nullptr || weights == nullptr || out == nullptr || w13 == nullptr ||
        w13_scale == nullptr || w2 == nullptr || w2_scale == nullptr || m < 1 || num_experts < 1 ||
        (nvfp4 != 0 && (w13_amax == nullptr || w2_amax == nullptr)))
        return -1;
    for (int64_t i = 0; i < m * K; ++i)
        if (ids[i] < 0 || ids[i] >= num_experts) return -1;
    ref::MoeRefLayer L;
    L.nvfp4 = nvfp4 != 0;
    L.w13 = w13, L.w13_scale = w13_scale, L.w2 = w2, L.w2_scale = w2_scale;
    L.w13_amax = w13_amax, L.w2_amax = w2_amax, L.w13_den = w13_den, L.w2_den = w2_den;
    std::vector<uint16_t> o(size_t(m) * K * H);
    parallel_for(m * K, [&](int64_t r) {
        ref::moe_expert_row(L, x + (r / K) * H, ids[r], weights[r], o.data() + r * H,
                            gate_up_tap != nullptr ? gate_up_tap + r * 2 * I : nullptr,
                            h_tap != nullptr ? h_tap + r * I : nullptr);
    });
    for (int64_t t = 0; t < m; ++t) {
        const uint16_t *rows[K];
        for (int k = 0; k < K; ++k) rows[k] = o.data() + (t * K + k) * H;
        ref::moe_combine_row(rows, out + t * H);
    }
    if (expert_out_tap != nullptr) std::copy(o.begin(), o.end(), expert_out_tap);
    return 0;
}

// Scalar conversions of quant_ref.h on host: x FP32 [n] -> E4M3 bytes and E2M1 codes (the GPU sweep's emulation half
// runs on device; this proves the host compile agrees).
CERE_KTEST int cere_ktest_inkling_cvt_ref(const float *x, int64_t n, uint8_t *e4m3, uint8_t *e2m1, void *) {
    if (x == nullptr || e4m3 == nullptr || e2m1 == nullptr || n < 0) return -1;
    for (int64_t i = 0; i < n; ++i) {
        e4m3[i] = ref::encode_e4m3_satfinite(x[i]);
        e2m1[i] = ref::encode_e2m1_satfinite(x[i]);
    }
    return 0;
}
