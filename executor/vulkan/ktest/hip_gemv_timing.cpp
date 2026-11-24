// VulkanAmd P1: same-session HIP reference timing of executor/hip/src/k2_mgemv.cpp (the HIP small-M GEMV) with the
// method of the Vulkan k2mx_test timing section (ktest_util.h time_gemv): max(3, copies) untimed warm-up calls, the
// settle sweeps (BF16 m 1 GEMVs at k 2,560 over the copies' whole span, ~KT_SETTLE_MS at 200 GB/s, default 20,
// <= 64; 0 = the P1 method without them), then 15 calls each between its own pair of hipEvents (GPU time), median /
// min / max; call i reads weight copy i % copies (rotating copies >= 1 GB defeat the caches for small weights; one
// copy = hot). Same shapes and m lists as k2mx_test, including its `xN` rows (N calls per event pair, per-call mean)
// and `--stability R`. Built by build.sh only when hipcc exists (HIP is the reference here, not part of the Vulkan
// executor).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "k2_mgemv.h"

#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
using namespace cere::hipk2;

static uint64_t g_xs = 0x9E3779B97F4A7C15ull;
static uint64_t xs() { g_xs ^= g_xs << 13; g_xs ^= g_xs >> 7; g_xs ^= g_xs << 17; return g_xs; }

struct T { double med, mn, mx; };
static T time_calls(uint8_t *x, uint8_t *w, uint64_t wstride, uint8_t *s, uint64_t sstride, uint32_t copies, void *y, uint32_t m, uint32_t n, uint32_t k,
                    uint32_t per) {
    const int reps = 15;
    std::vector<hipEvent_t> ev(2 * reps);
    for (auto &e : ev) HCK(hipEventCreate(&e));
    uint32_t c = 0;
    auto call = [&] {
        const uint32_t i = c++ % copies;
        if (gemv(reinterpret_cast<const __hip_bfloat16 *>(x), w + uint64_t(i) * wstride, s ? s + uint64_t(i) * sstride : nullptr, y, m, n, k, false, nullptr) != CERE_OK) std::exit(2);
    };
    for (uint32_t i = 0; i < std::max<uint32_t>(3, copies); ++i) call();
    const char *se = std::getenv("KT_SETTLE_MS");
    const double settle = se != nullptr ? std::atof(se) : 20.0;
    const uint64_t span = uint64_t(copies) * wstride;
    const uint32_t sweep_n = uint32_t(span / (2560 * 2));
    const uint32_t sweeps = settle <= 0 || sweep_n == 0 ? 0 : uint32_t(std::min(64.0, std::ceil(settle * 2e8 / double(span))));
    void *scratch = nullptr;
    if (sweeps > 0) HCK(hipMalloc(&scratch, size_t(sweep_n) * 8));
    for (uint32_t i = 0; i < sweeps; ++i)
        if (gemv(reinterpret_cast<const __hip_bfloat16 *>(x), w, nullptr, scratch, 1, sweep_n, 2560, false, nullptr) != CERE_OK) std::exit(2);
    for (int i = 0; i < reps; ++i) {
        HCK(hipEventRecord(ev[2 * i], nullptr));
        for (uint32_t j = 0; j < per; ++j) call();
        HCK(hipEventRecord(ev[2 * i + 1], nullptr));
    }
    HCK(hipDeviceSynchronize());
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1])); t.push_back(ms / per); }
    for (auto &e : ev) HCK(hipEventDestroy(e));
    if (scratch != nullptr) HCK(hipFree(scratch));
    std::sort(t.begin(), t.end());
    return {t[t.size() / 2], t.front(), t.back()};
}

struct Shape { const char *name; uint32_t n, k; uint64_t cold_bytes; };

static void time_shape(const Shape &sh, const std::vector<uint32_t> &ms, bool hot_too, uint32_t per = 1) {
    const uint64_t wb = uint64_t(sh.n) * sh.k * 2, qb = uint64_t(sh.n) * sh.k, sb = qb / 32;
    const uint32_t copies = uint32_t(std::max<uint64_t>(1, (sh.cold_bytes + qb - 1) / qb));
    std::vector<uint16_t> hw(wb * copies / 2); std::vector<uint8_t> hq(qb * copies), hs(sb * copies); std::vector<uint16_t> hx(16 * sh.k);
    for (auto &v : hw) v = uint16_t(0x3C00 + (xs() & 0x03FF) + ((xs() & 1) << 15));
    for (auto &b : hq) { b = uint8_t(xs()); if ((b & 0x7f) == 0x7f) b ^= 1; }
    for (auto &b : hs) b = uint8_t(112 + xs() % 11);
    for (auto &v : hx) v = uint16_t(0x3C00 + (xs() & 0x03FF) + ((xs() & 1) << 15));
    uint8_t *dw, *dq, *ds, *dx; void *dy;
    HCK(hipMalloc(&dw, hw.size() * 2 + 256)); HCK(hipMalloc(&dq, hq.size() + 256)); HCK(hipMalloc(&ds, hs.size() + 256));
    HCK(hipMalloc(&dx, hx.size() * 2 + 256)); HCK(hipMalloc(&dy, size_t(16) * sh.n * 4));
    HCK(hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice)); HCK(hipMemcpy(dq, hq.data(), hq.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(ds, hs.data(), hs.size(), hipMemcpyHostToDevice)); HCK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    for (uint32_t m : ms) for (int hot = 0; hot <= (hot_too ? 1 : 0); ++hot) {
        const uint32_t c = hot ? 1 : copies;
        const T tb = time_calls(dx, dw, wb, nullptr, 0, c, dy, m, sh.n, sh.k, per);
        const T tm = time_calls(dx, dq, qb, ds, sb, c, dy, m, sh.n, sh.k, per);
        std::string tag = hot ? "hot " : (copies > 1 ? "cold" : "    ");
        if (per != 1) tag += " x" + std::to_string(per);
        std::printf("hip      %-24s %s m %2u: bf16 %.3f ms (%.0f GB/s) [min %.3f max %.3f], mxfp8 %.3f ms (%.0f GB/s) [min %.3f max %.3f]\n",
                    sh.name, tag.c_str(), m, tb.med, wb / (tb.med * 1e-3) / 1e9, tb.mn, tb.mx,
                    tm.med, qb / (tm.med * 1e-3) / 1e9, tm.mn, tm.mx);
        std::fflush(stdout);
    }
    HCK(hipFree(dw)); HCK(hipFree(dq)); HCK(hipFree(ds)); HCK(hipFree(dx)); HCK(hipFree(dy));
}

int main(int argc, char **argv) {
    const Shape s320{"short-K n 10240 k 320", 10240, 320, 1000000000ull}, s640{"short-K n 10240 k 640", 10240, 640, 1000000000ull};
    const int stability = argc > 2 && std::string(argv[1]) == "--stability" ? std::atoi(argv[2]) : 0;
    if (stability > 0) {
        for (int r = 0; r < stability; ++r) {
            std::printf("round %d\n", r);
            for (uint32_t per : {1u, 20u}) { time_shape(s320, {4, 16}, false, per); time_shape(s640, {4, 16}, false, per); }
        }
        return 0;
    }
    time_shape({"lm_head 248320 x 2560", 248320, 2560, 0}, {1, 4, 5, 8, 16}, false);
    time_shape(s320, {1, 4, 16}, true);
    time_shape(s640, {1, 4, 16}, true);
    time_shape(s320, {1, 4, 16}, false, 20);
    time_shape(s640, {1, 4, 16}, false, 20);
    return 0;
}
