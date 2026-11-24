// AmdAmd PAR-6 step 2h: Track LoRA (executor/cuda/src/kernels/lora.cu, hipified) on gfx1151 against its header:
// y += scale(s) · B_s (A_s x) per row's slot, double-float sums, result within 1 bf16 ulp of an FP64 reference
// (including rows where the delta cancels the base), batch invariant (a row's bits are the same applied alone), a row
// without an adapter untouched. Target: o_proj-shaped (k 6,144, n 2,560), pool rank 16, adapters of rank 8 (padding).
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/lora.h"

namespace lora = cere::kernels::lora;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static double ulp_bf16(double v) { const double a = std::fabs(v); if (a < 1e-30) return std::ldexp(1.0, -133); int e; std::frexp(a, &e); return std::ldexp(1.0, e - 8); }

int main() {
    const uint32_t k = 6144, n = 2560, R = 16, r = 8, rows = 4;
    std::mt19937 g(21); std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd = [&](size_t c, float s) { std::vector<uint16_t> v(c); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    lora::Table table;
    std::vector<lora::TargetLayout> targets = {{0, uint64_t(R) * k, k, n}};
    lora::Entry e; static int dummy_w; e.w = &dummy_w; e.k = k; e.segments = 1; e.seg[0] = {0, 0, n};
    if (table.init(2, R, 64, targets, {e}) != CERE_OK) { std::printf("init failed\n"); return 2; }
    std::vector<std::vector<uint16_t>> A(2), B(2); const float scales[2] = {2.0f, 0.5f};
    for (int s = 0; s < 2; ++s) {
        A[s] = rnd(size_t(r) * k, 0.02f); B[s] = rnd(size_t(n) * r, 0.05f);
        const uint32_t ti = 0; const void *ap = A[s].data(), *bp = B[s].data();
        if (table.load(s, r, scales[s], 1, &ti, &ap, &bp) != CERE_OK) { std::printf("load %d failed\n", s); return 2; }
    }
    auto x = rnd(size_t(rows) * k, 1.f), y0 = rnd(size_t(rows) * n, 1.f);
    // row 3 cancels: its base equals minus slot 0's delta (rounded), so y lands near 0 where FP32 sums go wrong
    const int32_t slots[rows] = {0, 1, -1, 0};
    auto delta = [&](int s, uint32_t row, uint32_t c) {
        double acc = 0; for (uint32_t j = 0; j < r; ++j) { double t = 0; for (uint32_t i = 0; i < k; ++i) t += double(bf2f(x[row * k + i])) * bf2f(A[s][j * k + i]); acc += double(bf2f(B[s][c * r + j])) * t; }
        return scales[s] * acc; };
    for (uint32_t c = 0; c < n; ++c) y0[3 * n + c] = f2bf(float(-delta(0, 3, c)));
    void *dx, *dy; HCK(hipMalloc(&dx, x.size() * 2)); HCK(hipMalloc(&dy, y0.size() * 2));
    HCK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice)); HCK(hipMemcpy(dy, y0.data(), y0.size() * 2, hipMemcpyHostToDevice));
    table.activate_rows(slots, rows);
    if (table.apply(e, static_cast<const __hip_bfloat16 *>(dx), rows, static_cast<__hip_bfloat16 *>(dy), n, nullptr) != CERE_OK) { std::printf("apply failed\n"); return 2; }
    table.deactivate(); HCK(hipDeviceSynchronize());
    std::vector<uint16_t> y(y0.size()); HCK(hipMemcpy(y.data(), dy, y.size() * 2, hipMemcpyDeviceToHost));
    int64_t over = 0; double worst = 0;
    for (uint32_t row = 0; row < rows; ++row) for (uint32_t c = 0; c < n; ++c) {
        const double want = slots[row] < 0 ? double(bf2f(y0[row * n + c])) : double(bf2f(y0[row * n + c])) + delta(slots[row], row, c);
        const double err = std::fabs(bf2f(y[row * n + c]) - want) / ulp_bf16(want);
        worst = std::fmax(worst, err); over += err > 1.0;
    }
    const bool untouched = std::memcmp(&y[2 * n], &y0[2 * n], n * 2) == 0;
    std::printf("%-26s %s  4 rows (slots 0, 1, none, 0-cancelling): worst %.2f bf16 ulp vs FP64, %ld over 1 ulp; no-adapter row untouched %s\n",
                "apply (mixed rows)", over == 0 && untouched ? "PASS" : "FAIL", worst, (long)over, untouched ? "yes" : "NO");
    if (over || !untouched) ++g_fail;
    // batch invariance: each adapter row applied alone (m = 1, uniform slot) gives the same bits
    bool inv = true;
    for (uint32_t row = 0; row < rows; ++row) {
        if (slots[row] < 0) continue;
        HCK(hipMemcpy(dy, &y0[row * n], n * 2, hipMemcpyHostToDevice));
        table.activate(slots[row]);
        if (table.apply(e, static_cast<const __hip_bfloat16 *>(dx) + size_t(row) * k, 1, static_cast<__hip_bfloat16 *>(dy), n, nullptr) != CERE_OK) { std::printf("apply 1 failed\n"); return 2; }
        table.deactivate(); HCK(hipDeviceSynchronize());
        std::vector<uint16_t> one(n); HCK(hipMemcpy(one.data(), dy, n * 2, hipMemcpyDeviceToHost));
        inv = inv && std::memcmp(one.data(), &y[row * n], n * 2) == 0;
    }
    std::printf("%-26s %s  every adapter row bit-equal applied alone\n", "batch invariance", inv ? "PASS" : "FAIL");
    if (!inv) ++g_fail;
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
