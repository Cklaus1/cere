// AmdAmd PAR-6 step 2e: the K5 GDN pieces (executor/cuda/src/kernels/k5_gdn.cu, hipified) on gfx1151, against the
// header contract: conv1d_silu vs a reference that rounds each tap product to BF16 and sums in FP32 (SiLU in FP64),
// state continuity (two calls == one call, bit for bit; conv1d_state_at == the state conv1d_silu leaves), the
// verify chains (conv1d_silu_rows == sequential one-token calls, bit for bit, saves included); gating and l2norm vs
// FP64. Limits: 4e-3 rel-L2 for BF16 outputs, 1e-5 relative for the F32 gating outputs.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k5_gdn.h"

using bf16 = __hip_bfloat16;
namespace k5 = cere::k5;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 16)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
static void report(const char *n, bool ok, const char *d) { std::printf("%-30s %s  %s\n", n, ok ? "PASS" : "FAIL", d); if (!ok) ++g_fail; }
static bf16 *B(uint16_t *p) { return reinterpret_cast<bf16 *>(p); }

int main() {
    std::mt19937 g(9); std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    const int64_t T = 37, dim = 8192;  // Flash-Next's conv width: q|k|v of the GDN layer (2*16*128 + 48*128)
    auto x = rnd(T * dim, 1.f), w = rnd(dim * k5::kConvWidth, 0.5f), st0 = rnd(dim * k5::kConvState, 1.f);
    auto *dx = dev(x); auto *dw = dev(w);
    uint16_t *dy, *dst; HCK(hipMalloc(&dy, T * dim * 2)); HCK(hipMalloc(&dst, dim * k5::kConvState * 2));
    auto conv = [&](int64_t t0, int64_t n, uint16_t *state) { HCK(k5::conv1d_silu(B(dx + t0 * dim), dim, B(dw), B(state), B(dy + t0 * dim), dim, n, dim, nullptr)); };
    {   // one call vs reference
        HCK(hipMemcpy(dst, st0.data(), st0.size() * 2, hipMemcpyHostToDevice)); conv(0, T, dst);
        auto y = host(dy, T * dim); auto st = host(dst, dim * k5::kConvState);
        double num = 0, den = 0; bool state_ok = true;
        for (int64_t t = 0; t < T; ++t) for (int64_t c = 0; c < dim; ++c) {
            float acc = 0;
            for (int j = 0; j < k5::kConvWidth; ++j) {
                const int64_t tt = t - 3 + j; const float xv = tt < 0 ? bf2f(st0[c * 3 + (tt + 3)]) : bf2f(x[tt * dim + c]);
                acc += bf2f(f2bf(xv * bf2f(w[c * k5::kConvWidth + j])));
            }
            const double ref = double(acc) / (1 + std::exp(-double(acc))), got = bf2f(y[t * dim + c]);
            num += (got - ref) * (got - ref); den += ref * ref;
        }
        for (int64_t c = 0; c < dim; ++c) for (int j = 0; j < 3; ++j) state_ok = state_ok && st[c * 3 + j] == x[(T - 3 + j) * dim + c];
        char d[100]; std::snprintf(d, sizeof d, "T=%ld dim=%ld rel-L2 %.2e (limit 4e-3), state = last 3 inputs %s", (long)T, (long)dim, std::sqrt(num / den), state_ok ? "yes" : "NO");
        report("conv1d_silu", std::sqrt(num / den) <= 4e-3 && state_ok, d);
        // split 20 + 17 == one call
        auto y_one = y;
        HCK(hipMemcpy(dst, st0.data(), st0.size() * 2, hipMemcpyHostToDevice)); conv(0, 20, dst); conv(20, T - 20, dst);
        auto y_split = host(dy, T * dim); auto st_split = host(dst, dim * k5::kConvState);
        report("conv1d_silu split 20+17", y_split == y_one && st_split == st, "outputs and state bit-equal to one call");
        // state_at
        uint16_t *dsa; HCK(hipMalloc(&dsa, dim * 3 * 2)); HCK(k5::conv1d_state_at(B(dx), dim, B(dsa), T, dim, nullptr));
        report("conv1d_state_at", host(dsa, dim * 3) == st, "== the state conv1d_silu leaves");
    }
    {   // verify chains: rows 3/1/2 on distinct states == sequential one-token calls
        const int chains = 3, rows_of[3] = {3, 1, 2}, R = 6;
        k5::ConvChains cc{}; cc.chains = chains;
        std::vector<uint16_t *> states(chains), saves(R, nullptr);
        for (int c = 0, f = 0; c < chains; ++c) { cc.first[c] = f; cc.rows[c] = rows_of[c]; states[c] = dev(st0); cc.states[c] = B(states[c]); f += rows_of[c]; }
        for (int r = 0; r < R; ++r) { HCK(hipMalloc(&saves[r], dim * 3 * 2)); cc.saves[r] = B(saves[r]); }
        uint16_t *dyr; HCK(hipMalloc(&dyr, R * dim * 2));
        HCK(k5::conv1d_silu_rows(B(dx), dim, B(dw), cc, B(dyr), dim, dim, nullptr));
        auto Y = host(dyr, R * dim); bool eq = true;
        for (int c = 0, f = 0; c < chains; ++c) {
            auto *s1 = dev(st0);
            for (int i = 0; i < rows_of[c]; ++i) {
                const int r = f + i; HCK(k5::conv1d_silu(B(dx + r * dim), dim, B(dw), B(s1), B(dy), dim, 1, dim, nullptr));
                eq = eq && std::memcmp(host(dy, dim).data(), &Y[r * dim], dim * 2) == 0;
                eq = eq && host(s1, dim * 3) == host(i + 1 < rows_of[c] ? saves[r] : states[c], dim * 3);
            }
            f += rows_of[c];
        }
        report("conv1d_silu_rows", eq, "3 chains (3/1/2): outputs, saves, states bit-equal to sequential");
    }
    {   // gating vs FP64
        const int64_t tokens = 9, heads = 48;
        auto a = rnd(tokens * heads, 3.f), b = rnd(tokens * heads, 3.f), dt = rnd(heads, 1.f);
        std::vector<float> A_log(heads); for (auto &v : A_log) v = std::log(0.1f + std::fabs(nd(g)));
        a[0] = f2bf(30.f);  // softplus linear branch (x > 20)
        auto *da = dev(a); auto *db = dev(b); auto *ddt = dev(dt); auto *dA = dev(A_log); float *dg, *dbeta;
        HCK(hipMalloc(&dg, tokens * heads * 4)); HCK(hipMalloc(&dbeta, tokens * heads * 4));
        HCK(k5::gating(B(da), heads, B(db), heads, dA, B(ddt), dg, dbeta, tokens, heads, nullptr));
        auto gg = host(dg, tokens * heads), bb = host(dbeta, tokens * heads); double worst = 0; bool beta_ok = true;
        for (int64_t i = 0; i < tokens * heads; ++i) {
            const int h = int(i % heads); const double xx = double(bf2f(a[i])) + bf2f(dt[h]);
            // the contract's softplus: log(1 + exp(x)) with 1 + exp(x) rounded to FP32 (as the FP32 kernel and SGLang's
            // Triton kernel compute it; FP64 log1p would differ by the FP32 cancellation for very negative x)
            const double sp = xx > 20 ? xx : std::log(double(1.0f + float(std::exp(xx)))), ref = -std::exp(double(A_log[h])) * sp;
            worst = std::fmax(worst, std::fabs(gg[i] - ref) / std::fmax(std::fabs(ref), 1e-30));
            const float sig = float(1 / (1 + std::exp(-double(bf2f(b[i])))));
            beta_ok = beta_ok && std::fabs(bb[i] - bf2f(f2bf(sig))) <= 1e-2f * std::fabs(sig) + 1e-6f;  // bf16(sigmoid)
        }
        char d[100]; std::snprintf(d, sizeof d, "g worst rel err %.2e (limit 1e-5), beta = bf16(sigmoid(b)) %s", worst, beta_ok ? "yes" : "NO");
        report("gating", worst <= 1e-5 && beta_ok, d);
    }
    {   // l2norm vs FP64
        const int64_t rows = 7 * 16; const int64_t d128 = k5::kHeadDim; const float eps = 1e-6f;
        auto xx = rnd(rows * d128, 2.f); auto *dxx = dev(xx); uint16_t *dyy; HCK(hipMalloc(&dyy, rows * d128 * 2));
        HCK(k5::l2norm(B(dxx), B(dyy), rows, d128, eps, nullptr)); auto yy = host(dyy, rows * d128);
        double num = 0, den = 0;
        for (int64_t r = 0; r < rows; ++r) { double ss = 0; for (int i = 0; i < d128; ++i) ss += double(bf2f(xx[r * d128 + i])) * bf2f(xx[r * d128 + i]);
            for (int i = 0; i < d128; ++i) { const double ref = bf2f(xx[r * d128 + i]) / std::sqrt(ss + eps), got = bf2f(yy[r * d128 + i]); num += (got - ref) * (got - ref); den += ref * ref; } }
        char d[64]; std::snprintf(d, sizeof d, "rel-L2 %.2e (limit 4e-3)", std::sqrt(num / den)); report("l2norm", std::sqrt(num / den) <= 4e-3, d);
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
