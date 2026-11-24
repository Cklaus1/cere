// AmdAmd PAR-6 step 6: K3 hyper-connections (executor/cuda/src/kernels/k3_hc.cu, hipified: PDL no-ops, the cluster
// combine+norm dropped for its two-launch path, small-M GEMMs absent so hc_mix runs kUnfused) on gfx1151 at the
// model's shape (hc 4, hidden 2,560, lowrank 320), against k3_hc.h:
//  1. hc_mix at m 7 (hipBLASLt buckets) and 300 (k2_wmma): normed within 1 BF16 ulp of a host mirror of the Gemma
//     RMSNorm, mixed vs FP64 from the kernel's own normed (t = bf16(silu(normed W_down^T / hc)) kept at its BF16
//     rounding point): rel-L2 (limit 1e-2); repeat bit-identical;
//  2. hc_combine vs FP64: out rel-L2 (limit 4e-3);
//  3. hc_combine_norm == hc_combine then hc_mix's per-stream norm, bit for bit (m 7);
//  4. HcPlans::init_rows (rows 5): hc_mix of 5 rows == each row through the instantiated plans at m 1, bit for bit.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k3_hc.h"

namespace kn = cere::kernels;
using bf16 = __hip_bfloat16;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
#define CCK(x) do { cere_status s_ = (x); if (s_ != CERE_OK) { std::printf("cere %s:%d status %d\n", __FILE__, __LINE__, int(s_)); std::exit(2); } } while (0)
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
static const bf16 *B(const uint16_t *p) { return reinterpret_cast<const bf16 *>(p); }
static bf16 *B(uint16_t *p) { return reinterpret_cast<bf16 *>(p); }
static void check(const char *name, bool ok, const char *msg) { std::printf("%-22s %s  %s\n", name, ok ? "PASS" : "FAIL", msg); if (!ok) ++g_fail; }

constexpr uint32_t HC = 4, H = 2560, LR = 320, W = HC * H;

int main() {
    std::mt19937 g(41); std::normal_distribution<float> nd;
    auto rnd = [&](size_t n, float s, float mean = 0.f) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(mean + nd(g) * s); return v; };
    const kn::HcShape shape{HC, H, LR, 1e-6f};
    hipblasLtHandle_t lt; if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) return 2;
    const uint32_t MAXM = 300;
    kn::HcPlans plans; CCK(plans.init(lt, shape, MAXM));
    auto wnorm = rnd(W, 0.1f), wdown = rnd(size_t(LR) * W, 1.f / std::sqrt(float(W))), wup = rnd(size_t(W) * LR, 1.f / std::sqrt(float(LR))), winj = rnd(size_t(HC) * W, 1.f / std::sqrt(float(W)));
    auto *dwn = dev(wnorm), *dwd = dev(wdown), *dwu = dev(wup), *dwi = dev(winj);
    const size_t wsb = kn::hc_mix_workspace_bytes(shape, MAXM); void *dws; HCK(hipMalloc(&dws, wsb + 256));
    auto hyper = rnd(size_t(MAXM) * W, 1.f), block = rnd(size_t(MAXM) * H, 1.f);
    auto *dhy = dev(hyper), *dbo = dev(block);
    uint16_t *dnormed, *dmixed, *dout; HCK(hipMalloc(&dnormed, size_t(MAXM) * W * 2)); HCK(hipMalloc(&dmixed, size_t(MAXM) * H * 2)); HCK(hipMalloc(&dout, size_t(MAXM) * W * 2));
    for (uint32_t m : {7u, 300u}) {   // 1. hc_mix
        CCK(kn::hc_mix(plans, B(dhy), B(dwn), B(dwd), B(dwu), m, B(dnormed), B(dmixed), dws, nullptr));
        auto normed = host(dnormed, size_t(m) * W), mixed = host(dmixed, size_t(m) * H);
        int64_t nbad = 0;
        for (uint32_t r = 0; r < m; ++r) for (uint32_t s = 0; s < HC; ++s) {
            double ss = 0; for (uint32_t j = 0; j < H; ++j) { const double x = bf2f(hyper[size_t(r) * W + s * H + j]); ss += x * x; }
            const double rs = 1 / std::sqrt(ss / H + 1e-6);
            for (uint32_t j = 0; j < H; ++j) { const size_t i = size_t(r) * W + s * H + j;
                const uint16_t want = f2bf(float(bf2f(hyper[i]) * rs * (1 + double(bf2f(wnorm[s * H + j])))));
                nbad += std::abs(int(normed[i]) - int(want)) > 1; }
        }
        double num = 0, den = 0;
        for (uint32_t r = 0; r < m; r += (m > 16 ? 13 : 1)) {
            std::vector<double> t(LR);
            for (uint32_t o = 0; o < LR; ++o) { double a = 0; for (uint32_t i = 0; i < W; ++i) a += double(bf2f(normed[size_t(r) * W + i])) * bf2f(wdown[size_t(o) * W + i]);
                a /= HC; t[o] = bf2f(f2bf(float(a / (1 + std::exp(-a))))); }
            for (uint32_t j = 0; j < H; ++j) {
                double acc = 0;
                for (uint32_t s = 0; s < HC; ++s) { double u = 0; for (uint32_t o = 0; o < LR; ++o) u += t[o] * bf2f(wup[size_t(s * H + j) * LR + o]);
                    acc += 1 / (1 + std::exp(-u)) * bf2f(normed[size_t(r) * W + s * H + j]); }
                const double ref = acc / HC, e = bf2f(mixed[size_t(r) * H + j]) - ref; num += e * e; den += ref * ref;
            }
        }
        const double err = std::sqrt(num / den);
        CCK(kn::hc_mix(plans, B(dhy), B(dwn), B(dwd), B(dwu), m, B(dnormed), B(dmixed), dws, nullptr));
        const bool rep = host(dmixed, size_t(m) * H) == mixed && host(dnormed, size_t(m) * W) == normed;
        char msg[200]; std::snprintf(msg, sizeof msg, "m %3u: normed %ld outputs > 1 ulp; mixed rel-L2 %.2e vs FP64 (1e-2); repeat %s", m, (long)nbad, err, rep ? "bit-equal" : "DIFFERS");
        check("hc_mix", nbad == 0 && err <= 1e-2 && rep, msg);
    }
    {   // 2. hc_combine, 3. hc_combine_norm
        const uint32_t m = 7;
        CCK(kn::hc_mix(plans, B(dhy), B(dwn), B(dwd), B(dwu), m, B(dnormed), B(dmixed), dws, nullptr));
        auto normed = host(dnormed, size_t(m) * W);
        CCK(kn::hc_combine(shape, B(dbo), B(dhy), B(dnormed), B(dwi), m, B(dout), nullptr));
        auto out = host(dout, size_t(m) * W);
        double num = 0, den = 0;
        for (uint32_t r = 0; r < m; ++r) for (uint32_t s = 0; s < HC; ++s) {
            double d = 0; for (uint32_t i = 0; i < W; ++i) d += double(bf2f(normed[size_t(r) * W + i])) * bf2f(winj[size_t(s) * W + i]);
            const double a = 2 / (1 + std::exp(-d / HC));
            for (uint32_t j = 0; j < H; ++j) { const size_t i = size_t(r) * W + s * H + j; const double ref = bf2f(hyper[i]) + a * bf2f(block[size_t(r) * H + j]);
                const double e = bf2f(out[i]) - ref; num += e * e; den += ref * ref; }
        }
        const double err = std::sqrt(num / den);
        char msg[160]; std::snprintf(msg, sizeof msg, "m 7: out rel-L2 %.2e vs FP64 (4e-3)", err);
        check("hc_combine", err <= 4e-3, msg);
        // combine_norm: out2 / normed2 vs combine (above) then hc_mix's norm of `out` (with the next norm weight)
        auto wnext = rnd(W, 0.1f); auto *dwx = dev(wnext);
        uint16_t *dn2, *do2; HCK(hipMalloc(&dn2, size_t(m) * W * 2)); HCK(hipMalloc(&do2, size_t(m) * W * 2));
        HCK(hipMemcpy(dn2, dnormed, size_t(m) * W * 2, hipMemcpyDeviceToDevice));
        CCK(kn::hc_combine_norm(shape, B(dbo), B(dhy), B(dn2), B(dwi), B(dwx), m, B(do2), nullptr));
        uint16_t *dn3; HCK(hipMalloc(&dn3, size_t(m) * W * 2));
        CCK(kn::hc_mix(plans, B(dout), B(dwx), B(dwd), B(dwu), m, B(dn3), B(dmixed), dws, nullptr));
        const bool eq = host(do2, size_t(m) * W) == out && host(dn2, size_t(m) * W) == host(dn3, size_t(m) * W);
        check("hc_combine_norm", eq, "== hc_combine, then hc_mix's per-stream norm of its output: out and normed bit-equal");
    }
    {   // 4. rows plans
        kn::HcPlans rows; CCK(rows.init_rows(lt, plans, 5));
        CCK(kn::hc_mix(rows, B(dhy), B(dwn), B(dwd), B(dwu), 5, B(dnormed), B(dmixed), dws, nullptr));
        auto mr = host(dmixed, 5 * H);
        bool eq = true;
        for (uint32_t r = 0; r < 5; ++r) {
            CCK(kn::hc_mix(plans, B(dhy + size_t(r) * W), B(dwn), B(dwd), B(dwu), 1, B(dnormed), B(dmixed), dws, nullptr));
            auto one = host(dmixed, H); eq &= std::equal(one.begin(), one.end(), mr.begin() + size_t(r) * H);
        }
        check("rows plans", eq, "hc_mix of 5 rows on init_rows plans == each row at m 1 on the instantiated plans, bit for bit");
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
