// AmdAmd PAR-6 step 8: K13 BF16 MoE block of the MTP draft (executor/cuda/src/kernels/k13_moe_bf16.cu, hipified: PDL
// dropped, the E2M1 -> FP16 convert in software) on gfx1151 at H 256, E 16, top-4, I 128, S 128, against k13_moe_bf16.h:
//  1. BF16 experts vs a host FP64 mirror (K7's router / routing / combine, gate and up rounded to BF16, SwiGLU in
//     FP32 rounded to BF16, down rounded to BF16): rel-L2 (limit 4e-3) at m 3 (the <= 16-row tail) and 40; repeat
//     bit-identical;
//  2. NVFP4 W4A16 experts: out bit-equal to the BF16 call on BF16(RN(RN(e2m1 * scale) * global)) tensors, m 3 and 40;
//  3. `first`: rows [first, m) bit-equal to the first = 0 call, rows [0, first) not written (m 9, first 6);
//  4. rows plans: each row of a 5-row call == that row alone, bit for bit.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k13_moe_bf16.h"

namespace kn = cere::kernels;
using bf16 = __hip_bfloat16;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
#define CCK(x) do { cere_status s_ = (x); if (s_ != CERE_OK) { std::printf("cere %s:%d status %d\n", __FILE__, __LINE__, int(s_)); std::exit(2); } } while (0)
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float rb(double x) { return bf2f(f2bf(float(x))); }
static float e4m3f(uint8_t c) { const int s = c >> 7, e = (c >> 3) & 15, m = c & 7; const float v = e == 0 ? std::ldexp(m / 8.f, -6) : std::ldexp(1.f + m / 8.f, e - 7); return s ? -v : v; }
static const float kE2[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
static const bf16 *B(const uint16_t *p) { return reinterpret_cast<const bf16 *>(p); }
static void check(const char *name, bool ok, const char *msg) { std::printf("%-20s %s  %s\n", name, ok ? "PASS" : "FAIL", msg); if (!ok) ++g_fail; }

constexpr uint32_t H = 256, E = 16, K = 4, I = 128, S = 128;

int main() {
    std::mt19937 g(47); std::normal_distribution<float> nd;
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    const uint32_t MAXM = 64;
    auto router = rnd(size_t(E) * H, 0.1f), sg = rnd(size_t(S) * H, 1.f / 16), su = rnd(size_t(S) * H, 1.f / 16), sd = rnd(size_t(H) * S, 1.f / 11), seg = rnd(H, 0.06f);
    // NVFP4 experts and the BF16 tensors they dequantize to
    std::vector<uint8_t> guq(size_t(E) * 2 * I * H / 2), gus(size_t(E) * 2 * I * H / 16), dq(size_t(E) * H * I / 2), dsc(size_t(E) * H * I / 16);
    for (auto &b : guq) b = uint8_t(g()); for (auto &b : dq) b = uint8_t(g());
    for (auto &b : gus) b = uint8_t(0x30 + g() % 0x18); for (auto &b : dsc) b = uint8_t(0x30 + g() % 0x18);
    std::vector<float> gug(E * 2), dg(E); for (auto &x : gug) x = 0.01f * (1 + 0.1f * float(g() % 8)); for (auto &x : dg) x = 0.008f * (1 + 0.1f * float(g() % 8));
    auto deq = [&](const std::vector<uint8_t> &q, const std::vector<uint8_t> &sc, size_t rows, uint32_t cols, auto glob) {
        std::vector<uint16_t> w(rows * cols);
        for (size_t r = 0; r < rows; ++r) for (uint32_t c = 0; c < cols; ++c) {
            const int code = (q[(r * cols + c) / 2] >> (4 * (c % 2))) & 15; const float v = (code & 8) ? -kE2[code & 7] : kE2[code & 7];
            w[r * cols + c] = f2bf((v * e4m3f(sc[(r * cols + c) / 16])) * glob(r));
        }
        return w;
    };
    auto gu = deq(guq, gus, size_t(E) * 2 * I, H, [&](size_t r) { const size_t e = r / (2 * I); return gug[e * 2 + ((r % (2 * I)) >= I ? 1 : 0)]; });
    auto dn = deq(dq, dsc, size_t(E) * H, I, [&](size_t r) { return dg[r / H]; });
    kn::MoeDims dims{H, E, K, I, S};
    kn::MoeBf16Layer Lb{}; Lb.router = B(dev(router)); Lb.gate_up = B(dev(gu)); Lb.down = B(dev(dn));
    Lb.shared_gate = B(dev(sg)); Lb.shared_up = B(dev(su)); Lb.shared_down = B(dev(sd)); Lb.shared_expert_gate = B(dev(seg));
    kn::MoeBf16Layer Lq = Lb; Lq.gate_up = nullptr; Lq.down = nullptr;
    Lq.gate_up_q = dev(guq); Lq.gate_up_scale = dev(gus); Lq.gate_up_global = dev(gug); Lq.down_q = dev(dq); Lq.down_scale = dev(dsc); Lq.down_global = dev(dg);
    hipblasLtHandle_t lt; if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) return 2;
    kn::GemmPlan pr, pgu, pd;
    CCK(pr.init(lt, E, H, MAXM, kn::GemmOut::kBF16)); CCK(pgu.init(lt, S, H, MAXM, kn::GemmOut::kBF16)); CCK(pd.init(lt, H, S, MAXM, kn::GemmOut::kBF16));
    kn::MoePlans plans{&pr, &pgu, &pd};
    const size_t wsb = kn::moe_bf16_workspace_bytes(dims, MAXM); void *dws; HCK(hipMalloc(&dws, wsb + 256));
    auto x = rnd(size_t(MAXM) * H, 1.f); auto *dx = dev(x); uint16_t *dout; HCK(hipMalloc(&dout, size_t(MAXM) * H * 2));
    auto run = [&](const kn::MoeBf16Layer &L, const kn::MoePlans &P, uint32_t m, uint32_t first = 0, const uint16_t *xp = nullptr) {
        CCK(kn::moe_bf16_block(dims, L, P, B(xp ? xp : dx), m, first, reinterpret_cast<bf16 *>(dout), dws, wsb, nullptr));
        return host(dout, size_t(m) * H);
    };
    auto mirror = [&](uint32_t r) {
        std::vector<double> out(H); std::vector<float> xr(H); for (uint32_t i = 0; i < H; ++i) xr[i] = bf2f(x[r * H + i]);
        std::vector<float> lg(E); for (uint32_t e = 0; e < E; ++e) { double s = 0; for (uint32_t i = 0; i < H; ++i) s += double(xr[i]) * bf2f(router[e * H + i]); lg[e] = rb(s); }
        const float mx = *std::max_element(lg.begin(), lg.end()); std::vector<float> p(E); float sum = 0; for (uint32_t e = 0; e < E; ++e) { p[e] = std::exp(lg[e] - mx); sum += p[e]; }
        std::vector<uint32_t> idx(E); for (uint32_t e = 0; e < E; ++e) idx[e] = e;
        std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return lg[a] != lg[b] ? lg[a] > lg[b] : a < b; });
        float ks = 0; for (uint32_t k = 0; k < K; ++k) ks += p[idx[k]] / sum;
        std::vector<double> f(H, 0.0);
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t e = idx[k]; const float wk = (p[e] / sum) / ks; std::vector<float> h(I);
            for (uint32_t j = 0; j < I; ++j) { double a = 0, b = 0; for (uint32_t i = 0; i < H; ++i) { a += double(xr[i]) * bf2f(gu[(size_t(e) * 2 * I + j) * H + i]); b += double(xr[i]) * bf2f(gu[(size_t(e) * 2 * I + I + j) * H + i]); }
                const float ga = rb(a), ub = rb(b); h[j] = rb(double(ga) / (1 + std::exp(-double(ga))) * ub); }
            for (uint32_t o = 0; o < H; ++o) { double d = 0; for (uint32_t j = 0; j < I; ++j) d += double(h[j]) * bf2f(dn[(size_t(e) * H + o) * I + j]); f[o] += double(wk) * rb(d); }
        }
        std::vector<float> sh(S);
        for (uint32_t j = 0; j < S; ++j) { double a = 0, b = 0; for (uint32_t i = 0; i < H; ++i) { a += double(xr[i]) * bf2f(sg[j * H + i]); b += double(xr[i]) * bf2f(su[j * H + i]); }
            const float ga = rb(a), ub = rb(b); sh[j] = rb(double(ga) / (1 + std::exp(-double(ga))) * ub); }
        double gd = 0; for (uint32_t i = 0; i < H; ++i) gd += double(xr[i]) * bf2f(seg[i]);
        const double sig = 1 / (1 + std::exp(-gd));
        for (uint32_t o = 0; o < H; ++o) { double d = 0; for (uint32_t j = 0; j < S; ++j) d += double(sh[j]) * bf2f(sd[o * S + j]); out[o] = double(rb(f[o])) + sig * rb(d); }
        return out;
    };
    for (uint32_t m : {3u, 40u}) {  // 1, 2
        auto out = run(Lb, plans, m);
        double num = 0, den = 0;
        for (uint32_t r = 0; r < m; ++r) { auto ref = mirror(r); for (uint32_t o = 0; o < H; ++o) { const double e = bf2f(out[r * H + o]) - ref[o]; num += e * e; den += ref[o] * ref[o]; } }
        const double err = std::sqrt(num / den);
        const bool rep = run(Lb, plans, m) == out;
        char msg[160]; std::snprintf(msg, sizeof msg, "m %2u: out rel-L2 %.2e vs FP64 (4e-3); repeat %s", m, err, rep ? "bit-equal" : "DIFFERS");
        check("moe_bf16_block", err <= 4e-3 && rep, msg);
        auto outq = run(Lq, plans, m);
        int64_t diff = 0; for (size_t i = 0; i < out.size(); ++i) diff += outq[i] != out[i];
        std::snprintf(msg, sizeof msg, "m %2u: NVFP4 experts vs the BF16 call on their dequantized tensors: %ld of %zu differ", m, (long)diff, out.size());
        check("NVFP4 == BF16 (S2)", diff == 0, msg);
    }
    {   // 3. first
        auto full = run(Lb, plans, 9);
        HCK(hipMemset(dout, 0x55, size_t(9) * H * 2));
        CCK(kn::moe_bf16_block(dims, Lb, plans, B(dx), 9, 6, reinterpret_cast<bf16 *>(dout), dws, wsb, nullptr));
        auto part = host(dout, size_t(9) * H);
        bool tail = std::equal(part.begin() + 6 * H, part.end(), full.begin() + 6 * H), head = true;
        for (size_t i = 0; i < 6 * H; ++i) head &= part[i] == 0x5555;
        check("first", tail && head, tail && head ? "m 9, first 6: rows 6..8 == the first 0 call, rows 0..5 untouched" : "rows DIFFER or written");
    }
    {   // 4. rows plans
        kn::GemmPlan rr, rgu, rd; CCK(rr.init_rows(lt, pr, 5)); CCK(rgu.init_rows(lt, pgu, 5)); CCK(rd.init_rows(lt, pd, 5));
        kn::MoePlans rows{&rr, &rgu, &rd};
        auto all = run(Lb, rows, 5); bool eq = true;
        for (uint32_t r = 0; r < 5; ++r) { auto one = run(Lb, plans, 1, 0, dx + size_t(r) * H); eq &= std::equal(one.begin(), one.end(), all.begin() + size_t(r) * H); }
        check("rows plans", eq, "each row of a 5-row block == that row alone, bit for bit");
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
