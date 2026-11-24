// AmdAmd PAR-6 step 7: K7 NVFP4 MoE block (executor/cuda/src/kernels/k7_moe.cu with K0's helpers, hipified: PDL ->
// plain launches, software E2M1 rounding, v_rcp_f32, __dp4a on v_dot4_i32_iu8) on gfx1151, against k7_moe.h at
// H 256, E 16, top-4, I 128, S 128:
//  1. moe_route vs the host (FP32 softmax over all experts, k-th largest with ties to the lowest id, renormalized):
//     ids equal, weights within 2 FP32 ulp;
//  2. moe_block vs a host mirror of the header's arithmetic (K0's activation quantizer with exact reciprocals, the
//     W4A4 block dot products, alphas, BF16 rounding of gate / up / SwiGLU / down, the slot-order combine, the
//     sigmoid-gated BF16 shared expert): out rel-L2 (limit 1e-2: the device's approximate reciprocal can move a
//     block scale to a neighbouring FP8 code), m 7 and 40; repeat bit-identical;
//  3. batch invariance (the header: independent of the batch composition): with rows plans for the router and the
//     shared expert, each row of a 5-row call == that row alone, bit for bit.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k7_moe.h"

namespace kn = cere::kernels;
using bf16 = __hip_bfloat16;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
#define CCK(x) do { cere_status s_ = (x); if (s_ != CERE_OK) { std::printf("cere %s:%d status %d\n", __FILE__, __LINE__, int(s_)); std::exit(2); } } while (0)
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float rb(double x) { return bf2f(f2bf(float(x))); }
static float e4m3f(uint8_t c) { const int s = c >> 7, e = (c >> 3) & 15, m = c & 7; const float v = e == 0 ? std::ldexp(m / 8.f, -6) : std::ldexp(1.f + m / 8.f, e - 7); return s ? -v : v; }
static uint8_t to_e4m3(float x) {  // satfinite, round to nearest even (host reference of the FP8 constructor)
    if (!(x == x)) return 0x7f; uint8_t best = 0; float bd = 1e30f; const float ax = std::min(std::fabs(x), 448.f);
    for (int c = 0; c < 0x7f; ++c) { const float d = std::fabs(e4m3f(uint8_t(c)) - ax); if (d < bd || (d == bd && (c & 1) == 0)) { bd = d; best = uint8_t(c); } }
    return uint8_t(best | (x < 0 ? 0x80 : 0));
}
static const float kE2[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
static int e2m1(float x) {  // round to nearest even onto E2M1, satfinite
    const float a = std::fabs(x); int c;
    if (a <= .25f) c = 0; else if (a < .75f) c = 1; else if (a <= 1.25f) c = 2; else if (a < 1.75f) c = 3; else if (a <= 2.5f) c = 4; else if (a < 3.5f) c = 5; else if (a <= 5.f) c = 6; else c = 7;
    return c | (x < 0 ? 8 : 0);
}
static float e2f(int c) { return (c & 8) ? -kE2[c & 7] : kE2[c & 7]; }
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
static const bf16 *B(const uint16_t *p) { return reinterpret_cast<const bf16 *>(p); }
static void check(const char *name, bool ok, const char *msg) { std::printf("%-20s %s  %s\n", name, ok ? "PASS" : "FAIL", msg); if (!ok) ++g_fail; }

constexpr uint32_t H = 256, E = 16, K = 4, I = 128, S = 128;

// quantize one row (len n, multiple of 16) with K0's rule: codes (n) and block scales (n/16) -> dequantized values in
// units of the GEMM: value = e2m1(code) * e4m3(sf) / global_scale
static void q_row(const float *x, uint32_t n, float gs, std::vector<int> &codes, std::vector<uint8_t> &sfs) {
    codes.resize(n); sfs.resize(n / 16);
    for (uint32_t b = 0; b < n / 16; ++b) {
        float vmax = 0; for (int i = 0; i < 16; ++i) vmax = std::max(vmax, std::fabs(x[b * 16 + i]));
        const uint8_t sf = to_e4m3(gs * (vmax * (1.f / 6.f))); sfs[b] = sf;
        const float os = vmax != 0 ? 1.f / (e4m3f(sf) * (1.f / gs)) : 0.f;
        for (int i = 0; i < 16; ++i) codes[b * 16 + i] = e2m1(x[b * 16 + i] * os);
    }
}
// W4A4 dot of one activation row with one weight row (codes + scales, len n) -> FP32-accumulated sum (no alpha)
static double w4a4(const std::vector<int> &ac, const std::vector<uint8_t> &as, const uint8_t *wc, const uint8_t *ws, uint32_t n) {
    double acc = 0;
    for (uint32_t b = 0; b < n / 16; ++b) {
        double d = 0; for (int i = 0; i < 16; ++i) { const uint32_t j = b * 16 + i; const int c = (wc[j / 2] >> (4 * (j % 2))) & 15; d += double(e2f(ac[j])) * e2f(c); }
        acc += d * e4m3f(as[b]) * e4m3f(ws[b]);
    }
    return acc;
}

int main() {
    std::mt19937 g(43); std::normal_distribution<float> nd;
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    auto codes = [&](size_t n) { std::vector<uint8_t> v(n); for (auto &b : v) b = uint8_t(g()); return v; };
    auto scales = [&](size_t n) { std::vector<uint8_t> v(n); for (auto &b : v) b = uint8_t(0x30 + g() % 0x18); return v; };  // 0.5 .. ~3
    const uint32_t MAXM = 64;
    auto router = rnd(size_t(E) * H, 0.1f), sg = rnd(size_t(S) * H, 1.f / 16), su = rnd(size_t(S) * H, 1.f / 16), sd = rnd(size_t(H) * S, 1.f / 11), seg = rnd(H, 0.06f);
    auto gw = codes(size_t(E) * I * H / 2), uw = codes(size_t(E) * I * H / 2), dw = codes(size_t(E) * H * I / 2);
    auto gs = scales(size_t(E) * I * H / 16), us = scales(size_t(E) * I * H / 16), ds = scales(size_t(E) * H * I / 16);
    std::vector<float> g1(E), g2(E); const float a1 = 2688.f / 4.f, a2 = 2688.f / 2.f;
    for (uint32_t e = 0; e < E; ++e) { g1[e] = (1.f / a1) * 0.004f * (1 + 0.1f * e); g2[e] = (1.f / a2) * 0.004f * (1 + 0.05f * e); }
    kn::MoeDims dims{H, E, K, I, S};
    kn::MoeLayer L{};
    L.router = B(dev(router)); L.gate_weight = dev(gw); L.gate_scale = dev(gs); L.up_weight = dev(uw); L.up_scale = dev(us);
    L.down_weight = dev(dw); L.down_scale = dev(ds); L.g1_alphas = dev(g1); L.g2_alphas = dev(g2); L.a1_global_scale = a1; L.a2_global_scale = a2;
    L.shared_gate = B(dev(sg)); L.shared_up = B(dev(su)); L.shared_down = B(dev(sd)); L.shared_expert_gate = B(dev(seg));
    hipblasLtHandle_t lt; if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) return 2;
    kn::GemmPlan pr, pgu, pd;
    CCK(pr.init(lt, E, H, MAXM, kn::GemmOut::kBF16)); CCK(pgu.init(lt, S, H, MAXM, kn::GemmOut::kBF16)); CCK(pd.init(lt, H, S, MAXM, kn::GemmOut::kBF16));
    kn::MoePlans plans{&pr, &pgu, &pd};
    const size_t wsb = kn::moe_workspace_bytes(dims, MAXM); void *dws; HCK(hipMalloc(&dws, wsb + 256));
    auto x = rnd(size_t(MAXM) * H, 1.f); auto *dx = dev(x); uint16_t *dout; HCK(hipMalloc(&dout, size_t(MAXM) * H * 2));
    {   // 1. moe_route
        const uint32_t m = 40; std::vector<uint16_t> lg = rnd(size_t(m) * E, 2.f);
        for (uint32_t r = 0; r < m; r += 5) lg[r * E + 3] = lg[r * E + 9];  // ties
        auto *dl = dev(lg); int32_t *dids; float *dwts; HCK(hipMalloc(&dids, m * K * 4)); HCK(hipMalloc(&dwts, m * K * 4));
        CCK(kn::moe_route(B(dl), dids, dwts, m, E, K, nullptr));
        auto ids = host(dids, m * K); auto wts = host(dwts, m * K);
        int64_t idbad = 0; double wbad = 0;
        for (uint32_t r = 0; r < m; ++r) {
            std::vector<float> p(E); float mx = -1e30f; for (uint32_t e = 0; e < E; ++e) mx = std::max(mx, bf2f(lg[r * E + e]));
            float sum = 0; for (uint32_t e = 0; e < E; ++e) { p[e] = std::exp(bf2f(lg[r * E + e]) - mx); sum += p[e]; }
            std::vector<uint32_t> idx(E); for (uint32_t e = 0; e < E; ++e) idx[e] = e;
            std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return lg[r * E + a] != lg[r * E + b] ? bf2f(lg[r * E + a]) > bf2f(lg[r * E + b]) : a < b; });
            double ks = 0; for (uint32_t k = 0; k < K; ++k) ks += p[idx[k]] / sum;
            for (uint32_t k = 0; k < K; ++k) { idbad += uint32_t(ids[r * K + k]) != idx[k]; wbad = std::max(wbad, std::fabs(wts[r * K + k] - (p[idx[k]] / sum) / ks) / ((p[idx[k]] / sum) / ks)); }
        }
        char msg[160]; std::snprintf(msg, sizeof msg, "40 rows: %ld ids differ (ties included); weights worst rel %.1e (2.4e-7)", (long)idbad, wbad);
        check("moe_route", idbad == 0 && wbad <= 2.4e-7, msg);
    }
    auto mirror = [&](uint32_t r) {  // the host block for row r -> out[H] (double before the final rounding)
        std::vector<double> out(H);
        std::vector<float> xr(H); for (uint32_t i = 0; i < H; ++i) xr[i] = bf2f(x[r * H + i]);
        std::vector<float> lg(E); for (uint32_t e = 0; e < E; ++e) { double s = 0; for (uint32_t i = 0; i < H; ++i) s += double(xr[i]) * bf2f(router[e * H + i]); lg[e] = rb(s); }
        float mx = *std::max_element(lg.begin(), lg.end()); std::vector<float> p(E); float sum = 0; for (uint32_t e = 0; e < E; ++e) { p[e] = std::exp(lg[e] - mx); sum += p[e]; }
        std::vector<uint32_t> idx(E); for (uint32_t e = 0; e < E; ++e) idx[e] = e;
        std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return lg[a] != lg[b] ? lg[a] > lg[b] : a < b; });
        float ks = 0; for (uint32_t k = 0; k < K; ++k) ks += p[idx[k]] / sum;
        std::vector<int> ac; std::vector<uint8_t> as; q_row(xr.data(), H, a1, ac, as);
        std::vector<double> f(H, 0.0);
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t e = idx[k]; const float wk = (p[e] / sum) / ks;
            std::vector<float> hmid(I);
            for (uint32_t j = 0; j < I; ++j) {
                const size_t row = size_t(e) * I + j;
                const float gt = rb(w4a4(ac, as, &gw[row * H / 2], &gs[row * H / 16], H) * g1[e]);
                const float up = rb(w4a4(ac, as, &uw[row * H / 2], &us[row * H / 16], H) * g1[e]);
                hmid[j] = rb(double(gt) / (1 + std::exp(-double(gt))) * up);
            }
            std::vector<int> hc; std::vector<uint8_t> hs; q_row(hmid.data(), I, a2, hc, hs);
            for (uint32_t o = 0; o < H; ++o) { const size_t row = size_t(e) * H + o; f[o] += double(wk) * rb(w4a4(hc, hs, &dw[row * I / 2], &ds[row * I / 16], I) * g2[e]); }
        }
        std::vector<float> sh(S);
        for (uint32_t j = 0; j < S; ++j) { double a = 0, b = 0; for (uint32_t i = 0; i < H; ++i) { a += double(xr[i]) * bf2f(sg[j * H + i]); b += double(xr[i]) * bf2f(su[j * H + i]); }
            const float ga = rb(a), ub = rb(b); sh[j] = rb(double(ga) / (1 + std::exp(-double(ga))) * ub); }
        double gd = 0; for (uint32_t i = 0; i < H; ++i) gd += double(xr[i]) * bf2f(seg[i]);
        const double sig = 1 / (1 + std::exp(-gd));
        for (uint32_t o = 0; o < H; ++o) { double d = 0; for (uint32_t j = 0; j < S; ++j) d += double(sh[j]) * bf2f(sd[o * S + j]); out[o] = double(rb(f[o])) + sig * rb(d); }
        return out;
    };
    for (uint32_t m : {7u, 40u}) {  // 2. moe_block
        CCK(kn::moe_block(dims, L, plans, B(dx), m, reinterpret_cast<bf16 *>(dout), dws, wsb, nullptr));
        auto out = host(dout, size_t(m) * H);
        double num = 0, den = 0;
        for (uint32_t r = 0; r < m; ++r) { auto ref = mirror(r); for (uint32_t o = 0; o < H; ++o) { const double e = bf2f(out[r * H + o]) - ref[o]; num += e * e; den += ref[o] * ref[o]; } }
        const double err = std::sqrt(num / den);
        CCK(kn::moe_block(dims, L, plans, B(dx), m, reinterpret_cast<bf16 *>(dout), dws, wsb, nullptr));
        const bool rep = host(dout, size_t(m) * H) == out;
        char msg[160]; std::snprintf(msg, sizeof msg, "m %2u: out rel-L2 %.2e vs the host mirror (1e-2); repeat %s", m, err, rep ? "bit-equal" : "DIFFERS");
        check("moe_block", err <= 1e-2 && rep, msg);
    }
    {   // the small-m GEMV kernel (m <= 8) vs grouped_gemm_kernel (m 40): rows 0..6 bit-equal (cls 40 pins the
        // shared expert's GEMMs to the 40-row class, so only the routed GEMM's kernel differs)
        CCK(kn::moe_block(dims, L, plans, B(dx), 40, reinterpret_cast<bf16 *>(dout), dws, wsb, nullptr));
        auto big = host(dout, size_t(40) * H);
        kn::MoePlans p40 = plans; p40.cls = 40;
        CCK(kn::moe_block(dims, L, p40, B(dx), 7, reinterpret_cast<bf16 *>(dout), dws, wsb, nullptr));
        auto small = host(dout, size_t(7) * H);
        int64_t diff = 0; for (size_t i = 0; i < small.size(); ++i) diff += small[i] != big[i];
        char msg[160]; std::snprintf(msg, sizeof msg, "m 7 (GEMV kernel, cls 40) vs rows 0..6 of m 40 (GEMM kernel): %ld of %zu differ", (long)diff, small.size());
        check("GEMV == GEMM kernel", diff == 0, msg);
    }
    {   // the segment WMMA kernel (m > 8) vs the GEMV kernel (m <= 8): rows of the same tokens bit-equal (cls 40)
        CCK(kn::moe_block(dims, L, plans, B(dx), 40, reinterpret_cast<bf16 *>(dout), dws, wsb, nullptr));
        auto big = host(dout, size_t(40) * H);
        kn::MoePlans p40 = plans; p40.cls = 40;
        int64_t diff = 0;
        for (uint32_t r0 : {0u, 8u, 31u}) {
            CCK(kn::moe_block(dims, L, p40, B(dx + size_t(r0) * H), 1, reinterpret_cast<bf16 *>(dout), dws, wsb, nullptr));
            auto one = host(dout, H); for (uint32_t i = 0; i < H; ++i) diff += one[i] != big[size_t(r0) * H + i];
        }
        char msg[200]; std::snprintf(msg, sizeof msg, "rows 0, 8, 31 of m 40 (segment WMMA kernel) vs each row alone (GEMV, cls 40): %ld of %u differ", (long)diff, 3 * H);
        check("WMMA == GEMV kernel", diff == 0, msg);
    }
    {   // 3. batch invariance with rows plans
        kn::GemmPlan rr, rgu, rd; CCK(rr.init_rows(lt, pr, 5)); CCK(rgu.init_rows(lt, pgu, 5)); CCK(rd.init_rows(lt, pd, 5));
        kn::MoePlans rows{&rr, &rgu, &rd};
        CCK(kn::moe_block(dims, L, rows, B(dx), 5, reinterpret_cast<bf16 *>(dout), dws, wsb, nullptr));
        auto all = host(dout, 5 * H); bool eq = true;
        for (uint32_t r = 0; r < 5; ++r) {
            CCK(kn::moe_block(dims, L, plans, B(dx + size_t(r) * H), 1, reinterpret_cast<bf16 *>(dout), dws, wsb, nullptr));
            auto one = host(dout, H); eq &= std::equal(one.begin(), one.end(), all.begin() + size_t(r) * H);
        }
        check("batch invariance", eq, "rows plans: each row of a 5-row block == that row alone, bit for bit");
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
