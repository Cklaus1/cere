// AmdAmd PAR-6 step 3c: the whole K5 GDN mixer layer (executor/cuda/src/kernels/k5_gdn_layer.cu, hipified: K2 via the
// HIP adapter, K1 gated norm, conv1d / gating / l2norm hipified, delta_rule_chunk the hand-written HIP kernel,
// delta_rule_decode hipified) on gfx1151, at hidden 256 (the layer takes hidden_size; the GDN dims are the model's).
// Checks:
//  1. layer_forward PREFILL, 100 tokens (a full and a partial chunk) from a fresh state, vs an FP64 reference of
//     Qwen3_5GatedDeltaNet.forward from the same BF16 inputs (projections rounded to BF16 as the GEMMs store them;
//     conv1d + SiLU; gating; the exact gated delta rule on L2-normalised q, k; gated RMSNorm; out_proj): out rel-L2
//     (limit 2e-2: the chunk form's BF16 rounding points);
//  2. prefill 99 then DECODE 1 vs prefill 100: the last row's out rel-L2 (limit 2e-2: chunk vs decode arithmetic),
//     ssm state rel-L2 (limit 1e-2), the conv state (the last 3 raw in_proj_qkv rows) bit-equal (cls 128 both ways:
//     row 99 alone and inside the 100-row GEMM run the shift-invariant k2_wmma);
//  3. EndState at row 64 (FP32 ssm + conv state captured, then a 36-row PREFILL from them):
//     a. layer_core (the GDN mixer, the projections given): the capture leaves the call unchanged and rows 64..99 of
//        y, the final conv and ssm are bit-equal to the single call (the m210 D15 contract of the mixer);
//     b. layer_forward (with the K2 projections): the same, out rows 64..99 bit-equal. This needs a shift-invariant
//        K2 (hipBLASLt on gfx1151 is not: amd-amd design note K2-SHIFT; the plan runs k2_wmma for row classes > 16).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k5_gdn_layer.h"

namespace k5 = cere::k5;
using bf16 = __hip_bfloat16;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static double rb(double x) { return bf2f(f2bf(float(x))); }
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
static const bf16 *B(const uint16_t *p) { return reinterpret_cast<const bf16 *>(p); }
static bf16 *B(uint16_t *p) { return reinterpret_cast<bf16 *>(p); }
static double rel(const double *ref, const uint16_t *got, size_t n) {
    double a = 0, d = 0; for (size_t i = 0; i < n; ++i) { const double e = bf2f(got[i]) - ref[i]; a += e * e; d += ref[i] * ref[i]; }
    return std::sqrt(a / d);
}
static void check(const char *name, bool ok, const char *fmt, double a = 0, double b = 0, double c = 0) {
    std::printf("%-24s %s  ", name, ok ? "PASS" : "FAIL"); std::printf(fmt, a, b, c); std::printf("\n"); if (!ok) ++g_fail;
}

constexpr int HID = 256, CD = k5::kConvDim, KD = k5::kKeyDim, VD = k5::kValueDim, VH = k5::kValueHeads, D = 128;

int main() {
    std::mt19937 g(17); std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd = [&](size_t n, float s, float mean = 0.f) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(mean + nd(g) * s); return v; };
    const float ws = 1.f / std::sqrt(float(HID));
    auto Wqkv = rnd(size_t(CD) * HID, ws), Wz = rnd(size_t(VD) * HID, ws), Wb = rnd(size_t(VH) * HID, ws), Wa = rnd(size_t(VH) * HID, ws);
    auto Wout = rnd(size_t(HID) * VD, 1.f / std::sqrt(float(VD))), conv_w = rnd(size_t(CD) * 4, 0.5f);
    auto dt_bias = rnd(VH, 0.5f), norm_w = rnd(D, 0.1f, 1.f);
    std::vector<float> A_log(VH); for (auto &x : A_log) x = std::log(0.5f + std::fabs(nd(g)));
    const int64_t T = 100;
    auto hidden = rnd(size_t(T) * HID, 1.f);

    k5::LayerWeights w{B(dev(Wqkv)), B(dev(Wz)), B(dev(Wb)), B(dev(Wa)), B(dev(conv_w)), dev(A_log), B(dev(dt_bias)), B(dev(norm_w)), B(dev(Wout))};
    hipblasLtHandle_t lt; if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) return 2;
    cere::kernels::GemmPlan pq, pz, pba, po;
    if (pq.init(lt, CD, HID, 128, cere::kernels::GemmOut::kBF16) != CERE_OK || pz.init(lt, VD, HID, 128, cere::kernels::GemmOut::kBF16) != CERE_OK ||
        pba.init(lt, VH, HID, 128, cere::kernels::GemmOut::kBF16) != CERE_OK || po.init(lt, HID, VD, 128, cere::kernels::GemmOut::kBF16) != CERE_OK) { std::printf("plan init failed\n"); return 2; }
    k5::LayerGemms gm{&pq, &pz, &pba, &po};
    const size_t wsb = k5::layer_workspace_bytes(T); void *dws; HCK(hipMalloc(&dws, wsb));
    auto *dh = dev(hidden); uint16_t *dout; HCK(hipMalloc(&dout, size_t(T) * HID * 2));
    const std::vector<uint16_t> conv0(size_t(CD) * 3, 0), ssm0(size_t(VH) * D * D, 0);
    struct St { uint16_t *conv, *ssm; };
    auto fresh = [&]() { return St{dev(conv0), dev(ssm0)}; };
    auto fwd = [&](St s, uint32_t op, int64_t r0, int64_t n, const k5::EndState *end = nullptr) {
        const cere_status st = k5::layer_forward(gm, w, k5::LayerState{B(s.conv), B(s.ssm)}, op, B(dh + r0 * HID), n, HID,
                                                 B(dout + r0 * HID), dws, wsb, nullptr, 128, end);
        if (st != CERE_OK) { std::printf("layer_forward failed: %d\n", st); std::exit(2); }
    };

    // --- 1. prefill 100 vs FP64
    St s1 = fresh(); fwd(s1, CERE_OP_PREFILL, 0, T);
    auto out1 = host(dout, size_t(T) * HID); auto conv1 = host(s1.conv, conv0.size()); auto ssm1 = host(s1.ssm, ssm0.size());
    std::vector<double> ref(size_t(T) * HID), Sref(size_t(VH) * D * D, 0.0);
    {
        std::vector<double> mixed(size_t(T) * CD), z(size_t(T) * VD), bb(T * VH), aa(T * VH);
        auto proj = [&](const std::vector<uint16_t> &W, int n, std::vector<double> &y) {
            for (int64_t t = 0; t < T; ++t) for (int o = 0; o < n; ++o) {
                double s = 0; for (int i = 0; i < HID; ++i) s += double(bf2f(hidden[t * HID + i])) * bf2f(W[size_t(o) * HID + i]);
                y[t * n + o] = rb(s);
            }
        };
        proj(Wqkv, CD, mixed); proj(Wz, VD, z); proj(Wb, VH, bb); proj(Wa, VH, aa);
        std::vector<double> cv(size_t(T) * CD);
        for (int64_t t = 0; t < T; ++t) for (int c = 0; c < CD; ++c) {
            double s = 0; for (int j = 0; j < 4; ++j) { const int64_t tt = t - 3 + j; if (tt >= 0) s += double(bf2f(conv_w[c * 4 + j])) * mixed[tt * CD + c]; }
            cv[t * CD + c] = rb(s / (1 + std::exp(-s)));
        }
        std::vector<double> o(size_t(T) * VD);
        for (int vh = 0; vh < VH; ++vh) {
            const int kh = vh / 3; double *S = &Sref[size_t(vh) * D * D];
            for (int64_t t = 0; t < T; ++t) {
                double q[D], k[D], nq = 0, nk = 0;
                for (int c = 0; c < D; ++c) { q[c] = cv[t * CD + kh * D + c]; k[c] = cv[t * CD + KD + kh * D + c]; nq += q[c] * q[c]; nk += k[c] * k[c]; }
                for (int c = 0; c < D; ++c) { q[c] /= std::sqrt(nq + 1e-6); k[c] /= std::sqrt(nk + 1e-6); }
                const double x = aa[t * VH + vh] + bf2f(dt_bias[vh]);
                const double gg = -std::exp(double(A_log[vh])) * (x <= 20 ? std::log1p(std::exp(x)) : x);
                const double beta = rb(1 / (1 + std::exp(-bb[t * VH + vh]))), decay = std::exp(gg);
                for (int cv_ = 0; cv_ < D; ++cv_) {
                    double sk = 0; for (int ck = 0; ck < D; ++ck) { S[cv_ * D + ck] *= decay; sk += S[cv_ * D + ck] * k[ck]; }
                    const double dl = (cv[t * CD + 2 * KD + vh * D + cv_] - sk) * beta; double so = 0;
                    for (int ck = 0; ck < D; ++ck) { S[cv_ * D + ck] += dl * k[ck]; so += S[cv_ * D + ck] * q[ck]; }
                    o[t * VD + vh * D + cv_] = rb(so / std::sqrt(double(D)));
                }
            }
        }
        std::vector<double> y(size_t(T) * VD);
        for (int64_t t = 0; t < T; ++t) for (int vh = 0; vh < VH; ++vh) {
            double ss = 0; for (int c = 0; c < D; ++c) ss += o[t * VD + vh * D + c] * o[t * VD + vh * D + c];
            const double r = 1 / std::sqrt(ss / D + 1e-6);
            for (int c = 0; c < D; ++c) { const size_t i = t * VD + vh * D + c; y[i] = rb(o[i] * r * bf2f(norm_w[c]) / (1 + std::exp(-z[i]))); }
        }
        for (int64_t t = 0; t < T; ++t) for (int hh = 0; hh < HID; ++hh) {
            double s = 0; for (int i = 0; i < VD; ++i) s += y[t * VD + i] * bf2f(Wout[size_t(hh) * VD + i]);
            ref[t * HID + hh] = s;
        }
    }
    const double e1 = rel(ref.data(), out1.data(), ref.size()), es = rel(Sref.data(), ssm1.data(), Sref.size());
    check("prefill vs FP64", e1 <= 2e-2 && es <= 2e-2, "100 tokens: out rel-L2 %.2e, ssm state %.2e (limit 2e-2)", e1, es);

    // --- 2. prefill 99 + decode 1 vs prefill 100
    St s2 = fresh(); fwd(s2, CERE_OP_PREFILL, 0, T - 1); fwd(s2, CERE_OP_DECODE, T - 1, 1);
    auto out2 = host(dout, size_t(T) * HID); auto conv2 = host(s2.conv, conv0.size()); auto ssm2 = host(s2.ssm, ssm0.size());
    std::vector<double> last1(HID), s1d(ssm1.size()); for (int i = 0; i < HID; ++i) last1[i] = bf2f(out1[(T - 1) * HID + i]);
    for (size_t i = 0; i < ssm1.size(); ++i) s1d[i] = bf2f(ssm1[i]);
    const double e2 = rel(last1.data(), out2.data() + (T - 1) * HID, HID), e2s = rel(s1d.data(), ssm2.data(), s1d.size());
    int64_t cdiff = 0, culp = 0;
    for (size_t i = 0; i < conv1.size(); ++i) if (conv2[i] != conv1[i]) { ++cdiff; culp = std::max<int64_t>(culp, std::abs(int(conv2[i]) - int(conv1[i]))); }
    check("prefill 99 + decode 1", e2 <= 2e-2 && e2s <= 1e-2 && cdiff == 0, "last row out rel-L2 %.2e (2e-2), ssm %.2e (1e-2); conv state: %.0f entries differ",
          e2, e2s, double(cdiff));
    (void)culp;

    // --- 3a. EndState at row 64 on layer_core: the projections computed once, the mixer captured and resumed
    float *cap32; HCK(hipMalloc(&cap32, ssm0.size() * 4)); uint16_t *capconv; HCK(hipMalloc(&capconv, conv0.size() * 2));
    {
        uint16_t *mx, *zz, *bq, *aq, *y1, *y2;
        HCK(hipMalloc(&mx, size_t(T) * CD * 2)); HCK(hipMalloc(&zz, size_t(T) * VD * 2)); HCK(hipMalloc(&bq, T * VH * 2)); HCK(hipMalloc(&aq, T * VH * 2));
        HCK(hipMalloc(&y1, size_t(T) * VD * 2)); HCK(hipMalloc(&y2, size_t(T) * VD * 2));
        if (pq.run(B(dh), w.in_proj_qkv, T, mx, nullptr, 128) != CERE_OK || pz.run(B(dh), w.in_proj_z, T, zz, nullptr, 128) != CERE_OK ||
            pba.run(B(dh), w.in_proj_b, T, bq, nullptr, 128) != CERE_OK || pba.run(B(dh), w.in_proj_a, T, aq, nullptr, 128) != CERE_OK) return 2;
        const size_t cws = k5::core_workspace_bytes(T); void *dcw; HCK(hipMalloc(&dcw, cws));
        auto core = [&](St s, int64_t r0, int64_t n, uint16_t *y, const k5::EndState *e) {
            const cere_status st = k5::layer_core(w, k5::LayerState{B(s.conv), B(s.ssm)}, CERE_OP_PREFILL, B(mx + r0 * CD), B(zz + r0 * VD),
                                                  B(bq + r0 * VH), B(aq + r0 * VH), n, B(y + r0 * VD), dcw, cws, nullptr, e);
            if (st != CERE_OK) { std::printf("layer_core failed: %d\n", st); std::exit(2); }
        };
        St c1 = fresh(); core(c1, 0, T, y1, nullptr);
        auto yr = host(y1, size_t(T) * VD); auto cc1 = host(c1.conv, conv0.size()); auto cs1 = host(c1.ssm, ssm0.size());
        k5::EndState end{}; end.ssm_out32 = cap32; end.conv_out = B(capconv); end.row = 64;
        St c2 = fresh(); core(c2, 0, T, y2, &end);
        const bool neutral = host(y2, size_t(T) * VD) == yr && host(c2.conv, conv0.size()) == cc1 && host(c2.ssm, ssm0.size()) == cs1;
        HCK(hipMemset(y2, 0, size_t(T) * VD * 2));
        St c3{dev(host(capconv, conv0.size())), dev(ssm0)}; k5::EndState res{}; res.ssm_in32 = cap32;
        core(c3, 64, T - 64, y2, &res);
        auto yb = host(y2, size_t(T) * VD);
        bool cont = host(c3.conv, conv0.size()) == cc1 && host(c3.ssm, ssm0.size()) == cs1;
        for (size_t i = 64 * VD; i < yr.size(); ++i) cont &= yb[i] == yr[i];
        check("EndState (layer_core)", neutral && cont, "capture leaves the call unchanged (%.0f); rows 64..99, conv and ssm resumed from it bit-equal (%.0f)",
              neutral ? 1.0 : 0.0, cont ? 1.0 : 0.0);
    }
    // --- 3b. the same through layer_forward
    {
        k5::EndState end{}; end.ssm_out32 = cap32; end.conv_out = B(capconv); end.row = 64;
        St s3 = fresh(); fwd(s3, CERE_OP_PREFILL, 0, T, &end);
        const bool neutral = host(dout, size_t(T) * HID) == out1 && host(s3.ssm, ssm0.size()) == ssm1 && host(s3.conv, conv0.size()) == conv1;
        HCK(hipMemset(dout, 0, size_t(T) * HID * 2));
        St s4{dev(host(capconv, conv0.size())), dev(ssm0)}; k5::EndState res{}; res.ssm_in32 = cap32;
        fwd(s4, CERE_OP_PREFILL, 64, T - 64, &res);
        auto out4 = host(dout, size_t(T) * HID);
        int64_t diff = 0; for (size_t i = 64 * HID; i < out1.size(); ++i) diff += out4[i] != out1[i];
        std::vector<double> r64(out1.size() - 64 * HID); for (size_t i = 0; i < r64.size(); ++i) r64[i] = bf2f(out1[64 * HID + i]);
        bool cont = diff == 0 && host(s4.ssm, ssm0.size()) == ssm1 && host(s4.conv, conv0.size()) == conv1;
        std::printf("%-24s %s  capture neutral: %s; resumed rows 64..99: %ld of %zu outputs differ (rel-L2 %.2e), conv + ssm %s\n", "EndState (layer_forward)",
                    neutral && cont ? "PASS" : "FAIL", neutral ? "yes" : "NO", (long)diff, r64.size(), rel(r64.data(), out4.data() + 64 * HID, r64.size()),
                    cont ? "bit-equal" : "DIFFER");
        if (!neutral || !cont) ++g_fail;
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
