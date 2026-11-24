// AmdAmd PAR-6 step 2l: the K4 PLE forward (executor/cuda/src/kernels/k4_ple.cu, hipified, K2 via the HIP adapter)
// on gfx1151 against SGLang's Qwen4ExpPLELayer.forward (qwen4_exp.py:1162-1218, _short_conv :955) in FP64 from the
// kernel's own BF16 embedding: key/value projections, grouped Gemma RMSNorm (group 2,560) of key and the HC query,
// gate = sigmoid(signed_sqrt(sum(kn*qn)/sqrt(2560))), gated = gate*value, grouped norm, dilated depthwise conv
// (kernel 4, dilation 3, over 9 state columns + the step) + SiLU, delta = gated + conv. Checks: delta rel-L2 vs FP64
// (limit 2e-2: the chain rounds to BF16 at about ten points the reference does not), state continuity (one 9-token
// call == 5 + 4 tokens, bit for bit, with plans.cls = 16 so both pick the same GEMM bucket), conv state = the last 9
// gated-normed columns (via continuity), add_query (out = bf16(query + delta)) bit-equal to delta then the add.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k4_ple.h"

namespace k4 = cere::k4;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static const int64_t kMult[3] = {23703573157769LL, 20109073645365LL, 8052911324071LL};
constexpr int64_t kEos = 248044, H = k4::kHidden, C = k4::kChannels, E = k4::kEmbedDim;

int main() {
    std::mt19937_64 g(41); std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; };
    const int64_t rows = 16 * 1000; std::vector<uint8_t> table(rows * k4::kHeadDim);
    for (auto &b : table) { b = uint8_t(g() % 0x70); if (g() & 1) b |= 0x80; }  // small-magnitude finite E4M3
    std::vector<int64_t> vocab(16, 1000), offs(16); for (int h = 0; h < 16; ++h) offs[h] = h * 1000;
    std::vector<int64_t> mult(kMult, kMult + 3);
    auto key_proj = rnd(size_t(C) * E, 0.02f), value_proj = rnd(size_t(H) * E, 0.02f);
    auto norm_key = rnd(C, 0.1f), norm_query = rnd(C, 0.1f), norm_conv = rnd(C, 0.1f), conv_w = rnd(size_t(C) * 4, 0.3f);
    const uint16_t scale = f2bf(1.0f);
    k4::Weights w{}; w.table = dev(table); w.table_rows = rows; w.table_scale = reinterpret_cast<const __hip_bfloat16 *>(dev(std::vector<uint16_t>{scale}));
    w.multipliers = dev(mult); w.head_vocab_sizes = dev(vocab); w.head_offsets = dev(offs);
    w.key_proj = reinterpret_cast<const __hip_bfloat16 *>(dev(key_proj)); w.value_proj = reinterpret_cast<const __hip_bfloat16 *>(dev(value_proj));
    w.norm_key = reinterpret_cast<const __hip_bfloat16 *>(dev(norm_key)); w.norm_query = reinterpret_cast<const __hip_bfloat16 *>(dev(norm_query));
    w.norm_conv = reinterpret_cast<const __hip_bfloat16 *>(dev(norm_conv)); w.conv_weight = reinterpret_cast<const __hip_bfloat16 *>(dev(conv_w));
    w.eos_token_id = kEos; w.eps = 1e-6f; w.format = k4::TableFormat::kFp8;
    hipblasLtHandle_t lt; if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) return 2;
    cere::kernels::GemmPlan pk, pv;
    if (pk.init(lt, C, E, 16, cere::kernels::GemmOut::kBF16) != CERE_OK || pv.init(lt, H, E, 16, cere::kernels::GemmOut::kBF16) != CERE_OK) { std::printf("plan init failed\n"); return 2; }
    k4::Plans plans{&pk, &pv, 16};
    const int64_t T = 9; std::vector<int64_t> tokens(T); for (auto &x : tokens) x = int64_t(g() % 248320);
    auto query = rnd(size_t(T) * C, 1.f);
    const std::vector<int64_t> hist0 = {kEos, kEos}; const std::vector<uint16_t> conv0(size_t(C) * 9, 0);
    const size_t ws = k4::workspace_bytes(T); void *dws; HCK(hipMalloc(&dws, ws + 256));
    auto *dtok = dev(tokens); auto *dq = dev(query); uint16_t *dout; HCK(hipMalloc(&dout, size_t(T) * C * 2));
    auto step = [&](int64_t *dh, uint16_t *dcs, int64_t t0, int64_t t, bool add) {
        k4::State st{dh, reinterpret_cast<__hip_bfloat16 *>(dcs)};
        const cere_status s = k4::forward(w, plans, st, dtok + t0, t, reinterpret_cast<const __hip_bfloat16 *>(dq + t0 * C), add,
                                          reinterpret_cast<__hip_bfloat16 *>(dout + t0 * C), dws, ws, nullptr);
        if (s != CERE_OK) { std::printf("forward failed: %d\n", s); std::exit(2); }
    };
    // one call, delta only
    auto *dh1 = dev(hist0); auto *dc1 = dev(conv0); step(dh1, dc1, 0, T, false);
    auto delta = host(dout, size_t(T) * C); auto cs1 = host(dc1, size_t(C) * 9); auto h1 = host(dh1, 2);
    // the same split 5 + 4
    auto *dh2 = dev(hist0); auto *dc2 = dev(conv0); step(dh2, dc2, 0, 5, false); step(dh2, dc2, 5, T - 5, false);
    auto delta2 = host(dout, size_t(T) * C); auto cs2 = host(dc2, size_t(C) * 9); auto h2 = host(dh2, 2);
    const bool split_ok = delta2 == delta && cs2 == cs1 && h2 == h1 && h1[0] == tokens[T - 2] && h1[1] == tokens[T - 1];
    std::printf("%-22s %s  9 tokens in one call == 5 + 4: delta, conv state, history %s\n", "state continuity", split_ok ? "PASS" : "FAIL", split_ok ? "bit-equal" : "DIFFER");
    if (!split_ok) ++g_fail;
    // add_query: out = bf16(query + delta)
    auto *dh3 = dev(hist0); auto *dc3 = dev(conv0); step(dh3, dc3, 0, T, true); auto outq = host(dout, size_t(T) * C);
    int64_t addbad = 0; for (size_t i = 0; i < outq.size(); ++i) addbad += outq[i] != f2bf(bf2f(query[i]) + bf2f(delta[i]));
    std::printf("%-22s %s  out == bf16(query + delta): %ld of %ld differ\n", "add_query", addbad ? "FAIL" : "PASS", (long)addbad, (long)outq.size());
    if (addbad) ++g_fail;
    // FP64 reference from the kernel's own embedding
    uint16_t *demb; HCK(hipMalloc(&demb, size_t(T) * E * 2));
    if (k4::embed(w, dev(hist0), dtok, T, reinterpret_cast<__hip_bfloat16 *>(demb), nullptr) != CERE_OK) return 2;
    auto emb = host(demb, size_t(T) * E);
    std::vector<double> gn_hist(size_t(C) * (9 + T), 0.0);  // conv input over time per channel: 9 zero state + T
    std::vector<double> gated(size_t(T) * C);
    auto gnorm = [&](const double *x, const std::vector<uint16_t> &wv, int grp, double *y) {
        double ss = 0; for (int64_t i = 0; i < H; ++i) ss += x[i] * x[i];
        const double rs = 1 / std::sqrt(ss / H + 1e-6); for (int64_t i = 0; i < H; ++i) y[i] = x[i] * rs * (1 + double(bf2f(wv[grp * H + i])));
    };
    for (int64_t t = 0; t < T; ++t) {
        std::vector<double> key(C), value(H), kn(C), qn(C), q(C);
        for (int64_t o = 0; o < C; ++o) { double a = 0; for (int64_t i = 0; i < E; ++i) a += double(bf2f(emb[t * E + i])) * bf2f(key_proj[o * E + i]); key[o] = a; }
        for (int64_t o = 0; o < H; ++o) { double a = 0; for (int64_t i = 0; i < E; ++i) a += double(bf2f(emb[t * E + i])) * bf2f(value_proj[o * E + i]); value[o] = a; }
        for (int64_t i = 0; i < C; ++i) q[i] = bf2f(query[t * C + i]);
        for (int grp = 0; grp < 4; ++grp) {
            gnorm(&key[grp * H], norm_key, grp, &kn[grp * H]); gnorm(&q[grp * H], norm_query, grp, &qn[grp * H]);
            double dot = 0; for (int64_t i = 0; i < H; ++i) dot += kn[grp * H + i] * qn[grp * H + i];
            double gte = dot / std::sqrt(double(H)); gte = std::sqrt(std::fmax(std::fabs(gte), 1e-6)) * (gte > 0 ? 1 : gte < 0 ? -1 : 0);
            gte = 1 / (1 + std::exp(-gte));
            for (int64_t i = 0; i < H; ++i) gated[t * C + grp * H + i] = gte * value[i];
            std::vector<double> gnv(H); gnorm(&gated[t * C + grp * H], norm_conv, grp, gnv.data());
            for (int64_t i = 0; i < H; ++i) gn_hist[size_t(grp * H + i) * (9 + T) + 9 + t] = gnv[i];
        }
    }
    double num = 0, den = 0;
    for (int64_t t = 0; t < T; ++t) for (int64_t c = 0; c < C; ++c) {
        double acc = 0; for (int j = 0; j < 4; ++j) acc += double(bf2f(conv_w[c * 4 + j])) * gn_hist[size_t(c) * (9 + T) + 9 + t - 9 + j * 3];
        const double conv = acc / (1 + std::exp(-acc)), ref = gated[t * C + c] + conv, got = bf2f(delta[t * C + c]);
        num += (got - ref) * (got - ref); den += ref * ref;
    }
    const double e = std::sqrt(num / den);
    std::printf("%-22s %s  delta rel-L2 %.2e vs FP64 (limit 2e-2), 9 tokens x 10,240 channels\n", "forward vs FP64", e <= 2e-2 ? "PASS" : "FAIL", e);
    if (e > 2e-2) ++g_fail;
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
