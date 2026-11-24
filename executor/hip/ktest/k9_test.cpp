// AmdAmd PAR-6 step 5a: K9 QSA (executor/cuda/src/kernels/k9_qsa.cu, hipified: rocWMMA, plain copies, block top-k,
// 32 x 96-key splits for 64 KB of LDS) on gfx1151, against k9_qsa.h:
//  1. qsa_attention (sparse GQA over index rows, -1 padded) vs FP64 on the FP8 cache (rows 1, 5, 16: the split path;
//     40: the rows path) and BF16 rows (rows 1, 40): rel-L2 (limit 8e-3); qsa_attention_heads with rows_heads 0 / 3 /
//     6 / 12 all the same bytes (the header's promise); a row of a 16-row call == that row alone (split-K rows).
//     P4 (use_p4, executor/hip/src/p4_attention_hip.cpp sparse_kernel), rows 40: vs FP64 the same, rows 20..39 alone
//     (cls 40) and a repeat bit-equal to the 40-row call (row-partition invariance).
//  2. qsa_topk on random F32 logits (ties, -0 / +0, -inf past the visible blocks) vs a host port of the selection
//     (fast_topk's order-preserving uint32 keys, equal keys lowest block first, every visible block when <= 512, the
//     4-position expansion in ascending block order, the tail, -1 padding), impl 0 and 1, rows 3 and 40.
//  3. qsa_logits vs FP64 (rel <= 1e-5 per finite logit, -inf exactly where required), impl 0 / 1 / 2 the same bytes.
//  4. qsa_select == qsa_topk over qsa_logits (rows 3, seq_len 5,000).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/k9_qsa.h"

namespace k9 = cere::k9;
using bf16 = __hip_bfloat16;
static int g_fail = 0;
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
#define CCK(x) do { cere_status s_ = (x); if (s_ != CERE_OK) { std::printf("cere %s:%d status %d\n", __FILE__, __LINE__, int(s_)); std::exit(2); } } while (0)
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float e4m3(uint8_t c) { const int s = c >> 7, e = (c >> 3) & 15, m = c & 7; const float v = e == 0 ? std::ldexp(m / 8.f, -6) : std::ldexp(1.f + m / 8.f, e - 7); return s ? -v : v; }
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> host(const T *p, size_t n) { std::vector<T> h(n); HCK(hipDeviceSynchronize()); HCK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
static void check(const char *name, bool ok, const char *msg) { std::printf("%-26s %s  %s\n", name, ok ? "PASS" : "FAIL", msg); if (!ok) ++g_fail; }

constexpr int QH = k9::kQHeads, D = k9::kHeadDim, QW = QH * D, KW = k9::kKvHeads * D, RW = k9::kRowWidth, G = QH / k9::kKvHeads;

// the host selection (qsa_select's contract) of one row from its logits
static void ref_select(const float *lg, int64_t p, int32_t *out) {
    const int64_t vis = (p + 1) / 4;
    std::vector<int64_t> blocks;
    if (vis <= k9::kBlockTopk) { for (int64_t b = 0; b < vis; ++b) blocks.push_back(b); }
    else {
        auto key = [&](int64_t b) { uint32_t u; std::memcpy(&u, &lg[b], 4); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); };
        std::vector<int64_t> idx(vis); for (int64_t b = 0; b < vis; ++b) idx[b] = b;
        std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) { return key(a) > key(b); });
        blocks.assign(idx.begin(), idx.begin() + k9::kBlockTopk); std::sort(blocks.begin(), blocks.end());
    }
    int n = 0;
    for (int64_t b : blocks) for (int i = 0; i < 4; ++i) out[n++] = int32_t(4 * b + i);
    for (int64_t q = vis * 4; q <= p; ++q) out[n++] = int32_t(q);
    while (n < RW) out[n++] = -1;
}

int main() {
    std::mt19937 g(31); std::normal_distribution<float> nd; std::uniform_real_distribution<float> ud;
    const int64_t NK = 6000;  // context rows of K / V
    std::vector<uint8_t> kc(NK * KW), vc(NK * KW);
    for (auto &b : kc) { b = uint8_t(g() % 0x78); if (g() & 1) b |= 0x80; }
    for (auto &b : vc) { b = uint8_t(g() % 0x78); if (g() & 1) b |= 0x80; }
    std::vector<uint16_t> kb(NK * KW), vb(NK * KW); for (auto &x : kb) x = f2bf(nd(g)); for (auto &x : vb) x = f2bf(nd(g));
    auto *dkc = dev(kc), *dvc = dev(vc); auto *dkb = dev(kb), *dvb = dev(vb);
    // ---- 1. qsa_attention
    {
        const int64_t R = 40; std::vector<uint16_t> q(R * QW); for (auto &x : q) x = f2bf(nd(g) * 0.25f);
        std::vector<int32_t> idx(R * RW, -1);
        for (int64_t t = 0; t < R; ++t) {  // 1,200-2,051 valid distinct positions, ascending, then -1
            const int nv = 1200 + int(g() % 852); std::vector<int32_t> pos(NK); for (int i = 0; i < NK; ++i) pos[i] = i;
            std::shuffle(pos.begin(), pos.end(), g); std::sort(pos.begin(), pos.begin() + nv);
            for (int i = 0; i < nv; ++i) idx[t * RW + i] = pos[i];
        }
        idx[3 * RW + 0] = -1; for (int i = 0; i < RW; ++i) idx[7 * RW + i] = -1;  // row 7: no valid entry -> zeros
        auto *dq = dev(q); auto *didx = dev(idx); uint16_t *dout; HCK(hipMalloc(&dout, R * QW * 2));
        const size_t sb = std::max(k9::attention_scratch_bytes(16), k9::attention_scratch_bytes(R)); void *ds; HCK(hipMalloc(&ds, sb + 256));
        auto run = [&](bool fp8, int64_t rows, int64_t r0, int heads, bool p4 = false, int64_t cls = 0) {
            const bf16 *qq = reinterpret_cast<const bf16 *>(dq + r0 * QW);
            const int32_t *ii = didx + r0 * RW;
            CCK(fp8 ? k9::qsa_attention_heads(qq, QW, ii, rows, nullptr, 0, nullptr, 0, dkc, dvc, nullptr, k9::kSoftmaxScale, ds, sb, reinterpret_cast<bf16 *>(dout), QW, nullptr, p4, cls, heads)
                    : k9::qsa_attention_heads(qq, QW, ii, rows, reinterpret_cast<const bf16 *>(dkb), KW, reinterpret_cast<const bf16 *>(dvb), KW, nullptr, nullptr, nullptr, k9::kSoftmaxScale, ds, sb, reinterpret_cast<bf16 *>(dout), QW, nullptr, p4, cls, heads));
            return host(dout, size_t(rows) * QW);
        };
        for (int p4 = 0; p4 < 2; ++p4) for (bool fp8 : {true, false}) for (int64_t rows : p4 ? std::vector<int64_t>{40} : fp8 ? std::vector<int64_t>{1, 5, 16, 40} : std::vector<int64_t>{1, 40}) {
            auto out = run(fp8, rows, 0, k9::kRowsHeads, p4);
            double num = 0, den = 0; int64_t zero_bad = 0;
            for (int64_t t = 0; t < rows; ++t) for (int h = 0; h < QH; h += (rows > 16 ? 5 : 1)) {
                const int kvh = h / G; std::vector<double> s; std::vector<int32_t> js;
                for (int i = 0; i < RW; ++i) { const int32_t j = idx[t * RW + i]; if (j < 0) continue; js.push_back(j);
                    double d = 0; for (int c = 0; c < D; ++c) d += double(bf2f(q[t * QW + h * D + c])) * (fp8 ? e4m3(kc[j * KW + kvh * D + c]) : bf2f(kb[j * KW + kvh * D + c]));
                    s.push_back(d * k9::kSoftmaxScale); }
                if (js.empty()) { for (int c = 0; c < D; ++c) zero_bad += out[t * QW + h * D + c] != 0; continue; }
                const double mx = *std::max_element(s.begin(), s.end()); double l = 0; for (auto &x : s) { x = std::exp(x - mx); l += x; }
                for (int c = 0; c < D; ++c) { double o = 0; for (size_t i = 0; i < js.size(); ++i) o += s[i] * (fp8 ? e4m3(vc[js[i] * KW + kvh * D + c]) : bf2f(vb[js[i] * KW + kvh * D + c]));
                    o /= l; const double e = bf2f(out[t * QW + h * D + c]) - o; num += e * e; den += o * o; }
            }
            const double err = std::sqrt(num / den);
            bool same = true; if (!p4) for (int hh : {0, 3, 6}) same &= run(fp8, rows, 0, hh) == out;
            if (p4) {  // row-partition invariance: rows 20..39 alone (cls 40) == the 40-row call's
                auto part = run(fp8, 20, 20, k9::kRowsHeads, true, 40);
                same = std::equal(part.begin(), part.end(), out.begin() + 20 * QW) && run(fp8, rows, 0, k9::kRowsHeads, true) == out;
            }
            char msg[256]; std::snprintf(msg, sizeof msg, "rows %2ld: rel-L2 %.2e vs FP64 (8e-3), empty row zeros %s, %s %s", (long)rows, err,
                                         zero_bad ? "NO" : "yes", p4 ? "rows 20..39 alone + repeat" : "rows_heads 0/3/6/12", same ? "same bytes" : "DIFFER");
            check(p4 ? (fp8 ? "qsa_attention P4 fp8" : "qsa_attention P4 bf16") : (fp8 ? "qsa_attention fp8" : "qsa_attention bf16"), err <= 8e-3 && !zero_bad && same, msg);
        }
        {   // step 16: P4 on the FP8 cache with scratch decodes the call's K / V once into BF16 copies; capacities
            // covering every position and only some of them (the rest decoded per row) give the scratch-less bytes
            const auto base = run(true, 40, 0, k9::kRowsHeads, true);
            long diff = 0;
            for (int64_t cap : {NK + 7, NK / 2, int64_t(1)}) {
                const size_t bytes = 256 + size_t(cap) * 2 * 512 * 2;
                void *big; HCK(hipMalloc(&big, bytes + 256)); HCK(hipMemset(big, 0xA5, bytes));
                CCK(k9::qsa_attention_heads(reinterpret_cast<const bf16 *>(dq), QW, didx, 40, nullptr, 0, nullptr, 0, dkc, dvc, nullptr,
                                            k9::kSoftmaxScale, big, bytes, reinterpret_cast<bf16 *>(dout), QW, nullptr, true, 0, k9::kRowsHeads));
                diff += host(dout, size_t(40) * QW) != base;
                HCK(hipFree(big));
            }
            char msg[160]; std::snprintf(msg, sizeof msg, "40 rows, BF16 copies of all / half / 1 positions vs none: %ld of 3 differ", diff);
            check("P4 fp8 decoded copy", diff == 0, msg);
        }
        auto full = run(true, 16, 0, k9::kRowsHeads), one = run(true, 1, 9, k9::kRowsHeads);
        const bool eq = std::equal(one.begin(), one.end(), full.begin() + 9 * QW);
        check("split rows invariance", eq, "row 9 of a 16-row FP8 call == row 9 alone, bit for bit");
    }
    // ---- 2. qsa_topk
    {
        const int64_t S = 5000, R = 40, LD = S / 4;
        std::vector<int64_t> pos(R); for (int64_t t = 0; t < R; ++t) pos[t] = t < 4 ? int64_t(100 + 700 * t) : S - R + t;  // some rows with <= 512 visible
        std::vector<float> lg(R * LD);
        for (int64_t t = 0; t < R; ++t) for (int64_t b = 0; b < LD; ++b) {
            float x = std::round(nd(g) * 8.f) / 8.f;  // coarse values: many ties
            if (g() % 50 == 0) x = (g() & 1) ? -0.f : 0.f;
            lg[t * LD + b] = b < (pos[t] + 1) / 4 ? x : -INFINITY;
        }
        auto *dl = dev(lg); auto *dp = dev(pos); int32_t *dout; HCK(hipMalloc(&dout, R * RW * 4));
        std::vector<int32_t> want(R * RW); for (int64_t t = 0; t < R; ++t) ref_select(&lg[t * LD], pos[t], &want[t * RW]);
        for (int64_t rows : {int64_t(3), R}) for (int impl : {0, 1}) {
            CCK(k9::qsa_topk(dl, LD, dp, rows, S, impl, dout, nullptr));
            auto got = host(dout, size_t(rows) * RW);
            int64_t bad = 0; for (size_t i = 0; i < got.size(); ++i) bad += got[i] != want[i];
            char msg[160]; std::snprintf(msg, sizeof msg, "rows %2ld impl %d: %ld of %zu indices differ from the host selection", (long)rows, impl, (long)bad, got.size());
            check("qsa_topk", bad == 0, msg);
        }
    }
    // ---- 3, 4. qsa_logits, qsa_select
    {
        const int64_t S = 5000, R = 3, LD = S / 4, NB = S / 4;
        std::vector<uint16_t> qi(R * 4 * 128), keys(NB * 128); for (auto &x : qi) x = f2bf(nd(g)); for (auto &x : keys) x = f2bf(nd(g));
        auto *dqi = dev(qi), *dkeys = dev(keys); float *dl; HCK(hipMalloc(&dl, R * LD * 4));
        std::vector<int64_t> pos = {2047, 3001, S - 1}; auto *dp = dev(pos);
        std::vector<float> l0;
        bool same = true; double worst = 0; int64_t inf_bad = 0;
        for (int impl : {0, 1, 2}) {
            CCK(k9::qsa_logits(reinterpret_cast<const bf16 *>(dqi), reinterpret_cast<const bf16 *>(dkeys), nullptr, dp, R, S, dl, LD, impl, nullptr));
            auto l = host(dl, size_t(R) * LD);
            if (impl == 0) {
                l0 = l;
                for (int64_t t = 0; t < R; ++t) for (int64_t b = 0; b < LD; ++b) {
                    if (b >= (pos[t] + 1) / 4) { inf_bad += !(std::isinf(l[t * LD + b]) && l[t * LD + b] < 0); continue; }
                    double s = 0, sa = 0;
                    for (int h = 0; h < 4; ++h) { double d = 0, da = 0; for (int c = 0; c < 128; ++c) { const double pr = double(bf2f(qi[(t * 4 + h) * 128 + c])) * bf2f(keys[b * 128 + c]); d += pr; da += std::fabs(pr); }
                        if (d > 0) s += d; sa += da; }
                    worst = std::max(worst, std::fabs(l[t * LD + b] - s / k9::kScoreDivisor) / (sa / k9::kScoreDivisor + 1e-30));
                }
            } else same &= l == l0;
        }
        char msg[200]; std::snprintf(msg, sizeof msg, "3 rows x 1,250 blocks: worst |err| / sum|terms| %.1e (1e-5), -inf past visible %s, impl 0/1/2 %s", worst, inf_bad ? "NO" : "yes", same ? "same bytes" : "DIFFER");
        check("qsa_logits", worst <= 1e-5 && !inf_bad && same, msg);
        const size_t sb = k9::select_scratch_bytes(R, S); void *ds; HCK(hipMalloc(&ds, sb + 256)); int32_t *di, *dt; HCK(hipMalloc(&di, R * RW * 4)); HCK(hipMalloc(&dt, R * RW * 4));
        CCK(k9::qsa_select(reinterpret_cast<const bf16 *>(dqi), reinterpret_cast<const bf16 *>(dkeys), nullptr, dp, R, S, ds, sb, di, nullptr));
        CCK(k9::qsa_topk(dl, LD, dp, R, S, 1, dt, nullptr));  // dl holds impl 2's logits == impl 0's
        const bool eq = host(di, R * RW) == host(dt, R * RW);
        check("qsa_select", eq, "== qsa_topk(qsa_logits) for 3 rows at seq_len 5,000");
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
