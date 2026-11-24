// VulkanAmd P2: executor/hip/ktest/k6_test.cpp's contract rows on the Vulkan K6 dense causal GQA
// (src/kernels/k6_attention_vk.*: calls of <= 16 rows on the split-key kernel + combine, longer ones on the FP32
// prefill tile kernel), against k6::attention's contract:
// out[t, h] = softmax_j(q[t, h] . K[j] * scale) V[j], j = 0 .. pos0 + t, KV head h / 12, FP64 reference.
// The HIP test's rows, data (seed 29, same generation order) and bounds:
//  1. BF16 K/V rows and the FP8 E4M3 cache (scale 1.0; reference from the decoded bytes) at decode (1 token at pos0
//     100 and 1,500), a short extend (7 at 50), and prefill-size calls (64 at 0, 300 at 0, 2,048 at 0): rel-L2 vs FP64
//     (limit 8e-3), repeat bit-identical;
//  2. the class pin: rows 128..299 of a 300-row call (pos0 0) == a 172-row call at pos0 128 with cls 300, bit for bit;
//  3. the verify chain (dense_attention_rows, 5 rows at 1,700, FP8, gated) == the rows' one-token dense_attention calls;
//  4. timing (reported): prefill 2,048 / 512 rows, decode 1 row at 2,047 keys, a 4-row extend at 2,040; plus verify
//     chains of 4 / 8 / 16 rows (the HIP test does not time those; amd-amd step 10 did).
// Vulkan-only rows (the kernels claim more than the HIP test asserts, so the claim is tested):
//  5. row-partition invariance of both paths (a 16-row split call == its rows in two calls; a 300-row prefill call ==
//     100 + 200 rows), a 16-row chain at 2,032 == its one-token calls;
//  6. negative rows: arguments outside the contract are refused (CERE_ERR_ARGUMENT) before anything is recorded.
// P4-A rows (the cooperative-matrix prefill path, Options::prefill = kCoopmat; HIP's bounds, Vulkan rows):
//  7. vs FP64 (rel-L2 <= 8e-3) + repeat bit-equal: BF16 / FP8 at T 17 / 64 / 300 / 2,048 at 0, 100 at 1,000, 1,000 at
//     1,048; the max difference to the FP32 path is printed (the numerics change);
//  8. class pin and partitions (300 = 100 + 200 = 17 + 283 = 18 calls of 17) on the coopmat path, bit-exact; K / V past
//     row 200 replaced: rows < 200 bit-exact; dense_attention (gated: BF16 rows at 0, the FP8 cache at 700) vs FP64;
//  9. decode / split shapes (T 1, 7, 16 and the verify chain) are bit-identical with and without the option;
// 10. operand ranges: q rows x 2^90 with keys x 2^-100 (BF16 exponents far outside FP16: per-row scaling), V x 2^-24
//     (FP16-subnormal V: per-key scaling), V x 2^10 above key 100, V outliers 1e5 at one key (beyond FP16: exact fallback),
//     +inf in K at one key, NaN in one q row, an FP8 NaN byte in V: vs FP64 where finite, flagged rows bit-equal to
//     the FP32 path, unflagged rows within the bound;
// 11. negative rows: no / short workspace, an unknown Prefill value, pos0 + tokens > 2,048 on the coopmat path;
// 12. the device probe (FP16 subnormals in conversions and in the matrix unit) and timing: coopmat vs FP32 on the
//     timed prefill shapes plus T 1,000 at 1,048 (FP8: a second chunk) and T 1,024 at 0.
// Timing method: 3 untimed warm-up calls, then 15 calls each between its own pair of GPU timestamps (a barrier after
// each call: calls serialized as a HIP stream serializes them); median / min / max.
// Usage: k6_test <spv dir> [keys_per_split 64|128] [subgroup 32|64] [--time-only] [--p4-only]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "launch/vk_launch_testing.h"
#include "kernels/k6_attention_vk.h"
#include "ktest_util.h"

namespace k6 = cere::vkk6;
using kt::bf2f;
using kt::f2bf;
static int g_fail = 0;
static float e4m3(uint8_t c) {
    const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    const float v = e == 0 ? std::ldexp(m / 8.f, -6) : std::ldexp(1.f + m / 8.f, e - 7);
    return s ? -v : v;
}

constexpr int QH = k6::kQHeads, KVH = k6::kKvHeads, D = k6::kHeadDim, QW = QH * D, KW = KVH * D;

int main(int argc, char **argv) {
    auto ctx_owner = cere::vk::testing::create_or_die(argc > 1 ? argv[1] : "build/vulkan-k6/spv");
    cere::vk::Context &ctx = *ctx_owner;
    k6::Options opt;
    bool time_only = false, p4_only = false;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--time-only") == 0) time_only = true;
        else if (std::strcmp(argv[i], "--p4-only") == 0) p4_only = true;
        else if (i == 2) opt.keys_per_split = uint32_t(std::atoi(argv[i]));
        else if (i == 3) opt.subgroup = uint32_t(std::atoi(argv[i]));
    }
    std::printf("device: %s, driver %s %s; K6 options: keys_per_split %u, subgroup %u\n", ctx.info().name.c_str(),
                ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str(), opt.keys_per_split, opt.subgroup);

    std::mt19937 g(29); std::normal_distribution<float> nd;
    const int64_t NK = k6::kMaxContext;  // keys available
    std::vector<uint16_t> kb(NK * KW), vb(NK * KW);
    for (auto &x : kb) x = f2bf(nd(g));
    for (auto &x : vb) x = f2bf(nd(g));
    std::vector<uint8_t> kc(NK * KW), vc(NK * KW);
    for (auto &b : kc) { b = uint8_t(g() % 0x78); if (g() & 1) b |= 0x80; }  // finite E4M3 up to 240
    for (auto &b : vc) { b = uint8_t(g() % 0x78); if (g() & 1) b |= 0x80; }
    auto dk = kt::upload(ctx, kb), dv = kt::upload(ctx, vb), dkc = kt::upload(ctx, kc), dvc = kt::upload(ctx, vc);
    const int64_t TMAX = 2048;
    std::vector<uint16_t> q(TMAX * QW); for (auto &x : q) x = f2bf(nd(g) * 0.25f);
    auto dq = kt::upload(ctx, q);
    auto dout = kt::make(ctx, TMAX * QW * 2);
    k6::Options wopt = opt; wopt.prefill = k6::Prefill::kCoopmat;
    const size_t wsb = std::max(k6::workspace_bytes(TMAX, wopt), k6::workspace_bytes(16, wopt));
    auto dws = kt::make(ctx, wsb);
    const float scale = k6::kSoftmaxScale;

    auto rec = [&](bool fp8, int64_t T, int64_t pos0, int64_t qrow0, int64_t cls) {
        const uint64_t qq = dq.addr + uint64_t(qrow0) * QW * 2;
        return fp8 ? k6::attention(ctx, qq, T, pos0, 0, 0, 0, 0, dkc.addr, dvc.addr, 0, scale, dws.addr, wsb, dout.addr, cls, opt)
                   : k6::attention(ctx, qq, T, pos0, dk.addr, KW, dv.addr, KW, 0, 0, 0, scale, dws.addr, wsb, dout.addr, cls, opt);
    };
    auto rec2 = [&](const k6::Options &o, bool fp8, int64_t T, int64_t pos0) {
        return fp8 ? k6::attention(ctx, dq.addr, T, pos0, 0, 0, 0, 0, dkc.addr, dvc.addr, 0, scale, dws.addr, wsb, dout.addr, 0, o)
                   : k6::attention(ctx, dq.addr, T, pos0, dk.addr, KW, dv.addr, KW, 0, 0, 0, scale, dws.addr, wsb, dout.addr, 0, o);
    };
    auto run = [&](bool fp8, int64_t T, int64_t pos0, int64_t qrow0, int64_t cls = 0) {
        ctx.begin();
        const cere_status e = rec(fp8, T, pos0, qrow0, cls);
        if (e != CERE_OK) { std::printf("attention failed: %d (T %ld pos0 %ld)\n", int(e), (long)T, (long)pos0); std::exit(2); }
        ctx.barrier();
        ctx.submit_wait();
        return kt::host<uint16_t>(dout, size_t(T) * QW);
    };
    auto kval = [&](bool fp8, int64_t j, int c) { return fp8 ? double(e4m3(kc[j * KW + c])) : double(bf2f(kb[j * KW + c])); };
    auto vval = [&](bool fp8, int64_t j, int c) { return fp8 ? double(e4m3(vc[j * KW + c])) : double(bf2f(vb[j * KW + c])); };

    k6::Options copt = opt;
    copt.prefill = k6::Prefill::kCoopmat;
    if (!time_only && !p4_only) {
        struct C { int64_t T, pos0; } cases[] = {{1, 100}, {1, 1500}, {7, 50}, {64, 0}, {300, 0}, {2048, 0}};
        for (bool fp8 : {false, true}) for (auto cs : cases) {
            auto out = run(fp8, cs.T, cs.pos0, 0);
            double num = 0, den = 0;
            const int64_t step = cs.T > 256 ? 37 : 1;  // sample rows of the big calls (FP64 reference cost)
            for (int64_t t = 0; t < cs.T; t += step) for (int h = 0; h < QH; ++h) {
                const int kvh = h / (QH / KVH); const int64_t nk = cs.pos0 + t + 1;
                std::vector<double> s(nk); double mx = -1e300;
                for (int64_t j = 0; j < nk; ++j) { double d = 0; for (int c = 0; c < D; ++c) d += double(bf2f(q[t * QW + h * D + c])) * kval(fp8, j, kvh * D + c); s[j] = d * scale; mx = std::max(mx, s[j]); }
                double l = 0; for (auto &x : s) { x = std::exp(x - mx); l += x; }
                for (int c = 0; c < D; ++c) {
                    double o = 0; for (int64_t j = 0; j < nk; ++j) o += s[j] * vval(fp8, j, kvh * D + c);
                    o /= l; const double e = bf2f(out[t * QW + h * D + c]) - o; num += e * e; den += o * o;
                }
            }
            const double err = std::sqrt(num / den);
            auto again = run(fp8, cs.T, cs.pos0, 0);
            const bool ok = err <= 8e-3 && again == out;
            std::printf("%-6s T %4ld pos0 %4ld   %s  rel-L2 %.2e vs FP64 (limit 8e-3), repeat %s  [%s]\n", fp8 ? "fp8" : "bf16", (long)cs.T, (long)cs.pos0, ok ? "PASS" : "FAIL", err, again == out ? "bit-equal" : "DIFFERS",
                        cs.T > k6::kSplitMaxTokens ? "prefill FP32" : "split");
            if (!ok) ++g_fail;
        }
        for (bool fp8 : {false, true}) {  // class pin
            auto full = run(fp8, 300, 0, 0);
            auto part = run(fp8, 172, 128, 128, 300);
            int64_t diff = 0; for (size_t i = 0; i < part.size(); ++i) diff += part[i] != full[128 * QW + i];
            std::printf("%-6s class pin   %s  rows 128..299 at pos0 128 (cls 300) vs the 300-row call: %ld of %zu differ\n", fp8 ? "fp8" : "bf16", diff ? "FAIL" : "PASS", (long)diff, part.size());
            if (diff) ++g_fail;
        }
        // the verify chain vs one-token calls (the contract: row t == dense_attention at prefix pos0 + t)
        auto chain_check = [&](int64_t R, int64_t pos0, const char *label) {
            const size_t g_sz = size_t(R) * QW;
            std::vector<uint16_t> gate(g_sz); for (auto &x : gate) x = f2bf(nd(g));
            auto dgate = kt::upload(ctx, gate);
            auto dch = kt::make(ctx, g_sz * 2);
            ctx.begin();
            kt::must(k6::dense_attention_rows(ctx, dq.addr, dgate.addr, R, pos0, dkc.addr, dvc.addr, 0, dws.addr, wsb, dch.addr, QW, opt), "dense_attention_rows");
            ctx.barrier(); ctx.submit_wait();
            auto chain = kt::host<uint16_t>(dch, g_sz); int64_t diff = 0;
            for (int64_t t = 0; t < R; ++t) {
                ctx.begin();
                kt::must(k6::dense_attention(ctx, dq.addr + uint64_t(t) * QW * 2, dk.addr, dv.addr, KW, dgate.addr + uint64_t(t) * QW * 2, 1, pos0 + t, true,
                                             dkc.addr, dvc.addr, 0, dws.addr, wsb, dout.addr, QW, 0, opt), "dense_attention");
                ctx.barrier(); ctx.submit_wait();
                auto one = kt::host<uint16_t>(dout, QW); for (int c = 0; c < QW; ++c) diff += one[c] != chain[t * QW + c];
            }
            std::printf("%-6s chain rows   %s  dense_attention_rows (%ld rows at %ld) vs one-token dense_attention: %ld of %zu differ%s\n", "fp8", diff ? "FAIL" : "PASS",
                        (long)R, (long)pos0, (long)diff, g_sz, label);
            if (diff) ++g_fail;
            ctx.free(dgate); ctx.free(dch);
        };
        chain_check(5, 1700, "");
        // --- Vulkan-only rows ---
        chain_check(16, 2032, "  [extra]");
        for (bool fp8 : {false, true}) {  // row-partition invariance of the split path and of the prefill path
            struct P { int64_t T, pos0, cut; } ps[] = {{16, 1000, 7}, {300, 0, 100}};
            for (auto p : ps) {
                auto full = run(fp8, p.T, p.pos0, 0);
                auto a = run(fp8, p.cut, p.pos0, 0);
                auto b = run(fp8, p.T - p.cut, p.pos0 + p.cut, p.cut);
                int64_t diff = 0;
                for (size_t i = 0; i < a.size(); ++i) diff += a[i] != full[i];
                for (size_t i = 0; i < b.size(); ++i) diff += b[i] != full[size_t(p.cut) * QW + i];
                const bool same_path = (p.cut > k6::kSplitMaxTokens) == (p.T > k6::kSplitMaxTokens) && (p.T - p.cut > k6::kSplitMaxTokens) == (p.T > k6::kSplitMaxTokens);
                std::printf("%-6s partition   %s  %ld rows at %ld vs %ld + %ld rows (%s path%s): %ld of %zu differ  [extra]\n", fp8 ? "fp8" : "bf16", diff ? "FAIL" : "PASS",
                            (long)p.T, (long)p.pos0, (long)p.cut, (long)(p.T - p.cut), p.T > k6::kSplitMaxTokens ? "prefill" : "split",
                            same_path ? "" : ", NOT the same path", (long)diff, full.size());
                if (diff || !same_path) ++g_fail;
            }
        }
        {   // negative rows: refused before recording
            struct N { const char *what; int64_t T, pos0, cls; bool both; } ns[] = {
                {"cls < tokens", 172, 128, 100, false}, {"pos0 + tokens > 2048", 4, 2045, 0, false}, {"BF16 and FP8 sources both set", 1, 10, 0, true},
                {"negative pos0", 1, -1, 0, false}};
            for (auto n : ns) {
                ctx.begin();
                const cere_status e = n.both ? k6::attention(ctx, dq.addr, n.T, n.pos0, dk.addr, KW, dv.addr, KW, dkc.addr, dvc.addr, 0, scale, dws.addr, wsb, dout.addr, n.cls, opt)
                                             : k6::attention(ctx, dq.addr, n.T, n.pos0, 0, 0, 0, 0, dkc.addr, dvc.addr, 0, scale, dws.addr, wsb, dout.addr, n.cls, opt);
                ctx.submit_wait();
                const bool ok = e == CERE_ERR_ARGUMENT;
                std::printf("negative     %s  %s -> %s  [extra]\n", ok ? "PASS" : "FAIL", n.what, ok ? "CERE_ERR_ARGUMENT" : "accepted");
                if (!ok) ++g_fail;
            }
            ctx.begin();
            const cere_status e = k6::dense_attention_rows(ctx, dq.addr, dq.addr, 17, 100, dkc.addr, dvc.addr, 0, dws.addr, wsb, dout.addr, QW, opt);
            ctx.submit_wait();
            std::printf("negative     %s  dense_attention_rows with 17 rows -> %s  [extra]\n", e == CERE_ERR_ARGUMENT ? "PASS" : "FAIL", e == CERE_ERR_ARGUMENT ? "CERE_ERR_ARGUMENT" : "accepted");
            if (e != CERE_ERR_ARGUMENT) ++g_fail;
        }
    }

    // ===== P4-A: the cooperative-matrix prefill path =====
    if (!time_only) {
        uint32_t pr[4] = {0, 0, 0, 0};
        const cere_status ps = k6::coopmat_probe(ctx, pr);
        float f1, f2; std::memcpy(&f1, &pr[1], 4); std::memcpy(&f2, &pr[2], 4);
        std::printf("p4 probe     %s  f16(2^-20) bits 0x%04x (%s); coopmat 16 x 2^-20 = %g (%s), 16 x 2^-14 = %g; accumulator length %u  [info]\n",
                    ps == CERE_OK ? "INFO" : "FAIL", pr[0], pr[0] == 0x10 ? "subnormal kept" : "flushed", double(f1),
                    f1 == std::ldexp(1.f, -16) ? "subnormal operands kept" : "subnormal operands FLUSHED", double(f2), pr[3]);
        if (ps != CERE_OK) ++g_fail;
        {   // [info] the matrix unit and zero products (p4att_probe2.comp): D = A B + C with A[r][k] = 0 for k >= n_r, B1 vs
            // B2 differing only in rows k >= 3 (B2 zero there)
            const int W = 4096;
            std::mt19937 g3(11); std::uniform_real_distribution<float> ud(-1.f, 1.f);
            auto h16 = [](float f) {  // float -> FP16 bits (RNE; |f| in the normal range here)
                uint32_t u; std::memcpy(&u, &f, 4);
                const uint32_t s = (u >> 16) & 0x8000u; const int e = int((u >> 23) & 255) - 127 + 15; uint32_t m = u & 0x7FFFFFu;
                if (f == 0.f) return uint16_t(s);
                uint32_t r = (uint32_t(e) << 10) | (m >> 13); const uint32_t rem = m & 0x1FFFu;
                if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;
                return uint16_t(s | r);
            };
            std::vector<uint16_t> A(W * 256), B1(W * 256), B2(W * 256); std::vector<float> Cm(W * 256);
            for (int w = 0; w < W; ++w) for (int r = 0; r < 16; ++r) {
                const int n = (r + w) % 5 == 4 ? 0 : 3 + (r + w) % 13;  // some rows all zero
                for (int k = 0; k < 16; ++k) {
                    A[w * 256 + r * 16 + k] = k < n ? h16(std::ldexp(std::fabs(ud(g3)), 6 + int(g3() % 9))) : 0;  // p'-like
                    const uint16_t b = h16(ud(g3) * 16384.f);  // V'-like
                    B1[w * 256 + k * 16 + r] = b;
                    B2[w * 256 + k * 16 + r] = k < 3 ? b : 0;
                }
                for (int c = 0; c < 16; ++c) Cm[w * 256 + r * 16 + c] = ud(g3) * 1e6f;
            }
            // B2 must equal B1 wherever A is non-zero for some row: make every A column k >= 3 meet B rows that differ
            auto da = kt::upload(ctx, A), db1 = kt::upload(ctx, B1), db2 = kt::upload(ctx, B2), dc = kt::upload(ctx, Cm);
            auto dd1 = kt::make(ctx, W * 1024), dd2 = kt::make(ctx, W * 1024);
            struct { uint64_t a, b1, b2, c, d1, d2; } pc{da.addr, db1.addr, db2.addr, dc.addr, dd1.addr, dd2.addr};
            const cere::vk::Pipeline *pp = nullptr;
            kt::must(ctx.pipeline(cere::vk::PipelineSpec{"p4att_probe2.spv", sizeof(pc), 32, {}}, &pp), "probe2 pipeline");
            ctx.begin(); ctx.dispatch(*pp, &pc, W); ctx.barrier(); ctx.submit_wait();
            auto d1 = kt::host<float>(dd1, W * 256), d2 = kt::host<float>(dd2, W * 256);
            // only rows whose A is zero for k >= 3 (n_r == 3) have equal exact sums; compare every row's exact sum instead:
            int64_t bad1 = 0, cmp = 0, bad0 = 0, cmp0 = 0;
            for (int w = 0; w < W; ++w) for (int r = 0; r < 16; ++r) {
                const int n = (r + w) % 5 == 4 ? 0 : 3 + (r + w) % 13;
                for (int c = 0; c < 16; ++c) {
                    const size_t i = size_t(w) * 256 + r * 16 + c;
                    if (n == 3) { ++cmp; bad1 += std::memcmp(&d1[i], &d2[i], 4) != 0; }
                    if (n == 0) { ++cmp0; bad0 += std::memcmp(&d1[i], &Cm[i], 4) != 0 || std::memcmp(&d2[i], &Cm[i], 4) != 0; }
                }
            }
            std::printf("p4 probe2    INFO  coopMatMulAdd: A zero past column 3, B rows 3..15 random vs zero: %ld of %ld results change; A row all zero: %ld of %ld results != C  [info]\n",
                        (long)bad1, (long)cmp, (long)bad0, (long)cmp0);
            ctx.free(da); ctx.free(db1); ctx.free(db2); ctx.free(dc); ctx.free(dd1); ctx.free(dd2);
        }
        // generic run on any K/V source; o: coopmat or FP32 options
        auto runo = [&](const k6::Options &o, bool fp8, uint64_t qa, uint64_t ka, uint64_t va, int64_t T, int64_t pos0, int64_t qrow0, int64_t cls = 0) {
            const uint64_t qq = qa + uint64_t(qrow0) * QW * 2;
            std::memset(dout.map, 0x5A, size_t(T) * QW * 2);
            ctx.begin();
            const cere_status e = fp8 ? k6::attention(ctx, qq, T, pos0, 0, 0, 0, 0, ka, va, 0, scale, dws.addr, wsb, dout.addr, cls, o)
                                      : k6::attention(ctx, qq, T, pos0, ka, KW, va, KW, 0, 0, 0, scale, dws.addr, wsb, dout.addr, cls, o);
            if (e != CERE_OK) { std::printf("attention failed: %d (T %ld pos0 %ld)\n", int(e), (long)T, (long)pos0); std::exit(2); }
            ctx.barrier();
            ctx.submit_wait();
            return kt::host<uint16_t>(dout, size_t(T) * QW);
        };
        // FP64 reference rel-L2 over sampled rows (every `step`-th) whose reference is finite; skip(t, h) excludes rows
        auto ref_err = [&](const std::vector<uint16_t> &qh, auto kv, auto vv, int64_t T, int64_t pos0, const std::vector<uint16_t> &out,
                           int64_t step, auto skip) {
            double num = 0, den = 0;
            for (int64_t t = 0; t < T; t += step) for (int h = 0; h < QH; ++h) {
                if (skip(t, h)) continue;
                const int kvh = h / (QH / KVH); const int64_t nk = pos0 + t + 1;
                std::vector<double> sc(nk); double mx = -1e300;
                for (int64_t j = 0; j < nk; ++j) { double d = 0; for (int c = 0; c < D; ++c) d += double(bf2f(qh[t * QW + h * D + c])) * kv(j, kvh * D + c); sc[j] = d * scale; mx = std::max(mx, sc[j]); }
                double l = 0; for (auto &x : sc) { x = std::exp(x - mx); l += x; }
                for (int c = 0; c < D; ++c) {
                    double o = 0; for (int64_t j = 0; j < nk; ++j) o += sc[j] * vv(j, kvh * D + c);
                    o /= l; const double e = bf2f(out[t * QW + h * D + c]) - o; num += e * e; den += o * o;
                }
            }
            return std::sqrt(num / den);
        };
        auto noskip = [](int64_t, int) { return false; };
        auto kf = [&](bool fp8) { return [&, fp8](int64_t j, int c) { return kval(fp8, j, c); }; };
        auto vf = [&](bool fp8) { return [&, fp8](int64_t j, int c) { return vval(fp8, j, c); }; };
        // 7. accuracy + repeat
        struct C { int64_t T, pos0; } cases[] = {{17, 0}, {64, 0}, {300, 0}, {2048, 0}, {100, 1000}, {1000, 1048}};
        for (bool fp8 : {false, true}) for (auto cs : cases) {
            const uint64_t ka = fp8 ? dkc.addr : dk.addr, va = fp8 ? dvc.addr : dv.addr;
            auto out = runo(copt, fp8, dq.addr, ka, va, cs.T, cs.pos0, 0);
            const double err = ref_err(q, kf(fp8), vf(fp8), cs.T, cs.pos0, out, cs.T > 256 ? 37 : 1, noskip);
            auto again = runo(copt, fp8, dq.addr, ka, va, cs.T, cs.pos0, 0);
            auto f32 = runo(opt, fp8, dq.addr, ka, va, cs.T, cs.pos0, 0);
            double md = 0; int64_t nd_ = 0;
            for (size_t i = 0; i < out.size(); ++i) { const double d = std::fabs(double(bf2f(out[i])) - bf2f(f32[i])); md = std::max(md, d); nd_ += out[i] != f32[i]; }
            const bool ok = err <= 8e-3 && again == out;
            std::printf("%-6s p4 T %4ld pos0 %4ld   %s  rel-L2 %.2e vs FP64 (limit 8e-3), repeat %s; vs FP32 path: %ld of %zu differ, max |diff| %.3g  [coopmat]\n",
                        fp8 ? "fp8" : "bf16", (long)cs.T, (long)cs.pos0, ok ? "PASS" : "FAIL", err, again == out ? "bit-equal" : "DIFFERS", (long)nd_, out.size(), md);
            if (!ok) ++g_fail;
        }
        // 8. class pin + partitions
        for (bool fp8 : {false, true}) {
            const uint64_t ka = fp8 ? dkc.addr : dk.addr, va = fp8 ? dvc.addr : dv.addr;
            auto full = runo(copt, fp8, dq.addr, ka, va, 300, 0, 0);
            auto part = runo(copt, fp8, dq.addr, ka, va, 172, 128, 128, 300);
            int64_t diff = 0; for (size_t i = 0; i < part.size(); ++i) diff += part[i] != full[128 * QW + i];
            std::printf("%-6s p4 class pin %s  rows 128..299 at pos0 128 (cls 300) vs the 300-row call: %ld of %zu differ  [coopmat]\n", fp8 ? "fp8" : "bf16", diff ? "FAIL" : "PASS", (long)diff, part.size());
            if (diff) ++g_fail;
            for (int64_t cut : {int64_t(100), int64_t(17)}) {
                auto a = runo(copt, fp8, dq.addr, ka, va, cut, 0, 0);
                auto b = runo(copt, fp8, dq.addr, ka, va, 300 - cut, cut, cut);
                int64_t d2 = 0;
                for (size_t i = 0; i < a.size(); ++i) d2 += a[i] != full[i];
                for (size_t i = 0; i < b.size(); ++i) d2 += b[i] != full[size_t(cut) * QW + i];
                std::printf("%-6s p4 partition %s  300 rows at 0 vs %ld + %ld rows: %ld of %zu differ  [coopmat]\n", fp8 ? "fp8" : "bf16", d2 ? "FAIL" : "PASS", (long)cut, (long)(300 - cut), (long)d2, full.size());
                if (d2) ++g_fail;
            }
        }
        // 8a. every row at the end of a call: the 300-row call vs 17-row calls at 0, 17, .., 289 (+ the last 11 rows as part of
        // a 17-row call at 283): rows whose last 16-key chunk holds keys of later rows in one call and zeros in the other
        for (bool fp8 : {false, true}) {
            const uint64_t ka = fp8 ? dkc.addr : dk.addr, va = fp8 ? dvc.addr : dv.addr;
            auto full = runo(copt, fp8, dq.addr, ka, va, 300, 0, 0);
            int64_t d = 0;
            for (int64_t p0 = 0; p0 < 300; p0 += 17) {
                const int64_t a0 = std::min<int64_t>(p0, 283);
                auto part = runo(copt, fp8, dq.addr, ka, va, 17, a0, a0);
                for (size_t i = 0; i < part.size(); ++i) d += part[i] != full[size_t(a0) * QW + i];
            }
            std::printf("%-6s p4 partition %s  300 rows at 0 vs 18 calls of 17 rows: %ld differ  [coopmat]\n", fp8 ? "fp8" : "bf16", d ? "FAIL" : "PASS", (long)d);
            if (d) ++g_fail;
        }
        // 8b. masked-key independence: V (and K) of keys >= 200 replaced; rows < 200 must not change (keys past a row's
        // position share its last 32-key step with it)
        for (bool fp8 : {false, true}) {
            const uint64_t ka = fp8 ? dkc.addr : dk.addr, va = fp8 ? dvc.addr : dv.addr;
            auto base = runo(copt, fp8, dq.addr, ka, va, 300, 0, 0);
            std::vector<uint16_t> k2(kb), v2(vb); std::vector<uint8_t> kc2(kc), vc2(vc);
            std::mt19937 g2(7);
            for (size_t i = size_t(200) * KW; i < size_t(300) * KW; ++i) {
                k2[i] = f2bf(nd(g2) * 8.f); v2[i] = f2bf(nd(g2) * 8.f);
                kc2[i] = uint8_t(g2() % 0x78) | ((g2() & 1) ? 0x80 : 0); vc2[i] = uint8_t(g2() % 0x78) | ((g2() & 1) ? 0x80 : 0);
            }
            cere::vk::Buffer bk = fp8 ? kt::upload(ctx, kc2) : kt::upload(ctx, k2), bv = fp8 ? kt::upload(ctx, vc2) : kt::upload(ctx, v2);
            auto alt = runo(copt, fp8, dq.addr, bk.addr, bv.addr, 300, 0, 0);
            int64_t diff = 0;
            for (size_t i = 0; i < size_t(200) * QW; ++i) diff += alt[i] != base[i];
            std::printf("%-6s p4 masked    %s  K / V of keys 200..299 replaced: rows 0..199 %ld of %ld differ  [coopmat]\n", fp8 ? "fp8" : "bf16", diff ? "FAIL" : "PASS", (long)diff, (long)(200 * QW));
            if (diff) ++g_fail;
            ctx.free(bk); ctx.free(bv);
        }
        // 8c. dense_attention (the forward's call: gated, BF16 rows at prefix 0 / the FP8 cache after a prefix) on the
        // coopmat path vs FP64 (sigmoid(gate) * attention), repeat bit-equal
        {
            const int64_t T = 300;
            std::vector<uint16_t> gate(T * QW); for (auto &x : gate) x = f2bf(nd(g));
            auto dgate = kt::upload(ctx, gate); auto dg = kt::make(ctx, T * QW * 2);
            for (int c = 0; c < 2; ++c) {
                const bool fp8 = c == 1; const int64_t pre = fp8 ? 700 : 0;
                auto rund = [&]() {
                    std::memset(dg.map, 0x5A, size_t(T) * QW * 2);
                    ctx.begin();
                    kt::must(k6::dense_attention(ctx, dq.addr, dk.addr, dv.addr, KW, dgate.addr, T, pre, fp8, dkc.addr, dvc.addr, 0,
                                                 dws.addr, wsb, dg.addr, QW, 0, copt), "dense_attention coopmat");
                    ctx.barrier(); ctx.submit_wait();
                    return kt::host<uint16_t>(dg, size_t(T) * QW);
                };
                auto out = rund(), again = rund();
                double num = 0, den = 0;
                for (int64_t t = 0; t < T; t += 37) for (int h = 0; h < QH; ++h) {
                    const int kvh = h / (QH / KVH); const int64_t nk = pre + t + 1;
                    std::vector<double> sc(nk); double mx = -1e300;
                    for (int64_t j = 0; j < nk; ++j) { double d = 0; for (int cc = 0; cc < D; ++cc) d += double(bf2f(q[t * QW + h * D + cc])) * kval(fp8, j, kvh * D + cc); sc[j] = d * scale; mx = std::max(mx, sc[j]); }
                    double l = 0; for (auto &x : sc) { x = std::exp(x - mx); l += x; }
                    for (int cc = 0; cc < D; ++cc) {
                        double o = 0; for (int64_t j = 0; j < nk; ++j) o += sc[j] * vval(fp8, j, kvh * D + cc);
                        o = o / l / (1.0 + std::exp(-double(bf2f(gate[t * QW + h * D + cc]))));
                        const double e = bf2f(out[t * QW + h * D + cc]) - o; num += e * e; den += o * o;
                    }
                }
                const double err = std::sqrt(num / den);
                const bool ok = err <= 8e-3 && again == out;
                std::printf("%-6s p4 dense_att %s  dense_attention (gated) T %ld at %ld: rel-L2 %.2e vs FP64 (8e-3), repeat %s  [coopmat]\n", fp8 ? "fp8" : "bf16",
                            ok ? "PASS" : "FAIL", (long)T, (long)pre, err, again == out ? "bit-equal" : "DIFFERS");
                if (!ok) ++g_fail;
            }
            ctx.free(dgate); ctx.free(dg);
        }
        // 9. decode / split shapes unchanged by the option
        {
            int64_t diff = 0, n = 0;
            struct S { int64_t T, pos0; } ss[] = {{1, 1500}, {7, 50}, {16, 1000}, {1, 2047}};
            for (bool fp8 : {false, true}) for (auto c : ss) {
                const uint64_t ka = fp8 ? dkc.addr : dk.addr, va = fp8 ? dvc.addr : dv.addr;
                auto a = runo(copt, fp8, dq.addr, ka, va, c.T, c.pos0, 0), b = runo(opt, fp8, dq.addr, ka, va, c.T, c.pos0, 0);
                for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
                n += int64_t(a.size());
            }
            std::vector<uint16_t> gate(16 * QW); for (auto &x : gate) x = f2bf(nd(g));
            auto dgate = kt::upload(ctx, gate); auto dch = kt::make(ctx, 16 * QW * 2);
            std::vector<uint16_t> ch[2];
            for (int w = 0; w < 2; ++w) {
                ctx.begin();
                kt::must(k6::dense_attention_rows(ctx, dq.addr, dgate.addr, 16, 2032, dkc.addr, dvc.addr, 0, dws.addr, wsb, dch.addr, QW, w ? copt : opt), "rows");
                ctx.barrier(); ctx.submit_wait();
                ch[w] = kt::host<uint16_t>(dch, 16 * QW);
            }
            for (size_t i = 0; i < ch[0].size(); ++i) diff += ch[0][i] != ch[1][i];
            n += int64_t(ch[0].size());
            std::printf("p4 decode    %s  T 1 / 7 / 16 / 1 at 2,047 (bf16, fp8) and a 16-row chain with vs without the option: %ld of %ld differ  [coopmat]\n", diff ? "FAIL" : "PASS", (long)diff, (long)n);
            if (diff) ++g_fail;
            ctx.free(dgate); ctx.free(dch);
        }
        // 10. operand ranges (own K/V / q buffers)
        {
            const int64_t T = 300, NKR = 300;
            auto range_row = [&](const char *name, bool fp8, std::vector<uint16_t> qh, std::vector<uint16_t> kh, std::vector<uint16_t> vh,
                                 std::vector<uint8_t> kch, std::vector<uint8_t> vch, int64_t first_flag_token, int first_flag_heads, int flag_token, int flag_g) {
                auto bq = kt::upload(ctx, qh);
                cere::vk::Buffer bk, bv;
                if (fp8) { bk = kt::upload(ctx, kch); bv = kt::upload(ctx, vch); } else { bk = kt::upload(ctx, kh); bv = kt::upload(ctx, vh); }
                auto out = runo(copt, fp8, bq.addr, bk.addr, bv.addr, T, 0, 0);
                const auto fw = kt::host<uint32_t>(dws, size_t(T) * KVH * 3);  // attention(): the flags at workspace offset 0
                auto again = runo(copt, fp8, bq.addr, bk.addr, bv.addr, T, 0, 0);
                auto f32 = runo(opt, fp8, bq.addr, bk.addr, bv.addr, T, 0, 0);
                // flagged rows: tokens >= first_flag_token of the KV heads in the mask first_flag_heads (bit g); token
                // flag_token's KV head flag_g heads
                auto flagged = [&](int64_t t, int h) {
                    const int kvh = h / (QH / KVH);
                    return (first_flag_token >= 0 && t >= first_flag_token && ((first_flag_heads >> kvh) & 1)) || (t == flag_token && kvh == flag_g);
                };
                int64_t flag_wrong = 0, flag_set = 0;
                for (int64_t t = 0; t < T; ++t) for (int gg = 0; gg < KVH; ++gg) {
                    const bool f = (fw[(t * KVH + gg) * 3] | fw[(t * KVH + gg) * 3 + 1] | fw[(t * KVH + gg) * 3 + 2]) != 0;
                    flag_set += f; flag_wrong += f != flagged(t, gg * (QH / KVH));
                }
                int64_t fl_diff = 0, fl_n = 0, nonfinite = 0;
                for (int64_t t = 0; t < T; ++t) for (int h = 0; h < QH; ++h) for (int c = 0; c < D; ++c) {
                    const size_t i = size_t(t) * QW + h * D + c;
                    if (flagged(t, h)) { fl_diff += out[i] != f32[i]; ++fl_n; }
                    else nonfinite += (out[i] & 0x7F80) == 0x7F80;
                }
                auto kv = [&](int64_t j, int c) { return fp8 ? double(e4m3(kch[j * KW + c])) : double(bf2f(kh[j * KW + c])); };
                auto vv = [&](int64_t j, int c) { return fp8 ? double(e4m3(vch[j * KW + c])) : double(bf2f(vh[j * KW + c])); };
                const double err = ref_err(qh, kv, vv, T, 0, out, 7, flagged);
                const bool ok = err <= 8e-3 && fl_diff == 0 && nonfinite == 0 && again == out && flag_wrong == 0;
                std::printf("%-6s p4 range    %s  %s: unflagged rows rel-L2 %.2e vs FP64 (8e-3), %ld non-finite; flags %ld set, %ld unexpected; flagged rows vs FP32 path %ld of %ld differ; repeat %s  [coopmat]\n",
                            fp8 ? "fp8" : "bf16", ok ? "PASS" : "FAIL", name, err, (long)nonfinite, (long)flag_set, (long)flag_wrong, (long)fl_diff, (long)fl_n, again == out ? "bit-equal" : "DIFFERS");
                if (!ok) ++g_fail;
                ctx.free(bq); ctx.free(bk); ctx.free(bv);
            };
            std::vector<uint16_t> qh(q.begin(), q.begin() + T * QW), kh(kb.begin(), kb.begin() + NKR * KW), vh(vb.begin(), vb.begin() + NKR * KW);
            std::vector<uint8_t> kch(kc.begin(), kc.begin() + NKR * KW), vch(vc.begin(), vc.begin() + NKR * KW);
            auto scaled = [](std::vector<uint16_t> x, int e, size_t from, size_t to) { for (size_t i = from; i < to; ++i) x[i] = f2bf(std::ldexp(bf2f(x[i]), e)); return x; };
            // q rows of tokens 0..149 x 2^90, keys 0..199 x 2^-100 (scores unchanged in magnitude for mixed pairs)
            range_row("q x 2^90 (tokens < 150), K x 2^-100 (keys < 200)", false, scaled(qh, 90, 0, size_t(150) * QW), scaled(kh, -100, 0, size_t(200) * KW), vh, kch, vch, -1, 0, -1, 0);
            range_row("V x 2^-24 (all keys)", false, qh, kh, scaled(vh, -24, 0, vh.size()), kch, vch, -1, 0, -1, 0);
            range_row("V x 2^-24 for keys < 100, x 2^10 above", false, qh, kh, scaled(scaled(vh, -24, 0, size_t(100) * KW), 10, size_t(100) * KW, vh.size()), kch, vch, -1, 0, -1, 0);
            { auto v2 = vh; for (int c = 0; c < 8; ++c) v2[150 * KW + c * 37] = f2bf(100000.f * (c % 2 ? -1.f : 1.f));
              range_row("V outliers 1e5 (> FP16) at key 150, both KV heads (fallback rows >= 150)", false, qh, kh, v2, kch, vch, 150, 3, -1, 0); }
            { auto k2 = kh; k2[200 * KW + 300] = 0x7F80;  // +inf, KV head 1 (dims 256..511)
              range_row("+inf in K at key 200, KV head 1 (fallback rows >= 200 of KV head 1)", false, qh, k2, vh, kch, vch, 200, 2, -1, 0); }
            { auto q2 = qh; q2[50 * QW + 3 * D + 17] = 0x7FC0;  // NaN, head 3 (KV head 0)
              range_row("NaN in q (token 50, head 3: fallback of token 50, KV head 0)", false, q2, kh, vh, kch, vch, -1, 0, 50, 0); }
            { auto v3 = vch; v3[77 * KW + 5] = 0x7F;  // E4M3 NaN, KV head 0
              range_row("E4M3 NaN byte in V at key 77, KV head 0 (fallback rows >= 77 of KV head 0)", true, qh, kh, vh, kch, v3, 77, 1, -1, 0); }
        }
        // 11. negative rows
        {
            struct N { const char *what; cere_status want; cere_status got; } ns[4];
            auto neg = [&](const k6::Options &o, int64_t T, int64_t pos0, uint64_t ws, size_t wsn) {
                ctx.begin();
                const cere_status e = k6::attention(ctx, dq.addr, T, pos0, 0, 0, 0, 0, dkc.addr, dvc.addr, 0, scale, ws, wsn, dout.addr, 0, o);
                ctx.submit_wait();
                return e;
            };
            k6::Options bad = copt; bad.prefill = k6::Prefill(7);
            ns[0] = {"coopmat prefill without workspace", CERE_ERR_ARGUMENT, neg(copt, 300, 0, 0, 0)};
            ns[1] = {"coopmat prefill, workspace below the flags", CERE_ERR_ARGUMENT, neg(copt, 300, 0, dws.addr, 64)};
            ns[2] = {"an unknown Prefill value", CERE_ERR_ARGUMENT, neg(bad, 300, 0, dws.addr, wsb)};
            ns[3] = {"coopmat prefill, pos0 + tokens > 2,048", CERE_ERR_ARGUMENT, neg(copt, 100, 1949, dws.addr, wsb)};
            for (auto &n : ns) {
                const bool ok = n.got == n.want;
                std::printf("p4 negative  %s  %s -> %d (want %d)  [coopmat]\n", ok ? "PASS" : "FAIL", n.what, int(n.got), int(n.want));
                if (!ok) ++g_fail;
            }
        }
    }

    // timing
    auto time_it = [&](auto &&record) {
        const int reps = 15;
        ctx.begin();
        for (int i = 0; i < 3; ++i) { record(); ctx.barrier(); }
        for (int i = 0; i < reps; ++i) { ctx.stamp(2 * i); record(); ctx.stamp(2 * i + 1); ctx.barrier(); }
        ctx.submit_wait();
        std::vector<double> t;
        for (int i = 0; i < reps; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
        std::sort(t.begin(), t.end());
        return std::vector<double>{t[t.size() / 2], t.front(), t.back()};
    };
    for (bool fp8 : {false, true}) {
        struct TC { int64_t T, pos0; } tc[] = {{2048, 0}, {512, 0}, {1024, 0}, {1000, 1048}};
        for (auto c : tc) {
            if (!fp8 && c.pos0 > 0) continue;  // a later chunk reads the FP8 cache in the model
            for (int w = 0; w < 2; ++w) {
                auto t = time_it([&] { kt::must(rec2(w ? copt : opt, fp8, c.T, c.pos0), "attention"); });
                double pairs = 0; for (int64_t i = 0; i < c.T; ++i) pairs += double(c.pos0 + i + 1);
                const double tf = 4.0 * QH * D * pairs / (t[0] * 1e9);
                std::printf("%-6s timing p4    T %4ld pos0 %4ld: %.3f ms median of 15 (min %.3f, max %.3f) (%s, %.2f TFLOP/s causal-useful)\n", fp8 ? "fp8" : "bf16", (long)c.T,
                            (long)c.pos0, t[0], t[1], t[2], w ? "prefill coopmat" : "prefill FP32", tf);
            }
        }
    }
    if (!p4_only) for (bool fp8 : {false, true}) {
        struct TC { int64_t T, pos0; } tc[] = {{2048, 0}, {512, 0}, {1, 2047}, {4, 2040}};
        for (auto c : tc) {
            auto t = time_it([&] { kt::must(rec(fp8, c.T, c.pos0, 0, 0), "attention"); });
            double pairs = 0; for (int64_t i = 0; i < c.T; ++i) pairs += double(c.pos0 + i + 1);
            const double tf = 4.0 * QH * D * pairs / (t[0] * 1e9);
            std::printf("%-6s timing       T %4ld pos0 %4ld: %.3f ms median of 15 (min %.3f, max %.3f) (%s, %.2f TFLOP/s causal-useful)\n", fp8 ? "fp8" : "bf16", (long)c.T, (long)c.pos0,
                        t[0], t[1], t[2], c.T > 16 ? "prefill FP32" : "split", tf);
        }
    }
    if (!p4_only) {
        std::vector<uint16_t> gate(16 * QW); for (auto &x : gate) x = f2bf(nd(g));
        auto dgate = kt::upload(ctx, gate);
        struct CC { int64_t R, pos0; } cc[] = {{4, 2040}, {8, 2040}, {16, 2032}};
        for (auto c : cc) {
            auto t = time_it([&] { kt::must(k6::dense_attention_rows(ctx, dq.addr, dgate.addr, c.R, c.pos0, dkc.addr, dvc.addr, 0, dws.addr, wsb, dout.addr, QW, opt), "rows"); });
            std::printf("%-6s timing chain R %4ld pos0 %4ld: %.3f ms median of 15 (min %.3f, max %.3f) (dense_attention_rows, gated)\n", "fp8", (long)c.R, (long)c.pos0, t[0], t[1], t[2]);
        }
        auto t = time_it([&] { kt::must(k6::dense_attention(ctx, dq.addr, dk.addr, dv.addr, KW, dgate.addr, 1, 2047, true, dkc.addr, dvc.addr, 0, dws.addr, wsb, dout.addr, QW, 0, opt), "da"); });
        std::printf("%-6s timing step  T    1 pos0 2047: %.3f ms median of 15 (min %.3f, max %.3f) (dense_attention, gated)\n", "fp8", t[0], t[1], t[2]);
        ctx.free(dgate);
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
