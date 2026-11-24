// VulkanAmd P2: executor/hip/ktest/k9_test.cpp's contract rows on the Vulkan K9 QSA (src/kernels/k9_qsa_vk.*), same
// seed (31), data generation order, shapes and bounds:
//  1. qsa_attention (sparse GQA over index rows, -1 padded) vs FP64 on the FP8 cache (rows 1, 5, 16: the split path;
//     40: the rows path) and BF16 rows (rows 1, 40): rel-L2 (limit 8e-3); qsa_attention_heads with rows_heads 0 / 3 / 6
//     / 12 all the same bytes; an empty row is zeros. "P4" (use_p4, rows 40): vs FP64 the same, rows 20..39 alone
//     (cls 40) and a repeat bit-equal to the 40-row call. "P4 fp8 decoded copy": the FP8 rows path with scratch decodes
//     the call's K / V once into BF16 copies; copies of all / half / 1 positions vs none, same bytes. A row of a
//     16-row FP8 call == that row alone.
//  2. qsa_topk on random F32 logits (ties, -0 / +0, -inf past the visible blocks) vs a host port of the selection,
//     impl 0 and 1, rows 3 and 40 (impl 2, CUDA's cluster kernel, refused as on HIP).
//  3. qsa_logits vs FP64 (|err| / sum|terms| <= 1e-5 per finite logit, -inf exactly where required), impl 0 / 1 / 2
//     the same bytes.
//  4. qsa_select == qsa_topk over qsa_logits (rows 3, seq_len 5,000).
// Vulkan-only rows: an 8-row verify chain (FP8 and BF16) == its rows' one-row calls; qsa_select_step (device position)
// == qsa_select at seq_len = position + 1 (16K context); repeats bit-equal.
// Timing (--time; GPU timestamps, 3 warm-ups, median / min / max of 15, a barrier after each call): one-token decode
// (attention alone, and the K9 step: indexer_qk + qsa_compress_store_step + qsa_select_step + qsa_attention) on BF16
// and FP8 KV at a 2K (position 2,047) and a 16K (position 16,383) context; an 8-row verify chain (qsa_select +
// qsa_attention, FP8, 16K); a 4,096-row prefill attention (FP8 and BF16, 16K, 512 random blocks of 4 + tail per row,
// the amd-amd step 16 shape) and its qsa_select; P4-A: the coopmat rows path on the prefill calls and the S3 3,000-token
// prompt's call (3,000 BF16 rows, positions 0..2,999: dense prefix below 2,048, 512 random blocks + tail above; also on
// the FP8 cache). ktest/hip_k9_timing.cpp times HIP on the same data (same generator, same order).
// P4-A rows (the cooperative-matrix rows path, Options::prefill = kCoopmat; HIP's bound): the 40 HIP rows vs FP64 with
// cls 40 / 17 + 23 partitions / repeat bit-equal; the split side bit-identical with and without the option; 300 rows of
// the timing shapes (positions 2,048.. with 512 random blocks; positions 0..299 dense) vs FP64 + partitions; operand
// ranges (q x 2^100, K x 2^-110, V x 2^-30; a V outlier beyond the coopmat range, -inf in K, NaN in q, an E4M3 NaN byte in
// V): flags exactly the rows that read them, flagged rows bit-equal to the FP32 path, the others within the bound;
// negative rows (scratch missing / short / misaligned, an unknown Prefill value).
// Usage: k9_test <spv dir> [--time] [--time-only] [--subgroup 32|64] [--heads 0|3|6|12 (rows_heads of the timed calls)]
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "launch/vk_launch_testing.h"
#include "kernels/k9_qsa_vk.h"
#include "ktest_util.h"

namespace k9 = cere::vkk9;
using kt::bf2f;
using kt::f2bf;
static int g_fail = 0;
static float e4m3(uint8_t c) { const int s = c >> 7, e = (c >> 3) & 15, m = c & 7; const float v = e == 0 ? std::ldexp(m / 8.f, -6) : std::ldexp(1.f + m / 8.f, e - 7); return s ? -v : v; }
static void check(const char *name, bool ok, const char *msg) { std::printf("%-26s %s  %s\n", name, ok ? "PASS" : "FAIL", msg); if (!ok) ++g_fail; }

constexpr int QH = k9::kQHeads, D = k9::kHeadDim, QW = QH * D, KW = k9::kKvHeads * D, RW = k9::kRowWidth, G = QH / k9::kKvHeads;

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

// An index row as the timing uses it (amd-amd step 16): position p; when more than 512 blocks are visible, 512 random
// distinct visible blocks ascending, each expanded; then the tail; then -1.
static void timing_row(std::mt19937 &g, int64_t p, int32_t *out) {
    const int64_t vis = (p + 1) / 4;
    std::vector<int64_t> blocks;
    if (vis <= 512) { for (int64_t b = 0; b < vis; ++b) blocks.push_back(b); }
    else {
        std::vector<int64_t> all(vis); for (int64_t b = 0; b < vis; ++b) all[b] = b;
        for (int64_t i = 0; i < 512; ++i) { const int64_t j = i + int64_t(g() % uint32_t(vis - i)); std::swap(all[i], all[j]); }
        blocks.assign(all.begin(), all.begin() + 512); std::sort(blocks.begin(), blocks.end());
    }
    int n = 0;
    for (int64_t b : blocks) for (int i = 0; i < 4; ++i) out[n++] = int32_t(4 * b + i);
    for (int64_t q = vis * 4; q <= p; ++q) out[n++] = int32_t(q);
    while (n < RW) out[n++] = -1;
}

struct Stat { double med, mn, mx; };

int main(int argc, char **argv) {
    auto ctx_owner = cere::vk::testing::create_or_die(argc > 1 ? argv[1] : "build/vulkan-k9/spv");
    cere::vk::Context &ctx = *ctx_owner;
    bool time = false, time_only = false;
    k9::Options opt;
    int theads = k9::kRowsHeads;  // rows_heads of the timed attention calls
    for (int i = 2; i < argc; ++i) {
        if (kt::hook_arg(argc, argv, i)) continue;  // VulkanAutotune: --identity DIR, --samples (ktest_util.h)
        if (!std::strcmp(argv[i], "--time")) time = true;
        else if (!std::strcmp(argv[i], "--time-only")) time = time_only = true;
        else if (!std::strcmp(argv[i], "--subgroup") && i + 1 < argc) opt.subgroup = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--heads") && i + 1 < argc) theads = std::atoi(argv[++i]);
    }
    std::printf("device: %s, driver %s %s; subgroup %u, default rows_heads %d\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(),
                ctx.info().driver_info.c_str(), opt.subgroup, k9::kRowsHeads);
    auto must = [&](cere_status s, const char *what) { kt::must(s, what); };

  if (!time_only) {
    std::mt19937 g(31); std::normal_distribution<float> nd; std::uniform_real_distribution<float> ud;
    const int64_t NK = 6000;
    std::vector<uint8_t> kc(NK * KW), vc(NK * KW);
    for (auto &b : kc) { b = uint8_t(g() % 0x78); if (g() & 1) b |= 0x80; }
    for (auto &b : vc) { b = uint8_t(g() % 0x78); if (g() & 1) b |= 0x80; }
    std::vector<uint16_t> kb(NK * KW), vb(NK * KW); for (auto &x : kb) x = f2bf(nd(g)); for (auto &x : vb) x = f2bf(nd(g));
    auto dkc = kt::upload(ctx, kc), dvc = kt::upload(ctx, vc), dkb = kt::upload(ctx, kb), dvb = kt::upload(ctx, vb);
    // ---- 1. qsa_attention
    {
        const int64_t R = 40; std::vector<uint16_t> q(R * QW); for (auto &x : q) x = f2bf(nd(g) * 0.25f);
        std::vector<int32_t> idx(R * RW, -1);
        for (int64_t t = 0; t < R; ++t) {
            const int nv = 1200 + int(g() % 852); std::vector<int32_t> pos(NK); for (int i = 0; i < NK; ++i) pos[i] = i;
            std::shuffle(pos.begin(), pos.end(), g); std::sort(pos.begin(), pos.begin() + nv);
            for (int i = 0; i < nv; ++i) idx[t * RW + i] = pos[i];
        }
        idx[3 * RW + 0] = -1; for (int i = 0; i < RW; ++i) idx[7 * RW + i] = -1;
        auto dq = kt::upload(ctx, q); auto didx = kt::upload(ctx, idx); auto dout = kt::make(ctx, R * QW * 2);
        const size_t sb = std::max(k9::attention_scratch_bytes(16), k9::attention_scratch_bytes(R)); auto ds = kt::make(ctx, sb);
        auto run = [&](bool fp8, int64_t rows, int64_t r0, int heads, bool p4 = false, int64_t cls = 0, uint64_t scr = 0, size_t scrb = 0) {
            if (scr == ~uint64_t(0)) { scr = 0; scrb = 0; }   // no scratch at all
            else if (!scr) { scr = ds.addr; scrb = sb; }
            const uint64_t qq = dq.addr + uint64_t(r0) * QW * 2, ii = didx.addr + uint64_t(r0) * RW * 4;
            std::memset(dout.map, 0x5A, size_t(R) * QW * 2);
            ctx.begin();
            must(fp8 ? k9::qsa_attention_heads(ctx, qq, QW, ii, rows, 0, 0, 0, 0, dkc.addr, dvc.addr, 0, k9::kSoftmaxScale, scr, scrb, dout.addr, QW, p4, cls, heads, opt)
                     : k9::qsa_attention_heads(ctx, qq, QW, ii, rows, dkb.addr, KW, dvb.addr, KW, 0, 0, 0, k9::kSoftmaxScale, scr, scrb, dout.addr, QW, p4, cls, heads, opt),
                 "qsa_attention");
            ctx.barrier(); ctx.submit_wait();
            return kt::host<uint16_t>(dout, size_t(rows) * QW);
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
            if (p4) {
                auto part = run(fp8, 20, 20, k9::kRowsHeads, true, 40);
                same = std::equal(part.begin(), part.end(), out.begin() + 20 * QW) && run(fp8, rows, 0, k9::kRowsHeads, true) == out;
            }
            char msg[256]; std::snprintf(msg, sizeof msg, "rows %2ld: rel-L2 %.2e vs FP64 (8e-3), empty row zeros %s, %s %s", (long)rows, err,
                                         zero_bad ? "NO" : "yes", p4 ? "rows 20..39 alone + repeat" : "rows_heads 0/3/6/12", same ? "same bytes" : "DIFFER");
            check(p4 ? (fp8 ? "qsa_attention P4 fp8" : "qsa_attention P4 bf16") : (fp8 ? "qsa_attention fp8" : "qsa_attention bf16"), err <= 8e-3 && !zero_bad && same, msg);
        }
        {   // the HIP P4 decoded-copy row: scratch capacities covering all / half / 1 positions vs none
            const auto base = run(true, 40, 0, k9::kRowsHeads, true, 0, ~uint64_t(0), 0);
            long diff = 0;
            for (int64_t cap : {NK + 7, NK / 2, int64_t(1)}) {
                const size_t bytes = 256 + size_t(cap) * 2 * 512 * 2;
                auto big = kt::make(ctx, bytes); std::memset(big.map, 0xA5, bytes);
                diff += run(true, 40, 0, k9::kRowsHeads, true, 0, big.addr, bytes) != base;
                ctx.free(big);
            }
            char msg[200]; std::snprintf(msg, sizeof msg, "40 rows, BF16 copies of all / half / 1 positions vs none: %ld of 3 differ", diff);
            check("P4 fp8 decoded copy", diff == 0, msg);
        }
        auto full = run(true, 16, 0, k9::kRowsHeads), one = run(true, 1, 9, k9::kRowsHeads);
        check("split rows invariance", std::equal(one.begin(), one.end(), full.begin() + 9 * QW), "row 9 of a 16-row FP8 call == row 9 alone, bit for bit");
        for (bool fp8 : {true, false}) {   // Vulkan row: an 8-row verify chain == its rows' one-row calls; a repeat
            auto chain = run(fp8, 8, 24, k9::kRowsHeads);
            int64_t bad = 0;
            for (int64_t t = 0; t < 8; ++t) { auto r1 = run(fp8, 1, 24 + t, k9::kRowsHeads); bad += !std::equal(r1.begin(), r1.end(), chain.begin() + t * QW); }
            const bool rep = run(fp8, 8, 24, k9::kRowsHeads) == chain;
            char msg[160]; std::snprintf(msg, sizeof msg, "8 rows (24..31) vs 8 one-row calls: %ld rows differ; repeat %s [Vulkan row]", (long)bad, rep ? "bit-equal" : "DIFFERS");
            check(fp8 ? "verify chain fp8" : "verify chain bf16", bad == 0 && rep, msg);
        }
        // ===== P4-A: the cooperative-matrix prefill path (Options::prefill = kCoopmat) =====
        k9::Options copt = opt; copt.prefill = k9::Prefill::kCoopmat;
        const size_t csb = std::max(sb, k9::coopmat_scratch_bytes(320)); auto dcs = kt::make(ctx, csb);
        // run on any q / index / K / V buffers; rows of R0 (<= 300)
        auto runc = [&](const k9::Options &o, bool fp8, uint64_t qa, uint64_t ia, uint64_t ka, uint64_t va, int64_t rows, int64_t r0, int64_t cls = 0) {
            const uint64_t qq = qa + uint64_t(r0) * QW * 2, ii = ia + uint64_t(r0) * RW * 4;
            auto bo = kt::make(ctx, size_t(rows) * QW * 2);
            std::memset(bo.map, 0x5A, size_t(rows) * QW * 2);
            ctx.begin();
            must(fp8 ? k9::qsa_attention(ctx, qq, QW, ii, rows, 0, 0, 0, 0, ka, va, 0, k9::kSoftmaxScale, dcs.addr, csb, bo.addr, QW, true, cls, o)
                     : k9::qsa_attention(ctx, qq, QW, ii, rows, ka, KW, va, KW, 0, 0, 0, k9::kSoftmaxScale, dcs.addr, csb, bo.addr, QW, true, cls, o),
                 "qsa_attention p4");
            ctx.barrier(); ctx.submit_wait();
            auto h = kt::host<uint16_t>(bo, size_t(rows) * QW);
            ctx.free(bo);
            return h;
        };
        // rel-L2 vs FP64 over rows t (every step-th), heads (every hstep-th), skipping skip(t, kvh) rows
        auto ref = [&](const std::vector<uint16_t> &qh, const std::vector<int32_t> &ih, auto kv, auto vv, int64_t rows,
                       const std::vector<uint16_t> &out, int64_t step, int hstep, auto skip, int64_t *zero_bad) {
            double num = 0, den = 0;
            for (int64_t t = 0; t < rows; t += step) for (int h = 0; h < QH; h += hstep) {
                const int kvh = h / G; if (skip(t, kvh)) continue;
                std::vector<double> sc; std::vector<int32_t> js;
                for (int i = 0; i < RW; ++i) { const int32_t j = ih[t * RW + i]; if (j < 0) continue; js.push_back(j);
                    double d = 0; for (int c = 0; c < D; ++c) d += double(bf2f(qh[t * QW + h * D + c])) * kv(j, kvh * D + c);
                    sc.push_back(d * k9::kSoftmaxScale); }
                if (js.empty()) { for (int c = 0; c < D; ++c) *zero_bad += out[t * QW + h * D + c] != 0; continue; }
                const double mx = *std::max_element(sc.begin(), sc.end()); double l = 0; for (auto &x : sc) { x = std::exp(x - mx); l += x; }
                for (int c = 0; c < D; ++c) { double o = 0; for (size_t i = 0; i < js.size(); ++i) o += sc[i] * vv(js[i], kvh * D + c);
                    o /= l; const double e = bf2f(out[t * QW + h * D + c]) - o; num += e * e; den += o * o; }
            }
            return std::sqrt(num / den);
        };
        auto noskip = [](int64_t, int) { return false; };
        auto kvf = [&](bool fp8) { return [&, fp8](int64_t j, int c) { return fp8 ? double(e4m3(kc[j * KW + c])) : double(bf2f(kb[j * KW + c])); }; };
        auto vvf = [&](bool fp8) { return [&, fp8](int64_t j, int c) { return fp8 ? double(e4m3(vc[j * KW + c])) : double(bf2f(vb[j * KW + c])); }; };
        // the 40 HIP rows (random entry sets, an invalid first entry, an empty row)
        for (bool fp8 : {true, false}) {
            const uint64_t ka = fp8 ? dkc.addr : dkb.addr, va = fp8 ? dvc.addr : dvb.addr;
            auto out = runc(copt, fp8, dq.addr, didx.addr, ka, va, 40, 0);
            int64_t zb = 0;
            const double err = ref(q, idx, kvf(fp8), vvf(fp8), 40, out, 1, 5, noskip, &zb);
            auto part = runc(copt, fp8, dq.addr, didx.addr, ka, va, 20, 20, 40);
            auto a17 = runc(copt, fp8, dq.addr, didx.addr, ka, va, 17, 0), b23 = runc(copt, fp8, dq.addr, didx.addr, ka, va, 23, 17);
            const bool same = std::equal(part.begin(), part.end(), out.begin() + 20 * QW) && runc(copt, fp8, dq.addr, didx.addr, ka, va, 40, 0) == out &&
                              std::equal(a17.begin(), a17.end(), out.begin()) && std::equal(b23.begin(), b23.end(), out.begin() + 17 * QW);
            auto f32 = runc(opt, fp8, dq.addr, didx.addr, ka, va, 40, 0);
            int64_t ndf = 0; double md = 0;
            for (size_t i = 0; i < out.size(); ++i) { ndf += out[i] != f32[i]; md = std::max(md, std::fabs(double(bf2f(out[i])) - bf2f(f32[i]))); }
            char msg[300]; std::snprintf(msg, sizeof msg, "rows 40: rel-L2 %.2e vs FP64 (8e-3), empty row zeros %s, rows 20..39 alone (cls 40) + 17 + 23 + repeat %s; vs FP32 path %ld of %zu differ, max |diff| %.3g [coopmat]",
                                         err, zb ? "NO" : "yes", same ? "same bytes" : "DIFFER", (long)ndf, out.size(), md);
            check(fp8 ? "p4 qsa_attention fp8" : "p4 qsa_attention bf16", err <= 8e-3 && !zb && same, msg);
        }
        // the split side is unchanged by the option
        {
            int64_t diff = 0, n = 0;
            for (bool fp8 : {true, false}) for (int64_t rows : {int64_t(1), int64_t(5), int64_t(16)}) {
                const uint64_t ka = fp8 ? dkc.addr : dkb.addr, va = fp8 ? dvc.addr : dvb.addr;
                auto a = runc(copt, fp8, dq.addr, didx.addr, ka, va, rows, 3), b = runc(opt, fp8, dq.addr, didx.addr, ka, va, rows, 3);
                for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
                n += int64_t(a.size());
            }
            char msg[200]; std::snprintf(msg, sizeof msg, "rows 1 / 5 / 16 (fp8, bf16) with vs without the option: %ld of %ld differ [coopmat]", (long)diff, (long)n);
            check("p4 split unchanged", diff == 0, msg);
        }
        // 300 rows at positions 2,048 .. 2,347 (512 random blocks + tail, the timing rows' shape) and 300 dense-prefix
        // rows (positions 0 .. 299: every position, the first 2,048 rows of a prompt)
        {
            const int64_t RR = 300;
            std::vector<uint16_t> q3(RR * QW); for (auto &x : q3) x = f2bf(nd(g) * 0.25f);
            auto dq3 = kt::upload(ctx, q3);
            for (int shape = 0; shape < 2; ++shape) {
                std::vector<int32_t> i3(RR * RW);
                for (int64_t t = 0; t < RR; ++t) timing_row(g, shape ? t : 2048 + t, &i3[t * RW]);
                auto di3 = kt::upload(ctx, i3);
                for (bool fp8 : {true, false}) {
                    const uint64_t ka = fp8 ? dkc.addr : dkb.addr, va = fp8 ? dvc.addr : dvb.addr;
                    auto out = runc(copt, fp8, dq3.addr, di3.addr, ka, va, RR, 0);
                    int64_t zb = 0;
                    const double err = ref(q3, i3, kvf(fp8), vvf(fp8), RR, out, 13, 1, noskip, &zb);
                    auto a = runc(copt, fp8, dq3.addr, di3.addr, ka, va, 100, 0), b = runc(copt, fp8, dq3.addr, di3.addr, ka, va, 200, 100);
                    const bool same = std::equal(a.begin(), a.end(), out.begin()) && std::equal(b.begin(), b.end(), out.begin() + 100 * QW) &&
                                      runc(copt, fp8, dq3.addr, di3.addr, ka, va, RR, 0) == out;
                    char msg[300]; std::snprintf(msg, sizeof msg, "%s, 300 rows: rel-L2 %.2e vs FP64 (8e-3, every 13th row), 100 + 200 rows + repeat %s [coopmat]",
                                                 shape ? "positions 0..299 (dense prefix)" : "positions 2,048..2,347 (512 blocks + tail)", err, same ? "same bytes" : "DIFFER");
                    check(fp8 ? "p4 qsa_attention fp8" : "p4 qsa_attention bf16", err <= 8e-3 && !zb && same, msg);
                }
                ctx.free(di3);
            }
            ctx.free(dq3);
        }
        // routing: dense-prefix blocks on the dense kernel vs every row on the sparse kernel, the same bytes; calls whose
        // 16-row blocks fall differently (rows 8.. alone, rows 0..2,047 + a block straddling 2,048)
        {
            const int64_t RR = 320;
            std::vector<uint16_t> q3(RR * QW); for (auto &x : q3) x = f2bf(nd(g) * 0.25f);
            auto dq3 = kt::upload(ctx, q3);
            k9::Options sopt = copt; sopt.dense_blocks = false;
            for (int64_t p0 : {int64_t(0), int64_t(1900)}) {
                std::vector<int32_t> i3(RR * RW);
                for (int64_t t = 0; t < RR; ++t) timing_row(g, p0 + t, &i3[t * RW]);
                auto di3 = kt::upload(ctx, i3);
                for (bool fp8 : {true, false}) {
                    const uint64_t ka = fp8 ? dkc.addr : dkb.addr, va = fp8 ? dvc.addr : dvb.addr;
                    auto routed = runc(copt, fp8, dq3.addr, di3.addr, ka, va, RR, 0);
                    auto sparse = runc(sopt, fp8, dq3.addr, di3.addr, ka, va, RR, 0);
                    auto tail = runc(copt, fp8, dq3.addr, di3.addr, ka, va, RR - 8, 8);
                    int64_t d1 = 0, d2 = 0;
                    for (size_t i = 0; i < routed.size(); ++i) {
                        if (routed[i] != sparse[i] && d1 < 8)
                            std::printf("  routing diff: row %zu head %zu dim %zu: routed 0x%04x sparse 0x%04x\n", i / QW, (i % QW) / D, i % D, routed[i], sparse[i]);
                        d1 += routed[i] != sparse[i];
                    }
                    for (size_t i = 0; i < tail.size(); ++i) d2 += tail[i] != routed[8 * QW + i];
                    const auto rw = kt::host<int32_t>(dcs, size_t((RR - 8 + 15) / 16), k9::coopmat_scratch_bytes(RR - 8) - ((size_t((RR - 8 + 15) / 16) * 4 + 255) & ~size_t(255)));
                    int64_t dense_blocks = 0; for (int32_t w : rw) dense_blocks += w >= 0;
                    int64_t zb = 0;
                    const double err = ref(q3, i3, kvf(fp8), vvf(fp8), RR, routed, 17, 1, noskip, &zb);
                    char msg[300]; std::snprintf(msg, sizeof msg, "positions %ld..%ld: routed vs all-sparse %ld differ, rows 8.. alone %ld differ (%ld of %ld blocks dense there), rel-L2 %.2e vs FP64 [coopmat]",
                                                 (long)p0, (long)(p0 + RR - 1), (long)d1, (long)d2, (long)dense_blocks, (long)rw.size(), err);
                    check(fp8 ? "p4 routing fp8" : "p4 routing bf16", d1 == 0 && d2 == 0 && err <= 8e-3 && !zb && (p0 > 0 || dense_blocks > 0), msg);
                }
                ctx.free(di3);
            }
            ctx.free(dq3);
        }
        // operand ranges (BF16 rows unless noted): rows 0..39 of the HIP data
        {
            auto range = [&](const char *name, bool fp8, std::vector<uint16_t> qh, std::vector<uint16_t> kh, std::vector<uint16_t> vh,
                             std::vector<uint8_t> kch, std::vector<uint8_t> vch, auto expect) {
                auto bq = kt::upload(ctx, qh);
                cere::vk::Buffer bk, bv;
                if (fp8) { bk = kt::upload(ctx, kch); bv = kt::upload(ctx, vch); } else { bk = kt::upload(ctx, kh); bv = kt::upload(ctx, vh); }
                auto out = runc(copt, fp8, bq.addr, didx.addr, bk.addr, bv.addr, 40, 0);
                const auto fw3 = kt::host<uint32_t>(dcs, 40 * 2 * 3);
                std::vector<uint32_t> fw(80); for (int i = 0; i < 80; ++i) fw[i] = fw3[3 * i] | fw3[3 * i + 1] | fw3[3 * i + 2];
                auto again = runc(copt, fp8, bq.addr, didx.addr, bk.addr, bv.addr, 40, 0);
                auto f32 = runc(opt, fp8, bq.addr, didx.addr, bk.addr, bv.addr, 40, 0);
                int64_t fwrong = 0, fset = 0, fdiff = 0, fn = 0, nonfin = 0;
                for (int64_t t = 0; t < 40; ++t) for (int gg = 0; gg < 2; ++gg) { fset += fw[t * 2 + gg] != 0; fwrong += (fw[t * 2 + gg] != 0) != expect(t, gg); }
                for (int64_t t = 0; t < 40; ++t) for (int h = 0; h < QH; ++h) for (int c = 0; c < D; ++c) {
                    const size_t i = size_t(t) * QW + h * D + c;
                    if (expect(t, h / G)) { fdiff += out[i] != f32[i]; ++fn; } else nonfin += (out[i] & 0x7F80) == 0x7F80;
                }
                auto kv = [&](int64_t j, int c) { return fp8 ? double(e4m3(kch[j * KW + c])) : double(bf2f(kh[j * KW + c])); };
                auto vv = [&](int64_t j, int c) { return fp8 ? double(e4m3(vch[j * KW + c])) : double(bf2f(vh[j * KW + c])); };
                int64_t zb = 0;
                const double err = ref(qh, idx, kv, vv, 40, out, 3, 5, expect, &zb);
                const bool ok = err <= 8e-3 && !zb && fwrong == 0 && fdiff == 0 && nonfin == 0 && again == out;
                char msg[400]; std::snprintf(msg, sizeof msg, "%s: unflagged rel-L2 %.2e (8e-3), %ld non-finite; flags %ld set, %ld unexpected; flagged rows vs FP32 path %ld of %ld differ; repeat %s [coopmat]",
                                             name, err, (long)nonfin, (long)fset, (long)fwrong, (long)fdiff, (long)fn, again == out ? "bit-equal" : "DIFFERS");
                check(fp8 ? "p4 range fp8" : "p4 range bf16", ok, msg);
                ctx.free(bq); ctx.free(bk); ctx.free(bv);
            };
            auto none = [](int64_t, int) { return false; };
            auto scaled = [](std::vector<uint16_t> x, int e) { for (auto &y : x) y = f2bf(std::ldexp(bf2f(y), e)); return x; };
            std::vector<uint16_t> q40(q.begin(), q.begin() + 40 * QW);
            // does row t use position j (a valid entry)?
            auto uses = [&](int64_t t, int32_t j) { for (int i = 0; i < RW; ++i) if (idx[t * RW + i] == j) return true; return false; };
            range("q x 2^100, K x 2^-110, V x 2^-30", false, scaled(q40, 100), scaled(kb, -110), scaled(vb, -30), kc, vc, none);
            range("V x 2^12 (rows up to ~2^14.5, inside FP16)", false, q40, kb, scaled(vb, 12), kc, vc, none);
            int32_t jv = idx[5 * RW + 100];  // a position row 5 uses
            { auto v2 = vb; v2[size_t(jv) * KW + 3] = f2bf(-100000.f);
              range("V outlier -1e5 (beyond FP16) at one position, KV head 0", false, q40, kb, v2, kc, vc, [&](int64_t t, int gg) { return gg == 0 && uses(t, jv); }); }
            { auto k2 = kb; k2[size_t(jv) * KW + 256 + 9] = 0xFF80;  // -inf, KV head 1
              range("-inf in K at one position, KV head 1", false, q40, k2, vb, kc, vc, [&](int64_t t, int gg) { return gg == 1 && uses(t, jv); }); }
            { auto q2 = q40; q2[11 * QW + 20 * D + 2] = 0x7FC0;  // NaN, row 11 head 20 (KV head 1)
              range("NaN in q (row 11, head 20)", false, q2, kb, vb, kc, vc, [](int64_t t, int gg) { return t == 11 && gg == 1; }); }
            { auto v3 = vc; v3[size_t(jv) * KW + 130] = 0xFF;  // E4M3 NaN, KV head 0
              range("E4M3 NaN byte in V at one position, KV head 0", true, q40, kb, vb, kc, v3, [&](int64_t t, int gg) { return gg == 0 && uses(t, jv); }); }
        }
        {   // negative rows
            auto neg = [&](const k9::Options &o, uint64_t sc, size_t scb) {
                ctx.begin();
                const cere_status e = k9::qsa_attention(ctx, dq.addr, QW, didx.addr, 40, 0, 0, 0, 0, dkc.addr, dvc.addr, 0, k9::kSoftmaxScale, sc, scb, dout.addr, QW, true, 0, o);
                ctx.submit_wait();
                return e;
            };
            k9::Options bad = copt; bad.prefill = k9::Prefill(9);
            const cere_status e0 = neg(copt, 0, 0), e1 = neg(copt, dcs.addr, 64), e2 = neg(bad, dcs.addr, csb), e3 = neg(copt, dcs.addr + 16, csb - 16);
            char msg[240]; std::snprintf(msg, sizeof msg, "coopmat rows path: no scratch %d, scratch below the flags %d, unknown Prefill %d, scratch not 256-aligned %d (want %d each) [coopmat]",
                                         int(e0), int(e1), int(e2), int(e3), int(CERE_ERR_ARGUMENT));
            check("p4 negative", e0 == CERE_ERR_ARGUMENT && e1 == CERE_ERR_ARGUMENT && e2 == CERE_ERR_ARGUMENT && e3 == CERE_ERR_ARGUMENT, msg);
        }
        ctx.free(dcs);
    }
    // ---- 2. qsa_topk
    {
        const int64_t S = 5000, R = 40, LD = S / 4;
        std::vector<int64_t> pos(R); for (int64_t t = 0; t < R; ++t) pos[t] = t < 4 ? int64_t(100 + 700 * t) : S - R + t;
        std::vector<float> lg(R * LD);
        for (int64_t t = 0; t < R; ++t) for (int64_t b = 0; b < LD; ++b) {
            float x = std::round(nd(g) * 8.f) / 8.f;
            if (g() % 50 == 0) x = (g() & 1) ? -0.f : 0.f;
            lg[t * LD + b] = b < (pos[t] + 1) / 4 ? x : -INFINITY;
        }
        auto dl = kt::upload(ctx, lg); auto dp = kt::upload(ctx, pos); auto dout = kt::make(ctx, R * RW * 4);
        std::vector<int32_t> want(R * RW); for (int64_t t = 0; t < R; ++t) ref_select(&lg[t * LD], pos[t], &want[t * RW]);
        for (int64_t rows : {int64_t(3), R}) for (int impl : {0, 1}) {
            std::memset(dout.map, 0x7F, size_t(R) * RW * 4);
            ctx.begin(); must(k9::qsa_topk(ctx, dl.addr, LD, dp.addr, rows, S, impl, dout.addr, opt), "qsa_topk"); ctx.barrier(); ctx.submit_wait();
            auto got = kt::host<int32_t>(dout, size_t(rows) * RW);
            int64_t bad = 0; for (size_t i = 0; i < got.size(); ++i) bad += got[i] != want[i];
            char msg[160]; std::snprintf(msg, sizeof msg, "rows %2ld impl %d: %ld of %zu indices differ from the host selection", (long)rows, impl, (long)bad, got.size());
            check("qsa_topk", bad == 0, msg);
        }
        ctx.begin(); const cere_status s2 = k9::qsa_topk(ctx, dl.addr, LD, dp.addr, 3, S, 2, dout.addr, opt); ctx.submit_wait();
        std::printf("%-26s %s  impl 2 (CUDA's cluster kernel): status %d (HIP: hipErrorNotSupported)\n", "qsa_topk impl 2", s2 == CERE_ERR_UNSUPPORTED ? "REFUSED" : "FAIL", int(s2));
        if (s2 != CERE_ERR_UNSUPPORTED) ++g_fail;
    }
    // ---- 3, 4. qsa_logits, qsa_select
    {
        const int64_t S = 5000, R = 3, LD = S / 4, NB = S / 4;
        std::vector<uint16_t> qi(R * 4 * 128), keys(NB * 128); for (auto &x : qi) x = f2bf(nd(g)); for (auto &x : keys) x = f2bf(nd(g));
        auto dqi = kt::upload(ctx, qi), dkeys = kt::upload(ctx, keys); auto dl = kt::make(ctx, R * LD * 4);
        std::vector<int64_t> pos = {2047, 3001, S - 1}; auto dp = kt::upload(ctx, pos);
        std::vector<float> l0;
        bool same = true; double worst = 0; int64_t inf_bad = 0;
        for (int impl : {0, 1, 2}) {
            std::memset(dl.map, 0x7F, size_t(R) * LD * 4);
            ctx.begin(); must(k9::qsa_logits(ctx, dqi.addr, dkeys.addr, 0, dp.addr, R, S, dl.addr, LD, impl, opt), "qsa_logits"); ctx.barrier(); ctx.submit_wait();
            auto l = kt::host<float>(dl, size_t(R) * LD);
            if (impl == 0) {
                l0 = l;
                for (int64_t t = 0; t < R; ++t) for (int64_t b = 0; b < LD; ++b) {
                    if (b >= (pos[t] + 1) / 4) { inf_bad += !(std::isinf(l[t * LD + b]) && l[t * LD + b] < 0); continue; }
                    double s = 0, sa = 0;
                    for (int h = 0; h < 4; ++h) { double d = 0, da = 0; for (int c = 0; c < 128; ++c) { const double pr = double(bf2f(qi[(t * 4 + h) * 128 + c])) * bf2f(keys[b * 128 + c]); d += pr; da += std::fabs(pr); }
                        if (d > 0) { s += d; } sa += da; }
                    worst = std::max(worst, std::fabs(l[t * LD + b] - s / k9::kScoreDivisor) / (sa / k9::kScoreDivisor + 1e-30));
                }
            } else same &= std::memcmp(l.data(), l0.data(), l.size() * 4) == 0;
        }
        char msg[200]; std::snprintf(msg, sizeof msg, "3 rows x 1,250 blocks: worst |err| / sum|terms| %.1e (1e-5), -inf past visible %s, impl 0/1/2 %s", worst, inf_bad ? "NO" : "yes", same ? "same bytes" : "DIFFER");
        check("qsa_logits", worst <= 1e-5 && !inf_bad && same, msg);
        const size_t sb = k9::select_scratch_bytes(R, S); auto ds = kt::make(ctx, sb); auto di = kt::make(ctx, R * RW * 4), dt = kt::make(ctx, R * RW * 4);
        ctx.begin();
        must(k9::qsa_select(ctx, dqi.addr, dkeys.addr, 0, dp.addr, R, S, ds.addr, sb, di.addr, opt), "qsa_select");
        must(k9::qsa_topk(ctx, dl.addr, LD, dp.addr, R, S, 1, dt.addr, opt), "qsa_topk");
        ctx.barrier(); ctx.submit_wait();
        check("qsa_select", kt::host<int32_t>(di, R * RW) == kt::host<int32_t>(dt, R * RW), "== qsa_topk(qsa_logits) for 3 rows at seq_len 5,000");
    }
    // ---- Vulkan row: qsa_select_step (device position, 16K context, slot map) == qsa_select at seq_len = position + 1
    {
        const int64_t CTX = 16384, NB = CTX / 4;
        std::vector<uint16_t> qi(4 * 128), keys(NB * 128); for (auto &x : qi) x = f2bf(nd(g)); for (auto &x : keys) x = f2bf(nd(g));
        std::vector<int64_t> slots(CTX); for (int64_t p = 0; p < CTX; ++p) slots[p] = ((p / 4) ^ 37) * 4 + p % 4;  // groups stay aligned
        auto dqi = kt::upload(ctx, qi), dkeys = kt::upload(ctx, keys), dsl = kt::upload(ctx, slots);
        const size_t sb = k9::select_scratch_bytes(1, CTX); auto ds = kt::make(ctx, sb);
        auto da = kt::make(ctx, RW * 4), db = kt::make(ctx, RW * 4), dfault = kt::make(ctx, 4);
        int64_t bad = 0, cases = 0;
        for (int64_t p : {int64_t(2050), int64_t(2051), int64_t(9000), CTX - 2, CTX - 1}) {
            std::vector<int64_t> pv{p}; auto dpv = kt::upload(ctx, pv);
            const k9::Bounds b{0, CTX, std::min(CTX, p + 1 + (p % 3)), dfault.addr};  // a context of p + 1 .. p + 3 positions
            ctx.begin();
            must(k9::qsa_select_step(ctx, dqi.addr, dkeys.addr, dsl.addr, b, dpv.addr, CTX, ds.addr, sb, da.addr, opt), "select_step");
            ctx.barrier();
            must(k9::qsa_select(ctx, dqi.addr, dkeys.addr, dsl.addr, 0, 1, p + 1, ds.addr, sb, db.addr, opt), "select");
            ctx.barrier(); ctx.submit_wait();
            bad += kt::host<int32_t>(da, RW) != kt::host<int32_t>(db, RW); ++cases;
            ctx.free(dpv);
        }
        // a device position past the context is clamped to context - 1 and sets the fault word
        std::vector<int64_t> pv{CTX + 5}; auto dpv = kt::upload(ctx, pv);
        const k9::Bounds b{0, CTX, CTX, dfault.addr};
        ctx.begin();
        must(k9::qsa_select_step(ctx, dqi.addr, dkeys.addr, dsl.addr, b, dpv.addr, CTX, ds.addr, sb, da.addr, opt), "select_step");
        ctx.barrier();
        must(k9::qsa_select(ctx, dqi.addr, dkeys.addr, dsl.addr, 0, 1, CTX, ds.addr, sb, db.addr, opt), "select");
        ctx.barrier(); ctx.submit_wait();
        const bool clamp_ok = kt::host<int32_t>(da, RW) == kt::host<int32_t>(db, RW);
        const uint32_t fault = kt::host<uint32_t>(dfault, 1)[0];
        char msg[220]; std::snprintf(msg, sizeof msg, "%ld positions (2,050 .. 16,383, slot map): %ld differ from qsa_select at seq_len p + 1; position 16,389 -> 16,383 %s, fault %u [Vulkan row]",
                                     (long)cases, (long)bad, clamp_ok ? "yes" : "NO", fault);
        check("qsa_select_step", bad == 0 && clamp_ok && fault == 1, msg);
    }
  }

    if (time) {   // ---- timing
        std::mt19937 g(53); std::normal_distribution<float> nd;
        const int64_t CTX = 16384, NB = CTX / 4, PR = 4096;
        std::vector<uint8_t> kc(CTX * KW), vc(CTX * KW);
        for (auto &b : kc) { b = uint8_t(g() % 0x78); if (g() & 1) b |= 0x80; }
        for (auto &b : vc) { b = uint8_t(g() % 0x78); if (g() & 1) b |= 0x80; }
        std::vector<uint16_t> kb(CTX * KW), vb(CTX * KW); for (auto &x : kb) x = f2bf(nd(g)); for (auto &x : vb) x = f2bf(nd(g));
        std::vector<uint16_t> q(PR * QW); for (auto &x : q) x = f2bf(nd(g) * 0.25f);
        std::vector<int32_t> idx(PR * RW), idx2k(RW), idx16(RW);
        for (int64_t t = 0; t < PR; ++t) timing_row(g, CTX - PR + t, &idx[t * RW]);
        timing_row(g, 2047, idx2k.data()); timing_row(g, CTX - 1, idx16.data());
        std::vector<uint16_t> qk(16 * k9::kQkWidth), keys(NB * 128), w(128), qi(PR * 4 * 128);
        for (auto &x : qk) { x = f2bf(nd(g)); }
        for (auto &x : keys) { x = f2bf(nd(g)); }
        for (auto &x : w) { x = f2bf(nd(g) * 0.1f); }
        for (auto &x : qi) { x = f2bf(nd(g)); }
        std::vector<float> cs(size_t(CTX) * 64);
        for (int64_t p = 0; p < CTX; ++p) for (int i = 0; i < 32; ++i) { const double a = double(p) * std::pow(1e7, -2.0 * i / 64); cs[p * 64 + i] = float(std::cos(a)); cs[p * 64 + 32 + i] = float(std::sin(a)); }
        uint64_t sum = 1; for (auto x : idx) sum = sum * 1000003u + uint32_t(x); for (auto x : q) sum = sum * 1000003u + x; for (auto x : kc) sum = sum * 1000003u + x;
        std::printf("timing data checksum %016llx\n", (unsigned long long)sum);
        auto dkc = kt::upload(ctx, kc), dvc = kt::upload(ctx, vc), dkb = kt::upload(ctx, kb), dvb = kt::upload(ctx, vb), dq = kt::upload(ctx, q);
        auto didx = kt::upload(ctx, idx), di2 = kt::upload(ctx, idx2k), di16 = kt::upload(ctx, idx16);
        auto dqk = kt::upload(ctx, qk), dkeys = kt::upload(ctx, keys), dw = kt::upload(ctx, w), dcs = kt::upload(ctx, cs), dqi = kt::upload(ctx, qi);
        auto dout = kt::make(ctx, PR * QW * 2), dring = kt::make(ctx, 4 * 128 * 2), dqidx = kt::make(ctx, 16 * 4 * 128 * 2), dsel = kt::make(ctx, PR * RW * 4);
        const size_t asb = k9::attention_scratch_bytes(16), ssb = std::max(k9::select_scratch_bytes(PR, CTX), k9::select_scratch_bytes(16, CTX));
        auto das = kt::make(ctx, asb), dss = kt::make(ctx, ssb);
        std::vector<int64_t> p2k{2047}, p16{CTX - 1}, pv8(8); for (int i = 0; i < 8; ++i) pv8[i] = CTX - 8 + i;
        auto dp2k = kt::upload(ctx, p2k), dp16 = kt::upload(ctx, p16), dpv8 = kt::upload(ctx, pv8);
        // 3 untimed calls (stamped: the per-call estimate), then 15 calls each between its own pair of timestamps with a
        // barrier after each, submitted in command buffers of at most ~250 ms of GPU work (P4-A: the 4,096-row FP32 calls
        // made one 1.1 s command buffer; the launch layer's budget is 1,000 ms per submit)
        std::vector<double> last;  // the latest timeit's samples, issue order (the --samples line)
        auto timeit = [&](auto &&rec) {
            const int reps = 15;
            ctx.begin();
            for (int i = 0; i < 3; ++i) { ctx.stamp(2 * i); rec(); ctx.stamp(2 * i + 1); ctx.barrier(); }
            ctx.submit_wait();
            const double est = std::max(1e-3, ctx.stamp_ms(4, 5));
            const int per = std::max(1, std::min(reps, int(250.0 / est)));
            std::vector<double> t;
            for (int done = 0; done < reps;) {
                const int n = std::min(per, reps - done);
                ctx.begin();
                for (int i = 0; i < n; ++i) { ctx.stamp(2 * i); rec(); ctx.stamp(2 * i + 1); ctx.barrier(); }
                ctx.submit_wait();
                for (int i = 0; i < n; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
                done += n;
            }
            last = t;
            std::sort(t.begin(), t.end());
            return Stat{t[t.size() / 2], t.front(), t.back()};
        };
        auto attn = [&](bool fp8, uint64_t qq, uint64_t ii, int64_t rows) {
            // the model's K9 scratch: the attention partials (<= 16 rows) or the dead select logits tile (the FP8 copy)
            const uint64_t sc = rows <= 16 ? das.addr : dss.addr; const size_t scb = rows <= 16 ? asb : ssb;
            must(fp8 ? k9::qsa_attention_heads(ctx, qq, QW, ii, rows, 0, 0, 0, 0, dkc.addr, dvc.addr, 0, k9::kSoftmaxScale, sc, scb, dout.addr, QW, true, 0, theads, opt)
                     : k9::qsa_attention_heads(ctx, qq, QW, ii, rows, dkb.addr, KW, dvb.addr, KW, 0, 0, 0, k9::kSoftmaxScale, sc, scb, dout.addr, QW, true, 0, theads, opt), "attn");
        };
        auto step = [&](bool fp8, uint64_t dpos, int64_t p, uint64_t ii_unused) {
            (void)ii_unused;
            const k9::Bounds b{0, CTX, p + 1, 0};
            must(k9::indexer_qk(ctx, dqk.addr, k9::kQkWidth, dw.addr, 1e-6f, dcs.addr, b, 0, p, 1, dqidx.addr), "iq");
            must(k9::qsa_compress_store_step(ctx, dqk.addr + 512 * 2, dw.addr, 1e-6f, dcs.addr, b, 0, dpos, dring.addr, dkeys.addr), "cs");
            ctx.barrier();
            must(k9::qsa_select_step(ctx, dqidx.addr, dkeys.addr, 0, b, dpos, CTX, dss.addr, ssb, dsel.addr, opt), "sel");
            ctx.barrier();
            attn(fp8, dq.addr, dsel.addr, 1);
        };
        // --samples / --identity: the shape key is the name with every run of other characters as one '_'; the output
        // bytes are dout's first 4,096 rows (each call's rows; fewer-row calls leave the rest as the previous call's)
        auto pr = [&](const char *name, Stat s) {
            std::printf("TIME %-44s median %8.4f ms  (min %.4f, max %.4f)\n", name, s.med, s.mn, s.mx);
            std::string key;
            for (const char *c = name; *c; ++c) {
                const bool w = std::isalnum(static_cast<unsigned char>(*c)) != 0;
                if (w) key += *c; else if (!key.empty() && key.back() != '_') key += '_';
            }
            while (!key.empty() && key.back() == '_') key.pop_back();
            kt::samples(key, "warm", last);
            kt::identity(key, kt::host<uint8_t>(dout, size_t(PR) * QW * 2));
        };
        for (bool fp8 : {false, true}) {
            const char *f = fp8 ? "FP8" : "BF16";
            char n[96];
            std::snprintf(n, sizeof n, "attention 1 row, 2K ctx, %s", f); pr(n, timeit([&] { attn(fp8, dq.addr, di2.addr, 1); }));
            std::snprintf(n, sizeof n, "attention 1 row, 16K ctx, %s", f); pr(n, timeit([&] { attn(fp8, dq.addr, di16.addr, 1); }));
            std::snprintf(n, sizeof n, "K9 decode step, 2K ctx, %s", f); pr(n, timeit([&] { step(fp8, dp2k.addr, 2047, 0); }));
            std::snprintf(n, sizeof n, "K9 decode step, 16K ctx, %s", f); pr(n, timeit([&] { step(fp8, dp16.addr, CTX - 1, 0); }));
        }
        pr("select 1 row (select_step), 16K ctx", timeit([&] { const k9::Bounds b{0, CTX, CTX, 0}; must(k9::qsa_select_step(ctx, dqi.addr, dkeys.addr, 0, b, dp16.addr, CTX, dss.addr, ssb, dsel.addr, opt), "sel"); }));
        pr("verify 8 rows: select + attention, 16K, FP8", timeit([&] {
            must(k9::qsa_select(ctx, dqi.addr, dkeys.addr, 0, dpv8.addr, 8, CTX, dss.addr, ssb, dsel.addr, opt), "sel8");
            ctx.barrier(); attn(true, dq.addr, dsel.addr, 8); }));
        pr("verify 8 rows: attention only, 16K, FP8", timeit([&] { attn(true, dq.addr, didx.addr + uint64_t(PR - 8) * RW * 4, 8); }));
        pr("prefill 4,096 rows attention, 16K, FP8", timeit([&] { attn(true, dq.addr, didx.addr, PR); }));
        pr("prefill 4,096 rows attention, 16K, BF16", timeit([&] { attn(false, dq.addr, didx.addr, PR); }));
        pr("prefill 4,096 rows attention, 16K, FP8 no copy", timeit([&] {
            must(k9::qsa_attention_heads(ctx, dq.addr, QW, didx.addr, PR, 0, 0, 0, 0, dkc.addr, dvc.addr, 0, k9::kSoftmaxScale, 0, 0, dout.addr, QW, true, 0, theads, opt), "attn"); }));
        pr("prefill 512 rows attention, 16K, FP8", timeit([&] { attn(true, dq.addr, didx.addr, 512); }));
        pr("prefill 4,096 rows select, 16K", timeit([&] { must(k9::qsa_select(ctx, dqi.addr, dkeys.addr, 0, 0, PR, CTX, dss.addr, ssb, dsel.addr, opt), "selP"); }));
        // P4-A: the coopmat rows path on the same calls, and the S3 3,000-token prompt's sparse attention shape (one
        // 3,000-row BF16-rows call, positions 0..2,999: every position for rows below 2,048, 512 random blocks + tail above)
        {
            k9::Options copt = opt; copt.prefill = k9::Prefill::kCoopmat;
            auto attn_o = [&](const k9::Options &o, bool fp8, uint64_t ii, int64_t rows, uint64_t kk, uint64_t vv) {
                must(fp8 ? k9::qsa_attention_heads(ctx, dq.addr, QW, ii, rows, 0, 0, 0, 0, kk, vv, 0, k9::kSoftmaxScale, dss.addr, ssb, dout.addr, QW, true, 0, theads, o)
                         : k9::qsa_attention_heads(ctx, dq.addr, QW, ii, rows, kk, KW, vv, KW, 0, 0, 0, k9::kSoftmaxScale, dss.addr, ssb, dout.addr, QW, true, 0, theads, o), "attn_o");
            };
            pr("p4 coopmat: prefill 4,096 rows, 16K, FP8", timeit([&] { attn_o(copt, true, didx.addr, PR, dkc.addr, dvc.addr); }));
            pr("p4 coopmat: prefill 4,096 rows, 16K, BF16", timeit([&] { attn_o(copt, false, didx.addr, PR, dkb.addr, dvb.addr); }));
            pr("p4 coopmat: prefill 512 rows, 16K, FP8", timeit([&] { attn_o(copt, true, didx.addr, 512, dkc.addr, dvc.addr); }));
            const int64_t S3 = 3000;
            std::vector<int32_t> is3(S3 * RW);
            for (int64_t t = 0; t < S3; ++t) timing_row(g, t, &is3[t * RW]);
            auto dis3 = kt::upload(ctx, is3);
            pr("S3 3,000-token prompt, BF16 rows, FP32", timeit([&] { attn_o(opt, false, dis3.addr, S3, dkb.addr, dvb.addr); }));
            pr("p4 coopmat: S3 3,000-token prompt, BF16 rows", timeit([&] { attn_o(copt, false, dis3.addr, S3, dkb.addr, dvb.addr); }));
            pr("S3 3,000-token prompt as FP8 cache, FP32", timeit([&] { attn_o(opt, true, dis3.addr, S3, dkc.addr, dvc.addr); }));
            pr("p4 coopmat: S3 3,000-token prompt as FP8 cache", timeit([&] { attn_o(copt, true, dis3.addr, S3, dkc.addr, dvc.addr); }));
            ctx.free(dis3);
        }
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
