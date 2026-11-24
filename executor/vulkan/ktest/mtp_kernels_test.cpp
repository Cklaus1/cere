// VulkanAmd P3 (T3): contract rows of the MTP-local kernels on Vulkan (src/kernels/mtp_kernels_vk.*), from their CUDA
// definitions (executor/cuda/src/model/mtp.cu:43-115; the CUDA suite has no kernel-level rows for them, they are covered
// by the model tests, so the rows are the definitions' own: K1's FP64 bound for the norm, exactness for the rest):
//  norm_hidden: 1. the dispatch == vkk1::rmsnorm (Gemma, weight_groups 1) bit for bit at the dims vkk1::rmsnorm accepts
//     (2,560 and 8,192; it refuses the MTP's 10,240); 2. the MTP shape [5, 10,240] vs FP64 Gemma RMSNorm, rel-L2 <= 4e-3
//     (the K1 / CUDA FLOOR_OP bound, the HIP k1 test's data scales); 3. row independence (a 1-row call == that row of the
//     5-row call), in place (y == x) identical, repeat identical;
//  add_embedding: 4. bit-exact against the host rule bf16_rne(float(h) + float(e)) (NaN 0x7FFF) over every BF16 pattern
//     of hyper (65,536 x 10 placements) against random / special e (+-0, +-Inf, NaN, denormals, max), and over random
//     near-tie sums; untouched bytes past the call; 5. repeat identical;
//  argmax_map: 6. bit-exact against the host definition (lowest index of the maximum, NaN as -inf, -0 == +0) on rows of
//     65,536 hot logits (random, ties, NaN, all NaN, all -inf, +Inf, -0 / +0 ties) through a random map, on the vector
//     and the scalar path (count 65,535, a 2-byte-offset base) and small counts; 7. row independence and repeat;
//  8. negative: refusals (rows, count, alignment).
// Then decode-shape timing (the K1 batch method: us per call = GPU time of 100 dependent calls with a barrier after each
// / 100, median of 7 batches after 3 untimed). Built with -DT3_HIP_TIMING under hipcc it is instead the same-method HIP
// timing of the three CUDA kernels, hipified (build.d/p3_kernels.sh extracts them verbatim from gen/model/mtp.hip).
#ifdef T3_HIP_TIMING
#include <hip/hip_runtime.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#ifndef T3_HIP_TIMING
#include "launch/vk_launch_testing.h"
#include "kernels/k1_primitives_vk.h"
#include "kernels/mtp_kernels_vk.h"
#include "launch/vk_launch.h"
#endif

static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) {  // round to nearest even, NaN 0x7FFF (the CUDA __float2bfloat16 rule)
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fff;
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static std::vector<uint16_t> randbf(size_t n, std::mt19937 &g, float scale = 1.f) {
    std::normal_distribution<float> d(0.f, scale); std::vector<uint16_t> v(n);
    for (auto &x : v) x = f2bf(d(g));
    return v;
}
static double median(std::vector<double> t) { std::sort(t.begin(), t.end()); return t[t.size() / 2]; }

constexpr int HID = 2560, HCW = 4 * HID;
constexpr uint32_t HOT = 65536;  // hot lm_head rows (draft/token_map.u32le of the S3 artifact)
constexpr int kBatch = 100;
static const char *kShapes[] = {"norm_hidden 1 x 10240", "norm_hidden 4 x 10240", "add_embedding 1 x 4 x 2560", "add_embedding 4 x 4 x 2560",
                                "argmax_map 1 x 65536", "argmax_map 4 x 65536"};

#ifndef T3_HIP_TIMING
using namespace cere;
static int g_fail = 0;
static void report(const char *name, bool ok, const char *detail) {
    std::printf("%-44s %s  %s\n", name, ok ? "PASS" : "FAIL", detail);
    if (!ok) ++g_fail;
}
static vk::Buffer make(vk::Context &ctx, uint64_t bytes) {
    vk::Buffer b;
    if (!ctx.alloc(bytes + 256, b)) std::exit(2);
    std::memset(b.map, 0, bytes + 256);
    return b;
}
template <class T> static vk::Buffer up(vk::Context &ctx, const std::vector<T> &h) {
    vk::Buffer b = make(ctx, h.size() * sizeof(T)); std::memcpy(b.map, h.data(), h.size() * sizeof(T)); return b;
}
template <class T> static std::vector<T> host(const vk::Buffer &b, size_t n, size_t off = 0) {
    std::vector<T> h(n); std::memcpy(h.data(), static_cast<const uint8_t *>(b.map) + off, n * sizeof(T)); return h;
}
static void must(cere_status s, const char *what) { if (s != CERE_OK) { std::printf("FATAL %s returned %d\n", what, int(s)); std::exit(2); } }
static void sync(vk::Context &ctx, const std::function<cere_status()> &f, const char *what) { ctx.begin(); must(f(), what); ctx.barrier(); ctx.submit_wait(); }

static uint32_t argmax_ref(const uint16_t *row, uint32_t count) {  // mtp.cu:82-84: lowest index of the maximum, NaN as -inf
    uint32_t at = 0; float best = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const float raw = bf2f(row[i]); const float v = std::isnan(raw) ? -INFINITY : raw;
        if (i == 0 || v > best) { best = v; at = i; }
    }
    return at;
}

int main(int argc, char **argv) {
    auto ctx_owner = cere::vk::testing::create_or_die(argc > 1 ? argv[1] : "build/vulkan-t3/spv");
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str());
    std::mt19937 g(1234);
    const float eps = 1e-6f;

    // ---- norm_hidden
    {
        for (int dim : {2560, 8192}) {  // 1. the dispatch == vkk1::rmsnorm
            const int64_t rows = dim == 2560 ? 7 : 3;
            auto x = randbf(size_t(rows) * dim, g, 2.f), w = randbf(dim, g, 0.3f);
            auto dx = up(ctx, x), dw = up(ctx, w), y1 = make(ctx, size_t(rows) * dim * 2), y2 = make(ctx, size_t(rows) * dim * 2);
            sync(ctx, [&] { return vkmtp::detail::gemma_rows(ctx, dx.addr, dw.addr, eps, rows, dim, y1.addr); }, "gemma_rows");
            sync(ctx, [&] { return vkk1::rmsnorm(ctx, dx.addr, dim, dw.addr, 1, true, eps, rows, dim, y2.addr, dim); }, "rmsnorm");
            const bool eq = host<uint16_t>(y1, size_t(rows) * dim) == host<uint16_t>(y2, size_t(rows) * dim);
            char name[64]; std::snprintf(name, sizeof name, "norm_hidden dispatch == vkk1::rmsnorm @%d", dim);
            report(name, eq, eq ? "bit-identical (Gemma, weight_groups 1)" : "DIFFERS");
            for (auto *b : {&dx, &dw, &y1, &y2}) ctx.free(*b);
        }
        const int64_t M = 5;
        auto x = randbf(size_t(M) * HCW, g, 2.f), w = randbf(HCW, g, 0.3f);
        auto dx = up(ctx, x), dw = up(ctx, w), y = make(ctx, size_t(M) * HCW * 2), y1 = make(ctx, size_t(HCW) * 2);
        sync(ctx, [&] { return vkmtp::norm_hidden(ctx, dx.addr, dw.addr, eps, M, y.addr); }, "norm_hidden");
        auto out = host<uint16_t>(y, size_t(M) * HCW);
        double num = 0, den = 0;
        for (int64_t r = 0; r < M; ++r) {
            double ss = 0; for (int i = 0; i < HCW; ++i) { const double v = bf2f(x[r * HCW + i]); ss += v * v; }
            const double rs = 1.0 / std::sqrt(ss / HCW + eps);
            for (int i = 0; i < HCW; ++i) {
                const double ref = bf2f(x[r * HCW + i]) * rs * (1.0 + bf2f(w[i]));
                const double d = bf2f(out[r * HCW + i]) - ref; num += d * d; den += ref * ref;
            }
        }
        const double e = std::sqrt(num / den);
        sync(ctx, [&] { return vkmtp::norm_hidden(ctx, dx.addr, dw.addr, eps, M, y.addr); }, "norm_hidden again");
        const bool rep = host<uint16_t>(y, size_t(M) * HCW) == out;
        char d[160]; std::snprintf(d, sizeof d, "[5, 10240] rel-L2 %.2e vs FP64 (limit 4e-3); repeat bit-identical %s", e, rep ? "yes" : "NO");
        report("norm_hidden vs FP64 (MTP shape)", e <= 4e-3 && rep, d);
        bool rows_ok = true;
        for (int64_t r = 0; r < M; ++r) {
            sync(ctx, [&] { return vkmtp::norm_hidden(ctx, dx.addr + uint64_t(r) * HCW * 2, dw.addr, eps, 1, y1.addr); }, "norm_hidden 1");
            rows_ok = rows_ok && std::memcmp(host<uint16_t>(y1, HCW).data(), &out[r * HCW], HCW * 2) == 0;
        }
        sync(ctx, [&] { return vkmtp::norm_hidden(ctx, dx.addr, dw.addr, eps, M, dx.addr); }, "norm_hidden in place");
        const bool inplace = host<uint16_t>(dx, size_t(M) * HCW) == out;
        report("norm_hidden rows / in place", rows_ok && inplace, rows_ok && inplace ? "each 1-row call == its row of the 5-row call; y == x identical"
                                                                                    : "a 1-row call or the in-place call DIFFERS");
        for (auto *b : {&dx, &dw, &y, &y1}) ctx.free(*b);
    }

    // ---- add_embedding
    {
        const int64_t M = 64;  // 64 x 10,240 = every BF16 pattern 10 times
        std::vector<uint16_t> h(size_t(M) * HCW);
        for (size_t i = 0; i < h.size(); ++i) h[i] = uint16_t(i * 40503u);  // odd multiplier: a permutation of each 65,536 block
        std::vector<uint16_t> e = randbf(size_t(M) * HID, g, 3.f);
        const uint16_t specials[] = {0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0xffc1, 0x0001, 0x8001, 0x007f, 0x0080, 0x7f7f, 0xff7f, 0x3f80, 0xbf80, 0x0400};
        std::uniform_int_distribution<uint32_t> pick(0, 14);
        for (size_t i = 0; i < e.size(); i += 3) e[i] = specials[pick(g)];
        for (size_t i = 1; i < e.size(); i += 7) e[i] = uint16_t(g());  // arbitrary patterns
        auto dh = up(ctx, h), de = up(ctx, e);
        std::memset(static_cast<uint8_t *>(dh.map) + h.size() * 2, 0xEE, 256);
        sync(ctx, [&] { return vkmtp::add_embedding(ctx, dh.addr, de.addr, M); }, "add_embedding");
        auto out = host<uint16_t>(dh, h.size());
        int64_t bad = 0, nan_out = 0, denorm_in = 0;
        for (int64_t r = 0; r < M; ++r)
            for (int s = 0; s < 4; ++s)
                for (int c = 0; c < HID; ++c) {
                    const size_t i = size_t(r) * HCW + size_t(s) * HID + c;
                    const float a = bf2f(h[i]), b = bf2f(e[size_t(r) * HID + c]);
                    const uint16_t want = f2bf(a + b);
                    bad += out[i] != want; nan_out += want == 0x7fff;
                    denorm_in += std::fpclassify(a + b) == FP_SUBNORMAL;
                }
        auto tail = host<uint8_t>(dh, 256, h.size() * 2);
        const bool tail_ok = std::all_of(tail.begin(), tail.end(), [](uint8_t x) { return x == 0xEE; });
        char d[200];
        std::snprintf(d, sizeof d, "%ld of %zu differ from bf16_rne(h + e) (all 65,536 h patterns x 10; %ld NaN results, %ld denormal sums); tail untouched %s",
                      long(bad), h.size(), long(nan_out), long(denorm_in), tail_ok ? "yes" : "NO");
        report("add_embedding exact (all BF16 patterns)", bad == 0 && tail_ok, d);
        // near-tie sums: e within a few exponents of h, so the FP32 sum has bits below BF16's 8 and RNE decides
        std::vector<uint16_t> h2 = randbf(size_t(M) * HCW, g, 1.f), e2(size_t(M) * HID);
        for (auto &x : e2) { const float v = std::ldexp(float(int(g() % 512) - 256) / 256.f, -int(g() % 10)); x = f2bf(v); }
        auto dh2 = up(ctx, h2), de2 = up(ctx, e2);
        sync(ctx, [&] { return vkmtp::add_embedding(ctx, dh2.addr, de2.addr, M); }, "add_embedding 2");
        auto out2 = host<uint16_t>(dh2, h2.size());
        int64_t bad2 = 0, ties = 0;
        for (int64_t r = 0; r < M; ++r) for (int s = 0; s < 4; ++s) for (int c = 0; c < HID; ++c) {
            const size_t i = size_t(r) * HCW + size_t(s) * HID + c;
            const float sum = bf2f(h2[i]) + bf2f(e2[size_t(r) * HID + c]);
            uint32_t u; std::memcpy(&u, &sum, 4); ties += (u & 0xffffu) == 0x8000u;
            bad2 += out2[i] != f2bf(sum);
        }
        sync(ctx, [&] { std::memcpy(dh2.map, h2.data(), h2.size() * 2); return vkmtp::add_embedding(ctx, dh2.addr, de2.addr, M); }, "add_embedding again");
        const bool rep = host<uint16_t>(dh2, h2.size()) == out2;
        std::snprintf(d, sizeof d, "%ld of %zu differ (random h, e within 2^-9..1: %ld exact ties to even); repeat identical %s", long(bad2), h2.size(), long(ties),
                      rep ? "yes" : "NO");
        report("add_embedding exact (near-tie sums)", bad2 == 0 && rep, d);
        for (auto *b : {&dh, &de, &dh2, &de2}) ctx.free(*b);
    }

    // ---- argmax_map
    {
        const int R = 10;
        auto lg = randbf(size_t(R) * HOT, g, 3.f);
        auto row = [&](int r) { return &lg[size_t(r) * HOT]; };
        row(1)[777] = f2bf(40.f); row(1)[60000] = f2bf(40.f); row(1)[5] = f2bf(40.f);                 // tie: lowest (5)
        row(2)[100] = 0x7fc0; row(2)[9] = 0xffc0; row(2)[30000] = f2bf(30.f);                        // NaN ignored
        for (uint32_t i = 0; i < HOT; ++i) { row(3)[i] = (i % 3) ? 0x7fc0 : 0xffc1; }                    // all NaN -> 0
        for (uint32_t i = 0; i < HOT; ++i) { row(4)[i] = 0xff80; }                                        // all -inf -> 0
        for (uint32_t i = 0; i < HOT; ++i) { row(5)[i] = (i % 2) ? 0x7fc0 : 0xff80; } row(5)[4001] = 0xff7f;  // NaN / -inf, one finite
        row(6)[65535] = 0x7f80; row(6)[12] = 0x7fc0;                                                  // +inf at the last index
        for (uint32_t i = 0; i < HOT; ++i) { row(7)[i] = 0xbf80; } row(7)[300] = 0x8000; row(7)[200] = 0x0000;  // -0 / +0 tie: 200
        for (uint32_t i = 0; i < HOT; ++i) { row(8)[i] = 0xbf80; } row(8)[200] = 0x8000; row(8)[300] = 0x0000;  // tie: 200 (the -0)
        for (uint32_t i = 0; i < HOT; ++i) { row(9)[i] = f2bf(1.f); }                                     // all equal -> 0
        std::vector<uint32_t> map(HOT); std::iota(map.begin(), map.end(), 1000u); std::shuffle(map.begin(), map.end(), g);
        auto dl = up(ctx, lg), dm = up(ctx, map), dd = make(ctx, 4 * 64);
        sync(ctx, [&] { return vkmtp::argmax_map(ctx, dl.addr, HOT, dm.addr, R, dd.addr); }, "argmax_map");
        auto got = host<uint32_t>(dd, R);
        int bad = 0; std::vector<uint32_t> want(R);
        for (int r = 0; r < R; ++r) { want[r] = map[argmax_ref(row(r), HOT)]; bad += got[r] != want[r]; }
        const uint32_t expect_idx[R] = {argmax_ref(row(0), HOT), 5, 30000, 0, 0, 4001, 65535, 200, 200, 0};
        int bad_idx = 0; for (int r = 0; r < R; ++r) bad_idx += argmax_ref(row(r), HOT) != expect_idx[r];
        char d[200];
        std::snprintf(d, sizeof d, "%d of %d rows differ from map[host argmax] (random, tie, NaN, all NaN, all -inf, NaN/-inf + 1 finite, +inf, -0/+0 x2, all equal); "
                      "host ref vs hand indices: %d differ", bad, R, bad_idx);
        report("argmax_map exact, 65,536 (vector path)", bad == 0 && bad_idx == 0, d);
        // row independence + repeat
        bool ind = true;
        for (int r = 0; r < R; ++r) {
            sync(ctx, [&] { return vkmtp::argmax_map(ctx, dl.addr + uint64_t(r) * HOT * 2, HOT, dm.addr, 1, dd.addr + 64); }, "argmax_map 1");
            ind = ind && host<uint32_t>(dd, 1, 64)[0] == got[r];
        }
        sync(ctx, [&] { return vkmtp::argmax_map(ctx, dl.addr, HOT, dm.addr, R, dd.addr); }, "argmax_map again");
        ind = ind && host<uint32_t>(dd, R) == got;
        report("argmax_map rows / repeat", ind, ind ? "each 1-row call == its row of the 10-row call; repeat identical" : "DIFFERS");
        // scalar path: count 65,535 (odd) and a base offset by 2 bytes; small counts
        int bad_s = 0, tried = 0;
        for (uint32_t count : {65535u, 1u, 7u, 1000u, 8u}) {
            const int rr = 6;
            std::vector<uint16_t> l2 = randbf(size_t(rr) * count + 1, g, 2.f);
            if (count >= 8) { l2[1 + 3] = f2bf(99.f); l2[1 + count - 1] = f2bf(99.f); l2[1 + count + 2] = 0x7fc0; }
            auto d2 = up(ctx, l2);
            for (int off : {0, 1}) {  // off 1: base 2-byte aligned only -> scalar path
                sync(ctx, [&] { return vkmtp::argmax_map(ctx, d2.addr + uint64_t(off) * 2, count, dm.addr, rr, dd.addr); }, "argmax_map scalar");
                auto o = host<uint32_t>(dd, rr);
                for (int r = 0; r < rr; ++r) { ++tried; bad_s += o[r] != map[argmax_ref(&l2[size_t(off) + size_t(r) * count], count)]; }
            }
            ctx.free(d2);
        }
        std::snprintf(d, sizeof d, "%d of %d rows differ (counts 65,535 / 1 / 7 / 1,000 / 8, base aligned and +2 bytes, ties, NaN)", bad_s, tried);
        report("argmax_map exact, scalar path + small", bad_s == 0, d);
        // negative
        ctx.begin();
        bool ok = vkmtp::argmax_map(ctx, dl.addr, 0, dm.addr, 1, dd.addr) == CERE_ERR_ARGUMENT &&
                  vkmtp::argmax_map(ctx, dl.addr, HOT, dm.addr, 0, dd.addr) == CERE_ERR_ARGUMENT &&
                  vkmtp::argmax_map(ctx, dl.addr, HOT, dm.addr, 65536, dd.addr) == CERE_ERR_ARGUMENT &&
                  vkmtp::argmax_map(ctx, dl.addr + 1, HOT, dm.addr, 1, dd.addr) == CERE_ERR_ARGUMENT &&
                  vkmtp::argmax_map(ctx, dl.addr, HOT, dm.addr + 2, 1, dd.addr) == CERE_ERR_ARGUMENT &&
                  vkmtp::add_embedding(ctx, dl.addr + 8, dl.addr, 1) == CERE_ERR_ARGUMENT &&
                  vkmtp::add_embedding(ctx, dl.addr, dl.addr, 0) == CERE_ERR_ARGUMENT &&
                  vkmtp::add_embedding(ctx, dl.addr, dl.addr, 209716) == CERE_ERR_ARGUMENT &&
                  vkmtp::norm_hidden(ctx, dl.addr + 8, dm.addr, eps, 1, dd.addr) == CERE_ERR_ARGUMENT &&
                  vkmtp::norm_hidden(ctx, dl.addr, dm.addr, eps, 0, dd.addr) == CERE_ERR_ARGUMENT &&
                  vkmtp::norm_hidden(ctx, dl.addr, dm.addr, eps, 65536, dd.addr) == CERE_ERR_ARGUMENT &&
                  vkmtp::detail::gemma_rows(ctx, dl.addr, dm.addr, eps, 1, 100, dd.addr) == CERE_ERR_ARGUMENT;
        ctx.submit_wait();
        report("negative: refusals", ok, "count 0, rows 0 / 65,536, misaligned logits / map / hyper / x, rows * 20,480 B >= 4 GiB, dim % 8: ARGUMENT");
        for (auto *b : {&dl, &dm, &dd}) ctx.free(*b);
    }
    std::printf("contract rows: %s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);

    {   // decode-shape timing (K1 batch method)
        std::mt19937 gt(7);
        auto x = up(ctx, randbf(4 * HCW, gt, 2.f)), w = up(ctx, randbf(HCW, gt, 0.3f)), y = make(ctx, 4 * HCW * 2);
        auto hy = up(ctx, randbf(4 * HCW, gt)), e = up(ctx, randbf(4 * HID, gt));
        auto lg = up(ctx, randbf(size_t(4) * HOT, gt, 3.f)); std::vector<uint32_t> map(HOT); std::iota(map.begin(), map.end(), 0u);
        auto dm = up(ctx, map), dd = make(ctx, 64);
        std::vector<std::function<cere_status()>> calls = {
            [&] { return vkmtp::norm_hidden(ctx, x.addr, w.addr, 1e-6f, 1, y.addr); },
            [&] { return vkmtp::norm_hidden(ctx, x.addr, w.addr, 1e-6f, 4, y.addr); },
            [&] { return vkmtp::add_embedding(ctx, hy.addr, e.addr, 1); },
            [&] { return vkmtp::add_embedding(ctx, hy.addr, e.addr, 4); },
            [&] { return vkmtp::argmax_map(ctx, lg.addr, HOT, dm.addr, 1, dd.addr); },
            [&] { return vkmtp::argmax_map(ctx, lg.addr, HOT, dm.addr, 4, dd.addr); }};
        std::printf("\ndecode-shape timing: us per call = (GPU time between two timestamps around %d dependent calls, a barrier after each) / %d,\n"
                    "median of 7 such batches [min..max]; 3 untimed warm-up batches first\n", kBatch, kBatch);
        for (size_t i = 0; i < calls.size(); ++i) {
            const int reps = 10;
            ctx.begin();
            for (int c = 0; c < reps; ++c) {
                ctx.stamp(2 * c);
                for (int k = 0; k < kBatch; ++k) { must(calls[i](), "call"); ctx.barrier(); }
                ctx.stamp(2 * c + 1);
            }
            ctx.submit_wait();
            std::vector<double> t; for (int c = 3; c < reps; ++c) t.push_back(ctx.stamp_ms(2 * c, 2 * c + 1) * 1e3 / kBatch);
            std::printf("TIME vulkan | %-30s | %9.2f us [%.2f..%.2f]\n", kShapes[i], median(t), *std::min_element(t.begin(), t.end()), *std::max_element(t.begin(), t.end()));
        }
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

#else  // T3_HIP_TIMING: the three hipified CUDA kernels (verbatim from gen/model/mtp.hip), same shapes, hipEvents per batch
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
using bf16 = __hip_bfloat16;
constexpr uint32_t kHidden = HID, kHcHidden = HCW, kNormThreads = 256, kArgmaxThreads = 1024, kAddThreads = 256;
#include "hip_mtp_kernels.inc"
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
static bf16 *B(uint16_t *p) { return reinterpret_cast<bf16 *>(p); }
int main() {
    hipDeviceProp_t prop; HCK(hipGetDeviceProperties(&prop, 0)); std::printf("device %s\n", prop.gcnArchName);
    std::mt19937 gt(7);
    uint16_t *x = dev(randbf(4 * HCW, gt, 2.f)), *w = dev(randbf(HCW, gt, 0.3f)), *y; HCK(hipMalloc(&y, 4 * HCW * 2));
    uint16_t *hy = dev(randbf(4 * HCW, gt)), *e = dev(randbf(4 * HID, gt));
    uint16_t *lg = dev(randbf(size_t(4) * HOT, gt, 3.f)); std::vector<uint32_t> map(HOT); std::iota(map.begin(), map.end(), 0u);
    uint32_t *dm = dev(map), *dd; HCK(hipMalloc(&dd, 64));
    HCK(hipDeviceSynchronize());
    auto norm = [&](uint32_t m, hipStream_t s) { norm_hidden_kernel<<<m, kNormThreads, 0, s>>>(B(x), B(w), 1e-6f, B(y)); return hipGetLastError(); };
    auto add = [&](uint64_t m, hipStream_t s) {
        const uint64_t el = m * kHcHidden;
        add_embedding_kernel<<<uint32_t((el + kAddThreads - 1) / kAddThreads), kAddThreads, 0, s>>>(B(hy), B(e), el); return hipGetLastError(); };
    auto amax = [&](uint32_t c, hipStream_t s) { argmax_map_kernel<<<c, kArgmaxThreads, 0, s>>>(B(lg), HOT, dm, dd); return hipGetLastError(); };
    std::vector<std::function<hipError_t(hipStream_t)>> calls = {
        [&](hipStream_t s) { return norm(1, s); }, [&](hipStream_t s) { return norm(4, s); }, [&](hipStream_t s) { return add(1, s); },
        [&](hipStream_t s) { return add(4, s); }, [&](hipStream_t s) { return amax(1, s); }, [&](hipStream_t s) { return amax(4, s); }};
    std::printf("decode-shape timing: us per call = hipEvent time of %d back-to-back calls on one stream / %d, median of 7 batches [min..max]\n", kBatch, kBatch);
    hipStream_t st; HCK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    for (size_t i = 0; i < calls.size(); ++i) {
        const int reps = 10; std::vector<hipEvent_t> ev(2 * reps); for (auto &ee : ev) HCK(hipEventCreate(&ee));
        for (int c = 0; c < reps; ++c) {
            HCK(hipEventRecord(ev[2 * c], st));
            for (int k = 0; k < kBatch; ++k) HCK(calls[i](st));
            HCK(hipEventRecord(ev[2 * c + 1], st));
        }
        HCK(hipStreamSynchronize(st));
        std::vector<double> t; for (int c = 3; c < reps; ++c) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * c], ev[2 * c + 1])); t.push_back(ms * 1e3 / kBatch); }
        std::printf("TIME hip-stream | %-30s | %9.2f us [%.2f..%.2f]\n", kShapes[i], median(t), *std::min_element(t.begin(), t.end()), *std::max_element(t.begin(), t.end()));
    }
    return 0;
}
#endif
