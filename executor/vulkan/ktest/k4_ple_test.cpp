// VulkanAmd P2: the K4 PLE contract rows on Vulkan (shaders/k4_*.comp, src/kernels/k4_ple_vk.*), re-expressing the
// four HIP ktests with their seeds, data and bounds, plus the split-table rows this device needs:
//   hash     executor/hip/ktest/k4_hash_test.cpp: 3,900 windows (random, top of the vocab, every EOS pattern) with
//            Flash-Next's real constants, bit-exact vs a host port of SGLang's `_hash_contexts` (62,400 ids).
//   embed    executor/hip/ktest/k4_embed_test.cpp: FP8 gather (8 ids, 3 out of range) and embed (9 tokens, EOS in history
//            and step) bit-exact vs the host decode; history unchanged.
//            + split: the same table as 40 blocks == one block (all 16,000 rows); a 2,000,000-row FP8 table (320 MB) in
//            8 MiB blocks in the host heap == the same table in one buffer == host decode (every block boundary +-1 and
//            65,536 random ids; embed of 64 tokens over it).
//   k4n      executor/hip/ktest/k4n_test.cpp: NVFP4 gather of 4,096 ids (random bytes, NaN scales, out of range) for three
//            globals, bit-exact vs k4_nvfp4.h decode_row. + split: 2,000,000 NVFP4 rows (180 MB) in 8 MiB host-heap
//            blocks == one buffer == decode_row; embed over it.
//   forward  executor/hip/ktest/k4_forward_test.cpp: delta rel-L2 vs FP64 (limit 2e-2; HIP 3.48e-3 in amd-amd step 2l, 3.61e-3 same session), 9 tokens == 5 + 4
//            bit for bit (delta, conv state, history), add_query == bf16(query + delta). + forward_rows (3 sequences of 2/3/1
//            rows == sequential t = 1 calls, saves == the state after each row), prefill 40 == 17 + 23 with a capture at
//            row 20 == the state after 20 (every part at the step's row class, cls 40: the K2 GEMM for all three, as the
//            HIP test pins plans.cls = 16 for its 9 == 5 + 4 row, k4_forward_test.cpp:7-8, :48), repeat bit-identical,
//            refusals record nothing (cls < t too); prepared K2 weight info (prepare_projections) == the in-call scan.
//   time     GPU-timestamp timings at decode and prefill shapes (same shapes as k4_hip_timing).
//   large GB one NVFP4 table of about GB gigabytes in real ~4 GiB host-heap blocks: pattern fill, gather verified vs the
//            host decode at random ids across the whole table and at every block boundary, gather timed at spread ids.
// Usage: k4_ple_test <spv dir> [hash|embed|k4n|forward|time|large <GB>]... (default: hash embed k4n forward time).
// Exit code: 0 when every row passes, 1 on a FAIL, 2 on a harness error.
#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "launch/vk_launch_testing.h"
#include "kernels/k2_gemm_vk.h"
#include "kernels/k4_nvfp4.h"  // executor/cuda/src (header-only host decode, the HIP k4n_test's reference)
#include "kernels/k4_ple_vk.h"
#include "ktest_util.h"

using kt::bf2f;
using kt::f2bf;
namespace k4 = cere::vkk4;
namespace n4 = cere::k4::nvfp4;
using cere::vk::Buffer;
using cere::vk::Context;

static int g_fail = 0;
static void report(const char *n, bool ok, const std::string &d) {
    std::printf("%-26s %s  %s\n", n, ok ? "PASS" : "FAIL", d.c_str());
    if (!ok) ++g_fail;
}
static std::string fmt(const char *f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char *f, ...) {
    char b[512];
    va_list a;
    va_start(a, f);
    std::vsnprintf(b, sizeof b, f, a);
    va_end(a);
    return b;
}

// --- host references (as the HIP tests) ------------------------------------------------------------------------------
static const int64_t kMult[3] = {23703573157769LL, 20109073645365LL, 8052911324071LL};
static const int64_t kVocab[16] = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                                   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
static const int64_t kOffset[16] = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                                    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
constexpr int64_t kEos = 248044;
constexpr int64_t H = k4::kHidden, C = k4::kChannels, E = k4::kEmbedDim, D = k4::kHeadDim;

static int64_t py_mod(int64_t x, int64_t v) { int64_t r = x % v; return r < 0 ? r + v : r; }
// SGLang's hash of one [3]-token context row at its last position (k4_hash_test / k4_embed_test ref_hash).
static void ref_hash(const int64_t w[3], const int64_t *vocab, const int64_t *offs, int64_t out[16]) {
    int64_t prev_eos = -1;
    for (int i = 0; i < 2; ++i) if (w[i] == kEos) prev_eos = i;
    const int64_t pis = 2 - (prev_eos + 1), sh[3] = {w[2], pis >= 1 ? w[1] : kEos, pis >= 2 ? w[0] : kEos};
    for (int ng = 2; ng <= 3; ++ng) {
        int64_t mix = int64_t(uint64_t(sh[0]) * uint64_t(kMult[0]));
        for (int p = 1; p < ng; ++p) mix ^= int64_t(uint64_t(sh[p]) * uint64_t(kMult[p]));
        for (int h = 0; h < 8; ++h) { const int hd = (ng - 2) * 8 + h; out[hd] = py_mod(mix, vocab[hd]) + offs[hd]; }
    }
}
static uint16_t f2bf_nan(float f) { uint32_t u; std::memcpy(&u, &f, 4); if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7FFF; return f2bf(f); }  // canonical NaN (contract 5)
static float e4m3(uint8_t c) {  // OCP E4M3 (k4_embed_test)
    const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    if (e == 15 && m == 7) return NAN;
    const float v = e == 0 ? std::ldexp(m / 8.f, -6) : std::ldexp(1.f + m / 8.f, e - 7);
    return s ? -v : v;
}
// FP8 row reference (k4_embed_test row_ref): bf16(bf16(e4m3) * scale), zero row out of range.
static void fp8_row_ref(const uint8_t *row, bool valid, uint16_t scale, bool has_scale, uint16_t *dst) {
    for (int64_t j = 0; j < D; ++j) {
        if (!valid) { dst[j] = has_scale ? f2bf_nan(0.f * bf2f(scale)) : 0; continue; }
        const float v = bf2f(f2bf_nan(e4m3(row[j])));
        dst[j] = has_scale ? f2bf_nan(v * bf2f(scale)) : f2bf_nan(v);
    }
}

// --- harness ---------------------------------------------------------------------------------------------------------
template <class T> static Buffer up(Context &ctx, const std::vector<T> &h) { return kt::upload(ctx, h); }
static void sync(Context &ctx, const std::function<void()> &rec) { ctx.begin(); rec(); ctx.barrier(); ctx.submit_wait(); }

struct Times { double med, min, max, stream; };
// `rec(i)` records call i (calls touch distinct data where the caller wants cold rows). warm untimed calls, then reps calls
// each between its own timestamps (median / min / max), then samples x burst back-to-back calls (median per call).
static Times time_it(Context &ctx, const std::function<void(int)> &rec, int warm = 20, int reps = 15, int samples = 5, int burst = 20) {
    int c = 0;
    ctx.begin();
    for (int i = 0; i < warm; ++i) { rec(c++); ctx.barrier(); }
    for (int i = 0; i < reps; ++i) { ctx.stamp(2 * i); rec(c++); ctx.stamp(2 * i + 1); ctx.barrier(); }
    ctx.submit_wait();
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
    std::sort(t.begin(), t.end());
    ctx.begin();
    for (int s = 0; s < samples; ++s) {
        ctx.stamp(2 * s);
        for (int i = 0; i < burst; ++i) { rec(c++); ctx.barrier(); }
        ctx.stamp(2 * s + 1);
    }
    ctx.submit_wait();
    std::vector<double> u;
    for (int s = 0; s < samples; ++s) u.push_back(ctx.stamp_ms(2 * s, 2 * s + 1) / burst);
    std::sort(u.begin(), u.end());
    return {t[t.size() / 2], t.front(), t.back(), u[u.size() / 2]};
}
static int calls_of(int warm = 20, int reps = 15, int samples = 5, int burst = 20) { return warm + reps + samples * burst; }
static void print_times(const char *n, const Times &t, const std::string &extra = "") {
    std::printf("    %-40s call %9.2f us (min %.2f, max %.2f)  stream %9.2f us/call%s\n", n, t.med * 1e3, t.min * 1e3, t.max * 1e3,
                t.stream * 1e3, extra.c_str());
}

// --- hash ------------------------------------------------------------------------------------------------------------
static void test_hash(Context &ctx) {
    std::mt19937_64 g(99);
    std::vector<int64_t> win;
    auto add = [&](int64_t a, int64_t b, int64_t c) { win.insert(win.end(), {a, b, c}); };
    for (int i = 0; i < 3000; ++i) add(int64_t(g() % 248320), int64_t(g() % 248320), int64_t(g() % 248320));
    for (int i = 0; i < 500; ++i) add(248320 - 1 - int64_t(g() % 64), 248320 - 1 - int64_t(g() % 64), 248320 - 1 - int64_t(g() % 64));
    for (int e = 0; e < 8; ++e)
        for (int i = 0; i < 50; ++i)
            add(e & 1 ? kEos : int64_t(g() % 248320), e & 2 ? kEos : int64_t(g() % 248320), e & 4 ? kEos : int64_t(g() % 248320));
    const int64_t n = int64_t(win.size()) / 3;
    auto dw = up(ctx, win), dm = up(ctx, std::vector<int64_t>(kMult, kMult + 3)), dv = up(ctx, std::vector<int64_t>(kVocab, kVocab + 16)),
         doff = up(ctx, std::vector<int64_t>(kOffset, kOffset + 16));
    auto dids = kt::make(ctx, n * 16 * 8);
    sync(ctx, [&] { kt::must(k4::ngram_hash_windows(ctx, dw.addr, n, dm.addr, dv.addr, doff.addr, kEos, dids.addr), "hash"); });
    auto ids = kt::host<int64_t>(dids, n * 16);
    int64_t bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        int64_t want[16];
        ref_hash(&win[i * 3], kVocab, kOffset, want);
        for (int h = 0; h < 16; ++h) bad += ids[i * 16 + h] != want[h];
    }
    report("ngram_hash_windows", bad == 0, fmt("%ld windows (500 at the top of the vocab, 400 with EOS patterns): %ld of %ld ids differ", (long)n,
                                               (long)bad, (long)(n * 16)));
    // repeat: bit-identical
    auto dids2 = kt::make(ctx, n * 16 * 8);
    sync(ctx, [&] { kt::must(k4::ngram_hash_windows(ctx, dw.addr, n, dm.addr, dv.addr, doff.addr, kEos, dids2.addr), "hash"); });
    report("hash repeat", kt::host<int64_t>(dids2, n * 16) == ids, "second call bit-identical");
    report("hash refusals", k4::ngram_hash_windows(ctx, dw.addr + 4, n, dm.addr, dv.addr, doff.addr, kEos, dids.addr) == CERE_ERR_ARGUMENT &&
                                k4::ngram_hash_windows(ctx, dw.addr, -1, dm.addr, dv.addr, doff.addr, kEos, dids.addr) == CERE_ERR_ARGUMENT,
           "misaligned windows / n < 0 -> CERE_ERR_ARGUMENT");
    for (auto *b : {&dw, &dm, &dv, &doff, &dids, &dids2}) ctx.free(*b);
}

// --- split-table helpers -------------------------------------------------------------------------------------------
static bool make_table(Context &ctx, k4::SplitTable &t, k4::TableFormat f, int64_t rows, uint64_t block, k4::Memory mem, const char *what) {
    std::string why;
    if (!t.create(ctx, f, rows, block, mem, &why)) {
        report(what, false, "SplitTable::create refused: " + why);
        return false;
    }
    return true;
}
static void fill_table(k4::SplitTable &t, const std::vector<uint8_t> &bytes, const std::vector<uint8_t> *scales) {
    const uint32_t rb = k4::row_bytes(t.format());
    for (int64_t r = 0; r < t.view().rows; ++r) {
        std::memcpy(t.row(r), &bytes[size_t(r) * rb], rb);
        if (scales) std::memcpy(t.scales(r), &(*scales)[size_t(r) * n4::kRowScales], n4::kRowScales);
    }
}
static std::string geometry(const k4::SplitTable &t) {
    const auto &b = t.buffers();
    return fmt("%u blocks x %u rows (block bytes %llu .. %llu, memory type %s)", t.view().blocks, t.view().rows_per_block,
               (unsigned long long)b.back().size, (unsigned long long)b.front().size,
               t.memory_type() == UINT32_MAX ? "launch default" : std::to_string(t.memory_type()).c_str());
}
// Ids at every block boundary (+-1, +-0) and `random` uniform ids, plus out-of-range ones.
static std::vector<int64_t> boundary_ids(const k4::TableView &v, int random, uint64_t seed) {
    std::vector<int64_t> ids = {-1, -123456789, v.rows, v.rows + 1, int64_t(1) << 40, 0, v.rows - 1};
    for (uint32_t b = 1; b < v.blocks; ++b) {
        const int64_t r = int64_t(b) * v.rows_per_block;
        ids.insert(ids.end(), {r - 2, r - 1, r, r + 1});
    }
    std::mt19937_64 g(seed);
    for (int i = 0; i < random; ++i) ids.push_back(int64_t(g() % uint64_t(v.rows)));
    while (ids.size() % 16) ids.push_back(int64_t(g() % uint64_t(v.rows)));
    return ids;
}

// --- embed (FP8 gather and embed) ------------------------------------------------------------------------------------
static void test_embed(Context &ctx) {
    std::mt19937_64 g(31);
    const int64_t rows = 16 * 1000;
    std::vector<uint8_t> table(rows * D);
    for (auto &b : table) { b = uint8_t(g()); if ((b & 0x7f) == 0x7f) b ^= 1; }  // no NaN codes
    const uint16_t scale = f2bf(0.0625f * 3.f);
    auto ds = up(ctx, std::vector<uint16_t>{scale});
    k4::SplitTable one, split;
    if (!make_table(ctx, one, k4::TableFormat::kFp8, rows, 1ull << 31, k4::Memory::kDefault, "embed table")) return;
    if (!make_table(ctx, split, k4::TableFormat::kFp8, rows, 64 * 1024, k4::Memory::kHostHeap, "embed split table")) return;
    fill_table(one, table, nullptr);
    fill_table(split, table, nullptr);
    auto row_ref = [&](int64_t id, uint16_t *dst) { fp8_row_ref(id >= 0 && id < rows ? &table[id * D] : nullptr, id >= 0 && id < rows, scale, true, dst); };
    {   // gather, incl. out-of-range ids (HIP row)
        std::vector<int64_t> ids = {0, 1, 15999, 7, -1, 16000, 123456789, 8000};
        auto di = up(ctx, ids);
        auto dout = kt::make(ctx, ids.size() * D * 2);
        sync(ctx, [&] { kt::must(k4::gather(ctx, one.view(), di.addr, int64_t(ids.size()), ds.addr, dout.addr), "gather"); });
        auto out = kt::host<uint16_t>(dout, ids.size() * D);
        std::vector<uint16_t> want(ids.size() * D);
        for (size_t i = 0; i < ids.size(); ++i) row_ref(ids[i], &want[i * D]);
        int64_t diff = 0;
        for (size_t i = 0; i < want.size(); ++i) diff += out[i] != want[i];
        report("gather", diff == 0, fmt("8 ids (3 out of range -> zero rows): %ld of %ld values differ (bit-exact)", (long)diff, (long)want.size()));
        ctx.free(di); ctx.free(dout);
    }
    {   // every row through the 40-block split == one block == reference; without scale too
        std::vector<int64_t> ids(rows + 16);
        for (int64_t i = 0; i < rows; ++i) ids[i] = i;
        for (int i = 0; i < 16; ++i) ids[rows + i] = (i % 2) ? -1 - i : rows + i;
        auto di = up(ctx, ids);
        const size_t n = ids.size();
        auto o1 = kt::make(ctx, n * D * 2), o2 = kt::make(ctx, n * D * 2), o3 = kt::make(ctx, n * D * 2);
        sync(ctx, [&] {
            kt::must(k4::gather(ctx, one.view(), di.addr, int64_t(n), ds.addr, o1.addr), "gather");
            kt::must(k4::gather(ctx, split.view(), di.addr, int64_t(n), ds.addr, o2.addr), "gather");
            kt::must(k4::gather(ctx, split.view(), di.addr, int64_t(n), 0, o3.addr), "gather");
        });
        auto a = kt::host<uint16_t>(o1, n * D), b = kt::host<uint16_t>(o2, n * D), c = kt::host<uint16_t>(o3, n * D);
        int64_t diff = 0, refdiff = 0, nsdiff = 0;
        std::vector<uint16_t> want(D);
        for (size_t i = 0; i < n; ++i) {
            row_ref(ids[i], want.data());
            for (int64_t j = 0; j < D; ++j) refdiff += a[i * D + j] != want[j];
            fp8_row_ref(ids[i] >= 0 && ids[i] < rows ? &table[ids[i] * D] : nullptr, ids[i] >= 0 && ids[i] < rows, 0, false, want.data());
            for (int64_t j = 0; j < D; ++j) nsdiff += c[i * D + j] != want[j];
        }
        for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
        report("gather split (small)", diff == 0 && refdiff == 0 && nsdiff == 0,
               fmt("all 16,000 rows + 16 out of range, %s: split vs one block %ld differ, vs reference %ld, no-scale vs reference %ld",
                   geometry(split).c_str(), (long)diff, (long)refdiff, (long)nsdiff));
        for (auto *x : {&di, &o1, &o2, &o3}) ctx.free(*x);
    }
    {   // embed (HIP row), on the one-block and the split table
        std::vector<int64_t> vocab(16, 1000), offs(16);
        for (int h = 0; h < 16; ++h) offs[h] = h * 1000;
        std::vector<int64_t> mult(kMult, kMult + 3);
        const int64_t t = 9;
        std::vector<int64_t> tokens(t);
        for (auto &x : tokens) x = int64_t(g() % 248320);
        tokens[4] = kEos;
        std::vector<int64_t> history = {kEos, int64_t(g() % 248320)};
        auto dm = up(ctx, mult), dv = up(ctx, vocab), doff = up(ctx, offs), dh = up(ctx, history), dtok = up(ctx, tokens);
        k4::Weights w{};
        w.format = k4::TableFormat::kFp8; w.table = one.view(); w.table_scale = ds.addr;
        w.multipliers = dm.addr; w.head_vocab_sizes = dv.addr; w.head_offsets = doff.addr; w.eos_token_id = kEos;
        auto demb = kt::make(ctx, t * E * 2), demb2 = kt::make(ctx, t * E * 2);
        Buffer tab = kt::make(ctx, 2 * k4::kCallTableBytes);
        k4::Weights w2 = w; w2.table = split.view();
        sync(ctx, [&] {
            kt::must(k4::embed(ctx, w, dh.addr, dtok.addr, t, demb.addr, tab, 0), "embed");
            kt::must(k4::embed(ctx, w2, dh.addr, dtok.addr, t, demb2.addr, tab, k4::kCallTableBytes), "embed");
        });
        auto emb = kt::host<uint16_t>(demb, t * E), emb2 = kt::host<uint16_t>(demb2, t * E);
        auto hist_after = kt::host<int64_t>(dh, 2);
        std::vector<int64_t> ctxt = history;
        ctxt.insert(ctxt.end(), tokens.begin(), tokens.end());
        int64_t diff = 0;
        std::vector<uint16_t> row(D);
        for (int64_t i = 0; i < t; ++i) {
            int64_t ids[16];
            ref_hash(&ctxt[i], vocab.data(), offs.data(), ids);
            for (int h = 0; h < 16; ++h) { row_ref(ids[h], row.data()); for (int64_t j = 0; j < D; ++j) diff += emb[i * E + h * D + j] != row[j]; }
        }
        const bool hist_ok = hist_after == history;
        report("embed", diff == 0 && hist_ok, fmt("9 tokens after history (EOS in history and step): %ld of %ld values differ; history unchanged %s", (long)diff,
                                                  (long)(t * E), hist_ok ? "yes" : "NO"));
        report("embed split (small)", emb2 == emb, "the same embed over the 40-block table: bit-equal");
        for (auto *x : {&dm, &dv, &doff, &dh, &dtok, &demb, &demb2, &tab}) ctx.free(*x);
    }
    one.destroy(ctx); split.destroy(ctx);

    {   // 2,000,000-row FP8 table (320 MB): 8 MiB host-heap blocks vs one default buffer vs reference; embed over it
        const int64_t big = 2000000;
        std::vector<uint8_t> bt(size_t(big) * D);
        std::mt19937_64 gb(7);
        for (size_t i = 0; i < bt.size(); i += 8) { uint64_t v = gb(); std::memcpy(&bt[i], &v, 8); }
        for (auto &b : bt) if ((b & 0x7f) == 0x7f) b ^= 1;
        k4::SplitTable a1, a8;
        if (!make_table(ctx, a1, k4::TableFormat::kFp8, big, 1ull << 31, k4::Memory::kDefault, "fp8 320 MB one buffer")) return;
        if (!make_table(ctx, a8, k4::TableFormat::kFp8, big, 8ull << 20, k4::Memory::kHostHeap, "fp8 320 MB split")) return;
        fill_table(a1, bt, nullptr);
        fill_table(a8, bt, nullptr);
        auto ids = boundary_ids(a8.view(), 65536, 3);
        auto di = up(ctx, ids);
        const size_t n = ids.size();
        auto o1 = kt::make(ctx, n * D * 2), o8 = kt::make(ctx, n * D * 2);
        sync(ctx, [&] {
            kt::must(k4::gather(ctx, a1.view(), di.addr, int64_t(n), ds.addr, o1.addr), "gather");
            kt::must(k4::gather(ctx, a8.view(), di.addr, int64_t(n), ds.addr, o8.addr), "gather");
        });
        auto r1 = kt::host<uint16_t>(o1, n * D), r8 = kt::host<uint16_t>(o8, n * D);
        int64_t diff = 0, refdiff = 0;
        std::vector<uint16_t> want(D);
        for (size_t i = 0; i < n; ++i) {
            const bool ok = ids[i] >= 0 && ids[i] < big;
            fp8_row_ref(ok ? &bt[size_t(ids[i]) * D] : nullptr, ok, scale, true, want.data());
            for (int64_t j = 0; j < D; ++j) refdiff += r8[i * D + j] != want[j];
        }
        for (size_t i = 0; i < r1.size(); ++i) diff += r1[i] != r8[i];
        report("gather split (320 MB)", diff == 0 && refdiff == 0,
               fmt("%zu ids (%u boundaries x 4, 65,536 random, 5 out of range), %s: split vs one buffer %ld differ, vs reference %ld", n,
                   a8.view().blocks - 1, geometry(a8).c_str(), (long)diff, (long)refdiff));
        // embed over the split table: 16 heads x 125,000 rows
        std::vector<int64_t> vocab(16, 125000), offs(16), mult(kMult, kMult + 3);
        for (int h = 0; h < 16; ++h) offs[h] = h * 125000;
        const int64_t t = 64;
        std::vector<int64_t> tokens(t), history = {int64_t(gb() % 248320), kEos};
        for (auto &x : tokens) x = int64_t(gb() % 248320);
        tokens[10] = kEos; tokens[11] = kEos;
        auto dm = up(ctx, mult), dv = up(ctx, vocab), doff = up(ctx, offs), dh = up(ctx, history), dtok = up(ctx, tokens);
        k4::Weights w{};
        w.format = k4::TableFormat::kFp8; w.table = a8.view(); w.table_scale = ds.addr;
        w.multipliers = dm.addr; w.head_vocab_sizes = dv.addr; w.head_offsets = doff.addr; w.eos_token_id = kEos;
        k4::Weights w1 = w; w1.table = a1.view();
        auto e8 = kt::make(ctx, t * E * 2), e1 = kt::make(ctx, t * E * 2);
        Buffer tab = kt::make(ctx, 2 * k4::kCallTableBytes);
        sync(ctx, [&] {
            kt::must(k4::embed(ctx, w, dh.addr, dtok.addr, t, e8.addr, tab, 0), "embed");
            kt::must(k4::embed(ctx, w1, dh.addr, dtok.addr, t, e1.addr, tab, k4::kCallTableBytes), "embed");
        });
        auto emb8 = kt::host<uint16_t>(e8, t * E), emb1 = kt::host<uint16_t>(e1, t * E);
        std::vector<int64_t> ctxt = history;
        ctxt.insert(ctxt.end(), tokens.begin(), tokens.end());
        int64_t ediff = 0;
        for (int64_t i = 0; i < t; ++i) {
            int64_t hid[16];
            ref_hash(&ctxt[i], vocab.data(), offs.data(), hid);
            for (int h = 0; h < 16; ++h) {
                fp8_row_ref(&bt[size_t(hid[h]) * D], true, scale, true, want.data());
                for (int64_t j = 0; j < D; ++j) ediff += emb8[i * E + h * D + j] != want[j];
            }
        }
        report("embed split (320 MB)", ediff == 0 && emb8 == emb1, fmt("64 tokens (EOS pairs), 16 x 125,000-row heads over the split table: %ld of %ld values differ "
                                                                     "from the reference; == one buffer %s", (long)ediff, (long)(t * E), emb8 == emb1 ? "yes" : "NO"));
        for (auto *x : {&di, &o1, &o8, &dm, &dv, &doff, &dh, &dtok, &e8, &e1, &tab}) ctx.free(*x);
        a1.destroy(ctx); a8.destroy(ctx);
    }
    ctx.free(ds);
    // Refusals: an FP8 gather on an NVFP4-shaped view whose rows overflow 32-bit offsets, a null table.
    k4::TableView bad{};
    bad.desc = 64; bad.rows = 10; bad.rows_per_block = 0x7fffffff; bad.blocks = 1;
    k4::TableView none{};
    report("gather refusals", k4::gather(ctx, bad, 64, 16, 0, 64) == CERE_ERR_ARGUMENT && k4::gather(ctx, none, 64, 16, 0, 64) == CERE_ERR_ARGUMENT &&
                                   k4::gather_nvfp4(ctx, none, 64, 64, 16, 64) == CERE_ERR_ARGUMENT,
           "32-bit in-block offset overflow / empty view -> CERE_ERR_ARGUMENT");
}

// --- k4n (NVFP4 gather) -----------------------------------------------------------------------------------------------
static void test_k4n(Context &ctx) {
    std::mt19937 g(11);
    const int64_t rows = 5000, n = 4096;
    std::vector<uint8_t> table(size_t(rows) * n4::kRowBytes), scales(size_t(rows) * n4::kRowScales);
    for (auto &b : table) b = uint8_t(g());
    for (auto &b : scales) b = uint8_t(g());
    std::vector<int64_t> ids(n);
    for (auto &i : ids) i = int64_t(g() % (rows + 40)) - 20;
    k4::SplitTable one, split;
    if (!make_table(ctx, one, k4::TableFormat::kNvfp4, rows, 1ull << 31, k4::Memory::kHostHeap, "k4n table")) return;
    if (!make_table(ctx, split, k4::TableFormat::kNvfp4, rows, 32 * 1024, k4::Memory::kHostHeap, "k4n split table")) return;
    fill_table(one, table, &scales);
    fill_table(split, table, &scales);
    auto di = up(ctx, ids);
    auto dout = kt::make(ctx, size_t(n) * 160 * 2), dout2 = kt::make(ctx, size_t(n) * 160 * 2);
    for (float global : {0.0123f, 1.0f, 3.5e-3f}) {
        auto dg = up(ctx, std::vector<float>{global});
        sync(ctx, [&] {
            kt::must(k4::gather_nvfp4(ctx, one.view(), dg.addr, di.addr, n, dout.addr), "gather_nvfp4");
            kt::must(k4::gather_nvfp4(ctx, split.view(), dg.addr, di.addr, n, dout2.addr), "gather_nvfp4");
        });
        auto out = kt::host<uint16_t>(dout, size_t(n) * 160), out2 = kt::host<uint16_t>(dout2, size_t(n) * 160);
        std::vector<uint16_t> ref(160);
        const uint8_t zr[n4::kRowBytes] = {}, zs[n4::kRowScales] = {};
        long bad = 0;
        for (int64_t i = 0; i < n; ++i) {
            const bool in = ids[i] >= 0 && ids[i] < rows;
            n4::decode_row(in ? &table[ids[i] * n4::kRowBytes] : zr, in ? &scales[ids[i] * n4::kRowScales] : zs, global, ref.data());
            if (std::memcmp(ref.data(), &out[i * 160], 320) != 0) {
                if (++bad <= 3)
                    for (int e = 0; e < 160; ++e)
                        if (ref[e] != out[i * 160 + e]) { std::printf("  id %lld e %d ref %04x got %04x\n", (long long)ids[i], e, ref[e], out[i * 160 + e]); break; }
            }
        }
        report("gather_nvfp4", bad == 0, fmt("4096 ids (incl. out of range), global %g: %ld rows differ", global, bad));
        report("gather_nvfp4 split", out2 == out, fmt("the same over %s: %s", geometry(split).c_str(), out2 == out ? "bit-equal" : "DIFFER"));
        ctx.free(dg);
    }
    for (auto *x : {&di, &dout, &dout2}) ctx.free(*x);
    one.destroy(ctx); split.destroy(ctx);

    {   // 2,000,000 NVFP4 rows (180 MB) in 8 MiB host-heap blocks vs one buffer vs decode_row; embed over it
        const int64_t big = 2000000;
        std::vector<uint8_t> bt(size_t(big) * n4::kRowBytes), bs(size_t(big) * n4::kRowScales);
        std::mt19937_64 gb(13);
        for (size_t i = 0; i < bt.size(); i += 8) { uint64_t v = gb(); std::memcpy(&bt[i], &v, 8); }
        for (size_t i = 0; i < bs.size(); i += 8) { uint64_t v = gb(); std::memcpy(&bs[i], &v, std::min<size_t>(8, bs.size() - i)); }
        k4::SplitTable a1, a8;
        if (!make_table(ctx, a1, k4::TableFormat::kNvfp4, big, 1ull << 31, k4::Memory::kDefault, "nvfp4 180 MB one buffer")) return;
        if (!make_table(ctx, a8, k4::TableFormat::kNvfp4, big, 8ull << 20, k4::Memory::kHostHeap, "nvfp4 180 MB split")) return;
        fill_table(a1, bt, &bs);
        fill_table(a8, bt, &bs);
        const float global = 0.0123f;
        auto dg = up(ctx, std::vector<float>{global});
        auto bids = boundary_ids(a8.view(), 65536, 5);
        auto dbi = up(ctx, bids);
        const size_t nb = bids.size();
        auto o1 = kt::make(ctx, nb * 320), o8 = kt::make(ctx, nb * 320);
        sync(ctx, [&] {
            kt::must(k4::gather_nvfp4(ctx, a1.view(), dg.addr, dbi.addr, int64_t(nb), o1.addr), "gather_nvfp4");
            kt::must(k4::gather_nvfp4(ctx, a8.view(), dg.addr, dbi.addr, int64_t(nb), o8.addr), "gather_nvfp4");
        });
        auto r1 = kt::host<uint16_t>(o1, nb * 160), r8 = kt::host<uint16_t>(o8, nb * 160);
        long refbad = 0;
        std::vector<uint16_t> ref(160);
        const uint8_t zr[n4::kRowBytes] = {}, zs[n4::kRowScales] = {};
        for (size_t i = 0; i < nb; ++i) {
            const bool in = bids[i] >= 0 && bids[i] < big;
            n4::decode_row(in ? &bt[size_t(bids[i]) * n4::kRowBytes] : zr, in ? &bs[size_t(bids[i]) * n4::kRowScales] : zs, global, ref.data());
            refbad += std::memcmp(ref.data(), &r8[i * 160], 320) != 0;
        }
        report("gather_nvfp4 split (180 MB)", r1 == r8 && refbad == 0,
               fmt("%zu ids (%u boundaries x 4, 65,536 random, 5 out of range), %s: == one buffer %s, rows differing from decode_row %ld", nb,
                   a8.view().blocks - 1, geometry(a8).c_str(), r1 == r8 ? "yes" : "NO", refbad));
        std::vector<int64_t> vocab(16, 125000), offs(16), mult(kMult, kMult + 3);
        for (int h = 0; h < 16; ++h) offs[h] = h * 125000;
        const int64_t t = 64;
        std::vector<int64_t> tokens(t), history = {kEos, int64_t(gb() % 248320)};
        for (auto &x : tokens) x = int64_t(gb() % 248320);
        tokens[30] = kEos;
        auto dm = up(ctx, mult), dv = up(ctx, vocab), doff = up(ctx, offs), dh = up(ctx, history), dtok = up(ctx, tokens);
        k4::Weights w{};
        w.format = k4::TableFormat::kNvfp4; w.table = a8.view(); w.global_scale = dg.addr;
        w.multipliers = dm.addr; w.head_vocab_sizes = dv.addr; w.head_offsets = doff.addr; w.eos_token_id = kEos;
        k4::Weights w1 = w; w1.table = a1.view();
        auto e8 = kt::make(ctx, t * E * 2), e1 = kt::make(ctx, t * E * 2);
        Buffer tab = kt::make(ctx, 2 * k4::kCallTableBytes);
        sync(ctx, [&] {
            kt::must(k4::embed(ctx, w, dh.addr, dtok.addr, t, e8.addr, tab, 0), "embed");
            kt::must(k4::embed(ctx, w1, dh.addr, dtok.addr, t, e1.addr, tab, k4::kCallTableBytes), "embed");
        });
        auto emb8 = kt::host<uint16_t>(e8, t * E), emb1 = kt::host<uint16_t>(e1, t * E);
        std::vector<int64_t> ctxt = history;
        ctxt.insert(ctxt.end(), tokens.begin(), tokens.end());
        long ediff = 0;
        for (int64_t i = 0; i < t; ++i) {
            int64_t hid[16];
            ref_hash(&ctxt[i], vocab.data(), offs.data(), hid);
            for (int h = 0; h < 16; ++h) {
                n4::decode_row(&bt[size_t(hid[h]) * n4::kRowBytes], &bs[size_t(hid[h]) * n4::kRowScales], global, ref.data());
                ediff += std::memcmp(ref.data(), &emb8[i * E + h * D], 320) != 0;
            }
        }
        report("embed nvfp4 split", ediff == 0 && emb8 == emb1,
               fmt("64 tokens over the split NVFP4 table: %ld of 1,024 rows differ from decode_row; == one buffer %s", ediff, emb8 == emb1 ? "yes" : "NO"));
        // check_table: format/field mismatches refused
        k4::Weights bad = w; bad.table_scale = dg.addr;
        k4::Weights bad2 = w; bad2.global_scale = 0;
        k4::Weights bad3 = w; bad3.format = k4::TableFormat::kFp8;
        report("check_table refusals", k4::check_table(w) == CERE_OK && k4::check_table(bad) == CERE_ERR_ARGUMENT && k4::check_table(bad2) == CERE_ERR_ARGUMENT &&
                                            k4::check_table(bad3) == CERE_ERR_ARGUMENT &&
                                            k4::embed(ctx, bad, dh.addr, dtok.addr, t, e8.addr, tab, 0) == CERE_ERR_ARGUMENT,
               "NVFP4 with a BF16 scale / without a global / an FP8 view with a global -> CERE_ERR_ARGUMENT");
        for (auto *x : {&dg, &dbi, &o1, &o8, &dm, &dv, &doff, &dh, &dtok, &e8, &e1, &tab}) ctx.free(*x);
        a1.destroy(ctx); a8.destroy(ctx);
    }
}

// --- forward --------------------------------------------------------------------------------------------------------
struct FwdSetup {
    std::vector<uint16_t> key_proj, value_proj, norm_key, norm_query, norm_conv, conv_w;
    std::vector<uint8_t> table;
    std::vector<int64_t> vocab, offs, mult;
    k4::SplitTable st;
    std::vector<Buffer> bufs;
    k4::Weights w{};
};
// The HIP k4_forward_test's weights (mt19937_64 g(41), same order); the table split into 64 KiB host-heap blocks.
static bool fwd_setup(Context &ctx, std::mt19937_64 &g, FwdSetup &s) {
    std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd = [&](size_t n, float sc) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * sc); return v; };
    const int64_t rows = 16 * 1000;
    s.table.resize(rows * D);
    for (auto &b : s.table) { b = uint8_t(g() % 0x70); if (g() & 1) b |= 0x80; }
    s.vocab.assign(16, 1000); s.offs.resize(16);
    for (int h = 0; h < 16; ++h) s.offs[h] = h * 1000;
    s.mult.assign(kMult, kMult + 3);
    s.key_proj = rnd(size_t(C) * E, 0.02f); s.value_proj = rnd(size_t(H) * E, 0.02f);
    s.norm_key = rnd(C, 0.1f); s.norm_query = rnd(C, 0.1f); s.norm_conv = rnd(C, 0.1f); s.conv_w = rnd(size_t(C) * 4, 0.3f);
    if (!make_table(ctx, s.st, k4::TableFormat::kFp8, rows, 64 * 1024, k4::Memory::kHostHeap, "forward table")) return false;
    fill_table(s.st, s.table, nullptr);
    auto keep = [&](Buffer b) { s.bufs.push_back(b); return b.addr; };
    s.w.format = k4::TableFormat::kFp8; s.w.table = s.st.view(); s.w.table_scale = keep(up(ctx, std::vector<uint16_t>{f2bf(1.0f)}));
    s.w.multipliers = keep(up(ctx, s.mult)); s.w.head_vocab_sizes = keep(up(ctx, s.vocab)); s.w.head_offsets = keep(up(ctx, s.offs));
    s.w.key_proj = keep(up(ctx, s.key_proj)); s.w.value_proj = keep(up(ctx, s.value_proj));
    s.w.norm_key = keep(up(ctx, s.norm_key)); s.w.norm_query = keep(up(ctx, s.norm_query)); s.w.norm_conv = keep(up(ctx, s.norm_conv));
    s.w.conv_weight = keep(up(ctx, s.conv_w)); s.w.eos_token_id = kEos; s.w.eps = 1e-6f;
    return true;
}
static void fwd_free(Context &ctx, FwdSetup &s) { for (auto &b : s.bufs) ctx.free(b); s.st.destroy(ctx); }

static void test_forward(Context &ctx) {
    std::mt19937_64 g(41);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd = [&](size_t n, float sc) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * sc); return v; };
    FwdSetup s;
    if (!fwd_setup(ctx, g, s)) return;
    const k4::Weights &w = s.w;
    const int64_t T = 9;
    std::vector<int64_t> tokens(T);
    for (auto &x : tokens) x = int64_t(g() % 248320);
    auto query = rnd(size_t(T) * C, 1.f);
    const std::vector<int64_t> hist0 = {kEos, kEos};
    const std::vector<uint16_t> conv0(size_t(C) * 9, 0);
    const size_t wsb = k4::workspace_bytes(64);
    Buffer dws = kt::make(ctx, wsb + 256), tab = kt::make(ctx, 8 * k4::kCallTableBytes);
    Buffer dtok = up(ctx, tokens), dq = up(ctx, query), dout = kt::make(ctx, size_t(T) * C * 2);
    auto step = [&](Buffer &h, Buffer &cs, int64_t t0, int64_t t, bool add) {
        sync(ctx, [&] {
            kt::must(k4::forward(ctx, w, {h.addr, cs.addr}, dtok.addr + t0 * 8, t, dq.addr + t0 * C * 2, add, dout.addr + t0 * C * 2, dws.addr,
                                 wsb, tab, 0),
                     "forward");
        });
    };
    Buffer dh1 = up(ctx, hist0), dc1 = up(ctx, conv0);
    step(dh1, dc1, 0, T, false);
    auto delta = kt::host<uint16_t>(dout, size_t(T) * C), cs1 = kt::host<uint16_t>(dc1, size_t(C) * 9);
    auto h1 = kt::host<int64_t>(dh1, 2);
    Buffer dh2 = up(ctx, hist0), dc2 = up(ctx, conv0);
    step(dh2, dc2, 0, 5, false);
    step(dh2, dc2, 5, T - 5, false);
    auto delta2 = kt::host<uint16_t>(dout, size_t(T) * C), cs2 = kt::host<uint16_t>(dc2, size_t(C) * 9);
    auto h2 = kt::host<int64_t>(dh2, 2);
    const bool split_ok = delta2 == delta && cs2 == cs1 && h2 == h1 && h1[0] == tokens[T - 2] && h1[1] == tokens[T - 1];
    report("state continuity", split_ok, fmt("9 tokens in one call == 5 + 4: delta, conv state, history %s", split_ok ? "bit-equal" : "DIFFER"));
    Buffer dh3 = up(ctx, hist0), dc3 = up(ctx, conv0);
    step(dh3, dc3, 0, T, true);
    auto outq = kt::host<uint16_t>(dout, size_t(T) * C);
    int64_t addbad = 0;
    for (size_t i = 0; i < outq.size(); ++i) addbad += outq[i] != f2bf(bf2f(query[i]) + bf2f(delta[i]));
    report("add_query", addbad == 0, fmt("out == bf16(query + delta): %ld of %ld differ", (long)addbad, (long)outq.size()));
    // repeat: one 9-token call from the same state, bit-identical
    Buffer dh4 = up(ctx, hist0), dc4 = up(ctx, conv0);
    step(dh4, dc4, 0, T, false);
    report("forward repeat", kt::host<uint16_t>(dout, size_t(T) * C) == delta && kt::host<uint16_t>(dc4, size_t(C) * 9) == cs1,
           "the 9-token call again: delta and conv state bit-identical");
    // FP64 reference from the kernel's own embedding (HIP test)
    Buffer dh0 = up(ctx, hist0), demb = kt::make(ctx, size_t(T) * E * 2);
    sync(ctx, [&] { kt::must(k4::embed(ctx, w, dh0.addr, dtok.addr, T, demb.addr, tab, k4::kCallTableBytes), "embed"); });
    auto emb = kt::host<uint16_t>(demb, size_t(T) * E);
    {
        const auto &key_proj = s.key_proj, &value_proj = s.value_proj, &conv_w = s.conv_w;
        std::vector<double> gn_hist(size_t(C) * (9 + T), 0.0), gated(size_t(T) * C);
        auto gnorm = [&](const double *x, const std::vector<uint16_t> &wv, int grp, double *y) {
            double ss = 0;
            for (int64_t i = 0; i < H; ++i) ss += x[i] * x[i];
            const double rs = 1 / std::sqrt(ss / H + 1e-6);
            for (int64_t i = 0; i < H; ++i) y[i] = x[i] * rs * (1 + double(bf2f(wv[grp * H + i])));
        };
        for (int64_t t = 0; t < T; ++t) {
            std::vector<double> key(C), value(H), kn(C), qn(C), q(C);
            for (int64_t o = 0; o < C; ++o) { double a = 0; for (int64_t i = 0; i < E; ++i) a += double(bf2f(emb[t * E + i])) * bf2f(key_proj[o * E + i]); key[o] = a; }
            for (int64_t o = 0; o < H; ++o) { double a = 0; for (int64_t i = 0; i < E; ++i) a += double(bf2f(emb[t * E + i])) * bf2f(value_proj[o * E + i]); value[o] = a; }
            for (int64_t i = 0; i < C; ++i) q[i] = bf2f(query[t * C + i]);
            for (int grp = 0; grp < 4; ++grp) {
                gnorm(&key[grp * H], s.norm_key, grp, &kn[grp * H]);
                gnorm(&q[grp * H], s.norm_query, grp, &qn[grp * H]);
                double dot = 0;
                for (int64_t i = 0; i < H; ++i) dot += kn[grp * H + i] * qn[grp * H + i];
                double gte = dot / std::sqrt(double(H));
                gte = std::sqrt(std::fmax(std::fabs(gte), 1e-6)) * (gte > 0 ? 1 : gte < 0 ? -1 : 0);
                gte = 1 / (1 + std::exp(-gte));
                for (int64_t i = 0; i < H; ++i) gated[t * C + grp * H + i] = gte * value[i];
                std::vector<double> gnv(H);
                gnorm(&gated[t * C + grp * H], s.norm_conv, grp, gnv.data());
                for (int64_t i = 0; i < H; ++i) gn_hist[size_t(grp * H + i) * (9 + T) + 9 + t] = gnv[i];
            }
        }
        double num = 0, den = 0;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t c = 0; c < C; ++c) {
                double acc = 0;
                for (int j = 0; j < 4; ++j) acc += double(bf2f(conv_w[c * 4 + j])) * gn_hist[size_t(c) * (9 + T) + 9 + t - 9 + j * 3];
                const double conv = acc / (1 + std::exp(-acc)), ref = gated[t * C + c] + conv, got = bf2f(delta[t * C + c]);
                num += (got - ref) * (got - ref);
                den += ref * ref;
            }
        const double e = std::sqrt(num / den);
        report("forward vs FP64", e <= 2e-2, fmt("delta rel-L2 %.2e vs FP64 (limit 2e-2; HIP 3.48e-3 in amd-amd step 2l, 3.61e-3 same session), 9 tokens x 10,240 channels", e));
    }

    {   // forward_rows: 3 sequences of 2 / 3 / 1 rows == sequential t = 1 forward calls; saves == state after each row
        const uint32_t counts[3] = {2, 3, 1};
        const int64_t m = 6;
        std::vector<int64_t> rt(m);
        for (auto &x : rt) x = int64_t(g() % 248320);
        rt[3] = kEos;
        auto rq = rnd(size_t(m) * C, 1.f);
        std::vector<std::vector<int64_t>> h0(3);
        std::vector<std::vector<uint16_t>> c0(3);
        for (int q = 0; q < 3; ++q) { h0[q] = {int64_t(g() % 248320), q == 1 ? kEos : int64_t(g() % 248320)}; c0[q] = rnd(size_t(C) * 9, 0.5f); }
        Buffer drt = up(ctx, rt), drq = up(ctx, rq), dro = kt::make(ctx, size_t(m) * C * 2), dso = kt::make(ctx, size_t(m) * C * 2);
        std::vector<Buffer> hs, cs, sh, sc, ph, pc;
        for (int q = 0; q < 3; ++q) { hs.push_back(up(ctx, h0[q])); cs.push_back(up(ctx, c0[q])); ph.push_back(up(ctx, h0[q])); pc.push_back(up(ctx, c0[q])); }
        k4::RowsCall call;
        call.sequences = 3;
        uint32_t first = 0;
        for (int q = 0; q < 3; ++q) { call.states[q] = {hs[q].addr, cs[q].addr}; call.first[q] = first; call.count[q] = counts[q]; first += counts[q]; }
        for (int r = 0; r < m; ++r) {
            if (r == 1 || r == 5) continue;  // two rows without a save
            sh.push_back(kt::make(ctx, 16)); sc.push_back(kt::make(ctx, size_t(C) * 9 * 2));
            call.saves[r] = {sh.back().addr, sc.back().addr};
        }
        sync(ctx, [&] { kt::must(k4::forward_rows(ctx, w, call, drt.addr, m, drq.addr, true, dro.addr, dws.addr, wsb, tab, 0), "forward_rows"); });
        // sequential reference: each row alone (t = 1) on its sequence's private copy of the state
        int64_t outdiff = 0, savediff = 0, statediff = 0;
        std::vector<std::vector<int64_t>> want_h(m);
        std::vector<std::vector<uint16_t>> want_c(m);
        uint32_t r = 0;
        for (int q = 0; q < 3; ++q)
            for (uint32_t j = 0; j < counts[q]; ++j, ++r) {
                sync(ctx, [&] {
                    kt::must(k4::forward(ctx, w, {ph[q].addr, pc[q].addr}, drt.addr + r * 8, 1, drq.addr + r * C * 2, true, dso.addr + r * C * 2, dws.addr,
                                         wsb, tab, k4::kCallTableBytes),
                             "forward");
                });
                want_h[r] = kt::host<int64_t>(ph[q], 2);
                want_c[r] = kt::host<uint16_t>(pc[q], size_t(C) * 9);
            }
        auto ro = kt::host<uint16_t>(dro, size_t(m) * C), so = kt::host<uint16_t>(dso, size_t(m) * C);
        for (size_t i = 0; i < ro.size(); ++i) outdiff += ro[i] != so[i];
        size_t si = 0;
        for (int rr = 0; rr < m; ++rr) {
            if (rr == 1 || rr == 5) continue;
            savediff += kt::host<int64_t>(sh[si], 2) != want_h[rr];
            savediff += kt::host<uint16_t>(sc[si], size_t(C) * 9) != want_c[rr];
            ++si;
        }
        for (int q = 0; q < 3; ++q) {
            statediff += kt::host<int64_t>(hs[q], 2) != kt::host<int64_t>(ph[q], 2);
            statediff += kt::host<uint16_t>(cs[q], size_t(C) * 9) != kt::host<uint16_t>(pc[q], size_t(C) * 9);
        }
        report("forward_rows", outdiff == 0 && savediff == 0 && statediff == 0,
               fmt("3 sequences of 2/3/1 rows (add_query) == sequential t = 1 calls: out %ld of %ld differ, saves %ld of 8 differ, final states %ld of 6 differ",
                   (long)outdiff, (long)ro.size(), (long)savediff, (long)statediff));
        // refusals: sequences not tiling the rows; mismatched save pointers. State and out untouched.
        k4::RowsCall bad = call; bad.count[2] = 2;
        k4::RowsCall bad2 = call; bad2.saves[1] = {sh[0].addr, 0};
        auto before = kt::host<int64_t>(hs[0], 2);
        ctx.begin();
        const bool refused = k4::forward_rows(ctx, w, bad, drt.addr, m, drq.addr, true, dro.addr, dws.addr, wsb, tab, 0) == CERE_ERR_ARGUMENT &&
                             k4::forward_rows(ctx, w, bad2, drt.addr, m, drq.addr, true, dro.addr, dws.addr, wsb, tab, 0) == CERE_ERR_ARGUMENT &&
                             k4::forward_rows(ctx, w, call, drt.addr, 17, drq.addr, true, dro.addr, dws.addr, wsb, tab, 0) == CERE_ERR_ARGUMENT;
        ctx.submit_wait();
        report("forward_rows refusals", refused && kt::host<int64_t>(hs[0], 2) == before, "untiled sequences / half a save / m 17 -> CERE_ERR_ARGUMENT, state untouched");
        for (auto *v : {&hs, &cs, &sh, &sc, &ph, &pc}) for (auto &b : *v) ctx.free(b);
        for (auto *b : {&drt, &drq, &dro, &dso}) ctx.free(*b);
    }

    {   // prefill: 40 tokens in one call == 17 + 23 (GEMM row tiles and conv tiles cross differently); capture at row 20.
        // Every part passes the whole step's row class (cls 40, HIP k4::Plans::cls; k4_forward_test.cpp:48 pins 16 the same
        // way), so all three calls run the shift-invariant K2 GEMM; a part with its own class (17 + 23 at cls 0 is still
        // both >= 17, but a 9-row part would be <= 16) could switch to the GEMV, which is not bit-equal to the GEMM.
        const uint32_t kCls = 40;
        const int64_t P = 40;
        std::vector<int64_t> pt(P);
        for (auto &x : pt) x = int64_t(g() % 248320);
        pt[16] = kEos;
        auto pq = rnd(size_t(P) * C, 1.f);
        auto cinit = rnd(size_t(C) * 9, 0.5f);
        const std::vector<int64_t> hinit = {int64_t(g() % 248320), int64_t(g() % 248320)};
        Buffer dpt = up(ctx, pt), dpq = up(ctx, pq), dpo = kt::make(ctx, size_t(P) * C * 2), dpo2 = kt::make(ctx, size_t(P) * C * 2),
               dpo3 = kt::make(ctx, size_t(20) * C * 2);
        Buffer ha = up(ctx, hinit), ca = up(ctx, cinit), hb = up(ctx, hinit), cb = up(ctx, cinit), hc = up(ctx, hinit), cc = up(ctx, cinit);
        Buffer caph = kt::make(ctx, 16), capc = kt::make(ctx, size_t(C) * 9 * 2);
        const k4::Capture cap{20, caph.addr, capc.addr};
        sync(ctx, [&] {  // one call with a capture at row 20
            kt::must(k4::forward(ctx, w, {ha.addr, ca.addr}, dpt.addr, P, dpq.addr, false, dpo.addr, dws.addr, wsb, tab, 0, &cap, kCls), "forward");
        });
        sync(ctx, [&] {  // 17 + 23 (one command buffer, two call-table regions)
            kt::must(k4::forward(ctx, w, {hb.addr, cb.addr}, dpt.addr, 17, dpq.addr, false, dpo2.addr, dws.addr, wsb, tab, 0, nullptr, kCls), "forward");
            ctx.barrier();
            kt::must(k4::forward(ctx, w, {hb.addr, cb.addr}, dpt.addr + 17 * 8, P - 17, dpq.addr + 17 * C * 2, false, dpo2.addr + 17 * C * 2, dws.addr, wsb,
                                 tab, k4::kCallTableBytes, nullptr, kCls),
                     "forward");
        });
        sync(ctx, [&] {  // the state after 20 tokens
            kt::must(k4::forward(ctx, w, {hc.addr, cc.addr}, dpt.addr, 20, dpq.addr, false, dpo3.addr, dws.addr, wsb, tab, 0, nullptr, kCls), "forward");
        });
        auto o1 = kt::host<uint16_t>(dpo, size_t(P) * C), o2 = kt::host<uint16_t>(dpo2, size_t(P) * C), o3 = kt::host<uint16_t>(dpo3, size_t(20) * C);
        const bool o3_ok = std::equal(o3.begin(), o3.end(), o1.begin());
        const bool same = o1 == o2 && kt::host<int64_t>(ha, 2) == kt::host<int64_t>(hb, 2) &&
                          kt::host<uint16_t>(ca, size_t(C) * 9) == kt::host<uint16_t>(cb, size_t(C) * 9) && o3_ok;
        report("prefill continuity", same, fmt("40 tokens in one call == 17 + 23 == its first 20 alone, all at cls 40 (K2 GEMM): delta, conv state, "
                                               "history %s", same ? "bit-equal" : "DIFFER"));
        const bool cap_ok = kt::host<int64_t>(caph, 2) == kt::host<int64_t>(hc, 2) && kt::host<uint16_t>(capc, size_t(C) * 9) == kt::host<uint16_t>(cc, size_t(C) * 9);
        report("capture", cap_ok, "capture at row 20 of the 40-token call == the state after a 20-token call");
        // refusals record nothing and leave the state untouched
        auto hbefore = kt::host<int64_t>(ha, 2);
        auto cbefore = kt::host<uint16_t>(ca, size_t(C) * 9);
        ctx.begin();
        const k4::Capture badcap{5, caph.addr, capc.addr};
        k4::Weights wn = w; wn.key_proj += 2;
        const bool refused = k4::forward(ctx, w, {ha.addr, ca.addr}, dpt.addr, 0, dpq.addr, false, dpo.addr, dws.addr, wsb, tab, 0) == CERE_ERR_ARGUMENT &&
                             k4::forward(ctx, w, {ha.addr, ca.addr}, dpt.addr, P, dpq.addr, false, dpo.addr, dws.addr + 16, wsb, tab, 0) == CERE_ERR_ARGUMENT &&
                             k4::forward(ctx, w, {ha.addr, ca.addr}, dpt.addr, P, dpq.addr, false, dpo.addr, dws.addr, k4::workspace_bytes(P) - 1, tab, 0) ==
                                 CERE_ERR_ARGUMENT &&
                             k4::forward(ctx, w, {ha.addr, ca.addr}, dpt.addr, P, dpq.addr, false, dpo.addr, dws.addr, wsb, tab, 0, &badcap) == CERE_ERR_ARGUMENT &&
                             k4::forward(ctx, wn, {ha.addr, ca.addr}, dpt.addr, P, dpq.addr, false, dpo.addr, dws.addr, wsb, tab, 0) == CERE_ERR_ARGUMENT &&
                             k4::forward(ctx, w, {ha.addr, ca.addr}, dpt.addr, P, dpq.addr, false, dpo.addr, dws.addr, wsb, tab, 8) == CERE_ERR_ARGUMENT &&
                             k4::forward(ctx, w, {ha.addr, ca.addr}, dpt.addr, P, dpq.addr, false, dpo.addr, dws.addr, wsb, tab, 0, nullptr, P - 1) == CERE_ERR_ARGUMENT;
        ctx.submit_wait();
        report("forward refusals", refused && kt::host<int64_t>(ha, 2) == hbefore && kt::host<uint16_t>(ca, size_t(C) * 9) == cbefore,
               "t 0 / misaligned or short workspace / capture row < 9 / misaligned weight / misaligned call table / cls < t -> CERE_ERR_ARGUMENT, state untouched");
        {   // added: prepared K2 weight info (prepare_projections) == the weights scanned in the call (t 40, cls 0: the GEMM)
            Buffer info = kt::make(ctx, k4::proj_info_bytes());
            k4::Weights wp = w; wp.proj_info = info.addr;
            sync(ctx, [&] { kt::must(k4::prepare_projections(ctx, wp), "prepare_projections"); });
            Buffer h5 = up(ctx, hinit), c5 = up(ctx, cinit), h6 = up(ctx, hinit), c6 = up(ctx, cinit);
            sync(ctx, [&] { kt::must(k4::forward(ctx, wp, {h5.addr, c5.addr}, dpt.addr, P, dpq.addr, false, dpo.addr, dws.addr, wsb, tab, 0), "forward"); });
            sync(ctx, [&] { kt::must(k4::forward(ctx, w, {h6.addr, c6.addr}, dpt.addr, P, dpq.addr, false, dpo2.addr, dws.addr, wsb, tab, 0), "forward"); });
            const bool eq = kt::host<uint16_t>(dpo, size_t(P) * C) == kt::host<uint16_t>(dpo2, size_t(P) * C) &&
                            kt::host<uint16_t>(c5, size_t(C) * 9) == kt::host<uint16_t>(c6, size_t(C) * 9);
            report("prepared weight info", eq, fmt("t 40 with proj_info == the in-call weight scan: delta and conv state %s", eq ? "bit-equal" : "DIFFER"));
            for (auto *b : {&info, &h5, &c5, &h6, &c6}) ctx.free(*b);
        }
        for (auto *b : {&dpt, &dpq, &dpo, &dpo2, &dpo3, &ha, &ca, &hb, &cb, &hc, &cc, &caph, &capc}) ctx.free(*b);
    }
    for (auto *b : {&dws, &tab, &dtok, &dq, &dout, &dh1, &dc1, &dh2, &dc2, &dh3, &dc3, &dh4, &dc4, &dh0, &demb}) ctx.free(*b);
    fwd_free(ctx, s);
}

// --- timing ---------------------------------------------------------------------------------------------------------
// The same shapes as k4_hip_timing: hash 1 / 512 windows; gather FP8 / NVFP4 of 16 (one token) / 8,192 (512 tokens) ids
// spread over a 2,000,000-row table; embed (NVFP4, 16 x 125,000-row heads) t 1 / 512; forward t 1 / 16 / 512 (FP8 table).
static void test_time(Context &ctx) {
    std::printf("timing (GPU timestamps; call = one call between its own stamps, median of 15 after 20 warm-up; stream = 20 back-to-back / 20, median of 5):\n");
    std::mt19937_64 g(77);
    const int ncalls = calls_of();
    {
        std::vector<int64_t> win(size_t(512) * 3 * ncalls);
        for (auto &x : win) x = int64_t(g() % 248320);
        auto dw = up(ctx, win), dm = up(ctx, std::vector<int64_t>(kMult, kMult + 3)), dv = up(ctx, std::vector<int64_t>(kVocab, kVocab + 16)),
             doff = up(ctx, std::vector<int64_t>(kOffset, kOffset + 16));
        auto dids = kt::make(ctx, size_t(512) * 16 * 8);
        for (int n : {1, 512}) {
            auto t = time_it(ctx, [&](int i) { kt::must(k4::ngram_hash_windows(ctx, dw.addr + uint64_t(i) * n * 24, n, dm.addr, dv.addr, doff.addr, kEos, dids.addr), "hash"); });
            print_times(fmt("hash %d windows", n).c_str(), t);
        }
        for (auto *b : {&dw, &dm, &dv, &doff, &dids}) ctx.free(*b);
    }
    const int64_t big = 2000000;
    for (int nv = 0; nv < 2; ++nv) {
        const k4::TableFormat f = nv ? k4::TableFormat::kNvfp4 : k4::TableFormat::kFp8;
        k4::SplitTable tb;
        if (!make_table(ctx, tb, f, big, 8ull << 20, k4::Memory::kHostHeap, "timing table")) return;
        for (int64_t r = 0; r < big; ++r) {
            uint64_t v[20];
            for (auto &x : v) x = g();
            std::memcpy(tb.row(r), v, k4::row_bytes(f));
            if (nv) std::memcpy(tb.scales(r), v + 10, n4::kRowScales);
        }
        auto dsc = up(ctx, std::vector<uint16_t>{f2bf(0.5f)});
        auto dg = up(ctx, std::vector<float>{0.0123f});
        std::vector<int64_t> ids(size_t(8192) * ncalls);
        for (auto &x : ids) x = int64_t(g() % uint64_t(big));
        auto di = up(ctx, ids);
        auto dout = kt::make(ctx, size_t(8192) * 320);
        for (int n : {16, 8192}) {
            auto t = time_it(ctx, [&](int i) {
                const uint64_t ia = di.addr + uint64_t(i) * n * 8;
                kt::must(nv ? k4::gather_nvfp4(ctx, tb.view(), dg.addr, ia, n, dout.addr) : k4::gather(ctx, tb.view(), ia, n, dsc.addr, dout.addr), "gather");
            });
            print_times(fmt("gather %s %d ids (%s)", nv ? "nvfp4" : "fp8", n, n == 16 ? "1 token" : "512 tokens").c_str(), t,
                        fmt("  %.1f Mrows/s", n / (t.med * 1e-3) / 1e6));
        }
        if (nv) {  // embed t 1 / 512 over the NVFP4 table
            std::vector<int64_t> vocab(16, 125000), offs(16), mult(kMult, kMult + 3);
            for (int h = 0; h < 16; ++h) offs[h] = h * 125000;
            std::vector<int64_t> tokens(size_t(512) * ncalls);
            for (auto &x : tokens) x = int64_t(g() % 248320);
            auto dm = up(ctx, mult), dv = up(ctx, vocab), doff = up(ctx, offs), dh = up(ctx, std::vector<int64_t>{kEos, kEos}), dtok = up(ctx, tokens);
            k4::Weights w{};
            w.format = f; w.table = tb.view(); w.global_scale = dg.addr; w.multipliers = dm.addr; w.head_vocab_sizes = dv.addr; w.head_offsets = doff.addr;
            auto demb = kt::make(ctx, size_t(512) * E * 2);
            Buffer tab = kt::make(ctx, k4::kCallTableBytes);
            for (int t : {1, 512}) {
                auto tm = time_it(ctx, [&](int i) { kt::must(k4::embed(ctx, w, dh.addr, dtok.addr + uint64_t(i) * t * 8, t, demb.addr, tab, 0), "embed"); });
                print_times(fmt("embed nvfp4 t %d", t).c_str(), tm);
            }
            for (auto *b : {&dm, &dv, &doff, &dh, &dtok, &demb, &tab}) ctx.free(*b);
        }
        for (auto *b : {&dsc, &dg, &di, &dout}) ctx.free(*b);
        tb.destroy(ctx);
    }
    {   // forward
        std::mt19937_64 gf(41);
        FwdSetup s;
        if (!fwd_setup(ctx, gf, s)) return;
        std::normal_distribution<float> nd(0.f, 1.f);
        const int64_t TM = 512;
        std::vector<int64_t> tokens(TM);
        for (auto &x : tokens) x = int64_t(gf() % 248320);
        std::vector<uint16_t> q(size_t(TM) * C);
        for (auto &x : q) x = f2bf(nd(gf));
        Buffer dtok = up(ctx, tokens), dq = up(ctx, q), dout = kt::make(ctx, size_t(TM) * C * 2), dh = up(ctx, std::vector<int64_t>{kEos, kEos}),
               dc = kt::make(ctx, size_t(C) * 9 * 2);
        const size_t wsb = k4::workspace_bytes(TM);
        Buffer dws = kt::make(ctx, wsb + 256), tab = kt::make(ctx, k4::kCallTableBytes);
        Buffer pinfo = kt::make(ctx, k4::proj_info_bytes());  // the projections' K2 row info, prepared once (at load)
        s.w.proj_info = pinfo.addr;
        sync(ctx, [&] { kt::must(k4::prepare_projections(ctx, s.w), "prepare_projections"); });
        for (int t : {1, 16, 512}) {
            const bool heavy = t == 512;
            auto tm = time_it(ctx, [&](int) {
                kt::must(k4::forward(ctx, s.w, {dh.addr, dc.addr}, dtok.addr, t, dq.addr, true, dout.addr, dws.addr, wsb, tab, 0), "forward");
            }, heavy ? 3 : 20, 15, heavy ? 3 : 5, heavy ? 5 : 20);
            print_times(fmt("forward t %d (add_query)", t).c_str(), tm);
        }
        {   // forward_rows: one 4-row verify chain with 3 saves (an MTP verify step's PLE)
            std::vector<Buffer> sv;
            k4::RowsCall call;
            call.sequences = 1; call.states[0] = {dh.addr, dc.addr}; call.first[0] = 0; call.count[0] = 4;
            for (int r = 0; r < 3; ++r) {
                sv.push_back(kt::make(ctx, 16)); sv.push_back(kt::make(ctx, size_t(C) * 9 * 2));
                call.saves[r] = {sv[2 * r].addr, sv[2 * r + 1].addr};
            }
            auto tm = time_it(ctx, [&](int) {
                kt::must(k4::forward_rows(ctx, s.w, call, dtok.addr, 4, dq.addr, true, dout.addr, dws.addr, wsb, tab, 0), "forward_rows");
            });
            print_times("forward_rows 1 chain x 4 (3 saves)", tm);
            for (auto &b : sv) ctx.free(b);
        }
        for (auto *b : {&dtok, &dq, &dout, &dh, &dc, &dws, &tab, &pinfo}) ctx.free(*b);
        fwd_free(ctx, s);
    }
}

// --- large table -----------------------------------------------------------------------------------------------------
static uint64_t mix64(uint64_t x) {  // splitmix64 finalizer: the pattern of row r, word k is mix64(r * 16 + k)
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
static void pattern_row(int64_t r, uint8_t *bytes, uint8_t *scales) {
    uint64_t w[12];
    for (int k = 0; k < 12; ++k) w[k] = mix64(uint64_t(r) * 16 + k);
    std::memcpy(bytes, w, n4::kRowBytes);
    std::memcpy(scales, w + 10, n4::kRowScales);
}
static void test_large(Context &ctx, double gb) {
    const int64_t rows = int64_t(gb * 1e9 / (n4::kRowBytes + n4::kRowScales));
    std::printf("large table: NVFP4, %lld rows (%.2f GB of rows + scales), blocks <= maxBufferSize in the host heap\n", (long long)rows,
                rows * 90.0 / 1e9);
    k4::SplitTable tb;
    const auto t0 = std::chrono::steady_clock::now();
    if (!make_table(ctx, tb, k4::TableFormat::kNvfp4, rows, ~0ull, k4::Memory::kHostHeap, "large table")) return;
    const auto t1 = std::chrono::steady_clock::now();
    const int nth = 16;
    std::vector<std::thread> th;
    for (int k = 0; k < nth; ++k)
        th.emplace_back([&, k] {
            const int64_t a = rows * k / nth, b = rows * (k + 1) / nth;
            for (int64_t r = a; r < b; ++r) pattern_row(r, tb.row(r), tb.scales(r));
        });
    for (auto &x : th) x.join();
    const auto t2 = std::chrono::steady_clock::now();
    std::printf("    %s; allocate %.2f s, pattern fill %.2f s (%d threads)\n", geometry(tb).c_str(), std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double>(t2 - t1).count(), nth);
    const float global = 0.0123f;
    auto dg = up(ctx, std::vector<float>{global});
    // verify: every block boundary +-2 and 1,000,000 random ids across the table
    auto ids = boundary_ids(tb.view(), 1000000, 21);
    auto di = up(ctx, ids);
    const size_t n = ids.size();
    auto dout = kt::make(ctx, n * 320);
    sync(ctx, [&] { kt::must(k4::gather_nvfp4(ctx, tb.view(), dg.addr, di.addr, int64_t(n), dout.addr), "gather_nvfp4"); });
    auto out = kt::host<uint16_t>(dout, n * 160);
    long bad = 0;
    std::vector<uint16_t> ref(160);
    uint8_t rb[n4::kRowBytes], rs[n4::kRowScales];
    const uint8_t zr[n4::kRowBytes] = {}, zs[n4::kRowScales] = {};
    for (size_t i = 0; i < n; ++i) {
        const bool in = ids[i] >= 0 && ids[i] < rows;
        if (in) pattern_row(ids[i], rb, rs);
        n4::decode_row(in ? rb : zr, in ? rs : zs, global, ref.data());
        bad += std::memcmp(ref.data(), &out[i * 160], 320) != 0;
    }
    report("large table gather", bad == 0, fmt("%zu ids (%u block boundaries x 4, 1,000,000 random over all %lld rows): %ld rows differ from decode_row",
                                               n, tb.view().blocks - 1, (long long)rows, bad));
    ctx.free(di);
    ctx.free(dout);
    // timing at spread ids: distinct random ids per call
    std::mt19937_64 g(23);
    const int ncalls = calls_of();
    for (int nrows : {16, 8192, 1 << 20}) {
        const bool huge = nrows == (1 << 20);
        const int calls = huge ? calls_of(2, 15, 3, 5) : ncalls;
        std::vector<int64_t> tid(size_t(nrows) * (huge ? 1 : calls));
        for (auto &x : tid) x = int64_t(g() % uint64_t(rows));
        auto dti = up(ctx, tid);
        auto dto = kt::make(ctx, size_t(nrows) * 320);
        auto tm = time_it(ctx, [&](int i) {
            kt::must(k4::gather_nvfp4(ctx, tb.view(), dg.addr, dti.addr + (huge ? 0 : uint64_t(i) * nrows * 8), nrows, dto.addr), "gather_nvfp4");
        }, huge ? 2 : 20, 15, huge ? 3 : 5, huge ? 5 : 20);
        print_times(fmt("large gather nvfp4 %d ids", nrows).c_str(), tm,
                    fmt("  %.1f Mrows/s, %.1f GB/s of row+scale bytes", nrows / (tm.med * 1e-3) / 1e6, nrows * 90.0 / (tm.med * 1e-3) / 1e9));
        ctx.free(dti);
        ctx.free(dto);
    }
    ctx.free(dg);
    const auto t3 = std::chrono::steady_clock::now();
    tb.destroy(ctx);
    std::printf("    free %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t3).count());
}

int main(int argc, char **argv) {
    auto ctx_owner = cere::vk::testing::create_or_die(argc > 1 ? argv[1] : "build/vulkan-k4/spv");
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str());
    // the K2 GEMM's accumulator-map probe (once per context): forward's projections at row classes >= 17 need it
    std::printf("K2 GEMM probe: %s\n", cere::vkk2::gemm_probe(ctx) == CERE_OK ? "passed" : "FAILED (row classes >= 17 refused)");
    std::vector<std::string> sections;
    for (int i = 2; i < argc; ++i) sections.push_back(argv[i]);
    if (sections.empty()) sections = {"hash", "embed", "k4n", "forward", "time"};
    for (size_t i = 0; i < sections.size(); ++i) {
        const std::string &s = sections[i];
        std::printf("== %s\n", s.c_str());
        if (s == "hash") test_hash(ctx);
        else if (s == "embed") test_embed(ctx);
        else if (s == "k4n") test_k4n(ctx);
        else if (s == "forward") test_forward(ctx);
        else if (s == "time") test_time(ctx);
        else if (s == "large" && i + 1 < sections.size()) test_large(ctx, std::atof(sections[++i].c_str()));
        else { std::printf("unknown section %s\n", s.c_str()); return 2; }
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
