// VulkanAmd P2: executor/hip/ktest/k9_prep_test.cpp's contract rows on the Vulkan K9 index prep
// (src/kernels/k9_qsa_vk.*, shaders/k9_prep.comp), same seed (37), shapes and bounds:
//  1. indexer_qk vs the host mirror of its arithmetic (Gemma RMSNorm with the kernel's F32 sum order: 16 lanes x 8
//     sequential FMAs then a 32-lane xor tree; BF16 normed row; NeoX RoPE over 64 dims with BF16 cos / sin and every
//     product rounded to BF16): every output within 1 BF16 ulp, the ulp count reported;
//  2. qsa_compress_store: the compressed keys of 23 new tokens at pos0 2 (ring rows 0-1 from before) vs the host mirror
//     (F32 sequential mean of the 4 members -> BF16 -> norm -> RoPE at the first member's position), within 1 ulp;
//     groups not completed untouched; the ring holds the incomplete last group's members;
//  3. continuity: the same tokens as calls of 5 + 7 + 11, as 23 qsa_compress_store_step calls (device position) and
//     as one qsa_compress_store_at (device pos0): keys and the ring rows the contract defines bit-equal to the single
//     call.
// The rope table is computed on the host (cos / sin of p * 1e7^(-2i/64) in FP64, rounded to F32; the HIP test takes
// K1's device table): the mirror reads the same table the device reads, so the row is the same check.
// Vulkan-only rows (the kernels claim more than the HIP test asserts):
//  4. repeat bit-equal (indexer_qk and the compress);
//  5. Bounds: indexer_qk with device positions, two of them outside the rotary range, gives the clamped rows' bytes
//     and sets the fault word; in range it leaves the word 0; qsa_key_save_step vs the host rule (slots, j, n);
//     qsa_chains_step (2 chains, undo copies and steps in one dispatch) == the rows' per-row calls;
//  6. negative rows: arguments outside the contract are refused before anything is recorded; the mRoPE entries
//     (vision: a non-goal until after P3) are refused CERE_ERR_UNSUPPORTED.
// Usage: k9_prep_test <spv dir>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "launch/vk_launch_testing.h"
#include "kernels/k9_qsa_vk.h"
#include "ktest_util.h"

namespace k9 = cere::vkk9;
using kt::bf2f;
using kt::f2bf;
static int g_fail = 0;
static float rb(float x) { return bf2f(f2bf(x)); }
static void check(const char *name, bool ok, const char *msg) { std::printf("%-24s %s  %s\n", name, ok ? "PASS" : "FAIL", msg); if (!ok) ++g_fail; }

constexpr int DI = k9::kIndexHeadDim, HR = k9::kRotaryDim / 2;

// norm + rope of one 128 row (BF16-exact inputs x) -> BF16 bits (the HIP test's mirror)
static void ref_row(const float *x, const std::vector<uint16_t> &w, const float *cs, float eps, uint16_t *out) {
    float lane[32] = {};
    for (int l = 0; l < 16; ++l) for (int j = 0; j < 8; ++j) lane[l] = std::fmaf(x[l * 8 + j], x[l * 8 + j], lane[l]);
    for (int mask = 16; mask >= 1; mask >>= 1) { float t[32]; for (int l = 0; l < 32; ++l) t[l] = lane[l] + lane[l ^ mask]; std::memcpy(lane, t, sizeof t); }
    const float nf = 1.f / std::sqrt(lane[0] / float(DI) + eps);
    float row[DI]; for (int d = 0; d < DI; ++d) row[d] = rb((x[d] * nf) * (1.f + bf2f(w[d])));
    for (int d = 0; d < DI; ++d) {
        float o;
        if (d < HR) o = rb(rb(row[d] * rb(cs[d])) - rb(row[d + HR] * rb(cs[HR + d])));
        else if (d < 2 * HR) o = rb(rb(row[d] * rb(cs[d - HR])) + rb(row[d - HR] * rb(cs[HR + d - HR])));
        else o = row[d];
        out[d] = f2bf(o);
    }
}

int main(int argc, char **argv) {
    auto ctx_owner = cere::vk::testing::create_or_die(argc > 1 ? argv[1] : "build/vulkan-k9/spv");
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str());
    std::mt19937 g(37); std::normal_distribution<float> nd;
    const int64_t NPOS = 4096;
    std::vector<float> cs(size_t(NPOS) * k9::kRotaryDim);
    for (int64_t p = 0; p < NPOS; ++p) for (int i = 0; i < HR; ++i) {
        const double a = double(p) * std::pow(1e7, -2.0 * i / k9::kRotaryDim);
        cs[p * k9::kRotaryDim + i] = float(std::cos(a)); cs[p * k9::kRotaryDim + HR + i] = float(std::sin(a));
    }
    auto dcs = kt::upload(ctx, cs);
    const float eps = 1e-6f;
    std::vector<uint16_t> wq(DI), wk(DI); for (auto &x : wq) x = f2bf(nd(g) * 0.1f); for (auto &x : wk) x = f2bf(nd(g) * 0.1f);
    auto dwq = kt::upload(ctx, wq), dwk = kt::upload(ctx, wk);
    k9::Bounds bnd{0, NPOS, NPOS, 0};
    auto sync = [&](cere_status s, const char *what) { kt::must(s, what); ctx.barrier(); };
    {   // 1. indexer_qk
        const int64_t T = 37, pos0 = 1000;
        std::vector<uint16_t> qk(T * k9::kQkWidth); for (auto &x : qk) x = f2bf(nd(g) * 2.f);
        auto dqk = kt::upload(ctx, qk); auto dqi = kt::make(ctx, T * 4 * DI * 2);
        auto run = [&]() {
            ctx.begin(); sync(k9::indexer_qk(ctx, dqk.addr, k9::kQkWidth, dwq.addr, eps, dcs.addr, bnd, 0, pos0, T, dqi.addr), "indexer_qk"); ctx.submit_wait();
            return kt::host<uint16_t>(dqi, size_t(T) * 4 * DI);
        };
        auto got = run();
        int64_t one = 0, more = 0;
        for (int64_t t = 0; t < T; ++t) for (int h = 0; h < 4; ++h) {
            float x[DI]; for (int d = 0; d < DI; ++d) x[d] = bf2f(qk[t * k9::kQkWidth + h * DI + d]);
            uint16_t want[DI]; ref_row(x, wq, &cs[(pos0 + t) * k9::kRotaryDim], eps, want);
            for (int d = 0; d < DI; ++d) { const int u = std::abs(int(got[(t * 4 + h) * DI + d]) - int(want[d])); one += u == 1; more += u > 1; }
        }
        char msg[160]; std::snprintf(msg, sizeof msg, "37 tokens x 4 heads: %ld outputs at 1 ulp, %ld beyond", (long)one, (long)more);
        check("indexer_qk", more == 0, msg);
        const bool rep = run() == got;
        check("indexer_qk repeat", rep, "a second call bit-equal [Vulkan row]");
        {   // 5a. device positions with two values outside the rotary range [0, 4096): -5 -> row 0, 9999 -> row 4095
            std::vector<int64_t> pos(T); for (int64_t t = 0; t < T; ++t) pos[t] = pos0 + t;
            pos[3] = -5; pos[11] = 9999;
            std::vector<int64_t> clamped = pos; clamped[3] = 0; clamped[11] = NPOS - 1;
            auto dp = kt::upload(ctx, pos), dpc = kt::upload(ctx, clamped);
            auto dfault = kt::make(ctx, 4);
            k9::Bounds bf = bnd; bf.fault = dfault.addr;
            ctx.begin();
            sync(k9::indexer_qk(ctx, dqk.addr, k9::kQkWidth, dwq.addr, eps, dcs.addr, bf, dp.addr, 0, T, dqi.addr), "indexer_qk pos");
            ctx.submit_wait();
            auto a = kt::host<uint16_t>(dqi, size_t(T) * 4 * DI);
            const uint32_t f1 = kt::host<uint32_t>(dfault, 1)[0];
            std::memset(dfault.map, 0, 4);
            ctx.begin();
            sync(k9::indexer_qk(ctx, dqk.addr, k9::kQkWidth, dwq.addr, eps, dcs.addr, bf, dpc.addr, 0, T, dqi.addr), "indexer_qk clamped");
            ctx.submit_wait();
            auto b = kt::host<uint16_t>(dqi, size_t(T) * 4 * DI);
            const uint32_t f2 = kt::host<uint32_t>(dfault, 1)[0];
            // the in-range rows equal the host-position call's
            bool rows_ok = true; for (int64_t t = 0; t < T; ++t) if (t != 3 && t != 11) rows_ok &= std::equal(a.begin() + t * 4 * DI, a.begin() + (t + 1) * 4 * DI, got.begin() + t * 4 * DI);
            std::snprintf(msg, sizeof msg, "positions -5 / 9999 == rows 0 / 4095 %s, fault %u (want 1), in-range call fault %u (want 0), other rows == host-position call %s",
                          a == b ? "yes" : "NO", f1, f2, rows_ok ? "yes" : "NO");
            check("bounds clamp + fault", a == b && f1 == 1 && f2 == 0 && rows_ok, msg);
        }
    }
    {   // 2. compressed keys
        const int64_t T = 23, pos0 = 2, NB = 64;
        std::vector<uint16_t> kidx(T * DI), ring0(4 * DI, 0); for (auto &x : kidx) x = f2bf(nd(g)); for (int i = 0; i < 2 * DI; ++i) ring0[i] = f2bf(nd(g));
        auto dk = kt::upload(ctx, kidx);
        const std::vector<uint16_t> keys0(NB * DI, 0);
        auto r1 = kt::upload(ctx, ring0), k1s = kt::upload(ctx, keys0);
        ctx.begin(); sync(k9::qsa_compress_store(ctx, dk.addr, DI, dwk.addr, eps, dcs.addr, bnd, 0, 0, pos0, T, r1.addr, k1s.addr), "compress"); ctx.submit_wait();
        auto keys1 = kt::host<uint16_t>(k1s, keys0.size()), ring1 = kt::host<uint16_t>(r1, ring0.size());
        int64_t one = 0, more = 0, untouched_bad = 0;
        const int64_t gend = (pos0 + T) / 4;
        for (int64_t gg = 0; gg < NB; ++gg) {
            if (gg >= gend) { for (int d = 0; d < DI; ++d) untouched_bad += keys1[gg * DI + d] != 0; continue; }
            float acc[DI];
            for (int r = 0; r < 4; ++r) { const int64_t p = gg * 4 + r;
                for (int d = 0; d < DI; ++d) { const float f = p < pos0 ? bf2f(ring0[(p % 4) * DI + d]) : bf2f(kidx[(p - pos0) * DI + d]); acc[d] = r == 0 ? f : acc[d] + f; } }
            float x[DI]; for (int d = 0; d < DI; ++d) x[d] = rb(acc[d] * 0.25f);
            uint16_t want[DI]; ref_row(x, wk, &cs[(gg * 4) * k9::kRotaryDim], eps, want);
            for (int d = 0; d < DI; ++d) { const int u = std::abs(int(keys1[gg * DI + d]) - int(want[d])); one += u == 1; more += u > 1; }
        }
        auto ring_eq = [&](const std::vector<uint16_t> &r) {
            bool ok = true; for (int64_t p = gend * 4; p < pos0 + T; ++p) for (int d = 0; d < DI; ++d) ok &= r[(p % 4) * DI + d] == ring1[(p % 4) * DI + d];
            return ok; };
        bool ring_ok = true;
        for (int64_t p = gend * 4; p < pos0 + T; ++p) for (int d = 0; d < DI; ++d) ring_ok &= ring1[(p % 4) * DI + d] == kidx[(p - pos0) * DI + d];
        char msg[220]; std::snprintf(msg, sizeof msg, "%ld groups: %ld outputs at 1 ulp, %ld beyond; groups not completed untouched %s; ring holds the tail %s", (long)gend, (long)one, (long)more, untouched_bad ? "NO" : "yes", ring_ok ? "yes" : "NO");
        check("qsa_compress_store", more == 0 && !untouched_bad && ring_ok, msg);
        {   // repeat
            auto r1b = kt::upload(ctx, ring0), k1b = kt::upload(ctx, keys0);
            ctx.begin(); sync(k9::qsa_compress_store(ctx, dk.addr, DI, dwk.addr, eps, dcs.addr, bnd, 0, 0, pos0, T, r1b.addr, k1b.addr), "compress"); ctx.submit_wait();
            check("compress repeat", kt::host<uint16_t>(k1b, keys0.size()) == keys1 && kt::host<uint16_t>(r1b, ring0.size()) == ring1, "a second call bit-equal (keys, ring) [Vulkan row]");
        }
        // splits 5 + 7 + 11 (one command buffer, a barrier between calls)
        auto r2 = kt::upload(ctx, ring0), k2 = kt::upload(ctx, keys0);
        ctx.begin();
        int64_t at = 0; for (int64_t n : {5, 7, 11}) {
            sync(k9::qsa_compress_store(ctx, dk.addr + uint64_t(at) * DI * 2, DI, dwk.addr, eps, dcs.addr, bnd, 0, 0, pos0 + at, n, r2.addr, k2.addr), "compress split");
            at += n; }
        ctx.submit_wait();
        const bool split_ok = kt::host<uint16_t>(k2, keys0.size()) == keys1 && ring_eq(kt::host<uint16_t>(r2, ring0.size()));
        // step by step (device position): one position slot per token, all in one command buffer
        auto r3 = kt::upload(ctx, ring0), k3 = kt::upload(ctx, keys0);
        std::vector<int64_t> steps(T); for (int64_t t = 0; t < T; ++t) steps[t] = pos0 + t;
        auto dsteps = kt::upload(ctx, steps);
        ctx.begin();
        for (int64_t t = 0; t < T; ++t)
            sync(k9::qsa_compress_store_step(ctx, dk.addr + uint64_t(t) * DI * 2, dwk.addr, eps, dcs.addr, bnd, 0, dsteps.addr + uint64_t(t) * 8, r3.addr, k3.addr), "step");
        ctx.submit_wait();
        const bool step_ok = kt::host<uint16_t>(k3, keys0.size()) == keys1 && ring_eq(kt::host<uint16_t>(r3, ring0.size()));
        auto r4 = kt::upload(ctx, ring0), k4 = kt::upload(ctx, keys0);
        ctx.begin(); sync(k9::qsa_compress_store_at(ctx, dk.addr, DI, dwk.addr, eps, dcs.addr, bnd, 0, dsteps.addr, T, r4.addr, k4.addr), "at"); ctx.submit_wait();
        const bool at_ok = kt::host<uint16_t>(k4, keys0.size()) == keys1 && ring_eq(kt::host<uint16_t>(r4, ring0.size()));
        std::snprintf(msg, sizeof msg, "keys + the defined ring rows bit-equal to the 23-token call: 5+7+11 %s, 23 x step %s, _at %s", split_ok ? "yes" : "NO", step_ok ? "yes" : "NO", at_ok ? "yes" : "NO");
        check("compress continuity", split_ok && step_ok && at_ok, msg);

        {   // 5b. qsa_key_save_step vs the host rule: for positions 40..47 and chains n = 1..4 at j = 1..n
            std::vector<int64_t> slots(NPOS); for (int64_t p = 0; p < NPOS; ++p) slots[p] = (p ^ 8) ;  // groups of 4 stay together
            auto dsl = kt::upload(ctx, slots);
            std::vector<uint16_t> keysr(NB * DI); for (auto &x : keysr) x = f2bf(nd(g));
            auto dkr = kt::upload(ctx, keysr);
            int64_t bad = 0, cases = 0;
            for (int64_t at2 = 40; at2 < 48; ++at2) for (int64_t n = 1; n <= 4; ++n) for (int64_t j = 1; j <= n; ++j) {
                std::vector<int64_t> pv{at2}; auto dpv = kt::upload(ctx, pv);
                std::vector<uint16_t> sentinel(DI, 0xBEEF); auto dsv = kt::upload(ctx, sentinel);
                ctx.begin(); sync(k9::qsa_key_save_step(ctx, dkr.addr, dsl.addr, bnd, dpv.addr, j, n, dsv.addr), "key_save"); ctx.submit_wait();
                auto gotv = kt::host<uint16_t>(dsv, DI);
                const int64_t p = at2 - j, q = at2 + (3 - at2 % 4) % 4;
                std::vector<uint16_t> want = sentinel;
                if (q <= p + n) { const int64_t row = slots[q - 3] / 4; for (int d = 0; d < DI; ++d) want[d] = keysr[row * DI + d]; }
                bad += gotv != want; ++cases;
                ctx.free(dpv); ctx.free(dsv);
            }
            std::snprintf(msg, sizeof msg, "%ld cases (positions 40..47, n 1..4, j 1..n): %ld differ from the host rule [Vulkan row]", (long)cases, (long)bad);
            check("qsa_key_save_step", bad == 0, msg);
        }
    }
    {   // 5c. qsa_chains_step (one dispatch for the verify graph's rows) == the rows' per-row calls in row order
        const int64_t NB = 128;
        struct Chain { int first, rows; int64_t pos; };
        const Chain chains[2] = {{0, 3, 101}, {3, 5, 202}};
        std::vector<int64_t> slots(NB * 4); for (int64_t p = 0; p < NB * 4; ++p) slots[p] = ((p / 4) ^ 5) * 4 + p % 4;
        auto dsl = kt::upload(ctx, slots);
        std::vector<uint16_t> kid(8 * DI); for (auto &x : kid) x = f2bf(nd(g));
        auto dkid = kt::upload(ctx, kid);
        std::vector<uint16_t> ring0(4 * DI), keys0(NB * DI); for (auto &x : ring0) x = f2bf(nd(g)); for (auto &x : keys0) x = f2bf(nd(g));
        std::vector<int64_t> pos(8);
        for (const Chain &c : chains) for (int i = 0; i < c.rows; ++i) pos[c.first + i] = c.pos + i;
        auto dpos = kt::upload(ctx, pos);
        // per chain: ring, keys, undo rings / keys of its rows (two copies: chains_step and the reference)
        std::vector<cere::vk::Buffer> ring[2], keys[2], uring[2], ukey[2];
        for (int v = 0; v < 2; ++v) for (int c = 0; c < 2; ++c) { ring[v].push_back(kt::upload(ctx, ring0)); keys[v].push_back(kt::upload(ctx, keys0)); }
        for (int v = 0; v < 2; ++v) for (int r = 0; r < 8; ++r) { uring[v].push_back(kt::make(ctx, 4 * DI * 2)); ukey[v].push_back(kt::make(ctx, DI * 2)); }
        k9::ChainsStep cs_rows{};
        auto chain_of = [&](int r) { return r < 3 ? 0 : 1; };
        for (int c = 0; c < 2; ++c) { cs_rows.chain_first[c] = chains[c].first; cs_rows.chain_rows[c] = chains[c].rows; }
        for (int r = 0; r < 8; ++r) {
            const int c = chain_of(r), j = r - chains[c].first, n = chains[c].rows - 1;
            cs_rows.k_idx[r] = r == 6 ? 0 : dkid.addr + uint64_t(r) * DI * 2;  // row 6: no step (undo only)
            cs_rows.slots[r] = dsl.addr; cs_rows.ring[r] = ring[0][c].addr; cs_rows.keys[r] = keys[0][c].addr;
            cs_rows.undo_ring[r] = j >= 1 ? uring[0][r].addr : 0; cs_rows.undo_key[r] = j >= 1 ? ukey[0][r].addr : 0;
            cs_rows.context[r] = NB * 4; cs_rows.j[r] = j >= 1 ? j : 0; cs_rows.n[r] = n;
        }
        auto table = kt::make(ctx, k9::kChainsTableBytes);
        ctx.begin();
        sync(k9::qsa_chains_step(ctx, cs_rows, 2, dwk.addr, eps, dcs.addr, bnd, dpos.addr, table), "chains_step");
        ctx.submit_wait();
        for (int r = 0; r < 8; ++r) {  // reference: row order, one submission per row
            const int c = chain_of(r), j = r - chains[c].first, n = chains[c].rows - 1;
            const k9::Bounds br{0, NPOS, NB * 4, 0};
            if (j >= 1) {
                std::memcpy(uring[1][r].map, ring[1][c].map, 4 * DI * 2);
                ctx.begin(); sync(k9::qsa_key_save_step(ctx, keys[1][c].addr, dsl.addr, br, dpos.addr + uint64_t(r) * 8, j, n, ukey[1][r].addr), "key_save"); ctx.submit_wait();
            }
            if (r == 6) continue;
            ctx.begin(); sync(k9::qsa_compress_store_step(ctx, dkid.addr + uint64_t(r) * DI * 2, dwk.addr, eps, dcs.addr, br, dsl.addr, dpos.addr + uint64_t(r) * 8, ring[1][c].addr, keys[1][c].addr), "step"); ctx.submit_wait();
        }
        int64_t bad = 0;
        for (int c = 0; c < 2; ++c) { bad += kt::host<uint16_t>(ring[0][c], 4 * DI) != kt::host<uint16_t>(ring[1][c], 4 * DI); bad += kt::host<uint16_t>(keys[0][c], NB * DI) != kt::host<uint16_t>(keys[1][c], NB * DI); }
        for (int r = 0; r < 8; ++r) { bad += kt::host<uint16_t>(uring[0][r], 4 * DI) != kt::host<uint16_t>(uring[1][r], 4 * DI); bad += kt::host<uint16_t>(ukey[0][r], DI) != kt::host<uint16_t>(ukey[1][r], DI); }
        char msg[200]; std::snprintf(msg, sizeof msg, "2 chains (3 + 5 rows at 101 / 202, one row without a step): %ld of 20 buffers (rings, keys, undo rings, undo keys) differ from per-row calls [Vulkan row]", (long)bad);
        check("qsa_chains_step", bad == 0, msg);
    }
    {   // 6. negative rows
        auto dummy = kt::make(ctx, 1 << 16);
        const uint64_t a = dummy.addr;
        int refused = 0, total = 0;
        auto expect = [&](cere_status s, cere_status want) { ++total; refused += s == want; };
        ctx.begin();
        expect(k9::indexer_qk(ctx, a, 600, a, 1e-6f, a, bnd, 0, 0, 1, a), CERE_ERR_ARGUMENT);             // stride < 640
        expect(k9::indexer_qk(ctx, a, 640, a, 0.f, a, bnd, 0, 0, 1, a), CERE_ERR_ARGUMENT);               // eps 0
        expect(k9::indexer_qk(ctx, a, 640, a, 1e-6f, a, bnd, 0, NPOS, 1, a), CERE_ERR_ARGUMENT);          // rotary row past the table
        expect(k9::qsa_compress_store(ctx, a, 128, a, 1e-6f, a, bnd, 0, 0, NPOS - 2, 5, a, a), CERE_ERR_ARGUMENT);  // past the context
        expect(k9::qsa_compress_store_at(ctx, a, 128, a, 1e-6f, a, bnd, 0, a, 0, a, a), CERE_ERR_ARGUMENT);  // tokens 0
        expect(k9::qsa_key_save_step(ctx, a, 0, bnd, a, 1, 1, a), CERE_ERR_ARGUMENT);                     // slots null
        expect(k9::qsa_key_save_step(ctx, a, a, bnd, a, 3, 2, a), CERE_ERR_ARGUMENT);                     // j > n
        expect(k9::indexer_qk_mrope(ctx, a, 640, a, 1e-6f, a, bnd, a, 4, 0, 1, a), CERE_ERR_UNSUPPORTED);
        expect(k9::qsa_compress_store_mrope(ctx, a, 128, a, 1e-6f, a, bnd, a, 4, 0, 0, 0, 1, a, a), CERE_ERR_UNSUPPORTED);
        ctx.submit_wait();
        char msg[160]; std::snprintf(msg, sizeof msg, "%d of %d refused as expected (7 CERE_ERR_ARGUMENT, mRoPE x2 CERE_ERR_UNSUPPORTED) [Vulkan row]", refused, total);
        check("negative arguments", refused == total, msg);
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
