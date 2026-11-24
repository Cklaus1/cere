// VulkanAmd P3 (T3): contract rows of vkk5::chain_core (src/kernels/k5_gdn_chain_vk.*), the GDN verify chain
// (executor/cuda/src/kernels/k5_gdn_layer.h:110-138). The rows are the CUDA suite's (executor/tests/gpu/
// test_kernels_k5_gdn.py), on synthetic inputs at the model's GDN dims (the CUDA suite feeds SGLang's projections of real
// embedding rows; the contract rows are bit-exact, so the data only has to exercise the arithmetic):
//  1. DG-02 (test_chain_core_equals_per_row): after a CHAIN_PREFILL (27)-token PREFILL (vkk5::layer_core), consecutive
//     chains of CHAIN_CORE_ROWS (4, 1, 3, 2, 4, 9) rows, each from the state the last left, equal the same rows through
//     vkk5::layer_core DECODE one token at a time: y, both final states and the state after every row but the last
//     (the undo snapshots), byte for byte;
//  2. R16-B (test_chains_core_equals_per_row): one call over the chains of CHAINS_CORE_LAYOUTS ((4,4,4,4), (4), (9,3,1,3),
//     (2,5,1), (16), (1 x 16)), chain c on its own sequence (a 27-token PREFILL of its own window), equals the per-row
//     calls per chain: y, every chain's final states, the saves, byte for byte;
//  3. (extra) the call repeated from the same states is bit-identical; y aliasing mixed gives the same bytes;
//  4. negative (the CUDA header's errors): chains sharing a state, rows not the sum of chains.rows, a null save, a save
//     aliasing a state, rows 0 / 17, count 0, a small or misaligned workspace, a small table: ARGUMENT, nothing recorded.
// Then decode-shape timing (the K1 batch method: us per call = GPU time of 100 dependent calls with a barrier after each
// / 100, median of 7 batches after 3 untimed). Built with -DT3_HIP_TIMING under hipcc it is instead the same-method HIP
// timing of cere::k5::chain_core (reference). Exit 0 when every row passes.
#ifdef T3_HIP_TIMING
#include <hip/hip_runtime.h>
#include "kernels/k5_gdn_layer.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#ifndef T3_HIP_TIMING
#include "kernels/k5_gdn_chain_vk.h"
#include "kernels/k5_gdn_layer_vk.h"
#include "ktest_util.h"
#include "launch/vk_launch_testing.h"
#endif

static uint16_t f2bf_(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static double median(std::vector<double> t) { std::sort(t.begin(), t.end()); return t[t.size() / 2]; }

constexpr int CD = 10240, VD = 6144, VH = 48, D = 128;
constexpr size_t CONV_N = size_t(CD) * 3, SSM_N = size_t(VH) * D * D;
constexpr int kBatch = 100;
struct TShape { const char *name; int chains; int rows_each; };
static const TShape kShapes[] = {{"chain_core 1 chain x 1 row", 1, 1}, {"chain_core 1 chain x 4 rows (verify)", 1, 4},
                                 {"chain_core 1 chain x 16 rows", 1, 16}, {"chain_core 4 chains x 4 rows", 4, 4}};

// Synthetic layer data (seeded): conv weight, A_log, dt_bias, norm as k5_layer_test; per-token mixed / z / b / a.
struct Data {
    std::vector<uint16_t> conv_w, dt_bias, norm_w, mixed, z, b, a;
    std::vector<float> A_log;
};
static Data make_data(uint32_t seed, int64_t tokens) {
    std::mt19937 g(seed); std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd = [&](size_t n, float s, float mean = 0.f) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf_(mean + nd(g) * s); return v; };
    Data d;
    d.conv_w = rnd(size_t(CD) * 4, 0.5f); d.dt_bias = rnd(VH, 0.5f); d.norm_w = rnd(D, 0.1f, 1.f);
    d.A_log.resize(VH); for (auto &x : d.A_log) x = std::log(0.5f + std::fabs(nd(g)));
    d.mixed = rnd(size_t(tokens) * CD, 1.f); d.z = rnd(size_t(tokens) * VD, 1.f); d.b = rnd(size_t(tokens) * VH, 1.f); d.a = rnd(size_t(tokens) * VH, 1.f);
    return d;
}

#ifndef T3_HIP_TIMING
namespace k5 = cere::vkk5;
using cere::vk::Buffer;
static int g_fail = 0;
static void report(const char *name, bool ok, const char *detail) {
    std::printf("%-40s %s  %s\n", name, ok ? "PASS" : "FAIL", detail);
    if (!ok) ++g_fail;
}

struct Fixture {
    cere::vk::Context &ctx;
    Buffer conv_w, A_log, dt, norm, mixed, z, b, a, ws_core, ws_chain, y_ref, y, table;
    k5::LayerWeights w{};
    size_t core_bytes = 0, chain_bytes = 0;
    Fixture(cere::vk::Context &c, const Data &d, int64_t tokens) : ctx(c) {
        conv_w = kt::upload(ctx, d.conv_w); A_log = kt::upload(ctx, d.A_log); dt = kt::upload(ctx, d.dt_bias); norm = kt::upload(ctx, d.norm_w);
        mixed = kt::upload(ctx, d.mixed); z = kt::upload(ctx, d.z); b = kt::upload(ctx, d.b); a = kt::upload(ctx, d.a);
        w.conv1d = conv_w.addr; w.A_log = A_log.addr; w.dt_bias = dt.addr; w.norm = norm.addr;  // projections unused by the cores
        core_bytes = k5::core_workspace_bytes(64); ws_core = kt::make(ctx, core_bytes);
        chain_bytes = k5::chain_core_workspace_bytes(16); ws_chain = kt::make(ctx, chain_bytes);
        y_ref = kt::make(ctx, size_t(tokens) * VD * 2); y = kt::make(ctx, size_t(tokens) * VD * 2);
        table = kt::make(ctx, 4096);
    }
    uint64_t row(const Buffer &buf, int64_t r, int width) const { return buf.addr + uint64_t(r) * uint64_t(width) * 2; }
    // layer_core on rows [r0, r0 + n) into y_ref (PREFILL n >= 1 or DECODE n == 1), one submit.
    void core(k5::LayerState s, uint32_t op, int64_t r0, int64_t n) {
        ctx.begin();
        kt::must(k5::layer_core(ctx, w, s, op, row(mixed, r0, CD), row(z, r0, VD), row(b, r0, VH), row(a, r0, VH), n, row(y_ref, r0, VD), ws_core.addr,
                                core_bytes), "layer_core");
        ctx.barrier();
        ctx.submit_wait();
    }
};

struct State {
    Buffer conv, ssm;
    k5::LayerState ls() const { return {conv.addr, ssm.addr}; }
};
static State new_state(cere::vk::Context &ctx) { return {kt::make(ctx, CONV_N * 2), kt::make(ctx, SSM_N * 2)}; }
static void copy_state(const State &from, State &to) { std::memcpy(to.conv.map, from.conv.map, CONV_N * 2); std::memcpy(to.ssm.map, from.ssm.map, SSM_N * 2); }
static bool same_state(const State &x, const State &y) {
    return std::memcmp(x.conv.map, y.conv.map, CONV_N * 2) == 0 && std::memcmp(x.ssm.map, y.ssm.map, SSM_N * 2) == 0;
}
static void free_state(cere::vk::Context &ctx, State &s) { ctx.free(s.conv); ctx.free(s.ssm); }

// One chain_core call over `layout` (rows back to back from mixed row r0) on states[c], saves[r]; one submit.
static cere_status run_chains(Fixture &f, const std::vector<int> &layout, int64_t r0, std::vector<State> &states, std::vector<State> &saves,
                              uint64_t y_addr, bool submit = true) {
    k5::Chains ch{};
    ch.count = int64_t(layout.size());
    int64_t rows = 0;
    for (size_t c = 0; c < layout.size(); ++c) {
        ch.rows[c] = layout[c]; ch.states[c] = states[c].ls();
        for (int i = 0; i < layout[c]; ++i, ++rows) if (i + 1 < layout[c]) ch.saves[rows] = saves[rows].ls();
    }
    if (submit) f.ctx.begin();
    const cere_status st = k5::chain_core(f.ctx, f.w, ch, f.row(f.mixed, r0, CD), f.row(f.z, r0, VD), f.row(f.b, r0, VH), f.row(f.a, r0, VH), rows, y_addr,
                                          f.ws_chain.addr, f.chain_bytes, f.table, 0);
    if (submit) { f.ctx.barrier(); f.ctx.submit_wait(); }
    return st;
}

// The reference: each chain's rows through layer_core DECODE one at a time on ref[c]; snapshots after every row but the
// chain's last. Returns the snapshots indexed by the call's row.
static std::vector<State> per_row(Fixture &f, const std::vector<int> &layout, int64_t r0, std::vector<State> &ref) {
    std::vector<State> snaps;
    int64_t r = 0;
    for (size_t c = 0; c < layout.size(); ++c)
        for (int i = 0; i < layout[c]; ++i, ++r) {
            f.core(ref[c].ls(), CERE_OP_DECODE, r0 + r, 1);
            State s = new_state(f.ctx);
            if (i + 1 < layout[c]) copy_state(ref[c], s);
            snaps.push_back(s);
        }
    return snaps;
}

static bool compare(Fixture &f, const std::vector<int> &layout, int64_t r0, std::vector<State> &states, std::vector<State> &saves, std::vector<State> &ref,
                    std::vector<State> &snaps, int64_t *ydiff) {
    int64_t rows = 0; for (int n : layout) rows += n;
    const size_t yb = size_t(rows) * VD * 2, off = size_t(r0) * VD * 2;
    *ydiff = 0;
    const uint8_t *ya = static_cast<const uint8_t *>(f.y.map) + off, *yr = static_cast<const uint8_t *>(f.y_ref.map) + off;
    for (size_t i = 0; i < yb; i += 2) *ydiff += std::memcmp(ya + i, yr + i, 2) != 0;
    bool ok = *ydiff == 0;
    for (size_t c = 0; c < layout.size(); ++c) ok = ok && same_state(states[c], ref[c]);
    int64_t r = 0;
    for (size_t c = 0; c < layout.size(); ++c)
        for (int i = 0; i < layout[c]; ++i, ++r)
            if (i + 1 < layout[c]) ok = ok && same_state(saves[r], snaps[r]);
    return ok;
}

int main(int argc, char **argv) {
    auto ctx_owner = cere::vk::testing::create_or_die(argc > 1 ? argv[1] : "build/vulkan-t3/spv");
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str());
    const int64_t PREFILL = 27;

    // ---- 1. DG-02: consecutive chains on one sequence
    {
        const std::vector<int> chain_rows = {4, 1, 3, 2, 4, 9};
        int64_t total = PREFILL; for (int n : chain_rows) total += n;
        Data d = make_data(5443, total);
        Fixture f(ctx, d, total);
        State st = new_state(ctx), ref = new_state(ctx);
        f.core(st.ls(), CERE_OP_PREFILL, 0, PREFILL);
        copy_state(st, ref);
        int64_t t0 = PREFILL;
        for (size_t i = 0; i < chain_rows.size(); ++i) {
            const int n = chain_rows[i];
            std::vector<State> states = {st}, refs = {ref}, saves;
            for (int r = 0; r < n; ++r) saves.push_back(new_state(ctx));
            std::vector<State> snaps = per_row(f, {n}, t0, refs);
            kt::must(run_chains(f, {n}, t0, states, saves, f.row(f.y, t0, VD)), "chain_core");
            int64_t yd = 0;
            const bool ok = compare(f, {n}, t0, states, saves, refs, snaps, &yd);
            char name[64], det[160];
            std::snprintf(name, sizeof name, "DG-02 chain%zu_R%d (after %ld)", i, n, long(t0));
            std::snprintf(det, sizeof det, "y: %ld of %ld differ; conv + ssm state and %d undo saves %s layer_core DECODE per row",
                          long(yd), long(n) * VD, n - 1, ok ? "bit-equal to" : "DIFFER from");
            report(name, ok, det);
            for (auto &s : saves) free_state(ctx, s);
            for (auto &s : snaps) free_state(ctx, s);
            t0 += n;
        }
        free_state(ctx, st); free_state(ctx, ref);
    }

    // ---- 2. R16-B: one call over several sequences' chains
    const std::vector<std::vector<int>> layouts = {{4, 4, 4, 4}, {4}, {9, 3, 1, 3}, {2, 5, 1}, {16}, std::vector<int>(16, 1)};
    for (const auto &layout : layouts) {
        const int chains = int(layout.size());
        int rows = 0; for (int n : layout) rows += n;
        const int64_t total = PREFILL + chains + rows;
        Data d = make_data(5449, total);
        Fixture f(ctx, d, total);
        std::vector<State> states, refs, saves;
        for (int c = 0; c < chains; ++c) {
            states.push_back(new_state(ctx)); refs.push_back(new_state(ctx));
            f.core(states[c].ls(), CERE_OP_PREFILL, c, PREFILL);  // chain c's sequence: its own 27-token window
            copy_state(states[c], refs[c]);
        }
        for (int r = 0; r < rows; ++r) saves.push_back(new_state(ctx));
        const int64_t r0 = PREFILL + chains;
        std::vector<State> start;  // for the repeat / alias rows
        for (int c = 0; c < chains; ++c) { start.push_back(new_state(ctx)); copy_state(states[c], start[c]); }
        std::vector<State> snaps = per_row(f, layout, r0, refs);
        kt::must(run_chains(f, layout, r0, states, saves, f.row(f.y, r0, VD)), "chain_core");
        int64_t yd = 0;
        const bool ok = compare(f, layout, r0, states, saves, refs, snaps, &yd);
        std::string name = "R16-B chains_core_";
        for (int c = 0; c < chains; ++c) name += (c ? "-" : "") + std::to_string(layout[c]);
        char det[160];
        std::snprintf(det, sizeof det, "y: %ld of %ld differ; %d final states and %d saves %s per-row layer_core DECODE", long(yd), long(rows) * VD, chains,
                      rows - chains, ok ? "bit-equal to" : "DIFFER from");
        report(name.c_str(), ok, det);
        if (layout.size() == 4 && layout[0] == 9) {  // (extra) repeat and y aliasing mixed, on the (9,3,1,3) layout
            auto y1 = kt::host<uint16_t>(f.y, size_t(rows) * VD, size_t(r0) * VD * 2);
            std::vector<State> s2, sv2;
            for (int c = 0; c < chains; ++c) { s2.push_back(new_state(ctx)); copy_state(start[c], s2[c]); }
            for (int r = 0; r < rows; ++r) sv2.push_back(new_state(ctx));
            kt::must(run_chains(f, layout, r0, s2, sv2, f.y.addr + uint64_t(r0) * VD * 2), "chain_core repeat");
            bool rep = kt::host<uint16_t>(f.y, size_t(rows) * VD, size_t(r0) * VD * 2) == y1;
            for (int c = 0; c < chains; ++c) rep = rep && same_state(s2[c], states[c]);
            for (int r = 0; r < rows; ++r) rep = rep && same_state(sv2[r], saves[r]);
            report("repeat bit-identical (extra)", rep, "(9,3,1,3) from the same states: y, states, saves identical");
            // y aliasing mixed: a copy of the inputs whose mixed rows receive y
            auto mixed_copy = kt::upload(ctx, d.mixed);
            for (int c = 0; c < chains; ++c) copy_state(start[c], s2[c]);
            k5::Chains ch{}; ch.count = chains;
            for (int c = 0, r = 0; c < chains; ++c) { ch.rows[c] = layout[c]; ch.states[c] = s2[c].ls(); for (int i = 0; i < layout[c]; ++i, ++r) if (i + 1 < layout[c]) ch.saves[r] = sv2[r].ls(); }
            const uint64_t mx = mixed_copy.addr + uint64_t(r0) * CD * 2;
            ctx.begin();
            kt::must(k5::chain_core(ctx, f.w, ch, mx, f.row(f.z, r0, VD), f.row(f.b, r0, VH), f.row(f.a, r0, VH), rows, mx, f.ws_chain.addr, f.chain_bytes, f.table, 0),
                     "chain_core alias");
            ctx.barrier(); ctx.submit_wait();
            bool al = kt::host<uint16_t>(mixed_copy, size_t(rows) * VD, size_t(r0) * CD * 2) == y1;
            for (int c = 0; c < chains; ++c) al = al && same_state(s2[c], states[c]);
            report("y aliasing mixed (extra)", al, "y written over mixed: same y bytes and states");
            ctx.free(mixed_copy);
            for (auto &s : s2) free_state(ctx, s);
            for (auto &s : sv2) free_state(ctx, s);
        }
        for (auto *v : {&states, &refs, &saves, &snaps, &start}) for (auto &s : *v) free_state(ctx, s);
    }

    // ---- 4. negative rows
    {
        Data d = make_data(77, 32);
        Fixture f(ctx, d, 32);
        std::vector<State> st, sv;
        for (int i = 0; i < 4; ++i) st.push_back(new_state(ctx));
        for (int i = 0; i < 16; ++i) sv.push_back(new_state(ctx));
        for (auto &s : st) { std::memset(s.conv.map, 0x11, CONV_N * 2); std::memset(s.ssm.map, 0x22, SSM_N * 2); }
        std::memset(f.y.map, 0x33, size_t(32) * VD * 2);
        auto base = [&]() {  // a valid two-chain call (3 + 2 rows)
            k5::Chains ch{}; ch.count = 2; ch.rows[0] = 3; ch.rows[1] = 2; ch.states[0] = st[0].ls(); ch.states[1] = st[1].ls();
            ch.saves[0] = sv[0].ls(); ch.saves[1] = sv[1].ls(); ch.saves[3] = sv[3].ls();
            return ch;
        };
        int tried = 0, refused = 0;
        auto call = [&](const k5::Chains &ch, int64_t rows, uint64_t ws, size_t wsb, const Buffer &table, uint64_t toff = 0) {
            ++tried;
            const cere_status s = k5::chain_core(ctx, f.w, ch, f.mixed.addr, f.z.addr, f.b.addr, f.a.addr, rows, f.y.addr, ws, wsb, table, toff);
            if (s == CERE_ERR_ARGUMENT) ++refused; else std::printf("  bad call %d returned %d\n", tried, int(s));
        };
        ctx.begin();
        { auto c = base(); c.states[1] = st[0].ls(); call(c, 5, f.ws_chain.addr, f.chain_bytes, f.table); }                // chains share a state
        { auto c = base(); c.states[1].ssm = st[0].ssm.addr; call(c, 5, f.ws_chain.addr, f.chain_bytes, f.table); }        // share the ssm only
        { auto c = base(); call(c, 6, f.ws_chain.addr, f.chain_bytes, f.table); }                                          // rows != sum
        { auto c = base(); call(c, 4, f.ws_chain.addr, f.chain_bytes, f.table); }                                          // rows < sum
        { auto c = base(); c.saves[1] = {}; call(c, 5, f.ws_chain.addr, f.chain_bytes, f.table); }                         // null save
        { auto c = base(); c.saves[0] = st[1].ls(); call(c, 5, f.ws_chain.addr, f.chain_bytes, f.table); }                 // save aliases a state
        { auto c = base(); c.saves[0].conv = st[1].conv.addr; call(c, 5, f.ws_chain.addr, f.chain_bytes, f.table); }      // save conv aliases a state
        { auto c = base(); call(c, 0, f.ws_chain.addr, f.chain_bytes, f.table); }                                          // rows 0
        { k5::Chains c{}; c.count = 17; call(c, 17, f.ws_chain.addr, f.chain_bytes, f.table); }                            // rows / count 17
        { auto c = base(); c.count = 0; call(c, 5, f.ws_chain.addr, f.chain_bytes, f.table); }                             // count 0
        { auto c = base(); call(c, 5, f.ws_chain.addr, k5::chain_core_workspace_bytes(5) - 256, f.table); }               // small workspace
        { auto c = base(); call(c, 5, f.ws_chain.addr + 128, f.chain_bytes, f.table); }                                    // misaligned workspace
        { auto c = base(); call(c, 5, f.ws_chain.addr, f.chain_bytes, f.table, f.table.size - 512); }                     // table too small
        { auto c = base(); c.states[0] = {}; call(c, 5, f.ws_chain.addr, f.chain_bytes, f.table); }                        // null state
        ctx.barrier(); ctx.submit_wait();
        bool untouched = true;
        for (auto &s : st) {
            const auto cv = kt::host<uint8_t>(s.conv, CONV_N * 2), sm = kt::host<uint8_t>(s.ssm, SSM_N * 2);
            untouched = untouched && std::all_of(cv.begin(), cv.end(), [](uint8_t x) { return x == 0x11; }) &&
                        std::all_of(sm.begin(), sm.end(), [](uint8_t x) { return x == 0x22; });
        }
        const auto yy = kt::host<uint8_t>(f.y, size_t(32) * VD * 2);
        untouched = untouched && std::all_of(yy.begin(), yy.end(), [](uint8_t x) { return x == 0x33; });
        const bool sizes = k5::chain_core_workspace_bytes(0) == 0 && k5::chain_core_workspace_bytes(17) == 0 &&
                           k5::chain_core_workspace_bytes(4) == size_t(4) * CD * 2 + size_t(4) * VD * 2;
        char det[200];
        std::snprintf(det, sizeof det, "%d of %d bad calls ARGUMENT; states and y untouched after the submit: %s; workspace sizes as CUDA: %s", refused, tried,
                      untouched ? "yes" : "NO", sizes ? "yes" : "NO");
        report("negative: refusals record nothing", refused == tried && untouched && sizes, det);
        for (auto *v : {&st, &sv}) for (auto &s : *v) free_state(ctx, s);
    }
    std::printf("contract rows: %s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);

    {   // decode-shape timing (K1 batch method). States are updated in place by every call (the timed work is the same).
        Data d = make_data(91, 16);
        Fixture f(ctx, d, 16);
        std::vector<State> st, sv;
        for (int i = 0; i < 4; ++i) st.push_back(new_state(ctx));
        for (int i = 0; i < 16; ++i) sv.push_back(new_state(ctx));
        std::printf("\ndecode-shape timing: us per call = (GPU time between two timestamps around %d dependent calls, a barrier after each) / %d,\n"
                    "median of 7 such batches [min..max]; 3 untimed warm-up batches first\n", kBatch, kBatch);
        auto timed = [&](const char *name, const std::function<void()> &rec) {
            const int reps = 10;
            ctx.begin();
            for (int c = 0; c < reps; ++c) {
                ctx.stamp(2 * c);
                for (int i = 0; i < kBatch; ++i) { rec(); ctx.barrier(); }
                ctx.stamp(2 * c + 1);
            }
            ctx.submit_wait();
            std::vector<double> t; for (int c = 3; c < reps; ++c) t.push_back(ctx.stamp_ms(2 * c, 2 * c + 1) * 1e3 / kBatch);
            std::printf("TIME vulkan | %-40s | %9.2f us [%.2f..%.2f]\n", name, median(t), *std::min_element(t.begin(), t.end()), *std::max_element(t.begin(), t.end()));
        };
        for (const TShape &s : kShapes) {
            std::vector<int> layout(s.chains, s.rows_each);
            timed(s.name, [&] { kt::must(run_chains(f, layout, 0, st, sv, f.y.addr, false), "chain_core"); });
        }
        timed("layer_core DECODE 1 token (reference)", [&] {
            kt::must(k5::layer_core(ctx, f.w, st[0].ls(), CERE_OP_DECODE, f.mixed.addr, f.z.addr, f.b.addr, f.a.addr, 1, f.y.addr, f.ws_core.addr, f.core_bytes),
                     "layer_core");
        });
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

#else  // T3_HIP_TIMING: the same shapes and data through cere::k5::chain_core (hipified), hipEvents around each batch
#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)
using bf16 = __hip_bfloat16;
namespace k5 = cere::k5;
template <class T> static T *dev(const std::vector<T> &h) { T *p; HCK(hipMalloc(&p, h.size() * sizeof(T) + 256)); HCK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
static bf16 *B(uint16_t *p) { return reinterpret_cast<bf16 *>(p); }
static bf16 *zeros(size_t n) { void *p; HCK(hipMalloc(&p, n * 2 + 256)); HCK(hipMemset(p, 0, n * 2)); return static_cast<bf16 *>(p); }
int main() {
    hipDeviceProp_t prop; HCK(hipGetDeviceProperties(&prop, 0)); std::printf("device %s\n", prop.gcnArchName);
    Data d = make_data(91, 16);
    uint16_t *conv_w = dev(d.conv_w), *dt = dev(d.dt_bias), *norm = dev(d.norm_w), *mixed = dev(d.mixed), *z = dev(d.z), *b = dev(d.b), *a = dev(d.a);
    float *A_log = dev(d.A_log);
    k5::LayerWeights w{};
    w.conv1d = B(conv_w); w.A_log = A_log; w.dt_bias = B(dt); w.norm = B(norm);
    k5::LayerState st[4], sv[16];
    for (auto &s : st) { s.conv = zeros(CONV_N); s.ssm = zeros(SSM_N); }
    for (auto &s : sv) { s.conv = zeros(CONV_N); s.ssm = zeros(SSM_N); }
    const size_t wsb = k5::chain_core_workspace_bytes(16); void *ws; HCK(hipMalloc(&ws, wsb));
    bf16 *y = zeros(size_t(16) * VD);
    const size_t cwb = k5::core_workspace_bytes(1); void *cws; HCK(hipMalloc(&cws, cwb));
    HCK(hipDeviceSynchronize());
    std::printf("decode-shape timing: us per call = hipEvent time of %d back-to-back calls on one stream / %d, median of 7 batches [min..max]\n", kBatch, kBatch);
    hipStream_t s0; HCK(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
    auto timed = [&](const char *name, const std::function<cere_status()> &call) {
        const int reps = 10; std::vector<hipEvent_t> ev(2 * reps); for (auto &e : ev) HCK(hipEventCreate(&e));
        for (int c = 0; c < reps; ++c) {
            HCK(hipEventRecord(ev[2 * c], s0));
            for (int i = 0; i < kBatch; ++i) if (call() != CERE_OK) { std::printf("call failed\n"); std::exit(2); }
            HCK(hipEventRecord(ev[2 * c + 1], s0));
        }
        HCK(hipStreamSynchronize(s0));
        std::vector<double> t; for (int c = 3; c < reps; ++c) { float ms; HCK(hipEventElapsedTime(&ms, ev[2 * c], ev[2 * c + 1])); t.push_back(ms * 1e3 / kBatch); }
        std::printf("TIME hip-stream | %-40s | %9.2f us [%.2f..%.2f]\n", name, median(t), *std::min_element(t.begin(), t.end()), *std::max_element(t.begin(), t.end()));
    };
    for (const TShape &s : kShapes) {
        k5::Chains ch{}; ch.count = s.chains; int64_t rows = 0;
        for (int c = 0; c < s.chains; ++c) { ch.rows[c] = s.rows_each; ch.states[c] = st[c]; for (int i = 0; i < s.rows_each; ++i, ++rows) if (i + 1 < s.rows_each) ch.saves[rows] = sv[rows]; }
        timed(s.name, [&] { return k5::chain_core(w, ch, B(mixed), B(z), B(b), B(a), rows, y, ws, wsb, s0); });
    }
    timed("layer_core DECODE 1 token (reference)", [&] { return k5::layer_core(w, st[0], CERE_OP_DECODE, B(mixed), B(z), B(b), B(a), 1, y, cws, cwb, s0); });
    return 0;
}
#endif
