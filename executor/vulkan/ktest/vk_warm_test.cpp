// VulkanAmd P3 (pipeline warm-up, design 4.2): completeness of every kernel family's pipeline_specs list.
//
// For each family: a fresh Context, warm() of the family's list at the model's shapes (qwen38-flash-next: hidden 2560, hc 4 x
// lowrank 320, 512 experts top 10, MoE / shared inter 640, 24 q / 2 kv heads of 256, GDN 16 / 48 heads of 128, vocab 248,320),
// record pipeline_count(), then run a representative set of the family's public functions (decode m 1, the verify widths 5 / 8 /
// 16, a prefill size, BF16 and MXFP8 weights, drafts, both KV formats, ...) and assert that pipeline_count() did NOT grow: no
// pipeline was created lazily. Correctness of the outputs is NOT tested here (the buffers hold zeros); every call must return OK
// (a refused call would request no pipeline and prove nothing) and every submit must complete.
// Per family also:
//   exactness   : on a second, cold context the same calls run WITHOUT a warm-up; every list entry the calls requested is counted
//                 (used) and the unexercised ones are listed (a list that is longer than what the calls exercised is not a
//                 failure but is reported: "unused"). The pipelines the cold run created must all be in the list (same as the
//                 no-growth check, made from the other side: created == used + extra, extra must be 0).
//   negative    : a third context warms the list with one used entry removed and runs the calls: pipeline_count() must grow
//                 (by exactly 1), proving the no-growth check can fail.
// Finally the union of all families at the model shapes: warm() time on a cold context and the pipeline count.
// Usage: vk_warm_test <spv dir>. Exit 0 when every family passes.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "kernels/k12_sampler_vk.h"
#include "kernels/k13_moe_bf16_vk.h"
#include "kernels/k13s_spec_accept_vk.h"
#include "kernels/k1_primitives_vk.h"
#include "kernels/k2_gemm_vk.h"
#include "kernels/k2_mgemv_vk.h"
#include "kernels/k3_hc_vk.h"
#include "kernels/k4_ple_vk.h"
#include "kernels/k5_gdn_chain_vk.h"
#include "kernels/k5_gdn_chunk_vk.h"
#include "kernels/k5_gdn_layer_vk.h"
#include "kernels/k5_gdn_vk.h"
#include "kernels/k6_attention_vk.h"
#include "kernels/k6_prologue_vk.h"
#include "kernels/k7_moe_vk.h"
#include "kernels/k9_qsa_vk.h"
#include "kernels/mtp_kernels_vk.h"
#include "launch/vk_launch.h"

using namespace cere;
using Specs = std::vector<vk::PipelineSpec>;

static std::string g_dir;

// ---------------------------------------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------------------------------------
struct Arena {
    vk::Context &ctx;
    vk::Buffer b;
    uint64_t off = 0;
    Arena(vk::Context &c, uint64_t bytes) : ctx(c) {
        if (!ctx.alloc(bytes, b)) {
            std::printf("FATAL arena of %llu bytes: %s\n", (unsigned long long)bytes, ctx.last_error().c_str());
            std::exit(2);
        }
        std::memset(b.map, 0, bytes);
    }
    ~Arena() { ctx.free(b); }
    uint64_t take(uint64_t bytes) {
        const uint64_t a = b.addr + off;
        off += (bytes + 255) / 256 * 256 + 256;
        if (off > b.size) { std::printf("FATAL arena exhausted (%llu > %llu)\n", (unsigned long long)off, (unsigned long long)b.size); std::exit(2); }
        return a;
    }
    void *host(uint64_t addr) const { return static_cast<uint8_t *>(b.map) + (addr - b.addr); }
    // A vk::Buffer view of a sub-range (mapped, for the kernels' table parameters).
    vk::Buffer view(uint64_t bytes) {
        vk::Buffer v;
        v.addr = take(bytes);
        v.size = bytes;
        v.map = host(v.addr);
        return v;
    }
};

struct Rec {
    vk::Context &ctx;
    std::vector<std::string> errs;
    int calls = 0;
    explicit Rec(vk::Context &c) : ctx(c) {}
    // begin, the call, submit_wait: each call in its own command buffer; every status must be OK.
    void run(const char *what, const std::function<cere_status()> &f) {
        ++calls;
        cere_status st = ctx.begin();
        if (st != CERE_OK) { errs.push_back(std::string(what) + ": begin " + std::to_string(st)); return; }
        const cere_status cs = f();
        if (cs != CERE_OK) errs.push_back(std::string(what) + ": status " + std::to_string(cs));
        st = ctx.submit_wait();
        if (st != CERE_OK) errs.push_back(std::string(what) + ": submit " + std::to_string(st));
    }
};
// A probe submits its own command buffer: called directly, status recorded.
#define PROBE(r, call) do { ++(r).calls; const cere_status ps_ = (call); if (ps_ != CERE_OK) (r).errs.push_back(std::string(#call) + ": status " + std::to_string(ps_)); } while (0)
#define REC(r, call) (r).run(#call, [&]() -> cere_status { return (call); })

struct Family {
    std::string name;
    std::function<Specs()> specs;                         // at the model shapes (or the family's test shapes, said in the note)
    std::function<void(vk::Context &, Rec &)> calls;      // the representative calls
    std::string note;
};

static std::unique_ptr<vk::Context> new_ctx() {
    std::unique_ptr<vk::Context> c;
    std::string why;
    const cere_status s = vk::Context::create(0, &c, &why);
    if (s != CERE_OK) { std::printf("UNSUPPORTED/FATAL context: %s\n", why.c_str()); std::exit(3); }
    c->set_shader_dir(g_dir);
    return c;
}

static std::string spec_name(const vk::PipelineSpec &s) {
    std::string n = s.spv + " push " + std::to_string(s.push_bytes) + " sg " + std::to_string(s.required_subgroup);
    for (uint32_t v : s.spec) n += " spec " + std::to_string(v);
    return n;
}

static int g_fail = 0;

static bool run_family(const Family &f) {
    bool ok = true;
    const Specs list = f.specs();
    std::string detail;
    // --- 1. warm, then run: nothing may be created lazily -------------------------------------------------------------
    size_t warmed = 0, after = 0;
    int calls = 0;
    {
        auto c = new_ctx();
        std::string why;
        const cere_status sw = c->warm(list, &why);
        warmed = c->pipeline_count();
        if (sw != CERE_OK) { ok = false; detail += " warm FAILED (" + why + ");"; }
        Rec r(*c);
        f.calls(*c, r);
        calls = r.calls;
        after = c->pipeline_count();
        for (const std::string &e : r.errs) { ok = false; detail += " call error: " + e + ";"; }
        if (after != warmed) { ok = false; detail += " GREW " + std::to_string(warmed) + " -> " + std::to_string(after) + ";"; }
    }
    // --- 2. exactness: a cold run without warm-up; what it created vs the list ----------------------------------------
    size_t cold = 0, used = 0;
    std::vector<size_t> used_idx;
    std::string unused;
    {
        auto c = new_ctx();
        Rec r(*c);
        f.calls(*c, r);
        cold = c->pipeline_count();
        for (size_t i = 0; i < list.size(); ++i) {
            if (c->has_pipeline(list[i])) { ++used; used_idx.push_back(i); }
            else unused += (unused.empty() ? "" : ", ") + spec_name(list[i]);
        }
        if (cold != used) { ok = false; detail += " cold run created " + std::to_string(cold) + " pipelines but only " + std::to_string(used) + " are in the list;"; }
    }
    // --- 3. negative control: drop one used entry from the list ------------------------------------------------------
    bool neg_ok = false;
    std::string neg_detail = "no used entry";
    if (!used_idx.empty()) {
        Specs cut;
        const size_t drop = used_idx[used_idx.size() / 2];
        for (size_t i = 0; i < list.size(); ++i)
            if (i != drop) cut.push_back(list[i]);
        auto c = new_ctx();
        std::string why;
        c->warm(cut, &why);
        const size_t n0 = c->pipeline_count();
        Rec r(*c);
        f.calls(*c, r);
        const size_t n1 = c->pipeline_count();
        neg_ok = n1 == n0 + 1 && r.errs.empty();
        neg_detail = "dropped [" + spec_name(list[drop]) + "]: " + std::to_string(n0) + " -> " + std::to_string(n1);
    }
    if (!neg_ok) ok = false;
    std::printf("%-30s %s  list %3zu, warmed %3zu, calls %3d, after %3zu (grew %zd); cold run created %3zu (unused entries %zu)\n", f.name.c_str(),
                ok ? "PASS" : "FAIL", list.size(), warmed, calls, after, (ssize_t)after - (ssize_t)warmed, cold, list.size() - used);
    std::printf("%-30s %s  negative control: %s\n", "", neg_ok ? "PASS" : "FAIL", neg_detail.c_str());
    if (!unused.empty()) std::printf("%-30s       unused by the calls: %s\n", "", unused.c_str());
    if (!f.note.empty()) std::printf("%-30s       %s\n", "", f.note.c_str());
    if (!detail.empty()) std::printf("%-30s       %s\n", "", detail.c_str());
    if (!ok) ++g_fail;
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------------
// The model's shapes
// ---------------------------------------------------------------------------------------------------------------------
static const uint32_t kH = 2560, kLR = 320;
static const vkk3::HcShape kHc{4, kH, kLR, 1e-6f};
static const vkk7::MoeDims kModelDims{2560, 512, 10, 640, 640};  // the MoE blocks' model dims (K7 / K13 specs)
static const vkk7::MoeDims kTestDims{2560, 16, 4, 640, 640};     // the calls' dims: E 16 keeps the weights small; every pipeline selector is the same (epl16 as E <= 512)
static const vkk7::MoeDims kTestDims2{512, 16, 4, 256, 128};     // k < 512 (K13's 8-element variant), k <= 1024 lanes (K7)
static const std::vector<uint32_t> kBf16Ks = {320, 640, 2560, 6144, 10240};  // K3 up / shared down / hidden / out_proj / K3 down
static const std::vector<uint32_t> kMxKs = {640, 2560, 6144};

// ---------------------------------------------------------------------------------------------------------------------
// K1
// ---------------------------------------------------------------------------------------------------------------------
static void calls_k1(vk::Context &ctx, Rec &r) {
    Arena A(ctx, 64ull << 20);
    const uint64_t x = A.take(1 << 20), y = A.take(1 << 20), z = A.take(1 << 20), w = A.take(1 << 20), t = A.take(1 << 20), ids = A.take(1 << 16);
    const uint64_t cs = A.take(1 << 20), kc = A.take(1 << 20), vc = A.take(1 << 20), slots = A.take(1 << 16), at = A.take(1 << 16);
    const uint64_t qg = A.take(1 << 20), kk = A.take(1 << 20), qo = A.take(1 << 20), ko = A.take(1 << 20), go = A.take(1 << 20);
    const uint64_t lg = A.take(4 << 20), outp = A.take(1 << 16);
    REC(r, vkk1::embedding_gather(ctx, t, 32, 2560, ids, 1, y));
    REC(r, vkk1::rmsnorm(ctx, x, 2560, w, 4, true, 1e-6f, 4, 2560, y, 2560));
    REC(r, vkk1::rmsnorm(ctx, x, 128, w, 48, false, 1e-6f, 48, 128, y, 128));
    REC(r, vkk1::rmsnorm_gated_sigmoid(ctx, x, 128, z, 128, w, 1e-6f, 48, 128, y, 128));
    REC(r, vkk1::rmsnorm_gated_sigmoid(ctx, x, 2560, z, 2560, w, 1e-6f, 4, 2560, y, 2560));
    REC(r, vkk1::rope_table(ctx, cs, 128, 64, 1e7f));
    REC(r, vkk1::rope(ctx, x, 2048, y, 2048, cs, 0, 0, 0, 1, 8, 256, 64));
    REC(r, vkk1::sigmoid_mul(ctx, x, 6144, z, 6144, 1, 6144, y, 6144));
    REC(r, vkk1::top2_split(ctx, lg, 248320, 1, 248320, outp, outp + 64));
    REC(r, vkk1::kv_store_fp8(ctx, x, 512, y, 512, 1, 512, kc, vc, 0, 0));
    REC(r, vkk1::qk_norm_rope_gate(ctx, qg, 12288, kk, 512, w, w, cs, 0, 0, 0, 1, 24, 2, 256, 64, 1e-6f, qo, ko, go));
    REC(r, vkk1::kv_store_fp8_at(ctx, x, 512, y, 512, 1, 512, kc, vc, slots, at));
    {
        vkk1::KvRows rows{};
        for (int i = 0; i < 16; ++i) { rows.k_cache[i] = kc; rows.v_cache[i] = vc; rows.slots[i] = slots; }
        vk::Buffer tb = A.view(1024);
        REC(r, vkk1::kv_store_fp8_rows(ctx, x, 512, y, 512, 16, 512, rows, at, tb));
        vk::Buffer tb2 = A.view(1024);
        vkk1::CopySegment seg{kc, 1024, vc, 1024, 512, 2};
        REC(r, vkk1::copy_segments(ctx, &seg, 1, tb2));
    }
    REC(r, vkk1::top2(ctx, lg, 248320, 5, 248320, outp));
}

// ---------------------------------------------------------------------------------------------------------------------
// K2
// ---------------------------------------------------------------------------------------------------------------------
static void calls_k2_gemv(vk::Context &ctx, Rec &r) {
    Arena A(ctx, 128ull << 20);
    const uint32_t n = 16;
    const uint64_t x = A.take(16 * 10240 * 2), y = A.take(16 * 4096 * 4);
    for (uint32_t k : kBf16Ks) {
        const uint64_t w = A.take(uint64_t(n) * k * 2);
        for (uint32_t m = 1; m <= 16; ++m) REC(r, vkk2::gemv(ctx, x, w, 0, y, m, n, k, (m & 1) != 0));
    }
    for (uint32_t k : kMxKs) {
        const uint64_t q = A.take(uint64_t(n) * k), s = A.take(uint64_t(n) * k / 32);
        for (uint32_t m = 1; m <= 16; ++m) REC(r, vkk2::gemv(ctx, x, q, s, y, m, n, k, (m & 1) == 0));
        const uint64_t wd = A.take(uint64_t(n) * k * 2);
        REC(r, vkk2::mxfp8_dequantize(ctx, q, s, n, k, wd));
    }
}

static void calls_k2_gemm(vk::Context &ctx, Rec &r) {
    PROBE(r, vkk2::gemm_probe(ctx));
    Arena A(ctx, 256ull << 20);
    const uint32_t m = 300, n = 256, k = 2560;
    const uint64_t x = A.take(uint64_t(m) * k * 2), y = A.take(uint64_t(m) * n * 4), ws = A.take(vkk2::gemm_workspace_bytes(m, n)),
                   w = A.take(uint64_t(n) * k * 2), q = A.take(uint64_t(n) * k), s = A.take(uint64_t(n) * k / 32), wi = A.take(n * 4 + 64);
    REC(r, vkk2::gemm_prepare_weight(ctx, w, 0, n, k, wi));
    REC(r, vkk2::gemm_prepare_weight(ctx, q, s, n, k, wi));
    for (int f32 = 0; f32 < 2; ++f32) {
        REC(r, vkk2::gemm(ctx, x, w, 0, y, m, n, k, f32 != 0, ws));
        REC(r, vkk2::gemm(ctx, x, q, s, y, m, n, k, f32 != 0, ws, wi));
    }
    vkk2::GemmOptions fo;
    fo.route = vkk2::GemmRoute::kFp32;
    REC(r, vkk2::gemm(ctx, x, w, 0, y, m, n, k, false, ws, 0, &fo));
    REC(r, vkk2::run(ctx, x, w, 0, y, 8, n, k, false, 0, ws));         // class 8: the GEMV
    REC(r, vkk2::run(ctx, x, w, 0, y, 5, n, k, false, 20, ws));        // m 5 standing for class 20: the GEMM
    REC(r, vkk2::run(ctx, x, q, s, y, 17, n, k, true, 0, ws));         // MXFP8 class 17
    REC(r, vkk2::run(ctx, x, q, s, y, 3, n, k, true, 0, ws));          // MXFP8 GEMV (the routing's other side)
}

// ---------------------------------------------------------------------------------------------------------------------
// K3
// ---------------------------------------------------------------------------------------------------------------------
static void calls_k3(vk::Context &ctx, Rec &r) {
    PROBE(r, vkk2::gemm_probe(ctx));
    const uint32_t maxm = 300, W = 4 * kH;
    Arena A(ctx, 256ull << 20);
    const uint64_t cap = uint64_t(maxm) * W * 2;
    const uint64_t hyper = A.take(cap), nw = A.take(W * 2), wd = A.take(uint64_t(kLR) * W * 2), wu = A.take(uint64_t(W) * kLR * 2), normed = A.take(cap),
                   mixed = A.take(cap * 2), block = A.take(cap), inj = A.take(cap), out = A.take(cap), nnw = A.take(W * 2);
    vkk3::HcPlans plans, rows_plan;
    plans.init(kHc, maxm);
    rows_plan.init_rows(plans, maxm);
    const uint64_t ws = A.take(vkk3::hc_mix_workspace_bytes(kHc, maxm) + 256);
    for (uint32_t m : {1u, 2u, 3u, 5u, 8u, 16u, 17u, 300u}) REC(r, vkk3::hc_mix(ctx, plans, hyper, nw, wd, wu, m, normed, mixed, ws));
    REC(r, vkk3::hc_mix(ctx, plans, hyper, nw, wd, wu, 5, normed, mixed, ws, vkk3::HcMixPath::kAuto, 20));
    REC(r, vkk3::hc_mix(ctx, rows_plan, hyper, nw, wd, wu, 40, normed, mixed, ws));
    REC(r, vkk3::hc_mix_normed(ctx, plans, normed, wd, wu, 8, mixed, ws));
    REC(r, vkk3::hc_mix_normed(ctx, plans, normed, wd, wu, 64, mixed, ws));
    for (uint32_t m : {1u, 16u, 17u, 300u}) {
        REC(r, vkk3::hc_combine(ctx, kHc, block, hyper, normed, inj, m, out));
        REC(r, vkk3::hc_combine_norm(ctx, kHc, block, hyper, normed, inj, nnw, m, out));
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// K4
// ---------------------------------------------------------------------------------------------------------------------
static void calls_k4(vk::Context &ctx, Rec &r) {
    PROBE(r, vkk2::gemm_probe(ctx));
    Arena A(ctx, 256ull << 20);
    const uint64_t mult = A.take(64), hvs = A.take(16 * 8), hoff = A.take(16 * 8);
    int64_t *m3 = static_cast<int64_t *>(A.host(mult));
    for (int i = 0; i < 3; ++i) m3[i] = 1 + i;
    int64_t *hv = static_cast<int64_t *>(A.host(hvs)), *ho = static_cast<int64_t *>(A.host(hoff));
    for (int i = 0; i < 16; ++i) { hv[i] = 97; ho[i] = 0; }
    const uint64_t key = A.take(uint64_t(vkk4::kChannels) * vkk4::kEmbedDim * 2), val = A.take(uint64_t(vkk4::kHidden) * vkk4::kEmbedDim * 2);
    const uint64_t nk = A.take(vkk4::kChannels * 2), nq = A.take(vkk4::kChannels * 2), nc = A.take(vkk4::kChannels * 2), cw = A.take(vkk4::kChannels * 8);
    const uint64_t pinfo = A.take(vkk4::proj_info_bytes());
    const uint64_t hist = A.take(64), conv = A.take(vkk4::kChannels * vkk4::kConvStateLen * 2), hist2 = A.take(64), conv2 = A.take(vkk4::kChannels * vkk4::kConvStateLen * 2);
    const uint64_t tokens = A.take(64 * 8), query = A.take(64 * vkk4::kChannels * 2), out = A.take(64 * vkk4::kChannels * 2), emb = A.take(64 * vkk4::kEmbedDim * 2);
    const uint64_t scale = A.take(64), gscale = A.take(64);
    *static_cast<float *>(A.host(gscale)) = 1.0f;
    vk::Buffer call_table = A.view(16384);
    const uint64_t ws = A.take(vkk4::workspace_bytes(64));
    vkk4::SplitTable tab[2];
    for (int nv = 0; nv < 2; ++nv) {
        std::string why;
        if (!tab[nv].create(ctx, nv ? vkk4::TableFormat::kNvfp4 : vkk4::TableFormat::kFp8, 100, 1 << 20, vkk4::Memory::kDefault, &why)) {
            r.errs.push_back("SplitTable: " + why);
            return;
        }
    }
    for (int nv = 0; nv < 2; ++nv) {
        vkk4::Weights w;
        w.format = nv ? vkk4::TableFormat::kNvfp4 : vkk4::TableFormat::kFp8;
        w.table = tab[nv].view();
        w.table_scale = nv ? 0 : scale;
        w.global_scale = nv ? gscale : 0;
        w.multipliers = mult; w.head_vocab_sizes = hvs; w.head_offsets = hoff;
        w.key_proj = key; w.value_proj = val; w.norm_key = nk; w.norm_query = nq; w.norm_conv = nc; w.conv_weight = cw;
        w.proj_info = nv ? pinfo : 0;
        if (nv) REC(r, vkk4::prepare_projections(ctx, w));
        const vkk4::State st{hist, conv};
        for (int64_t t : {1, 2, 3, 5, 8, 16, 40}) REC(r, vkk4::forward(ctx, w, st, tokens, t, query, (t & 1) != 0, out, ws, vkk4::workspace_bytes(64), call_table, 0));
        REC(r, vkk4::forward(ctx, w, st, tokens, 5, query, true, out, ws, vkk4::workspace_bytes(64), call_table, 0, nullptr, 20));
        {
            const vkk4::Capture cap{10, hist2, conv2};
            REC(r, vkk4::forward(ctx, w, st, tokens, 16, query, false, out, ws, vkk4::workspace_bytes(64), call_table, 0, &cap));
        }
        {
            vkk4::RowsCall call;
            call.sequences = 2;
            call.states[0] = {hist, conv};
            call.states[1] = {hist2, conv2};
            call.first[0] = 0; call.count[0] = 3;
            call.first[1] = 3; call.count[1] = 2;
            REC(r, vkk4::forward_rows(ctx, w, call, tokens, 5, query, true, out, ws, vkk4::workspace_bytes(64), call_table, 0));
        }
        REC(r, vkk4::embed(ctx, w, hist, tokens, 8, emb, call_table, 0));
        if (nv) REC(r, vkk4::gather_nvfp4(ctx, w.table, gscale, tokens, 4, emb));
        else REC(r, vkk4::gather(ctx, w.table, tokens, 4, scale, emb));
    }
    REC(r, vkk4::ngram_hash_windows(ctx, A.take(64 * 8 * 4), 64, mult, hvs, hoff, vkk4::kEosTokenId, A.take(64 * 16 * 8)));
    for (int nv = 0; nv < 2; ++nv) tab[nv].destroy(ctx);
}

// ---------------------------------------------------------------------------------------------------------------------
// K5
// ---------------------------------------------------------------------------------------------------------------------
static void calls_k5_pieces(vk::Context &ctx, Rec &r) {
    Arena A(ctx, 128ull << 20);
    const uint64_t x = A.take(8 << 20), w = A.take(1 << 20), st = A.take(4 << 20), y = A.take(8 << 20), a = A.take(1 << 16), b = A.take(1 << 16), al = A.take(1 << 16),
                   dt = A.take(1 << 16), g = A.take(1 << 20), beta = A.take(1 << 20), qkv = A.take(4 << 20), o = A.take(4 << 20);
    vkk5::DecodeArgs da{qkv, qkv + 4096, qkv + 8192, a, b, al, dt, st, o, 0.0883883f};
    REC(r, vkk5::delta_rule_decode(ctx, da));
    vk::Buffer tb = A.view(4096);
    vkk5::DecodeRowsArgs ra{};
    ra.step = da;
    ra.chains = 2; ra.first[0] = 0; ra.rows[0] = 3; ra.first[1] = 3; ra.rows[1] = 2;
    ra.states[0] = st; ra.states[1] = st + (1 << 20);
    for (int i : {0, 1, 3}) ra.saves[i] = st + (2 << 20) + uint64_t(i) * 65536;
    ra.qkv_stride = 10240; ra.ab_stride = 48; ra.o_stride = 6144;
    REC(r, vkk5::delta_rule_decode_rows(ctx, ra, tb, 0));
    REC(r, vkk5::conv1d_silu(ctx, x, 10240, w, st, y, 10240, 1, 10240));
    REC(r, vkk5::conv1d_silu(ctx, x, 2048, w, st, y, 2048, 70, 2048));
    REC(r, vkk5::conv1d_state_at(ctx, x, 10240, st, 64, 10240));
    vkk5::ConvChains cc{};
    cc.chains = 2; cc.first[0] = 0; cc.rows[0] = 3; cc.first[1] = 3; cc.rows[1] = 2;
    cc.states[0] = st; cc.states[1] = st + (1 << 20);
    for (int i : {0, 1, 3}) cc.saves[i] = st + (2 << 20) + uint64_t(i) * 65536;
    REC(r, vkk5::conv1d_silu_rows(ctx, x, 10240, w, cc, y, 10240, 10240, tb, 0));
    REC(r, vkk5::gating(ctx, a, 48, b, 48, al, dt, g, beta, 70, 48));
    REC(r, vkk5::l2norm(ctx, x, y, 70 * 16, 128, 1e-6f));
}

static void calls_k5_chunk(vk::Context &ctx, Rec &r) {
    Arena A(ctx, 256ull << 20);
    const int64_t T = 70;
    const uint64_t q = A.take(T * 2048 * 2), k = A.take(T * 2048 * 2), v = A.take(T * 6144 * 2), g = A.take(T * 48 * 4), beta = A.take(T * 48 * 4),
                   st = A.take(48 * 128 * 128 * 4), o = A.take(T * 6144 * 2), ws = A.take(vkk5::chunk_workspace_bytes(T)), c32 = A.take(48 * 128 * 128 * 4),
                   i32 = A.take(48 * 128 * 128 * 4);
    vkk5::ChunkArgs ca;
    ca.q = q; ca.k = k; ca.v = v; ca.g = g; ca.beta = beta; ca.state = st; ca.o = o; ca.tokens = T; ca.scale = 0.0883883f;
    REC(r, vkk5::delta_rule_chunk(ctx, ca, ws));
    ca.state_in32 = i32; ca.capture32 = c32; ca.capture_chunk = 1;
    REC(r, vkk5::delta_rule_chunk(ctx, ca, ws));
    ca.tokens = 4096;
    ca.capture32 = 0; ca.state_in32 = 0;
    // a long prefill chunk: same pipelines
    Arena B(ctx, 512ull << 20);
    const int64_t T2 = 1024;
    vkk5::ChunkArgs cb;
    cb.q = B.take(T2 * 2048 * 2); cb.k = B.take(T2 * 2048 * 2); cb.v = B.take(T2 * 6144 * 2); cb.g = B.take(T2 * 48 * 4); cb.beta = B.take(T2 * 48 * 4);
    cb.state = B.take(48 * 128 * 128 * 4); cb.o = B.take(T2 * 6144 * 2); cb.tokens = T2; cb.scale = 0.0883883f;
    REC(r, vkk5::delta_rule_chunk(ctx, cb, B.take(vkk5::chunk_workspace_bytes(T2))));
}

static vkk5::LayerWeights make_layer_weights(Arena &A) {
    vkk5::LayerWeights w{};
    w.in_proj_qkv = A.take(10240ull * kH * 2);
    w.in_proj_z = A.take(6144ull * kH * 2);
    w.in_proj_b = A.take(48ull * kH * 2);
    w.in_proj_a = A.take(48ull * kH * 2);
    w.conv1d = A.take(10240 * 4 * 2);
    w.A_log = A.take(48 * 4);
    w.dt_bias = A.take(48 * 2);
    w.norm = A.take(128 * 2);
    w.out_proj = A.take(uint64_t(kH) * 6144 * 2);
    return w;
}

static void calls_k5_layer(vk::Context &ctx, Rec &r) {
    PROBE(r, vkk2::gemm_probe(ctx));
    Arena A(ctx, 768ull << 20);
    vkk5::LayerWeights w = make_layer_weights(A);
    const uint64_t conv = A.take(10240 * 3 * 2), ssm = A.take(48 * 128 * 128 * 4);
    const vkk5::LayerState state{conv, ssm};
    const int64_t maxt = 300;
    const uint64_t hidden = A.take(maxt * kH * 2), out = A.take(maxt * kH * 2);
    const uint64_t ws = A.take(vkk5::layer_workspace_bytes(maxt));
    const size_t wsb = vkk5::layer_workspace_bytes(maxt);
    w.info = A.take(vkk5::weight_info_bytes(kH));
    REC(r, vkk5::prepare_weights(ctx, w, kH));
    REC(r, vkk5::layer_forward(ctx, w, state, CERE_OP_DECODE, hidden, 1, kH, out, ws, wsb));
    for (int64_t t : {2, 3, 5, 16, 17, 300}) REC(r, vkk5::layer_forward(ctx, w, state, CERE_OP_PREFILL, hidden, t, kH, out, ws, wsb));
    REC(r, vkk5::layer_forward(ctx, w, state, CERE_OP_PREFILL, hidden, 5, kH, out, ws, wsb, 20));
    {
        vkk5::EndState end;
        end.ssm_in32 = A.take(48 * 128 * 128 * 4);
        end.ssm_out32 = A.take(48 * 128 * 128 * 4);
        end.conv_out = A.take(10240 * 3 * 2);
        end.row = 64;
        REC(r, vkk5::layer_forward(ctx, w, state, CERE_OP_PREFILL, hidden, 130, kH, out, ws, wsb, 0, &end));
    }
    // layer_core on its own (decode and prefill)
    const uint64_t mixed = A.take(maxt * 10240 * 2), z = A.take(maxt * 6144 * 2), b = A.take(maxt * 48 * 2), a = A.take(maxt * 48 * 2), y = A.take(maxt * 6144 * 2);
    const uint64_t cws = A.take(vkk5::core_workspace_bytes(maxt));
    REC(r, vkk5::layer_core(ctx, w, state, CERE_OP_DECODE, mixed, z, b, a, 1, y, cws, vkk5::core_workspace_bytes(maxt)));
    REC(r, vkk5::layer_core(ctx, w, state, CERE_OP_PREFILL, mixed, z, b, a, 70, y, cws, vkk5::core_workspace_bytes(maxt)));
}

// The layer with MXFP8 projections (the S3 artifact's format: E4M3 bytes [n, k] + UE8M0 scales [n, k / 32]).
static void calls_k5_layer_mx(vk::Context &ctx, Rec &r) {
    PROBE(r, vkk2::gemm_probe(ctx));
    Arena A(ctx, 512ull << 20);
    vkk5::LayerWeights w = make_layer_weights(A);
    const uint64_t H = kH;
    w.in_proj_qkv = A.take(10240 * H); w.in_proj_qkv_scales = A.take(10240 * H / 32);
    w.in_proj_z = A.take(6144 * H); w.in_proj_z_scales = A.take(6144 * H / 32);
    w.in_proj_b = A.take(48 * H); w.in_proj_b_scales = A.take(48 * H / 32);
    w.in_proj_a = A.take(48 * H); w.in_proj_a_scales = A.take(48 * H / 32);
    w.out_proj = A.take(H * 6144); w.out_proj_scales = A.take(H * 6144 / 32);
    const uint64_t conv = A.take(10240 * 3 * 2), ssm = A.take(48 * 128 * 128 * 4);
    const vkk5::LayerState state{conv, ssm};
    const int64_t maxt = 300;
    const uint64_t hidden = A.take(maxt * kH * 2), out = A.take(maxt * kH * 2);
    const uint64_t ws = A.take(vkk5::layer_workspace_bytes(maxt));
    const size_t wsb = vkk5::layer_workspace_bytes(maxt);
    REC(r, vkk5::layer_forward(ctx, w, state, CERE_OP_DECODE, hidden, 1, kH, out, ws, wsb));  // in-call weight scans
    for (int64_t t : {5, 17, 300}) REC(r, vkk5::layer_forward(ctx, w, state, CERE_OP_PREFILL, hidden, t, kH, out, ws, wsb));
    w.info = A.take(vkk5::weight_info_bytes(kH));
    REC(r, vkk5::prepare_weights(ctx, w, kH));
    REC(r, vkk5::layer_forward(ctx, w, state, CERE_OP_DECODE, hidden, 1, kH, out, ws, wsb));
    for (int64_t t : {2, 3, 5, 16, 17, 300}) REC(r, vkk5::layer_forward(ctx, w, state, CERE_OP_PREFILL, hidden, t, kH, out, ws, wsb));
    REC(r, vkk5::layer_forward(ctx, w, state, CERE_OP_PREFILL, hidden, 5, kH, out, ws, wsb, 20));
}

static void calls_k5_chain(vk::Context &ctx, Rec &r) {
    Arena A(ctx, 128ull << 20);
    vkk5::LayerWeights w{};
    w.conv1d = A.take(10240 * 4 * 2); w.A_log = A.take(48 * 4); w.dt_bias = A.take(48 * 2); w.norm = A.take(128 * 2);
    const int64_t rows = 5;
    const uint64_t mixed = A.take(rows * 10240 * 2), z = A.take(rows * 6144 * 2), b = A.take(rows * 48 * 2), a = A.take(rows * 48 * 2), y = A.take(rows * 6144 * 2);
    const uint64_t ws = A.take(vkk5::chain_core_workspace_bytes(rows));
    vk::Buffer tb = A.view(vkk5::kChainCoreTableBytes + 64);
    vkk5::Chains ch;
    ch.count = 2; ch.rows[0] = 3; ch.rows[1] = 2;
    for (int i = 0; i < 2; ++i) ch.states[i] = {A.take(10240 * 3 * 2), A.take(48 * 128 * 128 * 4)};
    for (int i : {0, 1, 3}) ch.saves[i] = {A.take(10240 * 3 * 2), A.take(48 * 128 * 128 * 4)};
    REC(r, vkk5::chain_core(ctx, w, ch, mixed, z, b, a, rows, y, ws, vkk5::chain_core_workspace_bytes(rows), tb, 0));
}

// ---------------------------------------------------------------------------------------------------------------------
// K6
// ---------------------------------------------------------------------------------------------------------------------
static void calls_k6_attention(vk::Context &ctx, Rec &r) {
    Arena A(ctx, 256ull << 20);
    const int64_t kMaxT = 40;
    const uint64_t q = A.take(kMaxT * 6144 * 2), kb = A.take(2048 * 512 * 2), vb = A.take(2048 * 512 * 2), kc = A.take(2048 * 512), vc = A.take(2048 * 512),
                   slots = A.take(2048 * 8), out = A.take(kMaxT * 6144 * 2), gate = A.take(kMaxT * 6144 * 2);
    const vkk6::Options opts[] = {{64, 64}, {128, 32}};
    for (const vkk6::Options &o : opts) {
        const uint64_t ws = A.take(vkk6::workspace_bytes(kMaxT, o));
        const size_t wsb = vkk6::workspace_bytes(kMaxT, o);
        for (int64_t t : {1, 5, 8, 16, 17, 40}) {
            REC(r, vkk6::attention(ctx, q, t, 100, kb, 512, vb, 512, 0, 0, 0, 0.0625f, ws, wsb, out, 0, o));
            REC(r, vkk6::attention(ctx, q, t, 100, 0, 0, 0, 0, kc, vc, slots, 0.0625f, ws, wsb, out, 0, o));
            REC(r, vkk6::dense_attention(ctx, q, kb, vb, 512, gate, t, 100, false, 0, 0, 0, ws, wsb, out, 6144, 0, o));
            REC(r, vkk6::dense_attention(ctx, q, 0, 0, 0, gate, t, 100, true, kc, vc, slots, ws, wsb, out, 6144, 0, o));
        }
        REC(r, vkk6::attention(ctx, q, 5, 100, kb, 512, vb, 512, 0, 0, 0, 0.0625f, ws, wsb, out, 20, o));   // class 20: prefill kernel
        for (int64_t t : {1, 5, 16}) REC(r, vkk6::dense_attention_rows(ctx, q, gate, t, 100, kc, vc, slots, ws, wsb, out, 6144, o));
    }
}

static void calls_k6_prologue(vk::Context &ctx, Rec &r) {
    Arena A(ctx, 128ull << 20);
    const uint64_t qg = A.take(40 * 12288 * 2), k = A.take(40 * 512 * 2), v = A.take(40 * 512 * 2), qn = A.take(256 * 2), kn = A.take(256 * 2), cs = A.take(4096 * 64 * 4),
                   kc = A.take(4096 * 512), vc = A.take(4096 * 512), qo = A.take(40 * 6144 * 2), ko = A.take(40 * 512 * 2), go = A.take(40 * 6144 * 2),
                   slots = A.take(4096 * 8), pos = A.take(64 * 8);
    for (int64_t t : {1, 5, 16, 40}) REC(r, vkk6::prologue(ctx, qg, 12288, k, 512, v, 512, qn, kn, 1e-6f, cs, 4096, 0, 0, slots, t, 0, kc, vc, qo, ko, go));
    REC(r, vkk6::prologue(ctx, qg, 12288, k, 512, v, 512, qn, kn, 1e-6f, cs, 4096, pos, 0, 0, 5, 100, kc, vc, qo, ko, go));
}

// ---------------------------------------------------------------------------------------------------------------------
// K7 / K13
// ---------------------------------------------------------------------------------------------------------------------
static vkk7::MoeLayer make_k7_layer(Arena &A, const vkk7::MoeDims &d) {
    vkk7::MoeLayer L;
    const uint64_t E = d.num_experts, H = d.hidden, I = d.inter, S = d.shared_inter;
    L.router = A.take(E * H * 2);
    L.gate_weight = A.take(E * I * H / 2); L.gate_scale = A.take(E * I * H / 16);
    L.up_weight = A.take(E * I * H / 2); L.up_scale = A.take(E * I * H / 16);
    L.down_weight = A.take(E * H * I / 2); L.down_scale = A.take(E * H * I / 16);
    L.g1_alphas = A.take(E * 4); L.g2_alphas = A.take(E * 4);
    for (uint64_t e = 0; e < E; ++e) {
        static_cast<float *>(A.host(L.g1_alphas))[e] = 1.0f;
        static_cast<float *>(A.host(L.g2_alphas))[e] = 1.0f;
    }
    L.a1_global_scale = 1.0f; L.a2_global_scale = 1.0f;
    L.shared_gate = A.take(S * H * 2); L.shared_up = A.take(S * H * 2); L.shared_down = A.take(H * S * 2); L.shared_expert_gate = A.take(H * 2);
    return L;
}

static void calls_k7_dims(vk::Context &ctx, Rec &r, const vkk7::MoeDims &d, bool options) {
    Arena A(ctx, 768ull << 20);
    const vkk7::MoeLayer L = make_k7_layer(A, d);
    const uint32_t maxm = 80;
    const uint64_t x = A.take(uint64_t(maxm) * d.hidden * 2), out = A.take(uint64_t(maxm) * d.hidden * 2), ws = A.take(vkk7::moe_workspace_bytes(d, maxm));
    const size_t wsb = vkk7::moe_workspace_bytes(d, maxm);
    PROBE(r, vkk7::probe(ctx));
    PROBE(r, vkk2::gemm_probe(ctx));
    REC(r, vkk7::moe_route(ctx, A.take(maxm * d.num_experts * 2), A.take(maxm * d.top_k * 4), A.take(maxm * d.top_k * 4), 5, d.num_experts, d.top_k));
    if (!options) {
        for (uint32_t m : {1u, 2u, 3u, 5u, 8u, 16u, 17u, 25u, 80u}) REC(r, vkk7::moe_block(ctx, d, L, x, m, out, ws, wsb));
        vkk7::MoeOptions o;
        o.cls = 20;
        REC(r, vkk7::moe_block(ctx, d, L, x, 5, out, ws, wsb, &o));   // class 20 (D15 resume part): K2 GEMM products on the GEMV path
        return;
    }
    // Non-default options: GemvTiles::kExpert at m 1..8, GroupPath::kSplit, GemmPath::kGemv / kCoopmat forced.
    for (uint32_t m = 1; m <= 8; ++m) {
        vkk7::MoeOptions o;
        o.tiles = vkk7::GemvTiles::kExpert;
        REC(r, vkk7::moe_block(ctx, d, L, x, m, out, ws, wsb, &o));
    }
    for (uint32_t m : {1u, 5u, 16u}) {
        vkk7::MoeOptions o;
        o.group = vkk7::GroupPath::kSplit;
        REC(r, vkk7::moe_block(ctx, d, L, x, m, out, ws, wsb, &o));
        o.gemm = vkk7::GemmPath::kCoopmat;
        REC(r, vkk7::moe_block(ctx, d, L, x, m, out, ws, wsb, &o));
        vkk7::MoeOptions g;
        g.gemm = vkk7::GemmPath::kGemv;
        g.group = vkk7::GroupPath::kFused;
        REC(r, vkk7::moe_block(ctx, d, L, x, m, out, ws, wsb, &g));
    }
    for (uint32_t m : {1u, 17u, 80u}) REC(r, vkk7::moe_block(ctx, d, L, x, m, out, ws, wsb));
}
// The block with an MXFP8 shared expert (the S3 format; the router stays BF16).
static void calls_k7_mx_dims(vk::Context &ctx, Rec &r, const vkk7::MoeDims &d) {
    Arena A(ctx, 768ull << 20);
    vkk7::MoeLayer L = make_k7_layer(A, d);
    const uint64_t H = d.hidden, S = d.shared_inter;
    L.shared_gate = A.take(S * H); L.shared_gate_scales = A.take(S * H / 32);
    L.shared_up = A.take(S * H); L.shared_up_scales = A.take(S * H / 32);
    L.shared_down = A.take(H * S); L.shared_down_scales = A.take(H * S / 32);
    const uint32_t maxm = 80;
    const uint64_t x = A.take(uint64_t(maxm) * d.hidden * 2), out = A.take(uint64_t(maxm) * d.hidden * 2), ws = A.take(vkk7::moe_workspace_bytes(d, maxm));
    const size_t wsb = vkk7::moe_workspace_bytes(d, maxm);
    PROBE(r, vkk7::probe(ctx));
    PROBE(r, vkk2::gemm_probe(ctx));
    for (uint32_t m : {1u, 2u, 3u, 5u, 8u, 16u, 17u, 25u, 80u}) REC(r, vkk7::moe_block(ctx, d, L, x, m, out, ws, wsb));
    vkk7::MoeOptions o;
    o.cls = 20;
    REC(r, vkk7::moe_block(ctx, d, L, x, 5, out, ws, wsb, &o));
    L.bf16_info = A.take(vkk7::moe_weight_info_bytes(d));
    REC(r, vkk7::moe_prepare_weights(ctx, d, L));
    for (uint32_t m : {1u, 17u}) REC(r, vkk7::moe_block(ctx, d, L, x, m, out, ws, wsb));
}
static void calls_k7_mx(vk::Context &ctx, Rec &r) { calls_k7_mx_dims(ctx, r, kTestDims); calls_k7_mx_dims(ctx, r, kTestDims2); }
static void calls_k7(vk::Context &ctx, Rec &r) { calls_k7_dims(ctx, r, kTestDims, false); calls_k7_dims(ctx, r, kTestDims2, false); }
static void calls_k7_opts(vk::Context &ctx, Rec &r) { calls_k7_dims(ctx, r, kTestDims, true); calls_k7_dims(ctx, r, kTestDims2, true); }

static void calls_k13_dims(vk::Context &ctx, Rec &r, const vkk7::MoeDims &d) {
    Arena A(ctx, 1024ull << 20);
    const uint64_t E = d.num_experts, H = d.hidden, I = d.inter, S = d.shared_inter;
    vkk13::MoeBf16Layer bf, fp;
    bf.router = fp.router = A.take(E * H * 2);
    bf.shared_gate = fp.shared_gate = A.take(S * H * 2);
    bf.shared_up = fp.shared_up = A.take(S * H * 2);
    bf.shared_down = fp.shared_down = A.take(H * S * 2);
    bf.shared_expert_gate = fp.shared_expert_gate = A.take(H * 2);
    bf.gate_up = A.take(E * 2 * I * H * 2);
    bf.down = A.take(E * H * I * 2);
    fp.gate_up_q = A.take(E * 2 * I * H / 2); fp.gate_up_scale = A.take(E * 2 * I * H / 16); fp.gate_up_global = A.take(E * 2 * 4);
    fp.down_q = A.take(E * H * I / 2); fp.down_scale = A.take(E * H * I / 16); fp.down_global = A.take(E * 4);
    for (uint64_t i = 0; i < E * 2; ++i) static_cast<float *>(A.host(fp.gate_up_global))[i] = 1.0f;
    for (uint64_t i = 0; i < E; ++i) static_cast<float *>(A.host(fp.down_global))[i] = 1.0f;
    const uint32_t maxm = 80;
    const uint64_t x = A.take(uint64_t(maxm) * H * 2), out = A.take(uint64_t(maxm) * H * 2), ws = A.take(vkk13::moe_bf16_workspace_bytes(d, maxm));
    const size_t wsb = vkk13::moe_bf16_workspace_bytes(d, maxm);
    PROBE(r, vkk2::gemm_probe(ctx));
    for (const vkk13::MoeBf16Layer *L : {&bf, &fp}) {
        for (uint32_t m : {1u, 2u, 3u, 5u, 8u, 16u, 17u, 80u}) REC(r, vkk13::moe_bf16_block(ctx, d, *L, x, m, 0, out, ws, wsb));
        REC(r, vkk13::moe_bf16_block(ctx, d, *L, x, 5, 4, out, ws, wsb));   // one tail row of a 5-row call (one-row GEMV)
        vkk13::MoeBf16Options o;
        o.group = vkk7::GroupPath::kSplit;
        REC(r, vkk13::moe_bf16_block(ctx, d, *L, x, 5, 0, out, ws, wsb, &o));
        o.group = vkk7::GroupPath::kAuto;
        o.cls = 20;
        REC(r, vkk13::moe_bf16_block(ctx, d, *L, x, 5, 0, out, ws, wsb, &o));
    }
}
static void calls_k13(vk::Context &ctx, Rec &r) { calls_k13_dims(ctx, r, kTestDims); calls_k13_dims(ctx, r, kTestDims2); }

// ---------------------------------------------------------------------------------------------------------------------
// K9
// ---------------------------------------------------------------------------------------------------------------------
static void calls_k9(vk::Context &ctx, Rec &r) {
    Arena A(ctx, 512ull << 20);
    const vkk9::Options opt{64};
    const int64_t ctxlen = 64;
    const uint64_t qk = A.take(40 * 640 * 2), qnw = A.take(256), knw = A.take(256), cs = A.take(4096 * 64 * 4), fault = A.take(16), pos = A.take(64 * 8),
                   slots = A.take(4096 * 8), ring = A.take(1 << 20), keys = A.take(1 << 20), qidx = A.take(40 * 4 * 128 * 2), kidx = A.take(40 * 128 * 2),
                   saved = A.take(1 << 16), logits = A.take(40 * 64 * 4), out = A.take(40 * 2051 * 4), scratch = A.take(16 << 20);
    const vkk9::Bounds b{0, 4096, 4096, fault};
    REC(r, vkk9::indexer_qk(ctx, qk, 640, qnw, 1e-6f, cs, b, 0, 0, 4, qidx));
    REC(r, vkk9::qsa_compress_store(ctx, kidx, 128, knw, 1e-6f, cs, b, 0, slots, 0, 9, ring, keys));
    REC(r, vkk9::qsa_compress_store_step(ctx, kidx, knw, 1e-6f, cs, b, slots, pos, ring, keys));
    REC(r, vkk9::qsa_compress_store_at(ctx, kidx, 128, knw, 1e-6f, cs, b, slots, pos, 5, ring, keys));
    REC(r, vkk9::qsa_key_save_step(ctx, keys, slots, b, pos, 1, 4, saved));
    {
        vkk9::ChainsStep rows{};
        rows.chain_first[0] = 0; rows.chain_rows[0] = 3;
        rows.chain_first[1] = 3; rows.chain_rows[1] = 2;
        for (int i = 0; i < 5; ++i) rows.context[i] = 64;
        vk::Buffer tb = A.view(vkk9::kChainsTableBytes + 64);
        REC(r, vkk9::qsa_chains_step(ctx, rows, 2, knw, 1e-6f, cs, b, pos, tb));
    }
    for (int impl : {0, 1, 2})
        for (int64_t rows : {1, 9}) REC(r, vkk9::qsa_logits(ctx, qidx, keys, slots, 0, rows, ctxlen, logits, ctxlen / 4, impl, opt));
    for (int impl : {0, 1}) REC(r, vkk9::qsa_topk(ctx, logits, ctxlen / 4, 0, 9, ctxlen, impl, out, opt));
    for (int64_t rows : {1, 9}) REC(r, vkk9::qsa_select(ctx, qidx, keys, slots, 0, rows, ctxlen, scratch, 16 << 20, out, opt));
    const vkk9::Bounds bstep{0, 4096, ctxlen, fault};
    REC(r, vkk9::qsa_select_step(ctx, qidx, keys, slots, bstep, pos, ctxlen, scratch, 16 << 20, out, opt));
    // sparse attention: BF16 KV (k, v) and the FP8 cache; split (class <= 16) and rows (> 16) paths; every rows_heads tile.
    const uint64_t q = A.take(40 * 6144 * 2), kb = A.take(2048 * 512 * 2), vb = A.take(2048 * 512 * 2), kc = A.take(2048 * 512), vc = A.take(2048 * 512),
                   idx = A.take(40 * 2051 * 4), o = A.take(40 * 6144 * 2), s2 = A.take(8 << 20);
    for (int heads : {0, 3, 6, 12})
        for (int64_t rows : {1, 5, 20}) {
            REC(r, vkk9::qsa_attention_heads(ctx, q, 6144, idx, rows, kb, 512, vb, 512, 0, 0, 0, 0.0625f, s2, 8 << 20, o, 6144, false, 0, heads, opt));
            REC(r, vkk9::qsa_attention_heads(ctx, q, 6144, idx, rows, 0, 0, 0, 0, kc, vc, slots, 0.0625f, s2, 8 << 20, o, 6144, false, 0, heads, opt));
        }
    REC(r, vkk9::qsa_attention(ctx, q, 6144, idx, 5, 0, 0, 0, 0, kc, vc, slots, 0.0625f, s2, 8 << 20, o, 6144, true, 20, opt));
}

// ---------------------------------------------------------------------------------------------------------------------
// K12 / K13S / MTP
// ---------------------------------------------------------------------------------------------------------------------
static void calls_k12(vk::Context &ctx, Rec &r) {
    const int64_t V = 248320;
    Arena A(ctx, 64ull << 20);
    const uint64_t logits = A.take(V * 2), proc = A.take(V * 4), counts = A.take(V * 4), mask = A.take(V / 8 + 64), bids = A.take(64 * 4), bvals = A.take(64 * 4),
                   ids = A.take(64 * 4), scores = A.take(64 * 4), probs = A.take(V * 4), lse = A.take(64), scratch = A.take(vkk12::scratch_bytes()),
                   keep = A.take(V / 8 + 64), us = A.take(64 * 8), seeds = A.take(64 * 8), drawn = A.take(64), fin = A.take(256), cid = A.take(64 * 4),
                   clp = A.take(64 * 4), tids = A.take(256 * 4), tlp = A.take(256 * 4), top2 = A.take(64);
    REC(r, vkk12::process(ctx, logits, V, 0, 0.f, 0.f, 1.f, 0, 0, 0, 0, 0, proc));
    REC(r, vkk12::process(ctx, logits, V, counts, 0.1f, 0.1f, 1.1f, mask, bids, bvals, 4, 4, proc));
    REC(r, vkk12::top2_f32(ctx, proc, V, ids, scores));
    REC(r, vkk12::probabilities(ctx, proc, V, 0.7f, probs, lse, scratch));
    REC(r, vkk12::keep_mask(ctx, probs, V, 40, 0.9f, 0.01f, keep, scratch));
    REC(r, vkk12::keep_mask(ctx, probs, V, 5000, 0.9f, 0.01f, keep, scratch));   // top_k > 1024: the masked pass
    REC(r, vkk12::draw_u(ctx, probs, keep, V, us, 3, ids, scratch));
    REC(r, vkk12::draw_seeded(ctx, probs, keep, V, seeds, 3, 7, ids, scratch));
    REC(r, vkk12::finalize(ctx, proc, top2, drawn, fin));
    REC(r, vkk12::top_logprobs(ctx, proc, V, 0.7f, cid, 0, 3, clp, tids, tlp, 20));
    REC(r, vkk12::count_commit(ctx, counts, ids, 5));
}

static void calls_k13s(vk::Context &ctx, Rec &r) {
    const int64_t V = 248320, n = 5;
    Arena A(ctx, 64ull << 20);
    const uint64_t probs = A.take((n + 1) * V * 4), masks = A.take((n + 1) * (V / 32 + 1) * 4), drafts = A.take(64 * 4), committed = A.take(64 * 4),
                   count = A.take(64), scratch = A.take(vkk13s::scratch_bytes(vkk13s::kMaxDrafts)), ids = A.take(1 << 12), acc = A.take(1 << 12),
                   tsc = A.take(vkk13s::trial_scratch_bytes());
    REC(r, vkk13s::spec_accept(ctx, probs, masks, V, drafts, n, 1, 0, committed, count, scratch));
    REC(r, vkk13s::spec_accept(ctx, probs, masks, V, 0, 0, 1, 0, committed, count, scratch));   // n = 0: only the bonus draw
    REC(r, vkk13s::row_trials(ctx, probs, masks, V, 7, 1, 1, 0, 1, 300, ids, acc, tsc));
}

static void calls_mtp(vk::Context &ctx, Rec &r) {
    Arena A(ctx, 64ull << 20);
    const uint64_t x = A.take(16 * 10240 * 2), w = A.take(10240 * 2), y = A.take(16 * 10240 * 2), e = A.take(16 * 10240 * 2), logits = A.take(16 * 248320 * 2),
                   map = A.take(248320 * 4), draft = A.take(64);
    REC(r, vkmtp::norm_hidden(ctx, x, w, 1e-6f, 5, y));
    REC(r, vkmtp::add_embedding(ctx, x, e, 5));
    REC(r, vkmtp::argmax_map(ctx, logits, 248320, map, 5, draft));
    REC(r, vkmtp::argmax_map(ctx, logits + 2, 248319, map, 1, draft));   // an unaligned row: the scalar path (same pipeline)
    REC(r, vkmtp::detail::gemma_rows(ctx, x, w, 1e-6f, 5, 2560, y));
}

// ---------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc < 2) { std::printf("usage: vk_warm_test <spv dir>\n"); return 2; }
    g_dir = argv[1];
    {
        auto c = new_ctx();
        const vk::DeviceInfo &i = c->info();
        std::printf("device: %s, driver %s %s, subgroup %u (%u..%u), coopmat %d\n", i.name.c_str(), i.driver_name.c_str(), i.driver_info.c_str(), i.subgroup_default,
                    i.subgroup_min, i.subgroup_max, int(i.coopmat));
        if (!i.coopmat) { std::printf("SKIP: the K2 GEMM / K7 coopmat families need cooperative matrices\n"); return 3; }
    }
    const std::vector<int> norm_dims = {2560, 128};
    const std::vector<Family> fams = {
        {"K1 primitives", [&] { return vkk1::pipeline_specs(norm_dims, {256}); }, calls_k1, "norm dims {2560, 128}, qk head_dim {256}"},
        {"K2 small-M GEMV", [&] { return vkk2::pipeline_specs(kBf16Ks, kMxKs, true); }, calls_k2_gemv, "BF16 k {320, 640, 2560, 6144, 10240}, MXFP8 k {640, 2560, 6144}, m 1..16, dequant"},
        {"K2 large-m GEMM", [&] { return vkk2::pipeline_specs(true, true, {2560}, {2560}); }, calls_k2_gemm, "BF16 + MXFP8, gemm / run (class <= 16 routes to the GEMV at k 2560) / prepare_weight / probe"},
        {"K3 hyper-connections", [&] { return vkk3::pipeline_specs(kHc, true); }, calls_k3, "HcShape{4, 2560, 320}, m 1 / 5 / 8 / 16 / 17 / 300, cls, rows plan"},
        {"K4 PLE", [&] { return vkk4::pipeline_specs(true, true, true); }, calls_k4, "FP8 and NVFP4 tables, forward t 1..40, cls 20, capture, forward_rows, embed, gather, hash"},
        {"K5 decode / pieces", [&] { return vkk5::piece_pipeline_specs(vkk5::kPiecesAll); }, calls_k5_pieces, "all seven pieces"},
        {"K5 chunk", [&] { return vkk5::chunk_pipeline_specs(); }, calls_k5_chunk, "T 70 (state in / capture), T 1024"},
        {"K5 layer", [&] { return vkk5::pipeline_specs({kH}, true); }, calls_k5_layer, "decode, prefill 5 / 16 / 17 / 300 / 130 with end state, cls 20, layer_core, prepare_weights"},
        {"K5 layer (MXFP8)", [&] { return vkk5::pipeline_specs({kH}, true, true); }, calls_k5_layer_mx,
         "MXFP8 projections (S3): decode, prefill 2..300, cls 20, in-call scans and prepare_weights"},
        {"K5 chain", [&] { return vkk5::pipeline_specs(); }, calls_k5_chain, "2 chains over 5 rows"},
        {"K6 attention", [&] { Specs s = vkk6::pipeline_specs({64, 64}, true, true); vkk2::append_unique(s, vkk6::pipeline_specs({128, 32}, true, true)); return s; },
         calls_k6_attention, "Options {64, 64} and {128, 32}; BF16 + FP8 KV; tokens 1 / 5 / 8 / 16 / 17 / 40, dense, rows, cls"},
        {"K6 prologue", [&] { return vkk6::pipeline_specs(); }, calls_k6_prologue, "tokens 1 / 5 / 16 / 40, with and without positions / slots"},
        {"K7 MoE (NVFP4)", [&] { Specs s = vkk7::pipeline_specs(kTestDims, true, true, false); vkk2::append_unique(s, vkk7::pipeline_specs(kTestDims2, true, true, false)); return s; },
         calls_k7, "calls at MoeDims{2560, 16, 4, 640, 640} and {512, 16, 4, 256, 128} (E 16 keeps weights small; E <= 512: the model's epl16 selector); m 1..80, cls 20"},
        {"K7 MoE (option variants)", [&] { Specs s = vkk7::pipeline_specs(kTestDims, true, true, true); vkk2::append_unique(s, vkk7::pipeline_specs(kTestDims2, true, true, true)); return s; },
         calls_k7_opts, "GemvTiles::kExpert m 1..8, GroupPath::kSplit / kFused, GemmPath forced"},
        {"K7 MoE (MXFP8 shared)", [&] { Specs s = vkk7::pipeline_specs(kTestDims, true, true, false, true); vkk2::append_unique(s, vkk7::pipeline_specs(kTestDims2, true, true, false, true)); return s; },
         calls_k7_mx, "MXFP8 shared expert (S3), BF16 router, both test dims, m 1..80, cls 20, moe_prepare_weights"},
        {"K9 QSA", [&] { return vkk9::pipeline_specs({64}, true, true, {0, 3, 6, 12}); }, calls_k9, "Options {64}; BF16 + FP8 KV; rows 1 / 5 / 20; rows_heads 0 / 3 / 6 / 12; impl 0 / 1 / 2"},
        {"K12 sampler", [&] { return vkk12::pipeline_specs(248320); }, calls_k12, "vocab 248,320, top_k 40 and 5000, bias, draws"},
        {"K13S spec accept", [&] { return vkk13s::pipeline_specs(); }, calls_k13s, "5 drafts, 0 drafts, row_trials"},
        {"K13 MoE BF16/NVFP4", [&] { Specs s = vkk13::pipeline_specs(kTestDims, true, true, true); vkk2::append_unique(s, vkk13::pipeline_specs(kTestDims2, true, true, true)); return s; },
         calls_k13, "BF16 and NVFP4 experts at the two test dims (k 2560 / 640 and k 512 / 256: both lane variants), tail 1 and > 1, cls 20, kSplit"},
        {"MTP kernels", [&] { return vkmtp::pipeline_specs({2560}); }, calls_mtp, "norm_hidden (10240), add_embedding, argmax_map, gemma_rows(2560)"},
    };
    int n = 0;
    for (const Family &f : fams) { run_family(f); ++n; }

    // ---- the union at the model's shapes: cold warm-up time and count ---------------------------------------------------
    Specs all;
    const auto add = [&](const Specs &s) { vkk2::append_unique(all, s); };
    add(vkk1::pipeline_specs({2560, 128}, {256}));
    add(vkk2::pipeline_specs(kBf16Ks, kMxKs, true));
    add(vkk2::pipeline_specs(true, true));
    add(vkk3::pipeline_specs(kHc, true));
    add(vkk4::pipeline_specs(true, true, true));
    add(vkk5::piece_pipeline_specs(vkk5::kPiecesAll));
    add(vkk5::chunk_pipeline_specs());
    add(vkk5::pipeline_specs({kH}, true));
    add(vkk5::pipeline_specs({kH}, true, true));
    add(vkk5::pipeline_specs());
    add(vkk6::pipeline_specs({64, 64}, true, true));
    add(vkk6::pipeline_specs());
    add(vkk7::pipeline_specs(kModelDims, true, true, false));
    add(vkk7::pipeline_specs(kModelDims, true, true, false, true));
    add(vkk9::pipeline_specs({64}, true, true, {vkk9::kRowsHeads}));
    add(vkk12::pipeline_specs(248320));
    add(vkk13s::pipeline_specs());
    add(vkk13::pipeline_specs(kModelDims, true, true, true));
    add(vkmtp::pipeline_specs());
    {
        auto c = new_ctx();
        std::string why;
        const auto t0 = std::chrono::steady_clock::now();
        const cere_status s = c->warm(all, &why);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const bool ok = s == CERE_OK && c->pipeline_count() == all.size();
        std::printf("UNION at the model shapes: %zu specs, pipeline_count %zu, warm() on a cold context %.1f ms (%.2f ms per pipeline), status %d %s  %s\n", all.size(),
                    c->pipeline_count(), ms, ms / double(all.size()), int(s), why.c_str(), ok ? "PASS" : "FAIL");
        if (!ok) ++g_fail;
        // a second warm() of the same list: all cached, nothing created
        const auto t1 = std::chrono::steady_clock::now();
        c->warm(all, &why);
        std::printf("UNION second warm() (all cached): %.3f ms, pipeline_count %zu\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count(), c->pipeline_count());
    }
    std::printf("%d families, %d FAIL\n", n, g_fail);
    return g_fail ? 1 : 0;
}
