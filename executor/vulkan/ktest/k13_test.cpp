// VulkanAmd P2: executor/hip/ktest/k13_test.cpp's contract rows on the Vulkan K13 BF16 MoE block (k13_moe_bf16_vk.h),
// same shapes (H 256, E 16, top-4, I 128, S 128), seed (47), data generation, FP64 mirror and bounds:
//  1. BF16 experts vs the host FP64 mirror (K7's router / routing / combine, gate and up rounded to BF16, SwiGLU in
//     FP32 rounded to BF16, down rounded to BF16): rel-L2 <= 4e-3 at m 3 and 40; repeat bit-identical;
//  2. NVFP4 W4A16 experts: out bit-equal to the BF16 call on the BF16(RN(RN(e2m1 * scale) * global)) tensors, m 3, 40;
//  3. `first`: rows [first, m) bit-equal to the first = 0 call, rows [0, first) not written (m 9, first 6);
//  4. "rows plans": each row of a 5-row call == that row alone, bit for bit (no plans on Vulkan: HIP's rows plans run
//     the small-M GEMV (k2_gemm_adapter.h run_base), which a class <= 16 call runs here too, so the 5-row call is the
//     plain call).
// The router / shared K2 products route by row class (MoeBf16Options::cls, k13_moe_bf16.h `plans.cls`): <= 16 the K2
// GEMV, >= 17 the K2 GEMM; m 40 and its class-40 comparisons run the GEMM on both sides.
// Added rows: K2 GEMM probe; split vs fused grouping bit-equal; prepared K2 weight info == in-call scan; refusals
// (cls < m too); at the model's shape (H 2,560, E 512, top-10, I 640, S 640, NVFP4 experts as the S3 artifact's MTP
// layer): the draft-extend call (m 5, first 4) == the m 1 call on row 4 (class <= 16 both), an m 64 call's rows ==
// alone at the same class (cls 64: the HIP k7_test.cpp:157 / :167 class pin), rel-L2 vs the mirror; then timing (GPU
// timestamps, median) at the MTP draft's widths.
// Usage: k13_test [spv_dir] [--no-timing] [--no-model] [--bf16-timing (model-shape BF16 experts, 5 GB)] [--explore]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "launch/vk_launch_testing.h"
#include "kernels/k13_moe_bf16_vk.h"
#include "kernels/k2_gemm_vk.h"
#include "ktest_util.h"

namespace kv = cere::vkk13;
using cere::vk::Buffer;
using cere::vk::Context;
using kt::bf2f;
using kt::f2bf;
static int g_fail = 0;
static void check(const char *name, bool ok, const char *msg) {
    std::printf("%-22s %s  %s\n", name, ok ? "PASS" : "FAIL", msg);
    std::fflush(stdout);
    if (!ok) ++g_fail;
}
static float rb(double x) { return bf2f(f2bf(float(x))); }
static float e4m3f(uint8_t c) { const int s = c >> 7, e = (c >> 3) & 15, m = c & 7; const float v = e == 0 ? std::ldexp(m / 8.f, -6) : std::ldexp(1.f + m / 8.f, e - 7); return s ? -v : v; }
static const float kE2[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

// One layer: NVFP4 experts (host + device) and, optionally, their BF16 dequantization (the HIP test's deq).
struct Model {
    kv::MoeDims d;
    std::vector<uint16_t> router, sg, su, sd, seg, gu, dn;
    std::vector<uint8_t> guq, gus, dq, dsc;
    std::vector<float> gug, dg;
    std::vector<Buffer> bufs;
    kv::MoeBf16Layer Lb, Lq;
    // the BF16 value of an NVFP4 weight: BF16(RN(RN(e2m1 * scale) * global)) (the HIP test's deq)
    static uint16_t deq1(const std::vector<uint8_t> &q, const std::vector<uint8_t> &sc, size_t idx, float glob) {
        const int code = (q[idx / 2] >> (4 * (idx % 2))) & 15; const float v = (code & 8) ? -kE2[code & 7] : kE2[code & 7];
        return f2bf((v * e4m3f(sc[idx / 16])) * glob);
    }
    float w_gu(size_t e, size_t row, size_t i) const {  // row in [0, 2 I)
        const size_t idx = (e * 2 * d.inter + row) * d.hidden + i;
        return gu.empty() ? bf2f(deq1(guq, gus, idx, gug[e * 2 + (row >= d.inter ? 1 : 0)])) : bf2f(gu[idx]);
    }
    float w_dn(size_t e, size_t o, size_t j) const {
        const size_t idx = (e * d.hidden + o) * d.inter + j;
        return dn.empty() ? bf2f(deq1(dq, dsc, idx, dg[e])) : bf2f(dn[idx]);
    }
    void make_bf16() {
        const uint32_t H = d.hidden, E = d.num_experts, I = d.inter;
        gu.resize(size_t(E) * 2 * I * H); dn.resize(size_t(E) * H * I);
        for (size_t r = 0; r < size_t(E) * 2 * I; ++r) { const size_t e = r / (2 * I); const float gl = gug[e * 2 + ((r % (2 * I)) >= I ? 1 : 0)]; for (uint32_t c = 0; c < H; ++c) gu[r * H + c] = deq1(guq, gus, r * H + c, gl); }
        for (size_t r = 0; r < size_t(E) * H; ++r) { const float gl = dg[r / H]; for (uint32_t c = 0; c < I; ++c) dn[r * I + c] = deq1(dq, dsc, r * I + c, gl); }
    }
    template <class T> uint64_t up(Context &ctx, const std::vector<T> &h) { bufs.push_back(kt::upload(ctx, h)); return bufs.back().addr; }
    void upload(Context &ctx, bool bf16) {
        kv::MoeBf16Layer L{};
        L.router = up(ctx, router); L.shared_gate = up(ctx, sg); L.shared_up = up(ctx, su); L.shared_down = up(ctx, sd); L.shared_expert_gate = up(ctx, seg);
        Lq = L;
        Lq.gate_up_q = up(ctx, guq); Lq.gate_up_scale = up(ctx, gus); Lq.gate_up_global = up(ctx, gug);
        Lq.down_q = up(ctx, dq); Lq.down_scale = up(ctx, dsc); Lq.down_global = up(ctx, dg);
        Lb = L;
        if (bf16) { Lb.gate_up = up(ctx, gu); Lb.down = up(ctx, dn); }
    }
    void release(Context &ctx) { for (auto &b : bufs) ctx.free(b); bufs.clear(); }
    // The HIP test's FP64 mirror of one BF16 row.
    std::vector<double> mirror(const uint16_t *x16) const {
        const uint32_t H = d.hidden, E = d.num_experts, K = d.top_k, I = d.inter, S = d.shared_inter;
        std::vector<double> out(H); std::vector<float> xr(H); for (uint32_t i = 0; i < H; ++i) xr[i] = bf2f(x16[i]);
        std::vector<float> lg(E); for (uint32_t e = 0; e < E; ++e) { double s = 0; for (uint32_t i = 0; i < H; ++i) s += double(xr[i]) * bf2f(router[size_t(e) * H + i]); lg[e] = rb(s); }
        const float mx = *std::max_element(lg.begin(), lg.end()); std::vector<float> p(E); float sum = 0; for (uint32_t e = 0; e < E; ++e) { p[e] = std::exp(lg[e] - mx); sum += p[e]; }
        std::vector<uint32_t> idx(E); for (uint32_t e = 0; e < E; ++e) idx[e] = e;
        std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return lg[a] != lg[b] ? lg[a] > lg[b] : a < b; });
        float ks = 0; for (uint32_t k = 0; k < K; ++k) ks += p[idx[k]] / sum;
        std::vector<double> f(H, 0.0);
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t e = idx[k]; const float wk = (p[e] / sum) / ks; std::vector<float> h(I);
            for (uint32_t j = 0; j < I; ++j) { double a = 0, b = 0; for (uint32_t i = 0; i < H; ++i) { a += double(xr[i]) * w_gu(e, j, i); b += double(xr[i]) * w_gu(e, I + j, i); }
                const float ga = rb(a), ub = rb(b); h[j] = rb(double(ga) / (1 + std::exp(-double(ga))) * ub); }
            for (uint32_t o = 0; o < H; ++o) { double dd = 0; for (uint32_t j = 0; j < I; ++j) dd += double(h[j]) * w_dn(e, o, j); f[o] += double(wk) * rb(dd); }
        }
        std::vector<float> sh(S);
        for (uint32_t j = 0; j < S; ++j) { double a = 0, b = 0; for (uint32_t i = 0; i < H; ++i) { a += double(xr[i]) * bf2f(sg[size_t(j) * H + i]); b += double(xr[i]) * bf2f(su[size_t(j) * H + i]); }
            const float ga = rb(a), ub = rb(b); sh[j] = rb(double(ga) / (1 + std::exp(-double(ga))) * ub); }
        double gd = 0; for (uint32_t i = 0; i < H; ++i) gd += double(xr[i]) * bf2f(seg[i]);
        const double sig = 1 / (1 + std::exp(-gd));
        for (uint32_t o = 0; o < H; ++o) { double dd = 0; for (uint32_t j = 0; j < S; ++j) dd += double(sh[j]) * bf2f(sd[size_t(o) * S + j]); out[o] = double(rb(f[o])) + sig * rb(dd); }
        return out;
    }
};

// The HIP test's generation, in its order (one engine and one normal distribution throughout, as the HIP test).
struct Gen {
    std::mt19937 g; std::normal_distribution<float> nd;
    explicit Gen(uint32_t seed) : g(seed) {}
    std::vector<uint16_t> rnd(size_t n, float s) { std::vector<uint16_t> v(n); for (auto &x : v) x = f2bf(nd(g) * s); return v; }
};
static void make_model(Model &M, Gen &G, const kv::MoeDims &d) {
    std::mt19937 &g = G.g;
    auto rnd = [&](size_t n, float s) { return G.rnd(n, s); };
    const uint32_t H = d.hidden, E = d.num_experts, I = d.inter, S = d.shared_inter;
    M.d = d;
    M.router = rnd(size_t(E) * H, 0.1f); M.sg = rnd(size_t(S) * H, 1.f / 16); M.su = rnd(size_t(S) * H, 1.f / 16);
    M.sd = rnd(size_t(H) * S, 1.f / 11); M.seg = rnd(H, 0.06f);
    M.guq.resize(size_t(E) * 2 * I * H / 2); M.gus.resize(size_t(E) * 2 * I * H / 16); M.dq.resize(size_t(E) * H * I / 2); M.dsc.resize(size_t(E) * H * I / 16);
    for (auto &b : M.guq) b = uint8_t(g());
    for (auto &b : M.dq) b = uint8_t(g());
    for (auto &b : M.gus) b = uint8_t(0x30 + g() % 0x18);
    for (auto &b : M.dsc) b = uint8_t(0x30 + g() % 0x18);
    M.gug.resize(E * 2); M.dg.resize(E);
    for (auto &x : M.gug) x = 0.01f * (1 + 0.1f * float(g() % 8));
    for (auto &x : M.dg) x = 0.008f * (1 + 0.1f * float(g() % 8));
}

static std::vector<uint16_t> run(Context &ctx, const Model &M, const kv::MoeBf16Layer &L, uint64_t x, uint32_t m, uint32_t first,
                                 Buffer &out, Buffer &ws, const kv::MoeBf16Options *o = nullptr) {
    ctx.begin();
    kt::must(kv::moe_bf16_block(ctx, M.d, L, x, m, first, out.addr, ws.addr, ws.size, o), "moe_bf16_block");
    ctx.barrier();
    ctx.submit_wait();
    return kt::host<uint16_t>(out, size_t(m) * M.d.hidden);
}
static long diff_rows(const std::vector<uint16_t> &a, size_t ar, const std::vector<uint16_t> &b, size_t br, size_t rows, uint32_t H) {
    long n = 0; for (size_t i = 0; i < rows * H; ++i) n += a[ar * H + i] != b[br * H + i]; return n;
}
static double rel_l2(const Model &M, const std::vector<uint16_t> &x, const std::vector<uint16_t> &out, const std::vector<uint32_t> &rows) {
    const uint32_t H = M.d.hidden; double num = 0, den = 0;
    for (uint32_t r : rows) { const auto ref = M.mirror(&x[size_t(r) * H]); for (uint32_t o = 0; o < H; ++o) { const double e = bf2f(out[size_t(r) * H + o]) - ref[o]; num += e * e; den += ref[o] * ref[o]; } }
    return std::sqrt(num / den);
}

int main(int argc, char **argv) {
    std::string spv = "build/vulkan-k13/spv";
    bool timing = true, model = true, bf16_timing = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--no-timing") timing = false;
        else if (a == "--no-model") model = false;
        else if (a == "--bf16-timing") bf16_timing = true;
        else spv = a;
    }
    auto ctx_owner = cere::vk::testing::create_or_die(spv);
    cere::vk::Context &ctx = *ctx_owner;
    std::printf("device: %s, driver %s %s\n", ctx.info().name.c_str(), ctx.info().driver_name.c_str(), ctx.info().driver_info.c_str());
    check("K2 GEMM probe", cere::vkk2::gemm_probe(ctx) == CERE_OK, "k2_gemm_probe: the K2 GEMM's accumulator map (router / shared row classes >= 17)");

    // ---- the HIP test's shape and data
    const uint32_t H = 256, E = 16, K = 4, I = 128, S = 128, MAXM = 64;
    Gen G(47);
    Model M;
    make_model(M, G, kv::MoeDims{H, E, K, I, S});
    M.make_bf16();
    M.upload(ctx, true);
    Buffer ws = kt::make(ctx, kv::moe_bf16_workspace_bytes(M.d, MAXM)), dout = kt::make(ctx, size_t(MAXM) * H * 2);
    const auto x = G.rnd(size_t(MAXM) * H, 1.f);
    Buffer dx = kt::upload(ctx, x);
    for (uint32_t m : {3u, 40u}) {  // 1, 2
        const auto out = run(ctx, M, M.Lb, dx.addr, m, 0, dout, ws);
        std::vector<uint32_t> rows(m); for (uint32_t r = 0; r < m; ++r) rows[r] = r;
        const double err = rel_l2(M, x, out, rows);
        const bool rep = run(ctx, M, M.Lb, dx.addr, m, 0, dout, ws) == out;
        char msg[160]; std::snprintf(msg, sizeof msg, "m %2u: out rel-L2 %.2e vs FP64 (4e-3); repeat %s", m, err, rep ? "bit-equal" : "DIFFERS");
        check("moe_bf16_block", err <= 4e-3 && rep, msg);
        const auto outq = run(ctx, M, M.Lq, dx.addr, m, 0, dout, ws);
        const long diff = diff_rows(outq, 0, out, 0, m, H);
        std::snprintf(msg, sizeof msg, "m %2u: NVFP4 experts vs the BF16 call on their dequantized tensors: %ld of %zu differ", m, diff, out.size());
        check("NVFP4 == BF16 (S2)", diff == 0, msg);
    }
    {   // 3. first
        const auto full = run(ctx, M, M.Lb, dx.addr, 9, 0, dout, ws);
        std::memset(dout.map, 0x55, size_t(9) * H * 2);
        const auto part = run(ctx, M, M.Lb, dx.addr, 9, 6, dout, ws);
        const bool tail = diff_rows(part, 6, full, 6, 3, H) == 0;
        bool head = true; for (size_t i = 0; i < 6 * H; ++i) head &= part[i] == 0x5555;
        check("first", tail && head, tail && head ? "m 9, first 6: rows 6..8 == the first 0 call, rows 0..5 untouched" : "rows DIFFER or written");
    }
    {   // 4. rows (plans)
        const auto all = run(ctx, M, M.Lb, dx.addr, 5, 0, dout, ws); long diff = 0;
        for (uint32_t r = 0; r < 5; ++r) diff += diff_rows(run(ctx, M, M.Lb, dx.addr + uint64_t(r) * H * 2, 1, 0, dout, ws), 0, all, r, 1, H);
        char msg[160]; std::snprintf(msg, sizeof msg, "each row of a 5-row block == that row alone, bit for bit: %ld of %u differ", diff, 5 * H);
        check("rows plans", diff == 0, msg);
    }
    {   // added: split vs fused grouping (and on NVFP4 experts)
        kv::MoeBf16Options sp; sp.group = cere::vkk7::GroupPath::kSplit;
        const long d1 = diff_rows(run(ctx, M, M.Lb, dx.addr, 40, 0, dout, ws, &sp), 0, run(ctx, M, M.Lb, dx.addr, 40, 0, dout, ws), 0, 40, H);
        const long d2 = diff_rows(run(ctx, M, M.Lq, dx.addr, 9, 2, dout, ws, &sp), 2, run(ctx, M, M.Lq, dx.addr, 9, 2, dout, ws), 2, 7, H);
        char msg[160]; std::snprintf(msg, sizeof msg, "m 40 split vs fused grouping %ld, NVFP4 m 9 first 2 split vs fused %ld differ", d1, d2);
        check("forced grouping", d1 == 0 && d2 == 0, msg);
    }
    {   // added: refusals
        kv::MoeBf16Layer mixed = M.Lq; mixed.gate_up = M.Lb.gate_up;
        kv::MoeBf16Layer half = M.Lq; half.down_global = 0;
        ctx.begin();
        const cere_status r1 = kv::moe_bf16_block(ctx, M.d, mixed, dx.addr, 3, 0, dout.addr, ws.addr, ws.size);
        const cere_status r2 = kv::moe_bf16_block(ctx, M.d, half, dx.addr, 3, 0, dout.addr, ws.addr, ws.size);
        const cere_status r3 = kv::moe_bf16_block(ctx, M.d, M.Lb, dx.addr, 3, 3, dout.addr, ws.addr, ws.size);
        const cere_status r4 = kv::moe_bf16_block(ctx, M.d, M.Lb, dx.addr, 65, 0, dout.addr, ws.addr, ws.size);
        kv::MoeBf16Options c5; c5.cls = 5;
        const cere_status r5 = kv::moe_bf16_block(ctx, M.d, M.Lb, dx.addr, 9, 0, dout.addr, ws.addr, ws.size, &c5);
        ctx.submit_wait();
        char msg[220]; std::snprintf(msg, sizeof msg, "BF16 + NVFP4 mixed %d, NVFP4 without down_global %d, first == m %d, short workspace %d, cls 5 < m 9 %d (ARGUMENT %d)", int(r1), int(r2), int(r3), int(r4), int(r5), int(CERE_ERR_ARGUMENT));
        check("refusals", r1 == CERE_ERR_ARGUMENT && r2 == CERE_ERR_ARGUMENT && r3 == CERE_ERR_ARGUMENT && r4 == CERE_ERR_ARGUMENT && r5 == CERE_ERR_ARGUMENT, msg);
    }
    {   // added: prepared K2 weight info == the weights scanned in the call (m 40, class 40: the K2 GEMM; m 9 first 3)
        Buffer info = kt::make(ctx, kv::moe_bf16_weight_info_bytes(M.d));
        kv::MoeBf16Layer Lp = M.Lb; Lp.bf16_info = info.addr;
        ctx.begin(); kt::must(kv::moe_bf16_prepare_weights(ctx, M.d, Lp), "moe_bf16_prepare_weights"); ctx.barrier(); ctx.submit_wait();
        const long d1 = diff_rows(run(ctx, M, Lp, dx.addr, 40, 0, dout, ws), 0, run(ctx, M, M.Lb, dx.addr, 40, 0, dout, ws), 0, 40, H);
        kv::MoeBf16Layer Lpq = M.Lq; Lpq.bf16_info = info.addr;
        const long d2 = diff_rows(run(ctx, M, Lpq, dx.addr, 40, 3, dout, ws), 3, run(ctx, M, M.Lq, dx.addr, 40, 3, dout, ws), 3, 37, H);
        char msg[160]; std::snprintf(msg, sizeof msg, "m 40 BF16: %ld, m 40 first 3 NVFP4: %ld outputs differ from the in-call scan", d1, d2);
        check("prepared weight info", d1 == 0 && d2 == 0, msg);
        ctx.free(info);
    }
    M.release(ctx); ctx.free(ws); ctx.free(dout); ctx.free(dx);
    {   // added: the HIP rows 1-2 at k >= 512 (H 512, I 512: the 16-elements-per-lane GEMV order the model's k use)
        Gen G3(48);
        Model W;
        make_model(W, G3, kv::MoeDims{512, 16, 4, 512, 128});
        W.make_bf16();
        W.upload(ctx, true);
        Buffer ws3 = kt::make(ctx, kv::moe_bf16_workspace_bytes(W.d, 40)), out3 = kt::make(ctx, size_t(40) * 512 * 2);
        const auto x3 = G3.rnd(size_t(40) * 512, 1.f);
        Buffer dx3 = kt::upload(ctx, x3);
        long diff = 0; double err = 0; bool rep = true;
        for (uint32_t m : {1u, 3u, 40u}) {
            const auto ob = run(ctx, W, W.Lb, dx3.addr, m, 0, out3, ws3);
            diff += diff_rows(run(ctx, W, W.Lq, dx3.addr, m, 0, out3, ws3), 0, ob, 0, m, 512);
            rep &= run(ctx, W, W.Lb, dx3.addr, m, 0, out3, ws3) == ob;
            if (m == 40) err = rel_l2(W, x3, ob, {0, 7, 20, 39});
        }
        char msg[200]; std::snprintf(msg, sizeof msg, "H 512, I 512, m 1 / 3 / 40: NVFP4 vs BF16 %ld differ; rel-L2 %.2e (4e-3); repeat %s", diff, err, rep ? "bit-equal" : "DIFFERS");
        check("k >= 512 (LE 16)", diff == 0 && err <= 4e-3 && rep, msg);
        W.release(ctx); ctx.free(ws3); ctx.free(out3); ctx.free(dx3);
    }

    if (model || timing) {
        // ---- the model's shape (the MTP layer: mtp.cu kMoeDims = the target's), NVFP4 experts (the S3 artifact)
        const uint32_t MH = 2560, ME = 512, MK = 10, MI = 640, MS = 640, POOL = 4096;
        const auto t0 = std::chrono::steady_clock::now();
        Gen G2(9);
        Model R;
        make_model(R, G2, kv::MoeDims{MH, ME, MK, MI, MS});
        const auto xp = G2.rnd(size_t(POOL) * MH, 1.f);
        if (bf16_timing) R.make_bf16();
        R.upload(ctx, bf16_timing);
        Buffer rinfo = kt::make(ctx, kv::moe_bf16_weight_info_bytes(R.d));  // K2 weight info, prepared once (at load)
        R.Lq.bf16_info = rinfo.addr; R.Lb.bf16_info = rinfo.addr;
        ctx.begin(); kt::must(kv::moe_bf16_prepare_weights(ctx, R.d, R.Lq), "moe_bf16_prepare_weights"); ctx.barrier(); ctx.submit_wait();
        Buffer px = kt::upload(ctx, xp);
        Buffer rws = kt::make(ctx, kv::moe_bf16_workspace_bytes(R.d, POOL)), rout = kt::make(ctx, size_t(POOL) * MH * 2);
        auto csum = [](const auto &v, uint64_t c) { for (auto e : v) c = c * 1000003u + uint64_t(e); return c; };
        std::printf("model shape: data checksum %016llx, set up in %.1f s\n", (unsigned long long)csum(xp, csum(R.dsc, csum(R.guq, csum(R.router, 1)))),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        if (model) {
            const auto ext = run(ctx, R, R.Lq, px.addr, 5, 4, rout, rws);  // the draft extend (last_only): row 4 only
            const auto one = run(ctx, R, R.Lq, px.addr + uint64_t(4) * MH * 2, 1, 0, rout, rws);
            const long d1 = diff_rows(ext, 4, one, 0, 1, MH);
            const auto b64 = run(ctx, R, R.Lq, px.addr, 64, 0, rout, rws);
            long d2 = 0;
            kv::MoeBf16Options c64; c64.cls = 64;  // the 64-row call's class (HIP k7_test.cpp:157's cls pin)
            for (uint32_t r : {0u, 33u, 63u}) d2 += diff_rows(run(ctx, R, R.Lq, px.addr + uint64_t(r) * MH * 2, 1, 0, rout, rws, &c64), 0, b64, r, 1, MH);
            const double err = rel_l2(R, xp, b64, {0, 33, 63});
            char msg[220]; std::snprintf(msg, sizeof msg, "NVFP4: m 5 first 4 row 4 vs m 1 %ld differ; m 64 rows 0, 33, 63 vs alone (cls 64) %ld differ; rel-L2 %.2e (4e-3)", d1, d2, err);
            check("model rows", d1 == 0 && d2 == 0 && err <= 4e-3, msg);
        }
        if (timing) {
            // GPU time of one call: 3 untimed, then `reps` calls each between its own pair of timestamps, a barrier after
            // each; call i on rows starting at (i * 61 * m) % (POOL - m) (other experts per call), as hip_k13_timing.
            struct Shape { uint32_t m, first; const char *what; };
            const Shape shapes[] = {{1, 0, "draft step"}, {2, 1, "extend last_only"}, {5, 4, "extend last_only"},
                                    {5, 0, "extend staged"}, {16, 0, "16 rows"}, {4096, 4095, "prompt extend last_only"}};
            std::printf("timing (model shape, GPU timestamps, rot):\n");
            for (int fmt = 0; fmt <= (bf16_timing ? 1 : 0); ++fmt) {
                const kv::MoeBf16Layer &L = fmt ? R.Lb : R.Lq;
                for (const auto &sh : shapes) {
                    const uint32_t m = sh.m; const int reps = m >= 1024 ? 7 : 15;
                    auto xa = [&](int i) { return px.addr + uint64_t(m < POOL ? (uint64_t(i) * 61 * m) % (POOL - m) : 0) * MH * 2; };
                    ctx.begin();
                    int c = 0;
                    for (int i = 0; i < 3; ++i, ++c) { kt::must(kv::moe_bf16_block(ctx, R.d, L, xa(c), m, sh.first, rout.addr, rws.addr, rws.size), "moe_bf16_block"); ctx.barrier(); }
                    for (int i = 0; i < reps; ++i, ++c) {
                        ctx.stamp(2 * i);
                        kt::must(kv::moe_bf16_block(ctx, R.d, L, xa(c), m, sh.first, rout.addr, rws.addr, rws.size), "moe_bf16_block");
                        ctx.stamp(2 * i + 1);
                        ctx.barrier();
                    }
                    ctx.submit_wait();
                    std::vector<double> t; for (int i = 0; i < reps; ++i) t.push_back(ctx.stamp_ms(2 * i, 2 * i + 1));
                    std::sort(t.begin(), t.end());
                    std::printf("  vulkan %s m %4u first %4u (%s): median %.3f ms (min %.3f, max %.3f, n %d)\n", fmt ? "bf16 " : "nvfp4", m, sh.first, sh.what,
                                t[t.size() / 2], t.front(), t.back(), reps);
                    std::fflush(stdout);
                }
            }
            // phase breakdown at m 1 (NVFP4), median of 7
            std::vector<std::vector<double>> ph(kv::kProfileStamps - 1);
            ctx.begin();
            for (int i = 0; i < 8; ++i) {
                kv::MoeBf16Options po; po.profile_stamp0 = int32_t(i * kv::kProfileStamps);
                kt::must(kv::moe_bf16_block(ctx, R.d, R.Lq, px.addr + uint64_t(i * 61 % (POOL - 1)) * MH * 2, 1, 0, rout.addr, rws.addr, rws.size, &po), "moe_bf16_block");
                ctx.barrier();
            }
            ctx.submit_wait();
            for (int i = 1; i < 8; ++i) for (uint32_t p = 0; p + 1 < kv::kProfileStamps; ++p) ph[p].push_back(ctx.stamp_ms(i * kv::kProfileStamps + p, i * kv::kProfileStamps + p + 1));
            const char *names[] = {"router+shared g/u+gates", "route+group+silu", "split grouping", "GEMM1+shared down", "GEMM2", "combine"};
            std::printf("  phases nvfp4 m 1:");
            for (uint32_t p = 0; p + 1 < kv::kProfileStamps; ++p) { std::sort(ph[p].begin(), ph[p].end()); std::printf(" %s %.3f |", names[p], ph[p][3]); }
            std::printf("\n");
        }
        R.release(ctx); ctx.free(px); ctx.free(rws); ctx.free(rout); ctx.free(rinfo);
    }
    std::printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
