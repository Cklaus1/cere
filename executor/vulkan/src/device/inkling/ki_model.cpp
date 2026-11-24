// Inkling S11: ki_model.h.
#include "device/inkling/ki_model.h"

#include <cstdio>
#include <cstring>
#include <utility>

#include "kernels/k2_gemm_vk.h"
#include "ref/inkling/attn_ref.h"

namespace cere::vkink {
namespace {

namespace ra = ref::attn;
constexpr uint32_t H = kHidden;
constexpr float kEps = 1e-6f, kRouteScale = 8.0f;
constexpr uint64_t kTail = 256;  // slack after every buffer (the kernels' vector loads may read past a row end)

// Every text tensor the forward binds with its exact size in bytes (the checkpoint's dtype x shape).
std::vector<std::pair<std::string, uint64_t>> text_tensors() {
    std::vector<std::pair<std::string, uint64_t>> t;
    const uint64_t bf = 2, f4 = 4;
    auto add = [&](const std::string &n, uint64_t bytes) { t.emplace_back("model.llm." + n, bytes); };
    add("embed.weight", uint64_t(kVocabRows) * H * bf);
    add("embed_norm.weight", H * bf);
    add("norm.weight", H * bf);
    add("unembed.weight", uint64_t(kVocabRows) * H * bf);
    add("attn_res.output_query", H * f4);
    add("middle_cycle_cross_pass_queries", 24 * H * f4);
    add("middle_cycle_cross_pass_kv_queries", 24 * H * f4);
    for (uint32_t l = 0; l < kLayers; ++l) {
        const bool full = l % 6 == 5, mx = l >= 20;
        const std::string p = "layers." + std::to_string(l) + ".";
        if (l > 0) add("attn_res.attn_queries." + std::to_string(l - 1), (full ? 3 : 1) * H * f4);
        add("attn_res.mlp_queries." + std::to_string(l), H * f4);
        if (full) {
            for (const char *n : {"q_input_norm", "k_input_norm", "v_input_norm"}) add(p + n + ".weight", H * bf);
            for (const char *n : {"k_sconv", "v_sconv"}) {
                add(p + "attn." + n + ".static_w", 4 * kKvRow * bf);
                add(p + "attn." + n + ".weight_proj", 32 * H * bf);
            }
        } else {
            add(p + "attn_norm.weight", H * bf);
            add(p + "attn.k_sconv.weight", kKvRow * 4 * bf);
            add(p + "attn.v_sconv.weight", kKvRow * 4 * bf);
        }
        add(p + "mlp_norm.weight", H * bf);
        add(p + "attn.wq_du.weight", uint64_t(kQRow) * H * bf);
        add(p + "attn.wk_dv.weight", uint64_t(kKvRow) * H * bf);
        add(p + "attn.wv_dv.weight", uint64_t(kKvRow) * H * bf);
        add(p + "attn.wr_du.weight", uint64_t(kRelRow) * H * bf);
        add(p + "attn.wo_ud.weight", uint64_t(H) * kQRow * bf);
        add(p + "attn.gate_attn.weight", uint64_t(kQRow) * H * bf);
        add(p + "attn.attn_sink_proj.weight", kHeads * H * bf);
        add(p + "attn.q_l2_clip.scale", kHeads * 128 * bf);
        add(p + "attn.rel_l2_clip.scale", kHeads * 16 * bf);
        add(p + "attn.rel_logits_proj.proj", 16 * kKvRow * bf);
        add(p + "attn_sconv.weight", H * 4 * bf);
        add(p + "mlp_sconv.weight", H * 4 * bf);
        add(p + "mlp.gate.weight", kExperts * H * bf);
        add(p + "mlp.gate.bias", kExperts * f4);
        const std::string e = p + "mlp.experts.";
        const uint64_t EH = uint64_t(kExperts) * H;
        if (mx) {  // MXFP8: E4M3 weights, one UE8M0 scale per 32
            add(e + "w13_weight", EH * 2 * kInter);
            add(e + "w13_weight_scale", EH * 2 * kInter / 32);
            add(e + "w2_weight", EH * kInter);
            add(e + "w2_weight_scale", EH * kInter / 32);
        } else {  // NVFP4: two E2M1 codes per byte, one E4M3 scale per 16, per-expert amax, outer denominator
            add(e + "w13_weight", EH * kInter);
            add(e + "w13_weight_scale", EH * 2 * kInter / 16);
            add(e + "w2_weight", EH * kInter / 2);
            add(e + "w2_weight_scale", EH * kInter / 16);
            add(e + "w13_weight_amax", kExperts * f4);
            add(e + "w2_weight_amax", kExperts * f4);
            add(e + "w13_weight_outer_scale_denominator", f4);
            add(e + "w2_weight_outer_scale_denominator", f4);
        }
        if (l >= kLoop0 && l < kLoop1) {
            for (const char *g : {"attn", "k", "v", "mlp"}) {
                const std::string c = p + "controllers." + g + ".";
                add(c + "down_weight", 64 * 2 * H * bf);
                add(c + "forget_weight", H * 64 * bf);
                add(c + "gain_weight", H * 64 * bf);
            }
        }
    }
    return t;
}

}  // namespace

std::vector<std::string> text_tensor_names() {
    std::vector<std::string> n;
    for (const auto &e : text_tensors()) n.push_back(e.first);
    return n;
}

Session::~Session() {
    if (ctx_ != nullptr)
        for (vk::Buffer &b : owned_) ctx_->free(b);
}

Forward::~Forward() {
    for (vk::Buffer &b : owned_) ctx_.free(b);
}

uint64_t Forward::session_bytes(uint64_t max_ctx) {
    uint64_t n = 0;
    for (uint32_t v = 0; v < kVisits; ++v) {
        const uint64_t rows = phys(v) % 6 == 5 ? max_ctx : uint64_t(ra::kRingSlots);
        n += 2 * (rows * kKvRow * 2 + kTail);
        n += 2 * 2 * (3 * kKvRow * 4 + kTail) + 2 * 2 * (3 * H * 4 + kTail);
    }
    return n;
}

// Split attention's span partials: a row block (rows <= kAttnKeyRows / keys, or 1) holds <= kAttnKeyRows / 256 + M
// (row, span) pairs, a single decode row <= kMaxContext / 256 + 1.
static uint64_t attn_part_bytes(uint64_t M) {
    return std::max<uint64_t>(kMaxContext / 256 + 1, kAttnKeyRows / 256 + M) * kHeads * vki::kAttnPartBytes;
}

std::vector<std::pair<uint64_t Forward::Scratch::*, uint64_t>> Forward::scratch_plan(uint64_t M, uint64_t LR) {
    using S = Scratch;
    return {
        {&S::ids, M * 4}, {&S::emb, M * H * 2}, {&S::tau, M * 4}, {&S::a, 3 * M * H * 4}, {&S::msgs, 4 * M * H * 4},
        {&S::t3, M * H * 4}, {&S::md, M * 64 * 4}, {&S::hq, M * H * 2}, {&S::hk, M * H * 2}, {&S::hv, M * H * 2}, {&S::q, M * kQRow * 2},
        {&S::r, M * kRelRow * 2}, {&S::sink, M * kHeads * 2}, {&S::gate, M * kQRow * 2}, {&S::kraw, M * kKvRow * 2},
        {&S::vraw, M * kKvRow * 2}, {&S::kd, M * 32 * 2}, {&S::vd, M * 32 * 2}, {&S::kcv, M * kKvRow * 2},
        {&S::vcv, M * kKvRow * 2}, {&S::k, M * kKvRow * 2}, {&S::v, M * kKvRow * 2}, {&S::ao, M * kQRow * 2},
        {&S::apart, attn_part_bytes(M)},
        {&S::o, M * H * 2}, {&S::co, M * H * 2}, {&S::res, M * H * 4}, {&S::mm, M * H * 4}, {&S::hn, M * H * 2},
        {&S::rids, M * 8 * 4}, {&S::rwts, M * 8 * 4}, {&S::rlg, M * kExperts * 4}, {&S::xq, M * H}, {&S::xs, M * H / 16},
        {&S::xa, M * 4}, {&S::rtab, vki::moe_tab_bytes(M * 8)},
        {&S::gu, M * 8 * kInter * 2}, {&S::vq, M * 8 * kInter}, {&S::vs, M * 8 * kInter / 16}, {&S::va, M * 8 * 4},
        {&S::y, M * 8 * H * 2}, {&S::dm, M * H * 2}, {&S::dc, M * H * 2}, {&S::fin, M * H * 4}, {&S::fn, M * H * 2},
        {&S::logits_bf, LR * uint64_t(kVocab) * 2}, {&S::logits, LR * uint64_t(kVocab) * 4},
    };
}

uint64_t Forward::scratch_bytes(const Options &opt) {
    const uint64_t M = opt.max_rows;
    uint64_t n = uint64_t(12) * 4 * H * 4 + kTail;  // merged message queries
    for (const auto &e : scratch_plan(M, opt.logit_rows)) n += e.second + kTail;
    n += kVisits * (M * H * 2 + kTail);  // visit outputs
    if (opt.k2) {  // workspace, and one row-info buffer per dense weight (<= 10 per layer and the unembed)
        n += vkk2::gemm_workspace_bytes(uint32_t(M), kVocab) + kTail;
        n += (uint64_t(kLayers) * 10) * ((uint64_t(kQRow) * 4 + 15) / 16 * 16 + kTail) + uint64_t(kVocab) * 4 + kTail;
    }
    return n;
}

cere_status Forward::alloc(uint64_t bytes, uint64_t *addr, std::vector<vk::Buffer> *owner) {
    vk::Buffer b;
    if (const cere_status st = ctx_.allocate(bytes + kTail, b, false); st != CERE_OK) return st;
    if (b.map == nullptr) {
        ctx_.free(b);
        return CERE_ERR_UNSUPPORTED;  // the forward writes inputs and reads logits through the mapping (UMA only)
    }
    std::memset(b.map, 0, bytes + kTail);
    owner->push_back(b);
    *addr = b.addr;
    return CERE_OK;
}

void *Forward::host(uint64_t addr) const {
    for (const vk::Buffer &b : owned_)
        if (b.addr == addr) return b.map;
    return nullptr;
}

cere_status Forward::fail(cere_status st) {
    if (open_) {
        ctx_.submit_wait();  // returns the latched error without running a half-recorded command buffer
        open_ = false;
    }
    (void)ctx_.wait_all();  // nothing stays in flight past an error
    return st;
}

// opt.prof without prof_split: a timestamp after every barrier (stamp 0 at begin), so each barrier site gets the GPU
// time of the ops recorded since the previous stamp, inside unsplit command buffers. With every command-buffer slot
// in flight, the oldest submit retires first.
cere_status Forward::begin() {
    if (ctx_.in_flight() >= ctx_.max_in_flight())
        if (const cere_status st = ctx_.wait(ctx_.last_retired() + 1); st != CERE_OK) return st;
    const cere_status st = ctx_.begin();
    if (st == CERE_OK) open_ = true;
    stamp_lines_.clear();
    if (st == CERE_OK && opt_.prof && !opt_.prof_split) {
        stamp_lines_.push_back(0);
        return ctx_.stamp(0);
    }
    return st;
}

cere_status Forward::drain() { return ctx_.wait_all(); }

// Without prof / trace the submit does not wait (begin and drain retire it); prof reads each submit's GPU times, so
// it waits.
cere_status Forward::submit(int line) {
    open_ = false;
    if (!opt_.prof && !opt_.trace) return ctx_.submit();
    const cere_status st = ctx_.submit_wait();
    if (st == CERE_OK && opt_.prof && opt_.prof_split) {
        auto &e = prof_[line];
        e.first += ctx_.last_submit_gpu_ms();
        e.second += 1;
    }
    if (st == CERE_OK && opt_.prof && !opt_.prof_split)
        for (uint32_t i = 1; i < stamp_lines_.size(); ++i) {
            auto &e = prof_[stamp_lines_[i]];
            e.first += ctx_.stamp_ms(i - 1, i);
            e.second += 1;
        }
    return st;
}

// After visit v (which ends on a barrier): close the command buffer, except inside a group of visits_per_submit visits
// of a short step (<= kOneSubmitRows rows ending at <= kOneSubmitCtx; long steps close every visit, ring timeout).
cere_status Forward::visit_cut(uint32_t M, uint32_t pos0, uint32_t v) {
    if (M <= kOneSubmitRows && uint64_t(pos0) + M <= kOneSubmitCtx && (v + 1) % opt_.visits_per_submit != 0)
        return CERE_OK;
    if (const cere_status st = submit(); st != CERE_OK) return st;
    return begin();
}

cere_status Forward::bar(int line) {
    if (opt_.trace || (opt_.prof && opt_.prof_split)) {
        if (const cere_status st = submit(line); st != CERE_OK) return st;
        return begin();
    }
    if (const cere_status st = ctx_.barrier(); st != CERE_OK) return st;
    if (opt_.prof && stamp_lines_.size() < vk::Context::kStamps) {
        stamp_lines_.push_back(line);
        return ctx_.stamp(uint32_t(stamp_lines_.size() - 1));
    }
    return CERE_OK;
}

#define KI_TRY(x)                                                                                                    \
    do {                                                                                                             \
        if (const cere_status st_ = (x); st_ != CERE_OK) {                                                           \
            if (opt_.trace) std::fprintf(stderr, "ki_model: %s failed (%d): %s\n", #x, int(st_), ctx_.last_error().c_str()); \
            return fail(st_);                                                                                        \
        }                                                                                                            \
        if (opt_.trace) std::fprintf(stderr, "op %s\n", #x);                                                         \
    } while (0)

cere_status Forward::bind(const Find &find, std::string *why) {
    std::vector<std::pair<std::string, uint64_t>> want = text_tensors();
    std::vector<TensorView> got(want.size());
    for (size_t i = 0; i < want.size(); ++i) {
        if (!find(want[i].first, &got[i]) || got[i].bytes != want[i].second || got[i].map == nullptr) {
            if (why != nullptr) *why = want[i].first;
            return CERE_ERR_ARGUMENT;
        }
        weight_bytes_ += got[i].bytes;
    }
    size_t i = 0;
    auto next = [&]() -> const TensorView & { return got[i++]; };  // in text_tensors() order
    auto f32 = [](const TensorView &t) {
        float f;
        std::memcpy(&f, t.map, 4);
        return f;
    };
    embed_ = next().addr;
    embed_norm_ = next().addr;
    norm_ = next().addr;
    unembed_ = next().addr;
    out_q_ = next().addr;
    const TensorView &qm = next(), &kv = next();
    {  // merged message queries (core_model_loading.py Inkling2LoopMergeMessageQueries): [j] = q, k, v, mlp
        if (const cere_status st = alloc(uint64_t(12) * 4 * H * 4, &cross_q_, &owned_); st != CERE_OK) return st;
        float *o = static_cast<float *>(host(cross_q_));
        const float *q = static_cast<const float *>(qm.map), *k = static_cast<const float *>(kv.map);
        for (uint32_t j = 0; j < 12; ++j) {
            std::memcpy(o + (j * 4 + 0) * H, q + (2 * j) * H, H * 4);
            std::memcpy(o + (j * 4 + 1) * H, k + (2 * j) * H, H * 4);
            std::memcpy(o + (j * 4 + 2) * H, k + (2 * j + 1) * H, H * 4);
            std::memcpy(o + (j * 4 + 3) * H, q + (2 * j + 1) * H, H * 4);
        }
    }
    for (uint32_t l = 0; l < kLayers; ++l) {
        Layer &y = L_[l];
        y.full = l % 6 == 5;
        y.fmt = l >= 20 ? 1 : 0;
        if (l > 0) y.attn_q = next().addr;
        y.mlp_q = next().addr;
        if (y.full) {
            y.norm_q = next().addr;
            y.norm_k = next().addr;
            y.norm_v = next().addr;
            y.k_conv = next().addr;
            y.k_dynp = next().addr;
            y.v_conv = next().addr;
            y.v_dynp = next().addr;
        } else {
            y.norm_q = y.norm_k = y.norm_v = next().addr;
            y.k_conv = next().addr;
            y.v_conv = next().addr;
        }
        y.mlp_norm = next().addr;
        y.wq = next().addr;
        y.wk = next().addr;
        y.wv = next().addr;
        y.wr = next().addr;
        y.wo = next().addr;
        y.wg = next().addr;
        y.ws = next().addr;
        y.q_scale = next().addr;
        y.rel_scale = next().addr;
        y.rel_proj = next().addr;
        y.attn_conv = next().addr;
        y.mlp_conv = next().addr;
        y.router_w = next().addr;
        y.router_b = next().addr;
        y.w13.w = next().addr;
        y.w13.ws = next().addr;
        y.w2.w = next().addr;
        y.w2.ws = next().addr;
        if (y.fmt == 0) {
            y.w13.wamax = next().addr;
            y.w2.wamax = next().addr;
            y.w13.den = f32(next());
            y.w2.den = f32(next());
        }
        if (l >= kLoop0 && l < kLoop1)
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 3; ++b) y.ctrl[a][b] = next().addr;
    }
    return CERE_OK;
}

cere_status Forward::init(const Find &find, const Options &opt, std::string *why) {
    if (ready_ || opt.max_rows == 0 || opt.max_rows > kMaxRows || opt.logit_rows == 0 ||
        opt.logit_rows > opt.max_rows) {
        return CERE_ERR_ARGUMENT;
    }
    opt_ = opt;
    if (const cere_status st = bind(find, why); st != CERE_OK) return st;
    for (const auto &e : scratch_plan(opt.max_rows, opt.logit_rows))
        if (const cere_status st = alloc(e.second, &(s_.*e.first), &owned_); st != CERE_OK) return st;
    const uint64_t M = opt.max_rows;
    for (uint32_t v = 0; v < kVisits; ++v)
        if (const cere_status st = alloc(M * H * 2, &out_[v], &owned_); st != CERE_OK) return st;
    std::vector<vk::PipelineSpec> specs = vki::pipeline_specs(vki::moe_cm_device(ctx_));
    if (opt.k2) {
        if (const cere_status st = alloc(vkk2::gemm_workspace_bytes(uint32_t(M), kVocab), &k2_ws_, &owned_);
            st != CERE_OK) {
            return st;
        }
        const std::vector<vk::PipelineSpec> k2 = vkk2::pipeline_specs(true, false, {kHidden, kQRow});
        specs.insert(specs.end(), k2.begin(), k2.end());
    }
    std::string w;
    if (const cere_status st = ctx_.warm(specs, &w); st != CERE_OK) {
        if (why != nullptr) *why = "pipelines: " + w;
        return st;
    }
    if (opt.k2 && M >= vkk2::kGemmMinM) {  // the GEMM's accumulator layout check (once per context)
        if (!vkk2::gemm_ready(ctx_) && vkk2::gemm_probe(ctx_) != CERE_OK) {
            if (why != nullptr) *why = "k2: no cooperative-matrix GEMM on this device";
            return CERE_ERR_UNSUPPORTED;
        }
    }
    // The NVFP4 prefill MoE's int8 cooperative-matrix route (same bits as the vector route) needs the layout check.
    if (vki::moe_cm_device(ctx_) && M >= kGroupedMoeTokens) {
        if (const cere_status st = vki::moe_probe(ctx_); st != CERE_OK && st != CERE_ERR_UNSUPPORTED) return st;
    }
    ready_ = true;
    return CERE_OK;
}

cere_status Forward::create_session(uint32_t max_ctx, std::unique_ptr<Session> *out) {
    if (!ready_ || max_ctx == 0 || max_ctx > kMaxContext) return CERE_ERR_ARGUMENT;
    std::unique_ptr<Session> s(new Session());
    s->ctx_ = &ctx_;
    s->max_ctx_ = max_ctx;
    for (uint32_t v = 0; v < kVisits; ++v) {
        Session::Visit &vs = s->v_[v];
        const uint64_t rows = L_[phys(v)].full ? max_ctx : uint64_t(ra::kRingSlots);
        if (const cere_status st = alloc(rows * kKvRow * 2, &vs.kc, &s->owned_); st != CERE_OK) return st;
        if (const cere_status st = alloc(rows * kKvRow * 2, &vs.vc, &s->owned_); st != CERE_OK) return st;
        const uint32_t C[4] = {kKvRow, kKvRow, H, H};
        for (int c = 0; c < 4; ++c)
            for (int k = 0; k < 2; ++k)
                if (const cere_status st = alloc(uint64_t(3) * C[c] * 4, &vs.st[c][k], &s->owned_); st != CERE_OK)
                    return st;
    }
    s->bytes_ = session_bytes(max_ctx);
    *out = std::move(s);
    return CERE_OK;
}

cere_status Forward::dense(uint64_t x, uint64_t w, uint64_t y, uint32_t M, uint32_t n, uint32_t k) {
    if (!opt_.k2) return vki::dense(ctx_, x, w, y, M, n, k, false);
    uint64_t info = 0;
    if (M >= vkk2::kGemmMinM) {
        for (const auto &e : winfo_)
            if (e.first == w) info = e.second;
        if (info == 0) {
            if (const cere_status st = alloc((uint64_t(n) * 4 + 15) / 16 * 16, &info, &owned_); st != CERE_OK) return st;
            if (const cere_status st = vkk2::gemm_prepare_weight(ctx_, w, 0, n, k, info); st != CERE_OK) return st;
            winfo_.emplace_back(w, info);
        }
        // GEMMs share one workspace (and the row info was just written): each waits for the previous one.
        if (const cere_status st = bar(); st != CERE_OK) return st;
    }
    return vkk2::run(ctx_, x, w, 0, y, M, n, k, false, 0, k2_ws_, info);
}

cere_status Forward::conv(Session &s, uint32_t v, int c, uint64_t x, uint64_t w, uint64_t dyn, uint64_t out,
                          uint32_t M, uint32_t C, bool dynamic) {
    Session::Visit &vs = s.v_[v];
    const uint64_t in = vs.st[c][vs.cur[c]], nx = vs.st[c][vs.cur[c] ^ 1];
    const vki::Tensor tx{x, vki::Dt::kBF16}, tw{w, vki::Dt::kBF16}, to{out, vki::Dt::kBF16};
    const cere_status st = dynamic ? vki::conv_dynamic(ctx_, tx, tw, {dyn, vki::Dt::kBF16}, to, in, nx, M, C, kKvHeads)
                                   : vki::conv_static(ctx_, tx, tw, to, in, nx, M, C);
    if (st == CERE_OK) vs.cur[c] ^= 1;
    return st;
}

cere_status Forward::rms(vki::Tensor x, uint64_t w, uint64_t out, uint32_t M) {
    return vki::rms_norm(ctx_, x, {w, vki::Dt::kBF16}, {out, vki::Dt::kBF16}, H, M, kEps);
}

cere_status Forward::layer(Session &ss, uint32_t v, const std::vector<uint64_t> &B, uint64_t x, uint32_t M,
                           uint32_t pos0, const uint64_t *prev) {
    const uint32_t l = phys(v);
    const Layer &y = L_[l];
    Session::Visit &vs = ss.v_[v];
    Scratch &s = s_;
    const bool boundary = l % 6 == 0, msgs = prev != nullptr && !opt_.no_messages;
    const uint64_t MH4 = uint64_t(M) * H * 4;
    auto states = [&](const std::vector<uint64_t> &list, uint64_t last, vki::Dt last_dt) {
        std::vector<vki::Tensor> t;
        for (uint64_t b : list) t.push_back({b, vki::Dt::kBF16});
        t.push_back({last, last_dt});
        return t;
    };
    if (msgs) {  // messages of loop layer j: depth_mix([prev x_in(12), prev out]; 4 queries) -> [4][M][H]
        const vki::Tensor st[2] = {{prev[0], vki::Dt::kBF16}, {prev[1], vki::Dt::kBF16}};
        KI_TRY(vki::depth_mix(ctx_, st, 2, cross_q_ + uint64_t(l - kLoop0) * 4 * H * 4, 4, s.msgs, M, kEps));
    }
    // attention inputs h_q, h_k, h_v (BF16)
    if (l == 0) {
        KI_TRY(rms({x, vki::Dt::kBF16}, y.norm_q, s.hq, M));
        KI_TRY(bar());
    } else {
        const std::vector<vki::Tensor> st = states(B, x, vki::Dt::kBF16);
        const uint32_t nq = y.full ? 3 : 1;
        KI_TRY(vki::depth_mix(ctx_, st.data(), uint32_t(st.size()), y.attn_q, nq, s.a, M, kEps));
        KI_TRY(bar());
        const uint64_t hs[3] = {s.hq, s.hk, s.hv}, nw[3] = {y.norm_q, y.norm_k, y.norm_v};
        const uint32_t ng = (y.full || msgs) ? 3 : 1;
        for (uint32_t g = 0; g < ng; ++g) {
            uint64_t a = s.a + (y.full ? g * MH4 : 0);
            if (msgs) {
                KI_TRY(vki::msg_gate(ctx_, a, s.msgs + g * MH4, y.ctrl[g][0], y.ctrl[g][1], y.ctrl[g][2], s.md, s.t3, M, kEps,
                                     opt_.msg_route));
                KI_TRY(bar());
                a = s.t3;
            }
            KI_TRY(rms({a, vki::Dt::kF32}, nw[g], hs[g], M));
            KI_TRY(bar());
        }
    }
    const bool shared = l == 0 || (!y.full && !msgs);
    const uint64_t hq = s.hq, hk = shared ? s.hq : s.hk, hv = shared ? s.hq : s.hv;
    // projections
    KI_TRY(dense(hq, y.wq, s.q, M, kQRow, H));
    KI_TRY(dense(hq, y.wr, s.r, M, kRelRow, H));
    KI_TRY(dense(hq, y.ws, s.sink, M, kHeads, H));
    KI_TRY(dense(hq, y.wg, s.gate, M, kQRow, H));
    KI_TRY(dense(hk, y.wk, s.kraw, M, kKvRow, H));
    KI_TRY(dense(hv, y.wv, s.vraw, M, kKvRow, H));
    if (y.full) {
        KI_TRY(dense(hk, y.k_dynp, s.kd, M, 32, H));
        KI_TRY(dense(hv, y.v_dynp, s.vd, M, 32, H));
    }
    KI_TRY(bar());
    const uint64_t tau = y.full ? s.tau : 0;
    KI_TRY(vki::l2_clip(ctx_, {s.q, vki::Dt::kBF16}, {y.q_scale, vki::Dt::kBF16}, tau, {s.q, vki::Dt::kBF16}, 128,
                        M * kHeads, kHeads, ra::kQkClip));
    KI_TRY(vki::l2_clip(ctx_, {s.r, vki::Dt::kBF16}, {y.rel_scale, vki::Dt::kBF16}, tau, {s.r, vki::Dt::kBF16}, 16,
                        M * kHeads, kHeads, ra::kRelClip));
    KI_TRY(conv(ss, v, 0, s.kraw, y.k_conv, s.kd, s.kcv, M, kKvRow, y.full));
    KI_TRY(conv(ss, v, 1, s.vraw, y.v_conv, s.vd, s.vcv, M, kKvRow, y.full));
    KI_TRY(bar());
    KI_TRY(vki::l2_clip(ctx_, {s.kcv, vki::Dt::kBF16}, {}, 0, {s.k, vki::Dt::kBF16}, 128, M * kKvHeads, kKvHeads,
                        ra::kQkClip));
    KI_TRY(vki::rms_norm(ctx_, {s.vcv, vki::Dt::kBF16}, {}, {s.v, vki::Dt::kBF16}, 128, M * kKvHeads, ra::kVEps));
    KI_TRY(bar());
    vki::AttnSeq sq;
    sq.kind = y.full ? 1 : 0;
    sq.pos0 = pos0;
    sq.n = M;
    sq.k = s.k;
    sq.v = s.v;
    sq.kc = vs.kc;
    sq.vc = vs.vc;
    vki::AttnRows ar;
    ar.q = s.q;
    ar.r = s.r;
    ar.proj = y.rel_proj;
    ar.sink = s.sink;
    ar.gate = s.gate;
    ar.out = s.ao;
    ar.part = s.apart;
    ar.part_bytes = attn_part_bytes(opt_.max_rows);
    {  // row blocks of <= kAttnKeyRows (query row x key) pairs, each block its own submit (AMD's ring timeout)
        const uint64_t keys = y.full ? uint64_t(pos0) + M : std::min<uint64_t>(uint64_t(pos0) + M, 1024);
        const uint32_t rb = uint32_t(std::max<uint64_t>(1, std::min<uint64_t>(M, kAttnKeyRows / keys)));
        for (uint32_t r0 = 0; r0 < M; r0 += rb) {
            if (r0 > 0) {
                KI_TRY(submit());
                KI_TRY(begin());
            }
            KI_TRY(vki::attention(ctx_, sq, ar, r0, rb, opt_.attn_route));
        }
    }
    KI_TRY(bar());
    KI_TRY(vki::kv_store(ctx_, sq));
    KI_TRY(dense(s.ao, y.wo, s.o, M, H, kQRow));
    KI_TRY(bar());
    KI_TRY(conv(ss, v, 2, s.o, y.attn_conv, 0, s.co, M, H, false));
    KI_TRY(bar());
    KI_TRY(vki::attn_residual(ctx_, x, s.co, s.res, M * H, boundary));
    KI_TRY(bar());
    std::vector<uint64_t> B2 = B;
    if (boundary) B2.push_back(x);
    {
        const std::vector<vki::Tensor> st = states(B2, s.res, vki::Dt::kF32);
        KI_TRY(vki::depth_mix(ctx_, st.data(), uint32_t(st.size()), y.mlp_q, 1, s.mm, M, kEps));
        KI_TRY(bar());
    }
    uint64_t mm = s.mm;
    if (msgs) {
        KI_TRY(vki::msg_gate(ctx_, mm, s.msgs + 3 * MH4, y.ctrl[3][0], y.ctrl[3][1], y.ctrl[3][2], s.md, s.t3, M, kEps,
                             opt_.msg_route));
        KI_TRY(bar());
        mm = s.t3;
    }
    KI_TRY(rms({mm, vki::Dt::kF32}, y.mlp_norm, s.hn, M));
    KI_TRY(bar());
    // experts
    const uint32_t pairs = M * 8;
    KI_TRY(vki::router(ctx_, s.hn, y.router_w, y.router_b, s.rlg, s.rids, s.rwts, M, kExperts, H, kRouteScale,
                       opt_.moe_mutant));
    if (y.fmt == 0)
        KI_TRY(vki::quant_nvfp4(ctx_, {s.hn, vki::Dt::kBF16}, s.xq, s.xs, s.xa, 0, H, M, opt_.moe_mutant));
    else
        KI_TRY(vki::quant_mxfp8(ctx_, {s.hn, vki::Dt::kBF16}, s.xq, s.xs, H, M));
    KI_TRY(bar());
    // Prefill: expert-grouped tiles (an expert's weights read once per 16 of its pairs); decode: per-pair tiles. Same bits.
    uint64_t tab = 0;
    if (M >= kGroupedMoeTokens) {
        tab = s.rtab;
        KI_TRY(vki::moe_bucket(ctx_, s.rids, tab, pairs, kExperts));
        KI_TRY(bar());
    }
    KI_TRY(vki::moe_gate_up(ctx_, y.fmt, {s.xq, s.xs, s.xa}, y.w13, s.rids, s.rwts, s.gu, 2 * kInter, H, pairs, tab));
    KI_TRY(bar());
    if (y.fmt == 0)
        KI_TRY(vki::quant_nvfp4(ctx_, {s.gu, vki::Dt::kBF16}, s.vq, s.vs, s.va, 0, kInter, pairs, opt_.moe_mutant));
    else
        KI_TRY(vki::quant_mxfp8(ctx_, {s.gu, vki::Dt::kBF16}, s.vq, s.vs, kInter, pairs));
    KI_TRY(bar());
    KI_TRY(vki::moe_down(ctx_, y.fmt, {s.vq, s.vs, s.va}, y.w2, s.rids, s.y, H, kInter, pairs, tab));
    KI_TRY(bar());
    KI_TRY(vki::moe_combine(ctx_, s.y, s.dm, H, M));
    KI_TRY(bar());
    KI_TRY(conv(ss, v, 3, s.dm, y.mlp_conv, 0, s.dc, M, H, false));
    KI_TRY(bar());
    KI_TRY(vki::layer_out(ctx_, s.res, s.dc, out_[v], M * H));
    KI_TRY(bar());
    return CERE_OK;
}

cere_status Forward::step(Session &ss, const uint32_t *ids, uint32_t M, uint32_t pos0) {
    if (!ready_ || ss.ctx_ != &ctx_ || M == 0 || M > opt_.max_rows || uint64_t(pos0) + M > ss.max_ctx_)
        return CERE_ERR_ARGUMENT;
    std::memcpy(host(s_.ids), ids, size_t(M) * 4);
    float *tau = static_cast<float *>(host(s_.tau));
    for (uint32_t i = 0; i < M; ++i) tau[i] = ra::tau(int64_t(pos0) + i);
    KI_TRY(begin());
    KI_TRY(vki::embed_norm(ctx_, embed_, s_.ids, embed_norm_, s_.emb, M, kEps));
    KI_TRY(bar());
    std::vector<uint64_t> B;
    uint64_t x = s_.emb;
    for (uint32_t v = 0; v < kLoop0; ++v) {
        if (const cere_status st = layer(ss, v, B, x, M, pos0, nullptr); st != CERE_OK) return st;
        KI_TRY(visit_cut(M, pos0, v));
        if (v % 6 == 0) B.push_back(x);
        x = out_[v];
    }
    const std::vector<uint64_t> preB = B;
    const uint64_t pre = x;
    uint64_t prev[2] = {0, 0};
    for (uint32_t p = 0; p < kPasses; ++p) {
        B = preB;
        x = pre;
        for (uint32_t j = 0; j < 12; ++j) {
            const uint32_t v = kLoop0 + 12 * p + j;
            if (const cere_status st = layer(ss, v, B, x, M, pos0, p == 0 ? nullptr : prev); st != CERE_OK) return st;
            KI_TRY(visit_cut(M, pos0, v));
            if (j % 6 == 0) B.push_back(x);
            x = out_[v];
        }
        prev[0] = B.back();  // x_in(12) of this pass (block_states[npre + 1:])
        prev[1] = x;
    }
    for (uint32_t v = 42; v < kVisits; ++v) {
        if (const cere_status st = layer(ss, v, B, x, M, pos0, nullptr); st != CERE_OK) return st;
        KI_TRY(visit_cut(M, pos0, v));
        if (phys(v) % 6 == 0) B.push_back(x);
        x = out_[v];
    }
    {
        std::vector<vki::Tensor> st;
        for (uint64_t b : B) st.push_back({b, vki::Dt::kBF16});
        st.push_back({x, vki::Dt::kBF16});
        KI_TRY(vki::depth_mix(ctx_, st.data(), uint32_t(st.size()), out_q_, 1, s_.fin, M, kEps));
    }
    KI_TRY(bar());
    KI_TRY(rms({s_.fin, vki::Dt::kF32}, norm_, s_.fn, M));
    KI_TRY(submit());
    KI_TRY(drain());
    return CERE_OK;
}

cere_status Forward::logits(uint32_t first, uint32_t rows) {
    if (!ready_ || rows == 0 || rows > opt_.logit_rows || first + rows > opt_.max_rows) return CERE_ERR_ARGUMENT;
    // One submit per kLogitRowsPerSubmit rows: the unembed of a 512-row chunk in one command buffer outlives amdgpu's
    // ring timeout on AMD (rows are independent, so the bits do not depend on the split).
    constexpr uint32_t kLogitRowsPerSubmit = 16;
    for (uint32_t r0 = 0; r0 < rows; r0 += kLogitRowsPerSubmit) {
        const uint32_t n = std::min(kLogitRowsPerSubmit, rows - r0);
        const uint64_t off = uint64_t(r0) * kVocab;
        KI_TRY(begin());
        KI_TRY(dense(s_.fn + uint64_t(first + r0) * H * 2, unembed_, s_.logits_bf + off * 2, n, kVocab, H));
        KI_TRY(bar());
        KI_TRY(vki::softcap(ctx_, {s_.logits_bf + off * 2, vki::Dt::kBF16}, s_.logits + off * 4, n * kVocab));
        KI_TRY(submit());
    }
    KI_TRY(drain());
    return CERE_OK;
}

}  // namespace cere::vkink
