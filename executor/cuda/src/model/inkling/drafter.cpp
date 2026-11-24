// Inkling 2 Mini S9 phase B: the block drafter on the device (drafter.h). The op order is DRAFTER.md sections 3-5;
// the arithmetic lives in kernels/inkling/drafter.h (bit-equal to executor/ref/inkling/drafter_ref.h), norm.h's
// embed_norm, linear.h and K2's GemmPlan (mlp.down_proj, markov_head.rnn_proj).
#include "model/inkling/drafter.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <map>
#include <set>

#include "inkling/drafter_ref.h"
#include "kernels/inkling/drafter.h"
#include "kernels/inkling/linear.h"
#include "kernels/inkling/norm.h"

namespace cere::inkling {

namespace {

namespace D = ref::drafter;
constexpr int64_t H = kHidden;
constexpr uint64_t kAlign = 256;
constexpr int64_t kRows = int64_t(kDraftBlock) * kMaxDraftSessions;     // 512 block rows of one draft call
constexpr int64_t kHidRows = int64_t(kMaxBlockDrafts) * kMaxDraftSessions;  // 448 drafted rows
constexpr int64_t kInter = 9216;
constexpr int64_t kRnnIn = 5120;
constexpr int64_t kDraftRingElems = int64_t(D::kWindow) * D::kKvRow;  // one K or V ring of a local layer
constexpr int64_t kCtxPlane = kMaxDraftCtxRows;

uint64_t align_up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

class Carver {
public:
    explicit Carver(uint8_t *base) : base_(base) {}
    template <typename T>
    T *take(uint64_t count) {
        T *p = base_ == nullptr ? nullptr : reinterpret_cast<T *>(base_ + used_);
        used_ += align_up(count * sizeof(T));
        return p;
    }
    uint64_t used() const { return used_; }

private:
    uint8_t *base_;
    uint64_t used_ = 0;
};

cere_status cuda_status(cudaError_t e) {
    if (e == cudaSuccess) return CERE_OK;
    cudaGetLastError();
    return e == cudaErrorInvalidValue ? CERE_ERR_ARGUMENT : CERE_ERR_INTERNAL;
}

#define DR_CUDA(expr)                                          \
    do {                                                       \
        const cudaError_t dr_e_ = (expr);                      \
        if (dr_e_ != cudaSuccess) return cuda_status(dr_e_);   \
    } while (0)
#define DR_TRY(expr)                                 \
    do {                                             \
        const cere_status dr_s_ = (expr);            \
        if (dr_s_ != CERE_OK) return dr_s_;          \
    } while (0)

uint64_t align16(uint64_t v) { return (v + 15) / 16 * 16; }

// The packed per-call images (offsets in bytes; device and host images share them).
struct CtxLayout {
    uint64_t rows, cs, src, dst, bytes;
};
CtxLayout ctx_layout(int64_t n) {
    CtxLayout l{};
    l.rows = 0;
    l.cs = align16(l.rows + uint64_t(n) * sizeof(draft::CtxRow));
    l.src = align16(l.cs + uint64_t(n) * D::kHeadDim * sizeof(float));
    l.dst = align16(l.src + uint64_t(n) * sizeof(int32_t));
    l.bytes = align16(l.dst + uint64_t(n) * sizeof(bf16 *));
    return l;
}
struct DraftLayout {
    uint64_t seqs, anchors, frnn, cs, bytes;
};
DraftLayout draft_layout(int64_t B) {
    DraftLayout l{};
    l.seqs = 0;  // [3 layers][B]
    l.anchors = align16(l.seqs + 3 * uint64_t(B) * sizeof(draft::BlockSeq));
    l.frnn = align16(l.anchors + uint64_t(B) * sizeof(uint32_t));
    l.cs = align16(l.frnn + uint64_t(B) * sizeof(bf16 *));
    l.bytes = align16(l.cs + uint64_t(B) * kDraftBlock * D::kHeadDim * sizeof(float));
    return l;
}

size_t attn_ws_max() {
    size_t m = 0;
    for (int64_t b = 1; b <= kMaxDraftSessions; ++b) m = std::max(m, draft::block_attention_workspace_bytes(b));
    return m;
}

// Carves (or, base null, sizes) the workspace in one fixed order.
struct Parts {
    float *alpha;
    bf16 *fold_w, *fold_b, *h0, *cx, *ck, *cv, *res, *x, *out, *q, *k, *v, *attn, *gate, *up, *hid, *base, *inp;
    float *proj;
    bf16 *state, *latent;
    unsigned long long *keys;
    uint32_t *ids;
    uint8_t *attn_ws, *ctx_meta, *draft_meta;
};
uint64_t carve(uint8_t *base, Parts *p) {
    Carver c(base);
    p->alpha = c.take<float>(D::kAlphaRows * D::kTaps);
    p->fold_w = c.take<bf16>(uint64_t(kUnpaddedVocab) * D::kRank);
    p->fold_b = c.take<bf16>(uint64_t(kUnpaddedVocab));
    p->h0 = c.take<bf16>(kRows * H);
    p->cx = c.take<bf16>(3 * kCtxPlane * H);
    p->ck = c.take<bf16>(3 * kCtxPlane * D::kKvRow);
    p->cv = c.take<bf16>(3 * kCtxPlane * D::kKvRow);
    p->res = c.take<bf16>(kRows * H);
    p->x = c.take<bf16>(kRows * H);
    p->out = c.take<bf16>(kRows * H);
    p->q = c.take<bf16>(kRows * D::kQRow);
    p->k = c.take<bf16>(kRows * D::kKvRow);
    p->v = c.take<bf16>(kRows * D::kKvRow);
    p->attn = c.take<bf16>(kRows * D::kQRow);
    p->gate = c.take<bf16>(kRows * kInter);
    p->up = c.take<bf16>(kRows * kInter);
    p->hid = c.take<bf16>(kHidRows * H);
    p->base = c.take<bf16>(uint64_t(kHidRows) * kLogitLd);
    p->inp = c.take<bf16>(kMaxDraftSessions * kRnnIn);
    p->proj = c.take<float>(kMaxDraftSessions * D::kProj);
    p->state = c.take<bf16>(kMaxDraftSessions * D::kRank);
    p->latent = c.take<bf16>(kMaxDraftSessions * D::kRank);
    p->keys = c.take<unsigned long long>(kMaxDraftSessions);
    p->ids = c.take<uint32_t>(kMaxDraftSessions * kMaxBlockDrafts);
    p->attn_ws = c.take<uint8_t>(attn_ws_max());
    p->ctx_meta = c.take<uint8_t>(ctx_layout(kMaxDraftCtxRows).bytes);
    p->draft_meta = c.take<uint8_t>(draft_layout(kMaxDraftSessions).bytes);
    return c.used();
}

// ---- weights binder ------------------------------------------------------------------------------------------------
class Binder {
public:
    Binder(const std::vector<TensorRef> &tensors, std::string *error) : error_(error) {
        for (const TensorRef &t : tensors) {
            if (t.name.rfind("model.", 0) == 0) continue;
            if (!by_name_.emplace(t.name, &t).second) fail(t.name, "duplicate");
        }
    }
    const bf16 *bf(const std::string &name, std::initializer_list<uint64_t> shape) {
        auto it = by_name_.find(name);
        if (it == by_name_.end()) {
            fail(name, "missing");
            return nullptr;
        }
        const TensorRef &t = *it->second;
        bool same = t.dtype == CERE_DTYPE_BF16 && t.ndim == shape.size();
        size_t d = 0;
        for (uint64_t s : shape) same = same && d < 4 && t.shape[d++] == s;
        if (!same) {
            fail(name, "dtype or shape differs from mtp.safetensors'");
            return nullptr;
        }
        if (t.ptr == nullptr) {
            fail(name, "null pointer");
            return nullptr;
        }
        used_.insert(name);
        return static_cast<const bf16 *>(t.ptr);
    }
    bool finish() {
        for (const auto &kv : by_name_) {
            if (used_.count(kv.first) == 0) fail(kv.first, "unknown drafter tensor");
        }
        return ok_;
    }

private:
    void fail(const std::string &name, const char *why) {
        if (ok_ && error_ != nullptr) *error_ = name + ": " + why;
        ok_ = false;
    }
    std::map<std::string, const TensorRef *> by_name_;
    std::set<std::string> used_;
    std::string *error_;
    bool ok_ = true;
};

}  // namespace

bool bind_draft_weights(const std::vector<TensorRef> &tensors, DraftWeights *out, std::string *error) {
    if (out == nullptr) return false;
    Binder b(tensors, error);
    DraftWeights w;
    const uint64_t Hu = uint64_t(H), Q = D::kQRow, KV = D::kKvRow, I = uint64_t(kInter), R = D::kRank;
    w.fuse_weight = b.bf("fuse_weight", {4, 4});
    w.mask_embed = b.bf("mask_embed", {Hu});
    w.norm = b.bf("norm.weight", {Hu});
    for (int i = 0; i < 3; ++i) {
        const std::string p = "layers." + std::to_string(i) + ".";
        DraftLayerWeights &l = w.layers[i];
        l.dflare_norm = b.bf(p + "dflare_norm.weight", {Hu});
        l.in_norm = b.bf(p + "input_layernorm.weight", {Hu});
        l.post_norm = b.bf(p + "post_attention_layernorm.weight", {Hu});
        l.q = b.bf(p + "self_attn.q_proj.weight", {Q, Hu});
        l.k = b.bf(p + "self_attn.k_proj.weight", {KV, Hu});
        l.v = b.bf(p + "self_attn.v_proj.weight", {KV, Hu});
        l.o = b.bf(p + "self_attn.o_proj.weight", {Hu, Q});
        l.q_norm = b.bf(p + "self_attn.q_norm.weight", {uint64_t(D::kHeadDim)});
        l.k_norm = b.bf(p + "self_attn.k_norm.weight", {uint64_t(D::kHeadDim)});
        l.k_ctx = b.bf(p + "self_attn.k_proj_ctx.weight", {KV, Hu});
        l.v_ctx = b.bf(p + "self_attn.v_proj_ctx.weight", {KV, Hu});
        l.k_norm_ctx = b.bf(p + "self_attn.k_norm_ctx.weight", {uint64_t(D::kHeadDim)});
        l.gate = b.bf(p + "mlp.gate_proj.weight", {I, Hu});
        l.up = b.bf(p + "mlp.up_proj.weight", {I, Hu});
        l.down = b.bf(p + "mlp.down_proj.weight", {Hu, I});
    }
    w.markov_norm = b.bf("markov_head.dflare_norm.weight", {Hu});
    w.markov_w1 = b.bf("markov_head.markov_w1.weight", {uint64_t(kVocab), R});
    w.rnn_w = b.bf("markov_head.rnn_proj.weight", {uint64_t(D::kProj), uint64_t(kRnnIn)});
    w.rnn_b = b.bf("markov_head.rnn_proj.bias", {uint64_t(D::kProj)});
    w.out_w = b.bf("markov_head.hidden_out.weight", {Hu, R});
    w.out_b = b.bf("markov_head.hidden_out.bias", {Hu});
    if (!b.finish()) return false;
    *out = w;
    return true;
}

// ---- sessions ------------------------------------------------------------------------------------------------------
uint64_t draft_session_bytes(int64_t max_ctx) {
    if (max_ctx < 1) return 0;
    return align_up(uint64_t(4) * kDraftRingElems * sizeof(bf16)) +
           align_up(uint64_t(2) * uint64_t(max_ctx) * D::kKvRow * sizeof(bf16)) + align_up(uint64_t(H) * sizeof(bf16));
}

DraftSession carve_draft_session(void *base, int64_t max_ctx) {
    DraftSession s;
    if (base == nullptr || max_ctx < 1) return s;
    Carver c(static_cast<uint8_t *>(base));
    s.ring = c.take<bf16>(uint64_t(4) * kDraftRingElems);
    s.full = c.take<bf16>(uint64_t(2) * uint64_t(max_ctx) * D::kKvRow);
    s.f_rnn = c.take<bf16>(uint64_t(H));
    s.max_ctx = max_ctx;
    return s;
}

// ---- Drafter -------------------------------------------------------------------------------------------------------
uint64_t Drafter::workspace_bytes() {
    Parts p{};
    return carve(nullptr, &p);
}

Drafter::~Drafter() {
    for (DraftGraph &g : graphs_) {
        if (g.exec != nullptr) cudaGraphExecDestroy(g.exec);
    }
    if (ctx_meta_host_ != nullptr) cudaFreeHost(ctx_meta_host_);
    if (draft_meta_host_ != nullptr) cudaFreeHost(draft_meta_host_);
    if (ctx_done_ != nullptr) cudaEventDestroy(ctx_done_);
    if (draft_done_ != nullptr) cudaEventDestroy(draft_done_);
    if (capture_ != nullptr) cudaStreamDestroy(capture_);
}

cere_status Drafter::init(const DraftWeights *w, const Weights *target, void *workspace, uint64_t bytes,
                          cublasLtHandle_t handle, cudaStream_t stream) {
    if (w == nullptr || target == nullptr || workspace == nullptr || handle == nullptr || bytes < workspace_bytes() ||
        reinterpret_cast<uintptr_t>(workspace) % kAlign != 0 || w_ != nullptr || target->unembed == nullptr ||
        target->embed == nullptr || target->embed_norm == nullptr || w->fuse_weight == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    Parts p{};
    carve(static_cast<uint8_t *>(workspace), &p);
    alpha_ = p.alpha;
    fold_w_ = p.fold_w;
    fold_b_ = p.fold_b;
    h0_ = p.h0;
    cx_ = p.cx;
    ck_ = p.ck;
    cv_ = p.cv;
    res_ = p.res;
    x_ = p.x;
    out_ = p.out;
    q_ = p.q;
    k_ = p.k;
    v_ = p.v;
    attn_ = p.attn;
    gate_ = p.gate;
    up_ = p.up;
    hid_ = p.hid;
    base_ = p.base;
    inp_ = p.inp;
    proj_ = p.proj;
    state_ = p.state;
    latent_ = p.latent;
    keys_ = p.keys;
    ids_ = p.ids;
    attn_ws_ = p.attn_ws;
    attn_ws_bytes_ = attn_ws_max();
    ctx_meta_dev_ = p.ctx_meta;
    draft_meta_dev_ = p.draft_meta;

    DR_TRY(down_plan_.init(handle, uint32_t(H), uint32_t(kInter), uint32_t(kRows), kernels::GemmOut::kBF16));
    DR_TRY(rnn_plan_.init(handle, uint32_t(D::kProj), uint32_t(kRnnIn), uint32_t(kMaxDraftSessions),
                          kernels::GemmOut::kF32));

    // alpha (drafter_ref.h fuse_alpha, on the host).
    uint16_t fw[D::kAlphaRows * D::kTaps];
    float alpha[D::kAlphaRows * D::kTaps];
    DR_CUDA(cudaMemcpyAsync(fw, w->fuse_weight, sizeof(fw), cudaMemcpyDeviceToHost, stream));
    DR_CUDA(cudaStreamSynchronize(stream));
    D::fuse_alpha(fw, alpha);
    DR_CUDA(cudaMemcpyAsync(alpha_, alpha, sizeof(alpha), cudaMemcpyHostToDevice, stream));
    // Rows 1..7 of every block: mask_embed.
    for (int64_t b = 0; b < kMaxDraftSessions; ++b) {
        for (int r = 1; r < kDraftBlock; ++r) {
            DR_CUDA(cudaMemcpyAsync(h0_ + (b * kDraftBlock + r) * H, w->mask_embed, H * sizeof(bf16),
                                    cudaMemcpyDeviceToDevice, stream));
        }
    }
    DR_CUDA(draft::fold(target->unembed, kUnpaddedVocab, w->out_w, w->out_b, fold_w_, fold_b_, stream));

    DR_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&ctx_meta_host_), ctx_layout(kMaxDraftCtxRows).bytes,
                          cudaHostAllocDefault));
    DR_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&draft_meta_host_), draft_layout(kMaxDraftSessions).bytes,
                          cudaHostAllocDefault));
    ctx_order_.reserve(uint64_t(kMaxDraftCtxRows));
    DR_CUDA(cudaEventCreateWithFlags(&ctx_done_, cudaEventDisableTiming));
    DR_CUDA(cudaEventCreateWithFlags(&draft_done_, cudaEventDisableTiming));
    DR_CUDA(cudaEventRecord(ctx_done_, stream));
    DR_CUDA(cudaEventRecord(draft_done_, stream));
    DR_CUDA(cudaStreamCreateWithFlags(&capture_, cudaStreamNonBlocking));
    DR_CUDA(cudaStreamSynchronize(stream));
    w_ = w;
    t_ = target;
    return CERE_OK;
}

const float *Drafter::rope_row(int64_t pos) {
    const uint64_t need = uint64_t(pos + 1) * D::kHeadDim;
    if (rope_.size() < need) {
        const int64_t have = int64_t(rope_.size() / D::kHeadDim);
        const int64_t to = std::max<int64_t>(pos + 1, (pos / 4096 + 1) * 4096);
        rope_.resize(uint64_t(to) * D::kHeadDim);
        for (int64_t p = have; p < to; ++p) {
            float *r = rope_.data() + uint64_t(p) * D::kHeadDim;
            D::rope_cos_sin(p, r, r + D::kHalf);
        }
    }
    return rope_.data() + uint64_t(pos) * D::kHeadDim;
}

cere_status Drafter::context(const DraftSession *const *sess, const int64_t *pos, const bool *frnn, const bf16 *taps,
                             int64_t tap_ld, int64_t n, cudaStream_t stream) {
    if (w_ == nullptr || sess == nullptr || pos == nullptr || frnn == nullptr || taps == nullptr || n < 1 ||
        n > kMaxDraftCtxRows || tap_ld < 4 * H || tap_ld % 8 != 0) {
        return CERE_ERR_ARGUMENT;
    }
    for (int64_t i = 0; i < n; ++i) {
        const DraftSession *s = sess[i];
        if (s == nullptr || s->ring == nullptr || s->full == nullptr || s->f_rnn == nullptr || pos[i] < 0 ||
            pos[i] >= s->max_ctx) {
            return CERE_ERR_ARGUMENT;
        }
    }
    const CtxLayout lay = ctx_layout(n);
    DR_CUDA(cudaEventSynchronize(ctx_done_));
    uint8_t *img = ctx_meta_host_;
    auto *rows = reinterpret_cast<draft::CtxRow *>(img + lay.rows);
    auto *cs = reinterpret_cast<float *>(img + lay.cs);
    auto *src = reinterpret_cast<int32_t *>(img + lay.src);
    auto *dst = reinterpret_cast<bf16 **>(img + lay.dst);
    int64_t nf = 0;
    for (int64_t i = 0; i < n; ++i) {
        const DraftSession &s = *sess[i];
        rows[i] = draft::CtxRow{s.ring, s.full, pos[i], s.max_ctx, 1};
        std::memcpy(cs + i * D::kHeadDim, rope_row(pos[i]), D::kHeadDim * sizeof(float));
        if (frnn[i]) {
            src[nf] = int32_t(i);
            dst[nf] = s.f_rnn;
            ++nf;
        }
    }
    // A ring slot written by several rows of the call keeps its largest position (the in-order result); a position
    // given twice for one session is refused (its stores would race).
    ctx_order_.resize(uint64_t(n));
    for (int64_t i = 0; i < n; ++i) ctx_order_[uint64_t(i)] = i;
    std::sort(ctx_order_.begin(), ctx_order_.end(), [&](int64_t a, int64_t b) {
        const int64_t sa = rows[a].pos % D::kWindow, sb = rows[b].pos % D::kWindow;
        if (rows[a].ring != rows[b].ring) return std::less<bf16 *>()(rows[a].ring, rows[b].ring);
        return sa != sb ? sa < sb : rows[a].pos > rows[b].pos;
    });
    for (int64_t j = 1; j < n; ++j) {
        const draft::CtxRow &prev = rows[ctx_order_[uint64_t(j - 1)]];
        draft::CtxRow &cur = rows[ctx_order_[uint64_t(j)]];
        if (cur.ring != prev.ring || cur.pos % D::kWindow != prev.pos % D::kWindow) continue;
        if (cur.pos == prev.pos) return CERE_ERR_ARGUMENT;
        cur.ring_write = 0;
    }
    DR_CUDA(cudaMemcpyAsync(ctx_meta_dev_, img, lay.bytes, cudaMemcpyHostToDevice, stream));
    DR_CUDA(cudaEventRecord(ctx_done_, stream));
    const uint8_t *dev = ctx_meta_dev_;
    const DraftWeights &w = *w_;

    // The 3 layers' mixes (alpha rows 0..2, dflare_norm_i) -> k_proj_ctx / v_proj_ctx -> k_norm_ctx + RoPE -> caches.
    draft::MixArgs m;
    m.taps = taps;
    m.tap_ld = tap_ld;
    m.rows = n;
    m.alpha = alpha_;
    m.a0 = 0;
    m.planes = 3;
    for (int l = 0; l < 3; ++l) m.w[l] = w.layers[l].dflare_norm;
    m.y = cx_;
    m.ldy = H;
    m.plane_stride = kCtxPlane * H;
    DR_CUDA(draft::mix_rmsnorm(m, stream));
    LinearArgs a;
    a.m = n;
    a.k = H;
    a.count = 6;
    for (int l = 0; l < 3; ++l) {
        a.seg[2 * l] = {cx_ + l * kCtxPlane * H, H, w.layers[l].k_ctx, ck_ + l * kCtxPlane * D::kKvRow, D::kKvRow,
                        D::kKvRow};
        a.seg[2 * l + 1] = {cx_ + l * kCtxPlane * H, H, w.layers[l].v_ctx, cv_ + l * kCtxPlane * D::kKvRow,
                            D::kKvRow, D::kKvRow};
    }
    DR_CUDA(linear(a, stream));
    const bf16 *const kn[3] = {w.layers[0].k_norm_ctx, w.layers[1].k_norm_ctx, w.layers[2].k_norm_ctx};
    DR_CUDA(draft::ctx_store(ck_, cv_, kCtxPlane * D::kKvRow, kn, reinterpret_cast<const float *>(dev + lay.cs),
                             reinterpret_cast<const draft::CtxRow *>(dev + lay.rows), n, stream));
    if (nf > 0) {
        draft::MixArgs f;
        f.taps = taps;
        f.tap_ld = tap_ld;
        f.src = reinterpret_cast<const int32_t *>(dev + lay.src);
        f.rows = nf;
        f.alpha = alpha_;
        f.a0 = 3;
        f.planes = 1;
        f.w[0] = w.markov_norm;
        f.dst = reinterpret_cast<bf16 *const *>(dev + lay.dst);
        DR_CUDA(draft::mix_rmsnorm(f, stream));
    }
    return CERE_OK;
}

cere_status Drafter::draft(const DraftSession *const *sess, const uint32_t *anchors, const int64_t *L, int64_t B,
                           int n_drafts, uint32_t *ids, cudaStream_t stream) {
    if (w_ == nullptr || sess == nullptr || anchors == nullptr || L == nullptr || ids == nullptr || B < 1 ||
        B > kMaxDraftSessions || n_drafts < 1 || n_drafts > kMaxBlockDrafts) {
        return CERE_ERR_ARGUMENT;
    }
    int64_t max_l = 0;
    for (int64_t b = 0; b < B; ++b) {
        const DraftSession *s = sess[b];
        if (s == nullptr || s->ring == nullptr || s->full == nullptr || s->f_rnn == nullptr || L[b] < 1 ||
            L[b] > s->max_ctx || anchors[b] >= uint32_t(kUnpaddedVocab)) {
            return CERE_ERR_ARGUMENT;
        }
        max_l = std::max(max_l, L[b]);
    }
    const DraftLayout lay = draft_layout(B);
    DR_CUDA(cudaEventSynchronize(draft_done_));
    uint8_t *img = draft_meta_host_;
    auto *seqs = reinterpret_cast<draft::BlockSeq *>(img + lay.seqs);
    auto *anc = reinterpret_cast<uint32_t *>(img + lay.anchors);
    auto *fr = reinterpret_cast<const bf16 **>(img + lay.frnn);
    auto *cs = reinterpret_cast<float *>(img + lay.cs);
    for (int64_t b = 0; b < B; ++b) {
        const DraftSession &s = *sess[b];
        const bf16 *ring0 = s.ring, *ring2 = s.ring + 2 * kDraftRingElems;
        seqs[0 * B + b] = draft::BlockSeq{ring0, ring0 + kDraftRingElems, L[b], 1};
        seqs[1 * B + b] = draft::BlockSeq{s.full, s.full + uint64_t(s.max_ctx) * D::kKvRow, L[b], 0};
        seqs[2 * B + b] = draft::BlockSeq{ring2, ring2 + kDraftRingElems, L[b], 1};
        anc[b] = anchors[b];
        fr[b] = s.f_rnn;
        for (int r = 0; r < kDraftBlock; ++r) {
            std::memcpy(cs + (b * kDraftBlock + r) * D::kHeadDim, rope_row(L[b] + r), D::kHeadDim * sizeof(float));
        }
    }
    DR_CUDA(cudaMemcpyAsync(draft_meta_dev_, img, lay.bytes, cudaMemcpyHostToDevice, stream));
    DR_CUDA(cudaEventRecord(draft_done_, stream));
    const int64_t bound = (max_l + kDraftBlock + kDraftPosStep - 1) / kDraftPosStep * kDraftPosStep;
    if (use_graphs && probe_logits == nullptr) {
        DR_TRY(replay(B, n_drafts, bound, stream));
    } else {
        DR_TRY(run_draft(B, n_drafts, bound, stream));
    }
    DR_CUDA(cudaMemcpyAsync(ids, ids_, uint64_t(B) * kMaxBlockDrafts * sizeof(uint32_t), cudaMemcpyDeviceToDevice,
                            stream));
    return CERE_OK;
}

cere_status Drafter::replay(int64_t B, int n_drafts, int64_t bound, cudaStream_t stream) {
    DraftGraph *g = nullptr;
    for (DraftGraph &x : graphs_) {
        if (x.used != 0 && x.blocks == B && x.n_drafts == n_drafts && x.bound == bound) g = &x;
    }
    if (g == nullptr) {
        g = &graphs_[0];
        for (DraftGraph &x : graphs_) {
            if (x.used < g->used) g = &x;
        }
        if (g->exec != nullptr) cudaGraphExecDestroy(g->exec);
        *g = DraftGraph{};
        // Captured on the Drafter's own stream (the caller's may be the legacy default stream); replayed on the
        // caller's, after the metadata upload.
        DR_CUDA(cudaStreamBeginCapture(capture_, cudaStreamCaptureModeThreadLocal));
        const cere_status st = run_draft(B, n_drafts, bound, capture_);
        cudaGraph_t graph = nullptr;
        cudaError_t e = cudaStreamEndCapture(capture_, &graph);
        if (st == CERE_OK && e == cudaSuccess) e = cudaGraphInstantiate(&g->exec, graph, 0);
        if (graph != nullptr) cudaGraphDestroy(graph);
        if (st != CERE_OK) return st;
        if (e != cudaSuccess) {
            g->exec = nullptr;
            return cuda_status(e);
        }
        g->blocks = B;
        g->n_drafts = n_drafts;
        g->bound = bound;
    }
    g->used = ++graph_clock_;
    DR_CUDA(cudaGraphLaunch(g->exec, stream));
    return CERE_OK;
}

// The whole draft call from the uploaded image: device work only (no host reads), so it captures.
cere_status Drafter::run_draft(int64_t B, int n, int64_t bound, cudaStream_t stream) {
    const DraftWeights &w = *w_;
    const DraftLayout lay = draft_layout(B);
    const uint8_t *dev = draft_meta_dev_;
    const auto *seqs = reinterpret_cast<const draft::BlockSeq *>(dev + lay.seqs);
    const auto *anc = reinterpret_cast<const uint32_t *>(dev + lay.anchors);
    const auto *frnn = reinterpret_cast<const bf16 *const *>(dev + lay.frnn);
    const auto *cs = reinterpret_cast<const float *>(dev + lay.cs);
    const int64_t rows = B * kDraftBlock;

    // Row 0 of each block: the anchor's normed target embedding (rows 1..7 hold mask_embed since init).
    DR_CUDA(embed_norm(t_->embed, anc, B, t_->embed_norm, kRmsEps, h0_, kDraftBlock * H, stream));
    for (int l = 0; l < 3; ++l) {
        const DraftLayerWeights &ly = w.layers[l];
        draft::AddNormArgs pre;
        pre.add_in = l == 0 ? nullptr : out_;
        pre.res_in = l == 0 ? h0_ : res_;
        pre.res_out = res_;
        pre.w = ly.in_norm;
        pre.y = x_;
        pre.rows = rows;
        DR_CUDA(draft::add_rmsnorm(pre, stream));
        LinearArgs qkv;
        qkv.m = rows;
        qkv.k = H;
        qkv.count = 3;
        qkv.seg[0] = {x_, H, ly.q, q_, D::kQRow, D::kQRow};
        qkv.seg[1] = {x_, H, ly.k, k_, D::kKvRow, D::kKvRow};
        qkv.seg[2] = {x_, H, ly.v, v_, D::kKvRow, D::kKvRow};
        DR_CUDA(linear(qkv, stream));
        DR_CUDA(draft::qk_norm_rope(q_, D::kQRow, ly.q_norm, k_, D::kKvRow, ly.k_norm, cs, rows, stream));
        draft::BlockAttnArgs at;
        at.seqs = seqs + l * B;
        at.blocks = B;
        at.local = D::is_local(l);
        at.q = q_;
        at.k = k_;
        at.v = v_;
        at.o = attn_;
        at.bound = bound;
        at.ws = attn_ws_;
        at.ws_bytes = attn_ws_bytes_;
        DR_CUDA(draft::block_attention(at, stream));
        LinearArgs o;
        o.m = rows;
        o.k = D::kQRow;
        o.count = 1;
        o.seg[0] = {attn_, D::kQRow, ly.o, out_, H, H};
        DR_CUDA(linear(o, stream));
        draft::AddNormArgs post;
        post.add_in = out_;
        post.res_in = res_;
        post.res_out = res_;
        post.w = ly.post_norm;
        post.y = x_;
        post.rows = rows;
        DR_CUDA(draft::add_rmsnorm(post, stream));
        LinearArgs gu;
        gu.m = rows;
        gu.k = H;
        gu.count = 2;
        gu.seg[0] = {x_, H, ly.gate, gate_, kInter, kInter};
        gu.seg[1] = {x_, H, ly.up, up_, kInter, kInter};
        DR_CUDA(linear(gu, stream));
        DR_CUDA(draft::silu_mul(gate_, up_, gate_, rows, kInter, stream));
        DR_TRY(down_plan_.run(gate_, ly.down, uint32_t(rows), out_, stream));
    }
    // Final norm of the drafted rows 1..n of every block -> hid_ [B * n].
    draft::AddNormArgs fin;
    fin.add_in = out_;
    fin.res_in = res_;
    fin.res_out = res_;
    fin.w = w.norm;
    fin.y = hid_;
    fin.rows = B * n;
    fin.group = kDraftBlock;
    fin.skip = 1;
    fin.take = n;
    DR_CUDA(draft::add_rmsnorm(fin, stream));
    // Base logits: one unembed GEMM over the B * n drafted rows.
    LinearArgs u;
    u.m = B * n;
    u.k = H;
    u.count = 1;
    u.seg[0] = {hid_, H, t_->unembed, base_, kLogitLd, kUnpaddedVocab};
    DR_CUDA(linear(u, stream));
    // Markov steps.
    DR_CUDA(cudaMemsetAsync(state_, 0, uint64_t(B) * D::kRank * sizeof(bf16), stream));
    DR_CUDA(cudaMemsetAsync(keys_, 0, uint64_t(B) * sizeof(unsigned long long), stream));
    for (int j = 0; j < n; ++j) {
        DR_CUDA(draft::markov_input(state_, w.markov_w1, hid_, n, j, frnn, anc, keys_, ids_, inp_, B, true, stream));
        DR_TRY(rnn_plan_.run(inp_, w.rnn_w, uint32_t(B), proj_, stream));
        DR_CUDA(draft::markov_gates(proj_, w.rnn_b, state_, latent_, B, stream));
        bf16 *lg = probe_logits == nullptr ? nullptr : probe_logits + uint64_t(j) * B * kLogitLd;
        DR_CUDA(draft::markov_vocab(latent_, fold_w_, fold_b_, base_ + uint64_t(j) * kLogitLd, int64_t(n) * kLogitLd,
                                    B, keys_, lg, kLogitLd, stream));
    }
    DR_CUDA(draft::markov_input(nullptr, nullptr, nullptr, 0, n, nullptr, anc, keys_, ids_, nullptr, B, false, stream));
    return CERE_OK;
}

}  // namespace cere::inkling
