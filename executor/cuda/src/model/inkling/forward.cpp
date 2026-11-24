// Inkling 2 Mini S5: forward.h.
#include "model/inkling/forward.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "kernels/inkling/attn.h"
#include "kernels/inkling/conv.h"
#include "kernels/inkling/depth.h"
#include "kernels/inkling/linear.h"
#include "kernels/inkling/msg.h"
#include "kernels/inkling/norm.h"
#include "kernels/inkling/softcap.h"
#include "model/inkling/logits.h"
#include "model/inkling/verify.h"

namespace cere::inkling {
namespace {

constexpr int64_t H = kHidden;
constexpr uint64_t kAlign = 256;
constexpr int64_t kRingElems = kRingSlots * kKvRow;  // one K or V ring of a visit

uint64_t align_up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

// A bump allocator over the workspace (or a dry run that only counts, base null).
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

#define INK_CUDA(expr)                                          \
    do {                                                        \
        const cudaError_t ink_e_ = (expr);                      \
        if (ink_e_ != cudaSuccess) return cuda_status(ink_e_);  \
    } while (0)
#define INK_TRY(expr)                                   \
    do {                                                \
        const cere_status ink_s_ = (expr);              \
        if (ink_s_ != CERE_OK) return ink_s_;           \
    } while (0)

size_t moe_bytes() {
    return std::max(moe_workspace_bytes(int(kMaxPrefillRows), true), moe_workspace_bytes(int(kMaxPrefillRows), false));
}

// The fused weighted RMSNorm of a one-query depth mix: y BF16 [n, 2304] = rms_w(mix), weight w.
DepthNorm row_norm(const bf16 *w, bf16 *y) {
    DepthNorm dn{};
    dn.w[0] = w;
    dn.y[0] = y;
    dn.ldy = H;
    dn.eps = kRmsEps;
    return dn;
}

}  // namespace

// The per-call metadata image (host pinned and device, the same bytes). A verify's: run_idx[r] is row r's index in
// its run; stage[i] the one sequence (rows [0, n) at "positions" 0 .. n - 1) that stores the call's sliding K/V of
// sliding visit i into the K/V stage.
struct Forward::Meta {
    uint32_t ids[kMaxPrefillRows];
    int64_t pos[kMaxPrefillRows];
    attn::Seq seqs[plan::kVisits][kMaxDecodeRows];
    float *conv[plan::kVisits][4][kMaxDecodeRows];
    int32_t run_idx[kMaxDecodeRows];
    attn::Seq stage[plan::kSlidingVisits];
};

uint64_t session_bytes(int64_t max_ctx) {
    return align_up(uint64_t(plan::kSlidingVisits) * 2 * kRingElems * sizeof(bf16)) +
           align_up(uint64_t(plan::kFullVisits) * 2 * uint64_t(max_ctx) * kKvRow * sizeof(bf16)) +
           align_up(uint64_t(plan::kVisits) * kConvVisitFloats * sizeof(float));
}

Session carve_session(void *base, int64_t max_ctx) {
    Carver c(static_cast<uint8_t *>(base));
    Session s;
    s.ring = c.take<bf16>(uint64_t(plan::kSlidingVisits) * 2 * kRingElems);
    s.full = c.take<bf16>(uint64_t(plan::kFullVisits) * 2 * uint64_t(max_ctx) * kKvRow);
    s.conv = c.take<float>(uint64_t(plan::kVisits) * kConvVisitFloats);
    s.max_ctx = max_ctx;
    return s;
}

cudaError_t reset_session(const Session &s, cudaStream_t stream) {
    return cudaMemsetAsync(s.conv, 0, uint64_t(plan::kVisits) * kConvVisitFloats * sizeof(float), stream);
}

uint64_t Forward::workspace_bytes() {
    // Mirrors init's carving (dry run).
    Carver c(nullptr);
    const uint64_t N = kMaxPrefillRows, NH = N * H;
    c.take<float>(uint64_t(plan::kLoopLen) * 4 * H);  // mq
    for (int i = 0; i < 2 + 3 + 3 + 2 + 1; ++i) c.take<bf16>(NH);  // h0 h6 h12[3] h18[3] ping[2] final
    for (int i = 0; i < 3; ++i) c.take<bf16>(NH);                   // x[3]
    c.take<bf16>(N * kQRow);                                        // q
    c.take<bf16>(N * kRelRow);                                      // rel
    c.take<bf16>(N * kSinkRow);                                     // sink
    c.take<bf16>(N * kQRow);                                        // gate
    for (int i = 0; i < 4; ++i) c.take<bf16>(N * kKvRow);           // klin vlin kc vc
    for (int i = 0; i < 2; ++i) c.take<bf16>(N * kDynRow);          // dynk dynv
    c.take<bf16>(N * kQRow);                                        // attn
    for (int i = 0; i < 5; ++i) c.take<bf16>(NH);                   // o oc xm mo mc
    c.take<float>(3 * NH);                                          // a[3]
    c.take<float>(4 * NH);                                          // msg[4]
    c.take<float>(NH);                                              // r
    c.take<float>(NH);                                              // u
    c.take<float>(msg_gate_norm_workspace_floats(N));               // msg_gate_norm ws
    c.take<int32_t>(N * kMoeTopK);
    c.take<float>(N * kMoeTopK);
    c.take<uint8_t>(moe_bytes());
    c.take<uint8_t>(attn::decode_workspace_bytes(kMaxDecodeRows));
    c.take<bf16>(uint64_t(kLogitRows) * kLogitLd);
    c.take<uint8_t>(sizeof(Meta));
    return c.used();
}

Forward::~Forward() {
    for (DecodeGraph &g : graphs_) {
        if (g.exec != nullptr) cudaGraphExecDestroy(g.exec);
    }
    if (meta_host_ != nullptr) cudaFreeHost(meta_host_);
    if (meta_done_ != nullptr) cudaEventDestroy(meta_done_);
    if (commit_host_ != nullptr) cudaFreeHost(commit_host_);
    if (commit_done_ != nullptr) cudaEventDestroy(commit_done_);
    if (capture_ != nullptr) cudaStreamDestroy(capture_);
}

uint64_t Forward::verify_bytes() {
    Carver c(nullptr);
    c.take<bf16>(uint64_t(kConvStageElems));
    c.take<bf16>(uint64_t(kKvStageElems));
    c.take<bf16>(uint64_t(kMaxPrefillRows) * kDraftTapLd);
    c.take<bf16>(uint64_t(kVerifyRows) * kDraftTapLd);
    c.take<CommitSeq>(uint64_t(kVerifyRows));
    return c.used();
}

cere_status Forward::attach_verify(void *base, uint64_t bytes) {
    if (base == nullptr || bytes < verify_bytes() || reinterpret_cast<uintptr_t>(base) % kAlign != 0 ||
        conv_stage_ != nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    Carver c(static_cast<uint8_t *>(base));
    conv_stage_ = c.take<bf16>(uint64_t(kConvStageElems));
    kv_stage_ = c.take<bf16>(uint64_t(kKvStageElems));
    dtaps_ = c.take<bf16>(uint64_t(kMaxPrefillRows) * kDraftTapLd);
    ctaps_ = c.take<bf16>(uint64_t(kVerifyRows) * kDraftTapLd);
    commit_dev_ = c.take<CommitSeq>(uint64_t(kVerifyRows));
    INK_CUDA(cudaMallocHost(&commit_host_, uint64_t(kVerifyRows) * sizeof(CommitSeq)));
    INK_CUDA(cudaEventCreateWithFlags(&commit_done_, cudaEventDisableTiming));
    return CERE_OK;
}

cere_status Forward::init(const Weights *w, void *workspace, uint64_t bytes, cudaStream_t stream) {
    if (w == nullptr || workspace == nullptr || bytes < workspace_bytes() ||
        reinterpret_cast<uintptr_t>(workspace) % kAlign != 0 || w_ != nullptr) {
        return CERE_ERR_ARGUMENT;
    }

    Carver c(static_cast<uint8_t *>(workspace));
    const uint64_t NH = uint64_t(kMaxPrefillRows) * H, Nr = uint64_t(kMaxPrefillRows);
    mq_ = c.take<float>(uint64_t(plan::kLoopLen) * 4 * H);
    h0_ = c.take<bf16>(NH);
    h6_ = c.take<bf16>(NH);
    for (auto &p : h12_) p = c.take<bf16>(NH);
    for (auto &p : h18_) p = c.take<bf16>(NH);
    for (auto &p : ping_) p = c.take<bf16>(NH);
    final_ = c.take<bf16>(NH);
    for (auto &p : x_) p = c.take<bf16>(NH);
    q_ = c.take<bf16>(Nr * kQRow);
    rel_ = c.take<bf16>(Nr * kRelRow);
    sink_ = c.take<bf16>(Nr * kSinkRow);
    gate_ = c.take<bf16>(Nr * kQRow);
    klin_ = c.take<bf16>(Nr * kKvRow);
    vlin_ = c.take<bf16>(Nr * kKvRow);
    kc_ = c.take<bf16>(Nr * kKvRow);
    vc_ = c.take<bf16>(Nr * kKvRow);
    dynk_ = c.take<bf16>(Nr * kDynRow);
    dynv_ = c.take<bf16>(Nr * kDynRow);
    attn_ = c.take<bf16>(Nr * kQRow);
    o_ = c.take<bf16>(NH);
    oc_ = c.take<bf16>(NH);
    xm_ = c.take<bf16>(NH);
    mo_ = c.take<bf16>(NH);
    mc_ = c.take<bf16>(NH);
    a_ = c.take<float>(3 * NH);
    msg_ = c.take<float>(4 * NH);
    r_ = c.take<float>(NH);
    u_ = c.take<float>(NH);
    msg_ws_ = c.take<float>(msg_gate_norm_workspace_floats(Nr));
    route_ids_ = c.take<int32_t>(Nr * kMoeTopK);
    route_w_ = c.take<float>(Nr * kMoeTopK);
    moe_ws_bytes_ = moe_bytes();
    moe_ws_ = c.take<uint8_t>(moe_ws_bytes_);
    attn_ws_bytes_ = attn::decode_workspace_bytes(kMaxDecodeRows);
    attn_ws_ = c.take<uint8_t>(attn_ws_bytes_);
    ulogits_ = c.take<bf16>(uint64_t(kLogitRows) * kLogitLd);
    meta_dev_ = c.take<uint8_t>(sizeof(Meta));
    if (c.used() != workspace_bytes()) return CERE_ERR_INTERNAL;

    INK_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&meta_host_), sizeof(Meta), cudaHostAllocDefault));
    std::memset(meta_host_, 0, sizeof(Meta));
    INK_CUDA(cudaEventCreateWithFlags(&meta_done_, cudaEventDisableTiming));
    INK_CUDA(cudaEventRecord(meta_done_, stream));
    INK_CUDA(cudaStreamCreateWithFlags(&capture_, cudaStreamNonBlocking));

    // Merged message queries [12][q, k, v, mlp][2304] (weights.h message_query_row).
    for (int i = 0; i < plan::kLoopLen; ++i) {
        for (int j = 0; j < 4; ++j) {
            INK_CUDA(cudaMemcpyAsync(mq_ + (size_t(i) * 4 + j) * H, message_query_row(*w, i, j), H * sizeof(float),
                                     cudaMemcpyDeviceToDevice, stream));
        }
    }
    INK_CUDA(cudaStreamSynchronize(stream));
    w_ = w;
    return CERE_OK;
}

cere_status Forward::upload(cudaStream_t stream) {
    INK_CUDA(cudaMemcpyAsync(meta_dev_, meta_host_, sizeof(Meta), cudaMemcpyHostToDevice, stream));
    INK_CUDA(cudaEventRecord(meta_done_, stream));
    return CERE_OK;
}

cere_status Forward::prefill(const Session &s, int64_t pos0, const uint32_t *ids, int64_t n, const Taps *taps,
                             const bf16 *image, cudaStream_t stream) {
    if (w_ == nullptr || ids == nullptr || n < 1 || n > kMaxPrefillRows || pos0 < 0 || s.ring == nullptr ||
        s.full == nullptr || s.conv == nullptr || pos0 + n > s.max_ctx) {
        return CERE_ERR_ARGUMENT;
    }
    for (int64_t i = 0; i < n; ++i) {
        if (ids[i] >= uint32_t(kUnpaddedVocab)) return CERE_ERR_ARGUMENT;
    }
    INK_CUDA(cudaEventSynchronize(meta_done_));
    Meta &m = *reinterpret_cast<Meta *>(meta_host_);
    for (int64_t i = 0; i < n; ++i) {
        m.ids[i] = ids[i];
        m.pos[i] = pos0 + i;
    }
    for (int v = 0; v < plan::kVisits; ++v) {
        const plan::Visit &x = plan::kPlan[size_t(v)];
        attn::Seq &q = m.seqs[v][0];
        q.row = 0;
        q.rows = n;
        q.pos0 = pos0;
        q.slots = nullptr;
        if (x.full) {
            q.k_cache = s.full + (uint64_t(x.full_index) * 2) * uint64_t(s.max_ctx) * kKvRow;
            q.v_cache = q.k_cache + uint64_t(s.max_ctx) * kKvRow;
        } else {
            q.k_cache = s.ring + (uint64_t(x.sliding_index) * 2) * kRingElems;
            q.v_cache = q.k_cache + kRingElems;
        }
        for (int c = 0; c < 4; ++c) m.conv[v][c][0] = s.conv + int64_t(v) * kConvVisitFloats + kConvStateOffset[c];
    }
    max_pos_ = pos0 + n;
    verify_ = false;
    image_ = image;
    splice_.clear();
    if (image != nullptr) {
        for (int64_t i = 0; i < n; ++i) {
            if (ids[i] != kImageToken) continue;
            if (!splice_.empty() && splice_.back().first + splice_.back().second == i) {
                ++splice_.back().second;
            } else {
                splice_.emplace_back(i, 1);
            }
        }
    }
    INK_TRY(upload(stream));
    return run(n, false, &s, taps, stream);
}

cere_status Forward::decode(const Session *sessions, const int64_t *pos, const uint32_t *ids, int64_t m,
                            const Taps *taps, cudaStream_t stream) {
    if (w_ == nullptr || sessions == nullptr || pos == nullptr || ids == nullptr || m < 1 || m > kMaxDecodeRows) {
        return CERE_ERR_ARGUMENT;
    }
    int64_t max_pos = 0;
    for (int64_t i = 0; i < m; ++i) {
        const Session &s = sessions[i];
        if (ids[i] >= uint32_t(kUnpaddedVocab) || pos[i] < 0 || pos[i] >= s.max_ctx || s.ring == nullptr ||
            s.full == nullptr || s.conv == nullptr) {
            return CERE_ERR_ARGUMENT;
        }
        for (int64_t j = 0; j < i; ++j) {
            if (sessions[j].conv == s.conv) return CERE_ERR_ARGUMENT;
        }
        max_pos = std::max(max_pos, pos[i] + 1);
    }
    INK_CUDA(cudaEventSynchronize(meta_done_));
    Meta &mt = *reinterpret_cast<Meta *>(meta_host_);
    for (int64_t i = 0; i < m; ++i) {
        mt.ids[i] = ids[i];
        mt.pos[i] = pos[i];
    }
    for (int v = 0; v < plan::kVisits; ++v) {
        const plan::Visit &x = plan::kPlan[size_t(v)];
        for (int64_t i = 0; i < m; ++i) {
            const Session &s = sessions[i];
            attn::Seq &q = mt.seqs[v][i];
            q.row = i;
            q.rows = 1;
            q.pos0 = pos[i];
            q.slots = nullptr;
            if (x.full) {
                q.k_cache = s.full + (uint64_t(x.full_index) * 2) * uint64_t(s.max_ctx) * kKvRow;
                q.v_cache = q.k_cache + uint64_t(s.max_ctx) * kKvRow;
            } else {
                q.k_cache = s.ring + (uint64_t(x.sliding_index) * 2) * kRingElems;
                q.v_cache = q.k_cache + kRingElems;
            }
            for (int c = 0; c < 4; ++c) {
                mt.conv[v][c][i] = s.conv + int64_t(v) * kConvVisitFloats + kConvStateOffset[c];
            }
        }
    }
    max_pos_ = (max_pos + kGraphPosStep - 1) / kGraphPosStep * kGraphPosStep;
    image_ = nullptr;
    splice_.clear();
    INK_TRY(upload(stream));
    verify_ = false;
    if (taps == nullptr) return replay(m, stream);
    return run(m, true, sessions, taps, stream);
}

cere_status Forward::verify(const Session *sessions, const int64_t *pos, const int64_t *rows, int64_t S,
                            const uint32_t *ids, cudaStream_t stream) {
    if (w_ == nullptr || conv_stage_ == nullptr || sessions == nullptr || pos == nullptr || rows == nullptr ||
        ids == nullptr || S < 1 || S > kMaxDecodeRows) {
        return CERE_ERR_ARGUMENT;
    }
    int64_t n = 0, max_pos = 0, max_rows = 0;
    for (int64_t i = 0; i < S; ++i) {
        const Session &s = sessions[i];
        if (rows[i] < 1 || rows[i] > kMaxRunRows || pos[i] < 0 || pos[i] + rows[i] > s.max_ctx ||
            s.ring == nullptr || s.full == nullptr || s.conv == nullptr) {
            return CERE_ERR_ARGUMENT;
        }
        for (int64_t j = 0; j < i; ++j) {
            if (sessions[j].conv == s.conv) return CERE_ERR_ARGUMENT;
        }
        n += rows[i];
        max_pos = std::max(max_pos, pos[i] + rows[i]);
        max_rows = std::max(max_rows, rows[i]);
    }
    if (n > kVerifyRows) return CERE_ERR_ARGUMENT;
    for (int64_t r = 0; r < n; ++r) {
        if (ids[r] >= uint32_t(kUnpaddedVocab)) return CERE_ERR_ARGUMENT;
    }
    INK_CUDA(cudaEventSynchronize(meta_done_));
    Meta &mt = *reinterpret_cast<Meta *>(meta_host_);
    int64_t first = 0;
    for (int64_t i = 0; i < S; ++i) {
        const Session &s = sessions[i];
        vsess_[i] = s;
        vpos_[i] = pos[i];
        vrows_[i] = rows[i];
        vfirst_[i] = first;
        for (int64_t j = 0; j < rows[i]; ++j) {
            mt.ids[first + j] = ids[first + j];
            mt.pos[first + j] = pos[i] + j;
            mt.run_idx[first + j] = int32_t(j);
        }
        for (int v = 0; v < plan::kVisits; ++v) {
            const plan::Visit &x = plan::kPlan[size_t(v)];
            attn::Seq &q = mt.seqs[v][i];
            q.row = first;
            q.rows = rows[i];
            q.pos0 = pos[i];
            q.slots = nullptr;
            if (x.full) {
                q.k_cache = s.full + (uint64_t(x.full_index) * 2) * uint64_t(s.max_ctx) * kKvRow;
                q.v_cache = q.k_cache + uint64_t(s.max_ctx) * kKvRow;
            } else {
                q.k_cache = s.ring + (uint64_t(x.sliding_index) * 2) * kRingElems;
                q.v_cache = q.k_cache + kRingElems;
            }
            for (int c = 0; c < 4; ++c) {
                float *state = s.conv + int64_t(v) * kConvVisitFloats + kConvStateOffset[c];
                for (int64_t j = 0; j < rows[i]; ++j) mt.conv[v][c][first + j] = state;
            }
        }
        first += rows[i];
    }
    for (int i = 0; i < plan::kSlidingVisits; ++i) {
        attn::Seq &q = mt.stage[i];
        q.row = 0;
        q.rows = n;
        q.pos0 = 0;
        q.slots = nullptr;
        q.k_cache = kv_stage_ + uint64_t(i) * 2 * kVerifyRows * kKvRow;
        q.v_cache = q.k_cache + kVerifyRows * kKvRow;
    }
    vseqs_ = S;
    vmax_rows_ = max_rows;
    verify_ = true;
    committed_ = false;
    max_pos_ = (max_pos + kGraphPosStep - 1) / kGraphPosStep * kGraphPosStep;
    image_ = nullptr;
    splice_.clear();
    INK_TRY(upload(stream));
    return replay(n, stream);
}

cere_status Forward::commit(const int64_t *keep, const bf16 **taps, cudaStream_t stream) {
    if (keep == nullptr || taps == nullptr || !verify_ || committed_) return CERE_ERR_ARGUMENT;
    int64_t max_keep = 0;
    for (int64_t i = 0; i < vseqs_; ++i) {
        if (keep[i] < 1 || keep[i] > vrows_[i]) return CERE_ERR_ARGUMENT;
        max_keep = std::max(max_keep, keep[i]);
    }
    INK_CUDA(cudaEventSynchronize(commit_done_));
    auto *cs = static_cast<CommitSeq *>(commit_host_);
    int64_t out = 0;
    for (int64_t i = 0; i < vseqs_; ++i) {
        cs[i] = CommitSeq{vfirst_[i], keep[i], vpos_[i], out, vsess_[i].conv, vsess_[i].ring};
        out += keep[i];
    }
    auto *dev = static_cast<CommitSeq *>(commit_dev_);
    INK_CUDA(cudaMemcpyAsync(dev, cs, uint64_t(vseqs_) * sizeof(CommitSeq), cudaMemcpyHostToDevice, stream));
    INK_CUDA(cudaEventRecord(commit_done_, stream));
    committed_ = true;
    INK_CUDA(conv_commit(conv_stage_, dev, vseqs_, stream));
    if (!mutants.verify_ring_in_place) INK_CUDA(ring_commit(kv_stage_, dev, vseqs_, max_keep, stream));
    // The kept rows' drafter taps: in place when they are already the leading rows (one session).
    if (vseqs_ == 1) {
        *taps = dtaps_;
    } else {
        INK_CUDA(gather_rows(dtaps_, kDraftTapLd, kDraftTapLd, dev, vseqs_, max_keep, ctaps_, kDraftTapLd, stream));
        *taps = ctaps_;
    }
    return CERE_OK;
}

int Forward::mutant_key() const {
    return ((int(mutants.moe) * 2 + int(mutants.no_messages)) * 2 + int(mutants.verify_conv_state)) * 2 +
           int(mutants.verify_ring_in_place);
}

cere_status Forward::replay(int64_t n, cudaStream_t stream) {
    const int mut = mutant_key();
    const int64_t seqs = verify_ ? vseqs_ : 0, max_rows = verify_ ? vmax_rows_ : 0;
    DecodeGraph *g = nullptr;
    for (DecodeGraph &x : graphs_) {
        if (x.used != 0 && x.rows == n && x.bound == max_pos_ && x.mutants == mut && x.seqs == seqs &&
            x.max_rows == max_rows) {
            g = &x;
        }
    }
    if (g == nullptr) {
        g = &graphs_[0];
        for (DecodeGraph &x : graphs_) {
            if (x.used < g->used) g = &x;
        }
        if (g->exec != nullptr) cudaGraphExecDestroy(g->exec);
        *g = DecodeGraph{};
        // Captured on the Forward's own stream (the caller's may be the legacy default stream, which cannot
        // capture); the replay goes to the caller's stream, after the metadata upload.
        INK_CUDA(cudaStreamBeginCapture(capture_, cudaStreamCaptureModeThreadLocal));
        Taps vt;
        vt.draft = dtaps_;
        const cere_status st = run(n, true, nullptr, verify_ ? &vt : nullptr, capture_);
        cudaGraph_t graph = nullptr;
        cudaError_t e = cudaStreamEndCapture(capture_, &graph);
        if (st == CERE_OK && e == cudaSuccess) e = cudaGraphInstantiate(&g->exec, graph, 0);
        if (graph != nullptr) cudaGraphDestroy(graph);
        if (st != CERE_OK) return st;
        if (e != cudaSuccess) {
            g->exec = nullptr;
            return cuda_status(e);
        }
        g->rows = n;
        g->bound = max_pos_;
        g->mutants = mut;
        g->seqs = seqs;
        g->max_rows = max_rows;
    }
    g->used = ++graph_clock_;
    INK_CUDA(cudaGraphLaunch(g->exec, stream));
    rows_ = n;
    return CERE_OK;
}

cere_status Forward::run(int64_t n, bool decode, const Session *sessions, const Taps *taps, cudaStream_t stream) {
    const Weights &w = *w_;
    const Meta *md = reinterpret_cast<const Meta *>(meta_dev_);
    const int64_t *positions = md->pos;
    const uint64_t NH = uint64_t(n) * H;
    const bool msgs_on = !mutants.no_messages;
    rows_ = 0;

    INK_CUDA(embed_norm(w.embed, md->ids, n, w.embed_norm, kRmsEps, h0_, H, stream));
    // Image placeholders take the vision tower's rows (prefill only; splice_ is empty otherwise).
    int64_t fed = 0;
    for (const auto &[row, count] : splice_) {
        INK_CUDA(cudaMemcpyAsync(h0_ + uint64_t(row) * H, image_ + uint64_t(fed) * H,
                                 uint64_t(count) * H * sizeof(bf16), cudaMemcpyDeviceToDevice, stream));
        fed += count;
    }

    // Conv over a visit's rows: one sequence (prefill, state in/out), one row per session (decode), or a verify's
    // runs (verify.h: state read only, the inputs staged for commit()).
    auto stage = [&](int v, int c) { return conv_stage_ + (int64_t(v) * 4 + c) * kConvStageRegion; };
    auto conv_static = [&](const bf16 *x, int64_t ldx, int C, const bf16 *cw, int v, int c, bf16 *y) {
        if (verify_) {
            return conv_static_verify(x, ldx, n, C, cw, md->conv[v][c], md->run_idx, y, C, stage(v, c),
                                      mutants.verify_conv_state, stream);
        }
        if (decode) return conv_static_decode(x, ldx, n, C, cw, md->conv[v][c], y, C, stream);
        return conv_static_prefill(x, ldx, n, C, cw, sessions[0].conv + int64_t(v) * kConvVisitFloats +
                                                         kConvStateOffset[c], y, C, stream);
    };
    auto conv_dynamic = [&](const bf16 *x, int C, const bf16 *sw, const bf16 *dyn, int v, int c, bf16 *y) {
        if (verify_) {
            return conv_dynamic_verify(x, C, n, C, sw, dyn, kDynRow, md->conv[v][c], md->run_idx, y, C, stage(v, c),
                                       mutants.verify_conv_state, stream);
        }
        if (decode) return conv_dynamic_decode(x, C, n, C, sw, dyn, kDynRow, md->conv[v][c], y, C, stream);
        return conv_dynamic_prefill(x, C, n, C, sw, dyn, kDynRow,
                                    sessions[0].conv + int64_t(v) * kConvVisitFloats + kConvStateOffset[c], y, C,
                                    stream);
    };
    auto bstate = [](const bf16 *p) { return DepthState{p, H, 0}; };

    const bf16 *h = h0_;
    for (int v = 0; v < plan::kVisits; ++v) {
        const plan::Visit &x = plan::kPlan[size_t(v)];
        const LayerWeights &lw = w.layers[x.layer];
        const int L = x.layer;
        const int pass12 = plan::is_loop_layer(L) ? x.pass : plan::kPasses - 1;
        if (L == plan::kLoopStart) h = h6_;  // every pass restarts from the pre-loop states (REF :1398-1404)
        bf16 *out = L == 5 ? h6_ : L == 11 ? h12_[x.pass] : L == 17 ? h18_[x.pass] : (h == ping_[0] ? ping_[1] : ping_[0]);

        // Block states B of this visit: h_6j with 6j < L.
        DepthStates st{};
        if (L > 0) st.s[st.count++] = bstate(h0_);
        if (L > 6) st.s[st.count++] = bstate(h6_);
        if (L > 12) st.s[st.count++] = bstate(h12_[pass12]);
        if (L > 18) st.s[st.count++] = bstate(h18_[plan::kPasses - 1]);
        const DepthStates block = st;

        const bool msgs = x.messages && msgs_on;
        if (msgs) {
            DepthStates ms{};
            ms.s[0] = bstate(h12_[x.pass - 1]);
            ms.s[1] = bstate(h18_[x.pass - 1]);
            ms.count = 2;
            INK_CUDA(depth_mix(ms, n, mq_ + size_t(x.loop_index) * 4 * H, 4, msg_, H, int64_t(NH), stream));
        }

        // Attention inputs x[0..2] (q, k, v), BF16.
        int n_in = 1;
        if (L == 0) {
            INK_CUDA(rms_norm(h, false, H, lw.in_norm[0], kRmsEps, n, x_[0], false, H, stream));
        } else {
            st.s[st.count++] = bstate(h);
            const int nq = x.full ? 3 : 1;
            if (msgs) {
                INK_CUDA(depth_mix(st, n, lw.attn_query, nq, a_, H, int64_t(NH), stream));
                MsgGateNorm gates[3];  // q, k, v gates and norms in one call (sliding: all three read a[0])
                for (int j = 0; j < 3; ++j) {
                    const GateWeights &g = lw.gates[j];
                    gates[j] = MsgGateNorm{x.full ? a_ + j * NH : a_, msg_ + j * NH, g.down, g.forget, g.gain,
                                           lw.in_norm[j], x_[j]};
                }
                INK_CUDA(msg_gate_norm(gates, 3, H, H, H, n, kRmsEps, msg_ws_, stream));
                n_in = 3;
            } else {
                DepthNorm dn = row_norm(lw.in_norm[0], x_[0]);
                for (int j = 1; j < nq; ++j) dn.w[j] = lw.in_norm[j], dn.y[j] = x_[j];
                INK_CUDA(depth_mix_norm(st, n, lw.attn_query, nq, dn, stream));
                n_in = nq;
            }
        }
        const bf16 *xq = x_[0], *xk = n_in == 3 ? x_[1] : x_[0], *xv = n_in == 3 ? x_[2] : x_[0];

        // Projections: the q input's four in one call, the k / v inputs' in another (linear.h segments).
        LinearArgs pq;
        pq.m = n;
        pq.k = H;
        pq.count = 4;
        pq.seg[0] = {xq, H, lw.wq, q_, kQRow, kQRow};
        pq.seg[1] = {xq, H, lw.wr, rel_, kRelRow, kRelRow};
        pq.seg[2] = {xq, H, lw.wsink, sink_, kSinkRow, kSinkRow};
        pq.seg[3] = {xq, H, lw.wgate, gate_, kQRow, kQRow};
        INK_CUDA(linear(pq, stream));
        LinearArgs pkv;
        pkv.m = n;
        pkv.k = H;
        pkv.count = x.full ? 4 : 2;
        pkv.seg[0] = {xk, H, lw.wk, klin_, kKvRow, kKvRow};
        pkv.seg[1] = {xv, H, lw.wv, vlin_, kKvRow, kKvRow};
        pkv.seg[2] = {xk, H, lw.k_proj, dynk_, kDynRow, kDynRow};
        pkv.seg[3] = {xv, H, lw.v_proj, dynv_, kDynRow, kDynRow};
        INK_CUDA(linear(pkv, stream));
        if (x.full) {
            INK_CUDA(conv_dynamic(klin_, kKvRow, lw.k_static, dynk_, v, 0, kc_));
            INK_CUDA(conv_dynamic(vlin_, kKvRow, lw.v_static, dynv_, v, 1, vc_));
        } else {
            INK_CUDA(conv_static(klin_, kKvRow, kKvRow, lw.k_conv, v, 0, kc_));
            INK_CUDA(conv_static(vlin_, kKvRow, kKvRow, lw.v_conv, v, 1, vc_));
        }

        // Attention.
        const int kind = x.full ? attn::kFull : attn::kSliding;
        attn::PostArgs pa;
        pa.kind = kind;
        pa.rows = n;
        pa.positions = positions;
        pa.q_in = q_;
        pa.q_scale = lw.q_scale;
        pa.q_out = q_;
        pa.r_in = rel_;
        pa.rel_scale = lw.rel_scale;
        pa.r_out = rel_;
        pa.k_in = kc_;
        pa.k_out = kc_;
        pa.v_in = vc_;
        pa.v_out = vc_;
        INK_CUDA(attn::post(pa, stream));
        attn::AttnArgs aa;
        aa.kind = kind;
        aa.seqs = md->seqs[v];
        aa.n_seqs = verify_ ? int(vseqs_) : decode ? int(n) : 1;
        aa.max_rows = verify_ ? vmax_rows_ : decode ? 1 : n;
        aa.max_pos = max_pos_;
        aa.q = q_;
        aa.r = rel_;
        aa.rel_proj = lw.rel_proj;
        aa.k = kc_;
        aa.v = vc_;
        aa.sink = sink_;
        aa.gate = gate_;
        aa.out = attn_;
        if (decode) {
            INK_CUDA(attn::decode(aa, n, attn_ws_, attn_ws_bytes_, stream));
        } else {
            INK_CUDA(attn::prefill(aa, stream));
        }
        if (verify_ && !x.full && !mutants.verify_ring_in_place) {
            // The sliding K/V rows wait in the stage for commit() (verify.h).
            INK_CUDA(attn::kv_store(kind, md->stage + x.sliding_index, 1, n, kc_, kKvRow, vc_, kKvRow, stream));
        } else {
            INK_CUDA(attn::kv_store(kind, aa.seqs, aa.n_seqs, aa.max_rows, kc_, kKvRow, vc_, kKvRow, stream));
        }
        LinearArgs po;
        po.m = n;
        po.k = kQRow;
        po.count = 1;
        po.seg[0] = {attn_, kQRow, lw.wo, o_, H, H};
        INK_CUDA(linear(po, stream));
        INK_CUDA(conv_static(o_, H, int(H), lw.attn_conv, v, 2, oc_));
        INK_CUDA(attn_residual(x.boundary, h, H, oc_, H, n, r_, H, stream));

        // MLP.
        DepthStates sm = block;
        if (x.boundary) sm.s[sm.count++] = bstate(h);
        sm.s[sm.count++] = DepthState{r_, H, 1};
        if (msgs) {
            INK_CUDA(depth_mix(sm, n, lw.mlp_query, 1, u_, H, 0, stream));
            const GateWeights &g = lw.gates[3];
            const MsgGateNorm gate{u_, msg_ + 3 * NH, g.down, g.forget, g.gain, lw.mlp_norm, xm_};
            INK_CUDA(msg_gate_norm(&gate, 1, H, H, H, n, kRmsEps, msg_ws_, stream));
        } else {
            INK_CUDA(depth_mix_norm(sm, n, lw.mlp_query, 1, row_norm(lw.mlp_norm, xm_), stream));
        }
        INK_CUDA(router(xm_, H, int(n), lw.router_w, lw.router_b, route_ids_, route_w_, nullptr, stream,
                        mutants.moe));
        INK_CUDA(moe_experts(xm_, H, int(n), route_ids_, route_w_, lw.moe, moe_ws_, moe_ws_bytes_, mo_, H, stream,
                             nullptr, mutants.moe));
        INK_CUDA(conv_static(mo_, H, int(H), lw.mlp_conv, v, 3, mc_));
        INK_CUDA(layer_out(r_, H, mc_, H, n, out, H, stream));

        if (taps != nullptr && taps->visits != nullptr) {
            INK_CUDA(cudaMemcpy2DAsync(taps->visits + v * taps->visit_stride, taps->ld * sizeof(bf16), out,
                                       H * sizeof(bf16), H * sizeof(bf16), size_t(n), cudaMemcpyDeviceToDevice,
                                       stream));
        }
        if (taps != nullptr && taps->draft != nullptr) {
            for (int t = 0; t < 3; ++t) {
                if (kDraftTapVisits[t] != v) continue;
                INK_CUDA(cudaMemcpy2DAsync(taps->draft + t * H, kDraftTapLd * sizeof(bf16), out, H * sizeof(bf16),
                                           H * sizeof(bf16), size_t(n), cudaMemcpyDeviceToDevice, stream));
            }
        }
        h = out;
    }

    // Final depth mix and norm (REF :1442-1443).
    DepthStates fs{};
    fs.s[0] = bstate(h0_);
    fs.s[1] = bstate(h6_);
    fs.s[2] = bstate(h12_[plan::kPasses - 1]);
    fs.s[3] = bstate(h18_[plan::kPasses - 1]);
    fs.s[4] = bstate(h);
    fs.count = 5;
    INK_CUDA(depth_mix_norm(fs, n, w.output_query, 1, row_norm(w.final_norm, final_), stream));
    if (taps != nullptr && taps->draft != nullptr) {
        INK_CUDA(cudaMemcpy2DAsync(taps->draft + 3 * H, kDraftTapLd * sizeof(bf16), final_, H * sizeof(bf16),
                                   H * sizeof(bf16), size_t(n), cudaMemcpyDeviceToDevice, stream));
    }
    if (taps != nullptr && taps->blocks != nullptr) {
        for (int b = 0; b < 4; ++b) {
            INK_CUDA(cudaMemcpy2DAsync(taps->blocks + b * H, taps->block_ld * sizeof(bf16), fs.s[b].ptr,
                                       H * sizeof(bf16), H * sizeof(bf16), size_t(n), cudaMemcpyDeviceToDevice,
                                       stream));
        }
    }
    rows_ = n;
    return CERE_OK;
}

cere_status Forward::logits(int64_t first, int64_t rows, float *out, int64_t ld, cudaStream_t stream) {
    if (w_ == nullptr || out == nullptr || first < 0 || rows < 1 || first + rows > rows_ || ld < kUnpaddedVocab) {
        return CERE_ERR_ARGUMENT;
    }
    for (int64_t b = 0; b < rows; b += kLogitRows) {
        const int64_t m = std::min(kLogitRows, rows - b);
        LinearArgs pu;
        pu.m = m;
        pu.k = H;
        pu.count = 1;
        pu.seg[0] = {final_ + (first + b) * H, H, w_->unembed, ulogits_, kLogitLd, kUnpaddedVocab};
        INK_CUDA(linear(pu, stream));
        float *o = out + b * ld;
        INK_CUDA(widen_logits(ulogits_, kLogitLd, m, kUnpaddedVocab, o, ld, stream));
        INK_CUDA(softcap(o, m, kUnpaddedVocab, ld, stream));
    }
    return CERE_OK;
}

cere_status Forward::argmax(const float *logits, int64_t rows, int64_t ld, int32_t *ids, cudaStream_t stream) {
    if (logits == nullptr || ids == nullptr || rows < 1 || ld < kUnpaddedVocab) return CERE_ERR_ARGUMENT;
    INK_CUDA(argmax_rows(logits, rows, kUnpaddedVocab, ld, ids, stream));
    return CERE_OK;
}

}  // namespace cere::inkling
