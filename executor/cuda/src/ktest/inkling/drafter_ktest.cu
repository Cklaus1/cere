// Inkling 2 Mini S9 phase B: kernel-test entry points of the drafter's kernels (executor/cuda/src/kernels/inkling/
// drafter.h), their CPU reference (executor/ref/inkling/drafter_ref.h) and the whole Drafter (model/inkling/
// drafter.h), for executor/tests/gpu/test_inkling_kernels_drafter.py and inkling_drafter_golden.py (prototypes in the
// test's docstring). GPU entries: device pointers; CPU (`_ref_`) entries: host pointers. Integers int64, the stream
// last (ignored by the CPU entries); 0 = success, CERE_ERR_ARGUMENT for arguments outside the contract,
// CERE_ERR_INTERNAL for a failed launch.
//
// Tables: a context row table is int64 [rows, 5] = (ring, full, pos, max_ctx, ring_write) (draft::CtxRow, device
// memory); a block table is int64 [blocks, 4] = (ctx_k, ctx_v, L, ring) (draft::BlockSeq; device memory with device
// cache pointers for the GPU entry, host memory with host cache pointers for the CPU entry; ring 1: ctx row j % 1024).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "cere_executor.h"
#include "inkling/drafter_ref.h"
#include "kernels/inkling/drafter.h"
#include "kernels/k2_gemm.h"
#include "model/inkling/drafter.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace {

namespace K = cere::inkling::draft;
namespace D = ref::drafter;
using K::bf16;

static_assert(sizeof(K::CtxRow) == 5 * sizeof(int64_t), "CtxRow is the int64 [5] row of a context row table");
static_assert(sizeof(K::BlockSeq) == 4 * sizeof(int64_t), "BlockSeq is the int64 [4] row of a block table");

int status_of(cudaError_t err) {
    if (err == cudaSuccess) return CERE_OK;
    cudaGetLastError();
    if (err == cudaErrorInvalidValue) return CERE_ERR_ARGUMENT;
    return CERE_ERR_INTERNAL;
}

const bf16 *in(const void *p) { return static_cast<const bf16 *>(p); }
bf16 *out(void *p) { return static_cast<bf16 *>(p); }
const uint16_t *hin(const void *p) { return static_cast<const uint16_t *>(p); }
uint16_t *hout(void *p) { return static_cast<uint16_t *>(p); }

template <typename F> void parallel_for(int64_t n, F f) {
    const int64_t workers = std::min<int64_t>(n, std::max(1u, std::thread::hardware_concurrency()));
    std::atomic<int64_t> next{0};
    std::vector<std::thread> pool;
    for (int64_t w = 0; w < workers; ++w)
        pool.emplace_back([&] {
            for (int64_t i = next++; i < n; i = next++) f(i);
        });
    for (auto &t : pool) t.join();
}

}  // namespace

// ---- GPU kernels -------------------------------------------------------------------------------------------------

// draft::mix_rmsnorm; src: device int32 [rows] or null; dst unused (planes at y + p * plane_stride).
CERE_KTEST int cere_ktest_inkling_draft_mix(const void *taps, int64_t tap_ld, const void *src, int64_t rows,
                                            const float *alpha, int64_t a0, int64_t planes, const void *w0,
                                            const void *w1, const void *w2, void *y, int64_t ldy, int64_t plane_stride,
                                            cudaStream_t stream) {
    K::MixArgs a;
    a.taps = in(taps);
    a.tap_ld = tap_ld;
    a.src = static_cast<const int32_t *>(src);
    a.rows = rows;
    a.alpha = alpha;
    a.a0 = int(a0);
    a.planes = int(planes);
    a.w[0] = in(w0);
    a.w[1] = in(w1);
    a.w[2] = in(w2);
    a.y = out(y);
    a.ldy = ldy;
    a.plane_stride = plane_stride;
    return status_of(K::mix_rmsnorm(a, stream));
}

CERE_KTEST int cere_ktest_inkling_draft_add_norm(const void *add_in, const void *res_in, void *res_out, const void *w,
                                                 void *y, int64_t rows, int64_t group, int64_t skip, int64_t take,
                                                 cudaStream_t stream) {
    K::AddNormArgs a;
    a.add_in = in(add_in);
    a.res_in = in(res_in);
    a.res_out = out(res_out);
    a.w = in(w);
    a.y = out(y);
    a.rows = rows;
    a.group = int(group);
    a.skip = int(skip);
    a.take = int(take);
    return status_of(K::add_rmsnorm(a, stream));
}

CERE_KTEST int cere_ktest_inkling_draft_silu_mul(const void *g, const void *u, void *act, int64_t rows, int64_t n,
                                                 cudaStream_t stream) {
    return status_of(K::silu_mul(in(g), in(u), out(act), rows, n, stream));
}

CERE_KTEST int cere_ktest_inkling_draft_qk_norm_rope(void *q, int64_t ldq, const void *q_norm, void *k, int64_t ldk,
                                                     const void *k_norm, const float *cs, int64_t rows,
                                                     cudaStream_t stream) {
    return status_of(K::qk_norm_rope(out(q), ldq, in(q_norm), out(k), ldk, in(k_norm), cs, rows, stream));
}

CERE_KTEST int cere_ktest_inkling_draft_ctx_store(const void *k, const void *v, int64_t layer_stride,
                                                  const void *kn0, const void *kn1, const void *kn2, const float *cs,
                                                  const void *rows_tab, int64_t rows, cudaStream_t stream) {
    const bf16 *const kn[3] = {in(kn0), in(kn1), in(kn2)};
    return status_of(K::ctx_store(in(k), in(v), layer_stride, kn, cs, static_cast<const K::CtxRow *>(rows_tab), rows,
                                  stream));
}

CERE_KTEST int64_t cere_ktest_inkling_draft_attention_ws_bytes(int64_t blocks) {
    return int64_t(K::block_attention_workspace_bytes(blocks));
}

// mutant: 0 none, 1 causal block mask, 2 per-row sliding window (draft::AttnMutant).
CERE_KTEST int cere_ktest_inkling_draft_attention(const void *seqs, int64_t blocks, int64_t local, const void *q,
                                                  const void *k, const void *v, void *o, int64_t bound, void *ws,
                                                  int64_t ws_bytes, int64_t mutant, cudaStream_t stream) {
    if (mutant < 0 || mutant > 2 || ws_bytes < 0 || (local != 0 && local != 1)) return CERE_ERR_ARGUMENT;
    K::BlockAttnArgs a;
    a.seqs = static_cast<const K::BlockSeq *>(seqs);
    a.blocks = blocks;
    a.local = local != 0;
    a.q = in(q);
    a.k = in(k);
    a.v = in(v);
    a.o = out(o);
    a.bound = bound;
    a.ws = ws;
    a.ws_bytes = size_t(ws_bytes);
    a.mutant = K::AttnMutant(mutant);
    return status_of(K::block_attention(a, stream));
}

// cere_ktest_inkling_draft_attention's call (no mutant) timed: after 3 warm-up calls, out double [1] (host) = GPU ms
// per call of `reps` back-to-back calls between two events.
CERE_KTEST int cere_ktest_inkling_draft_attention_time(const void *seqs, int64_t blocks, int64_t local,
                                                       const void *q, const void *k, const void *v, void *o,
                                                       int64_t bound, void *ws, int64_t ws_bytes, int64_t reps,
                                                       double *out, cudaStream_t stream) {
    if (out == nullptr || reps < 1) return CERE_ERR_ARGUMENT;
    const auto call = [&] {
        return cere_ktest_inkling_draft_attention(seqs, blocks, local, q, k, v, o, bound, ws, ws_bytes, 0, stream);
    };
    for (int i = 0; i < 3; ++i)
        if (const int st = call(); st != CERE_OK) return st;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, stream);
    for (int64_t i = 0; i < reps; ++i)
        if (const int st = call(); st != CERE_OK) return st;
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    out[0] = double(ms) / double(reps);
    return CERE_OK;
}

CERE_KTEST int cere_ktest_inkling_draft_gates(const float *acc, const void *bias, void *state, void *latent,
                                              int64_t blocks, cudaStream_t stream) {
    return status_of(K::markov_gates(acc, in(bias), out(state), out(latent), blocks, stream));
}

// keys: device uint64 [blocks], zeroed here, then the step's argmax keys (draft::key_id decodes them).
CERE_KTEST int cere_ktest_inkling_draft_vocab(const void *latent, const void *fold_w, const void *fold_b,
                                              const void *base, int64_t ld_base, int64_t blocks, void *keys,
                                              void *logits, int64_t ld_logits, cudaStream_t stream) {
    if (keys == nullptr || blocks < 1 || blocks > 64) return CERE_ERR_ARGUMENT;
    const cudaError_t e = cudaMemsetAsync(keys, 0, size_t(blocks) * sizeof(unsigned long long), stream);
    if (e != cudaSuccess) return status_of(e);
    return status_of(K::markov_vocab(in(latent), in(fold_w), in(fold_b), in(base), ld_base, blocks,
                                     static_cast<unsigned long long *>(keys), out(logits), ld_logits, stream));
}

CERE_KTEST int cere_ktest_inkling_draft_fold(const void *lm, int64_t rows, const void *hw, const void *hb,
                                             void *fold_w, void *fold_b, cudaStream_t stream) {
    return status_of(K::fold(in(lm), rows, in(hw), in(hb), out(fold_w), out(fold_b), stream));
}

// ---- CPU reference -----------------------------------------------------------------------------------------------

CERE_KTEST int cere_ktest_inkling_draft_ref_fuse_alpha(const void *w, float *alpha, cudaStream_t) {
    if (!w || !alpha) return CERE_ERR_ARGUMENT;
    D::fuse_alpha(hin(w), alpha);
    return CERE_OK;
}

// cs FP32 [rows, 128] = rope_cos_sin(pos[i]) (cos then sin).
CERE_KTEST int cere_ktest_inkling_draft_ref_rope(const int64_t *pos, int64_t rows, float *cs, cudaStream_t) {
    if (!pos || !cs || rows < 1) return CERE_ERR_ARGUMENT;
    parallel_for(rows, [&](int64_t i) { D::rope_cos_sin(pos[i], cs + i * D::kHeadDim, cs + i * D::kHeadDim + D::kHalf); });
    return CERE_OK;
}

// Row i: mix_rmsnorm(taps row i, alpha row arow, w) -> y + i * ldy.
CERE_KTEST int cere_ktest_inkling_draft_ref_mix(const void *taps, int64_t tap_ld, int64_t rows, const float *alpha,
                                                int64_t arow, const void *w, void *y, int64_t ldy, cudaStream_t) {
    if (!taps || !alpha || !w || !y || rows < 1 || arow < 0 || arow >= D::kAlphaRows || tap_ld < 4 * D::kHidden)
        return CERE_ERR_ARGUMENT;
    parallel_for(rows, [&](int64_t i) {
        D::mix_rmsnorm(hin(taps) + i * tap_ld, D::kHidden, alpha + arow * D::kTaps, hin(w), hout(y) + i * ldy);
    });
    return CERE_OK;
}

// Row r, head h < heads: head_norm_rope(x + r * ldx + h * 128, w, cs row r, ctx) -> out + r * ldo + h * 128.
CERE_KTEST int cere_ktest_inkling_draft_ref_head_norm_rope(const void *x, int64_t ldx, int64_t rows, int64_t heads,
                                                           const void *w, const float *cs, int64_t ctx, void *o,
                                                           int64_t ldo, cudaStream_t) {
    if (!x || !w || !cs || !o || rows < 1 || heads < 1) return CERE_ERR_ARGUMENT;
    parallel_for(rows, [&](int64_t r) {
        const float *c = cs + r * D::kHeadDim;
        for (int64_t h = 0; h < heads; ++h)
            D::head_norm_rope(hin(x) + r * ldx + h * D::kHeadDim, hin(w), c, c + D::kHalf, ctx != 0,
                              hout(o) + r * ldo + h * D::kHeadDim);
    });
    return CERE_OK;
}

// attend() for every block row and head: q [blocks * 8, 3072], k, v [blocks * 8, 1024] the blocks' own rows,
// o [blocks * 8, 3072]; layer 0..2 (0 and 2 local).
CERE_KTEST int cere_ktest_inkling_draft_ref_attention(const void *seqs, int64_t blocks, int64_t layer, const void *q,
                                                      const void *k, const void *v, void *o, cudaStream_t) {
    if (!seqs || blocks < 1 || layer < 0 || layer > 2 || !q || !k || !v || !o) return CERE_ERR_ARGUMENT;
    const int64_t *t = static_cast<const int64_t *>(seqs);
    parallel_for(blocks * D::kBlock * D::kHeads, [&](int64_t x) {
        const int64_t b = x / (D::kBlock * D::kHeads), r = x / D::kHeads % D::kBlock;
        const int h = int(x % D::kHeads);
        const int64_t *s = t + 4 * b;
        D::BlockKeys keys{s[2],
                          reinterpret_cast<const uint16_t *>(s[0]),
                          reinterpret_cast<const uint16_t *>(s[1]),
                          hin(k) + b * D::kBlock * D::kKvRow,
                          hin(v) + b * D::kBlock * D::kKvRow,
                          s[3] ? D::kWindow : 0};
        const int64_t row = b * D::kBlock + r;
        D::attend(keys, int(layer), hin(q) + row * D::kQRow + h * D::kHeadDim, h,
                  hout(o) + row * D::kQRow + h * D::kHeadDim);
    });
    return CERE_OK;
}

// proj BF16 [blocks, 768]; state BF16 [blocks, 256] in place; latent BF16 [blocks, 256].
CERE_KTEST int cere_ktest_inkling_draft_ref_gates(const void *proj, void *state, void *latent, int64_t blocks,
                                                  cudaStream_t) {
    if (!proj || !state || !latent || blocks < 1) return CERE_ERR_ARGUMENT;
    for (int64_t b = 0; b < blocks; ++b)
        D::markov_gates(hin(proj) + b * D::kProj, hout(state) + b * D::kRank, hout(latent) + b * D::kRank);
    return CERE_OK;
}

// markov_step per block: ids int64 [blocks], logits BF16 [blocks, ld_logits] (nullable).
CERE_KTEST int cere_ktest_inkling_draft_ref_vocab(const void *latent, const void *fold_w, const void *fold_b,
                                                  const void *base, int64_t ld_base, int64_t blocks, int64_t *ids,
                                                  void *logits, int64_t ld_logits, cudaStream_t) {
    if (!latent || !fold_w || !fold_b || !base || !ids || blocks < 1 || ld_base < D::kVocab ||
        (logits && ld_logits < D::kVocab))
        return CERE_ERR_ARGUMENT;
    // Split the vocab across threads per block; the first-max rule is resolved over the chunks in order.
    constexpr int64_t kChunks = 64;
    const int64_t per = (D::kVocab + kChunks - 1) / kChunks;
    for (int64_t b = 0; b < blocks; ++b) {
        std::vector<float> best_v(kChunks, -INFINITY);
        std::vector<uint32_t> best(kChunks, 0);
        const uint16_t *lat = hin(latent) + b * D::kRank;
        uint16_t *lg = logits ? hout(logits) + b * ld_logits : nullptr;
        parallel_for(kChunks, [&](int64_t c) {
            const int64_t v0 = c * per, v1 = std::min<int64_t>(D::kVocab, v0 + per);
            for (int64_t v = v0; v < v1; ++v) {
                const uint16_t l = D::step_logit(lat, hin(fold_w) + v * D::kRank, hin(fold_b)[v],
                                                 hin(base)[b * ld_base + v]);
                if (lg) lg[v] = l;
                const float x = ref::decode_bf16(l);
                if (x > best_v[c]) {
                    best_v[c] = x;
                    best[c] = uint32_t(v);
                }
            }
        });
        float bv = -INFINITY;
        uint32_t id = 0;
        for (int64_t c = 0; c < kChunks; ++c)
            if (best_v[c] > bv) {
                bv = best_v[c];
                id = best[c];
            }
        ids[b] = id;
    }
    return CERE_OK;
}

CERE_KTEST int cere_ktest_inkling_draft_ref_fold(const void *lm, int64_t rows, const void *hw, const void *hb,
                                                 void *fold_w, void *fold_b, cudaStream_t) {
    if (!lm || !hw || !hb || !fold_w || !fold_b || rows < 1) return CERE_ERR_ARGUMENT;
    parallel_for(rows, [&](int64_t r) {
        D::fold_row(hin(lm) + r * D::kHidden, hin(hw), hin(hb), hout(fold_w) + r * D::kRank, hout(fold_b) + r);
    });
    return CERE_OK;
}

// act = bf16(mul(div(g, add(1, exp(-g))), u)) element-wise over count elements.
CERE_KTEST int cere_ktest_inkling_draft_ref_silu_mul(const void *g, const void *u, void *act, int64_t count,
                                                     cudaStream_t) {
    if (!g || !u || !act || count < 1) return CERE_ERR_ARGUMENT;
    parallel_for((count + 4095) / 4096, [&](int64_t c) {
        for (int64_t e = c * 4096; e < std::min(count, (c + 1) * 4096); ++e) {
            const float x = ref::decode_bf16(hin(g)[e]);
            const float s = ref::div(x, ref::add(1.0f, ref::exp(-x)));
            hout(act)[e] = ref::encode_bf16(ref::mul(s, ref::decode_bf16(hin(u)[e])));
        }
    });
    return CERE_OK;
}

// ---- the whole Drafter ---------------------------------------------------------------------------------------------
namespace {

using namespace cere::inkling;

struct DevBuf {
    void *p = nullptr;
    DevBuf() = default;
    DevBuf(const DevBuf &) = delete;
    DevBuf &operator=(const DevBuf &) = delete;
    ~DevBuf() {
        if (p != nullptr) cudaFree(p);
    }
    cere_status alloc(uint64_t bytes) {
        if (cudaMalloc(&p, bytes) != cudaSuccess) {
            cudaGetLastError();
            p = nullptr;
            return CERE_ERR_CAPACITY;
        }
        return CERE_OK;
    }
};

struct DrafterState {
    DraftWeights dw;
    Weights tw;
    DevBuf ws;
    cublasLtHandle_t handle = nullptr;
    std::unique_ptr<Drafter> dr;
    std::vector<std::unique_ptr<DevBuf>> mem;
    std::vector<DraftSession> sess;
    ~DrafterState() {
        dr.reset();
        if (handle != nullptr) cere::kernels::gemm_handle_destroy(handle);
    }
};
std::unique_ptr<DrafterState> g_drafter;

bool session_ptrs(const int64_t *idx, int64_t n, std::vector<const DraftSession *> *out) {
    out->resize(size_t(n));
    for (int64_t i = 0; i < n; ++i) {
        if (idx[i] < 0 || idx[i] >= int64_t(g_drafter->sess.size())) return false;
        (*out)[size_t(i)] = &g_drafter->sess[size_t(idx[i])];
    }
    return true;
}

}  // namespace

// out int64 [2] (host) = (Drafter::workspace_bytes(), draft_session_bytes(max_ctx)).
CERE_KTEST int cere_ktest_inkling_drafter_sizes(int64_t max_ctx, int64_t *out, cudaStream_t) {
    if (out == nullptr) return CERE_ERR_ARGUMENT;
    out[0] = int64_t(Drafter::workspace_bytes());
    out[1] = int64_t(draft_session_bytes(max_ctx));
    return CERE_OK;
}

// Binds the drafter on the caller's device tensors (as cere_ktest_inkling_forward: tensors uint64 [count] device
// pointers, names NUL-separated, meta int64 [count, 6]) and the target's embed [201024, 2304], embed_norm [2304] and
// unembed [201024, 2304] (device BF16), then Drafter::init (the GPU fold). Replaces any open drafter.
CERE_KTEST int cere_ktest_inkling_drafter_open(const void *tensors, const void *names, const void *meta, int64_t count,
                                               const void *embed, const void *embed_norm, const void *unembed,
                                               cudaStream_t stream) {
    g_drafter.reset();
    if (!tensors || !names || !meta || count <= 0 || !embed || !embed_norm || !unembed) return CERE_ERR_ARGUMENT;
    const auto *ptrs = static_cast<const uint64_t *>(tensors);
    const auto *dims = static_cast<const int64_t *>(meta);
    const char *name = static_cast<const char *>(names);
    std::vector<TensorRef> refs(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        TensorRef &t = refs[size_t(i)];
        t.name = name;
        name += t.name.size() + 1;
        const int64_t *d = dims + 6 * i;
        if (d[1] < 0 || d[1] > 4) return CERE_ERR_ARGUMENT;
        t.dtype = uint32_t(d[0]);
        t.ndim = uint32_t(d[1]);
        for (int j = 0; j < 4; ++j) t.shape[j] = uint64_t(d[2 + j]);
        t.ptr = reinterpret_cast<const void *>(ptrs[i]);
    }
    auto s = std::make_unique<DrafterState>();
    std::string error;
    if (!bind_draft_weights(refs, &s->dw, &error)) {
        fprintf(stderr, "cere_ktest_inkling_drafter_open: %s\n", error.c_str());
        return CERE_ERR_ARGUMENT;
    }
    s->tw.embed = in(embed);
    s->tw.embed_norm = in(embed_norm);
    s->tw.unembed = in(unembed);
    if (const cere_status st = cere::kernels::gemm_handle_create(&s->handle); st != CERE_OK) return st;
    const uint64_t bytes = Drafter::workspace_bytes();
    if (const cere_status st = s->ws.alloc(bytes); st != CERE_OK) return st;
    s->dr = std::make_unique<Drafter>();
    if (const cere_status st = s->dr->init(&s->dw, &s->tw, s->ws.p, bytes, s->handle, stream); st != CERE_OK)
        return st;
    g_drafter = std::move(s);
    return CERE_OK;
}

CERE_KTEST int cere_ktest_inkling_drafter_close(cudaStream_t stream) {
    if (stream != nullptr) cudaStreamSynchronize(stream);
    g_drafter.reset();
    return CERE_OK;
}

// Replaces the open drafter's sessions by `count` zeroed sessions of max_ctx positions.
CERE_KTEST int cere_ktest_inkling_drafter_sessions(int64_t count, int64_t max_ctx, cudaStream_t stream) {
    if (!g_drafter || count < 1 || count > kMaxDraftSessions || max_ctx < 1) return CERE_ERR_ARGUMENT;
    cudaStreamSynchronize(stream);
    g_drafter->sess.clear();
    g_drafter->mem.clear();
    const uint64_t bytes = draft_session_bytes(max_ctx);
    for (int64_t i = 0; i < count; ++i) {
        auto m = std::make_unique<DevBuf>();
        if (const cere_status st = m->alloc(bytes); st != CERE_OK) return st;
        if (cudaMemsetAsync(m->p, 0, bytes, stream) != cudaSuccess) return status_of(cudaGetLastError());
        g_drafter->sess.push_back(carve_draft_session(m->p, max_ctx));
        g_drafter->mem.push_back(std::move(m));
    }
    return CERE_OK;
}

// out int64 [4] (host) = (ring, full, f_rnn, max_ctx) of session i (device pointers).
CERE_KTEST int cere_ktest_inkling_drafter_session(int64_t i, int64_t *out, cudaStream_t) {
    if (!g_drafter || out == nullptr || i < 0 || i >= int64_t(g_drafter->sess.size())) return CERE_ERR_ARGUMENT;
    const DraftSession &s = g_drafter->sess[size_t(i)];
    out[0] = int64_t(reinterpret_cast<uintptr_t>(s.ring));
    out[1] = int64_t(reinterpret_cast<uintptr_t>(s.full));
    out[2] = int64_t(reinterpret_cast<uintptr_t>(s.f_rnn));
    out[3] = s.max_ctx;
    return CERE_OK;
}

// Drafter::context: sess, pos, frnn host int64 [n]; taps device BF16 rows at tap_ld.
CERE_KTEST int cere_ktest_inkling_drafter_context(const int64_t *sess, const int64_t *pos, const int64_t *frnn,
                                                  const void *taps, int64_t tap_ld, int64_t n, cudaStream_t stream) {
    if (!g_drafter || !sess || !pos || !frnn || n < 1 || n > kMaxDraftCtxRows) return CERE_ERR_ARGUMENT;
    std::vector<const DraftSession *> sp;
    if (!session_ptrs(sess, n, &sp)) return CERE_ERR_ARGUMENT;
    std::unique_ptr<bool[]> f(new bool[size_t(n)]);
    for (int64_t i = 0; i < n; ++i) f[size_t(i)] = frnn[i] != 0;
    return g_drafter->dr->context(sp.data(), pos, f.get(), in(taps), tap_ld, n, stream);
}

namespace {
int draft_call(const int64_t *sess, const int64_t *anchors, const int64_t *L, int64_t B, int64_t n_drafts,
               int64_t graphs, void *ids, void *probe, cudaStream_t stream) {
    if (!g_drafter || !sess || !anchors || !L || B < 1 || B > kMaxDraftSessions || (graphs != 0 && graphs != 1) ||
        (graphs == 1 && probe != nullptr))
        return CERE_ERR_ARGUMENT;
    std::vector<const DraftSession *> sp;
    if (!session_ptrs(sess, B, &sp)) return CERE_ERR_ARGUMENT;
    std::vector<uint32_t> anc(static_cast<size_t>(B));
    for (int64_t b = 0; b < B; ++b) {
        if (anchors[b] < 0 || anchors[b] > int64_t(UINT32_MAX)) return CERE_ERR_ARGUMENT;
        anc[size_t(b)] = uint32_t(anchors[b]);
    }
    Drafter &d = *g_drafter->dr;
    d.use_graphs = graphs != 0;
    d.probe_logits = out(probe);
    const cere_status st = d.draft(sp.data(), anc.data(), L, B, int(n_drafts), static_cast<uint32_t *>(ids), stream);
    d.probe_logits = nullptr;
    d.use_graphs = true;
    return st;
}
}  // namespace

// Drafter::draft: sess, anchors, L host int64 [B]; ids device uint32 [B, 7]; graphs 1 = graph replay, 0 = eager;
// probe (eager only): device BF16 [n_drafts, B, kLogitLd] step logits, or null.
CERE_KTEST int cere_ktest_inkling_drafter_draft(const int64_t *sess, const int64_t *anchors, const int64_t *L,
                                                int64_t B, int64_t n_drafts, int64_t graphs, void *ids, void *probe,
                                                cudaStream_t stream) {
    return draft_call(sess, anchors, L, B, n_drafts, graphs, ids, probe, stream);
}

// Timing (after 3 warm-up calls): out double [2] (host) = (wall ms per call with a stream synchronize after each,
// GPU ms per call of `reps` back-to-back calls between two events).
CERE_KTEST int cere_ktest_inkling_drafter_draft_time(const int64_t *sess, const int64_t *anchors, const int64_t *L,
                                                     int64_t B, int64_t n_drafts, int64_t graphs, void *ids,
                                                     int64_t reps, double *out, cudaStream_t stream) {
    if (out == nullptr || reps < 1) return CERE_ERR_ARGUMENT;
    for (int i = 0; i < 3; ++i)
        if (const int st = draft_call(sess, anchors, L, B, n_drafts, graphs, ids, nullptr, stream); st != CERE_OK)
            return st;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return status_of(cudaGetLastError());
    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < reps; ++i) {
        if (const int st = draft_call(sess, anchors, L, B, n_drafts, graphs, ids, nullptr, stream); st != CERE_OK)
            return st;
        if (cudaStreamSynchronize(stream) != cudaSuccess) return status_of(cudaGetLastError());
    }
    out[0] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / double(reps);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, stream);
    for (int64_t i = 0; i < reps; ++i)
        if (const int st = draft_call(sess, anchors, L, B, n_drafts, graphs, ids, nullptr, stream); st != CERE_OK)
            return st;
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    out[1] = double(ms) / double(reps);
    return CERE_OK;
}

// Drafter::context timing, as cere_ktest_inkling_drafter_draft_time.
CERE_KTEST int cere_ktest_inkling_drafter_context_time(const int64_t *sess, const int64_t *pos, const int64_t *frnn,
                                                       const void *taps, int64_t tap_ld, int64_t n, int64_t reps,
                                                       double *out, cudaStream_t stream) {
    if (out == nullptr || reps < 1) return CERE_ERR_ARGUMENT;
    auto call = [&] { return cere_ktest_inkling_drafter_context(sess, pos, frnn, taps, tap_ld, n, stream); };
    for (int i = 0; i < 3; ++i)
        if (const int st = call(); st != CERE_OK) return st;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return status_of(cudaGetLastError());
    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < reps; ++i) {
        if (const int st = call(); st != CERE_OK) return st;
        if (cudaStreamSynchronize(stream) != cudaSuccess) return status_of(cudaGetLastError());
    }
    out[0] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / double(reps);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, stream);
    for (int64_t i = 0; i < reps; ++i)
        if (const int st = call(); st != CERE_OK) return st;
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    out[1] = double(ms) / double(reps);
    return CERE_OK;
}
