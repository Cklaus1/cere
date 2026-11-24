// Inkling 2 Mini S5: kernel-test entry points of the whole text forward (model/inkling/forward.h) on the caller's
// checkpoint tensors, for executor/tests/gpu/inkling_forward.py (G2 per-visit taps, G3 logits vs golden v2 arm Q,
// G4 determinism). Integers int64, the stream last; 0 = success, CERE_ERR_ARGUMENT for arguments outside the
// contract, other cere_status values from the forward.
//
// Tensors: `count` device pointers (uint64 [count]); names: their checkpoint names, each NUL-terminated, back to back;
// meta: int64 [count, 6] = (CERE_DTYPE_*, ndim, d0, d1, d2, d3). Every `model.llm.*` tensor of the text model must be
// present (model/inkling/weights.h bind_weights); others are ignored.
// Mutants (fail-first, G3): moe_mutant = kernels/inkling/moe.h MoeMutant value; no_messages = 1 runs passes 2-3
// without messages.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cere_executor.h"
#include "model/inkling/forward.h"
#include "model/inkling/verify.h"
#include "model/inkling/weights.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

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

// The model of one ktest call: weights bound on the caller's tensors, a Forward and its workspace.
struct Model {
    Weights w;
    DevBuf ws;
    Forward fwd;
};

cere_status load(const void *tensors, const void *names, const void *meta, int64_t count, int64_t moe_mutant,
                 int64_t no_messages, cudaStream_t stream, Model *m) {
    if (tensors == nullptr || names == nullptr || meta == nullptr || count <= 0 || moe_mutant < 0 ||
        moe_mutant > int64_t(MoeMutant::kAlwaysSix) || (no_messages != 0 && no_messages != 1)) {
        return CERE_ERR_ARGUMENT;
    }
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
    std::string error;
    if (!bind_weights(refs, &m->w, &error)) {
        fprintf(stderr, "cere_ktest_inkling: %s\n", error.c_str());
        return CERE_ERR_ARGUMENT;
    }
    const uint64_t bytes = Forward::workspace_bytes();
    if (const cere_status st = m->ws.alloc(bytes); st != CERE_OK) return st;
    if (const cere_status st = m->fwd.init(&m->w, m->ws.p, bytes, stream); st != CERE_OK) return st;
    m->fwd.mutants.moe = MoeMutant(moe_mutant);
    m->fwd.mutants.no_messages = no_messages != 0;
    return CERE_OK;
}

cere_status cuda(cudaError_t e) {
    if (e == cudaSuccess) return CERE_OK;
    cudaGetLastError();
    return e == cudaErrorMemoryAllocation ? CERE_ERR_CAPACITY : CERE_ERR_INTERNAL;
}

#define KT_TRY(expr)                              \
    do {                                          \
        const cere_status kt_s_ = (expr);         \
        if (kt_s_ != CERE_OK) return int(kt_s_);  \
    } while (0)

// Logits of the last call's rows [0, rows) in blocks of kLogitRows through `scratch` (FP32 [kLogitRows, kLogitLd]):
// argmax of row i to ids[i] (device), the full row to logits + (i - first) * kUnpaddedVocab when i >= first and
// logits is non-null (rows below `first` are not copied).
cere_status emit(Forward &f, int64_t rows, float *scratch, int32_t *ids, float *logits, int64_t first,
                 cudaStream_t stream) {
    for (int64_t b = 0; b < rows; b += kLogitRows) {
        const int64_t m = std::min(kLogitRows, rows - b);
        if (const cere_status st = f.logits(b, m, scratch, kLogitLd, stream); st != CERE_OK) return st;
        if (const cere_status st = f.argmax(scratch, m, kLogitLd, ids + b, stream); st != CERE_OK) return st;
        if (logits != nullptr) {
            const int64_t lo = std::max(b, first);
            if (lo < b + m) {
                const cudaError_t e = cudaMemcpy2DAsync(
                    logits + (lo - first) * kUnpaddedVocab, kUnpaddedVocab * sizeof(float),
                    scratch + (lo - b) * kLogitLd, kLogitLd * sizeof(float), kUnpaddedVocab * sizeof(float),
                    size_t(b + m - lo), cudaMemcpyDeviceToDevice, stream);
                if (e != cudaSuccess) return cuda(e);
            }
        }
    }
    return CERE_OK;
}

// Image placeholders among ids [0, n).
int64_t placeholders(const uint32_t *ids, int64_t n) {
    return std::count(ids, ids + n, kImageToken);
}

}  // namespace

// One sequence of n tokens (ids host int32 [n], each < 200058): positions [0, decode_from) prefilled in calls of
// `chunk` rows (1..4096), then positions [decode_from, n) decoded one call per token (teacher-forced on ids).
// decode_from in [1, n]. Outputs (device memory, each nullable but argmax):
//   visits BF16 [48, n, 2304]: the output of every visit at every position (G2 taps);
//   blocks BF16 [n, 4, 2304]: the final block states h_0, h_6, h_12 (pass 2), h_18;
//   argmax int32 [n]: the greedy id of every position's logits;
//   logits FP32 [n - logit_first, 200058]: the softcapped logits of positions >= logit_first (0 <= logit_first <= n).
// image: null, or device BF16 [image_rows, 2304], the vision tower's rows (cere_ktest_inkling_vision_encode) of the
// prompt's kImageToken placeholders in order; image_rows must equal the placeholders below decode_from.
CERE_KTEST int cere_ktest_inkling_forward(const void *tensors, const void *names, const void *meta, int64_t count,
                                          const void *ids, int64_t n, int64_t chunk, int64_t decode_from,
                                          int64_t moe_mutant, int64_t no_messages, void *visits, void *blocks,
                                          void *argmax, void *logits, int64_t logit_first, const void *image,
                                          int64_t image_rows, cudaStream_t stream) {
    if (ids == nullptr || argmax == nullptr || n < 1 || chunk < 1 || chunk > kMaxPrefillRows || decode_from < 1 ||
        decode_from > n || logit_first < 0 || logit_first > n) {
        return CERE_ERR_ARGUMENT;
    }
    const auto *tok = static_cast<const int32_t *>(ids);
    for (int64_t i = 0; i < n; ++i) {
        if (tok[i] < 0 || tok[i] >= kUnpaddedVocab) return CERE_ERR_ARGUMENT;
    }
    std::vector<uint32_t> u(tok, tok + n);
    if (image != nullptr && image_rows != placeholders(u.data(), decode_from)) return CERE_ERR_ARGUMENT;
    const auto *img = static_cast<const bf16 *>(image);
    Model m;
    KT_TRY(load(tensors, names, meta, count, moe_mutant, no_messages, stream, &m));
    DevBuf sess, scratch;
    KT_TRY(sess.alloc(session_bytes(n)));
    KT_TRY(scratch.alloc(uint64_t(kLogitRows) * kLogitLd * sizeof(float)));
    const Session s = carve_session(sess.p, n);
    KT_TRY(cuda(reset_session(s, stream)));
    auto *vis = static_cast<bf16 *>(visits);
    auto *blk = static_cast<bf16 *>(blocks);
    auto *am = static_cast<int32_t *>(argmax);
    auto *lg = static_cast<float *>(logits);
    int64_t pos = 0;
    while (pos < n) {
        const int64_t rows = pos < decode_from ? std::min(chunk, decode_from - pos) : 1;
        Taps taps;
        taps.visits = vis == nullptr ? nullptr : vis + pos * kHidden;
        taps.visit_stride = n * kHidden;
        taps.blocks = blk == nullptr ? nullptr : blk + pos * 4 * kHidden;
        const Taps *tp = vis == nullptr && blk == nullptr ? nullptr : &taps;
        if (pos < decode_from) {
            KT_TRY(m.fwd.prefill(s, pos, u.data() + pos, rows, tp, img, stream));
            if (img != nullptr) img += placeholders(u.data() + pos, rows) * kHidden;
        } else {
            KT_TRY(m.fwd.decode(&s, &pos, u.data() + pos, 1, tp, stream));
        }
        float *out = nullptr;
        int64_t first = rows;
        if (lg != nullptr && pos + rows > logit_first) {
            const int64_t lo = std::max(pos, logit_first);
            out = lg + (lo - logit_first) * kUnpaddedVocab;
            first = lo - pos;
        }
        KT_TRY(emit(m.fwd, rows, static_cast<float *>(scratch.p), am + pos, out, first, stream));
        pos += rows;
    }
    KT_TRY(cuda(cudaStreamSynchronize(stream)));
    return CERE_OK;
}

// S sessions decoding together (G3 decode, G4 sessions): session i prefills its prompt (prompts host int32,
// concatenated; lens host int64 [S], each >= 1) in calls of `chunk` rows, its first token comes from the prefill's
// last row, then gen - 1 decode calls of M = S rows (one per session) follow. forced host int32 [S, gen] or null:
// teacher forcing (decode step t feeds forced[i, t - 1]) instead of the greedy ids. S in [1, 64], gen >= 1.
// Outputs: ids host int32 [S, gen] (the greedy id of each step's logits); logits device FP32 [S, gen, 200058]
// (nullable). image: null, or device BF16 [image_rows, 2304], the rows of every prompt's kImageToken placeholders in
// prompt order (cere_ktest_inkling_forward); image_rows must equal the prompts' placeholders.
CERE_KTEST int cere_ktest_inkling_generate(const void *tensors, const void *names, const void *meta, int64_t count,
                                           const void *prompts, const void *lens, int64_t sessions, int64_t gen,
                                           const void *forced, int64_t chunk, int64_t moe_mutant, int64_t no_messages,
                                           void *out_ids, void *logits, const void *image, int64_t image_rows,
                                           cudaStream_t stream) {
    if (prompts == nullptr || lens == nullptr || out_ids == nullptr || sessions < 1 || sessions > kMaxDecodeRows ||
        gen < 1 || chunk < 1 || chunk > kMaxPrefillRows) {
        return CERE_ERR_ARGUMENT;
    }
    const auto *len = static_cast<const int64_t *>(lens);
    const auto *ptok = static_cast<const int32_t *>(prompts);
    const auto *ftok = static_cast<const int32_t *>(forced);
    std::vector<int64_t> off(size_t(sessions) + 1, 0);
    int64_t max_len = 0;
    for (int64_t i = 0; i < sessions; ++i) {
        if (len[i] < 1) return CERE_ERR_ARGUMENT;
        off[size_t(i) + 1] = off[size_t(i)] + len[i];
        max_len = std::max(max_len, len[i]);
    }
    std::vector<uint32_t> all(size_t(off.back()));
    for (int64_t i = 0; i < off.back(); ++i) {
        if (ptok[i] < 0 || ptok[i] >= kUnpaddedVocab) return CERE_ERR_ARGUMENT;
        all[size_t(i)] = uint32_t(ptok[i]);
    }
    if (image != nullptr && image_rows != placeholders(all.data(), off.back())) return CERE_ERR_ARGUMENT;
    const auto *img = static_cast<const bf16 *>(image);
    if (ftok != nullptr) {
        for (int64_t i = 0; i < sessions * gen; ++i) {
            if (ftok[i] < 0 || ftok[i] >= kUnpaddedVocab) return CERE_ERR_ARGUMENT;
        }
    }
    Model m;
    KT_TRY(load(tensors, names, meta, count, moe_mutant, no_messages, stream, &m));
    const int64_t ctx = max_len + gen;
    const uint64_t sb = session_bytes(ctx);
    DevBuf sess, scratch, dids;
    KT_TRY(sess.alloc(sb * uint64_t(sessions)));
    KT_TRY(scratch.alloc(uint64_t(kLogitRows) * kLogitLd * sizeof(float)));
    KT_TRY(dids.alloc(uint64_t(kMaxDecodeRows) * sizeof(int32_t)));
    std::vector<Session> ss(static_cast<size_t>(sessions));
    for (int64_t i = 0; i < sessions; ++i) {
        ss[size_t(i)] = carve_session(static_cast<uint8_t *>(sess.p) + sb * uint64_t(i), ctx);
        KT_TRY(cuda(reset_session(ss[size_t(i)], stream)));
    }
    auto *ids = static_cast<int32_t *>(out_ids);
    auto *lg = static_cast<float *>(logits);
    auto *sc = static_cast<float *>(scratch.p);
    auto *di = static_cast<int32_t *>(dids.p);
    const uint64_t step_floats = uint64_t(kUnpaddedVocab);
    // Logit row r of the last call -> session i, step t.
    auto take = [&](int64_t r, int64_t i, int64_t t) -> cere_status {
        if (lg == nullptr) return CERE_OK;
        return cuda(cudaMemcpyAsync(lg + (uint64_t(i) * gen + t) * step_floats, sc + r * kLogitLd,
                                    step_floats * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    };
    for (int64_t i = 0; i < sessions; ++i) {
        std::vector<uint32_t> u(ptok + off[size_t(i)], ptok + off[size_t(i) + 1]);
        const int64_t L = len[i];
        for (int64_t p = 0; p < L; p += chunk) {
            const int64_t rows = std::min(chunk, L - p);
            KT_TRY(m.fwd.prefill(ss[size_t(i)], p, u.data() + p, rows, nullptr, img, stream));
            if (img != nullptr) img += placeholders(u.data() + p, rows) * kHidden;
        }
        const int64_t last = (L - 1) % chunk;
        KT_TRY(m.fwd.logits(last, 1, sc, kLogitLd, stream));
        KT_TRY(m.fwd.argmax(sc, 1, kLogitLd, di, stream));
        KT_TRY(take(0, i, 0));
        int32_t id = 0;
        KT_TRY(cuda(cudaMemcpyAsync(&id, di, sizeof(id), cudaMemcpyDeviceToHost, stream)));
        KT_TRY(cuda(cudaStreamSynchronize(stream)));
        ids[i * gen] = id;
    }
    std::vector<int64_t> pos(static_cast<size_t>(sessions));
    std::vector<uint32_t> in(static_cast<size_t>(sessions));
    std::vector<int32_t> got(static_cast<size_t>(sessions));
    for (int64_t t = 1; t < gen; ++t) {
        for (int64_t i = 0; i < sessions; ++i) {
            pos[size_t(i)] = len[i] + t - 1;
            in[size_t(i)] = uint32_t(ftok != nullptr ? ftok[i * gen + t - 1] : ids[i * gen + t - 1]);
        }
        KT_TRY(m.fwd.decode(ss.data(), pos.data(), in.data(), sessions, nullptr, stream));
        KT_TRY(m.fwd.logits(0, sessions, sc, kLogitLd, stream));
        KT_TRY(m.fwd.argmax(sc, sessions, kLogitLd, di, stream));
        for (int64_t i = 0; i < sessions; ++i) KT_TRY(take(i, i, t));
        KT_TRY(cuda(cudaMemcpyAsync(got.data(), di, size_t(sessions) * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                    stream)));
        KT_TRY(cuda(cudaStreamSynchronize(stream)));
        for (int64_t i = 0; i < sessions; ++i) ids[i * gen + t] = got[size_t(i)];
    }
    KT_TRY(cuda(cudaStreamSynchronize(stream)));
    return CERE_OK;
}

// Speculative verify (S9 phase B, D-I-S9B; model/inkling/forward.h verify/commit) against teacher-forced plain decode:
// the sessions, prompts, gen, forced and outputs of cere_ktest_inkling_generate (forced required). Step 0 comes from
// the prefill; steps 1..gen-1 from verify rounds of all unfinished sessions together: session i at step t runs
// k = min(1 + (round + i) % run, gen - t) rows (every run length 1..run in turn; run in [1, 16]) at positions
// len[i] + t - 1 ..; its first a rows (1 <= a <= k, a hash of (seed, session, round)) feed forced[i, t - 1 + j], the
// rest a wrong token (a rejected draft); the a kept rows give steps t .. t + a - 1 and commit() keeps them. The
// outputs must equal generate's bit for bit (every kept row is the M = 1 decode of its step). mutant: 0 none,
// 1 verify_conv_state, 2 verify_ring_in_place (forward.h Mutants; each must change the outputs).
CERE_KTEST int cere_ktest_inkling_verify(const void *tensors, const void *names, const void *meta, int64_t count,
                                         const void *prompts, const void *lens, int64_t sessions, int64_t gen,
                                         const void *forced, int64_t chunk, int64_t run, int64_t seed, int64_t mutant,
                                         void *out_ids, void *logits, cudaStream_t stream) {
    if (prompts == nullptr || lens == nullptr || forced == nullptr || out_ids == nullptr || sessions < 1 ||
        run < 1 || run > kMaxRunRows || sessions * run > kVerifyRows || gen < 1 || chunk < 1 ||
        chunk > kMaxPrefillRows || mutant < 0 || mutant > 2) {
        return CERE_ERR_ARGUMENT;
    }
    const auto *len = static_cast<const int64_t *>(lens);
    const auto *ptok = static_cast<const int32_t *>(prompts);
    const auto *ftok = static_cast<const int32_t *>(forced);
    std::vector<int64_t> off(size_t(sessions) + 1, 0);
    int64_t max_len = 0;
    for (int64_t i = 0; i < sessions; ++i) {
        if (len[i] < 1) return CERE_ERR_ARGUMENT;
        off[size_t(i) + 1] = off[size_t(i)] + len[i];
        max_len = std::max(max_len, len[i]);
    }
    for (int64_t i = 0; i < off.back(); ++i) {
        if (ptok[i] < 0 || ptok[i] >= kUnpaddedVocab) return CERE_ERR_ARGUMENT;
    }
    for (int64_t i = 0; i < sessions * gen; ++i) {
        if (ftok[i] < 0 || ftok[i] >= kUnpaddedVocab) return CERE_ERR_ARGUMENT;
    }
    Model m;
    KT_TRY(load(tensors, names, meta, count, 0, 0, stream, &m));
    m.fwd.mutants.verify_conv_state = mutant == 1;
    m.fwd.mutants.verify_ring_in_place = mutant == 2;
    DevBuf vbuf;
    KT_TRY(vbuf.alloc(Forward::verify_bytes()));
    KT_TRY(m.fwd.attach_verify(vbuf.p, Forward::verify_bytes()));
    const int64_t ctx = max_len + gen;
    const uint64_t sb = session_bytes(ctx);
    DevBuf sess, scratch, dids;
    KT_TRY(sess.alloc(sb * uint64_t(sessions)));
    KT_TRY(scratch.alloc(uint64_t(kLogitRows) * kLogitLd * sizeof(float)));
    KT_TRY(dids.alloc(uint64_t(kMaxDecodeRows) * sizeof(int32_t)));
    std::vector<Session> ss(static_cast<size_t>(sessions));
    for (int64_t i = 0; i < sessions; ++i) {
        ss[size_t(i)] = carve_session(static_cast<uint8_t *>(sess.p) + sb * uint64_t(i), ctx);
        KT_TRY(cuda(reset_session(ss[size_t(i)], stream)));
    }
    auto *ids = static_cast<int32_t *>(out_ids);
    auto *lg = static_cast<float *>(logits);
    auto *sc = static_cast<float *>(scratch.p);
    auto *di = static_cast<int32_t *>(dids.p);
    const uint64_t step_floats = uint64_t(kUnpaddedVocab);
    auto take = [&](int64_t r, int64_t i, int64_t t) -> cere_status {
        if (lg == nullptr) return CERE_OK;
        return cuda(cudaMemcpyAsync(lg + (uint64_t(i) * gen + t) * step_floats, sc + r * kLogitLd,
                                    step_floats * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    };
    for (int64_t i = 0; i < sessions; ++i) {
        std::vector<uint32_t> u(ptok + off[size_t(i)], ptok + off[size_t(i) + 1]);
        const int64_t L = len[i];
        for (int64_t p = 0; p < L; p += chunk) {
            KT_TRY(m.fwd.prefill(ss[size_t(i)], p, u.data() + p, std::min(chunk, L - p), nullptr, nullptr, stream));
        }
        KT_TRY(m.fwd.logits((L - 1) % chunk, 1, sc, kLogitLd, stream));
        KT_TRY(m.fwd.argmax(sc, 1, kLogitLd, di, stream));
        KT_TRY(take(0, i, 0));
        int32_t id = 0;
        KT_TRY(cuda(cudaMemcpyAsync(&id, di, sizeof(id), cudaMemcpyDeviceToHost, stream)));
        KT_TRY(cuda(cudaStreamSynchronize(stream)));
        ids[i * gen] = id;
    }
    // splitmix64 of (seed, session, round, salt).
    auto hash = [&](int64_t i, int64_t round, uint64_t salt) {
        uint64_t z = uint64_t(seed) * 0x9e3779b97f4a7c15ull + uint64_t(i) * 0xbf58476d1ce4e5b9ull +
                     uint64_t(round) * 0x94d049bb133111ebull + salt;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    };
    std::vector<int64_t> step(static_cast<size_t>(sessions), 1);
    std::vector<Session> vs;
    std::vector<int64_t> vpos, vrows, vkeep, vsess, vfirst;
    std::vector<uint32_t> vids;
    std::vector<int32_t> got(static_cast<size_t>(kVerifyRows));
    for (int64_t round = 0;; ++round) {
        vs.clear(), vpos.clear(), vrows.clear(), vkeep.clear(), vsess.clear(), vfirst.clear(), vids.clear();
        for (int64_t i = 0; i < sessions; ++i) {
            const int64_t t = step[size_t(i)];
            if (t >= gen) continue;
            const int64_t k = std::min(1 + (round + i) % run, gen - t);
            const int64_t a = 1 + int64_t(hash(i, round, 2) % uint64_t(k));
            vs.push_back(ss[size_t(i)]);
            vpos.push_back(len[i] + t - 1);
            vrows.push_back(k);
            vkeep.push_back(a);
            vsess.push_back(i);
            vfirst.push_back(int64_t(vids.size()));
            for (int64_t j = 0; j < k; ++j) {
                const int32_t f = ftok[i * gen + t - 1 + j];
                vids.push_back(j < a ? uint32_t(f) : uint32_t((f + 1 + j) % kUnpaddedVocab));
            }
        }
        if (vs.empty()) break;
        const int64_t S = int64_t(vs.size()), n = int64_t(vids.size());
        KT_TRY(m.fwd.verify(vs.data(), vpos.data(), vrows.data(), S, vids.data(), stream));
        KT_TRY(m.fwd.logits(0, n, sc, kLogitLd, stream));
        KT_TRY(m.fwd.argmax(sc, n, kLogitLd, di, stream));
        for (int64_t q = 0; q < S; ++q) {
            for (int64_t j = 0; j < vkeep[size_t(q)]; ++j) {
                KT_TRY(take(vfirst[size_t(q)] + j, vsess[size_t(q)], step[size_t(vsess[size_t(q)])] + j));
            }
        }
        const bf16 *taps = nullptr;
        KT_TRY(m.fwd.commit(vkeep.data(), &taps, stream));
        KT_TRY(cuda(cudaMemcpyAsync(got.data(), di, size_t(n) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream)));
        KT_TRY(cuda(cudaStreamSynchronize(stream)));
        for (int64_t q = 0; q < S; ++q) {
            const int64_t i = vsess[size_t(q)];
            for (int64_t j = 0; j < vkeep[size_t(q)]; ++j) {
                ids[i * gen + step[size_t(i)] + j] = got[size_t(vfirst[size_t(q)] + j)];
            }
            step[size_t(i)] += vkeep[size_t(q)];
        }
    }
    KT_TRY(cuda(cudaStreamSynchronize(stream)));
    return CERE_OK;
}

// Verify pricing (D-I-S9B: the chain length is chosen from measured M-row step time): S sessions each prefill the
// n prompt ids (host int32, chunks of 4,096), then at position n: out[0] = the mean ms of a plain decode step of S
// rows, out[m] (m in [1, max_m], S * max_m <= 64) = the mean ms of a verify of S runs of m rows then a commit of one
// row each; both include the logits and argmax of every row. Graph replays: one warm-up call (captures), then
// `reps` timed calls (the state each call leaves does not change its time). out host double [max_m + 1].
CERE_KTEST int cere_ktest_inkling_verify_time(const void *tensors, const void *names, const void *meta,
                                              int64_t count, const void *ids, int64_t n, int64_t sessions,
                                              int64_t max_m, int64_t reps, void *out, cudaStream_t stream) {
    if (ids == nullptr || out == nullptr || n < 1 || sessions < 1 || max_m < 1 || max_m > kMaxRunRows ||
        sessions * max_m > kVerifyRows || reps < 1) {
        return CERE_ERR_ARGUMENT;
    }
    const auto *tok = static_cast<const int32_t *>(ids);
    for (int64_t i = 0; i < n; ++i) {
        if (tok[i] < 0 || tok[i] >= int32_t(kUnpaddedVocab)) return CERE_ERR_ARGUMENT;
    }
    const std::vector<uint32_t> u(tok, tok + n);
    Model m;
    KT_TRY(load(tensors, names, meta, count, 0, 0, stream, &m));
    DevBuf vbuf, sess, scratch, dids;
    KT_TRY(vbuf.alloc(Forward::verify_bytes()));
    KT_TRY(m.fwd.attach_verify(vbuf.p, Forward::verify_bytes()));
    const int64_t ctx = n + kMaxRunRows + 1;
    const uint64_t sb = session_bytes(ctx);
    KT_TRY(sess.alloc(sb * uint64_t(sessions)));
    KT_TRY(scratch.alloc(uint64_t(kLogitRows) * kLogitLd * sizeof(float)));
    KT_TRY(dids.alloc(uint64_t(kMaxDecodeRows) * sizeof(int32_t)));
    std::vector<Session> ss(static_cast<size_t>(sessions));
    for (int64_t i = 0; i < sessions; ++i) {
        ss[size_t(i)] = carve_session(static_cast<uint8_t *>(sess.p) + sb * uint64_t(i), ctx);
        KT_TRY(cuda(reset_session(ss[size_t(i)], stream)));
        for (int64_t p = 0; p < n; p += kMaxPrefillRows) {
            KT_TRY(m.fwd.prefill(ss[size_t(i)], p, u.data() + p, std::min(kMaxPrefillRows, n - p), nullptr, nullptr,
                                 stream));
        }
    }
    auto *sc = static_cast<float *>(scratch.p);
    auto *di = static_cast<int32_t *>(dids.p);
    struct Events {
        cudaEvent_t a = nullptr, b = nullptr;
        ~Events() {
            if (a != nullptr) cudaEventDestroy(a);
            if (b != nullptr) cudaEventDestroy(b);
        }
    } ev;
    KT_TRY(cuda(cudaEventCreate(&ev.a)));
    KT_TRY(cuda(cudaEventCreate(&ev.b)));
    const std::vector<int64_t> pos(static_cast<size_t>(sessions), n), keep(static_cast<size_t>(sessions), 1);
    const std::vector<uint32_t> in(static_cast<size_t>(sessions * max_m), u[size_t(n - 1)]);
    auto *o = static_cast<double *>(out);
    for (int64_t mm = 0; mm <= max_m; ++mm) {
        const std::vector<int64_t> rows(static_cast<size_t>(sessions), std::max<int64_t>(mm, 1));
        const int64_t total = sessions * std::max<int64_t>(mm, 1);
        auto once = [&]() -> int {
            if (mm == 0) {
                KT_TRY(m.fwd.decode(ss.data(), pos.data(), in.data(), sessions, nullptr, stream));
            } else {
                KT_TRY(m.fwd.verify(ss.data(), pos.data(), rows.data(), sessions, in.data(), stream));
            }
            KT_TRY(m.fwd.logits(0, total, sc, kLogitLd, stream));
            KT_TRY(m.fwd.argmax(sc, total, kLogitLd, di, stream));
            if (mm != 0) {
                const bf16 *taps = nullptr;
                KT_TRY(m.fwd.commit(keep.data(), &taps, stream));
            }
            return CERE_OK;
        };
        KT_TRY(cere_status(once()));
        KT_TRY(cuda(cudaEventRecord(ev.a, stream)));
        for (int64_t r = 0; r < reps; ++r) KT_TRY(cere_status(once()));
        KT_TRY(cuda(cudaEventRecord(ev.b, stream)));
        KT_TRY(cuda(cudaEventSynchronize(ev.b)));
        float ms = 0.0f;
        KT_TRY(cuda(cudaEventElapsedTime(&ms, ev.a, ev.b)));
        o[mm] = double(ms) / double(reps);
    }
    return CERE_OK;
}

// Forward workspace and session bytes (planning; no device work): out int64 [2] = (Forward::workspace_bytes(),
// session_bytes(ctx)).
CERE_KTEST int cere_ktest_inkling_sizes(int64_t ctx, void *out) {
    if (out == nullptr || ctx < 1) return CERE_ERR_ARGUMENT;
    auto *o = static_cast<int64_t *>(out);
    o[0] = int64_t(Forward::workspace_bytes());
    o[1] = int64_t(session_bytes(ctx));
    return CERE_OK;
}
