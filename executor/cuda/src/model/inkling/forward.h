// Inkling 2 Mini (inkling design note 3, D-I2..D-I4, slice S5): the text forward of one call on the device, prefill
// (one session, a chunk of up to 4,096 rows) or decode (up to 64 sessions, one row each), over the 48 visits of
// plan.h, then the final depth mix, the final norm and the logits. Arithmetic lives in the slice kernels
// (kernels/inkling/{norm,depth,msg,attn,conv,moe,softcap}.h, each bit-equal to its executor/ref/inkling/*_ref.h) and
// the BF16 projections in kernels/inkling/linear.h; this file only orders the ops and owns the buffers between them.
// The op order, the buffer of every boundary and its dtype are the forward-graph contract
// (an internal note).
//
// Per visit v (layer L = plan::kPlan[v].layer, pass p), rows N, h = the visit's BF16 input (REF :856-1149):
//   B = the block states h_6j with 6j < L (h_12 of pass p in the loop, of pass 2 after it; h_18 of pass 2)
//   L == 0: xq = rms_w(h, attn_norm) BF16; else a[j] = depth_mix(B ++ [h], attn_queries[L-1][j]) FP32, j < 1 (3 on
//   full layers); passes >= 1 (loop layers): m[0..3] = depth_mix([h_12, h_18 of pass p-1], message queries of
//   L - 6), sliding a[j] = msg_gate_j(a[0], m[j]) j < 3, full a[j] = msg_gate_j(a[j], m[j]); x[j] = rms_w(a[j],
//   attn_norm | q/k/v_input_norm) BF16 (one x on sliding layers without messages).
//   q = x0 Wq, r = x0 Wr, sink = x0 Wsink, gate = x0 Wgate, k = conv_K(x1 Wk [, x1 Wkdyn]), v = conv_V(x2 Wv
//   [, x2 Wvdyn]); attn::post (q, r, k, v in place); attention (prefill or decode) -> out [N, 3072]; attn::kv_store;
//   o = out Wo; oc = conv_attn(o); r = attn_residual(boundary, h, oc) FP32;
//   u = depth_mix(B [++ h if boundary] ++ [r], mlp_queries[L]) FP32; passes >= 1: u = msg_gate_mlp(u, m[3]);
//   xm = rms_w(u, mlp_norm) BF16; router; experts -> mo; mc = conv_mlp(mo); h_next = bf16(r + mc).
// Final: f = depth_mix([h_0, h_6, h_12 (pass 2), h_18, h_24], output_query); hidden = rms_w(f, norm) BF16;
// logits = softcap(float(bf16(hidden Wunembed[:200058]^T))).
//
// Determinism (G4, CERE D9): every kernel computes a row from that row's inputs only, and every projection runs one
// order for every M (linear.h), so a row's bits do not depend on the batch, the chunking or the session count.
//
// Decode graphs (S8): a decode call without taps replays a CUDA graph of its kernels, captured once per (rows,
// attention position bound, mutants). Every kernel reads the call's ids, positions and session tables from the
// metadata image (uploaded eagerly before the replay) and gets host-side bounds only (attn.h), and every decode call,
// eager or replayed, runs attention up to the bound (max position rounded up to kGraphPosStep; the spans past a row's
// position do nothing), so a replay is the eager call's launch sequence and bits.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cere_executor.h"
#include "kernels/inkling/moe.h"
#include "model/inkling/plan.h"
#include "model/inkling/weights.h"

namespace cere::inkling {

constexpr int64_t kMaxPrefillRows = 4096;  // rows of one prefill call (attn::kMaxPrefillRows)
constexpr int64_t kMaxDecodeRows = 64;     // sessions of one decode call (attn::kMaxDecodeRows)
constexpr int64_t kLogitRows = 64;         // rows of one unembed GEMM (logits() loops over blocks of them)
constexpr int64_t kRingSlots = 1024;       // sliding-window KV ring slots (D-I3)
constexpr uint32_t kImageToken = 200054;   // `<|unused_200054|>`, the image placeholder (config image_token_id)
constexpr int64_t kGraphPosStep = 1024;    // decode attention position bounds are multiples of this (4 key spans)
// Conv states of one visit, FP32 [3, C] each (row 0 the oldest; kernels/inkling/conv.h): K, V (C = 1024), the
// post-attention and the post-MLP conv (C = 2304), back to back.
constexpr int64_t kConvStateOffset[4] = {0, 3 * kKvRow, 6 * kKvRow, 6 * kKvRow + 3 * kHidden};
constexpr int64_t kConvVisitFloats = 6 * kKvRow + 6 * kHidden;  // 19,968

// One session's state, device memory the caller owns (D-I3, D-I4). Every visit owns its own KV and conv states:
//   ring: BF16 [40 sliding visits][K, V][1024 slots][1024]; key at position j in slot j % 1024 (attn.h kv_store);
//   full: BF16 [8 full visits][K, V][max_ctx][1024]; key at position j in row j (S2's contiguous per-session pool);
//   conv: FP32 [48 visits][kConvVisitFloats]; zero for a fresh sequence (reset_session).
struct Session {
    bf16 *ring = nullptr;
    bf16 *full = nullptr;
    float *conv = nullptr;
    int64_t max_ctx = 0;  // positions [0, max_ctx) fit the full KV
};

uint64_t session_bytes(int64_t max_ctx);  // 256-byte multiple
// The session carved from `base` (256-byte aligned, session_bytes(max_ctx) bytes).
Session carve_session(void *base, int64_t max_ctx);
// A fresh sequence: zero conv states (the KV holds nothing below the next call's pos0, so it is not cleared).
cudaError_t reset_session(const Session &s, cudaStream_t stream);

// Optional taps of one call (the forward ktest, G2; the drafter's context, S9 phase B): rows in call order.
struct Taps {
    bf16 *visits = nullptr;       // [48][..] the output of visit v, row i at visits + v * visit_stride + i * ld
    int64_t visit_stride = 0;
    int64_t ld = kHidden;
    bf16 *blocks = nullptr;       // [rows][4][2304] the final block states h_0, h_6, h_12 (pass 2), h_18
    int64_t block_ld = 4 * int64_t(kHidden);
    // [rows][4][2304] the drafter's taps (model/inkling/drafter.h context): the outputs of visits kDraftTapVisits,
    // then the final-norm output.
    bf16 *draft = nullptr;
};
constexpr int kDraftTapVisits[3] = {4, 22, 40};  // REF visits 5, 23, 41 entering (SGLang target_layer_ids)
constexpr int64_t kDraftTapLd = 4 * kHidden;

// Fail-first mutants (the forward ktest only; the device runs none): the MoE mutants of kernels/inkling/moe.h and
// passes 2-3 without messages (no message mix, no message gates); the verify's (verify.h): every verify row's conv
// window from its session's state alone (as a lone decode row), and the sliding K/V of every verify row stored in
// the ring in place (the rejected rows left in the ring; the ring commit then does nothing).
struct Mutants {
    MoeMutant moe = MoeMutant::kNone;
    bool no_messages = false;
    bool verify_conv_state = false;
    bool verify_ring_in_place = false;
};

class Forward {
public:
    // Device workspace of one Forward (256-byte aligned base).
    static uint64_t workspace_bytes();
    // Carves the workspace and builds the merged message queries (REF Inkling2LoopMergeMessageQueries) in it;
    // synchronizes `stream`. `w` must outlive the Forward.
    cere_status init(const Weights *w, void *workspace, uint64_t bytes, cudaStream_t stream);
    // Prefill: rows n in [1, 4096] of one session at positions pos0 .. pos0 + n - 1 (pos0 + n <= s.max_ctx; the
    // session holds positions < pos0 from earlier calls). ids host [n] (< 200058). image: null, or the vision
    // tower's rows (BF16 [placeholders of this call, 2304], model/inkling/vision.h) that replace the embedded rows of
    // the call's kImageToken placeholders in order (REF :1681-1694 masked_scatter; the rows are not normed).
    cere_status prefill(const Session &s, int64_t pos0, const uint32_t *ids, int64_t n, const Taps *taps,
                        const bf16 *image, cudaStream_t stream);
    // Decode: m in [1, 64] rows, row i one token ids[i] of sessions[i] at position pos[i] (distinct sessions).
    cere_status decode(const Session *sessions, const int64_t *pos, const uint32_t *ids, int64_t m, const Taps *taps,
                       cudaStream_t stream);

    // Speculative verify (S9 phase B, D-I-S9B; needs attach_verify). S distinct sessions, session i's run of
    // rows[i] in [1, kMaxRunRows] consecutive rows at positions pos[i] .. pos[i] + rows[i] - 1 (<= max_ctx), the
    // runs in session order, ids host [sum rows <= 64]. Each row's logits are bit-equal to those of the same token
    // decoded alone (M = 1) after the run's earlier rows were decoded: the conv windows read the run's earlier
    // inputs and the session's pre-verify state (no state written), the sliding K/V stay staged, the full visits
    // store in place; the drafter taps of the rows go to draft_taps(). Replayed as a CUDA graph keyed by the run
    // layout, the position bound and the mutants. Then commit() once.
    cere_status verify(const Session *sessions, const int64_t *pos, const int64_t *rows, int64_t S,
                       const uint32_t *ids, cudaStream_t stream);
    // Keeps keep[i] in [1, rows[i]] leading rows of each run of the last verify: advances the conv states over them
    // and stores their sliding K/V into the rings. Their drafter taps, compacted in session order, at *taps
    // (kDraftTapLd), valid until the next forward call.
    cere_status commit(const int64_t *keep, const bf16 **taps, cudaStream_t stream);
    // The verify buffers (verify.h stages, the drafter taps of a prefill chunk or a verify). 256-byte aligned base.
    static uint64_t verify_bytes();
    cere_status attach_verify(void *base, uint64_t bytes);
    bf16 *draft_taps() const { return dtaps_; }
    // Softcapped FP32 logits of rows [first, first + rows) of the last call's final hidden states: row i at
    // out + i * ld (ld >= 200058), columns [0, 200058).
    cere_status logits(int64_t first, int64_t rows, float *out, int64_t ld, cudaStream_t stream);
    // Greedy ids of `rows` logit rows (out of logits(), at ld): argmax, NaN the maximum, ties to the lowest id.
    cere_status argmax(const float *logits, int64_t rows, int64_t ld, int32_t *ids, cudaStream_t stream);
    // The last call's final hidden states (BF16 [rows, 2304]).
    const bf16 *hidden() const { return final_; }

    Mutants mutants;

private:
    struct Meta;
    cere_status run(int64_t n, bool decode, const Session *sessions, const Taps *taps, cudaStream_t stream);
    // run(n, decode) of a call without taps (a verify: with its drafter taps): replays (capturing first, if new) the
    // graph of (n, max_pos_, mutants, the verify layout).
    cere_status replay(int64_t n, cudaStream_t stream);
    cere_status upload(cudaStream_t stream);
    int mutant_key() const;

    const Weights *w_ = nullptr;
    // Workspace views (see workspace_bytes for the carving).
    float *mq_ = nullptr;  // merged message queries FP32 [12][4][2304]
    bf16 *h0_ = nullptr, *h6_ = nullptr, *h12_[3] = {}, *h18_[3] = {}, *ping_[2] = {}, *final_ = nullptr;
    bf16 *x_[3] = {}, *q_ = nullptr, *rel_ = nullptr, *sink_ = nullptr, *gate_ = nullptr;
    bf16 *klin_ = nullptr, *vlin_ = nullptr, *kc_ = nullptr, *vc_ = nullptr, *dynk_ = nullptr, *dynv_ = nullptr;
    bf16 *attn_ = nullptr, *o_ = nullptr, *oc_ = nullptr, *xm_ = nullptr, *mo_ = nullptr, *mc_ = nullptr;
    float *a_ = nullptr, *msg_ = nullptr, *r_ = nullptr, *u_ = nullptr, *msg_ws_ = nullptr;
    int32_t *route_ids_ = nullptr;
    float *route_w_ = nullptr;
    void *moe_ws_ = nullptr;
    size_t moe_ws_bytes_ = 0;
    void *attn_ws_ = nullptr;
    size_t attn_ws_bytes_ = 0;
    bf16 *ulogits_ = nullptr;  // BF16 [kLogitRows, kLogitLd]
    // Per-call metadata: a pinned host image and its device copy (ids, positions, attention Seq tables, conv state
    // pointer tables), one H2D copy per call; `meta_done_` guards the host image against reuse before that copy ends.
    uint8_t *meta_dev_ = nullptr;
    uint8_t *meta_host_ = nullptr;
    cudaEvent_t meta_done_ = nullptr;
    int64_t rows_ = 0;
    int64_t max_pos_ = 0;
    // The current verify call (verify_ false for prefill and decode): its sessions, positions and run rows.
    bool verify_ = false;
    int64_t vseqs_ = 0, vmax_rows_ = 0;
    Session vsess_[kMaxDecodeRows] = {};
    int64_t vpos_[kMaxDecodeRows] = {}, vrows_[kMaxDecodeRows] = {}, vfirst_[kMaxDecodeRows] = {};
    bool committed_ = true;
    // attach_verify's buffers: the conv input stage, the sliding K/V stage, the drafter taps (kMaxPrefillRows
    // rows), their compaction, and the commit table (pinned host image + device copy, `commit_done_` its guard).
    bf16 *conv_stage_ = nullptr, *kv_stage_ = nullptr, *dtaps_ = nullptr, *ctaps_ = nullptr;
    void *commit_dev_ = nullptr;
    void *commit_host_ = nullptr;
    cudaEvent_t commit_done_ = nullptr;
    // Decode and verify graphs, least recently used replaced first (used 0 = empty).
    struct DecodeGraph {
        int64_t rows = 0, bound = 0;
        int mutants = 0;
        int64_t seqs = 0, max_rows = 0;  // a verify's layout (seqs 0: a decode)
        cudaGraphExec_t exec = nullptr;
        uint64_t used = 0;
    };
    static constexpr int kDecodeGraphs = 8;
    DecodeGraph graphs_[kDecodeGraphs];
    uint64_t graph_clock_ = 0;
    cudaStream_t capture_ = nullptr;  // the captures' stream (replays go to the caller's)
    // The image rows of the current prefill call: (chunk row, count) runs of placeholders, fed from image_ in order.
    const bf16 *image_ = nullptr;
    std::vector<std::pair<int64_t, int64_t>> splice_;

public:
    Forward() = default;
    Forward(const Forward &) = delete;
    Forward &operator=(const Forward &) = delete;
    ~Forward();
};

}  // namespace cere::inkling
