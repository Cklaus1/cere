// Inkling 2 Mini S9 phase B (inkling design note 3.5, D-I-S9B): the MTP block drafter on the device. The drafter
// (SGLang 61915e54 Inkling2ForConditionalGenerationMTP; write-up an internal note, CPU reference of its
// kernels executor/ref/inkling/drafter_ref.h) proposes up to 7 greedy drafts per block from the target's 4 taps; the
// verify keeps only the target's own greedy tokens, so drafts change speed, never output.
//
// context(): per committed target row, its 3 layers' context K/V (mix_rmsnorm with dflare_norm_i -> k/v_proj_ctx ->
// k_norm_ctx + RoPE) into the session's caches, and on request the Markov head's f_rnn (mix row 3).
// draft(): per block, the 3-layer non-causal block forward over [anchor embed, mask_embed x 7] at positions
// L .. L + 7 (DRAFTER.md section 4), the final norm, the drafted rows' base logits (target unembed), then n_drafts
// sequential Markov steps (section 5): inp = [state, markov_w1[prev], h_j, f_rnn] -> rnn_proj + bias -> gates ->
// bias over the vocab (the fold) + base -> softcap -> first argmax.
// Kernels: kernels/inkling/drafter.h (+ norm.h embed_norm); projections linear() (K <= 3,072) and K2 GemmPlan
// (mlp.down_proj K 9,216, markov_head.rnn_proj K 5,120).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cere_executor.h"
#include "kernels/k2_gemm.h"
#include "model/inkling/forward.h"
#include "model/inkling/weights.h"

namespace cere::inkling {

constexpr int kDraftBlock = 8;
constexpr int kMaxBlockDrafts = 7;
constexpr int64_t kMaxDraftCtxRows = kMaxPrefillRows;  // 4096: context rows of one call
constexpr int64_t kMaxDraftSessions = kMaxDecodeRows;  // 64: blocks of one draft call
constexpr int64_t kDraftPosStep = kGraphPosStep;       // draft graph bounds: max L + 8 rounded up to this

struct DraftLayerWeights {
    const bf16 *dflare_norm = nullptr;  // [2304]
    const bf16 *in_norm = nullptr;      // input_layernorm [2304]
    const bf16 *post_norm = nullptr;    // post_attention_layernorm [2304]
    const bf16 *q = nullptr;            // self_attn.q_proj [3072, 2304]
    const bf16 *k = nullptr;            // self_attn.k_proj [1024, 2304]
    const bf16 *v = nullptr;            // self_attn.v_proj [1024, 2304]
    const bf16 *o = nullptr;            // self_attn.o_proj [2304, 3072]
    const bf16 *q_norm = nullptr;       // [128]
    const bf16 *k_norm = nullptr;       // [128]
    const bf16 *k_ctx = nullptr;        // self_attn.k_proj_ctx [1024, 2304]
    const bf16 *v_ctx = nullptr;        // self_attn.v_proj_ctx [1024, 2304]
    const bf16 *k_norm_ctx = nullptr;   // [128]
    const bf16 *gate = nullptr;         // mlp.gate_proj [9216, 2304]
    const bf16 *up = nullptr;           // mlp.up_proj [9216, 2304]
    const bf16 *down = nullptr;         // mlp.down_proj [2304, 9216]
};

// mtp.safetensors tensors as they reach the device (names exactly as in mtp.safetensors, no prefix:
// "fuse_weight", "mask_embed", "norm.weight", "layers.{0,1,2}.<...>", "markov_head.<...>"), all BF16.
struct DraftWeights {
    const bf16 *fuse_weight = nullptr;  // [4, 4]
    const bf16 *mask_embed = nullptr;   // [2304]
    const bf16 *norm = nullptr;         // [2304]
    DraftLayerWeights layers[3];
    const bf16 *markov_norm = nullptr;  // markov_head.dflare_norm [2304]
    const bf16 *markov_w1 = nullptr;    // markov_head.markov_w1 [201024, 256]
    const bf16 *rnn_w = nullptr;        // markov_head.rnn_proj.weight [768, 5120]
    const bf16 *rnn_b = nullptr;        // markov_head.rnn_proj.bias [768]
    const bf16 *out_w = nullptr;        // markov_head.hidden_out.weight [2304, 256]
    const bf16 *out_b = nullptr;        // markov_head.hidden_out.bias [2304]
};
// Binds every drafter tensor of `tensors` (TensorRef as in weights.h) by name, checking dtype and shape; tensors
// with the "model." prefix are ignored. False + *error on missing / wrong shape / duplicate / unknown drafter name.
bool bind_draft_weights(const std::vector<TensorRef> &tensors, DraftWeights *out, std::string *error);

// Per-session drafter state, carved from the session's memory.
struct DraftSession {
    bf16 *ring = nullptr;   // layers 0 and 2: [2][K, V][1024 slots][1024]; position j at slot j % 1024
    bf16 *full = nullptr;   // layer 1: [K, V][max_ctx][1024]; position j at row j
    bf16 *f_rnn = nullptr;  // [2304] the Markov head's context of the last committed position
    int64_t max_ctx = 0;
};
uint64_t draft_session_bytes(int64_t max_ctx);  // 256-byte multiple
DraftSession carve_draft_session(void *base, int64_t max_ctx);

class Drafter {
public:
    // Device workspace of one Drafter (persistent: fold_w BF16 [200058][256], fold_b, alpha; plus the scratch of
    // one context call of kMaxDraftCtxRows rows and one draft call of kMaxDraftSessions blocks). 256-aligned base.
    static uint64_t workspace_bytes();
    // Carves the workspace, computes alpha and the fold (on the GPU in FP64, k ascending, bit-equal to
    // drafter_ref.h fold_row) from target->unembed and the Markov head; synchronizes `stream`. `w` and `target`
    // outlive the Drafter. `handle`: the device's cuBLASLt handle for the K2 GemmPlan projections.
    cere_status init(const DraftWeights *w, const Weights *target, void *workspace, uint64_t bytes,
                     cublasLtHandle_t handle, cudaStream_t stream);
    // Context write: row r < n of taps (BF16; row r at taps + r * tap_ld, tap t at + t * 2304, tap order visits
    // 5, 23, 41 entering = CERE visit outputs 4, 22, 40, then the final-norm output) becomes session *sess[r]'s
    // ctx K/V of the 3 layers at position pos[r] (host arrays [n]); when frnn[r] is true, row r's mix (alpha row 3,
    // markov_head dflare_norm) becomes *sess[r]'s f_rnn. A local layer's ring slot written by several rows of the
    // call keeps the largest position (as in-order stores); a position given twice for one session is
    // CERE_ERR_ARGUMENT.
    cere_status context(const DraftSession *const *sess, const int64_t *pos, const bool *frnn, const bf16 *taps,
                        int64_t tap_ld, int64_t n, cudaStream_t stream);
    // Draft: block b < B of session *sess[b] starts at L[b] (its ctx holds positions < L[b]) with anchor id
    // anchors[b] (host arrays [B]); writes the first n_drafts (1..7) greedy drafts to ids [b * kMaxBlockDrafts + j]
    // (DEVICE uint32 buffer). Every per-call value reaches the kernels through one pinned host image + one H2D copy
    // (as Forward::upload), so the whole call replays as a CUDA graph keyed by (B, n_drafts, bound) where bound =
    // max L rounded up to a fixed step (LRU of 8 graphs, as Forward::replay).
    cere_status draft(const DraftSession *const *sess, const uint32_t *anchors, const int64_t *L, int64_t B,
                      int n_drafts, uint32_t *ids, cudaStream_t stream);

    // Eager draft() instead of the graph replay (timing and kernel tests; the same launch sequence and bits).
    bool use_graphs = true;
    // Kernel-test probe (eager draft() only): when non-null, step j's logits of block b land at
    // probe_logits + (j * B + b) * kLogitLd (BF16, 200058 used).
    bf16 *probe_logits = nullptr;

private:
    cere_status run_draft(int64_t B, int n_drafts, int64_t bound, cudaStream_t stream);
    cere_status replay(int64_t B, int n_drafts, int64_t bound, cudaStream_t stream);
    const float *rope_row(int64_t pos);  // RoPE rows staged by the host (drafter_ref.h rope_cos_sin), cached

    const DraftWeights *w_ = nullptr;
    const Weights *t_ = nullptr;
    kernels::GemmPlan down_plan_, rnn_plan_;
    // Persistent.
    float *alpha_ = nullptr;   // FP32 [4][4] (BF16 values)
    bf16 *fold_w_ = nullptr;   // [200058][256]
    bf16 *fold_b_ = nullptr;   // [200058]
    bf16 *h0_ = nullptr;       // [64 * 8][2304]: row 0 of each block the anchor's embed (per call), rows 1..7 mask
    // Context scratch.
    bf16 *cx_ = nullptr;       // [3][4096][2304] mixes
    bf16 *ck_ = nullptr;       // [3][4096][1024] k_proj_ctx outputs
    bf16 *cv_ = nullptr;       // [3][4096][1024] v_proj_ctx outputs
    // Draft scratch (rows = 8 * 64).
    bf16 *res_ = nullptr, *x_ = nullptr, *out_ = nullptr;  // [512][2304]
    bf16 *q_ = nullptr;                                    // [512][3072]
    bf16 *k_ = nullptr, *v_ = nullptr;                     // [512][1024]
    bf16 *attn_ = nullptr;                                 // [512][3072]
    bf16 *gate_ = nullptr, *up_ = nullptr;                 // [512][9216] (act in gate_)
    bf16 *hid_ = nullptr;                                  // [64 * 7][2304] the drafted rows' final hidden
    bf16 *base_ = nullptr;                                 // [64 * 7][kLogitLd] base logits
    bf16 *inp_ = nullptr;                                  // [64][5120]
    float *proj_ = nullptr;                                // [64][768]
    bf16 *state_ = nullptr, *latent_ = nullptr;            // [64][256]
    unsigned long long *keys_ = nullptr;                   // [64] the Markov steps' argmax keys
    uint32_t *ids_ = nullptr;                              // [64 * 7] the drafts (copied to the caller's ids)
    void *attn_ws_ = nullptr;
    size_t attn_ws_bytes_ = 0;
    // Per-call metadata images (host pinned + device), one H2D copy per call each, packed to the call's size; the
    // events guard the host images against reuse before their copy ends (as Forward::upload).
    uint8_t *ctx_meta_dev_ = nullptr, *ctx_meta_host_ = nullptr;
    uint8_t *draft_meta_dev_ = nullptr, *draft_meta_host_ = nullptr;
    cudaEvent_t ctx_done_ = nullptr, draft_done_ = nullptr;
    std::vector<float> rope_;  // [positions][128] cos then sin
    std::vector<int64_t> ctx_order_;  // context(): the rows by (ring, slot, position desc); kMaxDraftCtxRows reserved
    struct DraftGraph {
        int64_t blocks = 0, bound = 0;
        int n_drafts = 0;
        cudaGraphExec_t exec = nullptr;
        uint64_t used = 0;
    };
    static constexpr int kDraftGraphs = 8;
    DraftGraph graphs_[kDraftGraphs];
    uint64_t graph_clock_ = 0;
    cudaStream_t capture_ = nullptr;

public:
    Drafter() = default;
    Drafter(const Drafter &) = delete;
    Drafter &operator=(const Drafter &) = delete;
    ~Drafter();
};

}  // namespace cere::inkling
