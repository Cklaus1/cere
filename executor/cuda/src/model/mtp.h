// M2.9 (m29 design note D3, M29S2-04/-12/-19; build-loop-m29 design note C-MODEL): the native MTP (NEXTN) draft layer
// of the A4 model, SGLang's Qwen4ExpForCausalLMMTP (sglang/srt/models/qwen4_exp_mtp.py:73-216) on the native pieces.
//
// One MTP row at position p reads the target's (or the previous MTP step's) pre-final-mixer hc row at p and the token
// at p + 1:
//   e   = fc_embedding(gemma_rmsnorm_2560(embed(next_id)))                       (_fuse_residual_linear_shared)
//   h   = gemma_rmsnorm_10240(hc_in) viewed [4, 2,560]; x_s = fc_hidden(h_s) + e  (the 4 streams)
//   x   = one Qwen4Exp full-attention decoder layer over the hc row x (attn mix, attention with the MTP layer's own
//         KV at p in the MTP pool entry reading [0, p + 1), combine, mlp mix, K13 BF16 MoE, combine)  -> hc_out
//   y   = hyper_connection_mixer.mix(x);  draft = map[argmax(hot_lm_head · y)]
// SGLang's MTP model is a deepcopy of the text config with one full-attention layer and no PLE: RoPE is the target's
// (theta = cfg.rope_theta, partial 0.25; the checkpoint's `mtp.rope_theta` is not read by SGLang), so the MTP layer
// uses ModelWeights::rope_cos_sin, at the target's rotary positions (an image session: its prompt rows' mRoPE
// positions, later rows position + rope_delta; m213l design note C4). Everything runs on the Workspace's row
// buffers and shared scratch (aliased after the target's call, M29S2-12): no allocation, no synchronization.
#pragma once

#include "model/model.h"

namespace cere::model::mtp {

// Rows of the MTP rows plans (GemmPlan::init_rows): calls of m <= kRows rows run them, so row r is bit-equal at any m.
constexpr uint32_t kRows = kernels::GemmPlan::kMaxRows;

// Derived bytes (after the target's derived region): the hot lm_head BF16 [count, 2,560] and the map u32 [count].
uint64_t derived_bytes(const cere_qwen4_exp_config &cfg, uint64_t count);
// The shared-scratch bytes the MTP forward needs for up to max_m rows (the workspace's scratch is the max of this and
// the target's, M29S2-12).
uint64_t scratch_bytes(const cere_qwen4_exp_config &cfg, uint64_t max_m, uint64_t max_context_tokens, uint64_t count);
// Workspace::hc_rows bytes for max_m rows (BF16 [max_m, 10,240]).
uint64_t hc_rows_bytes(const cere_qwen4_exp_config &cfg, uint64_t max_m);
// Workspace growth of a drafting plan: workspace_bytes(cfg, t, ctx, count) - workspace_bytes(cfg, t, ctx).
uint64_t workspace_bytes(const cere_qwen4_exp_config &cfg, uint64_t max_m, uint64_t max_context_tokens,
                         uint64_t count);

// INSTANTIATE of a drafting model (w->has_draft), after instantiate_model: the hot lm_head rows (gathered from the
// target lm_head at the device map, which ALLOCATE's upload_map filled) and the Workspace::Mtp plans. Synchronizes
// `stream`. CERE_ERR_ARGUMENT without a draft; plan errors as GemmPlan::init.
cere_status instantiate(ModelWeights *w, Workspace *ws, cublasLtHandle_t gemm, cudaStream_t stream);

// Copies the plan's draft token map to w->mtp.map (device) and sets w->mtp.count; synchronizes `stream`.
cere_status upload_map(ModelWeights *w, const std::vector<uint32_t> &map, cudaStream_t stream);

// Kernel-test stage sink (K14 bisection, m29 design note D3): called on `stream` with each named intermediate of the
// forward (device BF16 [rows, cols], valid until the sink returns): "fusion.e" [m, 2,560], "fusion.hyper",
// "attn.normed" [m, 10,240], "attn.mixed" [m, 2,560], "attn.qkv" [m, 13,312] (before the prologue), "attn.out"
// [m, 6,144] (the o_proj input), "attn.block_out" [m, 2,560], "attn.hyper", "mlp.normed" [m, 10,240], "mlp.mixed",
// "mlp.block_out" [m, 2,560], "mixer.y" [m, 2,560]. A non-OK status ends the forward with it.
using StageSink = cere_status (*)(void *ctx, const char *name, const bf16 *values, uint64_t rows, uint64_t cols,
                                  cudaStream_t stream);

struct MtpCall {
    uint32_t m = 0;                      // rows, 1 .. ws.max_tokens
    const uint64_t *positions = nullptr; // host [m], consecutive: positions[0] + r
    // device [m] = positions (the spec_graph draft graph, model.h draft_graph): the attention reads the rows'
    // positions here (attention_rows_at), so a captured forward replays at any position; null: positions[0] on the host
    const int64_t *device_positions = nullptr;
    const uint32_t *next_ids = nullptr;  // device [m]: the token at positions[r] + 1 (< vocab)
    const bf16 *hc_in = nullptr;         // device [m, 10,240]: target (or previous MTP step) pre-mixer hc rows
    bf16 *hc_out = nullptr;              // device [m, 10,240]: the MTP layer's pre-mixer hc rows (may alias hc_in)
    uint32_t *draft_ids = nullptr;       // device [m] (last_only: [1]): map[argmax hot logits] per row
    // The runtime's draft-extend and decode steps (SGLang's LogitsProcessor prunes an extend to its last row): the
    // mixer, the hot lm_head and the argmax run on row m - 1 only, into draft_ids[0]; without a stage sink the K13
    // routed experts too, so hc_out rows [0, m - 1) are unspecified (the callers read row m - 1 alone).
    bool last_only = false;
    // m210 D15⁗′ class pin: 0, or the row count (>= m) of the cold call these rows belong to; every row-count
    // choice of the forward (rows plans, K2 buckets and pins, HC fusion, attention kernel) is made for it.
    uint32_t cls = 0;
    // The rows' rotary inputs (model.h Rope): an image session's draft-extend over prompt rows passes the target
    // call's prompt_rope (ws.positions3 still holds that call's columns); a null table (every other call) rotates at
    // position + the session's rope_delta (rope_table). device_positions (the draft graph) runs only with a null table
    // and a session at rope_delta 0.
    Rope rope{};
    // PrefixCapture G1: the MTP layer's EndKv of a chunk-0 end state's call (model.h end_kv at the MTP index), with
    // positions[0] = 0 (capture) or its rows (resume); null for every other call.
    const EndKv *end_kv = nullptr;
    StageSink stage_sink = nullptr;      // kernel tests only: null in every runtime call
    void *stage_ctx = nullptr;
};

// The MTP forward of `call` for one session: writes the MTP layer's KV / QSA keys at the rows' positions in the
// session's MTP pool entry (through s.slot_map) and reads its positions [0, positions[0] + m). Leaves the hot logits
// of the last chunk of rows nowhere (scratch). Enqueued on `stream`, no synchronization. CERE_ERR_ARGUMENT for a model
// without draft, a session without an MTP entry, m outside [1, ws.max_tokens] or positions beyond the context;
// CERE_ERR_INTERNAL on a launch failure.
cere_status forward(const ModelWeights &w, Workspace &ws, SessionState &s, const MtpCall &call, cudaStream_t stream);

// Kernel-test hook: forward with the hot logits [m, count] BF16 of every row also copied to `logits` (device).
cere_status forward_logits(const ModelWeights &w, Workspace &ws, SessionState &s, const MtpCall &call, bf16 *logits,
                           cudaStream_t stream);

// The draft of count (2 .. min(kRows, ws.mtp.rows)) distinct drafting sessions at once (m23 design note P11R-21):
// session p's extend over ns[p] rows at p0s[p] .. p0s[p] + ns[p] - 1, its rows in next_ids (device [Σ ns]) and hc
// (device [Σ ns, 10,240], read only) after session p - 1's (Σ ns <= ws.mtp.rows), then layout::kDraftSteps decode
// steps from its last row. One forward per pass over every session's rows (each launch once at all rows on the rows
// plans, the attention once per session on its MTP pool entry; the extend's MoE and last combine on each session's
// last row alone): drafts (device [1 + kDraftSteps, count], row 0 untouched) gets session p's extend draft in row 1
// and decode step k's in row k + 1, bit-equal to the session's own extend (MtpCall last_only) and one-row decode
// steps. `last` (device [count, 10,240]) holds each session's hc row of the steps. Enqueued on `stream`, no
// synchronization; CERE_ERR_ARGUMENT for a model without draft, a session without an MTP entry or repeated, or a
// step beyond the context.
cere_status draft_rows(const ModelWeights &w, Workspace &ws, SessionState *const *sessions, const uint64_t *p0s,
                       const uint32_t *ns, uint32_t count, const uint32_t *next_ids, bf16 *hc, bf16 *last,
                       uint32_t *drafts, cudaStream_t stream);

}  // namespace cere::model::mtp
