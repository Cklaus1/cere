// K8 (m22 design note M2.2b): the internal C++ interface between the CUDA device (device/), the forward pass of the
// A4 model (Qwen3.8-Flash-Next NVFP4, arch qwen4_exp) and the runtime that drives it. Nothing here is ABI.
//
// Lifecycle and ownership (O14: nothing is allocated and no module is loaded after INSTANTIATE):
//   estimate / ALLOCATE  memory_plan.cpp sizes the device arena with workspace_bytes / derived_bytes /
//                        session_bytes (x max_sessions) / pool_bytes and the mapped host arena with staging_bytes,
//                        and places the tensors in arena_order(plan). CudaDevice::allocate takes both arenas in one
//                        allocation each (plus the M2.11 snapshot arena, its own locked host allocation outside the
//                        mapped one) and calls bind_weights and bind_pool (pointers only; no tensor byte is read).
//   UPLOAD               cere_model_put fills the tensors; nothing here runs.
//   INSTANTIATE          CudaDevice::instantiate calls instantiate_model (model/forward.cu): derived tensors (F32
//                        A_log, MoE alphas and global scales, the RoPE table) and every GEMM plan (from the
//                        device's cuBLASLt handle).
//                        Host-side objects (plans) are built here and never again.
//   create_session       the device reserves the session's KV pages (host/kv_pool.h), then bind_session on the
//                        slot's bytes, session_reset and session_set_pages.
//   step                 forward (one prefill chunk or one decode token); the last call of a step with sampling or
//                        logprobs runs the K12 sampler stage on its scored row (m27 design note I9).
//   step_rows            forward_decode_rows (one token per session), each row with sampling or logprobs followed
//                        by its own K12 sampler stage (ledger CR4).
//
// Streams: every function that takes a cudaStream_t enqueues on it only (one executor stream, Q18); the device never
// uses the legacy default stream for model work. Errors: CERE_ERR_ARGUMENT for a violated precondition (checked
// before any launch), CERE_ERR_UNSUPPORTED for a model outside the compiled shapes, CERE_ERR_INTERNAL for a CUDA or
// cuBLASLt failure. No function throws; std::bad_alloc inside a host-side function is CERE_ERR_CAPACITY.
//
// The shapes are the compiled A4 support table (host/support.cpp): hidden 2,560, 48 layers (full attention at
// 3, 7, ..., 47), hc 4 x 2,560, 512 experts top-10, PLE at 0-based layer 1. `cfg` arguments are a config the host
// layer validated against that table.
#pragma once

#include "cere_executor.h"
#include "cere_model_constants.h"
#include "host/device.h"
#include "kernels/k2_gemm.h"
#include "kernels/k3_hc.h"
#include "kernels/k4_ple.h"
#include "kernels/k5_gdn_layer.h"
#include "kernels/k7_moe.h"
#include "kernels/k9_qsa.h"
#include "kernels/k13_moe_bf16.h"
#include "model/vision.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cere::research {
class Staging;
class Call;
struct Binding;
}  // namespace cere::research

namespace cere::model {

using bf16 = __nv_bfloat16;

// The model's dimensions come from its descriptor (executor/models/qwen38-flash-next.json, SelfTune D1): the build
// generates cere_model_constants.h (namespace cere::gen) from it with executor/tools/gen_model_header.py.
constexpr uint32_t kLayers = gen::kNumHiddenLayers;
constexpr uint32_t kHidden = gen::kHiddenSize;
constexpr uint32_t kHc = gen::kHcCount;
constexpr uint32_t kHcHidden = kHc * kHidden;  // 10,240: the HC state width
constexpr uint32_t kQHeads = gen::kNumAttentionHeads;
constexpr uint32_t kKvHeads = gen::kNumKeyValueHeads;
constexpr uint32_t kHeadDim = gen::kHeadDim;
constexpr uint32_t kQGateWidth = kQHeads * 2 * kHeadDim;  // 12,288: q_proj rows (per head q then gate)
constexpr uint32_t kKvWidth = kKvHeads * kHeadDim;        // 512: k_proj / v_proj rows, FP8 KV bytes per token
constexpr uint32_t kQkvWidth = kQGateWidth + 2 * kKvWidth;  // 13,312: the packed q|k|v projection
constexpr uint32_t kAttnOutWidth = kQHeads * kHeadDim;      // 6,144: o_proj input
constexpr uint32_t kRotaryDim = gen::kRotaryDim;
constexpr uint32_t kGdnValueHeads = gen::kLinearNumValueHeads;
constexpr uint32_t kExperts = gen::kNumExperts;
constexpr uint32_t kTopK = gen::kNumExpertsPerTok;
constexpr uint32_t kMoeInter = gen::kMoeIntermediateSize;
constexpr uint32_t kSharedInter = gen::kSharedExpertIntermediateSize;
constexpr uint32_t kVocab = gen::kVocabSize;
constexpr uint32_t kMaskWords = (kVocab + 31) / 32;  // 7,760: one vocab mask, 31,040 B (m28 design note A1)
static_assert(sizeof(gen::kPleLayerIds) == sizeof(uint32_t), "one PLE layer (config ple_layer_ids)");
constexpr uint32_t kPleLayer = gen::kPleLayerIds[0] - 1;  // 0-based (config ple_layer_ids [2] is 1-based)
// Rows of one prefill chunk at most (Q8: chunked prefill 4,096; K6 limits M2.2b to 2,048 by context).
constexpr uint64_t kMaxChunkTokens = gen::kGraphMaxChunkTokens;

// ---- Arena placement (memory_plan) -------------------------------------------------------------------------------

// The order memory_plan places tensors in. Tensors of one run abut with no alignment padding (packed[i] = true:
// order[i] starts where order[i - 1] ends); a run's first tensor is aligned as memory_plan aligns any tensor. Runs:
//   * per layer L, projection P in {gate, up, down} and suffix S in {weight, weight_scale, weight_scale_2,
//     input_scale}: `layers.L.mlp.experts.E.P_proj.S` for E = 0 .. 511 in expert order, so K7 sees [E, ...] arrays
//     (`weight_scale_2` / `input_scale` become F32 [E]);
//   * per full-attention layer: self_attn q_proj, k_proj, v_proj, so q|k|v is one [13,312, 2,560] weight (one GEMM,
//     as SGLang's fused QKVParallelLinear); with MXFP8 weights (track Fp4S3, fp4local design note FL-30, the S3
//     artifact) their `weight_scale` tensors form a second run, one UE8M0 [13,312, 80] scale array;
//   * the 128 PLE table shards `ngram_embedding.shard_N.weight` in numeric shard order (host arena), so the table is
//     one [rows, 160] FP8 array (or, NVFP4 table of fp4local design note D8, one [rows, 80] packed U8 array);
//   * an NVFP4 table's 128 block-scale shards `ngram_embedding.shard_N.weight_scale` in numeric shard order (host
//     arena), one [rows, 10] E4M3 array (CUR's checkpoint has none).
// Every other non-DISCARD tensor is its own run; runs follow the desc order of their first member. DISCARD tensors
// are absent from `order`.
// P3b build (m23 design note M23-D2): the up and gate `weight` / `weight_scale` runs of a layer are one run
// `layers.L.mlp.experts.*.w13.S`, member 2E = up_proj of expert E, 2E + 1 = its gate_proj, so the experts hold
// FlashInfer's w13 = [up; gate] per expert in place (no second copy).
struct ArenaOrder {
    std::vector<uint64_t> order;  // tensor indices into plan.tensors
    std::vector<bool> packed;     // parallel to order
};

// Never fails except by std::bad_alloc (propagated: memory_plan's callers map it to CAPACITY).
ArenaOrder arena_order(const host::ModelPlan &plan);
#if defined(CERE_MOE_P3B)
// The K7 build's order of the same plan (no w13 run), for the P3b arena-size ktest (p3b/p3b_ktest.cpp).
ArenaOrder arena_order_k7(const host::ModelPlan &plan);
// The shared kernel scratch bytes (the Workspace's only MoE-dependent region) for forward calls of up to t rows in a
// context of ctx tokens: out[0] this (P3b) build's, out[1] with K7's MoE workspace instead.
void scratch_bytes_p3b_k7(const cere_qwen4_exp_config &cfg, uint64_t t, uint64_t ctx, uint64_t out[2]);
#endif

// ---- Sizing (memory_plan) ----------------------------------------------------------------------------------------
// Pure arithmetic over a validated config; no CUDA call. Each region is a multiple of 256 bytes and is placed on a
// 256-byte boundary by memory_plan.

// Rows of the largest forward call: min(max_context_tokens, kMaxChunkTokens).
uint64_t workspace_rows(uint64_t max_context_tokens);
// Device bytes of the Workspace for forward calls of up to max_tokens rows in a context of max_context_tokens
// (activation rows, the shared kernel scratch = the largest K3/K4/K5/K6/K7 workspace joined by the K6 prologue + K9
// QSA scratch when the context exceeds k6::kMaxContext (m22c D5) and by the K12 sampler stage's buffers (m27 I9), one
// logits row, the F32 processed-logits row (m27 I9), top-2 rows, token ids).
// M2.9 (M29S2-12): with draft_count > 0 (a plan with drafts and a token map of that many ids) the shared scratch
// also covers mtp::scratch_bytes (the MTP forward aliases it after the target's call) and Workspace::hc_rows follows
// the gemm workspace; draft_count 0 is the plan without drafts, byte for byte.
uint64_t workspace_bytes(const cere_qwen4_exp_config &cfg, uint64_t max_tokens, uint64_t max_context_tokens,
                         uint64_t draft_count = 0);
// Mapped pinned host bytes of the Workspace's staging (token ids in, top-2 out, and staging_logprobs_rows entries of
// cere_token_logprobs out, one per scored row of a batched decode step) for max_tokens rows.
uint64_t staging_bytes(const cere_qwen4_exp_config &cfg, uint64_t max_tokens);
// The cere_token_logprobs entries of the staging: min(max_tokens, kernels::GemmPlan::kMaxRows).
uint64_t staging_logprobs_rows(uint64_t max_tokens);
// Device bytes of the K12 sampler stage's buffers (SamplerScratch), carved from the start of the shared scratch: one
// per model whatever the rows of a batched decode step (ledger W / CR4: the rows' stages run one after another).
size_t sampler_scratch_bytes(const cere_qwen4_exp_config &cfg);
// Device bytes of the model's derived tensors: F32 A_log per GDN layer, F32 g1/g2 alphas per layer, the F32 RoPE
// table [max_context_tokens, 64].
uint64_t derived_bytes(const cere_qwen4_exp_config &cfg, uint64_t max_context_tokens);
// Device bytes of one session slot for up to max_context_tokens: the slot map int64 [ctx] (m25 design note D3),
// GDN conv + SSM state per GDN layer (BF16), the QSA ring BF16 [4, 128] per full-attention layer (m22c D5), PLE
// history (2 x int64) and conv state (BF16 [10,240, 9]), the sampler's output counts u32 [vocab] (m27 I9).
// With `draft` (M2.9, M29S2-04/-19) the MTP QSA ring BF16 [4, 128] and the pending proposals u32 [4] follow.
uint64_t session_bytes(const cere_qwen4_exp_config &cfg, uint64_t max_context_tokens, bool draft = false);
// Device bytes of the paged KV pool of `pool_tokens` tokens (m25 D2): FP8 K and V [tokens, 512] and the QSA
// compressed keys BF16 [tokens / 4, 128] per full-attention layer.
// With `draft` (M2.9, M29S2-04) the MTP layer's entry follows: 1,088 B per token.
uint64_t pool_bytes(const cere_qwen4_exp_config &cfg, uint64_t pool_tokens, bool draft = false);
// M2.10 (m210 design note D6′, ledger M4): bytes of one fixed-state slot, the contiguous run of a session slot from
// the first GDN layer's conv state to out_counts (layout::fixed_state; 59,031,808 B on A4). 0 when the carving is not
// one run (never on the compiled support table).
uint64_t fixed_state_bytes(const cere_qwen4_exp_config &cfg);

// ---- Weights -----------------------------------------------------------------------------------------------------

// One GatedResidual (hyper-connection) in the checkpoint layout: hc_norm [10,240], input_mix_weight_down
// [320, 10,240], input_mix_weight_up [10,240, 320], block_inject_weight [4, 10,240] (null for the final mixer).
struct HcWeights {
    const bf16 *norm = nullptr;
    const bf16 *down = nullptr;
    const bf16 *up = nullptr;
    const bf16 *inject = nullptr;
};

// The full-attention mixer of one layer: qkv = q_proj | k_proj | v_proj packed [13,312, 2,560] (arena_order),
// q_norm / k_norm [256] (Gemma), o_proj [2,560, 6,144]; the QSA indexer (m22c F7): index_qk_proj
// [(4 + 1) * 128, 2,560] (q rows then k rows), q_layernorm / k_layernorm [128] (Gemma).
struct AttentionWeights {
    const bf16 *qkv = nullptr;
    const bf16 *q_norm = nullptr;
    const bf16 *k_norm = nullptr;
    const bf16 *o_proj = nullptr;
    const bf16 *indexer_qk = nullptr;
    const bf16 *indexer_q_norm = nullptr;
    const bf16 *indexer_k_norm = nullptr;
};

// Checkpoint sources of the MoE derived values (F32 [512] each, packed by arena_order).
struct MoeScaleSources {
    const float *gate_input_scale = nullptr;
    const float *up_input_scale = nullptr;
    const float *down_input_scale = nullptr;
    const float *gate_weight_scale_2 = nullptr;
    const float *down_weight_scale_2 = nullptr;
};

struct LayerWeights {
    bool full_attention = false;  // bit L of cfg.full_attention_layers
    HcWeights attn_hc;            // layers.L.attn_hyper_connection.*
    HcWeights mlp_hc;             // layers.L.mlp_hyper_connection.*
    AttentionWeights attn;        // full-attention layers only (else all null)
    k5::LayerWeights gdn{};       // GDN layers only (else all null); A_log points at the derived F32 copy
    const bf16 *gdn_A_log_bf16 = nullptr;  // the checkpoint A_log [48] (source of gdn.A_log)
    kernels::MoeLayer moe;        // g1/g2 alphas point into the derived region; a1/a2 set by instantiate_model
    MoeScaleSources moe_scales;
};

// M2.9 (m29 design note D3; model/mtp.h runs it): the MTP draft layer, the 31 `mtp.*` checkpoint tensors (DEVICE
// when the plan has drafts, else unbound) and its derived hot lm_head. The fusion: fc_embedding / fc_hidden
// [2,560, 2,560], pre_fc_norm_embedding [2,560] and pre_fc_norm_hidden [10,240] (Gemma); one full-attention decoder
// layer (mtp.layers.0: q|k|v packed by arena_order as the target's) with a BF16 MoE (K13); the final mixer
// mtp.hyper_connection_mixer (inject null). The draft lm_head is derived (layout::carve_mtp_derived).
struct MtpWeights {
    const bf16 *fc_embedding = nullptr;
    const bf16 *fc_hidden = nullptr;
    const bf16 *norm_embedding = nullptr;
    const bf16 *norm_hidden = nullptr;
    HcWeights attn_hc;
    HcWeights mlp_hc;
    HcWeights mixer;
    AttentionWeights attn;
    kernels::MoeBf16Layer moe;
    bf16 *lm_head = nullptr;    // derived BF16 [count, 2,560]: lm_head rows at map (mtp::instantiate)
    uint32_t *map = nullptr;    // derived u32 [count]: the draft token map (copied at ALLOCATE, mtp::upload_map)
    uint32_t count = 0;         // draft_token_map entries
};

// Typed device views of every tensor the forward reads, bound by name. HOST_PINNED tensors (the PLE table, its
// scale and index) are bound through the device alias of the mapped host arena (zero-copy).
struct ModelWeights {
    cere_qwen4_exp_config config{};
    uint64_t max_context_tokens = 0;
    const bf16 *embed = nullptr;    // embed_tokens [248,320, 2,560]
    const bf16 *lm_head = nullptr;  // lm_head [248,320, 2,560]
    HcWeights final_mixer;          // hyper_connection_mixer.* (inject null): replaces the final norm
    LayerWeights layers[kLayers];
    k4::Weights ple{};              // layer kPleLayer's PLE (table = 128 packed shards, table_rows = their sum)
    float *rope_cos_sin = nullptr;  // derived F32 [max_context_tokens, 64] (k1::rope_table, theta = cfg.rope_theta)
    uint8_t *derived = nullptr;     // the derived region (derived_bytes)
    // The vision tower (m213 design note D10): bound only when the plan's support entry has a vision config.
    vision::Weights vision;
    // M2.9: bound only when the plan has drafts (speculative_draft_tokens > 0); else has_draft false and mtp empty.
    bool has_draft = false;
    MtpWeights mtp;
    // Track Fp4S3 (fp4local design note FL-30, kernels/k2_mxfp8.h): the target GEMMs whose weights the artifact
    // stores as MXFP8 (the S3 FP4-Local artifact: the 384 dense projections, q|k|v one entry), keyed by the E4M3
    // pointer the layer views above hold in their bf16 fields; empty for BF16 artifacts (CUR, S2). Its staging is the
    // device arena's MXFP8 staging region (memory_plan.h, bind_mxfp8).
    kernels::k2m::Table mxfp8;
};

// Where a model's tensors live (CudaModel's arenas after ALLOCATE).
struct ArenaView {
    const host::ModelPlan *plan = nullptr;
    const uint64_t *offsets = nullptr;   // per tensor index: arena offset, UINT64_MAX for DISCARD
    uint8_t *device = nullptr;           // device arena base
    uint8_t *host_device = nullptr;      // device alias of the mapped host arena base
    uint8_t *derived = nullptr;          // device: the derived region (derived_bytes)
};

// Binds every tensor of ModelWeights by its checkpoint name (`model.language_model.` prefix, `lm_head.weight`, and the
// 333 `model.visual.*` tensors of model/vision.h when the plan's support entry has a vision config) via a
// name -> tensor index map over plan.tensors. Checks dtype, shape and placement of each, and that every arena_order
// run is contiguous (experts, q|k|v, PLE shards). Pointers only: no tensor byte is read, no CUDA call is made, so it
// runs at ALLOCATE (before UPLOAD). CERE_ERR_UNSUPPORTED with *error naming the first missing or mismatched tensor
// (e.g. "missing tensor model.language_model.layers.7.self_attn.q_norm.weight"); `out` is then unspecified.
// Track Fp4S3 (FL-30): each of the 384 target dense projections (attention q/k/v/o and index_qk_proj, GDN in_proj_*
// and out_proj, shared expert gate/up/down) binds as BF16 or as MXFP8 (model/mxfp8_plan.h: F8_E4M3 weight plus U8
// `weight_scale` [n, k / 32]); the MXFP8 ones go into out->mxfp8 (q|k|v one entry: both runs contiguous) and their
// views hold the E4M3 pointer. Mixed formats inside q|k|v, the GDN in-projections or the shared expert, and a scale of
// the wrong dtype / shape / placement are UNSUPPORTED (mxfp8::plan_of's text). The MTP layer's experts bind as BF16
// `gate_up_proj` / `down_proj`, or as the S3 artifact's NVFP4 W4A16 tensors (U8 `gate_up_proj` with `_scale` /
// `_scale_2`, kernels/k13_moe_bf16.h); its other tensors stay BF16.
cere_status bind_weights(const ArenaView &arena, ModelWeights *out, std::string *error);

// Device bytes of the encoder scratch of a vision plan of max_context_tokens: vision::scratch_bytes of the largest
// image (layout::max_image_patches). The memory plan sizes the workspace region to at least this; the encoder's
// activations alias the whole region (the encoder finishes before the forward's first use of the workspace).
uint64_t vision_scratch_bytes(const cere_qwen4_exp_vision_config &v, uint64_t max_context_tokens);

// The K12 sampler stage's device buffers (m27 design note I9), carved from the start of the shared scratch: the
// stage runs after every layer, so it never overlaps a layer kernel's use of that scratch.
struct SamplerScratch {
    float *probs = nullptr;                   // [vocab] fp32 probabilities of processed / T
    uint32_t *keep = nullptr;                 // [ceil(vocab / 32)] the D3 keep mask
    double *lse = nullptr;                    // [1]
    void *scratch = nullptr;                  // k12::scratch_bytes(): probabilities, keep_mask and the draw
    uint32_t *top2_ids = nullptr;             // [2] top2_f32 of the processed row
    float *top2_scores = nullptr;             // [2]
    uint32_t *drawn = nullptr;                // [1]
    cere_logit_bias *bias = nullptr;          // [vocab] the item's logit_bias, uploaded per sampled step
    cere_token_logprobs *logprobs = nullptr;  // [1] the D7 output before its copy to the host staging
    uint32_t *mask = nullptr;                 // [ceil(vocab / 32)] a constrained item's vocab mask (m28 D9), uploaded
                                              // per constrained step (rows' stages run one after another)
};

// M2.9 phase S (m29 design note D6, M29S2-13/-14): the sampler stage of every verify row of one forward_verify_rows
// call, kept per row until the chain's accept decision and its committed tokens' scores. Carved only for plans with
// drafts, in the device's draft-only region (spec_sampler_bytes; a spec-off arena has no such bytes, A1), rows =
// the verify call's rows (Workspace::max_rows of a drafting model).
struct SpecSampler {
    uint32_t rows = 0;               // 0 = not bound (no drafts)
    float *proc = nullptr;           // [rows, vocab] fp32 processed logits of verify row r (k12::process)
    float *probs = nullptr;          // [rows, vocab] fp32 probabilities of proc / T (temperature > 0 rows)
    uint32_t *keep = nullptr;        // [rows, ceil(vocab / 32)] the D3 keep masks
    uint32_t *top2_ids = nullptr;    // [rows, 2] top2_f32 of each processed row
    float *top2_scores = nullptr;    // [rows, 2]
    uint32_t *committed = nullptr;   // [rows] k13s::spec_accept's committed ids, chains at their first row
    uint32_t *count = nullptr;       // [rows] spec_accept's count, entry c for chain c
    uint32_t *counts = nullptr;      // [vocab] a penalized chain's out_counts plus its drafts so far (row j's counts)
    void *accept = nullptr;          // k13s::scratch_bytes(3)
};

// ---- Workspace ---------------------------------------------------------------------------------------------------

// The rows plan set of a batched decode step (m26 design note D6, I6): every GEMM forward_decode_rows runs at m = R
// (R = the model's max_sessions), each made by GemmPlan::init_rows from the Workspace's plan of the same shape, so a
// row of an m-row call runs the single-row decode's algorithm. Built by instantiate_model only when R > 1 (`rows` 0
// otherwise); the prefill / single-row plans above are never touched. PLE (k4::forward_rows, rows16 design note
// R16-A) runs its key / value GEMMs at m on ple_key / ple_value; the QSA indexer projection (inside the per-row
// attention) runs per row at m = 1 on the single-row plan (D5). The lm_head of an m-row call (a decode
// step's R rows, a verify call's Σ(n + 1) rows) runs once at m into Workspace::rows_logits on `lm_head`, the rows plan
// of Workspace::lm_head (m23 design note P11R-21: one weight read instead of m), bit-equal per row to the m = 1 call.
struct RowsPlans {
    uint32_t rows = 0;               // R; 0 = not built
    kernels::HcPlans hc;             // mix down / up
    kernels::GemmPlan attn_qkv;      // of Workspace::attn_qkv (its m <= 16 never reaches a pin range)
    kernels::GemmPlan attn_o;
    kernels::GemmPlan gdn_qkv;
    kernels::GemmPlan gdn_z;
    kernels::GemmPlan gdn_ba;
    kernels::GemmPlan gdn_out;
    kernels::GemmPlan ple_key;       // of Workspace::ple_key / ple_value (k4::forward_rows)
    kernels::GemmPlan ple_value;
    kernels::GemmPlan moe_router;    // with moe_shared_gate_up and moe_shared_down: moe_block's rows MoePlans
    kernels::GemmPlan moe_shared_gate_up;
    kernels::GemmPlan moe_shared_down;
    kernels::GemmPlan lm_head;       // of Workspace::lm_head, rows min(R, max_tokens) (Workspace::rows_logits' rows)
    // D6 MoE fallback: true runs moe_block once per row at m = 1 on the single-row plans (the solo call) instead of
    // once at m = R on the rows MoePlans (forward.cpp kMoeRowsPerRow, logged at INSTANTIATE; K2
    // test_moe_rows_invariant gates the fused path).
    bool moe_per_row = false;
};

// The decode graphs of one model (m23 design note P1, M23P1-01..07): a CUDA graph of an eligible DECODE step (the
// `graph` conditions of forward / forward_decode_rows), captured lazily on the first such step of its key and
// replayed after. Entry 0 is forward's single-row step (run_layers), entry r >= 1 forward_decode_rows' r-row step
// (run_rows); each graph is the step's H2D copies of ids (ids64 carrying the rows' positions in its tail,
// layout::decode_graph_fits), its launches and the D2H copy of top-2, so the host scalars of a step reach the graph
// only through the mapped staging. The key is what the captured launches bake in: per row the session's slot pointer
// and max_context_tokens (every per-session device pointer derives from the two for the session's life: bind_session
// carves the slot by the context, m25 D3) and its decode_dense bit. A step whose key differs recaptures and updates
// the executable in place (cudaGraphExecUpdate), instantiating anew when the topology changed.
//
// decode_step = spec_graph (m29 design note M29S2-18b, m23 design note P11R-20) adds the verify graphs: entry m
// of `verify` is forward_verify_rows' m-row VERIFY call (run_verify) of greedy chains (layout::spec_graph_chain_fits),
// with each row's position p + j in ids64's tail (uploaded with the ids, both stream-ordered ahead of the graph, and
// the top-2 download after it: a copy node at a graph's head delays its first kernel) and the undo copies inside (as
// k1::copy_segments kernels, the compressed-key one through k9::qsa_key_save_step). Its key: the verify_attention
// lever, then per chain the session's slot pointer, max_context_tokens and draft count n, then per chain its undo
// pointer (DraftAccept: the session's own snapshots, or a range of the model's pool in a call group with a chain longer
// than kUndoSnapshots). Entry `rows` of `draft` is one session's MTP draft (draft_graph: the extend of `rows` rows and
// the kDraftSteps decode steps at the device positions of ids64's head, layout::spec_draft_graph_fits); its key: the
// slot pointer, max_context_tokens, the hc rows, next ids and proposals pointers. A failed capture runs the same
// launches eagerly. A verify graph is kGraphSegments executables cut by graph_split in run_verify (after layer 0, then
// every kVerifySegmentLayers layers): a launch submits every node before the GPU starts it (~0.3 us per node, ~0.7 ms
// for the ~2,400 nodes of an m = 1 verify call, P11R-20), so the GPU starts on the one-layer head while the host
// launches the rest, a segment ahead. A draft graph is one segment per MTP forward (the extend, then each decode
// step); a decode graph is one segment.
constexpr uint32_t kVerifySegmentLayers = gen::kGraphVerifySegmentLayers;
constexpr uint32_t kGraphSegments = kLayers / kVerifySegmentLayers + 1;
static_assert(kLayers % kVerifySegmentLayers == 0, "verify graph segments split the layers evenly");
struct DecodeGraph {
    cudaGraphExec_t exec[kGraphSegments] = {};  // launched in order; [0, segments) are live
    uint32_t segments = 0;                      // 0: nothing captured
    uint64_t key[1 + 4 * kernels::GemmPlan::kMaxRows] = {};
    // the first capture printed `cere: decode_graph_capture_ms=<ms> rows=<r>` (a verify graph: spec_graph_capture_ms)
    bool logged = false;
    // the last instantiation failed for `key` (DM-5): steps of that key run eagerly instead of capturing and failing
    // again every step; another key, or a successful capture, clears it. Logged once as `cere: <what>_instantiate_failed`.
    bool failed = false;
};
struct GraphCounts {
    uint64_t captures = 0;  // graphs instantiated
    uint64_t updates = 0;   // executables updated in place for a new key
    uint64_t replays = 0;   // steps run by a graph
};
struct DecodeGraphs {
    // the decode_step lever (apply_levers): CERE_LEVER_DECODE_GRAPH or CERE_LEVER_DECODE_SPEC_GRAPH; off: every
    // step is eager
    bool enabled = false;
    bool spec = false;  // CERE_LEVER_DECODE_SPEC_GRAPH: eligible VERIFY calls and MTP drafts run through graphs
    DecodeGraph graph[kernels::GemmPlan::kMaxRows + 1];
    DecodeGraph verify[kernels::GemmPlan::kMaxRows + 1];
    DecodeGraph draft[kernels::GemmPlan::kMaxRows + 1];
    GraphCounts counts;         // of `graph`
    GraphCounts verify_counts;  // of `verify`
    GraphCounts draft_counts;   // of `draft`
    // A capture in progress (graph_step): graph_split closes the captured segment into closed[closed_n++] and
    // captures the next; outside a capture it does nothing.
    bool capturing = false;
    cudaGraph_t closed[kGraphSegments] = {};
    uint32_t closed_n = 0;
};

// The MoE shared-expert fork of a workspace (m23 design note P11R-23; kernels/k7_moe.h MoePlans side / fork /
// join): a non-blocking stream and two timing-free events of the model's device, created by instantiate_model
// (create keeps handles it already holds) and destroyed with the workspace. Every target-layer MoE block (K7 / P3b)
// of m <= 16 rows without hook points runs its shared expert on `side`, concurrently with the router and the routed
// experts (the MTP layer's BF16 block, K13, stays serial).
struct MoeFork {
    cudaStream_t side = nullptr;
    cudaEvent_t fork = nullptr;
    cudaEvent_t join = nullptr;
    MoeFork() = default;
    MoeFork(const MoeFork &) = delete;
    MoeFork &operator=(const MoeFork &) = delete;
    ~MoeFork();
    cere_status create();
    // The MoePlans of these GEMM plans (row class `cls`, MoePlans::cls) with this fork.
    kernels::MoePlans plans(const kernels::GemmPlan *router, const kernels::GemmPlan *shared_gate_up,
                            const kernels::GemmPlan *shared_down, uint32_t cls = 0) const {
        return kernels::MoePlans{router, shared_gate_up, shared_down, cls, side, fork, join};
    }
};

// Preallocated scratch of one model for forward calls of 1 .. max_tokens rows (one prefill chunk, or one decode
// token using row 0, or one batched decode step of up to max_rows sessions using rows 0 .. R - 1). One Workspace per
// model: forward calls of one model are serialized. Every buffer is row-major, rows first, 256-byte aligned.
struct Workspace {
    uint32_t max_tokens = 0;
    uint32_t max_rows = 1;  // the model's max_sessions (set by the device before instantiate_model); R of `rows`
    // device (carved from the device arena's workspace region)
    bf16 *hyper = nullptr;       // [T, 10,240] the HC streams; updated in place by every combine and the PLE add
    bf16 *normed = nullptr;      // [T, 10,240] hc_mix's normalized streams, kept for hc_combine
    bf16 *mixed = nullptr;       // [T, 2,560] block input (embedding output first, final hidden last)
    bf16 *block_out = nullptr;   // [T, 2,560] attention / GDN / MoE output before combine
    bf16 *qkv = nullptr;         // [T, 13,312] packed q|gate, k, v projection
    bf16 *attn_out = nullptr;    // [T, 6,144] gated attention output (o_proj input)
    bf16 *tap_routed = nullptr;  // [T, 2,560] MoeTaps.routed (written only when taps are requested)
    bf16 *tap_shared = nullptr;  // [T, 2,560] MoeTaps.shared
    int32_t *ids32 = nullptr;    // [T] token ids for the embedding gather
    int64_t *ids64 = nullptr;    // [T] token ids for PLE; a decode graph's row positions follow the R ids (P1)
    bf16 *logits = nullptr;      // [1, 248,320] one lm_head row (lm_head_last_row, m = 1 plan)
    float *proc_logits = nullptr;  // [1, 248,320] the processed row of the sampler stage (k12::process, m27 I9)
    cere_token_score *top2 = nullptr;  // [T] top-2 per scored row
    void *scratch = nullptr;     // shared kernel workspace (K3 mix, K4, K5, K6, K7, K9, K12; one call at a time)
    size_t scratch_bytes = 0;
    SamplerScratch sampler;      // aliases the start of `scratch`
    // [min(T, 16), 248,320] the lm_head rows of a batched decode step or a verify call (one m-row GEMM on
    // RowsPlans::lm_head, m23 design note P11, P11R-21): inside `scratch` right after `sampler`, idle once the final
    // mix has run (no memory delta: it lies inside `scratch`).
    bf16 *rows_logits = nullptr;
    void *gemm_workspace = nullptr;  // split-K partials of moe_shared_gate_up and attn_qkv (k2_gemm.h gemm_pins_for)
    size_t gemm_workspace_bytes = 0;
    uint64_t max_context_tokens = 0;  // the context the scratch is sized for (K9 above k6::kMaxContext)
    // mapped pinned host staging (carved from the host arena's staging region)
    int32_t *host_ids32 = nullptr;          // [T]
    int64_t *host_ids64 = nullptr;          // [T]
    cere_token_score *host_top2 = nullptr;  // [T]
    cere_token_logprobs *host_logprobs = nullptr;  // [staging_logprobs_rows(T)], entry r for row r
    // The K9 kernels' bound fault word (k9::Bounds::fault, audit KernelsAttention F3): zeroed at bind, set by a
    // kernel that clamped a device-resident position or rotary row, read and cleared by take_bound_fault.
    uint32_t *bound_fault = nullptr;
    // GEMM plans, created by instantiate_model (max_m = max_tokens unless noted); host objects.
    kernels::HcPlans hc;            // mix down / up, shared by every GatedResidual and the final mixer
    kernels::GemmPlan attn_qkv;     // n 13,312, k 2,560, BF16, gemm_pins_for's pins at m >= 17 (gemm_workspace)
    kernels::GemmPlan attn_o;       // n 2,560, k 6,144
    kernels::GemmPlan attn_index_qk;  // n 640, k 2,560: QSA index_qk_proj (m22c D2, contexts above 2,048)
    kernels::GemmPlan gdn_qkv;      // n 10,240, k 2,560
    kernels::GemmPlan gdn_z;        // n 6,144, k 2,560
    kernels::GemmPlan gdn_ba;       // n 48, k 2,560
    kernels::GemmPlan gdn_out;      // n 2,560, k 6,144
    kernels::GemmPlan ple_key;      // n 10,240, k 2,560
    kernels::GemmPlan ple_value;    // n 2,560, k 2,560
    kernels::GemmPlan moe_router;   // n 512, k 2,560
    kernels::GemmPlan moe_shared_gate_up;  // n 640, k 2,560, gemm_pins_for's pins at m >= 17 (gemm_workspace)
    kernels::GemmPlan moe_shared_down;     // n 2,560, k 640
    kernels::GemmPlan lm_head;      // n 248,320, k 2,560, BF16 logits, max_m 1
    // Images (m213 design note D10-D12, m213l design note VL-L2/L3): set by the device for a vision plan (else
    // null / 0 / empty plans). One image at a time: forward encodes each image a call overlaps into image_out and
    // copies the rows of that overlap to image_embeds at their call row.
    uint8_t *region = nullptr;        // the workspace region's base: the encoder scratch (aliases every buffer above)
    uint64_t region_bytes = 0;        // >= vision_scratch_bytes
    bf16 *image_out = nullptr;        // device [4,096, 2,560]: one image's merged rows (vision::encode's output)
    bf16 *image_embeds = nullptr;     // device [4,096, 2,560]: row r = the merged row spliced over the call's row r
    bf16 *image_patches = nullptr;    // device [16,384, 1,536]: one image's pixel patches
    uint8_t *host_image_patches = nullptr;  // mapped pinned host copy of image_patches (image_patch_bytes)
    uint64_t image_patch_bytes = 0;
    // device int64: one call's mRoPE positions, K1/K6's [3, n] then K9's [3, prefix % 4 + n] (prompt_rope)
    int64_t *positions3 = nullptr;
    int64_t *host_positions3 = nullptr;     // mapped pinned host copy of positions3
    uint32_t max_image_patches = 0;   // the encoder plans' max_m (0: no vision)
    vision::Plans vision;             // the encoder's seven F32 plans (instantiate_model)
    RowsPlans rows;                 // built when max_rows > 1 (m26 design note D6)
    // The prefill_attention lever (cere_levers, set by apply_levers): true routes K6/K9 calls of >= p4::kMinRows rows
    // (by their class pin, so a hot tail of a longer cold call too) to the P4 tensor-core kernel
    // (kernels/p4_attention.h). False (CERE_LEVER_PREFILL_CUDA_CORE) is the default.
    bool prefill_p4 = false;
    // The verify_attention lever (P11R-19, set by apply_levers): true runs a verify chain's full-attention rows as one
    // K6 prologue and one k6::dense_attention_rows launch (forward.cpp attention_chain), bit-identical to the per-row
    // calls; false (CERE_LEVER_VERIFY_PER_ROW) is the default. verify_chains counts the attention_chain calls (one per
    // batched chain and full-attention layer; ktest cere_ktest_verify_attention reads it).
    bool verify_batched = false;
    uint64_t verify_chains = 0;
    // The k2_weights lever (track Fp8W): the weights convert_fp8_weights converted at INSTANTIATE (empty unless the
    // model was created with CERE_LEVER_K2_WEIGHTS_FP8); apply_levers hands it to every plan (GemmPlan::use_fp8).
    kernels::k2f::Table fp8;
    // Track Fp4S3 (FL-30): the model's MXFP8 weights (ModelWeights::mxfp8, bind_mxfp8; null for a BF16 artifact);
    // apply_levers hands it to every plan (GemmPlan::use_mxfp8; the MTP layer's BF16 weights miss it).
    const kernels::k2m::Table *mxfp8 = nullptr;
    // Track LoRA (kernels/lora.h): the model's adapter pool (CudaModel::lora_reserve; null: none); apply_levers hands
    // it to every plan (GemmPlan::use_lora). forward / forward_decode_rows / forward_verify_rows activate it for their
    // sessions' SessionState::lora_slot and run eager (no decode or verify graph) while any row has an adapter.
    kernels::lora::Table *lora = nullptr;
    // WeightLab research (research/hook.h): the model's capture staging and plan memory (CudaDevice::research_reserve;
    // null: none). forward / forward_decode_rows / forward_verify_rows build a research hook for a call only when some
    // row's session has a plan (SessionState::research), point `research_call` at it for that call (null otherwise:
    // today's code path), and run such a call eager (no decode or verify graph).
    research::Staging *research = nullptr;
    research::Call *research_call = nullptr;
    MoeFork moe_fork;               // instantiate_model; MoePlans of every MoE block (MoeFork::plans)
    DecodeGraphs graphs;            // host handles; release_decode_graphs before the device memory goes
    // M2.9 (plans with drafts only, else null): the hc rows the MTP forward reads (model/mtp.h): the pre-final-mixer HC
    // rows of a prefill chunk or of the last forward_verify_rows call (first Σ rows), device BF16 [max_tokens, 10,240]
    // (layout::mtp_hc_rows_bytes), after the gemm workspace.
    bf16 *hc_rows = nullptr;
    // M2.9 phase S (plans with drafts only, else rows 0): the verify rows' sampler stages (SpecSampler).
    SpecSampler spec;
    // M2.9 MTP plans (mtp::instantiate; empty without drafts). The MTP layer also runs the target's attn_qkv, attn_o,
    // attn_index_qk, hc, moe_router, moe_shared_gate_up and moe_shared_down plans for m > 16 (same shapes).
    struct Mtp {
        kernels::GemmPlan fc;         // n 2,560, k 2,560, max_m 4 x max_tokens (fc_hidden runs on the 4 streams)
        kernels::GemmPlan lm_head;    // n count, k 2,560, max_m mtp::kLmChunk rows (hot logits in scratch chunks)
        // Rows plans (GemmPlan::init_rows) for m <= rows = min(16, max_tokens): row r is bit-equal at any m (fc_rows
        // runs fc_hidden's 4 m stream rows in chunks of up to 16).
        uint32_t rows = 0;
        kernels::HcPlans hc_rows;
        kernels::GemmPlan fc_rows;
        kernels::GemmPlan attn_qkv_rows;
        kernels::GemmPlan attn_o_rows;
        kernels::GemmPlan router_rows;
        kernels::GemmPlan shared_gate_up_rows;
        kernels::GemmPlan shared_down_rows;
        kernels::GemmPlan lm_head_rows;
    } mtp;
};

// Carves `ws`'s buffers from the device region (workspace_bytes(cfg, max_tokens, max_context_tokens) bytes) and the
// host staging region (staging_bytes, mapped pinned: host pointer). Pointers only; plans are left empty for
// instantiate_model.
cere_status bind_workspace(const cere_qwen4_exp_config &cfg, uint64_t max_tokens, uint64_t max_context_tokens,
                           uint8_t *device_region, uint8_t *host_staging, Workspace *ws, uint64_t draft_count = 0);

// Track Fp4S3 (fp4local design note FL-30): after bind_weights and bind_workspace, points weights->mxfp8 at its
// dequantize staging (device, 256-byte aligned: the arena's MXFP8 staging region, memory_plan.h, of
// mxfp8::plan_of(plan).staging_bytes) and ws->mxfp8 at the table (null when it is empty: BF16 artifacts, whose plan
// has no such region). CERE_ERR_INTERNAL when the region is shorter than the table needs (the plan arithmetic and the
// binder disagree), CERE_ERR_ARGUMENT for null pointers.
cere_status bind_mxfp8(ModelWeights *weights, bf16 *staging, uint64_t staging_bytes, Workspace *ws);

// Device bytes of SpecSampler for `rows` verify rows (256-byte aligned buffers); bind_spec_sampler carves them from
// `base` (the draft-only region's spec part) into ws->spec.
uint64_t spec_sampler_bytes(const cere_qwen4_exp_config &cfg, uint64_t rows);
void bind_spec_sampler(const cere_qwen4_exp_config &cfg, uint64_t rows, uint8_t *base, Workspace *ws);

// INSTANTIATE (every tensor uploaded): computes the derived tensors into weights->derived (A_log F32, g1/g2 alphas,
// a1/a2 global scales into weights->layers[L].moe, the RoPE table) and creates every plan of `ws` from `gemm`
// (the device's cuBLASLt handle). May copy small scale arrays to the host and synchronizes `stream` before return.
// The only function here that builds host objects; after it returns, forward allocates nothing. On a model with
// MXFP8 weights it also readies the W8A8 algorithms (GemmPlan::init_a8) of the prefill plans of an MXFP8 weight shape
// (k2-prefill design note P3; a plan cuBLASLt serves none for keeps the staging path).
cere_status instantiate_model(ModelWeights *weights, Workspace *ws, cublasLtHandle_t gemm, cudaStream_t stream);

// The k2_weights lever CERE_LEVER_K2_WEIGHTS_FP8 (track Fp8W), after instantiate_model and mtp::instantiate (the
// MTP draft lm_head is gathered from the BF16 lm_head): converts in place (kernels::k2f::convert_in_place, through
// ws->scratch) the dense projections the target's plans run (attention qkv / o_proj / indexer_qk, GDN qkv / z / b /
// a / out_proj, the shared expert gate / up / down, lm_head; router, hc and every MTP weight stay BF16), lists them
// in ws->fp8 and points its staging at lm_head's idle tail. Synchronizes `stream`. Once it started the BF16 bytes are
// gone (ws->fp8.started), so a failure leaves the model unusable. No allocation.
cere_status convert_fp8_weights(ModelWeights *weights, Workspace *ws, cudaStream_t stream);

// ABI v5 selection (host/device.h cere::host::Levers, already validated by the host): switches every GEMM plan of
// `ws` (the prefill/single-row plans, lm_head, the vision plans, the rows plans and the MTP plans; each HcPlans' down
// and up) to the small-M kernel iff k2_gemm is CERE_LEVER_K2_SMALL_M or CERE_LEVER_K2_SMALL_M_A8
// (GemmPlan::use_small_m),
// and onto the W8A8 path of the plans instantiate_model readied iff it is CERE_LEVER_K2_SMALL_M_A8 (GemmPlan::use_a8;
// numerics-changing, k2-prefill design note P3), sets ws->prefill_p4 iff
// prefill_attention is CERE_LEVER_PREFILL_P4, ws->verify_batched iff verify_attention is CERE_LEVER_VERIFY_BATCHED,
// ws->graphs.enabled iff decode_step is CERE_LEVER_DECODE_GRAPH or CERE_LEVER_DECODE_SPEC_GRAPH and ws->graphs.spec
// iff it is CERE_LEVER_DECODE_SPEC_GRAPH, destroying every captured decode and verify graph first
// (release_decode_graphs: a graph bakes the kernels it was captured with;
// the next eligible step captures anew). Called after instantiate_model and mtp::instantiate (whose plans start on the
// default, cuBLASLt) and by a hot swap. No allocation: every selection runs over the buffers bind_workspace carved
// whatever the levers (the small-M kernel, P4 and the decode graph take no workspace, the W8A8 path takes the MXFP8
// staging buffer every MXFP8 model has; Workspace::rows_logits, the
// small-M batched lm_head's rows, is carved for every model), so a swap between requests is safe. The other levers
// are the host's (draft_tokens) or checked only (moe_gemm). Every plan gets ws->fp8 (GemmPlan::use_fp8; null when
// empty): k2_weights is restart-only, so the table is fixed after INSTANTIATE; and ws->mxfp8 (GemmPlan::use_mxfp8,
// fixed at ALLOCATE).
void apply_levers(const cere::host::Levers &levers, Workspace *ws);
// Destroys every executable of ws->graphs (the model's destructor; the graphs reference its device memory).
void release_decode_graphs(Workspace *ws);

// ---- Sessions ----------------------------------------------------------------------------------------------------

// The model's paged KV pool (pool_bytes, m25 design note D2/D3): per full-attention layer FP8 E4M3 K and V
// [tokens, 512] (row = physical slot) and the QSA compressed keys BF16 [tokens / 4, 128] (row = physical slot / 4).
// Sessions own 64-token pages of it (host::PagePool); a page id covers every layer. Pages are never zeroed: a session
// reads only the slots of its own logical positions [0, position) (D3 stale-page rule).
struct KvPool {
    uint64_t tokens = 0;
    uint8_t *k_cache[kLayers] = {};
    uint8_t *v_cache[kLayers] = {};
    bf16 *qsa_keys[kLayers] = {};
    // M2.9 (M29S2-04, plans with drafts only): the MTP layer's FP8 K and V [tokens, 512] and QSA compressed keys.
    uint8_t *mtp_k = nullptr;
    uint8_t *mtp_v = nullptr;
    bf16 *mtp_qsa_keys = nullptr;
};

// Carves a KvPool from pool_bytes(cfg, pool_tokens, draft) device bytes at `base` (pointers only).
cere_status bind_pool(const cere_qwen4_exp_config &cfg, uint8_t *base, uint64_t pool_tokens, KvPool *out,
                      bool draft = false);

// Device state of one session slot (session_bytes) plus its view of the pool. Arrays are indexed by layer; entries
// of the other layer kind are null.
struct SessionState {
    uint8_t *slot = nullptr;          // the slot's device bytes (bind_session)
    uint64_t slot_bytes = 0;          // session_bytes(cfg, max_context_tokens)
    uint64_t position = 0;            // tokens consumed so far (the next token's 0-based position)
    uint64_t max_context_tokens = 0;  // <= ModelWeights::max_context_tokens
    int64_t *slot_map = nullptr;      // int64 [max_context_tokens]: logical position -> pool slot (session_set_pages)
    uint8_t *k_cache[kLayers] = {};   // full-attention layers: the pool's FP8 E4M3 K [pool tokens, 512], via slot_map
    uint8_t *v_cache[kLayers] = {};
    bf16 *qsa_keys[kLayers] = {};     // full-attention layers: the pool's QSA compressed keys, row slot_map[4b] / 4
    bf16 *qsa_ring[kLayers] = {};     // full-attention layers: QSA pending-key ring BF16 [4, 128]
    k5::LayerState gdn[kLayers] = {};  // GDN layers: conv BF16 [10,240, 3], ssm BF16 [48, 128, 128]
    k4::State ple{};                   // history int64 [2], conv_state BF16 [10,240, 9]
    uint32_t *out_counts = nullptr;    // u32 [vocab]: D5 output counts of sampled steps since the last PREFILL item
    // M2.9 (drafting plans only, bound when the pool has an MTP entry): the MTP layer's pool view (same slot map)
    // and its QSA ring BF16 [4, 128] and the pending MTP proposals u32 [4] in the slot: count (0..3), then 3 ids
    // (kind 10's last 16 bytes, host/snapshot_format.h).
    uint8_t *mtp_k_cache = nullptr;
    uint8_t *mtp_v_cache = nullptr;
    bf16 *mtp_qsa_keys = nullptr;
    bf16 *mtp_qsa_ring = nullptr;
    uint32_t *proposals = nullptr;
    // m213l design note C4: an image session's rope_delta (<= 0; 0 for text), set by set_rope_delta before its
    // prompt and kept across load_state. Every call outside the prompt's image rows rotates position p at
    // p + rope_delta (rope_table).
    int64_t rope_delta = 0;
    // Track LoRA: the adapter slot of Workspace::lora this session runs with (-1: the base model; set before its first
    // step, CudaDevice::set_session_lora).
    int32_t lora_slot = -1;
    // WeightLab research: the plan binding this session runs under (null: none; set before its first step,
    // CudaDevice::set_session_research; owned by the host, it outlives the session).
    const research::Binding *research = nullptr;
};

// Carves a SessionState from one slot of session_bytes(cfg, max_context_tokens, draft) device bytes and points its KV
// at `pool` (pointers only); draft = the pool has an MTP entry (pool.mtp_k non-null).
cere_status bind_session(const cere_qwen4_exp_config &cfg, uint8_t *slot, uint64_t max_context_tokens,
                         const KvPool &pool, SessionState *out);

// A fresh sequence: zeroes the slot's state (the slot map and the sampler's out_counts included, m27 I9/D5), sets the
// PLE history to EOS (k4::kEosTokenId) and position 0. Enqueued on `stream` (the caller synchronizes before the first
// forward or relies on stream order).
cere_status session_reset(const cere_qwen4_exp_config &cfg, SessionState *state, cudaStream_t stream);

// Writes the session's slot map from its pages (after session_reset): slot_map[p] = pages[p / 64] * 64 + p % 64 for
// p < max_context_tokens. `pages` are the n = ceil(max_context_tokens / 64) page ids the session reserved, in
// session order, each below pool_tokens / 64 (else CERE_ERR_ARGUMENT, nothing enqueued). The map is built on the host
// and enqueued from pageable memory (cudaMemcpyAsync stages it before returning); CERE_ERR_CAPACITY if the host
// buffer cannot be allocated.
cere_status session_set_pages(SessionState *state, const uint32_t *pages, uint64_t n, uint64_t pool_tokens,
                              cudaStream_t stream);

// M2.10 fixed state (m210 design note D6′, ledger M4: copy_fixed_state / load_fixed_state). The session's run
// starts at `state.slot + fixed_state_offset(cfg, state.max_context_tokens)` and is fixed_state_bytes(cfg) long; a
// state slot holds the same bytes in the same order, so capture and load are one device copy each.
uint64_t fixed_state_offset(const cere_qwen4_exp_config &cfg, uint64_t max_context_tokens);
// A SessionState view of a prefix node for save_copies / load_copies (D9 host tier, D12 export): its fixed state in
// the state slot at `state` (fixed_state_bytes) and its KV in `pool` through an explicit page list of n_pages ids
// (max_context_tokens = n_pages * 64, so the copies take that list). slot_map and out_counts are null: the kind 8 run
// (the sampler's out_counts, which a prefix never has) is planned with device null, which the caller writes as zeros
// on save and skips on load. On a model with drafts the view also holds the pool's MTP K/V rows and QSA keys, so a
// kind 10 run (Sessions O211: the host tier's MTP rows) covers them; it has no MTP ring and its proposals are planned
// with device null like kind 8. position 0 (set it to the node's length for save_copies). Pointers only.
cere_status bind_fixed_state(const cere_qwen4_exp_config &cfg, uint8_t *state, uint64_t n_pages, const KvPool &pool,
                             SessionState *out);

// m210 D15⁗′ (stage 2a: end capture + class-pinned resume) of one PREFILL call through views of state slot pairs
// (bind_fixed_state): slot `a` (standard layout) holds the GDN conv, PLE and ring state and, in each GDN layer's
// BF16 ssm region, the FP32 SSM of value heads 0..23; slot `b` holds heads 24..47 in its ssm regions. `cls` (0 = n)
// is the row count of the cold call these rows belong to: every row-count choice of the call is made for it.
// Capture (`row` > 0: a multiple of 64 strictly inside the call): the state as of row `row` goes to the pair `a`, `b`
// (GDN conv, FP32 SSM, PLE history and conv state; the caller copied the rest at the call start). Resume (`in_a`,
// `in_b` non-null): the GDN chunk rule starts from that pair's FP32 SSM instead of session.gdn[L].ssm. D15⁗″: one call
// may do both, through two disjoint pairs.
// PrefixCapture G1: a chunk-0 call (its cold call starts at position 0, so its dense attention reads its own BF16 K/V
// rows) has K/V runs in the state slots after a pair (Device::end_kv_slots): attention layer i (the target's
// full-attention layers ascending, then the MTP layer) holds K then V [rows, 512] BF16 at run + 2 i rows 512
// (end_kv). A resume attends over `in_kv`'s rows [0, in_kv_rows = the session position); a capture writes the rows
// [0, kv_rows = the session position + row) to `kv`: from the call's own rows, after a resume's saved ones. Null for
// any other call.
struct EndCall {
    uint32_t cls = 0;
    int64_t row = -1;
    const SessionState *a = nullptr;
    const SessionState *b = nullptr;
    const SessionState *in_a = nullptr;
    const SessionState *in_b = nullptr;
    bf16 *kv = nullptr;
    int64_t kv_rows = 0;
    const bf16 *in_kv = nullptr;
    int64_t in_kv_rows = 0;
};

// The bytes of a chunk-0 end state's K/V run (EndCall::kv) over `rows` rows and `layers` attention layers.
inline uint64_t end_kv_bytes(uint64_t layers, uint64_t rows) { return 2 * layers * rows * kKvWidth * sizeof(bf16); }

// Session state copy (m211 design note D2, D4; ledger M4): the device sections of a CERESNP1 image (every kind >= 1
// of `plan`; kind 0, the tokens, is the host's). Each section holds its kind's layers ascending and the logical rows
// [first_row, position) of the session's state (attention K/V rows, complete QSA blocks [first_row / r, position / r),
// the position mod r valid QSA ring rows, GDN conv and SSM state, PLE history and conv state, the sampler's
// out_counts u32 [vocab] (kind 8, m27 I12)), so neither the slot nor
// the pages enter the bytes: K/V rows and QSA keys are gathered from (save) or scattered to (load) the pool through
// `pages`, the session's n_pages = ceil(max_context_tokens / 64) page ids in session order (the ids its slot map was
// written from, session_set_pages; ledger X19), else CERE_ERR_ARGUMENT. `plan` must hold every section of kinds 1-8
// at exactly these sizes (host::snapshot::plan); another kind is CERE_ERR_UNSUPPORTED, a size off its derived value
// CERE_ERR_ARGUMENT. These functions only plan (no CUDA call): *out lists the contiguous device runs in image order
// (sections in plan order, each section's runs abutting from its offset), and the caller (CudaDevice::save_state /
// load_state) moves the bytes. On error *out is unspecified and nothing may be copied.
struct StateCopy {
    uint8_t *device;        // device bytes [device, device + bytes)
    uint64_t image_offset;  // image bytes [image_offset, image_offset + bytes)
    uint64_t bytes;         // may be 0 (an empty QSA ring)
    uint64_t zero_bytes;    // load only: device bytes right after the run to zero-fill (QSA ring rows past the valid)
};
// save_copies reads state at state.position; the state is unchanged.
cere_status save_copies(const cere_qwen4_exp_config &cfg, const SessionState &state, const uint32_t *pages,
                        uint64_t n_pages, const host::snapshot::Plan &plan, uint64_t first_row,
                        std::vector<StateCopy> *out);
// load_copies plans the writes into a fresh state (session_reset, position 0, else CERE_ERR_ARGUMENT) at `position`
// <= state.max_context_tokens (else CERE_ERR_CAPACITY); once every run is copied and zero-filled, the caller sets
// state.position = position and the next forward continues exactly as the saved state's.
cere_status load_copies(const cere_qwen4_exp_config &cfg, const SessionState &state, const uint32_t *pages,
                        uint64_t n_pages, const host::snapshot::Plan &plan, uint64_t first_row, uint64_t position,
                        std::vector<StateCopy> *out);

// ---- Forward -----------------------------------------------------------------------------------------------------

// Optional taps (executor/tests/gpu/test_numerics_b8_layers.py B8 names; m22p design note D3 detail names and D5
// patch points). forward takes a nullable ForwardTaps*; null costs nothing (no name formatting, MoE taps not
// requested, top-2 of the last row only). With `detail` 0 and `patch` null the arithmetic and launches are those of
// a null ForwardTaps plus the B8 copies (`emit`).
//
// Callbacks run on forward's thread, in this forward order (D = detail_sink, only with `detail` 1; E = emit;
// I = patch input point, return ignored; P = patch point, replaceable); L = layer, `cols` in brackets:
//   per layer L ([PLE add at L = 1] first; it has no point):
//     D native.layers.L.attn_mix.in [10,240]               the HC state (residual streams) the attn mix reads
//     D model.layers.L.attn_hyper_connection.hc_norm [10,240]  the mix's normed streams (ws.normed)
//     I native.layers.L.attn_mix.normed [10,240]           the same tensor, for a screen recomputing the mix
//     P native.layers.L.attn_mix.out [2,560]               the mix output (ws.mixed)
//     D native.layers.L.block.in [2,560]                   the attention block input (after P above)
//     full attention:
//       I native.layers.L.gemm.attn_qkv.in [2,560]   P native.layers.L.gemm.attn_qkv.out [13,312]
//       D model.layers.L.self_attn.qkv_proj [13,312]       packed q|gate, k, v (SGLang's qkv_proj order)
//       I native.layers.L.gemm.attn_o.in [6,144]     P native.layers.L.gemm.attn_o.out [2,560]
//       E model.layers.L.self_attn.o_proj [2,560]    P model.layers.L.self_attn.o_proj [2,560]
//     GDN (linear attention):
//       I/P native.layers.L.gemm.gdn_qkv.{in [2,560], out [10,240]}
//       I/P native.layers.L.gemm.gdn_z.{in [2,560], out [6,144]}
//       I/P native.layers.L.gemm.gdn_ba.{in [2,560], out [48]} twice: b, then a
//       D model.layers.L.linear_attn.in_proj_qkvz [16,384]   qkv | z (SGLang's fused in_proj_qkvz order)
//       D model.layers.L.linear_attn.in_proj_ba [96]         b | a (SGLang's fused in_proj_ba order)
//       D model.layers.L.linear_attn.norm [6,144]            the gated-RMSNorm output y (out_proj input)
//       I native.layers.L.gemm.gdn_out.in [6,144]    P native.layers.L.gemm.gdn_out.out [2,560]
//       E model.layers.L.linear_attn.out_proj [2,560]   P model.layers.L.linear_attn.out_proj [2,560]
//     D native.layers.L.mlp_mix.in [10,240]
//     D model.layers.L.mlp_hyper_connection.hc_norm [10,240]
//     I native.layers.L.mlp_mix.normed [10,240]
//     P native.layers.L.mlp_mix.out [2,560]
//     D native.layers.L.mlp.in [2,560]                     the MoE input (ws.mixed, after P above)
//     I native.layers.L.mlp.in [2,560]
//     MoE (kernels::MoeTaps.point, block order):
//       I/P native.layers.L.gemm.moe_router.{in [2,560], out [512]}
//       D model.layers.L.mlp.gate [512]                      router logits
//       I/P native.layers.L.gemm.moe_shared_gate_up.{in [2,560], out [640]} twice: gate, then up
//       D model.layers.L.mlp.shared_expert.gate_up_proj [1,280]   gate | up
//       I/P native.layers.L.gemm.moe_shared_down.{in [640], out [2,560]}
//     E model.layers.L.mlp.experts [2,560]                 routed sum (MoeTaps.routed)
//     E model.layers.L.mlp.shared_expert.down_proj [2,560] (MoeTaps.shared)
//     P model.layers.L.mlp.experts [2,560]                 a replacement enters the combine as MoeTaps.routed_in
//     P model.layers.L.mlp.shared_expert.down_proj [2,560] a replacement enters the combine as MoeTaps.shared_in
//   after the last layer:
//     D native.final_mix.in [10,240]   D native.final_mix.normed [10,240]   I native.final_mix.normed [10,240]
//     P native.final_mix.out [2,560]   E model [2,560]  (the final mixer output, lm_head input)
//   per scored row (the last `rows` rows, oldest first):
//     I native.logits [248,320] with rows = 1: that row's BF16 lm_head logits, before its top-2
// Not points: the PLE GEMMs (ple_key, ple_value: inside K4), the HC mix GEMMs (inside K3; their mix is a point) and
// the lm_head GEMM (its output is native.logits).
struct ForwardTaps {
    // Rows reported per tensor (emit, detail_sink) and scored: the last `last_rows` rows of the call (0 = all n;
    // > n = n). Patch callbacks always see all n rows.
    uint64_t last_rows = 0;
    void *ctx = nullptr;  // passed to every callback below
    // E (nullable): `values` is device BF16 [rows, cols] row-major (cols = 2,560), valid only until emit returns: the
    // sink must enqueue its copy on `stream` before returning. A non-OK status aborts the forward with that status.
    cere_status (*emit)(void *ctx, const char *name, const bf16 *values, uint64_t rows, uint64_t cols,
                        cudaStream_t stream) = nullptr;
    // D: 0 = none, 1 = every D name above through `detail_sink` (same contract as emit, any cols). GDN detail
    // concatenations are staged in the K5 core workspace and gate | up in the K7 SwiGLU region (both dead then).
    uint32_t detail = 0;
    cere_status (*detail_sink)(void *ctx, const char *name, const bf16 *values, uint64_t rows, uint64_t cols,
                               cudaStream_t stream) = nullptr;
    // I and P (nullable): `native` is device BF16 [rows, cols] (rows = n, except native.logits: 1), readable in
    // `stream` order during the call; `layer` is L (kLayers for native.final_mix.* and native.logits). At a P point a
    // non-null return is a device BF16 [n, cols]
    // replacement that forward copies over the native tensor on `stream` (then synchronizes `stream`, so the
    // pointer only needs to stay valid until the next callback); null keeps native. Returns of I points are
    // ignored. Any patch, and any detail, routes GDN layers through the decomposed K5 path (the same launches).
    const bf16 *(*patch)(void *ctx, const char *name, uint64_t layer, uint64_t rows, uint64_t cols,
                         const bf16 *native, cudaStream_t stream) = nullptr;
    // true: emit only "model.layers.L.self_attn.input" (ws.mixed, the full-attention block input after the HC mix,
    // [rows, 2,560] BF16) for every full-attention layer L, nothing else (no B8 name, no "model"), and stop after the
    // last layer: no final mixer, no logits, nothing copied to out_top2_last_or_all (K9 oracle, m22c Z2AC-04).
    // false: the B8 taps above.
    bool attn_inputs_only = false;
    // The encoder's taps (vision::Taps; nullable): only a call that runs the encoder reports them.
    const vision::Taps *vision = nullptr;
};

// The sampler stage's request (m27 design note I9): the work item's sampling copy made at submit (I8). `sampled`
// false is a greedy item that asked for logprobs only.
using SamplerArgs = host::ItemSampling;

// The sampler preconditions of forward and forward_decode_rows: CERE_ERR_ARGUMENT for top_logprobs above
// CERE_MAX_TOP_LOGPROBS or, when `sampled`, a parameter outside the K12 kernels' ranges or a logit_bias id >= vocab;
// CERE_OK otherwise (a vocab_mask, M2.8, is ceil(vocab / 32) host words the host validated at submit). Host only.
cere_status check_sampler(const SamplerArgs &s, uint32_t vocab);

// One forward call of n new tokens of `session`, appended at session.position, as `operation`: CERE_OP_PREFILL for
// the prompt (any n, including a 1-token prompt; a prompt longer than ws.max_tokens is several PREFILL calls with the
// state carried), CERE_OP_DECODE for one generated token (n = 1). The operation picks the GDN path as SGLang's
// backend does (K5: forward_extend's chunked delta rule vs forward_decode's single-token step); everything else is
// the same for both. Layer order follows SGLang qwen4_exp: embedding -> HC expansion (x4 streams);
// per layer: [PLE add at layer 1] -> attn HC mix -> GDN or full attention -> combine -> mlp HC mix -> MoE ->
// combine; then the final mixer (mix only) -> lm_head (one m = 1 row at a time) -> top-2.
// Full attention reads BF16 K/V of the call when session.position == 0 and the FP8 cache otherwise (K6).
//
// Preconditions (CERE_ERR_ARGUMENT before any launch, session untouched): 1 <= n <= ws.max_tokens; operation
// CERE_OP_PREFILL, or CERE_OP_DECODE with n = 1;
// position + n <= session.max_context_tokens; every token < vocab_size; taps (when non-null) with detail 0 or 1, a
// detail_sink when detail is 1, and at least one of emit / detail / patch; out_top2_last_or_all non-null host memory
// of 1 entry (taps null) or min(taps->last_rows or n, n) entries (scores of the reported rows, oldest first; entry r
// is the top-2 of the logits at that position, i.e. the prediction of the next token); sampler non-null only with
// taps null, sampler->bias ids unique and < vocab_size, top_logprobs <= CERE_MAX_TOP_LOGPROBS, vocab_mask null or
// ceil(vocab_size / 32) host words with an allowed id (M2.8); out_logprobs non-null only with sampler non-null.
//
// sampler null: the greedy path above, unchanged (D1). sampler non-null: the K12 stage (kernels/k12_sampler.h) on the
// scored row's BF16 logits: process (sampled: penalties from session.out_counts and logit_bias; else none; with the
// item's vocab_mask, uploaded to ws.sampler.mask, between the two: m28 design note D9) into
// ws.proc_logits, top2_f32, for a sampled item with temperature > 0 probabilities, keep_mask and a draw keyed by the
// position of the last fed token (D4, channels (0, 1), seed params.seed); out_top2_last_or_all[0] is the committed
// token's score over the processed row (the greedy path's score when `sampled` is false); out_logprobs (when
// non-null) gets D7's logprobs of the committed token and the top `top_logprobs`; a sampled item counts the committed
// token in session.out_counts (D5).
//
// Images (m213 design note D10, D11; m213l design note VL-L3; nullable, only on a model with
// ws.max_image_patches > 0): the session's prompt images and mRoPE positions, passed with every PREFILL call of an
// image session's prompt (any position, any chunk). Before anything else of the call, each image whose pad rows
// overlap the call's rows runs one encoder pass (its patches through the mapped staging to ws.image_patches,
// vision::encode with the workspace region as scratch, into ws.image_out) and the overlap's merged rows are copied to
// ws.image_embeds at their call rows; an image straddling calls is encoded once per call (deterministic launches,
// the same rows). The splice then puts those rows over ws.mixed after the embedding gather and before the HC
// expansion (kv::splice), and the PLE id stream carries each image's pad_id at that image's pad rows. The call's
// rows rotate at images->positions3's columns (K1/K6 [3, n]; K9 [3, position % 4 + n] from position - position % 4,
// prompt_rope). Every other call (DECODE, or a session without images) rotates position p at p +
// session.rope_delta (rope_table; the decode graph only at rope_delta 0). Extra preconditions (ARGUMENT): images
// only on a PREFILL of n <= layout::kMaxImageTokens rows, positions3 non-null with prompt_len >= position + n; each
// image t * h * w <= ws.max_image_patches (merged rows <= layout::kMaxImageTokens) with h, w even and pixels of
// t * h * w * 1,536 BF16 within ws.image_patch_bytes, its pad rows inside the prompt.
// With detail 1 the spliced ws.mixed is reported as D native.embed.spliced [2,560] (only with images); taps->vision
// (nullable) receives the encoder taps.
// Stream: everything is enqueued on `stream`; forward synchronizes `stream` (and the MoE fork's side stream) before it
// returns (the caller needs the token on the host for stop/limit checks). No allocation. On success session.position
// += n. `launched` (MF-2; this and the rows variants below set it): false when the call returned from a precondition
// before any device work (the session untouched), true from the first enqueue on, whatever the status then: a
// non-OK status with `launched` leaves the session's state undefined (the caller releases or resets it).
// `end` (nullable; m210 D15⁗′ stage 2a): a PREFILL's EndCall (class pin, capture or resume, above); ARGUMENT for one
// on a DECODE, with cls outside [n, ws.max_tokens], a capture row off the 64 grid or outside (0, n), both capture
// and resume, or a missing slot view. Null: the call unchanged.
// `segment` (nullable; MixedPD, mixed-pd design note): run the eager PREFILL call as layer segments. A call starts
// when segment->park->next_layer is 0 and runs layers [next_layer, min(next_layer + segment->layers, kLayers));
// short of kLayers it saves ws.hyper (and ws.normed while the next mix reads it) into the park, synchronizes and
// returns CERE_OK with segment->paused true, session.position unchanged and nothing scored. The caller then may run
// other forward calls of the model (decode rows, verify, MTP) and resumes the call with the same arguments: the ids
// and positions are staged again, the park restored, and the next layers run. The segment that runs the last layer
// finishes the call as an unsegmented one does (paused false, position += n). Every launch of the call and its
// inputs are the unsegmented call's, so its outputs and state are bit-identical. ARGUMENT for a segment on a DECODE,
// with taps, without park buffers, layers 0, or a resume whose n differs from the paused call's.
struct Park {
    bf16 *hyper = nullptr;   // [T, 10,240] (layout::park_bytes)
    bf16 *normed = nullptr;  // [T, 10,240]
    uint32_t next_layer = 0;  // the first layer of the paused call's next segment; 0: no call paused
    uint32_t n = 0;           // the paused call's rows
    bool normed_valid = false;  // ws.normed held the next mix's norm at the pause (combine's next_norm)
};
struct Segment {
    Park *park = nullptr;
    uint32_t layers = 0;  // the most layers this segment runs (>= kLayers: to the end)
    bool paused = false;  // out
};
cere_status forward(const ModelWeights &weights, Workspace &ws, SessionState &session, uint32_t operation,
                    const uint32_t *tokens, uint32_t n, ForwardTaps *taps, cere_token_score *out_top2_last_or_all,
                    const SamplerArgs *sampler, cere_token_logprobs *out_logprobs, cudaStream_t stream,
                    bool &launched, const host::StepImages *images = nullptr, const EndCall *end = nullptr,
                    Segment *segment = nullptr);

// One batched decode step (m26 design note D5, I5): row r feeds tokens[r] to sessions[r] at its own position and
// writes the top-2 of its next token to out[r], bit-identical to forward(CERE_OP_DECODE, n = 1) of that session alone
// (D6). Per layer: embedding and HC at m = rows; PLE (layer 1) as one k4::forward_rows at m = rows (each row its own
// sequence with that session's k4 State, GEMMs on RowsPlans::ple_key / ple_value); GDN
// projections at m = rows, then k5::layer_core once per row with that session's LayerState; attention QKV / O
// projections at m = rows, then the single-row decode attention once per row on that session's prefix (dense K6, or
// the K9 QSA path above 2,048 context with its indexer projection at m = 1); moe_block at m = rows on the rows
// MoePlans (or once per row on the single-row plans when ws.rows.moe_per_row); the final mixer at m = rows; lm_head
// and top-2 once at m = rows on the lm_head rows plan (RowsPlans::lm_head), then each row's sampler stage when
// samplers[r] is non-null.
//
// Preconditions (CERE_ERR_ARGUMENT before any launch, sessions untouched): ws.rows built (instantiate_model with
// ws.max_rows > 1), 1 <= rows <= min(ws.rows.rows, ws.max_tokens); sessions non-null and distinct; every
// position + 1 <= max_context_tokens; every token < vocab_size; taps (nullable) with an emit callback only (detail 0,
// no patch, not attn_inputs_only): each B8 tensor of forward's order is emitted once per row, rows in order, as that
// row's [1, 2,560] slice (the 2,560-column emit contract; last_rows is ignored); `out` holds `rows` entries;
// `samplers` null (every row greedy) or `rows` entries, entry r null (row r greedy) or passing check_sampler (else its
// status), only with taps null; `out_logprobs` null or `rows` entries, entry r non-null only with samplers[r].
//
// Sampling (ledger CR4): row r with samplers[r] runs forward's K12 stage on its own logits after its lm_head row, with
// its session's out_counts and the counter key sessions[r]->position (the position of the token it feeds), so out[r]
// and *out_logprobs[r] equal forward(CERE_OP_DECODE, n = 1, samplers[r]) of that session alone whatever the other
// rows are. The rows' stages run one after another on `stream` and share the Workspace's sampler scratch and
// proc_logits row, so the sampler adds no device memory per row beyond each session's out_counts.
// Stream and errors as forward: synchronizes `stream` before returning; on success every position += 1; a failure
// after the first launch returns its status with every session's state undefined.
cere_status forward_decode_rows(const ModelWeights &w, Workspace &ws, SessionState *const *sessions, uint32_t rows,
                                const uint32_t *tokens, ForwardTaps *taps, cere_token_score *out,
                                const SamplerArgs *const *samplers, cere_token_logprobs *const *out_logprobs,
                                cudaStream_t stream, bool &launched);

// The KV state of one full-attention layer of one session (a target layer or the MTP layer, m29 design note D3):
// the pool's FP8 K/V and QSA compressed keys through `slots` (logical position -> pool slot), and the QSA ring.
struct AttnKv {
    uint8_t *k_cache = nullptr;
    uint8_t *v_cache = nullptr;
    bf16 *qsa_ring = nullptr;
    bf16 *qsa_keys = nullptr;
    const int64_t *slots = nullptr;
    int64_t context = 0;  // the slot map's length: the session's max_context_tokens (k9::Bounds::context)
};

// The rotary inputs of one attention_rows call (m213l design note C4, C5). `table` is the K1 rope table base:
// rope_table(w, session) for every row past an image prompt's rows (a text session: w.rope_cos_sin itself, so its
// launches are the text launches). `positions3` (int64 device [3, n], axes t, h, w): an image prompt's rows at their
// mRoPE positions in K1/K6 (`table` the unshifted w.rope_cos_sin); `k9` (int64 device [3, k9_stride]) the same rows
// for K9, column c = logical position prefix - k9_lead + c, k9_lead = prefix % 4 (the first member of a group begun
// in an earlier call), both null for every other call.
struct Rope {
    const float *table = nullptr;
    const int64_t *positions3 = nullptr;
    const int64_t *k9 = nullptr;
    int64_t k9_stride = 0;
    int64_t k9_lead = 0;
};

// The table base of `s`'s calls outside its prompt's image rows: row p of it is the rotary position p + rope_delta on
// all three mRoPE axes (C4). Positions p + rope_delta >= 0: rope_delta = max mRoPE position + 1 - prompt length.
inline const float *rope_table(const ModelWeights &w, const SessionState &s) {
    return w.rope_cos_sin + s.rope_delta * int64_t(kRotaryDim);
}

// The k9::Bounds of a K9 call of a session with `context` positions whose rotary rows read `table` (a row base inside
// w.rope_cos_sin: rope_table(w, s) or w.rope_cos_sin itself): rows (w.rope_cos_sin - table) / 64 + [0, table rows).
inline k9::Bounds k9_bounds(const ModelWeights &w, const Workspace &ws, const float *table, uint64_t context) {
    const int64_t first = (w.rope_cos_sin - table) / int64_t(kRotaryDim);
    return {first, first + int64_t(w.max_context_tokens), int64_t(context), ws.bound_fault};
}

// Reads and clears ws.bound_fault after a drained call (audit KernelsAttention F3): set, a K9 kernel clamped an
// out-of-range device position or rotary row (the call computed at the clamped value, so its state is wrong); logs
// "cere: device_bound_fault code=<word>" and returns CERE_ERR_INTERNAL (the caller breaks the session). Else CERE_OK.
cere_status take_bound_fault(Workspace &ws);

// The Rope of an image prompt call of n rows at `prefix` whose positions forward staged into ws.positions3 (K1/K6's
// [3, n], then K9's [3, prefix % 4 + n]); the MTP extend over the same rows reads it too.
inline Rope prompt_rope(const ModelWeights &w, const Workspace &ws, uint64_t prefix, uint32_t n) {
    const int64_t lead = int64_t(prefix % 4);
    return Rope{w.rope_cos_sin, ws.positions3, ws.positions3 + 3 * uint64_t(n), lead + int64_t(n), lead};
}

// PrefixCapture G1: the BF16 K/V rows of one attention layer of a chunk-0 end state's runs (EndCall::kv, in_kv), [rows,
// 512] each. Resume (`in_k` non-null, prefix == in_rows): the rows attend over the saved rows [0, in_rows) and their
// own as the cold call's rows attended over its own BF16 rows, so with the class pin they equal that call's rows.
// Capture (`k` non-null, prefix < rows < prefix + n; prefix 0 unless the call resumes): attention_rows copies the rows
// [0, rows) its dense call attends over (the rotated K, the raw V) there: its own, after a resume's saved ones.
struct EndKv {
    bf16 *k = nullptr;
    bf16 *v = nullptr;
    int64_t rows = 0;
    const bf16 *in_k = nullptr;
    const bf16 *in_v = nullptr;
    int64_t in_rows = 0;
};

// Attention layer `index`'s EndKv of `call` (EndCall::kv or in_kv non-null).
inline EndKv end_kv(const EndCall &call, uint64_t index) {
    EndKv out;
    if (call.kv != nullptr) {
        out.k = call.kv + 2 * index * uint64_t(call.kv_rows) * kKvWidth;
        out.v = out.k + uint64_t(call.kv_rows) * kKvWidth;
        out.rows = call.kv_rows;
    }
    if (call.in_kv != nullptr) {
        out.in_k = call.in_kv + 2 * index * uint64_t(call.in_kv_rows) * kKvWidth;
        out.in_v = out.in_k + uint64_t(call.in_kv_rows) * kKvWidth;
        out.in_rows = call.in_kv_rows;
    }
    return out;
}

// The EndCall::kv index of target layer `layer`: the full-attention layers below it. end_kv_index(cfg, kLayers) is the
// MTP layer's (after every target one), and end_kv_index(cfg, kLayers) + 1 the run's layer count.
inline uint64_t end_kv_index(const cere_qwen4_exp_config &cfg, uint32_t layer) {
    uint64_t index = 0;
    for (uint32_t j = 0; j < layer && j < kLayers; ++j) index += (cfg.full_attention_layers >> j) & 1;
    return index;
}

// The attention between the projections for n rows of one session at positions prefix .. prefix + n - 1 (forward.cpp):
// `qkv` [n, 13,312] (q|gate, k, v; the prologue rewrites it), `mixed` [n, 2,560] (the QSA indexer input, contexts above
// 2,048) -> `attn_out` [n, 6,144] (the o_proj input), appending the rows' K/V (and compressed keys) to `kv`, rotated
// as `rope` says. Scratch is ws.scratch. Enqueued on `stream`, no synchronization; CERE_ERR_INTERNAL on a launch
// failure. `cls` (m210 D15⁗′; 0 = n): the rows of the cold call these rows belong to, for every row-count choice of
// the K9 path (the indexer projection's K2 plan, the K9 split and P4 choice) and of a dense K6 call with `end_kv`
// (EndKv; only on the dense path of a context above 2,048, else CERE_ERR_UNSUPPORTED); CERE_ERR_UNSUPPORTED for a pin
// (cls != n) on any other dense K6 call.
cere_status attention_rows(const ModelWeights &w, const AttentionWeights &aw, const Workspace &ws, const AttnKv &kv,
                           int64_t prefix, bf16 *qkv, const bf16 *mixed, bf16 *attn_out, uint32_t n, const Rope &rope,
                           cudaStream_t stream, uint32_t cls = 0, const EndKv *end_kv = nullptr);

// attention_rows at prefix = positions[0] read on the device (`positions` int64 device [n], positions[0] + r; the
// spec_graph draft graph's MTP layer): the same launches with K6's and K9's device-position twins
// (k9::qsa_compress_store_at, k6::prologue_rows_at, k6::dense_attention_at), bit for bit. Only for dense rows with a
// prefix (the caller checks positions[0] > 0 and positions[0] + n <= layout::kDenseContext, n <= 16) of a session with
// rope_delta 0 (the model's table).
cere_status attention_rows_at(const ModelWeights &w, const AttentionWeights &aw, const Workspace &ws, const AttnKv &kv,
                              const int64_t *positions, bf16 *qkv, const bf16 *mixed, bf16 *attn_out, uint32_t n,
                              cudaStream_t stream);

// ---- Speculation (m29 design note D4, D7 as M29S2-20, I8) -----------------------------------------------------

// One undo snapshot of a drafting session (M29S2-20). undo[j - 1] holds the state just before verify row j in
// {1, 2, 3} first mutates it (= the state after rows 0..j-1); commit of a <= n restores undo[a - 1]. Copies are taken
// in place right before the row's kernels; the GDN pieces in-kernel by k5::chain_core (m23 design note DG-02).
// Carved only for plans with drafts (layout::undo_bytes per snapshot, 3 per session), bound by the device.
struct UndoState {
    k5::LayerState gdn[kLayers] = {};  // GDN layers: conv + ssm copies
    k4::State ple{};                   // PLE history + conv copy
    bf16 *qsa_ring[kLayers] = {};      // full-attention layers above 2,048 context: the [4, 128] ring copy
    bf16 *qsa_key[kLayers] = {};       // the one compressed-key row the chain may complete ([128] each)
    int64_t key_row = -1;              // pool row of that compressed key (-1: none saved)
};

// One verify chain of one drafting session: tokens[0] is the session's last committed (unfed) token at position p,
// tokens[1..n] the drafts at p+1..p+n (n <= 3, p + n < max_context_tokens). `undo` is the session's 3 snapshots and
// `pages` its n_pages host page ids (for the QSA key row's pool slot). `sampler` (M2.9 phase S, null = greedy on the
// raw logits, phase G) is the item's sampler (check_sampler passed; a vocab_mask, GrammarSpec GS-02, is n + 1
// consecutive masks, row j's for verify row j); `logprobs` asks for the committed tokens' logprobs (only with a
// sampler).
struct VerifyRow {
    SessionState *session = nullptr;
    UndoState *undo = nullptr;  // [3]
    const uint32_t *tokens = nullptr;
    uint32_t n = 0;
    const uint32_t *pages = nullptr;
    uint64_t n_pages = 0;
    const SamplerArgs *sampler = nullptr;
    bool logprobs = false;
};

// Runs every chain's rows (Σ (n + 1) <= ws.rows.rows <= 16) as decode rows: embedding, HC, dense GEMMs and MoE at
// m = Σ rows on the rows plans exactly as forward_decode_rows (honouring moe_per_row); PLE as one k4::forward_rows and
// GDN as one k5::chain_core over every chain (a chain is a sequence on its own state, its rows in order, its undo
// saves taken in-kernel; per chain bit-equal to the one-row calls); attention and QSA advance one row at a time, in
// chain order, so verify row j's out entry is bit-equal to forward(CERE_OP_DECODE) at position p + j after rows
// 0..j-1 (D4); before row j >= 1 mutates a session's state it is copied to undo[j - 1].
// Writes the pre-final-mixer HC rows [Σ rows, 10,240] to ws.hc_rows when it is non-null (the MTP draft-extend input;
// SGLang's hc_hidden_states) and the top-2 of every row into out (row-major, chains in order). Sessions' positions are
// NOT advanced (verify_commit does). Taps: emit only, as forward_decode_rows.
//
// A chain with a sampler (phase S, m29 design note D6; needs ws.spec bound, else CERE_ERR_ARGUMENT) runs order 16's
// sampler stage on each of its rows right after that row's lm_head, into the row's ws.spec buffers: process (a sampled
// item: penalties from out_counts plus the drafts the row's prefix accepted, then the row's vocab mask when the chain
// is constrained, GrammarSpec GS-02, then logit_bias), top2_f32, and with temperature > 0 probabilities and keep_mask.
// With temperature > 0 the chain's committed ids are k13s::spec_accept's (seed = params.seed, key0 = p, the counter
// key plain sampling uses at p); with temperature 0 (or a greedy logprobs-only item) row j commits its processed
// top-1. out row j is then the score of row j's committed id over row j's processed logits (k12::finalize; the
// raw-logits top-2 for a greedy logprobs-only item), so the leading drafts equal to their row's out id are exactly the
// accepted ones (a rejected draft is never resampled), and out_logprobs (nullable, Σ rows entries; written for chains
// with `logprobs`) row j its logprobs (D7). Rows after a chain's first rejection carry unspecified ids. A sampled
// chain's row ids also land in ws.spec.committed from its first row on (verify_commit's input); out_counts are not
// touched here.
cere_status forward_verify_rows(const ModelWeights &w, Workspace &ws, const VerifyRow *rows, uint32_t row_count,
                                ForwardTaps *taps, cere_token_score *out, cudaStream_t stream, bool &launched,
                                cere_token_logprobs *out_logprobs = nullptr);

// Commits a in [1, n + 1] tokens of a verified chain at p: a <= n restores undo[a - 1] (GDN, PLE, QSA ring and the
// saved key row), a = n + 1 keeps the state; then position = p + a. Raw K/V rows past p + a are rewritten before any
// read (D8) and need no undo. A chain whose sampler is sampled counts its a committed ids into the session's
// out_counts (D5, M29S2-14): `committed` (device) is that chain's ids, ws.spec.committed at the chain's first row
// after forward_verify_rows (required then, else CERE_ERR_ARGUMENT).
cere_status verify_commit(const cere_qwen4_exp_config &cfg, const VerifyRow &row, uint32_t a, cudaStream_t stream,
                          const uint32_t *committed = nullptr);

// The spec_graph draft graph (decode_step = spec_graph, m29 design note M29S2-18b, m23 design note P11R-20): one
// drafting session's MTP draft through ws.graphs.draft[rows], the launches of the eager draft (device.cpp
// CudaDevice::draft) with the positions read on the device: host_ids64[0, rows + kDraftSteps) = p0 + i -> ids64 (one
// H2D copy, stream-ordered ahead of the graph), then the extend of `rows` rows at p0 (hc_in = hc_out = hc, next ids
// `next`, last_only, draft into proposals[1]) and layout::kDraftSteps one-row decode steps from the extend's last hc
// row (step k: next proposals[k], draft into proposals[k + 1]). The caller checks ws.graphs.spec and
// layout::spec_draft_graph_fits(p0, rows, s.max_context_tokens, ws.max_tokens), so every step the eager loop runs
// runs here too, and writes proposals[0] after it. Enqueued on `stream`, no synchronization (the caller
// synchronizes).
cere_status draft_graph(const ModelWeights &w, Workspace &ws, SessionState &s, uint64_t p0, uint32_t rows, bf16 *hc,
                        const uint32_t *next, uint32_t *proposals, cudaStream_t stream);

}  // namespace cere::model
