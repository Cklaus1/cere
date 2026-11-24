// K8 (m22 design note M2.2b): INSTANTIATE of the A4 model (derived tensors, GEMM plans) and the 48-layer qwen4_exp
// forward pass composed from the K1-K7 kernels, in SGLang's order (sglang/srt/models/qwen4_exp.py):
//   embed_tokens -> cat x hc_count (`_prepare_qwen4_exp_attn` :1284-1289); per layer: [PLE add at layer 1, query =
//   the HC state (residual is None), :1291-1304] -> attn_hyper_connection.mix -> GDN (`linear_attn`) or full attention
//   (`self_attention`) -> attn_hyper_connection.combine -> mlp_hyper_connection.mix -> MoE -> mlp combine
//   (:1309-1385); then hyper_connection_mixer.mix (:1668) -> lm_head (one m = 1 row per scored position) -> top-2.
// No allocation: every buffer comes from the Workspace and the SessionState; plans are built by instantiate_model.
#include "model/model.h"
#include "model/moe_backend.h"

#include "model/layout.h"
#include "model/mtp.h"
#include "model/plan.h"

#include "fault/fault_inject.h"

#include "research/hook.h"

#include "kernels/k1_primitives.h"
#include "kernels/k5_gdn_chunk.h"
#include "kernels/k6_attention.h"
#include "kernels/k9_qsa.h"
#include "kernels/k12_sampler.h"
#include "kernels/k13s_spec_accept.h"
#include "kernels/kv_vision.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <new>
#include <vector>

namespace cere::model {
namespace {

constexpr uint64_t kRowBytes = uint64_t(kHidden) * sizeof(bf16);

size_t align256(size_t n) { return (n + 255) & ~size_t(255); }

cere_status cuda_status(cudaError_t e) { return e == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL; }

#define CERE_TRY(expr)                        \
    do {                                      \
        const cere_status s_ = (expr);        \
        if (s_ != CERE_OK) return s_;         \
    } while (0)
#define CERE_TRY_CUDA(expr) CERE_TRY(cuda_status(expr))

float model_eps(const cere_qwen4_exp_config &cfg) { return float(cfg.rms_norm_eps); }

kernels::HcShape hc_shape(const cere_qwen4_exp_config &cfg) {
    return kernels::HcShape{kHc, kHidden, cfg.hc_lowrank, model_eps(cfg)};
}

constexpr kernels::MoeDims kMoeDims{kHidden, kExperts, kTopK, kMoeInter, kSharedInter};

// The taps of one forward call (model.h ForwardTaps): every helper is a no-op without its callback, and formats no
// name then. `rows` = the reported rows (the last `rows` of n). `cls` and `end`: the call's EndCall (model.h; 0 and
// null without one), every row-count choice made for cls (0 = n).
struct Tap {
    ForwardTaps *taps;  // nullable
    uint32_t n;
    uint64_t rows;
    cudaStream_t stream;
    uint32_t cls = 0;
    const EndCall *end = nullptr;

    // B8 emits; none under attn_inputs_only (then only attn_input emits).
    bool emitting() const { return taps != nullptr && taps->emit != nullptr && !taps->attn_inputs_only; }
    bool detail() const { return taps != nullptr && taps->detail != 0 && taps->detail_sink != nullptr; }
    bool patching() const { return taps != nullptr && taps->patch != nullptr; }
    bool hooked() const { return detail() || patching(); }

    // "<root>.layers.L.<suffix>", or "<root>.<suffix>" for layer kLayers (the final mixer).
    static void format(char (&name)[112], const char *root, uint32_t layer, const char *suffix) {
        if (layer < kLayers) {
            std::snprintf(name, sizeof(name), "%s.layers.%u.%s", root, layer, suffix);
        } else {
            std::snprintf(name, sizeof(name), "%s.%s", root, suffix);
        }
    }

    // E: one B8 tensor, the last `rows` of a [n, 2,560] buffer.
    cere_status emit(uint32_t layer, const char *suffix, const bf16 *values) const {
        if (!emitting()) return CERE_OK;
        char name[112];
        if (layer < kLayers) {
            format(name, "model", layer, suffix);
        } else {
            std::snprintf(name, sizeof(name), "%s", suffix);
        }
        return taps->emit(taps->ctx, name, values + (n - rows) * uint64_t(kHidden), rows, kHidden, stream);
    }

    // attn_inputs_only: "model.layers.L.self_attn.input", the last `rows` of ws.mixed before full attention L.
    cere_status attn_input(uint32_t layer, const bf16 *values) const {
        if (taps == nullptr || !taps->attn_inputs_only) return CERE_OK;
        char name[112];
        format(name, "model", layer, "self_attn.input");
        return taps->emit(taps->ctx, name, values + (n - rows) * uint64_t(kHidden), rows, kHidden, stream);
    }

    // D: the last `rows` of a [n, cols] buffer.
    cere_status detail(const char *root, uint32_t layer, const char *suffix, const bf16 *values, uint64_t cols) const {
        if (!detail()) return CERE_OK;
        char name[112];
        format(name, root, layer, suffix);
        return taps->detail_sink(taps->ctx, name, values + (n - rows) * cols, rows, cols, stream);
    }

    // I: a native.* input point of [n, cols]; the callback's return is ignored.
    cere_status input(uint32_t layer, const char *suffix, const bf16 *values, uint64_t cols) const {
        if (!patching()) return CERE_OK;
        char name[112];
        format(name, "native", layer, suffix);
        taps->patch(taps->ctx, name, layer, n, cols, values, stream);
        return CERE_OK;
    }

    // P: a replacement of the [n, cols] tensor `values` is copied over it in stream order (`*replaced` set).
    cere_status point(const char *root, uint32_t layer, const char *suffix, bf16 *values, uint64_t cols,
                      bool *replaced = nullptr) const {
        if (!patching()) return CERE_OK;
        char name[112];
        format(name, root, layer, suffix);
        const bf16 *r = taps->patch(taps->ctx, name, layer, n, cols, values, stream);
        if (r == nullptr) return CERE_OK;
        if (replaced != nullptr) *replaced = true;
        if (r == values) return CERE_OK;
        CERE_TRY_CUDA(cudaMemcpyAsync(values, r, uint64_t(n) * cols * sizeof(bf16), cudaMemcpyDeviceToDevice, stream));
        // The pointer only has to outlive the callback that returned it (model.h).
        return cuda_status(cudaStreamSynchronize(stream));
    }
};

// hc_mix of ws.hyper into ws.normed and ws.mixed; with `normed`, ws.normed already holds the mix's norm (the
// combine before it fused it, combine below) and the GEMMs and gate run alone.
cere_status run_mix(const kernels::HcPlans &plans, const Workspace &ws, const HcWeights &hc, uint32_t m, bool normed,
                    cudaStream_t stream, uint32_t cls = 0) {
    if (normed) {
        return kernels::hc_mix_normed(plans, ws.normed, hc.down, hc.up, m, ws.mixed, ws.scratch, stream,
                                      kernels::HcMixPath::kAuto, cls);
    }
    return kernels::hc_mix(plans, ws.hyper, hc.norm, hc.down, hc.up, m, ws.normed, ws.mixed, ws.scratch, stream,
                           kernels::HcMixPath::kAuto, cls);
}

// A GatedResidual.mix over ws.hyper (the attn / mlp mix of `layer`, or the final mixer for layer kLayers): ws.normed
// (kept for combine) and ws.mixed (the block input), with its D / I / P points. `normed` as run_mix.
cere_status mix(const Workspace &ws, const HcWeights &hc, uint32_t layer, const char *which, const Tap &tap,
                bool normed) {
    const auto run = [&] { return run_mix(ws.hc, ws, hc, tap.n, normed, tap.stream, tap.cls); };
    if (!tap.hooked()) return run();
    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), "%s_mix.in", which);
    CERE_TRY(tap.detail("native", layer, suffix, ws.hyper, kHcHidden));
    CERE_TRY(run());
    if (layer < kLayers) {
        std::snprintf(suffix, sizeof(suffix), "%s_hyper_connection.hc_norm", which);
        CERE_TRY(tap.detail("model", layer, suffix, ws.normed, kHcHidden));
    } else {
        CERE_TRY(tap.detail("native", layer, "final_mix.normed", ws.normed, kHcHidden));
    }
    std::snprintf(suffix, sizeof(suffix), "%s_mix.normed", which);
    CERE_TRY(tap.input(layer, suffix, ws.normed, kHcHidden));
    std::snprintf(suffix, sizeof(suffix), "%s_mix.out", which);
    CERE_TRY(tap.point("native", layer, suffix, ws.mixed, kHidden));
    if (layer >= kLayers) return CERE_OK;
    const bool attn = which[0] == 'a';
    CERE_TRY(tap.detail("native", layer, attn ? "block.in" : "mlp.in", ws.mixed, kHidden));
    if (!attn) CERE_TRY(tap.input(layer, "mlp.in", ws.mixed, kHidden));
    return CERE_OK;
}

// GatedResidual.combine of ws.block_out into ws.hyper (in place). With next_norm (the norm weight of the mix that
// follows) it also writes that mix's per-stream norm of the new ws.hyper into ws.normed (kernels::hc_combine_norm,
// m23 design note P5R, bit-identical), and the mix runs normed.
cere_status combine(const Workspace &ws, const HcWeights &hc, uint32_t n, const bf16 *next_norm, cudaStream_t stream) {
    if (next_norm == nullptr) {
        return kernels::hc_combine(ws.hc.shape, ws.block_out, ws.hyper, ws.normed, hc.inject, n, ws.hyper, stream);
    }
    return kernels::hc_combine_norm(ws.hc.shape, ws.block_out, ws.hyper, ws.normed, hc.inject, next_norm, n, ws.hyper,
                                    stream);
}

// The norm weight of the mix after layer L's mlp combine: layer L + 1's attn mix, or the final mixer's. None before
// the PLE layer, whose k4::forward rewrites ws.hyper between the two.
const bf16 *norm_after(const ModelWeights &w, uint32_t L) {
    if (L + 1 == kPleLayer) return nullptr;
    return L + 1 < kLayers ? w.layers[L + 1].attn_hc.norm : w.final_mixer.norm;
}

// SelfTune D2 (model/plan.h): the decode plan the single-session forward (DECODE, and every PREFILL chunk and MixedPD
// segment since ST1d), the batched decode forward and the verify forward walk. Its first op is embed.tokens, its last
// two the final norm and lm_head after every layer, and its slots are the layer kinds the support table compiles
// (executor/tests/host/test_plan.cpp checks the attention slots against full_attention_layers).
static_assert(plan::kDecode.size > 2 && plan::kDecode.ops[0].slot == plan::Slot::kEmbedTokens &&
                  plan::kDecode.ops[plan::kDecode.size - 2].slot == plan::Slot::kNorm &&
                  plan::kDecode.ops[plan::kDecode.size - 1].slot == plan::Slot::kLmHead &&
                  plan::kDecode.ops[plan::kDecode.size - 1].layer == kLayers,
              "the decode plan brackets the model's layers");

// The HC weights of a plan op's mix: layer L's attn or mlp mix, or the final mixer.
const HcWeights &plan_hc(const ModelWeights &w, uint32_t layer, plan::Mix mix) {
    if (mix == plan::Mix::kFinal) return w.final_mixer;
    return mix == plan::Mix::kAttn ? w.layers[layer].attn_hc : w.layers[layer].mlp_hc;
}

// The norm a combine op also writes into ws.normed (the hc_combine_norm rule fused it with the mix after it), or none.
const bf16 *fused_norm(const ModelWeights &w, const plan::Op &op) {
    return op.fused ? plan_hc(w, op.target_layer, op.target_mix).norm : nullptr;
}

// One K2 projection y = x w^T with its I / P points (native.layers.L.gemm.<plan>.{in,out}).
cere_status gemm(const kernels::GemmPlan &plan, const char *plan_name, const bf16 *x, const bf16 *w, bf16 *y,
                 uint32_t layer, const Tap &tap) {
    if (!tap.patching()) return plan.run(x, w, tap.n, y, tap.stream, tap.cls);
    char suffix[48];
    std::snprintf(suffix, sizeof(suffix), "gemm.%s.in", plan_name);
    CERE_TRY(tap.input(layer, suffix, x, plan.k()));
    CERE_TRY(plan.run(x, w, tap.n, y, tap.stream, tap.cls));
    std::snprintf(suffix, sizeof(suffix), "gemm.%s.out", plan_name);
    return tap.point("native", layer, suffix, y, plan.n());
}

// WeightLab research (research/hook.h): ws.research_call (null without a planned row in the call: every helper is then
// one pointer test) at a point of `layer` over call rows [row0, row0 + n) of the point's native tensor [n, cols].
using research::PointId;

bool researching(const Workspace &ws, PointId point, uint32_t layer) {
    return ws.research_call != nullptr && ws.research_call->acts(point, layer);
}

// `*edited` (nullable): set when the plan edited some row. An edit at native.layers.L.{attn,mlp}_mix.in (ws.hyper)
// makes the norm combine fused into ws.normed stale, so the mix after it recomputes it (normed false).
cere_status research_at(const Workspace &ws, PointId point, uint32_t layer, uint32_t row0, uint32_t n, bf16 *values,
                        uint32_t cols, cudaStream_t stream, bool *edited = nullptr) {
    if (!researching(ws, point, layer)) return CERE_OK;
    return ws.research_call->at(point, layer, row0, n, values, cols, stream, edited);
}

// The MoE block under the research hook: MoeTaps.point answers only "gemm.moe_router.out" (native.layers.L.
// gemm.moe_router.out) and "gemm.moe_shared_down.in" (native.layers.L.gemm.moe_shared_down.in, capture only): K7's
// hooked path, the shared expert then runs in stream rather than forked; MoeTaps.route the routing
// (native.layers.L.moe_route, the path unchanged), and MoeTaps.routed / shared expose model.layers.L.
// mlp.experts / mlp.shared_expert.down_proj; an edit there reruns only the combine (MoeTaps.*_in), as moe()'s patch.
struct ResearchMoe {
    research::Call *call;
    uint32_t layer;
    uint32_t row0;
};

cere_status research_moe_point(void *ctx, const char *point, bf16 *values, uint32_t rows, uint32_t cols,
                               cudaStream_t stream) {
    const ResearchMoe &h = *static_cast<const ResearchMoe *>(ctx);
    if (std::strcmp(point, "gemm.moe_router.out") == 0) {
        return h.call->at(PointId::kRouterOut, h.layer, h.row0, rows, values, cols, stream);
    }
    if (std::strcmp(point, "gemm.moe_shared_down.in") == 0) {
        return h.call->at(PointId::kSharedDownIn, h.layer, h.row0, rows, values, cols, stream);
    }
    return CERE_OK;
}

cere_status research_moe_route(void *ctx, int32_t *ids, float *weights, uint32_t rows, uint32_t top_k,
                               cudaStream_t stream) {
    const ResearchMoe &h = *static_cast<const ResearchMoe *>(ctx);
    return h.call->route(h.layer, h.row0, rows, ids, weights, top_k, stream);
}

// Call rows [row0, row0 + m): x [m, 2,560] -> out, `routed` / `shared` [m, 2,560] the MoeTaps outputs.
cere_status research_moe(const LayerWeights &lw, const Workspace &ws, const kernels::MoePlans &plans, uint32_t layer,
                         uint32_t row0, uint32_t m, const bf16 *x, bf16 *out, bf16 *routed, bf16 *shared,
                         cudaStream_t stream) {
    research::Call &call = *ws.research_call;
    ResearchMoe hook{&call, layer, row0};
    kernels::MoeTaps taps;
    const bool outputs = call.acts(PointId::kExperts, layer) || call.acts(PointId::kSharedExpert, layer);
    if (outputs) {
        taps.routed = routed;
        taps.shared = shared;
    }
    if (call.acts(PointId::kRouterOut, layer) || call.acts(PointId::kSharedDownIn, layer)) {
        taps.point = &research_moe_point;
    }
    if (call.acts(PointId::kRoute, layer)) taps.route = &research_moe_route;
    taps.ctx = &hook;
    const bool tapped = outputs || taps.point != nullptr || taps.route != nullptr;
    CERE_TRY(CERE_MOE_BLOCK(kMoeDims, lw.moe, plans, x, m, out, ws.scratch, ws.scratch_bytes, stream,
                            tapped ? &taps : nullptr));
    if (!outputs) return CERE_OK;
    bool edited = false;
    CERE_TRY(call.at(PointId::kExperts, layer, row0, m, routed, kHidden, stream, &edited));
    CERE_TRY(call.at(PointId::kSharedExpert, layer, row0, m, shared, kHidden, stream, &edited));
    if (!edited) return CERE_OK;
    kernels::MoeTaps given;
    given.routed_in = routed;
    given.shared_in = shared;
    return CERE_MOE_BLOCK(kMoeDims, lw.moe, plans, x, m, out, ws.scratch, ws.scratch_bytes, stream, &given);
}

}  // namespace

// Qwen3_5 attention between the HC mix and the combine (m22c D2): qkv_proj (packed q|gate, k, v) -> K6 prologue (QK
// norm, RoPE, gate, FP8 KV append at the session position) -> dense causal GQA (K6, prefix + n <= 2,048) or the K9 QSA
// path (indexer select + sparse GQA) -> sigmoid gate -> o_proj into ws.block_out. In a context above 2,048 every call
// from position 0 also projects the index keys and stores the compressed keys and the ring, so a later call above
// 2,048 sees every complete block. Scratch carving: model.cpp qsa_attention_scratch_bytes.
//
// attention_rows is the part between the projections for n rows of one session at positions prefix .. prefix + n - 1:
// `qkv` [n, 13,312] (q|gate, k, v), `mixed` [n, 2,560] (the indexer input, contexts above 2,048) -> `attn_out`
// [n, 6,144] (the o_proj input), on the layer's KV state `kv` (model.h AttnKv), rotated as `rope` says (model.h Rope:
// a text call's Rope{w.rope_cos_sin} gives the text launches argument for argument).
cere_status attention_rows(const ModelWeights &w, const AttentionWeights &aw, const Workspace &ws, const AttnKv &kv,
                           int64_t prefix, bf16 *qkv, const bf16 *mixed, bf16 *attn_out, uint32_t n, const Rope &rope,
                           cudaStream_t stream, uint32_t cls, const EndKv *end_kv) {
    const int64_t tokens = int64_t(n);
    const bf16 *k = qkv + kQGateWidth;
    const bf16 *v = k + kKvWidth;
    const float eps = model_eps(w.config);
    // KV rows and compressed-key rows of the paged pool through the session's slot map (m25 design note D3).
    const int64_t *slots = kv.slots;
    // A class pin (m210 D15⁗′) covers the K9 path and the dense K6 rows of a chunk-0 end state's resume (EndKv);
    // any other dense K6 call chooses by its own rows.
    const bool pinned = cls != 0 && cls != n;
    if (ws.max_context_tokens <= uint64_t(k6::kMaxContext)) {
        if (pinned || end_kv != nullptr) return CERE_ERR_UNSUPPORTED;
        CERE_TRY_CUDA(k6::full_attention(qkv, kQkvWidth, k, kQkvWidth, v, kQkvWidth, aw.q_norm,
                                         aw.k_norm, eps, rope.table, nullptr, rope.positions3, slots, tokens,
                                         prefix,
                                         prefix > 0, kv.k_cache, kv.v_cache, ws.scratch,
                                         ws.scratch_bytes, attn_out, kAttnOutWidth, stream, ws.prefill_p4));
        return CERE_OK;
    }
    const size_t t = size_t(n);
    uint8_t *p = static_cast<uint8_t *>(ws.scratch);
    bf16 *q = reinterpret_cast<bf16 *>(p);
    p += align256(t * kAttnOutWidth * sizeof(bf16));
    bf16 *k_rot = reinterpret_cast<bf16 *>(p);
    p += align256(t * kKvWidth * sizeof(bf16));
    bf16 *gate = reinterpret_cast<bf16 *>(p);
    p += align256(t * kAttnOutWidth * sizeof(bf16));
    bf16 *qk = reinterpret_cast<bf16 *>(p);
    p += align256(t * k9::kQkWidth * sizeof(bf16));
    const size_t rest = ws.scratch_bytes - size_t(p - static_cast<uint8_t *>(ws.scratch));
    // EndKv, checked before any launch: dense rows only; a resume right after its saved rows; a capture strictly
    // inside its call, at prefix 0 or after a resume; room in the scratch for a resume's row sets ahead of the dense
    // workspace.
    size_t end_bytes = 0;
    if (end_kv != nullptr) {
        const bool capture = end_kv->k != nullptr;
        const bool resume = end_kv->in_k != nullptr;
        const bool capture_ok = !capture || (end_kv->rows > prefix && end_kv->rows < prefix + tokens &&
                                             (prefix == 0 || resume) && end_kv->v != nullptr);
        const bool resume_ok = !resume || (end_kv->in_rows > 0 && prefix == end_kv->in_rows && end_kv->in_v != nullptr);
        if (prefix + tokens > int64_t(k6::kMaxContext) || !(capture || resume) || !capture_ok || !resume_ok) {
            return CERE_ERR_ARGUMENT;
        }
        if (resume) end_bytes = 2 * align256((size_t(prefix) + t) * kKvWidth * sizeof(bf16));
        if (rest < end_bytes + k6::dense_attention_workspace_bytes(tokens)) return CERE_ERR_ARGUMENT;
    }

    // Index keys of this call -> the ring and the compressed keys (every call, from position 0); an image prompt's
    // rows through K9's mRoPE twins (m213l design note C5).
    CERE_TRY(ws.attn_index_qk.run(mixed, aw.indexer_qk, n, qk, stream, cls));
    const bf16 *k_idx = qk + k9::kIndexHeads * k9::kIndexHeadDim;
    const k9::Bounds bounds = k9_bounds(w, ws, rope.table, uint64_t(kv.context));
    if (rope.k9 != nullptr) {
        CERE_TRY(k9::qsa_compress_store_mrope(k_idx, k9::kQkWidth, aw.indexer_k_norm, eps, rope.table, bounds, rope.k9,
                                              rope.k9_stride, rope.k9_lead, slots, prefix, tokens, kv.qsa_ring,
                                              kv.qsa_keys, stream));
    } else {
        CERE_TRY(k9::qsa_compress_store(k_idx, k9::kQkWidth, aw.indexer_k_norm, eps, rope.table, bounds, nullptr,
                                        slots, prefix, tokens, kv.qsa_ring, kv.qsa_keys, stream));
    }
    CERE_TRY_CUDA(k6::prologue(qkv, kQkvWidth, k, kQkvWidth, v, kQkvWidth, aw.q_norm, aw.k_norm, eps,
                               rope.table, int64_t(w.max_context_tokens), nullptr, rope.positions3, slots, tokens,
                               prefix,
                               kv.k_cache, kv.v_cache, q, k_rot, gate, stream));
    const bool dense = prefix + tokens <= int64_t(k6::kMaxContext);
    if (dense) {
        const size_t row_bytes = kKvWidth * sizeof(bf16);
        if (end_kv == nullptr || end_kv->in_k == nullptr) {
            if (pinned) return CERE_ERR_UNSUPPORTED;
            if (end_kv != nullptr) {
                // Capture: the cold call's own rows [0, rows), as its dense attention reads them.
                const size_t saved = size_t(end_kv->rows);
                CERE_TRY_CUDA(
                    cudaMemcpyAsync(end_kv->k, k_rot, saved * row_bytes, cudaMemcpyDeviceToDevice, stream));
                CERE_TRY_CUDA(cudaMemcpy2DAsync(end_kv->v, row_bytes, v, kQkvWidth * sizeof(bf16), row_bytes, saved,
                                                cudaMemcpyDeviceToDevice, stream));
            }
            CERE_TRY_CUDA(k6::dense_attention(q, k_rot, v, kQkvWidth, gate, tokens, prefix, prefix > 0,
                                              kv.k_cache, kv.v_cache, slots, p, rest,
                                              attn_out, kAttnOutWidth, stream, ws.prefill_p4));
            return CERE_OK;
        }
        // Resume: the saved rows [0, prefix) and this call's rows after them, BF16 as the cold call had them.
        const size_t saved = size_t(end_kv->in_rows);
        const size_t all = saved + t;
        bf16 *k_all = reinterpret_cast<bf16 *>(p);
        p += align256(all * row_bytes);
        bf16 *v_all = reinterpret_cast<bf16 *>(p);
        p += align256(all * row_bytes);
        CERE_TRY_CUDA(cudaMemcpyAsync(k_all, end_kv->in_k, saved * row_bytes, cudaMemcpyDeviceToDevice, stream));
        CERE_TRY_CUDA(cudaMemcpyAsync(k_all + saved * kKvWidth, k_rot, t * row_bytes, cudaMemcpyDeviceToDevice,
                                      stream));
        CERE_TRY_CUDA(cudaMemcpyAsync(v_all, end_kv->in_v, saved * row_bytes, cudaMemcpyDeviceToDevice, stream));
        CERE_TRY_CUDA(cudaMemcpy2DAsync(v_all + saved * kKvWidth, row_bytes, v, kQkvWidth * sizeof(bf16), row_bytes,
                                        t, cudaMemcpyDeviceToDevice, stream));
        if (end_kv->k != nullptr) {
            // D15⁗″: a capture past the resume: the rows [0, rows) of the same run, saved ones then its own.
            const size_t rows = size_t(end_kv->rows);
            CERE_TRY_CUDA(cudaMemcpyAsync(end_kv->k, k_all, rows * row_bytes, cudaMemcpyDeviceToDevice, stream));
            CERE_TRY_CUDA(cudaMemcpyAsync(end_kv->v, v_all, rows * row_bytes, cudaMemcpyDeviceToDevice, stream));
        }
        CERE_TRY_CUDA(k6::dense_attention(q, k_all, v_all, kKvWidth, gate, tokens, prefix, false, kv.k_cache,
                                          kv.v_cache, slots, p, rest - end_bytes, attn_out, kAttnOutWidth, stream,
                                          ws.prefill_p4, cls));
        return CERE_OK;
    }
    bf16 *attn = reinterpret_cast<bf16 *>(p);
    p += align256(t * kAttnOutWidth * sizeof(bf16));
    const size_t k9_bytes = ws.scratch_bytes - size_t(p - static_cast<uint8_t *>(ws.scratch));
    k9::Scratch s{};
    CERE_TRY(k9::carve_scratch(p, k9_bytes, tokens, int64_t(ws.max_context_tokens), &s));
    if (rope.k9 != nullptr) {
        CERE_TRY(k9::indexer_qk_mrope(qk, k9::kQkWidth, aw.indexer_q_norm, eps, rope.table, bounds, rope.k9,
                                      rope.k9_stride, rope.k9_lead, tokens, s.q_idx, stream));
    } else {
        CERE_TRY(k9::indexer_qk(qk, k9::kQkWidth, aw.indexer_q_norm, eps, rope.table, bounds, nullptr, prefix, tokens,
                                s.q_idx, stream));
    }
    CERE_TRY(k9::qsa_select(s.q_idx, kv.qsa_keys, slots, nullptr, tokens, prefix + tokens, s.work,
                            s.work_bytes, s.indices, stream));
    // K/V source as K6 (spec D3): the call's own BF16 rows without a prefix, else the FP8 cache.
    if (prefix == 0) {
        CERE_TRY(k9::qsa_attention(q, kAttnOutWidth, s.indices, tokens, k_rot, kKvWidth, v, kQkvWidth, nullptr,
                                   nullptr, nullptr, k9::kSoftmaxScale, s.work, s.work_bytes, attn, kAttnOutWidth,
                                   stream, ws.prefill_p4, cls));
    } else {
        CERE_TRY(k9::qsa_attention(q, kAttnOutWidth, s.indices, tokens, nullptr, 0, nullptr, 0,
                                   kv.k_cache, kv.v_cache, slots, k9::kSoftmaxScale,
                                   s.work, s.work_bytes, attn, kAttnOutWidth, stream, ws.prefill_p4, cls));
    }
    CERE_TRY_CUDA(k1::sigmoid_mul(attn, kAttnOutWidth, gate, kAttnOutWidth, tokens, kAttnOutWidth, attn_out,
                                  kAttnOutWidth, stream));
    return CERE_OK;
}

cere_status attention_rows_at(const ModelWeights &w, const AttentionWeights &aw, const Workspace &ws, const AttnKv &kv,
                              const int64_t *positions, bf16 *qkv, const bf16 *mixed, bf16 *attn_out, uint32_t n,
                              cudaStream_t stream) {
    const int64_t tokens = int64_t(n);
    const bf16 *k = qkv + kQGateWidth;
    const bf16 *v = k + kKvWidth;
    const float eps = model_eps(w.config);
    const int64_t *slots = kv.slots;
    // attention_rows' carving: q, k_rot, gate (k6::full_attention's on the dense plan), then the indexer's qk.
    const size_t t = size_t(n);
    uint8_t *p = static_cast<uint8_t *>(ws.scratch);
    bf16 *q = reinterpret_cast<bf16 *>(p);
    p += align256(t * kAttnOutWidth * sizeof(bf16));
    bf16 *k_rot = reinterpret_cast<bf16 *>(p);
    p += align256(t * kKvWidth * sizeof(bf16));
    bf16 *gate = reinterpret_cast<bf16 *>(p);
    p += align256(t * kAttnOutWidth * sizeof(bf16));
    if (ws.max_context_tokens > uint64_t(k6::kMaxContext)) {
        bf16 *qk = reinterpret_cast<bf16 *>(p);
        p += align256(t * k9::kQkWidth * sizeof(bf16));
        CERE_TRY(ws.attn_index_qk.run(mixed, aw.indexer_qk, n, qk, stream));
        CERE_TRY(k9::qsa_compress_store_at(qk + k9::kIndexHeads * k9::kIndexHeadDim, k9::kQkWidth, aw.indexer_k_norm,
                                           eps, w.rope_cos_sin, k9_bounds(w, ws, w.rope_cos_sin, uint64_t(kv.context)),
                                           slots, positions, tokens, kv.qsa_ring, kv.qsa_keys, stream));
    }
    const size_t rest = ws.scratch_bytes - size_t(p - static_cast<uint8_t *>(ws.scratch));
    CERE_TRY_CUDA(k6::prologue_rows_at(qkv, kQkvWidth, k, kQkvWidth, v, kQkvWidth, aw.q_norm, aw.k_norm, eps,
                                       w.rope_cos_sin, slots, positions, tokens, kv.k_cache, kv.v_cache, q, k_rot,
                                       gate, stream));
    CERE_TRY_CUDA(k6::dense_attention_at(q, gate, tokens, positions, kv.k_cache, kv.v_cache, slots, p, rest, attn_out,
                                         kAttnOutWidth, stream));
    return CERE_OK;
}

namespace {

cere_status copy_d2d(void *dst, const void *src, size_t bytes, cudaStream_t stream) {
    return cuda_status(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
}

// attention_core: attention_rows on target layer `layer` of `session` at the session position. full_attention runs it
// on the call's rows; forward_decode_rows on one row of each session (D5).
cere_status attention_core(const ModelWeights &w, const LayerWeights &lw, const Workspace &ws, SessionState &session,
                           uint32_t layer, bf16 *qkv, const bf16 *mixed, bf16 *attn_out, uint32_t n, const Rope &rope,
                           cudaStream_t stream, uint32_t cls = 0, const EndKv *end_kv = nullptr) {
    const AttnKv kv{session.k_cache[layer], session.v_cache[layer], session.qsa_ring[layer], session.qsa_keys[layer],
                    session.slot_map, int64_t(session.max_context_tokens)};
    return attention_rows(w, lw.attn, ws, kv, int64_t(session.position), qkv, mixed, attn_out, n, rope, stream, cls,
                          end_kv);
}

// True when a decode step of `session` (one token at session.position) attends densely: K6 over the FP8 cache. Above
// it a plan with a context above 2,048 runs the K9 QSA path (attention_core's `dense`).
bool decode_dense(const Workspace &ws, const SessionState &session) {
    return ws.max_context_tokens <= uint64_t(k6::kMaxContext) || session.position + 1 <= uint64_t(k6::kMaxContext);
}

// The decode graph's attention of one row (m23 design note P1, M23P1-03/04): attention_core at n = 1 and a prefix
// above 0 with every position-dependent launch reading the row's device position `position` (ws.ids64 + R + r, the
// tail after the R ids: layout::decode_graph_fits) instead of the host scalar, so a captured step replays at any
// position. Launch for launch the kernels of attention_core, bit for bit: each step kernel computes from *position
// exactly what its eager twin computes from session.position. The dense / sparse choice stays a host branch
// (decode_dense); it is part of the graph key (DecodeGraph::key), so a replay never crosses it. Only for a session at
// rope_delta 0 (the graphs' gates): the model's table.
cere_status attention_core_step(const ModelWeights &w, const LayerWeights &lw, const Workspace &ws,
                                SessionState &session, uint32_t layer, bf16 *qkv, const bf16 *mixed, bf16 *attn_out,
                                const int64_t *position, cudaStream_t stream) {
    const bf16 *k = qkv + kQGateWidth;
    const bf16 *v = k + kKvWidth;
    const float eps = model_eps(w.config);
    const int64_t *slots = session.slot_map;
    if (ws.max_context_tokens <= uint64_t(k6::kMaxContext)) {
        CERE_TRY_CUDA(k6::full_attention_step(qkv, kQkvWidth, k, kQkvWidth, v, kQkvWidth, lw.attn.q_norm,
                                              lw.attn.k_norm, eps, w.rope_cos_sin, slots, position,
                                              session.k_cache[layer], session.v_cache[layer], ws.scratch,
                                              ws.scratch_bytes, attn_out, kAttnOutWidth, stream));
        return CERE_OK;
    }
    uint8_t *p = static_cast<uint8_t *>(ws.scratch);
    bf16 *q = reinterpret_cast<bf16 *>(p);
    p += align256(kAttnOutWidth * sizeof(bf16));
    bf16 *k_rot = reinterpret_cast<bf16 *>(p);
    p += align256(kKvWidth * sizeof(bf16));
    bf16 *gate = reinterpret_cast<bf16 *>(p);
    p += align256(kAttnOutWidth * sizeof(bf16));
    bf16 *qk = reinterpret_cast<bf16 *>(p);
    p += align256(k9::kQkWidth * sizeof(bf16));
    const size_t rest = ws.scratch_bytes - size_t(p - static_cast<uint8_t *>(ws.scratch));

    CERE_TRY(ws.attn_index_qk.run(mixed, lw.attn.indexer_qk, 1, qk, stream));
    const k9::Bounds bounds = k9_bounds(w, ws, w.rope_cos_sin, session.max_context_tokens);
    CERE_TRY(k9::qsa_compress_store_step(qk + k9::kIndexHeads * k9::kIndexHeadDim, lw.attn.indexer_k_norm, eps,
                                         w.rope_cos_sin, bounds, slots, position, session.qsa_ring[layer],
                                         session.qsa_keys[layer], stream));
    CERE_TRY_CUDA(k6::prologue_step(qkv, kQkvWidth, k, kQkvWidth, v, kQkvWidth, lw.attn.q_norm, lw.attn.k_norm, eps,
                                    w.rope_cos_sin, slots, position, session.k_cache[layer], session.v_cache[layer],
                                    q, k_rot, gate, stream));
    if (decode_dense(ws, session)) {
        CERE_TRY_CUDA(k6::dense_attention_step(q, gate, session.k_cache[layer], session.v_cache[layer], slots,
                                               position, p, rest, attn_out, kAttnOutWidth, stream));
        return CERE_OK;
    }
    bf16 *attn = reinterpret_cast<bf16 *>(p);
    p += align256(kAttnOutWidth * sizeof(bf16));
    const size_t k9_bytes = ws.scratch_bytes - size_t(p - static_cast<uint8_t *>(ws.scratch));
    k9::Scratch s{};
    CERE_TRY(k9::carve_scratch(p, k9_bytes, 1, int64_t(ws.max_context_tokens), &s));
    CERE_TRY(k9::indexer_qk(qk, k9::kQkWidth, lw.attn.indexer_q_norm, eps, w.rope_cos_sin, bounds, position, 0, 1,
                            s.q_idx, stream));
    CERE_TRY(k9::qsa_select_step(s.q_idx, session.qsa_keys[layer], slots, bounds, position,
                                 int64_t(ws.max_context_tokens), s.work, s.work_bytes, s.indices, stream));
    CERE_TRY(k9::qsa_attention(q, kAttnOutWidth, s.indices, 1, nullptr, 0, nullptr, 0, session.k_cache[layer],
                               session.v_cache[layer], slots, k9::kSoftmaxScale, s.work, s.work_bytes, attn,
                               kAttnOutWidth, stream, ws.prefill_p4));
    CERE_TRY_CUDA(k1::sigmoid_mul(attn, kAttnOutWidth, gate, kAttnOutWidth, 1, kAttnOutWidth, attn_out, kAttnOutWidth,
                                  stream));
    return CERE_OK;
}

// `position` (device, n = 1): the decode graph's step (attention_core_step); null: the eager call rotated by `rope`.
cere_status full_attention(const ModelWeights &w, const LayerWeights &lw, const Workspace &ws, SessionState &session,
                           uint32_t layer, const Rope &rope, const int64_t *position, const Tap &tap) {
    CERE_TRY(gemm(ws.attn_qkv, "attn_qkv", ws.mixed, lw.attn.qkv, ws.qkv, layer, tap));
    CERE_TRY(research_at(ws, PointId::kAttnQkv, layer, 0, tap.n, ws.qkv, kQkvWidth, tap.stream));
    CERE_TRY(tap.detail("model", layer, "self_attn.qkv_proj", ws.qkv, kQkvWidth));
    if (position != nullptr) {
        CERE_TRY(attention_core_step(w, lw, ws, session, layer, ws.qkv, ws.mixed, ws.attn_out, position, tap.stream));
    } else {
        // A chunk-0 end state's attention rows (EndCall::kv, in_kv): its capture and / or resume of this layer's BF16
        // K/V.
        const bool end_kv = tap.end != nullptr && (tap.end->kv != nullptr || tap.end->in_kv != nullptr);
        const EndKv kv = end_kv ? model::end_kv(*tap.end, end_kv_index(w.config, layer)) : EndKv{};
        CERE_TRY(attention_core(w, lw, ws, session, layer, ws.qkv, ws.mixed, ws.attn_out, tap.n, rope, tap.stream,
                                tap.cls, end_kv ? &kv : nullptr));
    }
    return gemm(ws.attn_o, "attn_o", ws.attn_out, lw.attn.o_proj, ws.block_out, layer, tap);
}

// The GDN mixer into ws.block_out: k5::layer_forward, or with detail / patch taps (or a research plan acting at
// native.layers.L.gemm.gdn_{qkv,z}.out) the same launches spelled out over layer_forward's workspace layout
// (k5_gdn_layer.h layer_workspace_bytes: mixed [n, 10,240] then y [n, 6,144], z [n, 6,144], b and a [n, 48], then the
// core workspace), with the points between them. The call's EndCall (Tap) becomes the layer's k5::EndState over the
// slot views: a resume reads the FP32 SSM of the `in` pair, a capture writes the conv state and FP32 SSM to the out
// pair.
cere_status gdn(const LayerWeights &lw, const Workspace &ws, SessionState &session, uint32_t layer, uint32_t operation,
                const Tap &tap) {
    const uint32_t n = tap.n;
    k5::EndState end_state;
    const k5::EndState *end = nullptr;
    if (tap.end != nullptr && (tap.end->in_a != nullptr || tap.end->row > 0)) {
        if (tap.end->in_a != nullptr) {
            end_state.ssm_in32 = reinterpret_cast<const float *>(tap.end->in_a->gdn[layer].ssm);
            end_state.ssm_in32_hi = reinterpret_cast<const float *>(tap.end->in_b->gdn[layer].ssm);
        }
        if (tap.end->row > 0) {
            const k5::LayerState &a = tap.end->a->gdn[layer];
            end_state.ssm_out32 = reinterpret_cast<float *>(a.ssm);
            end_state.ssm_out32_hi = reinterpret_cast<float *>(tap.end->b->gdn[layer].ssm);
            end_state.conv_out = a.conv;
            end_state.row = tap.end->row;
        }
        end = &end_state;
    }
    const bool research = researching(ws, PointId::kGdnQkv, layer) || researching(ws, PointId::kGdnZ, layer);
    if (!tap.hooked() && !research) {
        const k5::LayerGemms gemms{&ws.gdn_qkv, &ws.gdn_z, &ws.gdn_ba, &ws.gdn_out};
        return k5::layer_forward(gemms, lw.gdn, session.gdn[layer], operation, ws.mixed, n, kHidden, ws.block_out,
                                 ws.scratch, ws.scratch_bytes, tap.stream, tap.cls, end);
    }
    constexpr uint64_t kAlign = 256;
    uint8_t *const base = static_cast<uint8_t *>(ws.scratch);
    uint64_t at = 0;
    const auto take = [&](uint64_t bytes) {
        uint8_t *p = base + at;
        at += (bytes + kAlign - 1) / kAlign * kAlign;
        return reinterpret_cast<bf16 *>(p);
    };
    bf16 *mixed = take(uint64_t(n) * k5::kConvDim * sizeof(bf16));
    bf16 *z = take(uint64_t(n) * k5::kValueDim * sizeof(bf16));
    bf16 *b = take(uint64_t(n) * k5::kValueHeads * sizeof(bf16));
    bf16 *a = take(uint64_t(n) * k5::kValueHeads * sizeof(bf16));
    bf16 *core = take(0);
    const size_t core_bytes = k5::core_workspace_bytes(n);
    constexpr uint64_t kQkvzWidth = k5::kConvDim + k5::kValueDim;  // 16,384
    // The core workspace (conv_out [n, 10,240] + o [n, 6,144] at least) stages the qkv | z and b | a details.
    if (core_bytes == 0 || at + core_bytes > ws.scratch_bytes || core_bytes < uint64_t(n) * kQkvzWidth * sizeof(bf16)) {
        return CERE_ERR_ARGUMENT;
    }
    CERE_TRY(gemm(ws.gdn_qkv, "gdn_qkv", ws.mixed, lw.gdn.in_proj_qkv, mixed, layer, tap));
    CERE_TRY(gemm(ws.gdn_z, "gdn_z", ws.mixed, lw.gdn.in_proj_z, z, layer, tap));
    CERE_TRY(gemm(ws.gdn_ba, "gdn_ba", ws.mixed, lw.gdn.in_proj_b, b, layer, tap));
    CERE_TRY(gemm(ws.gdn_ba, "gdn_ba", ws.mixed, lw.gdn.in_proj_a, a, layer, tap));
    CERE_TRY(research_at(ws, PointId::kGdnQkv, layer, 0, n, mixed, k5::kConvDim, tap.stream));
    CERE_TRY(research_at(ws, PointId::kGdnZ, layer, 0, n, z, k5::kValueDim, tap.stream));
    if (tap.detail()) {
        // Concatenations in SGLang's fused order; each sink enqueues its copy before the staging is reused.
        const auto concat = [&](const bf16 *left, uint64_t lc, const bf16 *right, uint64_t rc) -> cere_status {
            const uint64_t pitch = (lc + rc) * sizeof(bf16);
            CERE_TRY_CUDA(cudaMemcpy2DAsync(core, pitch, left, lc * sizeof(bf16), lc * sizeof(bf16), n,
                                            cudaMemcpyDeviceToDevice, tap.stream));
            CERE_TRY_CUDA(cudaMemcpy2DAsync(core + lc, pitch, right, rc * sizeof(bf16), rc * sizeof(bf16), n,
                                            cudaMemcpyDeviceToDevice, tap.stream));
            return CERE_OK;
        };
        CERE_TRY(concat(mixed, k5::kConvDim, z, k5::kValueDim));
        CERE_TRY(tap.detail("model", layer, "linear_attn.in_proj_qkvz", core, kQkvzWidth));
        CERE_TRY(concat(b, k5::kValueHeads, a, k5::kValueHeads));
        CERE_TRY(tap.detail("model", layer, "linear_attn.in_proj_ba", core, 2 * k5::kValueHeads));
    }
    CERE_TRY(k5::layer_core(lw.gdn, session.gdn[layer], operation, mixed, z, b, a, n, mixed, core, core_bytes,
                            tap.stream, end));
    CERE_TRY(tap.detail("model", layer, "linear_attn.norm", mixed, k5::kValueDim));
    return gemm(ws.gdn_out, "gdn_out", mixed, lw.gdn.out_proj, ws.block_out, layer, tap);
}

// The MoE block into ws.block_out with its taps: MoeTaps outputs whenever taps are set, the in-block points through
// moe_point, then the B8 entries and their patch points (a replacement reruns only the combine, MoeTaps.*_in).
struct MoeHook {
    const Tap *tap;
    uint32_t layer;
};

cere_status moe_point(void *ctx, const char *point, bf16 *values, uint32_t rows, uint32_t cols, cudaStream_t) {
    const MoeHook &h = *static_cast<const MoeHook *>(ctx);
    if (rows != h.tap->n) return CERE_ERR_INTERNAL;
    const size_t len = std::strlen(point);
    if (std::strncmp(point, "gemm.", 5) != 0) return h.tap->detail("model", h.layer, point, values, cols);
    if (len > 3 && std::strcmp(point + len - 3, ".in") == 0) return h.tap->input(h.layer, point, values, cols);
    return h.tap->point("native", h.layer, point, values, cols);
}

cere_status moe(const LayerWeights &lw, const Workspace &ws, uint32_t layer, const Tap &tap) {
    const kernels::MoePlans plans =
        ws.moe_fork.plans(&ws.moe_router, &ws.moe_shared_gate_up, &ws.moe_shared_down, tap.cls);
    if (ws.research_call != nullptr) {  // never with taps (forward refuses both)
        return research_moe(lw, ws, plans, layer, 0, tap.n, ws.mixed, ws.block_out, ws.tap_routed, ws.tap_shared,
                            tap.stream);
    }
    if (!tap.emitting() && !tap.hooked()) {
        return CERE_MOE_BLOCK(kMoeDims, lw.moe, plans, ws.mixed, tap.n, ws.block_out, ws.scratch, ws.scratch_bytes,
                              tap.stream);
    }
    const MoeHook hook{&tap, layer};
    kernels::MoeTaps moe_taps;
    moe_taps.routed = ws.tap_routed;
    moe_taps.shared = ws.tap_shared;
    if (tap.hooked()) {
        moe_taps.point = &moe_point;
        moe_taps.ctx = const_cast<MoeHook *>(&hook);
        moe_taps.detail = tap.detail();
    }
    CERE_TRY(CERE_MOE_BLOCK(kMoeDims, lw.moe, plans, ws.mixed, tap.n, ws.block_out, ws.scratch, ws.scratch_bytes,
                            tap.stream, &moe_taps));
    CERE_TRY(tap.emit(layer, "mlp.experts", ws.tap_routed));
    CERE_TRY(tap.emit(layer, "mlp.shared_expert.down_proj", ws.tap_shared));
    bool replaced = false;
    CERE_TRY(tap.point("model", layer, "mlp.experts", ws.tap_routed, kHidden, &replaced));
    CERE_TRY(tap.point("model", layer, "mlp.shared_expert.down_proj", ws.tap_shared, kHidden, &replaced));
    if (!replaced) return CERE_OK;
    kernels::MoeTaps given;
    given.routed_in = ws.tap_routed;
    given.shared_in = ws.tap_shared;
    return CERE_MOE_BLOCK(kMoeDims, lw.moe, plans, ws.mixed, tap.n, ws.block_out, ws.scratch, ws.scratch_bytes,
                          tap.stream, &given);
}

static_assert(sizeof(cere_logit_bias) == 8 && offsetof(cere_logit_bias, bias) == 4, "k12 bias stride 2");
static_assert(sizeof(cere_logprob_entry) == 8 && offsetof(cere_logprob_entry, logprob) == 4, "k12 top stride 2");

// The K12 sampler stage (model.h forward, m27 design note I9) on `logits`, the BF16 logits of the scored row. A
// greedy (logprobs-only) item's greedy top-2 is already in `score` (device); a sampled item's caller skips that top-2
// and finalize writes the committed token's whole score here; a logprobs request fills ws.sampler.logprobs (logprob
// and top entries; the host adds id and top_count). `key` is the session position of the last fed token (D4). A
// constrained item's vocab mask (m28 design note D9) goes to ws.sampler.mask, and process applies it after the
// penalties and before logit_bias; an unconstrained item passes no mask (the M2.7 stage, unchanged).
cere_status sample_row(Workspace &ws, const bf16 *logits, SessionState &session, const SamplerArgs &s, bool logprobs,
                       uint32_t key, cere_token_score *score, cudaStream_t stream) {
    const SamplerScratch &sc = ws.sampler;
    const cere_sampling_params &p = s.params;
    const uint64_t bias_count = s.sampled ? s.bias.size() : 0;
    if (bias_count > 0) {
        // Pageable source: the copy is staged before cudaMemcpyAsync returns.
        CERE_TRY_CUDA(cudaMemcpyAsync(sc.bias, s.bias.data(), bias_count * sizeof(cere_logit_bias),
                                      cudaMemcpyHostToDevice, stream));
    }
    const uint32_t *mask = nullptr;
    if (s.vocab_mask != nullptr) {
        // Pageable source as the bias: the ticket's mask may change once the step returns.
        CERE_TRY_CUDA(cudaMemcpyAsync(sc.mask, s.vocab_mask, kMaskWords * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                      stream));
        mask = sc.mask;
    }
    // D2; a greedy logprobs-only item has no penalty and no bias.
    if (s.sampled) {
        CERE_TRY_CUDA(k12::process(logits, kVocab, session.out_counts, p.frequency_penalty, p.presence_penalty,
                                   p.repetition_penalty, mask, &sc.bias->id, &sc.bias->bias, 2, int64_t(bias_count),
                                   ws.proc_logits, stream));
    } else {
        CERE_TRY_CUDA(k12::process(logits, kVocab, nullptr, 0.0f, 0.0f, 1.0f, mask, nullptr, nullptr, 1, 0,
                                   ws.proc_logits, stream));
    }
    const float temperature = s.sampled ? p.temperature : 0.0f;
    if (s.sampled) {
        CERE_TRY_CUDA(k12::top2_f32(ws.proc_logits, kVocab, sc.top2_ids, sc.top2_scores, stream));
        const uint32_t *drawn = nullptr;
        if (temperature != 0.0f) {
            CERE_TRY_CUDA(
                k12::probabilities(ws.proc_logits, kVocab, temperature, sc.probs, sc.lse, sc.scratch, stream));
            CERE_TRY_CUDA(k12::keep_mask(sc.probs, kVocab, int64_t(p.top_k), p.top_p, p.min_p, sc.keep, sc.scratch,
                                         stream));
            CERE_TRY_CUDA(k12::draw_seeded(sc.probs, sc.keep, kVocab, p.seed, 1, key, sc.drawn, sc.scratch, stream));
            drawn = sc.drawn;
        }
        CERE_TRY_CUDA(k12::finalize(ws.proc_logits, sc.top2_ids, drawn, score, stream));
    }
    if (logprobs) {
        CERE_TRY_CUDA(k12::top_logprobs(ws.proc_logits, kVocab, temperature, &score->id, 0, int64_t(s.top_logprobs),
                                        &sc.logprobs->logprob, &sc.logprobs->top[0].id, &sc.logprobs->top[0].logprob,
                                        2, stream));
    }
    // D5: only a sampled item counts its committed token.
    if (s.sampled) CERE_TRY_CUDA(k12::count_commit(session.out_counts, &score->id, 1, stream));
    return CERE_OK;
}

// The forward body after argument checks; returns with work enqueued (forward synchronizes). `position` (device, a
// DECODE at n = 1): the decode graph's step (attention_core_step); null: the eager forward. `range` (null: the whole
// call; MixedPD): only layers [range->first, range->last), the embedding and HC expansion only when first is 0 and
// the final mixer and scoring only when last is kLayers; range->normed carries whether ws.normed holds the next mix's
// norm into and out of the range.
struct LayerRange {
    uint32_t first = 0;
    uint32_t last = kLayers;
    bool normed = false;
};

cere_status run_layers(const ModelWeights &w, Workspace &ws, SessionState &session, uint32_t operation, uint32_t n,
                       uint64_t rows, ForwardTaps *taps, const SamplerArgs *sampler, bool logprobs,
                       const host::StepImages *images, const int64_t *position, cudaStream_t stream,
                       const EndCall *end, LayerRange *range = nullptr) {
    const Tap tap{taps, n, rows, stream, end != nullptr ? end->cls : 0, end};
    const uint32_t first = range != nullptr ? range->first : 0;
    const uint32_t last = range != nullptr ? range->last : kLayers;
    // Rotary inputs (m213l design note C4): an image prompt call's rows at their mRoPE positions; every other call
    // at position + rope_delta (a text session: the model's table, the text launches). Every segment's layers read it.
    const Rope rope = images != nullptr ? prompt_rope(w, ws, session.position, n) : Rope{rope_table(w, session)};
    // B8 taps (o_proj / out_proj, MoE, model) unless the caller asked for the full-attention block inputs only (Tap
    // emits nothing then but self_attn.input).
    const bool attn_inputs = taps != nullptr && taps->attn_inputs_only;
    const k4::Plans ple_plans{&ws.ple_key, &ws.ple_value, tap.cls};
    // An end capture's PLE state as of its row (EndCall), into slot a's PLE region.
    k4::Capture ple_capture{};
    const bool ple_captures = end != nullptr && end->row > 0;
    if (ple_captures) ple_capture = {end->row, end->a->ple.history, end->a->ple.conv_state};
    bool normed = range != nullptr && range->normed;  // ws.normed holds the next mix's norm (combine)
    // The decode plan (SelfTune D2; ST1d: every PREFILL chunk and MixedPD segment too), in its launch order:
    // embed.tokens when the call starts at layer 0, the layer ops of [first, last), then the final norm and lm_head
    // when it ends at kLayers. A paused segment (MixedPD) or block inputs only stop after the layers: no final mixer,
    // no logits, nothing scored (forward copies no top-2 out).
    for (const plan::Op &op : plan::kDecode) {
        const uint32_t L = op.layer;
        if (plan::per_layer(op) ? L < first || L >= last
                                : op.slot == plan::Slot::kEmbedTokens ? first != 0 : last < kLayers || attn_inputs) {
            continue;
        }
        switch (op.slot) {
        case plan::Slot::kEmbedTokens:
            // embed_tokens, then the image splice (m213 design note D11: the merged rows of every image over its pad
            // rows of this call, which forward copied to ws.image_embeds at their call rows), then the HC expansion:
            // hyper[t] = cat(x[t] for each of the hc streams).
            CERE_TRY_CUDA(k1::embedding_gather(w.embed, w.config.vocab_size, kHidden, ws.ids32, n, ws.mixed, stream));
            if (images != nullptr) {
                const uint64_t start = session.position;
                for (uint64_t i = 0; i < images->count; ++i) {
                    const cere::mrope::ImageSpan &s = images->image[i].span;
                    const uint64_t lo = std::max<uint64_t>(s.token_offset, start);
                    const uint64_t hi = std::min<uint64_t>(s.token_offset + cere::mrope::image_tokens(s), start + n);
                    if (lo >= hi) continue;
                    CERE_TRY_CUDA(kv::splice(ws.image_embeds + (lo - start) * kHidden, int64_t(hi - lo), int64_t(lo),
                                             int64_t(start), int64_t(n), kHidden, ws.mixed, stream));
                }
                CERE_TRY(tap.detail("native", kLayers, "embed.spliced", ws.mixed, kHidden));
            }
            for (uint32_t g = 0; g < kHc; ++g) {
                CERE_TRY_CUDA(cudaMemcpy2DAsync(ws.hyper + uint64_t(g) * kHidden, uint64_t(kHcHidden) * sizeof(bf16),
                                                ws.mixed, kRowBytes, kRowBytes, n, cudaMemcpyDeviceToDevice, stream));
            }
            break;
        case plan::Slot::kEmbedPle:
            CERE_TRY(k4::forward(w.ple, ple_plans, session.ple, ws.ids64, n, ws.hyper, true, ws.hyper, ws.scratch,
                                 ws.scratch_bytes, stream, ple_captures ? &ple_capture : nullptr));
            break;
        case plan::Slot::kMixHc: {
            const bool attn = op.mix == plan::Mix::kAttn;
            bool edited = false;
            CERE_TRY(research_at(ws, attn ? PointId::kAttnMixIn : PointId::kMlpMixIn, L, 0, n, ws.hyper, kHcHidden,
                                 stream, &edited));
            CERE_TRY(mix(ws, plan_hc(w, L, op.mix), L, attn ? "attn" : "mlp", tap, normed && !edited));
            CERE_TRY(research_at(ws, attn ? PointId::kAttnMixOut : PointId::kMlpMixOut, L, 0, n, ws.mixed, kHidden,
                                 stream));
            break;
        }
        case plan::Slot::kAttentionFull:
        case plan::Slot::kAttentionLinearGdn: {
            const LayerWeights &lw = w.layers[L];
            const bool full = op.slot == plan::Slot::kAttentionFull;
            const char *block = full ? "self_attn.o_proj" : "linear_attn.out_proj";
            if (full) {
                CERE_TRY(tap.attn_input(L, ws.mixed));
                CERE_TRY(full_attention(w, lw, ws, session, L, rope, position, tap));
            } else {
                CERE_TRY(gdn(lw, ws, session, L, operation, tap));
            }
            CERE_TRY(research_at(ws, full ? PointId::kOProj : PointId::kOutProj, L, 0, n, ws.block_out, kHidden,
                                 stream));
            CERE_TRY(tap.emit(L, block, ws.block_out));
            CERE_TRY(tap.point("model", L, block, ws.block_out, kHidden));
            break;
        }
        case plan::Slot::kFfnMoe:
            CERE_TRY(moe(w.layers[L], ws, L, tap));
            break;
        case plan::Slot::kCombineHc:
            CERE_TRY(combine(ws, plan_hc(w, L, op.mix), n, fused_norm(w, op), stream));
            normed = op.fused;
            break;
        case plan::Slot::kNorm:
            // SGLang's hc_hidden_states: the HC rows before the final mixer, the MTP extend input of a drafting model
            // (M2.9).
            if (ws.hc_rows != nullptr) {
                CERE_TRY(copy_d2d(ws.hc_rows, ws.hyper, uint64_t(n) * kHcHidden * sizeof(bf16), stream));
            }
            // hyper_connection_mixer.mix replaces the final norm: ws.mixed is the model output [n, 2,560].
            CERE_TRY(mix(ws, w.final_mixer, kLayers, "final", tap, normed));
            CERE_TRY(research_at(ws, PointId::kFinalMixOut, kLayers, 0, n, ws.mixed, kHidden, stream));
            CERE_TRY(tap.emit(kLayers, "model", ws.mixed));
            break;
        case plan::Slot::kLmHead:
            // lm_head one m = 1 row per scored position (bit-identical to the decode logits of that row), then top-2;
            // a sampled item's row skips it (sample_row's finalize writes that row's whole score).
            for (uint64_t r = 0; r < rows; ++r) {
                const uint32_t m = uint32_t(n - rows + r + 1);  // lm_head_last_row reads row m - 1
                CERE_TRY(kernels::lm_head_last_row(ws.lm_head, ws.mixed, m, w.lm_head, ws.logits, stream));
                if (tap.patching()) taps->patch(taps->ctx, "native.logits", kLayers, 1, kVocab, ws.logits, stream);
                if (ws.research_call != nullptr) {
                    CERE_TRY(ws.research_call->logits(uint32_t(n - rows + r), ws.logits, stream));
                }
                if (sampler != nullptr && sampler->sampled && r + 1 == rows) continue;
                CERE_TRY_CUDA(k1::top2(ws.logits, kVocab, 1, kVocab, ws.top2 + r, stream));
            }
            // The sampler stage reads the last row's logits (sampler is non-null only with taps null: rows = 1).
            if (sampler != nullptr) {
                CERE_TRY(sample_row(ws, ws.logits, session, *sampler, logprobs, uint32_t(session.position + n - 1),
                                    ws.top2 + rows - 1, stream));
            }
            CERE_TRY_CUDA(cudaMemcpyAsync(ws.host_top2, ws.top2, rows * sizeof(cere_token_score),
                                          cudaMemcpyDeviceToHost, stream));
            if (logprobs) {
                CERE_TRY_CUDA(cudaMemcpyAsync(ws.host_logprobs, ws.sampler.logprobs, sizeof(cere_token_logprobs),
                                              cudaMemcpyDeviceToHost, stream));
            }
            break;
        }
    }
    if (range != nullptr) range->normed = normed;
    return CERE_OK;
}

// ---- Batched decode step (m26 design note D5) -------------------------------------------------------------------

// D6 per-row MoE fallback: true runs moe_block once per row at m = 1 on the single-row plans (each row then is the
// solo call) instead of once at m = R on the rows MoePlans. Fused unless K2 test_moe_rows_invariant fails;
// instantiate_model logs the path ("cere: moe rows path fused|per_row").
constexpr bool kMoeRowsPerRow = false;

// One B8 tensor of a batched step: row r's [1, 2,560] slice of a [R, 2,560] buffer, once per row in row order.
cere_status emit_rows(ForwardTaps *taps, uint32_t layer, const char *suffix, const bf16 *values, uint32_t rows,
                      cudaStream_t stream) {
    if (taps == nullptr) return CERE_OK;
    char name[112];
    if (layer < kLayers) {
        Tap::format(name, "model", layer, suffix);
    } else {
        std::snprintf(name, sizeof(name), "%s", suffix);
    }
    for (uint32_t r = 0; r < rows; ++r) {
        CERE_TRY(taps->emit(taps->ctx, name, values + uint64_t(r) * kHidden, 1, kHidden, stream));
    }
    return CERE_OK;
}

// The GDN scratch of a batched step: the projections mixed [R, 10,240], z [R, 6,144], b and a [R, 48], the per-row
// outputs y [R, 6,144] (row r written only by row r's layer_core, or the verify call's k5::chain_core) and one core
// workspace that fits both a single-token layer_core and a chain_core of min(R, k5::kChainMaxRows) rows.
struct GdnRows {
    bf16 *mixed, *z, *b, *a, *y;
    void *core;
    size_t core_bytes;
    size_t bytes;  // from the scratch base
};

// `scratch` may be null (sizing only: the pointers are then unusable).
GdnRows carve_gdn_rows(void *scratch, uint32_t rows) {
    const size_t r = rows;
    const size_t sizes[5] = {r * k5::kConvDim * sizeof(bf16), r * k5::kValueDim * sizeof(bf16),
                             r * k5::kValueHeads * sizeof(bf16), r * k5::kValueHeads * sizeof(bf16),
                             r * k5::kValueDim * sizeof(bf16)};
    const uintptr_t base = reinterpret_cast<uintptr_t>(scratch);
    bf16 *p[5];
    size_t at = 0;
    for (int i = 0; i < 5; ++i) {
        p[i] = reinterpret_cast<bf16 *>(base + at);
        at += align256(sizes[i]);
    }
    const size_t core_bytes = std::max(k5::core_workspace_bytes(1),
                                       k5::chain_core_workspace_bytes(std::min<int64_t>(rows, k5::kChainMaxRows)));
    GdnRows g{p[0], p[1], p[2], p[3], p[4], reinterpret_cast<void *>(base + at), core_bytes, 0};
    g.bytes = at + g.core_bytes;
    return g;
}

// GDN of a batched step into ws.block_out: projections at m = R on the rows plans, k5::layer_core (DECODE) once per
// row with that session's LayerState, out_proj at m = R.
cere_status gdn_rows(const LayerWeights &lw, const Workspace &ws, SessionState *const *sessions, uint32_t rows,
                     uint32_t layer, cudaStream_t stream) {
    const RowsPlans &p = ws.rows;
    const GdnRows g = carve_gdn_rows(ws.scratch, rows);
    const kernels::GemmPlan *plans[4] = {&p.gdn_qkv, &p.gdn_z, &p.gdn_ba, &p.gdn_ba};
    const bf16 *weights[4] = {lw.gdn.in_proj_qkv, lw.gdn.in_proj_z, lw.gdn.in_proj_b, lw.gdn.in_proj_a};
    void *const outs[4] = {g.mixed, g.z, g.b, g.a};
    CERE_TRY(kernels::gemm_segments(plans, weights, outs, 4, ws.mixed, rows, stream));
    CERE_TRY(research_at(ws, PointId::kGdnQkv, layer, 0, rows, g.mixed, k5::kConvDim, stream));
    CERE_TRY(research_at(ws, PointId::kGdnZ, layer, 0, rows, g.z, k5::kValueDim, stream));
    for (uint32_t r = 0; r < rows; ++r) {
        CERE_TRY(k5::layer_core(lw.gdn, sessions[r]->gdn[layer], CERE_OP_DECODE, g.mixed + uint64_t(r) * k5::kConvDim,
                                g.z + uint64_t(r) * k5::kValueDim, g.b + uint64_t(r) * k5::kValueHeads,
                                g.a + uint64_t(r) * k5::kValueHeads, 1, g.y + uint64_t(r) * k5::kValueDim, g.core,
                                g.core_bytes, stream));
    }
    return p.gdn_out.run(g.y, lw.gdn.out_proj, rows, ws.block_out, stream);
}

// Full attention of a batched step into ws.block_out: QKV / O projections at m = R on the rows plans, the single-row
// decode attention (attention_core, n = 1) once per row on that session's prefix; with `positions` (device, row r at
// positions + r) the decode graph's attention_core_step instead.
cere_status attention_rows(const ModelWeights &w, const LayerWeights &lw, const Workspace &ws,
                           SessionState *const *sessions, uint32_t rows, uint32_t layer, const int64_t *positions,
                           cudaStream_t stream) {
    CERE_TRY(ws.rows.attn_qkv.run(ws.mixed, lw.attn.qkv, rows, ws.qkv, stream));
    CERE_TRY(research_at(ws, PointId::kAttnQkv, layer, 0, rows, ws.qkv, kQkvWidth, stream));
    for (uint32_t r = 0; r < rows; ++r) {
        bf16 *qkv = ws.qkv + uint64_t(r) * kQkvWidth;
        const bf16 *mixed = ws.mixed + uint64_t(r) * kHidden;
        bf16 *attn_out = ws.attn_out + uint64_t(r) * kAttnOutWidth;
        if (positions != nullptr) {
            CERE_TRY(attention_core_step(w, lw, ws, *sessions[r], layer, qkv, mixed, attn_out, positions + r, stream));
        } else {
            CERE_TRY(attention_core(w, lw, ws, *sessions[r], layer, qkv, mixed, attn_out, 1,
                                    Rope{rope_table(w, *sessions[r])}, stream));
        }
    }
    return ws.rows.attn_o.run(ws.attn_out, lw.attn.o_proj, rows, ws.block_out, stream);
}

// MoE of a batched step into ws.block_out: moe_block at m = R on the rows MoePlans, or the D6 per-row fallback; with
// taps, MoeTaps.routed / shared as the single-row forward requests them, emitted per row.
cere_status moe_rows(const LayerWeights &lw, const Workspace &ws, uint32_t rows, uint32_t layer, ForwardTaps *taps,
                     cudaStream_t stream) {
    if (ws.research_call != nullptr) {  // never with taps (forward_decode_rows refuses both)
        if (ws.rows.moe_per_row) {
            const kernels::MoePlans plans =
                ws.moe_fork.plans(&ws.moe_router, &ws.moe_shared_gate_up, &ws.moe_shared_down);
            for (uint32_t r = 0; r < rows; ++r) {
                const uint64_t at = uint64_t(r) * kHidden;
                CERE_TRY(research_moe(lw, ws, plans, layer, r, 1, ws.mixed + at, ws.block_out + at,
                                      ws.tap_routed + at, ws.tap_shared + at, stream));
            }
            return CERE_OK;
        }
        const kernels::MoePlans plans =
            ws.moe_fork.plans(&ws.rows.moe_router, &ws.rows.moe_shared_gate_up, &ws.rows.moe_shared_down);
        return research_moe(lw, ws, plans, layer, 0, rows, ws.mixed, ws.block_out, ws.tap_routed, ws.tap_shared,
                            stream);
    }
    if (ws.rows.moe_per_row) {
        const kernels::MoePlans plans =
            ws.moe_fork.plans(&ws.moe_router, &ws.moe_shared_gate_up, &ws.moe_shared_down);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint64_t at = uint64_t(r) * kHidden;
            kernels::MoeTaps row_taps;
            row_taps.routed = ws.tap_routed + at;
            row_taps.shared = ws.tap_shared + at;
            CERE_TRY(CERE_MOE_BLOCK(kMoeDims, lw.moe, plans, ws.mixed + at, 1, ws.block_out + at, ws.scratch,
                                    ws.scratch_bytes, stream, taps != nullptr ? &row_taps : nullptr));
        }
    } else {
        const kernels::MoePlans plans =
            ws.moe_fork.plans(&ws.rows.moe_router, &ws.rows.moe_shared_gate_up, &ws.rows.moe_shared_down);
        kernels::MoeTaps moe_taps;
        moe_taps.routed = ws.tap_routed;
        moe_taps.shared = ws.tap_shared;
        CERE_TRY(CERE_MOE_BLOCK(kMoeDims, lw.moe, plans, ws.mixed, rows, ws.block_out, ws.scratch, ws.scratch_bytes,
                                stream, taps != nullptr ? &moe_taps : nullptr));
    }
    CERE_TRY(emit_rows(taps, layer, "mlp.experts", ws.tap_routed, rows, stream));
    return emit_rows(taps, layer, "mlp.shared_expert.down_proj", ws.tap_shared, rows, stream);
}

// forward_decode_rows's body after argument checks; returns with work enqueued. Row r's sampler stage (samplers[r]
// non-null) reads row r of the step's logits; its logprobs leave the shared ws.sampler.logprobs for
// ws.host_logprobs[r] before the next row's stage writes them.
cere_status run_rows(const ModelWeights &w, Workspace &ws, SessionState *const *sessions, uint32_t rows,
                     ForwardTaps *taps, const SamplerArgs *const *samplers, cere_token_logprobs *const *out_logprobs,
                     const int64_t *positions, cudaStream_t stream) {
    const RowsPlans &p = ws.rows;
    CERE_TRY_CUDA(k1::embedding_gather(w.embed, w.config.vocab_size, kHidden, ws.ids32, rows, ws.mixed, stream));
    for (uint32_t g = 0; g < kHc; ++g) {
        CERE_TRY_CUDA(cudaMemcpy2DAsync(ws.hyper + uint64_t(g) * kHidden, uint64_t(kHcHidden) * sizeof(bf16),
                                        ws.mixed, kRowBytes, kRowBytes, rows, cudaMemcpyDeviceToDevice, stream));
    }
    // PLE as one k4::forward_rows (each row its own sequence on its session's k4 State; row r bit-equal to the solo
    // call by construction, rows16 design note R16-A).
    const k4::Plans ple_plans{&p.ple_key, &p.ple_value};
    k4::RowsCall ple_call;
    ple_call.sequences = rows;
    for (uint32_t r = 0; r < rows; ++r) {
        ple_call.states[r] = sessions[r]->ple;
        ple_call.first[r] = r;
        ple_call.count[r] = 1;
    }
    bool normed = false;  // ws.normed holds the next mix's norm (combine)
    const auto mix_rows = [&](const HcWeights &hc) { return run_mix(p.hc, ws, hc, rows, normed, stream); };
    // The decode plan's layer ops (SelfTune D2), in their launch order.
    for (const plan::Op &op : plan::kDecode) {
        if (!plan::per_layer(op)) continue;
        const uint32_t L = op.layer;
        const LayerWeights &lw = w.layers[L];
        switch (op.slot) {
        case plan::Slot::kEmbedPle:
            CERE_TRY(k4::forward_rows(w.ple, ple_plans, ple_call, ws.ids64, rows, ws.hyper, true, ws.hyper,
                                      ws.scratch, ws.scratch_bytes, stream));
            break;
        case plan::Slot::kMixHc: {
            const bool attn = op.mix == plan::Mix::kAttn;
            bool edited = false;
            CERE_TRY(research_at(ws, attn ? PointId::kAttnMixIn : PointId::kMlpMixIn, L, 0, rows, ws.hyper,
                                 kHcHidden, stream, &edited));
            if (edited) normed = false;
            CERE_TRY(mix_rows(plan_hc(w, L, op.mix)));
            CERE_TRY(research_at(ws, attn ? PointId::kAttnMixOut : PointId::kMlpMixOut, L, 0, rows, ws.mixed,
                                 kHidden, stream));
            break;
        }
        case plan::Slot::kAttentionFull:
        case plan::Slot::kAttentionLinearGdn: {
            const bool full = op.slot == plan::Slot::kAttentionFull;
            if (full) {
                CERE_TRY(attention_rows(w, lw, ws, sessions, rows, L, positions, stream));
            } else {
                CERE_TRY(gdn_rows(lw, ws, sessions, rows, L, stream));
            }
            CERE_TRY(research_at(ws, full ? PointId::kOProj : PointId::kOutProj, L, 0, rows, ws.block_out, kHidden,
                                 stream));
            CERE_TRY(emit_rows(taps, L, full ? "self_attn.o_proj" : "linear_attn.out_proj", ws.block_out, rows,
                               stream));
            break;
        }
        case plan::Slot::kFfnMoe:
            CERE_TRY(moe_rows(lw, ws, rows, L, taps, stream));
            break;
        case plan::Slot::kCombineHc:
            CERE_TRY(combine(ws, plan_hc(w, L, op.mix), rows, fused_norm(w, op), stream));
            normed = op.fused;
            break;
        default:
            break;
        }
    }
    CERE_TRY(mix_rows(w.final_mixer));
    CERE_TRY(research_at(ws, PointId::kFinalMixOut, kLayers, 0, rows, ws.mixed, kHidden, stream));
    CERE_TRY(emit_rows(taps, kLayers, "model", ws.mixed, rows, stream));
    // lm_head and top-2: once at m = R into ws.rows_logits on the rows plan of the lm_head (m23 design note P11R-21;
    // the small-M kernel under k2_gemm = small_m, P11), row r bit-equal to the m = 1 call on row r, so to the solo
    // step; once per row at m = 1 (lm_head_last_row reads row r) into ws.logits only beyond the plan's rows. A sampled
    // item's sampler stage then overwrites its whole score (finalize); every row's stage is keyed by the position of
    // the token it feeds (ledger CR4: forward's key for that session alone).
    const bool batched = rows <= ws.rows.lm_head.max_m();
    if (batched) {
        CERE_TRY(ws.rows.lm_head.run(ws.mixed, w.lm_head, rows, ws.rows_logits, stream));
        CERE_TRY_CUDA(k1::top2(ws.rows_logits, kVocab, rows, kVocab, ws.top2, stream));
    }
    for (uint32_t r = 0; r < rows; ++r) {
        const bf16 *logits = batched ? ws.rows_logits + uint64_t(r) * kVocab : ws.logits;
        if (!batched) {
            CERE_TRY(kernels::lm_head_last_row(ws.lm_head, ws.mixed, r + 1, w.lm_head, ws.logits, stream));
            CERE_TRY_CUDA(k1::top2(ws.logits, kVocab, 1, kVocab, ws.top2 + r, stream));
        }
        if (ws.research_call != nullptr) CERE_TRY(ws.research_call->logits(r, logits, stream));
        const SamplerArgs *sampler = samplers != nullptr ? samplers[r] : nullptr;
        if (sampler == nullptr) continue;
        const bool logprobs = out_logprobs != nullptr && out_logprobs[r] != nullptr;
        CERE_TRY(sample_row(ws, logits, *sessions[r], *sampler, logprobs, uint32_t(sessions[r]->position),
                            ws.top2 + r, stream));
        if (logprobs) {
            CERE_TRY_CUDA(cudaMemcpyAsync(ws.host_logprobs + r, ws.sampler.logprobs, sizeof(cere_token_logprobs),
                                          cudaMemcpyDeviceToHost, stream));
        }
    }
    CERE_TRY_CUDA(cudaMemcpyAsync(ws.host_top2, ws.top2, rows * sizeof(cere_token_score), cudaMemcpyDeviceToHost,
                                  stream));
    return CERE_OK;
}

// ---- Speculation: verify rows (m29 design note D4, M29S2-20) ---------------------------------------------------

// Bytes of each piece of session state a verify row mutates in place (the compiled A4 shapes, model.h SessionState).
constexpr size_t kGdnConvBytes = size_t(k5::kConvDim) * 3 * sizeof(bf16);                  // conv [10,240, 3]
constexpr size_t kGdnSsmBytes = size_t(k5::kValueHeads) * 128 * 128 * sizeof(bf16);        // ssm [48, 128, 128]
constexpr size_t kPleHistoryBytes = size_t(k4::kHistoryLen) * sizeof(int64_t);             // history [2]
constexpr size_t kPleConvBytes = size_t(k4::kChannels) * k4::kConvStateLen * sizeof(bf16);  // conv [10,240, 9]
constexpr size_t kQsaRingBytes = size_t(k9::kRatio) * k9::kIndexHeadDim * sizeof(bf16);     // ring [4, 128]
constexpr size_t kQsaKeyBytes = size_t(k9::kIndexHeadDim) * sizeof(bf16);                  // one compressed key

// One row of a verify call, flattened (chains in order, rows 0..n of each): its chain's session at the chain's
// position p (session.position stays p during the call), the chain's undo snapshots and j (0 = the committed token),
// and its chain (for the chain's sampler, phase S).
struct VerifyFlat {
    SessionState *session;
    UndoState *undo;
    uint64_t p;
    uint32_t j;
    const VerifyRow *chain;
};

// The D2D copies of a verify call. On the verify graph's path (`graph`: run_verify's positions non-null,
// m23 design note P11R-20) one k1::copy_segments launch, a kernel node once captured; else one cudaMemcpy(2D)Async
// per range. The bytes are the same; D12 native-test-profile-20261014T054614Z.json measured 1.6 ms per verify of idle
// next to the captured memcpy nodes (0.16 ms eager).
k1::CopySegment flat(void *dst, const void *src, size_t bytes) { return {dst, bytes, src, bytes, bytes, 1}; }

cere_status copy_ranges(bool graph, std::initializer_list<k1::CopySegment> ranges, cudaStream_t stream) {
    if (graph) return cuda_status(k1::copy_segments(ranges.begin(), int(ranges.size()), stream));
    for (const k1::CopySegment &s : ranges) {
        if (s.rows == 1) {
            CERE_TRY(copy_d2d(s.dst, s.src, s.width, stream));
            continue;
        }
        CERE_TRY_CUDA(cudaMemcpy2DAsync(s.dst, s.dst_pitch, s.src, s.src_pitch, s.width, s.rows,
                                        cudaMemcpyDeviceToDevice, stream));
    }
    return CERE_OK;
}

// The k4::forward_rows call of a verify call's PLE layer: each chain one sequence on its session's k4 State, row
// j < n of a chain saving into undo[j] (the state as of before row j + 1 mutates it; row 0 of a chain is rows[r] with
// j 0).
k4::RowsCall ple_verify_call(const VerifyFlat *rows, uint32_t m) {
    k4::RowsCall call;
    for (uint32_t r = 0; r < m; ++r) {
        const VerifyFlat &f = rows[r];
        if (f.j == 0) {
            call.states[call.sequences] = f.session->ple;
            call.first[call.sequences] = r;
            ++call.sequences;
        }
        ++call.count[call.sequences - 1];
        if (f.j < f.chain->n) call.saves[r] = f.undo[f.j].ple;
    }
    return call;
}

// The k5::chain_core chains of GDN layer L in a verify call, as ple_verify_call: each chain on its session's state,
// row j < n saving into undo[j].gdn[L].
k5::Chains gdn_verify_chains(const VerifyFlat *rows, uint32_t m, uint32_t L) {
    k5::Chains chains;
    for (uint32_t r = 0; r < m; ++r) {
        const VerifyFlat &f = rows[r];
        if (f.j == 0) chains.states[chains.count++] = f.session->gdn[L];
        ++chains.rows[chains.count - 1];
        if (f.j < f.chain->n) chains.saves[r] = f.undo[f.j].gdn[L];
    }
    return chains;
}

// The ring always (untouched by the dense K6 path, so then the copy is the unchanged ring), and the compressed key
// row the rest of the chain may complete (key_row, set by forward_verify_rows on the K9 path only). `position`
// (device, the row's p + j; the verify graph) takes the key row on the device instead (k9::qsa_key_save_step: the
// key_row rule from the position, on a plan with the K9 path, `qsa`), so no host scalar of p reaches the launch;
// `fault` is its k9::Bounds fault word (ws.bound_fault).
cere_status undo_qsa(const VerifyFlat &f, uint32_t layer, const int64_t *position, bool qsa, uint32_t *fault,
                     cudaStream_t stream) {
    if (f.j == 0) return CERE_OK;
    const UndoState &u = f.undo[f.j - 1];
    CERE_TRY(copy_ranges(position != nullptr, {flat(u.qsa_ring[layer], f.session->qsa_ring[layer], kQsaRingBytes)},
                         stream));
    if (position != nullptr) {
        if (!qsa) return CERE_OK;
        const k9::Bounds bounds{0, 0, int64_t(f.session->max_context_tokens), fault};  // rope fields unused
        return k9::qsa_key_save_step(f.session->qsa_keys[layer], f.session->slot_map, bounds, position, f.j,
                                     f.chain->n, u.qsa_key[layer], stream);
    }
    if (u.key_row < 0) return CERE_OK;
    return copy_d2d(u.qsa_key[layer], f.session->qsa_keys[layer] + uint64_t(u.key_row) * k9::kIndexHeadDim,
                    kQsaKeyBytes, stream);
}

static_assert(layout::kDenseContext == uint64_t(k6::kMaxContext), "spec_graph_chain_fits: K6's dense limit");
static_assert(k9::kChainsStepRows == k1::kKvRowsMax, "attention_chains: one k9::qsa_chains_step over the m rows");

// verify_attention = batched (m23 design note P11R-19, long-ctx design note LK-01): true when the chain of `rows`
// (its row 0) with c rows runs its full-attention layers as attention_chain. Either every row attends densely from
// the cache (p > 0, p + c <= k6::kMaxContext: no row reaches K9's sparse path) or, on a QSA plan, every row takes
// K9's sparse path (p >= k6::kMaxContext: row 0 already attends past the dense limit); a chain that straddles the
// limit, and a chain of one row (nothing to batch), stay per row.
bool chain_batched(const Workspace &ws, const VerifyFlat &row0, uint32_t c) {
    if (!ws.verify_batched || c <= 1 || row0.p == 0) return false;
    const bool sparse = ws.max_context_tokens > uint64_t(k6::kMaxContext) && row0.p >= uint64_t(k6::kMaxContext);
    return sparse || row0.p + c <= uint64_t(k6::kMaxContext);
}

// One full-attention layer of a chain's c rows (chain_batched), bit-identical to the per-row attention_core calls of
// run_verify. The QSA work of a QSA plan (contexts above 2,048) stays per row in row order, each row's undo_qsa right
// before its ring / compressed-key write (M29S2-20); its index-key GEMM is one call at c rows only when the plan runs
// the small-M kernel (row r bit-equal to the m = 1 call), else one call per row. Then one K6 prologue over the c rows
// (per-token arithmetic; every row's K/V lands in the FP8 cache before any row attends, and row t attends to
// positions <= p + t only). A dense chain then runs one k6::dense_attention_rows (row t bit-equal to the one-token
// call at p + t). A sparse chain (every row past the dense limit; eager only, the verify graph holds dense chains)
// runs K9's indexer query, selection and sparse attention over up to k9::kSplitMaxRows rows per call (row t of a call
// bit-equal to the one-row call at p + t: per-row index queries, logits and top-512 over the row's own visible blocks
// - a block a later row completes is past them - and the same split-K attention), then one sigmoid gate (LK-01).
// `positions` (device, rows' p + r; the verify graph, sessions at rope_delta 0 only) runs the device-position twins of
// the position-dependent launches (k9::qsa_compress_store_step, k6::prologue_rows_at, k6::dense_attention_rows_at),
// launch for launch. The eager launches rotate at p + r + rope_delta (rope_table, m213l design note C4).
cere_status attention_chain(const ModelWeights &w, const LayerWeights &lw, const Workspace &ws, const VerifyFlat *rows,
                            uint32_t c, uint32_t layer, const bf16 *qkv, const bf16 *mixed, bf16 *attn_out,
                            const int64_t *positions, cudaStream_t stream) {
    SessionState &s = *rows[0].session;
    const AttentionWeights &aw = lw.attn;
    const int64_t p = int64_t(rows[0].p);
    const int64_t *slots = s.slot_map;
    const float eps = model_eps(w.config);
    const float *table = rope_table(w, s);
    const k9::Bounds bounds = k9_bounds(w, ws, table, s.max_context_tokens);
    const size_t t = c;
    uint8_t *ptr = static_cast<uint8_t *>(ws.scratch);
    bf16 *q = reinterpret_cast<bf16 *>(ptr);
    ptr += align256(t * kAttnOutWidth * sizeof(bf16));
    bf16 *k_rot = reinterpret_cast<bf16 *>(ptr);
    ptr += align256(t * kKvWidth * sizeof(bf16));
    bf16 *gate = reinterpret_cast<bf16 *>(ptr);
    ptr += align256(t * kAttnOutWidth * sizeof(bf16));
    const bool qsa = ws.max_context_tokens > uint64_t(k6::kMaxContext);
    const bool sparse = qsa && p >= int64_t(k6::kMaxContext);
    if (sparse && positions != nullptr) return CERE_ERR_UNSUPPORTED;
    bf16 *qk = reinterpret_cast<bf16 *>(ptr);
    if (qsa) ptr += align256(t * k9::kQkWidth * sizeof(bf16));
    const size_t rest = ws.scratch_bytes - size_t(ptr - static_cast<uint8_t *>(ws.scratch));
    const bool index_batched = qsa && ws.attn_index_qk.small_m() && c <= ws.attn_index_qk.max_m();
    if (index_batched) CERE_TRY(ws.attn_index_qk.run(mixed, aw.indexer_qk, c, qk, stream));
    for (uint32_t r = 0; r < c; ++r) {
        const int64_t *position = positions != nullptr ? positions + r : nullptr;
        CERE_TRY(undo_qsa(rows[r], layer, position, qsa, ws.bound_fault, stream));
        if (!qsa) continue;
        bf16 *qk_r = qk + uint64_t(r) * k9::kQkWidth;  // every row's index q stays for a sparse chain's selection
        if (!index_batched) {
            CERE_TRY(ws.attn_index_qk.run(mixed + uint64_t(r) * kHidden, aw.indexer_qk, 1, qk_r, stream));
        }
        const bf16 *k_idx = qk_r + k9::kIndexHeads * k9::kIndexHeadDim;
        if (position != nullptr) {  // the verify graph: sessions at rope_delta 0, table = w.rope_cos_sin
            CERE_TRY(k9::qsa_compress_store_step(k_idx, aw.indexer_k_norm, eps, w.rope_cos_sin,
                                                 k9_bounds(w, ws, w.rope_cos_sin, s.max_context_tokens), slots,
                                                 position, s.qsa_ring[layer], s.qsa_keys[layer], stream));
        } else {
            CERE_TRY(k9::qsa_compress_store(k_idx, k9::kQkWidth, aw.indexer_k_norm, eps, table, bounds, nullptr,
                                            slots, p + r, 1, s.qsa_ring[layer], s.qsa_keys[layer], stream));
        }
    }
    const bf16 *k = qkv + kQGateWidth;
    const bf16 *v = k + kKvWidth;
    if (positions != nullptr) {
        CERE_TRY_CUDA(k6::prologue_rows_at(qkv, kQkvWidth, k, kQkvWidth, v, kQkvWidth, aw.q_norm, aw.k_norm, eps,
                                           w.rope_cos_sin, slots, positions, int64_t(c), s.k_cache[layer],
                                           s.v_cache[layer], q, k_rot, gate, stream));
        CERE_TRY_CUDA(k6::dense_attention_rows_at(q, gate, int64_t(c), positions, s.k_cache[layer], s.v_cache[layer],
                                                  slots, ptr, rest, attn_out, kAttnOutWidth, stream));
        return CERE_OK;
    }
    if (!sparse) {
        CERE_TRY_CUDA(k6::prologue(qkv, kQkvWidth, k, kQkvWidth, v, kQkvWidth, aw.q_norm, aw.k_norm, eps,
                                   table, int64_t(k6::kMaxContext), nullptr, nullptr, slots, int64_t(c), p,
                                   s.k_cache[layer], s.v_cache[layer], q, k_rot, gate, stream));
        CERE_TRY_CUDA(k6::dense_attention_rows(q, gate, int64_t(c), p, s.k_cache[layer], s.v_cache[layer], slots, ptr,
                                               rest, attn_out, kAttnOutWidth, stream));
        return CERE_OK;
    }
    // The sparse chain: attention_rows' K9 path (prefix p + r0, FP8 cache) per call of up to kSplitMaxRows rows.
    CERE_TRY_CUDA(k6::prologue(qkv, kQkvWidth, k, kQkvWidth, v, kQkvWidth, aw.q_norm, aw.k_norm, eps, table,
                               int64_t(w.max_context_tokens), nullptr, nullptr, slots, int64_t(c), p, s.k_cache[layer],
                               s.v_cache[layer], q, k_rot, gate, stream));
    bf16 *attn = reinterpret_cast<bf16 *>(ptr);
    ptr += align256(t * kAttnOutWidth * sizeof(bf16));
    const int64_t span = std::min<int64_t>(int64_t(c), k9::kSplitMaxRows);
    k9::Scratch ks{};
    CERE_TRY(k9::carve_scratch(ptr, ws.scratch_bytes - size_t(ptr - static_cast<uint8_t *>(ws.scratch)), span,
                               int64_t(ws.max_context_tokens), &ks));
    for (int64_t r0 = 0; r0 < int64_t(c); r0 += span) {
        const int64_t n = std::min<int64_t>(span, int64_t(c) - r0);
        CERE_TRY(k9::indexer_qk(qk + uint64_t(r0) * k9::kQkWidth, k9::kQkWidth, aw.indexer_q_norm, eps, table,
                                bounds, nullptr, p + r0, n, ks.q_idx, stream));
        CERE_TRY(k9::qsa_select(ks.q_idx, s.qsa_keys[layer], slots, nullptr, n, p + r0 + n, ks.work, ks.work_bytes,
                                ks.indices, stream));
        CERE_TRY(k9::qsa_attention(q + uint64_t(r0) * kAttnOutWidth, kAttnOutWidth, ks.indices, n, nullptr, 0, nullptr,
                                   0, s.k_cache[layer], s.v_cache[layer], slots, k9::kSoftmaxScale, ks.work,
                                   ks.work_bytes, attn + uint64_t(r0) * kAttnOutWidth, kAttnOutWidth, stream,
                                   ws.prefill_p4));
    }
    CERE_TRY_CUDA(k1::sigmoid_mul(attn, kAttnOutWidth, gate, kAttnOutWidth, int64_t(c), kAttnOutWidth, attn_out,
                                  kAttnOutWidth, stream));
    return CERE_OK;
}

// rows16 design note R16-C: true when the verify graph's call (positions non-null) runs every full-attention layer as
// one attention_chains over all m rows: the verify_attention lever is batched and every chain (one row included)
// attends densely from the cache (p > 0, p + c <= k6::kMaxContext). Else run_verify goes chain by chain.
bool chains_batched(const Workspace &ws, const VerifyFlat *rows, uint32_t m, const int64_t *positions) {
    if (positions == nullptr || !ws.verify_batched || m > uint32_t(k1::kKvRowsMax)) return false;
    for (uint32_t r = 0; r < m;) {
        const uint32_t c = rows[r].chain->n + 1;
        if (rows[r].p == 0 || rows[r].p + c > uint64_t(k6::kMaxContext)) return false;
        r += c;
    }
    return true;
}

// One full-attention layer of the verify graph's m rows over every sequence at once (chains_batched), bit-identical to
// attention_chain per chain: the QSA work of a QSA plan per row in row order as there (each row's undo_qsa before its
// ring / compressed-key write; the index-key GEMM one call at m rows on the small-M kernel, row r bit-equal to the
// m = 1 call, else one call per row, all before the QSA work), as one k9::qsa_chains_step launch (an internal design note
// spec.md FG-6: chain by chain, row by row, the undo_qsa and qsa_compress_store_step work byte for byte), then one
// k6::prologue_chains_at and one k6::dense_attention_chains_at over the m rows, row r on its sequence's caches and
// slot map at positions[r] (every row's K/V stored before any row attends; row r attends to its sequence's positions
// <= p + j).
cere_status attention_chains(const ModelWeights &w, const LayerWeights &lw, const Workspace &ws,
                             const VerifyFlat *rows, uint32_t m, uint32_t layer, const int64_t *positions,
                             cudaStream_t stream) {
    const AttentionWeights &aw = lw.attn;
    const float eps = model_eps(w.config);
    const size_t t = m;
    uint8_t *ptr = static_cast<uint8_t *>(ws.scratch);
    bf16 *q = reinterpret_cast<bf16 *>(ptr);
    ptr += align256(t * kAttnOutWidth * sizeof(bf16));
    bf16 *k_rot = reinterpret_cast<bf16 *>(ptr);
    ptr += align256(t * kKvWidth * sizeof(bf16));
    bf16 *gate = reinterpret_cast<bf16 *>(ptr);
    ptr += align256(t * kAttnOutWidth * sizeof(bf16));
    const bool qsa = ws.max_context_tokens > uint64_t(k6::kMaxContext);
    bf16 *qk = reinterpret_cast<bf16 *>(ptr);
    if (qsa) ptr += align256(t * k9::kQkWidth * sizeof(bf16));
    const size_t rest = ws.scratch_bytes - size_t(ptr - static_cast<uint8_t *>(ws.scratch));
    const bool index_batched = qsa && ws.attn_index_qk.small_m() && m <= ws.attn_index_qk.max_m();
    if (index_batched) CERE_TRY(ws.attn_index_qk.run(ws.mixed, aw.indexer_qk, m, qk, stream));
    k1::KvRows kv{};
    k9::ChainsStep step{};
    int chains = 0;
    for (uint32_t r = 0; r < m; ++r) {
        const VerifyFlat &f = rows[r];
        SessionState &s = *f.session;
        kv.k_cache[r] = s.k_cache[layer];
        kv.v_cache[r] = s.v_cache[layer];
        kv.slots[r] = s.slot_map;
        if (f.j == 0) {
            step.chain_first[chains] = int32_t(r);
            ++chains;
        } else if (chains == 0) {
            return CERE_ERR_INTERNAL;  // chains_batched: rows are whole chains from row 0
        }
        ++step.chain_rows[chains - 1];
        step.slots[r] = s.slot_map;
        step.ring[r] = s.qsa_ring[layer];
        step.keys[r] = s.qsa_keys[layer];
        step.context[r] = int64_t(s.max_context_tokens);
        if (f.j > 0) {
            const UndoState &u = f.undo[f.j - 1];
            step.undo_ring[r] = u.qsa_ring[layer];
            if (qsa) step.undo_key[r] = u.qsa_key[layer];
            step.j[r] = int32_t(f.j);
            step.n[r] = int32_t(f.chain->n);
        }
        if (!qsa) continue;
        bf16 *qk_r = qk + uint64_t(r) * k9::kQkWidth;
        if (!index_batched) {
            CERE_TRY(ws.attn_index_qk.run(ws.mixed + uint64_t(r) * kHidden, aw.indexer_qk, 1, qk_r, stream));
        }
        step.k_idx[r] = qk_r + k9::kIndexHeads * k9::kIndexHeadDim;
    }
    if (qsa || m > uint32_t(chains)) {
        // The verify graph: sessions at rope_delta 0, so every row's table is w.rope_cos_sin.
        const k9::Bounds bounds = k9_bounds(w, ws, w.rope_cos_sin, 1);  // context per row: step.context
        CERE_TRY(k9::qsa_chains_step(step, chains, aw.indexer_k_norm, eps, w.rope_cos_sin, bounds, positions,
                                     stream));
    }
    const bf16 *k = ws.qkv + kQGateWidth;
    const bf16 *v = k + kKvWidth;
    CERE_TRY_CUDA(k6::prologue_chains_at(ws.qkv, kQkvWidth, k, kQkvWidth, v, kQkvWidth, aw.q_norm, aw.k_norm, eps,
                                         w.rope_cos_sin, kv, positions, int64_t(m), q, k_rot, gate, stream));
    CERE_TRY_CUDA(k6::dense_attention_chains_at(q, gate, int64_t(m), positions, kv, ptr, rest, ws.attn_out,
                                                kAttnOutWidth, stream));
    return CERE_OK;
}

// Phase S (m29 design note D6): verify row r's sampler stage on `logits` (that row's lm_head output: its row of
// ws.rows_logits on the batched small-M lm_head, else ws.logits), as sample_row up to its draw, into the row's ws.spec
// buffers. A greedy logprobs-only item (its raw top-2, the greedy path's score, taken by the caller) processes without
// penalties or bias. A sampled row j >= 1 reads the chain's out_counts plus the drafts d_1..d_j (the counts plain
// decode has at p + j once they are committed; row j is consulted only when they were all accepted), kept in
// ws.spec.counts; row 0 reads out_counts itself, as plain decode at p. A constrained chain (GrammarSpec GS-02; always
// sampled) holds n + 1 consecutive masks from s.vocab_mask: row j's (the grammar's mask after d_1..d_j) goes to
// ws.sampler.mask, stream-ordered ahead of this row's process, as sample_row's at p + j.
cere_status verify_sample_stage(Workspace &ws, const bf16 *logits, const VerifyFlat &f, uint32_t r,
                                cudaStream_t stream) {
    const SamplerArgs &s = *f.chain->sampler;
    const SpecSampler &sp = ws.spec;
    const SamplerScratch &sc = ws.sampler;
    float *proc = sp.proc + uint64_t(r) * kVocab;
    if (!s.sampled) {
        return cuda_status(
            k12::process(logits, kVocab, nullptr, 0.0f, 0.0f, 1.0f, nullptr, nullptr, nullptr, 1, 0, proc, stream));
    }
    const cere_sampling_params &p = s.params;
    const uint64_t bias_count = s.bias.size();
    if (f.j == 0 && bias_count > 0) {
        // Pageable source: the copy is staged before cudaMemcpyAsync returns; one upload per chain.
        CERE_TRY_CUDA(cudaMemcpyAsync(sc.bias, s.bias.data(), bias_count * sizeof(cere_logit_bias),
                                      cudaMemcpyHostToDevice, stream));
    }
    const uint32_t *counts = f.session->out_counts;
    if (f.j > 0) {
        if (f.j == 1) CERE_TRY(copy_d2d(sp.counts, f.session->out_counts, kVocab * sizeof(uint32_t), stream));
        // Row j's input token is the draft d_j.
        CERE_TRY_CUDA(k12::count_commit(sp.counts, reinterpret_cast<const uint32_t *>(ws.ids32 + r), 1, stream));
        counts = sp.counts;
    }
    const uint32_t *mask = nullptr;
    if (s.vocab_mask != nullptr) {
        // Pageable source as sample_row's: the ticket's masks may change once the round returns.
        CERE_TRY_CUDA(cudaMemcpyAsync(sc.mask, s.vocab_mask + uint64_t(f.j) * kMaskWords, kMaskWords * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice, stream));
        mask = sc.mask;
    }
    CERE_TRY_CUDA(k12::process(logits, kVocab, counts, p.frequency_penalty, p.presence_penalty,
                               p.repetition_penalty, mask, &sc.bias->id, &sc.bias->bias, 2, int64_t(bias_count),
                               proc, stream));
    CERE_TRY_CUDA(k12::top2_f32(proc, kVocab, sp.top2_ids + 2 * uint64_t(r), sp.top2_scores + 2 * uint64_t(r), stream));
    if (p.temperature != 0.0f) {
        float *probs = sp.probs + uint64_t(r) * kVocab;
        CERE_TRY_CUDA(k12::probabilities(proc, kVocab, p.temperature, probs, sc.lse, sc.scratch, stream));
        CERE_TRY_CUDA(k12::keep_mask(probs, kVocab, int64_t(p.top_k), p.top_p, p.min_p,
                                     sp.keep + uint64_t(r) * kMaskWords, sc.scratch, stream));
    }
    return CERE_OK;
}

// Phase S after every row's stage: a sampled chain with temperature > 0 decides its committed ids with
// k13s::spec_accept (seed = params.seed, key0 = p: row j's counter key is p + j, plain sampling's at that position;
// committed zeroed first so the rows past the chain's count finalize a valid id); then every row of a sampled chain
// gets the score of its committed id (temperature 0: the processed top-1) and, when the chain asks, every row its
// logprobs, staged to ws.host_logprobs[r] (D7). `first` is the chain's first verify row, `c` its index.
cere_status verify_sample_commit(Workspace &ws, const VerifyRow &chain, uint32_t first, uint32_t c,
                                 cudaStream_t stream) {
    const SamplerArgs &s = *chain.sampler;
    const SpecSampler &sp = ws.spec;
    const SamplerScratch &sc = ws.sampler;
    const float temperature = s.sampled ? s.params.temperature : 0.0f;
    if (s.sampled && temperature != 0.0f) {
        CERE_TRY_CUDA(cudaMemsetAsync(sp.committed + first, 0, (chain.n + 1) * sizeof(uint32_t), stream));
        CERE_TRY_CUDA(k13s::spec_accept(sp.probs + uint64_t(first) * kVocab, sp.keep + uint64_t(first) * kMaskWords,
                                        kVocab, reinterpret_cast<const uint32_t *>(ws.ids32 + first + 1),
                                        int64_t(chain.n), s.params.seed, uint32_t(chain.session->position),
                                        sp.committed + first, sp.count + c, sp.accept, stream));
    } else if (s.sampled) {
        // Temperature 0: row j commits its processed top-1 (k12::finalize with no draw).
        CERE_TRY_CUDA(cudaMemcpy2DAsync(sp.committed + first, sizeof(uint32_t), sp.top2_ids + 2 * uint64_t(first),
                                        2 * sizeof(uint32_t), sizeof(uint32_t), chain.n + 1, cudaMemcpyDeviceToDevice,
                                        stream));
    }
    for (uint32_t r = first; r <= first + chain.n; ++r) {
        const float *proc = sp.proc + uint64_t(r) * kVocab;
        if (s.sampled) {
            CERE_TRY_CUDA(k12::finalize(proc, sp.top2_ids + 2 * uint64_t(r),
                                        temperature != 0.0f ? sp.committed + r : nullptr, ws.top2 + r, stream));
        }
        if (!chain.logprobs) continue;
        CERE_TRY_CUDA(k12::top_logprobs(proc, kVocab, temperature, &ws.top2[r].id, 0, int64_t(s.top_logprobs),
                                        &sc.logprobs->logprob, &sc.logprobs->top[0].id, &sc.logprobs->top[0].logprob,
                                        2, stream));
        CERE_TRY_CUDA(cudaMemcpyAsync(ws.host_logprobs + r, sc.logprobs, sizeof(cere_token_logprobs),
                                      cudaMemcpyDeviceToHost, stream));
    }
    return CERE_OK;
}

// Inside a graph_step capture (ws.graphs.capturing): ends the capture of the segment so far into
// ws.graphs.closed[closed_n++] and begins capturing the next on `stream`. Outside a capture, or with the last segment
// already open, nothing. A failure returns its status with the stream not capturing and nothing run (graph_step then
// discards the closed segments and runs the step eagerly).
cere_status graph_split(Workspace &ws, cudaStream_t stream) {
    DecodeGraphs &gs = ws.graphs;
    if (!gs.capturing || gs.closed_n + 1 >= kGraphSegments) return CERE_OK;
    cudaGraph_t graph = nullptr;
    const cudaError_t ended = cudaStreamEndCapture(stream, &graph);
    if (ended != cudaSuccess) {
        if (graph != nullptr) cudaGraphDestroy(graph);
        gs.capturing = false;
        return cuda_status(ended);
    }
    gs.closed[gs.closed_n++] = graph;
    const cudaError_t begun = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (begun != cudaSuccess) gs.capturing = false;
    return cuda_status(begun);
}

// forward_verify_rows's body after the ids upload and before the top-2 download; returns with work enqueued. As
// run_rows (greedy, emit taps only) except that attention, GDN and PLE take the rows' own positions p + j, each row's
// undo copies precede its first state mutation per layer, and the pre-final-mixer HC rows land in ws.hc_rows (when
// the MTP track carved it).
// `positions` (device [m], row r's p + j; the verify graph, every row dense with a prefix: layout::spec_graph_chain_fits)
// makes every position-dependent launch read the row's device position (attention_core_step per row,
// attention_chain's device-position twins, undo_qsa's device key row), so a captured call replays at any position.
cere_status run_verify(const ModelWeights &w, Workspace &ws, const VerifyFlat *rows, uint32_t m, ForwardTaps *taps,
                       const int64_t *positions, cudaStream_t stream) {
    const RowsPlans &p = ws.rows;
    const bool qsa = ws.max_context_tokens > uint64_t(k6::kMaxContext);
    CERE_TRY_CUDA(k1::embedding_gather(w.embed, w.config.vocab_size, kHidden, ws.ids32, m, ws.mixed, stream));
    const auto hyper_from_mixed = [&](uint32_t g) {
        return k1::CopySegment{ws.hyper + uint64_t(g) * kHidden, uint64_t(kHcHidden) * sizeof(bf16), ws.mixed, kRowBytes,
                               kRowBytes, m};
    };
    static_assert(kHc == 4, "run_verify copies the embedding into the kHc streams as one copy_ranges");
    CERE_TRY(copy_ranges(positions != nullptr,
                         {hyper_from_mixed(0), hyper_from_mixed(1), hyper_from_mixed(2), hyper_from_mixed(3)}, stream));
    const k4::Plans ple_plans{&p.ple_key, &p.ple_value};
    bool normed = false;  // ws.normed holds the next mix's norm (combine)
    const auto mix_rows = [&](const HcWeights &hc) { return run_mix(p.hc, ws, hc, m, normed, stream); };
    // The decode plan's layer ops (SelfTune D2, ST1c), in their launch order; the verify graph's segment cuts
    // follow layer L's mlp combine.
    for (const plan::Op &op : plan::kDecode) {
        if (!plan::per_layer(op)) continue;
        const uint32_t L = op.layer;
        const LayerWeights &lw = w.layers[L];
        switch (op.slot) {
        case plan::Slot::kEmbedPle:
            // One k4::forward_rows over every chain (a chain is a sequence): row j < n of a chain saves its state
            // into undo[j] in-kernel, the state before row j + 1 mutates it (M29S2-20).
            CERE_TRY(k4::forward_rows(w.ple, ple_plans, ple_verify_call(rows, m), ws.ids64, m, ws.hyper, true,
                                      ws.hyper, ws.scratch, ws.scratch_bytes, stream));
            break;
        case plan::Slot::kMixHc: {
            const bool attn = op.mix == plan::Mix::kAttn;
            bool edited = false;
            CERE_TRY(research_at(ws, attn ? PointId::kAttnMixIn : PointId::kMlpMixIn, L, 0, m, ws.hyper, kHcHidden,
                                 stream, &edited));
            if (edited) normed = false;
            CERE_TRY(mix_rows(plan_hc(w, L, op.mix)));
            CERE_TRY(research_at(ws, attn ? PointId::kAttnMixOut : PointId::kMlpMixOut, L, 0, m, ws.mixed, kHidden,
                                 stream));
            break;
        }
        case plan::Slot::kAttentionFull: {
            // attention_rows with each row at its own position: row j's K/V (and QSA ring / key) are written
            // before row j + 1 attends; or a chain at once (attention_chain, the verify_attention lever); or, in the
            // verify graph, every sequence's chain at once (attention_chains, rows16 design note R16-C).
            CERE_TRY(p.attn_qkv.run(ws.mixed, lw.attn.qkv, m, ws.qkv, stream));
            CERE_TRY(research_at(ws, PointId::kAttnQkv, L, 0, m, ws.qkv, kQkvWidth, stream));
            const bool all_chains = chains_batched(ws, rows, m, positions);
            if (all_chains) CERE_TRY(attention_chains(w, lw, ws, rows, m, L, positions, stream));
            for (uint32_t r = all_chains ? m : 0; r < m;) {
                const uint32_t c = rows[r].chain->n + 1;  // rows[r] is its chain's row 0
                if (chain_batched(ws, rows[r], c)) {
                    CERE_TRY(attention_chain(w, lw, ws, rows + r, c, L, ws.qkv + uint64_t(r) * kQkvWidth,
                                             ws.mixed + uint64_t(r) * kHidden,
                                             ws.attn_out + uint64_t(r) * kAttnOutWidth,
                                             positions != nullptr ? positions + r : nullptr, stream));
                    r += c;
                    continue;
                }
                for (const uint32_t end = r + c; r < end; ++r) {
                    const VerifyFlat &f = rows[r];
                    const int64_t *position = positions != nullptr ? positions + r : nullptr;
                    CERE_TRY(undo_qsa(f, L, position, qsa, ws.bound_fault, stream));
                    SessionState &s = *f.session;
                    bf16 *qkv = ws.qkv + uint64_t(r) * kQkvWidth;
                    const bf16 *mixed = ws.mixed + uint64_t(r) * kHidden;
                    bf16 *attn_out = ws.attn_out + uint64_t(r) * kAttnOutWidth;
                    s.position = f.p + f.j;
                    const cere_status st =
                        position != nullptr
                            ? attention_core_step(w, lw, ws, s, L, qkv, mixed, attn_out, position, stream)
                            : attention_core(w, lw, ws, s, L, qkv, mixed, attn_out, 1, Rope{rope_table(w, s)},
                                             stream);
                    s.position = f.p;
                    CERE_TRY(st);
                }
            }
            CERE_TRY(p.attn_o.run(ws.attn_out, lw.attn.o_proj, m, ws.block_out, stream));
            CERE_TRY(research_at(ws, PointId::kOProj, L, 0, m, ws.block_out, kHidden, stream));
            CERE_TRY(emit_rows(taps, L, "self_attn.o_proj", ws.block_out, m, stream));
            break;
        }
        case plan::Slot::kAttentionLinearGdn: {
            // GDN: the four in-projections share x, one launch on the small-M kernel (kernels::gemm_segments,
            // bit-equal to the separate calls); then one k5::chain_core over every chain (per chain bit-equal to
            // layer_core once per row in row order; rows16 design note R16-B), which writes undo[j] (the state
            // after row j, right before row j + 1 mutates it) in-kernel (m23 design note DG-02).
            const GdnRows g = carve_gdn_rows(ws.scratch, m);
            const kernels::GemmPlan *plans[4] = {&p.gdn_qkv, &p.gdn_z, &p.gdn_ba, &p.gdn_ba};
            const bf16 *weights[4] = {lw.gdn.in_proj_qkv, lw.gdn.in_proj_z, lw.gdn.in_proj_b, lw.gdn.in_proj_a};
            void *const outs[4] = {g.mixed, g.z, g.b, g.a};
            CERE_TRY(kernels::gemm_segments(plans, weights, outs, 4, ws.mixed, m, stream));
            CERE_TRY(research_at(ws, PointId::kGdnQkv, L, 0, m, g.mixed, k5::kConvDim, stream));
            CERE_TRY(research_at(ws, PointId::kGdnZ, L, 0, m, g.z, k5::kValueDim, stream));
            CERE_TRY(k5::chain_core(lw.gdn, gdn_verify_chains(rows, m, L), g.mixed, g.z, g.b, g.a, m, g.y, g.core,
                                    g.core_bytes, stream));
            CERE_TRY(p.gdn_out.run(g.y, lw.gdn.out_proj, m, ws.block_out, stream));
            CERE_TRY(research_at(ws, PointId::kOutProj, L, 0, m, ws.block_out, kHidden, stream));
            CERE_TRY(emit_rows(taps, L, "linear_attn.out_proj", ws.block_out, m, stream));
            break;
        }
        case plan::Slot::kFfnMoe:
            CERE_TRY(moe_rows(lw, ws, m, L, taps, stream));
            break;
        case plan::Slot::kCombineHc:
            CERE_TRY(combine(ws, plan_hc(w, L, op.mix), m, fused_norm(w, op), stream));
            normed = op.fused;
            if (op.mix == plan::Mix::kMlp && (L == 0 || (L + 1) % kVerifySegmentLayers == 0) && L + 1 < kLayers) {
                CERE_TRY(graph_split(ws, stream));
            }
            break;
        default:
            break;
        }
    }
    // SGLang's hc_hidden_states: the HC rows before the final mixer, the MTP draft-extend input.
    if (ws.hc_rows != nullptr) {
        CERE_TRY(copy_ranges(positions != nullptr, {flat(ws.hc_rows, ws.hyper, uint64_t(m) * kHcHidden * sizeof(bf16))},
                             stream));
    }
    CERE_TRY(mix_rows(w.final_mixer));
    CERE_TRY(research_at(ws, PointId::kFinalMixOut, kLayers, 0, m, ws.mixed, kHidden, stream));
    CERE_TRY(emit_rows(taps, kLayers, "model", ws.mixed, m, stream));
    // lm_head and top-2 as run_rows: once at m into ws.rows_logits on the lm_head rows plan (P11R-21; row r bit-equal
    // to the m = 1 call on row r, so verify == plain decode stays bit-identical), else once per row. A sampled row's
    // score is its committed id's (verify_sample_commit's finalize overwrites the batched top-2).
    const bool batched = m <= ws.rows.lm_head.max_m();
    if (batched) {
        CERE_TRY(ws.rows.lm_head.run(ws.mixed, w.lm_head, m, ws.rows_logits, stream));
        CERE_TRY_CUDA(k1::top2(ws.rows_logits, kVocab, m, kVocab, ws.top2, stream));
    }
    for (uint32_t r = 0; r < m; ++r) {
        const bf16 *logits = batched ? ws.rows_logits + uint64_t(r) * kVocab : ws.logits;
        const SamplerArgs *sampler = rows[r].chain->sampler;
        if (!batched) {
            CERE_TRY(kernels::lm_head_last_row(ws.lm_head, ws.mixed, r + 1, w.lm_head, ws.logits, stream));
            if (sampler == nullptr || !sampler->sampled) {
                CERE_TRY_CUDA(k1::top2(ws.logits, kVocab, 1, kVocab, ws.top2 + r, stream));
            }
        }
        if (ws.research_call != nullptr) CERE_TRY(ws.research_call->logits(r, logits, stream));
        if (sampler != nullptr) CERE_TRY(verify_sample_stage(ws, logits, rows[r], r, stream));
    }
    // Phase S: each sampled chain's decision once all of its rows' stages ran.
    for (uint32_t r = 0, c = 0; r < m; r += rows[r].chain->n + 1, ++c) {
        if (rows[r].chain->sampler != nullptr) CERE_TRY(verify_sample_commit(ws, *rows[r].chain, r, c, stream));
    }
    return CERE_OK;
}

// FP32 copies of the MoE scale sources of one layer, read to the host at INSTANTIATE.
struct MoeScales {
    std::vector<float> gate_in, up_in, down_in, gate_ws2, down_ws2;
};

}  // namespace

cere_status instantiate_model(ModelWeights *weights, Workspace *ws, cublasLtHandle_t gemm, cudaStream_t stream) {
    if (weights == nullptr || ws == nullptr || gemm == nullptr || ws->max_tokens == 0) return CERE_ERR_ARGUMENT;
    const cere_qwen4_exp_config &cfg = weights->config;
    try {
        // GEMM plans (host objects, pinned algorithms).
        const uint32_t max_m = ws->max_tokens;
        using kernels::GemmOut;
        CERE_TRY(ws->hc.init(gemm, hc_shape(cfg), max_m));
        // Every plan of a pinned shape takes gemm_pins_for's pins (k2_gemm.h: torch's pins of attn_qkv and
        // moe_shared_gate_up, the bucket re-pins of the others; a shape without pins gets nullptr) and the one
        // workspace region, except attn_index_qk below.
        const auto pinned = [&](kernels::GemmPlan &plan, uint32_t n, uint32_t k) {
            return plan.init(gemm, n, k, max_m, GemmOut::kBF16, kernels::gemm_pins_for(n, k, GemmOut::kBF16),
                             ws->gemm_workspace, ws->gemm_workspace_bytes);
        };
        CERE_TRY(pinned(ws->attn_qkv, kQkvWidth, kHidden));
        CERE_TRY(pinned(ws->attn_o, kHidden, kAttnOutWidth));
        if (ws->max_context_tokens > uint64_t(k6::kMaxContext)) {
            // moe_shared_gate_up's shape: torch's pins are that plan's, so this one keeps its buckets.
            CERE_TRY(ws->attn_index_qk.init(gemm, k9::kQkWidth, kHidden, max_m, GemmOut::kBF16));
        }
        CERE_TRY(pinned(ws->gdn_qkv, k5::kConvDim, kHidden));
        CERE_TRY(pinned(ws->gdn_z, k5::kValueDim, kHidden));
        CERE_TRY(pinned(ws->gdn_ba, k5::kValueHeads, kHidden));
        CERE_TRY(pinned(ws->gdn_out, kHidden, k5::kValueDim));
        CERE_TRY(pinned(ws->ple_key, k4::kChannels, k4::kEmbedDim));
        CERE_TRY(pinned(ws->ple_value, kHidden, k4::kEmbedDim));
        CERE_TRY(pinned(ws->moe_router, kExperts, kHidden));
        ws->moe_router.use_early_weights(true);  // BeatSglang S4: its w is a model weight (the rows plans copy it)
        CERE_TRY(pinned(ws->moe_shared_gate_up, kSharedInter, kHidden));
        CERE_TRY(pinned(ws->moe_shared_down, kHidden, kSharedInter));
        CERE_TRY(ws->moe_fork.create());
        // m = 1 per scored row (lm_head_last_row); the m-row calls of run_rows and run_verify run the rows plan below.
        CERE_TRY(ws->lm_head.init(gemm, kVocab, kHidden, 1, GemmOut::kBF16));
        // The k2_gemm lever's small_m_a8 (k2-prefill design note P3): the W8A8 algorithms of every prefill plan
        // whose shape an MXFP8 weight has, whatever the lever (a hot swap may select it). A plan cuBLASLt serves no
        // such algorithm keeps the staging path (k2m G1 checks that every served shape has them).
        if (!weights->mxfp8.empty()) {
            kernels::GemmPlan *const prefill[] = {
                &ws->attn_qkv, &ws->attn_o,  &ws->attn_index_qk, &ws->gdn_qkv,   &ws->gdn_z,
                &ws->gdn_ba,   &ws->gdn_out, &ws->ple_key,       &ws->ple_value, &ws->moe_router,
                &ws->moe_shared_gate_up, &ws->moe_shared_down};
            uint32_t eligible = 0;
            uint32_t ready = 0;
            for (kernels::GemmPlan *p : prefill) {
                if (p->max_m() != 0 && weights->mxfp8.has_shape(p->n(), p->k())) {
                    ++eligible;
                    ready += p->init_a8() == CERE_OK ? 1 : 0;
                }
            }
            std::fprintf(stderr, "cere: w8a8 plans %u/%u\n", ready, eligible);
        }
        // The encoder's plans (m213 design note D10), only on a model with a vision tower.
        if (weights->vision.bound() && ws->max_image_patches != 0) {
            CERE_TRY(vision::init_plans(&ws->vision, gemm, ws->max_image_patches));
        }

        // Rows plans of the batched decode step (m26 design note D6), only for a model of more than one session;
        // the plans above (prefill and single-row decode) are untouched.
        ws->rows.rows = 0;
        if (ws->max_rows > 1) {
            RowsPlans &r = ws->rows;
            const uint32_t rows = ws->max_rows;
            CERE_TRY(r.hc.init_rows(gemm, ws->hc, rows));
            CERE_TRY(r.attn_qkv.init_rows(gemm, ws->attn_qkv, rows));
            CERE_TRY(r.attn_o.init_rows(gemm, ws->attn_o, rows));
            CERE_TRY(r.gdn_qkv.init_rows(gemm, ws->gdn_qkv, rows));
            CERE_TRY(r.gdn_z.init_rows(gemm, ws->gdn_z, rows));
            CERE_TRY(r.gdn_ba.init_rows(gemm, ws->gdn_ba, rows));
            CERE_TRY(r.gdn_out.init_rows(gemm, ws->gdn_out, rows));
            CERE_TRY(r.ple_key.init_rows(gemm, ws->ple_key, rows));
            CERE_TRY(r.ple_value.init_rows(gemm, ws->ple_value, rows));
            CERE_TRY(r.moe_router.init_rows(gemm, ws->moe_router, rows));
            CERE_TRY(r.moe_shared_gate_up.init_rows(gemm, ws->moe_shared_gate_up, rows));
            CERE_TRY(r.moe_shared_down.init_rows(gemm, ws->moe_shared_down, rows));
            // Workspace::rows_logits holds min(max_tokens, 16) rows (rows <= kMaxRows already).
            const uint32_t lm_rows = std::min(rows, ws->max_tokens);
            CERE_TRY(r.lm_head.init_rows(gemm, ws->lm_head, lm_rows));
            r.moe_per_row = kMoeRowsPerRow;
            r.rows = rows;
            std::fprintf(stderr, "cere: moe rows path %s\n", kMoeRowsPerRow ? "per_row" : "fused");
            uint32_t lm_fused = 1;
            for (uint32_t m = 2; m <= lm_rows; ++m) lm_fused += r.lm_head.per_row(m) ? 0 : 1;
            std::fprintf(stderr, "cere: lm_head rows plan %u rows, %u fused\n", lm_rows, lm_fused);
        }

        // RoPE table [max_context_tokens, 64] (SGLang's cos/sin cache, theta = rope_theta).
        CERE_TRY_CUDA(k1::rope_table(weights->rope_cos_sin, int64_t(weights->max_context_tokens), int(kRotaryDim),
                                     float(cfg.rope_theta), stream));

        // Read every small source array to the host.
        std::vector<bf16> a_log_src(size_t(kLayers) * kGdnValueHeads);
        std::vector<MoeScales> scales(kLayers);
        const size_t moe_bytes = size_t(kExperts) * sizeof(float);
        for (uint32_t L = 0; L < kLayers; ++L) {
            LayerWeights &lw = weights->layers[L];
            if (!lw.full_attention) {
                if (lw.gdn_A_log_bf16 == nullptr || lw.gdn.A_log == nullptr) return CERE_ERR_ARGUMENT;
                CERE_TRY_CUDA(cudaMemcpyAsync(a_log_src.data() + size_t(L) * kGdnValueHeads, lw.gdn_A_log_bf16,
                                              kGdnValueHeads * sizeof(bf16), cudaMemcpyDeviceToHost, stream));
            }
            const MoeScaleSources &src = lw.moe_scales;
            if (src.gate_input_scale == nullptr || src.up_input_scale == nullptr || src.down_input_scale == nullptr ||
                src.gate_weight_scale_2 == nullptr || src.down_weight_scale_2 == nullptr ||
                lw.moe.g1_alphas == nullptr || lw.moe.g2_alphas == nullptr) {
                return CERE_ERR_ARGUMENT;
            }
            MoeScales &s = scales[L];
            for (std::vector<float> *v : {&s.gate_in, &s.up_in, &s.down_in, &s.gate_ws2, &s.down_ws2}) {
                v->resize(kExperts);
            }
            CERE_TRY_CUDA(cudaMemcpyAsync(s.gate_in.data(), src.gate_input_scale, moe_bytes, cudaMemcpyDeviceToHost,
                                          stream));
            CERE_TRY_CUDA(cudaMemcpyAsync(s.up_in.data(), src.up_input_scale, moe_bytes, cudaMemcpyDeviceToHost,
                                          stream));
            CERE_TRY_CUDA(cudaMemcpyAsync(s.down_in.data(), src.down_input_scale, moe_bytes, cudaMemcpyDeviceToHost,
                                          stream));
            CERE_TRY_CUDA(cudaMemcpyAsync(s.gate_ws2.data(), src.gate_weight_scale_2, moe_bytes,
                                          cudaMemcpyDeviceToHost, stream));
            CERE_TRY_CUDA(cudaMemcpyAsync(s.down_ws2.data(), src.down_weight_scale_2, moe_bytes,
                                          cudaMemcpyDeviceToHost, stream));
        }
        CERE_TRY_CUDA(cudaStreamSynchronize(stream));
        // MF-5: the activation scales are divided into (a1/a2_global_scale = 1 / max) and multiplied into the alphas: a
        // zero, negative or non-finite one would poison every MoE block. Checked for every layer before anything is
        // written to the device.
        for (uint32_t L = 0; L < kLayers; ++L) {
            const MoeScales &s = scales[L];
            for (uint32_t e = 0; e < kExperts; ++e) {
                for (const float v : {s.gate_in[e], s.up_in[e], s.down_in[e]}) {
                    if (!(std::isfinite(v) && v > 0.0f)) return CERE_ERR_ARGUMENT;
                }
            }
        }

        // Derived values (modelopt_quant.py process_weights_after_loading, flashinfer_cutlass; FP32 throughout).
        std::vector<float> a_log(size_t(kLayers) * kGdnValueHeads);
        std::vector<float> g1(size_t(kLayers) * kExperts), g2(size_t(kLayers) * kExperts);
        for (uint32_t L = 0; L < kLayers; ++L) {
            LayerWeights &lw = weights->layers[L];
            if (!lw.full_attention) {
                float *dst = a_log.data() + size_t(L) * kGdnValueHeads;
                for (uint32_t h = 0; h < kGdnValueHeads; ++h) {
                    dst[h] = __bfloat162float(a_log_src[size_t(L) * kGdnValueHeads + h]);
                }
                CERE_TRY_CUDA(cudaMemcpyAsync(const_cast<float *>(lw.gdn.A_log), dst, kGdnValueHeads * sizeof(float),
                                              cudaMemcpyHostToDevice, stream));
            }
            const MoeScales &s = scales[L];
            float w13_max = s.gate_in[0];
            float w2_max = s.down_in[0];
            for (uint32_t e = 0; e < kExperts; ++e) {
                w13_max = std::max(w13_max, std::max(s.gate_in[e], s.up_in[e]));
                w2_max = std::max(w2_max, s.down_in[e]);
            }
            float *g1_l = g1.data() + size_t(L) * kExperts;
            float *g2_l = g2.data() + size_t(L) * kExperts;
            for (uint32_t e = 0; e < kExperts; ++e) {
                g1_l[e] = w13_max * s.gate_ws2[e];
                g2_l[e] = w2_max * s.down_ws2[e];
            }
            CERE_TRY_CUDA(cudaMemcpyAsync(const_cast<float *>(lw.moe.g1_alphas), g1_l, moe_bytes,
                                          cudaMemcpyHostToDevice, stream));
            CERE_TRY_CUDA(cudaMemcpyAsync(const_cast<float *>(lw.moe.g2_alphas), g2_l, moe_bytes,
                                          cudaMemcpyHostToDevice, stream));
            lw.moe.a1_global_scale = 1.0f / w13_max;
            lw.moe.a2_global_scale = 1.0f / w2_max;
        }
        CERE_TRY_CUDA(cudaStreamSynchronize(stream));
#if defined(CERE_MOE_P3B)
        // P3b (p3b/p3b_moe.h, M23-D2): the runner and its pinned tactics, the device {a1, a2} of every layer, and
        // the w13 / down block scales swizzled in place (staged through the idle scratch).
        CERE_TRY(p3b::instantiate());
        std::vector<float> act(2 * size_t(kLayers));
        for (uint32_t L = 0; L < kLayers; ++L) {
            const kernels::MoeLayer &m = weights->layers[L].moe;
            act[2 * L] = m.a1_global_scale;
            act[2 * L + 1] = m.a2_global_scale;
            if (m.act_scales == nullptr) return CERE_ERR_ARGUMENT;
            CERE_TRY_CUDA(cudaMemcpyAsync(const_cast<float *>(m.act_scales), act.data() + 2 * L, 2 * sizeof(float),
                                          cudaMemcpyHostToDevice, stream));
            CERE_TRY(p3b::prepare_layer(kMoeDims, m, ws->scratch, ws->scratch_bytes, stream));
        }
        CERE_TRY_CUDA(cudaStreamSynchronize(stream));
#endif
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    return CERE_OK;
}

void apply_levers(const cere::host::Levers &levers, Workspace *ws) {
    // small_m_a8 (k2-prefill design note P3) is small_m plus the W8A8 path of the plans init_a8 readied.
    const bool small_m = levers.k2_gemm != CERE_LEVER_K2_CUBLASLT;
    const bool a8 = levers.k2_gemm == CERE_LEVER_K2_SMALL_M_A8;
    Workspace::Mtp &mtp = ws->mtp;
    RowsPlans &r = ws->rows;
    vision::Plans &v = ws->vision;
    // Every GemmPlan the model owns (model.h Workspace); an uninitialized plan (no vision, rows 0, no draft) takes
    // the flag too and is never run. Kernel-test plans (ktest/*) are their own and keep init()'s default.
    kernels::GemmPlan *const plans[] = {
        &ws->hc.down, &ws->hc.up, &ws->attn_qkv, &ws->attn_o, &ws->attn_index_qk, &ws->gdn_qkv, &ws->gdn_z,
        &ws->gdn_ba, &ws->gdn_out, &ws->ple_key, &ws->ple_value, &ws->moe_router, &ws->moe_shared_gate_up,
        &ws->moe_shared_down, &ws->lm_head, &v.patch, &v.qkv, &v.proj, &v.fc1, &v.fc2, &v.merge1, &v.merge2,
        &r.hc.down, &r.hc.up, &r.attn_qkv, &r.attn_o, &r.gdn_qkv, &r.gdn_z, &r.gdn_ba, &r.gdn_out, &r.ple_key,
        &r.ple_value, &r.moe_router, &r.moe_shared_gate_up, &r.moe_shared_down, &r.lm_head, &mtp.fc, &mtp.lm_head,
        &mtp.hc_rows.down, &mtp.hc_rows.up, &mtp.fc_rows, &mtp.attn_qkv_rows, &mtp.attn_o_rows, &mtp.router_rows,
        &mtp.shared_gate_up_rows, &mtp.shared_down_rows, &mtp.lm_head_rows};
    const kernels::k2f::Table *fp8 = ws->fp8.empty() ? nullptr : &ws->fp8;
    for (kernels::GemmPlan *p : plans) {
        p->use_small_m(small_m);
        p->use_a8(a8);  // a no-op on every plan init_a8 did not ready (rows, MTP, vision, lm_head, BF16 shapes)
        p->use_fp8(fp8);
        p->use_mxfp8(ws->mxfp8);  // track Fp4S3 (FL-30): null for a BF16 artifact; other weights miss it
        p->use_lora(ws->lora);    // track LoRA: null without a reserve; inactive unless a step's row has an adapter
    }
    ws->prefill_p4 = levers.prefill_attention == CERE_LEVER_PREFILL_P4;
    ws->verify_batched = levers.verify_attention == CERE_LEVER_VERIFY_BATCHED;
    // A captured graph bakes the launches (so the GEMM kernels) of the selection it was captured under.
    release_decode_graphs(ws);
    ws->graphs.enabled =
        levers.decode_step == CERE_LEVER_DECODE_GRAPH || levers.decode_step == CERE_LEVER_DECODE_SPEC_GRAPH;
    ws->graphs.spec = levers.decode_step == CERE_LEVER_DECODE_SPEC_GRAPH;
}

// The sampler preconditions of forward and forward_decode_rows (model.h): the ranges the K12 kernels accept.
cere_status check_sampler(const SamplerArgs &s, uint32_t vocab) {
    if (s.top_logprobs > CERE_MAX_TOP_LOGPROBS) return CERE_ERR_ARGUMENT;
    if (!s.sampled) return CERE_OK;
    const cere_sampling_params &p = s.params;
    if (!(p.temperature >= 0.0f) || p.top_k < 1 || p.top_k > vocab || !(p.top_p > 0.0f && p.top_p <= 1.0f) ||
        !(p.min_p >= 0.0f && p.min_p <= 1.0f) || s.bias.size() > vocab) {
        return CERE_ERR_ARGUMENT;
    }
    for (const cere_logit_bias &b : s.bias) {
        if (b.id >= vocab) return CERE_ERR_ARGUMENT;
    }
    return CERE_OK;
}

// model.h take_bound_fault: the word is written by the device through the mapped staging block and read here after
// the drain, so the read is volatile (never cached across calls).
cere_status take_bound_fault(Workspace &ws) {
    if (ws.bound_fault == nullptr) return CERE_OK;
    volatile uint32_t *word = ws.bound_fault;
    const uint32_t code = *word;
    if (code == 0) return CERE_OK;
    *word = 0;
    std::fprintf(stderr, "cere: device_bound_fault code=%u\n", unsigned(code));
    return CERE_ERR_INTERNAL;
}

namespace {

// The committed token's logprobs from the staged device entry: the device wrote logprob and the top entries; id,
// top_count and the unused entries come from here.
cere_token_logprobs staged_logprobs(const cere_token_logprobs &staged, uint32_t id, uint32_t top_count) {
    cere_token_logprobs lp{};
    lp.id = id;
    lp.logprob = staged.logprob;
    lp.top_count = top_count;
    for (uint32_t i = 0; i < top_count; ++i) lp.top[i] = staged.top[i];
    return lp;
}

// DecodeGraph::key entries of one row: the slot pointer, the session's max_context_tokens (bind_session carves the
// slot by it: the slot map is int64 [max_context_tokens], so the GDN, PLE and QSA-ring state after it move with it)
// and the decode_dense bit.
void row_key(const Workspace &ws, const SessionState &s, uint64_t *key) {
    key[0] = uint64_t(reinterpret_cast<uintptr_t>(s.slot));
    key[1] = s.max_context_tokens;
    key[2] = decode_dense(ws, s) ? 1 : 0;
}

// Destroys the executables of `g` and sets its segment count to 0.
void drop_graph(DecodeGraph &g) {
    for (uint32_t i = 0; i < g.segments; ++i) cudaGraphExecDestroy(g.exec[i]);
    std::fill(g.exec, g.exec + kGraphSegments, nullptr);
    g.segments = 0;
    g.failed = false;
}

// Synchronizes the step's stream and, when the model has one, the MoE fork's side stream (MF-3: shared-expert work
// enqueued on it must not outlive the call, whatever the status). The first error wins.
cudaError_t drain(cudaStream_t stream, cudaStream_t side) {
    cudaError_t e = cudaStreamSynchronize(stream);
    if (side != nullptr) {
        const cudaError_t s = cudaStreamSynchronize(side);
        if (e == cudaSuccess) e = s;
    }
    return e;
}

// Runs one step, `enqueue` (stream work only: no host read of device data, no host scalar a later step changes but
// `key`), through the graph `g` (ws.graphs.graph[index], .verify[m] or .draft[rows]; `counts` its counters): a replay
// when key[0, key_n) matches the graph's, else a capture of `enqueue` (thread-local capture mode, cut into segments
// where it calls graph_split) that updates the executables in place, or instantiates them when there are none, their
// segment count differs or an update is refused. A replay launches the segments in order. Returns with the launches
// enqueued; the caller synchronizes. A capture or an instantiation that fails runs the step eagerly instead
// (`enqueue` outside capture; nothing of the step had run, the captured work is recorded, not launched): a failed
// capture leaves the previous executables and key as they were, a failed instantiation leaves none (the next
// eligible step captures again). The first capture of `g` logs `cere: <what>_capture_ms=<ms> rows=<rows>`.
template <typename F>
cere_status graph_step(DecodeGraphs &gs, DecodeGraph &g, GraphCounts &counts, const char *what, uint32_t rows,
                       const uint64_t *key, size_t key_n, cudaStream_t stream, F &&enqueue) {
    if (g.failed && std::equal(key, key + key_n, g.key)) return enqueue();
    if (g.segments == 0 || !std::equal(key, key + key_n, g.key)) {
        const auto t0 = std::chrono::steady_clock::now();
        CERE_TRY_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        gs.closed_n = 0;
        gs.capturing = true;
        const cere_status st = enqueue();
        // graph_split clears `capturing` when it left the stream outside a capture (st is then its failure)
        const bool open = gs.capturing;
        gs.capturing = false;
        cudaGraph_t last = nullptr;
        const cudaError_t ended = open ? cudaStreamEndCapture(stream, &last) : cudaErrorStreamCaptureInvalidated;
        const uint32_t n = gs.closed_n + (ended == cudaSuccess ? 1 : 0);
        if (ended == cudaSuccess) gs.closed[gs.closed_n] = last;
        gs.closed_n = 0;
        const auto release = [&]() {
            for (uint32_t i = 0; i < n; ++i) cudaGraphDestroy(gs.closed[i]);
        };
        if (st != CERE_OK || ended != cudaSuccess) {
            if (last != nullptr && ended != cudaSuccess) cudaGraphDestroy(last);
            release();
            cudaGetLastError();
            return enqueue();
        }
        bool updated = g.segments == n;
        for (uint32_t i = 0; i < n && updated; ++i) {
            cudaGraphExecUpdateResultInfo info{};
            updated = cudaGraphExecUpdate(g.exec[i], gs.closed[i], &info) == cudaSuccess;
        }
        cudaError_t made = cudaSuccess;
        if (!updated) {
            cudaGetLastError();
            drop_graph(g);
            for (uint32_t i = 0; i < n && made == cudaSuccess; ++i) {
                made = cudaGraphInstantiate(&g.exec[i], gs.closed[i], 0);
                // OH-08: the instantiation fails (DM-5).
                CERE_FAULT_POINT(graph_instantiate, if (made == cudaSuccess) {
                    cudaGraphExecDestroy(g.exec[i]);
                    g.exec[i] = nullptr;
                    made = cudaErrorNotSupported;
                });
                if (made == cudaSuccess) g.segments = i + 1;
            }
        }
        release();
        if (made != cudaSuccess) {
            drop_graph(g);
            cudaGetLastError();
            // DM-5: latch the key: its next steps run eagerly (a fresh capture and instantiation every step would
            // fail the same way at ms of host cost each), reported once.
            std::copy(key, key + key_n, g.key);
            g.failed = true;
            std::fprintf(stderr, "cere: %s_instantiate_failed rows=%u error=%s (steps of this key run eagerly)\n", what,
                         rows, cudaGetErrorString(made));
            return enqueue();
        }
        std::copy(key, key + key_n, g.key);
        ++(updated ? counts.updates : counts.captures);
        if (!g.logged) {
            const std::chrono::duration<double, std::milli> ms = std::chrono::steady_clock::now() - t0;
            std::fprintf(stderr, "cere: %s_capture_ms=%.3f rows=%u\n", what, ms.count(), rows);
            g.logged = true;
        }
    }
    for (uint32_t i = 0; i < g.segments; ++i) CERE_TRY_CUDA(cudaGraphLaunch(g.exec[i], stream));
    ++counts.replays;
    return CERE_OK;
}

// WeightLab research: runs `enqueue` with ws.research_call at `call` when the call has a planned row (else exactly
// `enqueue()`), then enqueues the call's staged captures to the host ring; ws.research_call is null again on return.
template <typename F>
cere_status with_research(Workspace &ws, research::Call &call, cudaStream_t stream, F &&enqueue) {
    if (!call.active()) return enqueue();
    ws.research_call = &call;
    cere_status st = enqueue();
    ws.research_call = nullptr;
    if (st == CERE_OK) st = call.flush(stream);
    return st;
}

}  // namespace

cere_status forward(const ModelWeights &weights, Workspace &ws, SessionState &session, uint32_t operation,
                    const uint32_t *tokens, uint32_t n, ForwardTaps *taps, cere_token_score *out_top2_last_or_all,
                    const SamplerArgs *sampler, cere_token_logprobs *out_logprobs, cudaStream_t stream,
                    bool &launched, const host::StepImages *images, const EndCall *end, Segment *segment) {
    launched = false;
    // Preconditions, before any launch.
    if (tokens == nullptr || out_top2_last_or_all == nullptr || n == 0 || n > ws.max_tokens) return CERE_ERR_ARGUMENT;
    if (operation != CERE_OP_PREFILL && (operation != CERE_OP_DECODE || n != 1)) return CERE_ERR_ARGUMENT;
    if (session.position + n > session.max_context_tokens) return CERE_ERR_ARGUMENT;
    if (end != nullptr) {
        const bool capture = end->row >= 0;
        const bool resume = end->in_a != nullptr;
        const bool pairs_ok =
            (!capture || (end->a != nullptr && end->b != nullptr)) && (!resume || end->in_b != nullptr) &&
            (!capture || !resume ||
             (end->a != end->in_a && end->a != end->in_b && end->b != end->in_a && end->b != end->in_b));
        const bool kv_ok = end->kv == nullptr ||
                           (capture && end->kv_rows > 0 &&
                            uint64_t(end->kv_rows) == session.position + uint64_t(end->row) &&
                            (session.position == 0 || end->in_kv != nullptr));
        const bool in_kv_ok = end->in_kv == nullptr ||
                              (resume && end->in_kv_rows > 0 && session.position == uint64_t(end->in_kv_rows));
        if (operation != CERE_OP_PREFILL || (end->cls != 0 && (end->cls < n || end->cls > ws.max_tokens)) ||
            (capture && (end->row == 0 || end->row >= int64_t(n) || end->row % k5::kChunkTokens != 0)) ||
            !pairs_ok || !kv_ok || !in_kv_ok) {
            return CERE_ERR_ARGUMENT;
        }
    }
    if (taps != nullptr && (taps->detail > 1 || (taps->detail == 1 && taps->detail_sink == nullptr) ||
                            (taps->emit == nullptr && taps->detail == 0 && taps->patch == nullptr))) {
        return CERE_ERR_ARGUMENT;
    }
    const uint32_t vocab = weights.config.vocab_size;
    if (sampler != nullptr) {
        if (taps != nullptr) return CERE_ERR_ARGUMENT;
        if (const cere_status st = check_sampler(*sampler, vocab); st != CERE_OK) return st;
    } else if (out_logprobs != nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    for (uint32_t t = 0; t < n; ++t) {
        if (tokens[t] >= vocab) return CERE_ERR_ARGUMENT;
    }
    const uint64_t rows = taps == nullptr ? 1 : (taps->last_rows == 0 || taps->last_rows > n ? n : taps->last_rows);
    // MixedPD: a segmented call is an eager PREFILL without taps; a resume continues the paused call's rows.
    Park *park = segment != nullptr ? segment->park : nullptr;
    if (segment != nullptr) {
        segment->paused = false;
        if (operation != CERE_OP_PREFILL || taps != nullptr || park == nullptr || park->hyper == nullptr ||
            park->normed == nullptr || segment->layers == 0 || (park->next_layer != 0 && park->n != n)) {
            return CERE_ERR_ARGUMENT;
        }
    }
    const bool resume = park != nullptr && park->next_layer != 0;

    // Images (m213 design note D10, D11; m213l design note VL-L3): checks before any launch. Only an open prompt
    // carries them (the host passes none once the prompt is fed, and a session restored from a kind-11 snapshot has
    // none, M2.13 C14): later calls rotate through session.rope_delta.
    if (images != nullptr) {
        if (operation != CERE_OP_PREFILL || !weights.vision.bound() || ws.max_image_patches == 0 ||
            images->positions3 == nullptr || images->image == nullptr || images->count == 0 ||
            n > layout::kMaxImageTokens || images->prompt_len < session.position + n) {
            return CERE_ERR_ARGUMENT;
        }
        for (uint64_t i = 0; i < images->count; ++i) {
            const cere::host::SessionImage &im = images->image[i];
            const uint64_t p = uint64_t(im.span.grid_t) * im.span.grid_h * im.span.grid_w;
            const uint64_t merged = cere::mrope::image_tokens(im.span);
            if (im.span.grid_h % 2 != 0 || im.span.grid_w % 2 != 0 || p == 0 || p > ws.max_image_patches ||
                merged > layout::kMaxImageTokens || im.pixels.size() != p * vision::kPatchRow * sizeof(bf16) ||
                im.pixels.size() > ws.image_patch_bytes || im.span.token_offset + merged > images->prompt_len) {
                return CERE_ERR_ARGUMENT;
            }
        }
    }
    // WeightLab research: a planned session's rows [0, n) at its position (never with taps; such a call runs eager).
    research::Call research_call(ws.research);
    if (session.research != nullptr) {
        if (taps != nullptr) return CERE_ERR_ARGUMENT;
        CERE_TRY(research_call.add(session.research, 0, n, session.position));
    }
    launched = true;  // past the preconditions: every return below follows device work or a device failure

    // One encoder pass per image whose pad rows overlap the call's rows, before the call's first use of the workspace
    // (the encoder's scratch aliases the region): its patches through the mapped staging, vision::encode into
    // ws.image_out, then the overlap's merged rows to ws.image_embeds at their call rows (run_layers splices them).
    // The staging holds one image, so the stream drains before the next image overwrites it. MixedPD: a resumed call
    // encoded at its first segment (only that segment splices).
    static_assert(vision::kOutHidden == kHidden, "the merged rows are hidden-wide");
    if (images != nullptr && !resume) {
        const uint64_t first = session.position;
        for (uint64_t i = 0; i < images->count; ++i) {
            const cere::host::SessionImage &im = images->image[i];
            const uint64_t lo = std::max<uint64_t>(im.span.token_offset, first);
            const uint64_t hi = std::min<uint64_t>(im.span.token_offset + cere::mrope::image_tokens(im.span), first + n);
            if (lo >= hi) continue;
            std::memcpy(ws.host_image_patches, im.pixels.data(), im.pixels.size());
            cere_status st = cuda_status(cudaMemcpyAsync(ws.image_patches, ws.host_image_patches, im.pixels.size(),
                                                         cudaMemcpyHostToDevice, stream));
            if (st == CERE_OK) {
                st = vision::encode(weights.vision, ws.vision, ws.image_patches, im.span.grid_t, im.span.grid_h,
                                    im.span.grid_w, ws.region, ws.region_bytes, ws.image_out,
                                    taps == nullptr ? nullptr : taps->vision, stream);
            }
            if (st == CERE_OK) {
                st = copy_d2d(ws.image_embeds + (lo - first) * kHidden,
                              ws.image_out + (lo - im.span.token_offset) * kHidden, (hi - lo) * kRowBytes, stream);
            }
            const cudaError_t sync = cudaStreamSynchronize(stream);
            if (st != CERE_OK) return sync == cudaSuccess ? st : CERE_ERR_INTERNAL;
            CERE_TRY_CUDA(sync);
        }
    }

    // Token ids through the mapped staging (free: the previous forward synchronized the stream). PLE reads each
    // image's pad_id at its pad rows (D16); an image call's rotary positions are the prompt's columns (prompt_rope:
    // K1/K6's [3, n] of the call's rows, then K9's [3, lead + n] from the first member of the call's first group).
    for (uint32_t t = 0; t < n; ++t) {
        ws.host_ids32[t] = int32_t(tokens[t]);
        ws.host_ids64[t] = int64_t(tokens[t]);
    }
    static_assert(k9::kRatio == 4, "prompt_rope's K9 lead is position % 4");
    if (images != nullptr) {
        const uint64_t first = session.position;
        for (uint64_t i = 0; i < images->count; ++i) {
            const cere::host::SessionImage &im = images->image[i];
            const uint64_t lo = std::max<uint64_t>(im.span.token_offset, first);
            const uint64_t hi = std::min<uint64_t>(im.span.token_offset + cere::mrope::image_tokens(im.span),
                                                   first + n);
            for (uint64_t r = lo; r < hi; ++r) ws.host_ids64[r - first] = int64_t(im.pad_id);
        }
        const uint64_t lead = first % uint64_t(k9::kRatio);
        const uint64_t k9_cols = lead + n;
        int64_t *k9_positions = ws.host_positions3 + 3 * uint64_t(n);
        for (uint32_t a = 0; a < 3; ++a) {
            const int64_t *axis = images->positions3 + a * images->prompt_len;
            for (uint32_t t = 0; t < n; ++t) ws.host_positions3[a * n + t] = axis[first + t];
            for (uint64_t c = 0; c < k9_cols; ++c) k9_positions[a * k9_cols + c] = axis[first - lead + c];
        }
        CERE_TRY_CUDA(cudaMemcpyAsync(ws.positions3, ws.host_positions3, 3 * (n + k9_cols) * sizeof(int64_t),
                                      cudaMemcpyHostToDevice, stream));
    }
    // The decode graph (m23 design note P1, M23P1-01, M23P1-10): a greedy, unconstrained DECODE without taps or
    // logprobs after the first token whose position fits ids64's tail, of a session at rope_delta 0 (its launches read
    // the model's table), on either MoE backend; everything else stays eager. M2.9's VERIFY (forward_verify_rows),
    // draft-extend and MTP steps never come here (M23P1-11, M29S2-18): the host (api.cpp) decodes a drafting session
    // only through speculate_rows.
    // Track LoRA: a session with an adapter runs eager (Workspace::lora); the table is active for this call only.
    // WeightLab research: a call with a planned row runs eager too (research_call).
    const bool graph = ws.graphs.enabled && operation == CERE_OP_DECODE && taps == nullptr &&
                       sampler == nullptr && session.rope_delta == 0 && session.position > 0 && session.lora_slot < 0 &&
                       !research_call.active() && layout::decode_graph_fits(n, ws.max_tokens);
    const kernels::lora::Scope lora_scope(ws.lora, session.lora_slot);
    // MixedPD: a segment's layers; a resume restores the carried rows first, a pause saves them last (D2D, n rows).
    LayerRange range;
    if (park != nullptr) {
        range.first = park->next_layer;
        range.last = uint32_t(std::min<uint64_t>(uint64_t(range.first) + segment->layers, kLayers));
        range.normed = resume && park->normed_valid;
    }
    const uint64_t carried = uint64_t(n) * kHcHidden * sizeof(bf16);
    const auto enqueue = [&]() -> cere_status {
        CERE_TRY_CUDA(cudaMemcpyAsync(ws.ids32, ws.host_ids32, n * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CERE_TRY_CUDA(cudaMemcpyAsync(ws.ids64, ws.host_ids64, (graph ? 2 * n : n) * sizeof(int64_t),
                                      cudaMemcpyHostToDevice, stream));
        if (resume) {
            CERE_TRY(copy_d2d(ws.hyper, park->hyper, carried, stream));
            if (range.normed) CERE_TRY(copy_d2d(ws.normed, park->normed, carried, stream));
        }
        CERE_TRY(run_layers(weights, ws, session, operation, n, rows, taps, sampler, out_logprobs != nullptr, images,
                            graph ? ws.ids64 + n : nullptr, stream, end, park != nullptr ? &range : nullptr));
        if (park != nullptr && range.last < kLayers) {
            CERE_TRY(copy_d2d(park->hyper, ws.hyper, carried, stream));
            if (range.normed) CERE_TRY(copy_d2d(park->normed, ws.normed, carried, stream));
        }
        return CERE_OK;
    };
    cere_status status = CERE_OK;
    if (graph) {
        ws.host_ids64[n] = int64_t(session.position);
        uint64_t key[3];
        row_key(ws, session, key);
        status = graph_step(ws.graphs, ws.graphs.graph[0], ws.graphs.counts, "decode_graph", 1, key, 3, stream,
                            enqueue);
    } else {
        status = with_research(ws, research_call, stream, enqueue);
    }
    // OH-08: a failure after the launches whose status is not INTERNAL (MF-2), or a sticky launch failure that the
    // drain below reports (DM-2).
    CERE_FAULT_POINT(forward_launch, if (status == CERE_OK) status = CERE_ERR_CAPACITY);
    CERE_FAULT_POINT(forward_sticky, ::cere::fault::trap(stream));
    CERE_FAULT_POINT(bound_fault, if (ws.bound_fault != nullptr) *ws.bound_fault = k9::kBoundFault);
    // Drain the stream even on failure so no enqueued work outlives the call; the bound fault word is taken (and
    // cleared) whatever the status, before the position advances.
    const cudaError_t sync = drain(stream, ws.moe_fork.side);
    const cere_status bound = take_bound_fault(ws);
    if (park != nullptr && (status != CERE_OK || sync != cudaSuccess || bound != CERE_OK)) {
        park->next_layer = 0;  // nothing to resume
    }
    if (status != CERE_OK) return status;
    CERE_TRY_CUDA(sync);
    CERE_TRY(bound);
    if (park != nullptr && range.last < kLayers) {
        park->next_layer = range.last;
        park->n = n;
        park->normed_valid = range.normed;
        segment->paused = true;
        return CERE_OK;
    }
    if (park != nullptr) park->next_layer = 0;
    if (taps == nullptr || !taps->attn_inputs_only) {
        std::memcpy(out_top2_last_or_all, ws.host_top2, rows * sizeof(cere_token_score));
    }
    if (out_logprobs != nullptr) {
        *out_logprobs = staged_logprobs(*ws.host_logprobs, out_top2_last_or_all[0].id, sampler->top_logprobs);
    }
    session.position += n;
    return CERE_OK;
}

cere_status forward_decode_rows(const ModelWeights &w, Workspace &ws, SessionState *const *sessions, uint32_t rows,
                                const uint32_t *tokens, ForwardTaps *taps, cere_token_score *out,
                                const SamplerArgs *const *samplers, cere_token_logprobs *const *out_logprobs,
                                cudaStream_t stream, bool &launched) {
    launched = false;
    // Preconditions, before any launch.
    if (sessions == nullptr || tokens == nullptr || out == nullptr || ws.rows.rows == 0 || rows == 0 ||
        rows > ws.rows.rows || rows > ws.max_tokens || carve_gdn_rows(nullptr, rows).bytes > ws.scratch_bytes) {
        return CERE_ERR_ARGUMENT;
    }
    if (taps != nullptr &&
        (taps->emit == nullptr || taps->detail != 0 || taps->patch != nullptr || taps->attn_inputs_only)) {
        return CERE_ERR_ARGUMENT;
    }
    if (rows > staging_logprobs_rows(ws.max_tokens)) return CERE_ERR_ARGUMENT;
    const uint32_t vocab = w.config.vocab_size;
    for (uint32_t r = 0; r < rows; ++r) {
        const SessionState *s = sessions[r];
        if (s == nullptr || tokens[r] >= vocab || s->position + 1 > s->max_context_tokens) return CERE_ERR_ARGUMENT;
        for (uint32_t q = 0; q < r; ++q) {
            if (sessions[q] == s) return CERE_ERR_ARGUMENT;
        }
        const SamplerArgs *sampler = samplers != nullptr ? samplers[r] : nullptr;
        const bool logprobs = out_logprobs != nullptr && out_logprobs[r] != nullptr;
        if (sampler != nullptr) {
            if (taps != nullptr) return CERE_ERR_ARGUMENT;
            if (const cere_status st = check_sampler(*sampler, vocab); st != CERE_OK) return st;
        } else if (logprobs) {
            return CERE_ERR_ARGUMENT;
        }
    }
    // WeightLab research: row r of a planned session at its position (never with taps; such a step runs eager).
    research::Call research_call(ws.research);
    for (uint32_t r = 0; r < rows; ++r) {
        if (sessions[r]->research == nullptr) continue;
        if (taps != nullptr) return CERE_ERR_ARGUMENT;
        CERE_TRY(research_call.add(sessions[r]->research, r, 1, sessions[r]->position));
    }
    launched = true;  // everything below enqueues or fails after the first launch

    // Token ids through the mapped staging (free: the previous forward synchronized the stream).
    for (uint32_t r = 0; r < rows; ++r) {
        ws.host_ids32[r] = int32_t(tokens[r]);
        ws.host_ids64[r] = int64_t(tokens[r]);
    }
    bool graph = ws.graphs.enabled && taps == nullptr && rows <= kernels::GemmPlan::kMaxRows &&
                 !research_call.active() && layout::decode_graph_fits(rows, ws.max_tokens);
    for (uint32_t r = 0; r < rows && graph; ++r) {
        graph = (samplers == nullptr || samplers[r] == nullptr) && sessions[r]->position > 0 &&
                sessions[r]->rope_delta == 0 && sessions[r]->lora_slot < 0;
    }
    // Track LoRA: row r runs its session's adapter (none: base); any adapter row makes the step eager (above).
    int32_t lora_slots[kernels::lora::kMaxRows];
    for (uint32_t r = 0; r < kernels::lora::kMaxRows; ++r) {
        lora_slots[r] = r < rows ? sessions[r]->lora_slot : -1;
    }
    const kernels::lora::Scope lora_scope(ws.lora, lora_slots, rows);
    const auto enqueue = [&]() -> cere_status {
        CERE_TRY_CUDA(
            cudaMemcpyAsync(ws.ids32, ws.host_ids32, rows * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CERE_TRY_CUDA(cudaMemcpyAsync(ws.ids64, ws.host_ids64, (graph ? 2 * rows : rows) * sizeof(int64_t),
                                      cudaMemcpyHostToDevice, stream));
        return run_rows(w, ws, sessions, rows, taps, samplers, out_logprobs, graph ? ws.ids64 + rows : nullptr,
                        stream);
    };
    cere_status status = CERE_OK;
    if (graph) {
        uint64_t key[3 * kernels::GemmPlan::kMaxRows];
        for (uint32_t r = 0; r < rows; ++r) {
            ws.host_ids64[rows + r] = int64_t(sessions[r]->position);
            row_key(ws, *sessions[r], key + 3 * r);
        }
        status = graph_step(ws.graphs, ws.graphs.graph[rows], ws.graphs.counts, "decode_graph", rows, key,
                            3 * size_t(rows), stream, enqueue);
    } else {
        status = with_research(ws, research_call, stream, enqueue);
    }
    // Drain the stream even on failure so no enqueued work outlives the call (the bound fault word as forward).
    const cudaError_t sync = drain(stream, ws.moe_fork.side);
    const cere_status bound = take_bound_fault(ws);
    if (status != CERE_OK) return status;
    CERE_TRY_CUDA(sync);
    CERE_TRY(bound);
    std::memcpy(out, ws.host_top2, rows * sizeof(cere_token_score));
    for (uint32_t r = 0; r < rows; ++r) {
        if (out_logprobs != nullptr && out_logprobs[r] != nullptr) {
            *out_logprobs[r] = staged_logprobs(ws.host_logprobs[r], out[r].id, samplers[r]->top_logprobs);
        }
    }
    for (uint32_t r = 0; r < rows; ++r) {
        ++sessions[r]->position;
    }
    return CERE_OK;
}

void release_decode_graphs(Workspace *ws) {
    for (DecodeGraph &g : ws->graphs.graph) drop_graph(g);
    for (DecodeGraph &g : ws->graphs.verify) drop_graph(g);
    for (DecodeGraph &g : ws->graphs.draft) drop_graph(g);
}

MoeFork::~MoeFork() {
    if (join != nullptr) cudaEventDestroy(join);
    if (fork != nullptr) cudaEventDestroy(fork);
    if (side != nullptr) cudaStreamDestroy(side);
}

cere_status MoeFork::create() {
    if (side == nullptr) CERE_TRY_CUDA(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
    if (fork == nullptr) CERE_TRY_CUDA(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    if (join == nullptr) CERE_TRY_CUDA(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
    return CERE_OK;
}

cere_status forward_verify_rows(const ModelWeights &w, Workspace &ws, const VerifyRow *rows, uint32_t row_count,
                                ForwardTaps *taps, cere_token_score *out, cudaStream_t stream, bool &launched,
                                cere_token_logprobs *out_logprobs) {
    launched = false;
    constexpr uint32_t kMaxDrafts = uint32_t(layout::kMaxChainDrafts);
    constexpr uint32_t kMaxRows = kernels::GemmPlan::kMaxRows;
    // Preconditions, before any launch.
    if (rows == nullptr || out == nullptr || row_count == 0 || ws.rows.rows == 0) {
        return CERE_ERR_ARGUMENT;
    }
    if (taps != nullptr &&
        (taps->emit == nullptr || taps->detail != 0 || taps->patch != nullptr || taps->attn_inputs_only)) {
        return CERE_ERR_ARGUMENT;
    }
    const uint32_t vocab = w.config.vocab_size;
    const uint64_t max_rows = std::min<uint64_t>({ws.rows.rows, kMaxRows, ws.max_tokens});
    uint64_t m = 0;
    for (uint32_t c = 0; c < row_count; ++c) {
        const VerifyRow &v = rows[c];
        const SessionState *s = v.session;
        if (s == nullptr || v.tokens == nullptr || v.n > kMaxDrafts || (v.n > 0 && v.undo == nullptr)) {
            return CERE_ERR_ARGUMENT;
        }
        for (uint32_t q = 0; q < c; ++q) {
            if (rows[q].session == s) return CERE_ERR_ARGUMENT;
        }
        if (s->position >= s->max_context_tokens || v.n >= s->max_context_tokens - s->position ||
            v.n + 1 > max_rows - m) {
            return CERE_ERR_ARGUMENT;
        }
        for (uint32_t j = 0; j <= v.n; ++j) {
            if (v.tokens[j] >= vocab) return CERE_ERR_ARGUMENT;
        }
        if (v.sampler == nullptr) {
            if (v.logprobs) return CERE_ERR_ARGUMENT;
        } else {
            // Phase S: the draft-only region's stages hold every row; a constrained chain (GrammarSpec GS-02) holds
            // one mask per row (verify_sample_stage).
            if (m + v.n + 1 > ws.spec.rows || (v.logprobs && out_logprobs == nullptr)) return CERE_ERR_ARGUMENT;
            if (const cere_status st = check_sampler(*v.sampler, vocab); st != CERE_OK) return st;
        }
        m += v.n + 1;
    }
    if (carve_gdn_rows(nullptr, uint32_t(m)).bytes > ws.scratch_bytes) return CERE_ERR_ARGUMENT;

    // The K9 path (contexts above 2,048) completes group g's compressed key at position 4g + 3. Row j of a chain
    // (1..n) saves into undo[j - 1] the pool row of the first group it or a later row of the chain completes (the
    // first q >= p + j with q % 4 == 3 and q <= p + n; k9::qsa_key_save_step's rule), -1 when none; a chain of up to
    // kMaxChainDrafts (15) drafts completes at most four groups (verify_commit restores each one past the commit).
    const bool qsa = ws.max_context_tokens > uint64_t(k6::kMaxContext);
    int64_t key_rows[kMaxRows][kMaxDrafts] = {};
    for (uint32_t c = 0; c < row_count; ++c) {
        const VerifyRow &v = rows[c];
        const uint64_t p = v.session->position;
        constexpr uint64_t kRatio = uint64_t(k9::kRatio);
        for (uint32_t j = 1; j <= v.n; ++j) {
            key_rows[c][j - 1] = -1;
            const uint64_t q = p + j + (kRatio - 1 - (p + j) % kRatio) % kRatio;
            if (!qsa || q > p + v.n) continue;
            const uint64_t first = q + 1 - kRatio;  // 4g, the group's first position
            const uint64_t page = first / 64;
            if (v.pages == nullptr || page >= v.n_pages) return CERE_ERR_ARGUMENT;
            key_rows[c][j - 1] = int64_t((uint64_t(v.pages[page]) * 64 + first % 64) / kRatio);
        }
    }
    // The verify graph (decode_step = spec_graph, m29 design note M29S2-18b): greedy chains without taps, each
    // dense with a prefix (layout::spec_graph_chain_fits) of a session at rope_delta 0 (the graph reads the model's
    // table), whose m rows' positions fit ids64's tail; else eager.
    // WeightLab research: chain c's rows (p + j) of a planned session (never with taps; such a call runs eager). A
    // rejected row's capture is rewritten when its position is recomputed, and a released capture keeps the rows
    // below the session's fed count only (research/capture.h), so only accepted rows are ever read.
    research::Call research_call(ws.research);
    for (uint32_t c = 0, r = 0; c < row_count; r += rows[c].n + 1, ++c) {
        if (rows[c].session->research == nullptr) continue;
        if (taps != nullptr) return CERE_ERR_ARGUMENT;
        CERE_TRY(research_call.add(rows[c].session->research, r, rows[c].n + 1, rows[c].session->position));
    }
    bool graph = ws.graphs.spec && taps == nullptr && !research_call.active() &&
                 layout::decode_graph_fits(m, ws.max_tokens);
    for (uint32_t c = 0; c < row_count && graph; ++c) {
        graph = rows[c].sampler == nullptr && rows[c].session->rope_delta == 0 &&
                layout::spec_graph_chain_fits(rows[c].session->position, rows[c].n) && rows[c].session->lora_slot < 0;
    }
    // Track LoRA: every flat row of chain c runs chain c's adapter (none: base); any adapter makes the call eager.
    int32_t lora_slots[kMaxRows];
    for (uint32_t c = 0, at = 0; c < row_count; ++c) {
        for (uint32_t j = 0; j <= rows[c].n; ++j) lora_slots[at++] = rows[c].session->lora_slot;
    }
    const kernels::lora::Scope lora_scope(ws.lora, lora_slots, uint32_t(m));
    VerifyFlat flat[kMaxRows];
    uint32_t at = 0;
    for (uint32_t c = 0; c < row_count; ++c) {
        const VerifyRow &v = rows[c];
        const uint64_t p = v.session->position;
        for (uint32_t j = 0; j <= v.n; ++j) {
            flat[at] = VerifyFlat{v.session, v.undo, p, j, &v};
            ws.host_ids32[at] = int32_t(v.tokens[j]);
            ws.host_ids64[at] = int64_t(v.tokens[j]);
            if (graph) ws.host_ids64[m + at] = int64_t(p + j);
            ++at;
            if (j > 0) v.undo[j - 1].key_row = key_rows[c][j - 1];
        }
    }
    launched = true;  // the first enqueue follows
    // Token ids (and a graph's row positions) through the mapped staging (free: the previous forward synchronized
    // the stream), stream-ordered ahead of the graph: a copy node at a graph's head held its first kernel ~1.3 ms
    // behind the copy (D12 spec-1.2K-r1-small_m-spec_graph, P11R-20), so the verify graph holds kernels only.
    cere_status status = cuda_status(
        cudaMemcpyAsync(ws.ids32, ws.host_ids32, m * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    if (status == CERE_OK) {
        status = cuda_status(cudaMemcpyAsync(ws.ids64, ws.host_ids64, (graph ? 2 * m : m) * sizeof(int64_t),
                                             cudaMemcpyHostToDevice, stream));
    }
    const auto enqueue = [&]() -> cere_status {
        return run_verify(w, ws, flat, uint32_t(m), taps, graph ? ws.ids64 + m : nullptr, stream);
    };
    if (status == CERE_OK && graph) {
        uint64_t key[1 + 4 * kMaxRows];
        key[0] = ws.verify_batched ? 1 : 0;
        for (uint32_t c = 0; c < row_count; ++c) {
            const SessionState &s = *rows[c].session;
            key[1 + 3 * c] = uint64_t(reinterpret_cast<uintptr_t>(s.slot));
            key[2 + 3 * c] = s.max_context_tokens;
            key[3 + 3 * c] = rows[c].n;
        }
        // DraftAccept: the undo range each chain's captured copies write (a session's own snapshots or, in a call
        // group with a deep chain, a range of the model's pool).
        for (uint32_t c = 0; c < row_count; ++c) {
            key[1 + 3 * size_t(row_count) + c] = uint64_t(reinterpret_cast<uintptr_t>(rows[c].undo));
        }
        status = graph_step(ws.graphs, ws.graphs.verify[m], ws.graphs.verify_counts, "spec_graph", uint32_t(m), key,
                            1 + 4 * size_t(row_count), stream, enqueue);
    } else if (status == CERE_OK) {
        status = with_research(ws, research_call, stream, enqueue);
    }
    if (status == CERE_OK) {
        status = cuda_status(
            cudaMemcpyAsync(ws.host_top2, ws.top2, m * sizeof(cere_token_score), cudaMemcpyDeviceToHost, stream));
    }
    // verify_chains: the attention_chain calls of this call (captured or not), one per batched chain and
    // full-attention layer (with attention_chains, R16-C, the chains of two or more rows it batches likewise).
    if (status == CERE_OK) {
        uint64_t full_layers = 0;
        for (uint32_t L = 0; L < kLayers; ++L) full_layers += w.layers[L].full_attention ? 1 : 0;
        for (uint32_t r = 0; r < m; r += flat[r].chain->n + 1) {
            if (chain_batched(ws, flat[r], flat[r].chain->n + 1)) ws.verify_chains += full_layers;
        }
    }
    // Drain the stream even on failure so no enqueued work outlives the call (the bound fault word as forward).
    const cudaError_t sync = drain(stream, ws.moe_fork.side);
    const cere_status bound = take_bound_fault(ws);
    if (status != CERE_OK) return status;
    CERE_TRY_CUDA(sync);
    CERE_TRY(bound);
    std::memcpy(out, ws.host_top2, m * sizeof(cere_token_score));
    for (uint32_t c = 0, r = 0; c < row_count; r += rows[c].n + 1, ++c) {
        if (!rows[c].logprobs) continue;
        for (uint32_t q = r; q <= r + rows[c].n; ++q) {
            out_logprobs[q] = staged_logprobs(ws.host_logprobs[q], out[q].id, rows[c].sampler->top_logprobs);
        }
    }
    return CERE_OK;
}

cere_status verify_commit(const cere_qwen4_exp_config &cfg, const VerifyRow &row, uint32_t a, cudaStream_t stream,
                          const uint32_t *committed) {
    const bool counted = row.sampler != nullptr && row.sampler->sampled;
    if (row.session == nullptr || a == 0 || a > row.n + 1 || (a <= row.n && row.undo == nullptr) ||
        (counted && committed == nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    SessionState &s = *row.session;
    if (s.position >= s.max_context_tokens || a > s.max_context_tokens - s.position) return CERE_ERR_ARGUMENT;
    if (a <= row.n) {
        // The state after rows 0..a-1: undo[a - 1], taken right before row a's first mutation of each piece. Every
        // piece in one k1::copy_segments launch (m23 design note DG-03), the bytes of a cudaMemcpyAsync D2D per
        // piece: at most 2 per layer (GDN conv and SSM, or the QSA ring and compressed key) and the PLE history and
        // conv state.
        const UndoState &u = row.undo[a - 1];
        constexpr uint32_t kRestorePieces = 2 * kLayers + 2;
        static_assert(kRestorePieces <= uint32_t(k1::kMaxCopySegments), "verify_commit restores in one launch");
        k1::CopySegment pieces[kRestorePieces];
        uint32_t n = 0;
        for (uint32_t L = 0; L < kLayers; ++L) {
            if (layout::full_attention(cfg, L)) {
                pieces[n++] = flat(s.qsa_ring[L], u.qsa_ring[L], kQsaRingBytes);
                if (u.key_row >= 0) {
                    pieces[n++] = flat(s.qsa_keys[L] + uint64_t(u.key_row) * k9::kIndexHeadDim, u.qsa_key[L],
                                       kQsaKeyBytes);
                }
            } else {
                pieces[n++] = flat(s.gdn[L].conv, u.gdn[L].conv, kGdnConvBytes);
                pieces[n++] = flat(s.gdn[L].ssm, u.gdn[L].ssm, kGdnSsmBytes);
            }
        }
        pieces[n++] = flat(s.ple.history, u.ple.history, kPleHistoryBytes);
        pieces[n++] = flat(s.ple.conv_state, u.ple.conv_state, kPleConvBytes);
        CERE_TRY_CUDA(k1::copy_segments(pieces, int(n), stream));
        // DraftAccept / Chains: a chain of more than 4 drafts may complete further groups q1 + 4, q1 + 8, ... after the
        // first one q1 (>= p + a) that undo[a - 1] holds; undo[q - p] (taken before row q - p + 1) holds the key of
        // group q + 4, so the groups restore in order while the chain reaches them.
        if (u.key_row >= 0) {
            constexpr uint64_t kRatio = uint64_t(k9::kRatio);
            const uint64_t q1 = s.position + a + (kRatio - 1 - (s.position + a) % kRatio) % kRatio;
            for (uint64_t next = q1 - s.position; next < row.n && row.undo[next].key_row >= 0; next += kRatio) {
                const UndoState &v = row.undo[next];
                n = 0;
                for (uint32_t L = 0; L < kLayers; ++L) {
                    if (!layout::full_attention(cfg, L)) continue;
                    pieces[n++] = flat(s.qsa_keys[L] + uint64_t(v.key_row) * k9::kIndexHeadDim, v.qsa_key[L],
                                       kQsaKeyBytes);
                }
                CERE_TRY_CUDA(k1::copy_segments(pieces, int(n), stream));
            }
        }
    }
    // D5 (M29S2-14): only the committed tokens of a sampled item count, as plain sampled decode counts each.
    if (counted) CERE_TRY_CUDA(k12::count_commit(s.out_counts, committed, int64_t(a), stream));
    s.position += a;
    return CERE_OK;
}

// K6's split-K row bound (kSplitMaxTokens, k6_attention.h: 16) is the graphs' row bound.
static_assert(layout::kDraftGraphRows == uint64_t(kernels::GemmPlan::kMaxRows),
              "spec_draft_graph_fits: the draft graphs' row bound");

cere_status draft_graph(const ModelWeights &w, Workspace &ws, SessionState &s, uint64_t p0, uint32_t rows, bf16 *hc,
                        const uint32_t *next, uint32_t *proposals, cudaStream_t stream) {
    if (!ws.graphs.spec || hc == nullptr || next == nullptr || proposals == nullptr || s.rope_delta != 0 ||
        !layout::spec_draft_graph_fits(p0, rows, s.max_context_tokens, ws.max_tokens)) {
        return CERE_ERR_ARGUMENT;
    }
    constexpr uint32_t kSteps = uint32_t(layout::kDraftSteps);
    const uint32_t staged = rows + kSteps;
    uint64_t positions[layout::kDraftGraphRows + layout::kDraftSteps];
    for (uint32_t i = 0; i < staged; ++i) {
        positions[i] = p0 + i;
        ws.host_ids64[i] = int64_t(p0 + i);
    }
    // Stream-ordered ahead of the graph, as forward_verify_rows' ids (no copy node at a graph's head).
    CERE_TRY_CUDA(
        cudaMemcpyAsync(ws.ids64, ws.host_ids64, staged * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    const auto enqueue = [&]() -> cere_status {
        mtp::MtpCall call;
        call.m = rows;
        call.positions = positions;
        call.device_positions = ws.ids64;
        call.next_ids = next;
        call.hc_in = hc;
        call.hc_out = hc;
        call.draft_ids = proposals + 1;
        call.last_only = true;
        CERE_TRY(mtp::forward(w, ws, s, call, stream));
        bf16 *row = hc + uint64_t(rows - 1) * kHcHidden;
        for (uint32_t k = 1; k <= kSteps; ++k) {
            CERE_TRY(graph_split(ws, stream));  // one segment per MTP forward (model.h DecodeGraph)
            call.m = 1;
            call.positions = positions + rows + k - 1;
            call.device_positions = ws.ids64 + rows + k - 1;
            call.next_ids = proposals + k;
            call.hc_in = row;
            call.hc_out = row;
            call.draft_ids = proposals + k + 1;
            CERE_TRY(mtp::forward(w, ws, s, call, stream));
        }
        return CERE_OK;
    };
    const uint64_t key[] = {uint64_t(reinterpret_cast<uintptr_t>(s.slot)), s.max_context_tokens,
                            uint64_t(reinterpret_cast<uintptr_t>(hc)), uint64_t(reinterpret_cast<uintptr_t>(next)),
                            uint64_t(reinterpret_cast<uintptr_t>(proposals))};
    return graph_step(ws.graphs, ws.graphs.draft[rows], ws.graphs.draft_counts, "spec_draft_graph", rows, key,
                      sizeof(key) / sizeof(key[0]), stream, enqueue);
}

}  // namespace cere::model
