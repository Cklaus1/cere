// Inkling 2 Mini (inkling design note 3, slice S5): the text model's checkpoint tensors bound by name, as stored
// (no repack; the experts in S4's layout, kernels/inkling/moe.h MoeWeights). The device binds them on its arena
// (device/inkling/), the forward ktest on the caller's torch tensors (ktest/inkling/forward_ktest.cpp). Vision
// (`model.visual.*`, S7) and the drafter (mtp.safetensors, S9) are not bound here (D-I5).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_bf16.h>

#include "kernels/inkling/moe.h"
#include "model/inkling/plan.h"

namespace cere::inkling {

constexpr int kHidden = 2304;
constexpr int kQRow = 3072;      // 24 heads x 128
constexpr int kKvRow = 1024;     // 8 KV heads x 128
constexpr int kRelRow = 384;     // 24 heads x d_rel 16
constexpr int kSinkRow = 24;     // one sink logit per query head
constexpr int kDynRow = 32;      // dynamic conv taps: 8 KV heads x 4
constexpr int kConvTaps = 4;
constexpr int kGateRank = 64;    // loop message gate rank
constexpr int kVocab = 201024;   // embed / unembed rows (padded)
constexpr int kUnpaddedVocab = 200058;  // the logits the model produces (REF Inkling2UnpadVocab)
constexpr int kLogitLd = 200064;        // logits row stride (elements): kUnpaddedVocab rounded up to 8
constexpr float kRmsEps = 1e-6f;

// One tensor as the caller holds it (device memory).
struct TensorRef {
    std::string name;
    uint32_t dtype = 0;  // CERE_DTYPE_*
    uint32_t ndim = 0;
    uint64_t shape[4] = {0, 0, 0, 0};
    const void *ptr = nullptr;
};

// The message gate of one input of a loop layer (controllers.{attn,k,v,mlp}).
struct GateWeights {
    const bf16 *down = nullptr;    // [64, 4608]
    const bf16 *forget = nullptr;  // [2304, 64]
    const bf16 *gain = nullptr;    // [2304, 64]
};

struct LayerWeights {
    bool full = false;
    // Sliding: attn_norm (input_layernorm). Full: q/k/v_input_norm (index 0, 1, 2).
    const bf16 *in_norm[3] = {nullptr, nullptr, nullptr};
    const bf16 *wq = nullptr;     // [3072, 2304]
    const bf16 *wr = nullptr;     // [384, 2304]
    const bf16 *wsink = nullptr;  // [24, 2304]
    const bf16 *wgate = nullptr;  // [3072, 2304]
    const bf16 *wk = nullptr;     // [1024, 2304]
    const bf16 *wv = nullptr;     // [1024, 2304]
    const bf16 *wo = nullptr;     // [2304, 3072]
    // Sliding: the static K/V conv [1024, 1, 4]. Full: static_w [4, 1024] and weight_proj [32, 2304].
    const bf16 *k_conv = nullptr;
    const bf16 *v_conv = nullptr;
    const bf16 *k_static = nullptr;
    const bf16 *v_static = nullptr;
    const bf16 *k_proj = nullptr;
    const bf16 *v_proj = nullptr;
    const bf16 *q_scale = nullptr;    // [24, 128]
    const bf16 *rel_scale = nullptr;  // [24, 16]
    const bf16 *rel_proj = nullptr;   // [16, 1024]
    const bf16 *attn_conv = nullptr;  // attn_sconv [2304, 1, 4]
    const bf16 *mlp_norm = nullptr;   // post_attention_layernorm [2304]
    const bf16 *mlp_conv = nullptr;   // mlp_sconv [2304, 1, 4]
    const bf16 *router_w = nullptr;   // mlp.gate.weight [128, 2304]
    const float *router_b = nullptr;  // mlp.gate.bias [128] (e_score_correction_bias)
    MoeWeights moe;
    // attention_input query: [1, 2304] (sliding) or [3, 2304] (full; q, k, v); null at layer 0.
    const float *attn_query = nullptr;
    const float *mlp_query = nullptr;  // [2304]
    GateWeights gates[4];              // q (controllers.attn), k, v, mlp; loop layers only
};

struct Weights {
    const bf16 *embed = nullptr;       // [201024, 2304]
    const bf16 *embed_norm = nullptr;  // [2304]
    const bf16 *final_norm = nullptr;  // model.llm.norm [2304]
    const bf16 *unembed = nullptr;     // [201024, 2304]; rows < 200058 produce logits
    const float *output_query = nullptr;          // attn_res.output_query [2304]
    const float *cross_pass_queries = nullptr;    // middle_cycle_cross_pass_queries [24, 2304]
    const float *cross_pass_kv_queries = nullptr; // middle_cycle_cross_pass_kv_queries [24, 2304]
    LayerWeights layers[plan::kLayers];
};

// Binds every `model.llm.*` tensor of `tensors` by its checkpoint name, checking dtype and shape. Fails (returns
// false, *error names the tensor and the reason) on a missing tensor, a wrong dtype or shape, a duplicate, or a
// `model.llm.*` tensor the forward does not read. Tensors outside `model.llm.` (vision, drafter) are ignored here.
bool bind_weights(const std::vector<TensorRef> &tensors, Weights *out, std::string *error);

// The message queries of loop layer i in the merged order of REF Inkling2LoopMergeMessageQueries
// (core_model_loading.py :823-855): q = cross_pass_queries[2i], k = cross_pass_kv_queries[2i],
// v = cross_pass_kv_queries[2i + 1], mlp = cross_pass_queries[2i + 1]. Row j of the 4 (q, k, v, mlp) of layer i.
inline const float *message_query_row(const Weights &w, int i, int j) {
    switch (j) {
        case 0: return w.cross_pass_queries + size_t(2 * i) * kHidden;
        case 1: return w.cross_pass_kv_queries + size_t(2 * i) * kHidden;
        case 2: return w.cross_pass_kv_queries + size_t(2 * i + 1) * kHidden;
        default: return w.cross_pass_queries + size_t(2 * i + 1) * kHidden;
    }
}

}  // namespace cere::inkling
