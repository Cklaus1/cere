// Inkling 2 Mini S5: weights.h.
#include "model/inkling/weights.h"

#include <initializer_list>
#include <map>
#include <set>

#include "cere_executor.h"

namespace cere::inkling {
namespace {

constexpr int kLoopLayers = plan::kLoopLen;

class Binder {
public:
    Binder(const std::vector<TensorRef> &tensors, std::string *error) : error_(error) {
        for (const TensorRef &t : tensors) {
            if (t.name.rfind("model.llm.", 0) != 0) continue;
            if (!by_name_.emplace(t.name, &t).second) fail(t.name, "duplicate");
        }
    }

    // The tensor `name` with `dtype` and exactly `shape`, or null (and the first error recorded).
    const void *take(const std::string &name, uint32_t dtype, std::initializer_list<uint64_t> shape) {
        auto it = by_name_.find(name);
        if (it == by_name_.end()) {
            fail(name, "missing");
            return nullptr;
        }
        const TensorRef &t = *it->second;
        bool same = t.dtype == dtype && t.ndim == shape.size();
        size_t d = 0;
        for (uint64_t s : shape) same = same && t.shape[d++] == s;
        if (!same) {
            fail(name, "dtype or shape differs from the forward's");
            return nullptr;
        }
        if (t.ptr == nullptr) {
            fail(name, "null pointer");
            return nullptr;
        }
        used_.insert(name);
        return t.ptr;
    }
    const bf16 *bf(const std::string &name, std::initializer_list<uint64_t> shape) {
        return static_cast<const bf16 *>(take(name, CERE_DTYPE_BF16, shape));
    }
    const float *f32(const std::string &name, std::initializer_list<uint64_t> shape) {
        return static_cast<const float *>(take(name, CERE_DTYPE_F32, shape));
    }
    const uint8_t *u8(const std::string &name, uint32_t dtype, std::initializer_list<uint64_t> shape) {
        return static_cast<const uint8_t *>(take(name, dtype, shape));
    }

    bool finish() {
        for (const auto &kv : by_name_) {
            if (used_.count(kv.first) == 0) fail(kv.first, "not read by the forward");
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

void bind_layer(Binder &b, int l, LayerWeights *w) {
    const std::string p = "model.llm.layers." + std::to_string(l) + ".";
    const std::string a = p + "attn.";
    const uint64_t H = kHidden;
    w->full = plan::is_full_layer(l);
    if (w->full) {
        w->in_norm[0] = b.bf(p + "q_input_norm.weight", {H});
        w->in_norm[1] = b.bf(p + "k_input_norm.weight", {H});
        w->in_norm[2] = b.bf(p + "v_input_norm.weight", {H});
        w->k_static = b.bf(a + "k_sconv.static_w", {kConvTaps, kKvRow});
        w->v_static = b.bf(a + "v_sconv.static_w", {kConvTaps, kKvRow});
        w->k_proj = b.bf(a + "k_sconv.weight_proj", {kDynRow, H});
        w->v_proj = b.bf(a + "v_sconv.weight_proj", {kDynRow, H});
    } else {
        w->in_norm[0] = w->in_norm[1] = w->in_norm[2] = b.bf(p + "attn_norm.weight", {H});
        w->k_conv = b.bf(a + "k_sconv.weight", {kKvRow, 1, kConvTaps});
        w->v_conv = b.bf(a + "v_sconv.weight", {kKvRow, 1, kConvTaps});
    }
    w->wq = b.bf(a + "wq_du.weight", {kQRow, H});
    w->wr = b.bf(a + "wr_du.weight", {kRelRow, H});
    w->wsink = b.bf(a + "attn_sink_proj.weight", {kSinkRow, H});
    w->wgate = b.bf(a + "gate_attn.weight", {kQRow, H});
    w->wk = b.bf(a + "wk_dv.weight", {kKvRow, H});
    w->wv = b.bf(a + "wv_dv.weight", {kKvRow, H});
    w->wo = b.bf(a + "wo_ud.weight", {H, kQRow});
    w->q_scale = b.bf(a + "q_l2_clip.scale", {24, 128});
    w->rel_scale = b.bf(a + "rel_l2_clip.scale", {24, 16});
    w->rel_proj = b.bf(a + "rel_logits_proj.proj", {16, kKvRow});
    w->attn_conv = b.bf(p + "attn_sconv.weight", {H, 1, kConvTaps});
    w->mlp_norm = b.bf(p + "mlp_norm.weight", {H});
    w->mlp_conv = b.bf(p + "mlp_sconv.weight", {H, 1, kConvTaps});
    w->router_w = b.bf(p + "mlp.gate.weight", {kMoeExperts, H});
    w->router_b = b.f32(p + "mlp.gate.bias", {kMoeExperts});

    // Experts as stored (moe.h MoeWeights): NVFP4 layers 0-19, MXFP8 layers 20-23 (hf_quant_config.json).
    const std::string e = p + "mlp.experts.";
    const uint64_t E = kMoeExperts, I2 = 2 * kMoeInter, I = kMoeInter;
    MoeWeights &m = w->moe;
    m.nvfp4 = l < 20;
    m.num_experts = kMoeExperts;
    if (m.nvfp4) {
        m.w13 = b.u8(e + "w13_weight", CERE_DTYPE_U8, {E, I2, H / 2});
        m.w13_scale = b.u8(e + "w13_weight_scale", CERE_DTYPE_F8_E4M3, {E, I2, H / 16});
        m.w13_amax = b.f32(e + "w13_weight_amax", {E});
        m.w13_den = b.f32(e + "w13_weight_outer_scale_denominator", {1});
        m.w2 = b.u8(e + "w2_weight", CERE_DTYPE_U8, {E, H, I / 2});
        m.w2_scale = b.u8(e + "w2_weight_scale", CERE_DTYPE_F8_E4M3, {E, H, I / 16});
        m.w2_amax = b.f32(e + "w2_weight_amax", {E});
        m.w2_den = b.f32(e + "w2_weight_outer_scale_denominator", {1});
    } else {
        m.w13 = b.u8(e + "w13_weight", CERE_DTYPE_F8_E4M3, {E, I2, H});
        m.w13_scale = b.u8(e + "w13_weight_scale", CERE_DTYPE_U8, {E, I2, H / 32});
        m.w2 = b.u8(e + "w2_weight", CERE_DTYPE_F8_E4M3, {E, H, I});
        m.w2_scale = b.u8(e + "w2_weight_scale", CERE_DTYPE_U8, {E, H, I / 32});
    }

    // attn_res.attn_queries.i is layer i + 1's attention_input query, mlp_queries.i layer i's mlp_input query
    // (REF conversion_mapping.py :407-415, Inkling2IndexOffsetWeightRename offset 1).
    if (l > 0) {
        const std::string q = "model.llm.attn_res.attn_queries." + std::to_string(l - 1);
        w->attn_query = w->full ? b.f32(q, {3, H}) : b.f32(q, {H});
    }
    w->mlp_query = b.f32("model.llm.attn_res.mlp_queries." + std::to_string(l), {H});

    if (plan::is_loop_layer(l)) {
        static const char *const kGates[4] = {"attn", "k", "v", "mlp"};
        for (int g = 0; g < 4; ++g) {
            const std::string c = p + "controllers." + kGates[g] + ".";
            w->gates[g].down = b.bf(c + "down_weight", {kGateRank, 2 * H});
            w->gates[g].forget = b.bf(c + "forget_weight", {H, kGateRank});
            w->gates[g].gain = b.bf(c + "gain_weight", {H, kGateRank});
        }
    }
}

}  // namespace

bool bind_weights(const std::vector<TensorRef> &tensors, Weights *out, std::string *error) {
    Binder b(tensors, error);
    Weights w;
    const uint64_t H = kHidden;
    w.embed = b.bf("model.llm.embed.weight", {kVocab, H});
    w.embed_norm = b.bf("model.llm.embed_norm.weight", {H});
    w.final_norm = b.bf("model.llm.norm.weight", {H});
    w.unembed = b.bf("model.llm.unembed.weight", {kVocab, H});
    w.output_query = b.f32("model.llm.attn_res.output_query", {H});
    w.cross_pass_queries = b.f32("model.llm.middle_cycle_cross_pass_queries", {2 * kLoopLayers, H});
    w.cross_pass_kv_queries = b.f32("model.llm.middle_cycle_cross_pass_kv_queries", {2 * kLoopLayers, H});
    for (int l = 0; l < plan::kLayers; ++l) bind_layer(b, l, &w.layers[l]);
    if (!b.finish()) return false;
    *out = w;
    return true;
}

}  // namespace cere::inkling
