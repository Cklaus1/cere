// K8 model interface (model.h): arena order, sizing, weight/workspace/session binding and session reset of the A4
// model. instantiate_model and forward live in model/forward.cu.
#include "model/model.h"

#include "host/kv_pool.h"
#include "kernels/k4_nvfp4.h"
#include "kernels/k6_attention.h"
#include "kernels/k9_qsa.h"
#include "kernels/k12_sampler.h"
#include "kernels/k13s_spec_accept.h"
#include "model/layout.h"
#include "model/moe_backend.h"
#include "model/mtp.h"
#include "model/mxfp8_plan.h"

#include <algorithm>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cere::model {

namespace {

using layout::Carver;
using layout::full_attention;
using layout::kBf16;
using layout::kF32;
using layout::kI64;
using layout::Region;
using layout::rotary_dim;
constexpr std::string_view kLayerPrefix = "model.language_model.layers.";

// Parses "<digits>" at s[*at...] up to `stop`; false when absent.
bool parse_index(std::string_view s, size_t *at, char stop, uint64_t *out) {
    size_t i = *at;
    uint64_t v = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
        v = v * 10 + uint64_t(s[i] - '0');
        ++i;
    }
    if (i == *at || i - *at > 9 || i >= s.size() || s[i] != stop) return false;
    *out = v;
    *at = i + 1;
    return true;
}

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }

// The run a tensor belongs to (model.h ArenaOrder): key + member rank, or false for a singleton.
#if defined(CERE_MOE_P3B)
// `w13` (P3b, model.h ArenaOrder): the up/gate weight and weight_scale tensors of a layer form one run.
bool run_of(std::string_view name, std::string *key, uint64_t *member, bool w13) {
#else
bool run_of(std::string_view name, std::string *key, uint64_t *member) {
#endif
    // M2.9 (m29 design note D3): the MTP layer's q|k|v pack like the target layers' (one attn_qkv GEMM).
    constexpr std::string_view kMtpLayer = "mtp.layers.0.";
    if (starts_with(name, kMtpLayer)) {
        static constexpr std::string_view kMtpQkv[3] = {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                                                         "self_attn.v_proj.weight"};
        for (uint64_t i = 0; i < 3; ++i) {
            if (name.substr(kMtpLayer.size()) != kMtpQkv[i]) continue;
            *key = std::string(kMtpLayer) + "self_attn.qkv";
            *member = i;
            return true;
        }
        return false;
    }
    if (!starts_with(name, kLayerPrefix)) return false;
    size_t at = kLayerPrefix.size();
    uint64_t layer = 0;
    if (!parse_index(name, &at, '.', &layer)) return false;
    const std::string_view rest = name.substr(at);
    const std::string_view head = name.substr(0, at);  // "model.language_model.layers.L."
    constexpr std::string_view kExperts = "mlp.experts.";
    if (starts_with(rest, kExperts)) {
        size_t e = kExperts.size();
        uint64_t expert = 0;
        if (!parse_index(rest, &e, '.', &expert)) return false;
#if defined(CERE_MOE_P3B)
        if (w13) {
            static constexpr std::string_view kW13[2] = {"up_proj.", "gate_proj."};
            const std::string_view field = rest.substr(e);
            for (uint64_t g = 0; g < 2; ++g) {
                if (!starts_with(field, kW13[g])) continue;
                const std::string_view suffix = field.substr(kW13[g].size());
                if (suffix != "weight" && suffix != "weight_scale") break;
                *key = std::string(head) + "mlp.experts.*.w13." + std::string(suffix);
                *member = 2 * expert + g;
                return true;
            }
        }
#endif
        *key = std::string(head) + "mlp.experts.*." + std::string(rest.substr(e));
        *member = expert;
        return true;
    }
    static constexpr std::string_view kQkv[3] = {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                                                  "self_attn.v_proj.weight"};
    // Track Fp4S3 (fp4local design note FL-30): an MXFP8 q|k|v's UE8M0 scales, their own run (one [13,312, 80]).
    static constexpr std::string_view kQkvScale[3] = {"self_attn.q_proj.weight_scale", "self_attn.k_proj.weight_scale",
                                                       "self_attn.v_proj.weight_scale"};
    for (uint64_t i = 0; i < 3; ++i) {
        if (rest == kQkv[i] || rest == kQkvScale[i]) {
            *key = std::string(head) + (rest == kQkv[i] ? "self_attn.qkv" : "self_attn.qkv_scale");
            *member = i;
            return true;
        }
    }
    // The PLE table shards, and (NVFP4 table, fp4local design note D8) their per-shard E4M3 block scales.
    constexpr std::string_view kShard = "ple.ple_embedding.ngram_embedding.shard_";
    if (starts_with(rest, kShard)) {
        size_t s = kShard.size();
        uint64_t shard = 0;
        if (!parse_index(rest, &s, '.', &shard)) return false;
        const std::string_view field = rest.substr(s);
        if (field == "weight") {
            *key = std::string(head) + "ple.shards";
        } else if (field == "weight_scale") {
            *key = std::string(head) + "ple.shard_scales";
        } else {
            return false;
        }
        *member = shard;
        return true;
    }
    return false;
}

// ---- weight binding ----

class Binder {
public:
    Binder(const ArenaView &arena, std::string *error) : a_(arena), error_(error) {
        const auto &tensors = a_.plan->tensors;
        index_.reserve(tensors.size());
        for (uint64_t i = 0; i < tensors.size(); ++i) index_.emplace(tensors[i].name, i);
    }

    bool ok() const { return ok_; }

    // The storage of `name` after checking dtype, shape and placement; null (and the first error recorded) otherwise.
    template <typename T>
    T *get(const std::string &name, uint32_t dtype, std::initializer_list<uint64_t> shape,
           uint32_t placement = CERE_PLACEMENT_DEVICE, uint64_t *index_out = nullptr) {
        return get<T>(name, dtype, shape.begin(), shape.size(), placement, index_out);
    }

    // The same with the shape as `ndim` dims at `shape`.
    template <typename T>
    T *get(const std::string &name, uint32_t dtype, const uint64_t *shape, size_t ndim,
           uint32_t placement = CERE_PLACEMENT_DEVICE, uint64_t *index_out = nullptr) {
        const auto it = index_.find(name);
        if (it == index_.end()) return fail<T>("missing tensor " + name);
        const host::TensorPlan &t = a_.plan->tensors[it->second];
        if (t.dtype != dtype) return fail<T>("tensor " + name + ": dtype " + std::to_string(t.dtype) + ", expected " +
                                             std::to_string(dtype));
        if (t.placement != placement) {
            return fail<T>("tensor " + name + ": placement " + std::to_string(t.placement) + ", expected " +
                           std::to_string(placement));
        }
        bool shape_ok = t.ndim == ndim;
        for (size_t d = 0; shape_ok && d < ndim; ++d) shape_ok = t.shape[d] == shape[d];
        if (!shape_ok) {
            std::string have = "[";
            for (uint32_t i = 0; i < t.ndim; ++i) have += (i ? ", " : "") + std::to_string(t.shape[i]);
            std::string want = "[";
            for (size_t d = 0; d < ndim; ++d) want += (d ? ", " : "") + std::to_string(shape[d]);
            return fail<T>("tensor " + name + ": shape " + have + "], expected " + want + "]");
        }
        if (index_out != nullptr) *index_out = it->second;
        uint8_t *base = placement == CERE_PLACEMENT_HOST_PINNED ? a_.host_device : a_.device;
        return reinterpret_cast<T *>(base + a_.offsets[it->second]);
    }

    // `count` tensors name_of(i) of equal shape that must abut in index order (an arena_order run); returns the first.
    template <typename T, typename Name>
    T *run(uint64_t count, Name name_of, uint32_t dtype, std::initializer_list<uint64_t> shape,
           uint32_t placement = CERE_PLACEMENT_DEVICE) {
        T *first = nullptr;
        const uint8_t *expect = nullptr;
        for (uint64_t i = 0; i < count; ++i) {
            uint64_t index = 0;
            const std::string name = name_of(i);
            T *p = get<T>(name, dtype, shape, placement, &index);
            if (p == nullptr) return nullptr;
            if (i == 0) first = p;
            if (expect != nullptr && reinterpret_cast<const uint8_t *>(p) != expect) {
                return fail<T>("tensor " + name + " does not follow its predecessor in the arena (arena_order run)");
            }
            expect = reinterpret_cast<const uint8_t *>(p) + a_.plan->tensors[index].bytes;
        }
        return first;
    }

    uint64_t rows_of(const std::string &name) const {
        const auto it = index_.find(name);
        return it == index_.end() ? 0 : a_.plan->tensors[it->second].shape[0];
    }

    // The dtype of `name`, or UINT32_MAX when the plan has no such tensor.
    uint32_t dtype_of(const std::string &name) const {
        const auto it = index_.find(name);
        return it == index_.end() ? UINT32_MAX : a_.plan->tensors[it->second].dtype;
    }

    // Track Fp4S3 (fp4local design note FL-30): the GEMM weight `prefix`.weight [n, k], BF16 or MXFP8 (F8_E4M3
    // with U8 `prefix`.weight_scale [n, k / 32], into *scales; null for BF16). mxfp8::plan_of checked the format and
    // its group before, so the weight's dtype decides; the MXFP8 pointer is the E4M3 data, which the plans look up in
    // ModelWeights::mxfp8.
    const bf16 *dense(const std::string &prefix, uint64_t n, uint64_t k, const uint8_t **scales) {
        *scales = nullptr;
        const std::string weight = prefix + ".weight";
        if (dtype_of(weight) != CERE_DTYPE_F8_E4M3) return get<const bf16>(weight, CERE_DTYPE_BF16, {n, k});
        const bf16 *w = get<const bf16>(weight, CERE_DTYPE_F8_E4M3, {n, k});
        *scales = get<const uint8_t>(prefix + ".weight_scale", CERE_DTYPE_U8, {n, k / mxfp8::kGroupK});
        return w;
    }

private:
    template <typename T>
    T *fail(std::string message) {
        if (ok_ && error_ != nullptr) *error_ = std::move(message);
        ok_ = false;
        return nullptr;
    }

    const ArenaView &a_;
    std::string *error_;
    std::unordered_map<std::string_view, uint64_t> index_;
    bool ok_ = true;
};

std::string layer_name(uint32_t layer, const char *rest) {
    return std::string(kLayerPrefix) + std::to_string(layer) + "." + rest;
}

HcWeights bind_hc(Binder &b, const std::string &prefix, const cere_qwen4_exp_config &cfg, bool inject) {
    const uint64_t width = uint64_t(cfg.hc_count) * cfg.hidden_size;
    HcWeights w;
    w.norm = b.get<const bf16>(prefix + "hc_norm.weight", CERE_DTYPE_BF16, {width});
    w.down = b.get<const bf16>(prefix + "input_mix_weight_down.weight", CERE_DTYPE_BF16, {cfg.hc_lowrank, width});
    w.up = b.get<const bf16>(prefix + "input_mix_weight_up.weight", CERE_DTYPE_BF16, {width, cfg.hc_lowrank});
    if (inject) {
        w.inject = b.get<const bf16>(prefix + "block_inject_weight.weight", CERE_DTYPE_BF16, {cfg.hc_count, width});
    }
    return w;
}

// M2.9 (m29 design note D3): the 31 `mtp.*` tensors of a plan with drafts (model.h MtpWeights), all BF16; on the S3
// artifact (track Fp4S3) the 35 with the routed experts NVFP4 W4A16 (gate_up / down plus scales and globals).
cere_status bind_mtp(Binder &b, const cere_qwen4_exp_config &cfg, MtpWeights *out, std::string *error) {
    const uint64_t H = cfg.hidden_size;
    const uint64_t E = cfg.num_experts;
    const uint64_t I = cfg.moe_intermediate_size;
    const uint64_t S = cfg.shared_expert_intermediate_size;
    const uint64_t hc_width = uint64_t(cfg.hc_count) * H;
    const std::string root = "mtp.";
    const std::string layer = "mtp.layers.0.";
    const auto bf = [&b](const std::string &name, std::initializer_list<uint64_t> shape) {
        return b.get<const bf16>(name, CERE_DTYPE_BF16, shape);
    };
    MtpWeights m;
    m.fc_embedding = bf(root + "fc_embedding.weight", {H, H});
    m.fc_hidden = bf(root + "fc_hidden.weight", {H, H});
    m.norm_embedding = bf(root + "pre_fc_norm_embedding.weight", {H});
    m.norm_hidden = bf(root + "pre_fc_norm_hidden.weight", {hc_width});
    m.mixer = bind_hc(b, root + "hyper_connection_mixer.", cfg, false);
    m.attn_hc = bind_hc(b, layer + "attn_hyper_connection.", cfg, true);
    m.mlp_hc = bind_hc(b, layer + "mlp_hyper_connection.", cfg, true);
    const uint64_t q_rows = uint64_t(cfg.num_attention_heads) * 2 * cfg.head_dim;
    const uint64_t kv_rows = uint64_t(cfg.num_key_value_heads) * cfg.head_dim;
    const bf16 *q = bf(layer + "self_attn.q_proj.weight", {q_rows, H});
    const bf16 *k = bf(layer + "self_attn.k_proj.weight", {kv_rows, H});
    const bf16 *v = bf(layer + "self_attn.v_proj.weight", {kv_rows, H});
    if (q != nullptr && k != nullptr && v != nullptr && (k != q + q_rows * H || v != k + kv_rows * H)) {
        if (error != nullptr) *error = "tensors " + layer + "self_attn.{q,k,v}_proj.weight are not packed";
        return CERE_ERR_UNSUPPORTED;
    }
    m.attn.qkv = q;
    m.attn.q_norm = bf(layer + "self_attn.q_norm.weight", {cfg.head_dim});
    m.attn.k_norm = bf(layer + "self_attn.k_norm.weight", {cfg.head_dim});
    m.attn.o_proj = bf(layer + "self_attn.o_proj.weight", {H, uint64_t(cfg.num_attention_heads) * cfg.head_dim});
    const uint64_t idx_dim = cfg.indexer_head_dim;
    const uint64_t idx_rows = (uint64_t(cfg.indexer_n_heads) + cfg.indexer_kv_heads) * idx_dim;
    m.attn.indexer_qk = bf(layer + "self_attn.indexer.index_qk_proj.weight", {idx_rows, H});
    m.attn.indexer_q_norm = bf(layer + "self_attn.indexer.q_layernorm.weight", {idx_dim});
    m.attn.indexer_k_norm = bf(layer + "self_attn.indexer.k_layernorm.weight", {idx_dim});
    m.moe.router = bf(layer + "mlp.gate.weight", {E, H});
    // Track Fp4S3 (fp4local design note F7, FL-30): the S3 artifact's experts are NVFP4 W4A16 (U8 gate_up_proj,
    // kernels/k13_moe_bf16.h); the BF16 artifacts' are BF16. The dtype of gate_up_proj decides.
    const std::string experts = layer + "mlp.experts.";
    if (b.dtype_of(experts + "gate_up_proj") == CERE_DTYPE_U8) {
        m.moe.gate_up_q = b.get<const uint8_t>(experts + "gate_up_proj", CERE_DTYPE_U8, {E, 2 * I, H / 2});
        m.moe.gate_up_scale =
            b.get<const uint8_t>(experts + "gate_up_proj_scale", CERE_DTYPE_F8_E4M3, {E, 2 * I, H / 16});
        m.moe.gate_up_global = b.get<const float>(experts + "gate_up_proj_scale_2", CERE_DTYPE_F32, {E, 2});
        m.moe.down_q = b.get<const uint8_t>(experts + "down_proj", CERE_DTYPE_U8, {E, H, I / 2});
        m.moe.down_scale = b.get<const uint8_t>(experts + "down_proj_scale", CERE_DTYPE_F8_E4M3, {E, H, I / 16});
        m.moe.down_global = b.get<const float>(experts + "down_proj_scale_2", CERE_DTYPE_F32, {E});
    } else {
        m.moe.gate_up = bf(experts + "gate_up_proj", {E, 2 * I, H});
        m.moe.down = bf(experts + "down_proj", {E, H, I});
    }
    m.moe.shared_gate = bf(layer + "mlp.shared_expert.gate_proj.weight", {S, H});
    m.moe.shared_up = bf(layer + "mlp.shared_expert.up_proj.weight", {S, H});
    m.moe.shared_down = bf(layer + "mlp.shared_expert.down_proj.weight", {H, S});
    m.moe.shared_expert_gate = bf(layer + "mlp.shared_expert_gate.weight", {1, H});
    if (!b.ok()) return CERE_ERR_UNSUPPORTED;
    *out = m;
    return CERE_OK;
}

// The derived region (model.h derived_bytes, layout::carve_derived).
void bind_derived(const cere_qwen4_exp_config &cfg, uint64_t max_context_tokens, Carver &c, ModelWeights *w) {
    layout::carve_derived(cfg, max_context_tokens, c, [w](Region r, uint32_t l, uint8_t *p) {
        if (w == nullptr) return;
        float *f = reinterpret_cast<float *>(p);
        switch (r) {
            case Region::kALog: w->layers[l].gdn.A_log = f; break;
            case Region::kG1Alphas: w->layers[l].moe.g1_alphas = f; break;
            case Region::kG2Alphas: w->layers[l].moe.g2_alphas = f; break;
            case Region::kRope: w->rope_cos_sin = f; break;
            default: break;
        }
    });
}

// The full-attention part of the scratch above the dense limit (m22c D2/D5), in forward's carving order: the K6
// prologue outputs, the indexer projection qk [t, 640] BF16, then either dense_attention's workspace or the K9 path
// (the attention output before the gate, then the K9 scratch: index q, selected indices, logits tile).
size_t qsa_attention_scratch_bytes(uint64_t t, uint64_t ctx) {
    const auto align = [](uint64_t b) { return (b + layout::kAlign - 1) / layout::kAlign * layout::kAlign; };
    const size_t qk = size_t(align(t * uint64_t(k9::kQkWidth) * kBf16));
    const size_t k9_path = size_t(align(t * kAttnOutWidth * kBf16)) + k9::scratch_bytes(int64_t(t), int64_t(ctx));
    return k6::prologue_bytes(int64_t(t)) + qk + std::max(k6::dense_attention_workspace_bytes(int64_t(t)), k9_path);
}

// The K12 sampler stage's buffers (model.h SamplerScratch), carved from the start of the shared scratch.
void carve_sampler(const cere_qwen4_exp_config &cfg, Carver &c, SamplerScratch *out) {
    const uint64_t vocab = cfg.vocab_size;
    SamplerScratch s;
    s.probs = c.take<float>(vocab * sizeof(float));
    s.keep = c.take<uint32_t>((vocab + 31) / 32 * sizeof(uint32_t));
    s.lse = c.take<double>(sizeof(double));
    s.scratch = c.take<void>(k12::scratch_bytes());
    s.top2_ids = c.take<uint32_t>(2 * sizeof(uint32_t));
    s.top2_scores = c.take<float>(2 * sizeof(float));
    s.drawn = c.take<uint32_t>(sizeof(uint32_t));
    s.bias = c.take<cere_logit_bias>(vocab * sizeof(cere_logit_bias));
    s.logprobs = c.take<cere_token_logprobs>(sizeof(cere_token_logprobs));
    s.mask = c.take<uint32_t>((vocab + 31) / 32 * sizeof(uint32_t));  // M2.8 (m28 design note D9): 31,040 B
    if (out != nullptr) *out = s;
}

// The batched decode step's lm_head rows (Workspace::rows_logits) after the sampler buffers: BF16 [min(t, 16), vocab].
uint64_t rows_logits_bytes(const cere_qwen4_exp_config &cfg, uint64_t t) {
    return std::min<uint64_t>(t, kernels::GemmPlan::kMaxRows) * cfg.vocab_size * sizeof(bf16);
}

// The sampler buffers then the batched lm_head rows, from the start of the shared scratch.
size_t sampler_rows_bytes(const cere_qwen4_exp_config &cfg, uint64_t t) {
    Carver c{nullptr};
    carve_sampler(cfg, c, nullptr);
    c.take<bf16>(rows_logits_bytes(cfg, t));
    return size_t(c.size());
}

// The shared kernel scratch for forward calls of up to t rows in a context of ctx tokens: the largest K3 mix / K4 /
// K5 / K6 / K7 workspace and the K12 sampler stage's buffers with the batched lm_head rows; the attention part grows
// to qsa_attention_scratch_bytes when ctx exceeds the dense limit.
size_t scratch_bytes(const cere_qwen4_exp_config &cfg, uint64_t t, uint64_t ctx) {
    const kernels::HcShape hc{cfg.hc_count, cfg.hidden_size, cfg.hc_lowrank, float(cfg.rms_norm_eps)};
    const kernels::MoeDims moe{cfg.hidden_size, cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size,
                               cfg.shared_expert_intermediate_size};
    const size_t attention = ctx > uint64_t(k6::kMaxContext) ? qsa_attention_scratch_bytes(t, ctx)
                                                             : k6::full_attention_workspace_bytes(int64_t(t));
    return std::max({kernels::hc_mix_workspace_bytes(hc, uint32_t(t)), k4::workspace_bytes(int64_t(t)),
                     k5::layer_workspace_bytes(int64_t(t)), attention, CERE_MOE_WORKSPACE_BYTES(moe, uint32_t(t)),
                     sampler_rows_bytes(cfg, t)});
}
#if defined(CERE_MOE_P3B)
// scratch_bytes with K7's MoE workspace in place of P3b's, for the P3b arena-size ktest (p3b/p3b_ktest.cpp).
size_t scratch_bytes_k7(const cere_qwen4_exp_config &cfg, uint64_t t, uint64_t ctx) {
    const kernels::HcShape hc{cfg.hc_count, cfg.hidden_size, cfg.hc_lowrank, float(cfg.rms_norm_eps)};
    const kernels::MoeDims moe{cfg.hidden_size, cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size,
                               cfg.shared_expert_intermediate_size};
    const size_t attention = ctx > uint64_t(k6::kMaxContext) ? qsa_attention_scratch_bytes(t, ctx)
                                                             : k6::full_attention_workspace_bytes(int64_t(t));
    return std::max({kernels::hc_mix_workspace_bytes(hc, uint32_t(t)), k4::workspace_bytes(int64_t(t)),
                     k5::layer_workspace_bytes(int64_t(t)), attention, kernels::moe_workspace_bytes(moe, uint32_t(t)),
                     sampler_rows_bytes(cfg, t)});
}
#endif

// The workspace regions (model.h Workspace), in one order for sizing and binding: layout's row buffers, then the
// shared scratch.
void carve_workspace(const cere_qwen4_exp_config &cfg, uint64_t t, uint64_t ctx, Carver &c, Workspace *ws,
                     uint64_t draft_count = 0) {
    Workspace w;
    layout::carve_workspace_rows(cfg, t, c, [&w](Region r, uint32_t, uint8_t *p) {
        switch (r) {
            case Region::kHyper: w.hyper = reinterpret_cast<bf16 *>(p); break;
            case Region::kNormed: w.normed = reinterpret_cast<bf16 *>(p); break;
            case Region::kMixed: w.mixed = reinterpret_cast<bf16 *>(p); break;
            case Region::kBlockOut: w.block_out = reinterpret_cast<bf16 *>(p); break;
            case Region::kQkv: w.qkv = reinterpret_cast<bf16 *>(p); break;
            case Region::kAttnOut: w.attn_out = reinterpret_cast<bf16 *>(p); break;
            case Region::kTapRouted: w.tap_routed = reinterpret_cast<bf16 *>(p); break;
            case Region::kTapShared: w.tap_shared = reinterpret_cast<bf16 *>(p); break;
            case Region::kIds32: w.ids32 = reinterpret_cast<int32_t *>(p); break;
            case Region::kIds64: w.ids64 = reinterpret_cast<int64_t *>(p); break;
            case Region::kLogits: w.logits = reinterpret_cast<bf16 *>(p); break;
            case Region::kProcLogits: w.proc_logits = reinterpret_cast<float *>(p); break;
            case Region::kTop2: w.top2 = reinterpret_cast<cere_token_score *>(p); break;
            default: break;
        }
    });
    const size_t scratch = draft_count == 0 ? scratch_bytes(cfg, t, ctx)
                                            : std::max<size_t>(scratch_bytes(cfg, t, ctx),
                                                               mtp::scratch_bytes(cfg, t, ctx, draft_count));
    w.scratch = c.take<void>(scratch);
    w.scratch_bytes = scratch;
    Carver sampler{static_cast<uint8_t *>(w.scratch)};
    carve_sampler(cfg, sampler, &w.sampler);
    w.rows_logits = sampler.take<bf16>(rows_logits_bytes(cfg, t));
    // The pinned plans' split-K partials, moe_shared_gate_up's and attn_qkv's (forward.cpp instantiate_model selects
    // their pins through the same gemm_pins_for): one region sized by the largest need, since the plans run one at a
    // time on the forward's stream.
    size_t gemm_workspace = 0;
    for (const uint32_t n : {kSharedInter, kQkvWidth}) {
        const kernels::GemmPins *pins = kernels::gemm_pins_for(n, kHidden, kernels::GemmOut::kBF16);
        const size_t bytes = pins == nullptr ? 0 : kernels::gemm_pins_workspace_bytes(*pins, uint32_t(t));
        gemm_workspace = bytes > gemm_workspace ? bytes : gemm_workspace;
    }
    w.gemm_workspace = c.take<void>(gemm_workspace);
    w.gemm_workspace_bytes = gemm_workspace;
    // M2.9 (M29S2-12): the MTP forward's hc rows, only for a plan with drafts (else the region is unchanged).
    if (draft_count != 0) w.hc_rows = c.take<bf16>(mtp::hc_rows_bytes(cfg, t));
    if (ws != nullptr) {
        ws->max_tokens = uint32_t(t);
        ws->hyper = w.hyper;
        ws->normed = w.normed;
        ws->mixed = w.mixed;
        ws->block_out = w.block_out;
        ws->qkv = w.qkv;
        ws->attn_out = w.attn_out;
        ws->tap_routed = w.tap_routed;
        ws->tap_shared = w.tap_shared;
        ws->ids32 = w.ids32;
        ws->ids64 = w.ids64;
        ws->logits = w.logits;
        ws->proc_logits = w.proc_logits;
        ws->top2 = w.top2;
        ws->scratch = w.scratch;
        ws->scratch_bytes = w.scratch_bytes;
        ws->sampler = w.sampler;
        ws->rows_logits = w.rows_logits;
        ws->gemm_workspace = w.gemm_workspace;
        ws->gemm_workspace_bytes = w.gemm_workspace_bytes;
        ws->hc_rows = w.hc_rows;
        ws->max_context_tokens = ctx;
    }
}

void carve_staging(uint64_t t, Carver &c, Workspace *ws) {
    int32_t *ids32 = c.take<int32_t>(t * sizeof(int32_t));
    int64_t *ids64 = c.take<int64_t>(t * kI64);
    cere_token_score *top2 = c.take<cere_token_score>(t * sizeof(cere_token_score));
    cere_token_logprobs *logprobs =
        c.take<cere_token_logprobs>(staging_logprobs_rows(t) * sizeof(cere_token_logprobs));
    uint32_t *bound_fault = c.take<uint32_t>(sizeof(uint32_t));
    if (ws != nullptr) {
        ws->host_ids32 = ids32;
        ws->host_ids64 = ids64;
        ws->host_top2 = top2;
        ws->host_logprobs = logprobs;
        *bound_fault = 0;  // host memory (mapped pinned staging): no call has run yet
        ws->bound_fault = bound_fault;
    }
}

// One session slot (model.h session_bytes, layout::carve_session).
void bind_slot(const cere_qwen4_exp_config &cfg, uint64_t ctx, Carver &c, SessionState *s, bool draft = false) {
    const auto bind = [s](Region r, uint32_t l, uint8_t *p) {
        if (s == nullptr) return;
        switch (r) {
            case Region::kSlotMap: s->slot_map = reinterpret_cast<int64_t *>(p); break;
            case Region::kQsaRing: s->qsa_ring[l] = reinterpret_cast<bf16 *>(p); break;
            case Region::kGdnConv: s->gdn[l].conv = reinterpret_cast<bf16 *>(p); break;
            case Region::kGdnSsm: s->gdn[l].ssm = reinterpret_cast<bf16 *>(p); break;
            case Region::kPleHistory: s->ple.history = reinterpret_cast<int64_t *>(p); break;
            case Region::kPleConv: s->ple.conv_state = reinterpret_cast<bf16 *>(p); break;
            case Region::kOutCounts: s->out_counts = reinterpret_cast<uint32_t *>(p); break;
            case Region::kMtpQsaRing: s->mtp_qsa_ring = reinterpret_cast<bf16 *>(p); break;
            case Region::kProposals: s->proposals = reinterpret_cast<uint32_t *>(p); break;
            default: break;
        }
    };
    layout::carve_session(cfg, ctx, c, bind, draft);
}

// The paged KV pool (model.h pool_bytes, layout::carve_pool over the full-attention layers).
void bind_kv_pool(const cere_qwen4_exp_config &cfg, uint64_t tokens, Carver &c, KvPool *pool, bool draft = false) {
    const auto bind = [pool](Region r, uint32_t l, uint8_t *p) {
        if (pool == nullptr) return;
        switch (r) {
            case Region::kKCache: pool->k_cache[l] = p; break;
            case Region::kVCache: pool->v_cache[l] = p; break;
            case Region::kQsaKeys: pool->qsa_keys[l] = reinterpret_cast<bf16 *>(p); break;
            case Region::kMtpKCache: pool->mtp_k = p; break;
            case Region::kMtpVCache: pool->mtp_v = p; break;
            case Region::kMtpQsaKeys: pool->mtp_qsa_keys = reinterpret_cast<bf16 *>(p); break;
            default: break;
        }
    };
    layout::carve_pool(cfg, cfg.full_attention_layers, tokens, c, bind, draft);
}

}  // namespace

size_t sampler_scratch_bytes(const cere_qwen4_exp_config &cfg) {
    Carver c{nullptr};
    carve_sampler(cfg, c, nullptr);
    return size_t(c.size());
}

namespace {

// model.h SpecSampler, in one order for sizing and binding.
void carve_spec_sampler(const cere_qwen4_exp_config &cfg, uint64_t rows, Carver &c, SpecSampler *out) {
    const uint64_t vocab = cfg.vocab_size;
    const uint64_t words = (vocab + 31) / 32;
    SpecSampler s;
    s.rows = uint32_t(rows);
    s.proc = c.take<float>(rows * vocab * sizeof(float));
    s.probs = c.take<float>(rows * vocab * sizeof(float));
    s.keep = c.take<uint32_t>(rows * words * sizeof(uint32_t));
    s.top2_ids = c.take<uint32_t>(rows * 2 * sizeof(uint32_t));
    s.top2_scores = c.take<float>(rows * 2 * sizeof(float));
    s.committed = c.take<uint32_t>(rows * sizeof(uint32_t));
    s.count = c.take<uint32_t>(rows * sizeof(uint32_t));
    s.counts = c.take<uint32_t>(vocab * sizeof(uint32_t));
    s.accept = c.take<void>(k13s::scratch_bytes(3));
    if (out != nullptr) *out = s;
}

}  // namespace

uint64_t spec_sampler_bytes(const cere_qwen4_exp_config &cfg, uint64_t rows) {
    Carver c{nullptr};
    carve_spec_sampler(cfg, rows, c, nullptr);
    return c.size();
}

void bind_spec_sampler(const cere_qwen4_exp_config &cfg, uint64_t rows, uint8_t *base, Workspace *ws) {
    Carver c{base};
    carve_spec_sampler(cfg, rows, c, &ws->spec);
}

uint64_t staging_logprobs_rows(uint64_t max_tokens) {
    return std::min<uint64_t>(max_tokens, kernels::GemmPlan::kMaxRows);
}

// ---- arena order ----

#if defined(CERE_MOE_P3B)
namespace {
ArenaOrder order_of(const host::ModelPlan &plan, bool w13) {
#else
ArenaOrder arena_order(const host::ModelPlan &plan) {
#endif
    struct Run {
        std::vector<std::pair<uint64_t, uint64_t>> members;  // (rank, tensor index)
    };
    std::vector<Run> runs;
    std::unordered_map<std::string, size_t> keyed;
    std::string key;
    for (uint64_t i = 0; i < plan.tensors.size(); ++i) {
        const host::TensorPlan &t = plan.tensors[i];
        if (t.placement == CERE_PLACEMENT_DISCARD) continue;
        uint64_t member = 0;
#if defined(CERE_MOE_P3B)
        if (run_of(t.name, &key, &member, w13)) {
#else
        if (run_of(t.name, &key, &member)) {
#endif
            key += t.placement == CERE_PLACEMENT_HOST_PINNED ? "@host" : "@device";
            const auto [it, added] = keyed.emplace(key, runs.size());
            if (added) runs.emplace_back();
            runs[it->second].members.emplace_back(member, i);
        } else {
            runs.push_back(Run{{{0, i}}});
        }
    }
    ArenaOrder o;
    o.order.reserve(plan.tensors.size());
    o.packed.reserve(plan.tensors.size());
    for (Run &r : runs) {
        std::stable_sort(r.members.begin(), r.members.end(),
                         [](const auto &x, const auto &y) { return x.first < y.first; });
        for (size_t m = 0; m < r.members.size(); ++m) {
            o.order.push_back(r.members[m].second);
            o.packed.push_back(m != 0);
        }
    }
    return o;
}
#if defined(CERE_MOE_P3B)
}  // namespace

ArenaOrder arena_order(const host::ModelPlan &plan) { return order_of(plan, true); }
ArenaOrder arena_order_k7(const host::ModelPlan &plan) { return order_of(plan, false); }
void scratch_bytes_p3b_k7(const cere_qwen4_exp_config &cfg, uint64_t t, uint64_t ctx, uint64_t out[2]) {
    out[0] = scratch_bytes(cfg, t, ctx);
    out[1] = scratch_bytes_k7(cfg, t, ctx);
}
#endif

// ---- sizing ----

uint64_t workspace_rows(uint64_t max_context_tokens) {
    return std::min<uint64_t>(max_context_tokens, kMaxChunkTokens);
}

uint64_t workspace_bytes(const cere_qwen4_exp_config &cfg, uint64_t max_tokens, uint64_t max_context_tokens,
                         uint64_t draft_count) {
    Carver c{nullptr};
    carve_workspace(cfg, max_tokens, max_context_tokens, c, nullptr, draft_count);
    return c.size();
}

uint64_t staging_bytes(const cere_qwen4_exp_config &, uint64_t max_tokens) {
    Carver c{nullptr};
    carve_staging(max_tokens, c, nullptr);
    return c.size();
}

uint64_t derived_bytes(const cere_qwen4_exp_config &cfg, uint64_t max_context_tokens) {
    Carver c{nullptr};
    bind_derived(cfg, max_context_tokens, c, nullptr);
    return c.size();
}

uint64_t session_bytes(const cere_qwen4_exp_config &cfg, uint64_t max_context_tokens, bool draft) {
    Carver c{nullptr};
    bind_slot(cfg, max_context_tokens, c, nullptr, draft);
    return c.size();
}

uint64_t vision_scratch_bytes(const cere_qwen4_exp_vision_config &v, uint64_t max_context_tokens) {
    return vision::scratch_bytes(layout::max_image_patches(v, max_context_tokens));
}

uint64_t pool_bytes(const cere_qwen4_exp_config &cfg, uint64_t pool_tokens, bool draft) {
    Carver c{nullptr};
    bind_kv_pool(cfg, pool_tokens, c, nullptr, draft);
    return c.size();
}

uint64_t fixed_state_bytes(const cere_qwen4_exp_config &cfg) {
    const layout::FixedState f = layout::fixed_state(cfg, 1);
    return f.contiguous ? f.bytes : 0;
}

// ---- binding ----

cere_status bind_weights(const ArenaView &arena, ModelWeights *out, std::string *error) {
    try {
        if (arena.plan == nullptr || arena.offsets == nullptr || arena.device == nullptr || out == nullptr) {
            if (error != nullptr) *error = "bind_weights: null argument";
            return CERE_ERR_ARGUMENT;
        }
        const cere_qwen4_exp_config &cfg = arena.plan->config;
        if (cfg.num_hidden_layers != kLayers || cfg.hidden_size != kHidden || cfg.hc_count != kHc ||
            cfg.num_attention_heads != kQHeads || cfg.num_key_value_heads != kKvHeads || cfg.head_dim != kHeadDim ||
            cfg.ple_layer_ids != (uint64_t(1) << (kPleLayer + 1)) || rotary_dim(cfg) != kRotaryDim) {
            if (error != nullptr) *error = "bind_weights: config outside the compiled A4 shapes";
            return CERE_ERR_UNSUPPORTED;
        }
        // Track Fp4S3 (fp4local design note FL-30): the MXFP8 dense weights' formats, groups and scales
        // (model/mxfp8_plan.h), and the k2_weights lever, whose in-place conversion (fp8_weights.cpp) reads BF16.
        const mxfp8::Plan mx = mxfp8::plan_of(*arena.plan);
        if (!mx.error.empty()) {
            if (error != nullptr) *error = mx.error;
            return CERE_ERR_UNSUPPORTED;
        }
        if (mx.tensors != 0 && arena.plan->levers.k2_weights == CERE_LEVER_K2_WEIGHTS_FP8) {
            if (error != nullptr) {
                *error = "k2_weights = fp8 converts BF16 dense weights in place; this artifact's " +
                         std::to_string(mx.tensors) + " dense projections are MXFP8 (track Fp4S3: use k2_weights = "
                         "bf16, the MXFP8 GEMM path)";
            }
            return CERE_ERR_UNSUPPORTED;
        }
        Binder b(arena, error);
        ModelWeights w;
        w.config = cfg;
        w.max_context_tokens = arena.plan->max_context_tokens;
        const uint64_t H = cfg.hidden_size;
        const uint64_t V = cfg.vocab_size;
        const uint64_t E = cfg.num_experts;
        const uint64_t I = cfg.moe_intermediate_size;
        const uint64_t S = cfg.shared_expert_intermediate_size;
        const uint64_t hc_width = uint64_t(cfg.hc_count) * H;
        const uint64_t key_dim = uint64_t(cfg.linear_num_key_heads) * cfg.linear_key_head_dim;
        const uint64_t value_dim = uint64_t(cfg.linear_num_value_heads) * cfg.linear_value_head_dim;
        const uint64_t conv_dim = 2 * key_dim + value_dim;
        const uint64_t vh = cfg.linear_num_value_heads;
        const std::string root = "model.language_model.";
        // One GEMM weight of layer l: BF16, or MXFP8 listed in w.mxfp8 (Binder::dense).
        const auto dense = [&b, &w](const std::string &prefix, uint64_t n, uint64_t k) {
            const uint8_t *scales = nullptr;
            const bf16 *p = b.dense(prefix, n, k, &scales);
            if (p != nullptr && scales != nullptr) w.mxfp8.add(p, scales, uint32_t(n), uint32_t(k));
            return p;
        };

        w.embed = b.get<const bf16>(root + "embed_tokens.weight", CERE_DTYPE_BF16, {V, H});
        w.lm_head = b.get<const bf16>("lm_head.weight", CERE_DTYPE_BF16, {V, H});
        w.final_mixer = bind_hc(b, root + "hyper_connection_mixer.", cfg, false);

        for (uint32_t l = 0; l < kLayers && b.ok(); ++l) {
            LayerWeights &L = w.layers[l];
            const auto name = [l](const char *rest) { return layer_name(l, rest); };
            L.full_attention = full_attention(cfg, l);
            L.attn_hc = bind_hc(b, name("attn_hyper_connection."), cfg, true);
            L.mlp_hc = bind_hc(b, name("mlp_hyper_connection."), cfg, true);
            if (L.full_attention) {
                const uint64_t q_rows = uint64_t(cfg.num_attention_heads) * 2 * cfg.head_dim;
                const uint64_t kv_rows = uint64_t(cfg.num_key_value_heads) * cfg.head_dim;
                static const char *const kQkv[3] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"};
                const uint8_t *qs = nullptr;
                const uint8_t *ks = nullptr;
                const uint8_t *vs = nullptr;
                const bf16 *q = b.dense(name(kQkv[0]), q_rows, H, &qs);
                const bf16 *k = b.dense(name(kQkv[1]), kv_rows, H, &ks);
                const bf16 *v = b.dense(name(kQkv[2]), kv_rows, H, &vs);
                // Packed by arena_order: BF16 elements, or (MXFP8) E4M3 bytes and the scales' own run.
                const uint64_t esize = qs != nullptr ? 1 : sizeof(bf16);
                const auto at = [](const void *p, uint64_t bytes) { return static_cast<const uint8_t *>(p) + bytes; };
                if (q != nullptr && k != nullptr && v != nullptr &&
                    (at(k, 0) != at(q, q_rows * H * esize) || at(v, 0) != at(k, kv_rows * H * esize))) {
                    if (error != nullptr) {
                        *error = "tensors " + name("self_attn.{q,k,v}_proj.weight") + " are not packed";
                    }
                    return CERE_ERR_UNSUPPORTED;
                }
                if (qs != nullptr && ks != nullptr && vs != nullptr &&
                    (ks != qs + q_rows * H / mxfp8::kGroupK || vs != ks + kv_rows * H / mxfp8::kGroupK)) {
                    if (error != nullptr) {
                        *error = "tensors " + name("self_attn.{q,k,v}_proj.weight_scale") + " are not packed";
                    }
                    return CERE_ERR_UNSUPPORTED;
                }
                if (q != nullptr && qs != nullptr) w.mxfp8.add(q, qs, uint32_t(q_rows + 2 * kv_rows), uint32_t(H));
                L.attn.qkv = q;
                L.attn.q_norm = b.get<const bf16>(name("self_attn.q_norm.weight"), CERE_DTYPE_BF16, {cfg.head_dim});
                L.attn.k_norm = b.get<const bf16>(name("self_attn.k_norm.weight"), CERE_DTYPE_BF16, {cfg.head_dim});
                L.attn.o_proj = dense(name("self_attn.o_proj"), H, uint64_t(cfg.num_attention_heads) * cfg.head_dim);
                // QSA indexer (m22c F7): index_qk_proj [(n_heads + kv_heads) * head_dim, H] (q rows then k rows),
                // q_layernorm / k_layernorm [head_dim] (Gemma).
                const uint64_t idx_dim = cfg.indexer_head_dim;
                const uint64_t idx_rows = (uint64_t(cfg.indexer_n_heads) + cfg.indexer_kv_heads) * idx_dim;
                L.attn.indexer_qk = dense(name("self_attn.indexer.index_qk_proj"), idx_rows, H);
                L.attn.indexer_q_norm =
                    b.get<const bf16>(name("self_attn.indexer.q_layernorm.weight"), CERE_DTYPE_BF16, {idx_dim});
                L.attn.indexer_k_norm =
                    b.get<const bf16>(name("self_attn.indexer.k_layernorm.weight"), CERE_DTYPE_BF16, {idx_dim});
            } else {
                k5::LayerWeights &g = L.gdn;
                g.in_proj_qkv = dense(name("linear_attn.in_proj_qkv"), conv_dim, H);
                g.in_proj_z = dense(name("linear_attn.in_proj_z"), value_dim, H);
                g.in_proj_b = dense(name("linear_attn.in_proj_b"), vh, H);
                g.in_proj_a = dense(name("linear_attn.in_proj_a"), vh, H);
                g.conv1d = b.get<const bf16>(name("linear_attn.conv1d.weight"), CERE_DTYPE_BF16,
                                             {conv_dim, 1, cfg.linear_conv_kernel_dim});
                L.gdn_A_log_bf16 = b.get<const bf16>(name("linear_attn.A_log"), CERE_DTYPE_BF16, {vh});
                g.dt_bias = b.get<const bf16>(name("linear_attn.dt_bias"), CERE_DTYPE_BF16, {vh});
                g.norm = b.get<const bf16>(name("linear_attn.norm.weight"), CERE_DTYPE_BF16,
                                           {cfg.linear_value_head_dim});
                g.out_proj = dense(name("linear_attn.out_proj"), H, value_dim);
            }
            kernels::MoeLayer &m = L.moe;
            m.router = b.get<const bf16>(name("mlp.gate.weight"), CERE_DTYPE_BF16, {E, H});
            m.shared_gate = dense(name("mlp.shared_expert.gate_proj"), S, H);
            m.shared_up = dense(name("mlp.shared_expert.up_proj"), S, H);
            m.shared_down = dense(name("mlp.shared_expert.down_proj"), H, S);
            m.shared_expert_gate = b.get<const bf16>(name("mlp.shared_expert_gate.weight"), CERE_DTYPE_BF16, {1, H});
            const auto expert = [&](const char *proj, const char *suffix) {
                return [&name, proj, suffix](uint64_t e) {
                    return name("mlp.experts.") + std::to_string(e) + "." + proj + "_proj." + suffix;
                };
            };
#if defined(CERE_MOE_P3B)
            // M23-D2: w13 = [up; gate] per expert, one arena run (run_of); gate rows follow each expert's up rows.
            const auto w13 = [&name](const char *suffix) {
                return [&name, suffix](uint64_t i) {
                    return name("mlp.experts.") + std::to_string(i / 2) + (i % 2 ? ".gate_proj." : ".up_proj.") +
                           suffix;
                };
            };
            m.up_weight = b.run<const uint8_t>(2 * E, w13("weight"), CERE_DTYPE_U8, {I, H / 2});
            m.up_scale = b.run<const uint8_t>(2 * E, w13("weight_scale"), CERE_DTYPE_F8_E4M3, {I, H / 16});
            m.gate_weight = m.up_weight == nullptr ? nullptr : m.up_weight + I * (H / 2);
            m.gate_scale = m.up_scale == nullptr ? nullptr : m.up_scale + I * (H / 16);
#else
            m.gate_weight = b.run<const uint8_t>(E, expert("gate", "weight"), CERE_DTYPE_U8, {I, H / 2});
            m.gate_scale = b.run<const uint8_t>(E, expert("gate", "weight_scale"), CERE_DTYPE_F8_E4M3, {I, H / 16});
            m.up_weight = b.run<const uint8_t>(E, expert("up", "weight"), CERE_DTYPE_U8, {I, H / 2});
            m.up_scale = b.run<const uint8_t>(E, expert("up", "weight_scale"), CERE_DTYPE_F8_E4M3, {I, H / 16});
#endif
            m.down_weight = b.run<const uint8_t>(E, expert("down", "weight"), CERE_DTYPE_U8, {H, I / 2});
            m.down_scale = b.run<const uint8_t>(E, expert("down", "weight_scale"), CERE_DTYPE_F8_E4M3, {H, I / 16});
            MoeScaleSources &s = L.moe_scales;
            s.gate_input_scale = b.run<const float>(E, expert("gate", "input_scale"), CERE_DTYPE_F32, {});
            s.up_input_scale = b.run<const float>(E, expert("up", "input_scale"), CERE_DTYPE_F32, {});
            s.down_input_scale = b.run<const float>(E, expert("down", "input_scale"), CERE_DTYPE_F32, {});
            s.gate_weight_scale_2 = b.run<const float>(E, expert("gate", "weight_scale_2"), CERE_DTYPE_F32, {});
            s.down_weight_scale_2 = b.run<const float>(E, expert("down", "weight_scale_2"), CERE_DTYPE_F32, {});
#if defined(CERE_MOE_P3B)
            // up.weight_scale_2 is unused by the math (single-alpha semantics, k7_moe.h); P3b stores the device
            // {a1, a2} activation global scales FlashInfer reads in its first two floats at INSTANTIATE.
            m.act_scales = b.run<const float>(E, expert("up", "weight_scale_2"), CERE_DTYPE_F32, {});
#else
            // up.weight_scale_2 is unused (single-alpha semantics, k7_moe.h) but must be present and packed.
            b.run<const float>(E, expert("up", "weight_scale_2"), CERE_DTYPE_F32, {});
#endif
        }

        // PLE (layer kPleLayer).
        const auto ple = [](const char *rest) { return layer_name(kPleLayer, rest); };
        const uint64_t ngram_heads = uint64_t(cfg.ngram_size - 1) * cfg.heads_per_ngram;
        const uint64_t ple_row = cfg.ple_embed_dim / ngram_heads;
        k4::Weights &p = w.ple;
        const auto shard = [&ple](uint64_t i) {
            return ple("ple.ple_embedding.ngram_embedding.shard_") + std::to_string(i) + ".weight";
        };
        const auto shard_scale = [&ple](uint64_t i) {
            return ple("ple.ple_embedding.ngram_embedding.shard_") + std::to_string(i) + ".weight_scale";
        };
        const std::string fp8_scale = ple("ple.ple_embedding.ngram_embedding.weight_scale");
        const std::string nvfp4_global = ple("ple.ple_embedding.ngram_embedding.weight_scale_2");
        // The table format follows the shards' dtype (k4::TableFormat): U8 = packed NVFP4 rows with per-shard E4M3
        // block scales and one F32 global (fp4local design note D8; the runtime admits a U8 table only under the
        // artifact profile rtxpro6000bw-plehost-nvfp4-text-v1, native/upload.rs ple_table_format), else CUR's E4M3
        // rows with one BF16 scale. A tensor of the other format's scales is refused, so neither decode sees a mix.
        const bool nvfp4 = b.dtype_of(shard(0)) == CERE_DTYPE_U8;
        if (nvfp4 ? (ple_row != uint64_t(k4::nvfp4::kRowElements) || b.dtype_of(fp8_scale) != UINT32_MAX)
                  : (b.dtype_of(nvfp4_global) != UINT32_MAX || b.dtype_of(shard_scale(0)) != UINT32_MAX)) {
            if (error != nullptr) {
                *error = nvfp4 ? "PLE table: an NVFP4 table needs 160-element rows and no BF16 weight_scale"
                               : "PLE table: an E4M3 table takes no NVFP4 block scales or weight_scale_2";
            }
            return CERE_ERR_UNSUPPORTED;
        }
        p.format = nvfp4 ? k4::TableFormat::kNvfp4 : k4::TableFormat::kFp8;
        const uint64_t row_bytes = nvfp4 ? uint64_t(k4::nvfp4::kRowBytes) : ple_row;
        const uint32_t table_dtype = nvfp4 ? CERE_DTYPE_U8 : CERE_DTYPE_F8_E4M3;
        // Shards (and an NVFP4 table's block-scale shards) are checked by their own row count (a missing one is
        // reported by name), then for contiguity in shard order; the table is their concatenation.
        const uint8_t *expect = nullptr;
        const uint8_t *expect_scales = nullptr;
        for (uint64_t i = 0; i < cfg.split_ngram_parts && b.ok(); ++i) {
            const uint64_t rows = b.rows_of(shard(i));
            const uint8_t *s =
                b.get<const uint8_t>(shard(i), table_dtype, {rows, row_bytes}, CERE_PLACEMENT_HOST_PINNED);
            if (s == nullptr) break;
            if (i == 0) p.table = s;
            if (expect != nullptr && s != expect) {
                if (error != nullptr) *error = "tensor " + shard(i) + " does not follow its predecessor (PLE table)";
                return CERE_ERR_UNSUPPORTED;
            }
            expect = s + rows * row_bytes;
            p.table_rows += int64_t(rows);
            if (!nvfp4) continue;
            const uint8_t *sc = b.get<const uint8_t>(shard_scale(i), CERE_DTYPE_F8_E4M3,
                                                     {rows, uint64_t(k4::nvfp4::kRowScales)},
                                                     CERE_PLACEMENT_HOST_PINNED);
            if (sc == nullptr) break;
            if (i == 0) p.block_scales = sc;
            if (expect_scales != nullptr && sc != expect_scales) {
                if (error != nullptr) {
                    *error = "tensor " + shard_scale(i) + " does not follow its predecessor (PLE block scales)";
                }
                return CERE_ERR_UNSUPPORTED;
            }
            expect_scales = sc + rows * uint64_t(k4::nvfp4::kRowScales);
        }
        if (nvfp4) {
            p.global_scale = b.get<const float>(nvfp4_global, CERE_DTYPE_F32, {1}, CERE_PLACEMENT_HOST_PINNED);
        } else {
            p.table_scale = b.get<const bf16>(fp8_scale, CERE_DTYPE_BF16, {1}, CERE_PLACEMENT_HOST_PINNED);
        }
        p.multipliers = b.get<const int64_t>(ple("ple.ple_embedding.layer_multipliers"), CERE_DTYPE_I64,
                                             {cfg.ngram_size}, CERE_PLACEMENT_HOST_PINNED);
        p.head_vocab_sizes = b.get<const int64_t>(ple("ple.ple_embedding.ngram_heads_vocab_sizes"), CERE_DTYPE_I64,
                                                  {ngram_heads}, CERE_PLACEMENT_HOST_PINNED);
        p.head_offsets = b.get<const int64_t>(ple("ple.ple_embedding.ngram_heads_offsets"), CERE_DTYPE_I64,
                                              {ngram_heads}, CERE_PLACEMENT_HOST_PINNED);
        p.key_proj = b.get<const bf16>(ple("ple.key_proj.weight"), CERE_DTYPE_BF16, {hc_width, cfg.ple_embed_dim});
        p.value_proj = b.get<const bf16>(ple("ple.value_proj.weight"), CERE_DTYPE_BF16, {H, cfg.ple_embed_dim});
        p.norm_key = b.get<const bf16>(ple("ple.norm_key.weight"), CERE_DTYPE_BF16, {hc_width});
        p.norm_query = b.get<const bf16>(ple("ple.norm_query.weight"), CERE_DTYPE_BF16, {hc_width});
        p.norm_conv = b.get<const bf16>(ple("ple.norm_conv.weight"), CERE_DTYPE_BF16, {hc_width});
        p.conv_weight = b.get<const bf16>(ple("ple.conv1d.weight"), CERE_DTYPE_BF16,
                                          {hc_width, 1, cfg.ple_conv_kernel_size});
        p.eos_token_id = k4::kEosTokenId;
        p.eps = float(cfg.rms_norm_eps);
        if (!b.ok()) return CERE_ERR_UNSUPPORTED;

        // The vision tower (m213 design note D10), compiled for the A4 vision_config only.
        const host::Support *support = arena.plan->support;
        if (const cere_qwen4_exp_vision_config *v = support == nullptr ? nullptr : support->vision; v != nullptr) {
            if (v->depth != vision::kDepth || v->hidden_size != vision::kHidden ||
                v->intermediate_size != vision::kInter || v->out_hidden_size != vision::kOutHidden ||
                v->spatial_merge_size * v->spatial_merge_size != vision::kMergeUnit ||
                v->num_position_embeddings != vision::kPosSide * vision::kPosSide ||
                v->in_channels * v->temporal_patch_size * v->patch_size * v->patch_size != vision::kPatchRow) {
                if (error != nullptr) *error = "vision_config: not the compiled vision tower";
                return CERE_ERR_UNSUPPORTED;
            }
            for (uint32_t i = 0; i < vision::kTensorCount && b.ok(); ++i) {
                uint64_t shape[5];
                const uint32_t ndim = vision::tensor_shape(i, shape);
                w.vision.tensor[i] = b.get<const bf16>(vision::tensor_name(i), CERE_DTYPE_BF16, shape, ndim);
            }
            if (!b.ok()) return CERE_ERR_UNSUPPORTED;
        }

        // M2.9 (m29 design note D3): the MTP draft layer, only for a plan with drafts (its 31 `mtp.*` tensors, 35 on
        // the S3 artifact, are DEVICE then; without drafts they are DISCARD and nothing here reads them).
        if (arena.plan->speculative_draft_tokens != 0) {
            if (arena.plan->draft_token_map.empty()) {
                if (error != nullptr) *error = "bind_weights: drafts without a draft token map";
                return CERE_ERR_ARGUMENT;
            }
            if (const cere_status st = bind_mtp(b, cfg, &w.mtp, error); st != CERE_OK) return st;
            w.mtp.count = uint32_t(arena.plan->draft_token_map.size());
            w.has_draft = true;
        }

        w.derived = arena.derived;
        Carver c{arena.derived};
        bind_derived(cfg, w.max_context_tokens, c, &w);
        if (w.has_draft) {
            layout::carve_mtp_derived(cfg, w.mtp.count, c, [&w](Region r, uint32_t, uint8_t *p) {
                if (r == Region::kMtpLmHead) w.mtp.lm_head = reinterpret_cast<bf16 *>(p);
                if (r == Region::kMtpMap) w.mtp.map = reinterpret_cast<uint32_t *>(p);
            });
        }
        *out = w;
        return CERE_OK;
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
}

cere_status bind_workspace(const cere_qwen4_exp_config &cfg, uint64_t max_tokens, uint64_t max_context_tokens,
                           uint8_t *device_region, uint8_t *host_staging, Workspace *ws, uint64_t draft_count) {
    if (device_region == nullptr || host_staging == nullptr || ws == nullptr || max_tokens == 0 ||
        max_context_tokens == 0) {
        return CERE_ERR_ARGUMENT;
    }
    Carver d{device_region};
    carve_workspace(cfg, max_tokens, max_context_tokens, d, ws, draft_count);
    Carver h{host_staging};
    carve_staging(max_tokens, h, ws);
    return CERE_OK;
}

cere_status bind_mxfp8(ModelWeights *weights, bf16 *staging, uint64_t staging_bytes, Workspace *ws) {
    if (weights == nullptr || ws == nullptr) return CERE_ERR_ARGUMENT;
    kernels::k2m::Table &t = weights->mxfp8;
    if (t.empty()) {
        ws->mxfp8 = nullptr;
        return CERE_OK;
    }
    if (staging == nullptr || reinterpret_cast<uintptr_t>(staging) % layout::kAlign != 0) return CERE_ERR_ARGUMENT;
    if (staging_bytes < t.staging_needed()) return CERE_ERR_INTERNAL;
    t.set_staging(staging, staging_bytes);
    ws->mxfp8 = &t;
    return CERE_OK;
}

cere_status bind_pool(const cere_qwen4_exp_config &cfg, uint8_t *base, uint64_t pool_tokens, KvPool *out,
                      bool draft) {
    if (out == nullptr || (base == nullptr && pool_tokens != 0) || pool_tokens % host::kKvPageTokens != 0) {
        return CERE_ERR_ARGUMENT;
    }
    KvPool p;
    p.tokens = pool_tokens;
    Carver c{base};
    bind_kv_pool(cfg, pool_tokens, c, &p, draft);
    *out = p;
    return CERE_OK;
}

cere_status bind_session(const cere_qwen4_exp_config &cfg, uint8_t *slot, uint64_t max_context_tokens,
                         const KvPool &pool, SessionState *out) {
    if (slot == nullptr || out == nullptr || max_context_tokens == 0) return CERE_ERR_ARGUMENT;
    SessionState s;
    s.max_context_tokens = max_context_tokens;
    s.slot = slot;
    Carver c{slot};
    const bool draft = pool.mtp_k != nullptr;
    bind_slot(cfg, max_context_tokens, c, &s, draft);
    s.slot_bytes = c.size();
    for (uint32_t l = 0; l < kLayers; ++l) {
        s.k_cache[l] = pool.k_cache[l];
        s.v_cache[l] = pool.v_cache[l];
        s.qsa_keys[l] = pool.qsa_keys[l];
    }
    s.mtp_k_cache = pool.mtp_k;
    s.mtp_v_cache = pool.mtp_v;
    s.mtp_qsa_keys = pool.mtp_qsa_keys;
    *out = s;
    return CERE_OK;
}

cere_status session_reset(const cere_qwen4_exp_config &cfg, SessionState *state, cudaStream_t stream) {
    if (state == nullptr || state->slot == nullptr || state->ple.history == nullptr) return CERE_ERR_ARGUMENT;
    if (cudaMemsetAsync(state->slot, 0, state->slot_bytes, stream) != cudaSuccess) {
        cudaGetLastError();
        return CERE_ERR_INTERNAL;
    }
    // Pageable source: cudaMemcpyAsync stages it before returning, so a static array is safe.
    static const int64_t kEos[2] = {k4::kEosTokenId, k4::kEosTokenId};
    const uint64_t history = uint64_t(cfg.ngram_size == 0 ? 0 : cfg.ngram_size - 1);
    if (history > 2) return CERE_ERR_UNSUPPORTED;
    if (cudaMemcpyAsync(state->ple.history, kEos, history * kI64, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        cudaGetLastError();
        return CERE_ERR_INTERNAL;
    }
    state->position = 0;
    state->rope_delta = 0;
    return CERE_OK;
}

cere_status session_set_pages(SessionState *state, const uint32_t *pages, uint64_t n, uint64_t pool_tokens,
                              cudaStream_t stream) {
    if (state == nullptr || state->slot_map == nullptr || pages == nullptr ||
        n != host::kv_pages(state->max_context_tokens)) {
        return CERE_ERR_ARGUMENT;
    }
    const uint64_t pool_pages = pool_tokens / host::kKvPageTokens;
    for (uint64_t i = 0; i < n; ++i) {
        if (pages[i] >= pool_pages) return CERE_ERR_ARGUMENT;
    }
    try {
        std::vector<int64_t> map(state->max_context_tokens);
        for (uint64_t p = 0; p < map.size(); ++p) {
            map[p] = int64_t(uint64_t(pages[p / host::kKvPageTokens]) * host::kKvPageTokens + p % host::kKvPageTokens);
        }
        // Pageable source: cudaMemcpyAsync stages it before returning, so the vector may go out of scope.
        if (cudaMemcpyAsync(state->slot_map, map.data(), map.size() * kI64, cudaMemcpyHostToDevice, stream) !=
            cudaSuccess) {
            cudaGetLastError();
            return CERE_ERR_INTERNAL;
        }
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    return CERE_OK;
}

uint64_t fixed_state_offset(const cere_qwen4_exp_config &cfg, uint64_t max_context_tokens) {
    return layout::fixed_state(cfg, max_context_tokens).begin;
}

cere_status bind_fixed_state(const cere_qwen4_exp_config &cfg, uint8_t *state, uint64_t n_pages, const KvPool &pool,
                             SessionState *out) {
    if (state == nullptr || out == nullptr || n_pages == 0 || n_pages > UINT64_MAX / host::kKvPageTokens) {
        return CERE_ERR_ARGUMENT;
    }
    // Carved as a one-token slot whose run starts at `state`: the regions before the run (the slot map) and after it
    // (out_counts) are dropped below, so no pointer outside [state, state + fixed_state_bytes) is kept.
    const layout::FixedState f = layout::fixed_state(cfg, 1);
    if (!f.contiguous) return CERE_ERR_UNSUPPORTED;
    SessionState s;
    Carver c{state - f.begin};
    bind_slot(cfg, 1, c, &s);
    s.slot = state;
    s.slot_bytes = f.bytes;
    s.slot_map = nullptr;
    s.out_counts = nullptr;
    s.position = 0;
    s.max_context_tokens = n_pages * host::kKvPageTokens;
    for (uint32_t l = 0; l < kLayers; ++l) {
        s.k_cache[l] = pool.k_cache[l];
        s.v_cache[l] = pool.v_cache[l];
        s.qsa_keys[l] = pool.qsa_keys[l];
    }
    // Sessions (O211): the MTP K/V rows and QSA keys of a model with drafts (null without one), so a prefix node's
    // host-tier image carries its edge's MTP rows (kind 10); a node has no MTP ring rows or pending proposals.
    s.mtp_k_cache = pool.mtp_k;
    s.mtp_v_cache = pool.mtp_v;
    s.mtp_qsa_keys = pool.mtp_qsa_keys;
    *out = s;
    return CERE_OK;
}

namespace {

// One contiguous run of session-slot bytes inside a section, in section order.
struct Piece {
    uint8_t *device;
    uint64_t bytes;
};

// Appends the pool bytes of the session's logical rows [begin, end) of one layer's pool array `base`, `rows_per_page`
// rows of `row_bytes` per 64-token page, through the session's `pages` (logical page i is pool page pages[i]; D4:
// the slot map's gather, ledger X19): one piece per physically contiguous run, so identity pages copy as one run.
// False when a row lies past the session's pages.
bool pool_rows(uint8_t *base, const uint32_t *pages, uint64_t n_pages, uint64_t rows_per_page, uint64_t row_bytes,
               uint64_t begin, uint64_t end, std::vector<Piece> *out) {
    for (uint64_t r = begin; r < end;) {
        const uint64_t page = r / rows_per_page;
        if (page >= n_pages) return false;
        const uint64_t off = r % rows_per_page;
        const uint64_t n = std::min(rows_per_page - off, end - r);
        uint8_t *at = base + (uint64_t(pages[page]) * rows_per_page + off) * row_bytes;
        if (!out->empty() && out->back().device + out->back().bytes == at) {
            out->back().bytes += n * row_bytes;
        } else {
            out->push_back({at, n * row_bytes});
        }
        r += n;
    }
    return true;
}

// The pieces of section `kind` (1-8, 10) at rows [first_row, position), in the section's byte order (D2: layers
// ascending; kinds 1-3 gathered from the pool through `pages`, the others from the slot; kind 8 the sampler's
// out_counts, which a fresh state holds zeroed; kind 10 the MTP layer's K, V and compressed keys from its pool entry,
// its ring, then the pending proposals, host/snapshot_format.h draft_parts); false for another kind, kind 10 of a
// session without an MTP entry, a compress ratio that does not divide the page, or rows past the session's pages.
bool section_pieces(const cere_qwen4_exp_config &cfg, const SessionState &s, const uint32_t *pages, uint64_t n_pages,
                    uint32_t kind, uint64_t first_row, uint64_t position, std::vector<Piece> *out) {
    const uint64_t kv_width = uint64_t(cfg.num_key_value_heads) * cfg.head_dim;  // FP8: bytes per row
    const uint64_t key_row = uint64_t(cfg.indexer_head_dim) * kBf16;
    const uint64_t ratio = cfg.indexer_compress_ratio == 0 ? 1 : cfg.indexer_compress_ratio;
    const uint64_t conv_bytes = (2 * uint64_t(cfg.linear_num_key_heads) * cfg.linear_key_head_dim +
                                 uint64_t(cfg.linear_num_value_heads) * cfg.linear_value_head_dim) *
                                (cfg.linear_conv_kernel_dim == 0 ? 0 : cfg.linear_conv_kernel_dim - 1) * kBf16;
    const uint64_t ssm_bytes =
        uint64_t(cfg.linear_num_value_heads) * cfg.linear_key_head_dim * cfg.linear_value_head_dim * kBf16;
    const uint32_t layers = std::min<uint32_t>(cfg.num_hidden_layers, kLayers);
    out->clear();
    switch (kind) {
        case 1:
        case 2:
            for (uint32_t l = 0; l < layers; ++l) {
                if (!layout::full_attention(cfg, l)) continue;
                uint8_t *rows = kind == 1 ? s.k_cache[l] : s.v_cache[l];
                if (!pool_rows(rows, pages, n_pages, host::kKvPageTokens, kv_width, first_row, position, out)) {
                    return false;
                }
            }
            return true;
        case 3:
            // Block b's compressed key is pool row slot_map[r * b] / r: 64 / r rows per page.
            if (host::kKvPageTokens % ratio != 0) return false;
            for (uint32_t l = 0; l < layers; ++l) {
                if (!layout::full_attention(cfg, l)) continue;
                auto *keys = reinterpret_cast<uint8_t *>(s.qsa_keys[l]);
                if (!pool_rows(keys, pages, n_pages, host::kKvPageTokens / ratio, key_row, first_row / ratio,
                               position / ratio, out)) {
                    return false;
                }
            }
            return true;
        case 4:
            // Ring row p mod r holds position p of the incomplete last block (k9_qsa.h): rows [0, position mod r).
            for (uint32_t l = 0; l < layers; ++l) {
                if (!layout::full_attention(cfg, l)) continue;
                out->push_back({reinterpret_cast<uint8_t *>(s.qsa_ring[l]), position % ratio * key_row});
            }
            return true;
        case 5:
        case 6:
            for (uint32_t l = 0; l < layers; ++l) {
                if (layout::full_attention(cfg, l)) continue;
                if (kind == 5) {
                    out->push_back({reinterpret_cast<uint8_t *>(s.gdn[l].conv), conv_bytes});
                } else {
                    out->push_back({reinterpret_cast<uint8_t *>(s.gdn[l].ssm), ssm_bytes});
                }
            }
            return true;
        case 7: {
            const uint64_t history = uint64_t(cfg.ngram_size == 0 ? 0 : cfg.ngram_size - 1) * kI64;
            const uint64_t taps =
                uint64_t(cfg.ple_conv_kernel_size == 0 ? 0 : cfg.ple_conv_kernel_size - 1) * cfg.ngram_size;
            out->push_back({reinterpret_cast<uint8_t *>(s.ple.history), history});
            out->push_back({reinterpret_cast<uint8_t *>(s.ple.conv_state),
                            uint64_t(cfg.hc_count) * cfg.hidden_size * taps * kBf16});
            return true;
        }
        case 8:
            out->push_back({reinterpret_cast<uint8_t *>(s.out_counts), uint64_t(cfg.vocab_size) * sizeof(uint32_t)});
            return true;
        case 10:
            // One MTP layer (draft_parts sizes layers x rows; the native draft has exactly one). A prefix node's view
            // (bind_fixed_state) has no ring and no proposals: its depth is a multiple of 64, so no ring row is valid,
            // and its proposals are zeros in the image (device null), skipped on load.
            if (cfg.mtp_num_hidden_layers != 1 || s.mtp_k_cache == nullptr || s.mtp_v_cache == nullptr ||
                s.mtp_qsa_keys == nullptr || (s.mtp_qsa_ring == nullptr && position % ratio != 0) ||
                host::kKvPageTokens % ratio != 0) {
                return false;
            }
            if (!pool_rows(s.mtp_k_cache, pages, n_pages, host::kKvPageTokens, kv_width, first_row, position, out) ||
                !pool_rows(s.mtp_v_cache, pages, n_pages, host::kKvPageTokens, kv_width, first_row, position, out) ||
                !pool_rows(reinterpret_cast<uint8_t *>(s.mtp_qsa_keys), pages, n_pages, host::kKvPageTokens / ratio,
                           key_row, first_row / ratio, position / ratio, out)) {
                return false;
            }
            if (s.mtp_qsa_ring != nullptr) {
                out->push_back({reinterpret_cast<uint8_t *>(s.mtp_qsa_ring), position % ratio * key_row});
            }
            out->push_back({reinterpret_cast<uint8_t *>(s.proposals), host::snapshot::kDraftProposalBytes});
            return true;
        default:
            return false;
    }
}

// Every device section of `plan` (not a host kind: tokens, rope_delta) as copy runs in image order, checked against
// the section sizes.
cere_status plan_copies(const cere_qwen4_exp_config &cfg, const SessionState &s, const uint32_t *pages,
                        uint64_t n_pages, const host::snapshot::Plan &plan, uint64_t first_row, uint64_t position,
                        bool load, std::vector<StateCopy> *out) {
    if (first_row > position || pages == nullptr || n_pages != host::kv_pages(s.max_context_tokens)) {
        return CERE_ERR_ARGUMENT;
    }
    const uint64_t ring_bytes =
        uint64_t(cfg.indexer_compress_ratio == 0 ? 1 : cfg.indexer_compress_ratio) * cfg.indexer_head_dim * kBf16;
    out->clear();
    std::vector<Piece> pieces;
    for (const host::snapshot::Section &section : plan.sections) {
        if (((host::snapshot::kHostKinds >> section.kind) & 1) != 0) continue;
        if (!section_pieces(cfg, s, pages, n_pages, section.kind, first_row, position, &pieces)) {
            return CERE_ERR_UNSUPPORTED;
        }
        uint64_t at = section.offset;
        for (const Piece &p : pieces) {
            // The load zero-fills the QSA ring rows past the valid ones.
            const uint64_t zero = load && section.kind == host::snapshot::kQsaRing ? ring_bytes - p.bytes : 0;
            out->push_back({p.device, at, p.bytes, zero});
            at += p.bytes;
        }
        if (at - section.offset != section.bytes) return CERE_ERR_ARGUMENT;
    }
    return CERE_OK;
}

}  // namespace

cere_status save_copies(const cere_qwen4_exp_config &cfg, const SessionState &state, const uint32_t *pages,
                        uint64_t n_pages, const host::snapshot::Plan &plan, uint64_t first_row,
                        std::vector<StateCopy> *out) {
    if (state.slot == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    return plan_copies(cfg, state, pages, n_pages, plan, first_row, state.position, false, out);
}

cere_status load_copies(const cere_qwen4_exp_config &cfg, const SessionState &state, const uint32_t *pages,
                        uint64_t n_pages, const host::snapshot::Plan &plan, uint64_t first_row, uint64_t position,
                        std::vector<StateCopy> *out) {
    if (state.slot == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    if (state.position != 0) return CERE_ERR_ARGUMENT;
    if (position > state.max_context_tokens) return CERE_ERR_CAPACITY;
    return plan_copies(cfg, state, pages, n_pages, plan, first_row, position, true, out);
}

}  // namespace cere::model
