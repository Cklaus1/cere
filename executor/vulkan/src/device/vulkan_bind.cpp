#include "device/vulkan_bind.h"

#include "model/layout.h"
#include "model/mxfp8_plan.h"

#include <initializer_list>
#include <memory>
#include <new>
#include <string_view>
#include <unordered_map>

namespace cere::vkdev {

namespace layout = cere::model::layout;
using layout::Region;

namespace {

const char kLayerPrefix[] = "model.language_model.layers.";

std::string layer_name(uint32_t layer, const char *rest) {
    return std::string(kLayerPrefix) + std::to_string(layer) + "." + rest;
}

// Measuring carves (layout::Carver) report region starts as pointers; a non-null fake base turns them into offsets.
// Pointer arithmetic only, never dereferenced.
uint8_t *const kFakeBase = reinterpret_cast<uint8_t *>(uintptr_t(1) << 40);
uint64_t off_of(const uint8_t *p) { return uint64_t(p - kFakeBase); }

// CUDA's Binder (model.cpp:137-237) over addresses.
class Binder {
public:
    Binder(const host::ModelPlan &plan, const VkLayout &l, const std::vector<uint64_t> &addr, std::string *error)
        : plan_(plan), l_(l), addr_(addr), error_(error) {
        index_.reserve(plan.tensors.size());
        for (uint64_t i = 0; i < plan.tensors.size(); ++i) index_.emplace(plan.tensors[i].name, i);
    }
    bool ok() const { return ok_; }

    uint64_t get(const std::string &name, uint32_t dtype, std::initializer_list<uint64_t> shape,
                 uint32_t placement = CERE_PLACEMENT_DEVICE, uint64_t *index_out = nullptr) {
        const auto it = index_.find(name);
        if (it == index_.end()) return fail("missing tensor " + name);
        const host::TensorPlan &t = plan_.tensors[it->second];
        if (t.dtype != dtype) {
            return fail("tensor " + name + ": dtype " + std::to_string(t.dtype) + ", expected " + std::to_string(dtype));
        }
        if (t.placement != placement) {
            return fail("tensor " + name + ": placement " + std::to_string(t.placement) + ", expected " +
                        std::to_string(placement));
        }
        bool shape_ok = t.ndim == shape.size();
        size_t d = 0;
        for (uint64_t s : shape) {
            if (shape_ok) shape_ok = t.shape[d] == s;
            ++d;
        }
        if (!shape_ok) {
            std::string have = "[", want = "[";
            for (uint32_t i = 0; i < t.ndim; ++i) have += (i ? ", " : "") + std::to_string(t.shape[i]);
            d = 0;
            for (uint64_t s : shape) want += (d++ ? ", " : "") + std::to_string(s);
            return fail("tensor " + name + ": shape " + have + "], expected " + want + "]");
        }
        const uint64_t first = l_.piece_first[it->second], last = l_.piece_first[it->second + 1];
        // Only a PLE shard may be split (R4); a kernel takes one base address per tensor.
        if (first == last) return fail("tensor " + name + ": no storage in the Vulkan layout");
        if (last - first != 1 && placement != CERE_PLACEMENT_HOST_PINNED) {
            return fail("tensor " + name + ": split over Vulkan blocks");
        }
        if (index_out != nullptr) *index_out = it->second;
        const VkPiece &p = l_.pieces[first];
        return addr_[p.block] + p.offset;
    }

    // `count` tensors of equal shape that must abut (an arena_order run); returns the first's address.
    template <typename Name>
    uint64_t run(uint64_t count, Name name_of, uint32_t dtype, std::initializer_list<uint64_t> shape,
                 uint32_t placement = CERE_PLACEMENT_DEVICE) {
        uint64_t first = 0, expect = 0;
        for (uint64_t i = 0; i < count; ++i) {
            uint64_t index = 0;
            const std::string name = name_of(i);
            const uint64_t a = get(name, dtype, shape, placement, &index);
            if (a == 0) return 0;
            if (i == 0) first = a;
            if (i != 0 && a != expect) return fail("tensor " + name + " does not follow its predecessor (arena run)");
            expect = a + plan_.tensors[index].bytes;
        }
        return first;
    }

    uint32_t dtype_of(const std::string &name) const {
        const auto it = index_.find(name);
        return it == index_.end() ? UINT32_MAX : plan_.tensors[it->second].dtype;
    }
    uint64_t rows_of(const std::string &name) const {
        const auto it = index_.find(name);
        return it == index_.end() ? 0 : plan_.tensors[it->second].shape[0];
    }

    // Binder::dense (model.cpp:216-223): BF16, or MXFP8 (E4M3 weight + U8 weight_scale [n, k / 32]).
    VkDense dense(const std::string &prefix, uint64_t n, uint64_t k) {
        VkDense d;
        d.n = uint32_t(n);
        d.k = uint32_t(k);
        const std::string weight = prefix + ".weight";
        if (dtype_of(weight) != CERE_DTYPE_F8_E4M3) {
            d.w = get(weight, CERE_DTYPE_BF16, {n, k});
            return d;
        }
        d.w = get(weight, CERE_DTYPE_F8_E4M3, {n, k});
        d.scales = get(prefix + ".weight_scale", CERE_DTYPE_U8, {n, k / model::mxfp8::kGroupK});
        if (d.w != 0 && d.scales != 0) ++mxfp8_;
        return d;
    }
    uint32_t mxfp8_count() const { return mxfp8_; }

    uint64_t fail(std::string message) {
        if (ok_ && error_ != nullptr) *error_ = std::move(message);
        ok_ = false;
        return 0;
    }

private:
    const host::ModelPlan &plan_;
    const VkLayout &l_;
    const std::vector<uint64_t> &addr_;
    std::string *error_;
    std::unordered_map<std::string_view, uint64_t> index_;
    bool ok_ = true;
    uint32_t mxfp8_ = 0;
};

VkHcWeights bind_hc(Binder &b, const std::string &prefix, const cere_qwen4_exp_config &cfg, bool inject) {
    const uint64_t width = uint64_t(cfg.hc_count) * cfg.hidden_size;
    VkHcWeights w;
    w.norm = b.get(prefix + "hc_norm.weight", CERE_DTYPE_BF16, {width});
    w.down = b.get(prefix + "input_mix_weight_down.weight", CERE_DTYPE_BF16, {cfg.hc_lowrank, width});
    w.up = b.get(prefix + "input_mix_weight_up.weight", CERE_DTYPE_BF16, {width, cfg.hc_lowrank});
    if (inject) w.inject = b.get(prefix + "block_inject_weight.weight", CERE_DTYPE_BF16, {cfg.hc_count, width});
    return w;
}

// q|k|v of one attention layer as one GEMM (model.cpp:735-762): the three weights packed, and with MXFP8 the three
// scale tensors packed as their own run.
VkDense bind_qkv(Binder &b, const std::string &prefix, const cere_qwen4_exp_config &cfg) {
    const uint64_t H = cfg.hidden_size;
    const uint64_t q_rows = uint64_t(cfg.num_attention_heads) * 2 * cfg.head_dim;
    const uint64_t kv_rows = uint64_t(cfg.num_key_value_heads) * cfg.head_dim;
    const VkDense q = b.dense(prefix + "self_attn.q_proj", q_rows, H);
    const VkDense k = b.dense(prefix + "self_attn.k_proj", kv_rows, H);
    const VkDense v = b.dense(prefix + "self_attn.v_proj", kv_rows, H);
    if (!b.ok()) return VkDense{};
    const uint64_t esize = q.mxfp8() ? 1 : 2;
    if (q.mxfp8() != k.mxfp8() || q.mxfp8() != v.mxfp8()) {
        b.fail("tensors " + prefix + "self_attn.{q,k,v}_proj.weight mix BF16 and MXFP8");
        return VkDense{};
    }
    if (k.w != q.w + q_rows * H * esize || v.w != k.w + kv_rows * H * esize) {
        b.fail("tensors " + prefix + "self_attn.{q,k,v}_proj.weight are not packed");
        return VkDense{};
    }
    const uint64_t g = model::mxfp8::kGroupK;
    if (q.mxfp8() && (k.scales != q.scales + q_rows * H / g || v.scales != k.scales + kv_rows * H / g)) {
        b.fail("tensors " + prefix + "self_attn.{q,k,v}_proj.weight_scale are not packed");
        return VkDense{};
    }
    VkDense d = q;
    d.n = uint32_t(q_rows + 2 * kv_rows);
    return d;
}

void bind_attention(Binder &b, const std::string &prefix, const cere_qwen4_exp_config &cfg, VkAttentionWeights *a) {
    const uint64_t H = cfg.hidden_size;
    a->qkv = bind_qkv(b, prefix, cfg);
    a->q_norm = b.get(prefix + "self_attn.q_norm.weight", CERE_DTYPE_BF16, {cfg.head_dim});
    a->k_norm = b.get(prefix + "self_attn.k_norm.weight", CERE_DTYPE_BF16, {cfg.head_dim});
    a->o_proj = b.dense(prefix + "self_attn.o_proj", H, uint64_t(cfg.num_attention_heads) * cfg.head_dim);
    const uint64_t idx_dim = cfg.indexer_head_dim;
    const uint64_t idx_rows = (uint64_t(cfg.indexer_n_heads) + cfg.indexer_kv_heads) * idx_dim;
    a->indexer_qk = b.dense(prefix + "self_attn.indexer.index_qk_proj", idx_rows, H);
    a->indexer_q_norm = b.get(prefix + "self_attn.indexer.q_layernorm.weight", CERE_DTYPE_BF16, {idx_dim});
    a->indexer_k_norm = b.get(prefix + "self_attn.indexer.k_layernorm.weight", CERE_DTYPE_BF16, {idx_dim});
}

// bind_mtp (model.cpp:257-317).
void bind_mtp(Binder &b, const cere_qwen4_exp_config &cfg, VkMtpWeights *m) {
    const uint64_t H = cfg.hidden_size, E = cfg.num_experts, I = cfg.moe_intermediate_size,
                   S = cfg.shared_expert_intermediate_size, hc_width = uint64_t(cfg.hc_count) * H;
    const std::string root = "mtp.", layer = "mtp.layers.0.";
    m->fc_embedding = b.dense(root + "fc_embedding", H, H);
    m->fc_hidden = b.dense(root + "fc_hidden", H, H);
    m->norm_embedding = b.get(root + "pre_fc_norm_embedding.weight", CERE_DTYPE_BF16, {H});
    m->norm_hidden = b.get(root + "pre_fc_norm_hidden.weight", CERE_DTYPE_BF16, {hc_width});
    m->mixer = bind_hc(b, root + "hyper_connection_mixer.", cfg, false);
    m->attn_hc = bind_hc(b, layer + "attn_hyper_connection.", cfg, true);
    m->mlp_hc = bind_hc(b, layer + "mlp_hyper_connection.", cfg, true);
    bind_attention(b, layer, cfg, &m->attn);
    vkk13::MoeBf16Layer &x = m->moe;
    x.router = b.get(layer + "mlp.gate.weight", CERE_DTYPE_BF16, {E, H});
    const std::string experts = layer + "mlp.experts.";
    if (b.dtype_of(experts + "gate_up_proj") == CERE_DTYPE_U8) {
        x.gate_up_q = b.get(experts + "gate_up_proj", CERE_DTYPE_U8, {E, 2 * I, H / 2});
        x.gate_up_scale = b.get(experts + "gate_up_proj_scale", CERE_DTYPE_F8_E4M3, {E, 2 * I, H / 16});
        x.gate_up_global = b.get(experts + "gate_up_proj_scale_2", CERE_DTYPE_F32, {E, 2});
        x.down_q = b.get(experts + "down_proj", CERE_DTYPE_U8, {E, H, I / 2});
        x.down_scale = b.get(experts + "down_proj_scale", CERE_DTYPE_F8_E4M3, {E, H, I / 16});
        x.down_global = b.get(experts + "down_proj_scale_2", CERE_DTYPE_F32, {E});
    } else {
        x.gate_up = b.get(experts + "gate_up_proj", CERE_DTYPE_BF16, {E, 2 * I, H});
        x.down = b.get(experts + "down_proj", CERE_DTYPE_BF16, {E, H, I});
    }
    x.shared_gate = b.get(layer + "mlp.shared_expert.gate_proj.weight", CERE_DTYPE_BF16, {S, H});
    x.shared_up = b.get(layer + "mlp.shared_expert.up_proj.weight", CERE_DTYPE_BF16, {S, H});
    x.shared_down = b.get(layer + "mlp.shared_expert.down_proj.weight", CERE_DTYPE_BF16, {H, S});
    x.shared_expert_gate = b.get(layer + "mlp.shared_expert_gate.weight", CERE_DTYPE_BF16, {1, H});
}

}  // namespace

uint64_t vk_tensor_addr(const VkLayout &layout, const std::vector<uint64_t> &block_addr, uint64_t i) {
    if (i + 1 >= layout.piece_first.size()) return 0;
    const uint64_t first = layout.piece_first[i];
    if (first == layout.piece_first[i + 1]) return 0;
    const VkPiece &p = layout.pieces[first];
    return p.block < block_addr.size() ? block_addr[p.block] + p.offset : 0;
}

uint64_t vk_region_addr(const VkLayout &layout, const std::vector<uint64_t> &block_addr, VkRegionKind kind,
                        uint32_t index) {
    const VkRegion *r = vk_region(layout, kind, index);
    return r == nullptr || r->block >= block_addr.size() ? 0 : block_addr[r->block] + r->offset;
}

cere_status vk_bind_weights(const host::ModelPlan &plan, const VkLayout &layout, const std::vector<uint64_t> &block_addr,
                            VkModelWeights *out, std::string *error) {
    try {
        if (out == nullptr || layout.status != CERE_OK || block_addr.size() != layout.blocks.size() ||
            layout.piece_first.size() != plan.tensors.size() + 1) {
            if (error != nullptr) *error = "vk_bind_weights: malformed call";
            return CERE_ERR_ARGUMENT;
        }
        const cere_qwen4_exp_config &cfg = plan.config;
        const uint64_t ple_bits = cfg.ple_layer_ids;
        if (cfg.num_hidden_layers == 0 || cfg.num_hidden_layers > kMaxLayers || ple_bits == 0 ||
            (ple_bits & (ple_bits - 1)) != 0 || (ple_bits >> 1) >= (uint64_t(1) << cfg.num_hidden_layers)) {
            if (error != nullptr) *error = "vk_bind_weights: config outside the bound shapes (layers, one PLE layer)";
            return CERE_ERR_UNSUPPORTED;
        }
        if (plan.levers.k2_weights != CERE_LEVER_K2_WEIGHTS_BF16) {
            if (error != nullptr) *error = "k2_weights = fp8 is not built on Vulkan";
            return CERE_ERR_UNSUPPORTED;
        }
        Binder b(plan, layout, block_addr, error);
        auto w = std::make_unique<VkModelWeights>();
        w->config = cfg;
        w->max_context_tokens = plan.max_context_tokens;
        const uint64_t H = cfg.hidden_size, V = cfg.vocab_size, E = cfg.num_experts, I = cfg.moe_intermediate_size,
                       S = cfg.shared_expert_intermediate_size, hc_width = uint64_t(cfg.hc_count) * H;
        const uint64_t key_dim = uint64_t(cfg.linear_num_key_heads) * cfg.linear_key_head_dim;
        const uint64_t value_dim = uint64_t(cfg.linear_num_value_heads) * cfg.linear_value_head_dim;
        const uint64_t conv_dim = 2 * key_dim + value_dim, vh = cfg.linear_num_value_heads;
        const std::string root = "model.language_model.";

        w->embed = b.get(root + "embed_tokens.weight", CERE_DTYPE_BF16, {V, H});
        w->lm_head = b.get("lm_head.weight", CERE_DTYPE_BF16, {V, H});
        w->final_mixer = bind_hc(b, root + "hyper_connection_mixer.", cfg, false);

        const uint64_t moe_info = vk_region_addr(layout, block_addr, VkRegionKind::kMoeInfo);
        for (uint32_t l = 0; l < cfg.num_hidden_layers && b.ok(); ++l) {
            VkLayerWeights &L = w->layers[l];
            const auto name = [l](const char *rest) { return layer_name(l, rest); };
            L.full_attention = layout::full_attention(cfg, l);
            L.attn_hc = bind_hc(b, name("attn_hyper_connection."), cfg, true);
            L.mlp_hc = bind_hc(b, name("mlp_hyper_connection."), cfg, true);
            if (L.full_attention) {
                bind_attention(b, std::string(kLayerPrefix) + std::to_string(l) + ".", cfg, &L.attn);
            } else {
                L.gdn_dense[0] = b.dense(name("linear_attn.in_proj_qkv"), conv_dim, H);
                L.gdn_dense[1] = b.dense(name("linear_attn.in_proj_z"), value_dim, H);
                L.gdn_dense[2] = b.dense(name("linear_attn.in_proj_b"), vh, H);
                L.gdn_dense[3] = b.dense(name("linear_attn.in_proj_a"), vh, H);
                L.gdn_dense[4] = b.dense(name("linear_attn.out_proj"), H, value_dim);
                vkk5::LayerWeights &g = L.gdn;
                g.in_proj_qkv = L.gdn_dense[0].w;
                g.in_proj_z = L.gdn_dense[1].w;
                g.in_proj_b = L.gdn_dense[2].w;
                g.in_proj_a = L.gdn_dense[3].w;
                g.out_proj = L.gdn_dense[4].w;
                g.conv1d = b.get(name("linear_attn.conv1d.weight"), CERE_DTYPE_BF16, {conv_dim, 1, cfg.linear_conv_kernel_dim});
                L.gdn_A_log_bf16 = b.get(name("linear_attn.A_log"), CERE_DTYPE_BF16, {vh});
                g.dt_bias = b.get(name("linear_attn.dt_bias"), CERE_DTYPE_BF16, {vh});
                g.norm = b.get(name("linear_attn.norm.weight"), CERE_DTYPE_BF16, {cfg.linear_value_head_dim});
                g.A_log = 0;  // derived, below
                g.info = 0;   // winfo, below
            }
            vkk7::MoeLayer &m = L.moe;
            m.router = b.get(name("mlp.gate.weight"), CERE_DTYPE_BF16, {E, H});
            L.shared_dense[0] = b.dense(name("mlp.shared_expert.gate_proj"), S, H);
            L.shared_dense[1] = b.dense(name("mlp.shared_expert.up_proj"), S, H);
            L.shared_dense[2] = b.dense(name("mlp.shared_expert.down_proj"), H, S);
            m.shared_gate = L.shared_dense[0].w;
            m.shared_up = L.shared_dense[1].w;
            m.shared_down = L.shared_dense[2].w;
            m.shared_expert_gate = b.get(name("mlp.shared_expert_gate.weight"), CERE_DTYPE_BF16, {1, H});
            const auto expert = [&name](const char *proj, const char *suffix) {
                return [&name, proj, suffix](uint64_t e) {
                    return name("mlp.experts.") + std::to_string(e) + "." + proj + "_proj." + suffix;
                };
            };
            // The K7 build's runs (no w13; model.cpp:812-833).
            m.gate_weight = b.run(E, expert("gate", "weight"), CERE_DTYPE_U8, {I, H / 2});
            m.gate_scale = b.run(E, expert("gate", "weight_scale"), CERE_DTYPE_F8_E4M3, {I, H / 16});
            m.up_weight = b.run(E, expert("up", "weight"), CERE_DTYPE_U8, {I, H / 2});
            m.up_scale = b.run(E, expert("up", "weight_scale"), CERE_DTYPE_F8_E4M3, {I, H / 16});
            m.down_weight = b.run(E, expert("down", "weight"), CERE_DTYPE_U8, {H, I / 2});
            m.down_scale = b.run(E, expert("down", "weight_scale"), CERE_DTYPE_F8_E4M3, {H, I / 16});
            VkMoeScaleSources &s = L.moe_scales;
            s.gate_input_scale = b.run(E, expert("gate", "input_scale"), CERE_DTYPE_F32, {});
            s.up_input_scale = b.run(E, expert("up", "input_scale"), CERE_DTYPE_F32, {});
            s.down_input_scale = b.run(E, expert("down", "input_scale"), CERE_DTYPE_F32, {});
            s.gate_weight_scale_2 = b.run(E, expert("gate", "weight_scale_2"), CERE_DTYPE_F32, {});
            s.down_weight_scale_2 = b.run(E, expert("down", "weight_scale_2"), CERE_DTYPE_F32, {});
            b.run(E, expert("up", "weight_scale_2"), CERE_DTYPE_F32, {});  // unused, must be present and packed
            m.bf16_info = moe_info == 0 ? 0 : moe_info + uint64_t(l) * layout.moe_info_stride;
        }

        // PLE (model.cpp:836-915), the table over the split host-heap blocks (vk_ple_block_table).
        uint32_t ple_layer = 0;
        while (((ple_bits >> (ple_layer + 1)) & 1) == 0) ++ple_layer;  // ple_layer_ids is 1-based
        w->ple_layer = ple_layer;
        const auto ple = [ple_layer](const char *rest) { return layer_name(ple_layer, rest); };
        const uint64_t ngram_heads = uint64_t(cfg.ngram_size - 1) * cfg.heads_per_ngram;
        vkk4::Weights &p = w->ple;
        if (b.ok()) {
            const VkPleSplit &sp = layout.ple;
            const std::string shard0 = ple("ple.ple_embedding.ngram_embedding.shard_0.weight");
            if (!sp.present || sp.shards != cfg.split_ngram_parts) {
                b.fail("PLE table: " + std::to_string(sp.shards) + " shards in the layout, config split_ngram_parts " +
                       std::to_string(cfg.split_ngram_parts));
            } else if (sp.format != 1 || b.dtype_of(shard0) != CERE_DTYPE_U8) {
                // The S3 artifact's NVFP4 table; the FP8 (CUR) table's BF16 weight_scale is not bound here.
                b.fail("PLE table: only the NVFP4 table (U8 [rows, 80] shards + E4M3 scales) is bound on Vulkan");
            } else {
                for (uint64_t i = 0; i < cfg.split_ngram_parts && b.ok(); ++i) {
                    const std::string base = ple("ple.ple_embedding.ngram_embedding.shard_") + std::to_string(i);
                    b.get(base + ".weight", CERE_DTYPE_U8, {sp.shard_rows, vkk4::kNvfp4RowBytes},
                          CERE_PLACEMENT_HOST_PINNED);
                    b.get(base + ".weight_scale", CERE_DTYPE_F8_E4M3, {sp.shard_rows, vkk4::kNvfp4RowScales},
                          CERE_PLACEMENT_HOST_PINNED);
                }
                p.format = vkk4::TableFormat::kNvfp4;
                p.table.desc = vk_region_addr(layout, block_addr, VkRegionKind::kIoPleBlockTable);
                p.table.rows = int64_t(sp.rows);
                p.table.rows_per_block = sp.rows_per_block;
                p.table.blocks = sp.blocks;
                p.global_scale = b.get(ple("ple.ple_embedding.ngram_embedding.weight_scale_2"), CERE_DTYPE_F32, {1},
                                       CERE_PLACEMENT_HOST_PINNED);
            }
        }
        p.multipliers = b.get(ple("ple.ple_embedding.layer_multipliers"), CERE_DTYPE_I64, {cfg.ngram_size},
                              CERE_PLACEMENT_HOST_PINNED);
        p.head_vocab_sizes = b.get(ple("ple.ple_embedding.ngram_heads_vocab_sizes"), CERE_DTYPE_I64, {ngram_heads},
                                   CERE_PLACEMENT_HOST_PINNED);
        p.head_offsets = b.get(ple("ple.ple_embedding.ngram_heads_offsets"), CERE_DTYPE_I64, {ngram_heads},
                               CERE_PLACEMENT_HOST_PINNED);
        p.key_proj = b.get(ple("ple.key_proj.weight"), CERE_DTYPE_BF16, {hc_width, cfg.ple_embed_dim});
        p.value_proj = b.get(ple("ple.value_proj.weight"), CERE_DTYPE_BF16, {H, cfg.ple_embed_dim});
        p.norm_key = b.get(ple("ple.norm_key.weight"), CERE_DTYPE_BF16, {hc_width});
        p.norm_query = b.get(ple("ple.norm_query.weight"), CERE_DTYPE_BF16, {hc_width});
        p.norm_conv = b.get(ple("ple.norm_conv.weight"), CERE_DTYPE_BF16, {hc_width});
        p.conv_weight = b.get(ple("ple.conv1d.weight"), CERE_DTYPE_BF16, {hc_width, 1, cfg.ple_conv_kernel_size});
        p.eos_token_id = vkk4::kEosTokenId;
        p.eps = float(cfg.rms_norm_eps);
        if (!b.ok()) return CERE_ERR_UNSUPPORTED;

        if (plan.speculative_draft_tokens != 0) {
            if (plan.draft_token_map.empty()) {
                if (error != nullptr) *error = "vk_bind_weights: drafts without a draft token map";
                return CERE_ERR_ARGUMENT;
            }
            bind_mtp(b, cfg, &w->mtp);
            if (!b.ok()) return CERE_ERR_UNSUPPORTED;
            w->mtp.count = uint32_t(plan.draft_token_map.size());
            w->mtp.moe.bf16_info = moe_info == 0 ? 0 : moe_info + uint64_t(cfg.num_hidden_layers) * layout.moe_info_stride;
            w->has_draft = true;
        }
        w->mxfp8_tensors = b.mxfp8_count();

        // Derived (bind_derived, model.cpp:320-332; layout::carve_derived) and the MTP part (carve_mtp_derived).
        const uint64_t derived = vk_region_addr(layout, block_addr, VkRegionKind::kDerived);
        if (derived == 0) {
            if (error != nullptr) *error = "vk_bind_weights: the layout has no derived region";
            return CERE_ERR_ARGUMENT;
        }
        {
            layout::Carver c{kFakeBase};
            layout::carve_derived(cfg, plan.max_context_tokens, c, [&](Region r, uint32_t l, uint8_t *at) {
                const uint64_t a = derived + off_of(at);
                switch (r) {
                    case Region::kALog: w->layers[l].gdn.A_log = a; break;
                    case Region::kG1Alphas: w->layers[l].moe.g1_alphas = a; break;
                    case Region::kG2Alphas: w->layers[l].moe.g2_alphas = a; break;
                    case Region::kRope: w->rope_cos_sin = a; break;
                    default: break;
                }
            });
        }
        if (w->has_draft) {
            const uint64_t mtp = vk_region_addr(layout, block_addr, VkRegionKind::kMtpDerived);
            layout::Carver c{kFakeBase};
            layout::carve_mtp_derived(cfg, w->mtp.count, c, [&](Region r, uint32_t, uint8_t *at) {
                if (r == Region::kMtpLmHead) w->mtp.lm_head = mtp + off_of(at);
                if (r == Region::kMtpMap) w->mtp.map = mtp + off_of(at);
            });
        }

        // Prepared K2 row info (VkLayout::winfo): each entry's consumer view points at it.
        const uint64_t winfo = vk_region_addr(layout, block_addr, VkRegionKind::kWinfo);
        for (const VkWinfo &e : layout.winfo) {
            const uint64_t at = winfo + e.offset;
            const std::string &n = plan.tensors[e.tensor].name;
            const auto has = [&n](const char *s) { return n.find(s) != std::string::npos; };
            VkAttentionWeights *att = nullptr;
            if (n.rfind("mtp.", 0) == 0) {
                if (!w->has_draft) continue;
                if (has(".fc_embedding.")) w->mtp.fc_embedding.info = at;
                else if (has(".fc_hidden.")) w->mtp.fc_hidden.info = at;
                else att = &w->mtp.attn;
            } else {
                const size_t pos = n.find(kLayerPrefix);
                if (pos != 0) continue;
                const uint32_t l = uint32_t(std::stoul(n.substr(sizeof kLayerPrefix - 1)));
                if (l >= cfg.num_hidden_layers) continue;
                VkLayerWeights &L = w->layers[l];
                if (has(".ple.key_proj.")) w->ple.proj_info = at;
                else if (has(".linear_attn.in_proj_qkv.")) { L.gdn.info = at; L.gdn_dense[0].info = at; }
                else if (has(".linear_attn.in_proj_z.")) L.gdn_dense[1].info = at;
                else if (has(".linear_attn.in_proj_b.")) L.gdn_dense[2].info = at;
                else if (has(".linear_attn.in_proj_a.")) L.gdn_dense[3].info = at;
                else if (has(".linear_attn.out_proj.")) L.gdn_dense[4].info = at;
                else if (L.full_attention) att = &L.attn;
            }
            if (att != nullptr) {
                if (has(".self_attn.q_proj.")) att->qkv.info = at;
                else if (has(".self_attn.o_proj.")) att->o_proj.info = at;
                else if (has(".indexer.index_qk_proj.")) att->indexer_qk.info = at;
            }
        }
        *out = *w;
        return CERE_OK;
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    } catch (...) {
        if (error != nullptr) *error = "vk_bind_weights: unexpected exception";
        return CERE_ERR_INTERNAL;
    }
}

cere_status vk_bind_session(const host::ModelPlan &plan, const VkLayout &layout, const std::vector<uint64_t> &block_addr,
                            uint64_t slot_addr, uint64_t max_context_tokens, VkSessionState *out) {
    if (out == nullptr || slot_addr == 0 || max_context_tokens == 0 || max_context_tokens > plan.max_context_tokens) {
        return CERE_ERR_ARGUMENT;
    }
    const cere_qwen4_exp_config &cfg = plan.config;
    const bool draft = plan.speculative_draft_tokens != 0;
    VkSessionState s;
    s.slot = slot_addr;
    s.max_context_tokens = max_context_tokens;
    layout::Carver c{kFakeBase};
    layout::carve_session(
        cfg, max_context_tokens, c,
        [&](Region r, uint32_t l, uint8_t *at) {
            const uint64_t a = slot_addr + off_of(at);
            switch (r) {
                case Region::kSlotMap: s.slot_map = a; break;
                case Region::kGdnConv: s.gdn[l].conv = a; break;
                case Region::kGdnSsm: s.gdn[l].ssm = a; break;
                case Region::kPleHistory: s.ple.history = a; break;
                case Region::kPleConv: s.ple.conv_state = a; break;
                case Region::kQsaRing: s.qsa_ring[l] = a; break;
                case Region::kOutCounts: s.out_counts = a; break;
                case Region::kMtpQsaRing: s.mtp_qsa_ring = a; break;
                case Region::kProposals: s.proposals = a; break;
                default: break;
            }
        },
        draft);
    s.slot_bytes = c.size();
    if (s.slot_bytes > layout.session_bytes) return CERE_ERR_ARGUMENT;
    for (uint32_t l = 0; l < kMaxLayers && l < cfg.num_hidden_layers; ++l) {
        if (!layout::full_attention(cfg, l)) continue;
        s.k_cache[l] = vk_region_addr(layout, block_addr, VkRegionKind::kKCache, l);
        s.v_cache[l] = vk_region_addr(layout, block_addr, VkRegionKind::kVCache, l);
        s.qsa_keys[l] = vk_region_addr(layout, block_addr, VkRegionKind::kQsaKeys, l);
        if (s.k_cache[l] == 0 || s.v_cache[l] == 0 || s.qsa_keys[l] == 0) return CERE_ERR_ARGUMENT;
    }
    if (draft) {
        s.mtp_k_cache = vk_region_addr(layout, block_addr, VkRegionKind::kMtpKCache);
        s.mtp_v_cache = vk_region_addr(layout, block_addr, VkRegionKind::kMtpVCache);
        s.mtp_qsa_keys = vk_region_addr(layout, block_addr, VkRegionKind::kMtpQsaKeys);
    }
    *out = s;
    return CERE_OK;
}

cere_status vk_bind_fixed_state(const host::ModelPlan &plan, uint64_t state_addr, uint64_t max_context_tokens,
                                VkSessionState *out) {
    if (out == nullptr || state_addr == 0 || max_context_tokens == 0) return CERE_ERR_ARGUMENT;
    const cere_qwen4_exp_config &cfg = plan.config;
    const layout::FixedState f = layout::fixed_state(cfg, 1);
    if (!f.contiguous || state_addr < f.begin) return CERE_ERR_UNSUPPORTED;
    // model.cpp:1076-1105 bind_fixed_state: carved as a one-token slot whose run starts at `state_addr`; the regions
    // outside [state_addr, state_addr + f.bytes) (the slot map, out_counts) are dropped.
    const uint64_t base = state_addr - f.begin;
    VkSessionState s;
    s.slot = state_addr;
    s.slot_bytes = f.bytes;
    s.max_context_tokens = max_context_tokens;
    layout::Carver c{kFakeBase};
    layout::carve_session(cfg, 1, c, [&](Region r, uint32_t l, uint8_t *at) {
        const uint64_t a = base + off_of(at);
        switch (r) {
            case Region::kGdnConv: s.gdn[l].conv = a; break;
            case Region::kGdnSsm: s.gdn[l].ssm = a; break;
            case Region::kPleHistory: s.ple.history = a; break;
            case Region::kPleConv: s.ple.conv_state = a; break;
            case Region::kQsaRing: s.qsa_ring[l] = a; break;
            default: break;
        }
    });
    *out = s;
    return CERE_OK;
}

cere_status vk_bind_undo(const host::ModelPlan &plan, uint64_t base, uint64_t bytes, VkUndoState *out) {
    if (out == nullptr || base == 0) return CERE_ERR_ARGUMENT;
    VkUndoState u;
    layout::Carver c{kFakeBase};
    layout::carve_undo(plan.config, c, [&](Region r, uint32_t l, uint8_t *at) {
        const uint64_t a = base + off_of(at);
        switch (r) {
            case Region::kGdnConv: u.gdn[l].conv = a; break;
            case Region::kGdnSsm: u.gdn[l].ssm = a; break;
            case Region::kPleHistory: u.ple.history = a; break;
            case Region::kPleConv: u.ple.conv_state = a; break;
            case Region::kQsaRing: u.qsa_ring[l] = a; break;
            case Region::kUndoKeyRow: u.qsa_key[l] = a; break;
            default: break;
        }
    });
    if (c.size() > bytes) return CERE_ERR_ARGUMENT;
    *out = u;
    return CERE_OK;
}

cere_status vk_bind_workspace(const host::ModelPlan &plan, const VkLayout &layout,
                              const std::vector<uint64_t> &block_addr, VkWorkspace *out) {
    if (out == nullptr) return CERE_ERR_ARGUMENT;
    const uint64_t rows = vk_region_addr(layout, block_addr, VkRegionKind::kWorkspaceRows);
    if (rows == 0) return CERE_ERR_ARGUMENT;
    VkWorkspace w;
    w.rows = layout.workspace_rows;
    layout::Carver c{kFakeBase};
    layout::carve_workspace_rows(plan.config, layout.workspace_rows, c, [&](Region r, uint32_t, uint8_t *at) {
        const uint64_t a = rows + off_of(at);
        switch (r) {
            case Region::kHyper: w.hyper = a; break;
            case Region::kNormed: w.normed = a; break;
            case Region::kMixed: w.mixed = a; break;
            case Region::kBlockOut: w.block_out = a; break;
            case Region::kQkv: w.qkv = a; break;
            case Region::kAttnOut: w.attn_out = a; break;
            case Region::kTapRouted: w.tap_routed = a; break;
            case Region::kTapShared: w.tap_shared = a; break;
            case Region::kIds32: w.ids32 = a; break;
            case Region::kIds64: w.ids64 = a; break;
            case Region::kLogits: w.logits = a; break;
            case Region::kProcLogits: w.proc_logits = a; break;
            case Region::kTop2: w.top2 = a; break;
            default: break;
        }
    });
    w.scratch = vk_region_addr(layout, block_addr, VkRegionKind::kScratch);
    w.scratch_bytes = layout.scratch_bytes;
    w.gemm_ws = vk_region_addr(layout, block_addr, VkRegionKind::kGemmWorkspace);
    w.gemm_ws_bytes = layout.gemm_workspace_bytes;
    w.mtp_hc_rows = vk_region_addr(layout, block_addr, VkRegionKind::kMtpHcRows);
    *out = w;
    return CERE_OK;
}

}  // namespace cere::vkdev
