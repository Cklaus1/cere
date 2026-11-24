// FL-30 (track Fp4S3, fp4local design note "S3 perf plan"): the CUDA-free MXFP8 plan of model/mxfp8_plan.h (the
// staging bytes the memory plan carves before ALLOCATE, and the group checks bind_weights refuses with) on synthetic
// host plans with the A4 config's names and shapes: the S2 layout (every dense projection BF16: nothing changes), the
// S3 layout (the 384 target projections MXFP8: 360 GEMMs, staging = attention q|k|v's BF16 upcast), partial MXFP8,
// and every refusal (mixed formats in q|k|v, the GDN in-projections and the shared expert; a scale of the wrong dtype,
// shape or placement; a missing scale; a scale next to a BF16 weight; k not a multiple of 32). No tensor bytes.
#include "check.h"

#include "host/device.h"
#include "host/support.h"
#include "model/layout.h"
#include "model/mxfp8_plan.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

namespace mx = cere::model::mxfp8;
using cere::host::ModelPlan;
using cere::host::TensorPlan;

constexpr uint64_t kAttnQkvStaging = 13312ull * 2560 * 2;  // 68,157,440 B: q|k|v [13,312, 2,560] as BF16

void add(ModelPlan *p, const std::string &name, uint32_t dtype, std::vector<uint64_t> shape,
         uint32_t placement = CERE_PLACEMENT_DEVICE) {
    TensorPlan t{};
    t.name = name;
    t.dtype = dtype;
    t.placement = placement;
    t.ndim = uint32_t(shape.size());
    uint64_t elements = 1;
    for (size_t d = 0; d < shape.size(); ++d) {
        t.shape[d] = shape[d];
        elements *= shape[d];
    }
    t.bytes = elements * cere::host::dtype_size(dtype);
    p->tensors.push_back(t);
}

TensorPlan *find(ModelPlan *p, const std::string &name) {
    for (TensorPlan &t : p->tensors) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

void remove(ModelPlan *p, const std::string &name) {
    for (size_t i = 0; i < p->tensors.size(); ++i) {
        if (p->tensors[i].name == name) {
            p->tensors.erase(p->tensors.begin() + long(i));
            return;
        }
    }
}

// One dense projection `prefix` [n, k]: BF16, or MXFP8 (E4M3 weight + UE8M0 scale).
void dense(ModelPlan *p, const std::string &prefix, uint64_t n, uint64_t k, bool mxfp8) {
    if (!mxfp8) {
        add(p, prefix + ".weight", CERE_DTYPE_BF16, {n, k});
        return;
    }
    add(p, prefix + ".weight", CERE_DTYPE_F8_E4M3, {n, k});
    add(p, prefix + ".weight_scale", CERE_DTYPE_U8, {n, k / 32});
}

// The dense projections of the A4 target layers (plus a few non-candidates: router, lm_head, the MTP layer's q_proj),
// `mxfp8(prefix)` choosing each one's format.
template <typename Choose>
ModelPlan make_plan(Choose mxfp8) {
    ModelPlan p{};
    p.support = &cere::host::a4_support();
    p.config = p.support->config;
    const cere_qwen4_exp_config &c = p.config;
    const uint64_t H = c.hidden_size;
    const uint64_t q_rows = uint64_t(c.num_attention_heads) * 2 * c.head_dim;
    const uint64_t kv_rows = uint64_t(c.num_key_value_heads) * c.head_dim;
    const uint64_t attn_out = uint64_t(c.num_attention_heads) * c.head_dim;
    const uint64_t idx_rows = (uint64_t(c.indexer_n_heads) + c.indexer_kv_heads) * c.indexer_head_dim;
    const uint64_t key_dim = uint64_t(c.linear_num_key_heads) * c.linear_key_head_dim;
    const uint64_t value_dim = uint64_t(c.linear_num_value_heads) * c.linear_value_head_dim;
    const uint64_t S = c.shared_expert_intermediate_size;
    add(&p, "lm_head.weight", CERE_DTYPE_BF16, {c.vocab_size, H});
    for (uint32_t l = 0; l < c.num_hidden_layers; ++l) {
        const std::string P = mx::layer_prefix(l);
        const auto proj = [&](const char *rest, uint64_t n, uint64_t k) {
            dense(&p, P + rest, n, k, mxfp8(P + rest));
        };
        if (cere::model::layout::full_attention(c, l)) {
            proj("self_attn.q_proj", q_rows, H);
            proj("self_attn.k_proj", kv_rows, H);
            proj("self_attn.v_proj", kv_rows, H);
            proj("self_attn.o_proj", H, attn_out);
            proj("self_attn.indexer.index_qk_proj", idx_rows, H);
        } else {
            proj("linear_attn.in_proj_qkv", 2 * key_dim + value_dim, H);
            proj("linear_attn.in_proj_z", value_dim, H);
            proj("linear_attn.in_proj_b", c.linear_num_value_heads, H);
            proj("linear_attn.in_proj_a", c.linear_num_value_heads, H);
            proj("linear_attn.out_proj", H, value_dim);
        }
        add(&p, P + "mlp.gate.weight", CERE_DTYPE_BF16, {c.num_experts, H});
        proj("mlp.shared_expert.gate_proj", S, H);
        proj("mlp.shared_expert.up_proj", S, H);
        proj("mlp.shared_expert.down_proj", H, S);
    }
    add(&p, "mtp.layers.0.self_attn.q_proj.weight", CERE_DTYPE_BF16, {q_rows, H});
    return p;
}

ModelPlan s2_plan() {
    return make_plan([](const std::string &) { return false; });
}
ModelPlan s3_plan() {
    return make_plan([](const std::string &) { return true; });
}

bool contains(const std::string &s, const std::string &part) { return s.find(part) != std::string::npos; }

// The first full-attention and the first GDN layer of the A4 config.
std::string fa0() { return mx::layer_prefix(3); }
std::string gdn0() { return mx::layer_prefix(0); }

}  // namespace

int main() {
    CHECK(cere::model::layout::full_attention(cere::host::a4_support().config, 3));
    CHECK(!cere::model::layout::full_attention(cere::host::a4_support().config, 0));

    // S2 / CUR: every dense projection BF16; the plan is unchanged (no staging).
    {
        const mx::Plan r = mx::plan_of(s2_plan());
        CHECK(r.error.empty());
        CHECK_EQ(r.tensors, 0);
        CHECK_EQ(r.gemms, 0);
        CHECK_EQ(r.staging_bytes, 0);
    }
    // S3: the 384 target projections MXFP8 (36 GDN layers x 5, 12 attention layers x 5, 48 shared x 3); q|k|v one GEMM
    // (360 table entries); staging = q|k|v's BF16 upcast, the largest.
    {
        const mx::Plan r = mx::plan_of(s3_plan());
        CHECK(r.error.empty());
        CHECK_EQ(r.tensors, 384);
        CHECK_EQ(r.gemms, 360);
        CHECK_EQ(r.staging_bytes, kAttnQkvStaging);
        CHECK_EQ(r.staging_bytes, 68157440);
        CHECK_EQ(r.staging_bytes % cere::model::layout::kAlign, 0);
    }
    // Partial: only the attention o_proj MXFP8 (12 GEMMs): staging 2 x 2,560 x 6,144.
    {
        const mx::Plan r = mx::plan_of(make_plan([](const std::string &p) { return contains(p, "o_proj"); }));
        CHECK(r.error.empty());
        CHECK_EQ(r.tensors, 12);
        CHECK_EQ(r.gemms, 12);
        CHECK_EQ(r.staging_bytes, 2ull * 2560 * 6144);
    }
    // Partial: only the GDN in-projections (one group per layer, uniform): staging = in_proj_qkv's 2 x 10,240 x 2,560.
    {
        const mx::Plan r = mx::plan_of(make_plan([](const std::string &p) { return contains(p, "in_proj"); }));
        CHECK(r.error.empty());
        CHECK_EQ(r.tensors, 144);
        CHECK_EQ(r.gemms, 144);
        CHECK_EQ(r.staging_bytes, 2ull * 10240 * 2560);
    }
    // MXFP8 tensors outside the 384 (the MTP layer stays BF16 by contract) are not candidates: the binder's BF16
    // check refuses them by name.
    {
        ModelPlan p = s2_plan();
        remove(&p, "mtp.layers.0.self_attn.q_proj.weight");
        dense(&p, "mtp.layers.0.self_attn.q_proj", 8192, 2560, true);
        const mx::Plan r = mx::plan_of(p);
        CHECK(r.error.empty());
        CHECK_EQ(r.tensors, 0);
        CHECK_EQ(r.staging_bytes, 0);
    }

    // Refusals: mixed formats inside a group the forward runs as one unit.
    const auto mixed = [](const std::string &bf16_prefix) {
        return mx::plan_of(make_plan([bf16_prefix](const std::string &p) { return p != bf16_prefix; }));
    };
    {
        const mx::Plan r = mixed(fa0() + "self_attn.k_proj");
        CHECK(contains(r.error, "mixed") && contains(r.error, fa0() + "self_attn.q_proj.weight"));
    }
    {
        const mx::Plan r = mixed(gdn0() + "linear_attn.in_proj_a");
        CHECK(contains(r.error, "mixed") && contains(r.error, gdn0() + "linear_attn.in_proj_z.weight"));
    }
    {
        const mx::Plan r = mixed(gdn0() + "mlp.shared_expert.down_proj");
        CHECK(contains(r.error, "mixed") && contains(r.error, "mlp.shared_expert.gate_proj.weight"));
    }
    // A BF16 o_proj next to MXFP8 q|k|v is not a group: valid.
    CHECK(mixed(fa0() + "self_attn.o_proj").error.empty());
    CHECK(mixed(gdn0() + "linear_attn.out_proj").error.empty());

    // Refusals: the scale tensor of an MXFP8 weight.
    const std::string o = fa0() + "self_attn.o_proj";
    {
        ModelPlan p = s3_plan();
        find(&p, o + ".weight_scale")->dtype = CERE_DTYPE_F8_E4M3;  // an NVFP4-style E4M3 block scale
        const mx::Plan r = mx::plan_of(p);
        CHECK(contains(r.error, o + ".weight_scale") && contains(r.error, "expected U8"));
    }
    {
        ModelPlan p = s3_plan();
        find(&p, o + ".weight_scale")->shape[1] = 6144 / 16;  // 16-element blocks
        const mx::Plan r = mx::plan_of(p);
        CHECK(contains(r.error, o + ".weight_scale") && contains(r.error, "[2560, 192]"));
    }
    {
        ModelPlan p = s3_plan();
        find(&p, o + ".weight_scale")->shape[0] = 2559;
        CHECK(contains(mx::plan_of(p).error, o + ".weight_scale"));
    }
    {
        ModelPlan p = s3_plan();
        find(&p, o + ".weight_scale")->placement = CERE_PLACEMENT_HOST_PINNED;
        CHECK(contains(mx::plan_of(p).error, o + ".weight_scale"));
    }
    {
        ModelPlan p = s3_plan();
        remove(&p, o + ".weight_scale");
        const mx::Plan r = mx::plan_of(p);
        CHECK(contains(r.error, "missing tensor " + o + ".weight_scale"));
    }
    {
        ModelPlan p = s2_plan();
        add(&p, o + ".weight_scale", CERE_DTYPE_U8, {2560, 6144 / 32});  // a scale next to a BF16 weight
        CHECK(contains(mx::plan_of(p).error, o + ".weight_scale"));
    }
    {
        ModelPlan p = s3_plan();
        TensorPlan *w = find(&p, o + ".weight");
        w->shape[1] = 6144 + 16;  // k not a multiple of 32
        CHECK(contains(mx::plan_of(p).error, o + ".weight") && contains(mx::plan_of(p).error, "multiple of 32"));
    }
    {
        ModelPlan p = s3_plan();
        find(&p, o + ".weight")->ndim = 3;
        CHECK(contains(mx::plan_of(p).error, o + ".weight"));
    }

    return cere::test::finish("test_mxfp8_plan");
}
