// VulkanAmd P3 T2: the pipeline warm-up list of INSTANTIATE (design 3.6 step 1, T1 report section 11): the union of
// every kernel family's *_pipeline_specs list, so no pipeline is compiled after the model is sealed.
//
// The families' lists were being added while this was written; build_lib.sh greps each kernel header for the exact
// declaration it calls here and defines CERE_VKW_<FAMILY> in the generated vk_warm_have.h only when it is present, so
// a missing list is skipped (and named in InstantiateReport::warm_lists) instead of breaking the build.
#include "device/vulkan_warm.h"

#include "vk_warm_have.h"

#include "kernels/k12_sampler_vk.h"
#include "kernels/k13_moe_bf16_vk.h"
#include "kernels/k13s_spec_accept_vk.h"
#include "kernels/k1_primitives_vk.h"
#include "kernels/k2_gemm_vk.h"
#include "kernels/k2_mgemv_vk.h"
#include "kernels/k3_hc_vk.h"
#include "kernels/k4_ple_vk.h"
#include "kernels/k5_gdn_chain_vk.h"
#include "kernels/k5_gdn_layer_vk.h"
#include "kernels/k6_attention_vk.h"
#include "kernels/k6_prologue_vk.h"
#include "kernels/k7_moe_vk.h"
#include "kernels/k9_qsa_vk.h"
#include "kernels/mtp_kernels_vk.h"

namespace cere::vkdev {

std::vector<vk::PipelineSpec> vk_warm_specs(const cere_qwen4_exp_config &c, bool drafts, std::string *used, bool mx,
                                            bool coopmat) {
    std::vector<vk::PipelineSpec> s;
    std::string names, missing;
    const auto add = [&](const char *name, const std::vector<vk::PipelineSpec> &list) {
        vkk2::append_unique(s, list);
        names += std::string(names.empty() ? "" : ",") + name + "(" + std::to_string(list.size()) + ")";
    };
    const auto skip = [&](const char *name) { missing += std::string(missing.empty() ? "" : ",") + name; };
    const uint32_t H = c.hidden_size;
    const vkk7::MoeDims moe{c.hidden_size, c.num_experts, c.num_experts_per_tok, c.moe_intermediate_size,
                            c.shared_expert_intermediate_size};
    (void)H;
    (void)moe;
    (void)drafts;
    (void)mx;
    (void)skip;
    (void)coopmat;
#if defined(CERE_VKW_K1)
    add("k1", vkk1::pipeline_specs({int(c.hidden_size), int(c.linear_value_head_dim), int(c.indexer_head_dim)},
                                   {int(c.head_dim)}));
#else
    skip("k1");
#endif
#if defined(CERE_VKW_K2G)
    add("k2_gemm", vkk2::pipeline_specs(true, true));
#else
    skip("k2_gemm");
#endif
#if defined(CERE_VKW_K2V)
    // BF16 k: hc lowrank (K3 up), shared inter (K7/K13 down), hidden, GDN out, hc*hidden (K3 down); MXFP8 k: the S3
    // artifact's dense projections (hidden, 6,144 = attention / GDN out, 640 = shared-expert down).
    add("k2_gemv", vkk2::pipeline_specs({c.hc_lowrank, c.shared_expert_intermediate_size, H,
                                         c.linear_num_value_heads * c.linear_value_head_dim, c.hc_count * H},
                                        {H, c.num_attention_heads * c.head_dim, c.shared_expert_intermediate_size},
                                        false));
#else
    skip("k2_gemv");
#endif
#if defined(CERE_VKW_K3)
    add("k3", vkk3::pipeline_specs(vkk3::HcShape{c.hc_count, H, c.hc_lowrank, float(c.rms_norm_eps)}, true));
#else
    skip("k3");
#endif
#if defined(CERE_VKW_K4)
    add("k4", vkk4::pipeline_specs(false, true, true));
#else
    skip("k4");
#endif
#if defined(CERE_VKW_K5L)
    add("k5_layer", vkk5::pipeline_specs(std::vector<uint32_t>{H}, true, mx));
#else
    skip("k5_layer");
#endif
#if defined(CERE_VKW_K5C)
    add("k5_chain", vkk5::pipeline_specs());
#else
    skip("k5_chain");
#endif
#if defined(CERE_VKW_K6)
    // Both K/V formats: the first chunk (prefix 0) reads the call's BF16 K/V (vulkan_forward.cpp attention_rows).
    add("k6", vkk6::pipeline_specs(vkk6::Options{}, true, true));
    if (coopmat) add("k6_coopmat", vkk6::pipeline_specs(vkk6::Options{64, 64, vkk6::Prefill::kCoopmat}, true, true));
#else
    skip("k6");
#endif
#if defined(CERE_VKW_K6P)
    add("k6_prologue", vkk6::pipeline_specs());
#else
    skip("k6_prologue");
#endif
#if defined(CERE_VKW_K7)
    add("k7", vkk7::pipeline_specs(moe, true, true, false, mx));
#else
    skip("k7");
#endif
#if defined(CERE_VKW_K9)
    add("k9", vkk9::pipeline_specs(vkk9::Options{}, true, true, {vkk9::kRowsHeads}));
    if (coopmat) {
        add("k9_coopmat",
            vkk9::pipeline_specs(vkk9::Options{64, vkk9::Prefill::kCoopmat, true}, true, true, {vkk9::kRowsHeads}));
    }
#else
    skip("k9");
#endif
#if defined(CERE_VKW_K12)
    add("k12", vkk12::pipeline_specs(int64_t(c.vocab_size)));
#else
    skip("k12");
#endif
    if (drafts) {
#if defined(CERE_VKW_K13)
        add("k13", vkk13::pipeline_specs(moe, false, true, true));
#else
        skip("k13");
#endif
#if defined(CERE_VKW_K13S)
        add("k13s", vkk13s::pipeline_specs());
#else
        skip("k13s");
#endif
#if defined(CERE_VKW_MTP)
        add("mtp", vkmtp::pipeline_specs());
#else
        skip("mtp");
#endif
    }
    if (used != nullptr) *used = names + (missing.empty() ? "" : " missing:" + missing);
    return s;
}

}  // namespace cere::vkdev
