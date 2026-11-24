#include "host/support.h"

#include <cstring>

namespace cere::host {

namespace {

cere_qwen4_exp_config make_a4_config() {
    cere_qwen4_exp_config c{};
    c.struct_size = sizeof(c);
    c.abi_version = CERE_EXECUTOR_ABI_VERSION;
    c.vocab_size = 248320;
    c.hidden_size = 2560;
    c.num_hidden_layers = 48;
    c.full_attention_interval = 4;
    for (unsigned layer = 3; layer < 48; layer += 4) c.full_attention_layers |= uint64_t(1) << layer;
    c.num_attention_heads = 24;
    c.num_key_value_heads = 2;
    c.head_dim = 256;
    c.output_gate_type = CERE_OUTPUT_GATE_SIGMOID;
    c.rope_theta = 10000000.0;
    c.partial_rotary_factor = 0.25;
    c.mrope_interleaved = 1;
    c.mrope_section[0] = 11;
    c.mrope_section[1] = 11;
    c.mrope_section[2] = 10;
    c.indexer_n_heads = 4;
    c.indexer_head_dim = 128;
    c.indexer_kv_heads = 1;
    c.indexer_compress_ratio = 4;
    c.indexer_budget = 2048;
    c.linear_num_key_heads = 16;
    c.linear_num_value_heads = 48;
    c.linear_key_head_dim = 128;
    c.linear_value_head_dim = 128;
    c.linear_conv_kernel_dim = 4;
    c.mamba_ssm_dtype = CERE_DTYPE_F32;
    c.num_experts = 512;
    c.num_experts_per_tok = 10;
    c.moe_intermediate_size = 640;
    c.shared_expert_intermediate_size = 640;
    c.norm_topk_prob = 1;  // absent from config.json
    c.hc_count = 4;
    c.hc_lowrank = 320;
    c.ple_layer_ids = uint64_t(1) << 2;
    c.ple_embed_dim = 2560;
    c.heads_per_ngram = 8;
    c.ngram_size = 3;
    c.ple_conv_kernel_size = 4;
    c.ngram_vocab_size_base = 20000000;
    c.split_ngram_parts = 128;
    c.make_ngram_vocab_size_divisible_by = 128;
    c.rms_norm_eps = 1e-6;
    c.tie_word_embeddings = 0;
    c.mtp_num_hidden_layers = 1;
    c.max_position_embeddings = 262144;
    return c;
}

// $ART/config.json vision_config; layer_norm_eps is SGLang's (qwen3_vl.py:196, nn.LayerNorm eps=1e-6).
cere_qwen4_exp_vision_config make_a4_vision_config() {
    cere_qwen4_exp_vision_config v{};
    v.struct_size = sizeof(v);
    v.abi_version = CERE_EXECUTOR_ABI_VERSION;
    v.depth = 27;
    v.hidden_size = 1152;
    v.intermediate_size = 4304;
    v.num_heads = 16;
    v.patch_size = 16;
    v.temporal_patch_size = 2;
    v.spatial_merge_size = 2;
    v.in_channels = 3;
    v.out_hidden_size = 2560;
    v.num_position_embeddings = 2304;
    v.deepstack_count = 0;
    v.hidden_act = CERE_ACT_GELU_TANH;
    v.image_token_id = 248056;
    v.video_token_id = 248057;
    v.vision_start_token_id = 248053;
    v.vision_end_token_id = 248054;
    v.layer_norm_eps = 1e-6;
    return v;
}

constexpr uint32_t kA4Dtypes = dtype_bit(CERE_DTYPE_BF16) | dtype_bit(CERE_DTYPE_F32) |
                               dtype_bit(CERE_DTYPE_F8_E4M3) | dtype_bit(CERE_DTYPE_U8) | dtype_bit(CERE_DTYPE_I64);

}  // namespace

const Support &a4_support() {
    static const Support support{
        "qwen4_exp_nvfp4_a4",
        make_a4_config(),
        16,      // max_sessions (m23 design note P8; m26 D3 was 4): the n <= 16 rule, GemmPlan::kMaxRows rows per call
        262144,  // max_context_tokens (ctx262 design note CX-01: max_position_embeddings; m25 D6 was 200,000)
        1048576, // max_kv_pool_tokens (CX-01: 16,384 pages, a validation bound; the memory estimate refuses a pool
                 // that does not fit; Headroom's 300,032 before)
        kA4Dtypes,
        nullptr,
    };
    return support;
}

const Support &a4_vision_support() {
    static const cere_qwen4_exp_vision_config vision = make_a4_vision_config();
    // m213l design note VL-L1: a4_support's values (config, max_sessions, max_context_tokens, max_kv_pool_tokens,
    // dtypes) plus the vision tower, so the two entries never drift apart.
    static const Support support = [] {
        Support s = a4_support();
        s.name = "qwen4_exp_nvfp4_a4_vision";
        s.vision = &vision;
        return s;
    }();
    return support;
}

bool vision_matches(const Support &support, const cere_qwen4_exp_vision_config *vision) {
    if (support.vision == nullptr || vision == nullptr) return support.vision == vision;
    // 18 u32 then one double: no padding (ledger N), so a byte compare is a field compare.
    static_assert(sizeof(cere_qwen4_exp_vision_config) == 80, "cere_qwen4_exp_vision_config layout");
    constexpr size_t header = offsetof(cere_qwen4_exp_vision_config, depth);
    return std::memcmp(reinterpret_cast<const unsigned char *>(support.vision) + header,
                       reinterpret_cast<const unsigned char *>(vision) + header, sizeof(*vision) - header) == 0;
}

uint64_t patch_elements(const cere_qwen4_exp_vision_config &vision) {
    return uint64_t(vision.in_channels) * vision.temporal_patch_size * vision.patch_size * vision.patch_size;
}

bool config_matches(const Support &support, const cere_qwen4_exp_config &config) {
    // The struct has no padding (tests/fixtures/executor_abi_v5_layout.json), so a byte compare is a field compare.
    static_assert(sizeof(cere_qwen4_exp_config) == 216, "cere_qwen4_exp_config layout");
    constexpr size_t header = offsetof(cere_qwen4_exp_config, vocab_size);
    return std::memcmp(reinterpret_cast<const unsigned char *>(&support.config) + header,
                       reinterpret_cast<const unsigned char *>(&config) + header, sizeof(config) - header) == 0;
}

uint64_t dtype_size(uint32_t dtype) {
    switch (dtype) {
        case CERE_DTYPE_F64:
        case CERE_DTYPE_I64:
            return 8;
        case CERE_DTYPE_F32:
        case CERE_DTYPE_I32:
            return 4;
        case CERE_DTYPE_BF16:
        case CERE_DTYPE_F16:
        case CERE_DTYPE_I16:
            return 2;
        case CERE_DTYPE_F8_E4M3:
        case CERE_DTYPE_F8_E5M2:
        case CERE_DTYPE_U8:
        case CERE_DTYPE_I8:
        case CERE_DTYPE_BOOL:
            return 1;
        default:
            return 0;
    }
}

}  // namespace cere::host
