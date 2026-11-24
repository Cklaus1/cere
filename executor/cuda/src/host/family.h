#pragma once
// SelfTune ST4 S3a (selftune design note ST4): a library is built for one model, and the model's family
// (`gen::kFamily`, from its descriptor's model_type) decides which model descriptors it accepts. Header-only and
// CUDA-free: api.cpp compares a descriptor with its own library's family, and the host tests call these functions
// for every family, including those this build is not compiled for.

#include <cstdint>
#include <cstring>

#include "cere_executor.h"

namespace cere::host {

// The weight format (cere_model_desc.model_kind) of a family's checkpoints; 0 for a family the ABI does not define.
constexpr uint32_t family_model_kind(uint32_t family) {
    return family == CERE_FAMILY_QWEN4_EXP ? CERE_MODEL_QWEN4_EXP_NVFP4
           : family == CERE_FAMILY_SDAR    ? CERE_MODEL_SDAR_BF16
           : family == CERE_FAMILY_INKLING2 ? CERE_MODEL_INKLING2_NVFP4
                                           : 0;
}

// Why a library of `library_family` refuses a descriptor's family (CERE_ERR_UNSUPPORTED), by field name; null when
// the descriptor names the library's own family and that family's model_kind. A Qwen descriptor on an SDAR library
// and an SDAR descriptor on a Qwen library are both refused here, before the config is read.
constexpr const char *family_refusal(const cere_model_desc &desc, uint32_t library_family) {
    if (desc.model_kind != family_model_kind(library_family)) return "model_kind";
    if (desc.family != library_family) return "family";
    return nullptr;
}

// What an SDAR library refuses in a slot selection, by name (CERE_ERR_UNSUPPORTED); null when every choice is the
// registry default (crates/cere-runtime/src/levers.rs: what LeverSet::ffi sends for the default set, the only levers
// `cered` admits on family sdar, LeverSet::check_target): impl 0 in every slot but moe_gemm, whose default is p3b,
// and the draft window 0 of a model without drafts. Another choice names `<slot>.<impl>`; another draft window names
// the slot. An unknown slot or impl is left to resolve_selection (CERE_ERR_ARGUMENT).
constexpr const char *sdar_selection_refusal(const cere_slot_selection *selection) {
    if (selection == nullptr) return nullptr;
    for (uint32_t i = 0; i < selection->count && i < CERE_SLOT_MAX; ++i) {
        const cere_slot_choice &c = selection->choices[i];
        switch (c.slot) {
        case CERE_SLOT_K2_GEMM:
            if (c.impl == CERE_LEVER_K2_SMALL_M) return "k2_gemm.small_m";
            if (c.impl == CERE_LEVER_K2_SMALL_M_A8) return "k2_gemm.small_m_a8";
            if (c.impl == CERE_LEVER_K2_TAIL_SPLIT) return "k2_gemm.tail_split";
            break;
        case CERE_SLOT_PREFILL_ATTENTION:
            if (c.impl == CERE_LEVER_PREFILL_P4) return "prefill_attention.p4";
            break;
        case CERE_SLOT_DECODE_STEP:
            if (c.impl == CERE_LEVER_DECODE_GRAPH) return "decode_step.graph";
            if (c.impl == CERE_LEVER_DECODE_SPEC_GRAPH) return "decode_step.spec_graph";
            break;
        case CERE_SLOT_MOE_GEMM:
            if (c.impl == CERE_LEVER_MOE_K7) return "moe_gemm.k7";
            break;
        case CERE_SLOT_DRAFT_TOKENS:
            if (c.params[0] != 0) return "draft_tokens";
            break;
        case CERE_SLOT_VERIFY_ATTENTION:
            if (c.impl == CERE_LEVER_VERIFY_BATCHED) return "verify_attention.batched";
            break;
        case CERE_SLOT_K2_WEIGHTS:
            if (c.impl == CERE_LEVER_K2_WEIGHTS_FP8) return "k2_weights.fp8";
            break;
        default:
            break;
        }
    }
    return nullptr;
}

// What an SDAR library (DICE-8B: one session, eager, BF16 weights and BF16 KV, greedy, no draft head) refuses in a
// descriptor of its own family, by name (CERE_ERR_UNSUPPORTED); null when the descriptor is in scope. Out of scope by
// decision (ST4 plan, 2026-11-12): rows > 1, speculation, the prefix cache, snapshots, vision, MTP seal calibration
// and the selection refusals above. Header validity (struct_size, abi_version, block_version) is check_header's,
// before this; the config is sdar_config_refusal's, after it.
constexpr const char *sdar_refusal(const cere_model_desc &desc) {
    if (desc.max_sessions > 1) return "max_sessions";
    if (desc.speculative_draft_tokens != 0) return "speculative_draft_tokens";
    if (desc.draft_token_map != nullptr || desc.draft_token_map_count != 0) return "draft_token_map";
    if (desc.snapshot_host_bytes != 0) return "snapshot_host_bytes";
    if (desc.prefix_state_slots != 0) return "prefix_state_slots";
    if (desc.vision_config != nullptr) return "vision_config";
    if (desc.calibrate_windows != CERE_CALIBRATE_ALL) return "calibrate_windows";
    return sdar_selection_refusal(desc.selection);
}

// The leading fields every config struct shares (cere_qwen4_exp_config, cere_sdar_config).
struct ConfigHeader {
    uint32_t struct_size;
    uint32_t abi_version;
};

// A refusal: the field it names and the status it returns; field null for none.
struct Refusal {
    const char *field;
    cere_status status;
};

// A descriptor's `config` pointer, read before it is cast (ST4 D1: the pointee type follows `family`): whether its
// leading header is a `family` config of this ABI. The statuses are check_header's: a null pointer or a wrong size
// CERE_ERR_ARGUMENT, a wrong ABI version CERE_ERR_UNSUPPORTED.
inline Refusal config_refusal(const void *config, uint32_t family) {
    if (config == nullptr) return {"config", CERE_ERR_ARGUMENT};
    ConfigHeader h{};
    std::memcpy(&h, config, sizeof(h));
    const uint32_t want = family == CERE_FAMILY_SDAR       ? uint32_t(sizeof(cere_sdar_config))
                          : family == CERE_FAMILY_INKLING2 ? uint32_t(sizeof(cere_inkling2_config))
                                                           : uint32_t(sizeof(cere_qwen4_exp_config));
    if (h.struct_size != want) return {"config.struct_size", CERE_ERR_ARGUMENT};
    if (h.abi_version != CERE_EXECUTOR_ABI_VERSION) return {"config.abi_version", CERE_ERR_UNSUPPORTED};
    return {nullptr, CERE_OK};
}

// Why an SDAR config differs from the one the library was compiled for (`want`, from its descriptor), by field name;
// null when every value is equal (CERE_ERR_UNSUPPORTED otherwise, as for the Qwen support table).
constexpr const char *sdar_config_refusal(const cere_sdar_config &got, const cere_sdar_config &want) {
    if (got.vocab_size != want.vocab_size) return "config.vocab_size";
    if (got.hidden_size != want.hidden_size) return "config.hidden_size";
    if (got.intermediate_size != want.intermediate_size) return "config.intermediate_size";
    if (got.num_hidden_layers != want.num_hidden_layers) return "config.num_hidden_layers";
    if (got.num_attention_heads != want.num_attention_heads) return "config.num_attention_heads";
    if (got.num_key_value_heads != want.num_key_value_heads) return "config.num_key_value_heads";
    if (got.head_dim != want.head_dim) return "config.head_dim";
    if (got.block_size != want.block_size) return "config.block_size";
    if (!(got.rope_theta == want.rope_theta)) return "config.rope_theta";
    if (!(got.rms_norm_eps == want.rms_norm_eps)) return "config.rms_norm_eps";
    if (got.mask_token_id != want.mask_token_id) return "config.mask_token_id";
    if (got.tie_word_embeddings != want.tie_word_embeddings) return "config.tie_word_embeddings";
    if (got.max_position_embeddings != want.max_position_embeddings) return "config.max_position_embeddings";
    return nullptr;
}

// The largest draft window of an Inkling 2 Mini model with its MTP drafter (inkling design note D-I-S9B): a window
// is the verify rows per round, the unfed token plus window - 1 drafts (choose_drafts), and the drafter's block of 8
// gives at most 7 (Device::max_chain_drafts), so windows 2..8 run 1..7 drafts.
constexpr uint32_t kInkling2MaxDraftTokens = 8;

// What an Inkling 2 Mini library refuses in a descriptor of its own family, by name (CERE_ERR_UNSUPPORTED;
// inkling design note D-I5, D-I-S9B): a draft window other than off or 2..kInkling2MaxDraftTokens, any draft token
// map (the drafter is full-vocab), MTP seal calibration (the device times no rounds), the prefix cache (S6), and any
// non-default slot selection (no inkling lever exists yet; the same registry-default rule as sdar_selection_refusal).
// Header validity is check_header's, before this; the config is inkling2_config_refusal's, after it; a vision_config
// is plan_inkling2_model's (S6g).
constexpr const char *inkling2_refusal(const cere_model_desc &desc) {
    if (desc.speculative_draft_tokens == 1 || desc.speculative_draft_tokens > kInkling2MaxDraftTokens) {
        return "speculative_draft_tokens";
    }
    if (desc.draft_token_map != nullptr || desc.draft_token_map_count != 0) return "draft_token_map";
    if (desc.prefix_state_slots != 0) return "prefix_state_slots";
    if (desc.calibrate_windows != CERE_CALIBRATE_ALL) return "calibrate_windows";
    return sdar_selection_refusal(desc.selection);
}

// Why an Inkling config differs from the one the library was compiled for (`want`, from its descriptor), by field
// name; null when every value is equal.
constexpr const char *inkling2_config_refusal(const cere_inkling2_config &got, const cere_inkling2_config &want) {
    if (got.vocab_size != want.vocab_size) return "config.vocab_size";
    if (got.unpadded_vocab_size != want.unpadded_vocab_size) return "config.unpadded_vocab_size";
    if (got.hidden_size != want.hidden_size) return "config.hidden_size";
    if (got.num_hidden_layers != want.num_hidden_layers) return "config.num_hidden_layers";
    if (got.num_attention_heads != want.num_attention_heads) return "config.num_attention_heads";
    if (got.num_key_value_heads != want.num_key_value_heads) return "config.num_key_value_heads";
    if (got.head_dim != want.head_dim) return "config.head_dim";
    if (got.moe_intermediate_size != want.moe_intermediate_size) return "config.moe_intermediate_size";
    if (got.num_experts != want.num_experts) return "config.num_experts";
    if (got.num_experts_per_tok != want.num_experts_per_tok) return "config.num_experts_per_tok";
    if (got.sliding_window_size != want.sliding_window_size) return "config.sliding_window_size";
    if (got.rel_extent != want.rel_extent) return "config.rel_extent";
    if (got.d_rel != want.d_rel) return "config.d_rel";
    if (got.loop_core_start != want.loop_core_start) return "config.loop_core_start";
    if (got.loop_core_end != want.loop_core_end) return "config.loop_core_end";
    if (got.loop_num_passes != want.loop_num_passes) return "config.loop_num_passes";
    if (got.loop_gate_rank != want.loop_gate_rank) return "config.loop_gate_rank";
    if (got.n_layers_per_attn_res_block != want.n_layers_per_attn_res_block) return "config.n_layers_per_attn_res_block";
    if (got.conv_kernel_size != want.conv_kernel_size) return "config.conv_kernel_size";
    if (got.eos_token_id != want.eos_token_id) return "config.eos_token_id";
    if (got.local_layer_mask != want.local_layer_mask) return "config.local_layer_mask";
    if (got.mxfp8_layer_mask != want.mxfp8_layer_mask) return "config.mxfp8_layer_mask";
    if (!(got.rms_norm_eps == want.rms_norm_eps)) return "config.rms_norm_eps";
    if (!(got.final_logit_softcapping == want.final_logit_softcapping)) return "config.final_logit_softcapping";
    if (!(got.route_scale == want.route_scale)) return "config.route_scale";
    if (got.max_position_embeddings != want.max_position_embeddings) return "config.max_position_embeddings";
    return nullptr;
}

}  // namespace cere::host
