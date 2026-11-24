/* ABI v5 layout gate: compiled as C11 and C++17 by `make test-c`. Every struct size/offset and every constant is a
 * static assertion; the program prints the layout as JSON, which must equal tests/fixtures/executor_abi_v5_layout.json
 * byte for byte (the Rust projection compares its offset_of! values with that file). */
#include "cere_executor.h"

#include <stdio.h>

#ifdef __cplusplus
#include <type_traits>
#define ASSERT static_assert
#define ALIGNOF alignof
#define SAME_TYPE(expr, type) (std::is_same<decltype(expr), type>::value)
#else
#define ASSERT _Static_assert
#define ALIGNOF _Alignof
#define SAME_TYPE(expr, type) _Generic((expr), type: 1, default: 0)
#endif

#define FIELD_SIZE(type, field) sizeof(((type *)0)->field)
#define ASSERT_FIELD(type, field, offset, size)                                                                       \
    ASSERT(offsetof(type, field) == (offset), #type "." #field " offset");                                             \
    ASSERT(FIELD_SIZE(type, field) == (size), #type "." #field " size")

ASSERT(sizeof(void *) == 8, "LP64 target");
ASSERT(sizeof(cere_status) == 4, "status width");
ASSERT(sizeof(cere_execution_mode) == 4, "execution mode width");

/* Constants. */
ASSERT(CERE_EXECUTOR_ABI_VERSION == 5, "ABI version");
ASSERT(CERE_OK == 0, "ok");
ASSERT(CERE_PENDING == 1, "pending");
ASSERT(CERE_ERR_ARGUMENT == -1, "argument");
ASSERT(CERE_ERR_UNSUPPORTED == -2, "unsupported");
ASSERT(CERE_ERR_CAPACITY == -3, "capacity");
ASSERT(CERE_ERR_STALE_EPOCH == -4, "stale epoch");
ASSERT(CERE_ERR_BUSY == -5, "busy");
ASSERT(CERE_ERR_INTERNAL == -6, "internal");
ASSERT(CERE_OP_PREFILL == 1, "op prefill");
ASSERT(CERE_OP_DECODE == 2, "op decode");
ASSERT(CERE_OP_VERIFY == 3, "op verify");
ASSERT(CERE_OPS_PREFILL == 2, "ops prefill bit");
ASSERT(CERE_OPS_DECODE == 4, "ops decode bit");
ASSERT(CERE_OPS_VERIFY == 8, "ops verify bit");
ASSERT(CERE_OPS_CANCEL == 16, "ops cancel bit");
ASSERT(CERE_OPS_SNAPSHOT == 32, "ops snapshot bit");
ASSERT(CERE_OPS_RESTORE == 64, "ops restore bit");
ASSERT(CERE_WORK_IGNORE_STOP_TOKENS == 1, "work flag ignore stop tokens");
ASSERT(CERE_WORK_NO_SPECULATION == 2, "work flag no speculation");
ASSERT(CERE_WORK_LOOKUP_DRAFTS == 4, "work flag lookup drafts");
ASSERT(CERE_WORK_NO_MTP_DRAFTS == 8, "work flag no mtp drafts");
ASSERT(CERE_WORK_ADAPTIVE_DRAFTS == 16, "work flag adaptive drafts");
ASSERT(CERE_WORK_TAIL_DRAFTS == 0x20, "work flag tail drafts");
ASSERT(CERE_WORK_LONG_DRAFTS == 0x40, "work flag long drafts");
ASSERT(CERE_WORK_CHAIN_DRAFTS == 0x80, "work flag chain drafts");
ASSERT(CERE_WORK_KNOWN_FLAGS == 255, "work known flags");
ASSERT(CERE_OPS_PREFILL == (UINT64_C(1) << CERE_OP_PREFILL), "ops prefill = 1 << op");
ASSERT(CERE_OPS_DECODE == (UINT64_C(1) << CERE_OP_DECODE), "ops decode = 1 << op");
ASSERT(CERE_OPS_VERIFY == (UINT64_C(1) << CERE_OP_VERIFY), "ops verify = 1 << op");
ASSERT(CERE_EXECUTION_MODE_NATIVE == 1, "mode native");
ASSERT(CERE_EXECUTION_MODE_BACKEND_MANAGED == 2, "mode backend managed");
ASSERT(CERE_EXECUTION_MODE_SERIAL == 3, "mode serial");
ASSERT(CERE_EXECUTION_MODE_UNSUPPORTED == 4, "mode unsupported");
ASSERT(CERE_MODEL_QWEN4_EXP_NVFP4 == 1, "model kind");
ASSERT(CERE_MODEL_SDAR_BF16 == 2, "model kind sdar bf16");
ASSERT(CERE_MODEL_INKLING2_NVFP4 == 4, "model kind inkling2 nvfp4");
ASSERT(CERE_PLACEMENT_DEVICE == 1, "placement device");
ASSERT(CERE_PLACEMENT_HOST_PINNED == 2, "placement host pinned");
ASSERT(CERE_PLACEMENT_DISCARD == 3, "placement discard");
ASSERT(CERE_DTYPE_BF16 == 1, "dtype bf16");
ASSERT(CERE_DTYPE_F16 == 2, "dtype f16");
ASSERT(CERE_DTYPE_F32 == 3, "dtype f32");
ASSERT(CERE_DTYPE_F64 == 4, "dtype f64");
ASSERT(CERE_DTYPE_F8_E4M3 == 5, "dtype f8 e4m3");
ASSERT(CERE_DTYPE_F8_E5M2 == 6, "dtype f8 e5m2");
ASSERT(CERE_DTYPE_U8 == 7, "dtype u8");
ASSERT(CERE_DTYPE_I8 == 8, "dtype i8");
ASSERT(CERE_DTYPE_I16 == 9, "dtype i16");
ASSERT(CERE_DTYPE_I32 == 10, "dtype i32");
ASSERT(CERE_DTYPE_I64 == 11, "dtype i64");
ASSERT(CERE_DTYPE_BOOL == 12, "dtype bool");
ASSERT(CERE_OUTPUT_GATE_SIGMOID == 1, "output gate sigmoid");
ASSERT(CERE_ACT_GELU_TANH == 1, "act gelu tanh");
ASSERT(CERE_ACT_GELU_ERF == 2, "act gelu erf");
ASSERT(CERE_FINISH_NONE == 0, "finish none");
ASSERT(CERE_FINISH_LENGTH == 1, "finish length");
ASSERT(CERE_FINISH_STOP_TOKEN == 2, "finish stop token");
ASSERT(CERE_FINISH_CANCELLED == 3, "finish cancelled");
ASSERT(CERE_FINISH_ERROR == 4, "finish error");
ASSERT(CERE_TENSOR_MAX_DIMS == 8, "tensor rank limit");
ASSERT(CERE_MAX_TOP_LOGPROBS == 20, "top logprobs limit");
ASSERT(CERE_SAMPLING_KNOWN_FLAGS == 0, "sampling known flags");
ASSERT(CERE_LEVER_K2_CUBLASLT == 0, "lever k2 cublaslt");
ASSERT(CERE_LEVER_K2_SMALL_M == 1, "lever k2 small m");
ASSERT(CERE_LEVER_K2_SMALL_M_A8 == 2, "lever k2 small m a8");
ASSERT(CERE_LEVER_K2_TAIL_SPLIT == 3, "lever k2 tail split");
ASSERT(CERE_LEVER_PREFILL_CUDA_CORE == 0, "lever prefill cuda core");
ASSERT(CERE_LEVER_PREFILL_P4 == 1, "lever prefill p4");
ASSERT(CERE_LEVER_DECODE_EAGER == 0, "lever decode eager");
ASSERT(CERE_LEVER_DECODE_GRAPH == 1, "lever decode graph");
ASSERT(CERE_LEVER_DECODE_SPEC_GRAPH == 2, "lever decode spec graph");
ASSERT(CERE_LEVER_MOE_BUILD == 0, "lever moe build");
ASSERT(CERE_LEVER_MOE_P3B == 1, "lever moe p3b");
ASSERT(CERE_LEVER_MOE_K7 == 2, "lever moe k7");
ASSERT(CERE_LEVER_VERIFY_PER_ROW == 0, "lever verify per row");
ASSERT(CERE_LEVER_VERIFY_BATCHED == 1, "lever verify batched");
ASSERT(CERE_LEVER_K2_WEIGHTS_BF16 == 0, "lever k2 weights bf16");
ASSERT(CERE_LEVER_K2_WEIGHTS_FP8 == 1, "lever k2 weights fp8");
ASSERT(CERE_LEVER_SDAR_GEMM_IDENTICAL == 0, "lever sdar gemm identical");
ASSERT(CERE_LEVER_SDAR_GEMM_CHANGING == 1, "lever sdar gemm changing");
ASSERT(CERE_LEVER_DRAFT_WINDOW == 0, "lever draft window");
ASSERT(CERE_CALIBRATE_ALL == 0, "calibrate all");
ASSERT(CERE_CALIBRATE_MTP == 1, "calibrate mtp");
ASSERT(CERE_FAMILY_QWEN4_EXP == 0, "family qwen4 exp");
ASSERT(CERE_FAMILY_SDAR == 1, "family sdar");
ASSERT(CERE_FAMILY_INKLING2 == 3, "family inkling2");
ASSERT(CERE_SLOT_K2_GEMM == 1, "slot k2 gemm");
ASSERT(CERE_SLOT_PREFILL_ATTENTION == 2, "slot prefill attention");
ASSERT(CERE_SLOT_DECODE_STEP == 3, "slot decode step");
ASSERT(CERE_SLOT_MOE_GEMM == 4, "slot moe gemm");
ASSERT(CERE_SLOT_DRAFT_TOKENS == 5, "slot draft tokens");
ASSERT(CERE_SLOT_VERIFY_ATTENTION == 6, "slot verify attention");
ASSERT(CERE_SLOT_K2_WEIGHTS == 7, "slot k2 weights");
ASSERT(CERE_SLOT_SDAR_GEMM == 8, "slot sdar gemm");
ASSERT(CERE_SLOT_COUNT == 7, "slot count");
ASSERT(CERE_SLOT_BLOCK_VERSION == 1, "slot block version");
ASSERT(CERE_SLOT_MAX == 16, "slot max");
ASSERT(CERE_SLOT_PARAMS == 4, "slot params");
/* M2.10 (tasks/m210-spec.md D6, ledger D 104): bound of cere_model_desc.prefix_state_slots. */
ASSERT(CERE_MAX_PREFIX_STATE_SLOTS == 1024, "prefix state slots limit");
/* O79 (tasks/m210-spec.md PF1): bound of cere_executor_set_decode_rounds. */
ASSERT(CERE_MAX_DECODE_ROUNDS_PER_CHUNK == 64, "decode rounds per chunk limit");
/* MixedPD (tasks/mixed-pd-spec.md): bound of cere_executor_set_prefill_slice. */
ASSERT(CERE_MAX_PREFILL_SLICE_MS == 10000, "prefill slice ms limit");
/* BC4 (b) (tasks/batch-concurrency-spec.md): default and bound of cere_executor_set_prefill_age. */
ASSERT(CERE_DEFAULT_PREFILL_AGE_CALLS == 8, "prefill age calls default");
ASSERT(CERE_MAX_PREFILL_AGE_CALLS == 1024, "prefill age calls limit");
/* GrammarSpec (tasks/grammar-spec-spec.md GS-01): drafts of a constrained round. */
ASSERT(CERE_MAX_CONSTRAINED_DRAFTS == 3, "constrained drafts limit");
/* Track LoRA (tasks/lora-spec.md): adapter pool bounds and target modules of cere_model_lora_load. */
ASSERT(CERE_LORA_MAX_ADAPTERS == 64, "lora adapters limit");
ASSERT(CERE_LORA_MAX_RANK == 256, "lora rank limit");
ASSERT(CERE_LORA_MODULES == 8, "lora modules per layer");
ASSERT(CERE_LORA_Q_PROJ == 0, "lora q_proj");
ASSERT(CERE_LORA_K_PROJ == 1, "lora k_proj");
ASSERT(CERE_LORA_V_PROJ == 2, "lora v_proj");
ASSERT(CERE_LORA_O_PROJ == 3, "lora o_proj");
ASSERT(CERE_LORA_SHARED_GATE == 4, "lora shared gate");
ASSERT(CERE_LORA_SHARED_UP == 5, "lora shared up");
ASSERT(CERE_LORA_SHARED_DOWN == 6, "lora shared down");
/* S1 G4 (tasks/beat-sglang-spec.md): bound of cere_executor_set_decode_share. */
ASSERT(CERE_MAX_DECODE_SHARE_PCT == 1000, "decode share pct limit");
/* WeightLab (tasks/weightlab-spec.md WL-02): research plan bounds, kinds and capture dtypes. */
ASSERT(CERE_RESEARCH_MAX_PLANS == 64, "research plans limit");
ASSERT(CERE_RESEARCH_MAX_OPS == 128, "research ops limit");
ASSERT(CERE_RESEARCH_MAX_RANGES == 64, "research ranges limit");
ASSERT(CERE_RESEARCH_MAX_TOPK == 64, "research top_k limit");
ASSERT(CERE_RESEARCH_MAX_LOGIT_POSITIONS == 8, "research logits positions limit");
ASSERT(CERE_RESEARCH_MAX_PLAN_BYTES == 4194304, "research plan bytes");
ASSERT(CERE_RESEARCH_FINAL_LAYER == 48, "research final layer bit");
ASSERT(CERE_RESEARCH_CAPTURE == 0, "research capture");
ASSERT(CERE_RESEARCH_ADD == 1, "research add");
ASSERT(CERE_RESEARCH_ABLATE_ZERO == 2, "research ablate zero");
ASSERT(CERE_RESEARCH_ABLATE_MEAN == 3, "research ablate mean");
ASSERT(CERE_RESEARCH_REPLACE == 4, "research replace");
ASSERT(CERE_RESEARCH_EXPERT_BIAS == 5, "research expert bias");
ASSERT(CERE_RESEARCH_EXPERT_MASK == 6, "research expert mask");
ASSERT(CERE_RESEARCH_EXPERT_FORCE == 7, "research expert force");
ASSERT(CERE_RESEARCH_DTYPE_BF16 == 0, "research dtype bf16");
ASSERT(CERE_RESEARCH_DTYPE_I32 == 1, "research dtype i32");
ASSERT(CERE_RESEARCH_DTYPE_F32 == 2, "research dtype f32");

/* cere_executor_options */
ASSERT(sizeof(cere_executor_options) == 32, "options size");
ASSERT_FIELD(cere_executor_options, struct_size, 0, 4);
ASSERT_FIELD(cere_executor_options, abi_version, 4, 4);
ASSERT_FIELD(cere_executor_options, device_id, 8, 8);
ASSERT_FIELD(cere_executor_options, worker_epoch, 16, 8);
ASSERT_FIELD(cere_executor_options, memory_budget_bytes, 24, 8);

/* cere_execution_capabilities */
ASSERT(sizeof(cere_execution_capabilities) == 88, "capabilities size");
ASSERT_FIELD(cere_execution_capabilities, struct_size, 0, 4);
ASSERT_FIELD(cere_execution_capabilities, abi_version, 4, 4);
ASSERT_FIELD(cere_execution_capabilities, profile_generation, 8, 8);
ASSERT_FIELD(cere_execution_capabilities, worker_epoch, 16, 8);
ASSERT_FIELD(cere_execution_capabilities, decode_batch_min, 24, 8);
ASSERT_FIELD(cere_execution_capabilities, decode_batch_max, 32, 8);
ASSERT_FIELD(cere_execution_capabilities, max_context_tokens, 40, 8);
ASSERT_FIELD(cere_execution_capabilities, execution_mode, 48, 4);
ASSERT_FIELD(cere_execution_capabilities, native_decode_batch, 52, 1);
ASSERT_FIELD(cere_execution_capabilities, native_mixed_prefill_decode, 53, 1);
ASSERT_FIELD(cere_execution_capabilities, native_speculative_batch, 54, 1);
ASSERT_FIELD(cere_execution_capabilities, native_moe_grouping, 55, 1);
ASSERT_FIELD(cere_execution_capabilities, native_state_snapshot, 56, 1);
ASSERT_FIELD(cere_execution_capabilities, native_sampler, 57, 1);
ASSERT_FIELD(cere_execution_capabilities, native_token_mask, 58, 1);
ASSERT_FIELD(cere_execution_capabilities, speculative_draft_tokens, 59, 1);
ASSERT_FIELD(cere_execution_capabilities, native_prefix_cache, 60, 1);
ASSERT_FIELD(cere_execution_capabilities, native_image_input, 61, 1);
ASSERT_FIELD(cere_execution_capabilities, reserved, 62, 2);
ASSERT_FIELD(cere_execution_capabilities, operations, 64, 8);
ASSERT_FIELD(cere_execution_capabilities, kv_pool_tokens, 72, 8);
ASSERT_FIELD(cere_execution_capabilities, kv_page_tokens, 80, 4);
ASSERT_FIELD(cere_execution_capabilities, reserved2, 84, 4);

/* cere_completion_ticket */
ASSERT(sizeof(cere_completion_ticket) == 24, "ticket size");
ASSERT_FIELD(cere_completion_ticket, worker_epoch, 0, 8);
ASSERT_FIELD(cere_completion_ticket, model_generation, 8, 8);
ASSERT_FIELD(cere_completion_ticket, submission_id, 16, 8);

/* cere_qwen4_exp_config */
ASSERT(sizeof(cere_qwen4_exp_config) == 216, "config size");
ASSERT_FIELD(cere_qwen4_exp_config, struct_size, 0, 4);
ASSERT_FIELD(cere_qwen4_exp_config, abi_version, 4, 4);
ASSERT_FIELD(cere_qwen4_exp_config, vocab_size, 8, 4);
ASSERT_FIELD(cere_qwen4_exp_config, hidden_size, 12, 4);
ASSERT_FIELD(cere_qwen4_exp_config, num_hidden_layers, 16, 4);
ASSERT_FIELD(cere_qwen4_exp_config, full_attention_interval, 20, 4);
ASSERT_FIELD(cere_qwen4_exp_config, full_attention_layers, 24, 8);
ASSERT_FIELD(cere_qwen4_exp_config, num_attention_heads, 32, 4);
ASSERT_FIELD(cere_qwen4_exp_config, num_key_value_heads, 36, 4);
ASSERT_FIELD(cere_qwen4_exp_config, head_dim, 40, 4);
ASSERT_FIELD(cere_qwen4_exp_config, output_gate_type, 44, 4);
ASSERT_FIELD(cere_qwen4_exp_config, rope_theta, 48, 8);
ASSERT_FIELD(cere_qwen4_exp_config, partial_rotary_factor, 56, 8);
ASSERT_FIELD(cere_qwen4_exp_config, mrope_interleaved, 64, 4);
ASSERT_FIELD(cere_qwen4_exp_config, mrope_section, 68, 12);
ASSERT_FIELD(cere_qwen4_exp_config, indexer_n_heads, 80, 4);
ASSERT_FIELD(cere_qwen4_exp_config, indexer_head_dim, 84, 4);
ASSERT_FIELD(cere_qwen4_exp_config, indexer_kv_heads, 88, 4);
ASSERT_FIELD(cere_qwen4_exp_config, indexer_compress_ratio, 92, 4);
ASSERT_FIELD(cere_qwen4_exp_config, indexer_budget, 96, 4);
ASSERT_FIELD(cere_qwen4_exp_config, linear_num_key_heads, 100, 4);
ASSERT_FIELD(cere_qwen4_exp_config, linear_num_value_heads, 104, 4);
ASSERT_FIELD(cere_qwen4_exp_config, linear_key_head_dim, 108, 4);
ASSERT_FIELD(cere_qwen4_exp_config, linear_value_head_dim, 112, 4);
ASSERT_FIELD(cere_qwen4_exp_config, linear_conv_kernel_dim, 116, 4);
ASSERT_FIELD(cere_qwen4_exp_config, mamba_ssm_dtype, 120, 4);
ASSERT_FIELD(cere_qwen4_exp_config, num_experts, 124, 4);
ASSERT_FIELD(cere_qwen4_exp_config, num_experts_per_tok, 128, 4);
ASSERT_FIELD(cere_qwen4_exp_config, moe_intermediate_size, 132, 4);
ASSERT_FIELD(cere_qwen4_exp_config, shared_expert_intermediate_size, 136, 4);
ASSERT_FIELD(cere_qwen4_exp_config, norm_topk_prob, 140, 4);
ASSERT_FIELD(cere_qwen4_exp_config, hc_count, 144, 4);
ASSERT_FIELD(cere_qwen4_exp_config, hc_lowrank, 148, 4);
ASSERT_FIELD(cere_qwen4_exp_config, ple_layer_ids, 152, 8);
ASSERT_FIELD(cere_qwen4_exp_config, ple_embed_dim, 160, 4);
ASSERT_FIELD(cere_qwen4_exp_config, heads_per_ngram, 164, 4);
ASSERT_FIELD(cere_qwen4_exp_config, ngram_size, 168, 4);
ASSERT_FIELD(cere_qwen4_exp_config, ple_conv_kernel_size, 172, 4);
ASSERT_FIELD(cere_qwen4_exp_config, ngram_vocab_size_base, 176, 8);
ASSERT_FIELD(cere_qwen4_exp_config, split_ngram_parts, 184, 4);
ASSERT_FIELD(cere_qwen4_exp_config, make_ngram_vocab_size_divisible_by, 188, 4);
ASSERT_FIELD(cere_qwen4_exp_config, rms_norm_eps, 192, 8);
ASSERT_FIELD(cere_qwen4_exp_config, tie_word_embeddings, 200, 4);
ASSERT_FIELD(cere_qwen4_exp_config, mtp_num_hidden_layers, 204, 4);
ASSERT_FIELD(cere_qwen4_exp_config, max_position_embeddings, 208, 8);

/* cere_sdar_config (SelfTune ST4) */
ASSERT(sizeof(cere_sdar_config) == 72, "sdar config size");
ASSERT_FIELD(cere_sdar_config, struct_size, 0, 4);
ASSERT_FIELD(cere_sdar_config, abi_version, 4, 4);
ASSERT_FIELD(cere_sdar_config, vocab_size, 8, 4);
ASSERT_FIELD(cere_sdar_config, hidden_size, 12, 4);
ASSERT_FIELD(cere_sdar_config, intermediate_size, 16, 4);
ASSERT_FIELD(cere_sdar_config, num_hidden_layers, 20, 4);
ASSERT_FIELD(cere_sdar_config, num_attention_heads, 24, 4);
ASSERT_FIELD(cere_sdar_config, num_key_value_heads, 28, 4);
ASSERT_FIELD(cere_sdar_config, head_dim, 32, 4);
ASSERT_FIELD(cere_sdar_config, block_size, 36, 4);
ASSERT_FIELD(cere_sdar_config, rope_theta, 40, 8);
ASSERT_FIELD(cere_sdar_config, rms_norm_eps, 48, 8);
ASSERT_FIELD(cere_sdar_config, mask_token_id, 56, 4);
ASSERT_FIELD(cere_sdar_config, tie_word_embeddings, 60, 4);
ASSERT_FIELD(cere_sdar_config, max_position_embeddings, 64, 8);

/* cere_inkling2_config (tasks/inkling-spec.md S0) */
ASSERT(sizeof(cere_inkling2_config) == 128, "inkling2 config size");
ASSERT_FIELD(cere_inkling2_config, struct_size, 0, 4);
ASSERT_FIELD(cere_inkling2_config, abi_version, 4, 4);
ASSERT_FIELD(cere_inkling2_config, vocab_size, 8, 4);
ASSERT_FIELD(cere_inkling2_config, unpadded_vocab_size, 12, 4);
ASSERT_FIELD(cere_inkling2_config, hidden_size, 16, 4);
ASSERT_FIELD(cere_inkling2_config, num_hidden_layers, 20, 4);
ASSERT_FIELD(cere_inkling2_config, num_attention_heads, 24, 4);
ASSERT_FIELD(cere_inkling2_config, num_key_value_heads, 28, 4);
ASSERT_FIELD(cere_inkling2_config, head_dim, 32, 4);
ASSERT_FIELD(cere_inkling2_config, moe_intermediate_size, 36, 4);
ASSERT_FIELD(cere_inkling2_config, num_experts, 40, 4);
ASSERT_FIELD(cere_inkling2_config, num_experts_per_tok, 44, 4);
ASSERT_FIELD(cere_inkling2_config, sliding_window_size, 48, 4);
ASSERT_FIELD(cere_inkling2_config, rel_extent, 52, 4);
ASSERT_FIELD(cere_inkling2_config, d_rel, 56, 4);
ASSERT_FIELD(cere_inkling2_config, loop_core_start, 60, 4);
ASSERT_FIELD(cere_inkling2_config, loop_core_end, 64, 4);
ASSERT_FIELD(cere_inkling2_config, loop_num_passes, 68, 4);
ASSERT_FIELD(cere_inkling2_config, loop_gate_rank, 72, 4);
ASSERT_FIELD(cere_inkling2_config, n_layers_per_attn_res_block, 76, 4);
ASSERT_FIELD(cere_inkling2_config, conv_kernel_size, 80, 4);
ASSERT_FIELD(cere_inkling2_config, eos_token_id, 84, 4);
ASSERT_FIELD(cere_inkling2_config, local_layer_mask, 88, 4);
ASSERT_FIELD(cere_inkling2_config, mxfp8_layer_mask, 92, 4);
ASSERT_FIELD(cere_inkling2_config, rms_norm_eps, 96, 8);
ASSERT_FIELD(cere_inkling2_config, final_logit_softcapping, 104, 8);
ASSERT_FIELD(cere_inkling2_config, route_scale, 112, 8);
ASSERT_FIELD(cere_inkling2_config, max_position_embeddings, 120, 8);

/* cere_qwen4_exp_vision_config (ledger N) */
ASSERT(sizeof(cere_qwen4_exp_vision_config) == 80, "vision config size");
ASSERT_FIELD(cere_qwen4_exp_vision_config, struct_size, 0, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, abi_version, 4, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, depth, 8, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, hidden_size, 12, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, intermediate_size, 16, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, num_heads, 20, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, patch_size, 24, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, temporal_patch_size, 28, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, spatial_merge_size, 32, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, in_channels, 36, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, out_hidden_size, 40, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, num_position_embeddings, 44, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, deepstack_count, 48, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, hidden_act, 52, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, image_token_id, 56, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, video_token_id, 60, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, vision_start_token_id, 64, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, vision_end_token_id, 68, 4);
ASSERT_FIELD(cere_qwen4_exp_vision_config, layer_norm_eps, 72, 8);

/* cere_tensor_desc */
ASSERT(sizeof(cere_tensor_desc) == 104, "tensor desc size");
ASSERT_FIELD(cere_tensor_desc, struct_size, 0, 4);
ASSERT_FIELD(cere_tensor_desc, abi_version, 4, 4);
ASSERT_FIELD(cere_tensor_desc, name, 8, 8);
ASSERT_FIELD(cere_tensor_desc, dtype, 16, 4);
ASSERT_FIELD(cere_tensor_desc, placement, 20, 4);
ASSERT_FIELD(cere_tensor_desc, ndim, 24, 4);
ASSERT_FIELD(cere_tensor_desc, reserved, 28, 4);
ASSERT_FIELD(cere_tensor_desc, shape, 32, 64);
ASSERT_FIELD(cere_tensor_desc, bytes, 96, 8);

/* cere_slot_choice, cere_slot_selection (ABI v5) */
ASSERT(sizeof(cere_slot_choice) == 24, "slot choice size");
ASSERT_FIELD(cere_slot_choice, slot, 0, 4);
ASSERT_FIELD(cere_slot_choice, impl, 4, 4);
ASSERT_FIELD(cere_slot_choice, params, 8, 16);
ASSERT(sizeof(cere_slot_selection) == 400, "slot selection size");
ASSERT_FIELD(cere_slot_selection, struct_size, 0, 4);
ASSERT_FIELD(cere_slot_selection, abi_version, 4, 4);
ASSERT_FIELD(cere_slot_selection, block_version, 8, 4);
ASSERT_FIELD(cere_slot_selection, count, 12, 4);
ASSERT_FIELD(cere_slot_selection, choices, 16, 384);

/* cere_model_desc */
ASSERT(sizeof(cere_model_desc) == 136, "model desc size");
ASSERT_FIELD(cere_model_desc, struct_size, 0, 4);
ASSERT_FIELD(cere_model_desc, abi_version, 4, 4);
ASSERT_FIELD(cere_model_desc, model_kind, 8, 4);
ASSERT_FIELD(cere_model_desc, speculative_draft_tokens, 12, 4);
ASSERT_FIELD(cere_model_desc, config, 16, 8);
ASSERT_FIELD(cere_model_desc, tensors, 24, 8);
ASSERT_FIELD(cere_model_desc, tensor_count, 32, 8);
ASSERT_FIELD(cere_model_desc, max_sessions, 40, 8);
ASSERT_FIELD(cere_model_desc, max_context_tokens, 48, 8);
ASSERT_FIELD(cere_model_desc, stop_token_ids, 56, 8);
ASSERT_FIELD(cere_model_desc, stop_token_count, 64, 8);
ASSERT_FIELD(cere_model_desc, kv_pool_tokens, 72, 8);
ASSERT_FIELD(cere_model_desc, draft_token_map, 80, 8);
ASSERT_FIELD(cere_model_desc, draft_token_map_count, 88, 8);
ASSERT_FIELD(cere_model_desc, snapshot_host_bytes, 96, 8);
ASSERT_FIELD(cere_model_desc, prefix_state_slots, 104, 8);
ASSERT_FIELD(cere_model_desc, vision_config, 112, 8);
ASSERT_FIELD(cere_model_desc, selection, 120, 8);
ASSERT_FIELD(cere_model_desc, family, 128, 4);
ASSERT_FIELD(cere_model_desc, calibrate_windows, 132, 4);

/* cere_token_score */
ASSERT(sizeof(cere_token_score) == 16, "token score size");
ASSERT_FIELD(cere_token_score, id, 0, 4);
ASSERT_FIELD(cere_token_score, runner_up, 4, 4);
ASSERT_FIELD(cere_token_score, logit, 8, 4);
ASSERT_FIELD(cere_token_score, runner_up_logit, 12, 4);

/* cere_logit_bias, cere_sampling_params, cere_logprob_entry, cere_token_logprobs (ledger table N, M2.7) */
ASSERT(sizeof(cere_logit_bias) == 8, "logit bias size");
ASSERT_FIELD(cere_logit_bias, id, 0, 4);
ASSERT_FIELD(cere_logit_bias, bias, 4, 4);
ASSERT(sizeof(cere_sampling_params) == 88, "sampling params size");
ASSERT_FIELD(cere_sampling_params, struct_size, 0, 4);
ASSERT_FIELD(cere_sampling_params, abi_version, 4, 4);
ASSERT_FIELD(cere_sampling_params, temperature, 8, 4);
ASSERT_FIELD(cere_sampling_params, top_p, 12, 4);
ASSERT_FIELD(cere_sampling_params, min_p, 16, 4);
ASSERT_FIELD(cere_sampling_params, top_k, 20, 4);
ASSERT_FIELD(cere_sampling_params, frequency_penalty, 24, 4);
ASSERT_FIELD(cere_sampling_params, presence_penalty, 28, 4);
ASSERT_FIELD(cere_sampling_params, repetition_penalty, 32, 4);
ASSERT_FIELD(cere_sampling_params, flags, 36, 4);
ASSERT_FIELD(cere_sampling_params, seed, 40, 8);
ASSERT_FIELD(cere_sampling_params, logit_bias, 48, 8);
ASSERT_FIELD(cere_sampling_params, logit_bias_count, 56, 8);
ASSERT_FIELD(cere_sampling_params, suppress_token_ids, 64, 8);
ASSERT_FIELD(cere_sampling_params, suppress_token_count, 72, 8);
ASSERT_FIELD(cere_sampling_params, mask_until_committed, 80, 8);
ASSERT(sizeof(cere_logprob_entry) == 8, "logprob entry size");
ASSERT_FIELD(cere_logprob_entry, id, 0, 4);
ASSERT_FIELD(cere_logprob_entry, logprob, 4, 4);
ASSERT(sizeof(cere_token_logprobs) == 176, "token logprobs size");
ASSERT_FIELD(cere_token_logprobs, id, 0, 4);
ASSERT_FIELD(cere_token_logprobs, logprob, 4, 4);
ASSERT_FIELD(cere_token_logprobs, top_count, 8, 4);
ASSERT_FIELD(cere_token_logprobs, reserved, 12, 4);
ASSERT_FIELD(cere_token_logprobs, top, 16, 160);

/* cere_image_input (ledger N; M2.13-L C1 pixel_sha256) */
ASSERT(sizeof(cere_image_input) == 80, "image input size");
ASSERT_FIELD(cere_image_input, struct_size, 0, 4);
ASSERT_FIELD(cere_image_input, abi_version, 4, 4);
ASSERT_FIELD(cere_image_input, grid_t, 8, 4);
ASSERT_FIELD(cere_image_input, grid_h, 12, 4);
ASSERT_FIELD(cere_image_input, grid_w, 16, 4);
ASSERT_FIELD(cere_image_input, pad_id, 20, 4);
ASSERT_FIELD(cere_image_input, token_offset, 24, 8);
ASSERT_FIELD(cere_image_input, pixel_patches, 32, 8);
ASSERT_FIELD(cere_image_input, pixel_bytes, 40, 8);
ASSERT_FIELD(cere_image_input, pixel_sha256, 48, 32);

/* cere_work */
ASSERT(sizeof(cere_work) == 168, "work size");
ASSERT_FIELD(cere_work, struct_size, 0, 4);
ASSERT_FIELD(cere_work, abi_version, 4, 4);
ASSERT_FIELD(cere_work, operation, 8, 4);
ASSERT_FIELD(cere_work, flags, 12, 4);
ASSERT_FIELD(cere_work, expected_worker_epoch, 16, 8);
ASSERT_FIELD(cere_work, expected_session_revision, 24, 8);
ASSERT_FIELD(cere_work, session, 32, 8);
ASSERT_FIELD(cere_work, tokens, 40, 8);
ASSERT_FIELD(cere_work, token_count, 48, 8);
ASSERT_FIELD(cere_work, max_new_tokens, 56, 8);
ASSERT_FIELD(cere_work, reserved_peak_bytes, 64, 8);
ASSERT_FIELD(cere_work, out_tokens, 72, 8);
ASSERT_FIELD(cere_work, out_capacity, 80, 8);
ASSERT_FIELD(cere_work, out_scores, 88, 8);
ASSERT_FIELD(cere_work, stop_token_ids, 96, 8);
ASSERT_FIELD(cere_work, stop_token_count, 104, 8);
ASSERT_FIELD(cere_work, sampling, 112, 8);
ASSERT_FIELD(cere_work, out_logprobs, 120, 8);
ASSERT_FIELD(cere_work, top_logprobs, 128, 4);
ASSERT_FIELD(cere_work, capture_position, 132, 4);
ASSERT_FIELD(cere_work, vocab_mask, 136, 8);
ASSERT_FIELD(cere_work, vocab_mask_words, 144, 8);
ASSERT_FIELD(cere_work, images, 152, 8);
ASSERT_FIELD(cere_work, image_count, 160, 8);

/* cere_batch */
ASSERT(sizeof(cere_batch) == 48, "batch size");
ASSERT_FIELD(cere_batch, struct_size, 0, 4);
ASSERT_FIELD(cere_batch, abi_version, 4, 4);
ASSERT_FIELD(cere_batch, expected_worker_epoch, 8, 8);
ASSERT_FIELD(cere_batch, capability_profile_generation, 16, 8);
ASSERT_FIELD(cere_batch, items, 24, 8);
ASSERT_FIELD(cere_batch, item_count, 32, 8);
ASSERT_FIELD(cere_batch, reserved_peak_bytes, 40, 8);

/* cere_completion */
ASSERT(sizeof(cere_completion) == 64, "completion size");
ASSERT_FIELD(cere_completion, struct_size, 0, 4);
ASSERT_FIELD(cere_completion, abi_version, 4, 4);
ASSERT_FIELD(cere_completion, status, 8, 4);
ASSERT_FIELD(cere_completion, terminal, 12, 4);
ASSERT_FIELD(cere_completion, committed_tokens, 16, 8);
ASSERT_FIELD(cere_completion, session_revision, 24, 8);
ASSERT_FIELD(cere_completion, released_bytes, 32, 8);
ASSERT_FIELD(cere_completion, finish, 40, 4);
ASSERT_FIELD(cere_completion, cached_tokens, 44, 4);
ASSERT_FIELD(cere_completion, verify_steps, 48, 8);
ASSERT_FIELD(cere_completion, accepted_drafts, 56, 8);

/* cere_snapshot_info (M2.11, ledger N: 72 B, one size) */
ASSERT(sizeof(cere_snapshot_info) == 72, "snapshot info size");
ASSERT_FIELD(cere_snapshot_info, struct_size, 0, 4);
ASSERT_FIELD(cere_snapshot_info, abi_version, 4, 4);
ASSERT_FIELD(cere_snapshot_info, bytes, 8, 8);
ASSERT_FIELD(cere_snapshot_info, position, 16, 8);
ASSERT_FIELD(cere_snapshot_info, token_count, 24, 8);
ASSERT_FIELD(cere_snapshot_info, fed, 32, 8);
ASSERT_FIELD(cere_snapshot_info, layout_id, 40, 32);

/* cere_prefix_stats (appended to ABI v4: 104 B, one size) */
ASSERT(sizeof(cere_prefix_stats) == 104, "prefix stats size");
ASSERT_FIELD(cere_prefix_stats, struct_size, 0, 4);
ASSERT_FIELD(cere_prefix_stats, abi_version, 4, 4);
ASSERT_FIELD(cere_prefix_stats, alt_edges_on, 8, 4);
ASSERT_FIELD(cere_prefix_stats, reserved, 12, 4);
ASSERT_FIELD(cere_prefix_stats, hits, 16, 8);
ASSERT_FIELD(cere_prefix_stats, captures, 24, 8);
ASSERT_FIELD(cere_prefix_stats, capture_skipped, 32, 8);
ASSERT_FIELD(cere_prefix_stats, capture_page_mismatch, 40, 8);
ASSERT_FIELD(cere_prefix_stats, inserts, 48, 8);
ASSERT_FIELD(cere_prefix_stats, state_evictions, 56, 8);
ASSERT_FIELD(cere_prefix_stats, leaf_evictions, 64, 8);
ASSERT_FIELD(cere_prefix_stats, host_spills, 72, 8);
ASSERT_FIELD(cere_prefix_stats, host_loads, 80, 8);
ASSERT_FIELD(cere_prefix_stats, host_drops, 88, 8);
ASSERT_FIELD(cere_prefix_stats, alt_edges, 96, 8);

/* cere_prefix_capture_stats (appended to ABI v4: 48 B, one size) */
ASSERT(sizeof(cere_prefix_capture_stats) == 48, "prefix capture stats size");
ASSERT_FIELD(cere_prefix_capture_stats, struct_size, 0, 4);
ASSERT_FIELD(cere_prefix_capture_stats, abi_version, 4, 4);
ASSERT_FIELD(cere_prefix_capture_stats, skipped_prefill_slot, 8, 8);
ASSERT_FIELD(cere_prefix_capture_stats, skipped_decode_track, 16, 8);
ASSERT_FIELD(cere_prefix_capture_stats, skipped_end_pair, 24, 8);
ASSERT_FIELD(cere_prefix_capture_stats, hit_shortfall_tokens, 32, 8);
ASSERT_FIELD(cere_prefix_capture_stats, hit_shortfall_lookups, 40, 8);

/* WeightLab research structs (appended to ABI v4, one size each, no struct_size header: passed by pointer only to
 * the research functions) */
ASSERT(sizeof(cere_research_range) == 16, "research range size");
ASSERT_FIELD(cere_research_range, first, 0, 8);
ASSERT_FIELD(cere_research_range, end, 8, 8);
ASSERT(sizeof(cere_research_op) == 96, "research op size");
ASSERT_FIELD(cere_research_op, kind, 0, 4);
ASSERT_FIELD(cere_research_op, top_k, 4, 4);
ASSERT_FIELD(cere_research_op, point, 8, 8);
ASSERT_FIELD(cere_research_op, layers, 16, 8);
ASSERT_FIELD(cere_research_op, prompt, 24, 8);
ASSERT_FIELD(cere_research_op, prompt_count, 32, 4);
ASSERT_FIELD(cere_research_op, generated, 36, 4);
ASSERT_FIELD(cere_research_op, generated_max, 40, 8);
ASSERT_FIELD(cere_research_op, alpha, 48, 4);
ASSERT_FIELD(cere_research_op, col_begin, 52, 4);
ASSERT_FIELD(cere_research_op, col_end, 56, 4);
ASSERT_FIELD(cere_research_op, vector_rows, 60, 4);
ASSERT_FIELD(cere_research_op, vector, 64, 8);
ASSERT_FIELD(cere_research_op, vector_cols, 72, 4);
ASSERT_FIELD(cere_research_op, expert_count, 76, 4);
ASSERT_FIELD(cere_research_op, experts, 80, 8);
ASSERT_FIELD(cere_research_op, weights, 88, 8);
ASSERT(sizeof(cere_research_plan) == 48, "research plan size");
ASSERT_FIELD(cere_research_plan, ops, 0, 8);
ASSERT_FIELD(cere_research_plan, op_count, 8, 4);
ASSERT_FIELD(cere_research_plan, reserved, 12, 4);
ASSERT_FIELD(cere_research_plan, sha256, 16, 32);
ASSERT(sizeof(cere_research_tensor) == 48, "research tensor size");
ASSERT_FIELD(cere_research_tensor, name, 0, 8);
ASSERT_FIELD(cere_research_tensor, op, 8, 4);
ASSERT_FIELD(cere_research_tensor, layer, 12, 4);
ASSERT_FIELD(cere_research_tensor, dtype, 16, 4);
ASSERT_FIELD(cere_research_tensor, cols, 20, 4);
ASSERT_FIELD(cere_research_tensor, rows, 24, 8);
ASSERT_FIELD(cere_research_tensor, positions, 32, 8);
ASSERT_FIELD(cere_research_tensor, data, 40, 8);
ASSERT(sizeof(cere_research_capture) == 64, "research capture size");
ASSERT_FIELD(cere_research_capture, tensors, 0, 8);
ASSERT_FIELD(cere_research_capture, tensor_count, 8, 8);
ASSERT_FIELD(cere_research_capture, bytes, 16, 8);
ASSERT_FIELD(cere_research_capture, prefix_lookup_skipped, 24, 4);
ASSERT_FIELD(cere_research_capture, reserved, 28, 4);
ASSERT_FIELD(cere_research_capture, plan_sha256, 32, 32);

/* Function signatures, checked in unevaluated contexts (nothing is linked). */
typedef uint32_t (*abi_version_fn)(void);
typedef cere_status (*executor_create_fn)(cere_executor **, const cere_executor_options *);
typedef cere_status (*executor_destroy_fn)(cere_executor *);
typedef cere_status (*executor_capabilities_fn)(cere_executor *, const cere_model *, cere_execution_capabilities *);
typedef cere_status (*model_create_fn)(cere_executor *, const cere_model_desc *, cere_model **);
typedef cere_status (*model_put_fn)(cere_executor *, cere_model *, uint64_t, uint64_t, const void *, uint64_t);
typedef cere_status (*model_seal_fn)(cere_executor *, cere_model *);
typedef cere_status (*model_release_fn)(cere_executor *, cere_model *);
typedef cere_status (*session_create_fn)(cere_executor *, cere_model *, uint64_t, cere_session **);
typedef cere_status (*session_release_fn)(cere_executor *, cere_session *);
typedef cere_status (*executor_submit_fn)(cere_executor *, const cere_batch *, cere_completion_ticket *);
typedef cere_status (*executor_cancel_fn)(cere_executor *, cere_completion_ticket);
typedef cere_status (*executor_poll_fn)(cere_executor *, cere_completion_ticket, cere_completion *);
typedef cere_status (*executor_constrain_fn)(cere_executor *, cere_completion_ticket, uint64_t, const uint32_t *,
                                             uint64_t);
typedef cere_status (*session_snapshot_fn)(cere_executor *, cere_session *, uint64_t, cere_snapshot **);
typedef cere_status (*session_restore_fn)(cere_executor *, cere_session *, const cere_snapshot *);
typedef cere_status (*snapshot_get_info_fn)(cere_executor *, const cere_snapshot *, cere_snapshot_info *);
typedef cere_status (*snapshot_read_fn)(cere_executor *, const cere_snapshot *, uint64_t, void *, uint64_t);
typedef cere_status (*snapshot_import_fn)(cere_executor *, cere_model *, const void *, uint64_t, cere_snapshot **);
typedef cere_status (*snapshot_release_fn)(cere_executor *, cere_snapshot *);
/* M2.10 (tasks/m210-spec.md D2, D11, D12; ledger X): namespace, flush, export, import. */
typedef cere_status (*session_set_cache_namespace_fn)(cere_executor *, cere_session *, const uint8_t *);
typedef cere_status (*model_flush_prefix_cache_fn)(cere_executor *, cere_model *);
typedef cere_status (*prefix_cache_export_fn)(cere_executor *, cere_model *, const uint8_t *, const uint32_t *,
                                              uint64_t, cere_snapshot **);
typedef cere_status (*prefix_cache_import_fn)(cere_executor *, cere_model *, const uint8_t *,
                                              const cere_snapshot *);
typedef cere_status (*model_host_resident_fn)(cere_executor *, cere_model *, uint64_t, const void **);
typedef cere_status (*model_selection_fn)(cere_executor *, cere_model *, cere_slot_selection *);
typedef cere_status (*model_set_selection_fn)(cere_executor *, cere_model *, const cere_slot_selection *);
typedef cere_status (*executor_register_staging_fn)(cere_executor *, const void *, uint64_t);
typedef cere_status (*executor_unregister_staging_fn)(cere_executor *, const void *);
typedef cere_status (*executor_set_decode_rounds_fn)(cere_executor *, uint32_t);
typedef cere_status (*model_set_prefix_end_state_fn)(cere_executor *, cere_model *, uint32_t);
typedef cere_status (*model_set_prefix_alt_edges_fn)(cere_executor *, cere_model *, uint32_t);
typedef cere_status (*model_prefix_stats_fn)(cere_executor *, cere_model *, cere_prefix_stats *);
typedef cere_status (*model_prefix_capture_stats_fn)(cere_executor *, cere_model *, cere_prefix_capture_stats *);
typedef cere_status (*executor_set_prefill_slice_fn)(cere_executor *, uint32_t);
typedef cere_status (*executor_round_drafts_fn)(cere_executor *, cere_completion_ticket, uint64_t, uint32_t *, uint32_t,
                                                uint32_t *);
typedef cere_status (*executor_constrain_drafts_fn)(cere_executor *, cere_completion_ticket, uint64_t, const uint32_t *,
                                                    uint32_t, const uint32_t *, uint64_t);
typedef cere_status (*model_lora_reserve_fn)(cere_executor *, cere_model *, uint32_t, uint32_t);
typedef cere_status (*model_lora_load_fn)(cere_executor *, cere_model *, uint32_t, uint32_t, float, uint64_t,
                                          const uint32_t *, const void *const *, const void *const *);
typedef cere_status (*model_lora_unload_fn)(cere_executor *, cere_model *, uint32_t);
typedef cere_status (*session_set_lora_fn)(cere_executor *, cere_session *, uint32_t);
typedef cere_status (*executor_set_decode_share_fn)(cere_executor *, uint32_t);
typedef cere_status (*model_research_reserve_fn)(cere_executor *, cere_model *, uint64_t, uint64_t, uint32_t);
typedef cere_status (*model_research_plan_load_fn)(cere_executor *, cere_model *, uint32_t,
                                                   const cere_research_plan *);
typedef cere_status (*model_research_plan_unload_fn)(cere_executor *, cere_model *, uint32_t);
typedef cere_status (*session_set_research_fn)(cere_executor *, cere_session *, uint32_t, uint64_t, uint64_t,
                                               uint64_t *);
typedef cere_status (*model_research_capture_read_fn)(cere_executor *, cere_model *, uint64_t,
                                                      cere_research_capture *);
typedef cere_status (*model_research_capture_free_fn)(cere_executor *, cere_model *, uint64_t);
typedef cere_status (*model_prefill_signature_fn)(cere_executor *, const cere_model *, uint8_t *);
typedef cere_status (*executor_set_prefill_age_fn)(cere_executor *, uint32_t);
#define ASSERT_SIGNATURE(fn, type) ASSERT(SAME_TYPE(&fn, type), #fn " signature")
ASSERT_SIGNATURE(cere_executor_abi_version, abi_version_fn);
ASSERT_SIGNATURE(cere_executor_create, executor_create_fn);
ASSERT_SIGNATURE(cere_executor_destroy, executor_destroy_fn);
ASSERT_SIGNATURE(cere_executor_capabilities, executor_capabilities_fn);
ASSERT_SIGNATURE(cere_model_create, model_create_fn);
ASSERT_SIGNATURE(cere_model_put, model_put_fn);
ASSERT_SIGNATURE(cere_model_seal, model_seal_fn);
ASSERT_SIGNATURE(cere_model_release, model_release_fn);
ASSERT_SIGNATURE(cere_session_create, session_create_fn);
ASSERT_SIGNATURE(cere_session_release, session_release_fn);
ASSERT_SIGNATURE(cere_executor_submit, executor_submit_fn);
ASSERT_SIGNATURE(cere_executor_cancel, executor_cancel_fn);
ASSERT_SIGNATURE(cere_executor_poll, executor_poll_fn);
ASSERT_SIGNATURE(cere_executor_constrain, executor_constrain_fn);
ASSERT_SIGNATURE(cere_session_snapshot, session_snapshot_fn);
ASSERT_SIGNATURE(cere_session_restore, session_restore_fn);
ASSERT_SIGNATURE(cere_snapshot_get_info, snapshot_get_info_fn);
ASSERT_SIGNATURE(cere_snapshot_read, snapshot_read_fn);
ASSERT_SIGNATURE(cere_snapshot_import, snapshot_import_fn);
ASSERT_SIGNATURE(cere_snapshot_release, snapshot_release_fn);
ASSERT_SIGNATURE(cere_session_set_cache_namespace, session_set_cache_namespace_fn);
ASSERT_SIGNATURE(cere_model_flush_prefix_cache, model_flush_prefix_cache_fn);
ASSERT_SIGNATURE(cere_prefix_cache_export, prefix_cache_export_fn);
ASSERT_SIGNATURE(cere_prefix_cache_import, prefix_cache_import_fn);
ASSERT_SIGNATURE(cere_model_host_resident, model_host_resident_fn);
ASSERT_SIGNATURE(cere_model_selection, model_selection_fn);
ASSERT_SIGNATURE(cere_model_set_selection, model_set_selection_fn);
ASSERT_SIGNATURE(cere_executor_register_staging, executor_register_staging_fn);
ASSERT_SIGNATURE(cere_executor_unregister_staging, executor_unregister_staging_fn);
ASSERT_SIGNATURE(cere_executor_set_decode_rounds, executor_set_decode_rounds_fn);
ASSERT_SIGNATURE(cere_model_set_prefix_end_state, model_set_prefix_end_state_fn);
ASSERT_SIGNATURE(cere_model_set_prefix_alt_edges, model_set_prefix_alt_edges_fn);
ASSERT_SIGNATURE(cere_model_prefix_stats, model_prefix_stats_fn);
ASSERT_SIGNATURE(cere_model_prefix_capture_stats, model_prefix_capture_stats_fn);
ASSERT_SIGNATURE(cere_executor_set_prefill_slice, executor_set_prefill_slice_fn);
ASSERT_SIGNATURE(cere_executor_round_drafts, executor_round_drafts_fn);
ASSERT_SIGNATURE(cere_executor_constrain_drafts, executor_constrain_drafts_fn);
ASSERT_SIGNATURE(cere_model_lora_reserve, model_lora_reserve_fn);
ASSERT_SIGNATURE(cere_model_lora_load, model_lora_load_fn);
ASSERT_SIGNATURE(cere_model_lora_unload, model_lora_unload_fn);
ASSERT_SIGNATURE(cere_session_set_lora, session_set_lora_fn);
ASSERT_SIGNATURE(cere_executor_set_decode_share, executor_set_decode_share_fn);
ASSERT_SIGNATURE(cere_model_research_reserve, model_research_reserve_fn);
ASSERT_SIGNATURE(cere_model_research_plan_load, model_research_plan_load_fn);
ASSERT_SIGNATURE(cere_model_research_plan_unload, model_research_plan_unload_fn);
ASSERT_SIGNATURE(cere_session_set_research, session_set_research_fn);
ASSERT_SIGNATURE(cere_model_research_capture_read, model_research_capture_read_fn);
ASSERT_SIGNATURE(cere_model_research_capture_free, model_research_capture_free_fn);
ASSERT_SIGNATURE(cere_model_prefill_signature, model_prefill_signature_fn);
ASSERT_SIGNATURE(cere_executor_set_prefill_age, executor_set_prefill_age_fn);

/* JSON layout dump. Order is declaration order; the output is compared byte for byte with the committed fixture. */
static int first_item;

static void begin_list(const char *indent, const char *name) {
    printf("%s\"%s\": [", indent, name);
    first_item = 1;
}

static void end_list(const char *indent, int last) {
    printf("\n%s]%s\n", indent, last ? "" : ",");
}

static void item_prefix(void) {
    printf(first_item ? "\n" : ",\n");
    first_item = 0;
}

#define CONSTANT(name)                                                                                                \
    do {                                                                                                               \
        item_prefix();                                                                                                 \
        printf("    {\"name\": \"%s\", \"value\": %lld}", #name, (long long)(name));                                   \
    } while (0)

#define STRUCT_BEGIN(type)                                                                                            \
    do {                                                                                                               \
        item_prefix();                                                                                                 \
        printf("    {\"name\": \"%s\", \"size\": %u, \"align\": %u, \"fields\": [", #type, (unsigned)sizeof(type),     \
               (unsigned)ALIGNOF(type));                                                                              \
        first_item = 1;                                                                                                \
    } while (0)

#define STRUCT_FIELD(type, field)                                                                                     \
    do {                                                                                                               \
        printf("%s\n      {\"name\": \"%s\", \"offset\": %u, \"size\": %u}", first_item ? "" : ",", #field,           \
               (unsigned)offsetof(type, field), (unsigned)FIELD_SIZE(type, field));                                    \
        first_item = 0;                                                                                                \
    } while (0)

#define STRUCT_END()                                                                                                   \
    do {                                                                                                               \
        printf("\n    ]}");                                                                                            \
        first_item = 0;                                                                                                \
    } while (0)

static void dump_constants(void) {
    begin_list("  ", "constants");
    CONSTANT(CERE_EXECUTOR_ABI_VERSION);
    CONSTANT(CERE_OK);
    CONSTANT(CERE_PENDING);
    CONSTANT(CERE_ERR_ARGUMENT);
    CONSTANT(CERE_ERR_UNSUPPORTED);
    CONSTANT(CERE_ERR_CAPACITY);
    CONSTANT(CERE_ERR_STALE_EPOCH);
    CONSTANT(CERE_ERR_BUSY);
    CONSTANT(CERE_ERR_INTERNAL);
    CONSTANT(CERE_OP_PREFILL);
    CONSTANT(CERE_OP_DECODE);
    CONSTANT(CERE_OP_VERIFY);
    CONSTANT(CERE_OPS_PREFILL);
    CONSTANT(CERE_OPS_DECODE);
    CONSTANT(CERE_OPS_VERIFY);
    CONSTANT(CERE_OPS_CANCEL);
    CONSTANT(CERE_OPS_SNAPSHOT);
    CONSTANT(CERE_OPS_RESTORE);
    CONSTANT(CERE_WORK_IGNORE_STOP_TOKENS);
    CONSTANT(CERE_WORK_NO_SPECULATION);
    CONSTANT(CERE_WORK_LOOKUP_DRAFTS);
    CONSTANT(CERE_WORK_NO_MTP_DRAFTS);
    CONSTANT(CERE_WORK_ADAPTIVE_DRAFTS);
    CONSTANT(CERE_WORK_TAIL_DRAFTS);
    CONSTANT(CERE_WORK_LONG_DRAFTS);
    CONSTANT(CERE_WORK_CHAIN_DRAFTS);
    CONSTANT(CERE_WORK_KNOWN_FLAGS);
    CONSTANT(CERE_EXECUTION_MODE_NATIVE);
    CONSTANT(CERE_EXECUTION_MODE_BACKEND_MANAGED);
    CONSTANT(CERE_EXECUTION_MODE_SERIAL);
    CONSTANT(CERE_EXECUTION_MODE_UNSUPPORTED);
    CONSTANT(CERE_MODEL_QWEN4_EXP_NVFP4);
    CONSTANT(CERE_MODEL_SDAR_BF16);
    CONSTANT(CERE_MODEL_INKLING2_NVFP4);
    CONSTANT(CERE_PLACEMENT_DEVICE);
    CONSTANT(CERE_PLACEMENT_HOST_PINNED);
    CONSTANT(CERE_PLACEMENT_DISCARD);
    CONSTANT(CERE_DTYPE_BF16);
    CONSTANT(CERE_DTYPE_F16);
    CONSTANT(CERE_DTYPE_F32);
    CONSTANT(CERE_DTYPE_F64);
    CONSTANT(CERE_DTYPE_F8_E4M3);
    CONSTANT(CERE_DTYPE_F8_E5M2);
    CONSTANT(CERE_DTYPE_U8);
    CONSTANT(CERE_DTYPE_I8);
    CONSTANT(CERE_DTYPE_I16);
    CONSTANT(CERE_DTYPE_I32);
    CONSTANT(CERE_DTYPE_I64);
    CONSTANT(CERE_DTYPE_BOOL);
    CONSTANT(CERE_OUTPUT_GATE_SIGMOID);
    CONSTANT(CERE_ACT_GELU_TANH);
    CONSTANT(CERE_ACT_GELU_ERF);
    CONSTANT(CERE_FINISH_NONE);
    CONSTANT(CERE_FINISH_LENGTH);
    CONSTANT(CERE_FINISH_STOP_TOKEN);
    CONSTANT(CERE_FINISH_CANCELLED);
    CONSTANT(CERE_FINISH_ERROR);
    CONSTANT(CERE_TENSOR_MAX_DIMS);
    CONSTANT(CERE_MAX_TOP_LOGPROBS);
    CONSTANT(CERE_SAMPLING_KNOWN_FLAGS);
    CONSTANT(CERE_LEVER_K2_CUBLASLT);
    CONSTANT(CERE_LEVER_K2_SMALL_M);
    CONSTANT(CERE_LEVER_K2_SMALL_M_A8);
    CONSTANT(CERE_LEVER_K2_TAIL_SPLIT);
    CONSTANT(CERE_LEVER_PREFILL_CUDA_CORE);
    CONSTANT(CERE_LEVER_PREFILL_P4);
    CONSTANT(CERE_LEVER_DECODE_EAGER);
    CONSTANT(CERE_LEVER_DECODE_GRAPH);
    CONSTANT(CERE_LEVER_DECODE_SPEC_GRAPH);
    CONSTANT(CERE_LEVER_MOE_BUILD);
    CONSTANT(CERE_LEVER_MOE_P3B);
    CONSTANT(CERE_LEVER_MOE_K7);
    CONSTANT(CERE_LEVER_VERIFY_PER_ROW);
    CONSTANT(CERE_LEVER_VERIFY_BATCHED);
    CONSTANT(CERE_LEVER_K2_WEIGHTS_BF16);
    CONSTANT(CERE_LEVER_K2_WEIGHTS_FP8);
    CONSTANT(CERE_LEVER_SDAR_GEMM_IDENTICAL);
    CONSTANT(CERE_LEVER_SDAR_GEMM_CHANGING);
    CONSTANT(CERE_LEVER_DRAFT_WINDOW);
    CONSTANT(CERE_CALIBRATE_ALL);
    CONSTANT(CERE_CALIBRATE_MTP);
    CONSTANT(CERE_FAMILY_QWEN4_EXP);
    CONSTANT(CERE_FAMILY_SDAR);
    CONSTANT(CERE_FAMILY_INKLING2);
    CONSTANT(CERE_SLOT_K2_GEMM);
    CONSTANT(CERE_SLOT_PREFILL_ATTENTION);
    CONSTANT(CERE_SLOT_DECODE_STEP);
    CONSTANT(CERE_SLOT_MOE_GEMM);
    CONSTANT(CERE_SLOT_DRAFT_TOKENS);
    CONSTANT(CERE_SLOT_VERIFY_ATTENTION);
    CONSTANT(CERE_SLOT_K2_WEIGHTS);
    CONSTANT(CERE_SLOT_SDAR_GEMM);
    CONSTANT(CERE_SLOT_COUNT);
    CONSTANT(CERE_SLOT_BLOCK_VERSION);
    CONSTANT(CERE_SLOT_MAX);
    CONSTANT(CERE_SLOT_PARAMS);
    CONSTANT(CERE_MAX_PREFIX_STATE_SLOTS);
    CONSTANT(CERE_MAX_DECODE_ROUNDS_PER_CHUNK);
    CONSTANT(CERE_MAX_PREFILL_SLICE_MS);
    CONSTANT(CERE_DEFAULT_PREFILL_AGE_CALLS);
    CONSTANT(CERE_MAX_PREFILL_AGE_CALLS);
    CONSTANT(CERE_MAX_CONSTRAINED_DRAFTS);
    CONSTANT(CERE_LORA_MAX_ADAPTERS);
    CONSTANT(CERE_LORA_MAX_RANK);
    CONSTANT(CERE_LORA_MODULES);
    CONSTANT(CERE_LORA_Q_PROJ);
    CONSTANT(CERE_LORA_K_PROJ);
    CONSTANT(CERE_LORA_V_PROJ);
    CONSTANT(CERE_LORA_O_PROJ);
    CONSTANT(CERE_LORA_SHARED_GATE);
    CONSTANT(CERE_LORA_SHARED_UP);
    CONSTANT(CERE_LORA_SHARED_DOWN);
    CONSTANT(CERE_MAX_DECODE_SHARE_PCT);
    CONSTANT(CERE_RESEARCH_MAX_PLANS);
    CONSTANT(CERE_RESEARCH_MAX_OPS);
    CONSTANT(CERE_RESEARCH_MAX_RANGES);
    CONSTANT(CERE_RESEARCH_MAX_TOPK);
    CONSTANT(CERE_RESEARCH_MAX_LOGIT_POSITIONS);
    CONSTANT(CERE_RESEARCH_MAX_PLAN_BYTES);
    CONSTANT(CERE_RESEARCH_FINAL_LAYER);
    CONSTANT(CERE_RESEARCH_CAPTURE);
    CONSTANT(CERE_RESEARCH_ADD);
    CONSTANT(CERE_RESEARCH_ABLATE_ZERO);
    CONSTANT(CERE_RESEARCH_ABLATE_MEAN);
    CONSTANT(CERE_RESEARCH_REPLACE);
    CONSTANT(CERE_RESEARCH_EXPERT_BIAS);
    CONSTANT(CERE_RESEARCH_EXPERT_MASK);
    CONSTANT(CERE_RESEARCH_EXPERT_FORCE);
    CONSTANT(CERE_RESEARCH_DTYPE_BF16);
    CONSTANT(CERE_RESEARCH_DTYPE_I32);
    CONSTANT(CERE_RESEARCH_DTYPE_F32);
    end_list("  ", 0);
}

static void dump_structs(void) {
    begin_list("  ", "structs");
    STRUCT_BEGIN(cere_executor_options);
    STRUCT_FIELD(cere_executor_options, struct_size);
    STRUCT_FIELD(cere_executor_options, abi_version);
    STRUCT_FIELD(cere_executor_options, device_id);
    STRUCT_FIELD(cere_executor_options, worker_epoch);
    STRUCT_FIELD(cere_executor_options, memory_budget_bytes);
    STRUCT_END();

    STRUCT_BEGIN(cere_execution_capabilities);
    STRUCT_FIELD(cere_execution_capabilities, struct_size);
    STRUCT_FIELD(cere_execution_capabilities, abi_version);
    STRUCT_FIELD(cere_execution_capabilities, profile_generation);
    STRUCT_FIELD(cere_execution_capabilities, worker_epoch);
    STRUCT_FIELD(cere_execution_capabilities, decode_batch_min);
    STRUCT_FIELD(cere_execution_capabilities, decode_batch_max);
    STRUCT_FIELD(cere_execution_capabilities, max_context_tokens);
    STRUCT_FIELD(cere_execution_capabilities, execution_mode);
    STRUCT_FIELD(cere_execution_capabilities, native_decode_batch);
    STRUCT_FIELD(cere_execution_capabilities, native_mixed_prefill_decode);
    STRUCT_FIELD(cere_execution_capabilities, native_speculative_batch);
    STRUCT_FIELD(cere_execution_capabilities, native_moe_grouping);
    STRUCT_FIELD(cere_execution_capabilities, native_state_snapshot);
    STRUCT_FIELD(cere_execution_capabilities, native_sampler);
    STRUCT_FIELD(cere_execution_capabilities, native_token_mask);
    STRUCT_FIELD(cere_execution_capabilities, speculative_draft_tokens);
    STRUCT_FIELD(cere_execution_capabilities, native_prefix_cache);
    STRUCT_FIELD(cere_execution_capabilities, native_image_input);
    STRUCT_FIELD(cere_execution_capabilities, reserved);
    STRUCT_FIELD(cere_execution_capabilities, operations);
    STRUCT_FIELD(cere_execution_capabilities, kv_pool_tokens);
    STRUCT_FIELD(cere_execution_capabilities, kv_page_tokens);
    STRUCT_FIELD(cere_execution_capabilities, reserved2);
    STRUCT_END();

    STRUCT_BEGIN(cere_completion_ticket);
    STRUCT_FIELD(cere_completion_ticket, worker_epoch);
    STRUCT_FIELD(cere_completion_ticket, model_generation);
    STRUCT_FIELD(cere_completion_ticket, submission_id);
    STRUCT_END();

    STRUCT_BEGIN(cere_qwen4_exp_config);
    STRUCT_FIELD(cere_qwen4_exp_config, struct_size);
    STRUCT_FIELD(cere_qwen4_exp_config, abi_version);
    STRUCT_FIELD(cere_qwen4_exp_config, vocab_size);
    STRUCT_FIELD(cere_qwen4_exp_config, hidden_size);
    STRUCT_FIELD(cere_qwen4_exp_config, num_hidden_layers);
    STRUCT_FIELD(cere_qwen4_exp_config, full_attention_interval);
    STRUCT_FIELD(cere_qwen4_exp_config, full_attention_layers);
    STRUCT_FIELD(cere_qwen4_exp_config, num_attention_heads);
    STRUCT_FIELD(cere_qwen4_exp_config, num_key_value_heads);
    STRUCT_FIELD(cere_qwen4_exp_config, head_dim);
    STRUCT_FIELD(cere_qwen4_exp_config, output_gate_type);
    STRUCT_FIELD(cere_qwen4_exp_config, rope_theta);
    STRUCT_FIELD(cere_qwen4_exp_config, partial_rotary_factor);
    STRUCT_FIELD(cere_qwen4_exp_config, mrope_interleaved);
    STRUCT_FIELD(cere_qwen4_exp_config, mrope_section);
    STRUCT_FIELD(cere_qwen4_exp_config, indexer_n_heads);
    STRUCT_FIELD(cere_qwen4_exp_config, indexer_head_dim);
    STRUCT_FIELD(cere_qwen4_exp_config, indexer_kv_heads);
    STRUCT_FIELD(cere_qwen4_exp_config, indexer_compress_ratio);
    STRUCT_FIELD(cere_qwen4_exp_config, indexer_budget);
    STRUCT_FIELD(cere_qwen4_exp_config, linear_num_key_heads);
    STRUCT_FIELD(cere_qwen4_exp_config, linear_num_value_heads);
    STRUCT_FIELD(cere_qwen4_exp_config, linear_key_head_dim);
    STRUCT_FIELD(cere_qwen4_exp_config, linear_value_head_dim);
    STRUCT_FIELD(cere_qwen4_exp_config, linear_conv_kernel_dim);
    STRUCT_FIELD(cere_qwen4_exp_config, mamba_ssm_dtype);
    STRUCT_FIELD(cere_qwen4_exp_config, num_experts);
    STRUCT_FIELD(cere_qwen4_exp_config, num_experts_per_tok);
    STRUCT_FIELD(cere_qwen4_exp_config, moe_intermediate_size);
    STRUCT_FIELD(cere_qwen4_exp_config, shared_expert_intermediate_size);
    STRUCT_FIELD(cere_qwen4_exp_config, norm_topk_prob);
    STRUCT_FIELD(cere_qwen4_exp_config, hc_count);
    STRUCT_FIELD(cere_qwen4_exp_config, hc_lowrank);
    STRUCT_FIELD(cere_qwen4_exp_config, ple_layer_ids);
    STRUCT_FIELD(cere_qwen4_exp_config, ple_embed_dim);
    STRUCT_FIELD(cere_qwen4_exp_config, heads_per_ngram);
    STRUCT_FIELD(cere_qwen4_exp_config, ngram_size);
    STRUCT_FIELD(cere_qwen4_exp_config, ple_conv_kernel_size);
    STRUCT_FIELD(cere_qwen4_exp_config, ngram_vocab_size_base);
    STRUCT_FIELD(cere_qwen4_exp_config, split_ngram_parts);
    STRUCT_FIELD(cere_qwen4_exp_config, make_ngram_vocab_size_divisible_by);
    STRUCT_FIELD(cere_qwen4_exp_config, rms_norm_eps);
    STRUCT_FIELD(cere_qwen4_exp_config, tie_word_embeddings);
    STRUCT_FIELD(cere_qwen4_exp_config, mtp_num_hidden_layers);
    STRUCT_FIELD(cere_qwen4_exp_config, max_position_embeddings);
    STRUCT_END();

    STRUCT_BEGIN(cere_sdar_config);
    STRUCT_FIELD(cere_sdar_config, struct_size);
    STRUCT_FIELD(cere_sdar_config, abi_version);
    STRUCT_FIELD(cere_sdar_config, vocab_size);
    STRUCT_FIELD(cere_sdar_config, hidden_size);
    STRUCT_FIELD(cere_sdar_config, intermediate_size);
    STRUCT_FIELD(cere_sdar_config, num_hidden_layers);
    STRUCT_FIELD(cere_sdar_config, num_attention_heads);
    STRUCT_FIELD(cere_sdar_config, num_key_value_heads);
    STRUCT_FIELD(cere_sdar_config, head_dim);
    STRUCT_FIELD(cere_sdar_config, block_size);
    STRUCT_FIELD(cere_sdar_config, rope_theta);
    STRUCT_FIELD(cere_sdar_config, rms_norm_eps);
    STRUCT_FIELD(cere_sdar_config, mask_token_id);
    STRUCT_FIELD(cere_sdar_config, tie_word_embeddings);
    STRUCT_FIELD(cere_sdar_config, max_position_embeddings);
    STRUCT_END();

    STRUCT_BEGIN(cere_inkling2_config);
    STRUCT_FIELD(cere_inkling2_config, struct_size);
    STRUCT_FIELD(cere_inkling2_config, abi_version);
    STRUCT_FIELD(cere_inkling2_config, vocab_size);
    STRUCT_FIELD(cere_inkling2_config, unpadded_vocab_size);
    STRUCT_FIELD(cere_inkling2_config, hidden_size);
    STRUCT_FIELD(cere_inkling2_config, num_hidden_layers);
    STRUCT_FIELD(cere_inkling2_config, num_attention_heads);
    STRUCT_FIELD(cere_inkling2_config, num_key_value_heads);
    STRUCT_FIELD(cere_inkling2_config, head_dim);
    STRUCT_FIELD(cere_inkling2_config, moe_intermediate_size);
    STRUCT_FIELD(cere_inkling2_config, num_experts);
    STRUCT_FIELD(cere_inkling2_config, num_experts_per_tok);
    STRUCT_FIELD(cere_inkling2_config, sliding_window_size);
    STRUCT_FIELD(cere_inkling2_config, rel_extent);
    STRUCT_FIELD(cere_inkling2_config, d_rel);
    STRUCT_FIELD(cere_inkling2_config, loop_core_start);
    STRUCT_FIELD(cere_inkling2_config, loop_core_end);
    STRUCT_FIELD(cere_inkling2_config, loop_num_passes);
    STRUCT_FIELD(cere_inkling2_config, loop_gate_rank);
    STRUCT_FIELD(cere_inkling2_config, n_layers_per_attn_res_block);
    STRUCT_FIELD(cere_inkling2_config, conv_kernel_size);
    STRUCT_FIELD(cere_inkling2_config, eos_token_id);
    STRUCT_FIELD(cere_inkling2_config, local_layer_mask);
    STRUCT_FIELD(cere_inkling2_config, mxfp8_layer_mask);
    STRUCT_FIELD(cere_inkling2_config, rms_norm_eps);
    STRUCT_FIELD(cere_inkling2_config, final_logit_softcapping);
    STRUCT_FIELD(cere_inkling2_config, route_scale);
    STRUCT_FIELD(cere_inkling2_config, max_position_embeddings);
    STRUCT_END();

    STRUCT_BEGIN(cere_qwen4_exp_vision_config);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, struct_size);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, abi_version);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, depth);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, hidden_size);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, intermediate_size);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, num_heads);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, patch_size);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, temporal_patch_size);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, spatial_merge_size);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, in_channels);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, out_hidden_size);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, num_position_embeddings);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, deepstack_count);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, hidden_act);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, image_token_id);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, video_token_id);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, vision_start_token_id);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, vision_end_token_id);
    STRUCT_FIELD(cere_qwen4_exp_vision_config, layer_norm_eps);
    STRUCT_END();

    STRUCT_BEGIN(cere_tensor_desc);
    STRUCT_FIELD(cere_tensor_desc, struct_size);
    STRUCT_FIELD(cere_tensor_desc, abi_version);
    STRUCT_FIELD(cere_tensor_desc, name);
    STRUCT_FIELD(cere_tensor_desc, dtype);
    STRUCT_FIELD(cere_tensor_desc, placement);
    STRUCT_FIELD(cere_tensor_desc, ndim);
    STRUCT_FIELD(cere_tensor_desc, reserved);
    STRUCT_FIELD(cere_tensor_desc, shape);
    STRUCT_FIELD(cere_tensor_desc, bytes);
    STRUCT_END();

    STRUCT_BEGIN(cere_slot_choice);
    STRUCT_FIELD(cere_slot_choice, slot);
    STRUCT_FIELD(cere_slot_choice, impl);
    STRUCT_FIELD(cere_slot_choice, params);
    STRUCT_END();

    STRUCT_BEGIN(cere_slot_selection);
    STRUCT_FIELD(cere_slot_selection, struct_size);
    STRUCT_FIELD(cere_slot_selection, abi_version);
    STRUCT_FIELD(cere_slot_selection, block_version);
    STRUCT_FIELD(cere_slot_selection, count);
    STRUCT_FIELD(cere_slot_selection, choices);
    STRUCT_END();

    STRUCT_BEGIN(cere_model_desc);
    STRUCT_FIELD(cere_model_desc, struct_size);
    STRUCT_FIELD(cere_model_desc, abi_version);
    STRUCT_FIELD(cere_model_desc, model_kind);
    STRUCT_FIELD(cere_model_desc, speculative_draft_tokens);
    STRUCT_FIELD(cere_model_desc, config);
    STRUCT_FIELD(cere_model_desc, tensors);
    STRUCT_FIELD(cere_model_desc, tensor_count);
    STRUCT_FIELD(cere_model_desc, max_sessions);
    STRUCT_FIELD(cere_model_desc, max_context_tokens);
    STRUCT_FIELD(cere_model_desc, stop_token_ids);
    STRUCT_FIELD(cere_model_desc, stop_token_count);
    STRUCT_FIELD(cere_model_desc, kv_pool_tokens);
    STRUCT_FIELD(cere_model_desc, draft_token_map);
    STRUCT_FIELD(cere_model_desc, draft_token_map_count);
    STRUCT_FIELD(cere_model_desc, snapshot_host_bytes);
    STRUCT_FIELD(cere_model_desc, prefix_state_slots);
    STRUCT_FIELD(cere_model_desc, vision_config);
    STRUCT_FIELD(cere_model_desc, selection);
    STRUCT_FIELD(cere_model_desc, family);
    STRUCT_FIELD(cere_model_desc, calibrate_windows);
    STRUCT_END();

    STRUCT_BEGIN(cere_token_score);
    STRUCT_FIELD(cere_token_score, id);
    STRUCT_FIELD(cere_token_score, runner_up);
    STRUCT_FIELD(cere_token_score, logit);
    STRUCT_FIELD(cere_token_score, runner_up_logit);
    STRUCT_END();

    STRUCT_BEGIN(cere_logit_bias);
    STRUCT_FIELD(cere_logit_bias, id);
    STRUCT_FIELD(cere_logit_bias, bias);
    STRUCT_END();

    STRUCT_BEGIN(cere_sampling_params);
    STRUCT_FIELD(cere_sampling_params, struct_size);
    STRUCT_FIELD(cere_sampling_params, abi_version);
    STRUCT_FIELD(cere_sampling_params, temperature);
    STRUCT_FIELD(cere_sampling_params, top_p);
    STRUCT_FIELD(cere_sampling_params, min_p);
    STRUCT_FIELD(cere_sampling_params, top_k);
    STRUCT_FIELD(cere_sampling_params, frequency_penalty);
    STRUCT_FIELD(cere_sampling_params, presence_penalty);
    STRUCT_FIELD(cere_sampling_params, repetition_penalty);
    STRUCT_FIELD(cere_sampling_params, flags);
    STRUCT_FIELD(cere_sampling_params, seed);
    STRUCT_FIELD(cere_sampling_params, logit_bias);
    STRUCT_FIELD(cere_sampling_params, logit_bias_count);
    STRUCT_FIELD(cere_sampling_params, suppress_token_ids);
    STRUCT_FIELD(cere_sampling_params, suppress_token_count);
    STRUCT_FIELD(cere_sampling_params, mask_until_committed);
    STRUCT_END();

    STRUCT_BEGIN(cere_logprob_entry);
    STRUCT_FIELD(cere_logprob_entry, id);
    STRUCT_FIELD(cere_logprob_entry, logprob);
    STRUCT_END();

    STRUCT_BEGIN(cere_token_logprobs);
    STRUCT_FIELD(cere_token_logprobs, id);
    STRUCT_FIELD(cere_token_logprobs, logprob);
    STRUCT_FIELD(cere_token_logprobs, top_count);
    STRUCT_FIELD(cere_token_logprobs, reserved);
    STRUCT_FIELD(cere_token_logprobs, top);
    STRUCT_END();

    STRUCT_BEGIN(cere_image_input);
    STRUCT_FIELD(cere_image_input, struct_size);
    STRUCT_FIELD(cere_image_input, abi_version);
    STRUCT_FIELD(cere_image_input, grid_t);
    STRUCT_FIELD(cere_image_input, grid_h);
    STRUCT_FIELD(cere_image_input, grid_w);
    STRUCT_FIELD(cere_image_input, pad_id);
    STRUCT_FIELD(cere_image_input, token_offset);
    STRUCT_FIELD(cere_image_input, pixel_patches);
    STRUCT_FIELD(cere_image_input, pixel_bytes);
    STRUCT_FIELD(cere_image_input, pixel_sha256);
    STRUCT_END();

    STRUCT_BEGIN(cere_work);
    STRUCT_FIELD(cere_work, struct_size);
    STRUCT_FIELD(cere_work, abi_version);
    STRUCT_FIELD(cere_work, operation);
    STRUCT_FIELD(cere_work, flags);
    STRUCT_FIELD(cere_work, expected_worker_epoch);
    STRUCT_FIELD(cere_work, expected_session_revision);
    STRUCT_FIELD(cere_work, session);
    STRUCT_FIELD(cere_work, tokens);
    STRUCT_FIELD(cere_work, token_count);
    STRUCT_FIELD(cere_work, max_new_tokens);
    STRUCT_FIELD(cere_work, reserved_peak_bytes);
    STRUCT_FIELD(cere_work, out_tokens);
    STRUCT_FIELD(cere_work, out_capacity);
    STRUCT_FIELD(cere_work, out_scores);
    STRUCT_FIELD(cere_work, stop_token_ids);
    STRUCT_FIELD(cere_work, stop_token_count);
    STRUCT_FIELD(cere_work, sampling);
    STRUCT_FIELD(cere_work, out_logprobs);
    STRUCT_FIELD(cere_work, top_logprobs);
    STRUCT_FIELD(cere_work, capture_position);
    STRUCT_FIELD(cere_work, vocab_mask);
    STRUCT_FIELD(cere_work, vocab_mask_words);
    STRUCT_FIELD(cere_work, images);
    STRUCT_FIELD(cere_work, image_count);
    STRUCT_END();

    STRUCT_BEGIN(cere_batch);
    STRUCT_FIELD(cere_batch, struct_size);
    STRUCT_FIELD(cere_batch, abi_version);
    STRUCT_FIELD(cere_batch, expected_worker_epoch);
    STRUCT_FIELD(cere_batch, capability_profile_generation);
    STRUCT_FIELD(cere_batch, items);
    STRUCT_FIELD(cere_batch, item_count);
    STRUCT_FIELD(cere_batch, reserved_peak_bytes);
    STRUCT_END();

    STRUCT_BEGIN(cere_completion);
    STRUCT_FIELD(cere_completion, struct_size);
    STRUCT_FIELD(cere_completion, abi_version);
    STRUCT_FIELD(cere_completion, status);
    STRUCT_FIELD(cere_completion, terminal);
    STRUCT_FIELD(cere_completion, committed_tokens);
    STRUCT_FIELD(cere_completion, session_revision);
    STRUCT_FIELD(cere_completion, released_bytes);
    STRUCT_FIELD(cere_completion, finish);
    STRUCT_FIELD(cere_completion, cached_tokens);
    STRUCT_FIELD(cere_completion, verify_steps);
    STRUCT_FIELD(cere_completion, accepted_drafts);
    STRUCT_END();

    STRUCT_BEGIN(cere_snapshot_info);
    STRUCT_FIELD(cere_snapshot_info, struct_size);
    STRUCT_FIELD(cere_snapshot_info, abi_version);
    STRUCT_FIELD(cere_snapshot_info, bytes);
    STRUCT_FIELD(cere_snapshot_info, position);
    STRUCT_FIELD(cere_snapshot_info, token_count);
    STRUCT_FIELD(cere_snapshot_info, fed);
    STRUCT_FIELD(cere_snapshot_info, layout_id);
    STRUCT_END();

    STRUCT_BEGIN(cere_prefix_stats);
    STRUCT_FIELD(cere_prefix_stats, struct_size);
    STRUCT_FIELD(cere_prefix_stats, abi_version);
    STRUCT_FIELD(cere_prefix_stats, alt_edges_on);
    STRUCT_FIELD(cere_prefix_stats, reserved);
    STRUCT_FIELD(cere_prefix_stats, hits);
    STRUCT_FIELD(cere_prefix_stats, captures);
    STRUCT_FIELD(cere_prefix_stats, capture_skipped);
    STRUCT_FIELD(cere_prefix_stats, capture_page_mismatch);
    STRUCT_FIELD(cere_prefix_stats, inserts);
    STRUCT_FIELD(cere_prefix_stats, state_evictions);
    STRUCT_FIELD(cere_prefix_stats, leaf_evictions);
    STRUCT_FIELD(cere_prefix_stats, host_spills);
    STRUCT_FIELD(cere_prefix_stats, host_loads);
    STRUCT_FIELD(cere_prefix_stats, host_drops);
    STRUCT_FIELD(cere_prefix_stats, alt_edges);
    STRUCT_END();

    STRUCT_BEGIN(cere_prefix_capture_stats);
    STRUCT_FIELD(cere_prefix_capture_stats, struct_size);
    STRUCT_FIELD(cere_prefix_capture_stats, abi_version);
    STRUCT_FIELD(cere_prefix_capture_stats, skipped_prefill_slot);
    STRUCT_FIELD(cere_prefix_capture_stats, skipped_decode_track);
    STRUCT_FIELD(cere_prefix_capture_stats, skipped_end_pair);
    STRUCT_FIELD(cere_prefix_capture_stats, hit_shortfall_tokens);
    STRUCT_FIELD(cere_prefix_capture_stats, hit_shortfall_lookups);
    STRUCT_END();
    STRUCT_BEGIN(cere_research_range);
    STRUCT_FIELD(cere_research_range, first);
    STRUCT_FIELD(cere_research_range, end);
    STRUCT_END();
    STRUCT_BEGIN(cere_research_op);
    STRUCT_FIELD(cere_research_op, kind);
    STRUCT_FIELD(cere_research_op, top_k);
    STRUCT_FIELD(cere_research_op, point);
    STRUCT_FIELD(cere_research_op, layers);
    STRUCT_FIELD(cere_research_op, prompt);
    STRUCT_FIELD(cere_research_op, prompt_count);
    STRUCT_FIELD(cere_research_op, generated);
    STRUCT_FIELD(cere_research_op, generated_max);
    STRUCT_FIELD(cere_research_op, alpha);
    STRUCT_FIELD(cere_research_op, col_begin);
    STRUCT_FIELD(cere_research_op, col_end);
    STRUCT_FIELD(cere_research_op, vector_rows);
    STRUCT_FIELD(cere_research_op, vector);
    STRUCT_FIELD(cere_research_op, vector_cols);
    STRUCT_FIELD(cere_research_op, expert_count);
    STRUCT_FIELD(cere_research_op, experts);
    STRUCT_FIELD(cere_research_op, weights);
    STRUCT_END();
    STRUCT_BEGIN(cere_research_plan);
    STRUCT_FIELD(cere_research_plan, ops);
    STRUCT_FIELD(cere_research_plan, op_count);
    STRUCT_FIELD(cere_research_plan, reserved);
    STRUCT_FIELD(cere_research_plan, sha256);
    STRUCT_END();
    STRUCT_BEGIN(cere_research_tensor);
    STRUCT_FIELD(cere_research_tensor, name);
    STRUCT_FIELD(cere_research_tensor, op);
    STRUCT_FIELD(cere_research_tensor, layer);
    STRUCT_FIELD(cere_research_tensor, dtype);
    STRUCT_FIELD(cere_research_tensor, cols);
    STRUCT_FIELD(cere_research_tensor, rows);
    STRUCT_FIELD(cere_research_tensor, positions);
    STRUCT_FIELD(cere_research_tensor, data);
    STRUCT_END();
    STRUCT_BEGIN(cere_research_capture);
    STRUCT_FIELD(cere_research_capture, tensors);
    STRUCT_FIELD(cere_research_capture, tensor_count);
    STRUCT_FIELD(cere_research_capture, bytes);
    STRUCT_FIELD(cere_research_capture, prefix_lookup_skipped);
    STRUCT_FIELD(cere_research_capture, reserved);
    STRUCT_FIELD(cere_research_capture, plan_sha256);
    STRUCT_END();
    end_list("  ", 0);
}

static void dump_functions(void) {
    static const char *const names[] = {
        "cere_executor_abi_version", "cere_executor_create", "cere_executor_destroy", "cere_executor_capabilities",
        "cere_model_create",         "cere_model_put",       "cere_model_seal",       "cere_model_release",
        "cere_session_create",       "cere_session_release", "cere_executor_submit",  "cere_executor_cancel",
        "cere_executor_poll",        "cere_executor_constrain", "cere_session_snapshot", "cere_session_restore",
        "cere_snapshot_get_info",    "cere_snapshot_read",      "cere_snapshot_import",  "cere_snapshot_release",
        "cere_session_set_cache_namespace", "cere_model_flush_prefix_cache", "cere_prefix_cache_export",
        "cere_prefix_cache_import",          "cere_model_host_resident",      "cere_model_selection",
        "cere_model_set_selection",  "cere_executor_register_staging", "cere_executor_unregister_staging",
        "cere_executor_set_decode_rounds",
        "cere_model_set_prefix_end_state",
        "cere_model_set_prefix_alt_edges",
        "cere_model_prefix_stats",
        "cere_executor_set_prefill_slice",
        "cere_executor_round_drafts",
        "cere_executor_constrain_drafts",
        "cere_model_lora_reserve",   "cere_model_lora_load", "cere_model_lora_unload", "cere_session_set_lora",
        "cere_model_prefix_capture_stats",
        "cere_executor_set_decode_share",
        "cere_model_research_reserve",
        "cere_model_research_plan_load",
        "cere_model_research_plan_unload",
        "cere_session_set_research",
        "cere_model_research_capture_read",
        "cere_model_research_capture_free",
        "cere_model_prefill_signature",
        "cere_executor_set_prefill_age",
    };
    size_t i;
    begin_list("  ", "functions");
    for (i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        item_prefix();
        printf("    \"%s\"", names[i]);
    }
    end_list("  ", 1);
}

int main(void) {
    printf("{\n  \"abi_version\": %u,\n", (unsigned)CERE_EXECUTOR_ABI_VERSION);
    dump_constants();
    dump_structs();
    dump_functions();
    printf("}\n");
    return 0;
}
