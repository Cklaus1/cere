#ifndef CERE_EXECUTOR_H
#define CERE_EXECUTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI v5 (breaking vs v4; SelfTune ST2, tasks/selftune-spec.md D3/D4, Q5(a)): the perf levers become a versioned
 * slot-selection block. cere_levers is replaced by cere_slot_selection (a list of {slot, impl, params} choices, every
 * slot not listed at its default), cere_model_desc.levers becomes `selection` (same offset) and the desc grows by
 * `family` (the model family whose config `config` points at; only CERE_FAMILY_QWEN4_EXP) and `calibrate_windows`
 * (the former cere_levers field, a create-time choice, not a slot): 128 -> 136 B. The two lever functions keep
 * their export positions under new names: cere_model_selection (read back the active selection) and
 * cere_model_set_selection (dev/bench hot swap). A v4 caller is refused cleanly: its 128-B desc fails the
 * struct_size check (CERE_ERR_ARGUMENT) and abi_version 4 in any struct is CERE_ERR_UNSUPPORTED. Migration note:
 * executor/README.md. Layout: tests/fixtures/executor_abi_v5_layout.json (x86_64 LP64).
 *
 * v4 (breaking vs v3; PerfInt lever registry): cere_model_desc grows by `levers` (120 -> 128 B), a pointer to
 * the new cere_levers (null = every lever at its default, i.e. v3's behaviour), and two functions are appended to the
 * export order: cere_model_levers (read back the active levers) and cere_model_set_levers (dev/bench hot swap).
 * Every perf lever the library implements is selected at runtime by these values, not by a CMake option: the one
 * library runs every selection (only the MoE backend stays a build variant, which the moe_gemm lever checks). A v3
 * caller is refused cleanly: its 120-B desc fails the struct_size check (CERE_ERR_ARGUMENT) and abi_version 3 in any
 * struct is CERE_ERR_UNSUPPORTED.
 *
 * v3 (breaking vs v2): staged model creation (cere_model_create/put/seal replace the manifest-path loader), a typed
 * model config, token output buffers leased to completion, a finish reason, per-item stop tokens and an `operations`
 * bitmask in the capabilities.
 *
 * v3.L frame (tasks/native-abi-ledger.md, L0): the four shared structs carry their final layout
 * (cere_execution_capabilities 88 B, cere_model_desc 120 B (128 B since v4, 136 B since v5), cere_work 168 B,
 * cere_completion 64 B). A
 * field whose owner has not landed must be zero/null on input (non-zero: CERE_ERR_UNSUPPORTED) and reads 0 on output.
 * Each struct has exactly one valid struct_size, sizeof(T) of this header. */
#define CERE_EXECUTOR_ABI_VERSION UINT32_C(5)

/* Status codes. Every function returns one of these; on error every output stays null/zero. */
typedef int32_t cere_status;
/* Success; from cere_executor_poll, the completion is terminal and its ticket is released. */
#define CERE_OK INT32_C(0)
/* cere_executor_poll only: the work is not terminal yet; the completion carries the progress so far. */
#define CERE_PENDING INT32_C(1)
/* Null or foreign handle, struct_size other than sizeof(T) of this header, malformed descriptor,
 * overlapping/out-of-range/post-seal put, seal with a tiling gap, unknown or already released ticket, misuse of an
 * operation, a cere_work.flags bit outside CERE_WORK_KNOWN_FLAGS. */
#define CERE_ERR_ARGUMENT INT32_C(-1)
/* Wrong abi_version, a model config value, tensor dtype or batch shape outside the compiled support table, or an
 * operation this library does not implement (e.g. a build without a forward pass). */
#define CERE_ERR_UNSUPPORTED INT32_C(-2)
/* The memory estimate exceeds the device or host budget (nothing is allocated), the session/context limits are
 * exhausted, or an allocation failed. */
#define CERE_ERR_CAPACITY INT32_C(-3)
/* A worker epoch, model generation or session revision that is not the current one. */
#define CERE_ERR_STALE_EPOCH INT32_C(-4)
/* The object is in use: destroy/release with an outstanding ticket or live dependants, submit on a session that has
 * work in flight, seal while puts are in flight. */
#define CERE_ERR_BUSY INT32_C(-5)
/* An internal or device failure; also reported as cere_completion.status with finish CERE_FINISH_ERROR. */
#define CERE_ERR_INTERNAL INT32_C(-6)

/* cere_work.operation. VERIFY (M2.9 I4) needs a model with speculative_draft_tokens != 0 (else
 * CERE_ERR_UNSUPPORTED): `tokens` are 1..speculative_draft_tokens - 1 caller drafts for a decode-ready drafting
 * session (else CERE_ERR_ARGUMENT); one speculation round commits the drafts the model accepts in order plus its own
 * next token (the correction, or the bonus token after all accepted), then the ticket ends (LENGTH unless a stop
 * token finished it). Greedy: the same tokens a plain DECODE would commit. Sampled (M2.9 D6): each draft is kept with
 * its probability under the row's sampler distribution, else the token is drawn from that distribution without the
 * draft, so every committed token has plain sampling's distribution at its position (seeded output may differ from a
 * plain run's); each committed token's cere_token_score and logprobs are what plain sampled decoding reports for that
 * token at that position, and only committed tokens enter the output counts. */
#define CERE_OP_PREFILL UINT32_C(1)
#define CERE_OP_DECODE UINT32_C(2)
#define CERE_OP_VERIFY UINT32_C(3)
/* cere_execution_capabilities.operations: bit (1 << CERE_OP_*) per work operation, plus the non-work operations
 * cancel, snapshot and restore (M2.11: set iff cere_model_desc.snapshot_host_bytes > 0). */
#define CERE_OPS_PREFILL UINT64_C(0x2)
#define CERE_OPS_DECODE UINT64_C(0x4)
#define CERE_OPS_VERIFY UINT64_C(0x8)
#define CERE_OPS_CANCEL UINT64_C(0x10)
#define CERE_OPS_SNAPSHOT UINT64_C(0x20)
#define CERE_OPS_RESTORE UINT64_C(0x40)

/* cere_work.flags (ledger table F). Bit 0 (M2.12): no stop token (model or item list) finishes this ticket; the
 * budget and the context limit still do; VERIFY with bit 0 is CERE_ERR_ARGUMENT. Bit 1 (M2.9) is read per ticket: a
 * session drafts iff its model has speculative_draft_tokens != 0 and its PREFILL lacks bit 1 (image work carries
 * it); a DECODE with bit 1 on a drafting session runs its rounds with no drafts (one token per round, still followed
 * by the draft-extend step, D9), and VERIFY with bit 1, or on a session that does not draft, is CERE_ERR_ARGUMENT.
 * Stop-ignoring tickets also run with no drafts, and so do constrained (vocab_mask) tickets unless the caller masks
 * a round's drafts with cere_executor_constrain_drafts (GrammarSpec). Bit 2 (M2.9L, tasks/m29l-spec.md D5):
 * a ticket that drafts may verify suffix-lookup drafts (the tokens that followed the longest earlier occurrence of the
 * session's current suffix in its prompt and committed tokens) instead of the MTP proposals, chosen per round by the
 * executor's per-session policy; bit 3: the ticket never verifies MTP proposals (with bit 2: lookup drafts only;
 * without: no drafts). Either way the committed tokens are what VERIFY's caller drafts would commit (greedy: plain
 * decode's). Bits 2 and 3 do nothing on a ticket without drafts; VERIFY with bit 2 or 3 is CERE_ERR_ARGUMENT. Bit 4
 * (tasks/m29a-spec.md A2): a ticket that drafts lets its session's policy pick each round's draft count (and, with bit
 * 2, the source) from the request's own acceptance, pricing the windows with the executor's fixed calibrated round
 * costs instead of measured round times, so the picks depend only on the token stream; the committed tokens are
 * still plain decode's (greedy). Bit 4 does nothing on a ticket without drafts; VERIFY with bit 4 is
 * CERE_ERR_ARGUMENT. Bit 5 (DraftAccept): on a greedy ticket with bits 2 and 4 (not 3), when the suffix lookup agrees
 * with all three pending MTP proposals, the policy may verify the lookup's continuation past them (up to 8 drafts in
 * one chain) when its expected tokens per calibrated cost pay more; the committed tokens are still plain decode's.
 * Bit 5 does nothing otherwise; VERIFY with bit 5 is CERE_ERR_ARGUMENT. Bit 6 (TreeSpec): on a greedy ticket with bits
 * 2 and 4, the policy may verify a long suffix-lookup proposal (up to 8 drafts in one chain) on its own, without MTP
 * agreement, when its expected tokens per calibrated cost pay more; the committed tokens are still plain decode's.
 * Bit 6 does nothing otherwise; VERIFY with bit 6 is CERE_ERR_ARGUMENT. Bit 7 (Chains): the chains of bits 5 and 6
 * may hold up to the model's longest verify chain (up to 15 drafts, bounded by its undo snapshots: 3 per max_sessions)
 * instead of 8, priced the same way; the committed tokens are still plain decode's. Bit 7 does nothing otherwise;
 * VERIFY with bit 7 is CERE_ERR_ARGUMENT. Any other bit is CERE_ERR_ARGUMENT. */
#define CERE_WORK_IGNORE_STOP_TOKENS UINT32_C(0x1)
#define CERE_WORK_NO_SPECULATION UINT32_C(0x2)
#define CERE_WORK_LOOKUP_DRAFTS UINT32_C(0x4)
#define CERE_WORK_NO_MTP_DRAFTS UINT32_C(0x8)
#define CERE_WORK_ADAPTIVE_DRAFTS UINT32_C(0x10)
#define CERE_WORK_TAIL_DRAFTS UINT32_C(0x20)
#define CERE_WORK_LONG_DRAFTS UINT32_C(0x40)
#define CERE_WORK_CHAIN_DRAFTS UINT32_C(0x80)
#define CERE_WORK_KNOWN_FLAGS UINT32_C(0xFF)

/* cere_work.top_logprobs upper bound (ledger table K). */
#define CERE_MAX_TOP_LOGPROBS UINT32_C(20)
/* cere_sampling_params.flags (ledger table F2): no bits are assigned; any set bit is CERE_ERR_ARGUMENT. */
#define CERE_SAMPLING_KNOWN_FLAGS UINT32_C(0)

/* Mirrors execution_capabilities.execution_mode; zero is never a valid mode. */
typedef uint32_t cere_execution_mode;
#define CERE_EXECUTION_MODE_NATIVE UINT32_C(1)
#define CERE_EXECUTION_MODE_BACKEND_MANAGED UINT32_C(2)
#define CERE_EXECUTION_MODE_SERIAL UINT32_C(3)
#define CERE_EXECUTION_MODE_UNSUPPORTED UINT32_C(4)

/* cere_model_desc.model_kind: the weight format; each family has one (ST4: SDAR's checkpoint is plain BF16). */
#define CERE_MODEL_QWEN4_EXP_NVFP4 UINT32_C(1)
#define CERE_MODEL_SDAR_BF16 UINT32_C(2)
/* Inkling 2 Mini (tasks/inkling-spec.md section 2): NVFP4 experts (layers 0-19), MXFP8 experts (20-23), BF16/F32
 * rest, stored as the checkpoint has them. 3 is build/qwen35-4b's CERE_MODEL_QWEN3_5_BF16. */
#define CERE_MODEL_INKLING2_NVFP4 UINT32_C(4)
/* cere_model_desc.family (ABI v5): the model family, which names the config struct `config` points at. A library is
 * built for one model (SelfTune ST4, `gen::kFamily` of its descriptor) and accepts only that family with its
 * model_kind; any other family or kind is CERE_ERR_UNSUPPORTED. CERE_FAMILY_QWEN4_EXP: a cere_qwen4_exp_config;
 * CERE_FAMILY_SDAR (DICE-8B, block diffusion over Qwen3-8B dense layers): a cere_sdar_config; CERE_FAMILY_INKLING2
 * (Inkling 2 Mini, `inkling2_loop`): a cere_inkling2_config; each passed through the same `config` pointer (its
 * leading struct_size / abi_version fields name the struct). Value 2 is build/qwen35-4b's CERE_FAMILY_QWEN3_5. */
#define CERE_FAMILY_QWEN4_EXP UINT32_C(0)
#define CERE_FAMILY_SDAR UINT32_C(1)
#define CERE_FAMILY_INKLING2 UINT32_C(3)

/* cere_tensor_desc.placement */
#define CERE_PLACEMENT_DEVICE UINT32_C(1)
#define CERE_PLACEMENT_HOST_PINNED UINT32_C(2)
#define CERE_PLACEMENT_DISCARD UINT32_C(3) /* staged by the caller, only counted by the executor */

/* Tensor dtypes (safetensors names); also cere_qwen4_exp_config.mamba_ssm_dtype. */
#define CERE_DTYPE_BF16 UINT32_C(1)
#define CERE_DTYPE_F16 UINT32_C(2)
#define CERE_DTYPE_F32 UINT32_C(3)
#define CERE_DTYPE_F64 UINT32_C(4)
#define CERE_DTYPE_F8_E4M3 UINT32_C(5)
#define CERE_DTYPE_F8_E5M2 UINT32_C(6)
#define CERE_DTYPE_U8 UINT32_C(7)
#define CERE_DTYPE_I8 UINT32_C(8)
#define CERE_DTYPE_I16 UINT32_C(9)
#define CERE_DTYPE_I32 UINT32_C(10)
#define CERE_DTYPE_I64 UINT32_C(11)
#define CERE_DTYPE_BOOL UINT32_C(12)

/* cere_qwen4_exp_config.output_gate_type ("sigmoid"; an absent key is 0). */
#define CERE_OUTPUT_GATE_SIGMOID UINT32_C(1)

/* cere_qwen4_exp_vision_config.hidden_act (ledger K): GELU_TANH the tanh approximation (Qwen4Exp), GELU_ERF the exact
 * erf GELU (Inkling 2 Mini's vision tower, tasks/inkling-spec.md S6g) */
#define CERE_ACT_GELU_TANH UINT32_C(1)
#define CERE_ACT_GELU_ERF UINT32_C(2)

/* cere_completion.finish */
#define CERE_FINISH_NONE UINT32_C(0)       /* not terminal */
#define CERE_FINISH_LENGTH UINT32_C(1)     /* max_new_tokens or the session context limit reached */
#define CERE_FINISH_STOP_TOKEN UINT32_C(2) /* the stop token is committed and counted */
#define CERE_FINISH_CANCELLED UINT32_C(3)  /* committed tokens before the cancel stay valid */
#define CERE_FINISH_ERROR UINT32_C(4)      /* cere_completion.status carries the error */

#define CERE_TENSOR_MAX_DIMS 8

/* Opaque handles; a handle is valid only with the executor that created it. */
typedef struct cere_executor cere_executor;
typedef struct cere_model cere_model;
typedef struct cere_session cere_session;
/* M2.11: a session snapshot in the model's pinned host arena (cere_model_desc.snapshot_host_bytes). */
typedef struct cere_snapshot cere_snapshot;

/* Ledger-owned types: declared here, defined by their owner as `struct <tag> {...};`. Until then the pointers that
 * use them must be null. cere_sampling_params and cere_token_logprobs are defined below (M2.7). */
typedef struct cere_sampling_params cere_sampling_params;
typedef struct cere_token_logprobs cere_token_logprobs;
typedef struct cere_image_input cere_image_input;
typedef struct cere_qwen4_exp_vision_config cere_qwen4_exp_vision_config;

typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    const char *device_id; /* copied by create; caller retains ownership */
    uint64_t worker_epoch;
    uint64_t memory_budget_bytes; /* device bytes this executor may allocate */
} cere_executor_options;

typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t profile_generation; /* the model generation; cere_batch.capability_profile_generation must equal it */
    uint64_t worker_epoch;
    /* Decode rows of the model's plan (M2.6 D3): decode_batch_min 1, decode_batch_max = cere_model_desc.max_sessions
     * (R), native_decode_batch = 1 when R > 1 and the library has a forward pass; rows form across tickets (one
     * cere_work per batch), native_mixed_prefill_decode stays 0. */
    uint64_t decode_batch_min;
    uint64_t decode_batch_max;
    uint64_t max_context_tokens;
    cere_execution_mode execution_mode; /* one of CERE_EXECUTION_MODE_* */
    uint8_t native_decode_batch;
    uint8_t native_mixed_prefill_decode;
    uint8_t native_speculative_batch; /* M2.9: 1 iff the model has a draft and the library a forward pass */
    uint8_t native_moe_grouping;
    uint8_t native_state_snapshot;
    uint8_t native_sampler;           /* 1 = the device sampler is present (cere_work.sampling/out_logprobs) */
    uint8_t native_token_mask;        /* M2.8: 1 = cere_work.vocab_mask and cere_executor_constrain (forward present) */
    uint8_t speculative_draft_tokens; /* M2.9: the model's cere_model_desc value (0 or 4) under the same rule */
    uint8_t native_prefix_cache;      /* M2.10: 1 when the model was created with prefix_state_slots > 0 */
    uint8_t native_image_input;       /* 1 when the model has vision_config (M2.13) */
    uint8_t reserved[2];
    uint64_t operations; /* CERE_OPS_* bits; compared with the deployment capabilities' operations */
    uint64_t kv_pool_tokens; /* M2.5: paged KV pool tokens allocated (multiple of kv_page_tokens) */
    uint32_t kv_page_tokens; /* M2.5: tokens per KV page, 64 */
    uint32_t reserved2;
} cere_execution_capabilities;

typedef struct {
    uint64_t worker_epoch;
    uint64_t model_generation; /* model instance generation that accepted the submission */
    uint64_t submission_id;
} cere_completion_ticket;

/* The model's `text_config` values (config.json), fixed width. A key absent from config.json is 0, except
 * norm_topk_prob, whose absent value is 1 (SGLang's Qwen3Next default). Lists are bitmasks: full_attention_layers has
 * bit i set when layer_types[i] is "full_attention"; ple_layer_ids has bit n set for each config value n (1-based, as
 * in config.json). rope_theta, partial_rotary_factor, mrope_interleaved and mrope_section come from rope_parameters.
 * Strings are enums: output_gate_type CERE_OUTPUT_GATE_*, mamba_ssm_dtype CERE_DTYPE_* ("float32" = F32, "bfloat16" =
 * BF16, "float16" = F16). Every field must equal the compiled support table, else CERE_ERR_UNSUPPORTED. */
typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t num_hidden_layers;
    uint32_t full_attention_interval;
    uint64_t full_attention_layers;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t head_dim;
    uint32_t output_gate_type;
    double rope_theta;
    double partial_rotary_factor;
    uint32_t mrope_interleaved;
    uint32_t mrope_section[3];
    uint32_t indexer_n_heads;
    uint32_t indexer_head_dim;
    uint32_t indexer_kv_heads;
    uint32_t indexer_compress_ratio;
    uint32_t indexer_budget;
    uint32_t linear_num_key_heads;
    uint32_t linear_num_value_heads;
    uint32_t linear_key_head_dim;
    uint32_t linear_value_head_dim;
    uint32_t linear_conv_kernel_dim;
    uint32_t mamba_ssm_dtype;
    uint32_t num_experts;
    uint32_t num_experts_per_tok;
    uint32_t moe_intermediate_size;
    uint32_t shared_expert_intermediate_size;
    uint32_t norm_topk_prob;
    uint32_t hc_count;
    uint32_t hc_lowrank;
    uint64_t ple_layer_ids;
    uint32_t ple_embed_dim;
    uint32_t heads_per_ngram;
    uint32_t ngram_size;
    uint32_t ple_conv_kernel_size;
    uint64_t ngram_vocab_size_base;
    uint32_t split_ngram_parts;
    uint32_t make_ngram_vocab_size_divisible_by;
    double rms_norm_eps;
    uint32_t tie_word_embeddings;
    uint32_t mtp_num_hidden_layers;
    uint64_t max_position_embeddings;
} cere_qwen4_exp_config;

/* The config.json values of a CERE_FAMILY_SDAR model (SelfTune ST4, DICE-8B), fixed width, no padding: Qwen3-8B
 * dense GQA layers (q_norm / k_norm RMSNorm per head, SiLU-gated FFN, rope over the full head_dim) and the block
 * diffusion decode: block_size tokens per block, mask_token_id the masked position's token. Every field must equal
 * the library's compiled descriptor values, else CERE_ERR_UNSUPPORTED. */
typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_hidden_layers;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t head_dim;
    uint32_t block_size;
    double rope_theta;
    double rms_norm_eps;
    uint32_t mask_token_id;
    uint32_t tie_word_embeddings;
    uint64_t max_position_embeddings;
} cere_sdar_config;

/* The config.json values of a CERE_FAMILY_INKLING2 model (Inkling 2 Mini, tasks/inkling-spec.md section 3), fixed
 * width, no padding. 24 physical layers; layers loop_core_start .. loop_core_end - 1 run loop_num_passes times with
 * shared weights (48 visits). local_layer_mask bit i: layer i is sliding-window attention (else full attention);
 * mxfp8_layer_mask bit i: layer i's experts are MXFP8 (else NVFP4). No RoPE: relative position bias of d_rel
 * features over rel_extent distances. Every field must equal the library's compiled descriptor values, else
 * CERE_ERR_UNSUPPORTED. */
typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t vocab_size;
    uint32_t unpadded_vocab_size;
    uint32_t hidden_size;
    uint32_t num_hidden_layers;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t head_dim;
    uint32_t moe_intermediate_size;
    uint32_t num_experts;
    uint32_t num_experts_per_tok;
    uint32_t sliding_window_size;
    uint32_t rel_extent;
    uint32_t d_rel;
    uint32_t loop_core_start;
    uint32_t loop_core_end;
    uint32_t loop_num_passes;
    uint32_t loop_gate_rank;
    uint32_t n_layers_per_attn_res_block;
    uint32_t conv_kernel_size;
    uint32_t eos_token_id;
    uint32_t local_layer_mask;
    uint32_t mxfp8_layer_mask;
    double rms_norm_eps;
    double final_logit_softcapping;
    double route_scale;
    uint64_t max_position_embeddings;
} cere_inkling2_config;

/* The vision tower of a Qwen4ExpForConditionalGeneration model (M2.13, ledger N); must equal the device's compiled
 * support entry bit for bit (else CERE_ERR_UNSUPPORTED). */
struct cere_qwen4_exp_vision_config {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t depth;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_heads;
    uint32_t patch_size;
    uint32_t temporal_patch_size;
    uint32_t spatial_merge_size;
    uint32_t in_channels;
    uint32_t out_hidden_size;
    uint32_t num_position_embeddings;
    uint32_t deepstack_count;
    uint32_t hidden_act; /* CERE_ACT_* */
    uint32_t image_token_id;
    uint32_t video_token_id;
    uint32_t vision_start_token_id;
    uint32_t vision_end_token_id;
    double layer_norm_eps;
};

typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    const char *name;   /* NUL-terminated, copied by cere_model_create */
    uint32_t dtype;     /* CERE_DTYPE_* */
    uint32_t placement; /* CERE_PLACEMENT_* */
    uint32_t ndim;      /* 0..CERE_TENSOR_MAX_DIMS; shape entries at and after ndim are 0 */
    uint32_t reserved;
    uint64_t shape[CERE_TENSOR_MAX_DIMS];
    uint64_t bytes; /* product(shape) * dtype size */
} cere_tensor_desc;

/* Slot-selection values (ABI v5). Each slot selects which kernel/path runs over the same weights, layout and buffers;
 * every slot's default (impl 0, params 0) is v3's library bit for bit. The slots, impls, classes, numerics and typed
 * params are registered in crates/cere-runtime/src/levers.rs (tests/fixtures/lever_registry.json), where a slot is
 * named a lever (tasks/selftune-spec.md D-4). */
#define CERE_SLOT_K2_GEMM UINT32_C(1)           /* impl CERE_LEVER_K2_* */
#define CERE_SLOT_PREFILL_ATTENTION UINT32_C(2) /* impl CERE_LEVER_PREFILL_* */
#define CERE_SLOT_DECODE_STEP UINT32_C(3)       /* impl CERE_LEVER_DECODE_* */
#define CERE_SLOT_MOE_GEMM UINT32_C(4)          /* impl CERE_LEVER_MOE_* */
#define CERE_SLOT_DRAFT_TOKENS UINT32_C(5)      /* impl CERE_LEVER_DRAFT_WINDOW, params[0] the window */
#define CERE_SLOT_VERIFY_ATTENTION UINT32_C(6)  /* impl CERE_LEVER_VERIFY_* */
#define CERE_SLOT_K2_WEIGHTS UINT32_C(7)        /* impl CERE_LEVER_K2_WEIGHTS_* */
#define CERE_SLOT_COUNT UINT32_C(7)             /* slots 1..CERE_SLOT_COUNT exist: the shared slots of every family */
/* Family-scoped slots start above CERE_SLOT_COUNT (and stay below CERE_SLOT_MAX). SdarOptIn (O320): family sdar's
 * GEMM choice, fixed at create (restart-only: a set_selection value other than the model's is CERE_ERR_UNSUPPORTED).
 * Only an sdar library accepts it; any other library refuses the slot as CERE_ERR_ARGUMENT (outside
 * 1..CERE_SLOT_COUNT), and so does an sdar library built before it. An sdar library's cere_model_selection lists it
 * after the shared slots. */
#define CERE_SLOT_SDAR_GEMM UINT32_C(8)              /* impl CERE_LEVER_SDAR_GEMM_* (family sdar only) */
#define CERE_LEVER_SDAR_GEMM_IDENTICAL UINT32_C(0)   /* the pinned bit-identical cuBLASLt picks (default) */
#define CERE_LEVER_SDAR_GEMM_CHANGING UINT32_C(1)    /* split-K configs through a workspace (output-changing, opt-in) */
#define CERE_LEVER_K2_CUBLASLT UINT32_C(0)       /* K2 GEMM: cuBLASLt for every m (default) */
#define CERE_LEVER_K2_SMALL_M UINT32_C(1)        /* P11 small-M kernel for m <= 16 (+ K2Eff grouped split-K) */
/* small_m plus MXFP8 W8A8 (E4M3 activations, UE8M0 per 32 k) on MXFP8 weights at m > 16 (numerics-changing, opt-in
 * -v56; K2Prefill P3, BeatSglang S7). Every m <= 16 call runs small_m's kernels. */
#define CERE_LEVER_K2_SMALL_M_A8 UINT32_C(2)
/* VulkanAmd P4-B (tasks/vulkan-amd-spec.md): the Vulkan library's K2 GEMM (m > 16) carries a BF16 x row that is
 * inexact as FP16 only through at most 64 tiny "tail" values on the matrix unit without them plus an exact FP32 fma
 * chain over the tail, instead of recomputing the row's outputs on the FP32 route (numerics-changing within the K2
 * contract bounds, opt-in). Only a library whose device has the route accepts it (the Vulkan one); any other library
 * refuses it as CERE_ERR_UNSUPPORTED. Every m <= 16 call is unchanged (the GEMV). */
#define CERE_LEVER_K2_TAIL_SPLIT UINT32_C(3)
#define CERE_LEVER_PREFILL_CUDA_CORE UINT32_C(0) /* K6/K9 CUDA-core prefill attention (default) */
#define CERE_LEVER_PREFILL_P4 UINT32_C(1)        /* M2.3 P4 tensor-core prefill attention for calls of >= 17 rows */
#define CERE_LEVER_DECODE_EAGER UINT32_C(0)      /* decode step launched eagerly (default) */
#define CERE_LEVER_DECODE_GRAPH UINT32_C(1)      /* P1: eligible decode steps replay a CUDA graph (bit-equal to eager) */
#define CERE_LEVER_DECODE_SPEC_GRAPH UINT32_C(2) /* graph + VERIFY / MTP draft steps replay per-shape CUDA graphs (bit-equal) */
#define CERE_LEVER_MOE_BUILD UINT32_C(0)         /* the MoE backend the library was compiled with (CERE_MOE_BACKEND) */
#define CERE_LEVER_MOE_P3B UINT32_C(1)           /* UNSUPPORTED unless the library is a p3b build */
#define CERE_LEVER_MOE_K7 UINT32_C(2)            /* UNSUPPORTED unless the library is a k7 build */
/* The draft_tokens slot's one impl: params[0] is 0 (the model's speculative_draft_tokens) or
 * 2..speculative_draft_tokens, the verify window cap per speculation round (drafts <= window - 1); non-zero on a model
 * without drafts or above its speculative_draft_tokens: CERE_ERR_UNSUPPORTED. It caps the drafts the executor picks
 * (MTP proposals, suffix-lookup drafts); a VERIFY's caller drafts stay bounded by speculative_draft_tokens - 1 only.
 * Greedy commits plain decode's tokens whatever the cap. */
#define CERE_LEVER_DRAFT_WINDOW UINT32_C(0)
/* P11R-19: a VERIFY chain's full-attention rows (tasks/m23-spec.md); both bit-identical. */
#define CERE_LEVER_VERIFY_PER_ROW UINT32_C(0)   /* each verify row's attention is its own one-token call (default) */
#define CERE_LEVER_VERIFY_BATCHED UINT32_C(1)   /* one K6 prologue and one attention launch over a chain's rows */
/* Fp8W: the dense K2 weights' storage, chosen at create (restart-only: INSTANTIATE converts the bytes in place; a
 * set_selection value other than the model's is CERE_ERR_UNSUPPORTED). FP8: INSTANTIATE quantizes the attention, GDN,
 * shared-expert and lm_head projections to E4M3 with one FP32 scale per output row in their own storage; K2 then
 * runs them on the FP8 small-M kernel (m <= 16) or dequantized to BF16 (m > 16). */
#define CERE_LEVER_K2_WEIGHTS_BF16 UINT32_C(0)  /* the checkpoint's BF16 weights (default) */
#define CERE_LEVER_K2_WEIGHTS_FP8 UINT32_C(1)   /* E4M3 + FP32 per-row scales (numerics-changing, opt-in) */
/* cere_model_desc.calibrate_windows (LoadCrit LC-01; a cere_levers field before ABI v5): the speculation windows
 * cere_model_seal times for the adaptive policy's cost tables, fixed at create (the table is measured once, at seal).
 * Output-neutral: the seal's rounds only price the windows. MTP: the seal times windows 1..4 and extrapolates 5..16
 * by the prior shape, skipping the chain session; the entries 1..4 are measured as under ALL. A ticket with
 * CERE_WORK_TAIL/LONG/CHAIN_DRAFTS on such a model prices its tails by that extrapolation (it still commits plain
 * decode's tokens). Any other value is CERE_ERR_ARGUMENT. */
#define CERE_CALIBRATE_ALL UINT32_C(0)  /* every window the selection and the device's chains allow (default) */
#define CERE_CALIBRATE_MTP UINT32_C(1)  /* the MTP windows 1..4 only, for a caller that never sets work bits 5-7 */

/* cere_slot_selection.block_version: the layout of cere_slot_choice this header defines. */
#define CERE_SLOT_BLOCK_VERSION UINT32_C(1)
/* Capacity of cere_slot_selection.choices and of cere_slot_choice.params. */
#define CERE_SLOT_MAX UINT32_C(16)
#define CERE_SLOT_PARAMS UINT32_C(4)

/* One slot's choice: the impl (its CERE_LEVER_* value) and the impl's typed params (the registry's `launch` params, in
 * registry order; only the draft_tokens slot has one, its window). Every params entry the impl has no param for is
 * 0. */
typedef struct {
    uint32_t slot;                      /* CERE_SLOT_* */
    uint32_t impl;                      /* the slot's CERE_LEVER_* value */
    uint32_t params[CERE_SLOT_PARAMS];
} cere_slot_choice;

/* The model's slot selection (ABI v5, cere_model_desc.selection; cere_model_selection / cere_model_set_selection):
 * `count` choices, every slot not listed at its default. CERE_ERR_ARGUMENT: a bad header, block_version other than
 * CERE_SLOT_BLOCK_VERSION, count above CERE_SLOT_MAX, a slot outside 1..CERE_SLOT_COUNT (plus, on an sdar library,
 * CERE_SLOT_SDAR_GEMM) or listed twice, an impl
 * outside its slot's CERE_LEVER_* set, a non-zero param the impl does not have, or a non-zero byte in choices[count..].
 * CERE_ERR_UNSUPPORTED: a known choice the library or the model cannot run (a MoE backend other than the compiled
 * one, a bad draft window). */
typedef struct {
    uint32_t struct_size;   /* sizeof(cere_slot_selection) */
    uint32_t abi_version;   /* CERE_EXECUTOR_ABI_VERSION */
    uint32_t block_version; /* CERE_SLOT_BLOCK_VERSION */
    uint32_t count;         /* 0..CERE_SLOT_MAX */
    cere_slot_choice choices[CERE_SLOT_MAX];
} cere_slot_selection;

typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t model_kind; /* CERE_MODEL_* */
    /* M2.9: 0 (no speculation) or 4 (verify chain of the MTP draft: up to 3 drafts + 1 token per round); other
     * values are CERE_ERR_UNSUPPORTED, as a draft the device cannot run. */
    uint32_t speculative_draft_tokens;
    /* copied by cere_model_create; a cere_sdar_config when family is CERE_FAMILY_SDAR, a cere_inkling2_config when
     * family is CERE_FAMILY_INKLING2 (the caller casts) */
    const cere_qwen4_exp_config *config;
    const cere_tensor_desc *tensors;     /* copied by cere_model_create; put addresses tensors by index */
    uint64_t tensor_count;
    /* R: the plan's session count and decode rows; a decode round steps at most one row per session (M2.6). */
    uint64_t max_sessions;
    uint64_t max_context_tokens;
    const uint32_t *stop_token_ids; /* model stop tokens for every work item; copied by cere_model_create */
    uint64_t stop_token_count;
    /* 0 = the derived pool, max_sessions x ceil(max_context_tokens / 64) x 64 tokens (M2.5);
     * non-zero: a multiple of 64, >= max_context_tokens (else CERE_ERR_ARGUMENT) and <= the support
     * bound (else CERE_ERR_UNSUPPORTED). */
    uint64_t kv_pool_tokens;
    /* Ledger-owned fields; each must be 0/null until its owner lands (else CERE_ERR_UNSUPPORTED), except
     * draft_token_map (M2.9 I2): set exactly when speculative_draft_tokens != 0, 1..vocab strictly increasing ids
     * below vocab (else CERE_ERR_ARGUMENT); copied by cere_model_create. */
    const uint32_t *draft_token_map;
    uint64_t draft_token_map_count;
    uint64_t snapshot_host_bytes; /* M2.11: pinned host snapshot arena reserved at ALLOCATE; 0 = off */
    /* M2.10 (tasks/m210-spec.md D6): device state slots of the native prefix cache; 0 = cache off (M2.5's eager page
     * take at session create, unchanged). Non-zero needs a device with a forward pass and prefix support (else
     * CERE_ERR_UNSUPPORTED) and is <= CERE_MAX_PREFIX_STATE_SLOTS (else CERE_ERR_ARGUMENT); the model then reserves
     * (prefix_state_slots + max_sessions) fixed-state slots at ALLOCATE (one decode track slot per session). */
    uint64_t prefix_state_slots;
    /* Null = text-only: then no `model.visual.*` tensor may be in the desc (M2.13; copied by cere_model_create). */
    const cere_qwen4_exp_vision_config *vision_config;
    /* ABI v5: the model's slot selection, copied by cere_model_create; null = every slot at its default. */
    const cere_slot_selection *selection;
    uint32_t family;            /* CERE_FAMILY_*: the struct `config` points at */
    uint32_t calibrate_windows; /* CERE_CALIBRATE_* */
} cere_model_desc;

/* For a greedy item (`sampling` null, or temperature 0) `id` is the highest processed logit and `runner_up` the
 * second, ties to the lower id. For a sampled item `id` is the sampled token and `runner_up` the most probable other
 * token. `logit` is the token's processed logit (after penalties, mask and logit_bias, before the temperature) and
 * `runner_up_logit` the runner-up's (M2.7 I6); a null-`sampling` item reports the raw logits, unchanged from before
 * M2.7. */
typedef struct {
    uint32_t id;
    uint32_t runner_up;
    float logit;
    float runner_up_logit;
} cere_token_score;

/* One logit_bias entry: added to the processed logit of `id` (< vocab; ids unique within one item). */
typedef struct {
    uint32_t id;
    float bias;
} cere_logit_bias;

/* Per-item sampling (cere_work.sampling; copied at submit). Pipeline per scored row: BF16 logits -> fp32, then for
 * every id already generated by this session since its last PREFILL item (count c > 0) x - (frequency_penalty * c +
 * presence_penalty), then repetition scaling (x < 0: x * r, else x / r), then logit_bias. temperature 0 selects the
 * highest processed logit (greedy with processors); otherwise x / T, softmax, the top_k / top_p / min_p filter (min_p
 * 0: one joint mask; min_p > 0: sequential renormalized filters) and an inverse-CDF draw with u from the counter
 * hash of (seed, the session position of the last fed token). Ranges (else CERE_ERR_ARGUMENT, also for NaN):
 * temperature 0 or [1e-6, inf), top_p (0, 1], min_p [0, 1], top_k [1, vocab] (vocab = no top-k), frequency and
 * presence penalties [-2, 2], repetition_penalty (0, 2], flags within CERE_SAMPLING_KNOWN_FLAGS, logit_bias entries
 * finite with unique ids < vocab (null iff logit_bias_count is 0). suppress_token_ids, suppress_token_count and
 * mask_until_committed are ledger-owned (M2.12): non-zero/non-null is CERE_ERR_UNSUPPORTED. */
struct cere_sampling_params {
    uint32_t struct_size;
    uint32_t abi_version;
    float temperature;
    float top_p;
    float min_p;
    uint32_t top_k;
    float frequency_penalty;
    float presence_penalty;
    float repetition_penalty;
    uint32_t flags; /* CERE_SAMPLING_KNOWN_FLAGS bits */
    uint64_t seed;  /* always meaningful */
    const cere_logit_bias *logit_bias;
    uint64_t logit_bias_count;
    const uint32_t *suppress_token_ids; /* ledger-owned; must be null */
    uint64_t suppress_token_count;      /* ledger-owned; must be 0 */
    uint64_t mask_until_committed;      /* ledger-owned; must be 0 */
};

typedef struct {
    uint32_t id;
    float logprob;
} cere_logprob_entry;

/* The logprobs of one committed token (cere_work.out_logprobs entry): log_softmax of the processed logits (divided
 * by the temperature for a sampled item), unfiltered. `top` holds the `top_count` (= cere_work.top_logprobs) most
 * probable ids, descending, ties to the lower id; an id with probability 0 has logprob -inf. */
struct cere_token_logprobs {
    uint32_t id;
    float logprob;
    uint32_t top_count;
    uint32_t reserved; /* 0 */
    cere_logprob_entry top[CERE_MAX_TOP_LOGPROBS];
};

/* One image of a PREFILL prompt (M2.13, ledger N). Its span is tokens[token_offset - 1] = vision_start_token_id
 * followed by exactly grid_t * grid_h * grid_w / 4 image_token_id rows from token_offset. */
struct cere_image_input {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t grid_t;
    uint32_t grid_h;
    uint32_t grid_w;
    uint32_t pad_id;              /* non-zero: the id of this image's pad rows in the PLE id stream (M2.13 D16) */
    uint64_t token_offset;        /* row of the first image pad */
    const void *pixel_patches;    /* BF16 [grid_t * grid_h * grid_w, patch dim]; copied before return */
    uint64_t pixel_bytes;         /* exactly that size */
    uint8_t pixel_sha256[32];     /* SHA-256 of exactly the pixel_bytes of pixel_patches (M2.13-L C1): the prefix
                                   * cache keys the image's pad rows by it; all zero is CERE_ERR_ARGUMENT */
};

/* PREFILL appends `tokens` (the prompt) to an empty session, or from position h after a prefix hit (M2.10:
 * cere_completion.cached_tokens = h; the hit is resolved at the ticket's first prefill round), then generates up to
 * max_new_tokens. DECODE (token_count 0) continues the session from its last committed token. Generation stops after
 * a stop token (model or item list; it is committed and counted), at max_new_tokens, or at the session's context
 * limit. */
typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t operation; /* one of CERE_OP_* */
    uint32_t flags; /* CERE_WORK_* bits */
    uint64_t expected_worker_epoch;
    uint64_t expected_session_revision; /* the session revision after its last terminal completion */
    cere_session *session;
    const uint32_t *tokens; /* submit copies IDs before return; never retained */
    uint64_t token_count;
    uint64_t max_new_tokens; /* <= out_capacity */
    uint64_t reserved_peak_bytes;
    uint32_t *out_tokens;          /* caller-owned, leased to the executor until the terminal poll */
    uint64_t out_capacity;         /* entries of out_tokens (and of out_scores/out_logprobs when set) */
    cere_token_score *out_scores;  /* optional (null); leased like out_tokens */
    const uint32_t *stop_token_ids; /* copied before return; added to the model's stop tokens */
    uint64_t stop_token_count;
    /* M2.7 (null/0 = the greedy path, unchanged; non-null needs capabilities native_sampler 1, else
     * CERE_ERR_UNSUPPORTED). top_logprobs <= CERE_MAX_TOP_LOGPROBS and non-zero only with out_logprobs. */
    const cere_sampling_params *sampling;
    cere_token_logprobs *out_logprobs; /* optional (null); leased like out_tokens */
    uint32_t top_logprobs;
    /* M2.8 (tasks/m28-spec.md A1): vocab_mask_words 0 = unconstrained (vocab_mask null); otherwise it must equal
     * ceil(vocab / 32), `sampling` must be non-null and vocab_mask the non-all-zero mask of position 0 (id j allowed
     * iff (vocab_mask[j >> 5] >> (j & 31)) & 1, the xgrammar layout), copied at submit; else CERE_ERR_ARGUMENT. A
     * constrained ticket is never stepped without the mask of its next position (cere_executor_constrain), and VERIFY
     * on it is CERE_ERR_ARGUMENT. Ledger-owned fields below must be 0/null until their owner lands (else
     * CERE_ERR_UNSUPPORTED). */
    /* M2.10 (D4 (c)): PREFILL only; 0 = none, else a multiple of 64 with 0 < capture_position < token_count (else
     * CERE_ERR_ARGUMENT, also on DECODE/VERIFY). A hint: the prefix cache checkpoints the state there when the session
     * is namespaced, the model has prefix_state_slots > 0 and the point lies at least 8,192 tokens past the hit;
     * otherwise it is ignored (never an error that depends on cache contents). */
    uint32_t capture_position;
    const uint32_t *vocab_mask;
    uint64_t vocab_mask_words;
    /* PREFILL only, <= 32 images in prompt order with disjoint spans, each <= 4,096 merged rows (M2.13-L C2). */
    const cere_image_input *images;
    uint64_t image_count;
} cere_work;

typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t expected_worker_epoch;
    uint64_t capability_profile_generation;
    /* descriptors and input token IDs are copied before return; output buffers are leased until the terminal poll */
    const cere_work *items;
    uint64_t item_count;
    uint64_t reserved_peak_bytes; /* includes shared workspace, counted once */
} cere_batch;

typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    cere_status status; /* CERE_PENDING, CERE_OK, or the error of a CERE_FINISH_ERROR completion */
    uint32_t terminal;  /* zero until an actual completion/cancel fence */
    /* Published with release ordering: out_tokens/out_scores entries below it are final and readable. Never
     * decreases across polls of one ticket. */
    uint64_t committed_tokens;
    uint64_t session_revision;
    uint64_t released_bytes;
    uint32_t finish; /* CERE_FINISH_*; CERE_FINISH_NONE until terminal */
    uint32_t cached_tokens;  /* M2.10: prompt tokens this PREFILL ticket took from the prefix cache; 0 otherwise */
    /* M2.9: speculation rounds this ticket ran (each commits its accepted drafts + 1 token) and drafts accepted; both
     * 0 for a model without draft or a session that does not draft. */
    uint64_t verify_steps;
    uint64_t accepted_drafts;
} cere_completion;

/* M2.11: what a snapshot holds (CERESNP1 header fields). Output only. */
typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t bytes;       /* the snapshot's length (cere_snapshot_read range) */
    uint64_t position;    /* tokens fed into the session state */
    uint64_t token_count; /* prompt + committed tokens (position or position + 1) */
    uint64_t fed;         /* equals position (idle session) */
    uint8_t layout_id[32];
} cere_snapshot_info;

/* The prefix cache's counters of one model (cere_model_prefix_stats; appended to ABI v4 after
 * cere_model_set_prefix_alt_edges). Output only; every counter is a cumulative u64 since the model's creation (a
 * flush keeps them), so a reader that polls it takes deltas. `reserved` reads 0. */
typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t alt_edges_on; /* 1 when cere_model_set_prefix_alt_edges(1) is in effect, else 0 */
    uint32_t reserved;
    uint64_t hits;                  /* lookups that loaded a cached state (a request's cached_tokens > 0) */
    uint64_t captures;              /* states copied into a pool or track slot */
    uint64_t capture_skipped;       /* captures that found no state slot */
    uint64_t capture_page_mismatch; /* captures/imports refused (M210S-39): other pages for equal tokens */
    uint64_t inserts;               /* states the tree took */
    uint64_t state_evictions;       /* states evicted for a slot */
    uint64_t leaf_evictions;        /* leaves evicted for pages */
    uint64_t host_spills;           /* evicted stated leaves written to the host tier */
    uint64_t host_loads;            /* host entries loaded back */
    uint64_t host_drops;            /* host entries dropped */
    uint64_t alt_edges;             /* alternative edges added (m210 D17); 0 with alt_edges_on 0 */
} cere_prefix_stats;

/* The prefix cache's capture-skip sites and lost-hit counters of one model (cere_model_prefix_capture_stats; appended
 * to ABI v4 after cere_model_prefix_stats, tasks/prefix-token-spec.md PT0a). Output only, cumulative since the model's
 * creation like cere_prefix_stats. The three skipped_* sum to cere_prefix_stats.capture_skipped. */
typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t skipped_prefill_slot;  /* prefill call-end or split-point captures that found no state slot */
    uint64_t skipped_decode_track;  /* decode track captures that found no state slot, once per ticket */
    uint64_t skipped_end_pair;      /* end-state captures that found no adjacent slot pair */
    uint64_t hit_shortfall_tokens;  /* drafting lookups: tokens from the hit to the deepest prefill-chunk multiple
                                     * within the lookup's cap that its page match passed, where no draft state was */
    uint64_t hit_shortfall_lookups; /* drafting lookups with hit_shortfall_tokens > 0 */
} cere_prefix_capture_stats;

/* Every input/output struct uses size + ABI checks (struct_size other than sizeof(T) of this header:
 * CERE_ERR_ARGUMENT; wrong abi_version: CERE_ERR_UNSUPPORTED). Errors leave outputs null/zero. No exceptions/panics
 * cross this boundary. One executor
 * serializes its host API calls, with one exception: cere_model_put may run concurrently for different tensor_index
 * values between cere_model_create and cere_model_seal. Async GPU completion is represented only by tickets. */
uint32_t cere_executor_abi_version(void); /* CERE_EXECUTOR_ABI_VERSION of the library; check before any other call */
cere_status cere_executor_create(cere_executor **out, const cere_executor_options *options);
/* Refuse destruction while work is in flight (CERE_ERR_BUSY); caller must drain first. */
cere_status cere_executor_destroy(cere_executor *executor);
/* Model/profile-specific, not merely a property of the GPU. Requires a sealed model. */
cere_status cere_executor_capabilities(cere_executor *executor, const cere_model *model,
                                       cere_execution_capabilities *out);
/* ALLOCATE: validates the desc against the compiled support table (CERE_ERR_UNSUPPORTED) and the memory estimate
 * against the device and host budgets (CERE_ERR_CAPACITY, nothing allocated), then allocates every tensor. */
cere_status cere_model_create(cere_executor *executor, const cere_model_desc *desc, cere_model **out);
/* UPLOAD: copies `len` bytes to tensor `tensor_index` at byte `offset` before return. Overlapping, out-of-range,
 * empty or post-seal puts are CERE_ERR_ARGUMENT. DISCARD tensors only count bytes. May be called from several threads
 * at once for different tensor_index values; the executor synchronizes internally (one bounce-ring slot per caller). */
cere_status cere_model_put(cere_executor *executor, cere_model *model, uint64_t tensor_index, uint64_t offset,
                           const void *bytes, uint64_t len);
/* INSTANTIATE: requires exact tiling of every non-discarded tensor (gap: CERE_ERR_ARGUMENT) and no put in flight. */
cere_status cere_model_seal(cere_executor *executor, cere_model *model);
/* Refuse release while a session or a snapshot of the model exists (CERE_ERR_BUSY). Releasing an unsealed model aborts
 * the upload. */
cere_status cere_model_release(cere_executor *executor, cere_model *model);
cere_status cere_session_create(cere_executor *executor, cere_model *model, uint64_t max_context_tokens,
                                cere_session **out);
/* Refuse release while a ticket still references the session. */
cere_status cere_session_release(cere_executor *executor, cere_session *session);
/* item_count must be 1 (more: CERE_ERR_UNSUPPORTED); rows of a decode round form across submitted tickets. */
cere_status cere_executor_submit(cere_executor *executor, const cere_batch *batch, cere_completion_ticket *out);
/* Cancel is intent; the work stops at the next token boundary. Polling a terminal completion is the release
 * authority: after it the ticket is unknown (CERE_ERR_ARGUMENT) and the leased buffers return to the caller. */
cere_status cere_executor_cancel(cere_executor *executor, cere_completion_ticket ticket);
/* Each poll runs at most one executor-wide round, then reports `ticket` (M2.6 D2): a prefill round runs one chunk (at
 * most 4,096 tokens) of the oldest ticket with unconsumed prompt; a decode round steps every decode-ready ticket
 * (submission order) once; while both are pending, each prefill round is followed by the executor's decode rounds
 * per chunk (1 unless cere_executor_set_decode_rounds: the kinds alternate) before the next chunk, or, with a decode
 * share set (cere_executor_set_decode_share), that policy picks. A round commits at most one token per participant (a
 * speculation round: its accepted drafts and one more); a finish or error reached in a round is held on that ticket
 * and returned at its next poll, so a poll of one ticket may advance others. A poll runs no round when a round ran
 * since the ticket's submit or its last poll (another ticket's poll ran it): it reports that round's result first, so
 * polling each ticket in turn hands every round's results to all of them before the next round (S1 G3,
 * tasks/beat-sglang-spec.md). A row's precondition failure ends only that ticket. */
cere_status cere_executor_poll(cere_executor *executor, cere_completion_ticket ticket, cere_completion *out);
/* M2.8 (tasks/m28-spec.md A2, D9): the mask of a constrained ticket's next token, `position` = its committed tokens
 * (vocab_mask copied before return, layout and word count as cere_work.vocab_mask). CERE_ERR_ARGUMENT when position
 * is not the committed count, the words are not ceil(vocab / 32), the mask is null or all zero, or the ticket was
 * submitted unconstrained; unknown/stale ticket as cere_executor_poll. A poll of a constrained ticket without the
 * mask of its committed position returns CERE_PENDING with nothing stepped (after cancel and length); a decode round
 * leaves such a ticket out, and stepping consumes the mask. A constrained ticket commits at most one token per poll,
 * or up to draft_count + 1 after cere_executor_constrain_drafts. */
cere_status cere_executor_constrain(cere_executor *executor, cere_completion_ticket ticket, uint64_t position,
                                    const uint32_t *vocab_mask, uint64_t vocab_mask_words);
/* M2.11 snapshot/restore of an idle session (no ticket in flight). Every one is CERE_ERR_UNSUPPORTED on a model
 * created with snapshot_host_bytes 0. Snapshot copies the session state into the model's arena (in flight: BUSY;
 * revision other than the session's: STALE_EPOCH; arena full: CAPACITY); the session stays valid. Restore loads it
 * into a fresh session of the same model (submitted before, or another model: ARGUMENT; context below the
 * snapshot's position: CAPACITY). Import validates the bytes (tasks/m211-spec.md D14) into the model's arena; any
 * failure leaves the session and the arena unchanged. Read copies [offset, offset + len) of the bytes. */
cere_status cere_session_snapshot(cere_executor *executor, cere_session *session, uint64_t expected_session_revision,
                                  cere_snapshot **out);
cere_status cere_session_restore(cere_executor *executor, cere_session *session, const cere_snapshot *snapshot);
cere_status cere_snapshot_get_info(cere_executor *executor, const cere_snapshot *snapshot, cere_snapshot_info *out);
cere_status cere_snapshot_read(cere_executor *executor, const cere_snapshot *snapshot, uint64_t offset, void *dst,
                               uint64_t len);
cere_status cere_snapshot_import(cere_executor *executor, cere_model *model, const void *bytes, uint64_t len,
                                 cere_snapshot **out);
cere_status cere_snapshot_release(cere_executor *executor, cere_snapshot *snapshot);
/* M2.10 native prefix cache (tasks/m210-spec.md D2, D11, D12). Every one is CERE_ERR_UNSUPPORTED on a model created
 * with prefix_state_slots 0. The tree is per model; its root children are keyed by a 16-byte namespace.
 * set_cache_namespace: before the session's first submit (after it: CERE_ERR_ARGUMENT); a session without a namespace
 * neither looks up nor inserts. flush: drops every device and host entry (BUSY while a ticket of the model is in
 * flight). export: the deepest cached state of `tokens` under `namespace_id` (a prefix of at most token_count tokens,
 * a multiple of 64) as a snapshot in the model's arena (first_row 0; no cached prefix: CERE_ERR_ARGUMENT; arena
 * full: CAPACITY); import: inserts such a snapshot under `namespace_id` (BUSY while a ticket of the model is in
 * flight; another model's or a session snapshot with a pending token: ARGUMENT). */
#define CERE_MAX_PREFIX_STATE_SLOTS UINT64_C(1024)
cere_status cere_session_set_cache_namespace(cere_executor *executor, cere_session *session,
                                             const uint8_t namespace_id[16]);
cere_status cere_model_flush_prefix_cache(cere_executor *executor, cere_model *model);
cere_status cere_prefix_cache_export(cere_executor *executor, cere_model *model, const uint8_t namespace_id[16],
                                     const uint32_t *tokens, uint64_t token_count, cere_snapshot **out);
cere_status cere_prefix_cache_import(cere_executor *executor, cere_model *model, const uint8_t namespace_id[16],
                                     const cere_snapshot *snapshot);
/* UPLOAD without a put (tasks/m216-spec.md L4): for a HOST_PINNED tensor of an unsealed model whose host region was
 * attached at cere_model_create from a published shared segment (header checked there; the content digest is
 * verified before cere_model_seal succeeds, which fails CERE_ERR_INTERNAL on a mismatch), covers
 * the whole tensor as a put of all its bytes would and stores its read-only resident bytes in *out, valid until the
 * model is released. CERE_ERR_UNSUPPORTED when the model's host region is private or being created by this process
 * (put the bytes instead); CERE_ERR_ARGUMENT for a non-HOST_PINNED or empty tensor, a tensor already (partly)
 * covered, a sealed model or a null `out`. Callable concurrently with cere_model_put for other tensor_index values. */
cere_status cere_model_host_resident(cere_executor *executor, cere_model *model, uint64_t tensor_index,
                                     const void **out);
/* ABI v5: the model's active selection (the desc's, or the last cere_model_set_selection): every slot
 * 1..CERE_SLOT_COUNT once, in slot order (count CERE_SLOT_COUNT; an sdar library then lists CERE_SLOT_SDAR_GEMM, count
 * CERE_SLOT_COUNT + 1), as given (CERE_LEVER_MOE_BUILD reads back 0, not
 * the backend it names), the rest of `choices` zero. `out` struct_size/abi_version must be set by the caller (as
 * other caller-allocated outputs in this header); block_version and every other field are written. */
cere_status cere_model_selection(cere_executor *executor, cere_model *model, cere_slot_selection *out);
/* ABI v5 hot swap (dev/bench): replaces the sealed model's active selection between requests (every slot not listed
 * returns to its default). CERE_ERR_BUSY while any ticket of the executor is in flight (submitted and not yet returned
 * terminal by cere_executor_poll); else the same validation as the desc's selection. Every slot selects only which
 * kernel/path runs over the same weights, layout and buffers (hot-safe; the device keeps the buffers every selection
 * needs from ALLOCATE on), except k2_weights, fixed at create. On error the active selection is unchanged. */
cere_status cere_model_set_selection(cere_executor *executor, cere_model *model, const cere_slot_selection *selection);
/* Put staging (LoadFast2, tasks/m216-spec.md M216S-11; appended to ABI v4 like cere_model_host_resident, no struct
 * change): registers the caller's host range [base, base + bytes) as put staging. A cere_model_put to a DEVICE
 * tensor whose source lies wholly inside one registered range is copied straight from it (CUDA: the range is pinned
 * with cudaHostRegister and the DMA reads the caller's pages; no bounce-ring memcpy); every other put takes the
 * bounce ring as before, and the bytes put are the same either way. The range stays the caller's: it may write it
 * between puts (a put has finished reading its source when it returns) and must not free or unmap it while it is
 * registered. CERE_ERR_ARGUMENT for a null base, 0 bytes, or a range overlapping a registered one;
 * CERE_ERR_CAPACITY when the pin fails; CERE_ERR_UNSUPPORTED on a device without the direct path (nothing kept).
 * unregister ends the registration whose base is `base`: CERE_ERR_ARGUMENT when no registered range starts there,
 * CERE_ERR_BUSY while a put from the range is in flight (the range stays registered). cere_executor_destroy returns
 * CERE_ERR_BUSY while any range is registered. Both may run concurrently with cere_model_put; otherwise they follow
 * the host-call serialization. A caller that finds neither symbol uses plain puts (the bounce ring). */
cere_status cere_executor_register_staging(cere_executor *executor, const void *base, uint64_t bytes);
cere_status cere_executor_unregister_staging(cere_executor *executor, const void *base);
/* Round policy (O79, tasks/m210-spec.md PF1; appended to ABI v4 after the put staging functions, no struct change):
 * the decode rounds that run after each prefill round before the next prefill chunk while decode-ready tickets
 * remain, 1..CERE_MAX_DECODE_ROUNDS_PER_CHUNK (1 from cere_executor_create: the kinds alternate). Only the order of
 * rounds changes, not what a round computes (a chunk is the same chunk; decode rows are row-invariant, M2.6 D6), so
 * every ticket's tokens and scores stay its solo run's. Takes effect at the next round. CERE_ERR_ARGUMENT for a null
 * executor or a count outside the range. A caller that finds no symbol keeps the default. */
#define CERE_MAX_DECODE_ROUNDS_PER_CHUNK UINT32_C(64)
cere_status cere_executor_set_decode_rounds(cere_executor *executor, uint32_t decode_rounds_per_chunk);
/* Prefix end states (tasks/m210-spec.md D15⁗′ stage 2a, D15⁗″; appended to ABI v4 like cere_executor_register_staging,
 * no struct change): on = 1 lets a drafting cache-on PREFILL without a vocab mask also capture its state at
 * floor64(n - 1) when that point lies inside its last prefill call, cold or resumed from an end state (an end state,
 * two pool slots; a lookup hits one whose row-bit signature equals its own class's and resumes pinned to its own class,
 * bit-identical to cache-off); off (0, the default at create) captures as before. CERE_ERR_ARGUMENT for on > 1 or an
 * unknown model; CERE_ERR_UNSUPPORTED on prefix_state_slots 0; CERE_ERR_BUSY once any ticket of the model was submitted
 * (the setting is unchanged). A caller that finds no symbol leaves end states off. */
cere_status cere_model_set_prefix_end_state(cere_executor *executor, cere_model *model, uint32_t on);
/* Prefix alternative edges (tasks/m210-spec.md D17; appended to ABI v4 after cere_model_set_prefix_end_state, no
 * struct change): on = 1 lets a cache-on session capture states on its own path when the tree already holds the same
 * tokens over other sessions' pages (a second conversation that shares a header with a cached one but diverged before
 * its first state): its own pages become an alternative edge beside the other one, and a lookup takes the deepest
 * state over every alternative. Off (0, the default at create) refuses such captures (M210S-39, unchanged). A state
 * still holds only its own session's KV rows, so a hit equals its cache-off run as before. CERE_ERR_ARGUMENT for
 * on > 1 or an unknown model; CERE_ERR_UNSUPPORTED on a model created with prefix_state_slots 0; CERE_ERR_BUSY once
 * any ticket of the model was submitted (the setting is unchanged). A caller that finds no symbol leaves it off. */
cere_status cere_model_set_prefix_alt_edges(cere_executor *executor, cere_model *model, uint32_t on);
/* Prefix cache counters (appended to ABI v4 after cere_model_set_prefix_alt_edges; struct cere_prefix_stats): copies
 * the model's counters into *out, whose struct_size/abi_version the caller sets (struct_size other than
 * sizeof(cere_prefix_stats) or a null `out`: CERE_ERR_ARGUMENT; wrong abi_version: CERE_ERR_UNSUPPORTED, `out`
 * untouched on both). CERE_ERR_ARGUMENT for a null/unknown executor or model; CERE_ERR_UNSUPPORTED on a model created
 * with prefix_state_slots 0 (the body is then zeroed, header kept). Read-only: it takes no lock beyond the executor's
 * handle table and changes nothing, so it follows the host-call serialization like every other call (never
 * concurrently with a submit/poll/cancel of the same executor) and is allowed between requests and while tickets are
 * in flight (rounds run only inside cere_executor_poll, so between polls the counters are quiescent). A caller that
 * finds no symbol has no counters. */
cere_status cere_model_prefix_stats(cere_executor *executor, cere_model *model, cere_prefix_stats *out);
/* Layer-sliced prefill (MixedPD, tasks/mixed-pd-spec.md; appended to ABI v4 like cere_executor_set_decode_rounds, no
 * struct change): slice_ms > 0 lets a prefill call of a model created after this call (its plan then holds the park
 * region) pause at a layer boundary once its segment has run about slice_ms while another ticket is decode-ready,
 * so the executor's decode rounds run between the call's layer segments instead of after the whole chunk. A paused
 * call keeps its prompt chunk, end-state pair and position; the next prefill round resumes it before any other
 * prompt. The pause saves and restores the call's carried rows bit for bit and changes no launch of the call, so every
 * ticket's tokens and scores stay its solo run's; only the order of rounds changes. 0 (the default at create) runs
 * every call whole, as before. Takes effect at the next prefill round; a model created while it was 0 never slices.
 * A cancel of a ticket whose call is paused takes effect once that call has finished. CERE_ERR_ARGUMENT for a null
 * executor or slice_ms above CERE_MAX_PREFILL_SLICE_MS. A caller that finds no symbol keeps every call whole. */
#define CERE_MAX_PREFILL_SLICE_MS UINT32_C(10000)
cere_status cere_executor_set_prefill_slice(cere_executor *executor, uint32_t slice_ms);
/* GrammarSpec (tasks/grammar-spec-spec.md GS-01; appended to ABI v4 like cere_model_set_prefix_end_state, no struct
 * change): speculation rounds of a constrained ticket under per-row masks. round_drafts, while the ticket waits for
 * the mask of `position` (= its committed count, no mask present): freezes the drafts of its next round, the ones an
 * unconstrained ticket's round would pick (the session's draft policy and the ticket's bits 2-4; at most
 * CERE_MAX_CONSTRAINED_DRAFTS), writes the first min(their count, capacity) to `drafts` and that count to *count;
 * repeated calls at the same position return the same ids. *count is 0 when the round has no drafts: a session that
 * does not draft, bit 0 or 1, a ticket ending (finish, cancel, budget or context reached) or a policy pick of none.
 * constrain_drafts then gives that round's masks: `drafts` = the first draft_count (<= *count) frozen ids, and
 * `vocab_masks` = draft_count + 1 masks of vocab_mask_words words each (layout and check as cere_executor_constrain;
 * row 0 is the mask of `position`, row j the mask after drafts 1..j). The round verifies those drafts with row j's
 * mask on verify row j, so it commits 1..draft_count + 1 tokens, each what masked plain decoding commits at its
 * position (greedy: bit-identical; sampled: distribution-equal). The policy counts the frozen drafts the caller left
 * out as rejected; cere_executor_constrain at that position instead runs the round with no drafts and no policy
 * update. CERE_ERR_ARGUMENT: null count, null drafts with capacity > 0, a ticket submitted unconstrained or as VERIFY,
 * a position other than the committed count or a mask already present (round_drafts); no drafts frozen at
 * `position`, draft_count above the frozen count, ids other than the frozen ones, null drafts with draft_count > 0,
 * or a mask that fails cere_executor_constrain's check (constrain_drafts); unknown/stale ticket as cere_executor_poll.
 * A caller that finds neither symbol runs constrained tickets without drafts. On a cache-on model (M2.10) a drafting
 * session whose PREFILL carried a vocab mask keeps a non-drafting session's prefix-cache layout: it looks up any
 * state, splits its prompt at the D4 points and captures there, and its rounds end on each multiple of 64 they reach
 * and capture there as decode does; so a greedy constrained ticket commits the tokens and scores (bit for bit) it
 * commits under CERE_WORK_NO_SPECULATION on the same cache (GdFix). Drafting sessions without a mask keep the
 * one-call layout (m210 D15‴). */
#define CERE_MAX_CONSTRAINED_DRAFTS UINT32_C(3)
cere_status cere_executor_round_drafts(cere_executor *executor, cere_completion_ticket ticket, uint64_t position,
                                       uint32_t *drafts, uint32_t capacity, uint32_t *count);
cere_status cere_executor_constrain_drafts(cere_executor *executor, cere_completion_ticket ticket, uint64_t position,
                                           const uint32_t *drafts, uint32_t draft_count, const uint32_t *vocab_masks,
                                           uint64_t vocab_mask_words);
/* LoRA adapters (track LoRA, tasks/lora-spec.md; appended to ABI v4 like cere_model_set_prefix_end_state, no struct
 * change). An adapter adds s * B (A x) to a target projection's output: y = bf16(float(base) + s * sum_j B[o, j] *
 * (A[j, :] . x)), both sums accumulated in double-float and the result rounded once, A BF16 [rank, in] and B BF16
 * [out, rank] (PEFT lora_A / lora_B weights), s its scale (PEFT lora_alpha / r, or lora_alpha / sqrt(r) for rsLoRA).
 * Targets are qwen4_exp projections, target id = layer *
 * CERE_LORA_MODULES + module: q / k / v_proj (full-attention layers only: in hidden_size, out num_attention_heads *
 * head_dim * 2 for q (q and its gate, HF row order) and num_key_value_heads * head_dim for k / v), o_proj (in
 * num_attention_heads * head_dim, out hidden_size) and the shared expert's gate / up (in hidden_size, out
 * shared_expert_intermediate_size) and down (in shared_expert_intermediate_size, out hidden_size); the MTP layer is
 * never a target (drafts stay base-only). Requests without an adapter run the base bits exactly.
 * reserve: after cere_model_create and before cere_model_seal, once: carves a pool of max_adapters slots of rank up to
 * max_rank (device bytes counted against memory_budget_bytes: CERE_ERR_CAPACITY when they do not fit). load: copies one
 * adapter (count targets; a and b host pointers, read before return) into a free slot of a sealed model; targets left
 * out add nothing. unload: frees the slot (CERE_ERR_BUSY while a live session uses it). set_lora: binds a session to
 * a loaded slot before its first submit (as cere_session_set_cache_namespace). CERE_ERR_ARGUMENT for an unknown
 * model/session, a slot outside [0, max_adapters), a rank outside [1, max_rank], a non-finite scale, an unknown or
 * repeated target, a null pointer, a slot already loaded (load) or not loaded (unload, set_lora), or a session already
 * submitted; CERE_ERR_UNSUPPORTED on a device without LoRA or (load, unload, set_lora) a model without a reserve. A
 * session with an adapter takes no snapshot (CERE_ERR_UNSUPPORTED). A caller that finds no symbol serves no adapter. */
#define CERE_LORA_MAX_ADAPTERS UINT32_C(64)
#define CERE_LORA_MAX_RANK UINT32_C(256)
#define CERE_LORA_MODULES UINT32_C(8)
#define CERE_LORA_Q_PROJ UINT32_C(0)
#define CERE_LORA_K_PROJ UINT32_C(1)
#define CERE_LORA_V_PROJ UINT32_C(2)
#define CERE_LORA_O_PROJ UINT32_C(3)
#define CERE_LORA_SHARED_GATE UINT32_C(4)
#define CERE_LORA_SHARED_UP UINT32_C(5)
#define CERE_LORA_SHARED_DOWN UINT32_C(6)
cere_status cere_model_lora_reserve(cere_executor *executor, cere_model *model, uint32_t max_adapters,
                                    uint32_t max_rank);
cere_status cere_model_lora_load(cere_executor *executor, cere_model *model, uint32_t slot, uint32_t rank, float scale,
                                 uint64_t count, const uint32_t *targets, const void *const *a, const void *const *b);
cere_status cere_model_lora_unload(cere_executor *executor, cere_model *model, uint32_t slot);
cere_status cere_session_set_lora(cere_executor *executor, cere_session *session, uint32_t slot);
/* Prefix capture-skip sites and lost-hit counters (appended to ABI v4 after cere_model_prefix_stats; struct
 * cere_prefix_capture_stats, tasks/prefix-token-spec.md PT0a): the same contract as cere_model_prefix_stats (header
 * checks, errors, read-only, allowed with tickets in flight). A caller that finds no symbol has no site counters. */
cere_status cere_model_prefix_capture_stats(cere_executor *executor, cere_model *model,
                                            cere_prefix_capture_stats *out);
/* Decode share (S1 G4, tasks/beat-sglang-spec.md; appended to ABI v4 like cere_executor_set_decode_rounds, no struct
 * change): share_pct > 0 replaces the decode rounds per chunk with a time share. While two or more tickets have
 * unconsumed prompts, prefill rounds run back to back (oldest first, no decode round between). With one such ticket
 * left and decode-ready tickets waiting, each of its prefill rounds (a sliced call's segment included) owes the decode
 * rounds share_pct % of the round's wall time, and decode rounds run until that time is spent before its next prefill
 * round. Only the order of rounds changes: every ticket's tokens and scores stay its solo run's. 0 (the default at
 * create) keeps the decode rounds per chunk. Takes effect at the next round. CERE_ERR_ARGUMENT for a null executor or
 * share_pct above CERE_MAX_DECODE_SHARE_PCT. A caller that finds no symbol keeps the decode rounds per chunk. */
#define CERE_MAX_DECODE_SHARE_PCT UINT32_C(1000)
cere_status cere_executor_set_decode_share(cere_executor *executor, uint32_t share_pct);
/* WeightLab research (tasks/weightlab-spec.md WL-02; appended to ABI v4 after cere_executor_set_decode_share as
 * optional symbols 42-47, no existing struct changes). Opt-in observation (capture) and intervention on one model's
 * sessions. A model without a reserve, and a session without a plan, runs today's code path bit for bit.
 * Points are ForwardTaps names (executor/cuda/src/model/model.h) with the literal "L" for the layer, e.g.
 * "native.layers.L.attn_mix.out"; `layers` bit L selects layer L in [0, 48) and bit CERE_RESEARCH_FINAL_LAYER the
 * final-mixer points ("native.final_mix.out") and the logits points. Capture points:
 *   native.layers.L.{attn_mix,mlp_mix}.in [10,240]       the HC residual state the mix reads (read only)
 *   native.layers.L.{attn_mix,mlp_mix}.out [2,560]       mix outputs
 *   native.layers.L.gemm.attn_qkv.out [13,312]  native.layers.L.gemm.gdn_qkv.out [10,240]  gdn_z.out [6,144]
 *   model.layers.L.self_attn.o_proj / model.layers.L.linear_attn.out_proj [2,560]   the block output
 *   native.layers.L.gemm.moe_router.out [512]            router logits
 *   native.layers.L.moe_route                            chosen ids I32 [10] and gate weights F32 [10] (two parts:
 *                                                        ".ids", ".weights")
 *   model.layers.L.mlp.experts / model.layers.L.mlp.shared_expert.down_proj [2,560]   routed sum, shared expert
 *   native.layers.L.gemm.moe_shared_down.in [640]        the shared expert's down_proj input (SwiGLU output; read
 *                                                        only)
 *   native.final_mix.out [2,560]                         the lm_head input
 *   native.logits.topk                                   top_k ids I32 and logprobs F32 (log-softmax of the raw
 *                                                        BF16 logits), parts ".ids", ".logprobs"; top_k in [1, 64]
 *   native.logits [248,320]                              full BF16 logits; at most 8 positions per session
 * Positions are absolute sequence positions of the row fed (row p's tensors; its logits predict token p + 1):
 * `prompt` ranges [first, end) and `generated` (the rows that produce emitted tokens: positions prompt_tokens - 1
 * onward, at most generated_max of them; 0 = max_tokens). Logits points are scored rows only, so they take generated
 * positions only. Under speculation only committed rows are kept (a rejected draft's row is overwritten or dropped);
 * MTP draft forwards are never captured or edited. A capture holds the tensor after the plan's edits at that point.
 * Intervention kinds (CERE_RESEARCH_ADD ... EXPERT_FORCE, track WL-B, research/patch.h) edit P points in place for
 * the planned sessions' rows only; their fields: alpha (add), [col_begin, col_end) (0, 0 = every column), vector
 * (host BF16 [vector_rows, vector_cols], copied at plan_load), experts / weights / expert_count (expert kinds).
 * Any planned row runs eager (no decode or verify graph); a session with a plan takes no snapshot
 * (CERE_ERR_UNSUPPORTED).
 * reserve: after cere_model_create and before cere_model_seal, once: device_staging_bytes (>= 1 MiB) of device capture
 * staging plus max_plans * CERE_RESEARCH_MAX_PLAN_BYTES of plan memory (counted against memory_budget_bytes:
 * CERE_ERR_CAPACITY when they do not fit) and host_buffer_bytes of pinned host ring (CERE_ERR_CAPACITY when the pin
 * fails). plan_load: validates and copies `plan` (every pointer read before return) into the free `slot` of a sealed
 * model. plan_unload: CERE_ERR_BUSY while a live session uses it. set_research: binds a session to a loaded plan
 * before its first submit, once, given the request's prompt_tokens (>= 1) and max_tokens; when the plan captures, it
 * reserves the bound in the host ring (CERE_ERR_CAPACITY when it does not fit: nothing bound) and writes the capture
 * id (> 0; 0 = nothing captured) to *out_capture; a plan that captures prompt positions makes that session skip the
 * prefix-cache lookup (it prefills from 0; captures at its points still happen). A replace op's vector has 1 row or
 * at least as many rows as positions the op selects for the request (row i edits its i-th position). The bound: the
 * sum over capture ops and their layers of the op's selected positions x the point's row bytes; selected positions:
 * the prompt ranges cut at prompt_tokens united with [prompt_tokens - 1, prompt_tokens - 1 + g),
 * g = min(generated_max or max_tokens, max_tokens) (saturating u64 arithmetic). A capture holds the rows of the
 * selected positions below the session's consumed tokens at its release. capture_read and capture_free:
 * CERE_ERR_BUSY while its session lives; read fills *out, whose pointers stay valid until capture_free (the ring
 * bytes are freed). CERE_ERR_ARGUMENT for an unknown model/session/capture, a slot outside [0, max_plans), an unknown
 * point, a layer the point lacks, op_count 0 or above CERE_RESEARCH_MAX_OPS, unsorted or overlapping ranges, a kind's
 * field out of range, a slot already loaded (load) or not loaded (unload, set_research), or a session already
 * submitted or bound; CERE_ERR_UNSUPPORTED on a device without research or a model without a reserve. A caller that
 * finds no symbol serves no research plan. */
#define CERE_RESEARCH_MAX_PLANS UINT32_C(64)
#define CERE_RESEARCH_MAX_OPS UINT32_C(128)
#define CERE_RESEARCH_MAX_RANGES UINT32_C(64)
#define CERE_RESEARCH_MAX_TOPK UINT32_C(64)
#define CERE_RESEARCH_MAX_LOGIT_POSITIONS UINT32_C(8)
#define CERE_RESEARCH_MAX_PLAN_BYTES UINT64_C(4194304)
#define CERE_RESEARCH_FINAL_LAYER UINT32_C(48)
#define CERE_RESEARCH_CAPTURE UINT32_C(0)
#define CERE_RESEARCH_ADD UINT32_C(1)
#define CERE_RESEARCH_ABLATE_ZERO UINT32_C(2)
#define CERE_RESEARCH_ABLATE_MEAN UINT32_C(3)
#define CERE_RESEARCH_REPLACE UINT32_C(4)
#define CERE_RESEARCH_EXPERT_BIAS UINT32_C(5)
#define CERE_RESEARCH_EXPERT_MASK UINT32_C(6)
#define CERE_RESEARCH_EXPERT_FORCE UINT32_C(7)
#define CERE_RESEARCH_DTYPE_BF16 UINT32_C(0)
#define CERE_RESEARCH_DTYPE_I32 UINT32_C(1)
#define CERE_RESEARCH_DTYPE_F32 UINT32_C(2)
typedef struct cere_research_range {
    uint64_t first;
    uint64_t end;
} cere_research_range;
typedef struct cere_research_op {
    uint32_t kind;
    uint32_t top_k;
    const char *point;
    uint64_t layers;
    const cere_research_range *prompt;
    uint32_t prompt_count;
    uint32_t generated;
    uint64_t generated_max;
    float alpha;
    uint32_t col_begin;
    uint32_t col_end;
    uint32_t vector_rows;
    const void *vector;
    uint32_t vector_cols;
    uint32_t expert_count;
    const int32_t *experts;
    const float *weights;
} cere_research_op;
typedef struct cere_research_plan {
    const cere_research_op *ops;
    uint32_t op_count;
    uint32_t reserved;
    uint8_t sha256[32];
} cere_research_plan;
typedef struct cere_research_tensor {
    const char *name;
    uint32_t op;
    uint32_t layer;
    uint32_t dtype;
    uint32_t cols;
    uint64_t rows;
    const uint64_t *positions;
    const void *data;
} cere_research_tensor;
typedef struct cere_research_capture {
    const cere_research_tensor *tensors;
    uint64_t tensor_count;
    uint64_t bytes;
    uint32_t prefix_lookup_skipped;
    uint32_t reserved;
    uint8_t plan_sha256[32];
} cere_research_capture;
cere_status cere_model_research_reserve(cere_executor *executor, cere_model *model, uint64_t device_staging_bytes,
                                        uint64_t host_buffer_bytes, uint32_t max_plans);
cere_status cere_model_research_plan_load(cere_executor *executor, cere_model *model, uint32_t slot,
                                          const cere_research_plan *plan);
cere_status cere_model_research_plan_unload(cere_executor *executor, cere_model *model, uint32_t slot);
cere_status cere_session_set_research(cere_executor *executor, cere_session *session, uint32_t slot,
                                      uint64_t prompt_tokens, uint64_t max_tokens, uint64_t *out_capture);
cere_status cere_model_research_capture_read(cere_executor *executor, cere_model *model, uint64_t capture,
                                             cere_research_capture *out);
cere_status cere_model_research_capture_free(cere_executor *executor, cere_model *model, uint64_t capture);
/* Prefill selection signature (BC1 leftover, tasks/batch-concurrency-spec.md "Prefill selection signature";
 * appended to ABI v4 after the research functions as optional symbol 48, no existing struct changes). Writes to
 * out_sha256 the SHA-256 of a canonical UTF-8 text naming every choice that picks the sealed model's prefill kernels,
 * so two loads with the same digest ran the same prefill selection (a GPU run pins it per profile). Format v1, lines
 * '\n'-terminated, one fact per line in this order:
 *   cere-prefill-signature-v1
 *   host plan_captures_rule <R>           the version of the prefix cache's split rules (api.cpp kPlanCapturesRule)
 *   host prefill_chunk <C>                tokens of one prefill call (the device's prefill chunk)
 *   <device lines>                        the device's own facts. CUDA (model/prefill_signature.h selection_text):
 *     cublaslt <cublasLtGetVersion()>; per static GemmPins table "pins <n> <k> <out> cublaslt <v> ranges <count>"
 *     then per range "pin <m_lo> <m_hi> algo <id> tile <t> stages <s> splitk <k> reduction <r> partials <p> custom
 *     <c>" (k2_gemm.h kGemmPinsTables: torch's pins, then the bucket re-pins); per
 *     prefill GEMM plan (the plans INSTANTIATE builds at the workspace rows: hc down/up, attn_qkv, attn_o,
 *     attn_index_qk, gdn_*, ple_*, moe_router, moe_shared_*; not lm_head, not the decode rows plans)
 *     "plan <name> <n> <k> <out> max_m <M> buckets <B>" then per bucket "bucket <m_hi> algo <id> tile <t> stages <s>
 *     splitk <k> reduction <r> custom <c> swizzle <w> inner <i> cluster <u>" (cuBLASLt's algorithm configuration
 *     attributes); the P3b tactic table "p3b <count>" then per bucket "p3b_bucket <max_m> <g1 tile_m tile_n
 *     tile_k_bytes swap_ab finalize> <g2 ...>", or "p3b none" on a K7 build; "k5 chunk_tokens <64>"; "k6
 *     max_context <2048>". Toy: "toy prefill-selection v1" (and "toy bucket N" with ",prefill_sig=N").
 * CERE_ERR_ARGUMENT for a null executor or out_sha256, an unknown model or one not sealed (as
 * cere_executor_capabilities); CERE_ERR_UNSUPPORTED on a device without a prefill selection; CERE_ERR_INTERNAL when
 * the device cannot read a selection back (a cuBLASLt attribute read failed). A caller that finds no symbol records
 * no signature. */
cere_status cere_model_prefill_signature(cere_executor *executor, const cere_model *model, uint8_t out_sha256[32]);
/* BC4 (b)'s aging cap (tasks/batch-concurrency-spec.md BC4; appended to ABI v4 as optional symbol 49, no struct
 * change). Without a decode share, the executor's prefill rounds take the prompt ticket with the fewest prompt tokens
 * left, ties to the oldest. A prompt ticket is overtaken when a prefill call starts for a younger ticket (submitted
 * after it) while it could run (prompt not consumed, not waiting on an older ticket's shared prefix); its own call
 * resets the count. A ticket overtaken `calls` times runs the next call (the oldest first when several have), so no
 * ticket is overtaken more than `calls` times in a row and equal prompts keep FIFO order. When a ticket's prompt is
 * consumed after it was overtaken, one stderr line reports "cere: prefill_passed tokens=<prompt tokens> max=<most
 * times in a row> cap=<calls>". Only the order of calls changes, never a ticket's tokens. CERE_DEFAULT_PREFILL_AGE_CALLS
 * at create; takes effect at the next prefill round. CERE_ERR_ARGUMENT for a null executor or calls outside
 * 1..CERE_MAX_PREFILL_AGE_CALLS. A caller that finds no symbol keeps the default. */
#define CERE_DEFAULT_PREFILL_AGE_CALLS UINT32_C(8)
#define CERE_MAX_PREFILL_AGE_CALLS UINT32_C(1024)
cere_status cere_executor_set_prefill_age(cere_executor *executor, uint32_t calls);

#ifdef __cplusplus
}
#endif
#endif
