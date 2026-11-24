// SelfTune ST4 S3a (selftune design note ST4): a library is built for one model family (gen::kFamily). The SDAR toy
// library ($CERE_TOY_EXECUTOR_SDAR, the host sources built for executor/models/dice-8b.json) refuses, by name on
// stderr, every descriptor and choice outside DICE-8B's scope (rows > 1, speculation, graphs, MXFP8 / FP8 weights, the
// prefix cache, snapshots, vision, LoRA, research), a config that is not its compiled one, an unknown family and a
// Qwen descriptor; the Qwen toy library ($CERE_TOY_EXECUTOR) refuses an SDAR descriptor. The Inkling 2 Mini toy
// library ($CERE_TOY_EXECUTOR_INKLING, executor/models/inkling-2-mini.json, inkling design note S0) refuses vision,
// a draft window other than off or 2..8 and any draft token map (D-I-S9B), the prefix cache, LoRA, research, any config
// value other than its compiled one and, with the toy's null support entry, every in-scope descriptor as `device`.
// The family checks of host/family.h are also called directly, for every family.
#include "check.h"

#include "cere_executor.h"
#include "host/family.h"

#include <dlfcn.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

using ExecutorCreate = cere_status (*)(cere_executor **, const cere_executor_options *);
using ExecutorDestroy = cere_status (*)(cere_executor *);
using ModelCreate = cere_status (*)(cere_executor *, const cere_model_desc *, cere_model **);
using LoraReserve = cere_status (*)(cere_executor *, cere_model *, uint32_t, uint32_t);
using ResearchReserve = cere_status (*)(cere_executor *, cere_model *, uint64_t, uint64_t, uint32_t);

struct Lib {
    void *handle = nullptr;
    ExecutorCreate executor_create = nullptr;
    ExecutorDestroy executor_destroy = nullptr;
    ModelCreate model_create = nullptr;
    LoraReserve lora_reserve = nullptr;
    ResearchReserve research_reserve = nullptr;
    cere_executor *executor = nullptr;
};

template <typename F>
F symbol(void *lib, const char *name) {
    void *p = dlsym(lib, name);
    if (p == nullptr) std::fprintf(stderr, "dlsym %s: %s\n", name, dlerror());
    CHECK(p != nullptr);
    return reinterpret_cast<F>(p);
}

bool open_lib(const char *env, Lib *out) {
    const char *path = std::getenv(env);
    if (path == nullptr || path[0] != '/') std::fprintf(stderr, "%s must be an absolute library path\n", env);
    CHECK(path != nullptr && path[0] == '/');
    if (path == nullptr || path[0] != '/') return false;
    out->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (out->handle == nullptr) std::fprintf(stderr, "dlopen %s: %s\n", path, dlerror());
    CHECK(out->handle != nullptr);
    if (out->handle == nullptr) return false;
    out->executor_create = symbol<ExecutorCreate>(out->handle, "cere_executor_create");
    out->executor_destroy = symbol<ExecutorDestroy>(out->handle, "cere_executor_destroy");
    out->model_create = symbol<ModelCreate>(out->handle, "cere_model_create");
    out->lora_reserve = symbol<LoraReserve>(out->handle, "cere_model_lora_reserve");
    out->research_reserve = symbol<ResearchReserve>(out->handle, "cere_model_research_reserve");
    cere_executor_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = CERE_EXECUTOR_ABI_VERSION;
    options.device_id = "toy";
    options.worker_epoch = 3;
    options.memory_budget_bytes = 1 << 20;
    CHECK_EQ(out->executor_create(&out->executor, &options), CERE_OK);
    return out->executor != nullptr;
}

// What `fn` writes to stderr (fd 2) and its status.
template <typename F>
cere_status capture(F fn, std::string *err) {
    std::fflush(stderr);
    char path[] = "/tmp/cere-test-family-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    const int saved = dup(2);
    dup2(fd, 2);
    const cere_status st = fn();
    std::fflush(stderr);
    dup2(saved, 2);
    close(saved);
    err->clear();
    char buf[512];
    lseek(fd, 0, SEEK_SET);
    for (ssize_t n; (n = read(fd, buf, sizeof(buf))) > 0;) err->append(buf, size_t(n));
    close(fd);
    unlink(path);
    return st;
}

// DICE-8B's config (executor/models/dice-8b.json), the one the SDAR toy library was compiled for.
cere_sdar_config dice_config() {
    cere_sdar_config c{};
    c.struct_size = sizeof(c);
    c.abi_version = CERE_EXECUTOR_ABI_VERSION;
    c.vocab_size = 151936;
    c.hidden_size = 4096;
    c.intermediate_size = 12288;
    c.num_hidden_layers = 36;
    c.num_attention_heads = 32;
    c.num_key_value_heads = 8;
    c.head_dim = 128;
    c.block_size = 4;
    c.rope_theta = 1000000.0;
    c.rms_norm_eps = 1e-06;
    c.mask_token_id = 151669;
    c.tie_word_embeddings = 0;
    c.max_position_embeddings = 32768;
    return c;
}

// An SDAR descriptor that every check of the SDAR library passes (it is then refused as `device`: no device of this
// library runs an SDAR model before ST4 S3c).
struct Desc {
    cere_sdar_config config = dice_config();
    cere_slot_selection selection{};
    cere_model_desc desc{};
    Desc() {
        desc.struct_size = sizeof(desc);
        desc.abi_version = CERE_EXECUTOR_ABI_VERSION;
        desc.model_kind = CERE_MODEL_SDAR_BF16;
        desc.family = CERE_FAMILY_SDAR;
        desc.config = reinterpret_cast<const cere_qwen4_exp_config *>(&config);
        desc.max_sessions = 1;
        desc.max_context_tokens = 4096;
        desc.calibrate_windows = CERE_CALIBRATE_ALL;
        selection.struct_size = sizeof(selection);
        selection.abi_version = CERE_EXECUTOR_ABI_VERSION;
        selection.block_version = CERE_SLOT_BLOCK_VERSION;
    }
    Desc(const Desc &) = delete;
    Desc &operator=(const Desc &) = delete;
    // `desc` with one slot choice in its selection.
    void choose(uint32_t slot, uint32_t impl) {
        selection.count = 1;
        selection.choices[0] = cere_slot_choice{slot, impl, {0}};
        desc.selection = &selection;
    }
};

// Creating `d` on `lib` is refused with `status`, and stderr names `field` (`cere: refused <field> (family <name>)`).
void refused(const Lib &lib, const cere_model_desc &d, cere_status status, const char *field,
             const char *family = "sdar") {
    std::string err;
    cere_model *model = nullptr;
    const cere_status st = capture([&] { return lib.model_create(lib.executor, &d, &model); }, &err);
    const std::string want = std::string("cere: refused ") + field + " (family " + family + ")\n";
    if (st != status || err != want) {
        std::fprintf(stderr, "refused %s: status %d (want %d), stderr %s", field, int(st), int(status), err.c_str());
    }
    CHECK_EQ(st, status);
    CHECK(err == want);
    CHECK(model == nullptr);
}

void sdar_library_refusals(const Lib &sdar) {
    {
        Desc d;  // in scope: only the missing device refuses it
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "device");
    }
    {
        Desc d;  // a Qwen descriptor on the SDAR library
        d.desc.model_kind = CERE_MODEL_QWEN4_EXP_NVFP4;
        d.desc.family = CERE_FAMILY_QWEN4_EXP;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "model_kind");
    }
    {
        Desc d;  // an unknown family
        d.desc.family = 7;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "family");
    }
    {
        Desc d;  // an unknown model kind
        d.desc.model_kind = 9;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "model_kind");
    }
    {
        Desc d;  // rows > 1
        d.desc.max_sessions = 2;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "max_sessions");
    }
    {
        Desc d;  // speculation
        d.desc.speculative_draft_tokens = 4;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "speculative_draft_tokens");
    }
    {
        Desc d;
        const uint32_t map[1] = {0};
        d.desc.draft_token_map = map;
        d.desc.draft_token_map_count = 1;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "draft_token_map");
    }
    {
        Desc d;
        d.desc.snapshot_host_bytes = 1 << 20;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "snapshot_host_bytes");
    }
    {
        Desc d;
        d.desc.prefix_state_slots = 8;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "prefix_state_slots");
    }
    {
        Desc d;
        d.desc.vision_config = reinterpret_cast<const cere_qwen4_exp_vision_config *>(&d.config);
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "vision_config");
    }
    {
        Desc d;
        d.desc.calibrate_windows = CERE_CALIBRATE_MTP;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "calibrate_windows");
    }
    // Every slot choice other than the registry default: graphs, the small-M GEMMs (MXFP8 among them), P4, FP8
    // weights, the batched verify, the K7 MoE backend and any draft window.
    const struct {
        uint32_t slot, impl;
        const char *field;
    } choices[] = {
        {CERE_SLOT_DECODE_STEP, CERE_LEVER_DECODE_GRAPH, "decode_step.graph"},
        {CERE_SLOT_DECODE_STEP, CERE_LEVER_DECODE_SPEC_GRAPH, "decode_step.spec_graph"},
        {CERE_SLOT_K2_GEMM, CERE_LEVER_K2_SMALL_M, "k2_gemm.small_m"},
        {CERE_SLOT_K2_GEMM, CERE_LEVER_K2_SMALL_M_A8, "k2_gemm.small_m_a8"},
        {CERE_SLOT_K2_GEMM, CERE_LEVER_K2_TAIL_SPLIT, "k2_gemm.tail_split"},
        {CERE_SLOT_K2_WEIGHTS, CERE_LEVER_K2_WEIGHTS_FP8, "k2_weights.fp8"},
        {CERE_SLOT_PREFILL_ATTENTION, CERE_LEVER_PREFILL_P4, "prefill_attention.p4"},
        {CERE_SLOT_VERIFY_ATTENTION, CERE_LEVER_VERIFY_BATCHED, "verify_attention.batched"},
        {CERE_SLOT_MOE_GEMM, CERE_LEVER_MOE_K7, "moe_gemm.k7"},
    };
    for (const auto &c : choices) {
        Desc d;
        d.choose(c.slot, c.impl);
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, c.field);
    }
    {
        Desc d;  // a draft window: an SDAR model has no drafts
        d.choose(CERE_SLOT_DRAFT_TOKENS, CERE_LEVER_DRAFT_WINDOW);
        d.selection.choices[0].params[0] = 3;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "draft_tokens");
    }
    // The registry defaults, as `cered` sends them (LeverSet::ffi): every slot listed, MoE at p3b, window 0. In scope,
    // so the toy library gets as far as its device (no SDAR device on the toy).
    {
        Desc d;
        const uint32_t defaults[][2] = {
            {CERE_SLOT_K2_GEMM, CERE_LEVER_K2_CUBLASLT},         {CERE_SLOT_PREFILL_ATTENTION, CERE_LEVER_PREFILL_CUDA_CORE},
            {CERE_SLOT_DECODE_STEP, CERE_LEVER_DECODE_EAGER},    {CERE_SLOT_MOE_GEMM, CERE_LEVER_MOE_P3B},
            {CERE_SLOT_DRAFT_TOKENS, CERE_LEVER_DRAFT_WINDOW},   {CERE_SLOT_VERIFY_ATTENTION, CERE_LEVER_VERIFY_PER_ROW},
            {CERE_SLOT_K2_WEIGHTS, CERE_LEVER_K2_WEIGHTS_BF16},
        };
        d.selection.count = 0;
        for (const auto &c : defaults) d.selection.choices[d.selection.count++] = cere_slot_choice{c[0], c[1], {0}};
        d.desc.selection = &d.selection;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "device");
    }
    // The config pointer's pointee follows `family` (ST4 D1): its leading header is read before the cast.
    {
        Desc d;
        d.desc.config = nullptr;
        refused(sdar, d.desc, CERE_ERR_ARGUMENT, "config");
    }
    {
        cere_qwen4_exp_config qwen{};  // a Qwen-sized config under family sdar
        qwen.struct_size = sizeof(qwen);
        qwen.abi_version = CERE_EXECUTOR_ABI_VERSION;
        Desc d;
        d.desc.config = &qwen;
        refused(sdar, d.desc, CERE_ERR_ARGUMENT, "config.struct_size");
    }
    {
        Desc d;
        d.config.abi_version = CERE_EXECUTOR_ABI_VERSION - 1;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "config.abi_version");
    }
    // A config other than the compiled one, by field.
    {
        Desc d;
        d.config.hidden_size = 2048;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "config.hidden_size");
    }
    {
        Desc d;
        d.config.num_key_value_heads = 32;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "config.num_key_value_heads");
    }
    {
        Desc d;
        d.config.block_size = 8;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "config.block_size");
    }
    {
        Desc d;
        d.config.mask_token_id = 151643;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "config.mask_token_id");
    }
    {
        Desc d;
        d.config.rope_theta = 10000.0;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "config.rope_theta");
    }
    {
        Desc d;
        d.config.tie_word_embeddings = 1;
        refused(sdar, d.desc, CERE_ERR_UNSUPPORTED, "config.tie_word_embeddings");
    }
    // LoRA and research, refused before any model is looked up.
    std::string err;
    CHECK_EQ(capture([&] { return sdar.lora_reserve(sdar.executor, nullptr, 1, 8); }, &err), CERE_ERR_UNSUPPORTED);
    CHECK(err == "cere: refused lora (family sdar)\n");
    CHECK_EQ(capture([&] { return sdar.research_reserve(sdar.executor, nullptr, 1 << 20, 1 << 20, 1); }, &err),
             CERE_ERR_UNSUPPORTED);
    CHECK(err == "cere: refused research (family sdar)\n");
}

// Inkling 2 Mini's config (executor/models/inkling-2-mini.json), the one the Inkling toy library was compiled for.
cere_inkling2_config inkling_config() {
    cere_inkling2_config c{};
    c.struct_size = sizeof(c);
    c.abi_version = CERE_EXECUTOR_ABI_VERSION;
    c.vocab_size = 201024;
    c.unpadded_vocab_size = 200058;
    c.hidden_size = 2304;
    c.num_hidden_layers = 24;
    c.num_attention_heads = 24;
    c.num_key_value_heads = 8;
    c.head_dim = 128;
    c.moe_intermediate_size = 1152;
    c.num_experts = 128;
    c.num_experts_per_tok = 8;
    c.sliding_window_size = 1024;
    c.rel_extent = 1024;
    c.d_rel = 16;
    c.loop_core_start = 6;
    c.loop_core_end = 18;
    c.loop_num_passes = 3;
    c.loop_gate_rank = 64;
    c.n_layers_per_attn_res_block = 6;
    c.conv_kernel_size = 4;
    c.eos_token_id = 200006;
    c.local_layer_mask = 0x7df7df;  // every layer but 5, 11, 17, 23 (config.json local_layer_ids)
    c.mxfp8_layer_mask = 0xf00000;  // layers 20-23 (hf_quant_config.json)
    c.rms_norm_eps = 1e-06;
    c.final_logit_softcapping = 30.0;
    c.route_scale = 8.0;
    c.max_position_embeddings = 1048576;
    return c;
}

// An Inkling descriptor that every check of the Inkling library passes (then refused as `device` until S5).
struct InklingDesc {
    cere_inkling2_config config = inkling_config();
    cere_model_desc desc{};
    InklingDesc() {
        desc.struct_size = sizeof(desc);
        desc.abi_version = CERE_EXECUTOR_ABI_VERSION;
        desc.model_kind = CERE_MODEL_INKLING2_NVFP4;
        desc.family = CERE_FAMILY_INKLING2;
        desc.config = reinterpret_cast<const cere_qwen4_exp_config *>(&config);
        desc.max_sessions = 1;
        desc.max_context_tokens = 4096;
        desc.calibrate_windows = CERE_CALIBRATE_ALL;
    }
    InklingDesc(const InklingDesc &) = delete;
    InklingDesc &operator=(const InklingDesc &) = delete;
};

void inkling_library_refusals(const Lib &ink) {
    const char *const family = "inkling2_loop";
    {
        InklingDesc d;
        refused(ink, d.desc, CERE_ERR_UNSUPPORTED, "device", family);
    }
    {
        InklingDesc d;  // a DICE-8B descriptor's family and kind
        d.desc.family = CERE_FAMILY_SDAR;
        d.desc.model_kind = CERE_MODEL_SDAR_BF16;
        refused(ink, d.desc, CERE_ERR_UNSUPPORTED, "model_kind", family);
    }
    {
        InklingDesc d;  // a vision_config without its header (S6g: a valid one is the device's to match)
        cere_qwen4_exp_vision_config vision{};
        d.desc.vision_config = &vision;
        refused(ink, d.desc, CERE_ERR_ARGUMENT, "vision_config", family);
    }
    {
        InklingDesc d;  // D-I-S9B: the draft window is off or 2..8 (1..7 drafts of the drafter's block)
        for (const uint32_t window : {1u, 9u, 16u}) {
            d.desc.speculative_draft_tokens = window;
            refused(ink, d.desc, CERE_ERR_UNSUPPORTED, "speculative_draft_tokens", family);
        }
    }
    {
        InklingDesc d;  // every MTP drafter window passes the family checks (then the toy has no device)
        for (uint32_t window = 2; window <= cere::host::kInkling2MaxDraftTokens; ++window) {
            d.desc.speculative_draft_tokens = window;
            refused(ink, d.desc, CERE_ERR_UNSUPPORTED, "device", family);
        }
    }
    {
        InklingDesc d;  // the drafter is full-vocab: no draft token map, drafts on or off
        const uint32_t map[] = {0, 1, 2};
        d.desc.speculative_draft_tokens = 6;
        d.desc.draft_token_map = map;
        d.desc.draft_token_map_count = 3;
        refused(ink, d.desc, CERE_ERR_UNSUPPORTED, "draft_token_map", family);
    }
    {
        InklingDesc d;
        d.desc.prefix_state_slots = 8;
        refused(ink, d.desc, CERE_ERR_UNSUPPORTED, "prefix_state_slots", family);
    }
    {
        InklingDesc d;
        d.desc.calibrate_windows = CERE_CALIBRATE_MTP;
        refused(ink, d.desc, CERE_ERR_UNSUPPORTED, "calibrate_windows", family);
    }
    {
        InklingDesc d;
        d.config.struct_size = sizeof(cere_sdar_config);
        refused(ink, d.desc, CERE_ERR_ARGUMENT, "config.struct_size", family);
    }
    {
        InklingDesc d;
        d.config.loop_num_passes = 2;
        refused(ink, d.desc, CERE_ERR_UNSUPPORTED, "config.loop_num_passes", family);
    }
    {
        InklingDesc d;
        d.config.local_layer_mask = 0xffffff;
        refused(ink, d.desc, CERE_ERR_UNSUPPORTED, "config.local_layer_mask", family);
    }
    {
        InklingDesc d;
        d.config.mxfp8_layer_mask = 0;
        refused(ink, d.desc, CERE_ERR_UNSUPPORTED, "config.mxfp8_layer_mask", family);
    }
    {
        InklingDesc d;
        d.config.final_logit_softcapping = 0.0;
        refused(ink, d.desc, CERE_ERR_UNSUPPORTED, "config.final_logit_softcapping", family);
    }
    std::string err;
    CHECK_EQ(capture([&] { return ink.lora_reserve(ink.executor, nullptr, 1, 8); }, &err), CERE_ERR_UNSUPPORTED);
    CHECK(err == "cere: refused lora (family inkling2_loop)\n");
    CHECK_EQ(capture([&] { return ink.research_reserve(ink.executor, nullptr, 1 << 20, 1 << 20, 1); }, &err),
             CERE_ERR_UNSUPPORTED);
    CHECK(err == "cere: refused research (family inkling2_loop)\n");
}

// The Qwen library refuses an SDAR descriptor (its family check, before the config is read).
void qwen_library_refusals(const Lib &qwen) {
    Desc d;
    cere_model *model = nullptr;
    CHECK_EQ(qwen.model_create(qwen.executor, &d.desc, &model), CERE_ERR_UNSUPPORTED);
    CHECK(model == nullptr);
    // An SDAR-sized config under family qwen4_exp: its struct_size is not the Qwen config's.
    d.desc.model_kind = CERE_MODEL_QWEN4_EXP_NVFP4;
    d.desc.family = CERE_FAMILY_QWEN4_EXP;
    CHECK_EQ(qwen.model_create(qwen.executor, &d.desc, &model), CERE_ERR_ARGUMENT);
    CHECK(model == nullptr);
    // Nor an Inkling descriptor, on its family check.
    InklingDesc ink;
    CHECK_EQ(qwen.model_create(qwen.executor, &ink.desc, &model), CERE_ERR_UNSUPPORTED);
    CHECK(model == nullptr);
}

void family_functions() {
    using namespace cere::host;
    CHECK_EQ(family_model_kind(CERE_FAMILY_QWEN4_EXP), CERE_MODEL_QWEN4_EXP_NVFP4);
    CHECK_EQ(family_model_kind(CERE_FAMILY_SDAR), CERE_MODEL_SDAR_BF16);
    CHECK_EQ(family_model_kind(7), 0u);
    Desc sdar;
    cere_model_desc qwen = sdar.desc;
    qwen.model_kind = CERE_MODEL_QWEN4_EXP_NVFP4;
    qwen.family = CERE_FAMILY_QWEN4_EXP;
    CHECK(family_refusal(sdar.desc, CERE_FAMILY_SDAR) == nullptr);
    CHECK(family_refusal(qwen, CERE_FAMILY_QWEN4_EXP) == nullptr);
    CHECK(std::strcmp(family_refusal(sdar.desc, CERE_FAMILY_QWEN4_EXP), "model_kind") == 0);
    CHECK(std::strcmp(family_refusal(qwen, CERE_FAMILY_SDAR), "model_kind") == 0);
    cere_model_desc mixed = qwen;  // a Qwen weight format under family sdar
    mixed.family = CERE_FAMILY_SDAR;
    CHECK(std::strcmp(family_refusal(mixed, CERE_FAMILY_QWEN4_EXP), "family") == 0);
    // Both config sizes under the other family.
    cere_qwen4_exp_config qwen_config{};
    qwen_config.struct_size = sizeof(qwen_config);
    qwen_config.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK(config_refusal(&qwen_config, CERE_FAMILY_QWEN4_EXP).field == nullptr);
    CHECK(config_refusal(&sdar.config, CERE_FAMILY_SDAR).field == nullptr);
    CHECK(std::strcmp(config_refusal(&qwen_config, CERE_FAMILY_SDAR).field, "config.struct_size") == 0);
    CHECK(std::strcmp(config_refusal(&sdar.config, CERE_FAMILY_QWEN4_EXP).field, "config.struct_size") == 0);
    CHECK_EQ(config_refusal(&sdar.config, CERE_FAMILY_QWEN4_EXP).status, CERE_ERR_ARGUMENT);
    CHECK(sdar_config_refusal(sdar.config, dice_config()) == nullptr);
    // Inkling 2 Mini.
    CHECK_EQ(family_model_kind(CERE_FAMILY_INKLING2), CERE_MODEL_INKLING2_NVFP4);
    InklingDesc ink;
    CHECK(family_refusal(ink.desc, CERE_FAMILY_INKLING2) == nullptr);
    CHECK(std::strcmp(family_refusal(ink.desc, CERE_FAMILY_SDAR), "model_kind") == 0);
    CHECK(std::strcmp(family_refusal(sdar.desc, CERE_FAMILY_INKLING2), "model_kind") == 0);
    CHECK(inkling2_refusal(ink.desc) == nullptr);
    for (uint32_t window = 2; window <= kInkling2MaxDraftTokens; ++window) {
        ink.desc.speculative_draft_tokens = window;
        CHECK(inkling2_refusal(ink.desc) == nullptr);
    }
    ink.desc.speculative_draft_tokens = 1;
    CHECK(std::strcmp(inkling2_refusal(ink.desc), "speculative_draft_tokens") == 0);
    ink.desc.speculative_draft_tokens = kInkling2MaxDraftTokens + 1;
    CHECK(std::strcmp(inkling2_refusal(ink.desc), "speculative_draft_tokens") == 0);
    ink.desc.speculative_draft_tokens = 0;
    CHECK(config_refusal(&ink.config, CERE_FAMILY_INKLING2).field == nullptr);
    CHECK(std::strcmp(config_refusal(&ink.config, CERE_FAMILY_SDAR).field, "config.struct_size") == 0);
    CHECK(std::strcmp(config_refusal(&sdar.config, CERE_FAMILY_INKLING2).field, "config.struct_size") == 0);
    CHECK(inkling2_config_refusal(ink.config, inkling_config()) == nullptr);
    cere_inkling2_config other = inkling_config();
    other.route_scale = 2.5;
    CHECK(std::strcmp(inkling2_config_refusal(other, inkling_config()), "config.route_scale") == 0);
}

}  // namespace

int main() {
    family_functions();
    Lib sdar, qwen;
    if (open_lib("CERE_TOY_EXECUTOR_SDAR", &sdar)) {
        sdar_library_refusals(sdar);
        CHECK_EQ(sdar.executor_destroy(sdar.executor), CERE_OK);
    }
    if (open_lib("CERE_TOY_EXECUTOR", &qwen)) {
        qwen_library_refusals(qwen);
        CHECK_EQ(qwen.executor_destroy(qwen.executor), CERE_OK);
    }
    Lib ink;
    if (open_lib("CERE_TOY_EXECUTOR_INKLING", &ink)) {
        inkling_library_refusals(ink);
        CHECK_EQ(ink.executor_destroy(ink.executor), CERE_OK);
    }
    return cere::test::finish("test_family");
}
