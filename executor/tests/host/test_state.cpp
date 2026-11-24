// H2 acceptance: executor/model/session/ticket state machine of the host layer, driven through the C ABI against the
// CPU toy device (linked statically so the test can inspect what the device received).
#include "check.h"

#include "cere_executor.h"
#include "host/lora.h"
#include "host/sampler_ref.h"
#include "host/sha256.h"
#include "host/snapshot_format.h"
#include "host/state.h"
#include "host/support.h"
#include "model/layout.h"
#include "toy/toy_device.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cstdlib>
#include <new>
#include <unistd.h>

// Allocation-failure injection for the hardening sweeps (AllocFail below): while armed, the n-th global allocation
// (0-based) throws std::bad_alloc once, then the probe disarms itself. The tests are single-threaded around the armed
// call (the toy's hold_after / drafting rounds run on the polling thread).
namespace alloc_probe {
std::atomic<int64_t> countdown{-1};  // < 0: disarmed
std::atomic<bool> tripped{false};
}  // namespace alloc_probe

void *operator new(std::size_t bytes) {
    if (alloc_probe::countdown.load(std::memory_order_relaxed) >= 0 && alloc_probe::countdown.fetch_sub(1) == 0) {
        alloc_probe::countdown.store(-1);
        alloc_probe::tripped.store(true);
        throw std::bad_alloc();
    }
    if (void *p = std::malloc(bytes != 0 ? bytes : 1)) return p;
    throw std::bad_alloc();
}

// The replaced operators are inlined at their call sites, where GCC sees malloc/free pairs it cannot match to new/delete.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
#pragma GCC diagnostic pop

namespace {

constexpr uint64_t kEpoch = 7;
using cere::host::a4_support;

cere_executor *make_executor(const char *device_id = "toy", uint64_t budget = uint64_t(1) << 30,
                             uint64_t epoch = kEpoch) {
    cere_executor_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = CERE_EXECUTOR_ABI_VERSION;
    options.device_id = device_id;
    options.worker_epoch = epoch;
    options.memory_budget_bytes = budget;
    cere_executor *executor = nullptr;
    CHECK_EQ(cere_executor_create(&executor, &options), CERE_OK);
    CHECK(executor != nullptr);
    return executor;
}

cere::toy::ToyDevice &toy(cere_executor *executor) {
    return static_cast<cere::toy::ToyDevice &>(*executor->device);
}

// A small model: tensors of `sizes` bytes (U8, rank 1) with the given placements.
struct Desc {
    cere_qwen4_exp_config config;
    std::vector<std::string> names;
    std::vector<cere_tensor_desc> tensors;
    std::vector<uint32_t> stops;
    cere_model_desc desc{};

    Desc(const cere_qwen4_exp_config &cfg, const std::vector<uint64_t> &sizes,
         const std::vector<uint32_t> &placements = {}, uint64_t max_sessions = 2, uint64_t max_context = 64)
        : config(cfg) {
        for (size_t i = 0; i < sizes.size(); ++i) names.push_back("t" + std::to_string(i));
        for (size_t i = 0; i < sizes.size(); ++i) {
            cere_tensor_desc t{};
            t.struct_size = sizeof(t);
            t.abi_version = CERE_EXECUTOR_ABI_VERSION;
            t.name = names[i].c_str();
            t.dtype = CERE_DTYPE_U8;
            t.placement = i < placements.size() ? placements[i] : CERE_PLACEMENT_DEVICE;
            t.ndim = 1;
            t.shape[0] = sizes[i];
            t.bytes = sizes[i];
            tensors.push_back(t);
        }
        desc.struct_size = sizeof(desc);
        desc.abi_version = CERE_EXECUTOR_ABI_VERSION;
        desc.model_kind = CERE_MODEL_QWEN4_EXP_NVFP4;
        desc.max_sessions = max_sessions;
        desc.max_context_tokens = max_context;
        sync();
    }
    void sync() {
        desc.config = &config;
        desc.tensors = tensors.data();
        desc.tensor_count = tensors.size();
        desc.stop_token_ids = stops.empty() ? nullptr : stops.data();
        desc.stop_token_count = stops.size();
    }
};

Desc toy_desc(const std::vector<uint64_t> &sizes = {16, 32}, const std::vector<uint32_t> &placements = {}) {
    return Desc(cere::toy::fixture_config(), sizes, placements);
}

std::vector<uint8_t> pattern(uint64_t len, uint8_t seed) {
    std::vector<uint8_t> bytes(len);
    for (uint64_t i = 0; i < len; ++i) bytes[i] = uint8_t(seed + 31 * i);
    return bytes;
}

cere_model *sealed_model(cere_executor *executor, Desc &d) {
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    for (uint64_t i = 0; i < d.tensors.size(); ++i) {
        auto bytes = pattern(d.tensors[i].bytes, uint8_t(i));
        CHECK_EQ(cere_model_put(executor, model, i, 0, bytes.data(), bytes.size()), CERE_OK);
    }
    CHECK_EQ(cere_model_seal(executor, model), CERE_OK);
    return model;
}

struct Work {
    std::vector<uint32_t> prompt;
    std::vector<uint32_t> out;
    std::vector<cere_token_score> scores;
    std::vector<uint32_t> stops;
    cere_work item{};
    cere_batch batch{};

    Work(cere_session *session, uint32_t op, std::vector<uint32_t> tokens, uint64_t max_new, uint64_t generation,
         std::vector<uint32_t> stop_ids = {}, uint64_t revision = 0)
        : prompt(std::move(tokens)), out(max_new, 0xffffffffu), scores(max_new), stops(std::move(stop_ids)) {
        item.struct_size = sizeof(item);
        item.abi_version = CERE_EXECUTOR_ABI_VERSION;
        item.operation = op;
        item.expected_worker_epoch = kEpoch;
        item.expected_session_revision = revision;
        item.session = session;
        item.tokens = prompt.empty() ? nullptr : prompt.data();
        item.token_count = prompt.size();
        item.max_new_tokens = max_new;
        item.out_tokens = out.data();
        item.out_capacity = out.size();
        item.out_scores = scores.data();
        item.stop_token_ids = stops.empty() ? nullptr : stops.data();
        item.stop_token_count = stops.size();
        batch.struct_size = sizeof(batch);
        batch.abi_version = CERE_EXECUTOR_ABI_VERSION;
        batch.expected_worker_epoch = kEpoch;
        batch.capability_profile_generation = generation;
        batch.items = &item;
        batch.item_count = 1;
    }
};

cere_completion completion_header() {
    cere_completion c{};
    c.struct_size = sizeof(c);
    c.abi_version = CERE_EXECUTOR_ABI_VERSION;
    return c;
}

uint64_t generation_of(cere_executor *executor, cere_model *model) {
    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(cere_executor_capabilities(executor, model, &caps), CERE_OK);
    return caps.profile_generation;
}

// Polls to the terminal completion; checks committed_tokens never decreases on the way.
cere_completion drain(cere_executor *executor, cere_completion_ticket ticket, uint64_t *polls = nullptr) {
    cere_completion c = completion_header();
    uint64_t last = 0;
    uint64_t count = 0;
    for (;;) {
        c = completion_header();
        const cere_status st = cere_executor_poll(executor, ticket, &c);
        ++count;
        CHECK(c.committed_tokens >= last);
        last = c.committed_tokens;
        if (st == CERE_OK) {
            CHECK_EQ(c.terminal, 1);
            break;
        }
        CHECK_EQ(st, CERE_PENDING);
        CHECK_EQ(c.terminal, 0);
        CHECK_EQ(c.finish, CERE_FINISH_NONE);
        if (count > 10000) break;
    }
    if (polls) *polls = count;
    return c;
}

std::vector<uint32_t> toy_sequence(std::vector<uint32_t> tokens, uint64_t n, uint32_t vocab) {
    std::vector<uint32_t> out;
    for (uint64_t i = 0; i < n; ++i) {
        const uint32_t next = cere::toy::next_token(tokens.back(), tokens.size(), vocab);
        tokens.push_back(next);
        out.push_back(next);
    }
    return out;
}

// --- tests -------------------------------------------------------------------------------------------------------

void test_status_codes_documented() {
    std::ifstream in(CERE_HEADER_PATH);
    std::stringstream ss;
    ss << in.rdbuf();
    std::vector<std::string> lines;
    for (std::string line; std::getline(ss, line);) lines.push_back(line);
    for (const char *name : {"CERE_OK", "CERE_PENDING", "CERE_ERR_ARGUMENT", "CERE_ERR_UNSUPPORTED", "CERE_ERR_CAPACITY",
                             "CERE_ERR_STALE_EPOCH", "CERE_ERR_BUSY", "CERE_ERR_INTERNAL"}) {
        const std::string define = std::string("#define ") + name + " ";
        bool documented = false;
        for (size_t i = 1; i < lines.size(); ++i) {
            if (lines[i].rfind(define, 0) == 0) documented = lines[i - 1].find("*/") != std::string::npos;
        }
        if (!documented) std::fprintf(stderr, "undocumented status %s\n", name);
        CHECK(documented);
    }
}

void test_abi_checks() {
    CHECK_EQ(cere_executor_abi_version(), CERE_EXECUTOR_ABI_VERSION);
    cere_executor_options options{};
    options.struct_size = sizeof(options) - 8;
    options.abi_version = CERE_EXECUTOR_ABI_VERSION;
    options.device_id = "toy";
    cere_executor *executor = reinterpret_cast<cere_executor *>(0x1);
    CHECK_EQ(cere_executor_create(&executor, &options), CERE_ERR_ARGUMENT);
    CHECK(executor == nullptr);
    options.struct_size = sizeof(options);
    options.abi_version = 2;
    CHECK_EQ(cere_executor_create(&executor, &options), CERE_ERR_UNSUPPORTED);
    options.abi_version = CERE_EXECUTOR_ABI_VERSION;
    options.device_id = "gpu-that-is-not-a-toy";
    CHECK_EQ(cere_executor_create(&executor, &options), CERE_ERR_UNSUPPORTED);
    options.device_id = nullptr;
    CHECK_EQ(cere_executor_create(&executor, &options), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_create(nullptr, &options), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_destroy(nullptr), CERE_ERR_ARGUMENT);

    cere_executor *ok = make_executor();
    Desc d = toy_desc();
    d.desc.abi_version = 2;
    cere_model *model = reinterpret_cast<cere_model *>(0x1);
    CHECK_EQ(cere_model_create(ok, &d.desc, &model), CERE_ERR_UNSUPPORTED);
    CHECK(model == nullptr);
    d.desc.abi_version = CERE_EXECUTOR_ABI_VERSION;
    d.desc.struct_size = 8;
    CHECK_EQ(cere_model_create(ok, &d.desc, &model), CERE_ERR_ARGUMENT);
    d.desc.struct_size = sizeof(d.desc);
    d.tensors[1].struct_size = 4;
    d.sync();
    CHECK_EQ(cere_model_create(ok, &d.desc, &model), CERE_ERR_ARGUMENT);
    d.tensors[1].struct_size = sizeof(cere_tensor_desc);
    d.tensors[1].bytes = 31;  // shape says 32
    CHECK_EQ(cere_model_create(ok, &d.desc, &model), CERE_ERR_ARGUMENT);
    d.tensors[1].bytes = 32;
    d.tensors[1].name = d.tensors[0].name;  // duplicate name
    CHECK_EQ(cere_model_create(ok, &d.desc, &model), CERE_ERR_ARGUMENT);
    d.tensors[1].name = d.names[1].c_str();
    d.tensors[1].dtype = CERE_DTYPE_F16;  // known dtype, not in the support table
    d.tensors[1].shape[0] = 16;
    CHECK_EQ(cere_model_create(ok, &d.desc, &model), CERE_ERR_UNSUPPORTED);
    d.tensors[1].dtype = 99;
    CHECK_EQ(cere_model_create(ok, &d.desc, &model), CERE_ERR_ARGUMENT);
    d.tensors[1].dtype = CERE_DTYPE_U8;
    d.tensors[1].shape[0] = 32;
    d.desc.model_kind = 2;
    CHECK_EQ(cere_model_create(ok, &d.desc, &model), CERE_ERR_UNSUPPORTED);
    d.desc.model_kind = CERE_MODEL_QWEN4_EXP_NVFP4;
    d.config.abi_version = 2;
    CHECK_EQ(cere_model_create(ok, &d.desc, &model), CERE_ERR_UNSUPPORTED);
    d.config.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK(model == nullptr);
    CHECK_EQ(toy(ok).allocated_device_bytes(), 0);
    CHECK_EQ(cere_executor_destroy(ok), CERE_OK);
}

// Every config field (each array element separately) off by one -> UNSUPPORTED, for each compiled table entry.
void test_config_off_by_one() {
    struct Field {
        const char *name;
        size_t offset;
        char kind;  // 'u' u32, 'U' u64, 'd' double
    };
#define F(field, kind) Field{#field, offsetof(cere_qwen4_exp_config, field), kind}
    const std::vector<Field> fields = {
        F(vocab_size, 'u'), F(hidden_size, 'u'), F(num_hidden_layers, 'u'), F(full_attention_interval, 'u'),
        F(full_attention_layers, 'U'), F(num_attention_heads, 'u'), F(num_key_value_heads, 'u'), F(head_dim, 'u'),
        F(output_gate_type, 'u'), F(rope_theta, 'd'), F(partial_rotary_factor, 'd'), F(mrope_interleaved, 'u'),
        Field{"mrope_section[0]", offsetof(cere_qwen4_exp_config, mrope_section) + 0, 'u'},
        Field{"mrope_section[1]", offsetof(cere_qwen4_exp_config, mrope_section) + 4, 'u'},
        Field{"mrope_section[2]", offsetof(cere_qwen4_exp_config, mrope_section) + 8, 'u'}, F(indexer_n_heads, 'u'),
        F(indexer_head_dim, 'u'), F(indexer_kv_heads, 'u'), F(indexer_compress_ratio, 'u'), F(indexer_budget, 'u'),
        F(linear_num_key_heads, 'u'), F(linear_num_value_heads, 'u'), F(linear_key_head_dim, 'u'),
        F(linear_value_head_dim, 'u'), F(linear_conv_kernel_dim, 'u'), F(mamba_ssm_dtype, 'u'), F(num_experts, 'u'),
        F(num_experts_per_tok, 'u'), F(moe_intermediate_size, 'u'), F(shared_expert_intermediate_size, 'u'),
        F(norm_topk_prob, 'u'), F(hc_count, 'u'), F(hc_lowrank, 'u'), F(ple_layer_ids, 'U'), F(ple_embed_dim, 'u'),
        F(heads_per_ngram, 'u'), F(ngram_size, 'u'), F(ple_conv_kernel_size, 'u'), F(ngram_vocab_size_base, 'U'),
        F(split_ngram_parts, 'u'), F(make_ngram_vocab_size_divisible_by, 'u'), F(rms_norm_eps, 'd'),
        F(tie_word_embeddings, 'u'), F(mtp_num_hidden_layers, 'u'), F(max_position_embeddings, 'U'),
    };
#undef F
    size_t covered = 8;  // struct_size + abi_version
    for (const Field &f : fields) covered += f.kind == 'u' ? 4 : 8;
    CHECK_EQ(covered, sizeof(cere_qwen4_exp_config));  // every field is listed

    cere_executor *executor = make_executor();
    for (const cere_qwen4_exp_config &supported : {cere::toy::fixture_config(), a4_support().config}) {
        Desc base(supported, {16});
        cere_model *model = nullptr;
        CHECK_EQ(cere_model_create(executor, &base.desc, &model), CERE_OK);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        for (const Field &f : fields) {
            for (int delta : {+1, -1}) {
                Desc d(supported, {16});
                auto *bytes = reinterpret_cast<unsigned char *>(&d.config) + f.offset;
                if (f.kind == 'u') {
                    uint32_t v;
                    std::memcpy(&v, bytes, 4);
                    v += uint32_t(delta);
                    std::memcpy(bytes, &v, 4);
                } else if (f.kind == 'U') {
                    uint64_t v;
                    std::memcpy(&v, bytes, 8);
                    v += uint64_t(int64_t(delta));
                    std::memcpy(bytes, &v, 8);
                } else {
                    double v;
                    std::memcpy(&v, bytes, 8);
                    v += delta;
                    std::memcpy(bytes, &v, 8);
                }
                model = nullptr;
                const cere_status st = cere_model_create(executor, &d.desc, &model);
                if (st != CERE_ERR_UNSUPPORTED) std::fprintf(stderr, "field %s delta %d: %d\n", f.name, delta, st);
                CHECK_EQ(st, CERE_ERR_UNSUPPORTED);
                CHECK(model == nullptr);
            }
        }
    }
    CHECK_EQ(toy(executor).allocated_device_bytes(), 0);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_support_limits() {
    cere_executor *executor = make_executor();
    Desc d = toy_desc();
    d.desc.max_context_tokens = cere::toy::fixture_support().max_context_tokens + 1;
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_ERR_UNSUPPORTED);
    d.desc.max_context_tokens = 64;
    d.desc.max_sessions = cere::toy::fixture_support().max_sessions + 1;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_ERR_UNSUPPORTED);
    d.desc.max_sessions = 0;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_ERR_ARGUMENT);
    d.desc.max_sessions = 1;
    d.stops = {cere::toy::fixture_config().vocab_size};  // stop token outside the vocabulary
    d.sync();
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_ERR_ARGUMENT);
    CHECK(model == nullptr);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// m22c design note C2, raised by m25 design note D6 and ctx262 design note CX-01: the A4 support table serves
// the model's 262,144-token context (max_position_embeddings). An A4 desc at 262,144 is accepted, its capabilities
// report 262,144 and a session of 262,144 is created; 262,145 is refused as the +1 case above.
constexpr uint64_t kA4MaxContext = 262144;

void test_a4_context_limit() {
    CHECK_EQ(a4_support().max_context_tokens, kA4MaxContext);
    // The toy fixture (cered's toy deployments with the -v3 context) serves the same context.
    CHECK(cere::toy::fixture_support().max_context_tokens >= kA4MaxContext);
    cere_executor *executor = make_executor();
    Desc d(a4_support().config, {16}, {}, 1, kA4MaxContext);
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    if (model != nullptr) {
        auto bytes = pattern(16, 0);
        CHECK_EQ(cere_model_put(executor, model, 0, 0, bytes.data(), bytes.size()), CERE_OK);
        CHECK_EQ(cere_model_seal(executor, model), CERE_OK);
        cere_execution_capabilities caps{};
        caps.struct_size = sizeof(caps);
        caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
        CHECK_EQ(cere_executor_capabilities(executor, model, &caps), CERE_OK);
        CHECK_EQ(caps.max_context_tokens, kA4MaxContext);
        cere_session *session = nullptr;
        CHECK_EQ(cere_session_create(executor, model, kA4MaxContext + 1, &session), CERE_ERR_CAPACITY);
        CHECK_EQ(cere_session_create(executor, model, kA4MaxContext, &session), CERE_OK);
        CHECK_EQ(cere_session_release(executor, session), CERE_OK);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    }
    Desc over(a4_support().config, {16}, {}, 1, kA4MaxContext + 1);
    model = nullptr;
    CHECK_EQ(cere_model_create(executor, &over.desc, &model), CERE_ERR_UNSUPPORTED);
    CHECK(model == nullptr);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// m22c design note C10 (D5), paged by m25 design note D2/D5: the host plan of the A4 model at ctx 2,048, 200,000
// and (ctx262 design note CX-01) 262,144 (one session), printed and asserted against the D5 projection.
// session_bytes, pool_bytes and derived_bytes are the plan's own carving (model/layout.h, shared with model.cpp), and
// so are the workspace's row buffers (rows stop at 4,096, D4). The session slot holds no KV and no QSA compressed
// keys: those live in the pool (12 full-attention layers x (FP8 K + V [pages x 64, 512] + BF16 keys [pages x 16,
// 128])); the slot gains the slot map (int64 [ctx]).
// The K9 scratch term is GPU-only: the workspace's shared scratch (the largest K3-K7 workspace joined by
// k9::scratch_bytes) is sized by the CUDA kernels (k9_qsa.cu's logits tile rows), which this CUDA-free test does not
// link, so it is neither printed nor asserted here. test_model_k10_long.py test_plan_bytes asserts it through
// cere_ktest_model_plan_bytes: the K9 excess over the same rows' dense plan is <= D5's K9 terms plus the K9 path's row
// buffers.
void test_a4_plan_at_context() {
    namespace layout = cere::model::layout;
    const cere_qwen4_exp_config &cfg = a4_support().config;
    cere_executor *executor = make_executor();
    Desc d(cfg, {16}, {}, 1, kA4MaxContext);
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);  // the plan exists at 262,144
    if (model != nullptr) CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    constexpr uint64_t kFullLayers = 12;
    constexpr uint64_t kGdnLayers = 36;
    constexpr uint64_t kSlack = 256;  // per region: 256-byte alignment
    for (const uint64_t ctx : {uint64_t(2048), uint64_t(200000), kA4MaxContext}) {
        const uint64_t rows = ctx < 4096 ? ctx : 4096;
        layout::Carver w{nullptr};
        layout::carve_workspace_rows(cfg, rows, w, [](layout::Region, uint32_t, uint8_t *) {});
        const uint64_t workspace_rows_bytes = w.size();
        const uint64_t derived = layout::derived_bytes(cfg, ctx);
        const uint64_t session = layout::session_bytes(cfg, ctx);
        const uint64_t pool_tokens = (ctx + 63) / 64 * 64;  // the derived pool of one session (m25 D4)
        const uint64_t pool = layout::pool_bytes(cfg, cfg.full_attention_layers, pool_tokens);
        // D5 projection, region by region (each rounded up by at most kSlack).
        const uint64_t kv = kFullLayers * 2 * pool_tokens * 512;
        const uint64_t qsa = kFullLayers * (pool_tokens / 4) * 256;
        const uint64_t slot_map = ctx * 8;
        const uint64_t ring = kFullLayers * 4 * 256;
        const uint64_t gdn = kGdnLayers * (10240ull * 3 * 2 + 48ull * 128 * 128 * 2);
        const uint64_t ple = 2 * 8 + 10240ull * 9 * 2;
        const uint64_t counts = 248320ull * 4;  // m27 I9: the sampler's out_counts
        const uint64_t session_regions = 1 + 2 * kGdnLayers + 2 + kFullLayers + 1;
        const uint64_t session_projection = slot_map + ring + gdn + ple + counts + session_regions * kSlack;
        const uint64_t pool_projection = kv + qsa + 3 * kFullLayers * kSlack;
        const uint64_t rope = ctx * 64 * 4;
        const uint64_t alphas = 48ull * 2 * 512 * 4 + kGdnLayers * 48 * 4;
        const uint64_t derived_projection = rope + alphas + (48 * 2 + kGdnLayers + 1) * kSlack;
        // Row buffers, the BF16 logits row and (m27 I9) the F32 proc_logits row.
        const uint64_t rows_floor = rows * (2 * 10240 + 4 * 2560 + 13312 + 6144) * 2 + rows * (4 + 8 + 16) +
                                    248320ull * 2 + 248320ull * 4;
        const uint64_t rows_projection = rows_floor + 13 * kSlack;
        std::printf("C10 ctx %llu: workspace_rows %llu workspace_row_bytes %llu (shared scratch incl. K9: GPU-only, "
                    "K10 test_plan_bytes) derived_bytes %llu (projection %llu) session_bytes %llu (projection %llu) "
                    "pool_bytes %llu (projection %llu)\n",
                    static_cast<unsigned long long>(ctx), static_cast<unsigned long long>(rows),
                    static_cast<unsigned long long>(workspace_rows_bytes),
                    static_cast<unsigned long long>(derived), static_cast<unsigned long long>(derived_projection),
                    static_cast<unsigned long long>(session), static_cast<unsigned long long>(session_projection),
                    static_cast<unsigned long long>(pool), static_cast<unsigned long long>(pool_projection));
        CHECK(session <= session_projection);
        CHECK(session >= slot_map + ring + gdn + ple + counts);
        CHECK(pool <= pool_projection);
        CHECK(pool >= kv + qsa);
        CHECK(derived <= derived_projection);
        CHECK(derived >= rope + alphas);
        CHECK(workspace_rows_bytes <= rows_projection);
        CHECK(workspace_rows_bytes >= rows_floor);
        // M2.11 I12 (the K15 device peak equals the merge-base's, exact): the decode graph adds no workspace byte
        // (m23 design note P1, M23P1-03: its row positions ride in ids64's tail); the carve of main 66b69b56.
        CHECK_EQ(workspace_rows_bytes, ctx == 2048 ? uint64_t(207068160) : uint64_t(412646400));
        if (ctx == 200000) {
            CHECK_EQ(kv + qsa, uint64_t(2611200000));  // m25 D5 pool: 3,125 pages x 64 x (12,288 + 768) B
            CHECK_EQ(pool, uint64_t(2611200000));       // 256-aligned regions: no slack at 3,125 pages
            CHECK_EQ(slot_map, uint64_t(1600000));      // m25 D5 slot map
            CHECK_EQ(ring, uint64_t(12288));            // m22c D5 ring
            CHECK_EQ(rope, uint64_t(51200000));         // m25 D5 derived RoPE
        }
        if (ctx == kA4MaxContext) {
            CHECK_EQ(kv + qsa, uint64_t(3422552064));  // CX-01 pool: 4,096 pages x 64 x (12,288 + 768) B
            CHECK_EQ(pool, uint64_t(3422552064));       // 256-aligned regions: no slack at 4,096 pages
            CHECK_EQ(slot_map, uint64_t(2097152));      // int64 [262,144]
            CHECK_EQ(rope, uint64_t(67108864));         // F32 [262,144, 64] derived RoPE
        }
    }
}

// The decode graph (m23 design note P1, M23P1-03) stages a step's R row positions in ids64's tail [R, 2 R) of the
// T = workspace_rows (min(context, 4,096)) entries, one H2D copy with the ids. A step whose 2 R entries do not fit T
// runs eagerly (not a failure); the ids alone need R <= T, which forward_decode_rows already requires.
void test_decode_graph_fits() {
    namespace layout = cere::model::layout;
    CHECK(!layout::decode_graph_fits(1, 1));  // a 1-token plan: every step eager
    CHECK(layout::decode_graph_fits(1, 2));
    CHECK(layout::decode_graph_fits(1, 3));
    CHECK(!layout::decode_graph_fits(2, 3));  // a 3-token plan: rows 2-3 eager, row 1 by graph
    CHECK(!layout::decode_graph_fits(4, 7));
    CHECK(layout::decode_graph_fits(4, 8));
    CHECK(layout::decode_graph_fits(16, 4096));  // every shipped plan (T = 4,096) at kMaxRows
    for (uint64_t t = 1; t <= 40; ++t) {
        for (uint64_t rows = 1; rows <= t && rows <= 16; ++rows) {
            CHECK_EQ(layout::decode_graph_fits(rows, t), 2 * rows <= t);
        }
    }
}

// The verify graph (decode_step = spec_graph, m29 design note M29S2-18b, m23 design note P11R-20): a chain at
// position p with n drafts joins it only when every row attends densely from the cache: p > 0 (a prefilled session)
// and p + n + 1 <= 2,048 (k6::kMaxContext; no row reaches K9's sparse path). Other chains run the eager verify.
void test_spec_graph_chain_fits() {
    namespace layout = cere::model::layout;
    CHECK_EQ(layout::kDenseContext, uint64_t(2048));
    CHECK(!layout::spec_graph_chain_fits(0, 0));  // nothing fed yet
    CHECK(!layout::spec_graph_chain_fits(0, 3));
    CHECK(layout::spec_graph_chain_fits(1, 0));
    CHECK(layout::spec_graph_chain_fits(1, 3));
    CHECK(layout::spec_graph_chain_fits(2044, 3));  // rows at 2,044 .. 2,047
    CHECK(!layout::spec_graph_chain_fits(2045, 3));  // row 3 at 2,048: K9
    CHECK(layout::spec_graph_chain_fits(2047, 0));
    CHECK(!layout::spec_graph_chain_fits(2048, 0));
    CHECK(!layout::spec_graph_chain_fits(199999, 3));
    for (uint64_t p = 0; p <= 2060; ++p) {
        for (uint64_t n = 0; n <= 3; ++n) {
            CHECK_EQ(layout::spec_graph_chain_fits(p, n), p > 0 && p + n + 1 <= 2048);
        }
    }
}

// The draft graph (decode_step = spec_graph, m29 design note M29S2-18b, m23 design note P11R-20): an MTP
// draft-extend of `rows` rows at p0 then kDraftSteps decode steps joins it only when every row attends densely from
// the FP8 cache with a prefix, both decode steps run inside the context and the staged positions fit T.
void test_spec_draft_graph_fits() {
    namespace layout = cere::model::layout;
    CHECK_EQ(layout::kDraftSteps, uint64_t(2));
    CHECK_EQ(layout::kDraftGraphRows, uint64_t(16));
    CHECK(!layout::spec_draft_graph_fits(0, 1, 4096, 4096));    // no prefix
    CHECK(!layout::spec_draft_graph_fits(5, 0, 4096, 4096));    // nothing to extend
    CHECK(layout::spec_draft_graph_fits(1, 1, 4096, 4096));
    CHECK(layout::spec_draft_graph_fits(5, 16, 4096, 4096));
    CHECK(!layout::spec_draft_graph_fits(5, 17, 4096, 4096));   // past K6's split-K rows
    CHECK(layout::spec_draft_graph_fits(2042, 4, 4096, 4096));  // steps at 2,046 and 2,047
    CHECK(!layout::spec_draft_graph_fits(2043, 4, 4096, 4096)); // step 2 at 2,048: K9
    CHECK(!layout::spec_draft_graph_fits(100, 4, 105, 4096));   // step 2 past the context: the eager loop stops
    CHECK(layout::spec_draft_graph_fits(100, 4, 106, 4096));
    CHECK(!layout::spec_draft_graph_fits(100, 4, 4096, 5));     // 6 staged positions in T = 5
    for (uint64_t p = 0; p <= 2060; ++p) {
        for (uint64_t rows = 0; rows <= 17; ++rows) {
            CHECK_EQ(layout::spec_draft_graph_fits(p, rows, 4096, 4096),
                     p > 0 && rows >= 1 && rows <= 16 && p + rows + 2 <= 2048);
        }
    }
}

// A draft's MTP decode steps after its extend of `rows` rows at p0 (CudaDevice::draft): 2 while inside the context,
// fewer at its end, none while the prompt stays open. An image prompt's chunk before its last image ends sits below
// the text RoPE table (p + rope_delta < 0); its decode steps used to run there and K9's bounds refused them
// (model_vl43: chunked and long image prompts with speculation, CERE_ERR_ARGUMENT; cered HTTP 502).
void test_draft_decode_steps() {
    namespace layout = cere::model::layout;
    CHECK_EQ(layout::draft_decode_steps(0, 4096, 200000, false), uint64_t(2));   // a closed prompt's last chunk
    CHECK_EQ(layout::draft_decode_steps(0, 4096, 200000, true), uint64_t(0));    // a chunk before the prompt's last
    CHECK_EQ(layout::draft_decode_steps(0, 1001, 200000, true), uint64_t(0));    // VL43's 1,001-row chunk
    CHECK_EQ(layout::draft_decode_steps(1001, 1, 200000, false), uint64_t(2));   // a decode's extend
    CHECK_EQ(layout::draft_decode_steps(100, 4, 105, false), uint64_t(1));       // step 2 past the context
    CHECK_EQ(layout::draft_decode_steps(100, 4, 104, false), uint64_t(0));       // the extend ends the context
    CHECK_EQ(layout::draft_decode_steps(100, 4, 103, false), uint64_t(0));
    for (uint64_t ctx = 1; ctx <= 12; ++ctx) {
        for (uint64_t p = 0; p <= 10; ++p) {
            for (uint64_t rows = 1; rows <= 4; ++rows) {
                // the eager loop's old bound: a step at position q runs while q < ctx, at most kDraftSteps
                uint64_t eager = 0;
                for (uint64_t q = p + rows; eager < layout::kDraftSteps && q < ctx; ++q) ++eager;
                CHECK_EQ(layout::draft_decode_steps(p, rows, ctx, false), eager);
                CHECK_EQ(layout::draft_decode_steps(p, rows, ctx, true), uint64_t(0));
                // the draft graph replays both steps: it fits only where the eager loop runs both
                if (layout::spec_draft_graph_fits(p, rows, ctx, 4096)) {
                    CHECK_EQ(layout::draft_decode_steps(p, rows, ctx, false), layout::kDraftSteps);
                }
            }
        }
    }
}

// m29 design note M29S2-19/20 (memory_plan.h region 7): the draft-only region of a plan with drafts, by hand. One undo
// snapshot copies every in-place verify mutation in the session slot's shapes: per GDN layer conv BF16 [10,240, 3] and
// SSM BF16 [48, 128, 128], the PLE history (2 x int64, one 256-byte region) and conv BF16 [10,240, 9], and per
// full-attention layer the QSA ring BF16 [4, 128] plus one compressed-key row BF16 [128] (each its own 256-byte
// region). With R sessions the region is R x 3 snapshots, each a multiple of 256 (no alignment slack between them); a
// spec-off plan carves none of it (plan_layout, A1).
void test_a4_draft_region() {
    namespace layout = cere::model::layout;
    const cere_qwen4_exp_config &cfg = a4_support().config;
    constexpr uint64_t kFullLayers = 12;
    constexpr uint64_t kGdnLayers = 36;
    const uint64_t gdn = kGdnLayers * (10240ull * 3 * 2 + 48ull * 128 * 128 * 2);
    const uint64_t ple = 256 + 10240ull * 9 * 2;
    const uint64_t qsa = kFullLayers * (4 * 128 * 2 + 256);
    const uint64_t undo = layout::undo_bytes(cfg);
    std::printf("M29 draft region: undo_bytes %llu (gdn %llu ple %llu qsa %llu); region at R = 1, 2, 4: %llu %llu "
                "%llu\n",
                static_cast<unsigned long long>(undo), static_cast<unsigned long long>(gdn),
                static_cast<unsigned long long>(ple), static_cast<unsigned long long>(qsa),
                static_cast<unsigned long long>(1 * layout::kUndoSnapshots * undo),
                static_cast<unsigned long long>(2 * layout::kUndoSnapshots * undo),
                static_cast<unsigned long long>(4 * layout::kUndoSnapshots * undo));
    CHECK_EQ(undo, gdn + ple + qsa);
    CHECK_EQ(undo, uint64_t(59034880));
    CHECK_EQ(undo % layout::kAlign, uint64_t(0));
    CHECK_EQ(layout::kUndoSnapshots, uint64_t(3));
    CHECK_EQ(4 * layout::kUndoSnapshots * undo, uint64_t(708418560));
    // The snapshot regions mirror the session slot's: every tag carve_undo binds is a session tag except the key row.
    uint64_t gdn_regions = 0, rings = 0, key_rows = 0, ple_regions = 0;
    layout::Carver c{nullptr};
    layout::carve_undo(cfg, c, [&](layout::Region r, uint32_t l, uint8_t *) {
        switch (r) {
            case layout::Region::kGdnConv:
            case layout::Region::kGdnSsm: CHECK(!layout::full_attention(cfg, l)); ++gdn_regions; break;
            case layout::Region::kQsaRing: CHECK(layout::full_attention(cfg, l)); ++rings; break;
            case layout::Region::kUndoKeyRow: CHECK(layout::full_attention(cfg, l)); ++key_rows; break;
            case layout::Region::kPleHistory:
            case layout::Region::kPleConv: ++ple_regions; break;
            default: CHECK(false); break;
        }
    });
    CHECK_EQ(gdn_regions, 2 * kGdnLayers);
    CHECK_EQ(rings, kFullLayers);
    CHECK_EQ(key_rows, kFullLayers);
    CHECK_EQ(ple_regions, uint64_t(2));
}

void test_put_rules() {
    cere_executor *executor = make_executor();
    Desc d = toy_desc({16, 32, 8}, {CERE_PLACEMENT_DEVICE, CERE_PLACEMENT_HOST_PINNED, CERE_PLACEMENT_DISCARD});
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    auto b = pattern(64, 3);
    CHECK_EQ(cere_model_put(executor, model, 0, 0, b.data(), 8), CERE_OK);
    CHECK_EQ(cere_model_put(executor, model, 0, 4, b.data(), 8), CERE_ERR_ARGUMENT);   // overlaps [0, 8)
    CHECK_EQ(cere_model_put(executor, model, 0, 0, b.data(), 1), CERE_ERR_ARGUMENT);   // overlaps
    CHECK_EQ(cere_model_put(executor, model, 0, 12, b.data(), 8), CERE_ERR_ARGUMENT);  // past the end
    CHECK_EQ(cere_model_put(executor, model, 0, UINT64_MAX, b.data(), 2), CERE_ERR_ARGUMENT);  // overflow
    CHECK_EQ(cere_model_put(executor, model, 3, 0, b.data(), 1), CERE_ERR_ARGUMENT);   // no such tensor
    CHECK_EQ(cere_model_put(executor, model, 0, 8, nullptr, 8), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_put(executor, model, 0, 8, b.data(), 0), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_put(executor, reinterpret_cast<cere_model *>(&d), 0, 8, b.data(), 8), CERE_ERR_ARGUMENT);
    // Seal with a gap in tensor 0 ([8, 16) missing) and tensor 1 untouched.
    CHECK_EQ(cere_model_seal(executor, model), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_put(executor, model, 0, 8, b.data() + 8, 8), CERE_OK);
    CHECK_EQ(cere_model_put(executor, model, 1, 16, b.data() + 16, 16), CERE_OK);
    CHECK_EQ(cere_model_seal(executor, model), CERE_ERR_ARGUMENT);  // tensor 1 [0, 16) missing
    CHECK_EQ(cere_model_put(executor, model, 1, 0, b.data(), 16), CERE_OK);
    // DISCARD tensors need no bytes to seal; puts to them only count.
    CHECK_EQ(cere_model_put(executor, model, 2, 0, b.data(), 4), CERE_OK);
    CHECK_EQ(cere_model_put(executor, model, 2, 2, b.data(), 4), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_seal(executor, model), CERE_OK);
    CHECK_EQ(model->discarded_bytes.load(), 4);
    CHECK_EQ(model->uploaded_device_bytes.load(), 16);
    CHECK_EQ(model->uploaded_host_bytes.load(), 32);
    CHECK(toy(executor).tensor(*model->device, 0) == std::vector<uint8_t>(b.begin(), b.begin() + 16));
    CHECK(toy(executor).tensor(*model->device, 1) == std::vector<uint8_t>(b.begin(), b.begin() + 32));
    CHECK(toy(executor).tensor(*model->device, 2).empty());
    // Post-seal put and double seal.
    CHECK_EQ(cere_model_put(executor, model, 2, 4, b.data(), 4), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_seal(executor, model), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_ERR_ARGUMENT);  // already released
    CHECK_EQ(toy(executor).allocated_device_bytes(), 0);
    CHECK_EQ(toy(executor).allocated_host_bytes(), 0);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_concurrent_disjoint_puts() {
    cere_executor *executor = make_executor();
    std::vector<uint64_t> sizes;
    for (int i = 0; i < 16; ++i) sizes.push_back(4096 + 512 * uint64_t(i));
    Desc d = toy_desc(sizes);
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    std::atomic<int> errors{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            while (!go.load()) std::this_thread::yield();
            for (uint64_t i = uint64_t(t); i < sizes.size(); i += 4) {
                const auto bytes = pattern(sizes[i], uint8_t(i * 7));
                for (uint64_t off = 0; off < sizes[i]; off += 1024) {
                    const uint64_t len = std::min<uint64_t>(1024, sizes[i] - off);
                    if (cere_model_put(executor, model, i, off, bytes.data() + off, len) != CERE_OK) ++errors;
                }
            }
        });
    }
    go = true;
    for (auto &th : threads) th.join();
    CHECK_EQ(errors.load(), 0);
    CHECK_EQ(cere_model_seal(executor, model), CERE_OK);
    uint64_t total = 0;
    for (uint64_t i = 0; i < sizes.size(); ++i) {
        CHECK(toy(executor).tensor(*model->device, i) == pattern(sizes[i], uint8_t(i * 7)));
        total += sizes[i];
    }
    CHECK_EQ(model->uploaded_device_bytes.load(), total);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_concurrent_overlapping_puts() {
    cere_executor *executor = make_executor();
    for (int round = 0; round < 200; ++round) {
        Desc d = toy_desc({256});
        cere_model *model = nullptr;
        CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
        const auto a = pattern(256, 1);
        const auto b = pattern(256, 2);
        std::atomic<int> ready{0};
        cere_status results[2] = {CERE_OK, CERE_OK};
        auto run = [&](int who, uint64_t offset, const std::vector<uint8_t> &bytes) {
            ++ready;
            while (ready.load() < 2) std::this_thread::yield();
            results[who] = cere_model_put(executor, model, 0, offset, bytes.data() + offset, 160);
        };
        std::thread t0(run, 0, 0, std::cref(a));
        std::thread t1(run, 1, 96, std::cref(b));  // [0,160) and [96,256) overlap in [96,160)
        t0.join();
        t1.join();
        const int failed = (results[0] == CERE_ERR_ARGUMENT) + (results[1] == CERE_ERR_ARGUMENT);
        const int ok = (results[0] == CERE_OK) + (results[1] == CERE_OK);
        CHECK_EQ(failed, 1);
        CHECK_EQ(ok, 1);
        CHECK_EQ(cere_model_seal(executor, model), CERE_ERR_ARGUMENT);  // the loser's range is a gap
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    }
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_capacity() {
    // Device budget below the device tensors: CAPACITY, nothing allocated.
    cere_executor *executor = make_executor("toy", 40);
    Desc d = toy_desc({16, 32});
    cere_model *model = reinterpret_cast<cere_model *>(0x1);
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_ERR_CAPACITY);
    CHECK(model == nullptr);
    CHECK_EQ(toy(executor).allocated_device_bytes(), 0);
    CHECK_EQ(toy(executor).allocated_host_bytes(), 0);
    CHECK_EQ(toy(executor).allocations(), 0);
    CHECK(executor->models.empty());
    // Exactly at the budget fits; a second model then exceeds what is left.
    Desc fits = toy_desc({8, 32});
    CHECK_EQ(cere_model_create(executor, &fits.desc, &model), CERE_OK);
    cere_model *second = nullptr;
    Desc small = toy_desc({1});
    CHECK_EQ(cere_model_create(executor, &small.desc, &second), CERE_ERR_CAPACITY);
    CHECK(second == nullptr);
    CHECK_EQ(toy(executor).allocated_device_bytes(), 40);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    // Host budget below the pinned-host tensors.
    executor = make_executor("toy,host_bytes=20");
    Desc host = toy_desc({16, 32}, {CERE_PLACEMENT_DEVICE, CERE_PLACEMENT_HOST_PINNED});
    model = nullptr;
    CHECK_EQ(cere_model_create(executor, &host.desc, &model), CERE_ERR_CAPACITY);
    CHECK(model == nullptr);
    CHECK_EQ(toy(executor).allocations(), 0);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_generation_and_tokens() {
    cere_executor *executor = make_executor();
    Desc d = toy_desc();
    d.stops = {};
    d.sync();
    cere_model *model = sealed_model(executor, d);
    const uint64_t generation = generation_of(executor, model);
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;

    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(cere_executor_capabilities(executor, model, &caps), CERE_OK);
    CHECK_EQ(caps.worker_epoch, kEpoch);
    CHECK_EQ(caps.decode_batch_min, 1);
    CHECK_EQ(caps.decode_batch_max, 2);  // the plan's max_sessions (M2.6 D3)
    CHECK_EQ(caps.max_context_tokens, 64);
    CHECK_EQ(caps.execution_mode, CERE_EXECUTION_MODE_NATIVE);
    CHECK_EQ(caps.operations, CERE_OPS_PREFILL | CERE_OPS_DECODE | CERE_OPS_CANCEL);
    CHECK_EQ(caps.native_decode_batch, 1);
    CHECK_EQ(caps.native_mixed_prefill_decode + caps.native_speculative_batch + caps.native_moe_grouping +
                 caps.native_state_snapshot,
             0);

    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &session), CERE_OK);
    const std::vector<uint32_t> prompt = {5, 9, 69999};
    Work prefill(session, CERE_OP_PREFILL, prompt, 4, generation);
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &prefill.batch, &ticket), CERE_OK);
    CHECK_EQ(ticket.worker_epoch, kEpoch);
    CHECK_EQ(ticket.model_generation, generation);
    uint64_t polls = 0;
    cere_completion c = drain(executor, ticket, &polls);
    CHECK_EQ(polls, 4);  // one token per poll; the poll that commits the last token is terminal
    CHECK_EQ(c.status, CERE_OK);
    CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
    CHECK_EQ(c.committed_tokens, 4);
    CHECK_EQ(c.session_revision, 1);
    const auto expected = toy_sequence(prompt, 6, vocab);
    for (int i = 0; i < 4; ++i) {
        CHECK_EQ(prefill.out[i], expected[i]);
        CHECK_EQ(prefill.scores[i].id, expected[i]);
        CHECK_EQ(prefill.scores[i].runner_up, (expected[i] + 1) % vocab);
        CHECK(prefill.scores[i].logit > prefill.scores[i].runner_up_logit);
    }
    // The terminal poll released the ticket.
    c = completion_header();
    CHECK_EQ(cere_executor_poll(executor, ticket, &c), CERE_ERR_ARGUMENT);
    CHECK_EQ(c.committed_tokens, 0);
    CHECK_EQ(cere_executor_cancel(executor, ticket), CERE_ERR_ARGUMENT);

    // DECODE continues from the last committed token; a stale revision is refused.
    Work stale(session, CERE_OP_DECODE, {}, 2, generation, {}, 0);
    CHECK_EQ(cere_executor_submit(executor, &stale.batch, &ticket), CERE_ERR_STALE_EPOCH);
    Work decode(session, CERE_OP_DECODE, {}, 2, generation, {}, 1);
    CHECK_EQ(cere_executor_submit(executor, &decode.batch, &ticket), CERE_OK);
    c = drain(executor, ticket);
    CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
    CHECK_EQ(decode.out[0], expected[4]);
    CHECK_EQ(decode.out[1], expected[5]);
    CHECK_EQ(c.session_revision, 2);

    // A second PREFILL on a non-empty session, and a DECODE on an empty one, are misuse.
    Work again(session, CERE_OP_PREFILL, {1}, 1, generation, {}, 2);
    CHECK_EQ(cere_executor_submit(executor, &again.batch, &ticket), CERE_ERR_ARGUMENT);
    cere_session *fresh = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &fresh), CERE_OK);
    Work early(fresh, CERE_OP_DECODE, {}, 1, generation);
    CHECK_EQ(cere_executor_submit(executor, &early.batch, &ticket), CERE_ERR_ARGUMENT);
    Work verify(fresh, CERE_OP_VERIFY, {1}, 1, generation);
    CHECK_EQ(cere_executor_submit(executor, &verify.batch, &ticket), CERE_ERR_UNSUPPORTED);
    Work bad_token(fresh, CERE_OP_PREFILL, {vocab}, 1, generation);
    CHECK_EQ(cere_executor_submit(executor, &bad_token.batch, &ticket), CERE_ERR_ARGUMENT);
    Work small_out(fresh, CERE_OP_PREFILL, {1}, 2, generation);
    small_out.item.out_capacity = 1;
    CHECK_EQ(cere_executor_submit(executor, &small_out.batch, &ticket), CERE_ERR_ARGUMENT);
    Work two(fresh, CERE_OP_PREFILL, {1}, 1, generation);
    cere_work items[2] = {two.item, two.item};
    two.batch.items = items;
    two.batch.item_count = 2;  // rows > decode_batch_max
    CHECK_EQ(cere_executor_submit(executor, &two.batch, &ticket), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(ticket.submission_id, 0);  // outputs stay zero on error
    // Sessions are limited by max_sessions (2).
    cere_session *third = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &third), CERE_ERR_CAPACITY);
    CHECK(third == nullptr);
    CHECK_EQ(cere_session_create(executor, model, 65, &third), CERE_ERR_CAPACITY);
    CHECK_EQ(cere_session_release(executor, fresh), CERE_OK);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_session_release(executor, session), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_stop_and_context() {
    cere_executor *executor = make_executor();
    Desc d = toy_desc();
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    const std::vector<uint32_t> prompt = {11, 12};
    const auto expected = toy_sequence(prompt, 8, vocab);
    d.stops = {expected[5]};  // model stop token
    d.sync();
    cere_model *model = sealed_model(executor, d);
    const uint64_t generation = generation_of(executor, model);

    // Item stop token (third generated token) wins before the model stop token.
    cere_session *s1 = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &s1), CERE_OK);
    Work w1(s1, CERE_OP_PREFILL, prompt, 8, generation, {expected[2]});
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &w1.batch, &ticket), CERE_OK);
    cere_completion c = drain(executor, ticket);
    CHECK_EQ(c.finish, CERE_FINISH_STOP_TOKEN);
    CHECK_EQ(c.committed_tokens, 3);  // the stop token is committed and counted
    CHECK_EQ(w1.out[2], expected[2]);
    CHECK_EQ(w1.out[3], 0xffffffffu);  // nothing written past committed_tokens

    // Model stop token.
    cere_session *s2 = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &s2), CERE_OK);
    Work w2(s2, CERE_OP_PREFILL, prompt, 8, generation);
    CHECK_EQ(cere_executor_submit(executor, &w2.batch, &ticket), CERE_OK);
    c = drain(executor, ticket);
    CHECK_EQ(c.finish, CERE_FINISH_STOP_TOKEN);
    CHECK_EQ(c.committed_tokens, 6);
    CHECK_EQ(cere_session_release(executor, s1), CERE_OK);
    CHECK_EQ(cere_session_release(executor, s2), CERE_OK);

    // Context limit: 2 prompt tokens in a 4-token session leave room for 2 tokens.
    cere_session *s3 = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 4, &s3), CERE_OK);
    Work w3(s3, CERE_OP_PREFILL, {1, 2}, 5, generation);
    CHECK_EQ(cere_executor_submit(executor, &w3.batch, &ticket), CERE_OK);
    c = drain(executor, ticket);
    CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
    CHECK_EQ(c.committed_tokens, 2);
    Work w4(s3, CERE_OP_DECODE, {}, 1, generation, {}, 1);  // full session
    CHECK_EQ(cere_executor_submit(executor, &w4.batch, &ticket), CERE_OK);
    c = drain(executor, ticket);
    CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
    CHECK_EQ(c.committed_tokens, 0);
    CHECK_EQ(cere_session_release(executor, s3), CERE_OK);
    // A prompt that fills the context leaves no room for a token.
    cere_session *s4 = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 2, &s4), CERE_OK);
    Work w5(s4, CERE_OP_PREFILL, {1, 2}, 1, generation);
    CHECK_EQ(cere_executor_submit(executor, &w5.batch, &ticket), CERE_ERR_CAPACITY);
    CHECK_EQ(cere_session_release(executor, s4), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_stale_epoch() {
    cere_executor *executor = make_executor();
    Desc d = toy_desc();
    cere_model *model = sealed_model(executor, d);
    const uint64_t generation = generation_of(executor, model);
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &session), CERE_OK);
    cere_completion_ticket ticket{};
    Work w(session, CERE_OP_PREFILL, {1, 2}, 3, generation);
    w.batch.expected_worker_epoch = kEpoch - 1;
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_ERR_STALE_EPOCH);
    w.batch.expected_worker_epoch = kEpoch;
    w.item.expected_worker_epoch = kEpoch + 1;
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_ERR_STALE_EPOCH);
    w.item.expected_worker_epoch = kEpoch;
    w.batch.capability_profile_generation = generation + 1;
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_ERR_STALE_EPOCH);
    w.batch.capability_profile_generation = generation;
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);

    cere_completion_ticket stale = ticket;
    stale.worker_epoch = kEpoch - 1;
    cere_completion c = completion_header();
    CHECK_EQ(cere_executor_poll(executor, stale, &c), CERE_ERR_STALE_EPOCH);
    CHECK_EQ(cere_executor_cancel(executor, stale), CERE_ERR_STALE_EPOCH);
    stale = ticket;
    stale.model_generation = generation + 5;
    CHECK_EQ(cere_executor_poll(executor, stale, &c), CERE_ERR_STALE_EPOCH);
    CHECK_EQ(cere_executor_cancel(executor, stale), CERE_ERR_STALE_EPOCH);
    stale = ticket;
    stale.submission_id += 100;
    CHECK_EQ(cere_executor_poll(executor, stale, &c), CERE_ERR_ARGUMENT);
    // A wrong completion header is refused and nothing advances.
    c.struct_size = 8;
    CHECK_EQ(cere_executor_poll(executor, ticket, &c), CERE_ERR_ARGUMENT);
    c = drain(executor, ticket);
    CHECK_EQ(c.committed_tokens, 3);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_cancel_busy_and_release() {
    cere_executor *executor = make_executor();
    Desc d = toy_desc();
    cere_model *model = sealed_model(executor, d);
    const uint64_t generation = generation_of(executor, model);
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &session), CERE_OK);
    Work w(session, CERE_OP_PREFILL, {3, 4}, 20, generation);
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
    uint64_t last = 0;
    for (int i = 0; i < 3; ++i) {  // non-terminal polls: committed grows and never decreases
        cere_completion c = completion_header();
        CHECK_EQ(cere_executor_poll(executor, ticket, &c), CERE_PENDING);
        CHECK_EQ(c.status, CERE_PENDING);
        CHECK(c.committed_tokens >= last);
        CHECK_EQ(c.committed_tokens, uint64_t(i + 1));
        last = c.committed_tokens;
    }
    // In flight: a second submission on the session, releases and destroy are refused.
    cere_completion_ticket other{};
    Work second(session, CERE_OP_DECODE, {}, 1, generation);
    CHECK_EQ(cere_executor_submit(executor, &second.batch, &other), CERE_ERR_BUSY);
    CHECK_EQ(cere_session_release(executor, session), CERE_ERR_BUSY);
    CHECK_EQ(cere_model_release(executor, model), CERE_ERR_BUSY);
    CHECK_EQ(cere_executor_destroy(executor), CERE_ERR_BUSY);

    // Cancel is intent: the next poll is the terminal CANCELLED fence with the committed tokens kept.
    CHECK_EQ(cere_executor_cancel(executor, ticket), CERE_OK);
    CHECK_EQ(cere_executor_cancel(executor, ticket), CERE_OK);  // idempotent until the terminal poll
    cere_completion c = completion_header();
    CHECK_EQ(cere_executor_poll(executor, ticket, &c), CERE_OK);
    CHECK_EQ(c.terminal, 1);
    CHECK_EQ(c.finish, CERE_FINISH_CANCELLED);
    CHECK_EQ(c.status, CERE_OK);
    CHECK_EQ(c.committed_tokens, 3);
    CHECK_EQ(c.session_revision, 1);
    const auto expected = toy_sequence({3, 4}, 3, cere::toy::fixture_config().vocab_size);
    for (int i = 0; i < 3; ++i) CHECK_EQ(w.out[i], expected[i]);
    CHECK_EQ(w.out[3], 0xffffffffu);

    // After the terminal poll everything can be released; destroy succeeds.
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_destroy_with_ticket_outstanding() {
    cere_executor *executor = make_executor();
    Desc d = toy_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &session), CERE_OK);
    Work w(session, CERE_OP_PREFILL, {1}, 1, generation_of(executor, model));
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_ERR_BUSY);
    cere_completion c = completion_header();
    CHECK_EQ(cere_executor_poll(executor, ticket, &c), CERE_OK);  // max_new 1: the first poll is terminal
    CHECK_EQ(c.terminal, 1);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);  // idle sessions and models are freed with the executor
}

void test_internal_error() {
    // The toy fails its second forward call: the completion is terminal ERROR with CERE_ERR_INTERNAL.
    cere_executor *executor = make_executor("toy,fail_forward=1");
    Desc d = toy_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &session), CERE_OK);
    Work w(session, CERE_OP_PREFILL, {1, 2}, 5, generation_of(executor, model));
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
    cere_completion c = drain(executor, ticket);
    CHECK_EQ(c.finish, CERE_FINISH_ERROR);
    CHECK_EQ(c.status, CERE_ERR_INTERNAL);
    CHECK_EQ(c.committed_tokens, 1);
    // Not sticky: the device is fine afterwards, so a later session is created and served (DM-2 refuses only a
    // poisoned device).
    cere_session *next = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &next), CERE_OK);
    Work again(next, CERE_OP_PREFILL, {1, 2}, 2, generation_of(executor, model));
    cere_completion_ticket again_ticket{};
    CHECK_EQ(cere_executor_submit(executor, &again.batch, &again_ticket), CERE_OK);
    const cere_completion done = drain(executor, again_ticket);
    CHECK_EQ(done.finish, CERE_FINISH_LENGTH);
    CHECK_EQ(done.status, CERE_OK);
    CHECK_EQ(cere_session_release(executor, next), CERE_OK);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_sticky_fault() {
    // fail_sticky=1 (m32n design note R10): after the fail_forward call failed, every later forward call fails
    // too (a sticky device fault the readiness re-smoke must see). DM-2: the host then refuses every later call that
    // would use the device with CERE_ERR_INTERNAL instead of accepting work that can only fail; releases still work.
    cere_executor *executor = make_executor("toy,fail_forward=1,fail_sticky=1");
    Desc d = toy_desc();
    cere_model *model = sealed_model(executor, d);
    cere_model *pending = nullptr;  // created before the fault, sealed after it
    CHECK_EQ(cere_model_create(executor, &d.desc, &pending), CERE_OK);
    cere_session *session = nullptr, *spare = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &session), CERE_OK);
    CHECK_EQ(cere_session_create(executor, model, 64, &spare), CERE_OK);
    {
        Work w(session, CERE_OP_PREFILL, {1, 2}, 5, generation_of(executor, model));
        cere_completion_ticket ticket{};
        CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
        cere_completion c = drain(executor, ticket);
        CHECK_EQ(c.finish, CERE_FINISH_ERROR);
        CHECK_EQ(c.status, CERE_ERR_INTERNAL);
        CHECK_EQ(c.committed_tokens, 1);
    }
    cere_session *later = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &later), CERE_ERR_INTERNAL);
    CHECK(later == nullptr);
    cere_model *other = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &other), CERE_ERR_INTERNAL);
    CHECK(other == nullptr);
    CHECK_EQ(cere_model_seal(executor, pending), CERE_ERR_INTERNAL);
    {
        Work w(spare, CERE_OP_PREFILL, {1, 2}, 5, generation_of(executor, model));
        cere_completion_ticket ticket{};
        CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_ERR_INTERNAL);
    }
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_session_release(executor, spare), CERE_OK);
    CHECK_EQ(cere_model_release(executor, pending), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_release_device_option() {
    // m32n design note D10: cered gives every replica of a multi-replica deployment ",release_device=1"; the toy
    // accepts it (after other options, as cered appends it) and refuses any other value.
    cere_executor *executor = make_executor("toy,fail_forward=3,release_device=1");
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    for (const char *id :
         {"toy,release_device=0", "toy,release_device=2", "toy,release_device=", "toy,release_device"}) {
        cere_executor_options options{};
        options.struct_size = sizeof(options);
        options.abi_version = CERE_EXECUTOR_ABI_VERSION;
        options.device_id = id;
        options.memory_budget_bytes = uint64_t(1) << 30;
        cere_executor *refused = nullptr;
        CHECK_EQ(cere_executor_create(&refused, &options), CERE_ERR_ARGUMENT);
        CHECK(refused == nullptr);
    }
}

void test_create_busy_option() {
    // m32n design note D21: create_busy=2 makes the executor's first two creates BUSY with nothing kept (no model,
    // no allocation, no reservation); the third creates, seals and serves as without the option.
    cere_executor *executor = make_executor("toy,create_busy=2,release_device=1");
    Desc d = toy_desc();
    for (int i = 0; i < 2; ++i) {
        cere_model *model = reinterpret_cast<cere_model *>(0x1);
        CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_ERR_BUSY);
        CHECK(model == nullptr);
        CHECK_EQ(toy(executor).allocations(), 0);
        CHECK_EQ(toy(executor).allocated_device_bytes(), 0);
        CHECK_EQ(toy(executor).allocated_host_bytes(), 0);
    }
    cere_model *model = sealed_model(executor, d);
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &session), CERE_OK);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_create_capacity_delay_option() {
    // build/page-wait-wake: create_capacity_delay_ms=200 makes a create the pool cannot cover answer CAPACITY (nothing
    // kept) only after 200 ms; a create that fits and the host's own refusal (context above the plan) answer at once.
    cere_executor *executor = make_executor("toy,create_capacity_delay_ms=200");
    Desc d(cere::toy::fixture_config(), {16}, {}, 2, 128);
    d.desc.kv_pool_tokens = 128;
    cere_model *model = sealed_model(executor, d);
    const auto elapsed_ms = [](auto start) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    };
    cere_session *first = nullptr;
    auto start = std::chrono::steady_clock::now();
    CHECK_EQ(cere_session_create(executor, model, 128, &first), CERE_OK);
    CHECK(elapsed_ms(start) < 200);
    cere_session *second = reinterpret_cast<cere_session *>(0x1);
    start = std::chrono::steady_clock::now();
    CHECK_EQ(cere_session_create(executor, model, 64, &second), CERE_ERR_CAPACITY);
    CHECK(elapsed_ms(start) >= 200);
    CHECK(second == nullptr);
    CHECK_EQ(model->sessions, 1);
    CHECK_EQ(toy(executor).free_pages(*model->device), 0);
    start = std::chrono::steady_clock::now();
    CHECK_EQ(cere_session_create(executor, model, 129, &second), CERE_ERR_CAPACITY);
    CHECK(elapsed_ms(start) < 200);
    CHECK_EQ(cere_session_release(executor, first), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_hold_after() {
    // hold_after=2: after a session's second committed token its steps answer CERE_PENDING (no token, nothing
    // consumed), so the item stays pending at a token boundary until it is cancelled.
    cere_executor *executor = make_executor("toy,hold_after=2");
    Desc d = toy_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &session), CERE_OK);
    Work w(session, CERE_OP_PREFILL, {1, 2}, 5, generation_of(executor, model));
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
    cere_completion c = completion_header();
    for (int i = 0; i < 10; ++i) {
        c = completion_header();
        CHECK_EQ(cere_executor_poll(executor, ticket, &c), CERE_PENDING);
    }
    CHECK_EQ(c.committed_tokens, 2);
    CHECK_EQ(cere_executor_cancel(executor, ticket), CERE_OK);
    c = drain(executor, ticket);
    CHECK_EQ(c.finish, CERE_FINISH_CANCELLED);
    CHECK_EQ(c.status, CERE_OK);
    CHECK_EQ(c.committed_tokens, 2);
    const auto want = toy_sequence({1, 2}, 2, d.config.vocab_size);
    CHECK_EQ(w.out[0], want[0]);
    CHECK_EQ(w.out[1], want[1]);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    // The hold is bounded: past hold_ms the step fails the item, so a test that never cancels cannot hang.
    executor = make_executor("toy,hold_after=1,hold_ms=0");
    model = sealed_model(executor, d);
    CHECK_EQ(cere_session_create(executor, model, 64, &session), CERE_OK);
    Work v(session, CERE_OP_PREFILL, {1, 2}, 5, generation_of(executor, model));
    CHECK_EQ(cere_executor_submit(executor, &v.batch, &ticket), CERE_OK);
    c = drain(executor, ticket);
    CHECK_EQ(c.finish, CERE_FINISH_ERROR);
    CHECK_EQ(c.status, CERE_ERR_INTERNAL);
    CHECK_EQ(c.committed_tokens, 1);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_hold_gate() {
    // hold_gate=PATH: the hold_after hold lasts while PATH exists; once it is gone the session runs on to its end
    // and holds no more. A relative or empty path is refused.
    for (const char *bad : {"toy,hold_gate=", "toy,hold_gate=relative/path"}) {
        cere_executor_options options{};
        options.struct_size = sizeof(options);
        options.abi_version = CERE_EXECUTOR_ABI_VERSION;
        options.device_id = bad;
        options.worker_epoch = kEpoch;
        options.memory_budget_bytes = uint64_t(1) << 30;
        cere_executor *executor = nullptr;
        CHECK_EQ(cere_executor_create(&executor, &options), CERE_ERR_ARGUMENT);
    }
    const std::string gate = "/tmp/cere-toy-hold-gate-" + std::to_string(uint64_t(::getpid()));
    std::ofstream(gate).close();
    const std::string id = "toy,hold_after=2,hold_gate=" + gate;
    cere_executor *executor = make_executor(id.c_str());
    Desc d = toy_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &session), CERE_OK);
    Work w(session, CERE_OP_PREFILL, {1, 2}, 5, generation_of(executor, model));
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
    cere_completion c = completion_header();
    for (int i = 0; i < 10; ++i) {
        c = completion_header();
        CHECK_EQ(cere_executor_poll(executor, ticket, &c), CERE_PENDING);
    }
    CHECK_EQ(c.committed_tokens, 2);
    CHECK_EQ(std::remove(gate.c_str()), 0);
    c = drain(executor, ticket);
    CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
    CHECK_EQ(c.status, CERE_OK);
    CHECK_EQ(c.committed_tokens, 5);
    const auto want = toy_sequence({1, 2}, 5, d.config.vocab_size);
    for (size_t i = 0; i < want.size(); ++i) CHECK_EQ(w.out[i], want[i]);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_unsealed_and_foreign_handles() {
    cere_executor *executor = make_executor();
    cere_executor *other = make_executor();
    Desc d = toy_desc();
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 8, &session), CERE_ERR_ARGUMENT);  // not sealed
    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(cere_executor_capabilities(executor, model, &caps), CERE_ERR_ARGUMENT);
    CHECK_EQ(caps.operations, 0);
    auto b = pattern(32, 0);
    CHECK_EQ(cere_model_put(other, model, 0, 0, b.data(), 16), CERE_ERR_ARGUMENT);  // foreign handle
    CHECK_EQ(cere_model_release(other, model), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);  // aborting an unsealed upload
    CHECK_EQ(toy(executor).allocations(), 0);
    CHECK_EQ(cere_executor_destroy(other), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

std::string hex(const std::array<uint8_t, 32> &digest) {
    static const char kDigits[] = "0123456789abcdef";
    std::string out;
    for (uint8_t b : digest) {
        out += kDigits[b >> 4];
        out += kDigits[b & 15];
    }
    return out;
}

// C10 (m211 design note D3): the in-tree SHA-256 equals the FIPS 180-4 / NIST CSHA example digests, one-shot and
// streamed in uneven pieces across block boundaries.
void test_sha256_vectors() {
    struct Vector {
        std::string message;
        const char *digest;
    };
    const Vector vectors[] = {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
        {std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"},
    };
    for (const Vector &v : vectors) {
        CHECK(hex(cere::host::sha256(v.message.data(), v.message.size())) == v.digest);
        cere::host::Sha256 streamed;
        for (size_t at = 0, piece = 1; at < v.message.size(); piece = piece * 3 % 97 + 1) {
            const size_t n = std::min(piece, v.message.size() - at);
            streamed.update(v.message.data() + at, n);
            at += n;
        }
        CHECK(hex(streamed.finish()) == v.digest);
    }
}

// ---- C3 (format layer): the CERESNP1 grammar and D14's byte checks (m211 design note D2, D3, D14) ----

namespace snap = cere::host::snapshot;

void put32(std::vector<uint8_t> &b, uint64_t at, uint32_t v) {
    for (int i = 0; i < 4; ++i) b[at + i] = uint8_t(v >> (8 * i));
}
void put64(std::vector<uint8_t> &b, uint64_t at, uint64_t v) {
    for (int i = 0; i < 8; ++i) b[at + i] = uint8_t(v >> (8 * i));
}
uint32_t get32(const std::vector<uint8_t> &b, uint64_t at) {
    return uint32_t(b[at]) | uint32_t(b[at + 1]) << 8 | uint32_t(b[at + 2]) << 16 | uint32_t(b[at + 3]) << 24;
}
uint64_t get64(const std::vector<uint8_t> &b, uint64_t at) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = v << 8 | b[at + i];
    return v;
}
uint64_t entry_at(uint32_t i) { return snap::kHeaderBytes + uint64_t(i) * snap::kEntryBytes; }

// Header offsets (D2): magic 0, format 8, header_bytes 12, section_count 16, reserved 20, layout_id 24, position 56,
// token_count 64, fed 72, flags 80, first_row 88, total_bytes 96, reserved[24] 104.
constexpr uint64_t kAtFormat = 8, kAtHeaderBytes = 12, kAtSectionCount = 16, kAtReserved = 20, kAtLayout = 24,
                   kAtFed = 72, kAtFlags = 80, kAtFirstRow = 88, kAtTotal = 96, kAtReserved24 = 104;

// A snapshot with the given sections (kind, size; offsets assigned by snap::layout): every byte pre-filled 0xA5, the
// header and pads written by snap::write_header, kind 0 holding token ids 1, 2, ..., other bodies 0x5A.
std::vector<uint8_t> build(const snap::Header &fields, const std::vector<snap::Section> &sized) {
    snap::Plan plan;
    CHECK(snap::layout(sized, &plan));
    snap::Header header = fields;
    header.total_bytes = plan.total_bytes;
    std::vector<uint8_t> b(plan.total_bytes, 0xA5);
    snap::write_header(b.data(), header, plan);
    for (const snap::Section &s : plan.sections) {
        if (s.kind == snap::kTokens) {
            for (uint64_t i = 0; i * 4 < s.bytes; ++i) put32(b, s.offset + 4 * i, uint32_t(i + 1));
        } else {
            std::memset(b.data() + s.offset, 0x5A, s.bytes);
        }
    }
    return b;
}

snap::Header fields(uint64_t position, uint64_t token_count, uint64_t fed, uint64_t first_row = 0,
                   const cere_qwen4_exp_config &cfg = a4_support().config) {
    snap::Header h;
    h.layout_id = snap::layout_id(cfg);
    h.position = position;
    h.token_count = token_count;
    h.fed = fed;
    h.first_row = first_row;
    return h;
}

std::vector<snap::Section> sizes(uint64_t position, uint64_t token_count, uint64_t first_row = 0,
                                 uint32_t kinds = snap::kRequiredKinds,
                                 const cere_qwen4_exp_config &cfg = a4_support().config) {
    std::vector<snap::Section> out;
    for (uint32_t kind = 0; kind < 32; ++kind) {
        if (((kinds >> kind) & 1) == 0) continue;
        snap::Section s;
        s.kind = kind;
        CHECK(snap::section_bytes(cfg, kind, position, first_row, token_count, &s.bytes));
        out.push_back(s);
    }
    return out;
}

std::vector<uint8_t> valid(uint64_t position, uint64_t token_count, uint64_t fed) {
    return build(fields(position, token_count, fed), sizes(position, token_count));
}

cere_status validate(const std::vector<uint8_t> &b, uint64_t model_max = 73728, uint64_t session_max = 0) {
    snap::Expect e;
    e.config = &a4_support().config;
    e.layout_id = snap::layout_id(a4_support().config);
    e.model_max_context_tokens = model_max;
    e.session_max_context_tokens = session_max;
    e.known_flags = CERE_WORK_KNOWN_FLAGS;  // ledger F
    snap::Header h;
    snap::Plan p;
    return snap::validate(b.data(), b.size(), e, &h, &p);
}

void test_snapshot_writer() {
    const cere_qwen4_exp_config &cfg = a4_support().config;
    // Sizes (spec F2/F3): 12 attention layers x 512 FP8 per K/V row, QSA 12 x 128 BF16 per complete block of 4 and
    // per valid ring entry, GDN conv 36 x 10,240 x 3 BF16, GDN ssm 36 x 48 x 128 x 128 BF16, PLE 2 x i64 + 10,240 x 9
    // BF16, sampler out_counts 248,320 x u32 (kind 8, ledger CR13).
    const uint64_t want[9] = {6 * 4,   5 * 12 * 512, 5 * 12 * 512, 1 * 12 * 256, 1 * 12 * 256,
                              2211840, 56623104,     184336,       248320 * 4};
    for (uint32_t kind = 0; kind < 9; ++kind) {
        uint64_t bytes = 0;
        CHECK(snap::section_bytes(cfg, kind, 5, 0, 6, &bytes));
        CHECK_EQ(bytes, want[kind]);
    }
    uint64_t bytes = 0;
    CHECK(!snap::section_bytes(cfg, snap::kGrammar, 5, 0, 6, &bytes));
    CHECK(!snap::section_bytes(cfg, snap::kAttnK, 5, 6, 6, &bytes));  // first_row > position
    CHECK(!snap::section_bytes(cfg, snap::kAttnK, UINT64_MAX / 4, 0, 1, &bytes));  // u64 overflow
    // Totals pinned on both sides (Rust: native::snapshot::max_snapshot_bytes, config_native.rs).
    const uint64_t totals[3][2] = {{1, 60028928}, {5, 60081152}, {73728, 1022900992}};
    for (const auto &t : totals) {
        snap::Plan plan;
        CHECK(snap::plan(cfg, t[0], 0, t[0], &plan));
        CHECK_EQ(plan.total_bytes, t[1]);
    }
    // Kind 11 (M2.13 rope_delta, CR14): 8 bytes, in an image session's plan only (kinds 0-8 and 11), on the next 256-B
    // boundary after kind 8; text plans keep the totals above. A kind this build does not write has no plan.
    CHECK(snap::section_bytes(cfg, snap::kRopeDelta, 5, 0, 6, &bytes));
    CHECK_EQ(bytes, 8);
    CHECK_EQ(snap::kWritableKinds, 0x9FFu);
    const uint32_t image_kinds = snap::kRequiredKinds | 1u << snap::kRopeDelta;
    const uint64_t image_totals[2][2] = {{1, 60028936}, {5, 60081160}};
    for (const auto &t : image_totals) {
        snap::Plan plan;
        CHECK(snap::plan(cfg, t[0], 0, t[0], &plan, image_kinds));
        CHECK_EQ(plan.total_bytes, t[1]);
        CHECK_EQ(plan.sections.size(), 10);
        CHECK_EQ(plan.sections[0].offset, 512);  // 128 + 10 x 24 = 368, as 128 + 9 x 24 = 344, rounds to 512
        CHECK_EQ(plan.sections[9].kind, snap::kRopeDelta);
        CHECK_EQ(plan.sections[9].offset % 256, 0);
        CHECK_EQ(plan.sections[9].bytes, 8);
    }
    // Kind 10 (M2.9 D8, M29S2-07): in a drafting session's plan only: per MTP layer FP8 K and V rows, BF16 QSA
    // compressed keys and ring rows, then the 16-B pending proposals. A kind this build does not write (9) has no plan.
    const uint64_t kv_width = uint64_t(cfg.num_key_value_heads) * cfg.head_dim;
    const uint64_t ratio = cfg.indexer_compress_ratio == 0 ? 1 : cfg.indexer_compress_ratio;
    for (uint64_t position = 1; position <= 9; ++position) {
        CHECK(snap::section_bytes(cfg, snap::kDraft, position, 0, position, &bytes));
        const uint64_t qsa_rows = position / ratio + position % ratio;
        CHECK_EQ(bytes, cfg.mtp_num_hidden_layers * (2 * position * kv_width + 2 * qsa_rows * cfg.indexer_head_dim) +
                            snap::kDraftProposalBytes);
    }
    CHECK_EQ(snap::kDraftWritableKinds, 0xDFFu);
    snap::Plan draft;
    CHECK(snap::plan(cfg, 5, 0, 5, &draft, snap::kRequiredKinds | 1u << snap::kDraft));
    CHECK_EQ(draft.sections.size(), 10);
    CHECK_EQ(draft.sections[9].kind, snap::kDraft);
    CHECK_EQ(draft.sections[9].offset % 256, 0);
    // A drafting session's totals, pinned on both sides (Rust: max_snapshot_bytes(.., true), config_native.rs): kind
    // 10 on the next 256-B boundary after kind 8 (sessions design note SE-09 sizes the U1 arena with them).
    const uint64_t draft_totals[3][2] = {{1, 60030224}, {5, 60086800}, {73728, 1103117072}};
    for (const auto &t : draft_totals) {
        snap::Plan plan;
        CHECK(snap::plan(cfg, t[0], 0, t[0], &plan, snap::kRequiredKinds | 1u << snap::kDraft));
        CHECK_EQ(plan.total_bytes, t[1]);
    }
    snap::Plan refused;
    CHECK(!snap::plan(cfg, 5, 0, 5, &refused, snap::kRequiredKinds | 1u << snap::kGrammar));
    // Every position mod 4, token_count = fed and fed + 1: accepted, header and pads zero, sections aligned, and two
    // writes of the same state byte-identical.
    for (uint64_t position = 1; position <= 5; ++position) {
        for (uint64_t extra = 0; extra < 2; ++extra) {
            const std::vector<uint8_t> b = valid(position, position + extra, position);
            CHECK_EQ(validate(b), CERE_OK);
            CHECK(b == valid(position, position + extra, position));
            CHECK(std::memcmp(b.data(), snap::kMagic, 8) == 0);
            CHECK_EQ(b[kAtFormat], 1);
            CHECK_EQ(b[kAtHeaderBytes], 128);
            CHECK_EQ(b[kAtSectionCount], 9);
            CHECK_EQ(get64(b, kAtTotal), b.size());
            for (uint64_t i = kAtReserved24; i < 128; ++i) CHECK_EQ(b[i], 0);
            snap::Plan plan;
            CHECK(snap::plan(cfg, position, 0, position + extra, &plan));
            uint64_t end = entry_at(9);
            for (uint32_t i = 0; i < 9; ++i) {
                const snap::Section &s = plan.sections[i];
                CHECK_EQ(s.kind, i);
                CHECK_EQ(s.offset % 256, 0);
                CHECK_EQ(b[entry_at(i) + 4], 0);  // entry reserved
                for (uint64_t at = end; at < s.offset; ++at) CHECK_EQ(b[at], 0);
                end = s.offset + s.bytes;
            }
            CHECK_EQ(end, b.size());
        }
    }
    // layout_id: deterministic, and changed by the writable-kind set (M2.11's 0-7 without the sampler) and by the
    // config, never by placement.
    CHECK(snap::layout_id(cfg) == snap::layout_id(cfg));
    CHECK(snap::layout_id(cfg) != snap::layout_id(cfg, 0xFF));   // M2.11's kinds 0-7
    CHECK(snap::layout_id(cfg) != snap::layout_id(cfg, 0x1FF));  // kind 11 changed the layout of every snapshot
    cere_qwen4_exp_config other = cfg;
    other.vocab_size += 1;
    CHECK(snap::layout_id(cfg) != snap::layout_id(other));
}

// One case per D14 byte check; session/handle checks and "session and arena unchanged" are the ABI half (C3b).
void test_snapshot_rejections() {
    const std::vector<uint8_t> base = valid(5, 6, 5);
    CHECK_EQ(validate(base), CERE_OK);
    auto mutated = [&](const std::function<void(std::vector<uint8_t> &)> &f) {
        std::vector<uint8_t> b = base;
        f(b);
        return validate(b);
    };
    // len: short, long, != total_bytes.
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { b.resize(127); }), CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { b.push_back(0); }), CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put64(b, kAtTotal, b.size() - 1); }), CERE_ERR_ARGUMENT);
    // magic, header_bytes, reserved (header u32, header [24], entry), pad (after the table, between sections).
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { b[0] = 'X'; }), CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put32(b, kAtHeaderBytes, 129); }), CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { b[kAtReserved] = 1; }), CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { b[127] = 1; }), CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { b[entry_at(3) + 4] = 1; }), CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { b[entry_at(9) + 1] = 1; }), CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { b[get64(b, entry_at(0) + 8) + 6 * 4] = 1; }), CERE_ERR_ARGUMENT);
    // format, layout_id (UNSUPPORTED), after the ARGUMENT checks before them.
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put32(b, kAtFormat, 2); }), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) {
                 put32(b, kAtFormat, 2);
                 b[kAtReserved] = 1;
             }),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { b[kAtLayout + 31] ^= 1; }), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) {
                 b[kAtLayout] ^= 1;
                 put32(b, kAtSectionCount, 17);
             }),
             CERE_ERR_UNSUPPORTED);
    // Section table: count, u64 wrap, bounds, alignment, overlap, kind order and range.
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put32(b, kAtSectionCount, 17); }), CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) {
                 put64(b, entry_at(8) + 16, UINT64_MAX - get64(b, entry_at(8) + 8) + 1);
             }),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put64(b, entry_at(8) + 16, get64(b, entry_at(8) + 16) + 256); }),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put64(b, entry_at(1) + 8, get64(b, entry_at(1) + 8) + 1); }),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put64(b, entry_at(2) + 8, get64(b, entry_at(1) + 8)); }),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put32(b, entry_at(2), 1); }), CERE_ERR_ARGUMENT);
    auto with_kind = [&](uint32_t kind, uint64_t bytes) {
        std::vector<snap::Section> s = sizes(5, 6);
        snap::Section extra;
        extra.kind = kind;
        extra.bytes = bytes;
        s.push_back(extra);
        return validate(build(fields(5, 6, 5), s));
    };
    CHECK_EQ(with_kind(9, 0), CERE_ERR_ARGUMENT);
    CHECK_EQ(with_kind(12, 256), CERE_ERR_ARGUMENT);
    CHECK_EQ(with_kind(10, 0), CERE_ERR_UNSUPPORTED);  // no M2.9 here
    // Kind 11 (M2.13 rope_delta, ledger CR14): i64 of 8 bytes with -position < rope_delta <= 0 (the next DECODE's
    // rotary position position + rope_delta >= 1, cere::mrope::positions); UNSUPPORTED where it cannot be read (a
    // model without vision).
    const auto rope = [&](uint64_t bytes, int64_t delta) {
        std::vector<snap::Section> s = sizes(5, 6);
        snap::Section r;
        r.kind = snap::kRopeDelta;
        r.bytes = bytes;
        s.push_back(r);
        std::vector<uint8_t> b = build(fields(5, 6, 5), s);
        if (bytes >= 8) put64(b, get64(b, entry_at(9) + 8), uint64_t(delta));
        return b;
    };
    CHECK_EQ(validate(rope(8, -4)), CERE_OK);
    CHECK_EQ(validate(rope(8, 0)), CERE_OK);
    CHECK_EQ(validate(rope(0, 0)), CERE_ERR_ARGUMENT);
    CHECK_EQ(validate(rope(16, 0)), CERE_ERR_ARGUMENT);
    CHECK_EQ(validate(rope(8, 1)), CERE_ERR_ARGUMENT);
    CHECK_EQ(validate(rope(8, -5)), CERE_ERR_ARGUMENT);
    CHECK_EQ(validate(rope(8, INT64_MIN)), CERE_ERR_ARGUMENT);
    {
        snap::Expect e;
        e.config = &a4_support().config;
        e.layout_id = snap::layout_id(a4_support().config);
        e.model_max_context_tokens = 73728;
        e.known_flags = 0x3;
        e.readable_kinds = snap::kWritableKinds & ~(1u << snap::kRopeDelta);
        snap::Header h;
        snap::Plan p;
        const std::vector<uint8_t> b = rope(8, -4);
        CHECK_EQ(snap::validate(b.data(), b.size(), e, &h, &p), CERE_ERR_UNSUPPORTED);
        CHECK_EQ(snap::validate(base.data(), base.size(), e, &h, &p), CERE_OK);
    }
    // Missing kind 5, missing kind 8 (the sampler counts are always written, CR13); a size off by one.
    CHECK_EQ(validate(build(fields(5, 6, 5), sizes(5, 6, 0, snap::kRequiredKinds & ~(1u << 5)))), CERE_ERR_ARGUMENT);
    CHECK_EQ(validate(build(fields(5, 6, 5), sizes(5, 6, 0, snap::kRequiredKinds & ~(1u << 8)))), CERE_ERR_ARGUMENT);
    std::vector<snap::Section> off = sizes(5, 6);
    off[6].bytes -= 1;
    CHECK_EQ(validate(build(fields(5, 6, 5), off)), CERE_ERR_ARGUMENT);
    // first_row 1 (sizes consistent with it): reserved for M2.10.
    CHECK_EQ(validate(build(fields(5, 6, 5, 1), sizes(5, 6, 1))), CERE_ERR_ARGUMENT);
    // position 0, above the model's and the session's context: CAPACITY.
    CHECK_EQ(validate(build(fields(0, 0, 0), sizes(0, 0))), CERE_ERR_CAPACITY);
    CHECK_EQ(validate(base, 4), CERE_ERR_CAPACITY);
    CHECK_EQ(validate(base, 73728, 4), CERE_ERR_CAPACITY);
    CHECK_EQ(validate(base, 73728, 5), CERE_OK);
    // Idle invariant: fed == position, fed <= token_count <= fed + 1.
    CHECK_EQ(validate(build(fields(5, 5, 4), sizes(5, 5))), CERE_ERR_ARGUMENT);
    CHECK_EQ(validate(build(fields(5, 7, 5), sizes(5, 7))), CERE_ERR_ARGUMENT);
    CHECK_EQ(validate(build(fields(5, 4, 5), sizes(5, 4))), CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put64(b, kAtFed, 4); }), CERE_ERR_ARGUMENT);
    // Token ids < vocab (the never-fed last token included).
    const uint32_t vocab = a4_support().config.vocab_size;
    CHECK_EQ(mutated([&](std::vector<uint8_t> &b) { put32(b, get64(b, entry_at(0) + 8) + 20, vocab); }),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([&](std::vector<uint8_t> &b) { put32(b, get64(b, entry_at(0) + 8) + 20, vocab - 1); }),
             CERE_OK);
    // flags within CERE_WORK_KNOWN_FLAGS.
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put64(b, kAtFlags, 0x100); }), CERE_ERR_ARGUMENT);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put64(b, kAtFlags, 0x40); }), CERE_OK);  // TreeSpec bit 6
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put64(b, kAtFlags, 0x20); }), CERE_OK);  // DraftAccept bit 5
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put64(b, kAtFlags, 0xE); }), CERE_OK);  // M2.9L bits 2, 3
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put64(b, kAtFlags, 0x10); }), CERE_OK);  // M2.9A bit 4
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put64(b, kAtFlags, 0x2); }), CERE_OK);
    CHECK_EQ(mutated([](std::vector<uint8_t> &b) { put64(b, kAtFirstRow, 0); }), CERE_OK);
}

// --- M2.6 rows ------------------------------------------------------------------------------------------------------

Desc rows_desc(uint64_t max_sessions = 4, uint64_t max_context = 64) {
    return Desc(cere::toy::fixture_config(), {16, 32}, {}, max_sessions, max_context);
}

// A session prefilled with `prompt` through one PREFILL item of max_new 1 (decode-ready afterwards).
cere_session *prefilled_session(cere_executor *executor, cere_model *model, const std::vector<uint32_t> &prompt,
                                uint64_t max_context = 64) {
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, max_context, &session), CERE_OK);
    Work w(session, CERE_OP_PREFILL, prompt, 1, generation_of(executor, model));
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
    CHECK_EQ(drain(executor, ticket).finish, CERE_FINISH_LENGTH);
    return session;
}

void test_rows_duplicate_session_argument() {
    // I3: a repeated session (or a row count outside [1, max_sessions]) is ARGUMENT before anything runs.
    cere_executor *executor = make_executor();
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *a = prefilled_session(executor, model, {1, 2});
    cere_session *b = prefilled_session(executor, model, {3, 4});
    const uint32_t ta = a->tokens.back();
    const uint32_t tb = b->tokens.back();
    cere::host::RowArgs rows[5] = {{a->device.get(), ta}, {a->device.get(), ta}, {b->device.get(), tb},
                                   {b->device.get(), tb}, {b->device.get(), tb}};
    cere_token_score out[5] = {};
    cere_status status[5] = {77, 77, 77, 77, 77};
    CHECK_EQ(toy(executor).step_rows(*model->device, rows, 2, out, status), CERE_ERR_ARGUMENT);
    CHECK_EQ(toy(executor).step_rows(*model->device, rows + 1, 0, out, status), CERE_ERR_ARGUMENT);
    CHECK_EQ(toy(executor).step_rows(*model->device, rows, 5, out, status), CERE_ERR_ARGUMENT);  // above 4 sessions
    CHECK_EQ(toy(executor).step_rows(*model->device, rows + 1, 2, out, nullptr), CERE_ERR_ARGUMENT);
    for (cere_status st : status) CHECK_EQ(st, 77);  // untouched
    // Nothing was consumed: the distinct rows {a, b} step exactly as two solo decodes would.
    CHECK_EQ(toy(executor).step_rows(*model->device, rows + 1, 2, out, status), CERE_OK);
    CHECK_EQ(status[0], CERE_OK);
    CHECK_EQ(status[1], CERE_OK);
    const uint32_t vocab = d.config.vocab_size;
    CHECK_EQ(out[0].id, toy_sequence({1, 2}, 2, vocab)[1]);
    CHECK_EQ(out[1].id, toy_sequence({3, 4}, 2, vocab)[1]);
    CHECK_EQ(out[0].runner_up, (out[0].id + 1) % vocab);
    CHECK_EQ(cere_session_release(executor, a), CERE_OK);
    CHECK_EQ(cere_session_release(executor, b), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

uint64_t committed_of(cere_executor *executor, const cere_completion_ticket &ticket) {
    return executor->tickets.at(ticket.submission_id)->committed.load();
}

// One poll that must leave the ticket pending.
void poll_pending(cere_executor *executor, const cere_completion_ticket &ticket) {
    cere_completion c = completion_header();
    CHECK_EQ(cere_executor_poll(executor, ticket, &c), CERE_PENDING);
    CHECK_EQ(c.terminal, 0);
}

// One poll that must be terminal with `finish` (and `status` for CERE_FINISH_ERROR).
cere_completion poll_terminal(cere_executor *executor, const cere_completion_ticket &ticket, uint32_t finish,
                              cere_status status = CERE_OK) {
    cere_completion c = completion_header();
    CHECK_EQ(cere_executor_poll(executor, ticket, &c), CERE_OK);
    CHECK_EQ(c.terminal, 1);
    CHECK_EQ(c.finish, finish);
    CHECK_EQ(c.status, status);
    return c;
}

// A work item and its ticket.
struct Job {
    std::unique_ptr<Work> work;
    cere_completion_ticket ticket{};
};

Job submit_job(cere_executor *executor, cere_model *model, cere_session *session, uint32_t op,
               std::vector<uint32_t> prompt, uint64_t max_new) {
    Job job;
    job.work = std::make_unique<Work>(session, op, std::move(prompt), max_new, generation_of(executor, model),
                                      std::vector<uint32_t>{}, session->revision);
    CHECK_EQ(cere_executor_submit(executor, &job.work->batch, &job.ticket), CERE_OK);
    return job;
}

// What `prompt` produces alone: one PREFILL item of max_new `n` on a fresh single-session executor.
struct Solo {
    std::vector<uint32_t> out;
    std::vector<cere_token_score> scores;
};

Solo solo_run(const std::vector<uint32_t> &prompt, uint64_t n) {
    cere_executor *executor = make_executor();
    Desc d = rows_desc(1);
    cere_model *model = sealed_model(executor, d);
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &session), CERE_OK);
    Job job = submit_job(executor, model, session, CERE_OP_PREFILL, prompt, n);
    CHECK_EQ(drain(executor, job.ticket).committed_tokens, n);
    Solo solo{job.work->out, job.work->scores};
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    return solo;
}

// The first `count` tokens and scores of `w` equal the solo run of `prompt` from its token `offset` on (offset 1 for a
// DECODE item after a max_new 1 prefill), bit for bit.
void check_solo(const Work &w, uint64_t count, const std::vector<uint32_t> &prompt, uint64_t offset) {
    const Solo solo = solo_run(prompt, offset + count);
    for (uint64_t i = 0; i < count; ++i) {
        CHECK_EQ(w.out[i], solo.out[offset + i]);
        CHECK(std::memcmp(&w.scores[i], &solo.scores[offset + i], sizeof(cere_token_score)) == 0);
    }
    if (count < w.out.size()) CHECK_EQ(w.out[count], 0xffffffffu);  // nothing written past committed_tokens
}

std::vector<uint32_t> iota_prompt(uint32_t first, uint32_t count) {
    std::vector<uint32_t> p;
    for (uint32_t i = 0; i < count; ++i) p.push_back(first + i);
    return p;
}

void test_rows_round_commits_each() {
    // D2(b): one poll runs one decode round that commits one token to each of the 4 decode-ready tickets.
    cere_executor *executor = make_executor();
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    const std::vector<std::vector<uint32_t>> prompts = {{1, 2}, {3, 4, 5}, {6}, {7, 8}};
    std::vector<cere_session *> sessions;
    std::vector<Job> jobs;
    for (const auto &p : prompts) sessions.push_back(prefilled_session(executor, model, p));
    for (cere_session *s : sessions) jobs.push_back(submit_job(executor, model, s, CERE_OP_DECODE, {}, 4));
    poll_pending(executor, jobs[0].ticket);
    for (const Job &j : jobs) CHECK_EQ(committed_of(executor, j.ticket), 1);
    for (size_t i = 0; i < jobs.size(); ++i) {
        const cere_completion c = drain(executor, jobs[i].ticket);
        CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
        CHECK_EQ(c.committed_tokens, 4);
        check_solo(*jobs[i].work, 4, prompts[i], 1);
    }
    for (cere_session *s : sessions) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_rows_prefill_chunk_alternates() {
    // D2(a)/(c): a 10-token prompt at chunk 4 runs one chunk per prefill round (4, 4, 2), alternating with decode
    // rounds while both are pending; then both tickets decode in one round.
    cere_executor *executor = make_executor("toy,prefill_chunk=4");
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *ds = prefilled_session(executor, model, {1, 2});  // the last round was a prefill round
    Job dj = submit_job(executor, model, ds, CERE_OP_DECODE, {}, 10);
    cere_session *ps = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &ps), CERE_OK);
    const auto prompt = iota_prompt(11, 10);
    Job pj = submit_job(executor, model, ps, CERE_OP_PREFILL, prompt, 3);
    struct Seen {
        uint64_t fed, decode, prefill;
    };
    const std::vector<Seen> want = {{0, 1, 0}, {4, 1, 0}, {4, 2, 0}, {8, 2, 0},
                                    {8, 3, 0}, {10, 3, 1}, {11, 4, 2}, {12, 5, 3}};
    for (const Seen &w : want) {
        poll_pending(executor, dj.ticket);
        CHECK_EQ(ps->fed, w.fed);
        CHECK_EQ(committed_of(executor, dj.ticket), w.decode);
        CHECK_EQ(committed_of(executor, pj.ticket), w.prefill);
    }
    CHECK_EQ(poll_terminal(executor, pj.ticket, CERE_FINISH_LENGTH).committed_tokens, 3);
    CHECK_EQ(drain(executor, dj.ticket).committed_tokens, 10);
    check_solo(*pj.work, 3, prompt, 0);
    check_solo(*dj.work, 10, {1, 2}, 1);
    CHECK_EQ(cere_session_release(executor, ds), CERE_OK);
    CHECK_EQ(cere_session_release(executor, ps), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_rows_join_leave() {
    // A ticket submitted mid-decode joins the rows after its prefill; a terminal one leaves (its finish is returned at
    // its own next poll); every ticket's tokens and scores equal its solo run.
    cere_executor *executor = make_executor();
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *as = prefilled_session(executor, model, {1, 2});
    cere_session *bs = prefilled_session(executor, model, {3, 4});
    Job a = submit_job(executor, model, as, CERE_OP_DECODE, {}, 2);
    Job b = submit_job(executor, model, bs, CERE_OP_DECODE, {}, 6);
    poll_pending(executor, b.ticket);  // decode round {a, b}
    CHECK_EQ(committed_of(executor, a.ticket), 1);
    CHECK_EQ(committed_of(executor, b.ticket), 1);
    cere_session *cs = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &cs), CERE_OK);
    Job c = submit_job(executor, model, cs, CERE_OP_PREFILL, {5, 6, 7}, 3);
    poll_pending(executor, b.ticket);  // prefill round {c}
    CHECK_EQ(committed_of(executor, a.ticket), 1);
    CHECK_EQ(committed_of(executor, b.ticket), 1);
    CHECK_EQ(committed_of(executor, c.ticket), 1);
    poll_pending(executor, b.ticket);  // decode round {a, b, c}: a reaches its length
    CHECK_EQ(committed_of(executor, a.ticket), 2);
    CHECK_EQ(committed_of(executor, b.ticket), 2);
    CHECK_EQ(committed_of(executor, c.ticket), 2);
    // S1 G3: rounds ran since a's submit, so its poll reports a's finish without running one.
    CHECK_EQ(poll_terminal(executor, a.ticket, CERE_FINISH_LENGTH).committed_tokens, 2);
    CHECK_EQ(committed_of(executor, b.ticket), 2);
    CHECK_EQ(committed_of(executor, c.ticket), 2);
    poll_pending(executor, b.ticket);  // a left: round {b, c}, c reaches its length
    CHECK_EQ(committed_of(executor, b.ticket), 3);
    CHECK_EQ(committed_of(executor, c.ticket), 3);
    CHECK_EQ(poll_terminal(executor, c.ticket, CERE_FINISH_LENGTH).committed_tokens, 3);
    poll_pending(executor, b.ticket);  // c left: round {b}
    CHECK_EQ(committed_of(executor, b.ticket), 4);
    CHECK_EQ(drain(executor, b.ticket).committed_tokens, 6);
    check_solo(*a.work, 2, {1, 2}, 1);
    check_solo(*b.work, 6, {3, 4}, 1);
    check_solo(*c.work, 3, {5, 6, 7}, 0);
    for (cere_session *s : {as, bs, cs}) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_rows_cancel_one() {
    // A cancelled ticket leaves the next round; the others keep their rows; it ends CANCELLED at its own poll.
    cere_executor *executor = make_executor();
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    const std::vector<std::vector<uint32_t>> prompts = {{1, 2}, {3, 4}, {5, 6}};
    std::vector<cere_session *> sessions;
    std::vector<Job> jobs;
    for (const auto &p : prompts) sessions.push_back(prefilled_session(executor, model, p));
    for (cere_session *s : sessions) jobs.push_back(submit_job(executor, model, s, CERE_OP_DECODE, {}, 8));
    poll_pending(executor, jobs[0].ticket);
    CHECK_EQ(cere_executor_cancel(executor, jobs[1].ticket), CERE_OK);
    poll_pending(executor, jobs[0].ticket);
    poll_pending(executor, jobs[0].ticket);
    CHECK_EQ(committed_of(executor, jobs[0].ticket), 3);
    CHECK_EQ(committed_of(executor, jobs[1].ticket), 1);
    CHECK_EQ(committed_of(executor, jobs[2].ticket), 3);
    const cere_completion c = poll_terminal(executor, jobs[1].ticket, CERE_FINISH_CANCELLED);
    CHECK_EQ(c.committed_tokens, 1);
    CHECK_EQ(c.session_revision, 2);
    CHECK_EQ(drain(executor, jobs[0].ticket).finish, CERE_FINISH_LENGTH);
    CHECK_EQ(drain(executor, jobs[2].ticket).finish, CERE_FINISH_LENGTH);
    check_solo(*jobs[0].work, 8, prompts[0], 1);
    check_solo(*jobs[1].work, 1, prompts[1], 1);
    check_solo(*jobs[2].work, 8, prompts[2], 1);
    for (cere_session *s : sessions) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// One bad row (`bad`, a DECODE item of max_new 5) in a round with 3 healthy rows (max_new 12): only the bad ticket
// ends, with its single-row outcome; the healthy rows commit one token per poll, never stalling.
void run_one_bad_row(cere_executor *executor, cere_model *model, cere_session *bad, uint32_t finish,
                     cere_status status) {
    const std::vector<std::vector<uint32_t>> prompts = {{3, 4}, {5, 6}, {7, 8}};
    std::vector<cere_session *> sessions;
    std::vector<Job> jobs;
    for (const auto &p : prompts) sessions.push_back(prefilled_session(executor, model, p));
    Job x = submit_job(executor, model, bad, CERE_OP_DECODE, {}, 5);
    for (cere_session *s : sessions) jobs.push_back(submit_job(executor, model, s, CERE_OP_DECODE, {}, 12));
    poll_pending(executor, jobs[0].ticket);
    for (const Job &j : jobs) CHECK_EQ(committed_of(executor, j.ticket), 1);
    CHECK_EQ(poll_terminal(executor, x.ticket, finish, status).committed_tokens, 0);  // S1 G3: no round first
    for (uint64_t poll = 2; poll <= 11; ++poll) {  // no livelock over 10 polls: each commits one token to each row
        poll_pending(executor, jobs[0].ticket);
        for (const Job &j : jobs) CHECK_EQ(committed_of(executor, j.ticket), poll);
    }
    for (size_t i = 0; i < jobs.size(); ++i) {
        CHECK_EQ(drain(executor, jobs[i].ticket).committed_tokens, 12);
        check_solo(*jobs[i].work, 12, prompts[i], 1);
    }
    for (cere_session *s : sessions) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
}

void test_rows_one_bad_row() {
    // A broken session (its single-row decode failed, fail_forward=1) errors INTERNAL alone.
    cere_executor *executor = make_executor("toy,fail_forward=1");
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *broken = prefilled_session(executor, model, {1, 2});
    Job fail = submit_job(executor, model, broken, CERE_OP_DECODE, {}, 1);
    poll_terminal(executor, fail.ticket, CERE_FINISH_ERROR, CERE_ERR_INTERNAL);
    run_one_bad_row(executor, model, broken, CERE_FINISH_ERROR, CERE_ERR_INTERNAL);
    CHECK_EQ(cere_session_release(executor, broken), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    // An over-context session (full after its prefill) ends LENGTH alone, as its single-row step would.
    executor = make_executor();
    model = sealed_model(executor, d);
    cere_session *full = prefilled_session(executor, model, {1, 2}, 3);
    run_one_bad_row(executor, model, full, CERE_FINISH_LENGTH, CERE_OK);
    CHECK_EQ(cere_session_release(executor, full), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    // Ledger CR1 / M5 (m25 D3 per row): on a pooled model (pool 256 tokens = 4 pages, taken whole by a 3-token session
    // and three 64-token ones) a row at its session's context is CAPACITY inside step_rows before anything runs and
    // keeps its state (nothing consumed: the same answer again), while the other rows step as their solo runs. No row
    // ever needs a page.
    executor = make_executor();
    Desc pooled = rows_desc();
    pooled.desc.kv_pool_tokens = 256;
    model = sealed_model(executor, pooled);
    const cere::host::DeviceModel &pool = *model->device;
    const std::vector<std::vector<uint32_t>> prompts = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    std::vector<cere_session *> sessions;
    for (size_t i = 0; i < prompts.size(); ++i) {
        sessions.push_back(prefilled_session(executor, model, prompts[i], i == 0 ? 3 : 64));
    }
    CHECK_EQ(toy(executor).free_pages(pool), 0);
    const uint32_t vocab = pooled.config.vocab_size;
    cere::host::RowArgs rows[4];
    for (size_t i = 0; i < sessions.size(); ++i) rows[i] = {sessions[i]->device.get(), sessions[i]->tokens.back()};
    for (uint64_t round = 1; round <= 3; ++round) {
        cere_token_score out[4] = {};
        cere_status status[4] = {77, 77, 77, 77};
        out[0].id = 0xdeadu;
        CHECK_EQ(toy(executor).step_rows(*model->device, rows, 4, out, status), CERE_OK);
        // Round 1 fills the 3-token session (2 prompt tokens + 1 step); from round 2 it is at its context.
        CHECK_EQ(status[0], round == 1 ? CERE_OK : CERE_ERR_CAPACITY);
        if (round == 1) {
            CHECK_EQ(out[0].id, toy_sequence(prompts[0], 2, vocab)[1]);
        } else {
            CHECK_EQ(out[0].id, 0xdeadu);  // no score written for the refused row
        }
        for (size_t i = 1; i < 4; ++i) {
            CHECK_EQ(status[i], CERE_OK);
            CHECK_EQ(out[i].id, toy_sequence(prompts[i], round + 1, vocab)[round]);
            rows[i].token = out[i].id;
        }
        CHECK_EQ(toy(executor).free_pages(pool), 0);
    }
    for (cere_session *s : sessions) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(toy(executor).free_pages(pool), 4);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_rows_fail_forward_breaks_participants() {
    // fail_forward counts one per step/step_rows call. Three prefills are calls 0-2; the decode round {d0, d1, d2} is
    // call 3 and the prefill of p call 4.
    for (uint64_t fail : {3, 4}) {
        const std::string id = "toy,fail_forward=" + std::to_string(fail);
        cere_executor *executor = make_executor(id.c_str());
        Desc d = rows_desc();
        cere_model *model = sealed_model(executor, d);
        const std::vector<std::vector<uint32_t>> prompts = {{1, 2}, {3, 4}, {5, 6}};
        std::vector<cere_session *> sessions;
        std::vector<Job> jobs;
        for (const auto &p : prompts) sessions.push_back(prefilled_session(executor, model, p));
        for (cere_session *s : sessions) jobs.push_back(submit_job(executor, model, s, CERE_OP_DECODE, {}, 4));
        cere_session *ps = nullptr;
        CHECK_EQ(cere_session_create(executor, model, 64, &ps), CERE_OK);
        Job p = submit_job(executor, model, ps, CERE_OP_PREFILL, {9, 10}, 2);
        poll_pending(executor, p.ticket);  // decode round (call 3)
        if (fail == 3) {
            // Every participant breaks; the non-participant p is untouched and then runs as alone.
            CHECK_EQ(committed_of(executor, p.ticket), 0);
            CHECK_EQ(ps->fed, 0);
            for (const Job &j : jobs) {
                CHECK_EQ(poll_terminal(executor, j.ticket, CERE_FINISH_ERROR, CERE_ERR_INTERNAL).committed_tokens,
                         0);
            }
            CHECK_EQ(drain(executor, p.ticket).committed_tokens, 2);
            check_solo(*p.work, 2, {9, 10}, 0);
            Job again = submit_job(executor, model, sessions[0], CERE_OP_DECODE, {}, 1);  // the session is broken
            CHECK_EQ(drain(executor, again.ticket).status, CERE_ERR_INTERNAL);
        } else {
            // The 3-row round was one call: it ran; the next call (p's prefill) is the one that fails.
            for (const Job &j : jobs) CHECK_EQ(committed_of(executor, j.ticket), 1);
            poll_terminal(executor, p.ticket, CERE_FINISH_ERROR, CERE_ERR_INTERNAL);
            for (size_t i = 0; i < jobs.size(); ++i) {
                CHECK_EQ(drain(executor, jobs[i].ticket).committed_tokens, 4);
                check_solo(*jobs[i].work, 4, prompts[i], 1);
            }
        }
        for (cere_session *s : sessions) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        CHECK_EQ(cere_session_release(executor, ps), CERE_OK);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
}

void test_rows_pending_excludes_row() {
    // hold_after=3: a's row answers PENDING after its third token; it is skipped while b and c advance.
    cere_executor *executor = make_executor("toy,hold_after=3");
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *as = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &as), CERE_OK);
    Job a = submit_job(executor, model, as, CERE_OP_PREFILL, {1, 2}, 10);
    for (int i = 0; i < 4; ++i) poll_pending(executor, a.ticket);
    CHECK_EQ(committed_of(executor, a.ticket), 3);  // held
    cere_session *bs = nullptr;
    cere_session *cs = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &bs), CERE_OK);
    CHECK_EQ(cere_session_create(executor, model, 64, &cs), CERE_OK);
    Job b = submit_job(executor, model, bs, CERE_OP_PREFILL, {3, 4}, 3);
    Job c = submit_job(executor, model, cs, CERE_OP_PREFILL, {5, 6}, 3);
    poll_pending(executor, b.ticket);  // prefill b
    poll_pending(executor, b.ticket);  // prefill c: the pass goes on, no decode round between (BC4 (b))
    CHECK_EQ(committed_of(executor, b.ticket), 1);
    CHECK_EQ(committed_of(executor, c.ticket), 1);
    poll_pending(executor, b.ticket);  // decode {a held, b, c}
    CHECK_EQ(committed_of(executor, b.ticket), 2);
    CHECK_EQ(committed_of(executor, c.ticket), 2);
    poll_terminal(executor, b.ticket, CERE_FINISH_LENGTH);  // decode {a held, b, c}
    CHECK_EQ(committed_of(executor, c.ticket), 3);
    CHECK_EQ(committed_of(executor, a.ticket), 3);
    CHECK_EQ(drain(executor, c.ticket).committed_tokens, 3);
    CHECK_EQ(committed_of(executor, a.ticket), 3);
    CHECK_EQ(cere_executor_cancel(executor, a.ticket), CERE_OK);
    CHECK_EQ(poll_terminal(executor, a.ticket, CERE_FINISH_CANCELLED).committed_tokens, 3);
    check_solo(*a.work, 3, {1, 2}, 0);
    check_solo(*b.work, 3, {3, 4}, 0);
    check_solo(*c.work, 3, {5, 6}, 0);
    for (cere_session *s : {as, bs, cs}) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_rows_interleave_long_prefill() {
    // C10: a 14-token prompt at chunk 4 (4 chunks) submitted while 2 rows decode advances at most one chunk per round;
    // the decoding rows commit between chunks; a cancel of a decoding row lands within one chunk.
    cere_executor *executor = make_executor("toy,prefill_chunk=4");
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *as = prefilled_session(executor, model, {1, 2});
    cere_session *bs = prefilled_session(executor, model, {3, 4});
    Job a = submit_job(executor, model, as, CERE_OP_DECODE, {}, 20);
    Job b = submit_job(executor, model, bs, CERE_OP_DECODE, {}, 20);
    cere_session *ps = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &ps), CERE_OK);
    const auto prompt = iota_prompt(100, 14);
    Job p = submit_job(executor, model, ps, CERE_OP_PREFILL, prompt, 2);
    poll_pending(executor, b.ticket);  // decode {a, b}
    poll_pending(executor, b.ticket);  // chunk 1
    CHECK_EQ(ps->fed, 4);
    CHECK_EQ(committed_of(executor, a.ticket), 1);
    CHECK_EQ(cere_executor_cancel(executor, a.ticket), CERE_OK);
    CHECK_EQ(poll_terminal(executor, a.ticket, CERE_FINISH_CANCELLED).committed_tokens, 1);
    CHECK_EQ(ps->fed, 4);  // the cancel landed before the next chunk
    uint64_t fed = ps->fed;
    uint64_t decoded = committed_of(executor, b.ticket);
    bool last_prefill = true;  // the last round was chunk 1 (S1 G3: the terminal poll above ran none)
    uint64_t chunks = 1;
    while (committed_of(executor, p.ticket) == 0) {
        poll_pending(executor, b.ticket);
        const uint64_t fed_now = ps->fed;
        const uint64_t decoded_now = committed_of(executor, b.ticket);
        const bool prefill = fed_now != fed || committed_of(executor, p.ticket) != 0;
        CHECK(prefill != last_prefill);  // alternating
        if (prefill) {
            CHECK(fed_now - fed <= 4 || committed_of(executor, p.ticket) == 1);
            CHECK_EQ(decoded_now, decoded);
            ++chunks;
        } else {
            CHECK_EQ(decoded_now, decoded + 1);
        }
        fed = fed_now;
        decoded = decoded_now;
        last_prefill = prefill;
    }
    CHECK_EQ(chunks, 4);
    CHECK_EQ(ps->fed, 14);
    CHECK_EQ(drain(executor, p.ticket).committed_tokens, 2);
    CHECK_EQ(drain(executor, b.ticket).committed_tokens, 20);
    check_solo(*a.work, 1, {1, 2}, 1);
    check_solo(*b.work, 20, {3, 4}, 1);
    check_solo(*p.work, 2, prompt, 0);
    for (cere_session *s : {as, bs, ps}) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_rows_cancel_during_prefill() {
    // C10: cancelling the prefilling ticket mid-prompt ends it before its next chunk; the decoding row continues.
    cere_executor *executor = make_executor("toy,prefill_chunk=4");
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *ds = prefilled_session(executor, model, {1, 2});
    Job dj = submit_job(executor, model, ds, CERE_OP_DECODE, {}, 5);
    cere_session *ps = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &ps), CERE_OK);
    Job p = submit_job(executor, model, ps, CERE_OP_PREFILL, iota_prompt(100, 14), 2);
    poll_pending(executor, p.ticket);  // decode {d}
    poll_pending(executor, p.ticket);  // chunk 1
    CHECK_EQ(ps->fed, 4);
    CHECK_EQ(cere_executor_cancel(executor, p.ticket), CERE_OK);
    const cere_completion c = poll_terminal(executor, p.ticket, CERE_FINISH_CANCELLED);
    CHECK_EQ(c.committed_tokens, 0);
    CHECK_EQ(c.session_revision, 1);
    CHECK_EQ(ps->fed, 4);  // no further chunk ran
    CHECK_EQ(committed_of(executor, dj.ticket), 2);  // the cancelling poll's round was decode {d}
    CHECK_EQ(drain(executor, dj.ticket).committed_tokens, 5);
    check_solo(*dj.work, 5, {1, 2}, 1);
    CHECK_EQ(cere_session_release(executor, ds), CERE_OK);
    CHECK_EQ(cere_session_release(executor, ps), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// O79 (m210 design note PF1): two rows decode while two prompts prefill (chunk 5: p's 14 tokens in 3 chunks, q's 4
// in 1) under `rounds` decode rounds per chunk, set after the rows' own prefills (so 1 round is still owed). BC4 (b)
// (batch-concurrency design note): the chunks of one round-robin pass run back to back (p's first, then q's, which
// was submitted after p: the later short prompt waits one chunk, not p's whole prompt), and each pass is followed by
// exactly `rounds` decode rounds before the next (the rows outlast every gap). Every ticket's tokens and scores are
// its solo run's. Returns each ticket's tokens and scores (rows a, b, prompts p, q).
std::vector<Solo> rows_decode_rounds_run(uint32_t rounds) {
    cere_executor *executor = make_executor("toy,prefill_chunk=5");
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *as = prefilled_session(executor, model, {1, 2});
    cere_session *bs = prefilled_session(executor, model, {3, 4});
    CHECK_EQ(cere_executor_set_decode_rounds(executor, rounds), CERE_OK);
    Job a = submit_job(executor, model, as, CERE_OP_DECODE, {}, 56);
    Job b = submit_job(executor, model, bs, CERE_OP_DECODE, {}, 56);
    cere_session *ps = nullptr;
    cere_session *qs = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &ps), CERE_OK);
    CHECK_EQ(cere_session_create(executor, model, 64, &qs), CERE_OK);
    const auto p_prompt = iota_prompt(100, 14);
    const auto q_prompt = iota_prompt(200, 4);
    Job p = submit_job(executor, model, ps, CERE_OP_PREFILL, p_prompt, 3);
    Job q = submit_job(executor, model, qs, CERE_OP_PREFILL, q_prompt, 3);
    // A round is a prefill round when it consumed prompt (session.fed also counts the decoded tokens it feeds).
    const auto prompt_fed = [&] {
        return std::min<uint64_t>(ps->fed, p_prompt.size()) + std::min<uint64_t>(qs->fed, q_prompt.size());
    };
    std::string kinds;  // one letter per round until both prompts are consumed: P prefill, D decode
    while ((ps->fed < p_prompt.size() || qs->fed < q_prompt.size()) && kinds.size() < 100) {
        const uint64_t fed = prompt_fed();
        poll_pending(executor, a.ticket);
        kinds += prompt_fed() != fed ? 'P' : 'D';
    }
    const std::string gap(rounds, 'D');
    const std::string want = "DPP" + gap + "P" + gap + "P";
    CHECK(kinds == want);
    if (kinds != want) std::fprintf(stderr, "rounds %u: kinds %s, expected %s\n", rounds, kinds.c_str(), want.c_str());
    CHECK_EQ(drain(executor, a.ticket).committed_tokens, 56);
    CHECK_EQ(drain(executor, b.ticket).committed_tokens, 56);
    CHECK_EQ(drain(executor, p.ticket).committed_tokens, 3);
    CHECK_EQ(drain(executor, q.ticket).committed_tokens, 3);
    check_solo(*a.work, 56, {1, 2}, 1);
    check_solo(*b.work, 56, {3, 4}, 1);
    check_solo(*p.work, 3, p_prompt, 0);
    check_solo(*q.work, 3, q_prompt, 0);
    std::vector<Solo> out;
    for (const Job *j : {&a, &b, &p, &q}) out.push_back(Solo{j->work->out, j->work->scores});
    for (cere_session *s : {as, bs, ps, qs}) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    return out;
}

void test_rows_decode_rounds_per_chunk() {
    // O79 PF1: the schedule changes no ticket's bits; N in {4, 16} gives every ticket N = 1's tokens and scores.
    const std::vector<Solo> base = rows_decode_rounds_run(1);
    for (uint32_t rounds : {4u, 16u}) {
        const std::vector<Solo> run = rows_decode_rounds_run(rounds);
        CHECK_EQ(run.size(), base.size());
        for (size_t i = 0; i < run.size() && i < base.size(); ++i) {
            CHECK(run[i].out == base[i].out);
            CHECK(run[i].scores.size() == base[i].scores.size() &&
                  std::memcmp(run[i].scores.data(), base[i].scores.data(),
                              run[i].scores.size() * sizeof(cere_token_score)) == 0);
        }
    }
}

void test_rows_decode_rounds_setting() {
    // O79 PF1: 1..CERE_MAX_DECODE_ROUNDS_PER_CHUNK, else ARGUMENT (the setting unchanged); a change takes effect at
    // the next round (rounds still owed are capped by the new count).
    CHECK_EQ(cere_executor_set_decode_rounds(nullptr, 1), CERE_ERR_ARGUMENT);
    cere_executor *executor = make_executor("toy,prefill_chunk=4");
    CHECK_EQ(executor->decode_rounds_per_chunk, 1);
    CHECK_EQ(cere_executor_set_decode_rounds(executor, 0), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_set_decode_rounds(executor, CERE_MAX_DECODE_ROUNDS_PER_CHUNK + 1), CERE_ERR_ARGUMENT);
    CHECK_EQ(executor->decode_rounds_per_chunk, 1);
    CHECK_EQ(cere_executor_set_decode_rounds(executor, CERE_MAX_DECODE_ROUNDS_PER_CHUNK), CERE_OK);
    CHECK_EQ(executor->decode_rounds_per_chunk, 64);
    CHECK_EQ(cere_executor_set_decode_rounds(executor, 1), CERE_OK);
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *ds = prefilled_session(executor, model, {1, 2});  // its prefill round leaves 1 decode round owed
    Job dj = submit_job(executor, model, ds, CERE_OP_DECODE, {}, 20);
    poll_pending(executor, dj.ticket);  // decode, nothing owed after it
    CHECK_EQ(committed_of(executor, dj.ticket), 1);
    CHECK_EQ(cere_executor_set_decode_rounds(executor, 4), CERE_OK);
    cere_session *ps = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &ps), CERE_OK);
    const auto prompt = iota_prompt(100, 12);
    Job pj = submit_job(executor, model, ps, CERE_OP_PREFILL, prompt, 2);
    poll_pending(executor, dj.ticket);  // chunk 1
    CHECK_EQ(ps->fed, 4);
    poll_pending(executor, dj.ticket);  // decode 1 of 4
    poll_pending(executor, dj.ticket);  // decode 2 of 4
    CHECK_EQ(ps->fed, 4);
    CHECK_EQ(committed_of(executor, dj.ticket), 3);
    CHECK_EQ(cere_executor_set_decode_rounds(executor, 1), CERE_OK);  // 2 owed, capped to 1
    poll_pending(executor, dj.ticket);                                 // the last decode round owed
    CHECK_EQ(ps->fed, 4);
    CHECK_EQ(committed_of(executor, dj.ticket), 4);
    poll_pending(executor, dj.ticket);  // chunk 2
    CHECK_EQ(ps->fed, 8);
    poll_pending(executor, dj.ticket);  // decode: the kinds alternate from here on
    CHECK_EQ(committed_of(executor, dj.ticket), 5);
    poll_pending(executor, dj.ticket);  // chunk 3 (the last: commits p's first token)
    CHECK_EQ(ps->fed, 12);
    CHECK_EQ(drain(executor, pj.ticket).committed_tokens, 2);
    CHECK_EQ(drain(executor, dj.ticket).committed_tokens, 20);
    check_solo(*dj.work, 20, {1, 2}, 1);
    check_solo(*pj.work, 2, prompt, 0);
    for (cere_session *s : {ds, ps}) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// Track LoRA (lora design note): an adapter of the toy fixture config (one layer, no full-attention layer, shared
// expert width 0, hidden 32), whose targets are the three shared-expert modules: A [rank, 32] for gate and up, [rank,
// 0] for down; B [0, rank] for gate and up, [32, rank] for down. Empty tensors keep one element (non-null pointers).
struct ToyAdapter {
    uint32_t rank;
    float scale;
    std::vector<uint32_t> targets{CERE_LORA_SHARED_GATE, CERE_LORA_SHARED_UP, CERE_LORA_SHARED_DOWN};
    std::vector<std::vector<uint16_t>> a, b;
    std::vector<const void *> ap, bp;

    ToyAdapter(uint32_t r, float s, uint16_t seed) : rank(r), scale(s) {
        const cere_qwen4_exp_config cfg = cere::toy::fixture_config();
        for (uint32_t t : targets) {
            cere::host::lora::Dims dims;
            CHECK(cere::host::lora::dims(cfg, t, &dims));
            a.emplace_back(std::max<uint64_t>(1, uint64_t(rank) * dims.in));
            b.emplace_back(std::max<uint64_t>(1, uint64_t(dims.out) * rank));
            for (size_t i = 0; i < a.back().size(); ++i) a.back()[i] = uint16_t(seed + 7 * i + t);
            for (size_t i = 0; i < b.back().size(); ++i) b.back()[i] = uint16_t(seed + 13 * i + t);
        }
        for (size_t i = 0; i < targets.size(); ++i) {
            ap.push_back(a[i].data());
            bp.push_back(b[i].data());
        }
    }
    cere_status load(cere_executor *executor, cere_model *model, uint32_t slot) const {
        return cere_model_lora_load(executor, model, slot, rank, scale, targets.size(), targets.data(), ap.data(),
                                    bp.data());
    }
    // The toy's documented offset (toy_device.cpp lora_load): FNV-1a 64 over rank, scale bits and per target its id,
    // A bytes and B bytes (the tensors' real sizes), little-endian; 1 + digest mod (vocab - 1).
    uint32_t offset(uint32_t vocab) const {
        const cere_qwen4_exp_config cfg = cere::toy::fixture_config();
        uint64_t h = 14695981039346656037ull;
        const auto mix = [&h](const void *p, uint64_t n) {
            for (uint64_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t *>(p)[i]) * 1099511628211ull;
        };
        uint32_t bits = 0;
        std::memcpy(&bits, &scale, 4);
        mix(&rank, 4);
        mix(&bits, 4);
        for (size_t i = 0; i < targets.size(); ++i) {
            cere::host::lora::Dims dims;
            cere::host::lora::dims(cfg, targets[i], &dims);
            mix(&targets[i], 4);
            mix(a[i].data(), uint64_t(rank) * dims.in * 2);
            mix(b[i].data(), uint64_t(dims.out) * rank * 2);
        }
        return uint32_t(1 + h % (vocab - 1));
    }
};

void test_lora_api() {
    // Track LoRA: reserve before seal (once, bounds, budget), load into an empty slot of a sealed model with validated
    // targets, set_lora before the first submit to a loaded slot, unload BUSY while a session uses the slot.
    const cere_qwen4_exp_config cfg = cere::toy::fixture_config();
    const uint64_t per_rank = 2 * (uint64_t(32) + 0) + (0 + 32);  // gate, up: in 32 out 0; down: in 0 out 32
    CHECK_EQ(cere::host::lora::elements_per_rank(cfg), per_rank);
    CHECK_EQ(cere::host::lora::targets(cfg).size(), 3);
    CHECK_EQ(cere::host::lora::padded_rank(1), 8);
    CHECK_EQ(cere::host::lora::padded_rank(16), 16);
    CHECK_EQ(cere::host::lora::padded_rank(17), 24);
    const uint64_t pool = cere::host::lora::pool_bytes(cfg, 2, 12);
    CHECK_EQ(pool, 2 * 16 * per_rank * 2);

    cere_executor *executor = make_executor();
    Desc d = rows_desc();
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    const uint64_t reserved = executor->reserved_device_bytes;
    const uint64_t allocated = toy(executor).allocated_device_bytes();
    CHECK_EQ(cere_model_lora_reserve(nullptr, model, 2, 12), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_lora_reserve(executor, nullptr, 2, 12), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_lora_reserve(executor, model, 0, 12), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_lora_reserve(executor, model, CERE_LORA_MAX_ADAPTERS + 1, 12), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_lora_reserve(executor, model, 2, 0), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_lora_reserve(executor, model, 2, CERE_LORA_MAX_RANK + 1), CERE_ERR_ARGUMENT);
    CHECK_EQ(executor->reserved_device_bytes, reserved);
    CHECK_EQ(cere_model_lora_reserve(executor, model, 2, 12), CERE_OK);
    CHECK_EQ(executor->reserved_device_bytes, reserved + pool);
    CHECK_EQ(toy(executor).allocated_device_bytes(), allocated + pool);
    CHECK_EQ(cere_model_lora_reserve(executor, model, 2, 12), CERE_ERR_ARGUMENT);  // once
    const ToyAdapter adapter(12, 0.5f, 3);
    CHECK_EQ(adapter.load(executor, model, 0), CERE_ERR_ARGUMENT);  // before seal
    for (uint64_t i = 0; i < d.tensors.size(); ++i) {
        auto bytes = pattern(d.tensors[i].bytes, uint8_t(i));
        CHECK_EQ(cere_model_put(executor, model, i, 0, bytes.data(), bytes.size()), CERE_OK);
    }
    CHECK_EQ(cere_model_seal(executor, model), CERE_OK);
    CHECK_EQ(cere_model_lora_reserve(executor, model, 1, 8), CERE_ERR_ARGUMENT);  // sealed (and reserved)

    // Load arguments: every rejection leaves the slot empty.
    const auto load = [&](uint32_t slot, uint32_t rank, float scale, std::vector<uint32_t> targets) {
        return cere_model_lora_load(executor, model, slot, rank, scale, targets.size(), targets.data(),
                                    adapter.ap.data(), adapter.bp.data());
    };
    CHECK_EQ(load(2, 12, 0.5f, adapter.targets), CERE_ERR_ARGUMENT);  // slot past the pool
    CHECK_EQ(load(0, 0, 0.5f, adapter.targets), CERE_ERR_ARGUMENT);
    CHECK_EQ(load(0, 13, 0.5f, adapter.targets), CERE_ERR_ARGUMENT);  // above the reserved rank
    CHECK_EQ(load(0, 12, std::nanf(""), adapter.targets), CERE_ERR_ARGUMENT);
    CHECK_EQ(load(0, 12, std::numeric_limits<float>::infinity(), adapter.targets), CERE_ERR_ARGUMENT);
    CHECK_EQ(load(0, 12, 0.5f, {}), CERE_ERR_ARGUMENT);
    CHECK_EQ(load(0, 12, 0.5f, {CERE_LORA_SHARED_UP, CERE_LORA_SHARED_UP}), CERE_ERR_ARGUMENT);
    CHECK_EQ(load(0, 12, 0.5f, {CERE_LORA_Q_PROJ}), CERE_ERR_ARGUMENT);  // layer 0 is not full attention
    CHECK_EQ(load(0, 12, 0.5f, {7}), CERE_ERR_ARGUMENT);                 // no module 7
    CHECK_EQ(load(0, 12, 0.5f, {CERE_LORA_MODULES + CERE_LORA_SHARED_UP}), CERE_ERR_ARGUMENT);  // no layer 1
    const void *null_a[3] = {adapter.ap[0], nullptr, adapter.ap[2]};
    CHECK_EQ(cere_model_lora_load(executor, model, 0, 12, 0.5f, 3, adapter.targets.data(), null_a,
                                  adapter.bp.data()),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_lora_load(executor, model, 0, 12, 0.5f, 3, nullptr, adapter.ap.data(), adapter.bp.data()),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_lora_unload(executor, model, 0), CERE_ERR_ARGUMENT);  // empty
    cere_session *s = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &s), CERE_OK);
    CHECK_EQ(cere_session_set_lora(executor, s, 0), CERE_ERR_ARGUMENT);  // empty slot
    CHECK_EQ(adapter.load(executor, model, 0), CERE_OK);
    CHECK_EQ(adapter.load(executor, model, 0), CERE_ERR_ARGUMENT);  // occupied
    CHECK_EQ(load(1, 1, -2.0f, {CERE_LORA_SHARED_DOWN}), CERE_OK);   // any subset, any finite scale
    CHECK_EQ(cere_session_set_lora(nullptr, s, 0), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_session_set_lora(executor, nullptr, 0), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_session_set_lora(executor, s, 2), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_session_set_lora(executor, s, 1), CERE_OK);
    CHECK_EQ(cere_session_set_lora(executor, s, 0), CERE_OK);  // rebinding before the first submit
    CHECK_EQ(model->lora_sessions[0], 1);
    CHECK_EQ(model->lora_sessions[1], 0);
    CHECK_EQ(cere_model_lora_unload(executor, model, 1), CERE_OK);
    CHECK_EQ(cere_model_lora_unload(executor, model, 0), CERE_ERR_BUSY);
    Job job = submit_job(executor, model, s, CERE_OP_PREFILL, {1, 2}, 1);
    CHECK_EQ(drain(executor, job.ticket).committed_tokens, 1);
    CHECK_EQ(cere_session_set_lora(executor, s, 0), CERE_ERR_ARGUMENT);  // after the first submit
    CHECK_EQ(cere_model_release(executor, model), CERE_ERR_BUSY);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(model->lora_sessions[0], 0);
    CHECK_EQ(cere_model_lora_unload(executor, model, 0), CERE_OK);
    CHECK_EQ(cere_model_lora_unload(executor, model, 0), CERE_ERR_ARGUMENT);
    CHECK_EQ(adapter.load(executor, model, 0), CERE_OK);  // hot reload into the freed slot
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(executor->reserved_device_bytes, 0);
    CHECK_EQ(toy(executor).allocated_device_bytes(), 0);

    // A model without a pool: load, unload and set_lora are UNSUPPORTED; a pool past the budget is CAPACITY.
    model = sealed_model(executor, d);
    CHECK_EQ(cere_model_lora_reserve(executor, model, 1, 8), CERE_ERR_ARGUMENT);  // sealed
    CHECK_EQ(adapter.load(executor, model, 0), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_model_lora_unload(executor, model, 0), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_session_create(executor, model, 64, &s), CERE_OK);
    CHECK_EQ(cere_session_set_lora(executor, s, 0), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    executor = make_executor("toy", 48 + cere::host::lora::pool_bytes(cfg, 1, 8) - 1);  // the two tensors fit
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    CHECK_EQ(cere_model_lora_reserve(executor, model, 1, 8), CERE_ERR_CAPACITY);
    CHECK_EQ(cere_model_lora_reserve(executor, model, 1, 1), CERE_ERR_CAPACITY);  // rank 1 pads to 8
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_lora_rows() {
    // Track LoRA: an adapter session and a base session decode in the same rows; the base one is its solo run bit for
    // bit, the adapter one follows the toy's shifted sequence; two loads of the same bytes give the same offset.
    cere_executor *executor = make_executor();
    Desc d = rows_desc();
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    CHECK_EQ(cere_model_lora_reserve(executor, model, 2, 16), CERE_OK);
    for (uint64_t i = 0; i < d.tensors.size(); ++i) {
        auto bytes = pattern(d.tensors[i].bytes, uint8_t(i));
        CHECK_EQ(cere_model_put(executor, model, i, 0, bytes.data(), bytes.size()), CERE_OK);
    }
    CHECK_EQ(cere_model_seal(executor, model), CERE_OK);
    const ToyAdapter adapter(16, 4.0f, 9);
    CHECK_EQ(adapter.load(executor, model, 1), CERE_OK);
    const uint32_t vocab = d.config.vocab_size;
    const uint32_t off = adapter.offset(vocab);
    CHECK(off != 0);
    cere_session *base = prefilled_session(executor, model, {1, 2});
    cere_session *tuned = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &tuned), CERE_OK);
    CHECK_EQ(cere_session_set_lora(executor, tuned, 1), CERE_OK);
    Job tj = submit_job(executor, model, tuned, CERE_OP_PREFILL, {1, 2}, 12);
    Job bj = submit_job(executor, model, base, CERE_OP_DECODE, {}, 12);
    CHECK_EQ(drain(executor, tj.ticket).committed_tokens, 12);
    CHECK_EQ(drain(executor, bj.ticket).committed_tokens, 12);
    check_solo(*bj.work, 12, {1, 2}, 1);
    std::vector<uint32_t> seq{1, 2};
    for (uint64_t i = 0; i < 12; ++i) {
        const uint32_t next = uint32_t((uint64_t(cere::toy::next_token(seq.back(), seq.size(), vocab)) + off) % vocab);
        CHECK_EQ(tj.work->out[i], next);
        seq.push_back(next);
    }
    CHECK(tj.work->out != bj.work->out);
    for (cere_session *s : {base, tuned}) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    // The same bytes reloaded into the other slot score the same; the unloaded slot is reusable.
    CHECK_EQ(cere_model_lora_unload(executor, model, 1), CERE_OK);
    CHECK_EQ(adapter.load(executor, model, 0), CERE_OK);
    CHECK_EQ(cere_session_create(executor, model, 64, &tuned), CERE_OK);
    CHECK_EQ(cere_session_set_lora(executor, tuned, 0), CERE_OK);
    Job again = submit_job(executor, model, tuned, CERE_OP_PREFILL, {1, 2}, 12);
    CHECK_EQ(drain(executor, again.ticket).committed_tokens, 12);
    CHECK(again.work->out == tj.work->out);
    CHECK_EQ(cere_session_release(executor, tuned), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// MixedPD (mixed-pd design note): whether some ticket's prefill call is paused between layer segments.
bool any_parked(cere_executor *executor) {
    for (const auto &entry : executor->tickets) {
        if (entry.second->parked) return true;
    }
    return false;
}

// MixedPD: rows_decode_rounds_run's tickets (rows a, b decoding; prompts p of 14 tokens and q of 4 under chunk 5) on
// a toy whose sliced steps pause `segments` - 1 times, with the prefill slice set before (`before`) or after the
// model's create. Each poll's round is P (prompt consumed), D (a committed) or S (a paused segment: a ticket is
// parked and nothing else moved). Returns the kinds and each ticket's tokens and scores (a, b, p, q).
std::pair<std::string, std::vector<Solo>> prefill_slice_run(uint64_t segments, bool before) {
    const std::string id = "toy,prefill_chunk=5" + (segments > 1 ? ",prefill_segments=" + std::to_string(segments) : "");
    cere_executor *executor = make_executor(id.c_str());
    if (before) CHECK_EQ(cere_executor_set_prefill_slice(executor, 200), CERE_OK);
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    CHECK(model->plan.prefill_park == before);
    if (!before) CHECK_EQ(cere_executor_set_prefill_slice(executor, 200), CERE_OK);
    cere_session *as = prefilled_session(executor, model, {1, 2});  // alone: never sliced
    cere_session *bs = prefilled_session(executor, model, {3, 4});
    Job a = submit_job(executor, model, as, CERE_OP_DECODE, {}, 56);
    Job b = submit_job(executor, model, bs, CERE_OP_DECODE, {}, 56);
    cere_session *ps = nullptr;
    cere_session *qs = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &ps), CERE_OK);
    CHECK_EQ(cere_session_create(executor, model, 64, &qs), CERE_OK);
    const auto p_prompt = iota_prompt(100, 14);
    const auto q_prompt = iota_prompt(200, 4);
    Job p = submit_job(executor, model, ps, CERE_OP_PREFILL, p_prompt, 3);
    Job q = submit_job(executor, model, qs, CERE_OP_PREFILL, q_prompt, 3);
    const auto prompt_fed = [&] {
        return std::min<uint64_t>(ps->fed, p_prompt.size()) + std::min<uint64_t>(qs->fed, q_prompt.size());
    };
    std::string kinds;
    while ((ps->fed < p_prompt.size() || qs->fed < q_prompt.size()) && kinds.size() < 100) {
        const uint64_t fed = prompt_fed();
        const uint64_t committed = committed_of(executor, a.ticket);
        poll_pending(executor, a.ticket);
        if (prompt_fed() != fed) {
            kinds += 'P';
        } else if (committed_of(executor, a.ticket) != committed) {
            kinds += 'D';
        } else {
            CHECK(any_parked(executor));
            kinds += 'S';
        }
    }
    CHECK_EQ(drain(executor, a.ticket).committed_tokens, 56);
    CHECK_EQ(drain(executor, b.ticket).committed_tokens, 56);
    CHECK_EQ(drain(executor, p.ticket).committed_tokens, 3);
    CHECK_EQ(drain(executor, q.ticket).committed_tokens, 3);
    CHECK(!any_parked(executor));
    std::vector<Solo> out;
    for (const Job *j : {&a, &b, &p, &q}) out.push_back(Solo{j->work->out, j->work->scores});
    for (cere_session *s : {as, bs, ps, qs}) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    return {kinds, out};
}

void test_prefill_slice_rounds() {
    // MixedPD: with decode-ready rows waiting, every paused segment is a prefill round followed by the decode round
    // owed, so a call of 3 segments runs as S D S D P; the paused prompt resumes before any other one. BC4 (b): a
    // call that ends unpaused inside the slice passes on to the next prompt of the round-robin pass (p's first chunk,
    // then q's, back to back), then the decode round owed. Every ticket's tokens and scores are the unsliced run's
    // (rows_decode_rounds_run(1)) bit for bit. A model created before the setter has no park and never pauses.
    const std::vector<Solo> base = rows_decode_rounds_run(1);
    const auto [sliced_kinds, sliced] = prefill_slice_run(3, true);
    const std::string call = "SDSDP";
    const std::string want = "D" + call + call + "D" + call + "D" + call;
    CHECK(sliced_kinds == want);
    if (sliced_kinds != want) std::fprintf(stderr, "sliced kinds %s, expected %s\n", sliced_kinds.c_str(), want.c_str());
    const auto [late_kinds, late] = prefill_slice_run(3, false);
    CHECK(late_kinds == "DPPDPDP");
    if (late_kinds != "DPPDPDP") std::fprintf(stderr, "late kinds %s\n", late_kinds.c_str());
    for (const std::vector<Solo> *run : {&sliced, &late}) {
        CHECK_EQ(run->size(), base.size());
        for (size_t i = 0; i < run->size() && i < base.size(); ++i) {
            CHECK((*run)[i].out == base[i].out);
            CHECK((*run)[i].scores.size() == base[i].scores.size() &&
                  std::memcmp((*run)[i].scores.data(), base[i].scores.data(),
                              base[i].scores.size() * sizeof(cere_token_score)) == 0);
        }
    }
}

void test_prefill_slice_alone_and_cancel() {
    // MixedPD: a prompt with no decode-ready ticket waiting is never sliced (one round per chunk). A cancel of a
    // paused ticket waits for its call: its polls stay pending through the remaining segments (the decode rounds
    // between them run), the call consumes its chunk, and the next poll ends it CANCELLED with nothing more fed.
    cere_executor *executor = make_executor("toy,prefill_chunk=5,prefill_segments=3");
    CHECK_EQ(cere_executor_set_prefill_slice(executor, 200), CERE_OK);
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *ls = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &ls), CERE_OK);
    const auto prompt = iota_prompt(100, 14);
    Job lone = submit_job(executor, model, ls, CERE_OP_PREFILL, prompt, 1);
    uint64_t polls = 0;
    CHECK_EQ(drain(executor, lone.ticket, &polls).committed_tokens, 1);
    CHECK_EQ(polls, 3);
    check_solo(*lone.work, 1, prompt, 0);
    CHECK_EQ(cere_session_release(executor, ls), CERE_OK);

    cere_session *ds = prefilled_session(executor, model, {1, 2});
    Job dj = submit_job(executor, model, ds, CERE_OP_DECODE, {}, 20);
    cere_session *ps = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &ps), CERE_OK);
    Job pj = submit_job(executor, model, ps, CERE_OP_PREFILL, prompt, 2);
    poll_pending(executor, dj.ticket);  // decode (owed by ds's prefill)
    CHECK_EQ(committed_of(executor, dj.ticket), 1);
    poll_pending(executor, dj.ticket);  // chunk 1, segment 1: paused
    CHECK(executor->tickets.at(pj.ticket.submission_id)->parked);
    CHECK_EQ(ps->fed, 0);
    CHECK_EQ(cere_executor_cancel(executor, pj.ticket), CERE_OK);
    poll_pending(executor, pj.ticket);  // S1 G3: the rounds dj's polls ran are reported first, no round
    CHECK_EQ(committed_of(executor, dj.ticket), 1);
    CHECK(executor->tickets.at(pj.ticket.submission_id)->parked);
    poll_pending(executor, pj.ticket);  // decode
    CHECK_EQ(committed_of(executor, dj.ticket), 2);
    poll_pending(executor, pj.ticket);  // segment 2: paused again
    CHECK(executor->tickets.at(pj.ticket.submission_id)->parked);
    poll_pending(executor, pj.ticket);  // decode
    CHECK_EQ(committed_of(executor, dj.ticket), 3);
    const cere_completion c = poll_terminal(executor, pj.ticket, CERE_FINISH_CANCELLED);  // the call ends: chunk 1
    CHECK_EQ(c.committed_tokens, 0);
    CHECK_EQ(ps->fed, 5);
    CHECK_EQ(drain(executor, dj.ticket).committed_tokens, 20);
    CHECK_EQ(ps->fed, 5);  // no further chunk ran
    check_solo(*dj.work, 20, {1, 2}, 1);
    CHECK(!any_parked(executor));
    for (cere_session *s : {ds, ps}) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_prefill_slice_setting() {
    // MixedPD: 0 (off) ..= CERE_MAX_PREFILL_SLICE_MS, else ARGUMENT with the setting unchanged; the park is planned
    // for models created while it is set.
    CHECK_EQ(cere_executor_set_prefill_slice(nullptr, 200), CERE_ERR_ARGUMENT);
    cere_executor *executor = make_executor("toy");
    CHECK_EQ(executor->prefill_slice_ms, 0);
    CHECK_EQ(cere_executor_set_prefill_slice(executor, CERE_MAX_PREFILL_SLICE_MS + 1), CERE_ERR_ARGUMENT);
    CHECK_EQ(executor->prefill_slice_ms, 0);
    CHECK_EQ(cere_executor_set_prefill_slice(executor, CERE_MAX_PREFILL_SLICE_MS), CERE_OK);
    CHECK_EQ(executor->prefill_slice_ms, CERE_MAX_PREFILL_SLICE_MS);
    Desc d = rows_desc();
    cere_model *parked = sealed_model(executor, d);
    CHECK(parked->plan.prefill_park);
    CHECK_EQ(cere_executor_set_prefill_slice(executor, 0), CERE_OK);
    Desc e = rows_desc();
    cere_model *plain = sealed_model(executor, e);
    CHECK(!plain->plan.prefill_park);
    CHECK(parked->plan.prefill_park);
    CHECK_EQ(cere_model_release(executor, plain), CERE_OK);
    CHECK_EQ(cere_model_release(executor, parked), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_prefill_age_setting() {
    // BC4 (b)'s aging cap: CERE_DEFAULT_PREFILL_AGE_CALLS at create, 1 ..= CERE_MAX_PREFILL_AGE_CALLS, else ARGUMENT
    // with the setting unchanged.
    CHECK_EQ(cere_executor_set_prefill_age(nullptr, 8), CERE_ERR_ARGUMENT);
    cere_executor *executor = make_executor("toy");
    CHECK_EQ(executor->prefill_age_calls, CERE_DEFAULT_PREFILL_AGE_CALLS);
    CHECK_EQ(CERE_DEFAULT_PREFILL_AGE_CALLS, 8u);
    for (uint32_t bad : {0u, CERE_MAX_PREFILL_AGE_CALLS + 1}) {
        CHECK_EQ(cere_executor_set_prefill_age(executor, bad), CERE_ERR_ARGUMENT);
        CHECK_EQ(executor->prefill_age_calls, CERE_DEFAULT_PREFILL_AGE_CALLS);
    }
    for (uint32_t ok : {1u, 3u, CERE_MAX_PREFILL_AGE_CALLS}) {
        CHECK_EQ(cere_executor_set_prefill_age(executor, ok), CERE_OK);
        CHECK_EQ(executor->prefill_age_calls, ok);
    }
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// BC4 (b) (batch-concurrency design note, bar Sched): a 40-token prompt L submitted first, then twelve 4-token
// prompts a..l, under chunk 4 and max_new 1 (no decode round), with the aging cap `age` (0: the default). Returns the
// prompt of each prefill call in order ('L' or the short prompt's letter), and checks every reply equals its solo
// run's.
std::string prefill_age_run(uint32_t age) {
    cere_executor *executor = make_executor("toy,prefill_chunk=4");
    if (age != 0) CHECK_EQ(cere_executor_set_prefill_age(executor, age), CERE_OK);
    Desc d = rows_desc(13);
    cere_model *model = sealed_model(executor, d);
    std::vector<cere_session *> sessions(13, nullptr);
    std::vector<std::vector<uint32_t>> prompts{iota_prompt(1, 40)};
    for (uint32_t i = 0; i < 12; ++i) prompts.push_back(iota_prompt(41 + 4 * i, 4));
    std::vector<Job> jobs;
    for (size_t i = 0; i < sessions.size(); ++i) {
        CHECK_EQ(cere_session_create(executor, model, 64, &sessions[i]), CERE_OK);
        jobs.push_back(submit_job(executor, model, sessions[i], CERE_OP_PREFILL, prompts[i], 1));
    }
    std::string order;
    for (int poll = 0; poll < 100; ++poll) {
        std::vector<uint64_t> fed;
        for (cere_session *s : sessions) fed.push_back(s->fed);
        cere_completion c = completion_header();
        const cere_status st = cere_executor_poll(executor, jobs[0].ticket, &c);
        for (size_t i = 0; i < sessions.size(); ++i) {
            if (sessions[i]->fed != fed[i]) order += i == 0 ? 'L' : char('a' + i - 1);
        }
        if (st == CERE_OK) break;
        CHECK_EQ(st, CERE_PENDING);
    }
    for (size_t i = 1; i < jobs.size(); ++i) CHECK_EQ(drain(executor, jobs[i].ticket).committed_tokens, 1);
    for (size_t i = 0; i < jobs.size(); ++i) check_solo(*jobs[i].work, 1, prompts[i], 0);
    for (cere_session *s : sessions) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    return order;
}

void test_prefill_age_cap() {
    // Equal prompts keep their submission order (a..l), the shorter remaining prompt goes first, and L, overtaken by
    // `age` younger tickets' calls, runs the next call: it is never overtaken more than `age` times in a row. Every
    // reply equals its solo run's in each order.
    const std::string capped = prefill_age_run(3);
    CHECK(capped == "abcLdefLghiLjklLLLLLLL");
    if (capped != "abcLdefLghiLjklLLLLLLL") std::fprintf(stderr, "age 3 order %s\n", capped.c_str());
    const std::string by_default = prefill_age_run(0);  // CERE_DEFAULT_PREFILL_AGE_CALLS = 8
    CHECK(by_default == "abcdefghLijklLLLLLLLLL");
    if (by_default != "abcdefghLijklLLLLLLLLL") std::fprintf(stderr, "age 8 order %s\n", by_default.c_str());
    const std::string uncapped = prefill_age_run(CERE_MAX_PREFILL_AGE_CALLS);
    CHECK(uncapped == "abcdefghijklLLLLLLLLLL");
    if (uncapped != "abcdefghijklLLLLLLLLLL") std::fprintf(stderr, "age 1024 order %s\n", uncapped.c_str());
}

void test_poll_reports_round_first() {
    // S1 G3 (beat-sglang design note): two prompts of 2 chunks polled in turn (a driver pass: q, then p). A round
    // that q's poll ran is reported at p's next poll without another round, so p's first token is reported before
    // q's first chunk runs; a ticket polled twice in a row runs a round at the second poll. BC4 (b): p, the oldest of
    // equal prompts, then the one with fewer tokens left, runs both its chunks; a run of back-to-back prefill rounds
    // has as many calls as the round has prompts (2: p's chunk 2, then q's chunk 1), then the decode round owed.
    cere_executor *executor = make_executor("toy,prefill_chunk=4");
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *ps = nullptr;
    cere_session *qs = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &ps), CERE_OK);
    CHECK_EQ(cere_session_create(executor, model, 64, &qs), CERE_OK);
    const auto p_prompt = iota_prompt(100, 8);
    const auto q_prompt = iota_prompt(200, 8);
    Job p = submit_job(executor, model, ps, CERE_OP_PREFILL, p_prompt, 2);
    Job q = submit_job(executor, model, qs, CERE_OP_PREFILL, q_prompt, 2);
    const auto pending = [&](const Job &j) {
        cere_completion c = completion_header();
        CHECK_EQ(cere_executor_poll(executor, j.ticket, &c), CERE_PENDING);
        return c.committed_tokens;
    };
    CHECK_EQ(pending(q), 0);  // p's chunk 1
    CHECK_EQ(ps->fed, 4);
    CHECK_EQ(pending(p), 0);  // reported, no round
    CHECK_EQ(ps->fed, 4);
    CHECK_EQ(pending(q), 0);  // p's chunk 2 (fewest tokens left): p's first token
    CHECK_EQ(ps->fed, 8);
    CHECK_EQ(qs->fed, 0);
    CHECK_EQ(committed_of(executor, p.ticket), 1);
    CHECK_EQ(pending(p), 1);  // reported at once: q's first chunk has not run
    CHECK_EQ(qs->fed, 0);
    // p polled again runs a round: q's chunk 1, back to back with p's chunk 2 (a run of 2 calls, 2 prompts).
    CHECK_EQ(pending(p), 1);
    CHECK_EQ(qs->fed, 4);
    // Then the decode round owed after the run, where p reaches its length.
    CHECK_EQ(poll_terminal(executor, p.ticket, CERE_FINISH_LENGTH).committed_tokens, 2);
    CHECK_EQ(qs->fed, 4);
    CHECK_EQ(drain(executor, q.ticket).committed_tokens, 2);
    check_solo(*p.work, 2, p_prompt, 0);
    check_solo(*q.work, 2, q_prompt, 0);
    for (cere_session *s : {ps, qs}) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// S1 G4: rows_decode_rounds_run's tickets (rows a, b decoding; prompts p and q of 14 tokens each under chunk 5) with
// the decode share `share_pct` set after the rows' own prefills, on a toy whose sliced steps pause `segments` - 1
// times (the prefill slice set before the model's create when `segments` > 1). Each poll's round is P (prompt
// consumed), D (a decode row committed) or S (a paused segment). Returns the kinds and each ticket's tokens and
// scores (a, b, p, q).
std::pair<std::string, std::vector<Solo>> decode_share_run(uint32_t share_pct, uint64_t segments) {
    const std::string id = "toy,prefill_chunk=5" + (segments > 1 ? ",prefill_segments=" + std::to_string(segments) : "");
    cere_executor *executor = make_executor(id.c_str());
    if (segments > 1) CHECK_EQ(cere_executor_set_prefill_slice(executor, 200), CERE_OK);
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *as = prefilled_session(executor, model, {1, 2});
    cere_session *bs = prefilled_session(executor, model, {3, 4});
    CHECK_EQ(cere_executor_set_decode_share(executor, share_pct), CERE_OK);
    Job a = submit_job(executor, model, as, CERE_OP_DECODE, {}, 56);
    Job b = submit_job(executor, model, bs, CERE_OP_DECODE, {}, 56);
    cere_session *ps = nullptr;
    cere_session *qs = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &ps), CERE_OK);
    CHECK_EQ(cere_session_create(executor, model, 64, &qs), CERE_OK);
    const auto p_prompt = iota_prompt(100, 14);
    const auto q_prompt = iota_prompt(200, 14);
    Job p = submit_job(executor, model, ps, CERE_OP_PREFILL, p_prompt, 3);
    Job q = submit_job(executor, model, qs, CERE_OP_PREFILL, q_prompt, 3);
    const auto prompt_fed = [&] {
        return std::min<uint64_t>(ps->fed, p_prompt.size()) + std::min<uint64_t>(qs->fed, q_prompt.size());
    };
    const auto decoded = [&] {
        return committed_of(executor, a.ticket) + committed_of(executor, b.ticket) + committed_of(executor, p.ticket);
    };
    std::string kinds;
    while (qs->fed < q_prompt.size() && kinds.size() < 100) {
        const uint64_t fed = prompt_fed();
        const uint64_t done = decoded();
        poll_pending(executor, a.ticket);
        if (prompt_fed() != fed) {
            kinds += 'P';
        } else if (decoded() != done) {
            kinds += 'D';
        } else {
            CHECK(any_parked(executor));
            kinds += 'S';
        }
    }
    CHECK_EQ(drain(executor, a.ticket).committed_tokens, 56);
    CHECK_EQ(drain(executor, b.ticket).committed_tokens, 56);
    CHECK_EQ(drain(executor, p.ticket).committed_tokens, 3);
    CHECK_EQ(drain(executor, q.ticket).committed_tokens, 3);
    CHECK(!any_parked(executor));
    check_solo(*a.work, 56, {1, 2}, 1);
    check_solo(*b.work, 56, {3, 4}, 1);
    check_solo(*p.work, 3, p_prompt, 0);
    check_solo(*q.work, 3, q_prompt, 0);
    std::vector<Solo> out;
    for (const Job *j : {&a, &b, &p, &q}) out.push_back(Solo{j->work->out, j->work->scores});
    for (cere_session *s : {as, bs, ps, qs}) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    return {kinds, out};
}

void test_decode_share_rounds() {
    // S1 G4: while p and q both wait, p's 3 chunks run back to back (no decode round, no paused segment); q, the last
    // prompt, starts at once and owes the decode rows share x its rounds' wall time, so one or more decode rounds
    // run before each of its later chunks (and, sliced, after each paused segment). Every ticket's tokens and scores
    // are the same in each run (only the order of rounds changes). Share 0 keeps the decode rounds per chunk (1) after
    // each run of back-to-back prefill rounds (BC4 (b): the prompt with fewer tokens left first, p then q, a run as
    // long as the round has prompts: p p, then p q, then q alone per run).
    const auto [whole_kinds, whole] = decode_share_run(10, 1);
    const bool whole_ok = std::regex_match(whole_kinds, std::regex("PPPP(D+P){2}"));
    CHECK(whole_ok);
    if (!whole_ok) std::fprintf(stderr, "decode share kinds %s\n", whole_kinds.c_str());
    const auto [sliced_kinds, sliced] = decode_share_run(10, 3);
    const bool sliced_ok = std::regex_match(sliced_kinds, std::regex("PPPSD+SD+P(D+SD+SD+P){2}"));
    CHECK(sliced_ok);
    if (!sliced_ok) std::fprintf(stderr, "decode share sliced kinds %s\n", sliced_kinds.c_str());
    const auto [off_kinds, off] = decode_share_run(0, 1);
    CHECK(off_kinds == "DPPDPPDPDP");
    if (off_kinds != "DPPDPPDPDP") std::fprintf(stderr, "decode share 0 kinds %s\n", off_kinds.c_str());
    for (const std::vector<Solo> *run : {&whole, &sliced}) {
        CHECK_EQ(run->size(), off.size());
        for (size_t i = 0; i < run->size() && i < off.size(); ++i) {
            CHECK((*run)[i].out == off[i].out);
            CHECK((*run)[i].scores.size() == off[i].scores.size() &&
                  std::memcmp((*run)[i].scores.data(), off[i].scores.data(),
                              off[i].scores.size() * sizeof(cere_token_score)) == 0);
        }
    }
}

void test_decode_share_setting() {
    // S1 G4: 0 (off) ..= CERE_MAX_DECODE_SHARE_PCT, else ARGUMENT with the setting unchanged.
    CHECK_EQ(cere_executor_set_decode_share(nullptr, 100), CERE_ERR_ARGUMENT);
    cere_executor *executor = make_executor("toy");
    CHECK_EQ(executor->decode_share_pct, 0);
    CHECK_EQ(cere_executor_set_decode_share(executor, CERE_MAX_DECODE_SHARE_PCT + 1), CERE_ERR_ARGUMENT);
    CHECK_EQ(executor->decode_share_pct, 0);
    CHECK_EQ(cere_executor_set_decode_share(executor, CERE_MAX_DECODE_SHARE_PCT), CERE_OK);
    CHECK_EQ(executor->decode_share_pct, CERE_MAX_DECODE_SHARE_PCT);
    CHECK_EQ(cere_executor_set_decode_share(executor, 0), CERE_OK);
    CHECK_EQ(executor->decode_share_pct, 0);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

cere_execution_capabilities capabilities_of(cere_executor *executor, cere_model *model) {
    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(cere_executor_capabilities(executor, model, &caps), CERE_OK);
    return caps;
}

std::vector<uint32_t> range_ids(uint32_t first, uint32_t end) {
    std::vector<uint32_t> ids;
    for (uint32_t i = first; i < end; ++i) ids.push_back(i);
    return ids;
}

// m25 design note C1/C2 (D3, D4; ledger D 72, C 72-80): cere_model_desc.kv_pool_tokens 0 is the derived pool
// (max_sessions x round64(ctx)); non-zero must be a multiple of 64 and >= max_context_tokens (ARGUMENT) and <= the
// support table's max_kv_pool_tokens (UNSUPPORTED); capabilities report the pool and the 64-token page. Sessions
// reserve ceil(ctx / 64) pages at create (CAPACITY with nothing kept on a shortfall), steps never see the pool, and
// release returns the pages, which are reused lowest id first.
void test_kv_pool() {
    const cere_qwen4_exp_config &cfg = cere::toy::fixture_config();
    cere_executor *executor = make_executor();
    cere_model *model = nullptr;
    {
        Desc d(cfg, {16}, {}, 2, 128);
        for (const uint64_t pool : {uint64_t(65), uint64_t(193), uint64_t(64)}) {  // not x64 twice, then < ctx
            d.desc.kv_pool_tokens = pool;
            CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_ERR_ARGUMENT);
            CHECK(model == nullptr);
        }
        d.desc.kv_pool_tokens = cere::toy::fixture_support().max_kv_pool_tokens + 64;
        CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_ERR_UNSUPPORTED);
        CHECK(model == nullptr);
        CHECK_EQ(toy(executor).allocations(), 0);
    }
    {
        // 0: derived pool, 3 x round64(100) = 384; exactly three full-context sessions fit.
        Desc d(cfg, {16}, {}, 3, 100);
        model = sealed_model(executor, d);
        const cere_execution_capabilities caps = capabilities_of(executor, model);
        CHECK_EQ(caps.kv_pool_tokens, 384);
        CHECK_EQ(caps.kv_page_tokens, 64);
        CHECK_EQ(caps.reserved2, 0);
        cere_session *sessions[3] = {};
        for (cere_session *&s : sessions) CHECK_EQ(cere_session_create(executor, model, 100, &s), CERE_OK);
        for (cere_session *s : sessions) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    }
    {
        // A non-zero pool at the support bound (1,048,576, CX-01) is accepted and reported as allocated.
        Desc d(cfg, {16}, {}, 1, 200000);
        d.desc.kv_pool_tokens = cere::toy::fixture_support().max_kv_pool_tokens;
        model = sealed_model(executor, d);
        CHECK_EQ(capabilities_of(executor, model).kv_pool_tokens, 1048576);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    }
    // D9 (3) at toy scale: max_sessions 2, ctx 9,088 (142 pages), pool 12,288 (192 pages).
    Desc d(cfg, {16}, {}, 2, 9088);
    d.desc.kv_pool_tokens = 12288;
    model = sealed_model(executor, d);
    CHECK_EQ(capabilities_of(executor, model).kv_pool_tokens, 12288);
    const uint64_t generation = generation_of(executor, model);
    cere::toy::ToyDevice &device = toy(executor);
    const cere::host::DeviceModel &pool = *model->device;
    CHECK_EQ(device.free_pages(pool), 192);
    cere_session *first = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 9088, &first), CERE_OK);
    CHECK(device.session_pages(*first->device) == range_ids(0, 142));
    cere_session *second = reinterpret_cast<cere_session *>(0x1);
    CHECK_EQ(cere_session_create(executor, model, 9088, &second), CERE_ERR_CAPACITY);  // 50 pages free
    CHECK(second == nullptr);
    CHECK_EQ(model->sessions, 1);
    CHECK_EQ(device.free_pages(pool), 50);
    CHECK_EQ(cere_session_create(executor, model, 9089, &second), CERE_ERR_CAPACITY);  // > plan ctx, unchanged
    CHECK_EQ(cere_session_create(executor, model, 3200, &second), CERE_OK);           // exactly the 50 left
    CHECK(device.session_pages(*second->device) == range_ids(142, 192));
    CHECK_EQ(device.free_pages(pool), 0);
    {
        // The pool is full, yet a step never needs a page: prefill + decode run to their length.
        Work prefill(second, CERE_OP_PREFILL, std::vector<uint32_t>(3000, 7), 8, generation);
        cere_completion_ticket ticket{};
        CHECK_EQ(cere_executor_submit(executor, &prefill.batch, &ticket), CERE_OK);
        const cere_completion c = drain(executor, ticket);
        CHECK_EQ(c.status, CERE_OK);
        CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
        CHECK_EQ(c.committed_tokens, 8);
    }
    // Release returns the pages; the retry that failed now succeeds on the lowest ids again.
    CHECK_EQ(cere_session_release(executor, first), CERE_OK);
    CHECK_EQ(device.free_pages(pool), 142);
    cere_session *retry = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 9088, &retry), CERE_OK);
    CHECK(device.session_pages(*retry->device) == range_ids(0, 142));
    // A hole fills lowest first: release the first 142 pages again, a 64-token session takes page 0.
    CHECK_EQ(cere_session_release(executor, retry), CERE_OK);
    cere_session *small = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &small), CERE_OK);
    CHECK(device.session_pages(*small->device) == range_ids(0, 1));
    CHECK_EQ(cere_session_release(executor, small), CERE_OK);
    CHECK_EQ(cere_session_release(executor, second), CERE_OK);
    CHECK_EQ(device.free_pages(pool), 192);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// m25 design note C8, ledger CR1 (M2.5 x M2.6): the rows share the one pool. A 5-session plan at ctx 128 whose pool
// holds four contexts (512 tokens, 8 pages): four sessions reserve their pages at create, the fifth create is CAPACITY
// with nothing kept (a session slot is still free: the pool refuses, not the rows); the four then decode as rows to
// their full context with the pool empty, each ticket ending LENGTH with its solo tokens (no step or step_rows
// CAPACITY); a release lets the fifth create in on the lowest freed pages.
void test_kv_pool_rows() {
    cere_executor *executor = make_executor();
    Desc d(cere::toy::fixture_config(), {16}, {}, 5, 128);
    d.desc.kv_pool_tokens = 512;
    cere_model *model = sealed_model(executor, d);
    CHECK_EQ(capabilities_of(executor, model).kv_pool_tokens, 512);
    cere::toy::ToyDevice &device = toy(executor);
    const cere::host::DeviceModel &pool = *model->device;
    std::vector<cere_session *> sessions(4, nullptr);
    for (cere_session *&s : sessions) CHECK_EQ(cere_session_create(executor, model, 128, &s), CERE_OK);
    CHECK_EQ(device.free_pages(pool), 0);
    cere_session *fifth = reinterpret_cast<cere_session *>(0x1);
    CHECK_EQ(cere_session_create(executor, model, 128, &fifth), CERE_ERR_CAPACITY);
    CHECK(fifth == nullptr);
    CHECK_EQ(model->sessions, 4);
    CHECK_EQ(device.free_pages(pool), 0);
    // Each row: a 2-token prompt and 126 new tokens fill its 128-token context.
    const uint32_t vocab = d.config.vocab_size;
    std::vector<std::vector<uint32_t>> prompts;
    std::vector<Job> jobs;
    for (uint32_t i = 0; i < sessions.size(); ++i) {
        prompts.push_back({2 * i + 1, 2 * i + 2});
        jobs.push_back(submit_job(executor, model, sessions[i], CERE_OP_PREFILL, prompts.back(), 126));
    }
    for (size_t i = 0; i < jobs.size(); ++i) {
        const cere_completion c = drain(executor, jobs[i].ticket);
        CHECK_EQ(c.status, CERE_OK);
        CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
        CHECK_EQ(c.committed_tokens, 126);
        const auto want = toy_sequence(prompts[i], 126, vocab);
        for (size_t k = 0; k < want.size(); ++k) CHECK_EQ(jobs[i].work->out[k], want[k]);
    }
    CHECK_EQ(device.free_pages(pool), 0);
    CHECK_EQ(cere_session_release(executor, sessions[1]), CERE_OK);
    CHECK_EQ(device.free_pages(pool), 2);
    CHECK_EQ(cere_session_create(executor, model, 128, &fifth), CERE_OK);
    CHECK(device.session_pages(*fifth->device) == range_ids(2, 4));
    sessions[1] = fifth;
    for (cere_session *s : sessions) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(device.free_pages(pool), 8);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// Ledger L0 (native-abi-ledger design note): one struct_size per shared struct (the pre-L0 sizes are ARGUMENT),
// owner-less input fields must be zero/null (else UNSUPPORTED), flags bits per table F, owner-less outputs read 0.
// kv_pool_tokens has its owner (m25 design note C1): test_kv_pool covers it, and the frame's outputs report it.
void test_l0_frame() {
    cere_executor *executor = make_executor();
    Desc d = toy_desc();
    cere_model *model = nullptr;
    d.desc.struct_size = 72;  // pre-L0 cere_model_desc
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_ERR_ARGUMENT);
    d.desc.struct_size = sizeof(d.desc);
    static const uint32_t dummy[4] = {};
    const auto set_desc = [&](int field) {
        d.desc.speculative_draft_tokens = field == 0 ? 4 : 0;
        d.desc.kv_pool_tokens = field == 1 ? 64 : 0;
        d.desc.draft_token_map = field == 2 ? dummy : nullptr;
        d.desc.draft_token_map_count = field == 3 ? 1 : 0;
        d.desc.snapshot_host_bytes = field == 4 ? 1 : 0;
        d.desc.prefix_state_slots = field == 5 ? 1 : 0;
    };
    // kv_pool_tokens (1) is M2.5's (test_kv_pool), snapshot_host_bytes (4) M2.11's (test_snapshot_api),
    // prefix_state_slots (5) M2.10's (test_prefix_*), vision_config (D 112) M2.13's (test_images), the draft fields
    // (0, 2, 3) M2.9's (test_spec_abi: a lone one is ARGUMENT).
    for (const int field : {0, 2, 3}) {
        set_desc(field);
        CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_ERR_ARGUMENT);
        CHECK(model == nullptr);
    }
    set_desc(-1);
    model = sealed_model(executor, d);

    cere_execution_capabilities caps{};
    caps.struct_size = 72;  // pre-L0 size
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(cere_executor_capabilities(executor, model, &caps), CERE_ERR_ARGUMENT);
    std::memset(&caps, 0xff, sizeof(caps));
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(cere_executor_capabilities(executor, model, &caps), CERE_OK);
    CHECK_EQ(caps.native_sampler, 1);     // M2.7: the toy steps through host/sampler_ref
    CHECK_EQ(caps.native_token_mask, 1);  // M2.8 A3 (ledger C 58): vocab masks with the sampler
    CHECK_EQ(caps.speculative_draft_tokens + caps.native_prefix_cache + caps.native_image_input + caps.reserved[0] +
                 caps.reserved[1],
             0);
    CHECK_EQ(caps.kv_pool_tokens, 128);  // M2.5 D4: 0 = the derived pool, 2 sessions x 64 tokens
    CHECK_EQ(caps.kv_page_tokens, 64);
    CHECK_EQ(caps.reserved2, 0);
    const uint64_t generation = caps.profile_generation;

    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &session), CERE_OK);
    cere_completion_ticket ticket{};
    {
        Work w(session, CERE_OP_PREFILL, {1, 2}, 2, generation);
        w.item.struct_size = 112;  // pre-L0 cere_work
        CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_ERR_ARGUMENT);
        w.item.struct_size = sizeof(w.item);
        w.item.flags = 0x100;  // outside CERE_WORK_KNOWN_FLAGS
        CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_ERR_ARGUMENT);
        w.item.flags = UINT32_C(0x80000000);
        CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_ERR_ARGUMENT);
        w.item.flags = 0;
        // sampling/out_logprobs/top_logprobs have their owner (M2.7, test_sampling_validation).
        const auto set_work = [&](int field) {
            w.item.capture_position = field == 3 ? 64 : 0;
            w.item.vocab_mask = field == 4 ? dummy : nullptr;
            w.item.vocab_mask_words = field == 5 ? 1 : 0;
        };
        // images/image_count (W 152-160) have their owner (M2.13): test_images.
        for (int field = 3; field < 6; ++field) {
            set_work(field);
            // vocab_mask/vocab_mask_words have their owner (M2.8 A1, test_vocab_mask_submit): one without the other
            // is ARGUMENT; capture_position has its owner (M2.10 W 132, test_prefix_capture_position): 64 is not
            // inside this 2-token prompt, ARGUMENT.
            CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_ERR_ARGUMENT);
            CHECK_EQ(ticket.submission_id, 0);
        }
        set_work(-1);
        w.item.flags = CERE_WORK_NO_SPECULATION;  // accepted as a no-op from L0 on
        CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);

        cere_completion c = completion_header();
        c.struct_size = 48;  // pre-L0 cere_completion
        CHECK_EQ(cere_executor_poll(executor, ticket, &c), CERE_ERR_ARGUMENT);
        // One poll into a 0xff-filled completion with a valid header: the owner-less outputs read 0.
        std::memset(&c, 0xff, sizeof(c));
        c.struct_size = sizeof(c);
        c.abi_version = CERE_EXECUTOR_ABI_VERSION;
        const cere_status first = cere_executor_poll(executor, ticket, &c);
        CHECK(first == CERE_PENDING || first == CERE_OK);
        CHECK_EQ(c.cached_tokens, 0);  // M2.10 P 44: no prefix cache on this model (prefix_state_slots 0)
        CHECK_EQ(c.verify_steps, 0);
        CHECK_EQ(c.accepted_drafts, 0);
        if (first == CERE_PENDING) c = drain(executor, ticket);
        CHECK_EQ(c.committed_tokens, 2);
        CHECK_EQ(c.cached_tokens, 0);
        CHECK_EQ(c.verify_steps, 0);
        CHECK_EQ(c.accepted_drafts, 0);
        const auto expected = toy_sequence({1, 2}, 2, cere::toy::fixture_config().vocab_size);
        CHECK_EQ(w.out[0], expected[0]);
        CHECK_EQ(w.out[1], expected[1]);
    }
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// M2.13 (m213 design note C7): vision_config on the desc, images on PREFILL work (ledger N, D 112, W 152-160,
// C 61, F bit 1). The toy device and the CUDA device accept the A4 vision entry (a4_vision_support).
constexpr uint32_t kVisionStart = 248053;
constexpr uint32_t kImagePad = 248056;
constexpr uint32_t kVisionEnd = 248054;

struct VisionDesc : Desc {
    cere_qwen4_exp_vision_config vision;
    VisionDesc(uint64_t max_context = 2048, std::vector<uint64_t> sizes = {16}, uint64_t max_sessions = 1)
        : Desc(a4_support().config, sizes, {}, max_sessions, max_context),
          vision(*cere::host::a4_vision_support().vision) {
        desc.vision_config = &vision;
    }
};

// One image of grid (1, h, w) at `offset` (its first pad row) with a `fill`-filled patch buffer and its SHA-256 as
// pixel_sha256 (M2.13-L C1). A copy points pixel_patches at its own buffer (unless a test re-pointed it).
struct Image {
    std::vector<uint8_t> pixels;
    cere_image_input input{};
    Image(uint64_t offset, uint32_t h, uint32_t w, uint32_t pad_id = 1000001, uint8_t fill = 0x5a)
        : pixels(uint64_t(h) * w * 1536 * 2, fill) {
        input.struct_size = sizeof(input);
        input.abi_version = CERE_EXECUTOR_ABI_VERSION;
        input.grid_t = 1;
        input.grid_h = h;
        input.grid_w = w;
        input.pad_id = pad_id;
        input.token_offset = offset;
        input.pixel_patches = pixels.data();
        input.pixel_bytes = pixels.size();
        const std::array<uint8_t, 32> d = cere::host::sha256(pixels.data(), pixels.size());
        std::copy(d.begin(), d.end(), input.pixel_sha256);
    }
    Image(const Image &o) : pixels(o.pixels), input(o.input) { repoint(o); }
    Image &operator=(const Image &o) {
        pixels = o.pixels;
        input = o.input;
        repoint(o);
        return *this;
    }
    Image(Image &&) noexcept = default;  // moves keep the buffer
    Image &operator=(Image &&) noexcept = default;

private:
    void repoint(const Image &o) {
        if (o.input.pixel_patches == o.pixels.data()) input.pixel_patches = pixels.data();
    }
};

// A prompt with one image span per `pads` entry: text, vision_start, pads x image_pad, vision_end, ...
std::vector<uint32_t> image_prompt(const std::vector<uint64_t> &pads, std::vector<uint64_t> *offsets) {
    std::vector<uint32_t> p{1};
    for (uint64_t n : pads) {
        p.push_back(kVisionStart);
        offsets->push_back(p.size());
        p.insert(p.end(), n, kImagePad);
        p.push_back(kVisionEnd);
        p.push_back(2);
    }
    return p;
}

void test_images() {
    cere_executor *executor = make_executor();
    cere_model *model = nullptr;
    // Desc rules: header first, then the compiled entry bit for bit; visual tensors need vision_config.
    {
        VisionDesc v;
        v.vision.struct_size = 72;
        CHECK_EQ(cere_model_create(executor, &v.desc, &model), CERE_ERR_ARGUMENT);
        v.vision.struct_size = sizeof(v.vision);
        v.vision.abi_version = 2;
        CHECK_EQ(cere_model_create(executor, &v.desc, &model), CERE_ERR_UNSUPPORTED);
        v.vision.abi_version = CERE_EXECUTOR_ABI_VERSION;
        v.vision.depth += 1;
        CHECK_EQ(cere_model_create(executor, &v.desc, &model), CERE_ERR_UNSUPPORTED);
        v.vision.depth -= 1;
        v.vision.layer_norm_eps = 1e-5;
        CHECK_EQ(cere_model_create(executor, &v.desc, &model), CERE_ERR_UNSUPPORTED);
        Desc fixture = toy_desc();
        fixture.desc.vision_config = &v.vision;  // the fixture model has no vision entry
        CHECK_EQ(cere_model_create(executor, &fixture.desc, &model), CERE_ERR_UNSUPPORTED);
        Desc visual(a4_support().config, {16, 16}, {}, 1, 2048);
        visual.names[1] = "model.visual.patch_embed.proj.weight";
        visual.tensors[1].name = visual.names[1].c_str();
        CHECK_EQ(cere_model_create(executor, &visual.desc, &model), CERE_ERR_UNSUPPORTED);
        CHECK(model == nullptr);
    }
    // C 61: native_image_input is 1 iff the model has vision_config.
    const auto image_input = [&](cere_model *m) {
        cere_execution_capabilities caps{};
        caps.struct_size = sizeof(caps);
        caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
        CHECK_EQ(cere_executor_capabilities(executor, m, &caps), CERE_OK);
        return int(caps.native_image_input);
    };
    Desc text(a4_support().config, {16}, {}, 1, 2048);
    cere_model *text_model = sealed_model(executor, text);
    CHECK_EQ(image_input(text_model), 0);
    VisionDesc v(4096, {16, 16});
    v.names[1] = "model.visual.patch_embed.proj.weight";
    v.tensors[1].name = v.names[1].c_str();
    model = sealed_model(executor, v);
    CHECK_EQ(image_input(model), 1);
    const uint64_t generation = generation_of(executor, model);

    std::vector<uint64_t> offsets;
    const std::vector<uint32_t> prompt = image_prompt({4, 4}, &offsets);  // two grids (1, 4, 4)
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 2048, &session), CERE_OK);
    cere_completion_ticket ticket{};
    const auto submit = [&](Work &w, std::vector<Image> &images) {
        std::vector<cere_image_input> inputs;
        for (const Image &i : images) inputs.push_back(i.input);
        w.item.images = inputs.empty() ? nullptr : inputs.data();
        w.item.image_count = inputs.size();
        const cere_status st = cere_executor_submit(executor, &w.batch, &ticket);
        if (st != CERE_OK) CHECK_EQ(ticket.submission_id, 0);
        return st;
    };
    const auto images = [&]() { return std::vector<Image>{Image(offsets[0], 4, 4), Image(offsets[1], 4, 4)}; };
    {
        Work w(session, CERE_OP_PREFILL, prompt, 2, generation);
        w.item.flags = CERE_WORK_NO_SPECULATION;
        std::vector<Image> im = images();
        // Each refusal leaves the session untouched (the valid submit below still succeeds).
        const auto refused = [&](const std::function<void(std::vector<Image> &, Work &)> &edit, cere_status want) {
            std::vector<Image> bad = images();
            Work wb(session, CERE_OP_PREFILL, prompt, 2, generation);
            wb.item.flags = CERE_WORK_NO_SPECULATION;
            edit(bad, wb);
            CHECK_EQ(submit(wb, bad), want);
        };
        refused([](std::vector<Image> &b, Work &) {  // M2.13-L C1: an all-zero pixel_sha256
            std::fill(std::begin(b[0].input.pixel_sha256), std::end(b[0].input.pixel_sha256), uint8_t(0));
        }, CERE_ERR_ARGUMENT);
        refused([](std::vector<Image> &b, Work &) {  // C2: 4,160 merged rows > 4,096 per image
            b[0].input.grid_h = 128;
            b[0].input.grid_w = 130;
            b[0].input.pixel_bytes = uint64_t(128) * 130 * 1536 * 2;
        }, CERE_ERR_ARGUMENT);
        refused([](std::vector<Image> &b, Work &) { b[0].input.struct_size = 40; }, CERE_ERR_ARGUMENT);
        refused([](std::vector<Image> &b, Work &) { b[0].input.abi_version = 2; }, CERE_ERR_UNSUPPORTED);
        refused([](std::vector<Image> &b, Work &) { b[1].input.pad_id = 0; }, CERE_ERR_ARGUMENT);
        refused([](std::vector<Image> &b, Work &) { b[0].input.grid_w = 8; }, CERE_ERR_ARGUMENT);  // 32 pads, 16 rows
        refused([](std::vector<Image> &b, Work &) { b[1].input.grid_h = 1; }, CERE_ERR_ARGUMENT);  // odd grid
        refused([](std::vector<Image> &b, Work &) { b[1].input.grid_t = 0; }, CERE_ERR_ARGUMENT);
        refused([](std::vector<Image> &b, Work &) { b[0].input.pixel_bytes -= 2; }, CERE_ERR_ARGUMENT);
        refused([](std::vector<Image> &b, Work &) { b[0].input.pixel_patches = nullptr; }, CERE_ERR_ARGUMENT);
        refused([](std::vector<Image> &b, Work &) { b[0].input.token_offset += 1; }, CERE_ERR_ARGUMENT);  // no start
        refused([&](std::vector<Image> &b, Work &) { b[1].input.token_offset = prompt.size(); }, CERE_ERR_ARGUMENT);
        refused([](std::vector<Image> &b, Work &) { b[1] = b[0]; }, CERE_ERR_ARGUMENT);              // overlap
        refused([](std::vector<Image> &b, Work &) { std::swap(b[0], b[1]); }, CERE_ERR_ARGUMENT);    // unsorted
        refused([](std::vector<Image> &b, Work &) { b.pop_back(); }, CERE_ERR_ARGUMENT);  // unclaimed pad rows
        refused([](std::vector<Image> &b, Work &) { b.resize(33, b[0]); }, CERE_ERR_ARGUMENT);  // > 32 images
        refused([](std::vector<Image> &, Work &wb) { wb.item.operation = CERE_OP_VERIFY; }, CERE_ERR_ARGUMENT);
        {
            Work wb(session, CERE_OP_PREFILL, prompt, 2, generation);
            wb.item.flags = CERE_WORK_NO_SPECULATION;
            wb.item.images = &im[0].input;  // count 0 with a pointer
            wb.item.image_count = 0;
            CHECK_EQ(cere_executor_submit(executor, &wb.batch, &ticket), CERE_ERR_ARGUMENT);
            wb.item.images = nullptr;
            wb.item.image_count = 1;
            CHECK_EQ(cere_executor_submit(executor, &wb.batch, &ticket), CERE_ERR_ARGUMENT);
        }
        CHECK_EQ(submit(w, im), CERE_OK);
        // The patches were copied: the caller may reuse its buffer at once.
        std::fill(im[0].pixels.begin(), im[0].pixels.end(), 0);
        CHECK_EQ(session->images.size(), 2);
        CHECK_EQ(session->images[0].pixels.size(), 16 * 1536 * 2);
        CHECK_EQ(session->images[0].pixels[0], 0x5a);
        CHECK_EQ(session->images[1].span.token_offset, offsets[1]);
        CHECK_EQ(session->images[1].pad_id, 1000001);
        CHECK_EQ(session->rope_delta, -4);  // two merged 2x2 grids, each advancing 2 instead of 4 positions
        const cere_completion c = drain(executor, ticket);
        CHECK_EQ(c.committed_tokens, 2);
        const auto expected = toy_sequence(prompt, 2, a4_support().config.vocab_size);
        CHECK_EQ(w.out[0], expected[0]);  // the toy's tokens follow the ids (pixels do not reach it)
    }
    {
        Work d(session, CERE_OP_DECODE, {}, 1, generation, {}, 1);
        d.item.flags = CERE_WORK_NO_SPECULATION;
        std::vector<Image> im = images();
        CHECK_EQ(submit(d, im), CERE_ERR_ARGUMENT);  // DECODE with images
        im.clear();
        CHECK_EQ(submit(d, im), CERE_OK);  // DECODE of the image session without images
        drain(executor, ticket);
    }
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    // M2.13-L: a session context above indexer_budget (2,048) takes images (K9 rotates by positions3).
    CHECK_EQ(cere_session_create(executor, model, 2049, &session), CERE_OK);
    {
        Work w(session, CERE_OP_PREFILL, prompt, 2, generation);
        std::vector<Image> im = images();
        CHECK_EQ(submit(w, im), CERE_OK);  // flags 0: a model without a draft
        const cere_completion c = drain(executor, ticket);
        CHECK_EQ(c.committed_tokens, 2);
        CHECK_EQ(session->rope_delta, -4);
    }
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    // A text-only model refuses images.
    CHECK_EQ(cere_session_create(executor, text_model, 2048, &session), CERE_OK);
    {
        Work w(session, CERE_OP_PREFILL, prompt, 2, generation_of(executor, text_model));
        w.item.flags = CERE_WORK_NO_SPECULATION;
        std::vector<Image> im = images();
        CHECK_EQ(submit(w, im), CERE_ERR_UNSUPPORTED);
    }
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_model_release(executor, text_model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// m213 design note C11 (D12), resized by m213l design note D-L7: the image buffers memory_plan.cpp adds to a plan
// whose support entry has a vision config (layout.h, shared with the CUDA plan): the device encoder output of one
// image and the call's image embeddings, the device patch copy of one image, plus the same-size pinned host patch
// copy, and the mRoPE positions. The 897,862,112 B of model.visual.* weights are DEVICE tensors of the desc and are
// counted like every tensor; a text-only plan (support->vision null) gains nothing.
void test_image_memory() {
    namespace layout = cere::model::layout;
    const cere_qwen4_exp_vision_config &v = *cere::host::a4_vision_support().vision;
    CHECK_EQ(layout::kMaxImageTokens, uint64_t(4096));
    CHECK_EQ(layout::image_embed_bytes(v), uint64_t(20971520));   // [4,096, 2,560] BF16
    CHECK_EQ(layout::image_patch_bytes(v), uint64_t(50331648));   // 16,384 patches x 1,536 x 2 B
    CHECK_EQ(layout::image_embed_bytes(v) % layout::kAlign, 0);
    CHECK_EQ(layout::image_patch_bytes(v) % layout::kAlign, 0);
    // A 1280x900 screenshot (grid 1 x 56 x 80: 1,120 merged rows) and the census's largest image (grid 1 x 98 x 98:
    // 2,401 merged rows) each fit one image.
    CHECK(uint64_t(56) * 80 / 4 <= layout::kMaxImageTokens);
    CHECK(uint64_t(98) * 98 / 4 <= layout::kMaxImageTokens);
    CHECK_EQ(layout::max_image_patches(v, 2048), uint64_t(8192));
    CHECK_EQ(layout::max_image_patches(v, 200000), uint64_t(16384));
    CHECK_EQ(layout::image_positions_bytes(4096), uint64_t(196680));  // [3, 4,096] + [3, 4,099] int64
    CHECK(cere::host::a4_support().vision == nullptr);
}

// --- M2.7 sampling (m27 design note C4, I1-I8) ----------------------------------------------------------------

cere_sampling_params sampling_params(float temperature, uint32_t top_k, uint64_t seed = 5) {
    cere_sampling_params s{};
    s.struct_size = sizeof(s);
    s.abi_version = CERE_EXECUTOR_ABI_VERSION;
    s.temperature = temperature;
    s.top_p = 1.0f;
    s.top_k = top_k;
    s.repetition_penalty = 1.0f;
    s.seed = seed;
    return s;
}

struct Run {
    std::vector<uint32_t> tokens;
    std::vector<cere_token_score> scores;
    std::vector<cere_token_logprobs> logprobs;
};

// One PREFILL work item of `n` tokens after `prompt` on a fresh session.
Run run_item(cere_executor *executor, cere_model *model, uint64_t generation, const std::vector<uint32_t> &prompt,
             uint64_t n, const cere_sampling_params *sampling, uint32_t top_logprobs, bool want_logprobs) {
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &session), CERE_OK);
    Run run;
    run.logprobs.resize(n);
    {
        Work w(session, CERE_OP_PREFILL, prompt, n, generation);
        w.item.sampling = sampling;
        w.item.out_logprobs = want_logprobs ? run.logprobs.data() : nullptr;
        w.item.top_logprobs = top_logprobs;
        cere_completion_ticket ticket{};
        CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
        const cere_completion c = drain(executor, ticket);
        CHECK_EQ(c.committed_tokens, n);
        run.tokens = w.out;
        run.scores = w.scores;
    }
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    return run;
}

// The toy's logits for a sampled step (8 - j at id (next + j) mod vocab for j < 8, -inf elsewhere) as BF16 bits.
std::vector<uint16_t> expected_toy_row(uint32_t next, uint32_t vocab) {
    std::vector<uint16_t> row(vocab, 0xFF80);
    for (uint32_t j = 0; j < 8; ++j) {
        const float v = float(8 - j);
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        row[(uint64_t(next) + j) % vocab] = uint16_t(bits >> 16);
    }
    return row;
}

// The tokens a sampled item commits on the toy, recomputed step by step with sampler_ref (D2-D5: counts of this
// item's output, counter key = the position of the last consumed token, channels (0, 1)).
std::vector<uint32_t> expected_sampled(std::vector<uint32_t> tokens, uint64_t n, uint32_t vocab,
                                       const cere_sampling_params &s) {
    namespace sr = cere::sampler_ref;
    std::vector<uint32_t> counts(vocab, 0);
    std::vector<uint32_t> out;
    std::vector<sr::Bias> bias;
    for (uint64_t i = 0; i < s.logit_bias_count; ++i) bias.push_back({s.logit_bias[i].id, s.logit_bias[i].bias});
    for (uint64_t i = 0; i < n; ++i) {
        const uint32_t next = cere::toy::next_token(tokens.back(), tokens.size(), vocab);
        const std::vector<uint16_t> row = expected_toy_row(next, vocab);
        std::vector<float> x(vocab);
        sr::process(row.data(), vocab, counts.data(), s.frequency_penalty, s.presence_penalty, s.repetition_penalty,
                    nullptr, bias.data(), bias.size(), x.data());
        uint32_t id = sr::top2(x.data(), vocab).id;
        if (s.temperature != 0.0f) {
            std::vector<float> probs(vocab);
            sr::probabilities(x.data(), vocab, s.temperature, probs.data());
            const auto mask = sr::keep_mask(probs.data(), vocab, s.top_k, s.top_p, s.min_p);
            id = sr::draw(probs.data(), vocab, mask, sr::uniform(s.seed, uint32_t(tokens.size() - 1), 0, 1));
        }
        ++counts[id];
        tokens.push_back(id);
        out.push_back(id);
    }
    return out;
}

bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

void test_sampling_validation() {
    cere_executor *executor = make_executor();
    Desc d = toy_desc();
    cere_model *model = sealed_model(executor, d);
    const uint64_t generation = generation_of(executor, model);
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &session), CERE_OK);
    Work w(session, CERE_OP_PREFILL, {1, 2}, 2, generation);
    std::vector<cere_token_logprobs> lp(2);
    cere_completion_ticket ticket{};
    const auto submit = [&](const cere_sampling_params &s) {
        w.item.sampling = &s;
        const cere_status st = cere_executor_submit(executor, &w.batch, &ticket);
        w.item.sampling = nullptr;
        return st;
    };
    using Edit = std::function<void(cere_sampling_params &)>;
    const std::vector<Edit> arguments = {
        [](cere_sampling_params &s) { s.struct_size -= 1; },
        [](cere_sampling_params &s) { s.struct_size = 0; },
        [](cere_sampling_params &s) { s.flags = 1; },
        [](cere_sampling_params &s) { s.flags = UINT32_C(0x80000000); },
        [&](cere_sampling_params &s) { s.temperature = nan; },
        [](cere_sampling_params &s) { s.temperature = 5e-7f; },
        [](cere_sampling_params &s) { s.temperature = -1.0f; },
        [&](cere_sampling_params &s) { s.temperature = inf; },
        [](cere_sampling_params &s) { s.top_p = 0.0f; },
        [](cere_sampling_params &s) { s.top_p = 1.0001f; },
        [&](cere_sampling_params &s) { s.top_p = nan; },
        [](cere_sampling_params &s) { s.min_p = -0.1f; },
        [](cere_sampling_params &s) { s.min_p = 1.1f; },
        [&](cere_sampling_params &s) { s.min_p = nan; },
        [](cere_sampling_params &s) { s.top_k = 0; },
        [&](cere_sampling_params &s) { s.top_k = vocab + 1; },
        [](cere_sampling_params &s) { s.frequency_penalty = 2.1f; },
        [](cere_sampling_params &s) { s.frequency_penalty = -2.1f; },
        [&](cere_sampling_params &s) { s.frequency_penalty = nan; },
        [](cere_sampling_params &s) { s.presence_penalty = 2.5f; },
        [&](cere_sampling_params &s) { s.presence_penalty = nan; },
        [](cere_sampling_params &s) { s.repetition_penalty = 0.0f; },
        [](cere_sampling_params &s) { s.repetition_penalty = -1.0f; },
        [](cere_sampling_params &s) { s.repetition_penalty = 2.1f; },
        [&](cere_sampling_params &s) { s.repetition_penalty = nan; },
    };
    for (const Edit &edit : arguments) {
        cere_sampling_params s = sampling_params(0.7f, vocab);
        edit(s);
        CHECK_EQ(submit(s), CERE_ERR_ARGUMENT);
        CHECK_EQ(ticket.submission_id, 0);
    }
    {
        cere_sampling_params s = sampling_params(0.7f, vocab);
        s.abi_version = CERE_EXECUTOR_ABI_VERSION + 1;
        CHECK_EQ(submit(s), CERE_ERR_UNSUPPORTED);
    }
    // logit_bias: null iff empty, ids < vocab and unique, finite values.
    const std::vector<std::vector<cere_logit_bias>> bad_bias = {
        {{vocab, 1.0f}}, {{3, 1.0f}, {3, 2.0f}}, {{3, nan}}, {{3, inf}}, {{3, -inf}}};
    for (const auto &bias : bad_bias) {
        cere_sampling_params s = sampling_params(0.7f, vocab);
        s.logit_bias = bias.data();
        s.logit_bias_count = bias.size();
        CHECK_EQ(submit(s), CERE_ERR_ARGUMENT);
    }
    {
        const cere_logit_bias one{3, 1.0f};
        cere_sampling_params s = sampling_params(0.7f, vocab);
        s.logit_bias_count = 1;  // null list
        CHECK_EQ(submit(s), CERE_ERR_ARGUMENT);
        s.logit_bias = &one;
        s.logit_bias_count = 0;  // list without a count
        CHECK_EQ(submit(s), CERE_ERR_ARGUMENT);
    }
    // The M2.12 group is zero-required (ledger L0-2).
    static const uint32_t suppressed[1] = {4};
    for (int field = 0; field < 3; ++field) {
        cere_sampling_params s = sampling_params(0.7f, vocab);
        s.suppress_token_ids = field == 0 ? suppressed : nullptr;
        s.suppress_token_count = field == 1 ? 1 : 0;
        s.mask_until_committed = field == 2 ? 1 : 0;
        CHECK_EQ(submit(s), CERE_ERR_UNSUPPORTED);
    }
    // top_logprobs: <= CERE_MAX_TOP_LOGPROBS and only with out_logprobs.
    w.item.top_logprobs = 1;
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_ERR_ARGUMENT);
    w.item.out_logprobs = lp.data();
    w.item.top_logprobs = CERE_MAX_TOP_LOGPROBS + 1;
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_ERR_ARGUMENT);
    w.item.out_logprobs = nullptr;
    w.item.top_logprobs = 0;
    CHECK_EQ(ticket.submission_id, 0);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);

    // Every range boundary is accepted, and the inputs are copied at submit.
    const std::vector<Edit> boundaries = {
        [](cere_sampling_params &s) { s.temperature = 1e-6f; },
        [](cere_sampling_params &s) { s.temperature = 0.0f; },
        [](cere_sampling_params &s) { s.top_p = std::nextafter(0.0f, 1.0f); },
        [](cere_sampling_params &s) { s.min_p = 1.0f; },
        [](cere_sampling_params &s) { s.top_k = 1; },
        [](cere_sampling_params &s) { s.frequency_penalty = -2.0f; s.presence_penalty = 2.0f; },
        [](cere_sampling_params &s) { s.repetition_penalty = 2.0f; },
        [](cere_sampling_params &s) { s.repetition_penalty = std::nextafter(0.0f, 1.0f); },
    };
    for (const Edit &edit : boundaries) {
        cere_sampling_params s = sampling_params(0.7f, vocab);
        edit(s);
        run_item(executor, model, generation, {1, 2}, 2, &s, CERE_MAX_TOP_LOGPROBS, true);
    }
    {
        cere_session *copy_session = nullptr;
        CHECK_EQ(cere_session_create(executor, model, 32, &copy_session), CERE_OK);
        const uint32_t first = cere::toy::next_token(2, 2, vocab);
        std::vector<cere_logit_bias> bias = {{(first + 5) % vocab, 100.0f}, {vocab - 1, -100.0f}};
        cere_sampling_params s = sampling_params(0.0f, vocab);
        s.logit_bias = bias.data();
        s.logit_bias_count = bias.size();
        Work copy(copy_session, CERE_OP_PREFILL, {1, 2}, 1, generation);
        copy.item.sampling = &s;
        CHECK_EQ(cere_executor_submit(executor, &copy.batch, &ticket), CERE_OK);
        s = sampling_params(0.0f, vocab);  // overwritten after submit: the copy decides
        bias[0].bias = -100.0f;
        drain(executor, ticket);
        CHECK_EQ(copy.out[0], (first + 5) % vocab);
        CHECK_EQ(cere_session_release(executor, copy_session), CERE_OK);
    }
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_toy_sampling() {
    cere_executor *executor = make_executor();
    Desc d = toy_desc();
    cere_model *model = sealed_model(executor, d);
    const uint64_t generation = generation_of(executor, model);
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    const std::vector<uint32_t> prompt = {1, 2};
    const auto greedy = toy_sequence(prompt, 6, vocab);
    double sum8 = 0.0;
    for (int v = 1; v <= 8; ++v) sum8 += std::exp(double(v));
    const double lse8 = std::log(sum8);

    // sampling == null: the pre-M2.7 outputs, byte for byte.
    const Run plain = run_item(executor, model, generation, prompt, 6, nullptr, 0, false);
    CHECK(plain.tokens == greedy);
    for (size_t i = 0; i < greedy.size(); ++i) {
        CHECK_EQ(plain.scores[i].runner_up, (greedy[i] + 1) % vocab);
        CHECK(plain.scores[i].logit == 1.0f && plain.scores[i].runner_up_logit == 0.0f);
    }

    // Neutral greedy params with logprobs: the same tokens; logprobs of the 8 finite ids, then -inf ones.
    const cere_sampling_params neutral = sampling_params(0.0f, vocab);
    const Run g = run_item(executor, model, generation, prompt, 6, &neutral, CERE_MAX_TOP_LOGPROBS, true);
    CHECK(g.tokens == greedy);
    for (size_t i = 0; i < greedy.size(); ++i) {
        CHECK_EQ(g.scores[i].runner_up, (greedy[i] + 1) % vocab);
        CHECK(g.scores[i].logit == 8.0f && g.scores[i].runner_up_logit == 7.0f);
        const cere_token_logprobs &lp = g.logprobs[i];
        CHECK_EQ(lp.id, greedy[i]);
        CHECK(near(lp.logprob, 8.0 - lse8));
        CHECK_EQ(lp.top_count, CERE_MAX_TOP_LOGPROBS);
        CHECK_EQ(lp.reserved, 0);
        for (uint32_t j = 0; j < 8; ++j) {
            CHECK_EQ(lp.top[j].id, (greedy[i] + j) % vocab);
            CHECK(near(lp.top[j].logprob, 8.0 - j - lse8));
        }
        for (uint32_t j = 8; j < CERE_MAX_TOP_LOGPROBS; ++j) {
            CHECK(std::isinf(lp.top[j].logprob) && lp.top[j].logprob < 0.0f);
        }
    }

    // Logprobs without sampling: greedy tokens, top_count = top_logprobs.
    const Run only = run_item(executor, model, generation, prompt, 3, nullptr, 2, true);
    for (size_t i = 0; i < 3; ++i) {
        CHECK_EQ(only.tokens[i], greedy[i]);
        CHECK_EQ(only.logprobs[i].top_count, 2);
        CHECK(near(only.logprobs[i].logprob, 8.0 - lse8));
        CHECK_EQ(only.logprobs[i].top[1].id, (greedy[i] + 1) % vocab);
    }

    // Seeded draws: the same seed repeats, another seed gives the sampler_ref sequence of that seed.
    const cere_sampling_params s11 = sampling_params(1.0f, vocab, 11);
    const cere_sampling_params s12 = sampling_params(1.0f, vocab, 12);
    const Run a = run_item(executor, model, generation, prompt, 8, &s11, 1, true);
    const Run b = run_item(executor, model, generation, prompt, 8, &s11, 0, false);
    const Run c = run_item(executor, model, generation, prompt, 8, &s12, 0, false);
    CHECK(a.tokens == expected_sampled(prompt, 8, vocab, s11));
    CHECK(b.tokens == a.tokens);
    CHECK(c.tokens == expected_sampled(prompt, 8, vocab, s12));
    CHECK(c.tokens != a.tokens);
    for (size_t i = 0; i < 8; ++i) {
        // runner_up is the most probable other token; logprobs use log_softmax(x / T) over the 8 finite ids.
        const uint32_t next = cere::toy::next_token(i == 0 ? prompt.back() : a.tokens[i - 1], prompt.size() + i, vocab);
        const uint32_t j = (a.tokens[i] + vocab - next) % vocab;
        CHECK(j < 8);
        CHECK_EQ(a.scores[i].runner_up, j == 0 ? (next + 1) % vocab : next);
        CHECK(a.scores[i].logit == float(8 - j));
        CHECK(near(a.logprobs[i].logprob, 8.0 - j - lse8));
    }
    {
        cere_sampling_params filtered = sampling_params(0.8f, 3, 21);
        filtered.top_p = 0.9f;
        filtered.min_p = 0.2f;
        const Run f = run_item(executor, model, generation, prompt, 8, &filtered, 0, false);
        CHECK(f.tokens == expected_sampled(prompt, 8, vocab, filtered));
        cere_sampling_params top1 = sampling_params(0.8f, 1, 21);  // top_k 1: always the greedy token
        CHECK(run_item(executor, model, generation, prompt, 6, &top1, 0, false).tokens == greedy);
    }

    // logit_bias, applied to every step of the item.
    {
        const uint32_t first = greedy[0];
        const cere_logit_bias bias{(first + 3) % vocab, 100.0f};
        cere_sampling_params s = sampling_params(0.0f, vocab);
        s.logit_bias = &bias;
        s.logit_bias_count = 1;
        const Run r = run_item(executor, model, generation, prompt, 2, &s, 0, false);
        CHECK_EQ(r.tokens[0], (first + 3) % vocab);
        CHECK_EQ(r.tokens[1], cere::toy::next_token(r.tokens[0], 3, vocab));
        CHECK(r.scores[0].logit == 105.0f);
    }

    // Penalties count this item's output: the first token 2333 is the 8th finite id of the second step.
    {
        uint32_t x = 0;
        while ((31 * uint64_t(x) + 2) % vocab != 2333) ++x;
        const std::vector<uint32_t> p = {1, x};
        CHECK_EQ(cere::toy::next_token(2333, 3, vocab), 2326);
        cere_sampling_params s = sampling_params(0.0f, vocab);
        const Run before = run_item(executor, model, generation, p, 2, &s, 8, true);
        s.presence_penalty = 0.5f;
        const Run after = run_item(executor, model, generation, p, 2, &s, 8, true);
        for (const Run *r : {&before, &after}) {
            CHECK_EQ(r->tokens[0], 2333);
            CHECK_EQ(r->tokens[1], 2326);
            CHECK_EQ(r->logprobs[1].top[7].id, 2333);
        }
        CHECK(near(before.logprobs[1].top[7].logprob, 1.0 - lse8));
        const double lse_penalized = std::log(sum8 - std::exp(1.0) + std::exp(0.5));
        CHECK(near(after.logprobs[1].top[7].logprob, 0.5 - lse_penalized));
        // The count is per item: the first step of a fresh item is unpenalized.
        CHECK(near(after.logprobs[0].logprob, 8.0 - lse8));
    }
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

std::vector<std::string> lines_of(const std::string &path) {
    std::ifstream in(path);
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) lines.push_back(line);
    return lines;
}

// `record=<absolute path>`: one line of numbers per executed step (session, tokens fed, the sampling fields or
// `null`, the logprobs request or `null`, the committed token); a relative or empty path is refused.
void test_toy_record() {
    for (const char *bad : {"toy,record=", "toy,record=relative/path"}) {
        cere_executor_options options{};
        options.struct_size = sizeof(options);
        options.abi_version = CERE_EXECUTOR_ABI_VERSION;
        options.device_id = bad;
        options.worker_epoch = kEpoch;
        options.memory_budget_bytes = uint64_t(1) << 30;
        cere_executor *executor = nullptr;
        CHECK_EQ(cere_executor_create(&executor, &options), CERE_ERR_ARGUMENT);
    }
    const std::string path = "/tmp/cere-toy-record-" + std::to_string(uint64_t(::getpid())) + ".txt";
    std::remove(path.c_str());
    const std::string id = "toy,record=" + path;
    cere_executor *executor = make_executor(id.c_str());
    Desc d = toy_desc();
    cere_model *model = sealed_model(executor, d);
    const uint64_t generation = generation_of(executor, model);
    const std::vector<uint32_t> prompt = {1, 2, 3};
    const Run plain = run_item(executor, model, generation, prompt, 2, nullptr, 0, false);
    cere_sampling_params s = sampling_params(0.5f, 3, 99);
    s.top_p = 0.75f;
    s.min_p = 0.25f;
    s.frequency_penalty = 0.5f;
    s.presence_penalty = -1.5f;
    s.repetition_penalty = 1.25f;
    const cere_logit_bias bias[2] = {{4, 1.0f}, {5, -2.0f}};
    s.logit_bias = bias;
    s.logit_bias_count = 2;
    const Run sampled = run_item(executor, model, generation, prompt, 1, &s, 3, true);
    const Run only = run_item(executor, model, generation, prompt, 1, nullptr, 0, true);
    const auto t = [](uint32_t token) { return " token=" + std::to_string(token); };
    const std::vector<std::string> expected = {
        "session=1 count=3 sampling=null logprobs=null" + t(plain.tokens[0]),
        "session=1 count=1 sampling=null logprobs=null" + t(plain.tokens[1]),
        "session=2 count=3 temperature=0.5 top_k=3 top_p=0.75 min_p=0.25 frequency=0.5 presence=-1.5 "
        "repetition=1.25 logit_bias=2 seed=99 logprobs=3" + t(sampled.tokens[0]),
        "session=3 count=3 sampling=null logprobs=0" + t(only.tokens[0]),
    };
    CHECK(lines_of(path) == expected);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    std::remove(path.c_str());
}

// Ledger CR4 (M2.6 x M2.7): a decode round steps every decode-ready ticket of its model as one row of one
// step_rows, sampled and logprobs tickets included (RowArgs carries the row's ItemSampling and logprobs output).
// The toy records each row in row order, so a round of four tickets is four consecutive lines in submission order.
void test_rows_sampled_tickets_share_the_round() {
    const std::string path = "/tmp/cere-toy-rows-sampled-" + std::to_string(uint64_t(::getpid())) + ".txt";
    std::remove(path.c_str());
    const std::string id = "toy,record=" + path;
    cere_executor *executor = make_executor(id.c_str());
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    const std::vector<std::vector<uint32_t>> prompts = {{1, 2}, {3, 4, 5}, {6}, {7, 8}};
    std::vector<cere_session *> sessions;
    for (const auto &p : prompts) sessions.push_back(prefilled_session(executor, model, p));
    std::remove(path.c_str());  // the record of the four prefills
    const uint64_t n = 4;
    const cere_sampling_params s = sampling_params(2.0f, 8, 11);
    std::vector<cere_token_logprobs> logprobs(n);
    std::vector<Job> jobs;
    for (size_t i = 0; i < sessions.size(); ++i) {
        Job job;
        job.work = std::make_unique<Work>(sessions[i], CERE_OP_DECODE, std::vector<uint32_t>{}, n,
                                          generation_of(executor, model), std::vector<uint32_t>{},
                                          sessions[i]->revision);
        if (i == 1) job.work->item.sampling = &s;  // sampled
        if (i == 2) {                             // greedy with logprobs
            job.work->item.out_logprobs = logprobs.data();
            job.work->item.top_logprobs = 2;
        }
        CHECK_EQ(cere_executor_submit(executor, &job.work->batch, &job.ticket), CERE_OK);
        jobs.push_back(std::move(job));
    }
    // One poll: one decode round in which every ticket commits one token.
    poll_pending(executor, jobs[0].ticket);
    for (const Job &j : jobs) CHECK_EQ(committed_of(executor, j.ticket), 1);
    for (const Job &j : jobs) CHECK_EQ(drain(executor, j.ticket).committed_tokens, n);
    // Greedy rows match their solo runs.
    check_solo(*jobs[0].work, n, prompts[0], 1);
    check_solo(*jobs[3].work, n, prompts[3], 1);
    // The sampled ticket commits the sampler_ref sequence, which differs from the greedy one.
    std::vector<uint32_t> seq = prompts[1];
    seq.push_back(toy_sequence(prompts[1], 1, vocab)[0]);
    const std::vector<uint32_t> sampled = expected_sampled(seq, n, vocab, s);
    CHECK(jobs[1].work->out == sampled);
    CHECK(sampled != toy_sequence(seq, n, vocab));
    // The logprobs ticket commits the greedy tokens (its scores are the sampler stage's) with its logprobs filled.
    const Solo solo = solo_run(prompts[2], 1 + n);
    for (uint64_t i = 0; i < n; ++i) CHECK_EQ(jobs[2].work->out[i], solo.out[1 + i]);
    for (uint64_t i = 0; i < n; ++i) {
        CHECK_EQ(logprobs[i].id, jobs[2].work->out[i]);
        CHECK_EQ(logprobs[i].top_count, 2);
        CHECK(logprobs[i].logprob < 0.0f);
    }
    // Every round: one step_rows of the four rows, in submission order.
    std::vector<std::string> expected;
    const auto line = [](int session, const std::string &mid, uint32_t token) {
        return "session=" + std::to_string(session) + " count=1 " + mid + " token=" + std::to_string(token);
    };
    for (uint64_t i = 0; i < n; ++i) {
        expected.push_back(line(1, "sampling=null logprobs=null", jobs[0].work->out[i]));
        expected.push_back(line(2,
                                "temperature=2 top_k=8 top_p=1 min_p=0 frequency=0 presence=0 repetition=1 "
                                "logit_bias=0 seed=11 logprobs=null",
                                jobs[1].work->out[i]));
        expected.push_back(line(3, "sampling=null logprobs=2", jobs[2].work->out[i]));
        expected.push_back(line(4, "sampling=null logprobs=null", jobs[3].work->out[i]));
    }
    CHECK(lines_of(path) == expected);
    for (cere_session *session : sessions) CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    std::remove(path.c_str());
}

// One sampled DECODE item's outputs (tokens, scores, logprobs) after a 1-token prefill of `prompt`.
struct RowRun {
    std::vector<uint32_t> out;
    std::vector<cere_token_score> scores;
    std::vector<cere_token_logprobs> logprobs;
};

// Ledger CR4 / I10 on the toy: the target item (penalties, logit_bias, seed, top logprobs) run as the last of R
// concurrent tickets (R = 1..4; the others sampled with other seeds and penalties, or greedy) commits the same
// tokens, scores and logprobs at every R, i.e. the counter key (M8: seed and the session position) and the output
// counts are the row's own, whatever the batch composition and the row index; and those tokens are sampler_ref's.
void test_rows_sampler_invariance() {
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    const std::vector<uint32_t> target_prompt = {9, 10, 11};
    const std::vector<std::vector<uint32_t>> others = {{1, 2}, {3, 4, 5}, {6}};
    const uint64_t n = 12;
    const cere_logit_bias bias{17, 0.5f};
    cere_sampling_params target = sampling_params(1.5f, 6, 7);
    target.frequency_penalty = 0.75f;
    target.presence_penalty = 0.5f;
    target.repetition_penalty = 1.25f;
    target.logit_bias = &bias;
    target.logit_bias_count = 1;
    cere_sampling_params other_a = sampling_params(2.0f, 8, 101);
    other_a.frequency_penalty = -0.5f;
    const cere_sampling_params other_b = sampling_params(0.7f, 4, 7);  // the target's seed, another session
    const cere_sampling_params *other_params[3] = {&other_a, nullptr, &other_b};
    std::vector<RowRun> runs;
    for (uint32_t rows = 1; rows <= 4; ++rows) {
        cere_executor *executor = make_executor();
        Desc d = rows_desc();
        cere_model *model = sealed_model(executor, d);
        std::vector<cere_session *> sessions;
        for (uint32_t i = 0; i + 1 < rows; ++i) sessions.push_back(prefilled_session(executor, model, others[i]));
        sessions.push_back(prefilled_session(executor, model, target_prompt));
        RowRun run;
        run.logprobs.resize(n);
        std::vector<cere_token_logprobs> other_logprobs(n);
        std::vector<Job> jobs;
        for (uint32_t i = 0; i < rows; ++i) {
            Job job;
            job.work = std::make_unique<Work>(sessions[i], CERE_OP_DECODE, std::vector<uint32_t>{}, n,
                                              generation_of(executor, model), std::vector<uint32_t>{},
                                              sessions[i]->revision);
            if (i + 1 == rows) {
                job.work->item.sampling = &target;
                job.work->item.out_logprobs = run.logprobs.data();
                job.work->item.top_logprobs = 5;
            } else {
                job.work->item.sampling = other_params[i];
                if (i == 1) {
                    job.work->item.out_logprobs = other_logprobs.data();
                    job.work->item.top_logprobs = 3;
                }
            }
            CHECK_EQ(cere_executor_submit(executor, &job.work->batch, &job.ticket), CERE_OK);
            jobs.push_back(std::move(job));
        }
        for (const Job &j : jobs) CHECK_EQ(drain(executor, j.ticket).committed_tokens, n);
        run.out = jobs.back().work->out;
        run.scores = jobs.back().work->scores;
        // The co-batched sampled tickets commit their own sampler_ref sequences too.
        for (uint32_t i = 0; i + 1 < rows; ++i) {
            if (other_params[i] == nullptr) continue;
            std::vector<uint32_t> seq = others[i];
            seq.push_back(toy_sequence(others[i], 1, vocab)[0]);
            CHECK(jobs[i].work->out == expected_sampled(seq, n, vocab, *other_params[i]));
        }
        runs.push_back(std::move(run));
        for (cere_session *session : sessions) CHECK_EQ(cere_session_release(executor, session), CERE_OK);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
    std::vector<uint32_t> seq = target_prompt;
    seq.push_back(toy_sequence(target_prompt, 1, vocab)[0]);
    CHECK(runs[0].out == expected_sampled(seq, n, vocab, target));
    for (uint32_t r = 1; r < 4; ++r) {
        CHECK(runs[r].out == runs[0].out);
        CHECK(std::memcmp(runs[r].scores.data(), runs[0].scores.data(), n * sizeof(cere_token_score)) == 0);
        CHECK(std::memcmp(runs[r].logprobs.data(), runs[0].logprobs.data(), n * sizeof(cere_token_logprobs)) == 0);
    }
    for (uint64_t i = 0; i < n; ++i) {
        CHECK_EQ(runs[3].logprobs[i].id, runs[3].out[i]);
        CHECK_EQ(runs[3].logprobs[i].top_count, 5);
    }
}

// RowArgs (host/device.h): a row's logprobs output needs its sampling (ARGUMENT for the whole call, nothing runs);
// a sampled row is chosen by the sampler like `step` and its output counts are the session's own.
void test_rows_sampling_args() {
    cere_executor *executor = make_executor();
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *a = prefilled_session(executor, model, {1, 2});
    cere_session *b = prefilled_session(executor, model, {3, 4});
    const uint32_t vocab = d.config.vocab_size;
    cere_token_logprobs lp{};
    cere::host::RowArgs rows[2] = {{a->device.get(), a->tokens.back()}, {b->device.get(), b->tokens.back()}};
    rows[1].logprobs = &lp;
    cere_token_score out[2] = {};
    cere_status status[2] = {77, 77};
    CHECK_EQ(toy(executor).step_rows(*model->device, rows, 2, out, status), CERE_ERR_ARGUMENT);
    CHECK_EQ(status[0], 77);
    CHECK_EQ(status[1], 77);
    cere::host::ItemSampling sampling;
    sampling.sampled = true;
    sampling.params = sampling_params(1.0f, 8, 3);
    sampling.top_logprobs = 1;
    rows[1].sampling = &sampling;
    CHECK_EQ(toy(executor).step_rows(*model->device, rows, 2, out, status), CERE_OK);
    CHECK_EQ(status[0], CERE_OK);
    CHECK_EQ(status[1], CERE_OK);
    CHECK_EQ(out[0].id, toy_sequence({1, 2}, 2, vocab)[1]);  // the greedy row is untouched by its neighbour
    std::vector<uint32_t> seq = {3, 4, toy_sequence({3, 4}, 1, vocab)[0]};
    CHECK_EQ(out[1].id, expected_sampled(seq, 1, vocab, sampling.params)[0]);
    CHECK_EQ(lp.id, out[1].id);
    CHECK_EQ(lp.top_count, 1);
    CHECK_EQ(cere_session_release(executor, a), CERE_OK);
    CHECK_EQ(cere_session_release(executor, b), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// ---- C2 (host semantics over the toy): m211 design note D1, D4-D6, D14 and ledger N/K/C 56 ----

uint64_t snapshot_size(uint64_t position, uint64_t token_count) {
    snap::Plan plan;
    CHECK(snap::plan(cere::toy::fixture_config(), position, 0, token_count, &plan));
    return plan.total_bytes;
}

std::vector<uint8_t> read_all(cere_executor *executor, const cere_snapshot *s) {
    cere_snapshot_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(cere_snapshot_get_info(executor, s, &info), CERE_OK);
    std::vector<uint8_t> bytes(info.bytes);
    CHECK_EQ(cere_snapshot_read(executor, s, 0, bytes.data(), bytes.size()), CERE_OK);
    return bytes;
}

// Prefill `prompt` (generating `n`) and drain; returns the generated tokens.
std::vector<uint32_t> run(cere_executor *executor, cere_session *session, uint32_t op, std::vector<uint32_t> prompt,
                          uint64_t n, uint64_t generation, uint64_t revision, uint32_t flags = 0) {
    Work w(session, op, std::move(prompt), n, generation, {}, revision);
    w.item.flags = flags;
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
    const cere_completion c = drain(executor, ticket);
    CHECK_EQ(c.status, CERE_OK);
    return std::vector<uint32_t>(w.out.begin(), w.out.begin() + c.committed_tokens);
}

void test_snapshot_api() {
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    const std::vector<uint32_t> prompt = {5, 9, 69999};
    const auto expected = toy_sequence(prompt, 12, vocab);
    cere_executor *executor = make_executor();

    // Disabled (snapshot_host_bytes 0): no capability, every call on the model UNSUPPORTED.
    Desc off = toy_desc();
    cere_model *plain = sealed_model(executor, off);
    const uint64_t plain_generation = generation_of(executor, plain);
    cere_session *ps = nullptr;
    CHECK_EQ(cere_session_create(executor, plain, 32, &ps), CERE_OK);
    run(executor, ps, CERE_OP_PREFILL, prompt, 2, plain_generation, 0);
    cere_snapshot *s = reinterpret_cast<cere_snapshot *>(uintptr_t(1));
    CHECK_EQ(cere_session_snapshot(executor, ps, 1, &s), CERE_ERR_UNSUPPORTED);
    CHECK(s == nullptr);
    const std::vector<uint8_t> foreign = valid(1, 1, 1);
    CHECK_EQ(cere_snapshot_import(executor, plain, foreign.data(), foreign.size(), &s), CERE_ERR_UNSUPPORTED);

    // Enabled: native_state_snapshot 1 and the SNAPSHOT|RESTORE operations (ledger C 56, K).
    Desc on = toy_desc();
    on.desc.snapshot_host_bytes = uint64_t(1) << 20;
    cere_model *model = sealed_model(executor, on);
    const uint64_t generation = generation_of(executor, model);
    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(cere_executor_capabilities(executor, plain, &caps), CERE_OK);
    CHECK_EQ(caps.native_state_snapshot, 0);
    CHECK_EQ(caps.operations & (CERE_OPS_SNAPSHOT | CERE_OPS_RESTORE), 0);
    CHECK_EQ(cere_executor_capabilities(executor, model, &caps), CERE_OK);
    CHECK_EQ(caps.native_state_snapshot, 1);
    CHECK_EQ(caps.operations,
             CERE_OPS_PREFILL | CERE_OPS_DECODE | CERE_OPS_CANCEL | CERE_OPS_SNAPSHOT | CERE_OPS_RESTORE);
    CHECK(toy(executor).allocated_host_bytes() >= on.desc.snapshot_host_bytes);  // the arena is pinned at ALLOCATE

    cere_session *a = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &a), CERE_OK);
    CHECK_EQ(cere_session_snapshot(executor, a, 0, &s), CERE_ERR_ARGUMENT);  // nothing fed
    CHECK(s == nullptr);
    CHECK(run(executor, a, CERE_OP_PREFILL, prompt, 4, generation, 0, CERE_WORK_NO_SPECULATION) ==
          std::vector<uint32_t>(expected.begin(), expected.begin() + 4));

    // Stale revision, then the snapshot and its info (position = fed = 6, the last committed token unfed).
    CHECK_EQ(cere_session_snapshot(executor, a, 0, &s), CERE_ERR_STALE_EPOCH);
    CHECK(s == nullptr);
    cere_snapshot *s1 = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, a, 1, &s1), CERE_OK);
    CHECK(s1 != nullptr);
    cere_snapshot_info info{};
    info.struct_size = 72 - 8;
    info.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(cere_snapshot_get_info(executor, s1, &info), CERE_ERR_ARGUMENT);
    info.struct_size = sizeof(info);
    CHECK_EQ(cere_snapshot_get_info(executor, s1, &info), CERE_OK);
    CHECK_EQ(info.position, 6);
    CHECK_EQ(info.token_count, 7);
    CHECK_EQ(info.fed, 6);
    CHECK_EQ(info.bytes, snapshot_size(6, 7));
    const auto id = snap::layout_id(cere::toy::fixture_config());
    CHECK(std::memcmp(info.layout_id, id.data(), 32) == 0);
    const std::vector<uint8_t> bytes1 = read_all(executor, s1);
    CHECK_EQ(get64(bytes1, kAtFlags), CERE_WORK_NO_SPECULATION);  // the PREFILL's flags travel with the state
    std::vector<uint8_t> part(8);
    CHECK_EQ(cere_snapshot_read(executor, s1, info.bytes - 8, part.data(), 8), CERE_OK);
    CHECK_EQ(cere_snapshot_read(executor, s1, info.bytes - 7, part.data(), 8), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_snapshot_read(executor, s1, 0, nullptr, 8), CERE_ERR_ARGUMENT);

    // Two snapshots of one idle session are byte-identical.
    cere_snapshot *s2 = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, a, 1, &s2), CERE_OK);
    CHECK(read_all(executor, s2) == bytes1);
    CHECK_EQ(cere_snapshot_release(executor, s2), CERE_OK);
    CHECK_EQ(cere_snapshot_release(executor, s2), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_snapshot_get_info(executor, s2, &info), CERE_ERR_ARGUMENT);

    // Busy: a ticket in flight (the session's bit 1 is fixed by its PREFILL, ledger F: a DECODE carries it too).
    Work busy(a, CERE_OP_DECODE, {}, 4, generation, {}, 1);
    busy.item.flags = CERE_WORK_NO_SPECULATION;
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &busy.batch, &ticket), CERE_OK);
    CHECK_EQ(cere_session_snapshot(executor, a, 1, &s), CERE_ERR_BUSY);
    CHECK(s == nullptr);
    CHECK_EQ(drain(executor, ticket).committed_tokens, 4);
    CHECK(std::equal(busy.out.begin(), busy.out.end(), expected.begin() + 4));

    // Restore into a fresh session, then DECODE = the uninterrupted run; a second restore forks the same state.
    for (int fork = 0; fork < 2; ++fork) {
        cere_session *b = nullptr;
        CHECK_EQ(cere_session_create(executor, model, 32, &b), CERE_OK);
        CHECK_EQ(cere_session_restore(executor, b, s1), CERE_OK);
        CHECK(run(executor, b, CERE_OP_DECODE, {}, 4, generation, 0, CERE_WORK_NO_SPECULATION) ==
              std::vector<uint32_t>(expected.begin() + 4, expected.begin() + 8));
        // The restored session keeps the PREFILL flags.
        cere_snapshot *again = nullptr;
        CHECK_EQ(cere_session_snapshot(executor, b, 1, &again), CERE_OK);
        CHECK_EQ(get64(read_all(executor, again), kAtFlags), CERE_WORK_NO_SPECULATION);
        CHECK_EQ(cere_snapshot_release(executor, again), CERE_OK);
        // Not fresh any more.
        CHECK_EQ(cere_session_restore(executor, b, s1), CERE_ERR_ARGUMENT);
        CHECK_EQ(cere_session_release(executor, b), CERE_OK);
    }
    CHECK_EQ(cere_session_restore(executor, a, s1), CERE_ERR_ARGUMENT);   // submitted before
    cere_session *other = nullptr;
    CHECK_EQ(cere_session_create(executor, plain, 32, &other), CERE_OK);
    CHECK_EQ(cere_session_restore(executor, other, s1), CERE_ERR_ARGUMENT);  // another model
    cere_session *small = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 4, &small), CERE_OK);
    CHECK_EQ(cere_session_restore(executor, small, s1), CERE_ERR_CAPACITY);  // context below position
    CHECK_EQ(cere_session_restore(executor, small, nullptr), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_session_release(executor, small), CERE_OK);

    // Read + import round trip byte-identical, and the import restores like the original.
    cere_snapshot *imported = nullptr;
    CHECK_EQ(cere_snapshot_import(executor, model, bytes1.data(), bytes1.size(), &imported), CERE_OK);
    CHECK(read_all(executor, imported) == bytes1);
    std::vector<uint8_t> broken = bytes1;
    broken[0] ^= 1;  // magic
    s = reinterpret_cast<cere_snapshot *>(uintptr_t(1));
    CHECK_EQ(cere_snapshot_import(executor, model, broken.data(), broken.size(), &s), CERE_ERR_ARGUMENT);
    CHECK(s == nullptr);
    cere_session *c = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &c), CERE_OK);
    CHECK_EQ(cere_session_restore(executor, c, imported), CERE_OK);
    CHECK(run(executor, c, CERE_OP_DECODE, {}, 2, generation, 0) ==
          std::vector<uint32_t>(expected.begin() + 4, expected.begin() + 6));

    // A snapshot after a cancelled DECODE restores to the uninterrupted continuation.
    Work cancelled(c, CERE_OP_DECODE, {}, 10, generation, {}, 1);
    CHECK_EQ(cere_executor_submit(executor, &cancelled.batch, &ticket), CERE_OK);
    cere_completion done = completion_header();
    CHECK_EQ(cere_executor_poll(executor, ticket, &done), CERE_PENDING);
    CHECK_EQ(cere_executor_cancel(executor, ticket), CERE_OK);
    done = drain(executor, ticket);
    const uint64_t kept = 2 + done.committed_tokens;  // tokens generated by c so far
    cere_snapshot *after_cancel = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, c, 2, &after_cancel), CERE_OK);
    CHECK_EQ(cere_session_release(executor, a), CERE_OK);  // max_sessions 2
    cere_session *d2 = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &d2), CERE_OK);
    CHECK_EQ(cere_session_restore(executor, d2, after_cancel), CERE_OK);
    CHECK(run(executor, d2, CERE_OP_DECODE, {}, 2, generation, 0) ==
          std::vector<uint32_t>(expected.begin() + 4 + kept, expected.begin() + 6 + kept));

    // Model release is BUSY while snapshots exist (sessions released first).
    for (cere_session *x : {c, d2}) CHECK_EQ(cere_session_release(executor, x), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_ERR_BUSY);
    for (cere_snapshot *x : {s1, imported, after_cancel}) CHECK_EQ(cere_snapshot_release(executor, x), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);

    // Arena full: CAPACITY with nothing kept; a release makes room again.
    Desc tight = toy_desc();
    tight.desc.snapshot_host_bytes = (snapshot_size(6, 7) + 255) / 256 * 256;
    cere_model *tm = sealed_model(executor, tight);
    const uint64_t tg = generation_of(executor, tm);
    cere_session *t1 = nullptr;
    CHECK_EQ(cere_session_create(executor, tm, 32, &t1), CERE_OK);
    run(executor, t1, CERE_OP_PREFILL, prompt, 4, tg, 0);
    cere_snapshot *first = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, t1, 1, &first), CERE_OK);
    s = reinterpret_cast<cere_snapshot *>(uintptr_t(1));
    CHECK_EQ(cere_session_snapshot(executor, t1, 1, &s), CERE_ERR_CAPACITY);
    CHECK(s == nullptr);
    CHECK_EQ(cere_snapshot_import(executor, tm, bytes1.data(), bytes1.size(), &s), CERE_ERR_CAPACITY);
    CHECK(s == nullptr);
    std::vector<uint8_t> unflagged = bytes1;  // t1's PREFILL carried no flags
    put64(unflagged, kAtFlags, 0);
    CHECK(read_all(executor, first) == unflagged);  // the kept snapshot is untouched
    CHECK_EQ(cere_snapshot_release(executor, first), CERE_OK);
    CHECK_EQ(cere_session_snapshot(executor, t1, 1, &first), CERE_OK);
    CHECK_EQ(cere_snapshot_release(executor, first), CERE_OK);
    CHECK_EQ(cere_session_release(executor, t1), CERE_OK);
    CHECK_EQ(cere_model_release(executor, tm), CERE_OK);
    CHECK_EQ(cere_session_release(executor, ps), CERE_OK);
    CHECK_EQ(cere_session_release(executor, other), CERE_OK);
    CHECK_EQ(cere_model_release(executor, plain), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// ---- Ledger CR13 (M2.7 x M2.11, m27 design note I12/D5): snapshot kind 8 carries the session's out_counts ----

// Kind 8 of a snapshot image: the u32 [vocab] counts.
std::vector<uint32_t> image_counts(const std::vector<uint8_t> &image, uint32_t vocab) {
    CHECK_EQ(get32(image, kAtSectionCount), 9);
    CHECK_EQ(get32(image, entry_at(8)), snap::kSampler);
    const uint64_t at = get64(image, entry_at(8) + 8);
    CHECK_EQ(get64(image, entry_at(8) + 16), 4 * uint64_t(vocab));
    std::vector<uint32_t> counts(vocab);
    for (uint32_t i = 0; i < vocab; ++i) counts[i] = get32(image, at + 4 * uint64_t(i));
    return counts;
}

void test_snapshot_sampler_counts() {
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    // The PREFILL commits c1 = next_token(1204, 9) = 37,333, and the toy's next row has its top logit (8) at 37,333
    // again (30 c1 + 10 = 0 mod 70,000): frequency penalty 2 on the counted c1 makes 37,334 (7) win over it (6), so
    // a restore that dropped the counts would commit 37,333.
    const std::vector<uint32_t> prompt = {3, 1, 4, 1, 5, 9, 2, 6, 1204};
    cere_sampling_params s = sampling_params(0.0f, vocab);
    s.frequency_penalty = 2.0f;
    const std::vector<uint32_t> expected = expected_sampled(prompt, 4, vocab, s);
    CHECK_EQ(expected[0], 37333);
    CHECK_EQ(expected[1], 37334);
    std::vector<uint32_t> fed = prompt;
    fed.push_back(expected[0]);
    CHECK_EQ(expected_sampled(fed, 1, vocab, s)[0], 37333);  // the same step with the counts lost

    cere_executor *executor = make_executor();
    Desc on = toy_desc();
    on.desc.snapshot_host_bytes = uint64_t(1) << 22;
    cere_model *model = sealed_model(executor, on);
    const uint64_t generation = generation_of(executor, model);
    const auto item = [&](cere_session *session, uint32_t op, std::vector<uint32_t> tokens, uint64_t n,
                          uint64_t revision) {
        Work w(session, op, std::move(tokens), n, generation, {}, revision);
        w.item.sampling = &s;
        cere_completion_ticket ticket{};
        CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
        const cere_completion c = drain(executor, ticket);
        CHECK_EQ(c.status, CERE_OK);
        CHECK_EQ(c.committed_tokens, n);
        return w.out;
    };
    cere_session *a = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &a), CERE_OK);
    CHECK(item(a, CERE_OP_PREFILL, prompt, 1, 0) == std::vector<uint32_t>{expected[0]});
    cere_snapshot *snapshot = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, a, 1, &snapshot), CERE_OK);
    const std::vector<uint8_t> image = read_all(executor, snapshot);
    CHECK_EQ(image.size(), snapshot_size(9, 10));
    std::vector<uint32_t> want(vocab, 0);
    want[expected[0]] = 1;
    CHECK(image_counts(image, vocab) == want);

    // Restored into a fresh session, the counts penalise c1 as in the uninterrupted run; both end in the same state.
    cere_session *b = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &b), CERE_OK);
    CHECK_EQ(cere_session_restore(executor, b, snapshot), CERE_OK);
    cere_snapshot *again = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, b, 0, &again), CERE_OK);
    CHECK(read_all(executor, again) == image);
    CHECK_EQ(cere_snapshot_release(executor, again), CERE_OK);
    const std::vector<uint32_t> rest(expected.begin() + 1, expected.end());
    CHECK(item(b, CERE_OP_DECODE, {}, 3, 0) == rest);
    CHECK(item(a, CERE_OP_DECODE, {}, 3, 1) == rest);
    cere_snapshot *sa = nullptr;
    cere_snapshot *sb = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, a, 2, &sa), CERE_OK);
    CHECK_EQ(cere_session_snapshot(executor, b, 1, &sb), CERE_OK);
    const std::vector<uint8_t> after = read_all(executor, sa);
    CHECK(read_all(executor, sb) == after);
    for (uint32_t id : expected) ++want[id];
    want[expected[0]] -= 1;  // counted once above already
    CHECK(image_counts(after, vocab) == want);

    // A greedy session's kind 8 is all zero.
    CHECK_EQ(cere_session_release(executor, a), CERE_OK);  // max_sessions 2
    cere_session *g = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &g), CERE_OK);
    run(executor, g, CERE_OP_PREFILL, prompt, 2, generation, 0);
    cere_snapshot *sg = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, g, 1, &sg), CERE_OK);
    CHECK(image_counts(read_all(executor, sg), vocab) == std::vector<uint32_t>(vocab, 0));
    for (cere_session *x : {b, g}) CHECK_EQ(cere_session_release(executor, x), CERE_OK);
    for (cere_snapshot *x : {snapshot, sa, sb, sg}) CHECK_EQ(cere_snapshot_release(executor, x), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// ---- C3b (ABI half of D14): every rejection through cere_snapshot_import and cere_session_restore leaves the
// session and the arena unchanged (m211 design note C3, D14) ----
void test_snapshot_api_rejections() {
    const cere_qwen4_exp_config &cfg = cere::toy::fixture_config();
    const uint32_t vocab = cfg.vocab_size;
    cere_executor *executor = make_executor();
    Desc on = toy_desc();
    on.desc.snapshot_host_bytes = uint64_t(1) << 20;
    cere_model *model = sealed_model(executor, on);
    const uint64_t generation = generation_of(executor, model);
    cere_session *a = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &a), CERE_OK);
    run(executor, a, CERE_OP_PREFILL, {5, 9, 69999}, 4, generation, 0);
    cere_snapshot *s1 = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, a, 1, &s1), CERE_OK);
    const std::vector<uint8_t> good = read_all(executor, s1);  // position 6, token_count 7, fed 6
    uint8_t *stored = model->device->arena + executor->snapshots.at(s1)->offset;
    const uint64_t used = model->arena.used();
    const uint64_t count = model->snapshots;

    cere_session *fresh = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &fresh), CERE_OK);
    const auto unchanged = [&](const char *what) {
        if (!(fresh->tokens.empty() && fresh->fed == 0 && fresh->flags == 0 && !fresh->submitted &&
              model->arena.used() == used && model->snapshots == count &&
              std::memcmp(stored, good.data(), good.size()) == 0)) {
            std::fprintf(stderr, "C3b: %s changed the session or the arena\n", what);
            CHECK(false);
        }
    };
    // Same-length images go through both paths: import, and restore of the stored snapshot rewritten in place.
    const auto both = [&](const char *what, const std::vector<uint8_t> &b, cere_status want) {
        cere_snapshot *out = reinterpret_cast<cere_snapshot *>(uintptr_t(1));
        CHECK_EQ(cere_snapshot_import(executor, model, b.data(), b.size(), &out), want);
        CHECK(out == nullptr);
        unchanged(what);
        if (b.size() != good.size()) return;
        std::memcpy(stored, b.data(), b.size());
        CHECK_EQ(cere_session_restore(executor, fresh, s1), want);
        std::memcpy(stored, good.data(), good.size());
        unchanged(what);
    };
    const auto mutated = [&](const char *what, const std::function<void(std::vector<uint8_t> &)> &f,
                             cere_status want) {
        std::vector<uint8_t> b = good;
        f(b);
        both(what, b, want);
    };
    const auto at = [&](uint32_t i, uint64_t field) { return get64(good, entry_at(i) + field); };
    const uint64_t tokens_at = at(0, 8);

    // Handles: unknown and released.
    cere_snapshot *bogus = reinterpret_cast<cere_snapshot *>(uintptr_t(8));
    CHECK_EQ(cere_session_restore(executor, fresh, bogus), CERE_ERR_ARGUMENT);
    unchanged("unknown handle");
    cere_snapshot *gone = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, a, 1, &gone), CERE_OK);
    CHECK_EQ(cere_snapshot_release(executor, gone), CERE_OK);
    CHECK_EQ(cere_session_restore(executor, fresh, gone), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_snapshot_release(executor, gone), CERE_ERR_ARGUMENT);
    unchanged("released handle");
    // len short, long; total_bytes != len.
    both("len short", std::vector<uint8_t>(good.begin(), good.end() - 1), CERE_ERR_ARGUMENT);
    std::vector<uint8_t> longer = good;
    longer.push_back(0);
    both("len long", longer, CERE_ERR_ARGUMENT);
    both("len 127", std::vector<uint8_t>(good.begin(), good.begin() + 127), CERE_ERR_ARGUMENT);
    mutated("total_bytes", [](std::vector<uint8_t> &b) { put64(b, kAtTotal, b.size() - 1); }, CERE_ERR_ARGUMENT);
    // Magic, header_bytes, reserved and pad bytes.
    mutated("magic", [](std::vector<uint8_t> &b) { b[7] = '2'; }, CERE_ERR_ARGUMENT);
    mutated("header_bytes", [](std::vector<uint8_t> &b) { put32(b, kAtHeaderBytes, 256); }, CERE_ERR_ARGUMENT);
    mutated("header reserved", [](std::vector<uint8_t> &b) { b[kAtReserved + 3] = 1; }, CERE_ERR_ARGUMENT);
    mutated("header reserved[24]", [](std::vector<uint8_t> &b) { b[kAtReserved24] = 1; }, CERE_ERR_ARGUMENT);
    mutated("entry reserved", [](std::vector<uint8_t> &b) { b[entry_at(5) + 7] = 1; }, CERE_ERR_ARGUMENT);
    mutated("table pad", [](std::vector<uint8_t> &b) { b[entry_at(9)] = 1; }, CERE_ERR_ARGUMENT);
    mutated("section pad", [&](std::vector<uint8_t> &b) { b[tokens_at + at(0, 16)] = 1; }, CERE_ERR_ARGUMENT);
    // format 2, layout_id: UNSUPPORTED.
    mutated("format", [](std::vector<uint8_t> &b) { put32(b, kAtFormat, 2); }, CERE_ERR_UNSUPPORTED);
    mutated("layout_id", [](std::vector<uint8_t> &b) { b[kAtLayout + 5] ^= 0x10; }, CERE_ERR_UNSUPPORTED);
    // Section table.
    mutated("section_count 17", [](std::vector<uint8_t> &b) { put32(b, kAtSectionCount, 17); }, CERE_ERR_ARGUMENT);
    mutated("u64 wrap", [&](std::vector<uint8_t> &b) { put64(b, entry_at(8) + 16, UINT64_MAX - at(8, 8) + 1); },
            CERE_ERR_ARGUMENT);
    mutated("out of bounds", [&](std::vector<uint8_t> &b) { put64(b, entry_at(8) + 16, at(8, 16) + 256); },
            CERE_ERR_ARGUMENT);
    mutated("misaligned", [&](std::vector<uint8_t> &b) { put64(b, entry_at(1) + 8, at(1, 8) + 4); },
            CERE_ERR_ARGUMENT);
    // The toy's non-empty sections are kinds 0 (tokens), 7 (PLE) and 8 (sampler counts); kinds 1-6 have 0 bytes in
    // its 1-layer config.
    mutated("overlapping", [&](std::vector<uint8_t> &b) { put64(b, entry_at(7) + 8, at(0, 8)); }, CERE_ERR_ARGUMENT);
    mutated("duplicate kind", [](std::vector<uint8_t> &b) { put32(b, entry_at(2), 1); }, CERE_ERR_ARGUMENT);
    // Structure (rebuilt images of the toy's config, same length where the change allows it).
    const auto image = [&](uint64_t position, uint64_t token_count, uint64_t fed, uint64_t first_row = 0,
                           uint32_t kinds = snap::kRequiredKinds) {
        return build(fields(position, token_count, fed, first_row, cfg),
                     sizes(position, token_count, first_row, kinds, cfg));
    };
    auto extra = [&](uint32_t kind, uint64_t bytes) {
        std::vector<snap::Section> s = sizes(6, 7, 0, snap::kRequiredKinds, cfg);
        snap::Section e;
        e.kind = kind;
        e.bytes = bytes;
        s.push_back(e);
        return build(fields(6, 7, 6, 0, cfg), s);
    };
    both("kind 9", extra(9, 0), CERE_ERR_ARGUMENT);
    both("kind 12", extra(12, 256), CERE_ERR_ARGUMENT);
    both("kind 10 without M2.9", extra(10, 0), CERE_ERR_UNSUPPORTED);
    both("kind 11 on a model without vision", extra(11, 8), CERE_ERR_UNSUPPORTED);
    both("missing kind 5", image(6, 7, 6, 0, snap::kRequiredKinds & ~(1u << 5)), CERE_ERR_ARGUMENT);
    both("missing kind 8", image(6, 7, 6, 0, snap::kRequiredKinds & ~(1u << 8)), CERE_ERR_ARGUMENT);
    std::vector<snap::Section> off = sizes(6, 7, 0, snap::kRequiredKinds, cfg);
    off[0].bytes -= 1;
    both("size off by one", build(fields(6, 7, 6, 0, cfg), off), CERE_ERR_ARGUMENT);
    both("first_row 1", image(6, 7, 6, 1), CERE_ERR_ARGUMENT);
    both("position 0", image(0, 0, 0), CERE_ERR_CAPACITY);
    both("position above the model's context", image(65, 65, 65), CERE_ERR_CAPACITY);
    both("fed != position", image(6, 7, 5), CERE_ERR_ARGUMENT);
    both("token_count = fed + 2", image(6, 8, 6), CERE_ERR_ARGUMENT);
    mutated("token id = vocab", [&](std::vector<uint8_t> &b) { put32(b, tokens_at + 4 * 6, vocab); },
            CERE_ERR_ARGUMENT);
    mutated("unknown flag bit", [](std::vector<uint8_t> &b) { put64(b, kAtFlags, 0x100); }, CERE_ERR_ARGUMENT);
    // Above the session's context (restore only; import has no session).
    cere_session *small = nullptr;
    CHECK_EQ(cere_session_release(executor, a), CERE_OK);  // max_sessions 2
    CHECK_EQ(cere_session_create(executor, model, 5, &small), CERE_OK);
    CHECK_EQ(cere_session_restore(executor, small, s1), CERE_ERR_CAPACITY);
    CHECK(small->tokens.empty() && small->fed == 0 && !small->submitted);
    unchanged("session context");
    CHECK_EQ(cere_session_release(executor, small), CERE_OK);

    // Controls: the untouched image imports and restores.
    cere_snapshot *copy = nullptr;
    CHECK_EQ(cere_snapshot_import(executor, model, good.data(), good.size(), &copy), CERE_OK);
    CHECK(read_all(executor, copy) == good);
    CHECK_EQ(cere_session_restore(executor, fresh, s1), CERE_OK);
    CHECK_EQ(fresh->fed, 6);
    for (cere_snapshot *x : {s1, copy}) CHECK_EQ(cere_snapshot_release(executor, x), CERE_OK);
    CHECK_EQ(cere_session_release(executor, fresh), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// M211R-07 (review-m211 design note): a session whose prompt is still open (a chunked PREFILL cancelled between chunks,
// fed < prompt_tokens) has no state to save: cere_session_snapshot is CERE_ERR_ARGUMENT before the arena is touched,
// as CudaDevice::save_state refuses it.
void test_snapshot_refuses_open_prompt() {
    cere_executor *executor = make_executor("toy,prefill_chunk=4");
    Desc d = rows_desc();
    d.desc.snapshot_host_bytes = uint64_t(1) << 20;
    cere_model *model = sealed_model(executor, d);
    cere_session *s = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &s), CERE_OK);
    Job p = submit_job(executor, model, s, CERE_OP_PREFILL, iota_prompt(100, 14), 2);
    poll_pending(executor, p.ticket);  // chunk 1
    CHECK_EQ(s->fed, 4);
    CHECK_EQ(cere_executor_cancel(executor, p.ticket), CERE_OK);
    const cere_completion c = poll_terminal(executor, p.ticket, CERE_FINISH_CANCELLED);
    CHECK_EQ(s->fed, 4);
    CHECK(s->fed < s->prompt_tokens);
    const uint64_t used = model->arena.used();
    const uint64_t count = model->snapshots;
    cere_snapshot *out = reinterpret_cast<cere_snapshot *>(uintptr_t(1));
    CHECK_EQ(cere_session_snapshot(executor, s, c.session_revision, &out), CERE_ERR_ARGUMENT);
    CHECK(out == nullptr);
    CHECK_EQ(model->arena.used(), used);
    CHECK_EQ(model->snapshots, count);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// --- M2.8 vocab masks (m28 design note C6, A1-A3, A6, D9; ledger CR5, CR7) --------------------------------------

// ceil(vocab / 32) words of the toy fixture's vocab (A1).
uint64_t mask_words() { return (uint64_t(cere::toy::fixture_config().vocab_size) + 31) / 32; }

// A mask allowing exactly `ids` (xgrammar layout: word id >> 5, bit id & 31).
std::vector<uint32_t> mask_of_ids(std::initializer_list<uint32_t> ids) {
    std::vector<uint32_t> mask(mask_words(), 0);
    for (const uint32_t id : ids) mask[id >> 5] |= uint32_t(1) << (id & 31);
    return mask;
}

bool mask_allows(const std::vector<uint32_t> &mask, uint32_t id) { return (mask[id >> 5] >> (id & 31)) & 1u; }

// A constrained work item: `sampling` (default: greedy with processors, neutral), NO_SPECULATION (table F bit 1, set
// by cered on every constrained ticket) and `mask` as the mask of position 0.
struct Constrained {
    cere_sampling_params sampling;
    std::vector<uint32_t> mask;
    Work work;

    Constrained(cere_session *session, uint32_t op, std::vector<uint32_t> prompt, uint64_t max_new,
                uint64_t generation, std::vector<uint32_t> initial,
                const cere_sampling_params &params = sampling_params(0.0f, cere::toy::fixture_config().vocab_size))
        : sampling(params),
          mask(std::move(initial)),
          work(session, op, std::move(prompt), max_new, generation, {}, session->revision) {
        work.item.flags = CERE_WORK_NO_SPECULATION;
        work.item.sampling = &sampling;
        work.item.vocab_mask = mask.data();
        work.item.vocab_mask_words = mask.size();
    }
};

cere_status constrain(cere_executor *executor, const cere_completion_ticket &ticket, uint64_t position,
                      const std::vector<uint32_t> &mask) {
    return cere_executor_constrain(executor, ticket, position, mask.data(), mask.size());
}

// C6 / A1: the submit rules of cere_work.vocab_mask / vocab_mask_words; every refusal leaves the session idle and
// hands out no ticket. CR7: VERIFY on a constrained item is ARGUMENT (not the UNSUPPORTED of an unconstrained one).
// A3: capabilities native_token_mask is 1 with the sampler; a ",sampler=0" toy reports 0 and refuses the item.
void test_vocab_mask_submit() {
    cere_executor *executor = make_executor();
    Desc d = toy_desc();
    cere_model *model = sealed_model(executor, d);
    const uint64_t generation = generation_of(executor, model);
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &session), CERE_OK);
    const auto refused = [&](Constrained &c, cere_status want) {
        cere_completion_ticket ticket{};
        CHECK_EQ(cere_executor_submit(executor, &c.work.batch, &ticket), want);
        CHECK_EQ(ticket.submission_id, 0);
        CHECK_EQ(session->in_flight, 0);
    };
    {
        Constrained c(session, CERE_OP_PREFILL, {1, 2, 3}, 1, generation, mask_of_ids({5}));
        c.work.item.vocab_mask = nullptr;  // null mask, non-zero words
        refused(c, CERE_ERR_ARGUMENT);
        c.work.item.vocab_mask = c.mask.data();
        c.work.item.vocab_mask_words = 0;  // a mask without words
        refused(c, CERE_ERR_ARGUMENT);
        for (const uint64_t words : {mask_words() - 1, mask_words() + 1, uint64_t(7760)}) {
            c.work.item.vocab_mask_words = words;  // anything but ceil(vocab / 32) (7,760 is the A4 model's)
            refused(c, CERE_ERR_ARGUMENT);
        }
        c.work.item.vocab_mask_words = c.mask.size();
        c.work.item.sampling = nullptr;  // a mask needs `sampling`
        refused(c, CERE_ERR_ARGUMENT);
        c.work.item.sampling = &c.sampling;
        c.work.item.operation = CERE_OP_VERIFY;  // ledger CR7
        refused(c, CERE_ERR_ARGUMENT);
        c.work.item.operation = CERE_OP_PREFILL;
    }
    {
        Constrained zero(session, CERE_OP_PREFILL, {1, 2, 3}, 1, generation, mask_of_ids({}));
        refused(zero, CERE_ERR_ARGUMENT);  // all-zero initial mask
        // Only padding bits past the vocab (70,000 = 2,187.5 words): no allowed id, so the same refusal.
        zero.mask.back() = 0xffff0000u;
        CHECK_EQ(vocab % 32, 16);
        refused(zero, CERE_ERR_ARGUMENT);
    }
    {
        Constrained ok(session, CERE_OP_PREFILL, {1, 2, 3}, 1, generation, mask_of_ids({vocab - 1}));
        cere_completion_ticket ticket{};
        CHECK_EQ(cere_executor_submit(executor, &ok.work.batch, &ticket), CERE_OK);
        std::fill(ok.mask.begin(), ok.mask.end(), 0u);  // copied at submit
        const cere_completion c = drain(executor, ticket);
        CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
        CHECK_EQ(ok.work.out[0], vocab - 1);  // the toy's 96 is masked: the only allowed id above it
    }
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    cere_executor *plain = make_executor("toy,sampler=0");
    Desc pd = toy_desc();
    cere_model *pm = sealed_model(plain, pd);
    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(cere_executor_capabilities(plain, pm, &caps), CERE_OK);
    CHECK_EQ(caps.native_sampler, 0);
    CHECK_EQ(caps.native_token_mask, 0);
    cere_session *ps = nullptr;
    CHECK_EQ(cere_session_create(plain, pm, 32, &ps), CERE_OK);
    Constrained c(ps, CERE_OP_PREFILL, {1, 2, 3}, 1, caps.profile_generation, mask_of_ids({5}));
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(plain, &c.work.batch, &ticket), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_session_release(plain, ps), CERE_OK);
    CHECK_EQ(cere_model_release(plain, pm), CERE_OK);
    CHECK_EQ(cere_executor_destroy(plain), CERE_OK);
}

// C6 / A2 / D9: a constrained ticket polled without the mask of `committed` is CERE_PENDING with committed unchanged
// and no step (no record= line); cere_executor_constrain supplies it (only for position == committed, a valid mask, a
// constrained live ticket; STALE_EPOCH / unknown ticket as poll). The toy's masked rule: a masked `next` becomes the
// smallest allowed id above it, wrapping; an allowed `next` stays. Each step's record line carries its mask.
void test_vocab_mask_wait_and_constrain() {
    const std::string path = "/tmp/cere-toy-vocab-mask-" + std::to_string(uint64_t(::getpid())) + ".txt";
    std::remove(path.c_str());
    const std::string id = "toy,record=" + path;
    cere_executor *executor = make_executor(id.c_str());
    Desc d = toy_desc();
    cere_model *model = sealed_model(executor, d);
    const uint64_t generation = generation_of(executor, model);
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &session), CERE_OK);
    // The toy's own tokens after {1, 2, 3}: 96 (position 3); D9 picks 100 (the smallest allowed above 96, not 50).
    CHECK_EQ(toy_sequence({1, 2, 3}, 1, vocab)[0], 96);
    Constrained c(session, CERE_OP_PREFILL, {1, 2, 3}, 3, generation, mask_of_ids({50, 100, 150}));
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &c.work.batch, &ticket), CERE_OK);
    cere_completion out = completion_header();
    CHECK_EQ(cere_executor_poll(executor, ticket, &out), CERE_PENDING);
    CHECK_EQ(out.committed_tokens, 1);
    CHECK_EQ(c.work.out[0], 100);
    // No mask for position 1: pending, nothing committed, nothing stepped, however often it is polled.
    for (int i = 0; i < 3; ++i) {
        out = completion_header();
        CHECK_EQ(cere_executor_poll(executor, ticket, &out), CERE_PENDING);
        CHECK_EQ(out.status, CERE_PENDING);
        CHECK_EQ(out.finish, CERE_FINISH_NONE);
        CHECK_EQ(out.committed_tokens, 1);
    }
    CHECK_EQ(session->tokens.size(), 4);
    CHECK_EQ(lines_of(path).size(), 1);

    const std::vector<uint32_t> next = mask_of_ids({100, 200});
    CHECK_EQ(constrain(executor, ticket, 0, next), CERE_ERR_ARGUMENT);  // position != committed
    CHECK_EQ(constrain(executor, ticket, 2, next), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_constrain(executor, ticket, 1, next.data(), next.size() - 1), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_constrain(executor, ticket, 1, next.data(), 0), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_constrain(executor, ticket, 1, nullptr, next.size()), CERE_ERR_ARGUMENT);
    CHECK_EQ(constrain(executor, ticket, 1, mask_of_ids({})), CERE_ERR_ARGUMENT);  // all-zero
    CHECK_EQ(cere_executor_constrain(nullptr, ticket, 1, next.data(), next.size()), CERE_ERR_ARGUMENT);
    cere_completion_ticket stale = ticket;
    stale.worker_epoch = kEpoch + 1;
    CHECK_EQ(constrain(executor, stale, 1, next), CERE_ERR_STALE_EPOCH);
    stale = ticket;
    stale.model_generation = ticket.model_generation + 1;
    CHECK_EQ(constrain(executor, stale, 1, next), CERE_ERR_STALE_EPOCH);
    stale = ticket;
    stale.submission_id = ticket.submission_id + 100;
    CHECK_EQ(constrain(executor, stale, 1, next), CERE_ERR_ARGUMENT);  // unknown ticket
    poll_pending(executor, ticket);                                     // the refusals set nothing
    CHECK_EQ(committed_of(executor, ticket), 1);

    // Position 1: the toy's 31 * 100 + 4 = 3,104 is above every allowed id, so the pick wraps to 100.
    std::vector<uint32_t> copied = next;
    CHECK_EQ(constrain(executor, ticket, 1, copied), CERE_OK);
    std::fill(copied.begin(), copied.end(), 0u);  // copied before return
    poll_pending(executor, ticket);
    CHECK_EQ(committed_of(executor, ticket), 2);
    CHECK_EQ(c.work.out[1], 100);
    poll_pending(executor, ticket);  // and the mask was consumed by that step
    CHECK_EQ(committed_of(executor, ticket), 2);
    // Position 2: the toy's own 31 * 100 + 5 = 3,105 is allowed and stays.
    CHECK_EQ(constrain(executor, ticket, 2, mask_of_ids({3105, 3106})), CERE_OK);
    out = completion_header();
    CHECK_EQ(cere_executor_poll(executor, ticket, &out), CERE_OK);
    CHECK_EQ(out.finish, CERE_FINISH_LENGTH);
    CHECK_EQ(out.committed_tokens, 3);
    CHECK_EQ(c.work.out[2], 3105);
    CHECK_EQ(constrain(executor, ticket, 3, next), CERE_ERR_ARGUMENT);  // released by the terminal poll
    const std::vector<std::string> lines = lines_of(path);
    CHECK_EQ(lines.size(), 3);
    const std::vector<std::string> tails = {" mask=3 token=100", " mask=2 token=100", " mask=2 token=3105"};
    for (size_t i = 0; i < lines.size() && i < tails.size(); ++i) {
        CHECK(lines[i].size() > tails[i].size() &&
              lines[i].compare(lines[i].size() - tails[i].size(), tails[i].size(), tails[i]) == 0);
    }

    // An unconstrained ticket takes no mask, and its record lines carry none.
    cere_session *plain = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &plain), CERE_OK);
    Work w(plain, CERE_OP_PREFILL, {1, 2, 3}, 2, generation);
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
    CHECK_EQ(constrain(executor, ticket, 0, next), CERE_ERR_ARGUMENT);
    CHECK_EQ(drain(executor, ticket).committed_tokens, 2);
    CHECK(w.out == toy_sequence({1, 2, 3}, 2, vocab));
    for (const std::string &line : lines_of(path)) {
        if (line.find("session=2 ") == 0) CHECK(line.find("mask=") == std::string::npos);
    }
    CHECK_EQ(cere_session_release(executor, plain), CERE_OK);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    std::remove(path.c_str());
}

// C6 / D9 order: cancel comes before the mask wait (a ticket waiting for its mask cancels on its next poll).
void test_vocab_mask_cancel() {
    cere_executor *executor = make_executor();
    Desc d = toy_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 32, &session), CERE_OK);
    Constrained c(session, CERE_OP_PREFILL, {1, 2, 3}, 4, generation_of(executor, model), mask_of_ids({7}));
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &c.work.batch, &ticket), CERE_OK);
    poll_pending(executor, ticket);
    CHECK_EQ(c.work.out[0], 7);  // 96 is masked: the pick wraps to 7
    poll_pending(executor, ticket);
    CHECK_EQ(cere_executor_cancel(executor, ticket), CERE_OK);
    const cere_completion out = poll_terminal(executor, ticket, CERE_FINISH_CANCELLED);
    CHECK_EQ(out.committed_tokens, 1);
    CHECK_EQ(session->in_flight, 0);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// C5 on the toy: a sampled constrained item draws only allowed ids (here two of the toy's eight-id support), and its
// logprobs are post-mask (the two allowed ids carry the whole probability).
void test_vocab_mask_sampled() {
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    std::set<uint32_t> seen;
    for (uint64_t seed = 1; seed <= 40; ++seed) {
        cere_executor *executor = make_executor();
        Desc d = toy_desc();
        cere_model *model = sealed_model(executor, d);
        cere_session *session = nullptr;
        CHECK_EQ(cere_session_create(executor, model, 32, &session), CERE_OK);
        Constrained c(session, CERE_OP_PREFILL, {1, 2, 3}, 1, generation_of(executor, model), mask_of_ids({96, 98}),
                      sampling_params(2.0f, vocab, seed));
        std::vector<cere_token_logprobs> logprobs(1);
        c.work.item.out_logprobs = logprobs.data();
        c.work.item.top_logprobs = 3;
        cere_completion_ticket ticket{};
        CHECK_EQ(cere_executor_submit(executor, &c.work.batch, &ticket), CERE_OK);
        CHECK_EQ(drain(executor, ticket).committed_tokens, 1);
        const uint32_t token = c.work.out[0];
        CHECK(mask_allows(c.mask, token));
        seen.insert(token);
        CHECK_EQ(logprobs[0].id, token);
        CHECK_EQ(logprobs[0].top_count, 3);
        CHECK_EQ(logprobs[0].top[0].id, 96);
        CHECK_EQ(logprobs[0].top[1].id, 98);
        CHECK(std::isinf(logprobs[0].top[2].logprob) && logprobs[0].top[2].logprob < 0.0f);
        CHECK(std::fabs(std::exp(double(logprobs[0].top[0].logprob)) + std::exp(double(logprobs[0].top[1].logprob)) -
                        1.0) < 1e-5);
        CHECK_EQ(cere_session_release(executor, session), CERE_OK);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
    CHECK(seen == (std::set<uint32_t>{96, 98}));
}

// C6b (ledger CR5, M2.6 rounds): of two constrained DECODE tickets, one awaiting its mask is not stepped by the
// other's poll (that round has one row); with both masks present one round steps both rows, and each row commits a
// token of its own mask (each row's ItemSampling carries its own mask).
void test_rows_vocab_mask() {
    const std::string path = "/tmp/cere-toy-rows-mask-" + std::to_string(uint64_t(::getpid())) + ".txt";
    std::remove(path.c_str());
    const std::string id = "toy,record=" + path;
    cere_executor *executor = make_executor(id.c_str());
    Desc d = rows_desc();
    cere_model *model = sealed_model(executor, d);
    cere_session *a = prefilled_session(executor, model, {1, 2});
    cere_session *b = prefilled_session(executor, model, {3, 4});
    std::remove(path.c_str());  // the record of the two prefills
    const uint64_t generation = generation_of(executor, model);
    Constrained ca(a, CERE_OP_DECODE, {}, 3, generation, mask_of_ids({11}));
    Constrained cb(b, CERE_OP_DECODE, {}, 3, generation, mask_of_ids({22}));
    cere_completion_ticket ta{};
    cere_completion_ticket tb{};
    CHECK_EQ(cere_executor_submit(executor, &ca.work.batch, &ta), CERE_OK);
    CHECK_EQ(cere_executor_submit(executor, &cb.work.batch, &tb), CERE_OK);
    // Both masks present: one round, two rows, each token from its own mask.
    poll_pending(executor, ta);
    CHECK_EQ(committed_of(executor, ta), 1);
    CHECK_EQ(committed_of(executor, tb), 1);
    CHECK_EQ(ca.work.out[0], 11);
    CHECK_EQ(cb.work.out[0], 22);
    CHECK_EQ(lines_of(path).size(), 2);
    // Only b has its mask: a's poll runs a round of b alone; a waits, unstepped.
    CHECK_EQ(constrain(executor, tb, 1, mask_of_ids({33})), CERE_OK);
    poll_pending(executor, ta);
    CHECK_EQ(committed_of(executor, ta), 1);
    CHECK_EQ(committed_of(executor, tb), 2);
    CHECK_EQ(cb.work.out[1], 33);
    poll_pending(executor, tb);  // neither has a mask now: no round
    CHECK_EQ(committed_of(executor, ta), 1);
    CHECK_EQ(committed_of(executor, tb), 2);
    std::vector<std::string> lines = lines_of(path);
    CHECK_EQ(lines.size(), 3);
    if (lines.size() == 3) CHECK(lines[2].find("session=2 ") == 0);
    // Both masks again: one round of both rows.
    CHECK_EQ(constrain(executor, ta, 1, mask_of_ids({44})), CERE_OK);
    CHECK_EQ(constrain(executor, tb, 2, mask_of_ids({55})), CERE_OK);
    poll_pending(executor, ta);
    CHECK_EQ(committed_of(executor, ta), 2);
    CHECK_EQ(committed_of(executor, tb), 3);
    CHECK_EQ(ca.work.out[1], 44);
    CHECK_EQ(cb.work.out[2], 55);
    CHECK_EQ(lines_of(path).size(), 5);
    CHECK_EQ(poll_terminal(executor, tb, CERE_FINISH_LENGTH).committed_tokens, 3);
    CHECK_EQ(constrain(executor, ta, 2, mask_of_ids({66})), CERE_OK);
    CHECK_EQ(poll_terminal(executor, ta, CERE_FINISH_LENGTH).committed_tokens, 3);
    CHECK_EQ(ca.work.out[2], 66);
    CHECK_EQ(cere_session_release(executor, a), CERE_OK);
    CHECK_EQ(cere_session_release(executor, b), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    std::remove(path.c_str());
}

// m213 design note C14 (CR14; ledger "Snapshot section kinds": kind 11 `rope_delta` i64, writer rule): an image
// session's rotary state travels with its snapshot. The toy's DECODE token of an image session reads the rotary
// position (position + rope_delta, toy_device.h), so a restore that loses rope_delta changes the tokens. A build with
// kind 11: snapshot -> restore -> DECODE equals the uninterrupted session in tokens and in state (the restored
// session's snapshots are byte-identical to the original's, kind 11 = rope_delta); a text session's snapshot has no
// kind 11 and is unchanged. A build without kind 11 refuses the image session's snapshot (UNSUPPORTED, nothing kept).
void test_snapshot_image_session() {
    cere_executor *executor = make_executor();
    VisionDesc v(4096, {16, 16}, 3);
    v.names[1] = "model.visual.patch_embed.proj.weight";
    v.tensors[1].name = v.names[1].c_str();
    v.desc.snapshot_host_bytes = uint64_t(256) << 20;  // four A4 snapshots of a short session (~59 MB each)
    cere_model *model = sealed_model(executor, v);
    const uint64_t generation = generation_of(executor, model);
    const uint32_t vocab = a4_support().config.vocab_size;
    std::vector<uint64_t> offsets;
    const std::vector<uint32_t> prompt = image_prompt({4, 4}, &offsets);  // two grids (1, 4, 4): rope_delta -4
    const std::vector<Image> images{Image(offsets[0], 4, 4), Image(offsets[1], 4, 4)};
    std::vector<cere_image_input> inputs;
    for (const Image &i : images) inputs.push_back(i.input);
    // One drained work item of `s`: PREFILL of `prompt` (with the images unless `text`), else DECODE.
    const auto step = [&](cere_session *s, uint32_t op, uint64_t n, uint64_t revision, bool text = false) {
        Work w(s, op, op == CERE_OP_PREFILL ? prompt : std::vector<uint32_t>{}, n, generation, {}, revision);
        w.item.flags = CERE_WORK_NO_SPECULATION;
        if (op == CERE_OP_PREFILL && !text) {
            w.item.images = inputs.data();
            w.item.image_count = inputs.size();
        }
        cere_completion_ticket ticket{};
        CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
        const cere_completion c = drain(executor, ticket);
        CHECK_EQ(c.status, CERE_OK);
        return std::vector<uint32_t>(w.out.begin(), w.out.begin() + c.committed_tokens);
    };
    // The toy's image-session sequence: the PREFILL token at the prompt's positions, every DECODE token at
    // position + rope_delta (-4).
    std::vector<uint32_t> seq = prompt;
    seq.push_back(cere::toy::next_token(seq.back(), seq.size(), vocab));
    while (seq.size() < prompt.size() + 6) seq.push_back(cere::toy::next_token(seq.back(), seq.size() - 4, vocab));
    const std::vector<uint32_t> head(seq.begin() + prompt.size(), seq.begin() + prompt.size() + 2);
    const std::vector<uint32_t> tail(seq.begin() + prompt.size() + 2, seq.end());

    cere_session *a = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 2048, &a), CERE_OK);
    CHECK(step(a, CERE_OP_PREFILL, 2, 0) == head);
    CHECK_EQ(a->rope_delta, -4);
    const uint64_t used = model->arena.used();
    cere_snapshot *s1 = reinterpret_cast<cere_snapshot *>(uintptr_t(1));
    if (((snap::kWritableKinds >> snap::kRopeDelta) & 1) == 0) {
        // A build without kind 11 (ledger writer rule): the image session is never snapshotted partially.
        CHECK_EQ(cere_session_snapshot(executor, a, 1, &s1), CERE_ERR_UNSUPPORTED);
        CHECK(s1 == nullptr);
        CHECK_EQ(model->arena.used(), used);
        CHECK_EQ(model->snapshots, 0);
        CHECK_EQ(cere_session_release(executor, a), CERE_OK);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
        return;
    }
    CHECK_EQ(cere_session_snapshot(executor, a, 1, &s1), CERE_OK);
    const std::vector<uint8_t> bytes1 = read_all(executor, s1);
    // Kinds 0-8, then kind 11: 8 bytes, rope_delta i64 LE.
    CHECK_EQ(get64(bytes1, kAtSectionCount) & 0xffffffffu, 10);
    const uint64_t last = entry_at(9);
    CHECK_EQ(get64(bytes1, last) & 0xffffffffu, snap::kRopeDelta);
    CHECK_EQ(get64(bytes1, last + 16), 8);
    CHECK_EQ(int64_t(get64(bytes1, get64(bytes1, last + 8))), -4);
    // The uninterrupted run, and its state after it.
    CHECK(step(a, CERE_OP_DECODE, 4, 1) == tail);
    cere_snapshot *after = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, a, 2, &after), CERE_OK);
    const std::vector<uint8_t> bytes_after = read_all(executor, after);
    CHECK_EQ(cere_snapshot_release(executor, after), CERE_OK);
    CHECK_EQ(cere_session_release(executor, a), CERE_OK);

    // Restore (twice: a fork) and import: the same state before and after the same DECODE tokens.
    cere_snapshot *imported = nullptr;
    CHECK_EQ(cere_snapshot_import(executor, model, bytes1.data(), bytes1.size(), &imported), CERE_OK);
    for (cere_snapshot *from : {s1, s1, imported}) {
        cere_session *b = nullptr;
        CHECK_EQ(cere_session_create(executor, model, 2048, &b), CERE_OK);
        CHECK_EQ(cere_session_restore(executor, b, from), CERE_OK);
        CHECK_EQ(b->rope_delta, -4);
        cere_snapshot *again = nullptr;
        CHECK_EQ(cere_session_snapshot(executor, b, 0, &again), CERE_OK);
        CHECK(read_all(executor, again) == bytes1);
        CHECK_EQ(cere_snapshot_release(executor, again), CERE_OK);
        CHECK(step(b, CERE_OP_DECODE, 4, 0) == tail);
        CHECK_EQ(cere_session_snapshot(executor, b, 1, &again), CERE_OK);
        CHECK(read_all(executor, again) == bytes_after);
        CHECK_EQ(cere_snapshot_release(executor, again), CERE_OK);
        CHECK_EQ(cere_session_release(executor, b), CERE_OK);
    }

    // M2.13-L: CR19 is lifted: a session context above indexer_budget (2,048) restores the image session (K9 rotates
    // the prompt by positions3, every later row at position + rope_delta) and decodes as the original.
    cere_session *wide = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 4096, &wide), CERE_OK);
    CHECK_EQ(cere_session_restore(executor, wide, s1), CERE_OK);
    CHECK_EQ(wide->rope_delta, -4);
    CHECK(wide->image_session);
    CHECK(step(wide, CERE_OP_DECODE, 4, 0) == tail);
    CHECK_EQ(cere_session_release(executor, wide), CERE_OK);

    // A text session of the vision model: no kind 11 (the 8 sections of every text snapshot), and its restore decodes
    // as text.
    cere_session *t = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 2048, &t), CERE_OK);
    const auto text = toy_sequence(prompt, 6, vocab);
    CHECK(step(t, CERE_OP_PREFILL, 2, 0, true) == std::vector<uint32_t>(text.begin(), text.begin() + 2));
    cere_snapshot *ts = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, t, 1, &ts), CERE_OK);
    const std::vector<uint8_t> text_bytes = read_all(executor, ts);
    CHECK_EQ(get64(text_bytes, kAtSectionCount) & 0xffffffffu, 9);
    snap::Plan text_plan;
    CHECK(snap::plan(a4_support().config, prompt.size() + 1, 0, prompt.size() + 2, &text_plan));
    CHECK_EQ(text_bytes.size(), text_plan.total_bytes);
    CHECK_EQ(bytes1.size(), (text_plan.total_bytes + 255) / 256 * 256 + 8);  // kind 11 after a 256-B aligned pad
    CHECK_EQ(cere_session_release(executor, t), CERE_OK);
    CHECK_EQ(cere_session_create(executor, model, 2049, &t), CERE_OK);  // no image state: any context
    CHECK_EQ(cere_session_restore(executor, t, ts), CERE_OK);
    CHECK_EQ(t->rope_delta, 0);
    CHECK(step(t, CERE_OP_DECODE, 4, 0) == std::vector<uint32_t>(text.begin() + 2, text.end()));
    CHECK_EQ(cere_session_release(executor, t), CERE_OK);

    // A model without vision (the same config, so the same layout_id) cannot take the image session's state.
    Desc plain(a4_support().config, {16}, {}, 1, 4096);
    plain.desc.snapshot_host_bytes = uint64_t(64) << 20;
    cere_model *text_model = sealed_model(executor, plain);
    cere_snapshot *out = reinterpret_cast<cere_snapshot *>(uintptr_t(1));
    CHECK_EQ(cere_snapshot_import(executor, text_model, bytes1.data(), bytes1.size(), &out), CERE_ERR_UNSUPPORTED);
    CHECK(out == nullptr);
    CHECK_EQ(text_model->snapshots, 0);
    CHECK_EQ(cere_snapshot_import(executor, text_model, text_bytes.data(), text_bytes.size(), &out), CERE_OK);
    CHECK_EQ(cere_snapshot_release(executor, out), CERE_OK);

    for (cere_snapshot *x : {s1, imported, ts}) CHECK_EQ(cere_snapshot_release(executor, x), CERE_OK);
    CHECK_EQ(cere_model_release(executor, text_model), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

}  // namespace

// M2.12 C7 (m212 design note D9, ledger F): CERE_WORK_IGNORE_STOP_TOKENS is per ticket; under it neither the
// item's nor the model's stop tokens finish (PREFILL's first token included); the budget and the context still do.
// C7b: VERIFY with the bit is ARGUMENT (the driver sets NO_SPECULATION on such requests).
void test_ignore_stop_tokens() {
    cere_executor *executor = make_executor();
    Desc d = toy_desc();
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    const std::vector<uint32_t> prompt = {11, 12};
    const auto expected = toy_sequence(prompt, 12, vocab);
    d.stops = {expected[0], expected[5]};  // model stops: PREFILL's first token and a DECODE token
    d.sync();
    cere_model *model = sealed_model(executor, d);
    const uint64_t generation = generation_of(executor, model);

    cere_session *s = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &s), CERE_OK);
    Work prefill(s, CERE_OP_PREFILL, prompt, 4, generation, {expected[2]});
    prefill.item.flags = CERE_WORK_IGNORE_STOP_TOKENS;
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &prefill.batch, &ticket), CERE_OK);
    cere_completion c = drain(executor, ticket);
    CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
    CHECK_EQ(c.committed_tokens, 4);
    for (int i = 0; i < 4; ++i) CHECK_EQ(prefill.out[i], expected[i]);
    Work decode(s, CERE_OP_DECODE, {}, 4, generation, {}, 1);
    decode.item.flags = CERE_WORK_IGNORE_STOP_TOKENS | CERE_WORK_NO_SPECULATION;
    CHECK_EQ(cere_executor_submit(executor, &decode.batch, &ticket), CERE_OK);
    c = drain(executor, ticket);
    CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
    CHECK_EQ(c.committed_tokens, 4);
    CHECK_EQ(decode.out[1], expected[5]);  // the model stop token is an ordinary committed token
    // Without the bit the next ticket stops again (per ticket, not per session).
    Work plain(s, CERE_OP_DECODE, {}, 4, generation, {expected[9]}, 2);
    CHECK_EQ(cere_executor_submit(executor, &plain.batch, &ticket), CERE_OK);
    c = drain(executor, ticket);
    CHECK_EQ(c.finish, CERE_FINISH_STOP_TOKEN);
    CHECK_EQ(c.committed_tokens, 2);

    // The context limit still applies under the bit.
    cere_session *small = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 5, &small), CERE_OK);
    Work full(small, CERE_OP_PREFILL, prompt, 8, generation);
    full.item.flags = CERE_WORK_IGNORE_STOP_TOKENS;
    CHECK_EQ(cere_executor_submit(executor, &full.batch, &ticket), CERE_OK);
    c = drain(executor, ticket);
    CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
    CHECK_EQ(c.committed_tokens, 3);

    // C7b: VERIFY with bit 0 is ARGUMENT (before the VERIFY support check); without it, unsupported as before.
    cere_session *fresh = nullptr;
    CHECK_EQ(cere_session_release(executor, small), CERE_OK);
    CHECK_EQ(cere_session_create(executor, model, 32, &fresh), CERE_OK);
    Work verify(fresh, CERE_OP_VERIFY, {1}, 1, generation);
    verify.item.flags = CERE_WORK_IGNORE_STOP_TOKENS;
    CHECK_EQ(cere_executor_submit(executor, &verify.batch, &ticket), CERE_ERR_ARGUMENT);
    verify.item.flags = 0;
    CHECK_EQ(cere_executor_submit(executor, &verify.batch, &ticket), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_session_release(executor, fresh), CERE_OK);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// ---- M2.10 C3 (m210 design note Tier 1; D2, D3, D3′, D4/D4′, D7, D9, D11, D12, D15′, D16; W 132, P 44) ----
// The prefix cache through the ABI on the toy. The toy's fixed-state digest (ledger M4) is the analogue of the
// recurrent state a checkpoint carries: a hit run must end in the digest of a cold run over the same tokens.
namespace {

using PToks = std::vector<uint32_t>;
using cere::host::CaptureKind;
using cere::host::PrefixNode;

// Distinct token ids in [1, 69000] (the fixture vocab is 70,000); different seeds differ in their first page.
PToks ptoks(uint32_t seed, uint64_t n) {
    PToks t(n);
    for (uint64_t i = 0; i < n; ++i) t[i] = 1 + uint32_t((seed * 2654435761ull + i * 40503ull + seed * i) % 69000);
    return t;
}

PToks pjoin(const PToks &a, uint64_t a_len, const PToks &b) {
    PToks t(a.begin(), a.begin() + a_len);
    t.insert(t.end(), b.begin(), b.end());
    return t;
}

cere::host::CacheNamespace nsid(uint8_t b) {
    cere::host::CacheNamespace id{};
    id.fill(b);
    return id;
}

// The fixture model with M2.10's fields: `slots` prefix state slots, a KV pool of `pool` tokens (0 = derived) and a
// snapshot arena of `arena` bytes (the D9 host tier).
struct PrefixDesc : Desc {
    PrefixDesc(uint64_t slots, uint64_t max_sessions = 2, uint64_t max_context = 1024, uint64_t pool = 0,
               uint64_t arena = 0)
        : Desc(cere::toy::fixture_config(), {16, 32}, {}, max_sessions, max_context) {
        desc.prefix_state_slots = slots;
        desc.kv_pool_tokens = pool;
        desc.snapshot_host_bytes = arena;
    }
};

// A session of `context` tokens, namespaced by `id` unless it is 0.
cere_session *ns_session(cere_executor *executor, cere_model *model, uint64_t context, uint8_t id) {
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, context, &session), CERE_OK);
    if (session != nullptr && id != 0) {
        CHECK_EQ(cere_session_set_cache_namespace(executor, session, nsid(id).data()), CERE_OK);
    }
    return session;
}

uint64_t digest_of(const PToks &t) {
    uint64_t d = cere::toy::kDigestSeed;
    for (const uint32_t id : t) d = cere::toy::digest_step(d, id);
    return d;
}

uint64_t state_digest(cere_executor *executor, cere_session *session) {
    return toy(executor).session_state(*session->device).digest;
}

cere::host::PrefixCache &prefix_tree(cere_model *model) { return *model->prefix; }

// The stated node (device or host) at exactly `len` along `t` under namespace `id`, null when there is none.
const PrefixNode *stated_at(cere_model *model, uint8_t id, const PToks &t, uint64_t len) {
    const auto m = prefix_tree(model).match(nsid(id), t.data(), len, len, false);
    return m.h == len ? m.node : nullptr;
}

struct PRun {
    PToks out;
    std::vector<cere_token_score> scores;
    cere_completion c{};
};

// Submits one item and polls it to its terminal completion, checking cached_tokens == `cached` on every poll (P 44)
// and the terminal status `want`. `images` (M2.13-L) go with a PREFILL.
PRun prun(cere_executor *executor, cere_session *session, uint32_t op, const PToks &prompt, uint64_t max_new,
          uint64_t cached, uint32_t capture = 0, uint64_t revision = 0, cere_status want = CERE_OK,
          uint32_t flags = 0, const std::vector<cere_image_input> *images = nullptr) {
    Work w(session, op, prompt, max_new, session->model->generation, {}, revision);
    w.item.capture_position = capture;
    w.item.flags = flags;
    if (images != nullptr) {
        w.item.images = images->data();
        w.item.image_count = images->size();
    }
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
    PRun r;
    for (int i = 0; i < 100000; ++i) {
        r.c = completion_header();
        const cere_status st = cere_executor_poll(executor, ticket, &r.c);
        CHECK_EQ(r.c.cached_tokens, cached);
        if (st != CERE_PENDING) {
            CHECK_EQ(st, CERE_OK);
            break;
        }
    }
    CHECK_EQ(r.c.terminal, 1);
    CHECK_EQ(r.c.status, want);
    r.out.assign(w.out.begin(), w.out.begin() + r.c.committed_tokens);
    r.scores.assign(w.scores.begin(), w.scores.begin() + r.c.committed_tokens);
    return r;
}

// The session consumed `prompt` and every generated token but the last.
void check_cold_state(cere_executor *executor, cere_session *session, const PToks &prompt, const PRun &r) {
    PToks consumed = prompt;
    if (!r.out.empty()) consumed.insert(consumed.end(), r.out.begin(), r.out.end() - 1);
    CHECK(state_digest(executor, session) == digest_of(consumed));
}

uint64_t round256(uint64_t v) { return (v + 255) / 256 * 256; }

void test_prefix_namespace_api() {
    // D2: set_cache_namespace is UNSUPPORTED with slots 0 and ARGUMENT after the session's first submit (null id
    // too); an un-namespaced session on a cache-on model neither looks up nor inserts. C 60 native_prefix_cache.
    cere_executor *executor = make_executor();
    const PToks p = ptoks(1, 300);
    Desc plain = toy_desc();
    cere_model *off = sealed_model(executor, plain);
    cere_session *s0 = nullptr;
    CHECK_EQ(cere_session_create(executor, off, 32, &s0), CERE_OK);
    CHECK_EQ(cere_session_set_cache_namespace(executor, s0, nsid(1).data()), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_model_flush_prefix_cache(executor, off), CERE_ERR_UNSUPPORTED);
    cere_snapshot *out = reinterpret_cast<cere_snapshot *>(uintptr_t(1));
    CHECK_EQ(cere_prefix_cache_export(executor, off, nsid(1).data(), p.data(), p.size(), &out),
             CERE_ERR_UNSUPPORTED);
    CHECK(out == nullptr);
    PrefixDesc d(4, 2, 1024, 4096);
    cere_model *model = sealed_model(executor, d);
    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(cere_executor_capabilities(executor, off, &caps), CERE_OK);
    CHECK_EQ(caps.native_prefix_cache, 0);
    CHECK_EQ(cere_executor_capabilities(executor, model, &caps), CERE_OK);
    CHECK_EQ(caps.native_prefix_cache, 1);

    cere_session *s = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 1024, &s), CERE_OK);
    CHECK_EQ(cere_session_set_cache_namespace(executor, s, nullptr), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_session_set_cache_namespace(executor, s, nsid(1).data()), CERE_OK);
    CHECK_EQ(cere_session_set_cache_namespace(executor, s, nsid(2).data()), CERE_OK);  // until the first submit
    Work w(s, CERE_OP_PREFILL, p, 2, generation_of(executor, model));
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
    CHECK_EQ(cere_session_set_cache_namespace(executor, s, nsid(3).data()), CERE_ERR_ARGUMENT);  // in flight
    CHECK_EQ(drain(executor, ticket).status, CERE_OK);
    CHECK_EQ(cere_session_set_cache_namespace(executor, s, nsid(3).data()), CERE_ERR_ARGUMENT);  // after
    cere::host::PrefixCache &tree = prefix_tree(model);
    CHECK(stated_at(model, 2, p, 256) != nullptr);  // (a′) floor64(299) under the last namespace set
    CHECK(tree.root(nsid(1)) == nullptr);
    CHECK_EQ(tree.stats().inserts, 1);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);

    // No namespace: the same prompt neither hits nor captures, twice.
    for (int i = 0; i < 2; ++i) {
        cere_session *u = ns_session(executor, model, 1024, 0);
        check_cold_state(executor, u, p, prun(executor, u, CERE_OP_PREFILL, p, 2, 0));
        CHECK_EQ(cere_session_release(executor, u), CERE_OK);
    }
    CHECK_EQ(tree.stats().hits, 0);
    CHECK_EQ(tree.stats().captures, 1);
    CHECK_EQ(tree.stats().inserts, 1);
    // The namespaced repeat does hit (the lookup the un-namespaced sessions skipped).
    cere_session *r = ns_session(executor, model, 1024, 2);
    check_cold_state(executor, r, p, prun(executor, r, CERE_OP_PREFILL, p, 2, 256));
    CHECK_EQ(tree.stats().hits, 1);
    CHECK_EQ(cere_session_release(executor, r), CERE_OK);
    CHECK_EQ(cere_session_release(executor, s0), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_prefix_second_request() {
    // D3, P 44: the second request reports cached_tokens = h on every poll, ends in the cold state, and its tokens
    // and scores equal the cold-split run (the same prompt on a namespaced session after a flush).
    cere_executor *executor = make_executor();
    PrefixDesc d(4, 2, 1024, 4096);
    cere_model *model = sealed_model(executor, d);
    const PToks p = ptoks(2, 300);
    const PToks q = pjoin(p, 300, ptoks(3, 60));
    cere_session *a = ns_session(executor, model, 1024, 1);
    check_cold_state(executor, a, p, prun(executor, a, CERE_OP_PREFILL, p, 8, 0));
    cere_session *b = ns_session(executor, model, 1024, 1);
    const PRun hit = prun(executor, b, CERE_OP_PREFILL, q, 8, 256);
    check_cold_state(executor, b, q, hit);
    CHECK_EQ(hit.c.committed_tokens, 8);
    cere::host::PrefixCache &tree = prefix_tree(model);
    CHECK_EQ(tree.stats().hits, 1);
    // b's own (a′) floor64(359) = 320 lies past the shared path (b's pages there are a's): stored.
    const PrefixNode *deeper = stated_at(model, 1, q, 320);
    CHECK(deeper != nullptr && deeper->parent == stated_at(model, 1, p, 256));
    CHECK_EQ(tree.stats().capture_page_mismatch, 0);
    CHECK_EQ(cere_session_release(executor, a), CERE_OK);
    CHECK_EQ(cere_session_release(executor, b), CERE_OK);
    CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
    cere_session *cold = ns_session(executor, model, 1024, 1);
    const PRun split = prun(executor, cold, CERE_OP_PREFILL, q, 8, 0);
    check_cold_state(executor, cold, q, split);
    CHECK(split.out == hit.out);
    CHECK(std::memcmp(split.scores.data(), hit.scores.data(), hit.scores.size() * sizeof(cere_token_score)) == 0);
    CHECK_EQ(cere_session_release(executor, cold), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// BC4 (b) (batch-concurrency design note): three cache-on prompts under chunk 64 submitted together: a (330 tokens),
// b (a's first 290 and 10 others, 300) and c (300 others). The prompt with the fewest tokens left runs first, but b
// waits for a, which feeds b's shared prefix (prefix_wait), instead of starting cold: c (300) runs, then a, then b,
// which looks up a's capture at 256 (cached_tokens 256, as when it runs after a), with a decode round after each run
// of as many calls as the round has prompts. Every ticket's tokens and scores equal the run of the three in turn.
void test_prefix_shortest_first_wait() {
    const PToks p = ptoks(2, 330);
    const PToks q = pjoin(p, 290, ptoks(3, 10));
    const PToks r = ptoks(4, 300);
    PrefixDesc d(4, 3, 1024, 4096);
    std::vector<PRun> turn;
    {
        cere_executor *executor = make_executor("toy,prefill_chunk=64");
        cere_model *model = sealed_model(executor, d);
        const std::vector<std::pair<const PToks *, uint64_t>> items = {{&p, 0}, {&q, 256}, {&r, 0}};
        for (const auto &[prompt, cached] : items) {
            cere_session *s = ns_session(executor, model, 1024, 1);
            turn.push_back(prun(executor, s, CERE_OP_PREFILL, *prompt, 8, cached));
            CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        }
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
    cere_executor *executor = make_executor("toy,prefill_chunk=64");
    cere_model *model = sealed_model(executor, d);
    cere_session *as = ns_session(executor, model, 1024, 1);
    cere_session *bs = ns_session(executor, model, 1024, 1);
    cere_session *cs = ns_session(executor, model, 1024, 1);
    Job a = submit_job(executor, model, as, CERE_OP_PREFILL, p, 8);
    Job b = submit_job(executor, model, bs, CERE_OP_PREFILL, q, 8);
    Job c = submit_job(executor, model, cs, CERE_OP_PREFILL, r, 8);
    // One letter per round until b's prompt is consumed: the session whose prompt it fed, else D (decode).
    const auto fed = [&] { return std::array<uint64_t, 3>{as->fed, bs->fed, cs->fed}; };
    std::string kinds;
    while (bs->fed < q.size() && kinds.size() < 100) {
        const std::array<uint64_t, 3> before = fed();
        poll_pending(executor, c.ticket);
        const std::array<uint64_t, 3> after = fed();
        char kind = 'D';
        for (size_t i = 0; i < 3; ++i) {
            const uint64_t n = i == 0 ? p.size() : i == 1 ? q.size() : r.size();
            if (std::min(after[i], n) != std::min(before[i], n)) kind = char('a' + i);
        }
        kinds += kind;
    }
    CHECK(kinds == "cccccaDaaDaaDab");
    if (kinds != "cccccaDaaDaaDab") std::fprintf(stderr, "prefix kinds %s\n", kinds.c_str());
    const std::array<const Job *, 3> jobs = {&a, &b, &c};
    for (size_t i = 0; i < jobs.size(); ++i) {
        const cere_completion done = drain(executor, jobs[i]->ticket);
        CHECK_EQ(done.committed_tokens, 8);
        CHECK_EQ(done.cached_tokens, i == 1 ? 256 : 0);
        CHECK(PToks(jobs[i]->work->out.begin(), jobs[i]->work->out.begin() + 8) == turn[i].out);
        CHECK(std::memcmp(jobs[i]->work->scores.data(), turn[i].scores.data(), 8 * sizeof(cere_token_score)) == 0);
    }
    for (cere_session *s : {as, bs, cs}) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_prefix_internal_ticket() {
    // M210S-16: an INTERNAL ticket keeps the captures of its completed call ends and inserts nothing else: no
    // later split point, and its decode track capture is dropped (the track slot returns to the pool).
    {
        cere_executor *executor = make_executor("toy,prefill_chunk=64,fail_forward=2");
        PrefixDesc d(8, 1, 1024, 4096);
        cere_model *model = sealed_model(executor, d);
        const PToks p = ptoks(4, 300);
        cere_session *s = ns_session(executor, model, 1024, 1);
        prun(executor, s, CERE_OP_PREFILL, p, 4, 0, 0, 0, CERE_ERR_INTERNAL);
        CHECK(stated_at(model, 1, p, 64) != nullptr && stated_at(model, 1, p, 128) != nullptr);
        CHECK(prefix_tree(model).match(nsid(1), p.data(), 300, 299, false).h == 128);
        CHECK_EQ(prefix_tree(model).stats().inserts, 2);
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
    {
        cere_executor *executor = make_executor("toy,fail_forward=2");
        PrefixDesc d(8, 1, 1024, 4096);
        cere_model *model = sealed_model(executor, d);
        const PToks p = ptoks(5, 128);  // calls [0, 64) (a′) and [64, 128); the track capture at 128; DECODE fails
        cere_session *s = ns_session(executor, model, 1024, 1);
        prun(executor, s, CERE_OP_PREFILL, p, 4, 0, 0, 0, CERE_ERR_INTERNAL);
        CHECK(stated_at(model, 1, p, 64) != nullptr);
        CHECK(stated_at(model, 1, p, 128) == nullptr);
        CHECK_EQ(prefix_tree(model).stats().captures, 2);
        CHECK_EQ(prefix_tree(model).stats().inserts, 1);
        CHECK_EQ(s->track_slot, PrefixNode::kNoSlot);
        CHECK_EQ(prefix_tree(model).free_slots(), 8);
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
}

void test_prefix_cancel_keeps_captures() {
    // M210S-16: a cancel at a token boundary ends OK: the completed captures stay and the decode track capture is
    // inserted at the terminal poll; a request whose prompt reaches it hits there in the cold state.
    cere_executor *executor = make_executor("toy,hold_after=2");
    PrefixDesc d(4, 1, 1024, 4096);
    cere_model *model = sealed_model(executor, d);
    const PToks p = ptoks(6, 128);
    cere_session *s = ns_session(executor, model, 1024, 1);
    Work w(s, CERE_OP_PREFILL, p, 8, generation_of(executor, model));
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
    cere_completion c = completion_header();
    for (int i = 0; i < 16; ++i) {
        c = completion_header();
        CHECK_EQ(cere_executor_poll(executor, ticket, &c), CERE_PENDING);
    }
    CHECK_EQ(c.committed_tokens, 2);
    CHECK_EQ(s->tracked, 128);
    CHECK(stated_at(model, 1, p, 64) != nullptr && stated_at(model, 1, p, 128) == nullptr);
    CHECK_EQ(cere_executor_cancel(executor, ticket), CERE_OK);
    c = drain(executor, ticket);
    CHECK_EQ(c.status, CERE_OK);
    CHECK_EQ(c.finish, CERE_FINISH_CANCELLED);
    const PrefixNode *track = stated_at(model, 1, p, 128);
    CHECK(track != nullptr && track->kind == CaptureKind::kDecode);
    CHECK(stated_at(model, 1, p, 64) != nullptr);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    const PToks next = pjoin(p, 128, {w.out[0]});  // cap floor64(128) = 128
    cere_session *t = ns_session(executor, model, 1024, 1);
    check_cold_state(executor, t, next, prun(executor, t, CERE_OP_PREFILL, next, 1, 128));
    CHECK_EQ(cere_session_release(executor, t), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_prefix_busy_and_flush() {
    // D11, D12, D16: flush, export and import are BUSY while a ticket of the model is in flight; flush is BUSY while
    // a session holds a hit lock and OK once it is released; after a flush the tree is empty, every state slot and
    // every page (sessions released) free.
    cere_executor *executor = make_executor("toy,hold_after=1");
    PrefixDesc d(4, 2, 1024, 4096, uint64_t(4) << 20);
    cere_model *model = sealed_model(executor, d);
    cere::host::PrefixCache &tree = prefix_tree(model);
    const PToks p = ptoks(7, 300);
    cere_session *a = ns_session(executor, model, 1024, 1);
    prun(executor, a, CERE_OP_PREFILL, p, 1, 0);
    cere_snapshot *exported = nullptr;
    CHECK_EQ(cere_prefix_cache_export(executor, model, nsid(1).data(), p.data(), p.size(), &exported), CERE_OK);
    cere_session *b = ns_session(executor, model, 1024, 1);
    Work w(b, CERE_OP_PREFILL, ptoks(8, 100), 4, generation_of(executor, model));
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
    for (int i = 0; i < 4; ++i) {
        cere_completion c = completion_header();
        CHECK_EQ(cere_executor_poll(executor, ticket, &c), CERE_PENDING);
    }
    cere_snapshot *busy = nullptr;
    CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_ERR_BUSY);
    CHECK_EQ(cere_prefix_cache_export(executor, model, nsid(1).data(), p.data(), p.size(), &busy), CERE_ERR_BUSY);
    CHECK(busy == nullptr);
    CHECK_EQ(cere_prefix_cache_import(executor, model, nsid(2).data(), exported), CERE_ERR_BUSY);
    CHECK(tree.root(nsid(2)) == nullptr);
    CHECK_EQ(cere_executor_cancel(executor, ticket), CERE_OK);
    CHECK_EQ(drain(executor, ticket).status, CERE_OK);
    CHECK(tree.root(nsid(1)) != nullptr);
    CHECK_EQ(cere_session_release(executor, a), CERE_OK);
    CHECK_EQ(cere_session_release(executor, b), CERE_OK);
    CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
    CHECK(tree.root(nsid(1)) == nullptr);
    CHECK_EQ(tree.free_slots(), tree.slots());
    CHECK_EQ(toy(executor).free_pages(*model->device), 64);
    // A hit lock: flush BUSY until the holding session is released.
    CHECK_EQ(cere_prefix_cache_import(executor, model, nsid(1).data(), exported), CERE_OK);
    cere_session *h = ns_session(executor, model, 1024, 1);
    prun(executor, h, CERE_OP_PREFILL, pjoin(p, 300, ptoks(9, 10)), 1, 256);
    CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_ERR_BUSY);
    CHECK_EQ(cere_session_release(executor, h), CERE_OK);
    CHECK(!tree.locked());
    CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
    CHECK_EQ(toy(executor).free_pages(*model->device), 64);
    CHECK_EQ(cere_snapshot_release(executor, exported), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_prefix_export_import() {
    // D12: export writes the deepest cached state of the tokens as a CERESNP1 prefix (first_row 0, token_count =
    // position = fed = h); import inserts it under a namespace, here after a flush and into a fresh model (the
    // bytes moved through cere_snapshot_import), and a hit there ends in the cold state. An import onto a path the
    // namespace already holds stores nothing (M210S-39: its fresh pages are not the path's).
    cere_executor *executor = make_executor();
    PrefixDesc d(4, 1, 1024, 4096, uint64_t(4) << 20);
    cere_model *model = sealed_model(executor, d);
    cere::host::PrefixCache &tree = prefix_tree(model);
    const PToks p = ptoks(10, 300);
    const PToks q = pjoin(p, 300, ptoks(11, 40));
    cere_session *a = ns_session(executor, model, 1024, 1);
    prun(executor, a, CERE_OP_PREFILL, p, 1, 0);
    CHECK_EQ(cere_session_release(executor, a), CERE_OK);
    cere_snapshot *none = nullptr;
    CHECK_EQ(cere_prefix_cache_export(executor, model, nsid(3).data(), p.data(), p.size(), &none),
             CERE_ERR_ARGUMENT);  // nothing cached under namespace 3
    cere_snapshot *e = nullptr;
    CHECK_EQ(cere_prefix_cache_export(executor, model, nsid(1).data(), p.data(), p.size(), &e), CERE_OK);
    cere_snapshot_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(cere_snapshot_get_info(executor, e, &info), CERE_OK);
    CHECK_EQ(info.position, 256);
    CHECK_EQ(info.token_count, 256);
    CHECK_EQ(info.fed, 256);
    const std::vector<uint8_t> bytes = read_all(executor, e);
    CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
    CHECK_EQ(cere_prefix_cache_import(executor, model, nsid(2).data(), e), CERE_OK);
    const PrefixNode *imported = stated_at(model, 2, p, 256);
    CHECK(imported != nullptr && imported->kind == CaptureKind::kImport);
    CHECK(tree.root(nsid(1)) == nullptr);
    const uint64_t inserts = tree.stats().inserts;
    CHECK_EQ(cere_prefix_cache_import(executor, model, nsid(2).data(), e), CERE_OK);
    CHECK_EQ(tree.stats().inserts, inserts);
    CHECK_EQ(tree.stats().capture_page_mismatch, 1);
    CHECK(stated_at(model, 2, p, 256) == imported);
    cere_session *b = ns_session(executor, model, 1024, 2);
    check_cold_state(executor, b, q, prun(executor, b, CERE_OP_PREFILL, q, 2, 256));
    CHECK_EQ(cere_session_release(executor, b), CERE_OK);

    // A fresh model of the same layout: the exported bytes imported as a snapshot, then into its tree.
    PrefixDesc d2(4, 1, 1024, 4096, uint64_t(4) << 20);
    cere_model *fresh = sealed_model(executor, d2);
    cere_snapshot *moved = nullptr;
    CHECK_EQ(cere_snapshot_import(executor, fresh, bytes.data(), bytes.size(), &moved), CERE_OK);
    CHECK_EQ(cere_prefix_cache_import(executor, fresh, nsid(1).data(), moved), CERE_OK);
    cere_session *c = ns_session(executor, fresh, 1024, 1);
    check_cold_state(executor, c, q, prun(executor, c, CERE_OP_PREFILL, q, 2, 256));
    CHECK_EQ(cere_session_release(executor, c), CERE_OK);
    for (cere_snapshot *s : {e, moved}) CHECK_EQ(cere_snapshot_release(executor, s), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_prefix_restore() {
    // M2.11 restore into a fresh session of a cache-on model (a lazy session): it binds its own pages, decodes
    // equal to the original, and (namespaced) inserts its decode capture at the terminal poll.
    cere_executor *executor = make_executor();
    PrefixDesc d(4, 2, 1024, 4096, uint64_t(4) << 20);
    cere_model *model = sealed_model(executor, d);
    const PToks p = ptoks(12, 100);
    cere_session *a = ns_session(executor, model, 1024, 0);
    const PRun first = prun(executor, a, CERE_OP_PREFILL, p, 4, 0);
    cere_snapshot *s = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, a, 1, &s), CERE_OK);
    cere_session *b = ns_session(executor, model, 1024, 1);
    CHECK_EQ(cere_session_restore(executor, b, s), CERE_OK);
    CHECK(b->bound);
    CHECK_EQ(toy(executor).session_pages(*b->device).size(), 16);
    CHECK(state_digest(executor, b) == state_digest(executor, a));
    const PRun original = prun(executor, a, CERE_OP_DECODE, {}, 30, 0, 0, 1);
    const PRun restored = prun(executor, b, CERE_OP_DECODE, {}, 30, 0, 0, 0);
    CHECK(restored.out == original.out);
    CHECK(state_digest(executor, b) == state_digest(executor, a));
    PToks all = p;
    all.insert(all.end(), first.out.begin(), first.out.end());
    all.insert(all.end(), restored.out.begin(), restored.out.end());
    const PrefixNode *track = stated_at(model, 1, all, 128);
    CHECK(track != nullptr && track->kind == CaptureKind::kDecode);
    CHECK(prefix_tree(model).root(nsid(0)) == nullptr);  // a (no namespace) inserted nothing
    CHECK_EQ(prefix_tree(model).stats().inserts, 1);
    CHECK_EQ(cere_snapshot_release(executor, s), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_prefix_capture_position() {
    // W 132: PREFILL only, a multiple of 64 with 0 < p < token_count, else ARGUMENT (nothing submitted, the session
    // stays usable); never an error that depends on the cache: accepted and ignored with no namespace or slots 0,
    // and dropped when p − h < 8,192.
    cere_executor *executor = make_executor();
    PrefixDesc d(4, 2, 1024, 4096);
    cere_model *model = sealed_model(executor, d);
    const uint64_t generation = generation_of(executor, model);
    const PToks p = ptoks(13, 128);
    cere_session *s = ns_session(executor, model, 1024, 1);
    cere_completion_ticket ticket{};
    for (const uint32_t bad : {100u, 128u, 192u}) {  // not on the grid, == n, > n
        Work w(s, CERE_OP_PREFILL, p, 1, generation);
        w.item.capture_position = bad;
        CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_ERR_ARGUMENT);
    }
    prun(executor, s, CERE_OP_PREFILL, p, 1, 0, 64);
    Work decode(s, CERE_OP_DECODE, {}, 1, generation, {}, 1);
    decode.item.capture_position = 64;
    CHECK_EQ(cere_executor_submit(executor, &decode.batch, &ticket), CERE_ERR_ARGUMENT);
    prun(executor, s, CERE_OP_DECODE, {}, 1, 0, 0, 1);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    // p = 64 above was dropped (64 − 0 < 8,192): the (a′) state at 64 is a tail, not a boundary.
    const PrefixNode *tail = stated_at(model, 1, p, 64);
    CHECK(tail != nullptr && tail->kind == CaptureKind::kTail);
    const uint64_t captures = prefix_tree(model).stats().captures;
    // No namespace: a valid p is accepted and ignored.
    cere_session *u = ns_session(executor, model, 1024, 0);
    prun(executor, u, CERE_OP_PREFILL, ptoks(14, 300), 1, 0, 128);
    CHECK_EQ(prefix_tree(model).stats().captures, captures);
    CHECK_EQ(cere_session_release(executor, u), CERE_OK);
    // Slots 0: accepted and ignored; the ARGUMENT rule still holds.
    Desc plain = toy_desc();
    plain.desc.max_context_tokens = 1024;
    cere_model *off = sealed_model(executor, plain);
    cere_session *o = nullptr;
    CHECK_EQ(cere_session_create(executor, off, 1024, &o), CERE_OK);
    Work bad(o, CERE_OP_PREFILL, p, 1, generation_of(executor, off));
    bad.item.capture_position = 96;
    CHECK_EQ(cere_executor_submit(executor, &bad.batch, &ticket), CERE_ERR_ARGUMENT);
    prun(executor, o, CERE_OP_PREFILL, p, 1, 0, 64);
    CHECK_EQ(cere_session_release(executor, o), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_prefix_host_tier() {
    // D9: with a snapshot arena an evicted leaf state spills to the host tier; a later hit loads it back (taking
    // a slot, which spills the next LRU state) and ends in the cold state.
    cere_executor *executor = make_executor();
    PrefixDesc d(1, 1, 1024, 4096, uint64_t(8) << 20);  // 2 state slots
    cere_model *model = sealed_model(executor, d);
    cere::host::PrefixCache &tree = prefix_tree(model);
    const PToks x = ptoks(15, 200);
    const PToks y = ptoks(16, 200);
    const PToks z = ptoks(17, 200);
    for (const PToks *t : {&x, &y, &z}) {
        cere_session *s = ns_session(executor, model, 1024, 1);
        prun(executor, s, CERE_OP_PREFILL, *t, 1, 0);
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    }
    CHECK_EQ(tree.stats().host_spills, 1);
    const PrefixNode *spilled = stated_at(model, 1, x, 192);
    CHECK(spilled != nullptr && spilled->host && spilled->pages.empty());
    const PToks x2 = pjoin(x, 200, ptoks(18, 20));
    cere_session *s = ns_session(executor, model, 1024, 1);
    check_cold_state(executor, s, x2, prun(executor, s, CERE_OP_PREFILL, x2, 1, 192));
    CHECK_EQ(tree.stats().host_loads, 1);
    CHECK_EQ(tree.stats().host_spills, 2);  // y's state made room for the load
    const PrefixNode *loaded = stated_at(model, 1, x, 192);
    CHECK(loaded != nullptr && !loaded->host && loaded->slot != PrefixNode::kNoSlot);
    const PrefixNode *ys = stated_at(model, 1, y, 192);
    CHECK(ys != nullptr && ys->host);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_prefix_host_tier_internal() {
    // D9 [REVISED: M210S-42]: a prompt's tail state whose only child is the request's decode leaf spills to the host
    // tier too (the leaf first, then the tail once its children are host-resident), so a later prompt that shares
    // the prompt but not the reply still hits the tail, and the decode leaf stays loadable below it.
    cere_executor *executor = make_executor();
    PrefixDesc d(2, 1, 1024, 4096, uint64_t(16) << 20);  // 3 state slots
    cere_model *model = sealed_model(executor, d);
    cere::host::PrefixCache &tree = prefix_tree(model);
    const PToks x = ptoks(30, 200);
    PToks xs;
    for (uint32_t seed : {30u, 31u, 32u}) {
        const PToks t = ptoks(seed, 200);
        cere_session *s = ns_session(executor, model, 1024, 1);
        const PRun r = prun(executor, s, CERE_OP_PREFILL, t, 60, 0);
        if (seed == 30) {
            xs = t;
            xs.insert(xs.end(), r.out.begin(), r.out.end());
        }
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    }
    const PrefixNode *tail = stated_at(model, 1, x, 192);
    CHECK(tail != nullptr && tail->host && !tail->children.empty());
    const PrefixNode *leaf = stated_at(model, 1, xs, 256);
    CHECK(leaf != nullptr && leaf->host && leaf->parent == tail);
    const PToks x2 = pjoin(x, 200, ptoks(33, 20));
    cere_session *s = ns_session(executor, model, 1024, 1);
    check_cold_state(executor, s, x2, prun(executor, s, CERE_OP_PREFILL, x2, 1, 192));
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    // The decode leaf loads back below its (now device) tail: prompt + reply + a new turn hits at 256.
    const uint64_t loads = tree.stats().host_loads;
    const PToks next = pjoin(xs, 260, ptoks(34, 20));
    cere_session *t = ns_session(executor, model, 1024, 1);
    check_cold_state(executor, t, next, prun(executor, t, CERE_OP_PREFILL, next, 1, 256));
    CHECK(tree.stats().host_loads > loads);
    CHECK_EQ(cere_session_release(executor, t), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_prefix_host_tier_tail_last() {
    // R5 (M210S-43): with a host tier, a prompt with a boundary state (capture_position >= 8,192 past the hit), a
    // tail state and a decode leaf, run cold then again at once, hits at the tail floor64(n - 1) on the device: the
    // tail is never spilled before the older prompts' states and its own boundary.
    cere_executor *executor = make_executor();
    PrefixDesc d(4, 1, 12288, 40960, uint64_t(256) << 20);
    cere_model *model = sealed_model(executor, d);
    for (uint32_t seed = 40; seed < 44; ++seed) {
        const PToks p = ptoks(seed, 9000);
        cere_session *s = ns_session(executor, model, 12288, 1);
        prun(executor, s, CERE_OP_PREFILL, p, 60, 0, 8256);
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        cere_session *t = ns_session(executor, model, 12288, 1);
        prun(executor, t, CERE_OP_PREFILL, p, 1, 8960, 8256);
        CHECK_EQ(cere_session_release(executor, t), CERE_OK);
    }
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_prefix_arena_priority() {
    // D9 / ledger CR11: a session snapshot that does not fit the arena drops host entries (LRU) before CAPACITY.
    const uint64_t entry = round256(snapshot_size(192, 192));
    const uint64_t session_bytes = round256(snapshot_size(200, 201));
    CHECK(session_bytes <= 2 * entry);
    cere_executor *executor = make_executor();
    PrefixDesc d(1, 1, 1024, 4096, 2 * entry);  // 2 state slots, room for two host entries
    cere_model *model = sealed_model(executor, d);
    cere::host::PrefixCache &tree = prefix_tree(model);
    for (uint32_t seed = 20; seed < 24; ++seed) {
        cere_session *s = ns_session(executor, model, 1024, 1);
        prun(executor, s, CERE_OP_PREFILL, ptoks(seed, 200), 1, 0);
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    }
    CHECK_EQ(tree.stats().host_spills, 2);
    CHECK_EQ(tree.stats().host_drops, 0);
    cere_session *u = ns_session(executor, model, 1024, 0);
    prun(executor, u, CERE_OP_PREFILL, ptoks(24, 200), 1, 0);
    cere_snapshot *snapshot = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, u, 1, &snapshot), CERE_OK);
    CHECK_EQ(tree.stats().host_drops, session_bytes <= entry ? 1 : 2);
    CHECK(stated_at(model, 1, ptoks(20, 200), 192) == nullptr);  // the LRU entry went first
    CHECK_EQ(cere_snapshot_release(executor, snapshot), CERE_OK);
    CHECK_EQ(cere_session_release(executor, u), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_prefix_decode_capture() {
    // D4 (b): a decode capture at every committed multiple of 64 goes to the session's track slot and the latest
    // is inserted at the terminal poll (not before); a request whose prompt is the previous prompt + completion
    // hits at that point in the cold state.
    {
        cere_executor *executor = make_executor("toy,hold_after=95");
        PrefixDesc d(4, 1, 1024, 4096);
        cere_model *model = sealed_model(executor, d);
        const PToks p = ptoks(25, 100);
        cere_session *s = ns_session(executor, model, 1024, 1);
        Work w(s, CERE_OP_PREFILL, p, 100, generation_of(executor, model));
        cere_completion_ticket ticket{};
        CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
        cere_completion c = completion_header();
        for (int i = 0; i < 400; ++i) {
            c = completion_header();
            CHECK_EQ(cere_executor_poll(executor, ticket, &c), CERE_PENDING);
        }
        CHECK_EQ(c.committed_tokens, 95);
        CHECK_EQ(s->tracked, 192);
        PToks seq = p;
        seq.insert(seq.end(), w.out.begin(), w.out.begin() + 95);
        CHECK(stated_at(model, 1, seq, 192) == nullptr && stated_at(model, 1, seq, 128) == nullptr);
        CHECK_EQ(cere_executor_cancel(executor, ticket), CERE_OK);
        CHECK_EQ(drain(executor, ticket).status, CERE_OK);
        const PrefixNode *track = stated_at(model, 1, seq, 192);
        CHECK(track != nullptr && track->kind == CaptureKind::kDecode);
        CHECK(stated_at(model, 1, seq, 128) == nullptr);  // overwritten in the track slot
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
    cere_executor *executor = make_executor();
    PrefixDesc d(4, 1, 1024, 4096);
    cere_model *model = sealed_model(executor, d);
    const PToks p = ptoks(26, 100);
    cere_session *s = ns_session(executor, model, 1024, 1);
    const PRun first = prun(executor, s, CERE_OP_PREFILL, p, 100, 0);
    CHECK_EQ(first.c.finish, CERE_FINISH_LENGTH);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    PToks next = p;
    next.insert(next.end(), first.out.begin(), first.out.end());  // 200 tokens: cap floor64(199) = 192
    cere_session *t = ns_session(executor, model, 1024, 1);
    check_cold_state(executor, t, next, prun(executor, t, CERE_OP_PREFILL, next, 1, 192));
    CHECK_EQ(prefix_tree(model).stats().hits, 1);
    CHECK_EQ(cere_session_release(executor, t), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

}  // namespace

// --- M2.9 speculation (m29 design note) ---------------------------------------------------------------------------

// A sorted draft token map of the toy fixture (the toy's proposals ignore the map).
const std::vector<uint32_t> &draft_map() {
    static const std::vector<uint32_t> map = [] {
        std::vector<uint32_t> m;
        for (uint32_t i = 0; i < 1024; ++i) m.push_back(i * 68);
        return m;
    }();
    return map;
}

// Sets the M2.9 desc fields (I1, I2): `drafts` with `map` (null: no map).
void set_drafts(Desc &d, uint32_t drafts = 4, const std::vector<uint32_t> *map = &draft_map()) {
    d.desc.speculative_draft_tokens = drafts;
    d.desc.draft_token_map = map != nullptr ? map->data() : nullptr;
    d.desc.draft_token_map_count = map != nullptr ? map->size() : 0;
}

cere_execution_capabilities caps_of(cere_executor *executor, cere_model *model) {
    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(cere_executor_capabilities(executor, model, &caps), CERE_OK);
    return caps;
}

// M2.9 C3 (I1-I3): speculative_draft_tokens is 0 or 4 (else UNSUPPORTED); a draft token map exactly with 4, of
// 1 <= count strictly increasing ids below vocab (else ARGUMENT), copied into the plan; the device decides whether it
// runs the draft (toy ",draft=0": UNSUPPORTED, nothing allocated); capabilities report the draft.
void test_spec_abi() {
    cere_executor *executor = make_executor();
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    Desc d = toy_desc();
    cere_model *model = nullptr;
    const auto refused = [&](uint32_t drafts, const std::vector<uint32_t> *map, cere_status want) {
        set_drafts(d, drafts, map);
        CHECK_EQ(cere_model_create(executor, &d.desc, &model), want);
        CHECK(model == nullptr);
    };
    for (const uint32_t drafts : {1u, 2u, 3u, 5u, 255u}) refused(drafts, &draft_map(), CERE_ERR_UNSUPPORTED);
    refused(4, nullptr, CERE_ERR_ARGUMENT);  // no map
    const std::vector<uint32_t> empty;
    refused(4, &empty, CERE_ERR_ARGUMENT);  // count 0
    d.desc.draft_token_map = draft_map().data();
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_ERR_ARGUMENT);  // a pointer with count 0
    const std::vector<uint32_t> unsorted = {3, 2, 9};
    const std::vector<uint32_t> repeated = {3, 3, 9};
    const std::vector<uint32_t> outside = {3, 9, vocab};
    for (const auto *map : {&unsorted, &repeated, &outside}) refused(4, map, CERE_ERR_ARGUMENT);
    refused(0, &draft_map(), CERE_ERR_ARGUMENT);  // a map without drafts
    set_drafts(d, 0, nullptr);
    d.desc.draft_token_map_count = 1;  // a count without drafts
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_ERR_ARGUMENT);

    // count == vocab (every id) is the largest map; the plan holds a copy.
    std::vector<uint32_t> every(vocab);
    for (uint32_t i = 0; i < vocab; ++i) every[i] = i;
    set_drafts(d, 4, &every);
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    CHECK_EQ(model->plan.speculative_draft_tokens, 4);
    CHECK(model->plan.draft_token_map == every);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);

    set_drafts(d);
    model = sealed_model(executor, d);
    CHECK(model->plan.draft_token_map == draft_map());
    cere_execution_capabilities caps = caps_of(executor, model);
    CHECK_EQ(caps.native_speculative_batch, 1);
    CHECK_EQ(caps.speculative_draft_tokens, 4);
    CHECK_EQ(caps.operations, CERE_OPS_PREFILL | CERE_OPS_DECODE | CERE_OPS_VERIFY | CERE_OPS_CANCEL);
    set_drafts(d, 0, nullptr);
    cere_model *plain = sealed_model(executor, d);
    caps = caps_of(executor, plain);
    CHECK_EQ(caps.native_speculative_batch + caps.speculative_draft_tokens, 0);
    CHECK_EQ(caps.operations, CERE_OPS_PREFILL | CERE_OPS_DECODE | CERE_OPS_CANCEL);
    CHECK_EQ(cere_model_release(executor, plain), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    // A device without draft support refuses the desc before anything is allocated.
    cere_executor *nodraft = make_executor("toy,draft=0");
    set_drafts(d);
    CHECK_EQ(cere_model_create(nodraft, &d.desc, &model), CERE_ERR_UNSUPPORTED);
    CHECK(model == nullptr);
    CHECK_EQ(toy(nodraft).allocations(), 0);
    CHECK_EQ(cere_executor_destroy(nodraft), CERE_OK);
    // The draft options' grammar.
    for (const char *id : {"toy,draft=2", "toy,draft_miss=0", "toy,draft_miss=4", "toy,stateful=2", "toy,draft",
                           "toy,cycle=1", "toy,cycle=65"}) {
        cere_executor_options options{};
        options.struct_size = sizeof(options);
        options.abi_version = CERE_EXECUTOR_ABI_VERSION;
        options.device_id = id;
        options.worker_epoch = kEpoch;
        options.memory_budget_bytes = uint64_t(1) << 30;
        cere_executor *bad = nullptr;
        CHECK_EQ(cere_executor_create(&bad, &options), CERE_ERR_ARGUMENT);
        CHECK(bad == nullptr);
    }
}

// ---- M2.9 speculation on the toy (m29 design note C1, C2, C4, C5, C9; D8 kind 10) --------------------------------

namespace {

// One generation request: `prompt` (a PREFILL's prompt; a DECODE/VERIFY's tokens), up to `n` tokens, item stop ids,
// cere_work.flags and the session's context.
struct Gen {
    std::vector<uint32_t> prompt;
    uint64_t n = 8;
    std::vector<uint32_t> stops = {};
    uint32_t flags = 0;
    uint64_t context = 64;
};

// A ticket's tokens and scores up to its terminal committed_tokens, its terminal completion and the tokens each of
// its polls added.
struct Outcome {
    std::vector<uint32_t> out;
    std::vector<cere_token_score> scores;
    cere_completion c{};
    std::vector<uint64_t> bursts;
};

Job submit_gen(cere_executor *executor, cere_model *model, cere_session *session, uint32_t op, const Gen &g) {
    Job job;
    job.work = std::make_unique<Work>(session, op, g.prompt, g.n, generation_of(executor, model), g.stops,
                                      session->revision);
    job.work->item.flags = g.flags;
    CHECK_EQ(cere_executor_submit(executor, &job.work->batch, &job.ticket), CERE_OK);
    return job;
}

// One poll of `ticket` into `o` (a burst); true once terminal.
bool poll_into(cere_executor *executor, const cere_completion_ticket &ticket, const Work &w, Outcome &o) {
    cere_completion c = completion_header();
    const cere_status st = cere_executor_poll(executor, ticket, &c);
    const uint64_t last = o.c.committed_tokens;
    CHECK(c.committed_tokens >= last);
    o.bursts.push_back(c.committed_tokens - last);
    o.c = c;
    o.out.assign(w.out.begin(), w.out.begin() + c.committed_tokens);
    o.scores.assign(w.scores.begin(), w.scores.begin() + c.committed_tokens);
    if (st == CERE_PENDING) return false;
    CHECK_EQ(st, CERE_OK);
    CHECK_EQ(c.terminal, 1);
    return true;
}

Outcome finish(cere_executor *executor, const cere_completion_ticket &ticket, const Work &w, Outcome o = {}) {
    for (int i = 0; i < 10000 && !poll_into(executor, ticket, w, o); ++i) {
    }
    return o;
}

// Polls the jobs round-robin until every one is terminal.
std::vector<Outcome> finish_all(cere_executor *executor, std::vector<Job> &jobs) {
    std::vector<Outcome> out(jobs.size());
    std::vector<bool> done(jobs.size(), false);
    for (int i = 0; i < 10000 && std::count(done.begin(), done.end(), false) != 0; ++i) {
        for (size_t j = 0; j < jobs.size(); ++j) {
            if (!done[j]) done[j] = poll_into(executor, jobs[j].ticket, *jobs[j].work, out[j]);
        }
    }
    return out;
}

cere_executor *spec_executor(const std::string &device) { return make_executor(device.c_str()); }

// A sealed toy model with `drafts` (4 with the fixture's draft map, or 0) of `sessions` sessions.
cere_model *gen_model(cere_executor *executor, Desc &d, bool drafts) {
    if (drafts) set_drafts(d);
    return sealed_model(executor, d);
}

// `g` as one PREFILL on a fresh executor of `device`, its model with or without drafts.
Outcome gen_run(const std::string &device, const Gen &g, bool drafts) {
    cere_executor *executor = spec_executor(device);
    Desc d = rows_desc(1, std::max<uint64_t>(64, g.context));
    cere_model *model = gen_model(executor, d, drafts);
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, g.context, &session), CERE_OK);
    Job job = submit_gen(executor, model, session, CERE_OP_PREFILL, g);
    const Outcome o = finish(executor, job.ticket, *job.work);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    return o;
}

// Tokens, scores (bit for bit) and finish of `spec` equal `plain`'s.
void check_same(const Outcome &spec, const Outcome &plain) {
    CHECK_EQ(spec.c.committed_tokens, plain.c.committed_tokens);
    CHECK_EQ(spec.c.finish, plain.c.finish);
    CHECK_EQ(spec.c.status, plain.c.status);
    CHECK(spec.out == plain.out);
    CHECK(spec.scores.size() == plain.scores.size() &&
          std::memcmp(spec.scores.data(), plain.scores.data(), spec.scores.size() * sizeof(cere_token_score)) == 0);
    CHECK_EQ(plain.c.verify_steps + plain.c.accepted_drafts, 0);
}

// The toy's (verify_steps, accepted_drafts) for a drafting ticket that committed `total` tokens, the first `first` of
// them outside speculation (1 for a PREFILL's scoring step), under a commit limit `limit` (min(max_new_tokens,
// context - tokens before the ticket)); `miss` is draft_miss (0 = never); `drafts` false for a draft-0 ticket. Each
// round's chain is min(max_chain, room - 1) drafts (max_chain: the window - 1, 3 for speculative_draft_tokens 4 and
// draft_tokens - 1 under the ABI v4 draft_tokens lever); the miss-th is rejected (its correction commits), a stop
// truncates.
std::pair<uint64_t, uint64_t> predicted(uint64_t total, uint64_t limit, uint32_t miss, bool drafts = true,
                                        uint64_t first = 1, uint64_t max_chain = 3) {
    uint64_t steps = 0;
    uint64_t accepted = 0;
    for (uint64_t c = first; c < total; ++steps) {
        const uint64_t chain = drafts ? std::min<uint64_t>(max_chain, limit - c - 1) : 0;
        const uint64_t k = miss == 0 ? chain : std::min<uint64_t>(chain, miss - 1);
        const uint64_t commit = std::min(k + 1, total - c);
        accepted += std::min(k, commit);
        c += commit;
    }
    return {steps, accepted};
}

void check_counts(const Outcome &o, std::pair<uint64_t, uint64_t> want) {
    CHECK_EQ(o.c.verify_steps, want.first);
    CHECK_EQ(o.c.accepted_drafts, want.second);
}

struct DraftDevice {
    const char *id;
    uint32_t miss;
};
constexpr DraftDevice kDraftDevices[] = {{"toy,stateful=1", 0},
                                         {"toy,stateful=1,draft_miss=1", 1},
                                         {"toy,stateful=1,draft_miss=2", 2},
                                         {"toy,stateful=1,draft_miss=3", 3}};

// The index of the first occurrence of `v` in `seq`.
size_t first_index(const std::vector<uint32_t> &seq, uint32_t v) {
    return size_t(std::find(seq.begin(), seq.end(), v) - seq.begin());
}

}  // namespace

// C1 (D1, D5): on the stateful toy (a consumed wrong token changes every later token) a drafting model's greedy
// tokens and scores equal the plain model's, for every draft_miss, one session alone (R = 1) and four in one round
// (R = 4); each poll commits at most 4; the counters are the toy's predicted ones (accept patterns 0..3 per round);
// a stop token inside a chain truncates it; max_new_tokens and the context limit shorten the chain; a cancel lands on
// a round boundary and the session continues as plain.
void test_spec_equals_plain() {
    for (const DraftDevice &dev : kDraftDevices) {
        const std::vector<uint32_t> prompt = {5, 9, 11};
        std::vector<Gen> gens = {{prompt, 13}, {prompt, 6}, {prompt, 2}, {prompt, 1}, {prompt, 20, {}, 0, 10}};
        const Outcome full = gen_run(dev.id, {prompt, 13}, false);
        // A stop token inside the second round's chain (its first occurrence).
        CHECK_EQ(first_index(full.out, full.out[6]), 6);
        gens.push_back({prompt, 13, {full.out[6]}});
        for (const Gen &g : gens) {
            const Outcome plain = gen_run(dev.id, g, false);
            const Outcome spec = gen_run(dev.id, g, true);
            check_same(spec, plain);
            const uint64_t limit = std::min(g.n, g.context - g.prompt.size());
            check_counts(spec, predicted(spec.c.committed_tokens, limit, dev.miss));
            CHECK_EQ(spec.bursts.at(0), 1);  // the PREFILL's scoring step
            for (const uint64_t b : spec.bursts) CHECK(b <= 4);
            if (dev.miss == 0 && g.n == 13 && g.stops.empty()) {
                CHECK(spec.bursts == std::vector<uint64_t>({1, 4, 4, 4}));
            }
        }
        CHECK_EQ(gen_run(dev.id, gens.back(), true).c.finish, CERE_FINISH_STOP_TOKEN);

        // R = 4: four drafting tickets share each round, each equal to its plain run.
        cere_executor *executor = spec_executor(dev.id);
        Desc d = rows_desc(4);
        cere_model *model = gen_model(executor, d, true);
        const std::vector<Gen> four = {{{1, 2}, 13}, {{3, 4, 5}, 9}, {{6}, 6}, {{7, 8, 9, 10}, 11}};
        std::vector<cere_session *> sessions(four.size(), nullptr);
        std::vector<Job> jobs;
        for (size_t i = 0; i < four.size(); ++i) {
            CHECK_EQ(cere_session_create(executor, model, 64, &sessions[i]), CERE_OK);
            jobs.push_back(submit_gen(executor, model, sessions[i], CERE_OP_PREFILL, four[i]));
        }
        const std::vector<Outcome> outs = finish_all(executor, jobs);
        for (size_t i = 0; i < four.size(); ++i) {
            check_same(outs[i], gen_run(dev.id, four[i], false));
            check_counts(outs[i], predicted(four[i].n, four[i].n, dev.miss));
            CHECK_EQ(cere_session_release(executor, sessions[i]), CERE_OK);
        }

        // A cancel after two rounds keeps what they committed; the session's next DECODE continues the plain run.
        cere_session *s = nullptr;
        CHECK_EQ(cere_session_create(executor, model, 64, &s), CERE_OK);
        Job job = submit_gen(executor, model, s, CERE_OP_PREFILL, {prompt, 13});
        Outcome o;
        CHECK(!poll_into(executor, job.ticket, *job.work, o));
        CHECK(!poll_into(executor, job.ticket, *job.work, o));
        const uint64_t kept = o.c.committed_tokens;
        CHECK_EQ(kept, 1 + std::min<uint64_t>(4, dev.miss == 0 ? 4 : dev.miss));
        CHECK_EQ(cere_executor_cancel(executor, job.ticket), CERE_OK);
        o = finish(executor, job.ticket, *job.work, o);
        CHECK_EQ(o.c.finish, CERE_FINISH_CANCELLED);
        CHECK_EQ(o.c.committed_tokens, kept);
        Job next = submit_gen(executor, model, s, CERE_OP_DECODE, {{}, 13 - kept});
        const Outcome rest = finish(executor, next.ticket, *next.work);
        CHECK(std::vector<uint32_t>(full.out.begin(), full.out.begin() + kept) == o.out);
        CHECK(std::vector<uint32_t>(full.out.begin() + kept, full.out.end()) == rest.out);
        check_counts(rest, predicted(13 - kept, 13 - kept, dev.miss, true, 0));
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
}

// --- ABI v5 slot selection (SelfTune ST2; registry crates/cere-runtime/src/levers.rs) -------------------------------

namespace {

using cere::host::Levers;

Levers make_levers(uint32_t k2_gemm = CERE_LEVER_K2_CUBLASLT, uint32_t prefill = CERE_LEVER_PREFILL_CUDA_CORE,
                   uint32_t decode = CERE_LEVER_DECODE_EAGER, uint32_t moe = CERE_LEVER_MOE_BUILD,
                   uint32_t draft_tokens = 0) {
    Levers l;
    l.k2_gemm = k2_gemm;
    l.prefill_attention = prefill;
    l.decode_step = decode;
    l.moe_gemm = moe;
    l.draft_tokens = draft_tokens;
    return l;
}

bool same_levers(const Levers &a, const Levers &b) {
    return a.k2_gemm == b.k2_gemm && a.prefill_attention == b.prefill_attention && a.decode_step == b.decode_step &&
           a.moe_gemm == b.moe_gemm && a.draft_tokens == b.draft_tokens && a.verify_attention == b.verify_attention &&
           a.k2_weights == b.k2_weights;
}

// `l` with k2_weights = `weights`.
Levers with_k2_weights(Levers l, uint32_t weights) {
    l.k2_weights = weights;
    return l;
}

// An empty selection block (every slot at its default), header set.
cere_slot_selection empty_selection() {
    cere_slot_selection s{};
    s.struct_size = sizeof(s);
    s.abi_version = CERE_EXECUTOR_ABI_VERSION;
    s.block_version = CERE_SLOT_BLOCK_VERSION;
    return s;
}

// Appends one choice to `s`.
void add_choice(cere_slot_selection &s, uint32_t slot, uint32_t impl, uint32_t param0 = 0) {
    cere_slot_choice &c = s.choices[s.count++];
    c.slot = slot;
    c.impl = impl;
    c.params[0] = param0;
}

// `l` as a selection block listing every slot, in reverse slot order (the order is free).
cere_slot_selection to_selection(const Levers &l) {
    cere_slot_selection s = empty_selection();
    add_choice(s, CERE_SLOT_K2_WEIGHTS, l.k2_weights);
    add_choice(s, CERE_SLOT_VERIFY_ATTENTION, l.verify_attention);
    add_choice(s, CERE_SLOT_DRAFT_TOKENS, CERE_LEVER_DRAFT_WINDOW, l.draft_tokens);
    add_choice(s, CERE_SLOT_MOE_GEMM, l.moe_gemm);
    add_choice(s, CERE_SLOT_DECODE_STEP, l.decode_step);
    add_choice(s, CERE_SLOT_PREFILL_ATTENTION, l.prefill_attention);
    add_choice(s, CERE_SLOT_K2_GEMM, l.k2_gemm);
    return s;
}

// cere_model_selection of `model` (CHECKed OK and well formed: every slot once, in slot order, the rest zero),
// decoded; the toy device's copy equals it (the device received exactly the host's active selection, at ALLOCATE or
// by the last set_selection).
Levers active_levers(cere_executor *executor, cere_model *model) {
    cere_slot_selection out{};
    std::memset(&out, 0xFF, sizeof(out));  // every field past the header is written
    out.struct_size = sizeof(out);
    out.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(cere_model_selection(executor, model, &out), CERE_OK);
    CHECK_EQ(out.struct_size, uint32_t(sizeof(out)));
    CHECK_EQ(out.abi_version, CERE_EXECUTOR_ABI_VERSION);
    CHECK_EQ(out.block_version, CERE_SLOT_BLOCK_VERSION);
    CHECK_EQ(out.count, CERE_SLOT_COUNT);
    const cere_slot_choice zero{};
    for (uint32_t i = CERE_SLOT_COUNT; i < CERE_SLOT_MAX; ++i) {
        CHECK(std::memcmp(&out.choices[i], &zero, sizeof zero) == 0);
    }
    for (uint32_t i = 0; i < CERE_SLOT_COUNT; ++i) {
        CHECK_EQ(out.choices[i].slot, i + 1);
        const uint32_t params = out.choices[i].slot == CERE_SLOT_DRAFT_TOKENS ? 1 : 0;
        for (uint32_t p = params; p < CERE_SLOT_PARAMS; ++p) CHECK_EQ(out.choices[i].params[p], 0u);
    }
    Levers l;
    l.k2_gemm = out.choices[CERE_SLOT_K2_GEMM - 1].impl;
    l.prefill_attention = out.choices[CERE_SLOT_PREFILL_ATTENTION - 1].impl;
    l.decode_step = out.choices[CERE_SLOT_DECODE_STEP - 1].impl;
    l.moe_gemm = out.choices[CERE_SLOT_MOE_GEMM - 1].impl;
    CHECK_EQ(out.choices[CERE_SLOT_DRAFT_TOKENS - 1].impl, CERE_LEVER_DRAFT_WINDOW);
    l.draft_tokens = out.choices[CERE_SLOT_DRAFT_TOKENS - 1].params[0];
    l.verify_attention = out.choices[CERE_SLOT_VERIFY_ATTENTION - 1].impl;
    l.k2_weights = out.choices[CERE_SLOT_K2_WEIGHTS - 1].impl;
    CHECK(l.verify_attention <= CERE_LEVER_VERIFY_BATCHED);
    CHECK(l.k2_weights <= CERE_LEVER_K2_WEIGHTS_FP8);
    CHECK(same_levers(toy(executor).levers(*model->device), l));
    return l;
}

// cere_model_create of `d` with `selection` (null: none) on `executor`: its status, the model released on success.
cere_status create_status(cere_executor *executor, Desc &d, const cere_slot_selection *selection) {
    d.desc.selection = selection;
    cere_model *model = nullptr;
    const cere_status st = cere_model_create(executor, &d.desc, &model);
    CHECK_EQ(st == CERE_OK, model != nullptr);
    if (model != nullptr) CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    d.desc.selection = nullptr;
    return st;
}

cere_status create_status(cere_executor *executor, Desc &d, const Levers &levers) {
    const cere_slot_selection s = to_selection(levers);
    return create_status(executor, d, &s);
}

cere_status set_levers(cere_executor *executor, cere_model *model, const Levers &levers) {
    const cere_slot_selection s = to_selection(levers);
    return cere_model_set_selection(executor, model, &s);
}

// `g` as one PREFILL on a fresh executor of `device`, its model with drafts and the desc's selection `levers`.
Outcome gen_run_levers(const std::string &device, const Gen &g, const Levers &levers) {
    cere_executor *executor = spec_executor(device);
    Desc d = rows_desc(1);
    set_drafts(d);
    const cere_slot_selection s = to_selection(levers);
    d.desc.selection = &s;
    cere_model *model = sealed_model(executor, d);
    CHECK(same_levers(active_levers(executor, model), levers));
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, g.context, &session), CERE_OK);
    Job job = submit_gen(executor, model, session, CERE_OP_PREFILL, g);
    const Outcome o = finish(executor, job.ticket, *job.work);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    return o;
}

}  // namespace

// ABI v5: cere_model_desc.selection is validated and resolved by cere_model_create and read back by
// cere_model_selection (null or an empty block = every slot at its default; a partial block leaves the unlisted slots
// at theirs); the toy device holds the same set. Refusals: a bad struct_size, block_version or count, a slot outside
// 1..CERE_SLOT_COUNT or listed twice, an impl no library knows, a non-zero param the impl does not have or a non-zero
// choice past `count` -> ARGUMENT; abi_version 4 -> UNSUPPORTED; K7 MoE on the toy (a p3b build) and a draft window
// on a model without drafts, below 2 or above its speculative_draft_tokens -> UNSUPPORTED. The desc's family other
// than CERE_FAMILY_QWEN4_EXP is UNSUPPORTED, its calibrate_windows past CERE_CALIBRATE_MTP ARGUMENT. A v4 caller: its
// 128-B desc -> ARGUMENT, an abi_version 4 desc -> UNSUPPORTED. cere_model_selection refuses a null or mis-headed
// `out` and an unknown model.
void test_levers_desc_round_trip() {
    CHECK_EQ(sizeof(cere_slot_choice), size_t(24));
    CHECK_EQ(sizeof(cere_slot_selection), size_t(400));
    CHECK_EQ(sizeof(cere_model_desc), size_t(136));
    CHECK_EQ(offsetof(cere_model_desc, selection), size_t(120));
    cere_executor *executor = make_executor();
    const Levers zeros = make_levers();

    // Null selection: the defaults; so does an empty block.
    Desc d = toy_desc();
    cere_model *model = sealed_model(executor, d);
    CHECK(same_levers(active_levers(executor, model), zeros));
    CHECK_EQ(model->plan.calibrate_windows, CERE_CALIBRATE_ALL);
    const cere_slot_selection empty = empty_selection();
    CHECK_EQ(create_status(executor, d, &empty), CERE_OK);

    // cere_model_selection refusals.
    CHECK_EQ(cere_model_selection(executor, model, nullptr), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_selection(nullptr, model, nullptr), CERE_ERR_ARGUMENT);
    cere_slot_selection out = empty_selection();
    out.struct_size = sizeof(out) - 4;
    CHECK_EQ(cere_model_selection(executor, model, &out), CERE_ERR_ARGUMENT);
    out = empty_selection();
    out.abi_version = 4;
    CHECK_EQ(cere_model_selection(executor, model, &out), CERE_ERR_UNSUPPORTED);
    out = empty_selection();
    int other = 0;
    CHECK_EQ(cere_model_selection(executor, reinterpret_cast<cere_model *>(&other), &out), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);

    // Every runnable selection round-trips (MoE BUILD and P3B: the toy is a p3b build).
    // Fp8W: the FP8 K2 weights are a create-time selection every library accepts (the toy stores it).
    for (const Levers &l : {make_levers(CERE_LEVER_K2_SMALL_M), make_levers(0, CERE_LEVER_PREFILL_P4),
                            make_levers(CERE_LEVER_K2_SMALL_M, CERE_LEVER_PREFILL_P4, 0, CERE_LEVER_MOE_P3B),
                            make_levers(0, 0, 0, CERE_LEVER_MOE_BUILD),
                            with_k2_weights(make_levers(), CERE_LEVER_K2_WEIGHTS_FP8),
                            with_k2_weights(make_levers(CERE_LEVER_K2_SMALL_M), CERE_LEVER_K2_WEIGHTS_FP8),
                            // K2Prefill P3: small_m_a8 is a selection every library accepts (MXFP8-only effect).
                            make_levers(CERE_LEVER_K2_SMALL_M_A8),
                            with_k2_weights(make_levers(CERE_LEVER_K2_SMALL_M_A8), CERE_LEVER_K2_WEIGHTS_FP8)}) {
        Desc dl = toy_desc();
        const cere_slot_selection s = to_selection(l);
        dl.desc.selection = &s;
        cere_model *m = sealed_model(executor, dl);
        CHECK(same_levers(active_levers(executor, m), l));
        CHECK_EQ(cere_model_release(executor, m), CERE_OK);
    }
    // A partial block: only the listed slot moves.
    {
        cere_slot_selection s = empty_selection();
        add_choice(s, CERE_SLOT_PREFILL_ATTENTION, CERE_LEVER_PREFILL_P4);
        Desc dl = toy_desc();
        dl.desc.selection = &s;
        cere_model *m = sealed_model(executor, dl);
        CHECK(same_levers(active_levers(executor, m), make_levers(0, CERE_LEVER_PREFILL_P4)));
        CHECK_EQ(cere_model_release(executor, m), CERE_OK);
    }
    // LoadCrit LC-01: the MTP-only calibration is a desc field every library accepts (on a model without drafts the
    // seal times nothing either way); it is not a slot, so the selection reads back the defaults.
    {
        Desc dl = toy_desc();
        dl.desc.calibrate_windows = CERE_CALIBRATE_MTP;
        cere_model *m = sealed_model(executor, dl);
        CHECK_EQ(m->plan.calibrate_windows, CERE_CALIBRATE_MTP);
        CHECK(same_levers(active_levers(executor, m), zeros));
        CHECK_EQ(cere_model_release(executor, m), CERE_OK);
        dl.desc.calibrate_windows = CERE_CALIBRATE_MTP + 1;
        CHECK_EQ(create_status(executor, dl, nullptr), CERE_ERR_ARGUMENT);
    }
    // ABI v5 Q5(a): the only family is CERE_FAMILY_QWEN4_EXP.
    {
        Desc dl = toy_desc();
        dl.desc.family = CERE_FAMILY_QWEN4_EXP + 1;
        CHECK_EQ(create_status(executor, dl, nullptr), CERE_ERR_UNSUPPORTED);
    }

    // Header, block and unknown values -> ARGUMENT; known but not runnable -> UNSUPPORTED.
    Desc bad = toy_desc();
    cere_slot_selection s = to_selection(zeros);
    s.struct_size = sizeof(s) + 4;
    CHECK_EQ(create_status(executor, bad, &s), CERE_ERR_ARGUMENT);
    s = to_selection(zeros);
    s.abi_version = 4;
    CHECK_EQ(create_status(executor, bad, &s), CERE_ERR_UNSUPPORTED);
    s = to_selection(zeros);
    s.block_version = CERE_SLOT_BLOCK_VERSION + 1;
    CHECK_EQ(create_status(executor, bad, &s), CERE_ERR_ARGUMENT);
    s = to_selection(zeros);
    s.count = CERE_SLOT_MAX + 1;
    CHECK_EQ(create_status(executor, bad, &s), CERE_ERR_ARGUMENT);
    s = to_selection(zeros);
    s.choices[CERE_SLOT_COUNT].impl = 1;  // a non-zero choice past `count`
    CHECK_EQ(create_status(executor, bad, &s), CERE_ERR_ARGUMENT);
    for (const uint32_t slot : {0u, CERE_SLOT_COUNT + 1}) {
        s = empty_selection();
        add_choice(s, slot, 0);
        CHECK_EQ(create_status(executor, bad, &s), CERE_ERR_ARGUMENT);
    }
    s = empty_selection();
    add_choice(s, CERE_SLOT_K2_GEMM, CERE_LEVER_K2_SMALL_M);
    add_choice(s, CERE_SLOT_K2_GEMM, CERE_LEVER_K2_SMALL_M);  // listed twice
    CHECK_EQ(create_status(executor, bad, &s), CERE_ERR_ARGUMENT);
    s = empty_selection();
    add_choice(s, CERE_SLOT_K2_GEMM, CERE_LEVER_K2_CUBLASLT, 1);  // a param k2_gemm does not have
    CHECK_EQ(create_status(executor, bad, &s), CERE_ERR_ARGUMENT);
    s = empty_selection();
    add_choice(s, CERE_SLOT_DRAFT_TOKENS, CERE_LEVER_DRAFT_WINDOW);
    s.choices[0].params[1] = 1;  // draft_tokens has one param
    CHECK_EQ(create_status(executor, bad, &s), CERE_ERR_ARGUMENT);
    s = empty_selection();
    add_choice(s, CERE_SLOT_DRAFT_TOKENS, CERE_LEVER_DRAFT_WINDOW + 1);
    CHECK_EQ(create_status(executor, bad, &s), CERE_ERR_ARGUMENT);
    Levers l = make_levers();
    l.verify_attention = CERE_LEVER_VERIFY_BATCHED + 1;
    CHECK_EQ(create_status(executor, bad, l), CERE_ERR_ARGUMENT);
    CHECK_EQ(create_status(executor, bad, with_k2_weights(make_levers(), CERE_LEVER_K2_WEIGHTS_FP8 + 1)),
             CERE_ERR_ARGUMENT);
    // P11R-19: the batched verify attention is a selection every library runs (a model without drafts never verifies).
    l = make_levers();
    l.verify_attention = CERE_LEVER_VERIFY_BATCHED;
    CHECK_EQ(create_status(executor, bad, l), CERE_OK);
    for (const Levers &u : {make_levers(CERE_LEVER_K2_TAIL_SPLIT + 1), make_levers(0, 2), make_levers(0, 0, 3),
                            make_levers(0, 0, 0, 3)}) {
        CHECK_EQ(create_status(executor, bad, u), CERE_ERR_ARGUMENT);
    }
    // VulkanAmd P4-B: k2_gemm tail_split is the Vulkan device's route only (the toy has none).
    for (const Levers &u : {make_levers(0, 0, 0, CERE_LEVER_MOE_K7), make_levers(CERE_LEVER_K2_TAIL_SPLIT),
                            make_levers(0, 0, 0, 0, 2), make_levers(0, 0, 0, 0, 3)}) {
        CHECK_EQ(create_status(executor, bad, u), CERE_ERR_UNSUPPORTED);  // no drafts: a draft window unsupported
    }
    // P1 landed (build/perf-int): the decode graph is a selection every library runs.
    CHECK_EQ(create_status(executor, bad, make_levers(0, 0, CERE_LEVER_DECODE_GRAPH)), CERE_OK);
    CHECK_EQ(create_status(executor, bad, make_levers(0, 0, CERE_LEVER_DECODE_SPEC_GRAPH)), CERE_OK);

    // A draft window on a drafting model (speculative_draft_tokens 4): 2..4 accepted, 1 and 5 UNSUPPORTED.
    Desc drafting = rows_desc(1);
    set_drafts(drafting);
    for (uint32_t t : {2u, 3u, 4u}) CHECK_EQ(create_status(executor, drafting, make_levers(0, 0, 0, 0, t)), CERE_OK);
    for (uint32_t t : {1u, 5u}) {
        CHECK_EQ(create_status(executor, drafting, make_levers(0, 0, 0, 0, t)), CERE_ERR_UNSUPPORTED);
    }

    // A v4 caller: its 128-byte desc is a struct_size mismatch, a v4 stamp on the v5 size is UNSUPPORTED.
    Desc v4 = toy_desc();
    v4.desc.struct_size = 128;
    CHECK_EQ(create_status(executor, v4, nullptr), CERE_ERR_ARGUMENT);
    v4.desc.struct_size = sizeof(cere_model_desc);
    v4.desc.abi_version = 4;
    CHECK_EQ(create_status(executor, v4, nullptr), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// ABI v5 hot swap: cere_model_set_selection is CERE_ERR_BUSY while any ticket of the executor is in flight (its own
// model's or another's), until the terminal poll returned it, then replaces the active selection (read back by
// cere_model_selection and the device; an unlisted slot returns to its default); it refuses an unsealed or unknown
// model and a null selection (ARGUMENT) and validates as the desc's selection does; on every error the active
// selection is unchanged.
void test_set_levers_busy_and_validation() {
    cere_executor *executor = make_executor();
    const Levers zeros = make_levers();
    const Levers fast = make_levers(CERE_LEVER_K2_SMALL_M, CERE_LEVER_PREFILL_P4, CERE_LEVER_DECODE_GRAPH);

    Desc unsealed_desc = toy_desc();
    cere_model *unsealed = nullptr;
    CHECK_EQ(cere_model_create(executor, &unsealed_desc.desc, &unsealed), CERE_OK);
    CHECK_EQ(set_levers(executor, unsealed, fast), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_release(executor, unsealed), CERE_OK);

    Desc da = rows_desc(1);
    cere_model *a = sealed_model(executor, da);
    Desc db = rows_desc(1);
    cere_model *b = sealed_model(executor, db);
    const cere_slot_selection fast_sel = to_selection(fast);
    CHECK_EQ(cere_model_set_selection(nullptr, a, &fast_sel), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_set_selection(executor, a, nullptr), CERE_ERR_ARGUMENT);
    int other = 0;
    CHECK_EQ(cere_model_set_selection(executor, reinterpret_cast<cere_model *>(&other), &fast_sel), CERE_ERR_ARGUMENT);

    // A ticket of `b` in flight: `a` and `b` are both BUSY, unchanged; after the terminal poll both swap.
    cere_session *s = nullptr;
    CHECK_EQ(cere_session_create(executor, b, 64, &s), CERE_OK);
    Job job = submit_job(executor, b, s, CERE_OP_PREFILL, {1, 2, 3}, 4);
    CHECK_EQ(set_levers(executor, a, fast), CERE_ERR_BUSY);
    CHECK_EQ(set_levers(executor, b, fast), CERE_ERR_BUSY);
    CHECK(same_levers(active_levers(executor, a), zeros));
    CHECK(same_levers(active_levers(executor, b), zeros));
    const cere_completion c = drain(executor, job.ticket);
    CHECK_EQ(c.status, CERE_OK);
    CHECK_EQ(c.committed_tokens, 4u);
    CHECK_EQ(set_levers(executor, a, fast), CERE_OK);
    CHECK_EQ(set_levers(executor, b, fast), CERE_OK);
    CHECK(same_levers(active_levers(executor, a), fast));
    CHECK(same_levers(active_levers(executor, b), fast));

    // Validation as the desc's; the active selection stays `fast` after each refusal.
    cere_slot_selection sel = to_selection(zeros);
    sel.struct_size = 24;
    CHECK_EQ(cere_model_set_selection(executor, a, &sel), CERE_ERR_ARGUMENT);
    sel = to_selection(zeros);
    sel.abi_version = 4;
    CHECK_EQ(cere_model_set_selection(executor, a, &sel), CERE_ERR_UNSUPPORTED);
    sel = to_selection(zeros);
    sel.block_version = 0;
    CHECK_EQ(cere_model_set_selection(executor, a, &sel), CERE_ERR_ARGUMENT);
    Levers l = make_levers();
    l.verify_attention = CERE_LEVER_VERIFY_BATCHED + 1;
    CHECK_EQ(set_levers(executor, a, l), CERE_ERR_ARGUMENT);
    for (const Levers &u : {make_levers(CERE_LEVER_K2_TAIL_SPLIT + 1), make_levers(0, 2), make_levers(0, 0, 3),
                            make_levers(0, 0, 0, 3)}) {
        CHECK_EQ(set_levers(executor, a, u), CERE_ERR_ARGUMENT);
    }
    for (const Levers &u : {make_levers(0, 0, 0, CERE_LEVER_MOE_K7), make_levers(CERE_LEVER_K2_TAIL_SPLIT),
                            make_levers(0, 0, 0, 0, 2)}) {
        CHECK_EQ(set_levers(executor, a, u), CERE_ERR_UNSUPPORTED);
    }
    // SelfTune ST2 (c) (D6): the block carries launch params only. A param on an impl that has none (k2_gemm's
    // build-time k_step, say) is ARGUMENT; a draft window on this model without drafts (any window: none is in
    // range) is UNSUPPORTED. Both change nothing.
    sel = empty_selection();
    add_choice(sel, CERE_SLOT_K2_GEMM, CERE_LEVER_K2_SMALL_M, 64);
    CHECK_EQ(cere_model_set_selection(executor, a, &sel), CERE_ERR_ARGUMENT);
    for (const uint32_t window : {1u, 2u, 5u}) {
        sel = empty_selection();
        add_choice(sel, CERE_SLOT_DRAFT_TOKENS, CERE_LEVER_DRAFT_WINDOW, window);
        CHECK_EQ(cere_model_set_selection(executor, a, &sel), CERE_ERR_UNSUPPORTED);
    }
    CHECK(same_levers(active_levers(executor, a), fast));
    const Levers spec_graph = make_levers(0, 0, CERE_LEVER_DECODE_SPEC_GRAPH);
    CHECK_EQ(set_levers(executor, a, spec_graph), CERE_OK);
    CHECK(same_levers(active_levers(executor, a), spec_graph));
    CHECK_EQ(set_levers(executor, a, fast), CERE_OK);
    // A partial block: every unlisted slot returns to its default.
    sel = empty_selection();
    add_choice(sel, CERE_SLOT_DECODE_STEP, CERE_LEVER_DECODE_GRAPH);
    CHECK_EQ(cere_model_set_selection(executor, a, &sel), CERE_OK);
    CHECK(same_levers(active_levers(executor, a), make_levers(0, 0, CERE_LEVER_DECODE_GRAPH)));

    // Back to the defaults (an empty block); the model still serves (a request after the swaps commits as before).
    const cere_slot_selection empty = empty_selection();
    CHECK_EQ(cere_model_set_selection(executor, b, &empty), CERE_OK);
    CHECK(same_levers(active_levers(executor, b), zeros));
    Job next = submit_job(executor, b, s, CERE_OP_DECODE, {}, 2);
    CHECK_EQ(drain(executor, next.ticket).committed_tokens, 2u);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, a), CERE_OK);
    CHECK_EQ(cere_model_release(executor, b), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// Fp8W: k2_weights is fixed at cere_model_create (INSTANTIATE converts the weights in place), so set_selection refuses
// a value other than the model's active one with CERE_ERR_UNSUPPORTED (an unlisted k2_weights slot is its default,
// BF16) and an unknown one with CERE_ERR_ARGUMENT; the active selection is unchanged after each refusal, and every hot
// slot still swaps with the model's own k2_weights.
void test_set_levers_k2_weights_restart_only() {
    cere_executor *executor = make_executor();
    const Levers fast = make_levers(CERE_LEVER_K2_SMALL_M, CERE_LEVER_PREFILL_P4, CERE_LEVER_DECODE_GRAPH);
    for (const uint32_t weights : {CERE_LEVER_K2_WEIGHTS_BF16, CERE_LEVER_K2_WEIGHTS_FP8}) {
        const uint32_t flipped =
            weights == CERE_LEVER_K2_WEIGHTS_FP8 ? CERE_LEVER_K2_WEIGHTS_BF16 : CERE_LEVER_K2_WEIGHTS_FP8;
        const Levers created = with_k2_weights(make_levers(), weights);
        const cere_slot_selection created_sel = to_selection(created);
        Desc d = rows_desc(1);
        d.desc.selection = &created_sel;
        cere_model *m = sealed_model(executor, d);
        CHECK(same_levers(active_levers(executor, m), created));

        CHECK_EQ(set_levers(executor, m, with_k2_weights(make_levers(), flipped)), CERE_ERR_UNSUPPORTED);
        CHECK_EQ(set_levers(executor, m, with_k2_weights(fast, flipped)), CERE_ERR_UNSUPPORTED);
        CHECK_EQ(set_levers(executor, m, with_k2_weights(make_levers(), CERE_LEVER_K2_WEIGHTS_FP8 + 1)),
                 CERE_ERR_ARGUMENT);
        const cere_slot_selection empty = empty_selection();
        CHECK_EQ(cere_model_set_selection(executor, m, &empty),
                 weights == CERE_LEVER_K2_WEIGHTS_BF16 ? CERE_OK : CERE_ERR_UNSUPPORTED);
        CHECK(same_levers(active_levers(executor, m), created));

        const Levers swapped = with_k2_weights(fast, weights);
        CHECK_EQ(set_levers(executor, m, swapped), CERE_OK);
        CHECK(same_levers(active_levers(executor, m), swapped));
        CHECK_EQ(set_levers(executor, m, created), CERE_OK);
        CHECK(same_levers(active_levers(executor, m), created));

        // The model still serves after the refusals and swaps.
        cere_session *s = nullptr;
        CHECK_EQ(cere_session_create(executor, m, 64, &s), CERE_OK);
        Job job = submit_job(executor, m, s, CERE_OP_PREFILL, {1, 2, 3}, 4);
        CHECK_EQ(drain(executor, job.ticket).committed_tokens, 4u);
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        CHECK_EQ(cere_model_release(executor, m), CERE_OK);
    }
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// LoadCrit LC-01: calibrate_windows is a desc field fixed at cere_model_create (the seal measured the cost tables
// once), not a slot: cere_model_set_selection swaps every slot and the model keeps its own calibrate_windows.
void test_set_levers_calibrate_windows_kept() {
    cere_executor *executor = make_executor();
    const Levers fast = make_levers(CERE_LEVER_K2_SMALL_M, CERE_LEVER_PREFILL_P4, CERE_LEVER_DECODE_GRAPH);
    for (const uint32_t windows : {CERE_CALIBRATE_ALL, CERE_CALIBRATE_MTP}) {
        Desc d = rows_desc(1);
        set_drafts(d);
        d.desc.calibrate_windows = windows;
        cere_model *m = sealed_model(executor, d);
        CHECK_EQ(m->plan.calibrate_windows, windows);
        CHECK_EQ(set_levers(executor, m, fast), CERE_OK);
        CHECK(same_levers(active_levers(executor, m), fast));
        CHECK_EQ(m->plan.calibrate_windows, windows);
        CHECK_EQ(set_levers(executor, m, make_levers()), CERE_OK);
        CHECK_EQ(m->plan.calibrate_windows, windows);

        cere_session *s = nullptr;
        CHECK_EQ(cere_session_create(executor, m, 64, &s), CERE_OK);
        Job job = submit_job(executor, m, s, CERE_OP_PREFILL, {1, 2, 3}, 4);
        CHECK_EQ(drain(executor, job.ticket).committed_tokens, 4u);
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        CHECK_EQ(cere_model_release(executor, m), CERE_OK);
    }
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// ABI v5 draft_tokens slot (registry draft_tokens "3"/"2", contract test of the lever): on the stateful toy, for
// every draft_miss, a drafting model whose verify window is capped at the window param (2 or 3; 4 = the model's
// speculative_draft_tokens, the default's behaviour) commits plain decode's greedy tokens and scores bit for bit, and
// each round drafts at most window - 1 (the toy's predicted counters with that chain; every poll commits at most
// window). A hot swap between requests (cere_model_set_selection) takes effect at the next request's rounds, and
// window 0 restores the uncapped window.
void test_draft_tokens_lever_greedy_equals_plain() {
    for (const DraftDevice &dev : kDraftDevices) {
        const std::vector<uint32_t> prompt = {5, 9, 11};
        const Outcome full = gen_run(dev.id, {prompt, 13}, false);
        const std::vector<Gen> gens = {{prompt, 13}, {prompt, 9}, {prompt, 6}, {prompt, 2}, {prompt, 1},
                                       {prompt, 20, {}, 0, 10}, {prompt, 13, {full.out[6]}}};
        for (uint32_t window : {2u, 3u, 4u}) {
            for (const Gen &g : gens) {
                const Outcome plain = gen_run(dev.id, g, false);
                const Outcome spec = gen_run_levers(dev.id, g, make_levers(0, 0, 0, 0, window));
                check_same(spec, plain);
                const uint64_t limit = std::min(g.n, g.context - g.prompt.size());
                check_counts(spec, predicted(spec.c.committed_tokens, limit, dev.miss, true, 1, window - 1));
                CHECK_EQ(spec.bursts.at(0), 1);
                for (const uint64_t b : spec.bursts) CHECK(b <= window);
                if (window == 4) check_counts(spec, predicted(spec.c.committed_tokens, limit, dev.miss));
            }
        }
        // The RustLevers FFI case (native_ffi.rs): window 3, a 9-token PREFILL, draft_miss 0: 3 rounds, 5
        // accepted drafts, plain's tokens.
        if (dev.miss == 0) {
            const Outcome nine = gen_run_levers(dev.id, {{1, 2, 3}, 9}, make_levers(0, 0, 0, 0, 3));
            check_same(nine, gen_run(dev.id, {{1, 2, 3}, 9}, false));
            check_counts(nine, {3, 5});
        }

        // Hot swap on one sealed drafting model: 0 -> 2 -> 0, each request equal to plain with its window's counts.
        cere_executor *executor = spec_executor(dev.id);
        Desc d = rows_desc(1);
        cere_model *model = gen_model(executor, d, true);
        const Gen g = {prompt, 13};
        const Outcome plain = gen_run(dev.id, g, false);
        for (uint32_t window : {0u, 2u, 0u, 3u}) {
            CHECK_EQ(set_levers(executor, model, make_levers(0, 0, 0, 0, window)), CERE_OK);
            cere_session *s = nullptr;
            CHECK_EQ(cere_session_create(executor, model, g.context, &s), CERE_OK);
            Job job = submit_gen(executor, model, s, CERE_OP_PREFILL, g);
            const Outcome o = finish(executor, job.ticket, *job.work);
            check_same(o, plain);
            const uint64_t chain = window == 0 ? 3 : window - 1;
            check_counts(o, predicted(o.c.committed_tokens, g.n, dev.miss, true, 1, chain));
            CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        }
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
}

// C2 (I4, D1): VERIFY with caller drafts commits the plain prefix they match plus the plain correction (all correct:
// every draft and the bonus); the next DECODE equals the plain run (the rejected drafts never reached the stateful
// toy) and its first round uses fresh proposals (VERIFY ran draft-extend). Refusals: drafts 0 or >= 4, an id outside
// the vocab, flag bit 0 or 1, a vocab mask, a non-drafting (NO_SPECULATION) session, an empty or mid-prompt session
// -> ARGUMENT; a stale revision -> STALE_EPOCH; a model without drafts -> UNSUPPORTED.
void test_verify_forced() {
    const std::string device = "toy,stateful=1";
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    const std::vector<uint32_t> prompt = {4, 8, 15};
    const Outcome full = gen_run(device, {prompt, 16}, false);
    cere_executor *executor = spec_executor(device);
    Desc d = rows_desc(2);
    cere_model *model = gen_model(executor, d, true);
    struct Case {
        uint32_t drafts;
        int wrong;  // index of the wrong draft, -1 = none
    };
    for (const Case k : {Case{3, -1}, Case{3, 0}, Case{3, 1}, Case{3, 2}, Case{1, -1}, Case{2, -1}, Case{2, 1}}) {
        cere_session *s = prefilled_session(executor, model, prompt);
        std::vector<uint32_t> drafts(full.out.begin() + 1, full.out.begin() + 1 + k.drafts);
        if (k.wrong >= 0) drafts[k.wrong] = (drafts[k.wrong] + 1) % vocab;
        Job v = submit_gen(executor, model, s, CERE_OP_VERIFY, {drafts, 4});
        const Outcome vo = finish(executor, v.ticket, *v.work);
        const uint64_t a = k.wrong >= 0 ? uint64_t(k.wrong) + 1 : k.drafts + 1;
        CHECK_EQ(vo.c.finish, CERE_FINISH_LENGTH);
        CHECK_EQ(vo.c.status, CERE_OK);
        CHECK_EQ(vo.c.committed_tokens, a);
        CHECK_EQ(vo.c.verify_steps, 1);
        CHECK_EQ(vo.c.accepted_drafts, a - 1);
        CHECK(vo.out == std::vector<uint32_t>(full.out.begin() + 1, full.out.begin() + 1 + a));
        // The next DECODE: the plain continuation; its first round accepts 3 fresh proposals.
        Job next = submit_gen(executor, model, s, CERE_OP_DECODE, {{}, 8});
        Outcome no;
        CHECK(!poll_into(executor, next.ticket, *next.work, no));
        CHECK_EQ(no.c.committed_tokens, 4);
        CHECK_EQ(no.c.accepted_drafts, 3);
        no = finish(executor, next.ticket, *next.work, no);
        CHECK(no.out == std::vector<uint32_t>(full.out.begin() + 1 + a, full.out.begin() + 9 + a));
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    }
    // A VERIFY of max_new_tokens 2 runs one draft of 3 (chain shortened to the budget).
    {
        cere_session *s = prefilled_session(executor, model, prompt);
        Job v = submit_gen(executor, model, s, CERE_OP_VERIFY, {{full.out[1], full.out[2], full.out[3]}, 2});
        const Outcome vo = finish(executor, v.ticket, *v.work);
        CHECK(vo.out == std::vector<uint32_t>(full.out.begin() + 1, full.out.begin() + 3));
        CHECK_EQ(vo.c.accepted_drafts, 1);
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    }

    // Refusals on a decode-ready drafting session (nothing submitted, the session unchanged).
    cere_session *s = prefilled_session(executor, model, prompt);
    const uint64_t generation = generation_of(executor, model);
    const auto verify = [&](cere_session *session, std::vector<uint32_t> drafts, uint32_t flags = 0,
                            uint64_t revision_bump = 0, const uint32_t *mask = nullptr) {
        Work w(session, CERE_OP_VERIFY, std::move(drafts), 4, generation, {}, session->revision + revision_bump);
        w.item.flags = flags;
        const std::vector<uint32_t> words(mask_words(), ~0u);
        cere_sampling_params sampling = sampling_params(0.0f, vocab);
        if (mask != nullptr) {
            w.item.sampling = &sampling;
            w.item.vocab_mask = words.data();
            w.item.vocab_mask_words = words.size();
        }
        cere_completion_ticket ticket{};
        const cere_status st = cere_executor_submit(executor, &w.batch, &ticket);
        if (st == CERE_OK) drain(executor, ticket);
        return st;
    };
    const std::vector<uint32_t> good = {full.out[1]};
    const uint64_t revision = s->revision;
    CHECK_EQ(verify(s, {}), CERE_ERR_ARGUMENT);
    CHECK_EQ(verify(s, {full.out[1], full.out[2], full.out[3], full.out[4]}), CERE_ERR_ARGUMENT);
    CHECK_EQ(verify(s, {vocab}), CERE_ERR_ARGUMENT);
    CHECK_EQ(verify(s, good, CERE_WORK_IGNORE_STOP_TOKENS), CERE_ERR_ARGUMENT);
    CHECK_EQ(verify(s, good, CERE_WORK_NO_SPECULATION), CERE_ERR_ARGUMENT);
    CHECK_EQ(verify(s, good, CERE_WORK_LOOKUP_DRAFTS), CERE_ERR_ARGUMENT);  // M2.9L D5
    CHECK_EQ(verify(s, good, CERE_WORK_NO_MTP_DRAFTS), CERE_ERR_ARGUMENT);
    CHECK_EQ(verify(s, good, CERE_WORK_ADAPTIVE_DRAFTS), CERE_ERR_ARGUMENT);  // m29a design note A2
    CHECK_EQ(verify(s, good, CERE_WORK_TAIL_DRAFTS), CERE_ERR_ARGUMENT);  // DraftAccept
    CHECK_EQ(verify(s, good, CERE_WORK_LONG_DRAFTS), CERE_ERR_ARGUMENT);  // TreeSpec
    CHECK_EQ(verify(s, good, CERE_WORK_CHAIN_DRAFTS), CERE_ERR_ARGUMENT);  // Chains
    CHECK_EQ(verify(s, good, 0, 0, &good[0]), CERE_ERR_ARGUMENT);
    CHECK_EQ(verify(s, good, 0, 1), CERE_ERR_STALE_EPOCH);
    CHECK_EQ(s->tokens.size(), prompt.size() + 1);
    CHECK_EQ(s->revision, revision);
    // An empty session, and a session whose PREFILL carried NO_SPECULATION (not drafting).
    cere_session *empty = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &empty), CERE_OK);
    CHECK_EQ(verify(empty, good), CERE_ERR_ARGUMENT);
    Job nospec = submit_gen(executor, model, empty, CERE_OP_PREFILL, {prompt, 1, {}, CERE_WORK_NO_SPECULATION});
    CHECK_EQ(finish(executor, nospec.ticket, *nospec.work).c.finish, CERE_FINISH_LENGTH);
    CHECK(!empty->drafting);
    CHECK_EQ(verify(empty, good), CERE_ERR_ARGUMENT);
    CHECK_EQ(verify(s, good), CERE_OK);  // the same item on the drafting session runs
    CHECK_EQ(cere_session_release(executor, empty), CERE_OK);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    // A mid-prompt session (its chunked PREFILL cancelled after one chunk).
    executor = spec_executor("toy,prefill_chunk=2");
    Desc dc = rows_desc(2);
    model = gen_model(executor, dc, true);
    CHECK_EQ(cere_session_create(executor, model, 64, &s), CERE_OK);
    Job chunked = submit_gen(executor, model, s, CERE_OP_PREFILL, {{1, 2, 3, 4, 5, 6}, 4});
    Outcome co;
    CHECK(!poll_into(executor, chunked.ticket, *chunked.work, co));
    CHECK_EQ(cere_executor_cancel(executor, chunked.ticket), CERE_OK);
    CHECK_EQ(finish(executor, chunked.ticket, *chunked.work, co).c.finish, CERE_FINISH_CANCELLED);
    CHECK(s->fed < s->prompt_tokens);
    CHECK_EQ(verify(s, good), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    // A model without drafts.
    executor = spec_executor(device);
    Desc dp = rows_desc(2);
    model = gen_model(executor, dp, false);
    s = prefilled_session(executor, model, prompt);
    CHECK_EQ(verify(s, good), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// C3L (m29l design note D2, D4, D5): tickets with CERE_WORK_LOOKUP_DRAFTS (bit 2) and/or CERE_WORK_NO_MTP_DRAFTS
// (bit 3) commit exactly the plain model's greedy tokens and scores, on the periodic toy (",cycle=5": the suffix lookup
// matches and its drafts are accepted) and on the stateful toy (a consumed wrong lookup draft would change every later
// token), for a periodic and a plain prompt, with MTP proposals always right or always wrong (draft_miss=1). With the
// MTP always wrong the lookup drafts are what is accepted; bit 3 alone verifies no drafts at all; the counters keep
// meaning rounds and accepted drafts. A DECODE after a PREFILL continues with the same session's drafter and policy.
void test_spec_lookup_equals_plain() {
    const std::vector<uint32_t> periodic = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3};
    const std::vector<uint32_t> plain_prompt = {5, 9, 11};
    constexpr uint32_t kLookup = CERE_WORK_LOOKUP_DRAFTS;
    constexpr uint32_t kNoMtp = CERE_WORK_NO_MTP_DRAFTS;
    for (const char *dev : {"toy,cycle=5", "toy,cycle=5,draft_miss=1", "toy,stateful=1", "toy,stateful=1,draft_miss=1",
                            "toy,stateful=1,cycle=7"}) {
        for (const std::vector<uint32_t> &prompt : {periodic, plain_prompt}) {
            const Outcome plain = gen_run(dev, {prompt, 40}, false);
            for (const uint32_t flags : {kLookup, kLookup | kNoMtp, kNoMtp}) {
                const Outcome spec = gen_run(dev, {prompt, 40, {}, flags}, true);
                check_same(spec, plain);
                CHECK(spec.c.verify_steps >= 1);
                for (const uint64_t b : spec.bursts) CHECK(b <= 4);
                if (flags == kNoMtp) {  // no drafts: one token per round after the PREFILL's scoring step
                    CHECK_EQ(spec.c.verify_steps, 39);
                    CHECK_EQ(spec.c.accepted_drafts, 0);
                }
            }
        }
    }
    // The periodic toy with every MTP chain wrong at its first draft: MTP alone accepts nothing; with bit 2 (and
    // with bits 2 + 3) the lookup's drafts are accepted, so fewer rounds commit the same 40 tokens.
    for (const std::vector<uint32_t> &prompt : {periodic, plain_prompt}) {
        const char *dev = "toy,cycle=5,draft_miss=1";
        const Outcome mtp = gen_run(dev, {prompt, 40}, true);
        CHECK_EQ(mtp.c.accepted_drafts, 0);
        for (const uint32_t flags : {kLookup, kLookup | kNoMtp}) {
            const Outcome spec = gen_run(dev, {prompt, 40, {}, flags}, true);
            check_same(spec, gen_run(dev, {prompt, 40}, false));
            CHECK(spec.c.accepted_drafts >= 20);
            CHECK(spec.c.verify_steps < mtp.c.verify_steps);
        }
    }
    // A PREFILL then a DECODE on one session, both with bit 2: the DECODE continues the plain run.
    const char *dev = "toy,cycle=5,draft_miss=2";
    const Outcome full = gen_run(dev, {periodic, 30}, false);
    cere_executor *executor = spec_executor(dev);
    Desc d = rows_desc(1);
    cere_model *model = gen_model(executor, d, true);
    cere_session *s = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &s), CERE_OK);
    Job first = submit_gen(executor, model, s, CERE_OP_PREFILL, {periodic, 10, {}, kLookup});
    const Outcome a = finish(executor, first.ticket, *first.work);
    Job next = submit_gen(executor, model, s, CERE_OP_DECODE, {{}, 20, {}, kLookup});
    const Outcome b = finish(executor, next.ticket, *next.work);
    CHECK(std::vector<uint32_t>(full.out.begin(), full.out.begin() + 10) == a.out);
    CHECK(std::vector<uint32_t>(full.out.begin() + 10, full.out.end()) == b.out);
    CHECK(b.c.accepted_drafts > 0);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// m29a design note A4: `g` as one PREFILL on a fresh drafting executor of `device`, with the drafts its session's
// rounds verified (cere_session::verified_drafts).
std::pair<Outcome, uint64_t> gen_drafts(const std::string &device, const Gen &g) {
    cere_executor *executor = spec_executor(device);
    Desc d = rows_desc(1, std::max<uint64_t>(64, g.context));
    cere_model *model = gen_model(executor, d, true);
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, g.context, &session), CERE_OK);
    Job job = submit_gen(executor, model, session, CERE_OP_PREFILL, g);
    const Outcome o = finish(executor, job.ticket, *job.work);
    const uint64_t verified = session->verified_drafts;
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    return {o, verified};
}

// m29a design note A2, A4: CERE_WORK_ADAPTIVE_DRAFTS (bit 4, alone or with the lookup bits) commits exactly the plain
// model's greedy tokens and scores on the toys whose MTP proposals are always right, wrong at a chain position or
// stateful; the fixed policy (flags 0) verifies all 3 proposals every full round, unchanged. With the proposals always
// right the adaptive policy starts at 2 drafts (the prior) and every verified draft is accepted; with every chain
// wrong at its first draft it verifies fewer drafts than the fixed policy, and a rerun picks the same windows.
void test_spec_adaptive_equals_plain() {
    const std::vector<uint32_t> periodic = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3};
    const std::vector<uint32_t> plain_prompt = {5, 9, 11};
    constexpr uint32_t kAdaptive = CERE_WORK_ADAPTIVE_DRAFTS;
    constexpr uint32_t kLookup = CERE_WORK_LOOKUP_DRAFTS;
    for (const char *dev : {"toy", "toy,draft_miss=1", "toy,draft_miss=2", "toy,draft_miss=3", "toy,stateful=1",
                            "toy,stateful=1,draft_miss=1", "toy,cycle=5,draft_miss=1", "toy,stateful=1,cycle=7"}) {
        for (const std::vector<uint32_t> &prompt : {periodic, plain_prompt}) {
            const Outcome plain = gen_run(dev, {prompt, 40}, false);
            // DraftAccept: bit 5 on top (the toy's chains stay at 3 drafts, max_chain_drafts' default) changes nothing.
            for (const uint32_t flags :
                 {0u, kAdaptive, kAdaptive | kLookup, kAdaptive | kLookup | CERE_WORK_TAIL_DRAFTS}) {
                const auto [spec, verified] = gen_drafts(dev, {prompt, 40, {}, flags});
                check_same(spec, plain);
                CHECK(spec.c.verify_steps >= 1);
                CHECK(verified <= 3 * spec.c.verify_steps);
                CHECK(spec.c.accepted_drafts <= verified);
                for (const uint64_t b : spec.bursts) CHECK(b <= 4);
            }
        }
    }
    // Fixed: after the PREFILL's token, 9 rounds of 3 accepted drafts + 1 and a last round whose room the commit limit
    // cuts to 2 drafts.
    const auto [fixed, fixed_verified] = gen_drafts("toy", {plain_prompt, 40});
    CHECK_EQ(fixed.c.verify_steps, 10);
    CHECK_EQ(fixed_verified, 29);
    const auto [right, right_verified] = gen_drafts("toy", {plain_prompt, 40, {}, kAdaptive});
    CHECK_EQ(right.c.accepted_drafts, right_verified);
    CHECK(right_verified >= 2 * (right.c.verify_steps - 1));
    CHECK(right.c.verify_steps <= 13);
    // Every chain wrong at its first draft: the adaptive policy stops paying for windows that are always rejected.
    const auto [miss_fixed, miss_fixed_verified] = gen_drafts("toy,draft_miss=1", {plain_prompt, 40});
    const auto [miss, miss_verified] = gen_drafts("toy,draft_miss=1", {plain_prompt, 40, {}, kAdaptive});
    CHECK_EQ(miss_fixed.c.accepted_drafts, 0);
    CHECK_EQ(miss.c.accepted_drafts, 0);
    CHECK(2 * miss_verified < miss_fixed_verified);
    const auto [again, again_verified] = gen_drafts("toy,draft_miss=1", {plain_prompt, 40, {}, kAdaptive});
    CHECK_EQ(again_verified, miss_verified);
    CHECK_EQ(again.c.verify_steps, miss.c.verify_steps);
}

// O95 / m23 design note SC-01: on a device that times its rounds (the toy's timed_rounds=1), cere_model_seal
// measures the K2 table of the plan's own k2_gemm lever for a drafting model (host/spec_calibrate.h), with chain
// windows up to the device's longest chain (when the draft_tokens window allows a tail) whether or not its output is
// periodic (the chain rounds draft the recorded greedy stream, not lookups), and
// keeps the model's levers; its calibration sessions are gone (the model's sessions can still take every slot), and
// adaptive greedy tickets still commit plain decode's tokens. The adaptive tickets are priced by the model's table:
// one under which no draft pays makes them verify none. Without timed rounds, without drafts, or with a context the
// calibration session does not fit, the model keeps SpecCosts::untimed().
void test_spec_round_costs_seal() {
    using cere::host::SpecCosts;
    const auto is_untimed = [](const cere_model &m) {
        for (const SpecCosts &c : m.round_costs) {
            if (c.table() != SpecCosts::untimed().table()) return false;
        }
        return true;
    };
    const auto measured = [](const cere_model &m) {
        for (const SpecCosts &c : m.round_costs) {
            for (const double v : c.table()) {
                if (!(v > 0.0)) return false;
            }
        }
        return m.round_costs_measured;
    };
    struct Case {
        const char *device;
        bool drafts;
        uint64_t context;
        bool measured;
    };
    for (const Case &k : {Case{"toy", true, 2048, false}, Case{"toy,timed_rounds=1", false, 2048, false},
                          Case{"toy,timed_rounds=1", true, 64, false}, Case{"toy,timed_rounds=1", true, 2048, true},
                          Case{"toy,timed_rounds=1,chain=8,cycle=7", true, 2048, true},
                          Case{"toy,timed_rounds=1,chain=8", true, 2048, true}}) {
        cere_executor *executor = spec_executor(k.device);
        Desc d = rows_desc(2, k.context);
        cere_model *model = gen_model(executor, d, k.drafts);
        CHECK_EQ(measured(*model), k.measured);
        CHECK_EQ(is_untimed(*model), !k.measured);
        CHECK_EQ(model->plan.levers.k2_gemm, CERE_LEVER_K2_CUBLASLT);
        std::vector<cere_session *> sessions(2, nullptr);
        for (cere_session *&s : sessions) CHECK_EQ(cere_session_create(executor, model, k.context, &s), CERE_OK);
        for (cere_session *s : sessions) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
    // Only what the levers can pick is timed: the plan's own k2_gemm variant (the other table stays untimed()) and
    // chain windows past kMtpWindows only when the draft_tokens window leaves room for a DraftTail tail (window >= 4)
    // and calibrate_windows is ALL (LoadCrit LC-01: MTP times windows 1..4 only and prices 5..16 from window 4 by the
    // prior shape); every round is timed eager whatever the decode_step lever (no graph is instantiated at seal), and
    // the model keeps its levers.
    struct Pick {
        uint32_t k2_gemm;
        uint32_t draft_tokens;
        uint32_t decode;
        uint32_t calibrate;
        uint32_t deepest;  // the longest window timed
    };
    for (const Pick &p : {Pick{CERE_LEVER_K2_CUBLASLT, 0, CERE_LEVER_DECODE_EAGER, CERE_CALIBRATE_ALL, 9},
                          Pick{CERE_LEVER_K2_SMALL_M, 0, CERE_LEVER_DECODE_SPEC_GRAPH, CERE_CALIBRATE_ALL, 9},
                          Pick{CERE_LEVER_K2_SMALL_M_A8, 0, CERE_LEVER_DECODE_EAGER, CERE_CALIBRATE_ALL, 9},
                          Pick{CERE_LEVER_K2_CUBLASLT, 3, CERE_LEVER_DECODE_GRAPH, CERE_CALIBRATE_ALL, 4},
                          Pick{CERE_LEVER_K2_CUBLASLT, 0, CERE_LEVER_DECODE_EAGER, CERE_CALIBRATE_MTP, 4},
                          Pick{CERE_LEVER_K2_SMALL_M, 0, CERE_LEVER_DECODE_SPEC_GRAPH, CERE_CALIBRATE_MTP, 4},
                          Pick{CERE_LEVER_K2_CUBLASLT, 3, CERE_LEVER_DECODE_GRAPH, CERE_CALIBRATE_MTP, 4}}) {
        cere_executor *executor = spec_executor("toy,timed_rounds=1,chain=8");
        Desc d = rows_desc(2, 2048);
        const Levers levers =
            make_levers(p.k2_gemm, CERE_LEVER_PREFILL_CUDA_CORE, p.decode, CERE_LEVER_MOE_BUILD, p.draft_tokens);
        const cere_slot_selection sel = to_selection(levers);
        d.desc.selection = &sel;
        d.desc.calibrate_windows = p.calibrate;
        cere_model *model = gen_model(executor, d, true);
        d.desc.selection = nullptr;
        CHECK(same_levers(active_levers(executor, model), levers));
        CHECK_EQ(model->plan.calibrate_windows, p.calibrate);
        CHECK_EQ(toy(executor).spec_graph_rounds(), 0);
        const uint32_t own = p.k2_gemm != CERE_LEVER_K2_CUBLASLT ? 1 : 0;  // small_m_a8 uses small_m's table
        CHECK(model->round_costs_measured);
        CHECK(model->round_costs[own].table() != SpecCosts::untimed().table());
        CHECK(model->round_costs[1 - own].table() == SpecCosts::untimed().table());
        for (uint32_t w = 1; w <= 10; ++w) {
            CHECK_EQ(toy(executor).spec_rounds(1 - own, w), 0);
            CHECK_EQ(toy(executor).spec_rounds(own, w) > 0, w <= p.deepest);
        }
        if (p.deepest == 4) {  // windows past 4 extrapolated from window 4 by the prior shape (derive_round_costs)
            const auto ms = model->round_costs[own].table();
            for (uint32_t t = 5; t <= SpecCosts::kMaxWindow; ++t) {
                CHECK(ms[t - 1] == ms[3] * SpecCosts::shape(t) / SpecCosts::shape(4));
            }
        }
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
    const std::vector<uint32_t> prompt = {5, 9, 11};
    for (const char *dev : {"toy,timed_rounds=1", "toy,timed_rounds=1,draft_miss=2", "toy,timed_rounds=1,cycle=7"}) {
        const Outcome plain = gen_run(dev, {prompt, 40, {}, 0, 2048}, false);
        const auto [spec, verified] = gen_drafts(dev, {prompt, 40, {}, CERE_WORK_ADAPTIVE_DRAFTS, 2048});
        check_same(spec, plain);
        CHECK(verified > 0);
    }
    // A table under which only window 1 pays: the adaptive ticket verifies no draft and still commits plain's tokens.
    cere_executor *executor = spec_executor("toy,timed_rounds=1");
    Desc d = rows_desc(1, 2048);
    cere_model *model = gen_model(executor, d, true);
    CHECK(model->round_costs_measured);
    std::array<double, SpecCosts::kMaxWindow> steep_table{};
    steep_table.fill(100.0);
    steep_table[0] = 1.0;
    model->round_costs[0] = SpecCosts(steep_table);
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 2048, &session), CERE_OK);
    const Gen g{prompt, 40, {}, CERE_WORK_ADAPTIVE_DRAFTS, 2048};
    Job job = submit_gen(executor, model, session, CERE_OP_PREFILL, g);
    const Outcome steep = finish(executor, job.ticket, *job.work);
    CHECK_EQ(session->verified_drafts, 0);
    check_same(steep, gen_run("toy", {prompt, 40, {}, 0, 2048}, false));
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// LoadCrit LC-01 (why CERE_CALIBRATE_MTP is pick- and output-neutral for the drafters that never set work bits
// 5-7, e.g. -v29/-v22's mtp+lookup): a ticket with bits 2, 3 and 4 only reads its model's fixed table at windows
// 1..kMtpWindows (DraftPolicy::choose's room is kMaxDrafts; lengthen and extend run only under bits 5-7). On toys
// whose chains may carry 8 drafts, two models whose tables differ only past window 4 (cheap: every deeper window
// costs window 4's; dear: 100x) take exactly the same rounds (bursts, verify steps, accepted and verified drafts) and
// commit plain decode's tokens and scores. The same pair under bit 5, or bits 5 + 6, on the periodic toy does differ
// (the cheap table verifies chains past 3 drafts, the dear one never does), so the comparison can tell.
void test_spec_mtp_tickets_read_mtp_windows_only() {
    using cere::host::SpecCosts;
    const std::vector<uint32_t> periodic = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3};
    const std::vector<uint32_t> plain_prompt = {5, 9, 11};
    constexpr uint32_t kBase = CERE_WORK_ADAPTIVE_DRAFTS | CERE_WORK_LOOKUP_DRAFTS;
    const auto run = [](const char *dev, const Gen &g, bool cheap) {
        cere_executor *executor = spec_executor(dev);
        Desc d = rows_desc(1, std::max<uint64_t>(64, g.context));
        cere_model *model = gen_model(executor, d, true);
        std::array<double, SpecCosts::kMaxWindow> ms = SpecCosts::untimed().table();
        for (uint32_t t = 5; t <= SpecCosts::kMaxWindow; ++t) ms[t - 1] = cheap ? ms[3] : 100.0 * ms[3];
        for (SpecCosts &c : model->round_costs) c = SpecCosts(ms);
        cere_session *session = nullptr;
        CHECK_EQ(cere_session_create(executor, model, g.context, &session), CERE_OK);
        Job job = submit_gen(executor, model, session, CERE_OP_PREFILL, g);
        const Outcome o = finish(executor, job.ticket, *job.work);
        const uint64_t verified = session->verified_drafts;
        CHECK_EQ(cere_session_release(executor, session), CERE_OK);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
        return std::make_pair(o, verified);
    };
    for (const char *dev : {"toy,chain=8,cycle=5", "toy,chain=8,cycle=5,draft_miss=1",
                            "toy,chain=8,stateful=1,cycle=7,draft_miss=2", "toy,chain=8"}) {
        for (const std::vector<uint32_t> &prompt : {periodic, plain_prompt}) {
            const Outcome plain = gen_run(dev, {prompt, 40}, false);
            for (const uint32_t flags : {CERE_WORK_ADAPTIVE_DRAFTS, kBase, kBase | CERE_WORK_NO_MTP_DRAFTS}) {
                const auto [cheap, cheap_verified] = run(dev, {prompt, 40, {}, flags}, true);
                const auto [dear, dear_verified] = run(dev, {prompt, 40, {}, flags}, false);
                check_same(cheap, plain);
                check_same(dear, plain);
                CHECK(cheap.bursts == dear.bursts);
                CHECK_EQ(cheap.c.verify_steps, dear.c.verify_steps);
                CHECK_EQ(cheap.c.accepted_drafts, dear.c.accepted_drafts);
                CHECK_EQ(cheap_verified, dear_verified);
            }
        }
    }
    const char *dev = "toy,chain=8,cycle=5";
    const Outcome plain = gen_run(dev, {periodic, 40}, false);
    for (const uint32_t flags : {kBase | CERE_WORK_TAIL_DRAFTS, kBase | CERE_WORK_TAIL_DRAFTS | CERE_WORK_LONG_DRAFTS}) {
        const auto [cheap, cheap_verified] = run(dev, {periodic, 40, {}, flags}, true);
        const auto [dear, dear_verified] = run(dev, {periodic, 40, {}, flags}, false);
        check_same(cheap, plain);
        check_same(dear, plain);
        CHECK(*std::max_element(cheap.bursts.begin(), cheap.bursts.end()) > 4);
        CHECK(*std::max_element(dear.bursts.begin(), dear.bursts.end()) <= 4);
        CHECK(cheap_verified != dear_verified);
    }
}

// DraftTail: `g` on each of `count` sessions of one fresh drafting executor of `device`, submitted together so every
// round's speculate_rows call carries all of them; their outcomes and verified drafts.
std::vector<std::pair<Outcome, uint64_t>> gen_drafts_rows(const std::string &device, const Gen &g, uint32_t count) {
    cere_executor *executor = spec_executor(device);
    Desc d = rows_desc(count);
    cere_model *model = gen_model(executor, d, true);
    std::vector<cere_session *> sessions(count, nullptr);
    std::vector<Job> jobs;
    for (uint32_t i = 0; i < count; ++i) {
        CHECK_EQ(cere_session_create(executor, model, g.context, &sessions[i]), CERE_OK);
        jobs.push_back(submit_gen(executor, model, sessions[i], CERE_OP_PREFILL, g));
    }
    const std::vector<Outcome> outs = finish_all(executor, jobs);
    std::vector<std::pair<Outcome, uint64_t>> result;
    for (uint32_t i = 0; i < count; ++i) {
        result.emplace_back(outs[i], sessions[i]->verified_drafts);
        CHECK_EQ(cere_session_release(executor, sessions[i]), CERE_OK);
    }
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    return result;
}

// DraftTail (CERE_WORK_TAIL_DRAFTS, bit 5, on toys whose chains may carry 8 drafts): the round decides the tail. Every
// run commits the plain model's greedy tokens and scores. Without an agreeing lookup match (every MTP chain wrong at a
// position <= 3, so the lookup never agrees with it; or no periodic output) the rounds are exactly bit 4 + bit 2's:
// same rounds, verified drafts and bursts. With one (the periodic toy, proposals always right) a single session's
// rounds verify tails past 3 drafts and commit 40 tokens in fewer rounds. Two sessions in one call never take a tail
// (the single-row table does not price the longer verify): their rounds are bit 4 + bit 2's.
void test_spec_tail_policy() {
    const std::vector<uint32_t> periodic = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3};
    const std::vector<uint32_t> plain_prompt = {5, 9, 11};
    constexpr uint32_t kBase = CERE_WORK_ADAPTIVE_DRAFTS | CERE_WORK_LOOKUP_DRAFTS;
    constexpr uint32_t kTail = kBase | CERE_WORK_TAIL_DRAFTS;
    for (const char *dev : {"toy,chain=8,cycle=5,draft_miss=1", "toy,chain=8,cycle=5,draft_miss=2",
                            "toy,chain=8,cycle=5,draft_miss=3", "toy,chain=8,stateful=1,draft_miss=3",
                            "toy,chain=8,stateful=1,cycle=7,draft_miss=2"}) {
        for (const std::vector<uint32_t> &prompt : {periodic, plain_prompt}) {
            const Outcome plain = gen_run(dev, {prompt, 40}, false);
            const auto [base, base_verified] = gen_drafts(dev, {prompt, 40, {}, kBase});
            const auto [tail, tail_verified] = gen_drafts(dev, {prompt, 40, {}, kTail});
            check_same(base, plain);
            check_same(tail, plain);
            CHECK_EQ(tail.c.verify_steps, base.c.verify_steps);
            CHECK_EQ(tail.c.accepted_drafts, base.c.accepted_drafts);
            CHECK_EQ(tail_verified, base_verified);
            CHECK(tail.bursts == base.bursts);
        }
    }
    for (const std::vector<uint32_t> &prompt : {periodic, plain_prompt}) {
        const char *dev = "toy,chain=8,cycle=5";
        const Outcome plain = gen_run(dev, {prompt, 40}, false);
        const auto [base, base_verified] = gen_drafts(dev, {prompt, 40, {}, kBase});
        const auto [tail, tail_verified] = gen_drafts(dev, {prompt, 40, {}, kTail});
        check_same(base, plain);
        check_same(tail, plain);
        for (const uint64_t b : base.bursts) CHECK(b <= 4);
        for (const uint64_t b : tail.bursts) CHECK(b <= 9);
        CHECK(*std::max_element(tail.bursts.begin(), tail.bursts.end()) > 4);
        CHECK(tail_verified > 3 * (tail.c.verify_steps - 1));
        CHECK(tail.c.verify_steps < base.c.verify_steps);
        // Deterministic: a rerun takes the same rounds.
        const auto [again, again_verified] = gen_drafts(dev, {prompt, 40, {}, kTail});
        CHECK(again.bursts == tail.bursts);
        CHECK_EQ(again_verified, tail_verified);
        // Two sessions per call: no tail.
        const auto base_rows = gen_drafts_rows(dev, {prompt, 40, {}, kBase}, 2);
        const auto tail_rows = gen_drafts_rows(dev, {prompt, 40, {}, kTail}, 2);
        for (size_t i = 0; i < 2; ++i) {
            check_same(tail_rows[i].first, plain);
            CHECK_EQ(tail_rows[i].first.c.verify_steps, base_rows[i].first.c.verify_steps);
            CHECK_EQ(tail_rows[i].second, base_rows[i].second);
            CHECK(tail_rows[i].second <= 3 * tail_rows[i].first.c.verify_steps);
        }
    }
}

// TreeSpec (CERE_WORK_LONG_DRAFTS, bit 6, tree-spec design note TS-01, on toys whose chains may carry 8 drafts):
// a single drafting row's round may verify the lookup's first k > 3 drafts on its own (DraftPolicy::lengthen), with or
// without MTP agreement. Every run commits the plain model's greedy tokens and scores. On the cycle toys (periodic
// output from any prompt, so the lookup proposes long continuations) the bit-6 rounds burst past 4 and commit 40
// tokens in fewer rounds than bits 2 + 4; where every MTP chain goes wrong at a position <= 3 (draft_miss) bit 5's
// tail never fires (no agreement: its rounds are bits 2 + 4's) and bit 6's still do. Without periodic output there is
// no long proposal and the rounds are exactly bits 2 + 4's. Two sessions in one call take no long chain: the second
// session's rounds always share the call (verified <= 3 per round); the first one's first round runs alone while the
// second is in PREFILL, so it may verify one chain of up to kMaxTailDrafts.
void test_spec_long_policy() {
    const std::vector<uint32_t> periodic = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3};
    const std::vector<uint32_t> plain_prompt = {5, 9, 11};
    constexpr uint32_t kBase = CERE_WORK_ADAPTIVE_DRAFTS | CERE_WORK_LOOKUP_DRAFTS;
    constexpr uint32_t kTail = kBase | CERE_WORK_TAIL_DRAFTS;
    constexpr uint32_t kLong = kTail | CERE_WORK_LONG_DRAFTS;
    for (const char *dev :
         {"toy,chain=8,cycle=5", "toy,chain=8,cycle=5,draft_miss=1", "toy,chain=8,cycle=5,draft_miss=2"}) {
        const bool miss = std::string(dev).find("draft_miss") != std::string::npos;
        for (const std::vector<uint32_t> &prompt : {periodic, plain_prompt}) {
            const Outcome plain = gen_run(dev, {prompt, 40}, false);
            const auto [base, base_verified] = gen_drafts(dev, {prompt, 40, {}, kBase});
            const auto [tail, tail_verified] = gen_drafts(dev, {prompt, 40, {}, kTail});
            const auto [deep, deep_verified] = gen_drafts(dev, {prompt, 40, {}, kLong});
            check_same(deep, plain);
            check_same(tail, plain);
            if (miss) {
                CHECK(tail.bursts == base.bursts);
                CHECK_EQ(tail_verified, base_verified);
            }
            for (const uint64_t b : deep.bursts) CHECK(b <= 9);
            CHECK(*std::max_element(deep.bursts.begin(), deep.bursts.end()) > 4);
            CHECK(deep_verified > 3 * (deep.c.verify_steps - 1));
            CHECK(deep.c.verify_steps < base.c.verify_steps);
            // Deterministic: a rerun takes the same rounds.
            const auto [again, again_verified] = gen_drafts(dev, {prompt, 40, {}, kLong});
            CHECK(again.bursts == deep.bursts);
            CHECK_EQ(again_verified, deep_verified);
        }
    }
    for (const char *dev : {"toy,chain=8", "toy,chain=8,draft_miss=2", "toy,chain=8,stateful=1,draft_miss=3"}) {
        const Outcome plain = gen_run(dev, {plain_prompt, 40}, false);
        const auto [base, base_verified] = gen_drafts(dev, {plain_prompt, 40, {}, kBase});
        const auto [deep, deep_verified] = gen_drafts(dev, {plain_prompt, 40, {}, kLong});
        check_same(deep, plain);
        CHECK_EQ(deep.c.verify_steps, base.c.verify_steps);
        CHECK_EQ(deep_verified, base_verified);
        CHECK(deep.bursts == base.bursts);
    }
    for (const std::vector<uint32_t> &prompt : {periodic, plain_prompt}) {
        const char *dev = "toy,chain=8,cycle=5,draft_miss=2";
        const Outcome plain = gen_run(dev, {prompt, 40}, false);
        const auto deep_rows = gen_drafts_rows(dev, {prompt, 40, {}, kLong}, 2);
        check_same(deep_rows[0].first, plain);
        check_same(deep_rows[1].first, plain);
        CHECK(deep_rows[0].second <=
              3 * deep_rows[0].first.c.verify_steps + cere::host::DraftPolicy::kMaxTailDrafts - 3);
        CHECK(deep_rows[1].second <= 3 * deep_rows[1].first.c.verify_steps);
    }
}

// Chains (CERE_WORK_CHAIN_DRAFTS, bit 7, O123, chains design note): the chains of bits 5 and 6 may hold up to the
// device's max_chain_drafts (the toy's chain=N) instead of DraftPolicy::kMaxTailDrafts. Every run commits the plain
// model's greedy tokens and scores. On the periodic toy of chain=15 the bit-7 rounds burst past 9 (up to 16) and commit
// 48 tokens in fewer rounds than bits 2 + 4 + 5 + 6; with chain=8 (or without periodic output, or without bits 5 and 6)
// bit 7 changes no round. Two sessions in one call take no long chain (verified <= 3 per round past the first one's
// lone first round, which may verify one chain of up to max_chain_drafts).
void test_spec_chain_policy() {
    const std::vector<uint32_t> periodic = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3};
    const std::vector<uint32_t> plain_prompt = {5, 9, 11};
    constexpr uint32_t kBase = CERE_WORK_ADAPTIVE_DRAFTS | CERE_WORK_LOOKUP_DRAFTS;
    constexpr uint32_t kLong = kBase | CERE_WORK_TAIL_DRAFTS | CERE_WORK_LONG_DRAFTS;
    constexpr uint32_t kChain = kLong | CERE_WORK_CHAIN_DRAFTS;
    for (const char *dev : {"toy,chain=15,cycle=5", "toy,chain=15,cycle=5,draft_miss=2"}) {
        for (const std::vector<uint32_t> &prompt : {periodic, plain_prompt}) {
            const Outcome plain = gen_run(dev, {prompt, 48}, false);
            const auto [deep, deep_verified] = gen_drafts(dev, {prompt, 48, {}, kLong});
            const auto [chain, chain_verified] = gen_drafts(dev, {prompt, 48, {}, kChain});
            check_same(deep, plain);
            check_same(chain, plain);
            for (const uint64_t b : deep.bursts) CHECK(b <= 9);
            for (const uint64_t b : chain.bursts) CHECK(b <= 16);
            CHECK(*std::max_element(chain.bursts.begin(), chain.bursts.end()) > 9);
            CHECK(chain.c.verify_steps < deep.c.verify_steps);
            // Deterministic: a rerun takes the same rounds.
            const auto [again, again_verified] = gen_drafts(dev, {prompt, 48, {}, kChain});
            CHECK(again.bursts == chain.bursts);
            CHECK_EQ(again_verified, chain_verified);
            // Without bits 5 and 6, bit 7 does nothing.
            const auto [base, base_verified] = gen_drafts(dev, {prompt, 48, {}, kBase});
            const auto [lone, lone_verified] = gen_drafts(dev, {prompt, 48, {}, kBase | CERE_WORK_CHAIN_DRAFTS});
            CHECK(lone.bursts == base.bursts);
            CHECK_EQ(lone_verified, base_verified);
        }
    }
    // A device whose chains stop at 8, or no periodic output: bit 7 changes no round.
    for (const auto &[dev, prompt] : {std::pair<const char *, std::vector<uint32_t>>{"toy,chain=8,cycle=5", periodic},
                                      {"toy,chain=15", plain_prompt}}) {
        const Outcome plain = gen_run(dev, {prompt, 48}, false);
        const auto [deep, deep_verified] = gen_drafts(dev, {prompt, 48, {}, kLong});
        const auto [chain, chain_verified] = gen_drafts(dev, {prompt, 48, {}, kChain});
        check_same(chain, plain);
        CHECK(chain.bursts == deep.bursts);
        CHECK_EQ(chain_verified, deep_verified);
    }
    for (const std::vector<uint32_t> &prompt : {periodic, plain_prompt}) {
        const char *dev = "toy,chain=15,cycle=5";
        const Outcome plain = gen_run(dev, {prompt, 48}, false);
        const auto rows = gen_drafts_rows(dev, {prompt, 48, {}, kChain}, 2);
        check_same(rows[0].first, plain);
        check_same(rows[1].first, plain);
        CHECK(rows[0].second <= 3 * rows[0].first.c.verify_steps + 15 - 3);
        CHECK(rows[1].second <= 3 * rows[1].first.c.verify_steps);
    }
}

// C4 (D7): fail_forward counts a speculation round twice (verify launch, then draft-extend). A failure in the verify
// launch ends the ticket ERROR with nothing of that round committed (the session consumed nothing); one in the
// draft-extend reports the round's committed tokens on that poll and ends the ticket ERROR on the next; either way
// the session is broken (its next item ends ERROR).
void test_spec_fail_forward() {
    const std::vector<uint32_t> prompt = {2, 7, 1};
    const Outcome full = gen_run("toy", {prompt, 8}, false);
    for (const uint64_t fail : {1, 2}) {  // call 0 is the PREFILL's step
        cere_executor *executor = spec_executor("toy,fail_forward=" + std::to_string(fail));
        Desc d = rows_desc(2);
        cere_model *model = gen_model(executor, d, true);
        cere_session *s = nullptr;
        CHECK_EQ(cere_session_create(executor, model, 64, &s), CERE_OK);
        Job job = submit_gen(executor, model, s, CERE_OP_PREFILL, {prompt, 8});
        Outcome o;
        CHECK(!poll_into(executor, job.ticket, *job.work, o));
        CHECK_EQ(o.c.committed_tokens, 1);
        if (fail == 1) {
            CHECK(poll_into(executor, job.ticket, *job.work, o));
            CHECK_EQ(o.c.committed_tokens, 1);
            CHECK_EQ(s->tokens.size(), prompt.size() + 1);
            CHECK_EQ(s->fed, prompt.size());
        } else {
            CHECK(!poll_into(executor, job.ticket, *job.work, o));  // the committed round is reported
            CHECK_EQ(o.c.committed_tokens, 5);
            CHECK_EQ(o.c.verify_steps, 1);
            CHECK_EQ(o.c.accepted_drafts, 3);
            CHECK(poll_into(executor, job.ticket, *job.work, o));
            CHECK_EQ(o.c.committed_tokens, 5);
        }
        CHECK_EQ(o.c.finish, CERE_FINISH_ERROR);
        CHECK_EQ(o.c.status, CERE_ERR_INTERNAL);
        CHECK(o.out == std::vector<uint32_t>(full.out.begin(), full.out.begin() + o.c.committed_tokens));
        Job next = submit_gen(executor, model, s, CERE_OP_DECODE, {{}, 2});
        const Outcome no = finish(executor, next.ticket, *next.work);
        CHECK_EQ(no.c.finish, CERE_FINISH_ERROR);
        CHECK_EQ(no.c.committed_tokens, 0);
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
}

// C4 (D7): a held drafting row (toy hold_after: CERE_PENDING) commits and consumes nothing while the other row of
// the round advances; a cancel ends it on its boundary; a hold past hold_ms fails it before any commit (ERROR, the
// round's tokens not committed).
void test_spec_hold_after() {
    const std::vector<uint32_t> pa = {3, 1, 4};
    const std::vector<uint32_t> pb = {1, 5, 9};
    {
        cere_executor *executor = spec_executor("toy,hold_after=2,hold_count=1");
        Desc d = rows_desc(2);
        cere_model *model = gen_model(executor, d, true);
        cere_session *a = nullptr;
        cere_session *b = nullptr;
        CHECK_EQ(cere_session_create(executor, model, 64, &a), CERE_OK);
        CHECK_EQ(cere_session_create(executor, model, 64, &b), CERE_OK);
        Job ja = submit_gen(executor, model, a, CERE_OP_PREFILL, {pa, 12});
        Job jb = submit_gen(executor, model, b, CERE_OP_PREFILL, {pb, 9});
        Outcome oa;
        CHECK(!poll_into(executor, ja.ticket, *ja.work, oa));  // prefill a
        CHECK(!poll_into(executor, ja.ticket, *ja.work, oa));  // prefill b: the pass goes on (BC4 (b))
        CHECK_EQ(oa.c.committed_tokens, 1);
        CHECK(!poll_into(executor, ja.ticket, *ja.work, oa));  // a's first round (1 + 4), with b's
        CHECK_EQ(oa.c.committed_tokens, 5);
        const uint64_t fed = a->fed;
        Outcome ob = finish(executor, jb.ticket, *jb.work);  // b runs to the end while a is held
        check_same(ob, gen_run("toy", {pb, 9}, false));
        CHECK(!poll_into(executor, ja.ticket, *ja.work, oa));
        CHECK_EQ(oa.c.committed_tokens, 5);
        CHECK_EQ(a->fed, fed);
        CHECK_EQ(cere_executor_cancel(executor, ja.ticket), CERE_OK);
        oa = finish(executor, ja.ticket, *ja.work, oa);
        CHECK_EQ(oa.c.finish, CERE_FINISH_CANCELLED);
        CHECK_EQ(oa.c.committed_tokens, 5);
        CHECK(oa.out == std::vector<uint32_t>(gen_run("toy", {pa, 5}, false).out));
        CHECK_EQ(cere_session_release(executor, a), CERE_OK);
        CHECK_EQ(cere_session_release(executor, b), CERE_OK);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
    {
        cere_executor *executor = spec_executor("toy,hold_after=2,hold_ms=20");
        Desc d = rows_desc(2);
        cere_model *model = gen_model(executor, d, true);
        cere_session *a = nullptr;
        CHECK_EQ(cere_session_create(executor, model, 64, &a), CERE_OK);
        Job ja = submit_gen(executor, model, a, CERE_OP_PREFILL, {pa, 12});
        Outcome oa;
        CHECK(!poll_into(executor, ja.ticket, *ja.work, oa));
        CHECK(!poll_into(executor, ja.ticket, *ja.work, oa));
        CHECK(!poll_into(executor, ja.ticket, *ja.work, oa));  // held
        CHECK_EQ(oa.c.committed_tokens, 5);
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        oa = finish(executor, ja.ticket, *ja.work, oa);
        CHECK_EQ(oa.c.finish, CERE_FINISH_ERROR);
        CHECK_EQ(oa.c.status, CERE_ERR_INTERNAL);
        CHECK_EQ(oa.c.committed_tokens, 5);
        CHECK_EQ(oa.c.verify_steps, 1);
        CHECK_EQ(a->tokens.size(), pa.size() + 5);
        CHECK_EQ(cere_session_release(executor, a), CERE_OK);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
}

// C5 (D9 as M29S2-22): one round of four tickets: a drafting greedy ticket, a NO_SPECULATION one (its session never
// drafts: a plain row) joining later, one whose item stop token falls inside a chain and a stop-ignoring one cut by
// its context (draft 0 every round); each equals its plain run with the toy's predicted counters. Then per ticket
// (bit 1 per ticket, M29S2-23): a NO_SPECULATION DECODE and a constrained DECODE on the drafting session run as
// chains of 1 (draft 0) whose draft-extend still refreshes the proposals (the next DECODE accepts at once), and the
// whole sequence equals the same items on a plain model.
void test_spec_rows() {
    const DraftDevice dev = kDraftDevices[3];  // draft_miss=3: rounds of 2 accepted drafts + the correction
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    const Outcome c_full = gen_run(dev.id, {{6}, 12}, false);
    CHECK_EQ(first_index(c_full.out, c_full.out[5]), 5);  // round 2 commits 4, 5 (a draft), 6
    const std::vector<Gen> gens = {{{1, 2}, 12},
                                   {{3, 4, 5}, 9, {}, CERE_WORK_NO_SPECULATION},
                                   {{6}, 12, {c_full.out[5]}},
                                   {{7, 8, 9, 10}, 20, {}, CERE_WORK_IGNORE_STOP_TOKENS, 10}};
    cere_executor *executor = spec_executor(dev.id);
    Desc d = rows_desc(4);
    cere_model *model = gen_model(executor, d, true);
    std::vector<cere_session *> sessions(gens.size(), nullptr);
    std::vector<Job> jobs(gens.size());
    for (size_t i = 0; i < gens.size(); ++i) {
        CHECK_EQ(cere_session_create(executor, model, gens[i].context, &sessions[i]), CERE_OK);
        if (i != 1) jobs[i] = submit_gen(executor, model, sessions[i], CERE_OP_PREFILL, gens[i]);
    }
    std::vector<Outcome> outs(gens.size());
    for (int round = 0; round < 3; ++round) poll_into(executor, jobs[0].ticket, *jobs[0].work, outs[0]);
    jobs[1] = submit_gen(executor, model, sessions[1], CERE_OP_PREFILL, gens[1]);  // joins mid-decode
    std::vector<bool> done(gens.size(), false);
    for (int i = 0; i < 1000 && std::count(done.begin(), done.end(), false) != 0; ++i) {
        for (size_t j = 0; j < jobs.size(); ++j) {
            if (!done[j]) done[j] = poll_into(executor, jobs[j].ticket, *jobs[j].work, outs[j]);
        }
    }
    for (size_t i = 0; i < gens.size(); ++i) {
        const Outcome plain = gen_run(dev.id, gens[i], false);
        check_same(outs[i], plain);
        const uint64_t limit = std::min(gens[i].n, gens[i].context - gens[i].prompt.size());
        if (i == 1) {
            check_counts(outs[i], {0, 0});
        } else {
            check_counts(outs[i], predicted(outs[i].c.committed_tokens, limit, dev.miss, i != 3));
        }
    }
    CHECK_EQ(outs[2].c.finish, CERE_FINISH_STOP_TOKEN);
    CHECK_EQ(outs[3].c.finish, CERE_FINISH_LENGTH);
    CHECK_EQ(outs[3].c.committed_tokens, 6);
    CHECK(sessions[0]->drafting);
    CHECK(!sessions[1]->drafting);

    // The per-ticket sequence on session 0, replayed on a plain model.
    const auto script = [&](cere_executor *ex, cere_model *m, cere_session *s, bool drafting) {
        std::vector<Outcome> r;
        Job j1 = submit_gen(ex, m, s, CERE_OP_DECODE, {{}, 3, {}, CERE_WORK_NO_SPECULATION});
        r.push_back(finish(ex, j1.ticket, *j1.work));
        Job j2 = submit_gen(ex, m, s, CERE_OP_DECODE, {{}, 4});
        r.push_back(finish(ex, j2.ticket, *j2.work));
        // Constrained (flag bit 1 clear, 2 tokens: a chain of 2 would fit): greedy under masks allowing one id each.
        const uint32_t forced = (r.back().out.back() + 17) % vocab;
        Constrained c(s, CERE_OP_DECODE, {}, 2, generation_of(ex, m), mask_of_ids({forced}));
        c.work.item.flags = 0;
        cere_completion_ticket t{};
        CHECK_EQ(cere_executor_submit(ex, &c.work.batch, &t), CERE_OK);
        Outcome co;
        CHECK(!poll_into(ex, t, c.work, co));
        CHECK_EQ(co.c.committed_tokens, 1);
        CHECK_EQ(constrain(ex, t, 1, mask_of_ids({forced + 1})), CERE_OK);
        r.push_back(finish(ex, t, c.work, co));
        CHECK(r.back().out == std::vector<uint32_t>({forced, forced + 1}));
        Job j3 = submit_gen(ex, m, s, CERE_OP_DECODE, {{}, 4});
        r.push_back(finish(ex, j3.ticket, *j3.work));
        if (drafting) {
            check_counts(r[0], {3, 0});  // three chains of 1
            check_counts(r[1], predicted(4, 4, dev.miss, true, 0));
            CHECK_EQ(r[1].bursts.at(0), 3);  // fresh proposals right after the draft-0 rounds
            check_counts(r[2], {2, 0});
            check_counts(r[3], predicted(4, 4, dev.miss, true, 0));
            CHECK_EQ(r[3].bursts.at(0), 3);
        }
        return r;
    };
    const std::vector<Outcome> spec = script(executor, model, sessions[0], true);
    for (cere_session *s : sessions) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    executor = spec_executor(dev.id);
    Desc dp = rows_desc(1);
    model = gen_model(executor, dp, false);
    cere_session *s = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &s), CERE_OK);
    Job pre = submit_gen(executor, model, s, CERE_OP_PREFILL, gens[0]);
    check_same(outs[0], finish(executor, pre.ticket, *pre.work));
    const std::vector<Outcome> plain = script(executor, model, s, false);
    for (size_t i = 0; i < plain.size(); ++i) check_same(spec[i], plain[i]);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

namespace {

// The chi-square upper tail P(X >= x) with k degrees of freedom (Wilson-Hilferty).
double chi2_tail(double x, double k) {
    const double h = 2.0 / (9.0 * k);
    const double z = (std::cbrt(x / k) - (1.0 - h)) / std::sqrt(h);
    return 0.5 * std::erfc(z / std::sqrt(2.0));
}

// The chi-square p-value of `observed` counts (n samples) against probabilities `expected`; cells expecting fewer
// than 5 samples are pooled into one.
double chi2_p(const std::vector<uint64_t> &observed, const std::vector<double> &expected, uint64_t n) {
    double stat = 0.0;
    double pooled_e = 0.0;
    double pooled_o = 0.0;
    uint64_t cells = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double e = expected[i] * double(n);
        if (e < 5.0) {
            pooled_e += e;
            pooled_o += double(observed[i]);
            CHECK(e > 0.0 || observed[i] == 0);
            continue;
        }
        stat += (double(observed[i]) - e) * (double(observed[i]) - e) / e;
        ++cells;
    }
    if (pooled_e > 0.0) {
        stat += (pooled_o - pooled_e) * (pooled_o - pooled_e) / pooled_e;
        ++cells;
    }
    return chi2_tail(stat, double(cells - 1));
}

}  // namespace

// C9 (D6): sampled speculation preserves the plain distribution. On the toy_small_vocab model (vocab 64, temperature
// 1 over the toy's 8 finite logits) 200,000 seeded 4-token continuations (the PREFILL's plain draw, then one round of
// 2 drafts + bonus) match the analytic plain distribution: chi-square per position and over the 4-token sequence, p
// >= 1e-3 / 5 (Bonferroni); drafts are both accepted and rejected; a seeded run is reproducible; the accept (M8
// channels 4, 5) and resample (6, 7) draws at one (seed, position) are uncorrelated (Pearson |r| < 0.01 over 2^20).
void test_spec_sampled_distribution() {
    namespace sr = cere::sampler_ref;
    const uint32_t vocab = cere::toy::kSmallVocab;
    const std::vector<uint32_t> prompt = {3, 17};
    constexpr uint64_t kRuns = 200000;
    constexpr uint64_t kLen = 4;
    // p(x | last, position): the toy's row through the plain sampler stages.
    const auto conditional = [&](uint32_t last, uint64_t position) {
        const std::vector<uint16_t> row = expected_toy_row(cere::toy::next_token(last, position, vocab), vocab);
        std::vector<float> x(vocab);
        std::vector<uint32_t> counts(vocab, 0);
        sr::process(row.data(), vocab, counts.data(), 0.0f, 0.0f, 1.0f, nullptr, nullptr, 0, x.data());
        std::vector<float> probs(vocab);
        sr::probabilities(x.data(), vocab, 1.0f, probs.data());
        return std::vector<double>(probs.begin(), probs.end());
    };
    // The analytic distribution of the 4-token sequence (index = x0 * 64^3 + x1 * 64^2 + x2 * 64 + x3; zero cells
    // skipped) and its per-position marginals.
    std::map<uint64_t, double> joint;
    std::vector<std::vector<double>> marginal(kLen, std::vector<double>(vocab, 0.0));
    const std::function<void(uint32_t, uint64_t, uint64_t, double)> walk = [&](uint32_t last, uint64_t i,
                                                                               uint64_t index, double p) {
        if (i == kLen) {
            joint[index] += p;
            return;
        }
        const std::vector<double> q = conditional(last, prompt.size() + i);
        for (uint32_t x = 0; x < vocab; ++x) {
            if (q[x] == 0.0) continue;
            marginal[i][x] += p * q[x];
            walk(x, i + 1, index * vocab + x, p * q[x]);
        }
    };
    walk(prompt.back(), 0, 0, 1.0);

    cere_executor *executor = make_executor();
    Desc d(cere::toy::small_vocab_support().config, {16, 32}, {}, 1, 64);
    std::vector<uint32_t> map(vocab);
    for (uint32_t i = 0; i < vocab; ++i) map[i] = i;
    set_drafts(d, 4, &map);
    cere_model *model = sealed_model(executor, d);
    const uint64_t generation = generation_of(executor, model);
    // One continuation (seed); *rounds/*accepted its counters: a rejected draft ends a round, so 1..3 rounds.
    const auto continuation = [&](uint64_t seed, uint64_t *rounds, uint64_t *accepted) {
        cere_session *s = nullptr;
        CHECK_EQ(cere_session_create(executor, model, 64, &s), CERE_OK);
        Work w(s, CERE_OP_PREFILL, prompt, kLen, generation);
        const cere_sampling_params sampling = sampling_params(1.0f, vocab, seed);
        w.item.sampling = &sampling;
        cere_completion_ticket ticket{};
        CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
        const cere_completion c = drain(executor, ticket);
        CHECK_EQ(c.committed_tokens, kLen);
        CHECK(c.verify_steps >= 1 && c.verify_steps <= kLen - 1);
        CHECK(c.accepted_drafts + c.verify_steps == kLen - 1);  // every round commits its accepted drafts + 1
        *rounds = c.verify_steps;
        *accepted = c.accepted_drafts;
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        return w.out;
    };
    std::map<uint64_t, uint64_t> joint_seen;
    std::vector<std::vector<uint64_t>> seen(kLen, std::vector<uint64_t>(vocab, 0));
    uint64_t accepted_total = 0;
    uint64_t one_round = 0;
    for (uint64_t r = 0; r < kRuns; ++r) {
        uint64_t rounds = 0;
        uint64_t accepted = 0;
        const std::vector<uint32_t> out = continuation(r + 1, &rounds, &accepted);
        uint64_t index = 0;
        for (uint64_t i = 0; i < kLen; ++i) {
            ++seen[i][out[i]];
            index = index * vocab + out[i];
        }
        ++joint_seen[index];
        accepted_total += accepted;
        one_round += rounds == 1 ? 1 : 0;
    }
    const double bound = 1e-3 / double(kLen + 1);
    for (uint64_t i = 0; i < kLen; ++i) CHECK(chi2_p(seen[i], marginal[i], kRuns) >= bound);
    std::vector<double> joint_p;
    std::vector<uint64_t> joint_n;
    uint64_t inside = 0;
    for (const auto &cell : joint) {
        joint_p.push_back(cell.second);
        const auto it = joint_seen.find(cell.first);
        joint_n.push_back(it == joint_seen.end() ? 0 : it->second);
        inside += joint_n.back();
    }
    CHECK_EQ(inside, kRuns);  // no sequence outside the support
    CHECK(chi2_p(joint_n, joint_p, kRuns) >= bound);
    CHECK(one_round > 0);         // some continuations accept both drafts ...
    CHECK(one_round < kRuns);     // ... and some reject one (resampled, then another round)
    CHECK(accepted_total > 0);
    uint64_t r1 = 0, r2 = 0, a1 = 0, a2 = 0;
    CHECK(continuation(77, &r1, &a1) == continuation(77, &r2, &a2));  // seeded: reproducible
    CHECK_EQ(r1, r2);
    CHECK_EQ(a1, a2);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    // Accept vs resample draws at the same (seed, position).
    constexpr uint32_t kPositions = uint32_t(1) << 20;
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (uint32_t key = 0; key < kPositions; ++key) {
        const double x = sr::uniform(5, key, 4, 5);
        const double y = sr::uniform(5, key, 6, 7);
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    const double n = kPositions;
    const double r = (n * sxy - sx * sy) / std::sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    CHECK(std::fabs(r) < 0.01);
}

// batch-concurrency design note BC-D4 (F11): a sampled drafting ticket's draft picks depend only on its request and
// token stream. On toys that time their rounds (timed_rounds=1, as the CUDA device does), each of 4 seeded sampled
// CERE_WORK_LOOKUP_DRAFTS tickets (cered's sampled ticket under the -v52 drafter "mtp+lookup") commits the same
// tokens, scores (bit for bit), verify steps, accepted and verified drafts alone (one after another: single-row
// rounds), alone again after those rounds, and as one of the 4 rows of every round (no single-row round), all on one
// load as cered serves c1 then c4. Before BC-D4 its picks were priced by the model's wall-clock EMA of single-row
// rounds (`spec_costs`), which the solo rounds moved and the 4-row rounds never fed.
void test_spec_sampled_picks_invariant() {
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    const std::vector<std::vector<uint32_t>> prompts = {
        {5, 9, 11}, {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3}, {7, 3}, {12, 30, 2, 12, 30}};
    constexpr uint64_t kNew = 48;
    constexpr uint64_t kContext = 256;
    struct Result {
        Outcome o;
        uint64_t verified = 0;
    };
    for (const char *dev : {"toy,timed_rounds=1", "toy,timed_rounds=1,cycle=5", "toy,timed_rounds=1,draft_miss=2"}) {
        cere_executor *executor = spec_executor(dev);
        Desc d = rows_desc(4, kContext);
        cere_model *model = gen_model(executor, d, true);
        const uint64_t generation = generation_of(executor, model);
        std::vector<cere_sampling_params> params;
        for (size_t i = 0; i < prompts.size(); ++i) params.push_back(sampling_params(1.0f, vocab, 11 + i));
        // The requests `which`, submitted together (every round carries each of them until it ends).
        const auto run = [&](const std::vector<size_t> &which) {
            std::vector<cere_session *> sessions(which.size(), nullptr);
            std::vector<Job> jobs;
            for (size_t j = 0; j < which.size(); ++j) {
                CHECK_EQ(cere_session_create(executor, model, kContext, &sessions[j]), CERE_OK);
                Job job;
                job.work = std::make_unique<Work>(sessions[j], CERE_OP_PREFILL, prompts[which[j]], kNew, generation);
                job.work->item.flags = CERE_WORK_LOOKUP_DRAFTS;
                job.work->item.sampling = &params[which[j]];
                CHECK_EQ(cere_executor_submit(executor, &job.work->batch, &job.ticket), CERE_OK);
                jobs.push_back(std::move(job));
            }
            const std::vector<Outcome> outs = finish_all(executor, jobs);
            std::vector<Result> results;
            for (size_t j = 0; j < which.size(); ++j) {
                results.push_back({outs[j], sessions[j]->verified_drafts});
                CHECK_EQ(cere_session_release(executor, sessions[j]), CERE_OK);
            }
            return results;
        };
        const auto same = [](const Result &a, const Result &b) {
            CHECK_EQ(a.o.c.committed_tokens, b.o.c.committed_tokens);
            CHECK_EQ(a.o.c.finish, b.o.c.finish);
            CHECK(a.o.out == b.o.out);
            CHECK(a.o.scores.size() == b.o.scores.size() &&
                  std::memcmp(a.o.scores.data(), b.o.scores.data(), a.o.scores.size() * sizeof(cere_token_score)) ==
                      0);
            CHECK_EQ(a.o.c.verify_steps, b.o.c.verify_steps);
            CHECK_EQ(a.o.c.accepted_drafts, b.o.c.accepted_drafts);
            CHECK_EQ(a.verified, b.verified);
        };
        std::vector<Result> solo;
        for (size_t i = 0; i < prompts.size(); ++i) solo.push_back(run({i})[0]);
        same(run({0})[0], solo[0]);
        const std::vector<Result> rows = run({0, 1, 2, 3});
        uint64_t verified = 0;
        uint64_t accepted = 0;
        for (size_t i = 0; i < prompts.size(); ++i) {
            same(rows[i], solo[i]);
            CHECK_EQ(solo[i].o.c.committed_tokens, kNew);
            verified += solo[i].verified;
            accepted += solo[i].o.c.accepted_drafts;
        }
        CHECK(accepted > 0 && accepted < verified);  // sampled drafts both accepted and rejected
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
}

// --- GrammarSpec: constrained speculation (grammar-spec design note GS-01, GS-02; ledger CR7) -------------------

namespace {

// The tests' grammar over a ticket's committed tokens: `id` may follow `prefix` unless (id + sum(prefix)) % `modulus`
// is 0. Each draft prefix has its own mask, and the toy's own next token is masked out about once in `modulus`
// positions (the masked rule then picks another id, so the toy's proposals, its unmasked chain, go wrong there).
std::vector<uint32_t> grammar_mask(const std::vector<uint32_t> &prefix, uint32_t vocab, uint32_t modulus = 3) {
    uint64_t sum = 0;
    for (const uint32_t t : prefix) sum += t;
    std::vector<uint32_t> mask((uint64_t(vocab) + 31) / 32, 0);
    for (uint32_t id = 0; id < vocab; ++id) {
        if ((id + sum) % modulus != 0) mask[id >> 5] |= uint32_t(1) << (id & 31);
    }
    return mask;
}

// Whether every committed token of `o` is allowed by the grammar after the tokens before it.
bool grammar_valid(const Outcome &o, uint32_t vocab, uint32_t modulus = 3) {
    for (size_t i = 0; i < o.out.size(); ++i) {
        const std::vector<uint32_t> mask =
            grammar_mask(std::vector<uint32_t>(o.out.begin(), o.out.begin() + i), vocab, modulus);
        if (((mask[o.out[i] >> 5] >> (o.out[i] & 31)) & 1u) == 0) return false;
    }
    return true;
}

// One constrained ticket steered as cered's generation thread does (GS-04).
struct Steered {
    cere_completion_ticket ticket{};
    const Work *work = nullptr;
    Outcome o;
    uint64_t given = 0;    // the position whose mask(s) the ticket holds (submit: 0)
    uint64_t offered = 0;  // drafts cere_executor_round_drafts froze
    uint64_t kept = 0;     // of them, the drafts the grammar accepted (passed to cere_executor_constrain_drafts)
};

// One poll of `s`; true once terminal. A poll that leaves it waiting for the mask of `committed` is followed by
// cere_executor_constrain with the grammar's mask (`drafts` false) or by cere_executor_round_drafts and then
// cere_executor_constrain_drafts with the masks of the draft prefixes the grammar accepts (row j: after drafts
// 1..j; the first draft the grammar refuses ends them).
bool steer_poll(cere_executor *executor, Steered &s, bool drafts, uint32_t vocab, uint32_t modulus = 3) {
    if (poll_into(executor, s.ticket, *s.work, s.o)) return true;
    const uint64_t position = s.o.c.committed_tokens;
    if (position == s.given) return false;  // not stepped by that poll: it still holds its mask
    s.given = position;
    std::vector<uint32_t> prefix(s.work->out.begin(), s.work->out.begin() + position);
    std::vector<uint32_t> masks = grammar_mask(prefix, vocab, modulus);
    const uint64_t words = masks.size();
    if (!drafts) {
        CHECK_EQ(cere_executor_constrain(executor, s.ticket, position, masks.data(), words), CERE_OK);
        return false;
    }
    uint32_t ids[CERE_MAX_CONSTRAINED_DRAFTS] = {};
    uint32_t n = CERE_MAX_CONSTRAINED_DRAFTS + 1;
    CHECK_EQ(cere_executor_round_drafts(executor, s.ticket, position, ids, CERE_MAX_CONSTRAINED_DRAFTS, &n), CERE_OK);
    CHECK(n <= CERE_MAX_CONSTRAINED_DRAFTS);
    uint32_t m = 0;
    while (m < n && ((masks[m * words + (ids[m] >> 5)] >> (ids[m] & 31)) & 1u) != 0) {
        prefix.push_back(ids[m]);
        const std::vector<uint32_t> next = grammar_mask(prefix, vocab, modulus);
        masks.insert(masks.end(), next.begin(), next.end());
        ++m;
    }
    s.offered += n;
    s.kept += m;
    CHECK_EQ(cere_executor_constrain_drafts(executor, s.ticket, position, ids, m, masks.data(), words), CERE_OK);
    return false;
}

// A constrained item of `g` (its prompt, max_new_tokens, stops and flags) on `session`, the grammar's mask of
// position 0 at submit; `params` its sampling (default: greedy with processors).
struct GrammarItem {
    Constrained c;
    Steered s;

    GrammarItem(cere_executor *executor, cere_model *model, cere_session *session, uint32_t op, const Gen &g,
                uint32_t vocab, uint32_t modulus = 3, const cere_sampling_params *params = nullptr)
        : c(session, op, g.prompt, g.n, generation_of(executor, model), grammar_mask({}, vocab, modulus),
            params != nullptr ? *params : sampling_params(0.0f, vocab)) {
        c.work.item.flags = g.flags;
        c.work.stops = g.stops;
        c.work.item.stop_token_ids = c.work.stops.empty() ? nullptr : c.work.stops.data();
        c.work.item.stop_token_count = c.work.stops.size();
        s.work = &c.work;
        CHECK_EQ(cere_executor_submit(executor, &c.work.batch, &s.ticket), CERE_OK);
    }
};

// `g` as one constrained PREFILL under the grammar on a fresh executor of `device` (its model with or without
// drafts), steered with or without drafts; adds the run's offered and kept drafts to *offered / *kept.
Outcome grammar_run(const std::string &device, const Gen &g, bool model_drafts, bool drafts,
                    uint64_t *offered = nullptr, uint64_t *kept = nullptr) {
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    cere_executor *executor = spec_executor(device);
    Desc d = rows_desc(1, std::max<uint64_t>(64, g.context));
    cere_model *model = gen_model(executor, d, model_drafts);
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, g.context, &session), CERE_OK);
    GrammarItem item(executor, model, session, CERE_OP_PREFILL, g, vocab);
    for (int i = 0; i < 1000 && !steer_poll(executor, item.s, drafts, vocab); ++i) {
    }
    CHECK_EQ(item.s.o.c.terminal, 1);
    if (offered != nullptr) *offered += item.s.offered;
    if (kept != nullptr) *kept += item.s.kept;
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    return item.s.o;
}

uint64_t max_burst(const Outcome &o) {
    return o.bursts.empty() ? 0 : *std::max_element(o.bursts.begin(), o.bursts.end());
}

}  // namespace

// GS-R2 (B1 on the toy): a constrained greedy ticket steered with drafts (round_drafts, the grammar's masks of the
// accepted draft prefixes, constrain_drafts) commits the tokens and scores (bit for bit) and the finish of the same
// ticket on a plain model under cere_executor_constrain, on the stateful toy for every draft_miss, with the MTP
// source and the lookup / adaptive / tail / long / chain policy bits (a constrained ticket is sampled: never more
// than CERE_MAX_CONSTRAINED_DRAFTS drafts), a stop inside a chain and the context limit. Every token is allowed by
// the grammar; the grammar keeps some drafts and refuses others (mid-chain rejection), and kept drafts are accepted.
// The drafting model steered without drafts (cere_executor_constrain only) runs chains of 1 with the same tokens.
void test_grammar_spec_greedy_equals_masked_plain() {
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    constexpr uint32_t kPolicy = CERE_WORK_LOOKUP_DRAFTS | CERE_WORK_ADAPTIVE_DRAFTS;
    constexpr uint32_t kDeep = kPolicy | CERE_WORK_TAIL_DRAFTS | CERE_WORK_LONG_DRAFTS | CERE_WORK_CHAIN_DRAFTS;
    const std::vector<uint32_t> prompt = {5, 9, 11};
    const std::vector<uint32_t> periodic = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3};
    uint64_t offered = 0;
    uint64_t kept = 0;
    uint64_t accepted = 0;
    for (const DraftDevice &dev : kDraftDevices) {
        const Outcome full = grammar_run(dev.id, {prompt, 13}, false, false);
        const std::vector<Gen> gens = {{prompt, 13},
                                       {prompt, 2},
                                       {prompt, 1},
                                       {prompt, 13, {full.out[6]}},
                                       {prompt, 20, {}, 0, 10},
                                       {periodic, 16, {}, kPolicy},
                                       {periodic, 16, {}, kDeep}};
        for (const Gen &g : gens) {
            const Outcome plain = grammar_run(dev.id, g, false, false);
            const Outcome spec = grammar_run(dev.id, g, true, true, &offered, &kept);
            check_same(spec, plain);
            CHECK(grammar_valid(spec, vocab));
            CHECK(max_burst(spec) <= CERE_MAX_CONSTRAINED_DRAFTS + 1);
            CHECK(spec.c.accepted_drafts <= spec.c.committed_tokens);
            accepted += spec.c.accepted_drafts;
            const Outcome off = grammar_run(dev.id, g, true, false);
            CHECK(off.out == plain.out);
            CHECK_EQ(off.c.finish, plain.c.finish);
            CHECK_EQ(off.c.accepted_drafts, 0);
            CHECK_EQ(max_burst(off), 1);
        }
        CHECK_EQ(full.c.finish, CERE_FINISH_LENGTH);
    }
    CHECK(kept > 0 && kept < offered);
    CHECK(accepted > 0 && accepted <= kept);
}

// GS-02 rows: two constrained tickets steered with drafts and an unconstrained drafting ticket share speculation
// rounds (each row verifies under its own masks); each equals its solo run (masked plain / plain), and both
// constrained tickets accept drafts.
void test_grammar_spec_rows() {
    const DraftDevice dev = kDraftDevices[0];
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    const std::vector<Gen> gens = {{{1, 2}, 12}, {{6}, 10}, {{3, 4, 5}, 12}};
    cere_executor *executor = spec_executor(dev.id);
    Desc d = rows_desc(4);
    cere_model *model = gen_model(executor, d, true);
    std::vector<cere_session *> sessions(gens.size(), nullptr);
    for (cere_session *&s : sessions) CHECK_EQ(cere_session_create(executor, model, 64, &s), CERE_OK);
    GrammarItem a(executor, model, sessions[0], CERE_OP_PREFILL, gens[0], vocab);
    GrammarItem b(executor, model, sessions[1], CERE_OP_PREFILL, gens[1], vocab);
    Job c = submit_gen(executor, model, sessions[2], CERE_OP_PREFILL, gens[2]);
    Outcome oc;
    bool done_a = false, done_b = false, done_c = false;
    for (int i = 0; i < 1000 && !(done_a && done_b && done_c); ++i) {
        if (!done_a) done_a = steer_poll(executor, a.s, true, vocab);
        if (!done_b) done_b = steer_poll(executor, b.s, true, vocab);
        if (!done_c) done_c = poll_into(executor, c.ticket, *c.work, oc);
    }
    check_same(a.s.o, grammar_run(dev.id, gens[0], false, false));
    check_same(b.s.o, grammar_run(dev.id, gens[1], false, false));
    check_same(oc, gen_run(dev.id, gens[2], false));
    CHECK(a.s.o.c.accepted_drafts > 0);
    CHECK(b.s.o.c.accepted_drafts > 0);
    CHECK(oc.c.accepted_drafts > 0);
    for (cere_session *s : sessions) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// GS-01 (ledger CR7 amendment): cere_executor_round_drafts freezes a constrained ticket's drafts for position ==
// committed once (a repeat call returns the same ids; capacity bounds the copy, *count is the frozen count) and
// cere_executor_constrain_drafts takes a prefix of them with one mask per row; anything else is ARGUMENT
// (STALE_EPOCH as poll) and changes nothing. A round under row masks commits the accepted drafts and the masked
// correction in one poll; the record= lines carry each row's mask. A ticket that cannot draft (NO_SPECULATION, a
// model without drafts) offers 0; an unconstrained ticket is ARGUMENT.
void test_grammar_spec_api() {
    const std::string path = "/tmp/cere-toy-grammar-spec-" + std::to_string(uint64_t(::getpid())) + ".txt";
    std::remove(path.c_str());
    const std::string id = "toy,record=" + path;
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    const uint64_t words = mask_words();
    std::vector<uint32_t> all(words, ~uint32_t(0));
    all.back() = vocab % 32 == 0 ? ~uint32_t(0) : (uint32_t(1) << (vocab % 32)) - 1;
    cere_executor *executor = make_executor(id.c_str());
    Desc d = rows_desc(2);
    cere_model *model = gen_model(executor, d, true);
    const uint64_t generation = generation_of(executor, model);
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &session), CERE_OK);
    const std::vector<uint32_t> seq = toy_sequence({1, 2, 3}, 8, vocab);  // the toy's own (stateless) chain
    Constrained c(session, CERE_OP_PREFILL, {1, 2, 3}, 8, generation, all);
    c.work.item.flags = 0;
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &c.work.batch, &ticket), CERE_OK);
    uint32_t ids[CERE_MAX_CONSTRAINED_DRAFTS] = {};
    uint32_t n = 0;
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, 0, ids, 3, &n), CERE_ERR_ARGUMENT);  // holds its mask
    poll_pending(executor, ticket);
    CHECK_EQ(committed_of(executor, ticket), 1);
    CHECK_EQ(c.work.out[0], seq[0]);

    // round_drafts refusals.
    CHECK_EQ(cere_executor_round_drafts(nullptr, ticket, 1, ids, 3, &n), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, 1, ids, 3, nullptr), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, 1, nullptr, 1, &n), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, 0, ids, 3, &n), CERE_ERR_ARGUMENT);  // position
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, 2, ids, 3, &n), CERE_ERR_ARGUMENT);
    cere_completion_ticket stale = ticket;
    stale.worker_epoch = kEpoch + 1;
    CHECK_EQ(cere_executor_round_drafts(executor, stale, 1, ids, 3, &n), CERE_ERR_STALE_EPOCH);
    stale = ticket;
    stale.submission_id = ticket.submission_id + 100;
    CHECK_EQ(cere_executor_round_drafts(executor, stale, 1, ids, 3, &n), CERE_ERR_ARGUMENT);
    // constrain_drafts before the freeze.
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 1, nullptr, 0, all.data(), words), CERE_ERR_ARGUMENT);
    poll_pending(executor, ticket);  // the refusals set nothing
    CHECK_EQ(committed_of(executor, ticket), 1);

    // The freeze: the proposals after seq[0] (the toy proposes its own chain), the same on every call.
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, 1, ids, 3, &n), CERE_OK);
    CHECK_EQ(n, 3);
    CHECK(std::vector<uint32_t>(ids, ids + 3) == std::vector<uint32_t>(seq.begin() + 1, seq.begin() + 4));
    uint32_t again[CERE_MAX_CONSTRAINED_DRAFTS] = {7, 7, 7};
    n = 0;
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, 1, again, 1, &n), CERE_OK);
    CHECK_EQ(n, 3);
    CHECK_EQ(again[0], ids[0]);
    CHECK_EQ(again[1], 7);  // capacity 1: one id written
    n = 0;
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, 1, nullptr, 0, &n), CERE_OK);
    CHECK_EQ(n, 3);

    // constrain_drafts refusals: more drafts than frozen, ids not the frozen prefix, masks absent / wrongly sized /
    // one all-zero row, drafts null with a count; STALE_EPOCH as poll.
    std::vector<uint32_t> rows;
    const auto push = [&](std::initializer_list<uint32_t> allowed) {
        const std::vector<uint32_t> m = mask_of_ids(allowed);
        rows.insert(rows.end(), m.begin(), m.end());
    };
    for (int i = 0; i < 4; ++i) rows.insert(rows.end(), all.begin(), all.end());
    const uint32_t four[4] = {ids[0], ids[1], ids[2], ids[2]};
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 1, four, 4, rows.data(), words), CERE_ERR_ARGUMENT);
    const uint32_t wrong[2] = {ids[0], ids[1] + 1};
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 1, wrong, 2, rows.data(), words), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 1, ids, 2, nullptr, words), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 1, ids, 2, rows.data(), words - 1), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 1, nullptr, 2, rows.data(), words), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 2, ids, 2, rows.data(), words), CERE_ERR_ARGUMENT);
    std::vector<uint32_t> zero_row = rows;
    std::fill(zero_row.begin() + 2 * words, zero_row.begin() + 3 * words, 0u);
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 1, ids, 2, zero_row.data(), words), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 1, ids, 1, zero_row.data(), words), CERE_OK);
    // ... which the last call accepted (row 2 unread): constrain_drafts may replace it before the step.
    CHECK_EQ(cere_executor_constrain_drafts(nullptr, ticket, 1, ids, 1, rows.data(), words), CERE_ERR_ARGUMENT);
    stale = ticket;
    stale.worker_epoch = kEpoch + 1;
    CHECK_EQ(cere_executor_constrain_drafts(executor, stale, 1, ids, 1, rows.data(), words), CERE_ERR_STALE_EPOCH);

    // Rows: seq[1] alone, then {seq[2], x}, then {y, z, w} without seq[3]: drafts 1 and 2 are accepted and the
    // masked correction y (the smallest allowed above seq[3]) commits with them in one poll.
    const uint32_t y = (seq[3] + 1) % vocab;
    rows.clear();
    push({ids[0]});
    push({ids[1], (ids[1] + 2) % vocab});
    push({y, (y + 5) % vocab, (y + 9) % vocab});
    std::remove(path.c_str());
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 1, ids, 2, rows.data(), words), CERE_OK);
    std::fill(rows.begin(), rows.end(), 0u);  // copied before return
    cere_completion out = completion_header();
    CHECK_EQ(cere_executor_poll(executor, ticket, &out), CERE_PENDING);
    CHECK_EQ(out.committed_tokens, 4);
    CHECK(std::vector<uint32_t>(c.work.out.begin(), c.work.out.begin() + 4) ==
          std::vector<uint32_t>({seq[0], seq[1], seq[2], y}));
    CHECK_EQ(out.verify_steps, 1);
    CHECK_EQ(out.accepted_drafts, 2);
    const std::vector<std::string> lines = lines_of(path);
    CHECK_EQ(lines.size(), 3);
    const std::vector<std::string> tails = {" mask=1 token=" + std::to_string(seq[1]),
                                            " mask=2 token=" + std::to_string(seq[2]),
                                            " mask=3 token=" + std::to_string(y)};
    for (size_t i = 0; i < lines.size() && i < tails.size(); ++i) {
        CHECK(lines[i].size() > tails[i].size() &&
              lines[i].compare(lines[i].size() - tails[i].size(), tails[i].size(), tails[i]) == 0);
    }
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, 1, ids, 3, &n), CERE_ERR_ARGUMENT);  // a past position

    // Position 4: drafts frozen (3: 4 tokens left), then a plain cere_executor_constrain drops them: one token.
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, 4, ids, 3, &n), CERE_OK);
    CHECK_EQ(n, 3);
    CHECK_EQ(constrain(executor, ticket, 4, all), CERE_OK);
    CHECK(!executor->tickets.at(ticket.submission_id)->pick.has_value());
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 4, ids, 1, rows.data(), words), CERE_ERR_ARGUMENT);
    poll_pending(executor, ticket);
    CHECK_EQ(committed_of(executor, ticket), 5);
    // Position 5: 3 tokens left, 2 drafts; the grammar refuses the first (0 drafts kept): one token.
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, 5, ids, 3, &n), CERE_OK);
    CHECK_EQ(n, 2);
    std::vector<uint32_t> refuse = all;
    refuse[ids[0] >> 5] &= ~(uint32_t(1) << (ids[0] & 31));
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 5, nullptr, 0, refuse.data(), words), CERE_OK);
    poll_pending(executor, ticket);
    CHECK_EQ(committed_of(executor, ticket), 6);
    CHECK(c.work.out[5] != ids[0]);
    // Position 6: 2 tokens left, 1 draft, kept and accepted: the ticket ends (LENGTH) in that poll.
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, 6, ids, 3, &n), CERE_OK);
    CHECK_EQ(n, 1);
    rows.assign(all.begin(), all.end());
    rows.insert(rows.end(), all.begin(), all.end());
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 6, ids, 1, rows.data(), words), CERE_OK);
    out = completion_header();
    CHECK_EQ(cere_executor_poll(executor, ticket, &out), CERE_OK);
    CHECK_EQ(out.finish, CERE_FINISH_LENGTH);
    CHECK_EQ(out.committed_tokens, 8);
    CHECK_EQ(out.verify_steps, 4);
    CHECK_EQ(out.accepted_drafts, 3);
    CHECK_EQ(c.work.out[6], ids[0]);
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, 8, ids, 3, &n), CERE_ERR_ARGUMENT);  // released

    // NO_SPECULATION: offers 0 (and constrain_drafts takes 0 drafts); an unconstrained ticket is ARGUMENT.
    Constrained ns(session, CERE_OP_DECODE, {}, 2, generation, all);
    CHECK_EQ(cere_executor_submit(executor, &ns.work.batch, &ticket), CERE_OK);
    poll_pending(executor, ticket);
    n = 9;
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, 1, ids, 3, &n), CERE_OK);
    CHECK_EQ(n, 0);
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 1, ids, 1, rows.data(), words), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 1, nullptr, 0, all.data(), words), CERE_OK);
    CHECK_EQ(drain(executor, ticket).committed_tokens, 2);
    Work w(session, CERE_OP_DECODE, {}, 8, generation, {}, session->revision);
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
    poll_pending(executor, ticket);
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, committed_of(executor, ticket), ids, 3, &n),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, committed_of(executor, ticket), nullptr, 0, all.data(),
                                            words),
             CERE_ERR_ARGUMENT);
    drain(executor, ticket);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    // A model without drafts: a constrained ticket offers 0.
    executor = make_executor();
    Desc dp = rows_desc(1);
    model = gen_model(executor, dp, false);
    CHECK_EQ(cere_session_create(executor, model, 64, &session), CERE_OK);
    Constrained p(session, CERE_OP_PREFILL, {1, 2, 3}, 2, generation_of(executor, model), all);
    p.work.item.flags = 0;
    CHECK_EQ(cere_executor_submit(executor, &p.work.batch, &ticket), CERE_OK);
    poll_pending(executor, ticket);
    n = 9;
    CHECK_EQ(cere_executor_round_drafts(executor, ticket, 1, ids, 3, &n), CERE_OK);
    CHECK_EQ(n, 0);
    CHECK_EQ(cere_executor_constrain_drafts(executor, ticket, 1, nullptr, 0, all.data(), words), CERE_OK);
    CHECK_EQ(drain(executor, ticket).committed_tokens, 2);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    std::remove(path.c_str());
}

// GS-R2 (B2 on the toy, the C9 method): sampled constrained speculation preserves the masked plain distribution. On
// the toy_small_vocab model (vocab 64, temperature 1) 100,000 seeded 4-token constrained continuations steered with
// drafts under the grammar (modulus 4) match the analytic masked plain distribution (the toy's masked next, its row
// through the plain sampler stages under the prefix's mask): chi-square per position and over the sequence, p >=
// 1e-3 / 5; every token is allowed; the grammar refuses some drafts and keeps others, and kept drafts are both
// accepted and rejected.
void test_grammar_spec_sampled_distribution() {
    namespace sr = cere::sampler_ref;
    const uint32_t vocab = cere::toy::kSmallVocab;
    constexpr uint32_t kModulus = 3;  // not 4: 31 = -1 and 64 = 0 (mod 4) make the toy's drafts always allowed
    const std::vector<uint32_t> prompt = {3, 17};
    constexpr uint64_t kRuns = 100000;
    constexpr uint64_t kLen = 4;
    // p(x | outputs so far, last, position) under the grammar's mask.
    const auto conditional = [&](const std::vector<uint32_t> &prefix, uint32_t last, uint64_t position) {
        const std::vector<uint32_t> mask = grammar_mask(prefix, vocab, kModulus);
        uint32_t next = cere::toy::next_token(last, position, vocab);
        for (uint32_t i = 0; i < vocab && ((mask[next >> 5] >> (next & 31)) & 1u) == 0; ++i) {
            next = next + 1 == vocab ? 0 : next + 1;
        }
        const std::vector<uint16_t> row = expected_toy_row(next, vocab);
        std::vector<float> x(vocab);
        std::vector<uint32_t> counts(vocab, 0);
        sr::process(row.data(), vocab, counts.data(), 0.0f, 0.0f, 1.0f, mask.data(), nullptr, 0, x.data());
        std::vector<float> probs(vocab);
        sr::probabilities(x.data(), vocab, 1.0f, probs.data());
        return std::vector<double>(probs.begin(), probs.end());
    };
    std::map<uint64_t, double> joint;
    std::vector<std::vector<double>> marginal(kLen, std::vector<double>(vocab, 0.0));
    std::vector<uint32_t> outputs;
    const std::function<void(uint32_t, uint64_t, uint64_t, double)> walk = [&](uint32_t last, uint64_t i,
                                                                               uint64_t index, double p) {
        if (i == kLen) {
            joint[index] += p;
            return;
        }
        const std::vector<double> q = conditional(outputs, last, prompt.size() + i);
        for (uint32_t x = 0; x < vocab; ++x) {
            if (q[x] == 0.0) continue;
            marginal[i][x] += p * q[x];
            outputs.push_back(x);
            walk(x, i + 1, index * vocab + x, p * q[x]);
            outputs.pop_back();
        }
    };
    walk(prompt.back(), 0, 0, 1.0);

    cere_executor *executor = make_executor();
    Desc d(cere::toy::small_vocab_support().config, {16, 32}, {}, 1, 64);
    std::vector<uint32_t> map(vocab);
    for (uint32_t i = 0; i < vocab; ++i) map[i] = i;
    set_drafts(d, 4, &map);
    cere_model *model = sealed_model(executor, d);
    uint64_t offered = 0, kept = 0, accepted = 0;
    const auto continuation = [&](uint64_t seed, uint64_t *rounds) {
        cere_session *s = nullptr;
        CHECK_EQ(cere_session_create(executor, model, 64, &s), CERE_OK);
        const cere_sampling_params sampling = sampling_params(1.0f, vocab, seed);
        GrammarItem item(executor, model, s, CERE_OP_PREFILL, {prompt, kLen}, vocab, kModulus, &sampling);
        for (int i = 0; i < 100 && !steer_poll(executor, item.s, true, vocab, kModulus); ++i) {
        }
        CHECK_EQ(item.s.o.c.committed_tokens, kLen);
        CHECK(grammar_valid(item.s.o, vocab, kModulus));
        offered += item.s.offered;
        kept += item.s.kept;
        accepted += item.s.o.c.accepted_drafts;
        *rounds = item.s.o.c.verify_steps;
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        return item.s.o.out;
    };
    std::map<uint64_t, uint64_t> joint_seen;
    std::vector<std::vector<uint64_t>> seen(kLen, std::vector<uint64_t>(vocab, 0));
    for (uint64_t r = 0; r < kRuns; ++r) {
        uint64_t rounds = 0;
        const std::vector<uint32_t> out = continuation(r + 1, &rounds);
        uint64_t index = 0;
        for (uint64_t i = 0; i < kLen; ++i) {
            ++seen[i][out[i]];
            index = index * vocab + out[i];
        }
        ++joint_seen[index];
    }
    const double bound = 1e-3 / double(kLen + 1);
    for (uint64_t i = 0; i < kLen; ++i) CHECK(chi2_p(seen[i], marginal[i], kRuns) >= bound);
    std::vector<double> joint_p;
    std::vector<uint64_t> joint_n;
    uint64_t inside = 0;
    for (const auto &cell : joint) {
        joint_p.push_back(cell.second);
        const auto it = joint_seen.find(cell.first);
        joint_n.push_back(it == joint_seen.end() ? 0 : it->second);
        inside += joint_n.back();
    }
    CHECK_EQ(inside, kRuns);  // no sequence outside the support
    CHECK(chi2_p(joint_n, joint_p, kRuns) >= bound);
    CHECK(kept > 0 && kept < offered);       // the grammar keeps some drafts and refuses others
    CHECK(accepted > 0 && accepted < kept);  // kept drafts are accepted and rejected
    uint64_t r1 = 0, r2 = 0;
    CHECK(continuation(77, &r1) == continuation(77, &r2));  // seeded: reproducible
    CHECK_EQ(r1, r2);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// GS-03: a masked row with drafts needs explicit drafts (the device never verifies its own MTP proposals under row
// masks) of at most CERE_MAX_CONSTRAINED_DRAFTS: the toy's speculate_rows (chain=6: 4 drafts fit unmasked) refuses a
// masked row with null drafts or 4 drafts, nothing committed; 2 explicit drafts under 3 row masks run.
void test_grammar_spec_device_refuses_implicit_drafts() {
    cere_executor *executor = spec_executor("toy,stateful=1,chain=6");
    Desc d = rows_desc(1);
    cere_model *model = gen_model(executor, d, true);
    cere_session *session = prefilled_session(executor, model, {1, 2});
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    std::vector<uint32_t> masks;
    for (int i = 0; i < 5; ++i) {
        const std::vector<uint32_t> m = grammar_mask({}, vocab, 1000000);  // allows every id but 0
        masks.insert(masks.end(), m.begin(), m.end());
    }
    cere::host::ItemSampling sampling{};
    sampling.vocab_mask = masks.data();
    cere::host::SpecRowArgs row{};
    row.session = session->device.get();
    row.token = session->tokens.back();
    row.sampling = &sampling;
    row.max_commit = 4;
    row.draft_count = 2;
    cere::host::SpecRowResult result{};
    cere_status row_status = CERE_OK;
    cere::host::Device &device = *executor->device;
    CHECK_EQ(device.speculate_rows(*model->device, &row, 1, &result, &row_status), CERE_ERR_ARGUMENT);
    const uint32_t drafts[4] = {1, 2, 3, 4};
    row.drafts = drafts;
    row.draft_count = 4;
    row.max_commit = 5;
    CHECK_EQ(device.speculate_rows(*model->device, &row, 1, &result, &row_status), CERE_ERR_ARGUMENT);
    CHECK_EQ(result.committed, 0);
    row.draft_count = 2;
    CHECK_EQ(device.speculate_rows(*model->device, &row, 1, &result, &row_status), CERE_OK);
    CHECK_EQ(row_status, CERE_OK);
    CHECK(result.committed >= 1 && result.committed <= 3);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// D8 / M29S2-07: a drafting session's snapshot carries kind 10 (its pending proposals last) and the model's
// layout_id covers it; a restored session drafts and continues exactly as the original (its first round accepts the
// restored proposals); a NO_SPECULATION session's image has no kind 10 and restores non-drafting; a model without
// drafts keeps the M2.11 layout_id, and the drafting image does not import there.
void test_spec_snapshot() {
    const cere_qwen4_exp_config &cfg = cere::toy::fixture_config();
    const std::vector<uint32_t> prompt = {5, 9, 13};
    const Outcome full = gen_run("toy,stateful=1", {prompt, 13}, false);
    cere_executor *executor = spec_executor("toy,stateful=1");
    Desc d = rows_desc(4);
    d.desc.snapshot_host_bytes = uint64_t(1) << 26;
    cere_model *model = gen_model(executor, d, true);
    Desc dp = rows_desc(4);
    dp.desc.snapshot_host_bytes = uint64_t(1) << 26;
    cere_model *plain = gen_model(executor, dp, false);
    CHECK(plain->layout_id == snap::layout_id(cfg));
    CHECK(model->layout_id == snap::layout_id(cfg, snap::kDraftWritableKinds));
    CHECK(model->layout_id != plain->layout_id);

    cere_session *a = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &a), CERE_OK);
    Job pre = submit_gen(executor, model, a, CERE_OP_PREFILL, {prompt, 5});
    CHECK(finish(executor, pre.ticket, *pre.work).out == std::vector<uint32_t>(full.out.begin(), full.out.begin() + 5));
    cere_snapshot *image = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, a, a->revision, &image), CERE_OK);
    const std::vector<uint8_t> bytes = read_all(executor, image);
    CHECK_EQ(get64(bytes, kAtFlags), 0);
    CHECK_EQ(get32(bytes, kAtSectionCount), 10);
    CHECK_EQ(get32(bytes, entry_at(9)), snap::kDraft);
    const uint64_t end = get64(bytes, entry_at(9) + 8) + get64(bytes, entry_at(9) + 16);
    CHECK_EQ(get32(bytes, end - 16), 3);  // the pending proposals: the plain chain after the last committed token
    for (uint32_t k = 0; k < 3; ++k) CHECK_EQ(get32(bytes, end - 12 + 4 * k), full.out[5 + k]);
    cere_snapshot *foreign = nullptr;
    CHECK_EQ(cere_snapshot_import(executor, plain, bytes.data(), bytes.size(), &foreign), CERE_ERR_UNSUPPORTED);

    cere_session *b = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &b), CERE_OK);
    CHECK_EQ(cere_session_restore(executor, b, image), CERE_OK);
    CHECK(b->drafting);
    std::vector<Outcome> next;
    for (cere_session *s : {a, b}) {
        Job j = submit_gen(executor, model, s, CERE_OP_DECODE, {{}, 8});
        next.push_back(finish(executor, j.ticket, *j.work));
        CHECK(next.back().out == std::vector<uint32_t>(full.out.begin() + 5, full.out.end()));
        CHECK_EQ(next.back().bursts.at(0), 4);  // the (restored) proposals are accepted at once
        check_counts(next.back(), predicted(8, 8, 0, true, 0));
    }
    CHECK(std::memcmp(next[0].scores.data(), next[1].scores.data(), 8 * sizeof(cere_token_score)) == 0);

    // A NO_SPECULATION session: no kind 10, restores non-drafting.
    cere_session *c = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &c), CERE_OK);
    Job pc = submit_gen(executor, model, c, CERE_OP_PREFILL, {prompt, 5, {}, CERE_WORK_NO_SPECULATION});
    finish(executor, pc.ticket, *pc.work);
    cere_snapshot *plain_image = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, c, c->revision, &plain_image), CERE_OK);
    const std::vector<uint8_t> plain_bytes = read_all(executor, plain_image);
    CHECK_EQ(get64(plain_bytes, kAtFlags), CERE_WORK_NO_SPECULATION);
    CHECK_EQ(get32(plain_bytes, kAtSectionCount), 9);
    cere_session *e = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &e), CERE_OK);
    CHECK_EQ(cere_session_restore(executor, e, plain_image), CERE_OK);
    CHECK(!e->drafting);
    Job je = submit_gen(executor, model, e, CERE_OP_DECODE, {{}, 8});
    const Outcome eo = finish(executor, je.ticket, *je.work);
    CHECK(eo.out == std::vector<uint32_t>(full.out.begin() + 5, full.out.end()));
    check_counts(eo, {0, 0});

    for (cere_snapshot *s : {image, plain_image}) CHECK_EQ(cere_snapshot_release(executor, s), CERE_OK);
    for (cere_session *s : {a, b, c, e}) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, plain), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// Review M1 [INFERENCE] (m29a design note A6, T6 [REVISED: review-wave-fg M1]): a restored session's draft policy
// starts at its priors (restore needs a fresh session), so after a preempt/restore an adaptive greedy request picks
// other draft counts than the session that kept its learned policy; its committed tokens and scores are still plain
// greedy's (greedy accept does not depend on the draft count).
void test_spec_adaptive_restore_equals_plain() {
    const std::vector<uint32_t> prompt = {5, 9, 13};
    constexpr uint32_t kAdaptive = CERE_WORK_ADAPTIVE_DRAFTS;
    constexpr uint64_t kFirst = 13, kNext = 16;
    for (const char *dev : {"toy,draft_miss=1", "toy,stateful=1,draft_miss=1"}) {
        const Outcome full = gen_run(dev, {prompt, kFirst + kNext}, false);
        cere_executor *executor = spec_executor(dev);
        Desc d = rows_desc(4);
        d.desc.snapshot_host_bytes = uint64_t(1) << 26;
        cere_model *model = gen_model(executor, d, true);
        cere_session *a = nullptr;
        CHECK_EQ(cere_session_create(executor, model, 64, &a), CERE_OK);
        Job pre = submit_gen(executor, model, a, CERE_OP_PREFILL, {prompt, kFirst, {}, kAdaptive});
        CHECK(finish(executor, pre.ticket, *pre.work).out ==
              std::vector<uint32_t>(full.out.begin(), full.out.begin() + kFirst));
        cere_snapshot *image = nullptr;
        CHECK_EQ(cere_session_snapshot(executor, a, a->revision, &image), CERE_OK);
        cere_session *b = nullptr;
        CHECK_EQ(cere_session_create(executor, model, 64, &b), CERE_OK);
        CHECK_EQ(cere_session_restore(executor, b, image), CERE_OK);
        std::vector<uint64_t> verified;
        for (cere_session *s : {a, b}) {
            const uint64_t before = s->verified_drafts;
            Job j = submit_gen(executor, model, s, CERE_OP_DECODE, {{}, kNext, {}, kAdaptive});
            const Outcome o = finish(executor, j.ticket, *j.work);
            verified.push_back(s->verified_drafts - before);
            CHECK(o.out == std::vector<uint32_t>(full.out.begin() + kFirst, full.out.end()));
            CHECK(std::memcmp(o.scores.data(), full.scores.data() + kFirst, kNext * sizeof(cere_token_score)) == 0);
        }
        CHECK(verified[0] != verified[1]);  // the restored session's policy (at its priors) picked other windows
        CHECK_EQ(cere_snapshot_release(executor, image), CERE_OK);
        for (cere_session *s : {a, b}) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
}

// native-abi-ledger design note CR8 (M2.9 x M2.10, m210 D15⁗): with a draft model loaded a session that drafts (its
// PREFILL lacks NO_SPECULATION) never hits a NO_SPECULATION session's state: that session wrote no MTP K/V and split
// its prompt (test_spec_prefix_draft_hits covers the drafting hits; here the default 4,096-token chunk leaves a
// 300-token prompt no draft state). A NO_SPECULATION session of the same model hits. VERIFY rounds capture nothing:
// a drafting session whose decode crosses a multiple of 64 leaves no decode capture there, while a NO_SPECULATION
// session's does.
void test_spec_prefix_minimum() {
    cere_executor *executor = make_executor();
    PrefixDesc d(4, 2, 1024, 4096);
    set_drafts(d);
    cere_model *model = sealed_model(executor, d);
    cere::host::PrefixCache &tree = prefix_tree(model);
    const PToks p = ptoks(5, 300);
    cere_session *a = ns_session(executor, model, 1024, 1);
    check_cold_state(executor, a, p,
                     prun(executor, a, CERE_OP_PREFILL, p, 8, 0, 0, 0, CERE_OK, CERE_WORK_NO_SPECULATION));
    CHECK(!a->drafting);
    CHECK(stated_at(model, 1, p, 256) != nullptr);  // the NO_SPECULATION session's (a′) tail capture
    CHECK_EQ(cere_session_release(executor, a), CERE_OK);
    cere_session *b = ns_session(executor, model, 1024, 1);
    check_cold_state(executor, b, p, prun(executor, b, CERE_OP_PREFILL, p, 8, 0));  // drafting: no lookup
    CHECK_EQ(tree.stats().hits, 0);
    CHECK_EQ(cere_session_release(executor, b), CERE_OK);
    cere_session *c = ns_session(executor, model, 1024, 1);
    const PRun hit = prun(executor, c, CERE_OP_PREFILL, p, 8, 256, 0, 0, CERE_OK, CERE_WORK_NO_SPECULATION);
    check_cold_state(executor, c, p, hit);
    CHECK(!c->drafting);
    CHECK_EQ(tree.stats().hits, 1);
    CHECK_EQ(cere_session_release(executor, c), CERE_OK);

    // A 60-token prompt and 16 tokens: the decode crosses 64 (no prompt capture, floor64(59) = 0).
    const PToks q = ptoks(6, 60);
    const auto decoded = [&](const PRun &r) {
        PToks t = q;
        t.insert(t.end(), r.out.begin(), r.out.end());
        return t;
    };
    cere_session *e = ns_session(executor, model, 1024, 2);
    const PRun spec = prun(executor, e, CERE_OP_PREFILL, q, 16, 0);
    CHECK(e->drafting);
    CHECK(stated_at(model, 2, decoded(spec), 64) == nullptr);
    CHECK_EQ(cere_session_release(executor, e), CERE_OK);
    cere_session *f = ns_session(executor, model, 1024, 3);
    const PRun plain = prun(executor, f, CERE_OP_PREFILL, q, 16, 0, 0, 0, CERE_OK, CERE_WORK_NO_SPECULATION);
    CHECK(plain.out == spec.out);
    CHECK(stated_at(model, 3, decoded(plain), 64) != nullptr);
    CHECK_EQ(cere_session_release(executor, f), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// The `count=` of each record line of session `s` (toy record=: the tokens fed by each scoring call), in order.
std::vector<uint64_t> scored_counts(const std::string &path, uint32_t s) {
    std::vector<uint64_t> counts;
    const std::string head = "session=" + std::to_string(s) + " count=";
    for (const std::string &line : lines_of(path)) {
        if (line.rfind(head, 0) == 0) counts.push_back(std::stoull(line.substr(head.size())));
    }
    return counts;
}

// m210 design note D15‴ (CR8 "Minimum"): a drafting session never looks up, and a PREFILL split at the D4 points is
// not bit-identical to one call (D8 (b)), so splitting it changed its greedy output against the cache-off profile
// (live R3 on one spec + prefix profile: 48/54) for a capture it could never hit. On a cache-on model it runs the
// cache-off call layout. The toy is split-invariant, so its record shows the layout: the scoring call of a 300-token
// drafting PREFILL feeds all 300 tokens (no (a′) split at floor64(299) = 256), while a NO_SPECULATION session of the
// same model still splits there (44) and captures. Layout-neutral captures stay: with prefill_chunk=128 the drafting
// session's call ends at 128 and 256 are inserted as call ends.
void test_spec_prefix_unsplit() {
    const std::string path = "/tmp/cere-toy-spec-unsplit-" + std::to_string(uint64_t(::getpid())) + ".txt";
    const PToks p = ptoks(40, 300);
    for (const char *device : {"toy,record=", "toy,prefill_chunk=128,record="}) {
        const bool chunked = std::string(device).find("prefill_chunk") != std::string::npos;
        std::remove(path.c_str());
        const std::string id = device + path;
        cere_executor *executor = make_executor(id.c_str());
        PrefixDesc d(4, 2, 1024, 4096);
        set_drafts(d);
        cere_model *model = sealed_model(executor, d);
        cere::host::PrefixCache &tree = prefix_tree(model);
        cere_session *a = ns_session(executor, model, 1024, 1);
        check_cold_state(executor, a, p, prun(executor, a, CERE_OP_PREFILL, p, 1, 0));
        CHECK(a->drafting);
        CHECK_EQ(cere_session_release(executor, a), CERE_OK);
        if (chunked) {
            CHECK(scored_counts(path, 1) == std::vector<uint64_t>{44});  // [0, 128) [128, 256) [256, 300)
            for (const uint64_t at : {128u, 256u}) {
                const PrefixNode *end = stated_at(model, 1, p, at);
                CHECK(end != nullptr && end->kind == CaptureKind::kCallEnd);
            }
            CHECK_EQ(tree.stats().captures, 2);
        } else {
            CHECK(scored_counts(path, 1) == std::vector<uint64_t>{300});  // one call, as with the cache off
            CHECK(stated_at(model, 1, p, 256) == nullptr);
            CHECK_EQ(tree.stats().captures, 0);
            cere_session *b = ns_session(executor, model, 1024, 1);
            check_cold_state(executor, b, p,
                             prun(executor, b, CERE_OP_PREFILL, p, 1, 0, 0, 0, CERE_OK, CERE_WORK_NO_SPECULATION));
            CHECK(!b->drafting);
            CHECK(scored_counts(path, 2) == std::vector<uint64_t>{44});  // the (a′) split: [0, 256) [256, 300)
            const PrefixNode *tail = stated_at(model, 1, p, 256);
            CHECK(tail != nullptr && tail->kind == CaptureKind::kTail);
            CHECK_EQ(cere_session_release(executor, b), CERE_OK);
        }
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
    std::remove(path.c_str());
}

// The record= lines of session `s` (toy record=), each without its `session=` field: the tokens fed by each scoring
// call, the sampling, the mask's allowed count and the committed token.
std::vector<std::string> session_records(const std::string &path, uint32_t s) {
    std::vector<std::string> out;
    const std::string head = "session=" + std::to_string(s) + " ";
    for (const std::string &line : lines_of(path)) {
        if (line.rfind(head, 0) == 0) out.push_back(line.substr(head.size()));
    }
    return out;
}

// GdFix (grammar-spec design note F10, O238; qe_fast-gd-bisect-20261028: G6 changed under grammar_drafts "on"): on
// a cache-on model a greedy constrained ticket that drafts (grammar_drafts "on", GS-04) runs the prefix-cache layout
// of the same ticket under CERE_WORK_NO_SPECULATION ("off"): its 300-token PREFILL splits at the (a′) point 256 as
// the twin's does (record: a 44-token scoring call, not the one-call layout's 300), its speculation rounds end on each
// multiple of 64 they reach and capture there as decode does, and both leave the same states (kind for kind, none a
// draft state). A hot prompt that extends the output past 384 hits the decode capture at 384 on both (cached 384; the
// one-call layout looks up draft states only and missed). Cold and hot, the drafting ticket commits the twin's
// tokens and scores bit for bit, with the same record lines, and accepts drafts cold. The toy is split-invariant, so
// the layout shows in the record and the tree; on CUDA a split call is not bit-identical to one call (D8 (b)).
void test_grammar_spec_prefix_layout() {
    const std::string path = "/tmp/cere-toy-gs-layout-" + std::to_string(uint64_t(::getpid())) + ".txt";
    std::remove(path.c_str());
    const uint32_t vocab = cere::toy::fixture_config().vocab_size;
    const std::string id = "toy,stateful=1,record=" + path;
    cere_executor *executor = make_executor(id.c_str());
    PrefixDesc d(8, 2, 1024, 8192);
    set_drafts(d);
    cere_model *model = sealed_model(executor, d);
    // Session record ids follow creation order: twin cold 1, drafting cold 2, twin hot 3, drafting hot 4.
    const auto steered = [&](uint8_t ns, const PToks &prompt, uint64_t n, bool drafting) {
        cere_session *s = ns_session(executor, model, 1024, ns);
        const Gen g{prompt, n, {}, drafting ? 0u : uint32_t(CERE_WORK_NO_SPECULATION), 1024};
        GrammarItem item(executor, model, s, CERE_OP_PREFILL, g, vocab);
        for (int i = 0; i < 10000 && !steer_poll(executor, item.s, drafting, vocab); ++i) {
        }
        CHECK_EQ(item.s.o.c.terminal, 1);
        CHECK(s->drafting == drafting);
        CHECK(grammar_valid(item.s.o, vocab));
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        return item.s.o;
    };
    const PToks p = ptoks(70, 300);
    const Outcome twin = steered(1, p, 100, false);
    const Outcome spec = steered(2, p, 100, true);
    check_same(spec, twin);
    CHECK_EQ(twin.c.cached_tokens, 0);
    CHECK_EQ(spec.c.cached_tokens, 0);
    CHECK(spec.c.accepted_drafts > 0);
    CHECK(session_records(path, 1).front().rfind("count=44 ", 0) == 0);  // the (a′) split: [0, 256) [256, 300)
    CHECK(session_records(path, 2) == session_records(path, 1));
    const PToks seq = pjoin(p, p.size(), twin.out);
    CHECK(stated_at(model, 1, seq, 256) != nullptr && stated_at(model, 1, seq, 256)->kind == CaptureKind::kTail);
    CHECK(stated_at(model, 1, seq, 384) != nullptr && stated_at(model, 1, seq, 384)->kind == CaptureKind::kDecode);
    for (uint64_t at = 64; at <= 384; at += 64) {
        const PrefixNode *off = stated_at(model, 1, seq, at);
        const PrefixNode *on = stated_at(model, 2, seq, at);
        CHECK((off == nullptr) == (on == nullptr));
        if (off != nullptr && on != nullptr) {
            CHECK(on->kind == off->kind);
            CHECK(!on->draft && !off->draft);
        }
    }
    const PToks q(seq.begin(), seq.begin() + 390);
    const Outcome twin_hot = steered(1, q, 20, false);
    const Outcome spec_hot = steered(2, q, 20, true);
    check_same(spec_hot, twin_hot);
    CHECK_EQ(twin_hot.c.cached_tokens, 384);
    CHECK_EQ(spec_hot.c.cached_tokens, 384);
    CHECK(session_records(path, 4) == session_records(path, 3));
    CHECK(session_records(path, 3).front().rfind("count=6 ", 0) == 0);  // [384, 390) after the hit
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    std::remove(path.c_str());
}

// The prompt run of a drafting session (`ns` 0: un-namespaced) and its accepted drafts and verify steps.
struct DraftRun {
    PRun run;
    uint64_t accepted = 0;
    uint64_t steps = 0;
};

DraftRun draft_run(cere_executor *executor, cere_model *model, uint8_t ns, const PToks &p, uint64_t cached,
                   uint32_t flags = 0) {
    cere_session *s = ns_session(executor, model, 1024, ns);
    DraftRun r;
    r.run = prun(executor, s, CERE_OP_PREFILL, p, 12, cached, 0, 0, CERE_OK, flags);
    CHECK(s->drafting == ((flags & CERE_WORK_NO_SPECULATION) == 0));
    check_cold_state(executor, s, p, r.run);
    r.accepted = r.run.c.accepted_drafts;
    r.steps = r.run.c.verify_steps;
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    return r;
}

void check_same_run(const DraftRun &a, const DraftRun &b) {
    CHECK(a.run.out == b.run.out);
    CHECK_EQ(a.run.scores.size(), b.run.scores.size());
    CHECK(std::memcmp(a.run.scores.data(), b.run.scores.data(), a.run.scores.size() * sizeof(cere_token_score)) == 0);
}

// m210 D15⁗ (full ledger CR8): a drafting session hits a draft state, one captured by a drafting session at a
// multiple of the prefill chunk (here 128) inside its prompt, whose MTP row h - 1 took the hitting prompt's token h;
// else it falls back to the deepest such state above. The hot run's tokens, scores, accepted drafts and verify steps
// equal the cold run's: the toy pool records each row's token and MTP next id, so a hit over MTP rows the session's
// own prefill would not have written gets its first proposals wrong (the negative control marks a NO_SPECULATION
// state as a draft state by hand and loses accepted drafts). Drafting states serve NO_SPECULATION hits; NO_SPECULATION
// states never serve drafting ones; namespaces stay isolated.
void test_spec_prefix_draft_hits() {
    cere_executor *executor = make_executor("toy,prefill_chunk=128");
    PrefixDesc d(16, 2, 1024, 4096);
    set_drafts(d);
    cere_model *model = sealed_model(executor, d);
    cere::host::PrefixCache &tree = prefix_tree(model);
    const PToks p = ptoks(50, 300);
    const DraftRun cold = draft_run(executor, model, 1, p, 0);
    CHECK(cold.accepted > 0);
    for (const uint64_t at : {128u, 256u}) {
        const PrefixNode *n = stated_at(model, 1, p, at);
        CHECK(n != nullptr && n->draft && n->draft_next == p[at]);
    }
    const DraftRun hot = draft_run(executor, model, 1, p, 256);
    CHECK_EQ(tree.stats().hits, 1);
    check_same_run(hot, cold);
    CHECK_EQ(hot.accepted, cold.accepted);
    CHECK_EQ(hot.steps, cold.steps);
    CHECK_EQ(draft_run(executor, model, 9, p, 0).accepted, cold.accepted);  // another namespace: cold

    // Token 256 differs: the state at 256 drafted over another next id, so the hit falls back to 128.
    const PToks q = pjoin(p, 256, ptoks(51, 60));
    CHECK(q[256] != p[256]);
    const DraftRun q_hot = draft_run(executor, model, 1, q, 128);
    const DraftRun q_cold = draft_run(executor, model, 8, q, 0);
    check_same_run(q_hot, q_cold);
    CHECK_EQ(q_hot.accepted, q_cold.accepted);
    CHECK_EQ(q_hot.steps, q_cold.steps);

    // A NO_SPECULATION session hits the drafting session's state (its (a′) point is 256, so no split remains).
    check_same_run(draft_run(executor, model, 1, p, 256, CERE_WORK_NO_SPECULATION), cold);

    // NO_SPECULATION states (namespace 2: call end 128, (a′) 256) are never drafting hits.
    draft_run(executor, model, 2, p, 0, CERE_WORK_NO_SPECULATION);
    for (const uint64_t at : {128u, 256u}) {
        const PrefixNode *n = stated_at(model, 2, p, at);
        CHECK(n != nullptr && !n->draft);
    }
    const DraftRun no_hit = draft_run(executor, model, 2, p, 0);
    check_same_run(no_hit, cold);
    CHECK_EQ(no_hit.accepted, cold.accepted);
    // Negative control: forced to be a draft state, the NO_SPECULATION state at 256 (namespace 3, so nothing else
    // there is one) is hit and its missing MTP rows cost accepted drafts; the tokens stay the plain ones.
    draft_run(executor, model, 3, p, 0, CERE_WORK_NO_SPECULATION);
    auto *forced = const_cast<PrefixNode *>(stated_at(model, 3, p, 256));
    CHECK(forced != nullptr && !forced->draft);
    forced->draft = true;
    forced->draft_next = p[256];
    const DraftRun wrong = draft_run(executor, model, 3, p, 256);
    check_same_run(wrong, cold);
    CHECK(wrong.accepted < cold.accepted);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// Sessions (O211): on a model with drafts the D9 host tier keeps a draft state's mark and its edge's MTP rows (kind
// 10), so once namespace 1's draft states at 128 and 256 are spilled by other namespaces' captures, a drafting session
// hits 256 after loading both back, and its tokens, scores, accepted drafts and verify steps equal the never-spilled
// hit's and the cold run's. Negative control: a NO_SPECULATION state (its rows hold no MTP extend) spilled the same
// way and forced to be a draft state after the spill loses accepted drafts once loaded back; the tokens stay the
// plain ones.
void test_spec_prefix_host_tier_drafts() {
    cere_executor *executor = make_executor("toy,prefill_chunk=128");
    PrefixDesc d(2, 1, 1024, 4096, uint64_t(16) << 20);  // 3 state slots
    set_drafts(d);
    cere_model *model = sealed_model(executor, d);
    cere::host::PrefixCache &tree = prefix_tree(model);
    const PToks p = ptoks(90, 300);
    const auto spill_ns = [&](uint8_t id) {
        for (uint8_t other : {uint8_t(id + 1), uint8_t(id + 2)}) draft_run(executor, model, other, ptoks(other, 300), 0);
    };
    const DraftRun cold = draft_run(executor, model, 1, p, 0);
    CHECK(cold.accepted > 0);
    const DraftRun device_hit = draft_run(executor, model, 1, p, 256);
    check_same_run(device_hit, cold);
    CHECK_EQ(device_hit.accepted, cold.accepted);
    CHECK_EQ(tree.stats().host_spills, 0);
    spill_ns(1);
    for (const uint64_t at : {128u, 256u}) {
        const PrefixNode *n = stated_at(model, 1, p, at);
        CHECK(n != nullptr && n->host && n->draft && n->draft_next == p[at]);
    }
    const uint64_t loads = tree.stats().host_loads;
    const DraftRun reloaded = draft_run(executor, model, 1, p, 256);
    CHECK_EQ(tree.stats().host_loads, loads + 2);  // 128, then 256 below it
    check_same_run(reloaded, cold);
    CHECK_EQ(reloaded.accepted, device_hit.accepted);
    CHECK_EQ(reloaded.steps, device_hit.steps);

    draft_run(executor, model, 10, p, 0, CERE_WORK_NO_SPECULATION);
    spill_ns(10);
    auto *forced = const_cast<PrefixNode *>(stated_at(model, 10, p, 256));
    CHECK(forced != nullptr && forced->host && !forced->draft);
    forced->draft = true;
    forced->draft_next = p[256];
    const DraftRun wrong = draft_run(executor, model, 10, p, 256);
    check_same_run(wrong, cold);
    CHECK(wrong.accepted < cold.accepted);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// The end state (m210 D15⁗′) of namespace `id` at depth `b` along `t` for a drafting lookup of `t` (prefill chunk
// `chunk`): the node a drafting session with prompt t would hit there, null when it would not.
const PrefixNode *end_state_at(cere_model *model, uint8_t id, const PToks &t, uint64_t b, uint64_t chunk) {
    const auto m = prefix_tree(model).match(nsid(id), t.data(), t.size(), b, false, chunk);
    return m.h == b && m.node != nullptr && m.node->cls != 0 ? m.node : nullptr;
}

// Pool slots the tree's device states hold (an end state holds its pair and its chunk-0 K/V run).
uint64_t held_slots(cere_model *model, const std::vector<uint8_t> &ids) {
    uint64_t held = 0;
    std::vector<const PrefixNode *> todo;
    for (const uint8_t id : ids) {
        if (const PrefixNode *root = prefix_tree(model).root(nsid(id))) todo.push_back(root);
    }
    while (!todo.empty()) {
        const PrefixNode *n = todo.back();
        todo.pop_back();
        if (n->slot != PrefixNode::kNoSlot) held += n->held_slots();
        for (const auto &c : n->children) todo.push_back(c.second.get());
    }
    return held;
}

// m210 D15⁗′ stage 2a: a drafting cold call [s, e) that holds b = floor64(n − 1) strictly inside (s = floor_C(b) >= C;
// here C = 128, n = 350: b = 320, s = 256, class e − s = 94) captures an end state at b in two adjacent pool slots.
// A drafting session of an equal class (same n here, any tail after b + 1) hits it at b and resumes it in one call
// [b, e): the toy perturbs the session state unless that first step resumes with the state's class and ends at the
// cold call's end, so the hot run's tokens, scores, drafts and state equal the cold run's. A class mismatch
// (n = 370: class 114) is a miss that falls back to the chunk state at 256, and its own end capture there replaces
// the unlocked end state over other pages (O112); a mutant that forces the class to match resumes with the wrong class
// and its state differs. A NO_SPECULATION session never hits an end state, and its (a′) state at 320 replaces it (O112).
// The end state is never spilled or exported (the toy refuses to write one), its two slots are freed on eviction and
// flush.
void test_spec_prefix_end_states() {
    cere_executor *executor = make_executor("toy,prefill_chunk=128");
    PrefixDesc d(16, 2, 1024, 8192);
    set_drafts(d);
    cere_model *model = sealed_model(executor, d);
    CHECK_EQ(cere_model_set_prefix_end_state(executor, model, 1), CERE_OK);
    cere::host::PrefixCache &tree = prefix_tree(model);
    const PToks p = ptoks(60, 350);
    const DraftRun cold = draft_run(executor, model, 1, p, 0);
    const PrefixNode *e = end_state_at(model, 1, p, 320, 128);
    CHECK(e != nullptr && e->cls == 94 && e->draft && e->draft_next == p[320] && e->slot + 1 < tree.slots());
    CHECK(stated_at(model, 1, p, 320) == nullptr);  // invisible to every other lookup
    CHECK_EQ(held_slots(model, {1}), 4);             // 128, 256 and the pair
    CHECK_EQ(tree.free_slots(), tree.slots() - 4);
    const DraftRun hot = draft_run(executor, model, 1, p, 320);
    CHECK_EQ(tree.stats().hits, 1);
    check_same_run(hot, cold);
    CHECK_EQ(hot.accepted, cold.accepted);
    CHECK_EQ(hot.steps, cold.steps);
    // Same length, another tail after b + 1: the same class, a hit.
    const PToks t = pjoin(p, 330, ptoks(62, 20));
    const DraftRun t_cold = draft_run(executor, model, 7, t, 0);
    const DraftRun t_hot = draft_run(executor, model, 1, t, 320);
    check_same_run(t_hot, t_cold);
    CHECK_EQ(t_hot.accepted, t_cold.accepted);

    // κ mismatch: q extends p's first 340 tokens to 370 (class 114), so the end state is a miss and the chunk state
    // at 256 serves it; q's own end capture at 320 drops p's end state (unlocked, over other pages: O112) for its own.
    const PToks q = pjoin(p, 340, ptoks(61, 30));
    CHECK(end_state_at(model, 1, q, 320, 128) == nullptr);
    const DraftRun q_cold = draft_run(executor, model, 8, q, 0);
    const uint64_t free_before = tree.free_slots();
    const uint64_t evicted_before = tree.stats().state_evictions;
    const DraftRun q_hot = draft_run(executor, model, 1, q, 256);
    check_same_run(q_hot, q_cold);
    CHECK_EQ(q_hot.accepted, q_cold.accepted);
    CHECK_EQ(tree.free_slots(), free_before);
    CHECK_EQ(tree.stats().state_evictions, evicted_before + 1);
    CHECK(end_state_at(model, 1, p, 320, 128) == nullptr);
    const PrefixNode *qe = end_state_at(model, 1, q, 320, 128);
    CHECK(qe != nullptr && qe->cls == 114 && qe->draft_next == q[320]);
    // Mutant: the class forced to p's, so p hits q's state and resumes it with class 94: its state is not p's.
    if (qe != nullptr) {
        const_cast<PrefixNode *>(qe)->cls = 94;
        cere_session *m = ns_session(executor, model, 1024, 1);
        const PRun wrong = prun(executor, m, CERE_OP_PREFILL, p, 12, 320);
        PToks consumed = p;
        consumed.insert(consumed.end(), wrong.out.begin(), wrong.out.end() - 1);
        CHECK(state_digest(executor, m) != digest_of(consumed));
        CHECK_EQ(cere_session_release(executor, m), CERE_OK);
        const_cast<PrefixNode *>(qe)->cls = 114;
    }

    // A NO_SPECULATION session takes the deepest plain state (256), never the end state; its own (a′) capture at 320
    // drops the end state (O112: with end states off it would be captured there).
    check_same_run(draft_run(executor, model, 1, p, 256, CERE_WORK_NO_SPECULATION), cold);
    CHECK(end_state_at(model, 1, q, 320, 128) == nullptr);
    const PrefixNode *tail = stated_at(model, 1, p, 320);
    CHECK(tail != nullptr && tail->kind == CaptureKind::kTail && !tail->draft && tail->cls == 0);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);

    // A host tier and 4 + 2 slots: export serves the plain state below the end state; another prompt's states evict
    // ns 1's, and the end state leaves without a host image; slot accounting holds; a flush frees every slot.
    PrefixDesc h(4, 2, 1024, 8192, uint64_t(4) << 20);
    set_drafts(h);
    cere_model *tiered = sealed_model(executor, h);
    CHECK_EQ(cere_model_set_prefix_end_state(executor, tiered, 1), CERE_OK);
    cere::host::PrefixCache &tt = prefix_tree(tiered);
    draft_run(executor, tiered, 1, p, 0);
    CHECK(end_state_at(tiered, 1, p, 320, 128) != nullptr);
    cere_snapshot *exported = nullptr;
    CHECK_EQ(cere_prefix_cache_export(executor, tiered, nsid(1).data(), p.data(), p.size(), &exported), CERE_OK);
    cere_snapshot_info info{};
    info.struct_size = sizeof(info);
    info.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(cere_snapshot_get_info(executor, exported, &info), CERE_OK);
    CHECK_EQ(info.position, 256);
    CHECK_EQ(cere_snapshot_release(executor, exported), CERE_OK);
    const PToks r = ptoks(63, 350);
    draft_run(executor, tiered, 2, r, 0);
    CHECK(end_state_at(tiered, 2, r, 320, 128) != nullptr);
    CHECK(end_state_at(tiered, 1, p, 320, 128) == nullptr);
    CHECK(stated_at(tiered, 1, p, 320) == nullptr);  // no host image of it
    CHECK_EQ(tt.free_slots(), tt.slots() - held_slots(tiered, {1, 2}));
    CHECK_EQ(cere_model_flush_prefix_cache(executor, tiered), CERE_OK);
    CHECK_EQ(tt.free_slots(), tt.slots());
    CHECK_EQ(cere_model_release(executor, tiered), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// PrefixCapture G1: a drafting prompt below the prefill chunk (C = 512, n = 350: b = 320, s = 0, class 350) captures a
// chunk-0 end state at b when the device saves the cold call's K/V rows (",end_kv_slots=2": the pair plus a run of 2
// slots). The hot ticket hits at b and resumes it in one call [320, 350) whose run the toy checks (each slot holds the
// capture's marker), so its tokens, scores, drafts and state equal the cold run's. Without the run (a device that
// cannot serve chunk-0 end states, or a pool without 4 adjacent free slots: a skipped end pair) no end state is
// inserted and the hot ticket misses. A mutant that drops the node's run resumes without it and its state differs. A
// flush frees all 4 slots.
void test_spec_prefix_chunk0_end_states() {
    const PToks p = ptoks(70, 350);
    cere_executor *executor = make_executor("toy,prefill_chunk=512,end_kv_slots=2");
    PrefixDesc d(16, 2, 1024, 8192);
    set_drafts(d);
    cere_model *model = sealed_model(executor, d);
    CHECK_EQ(cere_model_set_prefix_end_state(executor, model, 1), CERE_OK);
    cere::host::PrefixCache &tree = prefix_tree(model);
    const DraftRun cold = draft_run(executor, model, 1, p, 0);
    const PrefixNode *e = end_state_at(model, 1, p, 320, 512);
    CHECK(e != nullptr && e->cls == 350 && e->kv_slots == 2 && e->draft && e->draft_next == p[320] &&
          e->slot + 3 < tree.slots());
    CHECK(stated_at(model, 1, p, 320) == nullptr);  // invisible to every other lookup
    CHECK_EQ(held_slots(model, {1}), 4);
    CHECK_EQ(tree.free_slots(), tree.slots() - 4);
    const DraftRun hot = draft_run(executor, model, 1, p, 320);
    check_same_run(hot, cold);
    CHECK_EQ(hot.accepted, cold.accepted);
    CHECK_EQ(hot.steps, cold.steps);
    // Mutant: the run dropped from the node, so the resume runs without the cold call's K/V: its state is not p's.
    if (e != nullptr) {
        auto *mutant = const_cast<PrefixNode *>(e);
        mutant->kv_slots = 0;
        cere_session *m = ns_session(executor, model, 1024, 1);
        const PRun wrong = prun(executor, m, CERE_OP_PREFILL, p, 12, 320);
        PToks consumed = p;
        consumed.insert(consumed.end(), wrong.out.begin(), wrong.out.end() - 1);
        CHECK(state_digest(executor, m) != digest_of(consumed));
        CHECK_EQ(cere_session_release(executor, m), CERE_OK);
        mutant->kv_slots = 2;
    }
    CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
    CHECK_EQ(tree.free_slots(), tree.slots());
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);

    // 3 pool slots (1 state slot + 2 sessions' track slots): no run of 4, so the end pair is skipped (counted) and the
    // hot ticket misses.
    PrefixDesc small(1, 2, 1024, 8192);
    set_drafts(small);
    cere_model *tight = sealed_model(executor, small);
    CHECK_EQ(cere_model_set_prefix_end_state(executor, tight, 1), CERE_OK);
    check_same_run(draft_run(executor, tight, 1, p, 0), cold);
    CHECK(end_state_at(tight, 1, p, 320, 512) == nullptr);
    CHECK_EQ(prefix_tree(tight).stats().skipped_end_pair, 1);
    check_same_run(draft_run(executor, tight, 1, p, 0), cold);
    CHECK_EQ(cere_model_release(executor, tight), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    // A device without chunk-0 end states: none captured, the hot ticket misses.
    cere_executor *plain = make_executor("toy,prefill_chunk=512");
    PrefixDesc pd(16, 2, 1024, 8192);
    set_drafts(pd);
    cere_model *pm = sealed_model(plain, pd);
    CHECK_EQ(cere_model_set_prefix_end_state(plain, pm, 1), CERE_OK);
    check_same_run(draft_run(plain, pm, 1, p, 0), cold);
    CHECK(end_state_at(pm, 1, p, 320, 512) == nullptr);
    CHECK_EQ(prefix_tree(pm).free_slots(), prefix_tree(pm).slots());
    check_same_run(draft_run(plain, pm, 1, p, 0), cold);
    CHECK_EQ(cere_model_release(plain, pm), CERE_OK);
    CHECK_EQ(cere_executor_destroy(plain), CERE_OK);
}

// m210 D15⁗″ (ResumeExact, S1 + S2): a conversation whose prompts extend each other inside one prefill chunk (C = 512,
// s = 512): n 700, 800, 900 (b 640, 768, 896; classes 188, 288, 388). With the toy's row-bit signature shared by the
// classes 100-512 (",row_sig=100-512") each turn hits the previous turn's end state (cached 640, then 768): σ, not the
// class, decides. Its resumed call also captures its own end state past the hit (one step resumes one pair and
// captures into another), so the next turn hits there and the chain 640 → 768 → 896 holds 7 slots. Every run's tokens,
// scores, drafts and state equal the cache-off cold run's (an un-namespaced session). Main's rule (σ = class, the toy's
// default) misses the end state at turns 2 and 3 (cached 512), and each turn's capture over its own pages drops the
// previous end-state leaf (O112). With ",row_sig=100-300" σ(388) is neither σ(288) nor σ(188): turn 3 takes the chunk
// state at 512 and its capture drops the whole unlocked end-state chain below 512 (O112 for chains) instead of being
// refused by it; with D17's alternative edges on (prod `-v52`) it drops it too (its tokens repeat the chain's first
// key). Before that, a mutant gives the 768 state the class 388 of q (p2 then another tail): q hits it and resumes it
// with a class whose σ is not its capture's, so the toy perturbs and q's state is not the cold one's.
void test_spec_prefix_end_chain() {
    const PToks p1 = ptoks(80, 700);
    const PToks p2 = pjoin(p1, 700, ptoks(81, 100));
    const PToks p3 = pjoin(p2, 800, ptoks(82, 100));
    struct Case {
        const char *id;
        uint64_t cached2;
        uint64_t cached3;
        uint64_t held;
        uint64_t evictions;
        bool mutant;
        bool alt;
    };
    for (const Case c : {Case{"toy,prefill_chunk=512,row_sig=100-512", 640, 768, 7, 0, false, false},
                         Case{"toy,prefill_chunk=512", 512, 512, 3, 2, false, false},
                         Case{"toy,prefill_chunk=512,row_sig=100-300", 640, 512, 3, 3, true, false},
                         Case{"toy,prefill_chunk=512,row_sig=100-300", 640, 512, 3, 3, true, true}}) {
        cere_executor *executor = make_executor(c.id);
        PrefixDesc d(32, 2, 1024, 16384);
        set_drafts(d);
        cere_model *model = sealed_model(executor, d);
        CHECK_EQ(cere_model_set_prefix_end_state(executor, model, 1), CERE_OK);
        if (c.alt) CHECK_EQ(cere_model_set_prefix_alt_edges(executor, model, 1), CERE_OK);
        cere::host::PrefixCache &tree = prefix_tree(model);
        struct Turn {
            const PToks &p;
            uint64_t cached;
            uint64_t b;
            uint64_t cls;
        };
        for (const Turn t : {Turn{p1, 0, 640, 188}, Turn{p2, c.cached2, 768, 288}, Turn{p3, c.cached3, 896, 388}}) {
            const DraftRun cold = draft_run(executor, model, 0, t.p, 0);
            CHECK(cold.accepted > 0);
            const DraftRun hot = draft_run(executor, model, 1, t.p, t.cached);
            check_same_run(hot, cold);
            CHECK_EQ(hot.accepted, cold.accepted);
            CHECK_EQ(hot.steps, cold.steps);
            const PrefixNode *e = end_state_at(model, 1, t.p, t.b, 512);
            CHECK(e != nullptr && e->cls == t.cls && e->draft && e->draft_next == t.p[t.b]);
            if (c.mutant && t.b == 768) {
                const PToks q = pjoin(p2, 800, ptoks(83, 100));
                auto *mutant = const_cast<PrefixNode *>(e);
                CHECK(end_state_at(model, 1, q, 768, 512) == nullptr);
                mutant->cls = 388;
                cere_session *m = ns_session(executor, model, 1024, 1);
                const PRun wrong = prun(executor, m, CERE_OP_PREFILL, q, 12, 768);
                PToks consumed = q;
                consumed.insert(consumed.end(), wrong.out.begin(), wrong.out.end() - 1);
                CHECK(state_digest(executor, m) != digest_of(consumed));
                CHECK_EQ(cere_session_release(executor, m), CERE_OK);
                mutant->cls = 288;
                CHECK_EQ(held_slots(model, {1}), 7);  // 512, 640, 768 and q's 896 below 768
            }
        }
        CHECK_EQ(held_slots(model, {1}), c.held);
        CHECK_EQ(tree.free_slots(), tree.slots() - c.held);
        CHECK_EQ(tree.stats().state_evictions, c.evictions);
        CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
        CHECK_EQ(tree.free_slots(), tree.slots());
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
}

// m210 D15⁗″ (ResumeExact): test_spec_prefix_end_chain's conversation (",row_sig=100-512") goes on to n 1,000 and
// 1,010 on a pool of 7 state slots (5 + 2 sessions), which its chain fills at turn 3 (the chunk state at 512 and the
// end-state pairs at 640, 768, 896). Turn 4 hits 896, whose path lock keeps 640 and 768 locked; no session hit them
// (`superseded_ends`), so its capture at 960 evicts the older, 640, instead of being skipped, and turn 5 hits 960.
// With a second session holding its hit on 640 (p1 again, not released) during turn 4, the capture evicts 768 and
// leaves 640. Main's take_run (unlocked end states, stale states, own states) skips that capture (`skipped_end_pair`
// 1, then 2) and turn 5 hits 896 again. Every run equals the cache-off cold run.
void test_spec_prefix_end_chain_superseded() {
    const PToks p1 = ptoks(84, 700);
    const PToks p2 = pjoin(p1, 700, ptoks(85, 100));
    const PToks p3 = pjoin(p2, 800, ptoks(86, 100));
    const PToks p4 = pjoin(p3, 900, ptoks(87, 100));
    const PToks p5 = pjoin(p4, 1000, ptoks(88, 10));
    for (const bool hold : {false, true}) {
        cere_executor *executor = make_executor("toy,prefill_chunk=512,row_sig=100-512");
        PrefixDesc d(5, 2, 1024, 16384);
        set_drafts(d);
        cere_model *model = sealed_model(executor, d);
        CHECK_EQ(cere_model_set_prefix_end_state(executor, model, 1), CERE_OK);
        cere::host::PrefixCache &tree = prefix_tree(model);
        CHECK_EQ(tree.slots(), 7);
        struct Turn {
            const PToks &p;
            uint64_t cached;
            uint64_t b;
            uint64_t held;
        };
        cere_session *m = nullptr;
        for (const Turn t : {Turn{p1, 0, 640, 3}, Turn{p2, 640, 768, 5}, Turn{p3, 768, 896, 7},
                             Turn{p4, 896, 960, 7}, Turn{p5, 960, 960, 7}}) {
            if (hold && &t.p == &p4) {
                m = ns_session(executor, model, 1024, 1);
                check_cold_state(executor, m, p1, prun(executor, m, CERE_OP_PREFILL, p1, 12, 640));
            }
            const DraftRun cold = draft_run(executor, model, 0, t.p, 0);
            const DraftRun hot = draft_run(executor, model, 1, t.p, t.cached);
            check_same_run(hot, cold);
            CHECK_EQ(hot.accepted, cold.accepted);
            CHECK_EQ(hot.steps, cold.steps);
            CHECK(end_state_at(model, 1, t.p, t.b, 512) != nullptr);
            CHECK_EQ(held_slots(model, {1}), t.held);
            if (m != nullptr) {
                CHECK_EQ(cere_session_release(executor, m), CERE_OK);
                m = nullptr;
            }
        }
        CHECK((end_state_at(model, 1, p4, 640, 512) != nullptr) == hold);
        CHECK((end_state_at(model, 1, p4, 768, 512) != nullptr) == !hold);
        CHECK(end_state_at(model, 1, p4, 896, 512) != nullptr);
        CHECK_EQ(tree.stats().state_evictions, 1);
        CHECK_EQ(tree.stats().skipped_end_pair, 0);
        CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
        CHECK_EQ(tree.free_slots(), tree.slots());
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
}

// m210 D15⁗″ (ResumeExact, S3): the chain in chunk 0 (C = 1,024 above every prompt, ",end_kv_slots=2"; the toy's σ
// shares the classes 65-512 as the CUDA device's dense bit does the classes up to 2,048): n 300, 420, 500 hit at 256
// (the first resume) then 384, and each resumed call [h, n) also captures, saving its K/V run as the resumed run's rows
// plus its own, so the next turn's resume finds a run of its own state (the toy checks each slot's marker); the chain
// holds 3 × 4 slots. Main's rule would miss turn 2 on class. n 700 (class 700, its own σ) hits nothing (cached 0), and
// its cold call's capture at 640 drops the chain (O112, 3 state evictions). Every run equals cold.
void test_spec_prefix_chunk0_end_chain() {
    cere_executor *executor = make_executor("toy,prefill_chunk=1024,end_kv_slots=2,row_sig=65-512");
    PrefixDesc d(32, 2, 1024, 16384);
    set_drafts(d);
    cere_model *model = sealed_model(executor, d);
    CHECK_EQ(cere_model_set_prefix_end_state(executor, model, 1), CERE_OK);
    cere::host::PrefixCache &tree = prefix_tree(model);
    const PToks p1 = ptoks(90, 300);
    const PToks p2 = pjoin(p1, 300, ptoks(91, 120));
    const PToks p3 = pjoin(p2, 420, ptoks(92, 80));
    const PToks p4 = pjoin(p3, 500, ptoks(93, 200));
    struct Turn {
        const PToks &p;
        uint64_t cached;
        uint64_t b;
        uint64_t held;
    };
    for (const Turn t : {Turn{p1, 0, 256, 4}, Turn{p2, 256, 384, 8}, Turn{p3, 384, 448, 12}, Turn{p4, 0, 640, 4}}) {
        const DraftRun cold = draft_run(executor, model, 0, t.p, 0);
        const DraftRun hot = draft_run(executor, model, 1, t.p, t.cached);
        check_same_run(hot, cold);
        CHECK_EQ(hot.accepted, cold.accepted);
        CHECK_EQ(hot.steps, cold.steps);
        const PrefixNode *e = end_state_at(model, 1, t.p, t.b, 1024);
        CHECK(e != nullptr && e->cls == t.p.size() && e->kv_slots == 2 && e->draft_next == t.p[t.b]);
        CHECK_EQ(held_slots(model, {1}), t.held);
    }
    CHECK_EQ(tree.stats().state_evictions, 3);
    CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
    CHECK_EQ(tree.free_slots(), tree.slots());
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// MixedPD: draft_run of `p` with a decoding ticket of an un-namespaced session waiting, so the prompt's calls are
// sliced; `parked_polls` counts the polls that left the prompt's call paused (each pause and the decode round owed
// after it: two per pause). The decoder's owed round may come before the prompt's lookup, so cached_tokens is checked
// on the terminal completion only.
DraftRun sliced_draft_run(cere_executor *executor, cere_model *model, uint8_t ns, const PToks &p, uint64_t cached,
                          uint64_t *parked_polls) {
    cere_session *ds = prefilled_session(executor, model, {1, 2}, 1024);
    Job dj = submit_job(executor, model, ds, CERE_OP_DECODE, {}, 400);
    cere_session *s = ns_session(executor, model, 1024, ns);
    Work w(s, CERE_OP_PREFILL, p, 12, s->model->generation, {}, 0);
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
    DraftRun r;
    *parked_polls = 0;
    for (int i = 0; i < 100000; ++i) {
        r.run.c = completion_header();
        const cere_status st = cere_executor_poll(executor, ticket, &r.run.c);
        if (st != CERE_PENDING) {
            CHECK_EQ(st, CERE_OK);
            break;
        }
        if (executor->tickets.at(ticket.submission_id)->parked) ++*parked_polls;
    }
    CHECK_EQ(r.run.c.terminal, 1);
    CHECK_EQ(r.run.c.cached_tokens, cached);
    CHECK_EQ(r.run.c.status, CERE_OK);
    r.run.out.assign(w.out.begin(), w.out.begin() + r.run.c.committed_tokens);
    r.run.scores.assign(w.scores.begin(), w.scores.begin() + r.run.c.committed_tokens);
    check_cold_state(executor, s, p, r.run);
    r.accepted = r.run.c.accepted_drafts;
    r.steps = r.run.c.verify_steps;
    CHECK_EQ(cere_executor_cancel(executor, dj.ticket), CERE_OK);
    CHECK_EQ(drain(executor, dj.ticket).finish, CERE_FINISH_CANCELLED);
    for (cere_session *x : {s, ds}) CHECK_EQ(cere_session_release(executor, x), CERE_OK);
    return r;
}

void test_prefill_slice_end_states() {
    // MixedPD × m210 D15⁗′ stage 2a: a sliced cold call [256, 350) takes its end-state pair at its first segment and
    // captures into it when the call ends; the hot run's resume call [320, 350) pauses too and resumes the state with
    // its class. Tokens, scores, accepted drafts and verify steps equal the unsliced executor's cold and hot runs.
    cere_executor *plain = make_executor("toy,prefill_chunk=128");
    PrefixDesc d(16, 2, 1024, 8192);
    set_drafts(d);
    cere_model *pm = sealed_model(plain, d);
    CHECK_EQ(cere_model_set_prefix_end_state(plain, pm, 1), CERE_OK);
    const PToks p = ptoks(60, 350);
    const DraftRun cold = draft_run(plain, pm, 1, p, 0);
    const DraftRun hot = draft_run(plain, pm, 1, p, 320);
    CHECK_EQ(cere_model_release(plain, pm), CERE_OK);
    CHECK_EQ(cere_executor_destroy(plain), CERE_OK);

    cere_executor *executor = make_executor("toy,prefill_chunk=128,prefill_segments=3");
    CHECK_EQ(cere_executor_set_prefill_slice(executor, 200), CERE_OK);
    PrefixDesc e(16, 2, 1024, 8192);
    set_drafts(e);
    cere_model *model = sealed_model(executor, e);
    CHECK_EQ(cere_model_set_prefix_end_state(executor, model, 1), CERE_OK);
    cere::host::PrefixCache &tree = prefix_tree(model);
    uint64_t parked_polls = 0;
    const DraftRun s_cold = sliced_draft_run(executor, model, 1, p, 0, &parked_polls);
    CHECK_EQ(parked_polls, 12);  // 3 calls x 2 pauses x 2 polls
    check_same_run(s_cold, cold);
    CHECK_EQ(s_cold.accepted, cold.accepted);
    CHECK_EQ(s_cold.steps, cold.steps);
    const PrefixNode *end_state = end_state_at(model, 1, p, 320, 128);
    CHECK(end_state != nullptr && end_state->cls == 94 && end_state->draft && end_state->draft_next == p[320]);
    CHECK_EQ(held_slots(model, {1}), 4);  // 128, 256 and the pair
    CHECK_EQ(tree.free_slots(), tree.slots() - 4);
    const DraftRun s_hot = sliced_draft_run(executor, model, 1, p, 320, &parked_polls);
    CHECK_EQ(parked_polls, 4);  // the one resume call
    CHECK_EQ(tree.stats().hits, 1);
    check_same_run(s_hot, hot);
    CHECK_EQ(s_hot.accepted, hot.accepted);
    CHECK_EQ(s_hot.steps, hot.steps);
    CHECK(!any_parked(executor));
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// m210 D15⁗′ stage 2a is opt-in (cere_model_set_prefix_end_state; `prefix_end_state` on the prefix profiles only): a
// drafting model with the setting off (the default, -v19) captures exactly the states main does (128 and 256 for n =
// 350, no end state at 320) and a hot drafting run hits only the chunk state at 256. Once the model had a ticket the
// setter is BUSY and the setting stays as it was (the next cold run still captures no end state); on > 1 is ARGUMENT
// and a cache-off model is UNSUPPORTED.
void test_spec_prefix_end_state_opt_in() {
    cere_executor *executor = make_executor("toy,prefill_chunk=128");
    PrefixDesc d(16, 2, 1024, 8192);
    set_drafts(d);
    cere_model *model = sealed_model(executor, d);
    cere::host::PrefixCache &tree = prefix_tree(model);
    CHECK_EQ(cere_model_set_prefix_end_state(executor, model, 2), CERE_ERR_ARGUMENT);
    const PToks p = ptoks(60, 350);
    const DraftRun cold = draft_run(executor, model, 1, p, 0);
    CHECK(end_state_at(model, 1, p, 320, 128) == nullptr);
    CHECK_EQ(held_slots(model, {1}), 2);  // 128 and 256 only
    CHECK_EQ(tree.free_slots(), tree.slots() - 2);
    const DraftRun hot = draft_run(executor, model, 1, p, 256);
    CHECK_EQ(tree.stats().hits, 1);
    check_same_run(hot, cold);
    CHECK_EQ(hot.accepted, cold.accepted);
    // After a ticket: BUSY, unchanged.
    CHECK_EQ(cere_model_set_prefix_end_state(executor, model, 1), CERE_ERR_BUSY);
    const PToks r = ptoks(63, 350);
    draft_run(executor, model, 2, r, 0);
    CHECK(end_state_at(model, 2, r, 320, 128) == nullptr);
    CHECK_EQ(held_slots(model, {1, 2}), 4);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    PrefixDesc off(0);
    set_drafts(off);
    cere_model *cache_off = sealed_model(executor, off);
    CHECK_EQ(cere_model_set_prefix_end_state(executor, cache_off, 1), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_model_release(executor, cache_off), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// The device states of namespaces `ids` that are not end states, as (namespace, depth, draft_next + 1 or 0).
std::set<std::array<uint64_t, 3>> plain_states(cere_model *model, const std::vector<uint8_t> &ids) {
    std::set<std::array<uint64_t, 3>> out;
    for (const uint8_t id : ids) {
        std::vector<const PrefixNode *> todo;
        if (const PrefixNode *root = prefix_tree(model).root(nsid(id))) todo.push_back(root);
        while (!todo.empty()) {
            const PrefixNode *n = todo.back();
            todo.pop_back();
            if (n->slot != PrefixNode::kNoSlot && n->cls == 0) {
                out.insert({id, n->depth, n->draft ? uint64_t(n->draft_next) + 1 : 0});
            }
            for (const auto &c : n->children) todo.push_back(c.second.get());
        }
    }
    return out;
}

// The depth a drafting lookup of `t` (prefill chunk `chunk`) under namespace `id` hits now (0: a miss).
uint64_t draft_hit(cere_model *model, uint8_t id, const PToks &t, uint64_t chunk) {
    const uint64_t cap = (t.size() - 1) / 64 * 64;
    return prefix_tree(model).match(nsid(id), t.data(), t.size(), cap, false, chunk).h;
}

// O112 (m210 design note D15⁗′ eviction): an end state never evicts another session's state. Its pair takes two
// free adjacent slots, else the slots of LRU unlocked end states, then of the capturing session's own LRU states on its
// path below its deepest one, and only when those can free a pair (else the capture is skipped); a single-slot capture
// evicts end states before any other state, except an end state its own ticket captured. Toy, prefill chunk 128, 4 + 2
// slots, no host tier; n = 350 captures drafts at 128 and 256 and an end state at 320 (4 slots), n = 300 drafts at 128
// and 256 only (2 slots; b = 256 is on the chunk grid).
void test_spec_prefix_end_state_eviction() {
    cere_executor *executor = make_executor("toy,prefill_chunk=128");
    const auto end_model = [&]() {
        PrefixDesc d(4, 2, 1024, 8192);
        set_drafts(d);
        cere_model *model = sealed_model(executor, d);
        CHECK_EQ(cere_model_set_prefix_end_state(executor, model, 1), CERE_OK);
        return model;
    };
    const auto drafts_kept = [&](cere_model *model, uint8_t id, const PToks &t) {
        for (const uint64_t at : {128u, 256u}) {
            const PrefixNode *n = stated_at(model, id, t, at);
            CHECK(n != nullptr && n->slot != PrefixNode::kNoSlot && n->draft && n->draft_next == t[at]);
        }
    };
    const PToks a = ptoks(70, 350), b = ptoks(71, 350), c = ptoks(72, 300), e = ptoks(73, 300);

    // (1) The pair evicts the older end state, not the older session's drafts (main evicted ns 1's 128 and 256).
    cere_model *model = end_model();
    cere::host::PrefixCache &tree = prefix_tree(model);
    draft_run(executor, model, 1, a, 0);
    draft_run(executor, model, 2, b, 0);
    drafts_kept(model, 1, a);
    drafts_kept(model, 2, b);
    CHECK(end_state_at(model, 1, a, 320, 128) == nullptr);
    CHECK(end_state_at(model, 2, b, 320, 128) != nullptr);
    CHECK_EQ(tree.stats().state_evictions, 1);
    CHECK_EQ(tree.free_slots(), 0);
    check_same_run(draft_run(executor, model, 1, a, 256), draft_run(executor, model, 7, a, 0));  // ns 1 hits 256
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);

    // (2) A draft capture evicts the end state first (main evicted ns 1's drafts for ns 3's 128 and 256).
    model = end_model();
    draft_run(executor, model, 1, a, 0);
    draft_run(executor, model, 2, c, 0);
    draft_run(executor, model, 3, e, 0);
    drafts_kept(model, 1, a);
    drafts_kept(model, 2, c);
    drafts_kept(model, 3, e);
    CHECK(end_state_at(model, 1, a, 320, 128) == nullptr);
    CHECK_EQ(prefix_tree(model).stats().state_evictions, 1);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);

    // (3) No free pair and no end state: the end capture is skipped and every draft stays (main evicted ns 1's). PT0a:
    // the skip counts at site end_pair only, and the total by the same one as before the split.
    model = end_model();
    draft_run(executor, model, 1, c, 0);
    draft_run(executor, model, 2, e, 0);
    const cere::host::PrefixStats before = prefix_tree(model).stats();
    draft_run(executor, model, 3, a, 0);
    drafts_kept(model, 1, c);
    drafts_kept(model, 2, e);
    drafts_kept(model, 3, a);
    CHECK(end_state_at(model, 3, a, 320, 128) == nullptr);
    const cere::host::PrefixStats &after = prefix_tree(model).stats();
    CHECK_EQ(after.capture_skipped, before.capture_skipped + 1);
    CHECK_EQ(after.skipped_end_pair, before.skipped_end_pair + 1);
    CHECK_EQ(after.skipped_prefill_slot, before.skipped_prefill_slot);
    CHECK_EQ(after.skipped_decode_track, before.skipped_decode_track);
    CHECK_EQ(after.skipped_prefill_slot + after.skipped_decode_track + after.skipped_end_pair, after.capture_skipped);
    CHECK_EQ(prefix_tree(model).stats().state_evictions, 0);
    CHECK_EQ(prefix_tree(model).free_slots(), 0);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);

    // (4) One session, more draft states than slots (n = 1,000: drafts 128 .. 896, end state at 960): its pair evicts
    // its own LRU drafts below its deepest one (896), so a repeat hits 960 (K16 (11) at 72K with 8 slots).
    model = end_model();
    const PToks g = ptoks(74, 1000);
    draft_run(executor, model, 1, g, 0);
    CHECK(end_state_at(model, 1, g, 960, 128) != nullptr);
    const PrefixNode *deepest = stated_at(model, 1, g, 896);
    CHECK(deepest != nullptr && deepest->slot != PrefixNode::kNoSlot && deepest->draft);
    const DraftRun g_hot = draft_run(executor, model, 1, g, 960);
    check_same_run(g_hot, draft_run(executor, model, 7, g, 0));
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);

    // (5) n = 384 (end state at 320, class 128) ends on the page grid: its prompt-end decode capture at 384 takes a
    // slot after the pair and evicts ns 1's LRU draft, not the end state its own ticket just captured.
    model = end_model();
    const PToks k = ptoks(75, 384);
    draft_run(executor, model, 1, a, 0);
    draft_run(executor, model, 2, k, 0);
    CHECK(end_state_at(model, 1, a, 320, 128) == nullptr);
    CHECK(end_state_at(model, 2, k, 320, 128) != nullptr);
    const DraftRun k_hot = draft_run(executor, model, 2, k, 320);
    check_same_run(k_hot, draft_run(executor, model, 7, k, 0));
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// PT0a (prefix-token design note): `hit_shortfall_tokens` sums, per drafting lookup, the tokens c − h between its
// hit h and the deepest prefill-chunk multiple c its page match k passed (c < k, within floor64(n − 1)) when no draft
// state was there; `hit_shortfall_lookups` counts those lookups. Toy, prefill chunk 128, 1 + 2 slots; p (n = 400)
// drafts at 128, 256 and 384, then two slots taken evict the interior 128 and 256 (their nodes stay, D7). q follows
// p to 330 (k = 320): it passed 256 and hits nothing, 256 lost (and it cannot re-capture there: its own pages are not
// the path's, M210S-39). A lookup over an empty tree, a hit at the deepest passed multiple (p at 384), a fallback hit
// whose page match ends at a multiple (r follows p to 256: c = 128 = h) and a NO_SPECULATION lookup lose nothing.
void test_spec_prefix_hit_shortfall() {
    cere_executor *executor = make_executor("toy,prefill_chunk=128");
    const PToks p = ptoks(76, 400);
    const PToks q = pjoin(p, 330, ptoks(77, 70));
    const auto prepared = [&](bool evict) {
        PrefixDesc d(1, 2, 1024, 4096);
        set_drafts(d);
        cere_model *model = sealed_model(executor, d);
        cere::host::PrefixCache &tree = prefix_tree(model);
        CHECK_EQ(tree.slots(), 3);
        draft_run(executor, model, 1, p, 0);  // a miss over an empty tree: k = 0
        CHECK_EQ(tree.stats().hit_shortfall_lookups, 0);
        if (evict) {
            uint32_t taken[2] = {};
            for (uint32_t &slot : taken) CHECK(tree.take_slot(&slot));
            for (const uint32_t slot : taken) tree.free_slot(slot);
            CHECK_EQ(tree.stats().state_evictions, 2);
            CHECK(stated_at(model, 1, p, 128) == nullptr && stated_at(model, 1, p, 256) == nullptr);
        }
        CHECK(stated_at(model, 1, p, 384) != nullptr);
        return model;
    };
    cere_model *model = prepared(true);
    const cere::host::PrefixStats *stats = &prefix_tree(model).stats();
    draft_run(executor, model, 1, q, 0);  // k = 320, c = 256, h = 0
    CHECK_EQ(stats->hit_shortfall_tokens, 256);
    CHECK_EQ(stats->hit_shortfall_lookups, 1);
    draft_run(executor, model, 1, p, 384);  // the hit is past every passed multiple
    CHECK_EQ(stats->hit_shortfall_tokens, 256);
    CHECK_EQ(stats->hit_shortfall_lookups, 1);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    model = prepared(false);
    stats = &prefix_tree(model).stats();
    const PToks r = pjoin(p, 256, ptoks(78, 100));
    CHECK(r[256] != p[256]);
    draft_run(executor, model, 1, r, 128);  // k = 256 ends at the draft state it cannot hit: c = 128 = h
    CHECK_EQ(stats->hit_shortfall_tokens, 0);
    CHECK_EQ(stats->hit_shortfall_lookups, 0);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    model = prepared(true);
    draft_run(executor, model, 1, q, 0, CERE_WORK_NO_SPECULATION);  // not a drafting lookup (k = 320, h = 0)
    CHECK_EQ(prefix_tree(model).stats().hit_shortfall_tokens, 0);
    CHECK_EQ(prefix_tree(model).stats().hit_shortfall_lookups, 0);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// O112: with end states on, every hit is at least as deep as the key-off hit and the end states only add hits, under
// state-slot pressure (6 + 2 slots for up to 24 states) and under page pressure (a 2,048-token pool: a session binds 16
// of its 32 pages, page eviction takes end-state leaves first); per namespace the deepest state other than an end
// state (the one a later turn hits) is never shallower than with them off (an end pair only takes end states and the
// session's own states below its deepest one). Six namespaces, interleaved: turn 1 (n = 350, end state at 320, class
// 94), turn 2 (turn 1 + 250 tokens, n = 600: drafts 384 and 512, end state at 576, class 88) and a retry of turn 2
// (two of them right after their turn 2). Turn 2 misses turn 1's end state (class 128 there) and drops it for its draft
// states (O112).
void test_spec_prefix_end_state_invariance() {
    cere_executor *executor = make_executor("toy,prefill_chunk=128");
    const uint8_t order[] = {1, 2, 1, 1, 3, 4, 2, 5, 3, 3, 6, 2, 4, 5, 4, 6, 5, 6};
    for (const auto &shape : {std::array<uint64_t, 2>{6, 8192}, std::array<uint64_t, 2>{16, 2048}}) {
        std::array<cere_model *, 2> models{};
        for (int on = 0; on < 2; ++on) {
            PrefixDesc d(shape[0], 2, 1024, shape[1]);
            set_drafts(d);
            models[on] = sealed_model(executor, d);
            CHECK_EQ(cere_model_set_prefix_end_state(executor, models[on], uint32_t(on)), CERE_OK);
        }
        std::array<int, 7> sent{};
        std::array<uint64_t, 2> cached{};
        uint64_t deeper = 0;
        for (const uint8_t id : order) {
            const PToks turn1 = ptoks(80 + id, 350);
            const PToks t = sent[id] == 0 ? turn1 : pjoin(turn1, 350, ptoks(90 + id, 250));
            ++sent[id];
            std::array<uint64_t, 2> h{};
            std::array<DraftRun, 2> runs;
            for (int on = 0; on < 2; ++on) {
                h[on] = draft_hit(models[on], id, t, 128);
                runs[on] = draft_run(executor, models[on], id, t, h[on]);
                cached[on] += h[on];
            }
            check_same_run(runs[1], runs[0]);
            CHECK(h[1] >= h[0]);
            deeper += h[1] > h[0] ? 1 : 0;
            for (const uint8_t ns : {1, 2, 3, 4, 5, 6}) {
                std::array<uint64_t, 2> deepest{};
                for (int on = 0; on < 2; ++on) {
                    for (const auto &s : plain_states(models[on], {ns})) deepest[on] = std::max(deepest[on], s[1]);
                }
                CHECK(deepest[1] >= deepest[0]);
            }
        }
        CHECK(deeper > 0);
        CHECK(cached[1] > cached[0]);
        for (cere_model *m : models) CHECK_EQ(cere_model_release(executor, m), CERE_OK);
    }
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// O247 (S10 EndPairHorizon, beat-sglang design note F10): a hot session's end pair first takes the slots of stale
// states (idle since before its previous use of the state it hits), oldest first, any session's, each path's deepest
// state excepted. Toy, prefill chunk 128, 7 + 2 slots, no host tier. Three n = 400 sessions (drafts 128, 256 and 384,
// no end state: 384 is on the chunk grid) fill the pool with draft states, as G6's capacity phase does; the hot
// session h (n = 200: a draft at 128, an end state at 192, class 72) runs cold, so its 128 is its only (deepest)
// state, it owns nothing to evict and its end pair is skipped (no hit: O112's order). An n = 300 session runs after it.
// h's next turn hits 128 and its pair evicts the remaining fillers' states below their deepest (older than h's cold
// turn), not the newer session's nor any path's deepest: the end state is captured and the turn after hits it. Main
// skips the pair again (the fillers are other sessions' draft states) and the third turn resumes at 128.
void test_spec_prefix_end_pair_horizon() {
    cere_executor *executor = make_executor("toy,prefill_chunk=128");
    PrefixDesc d(7, 2, 1024, 8192);
    set_drafts(d);
    cere_model *model = sealed_model(executor, d);
    CHECK_EQ(cere_model_set_prefix_end_state(executor, model, 1), CERE_OK);
    cere::host::PrefixCache &tree = prefix_tree(model);
    const auto draft_kept = [&](uint8_t id, const PToks &t, uint64_t at) {
        const PrefixNode *n = stated_at(model, id, t, at);
        CHECK(n != nullptr && n->slot != PrefixNode::kNoSlot && n->draft && n->draft_next == t[at]);
    };
    std::vector<PToks> fillers;
    for (const uint8_t id : {1, 2, 3}) {
        fillers.push_back(ptoks(100 + id, 400));
        draft_run(executor, model, id, fillers.back(), 0);
    }
    CHECK_EQ(tree.free_slots(), 0);
    const PToks h = ptoks(110, 200);
    draft_run(executor, model, 4, h, 0);  // cold: no hit, no horizon
    CHECK(end_state_at(model, 4, h, 192, 128) == nullptr);
    CHECK_EQ(tree.stats().skipped_end_pair, 1);
    const PToks newer = ptoks(105, 300);
    draft_run(executor, model, 5, newer, 0);
    // The three single-slot captures since (h's 128, the newer session's 128 and 256) took filler 1's three states
    // (take_slot's order, unchanged); fillers 2 and 3 keep 128, 256 and 384.
    CHECK(stated_at(model, 1, fillers[0], 128) == nullptr);
    for (uint8_t i = 1; i < 3; ++i) {
        for (const uint64_t at : {128u, 256u, 384u}) draft_kept(uint8_t(i + 1), fillers[i], at);
    }
    const uint64_t evictions = tree.stats().state_evictions;
    draft_run(executor, model, 4, h, 128);  // hits 128; its previous use was the cold turn
    CHECK_EQ(tree.stats().skipped_end_pair, 1);
    CHECK(end_state_at(model, 4, h, 192, 128) != nullptr);
    CHECK(tree.stats().state_evictions > evictions);
    for (const uint64_t at : {128u, 256u}) draft_kept(5, newer, at);  // used after h's cold turn: not stale
    for (uint8_t i = 1; i < 3; ++i) draft_kept(uint8_t(i + 1), fillers[i], 384);  // each path's deepest stays
    draft_kept(4, h, 128);
    const DraftRun hot = draft_run(executor, model, 4, h, 192);  // the end state hits
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    // The end-state hit is the cold run bit for bit (a fresh model: nothing cached).
    cere_model *fresh = sealed_model(executor, d);
    CHECK_EQ(cere_model_set_prefix_end_state(executor, fresh, 1), CERE_OK);
    check_same_run(hot, draft_run(executor, fresh, 4, h, 0));
    CHECK_EQ(cere_model_release(executor, fresh), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// O259 (ShortCache, beat-sglang design note "ShortCache (O259)"): a prompt below the prefill chunk captures no
// chunk-boundary state, so a hit-less turn of it never gets O247's reuse interval; its chunk-0 end pair takes its own
// ticket's clock as the horizon: the stale states (any session's, unlocked, used before it), oldest first, each path's
// deepest state other than an end state excepted. Toy, prefill chunk 128, ",end_kv_slots=2" (a run of 4), 12 + 2
// slots, no host tier. Two n = 897 fillers (drafts 128 .. 896, no end state: 896 is on the chunk grid) fill the pool,
// as G6's 72K capacity phase does; the short session s (n = 100: b = 64, s = 0, class 100) runs cold. Main skips its
// pair (no hit: O112's order, and the fillers are other sessions' draft states), so every turn misses; the branch
// takes the older filler's states below its deepest, keeps the newer filler's and both deepest states, and the next
// turn hits 64 bit-identical to a fresh cold run. A prompt whose end lies past chunk 0 keeps O112's order (n = 200:
// end state at 192 in chunk 1, its pair skipped while no end, superseded or own state frees a run).
void test_spec_prefix_short_end_pair() {
    cere_executor *executor = make_executor("toy,prefill_chunk=128,end_kv_slots=2");
    PrefixDesc d(12, 2, 1024, 8192);
    set_drafts(d);
    cere_model *model = sealed_model(executor, d);
    CHECK_EQ(cere_model_set_prefix_end_state(executor, model, 1), CERE_OK);
    cere::host::PrefixCache &tree = prefix_tree(model);
    const auto draft_kept = [&](uint8_t id, const PToks &t, uint64_t at) {
        const PrefixNode *n = stated_at(model, id, t, at);
        CHECK(n != nullptr && n->slot != PrefixNode::kNoSlot && n->draft && n->draft_next == t[at]);
    };
    const PToks f1 = ptoks(120, 897), f2 = ptoks(121, 897);
    draft_run(executor, model, 1, f1, 0);
    draft_run(executor, model, 2, f2, 0);
    for (uint64_t at = 128; at <= 896; at += 128) {
        draft_kept(1, f1, at);
        draft_kept(2, f2, at);
    }
    CHECK_EQ(tree.free_slots(), 0);
    CHECK_EQ(tree.stats().skipped_end_pair, 0);

    // The long prompt's cold turn: its end pair is skipped as on main (its 128 took the LRU filler state).
    const PToks lng = ptoks(123, 200);
    draft_run(executor, model, 4, lng, 0);
    CHECK(end_state_at(model, 4, lng, 192, 128) == nullptr);
    CHECK_EQ(tree.stats().skipped_end_pair, 1);
    draft_kept(4, lng, 128);

    const PToks s = ptoks(122, 100);
    const DraftRun cold = draft_run(executor, model, 3, s, 0);
    CHECK_EQ(tree.stats().skipped_end_pair, 1);
    const PrefixNode *e = end_state_at(model, 3, s, 64, 128);
    CHECK(e != nullptr && e->cls == 100 && e->kv_slots == 2 && e->draft && e->draft_next == s[64]);
    for (uint64_t at = 128; at <= 896; at += 128) draft_kept(2, f2, at);  // newer than f1's: not taken
    draft_kept(1, f1, 896);                                               // each path's deepest stays
    draft_kept(4, lng, 128);
    const DraftRun hot = draft_run(executor, model, 3, s, 64);  // the end state hits
    check_same_run(hot, cold);
    CHECK_EQ(hot.accepted, cold.accepted);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    // The end-state hit is the cold run bit for bit (a fresh model: nothing cached).
    cere_model *fresh = sealed_model(executor, d);
    CHECK_EQ(cere_model_set_prefix_end_state(executor, fresh, 1), CERE_OK);
    check_same_run(hot, draft_run(executor, fresh, 3, s, 0));
    CHECK_EQ(cere_model_release(executor, fresh), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// --- M2.13-L long-context images (m213l design note) ------------------------------------------------------------

// A vision model of the A4 entry on `executor`: the desc `v` with one visual tensor.
cere_model *vision_model(cere_executor *executor, VisionDesc &v) {
    v.names[1] = "model.visual.patch_embed.proj.weight";
    v.tensors[1].name = v.names[1].c_str();
    return sealed_model(executor, v);
}

// `lead` text ids, then one image span of `pads` rows (*offset = its first pad row), then text up to `n` tokens.
PToks vision_prompt(uint32_t seed, uint64_t lead, uint64_t pads, uint64_t n, uint64_t *offset) {
    PToks p = ptoks(seed, lead);
    p.push_back(kVisionStart);
    *offset = p.size();
    p.insert(p.end(), pads, kImagePad);
    p.push_back(kVisionEnd);
    const PToks tail = ptoks(seed + 1, n > p.size() ? n - p.size() : 0);
    p.insert(p.end(), tail.begin(), tail.end());
    return p;
}

std::vector<cere_image_input> inputs_of(const std::vector<Image> &images) {
    std::vector<cere_image_input> inputs;
    for (const Image &im : images) inputs.push_back(im.input);
    return inputs;
}

// Contract C6 written out: `prompt` with each pad row j of each image keyed 0x80000000 | (word[j % 8] & 0x7fffffff),
// word[] the 8 big-endian u32 of SHA-256(pixel_sha256 || grid_t || grid_h || grid_w as LE u32).
PToks image_key(const PToks &prompt, const std::vector<Image> &images) {
    PToks key = prompt;
    for (const Image &im : images) {
        cere::host::Sha256 h;
        h.update(im.input.pixel_sha256, sizeof(im.input.pixel_sha256));
        for (const uint32_t v : {im.input.grid_t, im.input.grid_h, im.input.grid_w}) {
            const uint8_t le[4] = {uint8_t(v), uint8_t(v >> 8), uint8_t(v >> 16), uint8_t(v >> 24)};
            h.update(le, sizeof(le));
        }
        const std::array<uint8_t, 32> d = h.finish();
        const uint64_t rows = uint64_t(im.input.grid_t) * im.input.grid_h * im.input.grid_w / 4;
        for (uint64_t j = 0; j < rows; ++j) {
            const uint8_t *w = d.data() + 4 * (j % 8);
            const uint32_t word = uint32_t(w[0]) << 24 | uint32_t(w[1]) << 16 | uint32_t(w[2]) << 8 | uint32_t(w[3]);
            key[im.input.token_offset + j] = 0x80000000u | (word & 0x7fffffffu);
        }
    }
    return key;
}

// The toy's `n` tokens of a session prefilled with `prompt` (the first scored at the prompt's position) and decoded at
// position + rope_delta (toy_device.h).
PToks toy_image_tokens(const PToks &prompt, uint64_t n, int64_t rope_delta, uint32_t vocab) {
    PToks seq = prompt;
    seq.push_back(cere::toy::next_token(seq.back(), seq.size(), vocab));
    while (seq.size() < prompt.size() + n) {
        seq.push_back(cere::toy::next_token(seq.back(), uint64_t(int64_t(seq.size()) + rope_delta), vocab));
    }
    return PToks(seq.begin() + int64_t(prompt.size()), seq.end());
}

struct ImageRun {
    PRun run;
    uint64_t digest = 0;
    int64_t rope_delta = 0;
    bool drafting = false;
};

// One PREFILL of `prompt` with `images` on a fresh session of `context` tokens (namespace `ns`, 0 = none) for
// `max_new` tokens, cached_tokens `cached` on every poll; the session's toy state digest after it.
ImageRun image_run(cere_executor *executor, cere_model *model, uint8_t ns, const PToks &prompt,
                   const std::vector<Image> &images, uint64_t max_new, uint64_t cached, uint32_t flags = 0,
                   uint64_t context = 2048) {
    cere_session *s = ns_session(executor, model, context, ns);
    const std::vector<cere_image_input> inputs = inputs_of(images);
    ImageRun r;
    r.run = prun(executor, s, CERE_OP_PREFILL, prompt, max_new, cached, 0, 0, CERE_OK, flags, &inputs);
    r.digest = state_digest(executor, s);
    r.rope_delta = s->rope_delta;
    r.drafting = s->drafting;
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    return r;
}

// Equal tokens, scores, drafts, rotary offset and state (the toy folds each pad row's pixels and grid into it).
void check_same_image_run(const ImageRun &a, const ImageRun &b) {
    CHECK(a.run.out == b.run.out);
    CHECK_EQ(a.run.scores.size(), b.run.scores.size());
    CHECK(std::memcmp(a.run.scores.data(), b.run.scores.data(), a.run.scores.size() * sizeof(cere_token_score)) == 0);
    CHECK_EQ(a.digest, b.digest);
    CHECK_EQ(a.rope_delta, b.rope_delta);
    CHECK_EQ(a.drafting, b.drafting);
    CHECK_EQ(a.run.c.accepted_drafts, b.run.c.accepted_drafts);
    CHECK_EQ(a.run.c.verify_steps, b.run.c.verify_steps);
}

// VL-L1: the vision entry is the text entry plus the vision tower (the same config, sessions, context, KV pool and
// dtypes), so a -v43 model plans its text paths as -v29's.
void test_vision_support_values() {
    const cere::host::Support &text = cere::host::a4_support();
    const cere::host::Support &vision = cere::host::a4_vision_support();
    CHECK(cere::host::config_matches(vision, text.config));
    CHECK_EQ(vision.max_sessions, text.max_sessions);
    CHECK_EQ(vision.max_context_tokens, text.max_context_tokens);
    CHECK_EQ(vision.max_kv_pool_tokens, text.max_kv_pool_tokens);
    CHECK_EQ(vision.dtype_mask, text.dtype_mask);
    CHECK(vision.vision != nullptr && text.vision == nullptr);
    CHECK(std::string(vision.name) == "qwen4_exp_nvfp4_a4_vision");
}

// VL-L2/VL-L3 (contract C2): an image anywhere in a long prompt. A 4,096-row image (the per-image cap) at rows
// 3,001..7,096 of a 7,200-token prompt prefills in 1,024-token chunks, each chunk holding some of its pad rows
// passing the image (five encodes of 4,096 rows in all), with the tokens and state of one 8,192-token call; its rows
// past the prompt rotate at position + 64 - 4,096. An image of 4,097 rows is ARGUMENT; an item takes 32 images, not 33.
void test_image_long_prompt() {
    const uint32_t vocab = a4_support().config.vocab_size;
    uint64_t at = 0;
    const PToks prompt = vision_prompt(70, 3000, 4096, 7200, &at);
    CHECK_EQ(at, 3001);
    const std::vector<Image> big{Image(at, 128, 128)};  // 64 x 64 merged rows
    std::array<ImageRun, 2> runs;
    const std::array<const char *, 2> devices{"toy,prefill_chunk=1024", "toy,prefill_chunk=8192"};
    for (int i = 0; i < 2; ++i) {
        cere_executor *executor = make_executor(devices[i]);
        VisionDesc v(8192, {16, 16});
        cere_model *model = vision_model(executor, v);
        runs[i] = image_run(executor, model, 0, prompt, big, 4, 0, 0, 8192);
        CHECK_EQ(toy(executor).image_encodes(), i == 0 ? 5 : 1);  // chunks [2,048, 3,072) .. [6,144, 7,168)
        CHECK_EQ(toy(executor).image_rows(), 4096);
        CHECK_EQ(runs[i].rope_delta, 64 - 4096);
        CHECK(runs[i].run.out == toy_image_tokens(prompt, 4, 64 - 4096, vocab));
        if (i == 0) {
            const uint64_t generation = generation_of(executor, model);
            const auto submit = [&](const PToks &p, const std::vector<Image> &images) {
                cere_session *s = ns_session(executor, model, 8192, 0);
                const std::vector<cere_image_input> inputs = inputs_of(images);
                Work w(s, CERE_OP_PREFILL, p, 1, generation);
                w.item.images = inputs.data();
                w.item.image_count = inputs.size();
                cere_completion_ticket ticket{};
                const cere_status st = cere_executor_submit(executor, &w.batch, &ticket);
                if (st == CERE_OK) CHECK_EQ(drain(executor, ticket).committed_tokens, 1);
                CHECK_EQ(cere_session_release(executor, s), CERE_OK);
                return st;
            };
            uint64_t over_at = 0;
            const PToks over = vision_prompt(71, 10, 4097, 4200, &over_at);
            CHECK_EQ(submit(over, {Image(over_at, 34, 482)}), CERE_ERR_ARGUMENT);  // 17 x 241 = 4,097 merged rows
            for (const uint64_t count : {32u, 33u}) {
                std::vector<uint64_t> offsets;
                const PToks p = image_prompt(std::vector<uint64_t>(count, 1), &offsets);
                std::vector<Image> images;
                for (const uint64_t o : offsets) images.emplace_back(o, 2, 2);
                CHECK_EQ(submit(p, images), count == 32 ? CERE_OK : CERE_ERR_ARGUMENT);
            }
        }
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
    check_same_image_run(runs[0], runs[1]);
}

// VL-L5 (contract C4): an image session decodes in step_rows beside a text session (one decode round steps both rows)
// and drafts after its images (MTP, and lookup drafts): every row past its prompt, verify row and MTP proposal rotates
// at position + rope_delta, so its tokens are the NO_SPECULATION run's and its drafts are accepted (a proposal at
// another rotary position misses in the toy); the text row beside it is unchanged.
void test_image_rows_and_drafts() {
    const uint32_t vocab = a4_support().config.vocab_size;
    std::vector<uint64_t> offsets;
    const PToks prompt = image_prompt({4, 4}, &offsets);  // two grids (1, 4, 4): rope_delta -4
    const PToks text{1, 2, 3};
    const PToks image_want = toy_image_tokens(prompt, 9, -4, vocab);
    const PToks text_want = toy_image_tokens(text, 9, 0, vocab);
    const std::vector<Image> images{Image(offsets[0], 4, 4), Image(offsets[1], 4, 4)};
    const std::vector<cere_image_input> inputs = inputs_of(images);
    for (const bool drafts : {false, true}) {
        cere_executor *executor = make_executor();
        VisionDesc v(2048, {16, 16}, 2);
        if (drafts) set_drafts(v);
        cere_model *model = vision_model(executor, v);
        const uint64_t generation = generation_of(executor, model);
        cere_session *a = ns_session(executor, model, 2048, 0);
        cere_session *b = ns_session(executor, model, 2048, 0);
        CHECK_EQ(prun(executor, a, CERE_OP_PREFILL, prompt, 1, 0, 0, 0, CERE_OK, 0, &inputs).out[0], image_want[0]);
        CHECK_EQ(prun(executor, b, CERE_OP_PREFILL, text, 1, 0).out[0], text_want[0]);
        CHECK_EQ(a->drafting, drafts);
        CHECK_EQ(a->rope_delta, -4);
        CHECK(a->images.empty() && a->positions3.empty());  // freed once the prompt is consumed
        Work da(a, CERE_OP_DECODE, {}, 8, generation, {}, 1);
        Work db(b, CERE_OP_DECODE, {}, 8, generation, {}, 1);
        cere_completion_ticket ta{};
        cere_completion_ticket tb{};
        CHECK_EQ(cere_executor_submit(executor, &da.batch, &ta), CERE_OK);
        CHECK_EQ(cere_executor_submit(executor, &db.batch, &tb), CERE_OK);
        if (!drafts) {  // one decode round: one step_rows of both rows
            poll_pending(executor, ta);
            CHECK_EQ(committed_of(executor, ta), 1);
            CHECK_EQ(committed_of(executor, tb), 1);
        }
        const cere_completion ca = drain(executor, ta);
        const cere_completion cb = drain(executor, tb);
        CHECK_EQ(ca.committed_tokens, 8);
        CHECK_EQ(cb.committed_tokens, 8);
        CHECK(da.out == PToks(image_want.begin() + 1, image_want.end()));
        CHECK(db.out == PToks(text_want.begin() + 1, text_want.end()));
        if (drafts) {
            CHECK(ca.verify_steps > 0 && ca.accepted_drafts > 0);
            CHECK(cb.verify_steps > 0 && cb.accepted_drafts > 0);
        }
        CHECK_EQ(cere_session_release(executor, a), CERE_OK);
        CHECK_EQ(cere_session_release(executor, b), CERE_OK);
        if (drafts) {
            const ImageRun mtp = image_run(executor, model, 0, prompt, images, 9, 0);
            const ImageRun lookup = image_run(executor, model, 0, prompt, images, 9, 0, CERE_WORK_LOOKUP_DRAFTS);
            const ImageRun plain = image_run(executor, model, 0, prompt, images, 9, 0, CERE_WORK_NO_SPECULATION);
            CHECK(mtp.drafting && lookup.drafting && !plain.drafting);
            CHECK(mtp.run.out == image_want);
            CHECK(lookup.run.out == image_want);
            CHECK(plain.run.out == image_want);
            CHECK(mtp.run.c.accepted_drafts > 0);
            CHECK_EQ(plain.run.c.verify_steps, 0);
        }
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
}

// VL-L7 (contract C6): an image session's prefix-cache path is its image key. The same image hits (the cold run's
// tokens, scores and state: the toy folds each pad row's pixels and grid into the state, so a wrong hit shows); other
// pixels, or another grid over the same pixel bytes, at the same ids miss; no state is keyed by the raw pad ids. Its
// tail and decode captures go along the key. A negative control: pixels the digest does not describe hit the digest's
// states (the executor trusts pixel_sha256, C1) and leave a state that is not their cold one. A restored image
// session (no key) neither looks up nor captures.
void test_prefix_image_keys() {
    const uint32_t vocab = a4_support().config.vocab_size;
    cere_executor *executor = make_executor();
    VisionDesc v(2048, {16, 16}, 2);
    v.desc.prefix_state_slots = 8;
    v.desc.snapshot_host_bytes = uint64_t(128) << 20;  // one A4 snapshot of a short session (~59 MB) and the host tier
    cere_model *model = vision_model(executor, v);
    cere::host::PrefixCache &tree = prefix_tree(model);
    uint64_t at = 0;
    const PToks prompt = vision_prompt(27, 1, 4, 208, &at);  // pad rows 2..5; tail capture at 192
    const std::vector<Image> a{Image(at, 4, 4)};
    const std::vector<Image> pixels{Image(at, 4, 4, 1000001, 0x33)};
    const std::vector<Image> grid{Image(at, 2, 8)};  // the same 16 patch rows of 0x5a: the same pixel_sha256
    CHECK(std::memcmp(grid[0].input.pixel_sha256, a[0].input.pixel_sha256, 32) == 0);
    const PToks key = image_key(prompt, a);

    const ImageRun cold = image_run(executor, model, 0, prompt, a, 70, 0);  // decodes across 256
    CHECK(cold.run.out == toy_image_tokens(prompt, 70, -2, vocab));
    const ImageRun first = image_run(executor, model, 1, prompt, a, 70, 0);
    check_same_image_run(first, cold);
    CHECK(stated_at(model, 1, key, 192) != nullptr);
    CHECK(stated_at(model, 1, prompt, 64) == nullptr);  // nothing along the raw pad ids
    PToks consumed = key;
    consumed.insert(consumed.end(), first.run.out.begin(), first.run.out.end() - 1);
    const PrefixNode *decoded = stated_at(model, 1, consumed, 256);
    CHECK(decoded != nullptr && decoded->kind == CaptureKind::kDecode);
    const uint64_t hits = tree.stats().hits;
    check_same_image_run(image_run(executor, model, 1, prompt, a, 70, 192), cold);
    CHECK_EQ(tree.stats().hits, hits + 1);

    const ImageRun pixels_cold = image_run(executor, model, 0, prompt, pixels, 70, 0);
    CHECK(pixels_cold.digest != cold.digest);
    check_same_image_run(image_run(executor, model, 1, prompt, pixels, 70, 0), pixels_cold);
    const ImageRun grid_cold = image_run(executor, model, 0, prompt, grid, 70, 0);
    CHECK(grid_cold.digest != cold.digest);
    check_same_image_run(image_run(executor, model, 1, prompt, grid, 70, 0), grid_cold);
    CHECK_EQ(tree.stats().hits, hits + 1);

    std::vector<Image> lie{Image(at, 4, 4, 1000001, 0x33)};
    std::copy(std::begin(a[0].input.pixel_sha256), std::end(a[0].input.pixel_sha256), lie[0].input.pixel_sha256);
    const ImageRun wrong = image_run(executor, model, 1, prompt, lie, 70, 192);
    CHECK(wrong.digest == cold.digest && wrong.digest != pixels_cold.digest);

    {
        cere_session *s = ns_session(executor, model, 2048, 2);
        const std::vector<cere_image_input> inputs = inputs_of(a);
        prun(executor, s, CERE_OP_PREFILL, prompt, 1, 0, 0, 0, CERE_OK, 0, &inputs);
        cere_snapshot *shot = nullptr;
        CHECK_EQ(cere_session_snapshot(executor, s, 1, &shot), CERE_OK);
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        cere_session *r = ns_session(executor, model, 2048, 2);
        CHECK_EQ(cere_session_restore(executor, r, shot), CERE_OK);
        CHECK(r->image_session && r->keys.empty());
        const uint64_t captures = tree.stats().captures;
        const PRun d = prun(executor, r, CERE_OP_DECODE, {}, 70, 0);  // decodes across 256
        const PToks want = toy_image_tokens(prompt, 71, -2, vocab);
        CHECK(d.out == PToks(want.begin() + 1, want.end()));
        CHECK_EQ(r->tracked, 0);
        CHECK_EQ(tree.stats().captures, captures);
        CHECK_EQ(cere_session_release(executor, r), CERE_OK);
        CHECK_EQ(cere_snapshot_release(executor, shot), CERE_OK);
    }
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// VL-L7 with m210 D15⁗/D15⁗′ (contract C6): a drafting image session's draft states and end state lie along its image
// key, a draft state's next id included (here the chunk state at 128 is inside the image, rows 121..136, so its next
// id is an image key word): the same image hits the end state at 320 with the cold run's tokens, scores, drafts and
// state; other pixels miss every state.
void test_prefix_image_end_state() {
    cere_executor *executor = make_executor("toy,prefill_chunk=128");
    VisionDesc v(1024, {16, 16}, 2);
    v.desc.prefix_state_slots = 16;
    v.desc.kv_pool_tokens = 8192;
    set_drafts(v);
    cere_model *model = vision_model(executor, v);
    CHECK_EQ(cere_model_set_prefix_end_state(executor, model, 1), CERE_OK);
    uint64_t at = 0;
    const PToks p = vision_prompt(63, 120, 16, 350, &at);
    CHECK_EQ(at, 121);
    const std::vector<Image> a{Image(at, 8, 8)};
    const PToks key = image_key(p, a);
    CHECK((key[128] & 0x80000000u) != 0);
    const ImageRun cold = image_run(executor, model, 7, p, a, 12, 0, 0, 1024);
    CHECK(cold.drafting && cold.run.c.accepted_drafts > 0);
    CHECK_EQ(cold.rope_delta, 4 - 16);
    check_same_image_run(image_run(executor, model, 1, p, a, 12, 0, 0, 1024), cold);
    const PrefixNode *chunk = stated_at(model, 1, key, 128);
    CHECK(chunk != nullptr && chunk->draft && chunk->draft_next == key[128]);
    const PrefixNode *e = end_state_at(model, 1, key, 320, 128);
    CHECK(e != nullptr && e->cls == 94 && e->draft && e->draft_next == key[320]);
    CHECK(stated_at(model, 1, p, 64) == nullptr);
    check_same_image_run(image_run(executor, model, 1, p, a, 12, 320, 0, 1024), cold);
    const std::vector<Image> b{Image(at, 8, 8, 1000001, 0x33)};
    const ImageRun b_cold = image_run(executor, model, 8, p, b, 12, 0, 0, 1024);
    CHECK(b_cold.digest != cold.digest);
    check_same_image_run(image_run(executor, model, 1, p, b, 12, 0, 0, 1024), b_cold);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// m210 D17 (PrefixHit): the production miss and its fix. Two drafting conversations X and Y (prefill chunk 128, end
// states on, as -v29) share a 100-token header: Y's prompts share page 0 with X's edge [0, 128) and diverge inside
// page 1. Turn 1 is 350 tokens, turns 2 and 3 append 250 each; X and Y alternate. With alternative edges off
// (M210S-39) every capture of Y is refused (its path runs through X's edge over X's pages), so Y never hits. With them
// on, Y's states go on an alternative edge over its own pages: with room, every turn from 2 on hits its previous
// turn's last chunk state (256, then 512); under page pressure (a 2,048-token pool, 16 of its 32 pages per session)
// the hits are what the tree holds at submit. Every hot run equals the cache-off run (an un-namespaced session):
// tokens, scores, accepted drafts, verify steps (D9). A flush returns every page.
void test_spec_prefix_alt_edges() {
    cere_executor *executor = make_executor("toy,prefill_chunk=128");
    const PToks header = ptoks(100, 100);
    for (const uint64_t pool : {8192u, 2048u}) {
        for (const uint32_t on : {0u, 1u}) {
            PrefixDesc d(16, 2, 1024, pool);
            set_drafts(d);
            cere_model *model = sealed_model(executor, d);
            cere::host::PrefixCache &tree = prefix_tree(model);
            CHECK_EQ(cere_model_set_prefix_end_state(executor, model, 1), CERE_OK);
            CHECK_EQ(cere_model_set_prefix_alt_edges(executor, model, on), CERE_OK);
            std::array<PToks, 2> conv{pjoin(header, 100, ptoks(101, 250)), pjoin(header, 100, ptoks(102, 250))};
            CHECK(conv[0][64] == conv[1][64] && conv[0][100] != conv[1][100]);
            uint64_t cached = 0;
            for (uint64_t turn = 1; turn <= 3; ++turn) {
                for (size_t c = 0; c < 2; ++c) {
                    if (turn > 1) conv[c] = pjoin(conv[c], conv[c].size(), ptoks(uint32_t(103 + 2 * turn + c), 250));
                    const uint64_t h = draft_hit(model, 1, conv[c], 128);
                    if (pool == 8192) CHECK_EQ(h, c == 0 || on != 0 ? 256 * (turn - 1) : 0);
                    const DraftRun hot = draft_run(executor, model, 1, conv[c], h);
                    const DraftRun cold = draft_run(executor, model, 0, conv[c], 0);
                    check_same_run(hot, cold);
                    CHECK_EQ(hot.accepted, cold.accepted);
                    CHECK_EQ(hot.steps, cold.steps);
                    cached += h;
                }
            }
            CHECK((tree.stats().capture_page_mismatch == 0) == (on != 0));
            CHECK((tree.stats().alt_edges > 0) == (on != 0));
            if (pool == 8192) CHECK_EQ(cached, on != 0 ? 1536 : 768);
            CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
            CHECK_EQ(toy(executor).free_pages(*model->device), pool / 64);
            CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        }
    }
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// m210 D17, one conversation (prod-t4 after the 06:31Z flush: a coding agent race qwen2, 390 of 391 large requests hit
// exactly 16,384 and never more, up to 90,672 prompt tokens). Turn 2 rewrites the conversation's own tokens past its
// last grid point but one, at 350 (the edge below the state at 256 keeps its first page [256, 320) and runs on over turn
// 1's pages to the draft state at 384), then turns 3 and 4 append 150 each. With alternative edges off (M210S-39) turn
// 2 hits 256 and its captures are refused (its path past 256 holds turn 1's pages), so every later turn hits 256 again.
// With them on, turn 2's states go on an alternative edge at 256 and each later turn hits its previous prompt's last
// grid state (640, then 768). Every hot run equals the cache-off run (D9).
void test_spec_prefix_alt_edges_rewrite() {
    cere_executor *executor = make_executor("toy,prefill_chunk=128");
    for (const uint32_t on : {0u, 1u}) {
        PrefixDesc d(16, 2, 1024, 8192);
        set_drafts(d);
        cere_model *model = sealed_model(executor, d);
        cere::host::PrefixCache &tree = prefix_tree(model);
        CHECK_EQ(cere_model_set_prefix_end_state(executor, model, 1), CERE_OK);
        CHECK_EQ(cere_model_set_prefix_alt_edges(executor, model, on), CERE_OK);
        const PToks t1 = ptoks(110, 600);
        const PToks t2 = pjoin(t1, 350, ptoks(111, 350));
        CHECK(t2.size() == 700 && t2[349] == t1[349] && t2[350] != t1[350]);
        const PToks t3 = pjoin(t2, 700, ptoks(112, 150));
        const PToks t4 = pjoin(t3, 850, ptoks(113, 150));
        const std::array<const PToks *, 4> turns{&t1, &t2, &t3, &t4};
        const std::array<uint64_t, 4> want_on{0, 256, 640, 768};
        const std::array<uint64_t, 4> want_off{0, 256, 256, 256};
        for (size_t i = 0; i < turns.size(); ++i) {
            const PToks &p = *turns[i];
            const uint64_t h = draft_hit(model, 1, p, 128);
            CHECK_EQ(h, on != 0 ? want_on[i] : want_off[i]);
            const DraftRun hot = draft_run(executor, model, 1, p, h);
            const DraftRun cold = draft_run(executor, model, 0, p, 0);
            check_same_run(hot, cold);
            CHECK_EQ(hot.accepted, cold.accepted);
            CHECK_EQ(hot.steps, cold.steps);
        }
        CHECK((tree.stats().capture_page_mismatch == 0) == (on != 0));
        CHECK_EQ(tree.stats().alt_edges, on != 0 ? 1 : 0);
        CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
        CHECK_EQ(toy(executor).free_pages(*model->device), 8192 / 64);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    }
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// m210 D17 is opt-in (cere_model_set_prefix_alt_edges; `prefix_alt_edges` on -v51 only): off by default; on > 1 is
// ARGUMENT; once the model had a ticket the setter is BUSY and the setting stays; a cache-off model is UNSUPPORTED.
void test_prefix_alt_edges_opt_in() {
    cere_executor *executor = make_executor("toy,prefill_chunk=128");
    PrefixDesc d(16, 2, 1024, 8192);
    set_drafts(d);
    cere_model *model = sealed_model(executor, d);
    cere::host::PrefixCache &tree = prefix_tree(model);
    CHECK(!tree.alt_edges());
    CHECK_EQ(cere_model_set_prefix_alt_edges(executor, model, 2), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_set_prefix_alt_edges(executor, model, 1), CERE_OK);
    CHECK(tree.alt_edges());
    CHECK_EQ(cere_model_set_prefix_alt_edges(executor, model, 0), CERE_OK);
    CHECK(!tree.alt_edges());
    draft_run(executor, model, 1, ptoks(60, 350), 0);
    CHECK_EQ(cere_model_set_prefix_alt_edges(executor, model, 1), CERE_ERR_BUSY);
    CHECK(!tree.alt_edges());
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    PrefixDesc off(0);
    set_drafts(off);
    cere_model *cache_off = sealed_model(executor, off);
    CHECK_EQ(cere_model_set_prefix_alt_edges(executor, cache_off, 1), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_model_release(executor, cache_off), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// m210 D17 page cost, census-shaped (an internal benchmark report: p99 of 9 in-flight sessions a
// minute). Nine drafting conversations (prefill chunk 128, end states on, as -v29) run their turns round-robin. Each
// shares a 100-token header and diverges inside page 1, like a coding agent conversations that diverge at their
// per-project memory path (token 1,742 of the system prompt). Turn 1 is 300 tokens, then each turn appends 150, for 4
// turns. Printed per config, alt edges off then on: turns that hit at least their previous prompt minus one chunk,
// cached tokens, tree pages after the run (pool pages minus free pages, with every session released), alt_edges,
// state and leaf evictions. Off, only conversation 0 takes states (M210S-39); on, each conversation owns an alternative
// edge. With ample slots and pages every turn from 2 on hits floor(prev / 128) * 128. The configs that apply pressure
// (32 slots as prod-t4, then also a 4,096-token pool) evict LRU. Every hot run equals the cache-off run (D9), and a
// flush returns every page.
void test_spec_prefix_alt_edges_census() {
    cere_executor *executor = make_executor("toy,prefill_chunk=128");
    const PToks header = ptoks(100, 100);
    constexpr size_t kConversations = 9;
    constexpr uint64_t kTurns = 4;
    struct Config {
        uint64_t slots;
        uint64_t pool;
    };
    for (const Config cfg : {Config{96, 16384}, Config{32, 16384}, Config{32, 4096}}) {
        for (const uint32_t on : {0u, 1u}) {
            PrefixDesc d(cfg.slots, 2, 1024, cfg.pool);
            set_drafts(d);
            cere_model *model = sealed_model(executor, d);
            cere::host::PrefixCache &tree = prefix_tree(model);
            CHECK_EQ(cere_model_set_prefix_end_state(executor, model, 1), CERE_OK);
            CHECK_EQ(cere_model_set_prefix_alt_edges(executor, model, on), CERE_OK);
            std::vector<PToks> conv;
            for (size_t c = 0; c < kConversations; ++c) {
                conv.push_back(pjoin(header, 100, ptoks(uint32_t(200 + c), 200)));
            }
            uint64_t cached = 0, full = 0;
            for (uint64_t turn = 1; turn <= kTurns; ++turn) {
                for (size_t c = 0; c < kConversations; ++c) {
                    const uint64_t prev = conv[c].size();
                    if (turn > 1) conv[c] = pjoin(conv[c], prev, ptoks(uint32_t(300 + 16 * turn + c), 150));
                    const uint64_t h = draft_hit(model, 1, conv[c], 128);
                    if (turn > 1 && h == prev / 128 * 128) ++full;
                    const DraftRun hot = draft_run(executor, model, 1, conv[c], h);
                    const DraftRun cold = draft_run(executor, model, 0, conv[c], 0);
                    check_same_run(hot, cold);
                    CHECK_EQ(hot.accepted, cold.accepted);
                    CHECK_EQ(hot.steps, cold.steps);
                    cached += h;
                }
            }
            const cere::host::PrefixStats &st = tree.stats();
            const uint64_t tree_pages = cfg.pool / 64 - toy(executor).free_pages(*model->device);
            std::printf("D17 census slots %llu pool %llu alt %u: full hits %llu/%llu cached %llu tree pages %llu "
                        "alt_edges %llu state_evictions %llu leaf_evictions %llu\n",
                        static_cast<unsigned long long>(cfg.slots), static_cast<unsigned long long>(cfg.pool), on,
                        static_cast<unsigned long long>(full),
                        static_cast<unsigned long long>(kConversations * (kTurns - 1)),
                        static_cast<unsigned long long>(cached), static_cast<unsigned long long>(tree_pages),
                        static_cast<unsigned long long>(st.alt_edges),
                        static_cast<unsigned long long>(st.state_evictions),
                        static_cast<unsigned long long>(st.leaf_evictions));
            if (cfg.slots == 96) {
                CHECK_EQ(full, on != 0 ? kConversations * (kTurns - 1) : kTurns - 1);
                CHECK_EQ(st.alt_edges, on != 0 ? kConversations - 1 : 0);
            }
            CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
            CHECK_EQ(toy(executor).free_pages(*model->device), cfg.pool / 64);
            CHECK_EQ(cere_model_release(executor, model), CERE_OK);
        }
    }
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// --- Host hardening (audit H1-H7, DM-2) ------------------------------------------------------------------------------

namespace {

// Arms the allocation probe for the n-th allocation made while it lives (0-based); tripped() tells whether it fired.
struct AllocFail {
    explicit AllocFail(int64_t n) {
        alloc_probe::tripped.store(false);
        alloc_probe::countdown.store(n);
    }
    ~AllocFail() { alloc_probe::countdown.store(-1); }
    bool tripped() const { return alloc_probe::tripped.load(); }
};

// More injection points than any call below makes allocations: a sweep that reaches it never saw the call complete.
constexpr int64_t kSweepLimit = 20000;

cere_snapshot *const kUnsetSnapshot = reinterpret_cast<cere_snapshot *>(uintptr_t(1));

// H1: a prefix image has token_count == fed (the cached tokens, no next token), a session image fed + 1; restoring the
// former would leave a session whose DECODE reads tokens[fed] past the end. The restore refuses it, untouched.
void test_restore_refuses_prefix_image() {
    cere_executor *executor = make_executor();
    PrefixDesc d(4, 2, 1024, 4096, uint64_t(4) << 20);
    cere_model *model = sealed_model(executor, d);
    const PToks p = ptoks(70, 300);
    cere_session *a = ns_session(executor, model, 1024, 1);
    prun(executor, a, CERE_OP_PREFILL, p, 1, 0);
    CHECK_EQ(cere_session_release(executor, a), CERE_OK);
    cere_snapshot *image = nullptr;
    CHECK_EQ(cere_prefix_cache_export(executor, model, nsid(1).data(), p.data(), p.size(), &image), CERE_OK);

    const uint64_t free_pages = toy(executor).free_pages(*model->device);
    cere_session *b = ns_session(executor, model, 1024, 0);
    CHECK_EQ(cere_session_restore(executor, b, image), CERE_ERR_ARGUMENT);
    CHECK(b->tokens.empty());
    CHECK(!b->submitted);
    CHECK_EQ(b->fed, 0);
    CHECK(!b->bound);
    CHECK_EQ(toy(executor).free_pages(*model->device), free_pages);
    // Untouched: the session still serves a prompt from scratch.
    check_cold_state(executor, b, p, prun(executor, b, CERE_OP_PREFILL, p, 2, 0));

    // The same guard at the submit: a DECODE needs a token that is not fed yet (tokens.size() > fed). Poke the
    // invariant away (as a bad restore would have) and the DECODE is refused instead of reading past the tokens.
    const uint64_t fed = b->fed;
    CHECK(b->tokens.size() > fed);
    b->fed = b->tokens.size();
    Work decode(b, CERE_OP_DECODE, {}, 1, generation_of(executor, model), {}, b->revision);
    cere_completion_ticket ticket{};
    CHECK_EQ(cere_executor_submit(executor, &decode.batch, &ticket), CERE_ERR_ARGUMENT);
    b->fed = fed;
    prun(executor, b, CERE_OP_DECODE, {}, 1, 0, 0, b->revision);

    CHECK_EQ(cere_session_release(executor, b), CERE_OK);
    CHECK_EQ(cere_snapshot_release(executor, image), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// H2: a drafting model's cache holds only NO_SPECULATION captures; the exported image must say so (header flags), or
// the import (which refuses a drafting image) could never take its own export back. The imported node stays draft-less.
void test_drafting_export_imports() {
    cere_executor *executor = make_executor();
    PrefixDesc d(4, 1, 1024, 4096, uint64_t(4) << 20);
    set_drafts(d);
    cere_model *model = sealed_model(executor, d);
    const PToks p = ptoks(71, 300);
    cere_session *a = ns_session(executor, model, 1024, 1);
    check_cold_state(executor, a, p,
                     prun(executor, a, CERE_OP_PREFILL, p, 8, 0, 0, 0, CERE_OK, CERE_WORK_NO_SPECULATION));
    CHECK_EQ(cere_session_release(executor, a), CERE_OK);
    cere_snapshot *image = nullptr;
    CHECK_EQ(cere_prefix_cache_export(executor, model, nsid(1).data(), p.data(), p.size(), &image), CERE_OK);
    CHECK_EQ(cere_prefix_cache_import(executor, model, nsid(2).data(), image), CERE_OK);
    const PrefixNode *imported = stated_at(model, 2, p, 256);
    CHECK(imported != nullptr && imported->kind == CaptureKind::kImport && !imported->draft);
    // The imported state serves a NO_SPECULATION request and ends in the cold state ...
    cere_session *b = ns_session(executor, model, 1024, 2);
    check_cold_state(executor, b, p, prun(executor, b, CERE_OP_PREFILL, p, 8, 256, 0, 0, CERE_OK, CERE_WORK_NO_SPECULATION));
    CHECK(!b->drafting);
    CHECK_EQ(cere_session_release(executor, b), CERE_OK);
    // ... and never a drafting one (no lookup: the state holds no draft next id).
    cere_session *c = ns_session(executor, model, 1024, 2);
    check_cold_state(executor, c, p, prun(executor, c, CERE_OP_PREFILL, p, 8, 0));
    CHECK(c->drafting);
    CHECK_EQ(cere_session_release(executor, c), CERE_OK);
    CHECK_EQ(cere_snapshot_release(executor, image), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// H3: a snapshot that can never fit the arena (not even with every host entry dropped) answers CAPACITY without
// dropping the host tier's entries first.
void test_unfittable_snapshot_destroys_nothing() {
    const uint64_t entry = round256(snapshot_size(192, 192));
    const uint64_t session_bytes = round256(snapshot_size(900, 901));
    CHECK(session_bytes > entry);
    cere_executor *executor = make_executor();
    PrefixDesc d(1, 1, 1024, 4096, entry);  // room for exactly one host entry
    cere_model *model = sealed_model(executor, d);
    cere::host::PrefixCache &tree = prefix_tree(model);
    const PToks x = ptoks(15, 200);
    for (const PToks &t : {x, ptoks(16, 200), ptoks(17, 200)}) {
        cere_session *s = ns_session(executor, model, 1024, 1);
        prun(executor, s, CERE_OP_PREFILL, t, 1, 0);
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    }
    CHECK_EQ(tree.stats().host_spills, 1);
    const PrefixNode *spilled = stated_at(model, 1, x, 192);
    CHECK(spilled != nullptr && spilled->host);
    const uint64_t used = model->arena.used();
    CHECK(used > 0);

    cere_session *u = ns_session(executor, model, 1024, 0);
    prun(executor, u, CERE_OP_PREFILL, ptoks(24, 900), 1, 0);
    cere_snapshot *snapshot = kUnsetSnapshot;
    CHECK_EQ(cere_session_snapshot(executor, u, 1, &snapshot), CERE_ERR_CAPACITY);
    CHECK(snapshot == nullptr);
    CHECK_EQ(tree.stats().host_drops, 0);
    CHECK_EQ(model->arena.used(), used);
    spilled = stated_at(model, 1, x, 192);
    CHECK(spilled != nullptr && spilled->host);
    CHECK_EQ(cere_session_release(executor, u), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// H4/H7: cere_prefix_cache_export under an allocation failure at every point: the path lock is released, no arena
// range or handle is left behind, and the export then works.
void test_export_allocation_failure() {
    bool completed = false;
    for (int64_t n = 0; n < kSweepLimit && !completed; ++n) {
        cere_executor *executor = make_executor();
        PrefixDesc d(4, 1, 1024, 4096, uint64_t(4) << 20);
        cere_model *model = sealed_model(executor, d);
        const PToks p = ptoks(72, 300);
        cere_session *a = ns_session(executor, model, 1024, 1);
        prun(executor, a, CERE_OP_PREFILL, p, 1, 0);
        CHECK_EQ(cere_session_release(executor, a), CERE_OK);
        cere_snapshot *out = kUnsetSnapshot;
        cere_status st = CERE_OK;
        bool tripped = false;
        {
            AllocFail arm(n);
            st = cere_prefix_cache_export(executor, model, nsid(1).data(), p.data(), p.size(), &out);
            tripped = arm.tripped();
        }
        if (tripped) {
            CHECK(st != CERE_OK);
            CHECK(out == nullptr);
            CHECK_EQ(model->arena.used(), 0);
            CHECK_EQ(model->snapshots, 0);
            CHECK(executor->snapshots.empty());
            CHECK(!prefix_tree(model).locked());
            CHECK_EQ(cere_prefix_cache_export(executor, model, nsid(1).data(), p.data(), p.size(), &out), CERE_OK);
        } else {
            CHECK_EQ(st, CERE_OK);
            completed = true;
            CHECK(n > 0);  // the sweep injected at least one failure before the call completed
        }
        CHECK_EQ(cere_snapshot_release(executor, out), CERE_OK);
        CHECK_EQ(model->arena.used(), 0);
        CHECK(!prefix_tree(model).locked());
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
    CHECK(completed);
}

// H7: cere_session_snapshot under an allocation failure at every point leaves no arena range behind.
void test_snapshot_allocation_failure() {
    bool completed = false;
    for (int64_t n = 0; n < kSweepLimit && !completed; ++n) {
        cere_executor *executor = make_executor();
        PrefixDesc d(4, 1, 1024, 4096, uint64_t(4) << 20);
        cere_model *model = sealed_model(executor, d);
        cere_session *a = ns_session(executor, model, 1024, 0);
        prun(executor, a, CERE_OP_PREFILL, ptoks(73, 100), 4, 0);
        cere_snapshot *out = kUnsetSnapshot;
        cere_status st = CERE_OK;
        bool tripped = false;
        {
            AllocFail arm(n);
            st = cere_session_snapshot(executor, a, 1, &out);
            tripped = arm.tripped();
        }
        if (tripped) {
            CHECK(st != CERE_OK);
            CHECK(out == nullptr);
            CHECK_EQ(model->arena.used(), 0);
            CHECK_EQ(model->snapshots, 0);
            CHECK(executor->snapshots.empty());
            CHECK_EQ(cere_session_snapshot(executor, a, 1, &out), CERE_OK);
        } else {
            CHECK_EQ(st, CERE_OK);
            completed = true;
            CHECK(n > 0);
        }
        CHECK_EQ(cere_snapshot_release(executor, out), CERE_OK);
        CHECK_EQ(model->arena.used(), 0);
        CHECK_EQ(cere_session_release(executor, a), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
    CHECK(completed);
}

// H6 (regression guard): cere_session_restore under an allocation failure at every point returns the session as it
// was (no tokens, nothing fed, no pages held), and the restore then works. The rollback after a device failure
// (load_state / set_rope_delta) has no injection point in the toy: not covered here.
void test_restore_allocation_failure() {
    bool completed = false;
    for (int64_t n = 0; n < kSweepLimit && !completed; ++n) {
        cere_executor *executor = make_executor();
        PrefixDesc d(4, 2, 1024, 4096, uint64_t(4) << 20);
        cere_model *model = sealed_model(executor, d);
        const PToks p = ptoks(74, 100);
        cere_session *a = ns_session(executor, model, 1024, 0);
        prun(executor, a, CERE_OP_PREFILL, p, 4, 0);
        cere_snapshot *image = nullptr;
        CHECK_EQ(cere_session_snapshot(executor, a, 1, &image), CERE_OK);
        cere_session *b = ns_session(executor, model, 1024, 1);
        const uint64_t free_pages = toy(executor).free_pages(*model->device);
        cere_status st = CERE_OK;
        bool tripped = false;
        {
            AllocFail arm(n);
            st = cere_session_restore(executor, b, image);
            tripped = arm.tripped();
        }
        if (tripped) {
            CHECK(st != CERE_OK);
            CHECK(b->tokens.empty());
            CHECK(!b->submitted);
            CHECK_EQ(b->fed, 0);
            CHECK_EQ(toy(executor).free_pages(*model->device), free_pages);
            CHECK_EQ(cere_session_restore(executor, b, image), CERE_OK);
        } else {
            CHECK_EQ(st, CERE_OK);
            completed = true;
            CHECK(n > 0);
        }
        CHECK(state_digest(executor, b) == state_digest(executor, a));
        CHECK_EQ(cere_snapshot_release(executor, image), CERE_OK);
        CHECK_EQ(cere_session_release(executor, a), CERE_OK);
        CHECK_EQ(cere_session_release(executor, b), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
    CHECK(completed);
}

// H5: the terminal poll under an allocation failure at every point: the ticket and the session's in_flight agree (a
// ticket that survives can be polled again, one that is gone leaves a releasable session), never a ticket-less
// session that stays busy forever.
void test_terminal_poll_allocation_failure() {
    bool completed = false;
    for (int64_t n = 0; n < kSweepLimit && !completed; ++n) {
        cere_executor *executor = make_executor("toy,hold_after=2");
        PrefixDesc d(4, 1, 1024, 4096);
        cere_model *model = sealed_model(executor, d);
        const PToks p = ptoks(6, 128);
        cere_session *s = ns_session(executor, model, 1024, 1);
        Work w(s, CERE_OP_PREFILL, p, 8, generation_of(executor, model));
        cere_completion_ticket ticket{};
        CHECK_EQ(cere_executor_submit(executor, &w.batch, &ticket), CERE_OK);
        for (int i = 0; i < 16; ++i) {
            cere_completion c = completion_header();
            CHECK_EQ(cere_executor_poll(executor, ticket, &c), CERE_PENDING);
        }
        CHECK_EQ(cere_executor_cancel(executor, ticket), CERE_OK);
        cere_completion c = completion_header();
        cere_status st = CERE_OK;
        bool tripped = false;
        {
            AllocFail arm(n);
            st = cere_executor_poll(executor, ticket, &c);
            tripped = arm.tripped();
        }
        if (tripped) {
            const bool ticket_kept = executor->tickets.count(ticket.submission_id) != 0;
            CHECK_EQ(ticket_kept, s->in_flight != 0);
            if (ticket_kept) {
                CHECK(st != CERE_OK);
                c = drain(executor, ticket);
                CHECK_EQ(c.finish, CERE_FINISH_CANCELLED);
            }
        } else {
            CHECK_EQ(st, CERE_OK);
            CHECK_EQ(c.terminal, 1);
            completed = true;
            CHECK(n > 0);
        }
        CHECK(executor->tickets.empty());
        CHECK_EQ(s->in_flight, 0);
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
    CHECK(completed);
}

}  // namespace

// --- WeightLab research (weightlab design note WL-02, WL-03; cere_executor.h "WeightLab research") ---------------

namespace {

constexpr uint64_t kResearchStaging = uint64_t(1) << 20;
constexpr uint64_t kLayer3 = uint64_t(1) << 3;  // full attention (layer % 4 == 3)
constexpr uint64_t kFinal = uint64_t(1) << CERE_RESEARCH_FINAL_LAYER;

// One cere_research_plan and the storage its pointers name (get() points them).
struct RPlan {
    std::vector<cere_research_op> ops;
    std::vector<std::vector<cere_research_range>> ranges;  // per op
    std::vector<std::vector<uint16_t>> vectors;            // per op: BF16 [vector_rows, vector_cols]
    cere_research_plan plan{};

    // A capture op of `point` at `layers` over the `prompt` ranges and, with `generated`, at most `gmax` generated
    // rows (0: max_tokens).
    void capture(const char *point, uint64_t layers, std::vector<cere_research_range> prompt, bool generated = false,
                 uint64_t gmax = 0, uint32_t top_k = 0) {
        cere_research_op op{};
        op.kind = CERE_RESEARCH_CAPTURE;
        op.point = point;
        op.layers = layers;
        op.generated = generated ? 1 : 0;
        op.generated_max = gmax;
        op.top_k = top_k;
        ops.push_back(op);
        ranges.push_back(std::move(prompt));
        vectors.emplace_back();
    }
    // A dense edit (`kind` add, ablate_mean or replace) of `point` with a [rows, cols] vector of 1.0.
    void edit(uint32_t kind, const char *point, uint64_t layers, uint32_t rows, uint32_t cols,
              std::vector<cere_research_range> prompt, bool generated) {
        capture(point, layers, std::move(prompt), generated);
        ops.back().kind = kind;
        ops.back().alpha = kind == CERE_RESEARCH_ADD ? 1.0f : 0.0f;
        ops.back().vector_rows = rows;
        ops.back().vector_cols = cols;
        vectors.back().assign(uint64_t(rows) * cols, 0x3f80);
    }
    const cere_research_plan *get() {
        for (size_t i = 0; i < ops.size(); ++i) {
            ops[i].prompt = ranges[i].empty() ? nullptr : ranges[i].data();
            ops[i].prompt_count = uint32_t(ranges[i].size());
            ops[i].vector = vectors[i].empty() ? nullptr : vectors[i].data();
        }
        plan.ops = ops.empty() ? nullptr : ops.data();
        plan.op_count = uint32_t(ops.size());
        for (uint32_t i = 0; i < 32; ++i) plan.sha256[i] = uint8_t(0xa0 + i);
        return &plan;
    }
};

// A model with a research reserve (`ring` host bytes, `plans` slots), sealed.
cere_model *research_model(cere_executor *executor, Desc &d, uint64_t ring = uint64_t(1) << 22, uint32_t plans = 4) {
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    CHECK_EQ(cere_model_research_reserve(executor, model, kResearchStaging, ring, plans), CERE_OK);
    for (uint64_t i = 0; i < d.tensors.size(); ++i) {
        auto bytes = pattern(d.tensors[i].bytes, uint8_t(i));
        CHECK_EQ(cere_model_put(executor, model, i, 0, bytes.data(), bytes.size()), CERE_OK);
    }
    CHECK_EQ(cere_model_seal(executor, model), CERE_OK);
    return model;
}

uint32_t dtype_size(uint32_t dtype) { return dtype == CERE_RESEARCH_DTYPE_BF16 ? 2 : 4; }

// Whether every element of `t` is the toy's (research/capture.h write_toy_row): BF16 bf16(p % 251 + L / 64), I32
// (p + L + j) % 512, F32 1 / (j + 1), row r at position positions[r].
bool toy_rows(const cere_research_tensor &t) {
    const auto *bytes = static_cast<const uint8_t *>(t.data);
    for (uint64_t r = 0; r < t.rows; ++r) {
        const uint64_t p = t.positions[r];
        for (uint32_t j = 0; j < t.cols; ++j) {
            const uint8_t *at = bytes + (r * t.cols + j) * dtype_size(t.dtype);
            if (t.dtype == CERE_RESEARCH_DTYPE_BF16) {
                uint16_t v;
                std::memcpy(&v, at, 2);
                if (v != cere::research::bf16_bits(float(p % 251) + float(t.layer) / 64.0f)) return false;
            } else if (t.dtype == CERE_RESEARCH_DTYPE_I32) {
                int32_t v;
                std::memcpy(&v, at, 4);
                if (v != int32_t((p + t.layer + j) % 512)) return false;
            } else {
                float v;
                std::memcpy(&v, at, 4);
                if (v != 1.0f / float(j + 1)) return false;
            }
        }
    }
    return true;
}

struct CapTensor {
    std::string name;
    uint32_t op, layer, dtype, cols;
    std::vector<uint64_t> positions;
    std::vector<uint8_t> data;
    bool operator==(const CapTensor &o) const {
        return name == o.name && op == o.op && layer == o.layer && dtype == o.dtype && cols == o.cols &&
               positions == o.positions && data == o.data;
    }
};

// A copy of a read capture (each tensor checked against the toy's rows).
std::vector<CapTensor> copy_capture(const cere_research_capture &c) {
    std::vector<CapTensor> out;
    uint64_t bytes = 0;
    for (uint64_t i = 0; i < c.tensor_count; ++i) {
        const cere_research_tensor &t = c.tensors[i];
        CHECK(toy_rows(t));
        const uint64_t n = t.rows * t.cols * dtype_size(t.dtype);
        bytes += n;
        const auto *data = static_cast<const uint8_t *>(t.data);
        out.push_back(CapTensor{t.name, t.op, t.layer, t.dtype, t.cols,
                                std::vector<uint64_t>(t.positions, t.positions + t.rows),
                                std::vector<uint8_t>(data, data + n)});
    }
    CHECK_EQ(c.bytes, bytes);
    return out;
}

// The positions plan: op 0 attn_mix.out at layers 3 and 47 over prompt [1, 3) and [4, 20); op 1 moe_route (two
// parts) at layer 0 over at most 2 generated rows; op 2 logits top-3 over the generated rows; op 3 attn_qkv at layer 3
// (full attention) over [0, 2) and the generated rows; op 4 an add at layer 0 (no capture).
RPlan positions_plan() {
    RPlan p;
    p.capture("native.layers.L.attn_mix.out", kLayer3 | (uint64_t(1) << 47), {{1, 3}, {4, 20}});
    p.capture("native.layers.L.moe_route", 1, {}, true, 2);
    p.capture("native.logits.topk", kFinal, {}, true, 0, 3);
    p.capture("native.layers.L.gemm.attn_qkv.out", kLayer3, {{0, 2}}, true);
    p.edit(CERE_RESEARCH_ADD, "native.layers.L.attn_mix.out", 1, 1, 2560, {}, true);
    return p;
}

// Its bound for prompt_tokens 6 and max_tokens 10: op 0 rows {1, 2, 4, 5} x 2 layers x 5,120 B; op 1 rows {5, 6} x
// (40 + 40) B; op 2 rows [5, 15) x (12 + 12) B; op 3 rows {0, 1} and [5, 15) x 26,624 B.
constexpr uint64_t kPositionsBound = 4 * 2 * 5120 + 2 * 80 + 10 * 24 + 12 * 26624;

struct CapRun {
    std::vector<CapTensor> tensors;
    uint64_t bound = 0;
    uint32_t skipped = 0;
    std::vector<uint32_t> out;
    std::vector<uint32_t> base_out;  // the same prompt on a session without a plan
    uint64_t verify_steps = 0;
    uint64_t accepted = 0;
};

// prompt {1..6} under positions_plan (max_tokens 10) on `device`: 4 tokens, as one PREFILL or (`split`) a PREFILL of
// 1 and a DECODE of 3; `drafts`: the model speculates (M2.9).
CapRun run_capture(const char *device, bool drafts, bool split) {
    cere_executor *executor = make_executor(device);
    Desc d = rows_desc(2);
    if (drafts) set_drafts(d);
    cere_model *model = research_model(executor, d);
    RPlan plan = positions_plan();
    CHECK_EQ(cere_model_research_plan_load(executor, model, 0, plan.get()), CERE_OK);
    const std::vector<uint32_t> prompt = {1, 2, 3, 4, 5, 6};
    CapRun run;
    uint64_t capture = 0;
    cere_session *s = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &s), CERE_OK);
    CHECK_EQ(cere_session_set_research(executor, s, 0, prompt.size(), 10, &capture), CERE_OK);
    CHECK(capture != 0);
    run.bound = s->research->bytes;
    CHECK_EQ(model->research_ring_alloc.used(), run.bound);
    for (const auto &[op, n] : split ? std::vector<std::pair<uint32_t, uint64_t>>{{CERE_OP_PREFILL, 1}, {CERE_OP_DECODE, 3}}
                                     : std::vector<std::pair<uint32_t, uint64_t>>{{CERE_OP_PREFILL, 4}}) {
        Job job = submit_gen(executor, model, s, op, {op == CERE_OP_PREFILL ? prompt : std::vector<uint32_t>{}, n});
        const Outcome o = finish(executor, job.ticket, *job.work);
        CHECK_EQ(o.c.committed_tokens, n);
        run.out.insert(run.out.end(), o.out.begin(), o.out.end());
        run.verify_steps += o.c.verify_steps;
        run.accepted += o.c.accepted_drafts;
    }
    cere_research_capture c{};
    CHECK_EQ(cere_model_research_capture_read(executor, model, capture, &c), CERE_ERR_BUSY);  // its session lives
    CHECK_EQ(cere_model_research_capture_free(executor, model, capture), CERE_ERR_BUSY);
    CHECK_EQ(cere_model_research_plan_unload(executor, model, 0), CERE_ERR_BUSY);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_research_capture_read(executor, model, capture, &c), CERE_OK);
    run.tensors = copy_capture(c);
    run.skipped = c.prefix_lookup_skipped;
    CHECK(std::memcmp(c.plan_sha256, plan.plan.sha256, 32) == 0);
    CHECK_EQ(cere_model_research_capture_free(executor, model, capture), CERE_OK);
    CHECK_EQ(model->research_ring_alloc.used(), 0);
    CHECK_EQ(cere_model_research_capture_read(executor, model, capture, &c), CERE_ERR_ARGUMENT);  // freed
    CHECK_EQ(cere_model_research_capture_free(executor, model, capture), CERE_ERR_ARGUMENT);

    // A session without a plan on the same model: the base run (the toy applies no edit either way).
    cere_session *b = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &b), CERE_OK);
    Job job = submit_gen(executor, model, b, CERE_OP_PREFILL, {prompt, 4});
    run.base_out = finish(executor, job.ticket, *job.work).out;
    CHECK_EQ(cere_session_release(executor, b), CERE_OK);
    CHECK_EQ(cere_model_research_plan_unload(executor, model, 0), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    return run;
}

void test_research_bf16() {
    CHECK_EQ(cere::research::bf16_bits(1.0f), 0x3f80);
    CHECK_EQ(cere::research::bf16_bits(250.734375f), 0x437b);  // 250 + 47 / 64 rounds to 251
    CHECK_EQ(cere::research::bf16_bits(-2.0f), 0xc000);
    CHECK_EQ(cere::research::bf16_bits(1.0f + 1.0f / 256.0f), 0x3f80);  // a tie rounds to even
}

void test_research_reserve() {
    // reserve: before seal, once, staging >= 1 MiB, ring > 0, plans in [1, 64]; staging + plans x 4 MiB count against
    // the budget (CAPACITY past it); the model's release returns every byte.
    cere_executor *executor = make_executor();
    Desc d = rows_desc();
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    const uint64_t reserved = executor->reserved_device_bytes;
    const uint64_t device = toy(executor).allocated_device_bytes();
    const uint64_t host = toy(executor).allocated_host_bytes();
    CHECK_EQ(cere_model_research_reserve(nullptr, model, kResearchStaging, 4096, 2), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_research_reserve(executor, nullptr, kResearchStaging, 4096, 2), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_research_reserve(executor, model, kResearchStaging - 1, 4096, 2), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_research_reserve(executor, model, kResearchStaging, 0, 2), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_research_reserve(executor, model, kResearchStaging, 4096, 0), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_research_reserve(executor, model, kResearchStaging, 4096, CERE_RESEARCH_MAX_PLANS + 1),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_research_reserve(executor, model, UINT64_MAX, 4096, 2), CERE_ERR_CAPACITY);
    CHECK_EQ(executor->reserved_device_bytes, reserved);
    const uint64_t bytes = kResearchStaging + 2 * CERE_RESEARCH_MAX_PLAN_BYTES;
    CHECK_EQ(cere_model_research_reserve(executor, model, kResearchStaging, 4096, 2), CERE_OK);
    CHECK_EQ(executor->reserved_device_bytes, reserved + bytes);
    CHECK_EQ(toy(executor).allocated_device_bytes(), device + bytes);
    CHECK_EQ(toy(executor).allocated_host_bytes(), host + 4096);
    CHECK_EQ(cere_model_research_reserve(executor, model, kResearchStaging, 4096, 2), CERE_ERR_ARGUMENT);  // once
    RPlan plan;
    plan.capture("native.layers.L.attn_mix.out", 1, {}, true);
    CHECK_EQ(cere_model_research_plan_load(executor, model, 0, plan.get()), CERE_ERR_ARGUMENT);  // before seal
    for (uint64_t i = 0; i < d.tensors.size(); ++i) {
        auto b = pattern(d.tensors[i].bytes, uint8_t(i));
        CHECK_EQ(cere_model_put(executor, model, i, 0, b.data(), b.size()), CERE_OK);
    }
    CHECK_EQ(cere_model_seal(executor, model), CERE_OK);
    CHECK_EQ(cere_model_research_reserve(executor, model, kResearchStaging, 4096, 2), CERE_ERR_ARGUMENT);  // sealed
    CHECK_EQ(cere_model_research_plan_load(executor, model, 0, plan.get()), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(executor->reserved_device_bytes, 0);
    CHECK_EQ(toy(executor).allocated_device_bytes(), 0);
    CHECK_EQ(toy(executor).allocated_host_bytes(), 0);

    // A model without a reserve: every other call is UNSUPPORTED.
    model = sealed_model(executor, d);
    cere_session *s = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &s), CERE_OK);
    uint64_t capture = 0;
    cere_research_capture c{};
    CHECK_EQ(cere_model_research_plan_load(executor, model, 0, plan.get()), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_model_research_plan_unload(executor, model, 0), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_session_set_research(executor, s, 0, 4, 4, &capture), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_model_research_capture_read(executor, model, 1, &c), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_model_research_capture_free(executor, model, 1), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);

    // Past the budget (the model's tensors fit): CAPACITY, nothing reserved.
    executor = make_executor();
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    const uint64_t tensors = executor->reserved_device_bytes;
    executor->memory_budget_bytes = tensors + kResearchStaging + CERE_RESEARCH_MAX_PLAN_BYTES - 1;
    CHECK_EQ(cere_model_research_reserve(executor, model, kResearchStaging, 4096, 1), CERE_ERR_CAPACITY);
    CHECK_EQ(executor->reserved_device_bytes, tensors);
    executor->memory_budget_bytes += 1;
    CHECK_EQ(cere_model_research_reserve(executor, model, kResearchStaging, 4096, 1), CERE_OK);
    CHECK_EQ(cere_model_research_reserve(executor, model, kResearchStaging - 1, 4096, 1), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_research_plan_refusals() {
    // plan_load refuses (ARGUMENT, the slot stays empty) every malformed plan of the header; plan_unload: ARGUMENT
    // for an empty or out-of-range slot.
    cere_executor *executor = make_executor();
    Desc d = rows_desc();
    cere_model *model = research_model(executor, d, 4096, 2);
    const auto load = [&](const std::function<void(RPlan &)> &make, uint32_t slot = 0) {
        RPlan p;
        make(p);
        if (p.plan.ops == nullptr) p.get();  // a plan its maker already pointed keeps its edits
        return cere_model_research_plan_load(executor, model, slot, &p.plan);
    };
    const auto one = [&](const std::function<void(cere_research_op &)> &edit) {
        return load([&](RPlan &p) {
            p.capture("native.layers.L.attn_mix.out", 1, {}, true);
            p.get();
            edit(p.ops[0]);
        });
    };
    CHECK_EQ(cere_model_research_plan_load(executor, model, 0, nullptr), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_research_plan_load(nullptr, model, 0, nullptr), CERE_ERR_ARGUMENT);
    CHECK_EQ(load([](RPlan &) {}), CERE_ERR_ARGUMENT);  // op_count 0
    {
        RPlan p;
        p.capture("native.layers.L.attn_mix.out", 1, {}, true);
        p.get();
        p.plan.op_count = CERE_RESEARCH_MAX_OPS + 1;  // checked before any op is read
        CHECK_EQ(cere_model_research_plan_load(executor, model, 0, &p.plan), CERE_ERR_ARGUMENT);
        p.plan.op_count = 1;
        p.plan.reserved = 1;
        CHECK_EQ(cere_model_research_plan_load(executor, model, 0, &p.plan), CERE_ERR_ARGUMENT);
        p.plan.reserved = 0;
        p.plan.ops = nullptr;
        CHECK_EQ(cere_model_research_plan_load(executor, model, 0, &p.plan), CERE_ERR_ARGUMENT);
    }
    static const cere_research_range unsorted[] = {{4, 6}, {1, 2}};
    static const cere_research_range overlapping[] = {{1, 4}, {3, 5}};
    static const cere_research_range empty[] = {{3, 3}};
    static const cere_research_range ok[] = {{0, 4}};
    CHECK_EQ(one([](cere_research_op &op) { op.kind = CERE_RESEARCH_EXPERT_FORCE + 1; }), CERE_ERR_ARGUMENT);
    CHECK_EQ(one([](cere_research_op &op) { op.point = "native.layers.L.nope"; }), CERE_ERR_ARGUMENT);
    CHECK_EQ(one([](cere_research_op &op) { op.point = "native.layers.3.attn_mix.out"; }), CERE_ERR_ARGUMENT);
    CHECK_EQ(one([](cere_research_op &op) { op.point = nullptr; }), CERE_ERR_ARGUMENT);
    CHECK_EQ(one([](cere_research_op &op) { op.layers = 0; }), CERE_ERR_ARGUMENT);
    CHECK_EQ(one([](cere_research_op &op) { op.layers = kFinal; }), CERE_ERR_ARGUMENT);  // not a final point
    CHECK_EQ(one([](cere_research_op &op) { op.layers = uint64_t(1) << 49; }), CERE_ERR_ARGUMENT);
    CHECK_EQ(one([](cere_research_op &op) { op.point = "native.layers.L.gemm.attn_qkv.out"; }),
             CERE_ERR_ARGUMENT);  // layer 0 is Gated DeltaNet
    CHECK_EQ(one([](cere_research_op &op) {
                 op.point = "native.layers.L.gemm.gdn_z.out";
                 op.layers = kLayer3;
             }),
             CERE_ERR_ARGUMENT);  // layer 3 is full attention
    CHECK_EQ(one([](cere_research_op &op) {
                 op.point = "native.final_mix.out";
                 op.layers = 1;
             }),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(one([](cere_research_op &op) {
                 op.prompt = unsorted;
                 op.prompt_count = 2;
             }),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(one([](cere_research_op &op) {
                 op.prompt = overlapping;
                 op.prompt_count = 2;
             }),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(one([](cere_research_op &op) {
                 op.prompt = empty;
                 op.prompt_count = 1;
             }),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(one([](cere_research_op &op) { op.prompt_count = 1; }), CERE_ERR_ARGUMENT);  // null ranges
    CHECK_EQ(one([](cere_research_op &op) {
                 op.prompt = ok;
                 op.prompt_count = CERE_RESEARCH_MAX_RANGES + 1;
             }),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(one([](cere_research_op &op) { op.generated = 2; }), CERE_ERR_ARGUMENT);
    CHECK_EQ(one([](cere_research_op &op) {
                 op.generated = 0;
                 op.generated_max = 3;
             }),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(one([](cere_research_op &op) { op.generated = 0; }), CERE_ERR_ARGUMENT);  // no positions
    CHECK_EQ(one([](cere_research_op &op) { op.top_k = 3; }), CERE_ERR_ARGUMENT);      // not native.logits.topk
    CHECK_EQ(one([](cere_research_op &op) { op.alpha = 1.0f; }), CERE_ERR_ARGUMENT);   // a capture with an edit field
    CHECK_EQ(one([](cere_research_op &op) { op.col_end = 4; }), CERE_ERR_ARGUMENT);
    CHECK_EQ(one([](cere_research_op &op) { op.expert_count = 1; }), CERE_ERR_ARGUMENT);
    // Logits points: generated rows only; native.logits takes generated_max in [1, 8]; top_k in [1, 64].
    const auto logits = [&](const char *point, uint32_t top_k, uint64_t gmax, bool prompt) {
        return one([=](cere_research_op &op) {
            op.point = point;
            op.layers = kFinal;
            op.top_k = top_k;
            op.generated_max = gmax;
            if (prompt) {
                op.prompt = ok;
                op.prompt_count = 1;
            }
        });
    };
    CHECK_EQ(logits("native.logits.topk", 0, 0, false), CERE_ERR_ARGUMENT);
    CHECK_EQ(logits("native.logits.topk", CERE_RESEARCH_MAX_TOPK + 1, 0, false), CERE_ERR_ARGUMENT);
    CHECK_EQ(logits("native.logits.topk", 5, 0, true), CERE_ERR_ARGUMENT);
    CHECK_EQ(logits("native.logits", 0, 0, false), CERE_ERR_ARGUMENT);
    CHECK_EQ(logits("native.logits", 0, CERE_RESEARCH_MAX_LOGIT_POSITIONS + 1, false), CERE_ERR_ARGUMENT);
    CHECK_EQ(logits("native.logits", 1, 4, false), CERE_ERR_ARGUMENT);
    CHECK_EQ(logits("native.logits", 0, 4, true), CERE_ERR_ARGUMENT);
    // Interventions: the point must take the kind, and WL-B's check_edit holds.
    CHECK_EQ(load([](RPlan &p) { p.edit(CERE_RESEARCH_ADD, "native.layers.L.attn_mix.in", 1, 1, 10240, {}, true); }),
             CERE_ERR_ARGUMENT);  // read only
    CHECK_EQ(load([](RPlan &p) {
                 p.edit(CERE_RESEARCH_ADD, "native.layers.L.gemm.moe_shared_down.in", 1, 1, 640, {}, true);
             }),
             CERE_ERR_ARGUMENT);  // read only
    CHECK_EQ(load([](RPlan &p) { p.edit(CERE_RESEARCH_ADD, "native.layers.L.attn_mix.out", 1, 1, 2559, {}, true); }),
             CERE_ERR_ARGUMENT);  // vector_cols != the point's
    CHECK_EQ(load([](RPlan &p) { p.edit(CERE_RESEARCH_ADD, "native.layers.L.attn_mix.out", 1, 2, 2560, {}, true); }),
             CERE_ERR_ARGUMENT);  // add takes one row
    CHECK_EQ(load([](RPlan &p) {
                 p.edit(CERE_RESEARCH_ADD, "native.layers.L.attn_mix.out", 1, 1, 2560, {}, true);
                 p.vectors.back().clear();  // vector_rows without a vector
             }),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(load([](RPlan &p) {
                 p.capture("native.layers.L.attn_mix.out", 1, {}, true);
                 p.ops.back().kind = CERE_RESEARCH_EXPERT_BIAS;
             }),
             CERE_ERR_ARGUMENT);  // not the router point
    static const int32_t experts[] = {3, 3};
    static const float weights[] = {0.5f, 0.5f};
    CHECK_EQ(load([](RPlan &p) {
                 p.capture("native.layers.L.moe_route", 1, {}, true);
                 p.ops.back().kind = CERE_RESEARCH_EXPERT_FORCE;
                 p.ops.back().experts = experts;
                 p.ops.back().weights = weights;
                 p.ops.back().expert_count = 2;
             }),
             CERE_ERR_ARGUMENT);  // a repeated expert
    // Operands above CERE_RESEARCH_MAX_PLAN_BYTES (a replace of 820 rows x 2,560 BF16 = 4,198,400 B).
    CHECK_EQ(load([](RPlan &p) {
                 p.edit(CERE_RESEARCH_REPLACE, "native.layers.L.attn_mix.out", 1, 820, 2560, {}, true);
             }),
             CERE_ERR_ARGUMENT);
    for (uint32_t slot = 0; slot < 2; ++slot) CHECK(model->research_plan[slot] == nullptr);

    // Valid: every kind at a point that takes it; slots; unload.
    static const int32_t one_expert[] = {7};
    static const float half[] = {0.5f};
    const auto all_kinds = [&](RPlan &p) {
        p.edit(CERE_RESEARCH_ADD, "native.layers.L.mlp_mix.out", 1, 1, 2560, {{0, 2}}, false);
        p.capture("native.layers.L.gemm.gdn_qkv.out", 1, {}, true);
        p.ops.back().kind = CERE_RESEARCH_ABLATE_ZERO;
        p.edit(CERE_RESEARCH_ABLATE_MEAN, "model.layers.L.linear_attn.out_proj", 1, 1, 2560, {}, true);
        p.edit(CERE_RESEARCH_REPLACE, "model.layers.L.self_attn.o_proj", kLayer3, 3, 2560, {}, true);
        p.capture("native.layers.L.gemm.moe_router.out", 1, {}, true);
        p.ops.back().kind = CERE_RESEARCH_EXPERT_MASK;
        p.ops.back().experts = one_expert;
        p.ops.back().expert_count = 1;
        p.capture("native.layers.L.gemm.moe_router.out", 2, {}, true);
        p.ops.back().kind = CERE_RESEARCH_EXPERT_BIAS;
        p.ops.back().experts = one_expert;
        p.ops.back().weights = half;
        p.ops.back().expert_count = 1;
        p.capture("native.layers.L.moe_route", 1, {}, true);
        p.ops.back().kind = CERE_RESEARCH_EXPERT_FORCE;
        p.ops.back().experts = one_expert;
        p.ops.back().weights = half;
        p.ops.back().expert_count = 1;
        p.capture("native.logits", kFinal, {}, true, CERE_RESEARCH_MAX_LOGIT_POSITIONS);
        p.capture("native.logits.topk", kFinal, {}, true, 0, CERE_RESEARCH_MAX_TOPK);
        p.capture("model.layers.L.mlp.experts", 3, {{0, 1}}, false);
        p.capture("model.layers.L.mlp.shared_expert.down_proj", 3, {{0, 1}}, false);
        p.capture("native.layers.L.attn_mix.in", 1, {{0, 1}}, false);
        p.capture("native.layers.L.mlp_mix.in", 1, {{0, 1}}, false);
        p.capture("native.layers.L.gemm.moe_shared_down.in", 1, {{0, 1}}, false);
        p.capture("native.final_mix.out", kFinal, {}, true);
        p.capture("native.layers.L.gemm.gdn_z.out", 1, {}, true);
    };
    CHECK_EQ(load(all_kinds, 2), CERE_ERR_ARGUMENT);  // slot outside [0, max_plans)
    CHECK_EQ(load(all_kinds, 1), CERE_OK);
    CHECK_EQ(load(all_kinds, 1), CERE_ERR_ARGUMENT);  // occupied
    CHECK(model->research_plan[1]->edits && model->research_plan[1]->captures);
    CHECK(model->research_plan[1]->captures_prompt);
    CHECK_EQ(cere_model_research_plan_unload(executor, model, 0), CERE_ERR_ARGUMENT);  // empty
    CHECK_EQ(cere_model_research_plan_unload(executor, model, 2), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_research_plan_unload(nullptr, model, 1), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_research_plan_unload(executor, model, 1), CERE_OK);
    CHECK_EQ(cere_model_research_plan_unload(executor, model, 1), CERE_ERR_ARGUMENT);
    CHECK_EQ(load(all_kinds, 1), CERE_OK);  // the freed slot loads again
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_research_bind() {
    // set_research: before the first submit, once, to a loaded slot, prompt_tokens >= 1, a replace vector of 1 row or
    // at least the op's positions; a capture-free plan writes capture 0; the ring holds the bound (CAPACITY past it,
    // nothing bound) until capture_free; unload is BUSY while a live session uses the slot.
    cere_executor *executor = make_executor();
    Desc d = rows_desc(4);
    // A ring of 4,096 B: moe_route rows are 80 B, so 51 rows fit.
    cere_model *model = research_model(executor, d, 4096, 3);
    RPlan route;
    route.capture("native.layers.L.moe_route", 1, {}, true);
    RPlan edit;
    edit.edit(CERE_RESEARCH_REPLACE, "native.layers.L.attn_mix.out", 1, 3, 2560, {{0, 2}}, true);
    CHECK_EQ(cere_model_research_plan_load(executor, model, 0, route.get()), CERE_OK);
    CHECK_EQ(cere_model_research_plan_load(executor, model, 1, edit.get()), CERE_OK);
    cere_session *s = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &s), CERE_OK);
    uint64_t capture = 99;
    CHECK_EQ(cere_session_set_research(nullptr, s, 0, 10, 30, &capture), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_session_set_research(executor, nullptr, 0, 10, 30, &capture), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_session_set_research(executor, s, 0, 10, 30, nullptr), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_session_set_research(executor, s, 2, 10, 30, &capture), CERE_ERR_ARGUMENT);  // not loaded
    CHECK_EQ(cere_session_set_research(executor, s, 3, 10, 30, &capture), CERE_ERR_ARGUMENT);  // outside
    CHECK_EQ(cere_session_set_research(executor, s, 0, 0, 30, &capture), CERE_ERR_ARGUMENT);   // prompt_tokens 0
    // The replace's 3 rows: prompt 2 + generated 1 = 3 positions fit, 2 + 2 do not.
    CHECK_EQ(cere_session_set_research(executor, s, 1, 4, 2, &capture), CERE_ERR_ARGUMENT);
    CHECK_EQ(capture, 99);
    CHECK(s->research == nullptr);
    CHECK_EQ(cere_session_set_research(executor, s, 1, 4, 1, &capture), CERE_OK);
    CHECK_EQ(capture, 0);  // nothing captured
    CHECK_EQ(model->research_ring_alloc.used(), 0);
    CHECK_EQ(cere_session_set_research(executor, s, 1, 4, 1, &capture), CERE_ERR_ARGUMENT);  // once
    CHECK_EQ(cere_model_research_plan_unload(executor, model, 1), CERE_ERR_BUSY);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_research_plan_unload(executor, model, 1), CERE_OK);

    // Ring accounting: 30 generated rows (prompt 10, max_tokens 30) = 2,400 B; a second such bind does not fit.
    cere_session *a = nullptr, *b = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &a), CERE_OK);
    CHECK_EQ(cere_session_create(executor, model, 64, &b), CERE_OK);
    uint64_t ca = 0, cb = 0;
    CHECK_EQ(cere_session_set_research(executor, a, 0, 10, 30, &ca), CERE_OK);
    CHECK(ca != 0);
    CHECK_EQ(a->research->bytes, 30 * 80);
    CHECK_EQ(cere::research::bound(*model->research_plan[0], 10, 30), 30 * 80);
    CHECK_EQ(model->research_ring_alloc.used(), 30 * 80);
    CHECK_EQ(cere_session_set_research(executor, b, 0, 10, 30, &cb), CERE_ERR_CAPACITY);
    CHECK(b->research == nullptr);
    CHECK_EQ(model->research_ring_alloc.used(), 30 * 80);
    CHECK_EQ(cere_session_set_research(executor, b, 0, 10, 20, &cb), CERE_OK);  // 1,600 B fit
    CHECK(cb != 0 && cb != ca);
    CHECK_EQ(cere_session_release(executor, b), CERE_OK);
    // After a submit: ARGUMENT.
    cere_session *c = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &c), CERE_OK);
    Job job = submit_job(executor, model, c, CERE_OP_PREFILL, {1, 2}, 1);
    CHECK_EQ(drain(executor, job.ticket).committed_tokens, 1);
    uint64_t cc = 0;
    CHECK_EQ(cere_session_set_research(executor, c, 0, 2, 1, &cc), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_session_release(executor, c), CERE_OK);
    // A capture never run: released at once, read with no rows; free returns its ring bytes.
    cere_research_capture cap{};
    CHECK_EQ(cere_model_research_capture_read(executor, model, cb, &cap), CERE_OK);
    CHECK_EQ(cap.tensor_count, 2);
    CHECK_EQ(cap.bytes, 0);
    CHECK_EQ(cap.tensors[0].rows, 0);
    CHECK(cap.tensors[0].positions == nullptr);
    CHECK(std::string(cap.tensors[0].name) == "native.layers.L.moe_route.ids/L0");
    CHECK(std::string(cap.tensors[1].name) == "native.layers.L.moe_route.weights/L0");
    CHECK_EQ(cap.tensors[0].dtype, CERE_RESEARCH_DTYPE_I32);
    CHECK_EQ(cap.tensors[1].dtype, CERE_RESEARCH_DTYPE_F32);
    CHECK_EQ(cap.tensors[1].cols, 10);
    CHECK_EQ(cap.prefix_lookup_skipped, 0);
    CHECK_EQ(cere_model_research_capture_read(executor, model, cb, nullptr), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_research_capture_read(executor, model, 12345, &cap), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_research_capture_read(executor, nullptr, cb, &cap), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_research_capture_free(executor, model, 12345), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_research_capture_free(executor, model, cb), CERE_OK);
    CHECK_EQ(model->research_ring_alloc.used(), 30 * 80);
    // a's capture outlives its unloaded plan and the model's release frees what is left.
    CHECK_EQ(cere_model_research_plan_unload(executor, model, 0), CERE_ERR_BUSY);
    CHECK_EQ(cere_session_release(executor, a), CERE_OK);
    CHECK_EQ(cere_model_research_plan_unload(executor, model, 0), CERE_OK);
    CHECK_EQ(cere_model_research_capture_read(executor, model, ca, &cap), CERE_OK);
    CHECK_EQ(cap.tensor_count, 2);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(executor->reserved_device_bytes, 0);
    CHECK_EQ(toy(executor).allocated_host_bytes(), 0);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_research_positions() {
    // Capture positions, names, order and rows under prefill + decode; the bound reserved at bind; rows kept below the
    // session's consumed tokens at release (4 of max_tokens 10 generated: positions < 9).
    const CapRun run = run_capture("toy", false, false);
    CHECK_EQ(run.bound, kPositionsBound);
    CHECK(run.out == run.base_out);
    CHECK_EQ(run.skipped, 1);  // op 0 and op 3 capture prompt positions
    const std::vector<std::pair<std::string, std::vector<uint64_t>>> want = {
        {"native.layers.L.attn_mix.out/L3", {1, 2, 4, 5}},
        {"native.layers.L.attn_mix.out/L47", {1, 2, 4, 5}},
        {"native.layers.L.moe_route.ids/L0", {5, 6}},
        {"native.layers.L.moe_route.weights/L0", {5, 6}},
        {"native.logits.topk.ids/L48", {5, 6, 7, 8}},
        {"native.logits.topk.logprobs/L48", {5, 6, 7, 8}},
        {"native.layers.L.gemm.attn_qkv.out/L3", {0, 1, 5, 6, 7, 8}},
    };
    CHECK_EQ(run.tensors.size(), want.size());
    const std::vector<uint32_t> ops = {0, 0, 1, 1, 2, 2, 3};
    const std::vector<uint32_t> layers = {3, 47, 0, 0, 48, 48, 3};
    const std::vector<uint32_t> cols = {2560, 2560, 10, 10, 3, 3, 13312};
    for (size_t i = 0; i < std::min(run.tensors.size(), want.size()); ++i) {
        CHECK(run.tensors[i].name == want[i].first);
        CHECK(run.tensors[i].positions == want[i].second);
        CHECK_EQ(run.tensors[i].op, ops[i]);
        CHECK_EQ(run.tensors[i].layer, layers[i]);
        CHECK_EQ(run.tensors[i].cols, cols[i]);
    }
    // A PREFILL of 1 and a DECODE of 3, and a chunked prefill (2 tokens a step), capture the same rows.
    const CapRun split = run_capture("toy", false, true);
    CHECK(split.tensors == run.tensors);
    CHECK(split.out == run.out);
    const CapRun chunked = run_capture("toy,prefill_chunk=2", false, false);
    CHECK(chunked.tensors == run.tensors);
    CHECK(chunked.out == run.out);
}

void test_research_verify() {
    // Speculation (M2.9): a drafting model whose verify rejects the 2nd draft of each chain keeps only committed rows;
    // the capture equals the plain model's, as do the tokens.
    for (const DraftDevice &dev : kDraftDevices) {
        const CapRun plain = run_capture(dev.id, false, false);
        const CapRun spec = run_capture(dev.id, true, false);
        CHECK(spec.verify_steps > 0);
        CHECK_EQ(plain.verify_steps, 0);
        CHECK(spec.out == plain.out);
        CHECK(spec.tensors == plain.tensors);
        const CapRun spec_split = run_capture(dev.id, true, true);
        CHECK(spec_split.tensors == plain.tensors);
    }
}

void test_research_lookup_skip() {
    // WL decision (e): a session whose plan captures prompt positions does not look up the prefix cache (it prefills
    // from 0, cached_tokens 0, prefix_lookup_skipped 1) but still inserts; a generated-only plan looks up as usual.
    cere_executor *executor = make_executor();
    PrefixDesc d(4, 2, 1024, 4096);
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &d.desc, &model), CERE_OK);
    CHECK_EQ(cere_model_research_reserve(executor, model, kResearchStaging, uint64_t(1) << 22, 2), CERE_OK);
    for (uint64_t i = 0; i < d.tensors.size(); ++i) {
        auto bytes = pattern(d.tensors[i].bytes, uint8_t(i));
        CHECK_EQ(cere_model_put(executor, model, i, 0, bytes.data(), bytes.size()), CERE_OK);
    }
    CHECK_EQ(cere_model_seal(executor, model), CERE_OK);
    RPlan prompt_plan;
    prompt_plan.capture("native.layers.L.attn_mix.out", 1, {{0, 300}});
    RPlan gen_plan;
    gen_plan.capture("native.layers.L.attn_mix.out", 1, {}, true);
    CHECK_EQ(cere_model_research_plan_load(executor, model, 0, prompt_plan.get()), CERE_OK);
    CHECK_EQ(cere_model_research_plan_load(executor, model, 1, gen_plan.get()), CERE_OK);
    const PToks p = ptoks(21, 300);
    cere::host::PrefixCache &tree = prefix_tree(model);
    cere_session *a = ns_session(executor, model, 1024, 1);
    const PRun cold = prun(executor, a, CERE_OP_PREFILL, p, 2, 0);
    CHECK_EQ(cere_session_release(executor, a), CERE_OK);
    CHECK_EQ(tree.stats().hits, 0);

    cere_session *b = ns_session(executor, model, 1024, 1);
    uint64_t cb = 0;
    CHECK_EQ(cere_session_set_research(executor, b, 0, p.size(), 2, &cb), CERE_OK);
    const PRun skipped = prun(executor, b, CERE_OP_PREFILL, p, 2, 0);
    check_cold_state(executor, b, p, skipped);
    CHECK(skipped.out == cold.out);
    CHECK_EQ(tree.stats().hits, 0);
    CHECK_EQ(cere_session_release(executor, b), CERE_OK);
    cere_research_capture c{};
    CHECK_EQ(cere_model_research_capture_read(executor, model, cb, &c), CERE_OK);
    CHECK_EQ(c.prefix_lookup_skipped, 1);
    CHECK_EQ(c.tensor_count, 1);
    CHECK_EQ(c.tensors[0].rows, 300);
    CHECK(toy_rows(c.tensors[0]));
    CHECK_EQ(cere_model_research_capture_free(executor, model, cb), CERE_OK);

    cere_session *g = ns_session(executor, model, 1024, 1);
    uint64_t cg = 0;
    CHECK_EQ(cere_session_set_research(executor, g, 1, p.size(), 2, &cg), CERE_OK);
    const PRun hit = prun(executor, g, CERE_OP_PREFILL, p, 2, 256);
    CHECK(hit.out == cold.out);
    CHECK_EQ(tree.stats().hits, 1);
    CHECK_EQ(cere_session_release(executor, g), CERE_OK);
    CHECK_EQ(cere_model_research_capture_read(executor, model, cg, &c), CERE_OK);
    CHECK_EQ(c.prefix_lookup_skipped, 0);
    CHECK_EQ(c.tensors[0].rows, 2);  // generated rows 299 and 300 (row 299 is never a hit)
    CHECK_EQ(c.tensors[0].positions[0], 299);
    CHECK_EQ(c.tensors[0].positions[1], 300);
    CHECK(toy_rows(c.tensors[0]));
    CHECK_EQ(cere_model_research_capture_free(executor, model, cg), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_research_snapshot() {
    // A session with a plan takes no snapshot and restores none (UNSUPPORTED); one without keeps both.
    cere_executor *executor = make_executor();
    Desc d = rows_desc();
    d.desc.snapshot_host_bytes = uint64_t(1) << 20;
    cere_model *model = research_model(executor, d, 4096, 1);
    RPlan plan;
    plan.capture("native.layers.L.moe_route", 1, {}, true);
    CHECK_EQ(cere_model_research_plan_load(executor, model, 0, plan.get()), CERE_OK);
    cere_session *r = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &r), CERE_OK);
    uint64_t capture = 0;
    CHECK_EQ(cere_session_set_research(executor, r, 0, 2, 2, &capture), CERE_OK);
    Job job = submit_job(executor, model, r, CERE_OP_PREFILL, {1, 2}, 2);
    CHECK_EQ(drain(executor, job.ticket).committed_tokens, 2);
    cere_snapshot *snap = nullptr;
    CHECK_EQ(cere_session_snapshot(executor, r, r->revision, &snap), CERE_ERR_UNSUPPORTED);
    CHECK(snap == nullptr);
    cere_session *base = prefilled_session(executor, model, {1, 2});
    CHECK_EQ(cere_session_snapshot(executor, base, base->revision, &snap), CERE_OK);
    cere_session *t = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &t), CERE_OK);
    uint64_t ct = 0;
    CHECK_EQ(cere_session_set_research(executor, t, 0, 2, 2, &ct), CERE_OK);
    CHECK_EQ(cere_session_restore(executor, t, snap), CERE_ERR_UNSUPPORTED);
    cere_session *u = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 64, &u), CERE_OK);
    CHECK_EQ(cere_session_restore(executor, u, snap), CERE_OK);
    CHECK_EQ(cere_session_set_research(executor, u, 0, 2, 2, &ct), CERE_ERR_ARGUMENT);  // a restore is a submit
    CHECK_EQ(cere_snapshot_release(executor, snap), CERE_OK);
    for (cere_session *s : {r, base, t, u}) CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

}  // namespace

int main() {
    test_status_codes_documented();
    test_abi_checks();
    test_config_off_by_one();
    test_a4_context_limit();
    test_a4_plan_at_context();
    test_decode_graph_fits();
    test_spec_graph_chain_fits();
    test_spec_draft_graph_fits();
    test_draft_decode_steps();
    test_a4_draft_region();
    test_support_limits();
    test_put_rules();
    test_concurrent_disjoint_puts();
    test_concurrent_overlapping_puts();
    test_capacity();
    test_generation_and_tokens();
    test_stop_and_context();
    test_stale_epoch();
    test_cancel_busy_and_release();
    test_destroy_with_ticket_outstanding();
    test_internal_error();
    test_sticky_fault();
    test_release_device_option();
    test_create_busy_option();
    test_create_capacity_delay_option();
    test_hold_after();
    test_hold_gate();
    test_rows_round_commits_each();
    test_rows_prefill_chunk_alternates();
    test_rows_join_leave();
    test_rows_cancel_one();
    test_rows_one_bad_row();
    test_rows_fail_forward_breaks_participants();
    test_rows_pending_excludes_row();
    test_rows_interleave_long_prefill();
    test_rows_cancel_during_prefill();
    test_rows_decode_rounds_per_chunk();
    test_rows_decode_rounds_setting();
    test_lora_api();
    test_lora_rows();
    test_research_bf16();
    test_research_reserve();
    test_research_plan_refusals();
    test_research_bind();
    test_research_positions();
    test_research_verify();
    test_research_lookup_skip();
    test_research_snapshot();
    test_prefill_slice_rounds();
    test_prefill_slice_alone_and_cancel();
    test_prefill_slice_setting();
    test_prefill_age_setting();
    test_prefill_age_cap();
    test_poll_reports_round_first();
    test_decode_share_rounds();
    test_decode_share_setting();
    test_rows_duplicate_session_argument();
    test_unsealed_and_foreign_handles();
    test_sha256_vectors();
    test_snapshot_writer();
    test_snapshot_rejections();
    test_snapshot_api();
    test_snapshot_sampler_counts();
    test_snapshot_api_rejections();
    test_snapshot_refuses_open_prompt();
    test_snapshot_image_session();
    test_kv_pool();
    test_kv_pool_rows();
    test_l0_frame();
    test_images();
    test_image_memory();
    test_sampling_validation();
    test_toy_sampling();
    test_toy_record();
    test_rows_sampled_tickets_share_the_round();
    test_rows_sampler_invariance();
    test_rows_sampling_args();
    test_ignore_stop_tokens();
    test_vocab_mask_submit();
    test_vocab_mask_wait_and_constrain();
    test_vocab_mask_cancel();
    test_vocab_mask_sampled();
    test_rows_vocab_mask();
    test_prefix_namespace_api();
    test_prefix_second_request();
    test_prefix_shortest_first_wait();
    test_prefix_internal_ticket();
    test_prefix_cancel_keeps_captures();
    test_prefix_busy_and_flush();
    test_prefix_export_import();
    test_prefix_restore();
    test_prefix_capture_position();
    test_prefix_host_tier();
    test_prefix_host_tier_internal();
    test_prefix_host_tier_tail_last();
    test_prefix_arena_priority();
    test_restore_refuses_prefix_image();
    test_drafting_export_imports();
    test_unfittable_snapshot_destroys_nothing();
    test_export_allocation_failure();
    test_snapshot_allocation_failure();
    test_restore_allocation_failure();
    test_terminal_poll_allocation_failure();
    test_prefix_decode_capture();
    test_spec_abi();
    test_spec_equals_plain();
    test_verify_forced();
    test_spec_lookup_equals_plain();
    test_spec_adaptive_equals_plain();
    test_spec_round_costs_seal();
    test_spec_mtp_tickets_read_mtp_windows_only();
    test_spec_tail_policy();
    test_spec_long_policy();
    test_spec_chain_policy();
    test_spec_fail_forward();
    test_spec_hold_after();
    test_spec_rows();
    test_spec_sampled_distribution();
    test_spec_sampled_picks_invariant();
    test_grammar_spec_greedy_equals_masked_plain();
    test_grammar_spec_rows();
    test_grammar_spec_api();
    test_grammar_spec_sampled_distribution();
    test_grammar_spec_device_refuses_implicit_drafts();
    test_spec_snapshot();
    test_spec_adaptive_restore_equals_plain();
    test_levers_desc_round_trip();
    test_set_levers_busy_and_validation();
    test_set_levers_k2_weights_restart_only();
    test_set_levers_calibrate_windows_kept();
    test_draft_tokens_lever_greedy_equals_plain();
    test_spec_prefix_minimum();
    test_spec_prefix_unsplit();
    test_grammar_spec_prefix_layout();
    test_spec_prefix_draft_hits();
    test_spec_prefix_host_tier_drafts();
    test_spec_prefix_end_states();
    test_spec_prefix_chunk0_end_states();
    test_spec_prefix_end_chain();
    test_spec_prefix_end_chain_superseded();
    test_spec_prefix_chunk0_end_chain();
    test_prefill_slice_end_states();
    test_spec_prefix_end_state_opt_in();
    test_spec_prefix_end_state_eviction();
    test_spec_prefix_hit_shortfall();
    test_spec_prefix_end_state_invariance();
    test_spec_prefix_end_pair_horizon();
    test_spec_prefix_short_end_pair();
    test_vision_support_values();
    test_image_long_prompt();
    test_image_rows_and_drafts();
    test_prefix_image_keys();
    test_prefix_image_end_state();
    test_spec_prefix_alt_edges();
    test_spec_prefix_alt_edges_rewrite();
    test_prefix_alt_edges_opt_in();
    test_spec_prefix_alt_edges_census();
    return cere::test::finish("test_state");
}
