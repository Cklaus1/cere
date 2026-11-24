// H2: the built toy libraries ($CERE_TOY_EXECUTOR, $CERE_TOY_EXECUTOR_NOFORWARD) export every ABI v4 function named in
// tests/fixtures/executor_abi_v5_layout.json and run a create/put/seal/session/submit/poll round trip through dlsym.
#include "check.h"

#include "cere_executor.h"
#include "toy/toy_device.h"

#include <dlfcn.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Api {
    uint32_t (*abi_version)(void);
    cere_status (*executor_create)(cere_executor **, const cere_executor_options *);
    cere_status (*executor_destroy)(cere_executor *);
    cere_status (*capabilities)(cere_executor *, const cere_model *, cere_execution_capabilities *);
    cere_status (*model_create)(cere_executor *, const cere_model_desc *, cere_model **);
    cere_status (*model_put)(cere_executor *, cere_model *, uint64_t, uint64_t, const void *, uint64_t);
    cere_status (*model_seal)(cere_executor *, cere_model *);
    cere_status (*model_release)(cere_executor *, cere_model *);
    cere_status (*session_create)(cere_executor *, cere_model *, uint64_t, cere_session **);
    cere_status (*session_release)(cere_executor *, cere_session *);
    cere_status (*submit)(cere_executor *, const cere_batch *, cere_completion_ticket *);
    cere_status (*cancel)(cere_executor *, cere_completion_ticket);
    cere_status (*poll)(cere_executor *, cere_completion_ticket, cere_completion *);
    cere_status (*constrain)(cere_executor *, cere_completion_ticket, uint64_t, const uint32_t *, uint64_t);
};

template <typename T>
void bind(void *lib, const char *name, T *slot) {
    void *sym = dlsym(lib, name);
    if (!sym) std::fprintf(stderr, "missing symbol %s\n", name);
    CHECK(sym != nullptr);
    *reinterpret_cast<void **>(slot) = sym;
}

std::vector<std::string> fixture_functions() {
    std::ifstream in(CERE_LAYOUT_FIXTURE);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    std::vector<std::string> names;
    size_t pos = text.find("\"functions\": [");
    CHECK(pos != std::string::npos);
    if (pos == std::string::npos) return names;
    const size_t end = text.find(']', pos);
    pos = text.find('[', pos) + 1;
    while (true) {
        const size_t open = text.find('"', pos);
        if (open == std::string::npos || open > end) break;
        const size_t close = text.find('"', open + 1);
        names.push_back(text.substr(open + 1, close - open - 1));
        pos = close + 1;
    }
    return names;
}

void *open_library(const char *env) {
    const char *path = std::getenv(env);
    if (!path || path[0] != '/') std::fprintf(stderr, "%s must be an absolute library path\n", env);
    CHECK(path != nullptr && path[0] == '/');
    if (!path) return nullptr;
    void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) std::fprintf(stderr, "dlopen %s: %s\n", path, dlerror());
    CHECK(lib != nullptr);
    return lib;
}

Api bind_api(void *lib) {
    const auto names = fixture_functions();
    // The 13 of M1, the six M2.11 snapshot functions, M2.8's constrain, M2.10's four prefix cache functions, M2.16's
    // host_resident, ABI v5's cere_model_selection and cere_model_set_selection, M216S-11's two put staging functions,
    // O79's cere_executor_set_decode_rounds, m210 D15''''' stage 2a's cere_model_set_prefix_end_state, m210 D17's
    // cere_model_set_prefix_alt_edges, the prefix cache counters' cere_model_prefix_stats, MixedPD's
    // cere_executor_set_prefill_slice, GrammarSpec's cere_executor_round_drafts and cere_executor_constrain_drafts,
    // Track LoRA's cere_model_lora_reserve / _load / _unload and cere_session_set_lora, PT0a's
    // cere_model_prefix_capture_stats, S1 G4's cere_executor_set_decode_share, WeightLab's six research functions
    // (cere_model_research_reserve / _plan_load / _plan_unload / _capture_read / _capture_free and
    // cere_session_set_research), BC1's cere_model_prefill_signature and BC4's cere_executor_set_prefill_age.
    CHECK_EQ(names.size(), 50);
    for (const std::string &name : names) CHECK(dlsym(lib, name.c_str()) != nullptr);
    Api api{};
    bind(lib, "cere_executor_abi_version", &api.abi_version);
    bind(lib, "cere_executor_create", &api.executor_create);
    bind(lib, "cere_executor_destroy", &api.executor_destroy);
    bind(lib, "cere_executor_capabilities", &api.capabilities);
    bind(lib, "cere_model_create", &api.model_create);
    bind(lib, "cere_model_put", &api.model_put);
    bind(lib, "cere_model_seal", &api.model_seal);
    bind(lib, "cere_model_release", &api.model_release);
    bind(lib, "cere_session_create", &api.session_create);
    bind(lib, "cere_session_release", &api.session_release);
    bind(lib, "cere_executor_submit", &api.submit);
    bind(lib, "cere_executor_cancel", &api.cancel);
    bind(lib, "cere_executor_poll", &api.poll);
    bind(lib, "cere_executor_constrain", &api.constrain);
    // Internal C++ symbols stay hidden.
    CHECK(dlsym(lib, "_ZN4cere4toy10next_tokenEjmj") == nullptr);
    return api;
}

// A sealed one-tensor model of the toy fixture with `max_sessions` sessions of up to 16 tokens.
cere_model *sealed_model(const Api &api, cere_executor *executor, const cere_qwen4_exp_config &config,
                         uint64_t max_sessions) {
    cere_tensor_desc tensor{};
    tensor.struct_size = sizeof(tensor);
    tensor.abi_version = CERE_EXECUTOR_ABI_VERSION;
    tensor.name = "model.embed_tokens.weight";
    tensor.dtype = CERE_DTYPE_BF16;
    tensor.placement = CERE_PLACEMENT_DEVICE;
    tensor.ndim = 2;
    tensor.shape[0] = 4;
    tensor.shape[1] = 8;
    tensor.bytes = 64;
    cere_model_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = CERE_EXECUTOR_ABI_VERSION;
    desc.model_kind = CERE_MODEL_QWEN4_EXP_NVFP4;
    desc.config = &config;
    desc.tensors = &tensor;
    desc.tensor_count = 1;
    desc.max_sessions = max_sessions;
    desc.max_context_tokens = 16;
    cere_model *model = nullptr;
    CHECK_EQ(api.model_create(executor, &desc, &model), CERE_OK);
    std::vector<uint8_t> bytes(64, 0x5a);
    CHECK_EQ(api.model_put(executor, model, 0, 0, bytes.data(), bytes.size()), CERE_OK);
    CHECK_EQ(api.model_seal(executor, model), CERE_OK);
    return model;
}

cere_executor *toy_executor(const Api &api) {
    cere_executor_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = CERE_EXECUTOR_ABI_VERSION;
    options.device_id = "toy";
    options.worker_epoch = 3;
    options.memory_budget_bytes = 1 << 20;
    cere_executor *executor = nullptr;
    CHECK_EQ(api.executor_create(&executor, &options), CERE_OK);
    return executor;
}

void round_trip(void *lib, bool forward) {
    if (!lib) return;
    const Api api = bind_api(lib);
    CHECK_EQ(api.abi_version(), CERE_EXECUTOR_ABI_VERSION);

    cere_executor *executor = toy_executor(api);
    const cere_qwen4_exp_config config = cere::toy::fixture_config();
    cere_model *model = sealed_model(api, executor, config, 1);

    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(api.capabilities(executor, model, &caps), CERE_OK);
    CHECK_EQ(caps.execution_mode, forward ? CERE_EXECUTION_MODE_NATIVE : CERE_EXECUTION_MODE_UNSUPPORTED);
    CHECK_EQ(caps.operations, forward ? (CERE_OPS_PREFILL | CERE_OPS_DECODE | CERE_OPS_CANCEL) : 0);

    cere_session *session = nullptr;
    CHECK_EQ(api.session_create(executor, model, 16, &session), CERE_OK);
    const uint32_t prompt[3] = {1, 2, 3};
    uint32_t out[1] = {0};
    cere_work item{};
    item.struct_size = sizeof(item);
    item.abi_version = CERE_EXECUTOR_ABI_VERSION;
    item.operation = CERE_OP_PREFILL;
    item.expected_worker_epoch = 3;
    item.session = session;
    item.tokens = prompt;
    item.token_count = 3;
    item.max_new_tokens = 1;
    item.out_tokens = out;
    item.out_capacity = 1;
    cere_batch batch{};
    batch.struct_size = sizeof(batch);
    batch.abi_version = CERE_EXECUTOR_ABI_VERSION;
    batch.expected_worker_epoch = 3;
    batch.capability_profile_generation = caps.profile_generation;
    batch.items = &item;
    batch.item_count = 1;
    cere_completion_ticket ticket{};
    if (forward) {
        CHECK_EQ(api.submit(executor, &batch, &ticket), CERE_OK);
        cere_completion c{};
        c.struct_size = sizeof(c);
        c.abi_version = CERE_EXECUTOR_ABI_VERSION;
        CHECK_EQ(api.poll(executor, ticket, &c), CERE_OK);
        CHECK_EQ(c.terminal, 1);
        CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
        CHECK_EQ(c.committed_tokens, 1);
        CHECK_EQ(out[0], (31u * 3u + 3u) % config.vocab_size);
        CHECK_EQ(api.cancel(executor, ticket), CERE_ERR_ARGUMENT);  // released by the terminal poll
    } else {
        CHECK_EQ(api.submit(executor, &batch, &ticket), CERE_ERR_UNSUPPORTED);
        CHECK_EQ(ticket.submission_id, 0);
    }
    CHECK_EQ(api.session_release(executor, session), CERE_OK);
    CHECK_EQ(api.model_release(executor, model), CERE_OK);
    CHECK_EQ(api.executor_destroy(executor), CERE_OK);
}

// C2 (M2.6 D3/I2): the caps report the plan's rows: decode_batch_max = max_sessions (1 and 4), native_decode_batch
// when it exceeds 1 (and the library has a forward pass); a batch of 2 items stays UNSUPPORTED (rows form across
// tickets, D1).
void caps_rows(void *lib, bool forward) {
    if (!lib) return;
    const Api api = bind_api(lib);
    const cere_qwen4_exp_config config = cere::toy::fixture_config();
    for (uint64_t rows : {1, 4}) {
        cere_executor *executor = toy_executor(api);
        cere_model *model = sealed_model(api, executor, config, rows);
        cere_execution_capabilities caps{};
        caps.struct_size = sizeof(caps);
        caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
        CHECK_EQ(api.capabilities(executor, model, &caps), CERE_OK);
        CHECK_EQ(caps.decode_batch_min, 1);
        CHECK_EQ(caps.decode_batch_max, rows);
        CHECK_EQ(caps.native_decode_batch, forward && rows > 1 ? 1u : 0u);
        CHECK_EQ(caps.native_mixed_prefill_decode, 0);

        cere_session *sessions[2] = {nullptr, nullptr};
        CHECK_EQ(api.session_create(executor, model, 16, &sessions[0]), CERE_OK);
        const uint32_t prompt[1] = {1};
        uint32_t out[1] = {0};
        cere_work items[2] = {};
        for (cere_work &item : items) {
            item.struct_size = sizeof(item);
            item.abi_version = CERE_EXECUTOR_ABI_VERSION;
            item.operation = CERE_OP_PREFILL;
            item.expected_worker_epoch = 3;
            item.session = sessions[0];
            item.tokens = prompt;
            item.token_count = 1;
            item.max_new_tokens = 1;
            item.out_tokens = out;
            item.out_capacity = 1;
        }
        if (rows > 1) {
            CHECK_EQ(api.session_create(executor, model, 16, &sessions[1]), CERE_OK);
            items[1].session = sessions[1];
        }
        cere_batch batch{};
        batch.struct_size = sizeof(batch);
        batch.abi_version = CERE_EXECUTOR_ABI_VERSION;
        batch.expected_worker_epoch = 3;
        batch.capability_profile_generation = caps.profile_generation;
        batch.items = items;
        batch.item_count = 2;
        cere_completion_ticket ticket{};
        CHECK_EQ(api.submit(executor, &batch, &ticket), CERE_ERR_UNSUPPORTED);
        CHECK_EQ(ticket.submission_id, 0);
        for (cere_session *s : sessions) {
            if (s != nullptr) CHECK_EQ(api.session_release(executor, s), CERE_OK);
        }
        CHECK_EQ(api.model_release(executor, model), CERE_OK);
        CHECK_EQ(api.executor_destroy(executor), CERE_OK);
    }
}

// M2.8 C6 (m28 design note A1-A3, D9) through dlsym: capabilities native_token_mask is 1 exactly when the library
// has a forward pass; a constrained PREFILL commits an allowed token, then waits (CERE_PENDING, nothing committed)
// until cere_executor_constrain supplies the mask of its next position.
void constrained_round_trip(void *lib, bool forward) {
    if (!lib) return;
    const Api api = bind_api(lib);
    cere_executor *executor = toy_executor(api);
    const cere_qwen4_exp_config config = cere::toy::fixture_config();
    cere_model *model = sealed_model(api, executor, config, 1);
    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(api.capabilities(executor, model, &caps), CERE_OK);
    CHECK_EQ(caps.native_token_mask, forward ? 1u : 0u);

    const uint64_t words = (config.vocab_size + 31) / 32;
    std::vector<uint32_t> mask(words, 0);
    mask[7 >> 5] |= 1u << (7 & 31);
    cere_session *session = nullptr;
    CHECK_EQ(api.session_create(executor, model, 16, &session), CERE_OK);
    const uint32_t prompt[3] = {1, 2, 3};
    uint32_t out[2] = {0, 0};
    cere_sampling_params sampling{};
    sampling.struct_size = sizeof(sampling);
    sampling.abi_version = CERE_EXECUTOR_ABI_VERSION;
    sampling.top_p = 1.0f;
    sampling.top_k = config.vocab_size;
    sampling.repetition_penalty = 1.0f;
    cere_work item{};
    item.struct_size = sizeof(item);
    item.abi_version = CERE_EXECUTOR_ABI_VERSION;
    item.operation = CERE_OP_PREFILL;
    item.flags = CERE_WORK_NO_SPECULATION;
    item.expected_worker_epoch = 3;
    item.session = session;
    item.tokens = prompt;
    item.token_count = 3;
    item.max_new_tokens = 2;
    item.out_tokens = out;
    item.out_capacity = 2;
    item.sampling = &sampling;
    item.vocab_mask = mask.data();
    item.vocab_mask_words = words;
    cere_batch batch{};
    batch.struct_size = sizeof(batch);
    batch.abi_version = CERE_EXECUTOR_ABI_VERSION;
    batch.expected_worker_epoch = 3;
    batch.capability_profile_generation = caps.profile_generation;
    batch.items = &item;
    batch.item_count = 1;
    cere_completion_ticket ticket{};
    if (forward) {
        CHECK_EQ(api.submit(executor, &batch, &ticket), CERE_OK);
        cere_completion c{};
        c.struct_size = sizeof(c);
        c.abi_version = CERE_EXECUTOR_ABI_VERSION;
        CHECK_EQ(api.poll(executor, ticket, &c), CERE_PENDING);
        CHECK_EQ(c.committed_tokens, 1);
        CHECK_EQ(out[0], 7);  // the toy's own token (96) is masked: the smallest allowed id above it, wrapping
        CHECK_EQ(api.poll(executor, ticket, &c), CERE_PENDING);
        CHECK_EQ(c.committed_tokens, 1);
        CHECK_EQ(api.constrain(executor, ticket, 0, mask.data(), words), CERE_ERR_ARGUMENT);
        mask.assign(words, 0);
        mask[9000 >> 5] |= 1u << (9000 & 31);
        CHECK_EQ(api.constrain(executor, ticket, 1, mask.data(), words), CERE_OK);
        CHECK_EQ(api.poll(executor, ticket, &c), CERE_OK);
        CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
        CHECK_EQ(c.committed_tokens, 2);
        CHECK_EQ(out[1], 9000);
        CHECK_EQ(api.constrain(executor, ticket, 2, mask.data(), words), CERE_ERR_ARGUMENT);  // released
    } else {
        CHECK_EQ(api.submit(executor, &batch, &ticket), CERE_ERR_UNSUPPORTED);
        CHECK_EQ(ticket.submission_id, 0);
    }
    CHECK_EQ(api.session_release(executor, session), CERE_OK);
    CHECK_EQ(api.model_release(executor, model), CERE_OK);
    CHECK_EQ(api.executor_destroy(executor), CERE_OK);
}

// M2.9 (m29 design note C2, C3, D1) through dlsym: a model with speculative_draft_tokens 4 reports the verify
// operation, native_speculative_batch and 4 drafts exactly when the library has a forward pass; its PREFILL commits
// the toy's greedy chain one token, then 4 per speculation round (3 accepted drafts), and a VERIFY of a wrong caller
// draft commits the plain correction alone. Without a forward pass the work is UNSUPPORTED.
void speculating_round_trip(void *lib, bool forward) {
    if (!lib) return;
    const Api api = bind_api(lib);
    cere_executor *executor = toy_executor(api);
    const cere_qwen4_exp_config config = cere::toy::fixture_config();
    cere_tensor_desc tensor{};
    tensor.struct_size = sizeof(tensor);
    tensor.abi_version = CERE_EXECUTOR_ABI_VERSION;
    tensor.name = "model.embed_tokens.weight";
    tensor.dtype = CERE_DTYPE_BF16;
    tensor.placement = CERE_PLACEMENT_DEVICE;
    tensor.ndim = 2;
    tensor.shape[0] = 4;
    tensor.shape[1] = 8;
    tensor.bytes = 64;
    const uint32_t map[3] = {1, 2, 3};
    cere_model_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = CERE_EXECUTOR_ABI_VERSION;
    desc.model_kind = CERE_MODEL_QWEN4_EXP_NVFP4;
    desc.speculative_draft_tokens = 4;
    desc.draft_token_map = map;
    desc.draft_token_map_count = 3;
    desc.config = &config;
    desc.tensors = &tensor;
    desc.tensor_count = 1;
    desc.max_sessions = 1;
    desc.max_context_tokens = 16;
    cere_model *model = nullptr;
    CHECK_EQ(api.model_create(executor, &desc, &model), CERE_OK);
    std::vector<uint8_t> bytes(64, 0x5a);
    CHECK_EQ(api.model_put(executor, model, 0, 0, bytes.data(), bytes.size()), CERE_OK);
    CHECK_EQ(api.model_seal(executor, model), CERE_OK);
    cere_execution_capabilities caps{};
    caps.struct_size = sizeof(caps);
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    CHECK_EQ(api.capabilities(executor, model, &caps), CERE_OK);
    CHECK_EQ(caps.operations, forward ? (CERE_OPS_PREFILL | CERE_OPS_DECODE | CERE_OPS_VERIFY | CERE_OPS_CANCEL) : 0);
    CHECK_EQ(caps.native_speculative_batch, forward ? 1u : 0u);
    CHECK_EQ(caps.speculative_draft_tokens, forward ? 4u : 0u);

    cere_session *session = nullptr;
    CHECK_EQ(api.session_create(executor, model, 16, &session), CERE_OK);
    const uint32_t prompt[3] = {1, 2, 3};
    uint32_t out[5] = {0, 0, 0, 0, 0};
    cere_work item{};
    item.struct_size = sizeof(item);
    item.abi_version = CERE_EXECUTOR_ABI_VERSION;
    item.operation = CERE_OP_PREFILL;
    item.expected_worker_epoch = 3;
    item.session = session;
    item.tokens = prompt;
    item.token_count = 3;
    item.max_new_tokens = 5;
    item.out_tokens = out;
    item.out_capacity = 5;
    cere_batch batch{};
    batch.struct_size = sizeof(batch);
    batch.abi_version = CERE_EXECUTOR_ABI_VERSION;
    batch.expected_worker_epoch = 3;
    batch.capability_profile_generation = caps.profile_generation;
    batch.items = &item;
    batch.item_count = 1;
    cere_completion_ticket ticket{};
    if (forward) {
        CHECK_EQ(api.submit(executor, &batch, &ticket), CERE_OK);
        cere_completion c{};
        c.struct_size = sizeof(c);
        c.abi_version = CERE_EXECUTOR_ABI_VERSION;
        CHECK_EQ(api.poll(executor, ticket, &c), CERE_PENDING);
        CHECK_EQ(c.committed_tokens, 1);
        CHECK_EQ(api.poll(executor, ticket, &c), CERE_OK);
        CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
        CHECK_EQ(c.committed_tokens, 5);
        CHECK_EQ(c.verify_steps, 1);
        CHECK_EQ(c.accepted_drafts, 3);
        std::vector<uint32_t> chain = {1, 2, 3};
        for (uint32_t i = 0; i < 6; ++i) {
            chain.push_back(cere::toy::next_token(chain.back(), chain.size(), config.vocab_size));
        }
        for (uint32_t i = 0; i < 5; ++i) CHECK_EQ(out[i], chain[3 + i]);

        // VERIFY: one wrong caller draft -> the plain correction alone.
        const uint32_t wrong[1] = {(chain[8] + 1) % config.vocab_size};
        item.operation = CERE_OP_VERIFY;
        item.tokens = wrong;
        item.token_count = 1;
        item.max_new_tokens = 2;
        item.expected_session_revision = c.session_revision;
        CHECK_EQ(api.submit(executor, &batch, &ticket), CERE_OK);
        c = cere_completion{};
        c.struct_size = sizeof(c);
        c.abi_version = CERE_EXECUTOR_ABI_VERSION;
        CHECK_EQ(api.poll(executor, ticket, &c), CERE_OK);
        CHECK_EQ(c.finish, CERE_FINISH_LENGTH);
        CHECK_EQ(c.committed_tokens, 1);
        CHECK_EQ(c.verify_steps, 1);
        CHECK_EQ(c.accepted_drafts, 0);
        CHECK_EQ(out[0], chain[8]);
    } else {
        CHECK_EQ(api.submit(executor, &batch, &ticket), CERE_ERR_UNSUPPORTED);
        CHECK_EQ(ticket.submission_id, 0);
    }
    CHECK_EQ(api.session_release(executor, session), CERE_OK);
    CHECK_EQ(api.model_release(executor, model), CERE_OK);
    CHECK_EQ(api.executor_destroy(executor), CERE_OK);
}

}  // namespace

int main() {
    void *toy = open_library("CERE_TOY_EXECUTOR");
    void *noforward = open_library("CERE_TOY_EXECUTOR_NOFORWARD");
    round_trip(toy, true);
    round_trip(noforward, false);
    caps_rows(toy, true);
    caps_rows(noforward, false);
    constrained_round_trip(toy, true);
    constrained_round_trip(noforward, false);
    speculating_round_trip(toy, true);
    speculating_round_trip(noforward, false);
    return cere::test::finish("test_library");
}
