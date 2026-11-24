// The C ABI of include/cere_executor.h over the host state machine. Only these functions are exported from a library
// built with -fvisibility=hidden; no exception leaves them.
#pragma GCC visibility push(default)
#include "cere_executor.h"
#pragma GCC visibility pop

#include "cere_model_constants.h"
#include "host/device.h"
#include "host/family.h"
#include "host/kv_pool.h"
#include "host/lora.h"
#include "host/sha256.h"
#include "host/spec_calibrate.h"
#include "host/state.h"
#include "host/support.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <optional>
#include <set>
#include <string>
#include <utility>

namespace {

using cere::host::Ticket;

// struct_size first (CERE_ERR_ARGUMENT), then abi_version (CERE_ERR_UNSUPPORTED).
template <typename T>
cere_status check_header(const T *p) {
    if (p == nullptr || p->struct_size != sizeof(T)) return CERE_ERR_ARGUMENT;
    if (p->abi_version != CERE_EXECUTOR_ABI_VERSION) return CERE_ERR_UNSUPPORTED;
    return CERE_OK;
}

// Zeroes every field after the header of an output struct whose header is valid.
template <typename T>
void clear_body(T *p) {
    const uint32_t size = p->struct_size;
    const uint32_t version = p->abi_version;
    *p = T{};
    p->struct_size = size;
    p->abi_version = version;
}

template <typename F>
cere_status guarded(F &&body) noexcept {
    try {
        return body();
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    } catch (...) {
        return CERE_ERR_INTERNAL;
    }
}

cere_model *find_model(cere_executor *executor, const cere_model *model) {
    auto it = executor->models.find(model);
    return it == executor->models.end() ? nullptr : it->second.get();
}

cere_session *find_session(cere_executor *executor, const cere_session *session) {
    auto it = executor->sessions.find(session);
    return it == executor->sessions.end() ? nullptr : it->second.get();
}

cere_snapshot *find_snapshot(cere_executor *executor, const cere_snapshot *snapshot) {
    auto it = executor->snapshots.find(snapshot);
    return it == executor->snapshots.end() ? nullptr : it->second.get();
}

// The kinds a snapshot of `model` may hold (M29S2-07): kind 10 only with a draft, so the layout_id of a model without
// one is unchanged by M2.9.
uint32_t writable_kinds(const cere_model &model) {
    namespace snap = cere::host::snapshot;
    return model.plan.speculative_draft_tokens != 0 ? snap::kDraftWritableKinds : snap::kWritableKinds;
}

// What a restore into `session` (or, with no session, an import) of `model` accepts (m211 design note D14): a model
// without vision has no image sessions, so it cannot read kind 11 (M2.13 CR14).
cere::host::snapshot::Expect snapshot_expect(const cere_model &model, uint64_t session_max_context_tokens) {
    namespace snap = cere::host::snapshot;
    snap::Expect expect;
    expect.config = &model.plan.config;
    expect.layout_id = model.layout_id;
    expect.model_max_context_tokens = model.plan.max_context_tokens;
    expect.session_max_context_tokens = session_max_context_tokens;
    expect.known_flags = CERE_WORK_KNOWN_FLAGS;
    if (model.plan.support->vision == nullptr) expect.readable_kinds &= ~(uint32_t(1) << snap::kRopeDelta);
    if (model.plan.speculative_draft_tokens != 0) expect.readable_kinds |= uint32_t(1) << snap::kDraft;
    return expect;
}

// DM-2 / DM-3: CERE_ERR_INTERNAL when the device suffered a sticky fault (its context cannot be trusted any more), or
// when `model`'s prefix tree saw the device refuse a page reference change (the page accounting is corrupted); the
// supervisor then reloads the card. OK otherwise.
cere_status refuse_faulted(const cere_executor *executor, const cere_model *model = nullptr) {
    if (executor->device->faulted()) return CERE_ERR_INTERNAL;
    if (model != nullptr && model->prefix != nullptr && model->prefix->pool_faulted()) return CERE_ERR_INTERNAL;
    return CERE_OK;
}

// H4: holds the +1 lock of one prefix path for its scope (export keeps the path on the device while it copies it);
// every exit, an exception included, unlocks it.
class PathLock {
public:
    explicit PathLock(cere::host::PrefixCache &cache) : cache_(cache) {}
    ~PathLock() { release(); }
    PathLock(const PathLock &) = delete;
    PathLock &operator=(const PathLock &) = delete;
    void hold(cere::host::PrefixNode *node) {
        cache_.lock(node);
        node_ = node;
    }
    void release() {
        if (node_ != nullptr) cache_.unlock(node_);
        node_ = nullptr;
    }

private:
    cere::host::PrefixCache &cache_;
    cere::host::PrefixNode *node_ = nullptr;
};

// Takes `len` bytes of the model's arena for a new snapshot handle (CERE_ERR_CAPACITY when no gap fits), fills them
// with `fill(dst)` and registers the handle; on any failure (an exception included) the arena is unchanged. H3: no
// host-tier entry is dropped for a `len` that cannot fit even with every droppable entry gone.
template <typename F>
cere_status add_snapshot(cere_executor *executor, cere_model &model, const cere::host::snapshot::Header &header,
                         uint64_t len, F &&fill, cere_snapshot **out) {
    // Each snapshot starts 256-B aligned in the arena (D2: sections are DMA aligned relative to the image).
    if (len > std::numeric_limits<uint64_t>::max() - 255) return CERE_ERR_CAPACITY;
    const uint64_t bytes = (len + 255) / 256 * 256;
    if (model.prefix != nullptr ? !model.prefix->could_fit_after_drops(bytes)
                                : bytes > model.arena.capacity() - model.arena.used()) {
        return CERE_ERR_CAPACITY;
    }
    uint64_t offset = 0;
    while (!model.arena.allocate(bytes, &offset)) {
        // M2.10 D9 arena priority (ledger M3, CR11): host-tier prefix entries are dropped, LRU, before CAPACITY.
        if (model.prefix == nullptr || !model.prefix->drop_host_lru()) return CERE_ERR_CAPACITY;
    }
    // The range is the guard's until the handle is registered: any throw below returns it to the arena.
    struct Range {
        cere::host::ArenaAllocator &arena;
        uint64_t offset;
        bool armed = true;
        ~Range() {
            if (armed) arena.free(offset);
        }
    } range{model.arena, offset};
    auto snapshot = std::make_unique<cere_snapshot>();
    if (const cere_status st = fill(model.device->arena + offset); st != CERE_OK) return st;
    snapshot->model = &model;
    snapshot->offset = offset;
    snapshot->info.bytes = len;
    snapshot->info.position = header.position;
    snapshot->info.token_count = header.token_count;
    snapshot->info.fed = header.fed;
    std::memcpy(snapshot->info.layout_id, header.layout_id.data(), header.layout_id.size());
    cere_snapshot *handle = snapshot.get();
    executor->snapshots.emplace(handle, std::move(snapshot));
    range.armed = false;
    ++model.snapshots;
    *out = handle;
    return CERE_OK;
}

// Epoch and generation are stale (CERE_ERR_STALE_EPOCH); an unknown submission is CERE_ERR_ARGUMENT.
cere_status find_ticket(cere_executor *executor, const cere_completion_ticket &ticket, Ticket **out) {
    if (ticket.worker_epoch != executor->worker_epoch) return CERE_ERR_STALE_EPOCH;
    auto it = executor->tickets.find(ticket.submission_id);
    if (it == executor->tickets.end()) return CERE_ERR_ARGUMENT;
    if (it->second->model_generation != ticket.model_generation) return CERE_ERR_STALE_EPOCH;
    *out = it->second.get();
    return CERE_OK;
}

bool tokens_valid(const uint32_t *ids, uint64_t count, uint32_t vocab) {
    if (count != 0 && ids == nullptr) return false;
    for (uint64_t i = 0; i < count; ++i) {
        if (ids[i] >= vocab) return false;
    }
    return true;
}

// The M2.7 fields of a work item (cere_executor.h: cere_sampling_params and cere_work.sampling/out_logprobs/
// top_logprobs) against a model of `vocab` tokens on a device whose sampler is present or not.
cere_status check_sampling(const cere_work &item, uint32_t vocab, bool sampler) {
    if (item.top_logprobs > CERE_MAX_TOP_LOGPROBS) return CERE_ERR_ARGUMENT;
    if (item.top_logprobs != 0 && item.out_logprobs == nullptr) return CERE_ERR_ARGUMENT;
    if (item.sampling == nullptr && item.out_logprobs == nullptr) return CERE_OK;
    if (!sampler) return CERE_ERR_UNSUPPORTED;
    if (item.sampling == nullptr) return CERE_OK;
    const cere_sampling_params &s = *item.sampling;
    if (const cere_status st = check_header(&s); st != CERE_OK) return st;
    // Comparisons are false for NaN, so a NaN field fails its range.
    const auto within = [](float v, float lo, float hi) { return v >= lo && v <= hi; };
    const bool ranges = (s.temperature == 0.0f || within(s.temperature, 1e-6f, std::numeric_limits<float>::max())) &&
                        s.top_p > 0.0f && s.top_p <= 1.0f && within(s.min_p, 0.0f, 1.0f) && s.top_k >= 1 &&
                        s.top_k <= vocab && within(s.frequency_penalty, -2.0f, 2.0f) &&
                        within(s.presence_penalty, -2.0f, 2.0f) && s.repetition_penalty > 0.0f &&
                        s.repetition_penalty <= 2.0f && (s.flags & ~CERE_SAMPLING_KNOWN_FLAGS) == 0;
    if (!ranges) return CERE_ERR_ARGUMENT;
    if ((s.logit_bias == nullptr) != (s.logit_bias_count == 0) || s.logit_bias_count > vocab) return CERE_ERR_ARGUMENT;
    std::vector<bool> seen(s.logit_bias_count == 0 ? 0 : vocab);
    for (uint64_t i = 0; i < s.logit_bias_count; ++i) {
        const cere_logit_bias &b = s.logit_bias[i];
        if (b.id >= vocab || seen[b.id] || !std::isfinite(b.bias)) return CERE_ERR_ARGUMENT;
        seen[b.id] = true;
    }
    // The M2.12 group (ledger table N, L0-2): zero-required until its owner lands.
    if (s.suppress_token_ids != nullptr || s.suppress_token_count != 0 || s.mask_until_committed != 0) {
        return CERE_ERR_UNSUPPORTED;
    }
    return CERE_OK;
}

// A vocab mask (m28 design note A1, D9; xgrammar layout: id j allowed iff (mask[j >> 5] >> (j & 31)) & 1) for a
// model of `vocab` tokens: exactly ceil(vocab / 32) words, non-null, with at least one allowed id below vocab (an
// all-zero mask leaves no token to commit). Anything else is CERE_ERR_ARGUMENT.
cere_status check_mask(const uint32_t *mask, uint64_t words, uint32_t vocab) {
    if (mask == nullptr || words != (uint64_t(vocab) + 31) / 32) return CERE_ERR_ARGUMENT;
    for (uint64_t w = 0; w < words; ++w) {
        const uint64_t ids = std::min<uint64_t>(32, vocab - w * 32);
        const uint32_t below = ids == 32 ? ~uint32_t(0) : (uint32_t(1) << ids) - 1;
        if ((mask[w] & below) != 0) return CERE_OK;
    }
    return CERE_ERR_ARGUMENT;
}

// Structure of one tensor descriptor: CERE_ERR_ARGUMENT unless every field is well formed.
cere_status validate_tensor(const cere_tensor_desc &t) {
    if (const cere_status st = check_header(&t); st != CERE_OK) return st;
    if (t.name == nullptr || t.name[0] == '\0') return CERE_ERR_ARGUMENT;
    const uint64_t element = cere::host::dtype_size(t.dtype);
    if (element == 0) return CERE_ERR_ARGUMENT;
    if (t.placement < CERE_PLACEMENT_DEVICE || t.placement > CERE_PLACEMENT_DISCARD) return CERE_ERR_ARGUMENT;
    if (t.ndim > CERE_TENSOR_MAX_DIMS) return CERE_ERR_ARGUMENT;
    uint64_t bytes = element;
    for (uint32_t d = 0; d < CERE_TENSOR_MAX_DIMS; ++d) {
        if (d >= t.ndim) {
            if (t.shape[d] != 0) return CERE_ERR_ARGUMENT;
            continue;
        }
        if (t.shape[d] != 0 && bytes > std::numeric_limits<uint64_t>::max() / t.shape[d]) return CERE_ERR_ARGUMENT;
        bytes *= t.shape[d];
    }
    if (bytes == 0 || bytes != t.bytes) return CERE_ERR_ARGUMENT;
    return CERE_OK;
}

// ABI v5 (SelfTune ST2; the registry: crates/cere-runtime/src/levers.rs): one validation for the desc's selection and
// for cere_model_set_selection, resolving it into `*out` (every slot not listed at its default; written only on
// CERE_OK). Header as every struct; a malformed block or a value no library knows is ARGUMENT (block_version, count,
// a slot outside 1..CERE_SLOT_COUNT or listed twice, an impl outside its slot's set, a non-zero param the impl does
// not have, a non-zero choice past `count`); a known choice this library or model cannot run is UNSUPPORTED: a MoE
// backend other than the compiled one (Device::moe_lever), k2_gemm tail_split on a device without that route
// (Device::k2_tail_split), a draft window on a model without drafts, below 2 or above its speculative_draft_tokens
// (the verify window is that many rows: the committed token plus up to window - 1 drafts).
cere_status resolve_selection(const cere::host::Device &device, uint32_t speculative_draft_tokens,
                              const cere_slot_selection *s, cere::host::Levers *out) {
    if (const cere_status st = check_header(s); st != CERE_OK) return st;
    if (s->block_version != CERE_SLOT_BLOCK_VERSION || s->count > CERE_SLOT_MAX) return CERE_ERR_ARGUMENT;
    const cere_slot_choice zero{};
    for (uint32_t i = s->count; i < CERE_SLOT_MAX; ++i) {
        if (std::memcmp(&s->choices[i], &zero, sizeof zero) != 0) return CERE_ERR_ARGUMENT;
    }
    // Per slot (index CERE_SLOT_*): its largest impl value and how many params its impls have.
    static constexpr uint32_t kMaxImpl[CERE_SLOT_COUNT + 1] = {
        0, CERE_LEVER_K2_TAIL_SPLIT, CERE_LEVER_PREFILL_P4, CERE_LEVER_DECODE_SPEC_GRAPH, CERE_LEVER_MOE_K7,
        CERE_LEVER_DRAFT_WINDOW, CERE_LEVER_VERIFY_BATCHED, CERE_LEVER_K2_WEIGHTS_FP8};
    static constexpr uint32_t kParams[CERE_SLOT_COUNT + 1] = {0, 0, 0, 0, 0, 1, 0, 0};
    cere::host::Levers l;
    bool seen[CERE_SLOT_COUNT + 1] = {};
    for (uint32_t i = 0; i < s->count; ++i) {
        const cere_slot_choice &c = s->choices[i];
        if (c.slot == 0 || c.slot > CERE_SLOT_COUNT || seen[c.slot] || c.impl > kMaxImpl[c.slot]) {
            return CERE_ERR_ARGUMENT;
        }
        seen[c.slot] = true;
        for (uint32_t p = kParams[c.slot]; p < CERE_SLOT_PARAMS; ++p) {
            if (c.params[p] != 0) return CERE_ERR_ARGUMENT;
        }
        switch (c.slot) {
            case CERE_SLOT_K2_GEMM: l.k2_gemm = c.impl; break;
            case CERE_SLOT_PREFILL_ATTENTION: l.prefill_attention = c.impl; break;
            case CERE_SLOT_DECODE_STEP: l.decode_step = c.impl; break;
            case CERE_SLOT_MOE_GEMM: l.moe_gemm = c.impl; break;
            case CERE_SLOT_DRAFT_TOKENS: l.draft_tokens = c.params[0]; break;
            case CERE_SLOT_VERIFY_ATTENTION: l.verify_attention = c.impl; break;
            default: l.k2_weights = c.impl; break;  // CERE_SLOT_K2_WEIGHTS
        }
    }
    if (l.moe_gemm != CERE_LEVER_MOE_BUILD && l.moe_gemm != device.moe_lever()) return CERE_ERR_UNSUPPORTED;
    if (l.k2_gemm == CERE_LEVER_K2_TAIL_SPLIT && !device.k2_tail_split()) return CERE_ERR_UNSUPPORTED;
    if (l.draft_tokens != 0 &&
        (speculative_draft_tokens == 0 || l.draft_tokens < 2 || l.draft_tokens > speculative_draft_tokens)) {
        return CERE_ERR_UNSUPPORTED;
    }
    *out = l;
    return CERE_OK;
}

#if CERE_GEN_FAMILY == CERE_FAMILY_SDAR
// The config this library was compiled for (its descriptor, executor/models/<model_id>.json).
constexpr cere_sdar_config kSdarConfig = {
    sizeof(cere_sdar_config),
    CERE_EXECUTOR_ABI_VERSION,
    cere::gen::kVocabSize,
    cere::gen::kHiddenSize,
    cere::gen::kIntermediateSize,
    cere::gen::kNumHiddenLayers,
    cere::gen::kNumAttentionHeads,
    cere::gen::kNumKeyValueHeads,
    cere::gen::kHeadDim,
    cere::gen::kDecodeBlockSize,
    double(cere::gen::kRopeTheta),
    cere::gen::kRmsNormEps,
    cere::gen::kDecodeMaskTokenId,
    cere::gen::kWeightsTieWordEmbeddings ? 1u : 0u,
    cere::gen::kMaxPositionEmbeddings,
};

cere_status refused(cere_status status, const char *field) {
    std::fprintf(stderr, "cere: refused %s (family sdar)\n", field);
    return status;
}

// SdarOptIn (O320): resolve_selection on an sdar library, which also has the family-scoped slot CERE_SLOT_SDAR_GEMM.
// The block without that slot goes through resolve_selection, so every shared-slot rule is the same; the slot itself
// is ARGUMENT listed twice, with an impl above CERE_LEVER_SDAR_GEMM_CHANGING or with a non-zero param. `*sdar_gemm`
// (unlisted: identical) is written only on CERE_OK, with `*levers`.
static_assert(CERE_SLOT_SDAR_GEMM > CERE_SLOT_COUNT && CERE_SLOT_SDAR_GEMM < CERE_SLOT_MAX,
              "family-scoped slots sit above the shared ones and inside the block");
cere_status resolve_sdar_selection(const cere::host::Device &device, const cere_slot_selection *s,
                                   cere::host::Levers *levers, uint32_t *sdar_gemm) {
    if (const cere_status st = check_header(s); st != CERE_OK) return st;
    if (s->block_version != CERE_SLOT_BLOCK_VERSION || s->count > CERE_SLOT_MAX) return CERE_ERR_ARGUMENT;
    const cere_slot_choice zero{};
    for (uint32_t i = s->count; i < CERE_SLOT_MAX; ++i) {
        if (std::memcmp(&s->choices[i], &zero, sizeof zero) != 0) return CERE_ERR_ARGUMENT;
    }
    cere_slot_selection shared = *s;
    shared.count = 0;
    for (cere_slot_choice &c : shared.choices) c = zero;
    uint32_t gemm = CERE_LEVER_SDAR_GEMM_IDENTICAL;
    bool seen = false;
    for (uint32_t i = 0; i < s->count; ++i) {
        const cere_slot_choice &c = s->choices[i];
        if (c.slot != CERE_SLOT_SDAR_GEMM) {
            shared.choices[shared.count++] = c;
            continue;
        }
        if (seen || c.impl > CERE_LEVER_SDAR_GEMM_CHANGING) return CERE_ERR_ARGUMENT;
        for (uint32_t p = 0; p < CERE_SLOT_PARAMS; ++p) {
            if (c.params[p] != 0) return CERE_ERR_ARGUMENT;
        }
        seen = true;
        gemm = c.impl;
    }
    cere::host::Levers l;
    if (const cere_status st = resolve_selection(device, 0, &shared, &l); st != CERE_OK) return st;
    *levers = l;
    *sdar_gemm = gemm;
    return CERE_OK;
}

// plan_model on an SDAR library (SelfTune ST4), after the descriptor's own header: the family and model_kind, what
// DICE-8B runs without, then the config's leading header (before the cast) and every config value, each refused by
// name; then the device's SDAR support entry (null on the host toy: refused as `device`) and the Qwen path's tensor,
// session and KV pool checks. On family sdar the plan's `config` holds only the struct header and vocab_size (the
// host's token checks read it); the device runs its compiled constants, which the config equals.
cere_status plan_sdar_model(const cere::host::Device &device, const cere_model_desc &desc,
                            cere::host::ModelPlan *plan, std::vector<uint32_t> *stops) {
    if (const char *why = cere::host::family_refusal(desc, CERE_FAMILY_SDAR)) {
        return refused(CERE_ERR_UNSUPPORTED, why);
    }
    if (const char *why = cere::host::sdar_refusal(desc)) return refused(CERE_ERR_UNSUPPORTED, why);
    if (const cere::host::Refusal r = cere::host::config_refusal(desc.config, CERE_FAMILY_SDAR); r.field != nullptr) {
        return refused(r.status, r.field);
    }
    cere_sdar_config config{};
    std::memcpy(&config, desc.config, sizeof(config));
    if (const char *why = cere::host::sdar_config_refusal(config, kSdarConfig)) {
        return refused(CERE_ERR_UNSUPPORTED, why);
    }
    const cere::host::Support *support = cere::host::sdar_support(device);
    if (support == nullptr) return refused(CERE_ERR_UNSUPPORTED, "device");
    if (desc.tensors == nullptr || desc.tensor_count == 0) return CERE_ERR_ARGUMENT;
    if (desc.max_sessions == 0 || desc.max_context_tokens == 0) return CERE_ERR_ARGUMENT;
    if (desc.stop_token_count != 0 && desc.stop_token_ids == nullptr) return CERE_ERR_ARGUMENT;
    cere::host::Levers levers;
    uint32_t sdar_gemm = CERE_LEVER_SDAR_GEMM_IDENTICAL;
    if (desc.selection != nullptr) {
        if (const cere_status st = resolve_sdar_selection(device, desc.selection, &levers, &sdar_gemm);
            st != CERE_OK) {
            return st;
        }
    }
    std::set<std::string> names;
    plan->tensors.reserve(desc.tensor_count);
    for (uint64_t i = 0; i < desc.tensor_count; ++i) {
        const cere_tensor_desc &t = desc.tensors[i];
        if (const cere_status st = validate_tensor(t); st != CERE_OK) return st;
        if (!names.insert(t.name).second) return CERE_ERR_ARGUMENT;
        if (t.placement != CERE_PLACEMENT_DISCARD && (support->dtype_mask & cere::host::dtype_bit(t.dtype)) == 0) {
            return CERE_ERR_UNSUPPORTED;
        }
        cere::host::TensorPlan plan_tensor{t.name, t.dtype, t.placement, t.ndim, {}, t.bytes};
        std::copy(t.shape, t.shape + CERE_TENSOR_MAX_DIMS, plan_tensor.shape);
        plan->tensors.push_back(std::move(plan_tensor));
    }
    if (desc.max_sessions > support->max_sessions) return CERE_ERR_UNSUPPORTED;
    if (desc.max_context_tokens > support->max_context_tokens) return CERE_ERR_UNSUPPORTED;
    if (!tokens_valid(desc.stop_token_ids, desc.stop_token_count, config.vocab_size)) return CERE_ERR_ARGUMENT;
    uint64_t kv_pool_tokens = 0;
    if (const cere_status st = cere::host::plan_kv_pool(desc.kv_pool_tokens, desc.max_sessions,
                                                        desc.max_context_tokens, support->max_kv_pool_tokens,
                                                        &kv_pool_tokens);
        st != CERE_OK) {
        return st;
    }
    plan->support = support;
    plan->config = cere_qwen4_exp_config{};
    plan->config.struct_size = config.struct_size;
    plan->config.abi_version = config.abi_version;
    plan->config.vocab_size = config.vocab_size;
    plan->max_sessions = desc.max_sessions;
    plan->max_context_tokens = desc.max_context_tokens;
    plan->kv_pool_tokens = kv_pool_tokens;
    plan->levers = levers;
    plan->sdar_gemm = sdar_gemm;
    plan->calibrate_windows = desc.calibrate_windows;
    stops->assign(desc.stop_token_ids, desc.stop_token_ids + desc.stop_token_count);
    return CERE_OK;
}
#endif

#if CERE_GEN_FAMILY == CERE_FAMILY_INKLING2
// Bit i set when layer i of the compiled descriptor's per-layer list equals `value`.
template <size_t N> constexpr uint32_t layer_mask(const char *const (&layers)[N], const char *value) {
    static_assert(N <= 32, "a layer mask holds 32 layers");
    uint32_t mask = 0;
    for (size_t i = 0; i < N; ++i) {
        const char *a = layers[i];
        const char *b = value;
        while (*a != '\0' && *a == *b) ++a, ++b;
        if (*a == *b) mask |= uint32_t(1) << i;
    }
    return mask;
}

// The config this library was compiled for (its descriptor, executor/models/inkling-2-mini.json).
constexpr cere_inkling2_config kInkling2Config = {
    sizeof(cere_inkling2_config),
    CERE_EXECUTOR_ABI_VERSION,
    cere::gen::kVocabSize,
    cere::gen::kUnpaddedVocabSize,
    cere::gen::kHiddenSize,
    cere::gen::kNumHiddenLayers,
    cere::gen::kNumAttentionHeads,
    cere::gen::kNumKeyValueHeads,
    cere::gen::kHeadDim,
    cere::gen::kMoeIntermediateSize,
    cere::gen::kNumExperts,
    cere::gen::kNumExpertsPerTok,
    cere::gen::kSlidingWindowSize,
    cere::gen::kRelativePositionRelExtent,
    cere::gen::kRelativePositionDRel,
    cere::gen::kLoopCoreStart,
    cere::gen::kLoopCoreEnd,
    cere::gen::kLoopNumPasses,
    cere::gen::kLoopGateRank,
    cere::gen::kNLayersPerAttnResBlock,
    cere::gen::kConvKernelSize,
    cere::gen::kEosTokenId,
    layer_mask(cere::gen::kLayerMixer, "sliding_attention"),
    layer_mask(cere::gen::kLayerExperts, "mxfp8"),
    cere::gen::kRmsNormEps,
    cere::gen::kFinalLogitSoftcapping,
    cere::gen::kRouteScale,
    cere::gen::kMaxPositionEmbeddings,
};

cere_status refused(cere_status status, const char *field) {
    std::fprintf(stderr, "cere: refused %s (family inkling2_loop)\n", field);
    return status;
}

// plan_model on an Inkling 2 Mini library (inkling design note S0, S6g, D-I-S9B), after the descriptor's own header:
// the family and model_kind, what the library runs without (inkling2_refusal: a draft window other than off or 2..8, a
// draft token map, the prefix cache), then the config's leading header and every config value, each refused by name;
// then a vision_config's header (S6g: null = text-only, else the device's vision entry must equal it, both refused as
// `vision_config`), the device's Inkling support entry (null on the toy: refused as `device`) and the tensor, session
// and KV pool checks. The plan's `config` holds only the struct header and vocab_size (the host's token checks read
// it); the device runs its compiled constants. With drafts on, the MTP drafter's tensors (mtp.safetensors names, no
// prefix) reach Device::allocate with the target's, and the device decides whether it runs them (draft_supported).
cere_status plan_inkling2_model(const cere::host::Device &device, const cere_model_desc &desc,
                                cere::host::ModelPlan *plan, std::vector<uint32_t> *stops) {
    if (const char *why = cere::host::family_refusal(desc, CERE_FAMILY_INKLING2)) {
        return refused(CERE_ERR_UNSUPPORTED, why);
    }
    if (const char *why = cere::host::inkling2_refusal(desc)) return refused(CERE_ERR_UNSUPPORTED, why);
    if (const cere::host::Refusal r = cere::host::config_refusal(desc.config, CERE_FAMILY_INKLING2);
        r.field != nullptr) {
        return refused(r.status, r.field);
    }
    cere_inkling2_config config{};
    std::memcpy(&config, desc.config, sizeof(config));
    if (const char *why = cere::host::inkling2_config_refusal(config, kInkling2Config)) {
        return refused(CERE_ERR_UNSUPPORTED, why);
    }
    if (desc.vision_config != nullptr && check_header(desc.vision_config) != CERE_OK) {
        return refused(CERE_ERR_ARGUMENT, "vision_config");
    }
    const cere::host::Support *support = cere::host::inkling2_support(device, desc.vision_config != nullptr);
    if (support == nullptr) return refused(CERE_ERR_UNSUPPORTED, "device");
    if (!cere::host::vision_matches(*support, desc.vision_config)) {
        return refused(CERE_ERR_UNSUPPORTED, "vision_config");
    }
    if (desc.tensors == nullptr || desc.tensor_count == 0) return CERE_ERR_ARGUMENT;
    if (desc.max_sessions == 0 || desc.max_context_tokens == 0) return CERE_ERR_ARGUMENT;
    if (desc.stop_token_count != 0 && desc.stop_token_ids == nullptr) return CERE_ERR_ARGUMENT;
    cere::host::Levers levers;
    if (desc.selection != nullptr) {
        if (const cere_status st = resolve_selection(device, desc.speculative_draft_tokens, desc.selection, &levers);
            st != CERE_OK) {
            return st;
        }
    }
    std::set<std::string> names;
    plan->tensors.reserve(desc.tensor_count);
    for (uint64_t i = 0; i < desc.tensor_count; ++i) {
        const cere_tensor_desc &t = desc.tensors[i];
        if (const cere_status st = validate_tensor(t); st != CERE_OK) return st;
        if (!names.insert(t.name).second) return CERE_ERR_ARGUMENT;
        if (t.placement != CERE_PLACEMENT_DISCARD && (support->dtype_mask & cere::host::dtype_bit(t.dtype)) == 0) {
            return CERE_ERR_UNSUPPORTED;
        }
        // A text-only desc places no vision tower tensor (m213 design note I-ABI; a discarded one is never read).
        if (desc.vision_config == nullptr && t.placement != CERE_PLACEMENT_DISCARD &&
            std::strncmp(t.name, "model.visual.", 13) == 0) {
            return CERE_ERR_UNSUPPORTED;
        }
        cere::host::TensorPlan plan_tensor{t.name, t.dtype, t.placement, t.ndim, {}, t.bytes};
        std::copy(t.shape, t.shape + CERE_TENSOR_MAX_DIMS, plan_tensor.shape);
        plan->tensors.push_back(std::move(plan_tensor));
    }
    if (desc.max_sessions > support->max_sessions) return CERE_ERR_UNSUPPORTED;
    if (desc.max_context_tokens > support->max_context_tokens) return CERE_ERR_UNSUPPORTED;
    if (!tokens_valid(desc.stop_token_ids, desc.stop_token_count, config.vocab_size)) return CERE_ERR_ARGUMENT;
    uint64_t kv_pool_tokens = 0;
    if (const cere_status st = cere::host::plan_kv_pool(desc.kv_pool_tokens, desc.max_sessions,
                                                        desc.max_context_tokens, support->max_kv_pool_tokens,
                                                        &kv_pool_tokens);
        st != CERE_OK) {
        return st;
    }
    plan->support = support;
    plan->config = cere_qwen4_exp_config{};
    plan->config.struct_size = config.struct_size;
    plan->config.abi_version = config.abi_version;
    plan->config.vocab_size = config.vocab_size;
    plan->max_sessions = desc.max_sessions;
    plan->max_context_tokens = desc.max_context_tokens;
    plan->kv_pool_tokens = kv_pool_tokens;
    plan->speculative_draft_tokens = desc.speculative_draft_tokens;  // the draft token map stays empty (full vocab)
    plan->levers = levers;
    plan->calibrate_windows = desc.calibrate_windows;
    stops->assign(desc.stop_token_ids, desc.stop_token_ids + desc.stop_token_count);
    return CERE_OK;
}
#endif

// Validates the descriptor against the device's support table and builds the plan.
cere_status plan_model(const cere::host::Device &device, const cere_model_desc *desc, cere::host::ModelPlan *plan,
                       std::vector<uint32_t> *stops) {
    if (const cere_status st = check_header(desc); st != CERE_OK) return st;
#if CERE_GEN_FAMILY == CERE_FAMILY_SDAR
    // SelfTune ST4: this library is built for an SDAR model (gen::kFamily). It refuses another family, what DICE-8B
    // runs without (rows > 1, speculation, graphs, MXFP8 / FP8, prefix cache, snapshots, vision) and a config that is
    // not its compiled one, each by name on stderr.
    return plan_sdar_model(device, *desc, plan, stops);
#endif
#if CERE_GEN_FAMILY == CERE_FAMILY_INKLING2
    // Inkling 2 Mini (inkling design note S0): the same shape for family inkling2_loop.
    return plan_inkling2_model(device, *desc, plan, stops);
#endif
    // Ledger-owned fields (native-abi-ledger design note table D) whose owner has not landed: zero/null or UNSUPPORTED.
    // kv_pool_tokens is M2.5's (checked below by plan_kv_pool); snapshot_host_bytes is M2.11's (checked below);
    // vision_config is M2.13's (D 112, below); the draft fields are M2.9's (I1, I2: below, once vocab is known);
    // prefix_state_slots is M2.10's (D 104, below); selection is ABI v5's (resolve_selection, after the draft fields).
    if (desc->speculative_draft_tokens != 0 && desc->speculative_draft_tokens != 4) return CERE_ERR_UNSUPPORTED;
    // M2.10 (ledger D 104): the prefix cache needs a device that implements it (prefix_supported, which for the toy
    // includes its forward pass; the CUDA device's forward_supported() only turns true once a model is instantiated,
    // so it is not asked here); at most CERE_MAX_PREFIX_STATE_SLOTS slots.
    if (desc->prefix_state_slots != 0) {
        if (!device.prefix_supported()) return CERE_ERR_UNSUPPORTED;
        if (desc->prefix_state_slots > CERE_MAX_PREFIX_STATE_SLOTS) return CERE_ERR_ARGUMENT;
    }
    // M2.11 (ledger D 96): an arena needs a device that saves and loads session state.
    if (desc->snapshot_host_bytes != 0 && !device.snapshot_supported()) return CERE_ERR_UNSUPPORTED;
    if (desc->model_kind != CERE_MODEL_QWEN4_EXP_NVFP4) return CERE_ERR_UNSUPPORTED;
    // ABI v5 (Q5(a)): the family names the config struct; only the qwen4 experimental family exists.
    if (desc->family != CERE_FAMILY_QWEN4_EXP) return CERE_ERR_UNSUPPORTED;
    if (const cere_status st = check_header(desc->config); st != CERE_OK) return st;
    // D 112 (M2.13): null = text-only; else a v3 vision_config that must equal a support entry's (below).
    if (desc->vision_config != nullptr) {
        if (const cere_status st = check_header(desc->vision_config); st != CERE_OK) return st;
    }
    if (desc->tensors == nullptr || desc->tensor_count == 0) return CERE_ERR_ARGUMENT;
    if (desc->max_sessions == 0 || desc->max_context_tokens == 0) return CERE_ERR_ARGUMENT;
    if (desc->stop_token_count != 0 && desc->stop_token_ids == nullptr) return CERE_ERR_ARGUMENT;
    // M2.9 I2: a draft token map exactly when drafts are on: 1 <= count <= vocab ids, strictly increasing, < vocab.
    if (desc->speculative_draft_tokens == 0) {
        if (desc->draft_token_map != nullptr || desc->draft_token_map_count != 0) return CERE_ERR_ARGUMENT;
    } else {
        const uint64_t count = desc->draft_token_map_count;
        if (desc->draft_token_map == nullptr || count == 0 || count > desc->config->vocab_size) {
            return CERE_ERR_ARGUMENT;
        }
        for (uint64_t i = 0; i < count; ++i) {
            const uint32_t id = desc->draft_token_map[i];
            if (id >= desc->config->vocab_size || (i > 0 && id <= desc->draft_token_map[i - 1])) {
                return CERE_ERR_ARGUMENT;
            }
        }
    }
    if (desc->calibrate_windows > CERE_CALIBRATE_MTP) return CERE_ERR_ARGUMENT;
    cere::host::Levers levers;
    if (desc->selection != nullptr) {
        if (const cere_status st =
                resolve_selection(device, desc->speculative_draft_tokens, desc->selection, &levers);
            st != CERE_OK) {
            return st;
        }
    }

    std::set<std::string> names;
    plan->tensors.reserve(desc->tensor_count);
    for (uint64_t i = 0; i < desc->tensor_count; ++i) {
        const cere_tensor_desc &t = desc->tensors[i];
        if (const cere_status st = validate_tensor(t); st != CERE_OK) return st;
        if (!names.insert(t.name).second) return CERE_ERR_ARGUMENT;
        // A text-only desc carries no vision tower tensor (m213 design note I-ABI).
        if (desc->vision_config == nullptr && std::strncmp(t.name, "model.visual.", 13) == 0) {
            return CERE_ERR_UNSUPPORTED;
        }
        cere::host::TensorPlan plan_tensor{t.name, t.dtype, t.placement, t.ndim, {}, t.bytes};
        std::copy(t.shape, t.shape + CERE_TENSOR_MAX_DIMS, plan_tensor.shape);
        plan->tensors.push_back(std::move(plan_tensor));
    }

    const cere::host::Support *support = nullptr;
    for (const cere::host::Support *candidate : device.supported()) {
        if (cere::host::config_matches(*candidate, *desc->config) &&
            cere::host::vision_matches(*candidate, desc->vision_config)) {
            support = candidate;
        }
    }
    if (support == nullptr) return CERE_ERR_UNSUPPORTED;
    for (const cere::host::TensorPlan &t : plan->tensors) {
        if (t.placement != CERE_PLACEMENT_DISCARD && (support->dtype_mask & cere::host::dtype_bit(t.dtype)) == 0) {
            return CERE_ERR_UNSUPPORTED;
        }
    }
    if (desc->max_sessions > support->max_sessions) return CERE_ERR_UNSUPPORTED;
    if (desc->max_context_tokens > support->max_context_tokens) return CERE_ERR_UNSUPPORTED;
    if (!tokens_valid(desc->stop_token_ids, desc->stop_token_count, desc->config->vocab_size)) {
        return CERE_ERR_ARGUMENT;
    }
    uint64_t kv_pool_tokens = 0;
    if (const cere_status st = cere::host::plan_kv_pool(desc->kv_pool_tokens, desc->max_sessions,
                                                        desc->max_context_tokens, support->max_kv_pool_tokens,
                                                        &kv_pool_tokens);
        st != CERE_OK) {
        return st;
    }
    plan->support = support;
    plan->config = *desc->config;
    plan->max_sessions = desc->max_sessions;
    plan->max_context_tokens = desc->max_context_tokens;
    plan->snapshot_host_bytes = desc->snapshot_host_bytes;
    plan->prefix_state_slots = desc->prefix_state_slots;
    plan->kv_pool_tokens = kv_pool_tokens;
    plan->speculative_draft_tokens = desc->speculative_draft_tokens;
    plan->draft_token_map.assign(desc->draft_token_map, desc->draft_token_map + desc->draft_token_map_count);
    plan->levers = levers;
    plan->calibrate_windows = desc->calibrate_windows;
    stops->assign(desc->stop_token_ids, desc->stop_token_ids + desc->stop_token_count);
    return CERE_OK;
}

// Ledger W 152-160 (M2.13), resized by M2.13-L D-L7 (contract C2): at most 32 images per PREFILL item, each at most
// 4,096 merged rows; image rows per item are bounded by the session context only.
constexpr uint64_t kMaxImages = 32;
constexpr uint64_t kMaxImageTokens = 4096;

// Validates the images of a PREFILL item against its (already valid) prompt and copies them in prompt order:
// each span is `vision_start` then exactly t*h*w/merge^2 `image_token_id` rows at token_offset (at most
// kMaxImageTokens), spans sorted and disjoint inside the prompt, no pad row outside a span, pixel_bytes = t*h*w patch
// rows of BF16, pixel_sha256 not all zero (M2.13-L C1: the executor keys the prefix cache by it and trusts it as the
// digest of the pixels). Else ARGUMENT (a wrong abi_version is UNSUPPORTED).
cere_status plan_images(const cere_work &item, const cere_qwen4_exp_vision_config &vision,
                        std::vector<cere::host::SessionImage> *out) {
    const uint64_t merge = vision.spatial_merge_size;
    const uint64_t row_bytes = cere::host::patch_elements(vision) * sizeof(uint16_t);
    uint64_t next = 0;  // the first row after the previous span
    uint64_t pads = 0;
    for (uint64_t i = 0; i < item.image_count; ++i) {
        const cere_image_input &im = item.images[i];
        if (const cere_status st = check_header(&im); st != CERE_OK) return st;
        if (im.pad_id == 0 || im.pixel_patches == nullptr || im.grid_t == 0) return CERE_ERR_ARGUMENT;
        if (im.grid_h == 0 || im.grid_w == 0 || im.grid_h % merge != 0 || im.grid_w % merge != 0) {
            return CERE_ERR_ARGUMENT;
        }
        uint64_t patches = 0;
        uint64_t bytes = 0;
        if (__builtin_mul_overflow(uint64_t(im.grid_t) * im.grid_h, uint64_t(im.grid_w), &patches) ||
            __builtin_mul_overflow(patches, row_bytes, &bytes) || bytes != im.pixel_bytes) {
            return CERE_ERR_ARGUMENT;
        }
        const uint64_t n = patches / (merge * merge);
        if (n > kMaxImageTokens) return CERE_ERR_ARGUMENT;
        if (std::all_of(im.pixel_sha256, im.pixel_sha256 + 32, [](uint8_t b) { return b == 0; })) {
            return CERE_ERR_ARGUMENT;
        }
        const uint64_t at = im.token_offset;
        if (at == 0 || at - 1 < next || at >= item.token_count || n > item.token_count - at) {
            return CERE_ERR_ARGUMENT;
        }
        if (item.tokens[at - 1] != vision.vision_start_token_id) return CERE_ERR_ARGUMENT;
        for (uint64_t r = at; r < at + n; ++r) {
            if (item.tokens[r] != vision.image_token_id) return CERE_ERR_ARGUMENT;
        }
        next = at + n;
        pads += n;
    }
    if (uint64_t(std::count(item.tokens, item.tokens + item.token_count, vision.image_token_id)) != pads) {
        return CERE_ERR_ARGUMENT;
    }
    out->reserve(item.image_count);
    for (uint64_t i = 0; i < item.image_count; ++i) {
        const cere_image_input &im = item.images[i];
        const auto *pixels = static_cast<const uint8_t *>(im.pixel_patches);
        cere::host::SessionImage image{{im.token_offset, im.grid_t, im.grid_h, im.grid_w}, im.pad_id, {},
                                       std::vector<uint8_t>(pixels, pixels + im.pixel_bytes)};
        std::copy(im.pixel_sha256, im.pixel_sha256 + 32, image.pixel_sha256.begin());
        out->push_back(std::move(image));
    }
    return CERE_OK;
}

// M2.13-L (m213l design note D-L5, contract C6): the prefix-cache key of an image prompt: its tokens, each pad row
// j of an image replaced by 0x80000000 | (word[j % 8] & 0x7fffffff), word[] the 8 big-endian u32 of
// SHA-256(pixel_sha256 || grid_t || grid_h || grid_w as LE u32). A key word is never a token id (< 2^31), and another
// image (pixels or grid) keys its rows differently but for a 2^-31-per-word chance (248 bits per image of 8+ rows).
std::vector<uint32_t> image_keys(const cere_work &item, const std::vector<cere::host::SessionImage> &images) {
    std::vector<uint32_t> keys(item.tokens, item.tokens + item.token_count);
    for (const cere::host::SessionImage &im : images) {
        cere::host::Sha256 h;
        h.update(im.pixel_sha256.data(), im.pixel_sha256.size());
        for (const uint32_t v : {im.span.grid_t, im.span.grid_h, im.span.grid_w}) {
            const uint8_t le[4] = {uint8_t(v), uint8_t(v >> 8), uint8_t(v >> 16), uint8_t(v >> 24)};
            h.update(le, sizeof(le));
        }
        const std::array<uint8_t, 32> d = h.finish();
        const uint64_t rows = cere::mrope::image_tokens(im.span);
        for (uint64_t j = 0; j < rows; ++j) {
            const uint8_t *w = d.data() + 4 * (j % 8);
            const uint32_t word = uint32_t(w[0]) << 24 | uint32_t(w[1]) << 16 | uint32_t(w[2]) << 8 | uint32_t(w[3]);
            keys[im.span.token_offset + j] = 0x80000000u | (word & 0x7fffffffu);
        }
    }
    return keys;
}

bool fits(uint64_t want, uint64_t budget, uint64_t reserved) { return reserved <= budget && want <= budget - reserved; }

// Under CERE_WORK_IGNORE_STOP_TOKENS (M2.12 D9) no token finishes the ticket: stop tokens are ordinary commits.
bool is_stop(const cere_model &model, const Ticket &ticket, uint32_t token) {
    if (ticket.ignore_stop_tokens) return false;
    return std::find(ticket.stop_tokens.begin(), ticket.stop_tokens.end(), token) != ticket.stop_tokens.end() ||
           std::find(model.stop_tokens.begin(), model.stop_tokens.end(), token) != model.stop_tokens.end();
}

// The StopFn of a speculating ticket (host/device.h): `ctx` is the Ticket.
bool ticket_stop(const void *ctx, uint32_t token) {
    const Ticket &ticket = *static_cast<const Ticket *>(ctx);
    return is_stop(*ticket.session->model, ticket, token);
}

// Commits `score` (and `logprobs`, for a ticket with out_logprobs) as the ticket's next token (consuming the
// session's fed tokens and, for a constrained ticket, the mask of that position: D9 step 4); returns the finish
// reason it reaches (CERE_FINISH_NONE while not terminal).
uint32_t commit(Ticket &ticket, const cere_token_score &score, const cere_token_logprobs &logprobs) {
    cere_session &session = *ticket.session;
    const uint64_t committed = ticket.committed.load(std::memory_order_relaxed);
    session.fed = session.tokens.size();
    session.tokens.push_back(score.id);
    if (!session.keys.empty()) session.keys.push_back(score.id);  // M2.13-L C6: a committed id keys as itself
    ticket.out_tokens[committed] = score.id;
    if (ticket.out_scores != nullptr) ticket.out_scores[committed] = score;
    if (ticket.out_logprobs != nullptr) ticket.out_logprobs[committed] = logprobs;
    ticket.mask_ready = false;
    ticket.committed.store(committed + 1, std::memory_order_release);
    if (is_stop(*session.model, ticket, score.id)) return CERE_FINISH_STOP_TOKEN;
    if (committed + 1 >= ticket.max_new_tokens || session.tokens.size() >= session.max_context_tokens) {
        return CERE_FINISH_LENGTH;
    }
    return CERE_FINISH_NONE;
}

void end(Ticket &ticket, uint32_t finish, cere_status error = CERE_OK) {
    ticket.finish_pending = finish;
    ticket.error_pending = error;
}

// M2.10 (m210 design note D2, D15′ (b)): a session that looks up, splits, captures and inserts: on a cache-on model,
// namespaced; an image session only with its image keys (M2.13-L C6: a restored one has none).
bool cache_on(const cere_session &session) {
    return session.model->prefix != nullptr && session.namespaced && (!session.image_session || !session.keys.empty());
}

// The version of plan_captures' split rules below, a line of the prefill selection signature
// (cere_model_prefill_signature): bump it whenever plan_captures' split rules change (a point added, dropped or
// moved), so a profile's pinned signature stops matching.
constexpr uint32_t kPlanCapturesRule = 1;

// D4/D4′: the split points of a cache-on PREFILL ticket fed [h, n) after a lookup whose page match is k: (a′)
// floor64(n − 1); (c) W 132's p when p − h >= 8,192; (d) floor64(min(k, n − 1)) when k > h. A point outside (h, n)
// is dropped; coinciding points are one capture whose kind is not (c) when another kind is there (D7 evicts (c)
// states first). The (a) call ends need no split: calls start at h. D15‴ (ledger CR8): a one-call-layout drafting
// session has none. A split call is not bit-identical to one call (D8 (b)), so it keeps the cache-off call layout and
// its greedy output equals the cache-off model's; it hits only at prefill-chunk multiples (D15⁗) and its (a) call
// ends are captured. GdFix: a masked-prompt drafting session splits as a non-drafting one does, so its greedy output
// equals the NO_SPECULATION ticket's (grammar_drafts "on" == "off").
void plan_captures(Ticket &ticket, const cere_session &session, uint64_t h, uint64_t k) {
    using cere::host::CaptureKind;
    ticket.capture_points.clear();
    if (!cache_on(session) || session.one_call_layout()) return;
    const uint64_t n = session.prompt_tokens;
    const auto floor64 = [](uint64_t v) { return v / cere::host::kKvPageTokens * cere::host::kKvPageTokens; };
    const auto add = [&](uint64_t point, CaptureKind kind) {
        if (point <= h || point >= n) return;
        for (auto &entry : ticket.capture_points) {
            if (entry.first != point) continue;
            if (entry.second == CaptureKind::kBoundary) entry.second = kind;
            return;
        }
        ticket.capture_points.emplace_back(point, kind);
    };
    add(floor64(n - 1), CaptureKind::kTail);
    if (ticket.capture_position != 0 && ticket.capture_position - std::min(ticket.capture_position, h) >= 8192) {
        add(ticket.capture_position, CaptureKind::kBoundary);
    }
    if (k > h) add(floor64(std::min(k, n - 1)), CaptureKind::kBranch);
    std::sort(ticket.capture_points.begin(), ticket.capture_points.end());
}

// D4′: the `limit` of a cache-on PREFILL call, so it ends at the ticket's next split point (0 = none).
uint64_t capture_limit(const Ticket &ticket, const cere_session &session) {
    for (const auto &entry : ticket.capture_points) {
        if (entry.first > session.fed) return entry.first - session.fed;
    }
    return 0;
}

// m210 D15⁗′ stage 2a: on a model with end states on (cere_model_set_prefix_end_state; the opt-in prefix profiles), a
// drafting cache-on PREFILL ticket that kept the cache-off layout (`draft_layout`, calls on the prefill-chunk grid from
// its hit h) captures an end state at b = floor64(n − 1) when b lies strictly inside its cold call [s, e) (s =
// floor_C(b), e = min(s + C, n)) and past h; the node's class is e − s. PrefixCapture G1: a chunk-0 one (s = 0, a
// prompt below C) also saves its K/V rows [0, b) (Device::end_kv_slots; none servable: no end state).
void plan_end(cere_executor *executor, Ticket &ticket, const cere_session &session, uint64_t h) {
    ticket.end_at = 0;
    ticket.end_kv_slots = 0;
    if (!cache_on(session) || !session.model->prefix_end_state || !session.draft_layout) return;
    if (session.prompt_tokens == 0) return;
    const uint64_t chunk = executor->device->prefill_chunk(*session.model->device);
    const uint64_t n = session.prompt_tokens;
    const uint64_t b = (n - 1) / cere::host::kKvPageTokens * cere::host::kKvPageTokens;
    if (chunk == 0) return;
    const uint64_t s = b / chunk * chunk;
    if (b == s || b <= h) return;
    const uint64_t cls = std::min(s + chunk, n) - s;
    if (s == 0) {
        ticket.end_kv_slots = executor->device->end_kv_slots(*session.model->device, b, cls);
        if (ticket.end_kv_slots == 0) return;
    }
    ticket.end_at = b;
    ticket.end_start = s;
    ticket.end_cls = cls;
}

// m210 D15⁗′: after a successful step that captured the end state into the run of slots from `pair` (the pair, then
// its end_kv_slots): inserts it at tokens[0, b) as a draft state (its MTP row b − 1 took tokens[b]) of class end_cls,
// else frees the run (M210S-39 path mismatch, counted; or a state already there).
void insert_end(cere_executor *executor, const Ticket &ticket, cere_session &session, uint32_t pair) {
    cere::host::PrefixCache &cache = *session.model->prefix;
    const std::vector<uint32_t> pages = executor->device->session_page_list(*session.device);
    ++cache.stats().captures;
    if (!cache.own_path(session.ns, cere::host::key_data(session), ticket.end_at, pages.data())) {
        ++cache.stats().capture_page_mismatch;
    } else if (cache.insert(session.ns, cere::host::key_data(session), ticket.end_at, pages.data(), pair,
                            cere::host::CaptureKind::kTail, ticket.submission_id,
                            cere::host::key_data(session)[ticket.end_at], ticket.end_cls, ticket.end_kv_slots)) {
        return;
    }
    cache.free_run(pair, 2 + ticket.end_kv_slots);
}

// D15⁗ (ledger CR8): the next id of a draft state captured by `session` at `at`, none when it is not one. A drafting
// session that kept the cache-off layout (`draft_layout`) captures one inside its prompt (`at` <= the prompt: the
// prompt-end decode capture included) at a multiple of the prefill chunk; its MTP row at - 1 took tokens[at] (the
// next prompt token, or the committed first token at the prompt's end) as its next id, keyed as the path is (an image
// pad row by its image key word, M2.13-L C6).
std::optional<uint32_t> draft_next(cere_executor *executor, const cere_session &session, uint64_t at) {
    if (!session.draft_layout || at == 0 || at > session.prompt_tokens || at >= session.tokens.size()) return {};
    const uint64_t chunk = executor->device->prefill_chunk(*session.model->device);
    if (chunk == 0 || at % chunk != 0) return {};
    return cere::host::key_data(session)[at];
}

// D4 (b′): copies the session's state (it has consumed tokens[0, at), a multiple of 64) into a pool slot and inserts
// it with key tokens[0, at) and the session's pages; a node that already holds a device state there is only marked
// used. No obtainable slot (every state locked or tracked): skipped, counted (D4′). A path whose existing nodes hold
// other pages than the session's (another ticket's edge, e.g. a (d) point inside it): skipped, counted (M210S-39).
// With alternative edges (D17) only a state on the session's own path counts as held, and the path never mismatches.
cere_status capture(cere_executor *executor, cere_session &session, uint64_t at, cere::host::CaptureKind kind,
                    uint64_t clock) {
    cere_model &model = *session.model;
    cere::host::PrefixCache &cache = *model.prefix;
    std::vector<uint32_t> pages;
    if (cache.alt_edges()) pages = executor->device->session_page_list(*session.device);
    const cere::host::PrefixCache::Match held =
        cache.match(session.ns, cere::host::key_data(session), at, at, true, 0, pages.empty() ? nullptr : pages.data());
    if (held.h == at) {
        cache.touch(held.node, clock);
        return CERE_OK;
    }
    if (pages.empty()) pages = executor->device->session_page_list(*session.device);
    if (!cache.own_path(session.ns, cere::host::key_data(session), at, pages.data())) {
        ++cache.stats().capture_page_mismatch;
        return CERE_OK;
    }
    uint32_t slot = 0;
    if (!cache.take_slot(&slot, clock)) {
        cache.stats().skip(cere::host::SkipSite::kPrefillSlot);
        return CERE_OK;
    }
    if (const cere_status st = executor->device->capture_state(*model.device, *session.device, slot); st != CERE_OK) {
        cache.free_slot(slot);
        return st;
    }
    ++cache.stats().captures;
    if (!cache.insert(session.ns, cere::host::key_data(session), at, pages.data(), slot, kind, clock,
                      draft_next(executor, session, at))) {
        cache.free_slot(slot);
    }
    return CERE_OK;
}

// D4 (a)/(a′)/(c)/(d): after a cache-on PREFILL call that ended inside the prompt, at a split point or on the 64 grid.
cere_status prefill_capture(cere_executor *executor, Ticket &ticket) {
    cere_session &session = *ticket.session;
    if (!cache_on(session)) return CERE_OK;
    const uint64_t at = session.fed;
    if (at % cere::host::kKvPageTokens != 0) return CERE_OK;  // split points are on the grid too
    cere::host::CaptureKind kind = cere::host::CaptureKind::kCallEnd;
    for (const auto &entry : ticket.capture_points) {
        if (entry.first == at) kind = entry.second;
    }
    return capture(executor, session, at, kind, ticket.submission_id);
}

// D4 (b)/D4′: after a commit, a cache-on session whose position is now a multiple of 64 copies its state into its
// track slot (taken from the pool when it has none, at the ticket's `clock`, O112; none obtainable: skipped, counted
// once per ticket at SkipSite::kDecodeTrack, PT0a, though every later boundary retries); inserted at the terminal poll.
cere_status decode_capture(cere_executor *executor, Ticket &ticket) {
    cere_session &session = *ticket.session;
    if (!cache_on(session) || session.fed % cere::host::kKvPageTokens != 0) return CERE_OK;
    cere_model &model = *session.model;
    cere::host::PrefixCache &cache = *model.prefix;
    if (session.track_slot == cere::host::PrefixNode::kNoSlot &&
        !cache.take_slot(&session.track_slot, ticket.submission_id)) {
        if (!ticket.decode_skip_counted) cache.stats().skip(cere::host::SkipSite::kDecodeTrack);
        ticket.decode_skip_counted = true;
        return CERE_OK;
    }
    if (const cere_status st = executor->device->capture_state(*model.device, *session.device, session.track_slot);
        st != CERE_OK) {
        return st;
    }
    ++cache.stats().captures;
    session.tracked = session.fed;
    return CERE_OK;
}

// GdFix: a speculation round of a masked-prompt drafting session (one_call_layout false) captures as its
// NO_SPECULATION twin's decode steps do (D4 (b)): its rounds end on every multiple of 64 they reach (`round_chain`), so
// each capture there holds the state the twin's decode capture holds (greedy speculation commits plain decoding's
// tokens). A one-call-layout session's rounds capture nothing (ledger CR8).
cere_status round_capture(cere_executor *executor, Ticket &ticket) {
    return ticket.session->one_call_layout() ? CERE_OK : decode_capture(executor, ticket);
}

// GdFix: the drafts a round of a masked-prompt drafting session on a cache-on model may verify so that it ends on, never
// past, the next multiple of 64 (`round_capture`); `chain` for any other session.
uint64_t round_chain(const cere_session &session, uint64_t chain) {
    if (!cache_on(session) || session.one_call_layout()) return chain;
    return std::min(chain, cere::host::kKvPageTokens - 1 - session.fed % cere::host::kKvPageTokens);
}

// D4 (b) at a terminal poll: a CERE_OK ticket (a cancel included) inserts the session's latest decode capture, key
// tokens[0, t), donating the track slot to the tree (the next capture takes another from the pool); a failed ticket
// drops it and frees the slot (M210S-16). A path over other pages than the session's (M210S-39) inserts nothing,
// counted; the session keeps its track slot.
void insert_track(cere_executor *executor, cere_session &session, uint64_t clock, bool ok) {
    if (session.tracked == 0) return;
    cere::host::PrefixCache &cache = *session.model->prefix;
    if (ok) {
        const std::vector<uint32_t> pages = executor->device->session_page_list(*session.device);
        if (!cache.own_path(session.ns, cere::host::key_data(session), session.tracked, pages.data())) {
            ++cache.stats().capture_page_mismatch;
        } else if (cache.insert(session.ns, cere::host::key_data(session), session.tracked, pages.data(),
                                session.track_slot, cere::host::CaptureKind::kDecode, clock,
                                draft_next(executor, session, session.tracked))) {
            session.track_slot = cere::host::PrefixNode::kNoSlot;
        }
    } else {
        cache.free_slot(session.track_slot);
        session.track_slot = cere::host::PrefixNode::kNoSlot;
    }
    session.tracked = 0;
}

// Whether a ticket of `model` is in flight (M2.10 D11, D12, D16: flush, export and import are BUSY then).
bool model_busy(const cere_executor *executor, const cere_model &model) {
    for (const auto &entry : executor->tickets) {
        if (entry.second->session->model == &model) return true;
    }
    return false;
}

cere::host::CacheNamespace cache_namespace(const uint8_t id[16]) {
    cere::host::CacheNamespace ns;
    std::memcpy(ns.data(), id, ns.size());
    return ns;
}

// The images of an image session's open prompt (m213 design note D10, D11), passed with every step of it; null for a
// text session and once the prompt is consumed (the device then rotates at position + rope_delta, set at submit).
const cere::host::StepImages *step_images(const cere_session &session, cere::host::StepImages *storage) {
    if (!session.image_session || session.fed >= session.prompt_tokens || session.images.empty()) return nullptr;
    *storage = {session.images.data(), session.images.size(), session.positions3.data(),
                session.positions3.size() / 3};
    return storage;
}

// One single-row prompt step of the ticket with its sampling (M2.7 I8): a prompt chunk (M2.6 D2a; an image session's
// with its images, M2.13-L: every image session decodes in rows and speculation as a text session). The token is
// committed when the call ends the fed tokens. M2.10: a lazy session's first prefill round attaches its prefix first
// (D3′; a failed bind ends the ticket, nothing consumed), and a cache-on prompt call ends at the ticket's next split
// point (D4′ `limit`).
// MixedPD: `slice` (a prompt chunk only) lets the call pause between layer segments (host/device.h StepSlice); a
// paused ticket keeps its end-state pair and repeats the same step (fed, limit, end) at its next prefill round.
void solo_step(cere_executor *executor, Ticket &ticket, cere::host::StepSlice *slice = nullptr) {
    cere_session &session = *ticket.session;
    const bool prompt = session.fed < session.prompt_tokens;
    if (prompt && !session.bound) {
        cere::host::PrefixAttach hit;
        if (const cere_status st = cere::host::attach_prefix(*executor, session, ticket.submission_id, &hit);
            st != CERE_OK) {
            return end(ticket, CERE_FINISH_ERROR, st);
        }
        ticket.cached_tokens = hit.h;
        plan_captures(ticket, session, hit.h, hit.k);
        plan_end(executor, ticket, session, hit.h);
    }
    const uint64_t count = session.tokens.size() - session.fed;
    uint64_t limit = prompt ? capture_limit(ticket, session) : 0;
    // m210 D15⁗′: the first step after an end-state hit resumes it and ends where the lookup's own cold call [s, e)
    // does; D15⁗″: with the lookup's class e − s (the node's class has the same row-bit signature, the match rule), so
    // its rows are that call's. The step whose call holds a ticket's end point captures there into a run of pool slots
    // (the pair and its end_kv_slots; O247 then O112: freed from states idle longer than the session's reuse
    // interval, then only from end states and the session's own states on its path below b; none obtainable: skipped,
    // counted): the call from the chunk start, or (D15⁗″) the resumed call when b lies past its hit in that chunk (in
    // chunk 0 only with the resumed state's K/V run, which the capture's run continues).
    cere::host::EndStep end_step;
    const cere::host::EndStep *end_arg = nullptr;
    uint32_t pair = cere::host::PrefixNode::kNoSlot;
    cere::host::PrefixCache *cache = session.model->prefix.get();
    bool captures = prompt && ticket.end_at != 0 && session.fed == ticket.end_start;
    if (prompt && session.end_resume) {
        const uint64_t chunk = executor->device->prefill_chunk(*session.model->device);
        const uint64_t s = session.fed / chunk * chunk;
        end_step.cls = std::min(s + chunk, session.prompt_tokens) - s;
        end_step.resume = true;
        end_step.in_slot = session.locked->slot;
        end_step.in_kv_slots = session.locked->kv_slots;
        end_arg = &end_step;
        limit = s + end_step.cls - session.fed;
        captures = ticket.end_at > session.fed && ticket.end_start == s && (s != 0 || end_step.in_kv_slots != 0);
    }
    if (captures) {
        if (ticket.parked) {
            pair = ticket.park_pair;  // the paused call's pair (none: its first segment skipped the capture)
        } else {
            // D17: with alternative edges the session's own path is the one through its pages. O247: a session with a
            // hit first takes the slots of states idle longer than its own reuse interval (its hit's previous use).
            // O259 (ShortCache): a prompt below the prefill chunk (its end state in chunk 0) captures no chunk-boundary
            // state, so without a hit it never gets a reuse interval; its horizon is its own ticket's clock (every
            // unlocked state last used before this ticket, each path's deepest state excepted, oldest first).
            std::vector<uint32_t> pages;
            if (cache->alt_edges()) pages = executor->device->session_page_list(*session.device);
            const uint64_t horizon = session.locked != nullptr ? session.locked->prev_state_clock
                                     : ticket.end_start == 0 ? ticket.submission_id
                                                             : 0;
            if (!cache->take_run(&pair, 2 + ticket.end_kv_slots, session.ns, cere::host::key_data(session),
                                 ticket.end_at, pages.empty() ? nullptr : pages.data(), horizon)) {
                cache->stats().skip(cere::host::SkipSite::kEndPair);
            }
        }
        if (pair != cere::host::PrefixNode::kNoSlot) {
            end_step.row = ticket.end_at - session.fed;
            end_step.slot = pair;
            end_step.kv_slots = ticket.end_kv_slots;
            end_arg = &end_step;
        }
    }
    uint64_t consumed = 0;
    cere_token_score score{};
    cere_token_logprobs logprobs{};
    cere::host::StepImages images;
    const cere_status st = executor->device->step(
        *session.model->device, *session.device, session.tokens.data() + session.fed, count, limit,
        step_images(session, &images), &consumed, ticket.sampling ? &*ticket.sampling : nullptr, &score,
        ticket.out_logprobs != nullptr ? &logprobs : nullptr, end_arg, slice);
    if (st == CERE_OK && slice != nullptr && slice->paused) {
        ticket.parked = true;  // nothing consumed: the next prefill round resumes the call
        ticket.park_pair = pair;
        return;
    }
    ticket.parked = false;
    ticket.park_pair = cere::host::PrefixNode::kNoSlot;
    const bool captured =
        pair != cere::host::PrefixNode::kNoSlot && st == CERE_OK && consumed != 0 && consumed <= count;
    if (pair != cere::host::PrefixNode::kNoSlot && !captured) cache->free_run(pair, 2 + ticket.end_kv_slots);
    if (st == CERE_PENDING) return;  // the device did not step; a later round retries
    session.end_resume = false;
    if (st != CERE_OK) return end(ticket, CERE_FINISH_ERROR, st);
    if (consumed == 0 || consumed > count) return end(ticket, CERE_FINISH_ERROR, CERE_ERR_INTERNAL);
    if (captured) insert_end(executor, ticket, session, pair);
    if (consumed < count) {
        session.fed += consumed;  // the prompt continues in a later prefill round
        if (const cere_status cap = prefill_capture(executor, ticket); cap != CERE_OK) {
            end(ticket, CERE_FINISH_ERROR, cap);
        }
        return;
    }
    if (prompt) {  // M2.13-L: the prompt is consumed: its images and positions are read no more (the keys stay)
        std::vector<cere::host::SessionImage>().swap(session.images);
        std::vector<int64_t>().swap(session.positions3);
    }
    const uint32_t finish = commit(ticket, score, logprobs);
    if (const cere_status cap = decode_capture(executor, ticket); cap != CERE_OK) {
        return end(ticket, CERE_FINISH_ERROR, cap);
    }
    if (finish != CERE_FINISH_NONE) end(ticket, finish);
}

// One step_rows of `rows` (row r is tickets[r]'s decode token with its sampling) and its per-row outcomes.
void rows_step(cere_executor *executor, cere_model &model, const std::vector<Ticket *> &tickets,
               std::vector<cere::host::RowArgs> &rows) {
    executor->round_scores.assign(rows.size(), cere_token_score{});
    executor->round_logprobs.assign(rows.size(), cere_token_logprobs{});
    executor->round_status.assign(rows.size(), CERE_ERR_ARGUMENT);  // kept by a whole-call ARGUMENT
    for (size_t r = 0; r < rows.size(); ++r) {
        rows[r].logprobs = tickets[r]->out_logprobs != nullptr ? &executor->round_logprobs[r] : nullptr;
    }
    const cere_status st = executor->device->step_rows(*model.device, rows.data(), uint32_t(rows.size()),
                                                       executor->round_scores.data(), executor->round_status.data());
    for (size_t r = 0; r < tickets.size(); ++r) {
        cere_status row = executor->round_status[r];
        if (st != CERE_OK && row == CERE_OK) row = st;
        if (row == CERE_PENDING) continue;  // excluded from this round (toy hold)
        if (row != CERE_OK) {
            end(*tickets[r], CERE_FINISH_ERROR, row);
            continue;
        }
        const uint32_t finish = commit(*tickets[r], executor->round_scores[r], executor->round_logprobs[r]);
        if (const cere_status cap = decode_capture(executor, *tickets[r]); cap != CERE_OK) {
            end(*tickets[r], CERE_FINISH_ERROR, cap);
        } else if (finish != CERE_FINISH_NONE) {
            end(*tickets[r], finish);
        }
    }
}

// The drafts of a speculating ticket's round (not a VERIFY; host/device.h SpecRowArgs): up to window - 1 pending
// proposals, capped at `chain` (the drafts that still fit before the bonus token). A ticket with flag bit 2 or 3
// (M2.9L, m29l design note D2, D5) or bit 4 (m29a design note A2) lets its session's policy pick the round's
// drafts instead (ticket.pick, ticket.lookup_match): none, the first n MTP proposals (not under bit 3), or the first k
// suffix-lookup drafts (bit 2), which go through *drafts as a VERIFY's caller drafts would (null: the pending MTP
// proposals). Every pick is priced by the model's fixed table for its k2_gemm lever (`round_costs`, measured at seal on
// this library, O95): batch-concurrency design note BC-D4, so a sampled ticket's picks, and with them its seeded
// draws, depend only on its request and token stream, never on wall time or on which rounds ran single-row. The
// window is the model's draft_tokens slot (its window param, ABI v5), 0 = speculative_draft_tokens; read here every
// round, so a set_selection between requests takes effect at the next
// round. It only shortens the drafts inside the device's allocated draft buffers (sized for speculative_draft_tokens),
// and a greedy round commits plain decode's tokens whatever its draft count (D4), so every window commits the same
// tokens. DraftAccept (bit 5): a greedy adaptive lookup ticket asks the lookup for up to the device's
// max_chain_drafts drafts and DraftPolicy::extend decides the round's tail (DraftTail: only as the only drafting row
// of the call, `rows` = the call's drafting rows; with a lookup match that agrees with the session's three pending
// MTP proposals; when the fixed table says it pays): the lookup's first k > 3 instead of the policy's pick. TreeSpec
// (bit 6, same greedy adaptive lookup tickets, MTP drafts or not): before that, DraftPolicy::lengthen may widen the
// pick to the lookup's first k > 3 drafts on its own (no MTP agreement needed; again only as the call's only drafting
// row, when the fixed table says it pays). Both chains hold at most DraftPolicy::kMaxTailDrafts drafts, or with Chains
// (bit 7, O123) the device's max_chain_drafts. A ticket with a sampling (every constrained one) never takes them.
uint32_t choose_drafts(cere::host::Device &device, Ticket &ticket, uint32_t rows, uint64_t chain,
                       const uint32_t **drafts) {
    cere_session &session = *ticket.session;
    *drafts = nullptr;
    const uint32_t window = session.model->plan.levers.draft_tokens != 0 ? session.model->plan.levers.draft_tokens
                                                                         : session.model->plan.speculative_draft_tokens;
    const uint32_t room = uint32_t(std::min<uint64_t>(window - 1, chain));
    if (!ticket.lookup_drafts && !ticket.no_mtp_drafts && !ticket.adaptive_drafts) return room;
    using cere::host::DraftPolicy;
    uint32_t proposed = 0;
    ticket.lookup_match = 0;
    // DraftAccept (bit 5) and TreeSpec (bit 6) chains past kMaxDrafts: greedy adaptive lookup rounds whose window is
    // not capped below kMaxDrafts + 1 (bit 5 also needs the MTP drafts the tail follows).
    const bool deep_ok =
        ticket.adaptive_drafts && ticket.lookup_drafts && !ticket.sampling && room >= DraftPolicy::kMaxDrafts;
    const bool tail = ticket.tail_drafts && !ticket.no_mtp_drafts && deep_ok;
    const bool lengthen = ticket.long_drafts && deep_ok;
    const uint32_t most = ticket.chain_drafts ? device.max_chain_drafts(*session.model->device)
                                              : std::min(device.max_chain_drafts(*session.model->device),
                                                         DraftPolicy::kMaxTailDrafts);
    const uint32_t deep = tail || lengthen ? uint32_t(std::min<uint64_t>(most, chain)) : 0;
    if (ticket.lookup_drafts && room != 0) {
        const uint32_t want = std::max(std::min(room, DraftPolicy::kMaxDrafts), deep);
        proposed = session.drafter.propose(session.tokens, want, ticket.lookup.data(), &ticket.lookup_match);
    }
    // small_m and small_m_a8 (K2Prefill P3) share decode/verify kernels (m <= 16): table 1 (k2_cost_table).
    const cere::host::SpecCosts &costs =
        session.model->round_costs[cere::host::k2_cost_table(session.model->plan.levers.k2_gemm)];
    ticket.pick = session.policy.choose(costs, !ticket.no_mtp_drafts, room, proposed, ticket.lookup_match);
    if (lengthen && proposed > DraftPolicy::kMaxDrafts) {
        ticket.pick = session.policy.lengthen(
            costs, *ticket.pick, {ticket.lookup.data(), proposed, ticket.lookup_match, nullptr, 0, rows, most});
    }
    if (tail && proposed > DraftPolicy::kMaxDrafts) {
        uint32_t mtp[DraftPolicy::kMaxDrafts] = {};
        const uint32_t pending = device.pending_drafts(*session.device, mtp, DraftPolicy::kMaxDrafts);
        ticket.pick = session.policy.extend(
            costs, *ticket.pick, {ticket.lookup.data(), proposed, ticket.lookup_match, mtp, pending, rows, most});
    }
    if (ticket.pick->source == DraftPolicy::Source::kLookup || ticket.pick->source == DraftPolicy::Source::kTail) {
        *drafts = ticket.lookup.data();
    }
    return ticket.pick->drafts;
}

// The speculation row of a drafting session's ticket (host/device.h SpecRowArgs): its commit budget, its drafts (a
// VERIFY's caller drafts; a constrained ticket's, GrammarSpec GS-01: the prefix of its frozen offer the caller masked
// with cere_executor_constrain_drafts, row j's mask at `mask` row j, its frozen pick kept for spec_step's observe;
// none for a NO_SPECULATION or stop-ignoring ticket; else choose_drafts) and its stop predicate.
cere::host::SpecRowArgs spec_row(cere::host::Device &device, Ticket &ticket, uint32_t rows) {
    cere_session &session = *ticket.session;
    const uint64_t budget = std::min(ticket.max_new_tokens - ticket.committed.load(std::memory_order_relaxed),
                                     session.max_context_tokens - session.tokens.size());
    cere::host::SpecRowArgs row;
    row.session = session.device.get();
    row.token = session.tokens[session.fed];
    row.max_commit = uint32_t(std::min<uint64_t>(budget, std::numeric_limits<uint32_t>::max()));
    const uint64_t chain = budget - 1;  // drafts that still fit before the bonus token
#if CERE_GEN_FAMILY == CERE_FAMILY_INKLING2
    // inkling design note D-I-S9B: on an Inkling library a ticket with a sampling (a draw, logprobs or a mask)
    // verifies no drafts, VERIFY included: draft_count 0 commits plain decode's one token per round.
    if (ticket.sampling) {
        ticket.pick.reset();
    } else
#endif
    if (ticket.constrained()) {
        // The pick stays only with the drafts it froze (cere_executor_constrain resets it).
        row.draft_count = uint32_t(std::min<uint64_t>(ticket.mask_rows - 1, chain));
        row.drafts = row.draft_count != 0 ? ticket.offer.data() : nullptr;
    } else {
        ticket.pick.reset();
        if (ticket.verify) {
            row.draft_count = uint32_t(std::min<uint64_t>(ticket.drafts.size(), chain));
            row.drafts = row.draft_count != 0 ? ticket.drafts.data() : nullptr;
        } else if (!ticket.no_speculation && !ticket.ignore_stop_tokens) {
            row.draft_count = choose_drafts(device, ticket, rows, round_chain(session, chain), &row.drafts);
        }
    }
    row.sampling = ticket.sampling ? &*ticket.sampling : nullptr;
    if (!ticket.ignore_stop_tokens) {
        row.is_stop = ticket_stop;
        row.stop_ctx = &ticket;
    }
    return row;
}

// One speculate_rows of the round's drafting tickets and its per-row outcomes. A row's committed tokens go through
// `commit` in order (the device truncated them after the first stop token and at max_commit, so only the last may
// finish the ticket); each row counts one verify step and its accepted drafts. A failure before the commit consumed
// nothing (ERROR now); one after it (draft-extend) reports the committed tokens on this poll and ends the ticket
// ERROR at the next round (D7). A VERIFY ends after its one round (LENGTH unless a stop token finished it). M2.9L
// (m29l design note D2): a policy-picked row's outcome feeds its session's policy (a draft was rejected iff the
// round committed its accepted drafts plus a corrected token; a round truncated by a stop draft commits only accepted
// drafts). No round's wall time prices a window (batch-concurrency design note BC-D4: spec_row's fixed table).
void spec_step(cere_executor *executor, cere_model &model) {
    const std::vector<Ticket *> &tickets = executor->spec_tickets;
    std::vector<cere::host::SpecRowArgs> &rows = executor->spec_rows;
    const size_t n = rows.size();
    executor->spec_results.assign(n, cere::host::SpecRowResult{});
    constexpr size_t kStride = cere::host::kMaxSpecChainDrafts + 1;  // one entry per committed token of a chain
    executor->spec_logprobs.assign(kStride * n, cere_token_logprobs{});
    executor->spec_status.assign(n, CERE_ERR_ARGUMENT);  // kept by a whole-call ARGUMENT
    for (size_t r = 0; r < n; ++r) {
        rows[r].logprobs = tickets[r]->out_logprobs != nullptr ? &executor->spec_logprobs[kStride * r] : nullptr;
    }
    const cere_status st = executor->device->speculate_rows(*model.device, rows.data(), uint32_t(n),
                                                            executor->spec_results.data(),
                                                            executor->spec_status.data());
    for (size_t r = 0; r < n; ++r) {
        Ticket &t = *tickets[r];
        cere_status row = executor->spec_status[r];
        if (st != CERE_OK && row == CERE_OK) row = st;
        if (row == CERE_PENDING) continue;  // excluded from this round (toy hold)
        const cere::host::SpecRowResult &res = executor->spec_results[r];
        if (row != CERE_OK && res.committed == 0) {
            end(t, CERE_FINISH_ERROR, row);
            continue;
        }
        // The device contract (host/device.h SpecRowResult): 1..n+1 tokens within max_commit, only the last a stop.
        bool valid = res.committed >= 1 && res.committed <= rows[r].draft_count + 1 &&
                     res.committed <= rows[r].max_commit && res.accepted <= res.committed;
        for (uint32_t i = 0; valid && i + 1 < res.committed; ++i) valid = !is_stop(model, t, res.scores[i].id);
        if (!valid) {
            end(t, CERE_FINISH_ERROR, CERE_ERR_INTERNAL);
            continue;
        }
        uint32_t finish = CERE_FINISH_NONE;
        for (uint32_t i = 0; i < res.committed; ++i) {
            finish = commit(t, res.scores[i], executor->spec_logprobs[kStride * r + i]);
        }
        t.verify_steps += 1;
        t.accepted_drafts += res.accepted;
        t.session->verified_drafts += rows[r].draft_count;
        if (t.pick) {
            const bool rejected = res.accepted < t.pick->drafts && res.committed == res.accepted + 1;
            t.session->policy.observe(*t.pick, res.accepted, rejected, t.lookup_match);
        }
        if (row != CERE_OK) {
            t.error_deferred = row;
        } else if (const cere_status cap = round_capture(executor, t); cap != CERE_OK) {
            end(t, CERE_FINISH_ERROR, cap);
        } else if (finish != CERE_FINISH_NONE) {
            end(t, finish);
        } else if (t.verify) {
            end(t, CERE_FINISH_LENGTH);
        }
    }
}

// Decode round (D2b): every decode-ready ticket of one model steps once, in submission order, as one row of one
// step_rows with its own sampling and logprobs (ledger CR4, host/device.h RowArgs); the tickets of drafting sessions
// run one speculation round each through one speculate_rows instead (M2.9 D9 as M29S2-22; the plain rows are
// unchanged). Models take turns: the round's model is that of the first ticket after the last decode round's first
// ticket.
void decode_round(cere_executor *executor, const std::vector<Ticket *> &ready) {
    const Ticket *first = ready.front();
    for (const Ticket *t : ready) {
        if (t->submission_id > executor->decode_cursor) {
            first = t;
            break;
        }
    }
    executor->decode_cursor = first->submission_id;
    cere_model &model = *first->session->model;
    std::vector<Ticket *> &tickets = executor->round_tickets;
    std::vector<cere::host::RowArgs> &rows = executor->round_rows;
    tickets.clear();
    rows.clear();
    executor->spec_tickets.clear();
    executor->spec_rows.clear();
    for (Ticket *t : ready) {
        cere_session &session = *t->session;
        if (session.model != &model) continue;
        if (session.drafting) {  // M2.9: a drafting session decodes only through speculate_rows
            executor->spec_tickets.push_back(t);
            continue;
        }
        tickets.push_back(t);
        rows.push_back(cere::host::RowArgs{session.device.get(), session.tokens[session.fed],
                                           t->sampling ? &*t->sampling : nullptr, nullptr});
    }
    // DraftTail: each row's drafts are picked knowing the call's drafting rows (DraftPolicy::extend's tail rule).
    for (Ticket *t : executor->spec_tickets) {
        executor->spec_rows.push_back(spec_row(*executor->device, *t, uint32_t(executor->spec_tickets.size())));
    }
    if (!rows.empty()) rows_step(executor, model, tickets, rows);
    if (!executor->spec_rows.empty()) spec_step(executor, model);
}

// BC4 (b) (batch-concurrency design note): whether `ticket`'s first prefill round waits for an older prompt of the
// round (`prompts`, submission order) that shares at least one prefill chunk of its key prefix and has not fed it yet.
// Only a lazy, cache-on ticket (its lookup is still ahead) waits, and only on a ticket of its own model and cache
// namespace, so its lookup finds the states the owner captures, as under FIFO order. The owner is the older prompt
// with the longest shared prefix at the ticket's first check; older-only, so no two tickets wait on each other.
bool prefix_wait(cere_executor *executor, Ticket &ticket, const std::vector<Ticket *> &prompts) {
    const cere_session &session = *ticket.session;
    if (session.bound || !cache_on(session)) return false;
    if (!ticket.share_checked) {
        ticket.share_checked = true;
        const uint64_t chunk = executor->device->prefill_chunk(*session.model->device);
        const uint32_t *key = cere::host::key_data(session);
        for (const Ticket *older : prompts) {
            if (older->submission_id >= ticket.submission_id) break;
            const cere_session &o = *older->session;
            if (o.model != session.model || !cache_on(o) || o.ns != session.ns) continue;
            const uint64_t n = std::min(o.prompt_tokens, session.prompt_tokens);
            const uint32_t *other = cere::host::key_data(o);
            const uint64_t shared = uint64_t(std::mismatch(key, key + n, other).first - key);
            if (shared >= chunk && shared > ticket.share_tokens) {
                ticket.share_owner = older->submission_id;
                ticket.share_tokens = shared;
            }
        }
    }
    if (ticket.share_owner == 0) return false;
    for (const Ticket *older : prompts) {
        if (older->submission_id == ticket.share_owner) return older->session->fed < ticket.share_tokens;
    }
    return false;  // the owner left the round (ended, cancelled, or past its prompt)
}

// BC4 (b): whether `ticket` could run a prefill call now: its prompt is not consumed, it is not ending, and it does not
// wait for an older ticket's shared prefix (prefix_wait).
bool can_prefill(cere_executor *executor, Ticket &ticket, const std::vector<Ticket *> &prompts) {
    const cere_session &s = *ticket.session;
    if (ticket.finish_pending != CERE_FINISH_NONE || s.fed >= s.prompt_tokens) return false;
    return !prefix_wait(executor, ticket, prompts);
}

// BC4 (b): the next prompt: of the prompt tickets that could run (can_prefill), the oldest one overtaken
// prefill_age_calls times (the aging cap, cere_executor_set_prefill_age), else the one with the fewest prompt tokens
// left, ties to the oldest (shortest remaining prompt first: a short prompt that arrives during a long one waits for
// at most the long one's call in progress, and prompts of equal length keep their submission order). Null when none
// is left. `prompts` is in submission order.
Ticket *next_prompt(cere_executor *executor, const std::vector<Ticket *> &prompts) {
    Ticket *best = nullptr;
    uint64_t best_left = 0;
    for (Ticket *t : prompts) {
        if (!can_prefill(executor, *t, prompts)) continue;
        if (t->overtaken >= executor->prefill_age_calls) return t;
        const uint64_t left = t->session->prompt_tokens - t->session->fed;
        if (best == nullptr || left < best_left) {
            best = t;
            best_left = left;
        }
    }
    return best;
}

// BC4 (b): a prefill call that starts for `prompt` (not a paused call's next segment) overtakes every older prompt
// ticket that could run; the prompt's own count resets. An older ticket's call never counts against a younger one, and
// next_prompt runs the oldest capped ticket first, so no count passes prefill_age_calls.
void count_overtakes(cere_executor *executor, Ticket &prompt, const std::vector<Ticket *> &prompts) {
    for (Ticket *t : prompts) {
        if (t->submission_id >= prompt.submission_id) break;
        if (!can_prefill(executor, *t, prompts)) continue;
        t->overtaken += 1;
        t->overtaken_max = std::max(t->overtaken_max, t->overtaken);
    }
    prompt.overtaken = 0;
}

// S1 G4 (beat-sglang design note, cere_executor_set_decode_share): the decode share round policy. While two or more
// prompts wait they run back to back, no decode round between them (FIFO: the queue's first tokens come first; BC4
// (b)'s shortest-first order is the share-0 policy's). With one prompt left and decode-ready tickets, decode gets
// `share` % of that prompt's prefill wall time: each of its prefill rounds (a MixedPD segment included) adds its wall
// time x share / 100 to the decode time owed, and decode rounds run while that is positive, each taking its own wall
// time off (an overrun is not carried: the owed time stops at 0). Nothing is owed across a round with no prompt or
// with more than one, so a new prompt starts at once. Only the order of rounds changes (row invariance, M2.6 D6).
void share_round(cere_executor *executor, Ticket *prompt, uint32_t prompts, const std::vector<Ticket *> &ready) {
    using clock = std::chrono::steady_clock;
    if (prompt == nullptr || ready.empty() || prompts > 1) {
        executor->decode_owed_ns = 0;
        if (prompt == nullptr) return decode_round(executor, ready);
        return solo_step(executor, *prompt);  // slice null: a paused call runs to its end
    }
    const clock::time_point start = clock::now();
    const auto spent = [&start]() {
        return int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - start).count());
    };
    if (executor->decode_owed_ns > 0) {
        decode_round(executor, ready);
        executor->decode_owed_ns = std::max<int64_t>(0, executor->decode_owed_ns - spent());
        return;
    }
    cere::host::StepSlice slice{executor->prefill_slice_ms, false};
    solo_step(executor, *prompt, executor->prefill_slice_ms != 0 ? &slice : nullptr);
    executor->decode_owed_ns += spent() * int64_t(executor->decode_share_pct) / 100;
}

// Runs at most one executor-wide round (D2). Every running, non-cancelled ticket is checked first (D7): one with no
// room left ends LENGTH alone, one whose last speculation round failed after its commit ends ERROR (M2.9 D7). Then a
// ticket whose prompt is not consumed runs one chunk (prefill round), or every decode-ready ticket steps once (decode
// round). BC4 (b) (batch-concurrency design note): the prompt with the fewest tokens left runs next (next_prompt; a
// lazy cache-on ticket first waits for an older one feeding its shared prefix, prefix_wait; a ticket overtaken
// prefill_age_calls times by younger tickets' calls runs first, count_overtakes), and prefill rounds run
// back to back in a run of at most as many calls as the round has prompt tickets; with decode-ready tickets each run,
// or each paused segment, is followed by the executor's decode rounds per chunk (O79 PF1; 1: the kinds alternate), and
// with a prefill slice set a run stops early once its prefill rounds took the slice (the next call gets what is left
// of it), so a decode gap stays near one slice. With a decode share set the share policy picks (share_round). A
// ticket's chunks and their layout are its own (BC-D3): only the order of calls changes, which changes no ticket's
// tokens (row invariance, M2.6 D6). A constrained ticket without its mask for `committed` takes part in neither
// (m28 design note D9 step 3, ledger CR5): it waits for cere_executor_constrain, and no other ticket's round steps
// it. MixedPD: with a prefill slice set and a decode-ready ticket waiting, the prefill round's call may pause between
// layer segments (each segment is a prefill round); a paused ticket is the next prefill round's prompt, cancelled or
// not, until its call ends.
void run_round(cere_executor *executor) {
    using clock = std::chrono::steady_clock;
    Ticket *parked = nullptr;
    std::vector<Ticket *> &prompts = executor->round_prompts;  // the tickets whose prompt is not consumed
    prompts.clear();
    std::vector<Ticket *> &ready = executor->round_ready;
    ready.clear();
    for (auto &entry : executor->tickets) {
        Ticket &t = *entry.second;
        if (t.finish_pending != CERE_FINISH_NONE) continue;
        if (t.error_deferred != CERE_OK) {  // D7: a failure after a speculation commit (tokens reported last poll)
            end(t, CERE_FINISH_ERROR, t.error_deferred);
            continue;
        }
        if (t.parked) {
            parked = &t;
            prompts.push_back(&t);
            continue;
        }
        if (t.cancel_requested) continue;
        const cere_session &session = *t.session;
        if (t.committed.load(std::memory_order_relaxed) >= t.max_new_tokens ||
            session.tokens.size() >= session.max_context_tokens) {
            end(t, CERE_FINISH_LENGTH);
        } else if (!t.mask_present()) {
            continue;
        } else if (session.fed < session.prompt_tokens) {
            prompts.push_back(&t);
        } else if (session.tokens.size() <= session.fed) {
            // H1: a decode row feeds tokens[fed]; a session with no such token (a state it was never given) is
            // an internal inconsistency, never read out of bounds.
            end(t, CERE_FINISH_ERROR, CERE_ERR_INTERNAL);
        } else {
            ready.push_back(&t);
        }
    }
    if (executor->decode_share_pct != 0) {
        Ticket *oldest = parked != nullptr ? parked : prompts.empty() ? nullptr : prompts.front();
        if (oldest != nullptr || !ready.empty()) share_round(executor, oldest, uint32_t(prompts.size()), ready);
        return;
    }
    Ticket *prompt = parked != nullptr ? parked : next_prompt(executor, prompts);
    if (prompt != nullptr && (ready.empty() || executor->decode_rounds_owed == 0)) {
        if (ready.empty()) {  // no decode gap to bound
            executor->prefill_run_ns = 0;
            executor->prefill_run_calls = 0;
        }
        const uint32_t slice_ms = executor->prefill_slice_ms;
        const int64_t slice_ns = int64_t(slice_ms) * 1000000;
        const int64_t left_ms = (slice_ns - executor->prefill_run_ns) / 1000000;
        cere::host::StepSlice slice{uint32_t(std::clamp<int64_t>(left_ms, 1, slice_ms == 0 ? 1 : slice_ms)), false};
        executor->prefill_run_calls += 1;
        if (prompt != parked) count_overtakes(executor, *prompt, prompts);
        const clock::time_point start = clock::now();
        solo_step(executor, *prompt, slice_ms != 0 && !ready.empty() ? &slice : nullptr);
        executor->prefill_run_ns +=
            int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - start).count());
        const cere_session &s = *prompt->session;
        if (!prompt->parked && s.fed >= s.prompt_tokens && prompt->overtaken_max != 0) {
            std::fprintf(stderr, "cere: prefill_passed tokens=%llu max=%u cap=%u\n",
                         static_cast<unsigned long long>(s.prompt_tokens), prompt->overtaken_max,
                         executor->prefill_age_calls);
            prompt->overtaken_max = 0;
        }
        // The run goes on (no decode round owed) while a prompt waits, the call did not pause, the run has had fewer
        // calls than the round has prompt tickets and, with a slice set, it has not used the slice up.
        const bool pass_on = !prompt->parked && executor->prefill_run_calls < prompts.size() &&
                             (slice_ms == 0 || executor->prefill_run_ns < slice_ns) &&
                             next_prompt(executor, prompts) != nullptr;
        executor->decode_rounds_owed = pass_on ? 0 : executor->decode_rounds_per_chunk;
    } else if (!ready.empty()) {
        if (executor->decode_rounds_owed != 0) executor->decode_rounds_owed -= 1;
        executor->prefill_run_ns = 0;
        executor->prefill_run_calls = 0;
        decode_round(executor, ready);
    }
}

// Keeps a model's in-flight put count for the duration of one put.
class PutScope {
public:
    explicit PutScope(cere_model &model) : model_(model) { model_.puts_in_flight.fetch_add(1); }
    ~PutScope() { model_.puts_in_flight.fetch_sub(1); }
    PutScope(const PutScope &) = delete;
    PutScope &operator=(const PutScope &) = delete;

private:
    cere_model &model_;
};

// O95 / m23 design note SC-01: a drafting model on a device whose round time is its window's cost gets the K2 cost
// table of its k2_gemm lever measured on this library at seal (host/spec_calibrate.h), kept for the life of the load.
// One stderr line reports that table and the load time it took, or why the model keeps SpecCosts::untimed()
// (capacity: the calibration session does not fit; unsupported: a window lacks rounds). Any other status is a device
// failure and fails the seal.
cere_status calibrate_costs(cere::host::Device &device, cere_model &m) {
    if (!device.times_spec_rounds() || !device.draft_supported(*m.device)) return CERE_OK;
    const auto started = std::chrono::steady_clock::now();
    std::array<cere::host::SpecCosts, 2> tables;
    const cere_status st = cere::host::calibrate_round_costs(device, *m.device, m.plan, &tables);
    const std::chrono::duration<double, std::milli> ms = std::chrono::steady_clock::now() - started;
    if (st == CERE_ERR_CAPACITY || st == CERE_ERR_UNSUPPORTED) {
        std::fprintf(stderr, "cere: spec_round_costs source=untimed reason=%s load_ms=%.1f\n",
                     st == CERE_ERR_CAPACITY ? "capacity" : "unsupported", ms.count());
        return CERE_OK;
    }
    if (st != CERE_OK) return st;
    m.round_costs = tables;
    m.round_costs_measured = true;
    std::string line = "cere: spec_round_costs source=measured load_ms=";
    char value[32];
    std::snprintf(value, sizeof value, "%.1f", ms.count());
    line += value;
    // small_m_a8 (K2Prefill P3) prices rounds as small_m (m <= 16 runs the same kernels): key small_m; tail_split
    // as cublaslt (k2_cost_table).
    const bool small_m = cere::host::k2_cost_table(m.plan.levers.k2_gemm) == 1;
    line += small_m ? " small_m=" : " cublaslt=";
    const auto &entries = tables[small_m ? 1 : 0].table();
    for (size_t t = 0; t < entries.size(); ++t) {
        std::snprintf(value, sizeof value, "%s%.3f", t == 0 ? "" : ",", entries[t]);
        line += value;
    }
    std::fprintf(stderr, "%s\n", line.c_str());
    return CERE_OK;
}

// cere_model_prefill_signature (BC1 leftover; format v1 in cere_executor.h): the canonical text the digest is taken
// of, or false when the device has no prefill selection.
bool prefill_signature_text(const cere::host::Device &device, const cere::host::DeviceModel &model, std::string *out) {
    std::string text = "cere-prefill-signature-v1\n";
    text += "host plan_captures_rule " + std::to_string(kPlanCapturesRule) + "\n";
    text += "host prefill_chunk " + std::to_string(device.prefill_chunk(model)) + "\n";
    if (!device.prefill_selection(model, text)) return false;
    *out = std::move(text);
    return true;
}

}  // namespace

extern "C" {

uint32_t cere_executor_abi_version(void) { return CERE_EXECUTOR_ABI_VERSION; }

cere_status cere_executor_create(cere_executor **out, const cere_executor_options *options) {
    return guarded([&]() -> cere_status {
        if (out == nullptr) return CERE_ERR_ARGUMENT;
        *out = nullptr;
        if (const cere_status st = check_header(options); st != CERE_OK) return st;
        if (options->device_id == nullptr) return CERE_ERR_ARGUMENT;
        auto executor = std::make_unique<cere_executor>();
        if (const cere_status st = cere::host::create_device(options->device_id, &executor->device); st != CERE_OK) {
            return st;
        }
        executor->worker_epoch = options->worker_epoch;
        executor->memory_budget_bytes = options->memory_budget_bytes;
        *out = executor.release();
        return CERE_OK;
    });
}

cere_status cere_executor_destroy(cere_executor *executor) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        if (!executor->tickets.empty() || !executor->staging.empty()) return CERE_ERR_BUSY;
        {
            std::unique_lock<std::shared_mutex> lock(executor->handles);
            for (const auto &entry : executor->models) {
                if (entry.second->puts_in_flight.load() != 0) return CERE_ERR_BUSY;
            }
        }
        delete executor;
        return CERE_OK;
    });
}

cere_status cere_executor_capabilities(cere_executor *executor, const cere_model *model,
                                       cere_execution_capabilities *out) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        if (const cere_status st = check_header(out); st != CERE_OK) return st;
        clear_body(out);
        const cere_model *m = find_model(executor, model);
        if (m == nullptr || !m->sealed.load()) return CERE_ERR_ARGUMENT;
        const bool forward = executor->device->forward_supported();
        out->profile_generation = m->generation;
        out->worker_epoch = executor->worker_epoch;
        // D3: a decode round steps up to one row per session of the plan.
        out->decode_batch_min = 1;
        out->decode_batch_max = m->plan.max_sessions;
        out->native_decode_batch = forward && m->plan.max_sessions > 1 ? 1 : 0;
        out->max_context_tokens = m->plan.max_context_tokens;
        out->execution_mode = forward ? CERE_EXECUTION_MODE_NATIVE : CERE_EXECUTION_MODE_UNSUPPORTED;
        out->operations = forward ? (CERE_OPS_PREFILL | CERE_OPS_DECODE | CERE_OPS_CANCEL) : 0;
        out->native_sampler = forward && executor->device->sampler_supported() ? 1 : 0;
        // M2.8 (m28 design note A3, ledger C 58): a mask needs `sampling` (A1), so it is native with the sampler.
        out->native_token_mask = out->native_sampler;
        // M2.11 (ledger C 56, K): snapshot and restore exactly when the model has an arena.
        if (forward && m->plan.snapshot_host_bytes > 0) {
            out->native_state_snapshot = 1;
            out->operations |= CERE_OPS_SNAPSHOT | CERE_OPS_RESTORE;
        }
        out->kv_pool_tokens = m->plan.kv_pool_tokens;  // m25 design note D4 (ledger C 72)
        out->kv_page_tokens = static_cast<uint32_t>(cere::host::kKvPageTokens);
        out->native_image_input = m->plan.support->vision != nullptr ? 1 : 0;  // C 61 (M2.13)
        out->native_prefix_cache = m->plan.prefix_state_slots > 0 ? 1 : 0;  // C 60 (M2.10)
        // M2.9 (I3, ledger C 54/59): a model with a draft speculates natively and accepts VERIFY.
        if (forward && m->plan.speculative_draft_tokens != 0) {
            out->native_speculative_batch = 1;
            out->speculative_draft_tokens = static_cast<uint8_t>(m->plan.speculative_draft_tokens);
            out->operations |= CERE_OPS_VERIFY;
        }
        return CERE_OK;
    });
}

cere_status cere_model_create(cere_executor *executor, const cere_model_desc *desc, cere_model **out) {
    return guarded([&]() -> cere_status {
        if (out == nullptr) return CERE_ERR_ARGUMENT;
        *out = nullptr;
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        if (const cere_status st = refuse_faulted(executor); st != CERE_OK) return st;
        auto model = std::make_unique<cere_model>();
        if (const cere_status st = plan_model(*executor->device, desc, &model->plan, &model->stop_tokens);
            st != CERE_OK) {
            return st;
        }
        std::unique_lock<std::shared_mutex> lock(executor->handles);
        // MixedPD: a model created with the prefill slice set keeps the park region (cere_executor_set_prefill_slice).
        model->plan.prefill_park = executor->prefill_slice_ms != 0;
        model->estimate = executor->device->estimate(model->plan);
        if (!fits(model->estimate.device_bytes, executor->memory_budget_bytes, executor->reserved_device_bytes) ||
            !fits(model->estimate.host_bytes, executor->device->host_budget_bytes(), executor->reserved_host_bytes)) {
            return CERE_ERR_CAPACITY;
        }
        model->ledgers = std::make_unique<cere::host::TensorLedger[]>(model->plan.tensors.size());
        if (const cere_status st = executor->device->allocate(model->plan, &model->device); st != CERE_OK) return st;
        // M2.9: the device decides whether it runs the model's draft (allocate already refuses one it cannot run).
        if (model->plan.speculative_draft_tokens != 0 && !executor->device->draft_supported(*model->device)) {
            return CERE_ERR_UNSUPPORTED;
        }
        model->executor = executor;
        model->generation = ++executor->last_generation;
        model->arena = cere::host::ArenaAllocator(model->plan.snapshot_host_bytes);
        model->layout_id = cere::host::snapshot::layout_id(model->plan.config, writable_kinds(*model));
        if (model->plan.prefix_state_slots > 0) {
            // M2.10 D1/D9: the model's tree; the host tier exists with a snapshot arena (ledger M3).
            model->prefix = std::make_unique<cere::host::PrefixCache>(
                *executor->device, *model->device, model->plan,
                model->plan.snapshot_host_bytes > 0 ? &model->arena : nullptr, model->layout_id);
        }
        cere_model *handle = model.get();
        executor->models.emplace(handle, std::move(model));
        executor->reserved_device_bytes += handle->estimate.device_bytes;
        executor->reserved_host_bytes += handle->estimate.host_bytes;
        *out = handle;
        return CERE_OK;
    });
}

cere_status cere_model_put(cere_executor *executor, cere_model *model, uint64_t tensor_index, uint64_t offset,
                           const void *bytes, uint64_t len) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        std::shared_lock<std::shared_mutex> lock(executor->handles);
        cere_model *m = find_model(executor, model);
        if (m == nullptr) return CERE_ERR_ARGUMENT;
        // Counted before the sealed check; seal sets `sealed` before reading the count.
        PutScope scope(*m);
        if (m->sealed.load()) return CERE_ERR_ARGUMENT;
        if (tensor_index >= m->plan.tensors.size() || bytes == nullptr || len == 0) return CERE_ERR_ARGUMENT;
        const cere::host::TensorPlan &tensor = m->plan.tensors[tensor_index];
        if (offset > tensor.bytes || len > tensor.bytes - offset) return CERE_ERR_ARGUMENT;
        cere::host::TensorLedger &ledger = m->ledgers[tensor_index];
        if (!ledger.reserve(offset, len)) return CERE_ERR_ARGUMENT;
        if (tensor.placement == CERE_PLACEMENT_DISCARD) {
            m->discarded_bytes.fetch_add(len);
            return CERE_OK;
        }
        // M216S-11: a DEVICE put from a registered staging range is copied straight from it; the range stays held
        // (unregister is BUSY) until the copy returns.
        const uintptr_t staged =
            tensor.placement == CERE_PLACEMENT_DEVICE ? executor->staging.acquire(bytes, len) : uintptr_t(0);
        const cere_status st =
            staged != 0 ? executor->device->write_staged(*m->device, tensor_index, offset, bytes, len)
                        : executor->device->write(*m->device, tensor_index, offset, bytes, len);
        if (staged != 0) executor->staging.release(staged);
        if (st != CERE_OK) {
            ledger.unreserve(offset);
            return st;
        }
        (tensor.placement == CERE_PLACEMENT_DEVICE ? m->uploaded_device_bytes : m->uploaded_host_bytes).fetch_add(len);
        return CERE_OK;
    });
}

cere_status cere_model_host_resident(cere_executor *executor, cere_model *model, uint64_t tensor_index,
                                     const void **out) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
        *out = nullptr;
        std::shared_lock<std::shared_mutex> lock(executor->handles);
        cere_model *m = find_model(executor, model);
        if (m == nullptr) return CERE_ERR_ARGUMENT;
        // A put in flight for seal (cere_model_put's rule): counted before the sealed check.
        PutScope scope(*m);
        if (m->sealed.load() || tensor_index >= m->plan.tensors.size()) return CERE_ERR_ARGUMENT;
        const cere::host::TensorPlan &tensor = m->plan.tensors[tensor_index];
        if (tensor.placement != CERE_PLACEMENT_HOST_PINNED || tensor.bytes == 0) return CERE_ERR_ARGUMENT;
        const uint8_t *bytes = nullptr;
        if (const cere_status st = executor->device->host_resident(*m->device, tensor_index, &bytes); st != CERE_OK) {
            return st;
        }
        if (!m->ledgers[tensor_index].reserve(0, tensor.bytes)) return CERE_ERR_ARGUMENT;
        *out = bytes;
        return CERE_OK;
    });
}

// m216 design note M216S-11: put staging ranges (header: concurrent with puts, otherwise serialized). The toy built
// with CERE_TOY_PRE_STAGING lacks these exports and the round policy's, as a library from before them does (the
// runtime's fallback tests).
#ifndef CERE_TOY_PRE_STAGING
cere_status cere_executor_register_staging(cere_executor *executor, const void *base, uint64_t bytes) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr || base == nullptr || bytes == 0) return CERE_ERR_ARGUMENT;
        const auto *p = static_cast<const uint8_t *>(base);
        return executor->staging.add(p, bytes, [&] { return executor->device->register_staging(p, bytes); });
    });
}

cere_status cere_executor_unregister_staging(cere_executor *executor, const void *base) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr || base == nullptr) return CERE_ERR_ARGUMENT;
        const auto *p = static_cast<const uint8_t *>(base);
        return executor->staging.remove(p, [&](uint64_t bytes) { executor->device->unregister_staging(p, bytes); });
    });
}

// O79 PF1: the decode rounds per prefill chunk (header: takes effect at the next round). The pre-staging toy lacks it
// too, as a library from before both appends does (the runtime's refusal test).
cere_status cere_executor_set_decode_rounds(cere_executor *executor, uint32_t decode_rounds_per_chunk) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr || decode_rounds_per_chunk == 0 ||
            decode_rounds_per_chunk > CERE_MAX_DECODE_ROUNDS_PER_CHUNK) {
            return CERE_ERR_ARGUMENT;
        }
        executor->decode_rounds_per_chunk = decode_rounds_per_chunk;
        executor->decode_rounds_owed = std::min(executor->decode_rounds_owed, decode_rounds_per_chunk);
        return CERE_OK;
    });
}

// MixedPD (mixed-pd design note): the prefill slice (header: models created after the call). The pre-staging toy
// lacks it too, as a library from before the append does (the runtime's refusal test).
cere_status cere_executor_set_prefill_slice(cere_executor *executor, uint32_t slice_ms) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr || slice_ms > CERE_MAX_PREFILL_SLICE_MS) return CERE_ERR_ARGUMENT;
        executor->prefill_slice_ms = slice_ms;
        return CERE_OK;
    });
}

// S1 G4 (beat-sglang design note): the decode share (header: takes effect at the next round). The pre-staging toy
// lacks it too, as a library from before the append does (the runtime's refusal test).
cere_status cere_executor_set_decode_share(cere_executor *executor, uint32_t share_pct) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr || share_pct > CERE_MAX_DECODE_SHARE_PCT) return CERE_ERR_ARGUMENT;
        executor->decode_share_pct = share_pct;
        executor->decode_owed_ns = 0;
        return CERE_OK;
    });
}
#endif

// BC4 (b)'s aging cap (header: takes effect at the next prefill round).
cere_status cere_executor_set_prefill_age(cere_executor *executor, uint32_t calls) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr || calls == 0 || calls > CERE_MAX_PREFILL_AGE_CALLS) return CERE_ERR_ARGUMENT;
        executor->prefill_age_calls = calls;
        return CERE_OK;
    });
}

// ABI v5: the active selection as stored (the desc's, or the last successful cere_model_set_selection): every slot
// once, in slot order.
cere_status cere_model_selection(cere_executor *executor, cere_model *model, cere_slot_selection *out) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        if (const cere_status st = check_header(out); st != CERE_OK) return st;
        clear_body(out);
        std::shared_lock<std::shared_mutex> lock(executor->handles);
        const cere_model *m = find_model(executor, model);
        if (m == nullptr) return CERE_ERR_ARGUMENT;
        const cere::host::Levers &l = m->plan.levers;
        const uint32_t impls[CERE_SLOT_COUNT] = {l.k2_gemm,          l.prefill_attention, l.decode_step,
                                                 l.moe_gemm,         CERE_LEVER_DRAFT_WINDOW, l.verify_attention,
                                                 l.k2_weights};
        out->block_version = CERE_SLOT_BLOCK_VERSION;
        out->count = CERE_SLOT_COUNT;
        for (uint32_t i = 0; i < CERE_SLOT_COUNT; ++i) {
            out->choices[i].slot = i + 1;
            out->choices[i].impl = impls[i];
        }
        out->choices[CERE_SLOT_DRAFT_TOKENS - 1].params[0] = l.draft_tokens;
#if CERE_GEN_FAMILY == CERE_FAMILY_SDAR
        out->choices[CERE_SLOT_COUNT].slot = CERE_SLOT_SDAR_GEMM;  // SdarOptIn: the family-scoped slot, last
        out->choices[CERE_SLOT_COUNT].impl = m->plan.sdar_gemm;
        out->count = CERE_SLOT_COUNT + 1;
#endif
        return CERE_OK;
    });
}

// ABI v5 hot swap. In flight = a ticket submitted and not yet returned terminal by cere_executor_poll (the poll erases
// it from executor->tickets), the cere_executor_destroy rule; the header serializes the host API calls (only puts run
// concurrently, and a sealed model takes none), so no submit or poll races this check, and with no ticket no forward
// call of any model runs (every round runs inside a poll): the device re-resolves its selections with nothing reading
// them. The device applies first and the plan's copy (read by spec_row and cere_model_selection) changes only on its
// success, so on error the active selection is unchanged. k2_weights is fixed at create (INSTANTIATE converted the
// weights' storage in place): a value other than the model's is UNSUPPORTED, nothing changed. calibrate_windows is a
// desc field, not a slot (the seal measured the cost tables once), so no swap reaches it. On an sdar library
// CERE_SLOT_SDAR_GEMM is fixed at create the same way (Gemms::init picked its algorithms and workspace).
cere_status cere_model_set_selection(cere_executor *executor, cere_model *model, const cere_slot_selection *selection) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        std::shared_lock<std::shared_mutex> lock(executor->handles);
        cere_model *m = find_model(executor, model);
        if (m == nullptr || !m->sealed.load()) return CERE_ERR_ARGUMENT;
        if (!executor->tickets.empty()) return CERE_ERR_BUSY;
        cere::host::Levers next;
#if CERE_GEN_FAMILY == CERE_FAMILY_SDAR
        uint32_t sdar_gemm = CERE_LEVER_SDAR_GEMM_IDENTICAL;
        if (const cere_status st = resolve_sdar_selection(*executor->device, selection, &next, &sdar_gemm);
            st != CERE_OK) {
            return st;
        }
        if (const char *why = cere::host::sdar_selection_refusal(selection)) return refused(CERE_ERR_UNSUPPORTED, why);
        if (sdar_gemm != m->plan.sdar_gemm) return CERE_ERR_UNSUPPORTED;
#else
        if (const cere_status st =
                resolve_selection(*executor->device, m->plan.speculative_draft_tokens, selection, &next);
            st != CERE_OK) {
            return st;
        }
#endif
        if (next.k2_weights != m->plan.levers.k2_weights) return CERE_ERR_UNSUPPORTED;
        if (const cere_status st = executor->device->set_levers(*m->device, next); st != CERE_OK) return st;
        m->plan.levers = next;
        return CERE_OK;
    });
}

// m210 D15⁗′ (appended to ABI v4, no struct change): end states on or off for the model's drafting PREFILLs. Fixed once
// the model has had a ticket (BUSY, nothing changed), so the tree never mixes policies.
cere_status cere_model_set_prefix_end_state(cere_executor *executor, cere_model *model, uint32_t on) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr || on > 1) return CERE_ERR_ARGUMENT;
        std::shared_lock<std::shared_mutex> lock(executor->handles);
        cere_model *m = find_model(executor, model);
        if (m == nullptr) return CERE_ERR_ARGUMENT;
        if (m->prefix == nullptr) return CERE_ERR_UNSUPPORTED;
        if (m->submitted) return CERE_ERR_BUSY;
        m->prefix_end_state = on != 0;
        return CERE_OK;
    });
}

// m210 D17 (appended to ABI v4, no struct change): alternative edges on or off for the model's prefix tree. Fixed
// once the model has had a ticket (BUSY, nothing changed), like the end-state setter.
cere_status cere_model_set_prefix_alt_edges(cere_executor *executor, cere_model *model, uint32_t on) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr || on > 1) return CERE_ERR_ARGUMENT;
        std::shared_lock<std::shared_mutex> lock(executor->handles);
        cere_model *m = find_model(executor, model);
        if (m == nullptr) return CERE_ERR_ARGUMENT;
        if (m->prefix == nullptr) return CERE_ERR_UNSUPPORTED;
        if (m->submitted) return CERE_ERR_BUSY;
        m->prefix->set_alt_edges(on != 0);
        return CERE_OK;
    });
}

// Prefix cache counters (appended to ABI v4 after cere_model_set_prefix_alt_edges; the PrefixHit diagnosis): a plain
// copy of host::PrefixStats, no tree walk, allocation or lock beyond the handle table. The counters are written only
// inside serialized calls (submit/poll/cancel/import/flush) and this one is such a call, so it reads them quiescent,
// with tickets in flight too (a round runs inside cere_executor_poll). The toy built with CERE_TOY_PRE_STAGING lacks
// it, as a library from before it does (the runtime's fallback tests).
#ifndef CERE_TOY_PRE_STAGING
cere_status cere_model_prefix_stats(cere_executor *executor, cere_model *model, cere_prefix_stats *out) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        if (const cere_status st = check_header(out); st != CERE_OK) return st;
        clear_body(out);
        std::shared_lock<std::shared_mutex> lock(executor->handles);
        const cere_model *m = find_model(executor, model);
        if (m == nullptr) return CERE_ERR_ARGUMENT;
        if (m->prefix == nullptr) return CERE_ERR_UNSUPPORTED;
        const cere::host::PrefixStats &s = m->prefix->stats();
        out->alt_edges_on = m->prefix->alt_edges() ? 1 : 0;
        out->hits = s.hits;
        out->captures = s.captures;
        out->capture_skipped = s.capture_skipped;
        out->capture_page_mismatch = s.capture_page_mismatch;
        out->inserts = s.inserts;
        out->state_evictions = s.state_evictions;
        out->leaf_evictions = s.leaf_evictions;
        out->host_spills = s.host_spills;
        out->host_loads = s.host_loads;
        out->host_drops = s.host_drops;
        out->alt_edges = s.alt_edges;
        return CERE_OK;
    });
}

// PT0a (prefix-token design note): the capture-skip sites and lost-hit counters of host::PrefixStats, appended to ABI
// v4 after cere_model_prefix_stats with its contract (a plain copy; the pre-staging toy lacks it too).
cere_status cere_model_prefix_capture_stats(cere_executor *executor, cere_model *model,
                                            cere_prefix_capture_stats *out) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        if (const cere_status st = check_header(out); st != CERE_OK) return st;
        clear_body(out);
        std::shared_lock<std::shared_mutex> lock(executor->handles);
        const cere_model *m = find_model(executor, model);
        if (m == nullptr) return CERE_ERR_ARGUMENT;
        if (m->prefix == nullptr) return CERE_ERR_UNSUPPORTED;
        const cere::host::PrefixStats &s = m->prefix->stats();
        out->skipped_prefill_slot = s.skipped_prefill_slot;
        out->skipped_decode_track = s.skipped_decode_track;
        out->skipped_end_pair = s.skipped_end_pair;
        out->hit_shortfall_tokens = s.hit_shortfall_tokens;
        out->hit_shortfall_lookups = s.hit_shortfall_lookups;
        return CERE_OK;
    });
}
#endif

cere_status cere_model_seal(cere_executor *executor, cere_model *model) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        if (const cere_status st = refuse_faulted(executor); st != CERE_OK) return st;
        std::shared_lock<std::shared_mutex> lock(executor->handles);
        cere_model *m = find_model(executor, model);
        if (m == nullptr || m->sealed.exchange(true)) return CERE_ERR_ARGUMENT;
        if (m->puts_in_flight.load() != 0) {
            m->sealed.store(false);
            return CERE_ERR_BUSY;
        }
        for (size_t i = 0; i < m->plan.tensors.size(); ++i) {
            const cere::host::TensorPlan &tensor = m->plan.tensors[i];
            if (tensor.placement != CERE_PLACEMENT_DISCARD && m->ledgers[i].covered() != tensor.bytes) {
                m->sealed.store(false);
                return CERE_ERR_ARGUMENT;
            }
        }
        if (const cere_status st = executor->device->instantiate(*m->device); st != CERE_OK) {
            m->sealed.store(false);
            return st;
        }
        if (const cere_status st = calibrate_costs(*executor->device, *m); st != CERE_OK) {
            m->sealed.store(false);
            return st;
        }
        return CERE_OK;
    });
}

cere_status cere_model_release(cere_executor *executor, cere_model *model) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        std::unique_lock<std::shared_mutex> lock(executor->handles);
        auto it = executor->models.find(model);
        if (it == executor->models.end()) return CERE_ERR_ARGUMENT;
        if (it->second->sessions != 0 || it->second->snapshots != 0 || it->second->puts_in_flight.load() != 0) {
            return CERE_ERR_BUSY;
        }
        executor->reserved_device_bytes -=
            it->second->estimate.device_bytes + it->second->lora_bytes + it->second->research_bytes;
        executor->reserved_host_bytes -= it->second->estimate.host_bytes;
        executor->models.erase(it);
        return CERE_OK;
    });
}

cere_status cere_session_create(cere_executor *executor, cere_model *model, uint64_t max_context_tokens,
                                cere_session **out) {
    return guarded([&]() -> cere_status {
        if (out == nullptr) return CERE_ERR_ARGUMENT;
        *out = nullptr;
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        cere_model *m = find_model(executor, model);
        if (m == nullptr || !m->sealed.load() || max_context_tokens == 0) return CERE_ERR_ARGUMENT;
        if (const cere_status st = refuse_faulted(executor, m); st != CERE_OK) return st;
        if (max_context_tokens > m->plan.max_context_tokens || m->sessions >= m->plan.max_sessions) {
            return CERE_ERR_CAPACITY;
        }
        // M2.10 D3′: a session of a cache-on model is created lazy (no pages) and admitted while the pool can still
        // give every unbound session its whole context: free + evictable − unbound >= kv_pages(context).
        const bool lazy = m->prefix != nullptr;
        const uint64_t want = cere::host::kv_pages(max_context_tokens);
        if (lazy && !m->prefix->admits(want, m->unbound_pages)) return CERE_ERR_CAPACITY;
        auto session = std::make_unique<cere_session>();
        session->model = m;
        session->max_context_tokens = max_context_tokens;
        session->tokens.reserve(max_context_tokens);
        if (const cere_status st =
                executor->device->create_session(*m->device, max_context_tokens, lazy, &session->device);
            st != CERE_OK) {
            return st;
        }
        session->lazy = lazy;
        session->bound = !lazy;
        cere_session *handle = session.get();
        executor->sessions.emplace(handle, std::move(session));  // nothing below throws
        if (lazy) m->unbound_pages += want;
        ++m->sessions;
        *out = handle;
        return CERE_OK;
    });
}

cere_status cere_session_release(cere_executor *executor, cere_session *session) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        auto it = executor->sessions.find(session);
        if (it == executor->sessions.end()) return CERE_ERR_ARGUMENT;
        if (it->second->in_flight != 0) return CERE_ERR_BUSY;
        cere::host::release_prefix(*it->second);  // M2.10 D7: unlocks its hit path, frees its track slot
        cere_session &released = *it->second;
        if (released.lora_slot >= 0) --released.model->lora_sessions[size_t(released.lora_slot)];
        // WeightLab: its capture keeps the rows below fed and becomes readable; the slot loses a live session.
        if (released.research != nullptr) {
            std::lock_guard<std::mutex> research_lock(released.model->research_mutex);
            cere::research::finalise(*released.research, released.fed);
            --released.model->research_sessions[released.research->slot];
        }
        --released.model->sessions;
        executor->sessions.erase(it);
        return CERE_OK;
    });
}

cere_status cere_executor_submit(cere_executor *executor, const cere_batch *batch, cere_completion_ticket *out) {
    return guarded([&]() -> cere_status {
        if (out == nullptr) return CERE_ERR_ARGUMENT;
        *out = cere_completion_ticket{};
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        if (const cere_status st = check_header(batch); st != CERE_OK) return st;
        if (batch->expected_worker_epoch != executor->worker_epoch) return CERE_ERR_STALE_EPOCH;
        if (batch->items == nullptr || batch->item_count == 0) return CERE_ERR_ARGUMENT;
        if (batch->item_count > 1) return CERE_ERR_UNSUPPORTED;  // one item per batch: rows form across tickets (D1)
        const cere_work &item = batch->items[0];
        if (const cere_status st = check_header(&item); st != CERE_OK) return st;
        // Table F: unknown bits are ARGUMENT; bit 0 ignores stop tokens (M2.12 D9); bit 1 (M2.9) keeps the ticket out
        // of speculation (its PREFILL's bit 1 keeps the session from drafting; M2.13-L: image work may draft).
        // VERIFY with bit 0 is ARGUMENT: such requests run without speculation (m212 D9 (ii), C7b). Bits 2 and 3
        // (M2.9L D5) and 4 (m29a design note A2) choose the drafts of a drafting ticket's rounds; a VERIFY's drafts
        // are the caller's: ARGUMENT.
        if ((item.flags & ~CERE_WORK_KNOWN_FLAGS) != 0) return CERE_ERR_ARGUMENT;
        constexpr uint32_t kVerifyRefused = CERE_WORK_IGNORE_STOP_TOKENS | CERE_WORK_LOOKUP_DRAFTS |
                                            CERE_WORK_NO_MTP_DRAFTS | CERE_WORK_ADAPTIVE_DRAFTS |
                                            CERE_WORK_TAIL_DRAFTS | CERE_WORK_LONG_DRAFTS | CERE_WORK_CHAIN_DRAFTS;
        if (item.operation == CERE_OP_VERIFY && (item.flags & kVerifyRefused) != 0) return CERE_ERR_ARGUMENT;
        // M2.8 (m28 design note A1, ledger W 136/144): vocab_mask_words 0 = unconstrained with a null mask; a
        // constrained item needs its mask and `sampling` (the mask applies in the sampler stage), and is never a
        // VERIFY (ledger CR7). The word count and the mask itself are checked against the model's vocab below.
        if ((item.vocab_mask == nullptr) != (item.vocab_mask_words == 0)) return CERE_ERR_ARGUMENT;
        if (item.vocab_mask_words != 0 && (item.sampling == nullptr || item.operation == CERE_OP_VERIFY)) {
            return CERE_ERR_ARGUMENT;
        }
        // M2.10 W 132 (D4 (c)): PREFILL only, a multiple of 64 strictly inside the prompt; checked against the item
        // alone, never against cache contents (slots 0, no namespace or images: accepted and ignored).
        if (item.capture_position != 0 &&
            (item.operation != CERE_OP_PREFILL || item.capture_position % cere::host::kKvPageTokens != 0 ||
             item.capture_position >= item.token_count)) {
            return CERE_ERR_ARGUMENT;
        }
        if (item.expected_worker_epoch != executor->worker_epoch) return CERE_ERR_STALE_EPOCH;
        cere_session *session = find_session(executor, item.session);
        if (session == nullptr) return CERE_ERR_ARGUMENT;
        cere_model &model = *session->model;
        if (batch->capability_profile_generation != model.generation) return CERE_ERR_STALE_EPOCH;
        if (const cere_status st = refuse_faulted(executor, &model); st != CERE_OK) return st;
        if (!executor->device->forward_supported()) return CERE_ERR_UNSUPPORTED;
        // W 152-160 (M2.13, M2.13-L): images need a vision model and a PREFILL item of at most kMaxImages images.
        const bool has_images = item.images != nullptr || item.image_count != 0;
        if (has_images) {
            if (item.images == nullptr || item.image_count == 0) return CERE_ERR_ARGUMENT;
            if (model.plan.support->vision == nullptr) return CERE_ERR_UNSUPPORTED;
            if (item.operation != CERE_OP_PREFILL) return CERE_ERR_ARGUMENT;  // DECODE or VERIFY with images
            if (item.image_count > kMaxImages) return CERE_ERR_ARGUMENT;
        }
        // M2.9 (I4): VERIFY needs a model with a draft (its other checks follow the session's).
        if (item.operation == CERE_OP_VERIFY && model.plan.speculative_draft_tokens == 0) return CERE_ERR_UNSUPPORTED;
        if (item.operation != CERE_OP_PREFILL && item.operation != CERE_OP_DECODE && item.operation != CERE_OP_VERIFY) {
            return CERE_ERR_ARGUMENT;
        }
        if (session->in_flight != 0) return CERE_ERR_BUSY;
        if (item.expected_session_revision != session->revision) return CERE_ERR_STALE_EPOCH;
        if (item.max_new_tokens == 0 || item.out_tokens == nullptr || item.out_capacity < item.max_new_tokens) {
            return CERE_ERR_ARGUMENT;
        }
        const uint32_t vocab = model.plan.config.vocab_size;
        if (!tokens_valid(item.stop_token_ids, item.stop_token_count, vocab)) return CERE_ERR_ARGUMENT;
        if (const cere_status st = check_sampling(item, vocab, executor->device->sampler_supported()); st != CERE_OK) {
            return st;
        }
        if (item.vocab_mask_words != 0) {
            if (const cere_status st = check_mask(item.vocab_mask, item.vocab_mask_words, vocab); st != CERE_OK) {
                return st;
            }
        }
        std::vector<cere::host::SessionImage> images;
        std::vector<int64_t> positions3;
        int64_t rope_delta = 0;
        std::vector<uint32_t> keys;
        if (item.operation == CERE_OP_PREFILL) {
            if (!session->tokens.empty() || item.token_count == 0) return CERE_ERR_ARGUMENT;
            if (!tokens_valid(item.tokens, item.token_count, vocab)) return CERE_ERR_ARGUMENT;
            if (item.token_count >= session->max_context_tokens) return CERE_ERR_CAPACITY;
            if (has_images) {
                if (const cere_status st = plan_images(item, *model.plan.support->vision, &images); st != CERE_OK) {
                    return st;
                }
                std::vector<cere::mrope::ImageSpan> spans;
                for (const cere::host::SessionImage &im : images) spans.push_back(im.span);
                positions3.resize(3 * item.token_count);
                if (!cere::mrope::positions(item.token_count, spans, positions3.data(), &rope_delta)) {
                    return CERE_ERR_ARGUMENT;
                }
                if (model.prefix != nullptr && session->namespaced) keys = image_keys(item, images);
            }
        } else if (item.operation == CERE_OP_DECODE) {
            if (item.token_count != 0 || session->tokens.size() <= session->fed) return CERE_ERR_ARGUMENT;
        } else {
            // I4: a VERIFY carries 1..speculative_draft_tokens - 1 caller drafts for a decode-ready drafting session
            // (a NO_SPECULATION session never drafts: ARGUMENT, as an empty or mid-prompt session).
            if (item.token_count == 0 || item.token_count >= model.plan.speculative_draft_tokens) {
                return CERE_ERR_ARGUMENT;
            }
            if (!tokens_valid(item.tokens, item.token_count, vocab)) return CERE_ERR_ARGUMENT;
            if (!session->drafting || session->fed < session->prompt_tokens || session->tokens.size() <= session->fed) {
                return CERE_ERR_ARGUMENT;
            }
            if ((item.flags & CERE_WORK_NO_SPECULATION) != 0) return CERE_ERR_ARGUMENT;
        }
        // M2.9 (D9 as M29S2-22): a session drafts iff its model has a draft and its PREFILL lacks NO_SPECULATION; the
        // device learns it before the first step, and every PREFILL's rope_delta too (M2.13-L: an image prompt's
        // mRoPE value, 0 for text, so a value left by a refused submit never survives).
        const bool drafting = item.operation == CERE_OP_PREFILL && model.plan.speculative_draft_tokens != 0 &&
                              (item.flags & CERE_WORK_NO_SPECULATION) == 0;
        if (item.operation == CERE_OP_PREFILL) {
            if (const cere_status st = executor->device->set_rope_delta(*model.device, *session->device, rope_delta);
                st != CERE_OK) {
                return st;
            }
        }
        if (drafting) {
            if (const cere_status st = executor->device->set_session_draft(*model.device, *session->device, true);
                st != CERE_OK) {
                return st;
            }
        }

        auto ticket = std::make_unique<Ticket>();
        ticket->submission_id = executor->last_submission + 1;
        ticket->seen_round = executor->rounds;
        ticket->model_generation = model.generation;
        ticket->session = session;
        ticket->max_new_tokens = item.max_new_tokens;
        ticket->out_tokens = item.out_tokens;
        ticket->out_scores = item.out_scores;
        ticket->out_logprobs = item.out_logprobs;
        if (item.sampling != nullptr || item.out_logprobs != nullptr) {
            cere::host::ItemSampling &s = ticket->sampling.emplace();
            s.top_logprobs = item.top_logprobs;
            if (item.sampling != nullptr) {
                s.sampled = true;
                s.params = *item.sampling;
                s.bias.assign(item.sampling->logit_bias, item.sampling->logit_bias + item.sampling->logit_bias_count);
                s.params.logit_bias = nullptr;
            }
        }
        if (item.vocab_mask_words != 0) {  // D9: the initial mask is position 0's
            // GrammarSpec GS-01: a ticket that may draft holds a mask row per verify row of its rounds.
            const bool may_draft = (drafting || session->drafting) &&
                                   (item.flags & (CERE_WORK_NO_SPECULATION | CERE_WORK_IGNORE_STOP_TOKENS)) == 0;
            const uint64_t mask_rows = may_draft ? 1 + CERE_MAX_CONSTRAINED_DRAFTS : 1;
            ticket->mask.assign(mask_rows * item.vocab_mask_words, 0);
            std::copy(item.vocab_mask, item.vocab_mask + item.vocab_mask_words, ticket->mask.begin());
            ticket->mask_words = uint32_t(item.vocab_mask_words);
            ticket->mask_position = 0;
            ticket->mask_ready = true;
            ticket->sampling->vocab_mask = ticket->mask.data();  // `mask` is never resized, so this stays valid
        }
        ticket->stop_tokens.assign(item.stop_token_ids, item.stop_token_ids + item.stop_token_count);
        ticket->ignore_stop_tokens = (item.flags & CERE_WORK_IGNORE_STOP_TOKENS) != 0;
        ticket->no_speculation = (item.flags & CERE_WORK_NO_SPECULATION) != 0;
        ticket->lookup_drafts = (item.flags & CERE_WORK_LOOKUP_DRAFTS) != 0;
        ticket->no_mtp_drafts = (item.flags & CERE_WORK_NO_MTP_DRAFTS) != 0;
        ticket->adaptive_drafts = (item.flags & CERE_WORK_ADAPTIVE_DRAFTS) != 0;
        ticket->tail_drafts = (item.flags & CERE_WORK_TAIL_DRAFTS) != 0;
        ticket->long_drafts = (item.flags & CERE_WORK_LONG_DRAFTS) != 0;
        ticket->chain_drafts = (item.flags & CERE_WORK_CHAIN_DRAFTS) != 0;
        if (item.operation == CERE_OP_VERIFY) {
            ticket->verify = true;
            ticket->drafts.assign(item.tokens, item.tokens + item.token_count);
        }
        ticket->reserved_peak_bytes = batch->reserved_peak_bytes;
        ticket->capture_position = item.capture_position;
        const uint64_t id = ticket->submission_id;
        executor->tickets.emplace(id, std::move(ticket));
        session->model->submitted = true;
        if (item.operation == CERE_OP_PREFILL) {
            session->tokens.assign(item.tokens, item.tokens + item.token_count);
            session->images = std::move(images);
            session->positions3 = std::move(positions3);
            session->keys = std::move(keys);
            session->rope_delta = rope_delta;
            session->image_session = has_images;
            session->flags = item.flags;
            session->prompt_tokens = item.token_count;
            session->drafting = drafting;
            session->masked_prompt = item.vocab_mask_words != 0;
        }
        session->submitted = true;
        executor->last_submission = id;
        session->in_flight = id;
        *out = cere_completion_ticket{executor->worker_epoch, model.generation, id};
        return CERE_OK;
    });
}

cere_status cere_executor_cancel(cere_executor *executor, cere_completion_ticket ticket) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        Ticket *t = nullptr;
        if (const cere_status st = find_ticket(executor, ticket, &t); st != CERE_OK) return st;
        t->cancel_requested = true;
        return CERE_OK;
    });
}

cere_status cere_executor_poll(cere_executor *executor, cere_completion_ticket ticket, cere_completion *out) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        if (const cere_status st = check_header(out); st != CERE_OK) return st;
        clear_body(out);
        Ticket *t = nullptr;
        if (const cere_status st = find_ticket(executor, ticket, &t); st != CERE_OK) return st;
        // S1 G3: a ticket not reported since the last round (another ticket's poll ran it) is reported without a
        // round, so each round's results reach every polled ticket before the next round runs.
        if (t->seen_round == executor->rounds) {
            run_round(executor);
            executor->rounds += 1;
        }
        t->seen_round = executor->rounds;
        uint32_t finish = t->finish_pending;
        // MixedPD: a cancelled ticket whose call is paused ends at the poll after the call does.
        if (finish == CERE_FINISH_NONE && t->cancel_requested && !t->parked) finish = CERE_FINISH_CANCELLED;
        cere_session &session = *t->session;
        out->committed_tokens = t->committed.load(std::memory_order_acquire);
        out->cached_tokens = static_cast<uint32_t>(t->cached_tokens);  // M2.10 P 44: resolved at the first round
        out->verify_steps = t->verify_steps;
        out->accepted_drafts = t->accepted_drafts;
        if (finish == CERE_FINISH_NONE) {
            out->status = CERE_PENDING;
            out->session_revision = session.revision;
            return CERE_PENDING;
        }
        // Terminal: the ticket is released here and the leased buffers return to the caller. H5: everything that
        // cannot throw runs first, the decode-track insert (it allocates) last, so an exception out of it fails this
        // poll with the ticket already gone; a ticket left in `tickets` with in_flight 0 would keep the model BUSY
        // for good.
        const uint64_t id = t->submission_id;
        session.in_flight = 0;
        ++session.revision;
        out->status = finish == CERE_FINISH_ERROR ? t->error_pending : CERE_OK;
        out->terminal = 1;
        out->finish = finish;
        out->session_revision = session.revision;
        out->released_bytes = t->reserved_peak_bytes;
        executor->tickets.erase(id);
        insert_track(executor, session, id, out->status == CERE_OK);  // M2.10 D4 (b)
        return CERE_OK;
    });
}

cere_status cere_executor_constrain(cere_executor *executor, cere_completion_ticket ticket, uint64_t position,
                                    const uint32_t *vocab_mask, uint64_t vocab_mask_words) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        Ticket *t = nullptr;
        if (const cere_status st = find_ticket(executor, ticket, &t); st != CERE_OK) return st;
        // m28 design note A2, D9: only a constrained ticket, only for the position it commits next.
        if (!t->constrained() || position != t->committed.load(std::memory_order_relaxed)) return CERE_ERR_ARGUMENT;
        const uint32_t vocab = t->session->model->plan.config.vocab_size;
        if (const cere_status st = check_mask(vocab_mask, vocab_mask_words, vocab); st != CERE_OK) return st;
        std::copy(vocab_mask, vocab_mask + vocab_mask_words, t->mask.begin());
        t->mask_position = position;
        t->mask_ready = true;
        t->mask_rows = 1;  // GrammarSpec GS-01: a round with no drafts
        t->pick.reset();   // and no policy outcome
        return CERE_OK;
    });
}

static_assert(CERE_MAX_CONSTRAINED_DRAFTS == cere::host::DraftPolicy::kMaxDrafts,
              "a constrained (sampled) ticket's policy picks at most kMaxDrafts drafts (choose_drafts)");

// GrammarSpec GS-01 (header: speculation rounds of a constrained ticket). The pre-staging toy lacks both, as a library
// from before them does (cered's refusal test of grammar_drafts = "on").
#ifndef CERE_TOY_PRE_STAGING
cere_status cere_executor_round_drafts(cere_executor *executor, cere_completion_ticket ticket, uint64_t position,
                                       uint32_t *drafts, uint32_t capacity, uint32_t *count) {
    return guarded([&]() -> cere_status {
        if (count == nullptr) return CERE_ERR_ARGUMENT;
        *count = 0;
        if (executor == nullptr || (drafts == nullptr && capacity != 0)) return CERE_ERR_ARGUMENT;
        Ticket *t = nullptr;
        if (const cere_status st = find_ticket(executor, ticket, &t); st != CERE_OK) return st;
        // GrammarSpec GS-01: a constrained ticket waiting for the mask of the position it commits next.
        if (!t->constrained() || t->verify || position != t->committed.load(std::memory_order_relaxed) ||
            t->mask_present()) {
            return CERE_ERR_ARGUMENT;
        }
        if (!t->offer_frozen || t->offer_position != position) {
            cere_session &session = *t->session;
            t->pick.reset();
            uint32_t n = 0;
            const bool runs = t->finish_pending == CERE_FINISH_NONE && !t->cancel_requested &&
                              t->error_deferred == CERE_OK && position < t->max_new_tokens &&
                              session.tokens.size() < session.max_context_tokens;
            if (runs && session.drafting && !t->no_speculation && !t->ignore_stop_tokens &&
                session.fed < session.tokens.size() && session.fed >= session.prompt_tokens &&
                t->mask.size() > t->mask_words) {
                const uint64_t room =
                    std::min(t->max_new_tokens - position, session.max_context_tokens - session.tokens.size());
                const uint64_t chain = round_chain(session, room - 1);
                const uint32_t *ids = nullptr;
                n = std::min(choose_drafts(*executor->device, *t, 1, chain, &ids), CERE_MAX_CONSTRAINED_DRAFTS);
                if (ids == nullptr) {
                    n = executor->device->pending_drafts(*session.device, t->offer.data(), n);
                } else {
                    std::copy(ids, ids + n, t->offer.begin());
                }
            }
            t->offer_frozen = true;
            t->offer_position = position;
            t->offer_count = n;
        }
        const uint32_t written = std::min(t->offer_count, capacity);
        std::copy(t->offer.begin(), t->offer.begin() + written, drafts);
        *count = t->offer_count;
        return CERE_OK;
    });
}

cere_status cere_executor_constrain_drafts(cere_executor *executor, cere_completion_ticket ticket, uint64_t position,
                                           const uint32_t *drafts, uint32_t draft_count, const uint32_t *vocab_masks,
                                           uint64_t vocab_mask_words) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr || (drafts == nullptr && draft_count != 0)) return CERE_ERR_ARGUMENT;
        Ticket *t = nullptr;
        if (const cere_status st = find_ticket(executor, ticket, &t); st != CERE_OK) return st;
        // GrammarSpec GS-01: a prefix of the drafts round_drafts froze for this position, one mask per verify row.
        if (!t->constrained() || position != t->committed.load(std::memory_order_relaxed) || !t->offer_frozen ||
            t->offer_position != position || draft_count > t->offer_count ||
            !std::equal(drafts, drafts + draft_count, t->offer.begin())) {
            return CERE_ERR_ARGUMENT;
        }
        const uint32_t vocab = t->session->model->plan.config.vocab_size;
        if (vocab_masks == nullptr) return CERE_ERR_ARGUMENT;
        for (uint32_t j = 0; j <= draft_count; ++j) {
            if (const cere_status st = check_mask(vocab_masks + j * vocab_mask_words, vocab_mask_words, vocab);
                st != CERE_OK) {
                return st;
            }
        }
        std::copy(vocab_masks, vocab_masks + (uint64_t(draft_count) + 1) * vocab_mask_words, t->mask.begin());
        t->mask_position = position;
        t->mask_ready = true;
        t->mask_rows = draft_count + 1;  // the frozen pick stays for the round's policy outcome
        return CERE_OK;
    });
}
#endif

cere_status cere_session_snapshot(cere_executor *executor, cere_session *session, uint64_t expected_session_revision,
                                  cere_snapshot **out) {
    return guarded([&]() -> cere_status {
        if (out == nullptr) return CERE_ERR_ARGUMENT;
        *out = nullptr;
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        cere_session *s = find_session(executor, session);
        if (s == nullptr) return CERE_ERR_ARGUMENT;
        cere_model &model = *s->model;
        if (const cere_status st = refuse_faulted(executor, &model); st != CERE_OK) return st;
        if (model.plan.snapshot_host_bytes == 0) return CERE_ERR_UNSUPPORTED;
        if (s->in_flight != 0) return CERE_ERR_BUSY;
        if (expected_session_revision != s->revision) return CERE_ERR_STALE_EPOCH;
        // Nothing fed yet: no state to save (D14 refuses position 0). A prompt still open (a chunked PREFILL cancelled
        // between chunks) has no image its own validate accepts (M211R-07; CudaDevice::save_state refuses it too).
        if (s->fed == 0 || s->fed < s->prompt_tokens) return CERE_ERR_ARGUMENT;
        namespace snap = cere::host::snapshot;
        // The session's kinds: 0-7, 10 (draft state) for a drafting session (M2.9, M29S2-07) and 11 (rope_delta) for
        // an image session (ledger CR14). Writer rule: a session whose state includes a kind this build cannot write
        // is refused instead of snapshotted partially; a device that cannot save kind 10 refuses it as UNSUPPORTED.
        const uint32_t kinds = snap::kRequiredKinds | (s->drafting ? uint32_t(1) << snap::kDraft : 0) |
                               (s->image_session ? uint32_t(1) << snap::kRopeDelta : 0);
        if ((kinds & ~writable_kinds(model)) != 0) return CERE_ERR_UNSUPPORTED;
        // Track LoRA: a snapshot carries no adapter, so a session with one is not snapshotted.
        if (s->lora_slot >= 0) return CERE_ERR_UNSUPPORTED;
        // WeightLab: neither is a session with a research plan.
        if (s->research != nullptr) return CERE_ERR_UNSUPPORTED;
        snap::Header header;
        header.layout_id = model.layout_id;
        header.position = s->fed;
        header.token_count = s->tokens.size();
        header.fed = s->fed;
        header.flags = s->flags;
        snap::Plan plan;
        if (!snap::plan(model.plan.config, header.position, 0, header.token_count, &plan, kinds)) {
            return CERE_ERR_CAPACITY;
        }
        header.total_bytes = plan.total_bytes;
        return add_snapshot(
            executor, model, header, plan.total_bytes,
            [&](uint8_t *dst) -> cere_status {
                snap::write_header(dst, header, plan);
                uint8_t *tokens = dst + plan.sections[0].offset;
                for (uint64_t i = 0; i < header.token_count; ++i) {
                    for (int b = 0; b < 4; ++b) tokens[4 * i + b] = uint8_t(s->tokens[i] >> (8 * b));
                }
                snap::write_rope_delta(dst, plan, s->rope_delta);
                const cere_status st =
                    executor->device->save_state(*model.device, *s->device, plan, header.position, dst);
                // Kind 10's writer rule (snapshot_format.h): count ids, the unused ones zero. A device copies its
                // proposal buffer as is, and a draft cut short by the end of the context (fewer than 3) leaves an
                // earlier round's ids after its count, which validate refuses (SE-09: a grow parks exactly there).
                uint32_t count = 0;
                uint32_t ids[3] = {};
                if (st == CERE_OK && snap::read_proposals(dst, plan, &count, ids) && count <= 3) {
                    snap::write_proposals(dst, plan, count, ids);
                }
                return st;
            },
            out);
    });
}

cere_status cere_session_restore(cere_executor *executor, cere_session *session, const cere_snapshot *snapshot) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        cere_session *s = find_session(executor, session);
        const cere_snapshot *snap = find_snapshot(executor, snapshot);
        if (s == nullptr || snap == nullptr) return CERE_ERR_ARGUMENT;
        // Restore needs a fresh session of the snapshot's model (D6); a snapshot exists only in a model with an arena.
        if (snap->model != s->model || s->submitted || s->in_flight != 0 || !s->tokens.empty()) {
            return CERE_ERR_ARGUMENT;
        }
        if (s->lora_slot >= 0) return CERE_ERR_UNSUPPORTED;  // Track LoRA: a snapshot's state is the base model's
        if (s->research != nullptr) return CERE_ERR_UNSUPPORTED;  // WeightLab: its rows below the image never ran
        cere_model &model = *s->model;
        if (const cere_status st = refuse_faulted(executor, &model); st != CERE_OK) return st;
        const uint8_t *bytes = model.device->arena + snap->offset;
        cere::host::snapshot::Header header;
        cere::host::snapshot::Plan plan;
        if (const cere_status st = cere::host::snapshot::validate(
                bytes, snap->info.bytes, snapshot_expect(model, s->max_context_tokens), &header, &plan);
            st != CERE_OK) {
            return st;
        }
        // H1: a session snapshot holds the next token to feed (token_count == fed + 1); a prefix-cache image
        // (token_count == fed) has none, so a session restored from it could never decode.
        if (header.token_count != header.fed + 1) return CERE_ERR_ARGUMENT;
        // An image session (kind 11) restores its rope_delta (M2.13-L: any context; 0 for a text session); it has no
        // image keys, so it never uses the prefix cache.
        int64_t rope_delta = 0;
        const bool image_session = cere::host::snapshot::read_rope_delta(bytes, plan, &rope_delta);
        // M2.10 D3′: a lazy session binds its own pages first (no prompt, so no lookup); CAPACITY when they cannot be.
        cere::host::PrefixAttach attached;
        if (const cere_status st = cere::host::attach_prefix(*executor, *s, executor->last_submission, &attached);
            st != CERE_OK) {
            return st;
        }
        // M2.9 (M29S2-07): an image with kind 10 is a drafting session's; the session drafts before its state loads.
        const bool drafting = std::any_of(plan.sections.begin(), plan.sections.end(),
                                          [](const auto &sec) { return sec.kind == cere::host::snapshot::kDraft; });
        // The tokens go in first (the one step that can throw: the buffer was reserved at create, so it does not in
        // practice), and one rollback undoes everything a failure below has changed (H6).
        const uint8_t *tokens = bytes + plan.sections[0].offset;
        s->tokens.resize(header.token_count);
        for (uint64_t i = 0; i < header.token_count; ++i) {
            s->tokens[i] = uint32_t(tokens[4 * i]) | uint32_t(tokens[4 * i + 1]) << 8 |
                           uint32_t(tokens[4 * i + 2]) << 16 | uint32_t(tokens[4 * i + 3]) << 24;
        }
        const auto rollback = [&] {
            s->tokens.clear();
            if (drafting) executor->device->set_session_draft(*model.device, *s->device, false);
        };
        if (drafting) {
            if (const cere_status st = executor->device->set_session_draft(*model.device, *s->device, true);
                st != CERE_OK) {
                rollback();
                return st;
            }
        }
        if (const cere_status st = executor->device->load_state(*model.device, *s->device, plan, header.position, bytes);
            st != CERE_OK) {
            rollback();
            return st;
        }
        if (const cere_status st = executor->device->set_rope_delta(*model.device, *s->device, rope_delta);
            st != CERE_OK) {
            rollback();
            return st;
        }
        s->fed = header.fed;
        s->flags = header.flags;
        s->rope_delta = rope_delta;
        s->image_session = image_session;
        s->drafting = drafting;
        s->submitted = true;
        return CERE_OK;
    });
}

cere_status cere_snapshot_get_info(cere_executor *executor, const cere_snapshot *snapshot, cere_snapshot_info *out) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        if (const cere_status st = check_header(out); st != CERE_OK) return st;
        clear_body(out);
        const cere_snapshot *snap = find_snapshot(executor, snapshot);
        if (snap == nullptr) return CERE_ERR_ARGUMENT;
        out->bytes = snap->info.bytes;
        out->position = snap->info.position;
        out->token_count = snap->info.token_count;
        out->fed = snap->info.fed;
        std::memcpy(out->layout_id, snap->info.layout_id, sizeof(out->layout_id));
        return CERE_OK;
    });
}

cere_status cere_snapshot_read(cere_executor *executor, const cere_snapshot *snapshot, uint64_t offset, void *dst,
                               uint64_t len) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        const cere_snapshot *snap = find_snapshot(executor, snapshot);
        if (snap == nullptr || dst == nullptr || len == 0) return CERE_ERR_ARGUMENT;
        if (offset > snap->info.bytes || len > snap->info.bytes - offset) return CERE_ERR_ARGUMENT;
        std::memcpy(dst, snap->model->device->arena + snap->offset + offset, len);
        return CERE_OK;
    });
}

cere_status cere_snapshot_import(cere_executor *executor, cere_model *model, const void *bytes, uint64_t len,
                                 cere_snapshot **out) {
    return guarded([&]() -> cere_status {
        if (out == nullptr) return CERE_ERR_ARGUMENT;
        *out = nullptr;
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        cere_model *m = find_model(executor, model);
        if (m == nullptr || !m->sealed.load()) return CERE_ERR_ARGUMENT;
        if (m->plan.snapshot_host_bytes == 0) return CERE_ERR_UNSUPPORTED;
        if (bytes == nullptr) return CERE_ERR_ARGUMENT;
        const auto *src = static_cast<const uint8_t *>(bytes);
        cere::host::snapshot::Header header;
        cere::host::snapshot::Plan plan;
        if (const cere_status st = cere::host::snapshot::validate(src, len, snapshot_expect(*m, 0), &header, &plan);
            st != CERE_OK) {
            return st;
        }
        return add_snapshot(
            executor, *m, header, len,
            [&](uint8_t *dst) -> cere_status {
                std::memcpy(dst, src, len);
                return CERE_OK;
            },
            out);
    });
}

cere_status cere_snapshot_release(cere_executor *executor, cere_snapshot *snapshot) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        auto it = executor->snapshots.find(snapshot);
        if (it == executor->snapshots.end()) return CERE_ERR_ARGUMENT;
        it->second->model->arena.free(it->second->offset);
        --it->second->model->snapshots;
        executor->snapshots.erase(it);
        return CERE_OK;
    });
}

cere_status cere_session_set_cache_namespace(cere_executor *executor, cere_session *session,
                                             const uint8_t namespace_id[16]) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        cere_session *s = find_session(executor, session);
        if (s == nullptr) return CERE_ERR_ARGUMENT;
        if (s->model->prefix == nullptr) return CERE_ERR_UNSUPPORTED;
        // D2: before the session's first submit (a restore counts as one), never after.
        if (namespace_id == nullptr || s->submitted) return CERE_ERR_ARGUMENT;
        s->ns = cache_namespace(namespace_id);
        s->namespaced = true;
        return CERE_OK;
    });
}

cere_status cere_model_flush_prefix_cache(cere_executor *executor, cere_model *model) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        cere_model *m = find_model(executor, model);
        if (m == nullptr) return CERE_ERR_ARGUMENT;
        if (m->prefix == nullptr) return CERE_ERR_UNSUPPORTED;
        // D11: idle only (no ticket of the model in flight, no session holding a hit lock).
        if (model_busy(executor, *m) || m->prefix->locked()) return CERE_ERR_BUSY;
        m->prefix->flush();
        return CERE_OK;
    });
}

cere_status cere_prefix_cache_export(cere_executor *executor, cere_model *model, const uint8_t namespace_id[16],
                                     const uint32_t *tokens, uint64_t token_count, cere_snapshot **out) {
    return guarded([&]() -> cere_status {
        namespace snap = cere::host::snapshot;
        if (out == nullptr) return CERE_ERR_ARGUMENT;
        *out = nullptr;
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        cere_model *m = find_model(executor, model);
        if (m == nullptr) return CERE_ERR_ARGUMENT;
        if (m->prefix == nullptr) return CERE_ERR_UNSUPPORTED;
        if (const cere_status st = refuse_faulted(executor, m); st != CERE_OK) return st;
        if (namespace_id == nullptr || tokens == nullptr || token_count == 0 ||
            !tokens_valid(tokens, token_count, m->plan.config.vocab_size)) {
            return CERE_ERR_ARGUMENT;
        }
        // The image is an M2.11 snapshot in the model's arena: none without one (as every snapshot call).
        if (m->plan.snapshot_host_bytes == 0) return CERE_ERR_UNSUPPORTED;
        if (model_busy(executor, *m)) return CERE_ERR_BUSY;  // D12, D16
        cere::host::PrefixCache &cache = *m->prefix;
        const cere::host::CacheNamespace ns = cache_namespace(namespace_id);
        const uint64_t cap = token_count / cere::host::kKvPageTokens * cere::host::kKvPageTokens;
        cere::host::PrefixCache::Match hit = cache.match(ns, tokens, token_count, cap, false);
        if (hit.node == nullptr) return CERE_ERR_ARGUMENT;  // no cached prefix
        // A host node (D9) is loaded back so the whole path is on the device; else the deepest device state.
        // H4: the path lock is released on every exit (an exception included).
        PathLock held(cache);
        held.hold(hit.node);
        if (hit.node->host && cache.reload(hit.node) != CERE_OK) {
            held.release();
            hit = cache.match(ns, tokens, token_count, cap, true);
            if (hit.node == nullptr) return CERE_ERR_ARGUMENT;
            held.hold(hit.node);
        }
        cere::host::PrefixNode *node = hit.node;
        snap::Header header;
        header.layout_id = m->layout_id;
        header.position = hit.h;
        header.token_count = hit.h;
        header.fed = hit.h;
        // H2: a drafting model's cached states carry no MTP K/V rows (an image's draft state is never captured
        // here), so the image is flagged NO_SPECULATION: its own validate (a drafting model refuses an image
        // without the flag) and the import that follows accept it, and the imported node stays draft-less.
        if (m->plan.speculative_draft_tokens != 0) header.flags = CERE_WORK_NO_SPECULATION;
        snap::Plan plan;
        if (!snap::plan(m->plan.config, hit.h, 0, hit.h, &plan, snap::kRequiredKinds)) return CERE_ERR_CAPACITY;
        header.total_bytes = plan.total_bytes;
        const std::vector<uint32_t> pages = cache.path_pages(node);
        const cere_status st = add_snapshot(
            executor, *m, header, plan.total_bytes,
            [&](uint8_t *dst) -> cere_status {
                snap::write_header(dst, header, plan);
                uint8_t *ids = dst + plan.sections[0].offset;
                for (uint64_t i = 0; i < hit.h; ++i) {
                    for (int b = 0; b < 4; ++b) ids[4 * i + b] = uint8_t(tokens[i] >> (8 * b));
                }
                return executor->device->save_prefix(*m->device, pages.data(), pages.size(), node->slot, plan, hit.h,
                                                     dst);
            },
            out);
        cache.touch(node, executor->last_submission);
        return st;
    });
}

cere_status cere_prefix_cache_import(cere_executor *executor, cere_model *model, const uint8_t namespace_id[16],
                                     const cere_snapshot *snapshot) {
    return guarded([&]() -> cere_status {
        namespace snap = cere::host::snapshot;
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        cere_model *m = find_model(executor, model);
        if (m == nullptr) return CERE_ERR_ARGUMENT;
        if (m->prefix == nullptr) return CERE_ERR_UNSUPPORTED;
        if (const cere_status st = refuse_faulted(executor, m); st != CERE_OK) return st;
        const cere_snapshot *s = find_snapshot(executor, snapshot);
        if (namespace_id == nullptr || s == nullptr || s->model != m) return CERE_ERR_ARGUMENT;
        if (model_busy(executor, *m)) return CERE_ERR_BUSY;  // D12, D16
        // An exported prefix: first_row 0 (validate refuses others), no pending token (token_count == position), a
        // positive multiple of 64, no image state (image sessions never insert, D15′ (b)).
        const uint8_t *bytes = m->device->arena + s->offset;
        snap::Header header;
        snap::Plan plan;
        if (const cere_status st = snap::validate(bytes, s->info.bytes, snapshot_expect(*m, 0), &header, &plan);
            st != CERE_OK) {
            return st;
        }
        int64_t rope_delta = 0;
        if (header.token_count != header.position || header.position % cere::host::kKvPageTokens != 0 ||
            snap::read_rope_delta(bytes, plan, &rope_delta)) {
            return CERE_ERR_ARGUMENT;
        }
        const uint64_t len = header.position;
        std::vector<uint32_t> ids(len);
        const uint8_t *src = bytes + plan.sections[0].offset;
        for (uint64_t i = 0; i < len; ++i) {
            ids[i] = uint32_t(src[4 * i]) | uint32_t(src[4 * i + 1]) << 8 | uint32_t(src[4 * i + 2]) << 16 |
                     uint32_t(src[4 * i + 3]) << 24;
        }
        // The prefix loads into pages of its own and a state slot; the tree then retains the edges it lacks and
        // the import's own references are dropped (pages of edges the tree already had are freed again). Its fresh
        // pages never equal a device path the namespace already holds over these tokens, so such an import stores
        // nothing (OK, counted, as an import onto an existing state): no state over foreign KV rows (M210S-39).
        cere::host::PrefixCache &cache = *m->prefix;
        const uint64_t n = cere::host::kv_pages(len);
        uint32_t slot = 0;
        if (!cache.take_slot(&slot)) return CERE_ERR_CAPACITY;
        std::vector<uint32_t> pages;
        cere_status st = cache.make_free(n) ? executor->device->take_pages(*m->device, n, &pages) : CERE_ERR_CAPACITY;
        if (st == CERE_OK && !cache.own_path(cache_namespace(namespace_id), ids.data(), len, pages.data())) {
            ++cache.stats().capture_page_mismatch;
            cache.free_slot(slot);
            if (const cere_status rel = executor->device->release_pages(*m->device, pages.data(), pages.size());
                rel != CERE_OK) {
                return rel;
            }
            return CERE_OK;
        }
        if (st == CERE_OK) {
            st = executor->device->load_prefix(*m->device, pages.data(), pages.size(), slot, plan, len, bytes);
        }
        if (st == CERE_OK) {
            if (!cache.insert(cache_namespace(namespace_id), ids.data(), len, pages.data(), slot,
                              cere::host::CaptureKind::kImport, executor->last_submission)) {
                cache.free_slot(slot);
            }
        } else {
            cache.free_slot(slot);
        }
        if (!pages.empty()) {
            if (const cere_status rel = executor->device->release_pages(*m->device, pages.data(), pages.size());
                rel != CERE_OK) {
                return rel;  // DM-3: the pool refused the release: a corrupted accounting is never reported as OK
            }
        }
        return st;
    });
}

// Track LoRA (lora design note; header: before seal / sealed, serialized). The pre-staging toy lacks these too, as a
// library from before them does (the runtime's refusal test).
#ifndef CERE_TOY_PRE_STAGING
cere_status cere_model_lora_reserve(cere_executor *executor, cere_model *model, uint32_t max_adapters,
                                    uint32_t max_rank) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr || max_adapters == 0 || max_adapters > CERE_LORA_MAX_ADAPTERS || max_rank == 0 ||
            max_rank > CERE_LORA_MAX_RANK) {
            return CERE_ERR_ARGUMENT;
        }
#if CERE_GEN_FAMILY == CERE_FAMILY_SDAR || CERE_GEN_FAMILY == CERE_FAMILY_INKLING2
        // SelfTune ST4: an SDAR library has no LoRA, refused before any model is looked up (Inkling 2 Mini too:
        // training and LoRA are out of scope, inkling design note section 14).
        return refused(CERE_ERR_UNSUPPORTED, "lora");
#endif
        std::unique_lock<std::shared_mutex> lock(executor->handles);
        cere_model *m = find_model(executor, model);
        if (m == nullptr || m->sealed.load() || m->lora_adapters != 0) return CERE_ERR_ARGUMENT;
        const uint64_t bytes = executor->device->lora_bytes(*m->device, max_adapters, max_rank);
        if (bytes == 0) return CERE_ERR_UNSUPPORTED;
        if (!fits(bytes, executor->memory_budget_bytes, executor->reserved_device_bytes)) return CERE_ERR_CAPACITY;
        if (const cere_status st = executor->device->lora_reserve(*m->device, max_adapters, max_rank);
            st != CERE_OK) {
            return st;
        }
        executor->reserved_device_bytes += bytes;
        m->lora_adapters = max_adapters;
        m->lora_rank = max_rank;
        m->lora_bytes = bytes;
        m->lora_loaded.assign(max_adapters, false);
        m->lora_sessions.assign(max_adapters, 0);
        return CERE_OK;
    });
}

cere_status cere_model_lora_load(cere_executor *executor, cere_model *model, uint32_t slot, uint32_t rank, float scale,
                                 uint64_t count, const uint32_t *targets, const void *const *a, const void *const *b) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        cere_model *m = find_model(executor, model);
        if (m == nullptr || !m->sealed.load()) return CERE_ERR_ARGUMENT;
        if (m->lora_adapters == 0) return CERE_ERR_UNSUPPORTED;
        if (slot >= m->lora_adapters || m->lora_loaded[slot]) return CERE_ERR_ARGUMENT;
        if (const cere_status st =
                cere::host::lora::check_load(m->plan.config, m->lora_rank, rank, scale, count, targets, a, b);
            st != CERE_OK) {
            return st;
        }
        if (const cere_status st = executor->device->lora_load(*m->device, slot, rank, scale, count, targets, a, b);
            st != CERE_OK) {
            return st;
        }
        m->lora_loaded[slot] = true;
        return CERE_OK;
    });
}

cere_status cere_model_lora_unload(cere_executor *executor, cere_model *model, uint32_t slot) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        cere_model *m = find_model(executor, model);
        if (m == nullptr) return CERE_ERR_ARGUMENT;
        if (m->lora_adapters == 0) return CERE_ERR_UNSUPPORTED;
        if (slot >= m->lora_adapters || !m->lora_loaded[slot]) return CERE_ERR_ARGUMENT;
        if (m->lora_sessions[slot] != 0) return CERE_ERR_BUSY;
        if (const cere_status st = executor->device->lora_unload(*m->device, slot); st != CERE_OK) return st;
        m->lora_loaded[slot] = false;
        return CERE_OK;
    });
}

cere_status cere_session_set_lora(cere_executor *executor, cere_session *session, uint32_t slot) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        cere_session *s = find_session(executor, session);
        if (s == nullptr) return CERE_ERR_ARGUMENT;
        cere_model &m = *s->model;
        if (m.lora_adapters == 0) return CERE_ERR_UNSUPPORTED;
        // Before the session's first submit (a restore counts as one), as cere_session_set_cache_namespace.
        if (s->submitted || slot >= m.lora_adapters || !m.lora_loaded[slot]) return CERE_ERR_ARGUMENT;
        if (const cere_status st = executor->device->set_session_lora(*m.device, *s->device, int32_t(slot));
            st != CERE_OK) {
            return st;
        }
        if (s->lora_slot >= 0) --m.lora_sessions[size_t(s->lora_slot)];
        s->lora_slot = int32_t(slot);
        ++m.lora_sessions[slot];
        return CERE_OK;
    });
}
#endif

// WeightLab research (weightlab design note WL-02; header: reserve before seal, the rest on a sealed model). Plans
// are validated and rows selected by research/plan.h, captures laid out and read by research/capture.h, alike in the
// CUDA library and the toy. The pre-staging toy lacks these, as a library from before them does.
#ifndef CERE_TOY_PRE_STAGING
// The least device staging a reserve takes: one row of the widest point (native.logits, 496,640 B) and the logits
// top-k scratch fit (the deployment table's minimum, 1 MiB).
constexpr uint64_t kResearchMinStagingBytes = uint64_t(1) << 20;

cere_status cere_model_research_reserve(cere_executor *executor, cere_model *model, uint64_t device_staging_bytes,
                                        uint64_t host_buffer_bytes, uint32_t max_plans) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr || device_staging_bytes < kResearchMinStagingBytes || host_buffer_bytes == 0 ||
            max_plans == 0 || max_plans > CERE_RESEARCH_MAX_PLANS) {
            return CERE_ERR_ARGUMENT;
        }
#if CERE_GEN_FAMILY == CERE_FAMILY_SDAR || CERE_GEN_FAMILY == CERE_FAMILY_INKLING2
        // SelfTune ST4: an SDAR library has no research points, refused before any model is looked up (nor an
        // Inkling 2 Mini library).
        return refused(CERE_ERR_UNSUPPORTED, "research");
#endif
        std::unique_lock<std::shared_mutex> lock(executor->handles);
        cere_model *m = find_model(executor, model);
        if (m == nullptr || m->sealed.load() || m->research_plans != 0) return CERE_ERR_ARGUMENT;
        if (!executor->device->research_supported()) return CERE_ERR_UNSUPPORTED;
        const uint64_t plan_bytes = uint64_t(max_plans) * CERE_RESEARCH_MAX_PLAN_BYTES;
        if (device_staging_bytes > std::numeric_limits<uint64_t>::max() - plan_bytes) return CERE_ERR_CAPACITY;
        const uint64_t bytes = device_staging_bytes + plan_bytes;
        if (!fits(bytes, executor->memory_budget_bytes, executor->reserved_device_bytes)) return CERE_ERR_CAPACITY;
        uint8_t *ring = nullptr;
        if (const cere_status st = executor->device->research_reserve(*m->device, device_staging_bytes, plan_bytes,
                                                                      host_buffer_bytes, &ring);
            st != CERE_OK) {
            return st;
        }
        executor->reserved_device_bytes += bytes;
        m->research_plans = max_plans;
        m->research_bytes = bytes;
        m->research_ring = ring;
        m->research_ring_alloc = cere::host::ArenaAllocator(host_buffer_bytes);
        m->research_plan.assign(max_plans, nullptr);
        m->research_sessions.assign(max_plans, 0);
        return CERE_OK;
    });
}

cere_status cere_model_research_plan_load(cere_executor *executor, cere_model *model, uint32_t slot,
                                          const cere_research_plan *plan) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        cere_model *m = find_model(executor, model);
        if (m == nullptr || !m->sealed.load()) return CERE_ERR_ARGUMENT;
        if (m->research_plans == 0) return CERE_ERR_UNSUPPORTED;
        std::lock_guard<std::mutex> research_lock(m->research_mutex);
        if (slot >= m->research_plans || m->research_plan[slot] != nullptr) return CERE_ERR_ARGUMENT;
        auto loaded = std::make_shared<cere::research::Plan>();
        if (const cere_status st = cere::research::load_plan(plan, loaded.get()); st != CERE_OK) return st;
        if (const cere_status st = executor->device->research_plan_load(*m->device, slot, *loaded); st != CERE_OK) {
            return st;
        }
        m->research_plan[slot] = std::move(loaded);
        return CERE_OK;
    });
}

cere_status cere_model_research_plan_unload(cere_executor *executor, cere_model *model, uint32_t slot) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        cere_model *m = find_model(executor, model);
        if (m == nullptr) return CERE_ERR_ARGUMENT;
        if (m->research_plans == 0) return CERE_ERR_UNSUPPORTED;
        std::lock_guard<std::mutex> research_lock(m->research_mutex);
        if (slot >= m->research_plans || m->research_plan[slot] == nullptr) return CERE_ERR_ARGUMENT;
        if (m->research_sessions[slot] != 0) return CERE_ERR_BUSY;
        // Captures keep their own reference to the plan; the slot's device plan memory is reused by the next load.
        m->research_plan[slot] = nullptr;
        return CERE_OK;
    });
}

cere_status cere_session_set_research(cere_executor *executor, cere_session *session, uint32_t slot,
                                      uint64_t prompt_tokens, uint64_t max_tokens, uint64_t *out_capture) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr || out_capture == nullptr) return CERE_ERR_ARGUMENT;
        cere_session *s = find_session(executor, session);
        if (s == nullptr) return CERE_ERR_ARGUMENT;
        cere_model &m = *s->model;
        if (m.research_plans == 0) return CERE_ERR_UNSUPPORTED;
        std::lock_guard<std::mutex> research_lock(m.research_mutex);
        // Before the session's first submit (a restore counts as one), once.
        if (s->submitted || s->research != nullptr || slot >= m.research_plans || m.research_plan[slot] == nullptr ||
            prompt_tokens == 0) {
            return CERE_ERR_ARGUMENT;
        }
        auto binding = std::make_shared<cere::research::Binding>();
        if (const cere_status st = cere::research::bind(m.research_plan[slot], slot, prompt_tokens, max_tokens,
                                                        binding.get());
            st != CERE_OK) {
            return st;
        }
        binding->lookup_skipped = binding->plan->captures_prompt;
        if (binding->bytes != 0) {
            if (!m.research_ring_alloc.allocate(binding->bytes, &binding->ring_offset)) return CERE_ERR_CAPACITY;
            binding->data = m.research_ring + binding->ring_offset;
        }
        if (const cere_status st = executor->device->set_session_research(*m.device, *s->device, binding.get());
            st != CERE_OK) {
            if (binding->bytes != 0) m.research_ring_alloc.free(binding->ring_offset);
            return st;
        }
        if (binding->plan->captures) {
            binding->id = m.research_next_capture++;
            m.research_captures.emplace(binding->id, binding);
        }
        ++m.research_sessions[slot];
        *out_capture = binding->id;
        s->research = std::move(binding);
        return CERE_OK;
    });
}

cere_status cere_model_research_capture_read(cere_executor *executor, cere_model *model, uint64_t capture,
                                             cere_research_capture *out) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
        cere_model *m = find_model(executor, model);
        if (m == nullptr) return CERE_ERR_ARGUMENT;
        if (m->research_plans == 0) return CERE_ERR_UNSUPPORTED;
        std::lock_guard<std::mutex> research_lock(m->research_mutex);
        const auto it = m->research_captures.find(capture);
        if (it == m->research_captures.end()) return CERE_ERR_ARGUMENT;
        cere::research::Binding &binding = *it->second;
        if (binding.live) return CERE_ERR_BUSY;
        if (const cere_status st = executor->device->research_sync(*m->device); st != CERE_OK) return st;
        cere::research::read(binding, out);
        return CERE_OK;
    });
}

cere_status cere_model_research_capture_free(cere_executor *executor, cere_model *model, uint64_t capture) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr) return CERE_ERR_ARGUMENT;
        cere_model *m = find_model(executor, model);
        if (m == nullptr) return CERE_ERR_ARGUMENT;
        if (m->research_plans == 0) return CERE_ERR_UNSUPPORTED;
        std::lock_guard<std::mutex> research_lock(m->research_mutex);
        const auto it = m->research_captures.find(capture);
        if (it == m->research_captures.end()) return CERE_ERR_ARGUMENT;
        if (it->second->live) return CERE_ERR_BUSY;
        if (it->second->bytes != 0) m->research_ring_alloc.free(it->second->ring_offset);
        it->second->data = nullptr;
        m->research_captures.erase(it);
        return CERE_OK;
    });
}
#endif

// BC1 leftover (batch-concurrency design note "Prefill selection signature"): the SHA-256 of
// prefill_signature_text. Sealed models only, as cere_executor_capabilities.
cere_status cere_model_prefill_signature(cere_executor *executor, const cere_model *model, uint8_t out_sha256[32]) {
    return guarded([&]() -> cere_status {
        if (executor == nullptr || out_sha256 == nullptr) return CERE_ERR_ARGUMENT;
        const cere_model *m = find_model(executor, model);
        if (m == nullptr || !m->sealed.load()) return CERE_ERR_ARGUMENT;
        std::string text;
        if (!prefill_signature_text(*executor->device, *m->device, &text)) return CERE_ERR_UNSUPPORTED;
        const std::array<uint8_t, 32> digest = cere::host::sha256(text.data(), text.size());
        std::memcpy(out_sha256, digest.data(), digest.size());
        return CERE_OK;
    });
}

}  // extern "C"
