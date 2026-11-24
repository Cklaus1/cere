#include "toy/toy_device.h"

#include "host/kv_pool.h"
#include "host/lora.h"
#include "host/sampler_ref.h"
#include "research/capture.h"
#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include <unistd.h>

namespace cere::toy {

namespace {

constexpr uint64_t kUnlimited = std::numeric_limits<uint64_t>::max();
constexpr uint64_t kDefaultHoldMs = 30000;
constexpr uint64_t kDefaultPrefillChunk = 4096;  // m22c D4
// m210 D15⁗′: the toy's perturbation of a session state resumed from an end state the wrong way (ToyDevice::step).
constexpr uint64_t kEndPerturb = 0x9e3779b97f4a7c15ull;

cere_qwen4_exp_config make_fixture_config() {
    cere_qwen4_exp_config c{};
    c.struct_size = sizeof(c);
    c.abi_version = CERE_EXECUTOR_ABI_VERSION;
    c.vocab_size = 70000;
    c.hidden_size = 32;
    c.num_hidden_layers = 1;
    c.full_attention_interval = 4;
    c.mamba_ssm_dtype = CERE_DTYPE_F32;
    c.num_experts = 2;
    c.num_experts_per_tok = 1;
    c.moe_intermediate_size = 16;
    c.norm_topk_prob = 1;  // absent from config.json
    c.ple_layer_ids = uint64_t(1) << 2;
    c.ngram_size = 3;
    c.split_ngram_parts = 4;
    c.mtp_num_hidden_layers = 1;
    return c;
}

// Parses an unsigned decimal; false on anything else (sign, empty, trailing bytes, overflow).
bool parse_u64(const std::string &text, uint64_t *out) {
    if (text.empty() || text[0] < '0' || text[0] > '9') return false;
    errno = 0;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end != text.c_str() + text.size()) return false;
    *out = value;
    return true;
}

}  // namespace

class ToyModel final : public cere::host::DeviceModel {
public:
    ToyModel(ToyDevice &device, const cere::host::ModelPlan &plan)
        : pool(cere::host::kv_pages(plan.kv_pool_tokens)),
          slots(plan.prefix_state_slots > 0 ? plan.prefix_state_slots + plan.max_sessions : 0),
          kv_ids(plan.prefix_state_slots > 0 && plan.speculative_draft_tokens != 0
                     ? cere::host::kv_pages(plan.kv_pool_tokens) * cere::host::kKvPageTokens
                     : 0),
          mtp_next(kv_ids.size()),
          levers(plan.levers),
          device_(device),
          plan_(plan) {}
    ~ToyModel() override {
        for (size_t i = 0; i < buffers.size(); ++i) {
            const cere::host::TensorPlan &t = plan_.tensors[i];
            if (t.placement == CERE_PLACEMENT_DISCARD) continue;
            const bool shared = segment != nullptr && t.placement == CERE_PLACEMENT_HOST_PINNED;
            (t.placement == CERE_PLACEMENT_DEVICE ? device_.device_bytes_ : device_.host_bytes_)
                .fetch_sub(shared ? t.bytes : buffers[i].size());
            device_.allocations_.fetch_sub(1);
        }
        device_.host_bytes_.fetch_sub(arena_bytes.size() + research_ring.size());
        device_.device_bytes_.fetch_sub(lora_bytes + research_device_bytes);
    }
    ToyModel(const ToyModel &) = delete;
    ToyModel &operator=(const ToyModel &) = delete;

    std::vector<std::vector<uint8_t>> buffers;  // one per tensor; empty for DISCARD and shared HOST_PINNED tensors
    std::vector<uint8_t> arena_bytes;           // the M2.11 snapshot arena (plan.snapshot_host_bytes)
    // Sharing on: the HOST_PINNED tensors live in `segment` at region_offsets[i] (host/host_share.h).
    std::unique_ptr<cere::host::SharedSegment> segment;
    std::vector<uint64_t> region_offsets;
    // The paged KV pool (m25 design note D3; M2.10 refcounts, D7): the toy counts pages, it stores no KV bytes.
    mutable std::mutex pool_mutex;
    cere::host::PagePool pool;
    // M2.10 (m210 design note D6′): the fixed-state slots, prefix_state_slots + max_sessions (none with 0).
    std::vector<ToyState> slots;
    // D15⁗ (ledger CR8) on a model with drafts and a prefix cache only: per pool row (page * 64 + row), the id + 1 of
    // the token the row holds (the target's KV, written by every consumed token) and of the next id the row's MTP
    // K/V was extended with (written by a drafting session's prefill steps); 0 = unwritten. A drafting session whose
    // MTP rows disagree with the tokens it consumed (a prefix it did not draft over, or a next id that is not its
    // token) drafts wrongly at its scoring step (propose).
    std::vector<uint32_t> kv_ids;
    std::vector<uint32_t> mtp_next;
    cere::host::Levers levers;  // ABI v5: the desc's selection, then every set_levers (ToyDevice::levers reads it)
    // Track LoRA (lora design note): per adapter slot the offset a bound session adds to every scored id (0 = an
    // empty slot; ToyDevice::lora_load), and the pool's device bytes (counted in allocated_device_bytes).
    std::vector<uint32_t> lora_offsets;
    uint64_t lora_bytes = 0;
    // WeightLab research: the capture ring (cere_model_research_reserve's host bytes; counted in allocated_host_bytes)
    // and the staging + plan memory the CUDA device would hold (counted in allocated_device_bytes).
    std::vector<uint8_t> research_ring;
    uint64_t research_device_bytes = 0;
    // MixedPD (prefill_segments): the session whose sliced step is paused (null: none) and its pauses so far.
    const cere::host::DeviceSession *parked = nullptr;
    uint64_t pauses = 0;
    uint32_t vocab() const { return plan_.config.vocab_size; }
    uint64_t max_sessions() const { return plan_.max_sessions; }
    const cere::host::ModelPlan &plan() const { return plan_; }

private:
    ToyDevice &device_;
    cere::host::ModelPlan plan_;
};

namespace {

class ToySession final : public cere::host::DeviceSession {
public:
    ToySession(ToyModel &model, uint64_t max_context_tokens, bool lazy, bool stateful, uint32_t cycle)
        : model(model), max_context(max_context_tokens), lazy(lazy), stateful(stateful), cycle(cycle) {}
    ~ToySession() override {
        std::lock_guard<std::mutex> lock(model.pool_mutex);
        if (model.parked == this) {  // MixedPD: a paused step of a released session never resumes
            model.parked = nullptr;
            model.pauses = 0;
        }
        cere::host::release_pages_logged(model.pool, pages.data(), pages.size(), "toy_session_destroy");
    }
    ToySession(const ToySession &) = delete;
    ToySession &operator=(const ToySession &) = delete;

    ToyModel &model;              // the host destroys sessions before their model
    uint64_t max_context;
    std::vector<uint32_t> pages;  // reserved at create (ascending ids) or bound (M2.10), released by the destructor
    bool lazy = false;            // M2.10 D3′: created without pages; `bound` once bind_pages gave them
    bool bound = false;
    uint64_t id = 0;              // record= session number (1-based, creation order)
    // The fixed state (tokens consumed, the last one and their running digest), as a ToyState.
    uint64_t length = 0;
    uint32_t last = 0;
    uint64_t digest = kDigestSeed;
    uint64_t hash = 0;       // ",stateful=1": rolling hash of every consumed token (0 otherwise)
    uint64_t generated = 0;  // tokens scored
    std::vector<uint32_t> counts;  // D5 output counts of sampled steps (allocated at the first one)
    bool holding = false;
    bool hold_done = false;  // a snapshot was taken (or refused), or the hold_gate file was gone: holds no more
    bool broken = false;     // a failed forward call (fail_forward) left the state undefined
    std::chrono::steady_clock::time_point held_since;
    const bool stateful;
    const uint32_t cycle;  // ",cycle=P": the target before the hash is (last + 1) mod P (0 = next_token)
    // M2.9: a drafting session's pending MTP proposals (the toy's draft: the greedy continuation of its last scored
    // token, its draft_miss-th proposal off by one); count 0 before the first scored step.
    bool drafting = false;
    uint32_t proposal_count = 0;
    uint32_t proposals[3] = {};
    // m210 D15⁗′: the class of the end state load_slot gave the session (0 = none), checked by its next step.
    uint64_t resume_cls = 0;
    uint32_t lora_off = 0;  // Track LoRA: the bound adapter's offset (0 = none)
    // WeightLab research: the bound plan (null: none); every consumed token writes its capture rows.
    const cere::research::Binding *research = nullptr;
    // M2.13-L (Device::set_rope_delta): every row scored outside an open prompt is at position + rope_delta.
    int64_t rope_delta = 0;
};

// The rolling hash of ",stateful=1" after consuming `token`.
uint64_t fold(uint64_t hash, uint32_t token) { return (hash ^ (uint64_t(token) + 1)) * 1099511628211ull; }

// The toy target rule: next_token(last, position) (position = tokens consumed + rope_delta), or (last + 1) mod P
// under ",cycle=P" (a periodic output, M2.9L: suffix lookups match it), plus the rolling hash for a stateful session,
// mod vocab.
uint32_t target(uint32_t last, uint64_t position, uint64_t hash, bool stateful, uint32_t cycle, uint32_t vocab) {
    const uint32_t next = cycle != 0 ? (last + 1) % cycle : next_token(last, position, vocab);
    return stateful ? uint32_t((uint64_t(next) + hash % vocab) % vocab) : next;
}

// The pool row (ToyModel::kv_ids, mtp_next) of the session's position `p`.
uint64_t pool_row(const ToySession &toy, uint64_t p) {
    return uint64_t(toy.pages[p / cere::host::kKvPageTokens]) * cere::host::kKvPageTokens + p % cere::host::kKvPageTokens;
}

// M2.13-L: the toy encoder word of `image`, FNV-1a 64 of its grid (t, h, w as LE u32) then its pixel bytes.
uint64_t image_word(const cere::host::SessionImage &image) {
    uint64_t h = kDigestSeed;
    const auto fold_byte = [&h](uint8_t b) { h = (h ^ b) * 0x100000001b3ull; };
    for (const uint32_t v : {image.span.grid_t, image.span.grid_h, image.span.grid_w}) {
        for (int b = 0; b < 4; ++b) fold_byte(uint8_t(v >> (8 * b)));
    }
    for (const uint8_t b : image.pixels) fold_byte(b);
    return h;
}

// M2.13-L: the pad rows of one `step`'s images (null StepImages: none) and their toy encoder words, each computed once
// per step: after the token id of pad row j of an image the digest folds (word ^ j) as two u32, so the state depends
// on the image's pixels and grid as the real model's does.
class PadWords {
public:
    explicit PadWords(const cere::host::StepImages *images) : images_(images) {
        if (images != nullptr) words_.assign(images->count, std::nullopt);
    }
    uint64_t fold(uint64_t digest, uint64_t p) {
        for (uint64_t k = 0; k < words_.size(); ++k) {
            const cere::host::SessionImage &im = images_->image[k];
            if (p < im.span.token_offset || p - im.span.token_offset >= cere::mrope::image_tokens(im.span)) continue;
            if (!words_[k]) words_[k] = image_word(im);
            const uint64_t v = *words_[k] ^ (p - im.span.token_offset);
            return digest_step(digest_step(digest, uint32_t(v)), uint32_t(v >> 32));
        }
        return digest;
    }

private:
    const cere::host::StepImages *images_;
    std::vector<std::optional<uint64_t>> words_;
};

// Consumes `count` tokens (no scoring): the ToyState (length, last, M2.10 digest, with `pads` the image rows' words)
// and the stateful hash advance, and on a model with drafts and a prefix cache the tokens' pool rows (D15⁗): the
// token, and no MTP row (a drafting prefill step writes it next, extend_mtp).
void consume(ToySession &toy, const uint32_t *tokens, uint64_t count, PadWords *pads = nullptr) {
    if (toy.research != nullptr) {
        for (uint64_t i = 0; i < count; ++i) cere::research::write_toy_row(*toy.research, toy.length + i);
    }
    for (uint64_t i = 0; i < count && !toy.model.kv_ids.empty(); ++i) {
        const uint64_t row = pool_row(toy, toy.length + i);
        toy.model.kv_ids[row] = tokens[i] + 1;
        toy.model.mtp_next[row] = 0;
    }
    for (uint64_t i = 0; i < count; ++i) {
        toy.digest = digest_step(toy.digest, tokens[i]);
        if (pads != nullptr) toy.digest = pads->fold(toy.digest, toy.length + i);
    }
    toy.length += count;
    toy.last = tokens[count - 1];
    if (toy.stateful) {
        for (uint64_t i = 0; i < count; ++i) toy.hash = fold(toy.hash, tokens[i]);
    }
}

// D15⁗: a drafting prefill step's MTP extend over the `n` tokens it just consumed: row i takes tokens[i + 1] as its
// next id, the last row `next` (the next prompt token, or the scored token).
void extend_mtp(ToySession &toy, const uint32_t *tokens, uint64_t n, uint32_t next) {
    if (toy.model.mtp_next.empty()) return;
    const uint64_t p0 = toy.length - n;
    for (uint64_t i = 0; i < n; ++i) {
        toy.model.mtp_next[pool_row(toy, p0 + i)] = (i + 1 < n ? tokens[i + 1] : next) + 1;
    }
}

// D15⁗: whether the session's MTP rows over its consumed tokens are the ones its own drafting prefill would have
// written: row p's next id is the token at p + 1 for every p < length - 1 (true without the row record).
bool draft_intact(const ToySession &toy) {
    if (toy.model.mtp_next.empty()) return true;
    for (uint64_t p = 0; p + 1 < toy.length; ++p) {
        if (toy.model.mtp_next[pool_row(toy, p)] != toy.model.kv_ids[pool_row(toy, p + 1)]) return false;
    }
    return true;
}

// Before anything runs: CERE_PENDING while held (CERE_ERR_INTERNAL once held past hold_ms; a snapshotted session, a
// session that found the `hold_gate` file gone and, past the executor's first hold_count holds, a new hold are not
// held), CERE_ERR_INTERNAL for a broken session, CERE_ERR_CAPACITY when `count` tokens do not fit the context.
cere_status precheck(ToySession &toy, uint64_t count, uint64_t hold_after, uint64_t hold_ms, uint64_t hold_count,
                     const std::string &hold_gate, std::atomic<uint64_t> &holds) {
    if (!toy.hold_done && toy.generated >= hold_after && !hold_gate.empty() && ::access(hold_gate.c_str(), F_OK) != 0) {
        // The gate file is gone: the session runs on and holds no more.
        toy.hold_done = true;
        toy.holding = false;
    }
    if (!toy.hold_done && toy.generated >= hold_after && (toy.holding || holds.load() < hold_count)) {
        // Held: the step does not run (not counted as a step) until the host sees a cancel, or fails past hold_ms.
        const auto now = std::chrono::steady_clock::now();
        if (!toy.holding) {
            toy.holding = true;
            toy.held_since = now;
            holds.fetch_add(1);
        }
        const auto held = std::chrono::duration_cast<std::chrono::milliseconds>(now - toy.held_since).count();
        if (uint64_t(held) >= hold_ms) return CERE_ERR_INTERNAL;
        return CERE_PENDING;
    }
    if (toy.broken) return CERE_ERR_INTERNAL;
    if (count > toy.max_context - toy.length) return CERE_ERR_CAPACITY;
    return CERE_OK;
}

// Consumes `count` tokens and scores the next one at rotary position length + `rope_delta`. Under a vocab mask
// (m28 design note D9) the toy's own rule: a masked `next` becomes the smallest allowed id above it, wrapping (the
// mask allows some id).
void forward(ToySession &toy, const uint32_t *tokens, uint64_t count, uint32_t vocab, const uint32_t *mask,
             cere_token_score *out, int64_t rope_delta = 0, PadWords *pads = nullptr) {
    consume(toy, tokens, count, pads);
    ++toy.generated;
    uint32_t next = target(toy.last, uint64_t(int64_t(toy.length) + rope_delta), toy.hash, toy.stateful, toy.cycle,
                           vocab);
    next = uint32_t((uint64_t(next) + toy.lora_off) % vocab);  // Track LoRA: before the mask, as a logit change
    const auto allowed = [mask](uint32_t id) { return ((mask[id >> 5] >> (id & 31)) & 1u) != 0; };
    if (mask != nullptr) {
        for (uint32_t i = 0; i < vocab && !allowed(next); ++i) next = next + 1 == vocab ? 0 : next + 1;
    }
    *out = cere_token_score{next, (next + 1) % vocab, 1.0f, 0.0f};
}

// The item's vocab mask (null: none or no sampling); a speculation row's verify row j has the j-th mask after it
// (GrammarSpec GS-02, host/device.h SpecRowArgs).
const uint32_t *mask_of(const cere::host::ItemSampling *sampling, uint32_t row = 0, uint32_t vocab = 0) {
    if (sampling == nullptr || sampling->vocab_mask == nullptr) return nullptr;
    return sampling->vocab_mask + uint64_t(row) * ((uint64_t(vocab) + 31) / 32);
}

// The M2.7 sampler stage on the scored token: replaces *out (the greedy `next`) with the CPU sampler reference's
// choice over toy_logits(next), with the row's vocab mask (`mask`, null = none) applied after the penalties (M2.8
// D9), and fills *logprobs (when non-null). M2.9: with `draft` set (a verify row judging that draft) a sampled row
// follows D6 (ledger M8): the draft is kept with probability p(draft) (coin on channels (4, 5)), else the token is
// drawn from p with the draft removed (channels (6, 7)), both keyed by the row's position like the plain draw; a greedy
// row commits its plain choice. *accepted tells whether the committed token is the draft.
void sample(ToySession &toy, const cere::host::ItemSampling &sampling, const uint32_t *mask, uint32_t vocab,
            cere_token_score *out, cere_token_logprobs *logprobs, const uint32_t *draft = nullptr,
            bool *accepted = nullptr) {
    const uint32_t next = out->id;
    namespace sr = cere::sampler_ref;
    const std::vector<uint16_t> row = toy_logits(next, vocab);
    std::vector<float> processed(vocab);
    float temperature = 0.0f;
    if (sampling.sampled) {
        const cere_sampling_params &p = sampling.params;
        if (toy.counts.empty()) toy.counts.assign(vocab, 0);
        std::vector<sr::Bias> bias;
        for (const cere_logit_bias &b : sampling.bias) bias.push_back(sr::Bias{b.id, b.bias});
        sr::process(row.data(), vocab, toy.counts.data(), p.frequency_penalty, p.presence_penalty,
                    p.repetition_penalty, mask, bias.data(), bias.size(), processed.data());
        temperature = p.temperature;
    } else {
        sr::process(row.data(), vocab, nullptr, 0.0f, 0.0f, 1.0f, mask, nullptr, 0, processed.data());
    }
    const sr::Top2 top = sr::top2(processed.data(), vocab);
    uint32_t id = top.id;
    uint32_t runner_up = top.runner_up;
    if (temperature != 0.0f) {
        const cere_sampling_params &p = sampling.params;
        std::vector<float> probs(vocab);
        sr::probabilities(processed.data(), vocab, temperature, probs.data());
        std::vector<uint32_t> kept_ids = sr::keep_mask(probs.data(), vocab, p.top_k, p.top_p, p.min_p);
        // D4: plain draw on channels (0, 1), keyed by the position of the last consumed token.
        const uint32_t key = uint32_t(toy.length - 1);
        if (draft == nullptr) {
            id = sr::draw(probs.data(), vocab, kept_ids, sr::uniform(p.seed, key, 0, 1));
        } else {
            double kept = 0.0;  // S, the fp64 kept total of the draw
            for (uint32_t i = 0; i < vocab; ++i) kept += sr::kept(kept_ids, i) ? double(probs[i]) : 0.0;
            const uint32_t d = *draft;
            if (sr::kept(kept_ids, d) && sr::uniform(p.seed, key, 4, 5) * kept < double(probs[d])) {
                id = d;
            } else {
                kept_ids[d / 32] &= ~(uint32_t(1) << (d % 32));  // a kept draft was not the sole kept id
                id = sr::draw(probs.data(), vocab, kept_ids, sr::uniform(p.seed, key, 6, 7));
            }
        }
        runner_up = id == top.id ? top.runner_up : top.id;
    }
    if (accepted != nullptr) *accepted = draft != nullptr && id == *draft;
    *out = cere_token_score{id, runner_up, processed[id], processed[runner_up]};
    if (logprobs != nullptr) {
        const sr::Logprobs lp = sr::logprobs(processed.data(), vocab, temperature, id, sampling.top_logprobs);
        *logprobs = cere_token_logprobs{};
        logprobs->id = id;
        logprobs->logprob = lp.committed;
        logprobs->top_count = uint32_t(lp.top.size());
        for (size_t i = 0; i < lp.top.size(); ++i) logprobs->top[i] = cere_logprob_entry{lp.top[i].id, lp.top[i].logprob};
    }
    if (sampling.sampled) sr::count_commit(toy.counts.data(), id);
}

// A drafting session's pending MTP proposals after it scored `pending` (unfed): the target chain continuing it
// (greedy, so a greedy verify accepts every proposal; at the session's rotary positions, M2.13-L), the draft_miss-th
// one replaced by the next id. Without an `intact` draft state (D15⁗) every proposal is replaced so.
void propose(ToySession &toy, uint32_t pending, uint32_t vocab, uint32_t miss, bool intact = true) {
    uint64_t hash = toy.stateful ? fold(toy.hash, pending) : 0;
    uint32_t last = pending;
    for (uint32_t k = 0; k < 3; ++k) {
        const uint64_t position = uint64_t(int64_t(toy.length + 1 + k) + toy.rope_delta);
        const uint32_t next = target(last, position, hash, toy.stateful, toy.cycle, vocab);
        toy.proposals[k] = (k + 1 == miss || !intact) ? (next + 1) % vocab : next;
        if (toy.stateful) hash = fold(hash, next);
        last = next;
    }
    toy.proposal_count = 3;
}

// Little-endian u32 in a CERESNP1 image (D2).
uint32_t get_u32(const uint8_t *p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

// M2.9 ",stateful=1": the rolling hash of the `count` u32 tokens at `tokens` (a CERESNP1 kind 0 section).
uint64_t stateful_hash(const uint8_t *tokens, uint64_t count) {
    uint64_t hash = 0;
    for (uint64_t i = 0; i < count; ++i) hash = fold(hash, get_u32(tokens + 4 * i));
    return hash;
}

void put_u32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i));
}

// M2.10: the ToyState digest's place in an image, the first 8 bytes of kind 7 (null when the plan has no kind 7 of
// at least 8 bytes).
const cere::host::snapshot::Section *digest_section(const cere::host::snapshot::Plan &plan) {
    for (const cere::host::snapshot::Section &s : plan.sections) {
        if (s.kind == cere::host::snapshot::kPle && s.bytes >= sizeof(uint64_t)) return &s;
    }
    return nullptr;
}

// Zeroes the device sections (kinds outside kHostKinds) of the image `dst`, then writes `digest` (kind 7) and
// `counts` (kind 8; empty = zero counts). CERE_ERR_INTERNAL when the counts do not fill kind 8.
cere_status write_device_sections(const cere::host::snapshot::Plan &plan, uint64_t digest,
                                  const std::vector<uint32_t> &counts, uint8_t *dst) {
    namespace snap = cere::host::snapshot;
    for (const snap::Section &s : plan.sections) {
        if (((snap::kHostKinds >> s.kind) & 1) != 0) continue;
        std::memset(dst + s.offset, 0, s.bytes);
        if (s.kind == snap::kSampler && !counts.empty()) {
            if (s.bytes != counts.size() * sizeof(uint32_t)) return CERE_ERR_INTERNAL;
            for (size_t i = 0; i < counts.size(); ++i) put_u32(dst + s.offset + 4 * i, counts[i]);
        }
    }
    if (const snap::Section *s = digest_section(plan)) {
        for (size_t i = 0; i < sizeof(digest); ++i) dst[s->offset + i] = uint8_t(digest >> (8 * i));
    }
    return CERE_OK;
}

// The ToyState of a validated image at `position` (>= 1): the last of its kind 0 tokens [0, position) and the digest
// of kind 7, or of those tokens when the plan has no room for it.
ToyState read_state(const cere::host::snapshot::Plan &plan, uint64_t position, const uint8_t *src) {
    const uint8_t *tokens = src + plan.sections.at(0).offset;
    ToyState state;
    state.length = position;
    state.last = get_u32(tokens + 4 * (position - 1));
    state.valid = true;
    if (const cere::host::snapshot::Section *s = digest_section(plan)) {
        state.digest = 0;
        for (size_t i = 0; i < sizeof(state.digest); ++i) state.digest |= uint64_t(src[s->offset + i]) << (8 * i);
    } else {
        for (uint64_t i = 0; i < position; ++i) state.digest = digest_step(state.digest, get_u32(tokens + 4 * i));
    }
    return state;
}

}  // namespace

uint32_t next_token(uint32_t last, uint64_t position, uint32_t vocab) {
    return uint32_t((31 * uint64_t(last) + position) % vocab);
}

std::vector<uint16_t> toy_logits(uint32_t next, uint32_t vocab) {
    std::vector<uint16_t> row(vocab, 0xFF80);  // BF16 -inf
    for (uint32_t j = 0; j < kToySupport && j < vocab; ++j) {
        float value = float(kToySupport - j);
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        row[(uint64_t(next) + j) % vocab] = uint16_t(bits >> 16);  // small integers are exact in BF16
    }
    return row;
}

cere_qwen4_exp_config fixture_config() { return make_fixture_config(); }

const cere::host::Support &fixture_support() {
    static const cere::host::Support support{
        "toy_fixture_model_src",
        make_fixture_config(),
        16,      // max_sessions: the shipped native profile's 16 rows (m23 design note P8; 8 before)
        262144,  // max_context_tokens: the A4 bound (ctx262 design note CX-01; m25 D6 200,000), for cered's toy
                 // deployments
        1048576, // max_kv_pool_tokens: the A4 bound (CX-01; Headroom's 300,032 before)
        cere::host::dtype_bit(CERE_DTYPE_BF16) | cere::host::dtype_bit(CERE_DTYPE_F32) |
            cere::host::dtype_bit(CERE_DTYPE_F8_E4M3) | cere::host::dtype_bit(CERE_DTYPE_U8) |
            cere::host::dtype_bit(CERE_DTYPE_I64),
        nullptr,
    };
    return support;
}

const cere::host::Support &small_vocab_support() {
    static const cere::host::Support support = [] {
        cere::host::Support s = fixture_support();
        s.name = "toy_small_vocab";
        s.config.vocab_size = kSmallVocab;
        return s;
    }();
    return support;
}

ToyDevice::ToyDevice(uint64_t host_budget_bytes, uint64_t fail_forward, uint64_t hold_after, uint64_t hold_ms,
                     uint64_t hold_count, std::string hold_gate, uint64_t snapshot_capacity, std::string restore_gate,
                     bool sampler, std::string record, uint64_t prefill_chunk, cere::host::ShareConfig share,
                     DraftOptions draft, std::string staging_gate, std::string tensor_digests)
    : host_budget_bytes_(host_budget_bytes),
      fail_forward_(fail_forward),
      hold_after_(hold_after),
      hold_ms_(hold_ms),
      hold_count_(hold_count),
      hold_gate_(std::move(hold_gate)),
      snapshot_capacity_(snapshot_capacity),
      restore_gate_(std::move(restore_gate)),
      sampler_(sampler),
      record_(std::move(record)),
      prefill_chunk_(prefill_chunk),
      share_(std::move(share)),
      draft_(draft),
      staging_gate_(std::move(staging_gate)),
      tensor_digests_(std::move(tensor_digests)) {}

uint64_t ToyDevice::region(const cere::host::ModelPlan &plan, std::vector<uint64_t> *offsets,
                           std::string *name) const {
    constexpr uint64_t kAlign = 256;
    offsets->assign(plan.tensors.size(), UINT64_MAX);
    uint64_t end = 0;
    for (size_t i = 0; i < plan.tensors.size(); ++i) {
        if (plan.tensors[i].placement != CERE_PLACEMENT_HOST_PINNED) continue;
        uint64_t at = 0;
        if (__builtin_add_overflow(end, kAlign - 1, &at)) return UINT64_MAX;
        at = at / kAlign * kAlign;
        (*offsets)[i] = at;
        if (__builtin_add_overflow(at, plan.tensors[i].bytes, &end)) return UINT64_MAX;
    }
    if (end != 0) *name = cere::host::segment_name(share_, "toy-v1", cere::host::region_description(plan, *offsets, end));
    return end;
}

std::vector<const cere::host::Support *> ToyDevice::supported() const {
    return {&fixture_support(), &small_vocab_support(), &cere::host::a4_support(), &cere::host::a4_vision_support()};
}

uint64_t ToyDevice::host_budget_bytes() const { return host_budget_bytes_; }

cere::host::MemoryEstimate ToyDevice::estimate(const cere::host::ModelPlan &plan) const {
    cere::host::MemoryEstimate estimate{0, 0};
    for (const cere::host::TensorPlan &t : plan.tensors) {
        if (t.placement == CERE_PLACEMENT_DEVICE) estimate.device_bytes += t.bytes;
        if (t.placement == CERE_PLACEMENT_HOST_PINNED && !share_.enabled()) estimate.host_bytes += t.bytes;
    }
    if (share_.enabled()) {
        // The region counts only when this process will create it (an attacher maps pages that already exist).
        try {
            std::vector<uint64_t> offsets;
            std::string name;
            const uint64_t bytes = region(plan, &offsets, &name);
            if (bytes == UINT64_MAX) return {UINT64_MAX, UINT64_MAX};
            if (bytes != 0 && !cere::host::segment_published(share_, name)) estimate.host_bytes += bytes;
        } catch (const std::bad_alloc &) {
            return {UINT64_MAX, UINT64_MAX};  // never fits a budget
        }
    }
    estimate.host_bytes += plan.snapshot_host_bytes;
    return estimate;
}

cere_status ToyDevice::allocate(const cere::host::ModelPlan &plan, std::unique_ptr<cere::host::DeviceModel> *out) {
    if (creates_.fetch_add(1) < create_busy_) return CERE_ERR_BUSY;
    if (plan.speculative_draft_tokens != 0 && !draft_.draft) return CERE_ERR_UNSUPPORTED;
    try {
        auto model = std::make_unique<ToyModel>(*this, plan);
        if (share_.enabled()) {
            std::string name;
            const uint64_t bytes = region(plan, &model->region_offsets, &name);
            if (bytes == UINT64_MAX) return CERE_ERR_CAPACITY;
            if (bytes != 0) {
                std::string error;
                cere::host::RegionRanges fill;  // L9: the puts that complete a chunk hash it
                for (size_t i = 0; i < plan.tensors.size(); ++i) {
                    const uint64_t at = model->region_offsets[i];
                    if (at != UINT64_MAX && plan.tensors[i].bytes != 0) fill.emplace_back(at, plan.tensors[i].bytes);
                }
                if (const cere_status st =
                        cere::host::SharedSegment::open(share_, name, bytes, false, fill, &model->segment, &error);
                    st != CERE_OK) {
                    std::fprintf(stderr, "cere: %s\n", error.c_str());
                    return st;
                }
            }
        }
        model->buffers.reserve(plan.tensors.size());
        for (const cere::host::TensorPlan &t : plan.tensors) {
            if (t.placement == CERE_PLACEMENT_DISCARD) {
                model->buffers.emplace_back();
                continue;
            }
            if (model->segment != nullptr && t.placement == CERE_PLACEMENT_HOST_PINNED) {
                model->buffers.emplace_back();
            } else {
                model->buffers.emplace_back(t.bytes);
            }
            (t.placement == CERE_PLACEMENT_DEVICE ? device_bytes_ : host_bytes_).fetch_add(t.bytes);
            allocations_.fetch_add(1);
        }
        model->arena_bytes.assign(plan.snapshot_host_bytes, 0);
        host_bytes_.fetch_add(plan.snapshot_host_bytes);
        model->arena = model->arena_bytes.empty() ? nullptr : model->arena_bytes.data();
        *out = std::move(model);
        return CERE_OK;
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;  // the partial model's destructor returned what it took
    }
}

cere_status ToyDevice::write(cere::host::DeviceModel &model, uint64_t tensor_index, uint64_t offset,
                             const void *bytes, uint64_t len) {
    auto &toy = static_cast<ToyModel &>(model);
    if (toy.segment != nullptr && toy.region_offsets[tensor_index] != UINT64_MAX) {
        std::string error;
        const cere_status st = toy.segment->write(toy.region_offsets[tensor_index] + offset, bytes, len, &error);
        if (st != CERE_OK) std::fprintf(stderr, "cere: %s\n", error.c_str());
        return st;
    }
    std::memcpy(toy.buffers[tensor_index].data() + offset, bytes, len);
    return CERE_OK;
}

cere_status ToyDevice::register_staging(const uint8_t *, uint64_t) {
    staging_ranges_.fetch_add(1);
    return CERE_OK;
}

void ToyDevice::unregister_staging(const uint8_t *, uint64_t) { staging_ranges_.fetch_sub(1); }

cere_status ToyDevice::write_staged(cere::host::DeviceModel &model, uint64_t tensor_index, uint64_t offset,
                                    const void *bytes, uint64_t len) {
    if (!staging_gate_.empty()) {
        staged_holding_.fetch_add(1);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(hold_ms_);
        while (::access(staging_gate_.c_str(), F_OK) == 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        staged_holding_.fetch_sub(1);
    }
    const cere_status st = write(model, tensor_index, offset, bytes, len);
    if (st == CERE_OK) {
        staged_puts_.fetch_add(1);
        staged_bytes_.fetch_add(len);
    }
    return st;
}

cere_status ToyDevice::host_resident(const cere::host::DeviceModel &model, uint64_t tensor_index,
                                     const uint8_t **out) const {
    const auto &toy = static_cast<const ToyModel &>(model);
    if (toy.segment == nullptr || toy.segment->creating()) return CERE_ERR_UNSUPPORTED;
    const uint64_t at = toy.region_offsets.at(tensor_index);
    if (at == UINT64_MAX) return CERE_ERR_ARGUMENT;
    *out = toy.segment->data() + at;
    return CERE_OK;
}

cere_status ToyDevice::instantiate(cere::host::DeviceModel &model) {
    auto &toy = static_cast<ToyModel &>(model);
    if (!tensor_digests_.empty()) {
        std::FILE *file = std::fopen(tensor_digests_.c_str(), "w");
        if (file == nullptr) return CERE_ERR_INTERNAL;
        for (uint64_t i = 0; i < toy.plan().tensors.size(); ++i) {
            if (toy.plan().tensors[i].placement == CERE_PLACEMENT_DISCARD) continue;
            uint64_t h = 0xcbf29ce484222325ull;
            for (const uint8_t b : tensor(model, i)) h = (h ^ b) * 0x100000001b3ull;
            std::fprintf(file, "%llu %016llx\n", static_cast<unsigned long long>(i), static_cast<unsigned long long>(h));
        }
        if (std::fclose(file) != 0) return CERE_ERR_INTERNAL;
    }
    if (toy.segment == nullptr) return CERE_OK;
    // m216 design note L4c: an attach's content digest is checked before the model is sealed (the CUDA device
    // checks it in the background during UPLOAD and joins here); a creator publishes.
    std::string error;
    cere_status st = toy.segment->verify(&error);
    if (st == CERE_OK) st = toy.segment->publish(&error);
    if (st != CERE_OK) std::fprintf(stderr, "cere: %s\n", error.c_str());
    return st;
}

cere_status ToyDevice::set_levers(cere::host::DeviceModel &model, const cere::host::Levers &levers) {
    static_cast<ToyModel &>(model).levers = levers;
    return CERE_OK;
}

cere::host::Levers ToyDevice::levers(const cere::host::DeviceModel &model) const {
    return static_cast<const ToyModel &>(model).levers;
}

uint64_t ToyDevice::spec_rounds(uint32_t k2_gemm, uint32_t window) const {
    return k2_gemm < 2 && window < 11 ? spec_rounds_[k2_gemm][window].load() : 0;
}

bool ToyDevice::forward_supported() const {
#ifdef CERE_TOY_NO_FORWARD
    return false;
#else
    return true;
#endif
}

bool ToyDevice::sampler_supported() const { return sampler_ && forward_supported(); }

bool ToyDevice::draft_supported(const cere::host::DeviceModel &model) const {
    return draft_.draft && static_cast<const ToyModel &>(model).plan().speculative_draft_tokens != 0;
}

cere_status ToyDevice::create_session(cere::host::DeviceModel &model, uint64_t max_context_tokens, bool lazy,
                                      std::unique_ptr<cere::host::DeviceSession> *out) {
    auto &toy = static_cast<ToyModel &>(model);
    const uint64_t pages = cere::host::kv_pages(max_context_tokens);
    std::unique_ptr<ToySession> session;
    try {
        session = std::make_unique<ToySession>(toy, max_context_tokens, lazy, draft_.stateful, draft_.cycle);
        session->pages.reserve(pages);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    if (!lazy) {  // M2.10 D3′: a lazy session takes its pages at bind_pages
        cere_status st = CERE_OK;
        {
            std::lock_guard<std::mutex> lock(toy.pool_mutex);
            st = toy.pool.take_pages(pages, &session->pages);
        }
        if (st != CERE_OK) {
            // create_capacity_delay_ms: the refusal answers late, outside the pool's mutex.
            if (st == CERE_ERR_CAPACITY && create_capacity_delay_ms_ != 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(create_capacity_delay_ms_));
            }
            return st;
        }
        session->bound = true;
    }
    session->id = sessions_.fetch_add(1) + 1;
    *out = std::move(session);
    return CERE_OK;
}

bool ToyDevice::fails_forward() {
    const uint64_t step = steps_.fetch_add(1);
    return step == fail_forward_ || (fail_sticky_ && step > fail_forward_);
}

cere_status ToyDevice::step(cere::host::DeviceModel &model, cere::host::DeviceSession &session,
                            const uint32_t *tokens, uint64_t count, uint64_t limit,
                            const cere::host::StepImages *images, uint64_t *consumed,
                            const cere::host::ItemSampling *sampling, cere_token_score *out,
                            cere_token_logprobs *logprobs, const cere::host::EndStep *end,
                            cere::host::StepSlice *slice) {
    if (slice != nullptr) slice->paused = false;
    if (!forward_supported()) return CERE_ERR_UNSUPPORTED;
    if (count == 0 || tokens == nullptr || consumed == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    *consumed = 0;
    auto &toy = static_cast<ToySession &>(session);
    if (!toy.bound) return CERE_ERR_ARGUMENT;  // M2.10: a lazy session steps once bound
    // M2.10 D4′: `limit` (0 = none) ends the call at the host's capture point.
    const uint64_t n = std::min({count, prefill_chunk_, limit != 0 ? limit : count});
    // m210 D15⁗′ (host/device.h EndStep): malformed before anything runs. D15⁗″: a step may resume one pair and
    // capture into another (disjoint runs); a chunk-0 capture past position 0 only from a resumed K/V run.
    auto &toy_model = static_cast<ToyModel &>(model);
    const bool capture = end != nullptr && end->row > 0;
    if (end != nullptr) {
        const uint64_t out_end = uint64_t(end->slot) + 2 + end->kv_slots;
        const uint64_t in_end = uint64_t(end->in_slot) + 2 + end->in_kv_slots;
        if ((!capture && !end->resume) || (capture && out_end > toy_model.slots.size()) ||
            (end->resume && in_end > toy_model.slots.size()) ||
            (capture && end->resume && end->slot < in_end && end->in_slot < out_end) ||
            (end->cls != 0 && end->cls < n) ||
            (capture && (end->row >= n || end->row % cere::host::kKvPageTokens != 0)) ||
            (capture && end->kv_slots != 0 && toy.length != 0 && !(end->resume && end->in_kv_slots != 0))) {
            return CERE_ERR_ARGUMENT;
        }
    }
    // M2.13-L (host/device.h StepImages): images come with the open prompt's steps only.
    if (images != nullptr && toy.length + n > images->prompt_len) return CERE_ERR_ARGUMENT;
    // MixedPD: a paused step resumes (its first segment passed precheck and fail_forward); another session's sliced
    // step waits for it.
    const bool resuming = toy_model.parked == &toy;
    if (toy_model.parked != nullptr && !resuming && slice != nullptr) return CERE_ERR_ARGUMENT;
    if (!resuming) {
        const cere_status st = precheck(toy, n, hold_after_, hold_ms_, hold_count_, hold_gate_, holds_);
        if (st != CERE_OK) return st;
        if (fails_forward()) {
            toy.broken = true;
            return CERE_ERR_INTERNAL;
        }
    }
    if (slice != nullptr && slice->budget_ms > 0 && toy_model.plan().prefill_park &&
        toy_model.pauses + 1 < prefill_segments_) {
        toy_model.parked = &toy;
        ++toy_model.pauses;
        slice->paused = true;
        return CERE_OK;
    }
    toy_model.parked = nullptr;
    toy_model.pauses = 0;
    *consumed = n;
    const uint64_t cls = end != nullptr && end->cls != 0 ? end->cls : n;
    // The toy's rule for an end state (D15⁗′, D15⁗″): the step after its load_slot is exact only when it resumes it
    // with a class of the same row-bit signature as the one it was captured with (row_sig) and ends where that class's
    // cold call ends (its chunk start + cls); any other first step, or a resume without such a state, perturbs the
    // session's state as a wrong FP32 resume or class would.
    if (toy.resume_cls != 0 || (end != nullptr && end->resume)) {
        const uint64_t start = toy.length / prefill_chunk_ * prefill_chunk_;
        // PrefixCapture G1: a chunk-0 end state (start 0) resumes exact only with its K/V run: `end_kv_slots` slots
        // after the pair, each holding the capture's marker (its row, digest and class).
        const uint32_t kv_slots = start == 0 ? end_kv_slots_ : 0;
        bool exact = end != nullptr && end->resume && toy.resume_cls != 0 &&
                     row_sig(model, start, toy.resume_cls) == row_sig(model, start, cls) &&
                     start + cls == toy.length + n && end->in_kv_slots == kv_slots;
        for (uint32_t i = 0; exact && i < kv_slots; ++i) {
            const ToyState &kv = toy_model.slots[end->in_slot + 2 + i];
            exact = !kv.valid && kv.length == toy.length && kv.digest == toy.digest && kv.cls == toy.resume_cls;
        }
        toy.resume_cls = 0;
        if (!exact) {
            toy.digest ^= kEndPerturb;
            if (toy.stateful) toy.hash ^= kEndPerturb;
        }
    }
    // M2.13-L: the step's images (any chunk of the open prompt): their pad rows fold the image words; each image with
    // pad rows in the step counts one encode (the CUDA device encodes it in this call).
    PadWords pads(images);
    if (images != nullptr) {
        for (uint64_t k = 0; k < images->count; ++k) {
            const cere::mrope::ImageSpan &s = images->image[k].span;
            const uint64_t lo = std::max(s.token_offset, toy.length);
            const uint64_t hi = std::min(s.token_offset + cere::mrope::image_tokens(s), toy.length + n);
            if (lo >= hi) continue;
            image_encodes_.fetch_add(1);
            image_rows_.fetch_add(hi - lo);
        }
    }
    if (capture) {
        // The state as of `row` into `slot` (class cls); `slot + 1` holds the rest of it, no state of its own.
        ToyState at{toy.length + end->row, tokens[end->row - 1], toy.digest, true, toy.hash, cls};
        for (uint64_t i = 0; i < end->row; ++i) {
            at.digest = pads.fold(digest_step(at.digest, tokens[i]), toy.length + i);
            if (toy.stateful) at.hash = fold(at.hash, tokens[i]);
        }
        toy_model.slots[end->slot] = at;
        toy_model.slots[end->slot + 1] = ToyState{};
        // PrefixCapture G1: the K/V run of a chunk-0 end state, each slot a marker of the capture (no state of its own).
        for (uint32_t i = 0; i < end->kv_slots; ++i) {
            toy_model.slots[end->slot + 2 + i] = ToyState{at.length, 0, at.digest, false, 0, cls};
        }
    }
    if (n < count) {  // one chunk of a longer prompt: nothing is scored
        consume(toy, tokens, n, &pads);
        if (toy.drafting) extend_mtp(toy, tokens, n, tokens[n]);  // D15⁗: the MTP rows of the chunk
        return CERE_OK;
    }
    // A step without images past an image session's prompt scores at its rotary position (M2.13-L); a prompt step
    // scores at the prompt's.
    const uint32_t vocab = static_cast<ToyModel &>(model).vocab();
    forward(toy, tokens, count, vocab, mask_of(sampling), out, images == nullptr ? toy.rope_delta : 0, &pads);
    if (sampling != nullptr) sample(toy, *sampling, mask_of(sampling), vocab, out, logprobs);
    // A drafting session's MTP extend runs with the step that scores (D2): its proposals continue the scored token,
    // and they are right only over an intact draft state (D15⁗: its own MTP rows, or a draft state's).
    if (toy.drafting) {
        extend_mtp(toy, tokens, count, out->id);
        propose(toy, out->id, vocab, draft_.miss, draft_intact(toy));
    }
    return record(toy.id, count, sampling, mask_of(sampling), logprobs, vocab, out->id);
}

cere_status ToyDevice::set_rope_delta(cere::host::DeviceModel &, cere::host::DeviceSession &session,
                                      int64_t rope_delta) {
    if (rope_delta > 0) return CERE_ERR_ARGUMENT;
    static_cast<ToySession &>(session).rope_delta = rope_delta;
    return CERE_OK;
}

cere_status ToyDevice::set_session_draft(cere::host::DeviceModel &model, cere::host::DeviceSession &session, bool on) {
    auto &toy = static_cast<ToySession &>(session);
    if (on && !draft_supported(model)) return CERE_ERR_UNSUPPORTED;
    toy.drafting = on;
    toy.proposal_count = 0;
    return CERE_OK;
}

// Track LoRA (lora design note): the toy's pool is the CUDA device's size (host/lora.h pool_bytes) and an adapter is
// one offset: 1 + (FNV-1a 64 over rank, scale bits, then per target in the given order its id, A and B bytes, all
// little-endian) mod (vocab - 1), added to every id a bound session scores (forward); MTP proposals stay the base's.
uint64_t ToyDevice::lora_bytes(const cere::host::DeviceModel &model, uint32_t adapters, uint32_t rank) const {
    return cere::host::lora::pool_bytes(static_cast<const ToyModel &>(model).plan().config, adapters, rank);
}

cere_status ToyDevice::lora_reserve(cere::host::DeviceModel &model, uint32_t adapters, uint32_t rank) {
    auto &toy = static_cast<ToyModel &>(model);
    try {
        toy.lora_offsets.assign(adapters, 0);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    toy.lora_bytes = lora_bytes(model, adapters, rank);
    device_bytes_.fetch_add(toy.lora_bytes);
    return CERE_OK;
}

cere_status ToyDevice::lora_load(cere::host::DeviceModel &model, uint32_t slot, uint32_t rank, float scale,
                                 uint64_t count, const uint32_t *targets, const void *const *a, const void *const *b) {
    auto &toy = static_cast<ToyModel &>(model);
    const cere_qwen4_exp_config &cfg = toy.plan().config;
    uint64_t h = 14695981039346656037ull;
    const auto mix = [&h](const void *p, uint64_t n) {
        const auto *bytes = static_cast<const uint8_t *>(p);
        for (uint64_t i = 0; i < n; ++i) h = (h ^ bytes[i]) * 1099511628211ull;
    };
    const auto mix_u32 = [&mix](uint32_t v) {
        const uint8_t le[4] = {uint8_t(v), uint8_t(v >> 8), uint8_t(v >> 16), uint8_t(v >> 24)};
        mix(le, 4);
    };
    uint32_t scale_bits = 0;
    std::memcpy(&scale_bits, &scale, 4);
    mix_u32(rank);
    mix_u32(scale_bits);
    for (uint64_t i = 0; i < count; ++i) {
        cere::host::lora::Dims d;
        cere::host::lora::dims(cfg, targets[i], &d);
        mix_u32(targets[i]);
        mix(a[i], uint64_t(rank) * d.in * 2);
        mix(b[i], uint64_t(d.out) * rank * 2);
    }
    const uint32_t vocab = toy.vocab();
    toy.lora_offsets[slot] = vocab > 1 ? uint32_t(1 + h % (vocab - 1)) : 0;
    return CERE_OK;
}

cere_status ToyDevice::lora_unload(cere::host::DeviceModel &model, uint32_t slot) {
    static_cast<ToyModel &>(model).lora_offsets[slot] = 0;
    return CERE_OK;
}

cere_status ToyDevice::set_session_lora(cere::host::DeviceModel &model, cere::host::DeviceSession &session,
                                        int32_t slot) {
    auto &toy = static_cast<ToyModel &>(model);
    static_cast<ToySession &>(session).lora_off = slot < 0 ? 0 : toy.lora_offsets[size_t(slot)];
    return CERE_OK;
}

cere_status ToyDevice::research_reserve(cere::host::DeviceModel &model, uint64_t staging_bytes, uint64_t plan_bytes,
                                        uint64_t host_bytes, uint8_t **ring) {
    auto &toy = static_cast<ToyModel &>(model);
    try {
        toy.research_ring.assign(host_bytes, 0);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    toy.research_device_bytes = staging_bytes + plan_bytes;
    host_bytes_.fetch_add(host_bytes);
    device_bytes_.fetch_add(toy.research_device_bytes);
    *ring = toy.research_ring.data();
    return CERE_OK;
}

cere_status ToyDevice::set_session_research(cere::host::DeviceModel &, cere::host::DeviceSession &session,
                                            const cere::research::Binding *binding) {
    static_cast<ToySession &>(session).research = binding;
    return CERE_OK;
}

cere_status ToyDevice::step_rows(cere::host::DeviceModel &model, const cere::host::RowArgs *rows, uint32_t row_count,
                                 cere_token_score *out, cere_status *row_status) {
    if (!forward_supported()) return CERE_ERR_UNSUPPORTED;
    auto &toy_model = static_cast<ToyModel &>(model);
    if (rows == nullptr || out == nullptr || row_status == nullptr || row_count == 0 ||
        row_count > toy_model.max_sessions()) {
        return CERE_ERR_ARGUMENT;
    }
    for (uint32_t r = 0; r < row_count; ++r) {
        if (rows[r].session == nullptr || (rows[r].logprobs != nullptr && rows[r].sampling == nullptr) ||
            !static_cast<ToySession &>(*rows[r].session).bound ||
            static_cast<ToySession &>(*rows[r].session).drafting) {  // M2.9: a drafting session decodes by speculation
            return CERE_ERR_ARGUMENT;
        }
        for (uint32_t q = 0; q < r; ++q) {
            if (rows[q].session == rows[r].session) return CERE_ERR_ARGUMENT;
        }
    }
    bool any = false;
    for (uint32_t r = 0; r < row_count; ++r) {
        row_status[r] = precheck(static_cast<ToySession &>(*rows[r].session), 1, hold_after_, hold_ms_, hold_count_,
                                 hold_gate_, holds_);
        any = any || row_status[r] == CERE_OK;
    }
    if (!any) return CERE_OK;  // nothing launches, so the call is not counted
    if (fails_forward()) {
        for (uint32_t r = 0; r < row_count; ++r) {
            if (row_status[r] != CERE_OK) continue;
            static_cast<ToySession &>(*rows[r].session).broken = true;
            row_status[r] = CERE_ERR_INTERNAL;
        }
        return CERE_ERR_INTERNAL;
    }
    cere_status recorded = CERE_OK;
    for (uint32_t r = 0; r < row_count; ++r) {
        if (row_status[r] != CERE_OK) continue;
        auto &toy = static_cast<ToySession &>(*rows[r].session);
        forward(toy, &rows[r].token, 1, toy_model.vocab(), mask_of(rows[r].sampling), &out[r], toy.rope_delta);
        // Ledger CR4: the row's own sampling, output counts and counter key (its session position), as `step`.
        if (rows[r].sampling != nullptr) {
            sample(toy, *rows[r].sampling, mask_of(rows[r].sampling), toy_model.vocab(), &out[r], rows[r].logprobs);
        }
        if (const cere_status st = record(toy.id, 1, rows[r].sampling, mask_of(rows[r].sampling), rows[r].logprobs,
                                          toy_model.vocab(), out[r].id);
            st != CERE_OK) {
            recorded = st;
        }
    }
    if (recorded != CERE_OK) {  // a failure after the launch: every participant is broken
        for (uint32_t r = 0; r < row_count; ++r) {
            if (row_status[r] != CERE_OK) continue;
            static_cast<ToySession &>(*rows[r].session).broken = true;
            row_status[r] = recorded;
        }
    }
    return recorded;
}

cere_status ToyDevice::speculate_rows(cere::host::DeviceModel &model, const cere::host::SpecRowArgs *rows,
                                      uint32_t row_count, cere::host::SpecRowResult *out, cere_status *row_status) {
    if (!forward_supported()) return CERE_ERR_UNSUPPORTED;
    auto &toy_model = static_cast<ToyModel &>(model);
    if (rows == nullptr || out == nullptr || row_status == nullptr || row_count == 0 ||
        row_count > toy_model.max_sessions()) {
        return CERE_ERR_ARGUMENT;
    }
    for (uint32_t r = 0; r < row_count; ++r) {
        const cere::host::SpecRowArgs &a = rows[r];
        // GrammarSpec GS-02: a masked row's drafts are the caller's, at most CERE_MAX_CONSTRAINED_DRAFTS (one mask
        // per verify row).
        const bool masked = a.sampling != nullptr && a.sampling->vocab_mask != nullptr;
        if (a.session == nullptr || !static_cast<ToySession &>(*a.session).bound ||
            !static_cast<ToySession &>(*a.session).drafting ||
            (a.logprobs != nullptr && a.sampling == nullptr) || a.max_commit == 0 ||
            a.draft_count > (a.drafts != nullptr ? max_chain_drafts(model) : 3u) ||
            a.draft_count >= a.max_commit || (a.drafts != nullptr && a.draft_count == 0) ||
            (masked && a.draft_count != 0 && (a.drafts == nullptr || a.draft_count > CERE_MAX_CONSTRAINED_DRAFTS))) {
            return CERE_ERR_ARGUMENT;
        }
        for (uint32_t q = 0; q < r; ++q) {
            if (rows[q].session == a.session) return CERE_ERR_ARGUMENT;
        }
    }
    bool any = false;
    for (uint32_t r = 0; r < row_count; ++r) {
        out[r] = cere::host::SpecRowResult{};
        row_status[r] = precheck(static_cast<ToySession &>(*rows[r].session), 1, hold_after_, hold_ms_, hold_count_,
                                 hold_gate_, holds_);
        any = any || row_status[r] == CERE_OK;
    }
    if (!any) return CERE_OK;  // nothing launches, so the call is not counted
    // small_m_a8 (K2Prefill P3) counts as small_m: the same decode/verify kernels and round-cost table 1.
    const uint32_t k2 = toy_model.levers.k2_gemm != CERE_LEVER_K2_CUBLASLT ? 1 : 0;
    for (uint32_t r = 0; r < row_count; ++r) {
        if (row_status[r] != CERE_OK) continue;
        spec_rounds_[k2][std::min(rows[r].draft_count + 1, 10u)].fetch_add(1);
        if (toy_model.levers.decode_step != CERE_LEVER_DECODE_EAGER) spec_graph_rounds_.fetch_add(1);
    }
    // Every participant is broken with `status`; out[r] keeps what it committed.
    const auto fail = [&](cere_status status) {
        for (uint32_t r = 0; r < row_count; ++r) {
            if (row_status[r] != CERE_OK) continue;
            static_cast<ToySession &>(*rows[r].session).broken = true;
            row_status[r] = status;
        }
        return status;
    };
    // The verify launch (one fail_forward count): a failure here consumes nothing.
    if (fails_forward()) return fail(CERE_ERR_INTERNAL);
    const uint32_t vocab = toy_model.vocab();
    cere_status recorded = CERE_OK;
    for (uint32_t r = 0; r < row_count; ++r) {
        if (row_status[r] != CERE_OK) continue;
        const cere::host::SpecRowArgs &a = rows[r];
        auto &toy = static_cast<ToySession &>(*a.session);
        const uint32_t *drafts = a.drafts != nullptr ? a.drafts : toy.proposals;
        const uint32_t n = a.drafts != nullptr ? a.draft_count : std::min(a.draft_count, toy.proposal_count);
        // Row i consumes `token` and scores the token after it; its result is committed, and the next row runs only
        // while that result is the row's draft (accepted), not a stop token and max_commit is not reached. Rows past
        // the first rejection never touch the session, so the commit equals the undo restore (M29S2-20, D5).
        uint32_t token = a.token;
        for (uint32_t i = 0;; ++i) {
            cere_token_score &score = out[r].scores[i];
            const uint32_t *mask = mask_of(a.sampling, i, vocab);
            forward(toy, &token, 1, vocab, mask, &score, toy.rope_delta);
            const uint32_t *draft = i < n ? &drafts[i] : nullptr;
            bool accepted = false;
            if (a.sampling != nullptr) {
                sample(toy, *a.sampling, mask, vocab, &score, a.logprobs != nullptr ? &a.logprobs[i] : nullptr, draft,
                       &accepted);
            } else {
                accepted = draft != nullptr && score.id == *draft;
            }
            // One record= line per committed token, as plain decode writes.
            if (const cere_status st = record(toy.id, 1, a.sampling, mask,
                                              a.logprobs != nullptr ? &a.logprobs[i] : nullptr, vocab, score.id);
                st != CERE_OK) {
                recorded = st;
            }
            ++out[r].committed;
            if (accepted) ++out[r].accepted;
            if (!accepted || out[r].committed == a.max_commit || (a.is_stop != nullptr && a.is_stop(a.stop_ctx, score.id))) {
                break;
            }
            token = score.id;
        }
        toy.proposal_count = 0;  // consumed; the draft-extend proposes again
    }
    if (recorded != CERE_OK) return fail(recorded);
    // The draft-extend launch (one fail_forward count): a failure here keeps the committed tokens in out[r].
    if (fails_forward()) return fail(CERE_ERR_INTERNAL);
    for (uint32_t r = 0; r < row_count; ++r) {
        if (row_status[r] != CERE_OK) continue;
        auto &toy = static_cast<ToySession &>(*rows[r].session);
        propose(toy, out[r].scores[out[r].committed - 1].id, vocab, draft_.miss);
    }
    return CERE_OK;
}

uint32_t ToyDevice::max_chain_drafts(const cere::host::DeviceModel &) const {
    return draft_.chain != 0 ? draft_.chain : 3;
}

uint32_t ToyDevice::pending_drafts(const cere::host::DeviceSession &session, uint32_t *ids, uint32_t max) const {
    const auto &toy = static_cast<const ToySession &>(session);
    if (!toy.drafting || ids == nullptr) return 0;
    const uint32_t n = std::min(toy.proposal_count, max);
    std::copy(toy.proposals, toy.proposals + n, ids);
    return n;
}

cere_status ToyDevice::record(uint64_t session, uint64_t count, const cere::host::ItemSampling *sampling,
                              const uint32_t *mask, const cere_token_logprobs *logprobs, uint32_t vocab,
                              uint32_t token) {
    if (record_.empty()) return CERE_OK;
    char line[512];
    int n = std::snprintf(line, sizeof(line), "session=%" PRIu64 " count=%" PRIu64, session, count);
    if (sampling != nullptr && sampling->sampled) {
        const cere_sampling_params &p = sampling->params;
        n += std::snprintf(line + n, sizeof(line) - n,
                           " temperature=%.9g top_k=%" PRIu32 " top_p=%.9g min_p=%.9g frequency=%.9g presence=%.9g"
                           " repetition=%.9g logit_bias=%zu seed=%" PRIu64,
                           double(p.temperature), p.top_k, double(p.top_p), double(p.min_p),
                           double(p.frequency_penalty), double(p.presence_penalty), double(p.repetition_penalty),
                           sampling->bias.size(), p.seed);
    } else {
        n += std::snprintf(line + n, sizeof(line) - n, " sampling=null");
    }
    if (logprobs != nullptr) {
        n += std::snprintf(line + n, sizeof(line) - n, " logprobs=%" PRIu32, sampling->top_logprobs);
    } else {
        n += std::snprintf(line + n, sizeof(line) - n, " logprobs=null");
    }
    if (mask != nullptr) {  // M2.8: the allowed ids of the row's mask
        uint64_t allowed = 0;
        for (uint32_t id = 0; id < vocab; ++id) allowed += (mask[id >> 5] >> (id & 31)) & 1u;
        n += std::snprintf(line + n, sizeof(line) - n, " mask=%" PRIu64, allowed);
    }
    std::snprintf(line + n, sizeof(line) - n, " token=%" PRIu32 "\n", token);
    const std::lock_guard<std::mutex> lock(record_mutex_);
    std::FILE *file = std::fopen(record_.c_str(), "a");
    if (file == nullptr) return CERE_ERR_INTERNAL;
    const bool written = std::fputs(line, file) >= 0;
    return std::fclose(file) == 0 && written ? CERE_OK : CERE_ERR_INTERNAL;
}

cere_status ToyDevice::save_state(cere::host::DeviceModel &, cere::host::DeviceSession &session,
                                  const cere::host::snapshot::Plan &plan, uint64_t position, uint8_t *dst) {
    auto &toy = static_cast<ToySession &>(session);
    if (toy.length != position) return CERE_ERR_INTERNAL;
    namespace snap = cere::host::snapshot;
    bool draft = false;
    for (const snap::Section &s : plan.sections) draft = draft || s.kind == snap::kDraft;
    if (draft != toy.drafting) return CERE_ERR_ARGUMENT;  // the host plans kind 10 exactly for a drafting session
    toy.hold_done = true;
    toy.holding = false;
    if (snapshots_.fetch_add(1) < snapshot_capacity_) return CERE_ERR_CAPACITY;
    // The toy keeps no KV or recurrent bytes (the host writes kinds 0 and 11): kind 7 carries the ToyState digest
    // (M2.10), kind 8 the session's sampler counts (ledger CR13; zero while it never sampled) and kind 10 (M2.9) the
    // pending proposals after zero MTP K/V and QSA parts; the rest is zero.
    if (const cere_status st = write_device_sections(plan, toy.digest, toy.counts, dst); st != CERE_OK) return st;
    snap::write_proposals(dst, plan, toy.proposal_count, toy.proposals);
    return CERE_OK;
}

cere_status ToyDevice::load_state(cere::host::DeviceModel &, cere::host::DeviceSession &session,
                                  const cere::host::snapshot::Plan &plan, uint64_t position, const uint8_t *src) {
    if (!restore_gate_.empty()) {
        // Refused while the gate file exists; the appended line tells the test one refusal happened.
        if (std::FILE *gate = std::fopen(restore_gate_.c_str(), "r+")) {
            const bool written = std::fseek(gate, 0, SEEK_END) == 0 && std::fputs("refused\n", gate) >= 0;
            return std::fclose(gate) == 0 && written ? CERE_ERR_CAPACITY : CERE_ERR_INTERNAL;
        }
    }
    auto &toy = static_cast<ToySession &>(session);
    if (!toy.bound) return CERE_ERR_ARGUMENT;  // M2.10 D3′: the host binds a lazy session's pages first
    namespace snap = cere::host::snapshot;
    uint32_t count = 0;
    uint32_t ids[3] = {};
    // The host sets the session drafting (set_session_draft) exactly when the image has kind 10.
    if (snap::read_proposals(src, plan, &count, ids) != toy.drafting) return CERE_ERR_ARGUMENT;
    const ToyState state = read_state(plan, position, src);
    toy.last = state.last;
    toy.length = state.length;
    toy.digest = state.digest;
    toy.hash = toy.stateful ? stateful_hash(src + plan.sections.at(0).offset, position) : 0;
    toy.proposal_count = count;
    std::copy(ids, ids + 3, toy.proposals);
    for (const cere::host::snapshot::Section &s : plan.sections) {
        if (s.kind != cere::host::snapshot::kSampler) continue;
        const uint8_t *counts = src + s.offset;
        const uint64_t n = s.bytes / sizeof(uint32_t);
        bool any = false;
        for (uint64_t i = 0; i < n && !any; ++i) any = get_u32(counts + 4 * i) != 0;
        // A session that never sampled keeps no counts (the first sampled step allocates them zeroed).
        if (!any) continue;
        toy.counts.resize(n);
        for (uint64_t i = 0; i < n; ++i) toy.counts[i] = get_u32(counts + 4 * i);
    }
    return CERE_OK;
}

std::vector<uint8_t> ToyDevice::tensor(const cere::host::DeviceModel &model, uint64_t tensor_index) const {
    const auto &toy = static_cast<const ToyModel &>(model);
    if (toy.segment != nullptr && toy.region_offsets.at(tensor_index) != UINT64_MAX) {
        const uint8_t *p = toy.segment->data() + toy.region_offsets[tensor_index];
        return std::vector<uint8_t>(p, p + toy.plan().tensors[tensor_index].bytes);
    }
    return toy.buffers.at(tensor_index);
}

uint64_t ToyDevice::free_pages(const cere::host::DeviceModel &model) const {
    const auto &toy = static_cast<const ToyModel &>(model);
    std::lock_guard<std::mutex> lock(toy.pool_mutex);
    return toy.pool.free_pages();
}

uint32_t ToyDevice::page_refs(const cere::host::DeviceModel &model, uint32_t id) const {
    const auto &toy = static_cast<const ToyModel &>(model);
    std::lock_guard<std::mutex> lock(toy.pool_mutex);
    return id < toy.pool.pages() ? toy.pool.refs(id) : 0;
}

cere_status ToyDevice::retain_pages(cere::host::DeviceModel &model, const uint32_t *ids, uint64_t n) {
    if (ids == nullptr && n != 0) return CERE_ERR_ARGUMENT;
    auto &toy = static_cast<ToyModel &>(model);
    std::lock_guard<std::mutex> lock(toy.pool_mutex);
    return toy.pool.retain(ids, n);
}

cere_status ToyDevice::release_pages(cere::host::DeviceModel &model, const uint32_t *ids, uint64_t n) {
    if (ids == nullptr && n != 0) return CERE_ERR_ARGUMENT;
    auto &toy = static_cast<ToyModel &>(model);
    std::lock_guard<std::mutex> lock(toy.pool_mutex);
    return toy.pool.release_pages(ids, n);
}

cere_status ToyDevice::take_pages(cere::host::DeviceModel &model, uint64_t n, std::vector<uint32_t> *out) {
    auto &toy = static_cast<ToyModel &>(model);
    std::lock_guard<std::mutex> lock(toy.pool_mutex);
    return toy.pool.take_pages(n, out);
}

cere_status ToyDevice::bind_pages(cere::host::DeviceModel &model, cere::host::DeviceSession &session,
                                  const uint32_t *prefix, uint64_t n_prefix) {
    auto &toy_model = static_cast<ToyModel &>(model);
    auto &toy = static_cast<ToySession &>(session);
    const uint64_t pages = cere::host::kv_pages(toy.max_context);
    if (!toy.lazy || toy.bound || n_prefix > pages || (n_prefix != 0 && prefix == nullptr)) return CERE_ERR_ARGUMENT;
    std::lock_guard<std::mutex> lock(toy_model.pool_mutex);
    for (uint64_t i = 0; i < n_prefix; ++i) {
        if (prefix[i] >= toy_model.pool.pages() || toy_model.pool.refs(prefix[i]) == 0) return CERE_ERR_ARGUMENT;
    }
    if (pages - n_prefix > toy_model.pool.free_pages()) return CERE_ERR_CAPACITY;
    // The page list's capacity was reserved at create, so nothing below allocates or fails.
    if (const cere_status st = toy_model.pool.retain(prefix, n_prefix); st != CERE_OK) return st;
    toy.pages.assign(prefix, prefix + n_prefix);
    if (const cere_status st = toy_model.pool.take_pages(pages - n_prefix, &toy.pages); st != CERE_OK) {
        cere::host::release_pages_logged(toy_model.pool, prefix, n_prefix, "toy_bind_pages");
        toy.pages.clear();
        return st;
    }
    toy.bound = true;
    return CERE_OK;
}

std::vector<uint32_t> ToyDevice::session_page_list(const cere::host::DeviceSession &session) const {
    return static_cast<const ToySession &>(session).pages;
}

uint64_t ToyDevice::session_position(const cere::host::DeviceSession &session) const {
    return static_cast<const ToySession &>(session).length;
}

uint64_t ToyDevice::row_sig(const cere::host::DeviceModel &, uint64_t, uint64_t cls) const {
    for (const auto &[lo, hi] : row_sig_) {
        if (cls >= lo && cls <= hi) return lo;
    }
    return cls;
}

bool ToyDevice::prefill_selection(const cere::host::DeviceModel &, std::string &out) const {
    out += "toy prefill-selection v1\n";
    if (prefill_sig_ != 0) out += "toy bucket " + std::to_string(prefill_sig_) + "\n";
    return true;
}

cere_status ToyDevice::capture_state(cere::host::DeviceModel &model, const cere::host::DeviceSession &session,
                                     uint32_t slot) {
    auto &toy_model = static_cast<ToyModel &>(model);
    const auto &toy = static_cast<const ToySession &>(session);
    if (slot >= toy_model.slots.size() || !toy.bound) return CERE_ERR_ARGUMENT;
    toy_model.slots[slot] = ToyState{toy.length, toy.last, toy.digest, true, toy.hash};
    return CERE_OK;
}

cere_status ToyDevice::load_slot(cere::host::DeviceModel &model, cere::host::DeviceSession &session, uint32_t slot,
                                 uint64_t position) {
    auto &toy_model = static_cast<ToyModel &>(model);
    auto &toy = static_cast<ToySession &>(session);
    // A slot holds the state of exactly `position` tokens; any mismatch is a host bookkeeping error.
    if (slot >= toy_model.slots.size() || !toy.bound || toy.length != 0 || position == 0 ||
        position % cere::host::kKvPageTokens != 0 || position > toy.max_context) {
        return CERE_ERR_ARGUMENT;
    }
    const ToyState &state = toy_model.slots[slot];
    if (!state.valid || state.length != position) return CERE_ERR_ARGUMENT;
    toy.length = state.length;
    toy.last = state.last;
    toy.digest = state.digest;
    toy.hash = state.hash;
    toy.resume_cls = state.cls;
    return CERE_OK;
}

cere_status ToyDevice::save_prefix(cere::host::DeviceModel &model, const uint32_t *pages, uint64_t n, uint32_t slot,
                                   const cere::host::snapshot::Plan &plan, uint64_t position, uint8_t *dst) {
    auto &toy_model = static_cast<ToyModel &>(model);
    if (slot >= toy_model.slots.size() || pages == nullptr || n < cere::host::kv_pages(position) || dst == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    const ToyState &state = toy_model.slots[slot];
    // An end state (D15⁗′) is never spilled or exported: its 2-slot FP32 layout has no image.
    if (!state.valid || state.length != position || state.cls != 0) return CERE_ERR_ARGUMENT;
    if (const cere_status st = write_device_sections(plan, state.digest, {}, dst); st != CERE_OK) {  // zero counts
        return st;
    }
    // Sessions (O211): kind 10 (a host-tier image on a model with drafts) carries the edge's MTP rows. The toy keeps
    // no K/V bytes, so the proposals part of kind 10 records its row record instead: ids[0] = 1 when every MTP row p
    // of the edge's [first_row, position - 1) holds the token at p + 1, ids[1] = row position - 1's next id + 1.
    namespace snap = cere::host::snapshot;
    for (const snap::Section &s : plan.sections) {
        if (s.kind != snap::kDraft || toy_model.mtp_next.empty()) continue;
        const uint64_t first = snap::read_first_row(dst);
        if (first >= position || s.bytes < snap::kDraftProposalBytes) return CERE_ERR_ARGUMENT;
        const auto row = [&](uint64_t p) {
            return uint64_t(pages[p / cere::host::kKvPageTokens]) * cere::host::kKvPageTokens +
                   p % cere::host::kKvPageTokens;
        };
        bool intact = true;
        for (uint64_t p = first; p + 1 < position; ++p) {
            intact = intact && toy_model.mtp_next[row(p)] == toy_model.kv_ids[row(p + 1)];
        }
        uint8_t *part = dst + s.offset + s.bytes - snap::kDraftProposalBytes;
        put_u32(part + 4, intact ? 1 : 0);
        put_u32(part + 8, toy_model.mtp_next[row(position - 1)]);
    }
    return CERE_OK;
}

cere_status ToyDevice::load_prefix(cere::host::DeviceModel &model, const uint32_t *pages, uint64_t n, uint32_t slot,
                                   const cere::host::snapshot::Plan &plan, uint64_t position, const uint8_t *src) {
    auto &toy_model = static_cast<ToyModel &>(model);
    if (slot >= toy_model.slots.size() || pages == nullptr || n < cere::host::kv_pages(position) || src == nullptr ||
        position == 0) {
        return CERE_ERR_ARGUMENT;
    }
    {
        std::lock_guard<std::mutex> lock(toy_model.pool_mutex);
        for (uint64_t i = 0; i < n; ++i) {
            if (pages[i] >= toy_model.pool.pages() || toy_model.pool.refs(pages[i]) == 0) return CERE_ERR_ARGUMENT;
        }
    }
    toy_model.slots[slot] = read_state(plan, position, src);
    if (draft_.stateful) toy_model.slots[slot].hash = stateful_hash(src + plan.sections.at(0).offset, position);
    // On a model with drafts and a prefix cache: the loaded rows [first_row, position) hold the image's tokens, and
    // their MTP rows what save_prefix recorded in kind 10 (Sessions O211), else none (unwritten).
    if (!toy_model.kv_ids.empty()) {
        namespace snap = cere::host::snapshot;
        const uint64_t first = snap::read_first_row(src);
        if (first >= position) return CERE_ERR_ARGUMENT;
        uint32_t intact = 0;
        uint32_t last = 0;
        for (const snap::Section &s : plan.sections) {
            if (s.kind != snap::kDraft || s.bytes < snap::kDraftProposalBytes) continue;
            const uint8_t *part = src + s.offset + s.bytes - snap::kDraftProposalBytes;
            intact = get_u32(part + 4);
            last = get_u32(part + 8);
        }
        const uint8_t *tokens = src + plan.sections.at(0).offset;
        for (uint64_t p = first; p < position; ++p) {
            const uint64_t row = uint64_t(pages[p / cere::host::kKvPageTokens]) * cere::host::kKvPageTokens +
                                 p % cere::host::kKvPageTokens;
            toy_model.kv_ids[row] = get_u32(tokens + 4 * p) + 1;
            toy_model.mtp_next[row] = intact == 0 ? 0 : p + 1 < position ? get_u32(tokens + 4 * (p + 1)) + 1 : last;
        }
    }
    return CERE_OK;
}

std::vector<uint32_t> ToyDevice::session_pages(const cere::host::DeviceSession &session) const {
    return static_cast<const ToySession &>(session).pages;
}

ToyState ToyDevice::session_state(const cere::host::DeviceSession &session) const {
    const auto &toy = static_cast<const ToySession &>(session);
    return ToyState{toy.length, toy.last, toy.digest, true, toy.hash};
}

ToyState ToyDevice::slot_state(const cere::host::DeviceModel &model, uint32_t slot) const {
    return static_cast<const ToyModel &>(model).slots.at(slot);
}

}  // namespace cere::toy

namespace cere::host {

// SelfTune ST4 S3c: the toy runs no SDAR model, so a toy sdar library refuses an in-scope descriptor as `device`.
const Support *sdar_support(const Device &) { return nullptr; }
// Inkling 2 Mini (inkling design note S0): nor an Inkling model, refused as `device` the same way.
const Support *inkling2_support(const Device &, bool) { return nullptr; }

cere_status create_device(const char *device_id, std::unique_ptr<Device> *out) {
    const std::string id(device_id);
    const size_t comma = id.find(',');
    if (id.substr(0, comma) != "toy") return CERE_ERR_UNSUPPORTED;
    uint64_t host_bytes = cere::toy::kUnlimited;
    uint64_t fail_forward = cere::toy::kUnlimited;
    uint64_t hold_after = cere::toy::kUnlimited;
    uint64_t hold_ms = cere::toy::kDefaultHoldMs;
    bool sampler = true;
    std::string record;
    std::string restore_gate;
    std::string hold_gate;
    std::string staging_gate;
    std::string tensor_digests;
    uint64_t snapshot_capacity = 0;
    uint64_t hold_count = cere::toy::kUnlimited;
    uint64_t prefill_chunk = cere::toy::kDefaultPrefillChunk;
    cere::toy::DraftOptions draft;
    bool fail_sticky = false;
    uint64_t create_busy = 0;
    uint64_t create_capacity_delay_ms = 0;
    uint64_t prefill_segments = 1;
    uint32_t end_kv_slots = 0;
    std::vector<std::pair<uint64_t, uint64_t>> row_sig;
    uint64_t prefill_sig = 0;
    bool timed_rounds = false;
    ShareConfig share;
    size_t pos = comma;
    while (pos != std::string::npos) {
        const size_t next = id.find(',', pos + 1);
        const std::string option = id.substr(pos + 1, next == std::string::npos ? std::string::npos : next - pos - 1);
        const size_t eq = option.find('=');
        if (eq == std::string::npos) return CERE_ERR_ARGUMENT;
        const std::string key = option.substr(0, eq);
        if (const int shared = parse_share_option(key, option.substr(eq + 1), &share); shared != 0) {
            if (shared < 0) return CERE_ERR_ARGUMENT;
            pos = next;
            continue;
        }
        if (key == "record" || key == "restore_gate" || key == "hold_gate" || key == "staging_gate" ||
            key == "tensor_digests") {
            std::string path = option.substr(eq + 1);
            if (path.empty() || path[0] != '/') return CERE_ERR_ARGUMENT;
            (key == "record"         ? record
             : key == "restore_gate" ? restore_gate
             : key == "hold_gate"    ? hold_gate
             : key == "staging_gate" ? staging_gate
                                     : tensor_digests) = std::move(path);
            pos = next;
            continue;
        }
        if (key == "row_sig") {  // LO-HI, 1 <= LO <= HI, disjoint from every earlier range
            const std::string range = option.substr(eq + 1);
            const size_t dash = range.find('-');
            uint64_t lo = 0;
            uint64_t hi = 0;
            if (dash == std::string::npos || !cere::toy::parse_u64(range.substr(0, dash), &lo) ||
                !cere::toy::parse_u64(range.substr(dash + 1), &hi) || lo == 0 || lo > hi) {
                return CERE_ERR_ARGUMENT;
            }
            for (const auto &[a, b] : row_sig) {
                if (lo <= b && a <= hi) return CERE_ERR_ARGUMENT;
            }
            row_sig.emplace_back(lo, hi);
            pos = next;
            continue;
        }
        uint64_t value = 0;
        if (!cere::toy::parse_u64(option.substr(eq + 1), &value)) return CERE_ERR_ARGUMENT;
        if (key == "host_bytes") {
            host_bytes = value;
        } else if (key == "fail_forward") {
            fail_forward = value;
        } else if (key == "fail_sticky" && value <= 1) {
            fail_sticky = value == 1;
        } else if (key == "create_busy") {
            create_busy = value;
        } else if (key == "create_capacity_delay_ms") {
            create_capacity_delay_ms = value;
        } else if (key == "timed_rounds" && value <= 1) {
            timed_rounds = value == 1;
        } else if (key == "hold_after") {
            hold_after = value;
        } else if (key == "hold_ms") {
            hold_ms = value;
        } else if (key == "snapshot_capacity") {
            snapshot_capacity = value;
        } else if (key == "hold_count") {
            hold_count = value;
        } else if (key == "release_device" && value == 1) {
            // device/cuda_device.h D10: the toy holds no device context to release.
        } else if (key == "sampler" && value <= 1) {
            sampler = value == 1;
        } else if (key == "prefill_chunk" && value != 0) {
            prefill_chunk = value;
        } else if (key == "prefill_segments" && value >= 2) {
            prefill_segments = value;
        } else if (key == "end_kv_slots" && value >= 1 && value <= 8) {
            end_kv_slots = uint32_t(value);
        } else if (key == "prefill_sig" && value >= 1) {
            prefill_sig = value;
        } else if (key == "draft" && value <= 1) {
            draft.draft = value == 1;
        } else if (key == "draft_miss" && value >= 1 && value <= 3) {
            draft.miss = uint32_t(value);
        } else if (key == "stateful" && value <= 1) {
            draft.stateful = value == 1;
        } else if (key == "cycle" && value >= 2 && value <= cere::toy::kSmallVocab) {
            draft.cycle = uint32_t(value);
        } else if (key == "chain" && value >= 4 && value <= cere::host::kMaxSpecChainDrafts) {
            draft.chain = uint32_t(value);
        } else {
            return CERE_ERR_ARGUMENT;
        }
        pos = next;
    }
    if (!share_config_complete(share)) return CERE_ERR_ARGUMENT;
    auto device = std::make_unique<cere::toy::ToyDevice>(host_bytes, fail_forward, hold_after, hold_ms, hold_count,
                                                         std::move(hold_gate), snapshot_capacity,
                                                         std::move(restore_gate), sampler, std::move(record),
                                                         prefill_chunk, std::move(share), draft,
                                                         std::move(staging_gate), std::move(tensor_digests));
    device->set_fail_sticky(fail_sticky);
    device->set_create_busy(create_busy);
    device->set_create_capacity_delay_ms(create_capacity_delay_ms);
    device->set_prefill_segments(prefill_segments);
    device->set_end_kv_slots(end_kv_slots);
    device->set_row_sig(std::move(row_sig));
    device->set_prefill_sig(prefill_sig);
    device->set_times_spec_rounds(timed_rounds);
    *out = std::move(device);
    return CERE_OK;
}

}  // namespace cere::host
