// Host-layer state behind the opaque ABI handles: executor -> models -> sessions -> tickets (no CUDA includes).
#pragma once

#include "cere_executor.h"
#include "host/device.h"
#include "host/mrope.h"
#include "host/prefix_cache.h"
#include "host/spec_lookup.h"
#include "research/capture.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cere::host {

// Byte ranges put into one tensor. A range is reserved before its copy starts, so two concurrent puts whose ranges
// overlap cannot both succeed; the ranges stay disjoint, so their total equals the covered bytes.
class TensorLedger {
public:
    // Reserves [offset, offset + len) (len > 0, no overflow); false when it overlaps a reserved range.
    bool reserve(uint64_t offset, uint64_t len);
    // Drops the range starting at `offset` (a failed copy leaves a gap).
    void unreserve(uint64_t offset);
    uint64_t covered() const;

private:
    mutable std::mutex mutex_;
    std::map<uint64_t, uint64_t> ranges_;  // start -> end (exclusive)
    uint64_t covered_ = 0;
};

// Put staging ranges (cere_executor_register_staging, m216 design note M216S-11): disjoint caller ranges, each with
// the puts copying from it right now. Thread-safe (puts look ranges up concurrently with each other and with
// register/unregister).
class StagingRanges {
public:
    // Adds [base, base + bytes) after `pin` (the device's register_staging) returns CERE_OK; CERE_ERR_ARGUMENT when
    // it overlaps a range (or wraps), else pin's status. One pin at a time.
    template <typename Pin>
    cere_status add(const uint8_t *base, uint64_t bytes, Pin &&pin) {
        std::lock_guard<std::mutex> lock(mutex_);
        const uintptr_t start = reinterpret_cast<uintptr_t>(base);
        if (start + bytes < start || overlaps(start, start + bytes)) return CERE_ERR_ARGUMENT;
        if (const cere_status st = pin(); st != CERE_OK) return st;
        ranges_.emplace(start, Range{start + bytes, 0});
        return CERE_OK;
    }
    // Removes the range starting at `base` after `unpin(bytes)`; CERE_ERR_ARGUMENT when no range starts there,
    // CERE_ERR_BUSY (kept) while a put holds it.
    template <typename Unpin>
    cere_status remove(const uint8_t *base, Unpin &&unpin) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = ranges_.find(reinterpret_cast<uintptr_t>(base));
        if (it == ranges_.end()) return CERE_ERR_ARGUMENT;
        if (it->second.puts != 0) return CERE_ERR_BUSY;
        unpin(it->second.end - it->first);
        ranges_.erase(it);
        return CERE_OK;
    }
    // Holds the range containing all of [p, p + len) for one put; 0 when none does (release is then not called).
    uintptr_t acquire(const void *p, uint64_t len);
    void release(uintptr_t start);
    bool empty() const;

private:
    struct Range {
        uintptr_t end;  // exclusive
        uint64_t puts;
    };
    bool overlaps(uintptr_t start, uintptr_t end) const;
    mutable std::mutex mutex_;
    std::map<uintptr_t, Range> ranges_;  // start -> range
};

// First-fit, lowest-offset sub-allocator over a model's snapshot arena (m211 design note D5, ledger M3):
// deterministic, so the same sequence of allocations and frees always yields the same offsets.
class ArenaAllocator {
public:
    explicit ArenaAllocator(uint64_t capacity = 0) : capacity_(capacity) {}
    // The lowest offset with `bytes` (> 0) free after it; false (nothing taken) when no gap is large enough.
    bool allocate(uint64_t bytes, uint64_t *offset);
    // Frees the range starting at `offset`.
    void free(uint64_t offset);
    // Whether `bytes` would fit now (allocate would succeed).
    bool fits(uint64_t bytes) const;
    uint64_t used() const { return used_; }
    uint64_t capacity() const { return capacity_; }
    // Bytes of the range starting at `offset` (0 when none starts there).
    uint64_t range_bytes(uint64_t offset) const;

private:
    uint64_t capacity_;
    std::map<uint64_t, uint64_t> ranges_;  // start -> end (exclusive)
    uint64_t used_ = 0;
};

// One submitted work item between submit and its terminal poll; owns copies of the inputs, leases the outputs.
struct Ticket {
    uint64_t submission_id = 0;
    uint64_t model_generation = 0;
    cere_session *session = nullptr;
    uint64_t max_new_tokens = 0;
    uint32_t *out_tokens = nullptr;
    cere_token_score *out_scores = nullptr;
    cere_token_logprobs *out_logprobs = nullptr;
    std::optional<ItemSampling> sampling;  // set when the item has sampling or out_logprobs
    std::vector<uint32_t> stop_tokens;     // the item's own stop tokens (the model's are on cere_model)
    uint64_t reserved_peak_bytes = 0;
    std::atomic<uint64_t> committed{0};  // published with release ordering after the output entries
    bool cancel_requested = false;
    bool ignore_stop_tokens = false;  // cere_work.flags bit 0 (M2.12 D9)
    // A round result awaiting this ticket's poll (M2.6 I4): the finish reason it reached (CERE_FINISH_NONE while
    // running) and the status of a CERE_FINISH_ERROR.
    uint32_t finish_pending = CERE_FINISH_NONE;
    cere_status error_pending = CERE_OK;
    // S1 G3 (beat-sglang design note): the executor's `rounds` when this ticket was last reported (or submitted);
    // below `rounds`, a round another ticket's poll ran may hold this ticket's result, so its poll reports first.
    uint64_t seen_round = 0;
    // Grammar (m28 design note A6, D9): a constrained ticket (cere_work.vocab_mask_words != 0) holds the mask of
    // its next token in `mask` (mask_words = ceil(vocab/32) words per row, sized once at submit; empty =
    // unconstrained). `mask_ready` says it is present for position `mask_position` (= committed); submit sets
    // position 0, cere_executor_constrain / cere_executor_constrain_drafts the later ones, and the commit of that
    // position clears it. GrammarSpec (grammar-spec design note GS-01): a constrained ticket of a drafting session
    // has 1 + CERE_MAX_CONSTRAINED_DRAFTS rows (row j = the mask after drafts 1..j), `mask_rows` of them set for
    // mask_position; `offer` holds the offer_count drafts cere_executor_round_drafts froze for offer_position (with
    // the policy's pick in `pick`), valid while `offer_frozen`.
    std::vector<uint32_t> mask;
    uint64_t mask_position = 0;
    bool mask_ready = false;
    uint32_t mask_words = 0;
    uint32_t mask_rows = 1;
    bool offer_frozen = false;
    uint64_t offer_position = 0;
    uint32_t offer_count = 0;
    std::array<uint32_t, CERE_MAX_CONSTRAINED_DRAFTS> offer{};
    bool constrained() const { return !mask.empty(); }
    // M2.9 (m29 design note D1, D9, I4): a NO_SPECULATION item proposes no drafts; a VERIFY item's caller drafts
    // (1..3 ids) run as its one speculation round; the completion counters (speculation rounds, committed drafts).
    bool no_speculation = false;
    bool verify = false;
    std::vector<uint32_t> drafts;
    uint64_t verify_steps = 0;
    uint64_t accepted_drafts = 0;
    // M2.9L (m29l design note D2, D5): flag bits 2 (lookup drafts may be verified) and 3 (MTP proposals never are),
    // bit 4 (m29a design note A2: the policy runs, priced by the model's round_costs[k2_small_m]), and this round's
    // policy pick (unset when the policy did not run) with its lookup drafts and their match length, kept until the
    // round's outcome.
    bool lookup_drafts = false;
    bool no_mtp_drafts = false;
    bool adaptive_drafts = false;
    // DraftAccept (bit 5, CERE_WORK_TAIL_DRAFTS): a greedy adaptive lookup ticket may append the lookup tail to the
    // three MTP proposals (DraftPolicy::extend); `lookup` holds up to the device's max_chain_drafts lookup drafts.
    bool tail_drafts = false;
    // TreeSpec (bit 6, CERE_WORK_LONG_DRAFTS): a greedy adaptive lookup ticket may verify up to
    // min(max_chain_drafts, DraftPolicy::kMaxTailDrafts) lookup drafts as one chain (DraftPolicy::lengthen), without
    // MTP agreement.
    bool long_drafts = false;
    // Chains (bit 7, CERE_WORK_CHAIN_DRAFTS, O123): the chains of bits 5 and 6 may hold up to max_chain_drafts drafts.
    bool chain_drafts = false;
    std::optional<DraftPolicy::Pick> pick;
    std::array<uint32_t, kMaxSpecChainDrafts> lookup{};
    uint32_t lookup_match = 0;
    // A speculation round that failed after its commit (D7): the tokens are reported on that poll, and the next
    // round ends the ticket CERE_FINISH_ERROR with this status.
    cere_status error_deferred = CERE_OK;
    // Whether the ticket may step now: unconstrained, or its mask for `committed` is present.
    bool mask_present() const {
        return mask.empty() || (mask_ready && mask_position == committed.load(std::memory_order_relaxed));
    }
    // M2.10 (m210 design note D3′, D4, D4′, D10): set at the ticket's first prefill round on a cache-on model:
    // the prompt tokens taken from the prefix cache (cere_completion.cached_tokens), and the split points of a
    // cache-on PREFILL ticket ((a′), (c), (d); ascending, each in (h, n)) with their capture kind. `capture_position`
    // is the item's W 132 hint, validated at submit.
    uint64_t capture_position = 0;
    uint64_t cached_tokens = 0;
    std::vector<std::pair<uint64_t, CaptureKind>> capture_points;
    // m210 D15⁗′ stage 2a: the end capture of a drafting cache-on PREFILL ticket (`end_at` = 0: none): depth b, the
    // start s of the cold call holding it and that call's class e − s. PrefixCapture G1: a chunk-0 one (s = 0) also
    // saves its K/V rows [0, b) into `end_kv_slots` slots after the pair (Device::end_kv_slots; 0 for s > 0).
    uint64_t end_at = 0;
    uint64_t end_start = 0;
    uint64_t end_cls = 0;
    uint32_t end_kv_slots = 0;
    // MixedPD (mixed-pd design note): the ticket's prefill call is paused between layer segments (StepSlice): it is
    // the next prefill round's prompt, cancelled or not, and its poll does not end it; `park_pair` is the end-state
    // pair the paused call captures into (PrefixNode::kNoSlot: none), taken at its first segment.
    bool parked = false;
    uint32_t park_pair = PrefixNode::kNoSlot;
    // BC4 (b) (batch-concurrency design note): the older prompt ticket whose prompt shares the longest key prefix
    // with this lazy, cache-on ticket's (`share_owner`, 0: none) and that prefix's length, found at the ticket's first
    // prefill round choice (`share_checked`). The ticket's lookup waits until the owner has fed the shared tokens, so
    // shortest-first keeps the hit FIFO order gives it.
    bool share_checked = false;
    uint64_t share_owner = 0;
    uint64_t share_tokens = 0;
    // BC4 (b)'s aging cap: the prefill calls that started for a younger ticket while this one could run since its own
    // last call (`overtaken`; at the executor's prefill_age_calls it runs the next call) and the most it reached
    // (`overtaken_max`, reported on stderr when its prompt is consumed).
    uint32_t overtaken = 0;
    uint32_t overtaken_max = 0;
    // PT0a (prefix-token design note): the ticket's decode track capture already found no slot and was counted
    // (SkipSite::kDecodeTrack): its later boundaries retry without counting again.
    bool decode_skip_counted = false;
};

}  // namespace cere::host

struct cere_model {
    cere_executor *executor = nullptr;
    cere::host::ModelPlan plan;
    std::vector<uint32_t> stop_tokens;
    uint64_t generation = 0;
    cere::host::MemoryEstimate estimate{};
    std::unique_ptr<cere::host::DeviceModel> device;
    std::unique_ptr<cere::host::TensorLedger[]> ledgers;  // one per tensor
    std::atomic<bool> sealed{false};
    std::atomic<uint64_t> puts_in_flight{0};
    std::atomic<uint64_t> uploaded_device_bytes{0};
    std::atomic<uint64_t> uploaded_host_bytes{0};
    std::atomic<uint64_t> discarded_bytes{0};
    uint64_t sessions = 0;
    uint64_t snapshots = 0;                     // live cere_snapshot handles in the arena
    cere::host::ArenaAllocator arena;           // over device->arena (plan.snapshot_host_bytes)
    std::array<uint8_t, 32> layout_id{};        // CERESNP1 layout_id of this model's config (D3)
    // M2.10 (m210 design note D1, D3′): the model's prefix tree (null with prefix_state_slots 0) and the pages
    // reserved for lazy sessions not yet bound (D3′ admission: kv_pages(context) per such session).
    std::unique_ptr<cere::host::PrefixCache> prefix;
    uint64_t unbound_pages = 0;
    // m210 D15⁗′ stage 2a: drafting PREFILLs capture end states (cere_model_set_prefix_end_state; off by default).
    // `submitted`: a ticket of the model was submitted; the setting is fixed from then on (BUSY).
    bool prefix_end_state = false;
    bool submitted = false;
    // O95 / m23 design note SC-01: the fixed tables that price every draft-policy round (m29a design note A3;
    // batch-concurrency design note BC-D4: sampled and non-adaptive tickets too), by k2_gemm (CERE_LEVER_K2_CUBLASLT,
    // CERE_LEVER_K2_SMALL_M; CERE_LEVER_K2_SMALL_M_A8 uses the small_m table, K2Prefill P3), set at seal for the life
    // of the load: measured on this library (host/spec_calibrate.h) when `round_costs_measured`, else
    // SpecCosts::untimed().
    std::array<cere::host::SpecCosts, 2> round_costs{cere::host::SpecCosts::untimed(),
                                                     cere::host::SpecCosts::untimed()};
    bool round_costs_measured = false;
    // Track LoRA (lora design note): the adapter pool of cere_model_lora_reserve (0 adapters = none), its device
    // bytes (counted in reserved_device_bytes), and per slot whether it holds an adapter and its live sessions.
    uint32_t lora_adapters = 0;
    uint32_t lora_rank = 0;
    uint64_t lora_bytes = 0;
    std::vector<bool> lora_loaded;
    std::vector<uint64_t> lora_sessions;
    // WeightLab research (weightlab design note WL-02; cere_model_research_*): the reserve (0 plans = none), its
    // device bytes (staging + plan memory, counted in reserved_device_bytes), the device's host ring and its
    // allocator, per plan slot the loaded plan (null: free) and its live sessions, and the captures by id (a session's
    // capture is listed from cere_session_set_research until cere_model_research_capture_free).
    uint32_t research_plans = 0;
    uint64_t research_bytes = 0;
    uint8_t *research_ring = nullptr;
    cere::host::ArenaAllocator research_ring_alloc;
    std::vector<std::shared_ptr<const cere::research::Plan>> research_plan;
    std::vector<uint64_t> research_sessions;
    std::map<uint64_t, std::shared_ptr<cere::research::Binding>> research_captures;
    uint64_t research_next_capture = 1;
    // Guards the research members above after the reserve (plan load/unload, set_research, session release,
    // capture read/free may run on different threads; the forward reads only a session's binding).
    std::mutex research_mutex;
};

struct cere_session {
    cere_model *model = nullptr;
    uint64_t max_context_tokens = 0;
    std::vector<uint32_t> tokens;  // prompt + committed tokens
    uint64_t fed = 0;              // tokens already consumed by the device session
    uint64_t flags = 0;            // cere_work.flags of the session's PREFILL (restored from a snapshot)
    uint64_t prompt_tokens = 0;    // tokens of the PREFILL prompt; the session decodes once fed reaches it
    uint64_t revision = 0;         // incremented by every terminal poll
    uint64_t in_flight = 0;        // submission ID of the outstanding ticket, 0 when idle
    // The PREFILL prompt's images, in prompt order; released once the prompt is consumed (M2.13-L: only an open
    // prompt's steps read them).
    std::vector<cere::host::SessionImage> images;
    std::vector<int64_t> positions3;  // mRoPE (D11) of the prompt, [3][prompt length]; empty without images
    // mRoPE (D11): every row past the prompt rotates at position + rope_delta (Device::set_rope_delta); 0 without
    // images.
    int64_t rope_delta = 0;
    // An image session: its PREFILL had images, or it was restored from a snapshot with kind 11 (rope_delta only:
    // a restored session is past its prompt, so images, positions3 and keys stay empty; M2.13 C14, ledger CR14).
    bool image_session = false;
    // M2.13-L (m213l design note D-L5, contract C6): an image session's prefix-cache key of `tokens` row by row
    // (each pad row j of an image is 0x80000000 | (word[j % 8] & 0x7fffffff), word[] the 8 big-endian u32 of
    // SHA-256(pixel_sha256 || grid_t || grid_h || grid_w as LE u32); every other row its token id), kept in step with
    // `tokens`. Empty for a text session (its key is `tokens`) and for a restored image session (no cache).
    std::vector<uint32_t> keys;
    // M2.9: the session drafts (set once from its PREFILL, or from a snapshot's kind 10): it decodes only through
    // Device::speculate_rows.
    bool drafting = false;
    // GdFix (grammar-spec design note F10, O238): its PREFILL carried a vocab mask (a constrained ticket; one that
    // drafts under grammar_drafts "on", GS-04). Set with `drafting`; false for a restored session.
    bool masked_prompt = false;
    // D15‴ (ledger CR8): a drafting session runs the cache-off call layout on a cache-on model (no split points, hits
    // only at draft states). A masked-prompt one runs the non-drafting layout instead (lookups, D4 splits, captures,
    // its decode captures included), so its greedy tokens are those of the same ticket under NO_SPECULATION.
    bool one_call_layout() const { return drafting && !masked_prompt; }
    // D15⁗ (ledger CR8): a one-call-layout session that looked up on a cache-on model, so it runs the cache-off call
    // layout from position 0 (a miss) or from a draft state's chunk multiple: its prompt captures at prefill-chunk
    // multiples are draft states (host/prefix_cache.h PrefixNode::draft).
    bool draft_layout = false;
    // m210 D15⁗′: its hit (`locked`) is an end state (PrefixNode::cls), resumed by its next prefill step.
    bool end_resume = false;
    bool submitted = false;        // a work item was ever accepted (restore needs a fresh session)
    // M2.9L (D1, D2): the suffix-lookup drafter over `tokens` and the draft policy of this session's requests.
    cere::host::SuffixDrafter drafter;
    cere::host::DraftPolicy policy;
    // m29a design note A5: drafts verified by this session's committed speculation rounds (any source), host-only
    // (no ABI counter): drafts per round = verified_drafts / the rounds' verify_steps.
    uint64_t verified_drafts = 0;
    std::unique_ptr<cere::host::DeviceSession> device;
    // M2.10 (m210 design note D2, D3′, D4 (b), D7). `lazy`: created without pages on a cache-on model; `bound`:
    // its pages are bound (always true for an eager session). A namespaced session (`ns`, set before its first
    // submit) looks up and inserts. `locked`: the node of its prefix hit, locked until release. `track_slot`: its
    // decode track slot (PrefixNode::kNoSlot = none) holding the state at `tracked` (0 = no uninserted capture).
    bool lazy = false;
    bool bound = true;
    bool namespaced = false;
    cere::host::CacheNamespace ns{};
    cere::host::PrefixNode *locked = nullptr;
    uint32_t track_slot = cere::host::PrefixNode::kNoSlot;
    uint64_t tracked = 0;
    // Track LoRA: the adapter slot this session runs with (cere_session_set_lora; -1 = the base model).
    int32_t lora_slot = -1;
    // WeightLab research: the plan this session runs under (cere_session_set_research; null = none).
    std::shared_ptr<cere::research::Binding> research;
};

namespace cere::host {

// The session's prefix-cache key rows (M2.13-L contract C6): `keys` for an image session, else `tokens`.
inline const uint32_t *key_data(const cere_session &session) {
    return session.keys.empty() ? session.tokens.data() : session.keys.data();
}

}  // namespace cere::host

// A snapshot in its model's arena (M2.11): bytes [offset, offset + info.bytes) hold the CERESNP1 image.
struct cere_snapshot {
    cere_model *model = nullptr;
    uint64_t offset = 0;
    cere_snapshot_info info{};
};

// Members are destroyed in reverse order: tickets, snapshots, sessions, models, then the device.
struct cere_executor {
    uint64_t worker_epoch = 0;
    uint64_t memory_budget_bytes = 0;
    std::unique_ptr<cere::host::Device> device;
    // Put staging ranges (cere_executor_register_staging); empty before cere_executor_destroy succeeds.
    cere::host::StagingRanges staging;
    // Held shared by cere_model_put (the one concurrent call), exclusively by calls that add or remove models.
    std::shared_mutex handles;
    std::unordered_map<const cere_model *, std::unique_ptr<cere_model>> models;
    std::unordered_map<const cere_session *, std::unique_ptr<cere_session>> sessions;
    std::unordered_map<const cere_snapshot *, std::unique_ptr<cere_snapshot>> snapshots;
    std::map<uint64_t, std::unique_ptr<cere::host::Ticket>> tickets;
    uint64_t last_generation = 0;
    uint64_t last_submission = 0;
    uint64_t reserved_device_bytes = 0;
    uint64_t reserved_host_bytes = 0;
    // Rounds (M2.6 D2, O79 PF1): the decode rounds per prefill chunk (cere_executor_set_decode_rounds) and the decode
    // rounds still owed before the next chunk while decode-ready tickets remain (each prefill round that ends a run
    // sets it to the count, each decode round takes one: count 1 alternates the kinds), the first ticket of the last
    // decode round (decode rounds rotate over models), and reused scratch (the decode-ready tickets, then one model's
    // tickets and rows with their per-row outputs). BC4 (b): the prefill rounds and their wall time since the last
    // decode round (a run of back-to-back prefill rounds), and the round's prompt tickets in submission order.
    uint32_t decode_rounds_per_chunk = 1;
    uint32_t decode_rounds_owed = 0;
    uint64_t prefill_run_calls = 0;
    int64_t prefill_run_ns = 0;
    std::vector<cere::host::Ticket *> round_prompts;
    // MixedPD: the prefill slice in ms (cere_executor_set_prefill_slice; 0 = off); each paused segment counts as a
    // prefill round for decode_rounds_owed.
    uint32_t prefill_slice_ms = 0;
    // BC4 (b): the aging cap in prefill calls (cere_executor_set_prefill_age): a prompt ticket overtaken this many
    // times by younger tickets' calls runs the next call.
    uint32_t prefill_age_calls = CERE_DEFAULT_PREFILL_AGE_CALLS;
    // S1 G4 (beat-sglang design note): the decode share in percent (cere_executor_set_decode_share; 0 = off, the
    // round policy above) and the decode wall time owed in ns (prefill rounds of the last waiting prompt add to it,
    // decode rounds take theirs off).
    uint32_t decode_share_pct = 0;
    int64_t decode_owed_ns = 0;
    // S1 G3: the rounds cere_executor_poll ran (a ticket whose seen_round is below it has a report pending).
    uint64_t rounds = 0;
    uint64_t decode_cursor = 0;
    std::vector<cere::host::Ticket *> round_ready;
    std::vector<cere::host::Ticket *> round_tickets;
    std::vector<cere::host::RowArgs> round_rows;
    std::vector<cere_token_score> round_scores;
    std::vector<cere_token_logprobs> round_logprobs;
    std::vector<cere_status> round_status;
    // M2.9: the round's drafting tickets, their speculate_rows rows and results (logprobs: 4 entries per row).
    std::vector<cere::host::Ticket *> spec_tickets;
    std::vector<cere::host::SpecRowArgs> spec_rows;
    std::vector<cere::host::SpecRowResult> spec_results;
    std::vector<cere_token_logprobs> spec_logprobs;
    std::vector<cere_status> spec_status;
};

namespace cere::host {

// D3/D3′ apply at a lazy session's first prefill round (or before an M2.11 restore, with no prompt): for a namespaced
// text session with a fresh prompt in tokens[0, prompt_tokens), looks up the deepest stated node within
// floor64(prompt_tokens - 1) (a host node is loaded back first, D9), locks its path, binds the session's pages (the
// path's, then its own, evicting LRU unlocked leaves first, D7) and loads the node's state, so fed = h with the
// prompt open. Without a hit (or a namespace) it binds own pages only. Sets out->h (the hit, 0 on a miss) and out->k
// (the page-match depth, D4 (d)). A no-op on an eager or already bound session. CERE_ERR_CAPACITY when the pages
// cannot be bound (nothing locked or bound).
struct PrefixAttach {
    uint64_t h = 0;
    uint64_t k = 0;
};
cere_status attach_prefix(cere_executor &executor, cere_session &session, uint64_t clock, PrefixAttach *out);

// The session's prefix-cache holdings at its release (D7): unlocks its hit path, frees its track slot and drops the
// D3′ reservation of a session never bound. Idempotent.
void release_prefix(cere_session &session);

}  // namespace cere::host
