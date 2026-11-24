// CPU toy device: host-memory tensors and a deterministic toy model, next = (31 * last + position) mod vocab, where
// position is the 0-based index of the generated token (the sequence length before it); every row an image session
// scores outside its open prompt (`step` without images, step_rows, speculate_rows, MTP proposals) uses its rotary
// position, that index plus rope_delta (m213 design note D11, Device::set_rope_delta), so a session that loses its
// rope_delta decodes other tokens. An image's pad rows fold the image's toy encoder word (FNV-1a 64 of its grid and
// pixels, xor the row index) into the state digest after their token id, so a state over other pixels or another
// grid at the same token ids has another digest (M2.13-L D-L5: a wrong prefix hit shows). device_id grammar:
// "toy" followed by optional ",host_bytes=N" (pinned-host budget, default unlimited), ",fail_forward=N" (the N-th
// forward call, 0-based, counting one per `step` and one per `step_rows` call that runs a row, returns
// CERE_ERR_INTERNAL and breaks every session it would have stepped: their later steps fail with CERE_ERR_INTERNAL),
// ",hold_after=N" (once a session has generated N tokens its steps, or its row of a `step_rows`, answer CERE_PENDING
// without running, so the work item waits at that token boundary for a cancel; a session that was snapshotted holds
// no more), ",hold_ms=N" (a hold that lasts N ms, default 30000, fails the step or row with CERE_ERR_INTERNAL
// instead), ",hold_count=N" (only the executor's first N holds happen, default unlimited), ",hold_gate=PATH" (an
// absolute path without commas: a session past hold_after that finds PATH gone runs on and holds no more, so a test
// ends the holds by removing PATH), ",snapshot_capacity=N" (the first N snapshots answer CERE_ERR_CAPACITY, session
// and arena unchanged; M2.11 C11), ",restore_gate=PATH" (an absolute path without commas: while PATH exists a restore
// answers CERE_ERR_CAPACITY, session and arena unchanged, and appends one line "refused" to PATH, so a test sees each
// refusal and ends them by removing PATH; M2.11 C11), ",staging_gate=PATH" (an absolute path without commas: while
// PATH exists, for at most hold_ms, a write_staged waits inside the device, staged_holding() counting it, so a test
// holds a put from a registered staging range in flight and ends the hold by removing PATH; m216 design note
// M216S-11), ",tensor_digests=PATH" (an absolute path without commas: INSTANTIATE writes PATH with one line
// "INDEX HEX" per stored tensor, HEX the 16-digit FNV-1a 64 of its bytes, so a test checks the uploaded bytes
// exactly; m216 design note L7),
// ",prefill_chunk=N" (N >= 1, default 4,096 as m22c D4: one `step` consumes at most N tokens and scores only when it
// consumes the rest of its tokens), ",end_kv_slots=K" (K >= 1; PrefixCapture G1: Device::end_kv_slots answers K for
// every chunk-0 end state, so a drafting prompt below the prefill chunk captures one; absent = 0, none),
// ",row_sig=LO-HI" (1 <= LO <= HI, repeatable, ranges disjoint; m210 D15⁗″: Device::row_sig gives every class in
// [LO, HI] one row-bit signature, LO, and every other class its own, the class; absent = σ = class everywhere),
// ",prefill_sig=N" (N >= 1; the prefill selection signature, cere_model_prefill_signature: Device::prefill_selection
// adds the line "toy bucket N" after "toy prefill-selection v1", as if one pinned bucket changed; absent = that line
// alone),
// ",sampler=0" (the device reports no sampler, so capabilities `native_sampler` and
// `native_token_mask` are 0 and sampled or constrained work is UNSUPPORTED) and ",record=PATH" (an absolute path
// without commas: every scored step, and every row of a `step_rows`, appends one line of numbers, "session=S count=N"
// (S numbers sessions from 1 in creation order, N = tokens fed by the call), then "sampling=null" or the sampled
// item's "temperature= top_k= top_p= min_p= frequency= presence= repetition= logit_bias=<count> seed=", then
// "logprobs=null" or "logprobs=<top_logprobs>", then for a constrained item (M2.8) "mask=<allowed ids of the row's
// mask>", then "token=<committed id>"). A step that does not fit the session's context is CERE_ERR_CAPACITY. Built
// with CERE_TOY_NO_FORWARD it reports no forward pass; built with CERE_TOY_PRE_STAGING the library lacks the two put
// staging exports (cere_executor_register_staging/unregister_staging), cere_executor_set_decode_rounds,
// cere_model_prefix_stats, cere_executor_set_prefill_slice, the two GrammarSpec exports
// (cere_executor_round_drafts/constrain_drafts), the four LoRA exports (cere_model_lora_reserve/load/unload,
// cere_session_set_lora), the six WeightLab research exports (cere_model_research_*, cere_session_set_research) and
// cere_model_prefix_capture_stats, as a library from before them. A step with sampling
// (M2.7) scores toy_logits(next) with the CPU sampler reference (host/sampler_ref.h), so a neutral greedy item still
// commits `next`; a `step_rows` row with sampling is scored the same way with its session's own counts and position
// (ledger CR4). Under an item's vocab mask (M2.8 D9) a masked `next` becomes the smallest allowed id above
// it, wrapping, and the sampler stage applies the mask too, so a constrained item always commits an allowed id.
// ",host_share=DIR,artifact=SHA256" (both or neither; host/host_share.h) puts the HOST_PINNED tensors in the shared
// segment of that artifact, packed in desc order at 256-byte aligned offsets (layout tag "toy-v1"); DEVICE tensors
// stay private and allocated_host_bytes() still counts the region. ",release_device=1" is accepted as on the CUDA
// device (device/cuda_device.h, m32n design note D10) and changes nothing: the toy holds no device context.
// ",create_busy=N" (m32n design note D21): the executor's first N ALLOCATEs answer CERE_ERR_BUSY with nothing kept,
// as a create of a shared segment that another model of the process is still creating does (host/host_share.cpp),
// so a test sees how the runtime treats a busy create that no sibling replica causes.
// ",create_capacity_delay_ms=N" (build/page-wait-wake): a cere_session_create whose pages the toy's pool cannot cover
// sleeps N ms (on the caller's thread) before it answers CERE_ERR_CAPACITY with nothing kept, so a test can land a
// cancel inside the host's create round trip; the host's own refusals (max_sessions, context, lazy admission) and a
// create that fits are not delayed. A test-only option: the CUDA device's parse_device_id refuses it as it refuses
// every option but release_device and the share pair (CERE_ERR_ARGUMENT).
// ",prefill_segments=N" (N >= 2; MixedPD, mixed-pd design note): on a model whose plan has prefill_park, a step
// with a StepSlice (budget_ms > 0) pauses N - 1 times (StepSlice::paused, nothing consumed) before it runs whole, as
// the CUDA device's layer segments do; another session's sliced step while one is paused is CERE_ERR_ARGUMENT.
#pragma once

#include "host/device.h"
#include "host/host_share.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace cere::toy {

uint32_t next_token(uint32_t last, uint64_t position, uint32_t vocab);

// Finite toy logits per sampled step (kToySupport ids; every other id is -inf).
constexpr uint32_t kToySupport = 8;
// The logits of a sampled step whose greedy token is `next` (BF16 bits, `vocab` entries): kToySupport - j at id
// (next + j) mod vocab for j < kToySupport, -inf elsewhere.
std::vector<uint16_t> toy_logits(uint32_t next, uint32_t vocab);

// M2.10 (m210 design note D4, ledger M4): the toy's fixed state, the analogue of the recurrent state a prefix
// checkpoint carries: tokens consumed, the last one, and a running FNV-1a digest of every consumed token id (so two
// sessions agree exactly when they consumed the same sequence, however it was split or restored). The logits do not
// read it. Sessions and state slots hold one; a CERESNP1 image holds the digest as u64 LE in the first 8 bytes of
// kind 7 (PLE: 16 bytes or more for every config the toy accepts), the rest of the device sections zero.
constexpr uint64_t kDigestSeed = 0xcbf29ce484222325ull;
constexpr uint64_t digest_step(uint64_t digest, uint32_t token) { return (digest ^ token) * 0x100000001b3ull; }
struct ToyState {
    uint64_t length = 0;
    uint32_t last = 0;
    uint64_t digest = kDigestSeed;
    bool valid = false;  // a slot that holds a captured or loaded state
    uint64_t hash = 0;   // M2.9 ",stateful=1": the rolling hash of the consumed tokens (0 otherwise)
    // m210 D15⁗′ stage 2a: an end state (captured inside a drafting PREFILL step through an EndStep) holds this
    // slot and the next one (left invalid) and records the class of its cold call (0 = a plain state).
    uint64_t cls = 0;
};

// tests/fixtures/model_src/config.json text_config mapped by the cere_qwen4_exp_config rule.
cere_qwen4_exp_config fixture_config();
// The fixture model's support entry (the toy also accepts small_vocab_support(), cere::host::a4_support() and
// a4_vision_support()).
const cere::host::Support &fixture_support();
// The fixture config with vocab_size kSmallVocab, support "toy_small_vocab" (M2.9 C9: test_spec_sampled_distribution
// samples 200,000 4-token continuations, each row an O(vocab) sampler pass, and checks them against the analytic
// plain distribution, which needs a vocab small enough to enumerate).
constexpr uint32_t kSmallVocab = 64;
const cere::host::Support &small_vocab_support();

// M2.9 options of the toy's draft (m29 design note I7): ",draft=0" (no draft support: a desc with
// speculative_draft_tokens 4 is UNSUPPORTED), ",draft_miss=K" (K in {1, 2, 3}: the K-th MTP proposal of every chain is
// wrong, the next id after the true one; absent = proposals never miss), ",stateful=1" (the target rule adds a
// rolling hash of every consumed token, so a session that consumed a wrong token decodes other tokens; all models) and
// ",cycle=P" (P in [2, kSmallVocab]; m29l design note C3L: the target before the hash is (last + 1) mod P, so the
// output repeats and suffix-lookup drafts match it) and ",chain=N" (N in [4, kMaxSpecChainDrafts]; DraftTail: the
// model's max_chain_drafts, so a row's caller drafts (a CERE_WORK_TAIL_DRAFTS lookup tail) may be N long; absent = 3,
// the Device default; MTP proposals stay 3, and pending_drafts reports them).
// A drafting session's proposals are the greedy target chain after its last scored token (made by the scoring `step`
// and by every round's draft-extend). speculate_rows runs each row's verify rows one after the other on the session
// (row i consumes the committed token before it), so rows past a rejection never touch the session; greedy accepts a
// draft equal to the plain choice, sampled follows D6 (accept coin channels (4, 5), resample with the draft removed
// channels (6, 7), key = the row's position; the bonus row is the plain draw). It writes one record= line per
// committed token. fail_forward counts a speculate_rows call that runs a row twice: its verify launch (a failure
// there consumes nothing) and then its draft-extend launch (a failure there keeps the committed tokens in the
// results); both break every participant. A drafting session's snapshot carries kind 10: zero MTP parts, then its
// pending proposals. D15⁗ (ledger CR8), on a model with drafts and a prefix cache: the pool records per row the token
// it holds and the next id of its MTP row (written by a drafting session's prefill steps, the MTP extend of the real
// device); a drafting session whose rows below its scoring step's position disagree (it hit a state it did not draft
// over, or one whose MTP row h - 1 took another token h) gets every proposal of that step wrong.
struct DraftOptions {
    bool draft = true;
    uint32_t miss = 0;  // 0 = never
    bool stateful = false;
    uint32_t cycle = 0;  // 0 = next_token
    uint32_t chain = 0;  // 0 = 3 drafts per chain
};

class ToyDevice final : public cere::host::Device {
public:
    ToyDevice(uint64_t host_budget_bytes, uint64_t fail_forward, uint64_t hold_after, uint64_t hold_ms,
              uint64_t hold_count, std::string hold_gate, uint64_t snapshot_capacity, std::string restore_gate,
              bool sampler, std::string record, uint64_t prefill_chunk, cere::host::ShareConfig share,
              DraftOptions draft = {}, std::string staging_gate = {}, std::string tensor_digests = {});
    // fail_sticky=1 (m32n design note R10): every forward call after the fail_forward one fails too, a sticky
    // device fault.
    void set_fail_sticky(bool sticky) { fail_sticky_ = sticky; }
    // The sticky fault has fired (the fail_forward call has run): DM-2, the host refuses the device's later calls.
    bool faulted() const override { return fail_sticky_ && steps_.load() > fail_forward_; }
    // create_busy=N (m32n design note D21): the first N `allocate` calls answer CERE_ERR_BUSY.
    void set_create_busy(uint64_t n) { create_busy_ = n; }
    // create_capacity_delay_ms=N (build/page-wait-wake): a create_session whose pages the pool cannot cover waits N ms
    // before it answers CERE_ERR_CAPACITY.
    void set_create_capacity_delay_ms(uint64_t ms) { create_capacity_delay_ms_ = ms; }
    // prefill_segments=N (MixedPD): a sliced step's segments (1 = it never pauses).
    void set_prefill_segments(uint64_t n) { prefill_segments_ = n; }
    void set_end_kv_slots(uint32_t k) { end_kv_slots_ = k; }
    // row_sig=LO-HI (m210 D15⁗″): the [LO, HI] class ranges that share one row-bit signature.
    void set_row_sig(std::vector<std::pair<uint64_t, uint64_t>> ranges) { row_sig_ = std::move(ranges); }
    // prefill_sig=N (cere_model_prefill_signature): the "toy bucket N" line of prefill_selection (0 = none).
    void set_prefill_sig(uint64_t n) { prefill_sig_ = n; }
    // timed_rounds=1 (O95, m23 design note SC-01): times_spec_rounds() is true, so cere_model_seal measures the
    // adaptive cost tables of a drafting model on the toy (host/spec_calibrate.h) as on the CUDA device.
    void set_times_spec_rounds(bool timed) { times_spec_rounds_ = timed; }
    bool times_spec_rounds() const override { return times_spec_rounds_; }

    std::vector<const cere::host::Support *> supported() const override;
    uint64_t host_budget_bytes() const override;
    cere::host::MemoryEstimate estimate(const cere::host::ModelPlan &plan) const override;
    cere_status allocate(const cere::host::ModelPlan &plan, std::unique_ptr<cere::host::DeviceModel> *out) override;
    cere_status write(cere::host::DeviceModel &model, uint64_t tensor_index, uint64_t offset, const void *bytes,
                      uint64_t len) override;
    cere_status host_resident(const cere::host::DeviceModel &model, uint64_t tensor_index,
                              const uint8_t **out) const override;
    // M216S-11 put staging: the toy has no DMA, so a staged write is the same memcpy, counted (and held by
    // staging_gate) so tests see which path a put took.
    cere_status register_staging(const uint8_t *base, uint64_t bytes) override;
    void unregister_staging(const uint8_t *base, uint64_t bytes) override;
    cere_status write_staged(cere::host::DeviceModel &model, uint64_t tensor_index, uint64_t offset, const void *bytes,
                             uint64_t len) override;
    cere_status instantiate(cere::host::DeviceModel &model) override;
    // ABI v5 slot selection: the toy is a p3b build (the shipped MoE backend), so moe_gemm BUILD and P3B are accepted
    // and K7 is UNSUPPORTED like a p3b CUDA library's. It has no kernels to select: k2_gemm, prefill_attention,
    // moe_gemm and k2_weights (the toy's weights stay as put; the host refuses a set_selection change of it) are
    // stored (levers() reads them back) and change nothing; draft_tokens is the host's (spec_row caps each round's
    // drafts at the window - 1, which the toy's speculate_rows honours as any draft_count).
    uint32_t moe_lever() const override { return CERE_LEVER_MOE_P3B; }
    cere_status set_levers(cere::host::DeviceModel &model, const cere::host::Levers &levers) override;
    bool forward_supported() const override;
    bool sampler_supported() const override;
    cere_status create_session(cere::host::DeviceModel &model, uint64_t max_context_tokens, bool lazy,
                               std::unique_ptr<cere::host::DeviceSession> *out) override;
    cere_status step(cere::host::DeviceModel &model, cere::host::DeviceSession &session, const uint32_t *tokens,
                     uint64_t count, uint64_t limit, const cere::host::StepImages *images, uint64_t *consumed,
                     const cere::host::ItemSampling *sampling, cere_token_score *out, cere_token_logprobs *logprobs,
                     const cere::host::EndStep *end, cere::host::StepSlice *slice) override;
    cere_status step_rows(cere::host::DeviceModel &model, const cere::host::RowArgs *rows, uint32_t row_count,
                          cere_token_score *out, cere_status *row_status) override;
    // M2.9: a plan with drafts allocates only with ",draft" on; its sessions draft when the host says so.
    bool draft_supported(const cere::host::DeviceModel &model) const override;
    cere_status set_session_draft(cere::host::DeviceModel &model, cere::host::DeviceSession &session, bool on) override;
    // M2.13-L: the rotary offset of the session's rows past its prompt (ToySession::rope_delta).
    cere_status set_rope_delta(cere::host::DeviceModel &model, cere::host::DeviceSession &session,
                               int64_t rope_delta) override;
    cere_status speculate_rows(cere::host::DeviceModel &model, const cere::host::SpecRowArgs *rows, uint32_t row_count,
                               cere::host::SpecRowResult *out, cere_status *row_status) override;
    // DraftTail: ",chain=N"'s N, else the Device default 3.
    uint32_t max_chain_drafts(const cere::host::DeviceModel &model) const override;
    // DraftTail: a drafting session's pending proposals (host copies), as the CUDA device's.
    uint32_t pending_drafts(const cere::host::DeviceSession &session, uint32_t *ids, uint32_t max) const override;
    // Track LoRA: an adapter shifts every id a bound session scores by its digest's offset (toy_device.cpp).
    uint64_t lora_bytes(const cere::host::DeviceModel &model, uint32_t adapters, uint32_t rank) const override;
    cere_status lora_reserve(cere::host::DeviceModel &model, uint32_t adapters, uint32_t rank) override;
    cere_status lora_load(cere::host::DeviceModel &model, uint32_t slot, uint32_t rank, float scale, uint64_t count,
                          const uint32_t *targets, const void *const *a, const void *const *b) override;
    cere_status lora_unload(cere::host::DeviceModel &model, uint32_t slot) override;
    cere_status set_session_lora(cere::host::DeviceModel &model, cere::host::DeviceSession &session,
                                 int32_t slot) override;
    // WeightLab research (weightlab design note WL-02): the toy validates plans as the CUDA library (research/plan.h,
    // the shared point table) and every consumed token of a bound session writes its capture rows
    // (research::write_toy_row); intervention ops are accepted and apply no edit. The staging and plan memory count
    // in allocated_device_bytes, the ring (a host buffer) in allocated_host_bytes.
    bool research_supported() const override { return true; }
    cere_status research_reserve(cere::host::DeviceModel &model, uint64_t staging_bytes, uint64_t plan_bytes,
                                 uint64_t host_bytes, uint8_t **ring) override;
    cere_status research_plan_load(cere::host::DeviceModel &, uint32_t, const cere::research::Plan &) override {
        return CERE_OK;
    }
    cere_status set_session_research(cere::host::DeviceModel &model, cere::host::DeviceSession &session,
                                     const cere::research::Binding *binding) override;
    // M2.11: the toy keeps no KV or recurrent bytes, so its device sections are zero except kind 8 (the sampler
    // counts), kind 10 (M2.9 proposals) and, since M2.10, the ToyState digest in kind 7; load rebuilds length and last
    // from kind 0 (m211 design note Interfaces "Toy implements all three").
    bool snapshot_supported() const override { return true; }
    cere_status save_state(cere::host::DeviceModel &model, cere::host::DeviceSession &session,
                           const cere::host::snapshot::Plan &plan, uint64_t position, uint8_t *dst) override;
    cere_status load_state(cere::host::DeviceModel &model, cere::host::DeviceSession &session,
                           const cere::host::snapshot::Plan &plan, uint64_t position, const uint8_t *src) override;
    // M2.10 (host/device.h "M2.10 prefix cache"): the refcounted pool, lazy sessions and ToyState slots; save/load
    // prefix write and read the M2.11 sections as save/load_state do (no KV bytes: `pages` are only checked).
    bool prefix_supported() const override { return forward_supported(); }
    uint64_t free_pages(const cere::host::DeviceModel &model) const override;
    uint32_t page_refs(const cere::host::DeviceModel &model, uint32_t id) const override;
    cere_status retain_pages(cere::host::DeviceModel &model, const uint32_t *ids, uint64_t n) override;
    cere_status release_pages(cere::host::DeviceModel &model, const uint32_t *ids, uint64_t n) override;
    cere_status take_pages(cere::host::DeviceModel &model, uint64_t n, std::vector<uint32_t> *out) override;
    cere_status bind_pages(cere::host::DeviceModel &model, cere::host::DeviceSession &session,
                           const uint32_t *prefix, uint64_t n_prefix) override;
    std::vector<uint32_t> session_page_list(const cere::host::DeviceSession &session) const override;
    uint64_t session_position(const cere::host::DeviceSession &session) const override;
    uint64_t prefill_chunk(const cere::host::DeviceModel &) const override { return prefill_chunk_; }
    uint32_t end_kv_slots(const cere::host::DeviceModel &, uint64_t, uint64_t) const override { return end_kv_slots_; }
    uint64_t row_sig(const cere::host::DeviceModel &, uint64_t s, uint64_t cls) const override;
    bool prefill_selection(const cere::host::DeviceModel &, std::string &out) const override;
    cere_status capture_state(cere::host::DeviceModel &model, const cere::host::DeviceSession &session,
                              uint32_t slot) override;
    cere_status load_slot(cere::host::DeviceModel &model, cere::host::DeviceSession &session, uint32_t slot,
                          uint64_t position) override;
    cere_status save_prefix(cere::host::DeviceModel &model, const uint32_t *pages, uint64_t n, uint32_t slot,
                            const cere::host::snapshot::Plan &plan, uint64_t position, uint8_t *dst) override;
    cere_status load_prefix(cere::host::DeviceModel &model, const uint32_t *pages, uint64_t n, uint32_t slot,
                            const cere::host::snapshot::Plan &plan, uint64_t position, const uint8_t *src) override;

    // Introspection for tests: live allocated bytes and buffers, and a copy of one tensor (empty for DISCARD).
    uint64_t allocated_device_bytes() const { return device_bytes_.load(); }
    uint64_t allocated_host_bytes() const { return host_bytes_.load(); }
    uint64_t allocations() const { return allocations_.load(); }
    // Put staging: registered ranges, staged writes (count and bytes), staged writes waiting on staging_gate now.
    uint64_t staging_ranges() const { return staging_ranges_.load(); }
    uint64_t staged_puts() const { return staged_puts_.load(); }
    uint64_t staged_bytes() const { return staged_bytes_.load(); }
    uint64_t staged_holding() const { return staged_holding_.load(); }
    // M2.13-L (m213l design note D-L3): `step` calls that consumed rows of an image (one per image per call, as the
    // CUDA device encodes each image once per prefill call that holds some of its pad rows) and the pad rows they
    // consumed, counted since the device was made.
    uint64_t image_encodes() const { return image_encodes_.load(); }
    uint64_t image_rows() const { return image_rows_.load(); }
    std::vector<uint8_t> tensor(const cere::host::DeviceModel &model, uint64_t tensor_index) const;
    // The selection the device holds for `model` (the desc's at ALLOCATE, then every set_levers).
    cere::host::Levers levers(const cere::host::DeviceModel &model) const;
    // O95 (m23 design note SC-01): the speculate_rows rows launched on any model whose k2_gemm lever is `k2_gemm`
    // (0 cuBLASLt, 1 small_m or small_m_a8) with verify window `window` (draft_count + 1, 1..10), counted since the
    // device was made; spec_graph_rounds: those launched while the model's decode_step lever was not
    // CERE_LEVER_DECODE_EAGER.
    uint64_t spec_rounds(uint32_t k2_gemm, uint32_t window) const;
    uint64_t spec_graph_rounds() const { return spec_graph_rounds_.load(); }
    // The model's free KV pool pages and a session's page ids (m25 design note D3).
    std::vector<uint32_t> session_pages(const cere::host::DeviceSession &session) const;
    // M2.10: a session's fixed state and a state slot's (tests compare a prefix hit's state with the cold run's).
    ToyState session_state(const cere::host::DeviceSession &session) const;
    ToyState slot_state(const cere::host::DeviceModel &model, uint32_t slot) const;

private:
    friend class ToyModel;
    // Appends the step's record= line (CERE_OK when there is no record); `sampling`/`logprobs` as given to step, `mask`
    // the scored row's vocab mask (null = none; a speculation row's verify row j: its j-th mask).
    cere_status record(uint64_t session, uint64_t count, const cere::host::ItemSampling *sampling, const uint32_t *mask,
                       const cere_token_logprobs *logprobs, uint32_t vocab, uint32_t token);
    // The shared region of `plan` (sharing on): offsets[i] per HOST_PINNED tensor (desc order, 256-byte aligned),
    // the region bytes (UINT64_MAX on overflow, 0 without HOST_PINNED tensors) and the segment name.
    uint64_t region(const cere::host::ModelPlan &plan, std::vector<uint64_t> *offsets, std::string *name) const;
    // Counts one forward call; true when it fails (the fail_forward call, and every later one with fail_sticky).
    bool fails_forward();
    uint64_t host_budget_bytes_;
    uint64_t fail_forward_;
    bool fail_sticky_ = false;
    uint64_t create_busy_ = 0;
    uint64_t create_capacity_delay_ms_ = 0;
    uint64_t prefill_segments_ = 1;
    uint32_t end_kv_slots_ = 0;  // ",end_kv_slots=K"
    std::vector<std::pair<uint64_t, uint64_t>> row_sig_;  // ",row_sig=LO-HI"
    uint64_t prefill_sig_ = 0;  // ",prefill_sig=N"
    std::atomic<uint64_t> creates_{0};  // allocate calls so far (create_busy)
    bool times_spec_rounds_ = false;
    std::atomic<uint64_t> spec_rounds_[2][11] = {};  // spec_rounds: [k2_gemm][window]
    std::atomic<uint64_t> spec_graph_rounds_{0};
    uint64_t hold_after_;
    uint64_t hold_ms_;
    uint64_t hold_count_;
    std::string hold_gate_;  // the hold_gate= path; empty = holds end only by a cancel, a snapshot or hold_ms
    uint64_t snapshot_capacity_;
    std::string restore_gate_;  // the restore_gate= path; empty = restores are never refused
    bool sampler_;
    std::string record_;  // the record= path; empty = no record
    std::mutex record_mutex_;
    std::atomic<uint64_t> sessions_{0};
    uint64_t prefill_chunk_;
    std::atomic<uint64_t> steps_{0};
    std::atomic<uint64_t> snapshots_{0};
    std::atomic<uint64_t> holds_{0};
    std::atomic<uint64_t> device_bytes_{0};
    std::atomic<uint64_t> host_bytes_{0};
    std::atomic<uint64_t> allocations_{0};
    cere::host::ShareConfig share_;  // sharing off when share_.enabled() is false
    DraftOptions draft_;
    std::string staging_gate_;  // the staging_gate= path; empty = staged writes never wait
    std::string tensor_digests_;  // the tensor_digests= path; empty = INSTANTIATE writes no digests
    std::atomic<uint64_t> staging_ranges_{0};
    std::atomic<uint64_t> staged_puts_{0};
    std::atomic<uint64_t> staged_bytes_{0};
    std::atomic<uint64_t> staged_holding_{0};
    std::atomic<uint64_t> image_encodes_{0};
    std::atomic<uint64_t> image_rows_{0};
};

}  // namespace cere::toy
