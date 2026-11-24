// Device interface of the host layer: the state machine in state.h/api.cpp drives one Device per executor. The CPU toy
// (executor/cuda/src/toy/) and the CUDA device implement it; no CUDA include appears here.
#pragma once

#include "cere_executor.h"
#include "host/mrope.h"
#include "host/snapshot_format.h"
#include "host/support.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cere::research {
struct Plan;
struct Binding;
}  // namespace cere::research

namespace cere::host {

// One tensor of cere_model_desc.tensors, in desc order (tensor_index is the index into ModelPlan::tensors).
struct TensorPlan {
    std::string name;
    uint32_t dtype;      // CERE_DTYPE_*
    uint32_t placement;  // CERE_PLACEMENT_*
    uint32_t ndim;
    uint64_t shape[CERE_TENSOR_MAX_DIMS];  // entries at and after ndim are 0
    uint64_t bytes;
};

// ABI v5 (SelfTune ST2): a model's resolved slot selection, one field per CERE_SLOT_* slot holding its impl's
// CERE_LEVER_* value (draft_tokens: its window param, 0 = the model's speculative_draft_tokens). host/api.cpp
// resolve_selection fills it from a validated cere_slot_selection, every slot not listed at its default (all 0).
struct Levers {
    uint32_t k2_gemm = CERE_LEVER_K2_CUBLASLT;
    uint32_t prefill_attention = CERE_LEVER_PREFILL_CUDA_CORE;
    uint32_t decode_step = CERE_LEVER_DECODE_EAGER;
    uint32_t moe_gemm = CERE_LEVER_MOE_BUILD;
    uint32_t draft_tokens = 0;
    uint32_t verify_attention = CERE_LEVER_VERIFY_PER_ROW;
    uint32_t k2_weights = CERE_LEVER_K2_WEIGHTS_BF16;
};

// The speculation round-cost table (index of the model's round_costs) a k2_gemm value prices by: 1 for small_m and
// small_m_a8 (whose m <= 16 kernels are small_m's; K2Prefill P3), 0 for cublaslt and tail_split (VulkanAmd P4-B: its
// m <= 16 calls are cublaslt's).
constexpr uint32_t k2_cost_table(uint32_t k2_gemm) {
    return k2_gemm == CERE_LEVER_K2_SMALL_M || k2_gemm == CERE_LEVER_K2_SMALL_M_A8 ? 1 : 0;
}

// A validated cere_model_desc: the config matched `support`, every tensor is well formed and allowed.
struct ModelPlan {
    const Support *support;
    cere_qwen4_exp_config config;
    std::vector<TensorPlan> tensors;
    uint64_t max_sessions;
    uint64_t max_context_tokens;
    uint64_t snapshot_host_bytes = 0;  // M2.11 snapshot arena (cere_model_desc.snapshot_host_bytes, 0 = none)
    uint64_t kv_pool_tokens;  // the paged KV pool allocated (host/kv_pool.h plan_kv_pool): a positive multiple of 64
    uint64_t prefix_state_slots = 0;  // M2.10 D6 (cere_model_desc.prefix_state_slots, 0 = cache off)
    // MixedPD (mixed-pd design note): the executor's prefill slice was set (cere_executor_set_prefill_slice) when the
    // model was created: the device keeps a park region for a paused prefill call (StepSlice).
    bool prefill_park = false;
    // M2.9 (spec I1, I2): cere_model_desc.speculative_draft_tokens (0 or 4) and its copied draft_token_map (strictly
    // increasing ids < vocab, 1 <= size <= vocab; empty iff 0). The draft tensors are the plan's `mtp.*` tensors.
    uint32_t speculative_draft_tokens = 0;
    std::vector<uint32_t> draft_token_map;
    // ABI v5: the active selection, validated and resolved (host/api.cpp resolve_selection) from
    // cere_model_desc.selection (null = every slot's default); replaced by cere_model_set_selection between requests.
    // The host reads draft_tokens (spec_row's verify window); the device resolves the kernel selections at
    // INSTANTIATE and in set_levers.
    Levers levers;
    // cere_model_desc.calibrate_windows (CERE_CALIBRATE_*): the seal's calibration depth (host/spec_calibrate.h),
    // fixed at create.
    uint32_t calibrate_windows = CERE_CALIBRATE_ALL;
#if CERE_GEN_FAMILY == CERE_FAMILY_SDAR
    // SdarOptIn (O320): the CERE_SLOT_SDAR_GEMM choice (CERE_LEVER_SDAR_GEMM_*), fixed at create. Only an sdar
    // library has the field, so no other family's plan changes.
    uint32_t sdar_gemm = CERE_LEVER_SDAR_GEMM_IDENTICAL;
#endif
};

struct MemoryEstimate {
    uint64_t device_bytes;
    uint64_t host_bytes;  // pinned (page-locked) host memory, including the snapshot arena
};

// One work item's sampling and logprobs request (m27 design note I8), copied at submit. When `sampled`, `params` is
// the item's cere_sampling_params (its logit_bias pointer cleared; the entries are `bias`). A `sampling == null` item
// that asks for logprobs only has `sampled` false and is scored greedily on the unprocessed logits. `top_logprobs` is
// the item's cere_work.top_logprobs; `vocab_mask` (ceil(vocab/32) host words, bit set = allowed, xgrammar layout) is
// a constrained item's mask for the token being scored (m28 design note D9; null = unconstrained). The host passes
// a constrained item only once that mask is present; the sampler stage sets disallowed ids to -inf after the
// penalties and before logit_bias.
struct ItemSampling {
    bool sampled = false;
    cere_sampling_params params{};
    std::vector<cere_logit_bias> bias;
    uint32_t top_logprobs = 0;
    const uint32_t *vocab_mask = nullptr;
};

// One image of a session's prompt, copied at submit (m213 design note I-ABI, ledger N).
struct SessionImage {
    cere::mrope::ImageSpan span;  // token_offset = its first pad row; grid (t, h, w) in patches
    uint32_t pad_id = 0;          // the PLE id of its pad rows (M2.13 D16)
    // cere_image_input.pixel_sha256 (M2.13-L contract C1): SHA-256 of `pixels`, as the caller gave it (non-zero); the
    // image's prefix-cache key (C6) is derived from it with the grid.
    std::array<uint8_t, 32> pixel_sha256{};
    std::vector<uint8_t> pixels;  // BF16 [t*h*w, patch_elements] in processor order
};

// The images of an image session's PREFILL prompt and its mRoPE positions (m213 design note D10, D11), passed with
// every step of that prompt (null for text sessions and for every step once the prompt is consumed; rows past the
// prompt rotate at position + the session's rope_delta, Device::set_rope_delta).
struct StepImages {
    const SessionImage *image = nullptr;  // in prompt order
    uint64_t count = 0;
    const int64_t *positions3 = nullptr;  // [3][prompt_len] (t, h, w rows; cere::mrope::positions)
    uint64_t prompt_len = 0;
};

// m210 design note D15⁗′ stage 2a: the end work of one PREFILL `step` of a drafting session on a prefix model,
// through end-node pairs of adjacent state slots. `cls` (0 = the step's own row count): the rows of the cold call the
// step's rows belong to; every row-count choice of the step (target and MTP extend) is made for it. Capture (`row` >
// 0: step-relative, a multiple of 64 strictly inside the step) into the pair `slot`, `slot + 1`: the session's fixed
// state at the step start goes to `slot` (capture_state), then the state as of `row` overwrites its GDN, SSM and PLE
// parts, the FP32 SSM of value heads 0..23 in `slot`'s SSM regions and of heads 24..47 in `slot + 1`'s. Resume
// (`resume`, the session's first step after load_slot(in_slot)): the GDN chunk rule starts from the pair `in_slot`,
// `in_slot + 1`'s FP32 SSM. D15⁗″ (ResumeExact): one step may do both, through two disjoint runs. `kv_slots` /
// `in_kv_slots` (PrefixCapture G1; 0 = none): a chunk-0 end state (its cold call starts at position 0, so its dense
// attention reads its own BF16 K/V rows) also holds those rows [0, b) of every attention layer in the slots after its
// pair: a capture writes the rows [0, position + row) there, a resume (the session then at b) attends over them.
struct EndStep {
    uint64_t cls = 0;
    uint64_t row = 0;
    bool resume = false;
    uint32_t slot = 0;
    uint32_t kv_slots = 0;
    uint32_t in_slot = 0;
    uint32_t in_kv_slots = 0;
};

// MixedPD (mixed-pd design note): a PREFILL `step` that may pause between layer segments of its forward call
// (model::forward `segment`) on a model whose plan has prefill_park. In: budget_ms > 0, the segment's wall-time
// budget. Out: `paused` true when the step returned CERE_OK with the call paused (*consumed 0, nothing scored, the
// session's position unchanged). The host then resumes it by repeating the step with the same tokens, count, limit,
// images, sampling and `end` (slice null: the call runs to its end) before any other PREFILL step of the model; any
// decode step (step_rows, speculate_rows, a DECODE `step`) may run between. A device or plan without the park runs
// the call whole (paused false).
struct StepSlice {
    uint32_t budget_ms = 0;
    bool paused = false;
};

// Device-side storage of one model (tensors, pinned host table, workspace). Destroying it releases every byte it
// holds (the host destroys it on cere_model_release, on a failed create, and on cere_executor_destroy, always after
// the model's sessions); destructors never throw.
class DeviceModel {
public:
    virtual ~DeviceModel() = default;
    // The snapshot arena of plan.snapshot_host_bytes in locked host memory, set by `allocate` (null when 0;
    // m211 design note D5).
    uint8_t *arena = nullptr;
};

// Device-side state of one session (KV and recurrent state). Destroying it releases that state (cere_session_release
// or cere_executor_destroy, never while a step is running).
class DeviceSession {
public:
    virtual ~DeviceSession() = default;
};

// One row of a batched decode step (M2.6 I3, internal C++, not ABI; ledger M9): the session, the one token it
// consumes and the row's sampling (ledger CR4). Later orders add members whose zero value means today's behaviour.
// `sampling` null is greedy on the raw logits; otherwise the row is chosen as `step` chooses its last token with that
// sampling (counter key = the session position of `token`, the session's own output counts), and `logprobs` (null or
// one entry; non-null only with `sampling`) receives the committed token's logprobs.
struct RowArgs {
    DeviceSession *session;
    uint32_t token;
    const ItemSampling *sampling = nullptr;
    cere_token_logprobs *logprobs = nullptr;
};

// M2.9 (m29 design note I7 as revised by M29S2-20..22; internal C++, not ABI). The host's stop predicate of one
// ticket: true when `token` finishes it (model or item stop token). Never set for a ticket that ignores stop tokens
// (such tickets carry CERE_WORK_NO_SPECULATION and propose no drafts).
using StopFn = bool (*)(const void *ctx, uint32_t token);

// One drafting session's speculation round (D1, D2, D5): verify row 0 consumes `token` (the last committed, unfed
// token) at the session's position p, rows 1..n consume the drafts at p+1..p+n. With `drafts` null the drafts are
// the session's pending MTP proposals and n = min(draft_count, pending proposals); with `drafts` set (VERIFY, a
// lookup tail after the MTP drafts, or a constrained ticket's masked drafts) they are the caller's and n =
// draft_count (1..Device::max_chain_drafts). draft_count 0 is a plain decode row of a drafting session (a
// NO_SPECULATION ticket, or max_commit 1): still followed by draft-extend, so the MTP state never has holes.
// GrammarSpec (grammar-spec design note GS-02): a constrained row's `sampling->vocab_mask` is row 0 of n + 1
// consecutive masks (ceil(vocab/32) words each; the host passes explicit drafts and n <= CERE_MAX_CONSTRAINED_DRAFTS):
// verify row j is sampled under mask j, as plain decode's row at position p + j would be, so every committed token is
// masked plain decode's (greedy bit-equal; sampled D6 over the masked distributions).
struct SpecRowArgs {
    DeviceSession *session = nullptr;
    uint32_t token = 0;
    const uint32_t *drafts = nullptr;
    uint32_t draft_count = 0;          // <= max_chain_drafts and <= max_commit - 1
    uint32_t max_commit = 1;           // >= 1: min(max_new_tokens - committed, max_context - tokens.size())
    const ItemSampling *sampling = nullptr;  // as RowArgs: null = greedy on the raw logits (phase G); else D6
    cere_token_logprobs *logprobs = nullptr; // null or [draft_count + 1] entries, one per committed token
    StopFn is_stop = nullptr;          // null = no stop token
    const void *stop_ctx = nullptr;
};

// The most drafts any device verifies in one chain (model::layout::kMaxChainDrafts on the CUDA device).
constexpr uint32_t kMaxSpecChainDrafts = 15;

// The outcome of one SpecRowArgs row: the session advanced exactly `committed` positions (a in [1, n + 1]: the
// accepted chain plus the bonus token, truncated after the first stop token and at max_commit, M29S2-21), so the
// device position always equals the host's tokens. scores[0, committed) are the committed tokens in order, each
// bit-equal to what plain decode scores at that position (D4); `accepted` counts the committed drafts.
struct SpecRowResult {
    uint32_t committed = 0;
    uint32_t accepted = 0;
    cere_token_score scores[kMaxSpecChainDrafts + 1] = {};
};

// The CUDA device (executor/cuda/src/device/) and the CPU toy (executor/cuda/src/toy/) implement this interface.
// Calls follow the host API's serialization (one call at a time per executor), except `write`, which may run
// concurrently for different tensors (and for disjoint ranges of one tensor): the device gives each concurrent
// caller its own staging slot (CUDA: one slot of the pinned bounce ring). Implementations never throw.
class Device {
public:
    virtual ~Device() = default;

    // Whether the device suffered a sticky fault (CUDA: an illegal address, launch failure, ECC or the like left the
    // context unusable until the process restarts). The host then answers every call that would use the device with
    // CERE_ERR_INTERNAL (a stable `cere: device_poisoned ...` line is written once, by the device) so the supervisor
    // reloads the card, instead of running more work on poisoned memory. Terminal polls of tickets already in flight
    // still run to their completion.
    virtual bool faulted() const { return false; }

    // The compiled support table entries this device accepts.
    virtual std::vector<const Support *> supported() const = 0;
    // Pinned host bytes this device may allocate in total.
    virtual uint64_t host_budget_bytes() const = 0;
    // Device and host bytes `allocate` will take for the plan, including workspace; the host compares them with the
    // executor's memory_budget_bytes and host_budget_bytes() (minus what live models reserved) before anything is
    // allocated, and returns CERE_ERR_CAPACITY without calling `allocate` when they do not fit.
    virtual MemoryEstimate estimate(const ModelPlan &plan) const = 0;
    // ALLOCATE: storage for every non-DISCARD tensor (DEVICE in device memory, HOST_PINNED in pinned host memory).
    // CERE_ERR_CAPACITY when an allocation fails; nothing is kept on error.
    virtual cere_status allocate(const ModelPlan &plan, std::unique_ptr<DeviceModel> *out) = 0;
    // UPLOAD: copies `len` bytes to a non-DISCARD tensor at `offset` before return (range already validated).
    virtual cere_status write(DeviceModel &model, uint64_t tensor_index, uint64_t offset, const void *bytes,
                              uint64_t len) = 0;
    // UPLOAD without a write (cere_model_host_resident, m216 design note L4): the bytes of HOST_PINNED tensor
    // `tensor_index` when the model's host region was attached from a published, digest-verified shared segment;
    // CERE_ERR_UNSUPPORTED when it is private or being created by this process.
    virtual cere_status host_resident(const DeviceModel &, uint64_t, const uint8_t **) const {
        return CERE_ERR_UNSUPPORTED;
    }
    // Put staging (cere_executor_register_staging, m216 design note M216S-11): makes [base, base + bytes) a source
    // the device copies DEVICE tensors from directly (CUDA: cudaHostRegister); CERE_ERR_UNSUPPORTED (the default)
    // without a direct path, CERE_ERR_CAPACITY when the pin fails (nothing kept). The host keeps the ranges disjoint,
    // calls these one at a time and unregisters a range only with no write_staged from it in flight.
    virtual cere_status register_staging(const uint8_t *, uint64_t) { return CERE_ERR_UNSUPPORTED; }
    virtual void unregister_staging(const uint8_t *, uint64_t) {}
    // `write` of a DEVICE tensor whose source lies wholly inside one registered range (same contract as `write`; the
    // bytes written are the same). May run concurrently with write and write_staged like write.
    virtual cere_status write_staged(DeviceModel &model, uint64_t tensor_index, uint64_t offset, const void *bytes,
                                     uint64_t len) {
        return write(model, tensor_index, offset, bytes, len);
    }
    // The CERE_LEVER_MOE_* value naming the MoE backend this library was compiled with (CUDA: CERE_MOE_BACKEND
    // p3b -> CERE_LEVER_MOE_P3B, k7 -> CERE_LEVER_MOE_K7; the toy reports P3B, the shipped backend). A desc or
    // set_selection moe_gemm impl other than CERE_LEVER_MOE_BUILD and this value is CERE_ERR_UNSUPPORTED (MoE is a
    // build variant: the slot only checks it).
    virtual uint32_t moe_lever() const = 0;
    // True when the device's K2 GEMM has the tail-split route (CERE_LEVER_K2_TAIL_SPLIT: the Vulkan device only). A
    // desc or set_selection k2_gemm tail_split on any other device is CERE_ERR_UNSUPPORTED.
    virtual bool k2_tail_split() const { return false; }
    // ABI v5 hot swap (cere_model_set_selection): re-resolves the sealed model's kernel selections from `levers`
    // (already validated by the host, same rules as the desc's). Called only between requests (no ticket in flight),
    // so no forward call reads a selection while it changes. Every selection runs over the buffers ALLOCATE carved
    // whatever the selection (the device's own comment proves it per slot), so nothing is allocated here. On error
    // the device keeps its previous selection.
    virtual cere_status set_levers(DeviceModel &model, const Levers &levers) = 0;
    // INSTANTIATE (cere_model_seal): every non-DISCARD tensor is fully written; build derived state, after which the
    // model's tensors are immutable. On error the host keeps the model unsealed.
    virtual cere_status instantiate(DeviceModel &model) = 0;
    // False when the library was built without a forward pass: capabilities report execution_mode UNSUPPORTED and
    // operations 0, submit returns CERE_ERR_UNSUPPORTED, and `step` is never called.
    virtual bool forward_supported() const = 0;
    // True when `step` and `step_rows` honour a non-null `sampling`/`logprobs` (capabilities native_sampler 1);
    // otherwise submit refuses an item with sampling or out_logprobs as CERE_ERR_UNSUPPORTED, `step` always gets
    // null/null and every RowArgs has null `sampling` and `logprobs`.
    virtual bool sampler_supported() const = 0;
    // Session state for up to max_context_tokens tokens (<= the plan's max_context_tokens). With `lazy` false (every
    // model with prefix_state_slots 0, byte for byte M2.5) it reserves kv_pages(max_context_tokens) pages of the
    // model's KV pool (host/kv_pool.h, lowest free ids first) for the session's life, so `step` never runs out of
    // pages; CERE_ERR_CAPACITY with nothing kept when the pages are not free or the state cannot be allocated. With
    // `lazy` true (M2.10 D3′, only on a prefix model) it takes the session slot and no page: the host binds the pages
    // with `bind_pages` before the session's first step or load. Destroying the session releases its pages (one
    // reference each, host/kv_pool.h refcounts: a page the prefix tree also holds stays allocated).
    virtual cere_status create_session(DeviceModel &model, uint64_t max_context_tokens, bool lazy,
                                       std::unique_ptr<DeviceSession> *out) = 0;
    // One step of the session's sequence: one forward pass of n = min(count, the device's prefill chunk, limit)
    // tokens (limit 0 = no limit; count >= 1; the chunk is 4,096 tokens on the served plan, m22c D4), *consumed = n.
    // When n == count it scores the next token into *out; when n < count (a prompt continues, M2.6 D2a, or a prefix
    // capture point ends the call, M2.10 D4′) *out and *logprobs are untouched and the host passes the rest in a later
    // call. The host feeds a prompt from its start (or from a prefix hit's position, `load_slot`), or the rest of a
    // prompt a call left partly consumed (still prompt tokens), and every generated token alone (count 1) once the
    // prompt is consumed. With `sampling` null the call that ends the tokens is greedy on the raw logits (`out` as
    // before M2.7). Otherwise the next token is chosen by the sampler (cere_executor.h, cere_sampling_params; the
    // counter key is the session position of the last consumed token), the session's output counts take the
    // committed token when `sampling->sampled`, and `logprobs` (null or one entry) receives the committed token's
    // logprob and its top `sampling->top_logprobs`. On error nothing of this call is consumed (a failure after the
    // first launch breaks the session: its later steps fail with CERE_ERR_INTERNAL). CERE_PENDING means the step did
    // not run (nothing consumed, no token): the work item stays pending at its token boundary and the host retries in
    // a later round. `images`: the session's prompt images (null for a text session; only on a model whose support
    // entry has a vision config), passed with every step of an image session's open prompt (M2.13-L: any chunk, any
    // position; after the prompt the session decodes like a text session at position + rope_delta, in step_rows and
    // speculate_rows too). `end` (null for every step but the D15⁗′ ones, EndStep): CERE_ERR_ARGUMENT
    // before anything runs for a malformed one (not a PREFILL, `slot + 1` outside the pool, cls below n). `slice`
    // (null for every step but a layer-sliced PREFILL round, StepSlice): the call may pause between layer segments.
    virtual cere_status step(DeviceModel &model, DeviceSession &session, const uint32_t *tokens, uint64_t count,
                             uint64_t limit, const StepImages *images, uint64_t *consumed,
                             const ItemSampling *sampling, cere_token_score *out, cere_token_logprobs *logprobs,
                             const EndStep *end, StepSlice *slice) = 0;
    // One batched decode step (M2.6 D2b/D7): row r consumes rows[r].token in rows[r].session and scores its next
    // token into out[r], bit-identical to a single-row `step` of that session with rows[r].sampling and
    // rows[r].logprobs (D6, ledger CR4: a row's sampled token depends on its own logits, seed, position and output
    // counts only, never on the other rows). row_count is in [1, the plan's max_sessions], every session is a
    // distinct decode-ready session of `model`. CERE_ERR_ARGUMENT for a malformed call (null pointer, row_count out
    // of range, a null or repeated session, a row with `logprobs` but no `sampling`) before anything runs,
    // row_status untouched. Otherwise every row_status[r] is set: before any launch each row's single-row
    // preconditions are checked (a broken session -> CERE_ERR_INTERNAL, no room for the token -> CERE_ERR_CAPACITY,
    // toy only: CERE_PENDING for a held row); such a row consumes nothing and the other rows run. The rows still
    // CERE_OK are the participants. CERE_OK: every participant consumed its token and out[r] (and *rows[r].logprobs)
    // holds its result. Any other return is a failure after the first launch: every participant's session is broken
    // (its later steps fail with CERE_ERR_INTERNAL), its row_status[r] is the returned status, and non-participants
    // are untouched.
    virtual cere_status step_rows(DeviceModel &model, const RowArgs *rows, uint32_t row_count, cere_token_score *out,
                                  cere_status *row_status) = 0;
    // M2.9: true when `model` was created with speculative_draft_tokens > 0 and this device runs its draft
    // (capabilities native_speculative_batch 1, operations | CERE_OPS_VERIFY). allocate refuses a plan with drafts as
    // CERE_ERR_UNSUPPORTED on a device that cannot run them.
    virtual bool draft_supported(const DeviceModel &) const { return false; }
    // M2.9: whether `session` drafts (the host sets it once, before the session's first step, from its first PREFILL
    // item: on iff the model has a draft and the item lacks CERE_WORK_NO_SPECULATION). A drafting session runs the
    // MTP extend in every `step` (D2: over the chunk's shifted ids; its last row also takes the scored first token)
    // and leaves the next proposals pending; after its prompt it decodes only through speculate_rows (never
    // step_rows or a count-1 `step`). Snapshot save/load of a drafting session carries its draft state (kind 10).
    virtual cere_status set_session_draft(DeviceModel &, DeviceSession &, bool on) {
        return on ? CERE_ERR_UNSUPPORTED : CERE_OK;
    }
    // M2.13-L (m213l design note D-L2, contract C4): the session's rotary offset past its prompt. Every row the
    // session consumes outside an open prompt (`step` with images null, step_rows, speculate_rows, the MTP draft rows
    // after the prompt) rotates at its position + rope_delta on all three mRoPE axes; 0 (every text session, the
    // default) keeps every launch as before. The host sets it before every PREFILL's first step (the mRoPE value of
    // an image prompt, 0 for text) and after every restore (kind 11's value, else 0). CERE_ERR_ARGUMENT for a
    // positive delta.
    virtual cere_status set_rope_delta(DeviceModel &model, DeviceSession &session, int64_t rope_delta) = 0;
    // M2.9: one speculation round for each row (distinct decode-ready drafting sessions of `model`, row_count in
    // [1, max_sessions]). Error contract as step_rows: CERE_ERR_ARGUMENT for a malformed call before anything runs
    // (row_status untouched); per-row preconditions set row_status without running that row. Per row: verify, accept
    // (D5 greedy / D6 sampled), truncate, commit (undo restore of rows >= a, M29S2-20), then draft-extend and two MTP
    // decode steps. A failure before commit: the row's session consumed nothing (status returned, session broken as
    // step_rows). A failure after commit: out[r] holds the committed tokens, row_status[r] the error, the session is
    // broken. CERE_OK: every participant's out[r] is set.
    virtual cere_status speculate_rows(DeviceModel &, const SpecRowArgs *, uint32_t, SpecRowResult *,
                                       cere_status *) {
        return CERE_ERR_UNSUPPORTED;
    }
    // DraftAccept (CERE_WORK_TAIL_DRAFTS): the most drafts one speculate_rows chain may carry on `model` (the CUDA
    // device: model::layout::kMaxChainDrafts within its verify rows and undo pool; 3 elsewhere).
    virtual uint32_t max_chain_drafts(const DeviceModel &) const { return 3; }
    // DraftAccept: the session's pending MTP proposals (host copies, no device work): writes up to `max` of them to
    // `ids` and returns their count (0 for a session that does not draft or a device without MTP proposals).
    virtual uint32_t pending_drafts(const DeviceSession &, uint32_t *, uint32_t) const { return 0; }
    // M2.9L (m29l design note D3), O95: whether a single-row speculate_rows call's wall time is the cost of its
    // verify window, so cere_model_seal measures the model's fixed cost tables (host/spec_calibrate.h). The CPU toy's
    // is host noise: its models keep SpecCosts::untimed() (deterministic choices).
    virtual bool times_spec_rounds() const { return false; }
    // M2.11 (m211 design note D2, D4, D5): whether this device saves and loads session state, so a model may carry
    // a snapshot arena (cere_model_desc.snapshot_host_bytes > 0; otherwise model creation is UNSUPPORTED).
    virtual bool snapshot_supported() const { return false; }
    // Writes the device sections (every kind >= 1 of `plan`) of an idle session that has consumed `position` tokens
    // (its prompt fully consumed, M2.6) into the image `dst` (plan.total_bytes, header and kind 0 already written by
    // the host) at the plan's offsets, in logical row order. The session is unchanged.
    virtual cere_status save_state(DeviceModel &, DeviceSession &, const snapshot::Plan &, uint64_t, uint8_t *) {
        return CERE_ERR_UNSUPPORTED;
    }
    // The reverse into a fresh session from a validated image (D14): afterwards the session has consumed `position`
    // tokens, is decode-ready (no open prompt) and its next step continues exactly as the saved session's would. On
    // error the session stays fresh.
    virtual cere_status load_state(DeviceModel &, DeviceSession &, const snapshot::Plan &, uint64_t,
                                   const uint8_t *) {
        return CERE_ERR_UNSUPPORTED;
    }

    // ---- M2.10 prefix cache (m210 design note D3′, D4′, D6′, D7, D9; ledger M4, M5, CR3). Called only on a model
    // whose plan has prefix_state_slots > 0, from the host's serialized calls. The device keeps the KV page pool and
    // its per-page reference counts (host/kv_pool.h); the host keeps the tree, the state-slot indices and every
    // policy. A state slot is an index in [0, prefix_state_slots + max_sessions) of the model's fixed-state pool
    // (D6′: the slot's contiguous GDN conv/SSM, PLE history/conv and QSA ring bytes, 59,031,808 B on the A4 config).
    // Whether this device implements the methods below (capabilities native_prefix_cache; a desc with
    // prefix_state_slots > 0 is CERE_ERR_UNSUPPORTED otherwise).
    virtual bool prefix_supported() const { return false; }
    // Pages of the model's pool with no reference.
    virtual uint64_t free_pages(const DeviceModel &) const { return 0; }
    // References of page `id` (0 = free).
    virtual uint32_t page_refs(const DeviceModel &, uint32_t) const { return 0; }
    // +1 / -1 reference on each of `n` allocated pages (a page is freed when its count reaches 0). The host calls
    // retain when the tree takes pages of a session and release when it drops a node's pages. CERE_ERR_INTERNAL (or
    // ARGUMENT for a null list) with nothing changed when the pool refuses an entry (an unallocated page, a double
    // release): the host then treats the model's page accounting as faulted.
    [[nodiscard]] virtual cere_status retain_pages(DeviceModel &, const uint32_t *, uint64_t) { return CERE_OK; }
    [[nodiscard]] virtual cere_status release_pages(DeviceModel &, const uint32_t *, uint64_t) { return CERE_OK; }
    // Appends `n` free pages (lowest ids first, ascending, one reference each) to *out for the tree's own use (D12
    // import: the pages of an imported prefix before the tree retains its edge). CERE_ERR_CAPACITY with nothing
    // taken when fewer than n are free (the host evicts first).
    virtual cere_status take_pages(DeviceModel &, uint64_t, std::vector<uint32_t> *) { return CERE_ERR_UNSUPPORTED; }
    // Binds a lazy session's pages: its page list becomes `prefix` (n_prefix allocated pages, one reference added
    // each: a prefix hit's path pages) followed by kv_pages(max_context_tokens) - n_prefix own pages (lowest free
    // first), and its slot map is written. CERE_ERR_CAPACITY with nothing kept when the own pages are not free (the
    // host evicts first); CERE_ERR_ARGUMENT for a bound or non-lazy session or n_prefix > kv_pages(context).
    virtual cere_status bind_pages(DeviceModel &, DeviceSession &, const uint32_t *, uint64_t) {
        return CERE_ERR_UNSUPPORTED;
    }
    // The session's page list (kv_pages(context) ids in logical order; empty for an unbound lazy session).
    virtual std::vector<uint32_t> session_page_list(const DeviceSession &) const { return {}; }
    // The session's position (tokens consumed).
    virtual uint64_t session_position(const DeviceSession &) const { return 0; }
    // Tokens of one prefill chunk of `model` (`step`'s n without a limit), 0 when unknown. D15⁗ (ledger CR8): a
    // drafting session hits only at a multiple of it, a call boundary of the cache-off layout.
    virtual uint64_t prefill_chunk(const DeviceModel &) const { return 0; }
    // The prefill selection signature (cere_model_prefill_signature, BC1): appends to `out` the device's facts that
    // pick a sealed model's prefill kernels and their bits (CUDA: model/prefill_signature.h selection_text; toy:
    // fixed lines), '\n'-terminated, one fact per line in a fixed order. False when the device has none (the call
    // answers CERE_ERR_UNSUPPORTED).
    virtual bool prefill_selection(const DeviceModel &, std::string & /*out*/) const { return false; }
    // PrefixCapture G1: the state slots after the pair a chunk-0 end state of `rows` saved rows needs for its K/V
    // (EndStep::kv_slots) when its cold call has `cls` rows; 0 when the model cannot serve that one.
    virtual uint32_t end_kv_slots(const DeviceModel &, uint64_t /*rows*/, uint64_t /*cls*/) const { return 0; }
    // m210 D15⁗″ (ResumeExact): the row-bit signature σ of a cold PREFILL call of `cls` rows from position `s` (a
    // multiple of prefill_chunk): a value that differs between two classes whenever some row-count choice of the call
    // can give a row other bits (every choice proven row-invariant left out). An end state of class κ at chunk start
    // s serves a drafting lookup of class κ′ iff σ(s, κ) == σ(s, κ′): its rows [s, b) are the κ′ cold call's. The
    // default, σ = cls, is D15⁗′'s equal-class rule.
    virtual uint64_t row_sig(const DeviceModel &, uint64_t /*s*/, uint64_t cls) const { return cls; }
    // copy_fixed_state (ledger M4): the session's fixed state -> state slot `slot` (one device copy, ordered on the
    // device stream before any later step). The session is unchanged.
    virtual cere_status capture_state(DeviceModel &, const DeviceSession &, uint32_t) { return CERE_ERR_UNSUPPORTED; }
    // load_fixed_state (ledger M4): state slot `slot` -> the fresh bound session's fixed state; the session is then
    // at `position` (a multiple of 64, its pages [0, position / 64) already hold the prefix's KV) with its prompt
    // open, so its next step is PREFILL (D3′). CERE_ERR_ARGUMENT for a session that consumed tokens.
    virtual cere_status load_slot(DeviceModel &, DeviceSession &, uint32_t, uint64_t) {
        return CERE_ERR_UNSUPPORTED;
    }
    // D9 host tier and D12 export: writes a prefix node's device sections into the snapshot image `dst` (plan from
    // snapshot::plan with that first_row and position; header and kind 0 written by the host): KV rows
    // [plan first_row, position) read through `pages` (page i holds rows [64 i, 64 i + 64); entries below
    // first_row / 64 may be any value), the fixed state from slot `slot`, zero output counts. A plan with kind 10 (the
    // host tier on a model with drafts, Sessions O211) also takes the MTP K/V rows and QSA keys of [first_row,
    // position); its proposals part is zeros (no ring rows: position is a multiple of 64).
    virtual cere_status save_prefix(DeviceModel &, const uint32_t *, uint64_t, uint32_t, const snapshot::Plan &,
                                    uint64_t, uint8_t *) {
        return CERE_ERR_UNSUPPORTED;
    }
    // The reverse: KV rows [first_row, position) into `pages` (allocated, page i as above), the fixed state into slot.
    virtual cere_status load_prefix(DeviceModel &, const uint32_t *, uint64_t, uint32_t, const snapshot::Plan &,
                                    uint64_t, const uint8_t *) {
        return CERE_ERR_UNSUPPORTED;
    }

    // ---- Track LoRA (lora design note; cere_model_lora_*, cere_session_set_lora). The host validates arguments
    // (host/lora.h check_load, slot bounds, slot state, budget) and serializes the calls; a device without LoRA keeps
    // these defaults (lora_bytes 0: the host answers CERE_ERR_UNSUPPORTED).
    // Device bytes a pool of `adapters` slots of rank up to `rank` takes for `model` (0 = no LoRA on this device).
    virtual uint64_t lora_bytes(const DeviceModel &, uint32_t, uint32_t) const { return 0; }
    // Before INSTANTIATE: allocates that pool (every slot empty). CERE_ERR_CAPACITY when the memory is not there.
    virtual cere_status lora_reserve(DeviceModel &, uint32_t, uint32_t) { return CERE_ERR_UNSUPPORTED; }
    // A sealed model: copies one validated adapter (slot, rank, scale, count targets with host A / B) into an empty
    // slot, ordered before any later step; targets left out add nothing. On error the slot stays empty.
    virtual cere_status lora_load(DeviceModel &, uint32_t, uint32_t, float, uint64_t, const uint32_t *,
                                  const void *const *, const void *const *) {
        return CERE_ERR_UNSUPPORTED;
    }
    // Empties a slot no session uses.
    virtual cere_status lora_unload(DeviceModel &, uint32_t) { return CERE_ERR_UNSUPPORTED; }
    // Binds a session that has not stepped to a loaded slot (-1: none, the base model).
    virtual cere_status set_session_lora(DeviceModel &, DeviceSession &, int32_t) { return CERE_ERR_UNSUPPORTED; }

    // ---- WeightLab research (weightlab design note WL-02; cere_model_research_*, cere_session_set_research).
    // The host validates plans (research/plan.h), selects rows and accounts the host ring (research/capture.h) and
    // serializes the calls; a device without research keeps these defaults (the host answers CERE_ERR_UNSUPPORTED).
    virtual bool research_supported() const { return false; }
    // Before INSTANTIATE: `staging_bytes` of device capture staging, `plan_bytes` of device plan memory
    // (max_plans x CERE_RESEARCH_MAX_PLAN_BYTES, slot i at i x CERE_RESEARCH_MAX_PLAN_BYTES) and `host_bytes` of
    // host ring (pinned on a GPU), whose base goes to *ring; the device frees them with the model. CERE_ERR_CAPACITY
    // when the memory is not there (nothing kept).
    virtual cere_status research_reserve(DeviceModel &, uint64_t, uint64_t, uint64_t, uint8_t **) {
        return CERE_ERR_UNSUPPORTED;
    }
    // A sealed model: copies a validated plan's operands (Op::*_offset) into slot `slot`'s plan memory, ordered before
    // any later step.
    virtual cere_status research_plan_load(DeviceModel &, uint32_t, const cere::research::Plan &) {
        return CERE_ERR_UNSUPPORTED;
    }
    // Binds a session that has not stepped to `binding` (owned by the host; it outlives the session): every row of
    // the session runs through the research hook from its first step on.
    virtual cere_status set_session_research(DeviceModel &, DeviceSession &, const cere::research::Binding *) {
        return CERE_ERR_UNSUPPORTED;
    }
    // Waits for every capture write enqueued so far (the host then reads the ring).
    virtual cere_status research_sync(DeviceModel &) { return CERE_OK; }
};

// Provided by the linked device implementation: CERE_ERR_UNSUPPORTED for a device_id it does not drive,
// CERE_ERR_ARGUMENT for a malformed one. The CUDA device's id is the decimal CUDA ordinal among the visible devices
// ("0" in the GPU0 container); empty, a leading zero, a sign or more than 9 digits is ARGUMENT; a non-decimal id,
// an ordinal without a visible device, a device other than compute capability 12.0 or one that cannot map host
// memory is UNSUPPORTED (device/cuda_device.h). The CUDA device also requires eager module loading
// (CUDA_MODULE_LOADING=EAGER in the process environment, O14): without it creation returns UNSUPPORTED.
cere_status create_device(const char *device_id, std::unique_ptr<Device> *out);

// SelfTune ST4 S3c: the support entry of the SDAR model `device` runs, null when it runs none. Provided by the
// linked device implementation of an sdar library (device/sdar/ returns its DICE-8B entry, the host toy null, so
// the toy sdar library refuses every in-scope descriptor as `device`); a Qwen library never calls it. Of the entry
// only name, max_sessions, max_context_tokens, max_kv_pool_tokens and dtype_mask are read: `config` is zero (the
// sdar config is checked against the library's compiled one, api.cpp) and `vision` null.
const Support *sdar_support(const Device &device);

// Inkling 2 Mini (inkling design note S0, S6g): the support entry of the Inkling model `device` runs, text-only or
// with the vision tower (`vision`: its entry's vision config is the one a descriptor's must equal), null when it runs
// none, read as sdar_support's is. Provided by the device implementation of an inkling2_loop library (device/inkling/;
// the host toy returns null, so a toy library refuses every in-scope descriptor as `device`).
const Support *inkling2_support(const Device &device, bool vision);

}  // namespace cere::host
