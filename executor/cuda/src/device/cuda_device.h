// The CUDA device of libcere_executor.so (m22 design note Q12, Q13, O14): A4 only (text-only, or with its vision
// tower: m213 design note D9, D10), one GPU per executor.
//
// device_id: the decimal CUDA ordinal among the devices visible to the process ("0" in the GPU0 container, where
// `--gpus device=<GPU0 UUID>` exposes one GPU), optionally followed by ",host_share=<dir>,artifact=<sha256>" (both or
// neither, any order; host/host_share.h) and ",release_device=1" (below). Empty, a leading zero ("00", "01"), a sign
// or more than 9 digits is
// CERE_ERR_ARGUMENT, as is a malformed, repeated, lone or unknown option (P1's former ",decode_graph=" too: the
// decode_step lever selects the graph); any other non-decimal id (e.g. "toy") and an ordinal without a visible device
// are CERE_ERR_UNSUPPORTED, as is a device that is not compute capability 12.0 (the library holds sm_120a code only)
// or cannot map host memory.
//
// Device release (m32n design note D10, `release_device=1`, an ordinal below device::kTrackedDevices, else
// CERE_ERR_ARGUMENT): when the last executor of this process on the ordinal is destroyed and it was created with the
// option, ~CudaDevice resets the device's primary context (cudaDeviceReset), so the card gets back the context, its
// eagerly loaded modules and every allocation of this process; a later executor on the ordinal creates a fresh
// context. The ordinal's device::context_generation is bumped first, so per-context caches are set again. A process
// that uses the device outside its executors (e.g. torch in a test) must not pass the option. Without it nothing
// changes.
//
// Decode graph (m23 design note P1, M23P1-01..07; lever decode_step, PI-02): with decode_step CERE_LEVER_DECODE_GRAPH
// the model runs an eligible DECODE step (greedy, unconstrained, text-only, no taps or logprobs, after the session's
// first token; K7 or P3b MoE, M23P1-10) as a CUDA graph, captured on the first step of its key and replayed after
// (model.h DecodeGraphs); bit-identical to the eager step (executor/tests/gpu/test_graph_decode.py). Every other step,
// and every step with decode_step CERE_LEVER_DECODE_EAGER (the default), is eager.
//
// Shared HOST_PINNED region (opportunity O47, build-loop-ple design note): with the share options the mapped block's
// HOST_PINNED tensors [0, region_bytes) live in the artifact's shared segment (layout tag "cuda-sm120-v1"), mapped
// MAP_SHARED, pinned in this process with cudaHostRegister + cudaHostGetDevicePointer; the staging gets its own small
// cudaHostAlloc(cudaHostAllocMapped) block. The registration path comes from cudaDevAttrHostRegisterReadOnlySupported
// only (printed as `cere: host_share_registration=read_only|rw_mprotect`): `read_only` maps PROT_READ and registers
// Mapped | Portable | ReadOnly; `rw_mprotect` (attribute 0) maps the region writable, registers Mapped | Portable and
// then mprotects it PROT_READ (the CPU mapping is read-only; the GPU-side write protection is lost, and no kernel
// writes a HOST_PINNED tensor: executor/tests/gpu/test_model_host_share.py). Every share or register failure fails
// closed (no private copy). The host estimate counts the region only while no published segment of that name
// exists. ~CudaModel unregisters before the segment unmaps (and the last mapper unlinks it). Each executor's mapping
// of the segment costs ~96 MiB of GPU page tables (51.2 GB / 4 KiB x 8 B) on every card of the process, with or
// without Portable (M3.2N G-7, m32n design note D5 note).
//
// Executor creation initializes the CUDA context and requires eager module loading (CUDA_MODULE_LOADING=EAGER in the
// process environment, checked through cuModuleGetLoadingMode; lazy loading is CERE_ERR_UNSUPPORTED), so every module
// of the library (its kernels and cuBLASLt's) is resident before the first model is created (O14: a lazily loaded
// kernel allocates device memory on a full GPU). It also creates the device's cuBLASLt handle (K2; the forward's GEMM
// plans, M2.2b, are created from it at INSTANTIATE) and allocates the bounce ring: 4 slots of 64 MiB pinned host
// memory, each with its own stream (Q13).
//
// Host budget: MemAvailable at executor creation (after the ring is pinned) minus kHostReserveBytes. The live scripts
// enforce the stricter floor (MemAvailable >= pinned PLE + 16 GiB + the deployed config's snapshot_host_bytes, the
// M2.11 host-RAM arena locked at ALLOCATE) before a native start; this budget only keeps a
// model from pinning the host's last memory.
//
// Lifecycle: ALLOCATE takes the whole memory plan (memory_plan.h) in one cudaMalloc (device_bytes), one
// cudaHostAlloc(cudaHostAllocMapped) (host_bytes: HOST_PINNED tensors and staging) and, with snapshot_host_bytes > 0,
// the M2.11 snapshot arena as its own host::LockedMemory (page-aligned anonymous mmap, zero-filled, mlock'd,
// MADV_DONTDUMP/MADV_DONTFORK; never device-mapped nor CUDA-registered, so it takes no GPU memory; a failed mmap,
// madvise or mlock, e.g. RLIMIT_MEMLOCK below the arena without CAP_IPC_LOCK, is CERE_ERR_CAPACITY with the reason on
// stderr), and binds the model's tensors, workspace and derived region by pointer
// (model::bind_weights, model::bind_workspace; no byte is read); UPLOAD copies DEVICE ranges through a ring slot
// (double-buffered 32 MiB halves, stream-synchronized before return) and HOST_PINNED ranges straight into the mapped
// host block; INSTANTIATE drains the device, then model::instantiate_model computes the derived tensors and creates
// every GEMM plan from the device's cuBLASLt handle; sessions take a preallocated slot (model::bind_session +
// model::session_reset). After INSTANTIATE nothing is allocated and no module is loaded.
//
// Memory ledger (host/memory_ledger.h; m211 design note R1 [REVISED 2026-10-11]): the device records every byte it
// allocates itself, exactly: the bounce ring (Held::kRing) at creation and, per model at ALLOCATE, the cudaMalloc
// (Held::kDevice), the mapped block (Held::kMapped) and the arena's locked pages (Held::kLocked); each is dropped
// when freed. This cudaMalloc and a LoRA reserve's pool (track LoRA, Held::kDevice too) are the executor's only device
// allocations (no cudaMallocAsync, no memory pool), so ledger().held(Held::kDevice) is the device bytes the executor
// holds; cuBLASLt's and the CUDA context's own memory are not the executor's and are not in it. The ktest cere_ktest_model_ledger reads it.
//
// Forward (M2.2b, M2.6): forward_supported() is true once a model is instantiated. `step` runs one forward call of
// the session's unconsumed tokens through model::forward on the executor stream (one prefill chunk of at most the
// workspace rows per call, PREFILL, until the prompt is consumed - M2.6 D2a; then one token per step, DECODE) and
// returns the next token's top-2 when the call ends its tokens; `step_rows` steps up to max_sessions decode-ready
// sessions at once through model::forward_decode_rows (M2.6 D5/D7; one row = the single-row DECODE forward). The
// host layer commits tokens, applies stop tokens, max_new_tokens, the context limit and cancel at token boundaries.
// A step beyond the session's max_context_tokens is CERE_ERR_CAPACITY before any launch. A failed step after the
// first launch leaves the session (every participant of a step_rows) unusable (every later step is
// CERE_ERR_INTERNAL); session release frees its slot only.
//
// Sampler (M2.7, m27 design note I9): sampler_supported() follows forward_supported() (the K12 kernels are always
// built). A step with sampling or logprobs passes them to the forward call that ends its tokens (the last prefill
// chunk, or the decode token); one with neither runs the greedy path unchanged (D1). A `step_rows` row with sampling
// runs the same K12 stage on its own logits (ledger CR4, model::forward_decode_rows). A PREFILL step that starts a
// prompt only runs on a session session_reset just zeroed (position 0), output counts included (D5).
//
// Prefix cache (M2.10, m210 design note D3′, D4′, D6′, D7): prefix_supported() is true (the forward is always
// built); a model with plan.prefix_state_slots > 0 holds the fixed-state pool after its KV pool (memory_plan.h
// region 6), creates lazy sessions (slot only; pages at bind_pages) and shares KV pages by reference count
// (host/kv_pool.h). `step` ends a call after min(chunk, count, limit) tokens; a call that ends inside a prompt leaves
// it open, so the capture after it copies the state of the call `[…, b)` (D4 (b′)), and load_slot leaves a hit
// session at h with its prompt open, so its next step is PREFILL (D3′).
#pragma once

#include "device/memory_plan.h"
#include "host/device.h"
#include "host/host_share.h"
#include "host/kv_pool.h"
#include "host/locked_memory.h"
#include "host/memory_ledger.h"
#include "kernels/k2_gemm.h"
#include "kernels/lora.h"
#include "model/layout.h"
#include "model/model.h"
#include "model/mtp.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace cere::device {

// Q13: 4 x 64 MiB pinned bounce slots, one per concurrent `write` caller or M2.11 state copy.
constexpr size_t kRingSlots = 4;
constexpr uint64_t kSlotBytes = uint64_t(64) << 20;
// m216 design note M216S-11: puts from registered staging (write_staged) DMA straight from the caller's pinned
// buffer, each on one of these streams for its duration (one per concurrent caller; a 17th waits).
constexpr size_t kStagedStreams = 16;
// m216 design note M216S-12: a model's host region is cudaHostRegister'd in pieces of this size, so the CUDA calls
// of ALLOCATE and UPLOAD on other threads run between pieces instead of waiting for the whole region.
constexpr uint64_t kRegisterChunkBytes = uint64_t(1) << 30;
// Host memory a model may never pin (see the host budget above).
constexpr uint64_t kHostReserveBytes = uint64_t(8) << 30;

class CudaDevice;
class CudaSession;

class CudaModel final : public host::DeviceModel {
public:
    CudaModel(int ordinal, ModelLayout layout, std::vector<uint32_t> placements);
    ~CudaModel() override;

    // Storage of a non-DISCARD tensor: device address (for HOST_PINNED, the mapped device alias of the host bytes).
    void *device_ptr(uint64_t tensor_index) const;
    // Host address of a HOST_PINNED tensor.
    uint8_t *host_ptr(uint64_t tensor_index) const;
    uint8_t *session_slot(uint64_t slot) const;
    // M2.10 state slot `slot` of the fixed-state pool (m210 design note D6′; slot < layout().state_slots).
    uint8_t *state_slot(uint64_t slot) const;
    // M2.9: undo snapshot j of session slot `slot` (layout_.undo_offset; null without drafts).
    uint8_t *undo_slot(uint64_t slot, uint64_t j) const;
    // DraftAccept: every undo snapshot of the model (max_sessions x kUndoSnapshots, index 3 slot + j, the bytes
    // undo_slot addresses), bound on first use; scratch of one speculate_rows call group whose chain is longer than
    // kUndoSnapshots (null without drafts).
    model::UndoState *undo_pool();
    // DraftAccept: the most drafts of one chain: min(kMaxChainDrafts, verify_rows - 1, the pool's snapshots).
    uint32_t max_chain_drafts() const;
    // M2.9: the plan's speculative_draft_tokens (0 = no draft-only region, no rows plans beyond max_sessions).
    uint32_t draft_tokens() const { return drafts_; }

    // K11 (ktest cere_ktest_session_set_pages, host/kv_pool.h PagePermutation): arms the page order the model's next
    // session create applies to its reserved pages.
    cere_status arm_page_order(const uint32_t *order, uint64_t n);
    // Free pages of the model's KV pool (m25 design note D3; ktest introspection).
    uint64_t free_pages() const;
    // References of page `id` (host/kv_pool.h; 0 = free, also for an id outside the pool).
    uint32_t page_refs(uint32_t id) const;

    const ModelLayout &layout() const { return layout_; }
    // Rows of the model's Workspace (the largest forward call).
    uint64_t workspace_rows() const { return workspace_.max_tokens; }
    // The model's Workspace (kernel tests only: the sampler stages' buffers after a step or a speculate_rows call).
    const model::Workspace &workspace() const { return workspace_; }

    // Rows per forward call of a multi-token step (kernel tests only; set while no step runs): 0 or a value above the
    // workspace's rows = workspace rows, so a prompt within the context limit is one forward call (M2.2b).
    void set_prefill_chunk(uint64_t tokens) { prefill_chunk_ = tokens; }

    // MixedPD (kernel tests only, cere_ktest_session_prefill_sliced; set while no step runs): the wall time per layer
    // and its call's prompt tokens that the next sliced PREFILL step plans its segment from (model::segment), fixed so
    // the cut is deterministic; the step then overwrites them with its own measurement. parked_layer: the paused
    // call's next layer (0: none paused).
    void set_park_cost(double ms_per_layer, uint64_t tokens) {
        park_ms_per_layer_ = ms_per_layer;
        park_tokens_ = tokens;
    }
    uint32_t parked_layer() const { return park_.next_layer; }

    // The model's decode graphs (m23 design note P1; ktest cere_ktest_decode_graph): `enabled` is the decode_step
    // lever (model::apply_levers); a kernel test may switch it while no step runs.
    model::DecodeGraphs &decode_graphs() { return workspace_.graphs; }

    // The verify_attention lever (m23 design note P11R-19; ktest cere_ktest_verify_attention): `verify_batched` is
    // the lever (model::apply_levers), `verify_chains` the count of chains it ran batched; a kernel test may switch it
    // while no verify runs.
    bool &verify_batched() { return workspace_.verify_batched; }
    uint64_t verify_chains() const { return workspace_.verify_chains; }

    // Shared mode (ktest cere_ktest_segment_digest/poke): the segment (nullptr in private mode), the device alias of
    // its region, and whether it was registered on the rw_mprotect path (without cudaHostRegisterReadOnly).
    const host::SharedSegment *segment() const { return segment_.get(); }
    uint8_t *host_device() const { return host_device_; }
    bool host_share_rw() const { return host_share_rw_; }
    // ktest cere_ktest_segment_ple_gather (M3.2N G-2): the CUDA ordinal and the bound weights (PLE table pointers).
    int ordinal() const { return ordinal_; }
    const model::ModelWeights &weights() const { return weights_; }

private:
    friend class CudaSession;
    friend class CudaDevice;
    int ordinal_;
    ModelLayout layout_;
    std::vector<uint32_t> placements_;  // per tensor, CERE_PLACEMENT_*
    uint8_t *device_ = nullptr;         // device arena (cudaMalloc)
    // Private mode: host_block_ is the whole mapped block (cudaHostAlloc, mapped) and host_ == host_block_. Shared
    // mode: host_ is the segment's region (cudaHostRegister'd while registered_) and host_block_ holds the staging only.
    uint8_t *host_ = nullptr;           // the HOST_PINNED tensors at layout_.offsets
    uint8_t *host_device_ = nullptr;    // device alias of host_
    uint8_t *host_block_ = nullptr;     // private: mmap'd, pinned by pinner_; shared: cudaHostAlloc(Mapped) staging
    uint64_t host_map_bytes_ = 0;       // private mode: the mmap length of host_block_ (0 = cudaHostAlloc'd)
    // m216 design note L3: private mode pins host_block_ on this thread while UPLOAD writes into it (populate, then
    // cudaHostRegister); L4c: shared mode verifies an attached segment's content digest and registers the region on
    // it. INSTANTIATE and the destructor join it first. pin_status_ is read only after the join.
    std::thread pinner_;
    cere_status pin_status_ = CERE_OK;
    void start_pin();
    void start_shared_pin(bool rw);
    cere_status join_pin();
    // load-soft design note LSF-01: the private pin registers the block only once every DEVICE byte of the plan is
    // put (device_put), or when INSTANTIATE or the destructor joins it (release_pin): a cudaHostRegister running beside
    // the device copies stalls them (~1 GB/s instead of ~28 GB/s), and the upload reads the HOST_PINNED tensors last.
    uint64_t device_put_target_ = 0;
    std::atomic<uint64_t> device_put_bytes_{0};
    std::mutex pin_mutex_;
    std::condition_variable pin_cv_;
    bool pin_go_ = false;
    void device_put(uint64_t bytes);
    void release_pin();
    // Registers [base, base + bytes) in kRegisterChunkBytes pieces (registered_ records each piece; the destructor
    // unregisters them, a failure keeps the pieces registered so far for it).
    cudaError_t register_region(uint8_t *base, uint64_t bytes, unsigned flags);
    uint8_t *staging_ = nullptr;        // the workspace's staging (in host_block_)
    std::unique_ptr<host::SharedSegment> segment_;  // shared mode; unmapped after the unregister in ~CudaModel
    std::vector<uint8_t *> registered_;  // the cudaHostRegister'd pieces of host_ (register_region)
    bool host_share_rw_ = false;        // shared mode: registered without cudaHostRegisterReadOnly (rw_mprotect)
    host::LockedMemory snapshot_;       // the M2.11 snapshot arena (DeviceModel::arena), not CUDA-registered
    host::LedgerEntry device_held_;     // device_ in the device's ledger (Held::kDevice)
    host::LedgerEntry host_held_;       // host_block_ (Held::kMapped)
    host::LedgerEntry region_held_;     // shared mode: the registered region (Held::kMapped)
    host::LedgerEntry snapshot_held_;   // snapshot_'s locked pages (Held::kLocked)
    mutable std::mutex slots_mutex_;    // guards slot_used_, pool_ and page_order_ (m25 F7)
    std::vector<bool> slot_used_;       // per session slot
    host::PagePool pool_;               // the paged KV pool's page ids (layout_.pool_tokens / 64)
    host::PagePermutation page_order_;  // K11's one-shot order for the next session create
    model::KvPool kv_{};                // the pool's device views, bound at ALLOCATE
    model::ModelWeights weights_{};     // bound at ALLOCATE, derived values at INSTANTIATE
    model::Workspace workspace_{};      // bound at ALLOCATE, plans at INSTANTIATE
    bool instantiated_ = false;
    // MF-1: instantiate_model began (it swizzles the MoE block scales in place, and a second swizzle of the same bytes
    // is wrong): a failed attempt is never retried, the model must be released.
    bool instantiate_attempted_ = false;
    uint64_t prefill_chunk_ = 0;
    uint32_t drafts_ = 0;               // plan.speculative_draft_tokens (M2.9)
    host::Levers levers_{};             // ABI v5: the active selection (plan.levers at ALLOCATE, then set_levers)
    std::vector<model::UndoState> undo_pool_;  // undo_pool(): bound on first use
    // MixedPD (mixed-pd design note): the paused prefill call (layout_.park_bytes; buffers null without the
    // region), its session (null: none paused), and the wall time per layer of the last paused segment or (BcSegment,
    // O283) whole sliced call from layer 0 (0: none yet), with that call's prompt tokens.
    model::Park park_{};
    const CudaSession *parked_ = nullptr;
    double park_ms_per_layer_ = 0.0;
    uint64_t park_tokens_ = 0;
    // Track LoRA (kernels/lora.h): the adapter pool of lora_reserve (null: none; Workspace::lora points at it), its
    // device bytes in the ledger (Held::kDevice) and the target index of each target id (-1: not a target).
    std::unique_ptr<kernels::lora::Table> lora_;
    host::LedgerEntry lora_held_;
    std::vector<int32_t> lora_index_;
    // WeightLab research (research/hook.h): the staging, plan memory and pinned host ring of research_reserve (null:
    // none; Workspace::research points at it), in the ledger as Held::kDevice and Held::kRing.
    std::unique_ptr<research::Staging> research_;
    host::LedgerEntry research_device_held_;
    host::LedgerEntry research_ring_held_;
};

class CudaSession final : public host::DeviceSession {
public:
    CudaSession(CudaModel &model, uint64_t slot, uint64_t max_context_tokens, std::vector<uint32_t> pages);
    ~CudaSession() override;

    // Tokens consumed so far.
    uint64_t position() const { return state_.position; }
    // The session's KV page ids in session order (page i holds logical positions [64 i, 64 i + 64)): reserved at
    // create, or (lazy, M2.10 D3′) a prefix hit's shared pages then its own, bound by bind_pages; empty while unbound.
    const std::vector<uint32_t> &pages() const { return pages_; }
    // A step left a prompt partly consumed (or load_slot opened it): the next step is PREFILL.
    bool prompt_open() const { return prompt_open_; }
    bool lazy() const { return lazy_; }
    // The sampler's output counts, device u32 [vocab] (kind 8; kernel tests only).
    const uint32_t *out_counts() const { return state_.out_counts; }

private:
    friend class CudaDevice;
    CudaModel &model_;
    uint64_t slot_;
    uint64_t max_context_tokens_;
    model::SessionState state_{};
    std::vector<uint32_t> pages_;  // one reference each, released by the destructor (host/kv_pool.h refcounts)
    // M2.9: the slot's 3 undo snapshots (model::UndoState, M29S2-20), bound at create when the model has drafts.
    model::UndoState undo_[model::layout::kUndoSnapshots] = {};
    bool broken_ = false;          // a forward failed after its first launch: the state is undefined
    bool prompt_open_ = false;     // a step consumed part of a prompt: the next step continues it as PREFILL (D2a)
    bool lazy_ = false;            // created without pages (M2.10 D3′): bind_pages binds them once
    bool drafting_ = false;        // M2.9 set_session_draft: the session decodes through speculate_rows
    // M2.9: host copy of state_.proposals (count, then 3 ids), refreshed after every draft run and snapshot load.
    uint32_t proposals_[4] = {};
};

class CudaDevice final : public host::Device {
public:
    // Parses and opens `device_id` (format above); `out` is set only on CERE_OK.
    static cere_status open(const char *device_id, std::unique_ptr<host::Device> *out);
    ~CudaDevice() override;

    std::vector<const host::Support *> supported() const override;
    uint64_t host_budget_bytes() const override;
    host::MemoryEstimate estimate(const host::ModelPlan &plan) const override;
    cere_status allocate(const host::ModelPlan &plan, std::unique_ptr<host::DeviceModel> *out) override;
    cere_status write(host::DeviceModel &model, uint64_t tensor_index, uint64_t offset, const void *bytes,
                      uint64_t len) override;
    cere_status host_resident(const host::DeviceModel &model, uint64_t tensor_index,
                              const uint8_t **out) const override;
    // m216 design note M216S-11: register_staging pins the caller's range (cudaHostRegister portable, one pin per
    // process shared by every device that registers the same range, L7; Held::kStaging);
    // write_staged copies a DEVICE put straight from it on a leased staged stream and synchronizes that stream.
    cere_status register_staging(const uint8_t *base, uint64_t bytes) override;
    void unregister_staging(const uint8_t *base, uint64_t bytes) override;
    cere_status write_staged(host::DeviceModel &model, uint64_t tensor_index, uint64_t offset, const void *bytes,
                             uint64_t len) override;
    cere_status instantiate(host::DeviceModel &model) override;
    // ABI v4: CERE_LEVER_MOE_P3B in a -DCERE_MOE_BACKEND=p3b build (CERE_MOE_P3B), else CERE_LEVER_MOE_K7.
    uint32_t moe_lever() const override;
    // ABI v5 hot swap: model::apply_levers on the instantiated model's Workspace (host-only flag flips, no device work);
    // before INSTANTIATE the selection is only stored (instantiate applies it).
    cere_status set_levers(host::DeviceModel &model, const host::Levers &levers) override;
    bool forward_supported() const override;
    bool faulted() const override;  // DM-2: a sticky CUDA error poisoned this ordinal (probe_sticky)
    // M2.9: true when the model's plan has speculative_draft_tokens > 0 (both MoE backends verify: P3b is row-invariant
    // at m 1..16, M23-D8, [REVISED: M29S2-06d]).
    bool draft_supported(const host::DeviceModel &model) const override;
    bool sampler_supported() const override;
    // `lazy` needs a model with a fixed-state pool (prefix_state_slots > 0), else CERE_ERR_ARGUMENT; it takes the
    // session slot and no page (the slot map stays zero until bind_pages) and consumes K11's armed order unapplied.
    cere_status create_session(host::DeviceModel &model, uint64_t max_context_tokens, bool lazy,
                               std::unique_ptr<host::DeviceSession> *out) override;
    // A session without pages (lazy, unbound) is CERE_ERR_ARGUMENT before any launch, here and in forward/step_rows.
    // `slice` (MixedPD): the call pauses at the layer boundary where its segment's planned layers end (StepSlice).
    cere_status step(host::DeviceModel &model, host::DeviceSession &session, const uint32_t *tokens, uint64_t count,
                     uint64_t limit, const host::StepImages *images, uint64_t *consumed,
                     const host::ItemSampling *sampling, cere_token_score *out, cere_token_logprobs *logprobs,
                     const host::EndStep *end, host::StepSlice *slice) override;
    // host/device.h contract. Per-row preconditions before any launch (row_status[r]): a broken session ->
    // CERE_ERR_INTERNAL; no room for the token -> CERE_ERR_CAPACITY; a session not decode-ready (nothing consumed
    // yet, or a prompt still open) or a token >= vocab_size -> CERE_ERR_ARGUMENT; a sampling outside
    // model::check_sampler -> its status. One participant runs the single-row DECODE forward with its sampling (A1);
    // more run model::forward_decode_rows. CERE_ERR_ARGUMENT for the whole call also covers a session of another
    // model and a row with logprobs but no sampling.
    cere_status step_rows(host::DeviceModel &model, const host::RowArgs *rows, uint32_t row_count,
                          cere_token_score *out, cere_status *row_status) override;
    // M2.9 (host/device.h contract). set_session_draft records the flag: on needs draft_supported(model) (else
    // CERE_ERR_UNSUPPORTED) and a session that has consumed nothing (else CERE_ERR_ARGUMENT); off always succeeds on
    // such a session. The device does not refuse `step` or step_rows of a drafting session: the target state stays
    // exact (only the MTP state those calls skip would differ, and draft-extend rewrites it every round, M29S2-20),
    // so "a drafting session decodes only through speculate_rows" is the host's rule.
    cere_status set_session_draft(host::DeviceModel &model, host::DeviceSession &session, bool on) override;
    // M2.13-L (host/device.h contract, m213l design note C4): records the session's rope_delta in its SessionState
    // (model::rope_table; the graphs serve rope_delta 0 only). CERE_ERR_ARGUMENT for a positive delta or a session of
    // another model.
    cere_status set_rope_delta(host::DeviceModel &model, host::DeviceSession &session, int64_t rope_delta) override;
    // Per-row preconditions before any launch (row_status[r]): a broken session -> CERE_ERR_INTERNAL; no room for
    // its n + 1 verify rows -> CERE_ERR_CAPACITY; a session not drafting or not decode-ready (as step_rows), a token
    // or draft >= vocab_size, max_commit 0, draft_count > 3 or >= max_commit, caller drafts with draft_count 0, a
    // sampling the K12 stage rejects (model::check_sampler), or a vocab_mask with drafts that are not the caller's or
    // more than CERE_MAX_CONSTRAINED_DRAFTS (GrammarSpec GS-02: a constrained row verifies its caller's
    // grammar-checked drafts, row j under mask j) -> CERE_ERR_ARGUMENT. A model without drafts is CERE_ERR_UNSUPPORTED
    // for the whole call.
    // `drafts` null: the session's pending MTP proposals (n = min(draft_count, their count)). The participants run
    // one verify_rows call per group (P8: consecutive rows whose chains fit the plan's verify_rows; one group up to
    // 4 rows); per row the accept (k = leading drafts equal to the previous row's committed id: the raw
    // top-1 for a greedy row (D5), the D6 decision of k13s::spec_accept for a sampled one, its processed top-1 at
    // temperature 0), a = k + 1 truncated after the first is_stop token and at max_commit, then model::verify_commit
    // (a sampled row counts its a committed ids, M29S2-14); `logprobs` receives the committed tokens' entries. Then
    // per participant the draft-extend over its a committed rows and the MTP decode steps (draft below) leave its
    // next proposals.
    cere_status speculate_rows(host::DeviceModel &model, const host::SpecRowArgs *rows, uint32_t row_count,
                               host::SpecRowResult *out, cere_status *row_status) override;
    // M2.9L D3: a GPU round's wall time is its window's cost (the stream is drained before speculate_rows returns).
    bool times_spec_rounds() const override { return true; }
    // DraftAccept: CudaModel::max_chain_drafts (3 for a model without drafts: never asked then).
    uint32_t max_chain_drafts(const host::DeviceModel &model) const override;
    // DraftAccept: the session's proposals_ (count, then the ids).
    uint32_t pending_drafts(const host::DeviceSession &session, uint32_t *ids, uint32_t max) const override;
    // M2.11 (m211 design note D2, D4): model::save_copies / load_copies moved between the session's device state and
    // the snapshot arena (plain locked host memory, memory_plan.h) through a leased ring slot on the executor stream:
    // save gathers the runs D2H into one 32 MiB half while the CPU copies the other, already landed half into `dst`;
    // load copies `src` into one half while the other's H2D scatter runs. Synchronized before return, even after an
    // error. A broken session is CERE_ERR_INTERNAL; save_state of a session whose prompt is still open (M2.6 chunked
    // prefill) and load_state into a session that has consumed tokens are CERE_ERR_ARGUMENT, and a failed load resets
    // the session (fresh again).
    bool snapshot_supported() const override { return true; }
    cere_status save_state(host::DeviceModel &model, host::DeviceSession &session, const host::snapshot::Plan &plan,
                           uint64_t position, uint8_t *dst) override;
    cere_status load_state(host::DeviceModel &model, host::DeviceSession &session, const host::snapshot::Plan &plan,
                           uint64_t position, const uint8_t *src) override;
    // save_state's copy without the open-prompt refusal: the state after a call that ended inside a prompt (D4 (b′)
    // call ends; the ktest digests of K16 (1)/(8)). Same other checks and errors as save_state.
    cere_status save_call_end_state(CudaModel &model, CudaSession &session, const host::snapshot::Plan &plan,
                                    uint64_t position, uint8_t *dst);

    // ---- M2.10 prefix cache (host/device.h contract; m210 design note D3′, D6′, D7, D9). Page ids and state slots
    // are checked against the model's pool (an id outside it, or a page with no reference where an allocated one is
    // required: CERE_ERR_ARGUMENT; retain/release ignore such ids). capture_state and load_slot are one D2D
    // cudaMemcpyAsync of layout().state_bytes between the session's fixed-state run (model::fixed_state_offset) and
    // the state slot, on the executor stream, not synchronized (stream order covers later steps and copies).
    // load_slot: CERE_ERR_ARGUMENT unless the session is bound, has consumed nothing, and 0 < position < its context,
    // position % 64 == 0. save_prefix/load_prefix take the plan's first_row from its kind 1 size and run
    // model::save_copies/load_copies over model::bind_fixed_state (the slot + `pages`) through the ring like
    // save_state/load_state (synchronized); kind 8 (output counts) is written as zeros and not loaded.
    bool prefix_supported() const override { return true; }
    uint64_t free_pages(const host::DeviceModel &model) const override;
    uint32_t page_refs(const host::DeviceModel &model, uint32_t id) const override;
    cere_status retain_pages(host::DeviceModel &model, const uint32_t *ids, uint64_t n) override;
    cere_status release_pages(host::DeviceModel &model, const uint32_t *ids, uint64_t n) override;
    cere_status take_pages(host::DeviceModel &model, uint64_t n, std::vector<uint32_t> *out) override;
    cere_status bind_pages(host::DeviceModel &model, host::DeviceSession &session, const uint32_t *prefix,
                           uint64_t n_prefix) override;
    std::vector<uint32_t> session_page_list(const host::DeviceSession &session) const override;
    uint64_t session_position(const host::DeviceSession &session) const override;
    uint64_t prefill_chunk(const host::DeviceModel &model) const override;
    // The prefill selection signature (cere_model_prefill_signature, BC1): model/prefill_signature.h selection_text of
    // the instantiated model's prefill plans and the static tables; false before INSTANTIATE. A cuBLASLt attribute
    // read failure throws (the call answers CERE_ERR_INTERNAL).
    bool prefill_selection(const host::DeviceModel &model, std::string &out) const override;
    // PrefixCapture G1: ceil(model::end_kv_bytes / state_bytes) when the cold call is dense (cls <= k6::kMaxContext,
    // above the K6-only contexts), with either prefill attention (P4's pinned resume equals the cold rows, BeatSglang
    // S7); else 0.
    uint32_t end_kv_slots(const host::DeviceModel &model, uint64_t rows, uint64_t cls) const override;
    // m210 D15⁗″ (ResumeExact): at the levers the row-bit proof covers (k2_gemm cublaslt or small_m, prefill_attention
    // cuda_core, k2_weights bf16) and a class of at least kRowSigMinRows, the choices that change a row's bits: the
    // split-K pin range at cls of moe_shared_gate_up and of attn_qkv (S2; 0 when that configuration does not split
    // K) and, at s = 0, whether the call is dense (S3: cls <= k6::kMaxContext). Every other choice (bucket
    // algorithms, P3b tactics, K6/K9 attention at s > 0) leaves a row's bits unchanged (S5, K16 (13) K16RowSig). Any
    // other levers or class: cls.
    uint64_t row_sig(const host::DeviceModel &model, uint64_t s, uint64_t cls) const override;
    static constexpr uint64_t kRowSigMinRows = 65;
    cere_status capture_state(host::DeviceModel &model, const host::DeviceSession &session, uint32_t slot) override;
    cere_status load_slot(host::DeviceModel &model, host::DeviceSession &session, uint32_t slot,
                          uint64_t position) override;
    cere_status save_prefix(host::DeviceModel &model, const uint32_t *pages, uint64_t n_pages, uint32_t slot,
                            const host::snapshot::Plan &plan, uint64_t position, uint8_t *dst) override;
    cere_status load_prefix(host::DeviceModel &model, const uint32_t *pages, uint64_t n_pages, uint32_t slot,
                            const host::snapshot::Plan &plan, uint64_t position, const uint8_t *src) override;
    // Track LoRA (kernels/lora.h): the pool is one cudaMalloc beside the arena, made at reserve (before INSTANTIATE,
    // whose apply_levers hands it to every plan) and freed with the model; load copies on the table's own stream.
    uint64_t lora_bytes(const host::DeviceModel &model, uint32_t adapters, uint32_t rank) const override;
    cere_status lora_reserve(host::DeviceModel &model, uint32_t adapters, uint32_t rank) override;
    cere_status lora_load(host::DeviceModel &model, uint32_t slot, uint32_t rank, float scale, uint64_t count,
                          const uint32_t *targets, const void *const *a, const void *const *b) override;
    cere_status lora_unload(host::DeviceModel &model, uint32_t slot) override;
    cere_status set_session_lora(host::DeviceModel &model, host::DeviceSession &session, int32_t slot) override;
    // WeightLab research (research/hook.h): reserve allocates the staging and plan memory (one cudaMalloc beside the
    // arena) and the pinned ring before INSTANTIATE, freed with the model; plan_load copies a plan's operands into its
    // slot on the executor stream and waits; a session's binding is read by every forward call of its rows.
    bool research_supported() const override { return true; }
    cere_status research_reserve(host::DeviceModel &model, uint64_t staging_bytes, uint64_t plan_bytes,
                                 uint64_t host_bytes, uint8_t **ring) override;
    cere_status research_plan_load(host::DeviceModel &model, uint32_t slot, const research::Plan &plan) override;
    cere_status set_session_research(host::DeviceModel &model, host::DeviceSession &session,
                                     const research::Binding *binding) override;
    cere_status research_sync(host::DeviceModel &model) override;

    // One model::forward call of n tokens as `operation` (CERE_OP_PREFILL / CERE_OP_DECODE) on the executor stream
    // (step's building block; kernel tests call it with taps). Same preconditions and errors as model::forward, plus
    // CERE_ERR_CAPACITY when position + n exceeds the session's max_context_tokens and CERE_ERR_INTERNAL on a session
    // left unusable by an earlier failure. `segment`: model::forward's (MixedPD; step owns the park and its session).
    cere_status forward(CudaModel &model, CudaSession &session, uint32_t operation, const uint32_t *tokens,
                        uint32_t n, model::ForwardTaps *taps, cere_token_score *out,
                        const model::SamplerArgs *sampler = nullptr, cere_token_logprobs *logprobs = nullptr,
                        const host::StepImages *images = nullptr, const model::EndCall *end = nullptr,
                        model::Segment *segment = nullptr);
    // M2.9 D2: after a forward of n rows (tokens, host [n]) on a drafting session, its MTP extend over them (draft at
    // positions position - n .. position - 1 on the hc rows the forward left in the workspace; next ids tokens[1, n)
    // then `next`: the next prompt token, or the token the forward scored). step() runs it after every forward;
    // kernel tests feeding a drafting session through forward() run it too, so every row the MTP attends to holds the
    // MTP's own K/V, never a pool page's earlier bytes. CERE_OK without work on a session that does not draft; a
    // failure breaks the session. `prompt_open`: the forward was a prompt chunk before the prompt's last (draft's,
    // no decode steps). `cls`: the extend's class pin (mtp::MtpCall::cls; 0 = none, D15⁗′). `rope`: the extend rows'
    // rotary inputs (an image prompt's chunk: the forward's model::prompt_rope; null: position + rope_delta).
    // `end_kv`: the MTP layer's EndKv of a chunk-0 end state's call (mtp::MtpCall::end_kv).
    cere_status extend_draft(CudaModel &model, CudaSession &session, const uint32_t *tokens, uint32_t n,
                             uint32_t next, bool prompt_open, uint32_t cls = 0, const model::Rope *rope = nullptr,
                             const model::EndKv *end_kv = nullptr);
    // M2.9 kernel-test entry (K14): one model::mtp::forward_logits call of `call` on `session` (a drafting model) on
    // the executor stream, synchronized before return; `logits` (device BF16 [m, draft count], nullable) gets the hot
    // logits. CERE_ERR_ARGUMENT for a model without drafts or a foreign session; otherwise mtp::forward's errors.
    cere_status mtp_forward(CudaModel &model, CudaSession &session, const model::mtp::MtpCall &call,
                            model::bf16 *logits);
    // One model::forward_decode_rows call (row r feeds tokens[r] to sessions[r], with samplers[r] and logprobs[r]
    // when those arrays are non-null) on the executor stream (step_rows's building block for more than one
    // participant; kernel tests call it with taps or samplers). Same preconditions and errors as
    // model::forward_decode_rows, plus CERE_ERR_INTERNAL on a session left unusable by an earlier failure; a failure
    // other than CERE_ERR_ARGUMENT leaves every row's session unusable.
    cere_status forward_rows(CudaModel &model, CudaSession *const *sessions, uint32_t rows, const uint32_t *tokens,
                             model::ForwardTaps *taps, cere_token_score *out,
                             const model::SamplerArgs *const *samplers = nullptr,
                             cere_token_logprobs *const *logprobs = nullptr);
    // M2.9: one model::forward_verify_rows call (chain c: sessions[c] feeds tokens[c][0 .. n[c]] at its position p,
    // its top-2 rows at out[Σ_{c' < c} (n[c'] + 1) ...]) then model::verify_commit of commit[c] in [1, n[c] + 1] tokens
    // per chain (speculate_rows's building block; kernel tests call it with taps or a forced commit). Same preconditions
    // as model::forward_verify_rows plus commit's range (CERE_ERR_ARGUMENT before anything runs), CERE_ERR_UNSUPPORTED
    // for a model without drafts and CERE_ERR_INTERNAL on a broken session; any later failure breaks every chain's
    // session.
    cere_status verify_rows(CudaModel &model, CudaSession *const *sessions, const uint32_t *const *tokens,
                            const uint32_t *n, uint32_t chains, model::ForwardTaps *taps, cere_token_score *out,
                            const uint32_t *commit);

    // The executor's own allocations, exact (see Memory ledger above).
    const host::MemoryLedger &ledger() const { return ledger_; }

private:
    struct Slot {
        uint8_t *host = nullptr;  // kSlotBytes of the ring
        cudaStream_t stream = nullptr;
        cudaEvent_t done[2] = {nullptr, nullptr};  // the copy out of each half has finished
    };
    class Lease;
    class StagedLease;

    CudaDevice(int ordinal, host::ShareConfig share, bool release);
    cere_status init();
    // Shared mode at ALLOCATE: maps the segment of the region [0, region_bytes) of `plan`/`l` and allocates the
    // staging block (see Shared HOST_PINNED region above); allocate registers the region at its end (M216S-12).
    cere_status allocate_shared(const host::ModelPlan &plan, CudaModel &m);
    // The segment name of `plan`/`l` (shared mode).
    std::string segment_name(const host::ModelPlan &plan, const ModelLayout &l) const;
    cere_status copy_to_device(uint8_t *dst, const uint8_t *src, uint64_t len);
    // The state runs through a leased ring slot on stream_ (save_state / load_state above); the stream is drained
    // before return, so no copy still touches the slot or the host image afterwards.
    cere_status stage_to_host(const std::vector<model::StateCopy> &copies, uint8_t *dst);
    cere_status stage_to_device(const std::vector<model::StateCopy> &copies, const uint8_t *src);
    // verify_rows's halves (speculate_rows accepts between them): the model::forward_verify_rows call (a failure
    // other than CERE_ERR_ARGUMENT breaks every chain's session) and the chains' verify_commit calls (enqueued, then
    // the stream drained; any failure breaks every chain's session).
    cere_status verify_chains(CudaModel &model, CudaSession *const *sessions, const model::VerifyRow *rows,
                              uint32_t chains, model::ForwardTaps *taps, cere_token_score *out,
                              cere_token_logprobs *out_logprobs = nullptr);
    cere_status commit_chains(CudaModel &model, CudaSession *const *sessions, const model::VerifyRow *rows,
                              const uint32_t *commit, uint32_t chains);
    // P8: one group of speculate_rows (the whole-call checks done): the per-row preconditions, one verify call of its
    // participants, the accept, the commit and the draft-extend, as speculate_rows describes; its rows' chains fit
    // the plan's verify_rows.
    cere_status speculate_group(CudaModel &model, const host::SpecRowArgs *rows, uint32_t row_count,
                                host::SpecRowResult *out, cere_status *row_status);
    // M2.9 D2 (F3): one drafting session's MTP extend over `rows` rows at positions p0 .. p0 + rows - 1 (the target's
    // pre-mixer hc rows `hc`, device [rows, 10,240], overwritten by the MTP's own; `next` device [rows], the token at
    // each row's position + 1), scored at its last row only (d1, the token after the session's unfed one at p0 +
    // rows), then layout::draft_decode_steps MTP decode steps at p0 + rows and p0 + rows + 1 (up to 2 while inside
    // the context, none when `prompt_open`: an extend of a prompt chunk before the prompt's last) (d2, d3): the
    // session's proposals are [count, d1, d2, d3], also copied to proposals_. Drains the stream; any failure breaks
    // the session. `cls`: the extend's class pin (MtpCall::cls; the decode steps take none); `rope`: the extend's
    // rotary inputs (MtpCall::rope; null: position + rope_delta, the decode steps' always); `end_kv`: the extend's
    // MtpCall::end_kv (the decode steps take none). The draft graph serves only `rope` and `end_kv` null at
    // rope_delta 0 with both decode steps.
    cere_status draft(CudaModel &model, CudaSession &session, uint64_t p0, uint32_t rows, model::bf16 *hc,
                      const uint32_t *next, uint32_t cls = 0, const model::Rope *rope = nullptr,
                      const model::EndKv *end_kv = nullptr, bool prompt_open = false);
    // P11R-21: whether draft_rows serves the `count` sessions of a group whose chain p committed commit[p] rows: 2 ..
    // kMaxRows sessions, Σ commit within the MTP rows plans, both decode steps of every session inside its context
    // (draft's eager loop runs both too), and the Workspace holds draft_rows's regions.
    bool draft_rows_fit(const CudaModel &model, CudaSession *const *sessions, const uint32_t *commit,
                        uint32_t count) const;
    // P11R-21: draft for the `count` sessions at once, each as draft(model, *sessions[p], position - commit[p],
    // commit[p], the verify's hc rows at base[p], ...) would, bit for bit (model::mtp::draft_rows: one MTP forward
    // per pass over every session's rows). The caller staged each session's committed ids contiguously at kExtendAt
    // of ws.ids32 and the proposal counts (1 + kDraftSteps) at kDraftAt. One drain; any failure breaks every session.
    cere_status draft_rows(CudaModel &model, CudaSession *const *sessions, const uint32_t *commit,
                           const uint32_t *base, uint32_t count);

    int ordinal_;
    host::ShareConfig share_;  // sharing on when share_.enabled()
    bool release_;             // release_device=1: the last executor on the ordinal resets its primary context
    uint64_t host_budget_ = 0;
    uint8_t *ring_ = nullptr;
    cudaStream_t stream_ = nullptr;  // the executor's stream (session reset, INSTANTIATE, every forward)
    cublasLtHandle_t gemm_ = nullptr;  // K2's per-device handle
    Slot slots_[kRingSlots];
    std::mutex ring_mutex_;
    std::condition_variable ring_free_;
    std::vector<size_t> free_slots_;
    cudaStream_t staged_streams_[kStagedStreams] = {};  // write_staged's streams (StagedLease)
    std::mutex staged_mutex_;
    std::condition_variable staged_free_;
    std::vector<size_t> free_staged_;
    std::atomic<bool> forward_ready_{false};  // a model was instantiated
    // DM-2: when a failed call leaves a sticky CUDA error behind (the context is dead: every later call on the ordinal
    // fails), poisons the ordinal (faulted(), one `cere: device_poisoned ordinal=<n> error=<name>` line) so the host
    // refuses new work with INTERNAL. Only for a tracked ordinal (below kTrackedDevices); cleared by the device release.
    void probe_sticky() const;
    host::MemoryLedger ledger_;               // declared before every entry that points at it (ring_held_ below)
    host::LedgerEntry ring_held_;             // ring_ (Held::kRing)
    // The registered staging ranges (Held::kStaging), by base; register/unregister_staging run one at a time.
    std::map<const uint8_t *, host::LedgerEntry> staging_held_;
    std::vector<uint64_t> mtp_positions_;     // draft's host MtpCall positions (grows to the largest MTP extend)
};

}  // namespace cere::device
