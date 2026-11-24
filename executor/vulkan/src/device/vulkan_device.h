// VulkanAmd P3 T2 (device half): VulkanDevice / VulkanModel / VulkanSession, the cere::host::Device of
// libcere_executor_vulkan.so (vulkan-amd-p3-design design note sections 1, 3, 4; task T2).
//
// What it does now (first-light scope: A4 text, one session, 2,049..32,768 tokens, no snapshot / LoRA / vision /
// MixedPD park; vulkan-amd design note P3; the device-tier prefix cache and EndStep since D15, below):
//   open        device id "N" (decimal Vulkan GPU ordinal, cuda_device.cpp:89-127's grammar); vk::Context::create
//   estimate    vk_plan_layout + vk_estimate (exact block sums); a plan the layout refuses estimates 0 so the host
//               reaches allocate, which returns the layout's own status (UNSUPPORTED / CAPACITY / ARGUMENT)
//   allocate    the layout's <= 4 GiB blocks (device-local, and host-heap for the PLE table), RADV's requirement and
//               coherence checked per block, ledger per block, names bound to addresses (vulkan_bind.h), the K4 block
//               table written; nothing kept on any error
//   write       memcpy into the persistent mapping at vk_locate's pieces (no GPU work, no lock: disjoint pieces)
//   instantiate probes (K2 GEMM, K7), pipeline warm-up (every *_pipeline_specs list the kernels provide), derived host
//               arithmetic (A_log F32, MoE alphas, a1/a2), one command buffer: RoPE table, every K2 row info
//               (VkLayout::winfo with each weight's own format), K7 row info per layer, MTP hot lm_head + K13 info
//   sessions    one slot, KV pages from host::PagePool, slot map + zeroed state + PLE EOS history through the mapping
//   levers      the Vulkan eager set (crates/cere-runtime/src/levers.rs arch `vulkan`) plus, P4-E, prefill_attention
//               p4 = the cooperative-matrix prefill attention (vkk6 / vkk9 Prefill::kCoopmat) on a device with
//               VK_KHR_cooperative_matrix whose seal-time probe (vkk6::coopmat_probe) passes; refused otherwise
//   forward     (T5, vulkan_forward.h) step: one PREFILL chunk (<= workspace_rows) or one DECODE token per call, one
//               command buffer per call; step_rows: the single-row DECODE (max_sessions 1); greedy and the K12 sampler
//               (sampling, logprobs, vocab mask); forward_supported / sampler_supported true once a model is sealed
//   speculation (T6, vulkan_spec.cpp + vulkan_forward.h): a plan with speculative_draft_tokens 4 and the MTP tensors
//               and token map (kDraftForwardImplemented); set_session_draft; a drafting session's PREFILL steps run
//               the MTP extend; speculate_rows (one participant): verify (one command buffer, rows = 1 + drafts <= 4),
//               accept on the host (greedy D5 / sampled D6 via k13s::spec_accept on the device), commit + draft-extend
//               + two MTP decode steps (one command buffer); max_chain_drafts 3; pending_drafts; times_spec_rounds
//               false (SpecCosts::untimed(), OD-6)
// Refused: a step with images / EndStep is UNSUPPORTED; a StepSlice runs whole. (P4-F: a constrained (vocab-mask) row
// with the caller's drafts, grammar drafts / GrammarSpec GS-02, runs with one mask per verify row.)
// Faults: any failed submit / device loss makes the Context sticky-faulted; faulted() reports it (one
// "cere: device_poisoned vk=<N> error=<VkResult>" line), the host answers INTERNAL (api.cpp refuse_faulted).
// Nothing here exits, aborts or throws (std::bad_alloc is caught and mapped to CERE_ERR_CAPACITY).
//
// ---- T5 interface (where the forward adds code) ----
//   VulkanDevice::ctx()            the vk::Context (one thread at a time: the host serializes every call but write)
//   VulkanModel::weights()         VkModelWeights: per layer VkLayerWeights (hc, attention, vkk5 view + gdn_dense with
//                                  MXFP8 scales, vkk7 view + shared_dense), PLE vkk4::Weights (table view on the 7
//                                  host-heap blocks), MTP (drafts), derived RoPE / A_log / alphas
//   VulkanModel::workspace()       VkWorkspace: carve_workspace_rows buffers, scratch, K2 GEMM workspace, MTP hc rows
//   VulkanModel::io(kind)          IoRegion {addr, host map, bytes} of the IO block (ids, top-2, logprobs, bound fault,
//                                  K12 bias / mask, table ring) for host writes before a submit / reads after it
//   VulkanModel::region_addr(..)   any VkLayout region (kUndo, kSpecSampler, ...)
//   VulkanSession::state()         VkSessionState: slot map, GDN conv/ssm, PLE history/conv, QSA rings, out counts,
//                                  the shared KV pool entries; position(), set_position(), broken flag
//   VulkanDevice::step / step_rows / speculate_rows   the methods T5 / T6 fill (now CERE_ERR_UNSUPPORTED)
#pragma once

#include "cere_executor.h"
#include "device/vk_memory_plan.h"
#include "device/vulkan_bind.h"
#include "device/vulkan_forward.h"
#include "host/device.h"
#include "host/kv_pool.h"
#include "host/memory_ledger.h"
#include "launch/vk_launch.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cere::vkdev {

// Policy (AGENTS.md 13).
constexpr uint64_t kHostReserveBytes = uint64_t(8) << 30;  // host_budget = MemAvailable at open - this (CUDA's rule)
constexpr uint32_t kMaxOrdinalDigits = 9;
// T6: the MTP draft and speculate_rows are built. False keeps every plan with drafts refused at allocate
// (draft_supported false): the switch for a build whose speculation is not proven lossless.
constexpr bool kDraftForwardImplemented = true;

class VulkanDevice;

// What INSTANTIATE did (logged once, kept for tests).
struct InstantiateReport {
    double probe_ms = 0, warm_ms = 0, host_ms = 0, gpu_ms = 0, total_ms = 0;
    size_t warm_specs = 0, pipelines = 0;
    uint32_t winfo_entries = 0, moe_info_layers = 0;
    bool mtp = false;  // T6: the draft token map, hot lm_head and K13 row info were prepared
    std::string warm_lists;  // the *_pipeline_specs lists used
    // P4-E: the coopmat prefill attention probe (vkk6::coopmat_probe words; the path needs FP16 subnormals kept:
    // out[0] == kCoopmatProbeF16 and out[1] == kCoopmatProbeProduct) and whether its pipelines were warmed.
    bool coopmat_probed = false, coopmat_ok = false;
    uint32_t coopmat_probe[4] = {};
    std::string coopmat_why;  // why the path is unavailable (empty when coopmat_ok)
};

// P4-E: the values vkk6::coopmat_probe must return for the coopmat prefill attention's error bounds to hold
// (shaders/p4att_common.comp; an internal benchmark report "What the forward must
// change" 1): FP16 subnormals kept in a shader conversion (f16(2^-20) = 0x0010) and in the matrix unit (16 products of
// 2^-20 x 1 = 2^-16).
constexpr uint32_t kCoopmatProbeF16 = 0x0010u;
constexpr uint32_t kCoopmatProbeProduct = 0x37800000u;

class VulkanModel final : public host::DeviceModel {
public:
    ~VulkanModel() override;

    const host::ModelPlan &plan() const { return plan_; }
    const VkLayout &layout() const { return layout_; }
    const std::string &layout_digest() const { return digest_; }
    const VkModelWeights &weights() const { return *weights_; }
    VkModelWeights &weights() { return *weights_; }
    const VkWorkspace &workspace() const { return ws_; }
    const std::vector<vk::Buffer> &blocks() const { return blocks_; }
    const std::vector<uint64_t> &block_addrs() const { return addrs_; }
    bool instantiated() const { return instantiated_; }
    // T5: the forward's per-model statistics (timings, command counts, the K2 exact-fallback counters).
    VkForwardStats &forward_stats() { return fstats_; }
    const VkForwardStats &forward_stats() const { return fstats_; }
    const InstantiateReport &instantiate_report() const { return report_; }
    const host::Levers &levers() const { return levers_; }
    // P4-E: the > 16-row prefill attention path the forward records for K6 dense / K9 sparse: the cooperative-matrix
    // kernels (Prefill::kCoopmat) when the active prefill_attention lever is p4 (the device accepts p4 only when the
    // seal-time probe passed), else the FP32 kernels (P2). <= 16-row calls (decode, verify, MTP decode) are the split
    // path either way. prefill_attention_path() names it for the logs and tests.
    bool prefill_coopmat() const { return levers_.prefill_attention == CERE_LEVER_PREFILL_P4 && report_.coopmat_ok; }
    const char *prefill_attention_path() const { return prefill_coopmat() ? "coopmat" : "fp32"; }

    uint64_t region_addr(VkRegionKind kind, uint32_t index = 0) const;
    // D15 prefix cache: state slot `slot` of the fixed-state pool (VkRegionKind::kStateSlots, layout_.state_bytes
    // each), 0 outside [0, layout_.state_slots).
    uint64_t state_slot_addr(uint64_t slot) const;
    uint8_t *region_map(VkRegionKind kind, uint32_t index = 0) const;
    struct IoRegion {
        uint64_t addr = 0;
        uint8_t *host = nullptr;
        uint64_t bytes = 0;
    };
    IoRegion io(VkRegionKind kind) const;
    // Host mapping of device address `addr` (inside one of this model's blocks), null otherwise.
    uint8_t *host_of(uint64_t addr) const;
    // Copies tensor bytes [offset, offset + len) out of the mappings (tests; the forward never needs it). ARGUMENT
    // outside the tensor or for a DISCARD tensor.
    cere_status read(uint64_t tensor, uint64_t offset, void *dst, uint64_t len) const;

    // Bytes held, as the ledger records them (device-local blocks = kDevice, host-heap = kMapped).
    uint64_t held_device_bytes() const;
    uint64_t held_host_bytes() const;

private:
    friend class VulkanDevice;
    friend class VulkanSession;
    explicit VulkanModel(std::shared_ptr<vk::Context> ctx) : ctx_(std::move(ctx)) {}

    std::shared_ptr<vk::Context> ctx_;
    host::ModelPlan plan_;
    VkLayout layout_;
    std::string digest_;
    std::vector<vk::Buffer> blocks_;
    std::vector<uint64_t> addrs_;
    std::vector<std::unique_ptr<host::LedgerEntry>> held_;
    std::unique_ptr<VkModelWeights> weights_;
    VkWorkspace ws_;
    host::Levers levers_;
    bool instantiated_ = false;
    bool instantiate_attempted_ = false;
    InstantiateReport report_;
    VkForwardStats fstats_;
    // Sessions (one slot per plan.max_sessions) and the KV page pool, under slots_mutex_.
    std::mutex slots_mutex_;
    std::vector<bool> slot_used_;
    std::unique_ptr<host::PagePool> pool_;
};

class VulkanSession final : public host::DeviceSession {
public:
    ~VulkanSession() override;
    const VkSessionState &state() const { return state_; }
    uint64_t slot() const { return slot_; }
    uint64_t max_context_tokens() const { return max_context_tokens_; }
    uint64_t position() const { return position_; }
    void set_position(uint64_t p) { position_ = p; }
    bool broken() const { return broken_; }
    void set_broken() { broken_ = true; }
    // The prompt continues in a later step (cuda_device.h prompt_open_: the rest of an open prompt is PREFILL).
    bool prompt_open() const { return prompt_open_; }
    void set_prompt_open(bool v) { prompt_open_ = v; }
    // D15 (M2.10 D3'): a lazy session (prefix model) has no pages until bind_pages.
    bool lazy() const { return lazy_; }
    const std::vector<uint32_t> &pages() const { return pages_; }
    VulkanModel &model() const { return model_; }
    // T6: the session drafts (set_session_draft); its pending MTP proposals (host copy: count 0..3, then the ids, as
    // CUDA's CudaSession::proposals_); its undo snapshots (layout::kUndoSnapshots, bound at create on a draft model).
    bool drafting() const { return drafting_; }
    const uint32_t *proposals() const { return proposals_; }
    void set_proposals(const uint32_t p[4]) {
        for (int i = 0; i < 4; ++i) proposals_[i] = p[i];
    }
    VkUndoState &undo(uint32_t j) { return undo_[j]; }
    uint32_t undo_count() const { return undo_count_; }

private:
    friend class VulkanDevice;
    VulkanSession(VulkanModel &m, uint64_t slot, uint64_t ctx, std::vector<uint32_t> pages)
        : model_(m), slot_(slot), max_context_tokens_(ctx), pages_(std::move(pages)) {}
    VulkanModel &model_;
    uint64_t slot_;
    uint64_t max_context_tokens_;
    std::vector<uint32_t> pages_;
    VkSessionState state_;
    uint64_t position_ = 0;
    bool broken_ = false;
    bool prompt_open_ = false;
    bool lazy_ = false;
    bool drafting_ = false;
    uint32_t proposals_[4] = {0, 0, 0, 0};
    VkUndoState undo_[kVkMaxChainDrafts];
    uint32_t undo_count_ = 0;
};

class VulkanDevice final : public host::Device {
public:
    // create_device's implementation: grammar "N" (decimal ordinal among the Vulkan GPUs, no sign, no leading zero,
    // <= 9 digits; else ARGUMENT), "N,key=value..." options: release_device / host_share keys are UNSUPPORTED (not
    // built on Vulkan), anything else ARGUMENT; a non-decimal id or no such device / missing feature UNSUPPORTED.
    static cere_status open(const char *device_id, std::unique_ptr<host::Device> *out);
    ~VulkanDevice() override;

    vk::Context &ctx() { return *ctx_; }
    const vk::Context &ctx() const { return *ctx_; }
    uint32_t ordinal() const { return ordinal_; }
    const host::MemoryLedger &ledger() const { return ledger_; }

    bool faulted() const override;
    std::vector<const host::Support *> supported() const override;
    uint64_t host_budget_bytes() const override { return host_budget_; }
    host::MemoryEstimate estimate(const host::ModelPlan &plan) const override;
    cere_status allocate(const host::ModelPlan &plan, std::unique_ptr<host::DeviceModel> *out) override;
    cere_status write(host::DeviceModel &model, uint64_t tensor_index, uint64_t offset, const void *bytes,
                      uint64_t len) override;
    uint32_t moe_lever() const override { return CERE_LEVER_MOE_K7; }
    bool k2_tail_split() const override { return true; }
    cere_status set_levers(host::DeviceModel &model, const host::Levers &levers) override;
    cere_status instantiate(host::DeviceModel &model) override;
    // True once a model was sealed (cuda_device.cpp:1082-1088: forward_ready_ after a successful instantiate).
    bool forward_supported() const override { return forward_ready_.load(); }
    bool sampler_supported() const override { return forward_supported(); }
    cere_status create_session(host::DeviceModel &model, uint64_t max_context_tokens, bool lazy,
                               std::unique_ptr<host::DeviceSession> *out) override;
    cere_status step(host::DeviceModel &model, host::DeviceSession &session, const uint32_t *tokens, uint64_t count,
                     uint64_t limit, const host::StepImages *images, uint64_t *consumed,
                     const host::ItemSampling *sampling, cere_token_score *out, cere_token_logprobs *logprobs,
                     const host::EndStep *end, host::StepSlice *slice) override;
    cere_status step_rows(host::DeviceModel &model, const host::RowArgs *rows, uint32_t row_count,
                          cere_token_score *out, cere_status *row_status) override;
    // T6 (vulkan_spec.cpp; cuda_device.cpp:1084-1086, 2104-2112, 2342-2554).
    bool draft_supported(const host::DeviceModel &model) const override;
    cere_status set_session_draft(host::DeviceModel &model, host::DeviceSession &session, bool on) override;
    cere_status speculate_rows(host::DeviceModel &model, const host::SpecRowArgs *rows, uint32_t row_count,
                               host::SpecRowResult *out, cere_status *row_status) override;
    uint32_t max_chain_drafts(const host::DeviceModel &model) const override;
    uint32_t pending_drafts(const host::DeviceSession &session, uint32_t *ids, uint32_t max) const override;
    // OD-6: no load-time round calibration (SpecCosts::untimed(), api.cpp calibrate_costs skips the model); the host
    // prices adaptive picks with the untimed table, fixed-window tickets need no costs.
    bool times_spec_rounds() const override { return false; }
    cere_status set_rope_delta(host::DeviceModel &model, host::DeviceSession &session, int64_t rope_delta) override;
    // M2.10 read-only introspection the host calls on every model.
    uint64_t free_pages(const host::DeviceModel &model) const override;
    uint32_t page_refs(const host::DeviceModel &model, uint32_t id) const override;
    std::vector<uint32_t> session_page_list(const host::DeviceSession &session) const override;
    uint64_t session_position(const host::DeviceSession &session) const override;
    uint64_t prefill_chunk(const host::DeviceModel &model) const override;
    // D15: the M2.10 prefix cache (cuda_device.cpp:1815-1990 restated; host/device.h :405-475). A lazy session
    // (create_session lazy) binds its pages at bind_pages; capture_state / load_slot copy the session's fixed-state
    // run to / from a state slot on the GPU (one k1 copy, submit_wait). end_kv_slots is CUDA's rule; row_sig keeps the
    // default (sigma = cls). No host tier (snapshot_supported false: save_prefix / load_prefix UNSUPPORTED).
    bool prefix_supported() const override { return true; }
    [[nodiscard]] cere_status retain_pages(host::DeviceModel &model, const uint32_t *ids, uint64_t n) override;
    [[nodiscard]] cere_status release_pages(host::DeviceModel &model, const uint32_t *ids, uint64_t n) override;
    cere_status take_pages(host::DeviceModel &model, uint64_t n, std::vector<uint32_t> *out) override;
    cere_status bind_pages(host::DeviceModel &model, host::DeviceSession &session, const uint32_t *prefix,
                           uint64_t n_prefix) override;
    uint32_t end_kv_slots(const host::DeviceModel &model, uint64_t rows, uint64_t cls) const override;
    cere_status capture_state(host::DeviceModel &model, const host::DeviceSession &session, uint32_t slot) override;
    cere_status load_slot(host::DeviceModel &model, host::DeviceSession &session, uint32_t slot,
                          uint64_t position) override;

    // allocate without the name binding (the device-internal test's synthetic plans, which carry no full model):
    // blocks, checks, ledger, K4 block table. allocate() = this + vk_bind_weights + workspace binding.
    cere_status allocate_storage(const host::ModelPlan &plan, std::unique_ptr<VulkanModel> *out, std::string *why);

    // The levers the Vulkan library runs (design 5.4): CERE_OK or UNSUPPORTED naming the first other one.
    static cere_status check_levers(const host::Levers &levers, std::string *why);
    // P4-E: the device half of the prefill_attention check: p4 needs VK_KHR_cooperative_matrix and, on a model whose
    // seal ran (m non-null), the passed coopmat probe. CERE_OK for cuda_core.
    cere_status check_prefill_attention(const VulkanModel *m, const host::Levers &levers, std::string *why) const;

private:
    VulkanDevice() = default;
    cere_status instantiate_impl(VulkanModel &m, std::string *why);
    // One GPU copy of `bytes` from src to dst (device addresses), recorded, submitted and waited for.
    cere_status copy_device(uint64_t dst, uint64_t src, uint64_t bytes);
    // session_set_pages (model.cpp:1044-1063) through the mapping: slot map position p -> page[p / 64] * 64 + p % 64.
    static cere_status write_slot_map(const VulkanModel &m, const VulkanSession &s);

    std::shared_ptr<vk::Context> ctx_;
    uint32_t ordinal_ = 0;
    uint64_t host_budget_ = 0;
    host::MemoryLedger ledger_;
    mutable std::atomic<bool> poisoned_reported_{false};
    std::atomic<bool> forward_ready_{false};
};

}  // namespace cere::vkdev
