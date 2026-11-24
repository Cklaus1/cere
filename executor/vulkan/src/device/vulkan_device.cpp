// VulkanAmd P3 T2 (device half): see vulkan_device.h. Cited CUDA counterparts (executor/cuda/src/device/cuda_device.cpp)
// are restated, never called: open :537-610, supported :616, estimate :626, allocate :695-847, write :873-894,
// instantiate :993-1052, set_levers :1062, create_session :1090-1170 (bind_session / session_reset / session_set_pages,
// model.cpp:1004-1063).
#include "device/vulkan_device.h"

#include "device/vulkan_warm.h"
#include "host/host_share.h"
#include "host/support.h"
#include "kernels/k13_moe_bf16_vk.h"
#include "kernels/k1_primitives_vk.h"
#include "kernels/k2_gemm_vk.h"
#include "kernels/k4_ple_vk.h"
#include "kernels/k6_attention_vk.h"
#include "kernels/k7_moe_vk.h"
#include "model/layout.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <string_view>
#include <unordered_map>

namespace cere::vkdev {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

uint64_t mem_available_bytes() {
    std::FILE *f = std::fopen("/proc/meminfo", "r");
    if (f == nullptr) return 0;
    char line[256];
    unsigned long long kib = 0;
    bool found = false;
    while (!found && std::fgets(line, sizeof line, f) != nullptr) {
        found = std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1;
    }
    std::fclose(f);
    return found ? uint64_t(kib) * 1024 : 0;
}

// cuda_device.cpp:89-127's grammar for a Vulkan ordinal. Options after a comma are parsed in full first (a malformed one
// is ARGUMENT, as on CUDA); a well-formed release_device or host_share option is UNSUPPORTED (neither is built here).
cere_status parse_device_id(const char *id, uint32_t *ordinal) {
    const std::string full(id);
    const size_t comma = full.find(',');
    const std::string s = full.substr(0, comma);
    if (s.empty()) return CERE_ERR_ARGUMENT;
    const size_t sign = (s[0] == '+' || s[0] == '-') ? 1 : 0;
    const std::string digits = s.substr(sign);
    const bool decimal = !digits.empty() && std::all_of(digits.begin(), digits.end(),
                                                         [](char c) { return c >= '0' && c <= '9'; });
    if (!decimal) return sign != 0 ? CERE_ERR_ARGUMENT : CERE_ERR_UNSUPPORTED;
    if (sign != 0 || (digits.size() > 1 && digits[0] == '0') || digits.size() > kMaxOrdinalDigits) {
        return CERE_ERR_ARGUMENT;
    }
    host::ShareConfig cfg;
    bool release = false, share = false;
    for (size_t pos = comma; pos != std::string::npos;) {
        const size_t next = full.find(',', pos + 1);
        const std::string option = full.substr(pos + 1, next == std::string::npos ? std::string::npos : next - pos - 1);
        const size_t eq = option.find('=');
        if (eq == std::string::npos) return CERE_ERR_ARGUMENT;
        const std::string key = option.substr(0, eq), value = option.substr(eq + 1);
        if (key == "release_device") {
            if (release || value != "1") return CERE_ERR_ARGUMENT;
            release = true;
        } else {
            const int r = host::parse_share_option(key, value, &cfg);
            if (r != 1) return CERE_ERR_ARGUMENT;
            share = true;
        }
        pos = next;
    }
    if (share && !host::share_config_complete(cfg)) return CERE_ERR_ARGUMENT;
    if (release || share) return CERE_ERR_UNSUPPORTED;
    *ordinal = uint32_t(std::stoul(digits));
    return CERE_OK;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

void say(const std::string &line) { std::fprintf(stderr, "cere: %s\n", line.c_str()); }

}  // namespace

// ---- VulkanModel ---------------------------------------------------------------------------------------------------

VulkanModel::~VulkanModel() {
    // T5: the forward's statistics, once (no prompt or token content: AGENTS.md 14).
    if (fstats_.calls > 0) {
        const VkForwardStats &f = fstats_;
        std::string fb, sp;
        for (uint32_t i = 0; i < uint32_t(VkGemmConsumer::kCount); ++i) {
            const VkFallbackCounter &k = f.fallback[i];
            if (k.rows == 0) continue;
            char one[128];
            std::snprintf(one, sizeof one, "%s%u:%llu/%llu", fb.empty() ? "" : ",", i, (unsigned long long)k.flagged,
                          (unsigned long long)k.rows);
            fb += one;
            if (k.split != 0) {
                std::snprintf(one, sizeof one, "%s%u:%llu", sp.empty() ? "" : ",", i, (unsigned long long)k.split);
                sp += one;
            }
        }
        std::string af;
        for (uint32_t i = 0; i < uint32_t(VkAttnPath::kCount); ++i) {
            const VkAttnFlagCounter &a = f.attn_flags[i];
            char one[160];
            std::snprintf(one, sizeof one, "%s%s:calls=%llu,rows=%llu,flagged_rows=%llu,flagged_pairs=%llu",
                          af.empty() ? "" : ";", i == 0 ? "k6" : "k9", (unsigned long long)a.calls,
                          (unsigned long long)a.rows, (unsigned long long)a.flagged_rows,
                          (unsigned long long)a.flagged_pairs);
            af += one;
        }
        std::fprintf(stderr,
                     "cere: vk_forward_stats calls=%llu prefill=%llu decode=%llu tokens=%llu prefill_gpu_ms=%.1f "
                     "decode_gpu_ms_mean=%.3f decode_record_ms_mean=%.3f exact_fallback_flagged/rows=[%s] split=[%s] "
                     "dropped=%llu prefill_attention=%s prefill_calls_coopmat/fp32=%llu/%llu attn_flags=[%s] "
                     "prefill_max_submit_gpu_ms=%.1f prefill_submits=%llu in_flight=%u prefill_gpu_gap_ms=%.1f/%llu "
                     "prefill_submit_call_ms=%.1f prefill_wait_call_ms=%.1f\n",
                     (unsigned long long)f.calls, (unsigned long long)f.prefill_calls,
                     (unsigned long long)f.decode_calls, (unsigned long long)f.tokens, f.prefill_gpu_ms,
                     f.decode_calls ? f.decode_gpu_ms / double(f.decode_calls) : 0.0,
                     f.decode_calls ? f.decode_record_ms / double(f.decode_calls) : 0.0, fb.c_str(), sp.c_str(),
                     (unsigned long long)f.fallback_dropped, prefill_attention_path(),
                     (unsigned long long)f.prefill_coopmat_calls, (unsigned long long)f.prefill_fp32_calls, af.c_str(),
                     f.prefill_max_submit_gpu_ms, (unsigned long long)f.prefill_submits.submits,
                     ctx_ ? ctx_->max_in_flight() : 0u, f.prefill_submits.gap_ms,
                     (unsigned long long)f.prefill_submits.gaps, f.prefill_submits.submit_call_ms,
                     f.prefill_submits.wait_call_ms);
        if (f.spec.rounds > 0 || f.spec.extends > 0) {
            const auto &p = f.spec;
            const double r = p.rounds ? double(p.rounds) : 1.0;
            std::fprintf(stderr,
                         "cere: vk_spec_stats rounds=%llu drafted=%llu accepted=%llu committed=%llu "
                         "accepted_hist=%llu/%llu/%llu/%llu verify_gpu_ms_mean=%.3f cd_gpu_ms_mean=%.3f "
                         "(restore %.3f draft %.3f) verify_record_ms_mean=%.3f cd_record_ms_mean=%.3f "
                         "host_accept_ms_mean=%.4f round_wall_ms_mean=%.3f extends=%llu extend_rows=%llu "
                         "extend_gpu_ms=%.1f max_submit_gpu_ms=%.2f/%.2f/%.2f restores=%llu key_restores=%llu "
                         "sparse_verify_rows=%llu\n",
                         (unsigned long long)p.rounds, (unsigned long long)p.drafted, (unsigned long long)p.accepted,
                         (unsigned long long)p.committed, (unsigned long long)p.accepted_hist[0],
                         (unsigned long long)p.accepted_hist[1], (unsigned long long)p.accepted_hist[2],
                         (unsigned long long)p.accepted_hist[3], p.verify_gpu_ms / r, p.cd_gpu_ms / r,
                         p.restore_gpu_ms / r, p.draft_gpu_ms / r, p.verify_record_ms / r, p.cd_record_ms / r,
                         p.host_accept_ms / r, (p.verify_wall_ms + p.cd_wall_ms + p.host_accept_ms) / r,
                         (unsigned long long)p.extends, (unsigned long long)p.extend_rows, p.extend_gpu_ms,
                         p.max_verify_gpu_ms, p.max_cd_gpu_ms, p.max_extend_gpu_ms, (unsigned long long)p.restores,
                         (unsigned long long)p.key_restores, (unsigned long long)p.sparse_rows);
        }
    }
    for (vk::Buffer &b : blocks_) ctx_->free(b);  // works on a faulted context (host handles only)
    blocks_.clear();
    held_.clear();  // drops the ledger entries
}

uint64_t VulkanModel::region_addr(VkRegionKind kind, uint32_t index) const {
    return vk_region_addr(layout_, addrs_, kind, index);
}

uint8_t *VulkanModel::region_map(VkRegionKind kind, uint32_t index) const {
    const VkRegion *r = vk_region(layout_, kind, index);
    if (r == nullptr || r->block >= blocks_.size() || blocks_[r->block].map == nullptr) return nullptr;
    return static_cast<uint8_t *>(blocks_[r->block].map) + r->offset;
}

VulkanModel::IoRegion VulkanModel::io(VkRegionKind kind) const {
    const VkRegion *r = vk_region(layout_, kind, 0);
    if (r == nullptr) return IoRegion{};
    return IoRegion{region_addr(kind), region_map(kind), r->bytes};
}

uint8_t *VulkanModel::host_of(uint64_t addr) const {
    for (const vk::Buffer &b : blocks_) {
        if (addr >= b.addr && addr < b.addr + b.size && b.map != nullptr) {
            return static_cast<uint8_t *>(b.map) + (addr - b.addr);
        }
    }
    return nullptr;
}

cere_status VulkanModel::read(uint64_t tensor, uint64_t offset, void *dst, uint64_t len) const {
    if (dst == nullptr || tensor >= plan_.tensors.size() || len == 0) return CERE_ERR_ARGUMENT;
    try {
        const std::vector<VkPiece> pieces = vk_locate(layout_, tensor, offset, len);
        if (pieces.empty()) return CERE_ERR_ARGUMENT;
        for (const VkPiece &p : pieces) {
            std::memcpy(static_cast<uint8_t *>(dst) + (p.tensor_offset - offset),
                        static_cast<const uint8_t *>(blocks_[p.block].map) + p.offset, p.bytes);
        }
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    return CERE_OK;
}

uint64_t VulkanModel::held_device_bytes() const {
    uint64_t s = 0;
    for (size_t b = 0; b < blocks_.size(); ++b) s += layout_.blocks[b].heap == VkHeap::kDeviceLocal ? held_[b]->bytes() : 0;
    return s;
}

uint64_t VulkanModel::held_host_bytes() const {
    uint64_t s = 0;
    for (size_t b = 0; b < blocks_.size(); ++b) s += layout_.blocks[b].heap == VkHeap::kHostHeap ? held_[b]->bytes() : 0;
    return s;
}

// ---- VulkanSession -------------------------------------------------------------------------------------------------

VulkanSession::~VulkanSession() {
    std::lock_guard<std::mutex> lock(model_.slots_mutex_);
    if (!pages_.empty()) host::release_pages_logged(*model_.pool_, pages_.data(), pages_.size(), "vk_session_release");
    if (slot_ < model_.slot_used_.size()) model_.slot_used_[slot_] = false;
}

// ---- VulkanDevice --------------------------------------------------------------------------------------------------

cere_status VulkanDevice::open(const char *device_id, std::unique_ptr<host::Device> *out) {
    if (device_id == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    uint32_t ordinal = 0;
    if (const cere_status st = parse_device_id(device_id, &ordinal); st != CERE_OK) return st;
    size_t shaders = 0;
    vk::embedded_shaders(&shaders);
    if (shaders == 0) {
        say("vk_open refused: the library was built without its embedded SPIR-V table");
        return CERE_ERR_UNSUPPORTED;
    }
    std::unique_ptr<vk::Context> ctx;
    std::string why;
    vk::CreateOptions opts;
    opts.max_in_flight = kPrefillSubmitsInFlight;  // P4-G: pipelined prefill segments (vulkan_forward.h)
    if (const cere_status st = vk::Context::create(ordinal, &ctx, &why, opts); st != CERE_OK) {
        // No such GPU / missing feature: UNSUPPORTED (the host's create refuses cleanly); the reason goes to stderr.
        say("vk_open refused ordinal=" + std::to_string(ordinal) + " status=" + std::to_string(st) + ": " + why);
        return st;
    }
    std::unique_ptr<VulkanDevice> d(new (std::nothrow) VulkanDevice());
    if (d == nullptr) return CERE_ERR_CAPACITY;
    d->ctx_ = std::shared_ptr<vk::Context>(ctx.release());
    d->ordinal_ = ordinal;
    const uint64_t available = mem_available_bytes();
    d->host_budget_ = available > kHostReserveBytes ? available - kHostReserveBytes : 0;
    *out = std::move(d);
    return CERE_OK;
}

VulkanDevice::~VulkanDevice() = default;

bool VulkanDevice::faulted() const {
    if (!ctx_->faulted()) return false;
    if (!poisoned_reported_.exchange(true)) {
        std::fprintf(stderr, "cere: device_poisoned vk=%u error=%s reason=%s\n", ordinal_,
                     vk::result_name(ctx_->fault_result()), ctx_->fault_reason().c_str());
    }
    return true;
}

std::vector<const host::Support *> VulkanDevice::supported() const { return {&host::a4_support()}; }

host::MemoryEstimate VulkanDevice::estimate(const host::ModelPlan &plan) const {
    try {
        const VkLayout l = vk_plan_layout(plan);
        // A plan outside the layout's scope estimates 0, so the host reaches allocate, which returns the layout's own
        // status (an out-of-scope plan is UNSUPPORTED, not CAPACITY); nothing is reserved because allocate refuses.
        if (l.status != CERE_OK) return {0, 0};
        return vk_estimate(l);
    } catch (const std::bad_alloc &) {
        return {UINT64_MAX, UINT64_MAX};
    }
}

cere_status VulkanDevice::check_levers(const host::Levers &l, std::string *why) {
    const auto no = [why](const char *m) {
        if (why != nullptr) *why = m;
        return CERE_ERR_UNSUPPORTED;
    };
    // P4-B / D7: tail_split = the GEMM's SPLIT x rows (Context::k2_tail_split, set per forward from the model's levers).
    if (l.k2_gemm != CERE_LEVER_K2_CUBLASLT && l.k2_gemm != CERE_LEVER_K2_TAIL_SPLIT)
        return no("k2_gemm: cublaslt or tail_split on Vulkan");
    // P4-E: p4 = the cooperative-matrix prefill attention; the device half (coopmat offered, probe passed) is
    // check_prefill_attention.
    if (l.prefill_attention != CERE_LEVER_PREFILL_CUDA_CORE && l.prefill_attention != CERE_LEVER_PREFILL_P4)
        return no("prefill_attention: cuda_core or p4 on Vulkan");
    if (l.decode_step != CERE_LEVER_DECODE_EAGER) return no("decode_step: eager only on Vulkan (no graphs)");
    if (l.moe_gemm != CERE_LEVER_MOE_BUILD && l.moe_gemm != CERE_LEVER_MOE_K7) return no("moe_gemm: k7 build");
    if (l.verify_attention != CERE_LEVER_VERIFY_PER_ROW) return no("verify_attention: per_row only on Vulkan");
    if (l.k2_weights != CERE_LEVER_K2_WEIGHTS_BF16) return no("k2_weights: bf16 only on Vulkan");
    return CERE_OK;
}

cere_status VulkanDevice::check_prefill_attention(const VulkanModel *m, const host::Levers &l, std::string *why) const {
    if (l.prefill_attention != CERE_LEVER_PREFILL_P4) return CERE_OK;
    const auto no = [why](const std::string &msg) {
        if (why != nullptr) *why = msg;
        return CERE_ERR_UNSUPPORTED;
    };
    if (!ctx_->info().coopmat) return no("prefill_attention p4: the device offers no VK_KHR_cooperative_matrix");
    if (m != nullptr && m->report_.coopmat_probed && !m->report_.coopmat_ok)
        return no("prefill_attention p4: " + m->report_.coopmat_why);
    return CERE_OK;
}

cere_status VulkanDevice::allocate_storage(const host::ModelPlan &plan, std::unique_ptr<VulkanModel> *out,
                                           std::string *why) {
    const auto t0 = Clock::now();
    if (ctx_->faulted()) return CERE_ERR_INTERNAL;
    std::unique_ptr<VulkanModel> m(new (std::nothrow) VulkanModel(ctx_));
    if (m == nullptr) return CERE_ERR_CAPACITY;
    m->plan_ = plan;
    m->layout_ = vk_plan_layout(plan);
    if (m->layout_.status != CERE_OK) {
        *why = "layout: " + m->layout_.error;
        return m->layout_.status;
    }
    m->digest_ = vk_layout_digest(plan, m->layout_);
    const VkLayout &l = m->layout_;
    m->blocks_.reserve(l.blocks.size());
    m->addrs_.reserve(l.blocks.size());
    m->held_.reserve(l.blocks.size());
    uint64_t req_max_slack = 0, req_over = 0;
    bool all_coherent = true;
    // Allocation order (an internal benchmark report): the device-local blocks that every
    // forward touches first, the workspace, then state, the KV pool and IO, then the rest (weights, the host-heap PLE
    // blocks) in index order. On an APU whose device-local heap is a small VRAM carve-out plus GTT (AMD: a 2 GiB
    // carve-out under RADV's unified heap), the kernel driver puts a block in the carve-out while it still fits, so
    // the first small allocations get it. In index order a model with drafts put a 0.82 GB weights block there and
    // its 1.18 GB workspace in GTT, and its prefill GPU time was 11 % above the same model without drafts (whose
    // workspace, KV pool, state and IO blocks all sat in the carve-out). The order changes no layout (block indices,
    // offsets, the digest), only which buffer the driver creates first; no output bit depends on it.
    std::vector<uint32_t> order;
    order.reserve(l.blocks.size());
    const auto hot = [](const VkBlock &x) {
        return x.heap == VkHeap::kDeviceLocal && x.purpose != VkPurpose::kWeights && x.purpose != VkPurpose::kPle;
    };
    for (const VkPurpose p : {VkPurpose::kWorkspace, VkPurpose::kState, VkPurpose::kKvPool, VkPurpose::kIo}) {
        for (size_t b = 0; b < l.blocks.size(); ++b) {
            if (hot(l.blocks[b]) && l.blocks[b].purpose == p) order.push_back(uint32_t(b));
        }
    }
    for (size_t b = 0; b < l.blocks.size(); ++b) {
        if (!hot(l.blocks[b])) order.push_back(uint32_t(b));
    }
    if (order.size() != l.blocks.size()) {
        *why = "block allocation order covers " + std::to_string(order.size()) + " of " +
               std::to_string(l.blocks.size()) + " blocks";
        return CERE_ERR_INTERNAL;
    }
    m->blocks_.resize(l.blocks.size());  // index-aligned with l.blocks; a block not allocated yet is empty
    m->addrs_.assign(l.blocks.size(), 0);
    m->held_.resize(l.blocks.size());
    for (const uint32_t b : order) {
        const VkBlock &blk = l.blocks[b];
        const bool host_heap = blk.heap == VkHeap::kHostHeap;
        // RADV assumption 1 (T2a open question): the buffer's memory requirement is exactly what the plan charges.
        // Context::allocate creates the buffer with these usage bits and allocates its requirement.
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = blk.bytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkDeviceBufferMemoryRequirements q{VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS};
        q.pCreateInfo = &bi;
        VkMemoryRequirements2 req{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        vkGetDeviceBufferMemoryRequirements(ctx_->device(), &q, &req);
        if (req.memoryRequirements.size > blk.bytes) {
            ++req_over;
            *why = "block " + std::to_string(b) + ": RADV requires " + std::to_string(req.memoryRequirements.size) +
                   " B for a " + std::to_string(blk.bytes) + " B buffer (the plan charges the buffer size)";
            return CERE_ERR_UNSUPPORTED;
        }
        req_max_slack = std::max<uint64_t>(req_max_slack, blk.bytes - req.memoryRequirements.size);
        vk::Buffer buf;
        std::string e;
        if (const cere_status st = ctx_->allocate(blk.bytes, buf, host_heap, &e); st != CERE_OK) {
            *why = "block " + std::to_string(b) + " (" + std::to_string(blk.bytes) + " B, " +
                   (host_heap ? "host heap" : "device-local") + "): " + e;
            return st;  // m's destructor frees the blocks allocated so far
        }
        m->blocks_[b] = buf;
        m->addrs_[b] = buf.addr;
        // RADV assumption 2: the chosen memory type is HOST_COHERENT (write / IO reads need no flush / invalidate).
        const VkMemoryPropertyFlags f = ctx_->memory_flags(buf.memory_type);
        if ((f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0 || buf.map == nullptr) {
            all_coherent = false;
            *why = "block " + std::to_string(b) + ": memory type " + std::to_string(buf.memory_type) + " (flags 0x" +
                   std::to_string(f) + ") is not host-coherent / mapped";
            return CERE_ERR_UNSUPPORTED;
        }
        auto entry = std::make_unique<host::LedgerEntry>();
        entry->hold(ledger_, host_heap ? host::Held::kMapped : host::Held::kDevice, blk.bytes);
        m->held_[b] = std::move(entry);
    }
    // The K4 block table (static from ALLOCATE on, design 4.2) and a clear bound-fault word.
    if (l.ple.present) {
        uint64_t table[128];
        if (!vk_ple_block_table(l, m->addrs_.data(), table)) {
            *why = "PLE block table";
            return CERE_ERR_INTERNAL;
        }
        uint8_t *dst = m->region_map(VkRegionKind::kIoPleBlockTable);
        if (dst == nullptr) {
            *why = "no IO region for the K4 block table";
            return CERE_ERR_INTERNAL;
        }
        std::memcpy(dst, table, sizeof table);
    }
    if (uint8_t *fault = m->region_map(VkRegionKind::kIoBoundFault); fault != nullptr) std::memset(fault, 0, 4);
    m->slot_used_.assign(plan.max_sessions, false);
    m->pool_ = std::make_unique<host::PagePool>(l.pool_tokens / host::kKvPageTokens);
    m->levers_ = plan.levers;
    uint32_t dev_blocks = 0, host_blocks = 0;
    for (const VkBlock &blk : l.blocks) (blk.heap == VkHeap::kHostHeap ? host_blocks : dev_blocks)++;
    char line[512];
    std::snprintf(line, sizeof line,
                  "vk_allocate vk=%u blocks=%zu device_blocks=%u host_blocks=%u device_bytes=%llu host_bytes=%llu "
                  "req_over=%llu req_max_slack=%llu coherent=%d layout=%.16s ms=%.1f",
                  ordinal_, l.blocks.size(), dev_blocks, host_blocks, (unsigned long long)l.device_bytes,
                  (unsigned long long)l.host_bytes, (unsigned long long)req_over, (unsigned long long)req_max_slack,
                  all_coherent ? 1 : 0, m->digest_.c_str(), ms_since(t0));
    say(line);
    *out = std::move(m);
    return CERE_OK;
}

cere_status VulkanDevice::allocate(const host::ModelPlan &plan, std::unique_ptr<host::DeviceModel> *out) {
    try {
        std::string why;
        cere_status st = check_levers(plan.levers, &why);
        if (st == CERE_OK) st = check_prefill_attention(nullptr, plan.levers, &why);
        if (st == CERE_OK && plan.speculative_draft_tokens != 0 && !kDraftForwardImplemented) {
            // device.h: allocate refuses a plan with drafts on a device that cannot run them.
            why = "speculative_draft_tokens " + std::to_string(plan.speculative_draft_tokens) +
                  ": the Vulkan MTP draft is disabled in this build (kDraftForwardImplemented)";
            st = CERE_ERR_UNSUPPORTED;
        }
        if (st == CERE_OK && plan.speculative_draft_tokens != 0 &&
            (plan.speculative_draft_tokens != kVkMaxChainDrafts + 1 || plan.draft_token_map.empty())) {
            // T6 scope: the ABI's one draft value (4: up to 3 drafts + 1 per round) with its token map.
            why = "speculative_draft_tokens " + std::to_string(plan.speculative_draft_tokens) +
                  ": the Vulkan draft runs 4 (3 drafts) with a draft token map only";
            st = CERE_ERR_UNSUPPORTED;
        }
        std::unique_ptr<VulkanModel> m;
        if (st == CERE_OK) st = allocate_storage(plan, &m, &why);
        if (st == CERE_OK) {
            m->weights_ = std::make_unique<VkModelWeights>();
            st = vk_bind_weights(m->plan_, m->layout_, m->addrs_, m->weights_.get(), &why);
        }
        if (st == CERE_OK) st = vk_bind_workspace(m->plan_, m->layout_, m->addrs_, &m->ws_);
        if (st == CERE_OK && plan.speculative_draft_tokens != 0) {
            // T6: everything the speculation path addresses must be carved (memory plan: drafts layout).
            const VkLayout &l = m->layout_;
            if (!m->weights_->has_draft || m->ws_.mtp_hc_rows == 0 || l.verify_rows < kVkMaxChainDrafts + 1 ||
                l.undo_bytes == 0 || m->region_addr(VkRegionKind::kSpecSampler) == 0 ||
                m->region_addr(VkRegionKind::kUndo, uint32_t(model::layout::kUndoSnapshots * plan.max_sessions - 1)) ==
                    0) {
                why = "drafts: the layout lacks an MTP / undo / spec-sampler region";
                st = CERE_ERR_INTERNAL;
            }
        }
        if (st != CERE_OK) {
            say("vk_allocate refused status=" + std::to_string(st) + ": " + why);
            return st;  // m (if any) frees every block it holds
        }
        *out = std::move(m);
        return CERE_OK;
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
}

cere_status VulkanDevice::write(host::DeviceModel &model, uint64_t tensor_index, uint64_t offset, const void *bytes,
                                uint64_t len) {
    // UPLOAD (design 3.5): a memcpy into the persistent, host-coherent mapping; no GPU work, so it does not touch the
    // Context and concurrent writes of disjoint ranges need no lock. The host validated the range; vk_locate
    // re-checks it (ARGUMENT outside the tensor or for a DISCARD tensor).
    auto &m = static_cast<VulkanModel &>(model);
    if (bytes == nullptr || len == 0) return CERE_ERR_ARGUMENT;
    try {
        const std::vector<VkPiece> pieces = vk_locate(m.layout_, tensor_index, offset, len);
        if (pieces.empty()) return CERE_ERR_ARGUMENT;
        for (const VkPiece &p : pieces) {
            std::memcpy(static_cast<uint8_t *>(m.blocks_[p.block].map) + p.offset,
                        static_cast<const uint8_t *>(bytes) + (p.tensor_offset - offset), p.bytes);
        }
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    return CERE_OK;
}

cere_status VulkanDevice::set_levers(host::DeviceModel &model, const host::Levers &levers) {
    auto &m = static_cast<VulkanModel &>(model);
    std::string why;
    if (const cere_status st = check_levers(levers, &why); st != CERE_OK) return st;  // previous selection kept
    if (const cere_status st = check_prefill_attention(&m, levers, &why); st != CERE_OK) {
        say("vk_set_levers refused: " + why);
        return st;
    }
    m.levers_ = levers;
    return CERE_OK;
}

cere_status VulkanDevice::instantiate(host::DeviceModel &model) {
    auto &m = static_cast<VulkanModel &>(model);
    if (ctx_->faulted()) return CERE_ERR_INTERNAL;
    if (m.instantiated_) return CERE_OK;
    m.instantiate_attempted_ = true;
    std::string why;
    cere_status st = CERE_OK;
    try {
        st = instantiate_impl(m, &why);
    } catch (const std::bad_alloc &) {
        st = CERE_ERR_CAPACITY;
        why = "out of host memory";
    }
    if (st != CERE_OK) {
        say("vk_instantiate failed status=" + std::to_string(st) + ": " + why);
        faulted();  // reports a device loss once
        return st;
    }
    m.instantiated_ = true;
    // T5: the forward runs this model only when every MXFP8 weight reaches its kernel with its scales; otherwise the
    // model stays sealed but the device reports no forward (execution_mode UNSUPPORTED), never a misread weight.
    std::string mx_why;
    bool ok = vk_forward_mxfp8_ok(m, &mx_why);
    if (ok) {
        forward_ready_ = true;
    } else {
        say("vk_instantiate: forward not available: " + mx_why);
    }
    say(std::string("vk_forward mxfp8_wiring=") + vk_forward_mxfp8_wiring() + " forward=" + (ok ? "1" : "0"));
    return CERE_OK;
}

cere_status VulkanDevice::instantiate_impl(VulkanModel &m, std::string *why) {
    const auto t0 = Clock::now();
    InstantiateReport rep;
    vk::Context &ctx = *ctx_;
    VkModelWeights &w = *m.weights_;
    const cere_qwen4_exp_config &c = m.plan_.config;
    const bool drafts = w.has_draft;
    // (1) The one-time probes (each its own submit, outside recording; T1 report section 11).
    {
        const auto t = Clock::now();
        if (cere_status st = vkk2::gemm_probe(ctx); st != CERE_OK) {
            *why = "vkk2::gemm_probe";
            return st;
        }
        if (cere_status st = vkk7::probe(ctx); st != CERE_OK) {
            *why = "vkk7::probe";
            return st;
        }
        // P4-E: the coopmat prefill attention's device facts (its own submit). A failed probe (or no coopmat) keeps
        // the model on the FP32 prefill attention: prefill_attention p4 is then refused (here, and by set_levers).
        rep.coopmat_probed = true;
        if (!ctx.info().coopmat) {
            rep.coopmat_why = "no VK_KHR_cooperative_matrix";
        } else {
            const cere_status st = vkk6::coopmat_probe(ctx, rep.coopmat_probe);
            if (st != CERE_OK) {
                if (ctx.faulted()) {
                    *why = "vkk6::coopmat_probe";
                    return CERE_ERR_INTERNAL;
                }
                rep.coopmat_why = "coopmat probe status " + std::to_string(st);
            } else if (rep.coopmat_probe[0] != kCoopmatProbeF16 || rep.coopmat_probe[1] != kCoopmatProbeProduct) {
                char b[160];
                std::snprintf(b, sizeof b, "coopmat probe: FP16 subnormals not kept (f16 0x%04x want 0x%04x, product 0x%08x "
                              "want 0x%08x)", rep.coopmat_probe[0], kCoopmatProbeF16, rep.coopmat_probe[1],
                              kCoopmatProbeProduct);
                rep.coopmat_why = b;
            } else {
                rep.coopmat_ok = true;
            }
        }
        // the report is the model's from here on (check_prefill_attention, prefill_coopmat read it)
        m.report_.coopmat_probed = true;
        m.report_.coopmat_ok = rep.coopmat_ok;
        std::memcpy(m.report_.coopmat_probe, rep.coopmat_probe, sizeof rep.coopmat_probe);
        m.report_.coopmat_why = rep.coopmat_why;
        if (const cere_status st = check_prefill_attention(&m, m.levers_, why); st != CERE_OK) return st;
        rep.probe_ms = ms_since(t);
    }
    // (2) Every pipeline the forward can use, so none compiles after readiness.
    {
        const auto t = Clock::now();
        // The MXFP8 K5 / K7 pipelines when any GDN or shared-expert weight of the bound model is MXFP8 (T5).
        bool mx = false;
        for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
            for (const VkDense &d : w.layers[L].gdn_dense) mx = mx || d.mxfp8();
            for (const VkDense &d : w.layers[L].shared_dense) mx = mx || d.mxfp8();
        }
        // P4-E: with the probe passed, the coopmat prefill pipelines too (any later set_selection may pick p4).
        const std::vector<vk::PipelineSpec> specs = vk_warm_specs(c, drafts, &rep.warm_lists, mx, rep.coopmat_ok);
        rep.warm_specs = specs.size();
        std::string e;
        if (cere_status st = ctx.warm(specs, &e); st != CERE_OK) {
            *why = "pipeline warm-up: " + e;
            return st;
        }
        rep.pipelines = ctx.pipeline_count();
        rep.warm_ms = ms_since(t);
    }
    // (3) Host arithmetic through the mappings (forward.cpp:1816-1893): A_log F32, MoE alphas, a1/a2 global scales.
    {
        const auto t = Clock::now();
        const uint32_t E = c.num_experts, vh = c.linear_num_value_heads;
        const auto f32s = [&m](uint64_t addr, uint32_t n, std::vector<float> *out) {
            const uint8_t *p = m.host_of(addr);
            if (p == nullptr) return false;
            out->resize(n);
            std::memcpy(out->data(), p, size_t(n) * 4);
            return true;
        };
        std::vector<float> gin, uin, din, gws2, dws2, g1(E), g2(E), alog(vh);
        for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
            VkLayerWeights &lw = w.layers[L];
            if (!lw.full_attention) {
                const uint8_t *src = m.host_of(lw.gdn_A_log_bf16);
                uint8_t *dst = m.host_of(lw.gdn.A_log);
                if (src == nullptr || dst == nullptr) {
                    *why = "A_log of layer " + std::to_string(L) + " not mapped";
                    return CERE_ERR_INTERNAL;
                }
                for (uint32_t h = 0; h < vh; ++h) {
                    uint16_t b;
                    std::memcpy(&b, src + 2 * h, 2);
                    alog[h] = bf16_to_f32(b);
                }
                std::memcpy(dst, alog.data(), size_t(vh) * 4);
            }
            const VkMoeScaleSources &s = lw.moe_scales;
            if (!f32s(s.gate_input_scale, E, &gin) || !f32s(s.up_input_scale, E, &uin) ||
                !f32s(s.down_input_scale, E, &din) || !f32s(s.gate_weight_scale_2, E, &gws2) ||
                !f32s(s.down_weight_scale_2, E, &dws2)) {
                *why = "MoE scale sources of layer " + std::to_string(L) + " not mapped";
                return CERE_ERR_INTERNAL;
            }
            // MF-5: zero / negative / non-finite activation scales would poison every MoE block.
            for (uint32_t e = 0; e < E; ++e) {
                for (const float v : {gin[e], uin[e], din[e]}) {
                    if (!(std::isfinite(v) && v > 0.0f)) {
                        *why = "layer " + std::to_string(L) + " expert " + std::to_string(e) +
                               ": an input_scale is not finite and positive";
                        return CERE_ERR_ARGUMENT;
                    }
                }
            }
            float w13_max = gin[0], w2_max = din[0];
            for (uint32_t e = 0; e < E; ++e) {
                w13_max = std::max(w13_max, std::max(gin[e], uin[e]));
                w2_max = std::max(w2_max, din[e]);
            }
            for (uint32_t e = 0; e < E; ++e) {
                g1[e] = w13_max * gws2[e];
                g2[e] = w2_max * dws2[e];
            }
            uint8_t *g1d = m.host_of(lw.moe.g1_alphas), *g2d = m.host_of(lw.moe.g2_alphas);
            if (g1d == nullptr || g2d == nullptr) {
                *why = "MoE alphas of layer " + std::to_string(L) + " not mapped";
                return CERE_ERR_INTERNAL;
            }
            std::memcpy(g1d, g1.data(), size_t(E) * 4);
            std::memcpy(g2d, g2.data(), size_t(E) * 4);
            lw.moe.a1_global_scale = 1.0f / w13_max;
            lw.moe.a2_global_scale = 1.0f / w2_max;
        }
        rep.host_ms = ms_since(t);
    }
    // (4) One command buffer: the RoPE table and every prepared K2 row info, then one submit.
    {
        const auto t = Clock::now();
        if (cere_status st = ctx.begin(); st != CERE_OK) {
            *why = "begin";
            return st;
        }
        cere_status st = vkk1::rope_table(ctx, w.rope_cos_sin, int64_t(m.plan_.max_context_tokens),
                                          int(model::layout::rotary_dim(c)), float(c.rope_theta));
        // Every VkLayout::winfo entry with its weight's own format (an MXFP8 weight is scanned with its scales; the
        // K5 / K4 info layouts are these entries abutting, vk_memory_plan.cpp:417-475).
        const uint64_t winfo = m.region_addr(VkRegionKind::kWinfo);
        std::unordered_map<std::string_view, uint64_t> by_name;
        for (const VkWinfo &e : m.layout_.winfo) {
            if (st != CERE_OK) break;
            const uint64_t wt = vk_tensor_addr(m.layout_, m.addrs_, e.tensor);
            const std::string &name = m.plan_.tensors[e.tensor].name;
            uint64_t scales = 0;
            if (m.plan_.tensors[e.tensor].dtype == CERE_DTYPE_F8_E4M3) {
                // the scale tensor of `<prefix>.weight` is `<prefix>.weight_scale`; for q|k|v the q scales start the
                // packed scale run (bind_qkv checked it).
                if (by_name.empty()) {
                    by_name.reserve(m.plan_.tensors.size());
                    for (uint64_t i = 0; i < m.plan_.tensors.size(); ++i) by_name.emplace(m.plan_.tensors[i].name, i);
                }
                const auto it = by_name.find(name + "_scale");
                if (it != by_name.end()) scales = vk_tensor_addr(m.layout_, m.addrs_, it->second);
                if (scales == 0) {
                    *why = "no scales for MXFP8 weight " + name;
                    st = CERE_ERR_INTERNAL;
                    break;
                }
            }
            st = vkk2::gemm_prepare_weight(ctx, wt, scales, e.n, e.k, winfo + e.offset);
            if (st != CERE_OK) *why = "gemm_prepare_weight " + name;
            ++rep.winfo_entries;
        }
        // K7 row info per layer in vkk7::moe_prepare_weights's layout (router E, shared gate S, up S, down H words,
        // 16-byte aligned parts), each part scanned with its own format: vkk7::moe_prepare_weights itself passes no
        // scales, and the S3 shared expert is MXFP8.
        const auto r16 = [](uint64_t b) { return (b + 15) / 16 * 16; };
        for (uint32_t L = 0; L < c.num_hidden_layers && st == CERE_OK; ++L) {
            const VkLayerWeights &lw = w.layers[L];
            const uint64_t base = lw.moe.bf16_info;
            const uint64_t gate = base + r16(uint64_t(c.num_experts) * 4);
            const uint64_t up = gate + r16(uint64_t(c.shared_expert_intermediate_size) * 4);
            const uint64_t down = up + r16(uint64_t(c.shared_expert_intermediate_size) * 4);
            st = vkk2::gemm_prepare_weight(ctx, lw.moe.router, 0, c.num_experts, c.hidden_size, base);
            const uint64_t parts[3] = {gate, up, down};
            for (int i = 0; i < 3 && st == CERE_OK; ++i) {
                const VkDense &d = lw.shared_dense[i];
                st = vkk2::gemm_prepare_weight(ctx, d.w, d.scales, d.n, d.k, parts[i]);
            }
            if (st != CERE_OK) *why = "K7 row info of layer " + std::to_string(L);
            ++rep.moe_info_layers;
        }
        // T6, drafts (mtp.cu instantiate :492-509, upload_map :485-490): the draft token map into the derived region
        // through the mapping (host-coherent, written before this command buffer is submitted), the hot lm_head = the
        // target lm_head rows at the map (a row gather, bit-exact; 32,768-row chunks as CUDA, K1 takes <= 65,535), and
        // the K13 row info of the MTP router / shared expert (kMoeInfo index num_hidden_layers).
        if (st == CERE_OK && drafts) {
            VkMtpWeights &mw = w.mtp;
            const std::vector<uint32_t> &map = m.plan_.draft_token_map;
            uint8_t *map_dst = m.host_of(mw.map);
            if (mw.count == 0 || map.size() != mw.count || map_dst == nullptr || mw.lm_head == 0 ||
                mw.moe.bf16_info == 0) {
                *why = "MTP derived region / draft token map not bound";
                st = CERE_ERR_INTERNAL;
            } else {
                std::memcpy(map_dst, map.data(), map.size() * sizeof(uint32_t));
                constexpr uint64_t kGatherRows = 32768;
                for (uint64_t r0 = 0; r0 < mw.count && st == CERE_OK; r0 += kGatherRows) {
                    const uint64_t n = std::min<uint64_t>(kGatherRows, mw.count - r0);
                    st = vkk1::embedding_gather(ctx, w.lm_head, int64_t(c.vocab_size), int(c.hidden_size),
                                                mw.map + r0 * 4, int64_t(n), mw.lm_head + r0 * c.hidden_size * 2);
                }
                if (st == CERE_OK) {
                    const vkk13::MoeDims dims{c.hidden_size, c.num_experts, c.num_experts_per_tok,
                                              c.moe_intermediate_size, c.shared_expert_intermediate_size};
                    st = vkk13::moe_bf16_prepare_weights(ctx, dims, mw.moe);
                }
                if (st != CERE_OK) *why = "MTP instantiate (hot lm_head gather / K13 row info)";
                rep.mtp = st == CERE_OK;
            }
        }
        if (st == CERE_OK) st = ctx.barrier();
        const cere_status sub = ctx.submit_wait();  // closes the command buffer whatever st was
        if (st == CERE_OK) st = sub;
        if (st != CERE_OK) {
            if (why->empty()) *why = "INSTANTIATE submit: " + std::string(ctx.faulted() ? ctx.fault_reason() : "");
            return st;
        }
        rep.gpu_ms = ms_since(t);
    }
    rep.total_ms = ms_since(t0);
    m.report_ = rep;
    char line[1024];
    std::snprintf(line, sizeof line,
                  "vk_instantiate vk=%u probes_ms=%.1f warm_ms=%.1f specs=%zu pipelines=%zu host_ms=%.1f gpu_ms=%.1f "
                  "winfo=%u moe_info=%u mtp=%d total_ms=%.1f layout=%.16s lists=%s prefill_attention=%s "
                  "coopmat_probe=%d(%08x,%08x,%08x,%u)%s%s",
                  ordinal_, rep.probe_ms, rep.warm_ms, rep.warm_specs, rep.pipelines, rep.host_ms, rep.gpu_ms,
                  rep.winfo_entries, rep.moe_info_layers, rep.mtp ? 1 : 0, rep.total_ms, m.digest_.c_str(),
                  rep.warm_lists.c_str(), m.prefill_attention_path(), rep.coopmat_ok ? 1 : 0, rep.coopmat_probe[0],
                  rep.coopmat_probe[1], rep.coopmat_probe[2], rep.coopmat_probe[3], rep.coopmat_why.empty() ? "" : " ",
                  rep.coopmat_why.c_str());
    say(line);
    return CERE_OK;
}

cere_status VulkanDevice::create_session(host::DeviceModel &model, uint64_t max_context_tokens, bool lazy,
                                         std::unique_ptr<host::DeviceSession> *out) {
    auto &m = static_cast<VulkanModel &>(model);
    if (lazy && m.layout_.state_slots == 0) return CERE_ERR_ARGUMENT;  // lazy sessions exist only on a prefix model
    if (out == nullptr || max_context_tokens == 0 || max_context_tokens > m.plan_.max_context_tokens) {
        return CERE_ERR_ARGUMENT;
    }
    if (ctx_->faulted()) return CERE_ERR_INTERNAL;
    const uint64_t n_pages = lazy ? 0 : host::kv_pages(max_context_tokens);
    uint64_t slot = 0;
    std::vector<uint32_t> pages;
    try {
        pages.reserve(n_pages);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    {
        std::lock_guard<std::mutex> lock(m.slots_mutex_);
        const auto free = std::find(m.slot_used_.begin(), m.slot_used_.end(), false);
        if (free == m.slot_used_.end()) return CERE_ERR_CAPACITY;
        if (const cere_status st = m.pool_->take_pages(n_pages, &pages); st != CERE_OK) return st;
        slot = uint64_t(free - m.slot_used_.begin());
        *free = true;
    }
    std::unique_ptr<VulkanSession> s(new (std::nothrow) VulkanSession(m, slot, max_context_tokens, std::move(pages)));
    if (s == nullptr) {
        std::lock_guard<std::mutex> lock(m.slots_mutex_);
        m.slot_used_[slot] = false;
        host::release_pages_logged(*m.pool_, pages.data(), pages.size(), "vk_session_create");
        return CERE_ERR_CAPACITY;
    }
    s->lazy_ = lazy;
    // From here the session's destructor returns the slot and the pages on every error.
    const uint64_t slot_addr = m.region_addr(VkRegionKind::kSession, uint32_t(slot));
    if (const cere_status st = vk_bind_session(m.plan_, m.layout_, m.addrs_, slot_addr, max_context_tokens, &s->state_);
        st != CERE_OK) {
        return st;
    }
    // T6: the slot's undo snapshots (undo[j] at index 3 slot + j, layout.h carve_undo), bound once.
    if (m.plan_.speculative_draft_tokens != 0) {
        for (uint32_t j = 0; j < kVkMaxChainDrafts; ++j) {
            const uint32_t index = uint32_t(slot * model::layout::kUndoSnapshots + j);
            const VkRegion *r = vk_region(m.layout_, VkRegionKind::kUndo, index);
            if (r == nullptr) return CERE_ERR_INTERNAL;
            if (const cere_status st = vk_bind_undo(m.plan_, m.region_addr(VkRegionKind::kUndo, index), r->bytes,
                                                    &s->undo_[j]);
                st != CERE_OK) {
                return CERE_ERR_INTERNAL;
            }
        }
        s->undo_count_ = kVkMaxChainDrafts;
        if (uint8_t *p = m.host_of(s->state_.proposals); p != nullptr) std::memset(p, 0, 16);
    }
    // session_reset (model.cpp:1026-1041) through the mapping: zeroed slot, PLE history = EOS; then session_set_pages
    // unless lazy (a lazy session's slot map stays zeroed until bind_pages writes it).
    uint8_t *slot_map = m.host_of(slot_addr);
    uint8_t *history = m.host_of(s->state_.ple.history);
    if (slot_map == nullptr || history == nullptr) return CERE_ERR_INTERNAL;
    std::memset(slot_map, 0, s->state_.slot_bytes);
    const uint64_t hist = m.plan_.config.ngram_size == 0 ? 0 : m.plan_.config.ngram_size - 1;
    if (hist > 2) return CERE_ERR_UNSUPPORTED;
    for (uint64_t i = 0; i < hist; ++i) {
        const int64_t eos = vkk4::kEosTokenId;
        std::memcpy(history + 8 * i, &eos, 8);
    }
    if (!lazy) {
        if (const cere_status st = write_slot_map(m, *s); st != CERE_OK) return st;
    }
    *out = std::move(s);
    return CERE_OK;
}

cere_status VulkanDevice::write_slot_map(const VulkanModel &m, const VulkanSession &s) {
    uint8_t *map = m.host_of(s.state_.slot_map);
    if (map == nullptr) return CERE_ERR_INTERNAL;
    const uint64_t pool_pages = m.layout_.pool_tokens / host::kKvPageTokens;
    for (uint64_t p = 0; p < s.max_context_tokens_; ++p) {
        const uint32_t page = s.pages_[p / host::kKvPageTokens];
        if (page >= pool_pages) return CERE_ERR_INTERNAL;
        const int64_t phys = int64_t(uint64_t(page) * host::kKvPageTokens + p % host::kKvPageTokens);
        std::memcpy(map + 8 * p, &phys, 8);
    }
    return CERE_OK;
}

// ---- D15 prefix cache (cuda_device.cpp:1815-1990) -------------------------------------------------------------------

cere_status VulkanDevice::retain_pages(host::DeviceModel &model, const uint32_t *ids, uint64_t n) {
    if (ids == nullptr && n != 0) return CERE_ERR_ARGUMENT;
    auto &m = static_cast<VulkanModel &>(model);
    std::lock_guard<std::mutex> lock(m.slots_mutex_);
    return m.pool_->retain(ids, n);
}

cere_status VulkanDevice::release_pages(host::DeviceModel &model, const uint32_t *ids, uint64_t n) {
    if (ids == nullptr && n != 0) return CERE_ERR_ARGUMENT;
    auto &m = static_cast<VulkanModel &>(model);
    std::lock_guard<std::mutex> lock(m.slots_mutex_);
    return m.pool_->release_pages(ids, n);
}

cere_status VulkanDevice::take_pages(host::DeviceModel &model, uint64_t n, std::vector<uint32_t> *out) {
    auto &m = static_cast<VulkanModel &>(model);
    if (out == nullptr) return CERE_ERR_ARGUMENT;
    std::lock_guard<std::mutex> lock(m.slots_mutex_);
    if (n > m.pool_->free_pages()) return CERE_ERR_CAPACITY;
    try {
        out->reserve(out->size() + n);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    return m.pool_->take_pages(n, out);
}

cere_status VulkanDevice::bind_pages(host::DeviceModel &model, host::DeviceSession &session, const uint32_t *prefix,
                                     uint64_t n_prefix) {
    auto &m = static_cast<VulkanModel &>(model);
    auto &s = static_cast<VulkanSession &>(session);
    if (&s.model_ != &m) return CERE_ERR_ARGUMENT;
    if (s.broken_) return CERE_ERR_INTERNAL;
    const uint64_t n_pages = host::kv_pages(s.max_context_tokens_);
    if (!s.lazy_ || !s.pages_.empty() || s.position_ != 0 || n_prefix > n_pages) return CERE_ERR_ARGUMENT;
    if (prefix == nullptr && n_prefix != 0) return CERE_ERR_ARGUMENT;
    std::vector<uint32_t> pages;
    try {
        pages.reserve(n_pages);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    {
        // One critical section: the shared pages gain the session's reference and its own pages are taken (lowest
        // free first) together, so CAPACITY keeps nothing.
        std::lock_guard<std::mutex> lock(m.slots_mutex_);
        for (uint64_t i = 0; i < n_prefix; ++i) {
            if (prefix[i] >= m.pool_->pages() || m.pool_->refs(prefix[i]) == 0) return CERE_ERR_ARGUMENT;
        }
        pages.assign(prefix, prefix + n_prefix);
        if (const cere_status st = m.pool_->take_pages(n_pages - n_prefix, &pages); st != CERE_OK) return st;
        if (const cere_status st = m.pool_->retain(prefix, n_prefix); st != CERE_OK) {
            host::release_pages_logged(*m.pool_, pages.data() + n_prefix, pages.size() - n_prefix, "vk_bind_pages");
            return st;
        }
        s.pages_ = std::move(pages);
    }
    return write_slot_map(m, s);
}

uint32_t VulkanDevice::end_kv_slots(const host::DeviceModel &model, uint64_t rows, uint64_t cls) const {
    // cuda_device.cpp:1916-1925: a chunk-0 end state's BF16 K/V rows are kept only when its cold call (cls > rows
    // rows) runs the dense K6 path over its own rows (cls <= kMaxContext = 2,048) on a context the > 2,048 plan
    // serves; the slots hold the rows [0, rows) of every full-attention layer (K then V, [rows, 512] BF16 each).
    // No MTP layer: a prefix model has no drafts (vk_plan_layout refuses the pair).
    const auto &m = static_cast<const VulkanModel &>(model);
    const cere_qwen4_exp_config &c = m.plan_.config;
    if (rows == 0 || cls <= rows || cls > uint64_t(vkk6::kMaxContext) ||
        m.plan_.max_context_tokens <= uint64_t(vkk6::kMaxContext) || m.layout_.state_bytes == 0) {
        return 0;
    }
    uint64_t layers = 0;
    for (uint32_t l = 0; l < c.num_hidden_layers && l < 64; ++l) layers += (c.full_attention_layers >> l) & 1;
    const uint64_t row_bytes = uint64_t(c.num_key_value_heads) * c.head_dim * 2;
    const uint64_t bytes = 2 * layers * rows * row_bytes;
    return uint32_t((bytes + m.layout_.state_bytes - 1) / m.layout_.state_bytes);
}

cere_status VulkanDevice::copy_device(uint64_t dst, uint64_t src, uint64_t bytes) {
    if (dst == 0 || src == 0 || bytes == 0 || (bytes & 15) != 0) return CERE_ERR_ARGUMENT;
    vk::Context &ctx = *ctx_;
    if (const cere_status st = ctx.begin(); st != CERE_OK) return st;
    // One k1 copy_segments dispatch: rows of 64 KiB (16-byte units), at most 65,535 rows per segment.
    constexpr uint64_t kRow = uint64_t(1) << 16;
    vkk1::CopySegment seg[vkk1::kMaxCopySegments];
    int count = 0;
    uint64_t at = 0;
    cere_status st = CERE_OK;
    while (at < bytes && st == CERE_OK) {
        if (count == vkk1::kMaxCopySegments) {
            st = CERE_ERR_INTERNAL;
            break;
        }
        const uint64_t rest = bytes - at;
        const uint64_t rows = rest >= kRow ? std::min<uint64_t>(rest / kRow, 65535) : 1;
        const uint64_t width = rest >= kRow ? kRow : rest;
        seg[count++] = vkk1::CopySegment{dst + at, width, src + at, width, width, rows};
        at += rows * width;
    }
    if (st == CERE_OK) {
        vk::TableRegion t{};
        st = ctx.table_alloc(sizeof(vkk1::CopySegment) * uint64_t(count), 16, &t);
        if (st == CERE_OK) {
            vk::Buffer tb = t.view();
            st = vkk1::copy_segments(ctx, seg, count, tb);
        }
    }
    const cere_status sub = ctx.submit_wait();  // closes the command buffer whatever st was
    return st != CERE_OK ? st : sub;
}

uint64_t VulkanModel::state_slot_addr(uint64_t slot) const {
    if (slot >= layout_.state_slots) return 0;
    const uint64_t base = region_addr(VkRegionKind::kStateSlots);
    return base == 0 ? 0 : base + slot * layout_.state_bytes;
}

cere_status VulkanDevice::capture_state(host::DeviceModel &model, const host::DeviceSession &session, uint32_t slot) {
    auto &m = static_cast<VulkanModel &>(model);
    const auto &s = static_cast<const VulkanSession &>(session);
    if (&s.model_ != &m || slot >= m.layout_.state_slots) return CERE_ERR_ARGUMENT;
    if (s.broken_) return CERE_ERR_INTERNAL;
    if (ctx_->faulted()) return CERE_ERR_INTERNAL;
    const uint64_t run = s.state_.slot + model::layout::fixed_state(m.plan_.config, s.max_context_tokens_).begin;
    return copy_device(m.state_slot_addr(slot), run, m.layout_.state_bytes);
}

cere_status VulkanDevice::load_slot(host::DeviceModel &model, host::DeviceSession &session, uint32_t slot,
                                    uint64_t position) {
    auto &m = static_cast<VulkanModel &>(model);
    auto &s = static_cast<VulkanSession &>(session);
    if (&s.model_ != &m || slot >= m.layout_.state_slots) return CERE_ERR_ARGUMENT;
    if (s.broken_) return CERE_ERR_INTERNAL;
    if (s.pages_.empty() || s.position_ != 0 || s.prompt_open_ || position == 0 ||
        position % host::kKvPageTokens != 0 || position >= s.max_context_tokens_) {
        return CERE_ERR_ARGUMENT;
    }
    if (ctx_->faulted()) return CERE_ERR_INTERNAL;
    const uint64_t run = s.state_.slot + model::layout::fixed_state(m.plan_.config, s.max_context_tokens_).begin;
    if (const cere_status st = copy_device(run, m.state_slot_addr(slot), m.layout_.state_bytes); st != CERE_OK) {
        if (ctx_->faulted()) s.broken_ = true;
        return st;
    }
    // The state of a session that consumed `position` prompt tokens in calls ending at a capture point: the slot map
    // was written at bind_pages and out_counts are still zero (no sampled step), so only the position and the open
    // prompt remain (D3').
    s.position_ = position;
    s.prompt_open_ = true;
    return CERE_OK;
}

// cuda_device.cpp:1241-1408 step: no images (vision), no MixedPD park (a StepSlice runs whole: paused false, allowed by
// device.h:139-145). D15: the EndStep (capture / resume through state-slot pairs) as cuda_device.cpp:1279-1347.
cere_status VulkanDevice::step(host::DeviceModel &model, host::DeviceSession &session, const uint32_t *tokens,
                               uint64_t count, uint64_t limit, const host::StepImages *images, uint64_t *consumed,
                               const host::ItemSampling *sampling, cere_token_score *out, cere_token_logprobs *logprobs,
                               const host::EndStep *end, host::StepSlice *slice) {
    auto &m = static_cast<VulkanModel &>(model);
    auto &s = static_cast<VulkanSession &>(session);
    if (slice != nullptr) slice->paused = false;
    if (consumed == nullptr || count == 0 || tokens == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    if (sampling == nullptr && logprobs != nullptr) return CERE_ERR_ARGUMENT;
    *consumed = 0;
    if (&s.model() != &m || !m.instantiated()) return CERE_ERR_ARGUMENT;
    if (images != nullptr) return CERE_ERR_UNSUPPORTED;  // vision: not built
    if (!forward_supported()) return CERE_ERR_UNSUPPORTED;
    if (s.broken()) return CERE_ERR_INTERNAL;
    if (s.pages().empty()) return CERE_ERR_ARGUMENT;  // a lazy session before bind_pages
    if (count > s.max_context_tokens() - s.position()) return CERE_ERR_CAPACITY;
    // cuda_device.cpp:1256: the prompt from its first step, chunk by chunk, is PREFILL; every later token DECODE.
    const uint32_t operation = (s.position() == 0 || s.prompt_open()) ? CERE_OP_PREFILL : CERE_OP_DECODE;
    const uint64_t chunk = prefill_chunk(m);
    const auto n = static_cast<uint32_t>(std::min(std::min(chunk, count), limit == 0 ? count : limit));
    if (operation == CERE_OP_DECODE && n != 1) return CERE_ERR_ARGUMENT;
    // D15''''' (host/device.h EndStep): the pairs' views and the call's EndCall, checked before anything runs.
    VkEndCall ecall;
    VkSessionState pair[2];
    VkSessionState in_pair[2];
    if (end != nullptr) {
        const bool capture = end->row > 0;
        const bool resume = end->resume;
        // A chunk-0 end state's K/V runs (EndStep::kv_slots, in_kv_slots): the capture's rows [0, position + row),
        // the resume's rows [0, position) after load_slot. The cold call's rows lie inside its class.
        const uint64_t position = s.position();
        const uint64_t kv_cls = end->cls != 0 ? end->cls : n;
        const uint64_t slots = m.layout_.state_slots;
        const uint64_t out_end = uint64_t(end->slot) + 2 + end->kv_slots;
        const uint64_t in_end = uint64_t(end->in_slot) + 2 + end->in_kv_slots;
        const bool capture_ok =
            !capture || (out_end <= slots && end->row < n &&
                         (end->kv_slots == 0 ||
                          ((position == 0 || (resume && end->in_kv_slots != 0)) && position + n <= kv_cls &&
                           end_kv_slots(model, position + end->row, kv_cls) == end->kv_slots)));
        const bool resume_ok =
            !resume || (in_end <= slots && (end->in_kv_slots == 0 ||
                                            (position > 0 && position + n <= kv_cls &&
                                             end_kv_slots(model, position, kv_cls) == end->in_kv_slots)));
        const bool disjoint = !capture || !resume || out_end <= end->in_slot || in_end <= end->slot;
        if (operation != CERE_OP_PREFILL || s.pages().empty() || (!capture && !resume) ||
            (end->cls != 0 && end->cls < n) || end->cls > m.layout_.workspace_rows || !capture_ok || !resume_ok ||
            !disjoint) {
            return CERE_ERR_ARGUMENT;
        }
        for (uint32_t i = 0; i < 2; ++i) {
            if (capture) {
                if (const cere_status st = vk_bind_fixed_state(m.plan_, m.state_slot_addr(end->slot + i),
                                                               s.max_context_tokens(), &pair[i]);
                    st != CERE_OK) {
                    return st;
                }
            }
            if (resume) {
                if (const cere_status st = vk_bind_fixed_state(m.plan_, m.state_slot_addr(end->in_slot + i),
                                                               s.max_context_tokens(), &in_pair[i]);
                    st != CERE_OK) {
                    return st;
                }
            }
        }
        ecall.cls = uint32_t(end->cls);
        if (capture) {
            ecall.row = int64_t(end->row);
            ecall.a = &pair[0];
            ecall.b = &pair[1];
            if (end->kv_slots != 0) {
                ecall.kv = m.state_slot_addr(end->slot + 2);
                ecall.kv_rows = int64_t(position + end->row);
            }
        }
        if (resume) {
            ecall.in_a = &in_pair[0];
            ecall.in_b = &in_pair[1];
            if (end->in_kv_slots != 0) {
                ecall.in_kv = m.state_slot_addr(end->in_slot + 2);
                ecall.in_kv_rows = int64_t(position);
            }
        }
        // A capture's slot first takes the whole fixed state as of the call start (rings and the rest); the forward
        // then overwrites its GDN, SSM and PLE parts as of `row` (submission order).
        if (capture) {
            if (const cere_status st = capture_state(model, session, end->slot); st != CERE_OK) {
                faulted();
                return st;
            }
        }
    }
    const bool last = n == count;  // only the call that ends the fed tokens scores (and samples) the next token
    cere_token_score score{};
    cere_token_logprobs lp{};
    VkForwardCall call;
    call.operation = operation;
    call.tokens = tokens;
    call.n = n;
    call.sampler = last ? sampling : nullptr;
    call.out = &score;
    call.out_logprobs = last && logprobs != nullptr ? &lp : nullptr;
    call.end = end != nullptr ? &ecall : nullptr;
    bool launched = false;
    const cere_status st = vk_forward(*ctx_, m, s, call, &launched);
    if (st != CERE_OK) {
        // cuda_device.cpp:1187-1192: after the first launch any failure leaves the session undefined (MF-2).
        if (launched) s.set_broken();
        faulted();  // prints the device_poisoned line once on a device loss
        return st;
    }
    s.set_prompt_open(!last);
    // T6 (cuda_device.cpp:1376-1401): a drafting session's MTP extend over the chunk (next ids = the chunk shifted by
    // one, then the scored token or the prompt's next token), its decode steps only after the prompt's last chunk. A
    // failure breaks the session (its target state already advanced).
    if (s.drafting()) {
        bool ext_launched = false;
        const cere_status ext = vk_extend_draft(*ctx_, m, s, tokens, n, last ? score.id : tokens[n], !last,
                                                &ext_launched);
        if (ext != CERE_OK) {
            s.set_broken();
            faulted();
            return ext;
        }
    }
    *consumed = n;
    if (last) {
        *out = score;
        if (logprobs != nullptr) *logprobs = lp;
    }
    return CERE_OK;
}

// cuda_device.cpp:2031-2102 step_rows: with max_sessions 1 one participant at most, which is the single-row DECODE
// forward (cuda_device.cpp:2079-2089); the batched forward_rows is not built (R = 1).
cere_status VulkanDevice::step_rows(host::DeviceModel &model, const host::RowArgs *rows, uint32_t row_count,
                                    cere_token_score *out, cere_status *row_status) {
    auto &m = static_cast<VulkanModel &>(model);
    if (rows == nullptr || out == nullptr || row_status == nullptr || row_count == 0 ||
        row_count > m.plan().max_sessions) {
        return CERE_ERR_ARGUMENT;
    }
    for (uint32_t r = 0; r < row_count; ++r) {
        const host::DeviceSession *s = rows[r].session;
        if (s == nullptr || &static_cast<const VulkanSession *>(s)->model() != &m) return CERE_ERR_ARGUMENT;
        if (rows[r].sampling == nullptr && rows[r].logprobs != nullptr) return CERE_ERR_ARGUMENT;
        for (uint32_t q = 0; q < r; ++q) {
            if (rows[q].session == s) return CERE_ERR_ARGUMENT;
        }
    }
    if (!m.instantiated()) return CERE_ERR_ARGUMENT;
    const uint32_t vocab = m.plan().config.vocab_size;
    uint32_t participants = 0, index = 0;
    for (uint32_t r = 0; r < row_count; ++r) {
        auto &s = static_cast<VulkanSession &>(*rows[r].session);
        cere_status st = CERE_OK;
        if (s.broken()) {
            st = CERE_ERR_INTERNAL;
        } else if (!forward_supported()) {
            st = CERE_ERR_UNSUPPORTED;
        } else if (s.position() >= s.max_context_tokens()) {
            st = CERE_ERR_CAPACITY;
        } else if (s.position() == 0 || s.prompt_open() || rows[r].token >= vocab) {
            st = CERE_ERR_ARGUMENT;
        } else if (rows[r].sampling != nullptr) {
            st = vk_check_sampler(*rows[r].sampling, vocab);
        }
        row_status[r] = st;
        if (st != CERE_OK) continue;
        ++participants;
        index = r;
    }
    if (participants == 0) return CERE_OK;
    if (participants > 1) return CERE_ERR_INTERNAL;  // unreachable: row_count <= max_sessions == 1
    auto &s = static_cast<VulkanSession &>(*rows[index].session);
    VkForwardCall call;
    call.operation = CERE_OP_DECODE;
    call.tokens = &rows[index].token;
    call.n = 1;
    call.sampler = rows[index].sampling;
    call.out = &out[index];
    call.out_logprobs = rows[index].logprobs;
    bool launched = false;
    const cere_status st = vk_forward(*ctx_, m, s, call, &launched);
    if (st != CERE_OK) {
        s.set_broken();  // cuda_device.cpp:2084-2087 (a participant's failure breaks its session)
        row_status[index] = st;
        faulted();
    }
    return st;
}

cere_status VulkanDevice::set_rope_delta(host::DeviceModel &, host::DeviceSession &, int64_t rope_delta) {
    if (rope_delta > 0) return CERE_ERR_ARGUMENT;
    return rope_delta == 0 ? CERE_OK : CERE_ERR_UNSUPPORTED;  // only an image prompt makes it negative
}

uint64_t VulkanDevice::free_pages(const host::DeviceModel &model) const {
    auto &m = const_cast<VulkanModel &>(static_cast<const VulkanModel &>(model));
    std::lock_guard<std::mutex> lock(m.slots_mutex_);
    return m.pool_->free_pages();
}

uint32_t VulkanDevice::page_refs(const host::DeviceModel &model, uint32_t id) const {
    auto &m = const_cast<VulkanModel &>(static_cast<const VulkanModel &>(model));
    std::lock_guard<std::mutex> lock(m.slots_mutex_);
    return id < m.pool_->pages() ? m.pool_->refs(id) : 0;
}

std::vector<uint32_t> VulkanDevice::session_page_list(const host::DeviceSession &session) const {
    return static_cast<const VulkanSession &>(session).pages();
}

uint64_t VulkanDevice::session_position(const host::DeviceSession &session) const {
    return static_cast<const VulkanSession &>(session).position();
}

uint64_t VulkanDevice::prefill_chunk(const host::DeviceModel &model) const {
    return static_cast<const VulkanModel &>(model).layout_.workspace_rows;
}

}  // namespace cere::vkdev

namespace cere::host {

cere_status create_device(const char *device_id, std::unique_ptr<Device> *out) {
    try {
        return cere::vkdev::VulkanDevice::open(device_id, out);
    } catch (const std::exception &) {
        return CERE_ERR_INTERNAL;
    }
}

// Only an sdar library calls it (api.cpp, #if CERE_GEN_FAMILY == CERE_FAMILY_SDAR); a Qwen library runs no SDAR model.
const Support *sdar_support(const Device &) { return nullptr; }

}  // namespace cere::host
