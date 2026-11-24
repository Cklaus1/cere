// VulkanAmd P3 T2: see vk_memory_plan.h for the rules (R1-R6), the scope and the refusals.
#include "device/vk_memory_plan.h"

#include "device/vk_arena_order.h"
#include "device/vk_kernel_sizes.h"
#include "host/sha256.h"
#include "host/support.h"
#include "model/layout.h"
#include "model/mxfp8_plan.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <initializer_list>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace cere::vkdev {

namespace {

namespace layout = cere::model::layout;

bool ends_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.substr(s.size() - p.size()) == p;
}
bool pow2(uint64_t v) { return v != 0 && (v & (v - 1)) == 0; }

// Saturating arithmetic as CUDA's memory_plan.cpp Calc: any overflow sets `overflow` (refused as CAPACITY).
struct Calc {
    bool overflow = false;
    uint64_t add(uint64_t a, uint64_t b) {
        uint64_t r = 0;
        overflow |= __builtin_add_overflow(a, b, &r);
        return r;
    }
    uint64_t mul(uint64_t a, uint64_t b) {
        uint64_t r = 0;
        overflow |= __builtin_mul_overflow(a, b, &r);
        return r;
    }
    uint64_t align(uint64_t v, uint64_t a) { return mul(add(v, a - 1) / a, a); }
};

// The dtypes the Vulkan kernels take (BF16 weights/activations, E4M3 MXFP8 weights / NVFP4 block scales / KV, U8 NVFP4
// codes and UE8M0 scales, F32 scalars, I64 PLE index): the A4 support's set (host/support.cpp kA4Dtypes).
bool vulkan_dtype(uint32_t dtype) {
    switch (dtype) {
        case CERE_DTYPE_BF16:
        case CERE_DTYPE_F32:
        case CERE_DTYPE_F8_E4M3:
        case CERE_DTYPE_U8:
        case CERE_DTYPE_I64: return true;
        default: return false;
    }
}

const char *dtype_name(uint32_t d) {
    switch (d) {
        case CERE_DTYPE_BF16: return "BF16";
        case CERE_DTYPE_F16: return "F16";
        case CERE_DTYPE_F32: return "F32";
        case CERE_DTYPE_F64: return "F64";
        case CERE_DTYPE_F8_E4M3: return "F8_E4M3";
        case CERE_DTYPE_F8_E5M2: return "F8_E5M2";
        case CERE_DTYPE_U8: return "U8";
        case CERE_DTYPE_I8: return "I8";
        case CERE_DTYPE_I16: return "I16";
        case CERE_DTYPE_I32: return "I32";
        case CERE_DTYPE_I64: return "I64";
        case CERE_DTYPE_BOOL: return "BOOL";
        default: return "?";
    }
}

// Next-fit placement into the blocks of one group (R2, R5).
class Packer {
public:
    Packer(VkLayout &l, Calc &k, uint64_t cap) : l_(l), k_(k), cap_(cap) {}

    void group(VkHeap heap, VkPurpose purpose) {
        heap_ = heap;
        purpose_ = purpose;
        cur_ = -1;
    }
    // Places `bytes` at alignment `a` (a == 1: abut the cursor). False when the item is larger than one block.
    bool place(uint64_t bytes, uint64_t a, uint32_t *block, uint64_t *offset) {
        if (bytes > cap_) return false;
        if (cur_ >= 0) {
            VkBlock &b = l_.blocks[size_t(cur_)];
            const uint64_t at = k_.align(b.used, a);
            if (!k_.overflow && at <= cap_ && bytes <= cap_ - at) {
                b.used = at + bytes;
                *block = uint32_t(cur_);
                *offset = at;
                return true;
            }
        }
        l_.blocks.push_back(VkBlock{heap_, purpose_, 0, bytes});
        cur_ = int64_t(l_.blocks.size() - 1);
        *block = uint32_t(cur_);
        *offset = 0;
        return true;
    }

private:
    VkLayout &l_;
    Calc &k_;
    uint64_t cap_;
    VkHeap heap_ = VkHeap::kDeviceLocal;
    VkPurpose purpose_ = VkPurpose::kWeights;
    int64_t cur_ = -1;
};

struct Fail {
    cere_status status;
    std::string why;
};

// Collected (tensor, piece) pairs, sorted into piece_first / pieces at the end.
struct PieceSink {
    std::vector<std::pair<uint64_t, VkPiece>> all;
    void add(uint64_t tensor, const VkPiece &p) { all.emplace_back(tensor, p); }
};

// Members [start, start + bytes) of a logically contiguous stream laid out as chunks (chunk c: stream bytes
// [chunk_start[c], chunk_start[c] + chunk_bytes[c]) at (chunk_block[c], chunk_offset[c])): the pieces of the member.
void stream_pieces(uint64_t tensor, uint64_t start, uint64_t bytes, const std::vector<uint64_t> &chunk_start,
                   const std::vector<uint64_t> &chunk_bytes, const std::vector<uint32_t> &chunk_block,
                   const std::vector<uint64_t> &chunk_offset, PieceSink *sink) {
    if (bytes == 0) {
        // A zero-byte member still has one (empty) piece where it would start.
        size_t c = size_t(std::upper_bound(chunk_start.begin(), chunk_start.end(), start) - chunk_start.begin());
        c = c == 0 ? 0 : c - 1;
        sink->add(tensor, VkPiece{chunk_block[c], chunk_offset[c] + (start - chunk_start[c]), 0, 0});
        return;
    }
    size_t c = size_t(std::upper_bound(chunk_start.begin(), chunk_start.end(), start) - chunk_start.begin()) - 1;
    uint64_t done = 0;
    while (done < bytes) {
        const uint64_t at = start + done;
        while (at >= chunk_start[c] + chunk_bytes[c]) ++c;
        const uint64_t in = at - chunk_start[c];
        const uint64_t take = std::min(bytes - done, chunk_bytes[c] - in);
        sink->add(tensor, VkPiece{chunk_block[c], chunk_offset[c] + in, done, take});
        done += take;
    }
}

uint64_t tensor_elements(const host::TensorPlan &t, Calc &k) {
    uint64_t n = 1;
    for (uint32_t d = 0; d < t.ndim && d < CERE_TENSOR_MAX_DIMS; ++d) n = k.mul(n, t.shape[d]);
    return n;
}

}  // namespace

const char *vk_purpose_name(VkPurpose p) {
    switch (p) {
        case VkPurpose::kWeights: return "weights";
        case VkPurpose::kWorkspace: return "workspace";
        case VkPurpose::kState: return "state";
        case VkPurpose::kKvPool: return "kv_pool";
        case VkPurpose::kIo: return "io";
        case VkPurpose::kPle: return "ple";
    }
    return "?";
}

const char *vk_region_name(VkRegionKind k) {
    switch (k) {
        case VkRegionKind::kDerived: return "derived";
        case VkRegionKind::kMtpDerived: return "mtp_derived";
        case VkRegionKind::kMoeInfo: return "moe_info";
        case VkRegionKind::kWinfo: return "k2_winfo";
        case VkRegionKind::kWorkspaceRows: return "workspace_rows";
        case VkRegionKind::kScratch: return "scratch";
        case VkRegionKind::kGemmWorkspace: return "gemm_workspace";
        case VkRegionKind::kMtpHcRows: return "mtp_hc_rows";
        case VkRegionKind::kSession: return "session";
        case VkRegionKind::kUndo: return "undo";
        case VkRegionKind::kSpecSampler: return "spec_sampler";
        case VkRegionKind::kStateSlots: return "state_slots";
        case VkRegionKind::kKCache: return "k_cache";
        case VkRegionKind::kVCache: return "v_cache";
        case VkRegionKind::kQsaKeys: return "qsa_keys";
        case VkRegionKind::kMtpKCache: return "mtp_k_cache";
        case VkRegionKind::kMtpVCache: return "mtp_v_cache";
        case VkRegionKind::kMtpQsaKeys: return "mtp_qsa_keys";
        case VkRegionKind::kIoIds32: return "io_ids32";
        case VkRegionKind::kIoIds64: return "io_ids64";
        case VkRegionKind::kIoTop2: return "io_top2";
        case VkRegionKind::kIoLogprobs: return "io_logprobs";
        case VkRegionKind::kIoBoundFault: return "io_bound_fault";
        case VkRegionKind::kIoBias: return "io_bias";
        case VkRegionKind::kIoMask: return "io_mask";
        case VkRegionKind::kIoPleBlockTable: return "io_ple_block_table";
        case VkRegionKind::kIoTableRing: return "io_table_ring";
    }
    return "?";
}

VkLayout vk_plan_layout(const host::ModelPlan &plan, const VkLimits &limits) {
    VkLayout l;
    l.limits = limits;
    const auto refuse = [&l](cere_status s, std::string why) {
        VkLayout r;
        r.limits = l.limits;
        r.status = s;
        r.error = std::move(why);
        return r;
    };
    // ---- limits ----
    if (limits.block_granule == 0 || !pow2(limits.region_alignment) || !pow2(limits.small_alignment) ||
        limits.region_alignment > limits.block_granule || limits.small_alignment > limits.region_alignment) {
        return refuse(CERE_ERR_ARGUMENT, "limits: alignments must be powers of two, small <= region <= granule");
    }
    const uint64_t cap = limits.max_buffer_bytes / limits.block_granule * limits.block_granule;
    if (cap == 0) return refuse(CERE_ERR_ARGUMENT, "limits: max_buffer_bytes holds no block_granule");

    // ---- first-light scope (UNSUPPORTED, never silently dropped) ----
    const cere_qwen4_exp_config &c = plan.config;
    if (plan.support == nullptr) return refuse(CERE_ERR_ARGUMENT, "plan has no support entry");
    if (plan.support->vision != nullptr) return refuse(CERE_ERR_UNSUPPORTED, "scope: vision plans are not supported");
    if (plan.max_sessions == 0) return refuse(CERE_ERR_ARGUMENT, "max_sessions 0");
    if (plan.max_sessions > limits.max_sessions) {
        return refuse(CERE_ERR_UNSUPPORTED, "scope: max_sessions " + std::to_string(plan.max_sessions) + " > " +
                                                std::to_string(limits.max_sessions));
    }
    if (plan.max_context_tokens <= limits.min_context_exclusive) {
        return refuse(CERE_ERR_UNSUPPORTED, "scope (OD-5): max_context_tokens " +
                                                std::to_string(plan.max_context_tokens) + " <= " +
                                                std::to_string(limits.min_context_exclusive) +
                                                " takes k6::full_attention, which Vulkan has not");
    }
    if (plan.max_context_tokens > limits.max_context_tokens) {
        return refuse(CERE_ERR_UNSUPPORTED, "scope: max_context_tokens " + std::to_string(plan.max_context_tokens) +
                                                " > " + std::to_string(limits.max_context_tokens));
    }
    if (plan.prefix_state_slots != 0 && plan.speculative_draft_tokens != 0) {
        // D15 scope: a drafting session's prefix states (draft states, end states with MTP K/V) are not ported.
        return refuse(CERE_ERR_UNSUPPORTED, "scope: prefix cache (prefix_state_slots) with speculative drafts");
    }
    if (plan.snapshot_host_bytes != 0) return refuse(CERE_ERR_UNSUPPORTED, "scope: snapshot arena");
    if (plan.prefill_park) return refuse(CERE_ERR_UNSUPPORTED, "scope: MixedPD park region (prefill_park)");
    if (plan.speculative_draft_tokens != 0 && plan.speculative_draft_tokens != 4) {
        return refuse(CERE_ERR_UNSUPPORTED, "scope: speculative_draft_tokens " +
                                                std::to_string(plan.speculative_draft_tokens) + " (0 or 4)");
    }
    if (plan.speculative_draft_tokens != 0 && plan.draft_token_map.empty()) {
        return refuse(CERE_ERR_ARGUMENT, "drafts without a draft token map");
    }
    if (plan.kv_pool_tokens < plan.max_context_tokens || plan.kv_pool_tokens % 64 != 0) {
        return refuse(CERE_ERR_ARGUMENT, "kv_pool_tokens " + std::to_string(plan.kv_pool_tokens) +
                                             " not a multiple of 64 >= max_context_tokens");
    }
    Calc k;
    for (size_t i = 0; i < plan.tensors.size(); ++i) {
        const host::TensorPlan &t = plan.tensors[i];
        if (t.placement == CERE_PLACEMENT_DISCARD) continue;
        if (t.placement != CERE_PLACEMENT_DEVICE && t.placement != CERE_PLACEMENT_HOST_PINNED) {
            return refuse(CERE_ERR_ARGUMENT, "tensor " + t.name + ": placement " + std::to_string(t.placement));
        }
        if (!vulkan_dtype(t.dtype)) {
            return refuse(CERE_ERR_UNSUPPORTED, "tensor " + t.name + ": dtype " + dtype_name(t.dtype) +
                                                    " is not taken by the Vulkan kernels");
        }
        Calc e;
        const uint64_t want = e.mul(tensor_elements(t, e), host::dtype_size(t.dtype));
        if (e.overflow || want != t.bytes) {
            return refuse(CERE_ERR_ARGUMENT, "tensor " + t.name + ": bytes " + std::to_string(t.bytes) +
                                                 " disagree with its shape");
        }
        (t.placement == CERE_PLACEMENT_DEVICE ? l.device_tensor_bytes : l.host_tensor_bytes) =
            k.add(t.placement == CERE_PLACEMENT_DEVICE ? l.device_tensor_bytes : l.host_tensor_bytes, t.bytes);
    }
    {
        const model::mxfp8::Plan mx = model::mxfp8::plan_of(plan);
        if (!mx.error.empty()) return refuse(CERE_ERR_UNSUPPORTED, "MXFP8: " + mx.error);
    }

    const bool drafts = plan.speculative_draft_tokens > 0;
    l.draft_count = drafts ? plan.draft_token_map.size() : 0;
    l.workspace_rows = std::min<uint64_t>(plan.max_context_tokens, gen::kGraphMaxChunkTokens);  // model.cpp:627-629
    const uint64_t t = l.workspace_rows;
    const uint64_t ctx = plan.max_context_tokens;

    // ---- arena runs and the PLE table geometry (R3, R4) ----
    const std::vector<ArenaRun> runs = arena_runs(plan);
    int64_t table_run = -1, scale_run = -1;
    for (size_t r = 0; r < runs.size(); ++r) {
        if (ends_with(runs[r].key, ".ple.shards@host")) {
            if (table_run >= 0) return refuse(CERE_ERR_UNSUPPORTED, "PLE: more than one host table run");
            table_run = int64_t(r);
        } else if (ends_with(runs[r].key, ".ple.shard_scales@host")) {
            if (scale_run >= 0) return refuse(CERE_ERR_UNSUPPORTED, "PLE: more than one host scale run");
            scale_run = int64_t(r);
        }
    }
    VkPleSplit &ple = l.ple;
    if (table_run >= 0) {
        const ArenaRun &tr = runs[size_t(table_run)];
        const host::TensorPlan &t0 = plan.tensors[tr.members[0]];
        ple.present = true;
        ple.shards = uint32_t(tr.members.size());
        ple.format = scale_run >= 0 ? 1 : 0;
        const uint64_t want_row = ple.format == 1 ? sizes::kK4Nvfp4RowBytes : 160;  // vkk4::row_bytes
        if (t0.ndim != 2 || t0.shape[0] == 0 || t0.shape[1] * host::dtype_size(t0.dtype) != want_row ||
            (ple.format == 1 && t0.dtype != CERE_DTYPE_U8) || (ple.format == 0 && t0.dtype != CERE_DTYPE_F8_E4M3)) {
            return refuse(CERE_ERR_UNSUPPORTED, "PLE: shard " + t0.name + " is not K4's " +
                                                    (ple.format == 1 ? "NVFP4 U8 [rows, 80]" : "FP8 E4M3 [rows, 160]"));
        }
        ple.shard_rows = t0.shape[0];
        ple.row_bytes = uint32_t(want_row);
        for (size_t m = 0; m < tr.members.size(); ++m) {
            const host::TensorPlan &tm = plan.tensors[tr.members[m]];
            if (tr.ranks[m] != m || tm.ndim != 2 || tm.shape[0] != t0.shape[0] || tm.shape[1] != t0.shape[1] ||
                tm.dtype != t0.dtype) {
                return refuse(CERE_ERR_UNSUPPORTED, "PLE: shards not uniform or not numbered 0..N-1 (" + tm.name + ")");
            }
        }
        if (scale_run >= 0) {
            const ArenaRun &sr = runs[size_t(scale_run)];
            ple.scale_bytes = sizes::kK4Nvfp4RowScales;
            if (sr.members.size() != tr.members.size()) {
                return refuse(CERE_ERR_UNSUPPORTED, "PLE: " + std::to_string(sr.members.size()) +
                                                        " scale shards for " + std::to_string(tr.members.size()) +
                                                        " table shards");
            }
            for (size_t m = 0; m < sr.members.size(); ++m) {
                const host::TensorPlan &tm = plan.tensors[sr.members[m]];
                if (sr.ranks[m] != m || tm.ndim != 2 || tm.shape[0] != ple.shard_rows ||
                    tm.shape[1] != sizes::kK4Nvfp4RowScales || tm.dtype != CERE_DTYPE_F8_E4M3) {
                    return refuse(CERE_ERR_UNSUPPORTED, "PLE: scale shard " + tm.name + " is not E4M3 [" +
                                                            std::to_string(ple.shard_rows) + ", 10]");
                }
            }
        }
        ple.rows = k.mul(ple.shard_rows, ple.shards);
        if (k.overflow || ple.rows >= (uint64_t(1) << 32)) {
            return refuse(CERE_ERR_UNSUPPORTED, "PLE: table rows outside [1, 2^32) (k4_ple_vk.h SplitTable::create)");
        }
        // Rows per block: whole chunk bytes within one block, 32-bit in-block byte offsets (k4_ple_vk.cpp:226-229).
        uint64_t max_rows = std::min<uint64_t>(cap / ple.row_bytes, ((uint64_t(1) << 32) - 1) / ple.row_bytes);
        if (ple.scale_bytes != 0) {
            max_rows = std::min<uint64_t>(max_rows, cap / ple.scale_bytes);
        }
        max_rows = std::min<uint64_t>(max_rows, ple.rows);
        if (max_rows == 0) {
            return refuse(CERE_ERR_UNSUPPORTED, "PLE: one " + std::to_string(ple.row_bytes) +
                                                    "-byte row does not fit one block: no row-boundary split");
        }
        const uint64_t rpb = max_rows >= ple.shard_rows ? max_rows / ple.shard_rows * ple.shard_rows : max_rows;
        ple.rows_per_block = uint32_t(rpb);
        ple.shards_per_block = rpb % ple.shard_rows == 0 ? uint32_t(rpb / ple.shard_rows) : 0;
        ple.blocks = uint32_t((ple.rows + rpb - 1) / rpb);
        if (ple.blocks > limits.max_ple_blocks) {
            return refuse(CERE_ERR_UNSUPPORTED, "PLE: " + std::to_string(ple.blocks) + " chunks > " +
                                                    std::to_string(limits.max_ple_blocks) + " (vkk4::kMaxBlocks)");
        }
    }

    // ---- WEIGHTS group (R1, R2, R3) ----
    Packer pack(l, k, cap);
    PieceSink sink;
    const auto tensor_align = [&](const host::TensorPlan &x) {
        return x.bytes < limits.small_tensor_bytes ? limits.small_alignment : limits.region_alignment;
    };
    // Places run `r` whole (member 0 aligned, the rest abutting). False: the run is larger than one block.
    const auto place_run = [&](const ArenaRun &r) {
        uint64_t total = 0;
        for (uint64_t i : r.members) total = k.add(total, plan.tensors[i].bytes);
        uint32_t block = 0;
        uint64_t at = 0;
        if (k.overflow || !pack.place(total, tensor_align(plan.tensors[r.members[0]]), &block, &at)) return false;
        for (uint64_t i : r.members) {
            sink.add(i, VkPiece{block, at, 0, plan.tensors[i].bytes});
            at += plan.tensors[i].bytes;
        }
        return true;
    };
    pack.group(VkHeap::kDeviceLocal, VkPurpose::kWeights);
    for (const ArenaRun &r : runs) {
        if (plan.tensors[r.members[0]].placement != CERE_PLACEMENT_DEVICE) continue;
        if (!place_run(r)) {
            return refuse(CERE_ERR_UNSUPPORTED, "weights: " + (r.key.empty() ? plan.tensors[r.members[0]].name : r.key) +
                                                    " is larger than one Vulkan buffer (" + std::to_string(cap) +
                                                    " B); only the PLE table may be split");
        }
    }
    // Derived regions (written at INSTANTIATE), after the weights in their blocks.
    const auto region = [&](VkRegionKind kind, uint32_t index, uint64_t bytes, uint64_t a) {
        uint32_t block = 0;
        uint64_t at = 0;
        if (k.overflow || !pack.place(bytes, a, &block, &at)) return false;
        l.regions.push_back(VkRegion{kind, index, block, at, bytes});
        return true;
    };
    const auto too_big = [&](VkRegionKind kind, uint64_t bytes) {
        return refuse(CERE_ERR_CAPACITY, std::string("region ") + vk_region_name(kind) + " (" + std::to_string(bytes) +
                                             " B) is larger than one Vulkan buffer (" + std::to_string(cap) + " B)");
    };
    const uint64_t ra = limits.region_alignment;
    {
        const uint64_t b = layout::derived_bytes(c, ctx);
        if (!region(VkRegionKind::kDerived, 0, b, ra)) return too_big(VkRegionKind::kDerived, b);
    }
    if (drafts) {
        const uint64_t b = layout::mtp_derived_bytes(c, l.draft_count);
        if (!region(VkRegionKind::kMtpDerived, 0, b, ra)) return too_big(VkRegionKind::kMtpDerived, b);
    }
    const sizes::MoeDims moe{c.hidden_size, c.num_experts, c.num_experts_per_tok, c.moe_intermediate_size,
                             c.shared_expert_intermediate_size};
    {
        l.moe_info_stride = sizes::k7_weight_info(moe);
        const uint64_t layers = uint64_t(c.num_hidden_layers) + (drafts ? 1 : 0);
        const uint64_t b = k.mul(l.moe_info_stride, layers);
        if (!region(VkRegionKind::kMoeInfo, 0, b, ra)) return too_big(VkRegionKind::kMoeInfo, b);
    }
    {
        // The K2 row info of every GEMM weight whose consumer takes prepared info, grouped per consumer in the order
        // its info layout expects (VkInfoConsumer). A group is added only when all its weights are present, DEVICE,
        // 2-D BF16 / E4M3 with k % 64 == 0 (K2 GEMM rule, k2_gemm_vk.h:69); q|k|v count once (the run's rows).
        std::unordered_map<std::string_view, uint64_t> by_name;
        by_name.reserve(plan.tensors.size());
        for (uint64_t i = 0; i < plan.tensors.size(); ++i) by_name.emplace(plan.tensors[i].name, i);
        uint64_t off = 0;
        const auto group = [&](VkInfoConsumer who, const std::string &prefix, std::initializer_list<const char *> parts) {
            std::vector<VkWinfo> g;
            for (const char *part : parts) {
                static const char *const kQkv[3] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"};
                const bool qkv = std::string_view(part) == "self_attn.qkv";
                const char *const single[1] = {part};
                const char *const *members = qkv ? kQkv : single;
                uint64_t n = 0, kk = 0, first = 0;
                for (size_t mi = 0; mi < (qkv ? 3u : 1u); ++mi) {
                    const char *member = members[mi];
                    const auto it = by_name.find(prefix + member + ".weight");
                    if (it == by_name.end()) return;
                    const host::TensorPlan &w = plan.tensors[it->second];
                    if (w.placement != CERE_PLACEMENT_DEVICE || w.ndim != 2 ||
                        (w.dtype != CERE_DTYPE_BF16 && w.dtype != CERE_DTYPE_F8_E4M3) || w.shape[1] % 64 != 0 ||
                        (kk != 0 && w.shape[1] != kk)) {
                        return;
                    }
                    if (n == 0) first = it->second;
                    n += w.shape[0];
                    kk = w.shape[1];
                }
                if (n > UINT32_MAX || kk > UINT32_MAX) return;
                g.push_back(VkWinfo{first, who, uint32_t(n), uint32_t(kk), 0});
            }
            for (VkWinfo &w : g) {
                w.offset = off;
                off = k.add(off, sizes::k2_winfo(w.n));
                l.gemm_max_n = std::max<uint64_t>(l.gemm_max_n, w.n);
                l.winfo.push_back(w);
            }
        };
        for (uint32_t layer = 0; layer < c.num_hidden_layers && layer < 64; ++layer) {
            const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
            if (layout::full_attention(c, layer)) {
                group(VkInfoConsumer::kForward, prefix,
                      {"self_attn.qkv", "self_attn.o_proj", "self_attn.indexer.index_qk_proj"});
            } else {
                group(VkInfoConsumer::kK5Layer, prefix,
                      {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b",
                       "linear_attn.in_proj_a", "linear_attn.out_proj"});
            }
            group(VkInfoConsumer::kK4Ple, prefix, {"ple.key_proj", "ple.value_proj"});
        }
        if (drafts) {
            group(VkInfoConsumer::kForward, "mtp.", {"fc_embedding", "fc_hidden"});
            group(VkInfoConsumer::kForward, "mtp.layers.0.",
                  {"self_attn.qkv", "self_attn.o_proj", "self_attn.indexer.index_qk_proj"});
        }
        if (!region(VkRegionKind::kWinfo, 0, off, ra)) return too_big(VkRegionKind::kWinfo, off);
    }

    // ---- WORKSPACE group ----
    {
        const uint64_t hidden = c.hidden_size, hc = c.hc_count;
        const uint64_t q_width = uint64_t(c.num_attention_heads) * c.head_dim;
        const uint64_t kv_width = uint64_t(c.num_key_value_heads) * c.head_dim;
        const uint64_t qk_width = uint64_t(c.indexer_n_heads + 1) * c.indexer_head_dim;
        const uint64_t vocab = c.vocab_size, words = (vocab + 31) / 32;
        const auto entry = [&](std::string name, std::string source, bool stale, uint64_t bytes) {
            l.workspace.push_back(VkWorkspaceEntry{std::move(name), std::move(source), stale, bytes});
        };
        entry("K3 hc_mix", "vkk3::hc_mix_workspace_bytes (sizes::k3_hc_mix_workspace)", true,
              sizes::k3_hc_mix_workspace(hc, hidden, c.hc_lowrank, t));
        entry("K4 PLE forward", "vkk4::workspace_bytes (sizes::k4_workspace)", true,
              sizes::k4_workspace(c.ple_embed_dim, hc * hidden, hidden, int64_t(t)));
        entry("K5 GDN layer", "vkk5::layer_workspace_bytes (sizes::k5_layer_workspace)", true,
              sizes::k5_layer_workspace(c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim,
                                        int64_t(t)));
        // model.cpp:337-342 qsa_attention_scratch_bytes with the Vulkan K6/K9 sizers (ctx > 2,048 always: OD-5).
        const uint64_t prologue = sizes::k6_prologue_cuda(q_width, kv_width, int64_t(t));
        const uint64_t qk = sizes::a256(t * qk_width * 2);
        // P4-E: the coopmat prefill attention's size (>= the FP32 path's): either selection runs in these buffers. Its K9
        // flags / routes (vkk9::coopmat_scratch_bytes) sit in the select region, far larger (vk_kernel_sizes_drift_test).
        const uint64_t k6 = sizes::k6_workspace(c.num_attention_heads, c.head_dim, int64_t(t), true);
        const uint64_t k9 = sizes::a256(t * q_width * 2) +
                            sizes::k9_scratch(c.num_attention_heads, c.head_dim, int64_t(t), int64_t(ctx));
        entry("attention: K6 prologue + qk + max(K6 dense, attn_out + K9)",
              "model.cpp:337 rule; vkk6::workspace_bytes (Prefill::kCoopmat), vkk9::scratch_bytes; prologue = CUDA k6 prologue_bytes "
              "(no Vulkan kernel, T3)",
              true, prologue + qk + std::max(k6, k9));
        entry("K7 MoE", "vkk7::moe_workspace_bytes (sizes::k7_moe_workspace)", true, sizes::k7_moe_workspace(moe, t));
        {
            // model.cpp:345-370 carve_sampler + rows_logits, with vkk12::scratch_bytes (k12_sampler_vk.cpp:66).
            layout::Carver s{nullptr};
            s.take<uint8_t>(vocab * 4);
            s.take<uint8_t>(words * 4);
            s.take<uint8_t>(8);
            s.take<uint8_t>(sizes::kK12Scratch);
            s.take<uint8_t>(8);
            s.take<uint8_t>(8);
            s.take<uint8_t>(4);
            s.take<uint8_t>(vocab * sizeof(cere_logit_bias));
            s.take<uint8_t>(sizeof(cere_token_logprobs));
            s.take<uint8_t>(words * 4);
            s.take<uint8_t>(std::min<uint64_t>(t, 16) * vocab * 2);
            entry("K12 sampler stage + rows lm_head", "model.cpp:345-370 carve_sampler/rows_logits, vkk12 scratch",
                  false, s.size());
        }
        if (drafts) {
            // mtp.cu:477-480 scratch_bytes with vkk13::moe_bf16_workspace_bytes.
            const uint64_t hot = sizes::a256(std::min<uint64_t>(16, t) * l.draft_count * 2);
            entry("MTP: max(K13 MoE, hot logits)", "mtp.cu:477 rule; vkk13::moe_bf16_workspace_bytes", true,
                  std::max(sizes::k13_moe_workspace(moe, t), hot));
        }
        for (const VkWorkspaceEntry &e : l.workspace) l.scratch_bytes = std::max(l.scratch_bytes, e.bytes);
        l.gemm_workspace_bytes = sizes::k2_gemm_workspace(t, l.gemm_max_n);
    }
    pack.group(VkHeap::kDeviceLocal, VkPurpose::kWorkspace);
    {
        layout::Carver rows{nullptr};
        layout::carve_workspace_rows(c, t, rows, [](layout::Region, uint32_t, uint8_t *) {});
        if (!region(VkRegionKind::kWorkspaceRows, 0, rows.size(), ra)) {
            return too_big(VkRegionKind::kWorkspaceRows, rows.size());
        }
        if (!region(VkRegionKind::kScratch, 0, l.scratch_bytes, ra)) {
            return too_big(VkRegionKind::kScratch, l.scratch_bytes);
        }
        if (!region(VkRegionKind::kGemmWorkspace, 0, l.gemm_workspace_bytes, ra)) {
            return too_big(VkRegionKind::kGemmWorkspace, l.gemm_workspace_bytes);
        }
        if (drafts) {
            const uint64_t b = layout::mtp_hc_rows_bytes(c, t);
            if (!region(VkRegionKind::kMtpHcRows, 0, b, ra)) return too_big(VkRegionKind::kMtpHcRows, b);
        }
    }

    // ---- STATE group ----
    pack.group(VkHeap::kDeviceLocal, VkPurpose::kState);
    l.session_bytes = layout::session_bytes(c, ctx, drafts);
    for (uint64_t s = 0; s < plan.max_sessions; ++s) {
        if (!region(VkRegionKind::kSession, uint32_t(s), l.session_bytes, ra)) {
            return too_big(VkRegionKind::kSession, l.session_bytes);
        }
    }
    if (drafts) {
        l.undo_bytes = layout::undo_bytes(c);
        for (uint64_t u = 0; u < layout::kUndoSnapshots * plan.max_sessions; ++u) {
            if (!region(VkRegionKind::kUndo, uint32_t(u), l.undo_bytes, ra)) return too_big(VkRegionKind::kUndo, l.undo_bytes);
        }
        l.verify_rows = layout::verify_rows(plan.max_sessions, t);
        // model.cpp:536-559 carve_spec_sampler with vkk13s::scratch_bytes(3) (k13s_spec_accept_vk.cpp:40).
        const uint64_t rows = l.verify_rows, vocab = c.vocab_size, words = (vocab + 31) / 32;
        layout::Carver s{nullptr};
        s.take<uint8_t>(k.mul(k.mul(rows, vocab), 4));
        s.take<uint8_t>(k.mul(k.mul(rows, vocab), 4));
        s.take<uint8_t>(rows * words * 4);
        s.take<uint8_t>(rows * 2 * 4);
        s.take<uint8_t>(rows * 2 * 4);
        s.take<uint8_t>(rows * 4);
        s.take<uint8_t>(rows * 4);
        s.take<uint8_t>(vocab * 4);
        s.take<uint8_t>(sizes::k13s_scratch(3));
        if (!region(VkRegionKind::kSpecSampler, 0, s.size(), ra)) return too_big(VkRegionKind::kSpecSampler, s.size());
    }
    // m210 D6': the fixed-state pool, one run so a chunk-0 end state's K/V run spans consecutive slots (as CUDA's).
    if (plan.prefix_state_slots != 0) {
        const layout::FixedState f = layout::fixed_state(c, 1);
        if (!f.contiguous || f.bytes == 0) return refuse(CERE_ERR_UNSUPPORTED, "prefix cache: fixed state not contiguous");
        l.state_bytes = f.bytes;
        l.state_slots = plan.prefix_state_slots + plan.max_sessions;
        const uint64_t b = k.mul(l.state_slots, l.state_bytes);
        if (k.overflow) return refuse(CERE_ERR_CAPACITY, "state slots size overflows uint64");
        if (!region(VkRegionKind::kStateSlots, 0, b, ra)) return too_big(VkRegionKind::kStateSlots, b);
    }

    // ---- KV POOL group: layout::carve_pool's entries, one item each (256-aligned like the carver) ----
    pack.group(VkHeap::kDeviceLocal, VkPurpose::kKvPool);
    l.pool_tokens = plan.kv_pool_tokens;
    {
        const uint64_t kv_width = uint64_t(c.num_key_value_heads) * c.head_dim;
        const uint64_t kv = k.mul(l.pool_tokens, kv_width);
        const uint64_t keys = k.mul(layout::qsa_blocks(c, l.pool_tokens), uint64_t(c.indexer_head_dim) * 2);
        if (k.overflow) return refuse(CERE_ERR_CAPACITY, "KV pool size overflows uint64");
        const auto entry = [&](VkRegionKind kind, uint32_t layer, uint64_t bytes) {
            if (!region(kind, layer, bytes, layout::kAlign)) return false;
            l.pool_bytes = k.add(k.align(l.pool_bytes, layout::kAlign), bytes);
            return true;
        };
        for (uint32_t layer = 0; layer < 64; ++layer) {
            if (((c.full_attention_layers >> layer) & 1) == 0) continue;
            if (!entry(VkRegionKind::kKCache, layer, kv)) return too_big(VkRegionKind::kKCache, kv);
            if (!entry(VkRegionKind::kVCache, layer, kv)) return too_big(VkRegionKind::kVCache, kv);
            if (!entry(VkRegionKind::kQsaKeys, layer, keys)) return too_big(VkRegionKind::kQsaKeys, keys);
        }
        if (drafts) {
            if (!entry(VkRegionKind::kMtpKCache, 0, kv)) return too_big(VkRegionKind::kMtpKCache, kv);
            if (!entry(VkRegionKind::kMtpVCache, 0, kv)) return too_big(VkRegionKind::kMtpVCache, kv);
            if (!entry(VkRegionKind::kMtpQsaKeys, 0, keys)) return too_big(VkRegionKind::kMtpQsaKeys, keys);
        }
        l.pool_bytes = k.align(l.pool_bytes, layout::kAlign);
    }

    // ---- IO group: model.cpp:471-487 carve_staging restated, plus the K12 uploads, K4 block table, table ring ----
    pack.group(VkHeap::kDeviceLocal, VkPurpose::kIo);
    {
        const uint64_t vocab = c.vocab_size, words = (vocab + 31) / 32;
        const uint64_t lp_rows = std::min<uint64_t>(t, 16);  // model.cpp:566 staging_logprobs_rows
        const uint64_t mask_rows = std::max<uint64_t>(1, l.verify_rows);
        const std::pair<VkRegionKind, uint64_t> io[] = {
            {VkRegionKind::kIoIds32, t * sizeof(int32_t)},
            {VkRegionKind::kIoIds64, t * sizeof(int64_t)},
            {VkRegionKind::kIoTop2, t * sizeof(cere_token_score)},
            {VkRegionKind::kIoLogprobs, lp_rows * sizeof(cere_token_logprobs)},
            {VkRegionKind::kIoBoundFault, sizeof(uint32_t)},
            {VkRegionKind::kIoBias, vocab * sizeof(cere_logit_bias)},
            {VkRegionKind::kIoMask, mask_rows * words * sizeof(uint32_t)},
            {VkRegionKind::kIoPleBlockTable, ple.present ? uint64_t(sizes::kK4BlockTableBytes) : 0},
            {VkRegionKind::kIoTableRing, limits.table_ring_bytes},
        };
        for (const auto &[kind, bytes] : io) {
            if (bytes == 0) continue;
            if (!region(kind, 0, bytes, ra)) return too_big(kind, bytes);
        }
    }

    // ---- PLE group (host heap): table chunks, then every other HOST_PINNED run (scales as chunks) ----
    pack.group(VkHeap::kHostHeap, VkPurpose::kPle);
    if (ple.present) {
        const ArenaRun &tr = runs[size_t(table_run)];
        std::vector<uint64_t> cs, cb, co;
        std::vector<uint32_t> cblk;
        for (uint32_t b = 0; b < ple.blocks; ++b) {
            const uint64_t r0 = uint64_t(b) * ple.rows_per_block;
            const uint64_t n = std::min<uint64_t>(ple.rows_per_block, ple.rows - r0);
            uint32_t block = 0;
            uint64_t at = 0;
            if (!pack.place(n * ple.row_bytes, ra, &block, &at)) {
                return refuse(CERE_ERR_INTERNAL, "PLE: a table chunk does not fit one block");
            }
            ple.table_block.push_back(block);
            ple.table_offset.push_back(at);
            cs.push_back(r0 * ple.row_bytes);
            cb.push_back(n * ple.row_bytes);
            cblk.push_back(block);
            co.push_back(at);
        }
        const uint64_t shard_bytes = ple.shard_rows * ple.row_bytes;
        for (size_t m = 0; m < tr.members.size(); ++m) {
            stream_pieces(tr.members[m], m * shard_bytes, shard_bytes, cs, cb, cblk, co, &sink);
        }
    }
    for (size_t r = 0; r < runs.size(); ++r) {
        const ArenaRun &run = runs[r];
        if (plan.tensors[run.members[0]].placement != CERE_PLACEMENT_HOST_PINNED || int64_t(r) == table_run) continue;
        if (int64_t(r) == scale_run) {
            std::vector<uint64_t> cs, cb, co;
            std::vector<uint32_t> cblk;
            for (uint32_t b = 0; b < ple.blocks; ++b) {
                const uint64_t r0 = uint64_t(b) * ple.rows_per_block;
                const uint64_t n = std::min<uint64_t>(ple.rows_per_block, ple.rows - r0);
                uint32_t block = 0;
                uint64_t at = 0;
                // Chunk 0 region-aligned, later chunks abut (one contiguous array while they share a block).
                if (!pack.place(n * ple.scale_bytes, b == 0 ? ra : 1, &block, &at)) {
                    return refuse(CERE_ERR_INTERNAL, "PLE: a scale chunk does not fit one block");
                }
                ple.scale_block.push_back(block);
                ple.scale_offset.push_back(at);
                cs.push_back(r0 * ple.scale_bytes);
                cb.push_back(n * ple.scale_bytes);
                cblk.push_back(block);
                co.push_back(at);
            }
            const uint64_t shard_bytes = ple.shard_rows * ple.scale_bytes;
            for (size_t m = 0; m < run.members.size(); ++m) {
                stream_pieces(run.members[m], m * shard_bytes, shard_bytes, cs, cb, cblk, co, &sink);
            }
            continue;
        }
        if (!place_run(run)) {
            return refuse(CERE_ERR_UNSUPPORTED, "host: " + (run.key.empty() ? plan.tensors[run.members[0]].name : run.key) +
                                                    " is larger than one Vulkan buffer; only the PLE table may be split");
        }
    }
    if (k.overflow) return refuse(CERE_ERR_CAPACITY, "a size overflowed uint64");

    // ---- finish: block sizes (R5), totals (R6), per-tensor piece index ----
    for (VkBlock &b : l.blocks) {
        b.bytes = std::max(k.align(b.used, limits.block_granule), limits.block_granule);
        if (b.bytes > cap) return refuse(CERE_ERR_INTERNAL, "block above the limit");
        (b.heap == VkHeap::kDeviceLocal ? l.device_bytes : l.host_bytes) =
            k.add(b.heap == VkHeap::kDeviceLocal ? l.device_bytes : l.host_bytes, b.bytes);
    }
    if (k.overflow) return refuse(CERE_ERR_CAPACITY, "a size overflowed uint64");
    std::stable_sort(sink.all.begin(), sink.all.end(),
                     [](const auto &x, const auto &y) { return x.first < y.first; });
    l.piece_first.assign(plan.tensors.size() + 1, 0);
    l.pieces.reserve(sink.all.size());
    size_t at = 0;
    for (size_t i = 0; i < plan.tensors.size(); ++i) {
        l.piece_first[i] = l.pieces.size();
        while (at < sink.all.size() && sink.all[at].first == i) l.pieces.push_back(sink.all[at++].second);
    }
    l.piece_first[plan.tensors.size()] = l.pieces.size();
    return l;
}

host::MemoryEstimate vk_estimate(const VkLayout &layout) {
    host::MemoryEstimate e{};
    e.device_bytes = layout.device_bytes;
    e.host_bytes = layout.host_bytes;
    return e;
}

std::string vk_layout_description(const host::ModelPlan &plan, const VkLayout &l) {
    std::string d;
    d.reserve(plan.tensors.size() * 96 + 4096);
    const auto num = [&d](uint64_t v) { d += std::to_string(v); };
    d += kVkLayoutTag;
    d += "\nlimits ";
    num(l.limits.max_buffer_bytes);
    d += ' ';
    num(l.limits.block_granule);
    d += ' ';
    num(l.limits.region_alignment);
    d += ' ';
    num(l.limits.small_alignment);
    d += ' ';
    num(l.limits.small_tensor_bytes);
    d += ' ';
    num(l.limits.table_ring_bytes);
    d += "\nstatus ";
    d += std::to_string(l.status);
    d += '\n';
    for (size_t b = 0; b < l.blocks.size(); ++b) {
        d += "block ";
        num(b);
        d += l.blocks[b].heap == VkHeap::kDeviceLocal ? " device " : " host ";
        d += vk_purpose_name(l.blocks[b].purpose);
        d += ' ';
        num(l.blocks[b].bytes);
        d += ' ';
        num(l.blocks[b].used);
        d += '\n';
    }
    for (size_t i = 0; i < plan.tensors.size() && i + 1 < l.piece_first.size(); ++i) {
        const host::TensorPlan &t = plan.tensors[i];
        d += t.name;
        d.push_back('\0');
        num(t.dtype);
        for (uint32_t s = 0; s < t.ndim && s < CERE_TENSOR_MAX_DIMS; ++s) {
            d += ',';
            num(t.shape[s]);
        }
        for (uint64_t p = l.piece_first[i]; p < l.piece_first[i + 1]; ++p) {
            d += '@';
            num(l.pieces[p].block);
            d += ':';
            num(l.pieces[p].offset);
            d += '+';
            num(l.pieces[p].bytes);
        }
        d += '\n';
    }
    for (const VkRegion &r : l.regions) {
        d += "region ";
        d += vk_region_name(r.kind);
        d += ' ';
        num(r.index);
        d += '@';
        num(r.block);
        d += ':';
        num(r.offset);
        d += '+';
        num(r.bytes);
        d += '\n';
    }
    for (const VkWinfo &w : l.winfo) {
        d += "winfo ";
        num(uint64_t(w.consumer));
        d += ' ';
        num(w.tensor);
        d += ' ';
        num(w.n);
        d += ' ';
        num(w.k);
        d += ' ';
        num(w.offset);
        d += '\n';
    }
    if (l.ple.present) {
        d += "ple ";
        num(l.ple.format);
        d += ' ';
        num(l.ple.rows);
        d += ' ';
        num(l.ple.rows_per_block);
        d += ' ';
        num(l.ple.blocks);
        d += '\n';
    }
    d += "totals ";
    num(l.device_bytes);
    d += ' ';
    num(l.host_bytes);
    d += '\n';
    return d;
}

std::string vk_layout_digest(const host::ModelPlan &plan, const VkLayout &layout) {
    const std::string d = vk_layout_description(plan, layout);
    const std::array<uint8_t, 32> h = host::sha256(d.data(), d.size());
    static const char *hex = "0123456789abcdef";
    std::string s;
    for (uint8_t b : h) {
        s.push_back(hex[b >> 4]);
        s.push_back(hex[b & 15]);
    }
    return s;
}

std::vector<VkPiece> vk_locate(const VkLayout &l, uint64_t tensor, uint64_t offset, uint64_t len) {
    std::vector<VkPiece> out;
    if (tensor + 1 >= l.piece_first.size()) return out;
    const uint64_t first = l.piece_first[tensor], last = l.piece_first[tensor + 1];
    if (first == last) return out;
    uint64_t total = 0;
    for (uint64_t p = first; p < last; ++p) total += l.pieces[p].bytes;
    if (offset > total || len > total - offset) return out;
    const uint64_t end = offset + len;
    for (uint64_t p = first; p < last && len > 0; ++p) {
        const VkPiece &q = l.pieces[p];
        const uint64_t a = std::max(offset, q.tensor_offset), b = std::min(end, q.tensor_offset + q.bytes);
        if (a >= b) continue;
        out.push_back(VkPiece{q.block, q.offset + (a - q.tensor_offset), a, b - a});
    }
    return out;
}

const VkRegion *vk_region(const VkLayout &l, VkRegionKind kind, uint32_t index) {
    for (const VkRegion &r : l.regions) {
        if (r.kind == kind && r.index == index) return &r;
    }
    return nullptr;
}

bool vk_ple_block_table(const VkLayout &l, const uint64_t *block_addr, uint64_t out[128]) {
    for (int i = 0; i < 128; ++i) out[i] = 0;
    if (!l.ple.present || l.ple.blocks > sizes::kK4MaxBlocks) return false;
    for (uint32_t b = 0; b < l.ple.blocks; ++b) {
        out[b] = block_addr[l.ple.table_block[b]] + l.ple.table_offset[b];
        if (l.ple.scale_bytes != 0) out[sizes::kK4MaxBlocks + b] = block_addr[l.ple.scale_block[b]] + l.ple.scale_offset[b];
    }
    return true;
}

}  // namespace cere::vkdev
