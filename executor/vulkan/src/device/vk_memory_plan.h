// VulkanAmd P3 T2 (vulkan-amd-p3-design design note section 3, task T2): the memory plan of one model on the Vulkan device.
// Pure host arithmetic (no Vulkan call, no allocation of model bytes), shared by VulkanDevice::estimate and ::allocate,
// so the estimate the host checks against its budgets is exactly what ALLOCATE takes (sum of the blocks).
//
// Why not CUDA's plan (executor/cuda/src/device/memory_plan.{h,cpp} plan_layout): every Vulkan buffer and allocation is
// <= 4 GiB - 4 on RADV (spec G0 consequence 1), so there is no contiguous device arena and no single mapped host block.
// The model becomes a list of BLOCKS (one VkBuffer + one VkDeviceMemory each), each <= VkLimits::max_buffer_bytes,
// rounded up to VkLimits::block_granule. Every byte a kernel touches is (block, offset): its buffer device address is
// block.addr + offset.
//
// Layout rules (deterministic: the same plan and limits give the same layout, byte for byte):
//   R1 Groups, in this block order: WEIGHTS (device-local: DEVICE tensors in arena order, then the derived tensors),
//      WORKSPACE, STATE (session slots, undo snapshots, spec sampler), KV POOL, IO (all device-local), then PLE (host
//      heap: the split PLE table, then every other HOST_PINNED tensor). A group never shares a block with another.
//   R2 Inside a group, items are placed next-fit in order: an item goes at the aligned cursor of the group's current
//      block when it fits there, else it opens a new block. No item is ever split, except the PLE table (R4).
//      Next-fit (not first-fit-decreasing) keeps the CUDA arena order, so a family the CUDA layout keeps together
//      (a layer's expert runs, q|k|v, ...) stays together and uploads stream contiguously; the slack it leaves at a
//      block end costs no memory because a block is sized to its content, not to the limit.
//   R3 Weight items are the CUDA arena runs (vk_arena_order.h, the K7 build's model::arena_order): member 0 aligned to
//      region_alignment (256 B) when the tensor is >= small_tensor_bytes (4 KiB), else small_alignment (16 B); later
//      members abut their predecessor (packed, so [E] expert arrays and q|k|v stay one array). A run larger than one
//      block is refused (CERE_ERR_UNSUPPORTED): kernels take one base address per run.
//   R4 The PLE table (run `layers.L.ple.shards@host`, HOST_PINNED) is the only split: in row chunks of rows_per_block
//      rows (vkk4::SplitTable geometry, k4_ple_vk.h:8-16: block b holds global rows [b rpb, (b+1) rpb)). rpb is the
//      largest whole number of shards whose bytes fit one block and whose in-block byte offsets stay < 2^32; only when
//      one shard is larger than a block does rpb fall back to the largest row count (shards then straddle blocks, and
//      vk_locate splits an upload write). Each table chunk is a 256-aligned item; its scales run (NVFP4) is split into
//      the same row chunks, which abut (one contiguous [rows, 10] array while they share a block). The K4 block table
//      (vk_ple_block_table) then holds table_base[b] = table chunk b's address, scale_base[b] = scale chunk b's.
//   R5 Every block's bytes = its content end rounded up to block_granule, and <= max_buffer_bytes.
//   R6 Accounting (OD-3): device-local blocks are device bytes, host-heap blocks host bytes; estimate == the sums.
//      VK_EXT_memory_budget is NOT read here; it would plug in at VulkanDevice::allocate, comparing each heap's
//      budget with the per-heap block sums of this layout before the first vkAllocateMemory (design 3.3 item 9).
//
// First-light scope (spec P3, design header): text only, max_sessions <= VkLimits::max_sessions (1), 2,048 <
// max_context_tokens <= VkLimits::max_context_tokens (32,768; OD-5 refuses <= 2,048), no snapshot arena, no vision, no
// MixedPD park; MTP drafts only as 0 or 4 speculative_draft_tokens. The prefix cache (D15: prefix_state_slots > 0) on
// a model without drafts only, its state slots in one block. Outside it: CERE_ERR_UNSUPPORTED with a message, never a
// silent drop.
#pragma once

#include "cere_executor.h"
#include "host/device.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cere::vkdev {

// The prefix of every layout description: a Vulkan layout can never be taken for a CUDA one (whose host_share tag is
// "cuda-sm120-v1", cuda_device.cpp:623) or the toy's ("toy-v1").
inline constexpr char kVkLayoutTag[] = "cere-vk-layout-v1";

// Offset of a tensor without storage (DISCARD).
inline constexpr uint64_t kVkNoStorage = UINT64_MAX;

enum class VkHeap : uint32_t {
    kDeviceLocal = 0,  // the launch layer's default Context::alloc: device-local + host-visible (RADV/UMA heap 1)
    kHostHeap = 1,     // host-visible, not device-local (heap 0, P0-5 "path B"): the PLE table
};

enum class VkPurpose : uint32_t {
    kWeights = 0,    // DEVICE tensors (arena order), then derived (A_log, alphas, RoPE, MTP hot lm_head, K2/K7 info)
    kWorkspace = 1,  // row buffers, shared kernel scratch, K2 GEMM workspace, MTP hc rows
    kState = 2,      // session slots, undo snapshots, spec sampler stages
    kKvPool = 3,     // paged FP8 K/V + QSA compressed keys per full-attention layer (+ the MTP entry)
    kIo = 4,         // host-written / host-read buffers: ids in, top-2 / logprobs out, bound fault, K12 uploads,
                     // the K4 block table, the descriptor-table ring
    kPle = 5,        // host heap: the split PLE table, its scales and the other HOST_PINNED tensors
};

// Limits and policy knobs (AGENTS.md 13: typed, not literals in the code).
struct VkLimits {
    uint64_t max_buffer_bytes = 4294967292ull;  // min(maxBufferSize, maxMemoryAllocationSize): 4 GiB - 4 on RADV (P0-5)
    uint64_t block_granule = 4096;              // block sizes are multiples (vkAllocateMemory page granularity)
    uint64_t region_alignment = 256;            // regions, workspaces, tensors >= small_tensor_bytes (CUDA's rule)
    uint64_t small_alignment = 16;              // tensors < small_tensor_bytes (the shaders' 16-byte vector loads)
    uint64_t small_tensor_bytes = 4096;
    uint64_t table_ring_bytes = uint64_t(1) << 20;  // descriptor-table ring in the IO block (design 4.2)
    // First-light scope.
    uint64_t max_sessions = 1;
    uint64_t min_context_exclusive = 2048;  // OD-5: max_context_tokens <= this (k6::full_attention plans) refused
    uint64_t max_context_tokens = 32768;
    uint32_t max_ple_blocks = 64;  // vkk4::kMaxBlocks (k4_ple_vk.h:60)
};

struct VkBlock {
    VkHeap heap;
    VkPurpose purpose;
    uint64_t bytes;  // allocation size: used rounded up to block_granule, <= max_buffer_bytes
    uint64_t used;   // end of the last item placed
};

// One contiguous piece of a tensor: bytes [tensor_offset, tensor_offset + bytes) of the tensor live at
// blocks[block] + offset. A tensor has one piece unless it is a PLE shard (or scale shard) straddling a chunk border.
struct VkPiece {
    uint32_t block;
    uint64_t offset;
    uint64_t tensor_offset;
    uint64_t bytes;
};

enum class VkRegionKind : uint32_t {
    // WEIGHTS group (after the DEVICE tensors), written at INSTANTIATE
    kDerived = 0,     // layout::carve_derived: per layer [F32 A_log], F32 g1/g2 alphas [E]; F32 RoPE [ctx, rotary]
    kMtpDerived,      // layout::carve_mtp_derived: hot lm_head BF16 [count, hidden], map u32 [count] (drafts only)
    kMoeInfo,         // vkk7/vkk13 MoeLayer::bf16_info, one stride per layer (index = layer; MTP at num_hidden_layers)
    kWinfo,           // K2 row info of the K5 / K4 / forward GEMM weights (VkLayout::winfo; K7/K13 use kMoeInfo)
    // WORKSPACE group
    kWorkspaceRows,   // layout::carve_workspace_rows for workspace_rows rows
    kScratch,         // the shared kernel scratch: max of VkLayout::workspace entries
    kGemmWorkspace,   // vkk2::gemm_workspace_bytes(workspace_rows, max GEMM n) for the forward's own vkk2::run calls
    kMtpHcRows,       // layout::mtp_hc_rows_bytes (drafts only)
    // STATE group
    kSession,         // layout::carve_session, index = slot
    kUndo,            // layout::carve_undo, index = 3 slot + j (drafts only)
    kSpecSampler,     // model::SpecSampler stages for verify_rows rows (drafts only)
    kStateSlots,      // prefix cache (D15): state_slots x state_bytes fixed-state slots, one run (prefix models only)
    // KV POOL group (layout::carve_pool's entries, index = layer)
    kKCache,
    kVCache,
    kQsaKeys,
    kMtpKCache,
    kMtpVCache,
    kMtpQsaKeys,
    // IO group
    kIoIds32,
    kIoIds64,
    kIoTop2,
    kIoLogprobs,
    kIoBoundFault,
    kIoBias,
    kIoMask,
    kIoPleBlockTable,  // the K4 SplitTable block table (vk_ple_block_table), static from ALLOCATE on
    kIoTableRing,
};

struct VkRegion {
    VkRegionKind kind;
    uint32_t index;
    uint32_t block;
    uint64_t offset;
    uint64_t bytes;
};

// One entry of the workspace table (the shared scratch is the max of the kScratch entries).
struct VkWorkspaceEntry {
    std::string name;
    std::string source;  // the Vulkan (or CUDA) function the bytes come from, restated in vk_kernel_sizes.h
    bool may_be_stale;   // sizes::kMayBeStale
    uint64_t bytes;
};

// Who reads a prepared K2 row-info entry (vkk2::gemm_prepare_weight, n uint32 words, 16-byte aligned parts).
enum class VkInfoConsumer : uint32_t {
    kK5Layer = 0,  // vkk5 LayerWeights::info of one GDN layer: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
                   // entries abut in that order (k5_gdn_layer_vk.cpp info_parts): the layer's info = the qkv entry
    kK4Ple = 1,    // vkk4 Weights::proj_info: key_proj then value_proj (k4_ple_vk.cpp:72 kValueInfoOff)
    kForward = 2,  // PROPOSED (T5): the forward's own vkk2::run calls: attention q|k|v (one packed GEMM), o_proj,
                   // indexer index_qk_proj per full-attention layer; MTP fc_embedding, fc_hidden, q|k|v, o, index_qk
};

// One prepared GEMM weight.
struct VkWinfo {
    uint64_t tensor;  // first tensor of the GEMM (q of a packed q|k|v run)
    VkInfoConsumer consumer;
    uint32_t n;       // rows (q|k|v: the run's summed rows)
    uint32_t k;
    uint64_t offset;  // inside the kWinfo region, 16-byte aligned
};

// The PLE split (R4), in vkk4::SplitTable geometry.
struct VkPleSplit {
    bool present = false;
    uint32_t format = 0;      // vkk4::TableFormat: 0 FP8 ([rows, 160], no scales), 1 NVFP4 ([rows, 80] + [rows, 10])
    uint64_t rows = 0;        // global rows = shards x shard_rows (>= the K4 view's table_rows)
    uint64_t shard_rows = 0;
    uint32_t shards = 0;
    uint32_t row_bytes = 0;
    uint32_t scale_bytes = 0;  // 0 for FP8
    uint32_t rows_per_block = 0;
    uint32_t blocks = 0;            // table chunks = SplitTable view.blocks
    uint32_t shards_per_block = 0;  // 0 when rows_per_block is not a whole number of shards (row fallback)
    std::vector<uint32_t> table_block;   // per chunk b: block index and offset of row b * rows_per_block
    std::vector<uint64_t> table_offset;
    std::vector<uint32_t> scale_block;   // per chunk b (NVFP4 only): block and offset of its first scale row
    std::vector<uint64_t> scale_offset;
};

struct VkLayout {
    cere_status status = CERE_OK;
    std::string error;  // why, when status != CERE_OK
    VkLimits limits;
    std::vector<VkBlock> blocks;
    // Per tensor of plan.tensors: pieces [piece_first[i], piece_first[i + 1]) of `pieces` (none for DISCARD).
    std::vector<uint64_t> piece_first;
    std::vector<VkPiece> pieces;
    std::vector<VkRegion> regions;
    std::vector<VkWinfo> winfo;
    std::vector<VkWorkspaceEntry> workspace;  // the scratch candidates, in table order
    VkPleSplit ple;
    // Sizes the device binds with (as CUDA's ModelLayout).
    uint64_t workspace_rows = 0;
    uint64_t scratch_bytes = 0;
    uint64_t gemm_workspace_bytes = 0;
    uint64_t gemm_max_n = 0;
    uint64_t draft_count = 0;  // draft token map entries (0 = no drafts)
    uint64_t verify_rows = 0;  // drafts only
    uint64_t session_bytes = 0;
    uint64_t undo_bytes = 0;
    // Prefix cache (m210 design note D6'): state slot i at kStateSlots + i * state_bytes (layout::fixed_state,
    // a 256-B multiple); state_slots = prefix_state_slots + max_sessions, 0 when the cache is off.
    uint64_t state_bytes = 0;
    uint64_t state_slots = 0;
    uint64_t pool_tokens = 0;
    uint64_t pool_bytes = 0;          // sum of the KV entries (equal to layout::pool_bytes when they share one block)
    uint64_t moe_info_stride = 0;
    // Tensor byte sums (no padding) and the totals.
    uint64_t device_tensor_bytes = 0;
    uint64_t host_tensor_bytes = 0;
    uint64_t device_bytes = 0;  // sum of device-local blocks
    uint64_t host_bytes = 0;    // sum of host-heap blocks
};

// The plan. status CERE_OK, or (nothing else valid in the layout):
//   CERE_ERR_UNSUPPORTED  outside the first-light scope; a dtype the Vulkan kernels do not take; an MXFP8 group the
//                         binder would refuse (model::mxfp8::plan_of); a non-PLE tensor or run larger than one block;
//                         a PLE table whose split cannot land on a row boundary or needs > max_ple_blocks chunks, or
//                         whose shards are not uniform / not numbered 0..N-1 / not of K4's row geometry
//   CERE_ERR_CAPACITY     a size overflowed uint64; a workspace, state, KV or IO region larger than one block
//   CERE_ERR_ARGUMENT     limits that cannot hold one granule, or a tensor whose bytes disagree with its shape
// May throw std::bad_alloc (the caller maps it to CERE_ERR_CAPACITY, as CUDA).
VkLayout vk_plan_layout(const host::ModelPlan &plan, const VkLimits &limits = VkLimits{});

// The host's capacity check (Device::estimate): exactly the block sums of `layout`.
host::MemoryEstimate vk_estimate(const VkLayout &layout);

// The layout description (identity / hash input; the Vulkan counterpart of host::region_description): kVkLayoutTag,
// the limits, every block, every tensor's pieces (name, dtype, shape), every region, the PLE split. Byte-stable.
std::string vk_layout_description(const host::ModelPlan &plan, const VkLayout &layout);
// Lower-case hex SHA-256 of vk_layout_description.
std::string vk_layout_digest(const host::ModelPlan &plan, const VkLayout &layout);

// The pieces of tensor `tensor`'s bytes [offset, offset + len) (Device::write: one memcpy per piece into the owning
// block's mapping). Empty when the range is outside the tensor or the tensor has no storage.
std::vector<VkPiece> vk_locate(const VkLayout &layout, uint64_t tensor, uint64_t offset, uint64_t len);

// The first region of `kind` with `index`, or null.
const VkRegion *vk_region(const VkLayout &layout, VkRegionKind kind, uint32_t index = 0);

// The K4 block table (vkk4::kBlockTableBytes = 1 KiB: u64 table_base[64], then u64 scale_base[64]) for blocks whose
// buffer device addresses are block_addr[0 .. blocks.size()). Unused entries are 0. False when the layout has no PLE
// table. The matching vkk4::TableView is {desc = address of the table, rows = the K4 table rows (<= ple.rows),
// rows_per_block = ple.rows_per_block, blocks = ple.blocks}.
bool vk_ple_block_table(const VkLayout &layout, const uint64_t *block_addr, uint64_t out[128]);

const char *vk_purpose_name(VkPurpose p);
const char *vk_region_name(VkRegionKind k);

}  // namespace cere::vkdev
