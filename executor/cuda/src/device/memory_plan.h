// Memory plan of one A4 model on the CUDA device (m22 design note "Memory plan", Q12): one device arena and one
// mapped pinned host block per model, plus the M2.11 snapshot arena (below). Pure arithmetic (no CUDA call), shared by
// Device::estimate and Device::allocate so the estimate the host checks against its budgets is exactly what ALLOCATE
// takes. The region sizes and the tensor order come from the model interface (model/model.h), which carves the same
// regions at bind time.
//
// Device arena, in order:
//   1. every DEVICE tensor, in model::arena_order (the weights: 77,843,711,744 B for the A4 artifact without MTP;
//      expert arrays, q|k|v and other arena_order runs packed without padding);
//   2. the workspace (model::workspace_bytes for model::workspace_rows(max_context_tokens) rows; Q12 "<= 4 GiB";
//      O14: no allocation after readiness); K2's GEMM plans run with zero cuBLASLt workspace except the pinned
//      split-K configurations of moe_shared_gate_up and attn_qkv, whose partials live in the workspace's one
//      gemm_workspace region (the larger k2_gemm.h gemm_pins_workspace_bytes: at most 4,259,840 B, attn_qkv at m 32);
//   3. the derived tensors (model::derived_bytes: F32 A_log, MoE alphas, RoPE table), filled at INSTANTIATE;
//   4. max_sessions session slots of model::session_bytes(max_context_tokens) each (slot map, recurrent state, conv
//      states, QSA ring, PLE history);
//   5. the paged KV pool (model::pool_bytes(plan.kv_pool_tokens), m25 design note D2/D3: FP8 KV and the QSA
//      compressed keys of every full-attention layer, shared by the sessions in 64-token pages);
//   6. M2.10 (m210 design note D6′): with plan.prefix_state_slots > 0, the fixed-state pool of
//      prefix_state_slots + max_sessions state slots of model::fixed_state_bytes each (59,031,808 B on A4: the
//      prefix checkpoints and one decode track slot per session); absent with 0 (the layout is then M2.5's).
//   7. (vision plans) the image buffers (m213 design note D12);
//   8. (plans with speculative_draft_tokens > 0 only; m29 design note M29S2-19/20) the draft-only region:
//      max_sessions x 3 undo snapshots of layout::undo_bytes each (model::UndoState), then the verify rows' sampler
//      stages (model::spec_sampler_bytes for layout::verify_rows rows; phase S, D6). A spec-off plan has no such
//      region (A1).
//   9. (MXFP8 artifacts only; track Fp4S3, fp4local design note FL-30) the MXFP8 dequantize staging of
//      ModelWeights::mxfp8 (model::bind_mxfp8): the largest MXFP8 GEMM's BF16 upcast, mxfp8::plan_of(plan)
//      .staging_bytes (68,157,440 B on the S3 artifact: the packed q|k|v, 13,312 x 2,560). Only the GEMMs on the
//      forward's stream (and the shared expert's fork stream, while the forward's stream runs no MXFP8 GEMM) write
//      it. A BF16 artifact's plan has no such region (byte for byte the plan without it).
// Mapped host block (cudaHostAllocMapped): every HOST_PINNED tensor in model::arena_order (the PLE table shards packed
// in shard order, scale and index: 51,200,246,042 B), then the workspace's staging (model::staging_bytes: token ids
// in, top-2 out). DISCARD tensors take no storage.
// Snapshot arena (M2.11, plan.snapshot_host_bytes; m211 design note D5): not in the mapped block but its own
// page-aligned, mlock'd allocation (host/locked_memory.h), never device-mapped nor CUDA-registered, so it takes no GPU
// memory. The host estimate (pinned_bytes, R1) counts the mapped block plus the arena.
#pragma once

#include "host/device.h"

#include <cstdint>
#include <vector>

namespace cere::device {

// Offset value of a DISCARD tensor.
constexpr uint64_t kNoStorage = UINT64_MAX;
// Regions and tensors of at least 4 KiB start on 256-byte boundaries (cuBLASLt and vectorized loads).
constexpr uint64_t kRegionAlignment = 256;

struct ModelLayout {
    std::vector<uint64_t> offsets;  // per tensor: offset in the device arena (DEVICE) or mapped block (HOST_PINNED)
    uint64_t weight_bytes;          // end of the DEVICE tensors in the device arena
    uint64_t workspace_rows;        // rows of the largest forward call (Workspace::max_tokens)
    uint64_t workspace_offset;      // device arena
    uint64_t workspace_bytes;
    uint64_t derived_offset;        // device arena
    uint64_t derived_bytes;
    uint64_t sessions_offset;       // device arena: slot i at sessions_offset + i * session_bytes
    uint64_t session_bytes;         // one slot, a multiple of 256
    uint64_t max_sessions;
    uint64_t pool_offset;           // device arena: the paged KV pool (model::bind_pool)
    uint64_t pool_bytes;
    uint64_t pool_tokens;           // plan.kv_pool_tokens, a multiple of the 64-token page
    uint64_t state_pool_offset;     // device arena: state slot i at state_pool_offset + i * state_bytes (M2.10 D6′)
    uint64_t state_bytes;           // one state slot: model::fixed_state_bytes, a multiple of 256
    uint64_t state_slots;           // plan.prefix_state_slots + max_sessions, or 0 when the cache is off
    // M2.9: the draft token map entries of a plan with drafts (0 = no drafts). Then the workspace has the MTP growth
    // (model::workspace_bytes' draft_count), the derived region ends with the MTP derived tensors
    // (model::mtp::derived_bytes), slots and pool carry the MTP entries (session_bytes / pool_bytes with draft).
    uint64_t draft_count;
    uint64_t staging_offset;        // mapped block
    uint64_t staging_bytes;
    uint64_t region_bytes;          // mapped block: end of the HOST_PINNED tensors (the shared region, cuda_device.h)
    // m213 design note D12: only when the plan's support entry has a vision config, else 0 (and no region). A
    // vision plan's workspace is also at least model::vision_scratch_bytes (the encoder's activations alias it).
    // Sizes per m213l design note VL-L2 (model::layout::kMaxImageTokens = 4,096 per image).
    uint64_t image_embed_offset;  // device arena: BF16 [4,096, out_hidden_size], a call's merged rows at call rows
    uint64_t image_embed_bytes;   // also the size of image_out
    uint64_t image_out_offset;    // device arena: BF16 [4,096, out_hidden_size], one image's encoder output
    uint64_t image_patch_offset;  // device arena: the device copy of the pixel patches
    uint64_t image_patch_bytes;   // one copy; the pinned host copy has the same size
    uint64_t image_staging_offset;  // host arena: the pinned copy of the pixel patches
    uint64_t image_positions_offset;  // device arena: int64 [3, 2 workspace_rows + 3] (layout::image_positions_bytes)
    uint64_t image_positions_bytes;   // one copy; the pinned host copy has the same size
    uint64_t image_positions_staging_offset;  // host arena
    // m29 design note M29S2-19/20: only when plan.speculative_draft_tokens > 0, else 0 (and no region).
    uint64_t undo_offset;     // device arena: slot i's undo[j] at undo_offset + (3 i + j) * undo_bytes
    uint64_t undo_bytes;      // one snapshot (layout::undo_bytes), a multiple of 256
    uint64_t verify_rows;     // rows of one verify call (layout::verify_rows)
    uint64_t spec_sampler_offset;  // device arena: model::SpecSampler of verify_rows rows
    uint64_t spec_sampler_bytes;   // model::spec_sampler_bytes
    uint64_t mxfp8_staging_offset;  // device arena: region 9 (0 bytes: none)
    uint64_t mxfp8_staging_bytes;
    // MixedPD (mixed-pd design note): only when plan.prefill_park, else 0 (and no region): the carried rows of a
    // paused prefill call (model::Park: hyper then normed, [workspace_rows, 10,240] BF16 each), last.
    uint64_t park_offset;  // device arena
    uint64_t park_bytes;
    uint64_t snapshot_bytes;        // M2.11 snapshot arena, its own allocation (snapshot_host_bytes)
    uint64_t device_bytes;  // device arena size
    uint64_t host_bytes;    // mapped block size
    uint64_t pinned_bytes;  // host estimate: the mapped block plus the snapshot arena
    bool overflow;  // a size overflowed uint64; device_bytes, host_bytes and pinned_bytes are then UINT64_MAX
};

// The host side of the plan: `mapped_end` is the end of the mapped block's regions (HOST_PINNED tensors, staging).
struct HostLayout {
    uint64_t mapped_bytes;    // the mapped block: mapped_end aligned to kRegionAlignment; the arena is not in it
    uint64_t snapshot_bytes;  // the snapshot arena (its own allocation)
    uint64_t pinned_bytes;    // mapped_bytes + snapshot_bytes aligned to kRegionAlignment (the host estimate)
    bool overflow;            // a size overflowed uint64; the sizes are then meaningless
};

inline HostLayout host_layout(uint64_t mapped_end, uint64_t snapshot_bytes) {
    HostLayout h{};
    uint64_t end = 0;
    h.overflow |= __builtin_add_overflow(mapped_end, kRegionAlignment - 1, &end);
    h.mapped_bytes = end / kRegionAlignment * kRegionAlignment;
    h.snapshot_bytes = snapshot_bytes;
    h.overflow |= __builtin_add_overflow(h.mapped_bytes, snapshot_bytes, &end);
    h.overflow |= __builtin_add_overflow(end, kRegionAlignment - 1, &end);
    h.pinned_bytes = end / kRegionAlignment * kRegionAlignment;
    return h;
}

// The host side when the HOST_PINNED region is shared (cuda_device.h): the private mapped block is the tail of the
// mapped block from `staging_offset` on (the staging and, with vision, the image staging; `region_bytes` of the shared
// segment are not in it), so a private offset `off` is at `off - staging_offset` in it.
inline HostLayout shared_host_layout(const ModelLayout &l) {
    return host_layout(l.host_bytes - l.staging_offset, l.snapshot_bytes);
}

// May throw std::bad_alloc (the callers map it to CERE_ERR_CAPACITY).
ModelLayout plan_layout(const host::ModelPlan &plan);

#if defined(CERE_MOE_P3B)
// The DEVICE weight bytes of `plan` placed in `order` / `packed` (a model::ArenaOrder; plan_layout's placement rule;
// UINT64_MAX on overflow): the P3b arena-size ktest compares model::arena_order with model::arena_order_k7
// (m23 design note M23-D2).
uint64_t weight_bytes(const host::ModelPlan &plan, const std::vector<uint64_t> &order,
                      const std::vector<bool> &packed);
#endif

}  // namespace cere::device
