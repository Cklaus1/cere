#include "device/memory_plan.h"

#include "host/support.h"
#include "model/layout.h"
#include "model/model.h"
#include "model/mtp.h"
#include "model/mxfp8_plan.h"

namespace cere::device {

namespace {

// Regions and tensors of at least 4 KiB start on kRegionAlignment (memory_plan.h) boundaries; smaller tensors
// (norms, per-layer scalars) on 16-byte boundaries. Members of an arena_order run after the first abut their
// predecessor (no padding), so ~300,000 expert scale scalars form [E] arrays.
constexpr uint64_t kSmallAlignment = 16;
constexpr uint64_t kSmallTensorBytes = 4096;

// Saturating arithmetic: any overflow sets `overflow` and the result is meaningless (the layout is then refused).
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
    // Places `bytes` at the end of `*cursor` aligned to `a`; returns the offset.
    uint64_t place(uint64_t *cursor, uint64_t bytes, uint64_t a) {
        const uint64_t at = align(*cursor, a);
        *cursor = add(at, bytes);
        return at;
    }
};

}  // namespace

#if defined(CERE_MOE_P3B)
uint64_t weight_bytes(const host::ModelPlan &plan, const std::vector<uint64_t> &order,
                      const std::vector<bool> &packed) {
    Calc k;
    uint64_t device = 0;
    for (size_t r = 0; r < order.size(); ++r) {
        const host::TensorPlan &t = plan.tensors[order[r]];
        const uint64_t a = packed[r] ? 1 : (t.bytes < kSmallTensorBytes ? kSmallAlignment : kRegionAlignment);
        if (t.placement == CERE_PLACEMENT_DEVICE) k.place(&device, t.bytes, a);
    }
    return k.overflow ? UINT64_MAX : device;
}
#endif

ModelLayout plan_layout(const host::ModelPlan &plan) {
    Calc k;
    ModelLayout l{};
    l.offsets.assign(plan.tensors.size(), kNoStorage);
    uint64_t device = 0;
    uint64_t host = 0;
    const model::ArenaOrder order = model::arena_order(plan);
    for (size_t r = 0; r < order.order.size(); ++r) {
        const uint64_t i = order.order[r];
        const host::TensorPlan &t = plan.tensors[i];
        const uint64_t a = order.packed[r] ? 1 : (t.bytes < kSmallTensorBytes ? kSmallAlignment : kRegionAlignment);
        if (t.placement == CERE_PLACEMENT_DEVICE) {
            l.offsets[i] = k.place(&device, t.bytes, a);
        } else if (t.placement == CERE_PLACEMENT_HOST_PINNED) {
            l.offsets[i] = k.place(&host, t.bytes, a);
        }
    }
    l.weight_bytes = device;
    const cere_qwen4_exp_config &c = plan.config;
    // M2.9 (m29 design note D3, M29S2-04/-12): a plan with drafts adds the MTP workspace growth, the MTP derived
    // tensors after the target's, the MTP QSA ring and proposals per slot and the MTP pool entry; draft_count 0 is
    // the plan without drafts, byte for byte.
    l.draft_count = plan.speculative_draft_tokens == 0 ? 0 : plan.draft_token_map.size();
    l.workspace_rows = model::workspace_rows(plan.max_context_tokens);
    l.workspace_bytes = model::workspace_bytes(c, l.workspace_rows, plan.max_context_tokens, l.draft_count);
    const cere_qwen4_exp_vision_config *vision = plan.support == nullptr ? nullptr : plan.support->vision;
    if (vision != nullptr) {
        l.workspace_bytes = std::max(l.workspace_bytes, model::vision_scratch_bytes(*vision, plan.max_context_tokens));
    }
    l.workspace_offset = k.place(&device, l.workspace_bytes, kRegionAlignment);
    l.derived_bytes = model::derived_bytes(c, plan.max_context_tokens) +
                      (l.draft_count == 0 ? 0 : model::mtp::derived_bytes(c, l.draft_count));
    l.derived_offset = k.place(&device, l.derived_bytes, kRegionAlignment);
    l.session_bytes = model::session_bytes(c, plan.max_context_tokens, l.draft_count != 0);
    l.max_sessions = plan.max_sessions;
    l.sessions_offset = k.place(&device, k.mul(l.session_bytes, l.max_sessions), kRegionAlignment);
    l.pool_tokens = plan.kv_pool_tokens;
    l.pool_bytes = model::pool_bytes(c, l.pool_tokens, l.draft_count != 0);
    l.pool_offset = k.place(&device, l.pool_bytes, kRegionAlignment);
    // m210 design note D6′: the fixed-state pool after the KV pool, only on a prefix-cache model (slots 0: none).
    l.state_bytes = model::fixed_state_bytes(c);
    l.state_slots = plan.prefix_state_slots == 0 ? 0 : k.add(plan.prefix_state_slots, l.max_sessions);
    l.state_pool_offset = k.place(&device, k.mul(l.state_bytes, l.state_slots), kRegionAlignment);
    l.region_bytes = host;
    l.staging_bytes = model::staging_bytes(c, l.workspace_rows);
    l.staging_offset = k.place(&host, l.staging_bytes, kRegionAlignment);
    // m213 design note D12 (C11): the image buffers, only with a vision config (text-only plans are unchanged).
    if (vision != nullptr) {
        const cere_qwen4_exp_vision_config &v = *vision;
        l.image_embed_bytes = model::layout::image_embed_bytes(v);
        l.image_embed_offset = k.place(&device, l.image_embed_bytes, kRegionAlignment);
        l.image_out_offset = k.place(&device, l.image_embed_bytes, kRegionAlignment);
        l.image_patch_bytes = model::layout::image_patch_bytes(v);
        l.image_patch_offset = k.place(&device, l.image_patch_bytes, kRegionAlignment);
        l.image_staging_offset = k.place(&host, l.image_patch_bytes, kRegionAlignment);
        l.image_positions_bytes = model::layout::image_positions_bytes(l.workspace_rows);
        l.image_positions_offset = k.place(&device, l.image_positions_bytes, kRegionAlignment);
        l.image_positions_staging_offset = k.place(&host, l.image_positions_bytes, kRegionAlignment);
    }
    // m29 design note M29S2-19/20: the draft-only region, only with drafts (spec-off plans are unchanged, A1): the
    // undo snapshots, then (phase S, D6) the verify rows' sampler stages (model::SpecSampler).
    if (plan.speculative_draft_tokens > 0) {
        l.undo_bytes = model::layout::undo_bytes(c);
        l.undo_offset = k.place(&device, k.mul(k.mul(l.undo_bytes, model::layout::kUndoSnapshots), l.max_sessions),
                                kRegionAlignment);
        l.verify_rows = model::layout::verify_rows(l.max_sessions, l.workspace_rows);
        l.spec_sampler_bytes = model::spec_sampler_bytes(c, l.verify_rows);
        l.spec_sampler_offset = k.place(&device, l.spec_sampler_bytes, kRegionAlignment);
    }
    // Track Fp4S3 (fp4local design note FL-30): the MXFP8 dequantize staging, last and only with MXFP8 weights (a
    // BF16 artifact's layout is unchanged). Sized from the plan's names and shapes, as bind_weights binds them.
    l.mxfp8_staging_bytes = model::mxfp8::plan_of(plan).staging_bytes;
    if (l.mxfp8_staging_bytes != 0) {
        l.mxfp8_staging_offset = k.place(&device, l.mxfp8_staging_bytes, kRegionAlignment);
    }
    // MixedPD (mixed-pd design note): the park region of a layer-sliced plan, after everything else (a plan
    // without it keeps its layout byte for byte).
    if (plan.prefill_park) {
        l.park_bytes = model::layout::park_bytes(c, l.workspace_rows);
        l.park_offset = k.place(&device, l.park_bytes, kRegionAlignment);
    }
    const HostLayout h = host_layout(host, plan.snapshot_host_bytes);
    l.snapshot_bytes = h.snapshot_bytes;
    l.device_bytes = k.align(device, kRegionAlignment);
    l.host_bytes = h.mapped_bytes;
    l.pinned_bytes = h.pinned_bytes;
    l.overflow = k.overflow || h.overflow;
    if (l.overflow) {
        l.device_bytes = UINT64_MAX;
        l.host_bytes = UINT64_MAX;
        l.pinned_bytes = UINT64_MAX;
    }
    return l;
}

}  // namespace cere::device
