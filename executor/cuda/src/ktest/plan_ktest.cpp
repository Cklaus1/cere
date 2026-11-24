// Memory-plan kernel-test entry point (executor/tests/gpu/test_kernels_k2_gemm.py DraftRegionTest): device::plan_layout
// of a weight-less A4 text plan (host arithmetic, no CUDA call). The DEVICE weights are placed first and do not depend
// on the plan's drafts, so the non-weight regions carry every spec-on/spec-off difference of this layout
// (m29 design note M29S2-19, A1). 0 = success, else a cere_status.
#include "device/memory_plan.h"
#include "host/support.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <new>

namespace {

int layout(int64_t max_sessions, int64_t max_context_tokens, int64_t kv_pool_tokens, int64_t drafts,
           int64_t prefix_state_slots, uint64_t *out, bool slots) {
    if (out == nullptr || max_sessions < 1 || max_context_tokens < 1 || kv_pool_tokens < 64 ||
        kv_pool_tokens % 64 != 0 || drafts < 0 || drafts > UINT32_MAX || prefix_state_slots < 0) {
        return CERE_ERR_ARGUMENT;
    }
    try {
        cere::host::ModelPlan plan{};
        plan.support = &cere::host::a4_support();
        plan.config = plan.support->config;
        plan.max_sessions = uint64_t(max_sessions);
        plan.max_context_tokens = uint64_t(max_context_tokens);
        plan.kv_pool_tokens = uint64_t(kv_pool_tokens);
        plan.speculative_draft_tokens = uint32_t(drafts);
        plan.prefix_state_slots = uint64_t(prefix_state_slots);
        const cere::device::ModelLayout l = cere::device::plan_layout(plan);
        if (l.overflow) return CERE_ERR_CAPACITY;
        out[0] = l.device_bytes;
        out[1] = l.pinned_bytes;
        out[2] = l.pool_offset + l.pool_bytes;
        out[3] = l.undo_offset;
        out[4] = l.undo_bytes;
        out[5] = l.workspace_rows;
        out[6] = l.spec_sampler_offset;
        out[7] = l.spec_sampler_bytes;
        out[8] = l.verify_rows;
        if (slots) {
            out[9] = l.state_bytes;
            out[10] = l.state_slots;
        }
        return CERE_OK;
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
}

}  // namespace

// out[0] device_bytes, out[1] pinned_bytes, out[2] pool_offset + pool_bytes (the end of the text plan's last region
// before M2.9), out[3] undo_offset, out[4] undo_bytes (0 = no draft-only region), out[5] workspace_rows,
// out[6] spec_sampler_offset, out[7] spec_sampler_bytes (phase S; 0 without drafts), out[8] verify_rows.
extern "C" __attribute__((visibility("default"))) int cere_ktest_plan_layout(int64_t max_sessions,
                                                                             int64_t max_context_tokens,
                                                                             int64_t kv_pool_tokens, int64_t drafts,
                                                                             uint64_t *out, cudaStream_t) {
    return layout(max_sessions, max_context_tokens, kv_pool_tokens, drafts, 0, out, false);
}

// batch-concurrency design note BC0 (c): cere_ktest_plan_layout with the plan's prefix_state_slots (M2.10 D6′), so a
// profile's whole arena (its prefix state pool included) is measured per R; out[0..8] as above, out[9] state_bytes,
// out[10] state_slots (prefix_state_slots + max_sessions, 0 with the cache off).
extern "C" __attribute__((visibility("default"))) int cere_ktest_plan_layout_slots(
    int64_t max_sessions, int64_t max_context_tokens, int64_t kv_pool_tokens, int64_t drafts,
    int64_t prefix_state_slots, uint64_t *out, cudaStream_t) {
    return layout(max_sessions, max_context_tokens, kv_pool_tokens, drafts, prefix_state_slots, out, true);
}
