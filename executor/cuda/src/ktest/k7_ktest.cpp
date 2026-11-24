// K7 kernel-test entry points (executor/tests/gpu/test_kernels_k7_moe.py, suite `kernels_k7`): the MoE router and
// the whole MoE block. Device pointers, int64 sizes, the stream last; 0 = success, else a cere_status. The block's
// K2 plans are pinned for this call (same algorithms as the executor's plans of the same shapes) and its workspace is
// stream-ordered memory released before return.
#include "kernels/k7_moe.h"
#include "ktest/k2_ktest.h"
#include "model/model.h"

#include <limits>

namespace cere::ktest {
namespace {

bool u32(int64_t v) { return v > 0 && v <= std::numeric_limits<int32_t>::max(); }

// rows 0: the block's plans pinned at max_m = m (cere_ktest_moe_block). rows > 0 (m <= rows): the executor's rows
// MoePlans (m26 design note D6): GemmPlan::init_rows(rows) of each instantiated plan (max_m 1: its bucket-0
// algorithm is the executor plan's). `routed` (nullable): MoeTaps.routed, the block's tap path. `fork`: the plans
// carry a model::MoeFork and the block must take it (m23 design note P11R-23), else CERE_ERR_ARGUMENT.
cere_status moe_block(const kernels::MoeDims &dims, const kernels::MoeLayer &layer, const void *x, int64_t m,
                      int64_t rows, void *out, void *routed, bool fork, cudaStream_t stream) {
    cublasLtHandle_t handle = gemm_handle();
    if (handle == nullptr) {
        return CERE_ERR_INTERNAL;
    }
    const uint32_t mm = static_cast<uint32_t>(m);
    const uint32_t max_m = rows == 0 ? mm : 1;
    kernels::GemmPlan router;
    kernels::GemmPlan shared_gate_up;
    kernels::GemmPlan shared_down;
    void *gemm_workspace = nullptr;
    cere_status s;
    if ((s = router.init(handle, dims.num_experts, dims.hidden, max_m, kernels::GemmOut::kBF16)) != CERE_OK ||
        (s = plan_init(&shared_gate_up, handle, dims.shared_inter, dims.hidden, max_m, kernels::GemmOut::kBF16,
                       &gemm_workspace, stream)) != CERE_OK) {
        return s;
    }
    const size_t bytes = kernels::moe_workspace_bytes(dims, mm);
    void *workspace = nullptr;
    s = shared_down.init(handle, dims.hidden, dims.shared_inter, max_m, kernels::GemmOut::kBF16);
    kernels::GemmPlan rows_router;
    kernels::GemmPlan rows_gate_up;
    kernels::GemmPlan rows_down;
    if (s == CERE_OK && rows > 0) {
        const uint32_t r = static_cast<uint32_t>(rows);
        if ((s = rows_router.init_rows(handle, router, r)) == CERE_OK &&
            (s = rows_gate_up.init_rows(handle, shared_gate_up, r)) == CERE_OK) {
            s = rows_down.init_rows(handle, shared_down, r);
        }
    }
    if (s == CERE_OK && bytes == 0) {
        s = CERE_ERR_ARGUMENT;
    }
    if (s == CERE_OK && cudaMallocAsync(&workspace, bytes, stream) != cudaSuccess) {
        s = CERE_ERR_CAPACITY;
    }
    model::MoeFork moe_fork;
    if (s == CERE_OK && fork) {
        s = moe_fork.create();
    }
    if (s == CERE_OK) {
        const kernels::MoePlans plans = rows == 0 ? moe_fork.plans(&router, &shared_gate_up, &shared_down)
                                                  : moe_fork.plans(&rows_router, &rows_gate_up, &rows_down);
        kernels::MoeTaps taps;
        taps.routed = static_cast<__nv_bfloat16 *>(routed);
        if (fork && !kernels::moe_shared_forks(plans, mm)) {
            s = CERE_ERR_ARGUMENT;
        } else {
            s = kernels::moe_block(dims, layer, plans, static_cast<const __nv_bfloat16 *>(x), mm,
                                   static_cast<__nv_bfloat16 *>(out), workspace, bytes, stream,
                                   routed == nullptr ? nullptr : &taps);
        }
        if (cudaFreeAsync(workspace, stream) != cudaSuccess && s == CERE_OK) {
            s = CERE_ERR_INTERNAL;
        }
    }
    // The shared gate/up plan's split-K workspace (k2_ktest.h plan_init), after its last run.
    if (gemm_workspace != nullptr && cudaFreeAsync(gemm_workspace, stream) != cudaSuccess && s == CERE_OK) {
        s = CERE_ERR_INTERNAL;
    }
    return s;
}

// The MoeLayer / MoeDims of the cere_ktest_moe_block arguments; false for a dimension outside [1, INT32_MAX].
bool block_args(const void *router_weight, const void *gate_weight, const void *gate_scale, const void *up_weight,
                const void *up_scale, const void *down_weight, const void *down_scale, const void *g1_alphas,
                const void *g2_alphas, float a1_global_scale, float a2_global_scale, const void *shared_gate_weight,
                const void *shared_up_weight, const void *shared_down_weight, const void *shared_expert_gate,
                int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter, int64_t shared_inter,
                kernels::MoeDims *dims, kernels::MoeLayer *layer) {
    if (!u32(m) || !u32(hidden) || !u32(num_experts) || !u32(top_k) || !u32(inter) || !u32(shared_inter)) {
        return false;
    }
    dims->hidden = static_cast<uint32_t>(hidden);
    dims->num_experts = static_cast<uint32_t>(num_experts);
    dims->top_k = static_cast<uint32_t>(top_k);
    dims->inter = static_cast<uint32_t>(inter);
    dims->shared_inter = static_cast<uint32_t>(shared_inter);
    layer->router = static_cast<const __nv_bfloat16 *>(router_weight);
    layer->gate_weight = static_cast<const uint8_t *>(gate_weight);
    layer->gate_scale = static_cast<const uint8_t *>(gate_scale);
    layer->up_weight = static_cast<const uint8_t *>(up_weight);
    layer->up_scale = static_cast<const uint8_t *>(up_scale);
    layer->down_weight = static_cast<const uint8_t *>(down_weight);
    layer->down_scale = static_cast<const uint8_t *>(down_scale);
    layer->g1_alphas = static_cast<const float *>(g1_alphas);
    layer->g2_alphas = static_cast<const float *>(g2_alphas);
    layer->a1_global_scale = a1_global_scale;
    layer->a2_global_scale = a2_global_scale;
    layer->shared_gate = static_cast<const __nv_bfloat16 *>(shared_gate_weight);
    layer->shared_up = static_cast<const __nv_bfloat16 *>(shared_up_weight);
    layer->shared_down = static_cast<const __nv_bfloat16 *>(shared_down_weight);
    layer->shared_expert_gate = static_cast<const __nv_bfloat16 *>(shared_expert_gate);
    return true;
}

}  // namespace
}  // namespace cere::ktest

extern "C" __attribute__((visibility("default"))) int cere_ktest_moe_router(const void *logits, void *topk_ids,
                                                                            void *topk_weights, int64_t m,
                                                                            int64_t num_experts, int64_t top_k,
                                                                            cudaStream_t stream) {
    if (!cere::ktest::u32(m) || !cere::ktest::u32(num_experts) || !cere::ktest::u32(top_k)) {
        return CERE_ERR_ARGUMENT;
    }
    return cere::kernels::moe_route(static_cast<const __nv_bfloat16 *>(logits), static_cast<int32_t *>(topk_ids),
                                    static_cast<float *>(topk_weights), static_cast<uint32_t>(m),
                                    static_cast<uint32_t>(num_experts), static_cast<uint32_t>(top_k), stream);
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_moe_block(
    const void *x, const void *router_weight, const void *gate_weight, const void *gate_scale, const void *up_weight,
    const void *up_scale, const void *down_weight, const void *down_scale, const void *g1_alphas,
    const void *g2_alphas, float a1_global_scale, float a2_global_scale, const void *shared_gate_weight,
    const void *shared_up_weight, const void *shared_down_weight, const void *shared_expert_gate, void *out,
    int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter, int64_t shared_inter,
    cudaStream_t stream) {
    cere::kernels::MoeDims dims;
    cere::kernels::MoeLayer layer;
    if (!cere::ktest::block_args(router_weight, gate_weight, gate_scale, up_weight, up_scale, down_weight, down_scale,
                                 g1_alphas, g2_alphas, a1_global_scale, a2_global_scale, shared_gate_weight,
                                 shared_up_weight, shared_down_weight, shared_expert_gate, m, hidden, num_experts,
                                 top_k, inter, shared_inter, &dims, &layer)) {
        return CERE_ERR_ARGUMENT;
    }
    return cere::ktest::moe_block(dims, layer, x, m, 0, out, nullptr, false, stream);
}

// m26 design note D6 (K2 test_moe_rows_invariant): cere_ktest_moe_block with the executor's rows MoePlans
// (rows in [1, GemmPlan::kMaxRows], m <= rows), or with rows 0 the plans of cere_ktest_moe_block; `routed`
// (nullable, BF16 [m, hidden]) receives MoeTaps.routed (the block then runs its tap path).
extern "C" __attribute__((visibility("default"))) int cere_ktest_moe_block_rows(
    const void *x, const void *router_weight, const void *gate_weight, const void *gate_scale, const void *up_weight,
    const void *up_scale, const void *down_weight, const void *down_scale, const void *g1_alphas,
    const void *g2_alphas, float a1_global_scale, float a2_global_scale, const void *shared_gate_weight,
    const void *shared_up_weight, const void *shared_down_weight, const void *shared_expert_gate, void *out,
    void *routed, int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter, int64_t shared_inter,
    int64_t rows, cudaStream_t stream) {
    cere::kernels::MoeDims dims;
    cere::kernels::MoeLayer layer;
    if (rows < 0 || rows > cere::kernels::GemmPlan::kMaxRows || (rows > 0 && m > rows) ||
        !cere::ktest::block_args(router_weight, gate_weight, gate_scale, up_weight, up_scale, down_weight, down_scale,
                                 g1_alphas, g2_alphas, a1_global_scale, a2_global_scale, shared_gate_weight,
                                 shared_up_weight, shared_down_weight, shared_expert_gate, m, hidden, num_experts,
                                 top_k, inter, shared_inter, &dims, &layer)) {
        return CERE_ERR_ARGUMENT;
    }
    return cere::ktest::moe_block(dims, layer, x, m, rows, out, routed, false, stream);
}

// The shared-expert fork (m23 design note P11R-23; test_kernels_k7_moe.py test_shared_fork_*):
// cere_ktest_moe_block_rows with `fork` 1: the plans carry a fork (model::MoeFork) and the block must take it
// (kernels::moe_shared_forks: m <= 16, else CERE_ERR_ARGUMENT); `fork` 0: cere_ktest_moe_block_rows.
extern "C" __attribute__((visibility("default"))) int cere_ktest_moe_block_fork(
    const void *x, const void *router_weight, const void *gate_weight, const void *gate_scale, const void *up_weight,
    const void *up_scale, const void *down_weight, const void *down_scale, const void *g1_alphas,
    const void *g2_alphas, float a1_global_scale, float a2_global_scale, const void *shared_gate_weight,
    const void *shared_up_weight, const void *shared_down_weight, const void *shared_expert_gate, void *out,
    void *routed, int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter, int64_t shared_inter,
    int64_t rows, int64_t fork, cudaStream_t stream) {
    cere::kernels::MoeDims dims;
    cere::kernels::MoeLayer layer;
    if (rows < 0 || rows > cere::kernels::GemmPlan::kMaxRows || (rows > 0 && m > rows) || fork < 0 || fork > 1 ||
        !cere::ktest::block_args(router_weight, gate_weight, gate_scale, up_weight, up_scale, down_weight, down_scale,
                                 g1_alphas, g2_alphas, a1_global_scale, a2_global_scale, shared_gate_weight,
                                 shared_up_weight, shared_down_weight, shared_expert_gate, m, hidden, num_experts,
                                 top_k, inter, shared_inter, &dims, &layer)) {
        return CERE_ERR_ARGUMENT;
    }
    return cere::ktest::moe_block(dims, layer, x, m, rows, out, routed, fork == 1, stream);
}
