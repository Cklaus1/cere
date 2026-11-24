// WeightLab intervention kernel-test entry points (weightlab design note WL-05..08;
// executor/tests/gpu/test_kernels_research_patch.py, suite `kernels_research`). Device pointers, int64 sizes, the stream
// last; 0 = success, else a cere_status. Row ranges are host int64 triples {begin, end, src} (research::RowRange).
#include "kernels/k7_moe.h"
#include "ktest/k2_ktest.h"
#include "model/model.h"
#include "research/patch.h"

#if defined(CERE_MOE_P3B)
#include "p3b/p3b_moe.h"
#endif

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>

namespace cere::ktest {
namespace {

namespace research = cere::research;

bool u32(int64_t v) { return v > 0 && v <= std::numeric_limits<int32_t>::max(); }
bool u32_or_zero(int64_t v) { return v >= 0 && v <= std::numeric_limits<int32_t>::max(); }

// The RowSet of `count` host triples; false past kMaxRanges or a field outside uint32 (apply_* checks the rest).
bool row_set(const int64_t *ranges, int64_t count, research::RowSet *out) {
    if (ranges == nullptr || count < 1 || count > int64_t(research::kMaxRanges)) return false;
    for (int64_t j = 0; j < count; ++j) {
        for (int f = 0; f < 3; ++f) {
            if (!u32_or_zero(ranges[3 * j + f])) return false;
        }
        out->range[j] = {uint32_t(ranges[3 * j]), uint32_t(ranges[3 * j + 1]), uint32_t(ranges[3 * j + 2])};
    }
    out->count = uint32_t(count);
    return true;
}

// The post-route hook of the block entry points. mode 1: record only; 2: overwrite every row's routing with
// `set_ids` / `set_weights` ([m, top_k], device) by a stream-ordered copy, then record; 3: research::apply_route
// (expert_force) on `rows`, then record. Record: the routing the experts run, into ids_out / weights_out.
struct RouteHook {
    int64_t mode = 0;
    const int32_t *set_ids = nullptr;
    const float *set_weights = nullptr;
    research::Edit force;
    research::RowSet rows;
    int32_t *ids_out = nullptr;
    float *weights_out = nullptr;
    uint32_t calls = 0;
};

cere_status route_hook(void *ctx, int32_t *ids, float *weights, uint32_t m, uint32_t top_k, cudaStream_t stream) {
    RouteHook &h = *static_cast<RouteHook *>(ctx);
    ++h.calls;
    const size_t n = size_t(m) * top_k;
    if (h.mode == 2 && (cudaMemcpyAsync(ids, h.set_ids, n * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream) !=
                            cudaSuccess ||
                        cudaMemcpyAsync(weights, h.set_weights, n * sizeof(float), cudaMemcpyDeviceToDevice,
                                        stream) != cudaSuccess)) {
        return CERE_ERR_INTERNAL;
    }
    cere_status s;
    if (h.mode == 3 && (s = research::apply_route(h.force, ids, weights, m, top_k, h.rows, stream)) != CERE_OK) {
        return s;
    }
    if ((h.ids_out != nullptr &&
         cudaMemcpyAsync(h.ids_out, ids, n * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) ||
        (h.weights_out != nullptr &&
         cudaMemcpyAsync(h.weights_out, weights, n * sizeof(float), cudaMemcpyDeviceToDevice, stream) != cudaSuccess)) {
        return CERE_ERR_INTERNAL;
    }
    return CERE_OK;
}

// The hook arguments of the block entry points (mode 0: no taps at all, the base call).
bool route_args(int64_t mode, const void *set_ids, const void *set_weights, const void *force_experts,
                const void *force_weights, int64_t force_count, const int64_t *ranges, int64_t range_count,
                void *ids_out, void *weights_out, RouteHook *h) {
    if (mode < 0 || mode > 3 || (mode == 2 && (set_ids == nullptr || set_weights == nullptr))) return false;
    if (mode == 3) {
        if (!u32(force_count) || !row_set(ranges, range_count, &h->rows)) return false;
        h->force.kind = research::Kind::kExpertForce;
        h->force.experts = static_cast<const int32_t *>(force_experts);
        h->force.weights = static_cast<const float *>(force_weights);
        h->force.count = uint32_t(force_count);
    }
    h->mode = mode;
    h->set_ids = static_cast<const int32_t *>(set_ids);
    h->set_weights = static_cast<const float *>(set_weights);
    h->ids_out = static_cast<int32_t *>(ids_out);
    h->weights_out = static_cast<float *>(weights_out);
    return true;
}

// One block call with the hook (backend 0: kernels::moe_block, K7 layout; 1: p3b::moe_block, P3b layout, p3b build
// only), plans pinned at max_m = m (fork 1: the plans carry a model::MoeFork and the block must take it, as
// k7_ktest.cpp's). `routed` (nullable): MoeTaps.routed. The hook must run exactly once when set.
cere_status hooked_block(int backend, const kernels::MoeDims &d, const kernels::MoeLayer &L, const void *x, int64_t m,
                         void *out, void *routed, RouteHook *hook, bool fork, cudaStream_t stream) {
    cublasLtHandle_t handle = gemm_handle();
    if (handle == nullptr) return CERE_ERR_INTERNAL;
    const uint32_t mm = uint32_t(m);
    kernels::GemmPlan router;
    kernels::GemmPlan shared_gate_up;
    kernels::GemmPlan shared_down;
    void *gemm_workspace = nullptr;
    cere_status s;
    if ((s = router.init(handle, d.num_experts, d.hidden, mm, kernels::GemmOut::kBF16)) != CERE_OK ||
        (s = plan_init(&shared_gate_up, handle, d.shared_inter, d.hidden, mm, kernels::GemmOut::kBF16,
                       &gemm_workspace, stream)) != CERE_OK) {
        return s;
    }
    s = shared_down.init(handle, d.hidden, d.shared_inter, mm, kernels::GemmOut::kBF16);
#if defined(CERE_MOE_P3B)
    const size_t bytes = backend == 1 ? p3b::moe_workspace_bytes(d, mm) : kernels::moe_workspace_bytes(d, mm);
#else
    const size_t bytes = kernels::moe_workspace_bytes(d, mm);
#endif
    void *workspace = nullptr;
    if (s == CERE_OK && bytes == 0) s = CERE_ERR_ARGUMENT;
    if (s == CERE_OK && cudaMallocAsync(&workspace, bytes, stream) != cudaSuccess) s = CERE_ERR_CAPACITY;
    model::MoeFork moe_fork;
    if (s == CERE_OK && fork) s = moe_fork.create();
    if (s == CERE_OK) {
        const kernels::MoePlans plans = moe_fork.plans(&router, &shared_gate_up, &shared_down);
        kernels::MoeTaps taps;
        taps.routed = static_cast<__nv_bfloat16 *>(routed);
        if (hook->mode != 0) {
            taps.route = route_hook;
            taps.ctx = hook;
        }
        const kernels::MoeTaps *t = hook->mode != 0 || routed != nullptr ? &taps : nullptr;
        const auto *xb = static_cast<const __nv_bfloat16 *>(x);
        auto *ob = static_cast<__nv_bfloat16 *>(out);
        if (fork && !kernels::moe_shared_forks(plans, mm)) {
            s = CERE_ERR_ARGUMENT;
        } else if (backend == 0) {
            s = kernels::moe_block(d, L, plans, xb, mm, ob, workspace, bytes, stream, t);
        } else {
#if defined(CERE_MOE_P3B)
            s = p3b::moe_block(d, L, plans, xb, mm, ob, workspace, bytes, stream, t);
#else
            s = CERE_ERR_UNSUPPORTED;
#endif
        }
        if (s == CERE_OK && hook->mode != 0 && hook->calls != 1) s = CERE_ERR_INTERNAL;
        if (cudaFreeAsync(workspace, stream) != cudaSuccess && s == CERE_OK) s = CERE_ERR_INTERNAL;
    }
    if (gemm_workspace != nullptr && cudaFreeAsync(gemm_workspace, stream) != cudaSuccess && s == CERE_OK) {
        s = CERE_ERR_INTERNAL;
    }
    return s;
}

}  // namespace
}  // namespace cere::ktest

// research::check_edit (host arrays `experts` int32 / `weights` FP32, nullable); the stream is unused.
extern "C" __attribute__((visibility("default"))) int cere_ktest_research_check(
    int64_t kind, int64_t router, int64_t point_cols, int64_t top_k, float alpha, int64_t col_begin, int64_t col_end,
    int64_t vector_rows, int64_t vector_cols, const void *experts, const void *weights, int64_t count, cudaStream_t) {
    using cere::ktest::u32_or_zero;
    if (!u32_or_zero(kind) || router < 0 || router > 1 || !u32_or_zero(point_cols) || !u32_or_zero(top_k) ||
        !u32_or_zero(col_begin) || !u32_or_zero(col_end) || !u32_or_zero(vector_rows) || !u32_or_zero(vector_cols) ||
        !u32_or_zero(count)) {
        return CERE_ERR_ARGUMENT;
    }
    return cere::research::check_edit(static_cast<cere::research::Kind>(kind), router == 1, uint32_t(point_cols),
                                      uint32_t(top_k), alpha, uint32_t(col_begin), uint32_t(col_end),
                                      uint32_t(vector_rows), uint32_t(vector_cols),
                                      static_cast<const int32_t *>(experts), static_cast<const float *>(weights),
                                      uint32_t(count));
}

// research::apply_point on `values` BF16 [n, cols] in place: kinds 1-6, `vector` BF16 [vector_rows, cols],
// `experts` int32 / `weights` FP32 [count] (device), `ranges` host int64 [range_count, 3].
extern "C" __attribute__((visibility("default"))) int cere_ktest_research_point(
    int64_t kind, float alpha, int64_t col_begin, int64_t col_end, const void *vector, int64_t vector_rows,
    const void *experts, const void *weights, int64_t count, void *values, int64_t n, int64_t cols,
    const int64_t *ranges, int64_t range_count, cudaStream_t stream) {
    using cere::ktest::u32;
    using cere::ktest::u32_or_zero;
    namespace research = cere::research;
    research::RowSet rows;
    if (!u32_or_zero(kind) || !u32_or_zero(col_begin) || !u32_or_zero(col_end) || !u32_or_zero(vector_rows) ||
        !u32_or_zero(count) || !u32(n) || !u32(cols) || !cere::ktest::row_set(ranges, range_count, &rows)) {
        return CERE_ERR_ARGUMENT;
    }
    research::Edit e;
    e.kind = static_cast<research::Kind>(kind);
    e.alpha = alpha;
    e.col_begin = uint32_t(col_begin);
    e.col_end = uint32_t(col_end);
    e.vector = static_cast<const __nv_bfloat16 *>(vector);
    e.vector_rows = uint32_t(vector_rows);
    e.experts = static_cast<const int32_t *>(experts);
    e.weights = static_cast<const float *>(weights);
    e.count = uint32_t(count);
    return research::apply_point(e, static_cast<__nv_bfloat16 *>(values), uint32_t(n), uint32_t(cols), rows, stream);
}

// research::apply_route (expert_force) on the routing ids int32 / weights FP32 [n, top_k] in place.
extern "C" __attribute__((visibility("default"))) int cere_ktest_research_route(
    const void *experts, const void *weights, int64_t count, void *topk_ids, void *topk_weights, int64_t n,
    int64_t top_k, const int64_t *ranges, int64_t range_count, cudaStream_t stream) {
    using cere::ktest::u32;
    namespace research = cere::research;
    research::RowSet rows;
    if (!u32(count) || !u32(n) || !u32(top_k) || !cere::ktest::row_set(ranges, range_count, &rows)) {
        return CERE_ERR_ARGUMENT;
    }
    research::Edit e;
    e.kind = research::Kind::kExpertForce;
    e.experts = static_cast<const int32_t *>(experts);
    e.weights = static_cast<const float *>(weights);
    e.count = uint32_t(count);
    return research::apply_route(e, static_cast<int32_t *>(topk_ids), static_cast<float *>(topk_weights), uint32_t(n),
                                 uint32_t(top_k), rows, stream);
}

// The K7 MoE block (kernels::moe_block; weights as cere_ktest_moe_block's) with the post-route hook (WL-08). `mode`
// 0: no taps (the base call; `routed` must be null); 1: a hook that only records the routing; 2: the hook overwrites
// every row's routing with `set_ids` int32 / `set_weights` FP32 [m, top_k]; 3: expert_force (`force_experts` int32 /
// `force_weights` FP32 [force_count], device) on the rows of `ranges` (host int64 [range_count, 3]). Modes 1-3 copy
// the routing the experts run into `ids_out` / `weights_out` (nullable). `routed` (nullable, BF16 [m, H]):
// MoeTaps.routed. `fork` as cere_ktest_moe_block_fork's.
extern "C" __attribute__((visibility("default"))) int cere_ktest_research_moe_block(
    const void *x, const void *router_weight, const void *gate_weight, const void *gate_scale, const void *up_weight,
    const void *up_scale, const void *down_weight, const void *down_scale, const void *g1_alphas,
    const void *g2_alphas, float a1_global_scale, float a2_global_scale, const void *shared_gate_weight,
    const void *shared_up_weight, const void *shared_down_weight, const void *shared_expert_gate, void *out,
    void *routed, void *ids_out, void *weights_out, int64_t mode, const void *set_ids, const void *set_weights,
    const void *force_experts, const void *force_weights, int64_t force_count, const int64_t *ranges,
    int64_t range_count, int64_t fork, int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k, int64_t inter,
    int64_t shared_inter, cudaStream_t stream) {
    using cere::ktest::u32;
    cere::ktest::RouteHook hook;
    if (!u32(m) || !u32(hidden) || !u32(num_experts) || !u32(top_k) || !u32(inter) || !u32(shared_inter) ||
        fork < 0 || fork > 1 || (mode == 0 && routed != nullptr) ||
        !cere::ktest::route_args(mode, set_ids, set_weights, force_experts, force_weights, force_count, ranges,
                                 range_count, ids_out, weights_out, &hook)) {
        return CERE_ERR_ARGUMENT;
    }
    cere::kernels::MoeDims d;
    d.hidden = uint32_t(hidden);
    d.num_experts = uint32_t(num_experts);
    d.top_k = uint32_t(top_k);
    d.inter = uint32_t(inter);
    d.shared_inter = uint32_t(shared_inter);
    cere::kernels::MoeLayer L;
    L.router = static_cast<const __nv_bfloat16 *>(router_weight);
    L.gate_weight = static_cast<const uint8_t *>(gate_weight);
    L.gate_scale = static_cast<const uint8_t *>(gate_scale);
    L.up_weight = static_cast<const uint8_t *>(up_weight);
    L.up_scale = static_cast<const uint8_t *>(up_scale);
    L.down_weight = static_cast<const uint8_t *>(down_weight);
    L.down_scale = static_cast<const uint8_t *>(down_scale);
    L.g1_alphas = static_cast<const float *>(g1_alphas);
    L.g2_alphas = static_cast<const float *>(g2_alphas);
    L.a1_global_scale = a1_global_scale;
    L.a2_global_scale = a2_global_scale;
    L.shared_gate = static_cast<const __nv_bfloat16 *>(shared_gate_weight);
    L.shared_up = static_cast<const __nv_bfloat16 *>(shared_up_weight);
    L.shared_down = static_cast<const __nv_bfloat16 *>(shared_down_weight);
    L.shared_expert_gate = static_cast<const __nv_bfloat16 *>(shared_expert_gate);
    return cere::ktest::hooked_block(0, d, L, x, m, out, routed, &hook, fork == 1, stream);
}

// The P3b MoE block (p3b::moe_block, P3b arena layout as cere_ktest_p3b_moe_block's; m's pinned tactic pair) with the
// post-route hook, `mode` / `routed` / `ids_out` / `weights_out` / `fork` as cere_ktest_research_moe_block's.
// CERE_ERR_UNSUPPORTED in a k7 build.
extern "C" __attribute__((visibility("default"))) int cere_ktest_research_p3b_moe_block(
    const void *x, const void *router_weight, const void *w13_weight, const void *w13_scale, const void *down_weight,
    const void *down_scale, const void *g1_alphas, const void *g2_alphas, const void *act_scales,
    const void *shared_gate_weight, const void *shared_up_weight, const void *shared_down_weight,
    const void *shared_expert_gate, void *out, void *routed, void *ids_out, void *weights_out, int64_t mode,
    const void *set_ids, const void *set_weights, const void *force_experts, const void *force_weights,
    int64_t force_count, const int64_t *ranges, int64_t range_count, int64_t fork, int64_t m, int64_t hidden,
    int64_t num_experts, int64_t top_k, int64_t inter, int64_t shared_inter, cudaStream_t stream) {
    using cere::ktest::u32;
    cere::ktest::RouteHook hook;
    if (!u32(m) || !u32(hidden) || !u32(num_experts) || !u32(top_k) || !u32(inter) || !u32(shared_inter) ||
        fork < 0 || fork > 1 || (mode == 0 && routed != nullptr) ||
        !cere::ktest::route_args(mode, set_ids, set_weights, force_experts, force_weights, force_count, ranges,
                                 range_count, ids_out, weights_out, &hook)) {
        return CERE_ERR_ARGUMENT;
    }
    cere::kernels::MoeDims d;
    d.hidden = uint32_t(hidden);
    d.num_experts = uint32_t(num_experts);
    d.top_k = uint32_t(top_k);
    d.inter = uint32_t(inter);
    d.shared_inter = uint32_t(shared_inter);
    cere::kernels::MoeLayer L;
    L.router = static_cast<const __nv_bfloat16 *>(router_weight);
    L.up_weight = static_cast<const uint8_t *>(w13_weight);
    L.up_scale = static_cast<const uint8_t *>(w13_scale);
    L.gate_weight = L.up_weight == nullptr ? nullptr : L.up_weight + size_t(inter) * size_t(hidden / 2);
    L.gate_scale = L.up_scale == nullptr ? nullptr : L.up_scale + size_t(inter) * size_t(hidden / 16);
    L.down_weight = static_cast<const uint8_t *>(down_weight);
    L.down_scale = static_cast<const uint8_t *>(down_scale);
    L.g1_alphas = static_cast<const float *>(g1_alphas);
    L.g2_alphas = static_cast<const float *>(g2_alphas);
    L.act_scales = static_cast<const float *>(act_scales);
    L.shared_gate = static_cast<const __nv_bfloat16 *>(shared_gate_weight);
    L.shared_up = static_cast<const __nv_bfloat16 *>(shared_up_weight);
    L.shared_down = static_cast<const __nv_bfloat16 *>(shared_down_weight);
    L.shared_expert_gate = static_cast<const __nv_bfloat16 *>(shared_expert_gate);
    return cere::ktest::hooked_block(1, d, L, x, m, out, routed, &hook, fork == 1, stream);
}
