// K13 kernel-test entry points (executor/tests/gpu/test_kernels_k13_moe_bf16.py, suite `kernels_k13`): the BF16
// MoE block of the MTP layer. Device pointers, int64 sizes, the stream last; 0 = success, else a cere_status. The
// block's K2 plans are pinned for the call (same algorithms as the executor's plans of the same shapes) and its
// workspace is stream-ordered memory released before return.
#include "kernels/k13_moe_bf16.h"
#include "ktest/k2_ktest.h"
#include "ktest/l2_flush.h"

#include <limits>

namespace cere::ktest {
namespace {

bool u32(int64_t v) { return v > 0 && v <= std::numeric_limits<int32_t>::max(); }

// The MoeBf16Layer / MoeDims of the entry-point arguments; false for a dimension outside [1, INT32_MAX] or rows
// outside [0, GemmPlan::kMaxRows] (rows > 0 needs m <= rows).
bool block_args(const void *router_weight, const void *gate_up_weight, const void *down_weight,
                const void *shared_gate_weight, const void *shared_up_weight, const void *shared_down_weight,
                const void *shared_expert_gate, int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k,
                int64_t inter, int64_t shared_inter, int64_t rows, kernels::MoeDims *dims,
                kernels::MoeBf16Layer *layer) {
    if (!u32(m) || !u32(hidden) || !u32(num_experts) || !u32(top_k) || !u32(inter) || !u32(shared_inter) ||
        rows < 0 || rows > kernels::GemmPlan::kMaxRows || (rows > 0 && m > rows)) {
        return false;
    }
    dims->hidden = static_cast<uint32_t>(hidden);
    dims->num_experts = static_cast<uint32_t>(num_experts);
    dims->top_k = static_cast<uint32_t>(top_k);
    dims->inter = static_cast<uint32_t>(inter);
    dims->shared_inter = static_cast<uint32_t>(shared_inter);
    layer->router = static_cast<const __nv_bfloat16 *>(router_weight);
    layer->gate_up = static_cast<const __nv_bfloat16 *>(gate_up_weight);
    layer->down = static_cast<const __nv_bfloat16 *>(down_weight);
    layer->shared_gate = static_cast<const __nv_bfloat16 *>(shared_gate_weight);
    layer->shared_up = static_cast<const __nv_bfloat16 *>(shared_up_weight);
    layer->shared_down = static_cast<const __nv_bfloat16 *>(shared_down_weight);
    layer->shared_expert_gate = static_cast<const __nv_bfloat16 *>(shared_expert_gate);
    return true;
}

// Runs the block `iterations` times on the same input. rows 0: the plans pinned at max_m = m; rows > 0 (m <= rows):
// the executor's rows MoePlans (m26 design note D6): GemmPlan::init_rows(rows) of each plan made at max_m 1 (its
// bucket-0 algorithm is the executor plan's). `first`: the block's routed rows [first, m) (moe_bf16_block).
// `elapsed_ms` (nullable, device float32 [1]): the mean wall time of one call over the iterations (CUDA events,
// after one untimed warm-up call). `cold` (nullable, host double [2]; elapsed_ms null): instead, one untimed call then
// `iterations` cold-L2 calls (l2_flush.h cold_us): the median and minimum µs of one call.
cere_status moe_bf16(const kernels::MoeDims &dims, const kernels::MoeBf16Layer &layer, const void *x, int64_t m,
                     int64_t rows, int64_t first, void *out, int64_t iterations, float *elapsed_ms, double *cold,
                     cudaStream_t stream) {
    if (first < 0 || first >= m || (cold != nullptr && elapsed_ms != nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
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
    const size_t bytes = kernels::moe_bf16_workspace_bytes(dims, mm);
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
    if (s == CERE_OK) {
        const kernels::MoePlans plans = rows == 0 ? kernels::MoePlans{&router, &shared_gate_up, &shared_down}
                                                  : kernels::MoePlans{&rows_router, &rows_gate_up, &rows_down};
        const auto run = [&] {
            return kernels::moe_bf16_block(dims, layer, plans, static_cast<const __nv_bfloat16 *>(x), mm,
                                           static_cast<uint32_t>(first), static_cast<__nv_bfloat16 *>(out),
                                           workspace, bytes, stream);
        };
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        if (elapsed_ms != nullptr) {
            s = run();  // warm-up
            if (s == CERE_OK && (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess ||
                                 cudaEventRecord(start, stream) != cudaSuccess)) {
                s = CERE_ERR_INTERNAL;
            }
        }
        if (cold != nullptr) {
            void *flush = nullptr;
            s = cudaMallocAsync(&flush, kL2FlushBytes, stream) == cudaSuccess
                    ? cold_us(flush, iterations, cold, stream, run)
                    : CERE_ERR_CAPACITY;
            if (flush != nullptr && cudaFreeAsync(flush, stream) != cudaSuccess && s == CERE_OK) {
                s = CERE_ERR_INTERNAL;
            }
        }
        for (int64_t i = 0; s == CERE_OK && cold == nullptr && i < iterations; ++i) {
            s = run();
        }
        if (s == CERE_OK && elapsed_ms != nullptr) {
            float total = 0.0f;
            if (cudaEventRecord(stop, stream) != cudaSuccess || cudaEventSynchronize(stop) != cudaSuccess ||
                cudaEventElapsedTime(&total, start, stop) != cudaSuccess) {
                s = CERE_ERR_INTERNAL;
            } else {
                const float mean = total / static_cast<float>(iterations);
                if (cudaMemcpyAsync(elapsed_ms, &mean, sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
                    cudaStreamSynchronize(stream) != cudaSuccess) {
                    s = CERE_ERR_INTERNAL;
                }
            }
        }
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
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

}  // namespace
}  // namespace cere::ktest

// out = moe_bf16_block(x) with rows 0 (the plans pinned at max_m = m) or rows in [1, GemmPlan::kMaxRows] with
// m <= rows (the executor's rows MoePlans); the routed experts and the combine on rows [first, m) only (first in
// [0, m), rows [0, first) of out not written).
extern "C" __attribute__((visibility("default"))) int cere_ktest_moe_bf16(
    const void *x, const void *router_weight, const void *gate_up_weight, const void *down_weight,
    const void *shared_gate_weight, const void *shared_up_weight, const void *shared_down_weight,
    const void *shared_expert_gate, void *out, int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k,
    int64_t inter, int64_t shared_inter, int64_t rows, int64_t first, cudaStream_t stream) {
    cere::kernels::MoeDims dims;
    cere::kernels::MoeBf16Layer layer;
    if (!cere::ktest::block_args(router_weight, gate_up_weight, down_weight, shared_gate_weight, shared_up_weight,
                                 shared_down_weight, shared_expert_gate, m, hidden, num_experts, top_k, inter,
                                 shared_inter, rows, &dims, &layer)) {
        return CERE_ERR_ARGUMENT;
    }
    return cere::ktest::moe_bf16(dims, layer, x, m, rows, first, out, 1, nullptr, nullptr, stream);
}

// cere_ktest_moe_bf16 timed: one warm-up call, then `iterations` calls between two CUDA events on the stream (plans
// and workspace made once, outside the timed region); `elapsed_ms` (device float32 [1]) receives the mean per call.
extern "C" __attribute__((visibility("default"))) int cere_ktest_moe_bf16_time(
    const void *x, const void *router_weight, const void *gate_up_weight, const void *down_weight,
    const void *shared_gate_weight, const void *shared_up_weight, const void *shared_down_weight,
    const void *shared_expert_gate, void *out, int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k,
    int64_t inter, int64_t shared_inter, int64_t rows, int64_t first, int64_t iterations, void *elapsed_ms,
    cudaStream_t stream) {
    cere::kernels::MoeDims dims;
    cere::kernels::MoeBf16Layer layer;
    if (iterations <= 0 || elapsed_ms == nullptr ||
        !cere::ktest::block_args(router_weight, gate_up_weight, down_weight, shared_gate_weight, shared_up_weight,
                                 shared_down_weight, shared_expert_gate, m, hidden, num_experts, top_k, inter,
                                 shared_inter, rows, &dims, &layer)) {
        return CERE_ERR_ARGUMENT;
    }
    return cere::ktest::moe_bf16(dims, layer, x, m, rows, first, out, iterations, static_cast<float *>(elapsed_ms),
                                 nullptr, stream);
}

// cere_ktest_moe_bf16 cold-L2 timed (l2_flush.h cold_us): one untimed call, then `iterations` calls each after an
// L2 flush; `us` (host double [2]) receives the median and minimum µs of one call.
extern "C" __attribute__((visibility("default"))) int cere_ktest_moe_bf16_cold(
    const void *x, const void *router_weight, const void *gate_up_weight, const void *down_weight,
    const void *shared_gate_weight, const void *shared_up_weight, const void *shared_down_weight,
    const void *shared_expert_gate, void *out, int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k,
    int64_t inter, int64_t shared_inter, int64_t rows, int64_t first, int64_t iterations, void *us,
    cudaStream_t stream) {
    cere::kernels::MoeDims dims;
    cere::kernels::MoeBf16Layer layer;
    if (iterations <= 0 || us == nullptr ||
        !cere::ktest::block_args(router_weight, gate_up_weight, down_weight, shared_gate_weight, shared_up_weight,
                                 shared_down_weight, shared_expert_gate, m, hidden, num_experts, top_k, inter,
                                 shared_inter, rows, &dims, &layer)) {
        return CERE_ERR_ARGUMENT;
    }
    return cere::ktest::moe_bf16(dims, layer, x, m, rows, first, out, iterations, nullptr, static_cast<double *>(us),
                                 stream);
}

// cere_ktest_moe_bf16 with NVFP4 W4A16 routed experts (MoeBf16Layer gate_up_q.., track Fp4S3): gate_up_q uint8
// [E, 2 I, H / 2], gate_up_scale uint8 (E4M3) [E, 2 I, H / 16], gate_up_global float32 [E, 2], down_q uint8
// [E, H, I / 2], down_scale uint8 [E, H, I / 16], down_global float32 [E]; the router and shared expert BF16 as
// cere_ktest_moe_bf16. iterations 0: one call; > 0: cold-L2 timed as cere_ktest_moe_bf16_cold (`us` host double [2]).
extern "C" __attribute__((visibility("default"))) int cere_ktest_moe_w4a16(
    const void *x, const void *router_weight, const void *gate_up_q, const void *gate_up_scale,
    const void *gate_up_global, const void *down_q, const void *down_scale, const void *down_global,
    const void *shared_gate_weight, const void *shared_up_weight, const void *shared_down_weight,
    const void *shared_expert_gate, void *out, int64_t m, int64_t hidden, int64_t num_experts, int64_t top_k,
    int64_t inter, int64_t shared_inter, int64_t rows, int64_t first, int64_t iterations, void *us,
    cudaStream_t stream) {
    cere::kernels::MoeDims dims;
    cere::kernels::MoeBf16Layer layer;
    if (iterations < 0 || (iterations > 0 && us == nullptr) || gate_up_q == nullptr || gate_up_scale == nullptr ||
        gate_up_global == nullptr || down_q == nullptr || down_scale == nullptr || down_global == nullptr ||
        !cere::ktest::block_args(router_weight, nullptr, nullptr, shared_gate_weight, shared_up_weight,
                                 shared_down_weight, shared_expert_gate, m, hidden, num_experts, top_k, inter,
                                 shared_inter, rows, &dims, &layer)) {
        return CERE_ERR_ARGUMENT;
    }
    layer.gate_up_q = static_cast<const uint8_t *>(gate_up_q);
    layer.gate_up_scale = static_cast<const uint8_t *>(gate_up_scale);
    layer.gate_up_global = static_cast<const float *>(gate_up_global);
    layer.down_q = static_cast<const uint8_t *>(down_q);
    layer.down_scale = static_cast<const uint8_t *>(down_scale);
    layer.down_global = static_cast<const float *>(down_global);
    return cere::ktest::moe_bf16(dims, layer, x, m, rows, first, out, iterations > 0 ? iterations : 1, nullptr,
                                 iterations > 0 ? static_cast<double *>(us) : nullptr, stream);
}
