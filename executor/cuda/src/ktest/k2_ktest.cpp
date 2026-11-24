// K2 kernel-test entry points (executor/tests/gpu/test_kernels_k2_gemm.py): the pinned BF16 GEMM, its prefill pins
// (ranges, version refusal) and the last-row-only lm_head. Device pointers, int64 sizes, the stream last; 0 = success,
// else a cere_status.
#include "ktest/k2_ktest.h"

#include <limits>

namespace cere::ktest {
namespace {

bool dims_ok(int64_t m, int64_t n, int64_t k) {
    constexpr int64_t kMax = std::numeric_limits<int32_t>::max();
    return m > 0 && n > 0 && k > 0 && m <= kMax && n <= kMax && k <= kMax;
}

// `iters` >= 1 calls plan.run(x, w_i, m, y), w_i = copy i % copies of the `copies` weights [copies, n, k] back to
// back in w; with iters > kTimedWarm, *us (host double) = mean µs per call over the calls after the first kTimedWarm
// (CUDA events on `stream`).
constexpr int64_t kTimedWarm = 5;

cere_status timed_runs(const kernels::GemmPlan &plan, const void *x, const void *w, uint32_t m, int64_t n, int64_t k,
                       int64_t iters, int64_t copies, void *y, void *us, cudaStream_t stream) {
    cere_status s = CERE_OK;
    cudaEvent_t start = nullptr, stop = nullptr;
    const bool timed = iters > kTimedWarm;
    if (timed && (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)) {
        s = CERE_ERR_INTERNAL;
    }
    for (int64_t i = 0; i < iters && s == CERE_OK; ++i) {
        if (timed && i == kTimedWarm && cudaEventRecord(start, stream) != cudaSuccess) {
            s = CERE_ERR_INTERNAL;
            break;
        }
        const auto *wi =
            static_cast<const __nv_bfloat16 *>(w) + static_cast<size_t>(i % copies) * size_t(n) * size_t(k);
        s = plan.run(static_cast<const __nv_bfloat16 *>(x), wi, m, y, stream);
    }
    if (s == CERE_OK && timed) {
        float ms = 0.0f;
        if (cudaEventRecord(stop, stream) != cudaSuccess || cudaEventSynchronize(stop) != cudaSuccess ||
            cudaEventElapsedTime(&ms, start, stop) != cudaSuccess) {
            s = CERE_ERR_INTERNAL;
        } else {
            *static_cast<double *>(us) = 1000.0 * double(ms) / double(iters - kTimedWarm);
        }
    }
    if (start != nullptr) {
        cudaEventDestroy(start);
    }
    if (stop != nullptr) {
        cudaEventDestroy(stop);
    }
    return s;
}

}  // namespace

cublasLtHandle_t gemm_handle() {
    static const cublasLtHandle_t handle = [] {
        cublasLtHandle_t h = nullptr;
        return kernels::gemm_handle_create(&h) == CERE_OK ? h : nullptr;
    }();
    return handle;
}

cere_status plan_init(kernels::GemmPlan *plan, cublasLtHandle_t handle, uint32_t n, uint32_t k, uint32_t max_m,
                      kernels::GemmOut out, void **workspace, cudaStream_t stream) {
    *workspace = nullptr;
    const kernels::GemmPins *pins = kernels::gemm_pins_for(n, k, out);
    const size_t bytes = pins == nullptr ? 0 : kernels::gemm_pins_workspace_bytes(*pins, max_m);
    if (bytes > 0 && cudaMallocAsync(workspace, bytes, stream) != cudaSuccess) {
        *workspace = nullptr;
        return CERE_ERR_CAPACITY;
    }
    const cere_status s = plan->init(handle, n, k, max_m, out, pins, *workspace, bytes);
    if (s != CERE_OK && *workspace != nullptr) {
        cudaFreeAsync(*workspace, stream);
        *workspace = nullptr;
    }
    return s;
}

cere_status gemm(const void *x, const void *w, int64_t m, int64_t n, int64_t k, void *y, kernels::GemmOut out,
                 cudaStream_t stream) {
    if (!dims_ok(m, n, k)) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = gemm_handle();
    if (handle == nullptr) {
        return CERE_ERR_INTERNAL;
    }
    kernels::GemmPlan plan;
    void *workspace = nullptr;
    cere_status s = plan_init(&plan, handle, static_cast<uint32_t>(n), static_cast<uint32_t>(k),
                              static_cast<uint32_t>(m), out, &workspace, stream);
    if (s == CERE_OK) {
        s = plan.run(static_cast<const __nv_bfloat16 *>(x), static_cast<const __nv_bfloat16 *>(w),
                     static_cast<uint32_t>(m), y, stream);
    }
    if (workspace != nullptr && cudaFreeAsync(workspace, stream) != cudaSuccess && s == CERE_OK) {
        s = CERE_ERR_INTERNAL;
    }
    return s;
}

cere_status lm_head_last(const void *hidden, const void *w, int64_t m, int64_t vocab, int64_t k, void *logits,
                         cudaStream_t stream) {
    if (!dims_ok(m, vocab, k)) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = gemm_handle();
    if (handle == nullptr) {
        return CERE_ERR_INTERNAL;
    }
    // The executor's lm_head plan covers m = 1 only: the last-row path never runs a larger M.
    kernels::GemmPlan plan;
    cere_status s =
        plan.init(handle, static_cast<uint32_t>(vocab), static_cast<uint32_t>(k), 1, kernels::GemmOut::kBF16);
    if (s != CERE_OK) {
        return s;
    }
    return kernels::lm_head_last_row(plan, static_cast<const __nv_bfloat16 *>(hidden), static_cast<uint32_t>(m),
                                     static_cast<const __nv_bfloat16 *>(w), logits, stream);
}

// y[m, n] = x[m, k] · w[n, k]^T through a GemmPlan(n, k, max_m = m, out) without pins: the bucket algorithm
// GemmPlan::init picks for m, the reference of the bucket re-pins (k2_gemm.h kFirstBucketRepin).
cere_status gemm_unpinned(const void *x, const void *w, int64_t m, int64_t n, int64_t k, void *y, kernels::GemmOut out,
                          cudaStream_t stream) {
    if (!dims_ok(m, n, k)) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = gemm_handle();
    if (handle == nullptr) {
        return CERE_ERR_INTERNAL;
    }
    kernels::GemmPlan plan;
    const cere_status s =
        plan.init(handle, static_cast<uint32_t>(n), static_cast<uint32_t>(k), static_cast<uint32_t>(m), out);
    if (s != CERE_OK) {
        return s;
    }
    return plan.run(static_cast<const __nv_bfloat16 *>(x), static_cast<const __nv_bfloat16 *>(w),
                    static_cast<uint32_t>(m), y, stream);
}

cere_status gemm_pin_ranges(int64_t n, int64_t k, kernels::GemmOut out, int64_t *ranges, int64_t cap,
                            int64_t *count) {
    if (!dims_ok(1, n, k) || ranges == nullptr || count == nullptr || cap < 0) {
        return CERE_ERR_ARGUMENT;
    }
    const kernels::GemmPins *pins = kernels::gemm_pins_for(static_cast<uint32_t>(n), static_cast<uint32_t>(k), out);
    if (pins == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    *count = pins->count;
    if (pins->count > cap) {
        return CERE_ERR_CAPACITY;
    }
    for (uint32_t i = 0; i < pins->count; ++i) {
        ranges[2 * i] = pins->ranges[i].m_lo;
        ranges[2 * i + 1] = pins->ranges[i].m_hi;
    }
    return CERE_OK;
}

cere_status gemm_pins_init(int64_t n, int64_t k, int64_t max_m, int64_t version_offset, cudaStream_t stream) {
    if (!dims_ok(max_m, n, k)) {
        return CERE_ERR_ARGUMENT;
    }
    const kernels::GemmPins *pins =
        kernels::gemm_pins_for(static_cast<uint32_t>(n), static_cast<uint32_t>(k), kernels::GemmOut::kBF16);
    cublasLtHandle_t handle = gemm_handle();
    if (pins == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    if (handle == nullptr) {
        return CERE_ERR_INTERNAL;
    }
    kernels::GemmPins shifted = *pins;
    shifted.cublaslt_version = static_cast<size_t>(static_cast<int64_t>(pins->cublaslt_version) + version_offset);
    const size_t bytes = kernels::gemm_pins_workspace_bytes(shifted, static_cast<uint32_t>(max_m));
    void *workspace = nullptr;
    if (bytes > 0 && cudaMallocAsync(&workspace, bytes, stream) != cudaSuccess) {
        return CERE_ERR_CAPACITY;
    }
    kernels::GemmPlan plan;
    cere_status s = plan.init(handle, static_cast<uint32_t>(n), static_cast<uint32_t>(k),
                              static_cast<uint32_t>(max_m), kernels::GemmOut::kBF16, &shifted, workspace, bytes);
    if (workspace != nullptr && cudaFreeAsync(workspace, stream) != cudaSuccess && s == CERE_OK) {
        s = CERE_ERR_INTERNAL;
    }
    return s;
}

cere_status gemm_rows(const void *x, const void *w, int64_t m, int64_t n, int64_t k, int64_t rows, int64_t out_f32,
                      int64_t fused_allowed, void *y, int64_t *per_row, cudaStream_t stream) {
    if (!dims_ok(m, n, k) || rows < 0 || rows > kernels::GemmPlan::kMaxRows || (rows > 0 && m > rows) ||
        out_f32 < 0 || out_f32 > 1 || per_row == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = gemm_handle();
    if (handle == nullptr) {
        return CERE_ERR_INTERNAL;
    }
    const kernels::GemmOut out = out_f32 != 0 ? kernels::GemmOut::kF32 : kernels::GemmOut::kBF16;
    // The instantiated plan: its bucket-0 (m = 1) algorithm is the executor plan's whatever its max_m and pins
    // (pins start at m = GemmPins::kPinMinM; bucket 0 is queried at m = 1 alone).
    kernels::GemmPlan instantiated;
    cere_status s = instantiated.init(handle, static_cast<uint32_t>(n), static_cast<uint32_t>(k),
                                      rows == 0 ? static_cast<uint32_t>(m) : 1, out);
    if (s != CERE_OK) {
        return s;
    }
    kernels::GemmPlan plan;
    if (rows > 0) {
        s = plan.init_rows(handle, instantiated, static_cast<uint32_t>(rows), static_cast<uint32_t>(fused_allowed));
        if (s != CERE_OK) {
            return s;
        }
    }
    const kernels::GemmPlan &run = rows == 0 ? instantiated : plan;
    *per_row = run.per_row(static_cast<uint32_t>(m)) ? 1 : 0;
    return run.run(static_cast<const __nv_bfloat16 *>(x), static_cast<const __nv_bfloat16 *>(w),
                   static_cast<uint32_t>(m), y, stream);
}

}  // namespace cere::ktest

// The plans of these entry points (and of cere_ktest_gemm_rows) are as GemmPlan::init leaves them: the k2_gemm lever's
// default, cuBLASLt for every m (CERE_LEVER_K2_CUBLASLT; test_levers_off_identity.py compares them with
// the merge-base library). cere_ktest_gemm_path and cere_ktest_gemm_segments switch the small-M kernel on per plan.
extern "C" __attribute__((visibility("default"))) int cere_ktest_gemm_bf16(const void *x, const void *w, int64_t m,
                                                                           int64_t n, int64_t k, void *y,
                                                                           cudaStream_t stream) {
    return cere::ktest::gemm(x, w, m, n, k, y, cere::kernels::GemmOut::kBF16, stream);
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_lm_head_last(const void *hidden, const void *w,
                                                                              int64_t m, int64_t vocab, int64_t k,
                                                                              void *logits, cudaStream_t stream) {
    return cere::ktest::lm_head_last(hidden, w, m, vocab, k, logits, stream);
}

// Host int64 outputs: ranges[2 * i], ranges[2 * i + 1] = m_lo, m_hi of pin range i of gemm_pins_for(n, k, out) (out
// F32 when out_f32 is 1, else BF16), and *count = the number of ranges (CERE_ERR_CAPACITY when it exceeds cap;
// CERE_ERR_ARGUMENT when the shape has none).
extern "C" __attribute__((visibility("default"))) int cere_ktest_gemm_pin_ranges(int64_t n, int64_t k, int64_t out_f32,
                                                                                 void *ranges, int64_t cap,
                                                                                 void *count, cudaStream_t stream) {
    (void)stream;
    if (out_f32 < 0 || out_f32 > 1) {
        return CERE_ERR_ARGUMENT;
    }
    const auto out = out_f32 != 0 ? cere::kernels::GemmOut::kF32 : cere::kernels::GemmOut::kBF16;
    return cere::ktest::gemm_pin_ranges(n, k, out, static_cast<int64_t *>(ranges), cap,
                                        static_cast<int64_t *>(count));
}

// r3-prefill design note (the bucket re-pins' reference): y = x · w^T (BF16, or F32 when out_f32 is 1) through a plan
// of max_m = m without pins, i.e. the bucket algorithm GemmPlan::init picks for m.
extern "C" __attribute__((visibility("default"))) int cere_ktest_gemm_unpinned(const void *x, const void *w, int64_t m,
                                                                               int64_t n, int64_t k, int64_t out_f32,
                                                                               void *y, cudaStream_t stream) {
    if (out_f32 < 0 || out_f32 > 1) {
        return CERE_ERR_ARGUMENT;
    }
    const auto out = out_f32 != 0 ? cere::kernels::GemmOut::kF32 : cere::kernels::GemmOut::kBF16;
    return cere::ktest::gemm_unpinned(x, w, m, n, k, y, out, stream);
}

// The status of GemmPlan::init(n, k, max_m, BF16) with gemm_pins_for's pins whose cublaslt_version is shifted by
// version_offset (0: the executor's pins), given a workspace of their size; no GEMM runs.
extern "C" __attribute__((visibility("default"))) int cere_ktest_gemm_pins_init(int64_t n, int64_t k, int64_t max_m,
                                                                                int64_t version_offset,
                                                                                cudaStream_t stream) {
    return cere::ktest::gemm_pins_init(n, k, max_m, version_offset, stream);
}

// m26 design note D6 / I6 (K1 test_rows_plan_invariant): y[m, n] = x[m, k] · w[n, k]^T (BF16, or F32 when out_f32
// is 1) through a rows plan GemmPlan::init_rows(instantiated, rows, fused_allowed) of the instantiated plan of (n, k,
// out), m in [1, rows]; rows 0 runs the instantiated plan itself (max_m = m), the single-row reference at m = 1.
// Host int64 *per_row = per_row(m) of the plan that ran (1: the m-row call ran as m single-row GEMMs).
extern "C" __attribute__((visibility("default"))) int cere_ktest_gemm_rows(const void *x, const void *w, int64_t m,
                                                                           int64_t n, int64_t k, int64_t rows,
                                                                           int64_t out_f32, int64_t fused_allowed,
                                                                           void *y, void *per_row,
                                                                           cudaStream_t stream) {
    return cere::ktest::gemm_rows(x, w, m, n, k, rows, out_f32, fused_allowed, y, static_cast<int64_t *>(per_row),
                                  stream);
}

// m23 design note P11: y = x · w^T through a plan of max_m = m whose small-M kernel is on (small_m 1, the k2_gemm
// lever CERE_LEVER_K2_SMALL_M's path for m <= 16) or off (0: cuBLASLt, the default lever). out_f32 1 = F32 y. iters >= 1
// calls of the plan; with iters > kWarm, *us (host double) = mean µs per call of the calls after the first kWarm.
// w holds `copies` >= 1 weights [copies, n, k] back to back; call i reads copy i % copies (copies whose total
// exceeds the L2 cache time weights streamed from DRAM, as a decode step reads them).
extern "C" __attribute__((visibility("default"))) int cere_ktest_gemm_path(const void *x, const void *w, int64_t m,
                                                                           int64_t n, int64_t k, int64_t out_f32,
                                                                           int64_t small_m, int64_t iters,
                                                                           int64_t copies, void *y, void *us,
                                                                           cudaStream_t stream) {
    constexpr int64_t kWarm = 5;
    if (m <= 0 || n <= 0 || k <= 0 || m > INT32_MAX || n > INT32_MAX || k > INT32_MAX || out_f32 < 0 || out_f32 > 1 ||
        small_m < 0 || small_m > 1 || iters < 1 || iters > 10'000 || copies < 1 || copies > 4096 || us == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = cere::ktest::gemm_handle();
    if (handle == nullptr) {
        return CERE_ERR_INTERNAL;
    }
    const auto out = out_f32 != 0 ? cere::kernels::GemmOut::kF32 : cere::kernels::GemmOut::kBF16;
    cere::kernels::GemmPlan plan;
    void *workspace = nullptr;
    cere_status s = cere::ktest::plan_init(&plan, handle, static_cast<uint32_t>(n), static_cast<uint32_t>(k),
                                           static_cast<uint32_t>(m), out, &workspace, stream);
    if (s == CERE_OK) {
        plan.use_small_m(small_m != 0);
        if (small_m != 0 && !plan.small_m()) {
            s = CERE_ERR_UNSUPPORTED;
        }
    }
    cudaEvent_t start = nullptr, stop = nullptr;
    const bool timed = iters > kWarm;
    if (s == CERE_OK && timed &&
        (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)) {
        s = CERE_ERR_INTERNAL;
    }
    for (int64_t i = 0; i < iters && s == CERE_OK; ++i) {
        if (timed && i == kWarm && cudaEventRecord(start, stream) != cudaSuccess) {
            s = CERE_ERR_INTERNAL;
            break;
        }
        const auto *wi = static_cast<const __nv_bfloat16 *>(w) + static_cast<size_t>(i % copies) * size_t(n) * size_t(k);
        s = plan.run(static_cast<const __nv_bfloat16 *>(x), wi, static_cast<uint32_t>(m), y, stream);
    }
    if (s == CERE_OK && timed) {
        float ms = 0.0f;
        if (cudaEventRecord(stop, stream) != cudaSuccess || cudaEventSynchronize(stop) != cudaSuccess ||
            cudaEventElapsedTime(&ms, start, stop) != cudaSuccess) {
            s = CERE_ERR_INTERNAL;
        } else {
            *static_cast<double *>(us) = 1000.0 * double(ms) / double(iters - kWarm);
        }
    }
    if (start != nullptr) {
        cudaEventDestroy(start);
    }
    if (stop != nullptr) {
        cudaEventDestroy(stop);
    }
    if (workspace != nullptr && cudaFreeAsync(workspace, stream) != cudaSuccess && s == CERE_OK) {
        s = CERE_ERR_INTERNAL;
    }
    return s;
}

// m23 design note P11 fusion: kernels::gemm_segments over `count` weights sharing x [m, k]. Host int64 arrays:
// w_ptrs[i] / y_ptrs[i] device pointers, ns[i] the rows of weight i; out_f32 as above. One plan of max_m = m per
// weight (the executor's GDN in-projections / shared gate-up shape).
extern "C" __attribute__((visibility("default"))) int cere_ktest_gemm_segments(const void *x, int64_t m, int64_t k,
                                                                               int64_t count, const int64_t *w_ptrs,
                                                                               const int64_t *ns,
                                                                               const int64_t *y_ptrs, int64_t out_f32,
                                                                               cudaStream_t stream) {
    constexpr int64_t kMax = 4;
    if (m <= 0 || m > INT32_MAX || k <= 0 || k > INT32_MAX || count <= 0 || count > kMax || w_ptrs == nullptr ||
        ns == nullptr || y_ptrs == nullptr || out_f32 < 0 || out_f32 > 1) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = cere::ktest::gemm_handle();
    if (handle == nullptr) {
        return CERE_ERR_INTERNAL;
    }
    const auto out = out_f32 != 0 ? cere::kernels::GemmOut::kF32 : cere::kernels::GemmOut::kBF16;
    cere::kernels::GemmPlan plans[kMax];
    void *workspaces[kMax] = {};
    const cere::kernels::GemmPlan *plan_ptrs[kMax] = {};
    const __nv_bfloat16 *w[kMax] = {};
    void *y[kMax] = {};
    cere_status s = CERE_OK;
    for (int64_t i = 0; i < count && s == CERE_OK; ++i) {
        if (ns[i] <= 0 || ns[i] > INT32_MAX) {
            s = CERE_ERR_ARGUMENT;
            break;
        }
        s = cere::ktest::plan_init(&plans[i], handle, static_cast<uint32_t>(ns[i]), static_cast<uint32_t>(k),
                                   static_cast<uint32_t>(m), out, &workspaces[i], stream);
        plans[i].use_small_m(true);  // the small-M kernel (the k2_gemm lever CERE_LEVER_K2_SMALL_M)
        plan_ptrs[i] = &plans[i];
        w[i] = reinterpret_cast<const __nv_bfloat16 *>(w_ptrs[i]);
        y[i] = reinterpret_cast<void *>(y_ptrs[i]);
    }
    if (s == CERE_OK) {
        s = cere::kernels::gemm_segments(plan_ptrs, w, y, static_cast<uint32_t>(count),
                                         static_cast<const __nv_bfloat16 *>(x), static_cast<uint32_t>(m), stream);
    }
    for (int64_t i = 0; i < count; ++i) {
        if (workspaces[i] != nullptr && cudaFreeAsync(workspaces[i], stream) != cudaSuccess && s == CERE_OK) {
            s = CERE_ERR_INTERNAL;
        }
    }
    return s;
}

// m23 design note P11 / D12: the launch k2s::gemm makes for inner dimension k over `count` segments of widths ns
// (host int64 [count]) on the current device; out: host int64 [9] = grid x, y, z, block x, y, z, cluster x, y, z (all
// 0 outside the kernel's contract).
// Host arithmetic only (D12 maps the profile's small-M kernel records to call shapes by it).
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2s_geometry(int64_t k, int64_t count,
                                                                              const int64_t *ns, void *out,
                                                                              cudaStream_t stream) {
    (void)stream;
    if (k <= 0 || k > INT32_MAX || count <= 0 || count > cere::kernels::k2s::kMaxSegments || ns == nullptr ||
        out == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    uint32_t widths[cere::kernels::k2s::kMaxSegments];
    for (int64_t i = 0; i < count; ++i) {
        if (ns[i] <= 0 || ns[i] > INT32_MAX) {
            return CERE_ERR_ARGUMENT;
        }
        widths[i] = static_cast<uint32_t>(ns[i]);
    }
    const cere::kernels::k2s::Geometry g =
        cere::kernels::k2s::geometry(static_cast<uint32_t>(k), widths, static_cast<uint32_t>(count));
    int64_t *dims = static_cast<int64_t *>(out);
    dims[0] = g.grid.x;
    dims[1] = g.grid.y;
    dims[2] = g.grid.z;
    dims[3] = g.block.x;
    dims[4] = g.block.y;
    dims[5] = g.block.z;
    dims[6] = g.cluster.x;
    dims[7] = g.cluster.y;
    dims[8] = g.cluster.z;
    return CERE_OK;
}

// m23 design note P11R-21: the m = 1 algorithm of the executor's plan of shape (n, k, out) (max_m 16, the verify
// rows) and the gemvx order the rows plans run it in. out: host int64 [5] = the loaded libcublasLt's
// cublasLtGetVersion(), the algorithm's CUBLASLT_ALGO_CONFIG_ID and CUSTOM_OPTION, and k2v::order_of's lanes (T) and
// chunk (S) for them (0 0: no identified order, the rows plans keep per-row calls where AlgoCheck refuses m).
extern "C" __attribute__((visibility("default"))) int cere_ktest_gemm_m1_algo(int64_t n, int64_t k, int64_t out_f32,
                                                                              void *out, cudaStream_t stream) {
    constexpr int64_t kVerifyRows = 16;
    if (n <= 0 || n > INT32_MAX || k <= 0 || k > INT32_MAX || out_f32 < 0 || out_f32 > 1 || out == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = cere::ktest::gemm_handle();
    if (handle == nullptr) {
        return CERE_ERR_INTERNAL;
    }
    const auto kind = out_f32 != 0 ? cere::kernels::GemmOut::kF32 : cere::kernels::GemmOut::kBF16;
    cere::kernels::GemmPlan plan;
    void *workspace = nullptr;
    cere_status s = cere::ktest::plan_init(&plan, handle, static_cast<uint32_t>(n), static_cast<uint32_t>(k),
                                           kVerifyRows, kind, &workspace, stream);
    int algo_id = -1;
    uint32_t custom_option = 0;
    if (s == CERE_OK) {
        s = plan.m1_algo(&algo_id, &custom_option);
    }
    if (s == CERE_OK) {
        const size_t version = cublasLtGetVersion();
        const cere::kernels::k2v::Order o =
            cere::kernels::k2v::order_of(version, algo_id, custom_option, static_cast<uint32_t>(k));
        int64_t *v = static_cast<int64_t *>(out);
        v[0] = static_cast<int64_t>(version);
        v[1] = algo_id;
        v[2] = custom_option;
        v[3] = o.lanes;
        v[4] = o.chunk;
    }
    if (workspace != nullptr && cudaFreeAsync(workspace, stream) != cudaSuccess && s == CERE_OK) {
        s = CERE_ERR_INTERNAL;
    }
    return s;
}

// m23 design note P11R-21: k2v::order_of for an explicit libcublasLt version (the refusal check: any version but
// the identified one maps every pairing to no order). out: host int64 [2] = lanes, chunk. Host arithmetic only.
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2v_order(int64_t cublaslt_version, int64_t algo_id,
                                                                           int64_t custom_option, int64_t k,
                                                                           void *out, cudaStream_t stream) {
    (void)stream;
    if (cublaslt_version < 0 || algo_id < INT32_MIN || algo_id > INT32_MAX || custom_option < 0 ||
        custom_option > UINT32_MAX || k <= 0 || k > INT32_MAX || out == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    const cere::kernels::k2v::Order o =
        cere::kernels::k2v::order_of(static_cast<size_t>(cublaslt_version), static_cast<int>(algo_id),
                                     static_cast<uint32_t>(custom_option), static_cast<uint32_t>(k));
    int64_t *v = static_cast<int64_t *>(out);
    v[0] = o.lanes;
    v[1] = o.chunk;
    return CERE_OK;
}

// m23 design note P11R-21: one k2v::gemv launch in an explicit order (lanes T, chunk S): y [m, n] (F32 when out_f32)
// = x [m, k] · w [n, k]^T. The perturbation check runs a known order on a shape served by another and expects bits
// that differ from the m = 1 call. CERE_ERR_ARGUMENT for an order or shape gemv() does not serve.
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2v_gemv(const void *x, int64_t m, int64_t k,
                                                                          const void *w, int64_t n, int64_t out_f32,
                                                                          int64_t lanes, int64_t chunk, void *y,
                                                                          cudaStream_t stream) {
    if (m <= 0 || m > cere::kernels::k2v::kMaxM || k <= 0 || k > INT32_MAX || n <= 0 || n > INT32_MAX ||
        out_f32 < 0 || out_f32 > 1 || lanes < 0 || lanes > UINT32_MAX || chunk < 0 || chunk > UINT32_MAX) {
        return CERE_ERR_ARGUMENT;
    }
    const cere::kernels::k2v::Order o{static_cast<uint32_t>(lanes), static_cast<uint32_t>(chunk)};
    const cudaError_t e = cere::kernels::k2v::gemv(
        static_cast<const __nv_bfloat16 *>(x), static_cast<uint32_t>(m), static_cast<uint32_t>(k),
        static_cast<const __nv_bfloat16 *>(w), static_cast<uint32_t>(n), y, out_f32 != 0, o, stream);
    if (e == cudaErrorInvalidValue) {
        return CERE_ERR_ARGUMENT;
    }
    return e == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

// m23 design note P11R-21: y [m, n] = x [m, k] · w [n, k]^T through a rows plan GemmPlan::init_rows(instantiated,
// rows, every m allowed, paths) of the instantiated plan of (n, k, out), m in [1, rows]; paths restricts the one-launch
// paths for an m AlgoCheck refuses on the rows layout (0 none, GemmPlan::kRowsGemv 1, kRowsBatched 2, 3 both; -1
// init_rows' default, the executor's rows plans).
// Host int64 info [3] = how m ran (0 one GEMM on the rows layout, m = 1 included; 1 m single-row calls; 2 one
// k2v::gemv launch; 3 one strided-batched cublasLtMatmul) and the rows plan's m = 1 algorithm (CUBLASLT_ALGO_CONFIG_ID,
// CUSTOM_OPTION). iters, copies and *us as cere_ktest_gemm_path (w holds `copies` weights back to back, call i reads
// copy i % copies; *us written when iters > 5).
extern "C" __attribute__((visibility("default"))) int cere_ktest_gemm_rows_path(const void *x, const void *w,
                                                                                int64_t m, int64_t n, int64_t k,
                                                                                int64_t rows, int64_t out_f32,
                                                                                int64_t paths, int64_t iters,
                                                                                int64_t copies, void *y, void *info,
                                                                                void *us, cudaStream_t stream) {
    using cere::kernels::GemmPlan;
    if (m <= 0 || n <= 0 || k <= 0 || m > rows || n > INT32_MAX || k > INT32_MAX || rows > GemmPlan::kMaxRows ||
        out_f32 < 0 || out_f32 > 1 || paths < -1 || paths > GemmPlan::kRowsAllPaths || iters < 1 || iters > 10'000 ||
        copies < 1 || copies > 4096 || info == nullptr || (iters > cere::ktest::kTimedWarm && us == nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = cere::ktest::gemm_handle();
    if (handle == nullptr) {
        return CERE_ERR_INTERNAL;
    }
    const auto out = out_f32 != 0 ? cere::kernels::GemmOut::kF32 : cere::kernels::GemmOut::kBF16;
    GemmPlan instantiated;
    cere_status s = instantiated.init(handle, static_cast<uint32_t>(n), static_cast<uint32_t>(k), 1, out);
    GemmPlan plan;
    if (s == CERE_OK) {
        s = paths < 0 ? plan.init_rows(handle, instantiated, static_cast<uint32_t>(rows))
                      : plan.init_rows(handle, instantiated, static_cast<uint32_t>(rows), ~uint32_t{0},
                                       static_cast<uint32_t>(paths));
    }
    int algo_id = -1;
    uint32_t custom_option = 0;
    if (s == CERE_OK) {
        s = plan.m1_algo(&algo_id, &custom_option);
    }
    if (s != CERE_OK) {
        return s;
    }
    const auto um = static_cast<uint32_t>(m);
    int64_t *v = static_cast<int64_t *>(info);
    v[0] = plan.per_row(um) ? 1 : plan.gemv_rows(um) ? 2 : plan.batched_rows(um) ? 3 : 0;
    v[1] = algo_id;
    v[2] = custom_option;
    return cere::ktest::timed_runs(plan, x, w, um, n, k, iters, copies, y, us, stream);
}

// m23 design note P11R-21 (gemm_m1_custom_option): y [1, n] = x [1, k] · w [n, k]^T through a plan of (n, k, out)
// of max_m 1 initialised with GemmPlan::init's m1_swap = `swap` (1 the executor's plans, 0 bucket 0's heuristic pick
// unswapped: main's algorithm before the swap). Host int64 ids [2] = that plan's m = 1 algorithm (ID, CUSTOM_OPTION).
// iters, copies and *us as cere_ktest_gemm_rows_path.
extern "C" __attribute__((visibility("default"))) int cere_ktest_gemm_m1_swap(const void *x, const void *w, int64_t n,
                                                                              int64_t k, int64_t out_f32, int64_t swap,
                                                                              int64_t iters, int64_t copies, void *y,
                                                                              void *ids, void *us,
                                                                              cudaStream_t stream) {
    if (n <= 0 || k <= 0 || n > INT32_MAX || k > INT32_MAX || out_f32 < 0 || out_f32 > 1 || swap < 0 || swap > 1 ||
        iters < 1 || iters > 10'000 || copies < 1 || copies > 4096 || ids == nullptr ||
        (iters > cere::ktest::kTimedWarm && us == nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = cere::ktest::gemm_handle();
    if (handle == nullptr) {
        return CERE_ERR_INTERNAL;
    }
    const auto out = out_f32 != 0 ? cere::kernels::GemmOut::kF32 : cere::kernels::GemmOut::kBF16;
    cere::kernels::GemmPlan plan;
    cere_status s = plan.init(handle, static_cast<uint32_t>(n), static_cast<uint32_t>(k), 1, out, nullptr, nullptr, 0,
                              swap != 0);
    int algo_id = -1;
    uint32_t custom_option = 0;
    if (s == CERE_OK) {
        s = plan.m1_algo(&algo_id, &custom_option);
    }
    if (s != CERE_OK) {
        return s;
    }
    int64_t *v = static_cast<int64_t *>(ids);
    v[0] = algo_id;
    v[1] = custom_option;
    return cere::ktest::timed_runs(plan, x, w, 1, n, k, iters, copies, y, us, stream);
}

// m23 design note P11R-21 / D12: the launch k2s::gemm_hc_gate makes for inner dimension k (the mix's lowrank) and
// `hidden` columns per stream; out as cere_ktest_k2s_geometry (all 0 outside the kernel's contract). Host only.
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2s_hc_gate_geometry(int64_t k, int64_t hidden,
                                                                                      void *out, cudaStream_t stream) {
    (void)stream;
    if (k <= 0 || k > INT32_MAX || hidden <= 0 || hidden > INT32_MAX || out == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    const cere::kernels::k2s::Geometry g =
        cere::kernels::k2s::hc_gate_geometry(static_cast<uint32_t>(k), static_cast<uint32_t>(hidden));
    int64_t *dims = static_cast<int64_t *>(out);
    dims[0] = g.grid.x;
    dims[1] = g.grid.y;
    dims[2] = g.grid.z;
    dims[3] = g.block.x;
    dims[4] = g.block.y;
    dims[5] = g.block.z;
    dims[6] = g.cluster.x;
    dims[7] = g.cluster.y;
    dims[8] = g.cluster.z;
    return CERE_OK;
}
