// K2 FP8 weights kernel-test entry points (executor/tests/gpu/test_kernels_k2_fp8.py; kernels/k2_fp8.h): the
// quantization, the in-place conversion, the dequantization, the FP8 small-M kernel and the GemmPlan hook. Device
// pointers, int64 sizes (host int64 arrays where named), the stream last; 0 = success, else a cere_status.
#include "kernels/k2_fp8.h"
#include "ktest/k2_ktest.h"

#include <cstdint>

namespace {

namespace k2f = cere::kernels::k2f;

constexpr int64_t kWarm = 5;

bool dim_ok(int64_t v) { return v > 0 && v <= INT32_MAX; }

cere_status status_of(cudaError_t e) {
    if (e == cudaErrorInvalidValue) return CERE_ERR_ARGUMENT;
    return e == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

// Runs `call(i)` iters times; with iters > kWarm, *us (host double) = mean µs per call after the first kWarm.
template <typename Call>
cere_status timed(int64_t iters, void *us, cudaStream_t stream, Call call) {
    cere_status s = CERE_OK;
    cudaEvent_t start = nullptr, stop = nullptr;
    const bool time = iters > kWarm && us != nullptr;
    if (time && (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)) {
        s = CERE_ERR_INTERNAL;
    }
    for (int64_t i = 0; i < iters && s == CERE_OK; ++i) {
        if (time && i == kWarm && cudaEventRecord(start, stream) != cudaSuccess) {
            s = CERE_ERR_INTERNAL;
            break;
        }
        s = call(i);
    }
    if (s == CERE_OK && time) {
        float ms = 0.0f;
        if (cudaEventRecord(stop, stream) != cudaSuccess || cudaEventSynchronize(stop) != cudaSuccess ||
            cudaEventElapsedTime(&ms, start, stop) != cudaSuccess) {
            s = CERE_ERR_INTERNAL;
        } else {
            *static_cast<double *>(us) = 1000.0 * double(ms) / double(iters - kWarm);
        }
    }
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    return s;
}

}  // namespace

// q E4M3 [rows, k] and scale FP32 [rows] of w BF16 [rows, k] (k2f::quantize_rows).
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2f_quantize(const void *w, int64_t rows, int64_t k,
                                                                              void *q, void *scale,
                                                                              cudaStream_t stream) {
    if (!dim_ok(rows) || !dim_ok(k)) return CERE_ERR_ARGUMENT;
    return status_of(k2f::quantize_rows(static_cast<const __nv_bfloat16 *>(w), uint32_t(rows), uint32_t(k),
                                        static_cast<uint8_t *>(q), static_cast<float *>(scale), stream));
}

// storage (BF16 [n, k] on entry, 2nk bytes) converted in place (k2f::convert_in_place) through `scratch`.
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2f_convert(void *storage, int64_t n, int64_t k,
                                                                             void *scratch, int64_t scratch_bytes,
                                                                             cudaStream_t stream) {
    if (!dim_ok(n) || !dim_ok(k) || scratch_bytes <= 0) return CERE_ERR_ARGUMENT;
    return status_of(
        k2f::convert_in_place(storage, uint32_t(n), uint32_t(k), scratch, size_t(scratch_bytes), stream));
}

// w BF16 [n, k] = the converted storage's q [n, k] times its scales (k2f::dequantize).
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2f_dequantize(const void *storage, int64_t n,
                                                                                int64_t k, void *w,
                                                                                cudaStream_t stream) {
    if (!dim_ok(n) || !dim_ok(k) || storage == nullptr) return CERE_ERR_ARGUMENT;
    return status_of(k2f::dequantize(static_cast<const uint8_t *>(storage),
                                     k2f::scales_of(storage, uint32_t(n), uint32_t(k)), uint32_t(n), uint32_t(k),
                                     static_cast<__nv_bfloat16 *>(w), stream));
}

// One k2f::gemm launch over `count` converted storages sharing x [m, k]: host int64 arrays w_ptrs (storages),
// ns (rows), y_ptrs; out_f32 1 = F32 outputs, else BF16. iters >= 1 launches; *us as timed() (nullptr allowed for
// iters <= 5). With copies > 1 the storages are w_ptrs[i] + c * stride bytes for c = launch % copies (a bench
// rotating over copies, count 1).
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2f_gemm(const void *x, int64_t m, int64_t k,
                                                                          int64_t count, const int64_t *w_ptrs,
                                                                          const int64_t *ns, const int64_t *y_ptrs,
                                                                          int64_t out_f32, int64_t iters,
                                                                          int64_t copies, int64_t stride, void *us,
                                                                          cudaStream_t stream) {
    if (!dim_ok(m) || !dim_ok(k) || count < 1 || count > int64_t{k2f::kMaxSegments} || w_ptrs == nullptr ||
        ns == nullptr || y_ptrs == nullptr || out_f32 < 0 || out_f32 > 1 || iters < 1 || iters > 10'000 ||
        copies < 1 || copies > 4096 || (copies > 1 && (count != 1 || stride <= 0))) {
        return CERE_ERR_ARGUMENT;
    }
    for (int64_t i = 0; i < count; ++i) {
        if (!dim_ok(ns[i])) return CERE_ERR_ARGUMENT;
    }
    return timed(iters, us, stream, [&](int64_t it) {
        k2f::Segment segs[k2f::kMaxSegments];
        for (int64_t i = 0; i < count; ++i) {
            const auto n = uint32_t(ns[i]);
            const auto *base = reinterpret_cast<const uint8_t *>(w_ptrs[i]) + (it % copies) * stride;
            segs[i] = k2f::Segment{base, k2f::scales_of(base, n, uint32_t(k)), reinterpret_cast<void *>(y_ptrs[i]), n};
        }
        return status_of(k2f::gemm(static_cast<const __nv_bfloat16 *>(x), uint32_t(m), uint32_t(k), segs,
                                   uint32_t(count), out_f32 != 0, stream));
    });
}

// The GemmPlan hook: y [m, n] = plan.run(x, storage) through a plan of max_m = m whose fp8 table lists `storage` (a
// converted [n, k]) with `staging` (bytes) as Table::staging; small_m as cere_ktest_gemm_path. m <= 16 runs the FP8
// kernel, m > 16 dequantizes into staging and runs the BF16 path there.
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2f_plan(const void *x, const void *storage,
                                                                          int64_t m, int64_t n, int64_t k,
                                                                          int64_t out_f32, int64_t small_m,
                                                                          void *staging, int64_t staging_bytes,
                                                                          void *y, cudaStream_t stream) {
    if (!dim_ok(m) || !dim_ok(n) || !dim_ok(k) || out_f32 < 0 || out_f32 > 1 || small_m < 0 || small_m > 1 ||
        staging_bytes < 0) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = cere::ktest::gemm_handle();
    if (handle == nullptr) return CERE_ERR_INTERNAL;
    const auto out = out_f32 != 0 ? cere::kernels::GemmOut::kF32 : cere::kernels::GemmOut::kBF16;
    cere::kernels::GemmPlan plan;
    void *workspace = nullptr;
    cere_status s = cere::ktest::plan_init(&plan, handle, uint32_t(n), uint32_t(k), uint32_t(m), out, &workspace,
                                           stream);
    k2f::Table table;
    table.add(storage, uint32_t(n), uint32_t(k));
    table.set_staging(static_cast<__nv_bfloat16 *>(staging), size_t(staging_bytes));
    if (s == CERE_OK) {
        plan.use_small_m(small_m != 0);
        plan.use_fp8(&table);
        s = plan.run(static_cast<const __nv_bfloat16 *>(x), static_cast<const __nv_bfloat16 *>(storage),
                     uint32_t(m), y, stream);
    }
    if (workspace != nullptr && cudaFreeAsync(workspace, stream) != cudaSuccess && s == CERE_OK) {
        s = CERE_ERR_INTERNAL;
    }
    return s;
}

// kernels::gemm_segments over `count` converted storages sharing x [m, k] (host int64 arrays w_ptrs, ns, y_ptrs),
// one plan of max_m = m per weight, all with one fp8 table listing every storage (no staging: m <= 16).
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2f_segments(const void *x, int64_t m, int64_t k,
                                                                              int64_t count, const int64_t *w_ptrs,
                                                                              const int64_t *ns,
                                                                              const int64_t *y_ptrs, int64_t out_f32,
                                                                              cudaStream_t stream) {
    constexpr int64_t kMax = 4;
    if (!dim_ok(m) || !dim_ok(k) || count < 1 || count > kMax || w_ptrs == nullptr || ns == nullptr ||
        y_ptrs == nullptr || out_f32 < 0 || out_f32 > 1) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = cere::ktest::gemm_handle();
    if (handle == nullptr) return CERE_ERR_INTERNAL;
    const auto out = out_f32 != 0 ? cere::kernels::GemmOut::kF32 : cere::kernels::GemmOut::kBF16;
    k2f::Table table;
    cere::kernels::GemmPlan plans[kMax];
    void *workspaces[kMax] = {};
    const cere::kernels::GemmPlan *pp[kMax] = {};
    const __nv_bfloat16 *w[kMax] = {};
    void *y[kMax] = {};
    cere_status s = CERE_OK;
    for (int64_t i = 0; i < count && s == CERE_OK; ++i) {
        if (!dim_ok(ns[i])) {
            s = CERE_ERR_ARGUMENT;
            break;
        }
        s = cere::ktest::plan_init(&plans[i], handle, uint32_t(ns[i]), uint32_t(k), uint32_t(m), out, &workspaces[i],
                                   stream);
        table.add(reinterpret_cast<const void *>(w_ptrs[i]), uint32_t(ns[i]), uint32_t(k));
        plans[i].use_small_m(true);
        plans[i].use_fp8(&table);
        pp[i] = &plans[i];
        w[i] = reinterpret_cast<const __nv_bfloat16 *>(w_ptrs[i]);
        y[i] = reinterpret_cast<void *>(y_ptrs[i]);
    }
    if (s == CERE_OK) {
        s = cere::kernels::gemm_segments(pp, w, y, uint32_t(count), static_cast<const __nv_bfloat16 *>(x),
                                         uint32_t(m), stream);
    }
    for (void *ws : workspaces) {
        if (ws != nullptr && cudaFreeAsync(ws, stream) != cudaSuccess && s == CERE_OK) s = CERE_ERR_INTERNAL;
    }
    return s;
}
