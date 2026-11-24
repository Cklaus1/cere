// K2 MXFP8 weights kernel-test entry points (executor/tests/gpu/test_kernels_k2m_mxfp8.py, suite `kernels_k2m`;
// kernels/k2_mxfp8.h, track Fp4S3): the exact dequantization, GemmPlan::run on an MXFP8 weight (the small-M MXFP8
// operand variant or the staging path) next to the same plan on a BF16 weight, gemm_segments, and their cold-L2 times;
// the W8A8 path (k2_gemm small_m_a8, k2-prefill design note P3; test_kernels_k2m_a8.py): x's quantization, the
// scale repack and the plans on it. Device pointers, int64 sizes (host int64 / double arrays where named), the stream
// last; 0 = success, else a cere_status.
#include "kernels/k2_mxfp8.h"
#include "ktest/k2_ktest.h"
#include "ktest/l2_flush.h"

#include <algorithm>
#include <cstdint>

namespace {

namespace k2m = cere::kernels::k2m;
using cere::kernels::GemmOut;
using cere::kernels::GemmPlan;

bool dim_ok(int64_t v) { return v > 0 && v <= INT32_MAX; }

// An init plan's max_m: max(m, cls), and at least k2s::kMaxM + 1 under small_m_a8 (small_m 2), so init_a8 has a W8A8
// bucket whatever the call's m, as on the executor's plans (max_m = max_tokens).
uint32_t plan_max_m(int64_t m, int64_t cls, int64_t small_m) {
    const int64_t at_least = small_m == 2 ? int64_t{cere::kernels::k2s::kMaxM} + 1 : 1;
    return uint32_t(std::max({m, cls, at_least}));
}

// A kL2FlushBytes device buffer of stream-ordered memory for cold_us when iters > 1 (nullptr otherwise).
cere_status flush_buffer(int64_t iters, void **flush, cudaStream_t stream) {
    *flush = nullptr;
    if (iters > 1 && cudaMallocAsync(flush, cere::ktest::kL2FlushBytes, stream) != cudaSuccess) {
        *flush = nullptr;
        return CERE_ERR_CAPACITY;
    }
    return CERE_OK;
}

// Runs `call` once (iters 1) or cold-L2 timed (iters > 1, us host double [2]: median, minimum µs per call).
template <typename Call>
cere_status run_or_time(int64_t iters, void *us, cudaStream_t stream, Call call) {
    void *flush = nullptr;
    cere_status s = flush_buffer(iters, &flush, stream);
    if (s == CERE_OK) {
        s = iters == 1 ? call() : cere::ktest::cold_us(flush, iters - 1, static_cast<double *>(us), stream, call);
    }
    if (flush != nullptr && cudaFreeAsync(flush, stream) != cudaSuccess && s == CERE_OK) s = CERE_ERR_INTERNAL;
    return s;
}

}  // namespace

// w BF16 [n, k] = the exact upcast of q E4M3 [n, k] with UE8M0 scales [n, k / 32] (k2m::dequantize).
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2m_dequantize(const void *q, const void *scales,
                                                                                int64_t n, int64_t k, void *w,
                                                                                cudaStream_t stream) {
    if (!dim_ok(n) || !dim_ok(k)) return CERE_ERR_ARGUMENT;
    const cudaError_t e = k2m::dequantize(static_cast<const uint8_t *>(q), static_cast<const uint8_t *>(scales),
                                          uint32_t(n), uint32_t(k), static_cast<__nv_bfloat16 *>(w), stream);
    if (e == cudaErrorInvalidValue) return CERE_ERR_ARGUMENT;
    return e == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

// q E4M3 [m, k] and its blocked UE8M0 scales (k2m::blocked_scale_bytes(m, k / 32) bytes at `scales`; padding bytes
// untouched) of x BF16 [m, k] (k2m::quantize_x).
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2m_quantize_x(const void *x, int64_t m, int64_t k,
                                                                               void *q, void *scales,
                                                                               cudaStream_t stream) {
    if (!dim_ok(m) || !dim_ok(k)) return CERE_ERR_ARGUMENT;
    const cudaError_t e = k2m::quantize_x(static_cast<const __nv_bfloat16 *>(x), uint32_t(m), uint32_t(k),
                                          static_cast<uint8_t *>(q), static_cast<uint8_t *>(scales), stream);
    if (e == cudaErrorInvalidValue) return CERE_ERR_ARGUMENT;
    return e == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

// The blocked layout (k2m::blocked_scale_bytes(n, k / 32) bytes at `blocked`) of row-major UE8M0 scales [n, k / 32]
// (k2m::repack_scales).
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2m_repack_scales(const void *scales, int64_t n,
                                                                                  int64_t k, void *blocked,
                                                                                  cudaStream_t stream) {
    if (!dim_ok(n) || !dim_ok(k)) return CERE_ERR_ARGUMENT;
    const cudaError_t e = k2m::repack_scales(static_cast<const uint8_t *>(scales), uint32_t(n), uint32_t(k),
                                             static_cast<uint8_t *>(blocked), stream);
    if (e == cudaErrorInvalidValue) return CERE_ERR_ARGUMENT;
    return e == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

// y [m, n] = plan.run(x, w, m, y, stream, cls) through a plan of (n, k, out):
// - rows 0: an init plan of max_m = plan_max_m (max(m, cls), at least 17 under small_m 2) with the executor's pins
//   (k2_ktest.h plan_init); rows in [1,
//   GemmPlan::kMaxRows] (m <= rows): the rows plan GemmPlan::init_rows(instantiated (n, k, max_m 1, out), rows,
//   fused_allowed, paths) (paths -1: init_rows' default, the executor's rows plans).
// - small_m 1 / 0: use_small_m (the k2_gemm lever's small-M path or cuBLASLt), after init. 2: the lever value
//   small_m_a8: use_small_m, and on a rows 0 plan init_a8 (its failure is the call's status) and use_a8.
// - scales null: w is BF16 [n, k] and the plan has no MXFP8 table (the reference); else w is E4M3 [n, k] with UE8M0
//   `scales` [n, k / 32], the only entry of the plan's MXFP8 table, whose staging buffer is `staging` (bytes; null / 0
//   allowed: then a staging-path call is refused CERE_ERR_UNSUPPORTED).
// - cls 0: the call's own m; else the row count the choice is made for (GemmPlan::run).
// iters 1: one call; > 1: one untimed call then iters - 1 cold-L2 calls (l2_flush.h cold_us), us host double [2] =
// median and minimum µs per call. Host int64 info [1] (nullable): how the BF16 plan runs that m (0 cuBLASLt, 1 m
// single-row calls, 2 one k2v::gemv launch, 3 one strided-batched cuBLASLt call, 4 the small-M kernel); 5 when the
// MXFP8 weight takes the W8A8 path.
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2m_plan(
    const void *x, const void *w, const void *scales, int64_t m, int64_t n, int64_t k, int64_t rows,
    int64_t fused_allowed, int64_t paths, int64_t out_f32, int64_t small_m, int64_t cls, void *staging,
    int64_t staging_bytes, int64_t iters, void *y, void *info, void *us, cudaStream_t stream) {
    if (!dim_ok(m) || !dim_ok(n) || !dim_ok(k) || rows < 0 || rows > GemmPlan::kMaxRows || (rows > 0 && m > rows) ||
        fused_allowed < 0 || fused_allowed > UINT32_MAX || paths < -1 || paths > GemmPlan::kRowsAllPaths ||
        out_f32 < 0 || out_f32 > 1 || small_m < 0 || small_m > 2 || cls < 0 || (cls != 0 && cls < m) ||
        cls > INT32_MAX || staging_bytes < 0 || iters < 1 || iters > 10'000 || (iters > 1 && us == nullptr) ||
        (rows > 0 && cls > rows)) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = cere::ktest::gemm_handle();
    if (handle == nullptr) return CERE_ERR_INTERNAL;
    const auto out = out_f32 != 0 ? GemmOut::kF32 : GemmOut::kBF16;
    const auto um = uint32_t(m);
    const auto ucls = uint32_t(cls);
    GemmPlan instantiated;
    GemmPlan rows_plan;
    void *workspace = nullptr;
    cere_status s = CERE_OK;
    if (rows == 0) {
        s = cere::ktest::plan_init(&instantiated, handle, uint32_t(n), uint32_t(k), plan_max_m(m, cls, small_m), out,
                                   &workspace, stream);
    } else {
        s = instantiated.init(handle, uint32_t(n), uint32_t(k), 1, out);
        if (s == CERE_OK) {
            s = paths < 0 ? rows_plan.init_rows(handle, instantiated, uint32_t(rows), uint32_t(fused_allowed))
                          : rows_plan.init_rows(handle, instantiated, uint32_t(rows), uint32_t(fused_allowed),
                                                uint32_t(paths));
        }
    }
    GemmPlan &plan = rows == 0 ? instantiated : rows_plan;
    k2m::Table table;
    if (s == CERE_OK) {
        plan.use_small_m(small_m != 0);
        if (small_m != 0 && !plan.small_m()) s = CERE_ERR_UNSUPPORTED;
    }
    if (s == CERE_OK && small_m == 2 && rows == 0) {
        s = plan.init_a8();
        plan.use_a8(true);
    }
    if (s == CERE_OK && scales != nullptr) {
        table.add(w, static_cast<const uint8_t *>(scales), uint32_t(n), uint32_t(k));
        table.set_staging(static_cast<__nv_bfloat16 *>(staging), size_t(staging_bytes));
        plan.use_mxfp8(&table);
    }
    if (s == CERE_OK && info != nullptr) {
        const uint32_t sel = cls != 0 ? ucls : um;
        int64_t how = 0;
        if (rows > 0) {
            how = plan.gemv_rows(um) ? 2 : plan.batched_rows(um) ? 3 : plan.per_row(um) ? 1 : plan.small_m() ? 4 : 0;
        } else if (plan.a8() && scales != nullptr && sel > cere::kernels::k2s::kMaxM) {
            how = 5;
        } else if (plan.small_m() && sel <= cere::kernels::k2s::kMaxM) {
            how = 4;
        }
        *static_cast<int64_t *>(info) = how;
    }
    if (s == CERE_OK) {
        s = run_or_time(iters, us, stream, [&] {
            return plan.run(static_cast<const __nv_bfloat16 *>(x), static_cast<const __nv_bfloat16 *>(w), um, y,
                            stream, ucls);
        });
    }
    if (workspace != nullptr && cudaFreeAsync(workspace, stream) != cudaSuccess && s == CERE_OK) {
        s = CERE_ERR_INTERNAL;
    }
    return s;
}

// kernels::gemm_segments over `count` (<= 4) weights sharing x [m, k] (host int64 arrays w_ptrs, scale_ptrs, ns,
// y_ptrs): per weight, rows 0: one init plan of max_m = plan_max_m (as cere_ktest_k2m_plan); rows in [1,
// GemmPlan::kMaxRows] (m <= rows): the
// executor's rows plan init_rows(rows) of an init plan of max_m 1. small_m as cere_ktest_k2m_plan (2 on rows 0 plans:
// init_a8 on each, its failure the call's status, and use_a8); scale_ptrs[i] 0: weight i is BF16 (no table entry),
// else E4M3 with those UE8M0 scales, listed in the one MXFP8 table every plan holds, with `staging` (bytes) as its
// staging buffer. `cls` as cere_ktest_k2m_plan (0: m; rows 0 only). iters / us as cere_ktest_k2m_plan.
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2m_segments(
    const void *x, int64_t m, int64_t k, int64_t count, const int64_t *w_ptrs, const int64_t *scale_ptrs,
    const int64_t *ns, const int64_t *y_ptrs, int64_t rows, int64_t out_f32, int64_t small_m, int64_t cls,
    void *staging, int64_t staging_bytes, int64_t iters, void *us, cudaStream_t stream) {
    constexpr int64_t kMax = 4;
    if (!dim_ok(m) || !dim_ok(k) || count < 1 || count > kMax || w_ptrs == nullptr || scale_ptrs == nullptr ||
        ns == nullptr || y_ptrs == nullptr || rows < 0 || rows > GemmPlan::kMaxRows || (rows > 0 && m > rows) ||
        out_f32 < 0 || out_f32 > 1 || small_m < 0 || small_m > 2 || cls < 0 || (cls != 0 && cls < m) ||
        (cls != 0 && rows > 0) || cls > INT32_MAX || staging_bytes < 0 || iters < 1 || iters > 10'000 ||
        (iters > 1 && us == nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = cere::ktest::gemm_handle();
    if (handle == nullptr) return CERE_ERR_INTERNAL;
    const auto out = out_f32 != 0 ? GemmOut::kF32 : GemmOut::kBF16;
    k2m::Table table;
    table.set_staging(static_cast<__nv_bfloat16 *>(staging), size_t(staging_bytes));
    GemmPlan instantiated[kMax];
    GemmPlan rows_plans[kMax];
    void *workspaces[kMax] = {};
    const GemmPlan *pp[kMax] = {};
    const __nv_bfloat16 *w[kMax] = {};
    void *y[kMax] = {};
    cere_status s = CERE_OK;
    for (int64_t i = 0; i < count && s == CERE_OK; ++i) {
        if (!dim_ok(ns[i])) {
            s = CERE_ERR_ARGUMENT;
            break;
        }
        if (rows == 0) {
            s = cere::ktest::plan_init(&instantiated[i], handle, uint32_t(ns[i]), uint32_t(k),
                                       plan_max_m(m, cls, small_m), out, &workspaces[i], stream);
        } else if ((s = instantiated[i].init(handle, uint32_t(ns[i]), uint32_t(k), 1, out)) == CERE_OK) {
            s = rows_plans[i].init_rows(handle, instantiated[i], uint32_t(rows));
        }
        if (s != CERE_OK) break;
        GemmPlan &plan = rows == 0 ? instantiated[i] : rows_plans[i];
        if (scale_ptrs[i] != 0) {
            table.add(reinterpret_cast<const void *>(w_ptrs[i]), reinterpret_cast<const uint8_t *>(scale_ptrs[i]),
                      uint32_t(ns[i]), uint32_t(k));
        }
        plan.use_small_m(small_m != 0);
        if (small_m == 2 && rows == 0) {
            if ((s = plan.init_a8()) != CERE_OK) break;
            plan.use_a8(true);
        }
        plan.use_mxfp8(&table);
        pp[i] = &plan;
        w[i] = reinterpret_cast<const __nv_bfloat16 *>(w_ptrs[i]);
        y[i] = reinterpret_cast<void *>(y_ptrs[i]);
    }
    if (s == CERE_OK) {
        s = run_or_time(iters, us, stream, [&] {
            return cere::kernels::gemm_segments(pp, w, y, uint32_t(count), static_cast<const __nv_bfloat16 *>(x),
                                                uint32_t(m), stream, uint32_t(cls));
        });
    }
    for (void *ws : workspaces) {
        if (ws != nullptr && cudaFreeAsync(ws, stream) != cudaSuccess && s == CERE_OK) s = CERE_ERR_INTERNAL;
    }
    return s;
}
