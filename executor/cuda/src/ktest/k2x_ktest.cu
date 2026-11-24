// K2 fast small-M kernel-test entry points (executor/tests/gpu/test_kernels_k2_fast.py; kernels/k2_fast.h): one MXFP8
// launch of the kernel k2x replaces (impl 0), of k2x (impl 1), of k2x's dispatch (impl 2) or of k2x's resident layout
// (impl 4), or the BF16 reference k2s::gemm (impl 3) on the same arguments, optionally timed cold (an l2_flush of ktest::kL2FlushBytes before every
// timed launch). Device pointers, int64 sizes (host int64 arrays where named), the stream last; 0 = success, else a
// cere_status.
#include "cere_executor.h"
#include "kernels/k2_fast.h"
#include "kernels/k2_small_m.h"
#include "ktest/l2_flush.h"

#include <cstdint>

namespace {

namespace k2s = cere::kernels::k2s;
namespace k2x = cere::kernels::k2x;

constexpr int64_t kMaxIters = 10'000;

bool dim_ok(int64_t v) { return v > 0 && v <= INT32_MAX; }

cere_status status_of(cudaError_t e) {
    if (e == cudaErrorInvalidValue) return CERE_ERR_ARGUMENT;
    return e == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

// tune: host int64 [3] (impl 1: q, lock, stages, 0 = the default's; impl 4: q, kc, stages, all 0 = the default) or
// nullptr (the default layout).
bool tune_of(const int64_t *tune, uint32_t (&t)[3]) {
    t[0] = t[1] = t[2] = 0;
    if (tune == nullptr) return true;
    for (int i = 0; i < 3; ++i) {
        if (tune[i] < 0 || tune[i] > 16) return false;
        t[i] = uint32_t(tune[i]);
    }
    return true;
}

// Runs `call()` iters times on `stream`. us (host f64 [iters]) nullptr: back to back, untimed; else each launch is
// preceded by an l2_flush (a 512 MiB buffer of its own) and us[i] = its event-timed µs.
template <typename Call>
cere_status run(int64_t iters, double *us, cudaStream_t stream, Call call) {
    if (iters < 1 || iters > kMaxIters) return CERE_ERR_ARGUMENT;
    if (us == nullptr) {
        for (int64_t i = 0; i < iters; ++i) {
            const cere_status s = status_of(call());
            if (s != CERE_OK) return s;
        }
        return CERE_OK;
    }
    void *buf = nullptr;
    if (cudaMallocAsync(&buf, cere::ktest::kL2FlushBytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cere_status s = cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess
                        ? CERE_OK
                        : CERE_ERR_INTERNAL;
    for (int64_t i = 0; i < iters && s == CERE_OK; ++i) {
        float ms = 0.0f;
        if (cere::ktest::l2_flush(buf, cere::ktest::kL2FlushBytes, stream) != cudaSuccess ||
            cudaEventRecord(start, stream) != cudaSuccess) {
            s = CERE_ERR_INTERNAL;
            break;
        }
        s = status_of(call());
        if (s == CERE_OK && (cudaEventRecord(stop, stream) != cudaSuccess || cudaEventSynchronize(stop) != cudaSuccess ||
                             cudaEventElapsedTime(&ms, start, stop) != cudaSuccess)) {
            s = CERE_ERR_INTERNAL;
        }
        us[i] = 1000.0 * double(ms);
    }
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    if (cudaFreeAsync(buf, stream) != cudaSuccess && s == CERE_OK) s = CERE_ERR_INTERNAL;
    return s;
}

}  // namespace

// One launch over `count` weights sharing x [m, k]: host int64 arrays w_ptrs, scale_ptrs, ns, y_ptrs. impl 0..2: w
// E4M3 [n, k] with UE8M0 scales [n, k / 32]; impl 0 k2s::gemm_mxfp8 (the kernel k2x replaces), 1 k2x::gemm (layout
// `tune`), 2 k2x::gemm_mxfp8 (the dispatch the executor runs), 4 k2x::resident (layout `tune`). impl 3: k2s::gemm on
// BF16 w [n, k] (the upcast reference; scale_ptrs unused, may be nullptr); impl 5 the same with `early` (the MoE
// router's launch, BeatSglang S4). tune: impl 1 and 4 only. out 0 BF16, 1 F32. iters, us: run().
extern "C" __attribute__((visibility("default"))) int cere_ktest_k2x_gemm(
    const void *x, int64_t m, int64_t k, int64_t count, const int64_t *w_ptrs, const int64_t *scale_ptrs,
    const int64_t *ns, const int64_t *y_ptrs, int64_t out, int64_t impl, const int64_t *tune, int64_t iters, void *us,
    cudaStream_t stream) {
    uint32_t t[3];
    if (x == nullptr || !dim_ok(m) || !dim_ok(k) || count < 1 || count > k2x::kMaxSegments || w_ptrs == nullptr ||
        (impl != 3 && impl != 5 && scale_ptrs == nullptr) || ns == nullptr || y_ptrs == nullptr || out < 0 ||
        out > 1 || impl < 0 || impl > 5 || (impl != 1 && impl != 4 && tune != nullptr) || !tune_of(tune, t)) {
        return CERE_ERR_ARGUMENT;
    }
    for (int64_t i = 0; i < count; ++i) {
        if (!dim_ok(ns[i])) return CERE_ERR_ARGUMENT;
    }
    const auto *xb = static_cast<const __nv_bfloat16 *>(x);
    const auto mm = uint32_t(m);
    const auto kk = uint32_t(k);
    const auto n = uint32_t(count);
    if (impl == 3 || impl == 5) {
        k2s::Segment segs[k2s::kMaxSegments];
        for (uint32_t i = 0; i < n; ++i) {
            segs[i] = k2s::Segment{reinterpret_cast<const __nv_bfloat16 *>(w_ptrs[i]),
                                   reinterpret_cast<void *>(y_ptrs[i]), uint32_t(ns[i])};
        }
        return run(iters, static_cast<double *>(us), stream,
                   [&] { return k2s::gemm(xb, mm, kk, segs, n, out == 1, stream, impl == 5); });
    }
    k2s::Mxfp8Segment segs[k2s::kMaxSegments];
    for (uint32_t i = 0; i < n; ++i) {
        segs[i] = k2s::Mxfp8Segment{reinterpret_cast<const uint8_t *>(w_ptrs[i]),
                                    reinterpret_cast<const uint8_t *>(scale_ptrs[i]),
                                    reinterpret_cast<void *>(y_ptrs[i]), uint32_t(ns[i])};
    }
    const k2x::Tune fast{t[0], t[1], t[2]};
    const k2x::ResidentTune resident{t[0], t[1], t[2]};
    return run(iters, static_cast<double *>(us), stream, [&] {
        return impl == 0   ? k2s::gemm_mxfp8(xb, mm, kk, segs, n, out == 1, stream)
               : impl == 1 ? k2x::gemm(xb, mm, kk, segs, n, out == 1, stream, fast)
               : impl == 4 ? k2x::resident(xb, mm, kk, segs, n, out == 1, stream, resident)
                           : k2x::gemm_mxfp8(xb, mm, kk, segs, n, out == 1, stream);
    });
}
