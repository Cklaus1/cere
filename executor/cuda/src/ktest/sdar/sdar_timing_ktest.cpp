// SelfTune ST6 S2 (selftune design note ST6, D-ST6-9): cold-L2 timing of family sdar's kernels, its GEMM and
// cuBLASLt's heuristic candidates for a GEMM shape (executor/tests/gpu/test_sdar_kernels_timing.py; prototypes in its
// docstring). Each `_cold` entry point takes its kernel entry point's arguments (sdar_ktest.cu), then `iters` and `us`
// (host double [2]): one untimed call, then `iters` calls each after an L2 flush (ktest/l2_flush.h cold_us), us = the
// median and minimum µs of one call. A refusal of the wrapped entry point is returned unchanged. Compiled only into
// a CERE_MODEL=<sdar descriptor> library's libcere_ktest; libcere_executor does not change.
#include <cstdint>
#include <vector>

#include <cublasLt.h>
#include <cuda_runtime.h>

#include "cere_executor.h"
#include "kernels/sdar/sdar_gemm.h"
#include "ktest/l2_flush.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

CERE_KTEST int cere_ktest_sdar_embed(const void *table, const void *ids, int64_t n, int64_t hidden, void *dst,
                                     cudaStream_t stream);
CERE_KTEST int cere_ktest_sdar_rmsnorm(const void *x, int64_t x_stride, const void *w, float eps, int64_t rows,
                                       int64_t dim, void *y, int64_t y_stride, cudaStream_t stream);
CERE_KTEST int cere_ktest_sdar_rope(void *x, int64_t stride, int64_t rows, int64_t heads, int64_t pos0, float theta,
                                    cudaStream_t stream);
CERE_KTEST int cere_ktest_sdar_kv_store(const void *k, int64_t k_stride, const void *v, int64_t v_stride, int64_t n,
                                        int64_t kv_heads, void *k_cache, void *v_cache, int64_t pos,
                                        cudaStream_t stream);
CERE_KTEST int cere_ktest_sdar_attention(const void *q, int64_t q_stride, const void *k_cache, const void *v_cache,
                                         int64_t past, const void *new_k, int64_t new_k_stride, const void *new_v,
                                         int64_t new_v_stride, int64_t n, int64_t heads, int64_t kv_heads, void *dst,
                                         int64_t out_stride, int64_t mask, cudaStream_t stream);
CERE_KTEST int cere_ktest_sdar_silu_mul(const void *gate_up, int64_t rows, int64_t inter, void *dst,
                                        cudaStream_t stream);
CERE_KTEST int cere_ktest_sdar_add(void *x, const void *y, int64_t n, cudaStream_t stream);
CERE_KTEST int cere_ktest_sdar_top_tie(const void *logits, int64_t stride, int64_t rows, int64_t vocab, void *dst,
                                       int64_t fp32, cudaStream_t stream);
CERE_KTEST int cere_ktest_sdar_unmask_select(const void *top, const void *masked, int64_t rows, void *index,
                                             cudaStream_t stream);

namespace {

using cere::ktest::cold_us;
using cere::ktest::kL2FlushBytes;

// `call` cold-L2 timed into us (host double [2]); the flush buffer is stream-ordered memory freed on return.
template <typename Call>
int timed(int64_t iters, void *us, cudaStream_t stream, Call call) {
    if (iters < 1 || us == nullptr) return CERE_ERR_ARGUMENT;
    void *flush = nullptr;
    if (cudaMallocAsync(&flush, kL2FlushBytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    cere_status s = cold_us(flush, iters, static_cast<double *>(us), stream, call);
    if (cudaFreeAsync(flush, stream) != cudaSuccess && s == CERE_OK) s = CERE_ERR_INTERNAL;
    if (cudaStreamSynchronize(stream) != cudaSuccess && s == CERE_OK) s = CERE_ERR_INTERNAL;
    return s;
}

bool ok(cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS; }

bool gemm_dims_ok(int64_t m, int64_t n, int64_t k) {
    return m >= 1 && n >= 1 && k >= 1 && m <= INT32_MAX && n <= INT32_MAX && k <= INT32_MAX;
}

}  // namespace

CERE_KTEST int cere_ktest_sdar_embed_cold(const void *table, const void *ids, int64_t n, int64_t hidden, void *dst,
                                          int64_t iters, void *us, cudaStream_t stream) {
    return timed(iters, us, stream, [&] { return cere_ktest_sdar_embed(table, ids, n, hidden, dst, stream); });
}

CERE_KTEST int cere_ktest_sdar_rmsnorm_cold(const void *x, int64_t x_stride, const void *w, float eps, int64_t rows,
                                            int64_t dim, void *y, int64_t y_stride, int64_t iters, void *us,
                                            cudaStream_t stream) {
    return timed(iters, us, stream,
                 [&] { return cere_ktest_sdar_rmsnorm(x, x_stride, w, eps, rows, dim, y, y_stride, stream); });
}

CERE_KTEST int cere_ktest_sdar_rope_cold(void *x, int64_t stride, int64_t rows, int64_t heads, int64_t pos0,
                                         float theta, int64_t iters, void *us, cudaStream_t stream) {
    return timed(iters, us, stream, [&] { return cere_ktest_sdar_rope(x, stride, rows, heads, pos0, theta, stream); });
}

CERE_KTEST int cere_ktest_sdar_kv_store_cold(const void *k, int64_t k_stride, const void *v, int64_t v_stride,
                                             int64_t n, int64_t kv_heads, void *k_cache, void *v_cache, int64_t pos,
                                             int64_t iters, void *us, cudaStream_t stream) {
    return timed(iters, us, stream, [&] {
        return cere_ktest_sdar_kv_store(k, k_stride, v, v_stride, n, kv_heads, k_cache, v_cache, pos, stream);
    });
}

CERE_KTEST int cere_ktest_sdar_attention_cold(const void *q, int64_t q_stride, const void *k_cache,
                                              const void *v_cache, int64_t past, const void *new_k,
                                              int64_t new_k_stride, const void *new_v, int64_t new_v_stride,
                                              int64_t n, int64_t heads, int64_t kv_heads, void *dst,
                                              int64_t out_stride, int64_t iters, void *us, cudaStream_t stream) {
    return timed(iters, us, stream, [&] {
        return cere_ktest_sdar_attention(q, q_stride, k_cache, v_cache, past, new_k, new_k_stride, new_v,
                                         new_v_stride, n, heads, kv_heads, dst, out_stride, 0, stream);
    });
}

CERE_KTEST int cere_ktest_sdar_silu_mul_cold(const void *gate_up, int64_t rows, int64_t inter, void *dst,
                                             int64_t iters, void *us, cudaStream_t stream) {
    return timed(iters, us, stream, [&] { return cere_ktest_sdar_silu_mul(gate_up, rows, inter, dst, stream); });
}

CERE_KTEST int cere_ktest_sdar_add_cold(void *x, const void *y, int64_t n, int64_t iters, void *us,
                                        cudaStream_t stream) {
    return timed(iters, us, stream, [&] { return cere_ktest_sdar_add(x, y, n, stream); });
}

CERE_KTEST int cere_ktest_sdar_top_tie_cold(const void *logits, int64_t stride, int64_t rows, int64_t vocab,
                                            void *dst, int64_t iters, void *us, cudaStream_t stream) {
    return timed(iters, us, stream,
                 [&] { return cere_ktest_sdar_top_tie(logits, stride, rows, vocab, dst, 0, stream); });
}

CERE_KTEST int cere_ktest_sdar_unmask_select_cold(const void *top, const void *masked, int64_t rows, void *index,
                                                  int64_t iters, void *us, cudaStream_t stream) {
    return timed(iters, us, stream,
                 [&] { return cere_ktest_sdar_unmask_select(top, masked, rows, index, stream); });
}

// y [m, n] = x [m, k] · w [n, k]^T through cere::sdar::Gemm (the model's plan: the heuristic's first zero-workspace
// algorithm for m), y BF16 (f32 0) or FP32 (f32 1).
CERE_KTEST int cere_ktest_sdar_gemm_cold(const void *x, const void *w, int64_t m, int64_t n, int64_t k, void *y,
                                         int64_t f32, int64_t iters, void *us, cudaStream_t stream) {
    if (!gemm_dims_ok(m, n, k) || (f32 != 0 && f32 != 1)) return CERE_ERR_ARGUMENT;
    cublasLtHandle_t handle = nullptr;
    if (!ok(cublasLtCreate(&handle))) return CERE_ERR_INTERNAL;
    int s;
    {
        cere::sdar::Gemm gemm;
        s = gemm.init(handle, uint32_t(n), uint32_t(k), uint32_t(m),
                      f32 ? cere::sdar::GemmOut::kF32 : cere::sdar::GemmOut::kBF16);
        if (s == CERE_OK) {
            s = timed(iters, us, stream, [&] {
                return gemm.run(static_cast<const __nv_bfloat16 *>(x), static_cast<const __nv_bfloat16 *>(w),
                                uint32_t(m), y, stream);
            });
        }
    }
    cublasLtDestroy(handle);
    return s;
}

// cuBLASLt's heuristic candidates for the same matmul (Gemm::init's descriptor and layouts), up to `top_n` of them
// with at most `workspace_bytes` of workspace, in the heuristic's order (candidate 0 with workspace 0 is the one Gemm
// picks). found (host int64 [1]) = the number returned; per candidate i < found: us[2i], us[2i+1] = its cold-L2
// median and minimum µs, -1 when cublasLtMatmul refuses it; ws (host int64 [top_n]) = its workspace bytes.
CERE_KTEST int cere_ktest_sdar_gemm_algos(const void *x, const void *w, int64_t m, int64_t n, int64_t k, void *y,
                                          int64_t f32, int64_t workspace_bytes, int64_t top_n, int64_t iters,
                                          void *us, void *found, void *ws, cudaStream_t stream) {
    if (!gemm_dims_ok(m, n, k) || (f32 != 0 && f32 != 1) || workspace_bytes < 0 || top_n < 1 || top_n > 64 ||
        iters < 1 || us == nullptr || found == nullptr || ws == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    double *out = static_cast<double *>(us);
    int64_t *nfound = static_cast<int64_t *>(found);
    int64_t *wsb = static_cast<int64_t *>(ws);
    *nfound = 0;
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    void *workspace = nullptr;
    const size_t limit = size_t(workspace_bytes);
    const cublasOperation_t trans = CUBLAS_OP_T, none = CUBLAS_OP_N;
    const cudaDataType_t dtype = f32 ? CUDA_R_32F : CUDA_R_16BF;
    std::vector<cublasLtMatmulHeuristicResult_t> results(static_cast<size_t>(top_n));
    int got = 0;
    int s = CERE_OK;
    if (!ok(cublasLtCreate(&handle)) || !ok(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F)) ||
        !ok(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans))) ||
        !ok(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &none, sizeof(none))) ||
        !ok(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, uint64_t(k), uint64_t(n), k)) ||
        !ok(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, uint64_t(k), uint64_t(m), k)) ||
        !ok(cublasLtMatrixLayoutCreate(&d, dtype, uint64_t(n), uint64_t(m), n)) ||
        !ok(cublasLtMatmulPreferenceCreate(&pref)) ||
        !ok(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit,
                                                 sizeof(limit)))) {
        s = CERE_ERR_INTERNAL;
    } else if (!ok(cublasLtMatmulAlgoGetHeuristic(handle, desc, a, b, d, d, pref, int(top_n), results.data(), &got))) {
        s = CERE_ERR_UNSUPPORTED;
    } else if (limit > 0 && cudaMalloc(&workspace, limit) != cudaSuccess) {
        workspace = nullptr;
        s = CERE_ERR_CAPACITY;
    }
    const float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < got && s == CERE_OK; ++i) {
        const cublasLtMatmulHeuristicResult_t &r = results[size_t(i)];
        wsb[i] = int64_t(r.workspaceSize);
        const int t = timed(iters, out + 2 * i, stream, [&] {
            return ok(cublasLtMatmul(handle, desc, &alpha, w, a, x, b, &beta, y, d, y, d, &r.algo, workspace,
                                     r.workspaceSize, stream))
                       ? CERE_OK
                       : CERE_ERR_INTERNAL;
        });
        if (t == CERE_ERR_INTERNAL) {
            out[2 * i] = out[2 * i + 1] = -1.0;
            if (cudaGetLastError() != cudaSuccess) s = CERE_ERR_INTERNAL;  // a sticky launch error, not a refusal
        } else if (t != CERE_OK) {
            s = t;
        }
    }
    if (s == CERE_OK) *nfound = got;
    if (workspace != nullptr) cudaFree(workspace);
    if (pref != nullptr) cublasLtMatmulPreferenceDestroy(pref);
    if (d != nullptr) cublasLtMatrixLayoutDestroy(d);
    if (b != nullptr) cublasLtMatrixLayoutDestroy(b);
    if (a != nullptr) cublasLtMatrixLayoutDestroy(a);
    if (desc != nullptr) cublasLtMatmulDescDestroy(desc);
    if (handle != nullptr) cublasLtDestroy(handle);
    return s;
}
