// AmdAmd PAR-6 step 2b: see k2_gemm_hip.h.
//
// hipBLASLt is column-major. The row-major y[m, n] = x[m, k] · w[n, k]^T is the column-major
// D (n x m, ld n) = op(A) · B with A = w seen as col-major (k x n, ld k), op(A) = T, and B = x seen as col-major
// (k x m, ld k), op(B) = N: a GEMM of M' = n, N' = m, K' = k.
#include "k2_gemm_hip.h"

#include <hipblaslt/hipblaslt-ext.hpp>

#include "k2_mgemv.h"
#include "k2_wmma.h"

namespace cere::hipk2 {
namespace {

uint32_t next_pow2(uint32_t v) {
    uint32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

}  // namespace

GemmPlan::~GemmPlan() {
    if (desc_) hipblasLtMatmulDescDestroy(desc_);
}

hipblasStatus_t GemmPlan::layouts(uint32_t m, hipblasLtMatrixLayout_t *a, hipblasLtMatrixLayout_t *b,
                                  hipblasLtMatrixLayout_t *c) const {
    hipblasStatus_t s = hipblasLtMatrixLayoutCreate(a, HIP_R_16BF, k_, n_, k_);
    if (s == HIPBLAS_STATUS_SUCCESS) s = hipblasLtMatrixLayoutCreate(b, HIP_R_16BF, k_, m, k_);
    if (s == HIPBLAS_STATUS_SUCCESS)
        s = hipblasLtMatrixLayoutCreate(c, out_ == GemmOut::kBF16 ? HIP_R_16BF : HIP_R_32F, n_, m, n_);
    return s;
}

cere_status GemmPlan::init(hipblasLtHandle_t handle, uint32_t n, uint32_t k, GemmOut out, uint32_t max_m) {
    if (!handle || n == 0 || k == 0 || max_m == 0 || desc_) return CERE_ERR_ARGUMENT;
    handle_ = handle;
    n_ = n;
    k_ = k;
    out_ = out;
    max_m_ = max_m;
    if (hipblasLtMatmulDescCreate(&desc_, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS)
        return CERE_ERR_INTERNAL;
    const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(desc_, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
    hipblasLtMatmulDescSetAttribute(desc_, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
    hipblasLtMatmulPreference_t pref = nullptr;
    if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) return CERE_ERR_INTERNAL;
    const uint64_t no_workspace = 0;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &no_workspace,
                                          sizeof no_workspace);
    const float alpha = 1.f, beta = 0.f;
    cere_status result = CERE_OK;
    for (uint32_t hi = 1, lo = 1; lo <= next_pow2(max_m); lo = hi + 1, hi <<= 1) {
        hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, bl = nullptr, cl = nullptr, al = nullptr;
        constexpr int kRequested = 16;
        hipblasLtMatmulHeuristicResult_t heur[kRequested];
        int returned = 0;
        bool pinned = false;
        if (layouts(hi, &a, &b, &c) == HIPBLAS_STATUS_SUCCESS &&
            layouts(lo, &al, &bl, &cl) == HIPBLAS_STATUS_SUCCESS &&
            hipblasLtMatmulAlgoGetHeuristic(handle_, desc_, a, b, c, c, pref, kRequested, heur, &returned) ==
                HIPBLAS_STATUS_SUCCESS) {
            for (int i = 0; i < returned && !pinned; ++i) {
                if (heur[i].state != HIPBLAS_STATUS_SUCCESS || heur[i].workspaceSize != 0) continue;
                size_t ws_hi = 1, ws_lo = 1;
                if (hipblaslt_ext::matmulIsAlgoSupported(handle_, desc_, &alpha, a, b, &beta, c, c, heur[i].algo,
                                                         ws_hi) != HIPBLAS_STATUS_SUCCESS ||
                    hipblaslt_ext::matmulIsAlgoSupported(handle_, desc_, &alpha, al, bl, &beta, cl, cl,
                                                         heur[i].algo, ws_lo) != HIPBLAS_STATUS_SUCCESS ||
                    ws_hi != 0 || ws_lo != 0)
                    continue;
                BucketPin pin{lo, hi, heur[i].algo, hipblaslt_ext::getIndexFromAlgo(heur[i].algo), {}};
                pin.kernel = hipblaslt_ext::getKernelNameFromAlgo(handle_, heur[i].algo);
                pins_.push_back(pin);
                pinned = true;
            }
        }
        for (auto *l : {a, b, c, al, bl, cl})
            if (l) hipblasLtMatrixLayoutDestroy(l);
        if (!pinned) {
            result = CERE_ERR_UNSUPPORTED;
            break;
        }
        if (hi >= next_pow2(max_m)) break;
    }
    hipblasLtMatmulPreferenceDestroy(pref);
    return result;
}

cere_status GemmPlan::run(const void *x, const void *w, void *y, uint32_t m, hipStream_t stream, uint32_t cls) const {
    if (!desc_ || m == 0 || m > max_m_ || !x || !w || !y || (cls != 0 && (cls < m || cls > max_m_))) return CERE_ERR_ARGUMENT;
    const uint32_t choice = cls != 0 ? cls : m;
    if (choice >= kWmmaMinM && k_ % 64 == 0)  // shift-invariant prefill GEMM (k2_gemm_hip.h)
        return gemm_wmma(static_cast<const __hip_bfloat16 *>(x), static_cast<const __hip_bfloat16 *>(w), y, m, n_, k_,
                         out_ == GemmOut::kBF16 ? WmmaOut::kBF16 : WmmaOut::kF32, stream);
    if (choice <= kGemvMaxM && k_ % 16 == 0)  // row-independent small-M GEMV (k2_mgemv.h)
        return gemv(static_cast<const __hip_bfloat16 *>(x), w, nullptr, y, m, n_, k_, out_ == GemmOut::kF32, stream);
    return run_blas(x, w, y, m, stream, choice);
}

cere_status GemmPlan::run_blas(const void *x, const void *w, void *y, uint32_t m, hipStream_t stream,
                               uint32_t choice) const {
    if (!desc_ || m == 0 || m > max_m_ || !x || !w || !y) return CERE_ERR_ARGUMENT;
    if (choice == 0) choice = m;
    const BucketPin *pin = nullptr;
    for (const auto &p : pins_)
        if (choice >= p.m_lo && choice <= p.m_hi) pin = &p;
    if (!pin) return CERE_ERR_INTERNAL;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cere_status st = CERE_OK;
    if (layouts(m, &a, &b, &c) != HIPBLAS_STATUS_SUCCESS) st = CERE_ERR_INTERNAL;
    const float alpha = 1.f, beta = 0.f;
    if (st == CERE_OK && hipblasLtMatmul(handle_, desc_, &alpha, w, a, x, b, &beta, y, c, y, c, &pin->algo, nullptr,
                                         0, stream) != HIPBLAS_STATUS_SUCCESS)
        st = CERE_ERR_INTERNAL;
    for (auto *l : {a, b, c})
        if (l) hipblasLtMatrixLayoutDestroy(l);
    return st;
}

}  // namespace cere::hipk2
