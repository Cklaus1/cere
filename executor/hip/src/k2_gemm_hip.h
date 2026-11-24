// AmdAmd PAR-6 step 2b: the K2 BF16 GEMM of the native executor on HIP (hipBLASLt), under CUDA K2's determinism
// contract (executor/cuda/src/kernels/k2_gemm.h):
//
//   y[m, n] = x[m, k] · w[n, k]^T ; x BF16 row-major [m, k]; w BF16 in the checkpoint layout [n, k]; y BF16 or F32
//   row-major [m, n]; FP32 accumulation, alpha 1, beta 0.
//
// Determinism: the algorithm is pinned when the plan is created, never chosen at run time. M is split into
// power-of-two buckets (1, 2, 3-4, 5-8, ..., up to next_pow2(max_m)); each bucket pins the first hipBLASLt heuristic
// result for M = its upper bound that runs with zero workspace (no split-K partial buffers) and that
// hipblaslt_ext::matmulIsAlgoSupported accepts at both ends of the bucket. The pin records hipBLASLt's solution
// index (hipblaslt_ext::getIndexFromAlgo) and kernel name. So the same (n, k, out, m) always runs the same kernel
// and repeated calls are bit-identical (asserted by the ktest). Buckets depend on m only.
//
// Not ported (CUDA K2 parts with NVIDIA-only hardware): the small-M tensor-core kernel (mma.sync m16n8k16, cp.async,
// clusters; batch invariance for m <= 16), the prefill pins recorded from torch's cuBLAS dispatch, the FP8/MXFP8
// paths.
//
// Rows class above kWmmaMinM (cls, or m when cls is 0; CUDA K2's kPinMinM boundary: m <= 16 keeps its bucket) with
// k % 64 == 0 run k2_wmma.h's GEMM instead of hipBLASLt: hipBLASLt on gfx1151 is not shift invariant (a row's bits
// depend on where the call's rows start; amd-amd design note K2-SHIFT), which breaks the D15 contract that a call
// from row 64 c continues the longer call exactly. k2_wmma gives every row the same bits in any call, at hipBLASLt's
// speed on the model's prefill shapes (k2w_test). Row classes <= 16 (k % 16 == 0) run k2_mgemv.h's GEMV: every row
// independent of the call (so batch invariant like CUDA's small-M kernel), one weight read for all rows, and the
// MXFP8 form of the same order for the S3 artifact. The hipBLASLt buckets stay pinned (run_blas) for the rest.
#pragma once

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <string>
#include <vector>

#include "cere_executor.h"

namespace cere::hipk2 {

enum class GemmOut : uint32_t { kBF16 = 0, kF32 = 1 };
constexpr uint32_t kWmmaMinM = 17;  // row classes >= this (with k % 64 == 0) run k2_wmma

struct BucketPin {
    uint32_t m_lo, m_hi;    // the bucket's M range, inclusive
    hipblasLtMatmulAlgo_t algo;
    int solution_index;     // hipblaslt_ext::getIndexFromAlgo
    std::string kernel;     // hipblaslt_ext::getKernelNameFromAlgo
};

class GemmPlan {
   public:
    GemmPlan() = default;
    GemmPlan(const GemmPlan &) = delete;
    GemmPlan &operator=(const GemmPlan &) = delete;
    ~GemmPlan();

    // Pins one algorithm per bucket up to next_pow2(max_m). CERE_ERR_UNSUPPORTED when a bucket has no zero-workspace
    // algorithm supported at both of its ends.
    cere_status init(hipblasLtHandle_t handle, uint32_t n, uint32_t k, GemmOut out, uint32_t max_m);

    // Enqueues one hipblasLtMatmul on `stream`. 1 <= m <= max_m. `cls` (CUDA K2's, m210 design note D15; 0 = m):
    // the rows of the call this one stands for (m <= cls <= max_m): the bucket is cls's while the GEMM runs m rows.
    cere_status run(const void *x, const void *w, void *y, uint32_t m, hipStream_t stream, uint32_t cls = 0) const;
    // The pinned hipBLASLt bucket alone (no k2_wmma / small-M GEMV routing): the ktests' reference and timing.
    cere_status run_blas(const void *x, const void *w, void *y, uint32_t m, hipStream_t stream, uint32_t cls = 0) const;
    uint32_t n() const { return n_; }
    uint32_t k() const { return k_; }
    uint32_t max_m() const { return max_m_; }
    GemmOut out() const { return out_; }

    const std::vector<BucketPin> &pins() const { return pins_; }

   private:
    hipblasStatus_t layouts(uint32_t m, hipblasLtMatrixLayout_t *a, hipblasLtMatrixLayout_t *b,
                            hipblasLtMatrixLayout_t *c) const;
    hipblasLtHandle_t handle_ = nullptr;
    hipblasLtMatmulDesc_t desc_ = nullptr;
    uint32_t n_ = 0, k_ = 0, max_m_ = 0;
    GemmOut out_ = GemmOut::kBF16;
    std::vector<BucketPin> pins_;
};

}  // namespace cere::hipk2
