// K2 helpers for kernel-test entry points (libcere_ktest.so only; the executor creates its handle and plans at
// INSTANTIATE instead).
#pragma once

#include "kernels/k2_gemm.h"

#include <cstdint>

namespace cere::ktest {

// The process-wide cuBLASLt handle of libcere_ktest.so, created on first use (nullptr if cublasLtCreate failed).
cublasLtHandle_t gemm_handle();

// Creates `plan` (max_m) as the executor creates its plan of shape (n, k, out): with kernels::gemm_pins_for's pins and,
// when they need one, a workspace of stream-ordered memory on `stream` returned in *workspace, which the caller
// releases with cudaFreeAsync on `stream` after the plan's last run (nullptr when none is needed or on failure).
cere_status plan_init(kernels::GemmPlan *plan, cublasLtHandle_t handle, uint32_t n, uint32_t k, uint32_t max_m,
                      kernels::GemmOut out, void **workspace, cudaStream_t stream);

// y[m, n] = x[m, k] · w[n, k]^T through a GemmPlan(n, k, max_m = m, out) pinned for this call: for a given (n, k, out,
// m) it runs the same algorithm as every executor plan of that shape (buckets and pins depend on m only).
cere_status gemm(const void *x, const void *w, int64_t m, int64_t n, int64_t k, void *y, kernels::GemmOut out,
                 cudaStream_t stream);

}  // namespace cere::ktest
