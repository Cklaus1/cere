// AmdAmd PAR-6 K2-SHIFT: a BF16 GEMM for gfx1151 whose every output element is a fixed function of its own x row
// and w row: y[m, n] = sum_k x[m, k] w[n, k], FP32 accumulation in one k order (WMMA 16x16x16 steps, k ascending),
// rounded once to BF16 (or stored FP32). No split-K, no atomics, no row-count or row-offset dependent choice: a row
// gives the same bits whichever call, row count or starting row it is computed in (hipBLASLt on gfx1151 does not:
// amd-amd design note OPEN K2-SHIFT), and repeated calls are bit-identical.
#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "cere_executor.h"

namespace cere::hipk2 {

enum class WmmaOut : uint32_t { kBF16 = 0, kF32 = 1 };

// x bf16 [m, k] (row stride k), w bf16 [n, k] (row stride k), y [m, n] (row stride n; BF16 or FP32 per `out`).
// Requires k % 64 == 0, x, w and y 16-byte aligned, m, n >= 1. CERE_ERR_ARGUMENT otherwise; CERE_ERR_INTERNAL for a
// failed launch. Enqueued on `stream`, no allocation.
cere_status gemm_wmma(const __hip_bfloat16 *x, const __hip_bfloat16 *w, void *y, uint32_t m, uint32_t n, uint32_t k,
                      WmmaOut out, hipStream_t stream);

}  // namespace cere::hipk2
