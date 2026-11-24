// AmdAmd PAR-6 step 14 (S3 artifact): K2's small-M GEMM and the MXFP8 weight upcast on gfx1151.
//
// gemv: y[r, j] = sum_i x[r, i] w[j, i] for r < m <= kGemvMaxM, w BF16 [n, k] or MXFP8 (q E4M3 [n, k] with UE8M0
// scales [n, k / 32]: w[j, i] = q[j, i] * 2^(s[j, i / 32] - 127), kernels/k2_mxfp8.h). One wave per column; lane l
// owns k elements 512 c + 16 l .. + 15 of every 512-element step c and accumulates them pair by pair in k order with
// gfx11's v_dot2_f32_bf16 (acc = w0 x0 + w1 x1 + acc in one instruction: deterministic, within ~1 FP32 ulp of the
// FMA chain, not correctly rounded; 4x fewer VALU ops per row than convert + FMA, which made m > 4 compute-bound),
// then the wave's fixed xor butterfly sums the 32 lanes. The element order depends on (n, k) only: every row's
// bits are the same whatever m, the call's first row or the cls it stands for (K2-SHIFT / D15 hold trivially), and an
// MXFP8 weight gives the bits of its exact BF16 upcast (the decoded value is the BF16 value; k2m's bit-identity
// contract). Reads every weight byte once per call: 1 B per element for MXFP8, the decode-time bandwidth win.
//
// dequantize: the exact BF16 upcast of an MXFP8 weight (k2m::dequantize's contract: k a positive multiple of 32, every
// scale byte in [k2m::kMinScale, kMaxScale], no E4M3 NaN byte; q and w 16-byte aligned).
#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "cere_executor.h"

namespace cere::hipk2 {

constexpr uint32_t kGemvMaxM = 16;

// scales == nullptr: w is BF16 [n, k]; else w is E4M3 [n, k] and scales UE8M0 [n, k / 32]. k % 16 == 0 (BF16) or
// k % 32 == 0 (MXFP8); x, w, y 16-byte aligned; 1 <= m <= kGemvMaxM. out_f32: y FP32, else BF16 (RN).
// CERE_ERR_ARGUMENT outside the contract, CERE_ERR_INTERNAL for a failed launch. No allocation.
cere_status gemv(const __hip_bfloat16 *x, const void *w, const uint8_t *scales, void *y, uint32_t m, uint32_t n,
                 uint32_t k, bool out_f32, hipStream_t stream);

hipError_t mxfp8_dequantize(const uint8_t *q, const uint8_t *scales, uint32_t n, uint32_t k, __hip_bfloat16 *w,
                            hipStream_t stream);

}  // namespace cere::hipk2
