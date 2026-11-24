// K2 gemvx-order rows kernel (m23 design note P11R-21): a multi-row BF16 GEMV whose every output element is
// summed in the order of cuBLASLt's gemvx kernel (libcublasLt 13.1.1.3, algoId 13), so row r of an m-row launch is
// bit-equal to the cuBLASLt m = 1 call on that row alone. The rows plans (k2_gemm.h GemmPlan::init_rows) run it for
// every m in [2, kMaxM] where the m = 1 algorithm is a gemvx variant of a known order and cuBLASLt refuses that
// algorithm at m: one launch reads the weights once for all m rows instead of m single-row calls.
//
// The order (decoded from the gemvx SASS and checked by exact emulation against cuBLASLt on the real weights):
// for output j, T leaves; leaf t starts at +0 and runs one FP32 FMA chain acc = fma(w[j, i], x[i], acc) over
// i = c * T * S + t * S + s for chunk c = 0, 1, ... and s = 0 .. S - 1 in increasing order (the products of BF16
// operands are exact in FP32); then a halving tree s[t] = s[t] + s[t + h] for h = T / 2 .. 1; then y = s[0] + 0
// (cuBLASLt's alpha = 1, beta = 0 epilogue: a -0 sum becomes +0), rounded to nearest to the output type. Each row's
// chains are independent of every other row, so rows share one weight read without changing a bit.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace cere::kernels::k2v {

constexpr uint32_t kMaxM = 16;

// One gemvx summation order: `lanes` = T leaves per output (the kernel's blockDim.x), `chunk` = S consecutive
// elements a leaf takes per chunk. lanes 0 = no known order.
struct Order {
    uint32_t lanes = 0;
    uint32_t chunk = 0;
};

// The order of cuBLASLt's m = 1 algorithm `algo_id` / `custom_option` at inner dimension k, as identified for
// libcublasLt `cublaslt_version` 130101 only (`lanes` 0 otherwise, and for every pairing not identified: those keep
// cuBLASLt's per-row calls). Identified: customOption 37 and 75 (blockDim.x 32) at k 2,560, 6,144 and 10,240, and 89
// (blockDim.x 8, the lm_head's) and 71 (the MTP hot lm_head's; InvBitId: the same bits as 89 on the lm_head) at
// k 2,560, all with S = 1; 82 (blockDim.x 32, the shared expert's down_proj) at k 640 with S = 20 (each leaf one
// contiguous block of k / T elements).
Order order_of(size_t cublaslt_version, int algo_id, uint32_t custom_option, uint32_t k);

// True when gemv() serves order `o` at inner dimension k (k a positive multiple of T * S and of 8): T 8 or 32 with
// S 1, or T 32 with S 20.
bool supported(Order o, uint32_t k);

// y[m, n] = x[m, k] · w[n, k]^T in order `o` for m in [1, kMaxM], f32: F32 output, else BF16. x, w 16-byte aligned,
// y aligned to its element. cudaErrorInvalidValue for arguments outside the contract; otherwise the launch status.
// No allocation, no synchronization.
cudaError_t gemv(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const __nv_bfloat16 *w, uint32_t n, void *y, bool f32,
                 Order o, cudaStream_t stream);

}  // namespace cere::kernels::k2v
