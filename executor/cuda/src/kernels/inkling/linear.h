// Inkling 2 Mini (inkling design note 4.1, G4, slice S8): the BF16 projections of the text forward (q, o-gate, rel,
// sink, k, v, the dynamic-conv taps, o, unembed), y[m, n] = x[m, k] · w[n, k]^T, BF16 operands, FP32 accumulation on
// the tensor cores (mma.sync m16n8k16), BF16 output (one RNE rounding). CONTRACT.md: the BF16 GEMMs' order is free
// (not bit-equal across backends); this file fixes it once for every M, so a row's bits never depend on the batch,
// the chunking, the session count or prefill vs decode (G4, CERE D9).
//
// The order of one output element y[t, f] (every kernel below, any m):
// - k is cut into S = k / 256 slices of 8 windows of 32; window [k0, k0 + 32) is two mma steps over the weight row f
//   (MMA operand A, row-major) and the input row t (operand B, column): step 0 takes k0 + 8c + {0, 1} into slots
//   (2c, 2c+1) and k0 + 8c + {2, 3} into (2c+8, 2c+9) for c = 0..3; step 1 the same with + 4;
// - slice s: p_s = the chain of its 16 mma steps from 0, in window order;
// - y = bf16_rn(((p_0 + p_1) + p_2) + ... + p_{S-1}) (FP32 adds in slice order).
// The two kernels run that order: narrow (m <= kLinearNarrowRows; one block per 16 rows of w, slice s on warp s,
// the partials added by one thread per output in slice order) and wide (blocks of 128 rows of w x 128 inputs, each
// warp chains the slices of its tile in order). An element's arithmetic is the same in both, wherever its row sits
// in an mma tile; `path` forces one for the ktest that checks it (test_inkling_kernels_linear.py); kWideMutant is the
// wide kernel with each window's two mma steps in the other order (fail-first, D-I-CPF), never used by the forward.
//
// Segments: one call runs up to kLinearMaxSegs projections that share k and m (e.g. q, gate, rel, sink of the q input
// and k, v, dyn of theirs), each with its own x, w, y. Contract: k a multiple of 256 in [256, 3072]; m in [1, 2^20];
// per segment n in [1, 2^31 - 1], x, w and y 16-byte aligned, ldx >= k and ldy >= n multiples of 8 (elements), w
// contiguous [n, k], y aliasing no x or w. No workspace; enqueues on `stream`, never synchronizes; returns the launch
// status (cudaErrorInvalidValue for a malformed argument, nothing launched).
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cere::inkling {

constexpr int kLinearMaxSegs = 8;
constexpr int64_t kLinearNarrowRows = 64;
constexpr int kLinearSliceK = 256;
constexpr int kLinearMaxK = 3072;

struct LinearSeg {
    const __nv_bfloat16 *x = nullptr;
    int64_t ldx = 0;
    const __nv_bfloat16 *w = nullptr;
    __nv_bfloat16 *y = nullptr;
    int64_t ldy = 0;
    int64_t n = 0;
};

struct LinearArgs {
    int64_t m = 0;
    int64_t k = 0;
    int count = 0;
    LinearSeg seg[kLinearMaxSegs];
};

enum class LinearPath : int { kAuto = 0, kNarrow = 1, kWide = 2, kWideMutant = 3 };

// kNarrow needs m <= kLinearNarrowRows.
cudaError_t linear(const LinearArgs &a, cudaStream_t stream, LinearPath path = LinearPath::kAuto);

}  // namespace cere::inkling
