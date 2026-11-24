// K2 fast small-M MXFP8 path (O73, m23 design note P11R-24): k2s::gemm_mxfp8's GEMM y[m, n] = x[m, k] · w[n, k]^T
// for 1 <= m <= kMaxM over up to kMaxSegments MXFP8 weights that share x (one launch): E4M3 [n, k] plus one UE8M0
// scale byte per 32 consecutive k of a row, w[j, i] = q[j, i] * 2^(s[j, i / 32] - 127) (fp4local design note F5).
//
// Numerics: bit-identical to k2s::gemm_mxfp8 (k2_small_m.h's gemm on the weights' exact BF16 upcast
// k2m::dequantize stores). Every output element y[r, j] is reduced in k2_small_m.h's order, a function of k alone:
// k is cut into 32-element steps and S = k2s::split_warps(k) slices; slice p takes steps p, p + S, ... in increasing
// order, chaining the same two BF16 mma.sync m16n8k16 per step with the same operand map into an FP32 accumulator
// that starts at zero (lane t of a step holds k = 32 s + 8 t .. + 7, MMA h takes k = 32 s + 8 t + 4 h + 0..3, x and
// w alike; row r of x and column j of w sit at an MMA row and column whose position changes no bit, as in
// k2_small_m.h); the kernel serves S <= 16 (k <= 6,144, every MXFP8 K2 weight), where the slice results are summed
// in slice order. The FP32 sum is stored as k2_small_m.h stores it (one BF16 round to nearest, or F32). The weight
// operand is formed in registers right before its MMA by S3's conversion (k2_mxfp8.cuh bf16x8, the exact upcast for
// every scale byte in [k2m::kMinScale, k2m::kMaxScale]), so the MMA inputs are the upcast's bits. No atomics, no
// workspace.
//
// Layout (no output bit depends on it): CTA b holds column tile b (8 Q columns of one segment, Q n-tiles of 8; the
// segments' tiles in segment order); its S / G warps each run G consecutive slices, one FP32 accumulator per slice.
// The tile's scale bytes are copied to shared memory first; its weights stream through a ring of shared-memory
// stages (cp.async, no register staging), one chunk of S consecutive steps (one step of every slice) per stage
// together with that chunk's x rows. After the last chunk the slice results go to shared memory and after a CTA
// barrier the CTA sums them in slice order and stores the tile. Every launch is a programmatic dependent launch as in
// k2_small_m.h, except that a CTA requests its scales and first stages' weights before it waits for the previous
// kernel (it reads x and writes y only after the wait): q and the scales must not be written by the kernels the
// launch depends on (model weights).
#pragma once

#include "kernels/k2_small_m.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace cere::kernels::k2x {

constexpr uint32_t kMaxM = 16;
constexpr uint32_t kMaxSegments = 4;

// A launch's layout (kernel tests sweep it; no output bit depends on it). Zero fields take the default layout's.
struct Tune {
    uint32_t q;       // n-tiles per column tile: 1, 2 or 4 (default: 4 past m = 8 when the 32-column tiles give every
                      // SM a CTA, else 2 when the 16-column tiles give every other SM one, else 1)
    uint32_t lock;    // G, slices per warp: 1, 2 or 4 dividing S (default: 2 with 8-column tiles when 2 divides S,
                      // else 1)
    uint32_t stages;  // ring stages, 1..min(ceil(k / 32 / S), 12) (default: 2 when the grid holds 2 CTAs per SM, else
                      // the deepest ring that keeps two CTAs per SM)
};

// True when the kernel serves inner dimension k: k2s::supported(k), k2s::split_warps(k) <= 16 (one slice group) and
// k a multiple of 128 (whole 4-byte scale words per row).
bool supported(uint32_t k);

// y_s = x · upcast(q_s, scales_s)^T for every segment s < count (1 <= count <= kMaxSegments), m in [1, kMaxM], f32:
// F32 output, else BF16. x and every q 16-byte aligned, every scales 4-byte aligned. cudaErrorInvalidValue outside
// the contract or for an invalid tune; otherwise the launch status. No allocation, no synchronization.
cudaError_t gemm(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const k2s::Mxfp8Segment *segments, uint32_t count,
                 bool f32, cudaStream_t stream, const Tune &tune = Tune{});

// The resident layout's tune (kernel tests sweep it; no output bit depends on it): all zero takes the default
// (k = 2,560: {2, 1, 8}; k = 6,144: {1, 4, 4} to m = 8, else {2, 3, 3}); otherwise one of the instantiated layouts
// {q, kc, stages}: k = 2,560 {2, 1, 8}, {1, 5, 4}; k = 6,144 {1, 4, 4}, {2, 3, 3}, {1, 3, 6}.
struct ResidentTune {
    uint32_t q;       // n-tiles per column tile (8 q columns): 1 or 2
    uint32_t kc;      // 16-step groups per ring item, dividing k / 512
    uint32_t stages;  // ring stages
};

// True when the resident layout serves inner dimension k: supported(k), k2s::split_warps(k) == 16 and k / 512 = 5 or
// 12 (k = 2,560 or 6,144).
bool resident_supported(uint32_t k);

// gemm's contract and bits (the reduction order above) in the resident layout (k2-decode-spec L1; no output bit
// depends on it): one CTA per SM (the grid is min(column tiles, SMs)) and 16 warps, warp p slice p, which loads its
// slice's x fragments into registers once, after the dependency wait. CTA b loops over column tiles b, b + grid, ..
// (8 q columns of one segment); each tile streams through a ring of `stages` shared-memory stages in items of kc
// consecutive 16-step groups (the tile's row pieces and their scale bytes, cp.async), the first stages - 1 items
// requested before the dependency wait (q and the scales must not be written by the kernels the launch depends on).
// After a tile's last item the slice results go to shared memory and the CTA sums them in slice order and stores the
// tile. cudaErrorInvalidValue outside gemm's contract, when !resident_supported(k) or for a tune that names no
// instantiated layout; otherwise the launch status. No allocation, no synchronization.
cudaError_t resident(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const k2s::Mxfp8Segment *segments,
                     uint32_t count, bool f32, cudaStream_t stream, const ResidentTune &tune = ResidentTune{});

// k2s::gemm_mxfp8's contract and bits: the resident layout (default tune) where it serves the call and was measured
// faster: k = 2,560 with at least 8,192 weight rows over the segments (the GDN in-projections, attention q|k|v) and
// k = 6,144 with at least 2,560 (attention o_proj, GDN out_proj), k2-decode-spec F4; else this kernel (default
// layout) at k <= 2,560 with at least 2,048 rows at m <= 8 (the shared expert's down; P11R-24); else
// k2s::gemm_mxfp8 (shared gate/up, the indexer).
cudaError_t gemm_mxfp8(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const k2s::Mxfp8Segment *segments,
                       uint32_t count, bool f32, cudaStream_t stream);

}  // namespace cere::kernels::k2x
