// K2 small-M path (m23 design note P11): the in-tree BF16 GEMM every GemmPlan runs for 1 <= m <= kMaxM (decode,
// batched rows, verify). y[m, n] = x[m, k] · w[n, k]^T over up to kMaxSegments weights that share x (one launch).
//
// Numerics: BF16 tensor-core MMA (mma.sync m16n8k16, FP32 accumulation), no atomics. Every launch computes a full
// 16-row tile whatever m is (rows past m are zeros and are not stored), so row r's instructions do not depend on m or
// on the other rows. Each output element y[r, j] is reduced in one fixed order that depends on k alone: k is cut into
// 32-element steps and into S = split_warps(k) slices; slice p takes steps p, p + S, p + 2S, ... in increasing order
// on one warp, chaining two MMAs per step into its FP32 accumulator (lane t of a step holds k = 32 s + 8 t .. + 7;
// MMA h takes k = 32 s + 8 t + 4 h + 0..3 of every t, the same map for x and w). S <= 16 (k <= 6,144): the slices
// are summed in slice order. Larger k: S = 16 G, G = min(8, ceil(k / 2,560)) groups (a slice holds at most 5 steps up
// to k = 20,480) [REVISED: P11R-14]; each group of 16 consecutive slices is summed in slice order, then the G group
// sums in group order [REVISED: P11R-13]. So y[r, j] is a function of (x[r, :], w[j, :], k) only: the same bits for
// every m in [1, kMaxM], every row position, every grouping of columns and every set of segments launched together
// (batch invariance and fusion equality), and repeated calls are bit-identical. BF16 output is one round-to-nearest
// of the FP32 sum.
//
// Layout (P11R-12; no output bit depends on it): one CTA of S warps when S <= 16, else a thread block cluster of G
// CTAs (one per group, 16 warps each; rank 0 sums the group sums through distributed shared memory); 16 or 8 columns
// per column tile from the column count, G and the SM count, so few columns (the router, hc_down, the shared expert)
// still fill every SM.
//
// Launch (P11R-21; no output bit depends on it): every launch is a programmatic dependent launch. A CTA first lets
// the stream's next kernel launch (griddepcontrol launch_dependents: only a kernel launched as a programmatic
// dependent, i.e. this one, starts early, and it waits as below), then waits for the previous kernel
// (griddepcontrol.wait) before it reads x or w or writes y. So a launch's CTAs are resident and set up while the
// previous kernel drains, and nothing a kernel before it writes is read early.
//
// Fused hc epilogues (P11R-21): gemm_hc_act and gemm_hc_gate run K3's hc_mix element-wise steps (k3_hc_math.cuh) on
// the FP32 sums in the GEMM's epilogue. Each sum is the one gemm(..., f32 = true) computes (same slices, same order),
// and the epilogue is K3's own inline arithmetic, so the outputs are bit-identical to gemm then K3's kernel.
//
// Early weights (K3Chain, k3chain design note; no output bit depends on it): gemm_hc_act, gemm_hc_gate and gemm
// (the MoE router, BeatSglang S4) with `early` copy each warp's first round of weights (kUnroll steps of its slice:
// the whole slice at k <= 2,560) into shared memory with cp.async after launch_dependents and before
// griddepcontrol.wait (the launch's dynamic shared memory, kUnroll * 512 bytes per warp and 8 columns), then run the
// same slices, MMAs and epilogue on those bytes. No registers hold them across the wait, so an early CTA stays as
// light as a plain one and can be resident while the kernel before it runs. x, normed and y are still read and
// written only after the wait. So no kernel right before an early launch on its stream may write w (the model's
// weights are written at load); the kernels' names and template arguments are unchanged (the early ones are overloads
// on their first parameter's type).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace cere::kernels::k2s {

constexpr uint32_t kMaxM = 16;
constexpr uint32_t kMaxSegments = 4;

struct Segment {
    const __nv_bfloat16 *w;  // [n, k], 16-byte aligned
    void *y;                 // [m, n] row-major, BF16 or F32 (the launch's)
    uint32_t n;
};

// True when the kernel serves inner dimension k (a positive multiple of 32: whole warp steps, and every row of x and
// w starts 16-byte aligned given an aligned base).
bool supported(uint32_t k);

// The slices per column (the split of k, one warp each), a function of k only.
uint32_t split_warps(uint32_t k);

// The launch gemm() makes for inner dimension k over segments of widths ns[0..count) on the current device: grid,
// block and cluster dims (D12 attributes the profile's kernel records to call shapes by them); zero dims outside the
// contract.
struct Geometry {
    dim3 grid;
    dim3 block;
    dim3 cluster;
};
Geometry geometry(uint32_t k, const uint32_t *ns, uint32_t count);

// y_s = x · w_s^T for every segment s < count (1 <= count <= kMaxSegments), m in [1, kMaxM], f32: F32 output, else
// BF16. x and every w 16-byte aligned. cudaErrorInvalidValue for arguments outside the contract; otherwise the launch
// status. No allocation, no synchronization. `early` (BeatSglang S4, beat-sglang design note): the early-weights
// launch above (the same bits), so every w must be a weight no kernel before the launch on its stream writes.
cudaError_t gemm(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const Segment *segments, uint32_t count, bool f32,
                 cudaStream_t stream, bool early = false);

// MXFP8 weights (track Fp4S3, kernels/k2_mxfp8.h): segment s's weight is q E4M3 [n, k] (16-byte aligned) with UE8M0
// scales [n, k / 32] (any alignment; k = 640's 20-byte rows included), every scale byte in [k2m::kMinScale,
// k2m::kMaxScale]. gemm_mxfp8 is gemm on the weights' exact BF16 upcasts (k2m::dequantize), bit for bit: the same
// kernel body, layout, slices and reduction order; a lane loads 8 E4M3 bytes per row and 32-k step (instead of 16
// BF16 bytes) and that row's scale byte, and turns them into the identical BF16 MMA operand (k2_mxfp8.cuh) before the
// same MMAs. The launch is geometry()'s for the same (k, ns, count). Contract and errors as gemm.
struct Mxfp8Segment {
    const uint8_t *q;       // E4M3 [n, k], 16-byte aligned
    const uint8_t *scales;  // UE8M0 [n, k / 32]
    void *y;                // [m, n] row-major, BF16 or F32 (the launch's)
    uint32_t n;
};
cudaError_t gemm_mxfp8(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const Mxfp8Segment *segments, uint32_t count,
                       bool f32, cudaStream_t stream);

// hc_mix's down projection with K3's activation fused (P11R-21): t[r, j] = hc::act(y[r, j]) for y = x · w^T [m, n] as
// gemm(x, m, k, {w, ., n}, 1, f32 = true) computes it; t BF16 [m, n]. Contract and errors as gemm.
cudaError_t gemm_hc_act(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const __nv_bfloat16 *w, uint32_t n,
                        __nv_bfloat16 *t, cudaStream_t stream, bool early = false);

// True when gemm_hc_gate serves inner dimension k: supported(k) and the hc::kHc streams' slices fit one CTA
// (hc::kHc * split_warps(k) <= 16, k < 640; the mix's lowrank is 320).
bool hc_gate_supported(uint32_t k);

// gemm_hc_gate's launch over `hidden` columns per stream (D12 attribution, as geometry); zero dims outside the contract.
Geometry hc_gate_geometry(uint32_t k, uint32_t hidden);

// hc_mix's up projection with K3's gate fused (P11R-21): u = t · w^T [m, hc::kHc * hidden] as gemm(t, m, k,
// {w, ., hc::kHc * hidden}, 1, f32 = true) computes it, then mixed[r, j] = hc::gate(u[r, g * hidden + j],
// normed[r, g, j] for g < hc::kHc); w [hc::kHc * hidden, k], normed BF16 [m, hc::kHc, hidden], mixed BF16 [m, hidden].
// cudaErrorInvalidValue outside the contract (t, w 16-byte aligned, m in [1, kMaxM], hc_gate_supported(k)).
cudaError_t gemm_hc_gate(const __nv_bfloat16 *t, uint32_t m, uint32_t k, const __nv_bfloat16 *w, uint32_t hidden,
                         const __nv_bfloat16 *normed, __nv_bfloat16 *mixed, cudaStream_t stream, bool early = false);

// How many times the early launches (gemm, gemm_hc_act, gemm_hc_gate with `early`) raised a kernel's dynamic shared
// memory limit in this process: once per kernel variant, context generation and larger size (audit KGM-4; ktest
// only).
uint64_t early_opt_ins();

}  // namespace cere::kernels::k2s
