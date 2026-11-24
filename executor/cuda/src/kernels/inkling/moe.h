// Inkling 2 Mini S4 (inkling design note 3.4, 5, G1): the routed-MoE kernels of family inkling2_loop: the router,
// the routed activation quantizers (NVFP4 fouroversix, MXFP8) and the W4A4 / W8A8 grouped expert GEMMs with the
// capped SwiGLU x gamma and the top-8 combine. Arithmetic = the CPU references executor/ref/inkling/{router,quant,
// moe}_ref.h bit for bit (G1): each function below names the ref routine it reproduces; the order of every reduction
// is documented at the top of those headers. Every function enqueues on `stream`, never synchronizes, allocates
// nothing and returns the launch status (cudaErrorInvalidValue for a malformed argument, nothing launched).
//
// Determinism (G4 / CERE D9): every output element is chained by one thread in a fixed order that depends only on
// its own token's row, routing and the weights (its block terms are pure functions of their blocks, wherever they
// are computed): no split-K, no atomics on values, no dependence on M, on the other
// tokens (which tokens share an expert), on the row order within an expert, or on the launch shape. A token's output
// is bit-identical alone, in any batch and in any chunk split. Run twice = bit-identical.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cere::inkling {

using bf16 = __nv_bfloat16;

constexpr int kMoeHidden = 2304;   // H
constexpr int kMoeInter = 1152;    // I (w13 rows = 2I, gate/up interleaved)
constexpr int kMoeExperts = 128;   // routed experts per layer
constexpr int kMoeTopK = 8;
constexpr int kMoeMaxTokens = 4096;  // M range of moe_experts / router: 1 .. kMoeMaxTokens (decode 1..64, prefill chunks)

// Fail-first mutants (executor/tests/gpu/test_inkling_kernels_moe.py only; the forward passes kNone):
//   kSoftmaxRouting: weights = softmax of the 8 selected logits x 8 (K7's softmax router) instead of sigmoid/sum;
//   kGlobalActScale: NVFP4 activation amax = kMutantGlobalAmax for every row (one per-layer scale, K7's) instead of
//     the per-token amax (the GEMM outer factor uses the same constant);
//   kGammaAfterDown: the SwiGLU without gamma; the expert output is bf16(mul(y, gamma)) after the down projection;
//   kAlwaysSix: NVFP4 block scale always the /6 candidate (no fouroversix);
//   kBlockOrder: the tensor-core NVFP4 GEMM (calls of more than 64 tokens) adds each K chunk's block terms in swapped
//   pairs (1, 0, 3, 2, ...) instead of in order (D-I-CPF's order change; the other GEMM paths ignore it);
//   kRouterOrder: the multi-row router (calls of more than 64 tokens) chains each lane's products over K descending
//   (D-I-CPF's order change; the one-row router ignores it).
enum class MoeMutant : int {
    kNone = 0,
    kSoftmaxRouting = 1,
    kGlobalActScale = 2,
    kGammaAfterDown = 3,
    kAlwaysSix = 4,
    kBlockOrder = 5,
    kRouterOrder = 6
};
constexpr float kMutantGlobalAmax = 64.0f;

// -- router (ref::route_row) -------------------------------------------------------------------------------------
// x: BF16 [m, kMoeHidden] at row stride x_stride (elements), the MLP input (after mlp_norm); w: BF16 [kMoeExperts,
// kMoeHidden] (mlp.gate.weight, contiguous); bias: FP32 [kMoeExperts] (mlp.gate.bias = e_score_correction_bias).
// Out: ids int32 [m, kMoeTopK] ascending expert ids (slot order), weights FP32 [m, kMoeTopK] (route_scale 8 applied);
// logits FP32 [m, kMoeExperts] (nullable; the router logits, for the ktests). One 8-CTA cluster per token (16
// experts per CTA, each logit one warp's ref::reduce_dot<256>); calls of more than 64 tokens: one cluster per 8 tokens
// (the same arithmetic per token, D-I-CPF); m in 1..65535.
cudaError_t router(const bf16 *x, int64_t x_stride, int m, const bf16 *w, const float *bias, int32_t *ids,
                   float *weights, float *logits, cudaStream_t stream, MoeMutant mutant = MoeMutant::kNone);

// -- activation quantizers (ref::nvfp4_quantize_row / ref::mxfp8_quantize_row) ------------------------------------
// x: BF16 [rows, k] at row stride x_stride; k a multiple of 64 (NVFP4) / 32 (MXFP8), k <= 4096.
// NVFP4 out: codes uint8 [rows, k/2] (element 2j low nibble of byte j), scales E4M3 [rows, k/16], amax FP32 [rows].
// MXFP8 out: values E4M3 [rows, k], exponents UE8M0 [rows, k/32].
cudaError_t quantize_nvfp4(const bf16 *x, int64_t x_stride, int rows, int k, uint8_t *codes, uint8_t *scales,
                           float *amax, cudaStream_t stream, MoeMutant mutant = MoeMutant::kNone);
cudaError_t quantize_mxfp8(const bf16 *x, int64_t x_stride, int rows, int k, uint8_t *values, uint8_t *exponents,
                           cudaStream_t stream);

// -- routed experts (ref::moe_experts_token) ----------------------------------------------------------------------
// One layer's expert weights, device pointers in the checkpoint's layout (never repacked), E = num_experts:
//   NVFP4 (layers 0-19): w13 uint8 [E, 2I, H/2], w13_scale E4M3 [E, 2I, H/16], w13_amax FP32 [E], w13_den FP32 [1]
//     (w13_weight_outer_scale_denominator); w2 uint8 [E, H, I/2], w2_scale E4M3 [E, H, I/16], w2_amax [E], w2_den [1].
//   MXFP8 (layers 20-23): w13 E4M3 [E, 2I, H], w13_scale UE8M0 [E, 2I, H/32]; w2 E4M3 [E, H, I], w2_scale [E, H, I/32];
//     amax/den null.
//   w13 row 2c = gate column c, row 2c + 1 = up column c (stored interleaved).
struct MoeWeights {
    bool nvfp4 = true;
    int32_t num_experts = kMoeExperts;
    const uint8_t *w13 = nullptr;
    const uint8_t *w13_scale = nullptr;
    const uint8_t *w2 = nullptr;
    const uint8_t *w2_scale = nullptr;
    const float *w13_amax = nullptr;
    const float *w2_amax = nullptr;
    const float *w13_den = nullptr;
    const float *w2_den = nullptr;
};

// Optional taps (ktests), each nullable, rows indexed by routed row id t * kMoeTopK + k:
//   gate_up BF16 [m*8, 2I] (bf16 of each GEMM1 element, interleaved), h BF16 [m*8, I] (capped SwiGLU x gamma),
//   expert_out BF16 [m*8, H] (each expert's BF16 output before the combine).
struct MoeTaps {
    bf16 *gate_up = nullptr;
    bf16 *h = nullptr;
    bf16 *expert_out = nullptr;
};

// Workspace of moe_experts for m tokens (256-byte aligned base; contents undefined between calls).
size_t moe_workspace_bytes(int m, bool nvfp4);

// out[t] = bf16( sum over slots k = 0..7 in order of bf16(expert ids[t,k](x[t], gamma = weights[t,k])) ), FP32 sum.
// x: BF16 [m, H] at x_stride (the same rows the router read); ids/weights: the router's outputs (ids ascending, each
// in [0, num_experts), distinct per token); out: BF16 [m, H] at out_stride (must not alias x). m in 1..kMoeMaxTokens.
// Launches: quantize x (per token), dispatch (one CTA: stable expert-sorted routed rows), GEMM1 + outer factor ->
// BF16 gate/up, SwiGLU x gamma + requantize (per routed row), GEMM2 + outer factor -> BF16 per routed row, combine.
cudaError_t moe_experts(const bf16 *x, int64_t x_stride, int m, const int32_t *ids, const float *weights,
                        const MoeWeights &w, void *workspace, size_t workspace_bytes, bf16 *out, int64_t out_stride,
                        cudaStream_t stream, const MoeTaps *taps = nullptr, MoeMutant mutant = MoeMutant::kNone);

}  // namespace cere::inkling
