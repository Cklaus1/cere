// P3b decode GEMMs (m23 design note P14): the routed experts' GEMM1 + SwiGLU + FP4 quantize and GEMM2 at
// m * top_k <= kMaxRows (decode and verify), in place of FlashInfer's TMA warp-specialized grouped GEMMs, doActivation
// and the stride setup between them (p3b_rows.h routed_rows). Bit-equal to them by construction:
//   * every output element is the sequential chain of FlashInfer's MMA (SM120_16x8x64_TN_VS<e2m1, e2m1, float, ue4m3,
//     16>: mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64, scale-factor byte / thread id 0)
//     over the k64 blocks in increasing k from a zero accumulator, as the CUTLASS SM120 block-scaled mainloop issues
//     it for every tile shape and swap (the P3 tactic sweep: every pinned pair is bit-equal to 128x128x128B no-swap),
//     the weights as the MMA's A, the tokens as its B;
//   * the epilogue is the TMA warp-specialized one's LinearCombination: D = BF16_rn(beta * C + alpha_e * acc) with
//     beta = 0 and C cleared, alpha_e the expert's global scale (quant.fp4.fcN.global_scale[e]);
//   * GEMM1's epilogue then runs doActivationKernel's body on the BF16 GEMM1 output (GLUAdaptor<SiLu>: silu(gate) *
//     linear on Array<float, 8>, gate = rows [I, 2I), times quant_fc2, quantizePackedFPXValue's PackedVec and
//     cvt_warp_fp16_to_fp4<bf16, 16, 8, false, DisableFastMath>), so GEMM2's FP4 A and its scale factors are
//     doActivation's bytes. They are written compact ([rows, I/2] FP4, [rows, I/16] UE4M3), not in the
//     per-expert-padded swizzled layout GEMM2's TMA reads.
// The NVFP4 4over6 quantization (FLASHINFER_NVFP4_4OVER6) is not supported here: routed_rows keeps the grouped GEMMs.
//
// math_reference: flashinfer
// FlashInfer 0.6.17 csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh (doActivationKernel, GLUAdaptor,
// quantizePackedFPXValue, getOffsetActivationSF), csrc/nv_internal/tensorrt_llm/kernels/quantization_utils.cuh
// (cvt_warp_fp16_to_fp4, get_sf_out_offset_128x4), moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl (the SM120
// epilogue: LinearCombination, beta 0 without C), cutlass/include/cute/arch/mma_sm120.hpp (SM120_16x8x64_TN_VS),
// cute/atom/mma_traits_sm120.hpp (the A / B / C / SFA / SFB register layouts).
#pragma once

#include "cere_executor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cere::p3b::decode {

// The largest m * top_k the decode GEMMs serve (small_prologue's: one CTA holds the whole routing).
constexpr int64_t kMaxRows = 256;
// The largest m * top_k routed_rows sends to the decode GEMMs on its own (RowsPath::kAuto): m <= 20 at top_k 10 (the
// plain step, the verify windows and the r4 spec round). Cold-L2 ktest at L23 (ktest/l2_flush before each graph
// replay, us, decode against grouped): m 1 27.9 / 44.3, m 4 85.1 / 96.2, m 5 97.2 / 108.5, m 16 258.8 / 272.8,
// m 20 316.9 / 321.5, m 25 411.6 / 380.6, so m 25 keeps the grouped GEMMs.
constexpr int64_t kRoutedMaxRows = 200;

// One active expert of the step, written by small_prologue (p3b_runner.cu) in increasing expert id: its permuted
// rows [first, first + rows). Slots past the active count hold expert -1. `active` = the active count (every slot).
struct Group {
    int32_t expert;
    int32_t first;
    int32_t rows;
    int32_t active;
};
static_assert(sizeof(Group) == 16, "one int4 per slot");

// The MMA chain: kForward is FlashInfer's (every k64 block in increasing k); kDropLast (ktest only, the fail-first
// perturbation) skips the last block.
enum class Order : int { kForward = 0, kDropLast = 1 };

struct Shape {
    int64_t m = 0;  // tokens: an expert's rows <= m (a token's top_k experts are distinct)
    int64_t hidden = 0;
    int64_t inter = 0;
    int64_t experts = 0;
    int64_t top_k = 0;
};

// The decode GEMMs serve `s`: 1 <= m * top_k <= kMaxRows, m <= 32, hidden and inter multiples of 64.
bool supported(const Shape &s);

// Group slots small_prologue writes and the kernels' grid covers: min(m * top_k, experts).
int64_t slots(const Shape &s);

// GEMM1 + SwiGLU + FP4 quantize. `x` / `x_sf`: GEMM1's A (permuted FP4 rows [m * top_k, hidden / 2], scale factors
// in FlashInfer's per-expert swizzled layout, getOffsetActivationSF) as small_prologue leaves them; `w13` [E, 2I, H/2]
// and `w13_sf` (swizzled) the layer's; `alpha` = quant.fp4.fc1.global_scale [E]; `quant_scale` (nullable) =
// quant.fp8.quant_fc2; `act_scale` = quant.fp4.fc2.act_global_scale, per expert when `per_expert_act_scale`. Out:
// `act` [m * top_k, I/2] FP4 and `act_sf` [m * top_k, I/16]. `ready` (nullable, [slots], zero at launch: small_prologue
// clears it): each CTA adds 1 to its slot's count after its last store, so gemm2 on the same `ready` waits per slot
// (I / 80 CTAs) instead of on this whole grid. `pdl`: launched as a programmatic dependent (waits on the prologue
// before any read) and triggers its dependents right after that wait.
cere_status gemm1_act(const Group *groups, const uint8_t *x, const uint8_t *x_sf, const uint8_t *w13,
                      const uint8_t *w13_sf, const float *alpha, const float *quant_scale, const float *act_scale,
                      bool per_expert_act_scale, bool disable_fast_math, uint8_t *act, uint8_t *act_sf, uint32_t *ready,
                      const Shape &s, Order order, bool pdl, cudaStream_t stream);

// GEMM2: `act` / `act_sf` (gemm1_act's), `down` [E, H, I/2] and `down_sf` (swizzled), `alpha` =
// quant.fp4.fc2.global_scale [E] -> `out` [m * top_k, H] BF16, permuted (GEMM2's D). `pdl`: reads `groups` and
// prefetches its weights before it waits on gemm1_act: on its slot's `ready` count (gemm1_act's, which must be
// launched immediately before as this kernel's programmatic primary) or, `ready` null, on gemm1_act's whole grid.
cere_status gemm2(const Group *groups, const uint8_t *act, const uint8_t *act_sf, const uint8_t *down,
                  const uint8_t *down_sf, const float *alpha, __nv_bfloat16 *out, const uint32_t *ready,
                  const Shape &s, Order order, bool pdl, cudaStream_t stream);

// ktest support: GEMM2's A scale factors as FlashInfer's doActivation leaves them (`sf`: per-expert swizzled,
// getOffsetActivationSF with K = inter) gathered to the compact [rows, inter / 16] layout gemm1_act writes, the
// expert of each permuted row from `expert_first_token_offset` [experts + 1].
cere_status compact_act_sf(const int64_t *expert_first_token_offset, const uint8_t *sf, int64_t rows, int64_t inter,
                           int64_t experts, uint8_t *out, cudaStream_t stream);

}  // namespace cere::p3b::decode
