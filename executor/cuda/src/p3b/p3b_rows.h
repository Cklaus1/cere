// P3b routed rows (m23 design note P13): FlashInfer's runMoe up to and including GEMM2, without its
// finalizeMoeRoutingKernel. The same launches in the same order on the same arguments as runMoe's non-min-latency
// path (FlashInfer 0.6.17 cutlass_fused_moe_kernels.cuh CutlassMoeFCRunner::runMoe; NVFP4 x NVFP4 -> BF16, SwiGLU,
// no bias / LoRA / AWQ, one EP rank, fused finalize off): the expert-map prologue, expandInputRowsKernelLauncher
// (above 256 rows p3b_runner.cu's expand_tiles, the same bytes), the TMA grouped-GEMM inputs, gemm1 (GEMM1 + SwiGLU
// + FP4 quantize; above 256 rows act_tiles, not doActivation) and gemm2's GEMM, so the BF16 rows are bit-equal to the
// rows runMoe's finalize reads. The finalize itself is K7's combine (k7_moe.h MoeTaps.routed_rows), which forms the
// routed row with finalizeMoeRoutingKernel's arithmetic. At m * top_k <= 256 (decode and verify) the expert-map
// prologue and expandInputRowsKernel run as one launch that writes the same maps and the same FP4 rows and scale
// factors (one CTA count, one thread layout, FlashInfer's quantizePackedFPXValue) in place of four launches, which
// also writes the decode GEMMs' group table; when the decode GEMMs serve the shape (p3b_decode.h, P14), they replace
// the stride setup, gemm1 and gemm2 (bit-equal rows).
//
// math_reference: flashinfer
// FlashInfer 0.6.17 csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh (runMoe, configureWsPtrs,
// getWorkspaceDeviceBufferSizes, setupTmaWarpSpecializedInputs, gemm1, gemm2).
#pragma once

#include "cere_executor.h"
#include "moe_kernels.h"
#include "p3b/p3b_decode.h"
#include "p3b/p3b_moe.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cere::p3b {

using FcRunner = tensorrt_llm::kernels::cutlass_kernels::CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1,
                                                                             __nv_bfloat16, __nv_bfloat16>;
using GemmRunner = tensorrt_llm::kernels::cutlass_kernels::MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16>;

struct RowsProblem {
    int64_t m = 0;  // tokens
    int64_t hidden = 0;
    int64_t inter = 0;
    int64_t experts = 0;
    int64_t top_k = 0;
    // ktest: kGrouped = FlashInfer's grouped GEMMs at every m (the decode GEMMs' reference), kDecode = the decode
    // GEMMs wherever decode::supported; kAuto (the model's) = the decode GEMMs up to decode::kRoutedMaxRows. And the
    // decode GEMMs' MMA chain (kDropLast: the fail-first perturbation).
    RowsPath path = RowsPath::kAuto;
    decode::Order order = decode::Order::kForward;
};

// routed_rows runs `p` on the decode GEMMs (else on FlashInfer's grouped GEMMs, doActivation and the stride setup).
bool rows_decoded(const RowsProblem &p);

// Workspace bytes of routed_rows (256-byte aligned regions; 0 when the GEMM runner cannot size its workspace).
// Monotone in m.
size_t rows_workspace_bytes(const GemmRunner &gemm, const RowsProblem &p);

// The routed experts' per-(token, slot) BF16 rows of `x` [m, hidden] under the routing `topk_ids` (int32) /
// `topk_weights` (FP32) [m, top_k], with the GEMM1 / GEMM2 tactics `gemm1` / `gemm2` (TMA warp-specialized; fused
// finalize off). `fc` supplies the TMA stride computation, `gemm` runs the grouped GEMMs. CERE_ERR_ARGUMENT for a short
// or misaligned workspace, CERE_ERR_INTERNAL when FlashInfer throws or a launch fails. Graph-capturable.
cere_status routed_rows(FcRunner &fc, GemmRunner &gemm,
                        const tensorrt_llm::cutlass_extensions::CutlassGemmConfig &gemm1,
                        const tensorrt_llm::cutlass_extensions::CutlassGemmConfig &gemm2, const RowsProblem &p,
                        const __nv_bfloat16 *x, const int32_t *topk_ids, const float *topk_weights,
                        const void *w13_weight, const void *down_weight,
                        const tensorrt_llm::kernels::cutlass_kernels::QuantParams &quant, void *workspace,
                        size_t workspace_bytes, cudaStream_t stream, Rows *out);

}  // namespace cere::p3b
