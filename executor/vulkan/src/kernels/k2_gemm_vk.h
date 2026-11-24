// VulkanAmd P2: K2's large-m GEMM (m > 16, prefill) on Vulkan, the executor/hip/src/k2_wmma.h role, and the K2 row
// routing of executor/hip/src/k2_gemm_hip.cpp GemmPlan::run (row class <= 16: the small-M GEMV; > 16: this GEMM).
//
// Contract (as k2_wmma.h / k2_mgemv.h): y[r, j] = sum_i x[r, i] w[j, i]; x BF16 [m, k]; w BF16 [n, k] (scales == 0) or
// MXFP8 (q E4M3 [n, k], scales UE8M0 [n, k / 32]); y BF16 (RNE, NaN 0x7FFF) or FP32 [m, n]. FP32 accumulation of exact
// products. Every output's bits are a function of its own x row and w row only: shift invariant (a call on rows
// s.. gives the full call's rows s..), row-count invariant, repeats bit-equal; an MXFP8 weight gives the bits of its
// BF16 upcast (k2m's contract).
//
// How (shaders/k2_gemm*.comp): FP16 cooperative matrices (RADV has no BF16 type) with every operand row carried as
// FP16 times its own power of two; a scan (k2_gemm_scan.comp) picks each row's shift and marks the rows for which
// that is not exact. P4-B: an x row whose only inexact values are at most 64 "tail" values (exponent below the row
// max's - 32, e.g. silu / sigmoid left tails) is SPLIT: the matrix unit takes the row without its tail and an exact
// FP32 fma chain over the tail values is added to each output's accumulator in the same workgroup (k2_gemm.comp).
// Any other inexact row (more tail values, Inf / NaN, |max| outside 2^-80 .. 2^100) and any inexact w row is FB:
// outputs touching it are recomputed on the exact FP32 route (k2_gemm_f32.comp, no matrix unit, ~0.3x the speed).
// Nothing is ever silently rounded: the route of an output depends on its two rows only. The info word (bit 31 FB,
// bit 30 SPLIT, bits 16..22 the tail count, bits 0..15 the shift) is a function of the row alone; consumers that
// count bit 31 (vulkan_forward.cpp's exact-fallback counters) count FB rows only.
//
// Not bit-equal across the 16 / 17 boundary: the GEMV (k2_mgemv_vk.h: FP32 FMA chains + a lane butterfly) and this
// GEMM (16-wide matrix-unit chunks) sum in different orders, as HIP's k2_mgemv and k2_wmma do. What callers rely on
// is the row class: `run` routes by cls (0 = m) exactly as HIP's GemmPlan::run, so a call of m rows standing for a
// class of cls rows (D15 resume of a prefill chunk, a verify tail) takes the class's kernel and gives the bits the
// class's full call gives (k2w_test "cls routing").
//
// Recording: the functions record into the context's open command buffer (Context::begin .. submit_wait), with
// barriers between their own dispatches; the caller orders its producers / consumers (Context::barrier).
// Refusals: CERE_ERR_UNSUPPORTED when the device has no VK_KHR_cooperative_matrix or the accumulator-layout probe
// (probe()) has not passed on this context; CERE_ERR_ARGUMENT outside the contract.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vkk2 {

constexpr uint32_t kGemmMinM = 17;  // row classes >= this run the GEMM (HIP's kWmmaMinM)

// Every pipeline gemm / run (class >= 17) / gemm_prepare_weight / gemm_probe can request: the probe's, and for each
// weight format asked (BF16 and / or MXFP8) the row scan (the BF16 scan of the activations is always included), the coopmat GEMM and the FP32 route. No shape parameter: no
// K2 GEMM pipeline depends on m, n or k. Model: both flags per the weights the family uses (BF16 for K3/K4/K5/K7/K13).
// run() routes a class <= 16 to vkk2::gemv: pass the k values (BF16 / MXFP8 weights) run() is called with to include those
// GEMV pipelines (pipeline_specs of k2_mgemv_vk.h); the families that call run() list their GEMV ks themselves and leave them empty.
std::vector<vk::PipelineSpec> pipeline_specs(bool bf16, bool mx, const std::vector<uint32_t> &gemv_bf16_ks = {},
                                             const std::vector<uint32_t> &gemv_mx_ks = {});

// Runs shaders/k2_gemm_probe.comp (one submit) and records whether the FP32 accumulator layout is the one
// k2_gemm.comp's epilogue assumes. Must be called once per context before gemm() / run() with a class > 16.
// CERE_ERR_UNSUPPORTED: no cooperative matrices, or a different layout.
cere_status gemm_probe(vk::Context &ctx);
// True when the context has cooperative matrices and gemm_probe passed on it (the probe result lives in the Context,
// vk::Probe::kK2GemmLayout). Consumers check it before recording anything that may route to the GEMM.
bool gemm_ready(const vk::Context &ctx);

// Bytes of the per-call workspace: the x rows' info words (m) and, when the caller passes no prepared weight info,
// the w rows' (n). 16-byte aligned, reusable after the call's work completes.
size_t gemm_workspace_bytes(uint32_t m, uint32_t n);

// Records the scan of a weight's rows into winfo (n uint32, 16-byte aligned): the weight's info words, a function of
// the weight alone, so a caller may scan once (at load) and pass winfo to every later gemm()/run() on that weight.
cere_status gemm_prepare_weight(vk::Context &ctx, uint64_t w, uint64_t scales, uint32_t n, uint32_t k, uint64_t winfo);

// Records the scan gemm() makes of its x rows into info (m uint32, 16-byte aligned): a weight scan, except that a
// row may be SPLIT when ctx.k2_tail_split() (lever k2_gemm tail_split, vulkan-amd design note D7). For callers
// that count or classify the routes of activations (forward counters, ktests).
cere_status gemm_scan_rows(vk::Context &ctx, uint64_t x, uint32_t m, uint32_t k, uint64_t info);

enum class GemmRoute : uint32_t {
    kAuto = 0,     // the coopmat GEMM, flagged rows' outputs on the FP32 route (the contract path)
    kFp32 = 1,     // every output on the exact FP32 route (timing and the tests' reference; same contract)
};

struct GemmOptions {
    GemmRoute route = GemmRoute::kAuto;
    int profile_stamp0 = -1;  // >= 0: timestamps stamp0 .. stamp0 + 3 around scan / GEMM / FP32 route (timing only)
};

// The GEMM for any m >= 1 (the caller chose it: m > 16, or a smaller call of a class > 16). x, w, y, ws 16-byte
// aligned; k % 64 == 0; x, w (and y) bytes < 2^32 (one <= 4 GiB buffer per tensor). ws: gemm_workspace_bytes(m, n)
// bytes. winfo: 0 = scan w in this call (into ws), else the weight's prepared info (gemm_prepare_weight).
// Records: scan(x) [+ scan(w)] -> barrier -> coopmat GEMM -> barrier -> FP32 route for flagged rows. The x rows' info
// words go to an armed K2 x-info tap's address when one matches (x, m, k) (vk::Context::arm_k2_info), else to ws.
cere_status gemm(vk::Context &ctx, uint64_t x, uint64_t w, uint64_t scales, uint64_t y, uint32_t m, uint32_t n, uint32_t k,
                 bool out_f32, uint64_t ws, uint64_t winfo = 0, const GemmOptions *opts = nullptr);

// K2's row routing (HIP GemmPlan::run): choice = cls ? cls : m (m <= cls when cls != 0); choice <= 16 -> vkk2::gemv
// (m <= 16 by the constraint; k % 16 BF16 / % 32 MXFP8; ws unused), choice >= 17 -> gemm (k % 64 == 0, else
// CERE_ERR_UNSUPPORTED: HIP's hipBLASLt fallback has no Vulkan counterpart). The same arguments as vkk2::gemv plus
// cls and the GEMM's workspace / prepared weight info.
cere_status run(vk::Context &ctx, uint64_t x, uint64_t w, uint64_t scales, uint64_t y, uint32_t m, uint32_t n, uint32_t k,
                bool out_f32, uint32_t cls, uint64_t ws, uint64_t winfo = 0);

}  // namespace cere::vkk2
