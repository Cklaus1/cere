// VulkanAmd P2: the whole K5 GDN mixer layer on Vulkan (Qwen3_5GatedDeltaNet.forward). Same contract as
// executor/cuda/src/kernels/k5_gdn_layer.h (the HIP build hipifies its .cu), composed from the Vulkan kernels:
//
//   mixed = hidden in_proj_qkv^T, z = hidden in_proj_z^T, b = hidden in_proj_b^T, a = hidden in_proj_a^T
//   PREFILL (1..4096 tokens): conv1d + SiLU over the q | k | v channel ranges into three planes (vkk5::conv1d_silu);
//     g, beta = vkk5::gating(a, b); o = vkk5::delta_rule_chunk (k5_gdn_chunk_vk.h)
//   DECODE (exactly 1 token): conv update over all 10,240 channels -> q | k | v; o = vkk5::delta_rule_decode
//   y = gated RMSNorm per value head (vkk1::rmsnorm_gated_sigmoid, eps kNormEps); out = y out_proj^T
//
// Projections (in_proj_qkv / z / b / a, out_proj): K2 (vkk2::run, BF16 out) routed by the row class as HIP's
// GemmPlan::run (k5_gdn_layer.h: "`cls` as kernels::GemmPlan::run (0 = tokens)"): class <= 16 the small-M GEMV (rows
// independent of the call), class >= 17 the large-m K2 GEMM (k2_gemm_vk.h: shift and row-count invariant, which D15
// resume needs: a resumed part passes the whole call's class). Not bit-equal across the 16 / 17 class boundary
// (k2_gemm_vk.h, as on HIP): a PREFILL + DECODE handoff that must reproduce the raw in_proj_qkv rows passes one class to
// both (the HIP k5_layer_test passes cls 128 to every call). Classes >= 17 need vkk2::gemm_probe on the context,
// cooperative matrices and hidden_size % 64 == 0 (CERE_ERR_UNSUPPORTED otherwise: no chunked-GEMV fallback).
//
// Differences from the CUDA header: buffer device addresses; no GemmPlans (their row class is the `cls` argument; the
// prepared K2 weight info is LayerWeights::info); hidden_size % 16 == 0;
// chain_core (verify chains) is not provided here: its pieces (conv1d_silu_rows, delta_rule_decode_rows) are in
// k5_gdn_vk.h and the composition is not part of the HIP layer test. CERE_OP_VERIFY is CERE_ERR_UNSUPPORTED.
// Every function records into the context's open command buffer with barriers between its dependent dispatches and
// none after the last: the caller records a barrier before reading out / y / state / end-state outputs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cere_executor.h"
#include "launch/vk_launch.h"

namespace cere::vkk5 {

// Model constants of the A4 config (text_config linear_* entries; executor/cuda reads them from the generated
// cere_model_constants.h, which the Vulkan build does not generate yet).
constexpr int kKeyDim = 16 * 128;     // 2,048
constexpr int kValueDim = 48 * 128;   // 6,144
constexpr int kConvDim = 2 * kKeyDim + kValueDim;  // 10,240: q | k | v
constexpr float kNormEps = 1e-6f;     // rms_norm_eps (the gated norm)

// Buffer device addresses of one layer's tensors (checkpoint layouts, as the CUDA header): in_proj_qkv [10240, hidden],
// in_proj_z [6144, hidden], in_proj_b / in_proj_a [48, hidden], conv1d [10240, 4], A_log f32 [48], dt_bias [48],
// norm [128], out_proj [hidden, 6144]; BF16 unless stated; 16-byte aligned.
// MXFP8 projections (the S3 artifact's: executor/cuda/src/model/mxfp8_plan.h kGdn): a projection whose `*_scales` is
// nonzero is MXFP8, its weight field then holds the E4M3 bytes [n, k] and `*_scales` the UE8M0 scales [n, k / 32]
// (k2_mgemv_vk.h / k2_gemm_vk.h; 0 = BF16, the default). Each projection's format is independent; an MXFP8 projection
// gives the bits of its BF16 upcast (vkk2's contract), so the layer's output does too. MXFP8 needs k % 32 (the
// in-projections: hidden_size % 32; out_proj: 6144).
struct LayerWeights {
    uint64_t in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, conv1d, A_log, dt_bias, norm, out_proj;
    // Optional K2 row info of the five projection weights (weight_info_bytes, written once at load by prepare_weights);
    // 0: each K2 GEMM scans its weight in the call.
    uint64_t info = 0;
    // UE8M0 scales of an MXFP8 projection (0 = that projection is BF16).
    uint64_t in_proj_qkv_scales = 0, in_proj_z_scales = 0, in_proj_b_scales = 0, in_proj_a_scales = 0, out_proj_scales = 0;
    bool any_mxfp8() const {
        return in_proj_qkv_scales || in_proj_z_scales || in_proj_b_scales || in_proj_a_scales || out_proj_scales;
    }
};

// conv bf16 [10240, 3] (oldest first), ssm bf16 [48, 128 (v), 128 (k)]; zeros = a fresh sequence; updated in place.
struct LayerState {
    uint64_t conv, ssm;
};

// One PREFILL call's end-state work (the CUDA header's EndState): resume from ssm_in32 (FP32 [48, 128, 128]); capture
// at `row` (a multiple of 64, 64 <= row < tokens) into ssm_out32 (FP32) and conv_out (bf16 [10240, 3]); *_hi split.
struct EndState {
    uint64_t ssm_in32 = 0;
    uint64_t ssm_out32 = 0;
    uint64_t conv_out = 0;
    int64_t row = -1;
    uint64_t ssm_in32_hi = 0;
    uint64_t ssm_out32_hi = 0;
};

// Every pipeline layer_core / layer_forward / prepare_weights can request: the K5 pieces (conv, conv_state, gating,
// decode), the chunk kernels (+ l2norm), K1's gated rmsnorm of width kHeadDim, K2's BF16 GEMV for k = each hidden size and
// kValueDim (m 1..16), and, when `gemm`, K2's GEMM + probe (row classes >= 17). `mx`: also the MXFP8 GEMV (same ks) and
// the MXFP8 GEMM / weight scan, for MXFP8 projections (the S3 artifact: mx true; a BF16 artifact: false). Model:
// hidden_sizes {2560}, gemm true, mx per the artifact.
std::vector<vk::PipelineSpec> pipeline_specs(const std::vector<uint32_t> &hidden_sizes, bool gemm, bool mx = false);

// Workspace of layer_core / layer_forward for max_tokens tokens (the CUDA header's carving, 256-byte aligned regions);
// 0 outside 1..4096. layer_workspace_bytes includes four K2 GEMM workspaces (vkk2::gemm_workspace_bytes(T, n) for the
// in-projections, recorded without barriers between them: n = 65,536 (qkv's, reused by out_proj for any hidden_size
// <= 65,536), 6,144, 48, 48).
size_t core_workspace_bytes(int64_t max_tokens);
size_t layer_workspace_bytes(int64_t max_tokens);

// Bytes of LayerWeights::info for hidden_size (qkv 10,240, z 6,144, b 48, a 48, out hidden_size words, 16-byte aligned
// parts); 0 for an invalid hidden_size.
size_t weight_info_bytes(int64_t hidden_size);
// Records the K2 weight scans of the five projections into w.info (16-byte aligned), each in its own format (`*_scales`);
// the caller puts a barrier after it. CERE_ERR_ARGUMENT for a null / misaligned pointer or hidden_size not a multiple of 64.
cere_status prepare_weights(vk::Context &ctx, const LayerWeights &w, int64_t hidden_size);

// The mixer between the projections: mixed bf16 [tokens, 10240], z [tokens, 6144], b / a [tokens, 48] -> y bf16
// [tokens, 6144] (may alias mixed but no other input). Errors as the CUDA header.
cere_status layer_core(vk::Context &ctx, const LayerWeights &w, LayerState state, uint32_t operation, uint64_t mixed,
                       uint64_t z, uint64_t b, uint64_t a, int64_t tokens, uint64_t y, uint64_t workspace,
                       size_t workspace_bytes, const EndState *end = nullptr);

// hidden bf16 [tokens, hidden_size] -> out bf16 [tokens, hidden_size] (may not alias hidden). `cls`: the projections'
// row class (CUDA layer_forward's cls; 0 = tokens, else tokens <= cls, CERE_ERR_ARGUMENT otherwise). CERE_ERR_ARGUMENT
// before anything is recorded for an MXFP8 in-projection with hidden_size % 32 != 0.
cere_status layer_forward(vk::Context &ctx, const LayerWeights &w, LayerState state, uint32_t operation, uint64_t hidden,
                          int64_t tokens, int64_t hidden_size, uint64_t out, uint64_t workspace, size_t workspace_bytes,
                          uint32_t cls = 0, const EndState *end = nullptr);

}  // namespace cere::vkk5
