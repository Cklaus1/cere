// K5 (m22 design note, m22n design note D7): the whole GDN linear-attention mixer of one Qwen3.8-Flash-Next
// linear_attention layer (Qwen3_5GatedDeltaNet.forward), routed by the operation as SGLang's GDN backend routes it:
//
//   mixed = hidden in_proj_qkv^T, z = hidden in_proj_z^T, b = hidden in_proj_b^T, a = hidden in_proj_a^T (K2 GEMMs)
//   PREFILL (forward_extend, any length 1..4096, including 1 token): conv1d + SiLU with the 3-step conv state ->
//     q | k | v; g, beta = gating(a, b); o = delta_rule_chunk (chunk-64 WY form, in-kernel L2 norm of q and k)
//   DECODE (forward_decode, exactly 1 token): conv update -> q | k | v; o = delta_rule_decode (gating and L2 norm
//     in-kernel, FlashInfer's BF16-state decode arithmetic)
//   y = gated RMSNorm per value head (K1, sigmoid gate, eps 1e-6); out = y out_proj^T (K2 GEMM)
//
// Both delta rules use scale 128^-0.5 and keep the state BF16 at rest.
//
// math_reference: flash-linear-attention
// math_reference: flashinfer
// math_reference: sglang
#pragma once

#include "cere_executor.h"
#include "cere_model_constants.h"
#include "kernels/k2_gemm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cere::k5 {

// Model constants of the A4 config (text_config linear_* entries).
constexpr int kKeyHeads = gen::kLinearNumKeyHeads;
constexpr int kValueHeads = gen::kLinearNumValueHeads;
constexpr int kKeyDim = kKeyHeads * gen::kLinearKeyHeadDim;      // 2,048
constexpr int kValueDim = kValueHeads * gen::kLinearValueHeadDim;  // 6,144
constexpr int kConvDim = 2 * kKeyDim + kValueDim;  // 10,240: q | k | v
constexpr float kNormEps = static_cast<float>(gen::kRmsNormEps);  // rms_norm_eps (the gated norm)

// Device pointers of one layer's tensors, checkpoint layouts (`linear_attn.*`): in_proj_qkv [10240, hidden],
// in_proj_z [6144, hidden], in_proj_b / in_proj_a [48, hidden], conv1d [10240, 4] (the [10240, 1, 4] tensor), A_log as
// F32 [48] (SGLang's float32 parameter; the checkpoint value is BF16 and converts exactly), dt_bias [48], norm [128],
// out_proj [hidden, 6144].
struct LayerWeights {
    const __nv_bfloat16 *in_proj_qkv;
    const __nv_bfloat16 *in_proj_z;
    const __nv_bfloat16 *in_proj_b;
    const __nv_bfloat16 *in_proj_a;
    const __nv_bfloat16 *conv1d;
    const float *A_log;
    const __nv_bfloat16 *dt_bias;
    const __nv_bfloat16 *norm;
    const __nv_bfloat16 *out_proj;
};

// GEMM plans (BF16 out) for the four projection shapes: qkv (n 10240, k hidden), z (6144, hidden), ba (48, hidden;
// used for both b and a) and out (hidden, 6144). Plans depend on shapes only and may be shared by all GDN layers.
struct LayerGemms {
    const kernels::GemmPlan *qkv;
    const kernels::GemmPlan *z;
    const kernels::GemmPlan *ba;
    const kernels::GemmPlan *out;
};

// Per-session recurrent state of one layer: conv bf16 [10240, 3] (oldest first), ssm bf16 [48, 128 (v), 128 (k)].
// Zeros = a fresh sequence; both are updated in place.
struct LayerState {
    __nv_bfloat16 *conv;
    __nv_bfloat16 *ssm;
};

// One PREFILL call's end state work (m210 D15⁗′; every member null / -1 = none, the call unchanged). Resume:
// `ssm_in32` FP32 [48, 128, 128] is the state the chunk rule starts from (state.ssm is then only written). Capture at
// `row` (a multiple of 64, 64 <= row < tokens): `ssm_out32` receives the FP32 h as of that row and `conv_out` bf16
// [10240, 3] the conv state there. Neither changes a computed value. A non-null `*_hi` holds value heads 24..47 of
// its FP32 state (k5 ChunkArgs: the base then holds heads 0..23).
struct EndState {
    const float *ssm_in32 = nullptr;
    float *ssm_out32 = nullptr;
    __nv_bfloat16 *conv_out = nullptr;
    int64_t row = -1;
    const float *ssm_in32_hi = nullptr;
    float *ssm_out32_hi = nullptr;
};

// Workspace of `layer_core` for `max_tokens` tokens: conv_out bf16 [T, 10240] (q | k | v; for a PREFILL the three
// contiguous planes q [T, 2048], k [T, 2048], v [T, 6144] that delta_rule_chunk reads, for a DECODE the one conv row),
// o bf16 [T, 6144], g and beta f32 [T, 48], then chunk_workspace_bytes(T) for delta_rule_chunk; every region starts
// 256-byte aligned. 0 when max_tokens is outside 1..kChunkMaxTokens.
size_t core_workspace_bytes(int64_t max_tokens);

// Workspace of `layer_forward` for `max_tokens` tokens: mixed bf16 [T, 10240] (the in_proj_qkv output; dead after
// the conv, then reused for y [T, 6144]), z bf16 [T, 6144], b and a bf16 [T, 48], then core_workspace_bytes(T).
// 0 when max_tokens is outside 1..kChunkMaxTokens.
size_t layer_workspace_bytes(int64_t max_tokens);

// The mixer between the projections, by `operation` (CERE_OP_PREFILL or CERE_OP_DECODE, cere_executor.h):
// mixed bf16 [tokens, 10240], z bf16 [tokens, 6144], b and a bf16 [tokens, 48] (all contiguous, read only) ->
// y bf16 [tokens, 6144] (the gated-norm output, may alias mixed but no other input). Uses w.conv1d, A_log, dt_bias
// and norm only. Enqueued on `stream` with no allocation and no synchronization. CERE_ERR_ARGUMENT for bad
// pointers/shapes, a DECODE of tokens != 1 (the model decodes one token per step) or a too-small workspace,
// CERE_ERR_UNSUPPORTED for CERE_OP_VERIFY, CERE_ERR_INTERNAL for a failed launch.
cere_status layer_core(const LayerWeights &w, LayerState state, uint32_t operation, const __nv_bfloat16 *mixed,
                       const __nv_bfloat16 *z, const __nv_bfloat16 *b, const __nv_bfloat16 *a, int64_t tokens,
                       __nv_bfloat16 *y, void *workspace, size_t workspace_bytes, cudaStream_t stream,
                       const EndState *end = nullptr);

// The most rows of one `chain_core` call, all its chains together: a verify step's rows.
constexpr int64_t kChainMaxRows = 16;

// The verify chains of one `chain_core` call, back to back from row 0: chain c has rows[c] >= 1 consecutive rows of
// the sequence states[c] holds (distinct sequences: no conv or ssm pointer shared between chains), and saves[r] is
// the undo snapshot of row r (conv and ssm, aliasing no state) for every row that is not its chain's last.
struct Chains {
    int64_t count = 0;  // 1..kChainMaxRows
    int64_t rows[kChainMaxRows] = {};
    LayerState states[kChainMaxRows] = {};
    LayerState saves[kChainMaxRows] = {};
};

// Workspace of `chain_core` for `rows` rows: conv_out bf16 [rows, 10240] and o bf16 [rows, 6144], each 256-byte
// aligned. 0 when rows is outside 1..kChainMaxRows.
size_t chain_core_workspace_bytes(int64_t rows);

// The DECODE core over the rows of a verify step's chains (m23 design note DG-02; all chains in one call,
// rows16 design note R16-B): `rows` = the sum of chains.rows, inputs as layer_core's (row r of mixed, z, b, a), y
// bf16 [rows, 6144] (may alias mixed but no other input). Bit-equal, per chain, to its rows' layer_core(DECODE) calls
// in order on states[c]: y, the final state, and saves[r] holding the state row r's call leaves (the undo snapshot
// right before row r + 1), written in-kernel. Three launches: conv1d_silu_rows, delta_rule_decode_rows (both one
// grid slice per chain) and one gated norm over every row. Same errors as layer_core (chains or rows outside
// 1..kChainMaxRows, rows not the sum of chains.rows, a null state, two chains sharing a state, a null save or one
// aliasing a state is CERE_ERR_ARGUMENT).
cere_status chain_core(const LayerWeights &w, const Chains &chains, const __nv_bfloat16 *mixed,
                       const __nv_bfloat16 *z, const __nv_bfloat16 *b, const __nv_bfloat16 *a, int64_t rows,
                       __nv_bfloat16 *y, void *workspace, size_t workspace_bytes, cudaStream_t stream);

// hidden bf16 [tokens, hidden_size] -> out bf16 [tokens, hidden_size] (may not alias hidden): the projections, then
// layer_core(operation), then out_proj; same errors as layer_core plus a GEMM plan not matching the shapes. `cls` as
// kernels::GemmPlan::run (0 = tokens); `end` (nullable, PREFILL only) as EndState.
cere_status layer_forward(const LayerGemms &gemms, const LayerWeights &w, LayerState state, uint32_t operation,
                          const __nv_bfloat16 *hidden, int64_t tokens, int64_t hidden_size, __nv_bfloat16 *out,
                          void *workspace, size_t workspace_bytes, cudaStream_t stream, uint32_t cls = 0,
                          const EndState *end = nullptr);

}  // namespace cere::k5
