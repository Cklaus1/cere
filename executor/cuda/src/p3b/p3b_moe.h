// P3b (m23 design note P3, M23-D1..D3): the MoE block of one layer with FlashInfer 0.6.17's SM120 CUTLASS fused
// MoE (CutlassMoeFCRunner<NVFP4, NVFP4, BF16, BF16>, vendored under executor/third_party/) for the routed experts.
// Built only with -DCERE_MOE_BACKEND=p3b; the call sites use CERE_MOE_BLOCK (model/moe_backend.h).
//
// Same contract as kernels::moe_block (k7_moe.h), with the layer in the P3b arena layout (M23-D2):
// * `up_weight` = w13 [E, 2I, H/2]: per expert [up; gate] (FlashInfer's w13 order, SGLang
//   `load_up_proj_weight_first`); `gate_weight` = up_weight + I * H/2 (expert stride 2I * H/2; K7 cannot read it);
// * `up_scale` = the w13 block scales [E, 2I, H/16] and `down_scale` [E, H, I/16], swizzled in place at INSTANTIATE
//   (prepare_layer; SGLang `swizzle_blockscale`); `gate_scale` = up_scale + I * H/16;
// * `act_scales` = device FP32 {a1, a2} (FlashInfer reads the activation global scales from device memory).
// Router: K7's (K2 router plan + moe_route), so ids and weights equal K7's. Routed experts: runMoe with the pinned
// tactic pair of m's bucket (p3b_tactic.h kPinnedBuckets) and fused finalize off; its finalized BF16 output is the
// routed row. Shared expert and combine (M23-D3, SGLang's order: `fused_gate_sigmoid_mul_add` onto the finalized
// routed output): K7's routed_in path (k7_moe.h MoeTaps), out = BF16(routed + sigmoid(x . g) * shared), FP32 inside.
#pragma once

#include "cere_executor.h"
#include "kernels/k7_moe.h"
#include "p3b/p3b_decode.h"
#include "p3b/p3b_tactic.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cere::p3b {

// INSTANTIATE: creates the current device's runner (once per device ordinal, m32n design note D4: each card has its
// own runner and mutex, never shared across ordinals) and resolves every bucket's pinned GEMM1 / GEMM2 tactics against
// its own lists. CERE_ERR_UNSUPPORTED when any resolves to zero or several tactics, or the device has none;
// CERE_ERR_INTERNAL when the runner cannot be built or the current ordinal has no slot. Idempotent per device (the
// first result is kept). Every entry point below uses the current device's runner.
cere_status instantiate();

// D4's refusal, for the ktest: the status the runner slot of `ordinal` has on the current device (built on first use
// with the current device): CERE_ERR_INTERNAL when it was built on another ordinal (a runner shared across cards is
// refused) or `ordinal` has no slot; `*device` (nullable) = the ordinal it was built on, -1 if none.
cere_status runner_binding(int ordinal, int *device);

// The runner's candidate lists as Tactics (fused finalize off), for the tactics ktest. `*n1` / `*n2` = list sizes;
// at most `cap` entries of each are written. Requires instantiate() == CERE_OK.
cere_status tactic_lists(Tactic *g1, size_t *n1, Tactic *g2, size_t *n2, size_t cap);

// The resolved indices (into tactic_lists) of the pinned pair of m's bucket (m >= 1). Requires instantiate() ==
// CERE_OK.
cere_status pinned_indices(uint32_t m, TacticIndices *pair);

// Swizzles `batches` row-major FP8 block-scale matrices [rows, cols] in place (rows % 128 == 0, cols % 4 == 0;
// SGLang `swizzle_blockscale`: (B, rows/128, 4, 32, cols/4, 4) -> permute(0, 1, 4, 3, 2, 5)), staging whole
// matrices through `scratch` (256-byte aligned). CERE_ERR_UNSUPPORTED when one matrix exceeds the scratch.
cere_status swizzle_scales(uint8_t *scales, uint64_t batches, uint32_t rows, uint32_t cols, void *scratch,
                           size_t scratch_bytes, cudaStream_t stream);

// INSTANTIATE, once per layer after the weights are loaded: swizzles the layer's w13 and down block scales in place.
cere_status prepare_layer(const kernels::MoeDims &dims, const kernels::MoeLayer &layer, void *scratch,
                          size_t scratch_bytes, cudaStream_t stream);

// Workspace bytes for up to max_m tokens (0 for invalid dims): K7's (shared expert and combine) + the router outputs,
// the routed BF16 rows and the runner's workspace.
size_t moe_workspace_bytes(const kernels::MoeDims &dims, uint32_t max_m);

// Workspace bytes of routed_experts for up to max_m tokens (0 for invalid dims): the runner's workspace and the
// source-to-destination map.
size_t routed_workspace_bytes(const kernels::MoeDims &dims, uint32_t max_m);

// The routed experts alone: runMoe on the given routing (`topk_ids` int32 / `topk_weights` FP32 [m, top_k]), its
// finalized BF16 rows into `out` [m, H]. `tactics` null: m's pinned pair; else that pair of indices into tactic_lists
// (the P3b tactic-sweep ktest; CERE_ERR_ARGUMENT when out of range). A tactic the runner cannot run fails with
// CERE_ERR_INTERNAL. Graph-capturable (no host synchronization).
cere_status routed_experts(const kernels::MoeDims &dims, const kernels::MoeLayer &layer, const __nv_bfloat16 *x,
                           const int32_t *topk_ids, const float *topk_weights, uint32_t m, __nv_bfloat16 *out,
                           void *workspace, size_t workspace_bytes, cudaStream_t stream,
                           const TacticIndices *tactics = nullptr);

// The expert rows routed_rows (p3b_rows.h) leaves in its workspace (valid until the workspace is reused).
struct Rows {
    const __nv_bfloat16 *rows = nullptr;  // [m * top_k, hidden], permuted (grouped by expert)
    const int32_t *pos = nullptr;         // [top_k, m]: row of (token t, slot k) = pos[k * m + t]
    // ktest: GEMM1's A (`x` FP4 [m * top_k, hidden / 2], `x_sf` per-expert swizzled; overwritten by GEMM2's A unless
    // `decoded`) and GEMM2's A as the path left it (`act` FP4 [m * top_k, inter / 2]; `act_sf` compact [m * top_k,
    // inter / 16] when `decoded`, else per-expert swizzled, getOffsetActivationSF); `offsets` =
    // expert_first_token_offset [experts + 1]; `groups` the decode group table (null unless `decoded`).
    const uint8_t *x = nullptr;
    const uint8_t *x_sf = nullptr;
    const uint8_t *act = nullptr;
    const uint8_t *act_sf = nullptr;
    const int64_t *offsets = nullptr;
    const decode::Group *groups = nullptr;
    bool decoded = false;
};

// routed_rows' GEMMs: kAuto (the model's) the decode GEMMs up to decode::kRoutedMaxRows, kGrouped FlashInfer's grouped
// GEMMs at every m, kDecode the decode GEMMs wherever decode::supported.
enum class RowsPath : int { kAuto = 0, kGrouped = 1, kDecode = 2 };

// ktest (m23 design note P14): routed_rows with m's pinned pair on the given routing and `path` (MMA chain `order`),
// into `*out`. Workspace: routed_workspace_bytes(dims, m). Graph-capturable.
cere_status routed_rows_view(const kernels::MoeDims &dims, const kernels::MoeLayer &layer, const __nv_bfloat16 *x,
                             const int32_t *topk_ids, const float *topk_weights, uint32_t m, RowsPath path,
                             decode::Order order, void *workspace, size_t workspace_bytes, cudaStream_t stream,
                             Rows *out);

// ktest: the model's routed_rows (RowsPath::kAuto) runs an m-token call on the decode GEMMs.
bool decode_routed(const kernels::MoeDims &dims, uint32_t m);

// ktest: the decode GEMMs alone on the layer's weights, scales and global scales (FlashInfer's environment-selected
// FP4 quantize fast math), as routed_rows launches them (programmatic dependents).
cere_status decode_gemm1(const kernels::MoeDims &dims, const kernels::MoeLayer &layer, uint32_t m,
                         const decode::Group *groups, const uint8_t *x, const uint8_t *x_sf, uint8_t *act,
                         uint8_t *act_sf, decode::Order order, cudaStream_t stream);
cere_status decode_gemm2(const kernels::MoeDims &dims, const kernels::MoeLayer &layer, uint32_t m,
                         const decode::Group *groups, const uint8_t *act, const uint8_t *act_sf, __nv_bfloat16 *out,
                         decode::Order order, cudaStream_t stream);

// kernels::moe_block's contract (k7_moe.h), P3b layout. Taps: `routed` receives the finalized routed rows; with
// `routed_in` given the block is K7's (no routed expert runs); router hook points and the post-route hook (`route`)
// as K7's. `topk_ids` (int32 [m, top_k]) / `topk_weights` (FP32 [m, top_k]), nullable, receive the routing the
// routed experts run (the router's output after `route`; the P3b pair ktest feeds them to FlashInfer's own
// cutlass_fused_moe). `tactics` as routed_experts' (null in the executor).
cere_status moe_block(const kernels::MoeDims &dims, const kernels::MoeLayer &layer, const kernels::MoePlans &plans,
                      const __nv_bfloat16 *x, uint32_t m, __nv_bfloat16 *out, void *workspace, size_t workspace_bytes,
                      cudaStream_t stream, const kernels::MoeTaps *taps = nullptr, int32_t *topk_ids = nullptr,
                      float *topk_weights = nullptr, const TacticIndices *tactics = nullptr);

}  // namespace cere::p3b
