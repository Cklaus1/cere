// WeightLab interventions on the device (weightlab design note "Interventions", WL-05..08, track WL-B): the
// elementwise edits a research plan applies at a P point and the expert force applied through the MoE post-route hook
// (kernels::MoeTaps.route). Kinds, limits and the host validation: research/patch_check.h. The plan parser and the
// per-call row selection are WL-A's (research/); this module only edits the rows it is given. No allocation and no
// synchronization; every launch is on `stream`.
//
// Arithmetic (all kinds bit-reproducible: per element, no reduction, fixed order):
//   add          y = bf16_rn(f32(y) + d), d = alpha * f32(v[c]) (both FP32 round-to-nearest, no contraction);
//                d == 0 leaves y's bits unchanged (alpha 0 is an exact no-op, -0 kept).
//   ablate_zero  y = +0.
//   ablate_mean  y = v[c].
//   replace      y = v[s, c], s = the row's source row (RowRange.src + offset; a [1, cols] vector broadcasts).
//   expert_bias  logits[e] = bf16_rn(f32(logits[e]) + b) for each listed (e, b); b == 0 leaves the bits unchanged.
//   expert_mask  logits[e] = -3.3895314e38 (0xFF7F, the most negative finite BF16; never -Inf: moe_route's NaN/+Inf
//                guard is the only non-finite handling). An empty list is a no-op.
//   expert_force (route hook) per selected row: slots 0..k-1 = the forced (experts[i], weights[i]); slots k..K-1 =
//                the router's picks in slot order that are not forced, first K-k of them, each weight w_j scaled as
//                f32(w_j * r) with r = (1 - sum_i weights[i]) / sum_j w_j (FP32, sums in slot order; r = 0 when either
//                sum is not positive). k == K replaces the whole routing.
// Columns: the dense kinds edit columns [col_begin, col_end) of the point (head-level ablation through
// gemm.attn_qkv.out / gemm.gdn_*.out column ranges), col_begin == col_end == 0 every column; the vector is always
// [cols] or [rows, cols] of the whole point. Points and their cols are WL-A's (cere_executor.h research block): the
// logit kinds act at native.layers.L.gemm.moe_router.out, expert_force at native.layers.L.moe_route.
#pragma once

#include "research/patch_check.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace cere::research {

// Rows [begin, end) of the call (0 <= begin < end <= n); `src` is the vector row of `begin` for replace.
struct RowRange {
    uint32_t begin = 0;
    uint32_t end = 0;
    uint32_t src = 0;
};

// The rows of one call an edit applies to: disjoint ranges, passed by value to the kernels (no device memory).
struct RowSet {
    RowRange range[kMaxRanges];
    uint32_t count = 0;
};

// One intervention at one point of one layer, its operands already on the device.
struct Edit {
    Kind kind = Kind::kAdd;
    float alpha = 0.0f;                        // add
    uint32_t col_begin = 0;                    // dense kinds: [col_begin, col_end) of the point's cols (0, 0: all)
    uint32_t col_end = 0;
    const __nv_bfloat16 *vector = nullptr;     // add / ablate_mean: [cols]; replace: [vector_rows, cols]
    uint32_t vector_rows = 0;
    const int32_t *experts = nullptr;          // expert kinds: distinct ids [count]
    const float *weights = nullptr;            // expert_bias: the bias per id; expert_force: the forced weights
    uint32_t count = 0;
};

// The dense and logit kinds on `values` [n, cols] (device BF16, the native tensor of the point) in place, rows of
// `rows` only (the others keep their bits). CERE_ERR_ARGUMENT for expert_force, an empty or out-of-range RowSet, a
// replace source row past vector_rows, or a null operand the kind needs; CERE_ERR_INTERNAL on a launch failure.
cere_status apply_point(const Edit &edit, __nv_bfloat16 *values, uint32_t n, uint32_t cols, const RowSet &rows,
                        cudaStream_t stream);

// expert_force on the routing ids int32 / weights FP32 [n, top_k] (device, in place; kernels::MoeTaps.route), rows
// of `rows` only. CERE_ERR_ARGUMENT for another kind, count outside [1, top_k] or an invalid RowSet.
cere_status apply_route(const Edit &edit, int32_t *topk_ids, float *topk_weights, uint32_t n, uint32_t top_k,
                        const RowSet &rows, cudaStream_t stream);

}  // namespace cere::research
