// WeightLab intervention kinds and their host validation (weightlab design note "Interventions", WL-05..08, track
// WL-B). Host-only C++ (no CUDA types): WL-A's plan_load calls check_edit for every intervention op, in the CUDA
// library and the toy build alike. The device side is research/patch.h.
#pragma once

#include "cere_executor.h"

#include <cstdint>

namespace cere::research {

// The intervention kinds, equal to cere_executor.h's CERE_RESEARCH_ADD ... CERE_RESEARCH_EXPERT_FORCE.
enum class Kind : uint32_t {
    kAdd = CERE_RESEARCH_ADD,
    kAblateZero = CERE_RESEARCH_ABLATE_ZERO,
    kAblateMean = CERE_RESEARCH_ABLATE_MEAN,
    kReplace = CERE_RESEARCH_REPLACE,
    kExpertBias = CERE_RESEARCH_EXPERT_BIAS,
    kExpertMask = CERE_RESEARCH_EXPERT_MASK,
    kExpertForce = CERE_RESEARCH_EXPERT_FORCE,
};

// Limits check_edit / apply_* enforce.
inline constexpr uint32_t kMaxRanges = CERE_RESEARCH_MAX_RANGES;  // RowSet ranges; more: one apply call per 64
inline constexpr uint32_t kMaxExperts = 1024;                     // kernels::k7 kMaxExperts
inline constexpr uint32_t kMaxTopK = 32;                          // kernels::k7 kMaxTopK
inline constexpr uint32_t kMaxVectorRows = 4096;

// Host validation of one intervention against its point (CERE_ERR_ARGUMENT on any violation): dense kinds (add,
// ablate_*, replace) at any point of `point_cols` (`router` either), 0 <= col_begin < col_end <= point_cols or both 0
// (every column), vector_cols == point_cols, add / ablate_mean vector_rows == 1, replace 1 <= vector_rows <=
// kMaxVectorRows, ablate_zero no vector, alpha finite (add only; other kinds alpha 0), no experts or weights;
// expert_bias / expert_mask `router` true (the router-logit point, point_cols = the expert count E), expert_force
// `router` false (its point is the routing, E passed as point_cols); expert kinds no columns, vector or alpha,
// 0 <= count <= E distinct ids in [0, E) (host arrays), `top_k` the model's MoE top_k (1..kMaxTopK, <= E),
// expert_mask no weights and leaving >= top_k experts unmasked, expert_bias finite weights (the bias per id),
// expert_force 1 <= count <= top_k with weights finite, >= 0 and summing to <= 1.
cere_status check_edit(Kind kind, bool router, uint32_t point_cols, uint32_t top_k, float alpha, uint32_t col_begin,
                       uint32_t col_end, uint32_t vector_rows, uint32_t vector_cols, const int32_t *experts,
                       const float *weights, uint32_t count);

}  // namespace cere::research
