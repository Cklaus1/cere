// WeightLab research plans (weightlab design note WL-02, cere_executor.h "WeightLab research"): the validated host
// copy of a cere_research_plan, the positions each op selects and the capture bound. Host-only C++ shared by the CUDA
// library and the CPU toy, so both validate, select and account alike; the Rust admission bound
// (crates/cere-runtime/src/native/research.rs) is the same formula as bound() below.
#pragma once

#include "cere_executor.h"
#include "research/patch_check.h"
#include "research/points.h"

#include <array>
#include <cstdint>
#include <vector>

namespace cere::research {

// [first, end) of absolute positions.
struct Interval {
    uint64_t first = 0;
    uint64_t end = 0;
};

// One op of a loaded plan, its operands copied.
struct Op {
    uint32_t kind = CERE_RESEARCH_CAPTURE;
    const PointInfo *point = nullptr;
    uint64_t layers = 0;  // bit L (bit 48: the final points), a non-empty subset of point_layers(*point)
    uint32_t top_k = 0;   // native.logits.topk only
    std::vector<Interval> prompt;
    bool generated = false;
    uint64_t generated_max = 0;  // 0 = max_tokens
    float alpha = 0.0f;
    uint32_t col_begin = 0;
    uint32_t col_end = 0;
    uint32_t vector_rows = 0;
    uint32_t vector_cols = 0;
    std::vector<uint16_t> vector;  // BF16 bits [vector_rows, vector_cols]
    std::vector<int32_t> experts;
    std::vector<float> weights;
    // Byte offsets of the operands in the slot's device plan memory (256-aligned; meaningful when present).
    uint64_t vector_offset = 0;
    uint64_t experts_offset = 0;
    uint64_t weights_offset = 0;

    bool captures() const { return kind == CERE_RESEARCH_CAPTURE; }
    Kind edit_kind() const { return Kind(kind); }
};

struct Plan {
    std::vector<Op> ops;
    std::array<uint8_t, 32> sha256{};
    bool captures = false;         // some op captures
    bool captures_prompt = false;  // some capture op names prompt ranges (its sessions skip the prefix lookup)
    bool edits = false;            // some op intervenes
    // Per PointId, the layers some op acts at (the forward's "does this call act at the point" test).
    std::array<uint64_t, size_t(PointId::kCount)> point_layers{};
    // Bytes of the operands in device plan memory (each 256-aligned), at most CERE_RESEARCH_MAX_PLAN_BYTES.
    uint64_t operand_bytes = 0;

    bool acts(PointId point, uint32_t layer) const { return (point_layers[size_t(point)] >> layer) & 1u; }
};

// Validates `in` against the point table and WL-B's check_edit and copies it into *out (every pointer read before
// return). CERE_ERR_ARGUMENT on anything cere_executor.h refuses: a null plan or ops, op_count 0 or above
// CERE_RESEARCH_MAX_OPS, `reserved` not 0, an unknown kind or point, an empty layer mask or one with a layer the point
// lacks, more than CERE_RESEARCH_MAX_RANGES prompt ranges, an empty / unsorted / overlapping range, `generated` other
// than 0 / 1, generated_max without `generated`, an op with no positions, prompt ranges at a logits point,
// native.logits without generated_max in [1, CERE_RESEARCH_MAX_LOGIT_POSITIONS], top_k outside [1, 64] at
// native.logits.topk (or non-zero elsewhere), a capture op with an intervention field, an intervention at a point
// that does not take its kind, check_edit's refusals, or operands above CERE_RESEARCH_MAX_PLAN_BYTES.
cere_status load_plan(const cere_research_plan *in, Plan *out);

// The positions op `op` selects for a request of `prompt_tokens` (>= 1) and `max_tokens`: its prompt ranges cut at
// prompt_tokens, united with the generated rows [prompt_tokens - 1, prompt_tokens - 1 + g), g = min(generated_max or
// max_tokens, max_tokens). Sorted, disjoint intervals (adjacent ones merged).
std::vector<Interval> select(const Op &op, uint64_t prompt_tokens, uint64_t max_tokens);
uint64_t count(const std::vector<Interval> &selection);

// The capture bound of a request: the sum over capture ops and their layers of |select(op)| x row_bytes. The Rust
// admission bound for the same plan, prompt_tokens and max_tokens is equal.
uint64_t bound(const Plan &plan, uint64_t prompt_tokens, uint64_t max_tokens);

// Layers of a mask in ascending order (bit 48 last).
std::vector<uint32_t> layers_of(uint64_t mask);

}  // namespace cere::research
