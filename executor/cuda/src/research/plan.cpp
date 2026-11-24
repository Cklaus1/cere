#include "research/plan.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace cere::research {
namespace {

constexpr uint64_t kAlign = 256;
constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();

uint64_t align(uint64_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

uint64_t sat_add(uint64_t a, uint64_t b) { return a > kMax - b ? kMax : a + b; }
uint64_t sat_mul(uint64_t a, uint64_t b) { return a != 0 && b > kMax / a ? kMax : a * b; }

bool no_edit_fields(const cere_research_op &op) {
    return op.alpha == 0.0f && op.col_begin == 0 && op.col_end == 0 &&
           op.vector_rows == 0 && op.vector == nullptr && op.vector_cols == 0 && op.expert_count == 0 &&
           op.experts == nullptr && op.weights == nullptr;
}

// Whether `point` takes intervention `kind`.
bool takes(const PointInfo &point, uint32_t kind) {
    switch (kind) {
        case CERE_RESEARCH_ADD:
        case CERE_RESEARCH_ABLATE_ZERO:
        case CERE_RESEARCH_ABLATE_MEAN:
        case CERE_RESEARCH_REPLACE:
            return point.dense;
        case CERE_RESEARCH_EXPERT_BIAS:
        case CERE_RESEARCH_EXPERT_MASK:
            return point.router;
        case CERE_RESEARCH_EXPERT_FORCE:
            return point.route;
        default:
            return false;
    }
}

cere_status load_op(const cere_research_op &in, Op *out, uint64_t *operand_bytes) {
    if (in.kind > CERE_RESEARCH_EXPERT_FORCE) return CERE_ERR_ARGUMENT;
    const PointInfo *point = find_point(in.point);
    if (point == nullptr) return CERE_ERR_ARGUMENT;
    if (in.layers == 0 || (in.layers & ~point_layers(*point)) != 0) return CERE_ERR_ARGUMENT;
    if (in.prompt_count > CERE_RESEARCH_MAX_RANGES || (in.prompt_count != 0 && in.prompt == nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    if (in.generated > 1 || (in.generated == 0 && in.generated_max != 0)) return CERE_ERR_ARGUMENT;
    if (in.prompt_count == 0 && in.generated == 0) return CERE_ERR_ARGUMENT;
    for (uint32_t i = 0; i < in.prompt_count; ++i) {
        const cere_research_range &r = in.prompt[i];
        if (r.end <= r.first || (i > 0 && r.first < in.prompt[i - 1].end)) return CERE_ERR_ARGUMENT;
    }
    // Logits points: scored rows only (generated positions).
    if (point->logits && in.prompt_count != 0) return CERE_ERR_ARGUMENT;
    if (point->id == PointId::kLogits &&
        (in.generated_max == 0 || in.generated_max > CERE_RESEARCH_MAX_LOGIT_POSITIONS)) {
        return CERE_ERR_ARGUMENT;
    }
    if (point->id == PointId::kLogitsTopk ? (in.top_k == 0 || in.top_k > CERE_RESEARCH_MAX_TOPK) : in.top_k != 0) {
        return CERE_ERR_ARGUMENT;
    }
    Op op;
    op.kind = in.kind;
    op.point = point;
    op.layers = in.layers;
    op.top_k = in.top_k;
    op.prompt.reserve(in.prompt_count);
    for (uint32_t i = 0; i < in.prompt_count; ++i) op.prompt.push_back(Interval{in.prompt[i].first, in.prompt[i].end});
    op.generated = in.generated != 0;
    op.generated_max = in.generated_max;
    if (in.kind == CERE_RESEARCH_CAPTURE) {
        if (!no_edit_fields(in)) return CERE_ERR_ARGUMENT;
        *out = std::move(op);
        return CERE_OK;
    }
    if (!takes(*point, in.kind)) return CERE_ERR_ARGUMENT;
    if ((in.vector_rows != 0) != (in.vector != nullptr)) return CERE_ERR_ARGUMENT;
    if (in.expert_count != 0 && in.experts == nullptr) return CERE_ERR_ARGUMENT;
    if (const cere_status st = check_edit(Kind(in.kind), point->router, point->edit_cols, kTopK, in.alpha,
                                          in.col_begin, in.col_end, in.vector_rows, in.vector_cols, in.experts,
                                          in.weights, in.expert_count);
        st != CERE_OK) {
        return st;
    }
    op.alpha = in.alpha;
    op.col_begin = in.col_begin;
    op.col_end = in.col_end;
    op.vector_rows = in.vector_rows;
    op.vector_cols = in.vector_cols;
    uint64_t offset = *operand_bytes;
    if (in.vector_rows != 0) {
        const uint64_t elements = uint64_t(in.vector_rows) * in.vector_cols;
        if (elements * 2 > CERE_RESEARCH_MAX_PLAN_BYTES) return CERE_ERR_ARGUMENT;
        op.vector.resize(elements);
        std::memcpy(op.vector.data(), in.vector, elements * 2);
        op.vector_offset = offset;
        offset += align(elements * 2);
    }
    if (in.expert_count != 0) {
        op.experts.assign(in.experts, in.experts + in.expert_count);
        op.experts_offset = offset;
        offset += align(uint64_t(in.expert_count) * sizeof(int32_t));
        if (in.weights != nullptr) {
            op.weights.assign(in.weights, in.weights + in.expert_count);
            op.weights_offset = offset;
            offset += align(uint64_t(in.expert_count) * sizeof(float));
        }
    }
    if (offset > CERE_RESEARCH_MAX_PLAN_BYTES) return CERE_ERR_ARGUMENT;
    *operand_bytes = offset;
    *out = std::move(op);
    return CERE_OK;
}

}  // namespace

cere_status load_plan(const cere_research_plan *in, Plan *out) {
    if (in == nullptr || out == nullptr || in->ops == nullptr || in->op_count == 0 ||
        in->op_count > CERE_RESEARCH_MAX_OPS || in->reserved != 0) {
        return CERE_ERR_ARGUMENT;
    }
    Plan plan;
    plan.ops.resize(in->op_count);
    for (uint32_t i = 0; i < in->op_count; ++i) {
        if (const cere_status st = load_op(in->ops[i], &plan.ops[i], &plan.operand_bytes); st != CERE_OK) return st;
        const Op &op = plan.ops[i];
        plan.point_layers[size_t(op.point->id)] |= op.layers;
        if (op.captures()) {
            plan.captures = true;
            plan.captures_prompt = plan.captures_prompt || !op.prompt.empty();
        } else {
            plan.edits = true;
        }
    }
    std::memcpy(plan.sha256.data(), in->sha256, plan.sha256.size());
    *out = std::move(plan);
    return CERE_OK;
}

std::vector<Interval> select(const Op &op, uint64_t prompt_tokens, uint64_t max_tokens) {
    std::vector<Interval> out;
    for (const Interval &r : op.prompt) {
        if (r.first >= prompt_tokens) break;
        out.push_back(Interval{r.first, std::min(r.end, prompt_tokens)});
    }
    if (op.generated && prompt_tokens != 0) {
        const uint64_t g = op.generated_max == 0 ? max_tokens : std::min(op.generated_max, max_tokens);
        if (g != 0) {
            const uint64_t first = prompt_tokens - 1;
            const uint64_t end = sat_add(first, g);
            // Prompt ranges end at or before prompt_tokens, so only the last can reach row prompt_tokens - 1.
            if (!out.empty() && out.back().end >= first) {
                out.back().end = std::max(out.back().end, end);
            } else {
                out.push_back(Interval{first, end});
            }
        }
    }
    // Merge ranges that touch (a later range starting at an earlier one's end).
    std::vector<Interval> merged;
    for (const Interval &r : out) {
        if (!merged.empty() && merged.back().end == r.first) {
            merged.back().end = r.end;
        } else {
            merged.push_back(r);
        }
    }
    return merged;
}

uint64_t count(const std::vector<Interval> &selection) {
    uint64_t n = 0;
    for (const Interval &r : selection) n = sat_add(n, r.end - r.first);
    return n;
}

uint64_t bound(const Plan &plan, uint64_t prompt_tokens, uint64_t max_tokens) {
    uint64_t bytes = 0;
    for (const Op &op : plan.ops) {
        if (!op.captures()) continue;
        const uint64_t rows = count(select(op, prompt_tokens, max_tokens));
        const uint64_t layers = uint64_t(__builtin_popcountll(op.layers));
        bytes = sat_add(bytes, sat_mul(sat_mul(rows, layers), row_bytes(*op.point, op.top_k)));
    }
    return bytes;
}

std::vector<uint32_t> layers_of(uint64_t mask) {
    std::vector<uint32_t> out;
    for (uint32_t L = 0; L <= kLayers; ++L) {
        if ((mask >> L) & 1u) out.push_back(L);
    }
    return out;
}

}  // namespace cere::research
