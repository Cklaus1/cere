// Host validation of WeightLab interventions (research/patch_check.h).
#include "research/patch_check.h"

#include <cmath>

namespace cere::research {
namespace {

bool dense(Kind kind) {
    return kind == Kind::kAdd || kind == Kind::kAblateZero || kind == Kind::kAblateMean || kind == Kind::kReplace;
}

}  // namespace

cere_status check_edit(Kind kind, bool router, uint32_t point_cols, uint32_t top_k, float alpha, uint32_t col_begin,
                       uint32_t col_end, uint32_t vector_rows, uint32_t vector_cols, const int32_t *experts,
                       const float *weights, uint32_t count) {
    const uint32_t k = uint32_t(kind);
    if (k < uint32_t(Kind::kAdd) || k > uint32_t(Kind::kExpertForce) || point_cols == 0 || !std::isfinite(alpha)) {
        return CERE_ERR_ARGUMENT;
    }
    if (dense(kind)) {
        const bool all = col_begin == 0 && col_end == 0;
        if ((!all && (col_begin >= col_end || col_end > point_cols)) || count != 0 || experts != nullptr ||
            weights != nullptr || (kind != Kind::kAdd && alpha != 0.0f)) {
            return CERE_ERR_ARGUMENT;
        }
        if (kind == Kind::kAblateZero) return vector_rows == 0 && vector_cols == 0 ? CERE_OK : CERE_ERR_ARGUMENT;
        if (vector_cols != point_cols) return CERE_ERR_ARGUMENT;
        if (kind == Kind::kReplace) {
            return vector_rows >= 1 && vector_rows <= kMaxVectorRows ? CERE_OK : CERE_ERR_ARGUMENT;
        }
        return vector_rows == 1 ? CERE_OK : CERE_ERR_ARGUMENT;
    }
    // Expert kinds: no columns, vector or alpha; E = point_cols.
    if ((kind == Kind::kExpertForce) == router || alpha != 0.0f || col_begin != 0 || col_end != 0 ||
        vector_rows != 0 || vector_cols != 0 || point_cols > kMaxExperts || top_k == 0 || top_k > kMaxTopK ||
        top_k > point_cols || count > point_cols || (count > 0 && experts == nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    bool seen[kMaxExperts] = {};
    for (uint32_t i = 0; i < count; ++i) {
        if (experts[i] < 0 || uint32_t(experts[i]) >= point_cols || seen[experts[i]]) return CERE_ERR_ARGUMENT;
        seen[experts[i]] = true;
    }
    switch (kind) {
        case Kind::kExpertMask:
            return weights == nullptr && point_cols - count >= top_k ? CERE_OK : CERE_ERR_ARGUMENT;
        case Kind::kExpertBias:
            if (count > 0 && weights == nullptr) return CERE_ERR_ARGUMENT;
            for (uint32_t i = 0; i < count; ++i) {
                if (!std::isfinite(weights[i])) return CERE_ERR_ARGUMENT;
            }
            return CERE_OK;
        default: {  // kExpertForce
            if (count == 0 || count > top_k || weights == nullptr) return CERE_ERR_ARGUMENT;
            double sum = 0.0;
            for (uint32_t i = 0; i < count; ++i) {
                if (!std::isfinite(weights[i]) || weights[i] < 0.0f) return CERE_ERR_ARGUMENT;
                sum += weights[i];
            }
            return sum <= 1.0 ? CERE_OK : CERE_ERR_ARGUMENT;
        }
    }
}

}  // namespace cere::research
