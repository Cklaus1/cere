// WeightLab interventions (research/patch.h; weightlab design note WL-05..08). Explicit _rn intrinsics keep every
// add and product a separately rounded FP32 operation whatever the compiler's contraction flags.
#include "research/patch.h"

#include <cstdint>

namespace cere::research {
namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kForceThreads = 128;
constexpr uint16_t kMostNegativeFiniteBf16 = 0xFF7F;

bool dense(Kind kind) {
    return kind == Kind::kAdd || kind == Kind::kAblateZero || kind == Kind::kAblateMean || kind == Kind::kReplace;
}

// The rows the RowSet selects (0 for an invalid set: count 0 or past kMaxRanges, an empty, overlapping, unordered or
// out-of-range range).
uint64_t selected(const RowSet &rows, uint32_t n) {
    if (rows.count == 0 || rows.count > kMaxRanges) return 0;
    uint64_t total = 0;
    uint32_t floor = 0;
    for (uint32_t j = 0; j < rows.count; ++j) {
        const RowRange &r = rows.range[j];
        if (r.begin >= r.end || r.end > n || r.begin < floor) return 0;
        floor = r.end;
        total += r.end - r.begin;
    }
    return total;
}

// Block i of a launch over the selected rows: its call row and its source row (replace).
__device__ __forceinline__ bool locate(const RowSet &rs, uint32_t i, uint32_t *row, uint32_t *offset_src) {
    for (uint32_t j = 0; j < rs.count; ++j) {
        const uint32_t len = rs.range[j].end - rs.range[j].begin;
        if (i < len) {
            *row = rs.range[j].begin + i;
            *offset_src = rs.range[j].src + i;
            return true;
        }
        i -= len;
    }
    return false;
}

__global__ void dense_kernel(uint32_t kind, float alpha, uint32_t col_begin, uint32_t col_end,
                             const __nv_bfloat16 *__restrict__ vector, uint32_t vector_rows,
                             __nv_bfloat16 *__restrict__ values, uint32_t cols, RowSet rs) {
    uint32_t r = 0;
    uint32_t s = 0;
    if (!locate(rs, blockIdx.x, &r, &s)) return;
    __nv_bfloat16 *row = values + uint64_t(r) * cols;
    const __nv_bfloat16 *v = vector == nullptr ? nullptr : vector + (vector_rows > 1 ? uint64_t(s) * cols : 0);
    for (uint32_t c = col_begin + threadIdx.x; c < col_end; c += blockDim.x) {
        switch (static_cast<Kind>(kind)) {
            case Kind::kAdd: {
                const float d = __fmul_rn(alpha, __bfloat162float(v[c]));
                if (d != 0.0f) row[c] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(row[c]), d));
                break;
            }
            case Kind::kAblateZero:
                row[c] = __ushort_as_bfloat16(0);
                break;
            case Kind::kAblateMean:
            case Kind::kReplace:
                row[c] = v[c];
                break;
            default:
                break;
        }
    }
}

__global__ void logit_kernel(uint32_t kind, const int32_t *__restrict__ experts, const float *__restrict__ weights,
                             uint32_t count, __nv_bfloat16 *__restrict__ values, uint32_t cols, RowSet rs) {
    uint32_t r = 0;
    uint32_t s = 0;
    if (!locate(rs, blockIdx.x, &r, &s)) return;
    __nv_bfloat16 *row = values + uint64_t(r) * cols;
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        const int32_t e = experts[i];
        if (static_cast<Kind>(kind) == Kind::kExpertMask) {
            row[e] = __ushort_as_bfloat16(kMostNegativeFiniteBf16);
        } else {
            const float b = weights[i];
            if (b != 0.0f) row[e] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(row[e]), b));
        }
    }
}

__global__ void force_kernel(const int32_t *__restrict__ experts, const float *__restrict__ forced, uint32_t k,
                             int32_t *__restrict__ ids, float *__restrict__ weights, uint32_t top_k, uint32_t total,
                             RowSet rs) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    uint32_t r = 0;
    uint32_t s = 0;
    if (!locate(rs, i, &r, &s)) return;
    int32_t *row_ids = ids + uint64_t(r) * top_k;
    float *row_w = weights + uint64_t(r) * top_k;
    int32_t kept_id[kMaxTopK];
    float kept_w[kMaxTopK];
    uint32_t kept = 0;
    float kept_sum = 0.0f;
    for (uint32_t j = 0; j < top_k && kept + k < top_k; ++j) {
        const int32_t id = row_ids[j];
        bool hit = false;
        for (uint32_t f = 0; f < k; ++f) hit = hit || experts[f] == id;
        if (hit) continue;
        kept_id[kept] = id;
        kept_w[kept] = row_w[j];
        kept_sum = __fadd_rn(kept_sum, row_w[j]);
        ++kept;
    }
    float forced_sum = 0.0f;
    for (uint32_t f = 0; f < k; ++f) forced_sum = __fadd_rn(forced_sum, forced[f]);
    const float rest = __fsub_rn(1.0f, forced_sum);
    const float scale = rest > 0.0f && kept_sum > 0.0f ? __fdiv_rn(rest, kept_sum) : 0.0f;
    for (uint32_t f = 0; f < k; ++f) {
        row_ids[f] = experts[f];
        row_w[f] = forced[f];
    }
    for (uint32_t t = 0; t < kept; ++t) {
        row_ids[k + t] = kept_id[t];
        row_w[k + t] = __fmul_rn(kept_w[t], scale);
    }
}

bool launched() { return cudaGetLastError() == cudaSuccess; }

}  // namespace

cere_status apply_point(const Edit &edit, __nv_bfloat16 *values, uint32_t n, uint32_t cols, const RowSet &rows,
                        cudaStream_t stream) {
    const uint64_t total = selected(rows, n);
    if (values == nullptr || n == 0 || cols == 0 || total == 0) return CERE_ERR_ARGUMENT;
    const uint32_t kind = uint32_t(edit.kind);
    if (dense(edit.kind)) {
        const bool all = edit.col_begin == 0 && edit.col_end == 0;
        const uint32_t col_begin = all ? 0 : edit.col_begin;
        const uint32_t col_end = all ? cols : edit.col_end;
        if (col_begin >= col_end || col_end > cols) return CERE_ERR_ARGUMENT;
        if (edit.kind != Kind::kAblateZero && (edit.vector == nullptr || edit.vector_rows == 0)) {
            return CERE_ERR_ARGUMENT;
        }
        if (edit.kind == Kind::kReplace && edit.vector_rows > 1) {
            for (uint32_t j = 0; j < rows.count; ++j) {
                const RowRange &r = rows.range[j];
                if (uint64_t(r.src) + (r.end - r.begin) > edit.vector_rows) return CERE_ERR_ARGUMENT;
            }
        }
        dense_kernel<<<unsigned(total), kThreads, 0, stream>>>(kind, edit.alpha, col_begin, col_end, edit.vector,
                                                               edit.vector_rows, values, cols, rows);
        return launched() ? CERE_OK : CERE_ERR_INTERNAL;
    }
    if (edit.kind != Kind::kExpertBias && edit.kind != Kind::kExpertMask) return CERE_ERR_ARGUMENT;
    if (edit.count == 0) return CERE_OK;  // an empty mask or bias: nothing to launch
    if (edit.count > cols || edit.experts == nullptr || (edit.kind == Kind::kExpertBias && edit.weights == nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    logit_kernel<<<unsigned(total), kThreads, 0, stream>>>(kind, edit.experts, edit.weights, edit.count, values, cols,
                                                           rows);
    return launched() ? CERE_OK : CERE_ERR_INTERNAL;
}

cere_status apply_route(const Edit &edit, int32_t *topk_ids, float *topk_weights, uint32_t n, uint32_t top_k,
                        const RowSet &rows, cudaStream_t stream) {
    const uint64_t total = selected(rows, n);
    if (edit.kind != Kind::kExpertForce || topk_ids == nullptr || topk_weights == nullptr || top_k == 0 ||
        top_k > kMaxTopK || edit.count == 0 || edit.count > top_k || edit.experts == nullptr ||
        edit.weights == nullptr || total == 0) {
        return CERE_ERR_ARGUMENT;
    }
    const unsigned blocks = unsigned((total + kForceThreads - 1) / kForceThreads);
    force_kernel<<<blocks, kForceThreads, 0, stream>>>(edit.experts, edit.weights, edit.count, topk_ids, topk_weights,
                                                       top_k, uint32_t(total), rows);
    return launched() ? CERE_OK : CERE_ERR_INTERNAL;
}

}  // namespace cere::research
