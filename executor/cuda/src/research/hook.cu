// WeightLab research hook of the CUDA forward (research/hook.h).
#include "research/hook.h"

#include <algorithm>
#include <cmath>
#include <new>

#define CERE_TRY(expr)                 \
    do {                               \
        const cere_status s_ = (expr); \
        if (s_ != CERE_OK) return s_;  \
    } while (0)

namespace cere::research {

namespace {

constexpr uint64_t kStagingAlign = 16;

cere_status from_cuda(cudaError_t e) { return e == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL; }

// The block of capture op `op` at `layer` and `part` (blocks in op, layer ascending, part order).
const Block &block_of(const Binding &b, uint32_t op, uint32_t layer, uint32_t part) {
    const Op &o = b.plan->ops[op];
    const uint64_t below = layer == 0 ? 0 : o.layers & ((uint64_t(1) << layer) - 1);
    const uint32_t index = uint32_t(__builtin_popcountll(below)) * o.point->parts + part;
    return b.blocks[b.first_block[op] + index];
}

bool at_layer(const Op &op, PointId point, uint32_t layer) {
    return op.point->id == point && ((op.layers >> layer) & 1u) != 0;
}

// The device form of intervention op `op` of a plan loaded at `plan` (its slot's plan memory).
Edit edit_of(const Op &op, const uint8_t *plan) {
    Edit e;
    e.kind = op.edit_kind();
    e.alpha = op.alpha;
    e.col_begin = op.col_begin;
    e.col_end = op.col_end;
    if (!op.vector.empty()) e.vector = reinterpret_cast<const __nv_bfloat16 *>(plan + op.vector_offset);
    e.vector_rows = op.vector_rows;
    if (!op.experts.empty()) e.experts = reinterpret_cast<const int32_t *>(plan + op.experts_offset);
    if (!op.weights.empty()) e.weights = reinterpret_cast<const float *>(plan + op.weights_offset);
    e.count = uint32_t(op.experts.size());
    return e;
}

// Order key of a BF16 logit: larger value first, then lower id; NaN below everything, -0 equal to +0.
__device__ __forceinline__ uint64_t logit_key(uint16_t u, uint32_t id) {
    uint32_t ord;
    if ((u & 0x7F80u) == 0x7F80u && (u & 0x007Fu) != 0) {
        ord = 0;  // NaN
    } else {
        if (u == 0x8000u) u = 0;
        ord = (u & 0x8000u) ? (~uint32_t(u) & 0xFFFFu) : (uint32_t(u) | 0x8000u);
    }
    return (uint64_t(ord) << 32) | uint64_t(~id);
}

__device__ __forceinline__ float bits_float(uint16_t u) {
    return __uint_as_float(uint32_t(u) << 16);
}

constexpr unsigned kTopkThreads = 1024;

// One block: log-sum-exp in FP32 (per-thread strided partials, then a fixed tree), then k passes each taking the
// largest key strictly below the previous pass's.
__global__ void __launch_bounds__(kTopkThreads)
    topk_kernel(const uint16_t *__restrict__ logits, uint32_t vocab, uint32_t k, int32_t *__restrict__ ids,
                float *__restrict__ logprobs) {
    __shared__ float fred[kTopkThreads / 32];
    __shared__ uint64_t kred[kTopkThreads / 32];
    __shared__ float s_max, s_lse;
    __shared__ uint64_t s_key;
    const unsigned t = threadIdx.x, lane = t & 31u, warp = t >> 5;
    // Max over the non-NaN logits.
    float m = -INFINITY;
    for (uint32_t i = t; i < vocab; i += kTopkThreads) {
        const float x = bits_float(logits[i]);
        if (x == x) m = fmaxf(m, x);
    }
    for (unsigned o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
    if (lane == 0) fred[warp] = m;
    __syncthreads();
    if (t == 0) {
        float v = fred[0];
        for (unsigned w = 1; w < kTopkThreads / 32; ++w) v = fmaxf(v, fred[w]);
        s_max = v;
    }
    __syncthreads();
    const float mx = s_max;
    float sum = 0.0f;
    if (mx != -INFINITY) {
        for (uint32_t i = t; i < vocab; i += kTopkThreads) {
            const float x = bits_float(logits[i]);
            if (x == x) sum = __fadd_rn(sum, expf(__fsub_rn(x, mx)));
        }
    }
    for (unsigned o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xFFFFFFFFu, sum, o));
    __syncthreads();
    if (lane == 0) fred[warp] = sum;
    __syncthreads();
    if (t == 0) {
        float v = 0.0f;
        for (unsigned w = 0; w < kTopkThreads / 32; ++w) v = __fadd_rn(v, fred[w]);
        s_lse = mx == -INFINITY ? -INFINITY : __fadd_rn(mx, logf(v));
        s_key = ~uint64_t(0);
    }
    __syncthreads();
    const float lse = s_lse;
    for (uint32_t pass = 0; pass < k; ++pass) {
        const uint64_t below = s_key;
        uint64_t best = 0;
        bool found = false;
        for (uint32_t i = t; i < vocab; i += kTopkThreads) {
            const uint64_t key = logit_key(logits[i], i);
            if (key < below && (!found || key > best)) {
                best = key;
                found = true;
            }
        }
        for (unsigned o = 16; o > 0; o >>= 1) {
            const uint64_t other = __shfl_xor_sync(0xFFFFFFFFu, best, o);
            best = other > best ? other : best;
        }
        __syncthreads();  // every thread read s_key
        if (lane == 0) kred[warp] = best;
        __syncthreads();
        if (t == 0) {
            uint64_t v = kred[0];
            for (unsigned w = 1; w < kTopkThreads / 32; ++w) v = kred[w] > v ? kred[w] : v;
            const uint32_t id = ~uint32_t(v);
            ids[pass] = int32_t(id);
            logprobs[pass] = __fsub_rn(bits_float(logits[id]), lse);
            s_key = v;
        }
        __syncthreads();
    }
}

}  // namespace

// ---- Staging -----------------------------------------------------------------------------------------------------

Staging::~Staging() {
    if (device_ != nullptr) cudaFree(device_);
    if (ring_ != nullptr) cudaFreeHost(ring_);
}

cere_status Staging::allocate(uint64_t staging_bytes, uint64_t plan_bytes, uint64_t host_bytes) {
    if (device_ != nullptr || staging_bytes == 0 || host_bytes == 0) return CERE_ERR_ARGUMENT;
    uint8_t *device = nullptr;
    if (cudaMalloc(&device, staging_bytes + plan_bytes) != cudaSuccess) {
        cudaGetLastError();
        return CERE_ERR_CAPACITY;
    }
    uint8_t *ring = nullptr;
    if (cudaHostAlloc(&ring, host_bytes, cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        cudaFree(device);
        return CERE_ERR_CAPACITY;
    }
    try {
        records_.reserve(256);
    } catch (const std::bad_alloc &) {
        cudaFree(device);
        cudaFreeHost(ring);
        return CERE_ERR_CAPACITY;
    }
    device_ = device;
    staging_bytes_ = staging_bytes;
    plan_bytes_ = plan_bytes;
    ring_ = ring;
    return CERE_OK;
}

cere_status Staging::take(uint64_t bytes, uint8_t *host, cudaStream_t stream, uint8_t **out) {
    if (bytes == 0 || bytes > staging_bytes_) return CERE_ERR_INTERNAL;
    uint64_t at = (used_ + kStagingAlign - 1) / kStagingAlign * kStagingAlign;
    if (at > staging_bytes_ || bytes > staging_bytes_ - at) {
        if (const cere_status st = flush(stream); st != CERE_OK) return st;
        at = 0;
    }
    records_.push_back(Record{at, host, bytes});
    used_ = at + bytes;
    *out = device_ + at;
    return CERE_OK;
}

cere_status Staging::room(uint64_t bytes, uint32_t count, cudaStream_t stream) {
    const uint64_t padded = (bytes + kStagingAlign - 1) / kStagingAlign * kStagingAlign;
    const uint64_t at = (used_ + kStagingAlign - 1) / kStagingAlign * kStagingAlign;
    if (padded * count > staging_bytes_) return CERE_ERR_INTERNAL;
    if (at <= staging_bytes_ && padded * count <= staging_bytes_ - at) return CERE_OK;
    return flush(stream);
}

cere_status Staging::flush(cudaStream_t stream) {
    // Ranges contiguous in both the staging and the ring go as one copy.
    size_t i = 0;
    while (i < records_.size()) {
        Record r = records_[i++];
        while (i < records_.size() && records_[i].offset == r.offset + r.bytes && records_[i].host == r.host + r.bytes) {
            r.bytes += records_[i++].bytes;
        }
        if (const cudaError_t e = cudaMemcpyAsync(r.host, device_ + r.offset, r.bytes, cudaMemcpyDeviceToHost, stream);
            e != cudaSuccess) {
            records_.clear();
            used_ = 0;
            return CERE_ERR_INTERNAL;
        }
    }
    records_.clear();
    used_ = 0;
    return CERE_OK;
}

// ---- Call --------------------------------------------------------------------------------------------------------

cere_status Call::add(const Binding *binding, uint32_t row, uint32_t count, uint64_t position) {
    if (binding == nullptr || count == 0) return CERE_OK;
    if (staging_ == nullptr || segments_ == kMaxSegments) return CERE_ERR_ARGUMENT;
    segment_[segments_++] = Segment{binding, row, count, position};
    return CERE_OK;
}

bool Call::acts(PointId point, uint32_t layer) const {
    for (uint32_t i = 0; i < segments_; ++i) {
        if (segment_[i].binding->plan->acts(point, layer)) return true;
    }
    return false;
}

bool Call::edits(PointId point, uint32_t layer) const {
    for (uint32_t i = 0; i < segments_; ++i) {
        const Plan &plan = *segment_[i].binding->plan;
        if (!plan.edits || !plan.acts(point, layer)) continue;
        for (const Op &op : plan.ops) {
            if (!op.captures() && at_layer(op, point, layer)) return true;
        }
    }
    return false;
}

template <typename F>
void Call::runs(const Segment &s, uint32_t op, uint32_t row0, uint32_t n, F &&f) {
    // The segment's rows inside [row0, row0 + n) and their positions.
    const uint32_t lo_row = std::max(s.row, row0);
    const uint32_t hi_row = std::min(s.row + s.count, row0 + n);
    if (lo_row >= hi_row) return;
    const uint64_t p_lo = s.position + (lo_row - s.row);
    const uint64_t p_hi = s.position + (hi_row - s.row);
    const std::vector<Interval> &sel = s.binding->selection[op];
    for (size_t i = 0; i < sel.size(); ++i) {
        const uint64_t lo = std::max(sel[i].first, p_lo);
        const uint64_t hi = std::min(sel[i].end, p_hi);
        if (lo >= hi) continue;
        const uint32_t offset = uint32_t(lo_row - row0 + (lo - p_lo));
        f(offset, hi - lo, s.binding->base[op][i] + (lo - sel[i].first));
    }
}

cere_status Call::capture(const Binding &b, uint32_t op, uint32_t layer, uint32_t part, uint64_t rank, uint64_t rows,
                          const uint8_t *src, cudaStream_t stream) {
    const Block &block = block_of(b, op, layer, part);
    const uint64_t row_bytes = block.row_bytes;
    if (row_bytes == 0 || row_bytes > staging_->capacity()) return CERE_ERR_INTERNAL;
    // As many rows per staged range as the staging holds (the widest row, one logits row, always fits).
    const uint64_t per_take = staging_->capacity() / row_bytes;
    while (rows != 0) {
        const uint64_t chunk = std::min(rows, per_take);
        uint8_t *dst = nullptr;
        CERE_TRY(staging_->take(chunk * row_bytes, b.row(block, rank), stream, &dst));
        if (cudaMemcpyAsync(dst, src, chunk * row_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
            return CERE_ERR_INTERNAL;
        }
        src += chunk * row_bytes;
        rank += chunk;
        rows -= chunk;
    }
    return CERE_OK;
}

cere_status Call::at(PointId point, uint32_t layer, uint32_t row0, uint32_t n, __nv_bfloat16 *values, uint32_t cols,
                     cudaStream_t stream, bool *edited) {
    for (uint32_t si = 0; si < segments_; ++si) {
        const Segment &s = segment_[si];
        const Binding &b = *s.binding;
        const Plan &plan = *b.plan;
        if (!plan.acts(point, layer)) continue;
        // Edits first (in op order), then the captures read the edited rows.
        for (uint32_t oi = 0; oi < plan.ops.size(); ++oi) {
            const Op &op = plan.ops[oi];
            if (op.captures() || !at_layer(op, point, layer)) continue;
            const Edit edit = edit_of(op, staging_->plan(b.slot));
            const bool per_row = op.kind == CERE_RESEARCH_REPLACE && op.vector_rows > 1;
            RowSet set;
            cere_status st = CERE_OK;
            const auto apply = [&]() {
                if (set.count == 0 || st != CERE_OK) return;
                st = apply_point(edit, values, n, cols, set, stream);
                if (edited != nullptr) *edited = true;
                set.count = 0;
            };
            runs(s, oi, row0, n, [&](uint32_t offset, uint64_t rows, uint64_t rank) {
                if (set.count == kMaxRanges) apply();
                set.range[set.count++] = RowRange{offset, uint32_t(offset + rows), per_row ? uint32_t(rank) : 0u};
            });
            apply();
            if (st != CERE_OK) return st;
        }
        if (b.data == nullptr) continue;
        for (uint32_t oi = 0; oi < plan.ops.size(); ++oi) {
            const Op &op = plan.ops[oi];
            if (!op.captures() || !at_layer(op, point, layer)) continue;
            if (block_of(b, oi, layer, 0).row_bytes != uint64_t(cols) * sizeof(__nv_bfloat16)) return CERE_ERR_INTERNAL;
            cere_status st = CERE_OK;
            runs(s, oi, row0, n, [&](uint32_t offset, uint64_t rows, uint64_t rank) {
                if (st != CERE_OK) return;
                const auto *src = reinterpret_cast<const uint8_t *>(values + uint64_t(offset) * cols);
                st = capture(b, oi, layer, 0, rank, rows, src, stream);
            });
            if (st != CERE_OK) return st;
        }
    }
    return CERE_OK;
}

cere_status Call::route(uint32_t layer, uint32_t row0, uint32_t n, int32_t *ids, float *weights, uint32_t top_k,
                        cudaStream_t stream) {
    for (uint32_t si = 0; si < segments_; ++si) {
        const Segment &s = segment_[si];
        const Binding &b = *s.binding;
        const Plan &plan = *b.plan;
        if (!plan.acts(PointId::kRoute, layer)) continue;
        for (uint32_t oi = 0; oi < plan.ops.size(); ++oi) {
            const Op &op = plan.ops[oi];
            if (op.captures() || !at_layer(op, PointId::kRoute, layer)) continue;
            const Edit edit = edit_of(op, staging_->plan(b.slot));
            RowSet set;
            cere_status st = CERE_OK;
            const auto apply = [&]() {
                if (set.count == 0 || st != CERE_OK) return;
                st = apply_route(edit, ids, weights, n, top_k, set, stream);
                set.count = 0;
            };
            runs(s, oi, row0, n, [&](uint32_t offset, uint64_t rows, uint64_t) {
                if (set.count == kMaxRanges) apply();
                set.range[set.count++] = RowRange{offset, uint32_t(offset + rows), 0u};
            });
            apply();
            if (st != CERE_OK) return st;
        }
        if (b.data == nullptr) continue;
        for (uint32_t oi = 0; oi < plan.ops.size(); ++oi) {
            const Op &op = plan.ops[oi];
            if (!op.captures() || !at_layer(op, PointId::kRoute, layer)) continue;
            const uint64_t row_bytes = uint64_t(top_k) * 4;
            if (block_of(b, oi, layer, 0).row_bytes != row_bytes || block_of(b, oi, layer, 1).row_bytes != row_bytes) {
                return CERE_ERR_INTERNAL;
            }
            cere_status st = CERE_OK;
            runs(s, oi, row0, n, [&](uint32_t offset, uint64_t rows, uint64_t rank) {
                if (st != CERE_OK) return;
                st = capture(b, oi, layer, 0, rank, rows,
                             reinterpret_cast<const uint8_t *>(ids + uint64_t(offset) * top_k), stream);
                if (st != CERE_OK) return;
                st = capture(b, oi, layer, 1, rank, rows,
                             reinterpret_cast<const uint8_t *>(weights + uint64_t(offset) * top_k), stream);
            });
            if (st != CERE_OK) return st;
        }
    }
    return CERE_OK;
}

cere_status Call::logits(uint32_t row, const __nv_bfloat16 *logits, cudaStream_t stream) {
    for (uint32_t si = 0; si < segments_; ++si) {
        const Segment &s = segment_[si];
        const Binding &b = *s.binding;
        if (b.data == nullptr || row < s.row || row >= s.row + s.count) continue;
        const Plan &plan = *b.plan;
        if (!plan.acts(PointId::kLogits, kLayers) && !plan.acts(PointId::kLogitsTopk, kLayers)) continue;
        for (uint32_t oi = 0; oi < plan.ops.size(); ++oi) {
            const Op &op = plan.ops[oi];
            if (!op.captures()) continue;
            const bool full = at_layer(op, PointId::kLogits, kLayers);
            if (!full && !at_layer(op, PointId::kLogitsTopk, kLayers)) continue;
            uint64_t rank = 0;
            if (!b.rank(oi, s.position + (row - s.row), &rank)) continue;
            if (full) {
                CERE_TRY(capture(b, oi, kLayers, 0, rank, 1, reinterpret_cast<const uint8_t *>(logits), stream));
                continue;
            }
            const Block &id_block = block_of(b, oi, kLayers, 0);
            const Block &lp_block = block_of(b, oi, kLayers, 1);
            uint8_t *ids = nullptr, *lps = nullptr;
            CERE_TRY(staging_->room(std::max(id_block.row_bytes, lp_block.row_bytes), 2, stream));
            CERE_TRY(staging_->take(id_block.row_bytes, b.row(id_block, rank), stream, &ids));
            CERE_TRY(staging_->take(lp_block.row_bytes, b.row(lp_block, rank), stream, &lps));
            CERE_TRY(logits_topk(logits, kVocab, op.top_k, reinterpret_cast<int32_t *>(ids),
                                 reinterpret_cast<float *>(lps), stream));
        }
    }
    return CERE_OK;
}

cere_status logits_topk(const __nv_bfloat16 *logits, uint32_t vocab, uint32_t k, int32_t *ids, float *logprobs,
                        cudaStream_t stream) {
    if (logits == nullptr || ids == nullptr || logprobs == nullptr || k == 0 || k > CERE_RESEARCH_MAX_TOPK ||
        vocab < k) {
        return CERE_ERR_ARGUMENT;
    }
    topk_kernel<<<1, kTopkThreads, 0, stream>>>(reinterpret_cast<const uint16_t *>(logits), vocab, k, ids, logprobs);
    return from_cuda(cudaGetLastError());
}

}  // namespace cere::research
