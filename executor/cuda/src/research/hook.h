// WeightLab research on the CUDA forward (weightlab design note WL-02..04, "Capture", "Interventions"): the device
// side of a model's research reserve (Staging) and the research hook of one forward call (Call). The forward
// (model/forward.cpp) builds a Call only when some row of the call belongs to a session bound to a plan
// (SessionState::research) and passes null otherwise, so a call without a planned row runs today's code path; it calls
// the hook at each research point the plan names, with the point's native tensor [n, cols] of a run of the call's rows.
// At a point the hook first applies the plan's interventions to the rows they select (WL-B's apply_point /
// apply_route, research/patch.h), then copies the selected rows (D2D) into the device staging, and flush() copies the
// staging asynchronously into the capture's slots of the pinned host ring (research/capture.h). Everything is enqueued
// on the call's stream; nothing synchronizes (the forward drains its stream before it returns, so a capture is on the
// host once the call returns).
//
// Rows: a call is a list of segments, each a run of rows [row, row + count) of one session at absolute positions
// [position, position + count) (run_layers: one segment of n rows; batched decode: one per row; verify: one per chain,
// row j at the chain's p + j). A rejected verify row lands in its position's capture slot like any row; the position is
// computed again (or the session ends below it) before the capture is read, so only committed rows survive
// (research/capture.h finalise()).
#pragma once

#include "research/capture.h"
#include "research/patch.h"
#include "research/points.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace cere::research {

// The device memory of a model's research reserve (CudaDevice::research_reserve; freed with the model): one cudaMalloc
// of staging_bytes of capture staging followed by the plan memory (slot i at i x CERE_RESEARCH_MAX_PLAN_BYTES), and
// the pinned host ring (cudaHostAlloc) the host carves captures from.
class Staging {
public:
    Staging() = default;
    ~Staging();
    Staging(const Staging &) = delete;
    Staging &operator=(const Staging &) = delete;

    // Allocates both; CERE_ERR_CAPACITY (nothing kept) when either allocation fails.
    cere_status allocate(uint64_t staging_bytes, uint64_t plan_bytes, uint64_t host_bytes);

    uint8_t *ring() const { return ring_; }
    uint8_t *plan(uint32_t slot) const { return device_ + staging_bytes_ + uint64_t(slot) * CERE_RESEARCH_MAX_PLAN_BYTES; }
    uint64_t plan_bytes() const { return plan_bytes_; }
    uint64_t capacity() const { return staging_bytes_; }

    // `bytes` (<= staging_bytes) of staging for rows bound to `host`; flushes first when they do not fit.
    cere_status take(uint64_t bytes, uint8_t *host, cudaStream_t stream, uint8_t **out);
    // Flushes when `count` takes of `bytes` each would not all fit, so they land without a flush between them (a
    // kernel then writes into all of them).
    cere_status room(uint64_t bytes, uint32_t count, cudaStream_t stream);
    // Copies every taken range to its host destination (stream order) and empties the staging.
    cere_status flush(cudaStream_t stream);

private:
    struct Record {
        uint64_t offset;
        uint8_t *host;
        uint64_t bytes;
    };
    uint8_t *device_ = nullptr;
    uint64_t staging_bytes_ = 0;
    uint64_t plan_bytes_ = 0;
    uint8_t *ring_ = nullptr;
    uint64_t used_ = 0;
    std::vector<Record> records_;  // the taken ranges not yet flushed (capacity kept across calls)
};

// The research hook of one forward call (see the file comment). At most kMaxSegments segments.
class Call {
public:
    static constexpr uint32_t kMaxSegments = 16;  // kernels::GemmPlan::kMaxRows: a batched or verify call's rows

    explicit Call(Staging *staging) : staging_(staging) {}

    // A run of `count` rows from call row `row` of a session bound to `binding` (null: no plan, ignored) at positions
    // [position, position + count). CERE_ERR_ARGUMENT past kMaxSegments or without staging.
    cere_status add(const Binding *binding, uint32_t row, uint32_t count, uint64_t position);
    bool active() const { return segments_ != 0; }

    // Whether some segment's plan acts (captures or edits) / edits at `point` of `layer` (kLayers: the final points).
    bool acts(PointId point, uint32_t layer) const;
    bool edits(PointId point, uint32_t layer) const;

    // The point's native BF16 tensor `values` [n, cols] holding call rows [row0, row0 + n): the plan's edits in place,
    // then the captures. `*edited` (nullable) is set when an edit was launched on some row.
    cere_status at(PointId point, uint32_t layer, uint32_t row0, uint32_t n, __nv_bfloat16 *values, uint32_t cols,
                   cudaStream_t stream, bool *edited = nullptr);
    // native.layers.L.moe_route: the routing ids int32 / weights FP32 [n, top_k] of call rows [row0, row0 + n)
    // (kernels::MoeTaps.route): expert_force edits, then the captures.
    cere_status route(uint32_t layer, uint32_t row0, uint32_t n, int32_t *ids, float *weights, uint32_t top_k,
                      cudaStream_t stream);
    // The logits points of scored call row `row`: `logits` is its BF16 [vocab] row.
    cere_status logits(uint32_t row, const __nv_bfloat16 *logits, cudaStream_t stream);
    // At the end of the call: the staged rows to the host ring.
    cere_status flush(cudaStream_t stream) { return staging_->flush(stream); }

private:
    struct Segment {
        const Binding *binding;
        uint32_t row;
        uint32_t count;
        uint64_t position;
    };
    // Op `op` of segment `s`'s plan at layer `layer`: its selected positions among call rows [row0, row0 + n), each as
    // f(row offset in [0, n), rows, rank of the first) for every selected run.
    template <typename F>
    static void runs(const Segment &s, uint32_t op, uint32_t row0, uint32_t n, F &&f);
    // Copies `rows` rows of part `part` of capture op `op` at `layer` from `src` (row stride = the part's row bytes)
    // into the capture rows from rank `rank`.
    cere_status capture(const Binding &b, uint32_t op, uint32_t layer, uint32_t part, uint64_t rank, uint64_t rows,
                        const uint8_t *src, cudaStream_t stream);

    Staging *staging_;
    Segment segment_[kMaxSegments];
    uint32_t segments_ = 0;
};

// The top `k` (1..64) ids of a BF16 [vocab] logits row by logit (ties: lowest id; NaN below every number) and their
// FP32 log-softmax values (x - max - log(sum exp(x - max)), NaN logits left out of the sum) into ids / logprobs [k]
// (device). One block, enqueued on `stream`. CERE_ERR_ARGUMENT for k outside [1, 64] or vocab < k.
cere_status logits_topk(const __nv_bfloat16 *logits, uint32_t vocab, uint32_t k, int32_t *ids, float *logprobs,
                        cudaStream_t stream);

}  // namespace cere::research
