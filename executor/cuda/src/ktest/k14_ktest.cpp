// K14 kernel-test entry point (executor/tests/gpu/test_model_k14_mtp.py, m29 design note D3): the MTP draft layer
// (model/mtp.h) of a drafting A4 model on a ktest handle (k8_ktest.cpp's cere_ktest_model_create), after a target
// PREFILL that supplies the pre-final-mixer hc rows.
#include "cere_executor.h"
#include "device/cuda_device.h"
#include "device/memory_plan.h"
#include "host/state.h"
#include "model/mtp.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

namespace {

using cere::model::bf16;

// The ktest handle of k8_ktest.cpp (cere_ktest_model_create): this file reads the first two members only.
struct K8ModelView {
    cere_executor *executor = nullptr;
    cere_model *model = nullptr;
};

// detail_sink of the target PREFILL: copies "native.final_mix.in" (the pre-final-mixer hc rows [n, 10,240]) of the
// chunk at row `row0` into `hc`; every other detail name is ignored.
struct HcSink {
    bf16 *hc = nullptr;
    uint64_t row0 = 0;
    bool seen = false;

    static cere_status sink(void *ctx, const char *name, const bf16 *values, uint64_t rows, uint64_t cols,
                            cudaStream_t stream) {
        auto *s = static_cast<HcSink *>(ctx);
        if (std::strcmp(name, "native.final_mix.in") != 0) return CERE_OK;
        if (cols != cere::model::kHcHidden) return CERE_ERR_INTERNAL;
        if (cudaMemcpyAsync(s->hc + s->row0 * cols, values, rows * cols * sizeof(bf16), cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
            cudaGetLastError();
            return CERE_ERR_INTERNAL;
        }
        s->seen = true;
        return CERE_OK;
    }
};

// The MTP stage names of model/mtp.h StageSink, in the order of cere_ktest_mtp_forward's `stages` pointers.
constexpr const char *kStageNames[] = {"fusion.e",   "fusion.hyper",  "attn.normed", "attn.mixed",
                                       "attn.qkv",   "attn.out",      "attn.block_out", "attn.hyper",
                                       "mlp.normed", "mlp.mixed",     "mlp.block_out",  "mixer.y"};
constexpr size_t kStages = sizeof(kStageNames) / sizeof(kStageNames[0]);

// Stage sink of the MTP calls: copies stage i of the call at row `row0` into stages[i] (device BF16 [n, cols]).
struct StageSink {
    const uint64_t *stages = nullptr;
    uint64_t row0 = 0;

    static cere_status sink(void *ctx, const char *name, const bf16 *values, uint64_t rows, uint64_t cols,
                            cudaStream_t stream) {
        auto *s = static_cast<StageSink *>(ctx);
        for (size_t i = 0; i < kStages; ++i) {
            if (std::strcmp(name, kStageNames[i]) != 0) continue;
            auto *dst = reinterpret_cast<bf16 *>(static_cast<uintptr_t>(s->stages[i])) + s->row0 * cols;
            if (cudaMemcpyAsync(dst, values, rows * cols * sizeof(bf16), cudaMemcpyDeviceToDevice, stream) !=
                cudaSuccess) {
                cudaGetLastError();
                return CERE_ERR_INTERNAL;
            }
            return CERE_OK;
        }
        return CERE_ERR_INTERNAL;
    }
};

}  // namespace

// cere_ktest_mtp_forward: on a fresh session of the model's context limit, PREFILL `tokens[0, n)` in chunks of the
// workspace rows with the pre-final-mixer hc rows of every position copied to `hc` (device BF16 [n, 10,240]), then
// the MTP forward over positions 0 .. n - 1 in calls of `mtp_chunk` rows (in order; row p reads hc[p] and
// next_ids[p], the token at p + 1): `hc_out` (device BF16 [n, 10,240]) the MTP pre-mixer hc rows, `logits` (device
// BF16 [n, draft count]) the hot logits, `drafts` (device u32 [n]) map[argmax]. `stages` (host u64 [12], or null):
// device pointers of BF16 [n, cols] buffers that receive the MTP stages of kStageNames (model/mtp.h StageSink) of
// every row. CERE_ERR_ARGUMENT for a null pointer, n = 0, n beyond the context, mtp_chunk outside [1, workspace rows]
// or a model without drafts.
extern "C" __attribute__((visibility("default"))) int cere_ktest_mtp_forward(
    uint64_t model, const uint32_t *tokens /*host [n]*/, uint64_t n, const uint32_t *next_ids /*device [n]*/,
    uint64_t mtp_chunk, void *hc, void *hc_out, void *logits, uint32_t *drafts, const uint64_t *stages /*host [12]*/,
    cudaStream_t stream) {
    auto *k = reinterpret_cast<K8ModelView *>(static_cast<uintptr_t>(model));
    if (k == nullptr || k->executor == nullptr || k->model == nullptr || tokens == nullptr || n == 0 ||
        next_ids == nullptr || hc == nullptr || hc_out == nullptr || logits == nullptr || drafts == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    auto &device = static_cast<cere::device::CudaDevice &>(*k->executor->device);
    auto &cmodel = static_cast<cere::device::CudaModel &>(*k->model->device);
    const uint64_t ws_rows = cmodel.workspace_rows();
    if (mtp_chunk == 0 || mtp_chunk > ws_rows || n > k->model->plan.max_context_tokens) return CERE_ERR_ARGUMENT;
    const uint64_t count = k->model->plan.draft_token_map.size();
    if (count == 0) return CERE_ERR_ARGUMENT;
    // The caller's buffers may still be written by its stream.
    if (cudaStreamSynchronize(stream) != cudaSuccess) return CERE_ERR_INTERNAL;

    cere_session *session = nullptr;
    if (const cere_status st = cere_session_create(k->executor, k->model, k->model->plan.max_context_tokens, &session);
        st != CERE_OK) {
        return st;
    }
    auto &csession = static_cast<cere::device::CudaSession &>(*session->device);
    cere_status st = CERE_OK;
    HcSink sink;
    sink.hc = static_cast<bf16 *>(hc);
    std::vector<cere_token_score> scores(ws_rows);
    for (uint64_t p0 = 0; p0 < n && st == CERE_OK; p0 += ws_rows) {
        const uint64_t rows = std::min(ws_rows, n - p0);
        cere::model::ForwardTaps taps;
        taps.ctx = &sink;
        taps.detail = 1;
        taps.detail_sink = &HcSink::sink;
        sink.row0 = p0;
        sink.seen = false;
        st = device.forward(cmodel, csession, CERE_OP_PREFILL, tokens + p0, uint32_t(rows), &taps, scores.data());
        if (st == CERE_OK && !sink.seen) st = CERE_ERR_INTERNAL;
    }
    std::vector<uint64_t> positions(mtp_chunk);
    StageSink stage_sink;
    stage_sink.stages = stages;
    for (uint64_t p0 = 0; p0 < n && st == CERE_OK; p0 += mtp_chunk) {
        const uint32_t m = uint32_t(std::min(mtp_chunk, n - p0));
        for (uint32_t r = 0; r < m; ++r) positions[r] = p0 + r;
        cere::model::mtp::MtpCall call;
        call.m = m;
        call.positions = positions.data();
        call.next_ids = next_ids + p0;
        call.hc_in = static_cast<const bf16 *>(hc) + p0 * cere::model::kHcHidden;
        call.hc_out = static_cast<bf16 *>(hc_out) + p0 * cere::model::kHcHidden;
        call.draft_ids = drafts + p0;
        if (stages != nullptr) {
            stage_sink.row0 = p0;
            call.stage_sink = &StageSink::sink;
            call.stage_ctx = &stage_sink;
        }
        st = device.mtp_forward(cmodel, csession, call, static_cast<bf16 *>(logits) + p0 * count);
    }
    cere_session_release(k->executor, session);
    return st;
}

// cere_ktest_mtp_plan_bytes (m29 design note D10', A1): device::plan_layout of the handle's plan with
// max_context_tokens, max_sessions and kv_pool_tokens replaced, either as is (`draft` 1: the plan's draft map and
// `mtp.*` placements; the handle must have drafts) or as the spec-off plan (`draft` 0: speculative_draft_tokens 0, no
// map, every `mtp.*` tensor DISCARD). Host arithmetic, no CUDA call. out (host u64 [12]): weight_bytes,
// workspace_rows, workspace_bytes, derived_bytes, session_bytes, pool_bytes, device_bytes, pinned_bytes, draft_count,
// then model::mtp::derived_bytes, hc_rows_bytes and workspace_bytes (the growth) at the plan's rows and draft count
// (0 when draft_count is 0). CERE_ERR_ARGUMENT for a null pointer, `draft` > 1, a zero size or `draft` 1 on a handle
// without drafts; CERE_ERR_CAPACITY when the layout overflows.
extern "C" __attribute__((visibility("default"))) int cere_ktest_mtp_plan_bytes(
    uint64_t model, uint64_t max_context_tokens, uint64_t max_sessions, uint64_t kv_pool_tokens, uint64_t draft,
    uint64_t *out /*host [12]*/, cudaStream_t) {
    auto *k = reinterpret_cast<K8ModelView *>(static_cast<uintptr_t>(model));
    if (k == nullptr || k->model == nullptr || out == nullptr || draft > 1 || max_context_tokens == 0 ||
        max_sessions == 0 || kv_pool_tokens == 0) {
        return CERE_ERR_ARGUMENT;
    }
    cere::host::ModelPlan plan = k->model->plan;
    if (draft == 1 && (plan.speculative_draft_tokens == 0 || plan.draft_token_map.empty())) return CERE_ERR_ARGUMENT;
    plan.max_context_tokens = max_context_tokens;
    plan.max_sessions = max_sessions;
    plan.kv_pool_tokens = kv_pool_tokens;
    if (draft == 0) {
        plan.speculative_draft_tokens = 0;
        plan.draft_token_map.clear();
        for (cere::host::TensorPlan &t : plan.tensors) {
            if (t.name.rfind("mtp.", 0) == 0) t.placement = CERE_PLACEMENT_DISCARD;
        }
    }
    cere::device::ModelLayout l{};
    try {
        l = cere::device::plan_layout(plan);
    } catch (const std::bad_alloc &) {
        return CERE_ERR_CAPACITY;
    }
    if (l.overflow) return CERE_ERR_CAPACITY;
    const uint64_t count = l.draft_count;
    const uint64_t values[12] = {
        l.weight_bytes, l.workspace_rows, l.workspace_bytes, l.derived_bytes, l.session_bytes, l.pool_bytes,
        l.device_bytes, l.pinned_bytes, count,
        count == 0 ? 0 : cere::model::mtp::derived_bytes(plan.config, count),
        count == 0 ? 0 : cere::model::mtp::hc_rows_bytes(plan.config, l.workspace_rows),
        count == 0 ? 0 : cere::model::mtp::workspace_bytes(plan.config, l.workspace_rows, max_context_tokens, count)};
    std::memcpy(out, values, sizeof(values));
    return CERE_OK;
}
