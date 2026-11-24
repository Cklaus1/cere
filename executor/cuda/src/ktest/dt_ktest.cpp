// DraftTrain kernel-test entry point (scripts/draft/run.py, drafttrain design note): greedy generation on a
// drafting ktest handle (k8_ktest.cpp's cere_ktest_model_create with a draft token map) that keeps the target's
// pre-final-mixer hc rows (the MTP layer's input, model/mtp.h) of the last prompt positions and of every decoded
// position. The rows come from Workspace::hc_rows, which model::forward fills on a drafting model after every call
// (forward.cpp, SGLang's hc_hidden_states); no tap runs, so the forward is the serving path's own.
#include "cere_executor.h"
#include "device/cuda_device.h"
#include "host/state.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

namespace {

using cere::model::bf16;

// The ktest handle of k8_ktest.cpp (cere_ktest_model_create): this file reads the first two members only.
struct K8ModelView {
    cere_executor *executor = nullptr;
    cere_model *model = nullptr;
};

// Copies rows [from, from + rows) of the workspace's hc rows to dst (device BF16 [rows, 10,240]); the forward that
// wrote them has returned, after its host copy of the scores (its stream is drained up to the hc copy).
cere_status keep_rows(const cere::model::Workspace &ws, uint64_t from, uint64_t rows, bf16 *dst) {
    if (rows == 0) return CERE_OK;
    const uint64_t width = cere::model::kHcHidden;
    if (cudaMemcpy(dst, ws.hc_rows + from * width, rows * width * sizeof(bf16), cudaMemcpyDeviceToDevice) !=
        cudaSuccess) {
        cudaGetLastError();
        return CERE_ERR_INTERNAL;
    }
    return CERE_OK;
}

uint64_t micros_since(std::chrono::steady_clock::time_point t0) {
    return uint64_t(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
}

}  // namespace

// cere_ktest_dt_generate: on a fresh session of the model's context limit, PREFILL prompt[0, n) in calls of the
// workspace rows, then greedy DECODE from the last prompt row's top-1: token g[0] is that top-1, and each further
// g[i] the top-1 of the DECODE call that fed g[i - 1]. Generation ends after a token in stops[0, stop_count)
// (committed, never fed), after max_new tokens or when the next feed would exceed the context. Kept rows, in position
// order: the `window` prompt positions n - window .. n - 1 (window <= n), then the positions n .. n + fed - 1 of the
// fed tokens g[0, fed). hc (device BF16 [window + max_new, 10,240]): the kept positions' pre-final-mixer hc rows.
// out_ids (host u32 [max_new]): g. out (host u64 [6]): committed tokens, fed tokens, finish (1 stop id, 2 max_new,
// 3 context), prefill microseconds, decode microseconds, kept rows. CERE_ERR_ARGUMENT for a null pointer, n = 0,
// window > n, max_new = 0, n beyond the context or a model without drafts (no hc rows).
extern "C" __attribute__((visibility("default"))) int cere_ktest_dt_generate(
    uint64_t model, const uint32_t *prompt /*host [n]*/, uint64_t n, uint64_t window, uint64_t max_new,
    const uint32_t *stops /*host [stop_count]*/, uint64_t stop_count, void *hc, uint32_t *out_ids /*host [max_new]*/,
    uint64_t *out /*host [6]*/, cudaStream_t stream) {
    auto *k = reinterpret_cast<K8ModelView *>(static_cast<uintptr_t>(model));
    if (k == nullptr || k->executor == nullptr || k->model == nullptr || prompt == nullptr || n == 0 ||
        window > n || max_new == 0 || hc == nullptr || out_ids == nullptr || out == nullptr ||
        (stop_count != 0 && stops == nullptr)) {
        return CERE_ERR_ARGUMENT;
    }
    auto &device = static_cast<cere::device::CudaDevice &>(*k->executor->device);
    auto &cmodel = static_cast<cere::device::CudaModel &>(*k->model->device);
    const cere::model::Workspace &ws = cmodel.workspace();
    const uint64_t context = k->model->plan.max_context_tokens;
    if (n > context || ws.hc_rows == nullptr) return CERE_ERR_ARGUMENT;
    const uint64_t ws_rows = cmodel.workspace_rows();
    // The caller's buffers may still be written by its stream.
    if (cudaStreamSynchronize(stream) != cudaSuccess) return CERE_ERR_INTERNAL;

    cere_session *session = nullptr;
    if (const cere_status st = cere_session_create(k->executor, k->model, context, &session); st != CERE_OK) {
        return st;
    }
    auto &csession = static_cast<cere::device::CudaSession &>(*session->device);
    auto *rows_out = static_cast<bf16 *>(hc);
    const uint64_t width = cere::model::kHcHidden;
    const uint64_t keep_from = n - window;
    cere_status st = CERE_OK;
    cere_token_score score{};
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t p0 = 0; p0 < n && st == CERE_OK; p0 += ws_rows) {
        const uint64_t rows = std::min(ws_rows, n - p0);
        st = device.forward(cmodel, csession, CERE_OP_PREFILL, prompt + p0, uint32_t(rows), nullptr, &score);
        if (st != CERE_OK) break;
        // The kept prompt rows of this call: positions max(p0, keep_from) .. p0 + rows - 1.
        const uint64_t from = std::max(p0, keep_from);
        if (from < p0 + rows) st = keep_rows(ws, from - p0, p0 + rows - from, rows_out + (from - keep_from) * width);
    }
    const uint64_t prefill_us = micros_since(t0);
    uint64_t committed = 0;
    uint64_t fed = 0;
    uint64_t finish = 0;
    const auto t1 = std::chrono::steady_clock::now();
    uint32_t next = score.id;
    while (st == CERE_OK) {
        out_ids[committed++] = next;
        if (std::find(stops, stops + stop_count, next) != stops + stop_count) {
            finish = 1;
            break;
        }
        if (committed == max_new) {
            finish = 2;
            break;
        }
        if (n + fed + 1 > context) {
            finish = 3;
            break;
        }
        st = device.forward(cmodel, csession, CERE_OP_DECODE, &next, 1, nullptr, &score);
        if (st != CERE_OK) break;
        st = keep_rows(ws, 0, 1, rows_out + (window + fed) * width);
        ++fed;
        next = score.id;
    }
    const uint64_t decode_us = micros_since(t1);
    cere_session_release(k->executor, session);
    if (st != CERE_OK) return st;
    out[0] = committed;
    out[1] = fed;
    out[2] = finish;
    out[3] = prefill_us;
    out[4] = decode_us;
    out[5] = window + fed;
    return CERE_OK;
}
