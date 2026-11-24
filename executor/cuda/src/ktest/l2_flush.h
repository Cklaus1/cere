// Cold-L2 timing for the kernel tests (libcere_ktest.so only; m23 design note P14): a replay timed after
// l2_flush finds none of its operands in L2, as in the model, where a layer's expert weights were last read a full step
// (gigabytes of other weights) earlier. The single-layer replays of a warm graph hold those weights in the 128 MB L2
// and overstate a weight-streaming kernel (the P14 decode GEMMs: L2-hot replays 2x faster than FlashInfer's grouped
// GEMMs, in the model 10 % slower).
//
// A read-only flush does not evict: the RTX PRO 6000's L2 keeps re-read lines over read-once ones (GPU1 2026-10-16,
// an 18 MB set read 5x then re-read: 6.1-6.5 us after no flush, after reading 512 MB or 1 GB once, or 512 MB twice;
// 17.0-17.5 us after writing 512 MB; 15.5-15.8 us for a set not read in the last 864 MB of reads). Writes do: l2_flush
// fills `bytes` of its buffer, then discards those lines (discard.global.L2: dropped without a write-back, so no
// write-back of the fill competes with the timed launch; 16.3-16.4 us).
#pragma once

#include "cere_executor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace cere::ktest {

// Writes `bytes` (a multiple of 128) of the scratch `buf` on `stream`, evicting what L2 held, then discards those
// lines (their content is undefined afterwards). The launch status.
cudaError_t l2_flush(void *buf, size_t bytes, cudaStream_t stream);

// The flush size the cold timings use: 4x the 128 MB L2 of the RTX PRO 6000 Blackwell.
constexpr size_t kL2FlushBytes = size_t{512} << 20;

// `timed` CUDA-graph replays of `run` (one call captured on its own stream after `warmup` replays): the span of each
// replay in microseconds into `us`. The caller's `stream` work is ordered before the capture stream's. `cold`: each
// timed replay follows an l2_flush (outside its span), so the replay reads its operands from DRAM, as in the model.
cere_status graph_spans(const std::function<cere_status(cudaStream_t)> &run, int64_t warmup, int64_t timed,
                        bool cold, float *us, cudaStream_t stream);

// Cold-L2 timing of `call` (a callable returning a cere_status, enqueuing its work on `stream`): one untimed call,
// then `iters` rounds of l2_flush(`flush`, kL2FlushBytes) and the call between two events on `stream`. us[0], us[1]
// (host doubles) = the median and the minimum µs of one call. `flush`: a device buffer of kL2FlushBytes. The first
// failing status (CERE_ERR_INTERNAL for a CUDA failure), else CERE_OK.
template <typename Call>
cere_status cold_us(void *flush, int64_t iters, double *us, cudaStream_t stream, Call call) {
    if (flush == nullptr || iters < 1 || us == nullptr) return CERE_ERR_ARGUMENT;
    cudaEvent_t start = nullptr, stop = nullptr;
    cere_status s = call();
    if (s == CERE_OK && (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)) {
        s = CERE_ERR_INTERNAL;
    }
    std::vector<double> spans;
    for (int64_t i = 0; i < iters && s == CERE_OK; ++i) {
        float ms = 0.0f;
        if (l2_flush(flush, kL2FlushBytes, stream) != cudaSuccess || cudaEventRecord(start, stream) != cudaSuccess) {
            s = CERE_ERR_INTERNAL;
            break;
        }
        s = call();
        if (s == CERE_OK &&
            (cudaEventRecord(stop, stream) != cudaSuccess || cudaEventSynchronize(stop) != cudaSuccess ||
             cudaEventElapsedTime(&ms, start, stop) != cudaSuccess)) {
            s = CERE_ERR_INTERNAL;
        }
        spans.push_back(1000.0 * double(ms));
    }
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    if (s == CERE_OK) {
        std::sort(spans.begin(), spans.end());
        us[0] = spans[spans.size() / 2];
        us[1] = spans[0];
    }
    return s;
}

}  // namespace cere::ktest
