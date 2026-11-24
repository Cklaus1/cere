// Cold-L2 timing (see l2_flush.h).
#include "ktest/l2_flush.h"

#include "cere_executor.h"

#include <cstdint>

namespace cere::ktest {
namespace {

constexpr size_t kLine = 128;  // L2 line bytes (discard.global.L2's size operand)

__global__ void l2_fill_kernel(uint4 *buf, size_t n, uint32_t v) {
    for (size_t i = size_t{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += size_t{gridDim.x} * blockDim.x) {
        buf[i] = make_uint4(v, v, v, v);
    }
}

__global__ void l2_discard_kernel(uint8_t *buf, size_t bytes) {
    for (size_t i = (size_t{blockIdx.x} * blockDim.x + threadIdx.x) * kLine; i < bytes;
         i += size_t{gridDim.x} * blockDim.x * kLine) {
        asm volatile("discard.global.L2 [%0], 128;" ::"l"(buf + i) : "memory");
    }
}

}  // namespace

cudaError_t l2_flush(void *buf, size_t bytes, cudaStream_t stream) {
    if (buf == nullptr || bytes == 0 || bytes % kLine != 0) return cudaErrorInvalidValue;
    int sms = 0;
    cudaError_t e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    if (e != cudaSuccess) return e;
    const unsigned grid = static_cast<unsigned>(sms * 4);
    l2_fill_kernel<<<grid, 512, 0, stream>>>(static_cast<uint4 *>(buf), bytes / sizeof(uint4), 0x5a5a5a5au);
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
    l2_discard_kernel<<<grid, 512, 0, stream>>>(static_cast<uint8_t *>(buf), bytes);
    return cudaGetLastError();
}

cere_status graph_spans(const std::function<cere_status(cudaStream_t)> &run, int64_t warmup, int64_t timed,
                        bool cold, float *us, cudaStream_t stream) {
    void *flush = nullptr;
    cudaStream_t own = nullptr;
    cudaEvent_t ready = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cere_status s = CERE_OK;
    const auto ck = [&s](cudaError_t e) {
        if (s == CERE_OK && e != cudaSuccess) s = CERE_ERR_INTERNAL;
        return s == CERE_OK;
    };
    if (ck(cudaStreamCreateWithFlags(&own, cudaStreamNonBlocking)) && ck(cudaEventCreate(&ready)) &&
        ck(cudaEventCreate(&start)) && ck(cudaEventCreate(&end)) &&
        (!cold || ck(cudaMalloc(&flush, kL2FlushBytes))) && ck(cudaEventRecord(ready, stream)) &&
        ck(cudaStreamWaitEvent(own, ready, 0)) && ck(cudaStreamSynchronize(own)) &&
        ck(cudaStreamBeginCapture(own, cudaStreamCaptureModeThreadLocal))) {
        const cere_status captured = run(own);
        const cudaError_t ended = cudaStreamEndCapture(own, &graph);
        if (captured != CERE_OK) s = captured;
        if (ck(ended) && ck(cudaGraphInstantiate(&exec, graph, 0))) {
            for (int64_t i = 0; i < warmup && ck(cudaGraphLaunch(exec, own)); ++i) {
            }
            for (int64_t i = 0; i < timed && s == CERE_OK; ++i) {
                float ms = 0.0f;
                if ((!cold || ck(l2_flush(flush, kL2FlushBytes, own))) && ck(cudaEventRecord(start, own)) &&
                    ck(cudaGraphLaunch(exec, own)) && ck(cudaEventRecord(end, own)) &&
                    ck(cudaEventSynchronize(end)) && ck(cudaEventElapsedTime(&ms, start, end))) {
                    us[i] = ms * 1000.0f;
                }
            }
            ck(cudaStreamSynchronize(own));
        }
    }
    if (exec != nullptr) cudaGraphExecDestroy(exec);
    if (graph != nullptr) cudaGraphDestroy(graph);
    if (end != nullptr) cudaEventDestroy(end);
    if (start != nullptr) cudaEventDestroy(start);
    if (ready != nullptr) cudaEventDestroy(ready);
    if (own != nullptr) cudaStreamDestroy(own);
    if (flush != nullptr) cudaFree(flush);
    return s;
}

}  // namespace cere::ktest

// Cold-L2 timing in Python (executor/tests/gpu/test_p3b_decode.py test_speed): one l2_flush of kL2FlushBytes on
// `stream` (a buffer of its own, stream-ordered). 0 = success, else a cere_status.
extern "C" __attribute__((visibility("default"))) int cere_ktest_l2_flush(cudaStream_t stream) {
    void *buf = nullptr;
    if (cudaMallocAsync(&buf, cere::ktest::kL2FlushBytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    const cudaError_t e = cere::ktest::l2_flush(buf, cere::ktest::kL2FlushBytes, stream);
    const cudaError_t f = cudaFreeAsync(buf, stream);
    return e == cudaSuccess && f == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}
