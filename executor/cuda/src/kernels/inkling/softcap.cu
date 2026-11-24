// Inkling 2 Mini (inkling design note 3.1, G1): the final logit softcap; contract in softcap.h, the order in
// executor/ref/inkling/softcap_ref.h. Compiled with -fmad=false.
#include "kernels/inkling/softcap.h"

#include <climits>

#include "kernels/inkling/pdl.cuh"
#include "inkling/softcap_ref.h"

namespace cere::inkling {

namespace {

constexpr int kThreads = 256;

__global__ void __launch_bounds__(kThreads) softcap_kernel(float *logits, int64_t vocab, int64_t ld) {
    pdl::enter();
    float *row = logits + int64_t(blockIdx.y) * ld;
    for (int64_t i = int64_t(blockIdx.x) * kThreads + threadIdx.x; i < vocab; i += int64_t(gridDim.x) * kThreads)
        row[i] = ref::softcap(row[i]);
}

}  // namespace

cudaError_t softcap(float *logits, int64_t rows, int64_t vocab, int64_t ld, cudaStream_t stream) {
    if (logits == nullptr || rows < 1 || rows > 65535 || vocab < 1 || vocab > INT_MAX || ld < vocab)
        return cudaErrorInvalidValue;
    const int64_t blocks = (vocab + kThreads - 1) / kThreads;
    const unsigned bx = unsigned(blocks < 1024 ? blocks : 1024);
    return pdl::launch(softcap_kernel, dim3(bx, unsigned(rows)), dim3(kThreads), 0, stream, logits, vocab, ld);
}

}  // namespace cere::inkling
