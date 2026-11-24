// Per-row BF16 digests (see row_digest.h).
#include "ktest/row_digest.h"

namespace cere::ktest {
namespace {

constexpr unsigned kThreads = 256;

__device__ __forceinline__ uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// One CTA per row.
__global__ void __launch_bounds__(kThreads) row_digest_kernel(const uint16_t *__restrict__ values, uint64_t cols,
                                                              uint64_t *__restrict__ out) {
    const uint16_t *row = values + uint64_t(blockIdx.x) * cols;
    uint64_t acc = 0;
    for (uint64_t c = threadIdx.x; c < cols; c += kThreads) acc += splitmix64(c << 16 | row[c]);
    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    __shared__ uint64_t warps[kThreads / 32];
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t sum = 0;
        for (unsigned w = 0; w < kThreads / 32; ++w) sum += warps[w];
        out[blockIdx.x] = splitmix64(sum ^ cols);
    }
}

}  // namespace

cudaError_t row_digests(const uint16_t *values, uint64_t rows, uint64_t cols, uint64_t *out, cudaStream_t stream) {
    if (rows == 0) return cudaSuccess;
    if (values == nullptr || out == nullptr || cols == 0 || rows > 0x7FFFFFFFull) return cudaErrorInvalidValue;
    row_digest_kernel<<<unsigned(rows), kThreads, 0, stream>>>(values, cols, out);
    return cudaGetLastError();
}

}  // namespace cere::ktest
