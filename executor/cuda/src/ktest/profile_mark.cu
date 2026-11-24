// D12 profile marker (see profile_mark.h).
#include "ktest/profile_mark.h"

namespace cere::ktest {
namespace {

__global__ void profile_mark_kernel() {}

}  // namespace

cudaError_t profile_mark(cudaStream_t stream) {
    profile_mark_kernel<<<1, 1, 0, stream>>>();
    return cudaGetLastError();
}

}  // namespace cere::ktest
