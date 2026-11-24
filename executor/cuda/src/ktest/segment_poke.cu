// O47 test-only device write into the shared HOST_PINNED region (see segment_poke.h).
#include "ktest/segment_poke.h"

namespace cere::ktest {
namespace {

__global__ void poke_kernel(volatile uint8_t *p, uint8_t value) { *p = value; }

}  // namespace

cudaError_t poke_byte(uint8_t *device_ptr, uint8_t value, cudaStream_t stream) {
    poke_kernel<<<1, 1, 0, stream>>>(device_ptr, value);
    return cudaGetLastError();
}

}  // namespace cere::ktest
