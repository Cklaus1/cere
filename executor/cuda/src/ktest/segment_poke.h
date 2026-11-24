// O47 test-only device write into the shared HOST_PINNED region (libcere_ktest.so only; k8_ktest.cpp
// cere_ktest_segment_poke, executor/tests/gpu/test_model_host_share.py).
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace cere::ktest {

// One GPU thread stores `value` at `device_ptr` (a device alias of mapped host memory) on `stream`; the caller
// synchronizes. The launch status.
cudaError_t poke_byte(uint8_t *device_ptr, uint8_t value, cudaStream_t stream);

}  // namespace cere::ktest
