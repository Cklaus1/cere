// D12 profile marker (m23 design note D12; libcere_ktest.so only; k8_ktest.cpp cere_ktest_forward_profile,
// executor/tests/gpu/test_profile_forward.py).
#pragma once

#include <cuda_runtime.h>

namespace cere::ktest {

// One single-thread kernel (`profile_mark_kernel`, it touches no memory) on `stream`, so a CUPTI kernel trace of a
// profiled call splits at it; the caller synchronizes. The launch status.
cudaError_t profile_mark(cudaStream_t stream);

}  // namespace cere::ktest
