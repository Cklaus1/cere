// Shared helpers of the K1/K6 kernel-test entry points (libcere_ktest.so only).
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "cere_executor.h"

namespace cere::ktest {

// CERE_OK for cudaSuccess, CERE_ERR_ARGUMENT for cudaErrorInvalidValue (a kernel contract violation), else
// CERE_ERR_INTERNAL.
int status_of(cudaError_t err);

// 0 <= v <= INT32_MAX.
bool fits_int(int64_t v);

}  // namespace cere::ktest
