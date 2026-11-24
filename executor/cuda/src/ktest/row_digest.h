// Per-row digests of a BF16 tensor (m210 design note D15⁗′ layout probe; libcere_ktest.so only; k8_ktest.cpp
// cere_ktest_forward_row_digests, executor/tests/gpu/test_model_k16_prefix.py K16 (10)).
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace cere::ktest {

// out[r] (device u64 [rows]) = a 64-bit digest of the bits of row r of values (device BF16 [rows, cols], row-major):
// the wrapping sum over columns c of splitmix64(c << 16 | bits[r][c]), finalized with cols. Equal rows give equal
// digests; distinct rows collide with probability ~2^-64. Deterministic (integer sums only). Enqueued on `stream`;
// rows 0 is a no-op. The launch status.
cudaError_t row_digests(const uint16_t *values, uint64_t rows, uint64_t cols, uint64_t *out, cudaStream_t stream);

}  // namespace cere::ktest
