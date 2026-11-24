// K2 MXFP8 weights: contract in k2_mxfp8.h, the per-element conversion in k2_mxfp8.cuh.
#include "kernels/k2_mxfp8.h"

#include "kernels/k2_mxfp8.cuh"

#include <cuda_fp8.h>

#include <algorithm>

namespace cere::kernels::k2m {
namespace {

constexpr uint32_t kDequantThreads = 256;
constexpr uint32_t kA8Threads = 256;

bool aligned16(const void *p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

// Thread i: E4M3 bytes 16 i .. 16 i + 15 of q (one scale group: 16 divides 32 and every row starts at a multiple of
// 32), their scale scales[16 i / 32] (row j's group g is byte j * k / 32 + g = (j * k + 32 g) / 32), 32 output bytes.
__global__ void __launch_bounds__(kDequantThreads)
    dequant_kernel(const uint4 *__restrict__ q, const uint8_t *__restrict__ scales, size_t chunks,
                   uint4 *__restrict__ w) {
    const size_t i = static_cast<size_t>(blockIdx.x) * kDequantThreads + threadIdx.x;
    if (i >= chunks) {
        return;
    }
    const uint4 v = q[i];
    const uint32_t add = exponent_add(scales[i / 2]);
    w[2 * i] = bf16x8(make_uint2(v.x, v.y), add);
    w[2 * i + 1] = bf16x8(make_uint2(v.z, v.w), add);
}

// Byte offset of scale (r, c) of a [., cols] scale matrix in the blocked layout (k2_mxfp8.h).
__device__ __forceinline__ size_t blocked_offset(uint32_t r, uint32_t c, uint32_t cols) {
    return (static_cast<size_t>(r / 128) * ((cols + 3) / 4) + c / 4) * 512 + (r % 32) * 16 + (r % 128 / 32) * 4 +
           c % 4;
}

// Thread i: group i of x (row i / cols, k 32 (i % cols) .. + 31; 64 contiguous bytes). amax = 1.f * 2^E (bits), and
// 448 = 1.75 * 2^8, so the smallest e with amax <= 448 * 2^e is E - 8, plus 1 when f > 0.75 (mantissa field above
// 0x600000); an F32 subnormal amax (exponent field 0) and amax 0 clamp to -127. e <= 121 for every input (field
// <= 255), so 2^-e is the normal F32 with biased exponent 127 - e, and x * 2^-e is exact before the one E4M3
// rounding.
__global__ void __launch_bounds__(kA8Threads)
    quantize_x_kernel(const uint4 *__restrict__ x, uint32_t cols, size_t groups, uint4 *__restrict__ q,
                      uint8_t *__restrict__ scales) {
    const size_t i = static_cast<size_t>(blockIdx.x) * kA8Threads + threadIdx.x;
    if (i >= groups) {
        return;
    }
    float v[kGroupK];
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
        const uint4 u = x[4 * i + j];
        const uint32_t words[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (uint32_t t = 0; t < 4; ++t) {
            v[8 * j + 2 * t] = __uint_as_float(words[t] << 16);
            v[8 * j + 2 * t + 1] = __uint_as_float(words[t] & 0xFFFF0000u);
        }
    }
    float amax = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kGroupK; ++j) {
        amax = fmaxf(amax, fabsf(v[j]));
    }
    const uint32_t bits = __float_as_uint(amax);
    const int32_t field = static_cast<int32_t>(bits >> 23);
    const int32_t e =
        field == 0 ? -127 : max(-127, min(127, field - 127 - 8 + ((bits & 0x7FFFFFu) > 0x600000u ? 1 : 0)));
    const float inv = __uint_as_float(static_cast<uint32_t>(127 - e) << 23);
    uint32_t out[8];
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j) {
        const __nv_fp8x2_storage_t lo =
            __nv_cvt_float2_to_fp8x2(make_float2(v[4 * j] * inv, v[4 * j + 1] * inv), __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(
            make_float2(v[4 * j + 2] * inv, v[4 * j + 3] * inv), __NV_SATFINITE, __NV_E4M3);
        out[j] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    }
    q[2 * i] = make_uint4(out[0], out[1], out[2], out[3]);
    q[2 * i + 1] = make_uint4(out[4], out[5], out[6], out[7]);
    scales[blocked_offset(static_cast<uint32_t>(i / cols), static_cast<uint32_t>(i % cols), cols)] =
        static_cast<uint8_t>(e + 127);
}

__global__ void __launch_bounds__(kA8Threads)
    repack_kernel(const uint8_t *__restrict__ scales, uint32_t cols, size_t count, uint8_t *__restrict__ blocked) {
    const size_t i = static_cast<size_t>(blockIdx.x) * kA8Threads + threadIdx.x;
    if (i < count) {
        blocked[blocked_offset(static_cast<uint32_t>(i / cols), static_cast<uint32_t>(i % cols), cols)] = scales[i];
    }
}

// Blocks of kA8Threads for `threads` (0 when the grid would exceed the launch limit).
uint32_t a8_blocks(size_t threads) {
    const size_t blocks = (threads + kA8Threads - 1) / kA8Threads;
    return blocks > 0x7fffffffu ? 0 : static_cast<uint32_t>(blocks);
}

}  // namespace

cudaError_t dequantize(const uint8_t *q, const uint8_t *scales, uint32_t n, uint32_t k, __nv_bfloat16 *w,
                       cudaStream_t stream) {
    if (q == nullptr || scales == nullptr || w == nullptr || !aligned16(q) || !aligned16(w) || n == 0 || k == 0 ||
        k % kGroupK != 0) {
        return cudaErrorInvalidValue;
    }
    const size_t chunks = size_t{n} * k / 16;
    const size_t blocks = (chunks + kDequantThreads - 1) / kDequantThreads;
    if (blocks > 0x7fffffffu) {
        return cudaErrorInvalidValue;
    }
    dequant_kernel<<<static_cast<uint32_t>(blocks), kDequantThreads, 0, stream>>>(
        reinterpret_cast<const uint4 *>(q), scales, chunks, reinterpret_cast<uint4 *>(w));
    return cudaGetLastError();
}

size_t blocked_scale_bytes(uint32_t rows, uint32_t cols) {
    return (size_t{rows} + 127) / 128 * 128 * ((size_t{cols} + 3) / 4 * 4);
}

cudaError_t quantize_x(const __nv_bfloat16 *x, uint32_t m, uint32_t k, uint8_t *q, uint8_t *scales,
                       cudaStream_t stream) {
    if (x == nullptr || q == nullptr || scales == nullptr || !aligned16(x) || !aligned16(q) || m == 0 || k == 0 ||
        k % kGroupK != 0) {
        return cudaErrorInvalidValue;
    }
    const size_t groups = size_t{m} * (k / kGroupK);
    const uint32_t blocks = a8_blocks(groups);
    if (blocks == 0) {
        return cudaErrorInvalidValue;
    }
    quantize_x_kernel<<<blocks, kA8Threads, 0, stream>>>(reinterpret_cast<const uint4 *>(x), k / kGroupK, groups,
                                                          reinterpret_cast<uint4 *>(q), scales);
    return cudaGetLastError();
}

cudaError_t repack_scales(const uint8_t *scales, uint32_t n, uint32_t k, uint8_t *blocked, cudaStream_t stream) {
    if (scales == nullptr || blocked == nullptr || n == 0 || k == 0 || k % kGroupK != 0) {
        return cudaErrorInvalidValue;
    }
    const size_t count = size_t{n} * (k / kGroupK);
    const uint32_t blocks = a8_blocks(count);
    if (blocks == 0) {
        return cudaErrorInvalidValue;
    }
    repack_kernel<<<blocks, kA8Threads, 0, stream>>>(scales, k / kGroupK, count, blocked);
    return cudaGetLastError();
}

void Table::add(const void *w, const uint8_t *scales, uint32_t n, uint32_t k) {
    const auto at = std::lower_bound(entries_.begin(), entries_.end(), w,
                                     [](const Entry &e, const void *p) { return e.w < p; });
    entries_.insert(at, Entry{w, scales, n, k});
}

const Entry *Table::find(const void *w) const {
    const auto at = std::lower_bound(entries_.begin(), entries_.end(), w,
                                     [](const Entry &e, const void *p) { return e.w < p; });
    return at != entries_.end() && at->w == w ? &*at : nullptr;
}

bool Table::has_shape(uint32_t n, uint32_t k) const {
    return std::any_of(entries_.begin(), entries_.end(), [&](const Entry &e) { return e.n == n && e.k == k; });
}

size_t Table::staging_needed() const {
    size_t bytes = 0;
    for (const Entry &e : entries_) {
        bytes = std::max(bytes, size_t{e.n} * e.k * sizeof(__nv_bfloat16));
    }
    return bytes;
}

}  // namespace cere::kernels::k2m
