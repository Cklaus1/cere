// K2 MXFP8 weights (track Fp4S3, fp4local design note "S3 perf plan", FL-30): the dense projections of the S3
// FP4-Local artifact (GDN in_proj_{qkv,z,a,b} and out_proj, attention q/k/v/o and indexer.index_qk_proj, shared expert
// gate/up/down) stay as the checkpoint stores them: q E4M3 [n, k] row-major (`X.weight`) and one UE8M0 scale byte per
// 32 consecutive k of a row, s [n, k / 32] (`X.weight_scale`). The weight's value is w[j, i] = q[j, i] * 2^(s[j, i /
// 32] - 127). With every scale byte in [kMinScale, kMaxScale] (`cerec convert --format s3` refuses any other, and any
// E4M3 NaN byte) every such value is a normal BF16 number, so it is exact in BF16: the BF16 tensor `cerec convert`'s
// S2 path stores for X.weight is exactly these values (fp4local design note D4).
//
// Bit-identity contract (S3 is gated by D9 token identity against the S2 upcast profile): every GemmPlan call on an
// MXFP8 weight (GemmPlan::use_mxfp8) returns the bits the same call returns on that weight's BF16 upcast. Where the BF16
// call runs the small-M kernel (k2_small_m.h), the MXFP8 call runs the same kernel, same tiles, same split and the same
// reduction order, with the E4M3 tile and its scale (one byte per row per 32-element k step) turned into the BF16
// operand by an exponent add before the MMA; everywhere else (cuBLASLt buckets and pins, the rows plans' gemv/batched/
// per-row paths) dequantize() writes the BF16 upcast into the table's staging buffer and the plan runs its BF16 path
// on that copy.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cere::kernels::k2m {

constexpr uint32_t kGroupK = 32;  // k per scale byte
// Scale bytes whose every E4M3 product is a normal BF16 (E4M3 magnitudes 2^-9 .. 448): 2^-9 * 2^(10 - 127) = 2^-126,
// 448 * 2^(246 - 127) < 2^128.
constexpr uint8_t kMinScale = 10;
constexpr uint8_t kMaxScale = 246;

// The exact BF16 upcast of an MXFP8 weight: w[j, i] = q[j, i] * 2^(scales[j, i / 32] - 127) for j < n, i < k (k a
// positive multiple of 32; q and w 16-byte aligned, scales any alignment). Bit-equal to cerec convert's S2 upcast.
// cudaErrorInvalidValue outside the contract, else the launch status. No allocation, no synchronization.
cudaError_t dequantize(const uint8_t *q, const uint8_t *scales, uint32_t n, uint32_t k, __nv_bfloat16 *w,
                       cudaStream_t stream);

// W8A8 operands (the k2_gemm lever's small_m_a8, k2-prefill design note P3; numerics-changing, opt-in): cuBLASLt's
// MXFP8 GEMM takes both operands as E4M3 with UE8M0 scales (CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0) in its blocked
// layout: a [rows, cols] scale matrix (cols = k / 32) stored as 128 x 4 tiles of 512 bytes, the tiles row-major
// (ceil(cols / 4) per tile row), byte (r % 32) * 16 + (r % 128 / 32) * 4 + c % 4 inside a tile. Padding bytes (rows
// past `rows`, columns past `cols`) are never written; they belong to no output.
// Bytes of that layout for `rows` x `cols`: rows rounded up to 128 times cols rounded up to 4.
size_t blocked_scale_bytes(uint32_t rows, uint32_t cols);
// x BF16 [m, k] row-major -> q E4M3 [m, k] row-major and its scales (blocked, `scales`): per row and 32 consecutive
// k, e = the smallest integer with amax <= 448 * 2^e (amax the group's largest |x|; e = -127 for an all-zero group),
// clamped to [-127, 127]; scale byte e + 127; q = the RN, satfinite E4M3 of x * 2^-e (so no element saturates unless
// the clamp did). k a positive multiple of 32, x and q 16-byte aligned. cudaErrorInvalidValue outside the contract,
// else the launch status. Row r's bytes depend on row r of x alone. No allocation, no synchronization.
cudaError_t quantize_x(const __nv_bfloat16 *x, uint32_t m, uint32_t k, uint8_t *q, uint8_t *scales,
                       cudaStream_t stream);
// An MXFP8 weight's row-major UE8M0 scales [n, k / 32] -> the blocked layout (`blocked`). k a positive multiple of 32.
// cudaErrorInvalidValue outside the contract, else the launch status.
cudaError_t repack_scales(const uint8_t *scales, uint32_t n, uint32_t k, uint8_t *blocked, cudaStream_t stream);

// The MXFP8 weights of one model (model binder) by the weight pointer the forward passes (the E4M3 bytes, cast to
// the BF16 pointer type every GemmPlan call takes), and the staging buffer of the dequantize-then-BF16 path (device,
// 256-byte aligned, at least the largest entry's 2 n k bytes; carved in the model's workspace), which the W8A8 path
// (GemmPlan::use_a8) holds its operands in instead.
struct Entry {
    const void *w;            // q E4M3 [n, k]
    const uint8_t *scales;    // UE8M0 [n, k / 32]
    uint32_t n;
    uint32_t k;
};
class Table {
public:
    void add(const void *w, const uint8_t *scales, uint32_t n, uint32_t k);  // keeps the entries sorted by w
    const Entry *find(const void *w) const;  // nullptr for a weight that is not MXFP8
    bool has_shape(uint32_t n, uint32_t k) const;  // some entry is [n, k]
    bool empty() const { return entries_.empty(); }
    size_t size() const { return entries_.size(); }
    // The largest entry's BF16 upcast, 2 n k bytes (0 when empty): the staging size the table needs.
    size_t staging_needed() const;
    void set_staging(__nv_bfloat16 *staging, size_t bytes) {
        staging_ = staging;
        staging_bytes_ = bytes;
    }
    __nv_bfloat16 *staging() const { return staging_; }
    size_t staging_bytes() const { return staging_bytes_; }

private:
    std::vector<Entry> entries_;
    __nv_bfloat16 *staging_ = nullptr;
    size_t staging_bytes_ = 0;
};

}  // namespace cere::kernels::k2m
