// K2 FP8 weights (track Fp8W; the k2_weights slot = CERE_LEVER_K2_WEIGHTS_FP8, opt-in, numerics-changing): the
// dense projections stored as E4M3 with one FP32 scale per output row, converted in place at INSTANTIATE.
//
// Storage of one converted weight w [n, k] (the BF16 tensor's own 2nk bytes): q E4M3 [n, k] at byte 0, then the FP32
// scales s [n] at byte n * k. Quantization (quantize_rows): s[j] = amax_j / 448 (amax_j = max |w[j, :]|, as FP32),
// q[j, i] = E4M3(w[j, i] / s[j]) rounded to nearest even and saturated to +-448 (a zero row: s 0 and q 0; a row
// holding a NaN or an Inf: s NaN and q 0, so gemm and dequantize both give NaN for it, KGM-4). q never holds the E4M3
// NaN bytes 0x7F / 0xFF, which gemm's in-register widening would read as +-480.
//
// gemm (m <= kMaxM): y[r, j] = s[j] * sum_i x[r, i] * q[j, i] with x BF16 and every q exactly as BF16 (E4M3 is a
// subset of BF16), BF16 tensor-core MMA (mma.sync m16n8k16) with FP32 accumulation, no atomics. Reduction order as
// k2_small_m.h but with 64-element steps: lane t of a step holds k = 64 s + 16 t .. + 15 (16 E4M3 bytes); MMA h (0..3)
// takes k = 64 s + 16 t + 4 h + 0..3, the same map for x and q; slice p of S = split_warps(k) <= 16 takes steps p,
// p + S, ... in increasing order on one warp, and the S slice sums are added in slice order, then multiplied by s[j]
// once. So y[r, j] is a function of (x[r, :], q[j, :], s[j], k) only: bit-identical for every m in [1, kMaxM], row
// position, column grouping and set of segments launched together, and across repeated calls. BF16 output is one
// round-to-nearest of the FP32 product. Every launch is a programmatic dependent launch (k2_small_m.h, P11R-21).
//
// m > kMaxM (prefill): GemmPlan::run dequantizes the weight to BF16 (dequantize: w'[j, i] = BF16(float(q[j, i]) *
// s[j])) into Table::staging and runs its BF16 path on w'.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cere::kernels::k2f {

constexpr uint32_t kMaxM = 16;
constexpr uint32_t kMaxSegments = 4;
constexpr uint32_t kStepK = 64;  // k per warp step

struct Segment {
    const uint8_t *q;    // E4M3 [n, k], 16-byte aligned
    const float *scale;  // FP32 [n]
    void *y;             // [m, n] row-major, BF16 or F32 (the launch's)
    uint32_t n;
};

// True when the kernels serve inner dimension k (a positive multiple of 64).
bool supported(uint32_t k);

// The slices per column (the split of k, one warp each), a function of k only.
uint32_t split_warps(uint32_t k);

// y_s = s_s * (x · q_s^T) for every segment (1 <= count <= kMaxSegments), m in [1, kMaxM], f32: F32 output, else
// BF16. x 16-byte aligned. cudaErrorInvalidValue outside the contract, else the launch status.
cudaError_t gemm(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const Segment *segments, uint32_t count, bool f32,
                 cudaStream_t stream);

// The quantization above: w BF16 [rows, k] -> q E4M3 [rows, k], scale FP32 [rows]. w must not overlap q or scale.
cudaError_t quantize_rows(const __nv_bfloat16 *w, uint32_t rows, uint32_t k, uint8_t *q, float *scale,
                          cudaStream_t stream);

// w'[j, i] = BF16(float(q[j, i]) * scale[j]) for j < n (q, w' 16-byte aligned).
cudaError_t dequantize(const uint8_t *q, const float *scale, uint32_t n, uint32_t k, __nv_bfloat16 *w,
                       cudaStream_t stream);

// Converts `storage` (BF16 [n, k] on entry) in place to the storage layout above: chunks of rows are copied to
// `scratch` and quantized back into the storage, the scales collected at the head of `scratch` and written last (FP8
// row r lands on bytes of BF16 rows <= r, already staged). Stream-ordered, no host synchronization.
// cudaErrorInvalidValue when k is unsupported, storage or scratch is misaligned (16 bytes) or scratch holds less than
// 4 n bytes (rounded up to 256) plus one row (2 k bytes).
cudaError_t convert_in_place(void *storage, uint32_t n, uint32_t k, void *scratch, size_t scratch_bytes,
                             cudaStream_t stream);

// The converted weights of one model (model::convert_fp8_weights) by the BF16 pointer the forward passes, and the
// staging buffer of the m > kMaxM path (the idle tail of a converted tensor's storage).
struct Entry {
    const void *w;
    uint32_t n;
    uint32_t k;
};
class Table {
public:
    void add(const void *w, uint32_t n, uint32_t k);  // keeps the entries sorted by w
    const Entry *find(const void *w) const;           // nullptr for a weight that is not converted
    bool empty() const { return entries_.empty(); }
    size_t size() const { return entries_.size(); }
    void set_staging(__nv_bfloat16 *staging, size_t bytes) {
        staging_ = staging;
        staging_bytes_ = bytes;
    }
    __nv_bfloat16 *staging() const { return staging_; }
    size_t staging_bytes() const { return staging_bytes_; }
    // Set when the conversion started (the BF16 bytes may be gone: INSTANTIATE is not retried).
    bool started = false;

private:
    std::vector<Entry> entries_;
    __nv_bfloat16 *staging_ = nullptr;
    size_t staging_bytes_ = 0;
};

// The scales of a converted weight [n, k] at `w`.
inline const float *scales_of(const void *w, uint32_t n, uint32_t k) {
    return reinterpret_cast<const float *>(static_cast<const uint8_t *>(w) + size_t{n} * k);
}

}  // namespace cere::kernels::k2f
