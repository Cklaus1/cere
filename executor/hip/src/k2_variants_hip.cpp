// AmdAmd PAR-6 step 9: the K2 GEMM variants on HIP (headers hipified from executor/cuda/src/kernels/k2_small_m.h,
// k2_fast.h, k2_fp8.h, k2_mxfp8.h, so every type the model layer holds is the CUDA one). Their kernels are NVIDIA
// tensor-core code (mma.sync, FP8 / MXFP8 MMA, cp.async) with no gfx1151 port: every `supported` predicate answers
// false and every kernel entry point refuses (hipErrorNotSupported, AGENTS.md rule 6), so callers take the paths
// their contracts define for "not available" (the BF16 K2 plan; hc_mix's kUnfused). The weight tables are real
// bookkeeping (sorted by pointer, as the CUDA ones), so a model binder can fill them; a GEMM asked to run an FP8 /
// MXFP8 table entry refuses in the HIP plan (k2_gemm_adapter.h).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <functional>

#include "kernels/k2_fast.h"
#include "kernels/k2_fp8.h"
#include "kernels/k2_mxfp8.h"
#include "kernels/k2_small_m.h"
#include "k2_mgemv.h"

namespace cere::kernels {

namespace k2s {
bool supported(uint32_t) { return false; }
bool hc_gate_supported(uint32_t) { return false; }
hipError_t gemm(const __hip_bfloat16 *, uint32_t, uint32_t, const Segment *, uint32_t, bool, hipStream_t, bool) {
    return hipErrorNotSupported;
}
hipError_t gemm_mxfp8(const __hip_bfloat16 *, uint32_t, uint32_t, const Mxfp8Segment *, uint32_t, bool, hipStream_t) {
    return hipErrorNotSupported;
}
hipError_t gemm_hc_act(const __hip_bfloat16 *, uint32_t, uint32_t, const __hip_bfloat16 *, uint32_t, __hip_bfloat16 *,
                       hipStream_t, bool) {
    return hipErrorNotSupported;
}
hipError_t gemm_hc_gate(const __hip_bfloat16 *, uint32_t, uint32_t, const __hip_bfloat16 *, uint32_t,
                        const __hip_bfloat16 *, __hip_bfloat16 *, hipStream_t, bool) {
    return hipErrorNotSupported;
}
}  // namespace k2s

namespace k2x {
bool supported(uint32_t) { return false; }
bool resident_supported(uint32_t) { return false; }
hipError_t gemm(const __hip_bfloat16 *, uint32_t, uint32_t, const k2s::Mxfp8Segment *, uint32_t, bool, hipStream_t,
                const Tune &) {
    return hipErrorNotSupported;
}
hipError_t resident(const __hip_bfloat16 *, uint32_t, uint32_t, const k2s::Mxfp8Segment *, uint32_t, bool, hipStream_t,
                    const ResidentTune &) {
    return hipErrorNotSupported;
}
hipError_t gemm_mxfp8(const __hip_bfloat16 *, uint32_t, uint32_t, const k2s::Mxfp8Segment *, uint32_t, bool,
                      hipStream_t) {
    return hipErrorNotSupported;
}
}  // namespace k2x

namespace k2f {
bool supported(uint32_t) { return false; }
hipError_t gemm(const __hip_bfloat16 *, uint32_t, uint32_t, const Segment *, uint32_t, bool, hipStream_t) {
    return hipErrorNotSupported;
}
hipError_t quantize_rows(const __hip_bfloat16 *, uint32_t, uint32_t, uint8_t *, float *, hipStream_t) {
    return hipErrorNotSupported;
}
hipError_t dequantize(const uint8_t *, const float *, uint32_t, uint32_t, __hip_bfloat16 *, hipStream_t) {
    return hipErrorNotSupported;
}
hipError_t convert_in_place(void *, uint32_t, uint32_t, void *, size_t, hipStream_t) { return hipErrorNotSupported; }
void Table::add(const void *w, uint32_t n, uint32_t k) {
    const auto at = std::lower_bound(entries_.begin(), entries_.end(), w,
                                     [](const Entry &e, const void *p) { return std::less<const void *>{}(e.w, p); });
    entries_.insert(at, Entry{w, n, k});
}
const Entry *Table::find(const void *w) const {
    const auto at = std::lower_bound(entries_.begin(), entries_.end(), w,
                                     [](const Entry &e, const void *p) { return std::less<const void *>{}(e.w, p); });
    return at != entries_.end() && at->w == w ? &*at : nullptr;
}
}  // namespace k2f

namespace k2m {
hipError_t dequantize(const uint8_t *q, const uint8_t *scales, uint32_t n, uint32_t k, __hip_bfloat16 *w,
                      hipStream_t stream) {  // the S3 staging path (k2_gemm_adapter.h GemmPlan::run_mxfp8)
    return hipk2::mxfp8_dequantize(q, scales, n, k, w, stream);
}
size_t blocked_scale_bytes(uint32_t rows, uint32_t cols) {  // the CUDA formula (sizes only; no HIP kernel uses them)
    return (size_t{rows} + 127) / 128 * 128 * ((size_t{cols} + 3) / 4 * 4);
}
hipError_t quantize_x(const __hip_bfloat16 *, uint32_t, uint32_t, uint8_t *, uint8_t *, hipStream_t) {
    return hipErrorNotSupported;
}
hipError_t repack_scales(const uint8_t *, uint32_t, uint32_t, uint8_t *, hipStream_t) { return hipErrorNotSupported; }
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
    for (const Entry &e : entries_) bytes = std::max(bytes, size_t{e.n} * e.k * sizeof(__hip_bfloat16));
    return bytes;
}
}  // namespace k2m

}  // namespace cere::kernels
