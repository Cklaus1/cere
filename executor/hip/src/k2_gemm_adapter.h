// AmdAmd PAR-6: the HIP stand-in for executor/cuda/src/kernels/k2_gemm.h, installed by gen_hip.py as
// build/hip/gen/kernels/k2_gemm.h, so hipified kernels and the model layer compile against the HIP GEMM
// (k2_gemm_hip.h: the small-M GEMV for row classes <= 16, the shift-invariant k2_wmma above) with CUDA's names and
// signatures. What CUDA K2 has and HIP does not, and what the HIP plan does instead:
//   - pins (torch's recorded cuBLASLt configurations): gemm_pins_for answers none, init takes no pins;
//   - the small-M / fast / FP8 tensor-core kernels: use_small_m / use_early_weights / use_a8 are recorded levers with
//     no effect (small_m() stays false; the HIP small-M GEMV runs whatever the lever); a plan given an FP8 table
//     refuses (CERE_ERR_UNSUPPORTED) a run on a weight the table holds;
//   - MXFP8 (the S3 artifact): a weight the k2m table holds runs the small-M GEMV on its E4M3 bytes (row classes
//     <= 16) or k2m::dequantize into the staging then the BF16 plan: either way its BF16 upcast's bits (k2m contract);
//   - rows plans: one small-M GEMV launch, bit-equal to CUDA's per-row fallback (each row is independent of the call);
//   - LoRA: as CUDA's run(), the base GEMM then lora::Table::apply for a weight the active table holds;
//   - gemm_segments: `count` run() calls, each bit-equal to plans[i]->run as CUDA's contract states.
#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "cere_executor.h"
#include "k2_gemm_hip.h"
#include "kernels/k2_fp8.h"
#include "kernels/k2_gemv.h"
#include "kernels/k2_mxfp8.h"
#include "k2_mgemv.h"
#include "kernels/k2_small_m.h"
#include "kernels/lora.h"

namespace cere::kernels {

using GemmOut = cere::hipk2::GemmOut;

inline cere_status gemm_handle_create(hipblasLtHandle_t *out) {
    return out && hipblasLtCreate(out) == HIPBLAS_STATUS_SUCCESS ? CERE_OK : CERE_ERR_INTERNAL;
}
inline void gemm_handle_destroy(hipblasLtHandle_t handle) {
    if (handle) hipblasLtDestroy(handle);
}

// CUDA's pinned-configuration records (cuBLASLt enums as plain integers here): none exist for HIP.
struct GemmConfig {
    int32_t algo_id;
    int32_t tile;
    int32_t stages;
    int32_t splitk;
    int32_t reduction;
    uint32_t partials;
};
struct GemmPinRange {
    uint32_t m_lo;
    uint32_t m_hi;
    GemmConfig config;
};
struct GemmPins {
    static constexpr uint32_t kPinMinM = 17;
    uint32_t n;
    uint32_t k;
    GemmOut out;
    const GemmPinRange *ranges;
    uint32_t count;
    size_t cublaslt_version;
    bool applies(uint32_t plan_n, uint32_t plan_k, GemmOut plan_out) const {
        return plan_n == n && plan_k == k && plan_out == out;
    }
};
inline size_t gemm_pins_workspace_bytes(const GemmPins &, uint32_t) { return 0; }
inline const GemmPins *gemm_pins_for(uint32_t, uint32_t, GemmOut) { return nullptr; }

class GemmPlan {
   public:
    static constexpr uint32_t kMaxRows = 16;
    static constexpr uint32_t kRowsGemv = 1, kRowsBatched = 2, kRowsAllPaths = 3;

    cere_status init(hipblasLtHandle_t handle, uint32_t n, uint32_t k, uint32_t max_m, GemmOut out,
                     const GemmPins *pins = nullptr, void * /*pin_workspace*/ = nullptr,
                     size_t /*pin_workspace_bytes*/ = 0, bool /*m1_swap*/ = true) {
        if (pins != nullptr) return CERE_ERR_UNSUPPORTED;  // gemm_pins_for gives none on HIP
        return plan_.init(handle, n, k, out, max_m);
    }
    // CUDA K2's rows plan (every row bit-equal to `instantiated`'s m = 1 call): on HIP always its per-row fallback, m
    // single-row calls of the m = 1 algorithm (no gemvx / strided-batch one-launch path), exact by construction.
    cere_status init_rows(hipblasLtHandle_t handle, const GemmPlan &instantiated, uint32_t rows,
                          uint32_t /*fused_allowed*/ = ~uint32_t{0}, uint32_t paths = kRowsAllPaths) {
        if (!handle || rows < 1 || rows > kMaxRows || instantiated.n() == 0 || paths == 0 || (paths & ~kRowsAllPaths))
            return CERE_ERR_ARGUMENT;
        rows_ = true;
        lora_ = instantiated.lora_;
        fp8_ = instantiated.fp8_;
        mxfp8_ = instantiated.mxfp8_;
        return plan_.init(handle, instantiated.n(), instantiated.k(), instantiated.out(), kMaxRows);
    }
    bool per_row(uint32_t m) const { return rows_ && m > 1 && m <= kMaxRows; }
    bool small_m() const { return false; }  // CUDA K2's tensor-core small-M kernel does not exist on HIP
    void use_small_m(bool on) { small_m_lever_ = on; }
    void use_early_weights(bool on) { early_ = on; }
    void use_fp8(const k2f::Table *table) { fp8_ = table; }
    const k2f::Table *fp8() const { return fp8_; }
    void use_mxfp8(const k2m::Table *table) { mxfp8_ = table; }
    const k2m::Table *mxfp8() const { return mxfp8_; }
    void use_lora(lora::Table *table) { lora_ = table; }
    lora::Table *lora() const { return lora_; }
    void use_a8(bool on) { a8_ = on; }
    cere_status init_a8() { return CERE_ERR_UNSUPPORTED; }  // CUDA's W8A8 (MXFP8 tensor-core) path: none on HIP
    cere_status run(const __hip_bfloat16 *x, const __hip_bfloat16 *w, uint32_t m, void *y, hipStream_t stream,
                    uint32_t cls = 0) const {
        if (fp8_ && fp8_->find(w)) return CERE_ERR_UNSUPPORTED;  // no per-tensor FP8 GEMM on HIP
        const k2m::Entry *mx = mxfp8_ ? mxfp8_->find(w) : nullptr;
        const cere_status s = mx ? run_mxfp8(*mx, x, m, y, stream, cls) : run_base(x, w, m, y, stream, cls);
        if (s != CERE_OK || lora_ == nullptr || !lora_->active()) return s;
        const lora::Entry *e = lora_->find(w);
        if (e == nullptr) return CERE_OK;
        if (out() != GemmOut::kBF16 || e->k != k()) return CERE_ERR_ARGUMENT;
        return lora_->apply(*e, x, m, static_cast<__hip_bfloat16 *>(y), n(), stream);
    }
    uint32_t n() const { return plan_.n(); }
    uint32_t k() const { return plan_.k(); }
    uint32_t max_m() const { return rows_ ? kMaxRows : plan_.max_m(); }
    GemmOut out() const { return plan_.out(); }

   private:
    // An MXFP8 weight (track Fp4S3): k2m's contract, the bits of its BF16 upcast. Row classes <= 16 run the small-M
    // GEMV straight on the E4M3 bytes (the BF16 weight's GEMV order, k2_mgemv.h); larger ones upcast into the table's
    // staging (k2m::dequantize) and run the BF16 plan on that copy.
    cere_status run_mxfp8(const k2m::Entry &e, const __hip_bfloat16 *x, uint32_t m, void *y, hipStream_t stream,
                          uint32_t cls) const {
        if (e.n != n() || e.k != k() || m < 1 || m > max_m() || !x || !y) return CERE_ERR_ARGUMENT;
        const uint32_t choice = cls != 0 ? cls : m;
        if (rows_ || choice <= cere::hipk2::kGemvMaxM)
            return cere::hipk2::gemv(x, e.w, e.scales, y, m, e.n, e.k, out() == GemmOut::kF32, stream);
        __hip_bfloat16 *staging = mxfp8_->staging();
        if (staging == nullptr || mxfp8_->staging_bytes() < size_t(e.n) * e.k * sizeof(__hip_bfloat16))
            return CERE_ERR_INTERNAL;
        if (k2m::dequantize(static_cast<const uint8_t *>(e.w), e.scales, e.n, e.k, staging, stream) != hipSuccess)
            return CERE_ERR_INTERNAL;
        return plan_.run(x, staging, y, m, stream, cls);
    }
    cere_status run_base(const __hip_bfloat16 *x, const __hip_bfloat16 *w, uint32_t m, void *y, hipStream_t stream,
                         uint32_t cls) const {
        if (!rows_) return plan_.run(x, w, y, m, stream, cls);
        if (m < 1 || m > kMaxRows || !x || !y) return CERE_ERR_ARGUMENT;
        // the small-M GEMV's rows are independent of the call: one launch gives the m single-row calls' bits
        if (plan_.k() % 16 == 0) return plan_.run(x, w, y, m, stream, 0);
        const size_t ybytes = size_t(plan_.n()) * (plan_.out() == GemmOut::kBF16 ? 2 : 4);
        for (uint32_t i = 0; i < m; ++i)
            if (const cere_status st = plan_.run(x + size_t(i) * plan_.k(), w, static_cast<char *>(y) + i * ybytes, 1, stream, 0);
                st != CERE_OK)
                return st;
        return CERE_OK;
    }
    cere::hipk2::GemmPlan plan_;
    bool rows_ = false, small_m_lever_ = false, early_ = false, a8_ = false;
    const k2f::Table *fp8_ = nullptr;
    const k2m::Table *mxfp8_ = nullptr;
    lora::Table *lora_ = nullptr;
};

// CUDA's last-row lm_head: row m - 1 alone through `plan` at m = 1 (prefill and decode give the same logits).
inline cere_status lm_head_last_row(const GemmPlan &plan, const __hip_bfloat16 *hidden, uint32_t m,
                                    const __hip_bfloat16 *w, void *logits, hipStream_t stream) {
    if (m < 1 || !hidden) return CERE_ERR_ARGUMENT;
    return plan.run(hidden + size_t(m - 1) * plan.k(), w, 1, logits, stream);
}

inline cere_status gemm_segments(const GemmPlan *const *plans, const __hip_bfloat16 *const *w, void *const *y,
                                 uint32_t count, const __hip_bfloat16 *x, uint32_t m, hipStream_t stream,
                                 uint32_t cls = 0) {
    for (uint32_t i = 0; i < count; ++i) {
        if (const cere_status st = plans[i]->run(x, w[i], m, y[i], stream, cls); st != CERE_OK) return st;
    }
    return CERE_OK;
}

}  // namespace cere::kernels
