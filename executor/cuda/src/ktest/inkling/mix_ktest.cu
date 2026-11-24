// Inkling 2 Mini (inkling design note G1, slice S3): kernel entry points of libcere_ktest.so over the residual-stream
// kernels (executor/cuda/src/kernels/inkling/{norm,conv,depth,msg,softcap}.h) and their fail-first mutants
// (mix_mutants.h), compared bit for bit with the CPU references of mix_ref_ktest.cpp by
// executor/tests/gpu/test_inkling_kernels_mix.py. Device pointers; each enqueues on `stream` and returns the launch
// status (0 = cudaSuccess).
#include <climits>
#include <cstdint>

#include "kernels/inkling/conv.h"
#include "kernels/inkling/depth.h"
#include "kernels/inkling/mix_mutants.h"
#include "kernels/inkling/msg.h"
#include "kernels/inkling/norm.h"
#include "kernels/inkling/softcap.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace ink = cere::inkling;

namespace {

int status(cudaError_t e) { return int(e); }

bool fits(int64_t v) { return v >= 0 && v <= INT_MAX; }

const ink::bf16 *b(const void *p) { return static_cast<const ink::bf16 *>(p); }
ink::bf16 *b(void *p) { return static_cast<ink::bf16 *>(p); }

}  // namespace

CERE_KTEST int cere_ktest_inkling_rms_norm(const void *x, int64_t x_f32, int64_t ldx, const void *w, float eps,
                                           int64_t rows, void *y, int64_t y_f32, int64_t ldy, cudaStream_t stream) {
    return status(ink::rms_norm(x, x_f32 != 0, ldx, b(w), eps, rows, y, y_f32 != 0, ldy, stream));
}

// ids int32 [rows] on the device (the caller keeps them < vocab).
CERE_KTEST int cere_ktest_inkling_embed_norm(const void *table, const void *ids, int64_t rows, const void *w, float eps,
                                             void *out, int64_t ldo, cudaStream_t stream) {
    return status(ink::embed_norm(b(table), static_cast<const uint32_t *>(ids), rows, b(w), eps, b(out), ldo, stream));
}

// dynamic 0: conv_static_prefill (w BF16 [C, 4]); 1: conv_dynamic_prefill (w = static_w BF16 [4, C], dyn). mutant 1:
// the no-residual mutant of the same conv.
CERE_KTEST int cere_ktest_inkling_conv_prefill(int64_t dynamic, int64_t mutant, const void *x, int64_t ldx, int64_t n,
                                               int64_t C, const void *w, const void *dyn, int64_t ld_dyn, float *state,
                                               void *y, int64_t ldy, cudaStream_t stream) {
    if (!fits(C)) return status(cudaErrorInvalidValue);
    const int c = int(C);
    if (dynamic)
        return status(mutant ? ink::mutant::conv_dynamic_prefill_no_residual(b(x), ldx, n, c, b(w), b(dyn), ld_dyn,
                                                                             state, b(y), ldy, stream)
                             : ink::conv_dynamic_prefill(b(x), ldx, n, c, b(w), b(dyn), ld_dyn, state, b(y), ldy,
                                                         stream));
    return status(mutant ? ink::mutant::conv_static_prefill_no_residual(b(x), ldx, n, c, b(w), state, b(y), ldy,
                                                                        stream)
                         : ink::conv_static_prefill(b(x), ldx, n, c, b(w), state, b(y), ldy, stream));
}

// M rows, row r with its own state: states = device array of M float* (int64 data pointers).
CERE_KTEST int cere_ktest_inkling_conv_decode(int64_t dynamic, const void *x, int64_t ldx, int64_t M, int64_t C,
                                              const void *w, const void *dyn, int64_t ld_dyn, const void *states,
                                              void *y, int64_t ldy, cudaStream_t stream) {
    if (!fits(C)) return status(cudaErrorInvalidValue);
    const auto st = static_cast<float *const *>(states);
    if (dynamic)
        return status(ink::conv_dynamic_decode(b(x), ldx, M, int(C), b(w), b(dyn), ld_dyn, st, b(y), ldy, stream));
    return status(ink::conv_static_decode(b(x), ldx, M, int(C), b(w), st, b(y), ldy, stream));
}

// States s0..s{count-1} at row stride ld (bit s of f32_mask: FP32, else BF16); mutant_w non-null: the weighted-norm
// mutant with that BF16 [2304] weight.
CERE_KTEST int cere_ktest_inkling_depth_mix(const void *s0, const void *s1, const void *s2, const void *s3,
                                            const void *s4, int64_t count, int64_t f32_mask, int64_t ld, int64_t rows,
                                            const float *queries, int64_t nq, float *out, int64_t ldo,
                                            int64_t out_qstride, const void *mutant_w, cudaStream_t stream) {
    if (count < 1 || count > ink::kDepthMaxStates || nq < 1 || nq > ink::kDepthMaxQueries)
        return status(cudaErrorInvalidValue);
    const void *src[ink::kDepthMaxStates] = {s0, s1, s2, s3, s4};
    ink::DepthStates states{};
    states.count = int(count);
    for (int s = 0; s < states.count; ++s) states.s[s] = {src[s], ld, int((f32_mask >> s) & 1)};
    if (mutant_w != nullptr)
        return status(ink::mutant::depth_mix_weighted_norm(states, rows, queries, int(nq), out, ldo, out_qstride,
                                                           b(mutant_w), stream));
    return status(ink::depth_mix(states, rows, queries, int(nq), out, ldo, out_qstride, stream));
}

// The fused depth mix + weighted RMSNorm (depth_mix_norm): weights w BF16 [nq, 2304] contiguous, outputs y BF16,
// query q's row r at y + q * y_qstride + r * ldy.
CERE_KTEST int cere_ktest_inkling_depth_mix_norm(const void *s0, const void *s1, const void *s2, const void *s3,
                                                 const void *s4, int64_t count, int64_t f32_mask, int64_t ld,
                                                 int64_t rows, const float *queries, int64_t nq, const void *w,
                                                 float eps, void *y, int64_t ldy, int64_t y_qstride,
                                                 cudaStream_t stream) {
    if (count < 1 || count > ink::kDepthMaxStates || nq < 1 || nq > ink::kDepthMaxQueries || w == nullptr ||
        y == nullptr)
        return status(cudaErrorInvalidValue);
    constexpr int64_t kRow = 2304;  // one weight row
    const void *src[ink::kDepthMaxStates] = {s0, s1, s2, s3, s4};
    ink::DepthStates states{};
    states.count = int(count);
    for (int s = 0; s < states.count; ++s) states.s[s] = {src[s], ld, int((f32_mask >> s) & 1)};
    ink::DepthNorm norm{};
    for (int q = 0; q < int(nq); ++q) {
        norm.w[q] = b(w) + q * kRow;
        norm.y[q] = b(y) + q * y_qstride;
    }
    norm.ldy = ldy;
    norm.eps = eps;
    return status(ink::depth_mix_norm(states, rows, queries, int(nq), norm, stream));
}

CERE_KTEST int cere_ktest_inkling_attn_residual(int64_t boundary, const void *h, int64_t ldh, const void *o,
                                                int64_t ldo, int64_t rows, float *r, int64_t ldr,
                                                cudaStream_t stream) {
    return status(ink::attn_residual(boundary != 0, b(h), ldh, b(o), ldo, rows, r, ldr, stream));
}

CERE_KTEST int cere_ktest_inkling_layer_out(const float *r, int64_t ldr, const void *d, int64_t ldd, int64_t rows,
                                            void *h, int64_t ldh, cudaStream_t stream) {
    return status(ink::layer_out(r, ldr, b(d), ldd, rows, b(h), ldh, stream));
}

// mutant 1: forget and message gates swapped. The workspace is stream-ordered scratch of this call.
CERE_KTEST int cere_ktest_inkling_msg_gate(int64_t mutant, const float *h, int64_t ldh, const float *msg, int64_t ldm,
                                           int64_t rows, const void *down, const void *forget, const void *gain,
                                           float eps, float *out, int64_t ldo, cudaStream_t stream) {
    if (rows < 1) return status(cudaErrorInvalidValue);
    float *ws = nullptr;
    const uint64_t bytes = ink::msg_gate_workspace_floats(rows) * sizeof(float);
    cudaError_t e = cudaMallocAsync(reinterpret_cast<void **>(&ws), bytes, stream);
    if (e != cudaSuccess) return status(e);
    e = mutant ? ink::mutant::msg_gate_swapped(h, ldh, msg, ldm, rows, b(down), b(forget), b(gain), eps, out, ldo, ws,
                                               stream)
               : ink::msg_gate(h, ldh, msg, ldm, rows, b(down), b(forget), b(gain), eps, out, ldo, ws, stream);
    const cudaError_t f = cudaFreeAsync(ws, stream);
    return status(e != cudaSuccess ? e : f);
}

// The fused gate + weighted norm (S8) over `count` gates: table host int64 [count, 7] = (h, msg, down, forget, gain,
// norm_w, y) device pointers. The workspace is stream-ordered scratch of this call.
CERE_KTEST int cere_ktest_inkling_msg_gate_norm(const int64_t *table, int64_t count, int64_t ldh, int64_t ldm,
                                                int64_t ldy, int64_t rows, float eps, cudaStream_t stream) {
    if (rows < 1 || count < 1 || count > ink::kMsgMaxGates) return status(cudaErrorInvalidValue);
    ink::MsgGateNorm gates[ink::kMsgMaxGates];
    for (int64_t i = 0; i < count; ++i) {
        const int64_t *p = table + 7 * i;
        const auto w = [&](int k) { return reinterpret_cast<const ink::bf16 *>(p[k]); };
        gates[i] = ink::MsgGateNorm{reinterpret_cast<const float *>(p[0]), reinterpret_cast<const float *>(p[1]),
                                    w(2), w(3), w(4), w(5), reinterpret_cast<ink::bf16 *>(p[6])};
    }
    float *ws = nullptr;
    const uint64_t bytes = ink::msg_gate_norm_workspace_floats(rows) * sizeof(float);
    cudaError_t e = cudaMallocAsync(reinterpret_cast<void **>(&ws), bytes, stream);
    if (e != cudaSuccess) return status(e);
    e = ink::msg_gate_norm(gates, int(count), ldh, ldm, ldy, rows, eps, ws, stream);
    const cudaError_t f = cudaFreeAsync(ws, stream);
    return status(e != cudaSuccess ? e : f);
}

CERE_KTEST int cere_ktest_inkling_softcap(float *logits, int64_t rows, int64_t vocab, int64_t ld,
                                          cudaStream_t stream) {
    return status(ink::softcap(logits, rows, vocab, ld, stream));
}
