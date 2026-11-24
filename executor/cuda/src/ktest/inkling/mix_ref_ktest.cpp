// Inkling 2 Mini (inkling design note G1, slice S3): CPU entry points of libcere_ktest.so over the residual-stream
// references (executor/ref/inkling/{norm,embed,conv,depth,msg,softcap}_ref.h), so the GPU suite
// (executor/tests/gpu/test_inkling_kernels_mix.py) compares each kernel (mix_ktest.cu) with its reference bit for bit.
// Host memory only (CPU tensors); the trailing stream the oracle appends is ignored. Each returns 0, or -1 for an
// argument it refuses. Layouts, strides and flags as the kernel entry points of mix_ktest.cu.
#include <climits>
#include <cstdint>
#include <vector>

#include "inkling/conv_ref.h"
#include "inkling/depth_ref.h"
#include "inkling/embed_ref.h"
#include "inkling/msg_ref.h"
#include "inkling/norm_ref.h"
#include "inkling/softcap_ref.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace {

float load(const void *p, bool f32, int64_t idx) {
    return f32 ? static_cast<const float *>(p)[idx] : ref::decode_bf16(static_cast<const uint16_t *>(p)[idx]);
}

void store(void *p, bool f32, int64_t idx, float v) {
    if (f32)
        static_cast<float *>(p)[idx] = v;
    else
        static_cast<uint16_t *>(p)[idx] = ref::encode_bf16(v);
}

bool fits(int64_t v, int64_t lo) { return v >= lo && v <= INT_MAX; }

}  // namespace

// y[r, :] = RMSNorm(x[r, :]), r < rows; x BF16 (x_f32 0) or FP32, w BF16 [2304] or null (unweighted), y BF16 or FP32.
CERE_KTEST int cere_ktest_inkling_ref_rms_norm(const void *x, int64_t x_f32, int64_t ldx, const uint16_t *w, float eps,
                                               int64_t rows, void *y, int64_t y_f32, int64_t ldy, void * /*stream*/) {
    if (x == nullptr || y == nullptr || rows < 0 || ldx < ref::kHidden || ldy < ref::kHidden) return -1;
    std::vector<float> xf(ref::kHidden), yf(ref::kHidden);
    for (int64_t r = 0; r < rows; ++r) {
        for (int i = 0; i < ref::kHidden; ++i) xf[i] = load(x, x_f32 != 0, r * ldx + i);
        ref::rms_norm<ref::kReduceHidden>(xf.data(), ref::kHidden, eps, w, yf.data());
        for (int i = 0; i < ref::kHidden; ++i) store(y, y_f32 != 0, r * ldy + i, yf[i]);
    }
    return 0;
}

// out[r, :] = embed_norm(table[ids[r], :]); ids int32 [rows] (< vocab, checked), table BF16 [vocab, 2304].
CERE_KTEST int cere_ktest_inkling_ref_embed_norm(const uint16_t *table, int64_t vocab, const int32_t *ids,
                                                 int64_t rows, const uint16_t *w, float eps, uint16_t *out,
                                                 int64_t ldo, void * /*stream*/) {
    if (table == nullptr || ids == nullptr || w == nullptr || out == nullptr || rows < 0 || ldo < ref::kHidden)
        return -1;
    for (int64_t r = 0; r < rows; ++r) {
        if (ids[r] < 0 || ids[r] >= vocab) return -1;
        ref::embed_norm_row(table, uint32_t(ids[r]), w, eps, out + r * ldo);
    }
    return 0;
}

// One sequence of n rows (state FP32 [3, C] in/out). dynamic 0: w BF16 [C, 4], dyn unused (null); dynamic 1:
// w = static_w BF16 [4, C], dyn BF16 [n, C / 128 * 4] at row stride ld_dyn.
CERE_KTEST int cere_ktest_inkling_ref_conv_seq(int64_t dynamic, const uint16_t *x, int64_t ldx, int64_t n, int64_t C,
                                               const uint16_t *w, const uint16_t *dyn, int64_t ld_dyn, float *state,
                                               uint16_t *y, int64_t ldy, void * /*stream*/) {
    if (x == nullptr || w == nullptr || state == nullptr || y == nullptr || !fits(n, 0) || !fits(C, 1) || ldx < C ||
        ldy < C)
        return -1;
    if (dynamic) {
        if (dyn == nullptr || C % ref::kConvHeadDim != 0 || ld_dyn < C / ref::kConvHeadDim * ref::kConvTaps) return -1;
        ref::conv_dynamic_seq(x, ldx, int(n), int(C), w, dyn, ld_dyn, state, y, ldy);
    } else {
        ref::conv_static_seq(x, ldx, int(n), int(C), w, state, y, ldy);
    }
    return 0;
}

// out[q][r, :] (out + q * out_qstride + r * ldo) = depth_mix over states s0..s{count-1} (row stride ld each; bit s
// of f32_mask: state s is FP32, else BF16), queries FP32 [nq, 2304].
CERE_KTEST int cere_ktest_inkling_ref_depth_mix(const void *s0, const void *s1, const void *s2, const void *s3,
                                                const void *s4, int64_t count, int64_t f32_mask, int64_t ld,
                                                int64_t rows, const float *queries, int64_t nq, float *out,
                                                int64_t ldo, int64_t out_qstride, void * /*stream*/) {
    const void *src[ref::kMaxDepthStates] = {s0, s1, s2, s3, s4};
    if (count < 1 || count > ref::kMaxDepthStates || rows < 0 || ld < ref::kHidden || queries == nullptr || nq < 1 ||
        nq > 4 || out == nullptr || ldo < ref::kHidden)
        return -1;
    for (int s = 0; s < count; ++s)
        if (src[s] == nullptr) return -1;
    std::vector<float> st(size_t(count) * ref::kHidden), o(size_t(nq) * ref::kHidden);
    const float *ptrs[ref::kMaxDepthStates];
    for (int s = 0; s < count; ++s) ptrs[s] = &st[size_t(s) * ref::kHidden];
    for (int64_t r = 0; r < rows; ++r) {
        for (int s = 0; s < count; ++s)
            for (int i = 0; i < ref::kHidden; ++i)
                st[size_t(s) * ref::kHidden + i] = load(src[s], (f32_mask >> s) & 1, r * ld + i);
        ref::depth_mix_row(ptrs, int(count), queries, ref::kHidden, int(nq), o.data(), ref::kHidden);
        for (int64_t q = 0; q < nq; ++q)
            for (int i = 0; i < ref::kHidden; ++i) out[q * out_qstride + r * ldo + i] = o[size_t(q) * ref::kHidden + i];
    }
    return 0;
}

// r[t, :] = attn_residual(boundary, h[t, :], o[t, :]) (h unread, may be null, at a boundary).
CERE_KTEST int cere_ktest_inkling_ref_attn_residual(int64_t boundary, const uint16_t *h, int64_t ldh, const uint16_t *o,
                                                    int64_t ldo, int64_t rows, float *r, int64_t ldr,
                                                    void * /*stream*/) {
    if ((!boundary && h == nullptr) || o == nullptr || r == nullptr || rows < 0) return -1;
    for (int64_t t = 0; t < rows; ++t)
        for (int i = 0; i < ref::kHidden; ++i)
            r[t * ldr + i] = ref::attn_residual(boundary != 0, boundary ? uint16_t(0) : h[t * ldh + i], o[t * ldo + i]);
    return 0;
}

// h_next[t, :] = layer_out(r[t, :], d[t, :]).
CERE_KTEST int cere_ktest_inkling_ref_layer_out(const float *r, int64_t ldr, const uint16_t *d, int64_t ldd,
                                                int64_t rows, uint16_t *h, int64_t ldh, void * /*stream*/) {
    if (r == nullptr || d == nullptr || h == nullptr || rows < 0) return -1;
    for (int64_t t = 0; t < rows; ++t)
        for (int i = 0; i < ref::kHidden; ++i) h[t * ldh + i] = ref::layer_out(r[t * ldr + i], d[t * ldd + i]);
    return 0;
}

// out[t, :] = msg_gate(h[t, :], msg[t, :]); down BF16 [64, 4608], forget / gain BF16 [2304, 64].
CERE_KTEST int cere_ktest_inkling_ref_msg_gate(const float *h, int64_t ldh, const float *msg, int64_t ldm,
                                               int64_t rows, const uint16_t *down, const uint16_t *forget,
                                               const uint16_t *gain, float eps, float *out, int64_t ldo,
                                               void * /*stream*/) {
    if (h == nullptr || msg == nullptr || down == nullptr || forget == nullptr || gain == nullptr || out == nullptr ||
        rows < 0)
        return -1;
    for (int64_t t = 0; t < rows; ++t)
        ref::msg_gate_row(h + t * ldh, msg + t * ldm, down, forget, gain, eps, out + t * ldo);
    return 0;
}

// logits[r, i] = softcap(logits[r, i]) in place, i < vocab.
CERE_KTEST int cere_ktest_inkling_ref_softcap(float *logits, int64_t rows, int64_t vocab, int64_t ld,
                                              void * /*stream*/) {
    if (logits == nullptr || rows < 0 || vocab < 0 || ld < vocab) return -1;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t i = 0; i < vocab; ++i) logits[r * ld + i] = ref::softcap(logits[r * ld + i]);
    return 0;
}
