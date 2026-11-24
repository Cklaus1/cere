#pragma once
// Inkling 2 Mini (inkling design note 3.2, G1): the short causal depthwise convolutions (kernel 4), static and
// dynamic, CPU reference (kernels: executor/cuda/src/kernels/inkling/conv.h). REF modeling_inkling2_loop.py:
// Inkling2LoopShortConvolution :329-393 (post-attention and post-MLP convs on the hidden, C = 2304; the sliding
// layers' K and V convs, C = 1024; FP32 weights, `.to(float32)`), Inkling2LoopDynamicShortConvolution :396-455 (the
// full layers' K and V, C = 1024 = 8 KV heads x 128). Every input is BF16 (o_proj / MoE / k-v linear outputs), the
// math FP32, the output rounded once to BF16 (the input dtype). The checkpoint's conv weights are BF16, widened
// exactly (REF loads them into FP32 parameters).
//
// State: FP32 [3, C] per conv per visit per session, state[j * C + c]; row 0 is the oldest. Before the first row it
// is zero (a fresh sequence). For the next row x[t] the window is win[0..3] = x[t-3], x[t-2], x[t-1], x[t] (FP32),
// entries before the call's first row read from the state (x[-3] = state row 0, x[-2] = row 1, x[-1] = row 2). After
// n rows the state is the last three entries of [state rows, x rows] in order (n >= 3: x[n-3], x[n-2], x[n-1]); a
// chunked prefill with the state carried equals the whole prefill bit for bit, and decode is n = 1.
//
// Static (w BF16 [C, 4], checkpoint `[C, 1, 4]`, tap j multiplies win[j]: conv1d with left padding 3):
//   acc = 0; for j = 0, 1, 2, 3: acc = fma(float(w[c, j]), win[j], acc)
//   y   = encode_bf16(div(add(acc, win[3]), kConvStaticScale))      kConvStaticScale = fp32(sqrt(5)) (0x400f1bbd)
// Dynamic (static_w BF16 [4, C], tap index = backward distance (lag), lag 0 is the current row; dyn BF16 = the
// weight_proj output [n, heads * 4], dyn[t, h * 4 + lag], one tap set per KV head h = c / 128):
//   tap[lag] = add(float(static_w[lag, c]), float(dyn[t, (c / 128) * 4 + lag]))
//   acc = 0; for lag = 0, 1, 2, 3: acc = fma(tap[lag], win[3 - lag], acc)
//   y   = encode_bf16(div(add(acc, win[3]), kConvDynamicScale))     kConvDynamicScale = sqrt(2 * 4 + 1) = 3
#include "common_ref.h"

namespace ref {

constexpr int kConvTaps = 4;
constexpr int kConvStateRows = 3;
constexpr int kConvHeadDim = 128;  // channels per KV head (one dynamic tap set each)
constexpr float kConvStaticScale = 2.236068010330200195f;  // fp32(sqrt(5)) = 0x400f1bbd
constexpr float kConvDynamicScale = 3.0f;

CERE_HD inline float conv_static_point(const float *win, const float *w) {
    float acc = 0.0f;
    for (int j = 0; j < kConvTaps; ++j) acc = fma(w[j], win[j], acc);
    return div(add(acc, win[3]), kConvStaticScale);
}

CERE_HD inline float conv_dynamic_tap(float static_w, float dyn) { return add(static_w, dyn); }

CERE_HD inline float conv_dynamic_point(const float *win, const float *tap) {
    float acc = 0.0f;
    for (int lag = 0; lag < kConvTaps; ++lag) acc = fma(tap[lag], win[3 - lag], acc);
    return div(add(acc, win[3]), kConvDynamicScale);
}

namespace conv_detail {
// The window of row t (relative to the call's first row) for channel c.
inline void window(const uint16_t *x, int64_t ldx, int t, int C, int c, const float *state, float *win) {
    for (int j = 0; j < kConvTaps; ++j) {
        const int r = t - 3 + j;
        win[j] = r >= 0 ? decode_bf16(x[int64_t(r) * ldx + c]) : state[(kConvStateRows + r) * C + c];
    }
}

inline void advance_state(const uint16_t *x, int64_t ldx, int n, int C, float *state) {
    for (int c = 0; c < C; ++c) {
        float old[kConvStateRows];
        for (int j = 0; j < kConvStateRows; ++j) old[j] = state[j * C + c];
        for (int j = 0; j < kConvStateRows; ++j) {
            const int r = n - kConvStateRows + j;
            state[j * C + c] = r >= 0 ? decode_bf16(x[int64_t(r) * ldx + c]) : old[kConvStateRows + r];
        }
    }
}
}  // namespace conv_detail

// One sequence of n rows: x BF16 [n, C] at row stride ldx, w BF16 [C, 4], state FP32 [3, C] in/out, y BF16 [n, C]
// at row stride ldy (y must not alias x).
inline void conv_static_seq(const uint16_t *x, int64_t ldx, int n, int C, const uint16_t *w, float *state,
                            uint16_t *y, int64_t ldy) {
    for (int t = 0; t < n; ++t) {
        for (int c = 0; c < C; ++c) {
            float win[kConvTaps], wf[kConvTaps];
            conv_detail::window(x, ldx, t, C, c, state, win);
            for (int j = 0; j < kConvTaps; ++j) wf[j] = decode_bf16(w[c * kConvTaps + j]);
            y[int64_t(t) * ldy + c] = encode_bf16(conv_static_point(win, wf));
        }
    }
    conv_detail::advance_state(x, ldx, n, C, state);
}

// One sequence of n rows: C = heads * 128, static_w BF16 [4, C], dyn BF16 [n, heads * 4] at row stride ld_dyn,
// otherwise as conv_static_seq.
inline void conv_dynamic_seq(const uint16_t *x, int64_t ldx, int n, int C, const uint16_t *static_w,
                             const uint16_t *dyn, int64_t ld_dyn, float *state, uint16_t *y, int64_t ldy) {
    for (int t = 0; t < n; ++t) {
        for (int c = 0; c < C; ++c) {
            float win[kConvTaps], tap[kConvTaps];
            conv_detail::window(x, ldx, t, C, c, state, win);
            for (int lag = 0; lag < kConvTaps; ++lag) {
                tap[lag] = conv_dynamic_tap(decode_bf16(static_w[lag * C + c]),
                                            decode_bf16(dyn[int64_t(t) * ld_dyn + (c / kConvHeadDim) * kConvTaps + lag]));
            }
            y[int64_t(t) * ldy + c] = encode_bf16(conv_dynamic_point(win, tap));
        }
    }
    conv_detail::advance_state(x, ldx, n, C, state);
}

}  // namespace ref
