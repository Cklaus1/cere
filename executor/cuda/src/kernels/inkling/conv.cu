// Inkling 2 Mini (inkling design note 3.2, G1): the short convolutions; contracts in conv.h, the exact order in
// executor/ref/inkling/conv_ref.h (each output is ref::conv_static_point / conv_dynamic_point of its window).
// Compiled with -fmad=false.
#include "kernels/inkling/conv.h"

#include <climits>

#include "kernels/inkling/mix_mutants.h"
#include "kernels/inkling/pdl.cuh"
#include "inkling/conv_ref.h"

namespace cere::inkling {

namespace {

constexpr int kThreads = 128;   // channels per block
constexpr int kTileRows = 32;   // prefill rows per block (sequential per thread, the window slides in registers)
constexpr int K = ref::kConvTaps;
constexpr int R = ref::kConvStateRows;

// Mut: the fail-first mutant (mix_mutants.h): the point without the residual and the scale.
template <bool Mut>
__device__ inline float static_point(const float *win, const float *w) {
    if (!Mut) return ref::conv_static_point(win, w);
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) acc = ref::fma(w[j], win[j], acc);
    return acc;
}

template <bool Mut>
__device__ inline float dynamic_point(const float *win, const float *tap) {
    if (!Mut) return ref::conv_dynamic_point(win, tap);
    float acc = 0.0f;
    for (int lag = 0; lag < K; ++lag) acc = ref::fma(tap[lag], win[3 - lag], acc);
    return acc;
}

// Rows [t0, min(t0 + kTileRows, n)) of channel c. The window of row t: x rows t-3..t, rows before 0 from the state.
template <bool Dynamic, bool Mut>
__global__ void __launch_bounds__(kThreads)
    conv_prefill_kernel(const uint16_t *x, int64_t ldx, int64_t n, int C, const uint16_t *w, const uint16_t *dyn,
                        int64_t ld_dyn, const float *state, uint16_t *y, int64_t ldy) {
    pdl::enter();
    const int c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= C) return;
    const int64_t t0 = int64_t(blockIdx.y) * kTileRows;
    const int64_t t1 = t0 + kTileRows < n ? t0 + kTileRows : n;
    float win[K];
    for (int j = 0; j < R; ++j) {
        const int64_t r = t0 - R + j;
        win[j + 1] = r >= 0 ? ref::decode_bf16(x[r * ldx + c]) : state[(R + r) * C + c];
    }
    float wf[K];
    if (!Dynamic)
        for (int j = 0; j < K; ++j) wf[j] = ref::decode_bf16(w[int64_t(c) * K + j]);
    for (int64_t t = t0; t < t1; ++t) {
        for (int j = 0; j < R; ++j) win[j] = win[j + 1];
        win[R] = ref::decode_bf16(x[t * ldx + c]);
        float o;
        if (Dynamic) {
            float tap[K];
            const uint16_t *d = dyn + t * ld_dyn + (c / ref::kConvHeadDim) * K;
            for (int lag = 0; lag < K; ++lag)
                tap[lag] = ref::conv_dynamic_tap(ref::decode_bf16(w[int64_t(lag) * C + c]), ref::decode_bf16(d[lag]));
            o = dynamic_point<Mut>(win, tap);
        } else {
            o = static_point<Mut>(win, wf);
        }
        y[t * ldy + c] = ref::encode_bf16(o);
    }
}

// conv_ref.h advance_state for channel c (after every output of the call read the old state).
__global__ void __launch_bounds__(kThreads)
    conv_advance_kernel(const uint16_t *x, int64_t ldx, int64_t n, int C, float *state) {
    pdl::enter();
    const int c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= C) return;
    float old[R], nu[R];
    for (int j = 0; j < R; ++j) old[j] = state[j * C + c];
    for (int j = 0; j < R; ++j) {
        const int64_t r = n - R + j;
        nu[j] = r >= 0 ? ref::decode_bf16(x[r * ldx + c]) : old[R + r];
    }
    for (int j = 0; j < R; ++j) state[j * C + c] = nu[j];
}

// Row `blockIdx.y` with its own state: output, then the state shifts by one row (the same thread reads it first).
// The channel's weights load before the PDL wait.
template <bool Dynamic>
__global__ void __launch_bounds__(kThreads)
    conv_decode_kernel(const uint16_t *x, int64_t ldx, int C, const uint16_t *w, const uint16_t *dyn, int64_t ld_dyn,
                       float *const *states, uint16_t *y, int64_t ldy) {
    pdl::trigger();
    const int c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= C) return;
    float wf[K];  // static: w[c, j]; dynamic: the static part w[lag, c] of tap lag
    for (int j = 0; j < K; ++j) wf[j] = ref::decode_bf16(Dynamic ? w[int64_t(j) * C + c] : w[int64_t(c) * K + j]);
    pdl::wait();
    const int64_t row = blockIdx.y;
    float *state = states[row];
    float win[K];
    for (int j = 0; j < R; ++j) win[j] = state[j * C + c];
    win[R] = ref::decode_bf16(x[row * ldx + c]);
    float o;
    if (Dynamic) {
        float tap[K];
        const uint16_t *d = dyn + row * ld_dyn + (c / ref::kConvHeadDim) * K;
        for (int lag = 0; lag < K; ++lag) tap[lag] = ref::conv_dynamic_tap(wf[lag], ref::decode_bf16(d[lag]));
        o = ref::conv_dynamic_point(win, tap);
    } else {
        o = ref::conv_static_point(win, wf);
    }
    y[row * ldy + c] = ref::encode_bf16(o);
    for (int j = 0; j < R; ++j) state[j * C + c] = win[j + 1];
}

const uint16_t *u16(const bf16 *p) { return reinterpret_cast<const uint16_t *>(p); }

bool shape_ok(const bf16 *x, int64_t ldx, int64_t rows, int C, const bf16 *y, int64_t ldy, bool dynamic,
              const bf16 *dyn, int64_t ld_dyn) {
    if (x == nullptr || y == nullptr || x == y || C < 1 || rows < 1 || ldx < C || ldy < C) return false;
    if (dynamic && (C % ref::kConvHeadDim != 0 || dyn == nullptr || ld_dyn < (C / ref::kConvHeadDim) * K))
        return false;
    // grid.y: the prefill's row tiles or the decode rows (65,535 at most)
    return rows <= INT_MAX;
}

template <bool Dynamic, bool Mut>
cudaError_t prefill(const bf16 *x, int64_t ldx, int64_t n, int C, const bf16 *w, const bf16 *dyn, int64_t ld_dyn,
                    float *state, bf16 *y, int64_t ldy, cudaStream_t stream) {
    if (!shape_ok(x, ldx, n, C, y, ldy, Dynamic, dyn, ld_dyn) || w == nullptr || state == nullptr)
        return cudaErrorInvalidValue;
    const int64_t tiles = (n + kTileRows - 1) / kTileRows;
    if (tiles > 65535) return cudaErrorInvalidValue;
    const unsigned cblocks = unsigned((C + kThreads - 1) / kThreads);
    const dim3 grid(cblocks, unsigned(tiles));
    const cudaError_t err = pdl::launch(conv_prefill_kernel<Dynamic, Mut>, grid, dim3(kThreads), 0, stream, u16(x), ldx,
                                        n, C, u16(w), u16(dyn), ld_dyn, state, reinterpret_cast<uint16_t *>(y), ldy);
    if (err != cudaSuccess) return err;
    return pdl::launch(conv_advance_kernel, dim3(cblocks), dim3(kThreads), 0, stream, u16(x), ldx, n, C, state);
}

template <bool Dynamic>
cudaError_t decode(const bf16 *x, int64_t ldx, int64_t M, int C, const bf16 *w, const bf16 *dyn, int64_t ld_dyn,
                   float *const *states, bf16 *y, int64_t ldy, cudaStream_t stream) {
    if (!shape_ok(x, ldx, M, C, y, ldy, Dynamic, dyn, ld_dyn) || w == nullptr || states == nullptr || M > 65535)
        return cudaErrorInvalidValue;
    const unsigned cblocks = unsigned((C + kThreads - 1) / kThreads);
    return pdl::launch(conv_decode_kernel<Dynamic>, dim3(cblocks, unsigned(M)), dim3(kThreads), 0, stream, u16(x), ldx,
                       C, u16(w), u16(dyn), ld_dyn, states, reinterpret_cast<uint16_t *>(y), ldy);
}

}  // namespace

cudaError_t conv_static_prefill(const bf16 *x, int64_t ldx, int64_t n, int C, const bf16 *w, float *state, bf16 *y,
                                int64_t ldy, cudaStream_t stream) {
    return prefill<false, false>(x, ldx, n, C, w, nullptr, 0, state, y, ldy, stream);
}

cudaError_t conv_static_decode(const bf16 *x, int64_t ldx, int64_t M, int C, const bf16 *w, float *const *states,
                               bf16 *y, int64_t ldy, cudaStream_t stream) {
    return decode<false>(x, ldx, M, C, w, nullptr, 0, states, y, ldy, stream);
}

cudaError_t conv_dynamic_prefill(const bf16 *x, int64_t ldx, int64_t n, int C, const bf16 *static_w, const bf16 *dyn,
                                 int64_t ld_dyn, float *state, bf16 *y, int64_t ldy, cudaStream_t stream) {
    return prefill<true, false>(x, ldx, n, C, static_w, dyn, ld_dyn, state, y, ldy, stream);
}

cudaError_t conv_dynamic_decode(const bf16 *x, int64_t ldx, int64_t M, int C, const bf16 *static_w, const bf16 *dyn,
                                int64_t ld_dyn, float *const *states, bf16 *y, int64_t ldy, cudaStream_t stream) {
    return decode<true>(x, ldx, M, C, static_w, dyn, ld_dyn, states, y, ldy, stream);
}

namespace mutant {

cudaError_t conv_static_prefill_no_residual(const bf16 *x, int64_t ldx, int64_t n, int C, const bf16 *w,
                                            float *state, bf16 *y, int64_t ldy, cudaStream_t stream) {
    return prefill<false, true>(x, ldx, n, C, w, nullptr, 0, state, y, ldy, stream);
}

cudaError_t conv_dynamic_prefill_no_residual(const bf16 *x, int64_t ldx, int64_t n, int C, const bf16 *static_w,
                                             const bf16 *dyn, int64_t ld_dyn, float *state, bf16 *y, int64_t ldy,
                                             cudaStream_t stream) {
    return prefill<true, true>(x, ldx, n, C, static_w, dyn, ld_dyn, state, y, ldy, stream);
}

}  // namespace mutant

}  // namespace cere::inkling
