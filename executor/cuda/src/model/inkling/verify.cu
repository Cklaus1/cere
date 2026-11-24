// Inkling 2 Mini S9 phase B: verify.h. Compiled with -fmad=false.
#include "model/inkling/verify.h"

#include "inkling/conv_ref.h"
#include "kernels/inkling/pdl.cuh"

namespace cere::inkling {
namespace {

constexpr int kThreads = 128;
constexpr int K = ref::kConvTaps;
constexpr int R = ref::kConvStateRows;
constexpr int64_t kRingElems = kRingSlots * kKvRow;
static_assert(kConvStateOffset[1] == R * kKvRow && kConvStateOffset[2] == 2 * R * kKvRow &&
                  kConvStateOffset[3] == 2 * R * kKvRow + R * kHidden,
              "conv_commit_kernel's offsets mirror forward.h kConvStateOffset");

// Row `blockIdx.y`: its window is the n = idx + 1 prefill window of its run's last row (conv_prefill_kernel's): x
// rows of the run before it, then the state; the output is conv_decode_kernel's point. No state is written.
template <bool Dynamic>
__global__ void __launch_bounds__(kThreads)
    conv_verify_kernel(const uint16_t *x, int64_t ldx, int C, const uint16_t *w, const uint16_t *dyn, int64_t ld_dyn,
                       float *const *states, const int32_t *idx, uint16_t *y, int64_t ldy, uint16_t *stage,
                       bool state_only) {
    pdl::trigger();
    const int c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= C) return;
    float wf[K];  // static: w[c, j]; dynamic: the static part w[lag, c] of tap lag
    for (int j = 0; j < K; ++j) wf[j] = ref::decode_bf16(Dynamic ? w[int64_t(j) * C + c] : w[int64_t(c) * K + j]);
    pdl::wait();
    const int64_t row = blockIdx.y;
    const float *state = states[row];
    const int64_t j = state_only ? 0 : idx[row];
    float win[K];
    for (int i = 0; i < R; ++i) {
        const int64_t r = j - R + i;  // index in the run
        win[i] = r >= 0 ? ref::decode_bf16(x[(row - j + r) * ldx + c]) : state[(R + r) * C + c];
    }
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
    stage[row * C + c] = x[row * ldx + c];
}

// Block (channel block, visit * 4 + conv, session): conv_ref.h advance_state over the kept rows of the stage.
__global__ void __launch_bounds__(kThreads) conv_commit_kernel(const uint16_t *stage, const CommitSeq *seqs) {
    pdl::enter();
    const int vc = blockIdx.y, v = vc / 4, cv = vc % 4;
    const int C = cv < 2 ? int(kKvRow) : int(kHidden);
    const int c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= C) return;
    const CommitSeq s = seqs[blockIdx.z];
    const uint16_t *x = stage + int64_t(vc) * kConvStageRegion + s.first * C;
    // kConvStateOffset[cv] (forward.h; a host array): K, V of kKvRow channels, then the two kHidden convs.
    const int64_t offset = cv < 2 ? int64_t(cv) * R * kKvRow : 2 * R * kKvRow + int64_t(cv - 2) * R * kHidden;
    float *state = s.conv + int64_t(v) * kConvVisitFloats + offset;
    float old[R], nu[R];
    for (int j = 0; j < R; ++j) old[j] = state[j * C + c];
    for (int j = 0; j < R; ++j) {
        const int64_t r = s.keep - R + j;
        nu[j] = r >= 0 ? ref::decode_bf16(x[r * C + c]) : old[R + r];
    }
    for (int j = 0; j < R; ++j) state[j * C + c] = nu[j];
}

// Block (kept row i, session, sliding visit * 2 + K/V): one 1,024-element row into its ring slot.
__global__ void __launch_bounds__(kThreads) ring_commit_kernel(const uint4 *stage, const CommitSeq *seqs) {
    pdl::enter();
    const CommitSeq s = seqs[blockIdx.y];
    const int64_t i = blockIdx.x;
    if (i >= s.keep) return;
    const int64_t kv = blockIdx.z;  // sliding_index * 2 + (0 K, 1 V)
    constexpr int64_t kRow16 = kKvRow / 8;
    const uint4 *src = stage + (kv * kVerifyRows + s.first + i) * kRow16;
    uint4 *dst = reinterpret_cast<uint4 *>(s.ring + kv * kRingElems + ((s.pos0 + i) % kRingSlots) * kKvRow);
    for (int64_t e = threadIdx.x; e < kRow16; e += kThreads) dst[e] = src[e];
}

__global__ void __launch_bounds__(kThreads)
    gather_kernel(const uint4 *src, int64_t ld_src, int64_t width, const CommitSeq *seqs, uint4 *dst, int64_t ld_dst) {
    pdl::enter();
    const CommitSeq s = seqs[blockIdx.y];
    const int64_t i = blockIdx.x;
    if (i >= s.keep) return;
    const uint4 *a = src + (s.first + i) * ld_src;
    uint4 *b = dst + (s.out + i) * ld_dst;
    for (int64_t e = threadIdx.x; e < width; e += kThreads) b[e] = a[e];
}

const uint16_t *u16(const bf16 *p) { return reinterpret_cast<const uint16_t *>(p); }

bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

template <bool Dynamic>
cudaError_t verify(const bf16 *x, int64_t ldx, int64_t M, int C, const bf16 *w, const bf16 *dyn, int64_t ld_dyn,
                   float *const *states, const int32_t *idx, bf16 *y, int64_t ldy, bf16 *stage, bool state_only,
                   cudaStream_t stream) {
    if (x == nullptr || y == nullptr || x == y || stage == nullptr || w == nullptr || states == nullptr ||
        idx == nullptr || C < 1 || M < 1 || M > kVerifyRows || ldx < C || ldy < C) {
        return cudaErrorInvalidValue;
    }
    if (Dynamic && (C % ref::kConvHeadDim != 0 || dyn == nullptr || ld_dyn < (C / ref::kConvHeadDim) * K)) {
        return cudaErrorInvalidValue;
    }
    const unsigned cblocks = unsigned((C + kThreads - 1) / kThreads);
    return pdl::launch(conv_verify_kernel<Dynamic>, dim3(cblocks, unsigned(M)), dim3(kThreads), 0, stream, u16(x),
                       ldx, C, u16(w), u16(dyn), ld_dyn, states, idx, reinterpret_cast<uint16_t *>(y), ldy,
                       reinterpret_cast<uint16_t *>(stage), state_only);
}

}  // namespace

cudaError_t conv_static_verify(const bf16 *x, int64_t ldx, int64_t M, int C, const bf16 *w, float *const *states,
                               const int32_t *idx, bf16 *y, int64_t ldy, bf16 *stage, bool state_only,
                               cudaStream_t stream) {
    return verify<false>(x, ldx, M, C, w, nullptr, 0, states, idx, y, ldy, stage, state_only, stream);
}

cudaError_t conv_dynamic_verify(const bf16 *x, int64_t ldx, int64_t M, int C, const bf16 *static_w, const bf16 *dyn,
                                int64_t ld_dyn, float *const *states, const int32_t *idx, bf16 *y, int64_t ldy,
                                bf16 *stage, bool state_only, cudaStream_t stream) {
    return verify<true>(x, ldx, M, C, static_w, dyn, ld_dyn, states, idx, y, ldy, stage, state_only, stream);
}

cudaError_t conv_commit(const bf16 *stage, const CommitSeq *seqs, int64_t S, cudaStream_t stream) {
    if (stage == nullptr || seqs == nullptr || S < 1 || S > kVerifyRows) return cudaErrorInvalidValue;
    const dim3 grid(unsigned((kHidden + kThreads - 1) / kThreads), unsigned(plan::kVisits * 4), unsigned(S));
    return pdl::launch(conv_commit_kernel, grid, dim3(kThreads), 0, stream, u16(stage), seqs);
}

cudaError_t ring_commit(const bf16 *kv_stage, const CommitSeq *seqs, int64_t S, int64_t max_keep,
                        cudaStream_t stream) {
    if (kv_stage == nullptr || !aligned16(kv_stage) || seqs == nullptr || S < 1 || S > kVerifyRows || max_keep < 1 ||
        max_keep > kMaxRunRows) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid(unsigned(max_keep), unsigned(S), unsigned(plan::kSlidingVisits * 2));
    return pdl::launch(ring_commit_kernel, grid, dim3(kThreads), 0, stream, reinterpret_cast<const uint4 *>(kv_stage),
                       seqs);
}

cudaError_t gather_rows(const bf16 *src, int64_t ld_src, int64_t width, const CommitSeq *seqs, int64_t S,
                        int64_t max_keep, bf16 *dst, int64_t ld_dst, cudaStream_t stream) {
    if (src == nullptr || dst == nullptr || !aligned16(src) || !aligned16(dst) || seqs == nullptr || S < 1 ||
        S > kVerifyRows || max_keep < 1 || max_keep > kMaxRunRows || width < 8 || width % 8 != 0 || ld_src % 8 != 0 ||
        ld_dst % 8 != 0 || ld_src < width || ld_dst < width) {
        return cudaErrorInvalidValue;
    }
    return pdl::launch(gather_kernel, dim3(unsigned(max_keep), unsigned(S)), dim3(kThreads), 0, stream,
                       reinterpret_cast<const uint4 *>(src), ld_src / 8, width / 8, seqs,
                       reinterpret_cast<uint4 *>(dst), ld_dst / 8);
}

}  // namespace cere::inkling
