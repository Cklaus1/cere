// Inkling 2 Mini (inkling design note 3.1, G1): the loop message gate; contract in msg.h, the exact order in
// executor/ref/inkling/msg_ref.h (and norm_ref.h for msg_gate_norm's weighted norm). Compiled with -fmad=false.
#include "kernels/inkling/msg.h"

#include <climits>

#include <cooperative_groups.h>

#include "kernels/inkling/mix_common.cuh"
#include "kernels/inkling/mix_mutants.h"
#include "kernels/inkling/pdl.cuh"
#include "inkling/msg_ref.h"

namespace cere::inkling {

namespace {

namespace cg = cooperative_groups;

using mix_detail::kPerThread;
using mix_detail::kRowThreads;

constexpr int kRankBatch = 8;  // down-projection outputs reduced per halving tree pass
// Decode-sized calls (rows <= kSplitRows) spread a row's 64 down-projection outputs over kGateRank / kDecodeRanks
// blocks; larger calls keep one block per row (the norms are recomputed per block).
constexpr int kSplitRows = 64;
constexpr int kDecodeRanks = 4;
constexpr int kMixThreads = ref::kGateRank;  // one channel per thread; the block's threads stage d[0..63]
static_assert(ref::kHidden % kMixThreads == 0, "2304 channels in whole mix blocks");
// msg_gate_norm's mix: a cluster of kNormCluster blocks per (row, gate), kNormChannels channels per block.
constexpr int kNormCluster = 8;
constexpr int kNormChannels = ref::kHidden / kNormCluster;  // 288
static_assert(kNormChannels * kNormCluster == ref::kHidden && kNormChannels % 32 == 0 &&
                  kNormChannels >= kRowThreads && kNormChannels >= ref::kGateRank,
              "whole warps; the norm's 256 lanes and the 64 d values staged by one block");

bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// The gates of one launch (gate i = blockIdx.z of the down kernel, blockIdx.y of the mix kernels); ws holds gate i's
// down projection at ws + i rows 64.
struct Gates {
    const float *h[kMsgMaxGates];
    const float *msg[kMsgMaxGates];
    const uint16_t *down[kMsgMaxGates];
    const uint16_t *forget[kMsgMaxGates];
    const uint16_t *gain[kMsgMaxGates];
    const uint16_t *norm_w[kMsgMaxGates];  // msg_gate_norm
    void *out[kMsgMaxGates];               // FP32 (msg_gate) or BF16 (msg_gate_norm)
    int64_t ldh, ldm, ldo, rows;
    float eps;
    float *ws;
};

__device__ __forceinline__ float *gate_ws(const Gates &a, int i) {
    return a.ws + int64_t(i) * a.rows * ref::kGateRank;
}

// 1-3. grid (rows, kGateRank / kRanks, gates): block (row, b, i) computes the two unweighted norms of gate i's row,
// the BF16 concat g (lane t of the 4608-wide dot holds i = t + 256 jj, jj < 18: the h half (jj < 9) then the msg
// half) and the down-projection outputs d[r] = bf16(reduce_dot<256>(down[r, :], g, 4608)) for r in [b kRanks,
// (b + 1) kRanks), kBatch per halving tree pass, into the gate's ws [rows, 64]. A one-pass block (kRanks <=
// kRankBatch: decode) loads its weights before the PDL wait.
template <int kRanks>
__global__ void __launch_bounds__(kRowThreads) msg_down_kernel(const Gates a) {
    constexpr int kBatch = kRanks < kRankBatch ? kRanks : kRankBatch;
    constexpr bool kOnePass = kRanks == kBatch;
    static_assert(kRanks % kBatch == 0 && ref::kGateRank % kRanks == 0, "whole tree passes, whole rank groups");
    pdl::trigger();
    __shared__ float red[kBatch][kRowThreads];
    const int64_t row = blockIdx.x;
    const int r_first = int(blockIdx.y) * kRanks;
    const int gi = int(blockIdx.z);
    const int t = threadIdx.x;
    const uint16_t *down = a.down[gi];
    [[maybe_unused]] uint16_t w1[kBatch][2 * kPerThread];
    if constexpr (kOnePass) {
#pragma unroll
        for (int k = 0; k < kBatch; ++k)
#pragma unroll
            for (int jj = 0; jj < 2 * kPerThread; ++jj)
                w1[k][jj] = down[int64_t(r_first + k) * ref::kGateIn + t + kRowThreads * jj];
    }
    pdl::wait();
    float g[2 * kPerThread];
    {
        const float *h = a.h[gi], *msg = a.msg[gi];
        float hv[kPerThread], mv[kPerThread];
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            hv[j] = h[row * a.ldh + t + kRowThreads * j];
            mv[j] = msg[row * a.ldm + t + kRowThreads * j];
        }
        const float rh = mix_detail::row_rms_inv(hv, a.eps, red[0]);
        const float rm = mix_detail::row_rms_inv(mv, a.eps, red[0]);
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            g[j] = ref::decode_bf16(ref::encode_bf16(ref::mul(hv[j], rh)));
            g[kPerThread + j] = ref::decode_bf16(ref::encode_bf16(ref::mul(mv[j], rm)));
        }
    }
    float *dout = gate_ws(a, gi);
    for (int r0 = r_first; r0 < r_first + kRanks; r0 += kBatch) {
        float acc[kBatch];
#pragma unroll
        for (int k = 0; k < kBatch; ++k) {
            [[maybe_unused]] const uint16_t *wr = down + int64_t(r0 + k) * ref::kGateIn + t;
            acc[k] = 0.0f;
#pragma unroll
            for (int jj = 0; jj < 2 * kPerThread; ++jj) {
                uint16_t wk;
                if constexpr (kOnePass) wk = w1[k][jj];
                else wk = wr[kRowThreads * jj];
                acc[k] = ref::fma(ref::decode_bf16(wk), g[jj], acc[k]);
            }
        }
        float out[kBatch];
        mix_detail::block_tree256_batch<kBatch>(acc, red, out);
        if (t == 0) {
#pragma unroll
            for (int k = 0; k < kBatch; ++k)
                dout[row * ref::kGateRank + r0 + k] = ref::decode_bf16(ref::encode_bf16(out[k]));
        }
    }
}

// 4. Channel c's forget / message gates from d (smem [64], 16-byte aligned): the two rank-64 dots (reduce_dot<64>:
// lane j = fma(w[j], d[j], 0), then the halving tree of ref::reduce_tree_ unrolled in registers), tanh. wf / wg: the
// channel's forget and gain rows, 8 BF16 per uint4.
constexpr int kVec = ref::kGateRank / 8;

__device__ __forceinline__ void load_gate_rows(const uint16_t *forget, const uint16_t *gain, int c, uint4 (&wf)[kVec],
                                               uint4 (&wg)[kVec]) {
    const uint4 *fr = reinterpret_cast<const uint4 *>(forget + int64_t(c) * ref::kGateRank);
    const uint4 *gr = reinterpret_cast<const uint4 *>(gain + int64_t(c) * ref::kGateRank);
#pragma unroll
    for (int v = 0; v < kVec; ++v) wf[v] = fr[v], wg[v] = gr[v];
}

__device__ __forceinline__ float gate_dot(const uint4 (&w)[kVec], const float *d) {
    float lane[ref::kGateRank];
    const uint16_t *wh = reinterpret_cast<const uint16_t *>(w);
    const float4 *d4 = reinterpret_cast<const float4 *>(d);
#pragma unroll
    for (int q = 0; q < ref::kGateRank / 4; ++q) {
        const float4 dv = d4[q];
        lane[4 * q] = ref::fma(ref::decode_bf16(wh[4 * q]), dv.x, 0.0f);
        lane[4 * q + 1] = ref::fma(ref::decode_bf16(wh[4 * q + 1]), dv.y, 0.0f);
        lane[4 * q + 2] = ref::fma(ref::decode_bf16(wh[4 * q + 2]), dv.z, 0.0f);
        lane[4 * q + 3] = ref::fma(ref::decode_bf16(wh[4 * q + 3]), dv.w, 0.0f);
    }
#pragma unroll
    for (int off = ref::kGateRank / 2; off >= 1; off /= 2) {
#pragma unroll
        for (int i = 0; i < off; ++i) lane[i] = ref::add(lane[i], lane[i + off]);
    }
    return ref::tanh(lane[0]);
}

// 5. The mix of channel c.
__device__ __forceinline__ float gate_mix(float f, float m, float hv, float mv) {
    return ref::add(ref::mul(ref::sub(1.0f, f), hv), ref::mul(m, mv));
}

// 4-5. grid (rows, 2304 / 64, gates): thread c of the row: the gates and the mix into FP32 out; the weight rows load
// before the PDL wait. Swap: the fail-first mutant (mix_mutants.h), forget and message gates exchanged.
template <bool Swap>
__global__ void __launch_bounds__(kMixThreads) msg_mix_kernel(const Gates a) {
    pdl::trigger();
    __shared__ __align__(16) float d[ref::kGateRank];
    const int64_t row = blockIdx.x;
    const int gi = int(blockIdx.y);
    const int t = threadIdx.x;
    const int c = int(blockIdx.z) * kMixThreads + t;
    uint4 wf[kVec], wg[kVec];
    load_gate_rows(a.forget[gi], a.gain[gi], c, wf, wg);
    pdl::wait();
    const float hv = a.h[gi][row * a.ldh + c], mv = a.msg[gi][row * a.ldm + c];
    d[t] = gate_ws(a, gi)[row * ref::kGateRank + t];
    __syncthreads();
    const float f = gate_dot(wf, d), m = gate_dot(wg, d);
    const float fg = Swap ? m : f, mg = Swap ? f : m;
    static_cast<float *>(a.out[gi])[row * a.ldo + c] = gate_mix(fg, mg, hv, mv);
}

// 4-5 + the weighted RMSNorm (norm_ref.h) into BF16: grid (kNormCluster rows, gates), a cluster per (row, gate);
// block rank b mixes channels [288 b, 288 b + 288) into shared u, then thread t < 256 reads the row's elements t +
// 256 j (j < 9) from the cluster's u (distributed shared memory) for the sum of squares in norm_ref.h's lane order,
// warp 0 runs the halving tree (warp_tree256), and every block scales its own channels: y = bf16(mul(w, mul(u,
// r))). Each block computes the same r. The gate and norm weights load before the PDL wait.
__global__ void __cluster_dims__(kNormCluster, 1, 1) __launch_bounds__(kNormChannels)
    msg_mix_norm_kernel(const Gates a) {
    pdl::trigger();
    __shared__ __align__(16) float d[ref::kGateRank];
    __shared__ float u[kNormChannels];
    __shared__ float red[kRowThreads];
    __shared__ float ss;
    cg::cluster_group cluster = cg::this_cluster();
    const int64_t row = blockIdx.x / kNormCluster;
    const int gi = int(blockIdx.y);
    const int t = threadIdx.x;
    const int c = int(cluster.block_rank()) * kNormChannels + t;
    uint4 wf[kVec], wg[kVec];
    load_gate_rows(a.forget[gi], a.gain[gi], c, wf, wg);
    const uint16_t nw = a.norm_w[gi][c];
    pdl::wait();
    const float hv = a.h[gi][row * a.ldh + c], mv = a.msg[gi][row * a.ldm + c];
    if (t < ref::kGateRank) d[t] = gate_ws(a, gi)[row * ref::kGateRank + t];
    __syncthreads();
    const float uv = gate_mix(gate_dot(wf, d), gate_dot(wg, d), hv, mv);
    u[t] = uv;
    cluster.sync();  // every block's u is written
    if (t < kRowThreads) {
        float x[kPerThread];
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            const int i = t + kRowThreads * j;
            x[j] = cluster.map_shared_rank(u, i / kNormChannels)[i % kNormChannels];
        }
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) acc = ref::fma(x[j], x[j], acc);
        red[t] = acc;
    }
    // This block's reads of the other blocks' u are done; they may exit once every block has arrived.
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    __syncthreads();
    if (t < 32) {
        const float v = mix_detail::warp_tree256(red, t);
        if (t == 0) ss = v;
    }
    __syncthreads();
    const float r = ref::rsqrt(ref::add(ref::div(ss, float(ref::kHidden)), a.eps));
    static_cast<uint16_t *>(a.out[gi])[row * a.ldo + c] =
        ref::encode_bf16(ref::mul(ref::decode_bf16(nw), ref::mul(uv, r)));
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");  // our u stays until every block has read it
}

cudaError_t launch_down(const Gates &a, int count, cudaStream_t stream) {
    if (a.rows <= kSplitRows)
        return pdl::launch(msg_down_kernel<kDecodeRanks>,
                           dim3(unsigned(a.rows), ref::kGateRank / kDecodeRanks, unsigned(count)), dim3(kRowThreads),
                           0, stream, a);
    return pdl::launch(msg_down_kernel<ref::kGateRank>, dim3(unsigned(a.rows), 1, unsigned(count)), dim3(kRowThreads),
                       0, stream, a);
}

template <bool Swap>
cudaError_t launch(const float *h, int64_t ldh, const float *msg, int64_t ldm, int64_t rows, const bf16 *down,
                   const bf16 *forget, const bf16 *gain, float eps, float *out, int64_t ldo, float *ws,
                   cudaStream_t stream) {
    if (h == nullptr || msg == nullptr || down == nullptr || forget == nullptr || gain == nullptr || out == nullptr ||
        ws == nullptr || rows < 1 || rows > INT_MAX || ldh < ref::kHidden || ldm < ref::kHidden ||
        ldo < ref::kHidden || !(eps >= 0.0f) || (out == h && ldo != ldh) || (out == msg && ldo != ldm) ||
        ws == h || ws == msg || ws == out || !aligned16(forget) || !aligned16(gain))
        return cudaErrorInvalidValue;
    Gates a{};
    a.h[0] = h;
    a.msg[0] = msg;
    a.down[0] = reinterpret_cast<const uint16_t *>(down);
    a.forget[0] = reinterpret_cast<const uint16_t *>(forget);
    a.gain[0] = reinterpret_cast<const uint16_t *>(gain);
    a.out[0] = out;
    a.ldh = ldh, a.ldm = ldm, a.ldo = ldo, a.rows = rows, a.eps = eps, a.ws = ws;
    const cudaError_t e = launch_down(a, 1, stream);
    if (e != cudaSuccess) return e;
    return pdl::launch(msg_mix_kernel<Swap>, dim3(unsigned(rows), 1, ref::kHidden / kMixThreads), dim3(kMixThreads), 0,
                       stream, a);
}

}  // namespace

cudaError_t msg_gate(const float *h, int64_t ldh, const float *msg, int64_t ldm, int64_t rows, const bf16 *down,
                     const bf16 *forget, const bf16 *gain, float eps, float *out, int64_t ldo, float *ws,
                     cudaStream_t stream) {
    return launch<false>(h, ldh, msg, ldm, rows, down, forget, gain, eps, out, ldo, ws, stream);
}

cudaError_t msg_gate_norm(const MsgGateNorm *gates, int count, int64_t ldh, int64_t ldm, int64_t ldy, int64_t rows,
                          float eps, float *ws, cudaStream_t stream) {
    if (gates == nullptr || count < 1 || count > kMsgMaxGates || ws == nullptr || rows < 1 ||
        rows > INT_MAX / kNormCluster || ldh < ref::kHidden || ldm < ref::kHidden || ldy < ref::kHidden ||
        !(eps >= 0.0f))
        return cudaErrorInvalidValue;
    Gates a{};
    for (int i = 0; i < count; ++i) {
        const MsgGateNorm &g = gates[i];
        if (g.h == nullptr || g.msg == nullptr || g.down == nullptr || g.forget == nullptr || g.gain == nullptr ||
            g.norm_w == nullptr || g.y == nullptr || ws == g.h || ws == g.msg || !aligned16(g.forget) ||
            !aligned16(g.gain))
            return cudaErrorInvalidValue;
        a.h[i] = g.h;
        a.msg[i] = g.msg;
        a.down[i] = reinterpret_cast<const uint16_t *>(g.down);
        a.forget[i] = reinterpret_cast<const uint16_t *>(g.forget);
        a.gain[i] = reinterpret_cast<const uint16_t *>(g.gain);
        a.norm_w[i] = reinterpret_cast<const uint16_t *>(g.norm_w);
        a.out[i] = g.y;
    }
    a.ldh = ldh, a.ldm = ldm, a.ldo = ldy, a.rows = rows, a.eps = eps, a.ws = ws;
    const cudaError_t e = launch_down(a, count, stream);
    if (e != cudaSuccess) return e;
    return pdl::launch(msg_mix_norm_kernel, dim3(unsigned(rows * kNormCluster), unsigned(count)), dim3(kNormChannels),
                       0, stream, a);
}

namespace mutant {

cudaError_t msg_gate_swapped(const float *h, int64_t ldh, const float *msg, int64_t ldm, int64_t rows,
                             const bf16 *down, const bf16 *forget, const bf16 *gain, float eps, float *out,
                             int64_t ldo, float *ws, cudaStream_t stream) {
    return launch<true>(h, ldh, msg, ldm, rows, down, forget, gain, eps, out, ldo, ws, stream);
}

}  // namespace mutant

}  // namespace cere::inkling
