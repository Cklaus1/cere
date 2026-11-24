// K2 small-M path: contract and reduction order in k2_small_m.h.
#include "kernels/k2_small_m.h"

#include "device/context_generation.h"
#include "kernels/k2_mxfp8.cuh"
#include "kernels/k3_hc_math.cuh"

#include <atomic>
#include <cooperative_groups.h>
#include <mutex>
#include <type_traits>

namespace cere::kernels::k2s {
namespace {

namespace cg = cooperative_groups;

constexpr uint32_t kStepK = 32;        // k per warp step: one 16-byte chunk per lane and row, two MMAs per n-tile
constexpr uint32_t kGroup = 16;        // slices per group (k2_small_m.h): the warps of one CTA when there are more
constexpr uint32_t kStepsPerWarp = 4;  // split_warps up to one group: about this many steps per slice
constexpr uint32_t kOneGroupSteps = 12;  // split_warps: one group (one CTA) while a slice holds at most this many steps
                                        // (k <= 6,144)
constexpr uint32_t kSliceSteps = 5;     // split_warps past one group: at most this many steps per slice (k <= 20,480;
                                        // hc_down's k = 10,240 runs 4 groups) [REVISED: P11R-14]
constexpr uint32_t kUnroll = 5;        // steps whose loads are issued before their MMAs (a router or shared expert
                                       // slice, 5 steps at k = 2,560, loads in one round)
constexpr uint32_t kMaxCluster = 8;    // the portable thread block cluster size: at most 8 groups

// What a launch stores per output element (small_m_kernel's first template argument, an integer so the profile's
// demangled kernel names read `small_m_kernel<O, TN, Hi>`, D12): y in BF16 or F32, or K3's activation of it.
constexpr uint32_t kOutBf16 = 0;
constexpr uint32_t kOutF32 = 1;
constexpr uint32_t kOutHcAct = 2;  // gemm_hc_act

struct LaunchSegment {
    const uint4 *w;
    void *y;
    uint32_t n;
    uint32_t tile_begin;  // first column tile of this segment
};

struct LaunchSegments {
    LaunchSegment s[kMaxSegments];
    uint32_t count;
};

// gemm's and gemm_hc_act's early launches (k2_small_m.h early weights): the same segments, a distinct parameter type
// so the early kernel is an overload of small_m_kernel (one demangled name, D12) whose body issues its weights before
// the wait.
struct EarlyLaunchSegments : LaunchSegments {};

// gemm_mxfp8's segments: the E4M3 rows and their scale bytes instead of BF16 rows (k2_small_m.h Mxfp8Segment).
struct Mxfp8LaunchSegment {
    const uint2 *q;  // 8 E4M3 bytes per chunk: chunk c of a row is k = 8 c .. 8 c + 7, as a BF16 row's uint4
    const uint8_t *scales;
    void *y;
    uint32_t n;
    uint32_t tile_begin;
};

struct Mxfp8LaunchSegments {
    Mxfp8LaunchSegment s[kMaxSegments];
    uint32_t count;
};

// The layout of one launch (none of it changes an output bit; see k2_small_m.h): tile_n columns per column tile
// (16 or 8); a tile's split_warps(k) slices run on `cluster` CTAs (one per group) of slices / cluster warps.
struct Layout {
    uint32_t tile_n;
    uint32_t cluster;
    uint32_t tiles;
    uint32_t warps;  // per CTA
};

// Weights are read once per launch: no L1 allocation, L2 256-byte prefetch.
__device__ __forceinline__ uint4 load_stream(const uint4 *p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint2 load_stream8(const uint2 *p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

// One lane's weight operand of one tile column per step, by weight format. BF16: the 16 bytes at k = 32 s + 8 t are
// the MMA operand as loaded. MXFP8 (gemm_mxfp8): the 8 E4M3 bytes at the same k and the row's scale byte of step s,
// turned into that BF16 operand (k2_mxfp8.cuh, exact) after the loads, before the MMAs. A column past n loads
// nothing: none(), whose operand is zero in both formats.
struct Bf16Row {
    using Raw = uint4;
    const uint4 *w;  // the row's chunk t
    __device__ __forceinline__ Raw load(size_t off, uint32_t) const { return load_stream(w + off); }
    __device__ __forceinline__ static Raw none() { return make_uint4(0, 0, 0, 0); }
    __device__ __forceinline__ static uint4 operand(const Raw &r) { return r; }
};

struct Mxfp8Row {
    struct Raw {
        uint2 q;
        uint32_t s;
    };
    const uint2 *q;         // the row's chunk t
    const uint8_t *scales;  // the row's k / 32 scale bytes (any alignment); L1-cached: a slice's steps share lines
    __device__ __forceinline__ Raw load(size_t off, uint32_t s) const {
        return Raw{load_stream8(q + off), __ldg(scales + s)};
    }
    __device__ __forceinline__ static Raw none() { return Raw{make_uint2(0, 0), 0}; }
    __device__ __forceinline__ static uint4 operand(const Raw &r) { return k2m::bf16x8(r.q, k2m::exponent_add(r.s)); }
};

// Column j's row of segment `seg` at lane chunk t (chunks = k / 8 per row).
__device__ __forceinline__ Bf16Row row_of(const LaunchSegment &seg, size_t j, uint32_t chunks, uint32_t t) {
    return Bf16Row{seg.w + j * chunks + t};
}

__device__ __forceinline__ Mxfp8Row row_of(const Mxfp8LaunchSegment &seg, size_t j, uint32_t chunks, uint32_t t) {
    return Mxfp8Row{seg.q + j * chunks + t, seg.scales + j * (chunks / 4)};
}

// Programmatic dependent launch (k2_small_m.h, P11R-21): lets the next kernel of the stream launch (it waits for this
// grid in its own dependency wait), then waits for the previous kernel's writes. A CTA reads nothing (x or w) before
// this, so a kernel test that writes the weights right before the launch is still read correctly. [REVISED: P11R-21
// S3: an L2 bulk prefetch of the tile's weight rows before the wait cost 0.5-0.9 ms per step in situ; removed]
__device__ __forceinline__ void launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

__device__ __forceinline__ void grid_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

__device__ __forceinline__ void dependency_wait() {
    launch_dependents();
    grid_wait();
}

// d += A · B for one m16n8k16 BF16 tile with FP32 accumulation (PTX fragment layout: a0/a2 row g, a1/a3 row g + 8;
// b0/b1 column g; d0/d1 row g, d2/d3 row g + 8, columns 2t, 2t + 1).
__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One warp step at k = 32 s: lane (g = lane / 4, t = lane % 4) holds x rows g and g + 8 and w column g of each of the
// Q n-tiles at k = 32 s + 8 t .. + 7 (16 bytes each). MMA h (0, 1) maps the tile's k pairs (2t, 2t + 1) and
// (2t + 8, 2t + 9) to k = 32 s + 8 t + 4 h + (0, 1) and (2, 3), the same map for x and w.
template <uint32_t Q>
__device__ __forceinline__ void step(float (&acc)[Q][4], const uint4 &x0, const uint4 &x1, const uint4 (&w)[Q]) {
#pragma unroll
    for (uint32_t q = 0; q < Q; ++q) {
        mma(acc[q], x0.x, x1.x, x0.y, x1.y, w[q].x, w[q].y);
        mma(acc[q], x0.z, x1.z, x0.w, x1.w, w[q].z, w[q].w);
    }
}

// One warp's slice: acc (zero on entry) += x · w over steps slice, slice + slices, ... < steps in increasing order,
// kUnroll steps' loads before their MMAs. Rows past m and columns past n contribute zeros (no load). Hi: rows 8..15
// may be live (m > 8); without it they are the zero registers they would have loaded, so no output bit depends on Hi.
// W: the weight format's row (Bf16Row, Mxfp8Row); its operands are the BF16 bits, so the MMAs are the same.
// Pre (early weights, BF16 rows): the first round's weights (s0 = slice) are read from `pre`, the warp's shared memory
// slots that first_round filled with cp.async (slot (u, q) at pre[(u * Q + q) * 32 + lane]); this thread's own
// copies, so waiting for its own cp.async groups is enough.
template <uint32_t Q, bool Hi, class W, bool Pre = false>
__device__ __forceinline__ void slice_mma(float (&acc)[Q][4], const uint4 *x, uint32_t m, uint32_t chunks,
                                          const W (&wc)[Q], const bool (&col)[Q], uint32_t slice, uint32_t slices,
                                          const uint4 *pre = nullptr) {
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t g = lane / 4;
    const uint32_t t = lane % 4;
    const uint32_t steps = chunks / 4;  // chunks = k / 8; 4 chunks per step
    const bool row0 = g < m;
    const bool row1 = Hi && g + 8 < m;
    const uint4 *xr0 = x + static_cast<size_t>(row0 ? g : 0) * chunks + t;
    const uint4 *xr1 = x + static_cast<size_t>(row1 ? g + 8 : 0) * chunks + t;
    const uint4 zero = make_uint4(0, 0, 0, 0);
    for (uint32_t s0 = slice; s0 < steps; s0 += kUnroll * slices) {
        uint4 xa[kUnroll], xb[kUnroll];
        typename W::Raw wv[kUnroll][Q];
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
            const uint32_t s = s0 + u * slices;
            const bool in = s < steps;
            const size_t off = static_cast<size_t>(s) * 4;
            xa[u] = in && row0 ? __ldg(xr0 + off) : zero;
            if constexpr (Hi) {
                xb[u] = in && row1 ? __ldg(xr1 + off) : zero;
            } else {
                xb[u] = zero;
            }
#pragma unroll
            for (uint32_t q = 0; q < Q; ++q) {
                if constexpr (Pre) {
                    wv[u][q] = !(in && col[q]) ? W::none()
                               : s0 == slice   ? pre[(u * Q + q) * 32 + lane]
                                               : wc[q].load(off, s);
                } else {
                    wv[u][q] = in && col[q] ? wc[q].load(off, s) : W::none();
                }
            }
        }
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
            if (s0 + u * slices < steps) {
                uint4 wo[Q];
#pragma unroll
                for (uint32_t q = 0; q < Q; ++q) {
                    wo[q] = W::operand(wv[u][q]);
                }
                step<Q>(acc, xa[u], xb[u], wo);
            }
        }
    }
}

// Early weights (k2_small_m.h): slice_mma's first round of BF16 weight loads (s0 = slice) as cp.async copies into the
// warp's shared memory slots `pre` (kUnroll * Q * 32 uint4), issued as one group; columns past n and steps past the
// slice copy nothing (slice_mma uses W::none() there, as it would have).
template <uint32_t Q>
__device__ __forceinline__ void first_round(uint4 *pre, uint32_t chunks, const Bf16Row (&wc)[Q],
                                            const bool (&col)[Q], uint32_t slice, uint32_t slices) {
    const uint32_t steps = chunks / 4;
    const uint32_t lane = threadIdx.x % 32;
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
        const uint32_t s = slice + u * slices;
        const size_t off = static_cast<size_t>(s) * 4;
#pragma unroll
        for (uint32_t q = 0; q < Q; ++q) {
            if (s < steps && col[q]) {
                const auto dst = static_cast<uint32_t>(__cvta_generic_to_shared(pre + (u * Q + q) * 32 + lane));
                asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(dst), "l"(wc[q].w + off)
                             : "memory");
            }
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// This thread's first_round copies have landed in shared memory.
__device__ __forceinline__ void first_round_wait() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }

// Dynamic shared memory of an early launch: first_round's slots of every warp of the CTA.
constexpr size_t early_bytes(uint32_t warps, uint32_t tile_n) {
    return size_t{warps} * kUnroll * (tile_n / 8) * 32 * sizeof(uint4);
}

// Each (row, column) partial of a warp's slice sits in exactly one of its lanes: partial[r][c] for the tile's TN
// columns.
template <uint32_t Q, uint32_t TN>
__device__ __forceinline__ void put_partial(float (&partial)[kMaxM][TN], const float (&acc)[Q][4]) {
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t g = lane / 4;
    const uint32_t t = lane % 4;
#pragma unroll
    for (uint32_t q = 0; q < Q; ++q) {
        const uint32_t c = 8 * q + 2 * t;
        partial[g][c] = acc[q][0];
        partial[g][c + 1] = acc[q][1];
        partial[g + 8][c] = acc[q][2];
        partial[g + 8][c + 1] = acc[q][3];
    }
}

// CTA b of a launch with clusters of `cluster` CTAs: column tile b / cluster, cluster rank b % cluster (1-D clusters
// of consecutive CTAs). Warp v of rank c runs slice c * warps + v of the tile's slices = cluster * warps; rank c's
// warps are slice group c when cluster > 1 (warps = kGroup). Segs: LaunchSegments (BF16) or Mxfp8LaunchSegments.
// Early (BF16 rows): the warp's first round of weights is copied into shared memory (first_round, the launch's dynamic
// shared memory) between launch_dependents and the wait (k2_small_m.h).
template <uint32_t O, uint32_t TN, bool Hi, class Segs, bool Early = false>
__device__ __forceinline__ void small_m_body(const Segs &segs, const uint4 *__restrict__ x, uint32_t m,
                                             uint32_t chunks, uint32_t cluster) {
    constexpr uint32_t Q = TN / 8;  // m16n8k16 n-tiles per warp
    __shared__ float partial[kGroup][kMaxM][TN];
    __shared__ float group_sum[kMaxM][TN];
    const uint32_t tile = blockIdx.x / cluster;
    const uint32_t rank = blockIdx.x % cluster;
    auto seg = segs.s[0];
#pragma unroll
    for (uint32_t i = 1; i < kMaxSegments; ++i) {
        if (i < segs.count && tile >= segs.s[i].tile_begin) {
            seg = segs.s[i];
        }
    }
    void *y = seg.y;
    const uint32_t n = seg.n;
    const uint32_t begin = seg.tile_begin;
    const uint32_t col0 = (tile - begin) * TN;
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warps = blockDim.x / 32;
    const uint32_t slices = warps * cluster;  // split_warps(k)
    const uint32_t g = lane / 4;
    if constexpr (Early) {
        launch_dependents();
    } else {
        dependency_wait();
    }
    bool col[Q];
    decltype(row_of(seg, 0, chunks, 0)) wc[Q];
#pragma unroll
    for (uint32_t q = 0; q < Q; ++q) {
        const uint32_t j = col0 + 8 * q + g;
        col[q] = j < n;
        wc[q] = row_of(seg, col[q] ? j : 0, chunks, lane % 4);
    }
    float acc[Q][4] = {};
    if constexpr (Early) {
        extern __shared__ uint4 early_slots[];
        uint4 *pre = early_slots + static_cast<size_t>(warp) * kUnroll * Q * 32;
        first_round<Q>(pre, chunks, wc, col, rank * warps + warp, slices);
        grid_wait();
        first_round_wait();
        slice_mma<Q, Hi, Bf16Row, true>(acc, x, m, chunks, wc, col, rank * warps + warp, slices, pre);
    } else {
        slice_mma<Q, Hi>(acc, x, m, chunks, wc, col, rank * warps + warp, slices);
    }
    put_partial<Q, TN>(partial[warp], acc);
    // Each CTA sums its warps' slices in slice order (one group, or all the slices when cluster == 1); with clusters,
    // rank 0 sums the group sums in group order through distributed shared memory.
    const auto store = [&](uint32_t r, uint32_t j, float sum) {
        const size_t at = static_cast<size_t>(r) * n + col0 + j;
        if constexpr (O == kOutF32) {
            static_cast<float *>(y)[at] = sum;
        } else if constexpr (O == kOutHcAct) {
            static_cast<__nv_bfloat16 *>(y)[at] = hc::act(sum);
        } else {
            static_cast<__nv_bfloat16 *>(y)[at] = __float2bfloat16_rn(sum);
        }
    };
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < m * TN; i += blockDim.x) {
        const uint32_t r = i / TN;
        const uint32_t j = i % TN;
        if (col0 + j < n) {
            float sum = partial[0][r][j];
            for (uint32_t v = 1; v < warps; ++v) {
                sum += partial[v][r][j];
            }
            if (cluster == 1) {
                store(r, j, sum);
            } else {
                group_sum[r][j] = sum;
            }
        }
    }
    if (cluster == 1) {
        return;
    }
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();  // every rank's group sums are in its shared memory
    if (rank == 0) {
        for (uint32_t i = threadIdx.x; i < m * TN; i += blockDim.x) {
            const uint32_t r = i / TN;
            const uint32_t j = i % TN;
            if (col0 + j < n) {
                float v[kMaxCluster];
#pragma unroll
                for (uint32_t c = 0; c < kMaxCluster; ++c) {
                    v[c] = c < cluster ? *cl.map_shared_rank(&group_sum[r][j], c) : 0.0f;
                }
                float sum = v[0];
#pragma unroll
                for (uint32_t c = 1; c < kMaxCluster; ++c) {
                    if (c < cluster) {
                        sum += v[c];
                    }
                }
                store(r, j, sum);
            }
        }
    }
    cl.sync();  // no rank exits while rank 0 reads its shared memory
}

// The kernels (BF16 and MXFP8 weights): one name, overloaded on the segments, so the profile's demangled names read
// `small_m_kernel<O, TN, Hi>` for both (D12 attributes a record by its launch, not its weight format).
template <uint32_t O, uint32_t TN, bool Hi>
__global__ void __launch_bounds__(kGroup * 32)
    small_m_kernel(const __grid_constant__ LaunchSegments segs, const uint4 *__restrict__ x, uint32_t m,
                   uint32_t chunks, uint32_t cluster) {
    small_m_body<O, TN, Hi>(segs, x, m, chunks, cluster);
}

template <uint32_t O, uint32_t TN, bool Hi>
__global__ void __launch_bounds__(kGroup * 32)
    small_m_kernel(const __grid_constant__ Mxfp8LaunchSegments segs, const uint4 *__restrict__ x, uint32_t m,
                   uint32_t chunks, uint32_t cluster) {
    small_m_body<O, TN, Hi>(segs, x, m, chunks, cluster);
}

template <uint32_t O, uint32_t TN, bool Hi>
__global__ void __launch_bounds__(kGroup * 32)
    small_m_kernel(const __grid_constant__ EarlyLaunchSegments segs, const uint4 *__restrict__ x, uint32_t m,
                   uint32_t chunks, uint32_t cluster) {
    small_m_body<O, TN, Hi, LaunchSegments, true>(segs, x, m, chunks, cluster);
}

// gemm_hc_gate's early launch: its weights' parameter type (the overload of hc_gate_kernel that issues them before
// the wait).
struct EarlyWeights {
    const uint4 *w;
};

// gemm_hc_gate: CTA b holds column tile j0 = b * TN of each of the hc::kHc streams (w rows g * hidden + j0 ..);
// warp v runs slice v % slices of stream v / slices (slices = split_warps(k), hc::kHc * slices <= kGroup). Each
// stream's column is summed in slice order, exactly as small_m_kernel's single-CTA sum, then gated in stream order.
// Early: the warp's first round of weights is copied into shared memory (first_round, the launch's dynamic shared
// memory) between launch_dependents and the wait (k2_small_m.h).
template <uint32_t TN, bool Hi, bool Early>
__device__ __forceinline__ void hc_gate_body(const uint4 *__restrict__ w, const uint4 *__restrict__ x,
                                             const __nv_bfloat16 *__restrict__ normed,
                                             __nv_bfloat16 *__restrict__ mixed, uint32_t m, uint32_t chunks,
                                             uint32_t hidden) {
    constexpr uint32_t Q = TN / 8;
    __shared__ float partial[kGroup][kMaxM][TN];
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t slices = blockDim.x / 32 / hc::kHc;
    const uint32_t stream = warp / slices;
    const uint32_t g = lane / 4;
    const uint32_t j0 = blockIdx.x * TN;
    if constexpr (Early) {
        launch_dependents();
    } else {
        dependency_wait();
    }
    bool col[Q];
    Bf16Row wc[Q];
#pragma unroll
    for (uint32_t q = 0; q < Q; ++q) {
        const uint32_t j = j0 + 8 * q + g;
        col[q] = j < hidden;
        wc[q] = Bf16Row{w + (col[q] ? static_cast<size_t>(stream) * hidden + j : 0) * chunks + lane % 4};
    }
    float acc[Q][4] = {};
    if constexpr (Early) {
        extern __shared__ uint4 early_slots[];
        uint4 *pre = early_slots + static_cast<size_t>(warp) * kUnroll * Q * 32;
        first_round<Q>(pre, chunks, wc, col, warp % slices, slices);
        grid_wait();
        first_round_wait();
        slice_mma<Q, Hi, Bf16Row, true>(acc, x, m, chunks, wc, col, warp % slices, slices, pre);
    } else {
        slice_mma<Q, Hi>(acc, x, m, chunks, wc, col, warp % slices, slices);
    }
    put_partial<Q, TN>(partial[warp], acc);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < m * TN; i += blockDim.x) {
        const uint32_t r = i / TN;
        const uint32_t c = i % TN;
        if (j0 + c < hidden) {
            float u[hc::kHc];
            __nv_bfloat16 nv[hc::kHc];
#pragma unroll
            for (uint32_t s = 0; s < hc::kHc; ++s) {
                float sum = partial[s * slices][r][c];
                for (uint32_t v = 1; v < slices; ++v) {
                    sum += partial[s * slices + v][r][c];
                }
                u[s] = sum;
                nv[s] = normed[(static_cast<size_t>(r) * hc::kHc + s) * hidden + j0 + c];
            }
            mixed[static_cast<size_t>(r) * hidden + j0 + c] = hc::gate(u, nv);
        }
    }
}

template <uint32_t TN, bool Hi>
__global__ void __launch_bounds__(kGroup * 32)
    hc_gate_kernel(const uint4 *__restrict__ w, const uint4 *__restrict__ x,
                   const __nv_bfloat16 *__restrict__ normed, __nv_bfloat16 *__restrict__ mixed, uint32_t m,
                   uint32_t chunks, uint32_t hidden) {
    hc_gate_body<TN, Hi, false>(w, x, normed, mixed, m, chunks, hidden);
}

template <uint32_t TN, bool Hi>
__global__ void __launch_bounds__(kGroup * 32)
    hc_gate_kernel(EarlyWeights w, const uint4 *__restrict__ x, const __nv_bfloat16 *__restrict__ normed,
                   __nv_bfloat16 *__restrict__ mixed, uint32_t m, uint32_t chunks, uint32_t hidden) {
    hc_gate_body<TN, Hi, true>(w.w, x, normed, mixed, m, chunks, hidden);
}

bool aligned16(const void *p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

uint64_t tiles_of(const uint32_t *ns, uint32_t count, uint32_t tile_n) {
    uint64_t tiles = 0;
    for (uint32_t i = 0; i < count; ++i) {
        tiles += (uint64_t{ns[i]} + tile_n - 1) / tile_n;
    }
    return tiles;
}

int sm_count() {
    int device = 0;
    int sms = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || sms <= 0) {
        return 0;
    }
    return sms;
}

// 16-column tiles when the tiles times the groups give every SM a CTA, else 8-column tiles; one CTA per slice group
// (a cluster of split_warps(k) / kGroup CTAs) when there is more than one group. {0, ...} outside the contract.
Layout layout(uint32_t k, const uint32_t *ns, uint32_t count) {
    const Layout none{0, 0, 0, 0};
    if (!supported(k) || ns == nullptr || count == 0 || count > kMaxSegments) {
        return none;
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (ns[i] == 0) {
            return none;
        }
    }
    const int sms = sm_count();
    if (sms == 0) {
        return none;
    }
    const uint32_t slices = split_warps(k);
    const uint32_t cluster = slices > kGroup ? slices / kGroup : 1;
    const uint32_t warps = slices / cluster;
    const uint64_t wide = tiles_of(ns, count, 16);
    const uint32_t tile_n = wide * cluster >= static_cast<uint64_t>(sms) ? 16 : 8;
    const uint64_t tiles = tile_n == 16 ? wide : tiles_of(ns, count, 8);
    if (tiles * cluster > 0x7fffffffu) {
        return none;
    }
    return Layout{tile_n, cluster, static_cast<uint32_t>(tiles), warps};
}

// gemm_hc_gate's layout: one CTA per column tile of `hidden` (tile_n by the rule above), hc::kHc * slices warps.
Layout gate_layout(uint32_t k, uint32_t hidden) {
    const Layout none{0, 0, 0, 0};
    if (!hc_gate_supported(k) || hidden == 0) {
        return none;
    }
    const int sms = sm_count();
    if (sms == 0) {
        return none;
    }
    const uint32_t wide = (hidden + 15) / 16;
    const uint32_t tile_n = wide >= static_cast<uint32_t>(sms) ? 16 : 8;
    return Layout{tile_n, 1, (hidden + tile_n - 1) / tile_n, hc::kHc * split_warps(k)};
}

// An early launch's dynamic shared memory (early_bytes; with the kernel's static shared memory the total passes the
// default 48 KiB at every hc shape, so the kernel opts in). The limit is state of the CUDA context, cached (audit
// KGM-4) per kernel variant (one EarlyOpted per launch site and variant) and ordinal below kTrackedDevices: the
// ordinal's context generation + 1 (device/context_generation.h) in bits 63..32 and the limit raised in it in bits
// 31..0 (0: never). A warm launch makes no driver call; a reset primary context (release_device=1), a larger `bytes`
// or an untracked ordinal raises it again. Raises are serialized, so a limit never drops within a context while
// another launch relies on it. The hc launches are captured in the decode graphs: the opt-in is host state only.
using EarlyOpted = std::atomic<uint64_t>[cere::device::kTrackedDevices];
std::atomic<uint64_t> early_raises{0};

cudaError_t set_early_bytes(const void *kernel, EarlyOpted &opted, cudaLaunchConfig_t &config, size_t bytes) {
    config.dynamicSmemBytes = bytes;
    int id = -1;
    if (const cudaError_t e = cudaGetDevice(&id); e != cudaSuccess) {
        return e;
    }
    const bool tracked = id >= 0 && id < cere::device::kTrackedDevices;
    const uint64_t generation = tracked ? uint64_t{cere::device::context_generation(id).load()} + 1 : 0;
    const auto raised = [&](uint64_t seen) { return seen >> 32 == generation && (seen & 0xffffffffu) >= bytes; };
    if (tracked && raised(opted[id].load(std::memory_order_acquire))) {
        return cudaSuccess;
    }
    static std::mutex raise;
    const std::lock_guard<std::mutex> lock(raise);
    if (tracked && raised(opted[id].load(std::memory_order_relaxed))) {
        return cudaSuccess;
    }
    early_raises.fetch_add(1, std::memory_order_relaxed);
    const cudaError_t e =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if (e == cudaSuccess && tracked) {
        opted[id].store(generation << 32 | bytes, std::memory_order_release);
    }
    return e;
}

// cudaLaunchKernelEx config of a launch: its cluster (when > 1) and programmatic stream serialization (P11R-21).
struct Launch {
    cudaLaunchAttribute attrs[2];
    cudaLaunchConfig_t config;

    Launch(dim3 grid, dim3 block, uint32_t cluster, cudaStream_t stream) : attrs{}, config{} {
        uint32_t count = 0;
        attrs[count].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[count].val.programmaticStreamSerializationAllowed = 1;
        ++count;
        if (cluster > 1) {
            attrs[count].id = cudaLaunchAttributeClusterDimension;
            attrs[count].val.clusterDim.x = cluster;
            attrs[count].val.clusterDim.y = 1;
            attrs[count].val.clusterDim.z = 1;
            ++count;
        }
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        config.attrs = attrs;
        config.numAttrs = count;
    }
};

// One launch over `segs` (LaunchSegments, EarlyLaunchSegments or Mxfp8LaunchSegments, every field but tile_begin set;
// ns their widths): layout(k, ns, count) and the small_m_kernel overload of Segs.
template <uint32_t O, class Segs>
cudaError_t launch(const __nv_bfloat16 *x, uint32_t m, uint32_t k, Segs segs, const uint32_t *ns,
                   cudaStream_t stream) {
    const Layout l = layout(k, ns, segs.count);
    if (l.tile_n == 0) {
        return cudaErrorInvalidValue;
    }
    uint32_t tiles = 0;  // < 2^31 when layout() accepts the segments
    for (uint32_t i = 0; i < segs.count; ++i) {
        segs.s[i].tile_begin = tiles;
        tiles += (ns[i] + l.tile_n - 1) / l.tile_n;
    }
    using Kernel = void (*)(Segs, const uint4 *, uint32_t, uint32_t, uint32_t);
    const bool hi = m > 8;
    const Kernel kernel = l.tile_n == 16
                              ? (hi ? static_cast<Kernel>(small_m_kernel<O, 16, true>)
                                    : static_cast<Kernel>(small_m_kernel<O, 16, false>))
                              : (hi ? static_cast<Kernel>(small_m_kernel<O, 8, true>)
                                    : static_cast<Kernel>(small_m_kernel<O, 8, false>));
    const auto *xv = reinterpret_cast<const uint4 *>(x);
    const uint32_t chunks = k / 8;
    Launch c(dim3(l.tiles * l.cluster, 1, 1), dim3(32 * l.warps, 1, 1), l.cluster, stream);
    if constexpr (std::is_same_v<Segs, EarlyLaunchSegments>) {
        static EarlyOpted opted[4];  // [2 * (tile_n == 16) + hi]: one per kernel variant
        const cudaError_t s = set_early_bytes(reinterpret_cast<const void *>(kernel), opted[2 * (l.tile_n == 16) + hi],
                                              c.config, early_bytes(l.warps, l.tile_n));
        if (s != cudaSuccess) {
            return s;
        }
    }
    const cudaError_t e = cudaLaunchKernelEx(&c.config, kernel, segs, xv, m, chunks, l.cluster);
    const cudaError_t last = cudaGetLastError();
    return e != cudaSuccess ? e : last;
}

bool args_ok(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const void *segments, uint32_t count) {
    return x != nullptr && aligned16(x) && m != 0 && m <= kMaxM && supported(k) && segments != nullptr &&
           count != 0 && count <= kMaxSegments;
}

// Segs: LaunchSegments, or EarlyLaunchSegments for an early launch (gemm, gemm_hc_act).
template <uint32_t O, class Segs = LaunchSegments>
cudaError_t launch_gemm(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const Segment *segments, uint32_t count,
                        cudaStream_t stream) {
    if (!args_ok(x, m, k, segments, count)) {
        return cudaErrorInvalidValue;
    }
    uint32_t ns[kMaxSegments];
    Segs segs{};
    for (uint32_t i = 0; i < count; ++i) {
        const Segment &s = segments[i];
        if (s.w == nullptr || !aligned16(s.w) || s.y == nullptr || s.n == 0) {
            return cudaErrorInvalidValue;
        }
        ns[i] = s.n;
        segs.s[i] = LaunchSegment{reinterpret_cast<const uint4 *>(s.w), s.y, s.n, 0};
    }
    segs.count = count;
    return launch<O>(x, m, k, segs, ns, stream);
}

template <uint32_t O>
cudaError_t launch_gemm_mxfp8(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const Mxfp8Segment *segments,
                              uint32_t count, cudaStream_t stream) {
    if (!args_ok(x, m, k, segments, count)) {
        return cudaErrorInvalidValue;
    }
    uint32_t ns[kMaxSegments];
    Mxfp8LaunchSegments segs{};
    for (uint32_t i = 0; i < count; ++i) {
        const Mxfp8Segment &s = segments[i];
        if (s.q == nullptr || !aligned16(s.q) || s.scales == nullptr || s.y == nullptr || s.n == 0) {
            return cudaErrorInvalidValue;
        }
        ns[i] = s.n;
        segs.s[i] = Mxfp8LaunchSegment{reinterpret_cast<const uint2 *>(s.q), s.scales, s.y, s.n, 0};
    }
    segs.count = count;
    return launch<O>(x, m, k, segs, ns, stream);
}

// gemm_hc_gate's launch of the hc_gate_kernel overload whose weights parameter is Wp (const uint4 *, or EarlyWeights
// with early_bytes of dynamic shared memory).
template <class Wp>
cudaError_t launch_gate(Launch &c, const Layout &l, Wp w, const uint4 *x, const __nv_bfloat16 *normed,
                        __nv_bfloat16 *mixed, uint32_t m, uint32_t chunks, uint32_t hidden) {
    using Kernel = void (*)(Wp, const uint4 *, const __nv_bfloat16 *, __nv_bfloat16 *, uint32_t, uint32_t, uint32_t);
    const bool hi = m > 8;
    const Kernel kernel = l.tile_n == 16 ? (hi ? static_cast<Kernel>(hc_gate_kernel<16, true>)
                                               : static_cast<Kernel>(hc_gate_kernel<16, false>))
                                         : (hi ? static_cast<Kernel>(hc_gate_kernel<8, true>)
                                               : static_cast<Kernel>(hc_gate_kernel<8, false>));
    if constexpr (std::is_same_v<Wp, EarlyWeights>) {
        static EarlyOpted opted[4];  // [2 * (tile_n == 16) + hi]: one per kernel variant
        const cudaError_t s = set_early_bytes(reinterpret_cast<const void *>(kernel), opted[2 * (l.tile_n == 16) + hi],
                                              c.config, early_bytes(l.warps, l.tile_n));
        if (s != cudaSuccess) {
            return s;
        }
    }
    const cudaError_t e = cudaLaunchKernelEx(&c.config, kernel, w, x, normed, mixed, m, chunks, hidden);
    const cudaError_t last = cudaGetLastError();
    return e != cudaSuccess ? e : last;
}

}  // namespace

uint64_t early_opt_ins() { return early_raises.load(std::memory_order_relaxed); }

bool supported(uint32_t k) { return k > 0 && k % kStepK == 0; }

uint32_t split_warps(uint32_t k) {
    const uint32_t steps = k / kStepK;
    if (steps <= kGroup * kOneGroupSteps) {
        const uint32_t s = steps / kStepsPerWarp;
        return s < 1 ? 1 : (s > kGroup ? kGroup : s);
    }
    const uint32_t groups = (steps + kGroup * kSliceSteps - 1) / (kGroup * kSliceSteps);
    return kGroup * (groups > kMaxCluster ? kMaxCluster : groups);
}

bool hc_gate_supported(uint32_t k) { return supported(k) && hc::kHc * split_warps(k) <= kGroup; }

Geometry geometry(uint32_t k, const uint32_t *ns, uint32_t count) {
    const Layout l = layout(k, ns, count);
    if (l.tile_n == 0) {
        return Geometry{dim3(0, 0, 0), dim3(0, 0, 0), dim3(0, 0, 0)};
    }
    return Geometry{dim3(l.tiles * l.cluster, 1, 1), dim3(32 * l.warps, 1, 1), dim3(l.cluster, 1, 1)};
}

Geometry hc_gate_geometry(uint32_t k, uint32_t hidden) {
    const Layout l = gate_layout(k, hidden);
    if (l.tile_n == 0) {
        return Geometry{dim3(0, 0, 0), dim3(0, 0, 0), dim3(0, 0, 0)};
    }
    return Geometry{dim3(l.tiles, 1, 1), dim3(32 * l.warps, 1, 1), dim3(1, 1, 1)};
}

cudaError_t gemm(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const Segment *segments, uint32_t count, bool f32,
                 cudaStream_t stream, bool early) {
    if (early) {
        return f32 ? launch_gemm<kOutF32, EarlyLaunchSegments>(x, m, k, segments, count, stream)
                   : launch_gemm<kOutBf16, EarlyLaunchSegments>(x, m, k, segments, count, stream);
    }
    return f32 ? launch_gemm<kOutF32>(x, m, k, segments, count, stream)
               : launch_gemm<kOutBf16>(x, m, k, segments, count, stream);
}

cudaError_t gemm_mxfp8(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const Mxfp8Segment *segments, uint32_t count,
                       bool f32, cudaStream_t stream) {
    return f32 ? launch_gemm_mxfp8<kOutF32>(x, m, k, segments, count, stream)
               : launch_gemm_mxfp8<kOutBf16>(x, m, k, segments, count, stream);
}

cudaError_t gemm_hc_act(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const __nv_bfloat16 *w, uint32_t n,
                        __nv_bfloat16 *t, cudaStream_t stream, bool early) {
    const Segment seg{w, t, n};
    return early ? launch_gemm<kOutHcAct, EarlyLaunchSegments>(x, m, k, &seg, 1, stream)
                 : launch_gemm<kOutHcAct>(x, m, k, &seg, 1, stream);
}

cudaError_t gemm_hc_gate(const __nv_bfloat16 *t, uint32_t m, uint32_t k, const __nv_bfloat16 *w, uint32_t hidden,
                         const __nv_bfloat16 *normed, __nv_bfloat16 *mixed, cudaStream_t stream, bool early) {
    if (t == nullptr || !aligned16(t) || m == 0 || m > kMaxM || w == nullptr || !aligned16(w) || normed == nullptr ||
        mixed == nullptr) {
        return cudaErrorInvalidValue;
    }
    const Layout l = gate_layout(k, hidden);
    if (l.tile_n == 0) {
        return cudaErrorInvalidValue;
    }
    const auto *xv = reinterpret_cast<const uint4 *>(t);
    const auto *wv = reinterpret_cast<const uint4 *>(w);
    const uint32_t chunks = k / 8;
    Launch c(dim3(l.tiles, 1, 1), dim3(32 * l.warps, 1, 1), 1, stream);
    return early ? launch_gate(c, l, EarlyWeights{wv}, xv, normed, mixed, m, chunks, hidden)
                 : launch_gate(c, l, wv, xv, normed, mixed, m, chunks, hidden);
}

}  // namespace cere::kernels::k2s
