// Inkling 2 Mini S8: the BF16 projections (contract and the order of one output element in linear.h). Compiled with
// -fmad=false; the only FP32 arithmetic outside the mma instructions is the slice adds, written as __fadd_rn.
#include "kernels/inkling/linear.h"

#include <climits>

#include "kernels/inkling/pdl.cuh"

namespace cere::inkling {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kWindow = 32;                                  // k per window (two mma steps)
constexpr int kSliceWindows = kLinearSliceK / kWindow;       // 8
constexpr int kMaxSlices = kLinearMaxK / kLinearSliceK;      // 12
constexpr int64_t kMaxRows = int64_t(1) << 20;

// d += A · B over one m16n8k16 step: a0..a3, b0..b1 in the PTX fragment layout (linear.h's slot mapping is in window).
__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One window [k0, k0 + 32) for lane (g = lane / 4, c = lane % 4): r0 / r1 = weight rows g / g + 8 and b = the input
// row g of the n8 tile, each the 8 values at k0 + 8c. Step 0: slots (2c, 2c+1) <- k0 + 8c + {0, 1}, (2c+8, 2c+9) <-
// k0 + 8c + {2, 3}; step 1 the same with + 4.
__device__ __forceinline__ void window(float (&d)[4], const uint4 &r0, const uint4 &r1, const uint4 &b) {
    mma(d, r0.x, r1.x, r0.y, r1.y, b.x, b.y);
    mma(d, r0.z, r1.z, r0.w, r1.w, b.z, b.w);
}

// Weights: a weak load (not ld.global.nc, which is outside the memory model: ptxas would move it across pdl::wait()'s
// fence to its first use), volatile so that it is issued in program order before the wait.
__device__ __forceinline__ uint4 load_stream(const bf16 *p) {
    uint4 v;
    asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
    return v;
}

// Inputs (written by the previous kernel): a weak load, which ptxas keeps after the PDL wait (an ld.global.nc may be
// hoisted above it).
__device__ __forceinline__ uint4 load_input(const bf16 *p) { return *reinterpret_cast<const uint4 *>(p); }

struct Tile {
    int seg;
    int64_t f0;  // first weight row of the tile
};

// Tiles are numbered segment by segment, ceil(n / rows) per segment.
__device__ __forceinline__ Tile find_tile(const LinearArgs &a, int64_t tile, int rows) {
    int s = 0;
    for (; s < a.count - 1; ++s) {
        const int64_t t = (a.seg[s].n + rows - 1) / rows;
        if (tile < t) break;
        tile -= t;
    }
    return {s, tile * rows};
}

// y[t, f .. f + 7] (f + 8 may pass n: the tail is written element by element).
__device__ __forceinline__ void store8(const LinearSeg &sg, int64_t t, int64_t f, const float (&v)[8]) {
    bf16 *y = sg.y + t * sg.ldy + f;
    if (f + 8 <= sg.n) {
        __align__(16) bf16 o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = __float2bfloat16_rn(v[j]);
        *reinterpret_cast<uint4 *>(y) = *reinterpret_cast<const uint4 *>(o);
    } else {
        for (int j = 0; j < 8 && f + j < sg.n; ++j) y[j] = __float2bfloat16_rn(v[j]);
    }
}

// ---- narrow: m <= 64 -------------------------------------------------------------------------------------------
// One block per 16 weight rows, warp s = slice s (blockDim = 32 S): the warp loads its 2 x 8 weight vectors first
// (8 KB in flight per warp, before the PDL wait: weights), then chains the 16 steps over NT n8 tiles of inputs; the
// partials meet in shared memory and one thread per (input row, 8 weight rows) adds them in slice order.
template <int NT>
__global__ void __launch_bounds__(32 * kMaxSlices) narrow_kernel(const LinearArgs a) {
    pdl::trigger();
    __shared__ float part[kMaxSlices][16][NT * 8];
    const int S = int(a.k / kLinearSliceK);
    const int warp = int(threadIdx.x) >> 5, lane = int(threadIdx.x) & 31, g = lane >> 2, c = lane & 3;
    const Tile tl = find_tile(a, blockIdx.x, 16);
    const LinearSeg &sg = a.seg[tl.seg];
    const int64_t k = a.k, kb = int64_t(warp) * kLinearSliceK + 8 * c;
    const int64_t r0 = tl.f0 + g, r1 = r0 + 8;
    const uint4 zero = make_uint4(0, 0, 0, 0);
    uint4 w0[kSliceWindows], w1[kSliceWindows];
#pragma unroll
    for (int i = 0; i < kSliceWindows; ++i) {
        w0[i] = r0 < sg.n ? load_stream(sg.w + r0 * k + kb + kWindow * i) : zero;
        w1[i] = r1 < sg.n ? load_stream(sg.w + r1 * k + kb + kWindow * i) : zero;
    }
    pdl::wait();
    float acc[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
#pragma unroll
    for (int i = 0; i < kSliceWindows; ++i) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int64_t row = t * 8 + g;
            const uint4 b = row < a.m ? load_input(sg.x + row * sg.ldx + kb + kWindow * i) : zero;
            window(acc[t], w0[i], w1[i], b);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        part[warp][g][t * 8 + 2 * c] = acc[t][0];
        part[warp][g][t * 8 + 2 * c + 1] = acc[t][1];
        part[warp][g + 8][t * 8 + 2 * c] = acc[t][2];
        part[warp][g + 8][t * 8 + 2 * c + 1] = acc[t][3];
    }
    __syncthreads();
    for (int q = int(threadIdx.x); q < NT * 16; q += int(blockDim.x)) {
        const int t = q >> 1, half = q & 1;
        const int64_t f = tl.f0 + 8 * half;
        if (t >= a.m || f >= sg.n) continue;
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float s = part[0][8 * half + j][t];
            for (int p = 1; p < S; ++p) s = __fadd_rn(s, part[p][8 * half + j][t]);
            v[j] = s;
        }
        store8(sg, t, f, v);
    }
}

// ---- wide: any m ----------------------------------------------------------------------------------------------
// Block tile 128 weight rows x 128 inputs, 8 warps of 32 x 64 (2 m16 x 8 n8), one block per SM: 3-deep cp.async
// pipeline of two windows per stage (one barrier per 64 mma steps of a warp, D-I-CPF). A stage holds per window the
// 256 rows (weights, then inputs) of 64 bytes (4 chunks of 8 values), so a fragment load's 8-lane phase (rows 2p,
// 2p + 1 x chunks 0..3) covers the 32 banks once. Each warp keeps its slice partials and the running total. Swap: the
// fail-first mutant (each window's two mma steps in the other order; LinearPath::kWideMutant).
constexpr int kWideRows = 128, kWideInputs = 128, kWideThreads = 256, kStages = 3, kStageWindows = 2;
constexpr int kWindowChunks = (kWideRows + kWideInputs) * (kWindow / 8);  // 16-byte chunks per window
constexpr int kStageChunks = kStageWindows * kWindowChunks;
constexpr size_t kWideSmem = size_t(kStages) * kStageChunks * 16;          // 96 KiB (dynamic)
constexpr int kOutPitch = kWideRows + 8;                                    // epilogue row (bf16), 16-byte multiple
static_assert(kWideSmem >= size_t(kWideInputs) * kOutPitch * 2, "the epilogue tile fits the stages");
static_assert(kSliceWindows % kStageWindows == 0, "a stage never straddles a slice");

__device__ __forceinline__ void cp_async16(void *dst, const void *src, bool ok) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src), "r"(ok ? 16 : 0));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <bool Swap>
__global__ void __launch_bounds__(kWideThreads, 1) wide_kernel(const LinearArgs a) {
    pdl::enter();
    extern __shared__ __align__(16) uint4 smem[];
    const Tile tl = find_tile(a, blockIdx.x, kWideRows);
    const LinearSeg &sg = a.seg[tl.seg];
    const int64_t t0 = int64_t(blockIdx.y) * kWideInputs, k = a.k;
    const int tid = int(threadIdx.x), warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int wf = (warp & 3) * 32, wt = (warp >> 2) * 64;
    const int iters = int(k / (kWindow * kStageWindows));

    // Thread tid loads chunk tid % 4 of weight rows tid / 4 and + 64 and of input rows tid / 4 and + 64, per window.
    static_assert(kWideThreads == 256 && kWideRows == 128 && kWideInputs == 128, "the load mapping below");
    const int lr = tid >> 2, lc = tid & 3;
    const bf16 *src[4];
    bool ok[4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int64_t f = tl.f0 + lr + 64 * h, t = t0 + lr + 64 * h;
        ok[h] = f < sg.n;
        src[h] = ok[h] ? sg.w + f * k + 8 * lc : sg.w;
        ok[2 + h] = t < a.m;
        src[2 + h] = ok[2 + h] ? sg.x + t * sg.ldx + 8 * lc : sg.x;
    }
    auto load = [&](int it) {
        uint4 *st = smem + (it % kStages) * kStageChunks + tid;
        const int kk = it * kWindow * kStageWindows;
#pragma unroll
        for (int win = 0; win < kStageWindows; ++win)
#pragma unroll
            for (int p = 0; p < 4; ++p)
                cp_async16(st + win * kWindowChunks + p * kWideThreads, src[p] + (ok[p] ? kk + kWindow * win : 0),
                           ok[p]);
    };

    float part[2][8][4], total[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) part[i][j][e] = total[i][j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < iters) load(s);
        cp_async_commit();
    }
    for (int it = 0; it < iters; ++it) {
        cp_async_wait<kStages - 2>();
        __syncthreads();
        if (it + kStages - 1 < iters) load(it + kStages - 1);
        cp_async_commit();
#pragma unroll 1  // one window's fragments live at a time (the 128 accumulators leave no room for two)
        for (int win = 0; win < kStageWindows; ++win) {
            const uint4 *st = smem + (it % kStages) * kStageChunks + win * kWindowChunks;
            const uint4 *bs = st + kWideRows * 4;
            uint4 r0[2], r1[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                r0[i] = st[(wf + 16 * i + g) * 4 + c];
                r1[i] = st[(wf + 16 * i + g + 8) * 4 + c];
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint4 b = bs[(wt + 8 * j + g) * 4 + c];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    if constexpr (Swap) {
                        mma(part[i][j], r0[i].z, r1[i].z, r0[i].w, r1[i].w, b.z, b.w);
                        mma(part[i][j], r0[i].x, r1[i].x, r0[i].y, r1[i].y, b.x, b.y);
                    } else {
                        window(part[i][j], r0[i], r1[i], b);
                    }
                }
            }
        }
        if (((it + 1) * kStageWindows) % kSliceWindows == 0) {
            const bool first = (it + 1) * kStageWindows == kSliceWindows;
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        total[i][j][e] = first ? part[i][j][e] : __fadd_rn(total[i][j][e], part[i][j][e]);
                        part[i][j][e] = 0.0f;
                    }
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    // Epilogue: the tile as BF16 [input][weight row] in shared memory, then 16-byte rows of y.
    bf16 *out = reinterpret_cast<bf16 *>(smem);
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int f = wf + 16 * i + g, t = wt + 8 * j + 2 * c;
            out[t * kOutPitch + f] = __float2bfloat16_rn(total[i][j][0]);
            out[(t + 1) * kOutPitch + f] = __float2bfloat16_rn(total[i][j][1]);
            out[t * kOutPitch + f + 8] = __float2bfloat16_rn(total[i][j][2]);
            out[(t + 1) * kOutPitch + f + 8] = __float2bfloat16_rn(total[i][j][3]);
        }
    __syncthreads();
    for (int q = tid; q < kWideInputs * (kWideRows / 8); q += kWideThreads) {
        const int t = q / (kWideRows / 8), chunk = q % (kWideRows / 8);
        const int64_t row = t0 + t, f = tl.f0 + 8 * chunk;
        if (row >= a.m || f >= sg.n) continue;
        const bf16 *src = out + t * kOutPitch + 8 * chunk;
        bf16 *y = sg.y + row * sg.ldy + f;
        if (f + 8 <= sg.n) {
            *reinterpret_cast<uint4 *>(y) = *reinterpret_cast<const uint4 *>(src);
        } else {
            for (int j = 0; j < 8 && f + j < sg.n; ++j) y[j] = src[j];
        }
    }
}

template <bool Swap>
cudaError_t launch_wide(const LinearArgs &a, int64_t tiles, cudaStream_t stream) {
    const cudaError_t e =
        cudaFuncSetAttribute(wide_kernel<Swap>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kWideSmem));
    if (e != cudaSuccess) return e;
    const dim3 grid(unsigned(tiles), unsigned((a.m + kWideInputs - 1) / kWideInputs));
    return pdl::launch(wide_kernel<Swap>, grid, dim3(kWideThreads), kWideSmem, stream, a);
}

bool aligned16(const void *p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace

cudaError_t linear(const LinearArgs &a, cudaStream_t stream, LinearPath path) {
    if (a.count < 1 || a.count > kLinearMaxSegs || a.k < kLinearSliceK || a.k > kLinearMaxK ||
        a.k % kLinearSliceK != 0 || a.m < 1 || a.m > kMaxRows) {
        return cudaErrorInvalidValue;
    }
    int64_t narrow_tiles = 0, wide_tiles = 0;
    for (int s = 0; s < a.count; ++s) {
        const LinearSeg &sg = a.seg[s];
        if (sg.x == nullptr || sg.w == nullptr || sg.y == nullptr || sg.n < 1 || sg.n > INT_MAX || !aligned16(sg.x) ||
            !aligned16(sg.w) || !aligned16(sg.y) || sg.ldx < a.k || sg.ldx % 8 != 0 || sg.ldy < sg.n ||
            sg.ldy % 8 != 0) {
            return cudaErrorInvalidValue;
        }
        narrow_tiles += (sg.n + 15) / 16;
        wide_tiles += (sg.n + kWideRows - 1) / kWideRows;
    }
    const bool narrow = path == LinearPath::kNarrow || (path == LinearPath::kAuto && a.m <= kLinearNarrowRows);
    if ((narrow && a.m > kLinearNarrowRows) || narrow_tiles > INT_MAX) return cudaErrorInvalidValue;
    if (narrow) {
        const dim3 grid{unsigned(narrow_tiles)}, block{unsigned(32 * (a.k / kLinearSliceK))};
        const int64_t nt = (a.m + 7) / 8;
        if (nt <= 1) return pdl::launch(narrow_kernel<1>, grid, block, 0, stream, a);
        if (nt <= 2) return pdl::launch(narrow_kernel<2>, grid, block, 0, stream, a);
        if (nt <= 4) return pdl::launch(narrow_kernel<4>, grid, block, 0, stream, a);
        return pdl::launch(narrow_kernel<8>, grid, block, 0, stream, a);
    }
    return path == LinearPath::kWideMutant ? launch_wide<true>(a, wide_tiles, stream)
                                           : launch_wide<false>(a, wide_tiles, stream);
}

}  // namespace cere::inkling
