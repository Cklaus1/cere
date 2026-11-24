// K2 FP8 weights: contract, storage and reduction order in k2_fp8.h.
#include "kernels/k2_fp8.h"

#include <cuda_fp8.h>

#include <algorithm>
#include <functional>

namespace cere::kernels::k2f {
namespace {

constexpr uint32_t kMaxWarps = 16;     // slices per column at most: one CTA, summed in slice order
constexpr uint32_t kStepsPerWarp = 5;  // split_warps: about this many 64-element steps per slice (two load batches)
constexpr uint32_t kUnroll = 4;        // steps whose loads are issued before their MMAs
constexpr uint32_t kOutBf16 = 0;
constexpr uint32_t kOutF32 = 1;
constexpr uint32_t kQuantThreads = 256;
constexpr uint32_t kDequantThreads = 256;

struct LaunchSegment {
    const uint4 *q;
    const float *scale;
    void *y;
    uint32_t n;
    uint32_t tile_begin;  // first column tile of this segment
};

struct LaunchSegments {
    LaunchSegment s[kMaxSegments];
    uint32_t count;
};

// Weights are read once per launch: no L1 allocation, L2 256-byte prefetch (as k2_small_m.cu).
__device__ __forceinline__ uint4 load_stream(const uint4 *p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// Programmatic dependent launch (k2_small_m.h, P11R-21): x is read only after the wait. The converted weights and
// their scales are written once, at INSTANTIATE (stream-synchronized before any forward), so a CTA issues its first
// weight loads and its scale load before the wait, overlapping the previous kernel's tail.
__device__ __forceinline__ void dependency_wait() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

// d += A · B for one m16n8k16 BF16 tile with FP32 accumulation (fragment layout as k2_small_m.cu).
__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Two E4M3 bytes of v (byte_perm selector `sel` puts them at bits 15:8 and 31:24) as the exact BF16 pair: the sign
// and the 7 exponent/mantissa bits placed in BF16's fields give e4m3 * 2^-120 (subnormal E4M3 to subnormal BF16
// alike), then one BF16 multiply by 2^120 (exact; BF16 arithmetic keeps subnormal inputs) makes every value normal.
// The E4M3 NaN bytes 0x7F / 0xFF come out as +-480 (dequant_kernel: NaN); quantize_kernel never emits them (KGM-4).
template <uint32_t Sel>
__device__ __forceinline__ uint32_t bf16_pair(uint32_t v) {
    const uint32_t t = __byte_perm(v, 0u, Sel);
    const uint32_t bits = (t & 0x80008000u) | ((t >> 4) & 0x07F007F0u);
    uint32_t out;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(out) : "r"(bits), "r"(0x7B807B80u));  // 0x7B80 = 2^120
    return out;
}

// One warp step at k = 64 s: lane (g = lane / 4, t = lane % 4) holds x rows g (xa) and g + 8 (xb) at k = 64 s + 16 t
// .. + 15 (two 16-byte chunks each) and q column g of each of the Q n-tiles there (16 bytes). MMA h takes k pairs
// (2t, 2t + 1) and (2t + 8, 2t + 9) of its tile from k = 64 s + 16 t + 4 h + (0, 1) and (2, 3).
template <uint32_t Q>
__device__ __forceinline__ void step(float (&acc)[Q][4], const uint4 (&xa)[2], const uint4 (&xb)[2],
                                     const uint4 (&w)[Q]) {
#pragma unroll
    for (uint32_t q = 0; q < Q; ++q) {
        mma(acc[q], xa[0].x, xb[0].x, xa[0].y, xb[0].y, bf16_pair<0x1404>(w[q].x), bf16_pair<0x3424>(w[q].x));
        mma(acc[q], xa[0].z, xb[0].z, xa[0].w, xb[0].w, bf16_pair<0x1404>(w[q].y), bf16_pair<0x3424>(w[q].y));
        mma(acc[q], xa[1].x, xb[1].x, xa[1].y, xb[1].y, bf16_pair<0x1404>(w[q].z), bf16_pair<0x3424>(w[q].z));
        mma(acc[q], xa[1].z, xb[1].z, xa[1].w, xb[1].w, bf16_pair<0x1404>(w[q].w), bf16_pair<0x3424>(w[q].w));
    }
}

// The weights of kUnroll steps s0, s0 + slices, ... (zeros past the steps and past n): loads issued, not awaited.
template <uint32_t Q>
__device__ __forceinline__ void load_weights(uint4 (&wv)[kUnroll][Q], const uint4 *const (&wc)[Q],
                                             const bool (&col)[Q], uint32_t s0, uint32_t slices, uint32_t steps) {
    const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
        const uint32_t s = s0 + u * slices;
        const bool in = s < steps;
#pragma unroll
        for (uint32_t q = 0; q < Q; ++q) {
            wv[u][q] = in && col[q] ? load_stream(wc[q] + static_cast<size_t>(s) * 4) : zero;
        }
    }
}

// One warp's slice: acc (zero on entry) += x · q over steps slice, slice + slices, ... < steps in increasing order.
// The weights of kUnroll steps are in flight at a time (the first batch was issued before the dependency wait, into
// wv); x (a few KB every CTA shares, L1/L2 resident) is loaded at its step. Rows past m and columns past n contribute
// zeros (no load); without Hi rows 8..15 are zero registers, so no output bit depends on Hi.
template <uint32_t Q, bool Hi>
__device__ __forceinline__ void slice_mma(float (&acc)[Q][4], uint4 (&wv)[kUnroll][Q], const uint4 *x, uint32_t m,
                                          uint32_t k, const uint4 *const (&wc)[Q], const bool (&col)[Q],
                                          uint32_t slice, uint32_t slices) {
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t g = lane / 4;
    const uint32_t t = lane % 4;
    const uint32_t steps = k / kStepK;
    const uint32_t xchunks = k / 8;
    const bool row0 = g < m;
    const bool row1 = Hi && g + 8 < m;
    const uint4 *xr0 = x + static_cast<size_t>(row0 ? g : 0) * xchunks + 2 * t;
    const uint4 *xr1 = x + static_cast<size_t>(row1 ? g + 8 : 0) * xchunks + 2 * t;
    const uint4 zero = make_uint4(0, 0, 0, 0);
    for (uint32_t s0 = slice; s0 < steps;) {
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
            const uint32_t s = s0 + u * slices;
            if (s < steps) {
                const size_t xoff = static_cast<size_t>(s) * 8;
                const uint4 xa[2] = {row0 ? __ldg(xr0 + xoff) : zero, row0 ? __ldg(xr0 + xoff + 1) : zero};
                uint4 xb[2] = {zero, zero};
                if constexpr (Hi) {
                    xb[0] = row1 ? __ldg(xr1 + xoff) : zero;
                    xb[1] = row1 ? __ldg(xr1 + xoff + 1) : zero;
                }
                step<Q>(acc, xa, xb, wv[u]);
            }
        }
        s0 += kUnroll * slices;
        if (s0 < steps) {
            load_weights<Q>(wv, wc, col, s0, slices, steps);
        }
    }
}

// CTA b: column tile b of the launch's segments; warp v runs slice v of split_warps(k) = blockDim.x / 32 slices.
// At most 64 registers without Hi (two 16-warp or four 8-warp CTAs per SM), so one CTA's load latency hides behind
// another's.
template <uint32_t O, uint32_t TN, bool Hi>
__global__ void __launch_bounds__(kMaxWarps * 32, Hi ? 1 : 2)
    fp8_kernel(const __grid_constant__ LaunchSegments segs, const uint4 *__restrict__ x, uint32_t m, uint32_t k) {
    constexpr uint32_t Q = TN / 8;
    __shared__ float partial[kMaxWarps][kMaxM][TN];
    const uint32_t tile = blockIdx.x;
    LaunchSegment seg = segs.s[0];
#pragma unroll
    for (uint32_t i = 1; i < kMaxSegments; ++i) {
        if (i < segs.count && tile >= segs.s[i].tile_begin) {
            seg = segs.s[i];
        }
    }
    const uint32_t col0 = (tile - seg.tile_begin) * TN;
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warps = blockDim.x / 32;
    const uint32_t g = lane / 4;
    const uint32_t t = lane % 4;
    const uint32_t wchunks = k / 16;
    bool col[Q];
    const uint4 *wc[Q];
#pragma unroll
    for (uint32_t q = 0; q < Q; ++q) {
        const uint32_t j = col0 + 8 * q + g;
        col[q] = j < seg.n;
        wc[q] = seg.q + static_cast<size_t>(col[q] ? j : 0) * wchunks + t;
    }
    // The reduction below gives thread i the column i % TN of every row it stores (blockDim.x % TN == 0).
    const uint32_t jc = threadIdx.x % TN;
    const float scale = col0 + jc < seg.n ? __ldg(seg.scale + col0 + jc) : 0.0f;
    uint4 wv[kUnroll][Q];
    load_weights<Q>(wv, wc, col, warp, warps, k / kStepK);
    dependency_wait();
    float acc[Q][4] = {};
    slice_mma<Q, Hi>(acc, wv, x, m, k, wc, col, warp, warps);
#pragma unroll
    for (uint32_t q = 0; q < Q; ++q) {
        const uint32_t c = 8 * q + 2 * t;
        partial[warp][g][c] = acc[q][0];
        partial[warp][g][c + 1] = acc[q][1];
        partial[warp][g + 8][c] = acc[q][2];
        partial[warp][g + 8][c + 1] = acc[q][3];
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < m * TN; i += blockDim.x) {
        const uint32_t r = i / TN;
        if (col0 + jc < seg.n) {
            float sum = partial[0][r][jc];
            for (uint32_t v = 1; v < warps; ++v) {
                sum += partial[v][r][jc];
            }
            const float y = sum * scale;
            const size_t at = static_cast<size_t>(r) * seg.n + col0 + jc;
            if constexpr (O == kOutF32) {
                static_cast<float *>(seg.y)[at] = y;
            } else {
                static_cast<__nv_bfloat16 *>(seg.y)[at] = __float2bfloat16_rn(y);
            }
        }
    }
}

// |v| folded into the running max a, non-finite-sticky (order-free): any NaN gives NaN, else any Inf gives Inf.
__device__ __forceinline__ float amax_fold(float a, float v) { return v > a || v != v ? v : a; }

// One CTA per row: the row's amax (a max: order-free), its scale, then every element. A row holding a NaN or an Inf
// (audit KGM-4) gets scale NaN and q 0, so both decodes give NaN (the GEMM's sum * NaN, dequant_kernel's 0 * NaN) as
// the BF16 weight's products would be non-finite. Every finite row saturates to +-448 (0x7E / 0xFE): quantize never
// emits 0x7F / 0xFF, the E4M3 NaN bytes bf16_pair decodes as +-480.
__global__ void __launch_bounds__(kQuantThreads)
    quantize_kernel(const __nv_bfloat16 *__restrict__ w, uint32_t k, uint8_t *__restrict__ q,
                    float *__restrict__ scale) {
    __shared__ float warp_max[kQuantThreads / 32];
    const size_t row = blockIdx.x;
    const __nv_bfloat16 *wr = w + row * k;
    float amax = 0.0f;
    for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
        amax = amax_fold(amax, fabsf(__bfloat162float(wr[i])));
    }
#pragma unroll
    for (uint32_t d = 16; d > 0; d /= 2) {
        amax = amax_fold(amax, __shfl_xor_sync(0xffffffffu, amax, d));
    }
    if (threadIdx.x % 32 == 0) {
        warp_max[threadIdx.x / 32] = amax;
    }
    __syncthreads();
    amax = warp_max[0];
    for (uint32_t v = 1; v < blockDim.x / 32; ++v) {
        amax = amax_fold(amax, warp_max[v]);
    }
    const bool finite = isfinite(amax);
    const float s = finite ? __fdiv_rn(amax, 448.0f) : __int_as_float(0x7FC00000);
    if (threadIdx.x == 0) {
        scale[row] = s;
    }
    uint8_t *qr = q + row * k;
    for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
        qr[i] = finite && s > 0.0f ? static_cast<uint8_t>(__nv_cvt_float_to_fp8(__fdiv_rn(__bfloat162float(wr[i]), s),
                                                                                __NV_SATFINITE, __NV_E4M3))
                                   : uint8_t{0};
    }
}

// Thread i: 16 E4M3 bytes at element 16 i (one row: k % 16 == 0) to 16 BF16.
__global__ void __launch_bounds__(kDequantThreads)
    dequant_kernel(const uint4 *__restrict__ q, const float *__restrict__ scale, uint32_t k, size_t chunks,
                   uint4 *__restrict__ w) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= chunks) {
        return;
    }
    const float s = scale[i * 16 / k];
    const uint4 v = q[i];
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
    uint32_t out[8];
#pragma unroll
    for (uint32_t b = 0; b < 4; ++b) {
#pragma unroll
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t pair = (words[b] >> (16 * h)) & 0xffffu;
            const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3);
            const float2 f = __half22float2(__half2(raw));
            const __nv_bfloat162 o = __floats2bfloat162_rn(f.x * s, f.y * s);
            out[2 * b + h] = *reinterpret_cast<const uint32_t *>(&o);
        }
    }
    w[2 * i] = make_uint4(out[0], out[1], out[2], out[3]);
    w[2 * i + 1] = make_uint4(out[4], out[5], out[6], out[7]);
}

bool aligned16(const void *p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

int sm_count() {
    int device = 0;
    int sms = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || sms <= 0) {
        return 0;
    }
    return sms;
}

uint64_t tiles_of(const Segment *segments, uint32_t count, uint32_t tile_n) {
    uint64_t tiles = 0;
    for (uint32_t i = 0; i < count; ++i) {
        tiles += (uint64_t{segments[i].n} + tile_n - 1) / tile_n;
    }
    return tiles;
}

template <uint32_t O>
cudaError_t launch_gemm(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const Segment *segments, uint32_t count,
                        cudaStream_t stream) {
    if (x == nullptr || !aligned16(x) || m == 0 || m > kMaxM || !supported(k) || segments == nullptr || count == 0 ||
        count > kMaxSegments) {
        return cudaErrorInvalidValue;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const Segment &s = segments[i];
        if (s.q == nullptr || !aligned16(s.q) || s.scale == nullptr || s.y == nullptr || s.n == 0) {
            return cudaErrorInvalidValue;
        }
    }
    const int sms = sm_count();
    if (sms == 0) {
        return cudaErrorInvalidValue;
    }
    // 16-column tiles when they give every SM a CTA, else 8 (no output bit depends on it).
    const uint64_t wide = tiles_of(segments, count, 16);
    const uint32_t tile_n = wide >= static_cast<uint64_t>(sms) ? 16 : 8;
    const uint64_t tiles = tile_n == 16 ? wide : tiles_of(segments, count, 8);
    if (tiles > 0x7fffffffu) {
        return cudaErrorInvalidValue;
    }
    LaunchSegments segs{};
    uint32_t begin = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const Segment &s = segments[i];
        segs.s[i] = LaunchSegment{reinterpret_cast<const uint4 *>(s.q), s.scale, s.y, s.n, begin};
        begin += (s.n + tile_n - 1) / tile_n;
    }
    segs.count = count;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(static_cast<uint32_t>(tiles), 1, 1);
    config.blockDim = dim3(32 * split_warps(k), 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;
    const auto *xv = reinterpret_cast<const uint4 *>(x);
    const bool hi = m > 8;
    cudaError_t e = cudaSuccess;
    if (tile_n == 16) {
        e = hi ? cudaLaunchKernelEx(&config, fp8_kernel<O, 16, true>, segs, xv, m, k)
               : cudaLaunchKernelEx(&config, fp8_kernel<O, 16, false>, segs, xv, m, k);
    } else {
        e = hi ? cudaLaunchKernelEx(&config, fp8_kernel<O, 8, true>, segs, xv, m, k)
               : cudaLaunchKernelEx(&config, fp8_kernel<O, 8, false>, segs, xv, m, k);
    }
    const cudaError_t last = cudaGetLastError();
    return e != cudaSuccess ? e : last;
}

}  // namespace

bool supported(uint32_t k) { return k > 0 && k % kStepK == 0; }

// About kStepsPerWarp steps per slice (8 warps at k = 2,560, so four CTAs share an SM), but at least the slices that
// give each warp one load batch while that needs at most 4 warps (a small k stays latency-bound: k = 640 runs 3).
uint32_t split_warps(uint32_t k) {
    const uint32_t steps = k / kStepK;
    const uint32_t one_batch = std::min((steps + kUnroll - 1) / kUnroll, uint32_t{4});
    return std::clamp(std::max(steps / kStepsPerWarp, one_batch), uint32_t{1}, kMaxWarps);
}

cudaError_t gemm(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const Segment *segments, uint32_t count, bool f32,
                 cudaStream_t stream) {
    return f32 ? launch_gemm<kOutF32>(x, m, k, segments, count, stream)
               : launch_gemm<kOutBf16>(x, m, k, segments, count, stream);
}

cudaError_t quantize_rows(const __nv_bfloat16 *w, uint32_t rows, uint32_t k, uint8_t *q, float *scale,
                          cudaStream_t stream) {
    if (w == nullptr || q == nullptr || scale == nullptr || rows == 0 || k == 0 || rows > 0x7fffffffu) {
        return cudaErrorInvalidValue;
    }
    quantize_kernel<<<rows, kQuantThreads, 0, stream>>>(w, k, q, scale);
    return cudaGetLastError();
}

cudaError_t dequantize(const uint8_t *q, const float *scale, uint32_t n, uint32_t k, __nv_bfloat16 *w,
                       cudaStream_t stream) {
    if (q == nullptr || scale == nullptr || w == nullptr || !aligned16(q) || !aligned16(w) || n == 0 ||
        !supported(k)) {
        return cudaErrorInvalidValue;
    }
    const size_t chunks = size_t{n} * k / 16;
    const size_t blocks = (chunks + kDequantThreads - 1) / kDequantThreads;
    if (blocks > 0x7fffffffu) {
        return cudaErrorInvalidValue;
    }
    dequant_kernel<<<static_cast<uint32_t>(blocks), kDequantThreads, 0, stream>>>(
        reinterpret_cast<const uint4 *>(q), scale, k, chunks, reinterpret_cast<uint4 *>(w));
    return cudaGetLastError();
}

cudaError_t convert_in_place(void *storage, uint32_t n, uint32_t k, void *scratch, size_t scratch_bytes,
                             cudaStream_t stream) {
    if (storage == nullptr || scratch == nullptr || n == 0 || !supported(k) || !aligned16(storage) ||
        !aligned16(scratch)) {
        return cudaErrorInvalidValue;
    }
    const size_t scale_bytes = (size_t{n} * 4 + 255) / 256 * 256;
    const size_t row_bytes = size_t{k} * 2;
    if (scratch_bytes < scale_bytes + row_bytes) {
        return cudaErrorInvalidValue;
    }
    const size_t chunk = std::min<size_t>(n, (scratch_bytes - scale_bytes) / row_bytes);
    auto *scales = static_cast<float *>(scratch);
    auto *stage = reinterpret_cast<__nv_bfloat16 *>(static_cast<uint8_t *>(scratch) + scale_bytes);
    auto *bytes = static_cast<uint8_t *>(storage);
    for (size_t r0 = 0; r0 < n; r0 += chunk) {
        const size_t rows = std::min<size_t>(chunk, n - r0);
        if (const cudaError_t e =
                cudaMemcpyAsync(stage, bytes + r0 * row_bytes, rows * row_bytes, cudaMemcpyDeviceToDevice, stream);
            e != cudaSuccess) {
            return e;
        }
        if (const cudaError_t e =
                quantize_rows(stage, static_cast<uint32_t>(rows), k, bytes + r0 * k, scales + r0, stream);
            e != cudaSuccess) {
            return e;
        }
    }
    return cudaMemcpyAsync(bytes + size_t{n} * k, scales, size_t{n} * 4, cudaMemcpyDeviceToDevice, stream);
}

void Table::add(const void *w, uint32_t n, uint32_t k) {
    const auto at = std::lower_bound(entries_.begin(), entries_.end(), w, [](const Entry &e, const void *p) {
        return std::less<const void *>{}(e.w, p);
    });
    entries_.insert(at, Entry{w, n, k});
}

const Entry *Table::find(const void *w) const {
    const auto at = std::lower_bound(entries_.begin(), entries_.end(), w, [](const Entry &e, const void *p) {
        return std::less<const void *>{}(e.w, p);
    });
    return at != entries_.end() && at->w == w ? &*at : nullptr;
}

}  // namespace cere::kernels::k2f
