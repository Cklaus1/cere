// P4 tensor-core prefill attention (m23 design note P4); contracts in p4_attention.h.
//
// One block of 4 warps per (row tile, KV head). A row tile is kRows = 16 * kMT (query token, query head) rows of one
// KV head: the sparse kernel takes one query token (12 heads in one 16-row MMA tile, as SGLang's BLOCK_M = 16), the
// dense kernel 32 consecutive rows of the flattened [tokens, 12] grid (2 m-tiles; a token may span two blocks). Keys
// stream in tiles of 32 through a two-stage cp.async ring (K and V rows of the KV head, BF16 or the raw FP8 bytes;
// invalid keys are zero-filled). Per tile:
//   S phase   warp w: scores of keys 8w .. 8w + 7 for every row (m16n8k16 over the 256 dims), scaled to base 2 and
//             masked, to shared memory;
//   softmax   4 threads per row: tile max, alpha, p = 2^(s - m) (F32), l, bf16(p) to shared memory;
//   P.V phase warp w: dims 64w .. 64w + 63 of every row, o = o * alpha + bf16(p) . v (m16n8k16).
// FP8 E4M3 K/V bytes are widened to BF16 exactly (cvt e4m3x2 -> f16x2 -> bf16x2) when the fragments are loaded.
//
// math_reference: sglang
#include "kernels/p4_attention.h"

#include <cmath>

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace cere::p4 {
namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / kWarp;
constexpr int kGroup = kQHeads / kKvHeads;  // query heads per KV head (12)
constexpr int kQWidth = kQHeads * kHeadDim;
constexpr int kKvWidth = kKvHeads * kHeadDim;
constexpr int kKeyTile = 32;                     // keys per tile; S phase: warp w owns keys 8w .. 8w + 7
constexpr int kDimsPerWarp = kHeadDim / kWarps;  // P.V phase: warp w owns dims 64w .. 64w + 63
constexpr int kQStride = kHeadDim + 8;           // BF16 per Q row in shared memory (528 B: ldmatrix rows hit 8 banks)
constexpr int kSStride = kKeyTile + 1;           // F32 per score row
constexpr int kPStride = kKeyTile + 8;           // BF16 per probability row (80 B)
constexpr int kSparseMT = 1;
constexpr int kDenseMT = 2;
constexpr float kLog2e = 1.4426950408889634f;

static_assert(kKeyTile == 8 * kWarps && kDimsPerWarp == 64 && kGroup == 12 && kHeadDim == 256,
              "the phases assume 4 warps, 32-key tiles, 12 query heads per KV head and head_dim 256");

template <int kMT, typename KvT>
struct Smem {
    static constexpr int kRows = 16 * kMT;
    static constexpr int kRowBytes = kHeadDim * static_cast<int>(sizeof(KvT)) + 16;  // padded K/V row
    static constexpr int kChunks = kHeadDim * static_cast<int>(sizeof(KvT)) / 16;    // 16-byte chunks per K/V row
    static constexpr size_t kQ = static_cast<size_t>(kRows) * kQStride * sizeof(bf16);
    static constexpr size_t kTile = static_cast<size_t>(kKeyTile) * kRowBytes;
    static constexpr size_t kS = static_cast<size_t>(kRows) * kSStride * sizeof(float);
    static constexpr size_t kP = static_cast<size_t>(kRows) * kPStride * sizeof(bf16);
    static constexpr size_t kOffKv = kQ;  // stage s: K tile at kOffKv + 2s * kTile, V tile right after it
    static constexpr size_t kOffS = kOffKv + 4 * kTile;
    static constexpr size_t kOffP = kOffS + kS;
    static constexpr size_t kOffStats = kOffP + kP;  // m, l, alpha [kRows] F32, then the sparse row's key end
    static constexpr size_t kBytes = kOffStats + 3 * static_cast<size_t>(kRows) * sizeof(float) + 16;
    static_assert(kOffKv % 16 == 0 && kTile % 16 == 0 && kOffS % 16 == 0 && kOffP % 16 == 0, "16-byte alignment");
};

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int kPending>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// D (16x8 F32) += A (16x16 BF16, row) * B (16x8 BF16, col).
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two FP8 E4M3 bytes (low byte first) -> BF16x2, exact (every E4M3 value is a BF16 value).
__device__ __forceinline__ uint32_t fp8x2_to_bf16x2(uint32_t pair) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3);
    const float2 f = __half22float2(__half2(h));
    const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
    return *reinterpret_cast<const uint32_t*>(&b);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// The rows a block covers and their key set.
struct Rows {
    int64_t t;         // sparse: the query token; dense: the first flattened row of the block
    const int32_t* idx;  // sparse: the token's index row
    int64_t key_end;   // sparse: last valid entry + 1; dense: pos0 + last token + 1
};

// Query token and query head (within the KV head's group) of block row r; false for a padding row.
template <bool kDense>
__device__ __forceinline__ bool row_of(const Rows& rs, int64_t tokens, int r, int64_t* t, int* hr) {
    if constexpr (kDense) {
        const int64_t f = rs.t + r;
        *t = f / kGroup;
        *hr = static_cast<int>(f % kGroup);
        return *t < tokens;
    } else {
        *t = rs.t;
        *hr = r;
        return r < kGroup;
    }
}

// Issues the cp.async copies of key tile j0 .. j0 + 31 (K and V rows of KV head g) into `stage`.
template <int kMT, typename KvT, bool kDense>
__device__ __forceinline__ void load_tile(uint32_t base, int stage, int64_t j0, const Rows& rs, const KvT* k,
                                          int64_t k_stride, const KvT* v, int64_t v_stride, const int64_t* slots,
                                          int g) {
    using S = Smem<kMT, KvT>;
    constexpr int kElemsPerChunk = 16 / static_cast<int>(sizeof(KvT));
    const uint32_t kdst = base + static_cast<uint32_t>(S::kOffKv + 2 * stage * S::kTile);
    const uint32_t vdst = kdst + static_cast<uint32_t>(S::kTile);
    for (int c = threadIdx.x; c < kKeyTile * S::kChunks; c += kThreads) {
        const int jj = c / S::kChunks;
        const int ch = c % S::kChunks;
        const int64_t j = j0 + jj;
        int64_t pos = -1;
        if (j < rs.key_end) pos = kDense ? j : static_cast<int64_t>(rs.idx[j]);
        const bool valid = pos >= 0;
        const int64_t row = valid ? (slots ? slots[pos] : pos) : 0;
        const int64_t off = static_cast<int64_t>(g) * kHeadDim + ch * kElemsPerChunk;
        const uint32_t dst = static_cast<uint32_t>(jj * S::kRowBytes + ch * 16);
        cp_async16(kdst + dst, valid ? static_cast<const void*>(k + row * k_stride + off) : k, valid);
        cp_async16(vdst + dst, valid ? static_cast<const void*>(v + row * v_stride + off) : v, valid);
    }
}

// Grid (row tiles, kKvHeads). Sparse: row tile = query token blockIdx.x (indices row of row_width entries); dense:
// flattened rows 32 * blockIdx.x .. + 31 of [tokens, 12], keys 0 .. pos0 + t causal.
template <int kMT, typename KvT, bool kDense>
__global__ void __launch_bounds__(kThreads) attention_kernel(const bf16* __restrict__ q, int64_t q_stride,
                                                             int64_t tokens, int64_t pos0,
                                                             const int32_t* __restrict__ indices, int64_t row_width,
                                                             const KvT* __restrict__ k, int64_t k_stride,
                                                             const KvT* __restrict__ v, int64_t v_stride,
                                                             const int64_t* __restrict__ slots, float score_scale,
                                                             bf16* __restrict__ out, int64_t out_stride) {
    using S = Smem<kMT, KvT>;
    constexpr int kRows = S::kRows;
    extern __shared__ __align__(16) unsigned char smem[];
    const uint32_t base = smem_u32(smem);
    float* s_score = reinterpret_cast<float*>(smem + S::kOffS);
    bf16* s_p = reinterpret_cast<bf16*>(smem + S::kOffP);
    float* s_m = reinterpret_cast<float*>(smem + S::kOffStats);
    float* s_l = s_m + kRows;
    float* s_alpha = s_l + kRows;
    int* s_end = reinterpret_cast<int*>(s_alpha + kRows);

    const int tid = threadIdx.x;
    const int warp = tid / kWarp;
    const int lane = tid % kWarp;
    const int g = blockIdx.y;

    Rows rs{};
    if constexpr (kDense) {
        rs.t = static_cast<int64_t>(blockIdx.x) * kRows;
        int64_t last = (rs.t + kRows - 1) / kGroup;
        if (last > tokens - 1) last = tokens - 1;
        rs.key_end = pos0 + last + 1;
        rs.idx = nullptr;
    } else {
        rs.t = blockIdx.x;
        rs.idx = indices + rs.t * row_width;
        if (tid == 0) *s_end = 0;
        __syncthreads();
        int last = 0;
        for (int64_t j = tid; j < row_width; j += kThreads)
            if (rs.idx[j] >= 0) last = static_cast<int>(j) + 1;
#pragma unroll
        for (int mask = kWarp / 2; mask >= 1; mask >>= 1) {
            const int o = __shfl_xor_sync(0xffffffffu, last, mask, kWarp);
            last = o > last ? o : last;
        }
        if (lane == 0) atomicMax(s_end, last);
    }
    for (int r = tid; r < kRows; r += kThreads) {
        s_m[r] = -INFINITY;
        s_l[r] = 0.0f;
    }
    __syncthreads();
    if constexpr (!kDense) rs.key_end = *s_end;

    // Q rows (zeros for padding rows), in the first cp.async group with key tile 0.
    for (int c = tid; c < kRows * (kHeadDim / 8); c += kThreads) {
        const int r = c / (kHeadDim / 8);
        const int ch = c % (kHeadDim / 8);
        int64_t t;
        int hr;
        const bool valid = row_of<kDense>(rs, tokens, r, &t, &hr);
        const bf16* src = valid ? q + t * q_stride + static_cast<int64_t>(g * kGroup + hr) * kHeadDim + ch * 8 : q;
        cp_async16(base + static_cast<uint32_t>((r * kQStride + ch * 8) * sizeof(bf16)), src, valid);
    }
    const int tiles = static_cast<int>((rs.key_end + kKeyTile - 1) / kKeyTile);
    if (tiles > 0) load_tile<kMT, KvT, kDense>(base, 0, 0, rs, k, k_stride, v, v_stride, slots, g);
    cp_async_commit();

    float o[kMT][kDimsPerWarp / 8][4];
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
        for (int nt = 0; nt < kDimsPerWarp / 8; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) o[mt][nt][e] = 0.0f;

    // ldmatrix x4 row address of lane L: rows (L % 8) + 8 * ((L / 8) % 2), column block 8 * (L / 16).
    const int lm_row = (lane % 8) + 8 * ((lane / 8) % 2);
    const int lm_col = 8 * (lane / 16);

    for (int i = 0; i < tiles; ++i) {
        const int stage = i & 1;
        const int64_t j0 = static_cast<int64_t>(i) * kKeyTile;
        if (i + 1 < tiles)
            load_tile<kMT, KvT, kDense>(base, stage ^ 1, j0 + kKeyTile, rs, k, k_stride, v, v_stride, slots, g);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();

        const unsigned char* k_tile = smem + S::kOffKv + 2 * stage * S::kTile;
        const unsigned char* v_tile = k_tile + S::kTile;

        {  // S phase: keys 8 warp .. 8 warp + 7 of the tile for every row
            float acc[kMT][4];
#pragma unroll
            for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[mt][e] = 0.0f;
#pragma unroll 2
            for (int kk = 0; kk < kHeadDim; kk += 32) {
                uint32_t b[4];  // B fragments of dims kk .. kk + 15 (b[0], b[1]) and kk + 16 .. kk + 31 (b[2], b[3])
                if constexpr (sizeof(KvT) == 2) {
                    // matrices (8 keys x 8 dims) at dims kk, kk + 8, kk + 16, kk + 24
                    ldmatrix_x4(b, base + static_cast<uint32_t>(k_tile - smem) +
                                       static_cast<uint32_t>((8 * warp + lane % 8) * S::kRowBytes +
                                                             (kk + 8 * (lane / 8)) * 2));
                } else {
                    const unsigned char* kr = k_tile + (8 * warp + lane / 4) * S::kRowBytes + kk + 2 * (lane % 4);
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        b[e] = fp8x2_to_bf16x2(*reinterpret_cast<const uint16_t*>(kr + 8 * e));
                }
#pragma unroll
                for (int mt = 0; mt < kMT; ++mt) {
                    uint32_t a[4];
                    const uint32_t qa = base + static_cast<uint32_t>(((16 * mt + lm_row) * kQStride + kk + lm_col) * 2);
                    ldmatrix_x4(a, qa);
                    mma_bf16(acc[mt], a, b[0], b[1]);
                    ldmatrix_x4(a, qa + 32);
                    mma_bf16(acc[mt], a, b[2], b[3]);
                }
            }
            const int col = 8 * warp + 2 * (lane % 4);
            bool key_ok[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int64_t j = j0 + col + e;
                key_ok[e] = j < rs.key_end && (kDense || rs.idx[j] >= 0);
            }
#pragma unroll
            for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = 16 * mt + lane / 4 + 8 * h;
                    int64_t limit = INT64_MAX;
                    if constexpr (kDense) {
                        int64_t t;
                        int hr;
                        row_of<kDense>(rs, tokens, r, &t, &hr);
                        limit = pos0 + (t < tokens ? t : tokens - 1);
                    }
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const bool ok = key_ok[e] && j0 + col + e <= limit;
                        s_score[r * kSStride + col + e] = ok ? acc[mt][2 * h + e] * score_scale : -INFINITY;
                    }
                }
            }
        }
        __syncthreads();

        {  // online softmax: 4 threads per row, 8 keys each
            const int r = tid / 4;
            const int qd = tid % 4;
            if (r < kRows) {  // warp-uniform (8 rows per warp)
                const float* srow = s_score + r * kSStride + 8 * qd;
                float sv[8];
                float mx = -INFINITY;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    sv[e] = srow[e];
                    mx = fmaxf(mx, sv[e]);
                }
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1, kWarp));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2, kWarp));
                const float m_old = s_m[r];
                const float m_new = fmaxf(m_old, mx);
                float alpha = 1.0f;
                float sum = 0.0f;
                float p[8];
                if (m_new == -INFINITY) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) p[e] = 0.0f;
                } else {
                    alpha = exp2f(m_old - m_new);
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        p[e] = exp2f(sv[e] - m_new);
                        sum += p[e];
                    }
                }
                bf16* prow = s_p + r * kPStride + 8 * qd;
#pragma unroll
                for (int e = 0; e < 8; e += 2)
                    *reinterpret_cast<__nv_bfloat162*>(prow + e) = __floats2bfloat162_rn(p[e], p[e + 1]);
                sum += __shfl_xor_sync(0xffffffffu, sum, 1, kWarp);
                sum += __shfl_xor_sync(0xffffffffu, sum, 2, kWarp);
                __syncwarp();
                if (qd == 0) {
                    s_m[r] = m_new;
                    s_l[r] = s_l[r] * alpha + sum;
                    s_alpha[r] = alpha;
                }
            }
        }
        __syncthreads();

        {  // P.V phase: dims 64 warp .. 64 warp + 63 of every row
#pragma unroll
            for (int mt = 0; mt < kMT; ++mt) {
                const float a0 = s_alpha[16 * mt + lane / 4];
                const float a1 = s_alpha[16 * mt + lane / 4 + 8];
#pragma unroll
                for (int nt = 0; nt < kDimsPerWarp / 8; ++nt) {
                    o[mt][nt][0] *= a0;
                    o[mt][nt][1] *= a0;
                    o[mt][nt][2] *= a1;
                    o[mt][nt][3] *= a1;
                }
            }
#pragma unroll
            for (int ks = 0; ks < kKeyTile; ks += 16) {
                uint32_t a[kMT][4];
#pragma unroll
                for (int mt = 0; mt < kMT; ++mt)
                    ldmatrix_x4(a[mt], base + static_cast<uint32_t>(S::kOffP +
                                                                    ((16 * mt + lm_row) * kPStride + ks + lm_col) * 2));
#pragma unroll
                for (int np = 0; np < kDimsPerWarp / 16; ++np) {
                    const int n0 = kDimsPerWarp * warp + 16 * np;
                    uint32_t b[4];  // dims n0 .. n0 + 7: b[0], b[1]; n0 + 8 .. n0 + 15: b[2], b[3]
                    if constexpr (sizeof(KvT) == 2) {
                        ldmatrix_x4_trans(b, base + static_cast<uint32_t>(v_tile - smem) +
                                                 static_cast<uint32_t>((ks + lm_row) * S::kRowBytes + (n0 + lm_col) * 2));
                    } else {
                        const unsigned char* vr = v_tile + (ks + 2 * (lane % 4)) * S::kRowBytes + n0 + lane / 4;
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            // e: (dims + 8 (e / 2), keys + 8 (e % 2))
                            const unsigned char* p0 = vr + (e % 2) * 8 * S::kRowBytes + (e / 2) * 8;
                            b[e] = fp8x2_to_bf16x2(static_cast<uint32_t>(p0[0]) |
                                                   (static_cast<uint32_t>(p0[S::kRowBytes]) << 8));
                        }
                    }
#pragma unroll
                    for (int mt = 0; mt < kMT; ++mt) {
                        mma_bf16(o[mt][2 * np], a[mt], b[0], b[1]);
                        mma_bf16(o[mt][2 * np + 1], a[mt], b[2], b[3]);
                    }
                }
            }
        }
        __syncthreads();
    }
    cp_async_wait<0>();

#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = 16 * mt + lane / 4 + 8 * h;
            int64_t t;
            int hr;
            if (!row_of<kDense>(rs, tokens, r, &t, &hr)) continue;
            const float l = s_l[r];
            bf16* orow = out + t * out_stride + static_cast<int64_t>(g * kGroup + hr) * kHeadDim;
#pragma unroll
            for (int nt = 0; nt < kDimsPerWarp / 8; ++nt) {
                const int d = kDimsPerWarp * warp + 8 * nt + 2 * (lane % 4);
                const float x0 = l > 0.0f ? o[mt][nt][2 * h] / l : 0.0f;
                const float x1 = l > 0.0f ? o[mt][nt][2 * h + 1] / l : 0.0f;
                *reinterpret_cast<__nv_bfloat162*>(orow + d) = __floats2bfloat162_rn(x0, x1);
            }
        }
    }
}

template <int kMT, typename KvT, bool kDense>
cudaError_t launch(unsigned grid_x, const bf16* q, int64_t q_stride, int64_t tokens, int64_t pos0,
                   const int32_t* indices, int64_t row_width, const KvT* k, int64_t k_stride, const KvT* v,
                   int64_t v_stride, const int64_t* slots, float scale, bf16* out, int64_t out_stride,
                   cudaStream_t stream) {
    constexpr size_t bytes = Smem<kMT, KvT>::kBytes;
    auto kernel = attention_kernel<kMT, KvT, kDense>;
    cudaError_t err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if (err != cudaSuccess) return err;
    kernel<<<dim3(grid_x, kKvHeads), kThreads, bytes, stream>>>(q, q_stride, tokens, pos0, indices, row_width, k,
                                                                k_stride, v, v_stride, slots, scale * kLog2e, out,
                                                                out_stride);
    return cudaGetLastError();
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// The K/V source pair of a call: BF16 rows (strides multiples of 8 elements) or the FP8 cache; exactly one.
bool kv_ok(const bf16* k, int64_t k_stride, const bf16* v, int64_t v_stride, const uint8_t* k_cache,
           const uint8_t* v_cache) {
    const bool bf16_kv = k && v && !k_cache && !v_cache;
    const bool fp8_kv = !k && !v && k_cache && v_cache;
    if (bf16_kv)
        return aligned16(k) && aligned16(v) && k_stride >= kKvWidth && v_stride >= kKvWidth && k_stride % 8 == 0 &&
               v_stride % 8 == 0;
    return fp8_kv && aligned16(k_cache) && aligned16(v_cache);
}

}  // namespace

cudaError_t dense_attention(const bf16* q, int64_t tokens, int64_t pos0, const bf16* k, int64_t k_stride,
                            const bf16* v, int64_t v_stride, const uint8_t* k_cache, const uint8_t* v_cache,
                            const int64_t* slots, float scale, bf16* out, cudaStream_t stream) {
    if (!q || !out || tokens < 1 || pos0 < 0 || !aligned16(q) || !aligned16(out) || !std::isfinite(scale) ||
        !kv_ok(k, k_stride, v, v_stride, k_cache, v_cache))
        return cudaErrorInvalidValue;
    constexpr int kRows = 16 * kDenseMT;
    const int64_t blocks = (tokens * kGroup + kRows - 1) / kRows;
    if (blocks > INT32_MAX) return cudaErrorInvalidValue;
    const unsigned grid = static_cast<unsigned>(blocks);
    if (k)
        return launch<kDenseMT, bf16, true>(grid, q, kQWidth, tokens, pos0, nullptr, 0, k, k_stride, v, v_stride,
                                            nullptr, scale, out, kQWidth, stream);
    return launch<kDenseMT, uint8_t, true>(grid, q, kQWidth, tokens, pos0, nullptr, 0, k_cache, kKvWidth, v_cache,
                                           kKvWidth, slots, scale, out, kQWidth, stream);
}

cudaError_t sparse_attention(const bf16* q, int64_t q_stride, const int32_t* indices, int64_t row_width, int64_t rows,
                             const bf16* k, int64_t k_stride, const bf16* v, int64_t v_stride, const uint8_t* k_cache,
                             const uint8_t* v_cache, const int64_t* slots, float scale, bf16* out, int64_t out_stride,
                             cudaStream_t stream) {
    if (!q || !indices || !out || rows < 1 || rows > INT32_MAX || row_width <= 0 || row_width > INT32_MAX ||
        q_stride < kQWidth || q_stride % 8 != 0 || out_stride < kQWidth || out_stride % 8 != 0 || !aligned16(q) ||
        !aligned16(out) || !std::isfinite(scale) || !kv_ok(k, k_stride, v, v_stride, k_cache, v_cache))
        return cudaErrorInvalidValue;
    const unsigned grid = static_cast<unsigned>(rows);
    if (k)
        return launch<kSparseMT, bf16, false>(grid, q, q_stride, rows, 0, indices, row_width, k, k_stride, v, v_stride,
                                              nullptr, scale, out, out_stride, stream);
    return launch<kSparseMT, uint8_t, false>(grid, q, q_stride, rows, 0, indices, row_width, k_cache, kKvWidth,
                                             v_cache, kKvWidth, slots, scale, out, out_stride, stream);
}

}  // namespace cere::p4
