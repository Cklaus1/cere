// AmdAmd Phase 2, P2-0 (tasks/amd-amd-spec.md): gemma4 full-attention prefill on gfx1151.
//
// One full-attention layer of gemma-4-e4b-it: H = 8 query heads, G = 2 KV heads (GQA 4:1), head dim D = 512,
// causal, n_q new tokens appended after `depth` cached tokens (query i sees keys 0 .. depth + i).
// Layout (token-major, like llama.cpp's KV cache): Q [n_q][H][D], K/V [n_k][G][D] bf16, O [n_q][H][D] f32.
//
//   ref   : CPU double, small shapes only (correctness oracle).
//   unfused: hipBLAS bf16 GEMM (S = QK^T, f32) -> masked softmax (P bf16) -> hipBLAS GEMM (O = PV), the shape of
//            llama.cpp's non-flash-attention path, which is what gemma4 runs on HIP today.
//   fa    : WMMA flash attention (v1), raw gfx11 wave32 intrinsics.
//
// Build (llama-amd:rocm7.2.4): hipcc --offload-arch=gfx1151 -O3 fa_gemma4.hip -lhipblas -o fa_gemma4
// Run: fa_gemma4 [--fa 1|2|3] [--check] [--depths 0,8192,16384,30000] [--nq 2048] [--reps 10]
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)
#define CB(x) do { hipblasStatus_t s_ = (x); if (s_ != HIPBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%d hipblas %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

constexpr int H = 8, G = 2, D = 512, HPG = H / G;
typedef __bf16 bf16;
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

// ---------------------------------------------------------------- unfused baseline
__global__ void softmax_causal(const float *S, bf16 *P, int n_q, int n_k, int depth, float scale) {
    // one block per (row, head-in-group); S/P [hh][n_q][n_k]
    const int q = blockIdx.x, hh = blockIdx.y;
    const float *s = S + ((size_t)hh * n_q + q) * n_k;
    bf16 *p = P + ((size_t)hh * n_q + q) * n_k;
    const int lim = depth + q;  // last visible key
    __shared__ float red[32];
    float m = -INFINITY;
    for (int k = threadIdx.x; k <= lim; k += blockDim.x) m = fmaxf(m, s[k] * scale);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = red[0];
    for (int w = 1; w < (int)blockDim.x / 32; ++w) m = fmaxf(m, red[w]);
    __syncthreads();
    float sum = 0.f;
    for (int k = threadIdx.x; k <= lim; k += blockDim.x) sum += __expf(s[k] * scale - m);
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    sum = 0.f;
    for (int w = 0; w < (int)blockDim.x / 32; ++w) sum += red[w];
    const float inv = 1.f / sum;
    for (int k = threadIdx.x; k < n_k; k += blockDim.x) p[k] = (bf16)(k <= lim ? __expf(s[k] * scale - m) * inv : 0.f);
}

struct Unfused {
    hipblasHandle_t h{};
    float *S{};
    bf16 *P{};
    size_t cap{};
    void ensure(size_t elems) {
        if (elems <= cap) return;
        if (S) { CK(hipFree(S)); CK(hipFree(P)); }
        CK(hipMalloc(&S, elems * sizeof(float)));
        CK(hipMalloc(&P, elems * sizeof(bf16)));
        cap = elems;
    }
    void run(const bf16 *Q, const bf16 *K, const bf16 *V, float *O, int n_q, int n_k, int depth, float scale) {
        ensure((size_t)HPG * n_q * n_k);
        const float one = 1.f, zero = 0.f;
        for (int g = 0; g < G; ++g) {
            // S^T (col-major n_k x n_q) = K_g (op T of col-major D x n_k) * Q_h (col-major D x n_q); batch = 4 heads
            CB(hipblasGemmStridedBatchedEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, n_k, n_q, D, &one,
                                           K + g * D, HIP_R_16BF, G * D, 0,
                                           Q + g * HPG * D, HIP_R_16BF, H * D, D, &zero,
                                           S, HIP_R_32F, n_k, (long long)n_q * n_k, HPG,
                                           HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
            softmax_causal<<<dim3(n_q, HPG), 256>>>(S, P, n_q, n_k, depth, scale);
            // O^T (col-major D x n_q, ld H*D) = V_g (col-major D x n_k) * P^T (col-major n_k x n_q)
            CB(hipblasGemmStridedBatchedEx(h, HIPBLAS_OP_N, HIPBLAS_OP_N, D, n_q, n_k, &one,
                                           V + g * D, HIP_R_16BF, G * D, 0,
                                           P, HIP_R_16BF, n_k, (long long)n_q * n_k, &zero,
                                           O + g * HPG * D, HIP_R_32F, H * D, D, HPG,
                                           HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        }
    }
};

// ---------------------------------------------------------------- WMMA flash attention (v1)
// Block = 4 waves (128 threads) for 16 query rows of one head; wave w owns head-dim slice [128w, 128w+128).
// Per 32-key block: each wave computes a partial S over its slice (16 WMMAs), partials are summed in LDS,
// wave 0 lanes 0..15 run the online softmax for one row each, then each wave does O_slice += P V_slice (16 WMMAs)
// with V staged in LDS (so B fragments, which need 16 keys of one d column per lane, read LDS, not strided HBM).
constexpr int BM = 16, BN = 32, WAVES = 4, DS = D / WAVES;  // DS = 128

__global__ __launch_bounds__(128) void fa_fwd(const bf16 *__restrict__ Q, const bf16 *__restrict__ K,
                                              const bf16 *__restrict__ V, float *__restrict__ O,
                                              int n_q, int n_k, int depth, float scale) {
    const int q0 = blockIdx.x * BM, h = blockIdx.y, g = h / HPG;
    const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, r = l & 15, hi = l >> 4;
    const int ws = w * DS;

    __shared__ float sPart[WAVES][BM][BN];  // 8 KB
    __shared__ float sS[BM][BN];            // 2 KB
    __shared__ bf16 sP[BM][BN];             // 1 KB
    __shared__ float sAlpha[BM], sL[BM];
    __shared__ bf16 sV[WAVES][BN][DS + 8];  // 4 x 32 x 136 x 2 = 34.8 KB (pad against bank conflicts)

    // Q fragments for this wave's slice: 8 k-steps of 16 dims; lane holds row r.
    bf16x16 qf[DS / 16];
    {
        const int qrow = min(q0 + r, n_q - 1);
        const bf16 *qp = Q + ((size_t)qrow * H + h) * D + ws;
#pragma unroll
        for (int ks = 0; ks < DS / 16; ++ks) qf[ks] = *(const bf16x16 *)(qp + ks * 16);
    }
    f32x8 acc[DS / 16];
#pragma unroll
    for (int nb = 0; nb < DS / 16; ++nb) acc[nb] = f32x8{};
    float m_run = -INFINITY, l_run = 0.f;  // wave 0, lanes 0..15: row r

    const int last_key = min(n_k - 1, depth + min(q0 + BM, n_q) - 1);
    for (int k0 = 0; k0 <= last_key; k0 += BN) {
        // stage this wave's V slice for keys k0..k0+31 (coalesced: each lane copies 8 bytes per pass)
        for (int i = l; i < BN * DS / 4; i += 32) {
            const int key = i / (DS / 4), c4 = (i % (DS / 4)) * 4;
            const int kk = min(k0 + key, n_k - 1);
            *(uint2 *)&sV[w][key][c4] = *(const uint2 *)(V + ((size_t)kk * G + g) * D + ws + c4);
        }
        // partial S over this slice: two 16-key n-blocks
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int key = min(k0 + j * 16 + r, n_k - 1);
            const bf16 *kp = K + ((size_t)key * G + g) * D + ws;
            f32x8 s = {};
#pragma unroll
            for (int ks = 0; ks < DS / 16; ++ks)
                s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf[ks], *(const bf16x16 *)(kp + ks * 16), s);
#pragma unroll
            for (int i = 0; i < 8; ++i) sPart[w][2 * i + hi][j * 16 + r] = s[i];
        }
        __syncthreads();
        // reduce partials, scale, mask
        for (int i = tid; i < BM * BN; i += 128) {
            const int row = i / BN, col = i % BN, key = k0 + col, qpos = q0 + row;
            const float v = (sPart[0][row][col] + sPart[1][row][col] + sPart[2][row][col] + sPart[3][row][col]) * scale;
            sS[row][col] = (key <= depth + qpos && key < n_k) ? v : -INFINITY;
        }
        __syncthreads();
        if (w == 0 && l < BM) {  // online softmax, one row per lane
            float mx = m_run;
            for (int c = 0; c < BN; ++c) mx = fmaxf(mx, sS[l][c]);
            const float alpha = (mx == -INFINITY) ? 1.f : __expf(m_run - mx);
            float sum = 0.f;
            for (int c = 0; c < BN; ++c) {
                const float p = (mx == -INFINITY) ? 0.f : __expf(sS[l][c] - mx);
                sP[l][c] = (bf16)p;
                sum += p;
            }
            l_run = l_run * alpha + sum;
            m_run = mx;
            sAlpha[l] = alpha;
        }
        __syncthreads();
        // rescale this wave's O slice and accumulate P V
        float a8[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) a8[i] = sAlpha[2 * i + hi];
        bf16x16 pf[2];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
#pragma unroll
            for (int k = 0; k < 16; ++k) pf[ks][k] = sP[r][ks * 16 + k];
#pragma unroll
        for (int nb = 0; nb < DS / 16; ++nb) {
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[nb][i] *= a8[i];
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                bf16x16 vf;
#pragma unroll
                for (int k = 0; k < 16; ++k) vf[k] = sV[w][ks * 16 + k][nb * 16 + r];
                acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf[ks], vf, acc[nb]);
            }
        }
        __syncthreads();  // sV / sP reuse
    }
    if (w == 0 && l < BM) sL[l] = l_run;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = 2 * i + hi, qpos = q0 + row;
        if (qpos >= n_q) continue;
        const float inv = 1.f / sL[row];
        float *op = O + ((size_t)qpos * H + h) * D + ws;
#pragma unroll
        for (int nb = 0; nb < DS / 16; ++nb) op[nb * 16 + r] = acc[nb][i] * inv;
    }
}

// ---------------------------------------------------------------- WMMA flash attention (v2)
// Block = one GQA group (4 heads) x 16 query rows = 64 rows, 16 waves (512 threads): wave (hh, w) owns head hh's
// 16 rows and head-dim slice [128w, 128w+128). K/V of the group are read once per 4 heads; partial S tiles are
// summed with LDS float atomics; the online softmax runs on all 512 threads (8 per row).
constexpr int B2_ROWS = HPG * BM;  // 64

__global__ __launch_bounds__(512) void fa_fwd2(const bf16 *__restrict__ Q, const bf16 *__restrict__ K,
                                               const bf16 *__restrict__ V, float *__restrict__ O,
                                               int n_q, int n_k, int depth, float scale) {
    const int q0 = blockIdx.x * BM, g = blockIdx.y;
    const int tid = threadIdx.x, wid = tid >> 5, l = tid & 31, r = l & 15, hi = l >> 4;
    const int hh = wid >> 2, w = wid & 3, h = g * HPG + hh, ws = w * DS;

    __shared__ float sS[B2_ROWS][BN];          // 8 KB, rows = hh * 16 + row
    __shared__ bf16 sP[B2_ROWS][BN];           // 4 KB
    __shared__ float sAlpha[B2_ROWS], sL[B2_ROWS];
    __shared__ bf16 sV[WAVES][BN][DS + 8];     // 34.8 KB, shared by the 4 heads

    bf16x16 qf[DS / 16];
    {
        const int qrow = min(q0 + r, n_q - 1);
        const bf16 *qp = Q + ((size_t)qrow * H + h) * D + ws;
#pragma unroll
        for (int ks = 0; ks < DS / 16; ++ks) qf[ks] = *(const bf16x16 *)(qp + ks * 16);
    }
    f32x8 acc[DS / 16];
#pragma unroll
    for (int nb = 0; nb < DS / 16; ++nb) acc[nb] = f32x8{};
    // softmax ownership: thread t handles row t/8, columns (t%8)*4 .. +4
    const int srow = tid >> 3, sc = (tid & 7) * 4;
    float m_run = -INFINITY, l_run = 0.f;

    const int last_key = min(n_k - 1, depth + min(q0 + BM, n_q) - 1);
    for (int k0 = 0; k0 <= last_key; k0 += BN) {
        for (int i = tid; i < B2_ROWS * BN; i += 512) (&sS[0][0])[i] = 0.f;
        // stage V slices (the 4 head-waves of slice w split the copy)
        for (int i = (hh * 32 + l); i < BN * DS / 4; i += 128) {
            const int key = i / (DS / 4), c4 = (i % (DS / 4)) * 4;
            const int kk = min(k0 + key, n_k - 1);
            *(uint2 *)&sV[w][key][c4] = *(const uint2 *)(V + ((size_t)kk * G + g) * D + ws + c4);
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int key = min(k0 + j * 16 + r, n_k - 1);
            const bf16 *kp = K + ((size_t)key * G + g) * D + ws;
            f32x8 s = {};
#pragma unroll
            for (int ks = 0; ks < DS / 16; ++ks)
                s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf[ks], *(const bf16x16 *)(kp + ks * 16), s);
#pragma unroll
            for (int i = 0; i < 8; ++i) atomicAdd(&sS[hh * BM + 2 * i + hi][j * 16 + r], s[i]);
        }
        __syncthreads();
        {   // online softmax, 8 threads per row
            const int qpos = q0 + (srow & (BM - 1));
            float v[4], mx = -INFINITY;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int key = k0 + sc + c;
                v[c] = (key <= depth + qpos && key < n_k) ? sS[srow][sc + c] * scale : -INFINITY;
                mx = fmaxf(mx, v[c]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor(mx, o));
            mx = fmaxf(mx, m_run);
            float sum = 0.f;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float p = (mx == -INFINITY) ? 0.f : __expf(v[c] - mx);
                sP[srow][sc + c] = (bf16)p;
                sum += p;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor(sum, o);
            const float alpha = (mx == -INFINITY) ? 1.f : __expf(m_run - mx);
            l_run = l_run * alpha + sum;
            m_run = mx;
            if ((tid & 7) == 0) sAlpha[srow] = alpha;
        }
        __syncthreads();
        float a8[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) a8[i] = sAlpha[hh * BM + 2 * i + hi];
        bf16x16 pf[2];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
#pragma unroll
            for (int k = 0; k < 16; ++k) pf[ks][k] = sP[hh * BM + r][ks * 16 + k];
#pragma unroll
        for (int nb = 0; nb < DS / 16; ++nb) {
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[nb][i] *= a8[i];
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                bf16x16 vf;
#pragma unroll
                for (int k = 0; k < 16; ++k) vf[k] = sV[w][ks * 16 + k][nb * 16 + r];
                acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf[ks], vf, acc[nb]);
            }
        }
        __syncthreads();
    }
    if ((tid & 7) == 0) sL[srow] = l_run;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = 2 * i + hi, qpos = q0 + row;
        if (qpos >= n_q) continue;
        const float inv = 1.f / sL[hh * BM + row];
        float *op = O + ((size_t)qpos * H + h) * D + ws;
#pragma unroll
        for (int nb = 0; nb < DS / 16; ++nb) op[nb * 16 + r] = acc[nb][i] * inv;
    }
}

// ---------------------------------------------------------------- WMMA flash attention (v3: v2 + V transposed in LDS, vector fragment reads)
// Block = one GQA group (4 heads) x 16 query rows = 64 rows, 16 waves (512 threads): wave (hh, w) owns head hh's
// 16 rows and head-dim slice [128w, 128w+128). K/V of the group are read once per 4 heads; partial S tiles are
// summed with LDS float atomics; the online softmax runs on all 512 threads (8 per row).

__global__ __launch_bounds__(512) void fa_fwd3(const bf16 *__restrict__ Q, const bf16 *__restrict__ K,
                                               const bf16 *__restrict__ V, float *__restrict__ O,
                                               int n_q, int n_k, int depth, float scale) {
    const int q0 = blockIdx.x * BM, g = blockIdx.y;
    const int tid = threadIdx.x, wid = tid >> 5, l = tid & 31, r = l & 15, hi = l >> 4;
    const int hh = wid >> 2, w = wid & 3, h = g * HPG + hh, ws = w * DS;

    __shared__ float sS[B2_ROWS][BN];          // 8 KB, rows = hh * 16 + row
    __shared__ __attribute__((aligned(32))) bf16 sP[B2_ROWS][BN];  // 4 KB
    __shared__ float sAlpha[B2_ROWS], sL[B2_ROWS];
    __shared__ __attribute__((aligned(32))) bf16 sVt[WAVES][DS][BN + 16];  // V^T: 4 x 128 x 48 x 2 = 48 KB

    bf16x16 qf[DS / 16];
    {
        const int qrow = min(q0 + r, n_q - 1);
        const bf16 *qp = Q + ((size_t)qrow * H + h) * D + ws;
#pragma unroll
        for (int ks = 0; ks < DS / 16; ++ks) qf[ks] = *(const bf16x16 *)(qp + ks * 16);
    }
    f32x8 acc[DS / 16];
#pragma unroll
    for (int nb = 0; nb < DS / 16; ++nb) acc[nb] = f32x8{};
    // softmax ownership: thread t handles row t/8, columns (t%8)*4 .. +4
    const int srow = tid >> 3, sc = (tid & 7) * 4;
    float m_run = -INFINITY, l_run = 0.f;

    const int last_key = min(n_k - 1, depth + min(q0 + BM, n_q) - 1);
    for (int k0 = 0; k0 <= last_key; k0 += BN) {
        for (int i = tid; i < B2_ROWS * BN; i += 512) (&sS[0][0])[i] = 0.f;
        // stage V slices (the 4 head-waves of slice w split the copy)
        for (int i = (hh * 32 + l); i < BN * DS / 4; i += 128) {
            const int key = i / (DS / 4), c4 = (i % (DS / 4)) * 4;
            const int kk = min(k0 + key, n_k - 1);
            const uint2 v4 = *(const uint2 *)(V + ((size_t)kk * G + g) * D + ws + c4);
            const bf16 *e = (const bf16 *)&v4;
            sVt[w][c4 + 0][key] = e[0]; sVt[w][c4 + 1][key] = e[1];
            sVt[w][c4 + 2][key] = e[2]; sVt[w][c4 + 3][key] = e[3];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int key = min(k0 + j * 16 + r, n_k - 1);
            const bf16 *kp = K + ((size_t)key * G + g) * D + ws;
            f32x8 s = {};
#pragma unroll
            for (int ks = 0; ks < DS / 16; ++ks)
                s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qf[ks], *(const bf16x16 *)(kp + ks * 16), s);
#pragma unroll
            for (int i = 0; i < 8; ++i) atomicAdd(&sS[hh * BM + 2 * i + hi][j * 16 + r], s[i]);
        }
        __syncthreads();
        {   // online softmax, 8 threads per row
            const int qpos = q0 + (srow & (BM - 1));
            float v[4], mx = -INFINITY;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int key = k0 + sc + c;
                v[c] = (key <= depth + qpos && key < n_k) ? sS[srow][sc + c] * scale : -INFINITY;
                mx = fmaxf(mx, v[c]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor(mx, o));
            mx = fmaxf(mx, m_run);
            float sum = 0.f;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float p = (mx == -INFINITY) ? 0.f : __expf(v[c] - mx);
                sP[srow][sc + c] = (bf16)p;
                sum += p;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor(sum, o);
            const float alpha = (mx == -INFINITY) ? 1.f : __expf(m_run - mx);
            l_run = l_run * alpha + sum;
            m_run = mx;
            if ((tid & 7) == 0) sAlpha[srow] = alpha;
        }
        __syncthreads();
        float a8[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) a8[i] = sAlpha[hh * BM + 2 * i + hi];
        bf16x16 pf[2];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
            pf[ks] = *(const bf16x16 *)&sP[hh * BM + r][ks * 16];
#pragma unroll
        for (int nb = 0; nb < DS / 16; ++nb) {
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[nb][i] *= a8[i];
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const bf16x16 vf = *(const bf16x16 *)&sVt[w][nb * 16 + r][ks * 16];
                acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf[ks], vf, acc[nb]);
            }
        }
        __syncthreads();
    }
    if ((tid & 7) == 0) sL[srow] = l_run;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = 2 * i + hi, qpos = q0 + row;
        if (qpos >= n_q) continue;
        const float inv = 1.f / sL[hh * BM + row];
        float *op = O + ((size_t)qpos * H + h) * D + ws;
#pragma unroll
        for (int nb = 0; nb < DS / 16; ++nb) op[nb * 16 + r] = acc[nb][i] * inv;
    }
}

static int g_fa_version = 3;
static void run_fa(const bf16 *Q, const bf16 *K, const bf16 *V, float *O, int n_q, int n_k, int depth, float scale) {
    if (g_fa_version == 1) fa_fwd<<<dim3((n_q + BM - 1) / BM, H), 128>>>(Q, K, V, O, n_q, n_k, depth, scale);
    else if (g_fa_version == 2) fa_fwd2<<<dim3((n_q + BM - 1) / BM, G), 512>>>(Q, K, V, O, n_q, n_k, depth, scale);
    else fa_fwd3<<<dim3((n_q + BM - 1) / BM, G), 512>>>(Q, K, V, O, n_q, n_k, depth, scale);
}

// ---------------------------------------------------------------- reference + harness
static void ref_attn(const std::vector<bf16> &Q, const std::vector<bf16> &K, const std::vector<bf16> &V,
                     std::vector<float> &O, int n_q, int n_k, int depth, float scale) {
    std::vector<double> s(n_k);
    for (int q = 0; q < n_q; ++q)
        for (int h = 0; h < H; ++h) {
            const int g = h / HPG, lim = depth + q;
            double mx = -1e300;
            for (int k = 0; k <= lim; ++k) {
                double a = 0;
                for (int d = 0; d < D; ++d) a += (double)(float)Q[((size_t)q * H + h) * D + d] * (double)(float)K[((size_t)k * G + g) * D + d];
                s[k] = a * scale;
                mx = std::max(mx, s[k]);
            }
            double sum = 0;
            for (int k = 0; k <= lim; ++k) sum += (s[k] = std::exp(s[k] - mx));
            for (int d = 0; d < D; ++d) {
                double a = 0;
                for (int k = 0; k <= lim; ++k) a += s[k] * (double)(float)V[((size_t)k * G + g) * D + d];
                O[((size_t)q * H + h) * D + d] = (float)(a / sum);
            }
        }
}

struct Buf {
    bf16 *Q, *K, *V;
    float *O1, *O2;
    std::vector<bf16> hQ, hK, hV;
    Buf(int n_q, int n_k, unsigned seed) : hQ((size_t)n_q * H * D), hK((size_t)n_k * G * D), hV((size_t)n_k * G * D) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto &x : hQ) x = (bf16)nd(rng);
        for (auto &x : hK) x = (bf16)nd(rng);
        for (auto &x : hV) x = (bf16)nd(rng);
        CK(hipMalloc(&Q, hQ.size() * 2)); CK(hipMalloc(&K, hK.size() * 2)); CK(hipMalloc(&V, hV.size() * 2));
        CK(hipMalloc(&O1, (size_t)n_q * H * D * 4)); CK(hipMalloc(&O2, (size_t)n_q * H * D * 4));
        CK(hipMemcpy(Q, hQ.data(), hQ.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(K, hK.data(), hK.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(V, hV.data(), hV.size() * 2, hipMemcpyHostToDevice));
    }
    ~Buf() { (void)hipFree(Q); (void)hipFree(K); (void)hipFree(V); (void)hipFree(O1); (void)hipFree(O2); }
};

static double max_rel(const std::vector<float> &a, const std::vector<float> &b) {
    double worst = 0, scale = 0;
    for (float x : b) scale = std::max(scale, (double)std::fabs(x));
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs((double)a[i] - b[i]));
    return worst / std::max(scale, 1e-30);
}

template <class F> static float time_ms(F f, int reps) {
    hipEvent_t a, b;
    CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int i = 0; i < 2; ++i) f();
    CK(hipDeviceSynchronize());
    std::vector<float> t;
    for (int i = 0; i < reps; ++i) {
        CK(hipEventRecord(a)); f(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float ms; CK(hipEventElapsedTime(&ms, a, b)); t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char **argv) {
    bool check = false;
    int n_q = 2048, reps = 10;
    std::vector<int> depths = {0, 8192, 16384, 30000};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--check") check = true;
        else if (a == "--nq" && i + 1 < argc) n_q = atoi(argv[++i]);
        else if (a == "--fa" && i + 1 < argc) g_fa_version = atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (a == "--depths" && i + 1 < argc) {
            depths.clear();
            for (char *t = strtok(argv[++i], ","); t; t = strtok(nullptr, ",")) depths.push_back(atoi(t));
        }
    }
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    printf("device %s, %d CUs; H=%d G=%d D=%d; fa v%d\n", prop.gcnArchName, prop.multiProcessorCount, H, G, D, g_fa_version);
    const float scale = 1.f / std::sqrt((float)D);  // a parameter of the kernel; the value does not affect timing
    Unfused uf;
    CB(hipblasCreate(&uf.h));

    if (check) {  // CPU double oracle on small shapes, including a ragged tail (n_q, n_k not multiples of 16/32)
        int bad = 0;
        for (auto [nq, dep] : std::vector<std::pair<int, int>>{{16, 0}, {37, 0}, {48, 41}, {33, 100}}) {
            const int nk = dep + nq;
            Buf b(nq, nk, 1234 + nq + dep);
            std::vector<float> ref((size_t)nq * H * D), o1(ref.size()), o2(ref.size());
            ref_attn(b.hQ, b.hK, b.hV, ref, nq, nk, dep, scale);
            uf.run(b.Q, b.K, b.V, b.O1, nq, nk, dep, scale);
            run_fa(b.Q, b.K, b.V, b.O2, nq, nk, dep, scale);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(o1.data(), b.O1, o1.size() * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(o2.data(), b.O2, o2.size() * 4, hipMemcpyDeviceToHost));
            const double e1 = max_rel(o1, ref), e2 = max_rel(o2, ref);
            const bool ok = e1 < 2e-2 && e2 < 2e-2;  // bf16 P and inputs: ~1e-3 expected
            bad += !ok;
            printf("check n_q=%d depth=%d: unfused max_rel %.2e, fa max_rel %.2e %s\n", nq, dep, e1, e2, ok ? "OK" : "FAIL");
        }
        if (bad) return 1;
    }

    printf("%8s %8s %12s %12s %8s %12s %12s\n", "depth", "n_k", "unfused_ms", "fa_ms", "speedup", "unfused_TF", "fa_TF");
    for (int dep : depths) {
        const int nk = dep + n_q;
        Buf b(n_q, nk, 7);
        const double pairs = (double)n_q * dep + (double)n_q * (n_q + 1) / 2;
        const double flops = 4.0 * pairs * D * H;
        const float tu = time_ms([&] { uf.run(b.Q, b.K, b.V, b.O1, n_q, nk, dep, scale); }, reps);
        const float tf = time_ms([&] { run_fa(b.Q, b.K, b.V, b.O2, n_q, nk, dep, scale); }, reps);
        std::vector<float> o1((size_t)n_q * H * D), o2(o1.size());
        CK(hipMemcpy(o1.data(), b.O1, o1.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(o2.data(), b.O2, o2.size() * 4, hipMemcpyDeviceToHost));
        printf("%8d %8d %12.2f %12.2f %7.2fx %12.2f %12.2f   (fa vs unfused max_rel %.1e)\n", dep, nk, tu, tf, tu / tf,
               flops / tu / 1e9, flops / tf / 1e9, max_rel(o2, o1));
    }
    return 0;
}
