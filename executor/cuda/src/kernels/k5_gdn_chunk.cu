// K5 chunked prefill (m22n design note D2, D6): the WY form of the gated delta rule in chunks of 64 tokens with the
// rounding points of FLA's chunk_gated_delta_rule as SGLang's Triton GDN backend runs it on sm_120.
//
// math_reference: flash-linear-attention
// math_reference: sglang
//
// Written from the math of sglang/kernels/ops/attention/fla/ chunk.py, cumsum.py, chunk_fwd.py, wy_fast.py,
// chunk_delta_h.py and chunk_o.py and from the PTX Triton emits for them (read offline); nothing is vendored. The
// rounding points and the FP32 operation order that reach the results are reproduced: BF16 tensor-core MMAs
// (m16n8k16, FP32 accumulate, the reduction dimension in ascending 16-wide steps) wherever FLA uses a BF16 tl.dot, TF32
// MMAs (m16n8k8) for the block merge of (I + A)^-1, Triton's exp lowering (ex2.approx.f32 of x * log2e), its scan tree
// for the per-chunk cumsum, its warp reduction tree for the 16x16 forward substitution and the FMA contractions LLVM
// forms (h = fma(h, exp(G_last), k^T v), o = fma(scale, X, P * scale)). Every product and sum below is written with
// explicit _rn intrinsics so the compiler cannot contract or reorder them differently.
#include "kernels/k5_gdn_chunk.h"

#include "cere_model_constants.h"
#include "kernels/k5_gdn.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cere::k5 {
namespace {

constexpr int kHK = gen::kLinearNumKeyHeads;    // key (and query) heads
constexpr int kHV = gen::kLinearNumValueHeads;  // value heads
constexpr int kGroup = kHV / kHK;
constexpr int kD = kHeadDim;  // 128 = key head dim = value head dim
constexpr int kBT = kChunkTokens;
constexpr int kBC = 16;  // FLA sub-block of the fused kkt + solve kernel
constexpr float kChunkL2Eps = 1e-6f;  // FLA l2norm_fwd default
constexpr float kLog2e = 1.44269502162933349609375f;  // 0x3FB8AA3B, Triton's exp lowering constant
constexpr unsigned kFull = 0xffffffffu;
constexpr int kThreads = 128;  // 4 warps in every kernel below
constexpr int kHBlockV = 32;   // value columns per CTA of the state recurrence (chunk_delta_h.py BV)
constexpr int kOBlockV = 64;   // value columns per CTA of the output kernel (chunk_o.py BV)
static_assert(kD == 128 && kBT == 64, "the tilings below are written for 128-wide heads and 64-token chunks");

using bf16 = __nv_bfloat16;

// ---- arithmetic as Triton lowers it ----------------------------------------------------------------------------

__device__ __forceinline__ float ex2_approx(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// tl.exp(x)
__device__ __forceinline__ float tl_exp(float x) { return ex2_approx(__fmul_rn(x, kLog2e)); }

// FLA safe_exp(x) = exp(where(x <= 0, x, -inf)) (op.py)
__device__ __forceinline__ float safe_exp(float x) {
    const float t = __fmul_rn(x, kLog2e);
    return ex2_approx(x <= 0.0f ? t : -INFINITY);
}

__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&p);
}

// Two adjacent BF16 values of global memory (4-byte aligned), zero when `valid` is false.
__device__ __forceinline__ uint32_t ld_pair(const bf16 *p, bool valid) {
    return valid ? __ldg(reinterpret_cast<const unsigned int *>(p)) : 0u;
}

__device__ __forceinline__ uint32_t tf32_operand(float x) {
    if constexpr (kChunkTf32Variant == 1) {
        uint32_t r;
        asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
        return r;
    } else {
        return __float_as_uint(x);
    }
}

// D (16x8 FP32) += A (16x16 BF16, row) * B (16x8 BF16, col).
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// D (16x8 FP32) += A (16x8 TF32, row) * B (8x8 TF32, col).
__device__ __forceinline__ void mma_tf32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 16 (4) bytes global -> shared by cp.async; the destination is zero-filled when `valid` is false (`src` must still
// be a mapped address).
__device__ __forceinline__ void cp16_zfill(void *dst, const void *src, bool valid) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp4_zfill(void *dst, const void *src, bool valid) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(d), "l"(src), "r"(valid ? 4 : 0) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
// All but the most recent cp.async group of this thread are complete.
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
// L2 prefetch of the line holding p (generic address); movement only, nothing reaches a register.
__device__ __forceinline__ void prefetch_l2(const void *p) { asm volatile("prefetch.L2 [%0];" ::"l"(p)); }

// The four 8x8 BF16 matrices of an m16n8k16 A fragment (rows addressed by lanes 8m .. 8m + 7 for matrix m).
__device__ __forceinline__ void ldsm_x4(uint32_t (&a)[4], const void *p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&a)[4], const void *p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}
// The two 8x8 BF16 matrices of an m16n8k16 B fragment from [n][k] rows (lanes 0 .. 15 address the rows).
__device__ __forceinline__ void ldsm_x2(uint32_t &b0, uint32_t &b1, const void *p) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(s));
}
// Every cp.async group of this thread is complete.
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }

// ---- 1. per-chunk inclusive cumsum of g (cumsum.py chunk_local_cumsum_scalar_kernel, BT 64, 8 warps) ------------
// Triton's scan: a Hillis-Steele scan of each 32-lane warp (x += shfl_up(x, d), d = 1, 2, 4, 8, 16), then the second
// warp adds the first warp's total. Rows past the end read 0 and are not stored.

__global__ void __launch_bounds__(kBT) chunk_cumsum_kernel(const float *__restrict__ g, float *__restrict__ G,
                                                           int64_t tokens) {
    __shared__ float total;
    const int i = threadIdx.x;
    const int lane = i & 31;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * kBT + i;
    const int head = blockIdx.y;
    float x = row < tokens ? g[row * kHV + head] : 0.0f;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
        const float y = __shfl_up_sync(kFull, x, d);
        if (lane >= d) {
            x = __fadd_rn(x, y);
        }
    }
    if (i == 31) {
        total = x;
    }
    __syncthreads();
    if (i >= 32) {
        x = __fadd_rn(total, x);
    }
    if (row < tokens) {
        G[row * kHV + head] = x;
    }
}

// ---- 2. A = (I + A_strict)^-1, then u and w (chunk_fwd.py kkt_solve kernel + wy_fast.py) ------------------------
// One CTA per (chunk, value head). A_strict = tril(diag(beta) (k k^T * safe_exp(G_i - G_j)), -1) is formed in FP32 by
// BF16 MMAs over the 128 key dims; the four 16x16 diagonal blocks are inverted by forward substitution (FP32 scalar
// ops in Triton's reduction tree), then merged by TF32 MMAs in the bracket order of chunk_fwd.py:270-304. A is used
// as BF16 by u = BF16(A BF16(v * beta)) and w = BF16(A BF16(k * beta * exp(G))).
// Movement (O-PTC4, bit-identical to the register-load form: same values, same MMA and FP32 op order): the chunk's k
// rows are staged by cp.async into the `ai` space for step 1 (ldmatrix fragments; `ai` is zeroed after step 1), v is
// prefetched into registers at the start; after step 3 the u / w B operands BF16(v * beta), BF16(k * beta * exp(G))
// are built once per element into a tile over the dead `af` (ldmatrix.trans), and u / w are staged over `ai` (dead
// once A is in registers) for 16-byte stores.

constexpr int kLdA = kBT + 4;  // FP32 row pitch of the 64x64 shared matrices
constexpr int kLdK = kD + 8;   // BF16 pitch of staged 128-wide chunk rows (k, w, q, h_c, operand tiles)
static_assert(kBT * kLdK * 2 <= kBT * kLdA * 4, "a BF16 row tile overlays one FP32 64x64 matrix");

struct KktShared {
    float af[kBT][kLdA];  // gated, masked, beta-scaled k k^T (lower blocks)
    float ai[kBT][kLdA];  // (I + A)^-1 (FP32 until the BF16 use)
    float tmp[4][kBC][kBC + 4];  // per-warp merge intermediates
    float G[kBT];
    float beta[kBT];
    float expG[kBT];
};

// One warp: D (16x16) = X (16x16) Y (16x16) [+ D] by TF32 MMAs (two 8-wide k steps per 8-wide column tile).
__device__ __forceinline__ void mm16_tf32(float (&d)[2][4], const float *x, int ldx, const float *y, int ldy,
                                          int lane) {
    const int g = lane >> 2;
    const int t = lane & 3;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const int k0 = ks * 8;
            const uint32_t a[4] = {tf32_operand(x[g * ldx + k0 + t]), tf32_operand(x[(g + 8) * ldx + k0 + t]),
                                   tf32_operand(x[g * ldx + k0 + t + 4]), tf32_operand(x[(g + 8) * ldx + k0 + t + 4])};
            mma_tf32(d[nt], a, tf32_operand(y[(k0 + t) * ldy + nt * 8 + g]),
                     tf32_operand(y[(k0 + t + 4) * ldy + nt * 8 + g]));
        }
    }
}

__device__ __forceinline__ void zero16(float (&d)[2][4]) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            d[nt][r] = 0.0f;
        }
    }
}

// Stores a 16x16 MMA result (optionally negated as Triton's 0 - x) to shared memory.
__device__ __forceinline__ void store16(float *out, int ld, const float (&d)[2][4], bool negate, int lane) {
    const int g = lane >> 2;
    const int t = lane & 3;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int c = nt * 8 + 2 * t;
        out[g * ld + c] = negate ? __fsub_rn(0.0f, d[nt][0]) : d[nt][0];
        out[g * ld + c + 1] = negate ? __fsub_rn(0.0f, d[nt][1]) : d[nt][1];
        out[(g + 8) * ld + c] = negate ? __fsub_rn(0.0f, d[nt][2]) : d[nt][2];
        out[(g + 8) * ld + c + 1] = negate ? __fsub_rn(0.0f, d[nt][3]) : d[nt][3];
    }
}

__global__ void __launch_bounds__(kThreads) chunk_kkt_wy_kernel(const bf16 *__restrict__ kn, const bf16 *__restrict__ v,
                                                              const float *__restrict__ G,
                                                              const float *__restrict__ beta, bf16 *__restrict__ u,
                                                              bf16 *__restrict__ w, int64_t tokens) {
    __shared__ __align__(16) KktShared s;
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    const int key_head = head / kGroup;
    const int64_t base = static_cast<int64_t>(chunk) * kBT;
    const int nvalid = static_cast<int>(tokens - base < kBT ? tokens - base : kBT);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int64_t vpitch = static_cast<int64_t>(kHV) * kD;  // row pitch of v, u, w
    const int64_t kpitch = static_cast<int64_t>(kHK) * kD;
    const bf16 *krow = kn + base * kpitch + key_head * kD;
    const bf16 *vrow = v + base * vpitch + head * kD;
    bf16(*kt)[kLdK] = reinterpret_cast<bf16(*)[kLdK]>(&s.ai[0][0]);  // raw k rows during step 1
    bf16(*ot)[kLdK] = reinterpret_cast<bf16(*)[kLdK]>(&s.af[0][0]);  // u / w B operand tile in step 4

    constexpr int kSeg = kBT * (kD / 8) / kThreads;  // 16-byte segments per thread (8)
#pragma unroll
    for (int e = 0; e < kSeg; ++e) {
        const int idx = tid + e * kThreads;
        const int j = idx / (kD / 8);
        const int c8 = (idx % (kD / 8)) * 8;
        cp16_zfill(&kt[j][c8], krow + static_cast<int64_t>(j < nvalid ? j : 0) * kpitch + c8, j < nvalid);
    }
    cp_commit();
    uint4 vraw[kSeg];
#pragma unroll
    for (int e = 0; e < kSeg; ++e) {
        const int idx = tid + e * kThreads;
        const int j = idx / (kD / 8);
        const int c8 = (idx % (kD / 8)) * 8;
        vraw[e] = j < nvalid ? __ldg(reinterpret_cast<const uint4 *>(vrow + static_cast<int64_t>(j) * vpitch + c8))
                             : make_uint4(0u, 0u, 0u, 0u);
    }
    if (tid < kBT) {
        const bool valid = tid < nvalid;
        const float gv = valid ? G[(base + tid) * kHV + head] : 0.0f;
        s.G[tid] = gv;
        s.beta[tid] = valid ? beta[(base + tid) * kHV + head] : 0.0f;
        s.expG[tid] = tl_exp(gv);
    }
    cp_wait0();
    __syncthreads();

    // Step 1: the 10 lower 16x16 blocks of k k^T (20 tiles of 16x8, 5 per warp), gate, mask, beta; k from kt.
    for (int unit = warp; unit < 20; unit += 4) {
        const int blk = unit >> 1;
        const int nt = unit & 1;
        // blk -> (bi, bj), bi >= bj: (0,0) (1,0) (1,1) (2,0) (2,1) (2,2) (3,0) (3,1) (3,2) (3,3)
        const int bi = blk >= 6 ? 3 : blk >= 3 ? 2 : blk >= 1 ? 1 : 0;
        const int bj = blk - bi * (bi + 1) / 2;
        const int r0 = bi * kBC;
        const int c0 = bj * kBC + nt * 8;
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int ks = 0; ks < kD / 16; ++ks) {
            const int k0 = ks * 16;
            uint32_t a[4];
            ldsm_x4(a, &kt[r0 + (lane & 7) + ((lane >> 3) & 1) * 8][k0 + ((lane >> 4) & 1) * 8]);
            uint32_t b0, b1;
            ldsm_x2(b0, b1, &kt[c0 + (lane & 7)][k0 + ((lane >> 3) & 1) * 8]);
            mma_bf16(d, a, b0, b1);
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = r0 + g + (r >= 2 ? 8 : 0);
            const int j = c0 + 2 * t + (r & 1);
            float val = __fmul_rn(d[r], safe_exp(__fsub_rn(s.G[i], s.G[j])));
            if (bi == bj && !((i > j) && i < nvalid && j < nvalid)) {
                val = 0.0f;
            }
            s.af[i][j] = __fmul_rn(val, s.beta[i]);
        }
    }
    __syncthreads();
    for (int e = tid; e < kBT * kLdA; e += kThreads) {
        (&s.ai[0][0])[e] = 0.0f;
    }
    __syncthreads();

    // Step 2: forward substitution on diagonal block `warp` (chunk_fwd.py:235-264); lane c < 16 owns column c.
    {
        const int d0 = warp * kBC;
        const int c = lane & (kBC - 1);
        if (lane < kBC) {
            for (int r = 0; r < kBC; ++r) {
                s.ai[d0 + r][d0 + c] = -s.af[d0 + r][d0 + c];
            }
        }
        __syncwarp();
        const int rows = nvalid - d0 < kBC ? nvalid - d0 : kBC;
        for (int i = 2; i < rows; ++i) {
            float a[kBC];
#pragma unroll
            for (int j = 0; j < kBC; ++j) {
                a[j] = j < i ? -s.af[d0 + i][d0 + j] : 0.0f;
            }
            // Triton's tree for sum_j a[j] * Ai[j][c] over the m16n8 accumulator layout: lane group g holds rows g and
            // g + 8 (fma(a_g, Ai_g, a_{g+8} Ai_{g+8})), then butterflies over row bits 4, 2, 1.
            float s0[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                s0[r] = __fmaf_rn(a[r], s.ai[d0 + r][d0 + c], __fmul_rn(a[r + 8], s.ai[d0 + r + 8][d0 + c]));
            }
            const float s10 = __fadd_rn(s0[0], s0[4]);
            const float s11 = __fadd_rn(s0[1], s0[5]);
            const float s12 = __fadd_rn(s0[2], s0[6]);
            const float s13 = __fadd_rn(s0[3], s0[7]);
            const float sum = __fadd_rn(__fadd_rn(s10, s12), __fadd_rn(s11, s13));
            const float row_c = __fadd_rn(c < i ? -s.af[d0 + i][d0 + c] : 0.0f, sum);
            __syncwarp();
            if (lane < kBC) {
                s.ai[d0 + i][d0 + c] = row_c;
            }
            __syncwarp();
        }
        if (lane < kBC) {
            for (int r = 0; r < kBC; ++r) {
                s.ai[d0 + r][d0 + c] = __fadd_rn(s.ai[d0 + r][d0 + c], r == c ? 1.0f : 0.0f);
            }
        }
    }
    __syncthreads();

    // Step 3: block merge by TF32 MMAs (chunk_fwd.py:270-304).
    auto AF = [&](int bi, int bj) { return &s.af[bi * kBC][bj * kBC]; };
    auto AI = [&](int bi, int bj) { return &s.ai[bi * kBC][bj * kBC]; };
    float *tmp = &s.tmp[warp][0][0];
    constexpr int kLdT = kBC + 4;
    float d[2][4];
    if (warp < 3) {  // Ai10, Ai21, Ai32 = -((Ai_ii A_ij) Ai_jj)
        const int bi = warp + 1;
        const int bj = warp;
        zero16(d);
        mm16_tf32(d, AI(bi, bi), kLdA, AF(bi, bj), kLdA, lane);
        store16(tmp, kLdT, d, false, lane);
        __syncwarp();
        zero16(d);
        mm16_tf32(d, tmp, kLdT, AI(bj, bj), kLdA, lane);
        store16(AI(bi, bj), kLdA, d, true, lane);
    }
    __syncthreads();
    if (warp < 2) {  // Ai20 = -(Ai22 (A20 Ai00 + A21 Ai10)), Ai31 = -(Ai33 (A31 Ai11 + A32 Ai21))
        const int bi = warp + 2;
        const int bj = warp;
        zero16(d);
        mm16_tf32(d, AF(bi, bj), kLdA, AI(bj, bj), kLdA, lane);
        mm16_tf32(d, AF(bi, bj + 1), kLdA, AI(bj + 1, bj), kLdA, lane);
        store16(tmp, kLdT, d, false, lane);
        __syncwarp();
        zero16(d);
        mm16_tf32(d, AI(bi, bi), kLdA, tmp, kLdT, lane);
        store16(AI(bi, bj), kLdA, d, true, lane);
    }
    __syncthreads();
    if (warp == 0) {  // Ai30 = -(Ai33 (A30 Ai00 + A31 Ai10 + A32 Ai20))
        zero16(d);
        mm16_tf32(d, AF(3, 0), kLdA, AI(0, 0), kLdA, lane);
        mm16_tf32(d, AF(3, 1), kLdA, AI(1, 0), kLdA, lane);
        mm16_tf32(d, AF(3, 2), kLdA, AI(2, 0), kLdA, lane);
        store16(tmp, kLdT, d, false, lane);
        __syncwarp();
        zero16(d);
        mm16_tf32(d, AI(3, 3), kLdA, tmp, kLdT, lane);
        store16(AI(3, 0), kLdA, d, true, lane);
    }
    __syncthreads();

    // Step 4: u then w; the B operand tile ot[j][n] = BF16(operand) overlays af (dead after step 3).
    const int m0 = warp * 16;
    uint32_t af[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const int k0 = ks * 16 + 2 * t;
        af[ks][0] = pack_bf16(s.ai[m0 + g][k0], s.ai[m0 + g][k0 + 1]);
        af[ks][1] = pack_bf16(s.ai[m0 + g + 8][k0], s.ai[m0 + g + 8][k0 + 1]);
        af[ks][2] = pack_bf16(s.ai[m0 + g][k0 + 8], s.ai[m0 + g][k0 + 9]);
        af[ks][3] = pack_bf16(s.ai[m0 + g + 8][k0 + 8], s.ai[m0 + g + 8][k0 + 9]);
    }
    // raw (8 BF16 of row j) -> ot[j][c8 ..] = BF16(x * beta[j] [* expG[j]])
    auto fill = [&](const uint4 (&raw)[kSeg], bool with_exp) {
#pragma unroll
        for (int e = 0; e < kSeg; ++e) {
            const int idx = tid + e * kThreads;
            const int j = idx / (kD / 8);
            const int c8 = (idx % (kD / 8)) * 8;
            float f[8];
            const __nv_bfloat162 *p = reinterpret_cast<const __nv_bfloat162 *>(&raw[e]);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float2 x = __bfloat1622float2(p[q]);
                f[2 * q] = x.x;
                f[2 * q + 1] = x.y;
            }
            const float bj = s.beta[j];
            const float ej = s.expG[j];
            uint4 out;
            uint32_t *o32 = reinterpret_cast<uint32_t *>(&out);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float lo = with_exp ? __fmul_rn(__fmul_rn(f[2 * q], bj), ej) : __fmul_rn(f[2 * q], bj);
                const float hi = with_exp ? __fmul_rn(__fmul_rn(f[2 * q + 1], bj), ej) : __fmul_rn(f[2 * q + 1], bj);
                o32[q] = pack_bf16(lo, hi);
            }
            *reinterpret_cast<uint4 *>(&ot[j][c8]) = out;
        }
    };
    // out tile (BF16, overlays ai: dead once af[][] is in registers) -> coalesced 16-byte stores of valid rows.
    bf16(*rt)[kLdK] = reinterpret_cast<bf16(*)[kLdK]>(&s.ai[0][0]);
    auto product = [&](bf16 *dst) {
#pragma unroll 2
        for (int nt = 0; nt < kD / 8; nt += 2) {
            float d0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float d1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                uint32_t b[4];
                ldsm_x4_trans(b, &ot[ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8][nt * 8 + (lane >> 4) * 8]);
                mma_bf16(d0, af[ks], b[0], b[1]);
                mma_bf16(d1, af[ks], b[2], b[3]);
            }
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int i = m0 + g + 8 * rr;
                const int col = nt * 8 + 2 * t;
                *reinterpret_cast<__nv_bfloat162 *>(&rt[i][col]) =
                    __halves2bfloat162(__float2bfloat16_rn(d0[2 * rr]), __float2bfloat16_rn(d0[2 * rr + 1]));
                *reinterpret_cast<__nv_bfloat162 *>(&rt[i][col + 8]) =
                    __halves2bfloat162(__float2bfloat16_rn(d1[2 * rr]), __float2bfloat16_rn(d1[2 * rr + 1]));
            }
        }
        __syncthreads();
        bf16 *drow = dst + base * vpitch + head * kD;
#pragma unroll
        for (int e = 0; e < kSeg; ++e) {
            const int idx = tid + e * kThreads;
            const int j = idx / (kD / 8);
            const int c8 = (idx % (kD / 8)) * 8;
            if (j < nvalid) {
                *reinterpret_cast<uint4 *>(drow + static_cast<int64_t>(j) * vpitch + c8) =
                    *reinterpret_cast<const uint4 *>(&rt[j][c8]);
            }
        }
    };
    fill(vraw, false);
    // k rows again (zero past the end) for w, in flight during u.
#pragma unroll
    for (int e = 0; e < kSeg; ++e) {
        const int idx = tid + e * kThreads;
        const int j = idx / (kD / 8);
        const int c8 = (idx % (kD / 8)) * 8;
        vraw[e] = j < nvalid ? __ldg(reinterpret_cast<const uint4 *>(krow + static_cast<int64_t>(j) * kpitch + c8))
                             : make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();
    product(u);
    __syncthreads();
    fill(vraw, true);
    __syncthreads();
    product(w);
}

// ---- 3. the state recurrence over chunks (chunk_delta_h.py, BV 32) -----------------------------------------------
// One CTA per (32 value columns, value head), sequential over chunks. The FP32 state slice h[kk, v] (128 x 32) lives
// in the accumulator layout of the k^T v MMA: warp w owns key rows [32 w, 32 w + 32) as two 16-row tiles x four
// 8-column tiles. Per chunk: h_c = BF16(h) (stored for the output kernel); t = u - w h_c^T (FP32), v_new = BF16(t)
// (overwrites u in place); h = fma(h, exp(G_last), k^T BF16(t * safe_exp(G_last - G))). Final state BF16(h).
// D15⁗′: h starts from `state_in32` (FP32, same layout) when it is non-null, and is stored as FP32 to `capture32` at
// the start of chunk `capture_chunk` (== nchunks: after the last chunk); both are plain copies of h. A non-null
// `*_hi` holds value heads [kHV / 2, kHV) of that state (the base pointer then heads [0, kHV / 2) only).
// Movement (O-PTC1, bit-identical to the register-load form: same operands, same MMA and FP32 op order): the chunk's
// k rows and w rows go through single shared buffers by cp.async (rows past the end zero-filled), each issued one
// phase ahead of its use: k(c) at the start of chunk c (used after the w h_c^T phase), w(c + 1) and G(c + 1) right
// after the w h_c^T phase of chunk c (used at the start of chunk c + 1). A fragments come from ldmatrix (.trans for
// k^T); BF16(t * decay) is stored transposed so its B fragments are 32-bit loads; h_c goes to `hs` in 16-byte rows,
// v_new to `uv` in BF16 pairs. Each chunk of the sequential loop still waited on loads issued less than a chunk
// earlier (38 % of DRAM peak at prefill), so every operand is also prefetched to L2 about a chunk ahead, with no
// register destination (the register prefetches of PrefillTc's state2 / state3 lost to scoreboard sharing): the
// cp.async and u loads then wait on L2, not DRAM.
// 48,640 B static smem keeps two CTAs per SM (the 192-CTA grid fits one wave on 188 SMs).

constexpr int kLdHc = kD + 8;   // BF16 pitch of hc [v][kk]
constexpr int kLdVt = kBT + 8;  // BF16 pitch of vbt [v][i]

__global__ void __launch_bounds__(kThreads) chunk_state_kernel(const bf16 *__restrict__ kn, const bf16 *__restrict__ w,
                                                               bf16 *__restrict__ uv, const float *__restrict__ G,
                                                               bf16 *__restrict__ state, bf16 *__restrict__ hs,
                                                               int64_t tokens, const float *__restrict__ state_in32,
                                                               float *__restrict__ capture32, int64_t capture_chunk,
                                                               const float *__restrict__ state_in32_hi,
                                                               float *__restrict__ capture32_hi) {
    __shared__ __align__(16) bf16 hc[kHBlockV][kLdHc];
    __shared__ __align__(16) bf16 vbt[kHBlockV][kLdVt];
    __shared__ __align__(16) bf16 sk[kBT][kLdK];
    __shared__ __align__(16) bf16 sw[kBT][kLdK];
    __shared__ __align__(16) float sG[2][kBT];
    const int v0 = blockIdx.x * kHBlockV;
    const int head = blockIdx.y;
    const int key_head = head / kGroup;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int64_t nchunks = (tokens + kBT - 1) / kBT;
    const int64_t vpitch = static_cast<int64_t>(kHV) * kD;
    const int64_t kpitch = static_cast<int64_t>(kHK) * kD;
    bf16 *st = state + static_cast<int64_t>(head) * kD * kD;
    const bool hi = head >= kHV / 2;
    const int64_t head32 = static_cast<int64_t>(head) * kD * kD;
    const int64_t half32 = static_cast<int64_t>(head - kHV / 2) * kD * kD;
    const float *st32 = state_in32 == nullptr ? nullptr
                        : hi && state_in32_hi != nullptr ? state_in32_hi + half32
                                                         : state_in32 + head32;
    float *cap32 = capture32 == nullptr ? nullptr
                   : hi && capture32_hi != nullptr ? capture32_hi + half32
                                                   : capture32 + head32;

    float h[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int kk = (2 * warp + mi) * 16 + g + (r >= 2 ? 8 : 0);
                const int vv = v0 + nt * 8 + 2 * t + (r & 1);
                const int64_t at = static_cast<int64_t>(vv) * kD + kk;
                h[mi][nt][r] = st32 != nullptr ? st32[at] : __bfloat162float(st[at]);
            }
        }
    }
    auto capture = [&]() {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int kk = (2 * warp + mi) * 16 + g + (r >= 2 ? 8 : 0);
                    const int vv = v0 + nt * 8 + 2 * t + (r & 1);
                    cap32[static_cast<int64_t>(vv) * kD + kk] = h[mi][nt][r];
                }
            }
        }
    };
    auto nvalid_of = [&](int64_t chunk) {
        const int64_t base = chunk * kBT;
        return static_cast<int>(tokens - base < kBT ? tokens - base : kBT);
    };
    // rows of a [tokens][pitch] BF16 source (128 wide at col0) of one chunk -> dst (rows past the end zero).
    auto stage_rows = [&](bf16 (*dst)[kLdK], const bf16 *src0, int64_t pitch, int nvalid) {
#pragma unroll
        for (int e = tid; e < kBT * (kD / 8); e += kThreads) {
            const int j = e / (kD / 8);
            const int c8 = (e % (kD / 8)) * 8;
            cp16_zfill(&dst[j][c8], src0 + static_cast<int64_t>(j < nvalid ? j : 0) * pitch + c8, j < nvalid);
        }
    };
    auto stage_w = [&](int64_t chunk) {  // w and G of `chunk`, one group
        const int nvalid = nvalid_of(chunk);
        stage_rows(sw, w + chunk * kBT * vpitch + head * kD, vpitch, nvalid);
        if (tid < kBT) {
            cp4_zfill(&sG[chunk & 1][tid], G + (chunk * kBT + (tid < nvalid ? tid : 0)) * kHV + head, tid < nvalid);
        }
        cp_commit();
    };
    // L2 prefetch (movement only): k and u of chunk c + 1 at the start of chunk c (staged / loaded at the start of
    // c + 1), w and G of chunk c + 2 (staged after the w h_c^T phase of c + 1). Each 32-byte sector once per grid:
    // a key head's k split over its kGroup * kVBlocks CTAs, a head's w over its kVBlocks CTAs, a chunk's G block (all
    // heads) over the grid; u is this CTA's own 64 bytes of each row.
    constexpr int kSec = 32 / static_cast<int>(sizeof(bf16));  // BF16 per sector
    constexpr int kRowSec = kD / kSec;                          // sectors of a 128-wide row (8)
    constexpr int kVBlocks = kD / kHBlockV;                     // CTAs per value head (4)
    constexpr int kKSplit = kGroup * kVBlocks;                  // CTAs per key head (12)
    constexpr int kGrid = kHV * kVBlocks;                       // CTAs (192)
    constexpr int kGSec = kBT * kHV * static_cast<int>(sizeof(float)) / 32;  // sectors of a chunk's G block (384)
    static_assert(kThreads * kVBlocks == kBT * kRowSec && kThreads * kKSplit >= kBT * kRowSec &&
                      kThreads == 2 * kBT && kThreads * kGrid >= kGSec,
                  "one prefetch per thread covers each tile");
    const int ksec = (head % kGroup) * kVBlocks + static_cast<int>(blockIdx.x) + tid * kKSplit;
    const int wsec = static_cast<int>(blockIdx.x) + tid * kVBlocks;
    const int gsec = head * kVBlocks + static_cast<int>(blockIdx.x) + tid * kGrid;
    auto prefetch_ku = [&](int64_t chunk) {
        if (chunk >= nchunks) {
            return;
        }
        const int64_t base = chunk * kBT;
        const int nvalid = nvalid_of(chunk);
        if (ksec < kBT * kRowSec && ksec / kRowSec < nvalid) {
            prefetch_l2(kn + (base + ksec / kRowSec) * kpitch + key_head * kD + (ksec % kRowSec) * kSec);
        }
        if (tid / 2 < nvalid) {
            prefetch_l2(uv + (base + tid / 2) * vpitch + head * kD + v0 + (tid % 2) * kSec);
        }
    };
    auto prefetch_wg = [&](int64_t chunk) {
        if (chunk >= nchunks) {
            return;
        }
        const int64_t base = chunk * kBT;
        const int nvalid = nvalid_of(chunk);
        if (wsec / kRowSec < nvalid) {
            prefetch_l2(w + (base + wsec / kRowSec) * vpitch + head * kD + (wsec % kRowSec) * kSec);
        }
        if (gsec < kGSec && gsec * 32 < nvalid * kHV * static_cast<int>(sizeof(float))) {
            prefetch_l2(reinterpret_cast<const char *>(G + base * kHV) + gsec * 32);
        }
    };

    stage_w(0);
    prefetch_wg(1);
    for (int64_t chunk = 0; chunk < nchunks; ++chunk) {
        const int64_t base = chunk * kBT;
        const int nvalid = nvalid_of(chunk);
        stage_rows(sk, kn + base * kpitch + key_head * kD, kpitch, nvalid);
        cp_commit();
        prefetch_ku(chunk + 1);
        prefetch_wg(chunk + 2);
        const int i0 = warp * 16;
        bf16 *ur = uv + base * vpitch + head * kD + v0;
        uint32_t u2[4][2];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int i = i0 + g + 8 * rr;
                u2[nt][rr] = ld_pair(ur + static_cast<int64_t>(i) * vpitch + nt * 8 + 2 * t, i < nvalid);
            }
        }
        if (cap32 != nullptr && chunk == capture_chunk) {
            capture();
        }
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int kk = (2 * warp + mi) * 16 + g + (r >= 2 ? 8 : 0);
                    const int vv = nt * 8 + 2 * t + (r & 1);
                    hc[vv][kk] = __float2bfloat16_rn(h[mi][nt][r]);
                }
            }
        }
        cp_wait1();  // w(c), G(c) in (k(c) may still be in flight)
        __syncthreads();
        const float *sGc = sG[chunk & 1];
        {
            bf16 *hsc = hs + (chunk * kHV + head) * kD * kD + static_cast<int64_t>(v0) * kD;
#pragma unroll
            for (int e = tid; e < kHBlockV * (kD / 8); e += kThreads) {
                const int vv = e / (kD / 8);
                const int c8 = (e % (kD / 8)) * 8;
                *reinterpret_cast<uint4 *>(hsc + static_cast<int64_t>(vv) * kD + c8) =
                    *reinterpret_cast<const uint4 *>(&hc[vv][c8]);
            }
        }
        const float g_last = sGc[nvalid - 1];
        float d[4][4] = {};
#pragma unroll
        for (int ks = 0; ks < kD / 16; ++ks) {
            const int k0 = ks * 16 + 2 * t;
            uint32_t a[4];
            ldsm_x4(a, &sw[i0 + (lane & 7) + ((lane >> 3) & 1) * 8][ks * 16 + ((lane >> 4) & 1) * 8]);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const bf16 *hrow = &hc[nt * 8 + g][0];
                mma_bf16(d[nt], a, *reinterpret_cast<const uint32_t *>(hrow + k0),
                         *reinterpret_cast<const uint32_t *>(hrow + k0 + 8));
            }
        }
        // Accumulator elements 2 rr and 2 rr + 1 of tile nt are columns nt * 8 + 2 t and + 1 of row i0 + g + 8 rr: v_new
        // goes out as one BF16 pair, each half rounded on its own as before.
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int i = i0 + g + 8 * rr;
                const uint32_t pair = u2[nt][rr];
                float tv[2];
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int vv = nt * 8 + 2 * t + c;
                    const float uval = __uint_as_float(c ? (pair & 0xffff0000u) : (pair << 16));
                    tv[c] = __fsub_rn(uval, d[nt][2 * rr + c]);
                    vbt[vv][i] = __float2bfloat16_rn(__fmul_rn(tv[c], safe_exp(__fsub_rn(g_last, sGc[i]))));
                }
                if (i < nvalid) {
                    *reinterpret_cast<__nv_bfloat162 *>(ur + static_cast<int64_t>(i) * vpitch + nt * 8 + 2 * t) =
                        __floats2bfloat162_rn(tv[0], tv[1]);
                }
            }
        }
        __syncthreads();  // sw free
        if (chunk + 1 < nchunks) {
            stage_w(chunk + 1);
        } else {
            cp_commit();
        }
        cp_wait1();  // k(c) in
        __syncthreads();
        {
            const float eg = tl_exp(g_last);
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int kkb = (2 * warp + mi) * 16;
                float e[4][4] = {};
#pragma unroll
                for (int ks = 0; ks < kBT / 16; ++ks) {
                    const int j0 = ks * 16 + 2 * t;
                    uint32_t a[4];
                    ldsm_x4_trans(a, &sk[ks * 16 + (lane & 7) + ((lane >> 4) & 1) * 8][kkb + ((lane >> 3) & 1) * 8]);
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        const int n = nt * 8 + g;
                        mma_bf16(e[nt], a, *reinterpret_cast<const uint32_t *>(&vbt[n][j0]),
                                 *reinterpret_cast<const uint32_t *>(&vbt[n][j0 + 8]));
                    }
                }
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        h[mi][nt][r] = __fmaf_rn(h[mi][nt][r], eg, e[nt][r]);
                    }
                }
            }
        }
        __syncthreads();
    }
    if (cap32 != nullptr && capture_chunk == nchunks) {
        capture();
    }

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int kk = (2 * warp + mi) * 16 + g + (r >= 2 ? 8 : 0);
                const int vv = v0 + nt * 8 + 2 * t + (r & 1);
                st[static_cast<int64_t>(vv) * kD + kk] = __float2bfloat16_rn(h[mi][nt][r]);
            }
        }
    }
}

// ---- 4. the output (chunk_o.py, BK 128, BV 64) ---------------------------------------------------------------------
// One CTA per (chunk, value head, 64 value columns); warp w owns chunk rows [16 w, 16 w + 16).
// X = (q h_c^T) * exp(G); P = BF16(tril_incl(q k^T * safe_exp(G_i - G_j))) v_new; o = BF16(fma(scale, X, P * scale)).
// Movement (O-PTC5, bit-identical to the register-load form): q, k and v_new are staged by cp.async (ldmatrix /
// ldmatrix.trans fragments); P v runs before X so the h_c block's cp.async (into the k space) overlaps it; o goes
// through this warp's own q rows for 16-byte stores. 44 KB smem keeps 2 CTAs per SM.

constexpr int kLdVo = kOBlockV + 8;
struct OutShared {
    bf16 qt[kBT][kLdK];  // q rows; then this warp's pm rows, then this warp's o rows
    bf16 kt[kBT][kLdK];  // k rows; then h_c rows v0 .. v0 + 63
    bf16 vt[kBT][kLdVo];  // v_new rows, columns v0 .. v0 + 63
    float sG[kBT];
};

__global__ void __launch_bounds__(kThreads) chunk_output_kernel(const bf16 *__restrict__ qn, const bf16 *__restrict__ kn,
                                                                 const bf16 *__restrict__ vnew,
                                                                 const bf16 *__restrict__ hs,
                                                                 const float *__restrict__ G, bf16 *__restrict__ o,
                                                                 int64_t tokens, float scale) {
    __shared__ __align__(16) OutShared s;
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    const int v0 = blockIdx.z * kOBlockV;
    const int key_head = head / kGroup;
    const int64_t base = static_cast<int64_t>(chunk) * kBT;
    const int nvalid = static_cast<int>(tokens - base < kBT ? tokens - base : kBT);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int64_t vpitch = static_cast<int64_t>(kHV) * kD;
    const int64_t kpitch = static_cast<int64_t>(kHK) * kD;
    const bf16 *qr = qn + base * kpitch + key_head * kD;
    const bf16 *kr = kn + base * kpitch + key_head * kD;
    const bf16 *hr = hs + (static_cast<int64_t>(chunk) * kHV + head) * kD * kD + static_cast<int64_t>(v0) * kD;
    const bf16 *vr = vnew + base * vpitch + head * kD + v0;
    constexpr int kSeg = kBT * (kD / 8) / kThreads;  // 8
#pragma unroll
    for (int e = 0; e < kSeg; ++e) {
        const int idx = tid + e * kThreads;
        const int j = idx / (kD / 8);
        const int c8 = (idx % (kD / 8)) * 8;
        const int64_t off = static_cast<int64_t>(j < nvalid ? j : 0) * kpitch + c8;
        cp16_zfill(&s.qt[j][c8], qr + off, j < nvalid);
        cp16_zfill(&s.kt[j][c8], kr + off, j < nvalid);
    }
    cp_commit();
#pragma unroll
    for (int e = 0; e < kBT * (kOBlockV / 8) / kThreads; ++e) {
        const int idx = tid + e * kThreads;
        const int j = idx / (kOBlockV / 8);
        const int c8 = (idx % (kOBlockV / 8)) * 8;
        cp16_zfill(&s.vt[j][c8], vr + static_cast<int64_t>(j < nvalid ? j : 0) * vpitch + c8, j < nvalid);
    }
    cp_commit();
    if (tid < kBT) {
        s.sG[tid] = tid < nvalid ? G[(base + tid) * kHV + head] : 0.0f;
    }
    cp_wait1();
    __syncthreads();

    const int i0 = warp * 16;
    uint32_t qa[kD / 16][4];
#pragma unroll
    for (int ks = 0; ks < kD / 16; ++ks) {
        ldsm_x4(qa[ks], &s.qt[i0 + (lane & 7) + ((lane >> 3) & 1) * 8][ks * 16 + ((lane >> 4) & 1) * 8]);
    }
    __syncwarp();
    bf16(*pm)[kLdK] = s.qt;  // this warp's rows only from here on
    for (int nt = 0; nt < kBT / 8; ++nt) {
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int ks = 0; ks < kD / 16; ks += 2) {
            uint32_t b[4];
            ldsm_x4(b, &s.kt[nt * 8 + (lane & 7)][ks * 16 + (lane >> 3) * 8]);
            mma_bf16(d, qa[ks], b[0], b[1]);
            mma_bf16(d, qa[ks + 1], b[2], b[3]);
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = i0 + g + (r >= 2 ? 8 : 0);
            const int jj = nt * 8 + 2 * t + (r & 1);
            const float val = __fmul_rn(d[r], safe_exp(__fsub_rn(s.sG[i], s.sG[jj])));
            pm[i][jj] = __float2bfloat16_rn(i >= jj ? val : 0.0f);
        }
    }
    __syncthreads();  // every warp is done with kt
#pragma unroll
    for (int e = 0; e < kSeg; ++e) {
        const int idx = tid + e * kThreads;
        const int n = idx / (kD / 8);
        const int c8 = (idx % (kD / 8)) * 8;
        cp16_zfill(&s.kt[n][c8], hr + static_cast<int64_t>(n) * kD + c8, true);
    }
    cp_commit();

    uint32_t pa[kBT / 16][4];
#pragma unroll
    for (int ks = 0; ks < kBT / 16; ++ks) {
        ldsm_x4(pa[ks], &pm[i0 + (lane & 7) + ((lane >> 3) & 1) * 8][ks * 16 + ((lane >> 4) & 1) * 8]);
    }
    cp_wait1();  // v_new
    __syncthreads();
    float p[kOBlockV / 8][4];
#pragma unroll
    for (int nt = 0; nt < kOBlockV / 8; nt += 2) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            p[nt][r] = 0.0f;
            p[nt + 1][r] = 0.0f;
        }
#pragma unroll
        for (int ks = 0; ks < kBT / 16; ++ks) {
            uint32_t b[4];
            ldsm_x4_trans(b, &s.vt[ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8][nt * 8 + (lane >> 4) * 8]);
            mma_bf16(p[nt], pa[ks], b[0], b[1]);
            mma_bf16(p[nt + 1], pa[ks], b[2], b[3]);
        }
    }
    cp_wait0();  // h_c
    __syncthreads();
    bf16(*ot)[kLdK] = s.qt;  // this warp's rows (pm is in registers)
#pragma unroll
    for (int nt = 0; nt < kOBlockV / 8; ++nt) {
        float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int ks = 0; ks < kD / 16; ks += 2) {
            uint32_t b[4];
            ldsm_x4(b, &s.kt[nt * 8 + (lane & 7)][ks * 16 + (lane >> 3) * 8]);
            mma_bf16(x, qa[ks], b[0], b[1]);
            mma_bf16(x, qa[ks + 1], b[2], b[3]);
        }
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int i = i0 + g + 8 * rr;
            const float e = tl_exp(s.sG[i]);
            const float o0 = __fmaf_rn(scale, __fmul_rn(x[2 * rr], e), __fmul_rn(p[nt][2 * rr], scale));
            const float o1 = __fmaf_rn(scale, __fmul_rn(x[2 * rr + 1], e), __fmul_rn(p[nt][2 * rr + 1], scale));
            *reinterpret_cast<__nv_bfloat162 *>(&ot[i][nt * 8 + 2 * t]) =
                __halves2bfloat162(__float2bfloat16_rn(o0), __float2bfloat16_rn(o1));
        }
    }
    __syncwarp();
    bf16 *orow = o + base * vpitch + head * kD + v0;
#pragma unroll
    for (int e = 0; e < 16 * (kOBlockV / 8) / 32; ++e) {
        const int idx = lane + e * 32;
        const int i = i0 + idx / (kOBlockV / 8);
        const int c8 = (idx % (kOBlockV / 8)) * 8;
        if (i < nvalid) {
            *reinterpret_cast<uint4 *>(orow + static_cast<int64_t>(i) * vpitch + c8) =
                *reinterpret_cast<const uint4 *>(&ot[i][c8]);
        }
    }
}

constexpr size_t kAlign = 256;

constexpr size_t aligned(size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

struct Layout {
    size_t qn, kn, G, uv, w, hs, total;
};

Layout layout(int64_t tokens) {
    const size_t T = static_cast<size_t>(tokens);
    const size_t nchunks = (T + kBT - 1) / kBT;
    Layout l{};
    size_t at = 0;
    l.qn = at;
    at += aligned(T * kHK * kD * sizeof(bf16));
    l.kn = at;
    at += aligned(T * kHK * kD * sizeof(bf16));
    l.G = at;
    at += aligned(T * kHV * sizeof(float));
    l.uv = at;
    at += aligned(T * kHV * kD * sizeof(bf16));
    l.w = at;
    at += aligned(T * kHV * kD * sizeof(bf16));
    l.hs = at;
    at += aligned(nchunks * kHV * kD * kD * sizeof(bf16));
    l.total = at;
    return l;
}

}  // namespace

size_t chunk_workspace_bytes(int64_t tokens) {
    if (tokens < 1 || tokens > kChunkMaxTokens) {
        return 0;
    }
    return layout(tokens).total;
}

cudaError_t delta_rule_chunk(const ChunkArgs &args, void *workspace, cudaStream_t stream) {
    const ChunkArgs &p = args;
    if (p.q == nullptr || p.k == nullptr || p.v == nullptr || p.g == nullptr || p.beta == nullptr ||
        p.state == nullptr || p.o == nullptr || workspace == nullptr || p.tokens < 1 || p.tokens > kChunkMaxTokens ||
        reinterpret_cast<uintptr_t>(workspace) % kAlign != 0 ||
        (p.state_in32_hi != nullptr && p.state_in32 == nullptr) ||
        (p.capture32_hi != nullptr && p.capture32 == nullptr) ||
        (p.capture32 != nullptr && (p.capture_chunk < 0 || p.capture_chunk > (p.tokens + kBT - 1) / kBT))) {
        return cudaErrorInvalidValue;
    }
    const int64_t T = p.tokens;
    const unsigned nchunks = static_cast<unsigned>((T + kBT - 1) / kBT);
    const Layout l = layout(T);
    char *ws = static_cast<char *>(workspace);
    bf16 *qn = reinterpret_cast<bf16 *>(ws + l.qn);
    bf16 *kn = reinterpret_cast<bf16 *>(ws + l.kn);
    float *G = reinterpret_cast<float *>(ws + l.G);
    bf16 *uv = reinterpret_cast<bf16 *>(ws + l.uv);
    bf16 *w = reinterpret_cast<bf16 *>(ws + l.w);
    bf16 *hs = reinterpret_cast<bf16 *>(ws + l.hs);

    cudaError_t err = l2norm(p.q, qn, T * kHK, kD, kChunkL2Eps, stream);
    if (err != cudaSuccess) {
        return err;
    }
    err = l2norm(p.k, kn, T * kHK, kD, kChunkL2Eps, stream);
    if (err != cudaSuccess) {
        return err;
    }
    chunk_cumsum_kernel<<<dim3(nchunks, kHV), kBT, 0, stream>>>(p.g, G, T);
    chunk_kkt_wy_kernel<<<dim3(nchunks, kHV), kThreads, 0, stream>>>(kn, p.v, G, p.beta, uv, w, T);
    chunk_state_kernel<<<dim3(kD / kHBlockV, kHV), kThreads, 0, stream>>>(kn, w, uv, G, p.state, hs, T, p.state_in32,
                                                                          p.capture32, p.capture_chunk,
                                                                          p.state_in32_hi, p.capture32_hi);
    chunk_output_kernel<<<dim3(nchunks, kHV, kD / kOBlockV), kThreads, 0, stream>>>(qn, kn, uv, hs, G, p.o, T,
                                                                                    p.scale);
    return cudaGetLastError();
}

}  // namespace cere::k5
