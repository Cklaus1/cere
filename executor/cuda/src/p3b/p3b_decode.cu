// P3b decode GEMMs (p3b_decode.h; m23 design note P14). One warp owns 16 weight rows (the MMA's M; GEMM1: the
// up rows j..j+15 and the gate rows I+j..I+j+15) over the whole K, so its accumulators see the k64 blocks in
// FlashInfer's order; the expert's tokens are the MMA's N (n8 tiles, at most kMaxTiles). Weights stream through a
// per-warp cp.async ring of stages of kStageK k64 blocks (16 rows x 64 bytes, 16-byte chunks XOR-swizzled by row
// bits 1..2 so ldmatrix reads them without bank conflicts) with their scale-factor words; the tokens and theirs are
// staged once per CTA. Every weight and scale-factor copy carries the L2::256B prefetch hint: a row's first copy pulls
// its next 256 bytes into L2, so DRAM streams whole 256-byte blocks rather than the 32- / 4-byte pieces one stage
// takes (the cold-L2 rate of the GEMM1 pattern: 0.57 TB/s without the hint, 1.32 with it, m23 design note P14).
#include "p3b/p3b_decode.h"

#include "device/context_generation.h"

#include "cutlass/array.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "tensorrt_llm/kernels/quantization_utils.cuh"

#include <algorithm>
#include <atomic>

namespace cere::p3b::decode {
namespace {

namespace tkk = tensorrt_llm::kernels;

constexpr int kWarp = 32;
constexpr int kTile = 16;                          // MMA M: weight rows per warp
constexpr int kStepBytes = 32;                     // one row's bytes per k64 block (64 FP4)
constexpr int kStageK = 2;                         // k64 blocks per ring stage
constexpr int kRowStage = kStageK * kStepBytes;    // 64: one row's bytes per stage (four 16-byte chunks)
constexpr int kTileBytes = kTile * kRowStage;      // 1024: one weight tile's stage
constexpr int kSfTileBytes = kTile * kStageK * 4;  // one 4-byte word (a k64 block's 4 UE4M3) per row and block
constexpr int kMaxTiles = 4;                       // n8 token tiles: m <= 32
constexpr int kTokenPad = 16;                      // staged token rows: stride row_bytes + 16 (bank spread)
constexpr int kScratchBytes = 2 * 8 * kTile * 2;   // per warp: two BF16 [8 tokens, 16 rows] epilogue tiles
constexpr int kMaxSmem = 99 * 1024;                // sm_120 opt-in shared memory per block

// GEMM1: kWarpsA warps per CTA, each one 16-row block of the intermediate (two weight tiles per stage),
// stages_a(tiles, one_wave) stages in flight per warp. 5 x 4 (the model's inter 640: 8 CTAs per expert) over P14b's
// 2 x 8 (20 CTAs): cold-L2 gemm1 + gemm2 pair -13 % at m 2, -5 % at m 4, -3 % at m 16, even at m 1 and 8 (O72,
// m23 design note P14c); each warp's chain is unchanged, so the outputs are bit-identical. At 1 and 2 token tiles
// (m <= 16) when the grid exceeds one CTA per SM, the ring is 3 and 2 stages deep, so two CTAs fit an SM (MG-02:
// <= 50,176 bytes each of sm_120's 102,400 with the 1 KB per-CTA reserve): a step's CTAs run in one wave, and the SM
// keeps as many bytes in flight; a grid within one CTA per SM keeps the 4-deep ring.
constexpr int kWarpsA = 5;
constexpr int kMaxStagesA = 4;
__host__ __device__ constexpr int stages_a(int tiles, bool one_wave) {
    return one_wave ? kMaxStagesA : tiles == 1 ? 3 : tiles == 2 ? 2 : kMaxStagesA;
}
constexpr int kStageA = 2 * kTileBytes + 2 * kSfTileBytes;
// GEMM2: kWarpsB warps per CTA, each 16 output rows (one weight tile per stage); kStagesB stages hold the whole K of
// the model's GEMM2 (inter 640: 5 stages), all in flight before the kernel waits on gemm1_act.
constexpr int kWarpsB = 4;
constexpr int kStagesB = 5;
constexpr int kStageB = kTileBytes + kSfTileBytes;

__device__ __forceinline__ uint32_t smem_addr(const void *p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// The copies' L2::256B prefetch: the L2 fills the aligned 256-byte block around the source (the weight streams'
// DRAM rate, see the file comment).
__device__ __forceinline__ void cp_async16(void *dst, const void *src) {
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async4(void *dst, const void *src) {
    asm volatile("cp.async.ca.shared.global.L2::256B [%0], [%1], 4;\n" ::"r"(smem_addr(dst)), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr)
                 : "memory");
}

// SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue4m3_t, 16>::fma (cute/arch/mma_sm120.hpp): d = a.b +
// d with the k16 block scales `sfa` (row (lane & 1) * 8 + lane / 4) and `sfb` (column lane / 4), byte / thread id 0.
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa,
                                    uint32_t sfb) {
    constexpr uint16_t kId = 0;
    const float c0 = d[0], c1 = d[1], c2 = d[2], c3 = d[3];
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa), "h"(kId), "h"(kId), "r"(sfb), "h"(kId), "h"(kId));
}

// The TMA warp-specialized epilogue's LinearCombination (Sm90LinearCombination, ElementSource = BF16 with C not
// loaded: its fragment cleared): Z = alpha * acc, I = beta * 0 + Z (beta = 0, a runtime value there as here), D = BF16
// round-to-nearest of I.
__device__ __forceinline__ __nv_bfloat16 epilogue(float alpha, float acc, float beta) {
    const float z = cutlass::multiplies<float>{}(alpha, acc);
    const float i = cutlass::multiply_add<float>{}(beta, 0.0f, z);
    return __float2bfloat16_rn(i);
}

// get_sf_out_offset_128x4 (quantization_utils.cuh) of row `row`, k64 block `kb` (its 4 k16 scale factors are one
// aligned word) in a matrix of `ktiles` k64 blocks per row.
__device__ __forceinline__ int64_t sf_word(int64_t row, int kb, int ktiles) {
    return (row >> 7) * ktiles * 512 + int64_t{kb} * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4;
}

// Whether stage `s` (k64 block s) enters the MMA chain: every block, unless kDropLast (Order::kDropLast, the ktest's
// fail-first perturbation) skips the last one.
template <bool kDropLast>
__device__ __forceinline__ bool k_used(int s, int ks) {
    return !kDropLast || s + 1 < ks;
}

// A tile row's 16-byte chunk `c` (k64 block c / 2, half c % 2) in its stage slot: chunks XOR-swizzled by row bits 1..2,
// so the 8 rows of one ldmatrix matrix (rows 64 bytes apart: two per 128-byte bank line) hit 8 distinct bank quads.
__device__ __forceinline__ int tile_offset(int r, int c) { return r * kRowStage + ((c ^ ((r >> 1) & 3)) << 4); }

// One weight tile's stage (k64 blocks k0 .. k0 + kStageK - 1) into its ring slot: lane l copies chunk l % 4 of rows
// l / 4 and l / 4 + 8.
__device__ __forceinline__ void issue_tile(uint8_t *slot, const uint8_t *rows, int row_bytes, int k0, int lane) {
    const int c = lane & 3;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int r = (lane >> 2) + 8 * half;
        cp_async16(slot + tile_offset(r, c), rows + static_cast<int64_t>(r) * row_bytes + k0 * kStepBytes + c * 16);
    }
}

// The A fragment (a0..a3: rows g / g + 8, k 8t.. / 32 + 8t..) of k64 block `b` of a tile slot: ldmatrix lane i
// addresses row (i & 7) + 8 * ((i >> 3) & 1) of half (i >> 4).
__device__ __forceinline__ void load_a(uint32_t (&a)[4], const uint8_t *slot, int b, int lane) {
    const int q = lane >> 3;
    const int r = (lane & 7) + ((q & 1) << 3);
    const int h = q >> 1;
    ldmatrix_x4(a, smem_addr(slot + tile_offset(r, 2 * b + h)));
}

// The expert's token rows [first, first + rows) of `src` (row_bytes each) and their scale-factor words into the
// staged token block (rows_pad rows, zeros past `rows`). `sf_word_of(r, kb)`: the word's byte offset in `src_sf`.
template <typename SfWord>
__device__ __forceinline__ void issue_tokens(uint8_t *tok, uint32_t *tok_sf, const uint8_t *src, const uint8_t *src_sf,
                                             int rows, int rows_pad, int row_bytes, int ks, SfWord sf_word_of) {
    const int chunks = row_bytes / 16;
    const int stride = row_bytes + kTokenPad;
    for (int i = threadIdx.x; i < rows_pad * chunks; i += blockDim.x) {
        const int r = i / chunks;
        const int c = i % chunks;
        uint8_t *dst = tok + r * stride + c * 16;
        if (r < rows) {
            cp_async16(dst, src + static_cast<int64_t>(r) * row_bytes + c * 16);
        } else {
            *reinterpret_cast<uint4 *>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
    for (int i = threadIdx.x; i < rows_pad * ks; i += blockDim.x) {
        const int r = i / ks;
        const int kb = i % ks;
        uint32_t *dst = tok_sf + r * (ks + 1) + kb;
        if (r < rows) {
            cp_async4(dst, src_sf + sf_word_of(r, kb));
        } else {
            *dst = 0;
        }
    }
}

// The B fragment of token tile `i` at k64 block `k`: b0 / b1 = token g's k 8t.. / 32 + 8t.., sfb its scales.
__device__ __forceinline__ void load_b(uint32_t &b0, uint32_t &b1, uint32_t &sfb, const uint8_t *tok,
                                       const uint32_t *tok_sf, int i, int k, int row_bytes, int ks, int lane) {
    const int n = i * 8 + (lane >> 2);
    const uint8_t *p = tok + n * (row_bytes + kTokenPad) + k * kStepBytes + (lane & 3) * 4;
    b0 = *reinterpret_cast<const uint32_t *>(p);
    b1 = *reinterpret_cast<const uint32_t *>(p + 16);
    sfb = tok_sf[n * (ks + 1) + k];
}

// The accumulator tile (d0..d3: row g / g + 8, token 2t / 2t + 1) as BF16 D into a [8 tokens, 16 rows] scratch.
__device__ __forceinline__ void store_d(__nv_bfloat16 *scratch, const float (&d)[4], float alpha, float beta,
                                        int lane) {
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        const int row = (lane >> 2) + 8 * (v >> 1);
        const int n = 2 * (lane & 3) + (v & 1);
        scratch[n * kTile + row] = epilogue(alpha, d[v], beta);
    }
}

__host__ __device__ size_t token_bytes(int rows_pad, int row_bytes, int ks) {
    return size_t(rows_pad) * (row_bytes + kTokenPad) + size_t(rows_pad) * (ks + 1) * 4;
}

int tiles_of(int64_t m) { return static_cast<int>((m + 7) / 8); }

size_t smem_a(const Shape &s, int stages) {
    return token_bytes(8 * tiles_of(s.m), static_cast<int>(s.hidden / 2), static_cast<int>(s.hidden / 64)) +
           size_t(kWarpsA) * (size_t(stages) * kStageA + kScratchBytes);
}

size_t smem_b(const Shape &s) {
    return token_bytes(8 * tiles_of(s.m), static_cast<int>(s.inter / 2), static_cast<int>(s.inter / 64)) +
           size_t(kWarpsB) * (size_t(kStagesB) * kStageB + kScratchBytes);
}

// GEMM1 + SwiGLU + FP4 quantize; warp w of CTA x owns the intermediate columns j0 = 16 * (x * kWarpsA + w) ..
// j0 + 15 of slot blockIdx.y's expert. `ready` (nullable): per-slot counts of finished CTAs, zero at entry.
template <bool kDisableFastMath, bool kDropLast, int kTiles, int kStagesA>
__global__ void __launch_bounds__(kWarpsA *kWarp)
    gemm1_act_kernel(const Group *__restrict__ groups, const uint8_t *__restrict__ x, const uint8_t *__restrict__ x_sf,
                     const uint8_t *__restrict__ w13, const uint8_t *__restrict__ w13_sf,
                     const float *__restrict__ alpha, const float *__restrict__ quant_scale,
                     const float *__restrict__ act_scale, bool per_expert_act_scale, uint8_t *__restrict__ act,
                     uint8_t *__restrict__ act_sf, uint32_t *__restrict__ ready, int hidden, int inter, float beta) {
    extern __shared__ __align__(128) uint8_t smem[];
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    const Group g = groups[blockIdx.y];
    // gemm2 may launch now (its CTAs read the group table, which small_prologue completed before the wait above, and
    // stream their weights before they wait on this grid's completion).
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    if (g.expert < 0) {
        return;
    }
    const int warp = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    const int ks = hidden / 64;
    const int stages = ks / kStageK;
    const int row_bytes = hidden / 2;
    const int rows_pad = 8 * kTiles;
    const int rows = min(g.rows, rows_pad);
    uint8_t *tok = smem;
    uint32_t *tok_sf = reinterpret_cast<uint32_t *>(tok + rows_pad * (row_bytes + kTokenPad));
    uint8_t *ring = smem + token_bytes(rows_pad, row_bytes, ks) + warp * (kStagesA * kStageA + kScratchBytes);
    auto *scratch = reinterpret_cast<__nv_bfloat16 *>(ring + kStagesA * kStageA);

    // GEMM1's A: small_prologue's FP4 rows and their scale factors (getOffsetActivationSF(e, first, hidden)), the
    // first commit group.
    const int j0 = (blockIdx.x * kWarpsA + warp) * kTile;
    const int64_t e = g.expert;
    const int64_t x_sf_base = (g.first + e * 127 + 127) / 128 * 128 * (hidden / 16);
    issue_tokens(tok, tok_sf, x + int64_t{g.first} * row_bytes, x_sf + x_sf_base, rows, rows_pad, row_bytes, ks,
                 [ks](int r, int kb) { return sf_word(r, kb, ks); });
    cp_commit();
    const uint8_t *up = w13 + (e * 2 * inter + j0) * row_bytes;
    const uint8_t *gate = up + int64_t{inter} * row_bytes;
    const uint8_t *w_sf = w13_sf + e * 2 * inter * (hidden / 16);
    const auto issue = [&](int s) {
        const int k0 = s * kStageK;
        uint8_t *slot = ring + (s % kStagesA) * kStageA;
        issue_tile(slot, up, row_bytes, k0, lane);
        issue_tile(slot + kTileBytes, gate, row_bytes, k0, lane);
        const int t = lane >> 4;  // lanes 0..15: up rows, 16..31: gate rows
        const int r = lane & 15;
#pragma unroll
        for (int b = 0; b < kStageK; ++b) {
            cp_async4(slot + 2 * kTileBytes + t * kSfTileBytes + (b * kTile + r) * 4,
                      w_sf + sf_word(t * inter + j0 + r, k0 + b, ks));
        }
    };
#pragma unroll 1
    for (int s = 0; s < kStagesA; ++s) {
        if (s < stages) issue(s);
        cp_commit();
    }
    cp_wait<kStagesA>();
    __syncthreads();

    const int tiles = (rows + 7) / 8;
    float acc_up[kTiles][4] = {};
    float acc_gate[kTiles][4] = {};
    const int sfa_row = ((lane & 1) << 3) + (lane >> 2);
#pragma unroll 1
    for (int s = 0; s < stages; ++s) {
        cp_wait<kStagesA - 1>();
        __syncwarp();
        const uint8_t *slot = ring + (s % kStagesA) * kStageA;
#pragma unroll
        for (int b = 0; b < kStageK; ++b) {
            const int k = s * kStageK + b;
            uint32_t a_up[4], a_gate[4];
            load_a(a_up, slot, b, lane);
            load_a(a_gate, slot + kTileBytes, b, lane);
            const uint8_t *sf = slot + 2 * kTileBytes + (b * kTile + sfa_row) * 4;
            const uint32_t sfa_up = *reinterpret_cast<const uint32_t *>(sf);
            const uint32_t sfa_gate = *reinterpret_cast<const uint32_t *>(sf + kSfTileBytes);
#pragma unroll
            for (int i = 0; i < kTiles; ++i) {
                if (i < tiles && k_used<kDropLast>(k, ks)) {
                    uint32_t b0, b1, sfb;
                    load_b(b0, b1, sfb, tok, tok_sf, i, k, row_bytes, ks, lane);
                    mma(acc_up[i], a_up, b0, b1, sfa_up, sfb);
                    mma(acc_gate[i], a_gate, b0, b1, sfa_gate, sfb);
                }
            }
        }
        __syncwarp();
        if (s + kStagesA < stages) issue(s + kStagesA);
        cp_commit();
    }

    // doActivationKernel's body on the BF16 GEMM1 output of this warp's 16 columns: lane 2n + h (n < 8) takes token
    // n's elements j0 + 8h .. j0 + 8h + 7 (lanes 16..31 repeat lanes 0..15 for the pair shuffle and store nothing).
    using GemmResultElem = cutlass::Array<__nv_bfloat16, 8>;
    using ComputeElem = cutlass::Array<float, 8>;
    const float a = alpha[e];
    const size_t act_scale_idx = per_expert_act_scale ? e : 0;
    const float q = quant_scale ? quant_scale[act_scale_idx] : 1.f;
    const float global_scale_val = act_scale ? act_scale[act_scale_idx] : 1.0f;
    const int n = (lane >> 1) & 7;
    const int h = lane & 1;
    cutlass::NumericArrayConverter<float, __nv_bfloat16, 8> to_float;
#pragma unroll
    for (int i = 0; i < kTiles; ++i) {
        if (i < tiles) {
            store_d(scratch, acc_up[i], a, beta, lane);
            store_d(scratch + 8 * kTile, acc_gate[i], a, beta, lane);
            __syncwarp();
            const ComputeElem linear_value =
                to_float(*reinterpret_cast<const GemmResultElem *>(scratch + n * kTile + 8 * h));
            const ComputeElem fc1_value =
                to_float(*reinterpret_cast<const GemmResultElem *>(scratch + 8 * kTile + n * kTile + 8 * h));
            // GLUAdaptor<cutlass::epilogue::thread::SiLu>: fn(gate) * linear.
            const ComputeElem gate_act = cutlass::epilogue::thread::SiLu<ComputeElem>{}(fc1_value) * linear_value;
            const ComputeElem post_act_val = gate_act * q;
            tkk::PackedVec<__nv_bfloat16> packed_vec{};
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                packed_vec.elts[v].x = static_cast<__nv_bfloat16>(post_act_val[v * 2 + 0]);
                packed_vec.elts[v].y = static_cast<__nv_bfloat16>(post_act_val[v * 2 + 1]);
            }
            const int token = i * 8 + n;
            const bool store = lane < 16 && token < rows;
            const int64_t row = int64_t{g.first} + token;
            uint8_t *sf_out = store && h == 0 ? act_sf + row * (inter / 16) + j0 / 16 : nullptr;
            const uint32_t res = tkk::cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, 8, false, kDisableFastMath>(
                packed_vec, global_scale_val, sf_out, 0.0f);
            if (store) {
                *reinterpret_cast<uint32_t *>(act + row * (inter / 2) + j0 / 2 + 4 * h) = res;
            }
            __syncwarp();
        }
    }
    // The slot's gemm2 CTAs wait on its count of finished CTAs (gridDim.x), not on the whole grid: every thread's
    // stores, then one release by thread 0 (the cooperative grid barrier's pattern).
    if (ready != nullptr) {
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            atomicAdd(ready + blockIdx.y, 1u);
        }
    }
}

// GEMM2; warp w of CTA x owns the output columns h0 = 16 * (x * kWarpsB + w) .. h0 + 15 of slot blockIdx.y's
// expert. The group and the weight stages are read before the wait on gemm1_act (small_prologue, which wrote the
// groups, completed before any gemm1_act CTA passed its own wait). `ready` null: the wait is on the whole gemm1_act
// grid; else on the slot's count reaching `ready_target` (its gemm1_act CTAs, all resident or done: this grid
// launches only after every gemm1_act CTA has triggered, so the spin cannot starve them).
template <bool kDropLast, int kTiles>
__global__ void __launch_bounds__(kWarpsB *kWarp)
    gemm2_kernel(const Group *__restrict__ groups, const uint8_t *__restrict__ act, const uint8_t *__restrict__ act_sf,
                 const uint8_t *__restrict__ down, const uint8_t *__restrict__ down_sf, const float *__restrict__ alpha,
                 __nv_bfloat16 *__restrict__ out, const uint32_t *__restrict__ ready, uint32_t ready_target,
                 int hidden, int inter, float beta) {
    extern __shared__ __align__(128) uint8_t smem[];
    const Group g = groups[blockIdx.y];
    if (g.expert < 0) {
        return;
    }
    const int warp = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    const int ks = inter / 64;
    const int stages = ks / kStageK;
    const int row_bytes = inter / 2;
    const int rows_pad = 8 * kTiles;
    const int rows = min(g.rows, rows_pad);
    uint8_t *tok = smem;
    uint32_t *tok_sf = reinterpret_cast<uint32_t *>(tok + rows_pad * (row_bytes + kTokenPad));
    uint8_t *ring = smem + token_bytes(rows_pad, row_bytes, ks) + warp * (kStagesB * kStageB + kScratchBytes);
    auto *scratch = reinterpret_cast<__nv_bfloat16 *>(ring + kStagesB * kStageB);

    const int h0 = (blockIdx.x * kWarpsB + warp) * kTile;
    const int64_t e = g.expert;
    const uint8_t *w = down + (e * hidden + h0) * row_bytes;
    const uint8_t *w_sf = down_sf + e * hidden * (inter / 16);
    const auto issue = [&](int s) {
        const int k0 = s * kStageK;
        uint8_t *slot = ring + (s % kStagesB) * kStageB;
        issue_tile(slot, w, row_bytes, k0, lane);
        const int b = lane >> 4;
        const int r = lane & 15;
        if (b < kStageK) {
            cp_async4(slot + kTileBytes + (b * kTile + r) * 4, w_sf + sf_word(h0 + r, k0 + b, ks));
        }
    };
#pragma unroll 1
    for (int s = 0; s < kStagesB; ++s) {
        if (s < stages) issue(s);
        cp_commit();
    }
    if (ready != nullptr) {
        if (threadIdx.x == 0) {
            uint32_t done = 0;
            for (;;) {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(done) : "l"(ready + blockIdx.y) : "memory");
                if (done >= ready_target) {
                    break;
                }
                __nanosleep(32);
            }
            __threadfence();
        }
        __syncthreads();
    } else {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaGridDependencySynchronize();
#endif
    }
    // GEMM2's A: gemm1_act's compact FP4 rows and scale factors.
    const int sf_row = inter / 16;
    issue_tokens(tok, tok_sf, act + int64_t{g.first} * row_bytes, act_sf + int64_t{g.first} * sf_row, rows, rows_pad,
                 row_bytes, ks, [sf_row](int r, int kb) { return int64_t{r} * sf_row + 4 * kb; });
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    const int tiles = (rows + 7) / 8;
    float acc[kTiles][4] = {};
    const int sfa_row = ((lane & 1) << 3) + (lane >> 2);
#pragma unroll 1
    for (int s = 0; s < stages; ++s) {
        cp_wait<kStagesB - 1>();
        __syncwarp();
        const uint8_t *slot = ring + (s % kStagesB) * kStageB;
#pragma unroll
        for (int b = 0; b < kStageK; ++b) {
            const int k = s * kStageK + b;
            uint32_t a[4];
            load_a(a, slot, b, lane);
            const uint32_t sfa = *reinterpret_cast<const uint32_t *>(slot + kTileBytes + (b * kTile + sfa_row) * 4);
#pragma unroll
            for (int i = 0; i < kTiles; ++i) {
                if (i < tiles && k_used<kDropLast>(k, ks)) {
                    uint32_t b0, b1, sfb;
                    load_b(b0, b1, sfb, tok, tok_sf, i, k, row_bytes, ks, lane);
                    mma(acc[i], a, b0, b1, sfa, sfb);
                }
            }
        }
        __syncwarp();
        if (s + kStagesB < stages) issue(s + kStagesB);
        cp_commit();
    }

    // D rows of token n: lane 2n + h stores columns h0 + 8h .. h0 + 8h + 7.
    const float a_e = alpha[e];
    const int n = (lane >> 1) & 7;
    const int h = lane & 1;
#pragma unroll
    for (int i = 0; i < kTiles; ++i) {
        if (i < tiles) {
            store_d(scratch, acc[i], a_e, beta, lane);
            __syncwarp();
            const int token = i * 8 + n;
            if (lane < 16 && token < rows) {
                *reinterpret_cast<uint4 *>(out + (int64_t{g.first} + token) * hidden + h0 + 8 * h) =
                    *reinterpret_cast<const uint4 *>(scratch + n * kTile + 8 * h);
            }
            __syncwarp();
        }
    }
}

__global__ void compact_act_sf_kernel(const int64_t *__restrict__ offsets, const uint8_t *__restrict__ sf,
                                      int64_t rows, int inter, int64_t experts, uint8_t *__restrict__ out) {
    const int cols = inter / 16;
    const int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i >= rows * cols) {
        return;
    }
    const int64_t p = i / cols;
    const int c = static_cast<int>(i % cols);
    // The expert of permuted row p: the last e with offsets[e] <= p (offsets[experts] = rows > p).
    int64_t lo = 0;
    int64_t hi = experts;
    while (lo < hi) {
        const int64_t mid = (lo + hi + 1) / 2;
        if (offsets[mid] <= p) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const int64_t base = (offsets[lo] + lo * 127 + 127) / 128 * 128 * cols;
    const int64_t r = p - offsets[lo];
    out[i] = sf[base + sf_word(r, c / 4, cols / 4) + c % 4];
}

// cudaLaunchKernelEx with programmatic stream serialization = pdl, after raising the kernel's dynamic shared memory
// limit to `smem` on the current device (per kernel and device: cudaFuncSetAttribute holds for the current device's
// context only, so each card raises its own (m32n design note D4), and a released context (D10, release_device=1)
// is raised again). The static is the instantiation's for `kKernel`, one entry per ordinal below
// cere::device::kTrackedDevices: the ordinal's context generation (device/context_generation.h) in the high 32 bits,
// the limit raised in that generation in the low 32; raised once per larger size per generation.
template <auto kKernel, typename... Args>
cere_status launch(dim3 grid, int threads, size_t smem, bool pdl, cudaStream_t stream, Args... args) {
    static std::atomic<uint64_t> raised[cere::device::kTrackedDevices]{};
    if (smem > 48 * 1024) {
        int ordinal = -1;
        if (cudaGetDevice(&ordinal) != cudaSuccess || ordinal < 0 || ordinal >= cere::device::kTrackedDevices) {
            cudaGetLastError();
            return CERE_ERR_INTERNAL;
        }
        const uint64_t generation = cere::device::context_generation(ordinal).load();
        const uint64_t entry = raised[ordinal].load();
        if ((entry >> 32) != generation || smem > (entry & 0xffffffffu)) {
            if (cudaFuncSetAttribute(kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)) !=
                cudaSuccess) {
                return CERE_ERR_INTERNAL;
            }
            raised[ordinal].store((generation << 32) | static_cast<uint32_t>(smem));
        }
    }
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(threads);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl;
    config.numAttrs = 1;
    config.attrs = attrs;
    return cudaLaunchKernelEx(&config, kKernel, args...) == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

template <bool kDisableFastMath, bool kDropLast, int kTiles, int kStagesA>
cere_status gemm1_act_tiles(dim3 grid, size_t smem, bool pdl, cudaStream_t stream, const Group *groups,
                            const uint8_t *x, const uint8_t *x_sf, const uint8_t *w13, const uint8_t *w13_sf,
                            const float *alpha, const float *quant_scale, const float *act_scale,
                            bool per_expert_act_scale, uint8_t *act, uint8_t *act_sf, uint32_t *ready, int hidden,
                            int inter) {
    return launch<&gemm1_act_kernel<kDisableFastMath, kDropLast, kTiles, kStagesA>>(
        grid, kWarpsA * kWarp, smem, pdl, stream, groups, x, x_sf, w13, w13_sf, alpha, quant_scale, act_scale,
        per_expert_act_scale, act, act_sf, ready, hidden, inter, 0.0f);
}

template <bool kDisableFastMath, bool kDropLast>
cere_status gemm1_act_order(int tiles, bool one_wave, dim3 grid, size_t smem, bool pdl, cudaStream_t stream,
                            const Group *groups, const uint8_t *x, const uint8_t *x_sf, const uint8_t *w13,
                            const uint8_t *w13_sf, const float *alpha, const float *quant_scale,
                            const float *act_scale, bool per_expert_act_scale, uint8_t *act, uint8_t *act_sf,
                            uint32_t *ready, int hidden, int inter) {
    switch (tiles) {
    case 1:
        return one_wave ? gemm1_act_tiles<kDisableFastMath, kDropLast, 1, stages_a(1, true)>(
                              grid, smem, pdl, stream, groups, x, x_sf, w13, w13_sf, alpha, quant_scale, act_scale,
                              per_expert_act_scale, act, act_sf, ready, hidden, inter)
                        : gemm1_act_tiles<kDisableFastMath, kDropLast, 1, stages_a(1, false)>(
                              grid, smem, pdl, stream, groups, x, x_sf, w13, w13_sf, alpha, quant_scale, act_scale,
                              per_expert_act_scale, act, act_sf, ready, hidden, inter);
    case 2:
        return one_wave ? gemm1_act_tiles<kDisableFastMath, kDropLast, 2, stages_a(2, true)>(
                              grid, smem, pdl, stream, groups, x, x_sf, w13, w13_sf, alpha, quant_scale, act_scale,
                              per_expert_act_scale, act, act_sf, ready, hidden, inter)
                        : gemm1_act_tiles<kDisableFastMath, kDropLast, 2, stages_a(2, false)>(
                              grid, smem, pdl, stream, groups, x, x_sf, w13, w13_sf, alpha, quant_scale, act_scale,
                              per_expert_act_scale, act, act_sf, ready, hidden, inter);
    case 3:
        return gemm1_act_tiles<kDisableFastMath, kDropLast, 3, kMaxStagesA>(grid, smem, pdl, stream, groups, x, x_sf,
                                                                            w13, w13_sf, alpha, quant_scale,
                                                                            act_scale, per_expert_act_scale, act,
                                                                            act_sf, ready, hidden, inter);
    case 4:
        return gemm1_act_tiles<kDisableFastMath, kDropLast, 4, kMaxStagesA>(grid, smem, pdl, stream, groups, x, x_sf,
                                                                            w13, w13_sf, alpha, quant_scale,
                                                                            act_scale, per_expert_act_scale, act,
                                                                            act_sf, ready, hidden, inter);
    default:
        return CERE_ERR_ARGUMENT;
    }
}

// The current device's SM count (cached per ordinal below cere::device::kTrackedDevices); 0 when unknown.
int sm_count() {
    static std::atomic<int> counts[cere::device::kTrackedDevices]{};
    int ordinal = -1;
    if (cudaGetDevice(&ordinal) != cudaSuccess || ordinal < 0 || ordinal >= cere::device::kTrackedDevices) {
        cudaGetLastError();
        return 0;
    }
    int count = counts[ordinal].load();
    if (count == 0) {
        if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, ordinal) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        counts[ordinal].store(count);
    }
    return count;
}

template <bool kDropLast, int kTiles>
cere_status gemm2_tiles(dim3 grid, size_t smem, bool pdl, cudaStream_t stream, const Group *groups,
                        const uint8_t *act, const uint8_t *act_sf, const uint8_t *down, const uint8_t *down_sf,
                        const float *alpha, __nv_bfloat16 *out, const uint32_t *ready, uint32_t ready_target,
                        int hidden, int inter) {
    return launch<&gemm2_kernel<kDropLast, kTiles>>(grid, kWarpsB * kWarp, smem, pdl, stream, groups, act, act_sf, down,
                                                    down_sf, alpha, out, ready, ready_target, hidden, inter, 0.0f);
}

template <bool kDropLast>
cere_status gemm2_order(int tiles, dim3 grid, size_t smem, bool pdl, cudaStream_t stream, const Group *groups,
                        const uint8_t *act, const uint8_t *act_sf, const uint8_t *down, const uint8_t *down_sf,
                        const float *alpha, __nv_bfloat16 *out, const uint32_t *ready, uint32_t ready_target,
                        int hidden, int inter) {
    switch (tiles) {
    case 1:
        return gemm2_tiles<kDropLast, 1>(grid, smem, pdl, stream, groups, act, act_sf, down, down_sf, alpha, out, ready,
                                         ready_target, hidden, inter);
    case 2:
        return gemm2_tiles<kDropLast, 2>(grid, smem, pdl, stream, groups, act, act_sf, down, down_sf, alpha, out, ready,
                                         ready_target, hidden, inter);
    case 3:
        return gemm2_tiles<kDropLast, 3>(grid, smem, pdl, stream, groups, act, act_sf, down, down_sf, alpha, out, ready,
                                         ready_target, hidden, inter);
    case 4:
        return gemm2_tiles<kDropLast, 4>(grid, smem, pdl, stream, groups, act, act_sf, down, down_sf, alpha, out, ready,
                                         ready_target, hidden, inter);
    default:
        return CERE_ERR_ARGUMENT;
    }
}

}  // namespace

bool supported(const Shape &s) {
    return s.m >= 1 && s.top_k >= 1 && s.experts >= 1 && s.m * s.top_k <= kMaxRows && tiles_of(s.m) <= kMaxTiles &&
           s.hidden > 0 && s.inter > 0 && s.hidden % (64 * kStageK) == 0 && s.inter % (64 * kStageK) == 0 &&
           s.inter % (kTile * kWarpsA) == 0 && s.hidden % (kTile * kWarpsB) == 0 &&
           smem_a(s, kMaxStagesA) <= kMaxSmem && smem_b(s) <= kMaxSmem;
}

int64_t slots(const Shape &s) { return std::min(s.m * s.top_k, s.experts); }

cere_status gemm1_act(const Group *groups, const uint8_t *x, const uint8_t *x_sf, const uint8_t *w13,
                      const uint8_t *w13_sf, const float *alpha, const float *quant_scale, const float *act_scale,
                      bool per_expert_act_scale, bool disable_fast_math, uint8_t *act, uint8_t *act_sf, uint32_t *ready,
                      const Shape &s, Order order, bool pdl, cudaStream_t stream) {
    if (!supported(s) || groups == nullptr || x == nullptr || x_sf == nullptr || w13 == nullptr || w13_sf == nullptr ||
        alpha == nullptr || act == nullptr || act_sf == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    const dim3 grid(static_cast<unsigned>(s.inter / kTile / kWarpsA), static_cast<unsigned>(slots(s)));
    const int tiles = tiles_of(s.m);
    const bool one_wave = int64_t(grid.x) * grid.y <= sm_count();
    const size_t smem = smem_a(s, stages_a(tiles, one_wave));
    ready = one_wave ? nullptr : ready;  // gemm2 waits on the whole grid then (see gemm2)
    const int hidden = static_cast<int>(s.hidden);
    const int inter = static_cast<int>(s.inter);
    const bool drop = order == Order::kDropLast;
    if (disable_fast_math) {
        return drop ? gemm1_act_order<true, true>(tiles, one_wave, grid, smem, pdl, stream, groups, x, x_sf, w13,
                                                      w13_sf, alpha, quant_scale, act_scale, per_expert_act_scale, act,
                                                      act_sf, ready, hidden, inter)
                        : gemm1_act_order<true, false>(tiles, one_wave, grid, smem, pdl, stream, groups, x, x_sf, w13,
                                                       w13_sf, alpha, quant_scale, act_scale, per_expert_act_scale,
                                                       act, act_sf, ready, hidden, inter);
    }
    return drop ? gemm1_act_order<false, true>(tiles, one_wave, grid, smem, pdl, stream, groups, x, x_sf, w13, w13_sf,
                                                   alpha, quant_scale, act_scale, per_expert_act_scale, act, act_sf,
                                                   ready, hidden, inter)
                    : gemm1_act_order<false, false>(tiles, one_wave, grid, smem, pdl, stream, groups, x, x_sf, w13,
                                                    w13_sf, alpha, quant_scale, act_scale, per_expert_act_scale, act,
                                                    act_sf, ready, hidden, inter);
}

cere_status gemm2(const Group *groups, const uint8_t *act, const uint8_t *act_sf, const uint8_t *down,
                  const uint8_t *down_sf, const float *alpha, __nv_bfloat16 *out, const uint32_t *ready,
                  const Shape &s, Order order, bool pdl, cudaStream_t stream) {
    if (!supported(s) || groups == nullptr || act == nullptr || act_sf == nullptr || down == nullptr ||
        down_sf == nullptr || alpha == nullptr || out == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    const dim3 grid(static_cast<unsigned>(s.hidden / kTile / kWarpsB), static_cast<unsigned>(slots(s)));
    const int tiles = tiles_of(s.m);
    const size_t smem = smem_b(s);
    const int hidden = static_cast<int>(s.hidden);
    const int inter = static_cast<int>(s.inter);
    // gemm1_act's CTAs per slot (its grid's x). A gemm1_act grid within one CTA per SM finishes as one wave, and the
    // whole-grid wait is the cheaper one there (MG-03: per-slot counters +1.9 us at m 1, 10 slots).
    const auto ready_target = static_cast<uint32_t>(s.inter / kTile / kWarpsA);
    const uint32_t *wait_on = int64_t(ready_target) * grid.y <= sm_count() ? nullptr : ready;
    return order == Order::kDropLast
               ? gemm2_order<true>(tiles, grid, smem, pdl, stream, groups, act, act_sf, down, down_sf, alpha, out,
                                   wait_on, ready_target, hidden, inter)
               : gemm2_order<false>(tiles, grid, smem, pdl, stream, groups, act, act_sf, down, down_sf, alpha, out,
                                    wait_on, ready_target, hidden, inter);
}

cere_status compact_act_sf(const int64_t *expert_first_token_offset, const uint8_t *sf, int64_t rows, int64_t inter,
                           int64_t experts, uint8_t *out, cudaStream_t stream) {
    if (expert_first_token_offset == nullptr || sf == nullptr || out == nullptr || rows <= 0 || inter <= 0 ||
        inter % 64 != 0 || experts <= 0) {
        return CERE_ERR_ARGUMENT;
    }
    const int64_t n = rows * (inter / 16);
    compact_act_sf_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        expert_first_token_offset, sf, rows, static_cast<int>(inter), experts, out);
    return cudaGetLastError() == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

}  // namespace cere::p3b::decode
