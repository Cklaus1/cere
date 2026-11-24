// K2 fast small-M path: contract, reduction order and layout in k2_fast.h.
#include "kernels/k2_fast.h"

#include "device/context_generation.h"
#include "kernels/k2_mxfp8.cuh"
#include "kernels/k2_small_m.h"

#include <atomic>
#include <mutex>

namespace cere::kernels::k2x {
namespace {

constexpr uint32_t kStepK = 32;      // k per warp step (k2_small_m.cu kStepK)
constexpr uint32_t kMaxSlices = 16;  // one slice group (k2_small_m.h): the slices are summed in slice order
constexpr uint32_t kMaxStages = 12;  // ring stages: a chunk is S steps, so k <= 6,144 has at most 12 chunks
constexpr uint32_t kXStepBytes = kStepK * 2;  // x bytes per step and row (BF16)
constexpr uint32_t kStepBytes = kStepK;       // weight bytes per step and row (E4M3)
constexpr uint32_t kMaxDevices = 32;          // device_of's per-ordinal attribute cache
// gemm_mxfp8 runs this kernel up to this k and on launches of at least kFastRows weight rows (GDN in-projections,
// attention q|k|v), or of at least kFastRowsLowM rows at m <= kFastLowM (the shared expert's down); everything else
// runs k2s::gemm_mxfp8, which the cold-L2 graph chain measured faster or equal there (attention o / GDN out at
// k = 6,144: k2x 0.73-0.99x below m = 16; shared gate/up and the indexer: 0.84-1.16x; GPU1 2026-10-18). The resident
// layout takes the calls it serves of at least kFastRows rows at k <= kFastMaxK (the GDN in-projections, attention
// q|k|v) and of at least kResidentRowsWide rows past it (attention o / GDN out at k = 6,144), k2-decode-spec F4.
constexpr uint32_t kFastMaxK = 2560;
constexpr uint64_t kFastRows = 8192;
constexpr uint64_t kFastRowsLowM = 2048;
constexpr uint32_t kFastLowM = 8;
constexpr uint64_t kResidentRowsWide = 2560;

// The output type (kernel template argument O).
constexpr uint32_t kOutBf16 = 0;
constexpr uint32_t kOutF32 = 1;

struct LaunchSegment {
    const uint8_t *q;
    const uint8_t *scales;
    void *y;
    uint32_t n;
    uint32_t tile_begin;  // first column tile of this segment
};

struct Args {
    LaunchSegment s[kMaxSegments];
    const uint8_t *x;
    uint32_t count;
    uint32_t m;
    uint32_t k;
    uint32_t steps;        // k / 32
    uint32_t slices;       // S = k2s::split_warps(k) <= kMaxSlices
    uint32_t chunks;       // ceil(steps / S): chunk c holds steps c S .. c S + S - 1
    uint32_t stages;       // ring stages, 1..min(chunks, kMaxStages)
    uint32_t w_stride;     // shared bytes per weight row of a stage (padded: 32 mod 128)
    uint32_t x_stride;     // shared bytes per x row of a stage (padded: 32 mod 128)
    uint32_t stage_bytes;  // TN w_stride + m x_stride
    uint32_t scale_bytes;  // TN steps rounded up to 128: the tile's scales, before the ring
};

// Shared row stride for `bytes` per row: a multiple of 16 that is 32 mod 128, so a warp's fragment reads (8 rows,
// 32 or 64 consecutive bytes each) take the fewest shared wavefronts.
__host__ __device__ constexpr uint32_t padded(uint32_t bytes) { return (bytes + 127) / 128 * 128 + 32; }

__device__ __forceinline__ uint32_t smem_addr(const void *p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Weights and x are copied to shared memory with cp.async (no register staging). Weights are read once per launch:
// L1 bypass and a 256-byte L2 prefetch. Every CTA reads the same x: L1 allocation, so the CTAs of an SM share it.
__device__ __forceinline__ void copy16(uint32_t dst, const uint8_t *src) {
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}

__device__ __forceinline__ void copy16_l1(uint32_t dst, const uint8_t *src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}

__device__ __forceinline__ void copy4(uint32_t dst, const uint8_t *src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

// cp.async.wait_group takes an immediate: all but the `pending` most recent groups complete.
__device__ __forceinline__ void wait_groups(uint32_t pending) {
    switch (pending) {
    case 0: asm volatile("cp.async.wait_group 0;" ::: "memory"); break;
    case 1: asm volatile("cp.async.wait_group 1;" ::: "memory"); break;
    case 2: asm volatile("cp.async.wait_group 2;" ::: "memory"); break;
    case 3: asm volatile("cp.async.wait_group 3;" ::: "memory"); break;
    case 4: asm volatile("cp.async.wait_group 4;" ::: "memory"); break;
    case 5: asm volatile("cp.async.wait_group 5;" ::: "memory"); break;
    case 6: asm volatile("cp.async.wait_group 6;" ::: "memory"); break;
    case 7: asm volatile("cp.async.wait_group 7;" ::: "memory"); break;
    case 8: asm volatile("cp.async.wait_group 8;" ::: "memory"); break;
    case 9: asm volatile("cp.async.wait_group 9;" ::: "memory"); break;
    case 10: asm volatile("cp.async.wait_group 10;" ::: "memory"); break;
    default: asm volatile("cp.async.wait_group 11;" ::: "memory"); break;
    }
}

__device__ __forceinline__ void launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void dependency_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// k2_small_m.cu mma: d += A · B, m16n8k16 BF16, FP32 accumulation.
__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One lane's 8 E4M3 weights of a step (k = 32 s + 8 t .. + 7) with their scale byte as the 16-byte BF16 operand
// k2_small_m's lane loads: S3's conversion (k2_mxfp8.cuh), the bits of the exact upcast k2m::dequantize stores.
__device__ __forceinline__ uint4 mxfp8_operand(uint2 q, uint32_t scale) {
    return k2m::bf16x8(q, k2m::exponent_add(scale));
}

// CTA b holds column tile b (TN = 8 Q columns of one segment) and runs its S slices on S / G warps, warp v slices
// G v .. G v + G - 1 (one accumulator each). The tile's scales go to shared memory first; its weights stream through
// a ring of `stages` shared stages, one chunk (S consecutive steps: one step of every slice) per stage, with the
// chunk's x rows. Scales and weights are requested before the dependency wait, x after it. Each warp chains its
// slices' steps in increasing order; the slice results go to shared memory (over the ring) and the CTA sums them in
// slice order and stores the tile.
template <uint32_t O, uint32_t Q, bool Hi, uint32_t G>
__global__ void __launch_bounds__(kMaxSlices * 32) fast_kernel(const __grid_constant__ Args a) {
    constexpr uint32_t TN = 8 * Q;
    extern __shared__ __align__(128) uint8_t smem[];
    const uint32_t tile = blockIdx.x;
    uint32_t seg = 0;
#pragma unroll
    for (uint32_t i = 1; i < kMaxSegments; ++i) {
        if (i < a.count && tile >= a.s[i].tile_begin) {
            seg = i;
        }
    }
    const LaunchSegment &sg = a.s[seg];
    const uint32_t col0 = (tile - sg.tile_begin) * TN;
    const uint32_t rows = sg.n - col0 < TN ? sg.n - col0 : TN;  // valid weight rows of the tile
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t warps = blockDim.x / 32;
    const uint32_t S = a.slices;
    const uint32_t steps = a.steps;
    const uint32_t chunks = a.chunks;
    const uint32_t stages = a.stages;
    const uint32_t m = a.m;
    const size_t w_row = static_cast<size_t>(steps) * kStepBytes;
    const uint8_t *w_tile = sg.q + static_cast<size_t>(col0) * w_row;
    uint8_t *const ring = smem + a.scale_bytes;

    // Rows past n are not copied: their columns' outputs are not stored, and an MMA column depends on its own
    // weights only.
    const auto issue_w = [&](uint32_t c) {
        const uint32_t s0 = c * S;
        const uint32_t units = (steps - s0 < S ? steps - s0 : S) * kStepBytes / 16;
        uint8_t *dst = ring + (c % stages) * a.stage_bytes;
        for (uint32_t r = warp; r < rows; r += warps) {
            const uint8_t *src = w_tile + r * w_row + s0 * kStepBytes;
            const uint32_t d = smem_addr(dst + r * a.w_stride);
            for (uint32_t u = lane; u < units; u += 32) {
                copy16(d + 16 * u, src + 16 * u);
            }
        }
    };
    const auto issue_x = [&](uint32_t c) {
        const uint32_t s0 = c * S;
        const uint32_t units = (steps - s0 < S ? steps - s0 : S) * kXStepBytes / 16;
        uint8_t *dst = ring + (c % stages) * a.stage_bytes + TN * a.w_stride;
        for (uint32_t r = warp; r < m; r += warps) {
            const uint8_t *src = a.x + static_cast<size_t>(r) * a.k * 2 + s0 * kXStepBytes;
            const uint32_t d = smem_addr(dst + r * a.x_stride);
            for (uint32_t u = lane; u < units; u += 32) {
                copy16_l1(d + 16 * u, src + 16 * u);
            }
        }
    };

    launch_dependents();
    {
        const uint8_t *src = sg.scales + static_cast<size_t>(col0) * steps;
        const uint32_t words = steps / 4;
        for (uint32_t r = warp; r < rows; r += warps) {
            const uint32_t d = smem_addr(smem + r * steps);
            for (uint32_t u = lane; u < words; u += 32) {
                copy4(d + 4 * u, src + r * steps + 4 * u);
            }
        }
    }
    for (uint32_t c = 0; c < stages; ++c) {
        issue_w(c);
        commit();
    }
    dependency_wait();
    for (uint32_t c = 0; c < stages; ++c) {
        issue_x(c);
        commit();
    }

    const uint32_t g = lane / 4;
    const uint32_t t = lane % 4;
    const bool row0 = g < m;
    const bool row1 = Hi && g + 8 < m;
    const uint4 zero = make_uint4(0, 0, 0, 0);
    const uint32_t p0 = warp * G;
    float acc[G][Q][4] = {};
    // Chunk c's group is complete once at most stages - 1 younger groups are pending: one group is committed per
    // chunk (a refill, possibly empty), so the count is the same for every c.
    for (uint32_t c = 0; c < chunks; ++c) {
        wait_groups(stages - 1);
        __syncthreads();
        const uint8_t *ws = ring + (c % stages) * a.stage_bytes;
        const uint8_t *xs = ws + TN * a.w_stride;
#pragma unroll
        for (uint32_t i = 0; i < G; ++i) {
            const uint32_t p = p0 + i;  // slice p's step in chunk c: c S + p
            const uint32_t s = c * S + p;
            if (s < steps) {
                const uint4 xa = row0 ? *reinterpret_cast<const uint4 *>(xs + g * a.x_stride + p * kXStepBytes +
                                                                         16 * t)
                                      : zero;
                uint4 xb = zero;
                if constexpr (Hi) {
                    xb = row1 ? *reinterpret_cast<const uint4 *>(xs + (g + 8) * a.x_stride + p * kXStepBytes + 16 * t)
                              : zero;
                }
#pragma unroll
                for (uint32_t q = 0; q < Q; ++q) {
                    const uint8_t *wrow = ws + (8 * q + g) * a.w_stride + p * kStepBytes;
                    const uint4 wv = mxfp8_operand(*reinterpret_cast<const uint2 *>(wrow + 8 * t),
                                                   smem[(8 * q + g) * steps + s]);
                    mma(acc[i][q], xa.x, xb.x, xa.y, xb.y, wv.x, wv.y);
                    mma(acc[i][q], xa.z, xb.z, xa.w, xb.w, wv.z, wv.w);
                }
            }
        }
        __syncthreads();  // every warp is done with stage c % stages
        if (c + stages < chunks) {
            issue_w(c + stages);
            issue_x(c + stages);
        }
        commit();
    }

    // The slice results over the ring (the loop's last barrier passed every read of it), summed in slice order.
    float(*partial)[kMaxM][TN] = reinterpret_cast<float(*)[kMaxM][TN]>(smem);
#pragma unroll
    for (uint32_t i = 0; i < G; ++i) {
#pragma unroll
        for (uint32_t q = 0; q < Q; ++q) {
            const uint32_t cc = 8 * q + 2 * t;
            partial[p0 + i][g][cc] = acc[i][q][0];
            partial[p0 + i][g][cc + 1] = acc[i][q][1];
            partial[p0 + i][g + 8][cc] = acc[i][q][2];
            partial[p0 + i][g + 8][cc + 1] = acc[i][q][3];
        }
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < m * TN; i += blockDim.x) {
        const uint32_t r = i / TN;
        const uint32_t c = i % TN;
        if (c < rows) {
            float sum = partial[0][r][c];
            for (uint32_t p = 1; p < S; ++p) {
                sum += partial[p][r][c];
            }
            const size_t at = static_cast<size_t>(r) * sg.n + col0 + c;
            if constexpr (O == kOutF32) {
                static_cast<float *>(sg.y)[at] = sum;
            } else {
                static_cast<__nv_bfloat16 *>(sg.y)[at] = __float2bfloat16_rn(sum);
            }
        }
    }
}

// The resident layout (k2_fast.h `resident`): one slice group of 16 slices, their x fragments in registers.
constexpr uint32_t kResidentSlices = 16;

struct ResidentArgs {
    LaunchSegment s[kMaxSegments];
    const uint8_t *x;
    uint32_t count;
    uint32_t m;
    uint32_t k;
    uint32_t steps;        // k / 32 = 16 U
    uint32_t tiles;        // column tiles of every segment
    uint32_t w_stride;     // shared bytes per weight row of a stage (padded: 32 mod 128)
    uint32_t s_stride;     // shared bytes per scale row of a stage (an odd count of words)
    uint32_t stage_bytes;  // TN (w_stride + s_stride), rounded up to 128
};

// One CTA per SM and 16 warps, warp p slice p: the warp loads its slice's x fragments (steps p, p + 16, ..,
// p + 16 (U - 1); rows g and g + 8) into registers once, after the dependency wait, and keeps them for the launch.
// CTA b loops over column tiles b, b + grid, .. (TN = 8 Q columns of one segment); a tile is U / KC ring items of
// KC consecutive 16-step groups (TN contiguous row pieces of 16 KC steps and their scale bytes), D stages, the first
// D - 1 items requested before the dependency wait. After a tile's last item the slice results go to shared memory
// (after the ring), the CTA sums them in slice order and stores the tile.
template <uint32_t O, uint32_t Q, bool Hi, uint32_t U, uint32_t KC, uint32_t D>
__global__ void __launch_bounds__(kResidentSlices * 32, 1) resident_kernel(const __grid_constant__ ResidentArgs a) {
    static_assert(U % KC == 0 && D >= 2 && D - 2 <= 11, "whole items per tile, a ring wait_groups serves");
    constexpr uint32_t NC = U / KC;
    constexpr uint32_t TN = 8 * Q;
    constexpr uint32_t kUnits = KC * kResidentSlices * kStepBytes / 16;  // 16-byte weight copies per row and item
    constexpr uint32_t kWords = KC * kResidentSlices / 4;                // 4-byte scale copies per row and item
    extern __shared__ __align__(128) uint8_t smem[];
    float(*partial)[kMaxM][TN] = reinterpret_cast<float(*)[kMaxM][TN]>(smem + D * a.stage_bytes);
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t p = threadIdx.x / 32;
    const uint32_t g = lane / 4;
    const uint32_t t = lane % 4;
    const uint32_t grid = gridDim.x;
    const uint32_t tiles = (a.tiles - blockIdx.x + grid - 1) / grid;  // this CTA's
    const uint32_t items = tiles * NC;
    const size_t w_row = a.k;  // E4M3 bytes per weight row
    const uint32_t steps = a.steps;
    const uint32_t m = a.m;

    const auto segment = [&](uint32_t tile) -> const LaunchSegment & {
        uint32_t s = 0;
#pragma unroll
        for (uint32_t i = 1; i < kMaxSegments; ++i) {
            if (i < a.count && tile >= a.s[i].tile_begin) {
                s = i;
            }
        }
        return a.s[s];
    };
    // Item it (local tile it / NC, step groups (it % NC) KC ..) into stage it % D. Rows past n are not copied: their
    // columns' outputs are not stored, and an MMA column depends on its own weights only.
    const auto issue = [&](uint32_t it) {
        const uint32_t tile = blockIdx.x + (it / NC) * grid;
        const uint32_t c = it % NC;
        const LaunchSegment &sg = segment(tile);
        const uint32_t col0 = (tile - sg.tile_begin) * TN;
        const uint32_t rows = sg.n - col0 < TN ? sg.n - col0 : TN;
        uint8_t *const st = smem + (it % D) * a.stage_bytes;
        const uint8_t *w = sg.q + static_cast<size_t>(col0) * w_row + c * (KC * kResidentSlices * kStepBytes);
        for (uint32_t i = threadIdx.x; i < rows * kUnits; i += blockDim.x) {
            const uint32_t r = i / kUnits;
            const uint32_t u = i % kUnits;
            copy16(smem_addr(st + r * a.w_stride + 16 * u), w + r * w_row + 16 * u);
        }
        const uint8_t *s = sg.scales + static_cast<size_t>(col0) * steps + c * (KC * kResidentSlices);
        uint8_t *const sd = st + TN * a.w_stride;
        for (uint32_t i = threadIdx.x; i < rows * kWords; i += blockDim.x) {
            const uint32_t r = i / kWords;
            const uint32_t u = i % kWords;
            copy4(smem_addr(sd + r * a.s_stride + 4 * u), s + r * steps + 4 * u);
        }
    };

    launch_dependents();
#pragma unroll
    for (uint32_t it = 0; it < D - 1; ++it) {
        if (it < items) {
            issue(it);
        }
        commit();
    }
    dependency_wait();

    const bool row0 = g < m;
    const bool row1 = Hi && g + 8 < m;
    const uint4 zero = make_uint4(0, 0, 0, 0);
    uint4 xa[U];
    uint4 xb[U];
#pragma unroll
    for (uint32_t u = 0; u < U; ++u) {
        const uint8_t *xs = a.x + (static_cast<size_t>(p + kResidentSlices * u) * kStepK) * 2 + 16 * t;
        xa[u] = row0 ? *reinterpret_cast<const uint4 *>(xs + static_cast<size_t>(g) * a.k * 2) : zero;
        xb[u] = row1 ? *reinterpret_cast<const uint4 *>(xs + static_cast<size_t>(g + 8) * a.k * 2) : zero;
    }

    for (uint32_t lt = 0; lt < tiles; ++lt) {
        float acc[Q][4] = {};
#pragma unroll
        for (uint32_t c = 0; c < NC; ++c) {
            // Item it is complete once at most D - 2 younger groups are pending (one group per item, possibly
            // empty); after the barrier every warp is done with item it - 1, whose stage the refill takes.
            const uint32_t it = lt * NC + c;
            wait_groups(D - 2);
            __syncthreads();
            if (it + D - 1 < items) {
                issue(it + D - 1);
            }
            commit();
            const uint8_t *ws = smem + (it % D) * a.stage_bytes;
            const uint8_t *ss = ws + TN * a.w_stride;
#pragma unroll
            for (uint32_t i = 0; i < KC; ++i) {
                const uint32_t u = c * KC + i;
                const uint32_t s = kResidentSlices * i + p;  // the slice's step within the item
#pragma unroll
                for (uint32_t q = 0; q < Q; ++q) {
                    const uint32_t r = 8 * q + g;
                    const uint4 wv = mxfp8_operand(
                        *reinterpret_cast<const uint2 *>(ws + r * a.w_stride + s * kStepBytes + 8 * t),
                        ss[r * a.s_stride + s]);
                    mma(acc[q], xa[u].x, xb[u].x, xa[u].y, xb[u].y, wv.x, wv.y);
                    mma(acc[q], xa[u].z, xb[u].z, xa[u].w, xb[u].w, wv.z, wv.w);
                }
            }
        }
        // The previous tile's sums finished before this tile's first barrier.
#pragma unroll
        for (uint32_t q = 0; q < Q; ++q) {
            const uint32_t cc = 8 * q + 2 * t;
            partial[p][g][cc] = acc[q][0];
            partial[p][g][cc + 1] = acc[q][1];
            partial[p][g + 8][cc] = acc[q][2];
            partial[p][g + 8][cc + 1] = acc[q][3];
        }
        __syncthreads();
        const uint32_t tile = blockIdx.x + lt * grid;
        const LaunchSegment &sg = segment(tile);
        const uint32_t col0 = (tile - sg.tile_begin) * TN;
        const uint32_t rows = sg.n - col0 < TN ? sg.n - col0 : TN;
        for (uint32_t i = threadIdx.x; i < m * TN; i += blockDim.x) {
            const uint32_t r = i / TN;
            const uint32_t c = i % TN;
            if (c < rows) {
                float sum = partial[0][r][c];
#pragma unroll
                for (uint32_t s = 1; s < kResidentSlices; ++s) {
                    sum += partial[s][r][c];
                }
                const size_t at = static_cast<size_t>(r) * sg.n + col0 + c;
                if constexpr (O == kOutF32) {
                    static_cast<float *>(sg.y)[at] = sum;
                } else {
                    static_cast<__nv_bfloat16 *>(sg.y)[at] = __float2bfloat16_rn(sum);
                }
            }
        }
    }
    wait_groups(0);  // the trailing (empty) groups
}

bool aligned(const void *p, uintptr_t to) { return reinterpret_cast<uintptr_t>(p) % to == 0; }

struct Device {
    int id;
    int sms;
    int smem_per_sm;     // shared memory per SM
    int smem_optin;      // dynamic shared memory per CTA with the opt-in
    int smem_reserved;   // per CTA, reserved by the system
};

// The current device's attributes, queried once per device (eager decode makes hundreds of K2 launches per step).
bool device_of(Device &d) {
    static std::atomic<bool> cached[kMaxDevices]{};
    static Device known[kMaxDevices]{};
    static std::mutex fill;
    int id = 0;
    if (cudaGetDevice(&id) != cudaSuccess || id < 0) {
        return false;
    }
    const bool cacheable = id < static_cast<int>(kMaxDevices);
    if (cacheable && cached[id].load(std::memory_order_acquire)) {
        d = known[id];
        return true;
    }
    d.id = id;
    if (cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, id) != cudaSuccess ||
        cudaDeviceGetAttribute(&d.smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, id) != cudaSuccess ||
        cudaDeviceGetAttribute(&d.smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, id) != cudaSuccess ||
        cudaDeviceGetAttribute(&d.smem_reserved, cudaDevAttrReservedSharedMemoryPerBlock, id) != cudaSuccess ||
        d.sms <= 0 || d.smem_optin <= 0) {
        return false;
    }
    if (cacheable) {
        const std::lock_guard<std::mutex> lock(fill);
        if (!cached[id].load(std::memory_order_relaxed)) {
            known[id] = d;
            cached[id].store(true, std::memory_order_release);
        }
    }
    return true;
}

// One launch of `kernel` (a fast_kernel or resident_kernel instance); its dynamic shared memory limit is raised once
// per device context (`opted`, one entry per ordinal below kTrackedDevices: the ordinal's context generation + 1
// (device/context_generation.h) in which the limit was raised, 0 = never) to the opt-in maximum: the limit is state
// of the CUDA context, so a reset primary context (release_device=1) raises it again, as p3b_decode.cu's launch.
using Opted = std::atomic<uint32_t>[cere::device::kTrackedDevices];

template <class A>
cudaError_t launch_kernel(void (*kernel)(A), Opted &opted, const A &a, const Device &dev, uint32_t grid,
                          uint32_t block, uint32_t smem, cudaStream_t stream) {
    const bool tracked = dev.id >= 0 && dev.id < cere::device::kTrackedDevices;
    const uint32_t generation = tracked ? cere::device::context_generation(dev.id).load() + 1 : 0;
    if (!tracked || opted[dev.id].load(std::memory_order_acquire) != generation) {
        const cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dev.smem_optin);
        if (e != cudaSuccess) {
            return e;
        }
        if (tracked) {
            opted[dev.id].store(generation, std::memory_order_release);
        }
    }
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(grid, 1, 1);
    config.blockDim = dim3(block, 1, 1);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;
    const cudaError_t e = cudaLaunchKernelEx(&config, kernel, a);
    const cudaError_t last = cudaGetLastError();
    return e != cudaSuccess ? e : last;
}

template <uint32_t O, uint32_t Q, bool Hi, uint32_t G>
cudaError_t launch_with(const Args &a, const Device &dev, uint32_t tiles, uint32_t smem, cudaStream_t stream) {
    static Opted opted{};
    return launch_kernel(fast_kernel<O, Q, Hi, G>, opted, a, dev, tiles, 32 * a.slices / G, smem, stream);
}

template <uint32_t O, uint32_t Q, uint32_t G>
cudaError_t launch_hi(const Args &a, const Device &dev, uint32_t tiles, uint32_t smem, cudaStream_t stream) {
    return a.m > 8 ? launch_with<O, Q, true, G>(a, dev, tiles, smem, stream)
                   : launch_with<O, Q, false, G>(a, dev, tiles, smem, stream);
}

template <uint32_t O, uint32_t Q>
cudaError_t launch_g(const Args &a, const Device &dev, uint32_t lock, uint32_t tiles, uint32_t smem,
                     cudaStream_t stream) {
    return lock == 1   ? launch_hi<O, Q, 1>(a, dev, tiles, smem, stream)
           : lock == 2 ? launch_hi<O, Q, 2>(a, dev, tiles, smem, stream)
                       : launch_hi<O, Q, 4>(a, dev, tiles, smem, stream);
}

template <uint32_t O>
cudaError_t launch(Args &a, const uint32_t *ns, const Tune &tune, cudaStream_t stream) {
    Device dev{};
    if (!device_of(dev)) {
        return cudaErrorInvalidValue;
    }
    uint32_t q = tune.q;
    if (q == 0) {
        // Wider tiles read x (every CTA's) fewer times per weight byte: 32-column tiles past m = 8 when they give
        // every SM a CTA, else 16-column tiles when they give every other SM one, else 8-column tiles (GPU1 sweep
        // 2026-10-17).
        const auto tiles_of = [&](uint32_t cols) {
            uint64_t t = 0;
            for (uint32_t i = 0; i < a.count; ++i) {
                t += (uint64_t{ns[i]} + cols - 1) / cols;
            }
            return t;
        };
        const auto sms = static_cast<uint64_t>(dev.sms);
        q = a.m > 8 && tiles_of(32) >= sms ? 4 : 2 * tiles_of(16) >= sms ? 2 : 1;
    }
    // 8-column tiles (a grid of few tiles) take two slices per warp, half the warps per CTA (back-to-back graph chain
    // sweep, GPU1 2026-10-17: -0.2..-1.5 us per launch on the router, shared expert and indexer shapes, the other
    // shapes within noise).
    const uint32_t lock = tune.lock != 0 ? tune.lock : q == 1 && a.slices % 2 == 0 ? 2 : 1;
    if ((q != 1 && q != 2 && q != 4) || (lock != 1 && lock != 2 && lock != 4) || a.slices % lock != 0) {
        return cudaErrorInvalidValue;
    }
    const uint32_t tn = 8 * q;
    a.chunks = (a.steps + a.slices - 1) / a.slices;
    a.w_stride = padded(a.slices * kStepBytes);
    a.x_stride = padded(a.slices * kXStepBytes);
    a.stage_bytes = tn * a.w_stride + a.m * a.x_stride;
    a.scale_bytes = (tn * a.steps + 127) / 128 * 128;
    const uint32_t partial = kMaxSlices * kMaxM * tn * 4;
    const auto smem_of = [&](uint32_t stages) {
        const uint32_t ring = a.scale_bytes + stages * a.stage_bytes;
        return ring > partial ? ring : partial;
    };
    uint64_t tiles = 0;
    for (uint32_t i = 0; i < a.count; ++i) {
        a.s[i].tile_begin = static_cast<uint32_t>(tiles);
        tiles += (uint64_t{ns[i]} + tn - 1) / tn;
    }
    if (tiles > 0x7fffffffu) {
        return cudaErrorInvalidValue;
    }
    const uint32_t most = a.chunks < kMaxStages ? a.chunks : kMaxStages;
    uint32_t stages = tune.stages;
    if (stages == 0) {
        // Two stages when the grid fills every SM twice (more resident CTAs beat a deeper ring, GPU1 sweep
        // 2026-10-17); a smaller grid takes the deepest ring that keeps two CTAs per SM.
        stages = 2;
        if (tiles < 2 * static_cast<uint64_t>(dev.sms)) {
            const uint32_t budget =
                static_cast<uint32_t>(dev.smem_per_sm) / 2 - static_cast<uint32_t>(dev.smem_reserved);
            stages = most;
            while (stages > 2 && smem_of(stages) > budget) {
                --stages;
            }
        }
        stages = stages < most ? stages : most;
    }
    if (stages < 1 || stages > most || smem_of(stages) > static_cast<uint32_t>(dev.smem_optin)) {
        return cudaErrorInvalidValue;
    }
    a.stages = stages;
    const auto n = static_cast<uint32_t>(tiles);
    const uint32_t smem = smem_of(stages);
    return q == 4   ? launch_g<O, 4>(a, dev, lock, n, smem, stream)
           : q == 2 ? launch_g<O, 2>(a, dev, lock, n, smem, stream)
                    : launch_g<O, 1>(a, dev, lock, n, smem, stream);
}

template <uint32_t O, uint32_t Q, bool Hi, uint32_t U, uint32_t KC, uint32_t D>
cudaError_t resident_with(const ResidentArgs &a, const Device &dev, uint32_t grid, uint32_t smem,
                          cudaStream_t stream) {
    static Opted opted{};
    return launch_kernel(resident_kernel<O, Q, Hi, U, KC, D>, opted, a, dev, grid, kResidentSlices * 32, smem, stream);
}

template <uint32_t O, uint32_t Q, uint32_t U, uint32_t KC, uint32_t D>
cudaError_t resident_hi(const ResidentArgs &a, const Device &dev, uint32_t grid, uint32_t smem, cudaStream_t stream) {
    return a.m > 8 ? resident_with<O, Q, true, U, KC, D>(a, dev, grid, smem, stream)
                   : resident_with<O, Q, false, U, KC, D>(a, dev, grid, smem, stream);
}

// The instantiated resident layouts (k2_fast.h ResidentTune), U = k / 512; cudaErrorInvalidValue for any other.
template <uint32_t O>
cudaError_t resident_launch(const ResidentArgs &a, uint32_t u, const ResidentTune &l, const Device &dev,
                            uint32_t grid, uint32_t smem, cudaStream_t stream) {
    const auto is = [&](uint32_t uu, uint32_t q, uint32_t kc, uint32_t stages) {
        return u == uu && l.q == q && l.kc == kc && l.stages == stages;
    };
    if (is(5, 2, 1, 8)) {
        return resident_hi<O, 2, 5, 1, 8>(a, dev, grid, smem, stream);
    }
    if (is(5, 1, 5, 4)) {
        return resident_hi<O, 1, 5, 5, 4>(a, dev, grid, smem, stream);
    }
    if (is(12, 1, 4, 4)) {
        return resident_hi<O, 1, 12, 4, 4>(a, dev, grid, smem, stream);
    }
    if (is(12, 2, 3, 3)) {
        return resident_hi<O, 2, 12, 3, 3>(a, dev, grid, smem, stream);
    }
    if (is(12, 1, 3, 6)) {
        return resident_hi<O, 1, 12, 3, 6>(a, dev, grid, smem, stream);
    }
    return cudaErrorInvalidValue;
}

// gemm's contract, the tune aside.
bool contract_ok(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const k2s::Mxfp8Segment *segments, uint32_t count) {
    if (x == nullptr || !aligned(x, 16) || m == 0 || m > kMaxM || !supported(k) || segments == nullptr ||
        count == 0 || count > kMaxSegments) {
        return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const k2s::Mxfp8Segment &s = segments[i];
        if (s.q == nullptr || !aligned(s.q, 16) || s.scales == nullptr || !aligned(s.scales, 4) || s.y == nullptr ||
            s.n == 0) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool supported(uint32_t k) {
    return k2s::supported(k) && k2s::split_warps(k) <= kMaxSlices && k % (4 * kStepK) == 0;
}

cudaError_t gemm(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const k2s::Mxfp8Segment *segments, uint32_t count,
                 bool f32, cudaStream_t stream, const Tune &tune) {
    if (!contract_ok(x, m, k, segments, count)) {
        return cudaErrorInvalidValue;
    }
    Args a{};
    a.m = m;
    a.k = k;
    a.steps = k / kStepK;
    a.slices = k2s::split_warps(k);
    a.x = reinterpret_cast<const uint8_t *>(x);
    a.count = count;
    uint32_t ns[kMaxSegments];
    for (uint32_t i = 0; i < count; ++i) {
        const k2s::Mxfp8Segment &s = segments[i];
        a.s[i] = LaunchSegment{s.q, s.scales, s.y, s.n, 0};
        ns[i] = s.n;
    }
    return f32 ? launch<kOutF32>(a, ns, tune, stream) : launch<kOutBf16>(a, ns, tune, stream);
}

bool resident_supported(uint32_t k) {
    const uint32_t groups = k / (kResidentSlices * kStepK);
    return supported(k) && k2s::split_warps(k) == kResidentSlices && k % (kResidentSlices * kStepK) == 0 &&
           (groups == 5 || groups == 12);
}

cudaError_t resident(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const k2s::Mxfp8Segment *segments,
                     uint32_t count, bool f32, cudaStream_t stream, const ResidentTune &tune) {
    if (!contract_ok(x, m, k, segments, count) || !resident_supported(k)) {
        return cudaErrorInvalidValue;
    }
    Device dev{};
    if (!device_of(dev)) {
        return cudaErrorInvalidValue;
    }
    const uint32_t u = k / (kResidentSlices * kStepK);
    ResidentTune l = tune;
    if (l.q == 0 && l.kc == 0 && l.stages == 0) {
        // k = 2,560: 16-column tiles, one 16-step group per item, 8 stages; k = 6,144: 8-column tiles of 4-group
        // items, 4 stages, to m = 8, else 16-column tiles of 3-group items, 3 stages (GPU sweep, k2-decode-spec F4).
        l = u == 5 ? ResidentTune{2, 1, 8} : m > 8 ? ResidentTune{2, 3, 3} : ResidentTune{1, 4, 4};
    }
    if ((l.q != 1 && l.q != 2) || l.kc == 0 || u % l.kc != 0 || l.stages < 2 || l.stages > 13) {
        return cudaErrorInvalidValue;
    }
    ResidentArgs a{};
    a.x = reinterpret_cast<const uint8_t *>(x);
    a.count = count;
    a.m = m;
    a.k = k;
    a.steps = k / kStepK;
    const uint32_t tn = 8 * l.q;
    uint64_t tiles = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const k2s::Mxfp8Segment &s = segments[i];
        a.s[i] = LaunchSegment{s.q, s.scales, s.y, s.n, static_cast<uint32_t>(tiles)};
        tiles += (uint64_t{s.n} + tn - 1) / tn;
    }
    if (tiles > 0x7fffffffu) {
        return cudaErrorInvalidValue;
    }
    a.tiles = static_cast<uint32_t>(tiles);
    a.w_stride = padded(l.kc * kResidentSlices * kStepBytes);
    a.s_stride = (l.kc * kResidentSlices / 4 + 1) * 4;  // an odd count of words: 4 kc + 1
    a.stage_bytes = (tn * (a.w_stride + a.s_stride) + 127) / 128 * 128;
    const uint64_t smem = uint64_t{l.stages} * a.stage_bytes + kResidentSlices * kMaxM * tn * 4;
    if (smem > static_cast<uint64_t>(dev.smem_optin)) {
        return cudaErrorInvalidValue;
    }
    const uint32_t grid = a.tiles < static_cast<uint32_t>(dev.sms) ? a.tiles : static_cast<uint32_t>(dev.sms);
    const auto bytes = static_cast<uint32_t>(smem);
    return f32 ? resident_launch<kOutF32>(a, u, l, dev, grid, bytes, stream)
               : resident_launch<kOutBf16>(a, u, l, dev, grid, bytes, stream);
}

cudaError_t gemm_mxfp8(const __nv_bfloat16 *x, uint32_t m, uint32_t k, const k2s::Mxfp8Segment *segments,
                       uint32_t count, bool f32, cudaStream_t stream) {
    uint64_t rows = 0;
    for (uint32_t i = 0; count <= kMaxSegments && segments != nullptr && i < count; ++i) {
        rows += segments[i].n;
    }
    if (!contract_ok(x, m, k, segments, count)) {
        return k2s::gemm_mxfp8(x, m, k, segments, count, f32, stream);
    }
    if (resident_supported(k) && rows >= (k > kFastMaxK ? kResidentRowsWide : kFastRows)) {
        return resident(x, m, k, segments, count, f32, stream);
    }
    const bool fast = k <= kFastMaxK && (rows >= kFastRows || (rows >= kFastRowsLowM && m <= kFastLowM));
    return fast ? gemm(x, m, k, segments, count, f32, stream) : k2s::gemm_mxfp8(x, m, k, segments, count, f32, stream);
}

}  // namespace cere::kernels::k2x
