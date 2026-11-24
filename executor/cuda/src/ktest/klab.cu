// KLAB kernel-lab entry points (m215 design note D2; suite executor/tests/gpu/test_klab_omma.py, `--gpu 1 test
// --suite klab`; prototypes in its docstring). libcere_ktest.so only. Device pointers, int64 sizes, the stream last;
// 0 = CERE_OK, CERE_ERR_ARGUMENT for arguments outside the contract, CERE_ERR_CAPACITY for a failed allocation or
// graph instantiation, CERE_ERR_INTERNAL for any other CUDA error.
//
// cere_ktest_klab_omma: one `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.
// e2m1.f32.ue4m3` (SASS OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X) per warp, instruction i on warp i. Every operand is
// the raw per-lane register: a [n][32][4], b [n][32][2], scale_a and scale_b [n][32], c and d [n][32][4] (lane-major),
// so the fragment layout and the scale-lane mapping are the caller's to pack and to probe. The selectors are PTX
// immediates; only PTX ISA 9.7.14.3 Table 37's values for scale_vec::4X are accepted (byte-id 0, thread-id-a {0, 1},
// thread-id-b {0, 1, 2, 3}); any other value is CERE_ERR_ARGUMENT before a launch (others are undefined behaviour).
//
// cere_ktest_klab_calib: the measured rows (F5), cudaEvent timing, one untimed warm-up and 5 timed repetitions.
// out (host double [7]): out[0] the work of one repetition (FLOP, bytes, barriers or nodes), out[1..5] the elapsed
// milliseconds of each repetition, out[6] the launch shape (blocks; row 4: the host milliseconds of
// cudaGraphInstantiate; row 5: 0).
//   row 0: OMMA NVFP4 m16n8k64, 8 independent chains per warp, SMs * size blocks of 128 threads (size 1..64).
//   row 1: HMMA BF16 m16n8k16, the same loop.
//   row 2: HBM read of an 8 GiB buffer (__ldcs, uint4), SMs * size blocks of 512 threads (size 1..64).
//   row 3: cooperative grid.sync, 128 threads per block; size 0 = one block per SM, 1 = full occupancy. Each
//          repetition is a 10,001-barrier launch minus a 1-barrier launch, so out[0] = 10,000 barriers.
//   row 4: CUDA graph replay of `size` 1-block no-op kernel nodes (size 1..1,000,000), captured on a private stream.
//   row 5: the same `size` no-op kernels launched one by one on a private stream.
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>

#include "cere_executor.h"

namespace {

namespace cg = cooperative_groups;

constexpr int kOmmaThreads = 128;
constexpr int kLoopThreads = 128;
constexpr int kLoopChains = 8;
constexpr int kLoopIters = 20000;
constexpr int kReadThreads = 512;
constexpr size_t kReadBytes = size_t{8} << 30;
constexpr int kSyncThreads = 128;
constexpr int kSyncIters = 10001;
constexpr int kReps = 5;
constexpr int64_t kMaxNodes = 1000000;

int status_of(cudaError_t err) {
    if (err == cudaSuccess) return CERE_OK;
    if (err == cudaErrorMemoryAllocation) return CERE_ERR_CAPACITY;
    return CERE_ERR_INTERNAL;
}

bool aligned(const void *p, uintptr_t bytes) { return p != nullptr && reinterpret_cast<uintptr_t>(p) % bytes == 0; }

__device__ __forceinline__ void omma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1, uint32_t sa, uint32_t sb) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

template <int kThreadA, int kThreadB>
__global__ void __launch_bounds__(kOmmaThreads)
    omma_batch(const uint4 *__restrict__ a, const uint2 *__restrict__ b, const uint32_t *__restrict__ scale_a,
               const uint32_t *__restrict__ scale_b, const float4 *__restrict__ c, float4 *__restrict__ d, int64_t n) {
    const int64_t warp = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    if (warp >= n) return;  // whole warps only: blockDim is a multiple of 32
    const int64_t t = warp * 32 + (threadIdx.x & 31);
    const uint4 av = a[t];
    const uint2 bv = b[t];
    const float4 cv = c[t];
    float acc[4] = {cv.x, cv.y, cv.z, cv.w};
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%12, %14};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(av.x), "r"(av.y), "r"(av.z), "r"(av.w), "r"(bv.x), "r"(bv.y), "r"(scale_a[t]), "r"(scale_b[t]), "n"(0),
          "n"(kThreadA), "n"(kThreadB));
    d[t] = make_float4(acc[0], acc[1], acc[2], acc[3]);
}

template <int kThreadA>
cudaError_t launch_omma(int64_t thread_id_b, dim3 grid, cudaStream_t stream, const uint4 *a, const uint2 *b,
                        const uint32_t *sa, const uint32_t *sb, const float4 *c, float4 *d, int64_t n) {
    switch (thread_id_b) {
        case 0: omma_batch<kThreadA, 0><<<grid, kOmmaThreads, 0, stream>>>(a, b, sa, sb, c, d, n); break;
        case 1: omma_batch<kThreadA, 1><<<grid, kOmmaThreads, 0, stream>>>(a, b, sa, sb, c, d, n); break;
        case 2: omma_batch<kThreadA, 2><<<grid, kOmmaThreads, 0, stream>>>(a, b, sa, sb, c, d, n); break;
        default: omma_batch<kThreadA, 3><<<grid, kOmmaThreads, 0, stream>>>(a, b, sa, sb, c, d, n); break;
    }
    return cudaGetLastError();
}

__device__ __forceinline__ void hmma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                     uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The probe's issue-rate loop (probe_calib.cu mma_loop): 8 independent accumulator chains per warp; the sum is
// stored only on an impossible value so the loop is not eliminated.
template <bool kFp4>
__global__ void __launch_bounds__(kLoopThreads) mma_loop(int iters, uint32_t seed, float *sink) {
    float d[kLoopChains][4] = {};
    const uint32_t a = seed ^ threadIdx.x;
    const uint32_t b = seed * 3u + threadIdx.x;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int c = 0; c < kLoopChains; ++c) {
            if (kFp4) {
                omma(d[c], a, a + c, a, b, b, a, 0x38383838u, 0x38383838u);
            } else {
                hmma(d[c], a, a + c, a, b, b, a);
            }
        }
    }
    float s = 0.f;
    for (int c = 0; c < kLoopChains; ++c) s += d[c][0] + d[c][1] + d[c][2] + d[c][3];
    if (s == 12345.f) sink[0] = s;
}

__global__ void read_bw(const uint4 *__restrict__ p, size_t n, unsigned long long *sink) {
    uint32_t acc = 0;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const uint4 v = __ldcs(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) atomicAdd(sink, 1ull);
}

__global__ void grid_sync(int iters, int *buf) {
    cg::grid_group g = cg::this_grid();
    for (int i = 0; i < iters; ++i) {
        if (threadIdx.x == 0) buf[blockIdx.x] += i;
        g.sync();
    }
}

__global__ void noop(int *p) {
    if (threadIdx.x == 0 && p != nullptr) p[blockIdx.x] = 0;
}

// Events and buffers of one calibration call, released on every path.
struct Calib {
    cudaEvent_t e0 = nullptr;
    cudaEvent_t e1 = nullptr;
    void *buf = nullptr;
    void *sink = nullptr;
    cudaStream_t own = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~Calib() {
        if (exec != nullptr) cudaGraphExecDestroy(exec);
        if (graph != nullptr) cudaGraphDestroy(graph);
        if (own != nullptr) {
            cudaStreamSynchronize(own);
            cudaStreamDestroy(own);
        }
        if (buf != nullptr) cudaFree(buf);
        if (sink != nullptr) cudaFree(sink);
        if (e0 != nullptr) cudaEventDestroy(e0);
        if (e1 != nullptr) cudaEventDestroy(e1);
    }
};

#define KLAB_CK(expr)                                  \
    do {                                               \
        const cudaError_t klab_err = (expr);           \
        if (klab_err != cudaSuccess) {                 \
            cudaGetLastError();                        \
            return status_of(klab_err);                \
        }                                              \
    } while (0)

// Milliseconds between e0 and e1 around `launch` (which records nothing itself) on `stream`.
template <typename Launch>
cudaError_t timed(Calib &k, cudaStream_t stream, Launch launch, float *ms) {
    cudaError_t err;
    if ((err = cudaEventRecord(k.e0, stream)) != cudaSuccess) return err;
    if ((err = launch()) != cudaSuccess) return err;
    if ((err = cudaEventRecord(k.e1, stream)) != cudaSuccess) return err;
    if ((err = cudaEventSynchronize(k.e1)) != cudaSuccess) return err;
    return cudaEventElapsedTime(ms, k.e0, k.e1);
}

int calib(int64_t row, int64_t size, double *out, cudaStream_t stream) {
    int dev = 0;
    KLAB_CK(cudaGetDevice(&dev));
    int sms = 0;
    KLAB_CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    Calib k;
    KLAB_CK(cudaEventCreate(&k.e0));
    KLAB_CK(cudaEventCreate(&k.e1));
    float ms = 0.f;
    if (row == 0 || row == 1) {
        const int blocks = sms * static_cast<int>(size);
        KLAB_CK(cudaMalloc(&k.sink, sizeof(float)));
        float *sink = static_cast<float *>(k.sink);
        auto launch = [&] {
            if (row == 0) {
                mma_loop<true><<<blocks, kLoopThreads, 0, stream>>>(kLoopIters, 7u, sink);
            } else {
                mma_loop<false><<<blocks, kLoopThreads, 0, stream>>>(kLoopIters, 7u, sink);
            }
            return cudaGetLastError();
        };
        const double flop_per = row == 0 ? 16.0 * 8 * 64 * 2 : 16.0 * 8 * 16 * 2;
        out[0] = flop_per * kLoopChains * kLoopIters * (blocks * (kLoopThreads / 32.0));
        out[6] = blocks;
        KLAB_CK(timed(k, stream, launch, &ms));
        for (int r = 0; r < kReps; ++r) {
            KLAB_CK(timed(k, stream, launch, &ms));
            out[1 + r] = ms;
        }
        return CERE_OK;
    }
    if (row == 2) {
        const int blocks = sms * static_cast<int>(size);
        KLAB_CK(cudaMalloc(&k.buf, kReadBytes));
        KLAB_CK(cudaMalloc(&k.sink, sizeof(unsigned long long)));
        KLAB_CK(cudaMemsetAsync(k.buf, 1, kReadBytes, stream));
        const uint4 *p = static_cast<const uint4 *>(k.buf);
        auto *sink = static_cast<unsigned long long *>(k.sink);
        auto launch = [&] {
            read_bw<<<blocks, kReadThreads, 0, stream>>>(p, kReadBytes / sizeof(uint4), sink);
            return cudaGetLastError();
        };
        out[0] = static_cast<double>(kReadBytes);
        out[6] = blocks;
        KLAB_CK(timed(k, stream, launch, &ms));
        for (int r = 0; r < kReps; ++r) {
            KLAB_CK(timed(k, stream, launch, &ms));
            out[1 + r] = ms;
        }
        return CERE_OK;
    }
    if (row == 3) {
        int cooperative = 0;
        KLAB_CK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, dev));
        if (!cooperative) return CERE_ERR_UNSUPPORTED;
        int per_sm = 1;
        if (size == 1) KLAB_CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, grid_sync, kSyncThreads, 0));
        const int blocks = sms * per_sm;
        KLAB_CK(cudaMalloc(&k.buf, static_cast<size_t>(blocks) * sizeof(int)));
        KLAB_CK(cudaMemsetAsync(k.buf, 0, static_cast<size_t>(blocks) * sizeof(int), stream));
        int *buf = static_cast<int *>(k.buf);
        auto launch_iters = [&](int iters) {
            return [&, iters]() mutable {
                void *args[] = {&iters, &buf};
                return cudaLaunchCooperativeKernel(reinterpret_cast<void *>(grid_sync), blocks, kSyncThreads, args, 0,
                                                   stream);
            };
        };
        out[0] = kSyncIters - 1;
        out[6] = blocks;
        float one = 0.f;
        KLAB_CK(timed(k, stream, launch_iters(kSyncIters), &ms));
        for (int r = 0; r < kReps; ++r) {
            KLAB_CK(timed(k, stream, launch_iters(1), &one));
            KLAB_CK(timed(k, stream, launch_iters(kSyncIters), &ms));
            out[1 + r] = static_cast<double>(ms) - one;
        }
        return CERE_OK;
    }
    // Rows 4 and 5: stream capture needs a stream other than the legacy default stream.
    const int nodes = static_cast<int>(size);
    KLAB_CK(cudaStreamCreateWithFlags(&k.own, cudaStreamNonBlocking));
    KLAB_CK(cudaMalloc(&k.buf, sizeof(int)));
    int *buf = static_cast<int *>(k.buf);
    auto launch_all = [&] {
        for (int i = 0; i < nodes; ++i) noop<<<1, 32, 0, k.own>>>(buf);
        return cudaGetLastError();
    };
    out[0] = nodes;
    if (row == 5) {
        out[6] = 0;
        KLAB_CK(timed(k, k.own, launch_all, &ms));
        for (int r = 0; r < kReps; ++r) {
            KLAB_CK(timed(k, k.own, launch_all, &ms));
            out[1 + r] = ms;
        }
        return CERE_OK;
    }
    KLAB_CK(cudaStreamBeginCapture(k.own, cudaStreamCaptureModeThreadLocal));
    const cudaError_t captured = launch_all();
    const cudaError_t ended = cudaStreamEndCapture(k.own, &k.graph);
    KLAB_CK(captured);
    KLAB_CK(ended);
    KLAB_CK(cudaStreamSynchronize(k.own));
    const auto t0 = std::chrono::steady_clock::now();
    const cudaError_t inst = cudaGraphInstantiate(&k.exec, k.graph, 0);
    const auto t1 = std::chrono::steady_clock::now();
    if (inst != cudaSuccess) {
        cudaGetLastError();
        return inst == cudaErrorMemoryAllocation ? CERE_ERR_CAPACITY : CERE_ERR_INTERNAL;
    }
    out[6] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto replay = [&] { return cudaGraphLaunch(k.exec, k.own); };
    KLAB_CK(timed(k, k.own, replay, &ms));
    for (int r = 0; r < kReps; ++r) {
        KLAB_CK(timed(k, k.own, replay, &ms));
        out[1 + r] = ms;
    }
    return CERE_OK;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int cere_ktest_klab_omma(
    const uint32_t *a, const uint32_t *b, const uint32_t *scale_a, const uint32_t *scale_b, const float *c, float *d,
    int64_t n, int64_t byte_id_a, int64_t thread_id_a, int64_t byte_id_b, int64_t thread_id_b, cudaStream_t stream) {
    if (byte_id_a != 0 || byte_id_b != 0 || (thread_id_a != 0 && thread_id_a != 1) || thread_id_b < 0 ||
        thread_id_b > 3) {
        return CERE_ERR_ARGUMENT;
    }
    constexpr int64_t kWarpsPerBlock = kOmmaThreads / 32;
    if (n <= 0 || n > int64_t{0x7FFFFFFF} * kWarpsPerBlock || !aligned(a, 16) || !aligned(b, 8) ||
        !aligned(scale_a, 4) || !aligned(scale_b, 4) || !aligned(c, 16) || !aligned(d, 16)) {
        return CERE_ERR_ARGUMENT;
    }
    const dim3 grid(static_cast<unsigned>((n + kWarpsPerBlock - 1) / kWarpsPerBlock));
    const auto *av = reinterpret_cast<const uint4 *>(a);
    const auto *bv = reinterpret_cast<const uint2 *>(b);
    const auto *cv = reinterpret_cast<const float4 *>(c);
    auto *dv = reinterpret_cast<float4 *>(d);
    const cudaError_t err = thread_id_a == 0
                                ? launch_omma<0>(thread_id_b, grid, stream, av, bv, scale_a, scale_b, cv, dv, n)
                                : launch_omma<1>(thread_id_b, grid, stream, av, bv, scale_a, scale_b, cv, dv, n);
    return err == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_klab_calib(int64_t row, int64_t size, double *out,
                                                                            cudaStream_t stream) {
    const bool ok = out != nullptr &&
                    (((row == 0 || row == 1 || row == 2) && size >= 1 && size <= 64) ||
                     (row == 3 && (size == 0 || size == 1)) || ((row == 4 || row == 5) && size >= 1 &&
                                                                size <= kMaxNodes));
    if (!ok) return CERE_ERR_ARGUMENT;
    for (int i = 0; i < 7; ++i) out[i] = 0.0;
    return calib(row, size, out, stream);
}
