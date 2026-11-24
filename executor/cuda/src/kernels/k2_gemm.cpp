// K2: cuBLASLt BF16 GEMM wrapper with per-bucket pinned, atomics-free algorithms, and the last-row-only lm_head.
// Contract in k2_gemm.h.
#include "kernels/k2_gemm.h"

#include "kernels/k2_fast.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>

namespace cere::kernels {
namespace {

constexpr uint32_t kHeuristicResults = 8;

uint32_t ceil_log2(uint32_t v) {
    uint32_t b = 0;
    while ((uint32_t{1} << b) < v) {
        ++b;
    }
    return b;
}

uint32_t bucket_upper(uint32_t b) { return uint32_t{1} << b; }
uint32_t bucket_lower(uint32_t b) { return b == 0 ? 1 : (uint32_t{1} << (b - 1)) + 1; }

// Largest power of two <= 16 dividing `bytes` (the alignment every column start of a matrix with that leading
// dimension in bytes keeps, given an aligned base pointer).
uint32_t ld_alignment(uint64_t bytes) {
    uint32_t a = 16;
    while (a > 1 && bytes % a != 0) {
        a /= 2;
    }
    return a;
}

bool aligned(const void *p, uint32_t alignment) {
    return reinterpret_cast<uintptr_t>(p) % alignment == 0;
}

cudaDataType_t out_type(GemmOut out) { return out == GemmOut::kF32 ? CUDA_R_32F : CUDA_R_16BF; }

// Column-major views of the row-major operands: A = w as [k, n] (ld k, op T), B = x as [k, m] (ld k, op N),
// D = y as [n, m] (ld n), so D = A^T B is y^T = w x^T.
struct Layouts {
    cublasLtMatrixLayoutOpaque_t a{}, b{}, d{};
};

cublasStatus_t init_layouts(Layouts *l, uint32_t n, uint32_t k, uint32_t m, GemmOut out) {
    cublasStatus_t s = cublasLtMatrixLayoutInit(&l->a, CUDA_R_16BF, k, n, k);
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatrixLayoutInit(&l->b, CUDA_R_16BF, k, m, k);
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatrixLayoutInit(&l->d, out_type(out), n, m, n);
    }
    return s;
}

// A rows plan's m rows as a strided batch of m single-row GEMMs (BATCH_COUNT m on every operand): the one weight
// (batch stride 0), x row r at r * k and y row r at r * n, i.e. the row-major rows buffers unchanged.
cublasStatus_t init_batched_layouts(Layouts *l, uint32_t n, uint32_t k, uint32_t m, GemmOut out) {
    cublasStatus_t s = init_layouts(l, n, k, 1, out);
    const int32_t count = static_cast<int32_t>(m);
    const int64_t strides[3] = {0, int64_t{k}, int64_t{n}};
    cublasLtMatrixLayoutOpaque_t *layouts[3] = {&l->a, &l->b, &l->d};
    for (int i = 0; i < 3 && s == CUBLAS_STATUS_SUCCESS; ++i) {
        s = cublasLtMatrixLayoutSetAttribute(layouts[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count));
        if (s == CUBLAS_STATUS_SUCCESS) {
            s = cublasLtMatrixLayoutSetAttribute(layouts[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strides[i],
                                                 sizeof(strides[i]));
        }
    }
    return s;
}

// W8A8 (GemmPlan::init_a8): A = w and B = x as E4M3 in init_layouts' views, D BF16.
cublasStatus_t init_a8_layouts(Layouts *l, uint32_t n, uint32_t k, uint32_t m) {
    cublasStatus_t s = cublasLtMatrixLayoutInit(&l->a, CUDA_R_8F_E4M3, k, n, k);
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatrixLayoutInit(&l->b, CUDA_R_8F_E4M3, k, m, k);
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatrixLayoutInit(&l->d, CUDA_R_16BF, n, m, n);
    }
    return s;
}

// The W8A8 descriptor: init()'s (FP32 compute and scale type, A op T, B op N) with VEC32_UE8M0 block scales on A (the
// weight's, `w_scales`) and B (x's, `x_scales`), both in k2m's blocked layout.
cublasStatus_t init_a8_desc(cublasLtMatmulDescOpaque_t *d, const void *w_scales, const void *x_scales) {
    const cublasOperation_t op_t = CUBLAS_OP_T;
    const cublasOperation_t op_n = CUBLAS_OP_N;
    const cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    cublasStatus_t s = cublasLtMatmulDescInit(d, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_scales, sizeof(w_scales));
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &x_scales, sizeof(x_scales));
    }
    return s;
}

// An atomics-free heuristic result (every bucket algorithm, init() and init_a8): SUCCESS, zero workspace, no
// split-K, reduction NONE, and AlgoCheck accepts it with zero workspace at both bucket ends.
bool atomics_free(cublasLtHandle_t handle, cublasLtMatmulDescOpaque_t *desc, Layouts *hi, Layouts *lo,
                  const cublasLtMatmulHeuristicResult_t &r) {
    if (r.state != CUBLAS_STATUS_SUCCESS || r.workspaceSize != 0) {
        return false;
    }
    int32_t splitk = 0;
    uint32_t scheme = 0;
    size_t written = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk),
                                             &written) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &scheme, sizeof(scheme),
                                             &written) != CUBLAS_STATUS_SUCCESS ||
        splitk > 1 || scheme != CUBLASLT_REDUCTION_SCHEME_NONE) {
        return false;
    }
    cublasLtMatmulHeuristicResult_t check_hi{};
    cublasLtMatmulHeuristicResult_t check_lo{};
    return cublasLtMatmulAlgoCheck(handle, desc, &hi->a, &hi->b, &hi->d, &hi->d, &r.algo, &check_hi) ==
               CUBLAS_STATUS_SUCCESS &&
           cublasLtMatmulAlgoCheck(handle, desc, &lo->a, &lo->b, &lo->d, &lo->d, &r.algo, &check_lo) ==
               CUBLAS_STATUS_SUCCESS &&
           check_hi.workspaceSize == 0 && check_lo.workspaceSize == 0;
}

// The first bucket whose every m exceeds k2s::kMaxM (bucket_lower 17): the W8A8 path's first bucket.
constexpr uint32_t kA8FirstBucket = 5;
static_assert(k2s::kMaxM == 16, "kA8FirstBucket is the bucket after m = k2s::kMaxM");
constexpr size_t kA8Align = 256;  // each W8A8 staging operand starts 256-byte aligned
// The scale pointers of init_a8's queries. cuBLASLt (13.1.1) answers a VEC32_UE8M0 heuristic query whose descriptor has
// no scale pointers CUBLAS_STATUS_INVALID_VALUE; the host-side queries (heuristic, AlgoCheck) read no device memory,
// and with this aligned placeholder they pick, for every K2 MXFP8 shape and bucket, the algorithm the real pointers
// get, which runs with the real pointers (klab a8diag, k2-prefill design note F7). matmul_a8 sets the real ones.
const void *const kA8QueryScales = reinterpret_cast<const void *>(uintptr_t{kA8Align});

size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

cere_status cuda_status(cudaError_t e) {
    return e == cudaSuccess ? CERE_OK : e == cudaErrorInvalidValue ? CERE_ERR_ARGUMENT : CERE_ERR_INTERNAL;
}

// Bytes per partial element of a split-K reduction scheme (0: no partials).
uint64_t partial_bytes(cublasLtReductionScheme_t scheme, GemmOut out) {
    switch (scheme) {
    case CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE:
        return 4;
    case CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE:
        return out == GemmOut::kF32 ? 4 : 2;
    default:
        return 0;
    }
}

// Workspace of `c` at m rows of n columns (GemmConfig::partials).
uint64_t config_workspace(const GemmConfig &c, uint32_t m, uint32_t n, GemmOut out) {
    return uint64_t{c.partials} * m * n * partial_bytes(c.reduction, out);
}

bool config_ok(const GemmConfig &c) {
    if (c.reduction == CUBLASLT_REDUCTION_SCHEME_NONE) {
        return c.splitk == 1 && c.partials == 0;
    }
    return (c.reduction == CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE ||
            c.reduction == CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE) &&
           c.splitk > 1 && c.partials > 0;
}

bool pins_ok(const GemmPins &pins) {
    if (pins.ranges == nullptr || pins.count == 0 || pins.count > GemmPlan::kMaxPins) {
        return false;
    }
    for (uint32_t i = 0; i < pins.count; ++i) {
        const GemmPinRange &r = pins.ranges[i];
        if (r.m_lo < GemmPins::kPinMinM || r.m_hi < r.m_lo || (i > 0 && r.m_lo <= pins.ranges[i - 1].m_hi) ||
            !config_ok(r.config)) {
            return false;
        }
    }
    return true;
}

// cublasLtMatmulAlgoInit(c.algo_id) with c's tile, stages, split-K, reduction scheme and custom option (every other
// configuration attribute stays 0).
cublasStatus_t init_config(cublasLtHandle_t handle, const GemmConfig &c, GemmOut out, cublasLtMatmulAlgo_t *algo) {
    cublasStatus_t s = cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                                              out_type(out), out_type(out), c.algo_id, algo);
    const uint32_t tile = c.tile;
    const uint32_t stages = c.stages;
    const int32_t splitk = c.splitk;
    const uint32_t scheme = c.reduction;
    const uint32_t custom = c.custom_option;
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile));
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages));
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk));
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &scheme,
                                                 sizeof(scheme));
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof(custom));
    }
    return s;
}

constexpr cublasLtReductionScheme_t kOutputType = CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE;
constexpr cublasLtReductionScheme_t kNoReduction = CUBLASLT_REDUCTION_SCHEME_NONE;

// kSharedGateUpPins (k2_gemm.h): torch's configuration per m range; the last field is GemmConfig::partials.
// Record: an internal benchmark report (`native_dev.py cublaslt --plan moe_shared_gate_up`:
// torch's pick at every m in [17, 4,096]; tests/test_k2_gemm_pins.py checks this table and kPinsCublasLt against it).
const GemmPinRange kSharedGateUpRanges[] = {
    {17, 28, {21, CUBLASLT_MATMUL_TILE_128x64, CUBLASLT_MATMUL_STAGES_64x3, 18, kOutputType, 14}},
    {29, 32, {21, CUBLASLT_MATMUL_TILE_128x64, CUBLASLT_MATMUL_STAGES_64x3, 17, kOutputType, 14}},
    {33, 52, {21, CUBLASLT_MATMUL_TILE_128x64, CUBLASLT_MATMUL_STAGES_64x4, 14, kOutputType, 14}},
    {53, 64, {23, CUBLASLT_MATMUL_TILE_128x64, CUBLASLT_MATMUL_STAGES_64x3, 17, kOutputType, 14}},
    {65, 96, {21, CUBLASLT_MATMUL_TILE_64x128, CUBLASLT_MATMUL_STAGES_32x6, 9, kOutputType, 8}},
    {97, 100, {21, CUBLASLT_MATMUL_TILE_128x64, CUBLASLT_MATMUL_STAGES_64x4, 8, kOutputType, 8}},
    {101, 128, {21, CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_32x4, 12, kOutputType, 10}},
    {129, 160, {21, CUBLASLT_MATMUL_TILE_128x64, CUBLASLT_MATMUL_STAGES_64x3, 6, kOutputType, 6}},
    {161, 192, {21, CUBLASLT_MATMUL_TILE_128x64, CUBLASLT_MATMUL_STAGES_32x6, 5, kOutputType, 5}},
    {193, 256, {21, CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_32x6, 8, kOutputType, 8}},
    {257, 384, {21, CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_32x4, 5, kOutputType, 5}},
    {385, 448, {21, CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_32x4, 4, kOutputType, 4}},
    {449, 472, {21, CUBLASLT_MATMUL_TILE_64x64, CUBLASLT_MATMUL_STAGES_64x6, 1, kNoReduction, 0}},
    {473, 544, {21, CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_32x4, 3, kOutputType, 3}},
    {545, 576, {21, CUBLASLT_MATMUL_TILE_64x64, CUBLASLT_MATMUL_STAGES_64x6, 1, kNoReduction, 0}},
    {577, 640, {21, CUBLASLT_MATMUL_TILE_128x64, CUBLASLT_MATMUL_STAGES_32x6, 1, kNoReduction, 0}},
    {641, 1120, {21, CUBLASLT_MATMUL_TILE_128x64, CUBLASLT_MATMUL_STAGES_64x4, 1, kNoReduction, 0}},
    {1121, 1152, {21, CUBLASLT_MATMUL_TILE_64x128, CUBLASLT_MATMUL_STAGES_64x4, 1, kNoReduction, 0}},
    // 1,153-1,536: torch's pick is INPLACE (workspace counters); the bucket algorithm stays.
    {1537, 1920, {21, CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_32x4, 1, kNoReduction, 0}},
    {1921, 2176, {21, CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_64x3, 1, kNoReduction, 0}},
    {2177, 2304, {21, CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_32x4, 1, kNoReduction, 0}},
    {2305, 2368, {21, CUBLASLT_MATMUL_TILE_256x64, CUBLASLT_MATMUL_STAGES_32x4, 1, kNoReduction, 0}},
    // 2,369-3,200: torch's pick is INPLACE; the bucket algorithm stays.
    {3201, 3896, {21, CUBLASLT_MATMUL_TILE_128x256, CUBLASLT_MATMUL_STAGES_32x3, 1, kNoReduction, 0}},
    {3897, 3968, {21, CUBLASLT_MATMUL_TILE_256x128, CUBLASLT_MATMUL_STAGES_32x3, 1, kNoReduction, 0}},
    {3969, 4096, {21, CUBLASLT_MATMUL_TILE_128x256, CUBLASLT_MATMUL_STAGES_32x3, 1, kNoReduction, 0}},
};

// kAttnQkvPins (k2_gemm.h): torch's configuration per m range; the last field is GemmConfig::partials.
// Record: an internal benchmark report (`native_dev.py cublaslt --plan attn_qkv`: torch's pick
// at every m in [17, 4,096]; tests/test_k2_gemm_pins.py checks this table and kPinsCublasLt against it).
const GemmPinRange kAttnQkvRanges[] = {
    {17, 32, {21, CUBLASLT_MATMUL_TILE_128x64, CUBLASLT_MATMUL_STAGES_32x6, 5, kOutputType, 5}},
    {33, 64, {21, CUBLASLT_MATMUL_TILE_128x64, CUBLASLT_MATMUL_STAGES_64x4, 1, kNoReduction, 0}},
    // 65-96: torch's pick is INPLACE (workspace counters); the bucket algorithm stays.
    {97, 128, {21, CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_32x4, 1, kNoReduction, 0}},
    {129, 192, {21, CUBLASLT_MATMUL_TILE_256x64, CUBLASLT_MATMUL_STAGES_32x4, 1, kNoReduction, 0}},
    // 193-256: INPLACE; the bucket algorithm stays.
    {257, 384, {21, CUBLASLT_MATMUL_TILE_256x128, CUBLASLT_MATMUL_STAGES_32x3, 1, kNoReduction, 0}},
    // 385-512: INPLACE; the bucket algorithm stays.
    {513, 640, {21, CUBLASLT_MATMUL_TILE_128x128, CUBLASLT_MATMUL_STAGES_32x4, 1, kNoReduction, 0}},
    // 641-768: INPLACE; the bucket algorithm stays.
    {769, 896, {21, CUBLASLT_MATMUL_TILE_256x128, CUBLASLT_MATMUL_STAGES_32x3, 1, kNoReduction, 0}},
    // 897-1,024: INPLACE; the bucket algorithm stays.
    {1025, 1152, {21, CUBLASLT_MATMUL_TILE_128x256, CUBLASLT_MATMUL_STAGES_32x3, 1, kNoReduction, 0}},
    {1153, 1280, {21, CUBLASLT_MATMUL_TILE_256x128, CUBLASLT_MATMUL_STAGES_32x3, 1, kNoReduction, 0}},
    // 1,281-1,536: INPLACE; the bucket algorithm stays.
    {1537, 1792, {21, CUBLASLT_MATMUL_TILE_128x256, CUBLASLT_MATMUL_STAGES_32x3, 1, kNoReduction, 0}},
    // 1,793-2,048: INPLACE; the bucket algorithm stays.
    {2049, 2176, {21, CUBLASLT_MATMUL_TILE_256x128, CUBLASLT_MATMUL_STAGES_32x3, 1, kNoReduction, 0}},
    {2177, 2304, {21, CUBLASLT_MATMUL_TILE_128x256, CUBLASLT_MATMUL_STAGES_32x3, 1, kNoReduction, 0}},
    {2305, 2944, {21, CUBLASLT_MATMUL_TILE_256x128, CUBLASLT_MATMUL_STAGES_32x3, 1, kNoReduction, 0}},
    {2945, 3072, {21, CUBLASLT_MATMUL_TILE_128x256, CUBLASLT_MATMUL_STAGES_32x3, 1, kNoReduction, 0}},
    {3073, 4096, {21, CUBLASLT_MATMUL_TILE_256x128, CUBLASLT_MATMUL_STAGES_32x3, 1, kNoReduction, 0}},
};

// cublasLtGetVersion() of the image's cuBLASLt 13.1.1 (cublas_api.h: CUBLAS_VERSION = major * 10,000 + minor * 100 +
// patch; both records' torch runtime value).
constexpr size_t kPinsCublasLt = 130101;

// Bucket re-pins (k2_gemm.h kGemmPinsTables from kFirstBucketRepin; r3-prefill design note): per shape the fastest
// configuration of the R3 lab (an internal note, GPU0 325 W, cuBLASLt 13.1.1) among the
// zero-workspace, split-K-free configurations bit-equal to the bucket algorithm at m 1,217, then swept over the bucket
// (m 1,025 + 32 j and 2,048; 30 reps, rotated weights, 2 seeds, bits checked at every m). A range is a maximal run of
// sweep m where the pin is bit-equal and < 0.99 x the bucket algorithm's time, holding a point <= 0.97 x; outside the
// ranges the plan runs the bucket algorithm. gdn_ba (n 48) is not re-pinned (< 1 µs a call). Record:
// an internal benchmark report (every sweep point: m, µs of the pin and of the bucket algorithm).
constexpr GemmPinRange kRepinGdnQkvRanges[] = {
    {1025, 1249, {67, CUBLASLT_MATMUL_TILE_112x320, CUBLASLT_MATMUL_STAGES_32xAUTO, 1, kNoReduction, 0, 91}},
    {1793, 1889, {67, CUBLASLT_MATMUL_TILE_112x320, CUBLASLT_MATMUL_STAGES_32xAUTO, 1, kNoReduction, 0, 91}},
};
constexpr GemmPinRange kRepinGdnZRanges[] = {
    {1025, 1217, {67, CUBLASLT_MATMUL_TILE_192x112, CUBLASLT_MATMUL_STAGES_32xAUTO, 1, kNoReduction, 0, 106}},
    {1793, 2048, {67, CUBLASLT_MATMUL_TILE_192x112, CUBLASLT_MATMUL_STAGES_32xAUTO, 1, kNoReduction, 0, 106}},
};
constexpr GemmPinRange kRepinOutRanges[] = {
    {1025, 1217, {67, CUBLASLT_MATMUL_TILE_128x136, CUBLASLT_MATMUL_STAGES_64xAUTO, 1, kNoReduction, 0, 104}},
};
constexpr GemmPinRange kRepinRouterRanges[] = {
    {1025, 1441, {21, CUBLASLT_MATMUL_TILE_64x64, CUBLASLT_MATMUL_STAGES_32x10, 1, kNoReduction, 0, 0}},
};
constexpr GemmPinRange kRepinSharedDownRanges[] = {
    {1025, 1281, {67, CUBLASLT_MATMUL_TILE_128x144, CUBLASLT_MATMUL_STAGES_64xAUTO, 1, kNoReduction, 0, 68}},
};
constexpr GemmPinRange kRepinHcDownRanges[] = {
    {1025, 1825, {67, CUBLASLT_MATMUL_TILE_64x56, CUBLASLT_MATMUL_STAGES_128xAUTO, 1, kNoReduction, 0, 142}},
};
constexpr GemmPinRange kRepinHcUpRanges[] = {
    {1153, 1281, {67, CUBLASLT_MATMUL_TILE_256x144, CUBLASLT_MATMUL_STAGES_32xAUTO, 1, kNoReduction, 0, 144}},
    {1793, 1985, {67, CUBLASLT_MATMUL_TILE_256x144, CUBLASLT_MATMUL_STAGES_32xAUTO, 1, kNoReduction, 0, 144}},
};

template <size_t N>
constexpr GemmPins repin(uint32_t n, uint32_t k, GemmOut out, const GemmPinRange (&ranges)[N]) {
    return GemmPins{n, k, out, ranges, uint32_t(N), kPinsCublasLt};
}

constexpr GemmPins kRepinGdnQkv = repin(10240, 2560, GemmOut::kBF16, kRepinGdnQkvRanges);  // gdn_qkv, ple_key
constexpr GemmPins kRepinGdnZ = repin(6144, 2560, GemmOut::kBF16, kRepinGdnZRanges);
constexpr GemmPins kRepinOut = repin(2560, 6144, GemmOut::kBF16, kRepinOutRanges);  // gdn_out, attn_o
constexpr GemmPins kRepinRouter = repin(512, 2560, GemmOut::kBF16, kRepinRouterRanges);
constexpr GemmPins kRepinSharedDown = repin(2560, 640, GemmOut::kBF16, kRepinSharedDownRanges);
constexpr GemmPins kRepinHcDown = repin(320, 10240, GemmOut::kF32, kRepinHcDownRanges);
constexpr GemmPins kRepinHcUp = repin(10240, 320, GemmOut::kF32, kRepinHcUpRanges);

// The algoId and customOption of a cuBLASLt algorithm.
cublasStatus_t algo_ids(const cublasLtMatmulAlgo_t &algo, int *algo_id, uint32_t *custom_option) {
    size_t written = 0;
    cublasStatus_t s =
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, algo_id, sizeof(*algo_id), &written);
    if (s == CUBLAS_STATUS_SUCCESS) {
        s = cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, custom_option,
                                                 sizeof(*custom_option), &written);
    }
    return s;
}

// Rows plans (m23 design note P11R-21): for the m AlgoCheck refuses on the rows layout and both one-launch paths
// serve, the first m at which the gemvx-order rows kernel (k2v::gemv, the weights read once per launch) is faster than
// the strided batch of the m = 1 algorithm (the weights read once per row, a cost linear in m), per (n, k, out); below
// it, and on every (n, k, out) not listed, the strided batch runs. Both paths give the same bits (test_rows_paths), so
// the choice is speed only. Measured with cold weights by test_kernels_k2_gemm.py test_rows_paths_bench (325 W, the
// k2v kernel tuned per shape class in k2_gemv.cu, an internal benchmark report).
// gdn_b / gdn_a (n 48) tie within 0.3 µs at every m and keep the strided batch.
struct RowsGemvFrom {
    uint32_t n;
    uint32_t k;
    GemmOut out;
    uint32_t m;
};
constexpr RowsGemvFrom kRowsGemvFrom[] = {
    {2560, 6144, GemmOut::kBF16, 2},    // attention o_proj, GDN out_proj (target and MTP)
    {6144, 2560, GemmOut::kBF16, 2},    // GDN in_proj_z
    {2560, 640, GemmOut::kBF16, 2},     // shared expert down_proj (target and MTP)
    {2560, 2560, GemmOut::kBF16, 2},    // MTP fc halves
    {248320, 2560, GemmOut::kBF16, 2},  // lm_head
    {65536, 2560, GemmOut::kBF16, 2},   // MTP hot lm_head (the -v10 draft token map)
    {512, 2560, GemmOut::kBF16, 4},     // router (target)
    {512, 2560, GemmOut::kF32, 4},      // router (MTP)
    {640, 2560, GemmOut::kBF16, 4},     // shared expert gate_proj / up_proj (target and MTP)
    {320, 10240, GemmOut::kF32, 6},     // hc input_mix down
};

uint32_t rows_gemv_from(uint32_t n, uint32_t k, GemmOut out) {
    for (const RowsGemvFrom &r : kRowsGemvFrom) {
        if (r.n == n && r.k == k && r.out == out) {
            return r.m;
        }
    }
    return GemmPlan::kMaxRows + 1;
}

}  // namespace

const GemmPins kSharedGateUpPins{640,
                                 2560,
                                 GemmOut::kBF16,
                                 kSharedGateUpRanges,
                                 uint32_t(sizeof(kSharedGateUpRanges) / sizeof(kSharedGateUpRanges[0])),
                                 kPinsCublasLt};

const GemmPins kAttnQkvPins{13312,
                            2560,
                            GemmOut::kBF16,
                            kAttnQkvRanges,
                            uint32_t(sizeof(kAttnQkvRanges) / sizeof(kAttnQkvRanges[0])),
                            kPinsCublasLt};

const GemmPins *const kGemmPinsTables[] = {&kSharedGateUpPins, &kAttnQkvPins,     &kRepinGdnQkv, &kRepinGdnZ,
                                           &kRepinOut,         &kRepinRouter,     &kRepinSharedDown,
                                           &kRepinHcDown,      &kRepinHcUp};
const uint32_t kGemmPinsTableCount = uint32_t(sizeof(kGemmPinsTables) / sizeof(kGemmPinsTables[0]));

const GemmPins *gemm_pins_for(uint32_t n, uint32_t k, GemmOut out) {
    for (uint32_t i = 0; i < kGemmPinsTableCount; ++i) {
        if (kGemmPinsTables[i]->applies(n, k, out)) {
            return kGemmPinsTables[i];
        }
    }
    return nullptr;
}

size_t gemm_pins_workspace_bytes(const GemmPins &pins, uint32_t max_m) {
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < pins.count && pins.ranges[i].m_lo <= max_m; ++i) {
        const GemmPinRange &r = pins.ranges[i];
        const uint64_t b = config_workspace(r.config, r.m_hi < max_m ? r.m_hi : max_m, pins.n, pins.out);
        bytes = b > bytes ? b : bytes;
    }
    return size_t(bytes);
}

uint32_t gemm_m1_custom_option(size_t cublaslt_version, int algo_id, uint32_t custom_option, uint32_t k) {
    constexpr int kGemvxAlgo = 13;
    constexpr uint32_t kFrom = 37;
    constexpr uint32_t kTo = 75;
    if (cublaslt_version != kPinsCublasLt || algo_id != kGemvxAlgo || custom_option != kFrom) {
        return custom_option;
    }
    const k2v::Order from = k2v::order_of(cublaslt_version, algo_id, kFrom, k);
    const k2v::Order to = k2v::order_of(cublaslt_version, algo_id, kTo, k);
    const bool same = from.lanes != 0 && from.lanes == to.lanes && from.chunk == to.chunk;
    return same ? kTo : custom_option;
}

cere_status gemm_handle_create(cublasLtHandle_t *out) {
    *out = nullptr;
    cublasLtHandle_t handle = nullptr;
    if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        return CERE_ERR_INTERNAL;
    }
    *out = handle;
    return CERE_OK;
}

void gemm_handle_destroy(cublasLtHandle_t handle) {
    if (handle != nullptr) {
        cublasLtDestroy(handle);
    }
}

cere_status GemmPlan::init(cublasLtHandle_t handle, uint32_t n, uint32_t k, uint32_t max_m, GemmOut out,
                           const GemmPins *pins, void *pin_workspace, size_t pin_workspace_bytes, bool m1_swap) {
    buckets_ = 0;
    rows_fused_ = 0;
    rows_allowed_ = 0;
    rows_gemv_ = 0;
    rows_batched_ = 0;
    small_m_ = false;  // the k2_gemm lever's default, cuBLASLt for every m (model::apply_levers switches it)
    a8_ready_ = false;
    a8_ = false;
    early_ = false;
    pin_count_ = 0;
    pin_range_ = nullptr;
    workspace_ = nullptr;
    workspace_bytes_ = 0;
    if (handle == nullptr || n == 0 || k == 0 || max_m == 0 || ceil_log2(max_m) >= kMaxBuckets ||
        (out != GemmOut::kBF16 && out != GemmOut::kF32)) {
        return CERE_ERR_ARGUMENT;
    }
    if (pins != nullptr) {
        if (!pins->applies(n, k, out) || !pins_ok(*pins)) {
            return CERE_ERR_ARGUMENT;
        }
        const size_t need = gemm_pins_workspace_bytes(*pins, max_m);
        if (need > pin_workspace_bytes || (need > 0 && (pin_workspace == nullptr || !aligned(pin_workspace, 256)))) {
            return CERE_ERR_ARGUMENT;
        }
        if (cublasLtGetVersion() != pins->cublaslt_version) {
            return CERE_ERR_UNSUPPORTED;
        }
    }
    const uint64_t out_bytes = out == GemmOut::kF32 ? 4 : 2;
    handle_ = handle;
    n_ = n;
    k_ = k;
    max_m_ = max_m;
    out_ = out;
    align_w_ = ld_alignment(uint64_t{k} * 2);
    align_x_ = align_w_;
    align_y_ = ld_alignment(uint64_t{n} * out_bytes);

    if (cublasLtMatmulDescInit(&desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        return CERE_ERR_INTERNAL;
    }
    const cublasOperation_t op_t = CUBLAS_OP_T;
    const cublasOperation_t op_n = CUBLAS_OP_N;
    if (cublasLtMatmulDescSetAttribute(&desc_, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(&desc_, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)) !=
            CUBLAS_STATUS_SUCCESS) {
        return CERE_ERR_INTERNAL;
    }

    cublasLtMatmulPreferenceOpaque_t pref{};
    const uint64_t workspace = 0;
    const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE;
    if (cublasLtMatmulPreferenceInit(&pref) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(&pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
                                             sizeof(workspace)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(&pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
                                             sizeof(reduction)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(&pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &align_w_,
                                             sizeof(align_w_)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(&pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &align_x_,
                                             sizeof(align_x_)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(&pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &align_y_,
                                             sizeof(align_y_)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(&pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &align_y_,
                                             sizeof(align_y_)) != CUBLAS_STATUS_SUCCESS) {
        return CERE_ERR_INTERNAL;
    }

    const uint32_t buckets = ceil_log2(max_m) + 1;
    for (uint32_t b = 0; b < buckets; ++b) {
        Layouts hi;
        Layouts lo;
        if (init_layouts(&hi, n, k, bucket_upper(b), out) != CUBLAS_STATUS_SUCCESS ||
            init_layouts(&lo, n, k, bucket_lower(b), out) != CUBLAS_STATUS_SUCCESS) {
            return CERE_ERR_INTERNAL;
        }
        cublasLtMatmulHeuristicResult_t results[kHeuristicResults]{};
        int returned = 0;
        const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(handle, &desc_, &hi.a, &hi.b, &hi.d, &hi.d, &pref,
                                                                 kHeuristicResults, results, &returned);
        if (hs != CUBLAS_STATUS_SUCCESS && hs != CUBLAS_STATUS_NOT_SUPPORTED) {
            return CERE_ERR_INTERNAL;
        }
        auto usable = [&](const cublasLtMatmulHeuristicResult_t &r) {
            return atomics_free(handle, &desc_, &hi, &lo, r);
        };
        int pick = -1;
        for (int i = 0; hs == CUBLAS_STATUS_SUCCESS && i < returned && pick < 0; ++i) {
            if (usable(results[i])) {
                pick = i;
            }
        }
        if (pick < 0) {
            return CERE_ERR_UNSUPPORTED;
        }
        algo_[b] = results[pick].algo;
        if (b == 0 && m1_swap) {
            // The m = 1 customOption swap (gemm_m1_custom_option): the heuristic's own result in the swapped
            // variant, or no plan.
            int algo_id = -1;
            uint32_t option = 0;
            if (algo_ids(results[pick].algo, &algo_id, &option) != CUBLAS_STATUS_SUCCESS) {
                return CERE_ERR_INTERNAL;
            }
            const uint32_t want = gemm_m1_custom_option(cublasLtGetVersion(), algo_id, option, k);
            bool swapped = want == option;
            for (int i = 0; i < returned && !swapped; ++i) {
                int id_i = -1;
                uint32_t option_i = 0;
                if (algo_ids(results[i].algo, &id_i, &option_i) == CUBLAS_STATUS_SUCCESS && id_i == algo_id &&
                    option_i == want && usable(results[i])) {
                    algo_[0] = results[i].algo;
                    swapped = true;
                }
            }
            if (!swapped) {
                return CERE_ERR_UNSUPPORTED;
            }
        }
    }

    uint32_t pin_count = 0;
    for (; pins != nullptr && pin_count < pins->count && pins->ranges[pin_count].m_lo <= max_m; ++pin_count) {
        const GemmPinRange &r = pins->ranges[pin_count];
        const uint32_t m_hi = r.m_hi < max_m ? r.m_hi : max_m;
        Layouts hi;
        Layouts lo;
        cublasLtMatmulAlgo_t algo{};
        if (init_layouts(&hi, n, k, m_hi, out) != CUBLAS_STATUS_SUCCESS ||
            init_layouts(&lo, n, k, r.m_lo, out) != CUBLAS_STATUS_SUCCESS) {
            return CERE_ERR_INTERNAL;
        }
        if (init_config(handle, r.config, out, &algo) != CUBLAS_STATUS_SUCCESS) {
            return CERE_ERR_UNSUPPORTED;
        }
        cublasLtMatmulHeuristicResult_t check_hi{};
        cublasLtMatmulHeuristicResult_t check_lo{};
        if (cublasLtMatmulAlgoCheck(handle, &desc_, &hi.a, &hi.b, &hi.d, &hi.d, &algo, &check_hi) !=
                CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulAlgoCheck(handle, &desc_, &lo.a, &lo.b, &lo.d, &lo.d, &algo, &check_lo) !=
                CUBLAS_STATUS_SUCCESS ||
            check_hi.workspaceSize > config_workspace(r.config, m_hi, n, out) ||
            check_lo.workspaceSize > config_workspace(r.config, r.m_lo, n, out)) {
            return CERE_ERR_UNSUPPORTED;
        }
        pin_algo_[pin_count] = algo;
    }
    if (pin_count > 0) {
        pin_range_ = pins->ranges;
        pin_count_ = pin_count;
        workspace_ = pin_workspace;
        workspace_bytes_ = pin_workspace_bytes;
    }
    buckets_ = buckets;
    return CERE_OK;
}

cere_status GemmPlan::m1_algo(int *algo_id, uint32_t *custom_option) const {
    if (buckets_ == 0 || algo_id == nullptr || custom_option == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    return algo_ids(algo_[0], algo_id, custom_option) == CUBLAS_STATUS_SUCCESS ? CERE_OK : CERE_ERR_INTERNAL;
}

cere_status GemmPlan::init_a8() {
    a8_ready_ = false;
    a8_ = false;
    if (buckets_ == 0 || rows_fused_ != 0 || out_ != GemmOut::kBF16) {
        return CERE_ERR_ARGUMENT;
    }
    if (k_ % k2m::kGroupK != 0 || max_m_ <= k2s::kMaxM) {
        return CERE_ERR_UNSUPPORTED;
    }
    align_q_ = ld_alignment(k_);
    cublasLtMatmulDescOpaque_t desc{};
    cublasLtMatmulPreferenceOpaque_t pref{};
    const uint64_t workspace = 0;
    const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_NONE;
    // Any refusal here (an FP8 descriptor or layout this cuBLASLt does not take) leaves the plan off the path.
    if (init_a8_desc(&desc, kA8QueryScales, kA8QueryScales) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceInit(&pref) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(&pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
                                             sizeof(workspace)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(&pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
                                             sizeof(reduction)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(&pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &align_q_,
                                             sizeof(align_q_)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(&pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &align_q_,
                                             sizeof(align_q_)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(&pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &align_y_,
                                             sizeof(align_y_)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceSetAttribute(&pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &align_y_,
                                             sizeof(align_y_)) != CUBLAS_STATUS_SUCCESS) {
        return CERE_ERR_UNSUPPORTED;
    }
    for (uint32_t b = kA8FirstBucket; b < buckets_; ++b) {
        Layouts hi;
        Layouts lo;
        if (init_a8_layouts(&hi, n_, k_, bucket_upper(b)) != CUBLAS_STATUS_SUCCESS ||
            init_a8_layouts(&lo, n_, k_, bucket_lower(b)) != CUBLAS_STATUS_SUCCESS) {
            return CERE_ERR_UNSUPPORTED;
        }
        cublasLtMatmulHeuristicResult_t results[kHeuristicResults]{};
        int returned = 0;
        if (cublasLtMatmulAlgoGetHeuristic(handle_, &desc, &hi.a, &hi.b, &hi.d, &hi.d, &pref, kHeuristicResults,
                                           results, &returned) != CUBLAS_STATUS_SUCCESS) {
            return CERE_ERR_UNSUPPORTED;
        }
        int pick = -1;
        for (int i = 0; i < returned && pick < 0; ++i) {
            if (atomics_free(handle_, &desc, &hi, &lo, results[i])) {
                pick = i;
            }
        }
        if (pick < 0) {
            return CERE_ERR_UNSUPPORTED;
        }
        algo8_[b] = results[pick].algo;
    }
    a8_ready_ = true;
    return CERE_OK;
}

size_t GemmPlan::a8_staging_bytes(uint32_t m, uint32_t n, uint32_t k) {
    const uint32_t cols = k / k2m::kGroupK;
    return align_up(size_t{m} * k, kA8Align) + align_up(k2m::blocked_scale_bytes(m, cols), kA8Align) +
           k2m::blocked_scale_bytes(n, cols);
}

cere_status GemmPlan::matmul_a8(const k2m::Entry &e, const uint8_t *xq, const uint8_t *xs, uint8_t *ws, uint32_t m,
                                void *y, cudaStream_t stream, uint32_t sel) const {
    if (!aligned(e.w, align_q_)) {
        return CERE_ERR_ARGUMENT;
    }
    const cere_status s = cuda_status(k2m::repack_scales(e.scales, n_, k_, ws, stream));
    if (s != CERE_OK) {
        return s;
    }
    cublasLtMatmulDescOpaque_t desc{};
    Layouts l;
    if (init_a8_desc(&desc, ws, xs) != CUBLAS_STATUS_SUCCESS ||
        init_a8_layouts(&l, n_, k_, m) != CUBLAS_STATUS_SUCCESS) {
        return CERE_ERR_INTERNAL;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasLtMatmul(handle_, &desc, &alpha, e.w, &l.a, xq, &l.b, &beta, y, &l.d, y, &l.d,
                          &algo8_[ceil_log2(sel)], nullptr, 0, stream) == CUBLAS_STATUS_SUCCESS
               ? CERE_OK
               : CERE_ERR_INTERNAL;
}

const k2m::Entry *GemmPlan::a8_entry(const __nv_bfloat16 *x, const __nv_bfloat16 *w, uint32_t m, const void *y,
                                     uint32_t cls) const {
    const uint32_t sel = cls != 0 ? cls : m;
    if (!a8_ || rows_fused_ != 0 || buckets_ == 0 || m == 0 || m > max_m_ || x == nullptr || w == nullptr ||
        y == nullptr || !aligned(x, align_x_) || !aligned(w, align_w_) || !aligned(y, align_y_) ||
        (cls != 0 && cls < m) || cls > max_m_ || sel <= k2s::kMaxM || mxfp8_ == nullptr ||
        (fp8_ != nullptr && fp8_->find(w) != nullptr)) {
        return nullptr;
    }
    const k2m::Entry *e = mxfp8_->find(w);
    return e != nullptr && e->n == n_ && e->k == k_ && e->scales != nullptr && aligned(e->w, align_q_) ? e : nullptr;
}

cere_status GemmPlan::init_rows(cublasLtHandle_t handle, const GemmPlan &instantiated, uint32_t rows,
                                uint32_t fused_allowed, uint32_t paths) {
    buckets_ = 0;
    rows_fused_ = 0;
    rows_allowed_ = 0;
    small_m_ = false;
    a8_ready_ = false;  // W8A8 is init() plans' alone
    a8_ = false;
    early_ = false;
    rows_gemv_ = 0;
    rows_batched_ = 0;
    gemv_order_ = {};
    pin_count_ = 0;
    pin_range_ = nullptr;
    workspace_ = nullptr;
    workspace_bytes_ = 0;
    if (handle == nullptr || instantiated.buckets_ == 0 || rows == 0 || rows > kMaxRows ||
        (paths & ~kRowsAllPaths) != 0) {
        return CERE_ERR_ARGUMENT;
    }
    handle_ = handle;
    n_ = instantiated.n_;
    k_ = instantiated.k_;
    max_m_ = rows;
    out_ = instantiated.out_;
    align_x_ = instantiated.align_x_;
    align_w_ = instantiated.align_w_;
    align_y_ = instantiated.align_y_;
    desc_ = instantiated.desc_;
    algo_[0] = instantiated.algo_[0];
    small_m_ = instantiated.small_m_;
    early_ = instantiated.early_;
    // The m AlgoCheck refuses on the rows layout run as one launch instead of m single-row calls: one k2v::gemv launch
    // where the m = 1 algorithm is a gemvx variant whose order is identified for this cuBLASLt (k2v::order_of), or one
    // strided-batched call of the m = 1 algorithm where AlgoCheck accepts it on that layout; where both serve, the
    // measured faster one (rows_gemv_from).
    int algo_id = -1;
    uint32_t custom_option = 0;
    if (instantiated.m1_algo(&algo_id, &custom_option) != CERE_OK) {
        return CERE_ERR_INTERNAL;
    }
    const k2v::Order order = k2v::order_of(cublasLtGetVersion(), algo_id, custom_option, k_);
    const bool gemv = (paths & kRowsGemv) != 0 && k2v::supported(order, k_) && rows <= k2v::kMaxM;
    const uint32_t gemv_from = rows_gemv_from(n_, k_, out_);
    // Both masks are built whatever small_m_ is, so the k2_gemm lever can switch the plan later (hot swap,
    // use_small_m) without re-initialising it: rows_allowed_ (fused_allowed's bits) is the mask under the small-M
    // kernel (rows <= kMaxRows = k2s::kMaxM: every allowed m runs one batch-invariant small-M launch), rows_fused_
    // (allowed and AlgoCheck-accepted) the mask under cuBLASLt's m = 1 algorithm.
    uint32_t allowed = 1;  // m = 1 is the instantiated plan's own call
    uint32_t fused = 1;
    for (uint32_t m = 2; m <= rows; ++m) {
        if ((fused_allowed & (uint32_t{1} << (m - 1))) == 0) {
            continue;
        }
        allowed |= uint32_t{1} << (m - 1);
        Layouts l;
        if (init_layouts(&l, n_, k_, m, out_) != CUBLAS_STATUS_SUCCESS) {
            return CERE_ERR_INTERNAL;
        }
        cublasLtMatmulHeuristicResult_t check{};
        if (cublasLtMatmulAlgoCheck(handle, &desc_, &l.a, &l.b, &l.d, &l.d, &algo_[0], &check) ==
                CUBLAS_STATUS_SUCCESS &&
            check.workspaceSize == 0) {
            fused |= uint32_t{1} << (m - 1);
        } else {
            bool batched = false;
            if ((paths & kRowsBatched) != 0 && !(gemv && m >= gemv_from)) {
                Layouts b;
                if (init_batched_layouts(&b, n_, k_, m, out_) != CUBLAS_STATUS_SUCCESS) {
                    return CERE_ERR_INTERNAL;
                }
                check = {};
                batched = cublasLtMatmulAlgoCheck(handle, &desc_, &b.a, &b.b, &b.d, &b.d, &algo_[0], &check) ==
                              CUBLAS_STATUS_SUCCESS &&
                          check.workspaceSize == 0;
            }
            if (batched) {
                fused |= uint32_t{1} << (m - 1);
                rows_batched_ |= uint32_t{1} << (m - 1);
            } else if (gemv) {
                fused |= uint32_t{1} << (m - 1);
                rows_gemv_ |= uint32_t{1} << (m - 1);
            }
        }
    }
    rows_fused_ = fused;
    gemv_order_ = order;
    rows_allowed_ = allowed;
    buckets_ = 1;
    return CERE_OK;
}

bool GemmPlan::per_row(uint32_t m) const {
    return rows_fused_ != 0 && m >= 1 && m <= max_m_ && (rows_mask() & (uint32_t{1} << (m - 1))) == 0;
}

cere_status GemmPlan::run_fp8(const k2f::Entry &e, const __nv_bfloat16 *x, uint32_t m, void *y, cudaStream_t stream,
                              uint32_t cls) const {
    if (e.n != n_ || e.k != k_ || !k2f::supported(k_)) {
        return CERE_ERR_ARGUMENT;
    }
    const auto *q = static_cast<const uint8_t *>(e.w);
    const float *scale = k2f::scales_of(e.w, n_, k_);
    if (cls <= k2f::kMaxM) {
        const k2f::Segment seg{q, scale, y, n_};
        return k2f::gemm(x, m, k_, &seg, 1, out_ == GemmOut::kF32, stream) == cudaSuccess ? CERE_OK
                                                                                         : CERE_ERR_INTERNAL;
    }
    __nv_bfloat16 *staging = fp8_->staging();
    if (staging == nullptr || fp8_->staging_bytes() < size_t{n_} * k_ * sizeof(__nv_bfloat16)) {
        return CERE_ERR_UNSUPPORTED;
    }
    if (k2f::dequantize(q, scale, n_, k_, staging, stream) != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    return run_base(x, staging, m, y, stream, cls);  // staging is no table entry: the BF16 path
}

// fp4local design note S3 (k2_mxfp8.h): the bits of run(x, upcast(w), ...). The choices are run()'s for a BF16 w,
// made the same way: where that call runs the small-M kernel (a rows plan with small-M on: one launch of m rows or m
// of one row; an init() plan: cls <= k2s::kMaxM with small-M on), the same launches through k2x::gemm_mxfp8
// (k2_fast.h: k2s::gemm_mxfp8's bits, the faster kernel per shape; P11R-24); every other path (cuBLASLt buckets and
// pins, the rows plans' gemv, batched and per-row calls) runs on the upcast dequantized into the table's staging
// buffer. W8A8 (use_a8, k2-prefill design note P3): an init() plan with the row choice above k2s::kMaxM instead
// quantizes x into the staging buffer and runs cuBLASLt's MXFP8 GEMM (numerics-changing, opt-in).
cere_status GemmPlan::run_mxfp8(const k2m::Entry &e, const __nv_bfloat16 *x, uint32_t m, void *y,
                                cudaStream_t stream, uint32_t cls) const {
    if (e.n != n_ || e.k != k_ || k_ % k2m::kGroupK != 0 || e.scales == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    const auto *q = static_cast<const uint8_t *>(e.w);
    if (small_m_ && (rows_fused_ != 0 || cls <= k2s::kMaxM)) {
        const bool fused = rows_fused_ == 0 || (rows_mask() & (uint32_t{1} << (m - 1))) != 0;
        const uint32_t calls = fused ? 1 : m;
        const uint32_t rows = fused ? m : 1;
        const size_t y_row = size_t{n_} * (out_ == GemmOut::kF32 ? 4 : 2);
        for (uint32_t r = 0; r < calls; ++r) {
            const k2s::Mxfp8Segment seg{q, e.scales, static_cast<char *>(y) + r * y_row, n_};
            if (k2x::gemm_mxfp8(x + size_t{r} * k_, rows, k_, &seg, 1, out_ == GemmOut::kF32, stream) !=
                cudaSuccess) {
                return CERE_ERR_INTERNAL;
            }
        }
        return CERE_OK;
    }
    if (a8_ && rows_fused_ == 0 && cls > k2s::kMaxM) {
        auto *base = reinterpret_cast<uint8_t *>(mxfp8_->staging());
        if (base == nullptr || mxfp8_->staging_bytes() < a8_staging_bytes(m, n_, k_)) {
            return CERE_ERR_UNSUPPORTED;
        }
        uint8_t *xs = base + align_up(size_t{m} * k_, kA8Align);
        uint8_t *ws = xs + align_up(k2m::blocked_scale_bytes(m, k_ / k2m::kGroupK), kA8Align);
        const cere_status s = cuda_status(k2m::quantize_x(x, m, k_, base, xs, stream));
        return s != CERE_OK ? s : matmul_a8(e, base, xs, ws, m, y, stream, cls);
    }
    __nv_bfloat16 *staging = mxfp8_->staging();
    if (staging == nullptr || mxfp8_->staging_bytes() < size_t{n_} * k_ * sizeof(__nv_bfloat16)) {
        return CERE_ERR_UNSUPPORTED;
    }
    if (k2m::dequantize(q, e.scales, n_, k_, staging, stream) != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    return run_base(x, staging, m, y, stream, cls);  // staging is no table entry: the BF16 path
}

cere_status GemmPlan::run(const __nv_bfloat16 *x, const __nv_bfloat16 *w, uint32_t m, void *y, cudaStream_t stream,
                          uint32_t cls) const {
    const cere_status s = run_base(x, w, m, y, stream, cls);
    if (s != CERE_OK || lora_ == nullptr || !lora_->active()) {
        return s;
    }
    const lora::Entry *e = lora_->find(w);
    if (e == nullptr) {
        return CERE_OK;
    }
    if (out_ != GemmOut::kBF16 || e->k != k_) {
        return CERE_ERR_ARGUMENT;
    }
    return lora_->apply(*e, x, m, static_cast<__nv_bfloat16 *>(y), n_, stream);
}

cere_status GemmPlan::run_base(const __nv_bfloat16 *x, const __nv_bfloat16 *w, uint32_t m, void *y,
                               cudaStream_t stream, uint32_t cls) const {
    if (buckets_ == 0 || m == 0 || m > max_m_ || x == nullptr || w == nullptr || y == nullptr ||
        !aligned(x, align_x_) || !aligned(w, align_w_) || !aligned(y, align_y_) || (cls != 0 && cls < m) ||
        cls > max_m_) {
        return CERE_ERR_ARGUMENT;
    }
    // The row count every choice below is made for (D15⁗′): the call's own m unless it stands for a longer one.
    const uint32_t sel = cls != 0 ? cls : m;
    if (fp8_ != nullptr) {
        if (const k2f::Entry *e = fp8_->find(w); e != nullptr) {
            return run_fp8(*e, x, m, y, stream, sel);
        }
    }
    if (mxfp8_ != nullptr) {
        if (const k2m::Entry *e = mxfp8_->find(w); e != nullptr) {
            return run_mxfp8(*e, x, m, y, stream, sel);
        }
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (rows_fused_ != 0) {
        // Rows plan: one call of m rows (small-M kernel, the m = 1 algorithm on the rows or the strided-batched layout,
        // or the k2v kernel in that algorithm's order), or m single-row calls where none serves m (or the kernel test
        // forced the per-row path).
        if (gemv_rows(m)) {
            return k2v::gemv(x, m, k_, w, n_, y, out_ == GemmOut::kF32, gemv_order_, stream) == cudaSuccess
                       ? CERE_OK
                       : CERE_ERR_INTERNAL;
        }
        if (batched_rows(m)) {
            Layouts b;
            if (init_batched_layouts(&b, n_, k_, m, out_) != CUBLAS_STATUS_SUCCESS) {
                return CERE_ERR_INTERNAL;
            }
            return cublasLtMatmul(handle_, const_cast<cublasLtMatmulDescOpaque_t *>(&desc_), &alpha, w, &b.a, x, &b.b,
                                  &beta, y, &b.d, y, &b.d, &algo_[0], nullptr, 0, stream) == CUBLAS_STATUS_SUCCESS
                       ? CERE_OK
                       : CERE_ERR_INTERNAL;
        }
        const bool fused = (rows_mask() & (uint32_t{1} << (m - 1))) != 0;
        const uint32_t calls = fused ? 1 : m;
        const uint32_t rows = fused ? m : 1;
        const size_t y_row = size_t{n_} * (out_ == GemmOut::kF32 ? 4 : 2);
        Layouts l;
        if (!small_m_ && init_layouts(&l, n_, k_, rows, out_) != CUBLAS_STATUS_SUCCESS) {
            return CERE_ERR_INTERNAL;
        }
        for (uint32_t r = 0; r < calls; ++r) {
            const __nv_bfloat16 *xr = x + size_t{r} * k_;
            char *yr = static_cast<char *>(y) + r * y_row;
            if (small_m_) {
                const k2s::Segment seg{w, yr, n_};
                if (k2s::gemm(xr, rows, k_, &seg, 1, out_ == GemmOut::kF32, stream, early_) != cudaSuccess) {
                    return CERE_ERR_INTERNAL;
                }
                continue;
            }
            const cublasStatus_t s =
                cublasLtMatmul(handle_, const_cast<cublasLtMatmulDescOpaque_t *>(&desc_), &alpha, w, &l.a, xr, &l.b,
                               &beta, yr, &l.d, yr, &l.d, &algo_[0], nullptr, 0, stream);
            if (s != CUBLAS_STATUS_SUCCESS) {
                return CERE_ERR_INTERNAL;
            }
        }
        return CERE_OK;
    }
    if (small_m_ && sel <= k2s::kMaxM) {
        const k2s::Segment seg{w, y, n_};
        return k2s::gemm(x, m, k_, &seg, 1, out_ == GemmOut::kF32, stream, early_) == cudaSuccess ? CERE_OK
                                                                                                   : CERE_ERR_INTERNAL;
    }
    Layouts l;
    if (init_layouts(&l, n_, k_, m, out_) != CUBLAS_STATUS_SUCCESS) {
        return CERE_ERR_INTERNAL;
    }
    for (uint32_t i = 0; i < pin_count_; ++i) {
        if (sel >= pin_range_[i].m_lo && sel <= pin_range_[i].m_hi) {
            const cublasStatus_t s =
                cublasLtMatmul(handle_, const_cast<cublasLtMatmulDescOpaque_t *>(&desc_), &alpha, w, &l.a, x, &l.b,
                               &beta, y, &l.d, y, &l.d, &pin_algo_[i], workspace_, workspace_bytes_, stream);
            return s == CUBLAS_STATUS_SUCCESS ? CERE_OK : CERE_ERR_INTERNAL;
        }
    }
    const cublasStatus_t s =
        cublasLtMatmul(handle_, const_cast<cublasLtMatmulDescOpaque_t *>(&desc_), &alpha, w, &l.a, x, &l.b, &beta, y,
                       &l.d, y, &l.d, &algo_[ceil_log2(sel)], nullptr, 0, stream);
    return s == CUBLAS_STATUS_SUCCESS ? CERE_OK : CERE_ERR_INTERNAL;
}

cere_status lm_head_last_row(const GemmPlan &plan, const __nv_bfloat16 *hidden, uint32_t m, const __nv_bfloat16 *w,
                             void *logits, cudaStream_t stream) {
    if (m == 0 || hidden == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    return plan.run(hidden + static_cast<size_t>(m - 1) * plan.k(), w, 1, logits, stream);
}

namespace {

static_assert(lora::kMaxRows == GemmPlan::kMaxRows, "a rows call's per-row adapter slots");

// A one-launch gemm_segments call's status, then each segment's LoRA pass (the run() fallbacks apply their own).
cere_status segments_done(cudaError_t launched, const GemmPlan *const *plans, const __nv_bfloat16 *const *w,
                          void *const *y, uint32_t count, const __nv_bfloat16 *x, uint32_t m, cudaStream_t stream) {
    if (launched == cudaErrorInvalidValue) {
        return CERE_ERR_ARGUMENT;
    }
    if (launched != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    for (uint32_t i = 0; i < count; ++i) {
        lora::Table *t = plans[i]->lora();
        const lora::Entry *e = t != nullptr && t->active() ? t->find(w[i]) : nullptr;
        if (e == nullptr) {
            continue;
        }
        if (plans[i]->out() != GemmOut::kBF16 || e->k != plans[i]->k()) {
            return CERE_ERR_ARGUMENT;
        }
        if (const cere_status s = t->apply(*e, x, m, static_cast<__nv_bfloat16 *>(y[i]), plans[i]->n(), stream);
            s != CERE_OK) {
            return s;
        }
    }
    return CERE_OK;
}

}  // namespace

cere_status gemm_segments(const GemmPlan *const *plans, const __nv_bfloat16 *const *w, void *const *y, uint32_t count,
                          const __nv_bfloat16 *x, uint32_t m, cudaStream_t stream, uint32_t cls) {
    const uint32_t sel = cls != 0 ? cls : m;
    if (plans == nullptr || w == nullptr || y == nullptr || count == 0) {
        return CERE_ERR_ARGUMENT;
    }
    // FP8 weights (k2_weights lever): one k2f launch when every segment is a converted weight of its plan's shape and
    // m <= k2f::kMaxM, else each plan's run() (which takes the FP8 path per weight).
    if (plans[0] != nullptr && plans[0]->fp8() != nullptr) {
        const k2f::Table *table = plans[0]->fp8();
        bool any = false;
        bool one = count <= k2f::kMaxSegments && sel <= k2f::kMaxM;
        k2f::Segment fsegs[k2f::kMaxSegments]{};
        for (uint32_t i = 0; i < count; ++i) {
            const GemmPlan *p = plans[i];
            const k2f::Entry *e = p != nullptr && w[i] != nullptr ? table->find(w[i]) : nullptr;
            any = any || e != nullptr;
            one = one && e != nullptr && p->fp8() == table && p->k() == plans[0]->k() &&
                  p->out() == plans[0]->out() && e->n == p->n() && e->k == p->k() && m <= p->max_m() &&
                  y[i] != nullptr && aligned(y[i], p->out() == GemmOut::kF32 ? 4 : 2);
            if (one) {
                fsegs[i] = k2f::Segment{static_cast<const uint8_t *>(e->w), k2f::scales_of(e->w, e->n, e->k), y[i],
                                        e->n};
            }
        }
        if (any && one) {
            return segments_done(
                k2f::gemm(x, m, plans[0]->k(), fsegs, count, plans[0]->out() == GemmOut::kF32, stream), plans, w, y,
                count, x, m, stream);
        }
        if (any) {
            for (uint32_t i = 0; i < count; ++i) {
                if (plans[i] == nullptr) {
                    return CERE_ERR_ARGUMENT;
                }
                const cere_status s = plans[i]->run(x, w[i], m, y[i], stream, cls);
                if (s != CERE_OK) {
                    return s;
                }
            }
            return CERE_OK;
        }
    }
    // W8A8 (use_a8, sel > k2s::kMaxM): two or more segments on the path share one quantize_x (each run() would
    // quantize x to the same bits). The other segments run first: a staging-path segment dequantizes its weight into
    // the buffer the shared E4M3 x then occupies.
    if (sel > k2s::kMaxM && count <= k2s::kMaxSegments) {
        const k2m::Entry *a8[k2s::kMaxSegments]{};
        const GemmPlan *first = nullptr;
        uint32_t shared = 0;
        size_t need = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const GemmPlan *p = plans[i];
            a8[i] = p != nullptr ? p->a8_entry(x, w[i], m, y[i], cls) : nullptr;
            if (a8[i] != nullptr && (first == nullptr || (p->mxfp8_ == first->mxfp8_ && p->k_ == first->k_))) {
                first = first == nullptr ? p : first;
                need = std::max(need, GemmPlan::a8_staging_bytes(m, p->n_, p->k_));
                ++shared;
            } else {
                a8[i] = nullptr;
            }
        }
        if (shared >= 2) {
            for (uint32_t i = 0; i < count; ++i) {
                if (a8[i] == nullptr) {
                    if (plans[i] == nullptr) {
                        return CERE_ERR_ARGUMENT;
                    }
                    const cere_status s = plans[i]->run(x, w[i], m, y[i], stream, cls);
                    if (s != CERE_OK) {
                        return s;
                    }
                }
            }
            auto *base = reinterpret_cast<uint8_t *>(first->mxfp8_->staging());
            if (base == nullptr || first->mxfp8_->staging_bytes() < need) {
                return CERE_ERR_UNSUPPORTED;
            }
            const uint32_t k = first->k_;
            uint8_t *xs = base + align_up(size_t{m} * k, kA8Align);
            uint8_t *ws = xs + align_up(k2m::blocked_scale_bytes(m, k / k2m::kGroupK), kA8Align);
            cere_status s = cuda_status(k2m::quantize_x(x, m, k, base, xs, stream));
            // The W8A8 segments' LoRA pass (run() applied it to the others above).
            const GemmPlan *a8_plans[k2s::kMaxSegments]{};
            const __nv_bfloat16 *a8_w[k2s::kMaxSegments]{};
            void *a8_y[k2s::kMaxSegments]{};
            uint32_t a8_count = 0;
            for (uint32_t i = 0; i < count && s == CERE_OK; ++i) {
                if (a8[i] != nullptr) {
                    s = plans[i]->matmul_a8(*a8[i], base, xs, ws, m, y[i], stream, sel);
                    a8_plans[a8_count] = plans[i];
                    a8_w[a8_count] = w[i];
                    a8_y[a8_count++] = y[i];
                }
            }
            return s != CERE_OK ? s : segments_done(cudaSuccess, a8_plans, a8_w, a8_y, a8_count, x, m, stream);
        }
    }
    bool one_launch = count <= k2s::kMaxSegments && sel <= k2s::kMaxM;
    for (uint32_t i = 0; i < count && one_launch; ++i) {
        const GemmPlan *p = plans[i];
        // Rows plans forced per row (kernel tests) keep their per-row calls.
        one_launch = p != nullptr && p->small_m() && p->k() == plans[0]->k() && p->out() == plans[0]->out() &&
                     m <= p->max_m() && !p->per_row(m);
    }
    // MXFP8 weights (track Fp4S3): the rule above decides as for their BF16 upcasts. Where it holds and every segment
    // is an MXFP8 weight of its plan, one k2x::gemm_mxfp8 launch (the BF16 launch's bits); a mix of MXFP8 and BF16
    // weights runs each plan's run() (a segment's y depends on x, its w and k alone, k2_small_m.h, so the bits are the
    // same).
    k2s::Mxfp8Segment msegs[k2s::kMaxSegments]{};
    uint32_t mx = 0;
    for (uint32_t i = 0; i < count && one_launch; ++i) {
        const k2m::Table *t = plans[i]->mxfp8();
        const k2m::Entry *e = t != nullptr && w[i] != nullptr ? t->find(w[i]) : nullptr;
        if (e == nullptr) {
            continue;
        }
        if (e->n != plans[i]->n() || e->k != plans[i]->k() || e->scales == nullptr || y[i] == nullptr ||
            !aligned(y[i], plans[i]->out() == GemmOut::kF32 ? 4 : 2)) {
            one_launch = false;  // run() refuses it
            break;
        }
        msegs[i] = k2s::Mxfp8Segment{static_cast<const uint8_t *>(e->w), e->scales, y[i], e->n};
        ++mx;
    }
    one_launch = one_launch && (mx == 0 || mx == count);
    if (!one_launch) {
        for (uint32_t i = 0; i < count; ++i) {
            if (plans[i] == nullptr) {
                return CERE_ERR_ARGUMENT;
            }
            const cere_status s = plans[i]->run(x, w[i], m, y[i], stream, cls);
            if (s != CERE_OK) {
                return s;
            }
        }
        return CERE_OK;
    }
    if (mx == count) {
        return segments_done(
            k2x::gemm_mxfp8(x, m, plans[0]->k(), msegs, count, plans[0]->out() == GemmOut::kF32, stream), plans, w, y,
            count, x, m, stream);
    }
    k2s::Segment segs[k2s::kMaxSegments];
    for (uint32_t i = 0; i < count; ++i) {
        if (w[i] == nullptr || y[i] == nullptr || !aligned(y[i], plans[i]->out() == GemmOut::kF32 ? 4 : 2)) {
            return CERE_ERR_ARGUMENT;
        }
        segs[i] = k2s::Segment{w[i], y[i], plans[i]->n()};
    }
    return segments_done(k2s::gemm(x, m, plans[0]->k(), segs, count, plans[0]->out() == GemmOut::kF32, stream), plans,
                         w, y, count, x, m, stream);
}

}  // namespace cere::kernels
