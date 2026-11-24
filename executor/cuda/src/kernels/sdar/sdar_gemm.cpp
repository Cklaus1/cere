// SelfTune ST4 S3c: sdar_gemm.h. Algorithm pins: SdarAlgoPin (O324). Output-changing configs: SdarOptIn (O320).
#include "kernels/sdar/sdar_gemm.h"

#include <atomic>
#include <cstdio>
#include <utility>

namespace cere::sdar {

namespace {

bool ok(cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS; }

// DICE-8B's pins (an internal benchmark report runs 2 and 3): kv (n 1024, k 4096) at m = 2..4,
// the heuristic's third zero-workspace candidate (algorithm 21, tile 64x64, stages 32x10, no split-K) instead of its
// first (tile 16x16, stages 128x2): 20.5 µs against 28.7 µs cold-L2 at each m, bit-identical on every probe input.
// qo (2.1 µs, one timer step) is within noise and not pinned; gate_up and down already run their fastest
// bit-identical algorithm, and so does lm_head at m = 4, the only m the forward runs it at (kScheduleBlock).
constexpr AlgoPin kPins[] = {
    {1024, 4096, 2, GemmOut::kBF16, 21, CUBLASLT_MATMUL_TILE_64x64, CUBLASLT_MATMUL_STAGES_32x10, 1,
     CUBLASLT_REDUCTION_SCHEME_NONE, 0, 0, 0, 0},
    {1024, 4096, 3, GemmOut::kBF16, 21, CUBLASLT_MATMUL_TILE_64x64, CUBLASLT_MATMUL_STAGES_32x10, 1,
     CUBLASLT_REDUCTION_SCHEME_NONE, 0, 0, 0, 0},
    {1024, 4096, 4, GemmOut::kBF16, 21, CUBLASLT_MATMUL_TILE_64x64, CUBLASLT_MATMUL_STAGES_32x10, 1,
     CUBLASLT_REDUCTION_SCHEME_NONE, 0, 0, 0, 0},
};
// Where the pins were measured: cuBLASLt 13.1.1 (cublasLtGetVersion() 130101) on sm 12.0 (RTX PRO 6000 Blackwell).
constexpr PinTarget kPinTarget{130101, 12, 0};

// SdarOptIn's changing table (the same probe, the fastest config whose 3 repeats are byte-equal, cold-L2 µs against
// the identical pick): kv at m = 2..4 split-K 27 through the workspace (12.3 against 20.5), down at m = 2 and 4
// split-K 4 (71.7 against 73.7) and at m = 3 split-K 8 (70.6 against 75.8). Every entry reduces in the compute type
// through the workspace (no atomics), at most kChangingWorkspaceBytes (down at m = 3).
constexpr AlgoPin kChanging[] = {
    {1024, 4096, 2, GemmOut::kBF16, 21, CUBLASLT_MATMUL_TILE_32x32, CUBLASLT_MATMUL_STAGES_64x1, 27,
     CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE, 0, 0, 0, 0},
    {1024, 4096, 3, GemmOut::kBF16, 21, CUBLASLT_MATMUL_TILE_32x32, CUBLASLT_MATMUL_STAGES_64x1, 27,
     CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE, 0, 0, 0, 0},
    {1024, 4096, 4, GemmOut::kBF16, 21, CUBLASLT_MATMUL_TILE_32x32, CUBLASLT_MATMUL_STAGES_64x1, 27,
     CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE, 0, 0, 0, 0},
    {4096, 12288, 2, GemmOut::kBF16, 21, CUBLASLT_MATMUL_TILE_32x32, CUBLASLT_MATMUL_STAGES_128x1, 4,
     CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE, 0, 0, 0, 0},
    {4096, 12288, 3, GemmOut::kBF16, 21, CUBLASLT_MATMUL_TILE_64x64, CUBLASLT_MATMUL_STAGES_32x10, 8,
     CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE, 0, 0, 0, 0},
    {4096, 12288, 4, GemmOut::kBF16, 21, CUBLASLT_MATMUL_TILE_32x32, CUBLASLT_MATMUL_STAGES_128x1, 4,
     CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE, 0, 0, 0, 0},
};
static_assert(CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE == 2, "the probe recorded reduction scheme 2");

// SelfTune ST4 S4a-style fail-first mutant (SdarOptIn): `changing_k_short` drops the last 128 k channels of every
// GEMM that runs a changing entry (each row's leading dimension stays k), so `dice_judge` and the changing ktest
// must both FAIL it. A plain build has k_used_ == k_.
#if defined(CERE_SDAR_MUTANT_CHANGING_K_SHORT)
constexpr uint32_t kMutantShortK = 128;
#endif

// One stderr line per outcome kind per process (PinUse bit), for the built-in table only.
std::atomic<uint32_t> g_logged{0};

void log_use(PinUse use, const AlgoPin &pin, const PinTarget &process, const PinTarget &table) {
    const uint32_t bit = 1u << uint32_t(use);
    if ((g_logged.fetch_or(bit) & bit) != 0) return;
    std::fprintf(stderr,
                 "cere: sdar gemm pins: %s (n %u k %u m %u; process cuBLASLt %zu sm %d.%d, table cuBLASLt %zu "
                 "sm %d.%d)\n",
                 pin_use_name(use), pin.n, pin.k, pin.m, process.cublaslt_version, process.cc_major,
                 process.cc_minor, table.cublaslt_version, table.cc_major, table.cc_minor);
}

}  // namespace

PinTable builtin_pins() { return {kPins, sizeof(kPins) / sizeof(kPins[0]), kPinTarget}; }

PinTable changing_pins() { return {kChanging, sizeof(kChanging) / sizeof(kChanging[0]), kPinTarget}; }

cere_status runtime_target(PinTarget *out) {
    int device = 0, major = 0, minor = 0;
    if (out == nullptr) return CERE_ERR_ARGUMENT;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    *out = {cublasLtGetVersion(), major, minor};
    return CERE_OK;
}

const char *pin_use_name(PinUse use) {
    switch (use) {
        case PinUse::kHeuristic: return "heuristic (no pin)";
        case PinUse::kPinned: return "pinned";
        case PinUse::kVersion: return "fallback to the heuristic: cuBLASLt version differs";
        case PinUse::kArch: return "fallback to the heuristic: compute capability differs";
        case PinUse::kRejected: return "fallback to the heuristic: cuBLASLt rejected the pinned config";
        case PinUse::kWorkspace: return "fallback to the heuristic: the pinned config needs a workspace";
        case PinUse::kChanging: return "changing (split-K through the workspace)";
    }
    return "unknown";
}

Gemm::~Gemm() { reset(); }

Gemm::Gemm(Gemm &&other) noexcept { *this = std::move(other); }

Gemm &Gemm::operator=(Gemm &&other) noexcept {
    if (this != &other) {
        reset();
        handle_ = std::exchange(other.handle_, nullptr);
        desc_ = std::exchange(other.desc_, nullptr);
        a_ = std::exchange(other.a_, nullptr);
        n_ = std::exchange(other.n_, 0);
        k_ = std::exchange(other.k_, 0);
        max_m_ = std::exchange(other.max_m_, 0);
        out_ = other.out_;
        table_ = other.table_;
        process_ = other.process_;
        builtin_ = other.builtin_;
        workspace_ = std::exchange(other.workspace_, nullptr);
        workspace_bytes_ = std::exchange(other.workspace_bytes_, 0);
        k_used_ = std::exchange(other.k_used_, 0);
        algo_ = std::move(other.algo_);
        valid_ = std::move(other.valid_);
        use_ = std::move(other.use_);
    }
    return *this;
}

void Gemm::reset() {
    if (a_ != nullptr) cublasLtMatrixLayoutDestroy(a_);
    if (desc_ != nullptr) cublasLtMatmulDescDestroy(desc_);
    a_ = nullptr;
    desc_ = nullptr;
    handle_ = nullptr;
    n_ = k_ = max_m_ = k_used_ = 0;
    workspace_ = nullptr;
    workspace_bytes_ = 0;
    algo_.clear();
    valid_.clear();
    use_.clear();
}

// Column-major view: y^T [n, m] = w [n, k] (row-major = column-major w^T [k, n], transposed) · x^T [k, m].
// A is w (op T), B is x^T (op N, ld k), D is y^T (ld n).
cere_status Gemm::init(cublasLtHandle_t handle, uint32_t n, uint32_t k, uint32_t max_m, GemmOut out) {
    reset();
    if (handle == nullptr || n == 0 || k == 0 || max_m == 0) return CERE_ERR_ARGUMENT;
    PinTarget process{};
    if (const cere_status st = runtime_target(&process); st != CERE_OK) return st;
    if (!ok(cublasLtMatmulDescCreate(&desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F))) {
        reset();
        return CERE_ERR_INTERNAL;
    }
    const cublasOperation_t trans = CUBLAS_OP_T, none = CUBLAS_OP_N;
    if (!ok(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans))) ||
        !ok(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_TRANSB, &none, sizeof(none)))) {
        reset();
        return CERE_ERR_INTERNAL;
    }
    n_ = n;
    k_ = k;
    k_used_ = k;
    if (const cere_status st = make_a(); st != CERE_OK) {
        reset();
        return st;
    }
    handle_ = handle;
    max_m_ = max_m;
    out_ = out;
    table_ = builtin_pins();
    process_ = process;
    builtin_ = true;
    algo_.resize(max_m);
    valid_.assign(max_m, 0);
    use_.assign(max_m, PinUse::kHeuristic);
    return CERE_OK;
}

void Gemm::set_pins(PinTable table, PinTarget process) {
    table_ = table;
    process_ = process;
    builtin_ = false;
    valid_.assign(max_m_, 0);
    use_.assign(max_m_, PinUse::kHeuristic);
}

PinUse Gemm::pin_use(uint32_t m) const { return m >= 1 && m <= max_m_ ? use_[m - 1] : PinUse::kHeuristic; }

// A is w [n, k] row-major, of which the layouts read the first k_used_ columns of each row (leading dimension k_).
cere_status Gemm::make_a() {
    if (a_ != nullptr) cublasLtMatrixLayoutDestroy(a_);
    a_ = nullptr;
    return ok(cublasLtMatrixLayoutCreate(&a_, CUDA_R_16BF, k_used_, n_, k_)) ? CERE_OK : CERE_ERR_INTERNAL;
}

cere_status Gemm::set_changing(PinTable table, PinTarget process, void *workspace, size_t workspace_bytes) {
    if (handle_ == nullptr) return CERE_ERR_ARGUMENT;
    std::vector<const AlgoPin *> entry(max_m_, nullptr);
    bool any = false;
    for (size_t i = 0; i < table.count; ++i) {
        const AlgoPin &p = table.pins[i];
        if (p.n == n_ && p.k == k_ && p.out == out_ && p.m >= 1 && p.m <= max_m_ && entry[p.m - 1] == nullptr) {
            entry[p.m - 1] = &p;
            any = true;
        }
    }
    if (!any) return CERE_OK;
    if (workspace == nullptr) return CERE_ERR_ARGUMENT;
    const auto refuse = [&](const char *why, uint32_t m) {
        std::fprintf(stderr, "cere: refused sdar_gemm.changing (%s n %u k %u m %u)\n", why, n_, k_, m);
        return CERE_ERR_UNSUPPORTED;
    };
    if (process.cublaslt_version != table.target.cublaslt_version) return refuse("cuBLASLt version", 0);
    if (process.cc_major != table.target.cc_major || process.cc_minor != table.target.cc_minor) {
        return refuse("compute capability", 0);
    }
    const uint32_t k_used = k_used_;
#if defined(CERE_SDAR_MUTANT_CHANGING_K_SHORT)
    if (k_ > kMutantShortK) {
        k_used_ = k_ - kMutantShortK;
        if (const cere_status st = make_a(); st != CERE_OK) return st;
    }
#endif
    std::vector<cublasLtMatmulAlgo_t> algo(max_m_);
    cere_status status = CERE_OK;
    for (uint32_t m = 1; m <= max_m_ && status == CERE_OK; ++m) {
        if (entry[m - 1] == nullptr) continue;
        cublasLtMatrixLayout_t b = nullptr, d = nullptr;
        status = layouts(m, &b, &d);
        if (status != CERE_OK) break;
        const PinUse use = pinned(*entry[m - 1], b, d, workspace_bytes, &algo[m - 1]);
        if (use == PinUse::kWorkspace) status = refuse("workspace", m);
        if (use == PinUse::kRejected) status = refuse("config", m);
        cublasLtMatrixLayoutDestroy(b);
        cublasLtMatrixLayoutDestroy(d);
    }
    if (status != CERE_OK) {
        if (k_used_ != k_used) {
            k_used_ = k_used;
            if (const cere_status st = make_a(); st != CERE_OK) return st;
        }
        return status;
    }
    if (k_used_ != k_used) valid_.assign(max_m_, 0);  // the mutant's layouts: every pick again
    for (uint32_t m = 1; m <= max_m_; ++m) {
        if (entry[m - 1] == nullptr) continue;
        algo_[m - 1] = algo[m - 1];
        use_[m - 1] = PinUse::kChanging;
        valid_[m - 1] = 1;
    }
    workspace_ = workspace;
    workspace_bytes_ = workspace_bytes;
    return CERE_OK;
}

cere_status Gemm::layouts(uint32_t m, cublasLtMatrixLayout_t *b, cublasLtMatrixLayout_t *d) const {
    const cudaDataType_t dtype = out_ == GemmOut::kF32 ? CUDA_R_32F : CUDA_R_16BF;
    if (!ok(cublasLtMatrixLayoutCreate(b, CUDA_R_16BF, k_used_, m, k_))) return CERE_ERR_INTERNAL;
    if (!ok(cublasLtMatrixLayoutCreate(d, dtype, n_, m, n_))) {
        cublasLtMatrixLayoutDestroy(*b);
        return CERE_ERR_INTERNAL;
    }
    return CERE_OK;
}

// The pinned config as an algorithm, accepted only if cublasLtMatmulAlgoCheck passes it within max_workspace bytes.
PinUse Gemm::pinned(const AlgoPin &pin, cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t d, size_t max_workspace,
                    cublasLtMatmulAlgo_t *algo) const {
    const cudaDataType_t dtype = out_ == GemmOut::kF32 ? CUDA_R_32F : CUDA_R_16BF;
    const uint16_t inner = uint16_t(pin.inner_shape), cluster = uint16_t(pin.cluster);
    if (!ok(cublasLtMatmulAlgoInit(handle_, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, dtype, dtype,
                                   pin.algo_id, algo)) ||
        !ok(cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &pin.tile, sizeof(pin.tile))) ||
        !ok(cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &pin.stages,
                                                 sizeof(pin.stages))) ||
        !ok(cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &pin.splitk,
                                                 sizeof(pin.splitk))) ||
        !ok(cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &pin.reduction,
                                                 sizeof(pin.reduction))) ||
        !ok(cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &pin.swizzle,
                                                 sizeof(pin.swizzle))) ||
        !ok(cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &pin.custom,
                                                 sizeof(pin.custom))) ||
        !ok(cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner,
                                                 sizeof(inner))) ||
        !ok(cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster,
                                                 sizeof(cluster)))) {
        return PinUse::kRejected;
    }
    cublasLtMatmulHeuristicResult_t check{};
    if (!ok(cublasLtMatmulAlgoCheck(handle_, desc_, a_, b, d, d, algo, &check))) return PinUse::kRejected;
    return check.workspaceSize <= max_workspace ? PinUse::kPinned : PinUse::kWorkspace;
}

// m's algorithm: the table's pin when one applies (sdar_gemm.h), else the heuristic's first zero-workspace one.
cere_status Gemm::pick(uint32_t m, cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t d) {
    const AlgoPin *pin = nullptr;
    for (size_t i = 0; i < table_.count && pin == nullptr; ++i) {
        const AlgoPin &p = table_.pins[i];
        if (p.n == n_ && p.k == k_ && p.m == m && p.out == out_) pin = &p;
    }
    PinUse use = PinUse::kHeuristic;
    if (pin != nullptr) {
        if (process_.cublaslt_version != table_.target.cublaslt_version) {
            use = PinUse::kVersion;
        } else if (process_.cc_major != table_.target.cc_major || process_.cc_minor != table_.target.cc_minor) {
            use = PinUse::kArch;
        } else {
            use = pinned(*pin, b, d, 0, &algo_[m - 1]);
        }
        if (builtin_) log_use(use, *pin, process_, table_.target);
    }
    if (use != PinUse::kPinned) {
        cublasLtMatmulPreference_t pref = nullptr;
        const size_t workspace = 0;
        cublasLtMatmulHeuristicResult_t result{};
        int found = 0;
        cere_status status = CERE_OK;
        if (!ok(cublasLtMatmulPreferenceCreate(&pref)) ||
            !ok(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
                                                     sizeof(workspace)))) {
            status = CERE_ERR_INTERNAL;
        } else if (!ok(cublasLtMatmulAlgoGetHeuristic(handle_, desc_, a_, b, d, d, pref, 1, &result, &found)) ||
                   found == 0) {
            status = CERE_ERR_UNSUPPORTED;
        } else {
            algo_[m - 1] = result.algo;
        }
        if (pref != nullptr) cublasLtMatmulPreferenceDestroy(pref);
        if (status != CERE_OK) return status;
    }
    use_[m - 1] = use;
    valid_[m - 1] = 1;
    return CERE_OK;
}

cere_status Gemm::run(const __nv_bfloat16 *x, const __nv_bfloat16 *w, uint32_t m, void *y, cudaStream_t stream) {
    if (handle_ == nullptr || m == 0 || m > max_m_ || x == nullptr || w == nullptr || y == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtMatrixLayout_t b = nullptr, d = nullptr;
    if (const cere_status st = layouts(m, &b, &d); st != CERE_OK) return st;
    cere_status status = valid_[m - 1] == 0 ? pick(m, b, d) : CERE_OK;
    if (status == CERE_OK) {
        const float alpha = 1.0f, beta = 0.0f;
        const bool changing = use_[m - 1] == PinUse::kChanging;
        if (!ok(cublasLtMatmul(handle_, desc_, &alpha, w, a_, x, b, &beta, y, d, y, d, &algo_[m - 1],
                               changing ? workspace_ : nullptr, changing ? workspace_bytes_ : 0, stream))) {
            status = CERE_ERR_INTERNAL;
        }
    }
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(d);
    return status;
}

}  // namespace cere::sdar
