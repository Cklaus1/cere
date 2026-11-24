// SelfTune ST4 S3c (selftune design note ST4): the GEMM of family sdar (DICE-8B), y = x · w^T on cuBLASLt with
// BF16 operands and an FP32 accumulator. Its own plan instead of K2's GemmPlan: K2 links the small-M kernel, whose
// hyper-connection epilogue is compiled for the Qwen model's stream count (kernels/k3_hc_math.cuh, kHcCount),
// which an sdar library's generated header does not have. One algorithm per row count m, picked once and kept. The
// default (lever `sdar_gemm` = identical) runs zero workspace only, so no split-K reduction through a workspace and
// no atomics: a call's bits depend only on m and its operands.
//
// The pick (SdarAlgoPin, O324): a pinned algorithm when the pin table (kPins) has an entry for (n, k, m, out),
// else the heuristic's first zero-workspace algorithm. An entry is in the table only when the probe
// (an internal benchmark report) measured it faster than the heuristic's pick by >= 2 timer steps on every run
// and bit-identical to it; the guard keeps it on that target. A pin falls back to the heuristic (fail-closed) when
// the cuBLASLt version or the device's compute capability differs from kPinTarget, cublasLtMatmulAlgoCheck rejects
// its config, or it needs a workspace. The first resolution of the built-in table in a process prints one
// `cere: sdar gemm pins:` line to stderr (pinned, or the fallback and its reason).
//
// SdarOptIn (O320, lever `sdar_gemm` = changing, restart class, opt-in only): set_changing replaces the pick for
// every m the changing table (changing_pins) lists for this (n, k, out) with its split-K config, which reduces
// through the caller's workspace with no atomics, so its bits still depend only on m and its operands but differ
// from the identical pick's. It never falls back: a cuBLASLt version or compute capability other than the table's,
// a config cublasLtMatmulAlgoCheck refuses, or a workspace larger than the caller's is refused (UNSUPPORTED), and
// the create that asked for it fails naming `sdar_gemm.changing`.
//
// modeling_sdar.py nn.Linear: bias-free; the BF16 output is one rounding of the FP32 accumulator.
// math_reference: DICE-8B
#pragma once

#include "cere_executor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cere::sdar {

enum class GemmOut : uint32_t {
    kBF16 = 0,  // y is __nv_bfloat16
    kF32 = 1,   // y is float
};

// One algorithm config (cublasLtMatmulAlgoConfigAttributes_t values) for one (n, k, m, out).
struct AlgoPin {
    uint32_t n, k, m;
    GemmOut out;
    int32_t algo_id;     // CUBLASLT_ALGO_CONFIG_ID
    uint32_t tile;       // cublasLtMatmulTile_t
    uint32_t stages;     // cublasLtMatmulStages_t
    int32_t splitk;      // CUBLASLT_ALGO_CONFIG_SPLITK_NUM
    uint32_t reduction;  // cublasLtReductionScheme_t
    uint32_t swizzle;    // CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING
    uint32_t custom;     // CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION
    uint32_t inner_shape;
    uint32_t cluster;
};

// The cuBLASLt version (cublasLtGetVersion()) and compute capability a table was measured on, or a process runs on.
struct PinTarget {
    size_t cublaslt_version;
    int cc_major, cc_minor;
};

struct PinTable {
    const AlgoPin *pins;
    size_t count;
    PinTarget target;
};

// How the algorithm for one m was picked.
enum class PinUse : uint32_t {
    kHeuristic = 0,  // no table entry: the heuristic's first zero-workspace algorithm
    kPinned = 1,     // the table's algorithm
    kVersion = 2,    // fallback to the heuristic: the cuBLASLt version differs from the table's target
    kArch = 3,       // fallback: the compute capability differs
    kRejected = 4,   // fallback: cublasLtMatmulAlgoInit / config attributes / cublasLtMatmulAlgoCheck refused the pin
    kWorkspace = 5,  // fallback: the pinned config needs a workspace
    kChanging = 6,   // the changing table's split-K config (set_changing; output differs from the identical pick)
};

// The built-in table (DICE-8B's measured pins) and this process's target (current device).
PinTable builtin_pins();
cere_status runtime_target(PinTarget *out);
// The changing table (SdarOptIn, measured by an internal benchmark report) and the workspace its
// largest entry needs; every entry needs at most kChangingWorkspaceBytes.
PinTable changing_pins();
constexpr size_t kChangingWorkspaceBytes = 393216;
const char *pin_use_name(PinUse use);

class Gemm {
public:
    Gemm() = default;
    ~Gemm();
    Gemm(const Gemm &) = delete;
    Gemm &operator=(const Gemm &) = delete;
    Gemm(Gemm &&other) noexcept;
    Gemm &operator=(Gemm &&other) noexcept;

    // w is [n, k] row-major (an nn.Linear weight), x [m, k] and y [m, n] row-major, 1 <= m <= max_m. Host only.
    // Uses builtin_pins() on runtime_target(). CERE_ERR_ARGUMENT for a null handle or a zero dimension,
    // CERE_ERR_INTERNAL for a cuBLASLt or CUDA failure.
    cere_status init(cublasLtHandle_t handle, uint32_t n, uint32_t k, uint32_t max_m, GemmOut out);
    // Test hook (ktest/sdar/sdar_gemm_pin_ktest.cpp): replace the table and the process's target before the first
    // run; forgets every pick. An overridden table prints no `cere: sdar gemm pins:` line.
    void set_pins(PinTable table, PinTarget process);
    // After init (and any set_pins): every m <= max_m the table lists for this (n, k, out) runs its entry, through
    // `workspace` (device, `workspace_bytes`). CERE_ERR_ARGUMENT for a null workspace when an entry applies or before
    // init; CERE_ERR_UNSUPPORTED, printing `cere: refused sdar_gemm.changing (<reason> n k m)`, when the process's
    // cuBLASLt version or compute capability differs from the table's, cuBLASLt refuses an entry, or it needs more
    // than workspace_bytes (nothing is changed then). A table with no entry for this shape changes nothing.
    cere_status set_changing(PinTable table, PinTarget process, void *workspace, size_t workspace_bytes);
    // Device pointers, 16-byte aligned. CERE_ERR_ARGUMENT for m outside [1, max_m]; CERE_ERR_UNSUPPORTED when
    // cuBLASLt offers no zero-workspace algorithm for m; CERE_ERR_INTERNAL when cublasLtMatmul fails.
    cere_status run(const __nv_bfloat16 *x, const __nv_bfloat16 *w, uint32_t m, void *y, cudaStream_t stream);
    // How m's algorithm was picked (kHeuristic before the first run with m).
    PinUse pin_use(uint32_t m) const;

    uint32_t n() const { return n_; }
    uint32_t k() const { return k_; }

private:
    void reset();
    cere_status layouts(uint32_t m, cublasLtMatrixLayout_t *b, cublasLtMatrixLayout_t *d) const;
    cere_status make_a();
    // The pinned config as an algorithm: kPinned when cuBLASLt accepts it within max_workspace bytes.
    PinUse pinned(const AlgoPin &pin, cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t d, size_t max_workspace,
                  cublasLtMatmulAlgo_t *algo) const;
    cere_status pick(uint32_t m, cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t d);

    cublasLtHandle_t handle_ = nullptr;
    cublasLtMatmulDesc_t desc_ = nullptr;
    cublasLtMatrixLayout_t a_ = nullptr;
    uint32_t n_ = 0, k_ = 0, max_m_ = 0;
    GemmOut out_ = GemmOut::kBF16;
    PinTable table_{nullptr, 0, {0, 0, 0}};
    PinTarget process_{0, 0, 0};
    bool builtin_ = true;
    // The changing entries' workspace (set_changing), null and 0 otherwise.
    void *workspace_ = nullptr;
    size_t workspace_bytes_ = 0;
    // The k the layouts use: k_, except in the changing_k_short mutant (sdar_gemm.cpp).
    uint32_t k_used_ = 0;
    // algo_[m - 1], picked on the first call with m (valid_[m - 1]); use_[m - 1] how.
    std::vector<cublasLtMatmulAlgo_t> algo_;
    std::vector<uint8_t> valid_;
    std::vector<PinUse> use_;
};

}  // namespace cere::sdar
