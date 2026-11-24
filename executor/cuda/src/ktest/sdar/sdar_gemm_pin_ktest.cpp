// SdarAlgoPin (O324; selftune design note ST6, DECISION (maintainer, 2026-11-13, SdarAlgoPin)): ktest entry points
// of cere::sdar::Gemm's algorithm pins (kernels/sdar/sdar_gemm.h), for executor/tests/gpu/test_sdar_kernels_pin.py.
// Compiled only into a CERE_MODEL=<sdar descriptor> library's libcere_ktest; libcere_executor does not change.
//
//   int cere_ktest_sdar_gemm_pins(void *out /*int64 [cap, 4]: n, k, m, f32*/, int64_t cap, void *target
//                                 /*int64 [6]: table cuBLASLt version, cc major, cc minor, then the process's*/,
//                                 cudaStream_t stream /*unused*/);
//       the built-in table's entries (returns the count, or a negative cere_status; cap < count is an argument
//       error) and both targets.
//   int cere_ktest_sdar_gemm_changing_pins(void *out, int64_t cap, void *target, cudaStream_t stream);
//       the same for the changing table (changing_pins, SdarOptIn O320).
//   int cere_ktest_sdar_gemm_pin(const void *x, const void *w, int64_t m, int64_t n, int64_t k, void *y,
//                                int64_t f32, int64_t mode, void *use /*int64 [1]*/, cudaStream_t stream);
//       one Gemm::run with m rows (a fresh Gemm, so the pick happens in this call), synchronised, and the PinUse it
//       picked by. mode: 0 the built-in table on this process (the library's own path); 1 no table (today's
//       heuristic: the reference a pin must equal bit for bit); 2 the built-in table, the process's cuBLASLt version
//       + 1; 3 the built-in table, the process's compute capability minor + 1; 4 a pin for (n, k, m) of an algorithm
//       id cuBLASLt does not have; 5 a pin for (n, k, m) whose tile is a cuBLASLt tile algorithm 21 does not offer
//       (the lowest id missing from its CUBLASLT_ALGO_CAP_TILE_IDS; an id outside the enum crashes
//       cublasLtMatmulAlgoCheck instead of failing it, so the built-in table holds only probed configs); 6 a pin for
//       (n, k, m) of the kv shape's split-K config (algorithm 21, tile 32x32, stages 64x1, split-K 27, reduction in
//       the compute type: needs a 180,224 B workspace at m = 2, an internal benchmark report). Modes 4-6 run on
//       this process's own target, so only the config guard can refuse them. SdarOptIn (O320, Gemm::set_changing,
//       through a kChangingWorkspaceBytes workspace): 7 the changing table on this process (lever sdar_gemm =
//       changing, the library's opt-in path); 8 the changing table, the process's cuBLASLt version + 1; 9 the
//       changing table, the process's compute capability minor + 1; 10 the changing table on this process with a
//       workspace of 256 B. Modes 8-10 must fail UNSUPPORTED (never fall back) for a shape the table lists.
#include <cstdint>

#include <cublasLt.h>
#include <cuda_runtime.h>

#include "cere_executor.h"
#include "kernels/sdar/sdar_gemm.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

CERE_KTEST int cere_ktest_sdar_gemm_pins(void *out, int64_t cap, void *target, cudaStream_t stream);
CERE_KTEST int cere_ktest_sdar_gemm_changing_pins(void *out, int64_t cap, void *target, cudaStream_t stream);
CERE_KTEST int cere_ktest_sdar_gemm_pin(const void *x, const void *w, int64_t m, int64_t n, int64_t k, void *y,
                                        int64_t f32, int64_t mode, void *use, cudaStream_t stream);

namespace {

using cere::sdar::AlgoPin;
using cere::sdar::GemmOut;
using cere::sdar::PinTable;
using cere::sdar::PinTarget;

constexpr int64_t kModes = 11;
constexpr size_t kSmallWorkspaceBytes = 256;

bool dims_ok(int64_t m, int64_t n, int64_t k) {
    return m >= 1 && n >= 1 && k >= 1 && m <= 4096 && n <= (int64_t(1) << 20) && k <= (int64_t(1) << 20) &&
           n % 8 == 0 && k % 8 == 0;
}

// The lowest tile id algorithm 21 does not offer for these types, 0 when it offers every one (or the query fails).
uint32_t unoffered_tile(cublasLtHandle_t handle, GemmOut out) {
    const cudaDataType_t dtype = out == GemmOut::kF32 ? CUDA_R_32F : CUDA_R_16BF;
    cublasLtMatmulAlgo_t algo{};
    uint32_t tiles[CUBLASLT_MATMUL_TILE_END] = {};
    size_t written = 0;
    if (cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, dtype, dtype, 21,
                               &algo) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tiles, sizeof(tiles), &written) !=
            CUBLAS_STATUS_SUCCESS) {
        return 0;
    }
    const size_t offered = written / sizeof(tiles[0]);
    for (uint32_t t = CUBLASLT_MATMUL_TILE_8x8; t < CUBLASLT_MATMUL_TILE_END; ++t) {
        bool found = false;
        for (size_t i = 0; i < offered && !found; ++i) found = tiles[i] == t;
        if (!found) return t;
    }
    return 0;
}

// A one-entry table for (n, k, m, out) on `target`; modes 4-6 (`tile`: mode 5's unoffered tile).
PinTable one_pin(AlgoPin *pin, int64_t mode, uint32_t n, uint32_t k, uint32_t m, GemmOut out, PinTarget target,
                 uint32_t tile) {
    *pin = {n, k, m, out, 21, CUBLASLT_MATMUL_TILE_64x64, CUBLASLT_MATMUL_STAGES_32x10, 1,
            CUBLASLT_REDUCTION_SCHEME_NONE, 0, 0, 0, 0};
    if (mode == 4) pin->algo_id = 1 << 20;
    if (mode == 5) pin->tile = tile;
    if (mode == 6) {
        pin->tile = CUBLASLT_MATMUL_TILE_32x32;
        pin->stages = CUBLASLT_MATMUL_STAGES_64x1;
        pin->splitk = 27;
        pin->reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    }
    return {pin, 1, target};
}

// `table`'s entries into out [cap, 4] and both targets into target [6]; returns the count.
int list_table(PinTable table, void *out, int64_t cap, void *target) {
    if (out == nullptr || target == nullptr || cap < 0) return CERE_ERR_ARGUMENT;
    if (int64_t(table.count) > cap) return CERE_ERR_ARGUMENT;
    PinTarget process{};
    if (const cere_status st = cere::sdar::runtime_target(&process); st != CERE_OK) return st;
    auto *o = static_cast<int64_t *>(out);
    for (size_t i = 0; i < table.count; ++i) {
        const AlgoPin &p = table.pins[i];
        o[4 * i] = p.n;
        o[4 * i + 1] = p.k;
        o[4 * i + 2] = p.m;
        o[4 * i + 3] = p.out == GemmOut::kF32 ? 1 : 0;
    }
    auto *t = static_cast<int64_t *>(target);
    t[0] = int64_t(table.target.cublaslt_version);
    t[1] = table.target.cc_major;
    t[2] = table.target.cc_minor;
    t[3] = int64_t(process.cublaslt_version);
    t[4] = process.cc_major;
    t[5] = process.cc_minor;
    return int(table.count);
}

}  // namespace

CERE_KTEST int cere_ktest_sdar_gemm_pins(void *out, int64_t cap, void *target, cudaStream_t) {
    return list_table(cere::sdar::builtin_pins(), out, cap, target);
}

CERE_KTEST int cere_ktest_sdar_gemm_changing_pins(void *out, int64_t cap, void *target, cudaStream_t) {
    return list_table(cere::sdar::changing_pins(), out, cap, target);
}

CERE_KTEST int cere_ktest_sdar_gemm_pin(const void *x, const void *w, int64_t m, int64_t n, int64_t k, void *y,
                                        int64_t f32, int64_t mode, void *use, cudaStream_t stream) {
    if (x == nullptr || w == nullptr || y == nullptr || use == nullptr || !dims_ok(m, n, k) ||
        (f32 != 0 && f32 != 1) || mode < 0 || mode >= kModes) {
        return CERE_ERR_ARGUMENT;
    }
    PinTarget process{};
    if (const cere_status st = cere::sdar::runtime_target(&process); st != CERE_OK) return st;
    cublasLtHandle_t handle = nullptr;
    if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) return CERE_ERR_INTERNAL;
    const GemmOut out = f32 ? GemmOut::kF32 : GemmOut::kBF16;
    int s;
    void *workspace = nullptr;
    if (mode >= 7 && cudaMalloc(&workspace, cere::sdar::kChangingWorkspaceBytes) != cudaSuccess) {
        cublasLtDestroy(handle);
        return CERE_ERR_CAPACITY;
    }
    {
        cere::sdar::Gemm gemm;
        s = gemm.init(handle, uint32_t(n), uint32_t(k), uint32_t(m), out);
        AlgoPin pin{};
        PinTarget other = process;
        if (mode == 1) gemm.set_pins({nullptr, 0, process}, process);
        if (mode == 2 || mode == 8) ++other.cublaslt_version;
        if (mode == 3 || mode == 9) ++other.cc_minor;
        if (mode == 2 || mode == 3) gemm.set_pins(cere::sdar::builtin_pins(), other);
        if (mode >= 4 && mode <= 6) {
            const uint32_t tile = mode == 5 ? unoffered_tile(handle, out) : 0;
            if (mode == 5 && tile == 0 && s == CERE_OK) s = CERE_ERR_UNSUPPORTED;
            gemm.set_pins(one_pin(&pin, mode, uint32_t(n), uint32_t(k), uint32_t(m), out, process, tile), process);
        }
        if (mode >= 7 && s == CERE_OK) {
            PinTable changing = cere::sdar::changing_pins();
            changing.target = other;  // modes 8 and 9: the table measured on another target
            s = gemm.set_changing(changing, process, workspace,
                                  mode == 10 ? kSmallWorkspaceBytes : cere::sdar::kChangingWorkspaceBytes);
        }
        if (s == CERE_OK) {
            s = gemm.run(static_cast<const __nv_bfloat16 *>(x), static_cast<const __nv_bfloat16 *>(w), uint32_t(m),
                         y, stream);
        }
        if (s == CERE_OK && cudaStreamSynchronize(stream) != cudaSuccess) s = CERE_ERR_INTERNAL;
        *static_cast<int64_t *>(use) = int64_t(gemm.pin_use(uint32_t(m)));
    }
    if (workspace != nullptr) cudaFree(workspace);
    cublasLtDestroy(handle);
    return s;
}
