// Inkling 2 Mini (inkling design note 3.6, slice S7): the vision tower's kernel-test entry point of libcere_ktest.so
// (model/inkling/vision.h), compared with the recorded reference tower by
// executor/tests/gpu/test_inkling_kernels_vision.py. Device pointers unless noted; enqueues on `stream`; returns a
// cere_status (0 = CERE_OK).
#include <cstdint>

#include "model/inkling/vision.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace iv = cere::model::inkling_vision;

namespace {

cublasLtHandle_t handle() {
    static const cublasLtHandle_t h = [] {
        cublasLtHandle_t created = nullptr;
        return cere::kernels::gemm_handle_create(&created) == CERE_OK ? created : nullptr;
    }();
    return h;
}

}  // namespace

// tensors: host array of iv::kTensorCount device pointers in iv::tensor_name order (names: cere_ktest_inkling_vision_
// tensor). grids: host int32 [count, 2] (h, w in patches). pixels BF16 [sum h * w, 1,200]; out BF16 [sum h * w / 4,
// 2,304]. mutant: bit 0 no class token, bit 1 RoPE axes swapped (fail-first), 0 in every other case. Plans (unpinned,
// max_m = this call's patches) and scratch are created per call; the call synchronizes `stream` before returning.
CERE_KTEST int cere_ktest_inkling_vision_encode(const void *const *tensors, const void *pixels, const int32_t *grids,
                                                int64_t count, int64_t mutant, void *out, cudaStream_t stream) {
    if (tensors == nullptr || grids == nullptr || count <= 0 || count > iv::kMaxImages || mutant < 0 || mutant > 3) {
        return CERE_ERR_ARGUMENT;
    }
    iv::Weights w;
    for (uint32_t i = 0; i < iv::kTensorCount; ++i) w.tensor[i] = static_cast<const iv::bf16 *>(tensors[i]);
    iv::Grid g[iv::kMaxImages];
    uint64_t patches = 0;
    for (int64_t i = 0; i < count; ++i) {
        if (grids[2 * i] <= 0 || grids[2 * i + 1] <= 0) return CERE_ERR_ARGUMENT;
        g[i] = iv::Grid{static_cast<uint32_t>(grids[2 * i]), static_cast<uint32_t>(grids[2 * i + 1])};
        patches += uint64_t(g[i].h) * g[i].w;
    }
    if (patches == 0 || patches % 4 != 0 || patches > iv::kMaxPatches || handle() == nullptr) return CERE_ERR_ARGUMENT;
    iv::Plans plans;
    cere_status s = iv::init_plans(&plans, handle(), static_cast<uint32_t>(patches));
    if (s != CERE_OK) return s;
    const uint64_t bytes = iv::scratch_bytes(patches, static_cast<uint32_t>(count));
    void *scratch = nullptr;
    if (cudaMallocAsync(&scratch, bytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    iv::Mutant m;
    m.no_class = (mutant & 1) != 0;
    m.swap_rope_axes = (mutant & 2) != 0;
    s = iv::encode(w, plans, static_cast<const iv::bf16 *>(pixels), g, static_cast<uint32_t>(count), scratch, bytes,
                   static_cast<iv::bf16 *>(out), stream, &m);
    const cudaError_t freed = cudaFreeAsync(scratch, stream);
    const cudaError_t synced = cudaStreamSynchronize(stream);
    if (s != CERE_OK) return s;
    return freed == cudaSuccess && synced == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

// Writes tensor i's checkpoint name (NUL-terminated, at most `cap` bytes) and shape (dims[2], unused 0); returns
// ndim, or CERE_ERR_ARGUMENT for i >= kTensorCount or a short buffer.
CERE_KTEST int cere_ktest_inkling_vision_tensor(int64_t i, char *name, int64_t cap, int64_t *dims) {
    if (i < 0 || i >= iv::kTensorCount || name == nullptr || dims == nullptr) return CERE_ERR_ARGUMENT;
    const std::string n = iv::tensor_name(static_cast<uint32_t>(i));
    if (static_cast<int64_t>(n.size()) + 1 > cap) return CERE_ERR_ARGUMENT;
    for (size_t c = 0; c <= n.size(); ++c) name[c] = c < n.size() ? n[c] : '\0';
    uint64_t shape[2] = {0, 0};
    const uint32_t ndim = iv::tensor_shape(static_cast<uint32_t>(i), shape);
    dims[0] = static_cast<int64_t>(shape[0]);
    dims[1] = static_cast<int64_t>(shape[1]);
    return static_cast<int>(ndim);
}
