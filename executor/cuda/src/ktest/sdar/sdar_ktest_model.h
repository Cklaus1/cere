// SelfTune ST4: what the whole-model entry points of family sdar share (sdar_model_ktest.cpp, the S3c replay;
// sdar_generate_ktest.cpp, the S4a free run): the model's weights bound on the caller's tensors, its GEMM plans, a
// workspace and one session's KV cache. Compiled only into a CERE_MODEL=<sdar descriptor> library.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <cublasLt.h>
#include <cuda_runtime.h>

#include "cere_executor.h"
#include "model/sdar/sdar_model.h"

namespace cere::sdar::ktest {

struct Device {
    void *ptr = nullptr;
    Device() = default;
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
    ~Device() {
        if (ptr != nullptr) cudaFree(ptr);
    }
};

struct Handle {
    cublasLtHandle_t h = nullptr;
    Handle() = default;
    Handle(const Handle &) = delete;
    Handle &operator=(const Handle &) = delete;
    ~Handle() {
        if (h != nullptr) cublasLtDestroy(h);
    }
};

// The model of one session of `context` tokens; `load` fills it in place.
struct Model {
    Weights weights;
    Handle handle;
    Gemms gemms;
    Device ws_mem;
    Device kv_mem;
    Workspace ws;
    bf16 *k_cache = nullptr;
    bf16 *v_cache = nullptr;
    uint64_t rows = 0;  // kv_rows(context)
};

// tensors: `count` device pointers (uint64); names: their checkpoint names, each NUL-terminated, back to back;
// dims: int64 [count, 2] (a 1-D tensor's second dim 0). Every mlp.up_proj must start where its gate_proj ends.
// sdar_gemm: the CERE_LEVER_SDAR_GEMM_* the Gemms run (SdarOptIn, O320; Gemms::init refuses any other value).
inline cere_status load(const void *tensors, const void *names, const void *dims, int64_t count, uint64_t context,
                        Model *m, uint32_t sdar_gemm = CERE_LEVER_SDAR_GEMM_IDENTICAL) {
    const auto *ptrs = static_cast<const uint64_t *>(tensors);
    const auto *shape = static_cast<const int64_t *>(dims);
    // The slots of the caller's tensors, offsets from the lowest pointer (bind_weights reads arena + offset).
    const uint64_t base = *std::min_element(ptrs, ptrs + count);
    std::vector<TensorSlot> slots(static_cast<size_t>(count));
    const char *name = static_cast<const char *>(names);
    for (int64_t i = 0; i < count; ++i) {
        TensorSlot &s = slots[size_t(i)];
        s.name = name;
        name += s.name.size() + 1;
        s.dtype = CERE_DTYPE_BF16;
        s.ndim = shape[2 * i + 1] == 0 ? 1 : 2;
        s.shape[0] = uint64_t(shape[2 * i]);
        s.shape[1] = uint64_t(shape[2 * i + 1]);
        s.bytes = s.shape[0] * (s.ndim == 2 ? s.shape[1] : 1) * sizeof(bf16);
        s.offset = ptrs[i] - base;
    }
    if (const cere_status st = bind_weights(slots, reinterpret_cast<const uint8_t *>(base), &m->weights);
        st != CERE_OK) {
        return st;
    }
    if (cublasLtCreate(&m->handle.h) != CUBLAS_STATUS_SUCCESS) return CERE_ERR_INTERNAL;
    if (const cere_status st = m->gemms.init(m->handle.h, sdar_gemm); st != CERE_OK) return st;
    if (cudaMalloc(&m->ws_mem.ptr, workspace_bytes()) != cudaSuccess ||
        cudaMalloc(&m->kv_mem.ptr, kv_bytes(context)) != cudaSuccess) {
        return CERE_ERR_CAPACITY;
    }
    m->ws = carve_workspace(static_cast<uint8_t *>(m->ws_mem.ptr));
    m->rows = kv_rows(context);
    m->k_cache = static_cast<bf16 *>(m->kv_mem.ptr);
    m->v_cache = m->k_cache + kv_bytes(context) / sizeof(bf16) / 2;
    return CERE_OK;
}

}  // namespace cere::sdar::ktest
