// K5 kernel-test entry points (libcere_ktest.so), the contract of executor/tests/gpu/test_kernels_k5_gdn.py: device
// pointers to contiguous row-major buffers, int64_t counts, float scalars, the stream last; 0 or a cere_status.
// The layer entry points create their GEMM plans and workspace per call and synchronize the stream before freeing
// the workspace (the executor preallocates both at INSTANTIATE instead). Operations are CERE_OP_PREFILL / DECODE.
#include "cere_executor.h"
#include "kernels/k1_primitives.h"
#include "kernels/k2_gemm.h"
#include "kernels/k5_gdn.h"
#include "kernels/k5_gdn_chunk.h"
#include "kernels/k5_gdn_layer.h"
#include "ktest/k2_ktest.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

using bf16 = __nv_bfloat16;
constexpr int64_t kHidden = 2560;  // A4 hidden_size

cere_status status_of(cudaError_t err) {
    if (err == cudaSuccess) {
        return CERE_OK;
    }
    return err == cudaErrorInvalidValue ? CERE_ERR_ARGUMENT : CERE_ERR_INTERNAL;
}

// Runs `run(workspace, bytes)` with a fresh workspace of `bytes`, synchronizes `stream` and frees it.
template <typename Run>
cere_status with_workspace(size_t bytes, cudaStream_t stream, Run run) {
    void *workspace = nullptr;
    if (bytes == 0) {
        return CERE_ERR_ARGUMENT;
    }
    if (cudaMalloc(&workspace, bytes) != cudaSuccess) {
        return CERE_ERR_CAPACITY;
    }
    cere_status st = run(workspace, bytes);
    const cudaError_t sync = cudaStreamSynchronize(stream);
    cudaFree(workspace);
    if (st == CERE_OK && sync != cudaSuccess) {
        st = CERE_ERR_INTERNAL;
    }
    return st;
}

cere_status layer(const void *hidden, const void *w_qkv, const void *w_z, const void *w_b, const void *w_a,
                  const void *conv_weight, const void *A_log, const void *dt_bias, const void *norm_weight,
                  const void *w_out, void *conv_state, void *ssm_state, void *out, int64_t tokens, uint32_t operation,
                  cudaStream_t stream) {
    if (tokens < 1 || tokens > cere::k5::kChunkMaxTokens) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = cere::ktest::gemm_handle();
    if (handle == nullptr) {
        return CERE_ERR_INTERNAL;
    }
    using cere::kernels::GemmOut;
    using cere::kernels::GemmPlan;
    const auto m = static_cast<uint32_t>(tokens);
    GemmPlan qkv, z, ba, o;
    cere_status st = qkv.init(handle, cere::k5::kConvDim, kHidden, m, GemmOut::kBF16);
    if (st == CERE_OK) {
        st = z.init(handle, cere::k5::kValueDim, kHidden, m, GemmOut::kBF16);
    }
    if (st == CERE_OK) {
        st = ba.init(handle, cere::k5::kValueHeads, kHidden, m, GemmOut::kBF16);
    }
    if (st == CERE_OK) {
        st = o.init(handle, kHidden, cere::k5::kValueDim, m, GemmOut::kBF16);
    }
    if (st != CERE_OK) {
        return st;
    }
    const cere::k5::LayerGemms gemms{&qkv, &z, &ba, &o};
    const cere::k5::LayerWeights w{
        static_cast<const bf16 *>(w_qkv),       static_cast<const bf16 *>(w_z),
        static_cast<const bf16 *>(w_b),         static_cast<const bf16 *>(w_a),
        static_cast<const bf16 *>(conv_weight), static_cast<const float *>(A_log),
        static_cast<const bf16 *>(dt_bias),     static_cast<const bf16 *>(norm_weight),
        static_cast<const bf16 *>(w_out),
    };
    const cere::k5::LayerState state{static_cast<bf16 *>(conv_state), static_cast<bf16 *>(ssm_state)};
    return with_workspace(cere::k5::layer_workspace_bytes(tokens), stream, [&](void *workspace, size_t bytes) {
        return cere::k5::layer_forward(gemms, w, state, operation, static_cast<const bf16 *>(hidden), tokens, kHidden,
                                       static_cast<bf16 *>(out), workspace, bytes, stream);
    });
}

}  // namespace

#pragma GCC visibility push(default)
extern "C" {

int cere_ktest_gdn_conv1d(const void *x, const void *weight, void *conv_state, void *y, int64_t tokens, int64_t dim,
                          cudaStream_t stream) {
    return status_of(cere::k5::conv1d_silu(static_cast<const bf16 *>(x), dim, static_cast<const bf16 *>(weight),
                                           static_cast<bf16 *>(conv_state), static_cast<bf16 *>(y), dim, tokens,
                                           dim, stream));
}

int cere_ktest_gdn_gating(const void *a, const void *b, const void *A_log, const void *dt_bias, void *g, void *beta,
                          int64_t tokens, int64_t heads, cudaStream_t stream) {
    return status_of(cere::k5::gating(static_cast<const bf16 *>(a), heads, static_cast<const bf16 *>(b), heads,
                                      static_cast<const float *>(A_log), static_cast<const bf16 *>(dt_bias),
                                      static_cast<float *>(g), static_cast<float *>(beta), tokens, heads, stream));
}

int cere_ktest_gdn_l2norm(const void *x, void *y, int64_t rows, int64_t dim, float eps, cudaStream_t stream) {
    return status_of(
        cere::k5::l2norm(static_cast<const bf16 *>(x), static_cast<bf16 *>(y), rows, dim, eps, stream));
}

// int32 [3] = {kConvVariant, kGatingVariant, kL2normVariant}: the lowering variants this library was built with.
int cere_ktest_gdn_variants(void *out, cudaStream_t stream) {
    static const int32_t variants[3] = {cere::k5::kConvVariant, cere::k5::kGatingVariant, cere::k5::kL2normVariant};
    return status_of(cudaMemcpyAsync(out, variants, sizeof(variants), cudaMemcpyHostToDevice, stream));
}

int cere_ktest_gdn_gated_rmsnorm(const void *x, const void *z, const void *weight, void *y, int64_t rows,
                                 int64_t dim, float eps, cudaStream_t stream) {
    if (rows < 1 || dim < 1 || dim > INT32_MAX) {
        return CERE_ERR_ARGUMENT;
    }
    return status_of(cere::k1::rmsnorm_gated_sigmoid(static_cast<const bf16 *>(x), dim, static_cast<const bf16 *>(z),
                                                     dim, static_cast<const bf16 *>(weight), eps, rows,
                                                     static_cast<int>(dim), static_cast<bf16 *>(y), dim, stream));
}

int cere_ktest_gdn_layer_core(const void *mixed_qkv, const void *z, const void *b, const void *a,
                              const void *conv_weight, const void *A_log, const void *dt_bias, const void *norm_weight,
                              void *conv_state, void *ssm_state, void *y, int64_t tokens, int64_t operation,
                              cudaStream_t stream) {
    if (operation < 0 || operation > UINT32_MAX) {
        return CERE_ERR_ARGUMENT;
    }
    cere::k5::LayerWeights w{};
    w.conv1d = static_cast<const bf16 *>(conv_weight);
    w.A_log = static_cast<const float *>(A_log);
    w.dt_bias = static_cast<const bf16 *>(dt_bias);
    w.norm = static_cast<const bf16 *>(norm_weight);
    const cere::k5::LayerState state{static_cast<bf16 *>(conv_state), static_cast<bf16 *>(ssm_state)};
    return with_workspace(cere::k5::core_workspace_bytes(tokens), stream, [&](void *workspace, size_t bytes) {
        return cere::k5::layer_core(w, state, static_cast<uint32_t>(operation), static_cast<const bf16 *>(mixed_qkv),
                                    static_cast<const bf16 *>(z), static_cast<const bf16 *>(b),
                                    static_cast<const bf16 *>(a), tokens, static_cast<bf16 *>(y), workspace, bytes,
                                    stream);
    });
}

// k5::chain_core on `rows` (1..16) rows of `chains` chains back to back, chain c rows chain_rows[c] (host int64
// [chains]), on its own state conv_states + c * [10240, 3], ssm_states + c * [48, 128, 128] (bf16, contiguous);
// inputs as cere_ktest_gdn_layer_core's at tokens = rows; conv_saves bf16 [rows, 10240, 3] and ssm_saves bf16
// [rows, 48, 128, 128] (row-indexed, contiguous; null when no chain has 2 rows) take the state after each row but its
// chain's last (rows16 design note R16-B).
int cere_ktest_gdn_chains_core(const void *mixed_qkv, const void *z, const void *b, const void *a,
                               const void *conv_weight, const void *A_log, const void *dt_bias,
                               const void *norm_weight, void *conv_states, void *ssm_states, void *conv_saves,
                               void *ssm_saves, void *y, int64_t rows, const void *chain_rows, int64_t chains,
                               cudaStream_t stream) {
    if (rows < 1 || rows > cere::k5::kChainMaxRows || chains < 1 || chains > rows || chain_rows == nullptr ||
        conv_states == nullptr || ssm_states == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    cere::k5::LayerWeights w{};
    w.conv1d = static_cast<const bf16 *>(conv_weight);
    w.A_log = static_cast<const float *>(A_log);
    w.dt_bias = static_cast<const bf16 *>(dt_bias);
    w.norm = static_cast<const bf16 *>(norm_weight);
    constexpr int64_t kConvElems = int64_t(cere::k5::kConvDim) * 3;
    constexpr int64_t kSsmElems = int64_t(cere::k5::kValueHeads) * 128 * 128;
    const int64_t *counts = static_cast<const int64_t *>(chain_rows);
    cere::k5::Chains set;
    set.count = chains;
    int64_t next = 0;
    for (int64_t c = 0; c < chains; ++c) {
        if (counts[c] < 1 || counts[c] > rows - next) {
            return CERE_ERR_ARGUMENT;
        }
        set.rows[c] = counts[c];
        set.states[c] = {static_cast<bf16 *>(conv_states) + c * kConvElems,
                         static_cast<bf16 *>(ssm_states) + c * kSsmElems};
        next += counts[c];
        for (int64_t r = next - counts[c]; r + 1 < next; ++r) {
            if (conv_saves == nullptr || ssm_saves == nullptr) {
                return CERE_ERR_ARGUMENT;
            }
            set.saves[r] = {static_cast<bf16 *>(conv_saves) + r * kConvElems,
                            static_cast<bf16 *>(ssm_saves) + r * kSsmElems};
        }
    }
    return with_workspace(cere::k5::chain_core_workspace_bytes(rows), stream, [&](void *workspace, size_t bytes) {
        return cere::k5::chain_core(w, set, static_cast<const bf16 *>(mixed_qkv), static_cast<const bf16 *>(z),
                                    static_cast<const bf16 *>(b), static_cast<const bf16 *>(a), rows,
                                    static_cast<bf16 *>(y), workspace, bytes, stream);
    });
}

// k5::chain_core on one chain of `rows` (1..16) rows: inputs as cere_ktest_gdn_layer_core's at tokens = rows;
// conv_saves bf16 [rows - 1, 10240, 3] and ssm_saves bf16 [rows - 1, 48, 128, 128] (contiguous, null when rows is 1)
// take the state after each row but the last.
int cere_ktest_gdn_chain_core(const void *mixed_qkv, const void *z, const void *b, const void *a,
                              const void *conv_weight, const void *A_log, const void *dt_bias, const void *norm_weight,
                              void *conv_state, void *ssm_state, void *conv_saves, void *ssm_saves, void *y,
                              int64_t rows, cudaStream_t stream) {
    const int64_t chain_rows[1] = {rows};
    return cere_ktest_gdn_chains_core(mixed_qkv, z, b, a, conv_weight, A_log, dt_bias, norm_weight, conv_state,
                                      ssm_state, conv_saves, ssm_saves, y, rows, chain_rows, 1, stream);
}

int cere_ktest_gdn_layer_prefill(const void *hidden, const void *w_qkv, const void *w_z, const void *w_b,
                                 const void *w_a, const void *conv_weight, const void *A_log, const void *dt_bias,
                                 const void *norm_weight, const void *w_out, void *conv_state, void *ssm_state,
                                 void *out, int64_t tokens, cudaStream_t stream) {
    return layer(hidden, w_qkv, w_z, w_b, w_a, conv_weight, A_log, dt_bias, norm_weight, w_out, conv_state,
                 ssm_state, out, tokens, CERE_OP_PREFILL, stream);
}

int cere_ktest_gdn_layer_decode(const void *hidden, const void *w_qkv, const void *w_z, const void *w_b,
                                const void *w_a, const void *conv_weight, const void *A_log, const void *dt_bias,
                                const void *norm_weight, const void *w_out, void *conv_state, void *ssm_state,
                                void *out, cudaStream_t stream) {
    return layer(hidden, w_qkv, w_z, w_b, w_a, conv_weight, A_log, dt_bias, norm_weight, w_out, conv_state,
                 ssm_state, out, 1, CERE_OP_DECODE, stream);
}

}  // extern "C"
#pragma GCC visibility pop
