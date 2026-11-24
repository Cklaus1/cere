// KV kernel-test entry points (executor/tests/gpu/test_kernels_kv_vision.py; m213 design note C9, D10, D11):
// device pointers, int64 sizes, the stream last; 0 = success (CERE_OK), CERE_ERR_ARGUMENT for arguments outside a
// kernel's contract, CERE_ERR_CAPACITY when scratch cannot be allocated, CERE_ERR_INTERNAL for a failed launch.
// Contracts in kernels/kv_vision.h; the F1 shapes (hidden 1152, 16 heads of 72, patch row 1536, pos_embed 48 x 48).
#include "kernels/kv_vision.h"
#include "ktest/k1_ktest.h"
#include "ktest/k2_ktest.h"

#include <cmath>

namespace kv = cere::kv;
using cere::ktest::status_of;
using kv::bf16;

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace {

constexpr int64_t kPatchRow = 1536;  // in_channels 3 * temporal_patch 2 * 16 * 16

// y[rows, n] = bias_act(x[rows, k] · w[n, k]^T) through an F32 K2 GEMM into scratch, as every encoder linear runs.
int linear(const void* x, const void* w, const void* bias, int64_t rows, int64_t n, int64_t k, kv::Act act, void* y,
           cudaStream_t stream) {
    float* acc = nullptr;
    if (cudaMallocAsync(reinterpret_cast<void**>(&acc), static_cast<size_t>(rows * n) * sizeof(float), stream) !=
        cudaSuccess)
        return CERE_ERR_CAPACITY;
    int status = cere::ktest::gemm(x, w, rows, n, k, acc, cere::kernels::GemmOut::kF32, stream);
    if (status == CERE_OK)
        status = status_of(kv::bias_act(acc, static_cast<const bf16*>(bias), nullptr, rows, static_cast<int>(n), act,
                                        static_cast<bf16*>(y), n, stream));
    const cudaError_t freed = cudaFreeAsync(acc, stream);
    if (status != CERE_OK) return status;
    return status_of(freed);
}

}  // namespace

// KV1: out [P, 1152] = bf16(pixels [P, 1536] · weight [1152, 1536]^T + bias) (Conv3d with stride = kernel).
CERE_KTEST int cere_ktest_vision_patch_embed(const void* pixels, const void* weight, const void* bias, int64_t P,
                                             void* out, cudaStream_t stream) {
    if (!pixels || !weight || !bias || !out || P <= 0 || P > 0x7fffffff) return CERE_ERR_ARGUMENT;
    return linear(pixels, weight, bias, P, kv::kHidden, kPatchRow, kv::Act::kBias, out, stream);
}

// KV2: out [t*h*w, 1152] = the interpolated pos_embed of one image (merge-window order), plus x when non-null.
CERE_KTEST int cere_ktest_vision_pos_embed(const void* table, int64_t num_position_embeddings, int64_t hidden,
                                           int64_t grid_t, int64_t grid_h, int64_t grid_w, int64_t align_corners,
                                           const void* x, void* out, cudaStream_t stream) {
    const int64_t side = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(num_position_embeddings))));
    if (hidden != kv::kHidden || side <= 0 || side * side != num_position_embeddings ||
        (align_corners != 0 && align_corners != 1))
        return CERE_ERR_ARGUMENT;
    return status_of(kv::pos_embed(static_cast<const bf16*>(table), static_cast<int>(side), grid_t, grid_h, grid_w,
                                   align_corners == 1, static_cast<const bf16*>(x), static_cast<bf16*>(out), stream));
}

// KV3: out [rows, dim] = nn.LayerNorm(dim, eps) of x [rows, dim] (contiguous).
CERE_KTEST int cere_ktest_vision_layernorm(const void* x, const void* weight, const void* bias, int64_t rows,
                                           int64_t dim, float eps, void* out, cudaStream_t stream) {
    if (!cere::ktest::fits_int(dim)) return CERE_ERR_ARGUMENT;
    return status_of(kv::layernorm(static_cast<const bf16*>(x), dim, static_cast<const bf16*>(weight),
                                   static_cast<const bf16*>(bias), eps, rows, static_cast<int>(dim),
                                   static_cast<bf16*>(out), dim, stream));
}

// KV3 contract check: kv::layernorm with the given row strides (a stride below dim must be refused, nothing run).
CERE_KTEST int cere_ktest_vision_layernorm_strided(const void* x, int64_t x_stride, const void* weight,
                                                   const void* bias, int64_t rows, int64_t dim, float eps, void* out,
                                                   int64_t y_stride, cudaStream_t stream) {
    if (!cere::ktest::fits_int(dim)) return CERE_ERR_ARGUMENT;
    return status_of(kv::layernorm(static_cast<const bf16*>(x), x_stride, static_cast<const bf16*>(weight),
                                   static_cast<const bf16*>(bias), eps, rows, static_cast<int>(dim),
                                   static_cast<bf16*>(out), y_stride, stream));
}

// The 2D vision RoPE of qkv with a table of n_pos rows built here.
int rope_with_table(const void* qkv, int64_t n_pos, int64_t grid_t, int64_t grid_h, int64_t grid_w, void* q_out,
                    void* k_out, cudaStream_t stream) {
    if (grid_h <= 0 || grid_w <= 0 || n_pos <= 0) return CERE_ERR_ARGUMENT;
    float* table = nullptr;
    if (cudaMallocAsync(reinterpret_cast<void**>(&table), static_cast<size_t>(n_pos) * 2 * kv::kRotaryHalf *
                                                              sizeof(float), stream) != cudaSuccess)
        return CERE_ERR_CAPACITY;
    int status = status_of(kv::rope_table(table, n_pos, stream));
    if (status == CERE_OK)
        status = status_of(kv::rope2d(static_cast<const bf16*>(qkv), kv::kQkvWidth, table, n_pos, grid_t, grid_h,
                                      grid_w, static_cast<bf16*>(q_out), static_cast<bf16*>(k_out), stream));
    const cudaError_t freed = cudaFreeAsync(table, stream);
    if (status != CERE_OK) return status;
    return status_of(freed);
}

// KV4: q_out / k_out [P, 1152] = the 2D vision RoPE of the q and k parts of qkv [P, 3456] for one image of grid
// (t, h, w), with the table built here (max(h, w) rows).
CERE_KTEST int cere_ktest_vision_rope(const void* qkv, int64_t grid_t, int64_t grid_h, int64_t grid_w, void* q_out,
                                      void* k_out, cudaStream_t stream) {
    return rope_with_table(qkv, grid_h > grid_w ? grid_h : grid_w, grid_t, grid_h, grid_w, q_out, k_out, stream);
}

// KV4 contract check: as cere_ktest_vision_rope with a table of table_rows rows (fewer than max(h, w) must be
// refused, nothing run).
CERE_KTEST int cere_ktest_vision_rope_rows(const void* qkv, int64_t grid_t, int64_t grid_h, int64_t grid_w,
                                           int64_t table_rows, void* q_out, void* k_out, cudaStream_t stream) {
    return rope_with_table(qkv, table_rows, grid_t, grid_h, grid_w, q_out, k_out, stream);
}

// KV5: out [P, 1152] = bidirectional 16 x 72 attention within each segment of seq_len rows.
CERE_KTEST int cere_ktest_vision_attention(const void* q, const void* k, int64_t qk_stride, const void* v,
                                           int64_t v_stride, int64_t P, int64_t seq_len, void* out,
                                           cudaStream_t stream) {
    return status_of(kv::attention(static_cast<const bf16*>(q), static_cast<const bf16*>(k), qk_stride,
                                   static_cast<const bf16*>(v), v_stride, P, seq_len, static_cast<bf16*>(out), stream));
}

// KV6: out [rows, n] = the epilogue of mode 0 (bias, + residual when non-null), 1 (bias + GELU tanh), 2 (bias +
// GELU erf) over acc [rows, n] F32. out may alias residual.
CERE_KTEST int cere_ktest_vision_bias_act(const void* acc, const void* bias, const void* residual, int64_t rows,
                                          int64_t n, int64_t mode, void* out, cudaStream_t stream) {
    if (!cere::ktest::fits_int(n) || mode < 0 || mode > 2) return CERE_ERR_ARGUMENT;
    return status_of(kv::bias_act(static_cast<const float*>(acc), static_cast<const bf16*>(bias),
                                  static_cast<const bf16*>(residual), rows, static_cast<int>(n),
                                  static_cast<kv::Act>(mode), static_cast<bf16*>(out), n, stream));
}

// KV7: the rows of embeds [image_rows, hidden] that fall in [chunk_start, chunk_start + n) over mixed [n, hidden].
CERE_KTEST int cere_ktest_vision_splice(const void* embeds, int64_t image_rows, int64_t token_offset,
                                        int64_t chunk_start, int64_t n, int64_t hidden, void* mixed,
                                        cudaStream_t stream) {
    return status_of(kv::splice(static_cast<const bf16*>(embeds), image_rows, token_offset, chunk_start, n, hidden,
                                static_cast<bf16*>(mixed), stream));
}
