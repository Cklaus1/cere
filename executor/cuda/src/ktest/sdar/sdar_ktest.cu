// SelfTune ST4 S3c: kernel-test entry points of family sdar's kernels (executor/tests/gpu/test_sdar_kernels_ops.py;
// prototypes in its docstring). Device pointers, int64 sizes, float scalars, the stream last; 0 = success (CERE_OK),
// CERE_ERR_ARGUMENT for arguments outside a kernel's contract, CERE_ERR_INTERNAL for a failed launch. Contracts in
// kernels/sdar/sdar_kernels.h; the fail-first mutants (attention's kCausal mask, top_tie without the BF16 rounding)
// in kernels/sdar/sdar_mutants.h. Compiled only into a CERE_MODEL=<sdar descriptor> library.
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

#include "cere_executor.h"
#include "kernels/sdar/sdar_kernels.h"
#include "kernels/sdar/sdar_mutants.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace {

using cere::sdar::bf16;
using cere::sdar::TopTie;

int status_of(cudaError_t err) {
    if (err == cudaSuccess) return CERE_OK;
    if (err == cudaErrorInvalidValue) return CERE_ERR_ARGUMENT;
    return CERE_ERR_INTERNAL;
}

// 0 <= v <= INT32_MAX: the int arguments of the kernel API.
bool fits_int(int64_t v) { return v >= 0 && v <= std::numeric_limits<int32_t>::max(); }

const bf16 *in(const void *p) { return static_cast<const bf16 *>(p); }
bf16 *out(void *p) { return static_cast<bf16 *>(p); }

}  // namespace

CERE_KTEST int cere_ktest_sdar_embed(const void *table, const void *ids, int64_t n, int64_t hidden, void *dst,
                                     cudaStream_t stream) {
    if (!fits_int(hidden)) return CERE_ERR_ARGUMENT;
    return status_of(cere::sdar::embed(in(table), static_cast<const uint32_t *>(ids), n, static_cast<int>(hidden),
                                       out(dst), stream));
}

CERE_KTEST int cere_ktest_sdar_rmsnorm(const void *x, int64_t x_stride, const void *w, float eps, int64_t rows,
                                       int64_t dim, void *y, int64_t y_stride, cudaStream_t stream) {
    if (!fits_int(dim)) return CERE_ERR_ARGUMENT;
    return status_of(cere::sdar::rmsnorm(in(x), x_stride, in(w), eps, rows, static_cast<int>(dim), out(y), y_stride,
                                         stream));
}

CERE_KTEST int cere_ktest_sdar_rope(void *x, int64_t stride, int64_t rows, int64_t heads, int64_t pos0, float theta,
                                    cudaStream_t stream) {
    if (!fits_int(heads)) return CERE_ERR_ARGUMENT;
    return status_of(cere::sdar::rope(out(x), stride, rows, static_cast<int>(heads), pos0, theta, stream));
}

CERE_KTEST int cere_ktest_sdar_kv_store(const void *k, int64_t k_stride, const void *v, int64_t v_stride, int64_t n,
                                        int64_t kv_heads, void *k_cache, void *v_cache, int64_t pos,
                                        cudaStream_t stream) {
    if (!fits_int(kv_heads)) return CERE_ERR_ARGUMENT;
    return status_of(cere::sdar::kv_store(in(k), k_stride, in(v), v_stride, n, static_cast<int>(kv_heads),
                                          out(k_cache), out(v_cache), pos, stream));
}

// mask: 0 = Mask::kBlockCausal (the model's), 1 = Mask::kCausal (the fail-first mutant).
CERE_KTEST int cere_ktest_sdar_attention(const void *q, int64_t q_stride, const void *k_cache, const void *v_cache,
                                         int64_t past, const void *new_k, int64_t new_k_stride, const void *new_v,
                                         int64_t new_v_stride, int64_t n, int64_t heads, int64_t kv_heads, void *dst,
                                         int64_t out_stride, int64_t mask, cudaStream_t stream) {
    if (!fits_int(heads) || !fits_int(kv_heads) || (mask != 0 && mask != 1)) return CERE_ERR_ARGUMENT;
    const cere::sdar::Mask m = mask == 0 ? cere::sdar::Mask::kBlockCausal : cere::sdar::Mask::kCausal;
    return status_of(cere::sdar::attention(in(q), q_stride, in(k_cache), in(v_cache), past, in(new_k), new_k_stride,
                                           in(new_v), new_v_stride, n, static_cast<int>(heads),
                                           static_cast<int>(kv_heads), out(dst), out_stride, m, stream));
}

CERE_KTEST int cere_ktest_sdar_silu_mul(const void *gate_up, int64_t rows, int64_t inter, void *dst,
                                        cudaStream_t stream) {
    if (!fits_int(inter)) return CERE_ERR_ARGUMENT;
    return status_of(cere::sdar::silu_mul(in(gate_up), rows, static_cast<int>(inter), out(dst), stream));
}

// x [n] in place, y [n].
CERE_KTEST int cere_ktest_sdar_add(void *x, const void *y, int64_t n, cudaStream_t stream) {
    return status_of(cere::sdar::add(out(x), in(y), n, stream));
}

// out: TopTie[rows] (20 bytes each: id, ties, runner_up as uint32, logit, runner_up_logit as float).
// fp32 != 0: the condition-1 mutant (no BF16 rounding before the comparisons).
CERE_KTEST int cere_ktest_sdar_top_tie(const void *logits, int64_t stride, int64_t rows, int64_t vocab, void *dst,
                                       int64_t fp32, cudaStream_t stream) {
    static_assert(sizeof(TopTie) == 20, "TopTie is 5 packed 4-byte fields (the suite reads it as int32 [rows, 5])");
    if (fp32 != 0 && fp32 != 1) return CERE_ERR_ARGUMENT;
    const float *l = static_cast<const float *>(logits);
    TopTie *t = static_cast<TopTie *>(dst);
    return status_of(fp32 ? cere::sdar::mutant::top_tie_fp32(l, stride, rows, vocab, t, stream)
                          : cere::sdar::top_tie(l, stride, rows, vocab, t, stream));
}

// index: int32 [1] (device).
CERE_KTEST int cere_ktest_sdar_unmask_select(const void *top, const void *masked, int64_t rows, void *index,
                                             cudaStream_t stream) {
    if (!fits_int(rows)) return CERE_ERR_ARGUMENT;
    return status_of(cere::sdar::unmask_select(static_cast<const TopTie *>(top), static_cast<const uint8_t *>(masked),
                                               static_cast<int>(rows), static_cast<int32_t *>(index), stream));
}
