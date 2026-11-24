// K1 kernel-test entry points (executor/tests/gpu/test_kernels_k1_primitives.py; prototypes in its docstring).
// Device pointers, int64 sizes, the stream last; 0 = success (CERE_OK), CERE_ERR_ARGUMENT for arguments outside a
// kernel's contract, CERE_ERR_INTERNAL for a failed launch. Contracts in kernels/k1_primitives.h.
#include "ktest/k1_ktest.h"

#include <limits>

#include "kernels/k1_primitives.h"

namespace cere::ktest {

int status_of(cudaError_t err) {
    if (err == cudaSuccess) return CERE_OK;
    if (err == cudaErrorInvalidValue) return CERE_ERR_ARGUMENT;
    return CERE_ERR_INTERNAL;
}

bool fits_int(int64_t v) { return v >= 0 && v <= std::numeric_limits<int32_t>::max(); }

}  // namespace cere::ktest

using cere::k1::bf16;
using cere::ktest::fits_int;
using cere::ktest::status_of;

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace {

// SmallFuse (m23 design note P5R): lets the stream's next kernel launch at once (griddepcontrol.launch_dependents),
// spins spin_ns on the global timer, then copies src to dst. A programmatic dependent launch behind it is resident
// during the spin; it reads dst's new bytes only if it waits for this grid (griddepcontrol.wait) before reading.
__global__ void pdl_late_copy_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int64_t bytes,
                                     uint64_t spin_ns) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    uint64_t start = 0;
    uint64_t now = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < spin_ns);
    const int64_t stride = int64_t(gridDim.x) * blockDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < bytes; i += stride) dst[i] = src[i];
}

}  // namespace

// dst = src (bytes), written spin_ns after the kernel starts; the stream's next kernel may launch before (see above).
CERE_KTEST int cere_ktest_pdl_late_copy(const void* src, void* dst, int64_t bytes, int64_t spin_ns,
                                        cudaStream_t stream) {
    if (!src || !dst || bytes <= 0 || spin_ns < 0) return CERE_ERR_ARGUMENT;
    pdl_late_copy_kernel<<<8, 256, 0, stream>>>(static_cast<const uint8_t*>(src), static_cast<uint8_t*>(dst), bytes,
                                                uint64_t(spin_ns));
    return status_of(cudaGetLastError());
}

CERE_KTEST int cere_ktest_embedding(const void* table, const void* ids, int64_t n_ids, int64_t vocab, int64_t hidden,
                                    void* out, cudaStream_t stream) {
    if (!fits_int(hidden)) return CERE_ERR_ARGUMENT;
    return status_of(cere::k1::embedding_gather(static_cast<const bf16*>(table), vocab, static_cast<int>(hidden),
                                                static_cast<const int32_t*>(ids), n_ids, static_cast<bf16*>(out),
                                                stream));
}

// A row of row_dim elements with one variance per group of group_size is row_dim / group_size kernel rows whose
// weights are the consecutive group_size slices of weight [row_dim]. weight_offset 1.0 = Gemma (1 + w), 0.0 = w.
CERE_KTEST int cere_ktest_rmsnorm(const void* x, const void* weight, int64_t n_rows, int64_t row_dim,
                                  int64_t group_size, float eps, float weight_offset, void* out, cudaStream_t stream) {
    if (n_rows < 0 || group_size <= 0 || !fits_int(group_size) || row_dim <= 0 || row_dim % group_size ||
        !fits_int(row_dim / group_size) || (weight_offset != 0.0f && weight_offset != 1.0f))
        return CERE_ERR_ARGUMENT;
    const int64_t groups = row_dim / group_size;
    return status_of(cere::k1::rmsnorm(static_cast<const bf16*>(x), group_size, static_cast<const bf16*>(weight),
                                       static_cast<int>(groups), weight_offset == 1.0f, eps, n_rows * groups,
                                       static_cast<int>(group_size), static_cast<bf16*>(out), group_size, stream));
}

CERE_KTEST int cere_ktest_rmsnorm_gated(const void* x, const void* z, const void* weight, int64_t n_rows,
                                        int64_t row_dim, float eps, void* out, cudaStream_t stream) {
    if (!fits_int(row_dim)) return CERE_ERR_ARGUMENT;
    return status_of(cere::k1::rmsnorm_gated_sigmoid(static_cast<const bf16*>(x), row_dim, static_cast<const bf16*>(z),
                                                     row_dim, static_cast<const bf16*>(weight), eps, n_rows,
                                                     static_cast<int>(row_dim), static_cast<bf16*>(out), row_dim,
                                                     stream));
}

CERE_KTEST int cere_ktest_rope_table(void* cos_sin, int64_t n_pos, int64_t rotary_dim, double rope_theta,
                                     cudaStream_t stream) {
    if (!fits_int(rotary_dim)) return CERE_ERR_ARGUMENT;
    return status_of(cere::k1::rope_table(static_cast<float*>(cos_sin), n_pos, static_cast<int>(rotary_dim),
                                          static_cast<float>(rope_theta), stream));
}

// Shared body of cere_ktest_rope / cere_ktest_rope3: exactly one of positions / positions3 is non-null.
static int rope_qk(const void* q, const void* k, const void* cos_sin, const int64_t* pos, const int64_t* pos3,
                   int64_t n_tokens, int64_t n_q_heads, int64_t n_kv_heads, int64_t head_dim, int64_t rotary_dim,
                   void* q_out, void* k_out, cudaStream_t stream) {
    if (!fits_int(n_q_heads) || !fits_int(n_kv_heads) || !fits_int(head_dim) || !fits_int(rotary_dim))
        return CERE_ERR_ARGUMENT;
    const float* table = static_cast<const float*>(cos_sin);
    const int64_t q_stride = n_q_heads * head_dim;
    const int64_t k_stride = n_kv_heads * head_dim;
    const int s = status_of(cere::k1::rope(static_cast<const bf16*>(q), q_stride, static_cast<bf16*>(q_out), q_stride,
                                           table, pos, pos3, 0, n_tokens, static_cast<int>(n_q_heads),
                                           static_cast<int>(head_dim), static_cast<int>(rotary_dim), stream));
    if (s != CERE_OK) return s;
    return status_of(cere::k1::rope(static_cast<const bf16*>(k), k_stride, static_cast<bf16*>(k_out), k_stride, table,
                                    pos, pos3, 0, n_tokens, static_cast<int>(n_kv_heads), static_cast<int>(head_dim),
                                    static_cast<int>(rotary_dim), stream));
}

CERE_KTEST int cere_ktest_rope(const void* q, const void* k, const void* cos_sin, const void* positions,
                               int64_t n_tokens, int64_t n_q_heads, int64_t n_kv_heads, int64_t head_dim,
                               int64_t rotary_dim, void* q_out, void* k_out, cudaStream_t stream) {
    if (!positions) return CERE_ERR_ARGUMENT;
    return rope_qk(q, k, cos_sin, static_cast<const int64_t*>(positions), nullptr, n_tokens, n_q_heads, n_kv_heads,
                   head_dim, rotary_dim, q_out, k_out, stream);
}

// M10 (m213 design note D11): cere_ktest_rope with positions3 [3][n_tokens] int64 (axes t, h, w) instead of
// positions; rotary_dim must be 64 (the interleaved mrope_section [11, 11, 10]).
CERE_KTEST int cere_ktest_rope3(const void* q, const void* k, const void* cos_sin, const void* positions3,
                                int64_t n_tokens, int64_t n_q_heads, int64_t n_kv_heads, int64_t head_dim,
                                int64_t rotary_dim, void* q_out, void* k_out, cudaStream_t stream) {
    if (!positions3) return CERE_ERR_ARGUMENT;
    return rope_qk(q, k, cos_sin, nullptr, static_cast<const int64_t*>(positions3), n_tokens, n_q_heads, n_kv_heads,
                   head_dim, rotary_dim, q_out, k_out, stream);
}

CERE_KTEST int cere_ktest_sigmoid_gate(const void* x, const void* gate, int64_t n_rows, int64_t row_dim, void* out,
                                       cudaStream_t stream) {
    if (!fits_int(row_dim)) return CERE_ERR_ARGUMENT;
    return status_of(cere::k1::sigmoid_mul(static_cast<const bf16*>(x), row_dim, static_cast<const bf16*>(gate),
                                           row_dim, n_rows, static_cast<int>(row_dim), static_cast<bf16*>(out),
                                           row_dim, stream));
}

CERE_KTEST int cere_ktest_top2(const void* logits, int64_t n_rows, int64_t vocab, void* ids, void* scores,
                               cudaStream_t stream) {
    return status_of(cere::k1::top2_split(static_cast<const bf16*>(logits), vocab, n_rows, vocab,
                                          static_cast<int32_t*>(ids), static_cast<float*>(scores), stream));
}

// Slots are trusted to lie in [0, n_slots) (the oracle draws them so); n_slots is not read on the device.
CERE_KTEST int cere_ktest_kv_store_fp8(const void* k, const void* v, const void* slots, int64_t n_tokens,
                                       int64_t row_dim, void* k_cache, void* v_cache, int64_t n_slots,
                                       cudaStream_t stream) {
    if (!slots || !fits_int(row_dim) || n_slots <= 0 || n_tokens > n_slots) return CERE_ERR_ARGUMENT;
    return status_of(cere::k1::kv_store_fp8(static_cast<const bf16*>(k), row_dim, static_cast<const bf16*>(v), row_dim,
                                            n_tokens, static_cast<int>(row_dim), static_cast<uint8_t*>(k_cache),
                                            static_cast<uint8_t*>(v_cache), static_cast<const int64_t*>(slots), 0,
                                            stream));
}

// ranges: host int64 [count, 6] rows (dst, dst_pitch, src, src_pitch, width, rows), device pointers as integers.
CERE_KTEST int cere_ktest_copy_segments(const void* ranges, int64_t count, cudaStream_t stream) {
    if (!ranges || count < 1 || count > cere::k1::kMaxCopySegments) return CERE_ERR_ARGUMENT;
    const int64_t* r = static_cast<const int64_t*>(ranges);
    cere::k1::CopySegment segments[cere::k1::kMaxCopySegments];
    for (int64_t i = 0; i < count; ++i, r += 6) {
        if (r[1] < 0 || r[3] < 0 || r[4] < 0 || r[5] < 0) return CERE_ERR_ARGUMENT;
        segments[i] = {reinterpret_cast<void*>(r[0]), uint64_t(r[1]), reinterpret_cast<const void*>(r[2]),
                       uint64_t(r[3]), uint64_t(r[4]), uint64_t(r[5])};
    }
    return status_of(cere::k1::copy_segments(segments, static_cast<int>(count), stream));
}
