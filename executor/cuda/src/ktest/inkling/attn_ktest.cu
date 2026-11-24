// Inkling 2 Mini S2: kernel-test entry points of the attention kernels (executor/cuda/src/kernels/inkling/attn.h) and
// their CPU reference (executor/ref/inkling/attn_ref.h), for executor/tests/gpu/test_inkling_kernels_attn.py
// (prototypes in its docstring). GPU entries: device pointers; CPU (`_ref_`) entries: host pointers. Integers int64,
// the stream last (ignored by the CPU entries); 0 = success, CERE_ERR_ARGUMENT for arguments outside the contract,
// CERE_ERR_INTERNAL for a failed launch.
//
// A sequence table is int64 [n_seqs, 6] = (row, rows, pos0, k_cache, v_cache, slots) per sequence, the layout of
// attn::Seq (device memory for the GPU entries; host memory with host cache pointers for the CPU entry).
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "cere_executor.h"
#include "kernels/inkling/attn.h"
#include "inkling/attn_ref.h"

#define CERE_KTEST extern "C" __attribute__((visibility("default")))

namespace {

namespace A = cere::inkling::attn;
namespace R = ref::attn;
using A::bf16;

static_assert(sizeof(A::Seq) == 6 * sizeof(int64_t), "Seq is the int64 [6] row of a ktest sequence table");

int status_of(cudaError_t err) {
    if (err == cudaSuccess) return CERE_OK;
    if (err == cudaErrorInvalidValue) return CERE_ERR_ARGUMENT;
    return CERE_ERR_INTERNAL;
}

bool fits_int(int64_t v) { return v >= 0 && v <= INT32_MAX; }

const bf16 *in(const void *p) { return static_cast<const bf16 *>(p); }
bf16 *out(void *p) { return static_cast<bf16 *>(p); }
const uint16_t *hin(const void *p) { return static_cast<const uint16_t *>(p); }
uint16_t *hout(void *p) { return static_cast<uint16_t *>(p); }

A::AttnArgs attn_args(int64_t kind, const void *seqs, int64_t n_seqs, int64_t max_rows, int64_t max_pos,
                      const void *q, int64_t q_stride, const void *r, int64_t r_stride, const void *rel_proj,
                      const void *k, int64_t k_stride, const void *v, int64_t v_stride, const void *sink,
                      int64_t sink_stride, const void *gate, int64_t gate_stride, void *o, int64_t out_stride,
                      int64_t mutant) {
    A::AttnArgs a;
    a.kind = int(kind);
    a.seqs = static_cast<const A::Seq *>(seqs);
    a.n_seqs = int(n_seqs);
    a.max_rows = max_rows;
    a.max_pos = max_pos;
    a.q = in(q), a.q_stride = q_stride;
    a.r = in(r), a.r_stride = r_stride;
    a.rel_proj = in(rel_proj);
    a.k = in(k), a.k_stride = k_stride;
    a.v = in(v), a.v_stride = v_stride;
    a.sink = in(sink), a.sink_stride = sink_stride;
    a.gate = in(gate), a.gate_stride = gate_stride;
    a.out = out(o), a.out_stride = out_stride;
    a.mutant = int(mutant);
    return a;
}

// Runs f(i) for i in [0, n) on up to hardware_concurrency threads (each index once; f writes disjoint outputs).
template <typename F> void parallel_for(int64_t n, F f) {
    const int64_t workers = std::min<int64_t>(n, std::max(1u, std::thread::hardware_concurrency()));
    std::atomic<int64_t> next{0};
    std::vector<std::thread> pool;
    for (int64_t w = 0; w < workers; ++w)
        pool.emplace_back([&] {
            for (int64_t i = next++; i < n; i = next++) f(i);
        });
    for (auto &t : pool) t.join();
}

}  // namespace

// ---- GPU ---------------------------------------------------------------------------------------------------------

// attn::post with contiguous rows (null inputs skip their part).
CERE_KTEST int cere_ktest_inkling_attn_post(int64_t kind, int64_t rows, const void *positions, const void *q_in,
                                            const void *q_scale, void *q_out, const void *r_in, const void *rel_scale,
                                            void *r_out, const void *k_in, void *k_out, const void *v_in, void *v_out,
                                            int64_t mutant, cudaStream_t stream) {
    A::PostArgs a;
    a.kind = int(kind);
    a.rows = rows;
    a.positions = static_cast<const int64_t *>(positions);
    a.q_in = in(q_in), a.q_scale = in(q_scale), a.q_out = out(q_out);
    a.r_in = in(r_in), a.rel_scale = in(rel_scale), a.r_out = out(r_out);
    a.k_in = in(k_in), a.k_out = out(k_out);
    a.v_in = in(v_in), a.v_out = out(v_out);
    a.mutant = int(mutant);
    return status_of(A::post(a, stream));
}

CERE_KTEST int cere_ktest_inkling_attn_prefill(int64_t kind, const void *seqs, int64_t n_seqs, int64_t max_rows,
                                               const void *q, int64_t q_stride, const void *r, int64_t r_stride,
                                               const void *rel_proj, const void *k, int64_t k_stride, const void *v,
                                               int64_t v_stride, const void *sink, int64_t sink_stride,
                                               const void *gate, int64_t gate_stride, void *o, int64_t out_stride,
                                               int64_t mutant, cudaStream_t stream) {
    if (!fits_int(n_seqs)) return CERE_ERR_ARGUMENT;
    return status_of(A::prefill(attn_args(kind, seqs, n_seqs, max_rows, 0, q, q_stride, r, r_stride, rel_proj, k,
                                          k_stride, v, v_stride, sink, sink_stride, gate, gate_stride, o, out_stride,
                                          mutant),
                                stream));
}

CERE_KTEST int64_t cere_ktest_inkling_attn_decode_workspace_bytes(int64_t total_rows) {
    return int64_t(A::decode_workspace_bytes(total_rows));
}

CERE_KTEST int cere_ktest_inkling_attn_decode(int64_t kind, const void *seqs, int64_t n_seqs, int64_t max_rows,
                                              int64_t max_pos, int64_t total_rows, const void *q, int64_t q_stride,
                                              const void *r, int64_t r_stride, const void *rel_proj, const void *k,
                                              int64_t k_stride, const void *v, int64_t v_stride, const void *sink,
                                              int64_t sink_stride, const void *gate, int64_t gate_stride, void *o,
                                              int64_t out_stride, void *workspace, int64_t workspace_bytes,
                                              int64_t mutant, cudaStream_t stream) {
    if (!fits_int(n_seqs) || workspace_bytes < 0) return CERE_ERR_ARGUMENT;
    return status_of(A::decode(attn_args(kind, seqs, n_seqs, max_rows, max_pos, q, q_stride, r, r_stride, rel_proj, k,
                                         k_stride, v, v_stride, sink, sink_stride, gate, gate_stride, o, out_stride,
                                         mutant),
                               total_rows, workspace, size_t(workspace_bytes), stream));
}

// `iters` back-to-back decode calls (contiguous rows) between two events; *ms (host float [1]) = mean milliseconds per
// call. One warm-up call first.
CERE_KTEST int cere_ktest_inkling_attn_decode_time(int64_t kind, const void *seqs, int64_t n_seqs, int64_t max_rows,
                                                   int64_t max_pos, int64_t total_rows, const void *q, const void *r,
                                                   const void *rel_proj, const void *k, const void *v,
                                                   const void *sink, const void *gate, void *o, void *workspace,
                                                   int64_t workspace_bytes, int64_t iters, void *ms,
                                                   cudaStream_t stream) {
    if (!fits_int(n_seqs) || workspace_bytes < 0 || iters < 1 || !ms) return CERE_ERR_ARGUMENT;
    const A::AttnArgs a = attn_args(kind, seqs, n_seqs, max_rows, max_pos, q, A::kQRow, r, A::kRelRow, rel_proj, k,
                                    A::kKvRow, v, A::kKvRow, sink, A::kHeads, gate, A::kQRow, o, A::kQRow, 0);
    cudaError_t err = A::decode(a, total_rows, workspace, size_t(workspace_bytes), stream);
    if (err != cudaSuccess) return status_of(err);
    cudaEvent_t e0, e1;
    if (cudaEventCreate(&e0) != cudaSuccess) return CERE_ERR_INTERNAL;
    if (cudaEventCreate(&e1) != cudaSuccess) {
        cudaEventDestroy(e0);
        return CERE_ERR_INTERNAL;
    }
    err = cudaEventRecord(e0, stream);
    for (int64_t i = 0; i < iters && err == cudaSuccess; ++i)
        err = A::decode(a, total_rows, workspace, size_t(workspace_bytes), stream);
    if (err == cudaSuccess) err = cudaEventRecord(e1, stream);
    if (err == cudaSuccess) err = cudaEventSynchronize(e1);
    float total = 0.0f;
    if (err == cudaSuccess) err = cudaEventElapsedTime(&total, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (err != cudaSuccess) return status_of(err);
    *static_cast<float *>(ms) = total / float(iters);
    return CERE_OK;
}

// `iters` back-to-back prefill calls (contiguous rows) between two events; *ms (host float [1]) = mean milliseconds per
// call. One warm-up call first.
CERE_KTEST int cere_ktest_inkling_attn_prefill_time(int64_t kind, const void *seqs, int64_t n_seqs, int64_t max_rows,
                                                    const void *q, const void *r, const void *rel_proj, const void *k,
                                                    const void *v, const void *sink, const void *gate, void *o,
                                                    int64_t iters, void *ms, cudaStream_t stream) {
    if (!fits_int(n_seqs) || iters < 1 || !ms) return CERE_ERR_ARGUMENT;
    const A::AttnArgs a = attn_args(kind, seqs, n_seqs, max_rows, 0, q, A::kQRow, r, A::kRelRow, rel_proj, k,
                                    A::kKvRow, v, A::kKvRow, sink, A::kHeads, gate, A::kQRow, o, A::kQRow, 0);
    cudaError_t err = A::prefill(a, stream);
    if (err != cudaSuccess) return status_of(err);
    cudaEvent_t e0, e1;
    if (cudaEventCreate(&e0) != cudaSuccess) return CERE_ERR_INTERNAL;
    if (cudaEventCreate(&e1) != cudaSuccess) {
        cudaEventDestroy(e0);
        return CERE_ERR_INTERNAL;
    }
    err = cudaEventRecord(e0, stream);
    for (int64_t i = 0; i < iters && err == cudaSuccess; ++i) err = A::prefill(a, stream);
    if (err == cudaSuccess) err = cudaEventRecord(e1, stream);
    if (err == cudaSuccess) err = cudaEventSynchronize(e1);
    float total = 0.0f;
    if (err == cudaSuccess) err = cudaEventElapsedTime(&total, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (err != cudaSuccess) return status_of(err);
    *static_cast<float *>(ms) = total / float(iters);
    return CERE_OK;
}

CERE_KTEST int cere_ktest_inkling_attn_kv_store(int64_t kind, const void *seqs, int64_t n_seqs, int64_t max_rows,
                                                const void *k, int64_t k_stride, const void *v, int64_t v_stride,
                                                cudaStream_t stream) {
    if (!fits_int(n_seqs)) return CERE_ERR_ARGUMENT;
    return status_of(A::kv_store(int(kind), static_cast<const A::Seq *>(seqs), int(n_seqs), max_rows, in(k), k_stride,
                                 in(v), v_stride, stream));
}

// ---- CPU reference (host memory) ----------------------------------------------------------------------------------

// The post ops of `rows` contiguous rows; positions int64 [rows] (full layers, q or r).
CERE_KTEST int cere_ktest_inkling_attn_ref_post(int64_t kind, int64_t rows, const void *positions, const void *q_in,
                                                const void *q_scale, void *q_out, const void *r_in,
                                                const void *rel_scale, void *r_out, const void *k_in, void *k_out,
                                                const void *v_in, void *v_out, cudaStream_t) {
    if ((kind != R::kSliding && kind != R::kFull) || rows < 0) return CERE_ERR_ARGUMENT;
    if ((q_in && (!q_scale || !q_out)) || (r_in && (!rel_scale || !r_out)) || (k_in && !k_out) || (v_in && !v_out))
        return CERE_ERR_ARGUMENT;
    const int64_t *pos = static_cast<const int64_t *>(positions);
    if (kind == R::kFull && (q_in || r_in) && !pos) return CERE_ERR_ARGUMENT;
    parallel_for(rows, [&](int64_t i) {
        const int64_t p = pos ? pos[i] : 0;
        if (q_in) R::q_post_row(int(kind), hin(q_in) + i * R::kQRow, hin(q_scale), p, hout(q_out) + i * R::kQRow);
        if (r_in)
            R::rel_post_row(int(kind), hin(r_in) + i * R::kRelRow, hin(rel_scale), p, hout(r_out) + i * R::kRelRow);
        if (k_in) R::k_post_row(hin(k_in) + i * R::kKvRow, hout(k_out) + i * R::kKvRow);
        if (v_in) R::v_post_row(hin(v_in) + i * R::kKvRow, hout(v_out) + i * R::kKvRow);
    });
    return CERE_OK;
}

// attention_row for the listed rows (check_rows int64 [n_check], or null with n_check = every row of the table);
// contiguous rows (q, gate, out [*, 3072], r [*, 384], k, v [*, 1024], sink [*, 24]); other out rows untouched.
CERE_KTEST int cere_ktest_inkling_attn_ref_attention(int64_t kind, const void *seqs, int64_t n_seqs, const void *q,
                                                     const void *r, const void *rel_proj, const void *k,
                                                     const void *v, const void *sink, const void *gate, void *o,
                                                     const void *check_rows, int64_t n_check, cudaStream_t) {
    if ((kind != R::kSliding && kind != R::kFull) || !seqs || n_seqs < 1 || n_check < 0) return CERE_ERR_ARGUMENT;
    if (!q || !r || !rel_proj || !k || !v || !sink || !gate || !o) return CERE_ERR_ARGUMENT;
    const A::Seq *table = static_cast<const A::Seq *>(seqs);
    std::vector<int64_t> rows;
    if (check_rows) {
        const int64_t *c = static_cast<const int64_t *>(check_rows);
        rows.assign(c, c + n_check);
    } else {
        for (int64_t s = 0; s < n_seqs; ++s)
            for (int64_t i = 0; i < table[s].rows; ++i) rows.push_back(table[s].row + i);
    }
    // Each row's sequence.
    std::vector<int64_t> owner(rows.size(), -1);
    for (size_t x = 0; x < rows.size(); ++x)
        for (int64_t s = 0; s < n_seqs; ++s)
            if (rows[x] >= table[s].row && rows[x] < table[s].row + table[s].rows) owner[x] = s;
    for (int64_t s : owner)
        if (s < 0) return CERE_ERR_ARGUMENT;
    parallel_for(int64_t(rows.size()), [&](int64_t x) {
        const A::Seq &s = table[owner[x]];
        R::Keys keys{int(kind),
                     s.row,
                     s.pos0,
                     hin(k),
                     hin(v),
                     R::kKvRow,
                     R::kKvRow,
                     reinterpret_cast<const uint16_t *>(s.k_cache),
                     reinterpret_cast<const uint16_t *>(s.v_cache),
                     s.slots};
        const int64_t row = rows[x];
        R::attention_row(keys, hin(q) + row * R::kQRow, hin(r) + row * R::kRelRow, hin(rel_proj),
                         hin(sink) + row * R::kHeads, hin(gate) + row * R::kQRow, s.pos0 + (row - s.row),
                         hout(o) + row * R::kQRow);
    });
    return CERE_OK;
}
