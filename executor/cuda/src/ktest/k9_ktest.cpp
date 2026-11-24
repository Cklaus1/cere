// K9 kernel-test entry points (executor/tests/gpu/test_kernels_k9_qsa.py; prototypes in its docstring): the QSA
// prep (index q + compressed-key store), the token selection and the sparse GQA of kernels/k9_qsa.h on device
// buffers. Device pointers, int64 sizes, the stream last; 0 = success, else a cere_status. Scratch is stream-ordered
// memory allocated and freed on the stream for the call.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kernels/k9_qsa.h"
#include "ktest/l2_flush.h"

namespace k9 = cere::k9;
using k9::bf16;

namespace {

int free_status(int status, void* scratch, cudaStream_t stream) {
    if (scratch == nullptr) return status;
    const cudaError_t freed = cudaFreeAsync(scratch, stream);
    if (status != CERE_OK) return status;
    return freed == cudaSuccess ? CERE_OK : CERE_ERR_INTERNAL;
}

// The k9::Bounds (audit KernelsAttention F3) the entry points below pass: open (no host check binds, no fault word)
// unless cere_ktest_qsa_set_bounds set them.
constexpr k9::Bounds kOpenBounds{INT64_MIN / 4, INT64_MAX / 4, INT64_MAX / 4, nullptr};
k9::Bounds g_bounds = kOpenBounds;

}  // namespace

// Sets the k9::Bounds of every later qsa entry point of the process (rows rope_first .. rope_end - 1 of the rope table
// passed, logical positions 0 .. context - 1; fault: a device uint32 word, or null); context 0 restores the open
// bounds. cere_ktest_qsa_chains_step bounds each row's logical position by its slot map's length (slots_len) instead
// of `context`. The stream is unused.
extern "C" __attribute__((visibility("default"))) int cere_ktest_qsa_set_bounds(int64_t rope_first, int64_t rope_end,
                                                                                 int64_t context, void* fault,
                                                                                 cudaStream_t) {
    g_bounds = context == 0 ? kOpenBounds : k9::Bounds{rope_first, rope_end, context, static_cast<uint32_t*>(fault)};
    return CERE_OK;
}

// indexer_qk then qsa_compress_store for `tokens` new tokens of one session at logical positions pos0 ..: qk
// [tokens, 640] BF16 (contiguous index_qk_proj output), positions [tokens] int64 (null: pos0 + t), slots int64
// (null: identity), q_norm/k_norm [128] BF16, cos_sin the K1 rope table [n_pos, 64] F32. Writes q_idx
// [tokens, 4, 128], the compressed keys of the groups this call completes and the ring [4, 128].
extern "C" __attribute__((visibility("default"))) int cere_ktest_qsa_compress(
    const void* qk, int64_t tokens, int64_t pos0, const void* positions, const void* slots, const void* q_norm,
    const void* k_norm, float eps, const void* cos_sin, void* ring, void* keys, void* q_idx, cudaStream_t stream) {
    const bf16* qk_rows = static_cast<const bf16*>(qk);
    const float* table = static_cast<const float*>(cos_sin);
    const int64_t* pos = static_cast<const int64_t*>(positions);
    cere_status status = k9::indexer_qk(qk_rows, k9::kQkWidth, static_cast<const bf16*>(q_norm), eps, table, g_bounds,
                                        pos, pos0, tokens, static_cast<bf16*>(q_idx), stream);
    if (status != CERE_OK) return status;
    return k9::qsa_compress_store(qk_rows ? qk_rows + k9::kIndexHeads * k9::kIndexHeadDim : nullptr, k9::kQkWidth,
                                  static_cast<const bf16*>(k_norm), eps, table, g_bounds, pos,
                                  static_cast<const int64_t*>(slots), pos0, tokens, static_cast<bf16*>(ring),
                                  static_cast<bf16*>(keys), stream);
}

// cere_ktest_qsa_compress through indexer_qk_mrope then qsa_compress_store_mrope (contract C5): positions3 int64
// [3][p3_stride] (axis-major; column c = logical position pos0 - p3_lead + c, p3_lead = pos0 % 4).
extern "C" __attribute__((visibility("default"))) int cere_ktest_qsa_compress_mrope(
    const void* qk, int64_t tokens, int64_t pos0, const void* positions3, int64_t p3_stride, int64_t p3_lead,
    const void* slots, const void* q_norm, const void* k_norm, float eps, const void* cos_sin, void* ring, void* keys,
    void* q_idx, cudaStream_t stream) {
    const bf16* qk_rows = static_cast<const bf16*>(qk);
    const float* table = static_cast<const float*>(cos_sin);
    const int64_t* p3 = static_cast<const int64_t*>(positions3);
    cere_status status = k9::indexer_qk_mrope(qk_rows, k9::kQkWidth, static_cast<const bf16*>(q_norm), eps, table,
                                              g_bounds, p3, p3_stride, p3_lead, tokens, static_cast<bf16*>(q_idx),
                                              stream);
    if (status != CERE_OK) return status;
    return k9::qsa_compress_store_mrope(qk_rows ? qk_rows + k9::kIndexHeads * k9::kIndexHeadDim : nullptr,
                                        k9::kQkWidth, static_cast<const bf16*>(k_norm), eps, table, g_bounds, p3,
                                        p3_stride, p3_lead, static_cast<const int64_t*>(slots), pos0, tokens,
                                        static_cast<bf16*>(ring), static_cast<bf16*>(keys), stream);
}

// attention_core_step's QSA launches of one decode token at the device position `position` (int64 [1]), in its
// order: qsa_compress_store_step (qk's index key -> the ring, the compressed key a completed group), indexer_qk at the
// device position (q_idx [1, 4, 128]) and qsa_select_step over a context of ctx (indices [1, 2051]); arguments as
// cere_ktest_qsa_compress, bounds as cere_ktest_qsa_set_bounds (context <= ctx).
extern "C" __attribute__((visibility("default"))) int cere_ktest_qsa_decode_step(
    const void* qk, const void* position, const void* slots, const void* q_norm, const void* k_norm, float eps,
    const void* cos_sin, int64_t ctx, void* ring, void* keys, void* q_idx, void* indices, cudaStream_t stream) {
    if (qk == nullptr || ctx < k9::kRatio) return CERE_ERR_ARGUMENT;
    const bf16* qk_row = static_cast<const bf16*>(qk);
    const float* table = static_cast<const float*>(cos_sin);
    const int64_t* pos = static_cast<const int64_t*>(position);
    const int64_t* sl = static_cast<const int64_t*>(slots);
    cere_status status = k9::qsa_compress_store_step(qk_row + k9::kIndexHeads * k9::kIndexHeadDim,
                                                     static_cast<const bf16*>(k_norm), eps, table, g_bounds, sl, pos,
                                                     static_cast<bf16*>(ring), static_cast<bf16*>(keys), stream);
    if (status != CERE_OK) return status;
    status = k9::indexer_qk(qk_row, k9::kQkWidth, static_cast<const bf16*>(q_norm), eps, table, g_bounds, pos, 0, 1,
                            static_cast<bf16*>(q_idx), stream);
    if (status != CERE_OK) return status;
    const size_t bytes = static_cast<size_t>(ctx / k9::kRatio) * sizeof(float);
    void* scratch = nullptr;
    if (cudaMallocAsync(&scratch, bytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    status = k9::qsa_select_step(static_cast<const bf16*>(q_idx), static_cast<const bf16*>(keys), sl, g_bounds, pos,
                                 ctx, scratch, bytes, static_cast<int32_t*>(indices), stream);
    return free_status(status, scratch, stream);
}

// k9::qsa_compress_store_at (the draft graph's MTP layer): `tokens` index keys (qk [tokens, 640]) at the device
// position `position` (int64 [1], the first token's) -> the ring and the compressed keys of the groups they complete;
// arguments as cere_ktest_qsa_compress, bounds as cere_ktest_qsa_set_bounds.
extern "C" __attribute__((visibility("default"))) int cere_ktest_qsa_compress_at(
    const void* qk, int64_t tokens, const void* position, const void* slots, const void* k_norm, float eps,
    const void* cos_sin, void* ring, void* keys, cudaStream_t stream) {
    if (qk == nullptr) return CERE_ERR_ARGUMENT;
    return k9::qsa_compress_store_at(static_cast<const bf16*>(qk) + k9::kIndexHeads * k9::kIndexHeadDim, k9::kQkWidth,
                                     static_cast<const bf16*>(k_norm), eps, static_cast<const float*>(cos_sin),
                                     g_bounds, static_cast<const int64_t*>(slots),
                                     static_cast<const int64_t*>(position), tokens, static_cast<bf16*>(ring),
                                     static_cast<bf16*>(keys), stream);
}

// qsa_select of `rows` query rows (positions_logical [rows] int64, null: seq_len - rows + t) over the compressed
// keys of a sequence of seq_len tokens: indices [rows, 2051] int32. scratch_bytes 0 = select_scratch_bytes(rows,
// seq_len) (the 128 MiB logits tile), else exactly that many bytes (smaller tiles). With logits non-null, also
// qsa_logits into logits [rows, seq_len / 4] F32.
extern "C" __attribute__((visibility("default"))) int cere_ktest_qsa_select(
    const void* q_idx, const void* keys, const void* slots, const void* positions_logical, int64_t rows,
    int64_t seq_len, int64_t scratch_bytes, void* indices, void* logits, cudaStream_t stream) {
    if (rows < 0 || seq_len <= 0 || scratch_bytes < 0) return CERE_ERR_ARGUMENT;
    const bf16* q = static_cast<const bf16*>(q_idx);
    const bf16* k = static_cast<const bf16*>(keys);
    const int64_t* sl = static_cast<const int64_t*>(slots);
    const int64_t* pl = static_cast<const int64_t*>(positions_logical);
    if (logits != nullptr) {
        const cere_status s = k9::qsa_logits(q, k, sl, pl, rows, seq_len, static_cast<float*>(logits),
                                             seq_len / k9::kRatio, 0, stream);
        if (s != CERE_OK) return s;
    }
    const size_t bytes =
        scratch_bytes > 0 ? static_cast<size_t>(scratch_bytes) : k9::select_scratch_bytes(rows, seq_len);
    void* scratch = nullptr;
    if (bytes > 0 && cudaMallocAsync(&scratch, bytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    const cere_status status =
        k9::qsa_select(q, k, sl, pl, rows, seq_len, scratch, bytes, static_cast<int32_t*>(indices), stream);
    return free_status(status, scratch, stream);
}

// k9::qsa_logits (track LongK9, long-k9 design note LK9-02) of `rows` query rows into logits [rows, seq_len / 4]
// F32; arguments as cere_ktest_qsa_select. impl 0 = qsa_select's choice, 1 = logits_kernel, 2 = logits_small_kernel.
// iters 0: one call; iters >= 1: one untimed call then `iters` cold-L2 calls (l2_flush.h cold_us: in the model the
// compressed keys come from DRAM), `us` (host double [2]) = the median and minimum µs of one call.
extern "C" __attribute__((visibility("default"))) int cere_ktest_qsa_logits(
    const void* q_idx, const void* keys, const void* slots, const void* positions_logical, int64_t rows,
    int64_t seq_len, int64_t impl, void* logits, int64_t iters, double* us, cudaStream_t stream) {
    if (iters < 0 || (iters > 0 && us == nullptr) || impl < 0 || impl > 2) return CERE_ERR_ARGUMENT;
    const auto call = [&]() {
        return k9::qsa_logits(static_cast<const bf16*>(q_idx), static_cast<const bf16*>(keys),
                              static_cast<const int64_t*>(slots), static_cast<const int64_t*>(positions_logical), rows,
                              seq_len, static_cast<float*>(logits), seq_len / k9::kRatio, static_cast<int>(impl),
                              stream);
    };
    if (iters == 0) return call();
    void* flush = nullptr;
    if (cudaMallocAsync(&flush, cere::ktest::kL2FlushBytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    return free_status(cere::ktest::cold_us(flush, iters, us, stream, call), flush, stream);
}

// k9::qsa_topk (track LongK9, long-k9 design note LK9-01) on a logits tile [rows, ld] F32 (positions_logical
// [rows] int64, null: seq_len - rows + t): indices [rows, 2051] int32. impl 0 = qsa_select's choice, 1 = one block
// per row, 2 = the cluster kernel. iters 0: one call; iters >= 1: `iters` CUDA-graph replays after 3 untimed ones
// (graph_spans, warm L2: in the model the logits kernel has just written the tile), `us` (host double [2]) = the
// median and minimum µs of one call.
extern "C" __attribute__((visibility("default"))) int cere_ktest_qsa_topk(
    const void* logits, int64_t ld, const void* positions_logical, int64_t rows, int64_t seq_len, int64_t impl,
    void* indices, int64_t iters, double* us, cudaStream_t stream) {
    if (iters < 0 || (iters > 0 && us == nullptr) || impl < 0 || impl > 2) return CERE_ERR_ARGUMENT;
    const auto run = [&](cudaStream_t s) {
        return k9::qsa_topk(static_cast<const float*>(logits), ld, static_cast<const int64_t*>(positions_logical),
                            rows, seq_len, static_cast<int>(impl), static_cast<int32_t*>(indices), s);
    };
    if (iters == 0) return run(stream);
    std::vector<float> spans(static_cast<size_t>(iters));
    const cere_status s = cere::ktest::graph_spans(run, 3, iters, false, spans.data(), stream);
    if (s != CERE_OK) return s;
    std::sort(spans.begin(), spans.end());
    us[0] = spans[spans.size() / 2];
    us[1] = spans[0];
    return CERE_OK;
}

// qsa_attention of `rows` query rows: q [rows, 24 * 256] BF16 (normed, rotated), indices [rows, 2051] int32. K/V:
// either k/v [*, 2 * 256] BF16 (contiguous; logical position j at row j) or k_cache/v_cache [*, 2 * 256] FP8 E4M3
// (scale 1.0; position j at row slots[j], null: j); the other pair null. out [rows, 24 * 256] BF16.
// use_p4 as k9::qsa_attention: cere_ktest_qsa_attention runs the default prefill_attention lever (K9's own kernel for
// every call), the _p4 entry forces the P4 tensor-core kernel for calls of >= 17 rows (m23 design note P4; fewer
// rows run K9's own kernel), and _p4_cls makes that choice for `cls` rows (k9::qsa_attention's class pin, cls >= rows:
// a hot tail of a cold call of cls rows; BeatSglang S7).
static int qsa_attention(const void* q, int64_t rows, const void* indices, const void* k, const void* v,
                         const void* k_cache, const void* v_cache, const void* slots, float scale, void* out,
                         cudaStream_t stream, bool use_p4, int64_t cls = 0) {
    if (rows < 0) return CERE_ERR_ARGUMENT;
    constexpr int64_t kQWidth = int64_t{k9::kQHeads} * k9::kHeadDim;
    constexpr int64_t kKvWidth = int64_t{k9::kKvHeads} * k9::kHeadDim;
    const size_t bytes = k9::attention_scratch_bytes(rows);
    void* scratch = nullptr;
    if (bytes > 0 && cudaMallocAsync(&scratch, bytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    const cere_status status = k9::qsa_attention(
        static_cast<const bf16*>(q), kQWidth, static_cast<const int32_t*>(indices), rows, static_cast<const bf16*>(k),
        kKvWidth, static_cast<const bf16*>(v), kKvWidth, static_cast<const uint8_t*>(k_cache),
        static_cast<const uint8_t*>(v_cache), static_cast<const int64_t*>(slots), scale, scratch, bytes,
        static_cast<bf16*>(out), kQWidth, stream, use_p4, cls);
    return free_status(status, scratch, stream);
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_qsa_attention(
    const void* q, int64_t rows, const void* indices, const void* k, const void* v, const void* k_cache,
    const void* v_cache, const void* slots, float scale, void* out, cudaStream_t stream) {
    return qsa_attention(q, rows, indices, k, v, k_cache, v_cache, slots, scale, out, stream, false);
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_qsa_attention_p4(
    const void* q, int64_t rows, const void* indices, const void* k, const void* v, const void* k_cache,
    const void* v_cache, const void* slots, float scale, void* out, cudaStream_t stream) {
    return qsa_attention(q, rows, indices, k, v, k_cache, v_cache, slots, scale, out, stream, true);
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_qsa_attention_p4_cls(
    const void* q, int64_t rows, const void* indices, const void* k, const void* v, const void* k_cache,
    const void* v_cache, const void* slots, float scale, int64_t cls, void* out, cudaStream_t stream) {
    return qsa_attention(q, rows, indices, k, v, k_cache, v_cache, slots, scale, out, stream, true, cls);
}

// k9::qsa_attention_heads (default prefill_attention lever, no P4) with the rows kernel's heads per warp given:
// rows_heads 0 = attention_kernel for every call (the pre-PT-02 / pre-LK9-02 path), 3/6/12 = rows_attention_kernel for
// calls of more than 16 rows (prefill-ttft design note PT-02) and split_attention_kernel for FP8-cache calls of up to
// 16 rows (long-k9 design note LK9-02). Arguments as cere_ktest_qsa_attention. iters 0: one call; iters >= 1: one
// untimed call then `iters` cold-L2 calls (l2_flush.h cold_us), `us` (host double [2]) = the median and minimum µs of
// one call.
extern "C" __attribute__((visibility("default"))) int cere_ktest_qsa_attention_heads(
    const void* q, int64_t rows, const void* indices, const void* k, const void* v, const void* k_cache,
    const void* v_cache, const void* slots, float scale, void* out, int64_t rows_heads, int64_t iters, double* us,
    cudaStream_t stream) {
    if (rows < 0 || iters < 0 || (iters > 0 && us == nullptr) || rows_heads < 0 || rows_heads > 12)
        return CERE_ERR_ARGUMENT;
    constexpr int64_t kQWidth = int64_t{k9::kQHeads} * k9::kHeadDim;
    constexpr int64_t kKvWidth = int64_t{k9::kKvHeads} * k9::kHeadDim;
    const size_t bytes = k9::attention_scratch_bytes(rows);
    void* scratch = nullptr;
    if (bytes > 0 && cudaMallocAsync(&scratch, bytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    const auto call = [&]() {
        return k9::qsa_attention_heads(
            static_cast<const bf16*>(q), kQWidth, static_cast<const int32_t*>(indices), rows,
            static_cast<const bf16*>(k), kKvWidth, static_cast<const bf16*>(v), kKvWidth,
            static_cast<const uint8_t*>(k_cache), static_cast<const uint8_t*>(v_cache),
            static_cast<const int64_t*>(slots), scale, scratch, bytes, static_cast<bf16*>(out), kQWidth, stream, false,
            0, static_cast<int>(rows_heads));
    };
    cere_status status = CERE_OK;
    if (iters == 0) {
        status = call();
    } else {
        void* flush = nullptr;
        status = cudaMallocAsync(&flush, cere::ktest::kL2FlushBytes, stream) == cudaSuccess
                     ? cere::ktest::cold_us(flush, iters, us, stream, call)
                     : CERE_ERR_CAPACITY;
        if (flush != nullptr && cudaFreeAsync(flush, stream) != cudaSuccess && status == CERE_OK) {
            status = CERE_ERR_INTERNAL;
        }
    }
    return free_status(status, scratch, stream);
}

// fuse-glue design note FG-6: one full-attention layer's verify QSA work over chains of layout [chains] (host int64,
// rows m = sum <= 16; chain c's row r - first is draft row j of n = layout[c] - 1), chain c on its own ring
// [4, 128], keys [keys_rows, 128] and slot map slots + c * slots_len, row r on its index key k_idx + 128 r, the device
// position positions[r] and its undo ring undo_rings + 512 r / undo key undo_keys + 128 r (all BF16 / int64). fused 1:
// one k9::qsa_chains_step; 0: attention_chains' per-row launches (row j >= 1: the ring's undo copy, with qsa its
// k9::qsa_key_save_step; with qsa then k9::qsa_compress_store_step). Row r's logical position is bounded by its slot
// map's length (ChainsStep::context = slots_len), its rotary row by cere_ktest_qsa_set_bounds.
extern "C" __attribute__((visibility("default"))) int cere_ktest_qsa_chains_step(
    const void* k_idx, const void* k_norm, float eps, const void* cos_sin, const void* positions, const void* layout,
    int64_t chains, const void* slots, int64_t slots_len, void* rings, void* keys, int64_t keys_rows,
    void* undo_rings, void* undo_keys, int64_t qsa, int64_t fused, cudaStream_t stream) {
    if (chains < 1 || chains > k9::kChainsStepRows || layout == nullptr || slots_len < 1 || keys_rows < 1)
        return CERE_ERR_ARGUMENT;
    constexpr int64_t kRing = int64_t{k9::kRatio} * k9::kIndexHeadDim;
    const auto* counts = static_cast<const int64_t*>(layout);
    const auto* key_rows = static_cast<const bf16*>(k_idx);
    const auto* pos = static_cast<const int64_t*>(positions);
    const auto* norm = static_cast<const bf16*>(k_norm);
    const auto* table = static_cast<const float*>(cos_sin);
    k9::ChainsStep step{};
    int64_t r = 0;
    for (int64_t c = 0; c < chains; ++c) {
        if (counts[c] < 1 || r + counts[c] > k9::kChainsStepRows) return CERE_ERR_ARGUMENT;
        step.chain_first[c] = static_cast<int32_t>(r);
        step.chain_rows[c] = static_cast<int32_t>(counts[c]);
        for (int64_t j = 0; j < counts[c]; ++j, ++r) {
            step.slots[r] = static_cast<const int64_t*>(slots) + c * slots_len;
            step.ring[r] = static_cast<bf16*>(rings) + c * kRing;
            step.keys[r] = static_cast<bf16*>(keys) + c * keys_rows * k9::kIndexHeadDim;
            step.context[r] = slots_len;
            if (j > 0) {
                step.undo_ring[r] = static_cast<bf16*>(undo_rings) + r * kRing;
                if (qsa != 0) step.undo_key[r] = static_cast<bf16*>(undo_keys) + r * k9::kIndexHeadDim;
                step.j[r] = static_cast<int32_t>(j);
                step.n[r] = static_cast<int32_t>(counts[c] - 1);
            }
            if (qsa != 0) step.k_idx[r] = key_rows + r * k9::kIndexHeadDim;
        }
    }
    if (fused != 0) return k9::qsa_chains_step(step, static_cast<int>(chains), norm, eps, table, g_bounds, pos, stream);
    const k9::Bounds bounds{g_bounds.rope_first, g_bounds.rope_end, slots_len, g_bounds.fault};
    for (int64_t row = 0; row < r; ++row) {
        if (step.undo_ring[row] != nullptr) {
            if (cudaMemcpyAsync(step.undo_ring[row], step.ring[row], kRing * sizeof(bf16), cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess)
                return CERE_ERR_INTERNAL;
            if (step.undo_key[row] != nullptr) {
                const cere_status s = k9::qsa_key_save_step(step.keys[row], step.slots[row], bounds, pos + row,
                                                            step.j[row], step.n[row], step.undo_key[row], stream);
                if (s != CERE_OK) return s;
            }
        }
        if (step.k_idx[row] == nullptr) continue;
        const cere_status s = k9::qsa_compress_store_step(step.k_idx[row], norm, eps, table, bounds, step.slots[row],
                                                          pos + row, step.ring[row], step.keys[row], stream);
        if (s != CERE_OK) return s;
    }
    return CERE_OK;
}
