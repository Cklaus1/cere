// K6 kernel-test entry points (executor/tests/gpu/test_kernels_k6_attn.py; prototypes in its docstring): one
// full-attention block, hidden -> q/k/v projections (K2 GEMM) -> kernels/k6_attention.h -> o_proj (K2 GEMM); and the
// D10 prologue alone (m22c design note, oracle rows in test_kernels_k9_qsa.py).
// The scratch buffers are allocated and freed on the stream for this call.
#include <cstdint>

#include "kernels/k6_attention.h"
#include "ktest/k1_ktest.h"
#include "ktest/k2_ktest.h"

using cere::k6::bf16;

namespace {

size_t align256(size_t n) { return (n + 255) & ~static_cast<size_t>(255); }

}  // namespace

// Shapes must equal the compiled support table (24 Q heads, 2 KV heads, head_dim 256, rotary_dim 64), kv_scale 1.0
// and softmax_scale 256^-0.5; anything else is CERE_ERR_ARGUMENT. positions must lie in [0, n_pos) and slots in
// [0, n_slots): both are device arrays the kernels read without a bounds check. use_p4 as k6::attention:
// cere_ktest_attn_block runs the default prefill_attention lever (CUDA-core kernels for every call),
// cere_ktest_attn_block_p4 forces the P4 tensor-core kernel for calls of >= 17 tokens (m23 design note P4; fewer
// tokens run K6's own kernels).
static int attn_block(const void* hidden, const void* w_q, const void* w_k, const void* w_v, const void* w_o,
                      const void* q_norm, const void* k_norm, const void* cos_sin, int64_t n_pos, const void* positions,
                      const void* slots, int64_t prefix_len, int64_t T, void* k_cache, void* v_cache, int64_t n_slots,
                      int64_t kv_from_cache, int64_t hidden_size, int64_t n_q_heads, int64_t n_kv_heads,
                      int64_t head_dim, int64_t rotary_dim, float eps, float kv_scale, float softmax_scale,
                      void* attn_out, void* out, cudaStream_t stream, bool use_p4) {
    namespace k6 = cere::k6;
    using cere::kernels::GemmOut;
    if (n_q_heads != k6::kQHeads || n_kv_heads != k6::kKvHeads || head_dim != k6::kHeadDim ||
        rotary_dim != k6::kRotaryDim || kv_scale != 1.0f || softmax_scale != k6::kSoftmaxScale ||
        (kv_from_cache != 0 && kv_from_cache != 1) || (kv_from_cache == 0 && prefix_len != 0) || T <= 0 ||
        prefix_len < 0 || prefix_len + T > k6::kMaxContext || prefix_len + T > n_slots || n_pos <= 0 ||
        hidden_size <= 0 || !positions || !slots)
        return CERE_ERR_ARGUMENT;
    constexpr int64_t kQGateWidth = 2 * k6::kQHeads * k6::kHeadDim;
    constexpr int64_t kKvWidth = k6::kKvHeads * k6::kHeadDim;
    constexpr int64_t kAttnWidth = k6::kQHeads * k6::kHeadDim;
    const size_t t = static_cast<size_t>(T);
    const size_t q_bytes = align256(t * kQGateWidth * sizeof(bf16));
    const size_t kv_bytes = align256(t * kKvWidth * sizeof(bf16));
    const size_t attn_ws = k6::full_attention_workspace_bytes(T);
    unsigned char* scratch = nullptr;
    if (cudaMallocAsync(reinterpret_cast<void**>(&scratch), q_bytes + 2 * kv_bytes + attn_ws, stream) != cudaSuccess)
        return CERE_ERR_CAPACITY;
    bf16* q_gate = reinterpret_cast<bf16*>(scratch);
    bf16* k = reinterpret_cast<bf16*>(scratch + q_bytes);
    bf16* v = reinterpret_cast<bf16*>(scratch + q_bytes + kv_bytes);
    void* ws = scratch + q_bytes + 2 * kv_bytes;

    int status = cere::ktest::gemm(hidden, w_q, T, kQGateWidth, hidden_size, q_gate, GemmOut::kBF16, stream);
    if (status == CERE_OK)
        status = cere::ktest::gemm(hidden, w_k, T, kKvWidth, hidden_size, k, GemmOut::kBF16, stream);
    if (status == CERE_OK)
        status = cere::ktest::gemm(hidden, w_v, T, kKvWidth, hidden_size, v, GemmOut::kBF16, stream);
    if (status == CERE_OK)
        status = cere::ktest::status_of(k6::full_attention(
            q_gate, kQGateWidth, k, kKvWidth, v, kKvWidth, static_cast<const bf16*>(q_norm),
            static_cast<const bf16*>(k_norm), eps, static_cast<const float*>(cos_sin),
            static_cast<const int64_t*>(positions), nullptr, static_cast<const int64_t*>(slots), T, prefix_len,
            kv_from_cache == 1, static_cast<uint8_t*>(k_cache), static_cast<uint8_t*>(v_cache), ws, attn_ws,
            static_cast<bf16*>(attn_out), kAttnWidth, stream, use_p4));
    if (status == CERE_OK)
        status = cere::ktest::gemm(attn_out, w_o, T, hidden_size, kAttnWidth, out, GemmOut::kBF16, stream);
    const cudaError_t freed = cudaFreeAsync(scratch, stream);
    if (status != CERE_OK) return status;
    return cere::ktest::status_of(freed);
}

#define CERE_KTEST_ATTN_BLOCK_PARAMS                                                                                  \
    const void *hidden, const void *w_q, const void *w_k, const void *w_v, const void *w_o, const void *q_norm,       \
        const void *k_norm, const void *cos_sin, int64_t n_pos, const void *positions, const void *slots,             \
        int64_t prefix_len, int64_t T, void *k_cache, void *v_cache, int64_t n_slots, int64_t kv_from_cache,          \
        int64_t hidden_size, int64_t n_q_heads, int64_t n_kv_heads, int64_t head_dim, int64_t rotary_dim, float eps,  \
        float kv_scale, float softmax_scale, void *attn_out, void *out, cudaStream_t stream
#define CERE_KTEST_ATTN_BLOCK_ARGS                                                                                    \
    hidden, w_q, w_k, w_v, w_o, q_norm, k_norm, cos_sin, n_pos, positions, slots, prefix_len, T, k_cache, v_cache,    \
        n_slots, kv_from_cache, hidden_size, n_q_heads, n_kv_heads, head_dim, rotary_dim, eps, kv_scale,              \
        softmax_scale, attn_out, out, stream

extern "C" __attribute__((visibility("default"))) int cere_ktest_attn_block(CERE_KTEST_ATTN_BLOCK_PARAMS) {
    return attn_block(CERE_KTEST_ATTN_BLOCK_ARGS, false);
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_attn_block_p4(CERE_KTEST_ATTN_BLOCK_PARAMS) {
    return attn_block(CERE_KTEST_ATTN_BLOCK_ARGS, true);
}

#undef CERE_KTEST_ATTN_BLOCK_PARAMS
#undef CERE_KTEST_ATTN_BLOCK_ARGS

// k6::prologue on device buffers: q_gate [T, 24*2*256], k [T, 2*256], v [T, 2*256] BF16 (contiguous), q_norm/k_norm
// [256] BF16, cos_sin the K1 rope table [n_pos, 64] F32 (n_pos = the context bound), positions [T] and slots
// [prefix_len + T] int64 (null positions: prefix_len + t; null slots: identity), k_cache/v_cache [n_slots, 2*256] FP8.
// Writes q_out [T, 24*256], k_out [T, 2*256], gate_out [T, 24*256] BF16 and the FP8 rows of the new tokens.
// prefix_len + T <= n_pos (no kMaxContext cap) and, with identity slots, <= n_slots; else CERE_ERR_ARGUMENT.
// Slots and positions are read without a bounds check (the caller keeps them in [0, n_slots) / [0, n_pos)).
static int attn_prologue(const void* q_gate, const void* k, const void* v, const void* q_norm, const void* k_norm,
                         const void* cos_sin, int64_t n_pos, const void* positions, const void* positions3,
                         const void* slots, int64_t prefix_len, int64_t T, void* k_cache, void* v_cache,
                         int64_t n_slots, float eps, void* q_out, void* k_out, void* gate_out, cudaStream_t stream) {
    namespace k6 = cere::k6;
    if (T <= 0 || prefix_len < 0 || n_pos <= 0 || n_slots <= 0 || prefix_len + T > n_pos ||
        (!slots && prefix_len + T > n_slots))
        return CERE_ERR_ARGUMENT;
    constexpr int64_t kQGateWidth = 2 * k6::kQHeads * k6::kHeadDim;
    constexpr int64_t kKvWidth = k6::kKvHeads * k6::kHeadDim;
    return cere::ktest::status_of(k6::prologue(
        static_cast<const bf16*>(q_gate), kQGateWidth, static_cast<const bf16*>(k), kKvWidth,
        static_cast<const bf16*>(v), kKvWidth, static_cast<const bf16*>(q_norm), static_cast<const bf16*>(k_norm), eps,
        static_cast<const float*>(cos_sin), n_pos, static_cast<const int64_t*>(positions),
        static_cast<const int64_t*>(positions3), static_cast<const int64_t*>(slots), T, prefix_len,
        static_cast<uint8_t*>(k_cache), static_cast<uint8_t*>(v_cache), static_cast<bf16*>(q_out),
        static_cast<bf16*>(k_out), static_cast<bf16*>(gate_out), stream));
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_attn_prologue(
    const void* q_gate, const void* k, const void* v, const void* q_norm, const void* k_norm, const void* cos_sin,
    int64_t n_pos, const void* positions, const void* slots, int64_t prefix_len, int64_t T, void* k_cache,
    void* v_cache, int64_t n_slots, float eps, void* q_out, void* k_out, void* gate_out, cudaStream_t stream) {
    return attn_prologue(q_gate, k, v, q_norm, k_norm, cos_sin, n_pos, positions, nullptr, slots, prefix_len, T,
                         k_cache, v_cache, n_slots, eps, q_out, k_out, gate_out, stream);
}

// M10 (m213 design note D11): cere_ktest_attn_prologue with positions3 [3][T] int64 (axes t, h, w; non-null,
// each < n_pos), which wins over positions (may be null).
extern "C" __attribute__((visibility("default"))) int cere_ktest_attn_prologue3(
    const void* q_gate, const void* k, const void* v, const void* q_norm, const void* k_norm, const void* cos_sin,
    int64_t n_pos, const void* positions, const void* positions3, const void* slots, int64_t prefix_len, int64_t T,
    void* k_cache, void* v_cache, int64_t n_slots, float eps, void* q_out, void* k_out, void* gate_out,
    cudaStream_t stream) {
    if (!positions3) return CERE_ERR_ARGUMENT;
    return attn_prologue(q_gate, k, v, q_norm, k_norm, cos_sin, n_pos, positions, positions3, slots, prefix_len, T,
                         k_cache, v_cache, n_slots, eps, q_out, k_out, gate_out, stream);
}

// The gated dense attention of `tokens` query rows over the FP8 cache through one of k6's decode/verify entry points
// (m23 design note P5R): q, gate [tokens, 24*256] BF16 (contiguous), k_cache/v_cache [n_slots, 2*256] FP8 rows
// holding keys 0 .. pos0 + tokens - 1 at slots[0 ..] (int64, device), pos (int64 [1], device) = pos0. mode 0
// dense_attention (eager, host pos0), 1 dense_attention_at (*pos), 2 dense_attention_rows (row t: the one-token
// call at pos0 + t), 3 dense_attention_rows_at (*pos). Writes out [tokens, 24*256] BF16. Modes 1-3 take
// 1 <= tokens <= 16; every mode pos0 + tokens <= 2,048. Else CERE_ERR_ARGUMENT.
extern "C" __attribute__((visibility("default"))) int cere_ktest_attn_dense(const void* q, const void* gate,
                                                                           int64_t tokens, int64_t pos0,
                                                                           const void* pos, const void* k_cache,
                                                                           const void* v_cache, const void* slots,
                                                                           int64_t mode, void* out,
                                                                           cudaStream_t stream) {
    namespace k6 = cere::k6;
    constexpr int64_t kAttnWidth = k6::kQHeads * k6::kHeadDim;
    if (!q || !gate || !pos || !k_cache || !v_cache || !slots || !out || mode < 0 || mode > 3 || tokens < 1 ||
        pos0 < 0 || pos0 + tokens > k6::kMaxContext || (mode > 0 && tokens > 16))
        return CERE_ERR_ARGUMENT;
    const size_t ws_bytes = k6::dense_attention_workspace_bytes(tokens);
    void* ws = nullptr;
    if (cudaMallocAsync(&ws, ws_bytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    const auto* qb = static_cast<const bf16*>(q);
    const auto* gb = static_cast<const bf16*>(gate);
    const auto* kc = static_cast<const uint8_t*>(k_cache);
    const auto* vc = static_cast<const uint8_t*>(v_cache);
    const auto* sl = static_cast<const int64_t*>(slots);
    const auto* p = static_cast<const int64_t*>(pos);
    auto* o = static_cast<bf16*>(out);
    cudaError_t err = cudaSuccess;
    switch (mode) {
        case 0:  // k and v are unread with kv_from_cache; q stands in for the non-null pointers
            err = k6::dense_attention(qb, qb, qb, kAttnWidth, gb, tokens, pos0, true, kc, vc, sl, ws, ws_bytes, o,
                                      kAttnWidth, stream, false);
            break;
        case 1:
            err = k6::dense_attention_at(qb, gb, tokens, p, kc, vc, sl, ws, ws_bytes, o, kAttnWidth, stream);
            break;
        case 2:
            err = k6::dense_attention_rows(qb, gb, tokens, pos0, kc, vc, sl, ws, ws_bytes, o, kAttnWidth, stream);
            break;
        default:
            err = k6::dense_attention_rows_at(qb, gb, tokens, p, kc, vc, sl, ws, ws_bytes, o, kAttnWidth, stream);
            break;
    }
    const cudaError_t freed = cudaFreeAsync(ws, stream);
    if (err != cudaSuccess) return cere::ktest::status_of(err);
    return cere::ktest::status_of(freed);
}

// k6::attention (CUDA-core kernels) over BF16 K/V rows with explicit row strides (elements): q [tokens, 24*256], k
// and v [tokens, max(k_stride, v_stride)] BF16 (rows at their stride), pos0 0, out [tokens, 24*256]. A stride below
// the 2*256 row width, or not a multiple of 8, returns CERE_ERR_ARGUMENT before any launch. 1 <= tokens <= 16.
extern "C" __attribute__((visibility("default"))) int cere_ktest_attn_kv_strides(const void* q, const void* k,
                                                                                const void* v, int64_t k_stride,
                                                                                int64_t v_stride, int64_t tokens,
                                                                                void* out, cudaStream_t stream) {
    namespace k6 = cere::k6;
    if (!q || !k || !v || !out || tokens < 1 || tokens > 16) return CERE_ERR_ARGUMENT;
    const size_t ws_bytes = k6::dense_attention_workspace_bytes(tokens);
    void* ws = nullptr;
    if (cudaMallocAsync(&ws, ws_bytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    const cudaError_t err = k6::attention(static_cast<const bf16*>(q), tokens, 0, static_cast<const bf16*>(k), k_stride,
                                          static_cast<const bf16*>(v), v_stride, nullptr, nullptr, nullptr,
                                          k6::kSoftmaxScale, ws, ws_bytes, static_cast<bf16*>(out), stream, false);
    const cudaError_t freed = cudaFreeAsync(ws, stream);
    if (err != cudaSuccess) return cere::ktest::status_of(err);
    return cere::ktest::status_of(freed);
}

// k6::attention with the prefill_attention lever at P4 and a class pin (BeatSglang S7): q [tokens, 24*256] BF16, k and
// v [pos0 + tokens, 2*256] BF16 (contiguous, key j at row j), the call's rows at pos0 .. pos0 + tokens - 1, cls 0 (=
// tokens) or >= tokens, out [tokens, 24*256]. A pin of cls >= 17 runs P4 for any tokens (the hot tail of a cold call
// of cls rows). tokens < 1, pos0 < 0 or pos0 + tokens above k6::kMaxContext: CERE_ERR_ARGUMENT.
extern "C" __attribute__((visibility("default"))) int cere_ktest_attn_p4_cls(const void* q, const void* k,
                                                                            const void* v, int64_t tokens,
                                                                            int64_t pos0, int64_t cls, void* out,
                                                                            cudaStream_t stream) {
    namespace k6 = cere::k6;
    constexpr int64_t kKvWidth = k6::kKvHeads * k6::kHeadDim;
    if (!q || !k || !v || !out || tokens < 1 || pos0 < 0 || pos0 + tokens > k6::kMaxContext) return CERE_ERR_ARGUMENT;
    const size_t ws_bytes = k6::dense_attention_workspace_bytes(tokens);
    void* ws = nullptr;
    if (cudaMallocAsync(&ws, ws_bytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    const cudaError_t err =
        k6::attention(static_cast<const bf16*>(q), tokens, pos0, static_cast<const bf16*>(k), kKvWidth,
                      static_cast<const bf16*>(v), kKvWidth, nullptr, nullptr, nullptr, k6::kSoftmaxScale, ws,
                      ws_bytes, static_cast<bf16*>(out), stream, true, cls);
    const cudaError_t freed = cudaFreeAsync(ws, stream);
    if (err != cudaSuccess) return cere::ktest::status_of(err);
    return cere::ktest::status_of(freed);
}

// rows16 design note R16-C: the verify graph's full-attention core of `chains` sequences' chains (chain_rows [chains]
// int64, host; rows = their sum, 1 .. 16): q_gate [rows, 24*2*256], k, v [rows, 2*256] BF16 (contiguous), q_norm /
// k_norm [256] BF16, cos_sin [n_pos, 64] F32, positions [rows] int64 (device; chain c's rows p_c, p_c + 1, ..., every
// p_c >= 1 and p_c + rows_c <= min(2,048, n_pos)), sequence c's slot map slots + c * n_slots (int64, device; values in
// [0, n_slots)) and FP8 caches k_caches / v_caches + c * n_slots * 2*256 bytes (keys 0 .. p_c - 1 at its slots).
// mode 0: per chain k6::prologue_rows_at then k6::dense_attention_rows_at (attention_chain's calls); mode 1: one
// k6::prologue_chains_at then one k6::dense_attention_chains_at over the rows. Writes out [rows, 24*256] BF16 and the
// rows' K/V cache rows. Else CERE_ERR_ARGUMENT.
extern "C" __attribute__((visibility("default"))) int cere_ktest_attn_chains(
    const void* q_gate, const void* k, const void* v, const void* q_norm, const void* k_norm, const void* cos_sin,
    const void* positions, const void* slots, void* k_caches, void* v_caches, int64_t n_slots,
    const int64_t* chain_rows, int64_t chains, float eps, int64_t mode, void* out, cudaStream_t stream) {
    namespace k6 = cere::k6;
    namespace k1 = cere::k1;
    constexpr int64_t kAttnWidth = k6::kQHeads * k6::kHeadDim;
    constexpr int64_t kQGateWidth = 2 * kAttnWidth;
    constexpr int64_t kKvWidth = k6::kKvHeads * k6::kHeadDim;
    if (!q_gate || !k || !v || !q_norm || !k_norm || !cos_sin || !positions || !slots || !k_caches || !v_caches ||
        !chain_rows || !out || n_slots < 1 || chains < 1 || chains > k1::kKvRowsMax || mode < 0 || mode > 1)
        return CERE_ERR_ARGUMENT;
    int64_t rows = 0;
    for (int64_t c = 0; c < chains; ++c) {
        if (chain_rows[c] < 1) return CERE_ERR_ARGUMENT;
        rows += chain_rows[c];
    }
    if (rows > k1::kKvRowsMax) return CERE_ERR_ARGUMENT;
    const size_t ws_bytes = k6::dense_attention_workspace_bytes(rows);
    const size_t act_bytes = static_cast<size_t>(rows) * (2 * kAttnWidth + kKvWidth) * sizeof(bf16);
    void* ws = nullptr;
    void* act = nullptr;
    if (cudaMallocAsync(&ws, ws_bytes, stream) != cudaSuccess) return CERE_ERR_CAPACITY;
    if (cudaMallocAsync(&act, act_bytes, stream) != cudaSuccess) {
        (void)cudaFreeAsync(ws, stream);
        return CERE_ERR_CAPACITY;
    }
    auto* q_out = static_cast<bf16*>(act);
    bf16* gate_out = q_out + rows * kAttnWidth;
    bf16* k_out = gate_out + rows * kAttnWidth;
    const auto* qg = static_cast<const bf16*>(q_gate);
    const auto* kb = static_cast<const bf16*>(k);
    const auto* vb = static_cast<const bf16*>(v);
    const auto* qn = static_cast<const bf16*>(q_norm);
    const auto* kn = static_cast<const bf16*>(k_norm);
    const auto* cs = static_cast<const float*>(cos_sin);
    const auto* pos = static_cast<const int64_t*>(positions);
    const auto* sl = static_cast<const int64_t*>(slots);
    auto* kc = static_cast<uint8_t*>(k_caches);
    auto* vc = static_cast<uint8_t*>(v_caches);
    auto* o = static_cast<bf16*>(out);
    const int64_t cache_bytes = n_slots * kKvWidth;
    cudaError_t err = cudaSuccess;
    if (mode == 0) {
        for (int64_t c = 0, r = 0; c < chains && err == cudaSuccess; r += chain_rows[c], ++c) {
            const int64_t n = chain_rows[c];
            err = k6::prologue_rows_at(qg + r * kQGateWidth, kQGateWidth, kb + r * kKvWidth, kKvWidth,
                                       vb + r * kKvWidth, kKvWidth, qn, kn, eps, cs, sl + c * n_slots, pos + r, n,
                                       kc + c * cache_bytes, vc + c * cache_bytes, q_out + r * kAttnWidth,
                                       k_out + r * kKvWidth, gate_out + r * kAttnWidth, stream);
            if (err != cudaSuccess) break;
            err = k6::dense_attention_rows_at(q_out + r * kAttnWidth, gate_out + r * kAttnWidth, n, pos + r,
                                              kc + c * cache_bytes, vc + c * cache_bytes, sl + c * n_slots, ws,
                                              ws_bytes, o + r * kAttnWidth, kAttnWidth, stream);
        }
    } else {
        k1::KvRows kv{};
        for (int64_t c = 0, r = 0; c < chains; ++c) {
            for (int64_t j = 0; j < chain_rows[c]; ++j, ++r) {
                kv.k_cache[r] = kc + c * cache_bytes;
                kv.v_cache[r] = vc + c * cache_bytes;
                kv.slots[r] = sl + c * n_slots;
            }
        }
        err = k6::prologue_chains_at(qg, kQGateWidth, kb, kKvWidth, vb, kKvWidth, qn, kn, eps, cs, kv, pos, rows, q_out,
                                     k_out, gate_out, stream);
        if (err == cudaSuccess)
            err = k6::dense_attention_chains_at(q_out, gate_out, rows, pos, kv, ws, ws_bytes, o, kAttnWidth, stream);
    }
    const cudaError_t freed_ws = cudaFreeAsync(ws, stream);
    const cudaError_t freed_act = cudaFreeAsync(act, stream);
    if (err != cudaSuccess) return cere::ktest::status_of(err);
    return cere::ktest::status_of(freed_ws != cudaSuccess ? freed_ws : freed_act);
}

// The attention blocks per (row, key split) of k6's one-token FP8-cache launch (k6::split_blocks): a decode step or
// MTP extend (chain 0) or a verify chain of `rows` rows (chain 1, 1 <= rows <= 16), written to blocks (int64 [1],
// host). Else CERE_ERR_ARGUMENT.
extern "C" __attribute__((visibility("default"))) int cere_ktest_attn_split_blocks(int64_t rows, int64_t chain,
                                                                                  void* blocks,
                                                                                  cudaStream_t /*stream*/) {
    if (!blocks || chain < 0 || chain > 1 || rows < 1 || rows > 16 || (chain == 0 && rows != 1))
        return CERE_ERR_ARGUMENT;
    *static_cast<int64_t*>(blocks) = cere::k6::split_blocks(rows, chain == 1);
    return 0;
}
