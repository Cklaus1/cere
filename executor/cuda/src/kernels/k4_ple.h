// K4 (m22 design note): the Qwen4-Exp PLE n-gram path of layer 1 (0-based; config `ple_layer_ids` [2] is 1-based,
// SGLang builds PLE where `(layer_id + 1) in ple_layer_ids`, qwen4_exp.py:1241).
//
// math_reference: sglang
//   python/sglang/srt/models/qwen4_exp.py: `Qwen4ExpPLEGroupedNorm` :376, `_hash_contexts` :603,
//   `_shift_right_ignore_eos` :658, `_gather_ple_embedding_from_pinned_kernel` :718, `Qwen4ExpPinnedHostEmbedding`
//   :750-870, `_short_conv` :955, `Qwen4ExpPLELayer.forward` :1162-1218, `_prepare_qwen4_exp_attn` :1276-1314;
//   python/sglang/kernels/ops/qwen4_ple.py (hash, gate/value, short-conv state).
//
// One PLE step over `t` new tokens of one sequence (forward):
//   ids[t, h]  = n-gram hash of the window (ctx[t], ctx[t+1], ctx[t+2]), ctx = history[2] ++ tokens (16 heads)
//   emb[t]     = bf16(fp8 table[ids[t, h]]) * scale, the 16 rows of 160 concatenated (2,560); an NVFP4 table
//                (fp4local design note D8) decodes each row instead (k4_nvfp4.h: E2M1 x E4M3 x F32 global, RNE)
//   key        = emb @ key_proj^T    [t, 4 * 2560]      value = emb @ value_proj^T  [t, 2560]
//   kn, qn     = grouped Gemma RMSNorm (group 2,560) of key and of the query (the 4-stream HC state; the residual
//                is None at every layer entry, qwen4_exp.py:1384-1385, so the query is the HC state itself)
//   gate[t, g] = sigmoid(signed_sqrt(sum(kn[t, g] * qn[t, g]) / sqrt(2560)))
//   gated      = gate[t, g] * value[t]                 gn = grouped Gemma RMSNorm(gated)
//   conv       = silu(depthwise conv1d(gn), kernel 4, dilation 3, over [conv_state (9) | gn])
//   delta      = gated + conv;   the layer adds it to the HC state before `attn_hyper_connection.mix`.
// History and conv state advance in place. Every intermediate is rounded to BF16 where SGLang's eager BF16 graph
// rounds (the fused decode kernels of qwen4_ple.py reproduce the same boundaries).
#pragma once

#include "cere_executor.h"
#include "cere_model_constants.h"
#include "kernels/k2_gemm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cere::k4 {

constexpr int kHidden = gen::kHiddenSize;             // hidden_size
constexpr int kHcCount = gen::kHcCount;               // hc_count
constexpr int kChannels = kHidden * kHcCount;         // 10,240: key / query / conv channels
constexpr int kNgramSize = gen::kNgramSize;           // ngram_size
constexpr int kHeadsPerNgram = gen::kHeadsPerNgram;   // heads_per_ngram
constexpr int kNgramHeads = (kNgramSize - 1) * kHeadsPerNgram;  // 16
constexpr int kEmbedDim = gen::kPleEmbedDim;          // ple_embed_dim
constexpr int kHeadDim = kEmbedDim / kNgramHeads;     // 160 FP8 bytes per table row
constexpr int kConvKernel = gen::kPleConvKernelSize;  // ple_conv_kernel_size
constexpr int kConvDilation = kNgramSize;             // short_conv_dilation = ngram_size
constexpr int kConvStateLen = (kConvKernel - 1) * kConvDilation;  // 9
constexpr int kHistoryLen = kNgramSize - 1;           // 2
// text_config.eos_token_id of the A4 model (SGLang's PLE EOS, qwen4_exp.py:460, :1607).
constexpr int64_t kEosTokenId = gen::kEosTokenId;

// The storage of the PLE table (fp4local design note D8, D14). kFp8 is the shipped checkpoint's (artifact profile
// `rtxpro6000bw-plehost-text-v1`): E4M3 rows times one BF16 scale. kNvfp4 is the FP4-Local checkpoint's (artifact
// profile `rtxpro6000bw-plehost-nvfp4-text-v1`, the only profile whose artifact carries such a table): 80 packed E2M1
// bytes + 10 E4M3 block scales per row and one F32 global (k4_nvfp4.h). The binder (model.cpp) picks it from the
// artifact's tensors; check_table refuses fields of the other format, so neither table reaches the other's decode.
enum class TableFormat : uint32_t { kFp8 = 0, kNvfp4 = 1 };

// Device-accessible weights and tables of one PLE layer.
struct Weights {
    const uint8_t *table;               // kFp8: E4M3 [table_rows, 160]; kNvfp4: packed E2M1 [table_rows, 80]
                                        // (mapped pinned host memory)
    int64_t table_rows;
    const __nv_bfloat16 *table_scale;   // kFp8: [1]; kNvfp4: null
    const int64_t *multipliers;         // [3] layer_multipliers
    const int64_t *head_vocab_sizes;    // [16] ngram_heads_vocab_sizes
    const int64_t *head_offsets;        // [16] ngram_heads_offsets
    const __nv_bfloat16 *key_proj;      // [10240, 2560]
    const __nv_bfloat16 *value_proj;    // [2560, 2560]
    const __nv_bfloat16 *norm_key;      // [10240]
    const __nv_bfloat16 *norm_query;    // [10240]
    const __nv_bfloat16 *norm_conv;     // [10240]
    const __nv_bfloat16 *conv_weight;   // [10240, 4] (checkpoint [10240, 1, 4])
    int64_t eos_token_id;
    float eps;
    TableFormat format = TableFormat::kFp8;
    const uint8_t *block_scales = nullptr;  // kNvfp4: E4M3 [table_rows, 10] (mapped pinned host memory); kFp8: null
    const float *global_scale = nullptr;    // kNvfp4: F32 [1] (mapped pinned host memory); kFp8: null
};

// Per-sequence state.
struct State {
    int64_t *history;            // [2] the two previous tokens, EOS for a new sequence
    __nv_bfloat16 *conv_state;   // [10240, 9] columns oldest -> newest, zero for a new sequence
};

// The K2 plans of the two projections (created at INSTANTIATE with max_m >= the largest step): key n = 10,240,
// value n = 2,560, both k = 2,560, BF16 output. `cls` as kernels::GemmPlan::run (m210 D15⁗′; 0 = the step's t).
struct Plans {
    const kernels::GemmPlan *key;
    const kernels::GemmPlan *value;
    uint32_t cls = 0;
};

// An end capture inside a step (m210 D15⁗′): the State as of `row` (9 <= row < t) written to `history` [2] and
// `conv_state` [10240, 9]; the step's own state and output are unchanged.
struct Capture {
    int64_t row;
    int64_t *history;
    __nv_bfloat16 *conv_state;
};

// Every function returns CERE_OK, CERE_ERR_ARGUMENT for invalid sizes, or CERE_ERR_INTERNAL for a launch failure.

// Ids of `n` explicit windows (t-2, t-1, t), as `_hash_contexts` on a [n, 3] context tensor.
cere_status ngram_hash_windows(const int64_t *windows, int64_t n, const int64_t *multipliers,
                               const int64_t *head_vocab_sizes, const int64_t *head_offsets, int64_t eos_token_id,
                               int64_t *ids, cudaStream_t stream);

// out[i] = bf16(fp8 table[ids[i]]) (times *scale when scale is not null); ids outside [0, table_rows) give zero rows.
cere_status gather(const uint8_t *table, int64_t table_rows, const int64_t *ids, int64_t n,
                   const __nv_bfloat16 *scale, __nv_bfloat16 *out, cudaStream_t stream);

// CERE_OK when the table fields of `w` are those of its format: a table and table_rows >= 0; kFp8 with no block
// scales and no global; kNvfp4 with no BF16 table_scale, non-null block_scales and global_scale, and an 8-byte aligned
// table. Else CERE_ERR_ARGUMENT (forward and embed check it before any launch).
cere_status check_table(const Weights &w);

// The NVFP4 gather (D8): out[i] = row ids[i] of the packed table decoded as k4_nvfp4.h, BF16 [n, 160]; an id outside
// [0, table_rows) decodes zero bytes (code 0, scale 0x00: +0 for a finite global), like the FP8 gather's zero rows.
// table 8-byte aligned; CERE_ERR_ARGUMENT for a null table, scale or global.
cere_status gather_nvfp4(const uint8_t *table, const uint8_t *block_scales, const float *global_scale,
                         int64_t table_rows, const int64_t *ids, int64_t n, __nv_bfloat16 *out, cudaStream_t stream);

// The embedding stage of forward on its own, by the same kernel: emb [t, 2560] of `t` >= 1 tokens after `history`
// [2] (left unchanged), each token's 16 n-gram rows decoded per w.format. CERE_ERR_ARGUMENT before any launch when
// check_table refuses or a pointer is null.
cere_status embed(const Weights &w, const int64_t *history, const int64_t *tokens, int64_t t, __nv_bfloat16 *emb,
                  cudaStream_t stream);

// Workspace bytes `forward` needs for up to `max_tokens` tokens per call.
size_t workspace_bytes(int64_t max_tokens);

// One PLE step over `t` >= 1 tokens. `query` is the HC state [t, 10240]. Writes `out` [t, 10240]:
//   out = query + delta when add_query (the model path; out may alias query), else out = delta.
// Advances state.history and state.conv_state in place. `workspace` is 256-B aligned and holds `workspace_size` >=
// workspace_bytes(t) bytes. CERE_ERR_ARGUMENT before any launch (history, conv state and out untouched) when t exceeds
// either plan's max_m, the workspace is missing, misaligned or short, a plan has the wrong shape, or a pointer is null
// or not 16-B aligned where K1/K2 require it, or check_table refuses the table fields.
cere_status forward(const Weights &w, const Plans &plans, const State &state, const int64_t *tokens, int64_t t,
                    const __nv_bfloat16 *query, bool add_query, __nv_bfloat16 *out, void *workspace,
                    size_t workspace_size, cudaStream_t stream, const Capture *capture = nullptr);

// Rows of one forward_rows call at most (kernels::GemmPlan::kMaxRows: a verify call's Σ(n + 1) <= 16).
constexpr uint32_t kMaxRows = 16;

// The sequences of one forward_rows call (rows16 design note R16-A): sequence s owns rows [first[s], first[s] +
// count[s]) of the call, back to back in order (first[0] = 0, first[s + 1] = first[s] + count[s], Σ count = m), with
// state states[s]. saves[r] (both pointers null: none): where the state of row r's sequence as of after row r goes
// (written in-kernel; the verify undo snapshot of m29 design note M29S2-20); never row r's sequence's own state.
struct RowsCall {
    uint32_t sequences = 0;
    State states[kMaxRows] = {};
    uint32_t first[kMaxRows] = {};
    uint32_t count[kMaxRows] = {};
    State saves[kMaxRows] = {};
};

// forward over `m` rows of several sequences at once: tokens / query / out [m, ...] as forward's, row r of sequence s
// after the rows of s before it. With `plans` the rows plans (GemmPlan::init_rows) of forward's plans, row r's out
// is bit-equal to forward(t = 1) on row r after forward(t = 1) on each earlier row of its sequence (same kernels and
// arithmetic per row; only the launches are shared), and every sequence's state ends as forward over its count[s]
// tokens leaves it. CERE_ERR_ARGUMENT before any launch for m outside [1, kMaxRows] or above a plan's max_m, a call
// whose sequences do not tile [0, m), and forward's other refusals.
cere_status forward_rows(const Weights &w, const Plans &plans, const RowsCall &call, const int64_t *tokens, int64_t m,
                         const __nv_bfloat16 *query, bool add_query, __nv_bfloat16 *out, void *workspace,
                         size_t workspace_size, cudaStream_t stream);

}  // namespace cere::k4
