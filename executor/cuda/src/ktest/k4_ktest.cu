// K4 kernel-test entry points (executor/tests/gpu/test_kernels_k4_ple.py): the PLE n-gram hash, the mapped pinned
// table gather and one PLE block step. Device (or UVA-mapped pinned host) pointers, int64 sizes, the stream last;
// 0 = success, else a cere_status.
#include "kernels/k4_ple.h"
#include "ktest/k2_ktest.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace cere::ktest {
namespace {

// One block step with the caller's capacities: the K2 plans are pinned for max_m = plan_max_m (the executor pins
// the same per-bucket algorithms at INSTANTIATE for its max_m) and the workspace holds
// k4::workspace_bytes(workspace_tokens) bytes (0: no workspace, a null pointer).
cere_status ple_block(const void *query, const void *tokens, int64_t n_tokens, void *ngram_history, void *conv_state,
                      const void *table, int64_t n_table_rows, const void *table_scale, const void *multipliers,
                      const void *head_vocab_sizes, const void *head_offsets, const void *key_proj,
                      const void *value_proj, const void *norm_key, const void *norm_query, const void *norm_conv,
                      const void *conv_weight, int64_t eos_token_id, float eps, void *out, int64_t plan_max_m,
                      int64_t workspace_tokens, cudaStream_t stream) {
    if (plan_max_m <= 0 || plan_max_m > int64_t(UINT32_MAX) || workspace_tokens < 0) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = gemm_handle();
    if (handle == nullptr) {
        return CERE_ERR_INTERNAL;
    }
    kernels::GemmPlan key_plan;
    kernels::GemmPlan value_plan;
    cere_status s = key_plan.init(handle, k4::kChannels, k4::kEmbedDim, uint32_t(plan_max_m), kernels::GemmOut::kBF16);
    if (s != CERE_OK) {
        return s;
    }
    s = value_plan.init(handle, k4::kHidden, k4::kEmbedDim, uint32_t(plan_max_m), kernels::GemmOut::kBF16);
    if (s != CERE_OK) {
        return s;
    }

    using bf16 = __nv_bfloat16;
    const k4::Weights w{
        static_cast<const uint8_t *>(table),
        n_table_rows,
        static_cast<const bf16 *>(table_scale),
        static_cast<const int64_t *>(multipliers),
        static_cast<const int64_t *>(head_vocab_sizes),
        static_cast<const int64_t *>(head_offsets),
        static_cast<const bf16 *>(key_proj),
        static_cast<const bf16 *>(value_proj),
        static_cast<const bf16 *>(norm_key),
        static_cast<const bf16 *>(norm_query),
        static_cast<const bf16 *>(norm_conv),
        static_cast<const bf16 *>(conv_weight),
        eos_token_id,
        eps,
    };
    const k4::Plans plans{&key_plan, &value_plan};
    const k4::State state{static_cast<int64_t *>(ngram_history), static_cast<bf16 *>(conv_state)};

    const size_t workspace_size = k4::workspace_bytes(workspace_tokens);
    void *workspace = nullptr;
    if (workspace_size != 0 && cudaMallocAsync(&workspace, workspace_size, stream) != cudaSuccess) {
        return CERE_ERR_INTERNAL;
    }
    s = k4::forward(w, plans, state, static_cast<const int64_t *>(tokens), n_tokens, static_cast<const bf16 *>(query),
                    false, static_cast<bf16 *>(out), workspace, workspace_size, stream);
    const bool freed = workspace == nullptr || cudaFreeAsync(workspace, stream) == cudaSuccess;
    if (s != CERE_OK) {
        return s;
    }
    return freed ? CERE_OK : CERE_ERR_INTERNAL;
}

// rows16 design note R16-A: `m` rows of `sequences` sequences (host int64 counts [sequences], back to back) with
// their states in histories int64 [sequences, 2] / conv_states BF16 [sequences, 10240, 9] (advanced in place), the
// per-row saves in save_histories [m, 2] / save_convs [m, 10240, 9] where host int64 save_mask[r] is 1 (row r's
// sequence state as of after row r). mode 0: the reference, the executor's pre-R16-A path: k4::forward(t = 1) once
// per row in order on the instantiated plans (max_m plan_max_m), each save a D2D copy of the state after its row;
// mode 1: one k4::forward_rows on their rows plans (GemmPlan::init_rows at k4::kMaxRows). small_m: use_small_m on
// every plan (the k2_gemm lever); CERE_ERR_UNSUPPORTED when it does not take.
cere_status ple_rows(const void *query, const void *tokens, int64_t m, int64_t sequences, const int64_t *counts,
                     void *histories, void *conv_states, void *save_histories, void *save_convs,
                     const int64_t *save_mask, const k4::Weights &w, bool add_query, void *out, int64_t plan_max_m,
                     int64_t mode, bool small_m, cudaStream_t stream) {
    using bf16 = __nv_bfloat16;
    if (m <= 0 || m > int64_t(k4::kMaxRows) || sequences <= 0 || sequences > m || plan_max_m < m ||
        plan_max_m > int64_t(UINT32_MAX) || (mode != 0 && mode != 1) || counts == nullptr || save_mask == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    cublasLtHandle_t handle = gemm_handle();
    if (handle == nullptr) return CERE_ERR_INTERNAL;
    kernels::GemmPlan key_plan, value_plan, key_rows, value_rows;
    cere_status s = key_plan.init(handle, k4::kChannels, k4::kEmbedDim, uint32_t(plan_max_m), kernels::GemmOut::kBF16);
    if (s == CERE_OK) {
        s = value_plan.init(handle, k4::kHidden, k4::kEmbedDim, uint32_t(plan_max_m), kernels::GemmOut::kBF16);
    }
    if (s == CERE_OK && mode == 1) s = key_rows.init_rows(handle, key_plan, k4::kMaxRows);
    if (s == CERE_OK && mode == 1) s = value_rows.init_rows(handle, value_plan, k4::kMaxRows);
    if (s != CERE_OK) return s;
    kernels::GemmPlan *key = mode == 1 ? &key_rows : &key_plan;
    kernels::GemmPlan *value = mode == 1 ? &value_rows : &value_plan;
    key->use_small_m(small_m);
    value->use_small_m(small_m);
    if (small_m && (!key->small_m() || !value->small_m())) return CERE_ERR_UNSUPPORTED;
    const k4::Plans plans{key, value};

    constexpr size_t kConvElems = size_t(k4::kChannels) * k4::kConvStateLen;
    const auto state_of = [&](void *hist, void *conv, int64_t i) {
        return k4::State{static_cast<int64_t *>(hist) + i * k4::kHistoryLen, static_cast<bf16 *>(conv) + i * kConvElems};
    };
    const size_t workspace_size = k4::workspace_bytes(m);
    void *workspace = nullptr;
    if (cudaMallocAsync(&workspace, workspace_size, stream) != cudaSuccess) return CERE_ERR_INTERNAL;
    const auto *tok = static_cast<const int64_t *>(tokens);
    const auto *q = static_cast<const bf16 *>(query);
    auto *o = static_cast<bf16 *>(out);
    if (mode == 1) {
        k4::RowsCall call;
        call.sequences = uint32_t(sequences);
        int64_t first = 0;
        for (int64_t i = 0; i < sequences && s == CERE_OK; ++i) {
            if (counts[i] <= 0 || counts[i] > m - first) s = CERE_ERR_ARGUMENT;
            call.states[i] = state_of(histories, conv_states, i);
            call.first[i] = uint32_t(first);
            call.count[i] = uint32_t(counts[i]);
            first += counts[i];
        }
        for (int64_t r = 0; r < m; ++r) {
            if (save_mask[r] != 0) call.saves[r] = state_of(save_histories, save_convs, r);
        }
        if (s == CERE_OK) {
            s = k4::forward_rows(w, plans, call, tok, m, q, add_query, o, workspace, workspace_size, stream);
        }
    } else {
        int64_t r = 0;
        for (int64_t i = 0; i < sequences && s == CERE_OK; ++i) {
            const k4::State st = state_of(histories, conv_states, i);
            for (int64_t j = 0; j < counts[i] && s == CERE_OK && r < m; ++j, ++r) {
                const uint64_t at = uint64_t(r) * k4::kChannels;
                s = k4::forward(w, plans, st, tok + r, 1, q + at, add_query, o + at, workspace, workspace_size, stream);
                if (s == CERE_OK && save_mask[r] != 0) {
                    const k4::State dst = state_of(save_histories, save_convs, r);
                    if (cudaMemcpyAsync(dst.history, st.history, k4::kHistoryLen * sizeof(int64_t),
                                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
                        cudaMemcpyAsync(dst.conv_state, st.conv_state, kConvElems * sizeof(bf16),
                                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                        s = CERE_ERR_INTERNAL;
                    }
                }
            }
        }
        if (s == CERE_OK && r != m) s = CERE_ERR_ARGUMENT;
    }
    const bool freed = cudaFreeAsync(workspace, stream) == cudaSuccess;
    if (s != CERE_OK) return s;
    return freed ? CERE_OK : CERE_ERR_INTERNAL;
}

}  // namespace
}  // namespace cere::ktest

extern "C" __attribute__((visibility("default"))) int cere_ktest_ple_hash(const void *windows, const void *multipliers,
                                                                          const void *head_vocab_sizes,
                                                                          const void *head_offsets, void *ids,
                                                                          int64_t n, int64_t eos_token_id,
                                                                          cudaStream_t stream) {
    return cere::k4::ngram_hash_windows(static_cast<const int64_t *>(windows), n,
                                        static_cast<const int64_t *>(multipliers),
                                        static_cast<const int64_t *>(head_vocab_sizes),
                                        static_cast<const int64_t *>(head_offsets), eos_token_id,
                                        static_cast<int64_t *>(ids), stream);
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_ple_gather(const void *table, int64_t n_table_rows,
                                                                            const void *ids, int64_t n, void *out,
                                                                            cudaStream_t stream) {
    return cere::k4::gather(static_cast<const uint8_t *>(table), n_table_rows, static_cast<const int64_t *>(ids), n,
                            nullptr, static_cast<__nv_bfloat16 *>(out), stream);
}

// FL-13 (fp4local design note D8): the NVFP4 row decode, ids outside [0, n_table_rows) give zero rows.
extern "C" __attribute__((visibility("default"))) int cere_ktest_ple_gather_nvfp4(const void *table,
                                                                                  const void *block_scales,
                                                                                  const void *global_scale,
                                                                                  int64_t n_table_rows,
                                                                                  const void *ids, int64_t n,
                                                                                  void *out, cudaStream_t stream) {
    return cere::k4::gather_nvfp4(static_cast<const uint8_t *>(table), static_cast<const uint8_t *>(block_scales),
                                  static_cast<const float *>(global_scale), n_table_rows,
                                  static_cast<const int64_t *>(ids), n, static_cast<__nv_bfloat16 *>(out), stream);
}

// The forward's embedding stage (ids from history ++ tokens, then the 16 rows of each token) of a table of `format`
// (0 FP8, 1 NVFP4) with whatever table fields the caller passes: check_table's refusals are observable here. The
// history is read, not advanced.
extern "C" __attribute__((visibility("default"))) int cere_ktest_ple_embed(
    uint32_t format, const void *table, int64_t n_table_rows, const void *table_scale, const void *block_scales,
    const void *global_scale, const void *multipliers, const void *head_vocab_sizes, const void *head_offsets,
    const void *ngram_history, const void *tokens, int64_t n_tokens, int64_t eos_token_id, void *emb,
    cudaStream_t stream) {
    if (format > uint32_t(cere::k4::TableFormat::kNvfp4)) return CERE_ERR_ARGUMENT;
    cere::k4::Weights w{};
    w.table = static_cast<const uint8_t *>(table);
    w.table_rows = n_table_rows;
    w.table_scale = static_cast<const __nv_bfloat16 *>(table_scale);
    w.multipliers = static_cast<const int64_t *>(multipliers);
    w.head_vocab_sizes = static_cast<const int64_t *>(head_vocab_sizes);
    w.head_offsets = static_cast<const int64_t *>(head_offsets);
    w.eos_token_id = eos_token_id;
    w.format = cere::k4::TableFormat(format);
    w.block_scales = static_cast<const uint8_t *>(block_scales);
    w.global_scale = static_cast<const float *>(global_scale);
    return cere::k4::embed(w, static_cast<const int64_t *>(ngram_history), static_cast<const int64_t *>(tokens),
                           n_tokens, static_cast<__nv_bfloat16 *>(emb), stream);
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_ple_block(
    const void *query, const void *tokens, int64_t n_tokens, void *ngram_history, void *conv_state, const void *table,
    int64_t n_table_rows, const void *table_scale, const void *multipliers, const void *head_vocab_sizes,
    const void *head_offsets, const void *key_proj, const void *value_proj, const void *norm_key,
    const void *norm_query, const void *norm_conv, const void *conv_weight, int64_t eos_token_id, float eps, void *out,
    cudaStream_t stream) {
    return cere::ktest::ple_block(query, tokens, n_tokens, ngram_history, conv_state, table, n_table_rows,
                                  table_scale, multipliers, head_vocab_sizes, head_offsets, key_proj, value_proj,
                                  norm_key, norm_query, norm_conv, conv_weight, eos_token_id, eps, out, n_tokens,
                                  n_tokens, stream);
}

extern "C" __attribute__((visibility("default"))) int cere_ktest_ple_block_planned(
    const void *query, const void *tokens, int64_t n_tokens, void *ngram_history, void *conv_state, const void *table,
    int64_t n_table_rows, const void *table_scale, const void *multipliers, const void *head_vocab_sizes,
    const void *head_offsets, const void *key_proj, const void *value_proj, const void *norm_key,
    const void *norm_query, const void *norm_conv, const void *conv_weight, int64_t eos_token_id, float eps, void *out,
    int64_t plan_max_m, int64_t workspace_tokens, cudaStream_t stream) {
    return cere::ktest::ple_block(query, tokens, n_tokens, ngram_history, conv_state, table, n_table_rows,
                                  table_scale, multipliers, head_vocab_sizes, head_offsets, key_proj, value_proj,
                                  norm_key, norm_query, norm_conv, conv_weight, eos_token_id, eps, out, plan_max_m,
                                  workspace_tokens, stream);
}

// rows16 design note R16-A (test_block_rows_bit_exact): cere::ktest::ple_rows on an FP8 table (k4::Weights as
// cere_ktest_ple_block's); counts and save_mask are host int64 arrays.
extern "C" __attribute__((visibility("default"))) int cere_ktest_ple_rows(
    const void *query, const void *tokens, int64_t m, int64_t sequences, const void *counts, void *histories,
    void *conv_states, void *save_histories, void *save_convs, const void *save_mask, const void *table,
    int64_t n_table_rows, const void *table_scale, const void *multipliers, const void *head_vocab_sizes,
    const void *head_offsets, const void *key_proj, const void *value_proj, const void *norm_key,
    const void *norm_query, const void *norm_conv, const void *conv_weight, int64_t eos_token_id, float eps,
    int64_t add_query, void *out, int64_t plan_max_m, int64_t mode, int64_t small_m, cudaStream_t stream) {
    using bf16 = __nv_bfloat16;
    const cere::k4::Weights w{
        static_cast<const uint8_t *>(table),
        n_table_rows,
        static_cast<const bf16 *>(table_scale),
        static_cast<const int64_t *>(multipliers),
        static_cast<const int64_t *>(head_vocab_sizes),
        static_cast<const int64_t *>(head_offsets),
        static_cast<const bf16 *>(key_proj),
        static_cast<const bf16 *>(value_proj),
        static_cast<const bf16 *>(norm_key),
        static_cast<const bf16 *>(norm_query),
        static_cast<const bf16 *>(norm_conv),
        static_cast<const bf16 *>(conv_weight),
        eos_token_id,
        eps,
    };
    return cere::ktest::ple_rows(query, tokens, m, sequences, static_cast<const int64_t *>(counts), histories,
                                 conv_states, save_histories, save_convs, static_cast<const int64_t *>(save_mask), w,
                                 add_query != 0, out, plan_max_m, mode, small_m != 0, stream);
}
