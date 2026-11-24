// The k2_weights lever's INSTANTIATE step (track Fp8W, kernels/k2_fp8.h): converts the dense projections to FP8.
#include "model/model.h"

#include <algorithm>
#include <vector>

namespace cere::model {
namespace {

constexpr size_t kAlign = 256;

size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

}  // namespace

cere_status convert_fp8_weights(ModelWeights *weights, Workspace *ws, cudaStream_t stream) {
    if (weights == nullptr || ws == nullptr || ws->scratch == nullptr || ws->fp8.started || !ws->fp8.empty()) {
        return CERE_ERR_ARGUMENT;
    }
    // Track Fp4S3 (FL-30): MXFP8 weights are not BF16 (bind_weights refuses the lever on them first).
    if (!weights->mxfp8.empty()) return CERE_ERR_UNSUPPORTED;
    // The tied-embedding case would convert the gather's table: not this model's layout.
    if (weights->lm_head == nullptr || weights->lm_head == weights->embed) {
        return CERE_ERR_UNSUPPORTED;
    }
    struct Item {
        const bf16 *w;
        uint32_t n;
        uint32_t k;
    };
    std::vector<Item> items;
    // A plan this model never initialized (e.g. attn_index_qk without the QSA indexer) never runs its weight.
    const auto add = [&items](const bf16 *w, const kernels::GemmPlan &plan) {
        if (w != nullptr && plan.n() != 0) items.push_back(Item{w, plan.n(), plan.k()});
    };
    for (uint32_t l = 0; l < kLayers; ++l) {
        const LayerWeights &lw = weights->layers[l];
        if (lw.full_attention) {
            add(lw.attn.qkv, ws->attn_qkv);
            add(lw.attn.o_proj, ws->attn_o);
            add(lw.attn.indexer_qk, ws->attn_index_qk);
        } else {
            add(lw.gdn.in_proj_qkv, ws->gdn_qkv);
            add(lw.gdn.in_proj_z, ws->gdn_z);
            add(lw.gdn.in_proj_b, ws->gdn_ba);
            add(lw.gdn.in_proj_a, ws->gdn_ba);
            add(lw.gdn.out_proj, ws->gdn_out);
        }
        add(lw.moe.shared_gate, ws->moe_shared_gate_up);
        add(lw.moe.shared_up, ws->moe_shared_gate_up);
        add(lw.moe.shared_down, ws->moe_shared_down);
    }
    add(weights->lm_head, ws->lm_head);
    for (const Item &it : items) {
        if (it.n == 0 || !kernels::k2f::supported(it.k)) return CERE_ERR_UNSUPPORTED;
    }
    // The m > kMaxM staging: the idle tail of lm_head's storage (2nk bytes, nk + 4n of them used). It must hold the
    // largest converted weight: checked before the conversion starts (MF-4), so an UNSUPPORTED here leaves the BF16
    // weights intact instead of half converted.
    const size_t n = ws->lm_head.n();
    const size_t k = ws->lm_head.k();
    const size_t used = align_up(n * k + 4 * n);
    size_t largest = 0;
    for (const Item &it : items) {
        if (it.w != weights->lm_head) largest = std::max(largest, size_t{it.n} * it.k * sizeof(bf16));
    }
    if (2 * n * k < used || 2 * n * k - used < largest) return CERE_ERR_UNSUPPORTED;
    // From here the BF16 bytes go: a failure leaves the model unusable (INSTANTIATE fails and is not retried), and is
    // INTERNAL.
    ws->fp8.started = true;
    for (const Item &it : items) {
        void *storage = const_cast<bf16 *>(it.w);
        if (kernels::k2f::convert_in_place(storage, it.n, it.k, ws->scratch, ws->scratch_bytes, stream) !=
            cudaSuccess) {
            return CERE_ERR_INTERNAL;
        }
        ws->fp8.add(it.w, it.n, it.k);
    }
    auto *base = reinterpret_cast<uint8_t *>(const_cast<bf16 *>(weights->lm_head));
    ws->fp8.set_staging(reinterpret_cast<bf16 *>(base + used), 2 * n * k - used);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return CERE_ERR_INTERNAL;
    return CERE_OK;
}

}  // namespace cere::model
