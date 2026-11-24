// Track LoRA (lora design note): the qwen4_exp targets of cere_model_lora_load (cere_executor.h) and the checks
// every device shares. A target id is layer * CERE_LORA_MODULES + module; A is BF16 [rank, in], B BF16 [out, rank].
#pragma once

#include "cere_executor.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace cere::host::lora {

// Devices store an adapter's rank padded to a multiple of kRankAlign (zero rows of A, zero columns of B), so any rank
// in [1, CERE_LORA_MAX_RANK] runs the same kernels.
constexpr uint32_t kRankAlign = 8;
constexpr uint32_t kModules = 7;  // CERE_LORA_Q_PROJ .. CERE_LORA_SHARED_DOWN
inline uint32_t padded_rank(uint32_t rank) { return (rank + kRankAlign - 1) / kRankAlign * kRankAlign; }

inline bool full_attention_layer(const cere_qwen4_exp_config &cfg, uint32_t layer) {
    return layer < 64 && ((cfg.full_attention_layers >> layer) & 1u) != 0;
}

struct Dims {
    uint32_t in = 0;
    uint32_t out = 0;
};

// The input and output widths of `target`; false for an unknown target (a module past CERE_LORA_SHARED_DOWN, a layer
// past num_hidden_layers, an attention module of a linear-attention layer).
inline bool dims(const cere_qwen4_exp_config &cfg, uint32_t target, Dims *out) {
    const uint32_t layer = target / CERE_LORA_MODULES;
    const uint32_t module = target % CERE_LORA_MODULES;
    if (layer >= cfg.num_hidden_layers || module >= kModules) return false;
    const uint32_t attn = cfg.num_attention_heads * cfg.head_dim;
    const uint32_t kv = cfg.num_key_value_heads * cfg.head_dim;
    const uint32_t h = cfg.hidden_size;
    const uint32_t s = cfg.shared_expert_intermediate_size;
    if (module <= CERE_LORA_O_PROJ && !full_attention_layer(cfg, layer)) return false;
    switch (module) {
    case CERE_LORA_Q_PROJ: *out = Dims{h, 2 * attn}; return true;  // q and its output gate, HF row order
    case CERE_LORA_K_PROJ:
    case CERE_LORA_V_PROJ: *out = Dims{h, kv}; return true;
    case CERE_LORA_O_PROJ: *out = Dims{attn, h}; return true;
    case CERE_LORA_SHARED_GATE:
    case CERE_LORA_SHARED_UP: *out = Dims{h, s}; return true;
    default: *out = Dims{s, h}; return true;  // CERE_LORA_SHARED_DOWN
    }
}

// Every target of the model, ascending ids (the pool's slot layout).
inline std::vector<uint32_t> targets(const cere_qwen4_exp_config &cfg) {
    std::vector<uint32_t> out;
    Dims d;
    for (uint32_t t = 0; t < cfg.num_hidden_layers * CERE_LORA_MODULES; ++t) {
        if (dims(cfg, t, &d)) out.push_back(t);
    }
    return out;
}

// BF16 elements of one slot per unit of padded rank: sum over targets of in + out (A rows and B columns).
inline uint64_t elements_per_rank(const cere_qwen4_exp_config &cfg) {
    uint64_t n = 0;
    Dims d;
    for (uint32_t t : targets(cfg)) {
        dims(cfg, t, &d);
        n += uint64_t(d.in) + d.out;
    }
    return n;
}

// Pool bytes of `adapters` slots of rank up to `rank`.
inline uint64_t pool_bytes(const cere_qwen4_exp_config &cfg, uint32_t adapters, uint32_t rank) {
    return uint64_t(adapters) * padded_rank(rank) * elements_per_rank(cfg) * 2;
}

// cere_model_lora_load's argument checks (CERE_ERR_ARGUMENT on any failure): rank in [1, max_rank], a finite scale,
// count in [1, targets], every target known and given once, every A and B pointer non-null.
inline cere_status check_load(const cere_qwen4_exp_config &cfg, uint32_t max_rank, uint32_t rank, float scale,
                              uint64_t count, const uint32_t *targets, const void *const *a, const void *const *b) {
    if (rank == 0 || rank > max_rank || !std::isfinite(scale) || count == 0 || targets == nullptr || a == nullptr ||
        b == nullptr) {
        return CERE_ERR_ARGUMENT;
    }
    std::vector<bool> seen(size_t(cfg.num_hidden_layers) * CERE_LORA_MODULES, false);
    if (count > seen.size()) return CERE_ERR_ARGUMENT;
    Dims d;
    for (uint64_t i = 0; i < count; ++i) {
        if (!dims(cfg, targets[i], &d) || seen[targets[i]] || a[i] == nullptr || b[i] == nullptr) {
            return CERE_ERR_ARGUMENT;
        }
        seen[targets[i]] = true;
    }
    return CERE_OK;
}

}  // namespace cere::host::lora
