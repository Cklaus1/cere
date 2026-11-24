// Compiled model support table: the exact configs, tensor dtypes and limits a device accepts (no CUDA includes).
#pragma once

#include "cere_executor.h"

#include <cstdint>

namespace cere::host {

struct Support {
    const char *name;
    cere_qwen4_exp_config config;  // every field must equal the desc's config bit for bit
    uint64_t max_sessions;         // upper bound of cere_model_desc.max_sessions
    uint64_t max_context_tokens;   // upper bound of cere_model_desc.max_context_tokens
    uint64_t max_kv_pool_tokens;   // upper bound of a non-zero cere_model_desc.kv_pool_tokens (host/kv_pool.h)
    uint32_t dtype_mask;           // bit (1 << CERE_DTYPE_*) per dtype allowed for DEVICE/HOST_PINNED tensors
    // Null = text-only (the desc's vision_config must be null); else it must equal the desc's bit for bit (M2.13).
    const cere_qwen4_exp_vision_config *vision;
};

// The A4 model (Qwen4ExpForConditionalGeneration NVFP4 text_config, m22 design note "Environment facts").
const Support &a4_support();

// The A4 model with its vision tower (m213 design note F2, ledger N): a4_support's values (m213l design note
// VL-L1: the same context and KV pool, any position for images) plus the vision_config of $ART/config.json.
const Support &a4_vision_support();

// Bitwise equality of every config field after the struct header.
bool config_matches(const Support &support, const cere_qwen4_exp_config &config);

// Null on both sides, or bitwise equality of every vision_config field after the struct header.
bool vision_matches(const Support &support, const cere_qwen4_exp_vision_config *vision);

// BF16 elements of one patch row of cere_image_input.pixel_patches: in_channels * temporal_patch * patch^2.
uint64_t patch_elements(const cere_qwen4_exp_vision_config &vision);

// Bytes per element of a CERE_DTYPE_* value; 0 for an unknown dtype.
uint64_t dtype_size(uint32_t dtype);

constexpr uint32_t dtype_bit(uint32_t dtype) { return uint32_t(1) << dtype; }

}  // namespace cere::host
