// VulkanAmd P3 T2 (device half): the model's tensors bound by name to buffer device addresses, and the per-layer weight
// structs of the Vulkan kernels. The Vulkan restatement of executor/cuda/src/model/model.cpp bind_weights (Binder
// :137-237, bind_hc :243-253, bind_mtp :257-317, bind_derived :320-332, bind_weights :673-962), which cannot be reused:
// model.h's ModelWeights holds CUDA pointer types and includes cuBLASLt (design section 0, OD-1).
//
// Pure host arithmetic over (plan, VkLayout, block addresses): no Vulkan call, no tensor byte read, so it runs at
// ALLOCATE (as CUDA binds at ALLOCATE, cuda_device.cpp allocate -> bind_weights) and a model the forward could not run
// is refused before any byte is uploaded. The same function is used by the host-only bind test with made-up addresses.
//
// What the forward (task T5/T6) reads from here: VkModelWeights. Every address is block.addr + offset of the tensor's
// one piece (vk_locate); runs (expert arrays, q|k|v and its MXFP8 scale run) are checked contiguous. MXFP8 dense
// projections (track Fp4S3, model/mxfp8_plan.h) carry their UE8M0 scale address in VkDense::scales (0 = BF16).
#pragma once

#include "cere_executor.h"
#include "device/vk_memory_plan.h"
#include "host/device.h"
#include "kernels/k13_moe_bf16_vk.h"
#include "kernels/k4_ple_vk.h"
#include "kernels/k5_gdn_layer_vk.h"
#include "kernels/k7_moe_vk.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cere::vkdev {

constexpr uint32_t kMaxLayers = 64;

// One dense GEMM weight [n, k]: BF16 (scales 0) or MXFP8 (E4M3 weight + UE8M0 scales [n, k / 32]); `info` is its
// prepared K2 row info (VkLayout::winfo / kMoeInfo, written at INSTANTIATE; 0 = not prepared: the K2 call scans w).
struct VkDense {
    uint64_t w = 0;
    uint64_t scales = 0;
    uint32_t n = 0, k = 0;
    uint64_t info = 0;
    bool mxfp8() const { return scales != 0; }
};

// One GatedResidual (hyper-connection): hc_norm [hc*H], input_mix_weight_down [lowrank, hc*H], input_mix_weight_up
// [hc*H, lowrank], block_inject_weight [hc, hc*H] (0 for a final mixer). BF16.
struct VkHcWeights {
    uint64_t norm = 0, down = 0, up = 0, inject = 0;
};

// Full-attention mixer: qkv = q|k|v packed [13,312, H] (one GEMM), q_norm / k_norm [256], o_proj [H, 6,144], the QSA
// indexer index_qk_proj [640, H] and its q / k layernorms [128].
struct VkAttentionWeights {
    VkDense qkv, o_proj, indexer_qk;
    uint64_t q_norm = 0, k_norm = 0, indexer_q_norm = 0, indexer_k_norm = 0;
};

// Checkpoint sources of the MoE derived values (F32 [E] each, packed runs).
struct VkMoeScaleSources {
    uint64_t gate_input_scale = 0, up_input_scale = 0, down_input_scale = 0, gate_weight_scale_2 = 0,
             down_weight_scale_2 = 0;
};

struct VkLayerWeights {
    bool full_attention = false;
    VkHcWeights attn_hc, mlp_hc;
    VkAttentionWeights attn;  // full-attention layers only
    // GDN layers only. `gdn` is the vkk5 view (A_log -> the derived F32 copy, info -> the layer's K2 row info);
    // gdn_dense[] the five projections with their formats (qkv, z, b, a, out): vkk5::LayerWeights has no scale field,
    // so an MXFP8 projection (the S3 artifact's) is visible only here (see vulkan_device.h "T5 interface").
    vkk5::LayerWeights gdn{};
    VkDense gdn_dense[5];
    uint64_t gdn_A_log_bf16 = 0;
    // MoE. `moe` is the vkk7 view (alphas -> derived, a1/a2 set at INSTANTIATE, bf16_info -> kMoeInfo of this layer);
    // shared_dense[] the shared expert's gate / up / down with their formats (MXFP8 on S3; vkk7::MoeLayer has no scale
    // field either).
    vkk7::MoeLayer moe{};
    VkDense shared_dense[3];
    VkMoeScaleSources moe_scales;
};

// The MTP draft layer (bound only for a plan with drafts).
struct VkMtpWeights {
    VkDense fc_embedding, fc_hidden;
    uint64_t norm_embedding = 0, norm_hidden = 0;
    VkHcWeights attn_hc, mlp_hc, mixer;
    VkAttentionWeights attn;
    vkk13::MoeBf16Layer moe{};
    uint64_t lm_head = 0;  // derived hot lm_head BF16 [count, H]
    uint64_t map = 0;      // derived u32 [count]
    uint32_t count = 0;
};

struct VkModelWeights {
    cere_qwen4_exp_config config{};
    uint64_t max_context_tokens = 0;
    uint64_t embed = 0;    // BF16 [V, H]
    uint64_t lm_head = 0;  // BF16 [V, H]
    VkHcWeights final_mixer;
    VkLayerWeights layers[kMaxLayers];
    uint32_t ple_layer = 0;
    vkk4::Weights ple{};   // table = the split table's TableView (desc = the IO block table), proj_info -> winfo
    uint64_t rope_cos_sin = 0;  // derived F32 [max_context_tokens, rotary_dim]
    bool has_draft = false;
    VkMtpWeights mtp;
    uint32_t mxfp8_tensors = 0;  // dense projections bound as MXFP8
};

// Binds every tensor the forward reads. block_addr[b] = device address of layout.blocks[b]. CERE_ERR_UNSUPPORTED with
// *error naming the first missing / mismatched tensor (as CUDA's Binder), CERE_ERR_ARGUMENT for a malformed call,
// CERE_ERR_CAPACITY on std::bad_alloc. `out` is written only on CERE_OK.
cere_status vk_bind_weights(const host::ModelPlan &plan, const VkLayout &layout, const std::vector<uint64_t> &block_addr,
                            VkModelWeights *out, std::string *error);

// The device address of tensor `i`'s first byte (0 for DISCARD / out of range).
uint64_t vk_tensor_addr(const VkLayout &layout, const std::vector<uint64_t> &block_addr, uint64_t i);

// The device address of region (kind, index), 0 when the layout has none.
uint64_t vk_region_addr(const VkLayout &layout, const std::vector<uint64_t> &block_addr, VkRegionKind kind,
                        uint32_t index = 0);

// One session slot carved at `slot` (address of the kSession region, carved for the session's context as CUDA's
// bind_session does, model.cpp:1004-1023), plus the pool entries every session shares.
struct VkSessionState {
    uint64_t slot = 0, slot_bytes = 0;
    uint64_t max_context_tokens = 0;
    uint64_t slot_map = 0;                 // int64 [ctx]
    vkk5::LayerState gdn[kMaxLayers] = {};  // conv, ssm per GDN layer
    vkk4::State ple{};                     // history int64 [2], conv BF16 [10240, 9]
    uint64_t qsa_ring[kMaxLayers] = {};    // BF16 [4, 128] per full-attention layer
    uint64_t out_counts = 0;               // u32 [vocab]
    uint64_t mtp_qsa_ring = 0, proposals = 0;  // drafts only
    uint64_t k_cache[kMaxLayers] = {}, v_cache[kMaxLayers] = {}, qsa_keys[kMaxLayers] = {};
    uint64_t mtp_k_cache = 0, mtp_v_cache = 0, mtp_qsa_keys = 0;
};

// Carves the slot at slot_addr for max_context_tokens (<= the plan's) and copies the pool entries of the layout.
cere_status vk_bind_session(const host::ModelPlan &plan, const VkLayout &layout, const std::vector<uint64_t> &block_addr,
                            uint64_t slot_addr, uint64_t max_context_tokens, VkSessionState *out);

// m210 design note D6' (model.cpp:1076-1105 bind_fixed_state, restated): the fixed state of a prefix state slot whose
// run starts at state_addr (layout::fixed_state's bytes), carved as a one-token slot: gdn conv/ssm, PLE history/conv
// and the QSA rings. slot_map, out_counts and the pool entries stay 0 (never addressed through a state slot).
cere_status vk_bind_fixed_state(const host::ModelPlan &plan, uint64_t state_addr, uint64_t max_context_tokens,
                                VkSessionState *out);

// T6: one undo snapshot (layout::carve_undo, the CUDA model.h UndoState restated for addresses) at `base` (the
// kUndo region of index 3 slot + j): per GDN layer the conv / SSM copies, the PLE history / conv copies, per
// full-attention layer the QSA ring copy and the one compressed-key row (kUndoKeyRow). key_row is the round's host
// value (forward.cpp forward_verify_rows: the pool row of the first QSA group this or a later verify row completes,
// -1 when none), set by the verify and read by the commit.
struct VkUndoState {
    vkk5::LayerState gdn[kMaxLayers] = {};
    vkk4::State ple{};
    uint64_t qsa_ring[kMaxLayers] = {};
    uint64_t qsa_key[kMaxLayers] = {};
    int64_t key_row = -1;
};
cere_status vk_bind_undo(const host::ModelPlan &plan, uint64_t base, uint64_t bytes, VkUndoState *out);

// The workspace regions the forward carves (layout::carve_workspace_rows at kWorkspaceRows), by layout::Region.
struct VkWorkspace {
    uint64_t rows = 0;  // workspace_rows
    uint64_t hyper = 0, normed = 0, mixed = 0, block_out = 0, qkv = 0, attn_out = 0, tap_routed = 0, tap_shared = 0,
             ids32 = 0, ids64 = 0, logits = 0, proc_logits = 0, top2 = 0;
    uint64_t scratch = 0, scratch_bytes = 0;
    uint64_t gemm_ws = 0, gemm_ws_bytes = 0;
    uint64_t mtp_hc_rows = 0;  // drafts only
};
cere_status vk_bind_workspace(const host::ModelPlan &plan, const VkLayout &layout,
                              const std::vector<uint64_t> &block_addr, VkWorkspace *out);

}  // namespace cere::vkdev
