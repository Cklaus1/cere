// VulkanAmd P3 T2: the Vulkan kernels' workspace / scratch sizers, restated host-only for the memory plan.
//
// Why restated: every `*_workspace_bytes` lives in a kernel translation unit that includes launch/vk_launch.h
// (<vulkan/vulkan.h>) and links against libvulkan, while the memory plan must stay pure host arithmetic (estimate runs
// before any device exists, and the host-only tests run without Vulkan). Each function below is a line-for-line copy of
// the cited Vulkan function, with the model dimensions passed in instead of the kernel's compile-time constants.
//
// THE ONE TABLE TO UPDATE. When a kernel's sizer changes, change its function here (and the citation); the drift test
// executor/vulkan/tests/vk_kernel_sizes_drift_test.cpp links the real kernel objects (libvulkan, no GPU needed) and
// fails on any difference, naming the entry. Entries marked kMayBeStale are those another worker is changing while the
// large-m GEMM is wired (task T4: K3/K4/K5 layer/K7/K13 projections move from 16-row GEMV chunks onto vkk2::run, which
// needs vkk2::gemm_workspace_bytes inside their workspaces), plus the K6 prologue, which has no Vulkan kernel yet (T3)
// and is the CUDA value.
//
// Citations are to executor/vulkan/src/kernels/<file> at the time of writing (files are untracked and moving; the drift
// test is the authority, not the line numbers).
#pragma once

#include <algorithm>
#include <cstdint>

namespace cere::vkdev::sizes {

enum Staleness : uint32_t {
    kStable = 0,      // the kernel is not being changed by another task as far as the design knows
    kMayBeStale = 1,  // the kernel's sizer is expected to change (T4 large-m wiring) or is a stand-in (T3)
};

inline constexpr uint64_t a16(uint64_t b) { return (b + 15) / 16 * 16; }
inline constexpr uint64_t a256(uint64_t b) { return (b + 255) / 256 * 256; }

// ---- K2 large-m GEMM: k2_gemm_vk.cpp:65 gemm_workspace_bytes(m, n) ----------------------------------------------
inline uint64_t k2_gemm_workspace(uint64_t m, uint64_t n) { return a16(m * 4) + a16(n * 4); }
// k2_gemm_vk.h:46-48 gemm_prepare_weight: winfo is n uint32 at a 16-byte aligned address (one per prepared weight).
inline uint64_t k2_winfo(uint64_t n) { return a16(n * 4); }

// ---- K3 hyper-connection mix: k3_hc_vk.cpp:48-52 hc_mix_workspace_bytes(shape, max_m) ------------------------------
// (T4 wiring added the K2 GEMM workspace for max(lowrank, hc x hidden) columns.) kMayBeStale.
inline constexpr uint64_t kK3MaxRows = 16383;  // k3_hc_vk.h:35
inline uint64_t k3_hc_mix_workspace(uint64_t hc, uint64_t hidden, uint64_t lowrank, uint64_t m) {
    return a256(m * lowrank * 4) + a256(m * lowrank * 2) + a256(m * hc * hidden * 4) +
           a256(k2_gemm_workspace(m, std::max(lowrank, hc * hidden)));
}

// ---- K4 PLE forward: k4_ple_vk.cpp:63-71 layout(), workspace_bytes(t) ------------------------------------------------
// (T4 wiring added the key_proj / value_proj K2 GEMM workspaces gk, gv.) kMayBeStale.
inline constexpr int64_t kK4MaxTokens = 16383;  // k4_ple_vk.cpp:41
inline uint64_t k4_workspace(uint64_t embed_dim, uint64_t channels, uint64_t hidden, int64_t t) {
    if (t <= 0) return 0;
    const uint64_t n = uint64_t(t);
    const uint64_t emb = a256(n * embed_dim * 2), wide = a256(n * channels * 2), narrow = a256(n * hidden * 2);
    return emb + 3 * wide + narrow + a256(k2_gemm_workspace(n, channels)) + a256(k2_gemm_workspace(n, hidden));
}
// k4_ple_vk.cpp:72, :332 proj_info_bytes(): Weights::proj_info = key_proj's K2 row info (channels words) then
// value_proj's (hidden words), 16-byte aligned parts.
inline uint64_t k4_proj_info(uint64_t channels, uint64_t hidden) { return a16(channels * 4) + a16(hidden * 4); }

// ---- K5 GDN chunk: k5_gdn_chunk_vk.cpp:44-74 carve / chunk_workspace_bytes(tokens) ---------------------------------
inline constexpr int64_t kK5ChunkTokens = 64;      // k5_gdn_chunk_vk.h:29 (BT)
inline constexpr int64_t kK5MaxTokens = 4096;      // k5_gdn_chunk_vk.h:30
inline uint64_t k5_chunk_workspace(uint64_t key_heads, uint64_t value_heads, uint64_t head_dim, int64_t tokens) {
    if (tokens < 1 || tokens > kK5MaxTokens) return 0;
    const uint64_t t = uint64_t(tokens), bt = uint64_t(kK5ChunkTokens), chunks = (t + bt - 1) / bt;
    const uint64_t kh = key_heads, vh = value_heads, d = head_dim;
    return a256(t * kh * d * 2) + a256(t * kh * d * 2) + a256(t * vh * d * 2) + a256(t * vh * 4) +
           a256(t * vh * 4) + a256(t * vh * d * 2) + a256(t * vh * d * 2) + a256(t * vh * d * 2) +
           a256(t * vh * bt * 2) + a256(chunks * vh * d * d * 2) + a256(t * vh * d * 2);
}

// ---- K5 GDN layer: k5_gdn_layer_vk.cpp:38-80 carve_core / carve_layer, layer_workspace_bytes -----------------------
// conv_dim = 2 key_dim + value_dim (10,240), value_dim = value_heads x head_dim (6,144). (T4 wiring added the four K2
// GEMM workspaces g_qkv (sized for n = kMaxHidden 65,536, reused by out_proj), g_z, g_b, g_a.) kMayBeStale.
inline constexpr uint64_t kK5MaxHidden = 65536;  // k5_gdn_layer_vk.cpp:52
inline uint64_t k5_layer_workspace(uint64_t key_heads, uint64_t value_heads, uint64_t head_dim, int64_t tokens) {
    if (tokens < 1 || tokens > kK5MaxTokens) return 0;
    const uint64_t t = uint64_t(tokens);
    const uint64_t key_dim = key_heads * head_dim, value_dim = value_heads * head_dim;
    const uint64_t conv_dim = 2 * key_dim + value_dim;
    const uint64_t layer = a256(t * conv_dim * 2) + a256(t * value_dim * 2) + a256(t * value_heads * 2) +
                           a256(t * value_heads * 2) + a256(k2_gemm_workspace(t, std::max(conv_dim, kK5MaxHidden))) +
                           a256(k2_gemm_workspace(t, value_dim)) + a256(k2_gemm_workspace(t, value_heads)) +
                           a256(k2_gemm_workspace(t, value_heads));
    const uint64_t core = a256(t * conv_dim * 2) + a256(t * value_dim * 2) + a256(t * value_heads * 4) +
                          a256(t * value_heads * 4) + a256(k5_chunk_workspace(key_heads, value_heads, head_dim, tokens));
    return layer + core;
}

// k5_gdn_layer_vk.cpp:149-174 info_parts / weight_info_bytes(hidden): LayerWeights::info = the K2 row info of
// in_proj_qkv (conv_dim words), in_proj_z (value_dim), in_proj_b, in_proj_a (value_heads each), out_proj (hidden).
inline uint64_t k5_weight_info(uint64_t key_heads, uint64_t value_heads, uint64_t head_dim, uint64_t hidden) {
    const uint64_t value_dim = value_heads * head_dim, conv_dim = 2 * key_heads * head_dim + value_dim;
    return a16(conv_dim * 4) + a16(value_dim * 4) + a16(value_heads * 4) + a16(value_heads * 4) + a16(hidden * 4);
}

// ---- K6 dense attention: k6_attention_vk.cpp:40-45 partials, :96-100 workspace_bytes(t, Options{64, 64}) -----------
inline constexpr int64_t kK6MaxContext = 2048;     // k6_attention_vk.h:37
inline constexpr int64_t kK6SplitMaxTokens = 16;   // k6_attention_vk.h:38
inline constexpr uint64_t kK6KeysPerSplit = 64;    // k6_attention_vk.h:45 Options default
// coopmat (P4-E): the size with Options::prefill = Prefill::kCoopmat (k6_attention_vk.cpp flags_bytes: one word per
// (token, KV head, head third) for calls of > 16 rows). The memory plan charges this one: every prefill_attention
// selection must run in the buffers ALLOCATE made (cere_model_set_selection is hot-safe).
inline constexpr uint64_t kK6KvHeads = 2;          // k6_attention_vk.h kKvHeads
inline uint64_t k6_workspace(uint64_t q_heads, uint64_t head_dim, int64_t t, bool coopmat = false) {
    if (t <= 0) return 0;
    const uint64_t split_rows = uint64_t(std::min<int64_t>(t, kK6SplitMaxTokens));
    const uint64_t r = split_rows * q_heads * (uint64_t(kK6MaxContext) / kK6KeysPerSplit);
    const uint64_t partials = a256(r * head_dim * 4) + a256(r * 2 * 4);
    const uint64_t flags = coopmat && t > kK6SplitMaxTokens ? a256(uint64_t(t) * kK6KvHeads * 3 * 4) : 0;
    return partials + a256(uint64_t(t) * q_heads * head_dim * 2) + flags;
}
// K6 prologue (q normed+rotated, k normed+rotated, gate): NO Vulkan kernel yet (design 2.3 item 1, task T3). This is
// the CUDA value, executor/cuda/src/kernels/k6_attention.cu:1153-1159 prologue_bytes. kMayBeStale.
inline uint64_t k6_prologue_cuda(uint64_t q_width, uint64_t kv_width, int64_t t) {
    if (t <= 0) return 0;
    const uint64_t n = uint64_t(t);
    return a256(n * q_width * 2) + a256(n * kv_width * 2) + a256(n * q_width * 2);
}

// ---- K9 QSA: k9_qsa_vk.cpp:107-138 partial_bytes_*, logits_tile_rows, select/attention/scratch_bytes ----------------
inline constexpr int64_t kK9Ratio = 4;                          // k9_qsa_vk.h:47
inline constexpr int64_t kK9IndexHeads = 4;                     // k9_qsa_vk.h:43
inline constexpr int64_t kK9IndexHeadDim = 128;                 // k9_qsa_vk.h:44
inline constexpr int64_t kK9RowWidth = 2048 + kK9Ratio - 1;     // k9_qsa_vk.h:50 (kTokenTopk + kRatio - 1)
inline constexpr int64_t kK9Tiles = (kK9RowWidth + 63) / 64;    // k9_qsa_vk.h:64
inline constexpr uint64_t kK9LogitsTileBytes = uint64_t(128) << 20;  // k9_qsa_vk.h:59
inline constexpr int64_t kK9SplitMaxRows = 16;                  // k9_qsa_vk.h:60
inline int64_t k9_logits_tile_rows(int64_t rows, int64_t ctx) {
    const int64_t keys = ctx / kK9Ratio;
    if (rows <= 0 || keys <= 0) return rows > 1 ? rows : 1;
    const int64_t block_q = 128 / kK9IndexHeads;
    const int64_t bytes_per_row = keys * 4;
    int64_t max_padded = int64_t(kK9LogitsTileBytes) / bytes_per_row;
    if (max_padded < block_q) max_padded = block_q;
    max_padded = max_padded / block_q * block_q;
    if (max_padded < block_q) max_padded = block_q;
    return rows < max_padded ? rows : max_padded;
}
inline uint64_t k9_scratch(uint64_t q_heads, uint64_t head_dim, int64_t rows, int64_t ctx) {
    if (rows <= 0) return 0;
    const uint64_t r = uint64_t(rows);
    const uint64_t sel = (ctx < kK9Ratio) ? 0 : a256(uint64_t(k9_logits_tile_rows(rows, ctx)) * uint64_t(ctx / kK9Ratio) * 4);
    const uint64_t att = rows > kK9SplitMaxRows ? 0
                                                : a256(r * q_heads * kK9Tiles * head_dim * 4) + a256(r * q_heads * kK9Tiles * 2 * 4);
    return a256(r * kK9IndexHeads * kK9IndexHeadDim * 2) + a256(r * uint64_t(kK9RowWidth) * 4) + std::max(sel, att);
}

// ---- K7 MoE (W4A4): k7_moe_vk.cpp:23-56 carve, :151-154 moe_workspace_bytes; :114-124 info_parts, :156 ----------
struct MoeDims {
    uint64_t hidden, experts, top_k, inter, shared_inter;
};
inline uint64_t k7_moe_workspace(const MoeDims &d, uint64_t m) {
    if (m == 0) return 0;
    const uint64_t r = m * d.top_k;
    return a256(m * d.experts * 2) + a256(r * 4) + a256(r * 4) + a256(m * d.hidden / 2) + a256(m * d.hidden / 16) +
           a256(m * 4) + a256((2 * d.experts + 2) * 4) + a256(r * 4) + a256(r * 4) + a256(r * d.inter * 2) +
           a256(r * d.inter / 2) + a256(r * d.inter / 16) + a256(r * d.hidden * 2) + a256(m * d.shared_inter * 2) +
           a256(m * d.shared_inter * 2) + a256(m * d.shared_inter * 2) + a256(m * d.hidden * 2) +
           a256(k2_gemm_workspace(m, d.experts)) + a256(k2_gemm_workspace(m, d.shared_inter)) +
           a256(k2_gemm_workspace(m, d.shared_inter)) + a256(k2_gemm_workspace(m, d.hidden));
}
// MoeLayer::bf16_info (router, shared gate, shared up, shared down K2 row info), per layer. k7_moe_vk.cpp:114-124/156;
// k13_moe_bf16_vk.cpp:94-102 has the same parts for MoeBf16Layer::bf16_info.
inline uint64_t k7_weight_info(const MoeDims &d) {
    return a16(d.experts * 4) + a16(d.shared_inter * 4) + a16(d.shared_inter * 4) + a16(d.hidden * 4);
}

// ---- K13 BF16/W4A16 MoE (MTP): k13_moe_bf16_vk.cpp:21-49 carve, :102-105 moe_bf16_workspace_bytes -----------------
inline uint64_t k13_moe_workspace(const MoeDims &d, uint64_t m) {
    if (m == 0) return 0;
    const uint64_t r = m * d.top_k;
    return a256(m * d.experts * 2) + a256(r * 4) + a256(r * 4) + a256(m * 4) + a256((2 * d.experts + 2) * 4) +
           a256(r * 4) + a256(r * 4) + a256(r * d.inter * 2) + a256(r * d.hidden * 2) + a256(m * d.shared_inter * 2) +
           a256(m * d.shared_inter * 2) + a256(m * d.shared_inter * 2) + a256(m * d.hidden * 2) +
           a256(k2_gemm_workspace(m, d.experts)) + a256(k2_gemm_workspace(m, d.shared_inter)) +
           a256(k2_gemm_workspace(m, d.shared_inter)) + a256(k2_gemm_workspace(m, d.hidden));
}

// ---- K12 sampler: k12_sampler_vk.cpp:10 kScratch, :66 scratch_bytes() ----------------------------------------------
inline constexpr uint64_t kK12Scratch = 38952;

// ---- K13S spec accept: k13s_spec_accept_vk.cpp:9 kRowPrefix, :24-38 carve, :40-44 scratch_bytes(n) ----------------
inline constexpr uint64_t kK13sRowPrefix = 8216;
inline constexpr int64_t kK13sMaxDrafts = 15;  // k13s_spec_accept_vk.h:16
inline uint64_t k13s_scratch(int64_t n) {
    if (n < 0) n = 0;
    if (n > kK13sMaxDrafts) n = kK13sMaxDrafts;
    const uint64_t u = uint64_t(n);
    return a256(kK12Scratch) + a256(4) + a256(u * kK13sRowPrefix) + a256(u * 4) + a256(u);
}

// ---- K4 split table: k4_ple_vk.h:55-61 (restated geometry constants of vkk4::SplitTable) ---------------------------
inline constexpr uint32_t kK4MaxBlocks = 64;                   // k4_ple_vk.h:60
inline constexpr uint32_t kK4BlockTableBytes = 16 * kK4MaxBlocks;  // k4_ple_vk.h:61: u64 table_base[64], scale_base[64]
inline constexpr uint32_t kK4CallTableBytes = 4096;            // k4_ple_vk.h:152
inline constexpr uint32_t kK4Nvfp4RowBytes = 80;               // k4_ple_vk.h:55
inline constexpr uint32_t kK4Nvfp4RowScales = 10;              // k4_ple_vk.h:56

}  // namespace cere::vkdev::sizes
