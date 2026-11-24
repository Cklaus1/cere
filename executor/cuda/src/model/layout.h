// The CUDA-free part of the A4 memory plan (model.h "Sizing"): the 256-byte aligned carving of the derived region, of
// one session slot, of the paged KV pool (m25 design note D2) and of the workspace's row buffers, shared by
// model.cpp (sizing and binding) and the host tests (executor/tests/host, built without CUDA: m22c design note C10).
// The workspace's shared scratch (the largest K3-K7/K9 kernel workspace) is model.cpp's: those sizes live in the CUDA
// kernels.
#pragma once

#include "cere_executor.h"

#include <cstdint>

namespace cere::model::layout {

constexpr uint64_t kAlign = 256;
constexpr uint64_t kBf16 = 2;
constexpr uint64_t kF32 = 4;
constexpr uint64_t kI64 = 8;

// Carves 256-byte aligned buffers out of a region; with a null base it only measures (the sizing functions).
struct Carver {
    uint8_t *base;
    uint64_t cursor = 0;

    template <typename T>
    T *take(uint64_t bytes) {
        cursor = (cursor + kAlign - 1) / kAlign * kAlign;
        T *p = base == nullptr ? nullptr : reinterpret_cast<T *>(base + cursor);
        cursor += bytes;
        return p;
    }
    uint64_t size() const { return (cursor + kAlign - 1) / kAlign * kAlign; }
};

inline bool full_attention(const cere_qwen4_exp_config &cfg, uint32_t layer) {
    return layer < 64 && ((cfg.full_attention_layers >> layer) & 1) != 0;
}

inline uint64_t rotary_dim(const cere_qwen4_exp_config &cfg) {
    return uint64_t(double(cfg.head_dim) * cfg.partial_rotary_factor);
}

// QSA compressed-key rows of a context (m22c design note F5): ceil(ctx / indexer_compress_ratio).
inline uint64_t qsa_blocks(const cere_qwen4_exp_config &cfg, uint64_t ctx) {
    const uint64_t ratio = cfg.indexer_compress_ratio == 0 ? 1 : cfg.indexer_compress_ratio;
    return (ctx + ratio - 1) / ratio;
}

enum class Region {
    kALog,        // derived: F32 [linear_num_value_heads] per GDN layer
    kG1Alphas,    // derived: F32 [num_experts] per layer
    kG2Alphas,    // derived: F32 [num_experts] per layer
    kRope,        // derived: F32 RoPE table [ctx, rotary_dim] (layer 0)
    kKCache,      // pool: FP8 [pool tokens, kv_heads * head_dim] per KV layer, row = physical slot
    kVCache,      // pool: FP8 [pool tokens, kv_heads * head_dim]
    kQsaKeys,     // pool: BF16 [pool tokens / ratio, indexer_head_dim] per KV layer, row = physical slot / ratio
    kSlotMap,     // session: int64 [ctx], logical position -> physical slot (layer 0)
    kQsaRing,     // session: BF16 [ratio, indexer_head_dim] per full-attention layer
    kGdnConv,     // session: BF16 [conv channels, conv_kernel - 1] per GDN layer
    kGdnSsm,      // session: BF16 [value_heads, key_dim, value_dim] per GDN layer
    kPleHistory,  // session: int64 [ngram_size - 1] (layer 0)
    kPleConv,     // session: BF16 [hc * hidden, (ple_conv_kernel - 1) * ngram_size] (layer 0)
    kOutCounts,   // session: u32 [vocab] the sampler's output counts (m27 I9/D5)
    // workspace row buffers (model.h Workspace), T = rows
    kHyper,       // BF16 [T, hc * hidden]
    kNormed,      // BF16 [T, hc * hidden]
    kMixed,       // BF16 [T, hidden]
    kBlockOut,    // BF16 [T, hidden]
    kQkv,         // BF16 [T, q|gate + k + v]
    kAttnOut,     // BF16 [T, heads * head_dim]
    kTapRouted,   // BF16 [T, hidden]
    kTapShared,   // BF16 [T, hidden]
    kIds32,       // int32 [T]
    kIds64,       // int64 [T]
    kLogits,      // BF16 [vocab]
    kProcLogits,  // F32 [vocab] the processed row of a sampled step (m27 I9)
    kTop2,        // cere_token_score [T]
    // M2.9 draft-only region (carve_undo): the one compressed-key row per full-attention layer a verify chain may
    // complete, BF16 [indexer_head_dim]. The snapshot's other copies reuse the session tags above.
    kUndoKeyRow,
    // M2.9 MTP draft layer (m29 design note D3, M29S2-04): carved only when drafts are on, after every region above
    kMtpKCache,   // pool: FP8 [pool tokens, kv_heads * head_dim], the MTP layer's K (same slot map as the target)
    kMtpVCache,   // pool: FP8 [pool tokens, kv_heads * head_dim]
    kMtpQsaKeys,  // pool: BF16 [pool tokens / ratio, indexer_head_dim]
    kMtpQsaRing,  // session: BF16 [ratio, indexer_head_dim], the MTP layer's QSA pending ring
    kProposals,   // session: u32 [4], the pending MTP proposals of a drafting session: count (0..3), then 3 ids
    kMtpLmHead,   // derived: BF16 [draft count, hidden], the hot lm_head (lm_head rows at the draft token map)
    kMtpMap,      // derived: u32 [draft count], the draft token map
};

// The workspace's row buffers for T rows, in Region order; the caller takes the shared scratch after them.
template <typename Bind>
void carve_workspace_rows(const cere_qwen4_exp_config &cfg, uint64_t t, Carver &c, Bind &&bind) {
    const uint64_t hidden = cfg.hidden_size;
    const uint64_t hc_width = uint64_t(cfg.hc_count) * hidden;
    const uint64_t qkv = uint64_t(cfg.num_attention_heads) * 2 * cfg.head_dim +
                         2 * uint64_t(cfg.num_key_value_heads) * cfg.head_dim;
    const uint64_t attn = uint64_t(cfg.num_attention_heads) * cfg.head_dim;
    bind(Region::kHyper, 0, c.take<uint8_t>(t * hc_width * kBf16));
    bind(Region::kNormed, 0, c.take<uint8_t>(t * hc_width * kBf16));
    bind(Region::kMixed, 0, c.take<uint8_t>(t * hidden * kBf16));
    bind(Region::kBlockOut, 0, c.take<uint8_t>(t * hidden * kBf16));
    bind(Region::kQkv, 0, c.take<uint8_t>(t * qkv * kBf16));
    bind(Region::kAttnOut, 0, c.take<uint8_t>(t * attn * kBf16));
    bind(Region::kTapRouted, 0, c.take<uint8_t>(t * hidden * kBf16));
    bind(Region::kTapShared, 0, c.take<uint8_t>(t * hidden * kBf16));
    bind(Region::kIds32, 0, c.take<uint8_t>(t * sizeof(int32_t)));
    bind(Region::kIds64, 0, c.take<uint8_t>(t * kI64));
    bind(Region::kLogits, 0, c.take<uint8_t>(uint64_t(cfg.vocab_size) * kBf16));
    bind(Region::kProcLogits, 0, c.take<uint8_t>(uint64_t(cfg.vocab_size) * kF32));
    bind(Region::kTop2, 0, c.take<uint8_t>(t * sizeof(cere_token_score)));
}

// The decode graph (m23 design note P1, M23P1-03) stages a step's R row positions in ids64's tail: host_ids64[R, 2 R)
// -> ids64[R, 2 R), one H2D copy with the ids, so the graph adds no device or host byte (M2.11 I12). A step whose
// 2 R entries do not fit the T = workspace_rows entries runs eagerly.
inline bool decode_graph_fits(uint64_t rows, uint64_t t) { return 2 * rows <= t; }

// The verify graph (decode_step = spec_graph, m29 design note M29S2-18b, m23 design note P11R-20) stages the m
// verify rows' positions the same way (decode_graph_fits(m, T)). A chain at position p with n drafts may join it when
// every row attends densely from the FP8 cache with a prefix: p > 0 and p + n + 1 <= kDenseContext (k6::kMaxContext,
// forward.cpp asserts the two agree), so no row reaches K9's sparse path or the prefix-0 prefill branch. The rest of
// the rule (greedy chains only: no sampler, no logprobs, no taps; m <= the graphs' row bound) is forward_verify_rows'.
constexpr uint64_t kDenseContext = 2048;
inline bool spec_graph_chain_fits(uint64_t p, uint64_t n) { return p > 0 && p + n + 1 <= kDenseContext; }

// The draft graph (decode_step = spec_graph, the same spec rows): one session's MTP draft after a commit, an extend of
// `rows` rows at p0 .. p0 + rows - 1 then kDraftSteps one-row MTP decode steps, its rows + kDraftSteps positions staged
// in ids64's head (host_ids64[0, rows + 2) -> ids64, one H2D copy; the VERIFY call before it has synchronized). It may
// run captured when every MTP row attends densely from the FP8 cache with a prefix (p0 > 0, p0 + rows + kDraftSteps
// <= kDenseContext), both decode steps run (p0 + rows + kDraftSteps <= the session's max_context_tokens `ctx`: the
// eager loop stops at the context), rows <= kDraftGraphRows (K6's split-K row bound and the graphs' row bound,
// forward.cpp asserts both) and the positions fit the T = workspace_rows entries.
constexpr uint64_t kDraftSteps = 2;
constexpr uint64_t kDraftGraphRows = 16;
inline bool spec_draft_graph_fits(uint64_t p0, uint64_t rows, uint64_t ctx, uint64_t t) {
    return p0 > 0 && rows >= 1 && rows <= kDraftGraphRows && p0 + rows + kDraftSteps <= kDenseContext &&
           p0 + rows + kDraftSteps <= ctx && rows + kDraftSteps <= t;
}

// The MTP decode steps of a draft after its extend of `rows` rows at p0 (CudaDevice::draft): kDraftSteps, fewer where
// the context `ctx` ends first (a step at p0 + rows + k runs while inside it), none while the prompt stays open (the
// extend of a prompt chunk that is not the prompt's last). An open prompt's next PREFILL call extends from p0 + rows
// again, so such steps' proposals and MTP rows would be overwritten unread; and their rotary position p + rope_delta
// (the decode steps' text rope) is below the table for an image prompt's rows before its last image ends (the
// prompt's whole rope_delta is negative), which K9's bounds refuse (CERE_ERR_ARGUMENT).
inline uint64_t draft_decode_steps(uint64_t p0, uint64_t rows, uint64_t ctx, bool prompt_open) {
    const uint64_t end = p0 + rows;
    if (prompt_open || end >= ctx) return 0;
    return ctx - end < kDraftSteps ? ctx - end : kDraftSteps;
}

// The derived region: per layer [F32 A_log (GDN only)], F32 g1 alphas [E], F32 g2 alphas [E]; then the F32 RoPE
// table [ctx, rotary_dim]. bind(Region, layer, uint8_t *) receives each region's start (null when measuring).
template <typename Bind>
void carve_derived(const cere_qwen4_exp_config &cfg, uint64_t ctx, Carver &c, Bind &&bind) {
    for (uint32_t l = 0; l < cfg.num_hidden_layers && l < 64; ++l) {
        if (!full_attention(cfg, l)) {
            bind(Region::kALog, l, c.take<uint8_t>(uint64_t(cfg.linear_num_value_heads) * kF32));
        }
        bind(Region::kG1Alphas, l, c.take<uint8_t>(uint64_t(cfg.num_experts) * kF32));
        bind(Region::kG2Alphas, l, c.take<uint8_t>(uint64_t(cfg.num_experts) * kF32));
    }
    bind(Region::kRope, 0, c.take<uint8_t>(ctx * rotary_dim(cfg) * kF32));
}

// The session slot's per-layer state shapes (carve_session; carve_undo copies the same shapes).
inline uint64_t gdn_conv_bytes(const cere_qwen4_exp_config &cfg) {
    const uint64_t conv_channels = 2 * uint64_t(cfg.linear_num_key_heads) * cfg.linear_key_head_dim +
                                   uint64_t(cfg.linear_num_value_heads) * cfg.linear_value_head_dim;
    const uint64_t conv_taps = cfg.linear_conv_kernel_dim == 0 ? 0 : cfg.linear_conv_kernel_dim - 1;
    return conv_channels * conv_taps * kBf16;
}
inline uint64_t gdn_ssm_bytes(const cere_qwen4_exp_config &cfg) {
    return uint64_t(cfg.linear_num_value_heads) * cfg.linear_key_head_dim * cfg.linear_value_head_dim * kBf16;
}
inline uint64_t ple_history_bytes(const cere_qwen4_exp_config &cfg) {
    return uint64_t(cfg.ngram_size == 0 ? 0 : cfg.ngram_size - 1) * kI64;
}
inline uint64_t ple_conv_bytes(const cere_qwen4_exp_config &cfg) {
    const uint64_t ple_channels = uint64_t(cfg.hc_count) * cfg.hidden_size;
    const uint64_t ple_taps =
        uint64_t(cfg.ple_conv_kernel_size == 0 ? 0 : cfg.ple_conv_kernel_size - 1) * cfg.ngram_size;
    return ple_channels * ple_taps * kBf16;
}
inline uint64_t qsa_key_row_bytes(const cere_qwen4_exp_config &cfg) { return uint64_t(cfg.indexer_head_dim) * kBf16; }
inline uint64_t qsa_ring_bytes(const cere_qwen4_exp_config &cfg) {
    return uint64_t(cfg.indexer_compress_ratio) * qsa_key_row_bytes(cfg);
}

// One session slot: the slot map, then per GDN layer the conv and SSM state, then the PLE history and conv state, then
// per full-attention layer the QSA pending ring (m22c D5), then the sampler's output counts (m27 I9). The
// full-attention layers' KV and compressed keys live in the model's pool (carve_pool), addressed through the slot map
// (m25 D2/D3). With `draft` (M2.9, M29S2-04/-19) the MTP layer's QSA ring and the pending proposals follow, so a
// plan without drafts is byte-identical.
template <typename Bind>
void carve_session(const cere_qwen4_exp_config &cfg, uint64_t ctx, Carver &c, Bind &&bind, bool draft = false) {
    bind(Region::kSlotMap, 0, c.take<uint8_t>(ctx * kI64));
    for (uint32_t l = 0; l < cfg.num_hidden_layers && l < 64; ++l) {
        if (full_attention(cfg, l)) continue;
        bind(Region::kGdnConv, l, c.take<uint8_t>(gdn_conv_bytes(cfg)));
        bind(Region::kGdnSsm, l, c.take<uint8_t>(gdn_ssm_bytes(cfg)));
    }
    bind(Region::kPleHistory, 0, c.take<uint8_t>(ple_history_bytes(cfg)));
    bind(Region::kPleConv, 0, c.take<uint8_t>(ple_conv_bytes(cfg)));
    for (uint32_t l = 0; l < cfg.num_hidden_layers && l < 64; ++l) {
        if (!full_attention(cfg, l)) continue;
        bind(Region::kQsaRing, l, c.take<uint8_t>(qsa_ring_bytes(cfg)));
    }
    bind(Region::kOutCounts, 0, c.take<uint8_t>(uint64_t(cfg.vocab_size) * sizeof(uint32_t)));
    if (draft) {
        bind(Region::kMtpQsaRing, 0, c.take<uint8_t>(qsa_ring_bytes(cfg)));
        bind(Region::kProposals, 0, c.take<uint8_t>(4 * sizeof(uint32_t)));
    }
}

// ---- M2.9 draft-only region (m29 design note M29S2-05, M29S2-19, M29S2-20) ------------------------------------
// Carved only for plans with speculative_draft_tokens > 0 (a spec-off arena is byte-identical, A1), after every other
// device region: max_sessions x kUndoSnapshots undo snapshots (model.h UndoState), slot i's undo[j] at index 3 i + j,
// then the verify rows' sampler stages (model.h SpecSampler, phase S).
// (Workspace::hc_rows is the workspace's, carved by the MTP forward.)
constexpr uint64_t kVerifyRows = 16;     // rows of one verify call: min(16, 4 x max_sessions) (M29S2-05)
constexpr uint64_t kUndoSnapshots = 3;   // per drafting session: undo[j - 1] before verify row j in {1, 2, 3}
// DraftAccept deep chains: the most drafts of one verify chain (the MTP drafts plus a lookup tail, or a long lookup
// chain). A call group with a chain longer than kUndoSnapshots draws its undo snapshots from the model's whole pool
// (max_sessions x kUndoSnapshots, contiguous ranges, at most the pool per call group): no extra bytes. Chains
// (O123, chains design note): 15, so a chain's 16 rows fill one verify call (kVerifyRows, GemmPlan::kMaxRows); a
// plan's chains stay within its pool (CudaModel::max_chain_drafts: 12 at max_sessions 4).
constexpr uint64_t kMaxChainDrafts = 15;
static_assert(kMaxChainDrafts + 1 <= kVerifyRows, "a chain fits one verify call");

// The verify call's rows of a plan with drafts (Workspace::max_rows): min(16, 4 x max_sessions, workspace rows).
inline uint64_t verify_rows(uint64_t max_sessions, uint64_t workspace_rows) {
    const uint64_t four = 4 * max_sessions;
    const uint64_t r = kVerifyRows < four ? kVerifyRows : four;
    return r < workspace_rows ? r : workspace_rows;
}

// One undo snapshot: per GDN layer the conv and SSM copies, the PLE history and conv copies, then per full-attention
// layer the QSA ring copy and the one compressed-key row (kUndoKeyRow), in the session slot's shapes.
template <typename Bind>
void carve_undo(const cere_qwen4_exp_config &cfg, Carver &c, Bind &&bind) {
    for (uint32_t l = 0; l < cfg.num_hidden_layers && l < 64; ++l) {
        if (full_attention(cfg, l)) continue;
        bind(Region::kGdnConv, l, c.take<uint8_t>(gdn_conv_bytes(cfg)));
        bind(Region::kGdnSsm, l, c.take<uint8_t>(gdn_ssm_bytes(cfg)));
    }
    bind(Region::kPleHistory, 0, c.take<uint8_t>(ple_history_bytes(cfg)));
    bind(Region::kPleConv, 0, c.take<uint8_t>(ple_conv_bytes(cfg)));
    for (uint32_t l = 0; l < cfg.num_hidden_layers && l < 64; ++l) {
        if (!full_attention(cfg, l)) continue;
        bind(Region::kQsaRing, l, c.take<uint8_t>(qsa_ring_bytes(cfg)));
        bind(Region::kUndoKeyRow, l, c.take<uint8_t>(qsa_key_row_bytes(cfg)));
    }
}

// One snapshot, a multiple of 256 (59,034,880 B on the A4 model).
inline uint64_t undo_bytes(const cere_qwen4_exp_config &cfg) {
    Carver c{nullptr};
    carve_undo(cfg, c, [](Region, uint32_t, uint8_t *) {});
    return c.size();
}

// The paged KV pool (m25 D2): for each layer of the KV layer set `layers` (bit l = layer l: the model's
// full_attention_layers here), FP8 K and V [tokens, kv_heads * head_dim] and the BF16 QSA
// compressed keys [tokens / ratio, indexer_head_dim]. `tokens` is the pool, a multiple of the 64-token page, so one
// page id addresses every layer. With `draft` (M2.9, M29S2-04) the MTP layer's entry (the same three tensors, 1,088 B
// per token) follows the target layers', so a plan without drafts is byte-identical.
template <typename Bind>
void carve_pool(const cere_qwen4_exp_config &cfg, uint64_t layers, uint64_t tokens, Carver &c, Bind &&bind,
                bool draft = false) {
    const uint64_t kv_width = uint64_t(cfg.num_key_value_heads) * cfg.head_dim;
    const uint64_t key_row = uint64_t(cfg.indexer_head_dim) * kBf16;
    for (uint32_t l = 0; l < 64; ++l) {
        if (((layers >> l) & 1) == 0) continue;
        bind(Region::kKCache, l, c.take<uint8_t>(tokens * kv_width));
        bind(Region::kVCache, l, c.take<uint8_t>(tokens * kv_width));
        bind(Region::kQsaKeys, l, c.take<uint8_t>(qsa_blocks(cfg, tokens) * key_row));
    }
    if (draft) {
        bind(Region::kMtpKCache, 0, c.take<uint8_t>(tokens * kv_width));
        bind(Region::kMtpVCache, 0, c.take<uint8_t>(tokens * kv_width));
        bind(Region::kMtpQsaKeys, 0, c.take<uint8_t>(qsa_blocks(cfg, tokens) * key_row));
    }
}

// M2.9 (m29 design note D3): the MTP part of the derived region, after the target's (carve_derived): the hot
// lm_head BF16 [count, hidden] (lm_head rows at the draft token map) then the device copy of the map u32 [count].
template <typename Bind>
void carve_mtp_derived(const cere_qwen4_exp_config &cfg, uint64_t count, Carver &c, Bind &&bind) {
    bind(Region::kMtpLmHead, 0, c.take<uint8_t>(count * cfg.hidden_size * kBf16));
    bind(Region::kMtpMap, 0, c.take<uint8_t>(count * sizeof(uint32_t)));
}

inline uint64_t mtp_derived_bytes(const cere_qwen4_exp_config &cfg, uint64_t count) {
    Carver c{nullptr};
    carve_mtp_derived(cfg, count, c, [](Region, uint32_t, uint8_t *) {});
    return c.size();
}

// M2.9: Workspace::hc_rows, the hc rows the MTP extend / draft-extend reads (the target's pre-mixer rows of a chunk or
// of the verify rows), BF16 [rows, hc * hidden] (D10': 80 MiB at 4,096 rows); carved only with drafts.
inline uint64_t mtp_hc_rows_bytes(const cere_qwen4_exp_config &cfg, uint64_t rows) {
    return (rows * cfg.hc_count * cfg.hidden_size * kBf16 + kAlign - 1) / kAlign * kAlign;
}

inline uint64_t derived_bytes(const cere_qwen4_exp_config &cfg, uint64_t ctx) {
    Carver c{nullptr};
    carve_derived(cfg, ctx, c, [](Region, uint32_t, uint8_t *) {});
    return c.size();
}

inline uint64_t session_bytes(const cere_qwen4_exp_config &cfg, uint64_t ctx, bool draft = false) {
    Carver c{nullptr};
    carve_session(cfg, ctx, c, [](Region, uint32_t, uint8_t *) {}, draft);
    return c.size();
}

// The fixed state of a session slot (m210 design note D6′, ledger M4): the one contiguous run [begin, begin + bytes)
// from the first GDN layer's conv state to the start of out_counts (GDN conv + SSM, PLE history + conv, QSA ring; never
// the slot map nor out_counts), 59,031,808 B on the A4 config (59,031,568 B + 240 B alignment padding after the PLE
// history). `contiguous` is false when a region other than those lies inside the run (carve_session changed). `bytes`
// does not depend on ctx (the slot map is the only ctx-sized region and comes first); `begin` does.
struct FixedState {
    uint64_t begin = 0;
    uint64_t bytes = 0;
    bool contiguous = true;
};

inline FixedState fixed_state(const cere_qwen4_exp_config &cfg, uint64_t ctx) {
    Carver c{nullptr};
    FixedState f;
    bool open = false;
    bool closed = false;
    uint64_t end = 0;  // where the previous region ended; the next starts at its aligned cursor
    carve_session(cfg, ctx, c, [&](Region r, uint32_t, uint8_t *) {
        const uint64_t start = (end + kAlign - 1) / kAlign * kAlign;
        end = c.cursor;
        if (r == Region::kGdnConv && !open) {
            open = true;
            f.begin = start;
        } else if (r == Region::kOutCounts && open && !closed) {
            closed = true;
            f.bytes = start - f.begin;
        } else if (open && !closed && r != Region::kGdnConv && r != Region::kGdnSsm && r != Region::kPleHistory &&
                   r != Region::kPleConv && r != Region::kQsaRing) {
            f.contiguous = false;
        }
    });
    if (!open || !closed) f.contiguous = false;
    return f;
}

// m213 design note D12 (C11), resized by m213l design note D-L7 (VL-L2): the image buffers of a model with
// vision_config, sized for one image of at most kMaxImageTokens merged tokens (the per-image cap; a request's image
// tokens are bounded by its context only, since forward encodes the images of each call one at a time). The encoder's
// activations alias the text workspace (the encoder finishes before layer 0).
constexpr uint64_t kMaxImageTokens = 4096;

// Device, twice: the encoder's output of one image (image_out) and the merged rows of a call's pad rows at their call
// rows (image_embeds; a call has at most the workspace rows <= kMaxImageTokens), each BF16
// [kMaxImageTokens, out_hidden_size] (20,971,520 B on the A4 model).
inline uint64_t image_embed_bytes(const cere_qwen4_exp_vision_config &v) {
    return kMaxImageTokens * v.out_hidden_size * kBf16;
}

// The pixel patches of one image, BF16 [kMaxImageTokens * merge^2, in_channels * temporal_patch * patch^2]
// (50,331,648 B on the A4 model): one in the device arena, one pinned in the host arena.
inline uint64_t image_patch_bytes(const cere_qwen4_exp_vision_config &v) {
    const uint64_t patches = kMaxImageTokens * v.spatial_merge_size * v.spatial_merge_size;
    return patches * v.in_channels * v.temporal_patch_size * v.patch_size * v.patch_size * kBf16;
}

// Patches of one image at most in a plan of max_context_tokens ctx: its merged rows stay below the context and within
// kMaxImageTokens.
inline uint64_t max_image_patches(const cere_qwen4_exp_vision_config &v, uint64_t ctx) {
    return (ctx < kMaxImageTokens ? ctx : kMaxImageTokens) * v.spatial_merge_size * v.spatial_merge_size;
}

// Device, and the same size pinned in the host arena: the mRoPE positions of one forward call (m213 design note
// D11, M10; m213l design note C5), int64 [3, rows] for K1/K6 then K9's [3, rows + 3] (the call's rows after up to
// three earlier members of its first group); rows = the workspace rows.
inline uint64_t image_positions_bytes(uint64_t rows) { return 3 * (2 * rows + 3) * kI64; }

// MixedPD (mixed-pd design note): the park region of a layer-sliced plan, model::Park's hyper then normed, BF16
// [rows, hc * hidden] each, 256-byte aligned (2 x 80 MiB at 4,096 rows).
inline uint64_t park_bytes(const cere_qwen4_exp_config &cfg, uint64_t rows) {
    return 2 * mtp_hc_rows_bytes(cfg, rows);
}

inline uint64_t pool_bytes(const cere_qwen4_exp_config &cfg, uint64_t layers, uint64_t tokens, bool draft = false) {
    Carver c{nullptr};
    carve_pool(cfg, layers, tokens, c, [](Region, uint32_t, uint8_t *) {}, draft);
    return c.size();
}

}  // namespace cere::model::layout
