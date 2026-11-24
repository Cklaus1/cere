// SelfTune D2 / ST1b-e (selftune design note D2, E2, E3): the decode plan, the op list the decode, prefill and
// verify forwards walk (forward.cpp `run_layers`, one session: a DECODE step and every PREFILL chunk or MixedPD
// segment, embed.tokens to lm_head; `run_rows`, BC2's batched rows; and `run_verify`, the verify chains), and the
// draft plan the MTP draft head's forward walks (mtp.cu `run`: embed.tokens is the MTP fusion, lm_head the hot
// lm_head and argmax). Both are built at compile time from the model descriptor's layer lists
// (executor/models/qwen38-flash-next.json through the generated cere_model_constants.h: gen::kLayerMixer,
// gen::kLayerFfn, gen::kLayerPle; gen::kDraftHeadLayerTypes), then the fusion pass applies its registered peephole
// rules over adjacent ops. CUDA-free, so the host tests check the builders and every rule
// (executor/tests/host/test_plan.cpp).
//
// Slots (D2) and the order the plan emits them, which is the launch order of the hand-written loops it replaces (the
// PDL chains between K1, K2 small-M, K3 and P3b need it unchanged, and the launch-sequence suite
// executor/tests/gpu/test_graph_launches.py compares it):
//   embed.tokens            token embedding (+ the image splice of a prompt call) and the HC expansion
//   per layer L:  [embed.ple]                        when the descriptor marks L as the PLE layer (k4)
//                 mix.hc (attn)  attention.full | attention.linear_gdn  mix.hc combine (attn)
//                 mix.hc (mlp)   ffn.moe                                mix.hc combine (mlp)
//   norm                    the final mixer (hyper_connection_mixer.mix replaces the final norm)
//   lm_head                 logits, top-2 and the sampler stage
// attention.full covers K6 (dense) and K9 (QSA): the sparse path is the impl a full-attention row takes past 2,048
// positions, chosen per row at run time (decode_dense), so attention.sparse_qsa is not an op of its own. A layer kind
// the executor has no slot implementation for (an `ffn` other than `moe`, a mixer other than full / linear attention)
// fails the build (static_assert below), never a launch.
#pragma once

#include <cstdint>

#include "cere_model_constants.h"

namespace cere::model::plan {

enum class Slot : uint8_t {
    kEmbedTokens,
    kEmbedPle,
    kMixHc,
    kAttentionFull,
    kAttentionLinearGdn,
    kFfnMoe,
    kCombineHc,
    kNorm,
    kLmHead,
};

// Which GatedResidual a mix.hc / combine op is: a layer's attn or mlp mix, or the final mixer (Slot::kNorm).
enum class Mix : uint8_t { kAttn, kMlp, kFinal };

struct Op {
    Slot slot = Slot::kEmbedTokens;
    Mix mix = Mix::kAttn;  // mix.hc, combine, norm
    uint16_t layer = 0;    // the layer the op belongs to; embed.tokens 0, norm and lm_head the layer count
    // hc_combine_norm (below). A combine: it also writes the next mix's norm into ws.normed (kernels::hc_combine_norm),
    // that mix's (target_layer, target_mix). A mix / norm: its norm comes from the combine before it.
    bool fused = false;
    uint16_t target_layer = 0;
    Mix target_mix = Mix::kAttn;
};

inline constexpr uint32_t kMaxLayers = 64;
inline constexpr uint32_t kOpsPerLayer = 7;  // embed.ple, two mixes, the mixer, the ffn, two combines
inline constexpr uint32_t kMaxOps = 1 + kMaxLayers * kOpsPerLayer + 2;

// A descriptor's layer list: per layer its `mixer` and `ffn` kinds and whether it is the PLE layer.
struct Layers {
    uint32_t count;
    const char *const *mixer;
    const char *const *ffn;
    const bool *ple;
};

struct Plan {
    Op ops[kMaxOps] = {};
    uint32_t size = 0;
    const char *error = nullptr;  // why the builder refused the layer list (the plan is then empty)
    uint32_t fused = 0;           // ops rewritten by the fusion pass

    constexpr const Op *begin() const { return ops; }
    constexpr const Op *end() const { return ops + size; }
};

constexpr bool same(const char *a, const char *b) {
    while (*a != '\0' && *a == *b) {
        ++a;
        ++b;
    }
    return *a == *b;
}

// A fusion rule (E2): a pattern over two adjacent ops, guarded by their kinds, and the rewrite.
struct Rule {
    const char *name;
    bool (*matches)(const Op &a, const Op &b);
    void (*apply)(Op &a, Op &b);
};

// hc_combine_norm (m23 design note P5R, kernels::hc_combine_norm; contract test K3's bit-identity of the fused
// kernel plus test_plan.cpp): a combine followed directly by a mix also writes that mix's per-stream norm, and the mix
// runs normed. An op between them (embed.ple, whose k4 rewrites ws.hyper) keeps them apart, so the rule never fires
// before the PLE layer.
constexpr bool combine_then_mix(const Op &a, const Op &b) {
    return a.slot == Slot::kCombineHc && (b.slot == Slot::kMixHc || b.slot == Slot::kNorm);
}
constexpr void fuse_combine_norm(Op &a, Op &b) {
    a.fused = true;
    a.target_layer = b.layer;
    a.target_mix = b.mix;
    b.fused = true;
}

inline constexpr Rule kRules[] = {{"hc_combine_norm", combine_then_mix, fuse_combine_norm}};

// An op of one layer (embed.ple, mix.hc, a mixer, ffn.moe, a combine): the walk of a layer range (MixedPD) runs only
// these; embed.tokens, the final norm and lm_head bracket the layers.
constexpr bool per_layer(const Op &op) {
    return op.slot != Slot::kEmbedTokens && op.slot != Slot::kNorm && op.slot != Slot::kLmHead;
}

constexpr Plan refuse(const char *error) {
    Plan p;
    p.error = error;
    return p;
}

// The plan of `layers`, fusion pass applied. fuse_final: the last combine also fuses into the final mixer's norm (a
// forward whose call runs the final mixer right after its layers); without it the rule never fires into Slot::kNorm.
constexpr Plan build_plan(const Layers &layers, bool fuse_final) {
    if (layers.count == 0 || layers.count > kMaxLayers) return refuse("layer count outside 1 .. kMaxLayers");
    Plan p;
    const auto add = [&p](Slot slot, uint32_t layer, Mix mix = Mix::kAttn) {
        Op op;
        op.slot = slot;
        op.layer = uint16_t(layer);
        op.mix = mix;
        p.ops[p.size++] = op;
    };
    add(Slot::kEmbedTokens, 0);
    for (uint32_t L = 0; L < layers.count; ++L) {
        Slot mixer = Slot::kAttentionFull;
        if (same(layers.mixer[L], "linear_attention")) {
            mixer = Slot::kAttentionLinearGdn;
        } else if (!same(layers.mixer[L], "full_attention")) {
            return refuse("a layer's mixer kind has no slot implementation");
        }
        if (!same(layers.ffn[L], "moe")) return refuse("a layer's ffn kind has no slot implementation");
        if (layers.ple[L]) add(Slot::kEmbedPle, L);
        add(Slot::kMixHc, L, Mix::kAttn);
        add(mixer, L);
        add(Slot::kCombineHc, L, Mix::kAttn);
        add(Slot::kMixHc, L, Mix::kMlp);
        add(Slot::kFfnMoe, L);
        add(Slot::kCombineHc, L, Mix::kMlp);
    }
    add(Slot::kNorm, layers.count, Mix::kFinal);
    add(Slot::kLmHead, layers.count);
    for (const Rule &rule : kRules) {
        for (uint32_t i = 0; i + 1 < p.size; ++i) {
            if (!rule.matches(p.ops[i], p.ops[i + 1])) continue;
            if (!fuse_final && p.ops[i + 1].slot == Slot::kNorm) continue;
            rule.apply(p.ops[i], p.ops[i + 1]);
            ++p.fused;
        }
    }
    return p;
}

// The decode plan of the model's layers (forward.cpp).
constexpr Plan build_decode(const Layers &layers) { return build_plan(layers, true); }

// The draft plan of an MTP draft head's layers (mtp.cu `run`, ST1e): the final mixer runs apart from the layers' call
// (mtp::draft_rows scores the gathered last rows of every session; a call without a score stops at hc_out), so the
// last combine stays unfused.
constexpr Plan build_draft(const Layers &layers) { return build_plan(layers, false); }

inline constexpr Plan kDecode =
    build_decode(Layers{gen::kNumHiddenLayers, gen::kLayerMixer, gen::kLayerFfn, gen::kLayerPle});
static_assert(kDecode.error == nullptr, "the descriptor's layer list has a kind without a slot implementation");

// The descriptor's draft_head lists its layers' mixer kinds; the MTP layer's ffn is the K13 BF16 MoE and it has no
// PLE (mtp.h). mtp.cu runs one layer (one set of MtpWeights).
static_assert(gen::kDraftHeadNumHiddenLayers == 1, "mtp.cu implements a one-layer draft head");
inline constexpr const char *kDraftFfn[gen::kDraftHeadNumHiddenLayers] = {"moe"};
inline constexpr bool kDraftPle[gen::kDraftHeadNumHiddenLayers] = {false};
inline constexpr Plan kDraft =
    build_draft(Layers{gen::kDraftHeadNumHiddenLayers, gen::kDraftHeadLayerTypes, kDraftFfn, kDraftPle});
static_assert(kDraft.error == nullptr, "the draft head's layer list has a kind without a slot implementation");

}  // namespace cere::model::plan
