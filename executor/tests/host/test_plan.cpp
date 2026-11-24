// SelfTune ST1b/ST1e (selftune design note D2, E2, E3): model/plan.h's decode and draft plan builders and their
// fusion rule. The Qwen3.8 plan walks the descriptor's 48 layers in the launch order of the loops it replaced, its
// attention slots equal the compiled support table's full_attention_layers (the config every model must match), and
// hc_combine_norm fuses every combine into the mix after it except before the PLE layer. The draft plan is the MTP
// layer's op list in mtp.cu's launch order, its last combine unfused (the final mixer runs apart). Synthetic layer
// lists check the rule's guard and the builders' refusals.
#include "check.h"

#include "host/support.h"
#include "model/plan.h"

#include <cstdint>

namespace {

namespace plan = cere::model::plan;
using plan::Mix;
using plan::Slot;

constexpr const char *kFull = "full_attention";
constexpr const char *kLinear = "linear_attention";
constexpr const char *kMoe = "moe";

// The op list the hand-written run_layers / run_rows emitted, layer by layer (the oracle; fusion flags apart).
void check_order(const plan::Plan &p, uint32_t layers, const char *const *mixer, const bool *ple) {
    uint32_t i = 0;
    const auto expect = [&](Slot slot, uint32_t layer, Mix mix) {
        CHECK(i < p.size);
        if (i >= p.size) return;
        CHECK(p.ops[i].slot == slot);
        CHECK_EQ(p.ops[i].layer, layer);
        if (slot == Slot::kMixHc || slot == Slot::kCombineHc || slot == Slot::kNorm) CHECK(p.ops[i].mix == mix);
        ++i;
    };
    expect(Slot::kEmbedTokens, 0, Mix::kAttn);
    for (uint32_t L = 0; L < layers; ++L) {
        if (ple[L]) expect(Slot::kEmbedPle, L, Mix::kAttn);
        expect(Slot::kMixHc, L, Mix::kAttn);
        expect(plan::same(mixer[L], kFull) ? Slot::kAttentionFull : Slot::kAttentionLinearGdn, L, Mix::kAttn);
        expect(Slot::kCombineHc, L, Mix::kAttn);
        expect(Slot::kMixHc, L, Mix::kMlp);
        expect(Slot::kFfnMoe, L, Mix::kAttn);
        expect(Slot::kCombineHc, L, Mix::kMlp);
    }
    expect(Slot::kNorm, layers, Mix::kFinal);
    expect(Slot::kLmHead, layers, Mix::kAttn);
    CHECK_EQ(i, p.size);
}

// hc_combine_norm's contract: a combine is fused exactly when a mix / norm follows it, and then names that mix; a
// mix / norm runs normed exactly when a combine precedes it; nothing else is touched.
void check_fusion(const plan::Plan &p) {
    uint32_t fused = 0;
    for (uint32_t i = 0; i < p.size; ++i) {
        const plan::Op &op = p.ops[i];
        const bool next_mix = i + 1 < p.size && (p.ops[i + 1].slot == Slot::kMixHc || p.ops[i + 1].slot == Slot::kNorm);
        const bool prev_combine = i > 0 && p.ops[i - 1].slot == Slot::kCombineHc;
        if (op.slot == Slot::kCombineHc) {
            CHECK_EQ(op.fused, next_mix);
            if (next_mix) {
                CHECK_EQ(op.target_layer, p.ops[i + 1].layer);
                CHECK(op.target_mix == p.ops[i + 1].mix);
            }
        } else if (op.slot == Slot::kMixHc || op.slot == Slot::kNorm) {
            CHECK_EQ(op.fused, prev_combine);
        } else {
            CHECK(!op.fused);
        }
        fused += op.fused ? 1 : 0;
    }
    CHECK_EQ(fused, p.fused * 2);
}

}  // namespace

int main() {
    namespace gen = cere::gen;
    // The Qwen3.8 descriptor's plan.
    const plan::Plan &q = plan::kDecode;
    CHECK(q.error == nullptr);
    CHECK_EQ(q.size, 1 + 1 + 48 * 6 + 2);
    check_order(q, gen::kNumHiddenLayers, gen::kLayerMixer, gen::kLayerPle);
    check_fusion(q);
    // Every combine fuses but layer 0's mlp combine (the PLE layer 1 follows it): 2 x 48 - 1 rules fired.
    CHECK_EQ(q.fused, 2 * 48 - 1);
    const uint64_t full = cere::host::a4_support().config.full_attention_layers;
    uint32_t mixers = 0;
    for (const plan::Op &op : q) {
        if (op.slot == Slot::kEmbedPle) CHECK_EQ(op.layer, gen::kPleLayerIds[0] - 1);
        if (op.slot != Slot::kAttentionFull && op.slot != Slot::kAttentionLinearGdn) continue;
        ++mixers;
        CHECK_EQ(op.slot == Slot::kAttentionFull, (full >> op.layer) & 1);
    }
    CHECK_EQ(mixers, 48);
    for (uint32_t i = 0; i + 1 < q.size; ++i) {
        if (q.ops[i + 1].slot != Slot::kEmbedPle) continue;
        CHECK(q.ops[i].slot == Slot::kCombineHc && !q.ops[i].fused);  // the guard: never fused before the PLE layer
        CHECK(!q.ops[i + 2].fused);                                    // and the PLE layer's attn mix runs unnormed
    }
    const plan::Op &last_combine = q.ops[q.size - 3];
    CHECK(last_combine.fused && last_combine.target_mix == Mix::kFinal && last_combine.target_layer == 48);

    // No PLE layer: the rule fires at every combine.
    {
        static constexpr const char *mixer[3] = {kLinear, kFull, kLinear};
        static constexpr const char *ffn[3] = {kMoe, kMoe, kMoe};
        static constexpr bool ple[3] = {false, false, false};
        constexpr plan::Plan p = plan::build_decode(plan::Layers{3, mixer, ffn, ple});
        CHECK(p.error == nullptr);
        check_order(p, 3, mixer, ple);
        check_fusion(p);
        CHECK_EQ(p.fused, 6);
    }
    // PLE at layer 0: no combine precedes it, so every combine still fuses; layer 0's attn mix runs unnormed.
    {
        static constexpr const char *mixer[2] = {kFull, kFull};
        static constexpr const char *ffn[2] = {kMoe, kMoe};
        static constexpr bool ple[2] = {true, false};
        constexpr plan::Plan p = plan::build_decode(plan::Layers{2, mixer, ffn, ple});
        check_order(p, 2, mixer, ple);
        check_fusion(p);
        CHECK_EQ(p.fused, 4);
    }
    // PLE at the last layer: the combine before it stays unfused.
    {
        static constexpr const char *mixer[2] = {kLinear, kLinear};
        static constexpr const char *ffn[2] = {kMoe, kMoe};
        static constexpr bool ple[2] = {false, true};
        constexpr plan::Plan p = plan::build_decode(plan::Layers{2, mixer, ffn, ple});
        check_order(p, 2, mixer, ple);
        check_fusion(p);
        CHECK_EQ(p.fused, 3);
    }
    // Refusals: a kind without a slot implementation, an empty or oversized layer list.
    {
        static constexpr const char *dense[1] = {"dense"};
        static constexpr const char *moe[1] = {kMoe};
        static constexpr const char *full[1] = {kFull};
        static constexpr const char *sliding[1] = {"sliding_attention"};
        static constexpr bool ple[1] = {false};
        constexpr plan::Plan ffn_dense = plan::build_decode(plan::Layers{1, full, dense, ple});
        CHECK(ffn_dense.error != nullptr && ffn_dense.size == 0);
        constexpr plan::Plan unknown = plan::build_decode(plan::Layers{1, sliding, moe, ple});
        CHECK(unknown.error != nullptr && unknown.size == 0);
        constexpr plan::Plan empty = plan::build_decode(plan::Layers{0, full, moe, ple});
        CHECK(empty.error != nullptr);
        constexpr plan::Plan big = plan::build_decode(plan::Layers{plan::kMaxLayers + 1, full, moe, ple});
        CHECK(big.error != nullptr);
    }
    // The draft plan (the descriptor's one-layer MTP head): embed.tokens (the fusion), attn mix, attention, the attn
    // combine fused into the mlp mix, MoE, an unfused combine, then the final mixer and lm_head.
    {
        const plan::Plan &d = plan::kDraft;
        CHECK(d.error == nullptr);
        CHECK_EQ(gen::kDraftHeadNumHiddenLayers, 1u);
        static constexpr bool ple[1] = {false};
        check_order(d, 1, gen::kDraftHeadLayerTypes, ple);
        CHECK_EQ(d.size, 9u);
        CHECK_EQ(d.fused, 1u);
        CHECK(d.ops[3].slot == Slot::kCombineHc && d.ops[3].fused && d.ops[3].target_mix == Mix::kMlp);
        CHECK(d.ops[4].slot == Slot::kMixHc && d.ops[4].fused);
        CHECK(d.ops[6].slot == Slot::kCombineHc && !d.ops[6].fused);
        CHECK(d.ops[7].slot == Slot::kNorm && !d.ops[7].fused);
        // The same layer list as a decode plan fuses the last combine too.
        constexpr plan::Plan as_decode = plan::build_decode(
            plan::Layers{gen::kDraftHeadNumHiddenLayers, gen::kDraftHeadLayerTypes, plan::kDraftFfn, plan::kDraftPle});
        CHECK_EQ(as_decode.fused, 2u);
        static constexpr const char *dense[1] = {"dense"};
        constexpr plan::Plan refused = plan::build_draft(plan::Layers{1, gen::kDraftHeadLayerTypes, dense, ple});
        CHECK(refused.error != nullptr && refused.size == 0);
    }
    return cere::test::finish("test_plan");
}
