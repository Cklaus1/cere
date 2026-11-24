// Inkling 2 Mini (inkling design note 3.1, D-I2, slice S5): the visit plan of executor/cuda/src/model/inkling/plan.h
// against the REF model loop (modeling_inkling2_loop.py :1352-1443) replayed here: pre-loop layers 0-5 once, loop
// layers 6-17 for passes 0-2 (passes >= 1 with messages and the cache offset num_hidden_layers + (p - 1) *
// num_loop_layers - loop_start), post-loop layers 18-23 once. Also the block-state count each visit sees (REF: a
// boundary layer appends its input before it runs; passes >= 1 restart from the pre-loop list) and the KV indices.
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

#include "../../cuda/src/model/inkling/plan.h"
#include "check.h"

namespace {

namespace P = cere::inkling::plan;

struct RefVisit {
    int layer;
    int pass;
    int cache_layer;
    bool messages;
    int blocks;  // block states the visit's depth mixes read (before its own input joins on a boundary)
};

// REF :1352-1443 with the config's loop_config {6, 18, 3} and 24 layers.
std::vector<RefVisit> ref_visits() {
    std::vector<RefVisit> out;
    int blocks = 0;
    auto run = [&](int layer, int pass, int offset, bool messages) {
        out.push_back({layer, pass, layer + offset, messages, blocks});
        if (layer % 6 == 0) ++blocks;  // decoder_layer.is_block_boundary: block_states += [input]
    };
    for (int l = 0; l < 6; ++l) run(l, 0, 0, false);
    const int pre_blocks = blocks;
    for (int l = 6; l < 18; ++l) run(l, 0, 0, false);
    for (int p = 1; p < 3; ++p) {
        blocks = pre_blocks;  // each pass restarts from the pre-loop states
        for (int l = 6; l < 18; ++l) run(l, p, 24 + (p - 1) * 12 - 6, true);
    }
    for (int l = 18; l < 24; ++l) run(l, 0, 0, false);  // continues from pass 2's list
    return out;
}

void test_against_ref() {
    const std::vector<RefVisit> ref = ref_visits();
    CHECK_EQ(int(ref.size()), P::kVisits);
    for (int v = 0; v < P::kVisits && v < int(ref.size()); ++v) {
        const P::Visit &x = P::kPlan[size_t(v)];
        const RefVisit &r = ref[size_t(v)];
        CHECK_EQ(x.layer, r.layer);
        CHECK_EQ(x.pass, r.pass);
        CHECK_EQ(x.cache_layer, r.cache_layer);
        CHECK_EQ(x.messages, r.messages);
        CHECK_EQ(x.boundary, r.layer % 6 == 0);
        CHECK_EQ((x.layer + P::kBlockEvery - 1) / P::kBlockEvery, r.blocks);  // h_6j with 6j < layer
        CHECK_EQ(P::layer_of(v), r.layer);
        CHECK_EQ(P::pass_of(v), r.pass);
        CHECK_EQ(P::visit_of(r.layer, r.pass), v);
        CHECK_EQ(x.attn_query, r.layer - 1);
        CHECK_EQ(x.loop_index, r.layer >= 6 && r.layer < 18 ? r.layer - 6 : -1);
    }
}

void test_kinds() {
    const int full_visits[P::kFullVisits] = {5, 11, 17, 23, 29, 35, 41, 47};
    std::set<int> full(full_visits, full_visits + P::kFullVisits);
    std::set<int> cache, sliding_idx;
    int next_full = 0;
    for (int v = 0; v < P::kVisits; ++v) {
        const P::Visit &x = P::kPlan[size_t(v)];
        CHECK_EQ(x.full, full.count(v) == 1);
        CHECK_EQ(x.full, x.layer == 5 || x.layer == 11 || x.layer == 17 || x.layer == 23);
        if (x.full) {
            CHECK_EQ(x.full_index, next_full);
            ++next_full;
            CHECK_EQ(x.sliding_index, -1);
        } else {
            CHECK_EQ(x.full_index, -1);
            CHECK(x.sliding_index >= 0 && x.sliding_index < P::kSlidingVisits);
            sliding_idx.insert(x.sliding_index);
        }
        cache.insert(x.cache_layer);
    }
    CHECK_EQ(next_full, P::kFullVisits);
    CHECK_EQ(int(sliding_idx.size()), P::kSlidingVisits);
    CHECK_EQ(int(cache.size()), P::kVisits);  // every visit owns its own cache layer
    CHECK_EQ(*cache.begin(), 0);
    CHECK_EQ(*cache.rbegin(), P::kVisits - 1);
    for (int p = 0; p < P::kPasses; ++p) {
        const int last = P::pass_last_visit(p);
        CHECK_EQ(P::kPlan[size_t(last)].layer, P::kLoopEnd - 1);
        CHECK_EQ(P::kPlan[size_t(last)].pass, p);
    }
    // MTP target_layer_ids (config): 1-based visit outputs, 48 = the final stream.
    CHECK_EQ(P::kMtpTargets[0], 5);
    CHECK_EQ(P::kMtpTargets[3], P::kVisits);
}

}  // namespace

int main() {
    test_against_ref();
    test_kinds();
    return cere::test::finish("test_inkling_plan");
}
