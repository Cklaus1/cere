// Inkling 2 Mini (inkling design note 3.1, D-I2, slice S5): the visit plan of the looped forward. The 24 physical
// decoder layers run as 48 virtual visits per token (REF modeling_inkling2_loop.py :1359-1440): layers 0-5 once
// (pass 0), layers 6-17 three times with shared weights (passes 0, 1, 2), layers 18-23 once. Weights stay per
// physical layer; every visit owns its own KV (sliding ring or full pool) and its 4 conv states (D-I3, D-I4).
// Header-only and CUDA-free (executor/tests/host/test_inkling_plan.cpp checks it against the REF loop).
//
// Residual-stream bookkeeping (REF :886-910, :1362-1443; executor/ref/inkling/depth_ref.h):
// - h_L is the BF16 input of a visit of layer L; h_0 the normed embedding. A boundary layer (L % 6 == 0: 0, 6, 12,
//   18) appends its input to the block-state list B, in slot L / 6 (h_0, h_6, h_12, h_18).
// - Every pass restarts from the pre-loop states (h_6, B = [h_0]), so h_6 (slot 1) is the same tensor in all three
//   passes and only h_12 (slot 2) differs per pass.
// - Passes 1 and 2 receive messages: for loop layer i = L - 6 the depth mix of [h_12 of the previous pass, the
//   previous pass's output] with message_input.query[i] (4 queries: q, k, v, mlp).
// - The post-loop layers continue from pass 2's list; the final mix reads [h_0, h_6, h_12 of pass 2, h_18, h_24].
#pragma once

#include <array>
#include <cstdint>

namespace cere::inkling::plan {

constexpr int kLayers = 24;
constexpr int kLoopStart = 6;
constexpr int kLoopEnd = 18;
constexpr int kLoopLen = kLoopEnd - kLoopStart;  // 12
constexpr int kPasses = 3;
constexpr int kVisits = kLayers + (kPasses - 1) * kLoopLen;  // 48
constexpr int kBlockEvery = 6;                                // n_layers_per_attn_res_block
constexpr int kBlockSlots = kLayers / kBlockEvery;            // 4: h_0, h_6, h_12, h_18
constexpr int kFullVisits = 8;
constexpr int kSlidingVisits = kVisits - kFullVisits;  // 40

// Layers 5, 11, 17 and 23 are full attention (config local_layer_ids lists every other layer).
constexpr bool is_full_layer(int layer) { return layer % kBlockEvery == kBlockEvery - 1; }
constexpr bool is_loop_layer(int layer) { return layer >= kLoopStart && layer < kLoopEnd; }

struct Visit {
    int layer = 0;          // physical layer (weights)
    int pass = 0;           // 0 outside the loop
    bool full = false;      // full attention (dynamic K/V conv, 3 attention queries, paged pool)
    bool boundary = false;  // layer % 6 == 0: the input joins the block states, the residual restarts
    bool messages = false;  // a loop layer in pass >= 1: message gates on q, k, v and the MLP input
    int loop_index = -1;    // layer - 6 for a loop layer
    int full_index = -1;    // 0..7 among the full visits (the KV pool of the visit)
    int sliding_index = -1; // 0..39 among the sliding visits (the KV ring of the visit)
    int cache_layer = 0;    // the REF DynamicCache layer of the visit (:1352-1355, cache_offset :1401)
    int attn_query = -1;    // attn_res.attn_queries index (layer - 1); -1 at layer 0 (no attention input mix)
};

constexpr int layer_of(int v) {
    return v < kLoopStart ? v : v < kLoopStart + kPasses * kLoopLen ? kLoopStart + (v - kLoopStart) % kLoopLen
                                                                    : v - (kPasses - 1) * kLoopLen;
}
constexpr int pass_of(int v) {
    return v < kLoopStart || v >= kLoopStart + kPasses * kLoopLen ? 0 : (v - kLoopStart) / kLoopLen;
}
// The visit of (layer, pass); pass must be 0 outside the loop.
constexpr int visit_of(int layer, int pass) {
    return layer < kLoopStart ? layer : layer < kLoopEnd ? layer + pass * kLoopLen : layer + (kPasses - 1) * kLoopLen;
}

constexpr std::array<Visit, kVisits> visits() {
    std::array<Visit, kVisits> out{};
    int full = 0, sliding = 0;
    for (int v = 0; v < kVisits; ++v) {
        Visit &x = out[size_t(v)];
        x.layer = layer_of(v);
        x.pass = pass_of(v);
        x.full = is_full_layer(x.layer);
        x.boundary = x.layer % kBlockEvery == 0;
        x.messages = is_loop_layer(x.layer) && x.pass > 0;
        x.loop_index = is_loop_layer(x.layer) ? x.layer - kLoopStart : -1;
        x.full_index = x.full ? full++ : -1;
        x.sliding_index = x.full ? -1 : sliding++;
        x.cache_layer = x.pass == 0 ? x.layer : kLayers + (x.pass - 1) * kLoopLen + x.layer - kLoopStart;
        x.attn_query = x.layer - 1;
    }
    return out;
}

constexpr std::array<Visit, kVisits> kPlan = visits();

// The last visit of pass p (its output is the pass's output h_18 of that pass).
constexpr int pass_last_visit(int pass) { return kLoopStart + (pass + 1) * kLoopLen - 1; }

// MTP target_layer_ids [5, 23, 41, 48] (config): visit outputs 1-based; 48 = the final stream (S9).
constexpr int kMtpTargets[4] = {5, 23, 41, 48};

}  // namespace cere::inkling::plan
