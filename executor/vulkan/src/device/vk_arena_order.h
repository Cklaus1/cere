// VulkanAmd P3 T2 (OD-2): the CUDA arena order rule, restated CUDA-free for the Vulkan memory plan.
//
// Source of truth: executor/cuda/src/model/model.cpp `run_of` (lines 54-132, the K7 build: no P3b `w13` branch,
// which only exists under CERE_MOE_P3B, lines 80-95) and `arena_order` (lines 574-616, again the non-P3b body), as
// documented in model/model.h `ArenaOrder` (lines 88-114). model.cpp includes CUDA kernel headers
// (kernels/k6_attention.h: cudaError_t, kernels/k9_qsa.h, ...), so it cannot be compiled into a host-only Vulkan
// build; this file restates the rule line for line instead of moving it (OD-1: nothing under executor/cuda changes).
// The Vulkan device runs the K7 MoE build (design section 1 `moe_lever()`), so the restated variant is the K7 one:
// every `mlp.experts.E.<field>` of a layer is one run per field, never the P3b up|gate `w13` pairing.
//
// The rule: a tensor belongs to a run (key, member rank) or is a singleton.
//   * `model.language_model.layers.L.mlp.experts.E.<rest>`      -> key `...layers.L.mlp.experts.*.<rest>`, rank E
//   * `model.language_model.layers.L.self_attn.{q,k,v}_proj.weight` -> key `...layers.L.self_attn.qkv`, rank 0/1/2
//   * `model.language_model.layers.L.self_attn.{q,k,v}_proj.weight_scale` -> key `...self_attn.qkv_scale`, rank 0/1/2
//   * `model.language_model.layers.L.ple.ple_embedding.ngram_embedding.shard_N.weight` -> `...ple.shards`, rank N
//   * `... .shard_N.weight_scale` -> `...ple.shard_scales`, rank N
//   * `mtp.layers.0.self_attn.{q,k,v}_proj.weight` -> `mtp.layers.0.self_attn.qkv`, rank 0/1/2
// The key gets "@host" or "@device" from the tensor's placement. DISCARD tensors are skipped. Runs keep the desc order
// of their first member; members are stably sorted by rank; member 0 is aligned like any tensor, later members abut
// (packed). Host test: executor/vulkan/tests/vk_memory_plan_test.cpp rows "arena_order.*".
#pragma once

#include "cere_executor.h"
#include "host/device.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cere::vkdev {

namespace arena_detail {

inline constexpr std::string_view kLayerPrefix = "model.language_model.layers.";

// model.cpp:38-50 parse_index: "<digits>" at s[*at...] up to `stop` (at most 9 digits).
inline bool parse_index(std::string_view s, size_t *at, char stop, uint64_t *out) {
    size_t i = *at;
    uint64_t v = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
        v = v * 10 + uint64_t(s[i] - '0');
        ++i;
    }
    if (i == *at || i - *at > 9 || i >= s.size() || s[i] != stop) return false;
    *out = v;
    *at = i + 1;
    return true;
}

inline bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }

}  // namespace arena_detail

// model.cpp:57-132 run_of (K7 build). True with *key / *member for a run member; false for a singleton.
inline bool run_of(std::string_view name, std::string *key, uint64_t *member) {
    using arena_detail::kLayerPrefix;
    using arena_detail::parse_index;
    using arena_detail::starts_with;
    constexpr std::string_view kMtpLayer = "mtp.layers.0.";
    if (starts_with(name, kMtpLayer)) {
        static constexpr std::string_view kMtpQkv[3] = {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                                                         "self_attn.v_proj.weight"};
        for (uint64_t i = 0; i < 3; ++i) {
            if (name.substr(kMtpLayer.size()) != kMtpQkv[i]) continue;
            *key = std::string(kMtpLayer) + "self_attn.qkv";
            *member = i;
            return true;
        }
        return false;
    }
    if (!starts_with(name, kLayerPrefix)) return false;
    size_t at = kLayerPrefix.size();
    uint64_t layer = 0;
    if (!parse_index(name, &at, '.', &layer)) return false;
    const std::string_view rest = name.substr(at);
    const std::string_view head = name.substr(0, at);
    constexpr std::string_view kExperts = "mlp.experts.";
    if (starts_with(rest, kExperts)) {
        size_t e = kExperts.size();
        uint64_t expert = 0;
        if (!parse_index(rest, &e, '.', &expert)) return false;
        *key = std::string(head) + "mlp.experts.*." + std::string(rest.substr(e));
        *member = expert;
        return true;
    }
    static constexpr std::string_view kQkv[3] = {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                                                  "self_attn.v_proj.weight"};
    static constexpr std::string_view kQkvScale[3] = {"self_attn.q_proj.weight_scale", "self_attn.k_proj.weight_scale",
                                                       "self_attn.v_proj.weight_scale"};
    for (uint64_t i = 0; i < 3; ++i) {
        if (rest == kQkv[i] || rest == kQkvScale[i]) {
            *key = std::string(head) + (rest == kQkv[i] ? "self_attn.qkv" : "self_attn.qkv_scale");
            *member = i;
            return true;
        }
    }
    constexpr std::string_view kShard = "ple.ple_embedding.ngram_embedding.shard_";
    if (starts_with(rest, kShard)) {
        size_t s = kShard.size();
        uint64_t shard = 0;
        if (!parse_index(rest, &s, '.', &shard)) return false;
        const std::string_view field = rest.substr(s);
        if (field == "weight") {
            *key = std::string(head) + "ple.shards";
        } else if (field == "weight_scale") {
            *key = std::string(head) + "ple.shard_scales";
        } else {
            return false;
        }
        *member = shard;
        return true;
    }
    return false;
}

// One run of the arena order: its key ("" for a singleton) and member tensor indices in placement order.
struct ArenaRun {
    std::string key;                // run_of key + "@host"/"@device"; empty for a singleton
    std::vector<uint64_t> members;  // tensor indices into plan.tensors, rank order
    std::vector<uint64_t> ranks;    // parallel to members: the run_of member rank (0 for a singleton)
};

// model.cpp:574-616 arena_order, returned as runs (the CUDA ArenaOrder is the concatenation: order = members of every
// run in turn, packed[i] = not the first member of its run). Throws std::bad_alloc only.
inline std::vector<ArenaRun> arena_runs(const host::ModelPlan &plan) {
    std::vector<ArenaRun> runs;
    std::unordered_map<std::string, size_t> keyed;
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> members;  // (rank, tensor) per run, desc order
    std::string key;
    for (uint64_t i = 0; i < plan.tensors.size(); ++i) {
        const host::TensorPlan &t = plan.tensors[i];
        if (t.placement == CERE_PLACEMENT_DISCARD) continue;
        uint64_t member = 0;
        if (run_of(t.name, &key, &member)) {
            key += t.placement == CERE_PLACEMENT_HOST_PINNED ? "@host" : "@device";
            const auto [it, added] = keyed.emplace(key, runs.size());
            if (added) {
                runs.push_back(ArenaRun{key, {}, {}});
                members.emplace_back();
            }
            members[it->second].emplace_back(member, i);
        } else {
            runs.push_back(ArenaRun{std::string(), {}, {}});
            members.push_back({{0, i}});
        }
    }
    for (size_t r = 0; r < runs.size(); ++r) {
        auto &m = members[r];
        std::stable_sort(m.begin(), m.end(), [](const auto &x, const auto &y) { return x.first < y.first; });
        runs[r].members.reserve(m.size());
        runs[r].ranks.reserve(m.size());
        for (const auto &[rank, index] : m) {
            runs[r].members.push_back(index);
            runs[r].ranks.push_back(rank);
        }
    }
    return runs;
}

}  // namespace cere::vkdev
