// VulkanAmd P3 T2: host-only tests of the Vulkan memory plan (executor/vulkan/src/device/vk_memory_plan.h). No Vulkan,
// no GPU. Rows print PASS / FAIL / SKIP; the exit status is non-zero when any row fails.
//
// Usage: vk_memory_plan_test [ARTIFACT_DIR [INVENTORY_JSON]]
//   ARTIFACT_DIR    the S3 artifact (read-only; only manifest.json, the safetensors HEADERS and the size of
//                   draft/token_map.u32le are read, never tensor payload bytes). Absent or unreadable: the real row SKIPs.
//   INVENTORY_JSON  configs/artifacts/qwen38-flash-next-fp4local-cere-s3.inventory.json: the placement totals the real
//                   plan must reproduce.
#include "device/vk_arena_order.h"
#include "device/vk_kernel_sizes.h"
#include "device/vk_memory_plan.h"
#include "host/kv_pool.h"
#include "host/support.h"
#include "model/layout.h"
#include "model/mxfp8_plan.h"

#include "mini_json.h"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace cere;
using namespace cere::vkdev;
namespace layout = cere::model::layout;

namespace {

int g_fail = 0, g_pass = 0, g_skip = 0;

void row(const char *name, bool ok, const std::string &detail = "") {
    std::printf("%-58s %s%s%s\n", name, ok ? "PASS" : "FAIL", detail.empty() ? "" : "  ", detail.c_str());
    (ok ? g_pass : g_fail)++;
}
void skip(const char *name, const std::string &why) {
    std::printf("%-58s SKIP  %s\n", name, why.c_str());
    ++g_skip;
}

std::string gib(uint64_t b) {
    char s[64];
    std::snprintf(s, sizeof s, "%.3f GiB", double(b) / double(uint64_t(1) << 30));
    return s;
}
std::string commas(uint64_t v) {
    std::string s = std::to_string(v);
    for (int i = int(s.size()) - 3; i > 0; i -= 3) s.insert(size_t(i), ",");
    return s;
}

host::TensorPlan T(const std::string &name, uint32_t dtype, uint32_t placement, std::initializer_list<uint64_t> shape) {
    host::TensorPlan t{};
    t.name = name;
    t.dtype = dtype;
    t.placement = placement;
    t.ndim = uint32_t(shape.size());
    uint64_t n = 1, d = 0;
    for (uint64_t s : shape) {
        t.shape[d++] = s;
        n *= s;
    }
    t.bytes = n * host::dtype_size(dtype);
    return t;
}

constexpr uint32_t DEV = CERE_PLACEMENT_DEVICE, HOST = CERE_PLACEMENT_HOST_PINNED, DISC = CERE_PLACEMENT_DISCARD;
const std::string L = "model.language_model.layers.";

host::ModelPlan base_plan() {
    host::ModelPlan p{};
    p.support = &host::a4_support();
    p.config = p.support->config;
    p.max_sessions = 1;
    p.max_context_tokens = 32768;
    p.kv_pool_tokens = 32768;
    return p;
}

// A small synthetic plan: expert runs, an MXFP8 q|k|v, singletons of both alignments, an NVFP4 PLE table of `shards`
// shards x `shard_rows` rows, PLE scalars, an MTP tensor and a DISCARD F16 tensor.
host::ModelPlan synthetic_plan(uint32_t drafts, uint32_t shards = 4, uint64_t shard_rows = 1000, uint32_t experts = 8) {
    host::ModelPlan p = base_plan();
    p.speculative_draft_tokens = drafts;
    if (drafts) {
        for (uint32_t i = 0; i < 4096; ++i) p.draft_token_map.push_back(i * 2);
    }
    auto &v = p.tensors;
    v.push_back(T("model.language_model.embed_tokens.weight", CERE_DTYPE_BF16, DEV, {1000, 2560}));
    for (uint32_t e = 0; e < experts; ++e) {
        const std::string x = L + "0.mlp.experts." + std::to_string(e) + ".gate_proj.";
        v.push_back(T(x + "weight", CERE_DTYPE_U8, DEV, {640, 1280}));
        v.push_back(T(x + "weight_scale", CERE_DTYPE_F8_E4M3, DEV, {640, 160}));
        v.push_back(T(x + "weight_scale_2", CERE_DTYPE_F32, DEV, {}));
        v.push_back(T(x + "input_scale", CERE_DTYPE_F32, DEV, {}));
    }
    for (const char *q : {"q", "k", "v"}) {
        const uint64_t n = q[0] == 'q' ? 12288 : 512;
        v.push_back(T(L + "3.self_attn." + q + "_proj.weight", CERE_DTYPE_F8_E4M3, DEV, {n, 2560}));
        v.push_back(T(L + "3.self_attn." + q + "_proj.weight_scale", CERE_DTYPE_U8, DEV, {n, 80}));
    }
    v.push_back(T(L + "0.attn_hyper_connection.hc_norm.weight", CERE_DTYPE_BF16, DEV, {10240}));
    v.push_back(T(L + "3.self_attn.q_norm.weight", CERE_DTYPE_BF16, DEV, {256}));
    v.push_back(T(L + "0.attn_hyper_connection.input_mix_weight_up.weight", CERE_DTYPE_BF16, DEV, {10240, 320}));
    const std::string ple = L + "1.ple.ple_embedding.";
    for (uint32_t s0 = 0; s0 < shards; ++s0) {  // table / scale shards interleaved, in reverse: arena_order sorts them
        const uint32_t id = shards - 1 - s0;
        v.push_back(T(ple + "ngram_embedding.shard_" + std::to_string(id) + ".weight", CERE_DTYPE_U8, HOST,
                      {shard_rows, 80}));
        v.push_back(T(ple + "ngram_embedding.shard_" + std::to_string(id) + ".weight_scale", CERE_DTYPE_F8_E4M3, HOST,
                      {shard_rows, 10}));
    }
    v.push_back(T(ple + "ngram_embedding.weight_scale_2", CERE_DTYPE_F32, HOST, {1}));
    v.push_back(T(ple + "layer_multipliers", CERE_DTYPE_I64, HOST, {3}));
    v.push_back(T(ple + "ngram_heads_offsets", CERE_DTYPE_I64, HOST, {16}));
    v.push_back(T("mtp.fc_hidden.weight", CERE_DTYPE_BF16, drafts ? DEV : DISC, {2560, 2560}));
    v.push_back(T("vision.patch_embed.weight", CERE_DTYPE_F16, DISC, {16, 16}));
    return p;
}

// ---- invariants ----
bool invariants(const host::ModelPlan &plan, const VkLayout &l, std::string *why) {
    const auto bad = [why](const std::string &m) {
        *why = m;
        return false;
    };
    if (l.status != CERE_OK) return bad("status " + std::to_string(l.status) + ": " + l.error);
    const VkLimits &lim = l.limits;
    uint64_t dev = 0, host = 0;
    for (size_t b = 0; b < l.blocks.size(); ++b) {
        const VkBlock &k = l.blocks[b];
        if (k.bytes % lim.block_granule != 0 || k.bytes > lim.max_buffer_bytes || k.used > k.bytes || k.bytes == 0) {
            return bad("block " + std::to_string(b) + " bytes " + std::to_string(k.bytes) + " used " +
                       std::to_string(k.used));
        }
        (k.heap == VkHeap::kDeviceLocal ? dev : host) += k.bytes;
    }
    const host::MemoryEstimate e = vk_estimate(l);
    if (dev != l.device_bytes || host != l.host_bytes || e.device_bytes != dev || e.host_bytes != host) {
        return bad("estimate != sum of blocks");
    }
    // No overlap: every piece and region per block.
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> spans(l.blocks.size());
    for (size_t i = 0; i < plan.tensors.size(); ++i) {
        const host::TensorPlan &t = plan.tensors[i];
        const uint64_t f = l.piece_first[i], n = l.piece_first[i + 1] - f;
        if (t.placement == CERE_PLACEMENT_DISCARD) {
            if (n != 0) return bad("DISCARD tensor " + t.name + " has storage");
            continue;
        }
        if (n == 0) return bad("tensor " + t.name + " not placed");
        uint64_t at = 0;
        for (uint64_t p = f; p < f + n; ++p) {
            const VkPiece &q = l.pieces[p];
            if (q.tensor_offset != at) return bad("tensor " + t.name + " pieces not contiguous");
            at += q.bytes;
            const VkBlock &blk = l.blocks[q.block];
            const bool want_host = t.placement == CERE_PLACEMENT_HOST_PINNED;
            if ((blk.heap == VkHeap::kHostHeap) != want_host) return bad("tensor " + t.name + " in the wrong heap");
            if (!want_host && blk.purpose != VkPurpose::kWeights) return bad("tensor " + t.name + " outside weights");
            if (q.offset + q.bytes > blk.used) return bad("tensor " + t.name + " past its block's end");
            if (q.bytes) spans[q.block].emplace_back(q.offset, q.offset + q.bytes);
        }
        if (at != t.bytes) return bad("tensor " + t.name + " pieces cover " + std::to_string(at) + " of " +
                                      std::to_string(t.bytes));
    }
    for (const VkRegion &r : l.regions) {
        if (r.offset % lim.region_alignment != 0) return bad(std::string("region ") + vk_region_name(r.kind) + " misaligned");
        if (r.offset + r.bytes > l.blocks[r.block].used) return bad("region past block end");
        if (r.bytes) spans[r.block].emplace_back(r.offset, r.offset + r.bytes);
    }
    for (size_t b = 0; b < spans.size(); ++b) {
        auto &s = spans[b];
        std::sort(s.begin(), s.end());
        for (size_t j = 1; j < s.size(); ++j) {
            if (s[j].first < s[j - 1].second) {
                return bad("overlap in block " + std::to_string(b) + " at " + std::to_string(s[j].first));
            }
        }
    }
    // Alignment of every run's first member (R3) and of the PLE chunks (R4).
    for (const ArenaRun &r : arena_runs(plan)) {
        const uint64_t i = r.members[0];
        const host::TensorPlan &t = plan.tensors[i];
        const VkPiece &q = l.pieces[l.piece_first[i]];
        const bool ple_split = r.key.find(".ple.shard") != std::string::npos;
        const uint64_t a = t.bytes < lim.small_tensor_bytes ? lim.small_alignment : lim.region_alignment;
        if (q.offset % (ple_split ? lim.region_alignment : a) != 0) return bad("tensor " + t.name + " misaligned");
        // Packed members abut.
        if (!ple_split) {
            uint64_t at = q.offset;
            for (uint64_t m : r.members) {
                const VkPiece &qm = l.pieces[l.piece_first[m]];
                if (qm.block != q.block || qm.offset != at) return bad("run " + r.key + " not packed");
                at += plan.tensors[m].bytes;
            }
        }
    }
    for (uint32_t b = 0; b < l.ple.blocks; ++b) {
        if (l.ple.table_offset[b] % 256 != 0) return bad("PLE chunk misaligned");
    }
    return true;
}

bool same_layout(const host::ModelPlan &plan, const VkLimits &lim) {
    const VkLayout a = vk_plan_layout(plan, lim), b = vk_plan_layout(plan, lim);
    return vk_layout_description(plan, a) == vk_layout_description(plan, b) &&
           vk_layout_digest(plan, a) == vk_layout_digest(plan, b);
}

// K4's SplitTable addressing (k4_ple_vk.h:8-16, k4_common.comp:97, :124-129) of global row r vs vk_locate on the
// row's shard tensor, with fake block addresses.
bool split_geometry_ok(const host::ModelPlan &plan, const VkLayout &l, std::string *why) {
    const VkPleSplit &s = l.ple;
    std::vector<uint64_t> addr(l.blocks.size());
    for (size_t b = 0; b < addr.size(); ++b) addr[b] = (uint64_t(b) + 1) << 40;
    uint64_t table[128];
    if (!vk_ple_block_table(l, addr.data(), table)) {
        *why = "no block table";
        return false;
    }
    if (uint64_t(s.blocks) * s.rows_per_block < s.rows || s.blocks > 64 ||
        uint64_t(s.rows_per_block) * s.row_bytes >= (uint64_t(1) << 32)) {
        *why = "view constraints (k4 view_ok) violated";
        return false;
    }
    // Shard tensors by rank.
    std::vector<uint64_t> shard(s.shards), scale(s.shards);
    for (const ArenaRun &r : arena_runs(plan)) {
        if (r.key.find(".ple.shards@host") != std::string::npos) {
            for (size_t m = 0; m < r.members.size(); ++m) shard[m] = r.members[m];
        }
        if (r.key.find(".ple.shard_scales@host") != std::string::npos) {
            for (size_t m = 0; m < r.members.size(); ++m) scale[m] = r.members[m];
        }
    }
    std::vector<uint64_t> probes = {0, 1, s.rows - 1, s.shard_rows - 1, s.shard_rows, s.rows_per_block - 1,
                                    s.rows_per_block, s.rows / 2};
    uint64_t x = 88172645463325252ull;
    for (int i = 0; i < 2000; ++i) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        probes.push_back(x % s.rows);
    }
    for (uint64_t r : probes) {
        if (r >= s.rows) continue;
        const uint64_t b = r / s.rows_per_block, in = r - b * s.rows_per_block;
        const uint64_t k4_row = table[b] + in * s.row_bytes;
        const std::vector<VkPiece> p = vk_locate(l, shard[r / s.shard_rows], (r % s.shard_rows) * s.row_bytes, s.row_bytes);
        if (p.size() != 1 || addr[p[0].block] + p[0].offset != k4_row) {
            *why = "row " + std::to_string(r) + ": K4 address differs from vk_locate";
            return false;
        }
        if (s.scale_bytes) {
            const uint64_t k4_scale = table[64 + b] + in * s.scale_bytes;
            const std::vector<VkPiece> q =
                vk_locate(l, scale[r / s.shard_rows], (r % s.shard_rows) * s.scale_bytes, s.scale_bytes);
            if (q.size() != 1 || addr[q[0].block] + q[0].offset != k4_scale) {
                *why = "row " + std::to_string(r) + ": K4 scale address differs from vk_locate";
                return false;
            }
        }
    }
    for (uint32_t b = s.blocks; b < 64; ++b) {
        if (table[b] != 0 || table[64 + b] != 0) {
            *why = "unused block-table entry not 0";
            return false;
        }
    }
    return true;
}

// ---- rows: arena order ----
void test_arena_order() {
    std::string key;
    uint64_t m = 0;
    bool ok = run_of(L + "7.mlp.experts.311.down_proj.weight_scale_2", &key, &m) &&
              key == L + "7.mlp.experts.*.down_proj.weight_scale_2" && m == 311;
    ok = ok && run_of(L + "11.self_attn.k_proj.weight", &key, &m) && key == L + "11.self_attn.qkv" && m == 1;
    ok = ok && run_of(L + "11.self_attn.v_proj.weight_scale", &key, &m) && key == L + "11.self_attn.qkv_scale" && m == 2;
    ok = ok && run_of(L + "1.ple.ple_embedding.ngram_embedding.shard_127.weight", &key, &m) &&
         key == L + "1.ple.shards" && m == 127;
    ok = ok && run_of(L + "1.ple.ple_embedding.ngram_embedding.shard_5.weight_scale", &key, &m) &&
         key == L + "1.ple.shard_scales" && m == 5;
    ok = ok && run_of("mtp.layers.0.self_attn.q_proj.weight", &key, &m) && key == "mtp.layers.0.self_attn.qkv" && m == 0;
    // Singletons: the CUDA rule's negatives.
    ok = ok && !run_of("mtp.layers.0.self_attn.o_proj.weight", &key, &m);
    ok = ok && !run_of("mtp.layers.0.mlp.experts.gate_up_proj", &key, &m);  // MTP experts: one 3-D tensor
    ok = ok && !run_of(L + "11.self_attn.o_proj.weight", &key, &m);
    ok = ok && !run_of(L + "1.ple.ple_embedding.ngram_embedding.weight_scale_2", &key, &m);
    ok = ok && !run_of(L + "x.mlp.experts.0.gate_proj.weight", &key, &m);
    ok = ok && !run_of(L + "1234567890.mlp.experts.0.gate_proj.weight", &key, &m);  // > 9 digits
    ok = ok && !run_of("lm_head.weight", &key, &m);
    row("arena_order.run_of names (model.cpp:57-132 rule)", ok);

    host::ModelPlan p = base_plan();
    p.tensors.push_back(T("a", CERE_DTYPE_BF16, DEV, {4}));
    p.tensors.push_back(T(L + "0.mlp.experts.1.up_proj.weight", CERE_DTYPE_U8, DEV, {2}));
    p.tensors.push_back(T("b", CERE_DTYPE_BF16, DISC, {4}));
    p.tensors.push_back(T(L + "0.mlp.experts.0.up_proj.weight", CERE_DTYPE_U8, DEV, {2}));
    p.tensors.push_back(T(L + "0.mlp.experts.0.up_proj.weight", CERE_DTYPE_U8, HOST, {2}));  // same key, other heap
    p.tensors.push_back(T("c", CERE_DTYPE_BF16, DEV, {4}));
    const std::vector<ArenaRun> r = arena_runs(p);
    const bool ok2 = r.size() == 4 && r[0].members == std::vector<uint64_t>{0} &&
                     r[1].members == std::vector<uint64_t>{3, 1} && r[1].key == L + "0.mlp.experts.*.up_proj.weight@device" &&
                     r[2].members == std::vector<uint64_t>{4} && r[2].key.find("@host") != std::string::npos &&
                     r[3].members == std::vector<uint64_t>{5};
    row("arena_order.runs: first-appearance order, rank sort, heap key, DISCARD", ok2);
}

// ---- rows: synthetic layouts ----
void test_synthetic() {
    std::string why;
    for (uint32_t drafts : {0u, 4u}) {
        const host::ModelPlan p = synthetic_plan(drafts);
        const VkLayout l = vk_plan_layout(p);
        const std::string name = std::string("synthetic.invariants drafts=") + std::to_string(drafts);
        row(name.c_str(), invariants(p, l, &why), why);
        why.clear();
        const std::string d = vk_layout_description(p, l);
        row((std::string("synthetic.description tag + determinism drafts=") + std::to_string(drafts)).c_str(),
            d.compare(0, std::strlen(kVkLayoutTag), kVkLayoutTag) == 0 && same_layout(p, VkLimits{}));
        // The KV entries share one block here: their offsets are layout::carve_pool's, from the first entry.
        const VkRegion *k0 = vk_region(l, VkRegionKind::kKCache, 3);
        bool pool_ok = k0 != nullptr;
        if (pool_ok) {
            // carve_pool calls bind after the take: the carver's cursor is then the region's END.
            layout::Carver c{nullptr};
            std::vector<uint64_t> ends;
            layout::carve_pool(p.config, p.config.full_attention_layers, p.kv_pool_tokens, c,
                               [&](layout::Region, uint32_t, uint8_t *) { ends.push_back(c.cursor); }, drafts != 0);
            pool_ok = c.size() == l.pool_bytes;
            size_t idx = 0;
            for (const VkRegion &r : l.regions) {
                if (r.kind < VkRegionKind::kKCache || r.kind > VkRegionKind::kMtpQsaKeys) continue;
                if (idx >= ends.size() || r.block != k0->block || r.offset - k0->offset + r.bytes != ends[idx]) {
                    pool_ok = false;
                }
                ++idx;
            }
            pool_ok = pool_ok && idx == ends.size();
        }
        row((std::string("synthetic.kv pool == layout::carve_pool drafts=") + std::to_string(drafts)).c_str(), pool_ok);
    }
    // Negative controls: the invariant checker sees an overlap, a block over the limit and a lying estimate.
    {
        const host::ModelPlan p = synthetic_plan(0);
        const VkLayout good = vk_plan_layout(p);
        std::string w;
        VkLayout bad = good;
        const uint64_t e0 = bad.piece_first[0], e1 = bad.piece_first[1];  // embed_tokens, then the next tensor
        bad.pieces[e1].block = bad.pieces[e0].block;
        bad.pieces[e1].offset = bad.pieces[e0].offset + 256;
        const bool overlap = !invariants(p, bad, &w);
        bad = good;
        bad.blocks[0].bytes = bad.limits.max_buffer_bytes + 4096;
        const bool over = !invariants(p, bad, &w);
        bad = good;
        bad.device_bytes += 4096;
        const bool est = !invariants(p, bad, &w);
        row("negative_control.invariants catch overlap / oversize / estimate", overlap && over && est);
    }
    // estimate == sum of blocks, explicitly.
    {
        const host::ModelPlan p = synthetic_plan(4);
        const VkLayout l = vk_plan_layout(p);
        uint64_t d = 0, h = 0;
        for (const VkBlock &b : l.blocks) (b.heap == VkHeap::kDeviceLocal ? d : h) += b.bytes;
        const host::MemoryEstimate e = vk_estimate(l);
        row("synthetic.estimate == sum of blocks", e.device_bytes == d && e.host_bytes == h && d > 0 && h > 0,
            "device " + commas(d) + " host " + commas(h));
    }
    // Forced small blocks (1 GiB): weights spill across blocks, the 28.8 GB S3-shaped PLE table splits on shard
    // boundaries into 26 chunks, and K4's addressing equals vk_locate.
    {
        VkLimits lim;
        lim.max_buffer_bytes = uint64_t(1) << 30;
        host::ModelPlan p = synthetic_plan(4, 128, 2500012);
        for (int i = 0; i < 6; ++i) {
            p.tensors.push_back(T("model.language_model.layers." + std::to_string(40 + i) + ".linear_attn.big" +
                                      std::to_string(i) + ".weight",
                                  CERE_DTYPE_BF16, DEV, {100000, 2048}));  // 409.6 MB each
        }
        const VkLayout l = vk_plan_layout(p, lim);
        std::string w;
        size_t weight_blocks = 0;
        for (const VkBlock &b : l.blocks) weight_blocks += b.purpose == VkPurpose::kWeights;
        row("small_blocks.invariants (1 GiB blocks)", invariants(p, l, &w), w);
        w.clear();
        row("small_blocks.weights spill over >= 3 blocks", weight_blocks >= 3, std::to_string(weight_blocks) + " blocks");
        row("small_blocks.PLE split on shard boundaries",
            l.ple.present && l.ple.shards_per_block == 5 && l.ple.rows_per_block == 5 * 2500012 && l.ple.blocks == 26,
            "rpb " + std::to_string(l.ple.rows_per_block) + " chunks " + std::to_string(l.ple.blocks) +
                " shards/chunk " + std::to_string(l.ple.shards_per_block));
        row("small_blocks.K4 SplitTable addressing == vk_locate", l.status == CERE_OK && split_geometry_ok(p, l, &w), w);
        w.clear();
        row("small_blocks.determinism", same_layout(p, lim));
    }
    // A shard larger than one block: the row fallback (shards straddle chunks, writes split by vk_locate).
    {
        VkLimits lim;
        lim.max_buffer_bytes = uint64_t(1) << 30;
        const host::ModelPlan p = synthetic_plan(0, 4, 20000000);  // 1.6 GB shards
        const VkLayout l = vk_plan_layout(p, lim);
        std::string w;
        row("row_fallback.invariants", invariants(p, l, &w), w);
        w.clear();
        const uint64_t want_rpb = (uint64_t(1) << 30) / 80;
        row("row_fallback.rows_per_block = block / 80 B", l.ple.shards_per_block == 0 && l.ple.rows_per_block == want_rpb,
            "rpb " + std::to_string(l.ple.rows_per_block) + " chunks " + std::to_string(l.ple.blocks));
        row("row_fallback.K4 SplitTable addressing == vk_locate", l.status == CERE_OK && split_geometry_ok(p, l, &w), w);
        // A write across a chunk border splits into two pieces that land at K4's addresses of the two rows.
        uint64_t straddle = 0;
        for (size_t t = 0; t < p.tensors.size(); ++t) {
            if (p.tensors[t].name.find("shard_0.weight") != std::string::npos && p.tensors[t].name.find("scale") == std::string::npos) {
                const std::vector<VkPiece> q = vk_locate(l, t, (want_rpb - 1) * 80, 160);
                straddle = q.size() == 2 && q[0].bytes == 80 && q[1].bytes == 80 && q[0].block != q[1].block ? 1 : 0;
            }
        }
        row("row_fallback.write across a chunk border = 2 pieces", straddle == 1);
    }
}

// ---- rows: refusals ----
void expect(const char *name, const host::ModelPlan &p, cere_status want, const VkLimits &lim = VkLimits{}) {
    const VkLayout l = vk_plan_layout(p, lim);
    row(name, l.status == want && !l.error.empty() && l.blocks.empty(),
        "status " + std::to_string(l.status) + ": " + l.error);
}

void test_refusals() {
    host::ModelPlan p;
    p = synthetic_plan(0);
    p.support = &host::a4_vision_support();
    expect("refuse.vision", p, CERE_ERR_UNSUPPORTED);
    p = synthetic_plan(0);
    p.max_sessions = 2;
    p.kv_pool_tokens = 65536;
    expect("refuse.max_sessions 2", p, CERE_ERR_UNSUPPORTED);
    p = synthetic_plan(0);
    p.max_context_tokens = 2048;
    p.kv_pool_tokens = 2048;
    expect("refuse.context 2,048 (OD-5)", p, CERE_ERR_UNSUPPORTED);
    p = synthetic_plan(0);
    p.max_context_tokens = 65536;
    p.kv_pool_tokens = 65536;
    expect("refuse.context 65,536 (> first light)", p, CERE_ERR_UNSUPPORTED);
    p = synthetic_plan(4);
    p.prefix_state_slots = 4;
    expect("refuse.prefix cache with drafts", p, CERE_ERR_UNSUPPORTED);
    p = synthetic_plan(0);
    p.prefix_state_slots = 1024;  // 1,025 x 59 MB fixed-state slots: one region larger than one 4 GiB block
    expect("refuse.state slots > one block", p, CERE_ERR_CAPACITY);
    {
        // D15: the fixed-state pool of a prefix model, one run of (prefix_state_slots + max_sessions) slots.
        p = synthetic_plan(0);
        p.prefix_state_slots = 16;
        const VkLayout l = vk_plan_layout(p);
        std::string w;
        const VkRegion *r = vk_region(l, VkRegionKind::kStateSlots);
        const layout::FixedState f = layout::fixed_state(p.config, 1);
        const bool ok = l.status == CERE_OK && invariants(p, l, &w) && r != nullptr && f.contiguous &&
                        l.state_bytes == f.bytes && l.state_slots == 17 && r->bytes == 17 * f.bytes &&
                        f.bytes % 256 == 0 && vk_region(vk_plan_layout(synthetic_plan(0)), VkRegionKind::kStateSlots) == nullptr;
        row("prefix.state slots region (16 + 1) x fixed_state", ok,
            "status " + std::to_string(l.status) + " " + l.error + w + " state_bytes " + std::to_string(l.state_bytes));
    }
    p = synthetic_plan(0);
    p.snapshot_host_bytes = 4096;
    expect("refuse.snapshot arena", p, CERE_ERR_UNSUPPORTED);
    p = synthetic_plan(0);
    p.prefill_park = true;
    expect("refuse.MixedPD park", p, CERE_ERR_UNSUPPORTED);
    p = synthetic_plan(4);
    p.speculative_draft_tokens = 3;
    expect("refuse.speculative_draft_tokens 3", p, CERE_ERR_UNSUPPORTED);
    p = synthetic_plan(4);
    p.draft_token_map.clear();
    expect("refuse.drafts without a token map", p, CERE_ERR_ARGUMENT);
    p = synthetic_plan(0);
    p.tensors.push_back(T("model.language_model.norm.weight", CERE_DTYPE_F16, DEV, {2560}));
    expect("refuse.dtype F16 (not a Vulkan kernel dtype)", p, CERE_ERR_UNSUPPORTED);
    p = synthetic_plan(0);
    p.tensors.push_back(T("model.language_model.x.weight", CERE_DTYPE_I32, HOST, {4}));
    expect("refuse.dtype I32 host tensor", p, CERE_ERR_UNSUPPORTED);
    p = synthetic_plan(0);
    p.tensors.push_back(T("model.language_model.huge.weight", CERE_DTYPE_U8, DEV, {uint64_t(5) << 30}));
    expect("refuse.device tensor > 4 GiB - 4", p, CERE_ERR_UNSUPPORTED);
    {
        VkLimits lim;
        lim.max_buffer_bytes = 6 << 20;  // 6 MiB: every tensor fits, the 8-expert gate_proj.weight run (6.55 MB) not
        p = synthetic_plan(0);
        const VkLayout l = vk_plan_layout(p, lim);
        row("refuse.device run > one block (experts run)",
            l.status == CERE_ERR_UNSUPPORTED && l.error.find("mlp.experts.*.gate_proj.weight@device") != std::string::npos,
            "status " + std::to_string(l.status) + ": " + l.error);
    }
    {
        VkLimits lim;
        lim.max_buffer_bytes = 64 << 20;  // weights fit; the workspace rows (~0.4 GB) do not
        p = synthetic_plan(0);
        expect("refuse.workspace region > one block", p, CERE_ERR_CAPACITY, lim);
    }
    p = synthetic_plan(0);
    p.kv_pool_tokens = uint64_t(1) << 60;
    expect("refuse.overflow (KV pool 2^60 tokens)", p, CERE_ERR_CAPACITY);
    p = synthetic_plan(0);
    p.kv_pool_tokens = uint64_t(1) << 24;  // 16 Mi tokens: one layer's K cache 8 GiB > one buffer
    expect("refuse.KV entry > one block", p, CERE_ERR_CAPACITY);
    {
        VkLimits lim;
        lim.max_buffer_bytes = uint64_t(1) << 30;
        p = synthetic_plan(0, 128, 20000000);  // 2.56e9 rows -> 191 chunks of 1 GiB
        expect("refuse.PLE needs > 64 chunks", p, CERE_ERR_UNSUPPORTED, lim);
    }
    {
        VkLimits lim;
        lim.max_buffer_bytes = 64;
        lim.block_granule = 64;
        lim.region_alignment = 64;
        p = synthetic_plan(0);
        expect("refuse.PLE row > one block (no row-boundary split)", p, CERE_ERR_UNSUPPORTED, lim);
    }
    p = synthetic_plan(0);
    for (auto &t : p.tensors) {
        if (t.name.find("shard_2.weight") != std::string::npos && t.name.find("scale") == std::string::npos) {
            t.name = L + "1.ple.ple_embedding.ngram_embedding.shard_9.weight";
        }
    }
    expect("refuse.PLE shards not numbered 0..N-1", p, CERE_ERR_UNSUPPORTED);
    p = synthetic_plan(0);
    for (auto &t : p.tensors) {
        if (t.name.find("shard_1.weight") != std::string::npos && t.name.find("scale") == std::string::npos) {
            t = T(t.name, CERE_DTYPE_U8, HOST, {999, 80});
        }
    }
    expect("refuse.PLE shards not uniform", p, CERE_ERR_UNSUPPORTED);
    p = synthetic_plan(0);
    p.tensors.erase(std::remove_if(p.tensors.begin(), p.tensors.end(),
                                   [](const host::TensorPlan &t) { return t.name.find("shard_3.weight_scale") != std::string::npos; }),
                    p.tensors.end());
    expect("refuse.PLE scale count != table count", p, CERE_ERR_UNSUPPORTED);
    p = synthetic_plan(0);
    for (auto &t : p.tensors) {
        if (t.name.find("ngram_embedding.shard_") != std::string::npos && t.name.find("scale") == std::string::npos) {
            t = T(t.name, CERE_DTYPE_U8, HOST, {1000, 96});
        }
    }
    expect("refuse.PLE shard not K4's row geometry", p, CERE_ERR_UNSUPPORTED);
    p = synthetic_plan(0);
    p.tensors.push_back(T(L + "0.linear_attn.in_proj_z.weight", CERE_DTYPE_F8_E4M3, DEV, {6144, 2560}));
    expect("refuse.MXFP8 weight without its scale (mxfp8::plan_of)", p, CERE_ERR_UNSUPPORTED);
    p = synthetic_plan(0);
    p.tensors[0].bytes += 1;
    expect("refuse.bytes disagree with shape", p, CERE_ERR_ARGUMENT);
    {
        VkLimits lim;
        lim.region_alignment = 384;
        expect("refuse.limits (alignment not a power of two)", synthetic_plan(0), CERE_ERR_ARGUMENT, lim);
    }
    // Positive control: a DISCARD tensor of an unsupported dtype is fine (synthetic_plan has one).
    {
        const VkLayout l = vk_plan_layout(synthetic_plan(0));
        row("accept.DISCARD F16 tensor takes no storage", l.status == CERE_OK, l.error);
    }
}

// ---- the real artifact ----
bool read_file(const std::string &path, std::string *out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream s;
    s << f.rdbuf();
    *out = s.str();
    return true;
}

uint32_t dtype_of(const std::string &s) {
    static const std::map<std::string, uint32_t> m = {
        {"BF16", CERE_DTYPE_BF16}, {"F16", CERE_DTYPE_F16},   {"F32", CERE_DTYPE_F32},
        {"F64", CERE_DTYPE_F64},   {"F8_E4M3", CERE_DTYPE_F8_E4M3}, {"F8_E5M2", CERE_DTYPE_F8_E5M2},
        {"U8", CERE_DTYPE_U8},     {"I8", CERE_DTYPE_I8},     {"I16", CERE_DTYPE_I16},
        {"I32", CERE_DTYPE_I32},   {"I64", CERE_DTYPE_I64},   {"BOOL", CERE_DTYPE_BOOL}};
    const auto it = m.find(s);
    return it == m.end() ? 0 : it->second;
}

// The S3 tensor table as the runtime builds the desc (crates/cere-runtime/src/native/upload.rs:647-700 model_spec):
// manifest `weights` files in manifest order, each file's tensors in data-offset order (crates/cere-compiler/src/
// safetensors.rs:263), placement by the family rules (configs/artifacts/qwen4_exp-nvfp4-plenvfp4-mxfp8.families.json:
// `.ple.ple_embedding.` = host_pinned, everything else device) mapped as upload.rs:188-199 (`draft` role = the `mtp.`
// families: DEVICE with drafts, else DISCARD). Headers only.
bool artifact_tensors(const std::string &dir, uint32_t drafts, std::vector<host::TensorPlan> *out, std::string *why) {
    std::string text;
    if (!read_file(dir + "/manifest.json", &text)) {
        *why = "cannot read " + dir + "/manifest.json";
        return false;
    }
    testjson::Value man;
    if (!testjson::parse(text, &man) || man.get("files") == nullptr) {
        *why = "manifest.json does not parse";
        return false;
    }
    for (const testjson::Value &f : man.get("files")->a) {
        const testjson::Value *role = f.get("role"), *path = f.get("path");
        if (role == nullptr || path == nullptr || role->s != "weights") continue;
        std::ifstream in(dir + "/" + path->s, std::ios::binary);
        uint64_t n = 0;
        if (!in.read(reinterpret_cast<char *>(&n), 8) || n == 0 || n > (uint64_t(1) << 30)) {
            *why = "cannot read the header of " + path->s;
            return false;
        }
        std::string header(n, '\0');
        if (!in.read(header.data(), std::streamsize(n))) {
            *why = "short header in " + path->s;
            return false;
        }
        testjson::Value h;
        if (!testjson::parse(header, &h)) {
            *why = "header of " + path->s + " does not parse";
            return false;
        }
        struct Entry {
            uint64_t begin, end;
            host::TensorPlan t;
        };
        std::vector<Entry> entries;
        for (const auto &[name, v] : h.o) {
            if (name == "__metadata__") continue;
            const testjson::Value *dt = v.get("dtype"), *shape = v.get("shape"), *off = v.get("data_offsets");
            if (dt == nullptr || shape == nullptr || off == nullptr || off->a.size() != 2) {
                *why = "tensor " + name + ": malformed header entry";
                return false;
            }
            host::TensorPlan t{};
            t.name = name;
            t.dtype = dtype_of(dt->s);
            t.ndim = uint32_t(shape->a.size());
            for (uint32_t d = 0; d < t.ndim; ++d) t.shape[d] = shape->a[d].u;
            t.bytes = off->a[1].u - off->a[0].u;
            const bool host_table = name.find(".ple.ple_embedding.") != std::string::npos;
            const bool draft = name.rfind("mtp.", 0) == 0;
            t.placement = draft ? (drafts ? DEV : DISC) : host_table ? HOST : DEV;
            entries.push_back(Entry{off->a[0].u, off->a[1].u, t});
        }
        std::stable_sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) {
            return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;
        });
        for (Entry &e : entries) out->push_back(std::move(e.t));
    }
    return true;
}

// The CUDA plan_layout placement rule (memory_plan.cpp:65-80) on the same arena runs: the CUDA device arena's weight
// bytes and mapped-block region bytes (alignment 256 / 16 below 4 KiB / packed members abut).
void cuda_rule(const host::ModelPlan &p, uint64_t *device, uint64_t *host) {
    *device = *host = 0;
    for (const ArenaRun &r : arena_runs(p)) {
        for (size_t m = 0; m < r.members.size(); ++m) {
            const host::TensorPlan &t = p.tensors[r.members[m]];
            const uint64_t a = m != 0 ? 1 : (t.bytes < 4096 ? 16 : 256);
            uint64_t &c = t.placement == DEV ? *device : *host;
            c = (c + a - 1) / a * a + t.bytes;
        }
    }
}

void real_row(const std::string &dir, const std::string &inventory) {
    struct stat st{};
    if (dir.empty() || stat((dir + "/manifest.json").c_str(), &st) != 0) {
        skip("real.S3 layout", "artifact not readable at '" + dir + "' (rerun with the artifact mounted)");
        return;
    }
    uint64_t inv_device = 0, inv_host = 0, inv_dev_count = 0, inv_host_count = 0;
    {
        std::string text;
        testjson::Value inv;
        if (!inventory.empty() && read_file(inventory, &text) && testjson::parse(text, &inv) &&
            inv.get("by_placement") != nullptr) {
            const testjson::Value *bp = inv.get("by_placement");
            if (bp->get("device")) {
                inv_device = bp->get("device")->get("bytes")->u;
                inv_dev_count = bp->get("device")->get("count")->u;
            }
            if (bp->get("host_pinned")) {
                inv_host = bp->get("host_pinned")->get("bytes")->u;
                inv_host_count = bp->get("host_pinned")->get("count")->u;
            }
        }
    }
    uint64_t draft_count = 0;
    if (stat((dir + "/draft/token_map.u32le").c_str(), &st) == 0) draft_count = uint64_t(st.st_size) / 4;
    for (uint32_t drafts : {4u, 0u}) {
        const auto t0 = std::chrono::steady_clock::now();
        host::ModelPlan p = base_plan();
        p.speculative_draft_tokens = drafts;
        if (drafts) {
            for (uint64_t i = 0; i < draft_count; ++i) p.draft_token_map.push_back(uint32_t(i));
        }
        uint64_t pool = 0;
        const cere_status ps = host::plan_kv_pool(0, p.max_sessions, p.max_context_tokens, p.support->max_kv_pool_tokens, &pool);
        p.kv_pool_tokens = pool;
        std::string why;
        const std::string tag = std::string("real.S3 drafts=") + std::to_string(drafts);
        if (ps != CERE_OK || !artifact_tensors(dir, drafts, &p.tensors, &why)) {
            row((tag + " tensor table").c_str(), false, why);
            continue;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const VkLayout l = vk_plan_layout(p);
        const auto t2 = std::chrono::steady_clock::now();
        row((tag + " invariants").c_str(), invariants(p, l, &why), why);
        if (l.status != CERE_OK) continue;
        uint64_t n_dev = 0, n_host = 0, n_disc = 0;
        for (const auto &t : p.tensors) (t.placement == DEV ? n_dev : t.placement == HOST ? n_host : n_disc)++;
        if (drafts && inv_device) {
            row((tag + " tensor sums == inventory by_placement").c_str(),
                l.device_tensor_bytes == inv_device && l.host_tensor_bytes == inv_host && n_dev == inv_dev_count &&
                    n_host == inv_host_count,
                "device " + commas(l.device_tensor_bytes) + " (" + commas(n_dev) + ") host " +
                    commas(l.host_tensor_bytes) + " (" + commas(n_host) + ")");
        }
        row((tag + " determinism (description twice)").c_str(), same_layout(p, VkLimits{}));
        if (drafts) {
            row((tag + " PLE = 7 chunks of 21 shards").c_str(),
                l.ple.blocks == 7 && l.ple.shards_per_block == 21 && l.ple.rows_per_block == 21 * l.ple.shard_rows);
            std::string w;
            row((tag + " K4 SplitTable addressing == vk_locate").c_str(), split_geometry_ok(p, l, &w), w);
            // K2 row-info groups in their kernels' layouts: per GDN layer 5 abutting K5 entries of
            // k5 weight_info_bytes; the PLE's value_proj entry at key + kValueInfoOff.
            const cere_qwen4_exp_config &cf = p.config;
            const uint64_t k5_bytes = sizes::k5_weight_info(cf.linear_num_key_heads, cf.linear_num_value_heads,
                                                            cf.linear_key_head_dim, cf.hidden_size);
            uint64_t k5_groups = 0, k4_groups = 0, fwd = 0;
            bool info_ok = true;
            for (size_t i = 0; i < l.winfo.size(); ++i) {
                const VkWinfo &wi = l.winfo[i];
                if (wi.consumer == VkInfoConsumer::kK5Layer && p.tensors[wi.tensor].name.find("in_proj_qkv") != std::string::npos) {
                    ++k5_groups;
                    info_ok = info_ok && i + 4 < l.winfo.size() &&
                              l.winfo[i + 4].offset + sizes::k2_winfo(l.winfo[i + 4].n) - wi.offset == k5_bytes &&
                              l.winfo[i + 4].n == cf.hidden_size;
                }
                if (wi.consumer == VkInfoConsumer::kK4Ple && p.tensors[wi.tensor].name.find("key_proj") != std::string::npos) {
                    ++k4_groups;
                    info_ok = info_ok && i + 1 < l.winfo.size() &&
                              l.winfo[i + 1].offset - wi.offset == sizes::a16(uint64_t(cf.hc_count) * cf.hidden_size * 4);
                }
                fwd += wi.consumer == VkInfoConsumer::kForward;
            }
            row((tag + " K2 row-info groups match K5 / K4 info layouts").c_str(), info_ok && k5_groups == 36 && k4_groups == 1,
                std::to_string(k5_groups) + " K5 layers, " + std::to_string(k4_groups) + " K4, " + std::to_string(fwd) +
                    " forward entries");
        }
        // ---- the numbers ----
        std::map<std::pair<int, int>, std::pair<uint64_t, uint64_t>> kinds;  // (heap, purpose) -> (count, bytes)
        uint64_t largest = 0;
        size_t largest_i = 0;
        for (size_t b = 0; b < l.blocks.size(); ++b) {
            auto &k = kinds[{int(l.blocks[b].heap), int(l.blocks[b].purpose)}];
            k.first++;
            k.second += l.blocks[b].bytes;
            if (l.blocks[b].bytes > largest) {
                largest = l.blocks[b].bytes;
                largest_i = b;
            }
        }
        std::printf("  --- S3, drafts=%u: %zu tensors (%s device, %s host, %s discard), tensor table %.1f s, layout %.2f s\n",
                    drafts, p.tensors.size(), commas(n_dev).c_str(), commas(n_host).c_str(), commas(n_disc).c_str(),
                    std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count());
        std::printf("  blocks: %zu\n", l.blocks.size());
        for (const auto &[k, v] : kinds) {
            std::printf("    %-12s %-10s %3" PRIu64 " blocks %18s B (%s)\n", k.first ? "host-heap" : "device-local",
                        vk_purpose_name(VkPurpose(k.second)), v.first, commas(v.second).c_str(), gib(v.second).c_str());
        }
        std::printf("  block table:\n");
        for (size_t b = 0; b < l.blocks.size(); ++b) {
            std::string what;
            std::map<std::string, int> names;
            for (const VkRegion &r : l.regions) {
                if (r.block == b) names[vk_region_name(r.kind)]++;
            }
            for (const auto &[n, c] : names) what += n + (c > 1 ? "x" + std::to_string(c) : "") + " ";
            std::printf("    #%-3zu %-12s %-9s %14s B used %14s%s%s\n", b,
                        l.blocks[b].heap == VkHeap::kDeviceLocal ? "device-local" : "host-heap",
                        vk_purpose_name(l.blocks[b].purpose), commas(l.blocks[b].bytes).c_str(),
                        commas(l.blocks[b].used).c_str(), what.empty() ? "" : "  + ", what.c_str());
        }
        std::printf("  device_bytes %s (%s), host_bytes %s (%s); largest block #%zu %s\n", commas(l.device_bytes).c_str(),
                    gib(l.device_bytes).c_str(), commas(l.host_bytes).c_str(), gib(l.host_bytes).c_str(), largest_i,
                    commas(largest).c_str());
        uint64_t cuda_dev = 0, cuda_host = 0;
        cuda_rule(p, &cuda_dev, &cuda_host);
        const cere_qwen4_exp_config &c = p.config;
        const bool d = drafts != 0;
        const uint64_t cuda_derived = layout::derived_bytes(c, p.max_context_tokens) +
                                      (d ? layout::mtp_derived_bytes(c, l.draft_count) : 0);
        const uint64_t mx = model::mxfp8::plan_of(p).staging_bytes;
        std::printf("  vs CUDA: tensor sums device %s host %s (inventory %s / %s)\n", commas(l.device_tensor_bytes).c_str(),
                    commas(l.host_tensor_bytes).c_str(), commas(inv_device).c_str(), commas(inv_host).c_str());
        std::printf("    CUDA rule (restated, memory_plan.cpp:65-80): weight_bytes %s (+%s padding), mapped region %s (+%s)\n",
                    commas(cuda_dev).c_str(), commas(cuda_dev - l.device_tensor_bytes).c_str(), commas(cuda_host).c_str(),
                    commas(cuda_host - l.host_tensor_bytes).c_str());
        uint64_t weights_used = 0;
        for (const VkBlock &b : l.blocks) weights_used += b.purpose == VkPurpose::kWeights ? b.used : 0;
        const VkRegion *der = vk_region(l, VkRegionKind::kDerived), *mder = vk_region(l, VkRegionKind::kMtpDerived),
                       *moe = vk_region(l, VkRegionKind::kMoeInfo), *wi = vk_region(l, VkRegionKind::kWinfo);
        std::printf("    Vulkan derived: layout %s (CUDA same: %s) + K7/K13 bf16_info %s (stride %s) + K2 winfo %s (%zu "
                    "weights, max n %s)\n",
                    commas(der->bytes + (mder ? mder->bytes : 0)).c_str(), commas(cuda_derived).c_str(),
                    commas(moe->bytes).c_str(), commas(l.moe_info_stride).c_str(), commas(wi->bytes).c_str(),
                    l.winfo.size(), commas(l.gemm_max_n).c_str());
        std::printf("    CUDA-only region absent on Vulkan: MXFP8 dequant staging %s (K2 reads MXFP8 directly)\n",
                    commas(mx).c_str());
        uint64_t sum_regions[6] = {};
        for (const VkBlock &b : l.blocks) sum_regions[int(b.purpose)] += b.bytes;
        std::printf("    Vulkan blocks by purpose (bytes incl. granule rounding): weights %s, workspace %s, state %s, "
                    "kv %s, io %s, ple %s; weights used %s\n",
                    commas(sum_regions[0]).c_str(), commas(sum_regions[1]).c_str(), commas(sum_regions[2]).c_str(),
                    commas(sum_regions[3]).c_str(), commas(sum_regions[4]).c_str(), commas(sum_regions[5]).c_str(),
                    commas(weights_used).c_str());
        std::printf("  PLE split: format %s, %u shards x %s rows = %s rows, row %u B + scale %u B; rows_per_block %s "
                    "(%u shards), %u chunks\n",
                    l.ple.format ? "NVFP4" : "FP8", l.ple.shards, commas(l.ple.shard_rows).c_str(), commas(l.ple.rows).c_str(),
                    l.ple.row_bytes, l.ple.scale_bytes, commas(l.ple.rows_per_block).c_str(), l.ple.shards_per_block,
                    l.ple.blocks);
        for (uint32_t b = 0; b < l.ple.blocks; ++b) {
            std::printf("    chunk %u: table #%u+%s (%s B), scales #%u+%s\n", b, l.ple.table_block[b],
                        commas(l.ple.table_offset[b]).c_str(),
                        commas(std::min<uint64_t>(l.ple.rows_per_block, l.ple.rows - uint64_t(b) * l.ple.rows_per_block) *
                               l.ple.row_bytes).c_str(),
                        l.ple.scale_block.empty() ? 0 : l.ple.scale_block[b],
                        commas(l.ple.scale_offset.empty() ? 0 : l.ple.scale_offset[b]).c_str());
        }
        std::printf("  workspace at %s rows, %s context (scratch = max of these):\n", commas(l.workspace_rows).c_str(),
                    commas(p.max_context_tokens).c_str());
        for (const VkWorkspaceEntry &e : l.workspace) {
            std::printf("    %-58s %14s B %s  [%s]\n", e.name.c_str(), commas(e.bytes).c_str(),
                        e.may_be_stale ? "MAY BE STALE" : "stable      ", e.source.c_str());
        }
        uint64_t ws_total = 0;
        for (const VkRegion &r : l.regions) {
            if (r.kind >= VkRegionKind::kWorkspaceRows && r.kind <= VkRegionKind::kMtpHcRows) {
                std::printf("    region %-16s %14s B\n", vk_region_name(r.kind), commas(r.bytes).c_str());
                ws_total += r.bytes;
            }
        }
        std::printf("    workspace total (rows + scratch + gemm ws%s) %s B (%s)\n", d ? " + mtp hc rows" : "",
                    commas(ws_total).c_str(), gib(ws_total).c_str());
        std::printf("  KV pool: %s tokens, %s B (%s); layout::pool_bytes %s\n", commas(l.pool_tokens).c_str(),
                    commas(l.pool_bytes).c_str(), gib(l.pool_bytes).c_str(),
                    commas(layout::pool_bytes(c, c.full_attention_layers, l.pool_tokens, d)).c_str());
        std::printf("  session slot %s B x %" PRIu64 "; undo %s B x %" PRIu64 "; verify rows %" PRIu64 "\n",
                    commas(l.session_bytes).c_str(), p.max_sessions, commas(l.undo_bytes).c_str(),
                    d ? 3 * p.max_sessions : 0, l.verify_rows);
        const VkRegion *ss = vk_region(l, VkRegionKind::kSpecSampler);
        std::printf("  spec sampler %s B; IO block regions:", ss ? commas(ss->bytes).c_str() : "0");
        for (const VkRegion &r : l.regions) {
            if (r.kind >= VkRegionKind::kIoIds32) std::printf(" %s=%s", vk_region_name(r.kind), commas(r.bytes).c_str());
        }
        std::printf("\n  layout digest %s (description %s B)\n", vk_layout_digest(p, l).c_str(),
                    commas(vk_layout_description(p, l).size()).c_str());
    }
}

}  // namespace

int main(int argc, char **argv) {
    const std::string dir = argc > 1 ? argv[1] : "";
    const std::string inventory = argc > 2 ? argv[2] : "";
    test_arena_order();
    test_synthetic();
    test_refusals();
    real_row(dir, inventory);
    std::printf("vk_memory_plan_test: %d PASS, %d FAIL, %d SKIP\n", g_pass, g_fail, g_skip);
    return g_fail == 0 ? 0 : 1;
}
