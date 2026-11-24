// VulkanAmd P3 T2: device-internal test of VulkanDevice (linked statically from the library's own objects). Rows print
// PASS / FAIL / SKIP; exit 1 on any FAIL.
//
// Usage: device_test [ARTIFACT_DIR]
//   host-only rows: vk_bind_weights on the real S3 tensor table (headers only; plain and drafts plans) with made-up
//     block addresses, and binding refusals; levers; device-id grammar is the smoke's.
//   GPU rows (Vulkan device 0, ~1.6 GB): allocate_storage of a small synthetic plan (no full model, so no binding),
//     write + read back through the mappings for every tensor (a PLE shard, expert runs, small tensors), refused writes,
//     ledger == plan, session create (slot map, EOS history, page accounting, second session CAPACITY), forward
//     refused, model release returns the ledger to 0, a simulated sticky fault refuses later work.
#include "device/vk_memory_plan.h"
#include "device/vulkan_bind.h"
#include "device/vulkan_device.h"
#include "host/kv_pool.h"
#include "kernels/k6_attention_vk.h"
#include "host/support.h"
#include "vk_artifact.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace cere;
using namespace cere::vkdev;

namespace {

int g_fail = 0, g_pass = 0, g_skip = 0;
void row(const std::string &name, bool ok, const std::string &detail = "") {
    std::printf("%-66s %s%s%s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.empty() ? "" : "  ", detail.c_str());
    (ok ? g_pass : g_fail)++;
}
void skip(const std::string &name, const std::string &why) {
    std::printf("%-66s SKIP  %s\n", name.c_str(), why.c_str());
    ++g_skip;
}

constexpr uint32_t DEV = CERE_PLACEMENT_DEVICE, HOST = CERE_PLACEMENT_HOST_PINNED, DISC = CERE_PLACEMENT_DISCARD;

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

host::ModelPlan base_plan() {
    host::ModelPlan p{};
    p.support = &host::a4_support();
    p.config = p.support->config;
    p.max_sessions = 1;
    p.max_context_tokens = 32768;
    p.kv_pool_tokens = 32768;
    return p;
}

// vk_memory_plan_test.cpp synthetic_plan: expert runs, MXFP8 q|k|v, small tensors, an NVFP4 PLE table, a DISCARD tensor.
host::ModelPlan synthetic_plan() {
    host::ModelPlan p = base_plan();
    const std::string L = "model.language_model.layers.";
    auto &v = p.tensors;
    v.push_back(T("model.language_model.embed_tokens.weight", CERE_DTYPE_BF16, DEV, {1000, 2560}));
    for (uint32_t e = 0; e < 8; ++e) {
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
    const std::string ple = L + "1.ple.ple_embedding.";
    for (uint32_t s = 0; s < 4; ++s) {
        v.push_back(T(ple + "ngram_embedding.shard_" + std::to_string(s) + ".weight", CERE_DTYPE_U8, HOST, {1000, 80}));
        v.push_back(T(ple + "ngram_embedding.shard_" + std::to_string(s) + ".weight_scale", CERE_DTYPE_F8_E4M3, HOST,
                      {1000, 10}));
    }
    v.push_back(T(ple + "ngram_embedding.weight_scale_2", CERE_DTYPE_F32, HOST, {1}));
    v.push_back(T(ple + "layer_multipliers", CERE_DTYPE_I64, HOST, {3}));
    v.push_back(T("mtp.fc_hidden.weight", CERE_DTYPE_BF16, DISC, {2560, 2560}));
    return p;
}

void bind_rows(const std::string &dir) {
    // Negative: the synthetic plan is not a full model.
    {
        const host::ModelPlan p = synthetic_plan();
        const VkLayout l = vk_plan_layout(p);
        std::vector<uint64_t> addr(l.blocks.size());
        for (size_t b = 0; b < addr.size(); ++b) addr[b] = (uint64_t(b) + 1) << 36;
        VkModelWeights w;
        std::string why;
        const cere_status st = vk_bind_weights(p, l, addr, &w, &why);
        row("bind: incomplete model UNSUPPORTED, names the tensor",
            st == CERE_ERR_UNSUPPORTED && why.find("tensor model.language_model.embed_tokens.weight: shape") == 0,
            why);
        std::vector<uint64_t> short_addr(addr.begin(), addr.end() - 1);
        row("bind: block address count != layout ARGUMENT",
            vk_bind_weights(p, l, short_addr, &w, &why) == CERE_ERR_ARGUMENT);
    }
    struct stat st {};
    if (dir.empty() || stat((dir + "/manifest.json").c_str(), &st) != 0) {
        skip("bind: real S3 tensor table", "artifact not readable at '" + dir + "'");
        return;
    }
    for (uint32_t drafts : {0u, 4u}) {
        const std::string tag = "bind: S3 drafts=" + std::to_string(drafts);
        std::vector<vktest::ArtTensor> art;
        std::string why;
        if (!vktest::artifact_tensors(dir, drafts, &art, &why)) {
            row(tag + " tensor table", false, why);
            continue;
        }
        host::ModelPlan p = base_plan();
        for (const auto &a : art) p.tensors.push_back(a.t);
        p.speculative_draft_tokens = drafts;
        if (drafts) {
            for (uint64_t i = 0, n = vktest::draft_map_count(dir); i < n; ++i) p.draft_token_map.push_back(uint32_t(i));
        }
        const VkLayout l = vk_plan_layout(p);
        if (l.status != CERE_OK) {
            row(tag + " layout", false, l.error);
            continue;
        }
        std::vector<uint64_t> addr(l.blocks.size());
        for (size_t b = 0; b < addr.size(); ++b) addr[b] = (uint64_t(b) + 1) << 36;
        auto w = std::make_unique<VkModelWeights>();
        const cere_status bs = vk_bind_weights(p, l, addr, w.get(), &why);
        row(tag + " vk_bind_weights OK", bs == CERE_OK, why);
        if (bs != CERE_OK) continue;
        // Spot checks against vk_locate / the layout.
        std::map<std::string, uint64_t> idx;
        for (uint64_t i = 0; i < p.tensors.size(); ++i) idx[p.tensors[i].name] = i;
        const auto at = [&](const std::string &n) { return vk_tensor_addr(l, addr, idx.at(n)); };
        const std::string L0 = "model.language_model.layers.0.", L3 = "model.language_model.layers.3.";
        bool ok = w->embed == at("model.language_model.embed_tokens.weight") && w->lm_head == at("lm_head.weight") &&
                  w->layers[0].gdn.in_proj_qkv == at(L0 + "linear_attn.in_proj_qkv.weight") &&
                  w->layers[0].gdn_dense[0].scales == at(L0 + "linear_attn.in_proj_qkv.weight_scale") &&
                  w->layers[3].attn.qkv.w == at(L3 + "self_attn.q_proj.weight") && w->layers[3].attn.qkv.n == 13312 &&
                  w->layers[3].attn.qkv.scales == at(L3 + "self_attn.q_proj.weight_scale") &&
                  w->layers[0].moe.gate_weight == at(L0 + "mlp.experts.0.gate_proj.weight") &&
                  w->layers[0].moe.down_scale == at(L0 + "mlp.experts.0.down_proj.weight_scale") &&
                  w->layers[0].shared_dense[2].scales == at(L0 + "mlp.shared_expert.down_proj.weight_scale") &&
                  w->ple.key_proj == at("model.language_model.layers.1.ple.key_proj.weight") &&
                  w->ple.table.blocks == l.ple.blocks && w->ple.table.rows_per_block == l.ple.rows_per_block &&
                  w->ple.table.desc == vk_region_addr(l, addr, VkRegionKind::kIoPleBlockTable) && w->ple_layer == 1;
        // Every derived / info pointer inside its region.
        uint32_t gdn = 0, info_ok = 0, attn = 0;
        for (uint32_t li = 0; li < 48; ++li) {
            const VkLayerWeights &lw = w->layers[li];
            if (lw.full_attention) {
                ++attn;
                info_ok += lw.attn.qkv.info != 0 && lw.attn.o_proj.info != 0 && lw.attn.indexer_qk.info != 0;
            } else {
                ++gdn;
                info_ok += lw.gdn.info != 0 && lw.gdn.A_log != 0 && lw.gdn_dense[4].info != 0;
            }
            ok = ok && lw.moe.g1_alphas != 0 && lw.moe.g2_alphas != 0 && lw.moe.bf16_info != 0;
        }
        ok = ok && gdn == 36 && attn == 12 && info_ok == 48 && w->ple.proj_info != 0 && w->rope_cos_sin != 0;
        if (drafts) {
            ok = ok && w->has_draft && w->mtp.moe.gate_up_q != 0 && w->mtp.moe.down_global != 0 &&
                 w->mtp.lm_head != 0 && w->mtp.map != 0 && w->mtp.count == p.draft_token_map.size() &&
                 w->mtp.fc_hidden.info != 0 && w->mtp.attn.qkv.info != 0 && !w->mtp.attn.qkv.mxfp8();
        } else {
            ok = ok && !w->has_draft;
        }
        row(tag + " addresses == vk_locate, derived / info bound", ok,
            std::to_string(w->mxfp8_tensors) + " MXFP8 dense projections");
        // Workspace + session carving.
        VkWorkspace ws;
        VkSessionState ss;
        const cere_status wst = vk_bind_workspace(p, l, addr, &ws);
        const cere_status sst =
            vk_bind_session(p, l, addr, vk_region_addr(l, addr, VkRegionKind::kSession), 32768, &ss);
        row(tag + " workspace + session carve", wst == CERE_OK && sst == CERE_OK && ws.scratch != 0 && ws.top2 != 0 &&
                                                    ss.slot_map != 0 && ss.gdn[0].ssm != 0 && ss.k_cache[3] != 0 &&
                                                    ss.out_counts != 0 && (drafts == 0 || ss.proposals != 0),
            "slot " + std::to_string(ss.slot_bytes) + " B of " + std::to_string(l.session_bytes));
        row(tag + " session > the plan's context ARGUMENT",
            vk_bind_session(p, l, addr, vk_region_addr(l, addr, VkRegionKind::kSession), 32769, &ss) == CERE_ERR_ARGUMENT);
    }
}

void lever_rows() {
    host::Levers l;
    row("levers: defaults accepted", VulkanDevice::check_levers(l, nullptr) == CERE_OK);
    l.moe_gemm = CERE_LEVER_MOE_K7;
    row("levers: moe_gemm k7 accepted", VulkanDevice::check_levers(l, nullptr) == CERE_OK);
    const auto refused = [](const char *name, auto set) {
        host::Levers x;
        set(x);
        std::string why;
        row(std::string("levers: ") + name + " UNSUPPORTED", VulkanDevice::check_levers(x, &why) == CERE_ERR_UNSUPPORTED, why);
    };
    refused("k2_gemm small_m", [](host::Levers &x) { x.k2_gemm = CERE_LEVER_K2_SMALL_M; });
    {  // P4-B / D7: tail_split = the GEMM's SPLIT x rows (Context::k2_tail_split)
        host::Levers x;
        x.k2_gemm = CERE_LEVER_K2_TAIL_SPLIT;
        row("levers: k2_gemm tail_split accepted", VulkanDevice::check_levers(x, nullptr) == CERE_OK);
    }
    refused("k2_gemm 4 (no such impl)", [](host::Levers &x) { x.k2_gemm = CERE_LEVER_K2_TAIL_SPLIT + 1; });
    {  // P4-E: p4 = the coopmat prefill attention; the static check accepts it (the device half needs coopmat + probe)
        host::Levers x;
        x.prefill_attention = CERE_LEVER_PREFILL_P4;
        row("levers: prefill p4 accepted by the static check (device half: gpu rows)", VulkanDevice::check_levers(x, nullptr) == CERE_OK);
    }
    refused("prefill_attention 2 (no such impl)", [](host::Levers &x) { x.prefill_attention = 2; });
    refused("decode graph", [](host::Levers &x) { x.decode_step = CERE_LEVER_DECODE_GRAPH; });
    refused("decode spec_graph", [](host::Levers &x) { x.decode_step = CERE_LEVER_DECODE_SPEC_GRAPH; });
    refused("moe p3b", [](host::Levers &x) { x.moe_gemm = CERE_LEVER_MOE_P3B; });
    refused("verify batched", [](host::Levers &x) { x.verify_attention = CERE_LEVER_VERIFY_BATCHED; });
    refused("k2_weights fp8", [](host::Levers &x) { x.k2_weights = CERE_LEVER_K2_WEIGHTS_FP8; });
}

void gpu_rows() {
    std::unique_ptr<host::Device> dev;
    const cere_status os = VulkanDevice::open("0", &dev);
    row("gpu: VulkanDevice::open(\"0\")", os == CERE_OK, "status " + std::to_string(os));
    if (os != CERE_OK) return;
    auto &vd = static_cast<VulkanDevice &>(*dev);
    std::printf("  device: %s (%s, %s), coopmat %d\n", vd.ctx().info().name.c_str(), vd.ctx().info().driver_name.c_str(),
                vd.ctx().info().driver_info.c_str(), int(vd.ctx().info().coopmat));
    row("gpu: supported() is the A4 text entry only",
        dev->supported().size() == 1 && dev->supported()[0] == &host::a4_support());
    row("gpu: forward / sampler not supported before a seal (T5: true after instantiate)",
        !dev->forward_supported() && !dev->sampler_supported());
    row("gpu: moe_lever == K7", dev->moe_lever() == CERE_LEVER_MOE_K7);

    const host::ModelPlan p = synthetic_plan();
    const host::MemoryEstimate est = dev->estimate(p);
    std::unique_ptr<VulkanModel> m;
    std::string why;
    const cere_status as = vd.allocate_storage(p, &m, &why);
    row("gpu: allocate_storage(synthetic plan)", as == CERE_OK, why);
    if (as != CERE_OK) return;
    row("gpu: ledger == estimate == layout block sums",
        m->held_device_bytes() == est.device_bytes && m->held_host_bytes() == est.host_bytes &&
            vd.ledger().held(host::Held::kDevice) == est.device_bytes &&
            vd.ledger().held(host::Held::kMapped) == est.host_bytes,
        std::to_string(est.device_bytes) + " device + " + std::to_string(est.host_bytes) + " host, " +
            std::to_string(m->blocks().size()) + " blocks");
    bool coherent = true;
    for (const vk::Buffer &b : m->blocks()) {
        coherent = coherent && (vd.ctx().memory_flags(b.memory_type) & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    }
    row("gpu: every block host-coherent and mapped", coherent);
    {   // P4-E: prefill_attention p4 (the coopmat prefill attention): the device half of the lever check, the probe
        const bool cm = vd.ctx().info().coopmat;
        host::Levers l4;
        l4.prefill_attention = CERE_LEVER_PREFILL_P4;
        std::string w4;
        const cere_status s4 = vd.check_prefill_attention(m.get(), l4, &w4);
        row("gpu: prefill_attention p4 on an unsealed model: OK with VK_KHR_cooperative_matrix, else UNSUPPORTED",
            s4 == (cm ? CERE_OK : CERE_ERR_UNSUPPORTED), w4);
        row("gpu: set_levers(p4) then back to cuda_core", dev->set_levers(*m, l4) == (cm ? CERE_OK : CERE_ERR_UNSUPPORTED) &&
            dev->set_levers(*m, host::Levers{}) == CERE_OK && m->levers().prefill_attention == CERE_LEVER_PREFILL_CUDA_CORE);
        row("gpu: unsealed model: prefill attention path fp32 (no probe yet)",
            std::string(m->prefill_attention_path()) == "fp32" && !m->instantiate_report().coopmat_probed);
        uint32_t pr[4] = {};
        const cere_status ps = cm ? vkk6::coopmat_probe(vd.ctx(), pr) : CERE_ERR_UNSUPPORTED;
        char d[160];
        std::snprintf(d, sizeof d, "status %d words %08x %08x %08x %u", ps, pr[0], pr[1], pr[2], pr[3]);
        row("gpu: vkk6::coopmat_probe: FP16 subnormals kept (the seal's gate)",
            !cm || (ps == CERE_OK && pr[0] == kCoopmatProbeF16 && pr[1] == kCoopmatProbeProduct), d);
    }
    {
        // The full allocate (with binding) refuses the incomplete model and keeps nothing.
        std::unique_ptr<host::DeviceModel> full;
        const uint64_t before = vd.ledger().held(host::Held::kDevice);
        const cere_status st = dev->allocate(p, &full);
        row("gpu: allocate(incomplete model) UNSUPPORTED, nothing kept",
            st == CERE_ERR_UNSUPPORTED && !full && vd.ledger().held(host::Held::kDevice) == before);
    }
    // Upload every non-DISCARD tensor in 3 uneven writes, read back.
    std::mt19937_64 rng(7);
    uint64_t written = 0, bad = 0;
    for (uint64_t i = 0; i < p.tensors.size(); ++i) {
        const host::TensorPlan &t = p.tensors[i];
        if (t.placement == DISC) continue;
        std::vector<uint8_t> src(t.bytes), back(t.bytes, 0xAA);
        for (auto &b : src) b = uint8_t(rng());
        const uint64_t a = t.bytes / 3, b2 = t.bytes - t.bytes / 5;
        const uint64_t cuts[4] = {0, a, b2, t.bytes};
        for (int c = 0; c < 3; ++c) {
            if (cuts[c + 1] > cuts[c] && dev->write(*m, i, cuts[c], src.data() + cuts[c], cuts[c + 1] - cuts[c]) != CERE_OK) ++bad;
        }
        if (m->read(i, 0, back.data(), t.bytes) != CERE_OK || back != src) ++bad;
        written += t.bytes;
    }
    row("gpu: write + read back every tensor (3 pieces each)", bad == 0,
        std::to_string(written) + " B, " + std::to_string(bad) + " mismatches");
    {
        const uint64_t last = p.tensors.size() - 1;  // DISCARD
        uint8_t x[16] = {};
        row("gpu: write of a DISCARD tensor ARGUMENT", dev->write(*m, last, 0, x, 16) == CERE_ERR_ARGUMENT);
        row("gpu: write past a tensor ARGUMENT", dev->write(*m, 0, p.tensors[0].bytes - 8, x, 16) == CERE_ERR_ARGUMENT);
        row("gpu: write of an unknown tensor ARGUMENT", dev->write(*m, 1u << 30, 0, x, 16) == CERE_ERR_ARGUMENT);
        row("gpu: write of null bytes ARGUMENT", dev->write(*m, 0, 0, nullptr, 16) == CERE_ERR_ARGUMENT);
    }
    {
        // The K4 block table: table_base[b] / scale_base[b] for the layout's chunks.
        uint64_t want[128];
        vk_ple_block_table(m->layout(), m->block_addrs().data(), want);
        const uint8_t *t = m->region_map(VkRegionKind::kIoPleBlockTable);
        row("gpu: K4 block table written at allocate", t != nullptr && std::memcmp(t, want, sizeof want) == 0);
    }
    // Sessions.
    {
        std::unique_ptr<host::DeviceSession> s, s2;
        const uint64_t free0 = dev->free_pages(*m);
        const cere_status cs = dev->create_session(*m, 32768, false, &s);
        row("gpu: create_session(32,768)", cs == CERE_OK, "status " + std::to_string(cs));
        if (cs == CERE_OK) {
            auto &vs = static_cast<VulkanSession &>(*s);
            const VkSessionState &st = vs.state();
            int64_t map0 = -1, map100 = -1, hist[2] = {0, 0};
            std::memcpy(&map0, m->host_of(st.slot_map), 8);
            std::memcpy(&map100, m->host_of(st.slot_map) + 8 * 100, 8);
            std::memcpy(hist, m->host_of(st.ple.history), 16);
            const uint8_t *ssm = m->host_of(st.gdn[0].ssm);
            bool zero = true;
            for (int k = 0; k < 4096; ++k) zero = zero && ssm[k] == 0;
            row("gpu: session slot map, EOS history, zeroed state",
                map0 == int64_t(vs.pages()[0]) * 64 && map100 == int64_t(vs.pages()[1]) * 64 + 36 &&
                    hist[0] == vkk4::kEosTokenId && hist[1] == vkk4::kEosTokenId && zero);
            row("gpu: session took kv_pages(32,768) pages",
                dev->free_pages(*m) == free0 - host::kv_pages(32768) && dev->session_page_list(*s).size() == 512 &&
                    dev->session_position(*s) == 0 && dev->page_refs(*m, vs.pages()[0]) == 1);
            row("gpu: second session CAPACITY (max_sessions 1)", dev->create_session(*m, 4096, false, &s2) == CERE_ERR_CAPACITY);
            row("gpu: lazy session ARGUMENT", dev->create_session(*m, 4096, true, &s2) == CERE_ERR_ARGUMENT);
            const uint32_t tok = 1;
            uint64_t consumed = 99;
            cere_token_score out{};
            // T5: the forward exists but runs only a sealed model (cuda_device.cpp:1176 rule: ARGUMENT), nothing
            // consumed; an image / EndStep step is UNSUPPORTED before that check is reached on a sealed model.
            row("gpu: step on an unsealed model ARGUMENT, nothing consumed",
                dev->step(*m, *s, &tok, 1, 0, nullptr, &consumed, nullptr, &out, nullptr, nullptr, nullptr) ==
                        CERE_ERR_ARGUMENT &&
                    consumed == 0);
            host::RowArgs ra{s.get(), 1};
            cere_status rs = CERE_PENDING;
            row("gpu: step_rows on an unsealed model ARGUMENT, row_status untouched",
                dev->step_rows(*m, &ra, 1, &out, &rs) == CERE_ERR_ARGUMENT && rs == CERE_PENDING);
            row("gpu: rope delta 0 OK / +1 ARGUMENT / -1 UNSUPPORTED",
                dev->set_rope_delta(*m, *s, 0) == CERE_OK && dev->set_rope_delta(*m, *s, 1) == CERE_ERR_ARGUMENT &&
                    dev->set_rope_delta(*m, *s, -1) == CERE_ERR_UNSUPPORTED);
            s.reset();
            row("gpu: session release returns its pages and slot",
                dev->free_pages(*m) == free0 && dev->create_session(*m, 4096, false, &s2) == CERE_OK);
            s2.reset();
        }
    }
    {
        host::Levers l;
        l.decode_step = CERE_LEVER_DECODE_GRAPH;
        const cere_status a = dev->set_levers(*m, l);
        row("gpu: set_levers(graph) UNSUPPORTED, previous kept", a == CERE_ERR_UNSUPPORTED &&
                                                                    m->levers().decode_step == CERE_LEVER_DECODE_EAGER);
        l.decode_step = CERE_LEVER_DECODE_EAGER;
        l.moe_gemm = CERE_LEVER_MOE_K7;
        row("gpu: set_levers(eager, k7) OK", dev->set_levers(*m, l) == CERE_OK && m->levers().moe_gemm == CERE_LEVER_MOE_K7);
    }
    m.reset();
    row("gpu: model release returns the ledger to 0",
        vd.ledger().held(host::Held::kDevice) == 0 && vd.ledger().held(host::Held::kMapped) == 0);
    // Simulated sticky fault (never touches the GPU).
    vd.ctx().inject_fault_for_test("device_test simulated fault");
    row("gpu: faulted() after a sticky fault", dev->faulted());
    std::unique_ptr<VulkanModel> m2;
    row("gpu: allocate after a fault INTERNAL", vd.allocate_storage(p, &m2, &why) == CERE_ERR_INTERNAL && !m2);
}

}  // namespace

int main(int argc, char **argv) {
    const std::string dir = argc > 1 ? argv[1] : "";
    bind_rows(dir);
    lever_rows();
    gpu_rows();
    std::printf("device_test: %d PASS, %d FAIL, %d SKIP\n", g_pass, g_fail, g_skip);
    return g_fail == 0 ? 0 : 1;
}
