// VulkanAmd P3 T2: full-size load of the S3 FP4-Local artifact through the real ABI on the Vulkan device (linked
// statically from the library's own objects so the test can reach the device's mappings for the read-back check;
// every model step goes through the cere_* functions). Rows print PASS / FAIL; exit 1 on any FAIL.
//
// Usage: load_s3 ARTIFACT_DIR [WINDOWS]
// Flow: desc from the safetensors headers (the runtime's model_spec order and placement, plain profile: drafts 0, so
// mtp.* is DISCARD) -> cere_model_create (estimate + allocate) -> cere_model_put of every non-DISCARD tensor streamed
// from the shards (a reader thread + the putting thread) -> cere_model_seal (instantiate) -> capabilities ->
// cere_session_create(32,768) -> read-back verify (sha256 of WINDOWS random <= 1 MiB windows, spread over every block, vs
// the file) -> refusals (submit, selection, release order) -> release -> destroy, with MemAvailable sampled every 100 ms.
// Needs ~108 GB of memory: check `free -g` first and hold /tmp/cere-gpu.lock.
#include "device/vulkan_device.h"
#include "host/sha256.h"
#include "host/state.h"
#include "vk_artifact.h"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <random>
#include <thread>

using namespace cere;
using namespace cere::vkdev;
using Clock = std::chrono::steady_clock;

namespace {

int g_fail = 0, g_pass = 0;
void row(const std::string &name, bool ok, const std::string &detail = "") {
    std::printf("%-60s %s%s%s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.empty() ? "" : "  ", detail.c_str());
    std::fflush(stdout);
    (ok ? g_pass : g_fail)++;
}
double secs(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }
double gb(uint64_t b) { return double(b) / 1e9; }

uint64_t mem_available() {
    std::FILE *f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kib = 0;
    while (std::fgets(line, sizeof line, f)) {
        if (std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1) break;
    }
    std::fclose(f);
    return uint64_t(kib) * 1024;
}

struct Sampler {
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> min_avail{UINT64_MAX};
    std::thread th;
    void start() {
        th = std::thread([this] {
            while (!stop.load()) {
                const uint64_t a = mem_available();
                uint64_t cur = min_avail.load();
                while (a < cur && !min_avail.compare_exchange_weak(cur, a)) {
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }
    void finish() {
        stop = true;
        if (th.joinable()) th.join();
    }
};

std::string hex(const std::array<uint8_t, 32> &d, size_t n = 8) {
    char s[65];
    for (size_t i = 0; i < n; ++i) std::snprintf(s + 2 * i, 3, "%02x", d[i]);
    return std::string(s, 2 * n);
}

struct Chunk {
    uint64_t tensor = 0, offset = 0, len = 0;
    std::vector<uint8_t> buf;
};

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: load_s3 ARTIFACT_DIR [WINDOWS]\n");
        return 2;
    }
    const std::string dir = argv[1];
    const int windows = argc > 2 ? std::atoi(argv[2]) : 64;
    const uint64_t avail0 = mem_available();
    std::printf("MemAvailable before: %.2f GB\n", gb(avail0));
    Sampler sampler;
    sampler.start();

    // ---- desc ----
    std::vector<vktest::ArtTensor> art;
    std::string why;
    if (!vktest::artifact_tensors(dir, 0, &art, &why)) {
        row("artifact tensor table", false, why);
        sampler.finish();
        return 1;
    }
    std::vector<cere_tensor_desc> tensors(art.size());
    uint64_t dev_bytes = 0, host_bytes = 0;
    for (size_t i = 0; i < art.size(); ++i) {
        cere_tensor_desc &d = tensors[i];
        d.struct_size = sizeof d;
        d.abi_version = CERE_EXECUTOR_ABI_VERSION;
        d.name = art[i].t.name.c_str();
        d.dtype = art[i].t.dtype;
        d.placement = art[i].t.placement;
        d.ndim = art[i].t.ndim;
        for (uint32_t k = 0; k < d.ndim; ++k) d.shape[k] = art[i].t.shape[k];
        d.bytes = art[i].t.bytes;
        if (d.placement == CERE_PLACEMENT_DEVICE) dev_bytes += d.bytes;
        if (d.placement == CERE_PLACEMENT_HOST_PINNED) host_bytes += d.bytes;
    }
    row("artifact tensor table (headers)", true,
        std::to_string(art.size()) + " tensors, device " + std::to_string(dev_bytes) + " B, host " +
            std::to_string(host_bytes) + " B");
    const cere_qwen4_exp_config config = host::a4_support().config;
    static const uint32_t stops[1] = {248044};
    cere_model_desc desc{};
    desc.struct_size = sizeof desc;
    desc.abi_version = CERE_EXECUTOR_ABI_VERSION;
    desc.model_kind = CERE_MODEL_QWEN4_EXP_NVFP4;
    desc.family = CERE_FAMILY_QWEN4_EXP;
    desc.config = &config;
    desc.tensors = tensors.data();
    desc.tensor_count = tensors.size();
    desc.max_sessions = 1;
    desc.max_context_tokens = 32768;
    desc.stop_token_ids = stops;
    desc.stop_token_count = 1;

    // ---- executor + model ----
    cere_executor *e = nullptr;
    cere_executor_options o{};
    o.struct_size = sizeof o;
    o.abi_version = CERE_EXECUTOR_ABI_VERSION;
    o.device_id = "0";
    o.worker_epoch = 1;
    o.memory_budget_bytes = uint64_t(90) << 30;
    row("cere_executor_create(\"0\")", cere_executor_create(&e, &o) == CERE_OK && e != nullptr);
    if (!e) {
        sampler.finish();
        return 1;
    }
    auto &vd = static_cast<VulkanDevice &>(*e->device);
    cere_model *model = nullptr;
    auto t0 = Clock::now();
    const cere_status cs = cere_model_create(e, &desc, &model);
    const double t_alloc = secs(t0);
    row("cere_model_create (estimate + allocate + bind)", cs == CERE_OK && model != nullptr,
        "status " + std::to_string(cs) + ", " + std::to_string(t_alloc) + " s");
    if (cs != CERE_OK) {
        cere_executor_destroy(e);
        sampler.finish();
        return 1;
    }
    auto &vm = static_cast<VulkanModel &>(*model->device);
    const VkLayout &L = vm.layout();
    {
        uint32_t dl = 0, hh = 0;
        uint64_t dlb = 0, hhb = 0;
        for (size_t b = 0; b < L.blocks.size(); ++b) {
            const bool h = L.blocks[b].heap == VkHeap::kHostHeap;
            (h ? hh : dl)++;
            (h ? hhb : dlb) += vm.blocks()[b].size;
            std::printf("  block %2zu %-11s %-9s planned %11llu allocated %11llu mem_type %u flags 0x%x\n", b,
                        h ? "host-heap" : "device", vk_purpose_name(L.blocks[b].purpose),
                        (unsigned long long)L.blocks[b].bytes, (unsigned long long)vm.blocks()[b].size,
                        vm.blocks()[b].memory_type, unsigned(vd.ctx().memory_flags(vm.blocks()[b].memory_type)));
        }
        bool same = true;
        for (size_t b = 0; b < L.blocks.size(); ++b) same = same && vm.blocks()[b].size == L.blocks[b].bytes;
        row("allocated blocks == plan (count, sizes, heaps)",
            same && dlb == L.device_bytes && hhb == L.host_bytes && model->estimate.device_bytes == L.device_bytes &&
                model->estimate.host_bytes == L.host_bytes,
            std::to_string(dl) + " device-local (" + std::to_string(dlb) + " B) + " + std::to_string(hh) +
                " host-heap (" + std::to_string(hhb) + " B), layout " + vm.layout_digest().substr(0, 16));
        row("ledger == allocated", vd.ledger().held(host::Held::kDevice) == dlb &&
                                       vd.ledger().held(host::Held::kMapped) == hhb);
    }

    // ---- upload ----
    t0 = Clock::now();
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Chunk> q;
    bool done = false, read_err = false;
    constexpr uint64_t kChunk = uint64_t(128) << 20;
    constexpr size_t kQueue = 4;
    std::atomic<uint64_t> read_bytes{0};
    std::thread reader([&] {
        std::map<std::string, int> fds;
        for (size_t i = 0; i < art.size() && !read_err; ++i) {
            if (art[i].t.placement == CERE_PLACEMENT_DISCARD) continue;
            int &fd = fds[art[i].file];
            if (fd == 0) fd = ::open((dir + "/" + art[i].file).c_str(), O_RDONLY);
            for (uint64_t off = 0; off < art[i].t.bytes; off += kChunk) {
                Chunk c;
                c.tensor = i;
                c.offset = off;
                c.len = std::min(kChunk, art[i].t.bytes - off);
                c.buf.resize(c.len);
                uint64_t got = 0;
                while (got < c.len) {
                    const ssize_t r = ::pread(fd, c.buf.data() + got, c.len - got, off_t(art[i].data_at + off + got));
                    if (r <= 0) break;
                    got += uint64_t(r);
                }
                if (got != c.len) {
                    std::lock_guard<std::mutex> l(mu);
                    read_err = true;
                    break;
                }
                read_bytes += c.len;
                std::unique_lock<std::mutex> l(mu);
                cv.wait(l, [&] { return q.size() < kQueue; });
                q.push_back(std::move(c));
                cv.notify_all();
            }
        }
        for (auto &kv : fds) ::close(kv.second);
        std::lock_guard<std::mutex> l(mu);
        done = true;
        cv.notify_all();
    });
    uint64_t put_bytes = 0, put_err = 0;
    double put_s = 0;
    for (;;) {
        Chunk c;
        {
            std::unique_lock<std::mutex> l(mu);
            cv.wait(l, [&] { return !q.empty() || done; });
            if (q.empty()) break;
            c = std::move(q.front());
            q.pop_front();
            cv.notify_all();
        }
        const auto tp = Clock::now();
        if (cere_model_put(e, model, c.tensor, c.offset, c.buf.data(), c.len) != CERE_OK) ++put_err;
        put_s += secs(tp);
        put_bytes += c.len;
    }
    reader.join();
    const double t_upload = secs(t0);
    row("upload: cere_model_put of every non-DISCARD tensor", !read_err && put_err == 0 && put_bytes == dev_bytes + host_bytes,
        std::to_string(put_bytes) + " B in " + std::to_string(t_upload) + " s = " +
            std::to_string(gb(put_bytes) / t_upload) + " GB/s end to end; put (memcpy) " +
            std::to_string(gb(put_bytes) / put_s) + " GB/s over " + std::to_string(put_s) + " s");
    {
        uint8_t x[16] = {};
        row("put past a tensor ARGUMENT", cere_model_put(e, model, 0, tensors[0].bytes - 8, x, 16) == CERE_ERR_ARGUMENT);
        row("put of an already written range ARGUMENT", cere_model_put(e, model, 0, 0, x, 16) == CERE_ERR_ARGUMENT);
    }

    // ---- seal ----
    {
        cere_execution_capabilities c{};
        c.struct_size = sizeof c;
        c.abi_version = CERE_EXECUTOR_ABI_VERSION;
        row("capabilities before seal ARGUMENT", cere_executor_capabilities(e, model, &c) == CERE_ERR_ARGUMENT);
    }
    t0 = Clock::now();
    const cere_status ss = cere_model_seal(e, model);
    const double t_seal = secs(t0);
    row("cere_model_seal (instantiate)", ss == CERE_OK, "status " + std::to_string(ss) + ", " + std::to_string(t_seal) + " s");
    {
        const InstantiateReport &r = vm.instantiate_report();
        std::printf("  instantiate: probes %.1f ms, warm %.1f ms (%zu specs, %zu pipelines), host %.1f ms, gpu %.1f ms, "
                    "winfo %u, moe info %u, total %.1f ms\n  warm lists: %s\n",
                    r.probe_ms, r.warm_ms, r.warm_specs, r.pipelines, r.host_ms, r.gpu_ms, r.winfo_entries,
                    r.moe_info_layers, r.total_ms, r.warm_lists.c_str());
    }
    if (ss == CERE_OK) {
        // Derived values readable through the mapping: A_log F32 of layer 0 == the BF16 source; RoPE row 1 cos != 0.
        const VkModelWeights &w = vm.weights();
        float alog = 0, rope1 = 0;
        uint16_t src = 0;
        std::memcpy(&alog, vm.host_of(w.layers[0].gdn.A_log), 4);
        std::memcpy(&src, vm.host_of(w.layers[0].gdn_A_log_bf16), 2);
        const uint32_t u = uint32_t(src) << 16;
        float want;
        std::memcpy(&want, &u, 4);
        std::memcpy(&rope1, vm.host_of(w.rope_cos_sin) + 4, 4);
        float rope0 = 0;
        std::memcpy(&rope0, vm.host_of(w.rope_cos_sin), 4);
        float a1 = w.layers[0].moe.a1_global_scale;
        row("derived: A_log F32 == BF16 source, RoPE row 0 cos == 1, a1 > 0", alog == want && rope0 == 1.0f && a1 > 0,
            "A_log[0] " + std::to_string(alog) + ", a1 " + std::to_string(a1));
    }

    cere_execution_capabilities caps{};
    caps.struct_size = sizeof caps;
    caps.abi_version = CERE_EXECUTOR_ABI_VERSION;
    const cere_status cap_st = cere_executor_capabilities(e, model, &caps);
    std::printf("  capabilities: status %d execution_mode %u operations 0x%llx decode_batch %llu..%llu max_context %llu "
                "native_decode_batch %u native_sampler %u native_token_mask %u native_speculative_batch %u "
                "speculative_draft_tokens %u native_state_snapshot %u native_prefix_cache %u native_image_input %u "
                "kv_pool_tokens %llu kv_page_tokens %u profile_generation %llu worker_epoch %llu\n",
                cap_st, unsigned(caps.execution_mode), (unsigned long long)caps.operations,
                (unsigned long long)caps.decode_batch_min, (unsigned long long)caps.decode_batch_max,
                (unsigned long long)caps.max_context_tokens, caps.native_decode_batch, caps.native_sampler,
                caps.native_token_mask, caps.native_speculative_batch, caps.speculative_draft_tokens,
                caps.native_state_snapshot, caps.native_prefix_cache, caps.native_image_input,
                (unsigned long long)caps.kv_pool_tokens, caps.kv_page_tokens,
                (unsigned long long)caps.profile_generation, (unsigned long long)caps.worker_epoch);
    // T5: the forward runs (prefill, decode, cancel; the K12 sampler), speculation does not (T6).
    const uint64_t ops = (1u << CERE_OP_PREFILL) | (1u << CERE_OP_DECODE);
    row("capabilities: native forward (prefill + decode + cancel, sampler), no verify / speculation",
        cap_st == CERE_OK && caps.execution_mode == CERE_EXECUTION_MODE_NATIVE && (caps.operations & ops) == ops &&
            (caps.operations & (1u << CERE_OP_VERIFY)) == 0 &&
            caps.native_sampler == 1 && caps.native_speculative_batch == 0 && caps.native_prefix_cache == 0 &&
            caps.native_image_input == 0 && caps.max_context_tokens == 32768 && caps.kv_pool_tokens == 32768);

    // ---- session ----
    cere_session *session = nullptr;
    t0 = Clock::now();
    const cere_status sc = cere_session_create(e, model, 32768, &session);
    const double t_session = secs(t0);
    row("cere_session_create(32,768)", sc == CERE_OK && session, "status " + std::to_string(sc) + ", " +
                                                                    std::to_string(t_session * 1000) + " ms");
    {
        cere_session *s2 = nullptr;
        row("second session CAPACITY (max_sessions 1)", cere_session_create(e, model, 1024, &s2) == CERE_ERR_CAPACITY);
    }

    // ---- read-back verify ----
    {
        t0 = Clock::now();
        std::mt19937_64 rng(20261115);
        // Candidate pieces per block (non-DISCARD tensors).
        std::vector<std::vector<std::pair<uint64_t, VkPiece>>> by_block(L.blocks.size());
        for (uint64_t i = 0; i < art.size(); ++i) {
            for (uint64_t p = L.piece_first[i]; p < L.piece_first[i + 1]; ++p) by_block[L.pieces[p].block].push_back({i, L.pieces[p]});
        }
        std::map<std::string, int> fds;
        int done_w = 0, bad = 0, blocks_hit = 0;
        uint64_t checked = 0;
        std::vector<uint8_t> a, b;
        for (int wi = 0; wi < windows; ++wi) {
            // Round-robin over blocks with tensors, then random.
            size_t blk = size_t(wi) % L.blocks.size();
            while (by_block[blk].empty()) blk = (blk + 1) % L.blocks.size();
            if (wi < int(L.blocks.size())) ++blocks_hit;
            const auto &cand = by_block[blk][rng() % by_block[blk].size()];
            const uint64_t ti = cand.first;
            const VkPiece &pc = cand.second;
            const uint64_t len = std::min<uint64_t>(uint64_t(1) << 20, pc.bytes);
            const uint64_t off = pc.tensor_offset + (pc.bytes > len ? rng() % (pc.bytes - len + 1) : 0);
            a.resize(len);
            b.resize(len);
            int &fd = fds[art[ti].file];
            if (fd == 0) fd = ::open((dir + "/" + art[ti].file).c_str(), O_RDONLY);
            const bool rd = ::pread(fd, a.data(), len, off_t(art[ti].data_at + off)) == ssize_t(len);
            const bool rm = vm.read(ti, off, b.data(), len) == CERE_OK;
            const auto ha = host::sha256(a.data(), len), hb = host::sha256(b.data(), len);
            if (!rd || !rm || ha != hb) {
                ++bad;
                std::printf("  MISMATCH block %zu tensor %s off %llu len %llu file %s mapped %s\n", blk,
                            art[ti].t.name.c_str(), (unsigned long long)off, (unsigned long long)len, hex(ha).c_str(),
                            hex(hb).c_str());
            }
            ++done_w;
            checked += len;
        }
        for (auto &kv : fds) ::close(kv.second);
        row("read-back: sha256 of windows over every block == the file", bad == 0 && done_w == windows,
            std::to_string(done_w) + " windows, " + std::to_string(checked) + " B, " + std::to_string(blocks_hit) +
                " blocks, " + std::to_string(bad) + " mismatches, " + std::to_string(secs(t0)) + " s");
    }

    // ---- refusals while loaded ----
    if (session != nullptr) {
        const uint32_t toks[3] = {760, 6511, 314};
        uint32_t out_tokens[8] = {};
        cere_work w{};
        w.struct_size = sizeof w;
        w.abi_version = CERE_EXECUTOR_ABI_VERSION;
        w.operation = CERE_OP_PREFILL;
        w.expected_worker_epoch = 1;
        w.session = session;
        w.tokens = toks;
        w.token_count = 3;
        w.max_new_tokens = 4;
        w.out_tokens = out_tokens;
        w.out_capacity = 8;
        cere_batch batch{};
        batch.struct_size = sizeof batch;
        batch.abi_version = CERE_EXECUTOR_ABI_VERSION;
        batch.expected_worker_epoch = 1;
        batch.capability_profile_generation = caps.profile_generation;
        batch.items = &w;
        batch.item_count = 1;
        cere_completion_ticket ticket{};
        const cere_status st = cere_executor_submit(e, &batch, &ticket);
        cere_completion c{};
        cere_status ps = st;
        for (int i = 0; st == CERE_OK && i < 64; ++i) {
            c = cere_completion{};
            c.struct_size = sizeof c;
            c.abi_version = CERE_EXECUTOR_ABI_VERSION;
            ps = cere_executor_poll(e, ticket, &c);
            if (c.terminal) break;
        }
        row("submit PREFILL (3 tokens) + 4 new tokens runs to terminal (T5)",
            st == CERE_OK && c.terminal && c.status == CERE_OK && c.committed_tokens >= 1,
            "submit " + std::to_string(st) + " poll " + std::to_string(ps) + " committed " +
                std::to_string(c.committed_tokens) + " finish " + std::to_string(c.finish));
    }
    {
        cere_slot_selection sel{};
        sel.struct_size = sizeof sel;
        sel.abi_version = CERE_EXECUTOR_ABI_VERSION;
        sel.block_version = CERE_SLOT_BLOCK_VERSION;
        sel.count = 1;
        sel.choices[0].slot = CERE_SLOT_DECODE_STEP;
        sel.choices[0].impl = CERE_LEVER_DECODE_GRAPH;
        row("set_selection(decode graph) UNSUPPORTED", cere_model_set_selection(e, model, &sel) == CERE_ERR_UNSUPPORTED);
        sel.choices[0].slot = CERE_SLOT_MOE_GEMM;
        sel.choices[0].impl = CERE_LEVER_MOE_K7;
        row("set_selection(moe k7) OK", cere_model_set_selection(e, model, &sel) == CERE_OK);
    }
    row("model_release with a live session BUSY", cere_model_release(e, model) == CERE_ERR_BUSY);
    row("session_release OK", session && cere_session_release(e, session) == CERE_OK);
    row("session_release again ARGUMENT", session && cere_session_release(e, session) == CERE_ERR_ARGUMENT);
    t0 = Clock::now();
    const cere_status rel = cere_model_release(e, model);
    const double t_release = secs(t0);
    row("model_release OK (frees every block)", rel == CERE_OK, std::to_string(t_release) + " s");
    row("model_release again ARGUMENT", cere_model_release(e, model) == CERE_ERR_ARGUMENT);
    row("ledger back to 0", vd.ledger().held(host::Held::kDevice) == 0 && vd.ledger().held(host::Held::kMapped) == 0);
    row("executor destroy OK", cere_executor_destroy(e) == CERE_OK);
    sampler.finish();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const uint64_t avail1 = mem_available();
    const uint64_t peak_used = avail0 - std::min(avail0, sampler.min_avail.load());
    std::printf("MemAvailable: before %.2f GB, minimum %.2f GB (peak use %.2f GB), after %.2f GB\n", gb(avail0),
                gb(sampler.min_avail.load()), gb(peak_used), gb(avail1));
    row("memory returned after destroy (MemAvailable within 2 GB of before)",
        avail1 + (uint64_t(2) << 30) >= avail0, std::to_string(gb(avail1)) + " vs " + std::to_string(gb(avail0)) + " GB");
    std::printf("timings: allocate %.2f s, upload %.1f s (%.2f GB/s), seal %.2f s, session %.1f ms\n", t_alloc, t_upload,
                gb(dev_bytes + host_bytes) / t_upload, t_seal, t_session * 1000);
    std::printf("load_s3: %d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
