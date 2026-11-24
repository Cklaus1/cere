// VulkanAmd P3 T6: the S3 speculation harness. Loads the S3 FP4-Local artifact WITH its MTP draft head
// (speculative_draft_tokens 4, the artifact's draft/token_map.u32le) through the cere_* ABI of a given executor library
// and, per prompt (token ids from tests/data/forward_s3_prompts.txt), runs in ONE model and session pool:
//   plain  : greedy decode of N tokens with CERE_WORK_NO_SPECULATION (the session does not draft: plain decode steps)
//   spec   : greedy decode of N tokens, speculating (MTP drafts, verify, accept), twice (determinism)
// and checks the lossless contract (executor/cuda/tests/host/test_state.cpp test_draft_tokens_lever_greedy_equals_plain,
// m29 design note R3): the speculating run's committed ids AND their cere_token_score (top-2 ids and both logits,
// bit for bit) equal the plain run's at every position. Then: a stop-token prompt (finish STOP inside a chain), the
// temperature-0 sampler path (phase S) == greedy, seeded sampled speculation (twice identical, draws inside top_k /
// top_p), a cancel after a few rounds followed by a DECODE continuation (compared with the plain run), negative rows
// (lever values outside the Vulkan set, a VERIFY on a session that does not draft), and the device-level rows of the
// static build (speculate_rows before the prompt, on a non-drafting session, malformed calls; GPU split per round).
// Per-round acceptance comes from the per-poll deltas of cere_completion (one speculation round per poll).
//
// Builds (executor/vulkan/build_lib.sh): spec_s3 (dlopen LIB: the Vulkan library, ABI v5, or the amd-amd HIP library,
// ABI v4) and spec_s3_static (the Vulkan library's objects linked in: device rows and statistics; LIB ignored).
// Usage: spec_s3 LIB ARTIFACT PROMPTS OUT_TRACE [N=64] [--prompts a,b] [--long name:N] [--quick]
//                [--prefill-attention cuda_core|p4]   (P4-E: the model's prefill_attention selection, as forward_s3)
//                [--k2-gemm cublaslt|tail_split]      (P4-B, D7, ABI v5 only: the model's k2_gemm selection, as forward_s3)
// Memory: ~107 GB; check MemAvailable first and hold /tmp/cere-gpu.lock.
#include "cere_executor.h"
#include "host/support.h"
#include "vk_artifact.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

#if defined(FS3_STATIC)
#include "device/vulkan_device.h"
#include "host/state.h"
#endif

using Clock = std::chrono::steady_clock;

namespace {

double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

struct Api {
    uint32_t (*abi_version)();
    cere_status (*executor_create)(cere_executor **, const cere_executor_options *);
    cere_status (*executor_destroy)(cere_executor *);
    cere_status (*capabilities)(cere_executor *, const cere_model *, cere_execution_capabilities *);
    cere_status (*model_create)(cere_executor *, const cere_model_desc *, cere_model **);
    cere_status (*model_put)(cere_executor *, cere_model *, uint64_t, uint64_t, const void *, uint64_t);
    cere_status (*model_seal)(cere_executor *, cere_model *);
    cere_status (*model_release)(cere_executor *, cere_model *);
    cere_status (*session_create)(cere_executor *, cere_model *, uint64_t, cere_session **);
    cere_status (*session_release)(cere_executor *, cere_session *);
    cere_status (*submit)(cere_executor *, const cere_batch *, cere_completion_ticket *);
    cere_status (*poll)(cere_executor *, cere_completion_ticket, cere_completion *);
    cere_status (*cancel)(cere_executor *, cere_completion_ticket);
};

template <typename F>
bool sym(void *h, const char *name, F *out) {
    *out = reinterpret_cast<F>(dlsym(h, name));
    if (*out == nullptr) std::fprintf(stderr, "missing symbol %s\n", name);
    return *out != nullptr;
}

struct LeversV4 {
    uint32_t struct_size, abi_version, k2_gemm, prefill_attention, decode_step, moe_gemm, draft_tokens,
        verify_attention, k2_weights, calibrate_windows;
};

struct Prompt {
    std::string name;
    std::vector<uint32_t> ids;
};

bool read_prompts(const std::string &path, std::vector<Prompt> *out) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream s(line);
        Prompt p;
        uint64_t count = 0;
        s >> p.name >> count;
        uint64_t v;
        while (s >> v) p.ids.push_back(uint32_t(v));
        if (p.ids.size() != count) return false;
        out->push_back(std::move(p));
    }
    return !out->empty();
}

int g_fail = 0, g_pass = 0;
void row(const std::string &name, bool ok, const std::string &detail = "") {
    std::printf("%-72s %s%s%s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.empty() ? "" : "  ", detail.c_str());
    std::fflush(stdout);
    (ok ? g_pass : g_fail)++;
}

struct Gen {
    cere_status status = CERE_OK, submit_status = CERE_OK;
    uint32_t finish = 0;
    std::vector<uint32_t> tokens;
    std::vector<cere_token_score> scores;
    std::vector<cere_token_logprobs> logprobs;
    double ttft_ms = 0, decode_ms = 0, total_ms = 0;
    uint64_t polls = 0, verify_steps = 0, accepted = 0, revision = 0;
    std::vector<uint32_t> round_committed, round_accepted;  // per poll after the first token
    cere_session *session = nullptr;                        // kept when asked
};

struct GenArgs {
    uint32_t operation = CERE_OP_PREFILL;
    const std::vector<uint32_t> *prompt = nullptr;
    uint64_t n_new = 0;
    uint32_t flags = 0;
    const cere_sampling_params *sampling = nullptr;
    uint32_t top_logprobs = 0;
    bool logprobs = false;
    std::vector<uint32_t> stop;
    cere_session *session = nullptr;  // DECODE / VERIFY on an existing session
    uint64_t revision = 0;
    bool keep_session = false;
    uint64_t cancel_after_polls = 0;  // > 0: cancel after that many polls
};

struct Runner {
    Api api{};
    uint32_t abi = 0;
    cere_executor *e = nullptr;
    cere_model *model = nullptr;
    cere_execution_capabilities caps{};

    template <typename T>
    void stamp(T *x) {
        x->struct_size = sizeof(T);
        x->abi_version = abi;
    }

    Gen generate(const GenArgs &a) {
        Gen g;
        cere_session *s = a.session;
        if (s == nullptr) {
            g.status = api.session_create(e, model, 32768, &s);
            if (g.status != CERE_OK) return g;
        }
        const uint64_t cap = std::max<uint64_t>(a.n_new, 1);
        std::vector<uint32_t> out(cap);
        std::vector<cere_token_score> scores(cap);
        std::vector<cere_token_logprobs> lps(cap);
        cere_work w{};
        stamp(&w);
        w.operation = a.operation;
        w.flags = a.flags;
        w.expected_worker_epoch = 1;
        w.expected_session_revision = a.revision;
        w.session = s;
        w.tokens = a.prompt != nullptr ? a.prompt->data() : nullptr;
        w.token_count = a.prompt != nullptr ? a.prompt->size() : 0;
        w.max_new_tokens = a.n_new;
        w.out_tokens = out.data();
        w.out_capacity = cap;
        w.out_scores = scores.data();
        w.sampling = a.sampling;
        w.out_logprobs = a.logprobs ? lps.data() : nullptr;
        w.top_logprobs = a.logprobs ? a.top_logprobs : 0;
        w.stop_token_ids = a.stop.empty() ? nullptr : a.stop.data();
        w.stop_token_count = a.stop.size();
        cere_batch b{};
        stamp(&b);
        b.expected_worker_epoch = 1;
        b.capability_profile_generation = caps.profile_generation;
        b.items = &w;
        b.item_count = 1;
        cere_completion_ticket ticket{};
        const auto t0 = Clock::now();
        g.submit_status = g.status = api.submit(e, &b, &ticket);
        if (g.status != CERE_OK) {
            if (a.session == nullptr && !a.keep_session) api.session_release(e, s);
            g.session = a.session == nullptr && !a.keep_session ? nullptr : s;
            return g;
        }
        Clock::time_point first{};
        bool have_first = false;
        uint64_t last_committed = 0, last_accepted = 0;
        for (;;) {
            cere_completion c{};
            stamp(&c);
            const cere_status ps = api.poll(e, ticket, &c);
            const auto tq = Clock::now();
            ++g.polls;
            if (ps != CERE_OK && ps != CERE_PENDING) {
                g.status = ps;
                break;
            }
            if (have_first && c.committed_tokens > last_committed) {
                g.round_committed.push_back(uint32_t(c.committed_tokens - last_committed));
                g.round_accepted.push_back(uint32_t(c.accepted_drafts - last_accepted));
            }
            if (!have_first && c.committed_tokens >= 1) {
                first = tq;
                have_first = true;
                g.ttft_ms = ms(t0, tq);
            }
            last_committed = c.committed_tokens;
            last_accepted = c.accepted_drafts;
            if (a.cancel_after_polls != 0 && g.polls == a.cancel_after_polls && !c.terminal) api.cancel(e, ticket);
            if (c.terminal) {
                g.status = c.status;
                g.finish = c.finish;
                const uint64_t k = c.committed_tokens;
                g.tokens.assign(out.begin(), out.begin() + k);
                g.scores.assign(scores.begin(), scores.begin() + k);
                if (a.logprobs) g.logprobs.assign(lps.begin(), lps.begin() + k);
                g.total_ms = ms(t0, tq);
                if (have_first) g.decode_ms = ms(first, tq);
                g.verify_steps = c.verify_steps;
                g.accepted = c.accepted_drafts;
                g.revision = c.session_revision;
                break;
            }
        }
        if (a.session == nullptr && !a.keep_session) {
            api.session_release(e, s);
        } else {
            g.session = s;
        }
        return g;
    }
};

uint32_t fbits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// The first position where two runs differ (ids or any score bit), or -1.
long first_diff(const Gen &a, const Gen &b, size_t from_a = 0, size_t from_b = 0) {
    const size_t na = a.tokens.size() - std::min(a.tokens.size(), from_a);
    const size_t nb = b.tokens.size() - std::min(b.tokens.size(), from_b);
    const size_t n = std::min(na, nb);
    for (size_t k = 0; k < n; ++k) {
        if (a.tokens[from_a + k] != b.tokens[from_b + k] ||
            std::memcmp(&a.scores[from_a + k], &b.scores[from_b + k], sizeof(cere_token_score)) != 0) {
            return long(k);
        }
    }
    return na == nb ? -1 : long(n);
}

std::string score_str(const cere_token_score &s) {
    char b[160];
    std::snprintf(b, sizeof b, "%u/%u %.6g/%.6g (%08x/%08x)", s.id, s.runner_up, double(s.logit),
                  double(s.runner_up_logit), fbits(s.logit), fbits(s.runner_up_logit));
    return b;
}

void trace_gen(FILE *tr, const std::string &tag, const Gen &g) {
    for (size_t k = 0; k < g.tokens.size(); ++k) {
        const cere_token_score &s = g.scores[k];
        std::fprintf(tr, "step %s %zu %u %u %u %.9g %.9g %08x %08x\n", tag.c_str(), k, g.tokens[k], s.id, s.runner_up,
                     double(s.logit), double(s.runner_up_logit), fbits(s.logit), fbits(s.runner_up_logit));
    }
    for (size_t r = 0; r < g.round_committed.size(); ++r) {
        std::fprintf(tr, "round %s %zu %u %u\n", tag.c_str(), r, g.round_committed[r], g.round_accepted[r]);
    }
    std::fflush(tr);
}

double tok_s(const Gen &g) {
    return g.tokens.size() > 1 && g.decode_ms > 0 ? double(g.tokens.size() - 1) / (g.decode_ms / 1000) : 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: spec_s3 LIB ARTIFACT PROMPTS OUT_TRACE [N] [--prompts a,b] [--long name:N] "
                             "[--quick]\n");
        return 2;
    }
    const std::string lib = argv[1], dir = argv[2], prompts_path = argv[3], trace_path = argv[4];
    uint64_t n_new = 64;
    std::string only, long_spec, prefill_attention, k2_gemm;
    bool quick = false;
    for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--prompts" && i + 1 < argc) {
            only = argv[++i];
        } else if (a == "--long" && i + 1 < argc) {
            long_spec = argv[++i];
        } else if (a == "--quick") {
            quick = true;
        } else if (a == "--prefill-attention" && i + 1 < argc) {
            prefill_attention = argv[++i];
            if (prefill_attention != "cuda_core" && prefill_attention != "p4") {
                std::fprintf(stderr, "--prefill-attention: cuda_core or p4\n");
                return 2;
            }
        } else if (a == "--k2-gemm" && i + 1 < argc) {
            k2_gemm = argv[++i];
            if (k2_gemm != "cublaslt" && k2_gemm != "tail_split") {
                std::fprintf(stderr, "--k2-gemm: cublaslt or tail_split\n");
                return 2;
            }
        } else {
            n_new = std::strtoull(a.c_str(), nullptr, 10);
        }
    }
    std::vector<Prompt> prompts;
    if (!read_prompts(prompts_path, &prompts)) {
        std::fprintf(stderr, "cannot read prompts %s\n", prompts_path.c_str());
        return 2;
    }
    if (!only.empty()) {
        std::vector<Prompt> keep;
        for (Prompt &p : prompts) {
            if (("," + only + ",").find("," + p.name + ",") != std::string::npos) keep.push_back(std::move(p));
        }
        prompts.swap(keep);
    }
    std::map<std::string, uint64_t> n_of;
    if (!long_spec.empty()) {
        const size_t colon = long_spec.find(':');
        n_of[long_spec.substr(0, colon)] = std::strtoull(long_spec.substr(colon + 1).c_str(), nullptr, 10);
    }

    Runner R;
#if defined(FS3_STATIC)
    (void)lib;
    R.api = Api{cere_executor_abi_version, cere_executor_create, cere_executor_destroy, cere_executor_capabilities,
                cere_model_create, cere_model_put, cere_model_seal, cere_model_release, cere_session_create,
                cere_session_release, cere_executor_submit, cere_executor_poll, cere_executor_cancel};
    std::printf("library: (static Vulkan objects)\n");
#else
    void *h = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (h == nullptr) {
        std::fprintf(stderr, "dlopen %s: %s\n", lib.c_str(), dlerror());
        return 2;
    }
    bool ok = sym(h, "cere_executor_abi_version", &R.api.abi_version) &&
              sym(h, "cere_executor_create", &R.api.executor_create) &&
              sym(h, "cere_executor_destroy", &R.api.executor_destroy) &&
              sym(h, "cere_executor_capabilities", &R.api.capabilities) &&
              sym(h, "cere_model_create", &R.api.model_create) && sym(h, "cere_model_put", &R.api.model_put) &&
              sym(h, "cere_model_seal", &R.api.model_seal) && sym(h, "cere_model_release", &R.api.model_release) &&
              sym(h, "cere_session_create", &R.api.session_create) &&
              sym(h, "cere_session_release", &R.api.session_release) &&
              sym(h, "cere_executor_submit", &R.api.submit) && sym(h, "cere_executor_poll", &R.api.poll) &&
              sym(h, "cere_executor_cancel", &R.api.cancel);
    if (!ok) return 2;
    std::printf("library: %s\n", lib.c_str());
#endif
    R.abi = R.api.abi_version();
    std::printf("abi_version: %u\n", R.abi);
    if (R.abi != 4 && R.abi != 5) {
        std::fprintf(stderr, "unsupported ABI %u\n", R.abi);
        return 2;
    }

    // ---- desc with drafts (vk_artifact.h: `mtp.` DEVICE) and the artifact's draft token map ----
    std::vector<vktest::ArtTensor> art;
    std::string why;
    if (!vktest::artifact_tensors(dir, 1, &art, &why)) {
        std::fprintf(stderr, "artifact: %s\n", why.c_str());
        return 2;
    }
    std::vector<uint32_t> draft_map;
    {
        std::ifstream f(dir + "/draft/token_map.u32le", std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        draft_map.resize(bytes.size() / 4);
        std::memcpy(draft_map.data(), bytes.data(), draft_map.size() * 4);
    }
    std::printf("draft token map: %zu ids\n", draft_map.size());
    std::vector<cere_tensor_desc> tensors(art.size());
    uint64_t total = 0;
    for (size_t i = 0; i < art.size(); ++i) {
        cere_tensor_desc &d = tensors[i];
        R.stamp(&d);
        d.name = art[i].t.name.c_str();
        d.dtype = art[i].t.dtype;
        d.placement = art[i].t.placement;
        d.ndim = art[i].t.ndim;
        for (uint32_t k = 0; k < d.ndim; ++k) d.shape[k] = art[i].t.shape[k];
        d.bytes = art[i].t.bytes;
        if (d.placement != CERE_PLACEMENT_DISCARD) total += d.bytes;
    }
    cere_qwen4_exp_config config = cere::host::a4_support().config;
    config.struct_size = sizeof config;
    config.abi_version = R.abi;
    cere_model_desc desc{};
    desc.struct_size = R.abi == 5 ? uint32_t(sizeof desc) : 128u;
    desc.abi_version = R.abi;
    desc.model_kind = CERE_MODEL_QWEN4_EXP_NVFP4;
    desc.speculative_draft_tokens = 4;
    desc.config = &config;
    desc.tensors = tensors.data();
    desc.tensor_count = tensors.size();
    desc.max_sessions = 1;
    desc.max_context_tokens = 32768;
    desc.stop_token_ids = nullptr;  // no model stop token: every run decodes N tokens unless an item stop list ends it
    desc.draft_token_map = draft_map.data();
    desc.draft_token_map_count = draft_map.size();
    const uint32_t pa = prefill_attention == "p4" ? CERE_LEVER_PREFILL_P4 : CERE_LEVER_PREFILL_CUDA_CORE;
    LeversV4 lv4{sizeof(LeversV4), 4, 0, pa, 0, CERE_LEVER_MOE_K7, 0, 0, 0, 0};
    cere_slot_selection psel{};
    psel.struct_size = sizeof psel;
    psel.abi_version = R.abi;
    psel.block_version = CERE_SLOT_BLOCK_VERSION;
    if (!k2_gemm.empty()) {
        psel.choices[psel.count].slot = CERE_SLOT_K2_GEMM;
        psel.choices[psel.count++].impl = k2_gemm == "tail_split" ? CERE_LEVER_K2_TAIL_SPLIT : CERE_LEVER_K2_CUBLASLT;
    }
    if (!prefill_attention.empty()) {
        psel.choices[psel.count].slot = CERE_SLOT_PREFILL_ATTENTION;
        psel.choices[psel.count++].impl = pa;
    }
    std::printf("k2_gemm: %s\n", k2_gemm.empty() ? "(library default)" : k2_gemm.c_str());
    std::printf("prefill_attention: %s\n", prefill_attention.empty() ? "(library default)" : prefill_attention.c_str());
    if (R.abi == 4) {
        if (!k2_gemm.empty()) {
            std::fprintf(stderr, "--k2-gemm: ABI v5 libraries only\n");
            return 2;
        }
        const void *levers = &lv4;
        std::memcpy(reinterpret_cast<uint8_t *>(&desc) + offsetof(cere_model_desc, selection), &levers, sizeof levers);
    } else {
        desc.family = CERE_FAMILY_QWEN4_EXP;
        if (psel.count != 0) desc.selection = &psel;
    }

    cere_executor_options o{};
    R.stamp(&o);
    o.device_id = "0";
    o.worker_epoch = 1;
    o.memory_budget_bytes = uint64_t(100) << 30;
    if (R.api.executor_create(&R.e, &o) != CERE_OK) {
        row("executor_create(\"0\")", false);
        return 1;
    }
    // ---- negative model rows (Vulkan library only: refused before any block is allocated) ----
    if (R.abi == 5) {
        const auto refused = [&](const char *name, const cere_slot_choice &choice, cere_status want) {
            cere_slot_selection sel{};
            R.stamp(&sel);
            sel.block_version = CERE_SLOT_BLOCK_VERSION;
            sel.count = 1;
            sel.choices[0] = choice;
            cere_model_desc d = desc;
            d.selection = &sel;
            cere_model *mm = nullptr;
            const cere_status st = R.api.model_create(R.e, &d, &mm);
            if (mm != nullptr) R.api.model_release(R.e, mm);
            row(std::string("negative: drafts model with ") + name + " refused", st == want,
                "status " + std::to_string(st));
        };
        refused("decode_step spec_graph", cere_slot_choice{CERE_SLOT_DECODE_STEP, CERE_LEVER_DECODE_SPEC_GRAPH, {}},
                CERE_ERR_UNSUPPORTED);
        refused("decode_step graph", cere_slot_choice{CERE_SLOT_DECODE_STEP, CERE_LEVER_DECODE_GRAPH, {}},
                CERE_ERR_UNSUPPORTED);
        refused("verify_attention batched",
                cere_slot_choice{CERE_SLOT_VERIFY_ATTENTION, CERE_LEVER_VERIFY_BATCHED, {}}, CERE_ERR_UNSUPPORTED);
        refused("k2_gemm small_m", cere_slot_choice{CERE_SLOT_K2_GEMM, CERE_LEVER_K2_SMALL_M, {}},
                CERE_ERR_UNSUPPORTED);
        cere_model_desc d3 = desc;
        d3.speculative_draft_tokens = 3;
        cere_model *mm = nullptr;
        const cere_status st3 = R.api.model_create(R.e, &d3, &mm);
        if (mm != nullptr) R.api.model_release(R.e, mm);
        row("negative: speculative_draft_tokens 3 refused", st3 == CERE_ERR_UNSUPPORTED || st3 == CERE_ERR_ARGUMENT,
            "status " + std::to_string(st3));
    }
    auto t0 = Clock::now();
    cere_status st = R.api.model_create(R.e, &desc, &R.model);
    row("model_create (drafts 4, token map " + std::to_string(draft_map.size()) + ")", st == CERE_OK,
        "status " + std::to_string(st) + ", " + std::to_string(ms(t0, Clock::now())) + " ms");
    if (st != CERE_OK) return 1;
#if defined(FS3_STATIC)
    {
        using namespace cere::vkdev;
        auto &vm = static_cast<VulkanModel &>(*R.model->device);
        uint64_t planned = 0, held = 0;
        for (size_t b = 0; b < vm.blocks().size(); ++b) {
            planned += vm.layout().blocks[b].bytes;
            held += vm.blocks()[b].size;
        }
        row("memory: allocated == planned with drafts (" + std::to_string(vm.blocks().size()) + " blocks)",
            planned == held && held == vm.held_device_bytes() + vm.held_host_bytes(),
            std::to_string(held) + " B held, device " + std::to_string(vm.held_device_bytes()) + " host " +
                std::to_string(vm.held_host_bytes()) + ", undo " + std::to_string(vm.layout().undo_bytes) +
                " x 3, verify_rows " + std::to_string(vm.layout().verify_rows));
    }
#endif
    // ---- upload (reader thread + puts) ----
    t0 = Clock::now();
    {
        std::mutex mu;
        std::condition_variable cv;
        struct Chunk {
            uint64_t tensor, offset;
            std::vector<uint8_t> buf;
        };
        std::deque<Chunk> q;
        bool done = false, err = false;
        constexpr uint64_t kChunk = uint64_t(128) << 20;
        std::thread reader([&] {
            std::map<std::string, int> fds;
            for (size_t i = 0; i < art.size() && !err; ++i) {
                if (art[i].t.placement == CERE_PLACEMENT_DISCARD) continue;
                int &fd = fds[art[i].file];
                if (fd == 0) fd = ::open((dir + "/" + art[i].file).c_str(), O_RDONLY);
                for (uint64_t off = 0; off < art[i].t.bytes; off += kChunk) {
                    Chunk c{i, off, std::vector<uint8_t>(std::min(kChunk, art[i].t.bytes - off))};
                    uint64_t got = 0;
                    while (got < c.buf.size()) {
                        const ssize_t r = ::pread(fd, c.buf.data() + got, c.buf.size() - got,
                                                  off_t(art[i].data_at + off + got));
                        if (r <= 0) break;
                        got += uint64_t(r);
                    }
                    std::unique_lock<std::mutex> l(mu);
                    if (got != c.buf.size()) {
                        err = true;
                        break;
                    }
                    cv.wait(l, [&] { return q.size() < 4; });
                    q.push_back(std::move(c));
                    cv.notify_all();
                }
            }
            for (auto &kv : fds) ::close(kv.second);
            std::lock_guard<std::mutex> l(mu);
            done = true;
            cv.notify_all();
        });
        uint64_t put = 0, bad = 0;
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
            if (R.api.model_put(R.e, R.model, c.tensor, c.offset, c.buf.data(), c.buf.size()) != CERE_OK) ++bad;
            put += c.buf.size();
        }
        reader.join();
        row("upload", !err && bad == 0 && put == total,
            std::to_string(put) + " B, " + std::to_string(ms(t0, Clock::now()) / 1000) + " s");
    }
    t0 = Clock::now();
    st = R.api.model_seal(R.e, R.model);
    row("model_seal", st == CERE_OK, std::to_string(ms(t0, Clock::now())) + " ms");
    if (st != CERE_OK) return 1;
    R.stamp(&R.caps);
    st = R.api.capabilities(R.e, R.model, &R.caps);
    std::printf("capabilities: status %d execution_mode %u operations 0x%llx native_sampler %u native_token_mask %u "
                "native_speculative_batch %u speculative_draft_tokens %u native_decode_batch %u max_context %llu\n",
                st, R.caps.execution_mode, (unsigned long long)R.caps.operations, R.caps.native_sampler,
                R.caps.native_token_mask, R.caps.native_speculative_batch, R.caps.speculative_draft_tokens,
                R.caps.native_decode_batch, (unsigned long long)R.caps.max_context_tokens);
    row("capabilities: native, prefill + decode + verify + cancel, speculative 1 / 4",
        st == CERE_OK && R.caps.execution_mode == CERE_EXECUTION_MODE_NATIVE &&
            (R.caps.operations & CERE_OPS_PREFILL) && (R.caps.operations & CERE_OPS_DECODE) &&
            (R.caps.operations & CERE_OPS_VERIFY) && (R.caps.operations & CERE_OPS_CANCEL) &&
            R.caps.native_speculative_batch == 1 && R.caps.speculative_draft_tokens == 4);

    FILE *tr = std::fopen(trace_path.c_str(), "w");
    std::fprintf(tr, "# spec_s3 trace lib=%s abi=%u n=%llu\n", lib.c_str(), R.abi, (unsigned long long)n_new);
    std::map<std::string, Gen> plain_of;
    std::printf("\n%-10s %6s %5s | %9s %9s %7s | %6s %8s %8s %6s | %s\n", "prompt", "prompt", "N", "plain t/s",
                "spec t/s", "speedup", "rounds", "drafted*", "accepted", "acc/rd", "commit-per-round hist 1/2/3/4");
    for (const Prompt &p : prompts) {
        const uint64_t N = n_of.count(p.name) ? n_of[p.name] : n_new;
        GenArgs ga;
        ga.prompt = &p.ids;
        ga.n_new = N;
        ga.flags = CERE_WORK_NO_SPECULATION;
#if defined(FS3_STATIC)
        // Per-prompt GPU split (forward stats deltas): the plain run's prefill, the speculating run's prefill and its
        // MTP extends (extend + decode steps; by op family with CERE_VK_FORWARD_PROFILE=1).
        const auto &fstats = static_cast<cere::vkdev::VulkanModel &>(*R.model->device).forward_stats();
        const cere::vkdev::VkForwardStats f0 = fstats;
#endif
        const Gen plain = R.generate(ga);
        plain_of[p.name] = plain;
#if defined(FS3_STATIC)
        const cere::vkdev::VkForwardStats f1 = fstats;
#endif
        ga.flags = 0;
        const Gen spec = R.generate(ga);
#if defined(FS3_STATIC)
        {
            const cere::vkdev::VkForwardStats &f2 = fstats;
            std::printf("  gpu %s: plain prefill %llu calls gpu %.1f ms wall %.1f ms | spec prefill %llu calls gpu %.1f ms "
                        "wall %.1f ms, extends %llu rows %llu gpu %.1f ms wall %.1f ms (max %.1f)\n",
                        p.name.c_str(), (unsigned long long)(f1.prefill_calls - f0.prefill_calls),
                        f1.prefill_gpu_ms - f0.prefill_gpu_ms, f1.prefill_wall_ms - f0.prefill_wall_ms,
                        (unsigned long long)(f2.prefill_calls - f1.prefill_calls), f2.prefill_gpu_ms - f1.prefill_gpu_ms,
                        f2.prefill_wall_ms - f1.prefill_wall_ms, (unsigned long long)(f2.spec.extends - f1.spec.extends),
                        (unsigned long long)(f2.spec.extend_rows - f1.spec.extend_rows),
                        f2.spec.extend_gpu_ms - f1.spec.extend_gpu_ms, f2.spec.extend_wall_ms - f1.spec.extend_wall_ms,
                        f2.spec.max_extend_gpu_ms);
            // By op family (the static build with taps, CERE_VK_FORWARD_PROFILE=1): the plain prefill, the speculating
            // prefill, the extends.
            const auto by_family = [&](const char *what, const double *a, const double *b) {
                std::string fam;
                for (uint32_t i = 0; i < uint32_t(cere::vkdev::VkProf::kCount); ++i) {
                    const double d = b[i] - a[i];
                    if (d <= 0) continue;
                    char s[64];
                    std::snprintf(s, sizeof s, " %s=%.2f", cere::vkdev::vk_prof_name(cere::vkdev::VkProf(i)), d);
                    fam += s;
                }
                if (!fam.empty()) std::printf("  %s profile %s (ms):%s\n", what, p.name.c_str(), fam.c_str());
            };
            by_family("plain prefill", f0.prof_ms, f1.prof_ms);
            by_family("spec prefill", f1.prof_ms, f2.prof_ms);
            by_family("extend", f1.spec.extend_prof_ms, f2.spec.extend_prof_ms);
        }
#endif
        const Gen spec2 = quick ? spec : R.generate(ga);
        trace_gen(tr, "plain:" + p.name, plain);
        trace_gen(tr, "spec:" + p.name, spec);
        if (!quick) trace_gen(tr, "spec2:" + p.name, spec2);
        std::fprintf(tr, "timing %s prompt %zu n %llu plain_ttft_ms %.3f plain_decode_ms %.3f spec_ttft_ms %.3f "
                         "spec_decode_ms %.3f verify_steps %llu accepted %llu\n",
                     p.name.c_str(), p.ids.size(), (unsigned long long)N, plain.ttft_ms, plain.decode_ms, spec.ttft_ms,
                     spec.decode_ms, (unsigned long long)spec.verify_steps, (unsigned long long)spec.accepted);
        const bool ran = plain.status == CERE_OK && spec.status == CERE_OK && plain.tokens.size() == N &&
                         spec.tokens.size() == N;
        const long d = first_diff(plain, spec);
        std::string detail = "plain " + std::to_string(plain.status) + "/" + std::to_string(plain.tokens.size()) +
                             " spec " + std::to_string(spec.status) + "/" + std::to_string(spec.tokens.size());
        if (d >= 0 && size_t(d) < std::min(plain.tokens.size(), spec.tokens.size())) {
            detail += "; FIRST MISMATCH step " + std::to_string(d) + ": plain " + score_str(plain.scores[size_t(d)]) +
                      " spec " + score_str(spec.scores[size_t(d)]);
        }
        row("LOSSLESS " + p.name + " (" + std::to_string(p.ids.size()) + " + " + std::to_string(N) +
                "): spec ids + top-2 logits == plain, bit for bit",
            ran && d < 0, detail);
        if (!quick) {
            row("determinism " + p.name + ": spec run x2 identical (ids, scores, rounds)",
                spec2.status == CERE_OK && first_diff(spec, spec2) < 0 && spec.round_committed == spec2.round_committed);
        }
        // Acceptance: one speculation round per poll after the first token (rounds committing 1..4).
        uint64_t hist[5] = {};
        for (uint32_t c : spec.round_committed) hist[std::min<uint32_t>(c, 4)]++;
        const uint64_t rounds = spec.verify_steps;
        std::printf("%-10s %6zu %5llu | %9.2f %9.2f %7.3f | %6llu %8llu %8llu %6.3f | %llu/%llu/%llu/%llu\n",
                    p.name.c_str(), p.ids.size(), (unsigned long long)N, tok_s(plain), tok_s(spec),
                    tok_s(plain) > 0 ? tok_s(spec) / tok_s(plain) : 0, (unsigned long long)rounds,
                    (unsigned long long)(3 * rounds), (unsigned long long)spec.accepted,
                    rounds ? double(spec.accepted) / double(rounds) : 0.0, (unsigned long long)hist[1],
                    (unsigned long long)hist[2], (unsigned long long)hist[3], (unsigned long long)hist[4]);
        std::printf("  %s: plain TTFT %.1f ms, spec TTFT %.1f ms (prefill + MTP extend), plain polls %llu, spec polls "
                    "%llu\n",
                    p.name.c_str(), plain.ttft_ms, spec.ttft_ms, (unsigned long long)plain.polls,
                    (unsigned long long)spec.polls);
    }
    // * drafted: 3 MTP proposals per round (speculative_draft_tokens 4 - 1), fewer only at the budget's end.

    const Prompt *cap = nullptr, *code = nullptr, *expl = nullptr;
    for (const Prompt &q : prompts) {
        if (q.name == "capital") cap = &q;
        if (q.name == "code") code = &q;
        if (q.name == "explain") expl = &q;
    }
    // ---- stop token inside a chain: the capital prompt with <|im_end|> (248046) and <|endoftext|> (248044) ----
    if (cap != nullptr) {
        GenArgs ga;
        ga.prompt = &cap->ids;
        ga.n_new = 64;
        ga.stop = {248046, 248044};
        ga.flags = CERE_WORK_NO_SPECULATION;
        const Gen plain = R.generate(ga);
        ga.flags = 0;
        const Gen spec = R.generate(ga);
        trace_gen(tr, "stop-plain:capital", plain);
        trace_gen(tr, "stop-spec:capital", spec);
        std::string last;
        for (size_t r = 0; r < spec.round_committed.size(); ++r) last += std::to_string(spec.round_committed[r]) + " ";
        row("stop token inside a chain (capital): same tokens, finish STOP, ends at the stop",
            plain.status == CERE_OK && spec.status == CERE_OK && plain.finish == CERE_FINISH_STOP_TOKEN &&
                spec.finish == CERE_FINISH_STOP_TOKEN && first_diff(plain, spec) < 0 && !spec.tokens.empty() &&
                (spec.tokens.back() == 248046 || spec.tokens.back() == 248044),
            "plain " + std::to_string(plain.tokens.size()) + " tokens finish " + std::to_string(plain.finish) +
                ", spec " + std::to_string(spec.tokens.size()) + " finish " + std::to_string(spec.finish) +
                ", rounds committed: " + last);
    }
    // ---- the phase-S path at temperature 0 (sampler, no penalties) == greedy, and logprobs ----
    const Prompt *sp_prompt = expl != nullptr ? expl : (prompts.empty() ? nullptr : &prompts[0]);
    if (sp_prompt != nullptr && R.caps.native_sampler) {
        const Gen &g0 = plain_of[sp_prompt->name];
        cere_sampling_params sp{};
        R.stamp(&sp);
        sp.temperature = 0.0f;
        sp.top_p = 1.0f;
        sp.top_k = 248320;
        sp.repetition_penalty = 1.0f;
        sp.seed = 1234;
        GenArgs ga;
        ga.prompt = &sp_prompt->ids;
        ga.n_new = g0.tokens.size();
        ga.sampling = &sp;
        ga.top_logprobs = 5;
        ga.logprobs = true;
        const Gen gz = R.generate(ga);
        bool lp_ok = gz.status == CERE_OK && gz.logprobs.size() == gz.tokens.size() && !gz.tokens.empty();
        for (size_t k = 0; k < gz.logprobs.size() && lp_ok; ++k) {
            lp_ok = gz.logprobs[k].id == gz.tokens[k] && gz.logprobs[k].top_count == 5 &&
                    gz.logprobs[k].top[0].id == gz.tokens[k];
        }
        row("spec sampler T 0 (phase S) ids == plain greedy ids (" + sp_prompt->name + ")",
            gz.status == CERE_OK && gz.tokens == g0.tokens,
            "committed " + std::to_string(gz.tokens.size()) + ", rounds " + std::to_string(gz.verify_steps) +
                ", accepted " + std::to_string(gz.accepted));
        row("spec sampler T 0: logprobs top[0] == committed", lp_ok);
        // Seeded sampled speculation, twice; the plain seeded run for the record (not a contract: D6 keeps the
        // distribution, not the draws).
        cere_sampling_params ss = sp;
        ss.temperature = 1.0f;
        ss.top_p = 0.9f;
        ss.top_k = 20;
        ss.seed = 20261115;
        ga.sampling = &ss;
        ga.top_logprobs = 20;
        ga.n_new = n_new;
        const Gen a = R.generate(ga);
        const Gen b = R.generate(ga);
        ga.flags = CERE_WORK_NO_SPECULATION;
        const Gen pl = R.generate(ga);
        bool same = a.status == CERE_OK && b.status == CERE_OK && a.tokens == b.tokens && a.tokens.size() == n_new;
        for (size_t k = 0; k < a.scores.size() && same; ++k) {
            same = std::memcmp(&a.scores[k], &b.scores[k], sizeof(cere_token_score)) == 0 &&
                   std::memcmp(&a.logprobs[k], &b.logprobs[k], sizeof(cere_token_logprobs)) == 0;
        }
        row("spec sampled (T 1, top_k 20, top_p 0.9, seeded) x2 identical (ids, scores, logprobs)", same,
            "rounds " + std::to_string(a.verify_steps) + " accepted " + std::to_string(a.accepted));
        uint64_t in_topk = 0, in_nucleus = 0, differs_from_greedy = 0, same_as_plain = 0;
        for (size_t k = 0; k < a.tokens.size(); ++k) {
            const cere_token_logprobs &lp = a.logprobs[k];
            double cum = 0;
            bool found = false;
            for (uint32_t j = 0; j < lp.top_count; ++j) {
                if (lp.top[j].id == a.tokens[k]) {
                    found = true;
                    break;
                }
                cum += std::exp(double(lp.top[j].logprob));
            }
            in_topk += found;
            in_nucleus += found && cum < double(ss.top_p);
            differs_from_greedy += lp.top[0].id != a.tokens[k];
            same_as_plain += k < pl.tokens.size() && pl.tokens[k] == a.tokens[k];
        }
        row("spec sampled: every draw inside top_k 20 and the top_p 0.9 nucleus",
            !a.tokens.empty() && in_topk == a.tokens.size() && in_nucleus == a.tokens.size(),
            std::to_string(in_topk) + "/" + std::to_string(a.tokens.size()) + " in top-k, " +
                std::to_string(in_nucleus) + " in nucleus, " + std::to_string(differs_from_greedy) +
                " != argmax; plain seeded run: " + std::to_string(same_as_plain) + " positions equal (no contract)");
        for (size_t k = 0; k < a.tokens.size(); ++k) {
            std::fprintf(tr, "sampled spec:%s %zu %u %.9g\n", sp_prompt->name.c_str(), k, a.tokens[k],
                         double(a.logprobs[k].logprob));
        }
    }
    // ---- cancel after a few rounds, then a DECODE continuation on the same session (no drafts per round) ----
    if (code != nullptr && plain_of.count("code")) {
        const Gen &pl = plain_of["code"];
        GenArgs ga;
        ga.prompt = &code->ids;
        ga.n_new = pl.tokens.size();
        ga.keep_session = true;
        ga.cancel_after_polls = 4;  // the prefill poll + 3 speculation rounds, then cancel
        const Gen a = R.generate(ga);
        const size_t c = a.tokens.size();
        GenArgs gb;
        gb.operation = CERE_OP_DECODE;
        gb.session = a.session;
        gb.revision = a.revision;
        gb.n_new = pl.tokens.size() > c ? pl.tokens.size() - c : 0;
        gb.flags = CERE_WORK_NO_SPECULATION;
        const Gen b = gb.n_new > 0 && a.session != nullptr ? R.generate(gb) : Gen{};
        bool ok_prefix = a.finish == CERE_FINISH_CANCELLED && c > 0 && c < pl.tokens.size();
        Gen joined = a;
        joined.tokens.insert(joined.tokens.end(), b.tokens.begin(), b.tokens.end());
        joined.scores.insert(joined.scores.end(), b.scores.begin(), b.scores.end());
        const long d = first_diff(pl, joined);
        row("cancel after 3 rounds (code): CANCELLED with a committed prefix == plain's",
            ok_prefix && first_diff(pl, a) == long(c), "finish " + std::to_string(a.finish) + ", committed " +
                                                           std::to_string(c) + ", status " + std::to_string(a.status));
        row("cancel: DECODE continuation (NO_SPECULATION rounds) == the plain run, ids + scores",
            b.status == CERE_OK && joined.tokens.size() == pl.tokens.size() && d < 0,
            "continued " + std::to_string(b.tokens.size()) + " (status " + std::to_string(b.status) + ", submit " +
                std::to_string(b.submit_status) + "), rounds " + std::to_string(b.verify_steps) +
                (d >= 0 ? ", first mismatch " + std::to_string(d) : ""));
        trace_gen(tr, "cancel:code", joined);
        // VERIFY with caller drafts on the continued session: the plain run's next tokens as drafts (all accepted).
        if (a.session != nullptr && b.status == CERE_OK) {
            const std::vector<uint32_t> none;
            GenArgs gv;
            gv.operation = CERE_OP_VERIFY;
            gv.session = a.session;
            gv.revision = b.revision;
            std::vector<uint32_t> drafts = {pl.tokens.back(), 0, 0};
            gv.prompt = &drafts;
            gv.n_new = 4;
            const Gen v = R.generate(gv);
            row("VERIFY with 3 caller drafts on a drafting session: one round, 1..4 committed",
                v.status == CERE_OK && v.verify_steps == 1 && !v.tokens.empty() && v.tokens.size() <= 4,
                "status " + std::to_string(v.status) + " submit " + std::to_string(v.submit_status) + " committed " +
                    std::to_string(v.tokens.size()) + " accepted " + std::to_string(v.accepted) + " finish " +
                    std::to_string(v.finish));
        }
        if (a.session != nullptr) R.api.session_release(R.e, a.session);
    }
    // ---- negative: VERIFY on a session that does not draft (its PREFILL had NO_SPECULATION) ----
    if (cap != nullptr) {
        GenArgs ga;
        ga.prompt = &cap->ids;
        ga.n_new = 1;
        ga.flags = CERE_WORK_NO_SPECULATION;
        ga.keep_session = true;
        const Gen a = R.generate(ga);
        std::vector<uint32_t> drafts = {1, 2};
        GenArgs gv;
        gv.operation = CERE_OP_VERIFY;
        gv.session = a.session;
        gv.revision = a.revision;
        gv.prompt = &drafts;
        gv.n_new = 3;
        const Gen v = a.session != nullptr ? R.generate(gv) : Gen{};
        row("negative: VERIFY on a non-drafting session refused (ARGUMENT)", v.submit_status == CERE_ERR_ARGUMENT,
            "submit status " + std::to_string(v.submit_status));
        if (a.session != nullptr) R.api.session_release(R.e, a.session);
    }

#if defined(FS3_STATIC)
    {
        using namespace cere::vkdev;
        auto &vd = static_cast<VulkanDevice &>(*R.e->device);
        auto &vm = static_cast<VulkanModel &>(*R.model->device);
        // Device rows (host/device.h contracts, cuda_device.cpp's statuses).
        std::unique_ptr<cere::host::DeviceSession> ds;
        cere_status cs = vd.create_session(vm, 32768, false, &ds);
        if (cs == CERE_OK) {
            cere::host::SpecRowArgs ra;
            ra.session = ds.get();
            ra.token = 1;
            ra.draft_count = 0;
            ra.max_commit = 4;
            cere::host::SpecRowResult res;
            cere_status rs = CERE_OK;
            const cere_status call_nodraft = vd.speculate_rows(vm, &ra, 1, &res, &rs);
            row("device: speculate_rows on a non-drafting session: row ARGUMENT, nothing run",
                call_nodraft == CERE_OK && rs == CERE_ERR_ARGUMENT, "call " + std::to_string(call_nodraft) + " row " +
                                                                         std::to_string(rs));
            row("device: set_session_draft(on) before the prompt OK", vd.set_session_draft(vm, *ds, true) == CERE_OK);
            rs = CERE_OK;
            const cere_status call_pre = vd.speculate_rows(vm, &ra, 1, &res, &rs);
            row("device: speculate_rows before prefill (position 0): row ARGUMENT, position unchanged",
                call_pre == CERE_OK && rs == CERE_ERR_ARGUMENT && vd.session_position(*ds) == 0,
                "call " + std::to_string(call_pre) + " row " + std::to_string(rs));
            row("device: speculate_rows(null rows) ARGUMENT", vd.speculate_rows(vm, nullptr, 1, &res, &rs) ==
                                                                  CERE_ERR_ARGUMENT);
            row("device: speculate_rows(row_count 2 > max_sessions) ARGUMENT",
                vd.speculate_rows(vm, &ra, 2, &res, &rs) == CERE_ERR_ARGUMENT);
            // A prompt step, then set_session_draft is refused (only before the first step).
            uint64_t consumed = 0;
            cere_token_score sc{};
            const uint32_t prompt[3] = {248045, 846, 198};
            const cere_status ps = vd.step(vm, *ds, prompt, 3, 0, nullptr, &consumed, nullptr, &sc, nullptr, nullptr,
                                           nullptr);
            uint32_t pend[4] = {};
            const uint32_t np = vd.pending_drafts(*ds, pend, 4);
            row("device: drafting prompt step runs the MTP extend: 3 pending proposals", ps == CERE_OK && np == 3,
                "step " + std::to_string(ps) + ", pending " + std::to_string(np));
            row("device: set_session_draft after the prompt ARGUMENT",
                vd.set_session_draft(vm, *ds, false) == CERE_ERR_ARGUMENT);
            // draft_count > max_chain_drafts (3): ARGUMENT before anything runs.
            ra.token = sc.id;
            ra.draft_count = 4;
            ra.max_commit = 8;
            uint32_t four[4] = {1, 2, 3, 4};
            ra.drafts = four;
            rs = CERE_OK;
            const cere_status call4 = vd.speculate_rows(vm, &ra, 1, &res, &rs);
            row("device: 4 caller drafts (> max_chain_drafts 3) ARGUMENT, position unchanged",
                call4 == CERE_OK && rs == CERE_ERR_ARGUMENT && vd.session_position(*ds) == 3,
                "row " + std::to_string(rs) + " max_chain_drafts " + std::to_string(vd.max_chain_drafts(vm)));
            // A constrained row with the caller's drafts (grammar drafts, GS-02): runs since P4-F, one mask per verify
            // row (here 3 all-ones masks: the row commits 1..3 tokens).
            cere::host::ItemSampling smp;
            smp.sampled = true;
            smp.params.temperature = 1.0f;
            smp.params.top_k = 20;
            smp.params.top_p = 1.0f;
            std::vector<uint32_t> mask((248320 + 31) / 32 * 4, 0xffffffffu);
            smp.vocab_mask = mask.data();
            ra.sampling = &smp;
            ra.draft_count = 2;
            rs = CERE_OK;
            const cere_status callm = vd.speculate_rows(vm, &ra, 1, &res, &rs);
            row("device: constrained row with 2 caller drafts (grammar drafts, P4-F) runs: 1..3 committed",
                callm == CERE_OK && rs == CERE_OK && res.committed >= 1 && res.committed <= 3,
                "row " + std::to_string(rs) + " committed " + std::to_string(res.committed));
            // More than CERE_MAX_CONSTRAINED_DRAFTS, or a constrained row with drafts but no caller ids: ARGUMENT.
            ra.drafts = nullptr;
            rs = CERE_OK;
            const cere_status calln = vd.speculate_rows(vm, &ra, 1, &res, &rs);
            row("device: constrained row with drafts but no caller ids ARGUMENT", calln == CERE_OK &&
                    rs == CERE_ERR_ARGUMENT, "row " + std::to_string(rs));
            ds.reset();
        } else {
            row("device: create_session", false, std::to_string(cs));
        }
        const VkForwardStats::Spec &p = vm.forward_stats().spec;
        const double r = p.rounds ? double(p.rounds) : 1;
        std::printf("spec stats: rounds %llu drafted %llu accepted %llu committed %llu hist(accepted 0/1/2/3) "
                    "%llu/%llu/%llu/%llu\n",
                    (unsigned long long)p.rounds, (unsigned long long)p.drafted, (unsigned long long)p.accepted,
                    (unsigned long long)p.committed, (unsigned long long)p.accepted_hist[0],
                    (unsigned long long)p.accepted_hist[1], (unsigned long long)p.accepted_hist[2],
                    (unsigned long long)p.accepted_hist[3]);
        std::printf("  per round means: verify gpu %.3f ms (record %.3f, wall %.3f) | commit+draft gpu %.3f ms "
                    "(restore %.3f, MTP draft %.3f; record %.3f, wall %.3f) | host accept %.4f ms\n",
                    p.verify_gpu_ms / r, p.verify_record_ms / r, p.verify_wall_ms / r, p.cd_gpu_ms / r,
                    p.restore_gpu_ms / r, p.draft_gpu_ms / r, p.cd_record_ms / r, p.cd_wall_ms / r,
                    p.host_accept_ms / r);
        std::printf("  max submit gpu: verify %.2f ms, commit+draft %.2f ms, prompt extend %.2f ms; extends %llu "
                    "(%llu rows, %.1f ms gpu); last verify %llu commands / %llu barriers, last commit+draft %llu / "
                    "%llu\n",
                    p.max_verify_gpu_ms, p.max_cd_gpu_ms, p.max_extend_gpu_ms, (unsigned long long)p.extends,
                    (unsigned long long)p.extend_rows, p.extend_gpu_ms, (unsigned long long)p.last_verify_commands,
                    (unsigned long long)p.last_verify_barriers, (unsigned long long)p.last_cd_commands,
                    (unsigned long long)p.last_cd_barriers);
        std::printf("  undo restores %llu (with a completed QSA key row %llu), verify rows on the K9 sparse path %llu\n",
                    (unsigned long long)p.restores, (unsigned long long)p.key_restores,
                    (unsigned long long)p.sparse_rows);
        const VkForwardStats &fs = vm.forward_stats();
        std::printf("  plain decode steps: %llu, gpu %.3f ms / record %.3f ms / wall %.3f ms per step\n",
                    (unsigned long long)fs.decode_calls, fs.decode_calls ? fs.decode_gpu_ms / fs.decode_calls : 0,
                    fs.decode_calls ? fs.decode_record_ms / fs.decode_calls : 0,
                    fs.decode_calls ? fs.decode_wall_ms / fs.decode_calls : 0);
    }
#endif
    std::fclose(tr);
    st = R.api.model_release(R.e, R.model);
    row("model_release", st == CERE_OK);
    row("executor_destroy", R.api.executor_destroy(R.e) == CERE_OK);
    std::printf("spec_s3: %d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
