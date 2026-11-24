// VulkanAmd P3 T5: the S3 forward harness. Loads the S3 FP4-Local artifact through the cere_* ABI of a given executor
// library, prefills fixed prompts (token ids from tests/data/forward_s3_prompts.txt) and greedy-decodes N tokens per
// prompt, recording per step the committed id and its cere_token_score (top-2 ids and logits); then the sampler rows
// (greedy-with-sampling == greedy, a seeded sampled run twice, top_k / top_p sanity from the logprobs). Writes a trace
// file that forward_s3_compare.py compares across libraries / runs.
//
// One source, two builds (executor/vulkan/tests/build_forward_s3.sh):
//   forward_s3         dlopen(LIB): any library of the ABI family. ABI v5 (this tree, the Vulkan library) and ABI v4
//                      (the amd-amd HIP library, 42 functions): the only layout difference on this path is
//                      cere_model_desc (v4: 128 B, `levers` where v5 has `selection`; v5: 136 B with `family` and
//                      `calibrate_windows`) and the abi_version stamped in every struct; v4 gets moe_gemm = k7 (its
//                      default p3b is CUDA-only), v5 the default (the build's MoE = K7 on Vulkan).
//   forward_s3_static  the Vulkan library's own objects linked in (LIB ignored), the forward compiled with
//                      CERE_VK_FORWARD_TAPS (per-layer hidden-state dump); adds the
//                      device-level rows: split prefill vs one call (VulkanDevice::step with `limit`), the per-layer tap
//                      dump of one prompt, the forward statistics (record / GPU ms, commands) and fallback counters.
// Usage: forward_s3 LIB ARTIFACT PROMPTS OUT_TRACE [N=64] [--no-sampler] [--prompts a,b] [--taps FILE]
//                   [--prefill-attention cuda_core|p4] [--k2-gemm cublaslt|tail_split] [--drafts]
// --prefill-attention (P4-E): the model's prefill_attention selection (ABI v5: cere_model_desc.selection; ABI v4: the
// levers block); default: not set (the library's default, cuda_core). On the Vulkan library p4 is the
// cooperative-matrix prefill attention. The static build prints per-prompt prefill GPU time, the longest submit, the
// attention flag and K2 split counters, and (env CERE_VK_FORWARD_PROFILE=1) the prefill GPU time by op family.
// --k2-gemm (P4-B, D7; ABI v5 only): the model's k2_gemm selection; tail_split = the Vulkan GEMM's SPLIT x rows (a
// library without that route refuses it at create). Default: not set (cublaslt: inexact x rows on the FP32 route).
// --drafts (VulkanPrefill B4 diagnostics): the model allocated WITH drafts as spec_s3 / the speculating profile load it
// (`mtp.` DEVICE, speculative_draft_tokens 4, the artifact's draft token map), every item submitted with
// CERE_WORK_NO_SPECULATION: the plain prefill / decode on a drafts model (its cost against the same run without).
// The rows that need a model without drafts (capabilities "no verify", the T6 negative rows) are skipped.
// Timings are wall clock around each cere_executor_poll (one round = one prefill chunk or one decode step).
// Memory: ~105 GB; check MemAvailable + GPUReclaim first and hold /tmp/cere-gpu.lock.
#include "cere_executor.h"
#include "host/support.h"
#include "vk_artifact.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
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
#include "kernels/k2_mgemv_vk.h"
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
    // D15 (optional: null when the library has none): the session's prefix-cache namespace.
    cere_status (*set_cache_namespace)(cere_executor *, cere_session *, const uint8_t *) = nullptr;
};

template <typename F>
bool sym(void *h, const char *name, F *out) {
    *out = reinterpret_cast<F>(dlsym(h, name));
    if (*out == nullptr) std::fprintf(stderr, "missing symbol %s\n", name);
    return *out != nullptr;
}

// ABI v4's cere_levers (the HIP library of the amd-amd checkout), restated: 10 u32 = 40 B.
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
    std::printf("%-64s %s%s%s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.empty() ? "" : "  ", detail.c_str());
    std::fflush(stdout);
    (ok ? g_pass : g_fail)++;
}

struct Gen {
    cere_status status = CERE_OK;
    uint32_t finish = 0;
    std::vector<uint32_t> tokens;
    std::vector<cere_token_score> scores;
    std::vector<cere_token_logprobs> logprobs;
    double ttft_ms = 0, decode_ms = 0, total_ms = 0, session_ms = 0;  // session_ms: session_create (outside TTFT)
    uint32_t cached = 0;  // cere_completion.cached_tokens (D15: prompt tokens taken from the prefix cache)
    std::vector<double> poll_ms;
    uint64_t polls = 0;
};

struct Runner {
    Api api{};
    uint32_t abi = 0;
    cere_executor *e = nullptr;
    cere_model *model = nullptr;
    cere_execution_capabilities caps{};
    uint32_t work_flags = 0;  // every item's cere_work.flags (--drafts: CERE_WORK_NO_SPECULATION)

    template <typename T>
    void stamp(T *x) {
        x->struct_size = sizeof(T);
        x->abi_version = abi;
    }

    // `ns` (D15): the session's prefix-cache namespace (16 B) before its first submit; null: none (no lookup).
    Gen generate(const std::vector<uint32_t> &prompt, uint64_t n_new, const cere_sampling_params *sp,
                 uint32_t top_logprobs, bool want_logprobs, const uint8_t *ns = nullptr) {
        Gen g;
        cere_session *s = nullptr;
        const auto ts = Clock::now();
        g.status = api.session_create(e, model, 32768, &s);
        g.session_ms = ms(ts, Clock::now());
        if (g.status != CERE_OK) return g;
        if (ns != nullptr) {
            g.status = api.set_cache_namespace != nullptr ? api.set_cache_namespace(e, s, ns) : CERE_ERR_UNSUPPORTED;
            if (g.status != CERE_OK) {
                api.session_release(e, s);
                return g;
            }
        }
        std::vector<uint32_t> out(n_new);
        std::vector<cere_token_score> scores(n_new);
        std::vector<cere_token_logprobs> lps(n_new);
        cere_work w{};
        stamp(&w);
        w.operation = CERE_OP_PREFILL;
        w.expected_worker_epoch = 1;
        w.session = s;
        w.tokens = prompt.data();
        w.token_count = prompt.size();
        w.max_new_tokens = n_new;
        w.flags = work_flags;
        w.out_tokens = out.data();
        w.out_capacity = n_new;
        w.out_scores = scores.data();
        w.sampling = sp;
        w.out_logprobs = want_logprobs ? lps.data() : nullptr;
        w.top_logprobs = want_logprobs ? top_logprobs : 0;
        cere_batch b{};
        stamp(&b);
        b.expected_worker_epoch = 1;
        b.capability_profile_generation = caps.profile_generation;
        b.items = &w;
        b.item_count = 1;
        cere_completion_ticket ticket{};
        const auto t0 = Clock::now();
        g.status = api.submit(e, &b, &ticket);
        if (g.status != CERE_OK) {
            api.session_release(e, s);
            return g;
        }
        Clock::time_point first{};
        bool have_first = false;
        for (;;) {
            cere_completion c{};
            stamp(&c);
            const auto tp = Clock::now();
            const cere_status ps = api.poll(e, ticket, &c);
            const auto tq = Clock::now();
            ++g.polls;
            g.poll_ms.push_back(ms(tp, tq));
            if (ps != CERE_OK && ps != CERE_PENDING) {
                g.status = ps;
                break;
            }
            if (!have_first && c.committed_tokens >= 1) {
                first = tq;
                have_first = true;
                g.ttft_ms = ms(t0, tq);
            }
            if (c.terminal) {
                g.status = c.status;
                g.finish = c.finish;
                const uint64_t k = c.committed_tokens;
                g.cached = c.cached_tokens;
                g.tokens.assign(out.begin(), out.begin() + k);
                g.scores.assign(scores.begin(), scores.begin() + k);
                if (want_logprobs) g.logprobs.assign(lps.begin(), lps.begin() + k);
                g.total_ms = ms(t0, tq);
                if (have_first) g.decode_ms = ms(first, tq);
                break;
            }
        }
        api.session_release(e, s);
        return g;
    }
};

uint32_t bf_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

#if defined(FS3_STATIC)
// ---- D8 screen (vulkan-amd design note D8): teacher-forced logits of every continuation row, per arm ------------
// Rows file: `<id> <ctx> <T> <T ids>` per line (scripts/live/amd/screen_d8.py tokenize). Arms: A = the default
// selection, A2 = A again, As = A with the PREFILL split at the largest multiple of 64 <= (ctx - 1) / 2, B =
// k2_gemm tail_split; ticket `floor` runs A, A2, As (+ free generation A, As), ticket `candidate` A, B (+ free
// generation B). Scored rows ctx - 1 .. T - 2 (row r predicts ids[r + 1]), all inside the last PREFILL call: the
// final mixer rows the call leaves in the workspace x lm_head by the m <= 16 K2 GEMV (BF16 out, the serving
// last-row path), row T - 1 computed too and compared with the call's own logits. One JSON line per row.
struct ScreenRow {
    std::string id;
    uint32_t ctx = 0;
    std::vector<uint32_t> ids;
};

bool read_screen_rows(const std::string &path, std::vector<ScreenRow> *out) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream s(line);
        ScreenRow r;
        uint64_t count = 0;
        s >> r.id >> r.ctx >> count;
        uint64_t v;
        while (s >> v) r.ids.push_back(uint32_t(v));
        if (r.ids.size() != count || r.ctx < 2 || uint64_t(r.ctx) + 1 >= r.ids.size()) return false;
        out->push_back(std::move(r));
    }
    return !out->empty();
}

struct ScreenArm {
    cere_status status = CERE_OK;
    std::string why;
    std::vector<uint16_t> logits;  // BF16 [rows, V]: rows ctx - 1 .. T - 1
    bool last_row_equal = false;   // row T - 1 == the call's own lm_head logits, bit for bit
    uint64_t scanned[2] = {0, 0}, flagged[2] = {0, 0}, split[2] = {0, 0};  // K2 counters: K3 up, K5 out_proj
    // Prefill attention: PREFILL calls on the coopmat (p4) / FP32 path, (token, KV head) pairs flagged to the exact
    // FP32 kernel by K6 dense / K9 sparse.
    uint64_t coopmat_calls = 0, fp32_calls = 0, attn_flagged[2] = {0, 0};
    double ms = 0;
};

float bf16_float(uint16_t x) {
    const uint32_t u = uint32_t(x) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

void log_softmax(const uint16_t *x, uint32_t V, std::vector<double> *lp) {
    lp->resize(V);
    double mx = -INFINITY;
    for (uint32_t i = 0; i < V; ++i) mx = std::max(mx, double(bf16_float(x[i])));
    double sum = 0;
    for (uint32_t i = 0; i < V; ++i) sum += std::exp(double(bf16_float(x[i])) - mx);
    const double lse = mx + std::log(sum);
    for (uint32_t i = 0; i < V; ++i) (*lp)[i] = double(bf16_float(x[i])) - lse;
}

uint32_t argmax_bf16(const uint16_t *x, uint32_t V) {  // lowest id on ties
    uint32_t best = 0;
    for (uint32_t i = 1; i < V; ++i) {
        if (bf16_float(x[i]) > bf16_float(x[best])) best = i;
    }
    return best;
}

class Screen {
  public:
    explicit Screen(Runner &r)
        : R(r), vd(static_cast<cere::vkdev::VulkanDevice &>(*r.e->device)),
          vm(static_cast<cere::vkdev::VulkanModel &>(*r.model->device)) {}

    // The arm's selection: `tail_split` -> k2_gemm tail_split, `p4` -> prefill_attention p4 (absent: the library
    // default of that slot).
    cere_status select(bool tail_split, bool p4) {
        cere_slot_selection sel{};
        R.stamp(&sel);
        sel.block_version = CERE_SLOT_BLOCK_VERSION;
        if (tail_split) {
            sel.choices[sel.count].slot = CERE_SLOT_K2_GEMM;
            sel.choices[sel.count].impl = CERE_LEVER_K2_TAIL_SPLIT;
            ++sel.count;
        }
        if (p4) {
            sel.choices[sel.count].slot = CERE_SLOT_PREFILL_ATTENTION;
            sel.choices[sel.count].impl = CERE_LEVER_PREFILL_P4;
            ++sel.count;
        }
        return cere_model_set_selection(R.e, R.model, &sel);
    }

    void score(const ScreenRow &r, uint64_t split, ScreenArm *a) {
        using namespace cere::vkdev;
        const auto t0 = Clock::now();
        const VkWorkspace &ws = vm.workspace();
        const uint32_t V = vm.weights().config.vocab_size, H = vm.weights().config.hidden_size;
        const uint64_t T = r.ids.size(), first = r.ctx - 1, rows = T - first;
        cere_session *s = nullptr;
        a->status = R.api.session_create(R.e, R.model, 32768, &s);
        if (a->status != CERE_OK) {
            a->why = "session_create";
            return;
        }
        const VkForwardStats f0 = vm.forward_stats();
        uint64_t start = 0;
        a->status = prefill(s, r.ids.data(), T, split, &start);
        const VkForwardStats &f1 = vm.forward_stats();
        const uint32_t cons[2] = {uint32_t(VkGemmConsumer::kHcUp), uint32_t(VkGemmConsumer::kGdnOut)};
        for (int i = 0; i < 2; ++i) {
            a->scanned[i] = f1.fallback[cons[i]].rows - f0.fallback[cons[i]].rows;
            a->flagged[i] = f1.fallback[cons[i]].flagged - f0.fallback[cons[i]].flagged;
            a->split[i] = f1.fallback[cons[i]].split - f0.fallback[cons[i]].split;
        }
        a->coopmat_calls = f1.prefill_coopmat_calls - f0.prefill_coopmat_calls;
        a->fp32_calls = f1.prefill_fp32_calls - f0.prefill_fp32_calls;
        for (uint32_t p = 0; p < uint32_t(cere::vkdev::VkAttnPath::kCount); ++p)
            a->attn_flagged[p] = f1.attn_flags[p].flagged_pairs - f0.attn_flags[p].flagged_pairs;
        uint8_t *y = vm.host_of(ws.logits);
        std::vector<uint16_t> own;
        if (a->status != CERE_OK) {
            a->why = "prefill";
        } else if (start > first) {
            a->status = CERE_ERR_INTERNAL;
            a->why = "scored rows straddle two PREFILL calls";
        } else if (y == nullptr) {
            a->status = CERE_ERR_INTERNAL;
            a->why = "workspace logits not host-mapped";
        } else {
            own.resize(V);
            std::memcpy(own.data(), y, size_t(V) * 2);  // the call's own last-row logits
        }
        // The serving last-row lm_head exactly (vulkan_forward.cpp kLmHead: m = 1 GEMV into ws.logits, BF16 out),
        // once per row: the call's rows stay in ws.mixed until the next call.
        cere::vk::Context &ctx = vd.ctx();
        if (a->status == CERE_OK) a->logits.resize(rows * V);
        for (uint64_t i = 0; a->status == CERE_OK && i < rows; ++i) {
            const uint64_t x = ws.mixed + (first + i - start) * H * 2;
            cere_status st = ctx.begin();
            if (st == CERE_OK) st = cere::vkk2::gemv(ctx, x, vm.weights().lm_head, 0, ws.logits, 1, V, H, false);
            if (st == CERE_OK) st = ctx.barrier();
            if (st == CERE_OK) st = ctx.submit_wait();
            if (st != CERE_OK) {
                a->status = st;
                a->why = "lm_head gemv";
                break;
            }
            std::memcpy(a->logits.data() + i * V, y, size_t(V) * 2);
        }
        a->last_row_equal =
            a->status == CERE_OK && std::memcmp(a->logits.data() + (rows - 1) * V, own.data(), size_t(V) * 2) == 0;
        R.api.session_release(R.e, s);
        a->ms = ms(t0, Clock::now());
    }

    // Greedy free generation of `n` tokens after ids[0 .. ctx) through the device step (no stop token).
    cere_status generate(const ScreenRow &r, uint64_t split, uint32_t n, std::vector<uint32_t> *out) {
        cere_session *s = nullptr;
        cere_status st = R.api.session_create(R.e, R.model, 32768, &s);
        if (st != CERE_OK) return st;
        uint64_t start = 0;
        st = prefill(s, r.ids.data(), r.ctx, split, &start);
        while (st == CERE_OK && out->size() < n) {
            out->push_back(last_.id);
            if (out->size() == n) break;
            uint64_t c = 0;
            const uint32_t next = last_.id;
            st = step(s, &next, 1, 0, &c, &last_);
        }
        R.api.session_release(R.e, s);
        return st;
    }

    uint32_t vocab() const { return vm.weights().config.vocab_size; }

  private:
    cere_status step(cere_session *s, const uint32_t *t, uint64_t count, uint64_t limit, uint64_t *consumed,
                     cere_token_score *out) {
        return vd.step(*R.model->device, *s->device, t, count, limit, nullptr, consumed, nullptr, out, nullptr, nullptr,
                       nullptr);
    }
    // PREFILL of ids[0 .. T) in calls of the device's chunk, the first ending at `split` when it is not 0; *start =
    // the last call's first position; last_ = the last call's score.
    cere_status prefill(cere_session *s, const uint32_t *ids, uint64_t T, uint64_t split, uint64_t *start) {
        uint64_t done = 0;
        while (done < T) {
            uint64_t c = 0;
            const cere_status st = step(s, ids + done, T - done, done == 0 ? split : 0, &c, &last_);
            if (st != CERE_OK) return st;
            if (c == 0) return CERE_ERR_INTERNAL;
            *start = done;
            done += c;
        }
        return CERE_OK;
    }

    Runner &R;
    cere::vkdev::VulkanDevice &vd;
    cere::vkdev::VulkanModel &vm;
    cere_token_score last_{};
};

uint64_t fnv1a(const void *p, size_t n) {
    const uint8_t *b = static_cast<const uint8_t *>(p);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// `lever` (D8): tail_split (the D8 screen: A = the library default, B = k2_gemm tail_split) or p4 (its own screen,
// after tail_split's PASS: A = the new default k2_gemm tail_split, B = A + prefill_attention p4).
int run_screen(Runner &R, const std::string &rows_path, const std::string &out_path, const std::string &ticket,
               const std::string &lever, uint32_t gen_n, uint64_t max_rows) {
    std::vector<ScreenRow> rows;
    if (!read_screen_rows(rows_path, &rows)) {
        std::fprintf(stderr, "cannot read screen rows %s\n", rows_path.c_str());
        return 2;
    }
    if (max_rows != 0 && rows.size() > max_rows) rows.resize(max_rows);
    FILE *out = std::fopen(out_path.c_str(), "w");
    if (out == nullptr) return 2;
    Screen S(R);
    const uint32_t V = S.vocab();
    const bool floor = ticket == "floor";
    const bool p4 = lever == "p4";  // A's k2_gemm: tail_split on the p4 screen, the library default on D8's
    const auto t_all = Clock::now();
    int bad = 0;
    for (const ScreenRow &r : rows) {
        const uint64_t T = r.ids.size(), n = T - r.ctx;  // scored rows
        const uint64_t split = uint64_t((r.ctx - 1) / 2) / 64 * 64;
        // A's logits stay for the row; every other arm is compared with A as soon as it ran, then freed (an arm's
        // logits are up to ~1 GB).
        ScreenArm A;
        std::string arms_json, summary;
        std::map<std::string, std::vector<uint32_t>> gen;
        std::map<std::string, cere_status> gen_st;
        const auto emit = [&](const std::string &name, ScreenArm &a, bool is_a) {
            const bool ok = a.status == CERE_OK;
            bad += !ok || !a.last_row_equal;
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                          "%s\"%s\":{\"status\":%d,\"why\":\"%s\",\"digest\":\"%016llx\",\"last_row_equal\":%s,"
                          "\"ms\":%.1f,\"hc_up\":[%llu,%llu,%llu],\"gdn_out\":[%llu,%llu,%llu],"
                          "\"attn\":[%llu,%llu,%llu,%llu]",
                          arms_json.empty() ? "" : ",", name.c_str(), a.status, a.why.c_str(),
                          ok ? (unsigned long long)fnv1a(a.logits.data(), size_t(n) * V * 2) : 0ull,
                          a.last_row_equal ? "true" : "false", a.ms, (unsigned long long)a.scanned[0],
                          (unsigned long long)a.flagged[0], (unsigned long long)a.split[0],
                          (unsigned long long)a.scanned[1], (unsigned long long)a.flagged[1],
                          (unsigned long long)a.split[1], (unsigned long long)a.coopmat_calls,
                          (unsigned long long)a.fp32_calls, (unsigned long long)a.attn_flagged[0],
                          (unsigned long long)a.attn_flagged[1]);
            arms_json += buf;
            summary += name + " status " + std::to_string(a.status) + " " + std::to_string(int(a.ms)) +
                       " ms last-row " + (a.last_row_equal ? "equal" : "DIFFERS") + "; ";
            if (!ok) {
                arms_json += "}";
                return;
            }
            std::vector<double> la, lx;
            std::string nll = ",\"nll\":[", kl = ",\"kl\":[", top1;
            const bool cmp = !is_a && A.status == CERE_OK;
            for (uint64_t i = 0; i < n; ++i) {
                const uint16_t *px = a.logits.data() + i * V;
                log_softmax(px, V, &lx);
                std::snprintf(buf, sizeof(buf), "%s%.6f", i ? "," : "", -lx[r.ids[r.ctx + i]]);
                nll += buf;
                if (!cmp) continue;
                const uint16_t *pa = A.logits.data() + i * V;
                log_softmax(pa, V, &la);
                double d = 0;
                for (uint32_t j = 0; j < V; ++j) d += std::exp(la[j]) * (la[j] - lx[j]);
                std::snprintf(buf, sizeof(buf), "%s%.9g", i ? "," : "", d);
                kl += buf;
                top1 += argmax_bf16(pa, V) == argmax_bf16(px, V) ? '1' : '0';
            }
            arms_json += nll + "]";
            if (cmp) {
                const bool equal = std::memcmp(a.logits.data(), A.logits.data(), size_t(n) * V * 2) == 0;
                arms_json += std::string(",\"equal_to_A\":") + (equal ? "true" : "false") + kl + "],\"top1\":\"" +
                             top1 + "\"";
            }
            arms_json += "}";
        };
        // An arm: `cand` false = A's selection, true = the candidate's (B).
        const auto run = [&](const std::string &name, bool cand, uint64_t sp, bool free_gen) {
            ScreenArm other;
            ScreenArm &a = name == "A" ? A : other;
            const cere_status sel = S.select(p4 || cand, p4 && cand);
            if (sel != CERE_OK) {
                a.status = sel;
                a.why = "set_selection";
            } else {
                S.score(r, sp, &a);
                if (free_gen) gen_st[name] = S.generate(r, sp, gen_n, &gen[name]);
            }
            emit(name, a, name == "A");
        };
        if (floor) {
            run("A", false, 0, true);
            run("A2", false, 0, false);
            run("As", false, split, true);
        } else {
            run("A", false, 0, false);
            run("B", true, 0, true);
        }
        if (S.select(false, false) != CERE_OK) ++bad;
        std::fprintf(out,
                     "{\"ticket\":\"%s\",\"lever\":\"%s\",\"id\":\"%s\",\"T\":%llu,\"ctx\":%u,\"split\":%llu,"
                     "\"arms\":{%s",
                     ticket.c_str(), lever.c_str(), r.id.c_str(), (unsigned long long)T, r.ctx,
                     (unsigned long long)(floor ? split : 0), arms_json.c_str());
        std::fprintf(out, "},\"gen\":{");
        bool first_arm = true;
        for (const auto &kv : gen) {
            std::fprintf(out, "%s\"%s\":{\"status\":%d,\"tokens\":[", first_arm ? "" : ",", kv.first.c_str(),
                         gen_st[kv.first]);
            for (size_t i = 0; i < kv.second.size(); ++i) std::fprintf(out, "%s%u", i ? "," : "", kv.second[i]);
            std::fprintf(out, "]}");
            first_arm = false;
            bad += gen_st[kv.first] != CERE_OK;
        }
        std::fprintf(out, "}}\n");
        std::fflush(out);
        std::printf("screen %s %s T %llu scored %llu: %selapsed %.1f s\n", ticket.c_str(), r.id.c_str(),
                    (unsigned long long)T, (unsigned long long)n, summary.c_str(), ms(t_all, Clock::now()) / 1000);
        std::fflush(stdout);
    }
    std::fclose(out);
    row("screen " + ticket + ": every call OK, every last-row lm_head check equal", bad == 0,
        std::to_string(rows.size()) + " rows, " + std::to_string(ms(t_all, Clock::now()) / 1000) + " s");
    return bad == 0 ? 0 : 1;
}
#endif

}  // namespace

int main(int argc, char **argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: forward_s3 LIB ARTIFACT PROMPTS OUT_TRACE [N] [--no-sampler] [--prompts a,b] "
                             "[--taps FILE] [--screen ROWS OUT floor|candidate [--screen-lever tail_split|p4] "
                             "[--screen-gen N] [--screen-max-rows N]] [--prefix-slots N]\n");
        return 2;
    }
    const std::string lib = argv[1], dir = argv[2], prompts_path = argv[3], trace_path = argv[4];
    uint64_t n_new = 64;
    bool sampler_rows = true, drafts = false;
    std::string only, taps_path, consistency, prefill_attention, k2_gemm;
    std::string screen_rows, screen_out, screen_ticket;  // D8 screen (static build): see run_screen
    std::string screen_lever = "tail_split";
    uint32_t screen_gen = 128;
    uint64_t screen_max_rows = 0;
    uint64_t prefix_slots = 0;  // D15 (static build): cere_model_desc.prefix_state_slots; > 0 runs the D15 rows
    for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--screen" && i + 3 < argc) {
            screen_rows = argv[++i];
            screen_out = argv[++i];
            screen_ticket = argv[++i];
            if (screen_ticket != "floor" && screen_ticket != "candidate") {
                std::fprintf(stderr, "--screen: ticket floor or candidate\n");
                return 2;
            }
#if !defined(FS3_STATIC)
            std::fprintf(stderr, "--screen: static build only (forward_s3_static)\n");
            return 2;
#endif
        } else if (a == "--screen-gen" && i + 1 < argc) {
            screen_gen = uint32_t(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--screen-max-rows" && i + 1 < argc) {
            screen_max_rows = std::strtoull(argv[++i], nullptr, 10);
        } else if (a == "--screen-lever" && i + 1 < argc) {
            screen_lever = argv[++i];
            if (screen_lever != "tail_split" && screen_lever != "p4") {
                std::fprintf(stderr, "--screen-lever: tail_split or p4\n");
                return 2;
            }
        } else if (a == "--no-sampler") {
            sampler_rows = false;
        } else if (a == "--prompts" && i + 1 < argc) {
            only = argv[++i];
        } else if (a == "--consistency" && i + 1 < argc) {
            consistency = argv[++i];
        } else if (a == "--taps" && i + 1 < argc) {
            taps_path = argv[++i];
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
        } else if (a == "--prefix-slots" && i + 1 < argc) {
            prefix_slots = std::strtoull(argv[++i], nullptr, 10);
#if !defined(FS3_STATIC)
            std::fprintf(stderr, "--prefix-slots: static build only (forward_s3_static)\n");
            return 2;
#endif
        } else if (a == "--drafts") {
            drafts = true;
        } else {
            n_new = std::strtoull(a.c_str(), nullptr, 10);
        }
    }
#if !defined(FS3_STATIC)
    (void)screen_gen;
    (void)screen_max_rows;
#endif
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

    Runner R;
#if defined(FS3_STATIC)
    (void)lib;
    R.api = Api{cere_executor_abi_version, cere_executor_create, cere_executor_destroy, cere_executor_capabilities,
                cere_model_create, cere_model_put, cere_model_seal, cere_model_release, cere_session_create,
                cere_session_release, cere_executor_submit, cere_executor_poll, cere_session_set_cache_namespace};
    std::printf("library: (static Vulkan objects, taps test build)\n");
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
              sym(h, "cere_executor_submit", &R.api.submit) && sym(h, "cere_executor_poll", &R.api.poll);
    if (!ok) return 2;
    std::printf("library: %s\n", lib.c_str());
#endif
    R.abi = R.api.abi_version();
    std::printf("abi_version: %u\n", R.abi);
    if (R.abi != 4 && R.abi != 5) {
        std::fprintf(stderr, "unsupported ABI %u\n", R.abi);
        return 2;
    }

    // ---- desc (vk_artifact.h: the runtime's model_spec order and placement, plain profile; --drafts: spec_s3's) ----
    std::vector<vktest::ArtTensor> art;
    std::string why;
    if (!vktest::artifact_tensors(dir, drafts ? 1 : 0, &art, &why)) {
        std::fprintf(stderr, "artifact: %s\n", why.c_str());
        return 2;
    }
    std::vector<uint32_t> draft_map;
    if (drafts) {
        std::ifstream f(dir + "/draft/token_map.u32le", std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        draft_map.resize(bytes.size() / 4);
        std::memcpy(draft_map.data(), bytes.data(), draft_map.size() * 4);
        R.work_flags = CERE_WORK_NO_SPECULATION;
        std::printf("drafts: speculative_draft_tokens 4, draft token map %zu ids, items CERE_WORK_NO_SPECULATION\n",
                    draft_map.size());
    }
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
    desc.config = &config;
    desc.prefix_state_slots = prefix_slots;  // D15: 0 (the default) keeps the cache off
    desc.tensors = tensors.data();
    desc.tensor_count = tensors.size();
    desc.max_sessions = 1;
    desc.max_context_tokens = 32768;
    desc.stop_token_ids = nullptr;  // no stop token: every prompt decodes exactly N tokens on both libraries
    desc.stop_token_count = 0;
    if (drafts) {
        desc.speculative_draft_tokens = 4;
        desc.draft_token_map = draft_map.data();
        desc.draft_token_map_count = draft_map.size();
    }
    const uint32_t pa = prefill_attention == "p4" ? CERE_LEVER_PREFILL_P4 : CERE_LEVER_PREFILL_CUDA_CORE;
    LeversV4 lv4{sizeof(LeversV4), 4, 0, pa, 0, CERE_LEVER_MOE_K7, 0, 0, 0, 0};
    cere_slot_selection sel{};
    sel.struct_size = sizeof sel;
    sel.abi_version = R.abi;
    sel.block_version = CERE_SLOT_BLOCK_VERSION;
    if (!k2_gemm.empty()) {
        sel.choices[sel.count].slot = CERE_SLOT_K2_GEMM;
        sel.choices[sel.count++].impl = k2_gemm == "tail_split" ? CERE_LEVER_K2_TAIL_SPLIT : CERE_LEVER_K2_CUBLASLT;
    }
    if (!prefill_attention.empty()) {
        sel.choices[sel.count].slot = CERE_SLOT_PREFILL_ATTENTION;
        sel.choices[sel.count++].impl = pa;
    }
    std::printf("k2_gemm: %s\n", k2_gemm.empty() ? "(library default)" : k2_gemm.c_str());
    std::printf("prefill_attention: %s\n", prefill_attention.empty() ? "(library default)" : prefill_attention.c_str());
    if (R.abi == 4) {
        if (!k2_gemm.empty()) {
            std::fprintf(stderr, "--k2-gemm: ABI v5 libraries only\n");
            return 2;
        }
        static_assert(offsetof(cere_model_desc, selection) == 120, "v4 levers sit where v5 has selection");
        const void *levers = &lv4;
        std::memcpy(reinterpret_cast<uint8_t *>(&desc) + offsetof(cere_model_desc, selection), &levers, sizeof levers);
    } else {
        desc.family = CERE_FAMILY_QWEN4_EXP;
        if (sel.count != 0) desc.selection = &sel;
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
    auto t0 = Clock::now();
    cere_status st = R.api.model_create(R.e, &desc, &R.model);
    row("model_create", st == CERE_OK, "status " + std::to_string(st) + ", " + std::to_string(ms(t0, Clock::now())) + " ms");
    if (st != CERE_OK) return 1;
#if defined(FS3_STATIC)
    {  // diagnostics: the blocks' device address ranges (FS3_ADDRS set: then exit)
        auto &vm = static_cast<cere::vkdev::VulkanModel &>(*R.model->device);
        for (size_t b = 0; b < vm.blocks().size(); ++b) {
            std::printf("block %2zu %-10s addr 0x%llx .. 0x%llx\n", b, cere::vkdev::vk_purpose_name(vm.layout().blocks[b].purpose),
                        (unsigned long long)vm.blocks()[b].addr,
                        (unsigned long long)(vm.blocks()[b].addr + vm.blocks()[b].size));
        }
        const auto &ws = vm.workspace();
        std::printf("scratch 0x%llx +%llu gemm_ws 0x%llx\n", (unsigned long long)ws.scratch,
                    (unsigned long long)ws.scratch_bytes, (unsigned long long)ws.gemm_ws);
        if (std::getenv("FS3_ADDRS") != nullptr) {
            R.api.model_release(R.e, R.model);
            R.api.executor_destroy(R.e);
            return 0;
        }
    }
#endif
    // ---- upload (a reader thread + puts), as load_s3 ----
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
                "native_speculative_batch %u native_decode_batch %u max_context %llu\n",
                st, R.caps.execution_mode, (unsigned long long)R.caps.operations, R.caps.native_sampler,
                R.caps.native_token_mask, R.caps.native_speculative_batch, R.caps.native_decode_batch,
                (unsigned long long)R.caps.max_context_tokens);
    row("capabilities: native, prefill + decode + cancel, no verify" + std::string(drafts ? " (--drafts: verify)" : ""),
        st == CERE_OK && R.caps.execution_mode == CERE_EXECUTION_MODE_NATIVE &&
            (R.caps.operations & (1u << CERE_OP_PREFILL)) && (R.caps.operations & (1u << CERE_OP_DECODE)) &&
            bool(R.caps.operations & (1u << CERE_OP_VERIFY)) == drafts);

    // T6 negative rows: speculation requested on a model allocated WITHOUT drafts (skipped with --drafts).
    if (!drafts && !prompts.empty()) {
        cere_session *s = nullptr;
        if (R.api.session_create(R.e, R.model, 32768, &s) == CERE_OK) {
            const uint32_t ids[2] = {1, 2};
            uint32_t out[3];
            cere_work w{};
            R.stamp(&w);
            w.operation = CERE_OP_VERIFY;
            w.expected_worker_epoch = 1;
            w.session = s;
            w.tokens = ids;
            w.token_count = 2;
            w.max_new_tokens = 3;
            w.out_tokens = out;
            w.out_capacity = 3;
            cere_batch b{};
            R.stamp(&b);
            b.expected_worker_epoch = 1;
            b.capability_profile_generation = R.caps.profile_generation;
            b.items = &w;
            b.item_count = 1;
            cere_completion_ticket ticket{};
            const cere_status vs = R.api.submit(R.e, &b, &ticket);
            row("negative: VERIFY on a model without drafts refused (UNSUPPORTED)", vs == CERE_ERR_UNSUPPORTED,
                "status " + std::to_string(vs));
#if defined(FS3_STATIC)
            {
                auto &vd = static_cast<cere::vkdev::VulkanDevice &>(*R.e->device);
                auto &vm = static_cast<cere::vkdev::VulkanModel &>(*R.model->device);
                row("negative: device draft_supported false on a model without drafts", !vd.draft_supported(vm));
                row("negative: set_session_draft(on) on a model without drafts UNSUPPORTED",
                    vd.set_session_draft(vm, *s->device, true) == CERE_ERR_UNSUPPORTED);
                cere::host::SpecRowArgs ra;
                ra.session = s->device.get();
                ra.max_commit = 1;
                cere::host::SpecRowResult res;
                cere_status rs = CERE_OK;
                row("negative: speculate_rows on a model without drafts UNSUPPORTED",
                    vd.speculate_rows(vm, &ra, 1, &res, &rs) == CERE_ERR_UNSUPPORTED);
            }
#endif
            R.api.session_release(R.e, s);
        }
    }

    FILE *tr = std::fopen(trace_path.c_str(), "w");
    std::fprintf(tr, "# forward_s3 trace lib=%s abi=%u n=%llu\n", lib.c_str(), R.abi, (unsigned long long)n_new);
    std::map<std::string, Gen> greedy;
    const bool prefill_only = std::getenv("FS3_PREFILL_ONLY") != nullptr;  // diagnostics: skip generation
#if defined(FS3_STATIC)
    auto &svm = static_cast<cere::vkdev::VulkanModel &>(*R.model->device);
    std::printf("device prefill attention path: %s (coopmat probe %s)\n", svm.prefill_attention_path(),
                svm.instantiate_report().coopmat_ok ? "passed" : svm.instantiate_report().coopmat_why.c_str());
    row("prefill attention path == the selection",
        std::string(svm.prefill_attention_path()) == (prefill_attention == "p4" ? "coopmat" : "fp32"),
        svm.prefill_attention_path());
    const size_t pipelines_at_seal = static_cast<cere::vkdev::VulkanDevice &>(*R.e->device).ctx().pipeline_count();
    if (!screen_rows.empty()) {
        std::fclose(tr);
        return run_screen(R, screen_rows, screen_out, screen_ticket, screen_lever, screen_gen, screen_max_rows);
    }
#endif
    // FS3_GAP_MS (diagnostic): idle the GPU this long before each prompt, as a server between requests does.
    const char *gap_env = std::getenv("FS3_GAP_MS");
    const long gap_ms = gap_env == nullptr ? 0 : std::atol(gap_env);
    for (const Prompt &p : prompts) {
        if (prefill_only) break;
        if (gap_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
#if defined(FS3_STATIC)
        const cere::vkdev::VkForwardStats f0 = svm.forward_stats();
#endif
        const Gen g = R.generate(p.ids, n_new, nullptr, 0, false);
#if defined(FS3_STATIC)
        {
            using namespace cere::vkdev;
            const VkForwardStats &f1 = svm.forward_stats();
            uint64_t fb = 0, sp = 0, rows = 0;
            for (uint32_t i = 0; i < uint32_t(VkGemmConsumer::kCount); ++i) {
                fb += f1.fallback[i].flagged - f0.fallback[i].flagged;
                sp += f1.fallback[i].split - f0.fallback[i].split;
                rows += f1.fallback[i].rows - f0.fallback[i].rows;
            }
            uint64_t ar = 0, af = 0, ap = 0;
            for (uint32_t i = 0; i < uint32_t(VkAttnPath::kCount); ++i) {
                ar += f1.attn_flags[i].rows - f0.attn_flags[i].rows;
                af += f1.attn_flags[i].flagged_rows - f0.attn_flags[i].flagged_rows;
                ap += f1.attn_flags[i].flagged_pairs - f0.attn_flags[i].flagged_pairs;
            }
            const uint64_t dc = f1.decode_calls - f0.decode_calls;
            std::printf("  gpu %s: prefill %llu calls, gpu %.1f ms, longest submit %.1f ms; decode gpu mean %.3f ms over "
                        "%llu; K2 x rows scanned %llu FB %llu split %llu; coopmat attention rows %llu flagged rows %llu "
                        "(pairs %llu)\n",
                        p.name.c_str(), (unsigned long long)(f1.prefill_calls - f0.prefill_calls),
                        f1.prefill_gpu_ms - f0.prefill_gpu_ms, f1.prefill_max_submit_gpu_ms,
                        dc ? (f1.decode_gpu_ms - f0.decode_gpu_ms) / double(dc) : 0.0, (unsigned long long)dc,
                        (unsigned long long)rows, (unsigned long long)fb, (unsigned long long)sp,
                        (unsigned long long)ar, (unsigned long long)af, (unsigned long long)ap);
            // P4-G: segment timing of this prompt's prefill (wall = record + submit + wait + read on the host)
            const VkSubmitBreakdown &b1 = f1.prefill_submits, &b0 = f0.prefill_submits;
            const uint64_t ns = b1.submits - b0.submits, ng = b1.gaps - b0.gaps, nc = b1.calibrated - b0.calibrated;
            std::printf("  submits %s: in flight %u, prefill wall %.1f ms - gpu %.1f ms = %.1f ms over %llu submits (%.2f ms "
                        "each); GPU idle between segments %.1f ms over %llu gaps (max %.2f); submit calls %.1f ms, wait calls "
                        "%.1f ms; calibrated %llu: start after submit %.3f ms, retire after end %.3f ms (means); host after the "
                        "last retire %.1f ms, counters summed at retires %.1f ms\n",
                        p.name.c_str(), f1.prefill_last_depth, f1.prefill_wall_ms - f0.prefill_wall_ms,
                        f1.prefill_gpu_ms - f0.prefill_gpu_ms,
                        (f1.prefill_wall_ms - f0.prefill_wall_ms) - (f1.prefill_gpu_ms - f0.prefill_gpu_ms),
                        (unsigned long long)ns,
                        ns ? ((f1.prefill_wall_ms - f0.prefill_wall_ms) - (f1.prefill_gpu_ms - f0.prefill_gpu_ms)) / double(ns) : 0.0,
                        b1.gap_ms - b0.gap_ms, (unsigned long long)ng, b1.max_gap_ms, b1.submit_call_ms - b0.submit_call_ms,
                        b1.wait_call_ms - b0.wait_call_ms, (unsigned long long)nc,
                        nc ? (b1.start_after_submit_ms - b0.start_after_submit_ms) / double(nc) : 0.0,
                        nc ? (b1.retire_after_end_ms - b0.retire_after_end_ms) / double(nc) : 0.0,
                        f1.prefill_read_ms - f0.prefill_read_ms, f1.prefill_count_ms - f0.prefill_count_ms);
        }
#endif
        greedy[p.name] = g;
        row("greedy " + p.name + " (" + std::to_string(p.ids.size()) + " tokens)",
            g.status == CERE_OK && g.tokens.size() == n_new,
            "status " + std::to_string(g.status) + " finish " + std::to_string(g.finish) + " committed " +
                std::to_string(g.tokens.size()));
        const double dec_tps = g.tokens.size() > 1 ? double(g.tokens.size() - 1) / (g.decode_ms / 1000) : 0;
        std::printf("  timing %s: prompt %zu, TTFT %.1f ms (%.0f prompt tok/s), decode %zu tok in %.1f ms = %.2f tok/s, "
                    "polls %llu, session_create %.1f ms\n",
                    p.name.c_str(), p.ids.size(), g.ttft_ms, double(p.ids.size()) / (g.ttft_ms / 1000),
                    g.tokens.size() ? g.tokens.size() - 1 : 0, g.decode_ms, dec_tps, (unsigned long long)g.polls,
                    g.session_ms);
        std::fprintf(tr, "timing %s prompt %zu ttft_ms %.3f decode_tokens %zu decode_ms %.3f\n", p.name.c_str(),
                     p.ids.size(), g.ttft_ms, g.tokens.size() ? g.tokens.size() - 1 : 0, g.decode_ms);
        for (size_t k = 0; k < g.tokens.size(); ++k) {
            const cere_token_score &s = g.scores[k];
            std::fprintf(tr, "step %s %zu %u %u %u %.9g %.9g %08x %08x\n", p.name.c_str(), k, g.tokens[k], s.id,
                         s.runner_up, double(s.logit), double(s.runner_up_logit), bf_bits(s.logit),
                         bf_bits(s.runner_up_logit));
        }
        std::fflush(tr);
    }

    // Path consistency on ONE library (--consistency name:k1,k2,..): the step-k score of the greedy decode run vs the
    // score of a fresh session that prefills prompt + the first k greedy tokens in one PREFILL call (the same position,
    // other kernels: GEMM classes, chunked GDN, prefill attention). Its spread is the library's own path noise, the
    // yardstick for the cross-device differences.
    if (!consistency.empty()) {
        const size_t colon = consistency.find(':');
        const std::string name = consistency.substr(0, colon);
        const Gen *g = greedy.count(name) ? &greedy[name] : nullptr;
        const Prompt *pp = nullptr;
        for (const Prompt &q : prompts) {
            if (q.name == name) pp = &q;
        }
        std::istringstream ks(consistency.substr(colon + 1));
        std::string kk;
        while (g != nullptr && pp != nullptr && std::getline(ks, kk, ',')) {
            const size_t k = std::strtoull(kk.c_str(), nullptr, 10);
            if (k == 0 || k >= g->tokens.size()) continue;
            std::vector<uint32_t> ids = pp->ids;
            ids.insert(ids.end(), g->tokens.begin(), g->tokens.begin() + k);
            const Gen c = R.generate(ids, 1, nullptr, 0, false);
            if (c.status != CERE_OK || c.scores.empty()) continue;
            const cere_token_score &d = g->scores[k], &q = c.scores[0];
            std::printf("consistency %s step %zu: decode top-2 %u/%u %.4f/%.4f | prefill(%zu) top-2 %u/%u %.4f/%.4f | "
                        "d top1 logit %.4f\n",
                        name.c_str(), k, d.id, d.runner_up, double(d.logit), double(d.runner_up_logit), ids.size(),
                        q.id, q.runner_up, double(q.logit), double(q.runner_up_logit), double(q.logit - d.logit));
            std::fprintf(tr, "consistency %s %zu %u %u %.9g %.9g %u %u %.9g %.9g\n", name.c_str(), k, d.id,
                         d.runner_up, double(d.logit), double(d.runner_up_logit), q.id, q.runner_up, double(q.logit),
                         double(q.runner_up_logit));
        }
    }

    if (sampler_rows && R.caps.native_sampler && !prompts.empty()) {
        // The "explain" prompt when present (an open-ended answer: the draws have entropy to show), else the first.
        const Prompt *pp = &prompts[0];
        for (const Prompt &q : prompts) {
            if (q.name == "explain") pp = &q;
        }
        const Prompt &p = *pp;
        const Gen &g0 = greedy[p.name];
        // (1) temperature 0 with processors (no penalties) and logprobs: the committed ids equal greedy's.
        cere_sampling_params sp{};
        R.stamp(&sp);
        sp.temperature = 0.0f;
        sp.top_p = 1.0f;
        sp.top_k = 248320;
        sp.repetition_penalty = 1.0f;
        sp.seed = 1234;
        const Gen gz = R.generate(p.ids, n_new, &sp, 5, true);
        row("sampler: temperature 0 == greedy ids (" + p.name + ")",
            gz.status == CERE_OK && gz.tokens == g0.tokens, "committed " + std::to_string(gz.tokens.size()));
        bool lp_ok = gz.status == CERE_OK && !gz.logprobs.empty();
        for (size_t k = 0; k < gz.logprobs.size() && lp_ok; ++k) {
            lp_ok = gz.logprobs[k].id == gz.tokens[k] && gz.logprobs[k].top_count == 5 &&
                    gz.logprobs[k].top[0].id == gz.tokens[k];
        }
        row("sampler: greedy logprobs top[0] == committed", lp_ok);
        // (2) seeded sampled, twice: identical; each draw inside top_k and the top_p nucleus (from its logprobs).
        cere_sampling_params ss = sp;
        ss.temperature = 1.0f;
        ss.top_p = 0.9f;
        ss.top_k = 20;
        ss.seed = 20261115;
        const Gen a = R.generate(p.ids, n_new, &ss, 20, true);
        const Gen b = R.generate(p.ids, n_new, &ss, 20, true);
        bool same = a.status == CERE_OK && b.status == CERE_OK && a.tokens == b.tokens && a.tokens.size() == n_new;
        for (size_t k = 0; k < a.scores.size() && same; ++k) {
            same = std::memcmp(&a.scores[k], &b.scores[k], sizeof(cere_token_score)) == 0 &&
                   std::memcmp(&a.logprobs[k], &b.logprobs[k], sizeof(cere_token_logprobs)) == 0;
        }
        row("sampler: seeded sampled run x2 identical (ids, scores, logprobs)", same);
        uint64_t in_topk = 0, in_nucleus = 0, differs_from_greedy = 0;
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
        }
        row("sampler: T 1.0: every draw inside top_k 20 and the top_p 0.9 nucleus",
            in_topk == a.tokens.size() && in_nucleus == a.tokens.size(),
            std::to_string(in_topk) + "/" + std::to_string(a.tokens.size()) + " in top-k, " +
                std::to_string(in_nucleus) + " in nucleus, " + std::to_string(differs_from_greedy) +
                " draws != the argmax");
        for (size_t k = 0; k < a.tokens.size(); ++k) {
            std::fprintf(tr, "sampled %s %zu %u %.9g\n", p.name.c_str(), k, a.tokens[k], double(a.logprobs[k].logprob));
        }
    }

#if defined(FS3_STATIC)
    {
        using namespace cere::vkdev;
        auto &vd = static_cast<VulkanDevice &>(*R.e->device);
        auto &vm = static_cast<VulkanModel &>(*R.model->device);
        const VkForwardStats &fs = vm.forward_stats();
        std::printf("forward stats: calls %llu (prefill %llu, decode %llu), tokens %llu\n",
                    (unsigned long long)fs.calls, (unsigned long long)fs.prefill_calls,
                    (unsigned long long)fs.decode_calls, (unsigned long long)fs.tokens);
        std::printf("  prefill: record %.1f ms, gpu %.1f ms, wall %.1f ms (sums)\n", fs.prefill_record_ms,
                    fs.prefill_gpu_ms, fs.prefill_wall_ms);
        {
            const VkSubmitBreakdown &b = fs.prefill_submits;
            std::printf("  prefill segments: %llu submits, %u in flight; frame gpu %.1f ms (longest %.1f); GPU idle between "
                        "segments %.1f ms over %llu gaps (%.3f ms mean, max %.3f); submit calls %.1f ms (%.3f mean); wait "
                        "calls %.1f ms; calibrated %llu: start after submit %.3f ms, retire after end %.3f ms (means)\n",
                        (unsigned long long)b.submits, vd.ctx().max_in_flight(), b.frame_gpu_ms, b.max_frame_gpu_ms,
                        b.gap_ms, (unsigned long long)b.gaps, b.gaps ? b.gap_ms / double(b.gaps) : 0.0, b.max_gap_ms,
                        b.submit_call_ms, b.submits ? b.submit_call_ms / double(b.submits) : 0.0, b.wait_call_ms,
                        (unsigned long long)b.calibrated, b.calibrated ? b.start_after_submit_ms / double(b.calibrated) : 0.0,
                        b.calibrated ? b.retire_after_end_ms / double(b.calibrated) : 0.0);
            row("prefill command buffers under the budget hint (frame stamps)", b.max_frame_gpu_ms <= cere::vk::kMaxSubmitGpuMs,
                std::to_string(b.max_frame_gpu_ms) + " ms");
        }
        std::printf("  decode: per step record %.3f ms, gpu %.3f ms, wall %.3f ms (means over %llu)\n",
                    fs.decode_calls ? fs.decode_record_ms / fs.decode_calls : 0,
                    fs.decode_calls ? fs.decode_gpu_ms / fs.decode_calls : 0,
                    fs.decode_calls ? fs.decode_wall_ms / fs.decode_calls : 0, (unsigned long long)fs.decode_calls);
        {
            const VkSubmitBreakdown &b = fs.decode_submits;
            std::printf("  decode segments: %llu submits over %llu steps; GPU idle between a step's segments %.3f ms over "
                        "%llu gaps (mean, max %.3f); submit calls %.3f ms mean; calibrated %llu steps: last retire after "
                        "the GPU end %.3f ms (mean)\n",
                        (unsigned long long)b.submits, (unsigned long long)fs.decode_calls,
                        b.gaps ? b.gap_ms / double(b.gaps) : 0.0, (unsigned long long)b.gaps, b.max_gap_ms,
                        b.submits ? b.submit_call_ms / double(b.submits) : 0.0, (unsigned long long)fs.decode_calibrated,
                        fs.decode_calibrated ? fs.decode_retire_after_end_ms / double(fs.decode_calibrated) : 0.0);
        }
        std::printf("  last call: op %u rows %llu commands %llu forward barriers %llu ring %llu B record %.3f ms "
                    "wait %.3f gpu %.3f\n",
                    fs.last_operation, (unsigned long long)fs.last_rows, (unsigned long long)fs.last_commands,
                    (unsigned long long)fs.last_barriers, (unsigned long long)fs.last_ring_bytes, fs.last_record_ms,
                    fs.last_wait_ms, fs.last_gpu_ms);
        for (uint32_t i = 0; i < uint32_t(VkGemmConsumer::kCount); ++i) {
            const VkFallbackCounter &k = fs.fallback[i];
            std::printf("  exact-fallback %-40s rows %10llu flagged %8llu (%.4f%%) scans %llu\n",
                        vk_gemm_consumer_name(VkGemmConsumer(i)), (unsigned long long)k.rows,
                        (unsigned long long)k.flagged, k.rows ? 100.0 * double(k.flagged) / double(k.rows) : 0.0,
                        (unsigned long long)k.scans);
        }
        std::printf("  exact-fallback scans dropped: %llu\n", (unsigned long long)fs.fallback_dropped);
        for (uint32_t i = 0; i < uint32_t(VkGemmConsumer::kCount); ++i) {
            const VkFallbackCounter &k = fs.fallback[i];
            std::printf("  split-route %-43s rows %10llu split %8llu (%.4f%%)\n", vk_gemm_consumer_name(VkGemmConsumer(i)),
                        (unsigned long long)k.rows, (unsigned long long)k.split,
                        k.rows ? 100.0 * double(k.split) / double(k.rows) : 0.0);
        }
        std::printf("  prefill attention: path %s, prefill calls coopmat %llu / fp32 %llu, longest prefill submit %.1f ms "
                    "(budget hint %.0f ms)\n", vm.prefill_attention_path(),
                    (unsigned long long)fs.prefill_coopmat_calls, (unsigned long long)fs.prefill_fp32_calls,
                    fs.prefill_max_submit_gpu_ms, cere::vk::kMaxSubmitGpuMs);
        for (uint32_t i = 0; i < uint32_t(VkAttnPath::kCount); ++i) {
            const VkAttnFlagCounter &a = fs.attn_flags[i];
            std::printf("  coopmat flags %-9s calls %6llu rows %10llu flagged rows %8llu (%.4f%%) pairs %8llu words %8llu\n",
                        i == 0 ? "K6 dense" : "K9 sparse", (unsigned long long)a.calls, (unsigned long long)a.rows,
                        (unsigned long long)a.flagged_rows, a.rows ? 100.0 * double(a.flagged_rows) / double(a.rows) : 0.0,
                        (unsigned long long)a.flagged_pairs, (unsigned long long)a.flagged_words);
        }
        row("prefill submits under the launch layer's budget hint", fs.prefill_max_submit_gpu_ms <= cere::vk::kMaxSubmitGpuMs,
            std::to_string(fs.prefill_max_submit_gpu_ms) + " ms");
        row("no pipeline created after the seal (warm list complete)", vd.ctx().pipeline_count() == pipelines_at_seal,
            std::to_string(pipelines_at_seal) + " at seal, " + std::to_string(vd.ctx().pipeline_count()) + " now");
        if (fs.prof_segments > 0) {
            double tot = 0;
            for (uint32_t i = 0; i < uint32_t(VkProf::kCount); ++i) tot += fs.prof_ms[i];
            std::printf("  prefill GPU time by op family (%llu segments, %llu marks dropped; sum %.1f ms):\n",
                        (unsigned long long)fs.prof_segments, (unsigned long long)fs.prof_dropped, tot);
            for (uint32_t i = 0; i < uint32_t(VkProf::kCount); ++i) {
                if (fs.prof_ms[i] == 0) continue;
                std::printf("    %-22s %10.1f ms %6.2f%%\n", vk_prof_name(VkProf(i)), fs.prof_ms[i], 100.0 * fs.prof_ms[i] / tot);
            }
        }

        // One decode step's record / command profile and the prefill-chunk profile come from the stats of the last
        // call of each kind: run one prefill + one decode of the first prompt directly on the device.
        const VkWorkspace &ws = vm.workspace();
        const uint32_t V = vm.weights().config.vocab_size;
        const auto step = [&](cere_session *s, const uint32_t *toks, uint64_t count, uint64_t limit, uint64_t *consumed,
                              cere_token_score *out) {
            return vd.step(*R.model->device, *s->device, toks, count, limit, nullptr, consumed, nullptr, out, nullptr,
                           nullptr, nullptr);
        };
        const auto logits = [&](std::vector<uint8_t> *dst) {
            dst->resize(size_t(V) * 2);
            std::memcpy(dst->data(), vm.host_of(ws.logits), dst->size());
        };
        if (prefix_slots > 0) {
            // D15 (bar B9; vulkan-amd design note P4 second pass): on a prefix model (cere_model_desc.prefix_state_slots
            // > 0; its sessions are lazy) a split prefill through the EndStep path equals one call bit for bit (logits
            // and ids): the cold call captures the state at b = floor64(split) into the state-slot pair 0 / 1 (+ the
            // BF16 K/V run of a chunk-0 end state, Device::end_kv_slots), then a fresh session sharing the cold pages
            // [0, b) loads slot 0 at b and resumes pinned to the cold call's class (cuda_device.cpp:1279-1347). Then
            // the same across the 4,096-row chunk boundary, and a prefix-cache hit through the ABI equals its cold
            // prefill. Every row is bit-exact; the session API admits one session at a time (max_sessions 1), so the
            // cold session's pages [0, b) are retained across its release.
            using cere::host::EndStep;
            constexpr uint64_t kPage = cere::host::kKvPageTokens;
            // A resumed session's context: the host admits a lazy session only while the pool's free pages cover its
            // whole context (api.cpp session_create), and the cold pages the test retains are outside the host's
            // cache, so the resume asks for 8,192 tokens (the prompts' 5,000 fit) instead of the 32,768 pool.
            constexpr uint64_t kResumeCtx = 8192;
            const auto bind = [&](cere_session *s, const uint32_t *prefix, uint64_t n_prefix) {
                return vd.bind_pages(*R.model->device, *s->device, prefix, n_prefix);
            };
            const auto run = [&](cere_session *s, const uint32_t *toks, uint64_t count, uint64_t limit,
                                 const EndStep *end, uint64_t *consumed, cere_token_score *out) {
                return vd.step(*R.model->device, *s->device, toks, count, limit, nullptr, consumed, nullptr, out,
                               nullptr, end, nullptr);
            };
            const auto differing = [&](const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) {
                uint64_t d = 0;
                for (size_t i = 0; i + 1 < a.size(); i += 2) d += a[i] != b[i] || a[i + 1] != b[i + 1];
                return d;
            };
            // A cold prefill of `ids` by plain calls (limit 0: one call per 4,096-row chunk): status, score, logits.
            struct Ref {
                cere_status st = CERE_OK;
                cere_token_score score{};
                std::vector<uint8_t> logits;
            };
            const auto cold = [&](const std::vector<uint32_t> &ids) {
                Ref r;
                cere_session *s = nullptr;
                r.st = R.api.session_create(R.e, R.model, 32768, &s);
                if (r.st != CERE_OK) return r;
                r.st = bind(s, nullptr, 0);
                for (uint64_t at = 0; r.st == CERE_OK && at < ids.size();) {
                    uint64_t c = 0;
                    r.st = run(s, ids.data() + at, ids.size() - at, 0, nullptr, &c, &r.score);
                    at += c;
                }
                logits(&r.logits);
                R.api.session_release(R.e, s);
                return r;
            };
            // One split case: `before` plain tokens, then the capture call (its chunk starts at `before`; the call
            // covers `cold_n` rows, capture at row `cap_row`), the resume at before + cap_row pinned to class cold_n over
            // the rest of that chunk, then plain calls to the end. Bit-compared with `ref`.
            const auto split_case = [&](const std::string &name, const std::vector<uint32_t> &ids, uint64_t before,
                                        uint64_t cold_n, uint64_t cap_row, const Ref &ref) {
                const uint64_t b = before + cap_row;
                const uint32_t kv = before == 0 ? vd.end_kv_slots(*R.model->device, b, cold_n) : 0;
                cere_status st = CERE_OK;
                Ref cap, res;
                std::vector<uint32_t> keep;
                {  // cold, capturing
                    cere_session *s = nullptr;
                    st = R.api.session_create(R.e, R.model, 32768, &s);
                    if (st == CERE_OK) st = bind(s, nullptr, 0);
                    uint64_t at = 0;
                    while (st == CERE_OK && at < ids.size()) {
                        uint64_t c = 0;
                        EndStep e{};
                        e.row = cap_row;
                        e.slot = 0;
                        e.kv_slots = before == 0 ? kv : 0;
                        st = run(s, ids.data() + at, ids.size() - at, 0, at == before ? &e : nullptr, &c, &cap.score);
                        if (st == CERE_OK && at == before && c != cold_n) st = CERE_ERR_INTERNAL;
                        at += c;
                    }
                    cap.st = st;
                    logits(&cap.logits);
                    if (st == CERE_OK) {
                        const auto &pg = static_cast<VulkanSession &>(*s->device).pages();
                        keep.assign(pg.begin(), pg.begin() + b / kPage);
                        st = vd.retain_pages(*R.model->device, keep.data(), keep.size());
                        if (st != CERE_OK) keep.clear();
                    }
                    if (s != nullptr) R.api.session_release(R.e, s);
                }
                {  // resumed from the pair at b
                    cere_session *s = nullptr;
                    if (st == CERE_OK) st = R.api.session_create(R.e, R.model, kResumeCtx, &s);
                    if (st == CERE_OK) st = bind(s, keep.data(), keep.size());
                    if (st == CERE_OK) st = vd.load_slot(*R.model->device, *s->device, 0, b);
                    uint64_t at = b;
                    if (st == CERE_OK) {
                        EndStep e{};
                        e.cls = cold_n;
                        e.resume = true;
                        e.in_slot = 0;
                        e.in_kv_slots = before == 0 ? kv : 0;
                        uint64_t c = 0;
                        st = run(s, ids.data() + b, ids.size() - b, before + cold_n - b, &e, &c, &res.score);
                        if (st == CERE_OK && c != before + cold_n - b) st = CERE_ERR_INTERNAL;
                        at += c;
                    }
                    while (st == CERE_OK && at < ids.size()) {
                        uint64_t c = 0;
                        st = run(s, ids.data() + at, ids.size() - at, 0, nullptr, &c, &res.score);
                        at += c;
                    }
                    res.st = st;
                    logits(&res.logits);
                    if (s != nullptr) R.api.session_release(R.e, s);
                    if (!keep.empty() && vd.release_pages(*R.model->device, keep.data(), keep.size()) != CERE_OK) {
                        res.st = CERE_ERR_INTERNAL;
                    }
                }
                const uint64_t d_cap = differing(ref.logits, cap.logits), d_res = differing(ref.logits, res.logits);
                const bool ok = ref.st == CERE_OK && cap.st == CERE_OK && res.st == CERE_OK && d_cap == 0 &&
                                d_res == 0 && std::memcmp(&ref.score, &cap.score, sizeof ref.score) == 0 &&
                                std::memcmp(&ref.score, &res.score, sizeof ref.score) == 0;
                std::printf("d15 %s: capture at %llu (call [%llu, %llu), kv slots %u): status %d/%d/%d, top-1 %u/%u/%u, "
                            "logits differing capture %llu resume %llu of %u, top-1 logit %.9g / %.9g / %.9g\n",
                            name.c_str(), (unsigned long long)b, (unsigned long long)before,
                            (unsigned long long)(before + cold_n), before == 0 ? kv : 0u, ref.st, cap.st, res.st,
                            ref.score.id, cap.score.id, res.score.id, (unsigned long long)d_cap,
                            (unsigned long long)d_res, V, double(ref.score.logit), double(cap.score.logit),
                            double(res.score.logit));
                row("D15 split " + name + ": capture call and resume == one call, logits and ids bit for bit", ok);
            };
            const Prompt *lp = &prompts.back();
            for (const Prompt &p : prompts) {
                if (p.ids.size() > lp->ids.size()) lp = &p;
            }
            // B9's shape: the first 2,040 tokens of the long prompt (one chunk-0 call of class 2,040 <= kMaxContext,
            // so its end state keeps BF16 K/V, Device::end_kv_slots). A chunk-0 call of class > 2,048 has no servable
            // end state (CUDA's rule, cuda_device.cpp:1916-1925: its K9 reads its own BF16 rows, a resume the FP8
            // cache), so the full 3,000-token prompt is not split at chunk 0.
            const std::vector<uint32_t> b9(lp->ids.begin(), lp->ids.begin() + std::min<size_t>(lp->ids.size(), 2040));
            const uint64_t L = b9.size();
            const Ref one = cold(b9);
            for (const uint64_t split : {uint64_t(1500), uint64_t(2048), uint64_t(L - 1)}) {
                if (split == 0 || split >= L) continue;  // as the cls-0 rows below
                const uint64_t b = split / kPage * kPage;
                split_case(std::to_string(split) + " of " + std::to_string(L), b9, 0, L, b, one);
            }
            // Across the 4,096-row chunk boundary: the long prompt repeated to 5,000 tokens (chunk 0 = 4,096 rows,
            // chunk 1 = 904 rows of class 904). At 4,096 the cold call ends, so the split is a plain state copy
            // (capture_state after chunk 0, load_slot, then plain calls: the host cache's own path); inside chunk 1
            // it is an end state (capture at 4,160 and at floor64(L - 1) = 4,992, resume pinned to class 904).
            const uint64_t chunk = ws.rows;
            std::vector<uint32_t> big(5000);
            for (size_t i = 0; i < big.size(); ++i) big[i] = lp->ids[i % lp->ids.size()];
            const Ref big_one = cold(big);
            {
                Ref cap, res;
                std::vector<uint32_t> keep;
                cere_session *s = nullptr;
                cere_status st = R.api.session_create(R.e, R.model, 32768, &s);
                if (st == CERE_OK) st = bind(s, nullptr, 0);
                uint64_t at = 0;
                while (st == CERE_OK && at < big.size()) {
                    uint64_t c = 0;
                    st = run(s, big.data() + at, big.size() - at, 0, nullptr, &c, &cap.score);
                    at += c;
                    if (st == CERE_OK && at == chunk) {
                        st = vd.capture_state(*R.model->device, *s->device, 0);
                        const auto &pg = static_cast<VulkanSession &>(*s->device).pages();
                        keep.assign(pg.begin(), pg.begin() + chunk / kPage);
                        if (st == CERE_OK) st = vd.retain_pages(*R.model->device, keep.data(), keep.size());
                        if (st != CERE_OK) keep.clear();
                    }
                }
                cap.st = st;
                logits(&cap.logits);
                if (s != nullptr) R.api.session_release(R.e, s);
                s = nullptr;
                if (st == CERE_OK) st = R.api.session_create(R.e, R.model, kResumeCtx, &s);
                if (st == CERE_OK) st = bind(s, keep.data(), keep.size());
                if (st == CERE_OK) st = vd.load_slot(*R.model->device, *s->device, 0, chunk);
                at = chunk;
                while (st == CERE_OK && at < big.size()) {
                    uint64_t c = 0;
                    st = run(s, big.data() + at, big.size() - at, 0, nullptr, &c, &res.score);
                    at += c;
                }
                res.st = st;
                logits(&res.logits);
                if (s != nullptr) R.api.session_release(R.e, s);
                if (!keep.empty() && vd.release_pages(*R.model->device, keep.data(), keep.size()) != CERE_OK) {
                    res.st = CERE_ERR_INTERNAL;
                }
                const uint64_t d_cap = differing(big_one.logits, cap.logits);
                const uint64_t d_res = differing(big_one.logits, res.logits);
                const bool ok = big_one.st == CERE_OK && cap.st == CERE_OK && res.st == CERE_OK && d_cap == 0 &&
                                d_res == 0 && std::memcmp(&big_one.score, &cap.score, sizeof cap.score) == 0 &&
                                std::memcmp(&big_one.score, &res.score, sizeof res.score) == 0;
                std::printf("d15 chunk boundary 4096 of 5000 (state copy): status %d/%d/%d, top-1 %u/%u/%u, logits "
                            "differing capture %llu resume %llu of %u, top-1 logit %.9g / %.9g / %.9g\n",
                            big_one.st, cap.st, res.st, big_one.score.id, cap.score.id, res.score.id,
                            (unsigned long long)d_cap, (unsigned long long)d_res, V, double(big_one.score.logit),
                            double(cap.score.logit), double(res.score.logit));
                row("D15 split 4096 of 5000 (chunk boundary, state copy) == one prefill, logits and ids bit for bit",
                    ok);
            }
            split_case("4160 of 5000 (chunk 1)", big, chunk, big.size() - chunk, 4160 - chunk, big_one);
            split_case("4999 of 5000 (chunk 1)", big, chunk, big.size() - chunk, 4992 - chunk, big_one);
            // A prefix-cache hit through the ABI equals its cold prefill: the same prompt twice under one namespace
            // (the first fills the cache at the host's capture points, the second takes cached tokens), greedy.
            const std::vector<uint32_t> *const hit_prompts[] = {&b9, &lp->ids, &big};
            for (const std::vector<uint32_t> *ids : hit_prompts) {
                uint8_t ns[16] = {'v', 'k', 'd', '1', '5'};
                ns[15] = uint8_t(ids->size() & 0xFF);
                const Gen g0 = R.generate(*ids, 16, nullptr, 0, false, ns);
                const Gen g1 = R.generate(*ids, 16, nullptr, 0, false, ns);
                const bool same = g0.status == CERE_OK && g1.status == CERE_OK && g0.tokens == g1.tokens &&
                                  g0.scores.size() == g1.scores.size() &&
                                  std::memcmp(g0.scores.data(), g1.scores.data(),
                                              g0.scores.size() * sizeof(cere_token_score)) == 0;
                std::printf("d15 prefix hit %zu tokens: status %d/%d, cached %u/%u, tokens %zu/%zu, first %u/%u\n",
                            ids->size(), g0.status, g1.status, g0.cached, g1.cached, g0.tokens.size(),
                            g1.tokens.size(), g0.tokens.empty() ? 0u : g0.tokens[0],
                            g1.tokens.empty() ? 0u : g1.tokens[0]);
                row("D15 prefix hit " + std::to_string(ids->size()) + ": cached > 0 and == cold, ids and scores bit for bit",
                    same && g0.cached == 0 && g1.cached > 0,
                    "cached " + std::to_string(g1.cached));
            }
            goto static_done;  // the cls-0 rows below run on eager sessions (cache-off model)
        }
        // Split prefill vs one call (the long prompt when present).
        const Prompt *lp = &prompts.back();
        for (const Prompt &p : prompts) {
            if (p.ids.size() > lp->ids.size()) lp = &p;
        }
        const uint64_t L = lp->ids.size();
        cere_token_score one{};
        std::vector<uint8_t> one_logits;
        {
            cere_session *s = nullptr;
            if (R.api.session_create(R.e, R.model, 32768, &s) != CERE_OK) {
                row("static rows: session_create", false);
                goto static_done;
            }
            uint64_t c = 0;
            const cere_status a = step(s, lp->ids.data(), L, 0, &c, &one);
            logits(&one_logits);
            std::printf("prefill one call %s (%llu): status %d, consumed %llu, record %.1f ms, gpu %.1f ms, wait %.1f "
                        "ms, commands %llu, forward barriers %llu, ring %llu B, submits %llu\n",
                        lp->name.c_str(), (unsigned long long)L, a, (unsigned long long)c, fs.last_record_ms,
                        fs.last_gpu_ms, fs.last_wait_ms, (unsigned long long)fs.last_commands,
                        (unsigned long long)fs.last_barriers, (unsigned long long)fs.last_ring_bytes,
                        (unsigned long long)fs.last_submits);
            {  // diagnostics: the session state after the prefill, read through the mappings
                auto &vs = static_cast<VulkanSession &>(*s->device);
                const VkSessionState &ss = vs.state();
                const auto &cfg = vm.weights().config;
                const int64_t *map = reinterpret_cast<const int64_t *>(vm.host_of(ss.slot_map));
                uint64_t map_bad = 0;
                for (uint64_t pp = 0; pp < ss.max_context_tokens; ++pp) {
                    const int64_t want = int64_t(vs.pages()[pp / 64]) * 64 + int64_t(pp % 64);
                    map_bad += map[pp] != want;
                }
                const auto nonfinite_bf16 = [&](uint64_t addr, uint64_t count) {
                    const uint16_t *v = reinterpret_cast<const uint16_t *>(vm.host_of(addr));
                    uint64_t bad = 0;
                    for (uint64_t i = 0; i < count; ++i) bad += (v[i] & 0x7F80u) == 0x7F80u;
                    return bad;
                };
                uint64_t gdn_bad = 0, ring_bad = 0;
                for (uint32_t l = 0; l < cfg.num_hidden_layers; ++l) {
                    if (ss.gdn[l].conv) gdn_bad += nonfinite_bf16(ss.gdn[l].conv, 10240 * 3) + nonfinite_bf16(ss.gdn[l].ssm, 48 * 128 * 128);
                    if (ss.qsa_ring[l]) ring_bad += nonfinite_bf16(ss.qsa_ring[l], 4 * 128);
                }
                const int64_t *hist = reinterpret_cast<const int64_t *>(vm.host_of(ss.ple.history));
                std::printf("state after prefill: slot map entries wrong %llu, GDN non-finite %llu, QSA ring non-finite "
                            "%llu, PLE history %lld %lld, position %llu\n",
                            (unsigned long long)map_bad, (unsigned long long)gdn_bad, (unsigned long long)ring_bad,
                            (long long)hist[0], (long long)hist[1], (unsigned long long)vs.position());
                if (std::getenv("FS3_PREFILL_ONLY") != nullptr) {
                    R.api.session_release(R.e, s);
                    goto static_done;
                }
            }
            // One decode step after it (the step profile at ~3,000 positions).
            cere_token_score d{};
            uint64_t c2 = 0;
            const uint32_t next = one.id;
            const cere_status b = step(s, &next, 1, 0, &c2, &d);
            std::printf("decode step after %llu: status %d, record %.3f ms, gpu %.3f ms, wait %.3f ms, read %.3f ms, "
                        "commands %llu, forward barriers %llu, ring %llu B\n",
                        (unsigned long long)L, b, fs.last_record_ms, fs.last_gpu_ms, fs.last_wait_ms, fs.last_read_ms,
                        (unsigned long long)fs.last_commands, (unsigned long long)fs.last_barriers,
                        (unsigned long long)fs.last_ring_bytes);
            R.api.session_release(R.e, s);
        }
        for (const uint64_t split : {uint64_t(1500), uint64_t(2048), uint64_t(L - 1)}) {
            if (split == 0 || split >= L) continue;
            cere_session *s = nullptr;
            if (R.api.session_create(R.e, R.model, 32768, &s) != CERE_OK) break;
            uint64_t c1 = 0, c2 = 0;
            cere_token_score two{};
            const cere_status a = step(s, lp->ids.data(), L, split, &c1, &two);
            const cere_status b = step(s, lp->ids.data() + c1, L - c1, 0, &c2, &two);
            std::vector<uint8_t> two_logits;
            logits(&two_logits);
            uint64_t differ = 0;
            double max_abs = 0;
            for (uint32_t i = 0; i < V; ++i) {
                uint16_t x, y;
                std::memcpy(&x, one_logits.data() + 2 * i, 2);
                std::memcpy(&y, two_logits.data() + 2 * i, 2);
                if (x != y) {
                    ++differ;
                    const uint32_t ux = uint32_t(x) << 16, uy = uint32_t(y) << 16;
                    float fx, fy;
                    std::memcpy(&fx, &ux, 4);
                    std::memcpy(&fy, &uy, 4);
                    max_abs = std::max(max_abs, double(std::fabs(fx - fy)));
                }
            }
            const bool bit_equal = differ == 0 && std::memcmp(&one, &two, sizeof one) == 0;
            std::printf("split prefill %llu + %llu vs one call: status %d/%d, top-1 %u/%u, logits differing %llu of %u "
                        "(max |d| %.4g), top-1 logit %.6g vs %.6g\n",
                        (unsigned long long)c1, (unsigned long long)c2, a, b, one.id, two.id,
                        (unsigned long long)differ, V, max_abs, double(one.logit), double(two.logit));
            row("split prefill " + std::to_string(c1) + "+" + std::to_string(c2) + ": same top-1 (bits reported)",
                a == CERE_OK && b == CERE_OK && one.id == two.id,
                bit_equal ? "bit-identical" : "not bit-identical (cls 0 per call: see report)");
            R.api.session_release(R.e, s);
        }
#if defined(CERE_VK_FORWARD_TAPS)
        if (!taps_path.empty()) {
            // Per-layer hidden state (last row of the HC state after each layer, then the final mixer output) of the
            // first prompt's prefill.
            const Prompt &p = prompts.front();
            cere_session *s = nullptr;
            if (R.api.session_create(R.e, R.model, 32768, &s) != CERE_OK) goto static_done;
            uint64_t c = 0;
            cere_token_score sc{};
            const cere_status a = step(s, p.ids.data(), p.ids.size(), 0, &c, &sc);
            const auto &cfg = vm.weights().config;
            const uint64_t row = uint64_t(cfg.hc_count) * cfg.hidden_size * 2;
            FILE *tf = std::fopen(taps_path.c_str(), "wb");
            std::fwrite(vm.host_of(ws.tap_shared), 1, row * (cfg.num_hidden_layers + 1), tf);
            std::fwrite(vm.host_of(ws.logits), 1, size_t(V) * 2, tf);
            std::fclose(tf);
            std::printf("taps: %s status %d, %u layers x %llu B + final + logits -> %s\n", p.name.c_str(), a,
                        cfg.num_hidden_layers, (unsigned long long)row, taps_path.c_str());
            R.api.session_release(R.e, s);
        }
#endif
    }
static_done:
#endif
    std::fclose(tr);
    st = R.api.model_release(R.e, R.model);
    row("model_release", st == CERE_OK);
    row("executor_destroy", R.api.executor_destroy(R.e) == CERE_OK);
    std::printf("forward_s3: %d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
