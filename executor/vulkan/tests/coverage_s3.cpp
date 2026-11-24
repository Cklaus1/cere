// VulkanAmd P4-F: coverage of what first light (P3) left "not exercised", on the real S3 model through the cere_* ABI
// of the Vulkan library (vulkan-amd design note "Open after G3"; report an internal benchmark report
// 20261116.md). One model per run (--model plain: speculative_draft_tokens 0, the plain deployment's model; --model
// drafts: 4 drafts + the artifact's token map, the speculating deployment's), parts chosen with --parts:
//
//   long      a prompt filling exactly one 4,096-row prefill chunk (x2, deterministic), then a context ladder stepped up
//             under a GPU-time guard (--ladder, default 16384,24576), a 32,760-token prompt decoded to the 32,768 limit
//             (x2, deterministic, finish LENGTH), a 32,767-token prompt (the longest the host takes: 1 token, LENGTH),
//             a 32,768- and a 32,769-token prompt (CAPACITY at submit), a DECODE on the full session (terminal LENGTH,
//             nothing stepped); static build: the device step at position 32,767 (writes the last KV slot) and at
//             32,768 (CAPACITY, session not broken), memory held == planned before / after, free KV pages returned,
//             the longest submit of the run against vk::kMaxSubmitGpuMs.
//   mask      vocab masks end to end (cere_work.vocab_mask + cere_executor_constrain): greedy (temperature 0) under a
//             fixed 8-id mask (step 0 = the best allowed id of the unmasked step-0 logprobs; logprobs renormalized over
//             the mask), per-step alternating masks, a seeded sampled run under the mask, an all-ones mask == the
//             unmasked sampler run bit for bit, unconstrained runs before / after unchanged, malformed masks refused.
//             On a drafts model also the grammar_drafts rows: "off" (cered adds CERE_WORK_NO_SPECULATION to a
//             constrained ticket) == the drafting session's constrained ticket without drafts, and "on"
//             (round_drafts + constrain_drafts) refused cleanly by the device (ticket ERROR UNSUPPORTED, nothing run).
//   cancel    cancel after 1 and 2 prefill chunks of a 10,000-token prompt and after 10 decode polls: CANCELLED, the
//             committed prefix == the reference's, session released, pages returned (static), and the next sessions'
//             outputs == references made before the cancels, bit for bit.
//   levers    hot set_selection rows: every slot at its one Vulkan value explicitly == the default selection's output;
//             prefill_attention toggled and back == the reference; every value the Vulkan device refuses ->
//             UNSUPPORTED (or ARGUMENT for values outside the ABI), selection unchanged, output unchanged after.
//   window    (drafts) the draft_tokens window at 2, 3 and 4: greedy speculation == plain bit for bit, each round
//             commits <= window tokens; window 1 / 5 refused.
//   r5        (drafts) sampled speculation distribution equality, m29 design note R5 restated: T 1.0, top_p 0.95,
//             min_p 0, top_k = vocab; per prompt (--r5-prompts, default explain,code) --r5-samples (512) seeded draws of
//             --r5-tokens (4) tokens on each side (plain: CERE_WORK_NO_SPECULATION, seeds 1..S; spec: seeds
//             100001..100000+S, disjoint so the two samples are independent); per prompt and position a two-sample
//             Pearson chi-square on the drawn ids (bins whose expected count is below 5 on either side pooled, as
//             scripts/live/native_spec.py pooled_two_sample); gated: every testable p >= 1e-3 / m (Bonferroni over the
//             m testable tests) and speculation engaged (verify_steps > 0). A sensitivity control (reported, not gated):
//             plain at T 1.0 vs plain at T 0.7 on the first prompt must be rejected for the test to have power.
//   longspec  (drafts) greedy speculation == plain at long contexts (--longspec, default 8192,16384,32000), stepped
//             up under the guard (the MTP extend of a 4,096-row chunk is one command buffer).
//
// Usage: coverage_s3 LIB ARTIFACT PROMPTS TRACE --model plain|drafts --parts a,b,.. [--prefill-attention cuda_core|p4]
//        [--ladder N,..] [--longspec N,..] [--r5-samples S] [--r5-tokens T] [--r5-prompts a,b] [--guard-ms G]
// PROMPTS: executor/vulkan/tests/data/spec_s3_prompts.txt (capital, code, explain, long); long prompts are the `long`
// prompt's chat header + its document body cycled + its chat tail. The trace holds `step` lines (forward_s3_compare.py
// format) and `r5` count lines; no text. Memory: ~105 GB (plain) / ~107 GB (drafts); hold /tmp/cere-gpu.lock.
// Builds (build_lib.sh): coverage_s3 (dlopen LIB, ABI v5) and coverage_s3_static (the library's objects; device rows,
// the guard and submit maxima; LIB ignored). The guard (--guard-ms, default 1500) needs the static build: after every
// poll the longest submit so far is compared with it and the run stops stepping up (the ticket is cancelled) above it.
#include "cere_executor.h"
#include "host/support.h"
#include "vk_artifact.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <set>
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
    cere_status (*constrain)(cere_executor *, cere_completion_ticket, uint64_t, const uint32_t *, uint64_t);
    cere_status (*round_drafts)(cere_executor *, cere_completion_ticket, uint64_t, uint32_t *, uint32_t, uint32_t *);
    cere_status (*constrain_drafts)(cere_executor *, cere_completion_ticket, uint64_t, const uint32_t *, uint32_t,
                                    const uint32_t *, uint64_t);
    cere_status (*set_selection)(cere_executor *, cere_model *, const cere_slot_selection *);
    cere_status (*selection)(cere_executor *, cere_model *, cere_slot_selection *);
};

template <typename F>
bool sym(void *h, const char *name, F *out) {
    *out = reinterpret_cast<F>(dlsym(h, name));
    if (*out == nullptr) std::fprintf(stderr, "missing symbol %s\n", name);
    return *out != nullptr;
}

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

std::vector<uint64_t> parse_list(const std::string &s) {
    std::vector<uint64_t> v;
    std::istringstream is(s);
    std::string x;
    while (std::getline(is, x, ',')) {
        if (!x.empty()) v.push_back(std::strtoull(x.c_str(), nullptr, 10));
    }
    return v;
}

int g_fail = 0, g_pass = 0;
void row(const std::string &name, bool ok, const std::string &detail = "") {
    std::printf("%-80s %s%s%s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.empty() ? "" : "  ", detail.c_str());
    std::fflush(stdout);
    (ok ? g_pass : g_fail)++;
}
void info(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void info(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);
    std::fflush(stdout);
}

constexpr uint32_t kVocab = 248320;
constexpr uint64_t kMaskWords = (kVocab + 31) / 32;
constexpr uint32_t kEnd = 0xFFFFFFFFu;  // r5: a position the sample did not reach

struct Mask {
    std::vector<uint32_t> w;
    std::set<uint32_t> ids;
    explicit Mask(const std::set<uint32_t> &allow) : w(kMaskWords, 0), ids(allow) {
        for (uint32_t id : allow) w[id >> 5] |= 1u << (id & 31);
    }
    bool has(uint32_t id) const { return ids.count(id) != 0; }
};

struct Gen {
    cere_status status = CERE_OK, submit_status = CERE_OK, release_status = CERE_OK;
    cere_status constrain_status = CERE_OK;  // first non-OK constrain / round_drafts / constrain_drafts
    uint32_t finish = 0;
    std::vector<uint32_t> tokens;
    std::vector<cere_token_score> scores;
    std::vector<cere_token_logprobs> logprobs;
    double ttft_ms = 0, decode_ms = 0, total_ms = 0;
    uint64_t polls = 0, verify_steps = 0, accepted = 0, revision = 0;
    uint64_t grammar_rounds = 0;  // rounds given drafts through constrain_drafts
    std::vector<uint32_t> round_committed;
    bool guard_stopped = false;
    cere_session *session = nullptr;
};

struct GenArgs {
    uint32_t operation = CERE_OP_PREFILL;
    const std::vector<uint32_t> *prompt = nullptr;
    uint64_t n_new = 0;
    uint32_t flags = 0;
    const cere_sampling_params *sampling = nullptr;
    uint32_t top_logprobs = 0;
    bool logprobs = false;
    cere_session *session = nullptr;
    uint64_t revision = 0;
    bool keep_session = false;
    uint64_t cancel_after_polls = 0;
    std::function<const Mask *(uint64_t)> mask;  // constrained: the mask of position k
    bool grammar_on = false;                     // round_drafts + constrain_drafts (grammar_drafts "on")
};

// Static build: called after every poll; returns false to stop (the ticket is cancelled: the guard).
std::function<bool()> g_after_poll;

struct Runner {
    Api api{};
    uint32_t abi = 5;
    cere_executor *e = nullptr;
    cere_model *model = nullptr;
    cere_execution_capabilities caps{};
    cere_slot_choice base_choice{};  // the run's prefill_attention choice (slot 0 = none)

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
        if (a.mask) {
            const Mask *m0 = a.mask(0);
            w.vocab_mask = m0->w.data();
            w.vocab_mask_words = kMaskWords;
        }
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
            if (a.session == nullptr && !a.keep_session) {
                g.release_status = api.session_release(e, s);
            } else {
                g.session = s;
            }
            return g;
        }
        Clock::time_point first{};
        bool have_first = false, cancelled = false;
        uint64_t last_committed = 0, given = 0;  // the mask of position 0 came with the work
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
            }
            if (!have_first && c.committed_tokens >= 1) {
                first = tq;
                have_first = true;
                g.ttft_ms = ms(t0, tq);
            }
            last_committed = c.committed_tokens;
            if (!c.terminal && !cancelled) {
                bool stop = a.cancel_after_polls != 0 && g.polls == a.cancel_after_polls;
                if (g_after_poll && !g_after_poll()) {
                    stop = true;
                    g.guard_stopped = true;
                }
                if (stop) {
                    api.cancel(e, ticket);
                    cancelled = true;
                }
            }
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
            // A constrained ticket waits for the mask of its committed position (cere_executor_constrain).
            if (a.mask && !cancelled && c.committed_tokens >= 1 && c.committed_tokens != given) {
                const uint64_t pos = c.committed_tokens;
                given = pos;
                cere_status cs = CERE_OK;
                if (a.grammar_on) {
                    uint32_t drafts[CERE_MAX_CONSTRAINED_DRAFTS] = {}, count = 0;
                    cs = api.round_drafts(e, ticket, pos, drafts, CERE_MAX_CONSTRAINED_DRAFTS, &count);
                    // Cut at the first frozen draft its row's mask excludes (a grammar would stop there).
                    uint32_t keep = 0;
                    while (cs == CERE_OK && keep < count && a.mask(pos + keep)->has(drafts[keep])) ++keep;
                    count = keep;
                    if (cs == CERE_OK && count > 0) {
                        std::vector<uint32_t> masks;
                        for (uint32_t j = 0; j <= count; ++j) {
                            const Mask *m = a.mask(pos + j);
                            masks.insert(masks.end(), m->w.begin(), m->w.end());
                        }
                        cs = api.constrain_drafts(e, ticket, pos, drafts, count, masks.data(), kMaskWords);
                        if (cs == CERE_OK) ++g.grammar_rounds;
                    } else if (cs == CERE_OK) {
                        cs = api.constrain(e, ticket, pos, a.mask(pos)->w.data(), kMaskWords);
                    }
                } else {
                    cs = api.constrain(e, ticket, pos, a.mask(pos)->w.data(), kMaskWords);
                }
                if (cs != CERE_OK && g.constrain_status == CERE_OK) g.constrain_status = cs;
            }
        }
        if (a.session == nullptr && !a.keep_session) {
            g.release_status = api.session_release(e, s);
        } else {
            g.session = s;
        }
        return g;
    }

    // set_selection with the run's base choice (prefill_attention) plus `extra` (a slot listed in both: extra wins).
    cere_status select(const std::vector<cere_slot_choice> &extra) {
        cere_slot_selection sel{};
        stamp(&sel);
        sel.block_version = CERE_SLOT_BLOCK_VERSION;
        std::map<uint32_t, cere_slot_choice> by;
        if (base_choice.slot != 0) by[base_choice.slot] = base_choice;
        for (const cere_slot_choice &c : extra) by[c.slot] = c;
        for (const auto &kv : by) sel.choices[sel.count++] = kv.second;
        return api.set_selection(e, model, &sel);
    }
    std::string selection_str() {
        cere_slot_selection out{};
        stamp(&out);
        if (api.selection(e, model, &out) != CERE_OK) return "?";
        std::string s;
        for (uint32_t i = 0; i < out.count; ++i) {
            s += std::to_string(out.choices[i].slot) + ":" + std::to_string(out.choices[i].impl);
            if (out.choices[i].params[0] != 0) s += "(" + std::to_string(out.choices[i].params[0]) + ")";
            s += i + 1 < out.count ? "," : "";
        }
        return s;
    }
};

uint32_t fbits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

long first_diff(const Gen &a, const Gen &b) {
    const size_t n = std::min(a.tokens.size(), b.tokens.size());
    for (size_t k = 0; k < n; ++k) {
        if (a.tokens[k] != b.tokens[k] || std::memcmp(&a.scores[k], &b.scores[k], sizeof(cere_token_score)) != 0) {
            return long(k);
        }
    }
    return a.tokens.size() == b.tokens.size() ? -1 : long(n);
}
bool same(const Gen &a, const Gen &b) { return first_diff(a, b) < 0; }

std::string gstr(const Gen &g) {
    return "status " + std::to_string(g.status) + " submit " + std::to_string(g.submit_status) + " finish " +
           std::to_string(g.finish) + " committed " + std::to_string(g.tokens.size());
}

FILE *g_trace = nullptr;
void trace_gen(const std::string &tag, const Gen &g) {
    for (size_t k = 0; k < g.tokens.size(); ++k) {
        const cere_token_score &s = g.scores[k];
        std::fprintf(g_trace, "step %s %zu %u %u %u %.9g %.9g %08x %08x\n", tag.c_str(), k, g.tokens[k], s.id,
                     s.runner_up, double(s.logit), double(s.runner_up_logit), fbits(s.logit), fbits(s.runner_up_logit));
    }
    std::fflush(g_trace);
}

// ---- statistics: scripts/live/native_spec.py pooled_two_sample, restated --------------------------------------------

// Regularized upper incomplete gamma Q(a, x) (series for x < a + 1, else continued fraction; Numerical Recipes 6.2).
double gamma_q(double a, double x) {
    if (x <= 0) return 1.0;
    const double gln = std::lgamma(a);
    if (x < a + 1) {
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 0; n < 100000; ++n) {
            ap += 1;
            del *= x / ap;
            sum += del;
            if (std::fabs(del) < std::fabs(sum) * 1e-15) break;
        }
        return 1.0 - sum * std::exp(-x + a * std::log(x) - gln);
    }
    double b = x + 1 - a, c = 1.0 / 1e-300, d = 1.0 / b, h = d;
    for (int i = 1; i < 100000; ++i) {
        const double an = -i * (i - a);
        b += 2;
        d = an * d + b;
        if (std::fabs(d) < 1e-300) d = 1e-300;
        c = b + an / c;
        if (std::fabs(c) < 1e-300) c = 1e-300;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1) < 1e-15) break;
    }
    return std::exp(-x + a * std::log(x) - gln) * h;
}
double chi2_sf(double x, int df) { return gamma_q(0.5 * df, 0.5 * x); }

struct Chi2 {
    bool testable = false;
    int bins = 0, df = 0;
    double stat = 0, p = 1;
    uint64_t na = 0, nb = 0;
};
constexpr double kExpectedMin = 5.0;  // sampler_live.EXPECTED_MIN

Chi2 pooled_two_sample(const std::map<uint32_t, uint64_t> &a, const std::map<uint32_t, uint64_t> &b) {
    Chi2 r;
    for (const auto &kv : a) r.na += kv.second;
    for (const auto &kv : b) r.nb += kv.second;
    const double na = double(r.na), nb = double(r.nb), n = na + nb;
    if (r.na == 0 || r.nb == 0) return r;
    std::set<uint32_t> keys;
    for (const auto &kv : a) keys.insert(kv.first);
    for (const auto &kv : b) keys.insert(kv.first);
    std::vector<std::pair<double, double>> kept;
    double pa = 0, pb = 0;
    for (uint32_t k : keys) {
        const double oa = a.count(k) ? double(a.at(k)) : 0, ob = b.count(k) ? double(b.at(k)) : 0, col = oa + ob;
        if (col == 0) continue;
        if (std::min(na * col / n, nb * col / n) >= kExpectedMin) {
            kept.emplace_back(oa, ob);
        } else {
            pa += oa;
            pb += ob;
        }
    }
    if (pa + pb > 0) {
        const double pcol = pa + pb;
        if (std::min(na * pcol / n, nb * pcol / n) >= kExpectedMin || kept.empty()) {
            kept.emplace_back(pa, pb);
        } else {
            auto smallest = std::min_element(kept.begin(), kept.end(), [](const auto &x, const auto &y) {
                return x.first + x.second < y.first + y.second;
            });
            smallest->first += pa;
            smallest->second += pb;
        }
    }
    r.bins = int(kept.size());
    if (kept.size() < 2) return r;
    for (const auto &kc : kept) {
        const double col = kc.first + kc.second;
        const double ea = na * col / n, eb = nb * col / n;
        r.stat += (kc.first - ea) * (kc.first - ea) / ea + (kc.second - eb) * (kc.second - eb) / eb;
    }
    r.df = r.bins - 1;
    r.p = chi2_sf(r.stat, r.df);
    r.testable = true;
    return r;
}

// A long prompt: the `long` prompt's chat header (3 ids) + its document body cycled + its chat tail (9 ids).
std::vector<uint32_t> long_prompt(const std::vector<uint32_t> &base, uint64_t n) {
    constexpr size_t kHead = 3, kTail = 9;
    std::vector<uint32_t> v(base.begin(), base.begin() + kHead);
    const std::vector<uint32_t> body(base.begin() + kHead, base.end() - kTail);
    for (size_t i = 0; v.size() + kTail < n; ++i) v.push_back(body[i % body.size()]);
    v.insert(v.end(), base.end() - kTail, base.end());
    v.resize(n);  // n < kHead + kTail never asked
    return v;
}

uint64_t proc_status_kb(const char *key) {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind(key, 0) == 0) return std::strtoull(line.c_str() + std::strlen(key), nullptr, 10);
    }
    return 0;
}
// amdgpu's own counters (the whole GPU, every process): GTT and VRAM used, bytes; 0 when not readable.
uint64_t amdgpu_used(const char *what) {
    for (int c = 0; c < 8; ++c) {
        std::ifstream f("/sys/class/drm/card" + std::to_string(c) + "/device/mem_info_" + what + "_used");
        uint64_t v = 0;
        if (f >> v) return v;
    }
    return 0;
}
std::string mem_line() {
    char b[200];
    std::snprintf(b, sizeof b, "VmRSS %.2f GB VmHWM %.2f GB, amdgpu gtt_used %.2f GB vram_used %.2f GB",
                  proc_status_kb("VmRSS:") / 1048576.0, proc_status_kb("VmHWM:") / 1048576.0,
                  amdgpu_used("gtt") / 1e9, amdgpu_used("vram") / 1e9);
    return b;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: coverage_s3 LIB ARTIFACT PROMPTS TRACE --model plain|drafts --parts a,b ...\n");
        return 2;
    }
    const std::string lib = argv[1], dir = argv[2], prompts_path = argv[3], trace_path = argv[4];
    std::string model_kind = "plain", parts_s, prefill_attention, r5_prompts_s = "explain,code";
    std::vector<uint64_t> ladder = {16384, 24576}, longspec = {8192, 16384, 32000};
    uint64_t r5_samples = 512, r5_tokens = 4;
    double guard_ms = 1500;
    for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--model" && more) {
            model_kind = argv[++i];
        } else if (a == "--parts" && more) {
            parts_s = argv[++i];
        } else if (a == "--prefill-attention" && more) {
            prefill_attention = argv[++i];
        } else if (a == "--ladder" && more) {
            ladder = parse_list(argv[++i]);
        } else if (a == "--longspec" && more) {
            longspec = parse_list(argv[++i]);
        } else if (a == "--r5-samples" && more) {
            r5_samples = std::strtoull(argv[++i], nullptr, 10);
        } else if (a == "--r5-tokens" && more) {
            r5_tokens = std::strtoull(argv[++i], nullptr, 10);
        } else if (a == "--r5-prompts" && more) {
            r5_prompts_s = argv[++i];
        } else if (a == "--guard-ms" && more) {
            guard_ms = std::strtod(argv[++i], nullptr);
        } else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if ((model_kind != "plain" && model_kind != "drafts") ||
        (!prefill_attention.empty() && prefill_attention != "cuda_core" && prefill_attention != "p4")) {
        std::fprintf(stderr, "--model plain|drafts, --prefill-attention cuda_core|p4\n");
        return 2;
    }
    const bool drafts = model_kind == "drafts";
    const auto part = [&](const char *p) { return ("," + parts_s + ",").find(std::string(",") + p + ",") != std::string::npos; };
    std::vector<Prompt> prompts;
    if (!read_prompts(prompts_path, &prompts)) {
        std::fprintf(stderr, "cannot read prompts %s\n", prompts_path.c_str());
        return 2;
    }
    std::map<std::string, const Prompt *> P;
    for (const Prompt &p : prompts) P[p.name] = &p;
    for (const char *need : {"capital", "code", "explain", "long"}) {
        if (!P.count(need)) {
            std::fprintf(stderr, "prompt %s missing\n", need);
            return 2;
        }
    }

    Runner R;
#if defined(FS3_STATIC)
    (void)lib;
    R.api = Api{cere_executor_abi_version, cere_executor_create, cere_executor_destroy, cere_executor_capabilities,
                cere_model_create, cere_model_put, cere_model_seal, cere_model_release, cere_session_create,
                cere_session_release, cere_executor_submit, cere_executor_poll, cere_executor_cancel,
                cere_executor_constrain, cere_executor_round_drafts, cere_executor_constrain_drafts,
                cere_model_set_selection, cere_model_selection};
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
              sym(h, "cere_executor_cancel", &R.api.cancel) && sym(h, "cere_executor_constrain", &R.api.constrain) &&
              sym(h, "cere_executor_round_drafts", &R.api.round_drafts) &&
              sym(h, "cere_executor_constrain_drafts", &R.api.constrain_drafts) &&
              sym(h, "cere_model_set_selection", &R.api.set_selection) &&
              sym(h, "cere_model_selection", &R.api.selection);
    if (!ok) return 2;
    std::printf("library: %s\n", lib.c_str());
#endif
    if (R.api.abi_version() != 5) {
        std::fprintf(stderr, "ABI %u: coverage_s3 runs the ABI v5 (Vulkan) library only\n", R.api.abi_version());
        return 2;
    }
    std::printf("model: %s, parts: %s, prefill_attention: %s, guard %.0f ms\n", model_kind.c_str(), parts_s.c_str(),
                prefill_attention.empty() ? "(default)" : prefill_attention.c_str(), guard_ms);

    // ---- desc (vk_artifact.h order and placement; drafts: mtp.* DEVICE + the artifact's token map) ----
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
    desc.struct_size = sizeof desc;
    desc.abi_version = R.abi;
    desc.model_kind = CERE_MODEL_QWEN4_EXP_NVFP4;
    desc.family = CERE_FAMILY_QWEN4_EXP;
    desc.config = &config;
    desc.tensors = tensors.data();
    desc.tensor_count = tensors.size();
    desc.max_sessions = 1;
    desc.max_context_tokens = 32768;
    if (drafts) {
        desc.speculative_draft_tokens = 4;
        desc.draft_token_map = draft_map.data();
        desc.draft_token_map_count = draft_map.size();
    }
    cere_slot_selection psel{};
    R.stamp(&psel);
    psel.block_version = CERE_SLOT_BLOCK_VERSION;
    if (!prefill_attention.empty()) {
        R.base_choice = cere_slot_choice{CERE_SLOT_PREFILL_ATTENTION,
                                         prefill_attention == "p4" ? CERE_LEVER_PREFILL_P4 : CERE_LEVER_PREFILL_CUDA_CORE,
                                         {}};
        psel.count = 1;
        psel.choices[0] = R.base_choice;
        desc.selection = &psel;
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
    // ---- create-time refusals of draft windows (resolve_selection runs before any allocation) ----
    if (part("window") || part("levers")) {
        for (const uint32_t wv : {1u, 5u, 2u}) {
            if (drafts && wv == 2) continue;  // 2 is valid with drafts (the window rows)
            cere_slot_selection sel = psel;
            sel.choices[sel.count] = cere_slot_choice{CERE_SLOT_DRAFT_TOKENS, CERE_LEVER_DRAFT_WINDOW, {wv}};
            sel.count += 1;
            cere_model_desc d = desc;
            d.selection = &sel;
            cere_model *mm = nullptr;
            const cere_status st = R.api.model_create(R.e, &d, &mm);
            if (mm != nullptr) R.api.model_release(R.e, mm);
            row("negative: model_create (" + model_kind + ") with draft_tokens window " + std::to_string(wv) +
                    " UNSUPPORTED",
                st == CERE_ERR_UNSUPPORTED, "status " + std::to_string(st));
        }
    }
    const std::string mem_before_create = mem_line();
    auto t0 = Clock::now();
    cere_status st = R.api.model_create(R.e, &desc, &R.model);
    row("model_create (" + model_kind + ")", st == CERE_OK,
        "status " + std::to_string(st) + ", " + std::to_string(ms(t0, Clock::now())) + " ms");
    if (st != CERE_OK) return 1;
    // ---- upload (reader thread + puts, as load_s3) ----
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
    row("capabilities: native, sampler, token mask, cancel" + std::string(drafts ? ", verify" : ""),
        st == CERE_OK && R.caps.execution_mode == CERE_EXECUTION_MODE_NATIVE && R.caps.native_sampler &&
            R.caps.native_token_mask && (R.caps.operations & CERE_OPS_CANCEL) &&
            (drafts == ((R.caps.operations & CERE_OPS_VERIFY) != 0)),
        "operations 0x" + [&] { char b[32]; std::snprintf(b, sizeof b, "%llx", (unsigned long long)R.caps.operations); return std::string(b); }() +
            " max_context " + std::to_string(R.caps.max_context_tokens));
    std::printf("selection at seal: %s\n", R.selection_str().c_str());
    std::printf("memory: before create %s\n        after seal    %s\n", mem_before_create.c_str(), mem_line().c_str());

    g_trace = std::fopen(trace_path.c_str(), "w");
    std::fprintf(g_trace, "# coverage_s3 trace lib=%s model=%s parts=%s\n", lib.c_str(), model_kind.c_str(),
                 parts_s.c_str());
    const uint32_t plain_flags = drafts ? CERE_WORK_NO_SPECULATION : 0u;

#if defined(FS3_STATIC)
    using namespace cere::vkdev;
    auto &vd = static_cast<VulkanDevice &>(*R.e->device);
    auto &vm = static_cast<VulkanModel &>(*R.model->device);
    uint64_t planned = 0, held = 0;
    for (size_t b = 0; b < vm.blocks().size(); ++b) {
        planned += vm.layout().blocks[b].bytes;
        held += vm.blocks()[b].size;
    }
    const uint64_t free0 = vd.free_pages(vm);
    const auto held_now = [&] { return vm.held_device_bytes() + vm.held_host_bytes(); };
    row("memory: allocated == planned == ledger (" + std::to_string(vm.blocks().size()) + " blocks)",
        planned == held && held == held_now(),
        std::to_string(planned) + " B planned, device " + std::to_string(vm.held_device_bytes()) + " host " +
            std::to_string(vm.held_host_bytes()) + ", KV pool pages free " + std::to_string(free0) +
            ", workspace rows " + std::to_string(vd.prefill_chunk(vm)));
    // The longest submit of each kind so far; the guard compares the overall maximum after every poll.
    double max_other_ms = 0;
    const auto max_submit = [&] {
        const VkForwardStats &fs = vm.forward_stats();
        return std::max({fs.prefill_max_submit_gpu_ms, fs.spec.max_verify_gpu_ms, fs.spec.max_cd_gpu_ms,
                         fs.spec.max_extend_gpu_ms, max_other_ms});
    };
    g_after_poll = [&] {
        max_other_ms = std::max(max_other_ms, vd.ctx().last_submit_gpu_ms());
        return max_submit() <= guard_ms && !vd.faulted();
    };
    const auto submit_str = [&] {
        const VkForwardStats &fs = vm.forward_stats();
        char b[240];
        std::snprintf(b, sizeof b, "longest submit %.1f ms (prefill segment %.1f, verify %.1f, commit+draft %.1f, "
                                   "MTP extend %.1f, last-of-poll %.1f)", max_submit(), fs.prefill_max_submit_gpu_ms,
                      fs.spec.max_verify_gpu_ms, fs.spec.max_cd_gpu_ms, fs.spec.max_extend_gpu_ms, max_other_ms);
        return std::string(b);
    };
#endif

    // =================================================================================================================
    // long: a full 4,096-row chunk, the ladder, the 32,768 limit.
    // =================================================================================================================
    if (part("long")) {
        std::printf("\n== long (%s) ==\n", model_kind.c_str());
        const std::vector<uint32_t> &base = P["long"]->ids;
        bool stopped = false;
        const auto run = [&](const std::vector<uint32_t> &ids, uint64_t n_new, bool keep = false) {
            GenArgs ga;
            ga.prompt = &ids;
            ga.n_new = n_new;
            ga.flags = plain_flags;
            ga.keep_session = keep;
            const Gen g = R.generate(ga);
            if (g.guard_stopped) stopped = true;
            return g;
        };
        // (1) exactly one full chunk.
        {
            const std::vector<uint32_t> ids = long_prompt(base, 4096);
            const Gen a = run(ids, 16), b = run(ids, 16);
            trace_gen("full4096", a);
            row("long: 4,096-token prompt (one full prefill chunk) + 16 greedy tokens", a.status == CERE_OK &&
                    a.tokens.size() == 16, gstr(a) + ", TTFT " + std::to_string(int(a.ttft_ms)) + " ms");
            row("long: 4,096-token prompt deterministic (x2: ids + scores bit for bit)", same(a, b) && b.tokens.size() == 16);
#if defined(FS3_STATIC)
            info("  %s\n", submit_str().c_str());
#endif
        }
        // (2) the ladder (stepped up; the guard stops it).
        for (uint64_t n : ladder) {
            if (stopped) break;
            const std::vector<uint32_t> ids = long_prompt(base, n);
            const Gen g = run(ids, 4);
            trace_gen("ladder" + std::to_string(n), g);
            row("long: ladder " + std::to_string(n) + "-token prompt + 4 tokens", g.status == CERE_OK &&
                    g.tokens.size() == 4 && !g.guard_stopped,
                gstr(g) + ", TTFT " + std::to_string(int(g.ttft_ms)) + " ms (" +
                    std::to_string(int(double(n) / (g.ttft_ms / 1000))) + " tok/s)");
#if defined(FS3_STATIC)
            info("  %s; %s\n", submit_str().c_str(), mem_line().c_str());
#endif
        }
        // (3) 32,760 tokens decoded to the limit, twice.
        Gen lim_a, lim_b;
        if (!stopped) {
            const std::vector<uint32_t> ids = long_prompt(base, 32760);
            lim_a = run(ids, 64, true);
            trace_gen("limit32760", lim_a);
            row("long: 32,760-token prompt decoded to the 32,768 limit: 8 tokens, finish LENGTH",
                lim_a.status == CERE_OK && lim_a.tokens.size() == 8 && lim_a.finish == CERE_FINISH_LENGTH &&
                    !lim_a.guard_stopped,
                gstr(lim_a) + ", TTFT " + std::to_string(int(lim_a.ttft_ms)) + " ms, decode " +
                    std::to_string(int(lim_a.decode_ms)) + " ms");
#if defined(FS3_STATIC)
            info("  %s; %s\n", submit_str().c_str(), mem_line().c_str());
#endif
            // One past the limit through the ABI: a DECODE on the full session ends LENGTH with nothing stepped.
            if (lim_a.session != nullptr && lim_a.status == CERE_OK) {
#if defined(FS3_STATIC)
                const uint64_t calls0 = vm.forward_stats().calls;
#endif
                GenArgs gd;
                gd.operation = CERE_OP_DECODE;
                gd.session = lim_a.session;
                gd.revision = lim_a.revision;
                gd.n_new = 4;
                gd.flags = plain_flags;
                const Gen d = R.generate(gd);
                bool none_stepped = true;
#if defined(FS3_STATIC)
                none_stepped = vm.forward_stats().calls == calls0;
#endif
                row("long: DECODE on the full session (32,768 tokens): terminal LENGTH, 0 committed, nothing stepped",
                    d.submit_status == CERE_OK && d.status == CERE_OK && d.finish == CERE_FINISH_LENGTH &&
                        d.tokens.empty() && none_stepped,
                    gstr(d));
#if defined(FS3_STATIC)
                // The device at the limit: the session's device position is 32,767 (the last committed token not yet
                // fed); a step there writes the last KV slot; the next one is CAPACITY and breaks nothing.
                auto &vs = static_cast<VulkanSession &>(*lim_a.session->device);
                const uint64_t pos = vs.position();
                cere_token_score sc{}, sc2{};
                uint64_t consumed = 0, consumed2 = 0;
                const uint32_t tok = lim_a.tokens.back();
                const cere_status s1 = vd.step(vm, vs, &tok, 1, 0, nullptr, &consumed, nullptr, &sc, nullptr, nullptr,
                                               nullptr);
                const uint64_t pos1 = vs.position();
                const cere_status s2 = vd.step(vm, vs, &sc.id, 1, 0, nullptr, &consumed2, nullptr, &sc2, nullptr,
                                               nullptr, nullptr);
                const bool finite = std::isfinite(sc.logit) && std::isfinite(sc.runner_up_logit) && sc.id < kVocab;
                row("long: device step at position 32,767 (the last KV slot) OK, finite top-2",
                    pos == 32767 && s1 == CERE_OK && consumed == 1 && pos1 == 32768 && finite,
                    "position " + std::to_string(pos) + " -> " + std::to_string(pos1) + ", status " +
                        std::to_string(s1) + ", top-1 " + std::to_string(sc.id));
                row("long: device step at position 32,768 CAPACITY, nothing consumed, session not broken",
                    s2 == CERE_ERR_CAPACITY && consumed2 == 0 && !vs.broken() && vs.position() == 32768,
                    "status " + std::to_string(s2));
#endif
                const cere_status rs = R.api.session_release(R.e, lim_a.session);
                row("long: full session released", rs == CERE_OK, "status " + std::to_string(rs));
            }
            if (!stopped) {
                lim_b = run(ids, 64);
                row("long: 32,760-token prompt to the limit deterministic (x2: ids + scores bit for bit)",
                    lim_b.status == CERE_OK && same(lim_a, lim_b), gstr(lim_b));
            }
        }
        // (4) the longest prompt the host takes, and the ones past it.
        if (!stopped) {
            const std::vector<uint32_t> ids = long_prompt(base, 32767);
            const Gen g = run(ids, 4);
            trace_gen("limit32767", g);
            row("long: 32,767-token prompt (the longest accepted): 1 token, finish LENGTH",
                g.status == CERE_OK && g.tokens.size() == 1 && g.finish == CERE_FINISH_LENGTH && !g.guard_stopped,
                gstr(g) + ", TTFT " + std::to_string(int(g.ttft_ms)) + " ms");
        }
        for (const uint64_t n : {uint64_t(32768), uint64_t(32769)}) {
            const std::vector<uint32_t> ids = long_prompt(base, n);
            const Gen g = run(ids, 4);
            row("long: " + std::to_string(n) + "-token prompt refused at submit (CAPACITY), session released",
                g.submit_status == CERE_ERR_CAPACITY && g.tokens.empty() && g.release_status == CERE_OK, gstr(g));
        }
        row("long: no guard stop (every submit <= " + std::to_string(int(guard_ms)) + " ms)", !stopped);
#if defined(FS3_STATIC)
        row("long: longest submit of the run under the 1,000 ms budget hint (vk::kMaxSubmitGpuMs)",
            max_submit() <= cere::vk::kMaxSubmitGpuMs, submit_str());
        row("long: memory held == planned after the long sessions; KV pages all returned",
            held_now() == planned && vd.free_pages(vm) == free0,
            std::to_string(held_now()) + " B held, free pages " + std::to_string(vd.free_pages(vm)) + "/" +
                std::to_string(free0) + "; " + mem_line());
#else
        info("  memory after long: %s\n", mem_line().c_str());
#endif
    }

    // =================================================================================================================
    // mask: vocab masks end to end.
    // =================================================================================================================
    if (part("mask")) {
        std::printf("\n== mask (%s) ==\n", model_kind.c_str());
        const Prompt &p = *P["explain"];
        cere_sampling_params g0{};
        R.stamp(&g0);
        g0.temperature = 0.0f;
        g0.top_p = 1.0f;
        g0.top_k = kVocab;
        g0.repetition_penalty = 1.0f;
        g0.seed = 1234;
        const auto plain_greedy = [&](const std::vector<uint32_t> &ids, uint64_t n) {
            GenArgs ga;
            ga.prompt = &ids;
            ga.n_new = n;
            ga.flags = plain_flags;
            return R.generate(ga);
        };
        const Gen ref_before = plain_greedy(p.ids, 32);
        // Unmasked sampler run at T 0 with the top-20 logprobs: step 0's ranking picks the masks.
        GenArgs gu;
        gu.prompt = &p.ids;
        gu.n_new = 32;
        gu.flags = plain_flags;
        gu.sampling = &g0;
        gu.logprobs = true;
        gu.top_logprobs = 20;
        const Gen unmasked = R.generate(gu);
        if (unmasked.status != CERE_OK || unmasked.logprobs.empty()) {
            row("mask: unmasked sampler reference", false, gstr(unmasked));
        } else {
            const cere_token_logprobs &l0 = unmasked.logprobs[0];
            // Mask A: ranks 2..5 of step 0 (so the mask binds: the argmax is excluded) + 4 ids outside the top 20.
            std::set<uint32_t> a_ids, b_ids, top20;
            for (uint32_t j = 0; j < l0.top_count; ++j) top20.insert(l0.top[j].id);
            for (uint32_t j = 1; j <= 4; ++j) a_ids.insert(l0.top[j].id);
            for (uint32_t id = 1000; a_ids.size() < 8; id += 997) {
                if (!top20.count(id)) a_ids.insert(id);
            }
            // Mask B: ranks 6..9 + 4 other ids outside the top 20 (disjoint from A).
            for (uint32_t j = 5; j <= 8; ++j) b_ids.insert(l0.top[j].id);
            for (uint32_t id = 50000; b_ids.size() < 8; id += 1013) {
                if (!top20.count(id) && !a_ids.count(id)) b_ids.insert(id);
            }
            const Mask A(a_ids), B(b_ids);
            std::set<uint32_t> all;
            for (uint32_t id = 0; id < kVocab; ++id) all.insert(id);
            const Mask ALL(all);
            // (1) greedy under A.
            GenArgs gm = gu;
            gm.mask = [&](uint64_t) { return &A; };
            const Gen ga = R.generate(gm);
            trace_gen("maskA:" + p.name, ga);
            uint64_t inside = 0, lp_inside = 0, lp_entries = 0;
            for (size_t k = 0; k < ga.tokens.size(); ++k) {
                inside += A.has(ga.tokens[k]);
                for (uint32_t j = 0; j < ga.logprobs[k].top_count; ++j) {
                    if (ga.logprobs[k].top[j].logprob > -1e30f) {
                        ++lp_entries;
                        lp_inside += A.has(ga.logprobs[k].top[j].id);
                    }
                }
            }
            row("mask: greedy under an 8-id mask: 32 tokens, every one inside the mask",
                ga.status == CERE_OK && ga.tokens.size() == 32 && inside == ga.tokens.size() &&
                    ga.constrain_status == CERE_OK,
                gstr(ga) + ", inside " + std::to_string(inside) + ", polls " + std::to_string(ga.polls));
            row("mask: every finite top logprob entry is a masked-in id (masked-out ids -inf)",
                lp_entries > 0 && lp_inside == lp_entries,
                std::to_string(lp_inside) + "/" + std::to_string(lp_entries) + " entries inside");
            // Step 0: the best allowed id of the unmasked ranking (rank 2), its logprob renormalized over the mask.
            if (!ga.tokens.empty()) {
                double lse = -INFINITY;
                for (uint32_t j = 0; j < l0.top_count; ++j) {
                    if (!A.has(l0.top[j].id)) continue;
                    const double x = l0.top[j].logprob;
                    lse = lse == -INFINITY ? x : std::max(lse, x) + std::log1p(std::exp(-std::fabs(lse - x)));
                }
                const double want = double(l0.top[1].logprob) - lse;
                const double got = ga.logprobs[0].logprob;
                row("mask: step 0 = the unmasked rank-2 id (the best allowed), logprob renormalized over the mask",
                    ga.tokens[0] == l0.top[1].id && std::fabs(got - want) < 2e-3,
                    "rank-2 id match " + std::to_string(ga.tokens[0] == l0.top[1].id) + ", logprob " +
                        std::to_string(got) + " vs " + std::to_string(want) + " (outside-top-20 ids neglected)");
                row("mask: step 0 score logit == the unmasked logit of that id (mask changes no allowed logit)",
                    ga.scores[0].logit == [&] {
                        // the unmasked runner-up (rank 2) logit is reported in the unmasked step-0 score
                        return unmasked.scores[0].runner_up_logit;
                    }(),
                    std::to_string(ga.scores[0].logit) + " vs " + std::to_string(unmasked.scores[0].runner_up_logit));
            }
            // (2) per-step masks: A at even positions, B at odd ones.
            GenArgs gab = gu;
            gab.logprobs = false;
            gab.mask = [&](uint64_t k) { return k % 2 == 0 ? &A : &B; };
            const Gen gx = R.generate(gab);
            trace_gen("maskAB:" + p.name, gx);
            uint64_t ok_k = 0;
            for (size_t k = 0; k < gx.tokens.size(); ++k) ok_k += (k % 2 == 0 ? A : B).has(gx.tokens[k]);
            row("mask: alternating masks A/B per position: every token inside its position's mask",
                gx.status == CERE_OK && gx.tokens.size() == 32 && ok_k == gx.tokens.size(),
                gstr(gx) + ", inside " + std::to_string(ok_k));
            // (3) seeded sampled under A.
            cere_sampling_params s1 = g0;
            s1.temperature = 1.0f;
            s1.top_k = 20;
            s1.top_p = 1.0f;
            s1.seed = 20261116;
            GenArgs gs = gu;
            gs.sampling = &s1;
            gs.mask = [&](uint64_t) { return &A; };
            const Gen gsa = R.generate(gs), gsb = R.generate(gs);
            std::set<uint32_t> distinct;
            uint64_t in_s = 0;
            for (uint32_t t : gsa.tokens) {
                in_s += A.has(t);
                distinct.insert(t);
            }
            row("mask: sampled (T 1, top_k 20, seeded) under the mask: every draw inside, x2 identical",
                gsa.status == CERE_OK && gsa.tokens.size() == 32 && in_s == gsa.tokens.size() && same(gsa, gsb),
                gstr(gsa) + ", inside " + std::to_string(in_s) + ", distinct ids " + std::to_string(distinct.size()));
            // (4) an all-ones mask == the unmasked sampler run (ids, scores, logprobs).
            GenArgs gall = gu;
            gall.mask = [&](uint64_t) { return &ALL; };
            const Gen gal = R.generate(gall);
            bool lp_same = gal.logprobs.size() == unmasked.logprobs.size();
            for (size_t k = 0; k < gal.logprobs.size() && lp_same; ++k) {
                lp_same = std::memcmp(&gal.logprobs[k], &unmasked.logprobs[k], sizeof(cere_token_logprobs)) == 0;
            }
            row("mask: all-ones mask == unmasked (T 0 sampler): ids, scores, logprobs bit for bit",
                gal.status == CERE_OK && same(gal, unmasked) && lp_same, gstr(gal));
            // (5) unconstrained greedy after the masked runs == before.
            const Gen ref_after = plain_greedy(p.ids, 32);
            trace_gen("unmasked:" + p.name, ref_after);
            row("mask: unconstrained greedy after the masked runs == before (bit for bit)",
                ref_before.status == CERE_OK && same(ref_before, ref_after), gstr(ref_after));
            row("mask: T 0 sampler ids == plain greedy ids (unmasked)", unmasked.tokens == ref_before.tokens);
            // (6) malformed masks.
            const auto bad = [&](const char *name, const uint32_t *words, uint64_t n_words, bool with_sampling,
                                 uint32_t op_verify) {
                cere_session *s = nullptr;
                R.api.session_create(R.e, R.model, 32768, &s);
                std::vector<uint32_t> outv(4);
                cere_work w{};
                R.stamp(&w);
                w.operation = CERE_OP_PREFILL;
                w.flags = plain_flags;
                w.expected_worker_epoch = 1;
                w.session = s;
                w.tokens = p.ids.data();
                w.token_count = p.ids.size();
                w.max_new_tokens = 4;
                w.out_tokens = outv.data();
                w.out_capacity = 4;
                w.sampling = with_sampling ? &g0 : nullptr;
                w.vocab_mask = words;
                w.vocab_mask_words = n_words;
                cere_batch b{};
                R.stamp(&b);
                b.expected_worker_epoch = 1;
                b.capability_profile_generation = R.caps.profile_generation;
                b.items = &w;
                b.item_count = 1;
                cere_completion_ticket t{};
                const cere_status sst = R.api.submit(R.e, &b, &t);
                R.api.session_release(R.e, s);
                (void)op_verify;
                row(std::string("negative: ") + name + " refused (ARGUMENT)", sst == CERE_ERR_ARGUMENT,
                    "status " + std::to_string(sst));
            };
            const std::vector<uint32_t> zero(kMaskWords, 0);
            bad("all-zero mask", zero.data(), kMaskWords, true, 0);
            bad("mask with the wrong word count", A.w.data(), kMaskWords - 1, true, 0);
            bad("mask without sampling params", A.w.data(), kMaskWords, false, 0);
            bad("mask words without a mask", nullptr, kMaskWords, true, 0);
#if defined(FS3_STATIC)
            row("mask: KV pages all returned after the mask rows", vd.free_pages(vm) == free0,
                std::to_string(vd.free_pages(vm)) + "/" + std::to_string(free0));
#endif
            // (7) grammar_drafts (drafts model).
            if (drafts) {
                GenArgs goff = gu;  // "off": cered adds NO_SPECULATION to a constrained ticket
                goff.logprobs = false;
                goff.mask = [&](uint64_t) { return &A; };
                goff.flags = CERE_WORK_NO_SPECULATION;
                const Gen off = R.generate(goff);
                GenArgs gdr = goff;  // a drafting session's constrained ticket that is given no drafts
                gdr.flags = 0;
                const Gen dr = R.generate(gdr);
                trace_gen("gd_off:" + p.name, off);
                trace_gen("gd_drafting:" + p.name, dr);
                uint64_t in_off = 0;
                for (uint32_t t : off.tokens) in_off += A.has(t);
                row("grammar_drafts off: constrained ticket with NO_SPECULATION on the drafts model: 32 tokens "
                    "inside the mask, no speculation round, ids == the masked greedy run",
                    off.status == CERE_OK && off.tokens.size() == 32 && in_off == 32 && off.verify_steps == 0 &&
                        off.tokens == ga.tokens,
                    gstr(off) + ", verify_steps " + std::to_string(off.verify_steps));
                row("grammar_drafts: drafting session's constrained ticket without drafts == the NO_SPECULATION one "
                    "(ids + scores bit for bit)",
                    dr.status == CERE_OK && same(off, dr), gstr(dr) + ", verify_steps " + std::to_string(dr.verify_steps) +
                                                               " accepted " + std::to_string(dr.accepted));
                // "on" (round_drafts + constrain_drafts, P4-F): the frozen drafts cut at the first id its row's mask
                // excludes (as a grammar would), one mask per verify row. Greedy: == "off" bit for bit (GS-01).
                GenArgs gon = gdr;
                gon.grammar_on = true;
                const Gen on = R.generate(gon);
                trace_gen("gd_on:" + p.name, on);
                row("grammar_drafts on, 8-id mask: greedy ids + scores == off, bit for bit",
                    on.status == CERE_OK && on.constrain_status == CERE_OK && same(off, on),
                    gstr(on) + ", rounds given drafts " + std::to_string(on.grammar_rounds) + ", accepted " +
                        std::to_string(on.accepted) + ", constrain status " + std::to_string(on.constrain_status));
                // A mask the model's own continuation satisfies (the unmasked greedy ids + A): drafts get accepted.
                std::set<uint32_t> m_ids(a_ids);
                for (uint32_t t : ref_before.tokens) m_ids.insert(t);
                for (uint32_t t : unmasked.tokens) m_ids.insert(t);
                const Mask M(m_ids);
                GenArgs moff = goff, mon = gon;
                moff.mask = [&](uint64_t) { return &M; };
                mon.mask = [&](uint64_t) { return &M; };
                const Gen off_m = R.generate(moff), on_m = R.generate(mon);
                trace_gen("gd_offM:" + p.name, off_m);
                trace_gen("gd_onM:" + p.name, on_m);
                uint64_t in_m = 0;
                for (uint32_t t : on_m.tokens) in_m += M.has(t);
                row("grammar_drafts on, " + std::to_string(m_ids.size()) + "-id mask the continuation fits: greedy == "
                    "off bit for bit, drafts accepted, every token inside",
                    on_m.status == CERE_OK && same(off_m, on_m) && on_m.accepted > 0 && in_m == on_m.tokens.size() &&
                        on_m.tokens.size() == 32,
                    gstr(on_m) + ", rounds given drafts " + std::to_string(on_m.grammar_rounds) + ", verify " +
                        std::to_string(on_m.verify_steps) + ", accepted " + std::to_string(on_m.accepted));
                // Sampled "on" under M: seeded x2 identical, every draw inside (distribution: part r5grammar).
                cere_sampling_params s2 = g0;
                s2.temperature = 1.0f;
                s2.top_p = 0.95f;
                s2.seed = 777;
                GenArgs son = mon;
                son.sampling = &s2;
                const Gen sa = R.generate(son), sb = R.generate(son);
                uint64_t in_sa = 0;
                for (uint32_t t : sa.tokens) in_sa += M.has(t);
                row("grammar_drafts on, sampled (T 1, top_p 0.95, seeded) under the mask: x2 identical, every draw "
                    "inside",
                    sa.status == CERE_OK && sa.tokens.size() == 32 && same(sa, sb) && in_sa == sa.tokens.size(),
                    gstr(sa) + ", rounds given drafts " + std::to_string(sa.grammar_rounds) + ", accepted " +
                        std::to_string(sa.accepted));
                const Gen after_on = R.generate(goff);
                row("grammar_drafts: the off run again after the on runs, bit for bit", same(off, after_on),
                    gstr(after_on));
                // VERIFY on a constrained ticket is ARGUMENT (cere_work.vocab_mask contract).
                GenArgs gk = goff;
                gk.flags = 0;
                gk.n_new = 1;
                gk.keep_session = true;
                const Gen k1 = R.generate(gk);
                if (k1.session != nullptr) {
                    std::vector<uint32_t> outv(4), dv = {k1.tokens.empty() ? 1u : k1.tokens[0]};
                    cere_work w{};
                    R.stamp(&w);
                    w.operation = CERE_OP_VERIFY;
                    w.expected_worker_epoch = 1;
                    w.expected_session_revision = k1.revision;
                    w.session = k1.session;
                    w.tokens = dv.data();
                    w.token_count = 1;
                    w.max_new_tokens = 2;
                    w.out_tokens = outv.data();
                    w.out_capacity = 4;
                    w.sampling = &g0;
                    w.vocab_mask = A.w.data();
                    w.vocab_mask_words = kMaskWords;
                    cere_batch b{};
                    R.stamp(&b);
                    b.expected_worker_epoch = 1;
                    b.capability_profile_generation = R.caps.profile_generation;
                    b.items = &w;
                    b.item_count = 1;
                    cere_completion_ticket t{};
                    const cere_status vs = R.api.submit(R.e, &b, &t);
                    row("negative: VERIFY with a vocab mask refused (ARGUMENT)", vs == CERE_ERR_ARGUMENT,
                        "status " + std::to_string(vs));
                    R.api.session_release(R.e, k1.session);
                }
            }
        }
    }

    // =================================================================================================================
    // cancel: mid-prefill (between chunk calls) and mid-decode; then bit-identical later sessions.
    // =================================================================================================================
    if (part("cancel")) {
        std::printf("\n== cancel (%s, %s sessions) ==\n", model_kind.c_str(), drafts ? "drafting" : "plain");
        const std::vector<uint32_t> big = long_prompt(P["long"]->ids, 10000);
        const Prompt &q = *P["explain"];
        const auto gen = [&](const std::vector<uint32_t> &ids, uint64_t n, uint64_t cancel_after) {
            GenArgs ga;
            ga.prompt = &ids;
            ga.n_new = n;
            ga.cancel_after_polls = cancel_after;
            return R.generate(ga);  // default flags: a drafting session on the drafts model
        };
        const Gen ref_big = gen(big, 16, 0), ref_q = gen(q.ids, 48, 0);
        trace_gen("cancel_ref_big", ref_big);
        trace_gen("cancel_ref_q", ref_q);
        row("cancel: references (10,000-token prompt + 16, explain + 48)",
            ref_big.status == CERE_OK && ref_big.tokens.size() == 16 && ref_q.status == CERE_OK &&
                ref_q.tokens.size() == 48,
            gstr(ref_big) + " / " + gstr(ref_q) + ", prefill polls " + std::to_string(ref_big.polls));
        for (const uint64_t after : {uint64_t(1), uint64_t(2)}) {
            const Gen c = gen(big, 16, after);
            bool pages = true;
#if defined(FS3_STATIC)
            pages = vd.free_pages(vm) == free0;
#endif
            row("cancel: after " + std::to_string(after) + " of 3 prefill chunk calls: CANCELLED, 0 committed, "
                    "session released, pages returned",
                c.status == CERE_OK && c.finish == CERE_FINISH_CANCELLED && c.tokens.empty() &&
                    c.release_status == CERE_OK && pages,
                gstr(c) + ", polls " + std::to_string(c.polls));
        }
        {
            const Gen a = gen(q.ids, 48, 0), b = gen(big, 16, 0);
            row("cancel: after the prefill cancels, the next sessions == the references (bit for bit)",
                same(a, ref_q) && same(b, ref_big), gstr(a) + " / " + gstr(b));
        }
        {
            const Gen c = gen(q.ids, 48, 10);
            bool pages = true;
#if defined(FS3_STATIC)
            pages = vd.free_pages(vm) == free0;
#endif
            row("cancel: after 10 decode polls: CANCELLED, committed prefix == the reference's, pages returned",
                c.status == CERE_OK && c.finish == CERE_FINISH_CANCELLED && !c.tokens.empty() &&
                    c.tokens.size() < 48 && first_diff(ref_q, c) == long(c.tokens.size()) &&
                    c.release_status == CERE_OK && pages,
                gstr(c) + ", rounds " + std::to_string(c.verify_steps));
            const Gen a = gen(q.ids, 48, 0);
            row("cancel: after the decode cancel, the next session == the reference (bit for bit)", same(a, ref_q),
                gstr(a));
        }
#if defined(FS3_STATIC)
        row("cancel: memory held == planned, KV pages all free", held_now() == planned && vd.free_pages(vm) == free0,
            std::to_string(vd.free_pages(vm)) + "/" + std::to_string(free0));
#endif
    }

    // =================================================================================================================
    // levers: hot set_selection rows.
    // =================================================================================================================
    if (part("levers")) {
        std::printf("\n== levers (%s) ==\n", model_kind.c_str());
        const Prompt &p = *P["long"];
        const auto greedy = [&] {
            GenArgs ga;
            ga.prompt = &p.ids;
            ga.n_new = 16;
            ga.flags = plain_flags;
            return R.generate(ga);
        };
        const std::string sel0 = R.selection_str();
        const Gen ref = greedy();
        trace_gen("levers_ref", ref);
        // Every slot at its one Vulkan value, explicitly.
        std::vector<cere_slot_choice> all = {
            {CERE_SLOT_K2_GEMM, CERE_LEVER_K2_CUBLASLT, {}},      {CERE_SLOT_DECODE_STEP, CERE_LEVER_DECODE_EAGER, {}},
            {CERE_SLOT_MOE_GEMM, CERE_LEVER_MOE_K7, {}},          {CERE_SLOT_VERIFY_ATTENTION, CERE_LEVER_VERIFY_PER_ROW, {}},
            {CERE_SLOT_K2_WEIGHTS, CERE_LEVER_K2_WEIGHTS_BF16, {}}};
        if (drafts) all.push_back({CERE_SLOT_DRAFT_TOKENS, CERE_LEVER_DRAFT_WINDOW, {4}});
        const cere_status sa = R.select(all);
        const Gen ga = greedy();
        row("levers: every slot explicit at its Vulkan value (k2 cublaslt, eager, moe k7, per_row, bf16" +
                std::string(drafts ? ", window 4" : "") + ") == the default selection's output",
            sa == CERE_OK && same(ref, ga), "set " + std::to_string(sa) + ", read back " + R.selection_str());
        // prefill_attention toggled and back.
        const uint32_t cur = R.base_choice.slot != 0 ? R.base_choice.impl : CERE_LEVER_PREFILL_CUDA_CORE;
        const uint32_t other = cur == CERE_LEVER_PREFILL_P4 ? CERE_LEVER_PREFILL_CUDA_CORE : CERE_LEVER_PREFILL_P4;
        const cere_status so = R.select({{CERE_SLOT_PREFILL_ATTENTION, other, {}}});
        const Gen gother = greedy();
        const cere_status sb = R.select({});
        const Gen gback = greedy();
        trace_gen("levers_other_prefill", gother);
        row("levers: prefill_attention hot-swapped to the other path and back: output == the reference",
            so == CERE_OK && sb == CERE_OK && gother.status == CERE_OK && same(ref, gback),
            "other path: " + std::string(same(ref, gother) ? "identical" : "differs at step " +
                                                                 std::to_string(first_diff(ref, gother))) +
                " (not a contract: different kernels)");
        // P4-B / D7: k2_gemm tail_split (the GEMM's SPLIT x rows) hot-swapped on and back to the default.
        const cere_status ts = R.select({{CERE_SLOT_K2_GEMM, CERE_LEVER_K2_TAIL_SPLIT, {}}});
        const std::string tsel = R.selection_str();
        const Gen gsplit = greedy();
        const cere_status tb = R.select({});
        const Gen gtback = greedy();
        trace_gen("levers_k2_tail_split", gsplit);
        row("levers: k2_gemm tail_split hot-swapped on and back: output == the reference",
            ts == CERE_OK && tb == CERE_OK && gsplit.status == CERE_OK && same(ref, gtback),
            "selection " + tsel + "; split: " + std::string(same(ref, gsplit) ? "identical" : "differs at step " +
                                                                 std::to_string(first_diff(ref, gsplit))) +
                " (numerics-changing within the K2 contract: not a bit contract)");
        // Refused values: status, selection unchanged.
        struct Neg {
            const char *name;
            cere_slot_choice c;
            cere_status want;
        };
        std::vector<Neg> negs = {
            {"k2_gemm small_m", {CERE_SLOT_K2_GEMM, CERE_LEVER_K2_SMALL_M, {}}, CERE_ERR_UNSUPPORTED},
            {"k2_gemm small_m_a8", {CERE_SLOT_K2_GEMM, CERE_LEVER_K2_SMALL_M_A8, {}}, CERE_ERR_UNSUPPORTED},
            {"decode_step graph", {CERE_SLOT_DECODE_STEP, CERE_LEVER_DECODE_GRAPH, {}}, CERE_ERR_UNSUPPORTED},
            {"decode_step spec_graph", {CERE_SLOT_DECODE_STEP, CERE_LEVER_DECODE_SPEC_GRAPH, {}}, CERE_ERR_UNSUPPORTED},
            {"moe_gemm p3b", {CERE_SLOT_MOE_GEMM, CERE_LEVER_MOE_P3B, {}}, CERE_ERR_UNSUPPORTED},
            {"verify_attention batched", {CERE_SLOT_VERIFY_ATTENTION, CERE_LEVER_VERIFY_BATCHED, {}}, CERE_ERR_UNSUPPORTED},
            {"k2_weights fp8 (fixed at create)", {CERE_SLOT_K2_WEIGHTS, CERE_LEVER_K2_WEIGHTS_FP8, {}}, CERE_ERR_UNSUPPORTED},
            {"draft_tokens window 1", {CERE_SLOT_DRAFT_TOKENS, CERE_LEVER_DRAFT_WINDOW, {1}}, CERE_ERR_UNSUPPORTED},
            {"draft_tokens window 5", {CERE_SLOT_DRAFT_TOKENS, CERE_LEVER_DRAFT_WINDOW, {5}}, CERE_ERR_UNSUPPORTED},
            {"k2_gemm impl 4 (outside the ABI)", {CERE_SLOT_K2_GEMM, CERE_LEVER_K2_TAIL_SPLIT + 1, {}}, CERE_ERR_ARGUMENT},
            {"prefill_attention impl 2 (outside the ABI)", {CERE_SLOT_PREFILL_ATTENTION, 2, {}}, CERE_ERR_ARGUMENT},
        };
        if (!drafts) negs.push_back({"draft_tokens window 2 (model without drafts)",
                                     {CERE_SLOT_DRAFT_TOKENS, CERE_LEVER_DRAFT_WINDOW, {2}}, CERE_ERR_UNSUPPORTED});
        for (const Neg &n : negs) {
            const cere_status s = R.select({n.c});
            const std::string now = R.selection_str();
            row(std::string("negative: set_selection ") + n.name + " refused, selection unchanged", s == n.want &&
                    now == sel0, "status " + std::to_string(s) + ", selection " + now);
        }
        const Gen gafter = greedy();
        row("levers: output after the refused swaps == the reference (bit for bit)", same(ref, gafter), gstr(gafter));
    }

    // =================================================================================================================
    // window: the draft_tokens window 2 / 3 / 4 (drafts model).
    // =================================================================================================================
    if (part("window") && drafts) {
        std::printf("\n== window ==\n");
        std::map<std::string, Gen> plain_of;
        for (const char *name : {"capital", "code", "explain"}) {
            GenArgs ga;
            ga.prompt = &P[name]->ids;
            ga.n_new = 64;
            ga.flags = CERE_WORK_NO_SPECULATION;
            plain_of[name] = R.generate(ga);
        }
        for (const uint32_t wv : {2u, 3u, 4u}) {
            const cere_status s = R.select({{CERE_SLOT_DRAFT_TOKENS, CERE_LEVER_DRAFT_WINDOW, {wv}}});
            row("window " + std::to_string(wv) + ": set_selection OK", s == CERE_OK, "selection " + R.selection_str());
            for (const char *name : {"capital", "code", "explain"}) {
                GenArgs ga;
                ga.prompt = &P[name]->ids;
                ga.n_new = 64;
                const Gen g = R.generate(ga);
                trace_gen("window" + std::to_string(wv) + ":" + name, g);
                const Gen &pl = plain_of[name];
                uint32_t most = 0;
                for (uint32_t c : g.round_committed) most = std::max(most, c);
                const long d = first_diff(pl, g);
                row("LOSSLESS window " + std::to_string(wv) + " " + name + " (64): spec ids + top-2 logits == plain, "
                    "each round commits <= " + std::to_string(wv),
                    pl.status == CERE_OK && g.status == CERE_OK && g.tokens.size() == 64 && d < 0 && most <= wv &&
                        g.verify_steps > 0,
                    "rounds " + std::to_string(g.verify_steps) + " accepted " + std::to_string(g.accepted) +
                        " max per round " + std::to_string(most) + (d >= 0 ? ", FIRST MISMATCH " + std::to_string(d) : ""));
            }
        }
        R.select({});
        row("window: selection restored to the default", R.selection_str().find("5:0(") == std::string::npos,
            R.selection_str());
    }

    // =================================================================================================================
    // r5: sampled speculation distribution equality (drafts model).
    // =================================================================================================================
    if ((part("r5") || part("r5grammar")) && drafts) {
        std::printf("\n== r5: %llu samples x %llu tokens per prompt and side ==\n", (unsigned long long)r5_samples,
                    (unsigned long long)r5_tokens);
        std::vector<std::string> r5_prompts;
        {
            std::istringstream is(r5_prompts_s);
            std::string x;
            while (std::getline(is, x, ',')) r5_prompts.push_back(x);
        }
        cere_sampling_params sp{};
        R.stamp(&sp);
        sp.temperature = 1.0f;
        sp.top_p = 0.95f;
        sp.min_p = 0.0f;
        sp.top_k = kVocab;
        sp.repetition_penalty = 1.0f;
        struct Side {
            std::vector<std::vector<uint32_t>> draws;
            uint64_t verify_steps = 0, accepted = 0, bad = 0, grammar_rounds = 0;
            double ms = 0;
        };
        const auto draw = [&](const Prompt &p, uint32_t flags, uint64_t seed0, float temperature,
                              const Mask *mask = nullptr, bool grammar = false) {
            Side s;
            const auto t1 = Clock::now();
            for (uint64_t i = 0; i < r5_samples; ++i) {
                cere_sampling_params x = sp;
                x.temperature = temperature;
                x.seed = seed0 + i;
                GenArgs ga;
                ga.prompt = &p.ids;
                ga.n_new = r5_tokens;
                ga.flags = flags;
                ga.sampling = &x;
                if (mask != nullptr) ga.mask = [mask](uint64_t) { return mask; };
                ga.grammar_on = grammar;
                const Gen g = R.generate(ga);
                if (g.status != CERE_OK || g.constrain_status != CERE_OK) ++s.bad;
                s.grammar_rounds += g.grammar_rounds;
                std::vector<uint32_t> k(g.tokens.begin(), g.tokens.end());
                k.resize(r5_tokens, kEnd);
                s.draws.push_back(k);
                s.verify_steps += g.verify_steps;
                s.accepted += g.accepted;
            }
            s.ms = ms(t1, Clock::now());
            return s;
        };
        const auto counts = [&](const Side &s, uint64_t pos) {
            std::map<uint32_t, uint64_t> m;
            for (const auto &d : s.draws) m[d[pos]]++;
            return m;
        };
        std::vector<std::pair<std::string, Chi2>> tests;
        uint64_t steps = 0, bad = 0;
        for (const std::string &name : r5_prompts) {
            if (!part("r5")) break;
            if (!P.count(name)) continue;
            const Prompt &p = *P[name];
            const Side pl = draw(p, CERE_WORK_NO_SPECULATION, 1, 1.0f);
            const Side spc = draw(p, 0, 100001, 1.0f);
            steps += spc.verify_steps;
            bad += pl.bad + spc.bad;
            info("  %s: plain %.1f s, spec %.1f s (rounds %llu, accepted %llu, accept length %.3f)\n", name.c_str(),
                 pl.ms / 1000, spc.ms / 1000, (unsigned long long)spc.verify_steps, (unsigned long long)spc.accepted,
                 spc.verify_steps ? double(r5_samples * (r5_tokens - 1)) / double(spc.verify_steps) : 0.0);
            for (uint64_t pos = 0; pos < r5_tokens; ++pos) {
                const auto ca = counts(pl, pos), cb = counts(spc, pos);
                const Chi2 c = pooled_two_sample(ca, cb);
                tests.emplace_back(name + "/" + std::to_string(pos), c);
                info("  r5 %s pos %llu: distinct plain %zu spec %zu, bins %d, chi2 %.3f df %d p %.4g%s\n", name.c_str(),
                     (unsigned long long)pos, ca.size(), cb.size(), c.bins, c.stat, c.df, c.p,
                     c.testable ? "" : " (not testable)");
                for (const auto &kv : ca) std::fprintf(g_trace, "r5 %s %llu plain %u %llu\n", name.c_str(), (unsigned long long)pos, kv.first, (unsigned long long)kv.second);
                for (const auto &kv : cb) std::fprintf(g_trace, "r5 %s %llu spec %u %llu\n", name.c_str(), (unsigned long long)pos, kv.first, (unsigned long long)kv.second);
            }
            std::fflush(g_trace);
            // Sensitivity control on the first prompt: plain T 1.0 vs plain T 0.7 (reported only).
            if (&name == &r5_prompts.front()) {
                const Side cold = draw(p, CERE_WORK_NO_SPECULATION, 200001, 0.7f);
                double pmin = 1;
                std::string ps;
                for (uint64_t pos = 0; pos < r5_tokens; ++pos) {
                    const Chi2 c = pooled_two_sample(counts(pl, pos), counts(cold, pos));
                    if (c.testable) pmin = std::min(pmin, c.p);
                    char b[64];
                    std::snprintf(b, sizeof b, "%s%.3g", pos ? "/" : "", c.testable ? c.p : 1.0);
                    ps += b;
                }
                info("  r5 sensitivity control (reported): %s plain T 1.0 vs plain T 0.7, p per position %s, min %.3g\n",
                     name.c_str(), ps.c_str(), pmin);
            }
        }
        if (part("r5")) {
        uint64_t m = 0;
        for (const auto &t : tests) m += t.second.testable;
        const double thr = m ? 1e-3 / double(m) : 0;
        std::string low;
        double pmin = 1;
        for (const auto &t : tests) {
            if (!t.second.testable) continue;
            pmin = std::min(pmin, t.second.p);
            if (t.second.p < thr) low += t.first + " ";
        }
        row("R5 sampled speculation == plain in distribution: every testable p >= 1e-3/m (Bonferroni), "
            "speculation engaged, every draw OK",
            m > 0 && low.empty() && steps > 0 && bad == 0,
            "testable " + std::to_string(m) + " of " + std::to_string(tests.size()) + ", threshold " +
                std::to_string(thr) + ", min p " + std::to_string(pmin) + ", rounds " + std::to_string(steps) +
                ", failed draws " + std::to_string(bad) + (low.empty() ? "" : ", LOW: " + low));
        }
        // r5grammar (P4-F grammar drafts): the same statistic, constrained tickets under one mask (the union of the
        // unmasked greedy run's top-20 ids over its first 8 steps): grammar_drafts "on" (round_drafts +
        // constrain_drafts, spec seeds) vs "off" (NO_SPECULATION, plain seeds), first prompt only.
        if (part("r5grammar") && !r5_prompts.empty() && P.count(r5_prompts.front())) {
            const Prompt &p = *P[r5_prompts.front()];
            cere_sampling_params g0 = sp;
            g0.temperature = 0.0f;
            GenArgs gu;
            gu.prompt = &p.ids;
            gu.n_new = 8;
            gu.flags = CERE_WORK_NO_SPECULATION;
            gu.sampling = &g0;
            gu.logprobs = true;
            gu.top_logprobs = 20;
            const Gen u = R.generate(gu);
            std::set<uint32_t> ids;
            for (const cere_token_logprobs &l : u.logprobs) {
                for (uint32_t j = 0; j < l.top_count; ++j) ids.insert(l.top[j].id);
            }
            const Mask M(ids);
            const Side off = draw(p, CERE_WORK_NO_SPECULATION, 300001, 1.0f, &M, false);
            const Side on = draw(p, 0, 400001, 1.0f, &M, true);
            uint64_t outside = 0, mg = 0;
            std::string low;
            double pmin = 1;
            std::vector<Chi2> gt;
            for (uint64_t pos = 0; pos < r5_tokens; ++pos) {
                const auto ca = counts(off, pos), cb = counts(on, pos);
                for (const auto &kv : cb) outside += kv.first != kEnd && !M.has(kv.first) ? kv.second : 0;
                for (const auto &kv : ca) outside += kv.first != kEnd && !M.has(kv.first) ? kv.second : 0;
                const Chi2 c = pooled_two_sample(ca, cb);
                gt.push_back(c);
                mg += c.testable;
                info("  r5grammar %s pos %llu: distinct off %zu on %zu, bins %d, chi2 %.3f df %d p %.4g%s\n",
                     p.name.c_str(), (unsigned long long)pos, ca.size(), cb.size(), c.bins, c.stat, c.df, c.p,
                     c.testable ? "" : " (not testable)");
                for (const auto &kv : ca) std::fprintf(g_trace, "r5g %s %llu off %u %llu\n", p.name.c_str(), (unsigned long long)pos, kv.first, (unsigned long long)kv.second);
                for (const auto &kv : cb) std::fprintf(g_trace, "r5g %s %llu on %u %llu\n", p.name.c_str(), (unsigned long long)pos, kv.first, (unsigned long long)kv.second);
            }
            const double thr = mg ? 1e-3 / double(mg) : 0;
            for (uint64_t pos = 0; pos < gt.size(); ++pos) {
                if (!gt[pos].testable) continue;
                pmin = std::min(pmin, gt[pos].p);
                if (gt[pos].p < thr) low += std::to_string(pos) + " ";
            }
            info("  r5grammar: mask %zu ids; off %.1f s, on %.1f s (rounds %llu, given drafts %llu, accepted %llu)\n",
                 ids.size(), off.ms / 1000, on.ms / 1000, (unsigned long long)on.verify_steps,
                 (unsigned long long)on.grammar_rounds, (unsigned long long)on.accepted);
            row("R5 grammar drafts: constrained sampled on == off in distribution (every testable p >= 1e-3/m), "
                "drafts given and accepted, every draw inside the mask",
                mg > 0 && low.empty() && on.grammar_rounds > 0 && on.accepted > 0 && outside == 0 && off.bad == 0 &&
                    on.bad == 0,
                "testable " + std::to_string(mg) + ", threshold " + std::to_string(thr) + ", min p " +
                    std::to_string(pmin) + ", draws outside " + std::to_string(outside) + ", failed " +
                    std::to_string(off.bad + on.bad) + (low.empty() ? "" : ", LOW positions: " + low));
        }
    }

    // =================================================================================================================
    // longspec: greedy speculation == plain at long contexts (drafts model), stepped up under the guard.
    // =================================================================================================================
    if (part("longspec") && drafts) {
        std::printf("\n== longspec ==\n");
        for (uint64_t n : longspec) {
            const std::vector<uint32_t> ids = long_prompt(P["long"]->ids, n);
            GenArgs ga;
            ga.prompt = &ids;
            ga.n_new = 32;
            ga.flags = CERE_WORK_NO_SPECULATION;
            const Gen pl = R.generate(ga);
            if (pl.guard_stopped) {
                row("longspec " + std::to_string(n) + ": guard stop (plain)", false);
                break;
            }
            ga.flags = 0;
            const Gen sp = R.generate(ga);
            trace_gen("longspec_plain" + std::to_string(n), pl);
            trace_gen("longspec_spec" + std::to_string(n), sp);
            const long d = first_diff(pl, sp);
            row("LOSSLESS longspec " + std::to_string(n) + " (+32): spec ids + top-2 logits == plain, bit for bit",
                pl.status == CERE_OK && sp.status == CERE_OK && pl.tokens.size() == 32 && d < 0 && sp.verify_steps > 0 &&
                    !sp.guard_stopped,
                "rounds " + std::to_string(sp.verify_steps) + " accepted " + std::to_string(sp.accepted) +
                    ", TTFT plain " + std::to_string(int(pl.ttft_ms)) + " spec " + std::to_string(int(sp.ttft_ms)) +
                    " ms" + (d >= 0 ? ", FIRST MISMATCH " + std::to_string(d) : ""));
#if defined(FS3_STATIC)
            info("  %s\n", submit_str().c_str());
#endif
            if (sp.guard_stopped) break;
        }
    }

#if defined(FS3_STATIC)
    row("run: longest submit under the 1,000 ms budget hint", max_submit() <= cere::vk::kMaxSubmitGpuMs, submit_str());
    row("run: memory held == planned, KV pages all free, device not faulted",
        held_now() == planned && vd.free_pages(vm) == free0 && !vd.faulted(),
        std::to_string(held_now()) + " B, pages " + std::to_string(vd.free_pages(vm)) + "/" + std::to_string(free0));
    g_after_poll = nullptr;
#endif
    std::printf("memory at end: %s\n", mem_line().c_str());
    std::fclose(g_trace);
    st = R.api.model_release(R.e, R.model);
    row("model_release", st == CERE_OK);
    row("executor_destroy", R.api.executor_destroy(R.e) == CERE_OK);
    std::printf("memory after release: %s\n", mem_line().c_str());
    std::printf("coverage_s3: %d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
