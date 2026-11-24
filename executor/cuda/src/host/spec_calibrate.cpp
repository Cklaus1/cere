#include "host/spec_calibrate.h"

#include <algorithm>
#include <chrono>
#include <memory>

namespace cere::host {

namespace {

using Calib = SpecCalibration;

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    if (v.size() % 2 == 1) return v[mid];
    const double upper = v[mid];
    return (*std::max_element(v.begin(), v.begin() + mid) + upper) / 2.0;
}

// kPeriod seeded ordinary ids repeated: [1000, 100000) on a full vocabulary (the Qwen vocabularies' special ids sit
// above 151,642), the whole vocabulary on a small one (the CPU toy's).
std::vector<uint32_t> periodic_prompt(uint32_t vocab) {
    const bool full = vocab > 100000;
    const uint64_t lo = full ? 1000 : 0;
    const uint64_t span = (full ? 100000 : vocab) - lo;
    std::vector<uint32_t> ids(Calib::kPromptTokens);
    uint64_t x = Calib::kPromptSeed;
    for (uint32_t i = 0; i < Calib::kPeriod; ++i) {
        x += 0x9e3779b97f4a7c15ull;  // splitmix64
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        ids[i] = uint32_t(lo + (z ^ (z >> 31)) % span);
    }
    for (uint32_t i = Calib::kPeriod; i < Calib::kPromptTokens; ++i) ids[i] = ids[i - Calib::kPeriod];
    return ids;
}

// One single-row round, timed as spec_step times it; committed in [1, max_commit].
cere_status timed_round(Device &device, DeviceModel &model, const SpecRowArgs &row, SpecRowResult *res, double *ms) {
    cere_status row_status = CERE_ERR_ARGUMENT;
    const auto started = std::chrono::steady_clock::now();
    const cere_status st = device.speculate_rows(model, &row, 1, res, &row_status);
    const std::chrono::duration<double, std::milli> took = std::chrono::steady_clock::now() - started;
    *ms = took.count();
    if (st != CERE_OK) return st;
    if (row_status != CERE_OK) return row_status == CERE_PENDING ? CERE_ERR_INTERNAL : row_status;
    if (res->committed == 0 || res->committed > row.max_commit) return CERE_ERR_INTERNAL;
    return CERE_OK;
}

}  // namespace

bool derive_round_costs(const SpecRoundTimes &times, uint32_t max_window, SpecCosts *out) {
    constexpr uint32_t kMtp = Calib::kMtpWindows;
    if (max_window < kMtp || max_window > SpecCosts::kMaxWindow) return false;
    std::array<double, SpecCosts::kMaxWindow> ms{};
    for (uint32_t t = 1; t <= kMtp; ++t) {
        ms[t - 1] = median(times.mtp[t]);
        if (!(ms[t - 1] > 0.0)) return false;
    }
    if (max_window > kMtp) {
        const double base = median(times.chain[kMtp]);
        if (!(base > 0.0)) return false;
        for (uint32_t t = kMtp + 1; t <= max_window; ++t) {
            const double chain = median(times.chain[t]);
            if (!(chain > 0.0)) return false;
            ms[t - 1] = chain * ms[kMtp - 1] / base;
        }
    }
    // Chains (O123): past a measured chain window, the measured chain windows' per-row slope (a verify row costs about
    // the same at every depth, chains design note F1); the prior shape when there is none or it is not positive.
    const double slope =
        max_window > kMtp ? (ms[max_window - 1] - ms[kMtp - 1]) / double(max_window - kMtp) : 0.0;
    for (uint32_t t = max_window + 1; t <= SpecCosts::kMaxWindow; ++t) {
        ms[t - 1] = slope > 0.0 ? ms[max_window - 1] + slope * double(t - max_window)
                                : ms[max_window - 1] * SpecCosts::shape(t) / SpecCosts::shape(max_window);
    }
    *out = SpecCosts(ms);
    return true;
}

uint32_t calibration_window(const Device &device, const DeviceModel &model, const ModelPlan &plan) {
    if (plan.calibrate_windows == CERE_CALIBRATE_MTP) return Calib::kMtpWindows;
    const uint32_t window = plan.levers.draft_tokens != 0 ? plan.levers.draft_tokens : plan.speculative_draft_tokens;
    if (window < DraftPolicy::kMaxDrafts + 1) return Calib::kMtpWindows;
    return std::min(SpecCosts::kMaxWindow, std::min(device.max_chain_drafts(model), DraftPolicy::kMaxTailDrafts) + 1);
}

cere_status measure_round_costs(Device &device, DeviceModel &model, const ModelPlan &plan, SpecRoundTimes *out) {
    constexpr uint32_t kMtp = Calib::kMtpWindows;
    const uint32_t deepest = calibration_window(device, model, plan);
    if (deepest < kMtp) return CERE_ERR_UNSUPPORTED;
    uint64_t mtp_cycle = 0;  // tokens one mtp cycle commits at most
    for (uint32_t t = 1; t <= kMtp; ++t) mtp_cycle += t;
    uint64_t chain_cycle = 0;  // tokens one chain cycle commits at most (0: no chain window)
    for (uint32_t t = kMtp; deepest > kMtp && t <= deepest; ++t) chain_cycle += t;
    // The stream the chain session replays from its first token (the mtp rounds' tokens included): kChainCycles chain
    // cycles; the mtp session may overshoot it by kMtp - 1.
    const uint64_t stream = chain_cycle == 0 ? 0 : Calib::kChainCycles * chain_cycle + 1;
    const uint64_t context = Calib::kPromptTokens + std::max<uint64_t>(Calib::kMaxCycles * mtp_cycle, stream) + kMtp;
    if (context > plan.max_context_tokens) return CERE_ERR_CAPACITY;

    // A drafting session of `context` fed the synthetic prompt; *first is its greedy next token.
    const std::vector<uint32_t> prompt = periodic_prompt(plan.config.vocab_size);
    const auto open = [&](std::unique_ptr<DeviceSession> *session, uint32_t *first) -> cere_status {
        if (const cere_status st = device.create_session(model, context, false, session); st != CERE_OK) return st;
        if (const cere_status st = device.set_session_draft(model, **session, true); st != CERE_OK) return st;
        for (uint64_t fed = 0; fed < Calib::kPromptTokens;) {
            const uint64_t count = Calib::kPromptTokens - fed;
            uint64_t consumed = 0;
            cere_token_score score{};
            const cere_status st = device.step(model, **session, prompt.data() + fed, count, 0, nullptr, &consumed,
                                               nullptr, &score, nullptr, nullptr, nullptr);
            if (st != CERE_OK) return st == CERE_PENDING ? CERE_ERR_INTERNAL : st;
            if (consumed == 0 || consumed > count) return CERE_ERR_INTERNAL;
            fed += consumed;
            if (fed == Calib::kPromptTokens) *first = score.id;
        }
        return CERE_OK;
    };
    SpecRoundTimes times;
    std::vector<uint32_t> history;  // the greedy stream after the prompt
    history.reserve(context);
    {
        std::unique_ptr<DeviceSession> session;
        uint32_t first = 0;
        if (const cere_status st = open(&session, &first); st != CERE_OK) return st;
        history.push_back(first);
        const auto round = [&](uint32_t t, double *ms) -> cere_status {
            SpecRowArgs row;
            row.session = session.get();
            row.token = history.back();
            row.draft_count = t - 1;
            row.max_commit = t;
            SpecRowResult res;
            if (const cere_status st = timed_round(device, model, row, &res, ms); st != CERE_OK) return st;
            for (uint32_t i = 0; i < res.committed; ++i) history.push_back(res.scores[i].id);
            return CERE_OK;
        };
        const auto pending = [&] {
            uint32_t ids[DraftPolicy::kMaxDrafts] = {};
            return device.pending_drafts(*session, ids, DraftPolicy::kMaxDrafts);
        };
        const auto mtp_full = [&] {
            for (uint32_t t = 1; t <= kMtp; ++t) {
                if (times.mtp[t].size() < Calib::kRounds) return false;
            }
            return true;
        };
        std::array<bool, kMtp + 1> warm{};  // the window's first round ran (its graph capture, not timed)
        for (uint32_t cycle = 0; cycle < Calib::kMaxCycles && !mtp_full(); ++cycle) {
            for (uint32_t t = 1; t <= kMtp; ++t) {
                if (t > 1 && pending() < t - 1) continue;
                double ms = 0.0;
                if (const cere_status st = round(t, &ms); st != CERE_OK) return st;
                if (warm[t]) times.mtp[t].push_back(ms);
                warm[t] = true;
            }
        }
        if (!mtp_full()) return CERE_ERR_UNSUPPORTED;
        while (history.size() < stream) {
            double ms = 0.0;
            if (const cere_status st = round(1 + std::min(pending(), kMtp - 1), &ms); st != CERE_OK) return st;
        }
    }
    if (chain_cycle > 0) {
        std::unique_ptr<DeviceSession> session;
        uint32_t first = 0;
        if (const cere_status st = open(&session, &first); st != CERE_OK) return st;
        if (first != history[0]) return CERE_ERR_UNSUPPORTED;
        const auto chain_full = [&] {
            for (uint32_t t = kMtp; t <= deepest; ++t) {
                if (times.chain[t].size() < Calib::kRounds) return false;
            }
            return true;
        };
        std::array<bool, SpecCosts::kMaxWindow + 1> warm{};
        size_t pos = 0;  // the session's last token is history[pos]
        for (uint32_t cycle = 0; cycle < Calib::kMaxCycles && !chain_full(); ++cycle) {
            for (uint32_t t = kMtp; t <= deepest; ++t) {
                if (pos + t > history.size()) break;
                SpecRowArgs row;
                row.session = session.get();
                row.token = history[pos];
                row.drafts = history.data() + pos + 1;
                row.draft_count = t - 1;
                row.max_commit = t;
                SpecRowResult res;
                double ms = 0.0;
                if (const cere_status st = timed_round(device, model, row, &res, &ms); st != CERE_OK) return st;
                for (uint32_t i = 0; i < res.committed; ++i) {
                    if (pos + 1 + i >= history.size() || res.scores[i].id != history[pos + 1 + i]) {
                        return CERE_ERR_UNSUPPORTED;
                    }
                }
                pos += res.committed;
                if (warm[t] && res.committed == t) times.chain[t].push_back(ms);
                warm[t] = true;
            }
        }
        if (!chain_full()) return CERE_ERR_UNSUPPORTED;
    }
    *out = std::move(times);
    return CERE_OK;
}

cere_status calibrate_round_costs(Device &device, DeviceModel &model, const ModelPlan &plan,
                                  std::array<SpecCosts, 2> *out) {
    const uint32_t own = k2_cost_table(plan.levers.k2_gemm);
    std::array<SpecCosts, 2> tables{SpecCosts::untimed(), SpecCosts::untimed()};
    // Eager rounds: a graph instantiated here keeps device memory after it is destroyed (GPU1 probe, SC-01), and the
    // eager table matches the graph path's within 1 % per window.
    Levers eager = plan.levers;
    eager.decode_step = CERE_LEVER_DECODE_EAGER;
    cere_status st = device.set_levers(model, eager);
    SpecRoundTimes times;
    if (st == CERE_OK) st = measure_round_costs(device, model, plan, &times);
    if (st == CERE_OK && !derive_round_costs(times, calibration_window(device, model, plan), &tables[own])) {
        st = CERE_ERR_UNSUPPORTED;
    }
    const cere_status restored = device.set_levers(model, plan.levers);
    if (st != CERE_OK) return st;
    if (restored != CERE_OK) return restored;
    *out = tables;
    return CERE_OK;
}

}  // namespace cere::host
