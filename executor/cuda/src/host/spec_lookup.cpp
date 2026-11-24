#include "host/spec_lookup.h"

#include <algorithm>

namespace cere::host {

namespace {

uint64_t mix(uint64_t x) {  // splitmix64 finalizer
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

double rate(double ok, double bad, double prior) {
    return (ok + prior * DraftPolicy::kPriorWeight) / (ok + bad + DraftPolicy::kPriorWeight);
}

}  // namespace

uint64_t SuffixDrafter::key_at(const uint32_t *tokens, uint64_t end) {
    const uint64_t a = tokens[end - 2];
    const uint64_t b = tokens[end - 1];
    const uint64_t c = tokens[end];
    return (mix(a | (b << 32)) ^ mix(c + 0x9e3779b97f4a7c15ull)) | 1u;
}

SuffixDrafter::Slot *SuffixDrafter::find(uint64_t key, bool insert) {
    if (table_.empty()) return nullptr;
    const uint64_t mask = table_.size() - 1;
    for (uint64_t i = key & mask;; i = (i + 1) & mask) {
        Slot &s = table_[i];
        if (s.key == key) return &s;
        if (s.key == 0) {
            if (!insert) return nullptr;
            s.key = key;
            ++used_;
            return &s;
        }
    }
}

void SuffixDrafter::insert(uint64_t key, uint32_t end) {
    if ((used_ + 1) * 2 > table_.size()) {
        std::vector<Slot> old = std::move(table_);
        table_.assign(std::max<size_t>(1024, old.size() * 2), Slot{});
        used_ = 0;
        for (const Slot &s : old) {
            if (s.key != 0) *find(s.key, true) = s;
        }
    }
    Slot &s = *find(key, true);
    for (uint32_t w = std::min(s.n, kWays - 1); w > 0; --w) s.pos[w] = s.pos[w - 1];
    s.pos[0] = end;
    s.n = std::min(s.n + 1, kWays);
}

uint32_t SuffixDrafter::propose(const std::vector<uint32_t> &tokens, uint32_t max_k, uint32_t *out, uint32_t *match) {
    *match = 0;
    const uint64_t L = tokens.size();
    const uint32_t *t = tokens.data();
    if (L < indexed_) {  // not the history this drafter indexed: start over
        table_.clear();
        used_ = 0;
        indexed_ = 0;
    }
    // Every trigram end before the last token (the current suffix's own end is not a candidate).
    for (uint64_t e = std::max<uint64_t>(indexed_, 3) - 1; e + 1 < L; ++e) insert(key_at(t, e), uint32_t(e));
    indexed_ = L;
    if (max_k == 0 || L < kMinMatch + 1) return 0;
    const Slot *s = find(key_at(t, L - 1), false);
    if (s == nullptr) return 0;
    uint32_t best = 0;
    uint64_t best_end = 0;
    for (uint32_t w = 0; w < s->n; ++w) {  // most recent first, so a strictly longer match is needed to replace
        const uint64_t e = s->pos[w];
        uint32_t m = 0;
        while (m < kMaxMatch && m <= e && t[e - m] == t[L - 1 - m]) ++m;
        if (m > best) {
            best = m;
            best_end = e;
        }
    }
    if (best < kMinMatch) return 0;  // a hash collision, or no earlier trigram
    *match = best;
    const uint64_t period = L - 1 - best_end;
    for (uint32_t j = 0; j < max_k; ++j) {
        uint64_t i = best_end + 1 + j;
        while (i >= L) i -= period;  // past the history: the proposal repeats with the match's period
        out[j] = t[i];
    }
    return max_k;
}

SpecCosts::SpecCosts() : SpecCosts(untimed()) {}

SpecCosts SpecCosts::untimed() {
    std::array<double, kMaxWindow> ms{};
    for (uint32_t t = 1; t <= kMaxWindow; ++t) ms[t - 1] = 1.0 + kUntimedSlope * double(t - 1);
    return SpecCosts(ms);
}

double SpecCosts::cost(uint32_t window) const { return ms_[std::clamp<uint32_t>(window, 1, kMaxWindow) - 1]; }

uint32_t DraftPolicy::bucket(uint32_t match) { return match < 4 ? 0 : match < 8 ? 1 : match < 16 ? 2 : 3; }

double DraftPolicy::mtp_rate(uint32_t position) const {
    const uint32_t j = std::clamp<uint32_t>(position, 1, kMaxDrafts) - 1;
    return rate(mtp_ok_[j], mtp_bad_[j], kMtpPrior);
}

double DraftPolicy::lookup_rate(uint32_t match) const {
    const uint32_t b = bucket(match);
    return rate(lookup_ok_[b], lookup_bad_[b], kLookupPrior[b]);
}

double DraftPolicy::tail_rate(uint32_t match) const {
    const uint32_t b = bucket(match);
    return rate(tail_ok_[b], tail_bad_[b], kLookupPrior[b]);
}

double DraftPolicy::lookup_q(uint32_t match) const {
    const uint32_t b = bucket(match);
    const double q = lookup_rate(match);
    return b != 0 && lookup_ok_[b] + lookup_bad_[b] < kExplore ? std::max(q, kOptimistic) : q;
}

double DraftPolicy::value(const SpecCosts &costs, Pick pick, uint32_t match) const {
    double expected = 1.0;
    double chain = 1.0;
    if (pick.source == Source::kMtp) {
        for (uint32_t n = 1; n <= pick.drafts; ++n) expected += (chain *= mtp_rate(n));
    } else if (pick.source == Source::kLookup) {
        const double q = lookup_q(match);
        for (uint32_t k = 1; k <= pick.drafts; ++k) expected += (chain *= q);
    }
    return expected / costs.cost(pick.drafts + 1);
}

DraftPolicy::Pick DraftPolicy::lengthen(const SpecCosts &costs, Pick base, const Tail &round) const {
    const uint32_t most = std::min({round.length, round.most, SpecCosts::kMaxWindow - 1});
    if (round.rows != 1 || most <= kMaxDrafts || round.match < SuffixDrafter::kMinMatch || round.lookup == nullptr) {
        return base;
    }
    const double base_value = value(costs, base, round.match);
    double best_value = base.source == Source::kLookup ? base_value : base_value * (1.0 + kMargin);
    const double q = lookup_q(round.match);
    double expected = 1.0;
    double chain = 1.0;
    Pick best = base;
    for (uint32_t k = 1; k <= most; ++k) {
        expected += (chain *= q);
        if (k <= kMaxDrafts) continue;  // choose's windows
        if (const double v = expected / costs.cost(k + 1); v > best_value) {
            best = {Source::kLookup, k};
            best_value = v;
        }
    }
    return best;
}

DraftPolicy::Pick DraftPolicy::extend(const SpecCosts &costs, Pick base, const Tail &tail) const {
    const uint32_t most = std::min({tail.length, tail.most, SpecCosts::kMaxWindow - 1});
    if (tail.rows != 1 || most <= kMaxDrafts || tail.match < SuffixDrafter::kMinMatch || tail.lookup == nullptr ||
        tail.mtp == nullptr || tail.mtp_count < kMaxDrafts ||
        !std::equal(tail.mtp, tail.mtp + kMaxDrafts, tail.lookup)) {
        return base;
    }
    const uint32_t match = tail.match;
    double best_value = value(costs, base, match);
    double expected = 1.0;
    double chain = 1.0;
    for (uint32_t n = 1; n <= kMaxDrafts; ++n) expected += (chain *= mtp_rate(n));
    const double t = tail_rate(match);
    Pick best = base;
    for (uint32_t k = kMaxDrafts + 1; k <= most; ++k) {
        expected += (chain *= t);
        if (const double v = expected / costs.cost(k + 1); v > best_value) {
            best = {Source::kTail, k};
            best_value = v;
        }
    }
    return best;
}

DraftPolicy::Pick DraftPolicy::choose(const SpecCosts &costs, bool mtp, uint32_t max_drafts, uint32_t lookup,
                                      uint32_t match) const {
    const uint32_t room = std::min(max_drafts, kMaxDrafts);
    Pick best;
    double best_value = 1.0 / costs.cost(1);
    if (mtp) {
        double expected = 1.0;
        double chain = 1.0;
        for (uint32_t n = 1; n <= room; ++n) {
            chain *= mtp_rate(n);
            expected += chain;
            const double value = expected / costs.cost(n + 1);
            if (value > best_value) {
                best = {Source::kMtp, n};
                best_value = value;
            }
        }
    }
    const uint32_t avail = std::min(lookup, room);
    if (avail != 0) {
        const double q = lookup_q(match);
        double expected = 1.0;
        double chain = 1.0;
        Pick pick;
        double value = 0.0;
        for (uint32_t k = 1; k <= avail; ++k) {
            chain *= q;
            expected += chain;
            if (const double v = expected / costs.cost(k + 1); v > value) {
                pick = {Source::kLookup, k};
                value = v;
            }
        }
        if (value > best_value * (1.0 + kMargin)) best = pick;
    }
    return best;
}

void DraftPolicy::observe(Pick pick, uint32_t accepted, bool rejected, uint32_t match) {
    for (auto *counts : {&mtp_ok_, &mtp_bad_}) {
        for (double &c : *counts) c *= kDecay;
    }
    for (auto *counts : {&lookup_ok_, &lookup_bad_, &tail_ok_, &tail_bad_}) {
        for (double &c : *counts) c *= kDecay;
    }
    if (pick.source == Source::kMtp || pick.source == Source::kTail) {
        for (uint32_t j = 0; j < accepted && j < kMaxDrafts; ++j) mtp_ok_[j] += 1.0;
        if (rejected && accepted < kMaxDrafts) mtp_bad_[accepted] += 1.0;
        if (pick.source == Source::kTail && accepted >= kMaxDrafts) {
            const uint32_t b = bucket(match);
            tail_ok_[b] += double(accepted - kMaxDrafts);
            if (rejected) tail_bad_[b] += 1.0;
        }
    } else if (pick.source == Source::kLookup) {
        const uint32_t b = bucket(match);
        lookup_ok_[b] += double(accepted);
        if (rejected) lookup_bad_[b] += 1.0;
    }
}

}  // namespace cere::host
