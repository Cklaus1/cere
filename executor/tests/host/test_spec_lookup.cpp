// m29l design note C1L, C2L: the suffix-lookup drafter and the draft policy of executor/cuda/src/host/spec_lookup.h.
#include "check.h"

#include "host/spec_calibrate.h"
#include "host/spec_lookup.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace {

using cere::host::DraftPolicy;
using cere::host::SpecCosts;
using cere::host::SuffixDrafter;
using Pick = DraftPolicy::Pick;
using Source = DraftPolicy::Source;

struct Proposal {
    std::vector<uint32_t> tokens;
    uint32_t match = 0;
};

Proposal propose(SuffixDrafter &d, const std::vector<uint32_t> &tokens, uint32_t max_k = 3) {
    uint32_t out[8] = {};
    uint32_t match = 99;
    const uint32_t n = d.propose(tokens, max_k, out, &match);
    return {std::vector<uint32_t>(out, out + n), match};
}

Proposal fresh(const std::vector<uint32_t> &tokens, uint32_t max_k = 3) {
    SuffixDrafter d;
    return propose(d, tokens, max_k);
}

// The rule of D1 by brute force: the kWays most recent earlier ends of the current trigram, the longest backward match
// (capped at kMaxMatch), the most recent on ties; the continuation repeats with the period when it reaches the end.
Proposal oracle(const std::vector<uint32_t> &t, uint32_t max_k) {
    const uint64_t L = t.size();
    if (L < SuffixDrafter::kMinMatch + 1 || max_k == 0) return {};
    std::vector<uint64_t> ends;
    for (uint64_t e = L - 1; e-- > SuffixDrafter::kMinMatch - 1;) {  // e = L - 2 down to 2
        if (t[e] == t[L - 1] && t[e - 1] == t[L - 2] && t[e - 2] == t[L - 3]) ends.push_back(e);
        if (ends.size() == SuffixDrafter::kWays) break;
    }
    uint64_t best_e = 0;
    uint32_t best = 0;
    for (const uint64_t e : ends) {
        uint32_t m = 0;
        while (m < SuffixDrafter::kMaxMatch && m <= e && t[e - m] == t[L - 1 - m]) ++m;
        if (m > best) {
            best = m;
            best_e = e;
        }
    }
    if (best < SuffixDrafter::kMinMatch) return {};
    Proposal p;
    p.match = best;
    const uint64_t period = L - 1 - best_e;
    for (uint32_t j = 0; j < max_k; ++j) {
        uint64_t i = best_e + 1 + j;
        while (i >= L) i -= period;
        p.tokens.push_back(i < L ? t[i] : 0);
    }
    return p;
}

// A deterministic token stream over `vocab` ids with copied spans (so trigrams recur, some more than kWays times).
std::vector<uint32_t> stream(uint64_t n, uint32_t vocab, uint64_t seed) {
    std::vector<uint32_t> t;
    uint64_t x = seed;
    const auto next = [&x] {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        return x >> 33;
    };
    while (t.size() < n) {
        if (t.size() > 64 && next() % 3 == 0) {
            const uint64_t len = 4 + next() % 40;
            const uint64_t from = next() % (t.size() - 1);
            for (uint64_t i = 0; i < len && t.size() < n; ++i) t.push_back(t[from + i]);
        } else {
            t.push_back(uint32_t(next() % vocab));
        }
    }
    return t;
}

void test_no_match() {
    CHECK(fresh({}).tokens.empty());
    CHECK(fresh({1, 2, 3}).tokens.empty());
    const Proposal p = fresh({1, 2, 3, 4, 5});
    CHECK(p.tokens.empty());
    CHECK_EQ(p.match, 0);
    // Only two tokens of the suffix recur: under kMinMatch.
    CHECK(fresh({2, 3, 4, 9, 1, 2, 3}).tokens.empty());
}

void test_basic_match_and_bounds() {
    const std::vector<uint32_t> t = {1, 2, 3, 4, 5, 6, 9, 1, 2, 3};
    Proposal p = fresh(t);
    CHECK(p.tokens == std::vector<uint32_t>({4, 5, 6}));
    CHECK_EQ(p.match, 3);
    CHECK(fresh(t, 1).tokens == std::vector<uint32_t>({4}));
    p = fresh(t, 0);
    CHECK(p.tokens.empty());
    CHECK_EQ(p.match, 0);
}

void test_longest_then_most_recent() {
    // "7 1 2 3" matches the first occurrence for 4 tokens, the second for 3: the longer wins although older.
    Proposal p = fresh({7, 1, 2, 3, 10, 11, 12, 8, 1, 2, 3, 20, 21, 22, 7, 1, 2, 3});
    CHECK(p.tokens == std::vector<uint32_t>({10, 11, 12}));
    CHECK_EQ(p.match, 4);
    // Two matches of 3: the most recent wins.
    p = fresh({1, 2, 3, 10, 5, 1, 2, 3, 20, 6, 1, 2, 3});
    CHECK(p.tokens == std::vector<uint32_t>({20, 6, 1}));
    CHECK_EQ(p.match, 3);
    // The match is capped at kMaxMatch.
    std::vector<uint32_t> run(80, 4);
    p = fresh(run);
    CHECK_EQ(p.match, SuffixDrafter::kMaxMatch);
}

void test_periodic_tail() {
    Proposal p = fresh({9, 4, 4, 4, 4});
    CHECK(p.tokens == std::vector<uint32_t>({4, 4, 4}));
    CHECK_EQ(p.match, 3);
    p = fresh({4, 5, 6, 4, 5, 6, 4});
    CHECK(p.tokens == std::vector<uint32_t>({5, 6, 4}));
    CHECK_EQ(p.match, 4);
    // Period 2 wrapping twice past the end.
    p = fresh({8, 1, 2, 1, 2, 1});
    CHECK(p.tokens == std::vector<uint32_t>({2, 1, 2}));
}

// Incremental indexing (one drafter over the growing history) equals a fresh drafter and the brute-force rule at every
// length, on small vocabularies (many collisions, trigrams beyond kWays) and on full-size ids.
void test_incremental_equals_oracle() {
    for (const uint32_t vocab : {3u, 20u, 248320u}) {
        const std::vector<uint32_t> t = stream(3000, vocab, vocab);
        SuffixDrafter grow;
        uint64_t proposals = 0;
        for (uint64_t L = 1; L <= t.size(); ++L) {
            const std::vector<uint32_t> prefix(t.begin(), t.begin() + L);
            const Proposal a = propose(grow, prefix);
            const Proposal want = oracle(prefix, 3);
            CHECK(a.tokens == want.tokens);
            CHECK_EQ(a.match, want.match);
            CHECK_EQ(grow.indexed(), L);
            if (L % 97 == 0) {
                const Proposal b = fresh(prefix);
                CHECK(b.tokens == a.tokens);
                CHECK_EQ(b.match, a.match);
            }
            proposals += a.tokens.empty() ? 0 : 1;
        }
        CHECK(proposals > 100);
    }
}

// A long history (the table grows many times) still matches the rule at sampled lengths.
void test_long_history() {
    const std::vector<uint32_t> t = stream(65536, 248320, 7);
    SuffixDrafter d;
    for (uint64_t L = 1024; L <= t.size(); L += 4093) {
        const std::vector<uint32_t> prefix(t.begin(), t.begin() + L);
        const Proposal a = propose(d, prefix);
        const Proposal want = oracle(prefix, 3);
        CHECK(a.tokens == want.tokens);
        CHECK_EQ(a.match, want.match);
    }
    CHECK_EQ(d.indexed(), 65536 - (65536 - 1024) % 4093);
}

bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) <= eps; }

// The fixed table whose first windows cost `head` (ms) and whose later windows follow the prior shape from the last.
SpecCosts ladder(std::initializer_list<double> head) {
    std::array<double, SpecCosts::kMaxWindow> ms{};
    std::copy(head.begin(), head.end(), ms.begin());
    const uint32_t n = uint32_t(head.size());
    for (uint32_t t = n + 1; t <= SpecCosts::kMaxWindow; ++t) {
        ms[t - 1] = ms[n - 1] * SpecCosts::shape(t) / SpecCosts::shape(n);
    }
    return SpecCosts(ms);
}

SpecCosts flat() { return ladder({10.0, 10.0, 10.0, 10.0}); }

void test_policy_priors() {
    CHECK_EQ(DraftPolicy::bucket(0), 0);
    CHECK_EQ(DraftPolicy::bucket(3), 0);
    CHECK_EQ(DraftPolicy::bucket(4), 1);
    CHECK_EQ(DraftPolicy::bucket(7), 1);
    CHECK_EQ(DraftPolicy::bucket(8), 2);
    CHECK_EQ(DraftPolicy::bucket(15), 2);
    CHECK_EQ(DraftPolicy::bucket(16), 3);
    CHECK_EQ(DraftPolicy::bucket(1000), 3);
    const DraftPolicy p;
    for (uint32_t j = 1; j <= 3; ++j) CHECK(near(p.mtp_rate(j), DraftPolicy::kMtpPrior));
    CHECK(near(p.lookup_rate(3), DraftPolicy::kLookupPrior[0]));
    CHECK(near(p.lookup_rate(20), DraftPolicy::kLookupPrior[3]));
    const SpecCosts c = flat();
    // A short match loses to the MTP, a long one wins; a one-token lookup loses to a three-token MTP window.
    Pick k = p.choose(c, true, 3, 3, 3);
    CHECK(k.source == Source::kMtp);
    CHECK_EQ(k.drafts, 3);
    k = p.choose(c, true, 3, 3, 16);
    CHECK(k.source == Source::kLookup);
    CHECK_EQ(k.drafts, 3);
    // Even an unexplored bucket's optimistic rate (kOptimistic) leaves a one-token lookup below a three-token MTP.
    k = p.choose(c, true, 3, 1, 16);
    CHECK(k.source == Source::kMtp);
    // The round's room bounds both sources.
    k = p.choose(c, true, 1, 3, 3);
    CHECK(k.source == Source::kMtp);
    CHECK_EQ(k.drafts, 1);
    k = p.choose(c, true, 0, 3, 16);
    CHECK(k.source == Source::kNone);
    CHECK_EQ(k.drafts, 0);
    k = p.choose(c, true, 9, 0, 0);
    CHECK_EQ(k.drafts, 3);
    // Without the MTP: the lookup if any, else none.
    k = p.choose(c, false, 3, 2, 3);
    CHECK(k.source == Source::kLookup);
    CHECK_EQ(k.drafts, 2);
    k = p.choose(c, false, 3, 0, 0);
    CHECK(k.source == Source::kNone);
    CHECK_EQ(k.drafts, 0);
}

void test_policy_cost() {
    // Each extra verify row costs twice a 1-row round: no window pays for itself.
    const SpecCosts c = ladder({1.0, 3.0, 5.0, 7.0});
    const DraftPolicy p;
    Pick k = p.choose(c, true, 3, 3, 20);
    CHECK(k.source == Source::kNone);
    CHECK_EQ(k.drafts, 0);
    // Moderately steep: the MTP's first draft only (E(1)/c(2) = 1.68/1.5 beats E(2)/c(3) = 2.14/2.2 and none).
    const SpecCosts m = ladder({1.0, 1.5, 2.2, 3.0});
    k = p.choose(m, true, 3, 0, 0);
    CHECK(k.source == Source::kMtp);
    CHECK_EQ(k.drafts, 1);
}

void test_policy_explores() {
    const SpecCosts c = flat();
    // The shortest matches (bucket 0) are never explored: their prior loses to the MTP.
    DraftPolicy p;
    CHECK(p.choose(c, true, 3, 3, 3).source == Source::kMtp);
    // A medium match without evidence is tried; one rejected round is evidence, and the MTP wins again.
    CHECK(p.choose(c, true, 3, 3, 5).source == Source::kLookup);
    p.observe({Source::kLookup, 3}, 0, true, 5);
    CHECK(p.choose(c, true, 3, 3, 5).source == Source::kMtp);
    // Once that evidence has decayed below kExplore (rounds that verify nothing still decay it), it is tried again.
    int rounds = 0;
    while (p.choose(c, true, 3, 3, 5).source == Source::kMtp && rounds < 1000) {
        p.observe({Source::kNone, 0}, 0, false, 0);
        ++rounds;
    }
    CHECK(rounds > 1 && rounds < 1000);
    const Pick again = p.choose(c, true, 3, 3, 5);
    CHECK(again.source == Source::kLookup);
    CHECK_EQ(again.drafts, 3);
}

void test_policy_learns() {
    const SpecCosts c = flat();
    DraftPolicy p;
    // Long matches that are always rejected at the first draft: q falls until the MTP wins again.
    for (int i = 0; i < 60; ++i) p.observe({Source::kLookup, 3}, 0, true, 20);
    CHECK(p.lookup_rate(20) < 0.15);
    CHECK(near(p.lookup_rate(3), DraftPolicy::kLookupPrior[0], 1e-12));  // other buckets untouched
    CHECK(p.choose(c, true, 3, 3, 20).source == Source::kMtp);
    // MTP rounds that keep the first draft and reject the second: p1 up, p2 down, p3 unobserved (prior).
    for (int i = 0; i < 200; ++i) p.observe({Source::kMtp, 3}, 1, true, 0);
    CHECK(p.mtp_rate(1) > 0.95);
    CHECK(p.mtp_rate(2) < 0.1);
    CHECK(near(p.mtp_rate(3), DraftPolicy::kMtpPrior, 1e-12));
    // Meanwhile the lookup's counts decayed back towards its prior, so a long match is tried again.
    CHECK(p.lookup_rate(20) > DraftPolicy::kLookupPrior[3] - 0.01);
    CHECK(p.choose(c, true, 3, 3, 20).source == Source::kLookup);
    // A truncated round (a stop token or the commit limit, no rejection) counts its accepted drafts only.
    DraftPolicy t;
    for (int i = 0; i < 60; ++i) t.observe({Source::kLookup, 3}, 1, false, 5);
    CHECK(t.lookup_rate(5) > 0.95);
    // Accepted lookup drafts lift q above the MTP's E: the lookup wins on a medium match (after one rejected round
    // ends its exploration).
    DraftPolicy w;
    w.observe({Source::kLookup, 3}, 0, true, 5);
    CHECK(w.choose(c, true, 3, 3, 5).source == Source::kMtp);
    for (int i = 0; i < 20; ++i) w.observe({Source::kLookup, 3}, 3, false, 5);
    CHECK(w.choose(c, true, 3, 3, 5).source == Source::kLookup);
}

// m29a design note A3: a fixed table prices each window at its entry (windows clamped to 1..kMaxWindow) and
// table() reads its entries back. O95: untimed() is the fixed table 1 + kUntimedSlope (t - 1), and the default table.
// batch-concurrency design note BC-D4: a table has no observe(); nothing a round measures changes it.
void test_costs_fixed() {
    const std::array<double, SpecCosts::kMaxWindow> want = {1.0, 1.5, 2.25, 4.0, 4.5,  5.0, 5.5, 6.0,
                                                            6.5, 7.0, 7.5,  8.0, 8.5, 9.0, 9.5, 10.0};
    const SpecCosts c(want);
    for (uint32_t t = 1; t <= SpecCosts::kMaxWindow; ++t) CHECK(near(c.cost(t), want[t - 1]));
    CHECK(near(c.cost(0), 1.0));
    CHECK(near(c.cost(SpecCosts::kMaxWindow + 1), 10.0));
    CHECK(c.table() == want);
    const SpecCosts untimed = SpecCosts::untimed();
    for (uint32_t t = 1; t <= SpecCosts::kMaxWindow; ++t) {
        CHECK(near(untimed.cost(t), 1.0 + SpecCosts::kUntimedSlope * (t - 1)));
        CHECK(near(untimed.table()[t - 1], 1.0 + SpecCosts::kUntimedSlope * (t - 1)));
    }
    CHECK(SpecCosts().table() == untimed.table());
}

// O95 / m23 design note SC-01: derive_round_costs takes each mtp window's median, scales the chain windows by the
// medians' ratio at kMtpWindows, extrapolates past the model's deepest window (Chains: by the measured chain windows'
// per-row slope, the prior shape when the deepest is kMtpWindows or that slope is not positive), and refuses an
// empty series, a non-positive median or a window outside [kMtpWindows, kMaxWindow].
void test_derive_round_costs() {
    using cere::host::SpecCalibration;
    using cere::host::SpecRoundTimes;
    constexpr uint32_t kMtp = SpecCalibration::kMtpWindows;
    SpecRoundTimes times;
    for (uint32_t t = 1; t <= kMtp; ++t) times.mtp[t] = {10.0 * t + 5.0, 10.0 * t, 10.0 * t + 1.0, 10.0 * t - 1.0};
    for (uint32_t t = kMtp; t <= SpecCosts::kMaxWindow; ++t) times.chain[t] = {20.0 * t, 20.0 * t + 2.0, 20.0 * t - 2.0};
    SpecCosts got;
    CHECK(cere::host::derive_round_costs(times, SpecCosts::kMaxWindow, &got));
    // mtp medians 10t + 0.5 (even count: the two middle samples' mean); chain t costs 20t x (40.5 / 80).
    for (uint32_t t = 1; t <= kMtp; ++t) CHECK(near(got.cost(t), 10.0 * t + 0.5));
    for (uint32_t t = kMtp + 1; t <= SpecCosts::kMaxWindow; ++t) CHECK(near(got.cost(t), 20.0 * t * 40.5 / 80.0));
    // A model whose chains stop at kMtpWindows: the deeper windows follow the prior shape from window kMtpWindows.
    SpecCosts shallow;
    CHECK(cere::host::derive_round_costs(times, kMtp, &shallow));
    for (uint32_t t = kMtp + 1; t <= SpecCosts::kMaxWindow; ++t) {
        CHECK(near(shallow.cost(t), 40.5 * SpecCosts::shape(t) / SpecCosts::shape(kMtp)));
    }
    // Chains: a model calibrated to window 9 prices windows 10..16 by the slope of windows 4..9 (linear here, so the
    // extrapolation is the series itself), and by the shape when the chain windows do not grow.
    SpecCosts nine;
    CHECK(cere::host::derive_round_costs(times, 9, &nine));
    for (uint32_t t = 1; t <= SpecCosts::kMaxWindow; ++t) CHECK(near(nine.cost(t), got.cost(t)));
    SpecRoundTimes flat = times;
    for (uint32_t t = kMtp; t <= 9; ++t) flat.chain[t] = {80.0};
    SpecCosts level;
    CHECK(cere::host::derive_round_costs(flat, 9, &level));
    for (uint32_t t = 10; t <= SpecCosts::kMaxWindow; ++t) {
        CHECK(near(level.cost(t), 40.5 * SpecCosts::shape(t) / SpecCosts::shape(9)));
    }
    std::array<double, SpecCosts::kMaxWindow> ones{};
    for (uint32_t t = 1; t <= SpecCosts::kMaxWindow; ++t) ones[t - 1] = double(t);
    const SpecCosts untouched(ones);
    SpecCosts out = untouched;
    CHECK(!cere::host::derive_round_costs(times, kMtp - 1, &out));
    CHECK(!cere::host::derive_round_costs(times, SpecCosts::kMaxWindow + 1, &out));
    SpecRoundTimes missing = times;
    missing.chain[7].clear();
    CHECK(!cere::host::derive_round_costs(missing, SpecCosts::kMaxWindow, &out));
    CHECK(cere::host::derive_round_costs(missing, 6, &out));  // window 7 is past the model's deepest
    out = untouched;
    SpecRoundTimes zero = times;
    zero.mtp[2] = {0.0, 0.0, 0.0};
    CHECK(!cere::host::derive_round_costs(zero, kMtp, &out));
    for (uint32_t t = 1; t <= SpecCosts::kMaxWindow; ++t) CHECK(near(out.cost(t), double(t)));
}

// DraftAccept (CERE_WORK_TAIL_DRAFTS): DraftPolicy::extend keeps `base` without a tail longer than the MTP drafts,
// appends the tail when E(k) = 1 + p1 + p1 p2 + p1 p2 p3 (1 + t + .. + t^(k-3)) over cost(k + 1) beats the base pick
// (a long match's prior t = 0.88 on a flat table), stops paying once the tail's bucket keeps rejecting, and observing a
// kTail round credits the MTP positions and the tail's bucket. DraftTail: the round decides the tail, so without a
// lookup match (none, shorter than kMinMatch, or one that disagrees with the pending MTP proposals), without three
// pending proposals or with another drafting row in the call it keeps `base` whatever the table says.
void test_policy_tail() {
    using Tail = DraftPolicy::Tail;
    const SpecCosts flat({1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4});
    DraftPolicy p;
    const Pick base = p.choose(flat, true, 3, 0, 0);
    CHECK(base.source == Source::kMtp);
    CHECK_EQ(base.drafts, 3);
    const uint32_t lookup[8] = {7, 8, 9, 10, 11, 12, 13, 14};
    const uint32_t agree[3] = {7, 8, 9};
    const uint32_t differ[3] = {7, 8, 5};
    const auto round = [&](uint32_t length, uint32_t match, const uint32_t *mtp, uint32_t count, uint32_t rows) {
        return Tail{lookup, length, match, mtp, count, rows};
    };
    CHECK(p.extend(flat, base, round(3, 20, agree, 3, 1)).source == Source::kMtp);
    CHECK(p.extend(flat, base, round(0, 0, agree, 3, 1)).source == Source::kMtp);
    const Pick tail = p.extend(flat, base, round(8, 20, agree, 3, 1));
    CHECK(tail.source == Source::kTail);
    CHECK_EQ(tail.drafts, 8);
    CHECK_EQ(p.extend(flat, base, round(6, 20, agree, 3, 1)).drafts, 6);
    // No match: a short match, a lookup that disagrees with the MTP chain, fewer than three proposals, no proposal.
    CHECK(p.extend(flat, base, round(8, SuffixDrafter::kMinMatch - 1, agree, 3, 1)).source == Source::kMtp);
    CHECK(p.extend(flat, base, round(8, 20, differ, 3, 1)).source == Source::kMtp);
    CHECK(p.extend(flat, base, round(8, 20, agree, 2, 1)).source == Source::kMtp);
    CHECK(p.extend(flat, base, round(8, 20, nullptr, 0, 1)).source == Source::kMtp);
    CHECK(p.extend(flat, base, Tail{nullptr, 8, 20, agree, 3, 1}).source == Source::kMtp);
    // Another drafting row in the call: the single-row table does not price the longer verify.
    for (const uint32_t rows : {0u, 2u, 4u}) {
        CHECK(p.extend(flat, base, round(8, 20, agree, 3, rows)).source == Source::kMtp);
    }
    // A non-MTP base comes back unchanged without a match.
    const Pick look{Source::kLookup, 2};
    const Pick kept = p.extend(flat, look, round(8, 20, differ, 3, 1));
    CHECK(kept.source == Source::kLookup);
    CHECK_EQ(kept.drafts, 2);
    // A steep table prices the tail out.
    const SpecCosts steep({1.0, 1.05, 1.1, 1.15, 3.0, 4.0, 5.0, 6.0, 7.0});
    CHECK(p.extend(steep, base, round(8, 20, agree, 3, 1)).source == Source::kMtp);
    // Every tail draft rejected: the bucket's rate falls and the tail stops paying; MTP evidence kept.
    for (int i = 0; i < 40; ++i) p.observe(tail, 3, true, 20);
    CHECK(p.tail_rate(20) < 0.1);
    CHECK(p.mtp_rate(3) > DraftPolicy::kMtpPrior);
    CHECK(p.extend(flat, base, round(8, 20, agree, 3, 1)).source == Source::kMtp);
    // Another bucket keeps its prior.
    CHECK(near(p.tail_rate(5), DraftPolicy::kLookupPrior[1]));
}

// TreeSpec (CERE_WORK_LONG_DRAFTS, tree-spec design note TS-01): DraftPolicy::lengthen widens choose's pick to the
// lookup's first k > 3 drafts (k <= min(length, kMaxTailDrafts)) when E(k) = 1 + q + .. + q^k over cost(k + 1) beats
// the base pick (an unexplored long match is valued at kOptimistic 0.9 on a flat table: E(8)/c(9) = 6.13/1.4 against
// the MTP's 2.46/1.15), from an MTP base or from choose's own 3-draft lookup window, without MTP agreement. It keeps
// `base` with no more than 3 lookup drafts, a match shorter than kMinMatch, no proposal, another drafting row in the
// call, a short match's prior (bucket 0, q 0.45, never explored), a steep table, or once the bucket keeps rejecting.
void test_policy_long() {
    using Tail = DraftPolicy::Tail;
    const SpecCosts flat({1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4});
    DraftPolicy p;
    const uint32_t lookup[8] = {7, 8, 9, 10, 11, 12, 13, 14};
    const auto round = [&](uint32_t length, uint32_t match, uint32_t rows) {
        return Tail{lookup, length, match, nullptr, 0, rows};
    };
    const Pick mtp = p.choose(flat, true, 3, 0, 0);
    CHECK(mtp.source == Source::kMtp);
    CHECK_EQ(mtp.drafts, 3);
    Pick k = p.lengthen(flat, mtp, round(8, 20, 1));
    CHECK(k.source == Source::kLookup);
    CHECK_EQ(k.drafts, 8);
    CHECK_EQ(p.lengthen(flat, mtp, round(6, 20, 1)).drafts, 6);
    // From choose's own lookup window (3 drafts of the same proposal).
    const Pick look = p.choose(flat, true, 3, 8, 20);
    CHECK(look.source == Source::kLookup);
    CHECK_EQ(look.drafts, 3);
    k = p.lengthen(flat, look, round(8, 20, 1));
    CHECK(k.source == Source::kLookup);
    CHECK_EQ(k.drafts, 8);
    // Kept: no proposal past 3 drafts, a short match, none, other rows in the call, bucket 0's prior.
    for (const Tail &r : {round(3, 20, 1), round(0, 0, 1), round(8, SuffixDrafter::kMinMatch - 1, 1),
                          Tail{nullptr, 8, 20, nullptr, 0, 1}, round(8, 20, 0), round(8, 20, 2), round(8, 3, 1)}) {
        k = p.lengthen(flat, mtp, r);
        CHECK(k.source == Source::kMtp);
        CHECK_EQ(k.drafts, 3);
    }
    // A steep table prices the long chain out.
    const SpecCosts steep({1.0, 1.05, 1.1, 1.15, 3.0, 4.0, 5.0, 6.0, 7.0});
    CHECK(p.lengthen(steep, mtp, round(8, 20, 1)).source == Source::kMtp);
    // Long lookup rounds whose first draft is always rejected: the bucket's rate falls and the chain stops paying.
    for (int i = 0; i < 40; ++i) p.observe({Source::kLookup, 8}, 0, true, 20);
    CHECK(p.lookup_rate(20) < 0.1);
    CHECK(p.lengthen(flat, mtp, round(8, 20, 1)).source == Source::kMtp);
    // Fully accepted ones credit the bucket with every accepted draft.
    DraftPolicy q;
    for (int i = 0; i < 20; ++i) q.observe({Source::kLookup, 8}, 8, false, 20);
    CHECK(q.lookup_rate(20) > 0.99);
    CHECK_EQ(q.lengthen(flat, mtp, round(8, 20, 1)).drafts, 8);
    // Chains (CERE_WORK_CHAIN_DRAFTS, O123): `most` caps the chain (kMaxTailDrafts by default) up to kMaxWindow - 1;
    // the same rule prices the longer windows, and a steep tail of the table keeps the chain short. The tail's prior
    // (t = 0.88) pays to 12 drafts only on a near-flat table (0.01 per row: each draft adds > 1 % of E).
    std::array<double, SpecCosts::kMaxWindow> grow{};
    for (uint32_t t = 1; t <= SpecCosts::kMaxWindow; ++t) grow[t - 1] = 1.0 + 0.05 * (t - 1);
    const SpecCosts wide(grow);
    const uint32_t chain[15] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    const uint32_t agree[3] = {7, 8, 9};
    CHECK_EQ(q.lengthen(wide, mtp, Tail{chain, 15, 20, nullptr, 0, 1}).drafts, DraftPolicy::kMaxTailDrafts);
    CHECK_EQ(q.lengthen(wide, mtp, Tail{chain, 15, 20, nullptr, 0, 1, 12}).drafts, 12);
    CHECK_EQ(q.lengthen(wide, mtp, Tail{chain, 15, 20, nullptr, 0, 1, 15}).drafts, 15);
    CHECK_EQ(q.lengthen(wide, mtp, Tail{chain, 10, 20, nullptr, 0, 1, 15}).drafts, 10);
    CHECK_EQ(q.lengthen(wide, mtp, Tail{chain, 15, 20, nullptr, 0, 1, 40}).drafts, SpecCosts::kMaxWindow - 1);
    std::array<double, SpecCosts::kMaxWindow> gentle{};
    for (uint32_t w = 1; w <= SpecCosts::kMaxWindow; ++w) gentle[w - 1] = 1.0 + 0.01 * (w - 1);
    const SpecCosts near_flat(gentle);
    const Pick tail = q.extend(near_flat, mtp, Tail{chain, 15, 20, agree, 3, 1, 12});
    CHECK(tail.source == Source::kTail);
    CHECK_EQ(tail.drafts, 12);
    CHECK_EQ(q.extend(near_flat, mtp, Tail{chain, 15, 20, agree, 3, 1}).drafts, DraftPolicy::kMaxTailDrafts);
    CHECK_EQ(q.extend(wide, mtp, Tail{chain, 15, 20, agree, 3, 1, 12}).drafts, 10);  // E(k)/c peaks at k = 10
    std::array<double, SpecCosts::kMaxWindow> cliff = grow;
    for (uint32_t w = 10; w <= SpecCosts::kMaxWindow; ++w) cliff[w - 1] = 10.0;
    CHECK_EQ(q.lengthen(SpecCosts(cliff), mtp, Tail{chain, 15, 20, nullptr, 0, 1, 15}).drafts, 8);
}

// m29a design note A2: the adaptive policy (DraftPolicy priced by a fixed table) picks the draft count from the
// request's own MTP acceptance: the full window at the prior on both K2 tables (default, A3 [REVISED: DefaultAll]:
// E(3)/c(4) = 2.46/18.80 beats E(2)/c(3) = 2.14/17.78), the full window while every draft is accepted, one draft
// when the second is always rejected, none when the first always is, and drafts again once that evidence decays. Two
// policies fed the same outcomes pick the same windows. The tables are the last compiled ones (library c772467a +
// 588cb097's deep windows, model_k8 "A3 window t" / "A3D window t"; O95 replaced them by the seal-time tables).
void test_policy_adaptive() {
    const SpecCosts k({15.279, 16.533, 17.778, 18.803, 20.972, 21.851, 22.453, 23.225, 25.891});
    const SpecCosts small_m({13.665, 14.778, 15.880, 16.887, 18.058, 18.898, 19.596, 20.362, 21.351});
    CHECK_EQ(DraftPolicy().choose(small_m, true, 3, 0, 0).drafts, 3);
    DraftPolicy p;
    Pick prior = p.choose(k, true, 3, 0, 0);
    CHECK(prior.source == Source::kMtp);
    CHECK_EQ(prior.drafts, 3);
    // A third draft accepted one round in six (p3 about 0.21) pays on both tables of the L2+L3 library (window 4 at
    // 1.23-1.24x window 1; DG-01's c54ab336 tables, at 1.29x, verified two drafts); one round in eight (p3 about 0.18)
    // no longer pays on the default table.
    DraftPolicy sixth;
    for (int i = 0; i < 100; ++i) sixth.observe({Source::kMtp, 3}, i % 6 == 0 ? 3 : 2, i % 6 != 0, 0);
    CHECK_EQ(sixth.choose(k, true, 3, 0, 0).drafts, 3);
    CHECK_EQ(sixth.choose(small_m, true, 3, 0, 0).drafts, 3);
    DraftPolicy third;
    for (int i = 0; i < 100; ++i) third.observe({Source::kMtp, 3}, i % 8 == 0 ? 3 : 2, i % 8 != 0, 0);
    CHECK_EQ(third.choose(k, true, 3, 0, 0).drafts, 2);
    for (int i = 0; i < 50; ++i) p.observe({Source::kMtp, 3}, 3, false, 0);
    CHECK_EQ(p.choose(k, true, 3, 0, 0).drafts, 3);
    CHECK_EQ(p.choose(k, true, 1, 0, 0).drafts, 1);  // the round's room still bounds it
    DraftPolicy one;
    for (int i = 0; i < 100; ++i) one.observe({Source::kMtp, 3}, 1, true, 0);
    const Pick k1 = one.choose(k, true, 3, 0, 0);
    CHECK(k1.source == Source::kMtp);
    CHECK_EQ(k1.drafts, 1);
    // The first draft always rejected: no window pays, so none is verified; MTP evidence then only decays (a round
    // without drafts observes nothing), and the policy tries the MTP again.
    DraftPolicy none;
    for (int i = 0; i < 100; ++i) none.observe({Source::kMtp, 3}, 0, true, 0);
    CHECK(none.choose(k, true, 3, 0, 0).source == Source::kNone);
    int rounds = 0;
    while (none.choose(k, true, 3, 0, 0).source == Source::kNone && rounds < 1000) {
        none.observe({Source::kNone, 0}, 0, false, 0);
        ++rounds;
    }
    CHECK(rounds > 1 && rounds < 1000);
    // Determinism: the same outcome stream gives the same picks.
    DraftPolicy a, b;
    const uint32_t outcomes[] = {3, 0, 1, 2, 0, 0, 3, 1, 0, 0, 0, 2};
    for (int i = 0; i < 200; ++i) {
        const Pick pa = a.choose(k, true, 3, 0, 0);
        const Pick pb = b.choose(k, true, 3, 0, 0);
        CHECK(pa.source == pb.source);
        CHECK_EQ(pa.drafts, pb.drafts);
        const uint32_t acc = std::min(outcomes[i % 12], pa.drafts);
        a.observe(pa, acc, acc < pa.drafts, 0);
        b.observe(pb, acc, acc < pb.drafts, 0);
    }
}

}  // namespace

int main() {
    test_no_match();
    test_basic_match_and_bounds();
    test_longest_then_most_recent();
    test_periodic_tail();
    test_incremental_equals_oracle();
    test_long_history();
    test_policy_priors();
    test_policy_explores();
    test_policy_cost();
    test_derive_round_costs();
    test_policy_learns();
    test_costs_fixed();
    test_policy_adaptive();
    test_policy_tail();
    test_policy_long();
    return cere::test::finish("test_spec_lookup");
}
