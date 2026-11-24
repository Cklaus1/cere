// M2.9L (m29l design note D1-D3): the suffix-lookup drafter and the per-request draft policy of a drafting session
// whose ticket carries CERE_WORK_LOOKUP_DRAFTS. Host-only, no CUDA: the lookup drafts it picks go through
// Device::speculate_rows as caller drafts, whose verify, accept and commit are the VERIFY path's (D4), so the committed
// tokens are plain decode's whatever it picks. Adapted from Niko1221/Strata@fb58e0d:include/strata/spec/suffix_drafter.hpp
// and Niko1221/Strata@fb58e0d:include/strata/spec/draft_policy.hpp (one window of at most 3 drafts here, D2).
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace cere::host {

// Proposes the tokens that followed the longest earlier occurrence of the history's current suffix (D1). The history
// is a session's tokens (prompt + committed, append-only); every trigram of it maps to its kWays most recent end
// positions in an open-addressing table that doubles at half load, and a proposal checks those candidates, extends
// each match backwards up to kMaxMatch tokens and takes the longest (the most recent on ties). A continuation that
// runs into the suffix itself repeats with the match's period. Matches shorter than kMinMatch propose nothing.
class SuffixDrafter {
public:
    static constexpr uint32_t kMinMatch = 3;  // the indexed n-gram
    static constexpr uint32_t kMaxMatch = 32;
    static constexpr uint32_t kWays = 4;

    // Indexes `tokens` past what earlier calls saw (the same growing history every time), then writes up to `max_k`
    // tokens that would follow tokens.back() into `out`; returns how many (0: no match of at least kMinMatch, or
    // max_k 0) and sets *match to the match length behind them (0 when none).
    uint32_t propose(const std::vector<uint32_t> &tokens, uint32_t max_k, uint32_t *out, uint32_t *match);
    // History tokens indexed so far.
    uint64_t indexed() const { return indexed_; }

private:
    struct Slot {
        uint64_t key = 0;          // trigram hash | 1 (0 = empty)
        uint32_t n = 0;            // positions held
        uint32_t pos[kWays] = {};  // end positions (the index of the trigram's last token), most recent first
    };
    static uint64_t key_at(const uint32_t *tokens, uint64_t end);
    Slot *find(uint64_t key, bool insert);
    void insert(uint64_t key, uint32_t end);

    std::vector<Slot> table_;
    uint64_t used_ = 0;
    uint64_t indexed_ = 0;
};

// The cost of a verify window t = 1 + drafts (D3): a fixed table (m29a design note A3), window t at its entry, so
// the policy's picks depend only on the token stream (reproducible runs; batch-concurrency design note BC-D4: every
// ticket class, sampled included, so a seeded ticket's draws are the same alone, after any traffic and as one row of
// many). O95 / m23 design note SC-01: the table is the loaded library's own, measured at cere_model_seal
// (host/spec_calibrate.h calibrate_round_costs) on a device whose round time is its window's cost and kept for the
// life of the load; otherwise (the CPU toy, or a model the calibration session does not fit) it is untimed(). Only
// ratios matter to the policy. shape() is the prior shape 1 + kSlope (t - 1) the calibration extrapolates by.
class SpecCosts {
public:
    static constexpr uint32_t kMaxWindow = 16;  // Chains (O123): a chain of up to 15 drafts (kMaxSpecChainDrafts)
    static constexpr double kSlope = 0.35;
    // untimed()'s growth per extra verify row: the CUDA device's tables grew about 8 % per row (library c772467a,
    // model_k8 "A3 window t" / "A3D window t": window 9 at 1.56x window 1 under small_m, 1.69x under cuBLASLt).
    static constexpr double kUntimedSlope = 0.08;

    // untimed()'s table.
    SpecCosts();
    // The fixed table `ms` (window t costs ms[t - 1], every entry > 0).
    explicit SpecCosts(const std::array<double, kMaxWindow> &ms) : ms_(ms) {}
    // The table window t costs 1 + kUntimedSlope (t - 1) of: the table of a model whose device does not time its
    // rounds (deterministic picks on the CPU toy) or whose calibration session does not fit.
    static SpecCosts untimed();
    // The entries (window t at [t - 1]).
    const std::array<double, kMaxWindow> &table() const { return ms_; }

    // Window t's entry; t is clamped to 1..kMaxWindow.
    double cost(uint32_t window) const;
    static double shape(uint32_t window) { return 1.0 + kSlope * double(window - 1); }

private:
    std::array<double, kMaxWindow> ms_{};
};

// The per-request policy (D2): each round, the window with the most expected committed tokens per unit cost among no
// drafts, the MTP's first n proposals (n = 1..max) and the lookup's first k (k = 1..min(proposal, max)):
//   MTP     E(n) = 1 + p1 + p1 p2 + ... + p1..pn, p_j = the acceptance of the j-th MTP draft given the earlier ones
//   lookup  E(k) = 1 + q + q^2 + ... + q^k, q = the acceptance of lookup drafts whose match length is in that bucket
// the lookup taken only when it beats the best other window by kMargin. Rates are decayed counts plus prior
// pseudo-counts; every observed round decays every count, so a source not tried for a while drifts back to its prior.
// Counts only grow when the lookup is verified, so a bucket >= 1 whose decayed evidence (accepted + rejected) is below
// kExplore is valued at q = max(q, kOptimistic): a match length never tried (or not for ~60 rounds) gets one round
// when its window pays at that rate, and the round's outcome settles it. Bucket 0 (the shortest matches) is never
// explored. It only chooses which drafts are verified, never a token.
class DraftPolicy {
public:
    enum class Source : uint8_t { kNone, kMtp, kLookup, kTail };
    struct Pick {
        Source source = Source::kNone;
        uint32_t drafts = 0;
    };
    static constexpr uint32_t kMaxDrafts = 3;
    static constexpr uint32_t kBuckets = 4;
    static constexpr double kMargin = 0.03;
    static constexpr double kDecay = 0.95;
    static constexpr double kPriorWeight = 2.0;
    static constexpr double kMtpPrior = 0.68;  // 1 + p + p^2 + p^3 = 2.53 (M2.9 R7: accept length 2.47)
    static constexpr std::array<double, kBuckets> kLookupPrior = {0.45, 0.6, 0.75, 0.88};
    static constexpr double kExplore = 0.05;
    static constexpr double kOptimistic = 0.9;
    // DraftAccept (CERE_WORK_TAIL_DRAFTS): the tail's per-draft acceptance prior by match bucket (the lookup's; its
    // counts are separate: a tail draft follows three accepted MTP drafts the lookup agreed with).
    static constexpr uint32_t kMaxTailDrafts = 8;
    // DraftTail: what one round offers a tail. `lookup` holds the lookup proposal's `length` drafts (0 = no match),
    // `match` its match length, `mtp` the session's `mtp_count` pending MTP proposals, `rows` the drafting rows of
    // the round's speculate_rows call and `most` the longest chain the round may verify (kMaxTailDrafts; Chains,
    // CERE_WORK_CHAIN_DRAFTS: the device's max_chain_drafts).
    struct Tail {
        const uint32_t *lookup = nullptr;
        uint32_t length = 0;
        uint32_t match = 0;
        const uint32_t *mtp = nullptr;
        uint32_t mtp_count = 0;
        uint32_t rows = 1;
        uint32_t most = kMaxTailDrafts;
    };
    // A round's tail decision: `base` (choose's pick) unless the round is the call's only drafting row (SpecCosts
    // prices single-row rounds; with other rows a longer chain can split the call into two verify groups), the lookup
    // has a match of at least SuffixDrafter::kMinMatch with more than kMaxDrafts drafts, the session has kMaxDrafts
    // pending MTP proposals equal to the lookup's first kMaxDrafts and the tail pays more: the MTP proposals and
    // lookup drafts 4..k (k <= min(length, most)), valued
    //   E(k) = 1 + p1 + p1 p2 + p1 p2 p3 (1 + t + ... + t^(k - 3)), t = the tail rate of the match length's bucket
    // over cost(k + 1), beat value(base). A kTail pick of k drafts verifies the lookup proposal's first k. Without a
    // match (or with other rows) the round is exactly CERE_WORK_LOOKUP_DRAFTS + adaptive's.
    Pick extend(const SpecCosts &costs, Pick base, const Tail &tail) const;
    // TreeSpec (CERE_WORK_LONG_DRAFTS): a round's long-lookup decision, made after choose and before extend. `base`
    // (choose's pick) unless the round is the call's only drafting row (as extend), the lookup has a match of at
    // least SuffixDrafter::kMinMatch with more than kMaxDrafts drafts and its first k drafts (kMaxDrafts < k <=
    // min(length, most)), valued as choose values a lookup window (E(k) = 1 + q + ... + q^k over cost(k + 1),
    // the bucket's q, kOptimistic while unexplored), beat value(base), by kMargin unless `base` is itself a lookup
    // pick: choose's lookup window widened past kMaxDrafts. A kLookup pick of k drafts verifies the lookup proposal's
    // first k, observed as any lookup round. `round.mtp` is not read.
    Pick lengthen(const SpecCosts &costs, Pick base, const Tail &round) const;

    // `mtp`: MTP proposals may be verified; `max_drafts`: drafts that fit the round (clamped to kMaxDrafts);
    // `lookup`: the lookup proposal's length (0 = none) and `match` its match length.
    Pick choose(const SpecCosts &costs, bool mtp, uint32_t max_drafts, uint32_t lookup, uint32_t match) const;
    // After the round: what it verified, the drafts accepted and whether a draft was rejected (a draft left unverified
    // by a stop token or the commit limit is neither accepted nor rejected).
    void observe(Pick pick, uint32_t accepted, bool rejected, uint32_t match);
    double mtp_rate(uint32_t position) const;  // p_j, position 1..kMaxDrafts
    double lookup_rate(uint32_t match) const;  // q of the match length's bucket
    double tail_rate(uint32_t match) const;    // t of the match length's bucket
    static uint32_t bucket(uint32_t match);

private:
    std::array<double, kMaxDrafts> mtp_ok_{}, mtp_bad_{};
    std::array<double, kBuckets> lookup_ok_{}, lookup_bad_{};
    std::array<double, kBuckets> tail_ok_{}, tail_bad_{};
    double value(const SpecCosts &costs, Pick pick, uint32_t match) const;  // E / cost of an MTP or lookup pick
    double lookup_q(uint32_t match) const;  // q of a lookup pick: the bucket's rate, kOptimistic while unexplored
};

}  // namespace cere::host
