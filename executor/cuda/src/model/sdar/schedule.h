// SelfTune ST4 S3c (selftune design note ST4, plan S3 "DECODE POLICY", conditions 3 and 8): the block-diffusion
// schedule of family sdar (DICE-8B), shared by the CUDA device (device/sdar/) and the host toy of an sdar library
// (toy/sdar/). Header-only and CUDA-free: the forward passes are a `Forward` the device implements.
//
// The host keeps its per-token step contract (host/device.h `step`): every call consumes tokens and scores the next
// one. The schedule maps it onto blocks of kBlock positions (block b = positions [4b, 4b + 4)):
// - A prompt: its full blocks are prefilled (block-causal attention, KV stored); the rest of the prompt (0..3 tokens)
//   opens the first decode block, whose other positions hold the mask token.
// - A decode block at base: one denoise forward per masked position over the block's 4 rows (their K/V are written
//   to the session's tail slots [base, base + 4) and read from there, never committed); each picks, per masked row,
//   the lowest id among the BF16-rounded maximal logits and their count (ties), and unmasks the row with the fewest
//   ties, the lowest row among equals (the reference's low_confidence_static with top_k 1: confidence = 1 / ties).
//   Then one commit forward over the final 4 tokens stores their KV, and the session's KV position advances to
//   base + 4. The generated tokens wait in `pending`: the call returns the first, and each later call that feeds the
//   pending front back (count 1) returns the next one with no forward. So the host commits one token per call while
//   the device runs whole blocks: a stop token or max_new_tokens ends the output, never the block (condition 8).
// - Any other call (a new prompt, or a fed token other than the pending front: a cancelled or finished ticket's
//   session continuing) drops `pending` and rolls the KV position back to the block holding the session's last fed
//   token: a partly handed-out block's rows are recomputed from the fed tokens (a denoise rewrites the tail slots
//   before any read), so no uncommitted or stale row is ever attended (condition 3).
// - A failed forward leaves the KV position and the fed tokens as they were and breaks the session (host/device.h:
//   a failure after the first launch; its later steps fail with CERE_ERR_INTERNAL).
#pragma once

#include "cere_executor.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace cere::sdar {

constexpr uint64_t kScheduleBlock = 4;

// One row's pick of a denoise forward (the layout of kernels/sdar/sdar_kernels.h TopTie): the lowest id among the
// BF16-rounded maximal logits, their count, the runner-up and both BF16 values as floats.
struct RowPick {
    uint32_t id = 0;
    uint32_t ties = 0;
    uint32_t runner_up = 0;
    float logit = 0.0f;
    float runner_up_logit = 0.0f;
};

// The unmask choice over a block's picks: the masked row with the fewest ties, the lowest row among equals; -1 when
// no row is masked.
inline int32_t select_unmask(const RowPick *picks, const bool *masked, int rows) {
    int32_t best = -1;
    for (int r = 0; r < rows; ++r) {
        if (!masked[r]) continue;
        if (best < 0 || picks[r].ties < picks[best].ties) best = r;
    }
    return best;
}

// The forward passes of one session (the device binds its session to them).
class Forward {
public:
    virtual ~Forward() = default;
    // Prefill: n rows (n % 4 == 0) at positions [pos, pos + n) (pos % 4 == 0), block-causal over the stored rows
    // [0, pos) and themselves; stores their KV. No logits.
    virtual cere_status prefill(const uint32_t *tokens, uint64_t n, uint64_t pos) = 0;
    // Denoise: the block's 4 rows at base (masked rows hold the mask token), K/V in the tail slots, nothing committed;
    // picks[r] for every row, *index = select_unmask(picks, masked, 4).
    virtual cere_status denoise(const uint32_t *block, const bool *masked, uint64_t base, RowPick *picks,
                                int32_t *index) = 0;
    // Commit: the block's final 4 tokens at base; stores their KV. No logits.
    virtual cere_status commit(const uint32_t *block, uint64_t base) = 0;
};

// One session's schedule state.
struct Session {
    uint64_t capacity = 0;          // max_context_tokens of the session
    std::vector<uint32_t> fed;      // every token the host fed, in order (the host's position is fed.size())
    uint64_t kv = 0;                // rows [0, kv) hold committed KV (a multiple of kScheduleBlock)
    std::vector<uint32_t> pending;  // generated tokens of the last block not yet fed back, front first
    std::vector<RowPick> pending_picks;
    bool broken = false;
};

// KV rows a session of max_context_tokens needs: the last block a step can open starts at floor(max / 4) * 4 (a
// session holding max fed tokens) and is written whole.
constexpr uint64_t kv_rows(uint64_t max_context_tokens) {
    return max_context_tokens / kScheduleBlock * kScheduleBlock + kScheduleBlock;
}

inline cere_token_score score_of(const RowPick &p) {
    cere_token_score s{};
    s.id = p.id;
    s.runner_up = p.runner_up;
    s.logit = p.logit;
    s.runner_up_logit = p.runner_up_logit;
    return s;
}

// host/device.h `step` on an sdar session: consumes n = min(count, chunk) tokens (chunk 0 = no limit; the host's
// `limit` is always 0 on family sdar: no prefix capture), and when n == count scores the next token into *out.
// mask_id: the config's mask token. A failed forward restores the fed tokens and the KV position of the call's start
// and breaks the session.
inline cere_status step(Forward &forward, Session &s, const uint32_t *tokens, uint64_t count, uint64_t chunk,
                        uint32_t mask_id, uint64_t *consumed, cere_token_score *out) {
    if (s.broken) return CERE_ERR_INTERNAL;
    if (tokens == nullptr || count == 0 || consumed == nullptr || out == nullptr) return CERE_ERR_ARGUMENT;
    const uint64_t entry_fed = s.fed.size();
    const uint64_t entry_kv = s.kv;
    const auto fail = [&](cere_status st) {
        s.fed.resize(entry_fed);
        s.kv = entry_kv;
        s.pending.clear();
        s.pending_picks.clear();
        s.broken = true;
        return st;
    };
    uint64_t n = 1;
    if (count == 1 && !s.pending.empty() && tokens[0] == s.pending.front()) {
        if (s.fed.size() + 1 > s.capacity) return CERE_ERR_CAPACITY;
        s.fed.push_back(tokens[0]);
        s.pending.erase(s.pending.begin());
        s.pending_picks.erase(s.pending_picks.begin());
        if (!s.pending.empty()) {
            *consumed = 1;
            *out = score_of(s.pending_picks.front());
            return CERE_OK;
        }
    } else {
        n = chunk == 0 ? count : std::min(count, chunk);
        if (s.fed.size() + n > s.capacity) return CERE_ERR_CAPACITY;
        s.pending.clear();
        s.pending_picks.clear();
        // Roll back to the block holding the last fed token (a block that ran ahead of the host).
        s.kv = std::min<uint64_t>(s.kv, s.fed.size() / kScheduleBlock * kScheduleBlock);
        s.fed.insert(s.fed.end(), tokens, tokens + n);
        const uint64_t full = s.fed.size() / kScheduleBlock * kScheduleBlock;
        if (full > s.kv) {
            if (const cere_status st = forward.prefill(s.fed.data() + s.kv, full - s.kv, s.kv); st != CERE_OK) {
                return fail(st);
            }
            s.kv = full;
        }
        if (n < count) {
            *consumed = n;
            return CERE_OK;
        }
    }
    // A decode block at base = kv: the fed tokens past kv (0..3), then masks.
    const uint64_t base = s.kv;
    const uint64_t known = s.fed.size() - base;
    std::array<uint32_t, kScheduleBlock> block{};
    std::array<bool, kScheduleBlock> masked{};
    std::array<RowPick, kScheduleBlock> picked{};
    for (uint64_t r = 0; r < kScheduleBlock; ++r) {
        masked[r] = r >= known;
        block[r] = masked[r] ? mask_id : s.fed[base + r];
    }
    for (uint64_t i = known; i < kScheduleBlock; ++i) {
        std::array<RowPick, kScheduleBlock> picks{};
        int32_t index = -1;
        if (const cere_status st = forward.denoise(block.data(), masked.data(), base, picks.data(), &index);
            st != CERE_OK) {
            return fail(st);
        }
        if (index < 0 || index >= int32_t(kScheduleBlock) || !masked[index]) return fail(CERE_ERR_INTERNAL);
        block[index] = picks[index].id;
        picked[index] = picks[index];
        masked[index] = false;
    }
    if (const cere_status st = forward.commit(block.data(), base); st != CERE_OK) return fail(st);
    s.kv = base + kScheduleBlock;
    s.pending.assign(block.begin() + known, block.end());
    s.pending_picks.assign(picked.begin() + known, picked.end());
    *consumed = n;
    *out = score_of(s.pending_picks.front());
    return CERE_OK;
}

}  // namespace cere::sdar
