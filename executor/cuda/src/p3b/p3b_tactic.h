// P3b (m23 design note P3, F9): the GEMM tactics of FlashInfer's CUTLASS fused-MoE runner are pinned by
// configuration, never by index and never by an autotuner at run time. INSTANTIATE maps each candidate of the
// runner's own list (getTactics(GEMM_1 / GEMM_2), CutlassGemmConfig::tile_config_sm120 / swap_ab /
// epilogue_fusion_type) to a Tactic and resolves every pinned one to its index; zero or more than one match fails
// closed with CERE_ERR_UNSUPPORTED. One pinned pair per m bucket ([REVISED: M23T-01]), each chosen by the offline
// sweep (executor/tests/gpu/test_p3blab_sweep.py) among the pairs bit-equal to the reference pair. Pure C++
// (host-testable: executor/tests/host/test_p3b_tactic.cpp).
#pragma once

#include "cere_executor.h"

#include <cstddef>
#include <cstdint>

namespace cere::p3b {

struct Tactic {
    int tile_m = 0;        // CTA shape M
    int tile_n = 0;        // CTA shape N
    int tile_k_bytes = 0;  // CTA shape K in bytes (the "B" of CutlassTileConfigSM120::CtaShape<M>x<N>x<K>B)
    bool swap_ab = false;
    bool finalize = false;  // epilogue FINALIZE (fused finalize); P3b runs with fused finalize off (F9)

    constexpr bool operator==(const Tactic &o) const {
        return tile_m == o.tile_m && tile_n == o.tile_n && tile_k_bytes == o.tile_k_bytes && swap_ab == o.swap_ab &&
               finalize == o.finalize;
    }
};

// A tactic pair as positions in the runner's GEMM1 / GEMM2 lists (p3b_moe.h tactic_lists).
struct TacticIndices {
    size_t gemm1 = 0;
    size_t gemm2 = 0;
};

// The reference pair: FlashInfer's untuned default (F9; the N2 probe's tactics 0 and 20), CtaShape128x128x128B, no
// swapAB, epilogue NONE, for GEMM1 and GEMM2. Every pinned pair is bit-equal to it at every m the sweep and P3's pair
// row run (the n <= 16 rule, [REVISED: M23T-01]).
constexpr Tactic kReference{128, 128, 128, false, false};

// The pinned pair of the m in (the previous bucket's max_m, max_m].
struct PinnedBucket {
    uint32_t max_m = 0;
    Tactic gemm1;
    Tactic gemm2;
};

// The pins (an internal benchmark report lab report native-test-p3blab-20261012T145750Z): every
// pair the runner runs was bit-equal to the reference and repeatable at every swept m (1..16, 32..4096) on layers 0,
// 23 and 47, so a per-m choice inside [1, 16] outputs exactly what the reference pair does at every m (the n <= 16
// rule's batched = single-row identity is the reference pair's, [REVISED: M23T-01]).
// Each bucket's pair is within 1 % of the per-m fastest at every swept m it covers. max_m strictly increasing, the
// last bucket open-ended.
constexpr PinnedBucket kPinnedBuckets[] = {
    {6, {128, 32, 64, true, false}, {128, 32, 64, true, false}},
    {512, {128, 32, 128, true, false}, {128, 32, 128, true, false}},
    {2048, {128, 32, 128, true, false}, {128, 128, 64, false, false}},
    {UINT32_MAX, {256, 128, 64, true, false}, {256, 128, 64, true, false}},
};
constexpr size_t kPinnedBucketCount = sizeof(kPinnedBuckets) / sizeof(kPinnedBuckets[0]);

// `*index` = the position of the one candidate equal to `want`. CERE_ERR_UNSUPPORTED (index untouched) when none or
// more than one is; CERE_ERR_ARGUMENT for a null `index`, or null `candidates` with count > 0.
inline cere_status resolve_tactic(const Tactic *candidates, size_t count, const Tactic &want, size_t *index) {
    if (index == nullptr || (candidates == nullptr && count > 0)) {
        return CERE_ERR_ARGUMENT;
    }
    size_t found = count;
    for (size_t i = 0; i < count; ++i) {
        if (candidates[i] == want) {
            if (found != count) {
                return CERE_ERR_UNSUPPORTED;
            }
            found = i;
        }
    }
    if (found == count) {
        return CERE_ERR_UNSUPPORTED;
    }
    *index = found;
    return CERE_OK;
}

// The bucket of `m`: the first of `buckets[count]` whose max_m >= m; `count` for m = 0 or m above every bucket.
inline size_t bucket_of(const PinnedBucket *buckets, size_t count, uint32_t m) {
    if (m == 0 || buckets == nullptr) {
        return count;
    }
    for (size_t b = 0; b < count; ++b) {
        if (m <= buckets[b].max_m) {
            return b;
        }
    }
    return count;
}

// `indices[b]` = the GEMM1 / GEMM2 positions of bucket b's pair in the runner's lists `g1[n1]` / `g2[n2]`, for every
// bucket. CERE_ERR_ARGUMENT for null pointers or a malformed table (empty, max_m 0 or not strictly increasing, the
// last bucket not open-ended at UINT32_MAX); CERE_ERR_UNSUPPORTED when a pin matches zero or several candidates.
// `indices` is untouched unless every bucket resolves.
inline cere_status resolve_pins(const Tactic *g1, size_t n1, const Tactic *g2, size_t n2, const PinnedBucket *buckets,
                                size_t count, TacticIndices *indices) {
    if (indices == nullptr || buckets == nullptr || count == 0 || (g1 == nullptr && n1 > 0) ||
        (g2 == nullptr && n2 > 0)) {
        return CERE_ERR_ARGUMENT;
    }
    uint32_t previous = 0;
    for (size_t b = 0; b < count; ++b) {
        if (buckets[b].max_m <= previous) {
            return CERE_ERR_ARGUMENT;
        }
        previous = buckets[b].max_m;
    }
    if (previous != UINT32_MAX) {
        return CERE_ERR_ARGUMENT;
    }
    for (size_t b = 0; b < count; ++b) {
        TacticIndices pair;
        cere_status s = resolve_tactic(g1, n1, buckets[b].gemm1, &pair.gemm1);
        if (s == CERE_OK) {
            s = resolve_tactic(g2, n2, buckets[b].gemm2, &pair.gemm2);
        }
        if (s != CERE_OK) {
            return s;
        }
    }
    for (size_t b = 0; b < count; ++b) {
        resolve_tactic(g1, n1, buckets[b].gemm1, &indices[b].gemm1);
        resolve_tactic(g2, n2, buckets[b].gemm2, &indices[b].gemm2);
    }
    return CERE_OK;
}

}  // namespace cere::p3b
