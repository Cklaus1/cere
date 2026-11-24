// m23 design note P3 (P3b), build-loop T4: the tactic resolver of executor/cuda/src/p3b/p3b_tactic.h. P3b pins its
// GEMM1 and GEMM2 tactics by configuration (tile shape, swapAB, epilogue), one pair per m bucket ([REVISED: M23T-01]);
// INSTANTIATE resolves each to the runner's own tactic index and fails closed (CERE_ERR_UNSUPPORTED) when zero or more
// than one candidate matches.
#include "check.h"

#include "p3b/p3b_tactic.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using cere::p3b::bucket_of;
using cere::p3b::kPinnedBucketCount;
using cere::p3b::kPinnedBuckets;
using cere::p3b::kReference;
using cere::p3b::PinnedBucket;
using cere::p3b::resolve_pins;
using cere::p3b::resolve_tactic;
using cere::p3b::Tactic;
using cere::p3b::TacticIndices;

constexpr size_t kSentinel = 12345;

cere_status resolve(const std::vector<Tactic> &list, const Tactic &want, size_t *index) {
    *index = kSentinel;
    return resolve_tactic(list.data(), list.size(), want, index);
}

// The reference pair is the untuned default of F9 (probe tactics 0 and 20): 128x128x128B, no swapAB, epilogue NONE;
// fused finalize is never pinned.
void test_reference_pair() {
    CHECK_EQ(kReference.tile_m, 128);
    CHECK_EQ(kReference.tile_n, 128);
    CHECK_EQ(kReference.tile_k_bytes, 128);
    CHECK(!kReference.swap_ab);
    CHECK(!kReference.finalize);
}

// The shipped table: max_m strictly increasing from >= 1, the last bucket open-ended, no FINALIZE pin, every m in
// [1, 2^32) in exactly one bucket.
void test_shipped_table() {
    CHECK(kPinnedBucketCount > 0);
    uint32_t previous = 0;
    for (size_t b = 0; b < kPinnedBucketCount; ++b) {
        CHECK(kPinnedBuckets[b].max_m > previous);
        previous = kPinnedBuckets[b].max_m;
        CHECK(!kPinnedBuckets[b].gemm1.finalize);
        CHECK(!kPinnedBuckets[b].gemm2.finalize);
    }
    CHECK_EQ(previous, UINT32_MAX);
    CHECK_EQ(bucket_of(kPinnedBuckets, kPinnedBucketCount, 1), 0);
    CHECK_EQ(bucket_of(kPinnedBuckets, kPinnedBucketCount, UINT32_MAX), kPinnedBucketCount - 1);
}

// The shipped pins are the sweep's buckets (an internal benchmark report): m 1..6 GEMM1/GEMM2
// 128x32x64B swapAB, 7..512 128x32x128B swapAB, 513..2048 GEMM1 128x32x128B swapAB + GEMM2 128x128x64B, above 2048
// 256x128x64B swapAB on both; epilogue NONE everywhere.
void test_shipped_pins() {
    const Tactic t32x64{128, 32, 64, true, false};
    const Tactic t32x128{128, 32, 128, true, false};
    const Tactic t128x64{128, 128, 64, false, false};
    const Tactic t256x64{256, 128, 64, true, false};
    struct Want {
        uint32_t m;
        Tactic gemm1;
        Tactic gemm2;
    };
    const std::vector<Want> want = {{1, t32x64, t32x64},       {6, t32x64, t32x64},       {7, t32x128, t32x128},
                                    {16, t32x128, t32x128},    {17, t32x128, t32x128},    {512, t32x128, t32x128},
                                    {513, t32x128, t128x64},   {2048, t32x128, t128x64},  {2049, t256x64, t256x64},
                                    {4096, t256x64, t256x64},  {UINT32_MAX, t256x64, t256x64}};
    CHECK_EQ(kPinnedBucketCount, 4);
    for (const Want &w : want) {
        const size_t b = bucket_of(kPinnedBuckets, kPinnedBucketCount, w.m);
        CHECK(b < kPinnedBucketCount);
        if (b < kPinnedBucketCount) {
            CHECK(kPinnedBuckets[b].gemm1 == w.gemm1);
            CHECK(kPinnedBuckets[b].gemm2 == w.gemm2);
        }
    }
}

// Exactly one match: its index.
void test_one_match() {
    const std::vector<Tactic> list = {{128, 32, 128, false, false}, {128, 128, 128, true, false},
                                      {128, 128, 128, false, true}, {128, 128, 128, false, false},
                                      {256, 128, 128, false, false}};
    size_t index = 0;
    CHECK_EQ(resolve(list, kReference, &index), CERE_OK);
    CHECK_EQ(index, 3);
    CHECK_EQ(resolve(list, Tactic{128, 128, 128, true, false}, &index), CERE_OK);
    CHECK_EQ(index, 1);
    CHECK_EQ(resolve(list, Tactic{128, 128, 128, false, true}, &index), CERE_OK);
    CHECK_EQ(index, 2);
}

// No match (absent tile, absent swap variant, empty list): UNSUPPORTED, index untouched.
void test_no_match() {
    const std::vector<Tactic> list = {{128, 32, 128, false, false}, {128, 128, 256, false, false},
                                      {128, 128, 128, true, false}, {128, 128, 128, false, true}};
    size_t index = 0;
    CHECK_EQ(resolve(list, kReference, &index), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(index, kSentinel);
    CHECK_EQ(resolve({}, kReference, &index), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(index, kSentinel);
}

// Two matches (the configuration does not identify one tactic): UNSUPPORTED, index untouched.
void test_duplicate_match() {
    const std::vector<Tactic> list = {{128, 128, 128, false, false}, {128, 64, 128, false, false},
                                      {128, 128, 128, false, false}};
    size_t index = 0;
    CHECK_EQ(resolve(list, kReference, &index), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(index, kSentinel);
}

// Null pointers: ARGUMENT.
void test_arguments() {
    const Tactic one = kReference;
    size_t index = kSentinel;
    CHECK_EQ(resolve_tactic(&one, 1, kReference, nullptr), CERE_ERR_ARGUMENT);
    CHECK_EQ(resolve_tactic(nullptr, 1, kReference, &index), CERE_ERR_ARGUMENT);
    CHECK_EQ(index, kSentinel);
}

const std::vector<PinnedBucket> kTable = {{1, {128, 32, 128, true, false}, {128, 128, 128, false, false}},
                                          {16, {128, 64, 128, true, false}, {128, 32, 128, true, false}},
                                          {UINT32_MAX, kReference, kReference}};
const std::vector<Tactic> kGemm1 = {{128, 128, 128, false, false}, {128, 32, 128, true, false},
                                    {128, 64, 128, true, false}};
const std::vector<Tactic> kGemm2 = {{128, 32, 128, true, false}, {128, 128, 128, false, false}};

// The bucket of m: the first whose max_m >= m; m = 0 or above every bucket: the count.
void test_bucket_of() {
    const PinnedBucket *t = kTable.data();
    CHECK_EQ(bucket_of(t, 3, 0), 3);
    CHECK_EQ(bucket_of(t, 3, 1), 0);
    CHECK_EQ(bucket_of(t, 3, 2), 1);
    CHECK_EQ(bucket_of(t, 3, 16), 1);
    CHECK_EQ(bucket_of(t, 3, 17), 2);
    CHECK_EQ(bucket_of(t, 3, 4096), 2);
    CHECK_EQ(bucket_of(t, 2, 17), 2);
}

// Every bucket's pair resolved to the runner's indices.
void test_resolve_pins() {
    std::vector<TacticIndices> out(3, TacticIndices{kSentinel, kSentinel});
    CHECK_EQ(resolve_pins(kGemm1.data(), kGemm1.size(), kGemm2.data(), kGemm2.size(), kTable.data(), kTable.size(),
                          out.data()),
             CERE_OK);
    CHECK_EQ(out[0].gemm1, 1);
    CHECK_EQ(out[0].gemm2, 1);
    CHECK_EQ(out[1].gemm1, 2);
    CHECK_EQ(out[1].gemm2, 0);
    CHECK_EQ(out[2].gemm1, 0);
    CHECK_EQ(out[2].gemm2, 1);
}

// One pin of one bucket unmatched (or matched twice): UNSUPPORTED, every index untouched.
void test_resolve_pins_fails_closed() {
    const std::vector<Tactic> missing = {kGemm1[0], kGemm1[1]};
    const std::vector<Tactic> twice = {kGemm2[0], kGemm2[1], kGemm2[0]};
    std::vector<TacticIndices> out(3, TacticIndices{kSentinel, kSentinel});
    CHECK_EQ(resolve_pins(missing.data(), missing.size(), kGemm2.data(), kGemm2.size(), kTable.data(), kTable.size(),
                          out.data()),
             CERE_ERR_UNSUPPORTED);
    CHECK_EQ(resolve_pins(kGemm1.data(), kGemm1.size(), twice.data(), twice.size(), kTable.data(), kTable.size(),
                          out.data()),
             CERE_ERR_UNSUPPORTED);
    for (const TacticIndices &i : out) {
        CHECK_EQ(i.gemm1, kSentinel);
        CHECK_EQ(i.gemm2, kSentinel);
    }
}

// A malformed table (empty, max_m 0, not increasing, last bucket bounded) or null pointers: ARGUMENT, untouched.
void test_resolve_pins_arguments() {
    std::vector<TacticIndices> out(3, TacticIndices{kSentinel, kSentinel});
    const auto run = [&](const std::vector<PinnedBucket> &table) {
        return resolve_pins(kGemm1.data(), kGemm1.size(), kGemm2.data(), kGemm2.size(), table.data(), table.size(),
                            out.data());
    };
    CHECK_EQ(run({}), CERE_ERR_ARGUMENT);
    CHECK_EQ(run({{0, kReference, kReference}, {UINT32_MAX, kReference, kReference}}), CERE_ERR_ARGUMENT);
    CHECK_EQ(run({{16, kReference, kReference}, {16, kReference, kReference}, {UINT32_MAX, kReference, kReference}}),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(run({{16, kReference, kReference}, {4096, kReference, kReference}}), CERE_ERR_ARGUMENT);
    CHECK_EQ(resolve_pins(kGemm1.data(), kGemm1.size(), kGemm2.data(), kGemm2.size(), kTable.data(), kTable.size(),
                          nullptr),
             CERE_ERR_ARGUMENT);
    CHECK_EQ(resolve_pins(nullptr, 1, kGemm2.data(), kGemm2.size(), kTable.data(), kTable.size(), out.data()),
             CERE_ERR_ARGUMENT);
    for (const TacticIndices &i : out) {
        CHECK_EQ(i.gemm1, kSentinel);
        CHECK_EQ(i.gemm2, kSentinel);
    }
}

}  // namespace

int main() {
    test_reference_pair();
    test_shipped_table();
    test_shipped_pins();
    test_one_match();
    test_no_match();
    test_duplicate_match();
    test_arguments();
    test_bucket_of();
    test_resolve_pins();
    test_resolve_pins_fails_closed();
    test_resolve_pins_arguments();
    return cere::test::finish("test_p3b_tactic");
}
