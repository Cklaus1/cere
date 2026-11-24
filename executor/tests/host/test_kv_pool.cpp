// m25 design note C2 (host half): the paged pool size check (D4) and the page allocator (D3) of
// executor/cuda/src/host/kv_pool.h. The C-API cases (desc field, toy create CAPACITY) live in test_state.cpp.
#include "check.h"

#include "host/kv_pool.h"
#include "host/support.h"
#include "toy/toy_device.h"

#include <cstdint>
#include <vector>

namespace {

using cere::host::kKvPageTokens;
using cere::host::kv_pages;
using cere::host::PagePermutation;
using cere::host::PagePool;
using cere::host::plan_kv_pool;

constexpr uint64_t kSentinel = 0xdeadbeef;

cere_status pool(uint64_t requested, uint64_t sessions, uint64_t ctx, uint64_t support_max, uint64_t *out) {
    *out = kSentinel;
    return plan_kv_pool(requested, sessions, ctx, support_max, out);
}

// D2/D6: page 64; the A4 and toy support tables bound a non-zero pool at 1,048,576 tokens = 16,384 pages
// (ctx262 design note CX-01, Headroom's 300,032 before: a validation bound, the memory estimate bounds the rest).
void test_page_and_support_bound() {
    CHECK_EQ(kKvPageTokens, 64);
    CHECK_EQ(kv_pages(1), 1);
    CHECK_EQ(kv_pages(64), 1);
    CHECK_EQ(kv_pages(65), 2);
    CHECK_EQ(kv_pages(200000), 3125);
    CHECK_EQ(kv_pages(262144), 4096);
    CHECK_EQ(kv_pages(300032), 4688);
    CHECK_EQ(kv_pages(1048576), 16384);
    CHECK_EQ(cere::host::a4_support().max_kv_pool_tokens, 1048576);
    CHECK_EQ(cere::host::a4_support().max_kv_pool_tokens % kKvPageTokens, 0);
    CHECK_EQ(cere::toy::fixture_support().max_kv_pool_tokens, 1048576);
    uint64_t out = 0;
    CHECK_EQ(pool(524288, 4, 262144, 1048576, &out), CERE_OK);  // the -v44 profile's desc
    CHECK_EQ(out, 524288);
    CHECK_EQ(pool(1048640, 4, 262144, 1048576, &out), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(out, kSentinel);
}

// D4: 0 = max_sessions × round64(ctx), the contiguous capacity of today; the derived pool is not bounded by the
// support's max_kv_pool_tokens (the memory estimate bounds it, as the sessions region did).
void test_zero_is_the_derived_pool() {
    uint64_t out = 0;
    CHECK_EQ(pool(0, 1, 200000, 200000, &out), CERE_OK);
    CHECK_EQ(out, 200000);
    CHECK_EQ(pool(0, 2, 9088, 200000, &out), CERE_OK);
    CHECK_EQ(out, 2 * 142 * 64);
    CHECK_EQ(pool(0, 1, 100, 200000, &out), CERE_OK);
    CHECK_EQ(out, 128);
    CHECK_EQ(pool(0, 4, 200000, 200000, &out), CERE_OK);
    CHECK_EQ(out, 800000);
}

// D4: non-zero not ×64 or < max_context_tokens -> ARGUMENT; > support -> UNSUPPORTED; *out untouched on error.
void test_non_zero_checks() {
    uint64_t out = 0;
    CHECK_EQ(pool(12300, 2, 9088, 200000, &out), CERE_ERR_ARGUMENT);  // not a multiple of 64
    CHECK_EQ(out, kSentinel);
    CHECK_EQ(pool(200001, 1, 200000, 200000, &out), CERE_ERR_ARGUMENT);
    CHECK_EQ(out, kSentinel);
    CHECK_EQ(pool(9024, 2, 9088, 200000, &out), CERE_ERR_ARGUMENT);  // below one session's context
    CHECK_EQ(out, kSentinel);
    CHECK_EQ(pool(200064, 1, 200000, 200000, &out), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(out, kSentinel);
    CHECK_EQ(pool(200000, 1, 200000, 200000, &out), CERE_OK);
    CHECK_EQ(out, 200000);
    CHECK_EQ(pool(12288, 2, 9088, 200000, &out), CERE_OK);  // K11 (3)'s desc: two 9,088 sessions do not fit
    CHECK_EQ(out, 12288);
    CHECK_EQ(pool(9088, 1, 9088, 200000, &out), CERE_OK);  // pool = ctx, a multiple of 64
    CHECK_EQ(out, 9088);
}

bool ascending_from(const std::vector<uint32_t> &ids, uint32_t first) {
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != first + i) return false;
    }
    return true;
}

// D3: a session reserves kv_pages(ctx) at create; a shortfall takes nothing; release frees; lowest-first reuse.
void test_take_release_lowest_first() {
    PagePool p(12288 / kKvPageTokens);  // 192 pages
    CHECK_EQ(p.pages(), 192);
    CHECK_EQ(p.free_pages(), 192);
    std::vector<uint32_t> first;
    CHECK_EQ(p.take_pages(kv_pages(9088), &first), CERE_OK);
    CHECK_EQ(first.size(), 142);
    CHECK(ascending_from(first, 0));
    CHECK_EQ(p.free_pages(), 50);

    std::vector<uint32_t> second{7};  // CAPACITY leaves *out as it was
    CHECK_EQ(p.take_pages(kv_pages(9088), &second), CERE_ERR_CAPACITY);
    CHECK_EQ(second.size(), 1);
    CHECK_EQ(second[0], 7);
    CHECK_EQ(p.free_pages(), 50);

    CHECK_EQ(p.release_pages(first.data(), first.size()), CERE_OK);
    CHECK_EQ(p.free_pages(), 192);
    second.clear();
    CHECK_EQ(p.take_pages(kv_pages(9088), &second), CERE_OK);  // the retry succeeds with the same ids
    CHECK(second == first);
    std::vector<uint32_t> rest;
    CHECK_EQ(p.take_pages(50, &rest), CERE_OK);
    CHECK(ascending_from(rest, 142));
    CHECK_EQ(p.free_pages(), 0);
    CHECK_EQ(p.take_pages(1, &rest), CERE_ERR_CAPACITY);
    CHECK_EQ(rest.size(), 50);
}

// Lowest free ids first, wherever the holes are; the order of a release does not matter.
void test_holes_fill_lowest_first() {
    PagePool p(8);
    std::vector<uint32_t> a, b, c;
    CHECK_EQ(p.take_pages(3, &a), CERE_OK);  // 0 1 2
    CHECK_EQ(p.take_pages(2, &b), CERE_OK);  // 3 4
    const uint32_t hole[] = {2, 0};
    CHECK_EQ(p.release_pages(hole, 2), CERE_OK);
    CHECK_EQ(p.free_pages(), 5);
    CHECK_EQ(p.take_pages(3, &c), CERE_OK);
    CHECK(c == (std::vector<uint32_t>{0, 2, 5}));
    std::vector<uint32_t> none;
    CHECK_EQ(p.take_pages(0, &none), CERE_OK);
    CHECK(none.empty());
}

void test_empty_pool() {
    PagePool p(0);
    std::vector<uint32_t> out;
    CHECK_EQ(p.take_pages(1, &out), CERE_ERR_CAPACITY);
    CHECK(out.empty());
    CHECK_EQ(p.take_pages(0, &out), CERE_OK);
}

// m210 design note D7 (ledger M5/CR3): a taken page has one reference; retain adds one per entry; release drops one
// and frees the page at 0, so a page the prefix tree also holds outlives the session that took it.
void test_refcounts() {
    PagePool p(4);
    std::vector<uint32_t> a;
    CHECK_EQ(p.take_pages(3, &a), CERE_OK);  // 0 1 2
    CHECK_EQ(p.refs(0), 1);
    CHECK_EQ(p.refs(3), 0);
    CHECK_EQ(p.retain(a.data(), 2), CERE_OK);  // the tree holds pages 0 and 1
    CHECK_EQ(p.refs(0), 2);
    CHECK_EQ(p.refs(1), 2);
    CHECK_EQ(p.refs(2), 1);
    CHECK_EQ(p.release_pages(a.data(), a.size()), CERE_OK);  // the session ends: only page 2 is free again
    CHECK_EQ(p.free_pages(), 2);
    CHECK_EQ(p.refs(0), 1);
    CHECK_EQ(p.refs(2), 0);
    std::vector<uint32_t> b;
    CHECK_EQ(p.take_pages(2, &b), CERE_OK);  // lowest free: 2 3, never a held page
    CHECK(b == (std::vector<uint32_t>{2, 3}));
    const uint32_t twice[] = {0, 0};
    CHECK_EQ(p.retain(twice, 2), CERE_OK);  // one reference per entry
    CHECK_EQ(p.refs(0), 3);
    CHECK_EQ(p.release_pages(twice, 2), CERE_OK);
    CHECK_EQ(p.release_pages(a.data(), 2), CERE_OK);  // the tree drops its node
    CHECK_EQ(p.refs(0), 0);
    CHECK_EQ(p.refs(1), 0);
    CHECK_EQ(p.free_pages(), 2);
}

// DM-3/DM-4: a release or retain the accounting cannot honor is INTERNAL and changes nothing (a double release used to
// wrap a count to 0xFFFFFFFF and leak the page for good).
void test_refcount_misuse_is_refused() {
    PagePool p(4);
    std::vector<uint32_t> a;
    CHECK_EQ(p.take_pages(2, &a), CERE_OK);  // 0 1
    CHECK_EQ(p.release_pages(a.data(), 2), CERE_OK);
    CHECK_EQ(p.free_pages(), 4);
    CHECK_EQ(p.release_pages(a.data(), 2), CERE_ERR_INTERNAL);  // double release
    CHECK_EQ(p.refs(0), 0);
    CHECK_EQ(p.refs(1), 0);
    CHECK_EQ(p.free_pages(), 4);

    CHECK_EQ(p.retain(a.data(), 1), CERE_ERR_INTERNAL);  // a free page has no reference to add to
    CHECK_EQ(p.refs(0), 0);
    const uint32_t beyond[] = {4};
    CHECK_EQ(p.release_pages(beyond, 1), CERE_ERR_INTERNAL);  // out of range
    CHECK_EQ(p.retain(beyond, 1), CERE_ERR_INTERNAL);

    std::vector<uint32_t> b;
    CHECK_EQ(p.take_pages(2, &b), CERE_OK);  // 0 1 again, one reference each
    const uint32_t dup[] = {0, 0};
    CHECK_EQ(p.release_pages(dup, 2), CERE_ERR_INTERNAL);  // one reference cannot be dropped twice
    CHECK_EQ(p.refs(0), 1);  // atomic: nothing was dropped
    CHECK_EQ(p.free_pages(), 2);
    const uint32_t mixed[] = {1, 3};
    CHECK_EQ(p.release_pages(mixed, 2), CERE_ERR_INTERNAL);  // page 3 is free: page 1 keeps its reference
    CHECK_EQ(p.refs(1), 1);
    CHECK_EQ(p.free_pages(), 2);
    CHECK_EQ(p.retain(mixed, 2), CERE_ERR_INTERNAL);
    CHECK_EQ(p.refs(1), 1);

    CHECK_EQ(p.retain(dup, 2), CERE_OK);  // refs 3: the same id twice is fine while the count covers it
    CHECK_EQ(p.release_pages(dup, 2), CERE_OK);
    CHECK_EQ(p.refs(0), 1);
    CHECK_EQ(p.release_pages(b.data(), 2), CERE_OK);
    CHECK_EQ(p.free_pages(), 4);
}

// K11's page order (ktest cere_ktest_session_set_pages, ledger R as amended by the owner 2026-10-10): armed on a model,
// applied to and consumed by its next session create, so a second session never inherits it.
void test_page_order_is_one_shot() {
    PagePermutation order;
    CHECK(!order.armed());
    std::vector<uint32_t> pages = {4, 5, 6, 7};
    CHECK_EQ(order.apply(&pages), CERE_OK);  // nothing armed: the reserved order stays
    CHECK(pages == (std::vector<uint32_t>{4, 5, 6, 7}));

    const uint32_t reversed[4] = {3, 2, 1, 0};
    CHECK_EQ(order.arm(reversed, 4), CERE_OK);
    CHECK(order.armed());
    CHECK_EQ(order.apply(&pages), CERE_OK);
    CHECK(pages == (std::vector<uint32_t>{7, 6, 5, 4}));
    CHECK(!order.armed());  // consumed: the next create keeps its ascending pages
    std::vector<uint32_t> next = {0, 1, 2, 3};
    CHECK_EQ(order.apply(&next), CERE_OK);
    CHECK(next == (std::vector<uint32_t>{0, 1, 2, 3}));

    const uint32_t interleaved[4] = {0, 2, 1, 3};
    CHECK_EQ(order.arm(interleaved, 4), CERE_OK);
    std::vector<uint32_t> three = {0, 1, 2};
    CHECK_EQ(order.apply(&three), CERE_ERR_ARGUMENT);  // another page count: refused, and still consumed
    CHECK(three == (std::vector<uint32_t>{0, 1, 2}));
    CHECK(!order.armed());

    CHECK_EQ(order.arm(interleaved, 4), CERE_OK);
    order.clear();  // model release
    CHECK(!order.armed());
}

void test_page_order_refusals() {
    PagePermutation order;
    const uint32_t repeated[3] = {0, 1, 1};
    const uint32_t out_of_range[3] = {0, 1, 3};
    const uint32_t ok[2] = {1, 0};
    CHECK_EQ(order.arm(nullptr, 2), CERE_ERR_ARGUMENT);
    CHECK_EQ(order.arm(ok, 0), CERE_ERR_ARGUMENT);
    CHECK_EQ(order.arm(repeated, 3), CERE_ERR_ARGUMENT);
    CHECK_EQ(order.arm(out_of_range, 3), CERE_ERR_ARGUMENT);
    CHECK(!order.armed());  // a refused arming arms nothing
    CHECK_EQ(order.arm(ok, 2), CERE_OK);
    CHECK_EQ(order.arm(repeated, 3), CERE_ERR_ARGUMENT);
    CHECK(order.armed());  // and keeps an earlier one
}

}  // namespace

int main() {
    test_page_and_support_bound();
    test_zero_is_the_derived_pool();
    test_non_zero_checks();
    test_take_release_lowest_first();
    test_holes_fill_lowest_first();
    test_empty_pool();
    test_refcounts();
    test_refcount_misuse_is_refused();
    test_page_order_is_one_shot();
    test_page_order_refusals();
    return cere::test::finish("test_kv_pool");
}
