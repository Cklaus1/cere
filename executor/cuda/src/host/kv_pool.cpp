#include "host/kv_pool.h"

#include <cstdio>

namespace cere::host {

cere_status plan_kv_pool(uint64_t requested, uint64_t max_sessions, uint64_t max_context_tokens, uint64_t support_max,
                         uint64_t *out) {
    if (requested == 0) {
        *out = max_sessions * kv_pages(max_context_tokens) * kKvPageTokens;
        return CERE_OK;
    }
    if (requested % kKvPageTokens != 0 || requested < max_context_tokens) return CERE_ERR_ARGUMENT;
    if (requested > support_max) return CERE_ERR_UNSUPPORTED;
    *out = requested;
    return CERE_OK;
}

PagePool::PagePool(uint64_t pages) : refs_(pages, 0), free_(pages) {}

cere_status PagePool::take_pages(uint64_t n, std::vector<uint32_t> *out) {
    if (n > free_) return CERE_ERR_CAPACITY;
    const size_t before = out->size();
    out->reserve(before + n);
    uint64_t taken = 0;
    for (uint64_t id = 0; taken < n && id < refs_.size(); ++id) {
        if (refs_[id] != 0) continue;
        refs_[id] = 1;
        out->push_back(static_cast<uint32_t>(id));
        ++taken;
    }
    if (taken != n) {  // free_ disagrees with the reference counts: a corrupted pool, nothing is taken
        for (size_t i = before; i < out->size(); ++i) refs_[(*out)[i]] = 0;
        out->resize(before);
        return CERE_ERR_INTERNAL;
    }
    free_ -= n;
    return CERE_OK;
}

cere_status PagePool::retain(const uint32_t *ids, uint64_t n) {
    uint64_t i = 0;
    for (; i < n; ++i) {
        if (ids[i] >= refs_.size() || refs_[ids[i]] == 0) break;
        ++refs_[ids[i]];
    }
    if (i == n) return CERE_OK;
    while (i-- > 0) --refs_[ids[i]];  // undo the entries already counted
    return CERE_ERR_INTERNAL;
}

cere_status PagePool::release_pages(const uint32_t *ids, uint64_t n) {
    uint64_t i = 0;
    for (; i < n; ++i) {
        if (ids[i] >= refs_.size() || refs_[ids[i]] == 0) break;
        if (--refs_[ids[i]] == 0) ++free_;
    }
    if (i == n) return CERE_OK;
    while (i-- > 0) {  // undo the entries already dropped
        if (refs_[ids[i]]++ == 0) --free_;
    }
    return CERE_ERR_INTERNAL;
}

void release_pages_logged(PagePool &pool, const uint32_t *ids, uint64_t n, const char *where) noexcept {
    if (const cere_status st = pool.release_pages(ids, n); st != CERE_OK) {
        std::fprintf(stderr, "cere: page_pool_fault op=release where=%s status=%d\n", where, int(st));
    }
}

cere_status PagePermutation::arm(const uint32_t *order, uint64_t n) {
    if (order == nullptr || n == 0) return CERE_ERR_ARGUMENT;
    std::vector<uint8_t> seen(n, 0);
    for (uint64_t i = 0; i < n; ++i) {
        if (order[i] >= n || seen[order[i]] != 0) return CERE_ERR_ARGUMENT;
        seen[order[i]] = 1;
    }
    order_.assign(order, order + n);
    return CERE_OK;
}

cere_status PagePermutation::apply(std::vector<uint32_t> *pages) {
    if (order_.empty()) return CERE_OK;
    std::vector<uint32_t> order;
    order.swap(order_);
    if (order.size() != pages->size()) return CERE_ERR_ARGUMENT;
    // The permuted list is built in the order's own storage, so applying allocates nothing.
    for (uint32_t &entry : order) entry = (*pages)[entry];
    pages->swap(order);
    return CERE_OK;
}

}  // namespace cere::host
