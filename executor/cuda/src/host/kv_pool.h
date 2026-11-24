// Paged KV pool accounting (m25 design note D2-D4; ledger D 72 and M5), shared by the host plan check, the CUDA
// device and the CPU toy (which counts pages, no bytes). No CUDA include.
#pragma once

#include "cere_executor.h"

#include <cstdint>
#include <vector>

namespace cere::host {

// Tokens per page (D2): SGLang's page for compressed QSA, a multiple of the indexer compress ratio 4.
constexpr uint64_t kKvPageTokens = 64;

// Pages that hold `tokens` tokens: ceil(tokens / 64).
constexpr uint64_t kv_pages(uint64_t tokens) { return (tokens + kKvPageTokens - 1) / kKvPageTokens; }

// The pool of a validated plan (D4). `requested` is cere_model_desc.kv_pool_tokens: 0 = the derived pool,
// max_sessions × kv_pages(max_context_tokens) × 64, so every session gets its full context as with contiguous slots;
// non-zero must be a multiple of 64 and >= max_context_tokens (else CERE_ERR_ARGUMENT) and <= `support_max` (the
// support table's max_kv_pool_tokens, else CERE_ERR_UNSUPPORTED). Called after max_sessions and max_context_tokens
// passed their support bounds. *out is written only on CERE_OK.
cere_status plan_kv_pool(uint64_t requested, uint64_t max_sessions, uint64_t max_context_tokens, uint64_t support_max,
                         uint64_t *out);

// The page allocator (D3) with M2.10's per-page reference counts (m210 design note D7, ledger M5/CR3): a free list
// over page ids [0, pages), sized once at ALLOCATE, so taking, retaining and releasing pages allocates nothing. A page
// taken by a session has one reference; the prefix tree adds one per node holding it, and the page is free again when
// its count reaches 0 (an M2.5 session's page goes 1 -> 0 on release, as before). Not synchronized: the owner holds its
// lock (CUDA: slots_mutex_).
class PagePool {
public:
    explicit PagePool(uint64_t pages);

    uint64_t pages() const { return static_cast<uint64_t>(refs_.size()); }
    uint64_t free_pages() const { return free_; }
    // References of page `id` (0 = free).
    uint32_t refs(uint32_t id) const { return refs_[id]; }

    // Appends the `n` lowest free page ids to *out in ascending order (deterministic), one reference each.
    // CERE_ERR_CAPACITY when fewer than `n` are free: nothing is taken and *out is unchanged.
    cere_status take_pages(uint64_t n, std::vector<uint32_t> *out);
    // +1 reference on each of `n` allocated pages (a page may appear more than once: one reference per entry).
    // CERE_ERR_INTERNAL, with nothing changed, for an id outside the pool or a page with no reference (a caller bug:
    // the pool is never silently corrupted).
    [[nodiscard]] cere_status retain(const uint32_t *ids, uint64_t n);
    // -1 reference on each of `n` allocated pages; a page whose count reaches 0 is free again. CERE_ERR_INTERNAL,
    // with nothing changed, when an entry has no reference left to drop (a double release, or more entries of a page
    // than it has references).
    [[nodiscard]] cere_status release_pages(const uint32_t *ids, uint64_t n);

private:
    std::vector<uint32_t> refs_;  // 0 = free
    uint64_t free_;
};

// For destructors and cleanup paths that have no status to return: releases like PagePool::release_pages and, when
// the pool refuses (a double release), writes the stable line "cere: page_pool_fault op=release where=<where>
// status=<n>" to stderr. The pool is left exactly as the refusal left it.
void release_pages_logged(PagePool &pool, const uint32_t *ids, uint64_t n, const char *where) noexcept;

// K11's one-shot page order (ktest `cere_ktest_session_set_pages(model, ids, n)`, ledger R as amended by the owner
// 2026-10-10: no ktest session handle exists before M2.11): armed on a model, applied by its next session create to
// the pages that create reserved, and consumed by that create whatever its outcome; cleared on model release. Not
// synchronized: the owner holds its lock (CUDA: slots_mutex_).
class PagePermutation {
public:
    // `order` (n entries): a permutation of [0, n), the index into the reserved list of each page in session order.
    // CERE_ERR_ARGUMENT for a null order, n == 0, an index >= n or a repeated index (an earlier arming is kept);
    // otherwise replaces any earlier arming.
    cere_status arm(const uint32_t *order, uint64_t n);
    bool armed() const { return !order_.empty(); }
    void clear() { order_.clear(); }
    // Disarms. Nothing armed: CERE_OK, *pages unchanged. An arming of another size than *pages: CERE_ERR_ARGUMENT,
    // *pages unchanged. Otherwise (*pages)[i] becomes the old (*pages)[order[i]].
    cere_status apply(std::vector<uint32_t> *pages);

private:
    std::vector<uint32_t> order_;
};

}  // namespace cere::host
