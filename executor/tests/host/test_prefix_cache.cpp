// C2 (m210 design note Tier 1): the M2.10 prefix tree on the CPU toy device. Each case drives the tree through the
// ABI (namespaced sessions on a model with prefix_state_slots > 0) or, for the pure tree rules, through
// host/prefix_cache.h directly, and checks the tree, the page refcounts and the toy's fixed-state digest (ledger M4):
// a hit's state must equal the state of a cold run over the same tokens.
#include "check.h"
#include "cere_executor.h"
#include "host/kv_pool.h"
#include "host/prefix_cache.h"
#include "host/state.h"
#include "toy/toy_device.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

using cere::host::CacheNamespace;
using cere::host::CaptureKind;
using cere::host::PrefixCache;
using cere::host::PrefixNode;
using Tokens = std::vector<uint32_t>;

constexpr uint64_t kEpoch = 3;

cere_executor *make_executor(const char *device_id = "toy") {
    cere_executor_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = CERE_EXECUTOR_ABI_VERSION;
    options.device_id = device_id;
    options.worker_epoch = kEpoch;
    options.memory_budget_bytes = uint64_t(1) << 30;
    cere_executor *executor = nullptr;
    CHECK_EQ(cere_executor_create(&executor, &options), CERE_OK);
    return executor;
}

cere::toy::ToyDevice &toy(cere_executor *executor) {
    return static_cast<cere::toy::ToyDevice &>(*executor->device);
}

// The fixture model with M2.10's fields: `slots` prefix state slots, `pool` KV pool tokens (0 = derived) and a
// snapshot arena of `arena` bytes (the D9 host tier).
struct Model {
    cere_qwen4_exp_config config = cere::toy::fixture_config();
    std::vector<std::string> names{"t0", "t1"};
    std::vector<cere_tensor_desc> tensors;
    cere_model_desc desc{};

    Model(uint64_t slots, uint64_t max_sessions, uint64_t max_context, uint64_t pool = 0, uint64_t arena = 0) {
        for (size_t i = 0; i < names.size(); ++i) {
            cere_tensor_desc t{};
            t.struct_size = sizeof(t);
            t.abi_version = CERE_EXECUTOR_ABI_VERSION;
            t.name = names[i].c_str();
            t.dtype = CERE_DTYPE_U8;
            t.placement = CERE_PLACEMENT_DEVICE;
            t.ndim = 1;
            t.shape[0] = 16;
            t.bytes = 16;
            tensors.push_back(t);
        }
        desc.struct_size = sizeof(desc);
        desc.abi_version = CERE_EXECUTOR_ABI_VERSION;
        desc.model_kind = CERE_MODEL_QWEN4_EXP_NVFP4;
        desc.config = &config;
        desc.tensors = tensors.data();
        desc.tensor_count = tensors.size();
        desc.max_sessions = max_sessions;
        desc.max_context_tokens = max_context;
        desc.kv_pool_tokens = pool;
        desc.snapshot_host_bytes = arena;
        desc.prefix_state_slots = slots;
    }
};

cere_model *sealed(cere_executor *executor, Model &m) {
    cere_model *model = nullptr;
    CHECK_EQ(cere_model_create(executor, &m.desc, &model), CERE_OK);
    const std::vector<uint8_t> bytes(16, 7);
    for (uint64_t i = 0; i < m.tensors.size(); ++i) {
        CHECK_EQ(cere_model_put(executor, model, i, 0, bytes.data(), bytes.size()), CERE_OK);
    }
    CHECK_EQ(cere_model_seal(executor, model), CERE_OK);
    return model;
}

CacheNamespace ns(uint8_t b) {
    CacheNamespace id{};
    id.fill(b);
    return id;
}

// A session of `context` tokens, namespaced by `id` unless it is 0.
cere_session *open(cere_executor *executor, cere_model *model, uint64_t context, uint8_t id) {
    cere_session *session = nullptr;
    CHECK_EQ(cere_session_create(executor, model, context, &session), CERE_OK);
    if (session != nullptr && id != 0) {
        CHECK_EQ(cere_session_set_cache_namespace(executor, session, ns(id).data()), CERE_OK);
    }
    return session;
}

// Distinct pseudo-random token ids in [1, 69000] (the fixture vocab is 70,000); different seeds differ in page 0.
Tokens toks(uint32_t seed, uint64_t n) {
    Tokens t(n);
    for (uint64_t i = 0; i < n; ++i) t[i] = 1 + uint32_t((seed * 2654435761ull + i * 40503ull + seed * i) % 69000);
    return t;
}

Tokens join(const Tokens &a, uint64_t a_len, const Tokens &b) {
    Tokens t(a.begin(), a.begin() + a_len);
    t.insert(t.end(), b.begin(), b.end());
    return t;
}

// The toy's fixed-state digest of a session that consumed exactly `t` (ledger M4).
uint64_t digest(const Tokens &t) {
    uint64_t d = cere::toy::kDigestSeed;
    for (const uint32_t id : t) d = cere::toy::digest_step(d, id);
    return d;
}

uint64_t session_digest(cere_executor *executor, cere_session *session) {
    return toy(executor).session_state(*session->device).digest;
}

struct Job {
    Tokens prompt;
    Tokens out;
    std::vector<cere_token_score> scores;
    cere_work item{};
    cere_batch batch{};
    cere_completion_ticket ticket{};

    Job(cere_session *session, uint32_t op, Tokens tokens, uint64_t max_new, uint32_t capture = 0,
        uint64_t revision = 0)
        : prompt(std::move(tokens)), out(max_new, 0xffffffffu), scores(max_new) {
        item.struct_size = sizeof(item);
        item.abi_version = CERE_EXECUTOR_ABI_VERSION;
        item.operation = op;
        item.expected_worker_epoch = kEpoch;
        item.expected_session_revision = revision;
        item.session = session;
        item.tokens = prompt.empty() ? nullptr : prompt.data();
        item.token_count = prompt.size();
        item.max_new_tokens = max_new;
        item.out_tokens = out.data();
        item.out_capacity = out.size();
        item.out_scores = scores.data();
        item.capture_position = capture;
        batch.struct_size = sizeof(batch);
        batch.abi_version = CERE_EXECUTOR_ABI_VERSION;
        batch.expected_worker_epoch = kEpoch;
        batch.capability_profile_generation = session->model->generation;
        batch.items = &item;
        batch.item_count = 1;
    }
    cere_status submit(cere_executor *executor) { return cere_executor_submit(executor, &batch, &ticket); }
};

cere_completion completion_header() {
    cere_completion c{};
    c.struct_size = sizeof(c);
    c.abi_version = CERE_EXECUTOR_ABI_VERSION;
    return c;
}

// One poll; *done when it was terminal.
cere_completion poll(cere_executor *executor, cere_completion_ticket ticket, bool *done) {
    cere_completion c = completion_header();
    const cere_status st = cere_executor_poll(executor, ticket, &c);
    *done = st == CERE_OK;
    if (!*done) CHECK_EQ(st, CERE_PENDING);
    return c;
}

// Polls to the terminal completion; every poll reports `cached` as cached_tokens (P 44).
cere_completion drain(cere_executor *executor, cere_completion_ticket ticket, uint64_t cached) {
    for (uint64_t i = 0; i < 100000; ++i) {
        bool done = false;
        const cere_completion c = poll(executor, ticket, &done);
        CHECK_EQ(c.cached_tokens, cached);
        if (done) return c;
    }
    CHECK(false);
    return completion_header();
}

// A PREFILL of `prompt` (max_new tokens) on `session`, drained; checks status OK, the hit `cached` and the state
// digest (the device consumed the prompt and every generated token but the last).
Job prefill(cere_executor *executor, cere_session *session, const Tokens &prompt, uint64_t max_new, uint64_t cached,
            uint32_t capture = 0) {
    Job job(session, CERE_OP_PREFILL, prompt, max_new, capture);
    CHECK_EQ(job.submit(executor), CERE_OK);
    const cere_completion c = drain(executor, job.ticket, cached);
    CHECK_EQ(c.status, CERE_OK);
    CHECK_EQ(c.committed_tokens, max_new);
    Tokens consumed = prompt;
    consumed.insert(consumed.end(), job.out.begin(), job.out.end() - 1);
    CHECK(session_digest(executor, session) == digest(consumed));
    return job;
}

// prefill() on a fresh session namespaced by `id`, released afterwards.
void once(cere_executor *executor, cere_model *model, uint64_t context, uint8_t id, const Tokens &prompt,
          uint64_t cached, uint32_t capture = 0) {
    cere_session *session = open(executor, model, context, id);
    prefill(executor, session, prompt, 1, cached, capture);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
}

PrefixCache &tree(cere_model *model) { return *model->prefix; }

// The node of `ns` whose edge ends exactly at `len` along tokens `t` (stated or not), null when there is none.
const PrefixNode *node_at(cere_model *model, uint8_t id, const Tokens &t, uint64_t len) {
    const PrefixNode *node = tree(model).root(ns(id));
    uint64_t depth = 0;
    while (node != nullptr && depth < len) {
        cere::host::PageKey key;
        std::memcpy(key.data(), t.data() + depth, sizeof(key));
        const auto it = node->children.find(key);
        if (it == node->children.end()) return nullptr;
        node = it->second.get();
        for (uint64_t i = 0; i < node->key.size(); ++i) {
            if (depth + i >= t.size() || node->key[i] != t[depth + i]) return nullptr;
        }
        depth = node->depth;
    }
    return depth == len ? node : nullptr;
}

// The stated node at exactly `len` (a device slot or a host image), null when none.
const PrefixNode *state_at(cere_model *model, uint8_t id, const Tokens &t, uint64_t len) {
    const PrefixNode *node = node_at(model, id, t, len);
    return node != nullptr && node->stated() ? node : nullptr;
}

uint64_t free_pages(cere_executor *executor, cere_model *model) { return toy(executor).free_pages(*model->device); }

uint32_t refs(cere_executor *executor, cere_model *model, uint32_t page) {
    return toy(executor).page_refs(*model->device, page);
}

// --- tests -------------------------------------------------------------------------------------------------------

void test_tree_rules() {
    // Page-aligned insert/match through host/prefix_cache.h: a split parent gets no state (F4), truncation to the
    // deepest state within the cap, namespaces never share, lock/unlock, the slot pool (take free, else the LRU
    // unlocked state, else none when every state is locked) and LRU leaf eviction that walks up to the root.
    cere_executor *executor = make_executor();
    Model m(4, 1, 1024);
    cere_model *model = sealed(executor, m);
    PrefixCache &cache = tree(model);
    CHECK_EQ(cache.slots(), 5);  // prefix_state_slots + max_sessions (D6′)
    // An un-namespaced session binds its pages (16 ids) without touching the tree.
    const Tokens x = toks(1, 300);
    cere_session *session = open(executor, model, 1024, 0);
    prefill(executor, session, x, 1, 0);
    CHECK_EQ(cache.stats().captures, 0);
    CHECK(cache.root(ns(1)) == nullptr);
    const std::vector<uint32_t> pages = toy(executor).session_pages(*session->device);
    CHECK_EQ(pages.size(), 16);
    for (const uint32_t p : pages) CHECK_EQ(refs(executor, model, p), 1);

    uint32_t s[5] = {};
    for (uint32_t i = 0; i < 3; ++i) {
        CHECK(cache.take_slot(&s[i]));
        CHECK_EQ(s[i], i);  // the lowest free slot
    }
    CHECK(cache.insert(ns(1), x.data(), 256, pages.data(), s[0], CaptureKind::kTail, 10));
    const PrefixNode *xl = state_at(model, 1, x, 256);
    CHECK(xl != nullptr && xl->slot == s[0] && xl->pages.size() == 4 && xl->clock == 10);
    for (uint32_t i = 0; i < 4; ++i) CHECK_EQ(refs(executor, model, pages[i]), 2);  // session + tree
    // y shares x's first two pages: the edge splits at 128 and the split parent holds no state.
    const Tokens y = join(x, 128, toks(2, 172));
    CHECK(cache.insert(ns(1), y.data(), 256, pages.data(), s[1], CaptureKind::kTail, 11));
    const PrefixNode *split = node_at(model, 1, x, 128);
    CHECK(split != nullptr && !split->stated() && split->children.size() == 2 && split->pages.size() == 2);
    CHECK(state_at(model, 1, x, 256) == xl && xl->parent == split && xl->pages.size() == 2);
    const PrefixNode *yl = state_at(model, 1, y, 256);
    CHECK(yl != nullptr && yl->slot == s[1] && yl->parent == split);
    CHECK_EQ(refs(executor, model, pages[0]), 2);  // the split moved the tree's reference, it took none
    CHECK_EQ(refs(executor, model, pages[2]), 3);  // x's edge and y's edge (the same ids here) + the session
    // M210S-39: a state is attached only where the path's pages are the inserter's own. Other page ids for [0, 192)
    // (another session's KV rows over the same tokens) are refused with the tree unchanged.
    std::vector<uint32_t> foreign(pages.begin(), pages.end());
    foreign[1] += 100;
    CHECK(!cache.own_path(ns(1), x.data(), 192, foreign.data()));
    CHECK(cache.own_path(ns(1), x.data(), 192, pages.data()));
    CHECK(!cache.insert(ns(1), x.data(), 192, foreign.data(), s[2], CaptureKind::kBranch, 12));
    CHECK(node_at(model, 1, x, 192) == nullptr && split->children.size() == 2);
    // Foreign pages only past the existing path are the inserter's new edge: allowed.
    foreign[1] = pages[1];
    CHECK(cache.own_path(ns(1), toks(30, 256).data(), 256, foreign.data()));
    // Truncation: the deepest stated node within the cap; the split parent is passed over.
    PrefixCache::Match hit = cache.match(ns(1), x.data(), 300, 299 / 64 * 64, false);
    CHECK(hit.node == xl && hit.h == 256 && hit.k == 256);
    hit = cache.match(ns(1), x.data(), 300, 255, false);
    CHECK(hit.node == nullptr && hit.h == 0 && hit.k == 256);
    hit = cache.match(ns(1), join(x, 200, toks(3, 100)).data(), 300, 256, false);
    CHECK(hit.node == nullptr && hit.k == 192);  // page 3 differs: the page match ends at 192
    CHECK(cache.match(ns(2), x.data(), 300, 256, false).node == nullptr);  // namespaces never share
    // The split parent takes a state of its own key; a second insert there keeps the first state.
    CHECK(cache.insert(ns(1), x.data(), 128, pages.data(), s[2], CaptureKind::kCallEnd, 12));
    CHECK(split->slot == s[2]);
    CHECK(!cache.insert(ns(1), x.data(), 128, pages.data(), s[2], CaptureKind::kCallEnd, 13));
    CHECK_EQ(cache.match(ns(1), x.data(), 300, 255, false).h, 128);
    CHECK_EQ(cache.stats().inserts, 3);
    // Locks count on the node and every ancestor below the root.
    PrefixNode *xn = const_cast<PrefixNode *>(xl);
    PrefixNode *yn = const_cast<PrefixNode *>(yl);
    cache.lock(xn);
    CHECK(xl->lock == 1 && split->lock == 1 && yl->lock == 0 && cache.locked());
    // Two more slots taken (as decode track slots would be); then every state is on a locked path: none left.
    CHECK(cache.take_slot(&s[3]) && cache.take_slot(&s[4]));
    CHECK_EQ(cache.free_slots(), 0);
    cache.lock(yn);
    CHECK_EQ(split->lock, 2);
    uint32_t extra = 99;
    CHECK(!cache.take_slot(&extra));
    CHECK_EQ(cache.stats().state_evictions, 0);
    // Unlocking y frees its leaf state for the pool: the leaf is deleted (its pages released); the stated split
    // parent stays.
    cache.unlock(yn);
    CHECK(cache.take_slot(&extra));
    CHECK_EQ(extra, s[1]);
    CHECK_EQ(cache.stats().state_evictions, 1);
    CHECK(node_at(model, 1, y, 256) == nullptr && state_at(model, 1, x, 128) == split);
    CHECK_EQ(split->children.size(), 1);
    CHECK_EQ(refs(executor, model, pages[2]), 2);
    cache.unlock(xn);
    CHECK(!cache.locked() && xl->lock == 0 && split->lock == 0);
    // make_free: the LRU unlocked leaf first (x's), then the parent it leaves childless, then the namespace root is
    // gone. The session still holds every page, so nothing frees and make_free reports false.
    const uint64_t free = free_pages(executor, model);
    CHECK(!cache.make_free(free + 1));
    CHECK_EQ(cache.stats().leaf_evictions, 2);
    CHECK(cache.root(ns(1)) == nullptr);
    for (const uint32_t p : pages) CHECK_EQ(refs(executor, model, p), 1);
    for (const uint32_t slot : {extra, s[3], s[4]}) cache.free_slot(slot);
    CHECK_EQ(cache.free_slots(), 5);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(free_pages(executor, model), 16);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_deep_hit() {
    // A hit at 150,016 on a 200,000-token pool (3,125 pages): the first request's call ends (every 4,096) and its
    // (a′) floor64(n − 1) are captured into 5 slots; the second session binds the 2,344 shared pages plus 781 own
    // (the whole pool), resumes at h and ends with the cold run's state. Its own (a′) finds every state on its
    // locked path, so the capture is skipped and counted.
    cere_executor *executor = make_executor();
    Model m(4, 1, 200000, 200000);
    cere_model *model = sealed(executor, m);
    const Tokens p = toks(4, 150017);
    once(executor, model, 200000, 1, p, 0);
    PrefixCache &cache = tree(model);
    CHECK_EQ(cache.stats().captures, 37);  // 36 call ends + (a′)
    CHECK_EQ(cache.stats().inserts, 37);
    CHECK_EQ(cache.stats().state_evictions, 32);
    for (uint64_t at = 4096; at <= 147456; at += 4096) {
        const PrefixNode *node = node_at(model, 1, p, at);
        CHECK(node != nullptr && node->kind == CaptureKind::kCallEnd);
        CHECK_EQ(node != nullptr && node->stated(), at >= 135168);  // the 4 newest call ends keep their states
    }
    const PrefixNode *tail = state_at(model, 1, p, 150016);
    CHECK(tail != nullptr && tail->kind == CaptureKind::kTail && tail->children.empty());
    CHECK_EQ(free_pages(executor, model), 3125 - 2344);
    const Tokens q = join(p, 150016, toks(5, 100));
    cere_session *session = open(executor, model, 200000, 1);
    prefill(executor, session, q, 1, 150016);
    CHECK_EQ(cache.stats().hits, 1);
    CHECK_EQ(cache.stats().capture_skipped, 1);
    CHECK_EQ(cache.stats().skipped_prefill_slot, 1);  // PT0a: its (a′) capture
    CHECK_EQ(free_pages(executor, model), 0);
    const std::vector<uint32_t> bound = toy(executor).session_pages(*session->device);
    const std::vector<uint32_t> shared = cache.path_pages(tail);
    CHECK_EQ(bound.size(), 3125);
    CHECK_EQ(shared.size(), 2344);
    CHECK(std::equal(shared.begin(), shared.end(), bound.begin()));
    CHECK_EQ(refs(executor, model, shared[0]), 2);
    CHECK(tail->lock == 1);
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK(!cache.locked());
    CHECK_EQ(free_pages(executor, model), 3125 - 2344);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_aligned_prompt() {
    // n % 64 == 0: (a′) is floor64(n − 1) = n − 64; the state at n comes from the decode capture after the
    // prompt's last call, inserted at the terminal poll. A repeat matches at most floor64(n − 1) (the cap).
    cere_executor *executor = make_executor();
    Model m(4, 1, 1024);
    cere_model *model = sealed(executor, m);
    const Tokens p = toks(6, 256);
    once(executor, model, 1024, 1, p, 0);
    const PrefixNode *a = state_at(model, 1, p, 192);
    const PrefixNode *b = state_at(model, 1, p, 256);
    CHECK(a != nullptr && a->kind == CaptureKind::kTail);
    CHECK(b != nullptr && b->kind == CaptureKind::kDecode && b->parent == a);
    CHECK_EQ(tree(model).stats().inserts, 2);
    once(executor, model, 1024, 1, p, 192);
    CHECK_EQ(tree(model).stats().hits, 1);
    // A prompt of 257 tokens reaches the decode-captured state at 256.
    Tokens longer = p;
    longer.push_back(5);
    once(executor, model, 1024, 1, longer, 256);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_branching() {
    // (d) under M210S-39: b diverges from a inside a's edge; its lookup misses (no state within the match) and the
    // page match k = 512 splits b's calls at 512, but the path [0, 512) holds a's pages, not b's: the (d) capture
    // and b's (a′) (whose path runs through a's edge) are skipped and counted, the tree is unchanged and b ends in
    // the cold state. A request that hits a's state binds a's pages, so its (a′) past the path is stored.
    cere_executor *executor = make_executor();
    Model m(8, 1, 2048, 4096);  // room for a's edge beside a session's 32 pages (no leaf eviction at bind)
    cere_model *model = sealed(executor, m);
    PrefixCache &cache = tree(model);
    const Tokens a = toks(7, 1000);
    const Tokens b = join(a, 512, toks(8, 488));
    once(executor, model, 2048, 1, a, 0);
    const PrefixNode *tail = state_at(model, 1, a, 960);
    CHECK(tail != nullptr && tail->kind == CaptureKind::kTail);
    CHECK_EQ(cache.stats().inserts, 1);
    once(executor, model, 2048, 1, b, 0);
    CHECK_EQ(cache.stats().capture_page_mismatch, 2);  // (d) at 512 and (a′) at 960
    CHECK_EQ(cache.stats().captures, 1);
    CHECK_EQ(cache.stats().inserts, 1);
    CHECK(node_at(model, 1, a, 512) == nullptr && state_at(model, 1, a, 960) == tail);
    CHECK_EQ(cache.root(ns(1))->children.size(), 1);
    CHECK_EQ(cache.free_slots(), 8);
    // c hits a's state at 960 (a's pages [0, 960) bound shared) and stores its (a′) at 1,024 below a's leaf.
    const Tokens c = join(a, 960, toks(9, 100));
    once(executor, model, 2048, 1, c, 960);
    const PrefixNode *deeper = state_at(model, 1, c, 1024);
    CHECK(deeper != nullptr && deeper->kind == CaptureKind::kTail && deeper->parent == tail);
    CHECK_EQ(cache.stats().capture_page_mismatch, 2);
    CHECK_EQ(cache.stats().hits, 1);
    // A repeat of b still misses and still stores nothing.
    once(executor, model, 2048, 1, b, 0);
    CHECK_EQ(cache.stats().capture_page_mismatch, 4);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_refcount_outlives_session() {
    // D7: the tree and the hit session share s1's first pages; they outlive s1 and are freed at refcount 0 (flush).
    cere_executor *executor = make_executor();
    Model m(4, 2, 512);
    cere_model *model = sealed(executor, m);
    const Tokens p = toks(11, 200);
    cere_session *s1 = open(executor, model, 512, 1);
    prefill(executor, s1, p, 1, 0);
    const std::vector<uint32_t> own = toy(executor).session_pages(*s1->device);
    cere_session *s2 = open(executor, model, 512, 1);
    prefill(executor, s2, join(p, 192, toks(12, 50)), 1, 192);
    const std::vector<uint32_t> hit = toy(executor).session_pages(*s2->device);
    CHECK(std::equal(own.begin(), own.begin() + 3, hit.begin()));
    CHECK_EQ(refs(executor, model, own[0]), 3);
    CHECK_EQ(refs(executor, model, own[3]), 1);
    CHECK_EQ(cere_session_release(executor, s1), CERE_OK);
    CHECK_EQ(refs(executor, model, own[0]), 2);
    CHECK_EQ(refs(executor, model, own[3]), 0);
    CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_ERR_BUSY);  // s2 holds its hit lock
    CHECK_EQ(cere_session_release(executor, s2), CERE_OK);
    CHECK_EQ(refs(executor, model, own[0]), 1);
    CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
    CHECK_EQ(refs(executor, model, own[0]), 0);
    CHECK_EQ(free_pages(executor, model), 16);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_evict_before_capacity() {
    // D3′/D7 on a 10-page pool of 10-page sessions: admission counts unlocked tree pages as available, binding
    // evicts LRU unlocked leaves before CAPACITY, and only a locked path makes create CAPACITY.
    cere_executor *executor = make_executor();
    Model m(4, 2, 640, 640);
    cere_model *model = sealed(executor, m);
    const Tokens p = toks(13, 200);
    once(executor, model, 640, 1, p, 0);
    CHECK(state_at(model, 1, p, 192) != nullptr);
    CHECK_EQ(free_pages(executor, model), 7);
    const Tokens r = toks(14, 100);
    once(executor, model, 640, 1, r, 0);  // binds 10 own pages: p's leaf goes first
    CHECK_EQ(tree(model).stats().leaf_evictions, 1);
    CHECK(node_at(model, 1, p, 192) == nullptr && state_at(model, 1, r, 64) != nullptr);
    CHECK_EQ(free_pages(executor, model), 9);
    cere_session *s3 = open(executor, model, 640, 1);
    prefill(executor, s3, join(r, 64, toks(15, 36)), 1, 64);
    CHECK_EQ(free_pages(executor, model), 0);
    cere_session *s4 = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 640, &s4), CERE_ERR_CAPACITY);  // r's page is locked by s3
    CHECK(s4 == nullptr);
    CHECK_EQ(cere_session_release(executor, s3), CERE_OK);
    CHECK_EQ(cere_session_create(executor, model, 640, &s4), CERE_OK);
    // A second unbound session would need the same pages: the D3′ reservation refuses it.
    cere_session *s5 = nullptr;
    CHECK_EQ(cere_session_create(executor, model, 640, &s5), CERE_ERR_CAPACITY);
    CHECK_EQ(cere_session_release(executor, s4), CERE_OK);
    CHECK_EQ(cere_session_create(executor, model, 640, &s5), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_lru_logical_clock() {
    // D7: LRU by the logical clock (submission id), not by the order of insertion: t1's decode state is inserted at
    // its terminal poll after t2's prefill state, but t1 was submitted first, so it is evicted first.
    cere_executor *executor = make_executor();
    Model m(1, 2, 1024);
    cere_model *model = sealed(executor, m);
    const Tokens x = toks(16, 64);
    const Tokens y = toks(17, 200);
    cere_session *s1 = open(executor, model, 1024, 1);
    cere_session *s2 = open(executor, model, 1024, 1);
    Job t1(s1, CERE_OP_PREFILL, x, 70);
    CHECK_EQ(t1.submit(executor), CERE_OK);
    Job t2(s2, CERE_OP_PREFILL, y, 1);
    CHECK_EQ(t2.submit(executor), CERE_OK);
    CHECK_EQ(drain(executor, t2.ticket, 0).status, CERE_OK);
    CHECK_EQ(drain(executor, t1.ticket, 0).status, CERE_OK);
    Tokens x128 = x;
    x128.insert(x128.end(), t1.out.begin(), t1.out.begin() + 64);
    const PrefixNode *old = state_at(model, 1, x128, 128);
    const PrefixNode *young = state_at(model, 1, y, 192);
    CHECK(old != nullptr && old->kind == CaptureKind::kDecode && old->clock == t1.ticket.submission_id);
    CHECK(young != nullptr && young->clock == t2.ticket.submission_id);
    CHECK(old != nullptr && young != nullptr && old->seq > young->seq);
    CHECK_EQ(cere_session_release(executor, s1), CERE_OK);
    CHECK_EQ(cere_session_release(executor, s2), CERE_OK);
    once(executor, model, 1024, 1, toks(18, 200), 0);  // the third slot was free
    CHECK_EQ(tree(model).stats().state_evictions, 0);
    once(executor, model, 1024, 1, toks(19, 200), 0);
    CHECK_EQ(tree(model).stats().state_evictions, 1);
    CHECK(node_at(model, 1, x128, 128) == nullptr);
    CHECK(state_at(model, 1, y, 192) != nullptr);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_boundary_states() {
    // D4 (c) + (a′) in one ticket take two slots; D7 evicts (c) states first, so the boundary goes before an older
    // (a′) state; the internal node keeps its pages without a state; evicting the state of the leaf below it then
    // deletes the leaf and the state-less childless boundary node. (prefill_chunk 16,384: no call-end captures.)
    cere_executor *executor = make_executor("toy,prefill_chunk=16384");
    Model m(2, 1, 16384, 32768);  // a 256-page session binds beside the tree's pages (no leaf eviction at bind)
    cere_model *model = sealed(executor, m);
    PrefixCache &cache = tree(model);
    const Tokens x = toks(20, 200);
    const Tokens y = toks(21, 9100);
    once(executor, model, 16384, 1, x, 0);
    once(executor, model, 16384, 1, y, 0, 8448);
    const PrefixNode *boundary = state_at(model, 1, y, 8448);
    const PrefixNode *tail = state_at(model, 1, y, 9088);
    CHECK(boundary != nullptr && boundary->kind == CaptureKind::kBoundary && boundary->depth == 8448);
    CHECK(tail != nullptr && tail->kind == CaptureKind::kTail && tail->parent == boundary);
    CHECK(node_at(model, 1, y, 4096) == nullptr && node_at(model, 1, y, 8192) == nullptr);
    CHECK_EQ(cache.free_slots(), 0);
    const uint32_t boundary_page = boundary->pages.at(0);
    once(executor, model, 16384, 1, toks(22, 200), 0);
    CHECK(node_at(model, 1, y, 8448) == boundary && !boundary->stated());  // the (c) state went, not x's older one
    CHECK_EQ(refs(executor, model, boundary_page), 1);                    // the internal node keeps its pages
    CHECK(state_at(model, 1, x, 192) != nullptr);
    once(executor, model, 16384, 1, toks(23, 200), 0);  // LRU: x's state (the oldest clock)
    CHECK(node_at(model, 1, x, 192) == nullptr);
    once(executor, model, 16384, 1, toks(24, 200), 0);  // LRU: y's tail; its state-less parent goes with it
    CHECK(node_at(model, 1, y, 9088) == nullptr && node_at(model, 1, y, 8448) == nullptr);
    CHECK_EQ(refs(executor, model, boundary_page), 0);
    CHECK_EQ(free_pages(executor, model), 512 - 9);  // three 3-page leaves remain
    CHECK_EQ(cache.stats().state_evictions, 3);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_skip_when_all_locked() {
    // D4′: a capture that finds every slot on a locked path or a session's track slot is skipped and counted
    // (prefix_capture_skipped); the ticket still completes with the cold state.
    cere_executor *executor = make_executor("toy,hold_after=1");
    Model m(1, 2, 1024);
    cere_model *model = sealed(executor, m);
    PrefixCache &cache = tree(model);
    const Tokens x = toks(25, 100);
    const Tokens y = toks(26, 100);
    once(executor, model, 1024, 1, x, 0);
    once(executor, model, 1024, 1, y, 0);
    CHECK_EQ(cache.free_slots(), 1);
    // s2 hits x@64 (locked) and decode-captures at 128 into the last free slot, then holds at a token boundary.
    const Tokens x2 = join(x, 64, toks(27, 64));
    cere_session *s2 = open(executor, model, 1024, 1);
    Job held(s2, CERE_OP_PREFILL, x2, 4);
    CHECK_EQ(held.submit(executor), CERE_OK);
    cere_completion c = completion_header();
    for (int i = 0; i < 8; ++i) {
        bool done = false;
        c = poll(executor, held.ticket, &done);
        CHECK(!done);
        CHECK_EQ(c.cached_tokens, 64);
    }
    CHECK_EQ(c.committed_tokens, 1);
    CHECK_EQ(cache.free_slots(), 0);
    CHECK_EQ(s2->tracked, 128);
    // s1 hits y@64 (locked): its (a′) at 192 finds no slot.
    const Tokens y2 = join(y, 64, toks(28, 136));
    cere_session *s1 = open(executor, model, 1024, 1);
    prefill(executor, s1, y2, 1, 64);
    CHECK_EQ(cache.stats().capture_skipped, 1);
    CHECK_EQ(cache.stats().skipped_prefill_slot, 1);  // PT0a: a prefill capture, not the held decode
    CHECK_EQ(cache.stats().state_evictions, 0);
    CHECK(node_at(model, 1, y2, 192) == nullptr);
    // The cancelled ticket ends OK at its token boundary and inserts its decode capture.
    CHECK_EQ(cere_executor_cancel(executor, held.ticket), CERE_OK);
    c = drain(executor, held.ticket, 64);
    CHECK_EQ(c.status, CERE_OK);
    CHECK_EQ(c.finish, CERE_FINISH_CANCELLED);
    Tokens x128 = x2;
    const PrefixNode *decoded = state_at(model, 1, x128, 128);
    CHECK(decoded != nullptr && decoded->kind == CaptureKind::kDecode);
    CHECK_EQ(cere_session_release(executor, s1), CERE_OK);
    CHECK_EQ(cere_session_release(executor, s2), CERE_OK);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_namespace_isolation() {
    // D2: root children are keyed by the namespace; the same prompt under another namespace misses and inserts its
    // own path.
    cere_executor *executor = make_executor();
    Model m(4, 1, 1024, 4096);
    cere_model *model = sealed(executor, m);
    const Tokens p = toks(29, 300);
    once(executor, model, 1024, 1, p, 0);
    once(executor, model, 1024, 2, p, 0);
    const PrefixNode *a = state_at(model, 1, p, 256);
    const PrefixNode *b = state_at(model, 2, p, 256);
    CHECK(a != nullptr && b != nullptr && a != b);
    CHECK(a->pages != b->pages);
    once(executor, model, 1024, 2, p, 256);
    once(executor, model, 1024, 3, p, 0);
    CHECK_EQ(tree(model).stats().hits, 1);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_alt_edge_rules() {
    // m210 D17 on the tree: with alternative edges on, an insert over other page ids for tokens the tree holds is never
    // refused (own_path) and adds an edge beside the existing one (counted); a match takes the deepest state over all
    // alternatives, the most recently used among equally deep ones, and k over all of them; given a session's pages,
    // only its own path. An unlocked end-state leaf over other pages yields to an insert whose tokens repeat its whole
    // key (O112) and stays beside one that diverges inside it. LRU leaf eviction erases exactly the evicted
    // alternative: a locked one beside it stays.
    cere_executor *executor = make_executor();
    Model m(10, 1, 1024);
    cere_model *model = sealed(executor, m);
    PrefixCache &cache = tree(model);
    CHECK(!cache.alt_edges());
    CHECK_EQ(cere_model_set_prefix_alt_edges(executor, model, 1), CERE_OK);
    CHECK(cache.alt_edges());
    cere_session *session = open(executor, model, 1024, 0);
    prefill(executor, session, toks(1, 300), 1, 0);
    const std::vector<uint32_t> pages = toy(executor).session_pages(*session->device);
    CHECK_EQ(pages.size(), 16);
    const std::vector<uint32_t> other(pages.rbegin(), pages.rend());  // other ids for the same rows
    std::vector<uint32_t> mixed(pages.begin(), pages.begin() + 4);     // a session that hit pages' state at 256
    mixed.insert(mixed.end(), other.begin() + 4, other.end());
    const auto slot = [&]() {
        uint32_t s = 0;
        CHECK(cache.take_slot(&s));
        return s;
    };
    const Tokens t = toks(1, 600);

    CHECK(cache.insert(ns(1), t.data(), 256, pages.data(), slot(), CaptureKind::kTail, 10));
    CHECK(cache.own_path(ns(1), t.data(), 192, other.data()));
    CHECK(cache.insert(ns(1), t.data(), 128, other.data(), slot(), CaptureKind::kCallEnd, 11));
    CHECK_EQ(cache.stats().alt_edges, 1);
    const PrefixNode *root = cache.root(ns(1));
    CHECK(root != nullptr && root->children.size() == 2);
    const PrefixNode *mine = cache.match(ns(1), t.data(), 300, 256, false, 0, pages.data()).node;
    const PrefixNode *theirs = cache.match(ns(1), t.data(), 300, 256, false, 0, other.data()).node;
    CHECK(mine != nullptr && mine->depth == 256 && mine->pages.front() == pages[0] && mine->parent == root);
    CHECK(theirs != nullptr && theirs->depth == 128 && theirs->pages.front() == other[0] && theirs->parent == root);
    PrefixCache::Match hit = cache.match(ns(1), t.data(), 300, 256, false);
    CHECK(hit.node == mine && hit.h == 256 && hit.k == 256);
    hit = cache.match(ns(1), t.data(), 300, 255, false);
    CHECK(hit.node == theirs && hit.h == 128 && hit.k == 256);
    // Equally deep: other's 256 (used at 12) wins over pages' (10) until pages' is used again.
    CHECK(cache.insert(ns(1), t.data(), 256, other.data(), slot(), CaptureKind::kTail, 12));
    CHECK_EQ(cache.stats().alt_edges, 1);  // below its own 128 edge: nothing beside it
    const PrefixNode *theirs256 = cache.match(ns(1), t.data(), 300, 256, false, 0, other.data()).node;
    CHECK(theirs256 != nullptr && theirs256->depth == 256 && theirs256->parent == theirs);
    CHECK(cache.match(ns(1), t.data(), 300, 256, false).node == theirs256);
    cache.touch(const_cast<PrefixNode *>(mine), 13);
    CHECK(cache.match(ns(1), t.data(), 300, 256, false).node == mine);

    // An end-state leaf (class 94 for n = 350, chunk 128) over mixed's pages below pages' 256 yields to pages' insert
    // at 384 (t repeats its whole key [256, 320)), a state eviction that frees its pair.
    uint32_t pair = 0;
    CHECK(cache.take_run(&pair, 2, ns(1), t.data(), 320, mixed.data()));
    CHECK(cache.insert(ns(1), t.data(), 320, mixed.data(), pair, CaptureKind::kTail, 14, t[320], 94));
    const PrefixNode *end = cache.match(ns(1), t.data(), 350, 320, false, 128, mixed.data()).node;
    CHECK(end != nullptr && end->cls == 94 && end->parent == mine);
    const uint64_t free_slots = cache.free_slots();
    const uint64_t evictions = cache.stats().state_evictions;
    CHECK(cache.insert(ns(1), t.data(), 384, pages.data(), slot(), CaptureKind::kTail, 15));
    CHECK_EQ(cache.stats().state_evictions, evictions + 1);
    CHECK_EQ(cache.free_slots(), free_slots + 1);  // the pair freed, one slot taken
    CHECK(cache.match(ns(1), t.data(), 350, 320, false, 128).node == nullptr);
    const PrefixNode *mine384 = cache.match(ns(1), t.data(), 400, 384, false, 0, pages.data()).node;
    CHECK(mine384 != nullptr && mine384->depth == 384 && mine384->parent == mine && mine->children.size() == 1);
    CHECK_EQ(cache.stats().alt_edges, 1);

    // y diverges from t inside an end-state leaf [256, 448) over mixed's pages: pages' insert at 512 leaves it and adds
    // its own edge beside it.
    const Tokens y = join(t, 320, toks(2, 280));
    CHECK(cache.insert(ns(2), t.data(), 256, pages.data(), slot(), CaptureKind::kTail, 16));
    CHECK(cache.take_run(&pair, 2, ns(2), y.data(), 448, mixed.data()));
    CHECK(cache.insert(ns(2), y.data(), 448, mixed.data(), pair, CaptureKind::kTail, 17, y[448], 116));
    const PrefixNode *yend = cache.match(ns(2), y.data(), 500, 448, false, 128, mixed.data()).node;
    CHECK(yend != nullptr && yend->cls == 116 && yend->depth == 448);
    const uint64_t kept = cache.stats().state_evictions;
    CHECK(cache.insert(ns(2), t.data(), 512, pages.data(), slot(), CaptureKind::kTail, 18));
    CHECK_EQ(cache.stats().state_evictions, kept);
    CHECK_EQ(cache.stats().alt_edges, 2);
    CHECK(cache.match(ns(2), y.data(), 500, 448, false, 128).node == yend);
    const PrefixNode *base = cache.match(ns(2), t.data(), 300, 256, false).node;
    CHECK(base != nullptr && base->children.size() == 2);
    CHECK_EQ(cache.match(ns(2), t.data(), 600, 512, false).h, 512);

    // Eviction: with pages' 384 locked, every other leaf goes (the session holds every page, so nothing frees);
    // erasing other's alternatives leaves pages' edge beside them.
    cache.lock(const_cast<PrefixNode *>(mine384));
    CHECK(!cache.make_free(free_pages(executor, model) + 1));
    CHECK(cache.root(ns(2)) == nullptr);
    root = cache.root(ns(1));
    CHECK(root != nullptr && root->children.size() == 1 && root->children.begin()->second.get() == mine);
    CHECK(cache.match(ns(1), t.data(), 400, 384, false).node == mine384);
    cache.unlock(const_cast<PrefixNode *>(mine384));
    CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
    CHECK_EQ(cache.free_slots(), cache.slots());
    CHECK_EQ(cere_session_release(executor, session), CERE_OK);
    CHECK_EQ(free_pages(executor, model), 16);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_row_sig_match() {
    // m210 D15⁗″ (ResumeExact): a drafting lookup (draft chunk 128) takes an end state at b = 320 (s = 256) whose class
    // κ has the lookup's own class κ′'s row-bit signature, σ(256, κ) == σ(256, κ′) (Device::row_sig). The toy's
    // default σ is the class (D15⁗′'s equal-class rule): the node's κ 94 (n = 350) serves n = 350 only, and κ′ 114
    // (n = 370) takes the draft state at 256. ",row_sig=65-127" gives 94 and 114 one σ, so n = 370 hits at 320;
    // ",row_sig=65-100,row_sig=101-127" splits them and n = 370 misses again. A non-drafting lookup never takes an end
    // state, whatever σ.
    for (const char *id : {"toy", "toy,row_sig=65-127", "toy,row_sig=65-100,row_sig=101-127"}) {
        const bool shared = std::string(id) == "toy,row_sig=65-127";
        cere_executor *executor = make_executor(id);
        Model m(10, 1, 1024);
        cere_model *model = sealed(executor, m);
        PrefixCache &cache = tree(model);
        cere_session *session = open(executor, model, 1024, 0);
        const Tokens t = toks(3, 400);
        prefill(executor, session, t, 1, 0);
        const std::vector<uint32_t> pages = toy(executor).session_pages(*session->device);
        uint32_t slot = 0;
        CHECK(cache.take_slot(&slot));
        CHECK(cache.insert(ns(1), t.data(), 256, pages.data(), slot, CaptureKind::kTail, 10, t[256]));
        uint32_t pair = 0;
        CHECK(cache.take_run(&pair, 2, ns(1), t.data(), 320, pages.data()));
        CHECK(cache.insert(ns(1), t.data(), 320, pages.data(), pair, CaptureKind::kTail, 11, t[320], 94));
        const PrefixCache::Match same = cache.match(ns(1), t.data(), 350, 320, false, 128);
        CHECK(same.h == 320 && same.node != nullptr && same.node->cls == 94);
        const PrefixCache::Match longer = cache.match(ns(1), t.data(), 370, 320, false, 128);
        CHECK_EQ(longer.h, shared ? 320 : 256);
        CHECK(longer.node != nullptr && longer.node->cls == (shared ? 94u : 0u));
        CHECK_EQ(cache.match(ns(1), t.data(), 370, 320, false).h, 256);
        CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
        CHECK_EQ(cache.free_slots(), cache.slots());
        CHECK_EQ(cere_session_release(executor, session), CERE_OK);
        CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
    }
}

void test_alt_edges_branching() {
    // m210 D17: test_branching with alternative edges on. b diverges from a inside a's edge; its (d) at 512 and (a′)
    // at 960 go on an alternative edge over b's own pages beside a's (nothing refused), so a repeat of b hits its own
    // 960 state over its own pages and a repeat of a still hits a's; c, sharing [0, 512) with both, hits b's (d)
    // state. Every run ends in its cold state (prefill checks the digest). With the setting off the same sequence is
    // test_branching's (b never hits).
    cere_executor *executor = make_executor();
    const Tokens a = toks(7, 1000);
    const Tokens b = join(a, 512, toks(8, 488));
    const Tokens c = join(a, 512, toks(9, 300));
    for (const uint32_t on : {0u, 1u}) {
        Model m(8, 1, 2048, 4096);
        cere_model *model = sealed(executor, m);
        PrefixCache &cache = tree(model);
        CHECK_EQ(cere_model_set_prefix_alt_edges(executor, model, on), CERE_OK);
        once(executor, model, 2048, 1, a, 0);
        const PrefixNode *tail = cache.match(ns(1), a.data(), 1000, 960, false).node;
        CHECK(tail != nullptr && tail->depth == 960);
        once(executor, model, 2048, 1, b, 0);
        CHECK_EQ(cache.stats().capture_page_mismatch, on ? 0 : 2);
        CHECK_EQ(cache.stats().inserts, on ? 3 : 1);
        CHECK_EQ(cache.stats().alt_edges, on);
        CHECK_EQ(cache.root(ns(1))->children.size(), on ? 2 : 1);
        // b again: on, its own 960 over its own pages (a's [0, 512) pages are not bound); off, a miss.
        cere_session *s = open(executor, model, 2048, 1);
        prefill(executor, s, b, 1, on ? 960 : 0);
        if (on) {
            const PrefixNode *own = cache.match(ns(1), b.data(), 1000, 960, false).node;
            CHECK(own != nullptr && own != tail && own->depth == 960);
            const std::vector<uint32_t> bound = toy(executor).session_pages(*s->device);
            const std::vector<uint32_t> path = cache.path_pages(own);
            CHECK(path.size() == 15 && std::equal(path.begin(), path.end(), bound.begin()));
            CHECK(cache.path_pages(tail).front() != bound.front());
        }
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        once(executor, model, 2048, 1, a, 960);
        once(executor, model, 2048, 1, c, on ? 512 : 0);
        CHECK_EQ(cache.stats().hits, on ? 3 : 1);
        CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
        CHECK_EQ(free_pages(executor, model), 64);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    }
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

void test_alt_edges_eviction() {
    // m210 D17 under page pressure (D3′/D7): alternative edges are ordinary tree leaves. A 20-page pool of 10-page
    // sessions: a (n = 300, (a′) 256) and b (a's first page, then other tokens: (d) 64 and (a′) 256 on an alternative
    // edge over b's pages) hold 8 pages. A session s hits a's 256 and stays open (a's path locked); c (a's first page,
    // then others) hits b's (d) state at 64 and must evict for its 9 own pages: the LRU unlocked leaf (b's [64, 256))
    // goes, a's locked alternative and b's locked (d) node stay. Afterwards a repeat of a still hits 256, of b only 64.
    cere_executor *executor = make_executor();
    Model m(8, 2, 640, 1280);
    cere_model *model = sealed(executor, m);
    PrefixCache &cache = tree(model);
    CHECK_EQ(cere_model_set_prefix_alt_edges(executor, model, 1), CERE_OK);
    const Tokens a = toks(20, 300);
    const Tokens b = join(a, 64, toks(21, 236));
    const Tokens c = join(a, 64, toks(22, 236));
    once(executor, model, 640, 1, a, 0);
    once(executor, model, 640, 1, b, 0);
    CHECK_EQ(cache.stats().alt_edges, 1);
    CHECK_EQ(cache.stats().capture_page_mismatch, 0);
    CHECK_EQ(cache.match(ns(1), b.data(), 300, 256, false).h, 256);
    CHECK_EQ(free_pages(executor, model), 12);
    cere_session *s = open(executor, model, 640, 1);
    prefill(executor, s, join(a, 256, toks(23, 44)), 1, 256);
    CHECK_EQ(free_pages(executor, model), 6);
    const uint64_t evicted = cache.stats().leaf_evictions;
    cere_session *s2 = open(executor, model, 640, 1);
    prefill(executor, s2, c, 1, 64);
    CHECK_EQ(cache.stats().leaf_evictions, evicted + 1);
    CHECK_EQ(cache.match(ns(1), b.data(), 300, 256, false).h, 64);  // b's (a′) leaf went, its (d) state stayed
    const PrefixNode *a256 = cache.match(ns(1), a.data(), 300, 256, false).node;
    CHECK(a256 != nullptr && a256->depth == 256 && a256->lock == 1);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);
    CHECK_EQ(cere_session_release(executor, s2), CERE_OK);
    once(executor, model, 640, 1, a, 256);
    once(executor, model, 640, 1, b, 64);
    CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
    CHECK_EQ(free_pages(executor, model), 20);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// cere_prefix_stats with a valid header.
cere_prefix_stats stats_header() {
    cere_prefix_stats s{};
    s.struct_size = sizeof(s);
    s.abi_version = CERE_EXECUTOR_ABI_VERSION;
    return s;
}

cere_prefix_stats read_stats(cere_executor *executor, cere_model *model) {
    cere_prefix_stats s = stats_header();
    CHECK_EQ(cere_model_prefix_stats(executor, model, &s), CERE_OK);
    return s;
}

// Every exported counter equals the tree's own (the export is a read of PrefixStats, nothing derived).
void check_stats_equal_tree(const cere_prefix_stats &s, const cere::host::PrefixStats &t, uint32_t alt_on) {
    CHECK_EQ(s.struct_size, sizeof(cere_prefix_stats));
    CHECK_EQ(s.abi_version, CERE_EXECUTOR_ABI_VERSION);
    CHECK_EQ(s.alt_edges_on, alt_on);
    CHECK_EQ(s.reserved, 0);
    CHECK_EQ(s.hits, t.hits);
    CHECK_EQ(s.captures, t.captures);
    CHECK_EQ(s.capture_skipped, t.capture_skipped);
    CHECK_EQ(s.capture_page_mismatch, t.capture_page_mismatch);
    CHECK_EQ(s.inserts, t.inserts);
    CHECK_EQ(s.state_evictions, t.state_evictions);
    CHECK_EQ(s.leaf_evictions, t.leaf_evictions);
    CHECK_EQ(s.host_spills, t.host_spills);
    CHECK_EQ(s.host_loads, t.host_loads);
    CHECK_EQ(s.host_drops, t.host_drops);
    CHECK_EQ(s.alt_edges, t.alt_edges);
}

void test_prefix_stats_export() {
    // cere_model_prefix_stats (ABI v4 append): the tree's counters through the C ABI. test_alt_edges_branching's
    // scenario: b diverges from a inside a's edge. With alternative edges off its (d) and (a′) captures are refused by
    // M210S-39 (capture_page_mismatch 2, alt_edges 0); on, nothing is refused and one alternative edge is added
    // (capture_page_mismatch 0, alt_edges 1). Counters never decrease (a flush keeps them), a read changes nothing
    // and is allowed with a ticket in flight; the refusals leave an output zero.
    cere_executor *executor = make_executor();
    const Tokens a = toks(7, 1000);
    const Tokens b = join(a, 512, toks(8, 488));
    for (const uint32_t on : {0u, 1u}) {
        Model m(8, 1, 2048, 4096);
        cere_model *model = sealed(executor, m);
        PrefixCache &cache = tree(model);
        CHECK_EQ(cere_model_set_prefix_alt_edges(executor, model, on), CERE_OK);
        const cere_prefix_stats fresh = read_stats(executor, model);
        check_stats_equal_tree(fresh, cache.stats(), on);
        CHECK_EQ(fresh.hits + fresh.captures + fresh.inserts + fresh.capture_page_mismatch + fresh.alt_edges, 0);

        once(executor, model, 2048, 1, a, 0);
        const cere_prefix_stats after_a = read_stats(executor, model);
        check_stats_equal_tree(after_a, cache.stats(), on);
        CHECK(after_a.captures > 0 && after_a.inserts > 0);
        CHECK_EQ(after_a.capture_page_mismatch, 0);
        CHECK_EQ(after_a.alt_edges, 0);

        once(executor, model, 2048, 1, b, 0);
        const cere_prefix_stats after_b = read_stats(executor, model);
        check_stats_equal_tree(after_b, cache.stats(), on);
        CHECK_EQ(after_b.capture_page_mismatch, on ? 0 : 2);
        CHECK_EQ(after_b.alt_edges, on);
        CHECK_EQ(after_b.inserts, on ? 3 : 1);
        CHECK(after_b.captures >= after_a.captures && after_b.inserts >= after_a.inserts);

        // A hit is counted; a read in between changes nothing; one ticket in flight does not refuse it.
        cere_session *s = open(executor, model, 2048, 1);
        Job job(s, CERE_OP_PREFILL, a, 1);
        CHECK_EQ(job.submit(executor), CERE_OK);
        const cere_prefix_stats flying = read_stats(executor, model);
        const cere_prefix_stats again = read_stats(executor, model);
        CHECK(std::memcmp(&flying, &again, sizeof(flying)) == 0);
        check_stats_equal_tree(flying, cache.stats(), on);
        const cere_completion c = drain(executor, job.ticket, 960);
        CHECK_EQ(c.status, CERE_OK);
        CHECK_EQ(cere_session_release(executor, s), CERE_OK);
        const cere_prefix_stats after_hit = read_stats(executor, model);
        check_stats_equal_tree(after_hit, cache.stats(), on);
        CHECK_EQ(after_hit.hits, after_b.hits + 1);

        CHECK_EQ(cere_model_flush_prefix_cache(executor, model), CERE_OK);
        const cere_prefix_stats flushed = read_stats(executor, model);
        CHECK_EQ(flushed.hits, after_hit.hits);
        CHECK_EQ(flushed.captures, after_hit.captures);
        CHECK_EQ(flushed.capture_page_mismatch, after_hit.capture_page_mismatch);
        CHECK_EQ(flushed.alt_edges, after_hit.alt_edges);
        CHECK_EQ(cere_model_release(executor, model), CERE_OK);
    }

    // Refusals. A model without a prefix cache is UNSUPPORTED with the body zeroed and the header kept; a bad
    // struct_size is ARGUMENT, a bad abi_version UNSUPPORTED, both leaving the struct untouched; null pointers and
    // foreign handles are ARGUMENT.
    Model off(0, 1, 1024);
    cere_model *cache_off = sealed(executor, off);
    cere_prefix_stats s = stats_header();
    s.hits = 77;
    CHECK_EQ(cere_model_prefix_stats(executor, cache_off, &s), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(s.hits, 0);
    CHECK_EQ(s.struct_size, sizeof(s));
    CHECK_EQ(s.abi_version, CERE_EXECUTOR_ABI_VERSION);
    Model on(4, 1, 1024);
    cere_model *cached = sealed(executor, on);
    cere_prefix_stats bad = stats_header();
    bad.struct_size = sizeof(bad) - 8;
    bad.hits = 5;
    CHECK_EQ(cere_model_prefix_stats(executor, cached, &bad), CERE_ERR_ARGUMENT);
    CHECK_EQ(bad.hits, 5);
    bad = stats_header();
    bad.abi_version = CERE_EXECUTOR_ABI_VERSION - 1;
    bad.hits = 5;
    CHECK_EQ(cere_model_prefix_stats(executor, cached, &bad), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(bad.hits, 5);
    CHECK_EQ(cere_model_prefix_stats(executor, cached, nullptr), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_prefix_stats(nullptr, cached, &s), CERE_ERR_ARGUMENT);
    cere_prefix_stats unknown = stats_header();
    CHECK_EQ(cere_model_prefix_stats(executor, reinterpret_cast<cere_model *>(&unknown), &unknown), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_prefix_stats(executor, nullptr, &unknown), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

// cere_prefix_capture_stats with a valid header.
cere_prefix_capture_stats sites_header() {
    cere_prefix_capture_stats s{};
    s.struct_size = sizeof(s);
    s.abi_version = CERE_EXECUTOR_ABI_VERSION;
    return s;
}

// cere_model_prefix_capture_stats, checked against the tree's own counters; the sites sum to capture_skipped.
cere_prefix_capture_stats read_sites(cere_executor *executor, cere_model *model) {
    cere_prefix_capture_stats s = sites_header();
    CHECK_EQ(cere_model_prefix_capture_stats(executor, model, &s), CERE_OK);
    const cere::host::PrefixStats &t = tree(model).stats();
    CHECK_EQ(s.struct_size, sizeof(s));
    CHECK_EQ(s.abi_version, CERE_EXECUTOR_ABI_VERSION);
    CHECK_EQ(s.skipped_prefill_slot, t.skipped_prefill_slot);
    CHECK_EQ(s.skipped_decode_track, t.skipped_decode_track);
    CHECK_EQ(s.skipped_end_pair, t.skipped_end_pair);
    CHECK_EQ(s.hit_shortfall_tokens, t.hit_shortfall_tokens);
    CHECK_EQ(s.hit_shortfall_lookups, t.hit_shortfall_lookups);
    CHECK_EQ(s.skipped_prefill_slot + s.skipped_decode_track + s.skipped_end_pair, t.capture_skipped);
    return s;
}

void test_capture_skip_sites() {
    // PT0a (prefix-token design note F12): a session's decode track capture that finds no slot is retried at every
    // later 64-token boundary its decode reaches; it counts once per ticket (site decode_track), not once per
    // boundary. Two slots, both on the hit path of the decoding session (64 and 192, locked): its 1,281-token decode
    // from 193 crosses 20 boundaries (256 .. 1,472), none can take a slot and nothing is evicted. A second ticket of
    // the session counts once more. The export's refusals follow cere_model_prefix_stats.
    cere_executor *executor = make_executor();
    Model m(1, 1, 4096);
    cere_model *model = sealed(executor, m);
    PrefixCache &cache = tree(model);
    CHECK_EQ(cache.slots(), 2);
    const Tokens x = toks(29, 200);
    once(executor, model, 4096, 1, Tokens(x.begin(), x.begin() + 100), 0);  // (a′) at 64
    once(executor, model, 4096, 1, x, 64);                                   // (a′) at 192
    CHECK(state_at(model, 1, x, 64) != nullptr && state_at(model, 1, x, 192) != nullptr);
    CHECK_EQ(cache.free_slots(), 0);
    CHECK_EQ(read_sites(executor, model).skipped_decode_track, 0);
    const uint64_t captures = cache.stats().captures;
    cere_session *s = open(executor, model, 4096, 1);
    const Tokens prompt(x.begin(), x.begin() + 193);
    prefill(executor, s, prompt, 1281, 192);
    cere_prefix_capture_stats sites = read_sites(executor, model);
    CHECK_EQ(sites.skipped_decode_track, 1);  // 20 boundaries, one ticket
    CHECK_EQ(sites.skipped_prefill_slot, 0);
    CHECK_EQ(sites.skipped_end_pair, 0);
    CHECK_EQ(cache.stats().capture_skipped, 1);
    CHECK_EQ(cache.stats().captures, captures);
    CHECK_EQ(cache.stats().state_evictions, 0);
    CHECK_EQ(s->track_slot, PrefixNode::kNoSlot);
    // The session's next ticket (a DECODE of 130 tokens over 2 more boundaries, 1,536 and 1,600) counts once more.
    Job second(s, CERE_OP_DECODE, {}, 130, 0, s->revision);
    CHECK_EQ(second.submit(executor), CERE_OK);
    CHECK_EQ(drain(executor, second.ticket, 0).status, CERE_OK);
    sites = read_sites(executor, model);
    CHECK_EQ(sites.skipped_decode_track, 2);
    CHECK_EQ(sites.skipped_prefill_slot + sites.skipped_end_pair, 0);
    CHECK_EQ(cache.stats().capture_skipped, 2);
    CHECK_EQ(cere_session_release(executor, s), CERE_OK);

    // Refusals, as cere_model_prefix_stats: a model without a prefix cache is UNSUPPORTED (body zeroed, header kept),
    // a bad struct_size ARGUMENT and a bad abi_version UNSUPPORTED (struct untouched), null and foreign handles
    // ARGUMENT.
    Model off(0, 1, 1024);
    cere_model *cache_off = sealed(executor, off);
    cere_prefix_capture_stats z = sites_header();
    z.skipped_end_pair = 77;
    CHECK_EQ(cere_model_prefix_capture_stats(executor, cache_off, &z), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(z.skipped_end_pair, 0);
    CHECK_EQ(z.struct_size, sizeof(z));
    cere_prefix_capture_stats bad = sites_header();
    bad.struct_size = sizeof(bad) - 8;
    bad.skipped_end_pair = 5;
    CHECK_EQ(cere_model_prefix_capture_stats(executor, model, &bad), CERE_ERR_ARGUMENT);
    CHECK_EQ(bad.skipped_end_pair, 5);
    bad = sites_header();
    bad.abi_version = CERE_EXECUTOR_ABI_VERSION - 1;
    bad.skipped_end_pair = 5;
    CHECK_EQ(cere_model_prefix_capture_stats(executor, model, &bad), CERE_ERR_UNSUPPORTED);
    CHECK_EQ(bad.skipped_end_pair, 5);
    CHECK_EQ(cere_model_prefix_capture_stats(executor, model, nullptr), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_prefix_capture_stats(nullptr, model, &z), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_prefix_capture_stats(executor, reinterpret_cast<cere_model *>(&z), &z), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_model_prefix_capture_stats(executor, nullptr, &z), CERE_ERR_ARGUMENT);
    CHECK_EQ(cere_executor_destroy(executor), CERE_OK);
}

}  // namespace

int main() {
    test_tree_rules();
    test_deep_hit();
    test_aligned_prompt();
    test_branching();
    test_refcount_outlives_session();
    test_evict_before_capacity();
    test_lru_logical_clock();
    test_boundary_states();
    test_skip_when_all_locked();
    test_namespace_isolation();
    test_alt_edge_rules();
    test_row_sig_match();
    test_alt_edges_branching();
    test_alt_edges_eviction();
    test_prefix_stats_export();
    test_capture_skip_sites();
    return cere::test::finish("test_prefix_cache");
}
