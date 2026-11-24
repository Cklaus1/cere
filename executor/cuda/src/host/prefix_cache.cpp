#include "host/prefix_cache.h"

#include "host/kv_pool.h"
#include "host/snapshot_format.h"
#include "host/state.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace cere::host {

namespace {

PageKey page_key(const uint32_t *tokens) {
    PageKey key;
    std::copy(tokens, tokens + kKvPageTokens, key.begin());
    return key;
}

// LRU order (D7): the older logical clock, then the older use; equal uses (one touch marks a whole path) evict the
// deeper node first.
bool older(const PrefixNode &a, const PrefixNode &b) {
    if (a.clock != b.clock) return a.clock < b.clock;
    if (a.seq != b.seq) return a.seq < b.seq;
    return a.depth > b.depth;
}

// State LRU (D7): the older logical clock of the state's own last use, then the older use.
bool older_state(const PrefixNode &a, const PrefixNode &b) {
    if (a.state_clock != b.state_clock) return a.state_clock < b.state_clock;
    if (a.state_seq != b.state_seq) return a.state_seq < b.state_seq;
    return a.depth > b.depth;
}

bool state_victim(const PrefixNode &n) { return n.slot != PrefixNode::kNoSlot && n.lock == 0; }
// The sessions whose hit is `n` itself: lock() counts a hit on the node and on each ancestor below the root (an edge
// split copies the child's count to the new parent), so n.lock less its children's counts.
uint64_t own_locks(const PrefixNode &n) {
    uint64_t below = 0;
    for (const auto &c : n.children) below += c.second->lock;
    return n.lock - below;
}
// O112 (D15⁗′ eviction): an unlocked end state, the only state an end-state pair may evict.
bool end_victim(const PrefixNode &n) { return n.cls != 0 && state_victim(n); }
// A host entry the arena may drop: unlocked and childless (M210S-42: a host node's host children go before it).
bool host_victim(const PrefixNode &n) { return n.host && n.lock == 0 && n.children.empty(); }
// A device leaf: on the device, unlocked, and every child (if any) a host node.
bool leaf_victim(const PrefixNode &n) {
    if (n.host || n.lock != 0) return false;
    for (const auto &c : n.children) {
        if (!c.second->host) return false;
    }
    return true;
}
// O112: a device leaf holding an end state, the first page victim.
bool end_leaf_victim(const PrefixNode &n) { return n.cls != 0 && n.slot != PrefixNode::kNoSlot && leaf_victim(n); }

// State order (D7, O112): end states first (except one touched at `keep`, the capturing ticket's clock: rank by kind),
// then (c) boundary states, then the rest; LRU within a rank.
int state_rank(const PrefixNode &n, uint64_t keep) {
    if (n.cls != 0 && (keep == 0 || n.state_clock != keep)) return 0;
    return n.kind == CaptureKind::kBoundary ? 1 : 2;
}

// The lowest i with slots i .. i + count - 1 free.
std::optional<size_t> first_free_run(const std::vector<uint8_t> &used, uint32_t count) {
    size_t run = 0;
    for (size_t i = 0; i < used.size(); ++i) {
        run = used[i] == 0 ? run + 1 : 0;
        if (run == count) return i + 1 - count;
    }
    return std::nullopt;
}

// O112: an unlocked end-state leaf, which an insert over other pages drops instead of being refused by it (M210S-39).
// D15⁗″: or an unlocked end state whose subtree holds only unlocked end states and state-less nodes (edge splits, or
// an evicted state's node): resumed calls' captures below it, the same conversation's earlier prompt ends.
bool end_only(const PrefixNode &n) {
    if (n.lock != 0 || n.host || (n.cls == 0 && n.slot != PrefixNode::kNoSlot)) return false;
    for (const auto &c : n.children) {
        if (!end_only(*c.second)) return false;
    }
    return true;
}
bool end_yields(const PrefixNode &n) { return n.cls != 0 && end_only(n); }

template <typename F>
void walk(PrefixNode &node, F &f) {
    for (auto &c : node.children) {
        f(*c.second);
        walk(*c.second, f);
    }
}

// The edge `matched` tokens of `child` that equal tokens[depth, ...) in whole pages, at most up to `limit`; the first
// page is equal by the children key.
uint64_t edge_match(const PrefixNode &child, const uint32_t *tokens, uint64_t depth, uint64_t limit) {
    uint64_t matched = kKvPageTokens;
    while (matched < child.key.size() && depth + matched + kKvPageTokens <= limit &&
           std::equal(child.key.data() + matched, child.key.data() + matched + kKvPageTokens,
                      tokens + depth + matched)) {
        matched += kKvPageTokens;
    }
    return matched;
}

// D17: the child of `node` at tokens[depth, ...) on the path of the session with `pages`: the device edge whose first
// page is the session's (its own edge, or one of the path it hit); children.end() when there is none.
template <typename Node>
auto own_child(Node &node, const uint32_t *tokens, uint64_t depth, const uint32_t *pages)
    -> decltype(node.children.begin()) {
    const auto range = node.children.equal_range(page_key(tokens + depth));
    for (auto it = range.first; it != range.second; ++it) {
        if (!it->second->host && it->second->pages.front() == pages[depth / kKvPageTokens]) return it;
    }
    return node.children.end();
}

}  // namespace

PrefixCache::PrefixCache(Device &device, DeviceModel &model, const ModelPlan &plan, ArenaAllocator *arena,
                         const std::array<uint8_t, 32> &layout_id)
    : device_(device),
      model_(model),
      config_(plan.config),
      arena_(plan.snapshot_host_bytes > 0 ? arena : nullptr),
      // Sessions (O211): a model with drafts (allocate refuses one its device cannot run) spills every edge with its
      // MTP rows, so a reloaded draft state is a drafting hit and a reloaded parent keeps its children's MTP rows.
      spill_kinds_(plan.speculative_draft_tokens != 0 ? snapshot::kRequiredKinds | 1u << snapshot::kDraft
                                                       : snapshot::kRequiredKinds),
      layout_id_(layout_id),
      slot_used_(plan.prefix_state_slots + plan.max_sessions, 0) {}

template <typename F>
void PrefixCache::each(F &&f) const {
    for (const auto &root : roots_) walk(*root.second, f);
}

PrefixNode *PrefixCache::lru(bool (*eligible)(const PrefixNode &), bool state_order, uint64_t keep) const {
    PrefixNode *best = nullptr;
    each([&](PrefixNode &n) {
        if (!eligible(n)) return;
        if (best != nullptr && state_order && state_rank(n, keep) != state_rank(*best, keep)) {
            if (state_rank(n, keep) < state_rank(*best, keep)) best = &n;
            return;
        }
        if (best == nullptr || (state_order ? older_state(n, *best) : older(n, *best))) best = &n;
    });
    return best;
}

std::vector<PrefixNode *> PrefixCache::path_states(const CacheNamespace &ns, const uint32_t *tokens, uint64_t len,
                                                   const uint32_t *pages) const {
    std::vector<PrefixNode *> out;
    const auto root = roots_.find(ns);
    if (root == roots_.end()) return out;
    const PrefixNode *node = root->second.get();
    uint64_t depth = 0;
    while (depth + kKvPageTokens <= len) {
        const auto it = pages != nullptr ? own_child(*node, tokens, depth, pages)
                                         : node->children.find(page_key(tokens + depth));
        if (it == node->children.end()) break;
        PrefixNode *child = it->second.get();
        if (child->host || child->depth > len || !std::equal(child->key.begin(), child->key.end(), tokens + depth)) {
            break;
        }
        if (child->slot != PrefixNode::kNoSlot) out.push_back(child);
        node = child;
        depth = child->depth;
    }
    return out;
}

std::vector<PrefixNode *> PrefixCache::own_states(const CacheNamespace &ns, const uint32_t *tokens, uint64_t len,
                                                  const uint32_t *pages) const {
    std::vector<PrefixNode *> out = path_states(ns, tokens, len, pages);
    out.erase(std::remove_if(out.begin(), out.end(), [](const PrefixNode *n) { return n->cls != 0; }), out.end());
    if (!out.empty()) out.pop_back();  // the deepest one stays
    out.erase(std::remove_if(out.begin(), out.end(), [](const PrefixNode *n) { return !state_victim(*n); }),
              out.end());
    return out;
}

std::vector<PrefixNode *> PrefixCache::superseded_ends(const CacheNamespace &ns, const uint32_t *tokens, uint64_t len,
                                                       const uint32_t *pages) const {
    std::vector<PrefixNode *> out = path_states(ns, tokens, len, pages);
    if (!out.empty()) out.pop_back();  // the deepest state (an end-state hit: the one the capturing call resumes) stays
    out.erase(std::remove_if(out.begin(), out.end(),
                             [](const PrefixNode *n) { return n->cls == 0 || own_locks(*n) != 0; }),
              out.end());
    return out;
}

namespace {

// D3 (D17: over every alternative edge, or with `pages` the session's own path) from `node` down; see match.
struct MatchWalk {
    const uint32_t *tokens;
    uint64_t n;
    uint64_t cap;
    bool device_only;
    uint64_t draft_chunk;
    const uint32_t *pages;
    const Device &device;
    const DeviceModel &model;
    PrefixCache::Match m;

    void from(const PrefixNode &node) {
        const uint64_t depth = node.depth;
        if (depth + kKvPageTokens > n) return;
        const auto range = node.children.equal_range(page_key(tokens + depth));
        for (auto it = range.first; it != range.second; ++it) {
            const PrefixNode &child = *it->second;
            if (pages != nullptr && (child.host || child.pages.front() != pages[depth / kKvPageTokens])) continue;
            const uint64_t matched = edge_match(child, tokens, depth, n);
            m.k = std::max(m.k, depth + matched);
            if (matched < child.key.size()) continue;
            const uint64_t at = child.depth;
            // D15⁗ draft states at chunk multiples; D15⁗′ end states off the grid, of a class whose row-bit signature
            // is the lookup's own class's (D15⁗″).
            bool draft_ok = child.cls == 0;
            if (draft_chunk != 0) {
                const uint64_t s = at / draft_chunk * draft_chunk;
                const bool next_ok = child.draft && at < n && tokens[at] == child.draft_next;
                draft_ok = child.cls == 0 ? next_ok && at == s
                                          : next_ok && at != s &&
                                                device.row_sig(model, s, child.cls) ==
                                                    device.row_sig(model, s, std::min(s + draft_chunk, n) - s);
            }
            if (child.stated() && at <= cap && (!device_only || child.slot != PrefixNode::kNoSlot) && draft_ok &&
                (m.node == nullptr || at > m.h || (at == m.h && older_state(*m.node, child)))) {
                m.node = it->second.get();
                m.h = at;
            }
            from(child);
        }
    }
};

}  // namespace

PrefixCache::Match PrefixCache::match(const CacheNamespace &ns, const uint32_t *tokens, uint64_t n, uint64_t cap,
                                      bool device_only, uint64_t draft_chunk, const uint32_t *pages) const {
    auto root = roots_.find(ns);
    if (root == roots_.end()) return Match{};
    MatchWalk walk{tokens, n, cap, device_only, draft_chunk, pages, device_, model_, Match{}};
    walk.from(*root->second);
    return walk.m;
}

void PrefixCache::lock(PrefixNode *node) {
    for (PrefixNode *p = node; p->parent != nullptr; p = p->parent) ++p->lock;
}

void PrefixCache::unlock(PrefixNode *node) {
    for (PrefixNode *p = node; p->parent != nullptr; p = p->parent) --p->lock;
}

void PrefixCache::touch(PrefixNode *node, uint64_t clock) {
    // max keeps a parent at least as recent as any child when an older submission's insert lands later (decode
    // inserts happen at the terminal poll).
    const uint64_t seq = ++seq_;
    if (clock > node->state_clock) {  // O247: a later use; a repeat touch at the same clock keeps the previous use
        node->prev_state_clock = node->state_clock;
        node->state_clock = clock;
    }
    node->state_seq = seq;
    for (PrefixNode *p = node; p->parent != nullptr; p = p->parent) {
        p->clock = std::max(p->clock, clock);
        p->seq = seq;
    }
}

std::vector<uint32_t> PrefixCache::path_pages(const PrefixNode *node) const {
    std::vector<uint32_t> out;
    if (node == nullptr) return out;
    out.resize(node->depth / kKvPageTokens);
    for (const PrefixNode *p = node; p->parent != nullptr; p = p->parent) {
        std::copy(p->pages.begin(), p->pages.end(), out.begin() + p->start() / kKvPageTokens);
    }
    return out;
}

std::vector<uint32_t> PrefixCache::path_tokens(const PrefixNode *node) const {
    std::vector<uint32_t> out(node->depth);
    for (const PrefixNode *p = node; p->parent != nullptr; p = p->parent) {
        std::copy(p->key.begin(), p->key.end(), out.begin() + p->start());
    }
    return out;
}

bool PrefixCache::take_slot(uint32_t *slot, uint64_t keep) {
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < slot_used_.size(); ++i) {
            if (slot_used_[i] != 0) continue;
            slot_used_[i] = 1;
            *slot = uint32_t(i);
            return true;
        }
        if (pass == 1) break;
        PrefixNode *victim = lru(state_victim, true, keep);
        if (victim == nullptr) return false;
        ++stats_.state_evictions;
        evict_state(victim);
    }
    return false;
}

namespace {

// O247 (S10 EndPairHorizon): appends to `out` the unlocked states in the subtree of `node` last used before `horizon`
// that are end states or have a device state other than an end state below them (each path's deepest such state
// stays, as the capturing session's own deepest does in own_states); whether that subtree, `node` included, holds a
// device state other than an end state.
bool collect_stale(PrefixNode &node, uint64_t horizon, std::vector<PrefixNode *> &out) {
    bool below = false;
    for (auto &c : node.children) below = collect_stale(*c.second, horizon, out) || below;
    if (node.parent != nullptr && (node.cls != 0 || below) && node.state_clock < horizon && state_victim(node)) {
        out.push_back(&node);
    }
    return below || (node.slot != PrefixNode::kNoSlot && node.cls == 0);
}

}  // namespace

std::vector<PrefixNode *> PrefixCache::stale_states(uint64_t horizon) const {
    std::vector<PrefixNode *> out;
    if (horizon == 0) return out;
    for (const auto &root : roots_) collect_stale(*root.second, horizon, out);
    return out;
}

bool PrefixCache::take_run(uint32_t *slot, uint32_t count, const CacheNamespace &ns, const uint32_t *tokens,
                           uint64_t len, const uint32_t *pages, uint64_t horizon) {
    // Nothing is evicted unless freeing every candidate (each superseded end state, each stale state, each end state,
    // the own states) leaves such a run.
    std::vector<uint8_t> used = slot_used_;
    const auto release = [&](const PrefixNode &n) {
        std::fill_n(used.begin() + n.slot, n.held_slots(), uint8_t{0});
    };
    each([&](PrefixNode &n) {
        if (end_victim(n)) release(n);
    });
    for (const PrefixNode *n : superseded_ends(ns, tokens, len, pages)) release(*n);
    for (const PrefixNode *n : stale_states(horizon)) release(*n);
    for (const PrefixNode *n : own_states(ns, tokens, len, pages)) release(*n);
    if (!first_free_run(used, count)) return false;
    for (;;) {
        if (const std::optional<size_t> i = first_free_run(slot_used_, count)) {
            std::fill_n(slot_used_.begin() + *i, count, uint8_t{1});
            *slot = uint32_t(*i);
            return true;
        }
        // ResumeExact first: the oldest superseded end state; then O247: the oldest stale state (each recollected, as
        // an eviction may remove or spill other nodes).
        PrefixNode *victim = nullptr;
        for (PrefixNode *n : superseded_ends(ns, tokens, len, pages)) {
            if (victim == nullptr || older_state(*n, *victim)) victim = n;
        }
        if (victim == nullptr) {
            for (PrefixNode *n : stale_states(horizon)) {
                if (victim == nullptr || older_state(*n, *victim)) victim = n;
            }
        }
        if (victim == nullptr) victim = lru(end_victim, true);
        if (victim == nullptr) {
            for (PrefixNode *n : own_states(ns, tokens, len, pages)) {
                if (victim == nullptr || older_state(*n, *victim)) victim = n;
            }
        }
        if (victim == nullptr) return false;
        ++stats_.state_evictions;
        evict_state(victim);
    }
}

void PrefixCache::free_slot(uint32_t slot) { slot_used_[slot] = 0; }

void PrefixCache::free_run(uint32_t slot, uint32_t count) { std::fill_n(slot_used_.begin() + slot, count, uint8_t{0}); }

void PrefixCache::free_state(PrefixNode *node) {
    if (node->slot != PrefixNode::kNoSlot) free_run(node->slot, node->held_slots());
    node->slot = PrefixNode::kNoSlot;
    node->draft = false;
    node->cls = 0;
    node->kv_slots = 0;
}

uint64_t PrefixCache::free_slots() const { return uint64_t(std::count(slot_used_.begin(), slot_used_.end(), 0)); }

const PrefixNode *PrefixCache::root(const CacheNamespace &ns) const {
    const auto it = roots_.find(ns);
    return it == roots_.end() ? nullptr : it->second.get();
}

// D7: a state evicted from a leaf deletes the leaf (D9: spills it to the host tier when that is on); an internal
// node keeps its pages and loses only the state. M210S-42/43: with a host tier, an internal node first spills its
// unlocked stated device children that are leaves themselves (a prompt tail's decode leaf), then, once every child
// is host-resident, spills like a leaf and keeps its host children. The LRU order is unchanged, so a tail goes
// after its older boundary states.
void PrefixCache::evict_state(PrefixNode *node) {
    if (arena_ != nullptr && !node->children.empty() && node->cls == 0) {
        std::vector<PrefixNode *> kids;
        for (auto &c : node->children) {
            PrefixNode *k = c.second.get();
            if (!k->host && k->slot != PrefixNode::kNoSlot && leaf_victim(*k)) kids.push_back(k);
        }
        for (PrefixNode *k : kids) {
            if (!spill(k)) break;
        }
    }
    if (leaf_victim(*node) && spill(node)) return;
    if (node->children.empty()) {
        remove(node);
        return;
    }
    free_state(node);
}

void PrefixCache::evict_leaf(PrefixNode *node) {
    ++stats_.leaf_evictions;
    if (node->slot != PrefixNode::kNoSlot && spill(node)) return;
    drop_host_children(node);
    remove(node);
}

// D9: the stated device leaf's own edge [start, depth) and state as a CERESNP1 image (first_row = the edge's start,
// kind 0 = the path's tokens; on a model with drafts kind 10 = the edge's MTP rows) in the arena, dropping LRU host
// entries while it does not fit. False (nothing changed) without a host tier, when it cannot fit or when the device
// cannot write it.
bool PrefixCache::spill(PrefixNode *node) {
    if (arena_ == nullptr || node->cls != 0) return false;  // an end state has no image (D15⁗′)
    namespace snap = snapshot;
    snap::Plan plan;
    if (!snap::plan(config_, node->depth, node->start(), node->depth, &plan, spill_kinds_)) return false;
    snap::Header header;
    header.layout_id = layout_id_;
    header.position = node->depth;
    header.token_count = node->depth;
    header.fed = node->depth;
    header.first_row = node->start();
    header.total_bytes = plan.total_bytes;
    if (plan.total_bytes > UINT64_MAX - 255) return false;
    const uint64_t bytes = (plan.total_bytes + 255) / 256 * 256;  // 256-B aligned in the arena, as snapshots
    // H3: dropping host entries is destructive, so none is dropped for an image no amount of dropping can fit.
    if (!could_fit_after_drops(bytes)) return false;
    uint64_t offset = 0;
    while (!arena_->allocate(bytes, &offset)) {
        if (!drop_host_lru()) return false;
    }
    uint8_t *dst = model_.arena + offset;
    snap::write_header(dst, header, plan);
    const std::vector<uint32_t> tokens = path_tokens(node);
    uint8_t *kind0 = dst + plan.sections[0].offset;
    for (uint64_t i = 0; i < tokens.size(); ++i) {
        for (int b = 0; b < 4; ++b) kind0[4 * i + b] = uint8_t(tokens[i] >> (8 * b));
    }
    const std::vector<uint32_t> pages = path_pages(node);
    if (device_.save_prefix(model_, pages.data(), pages.size(), node->slot, plan, node->depth, dst) != CERE_OK) {
        arena_->free(offset);
        return false;
    }
    release_pages(node->pages.data(), node->pages.size());
    node->pages.clear();
    // Sessions (O211): the image carries the edge's MTP rows (kind 10) on a model with drafts, so a draft state
    // stays one: the reload writes those rows back and a drafting session hits it as it would on the device.
    const bool draft = node->draft;
    const uint32_t draft_next = node->draft_next;
    free_state(node);
    node->draft = draft;
    node->draft_next = draft_next;
    node->host = true;
    node->host_offset = offset;
    ++stats_.host_spills;
    return true;
}

// A host child cannot outlive its parent's pages or image (D9: loading back needs the parent path on the device);
// drops the whole host subtree below `node`.
void PrefixCache::drop_host_children(PrefixNode *node) {
    while (!node->children.empty()) {
        PrefixNode *child = node->children.begin()->second.get();
        drop_host_children(child);
        ++stats_.host_drops;
        erase(child);
    }
}

void PrefixCache::drop_end_chain(PrefixNode *node) {
    while (!node->children.empty()) drop_end_chain(node->children.begin()->second.get());
    if (node->cls != 0) ++stats_.state_evictions;
    erase(node);
}

// Releases the node's pages, slot and arena bytes and deletes it (it has no children).
void PrefixCache::erase(PrefixNode *node) {
    if (!node->pages.empty()) release_pages(node->pages.data(), node->pages.size());
    free_state(node);
    if (node->host) arena_->free(node->host_offset);
    auto &siblings = node->parent->children;
    const auto range = siblings.equal_range(page_key(node->key.data()));
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second.get() == node) {
            siblings.erase(it);
            return;
        }
    }
}

// Deletes the node, then every state-less childless unlocked ancestor (D7), then an empty namespace root.
void PrefixCache::remove(PrefixNode *node) {
    PrefixNode *parent = node->parent;
    erase(node);
    while (parent->parent != nullptr && !parent->stated() && parent->children.empty() && parent->lock == 0) {
        PrefixNode *up = parent->parent;
        erase(parent);
        parent = up;
    }
    if (parent->parent != nullptr || !parent->children.empty()) return;
    for (auto it = roots_.begin(); it != roots_.end(); ++it) {
        if (it->second.get() == parent) {
            roots_.erase(it);
            return;
        }
    }
}

bool PrefixCache::own_path(const CacheNamespace &ns, const uint32_t *tokens, uint64_t len,
                           const uint32_t *pages) const {
    if (alt_edges_) return true;  // D17: the insert adds the session's own edges beside other ones
    const auto root = roots_.find(ns);
    if (root == roots_.end()) return true;
    const PrefixNode *node = root->second.get();
    uint64_t depth = 0;
    while (depth < len) {
        const auto it = node->children.find(page_key(tokens + depth));
        if (it == node->children.end()) return true;  // the rest is new edge with the inserter's pages
        const PrefixNode &child = *it->second;
        if (child.host) return true;  // replaced by (or dropped for) the inserter's pages
        const uint64_t matched = edge_match(child, tokens, depth, len);
        if (!std::equal(child.pages.begin(), child.pages.begin() + matched / kKvPageTokens,
                        pages + depth / kKvPageTokens)) {
            return end_yields(child);  // O112: dropped for the inserter's pages
        }
        if (matched < child.key.size()) return true;  // a split there keeps the matched (equal) pages
        depth = child.depth;
        node = &child;
    }
    return true;
}

bool PrefixCache::insert(const CacheNamespace &ns, const uint32_t *tokens, uint64_t len, const uint32_t *pages,
                         uint32_t slot, CaptureKind kind, uint64_t clock, std::optional<uint32_t> draft_next,
                         uint64_t cls, uint32_t kv_slots) {
    if (!own_path(ns, tokens, len, pages)) return false;  // M210S-39: never a state over foreign KV rows
    std::unique_ptr<PrefixNode> &root = roots_[ns];
    if (root == nullptr) root = std::make_unique<PrefixNode>();
    PrefixNode *node = root.get();
    uint64_t depth = 0;
    while (depth < len) {
        auto it = alt_edges_ ? own_child(*node, tokens, depth, pages) : node->children.find(page_key(tokens + depth));
        if (alt_edges_ && it == node->children.end()) {
            // D17: no edge over the session's pages here. An unlocked end-state leaf (D15⁗″: or chain) whose whole key
            // the session's tokens repeat yields (O112: the same conversation's next prompt supersedes it), counted as
            // state evictions; otherwise the rest becomes an alternative edge beside the other sessions' ones.
            const auto range = node->children.equal_range(page_key(tokens + depth));
            const auto yields = std::find_if(range.first, range.second, [&](const auto &entry) {
                const PrefixNode &child = *entry.second;
                return !child.host && end_yields(child) && child.depth <= len &&
                       edge_match(child, tokens, depth, len) == child.key.size();
            });
            if (yields != range.second) {
                drop_end_chain(yields->second.get());
                continue;
            }
            if (range.first != range.second) ++stats_.alt_edges;
        }
        if (it == node->children.end()) {
            auto child = std::make_unique<PrefixNode>();
            child->parent = node;
            child->key.assign(tokens + depth, tokens + len);
            child->pages.assign(pages + depth / kKvPageTokens, pages + len / kKvPageTokens);
            child->depth = len;
            retain_pages(child->pages.data(), child->pages.size());
            PrefixNode *raw = child.get();
            node->children.emplace(page_key(tokens + depth), std::move(child));
            node = raw;
            break;
        }
        PrefixNode *child = it->second.get();
        const uint64_t matched = edge_match(*child, tokens, depth, len);
        if (child->host) {
            if (matched == child->key.size() && child->depth == len) {
                // The key ends on the host node: it takes the inserter's pages and state instead of its image.
                ++stats_.host_drops;
                arena_->free(child->host_offset);
                child->host = false;
                child->pages.assign(pages + child->start() / kKvPageTokens, pages + len / kKvPageTokens);
                retain_pages(child->pages.data(), child->pages.size());
                node = child;
                break;
            }
            // Inserting below or beside a host node drops it with its host subtree, and the next pass adds a
            // fresh edge.
            ++stats_.host_drops;
            drop_host_children(child);
            erase(child);
            continue;
        }
        if (!alt_edges_ && end_yields(*child) &&
            !std::equal(child->pages.begin(), child->pages.begin() + matched / kKvPageTokens,
                        pages + depth / kKvPageTokens)) {
            // O112: an unlocked end-state leaf (D15⁗″: or chain) over other pages than the inserter's yields (with end
            // states off no node would be there); counted as state evictions, and the next pass adds a fresh edge.
            drop_end_chain(child);
            continue;
        }
        // D17: an own edge is the session's as far as its pages are (in practice all of the matched ones).
        uint64_t own = matched;
        if (alt_edges_) {
            const auto first = child->pages.begin();
            own = uint64_t(std::mismatch(first, first + matched / kKvPageTokens, pages + depth / kKvPageTokens).first -
                           first) *
                  kKvPageTokens;
        }
        if (own < child->key.size()) {
            // Split: the new parent takes the matched pages and no state (F4); the child keeps the rest.
            auto parent = std::make_unique<PrefixNode>();
            parent->parent = node;
            parent->key.assign(child->key.begin(), child->key.begin() + own);
            parent->pages.assign(child->pages.begin(), child->pages.begin() + own / kKvPageTokens);
            parent->depth = depth + own;
            parent->lock = child->lock;
            parent->clock = child->clock;
            parent->seq = child->seq;
            child->key.erase(child->key.begin(), child->key.begin() + own);
            child->pages.erase(child->pages.begin(), child->pages.begin() + own / kKvPageTokens);
            child->parent = parent.get();
            parent->children.emplace(page_key(child->key.data()), std::move(it->second));
            it->second = std::move(parent);
            child = it->second.get();
        }
        node = child;
        depth = child->depth;
    }
    touch(node, clock);
    if (node->slot != PrefixNode::kNoSlot) return false;
    node->slot = slot;
    node->kind = kind;
    node->draft = draft_next.has_value();
    node->draft_next = draft_next.value_or(0);
    node->cls = cls;
    node->kv_slots = cls != 0 ? kv_slots : 0;
    ++stats_.inserts;
    return true;
}

bool PrefixCache::make_free(uint64_t need) {
    while (device_.free_pages(model_) < need) {
        PrefixNode *victim = lru(end_leaf_victim, false);
        if (victim == nullptr) victim = lru(leaf_victim, false);
        if (victim == nullptr) return false;
        evict_leaf(victim);
    }
    return true;
}

uint64_t PrefixCache::evictable_pages() const {
    uint64_t pages = 0;
    each([&](PrefixNode &n) {
        if (n.lock != 0) return;
        for (const uint32_t id : n.pages) pages += device_.page_refs(model_, id) == 1 ? 1 : 0;
    });
    return pages;
}

bool PrefixCache::admits(uint64_t want, uint64_t unbound) const {
    const uint64_t available = device_.free_pages(model_) + evictable_pages();
    return available >= unbound && available - unbound >= want;
}

bool PrefixCache::drop_host_lru() {
    PrefixNode *victim = lru(host_victim, false);
    if (victim == nullptr) return false;
    ++stats_.host_drops;
    remove(victim);
    return true;
}

bool PrefixCache::could_fit_after_drops(uint64_t bytes) const {
    if (arena_ == nullptr) return false;
    uint64_t droppable = 0;
    each([&](const PrefixNode &n) {
        if (n.host && n.lock == 0) droppable += arena_->range_bytes(n.host_offset);
    });
    return bytes <= arena_->capacity() - (arena_->used() - droppable);
}

void PrefixCache::retain_pages(const uint32_t *ids, uint64_t n) {
    if (const cere_status st = device_.retain_pages(model_, ids, n); st != CERE_OK) pool_fault("retain", st);
}

void PrefixCache::release_pages(const uint32_t *ids, uint64_t n) {
    if (const cere_status st = device_.release_pages(model_, ids, n); st != CERE_OK) pool_fault("release", st);
}

void PrefixCache::pool_fault(const char *op, cere_status st) {
    if (!pool_fault_) std::fprintf(stderr, "cere: page_pool_fault op=%s where=prefix_cache status=%d\n", op, int(st));
    pool_fault_ = true;
}

cere_status PrefixCache::reload(PrefixNode *node) {
    if (!node->host) return CERE_OK;
    // M210S-42: a host node's parent may be host too; its path loads first (the caller's lock covers it).
    if (node->parent->parent != nullptr) {
        if (const cere_status st = reload(node->parent); st != CERE_OK) return st;
    }
    namespace snap = snapshot;
    snap::Plan plan;
    if (!snap::plan(config_, node->depth, node->start(), node->depth, &plan, spill_kinds_)) return CERE_ERR_INTERNAL;
    uint32_t slot = PrefixNode::kNoSlot;
    if (!take_slot(&slot)) return CERE_ERR_CAPACITY;
    const uint64_t own = node->key.size() / kKvPageTokens;
    make_free(own);
    std::vector<uint32_t> pages = path_pages(node->parent);
    const size_t base = pages.size();
    if (device_.take_pages(model_, own, &pages) != CERE_OK) {
        free_slot(slot);
        return CERE_ERR_CAPACITY;
    }
    if (const cere_status st = device_.load_prefix(model_, pages.data(), pages.size(), slot, plan, node->depth,
                                                   model_.arena + node->host_offset);
        st != CERE_OK) {
        release_pages(pages.data() + base, own);
        free_slot(slot);
        return st;
    }
    node->pages.assign(pages.begin() + base, pages.end());  // the take's reference is the tree's
    arena_->free(node->host_offset);
    node->host = false;
    node->slot = slot;
    ++stats_.host_loads;
    return CERE_OK;
}

bool PrefixCache::locked() const {
    for (const auto &root : roots_) {
        for (const auto &c : root.second->children) {
            if (c.second->lock != 0) return true;
        }
    }
    return false;
}

void PrefixCache::flush() {
    each([&](PrefixNode &n) {
        if (!n.pages.empty()) release_pages(n.pages.data(), n.pages.size());
        free_state(&n);
        if (n.host) arena_->free(n.host_offset);
    });
    roots_.clear();
}

cere_status attach_prefix(cere_executor &executor, cere_session &session, uint64_t clock, PrefixAttach *out) {
    *out = PrefixAttach{};
    cere_model &model = *session.model;
    PrefixCache *cache = model.prefix.get();
    if (cache == nullptr || session.bound) return CERE_OK;
    Device &device = *executor.device;
    const uint64_t want = kv_pages(session.max_context_tokens);
    PrefixCache::Match hit;
    uint64_t draft_cap = 0;  // a drafting lookup's cap floor64(n − 1) (0: no drafting lookup)
    // Ledger CR8 (M2.9 x M2.10, D15⁗): a one-call-layout drafting session hits only a draft state (PrefixNode::draft)
    // at a multiple of the prefill chunk whose MTP row h - 1 took this prompt's token h, so its target state, KV and MTP
    // K/V over [0, h) are its cold run's and its calls after h keep the cache-off layout (it has no split points,
    // D15‴, api.cpp `plan_captures`). Its captures at chunk multiples inside its prompt are draft states
    // (`draft_layout`). A device without a known prefill chunk gives it no lookup and no draft state. GdFix: a
    // masked-prompt drafting session looks up as a non-drafting one does (any state), as its NO_SPECULATION twin would;
    // MTP rows its hit lacks cost accepted drafts only.
    const bool one_call = session.one_call_layout();
    const uint64_t draft_chunk = one_call ? device.prefill_chunk(*model.device) : 0;
    // M2.13-L C6: an image session looks up only by its image keys (a restored one has none and no prompt). WeightLab
    // (WL decision (e)): a session whose research plan captures prompt positions does not look up (it prefills from
    // 0, so every prompt row it captures is computed); it still binds its own pages and captures states.
    const bool research_skip = session.research != nullptr && session.research->plan->captures_prompt;
    if (session.namespaced && (!session.image_session || !session.keys.empty()) && session.fed == 0 &&
        session.prompt_tokens != 0 && (!one_call || draft_chunk != 0) && !research_skip) {
        session.draft_layout = one_call;
        const uint64_t n = session.prompt_tokens;
        const uint64_t cap = (n - 1) / kKvPageTokens * kKvPageTokens;  // >= 1 row computes the last logits
        if (one_call) draft_cap = cap;
        hit = cache->match(session.ns, key_data(session), n, cap, false, draft_chunk);
        if (hit.node != nullptr) {
            cache->lock(hit.node);
            if (hit.node->host && cache->reload(hit.node) != CERE_OK) {
                // No slot or pages for the load-back: the deepest device state instead.
                cache->unlock(hit.node);
                const uint64_t k = hit.k;
                hit = cache->match(session.ns, key_data(session), n, cap, true, draft_chunk);
                hit.k = k;
                if (hit.node != nullptr) cache->lock(hit.node);
            }
        }
    }
    // D3′: the locked path's pages first, then the session's own, evicting LRU unlocked leaves before CAPACITY.
    const std::vector<uint32_t> prefix = cache->path_pages(hit.node);
    cache->make_free(want - prefix.size());
    if (const cere_status st = device.bind_pages(*model.device, *session.device, prefix.data(), prefix.size());
        st != CERE_OK) {
        if (hit.node != nullptr) cache->unlock(hit.node);
        return st;
    }
    session.bound = true;
    model.unbound_pages -= want;
    out->k = hit.k;
    // PT0a (prefix-token design note): the deepest prefill-chunk multiple c <= cap that a drafting lookup's page
    // match passed (c < k: token c matched too, so a draft state there would have been hit) lies c − h past its hit
    // when no draft state was at c (evicted, or never captured as one).
    if (draft_cap != 0 && hit.k != 0) {
        const uint64_t c = std::min(hit.k - 1, draft_cap) / draft_chunk * draft_chunk;
        if (c > hit.h) {
            cache->stats().hit_shortfall_tokens += c - hit.h;
            ++cache->stats().hit_shortfall_lookups;
        }
    }
    if (hit.node == nullptr) return CERE_OK;
    if (const cere_status st = device.load_slot(*model.device, *session.device, hit.node->slot, hit.h);
        st != CERE_OK) {
        cache->unlock(hit.node);
        return st;
    }
    cache->touch(hit.node, clock);
    session.locked = hit.node;
    session.fed = hit.h;
    ++cache->stats().hits;
    session.end_resume = hit.node->cls != 0;  // D15⁗′: its first step resumes the end state
    out->h = hit.h;
    return CERE_OK;
}

void release_prefix(cere_session &session) {
    cere_model &model = *session.model;
    if (model.prefix == nullptr) return;
    if (session.locked != nullptr) model.prefix->unlock(session.locked);
    session.locked = nullptr;
    if (session.track_slot != PrefixNode::kNoSlot) model.prefix->free_slot(session.track_slot);
    session.track_slot = PrefixNode::kNoSlot;
    session.tracked = 0;
    if (!session.bound) model.unbound_pages -= kv_pages(session.max_context_tokens);
    session.bound = true;
}

}  // namespace cere::host
