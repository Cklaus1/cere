// M2.10 native prefix cache (m210 design note D1-D4′, D6′, D7, D9, D11, D12; ledger M3, M4, M5, CR3): one radix
// tree per model over whole 64-token KV pages, device-independent (it drives the model's Device through the M2.10
// methods of host/device.h) and toy-tested. Mutated only from the executor's serialized calls (D16). No CUDA include.
#pragma once

#include "cere_executor.h"
#include "host/device.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace cere::host {

class ArenaAllocator;

// The 16-byte isolation key of a session (D2): root children of the tree are keyed by it.
using CacheNamespace = std::array<uint8_t, 16>;

// Why a state was captured (D4): (a) a call end on the 64 grid, (a′) floor64(n − 1), (c) W 132's boundary, (d) the
// branching point, (b) a decode track capture, or a D12 import. State eviction takes end states (PrefixNode::cls,
// O112) first, then (c) states (D7).
enum class CaptureKind : uint8_t { kCallEnd, kTail, kBoundary, kBranch, kDecode, kImport };

// Where a capture found no obtainable slot (prefix-token design note PT0a, F12): a prefill call-end or split-point
// capture (`prefill_slot`), a session's decode track capture (`decode_track`, counted once per ticket however many
// 64-token boundaries retry it) or an end-state run (`end_pair`, O112's take_run: the pair plus any chunk-0 K/V slots).
enum class SkipSite : uint8_t { kPrefillSlot, kDecodeTrack, kEndPair };

// Counters for tests and ktests (K16): `capture_skipped` is ktest `prefix_capture_skipped` (a prefill or decode
// capture that found no state slot, D4′), the sum of its three sites (`skip`); `capture_page_mismatch` counts states
// not stored because the tree's path to them holds other page ids than the capturing session's (M210S-39; never with
// alternative edges, m210 D17); `alt_edges` counts the edges an insert added beside an edge over the same first page
// of tokens and other pages. `hit_shortfall_tokens` (PT0a) sums, over drafting lookups whose page match k passed a
// prefill-chunk multiple c = floor_C(min(k − 1, floor64(n − 1))) deeper than their hit h, the tokens c − h lost to a
// draft state missing at c (evicted, or never captured as one); `hit_shortfall_lookups` counts those lookups.
struct PrefixStats {
    uint64_t hits = 0;                  // lookups that loaded a state (D3)
    uint64_t captures = 0;              // states copied into a pool or track slot
    uint64_t capture_skipped = 0;       // captures without an obtainable slot (the sum of the three sites below)
    uint64_t skipped_prefill_slot = 0;  // ... a prefill call-end or split-point capture
    uint64_t skipped_decode_track = 0;  // ... a decode track capture (once per ticket)
    uint64_t skipped_end_pair = 0;      // ... an end-state pair
    uint64_t inserts = 0;               // states the tree took
    uint64_t state_evictions = 0;       // states evicted for a slot (D7)
    uint64_t leaf_evictions = 0;        // leaves evicted for pages (D7)
    uint64_t host_spills = 0;           // evicted stated leaves written to the host tier (D9)
    uint64_t host_loads = 0;            // host entries loaded back (D9)
    uint64_t host_drops = 0;            // host entries dropped (arena priority, LRU, a parent leaving the device)
    uint64_t capture_page_mismatch = 0;  // captures/imports refused: the tree's path pages are not the session's
    uint64_t alt_edges = 0;              // D17: alternative edges added (a session's own pages beside another's)
    uint64_t hit_shortfall_tokens = 0;   // PT0a: drafting lookup tokens lost to a missing draft state
    uint64_t hit_shortfall_lookups = 0;  // PT0a: drafting lookups with hit_shortfall_tokens > 0

    // One capture skipped at `site`: the site's counter and the total.
    void skip(SkipSite site) {
        ++capture_skipped;
        switch (site) {
        case SkipSite::kPrefillSlot: ++skipped_prefill_slot; break;
        case SkipSite::kDecodeTrack: ++skipped_decode_track; break;
        case SkipSite::kEndPair: ++skipped_end_pair; break;
        }
    }
};

// The first page of an edge: a node's children are keyed by it.
using PageKey = std::array<uint32_t, 64>;

// One node: the edge from its parent is `key` (whole pages), held on the device by `pages` (one tree reference each)
// or, for a host node (D9), by a CERESNP1 own-edge image in the model's arena (first_row = the edge's start; on a
// model with drafts it carries the edge's MTP rows too, kind 10). A node holds at most one state: a pool slot, or the
// host image. Host nodes are always leaves.
struct PrefixNode {
    static constexpr uint32_t kNoSlot = UINT32_MAX;
    PrefixNode *parent = nullptr;  // null for a namespace root
    std::vector<uint32_t> key;     // the edge's tokens, a positive multiple of 64 (empty for a root)
    std::vector<uint32_t> pages;   // the edge's page ids (key.size() / 64; empty for a host node or a root)
    uint64_t depth = 0;            // tokens from the namespace root to the end of the edge
    // One child per first page of tokens; with alternative edges on (m210 D17) several children may share one, each
    // over other pages (another session's KV rows of the same tokens).
    std::multimap<PageKey, std::unique_ptr<PrefixNode>> children;
    uint32_t slot = kNoSlot;  // the state's pool slot
    bool host = false;        // the state and edge live in the arena at host_offset
    uint64_t host_offset = 0;
    CaptureKind kind = CaptureKind::kCallEnd;
    uint64_t lock = 0;   // sessions whose hit path runs through this node (D7: never evicted while > 0)
    uint64_t clock = 0;  // the logical clock (submission id) of the last use of the node's path (leaf LRU, D7)
    uint64_t seq = 0;    // tie-break of equal clocks: the tree's use counter
    // The last use of the node's own state (its insert, a hit loading it, an export): state LRU (D7). A path touch
    // does not refresh the states above the node, so the states of one ticket's call ends age in capture order.
    uint64_t state_clock = 0;
    uint64_t state_seq = 0;
    // O247 (S10 EndPairHorizon): state_clock before the touch that last advanced it, the state's previous use. A hit
    // reads it as its session's reuse horizon (PrefixCache::take_run).
    uint64_t prev_state_clock = 0;
    // D15⁗ (ledger CR8): the state is a draft state, so a drafting session may hit it. A drafting session that kept
    // the cache-off call layout captured it inside its prompt at a multiple of the prefill chunk, so the path's pages
    // carry its MTP K/V rows, and MTP row depth - 1 was computed with `draft_next` (the token at `depth`) as its next
    // id. Set with the slot; a state of any other session or an import is never one. A host-tier node keeps the mark
    // (Sessions, O211): its image carries the edge's MTP rows and the reload writes them back.
    bool draft = false;
    uint32_t draft_next = 0;
    // m210 D15⁗′ stage 2a: an end state when > 0. A drafting session whose cache-off call [s, e) (s = floor_C(b), the
    // prefill chunk C; e = min(s + C, n)) holds b = floor64(n − 1) strictly inside captured it at depth b during
    // that call (host/device.h EndStep) into adjacent pool slots, `slot` and `slot + 1` (its FP32 SSM halves), as a
    // draft state; `cls` = e − s is its class key κ. Only a drafting lookup of an equal own class hits it, and the
    // hit resumes it in its first call. Never spilled, reloaded or exported (no image holds it).
    uint64_t cls = 0;
    // PrefixCapture G1: a chunk-0 end state (s = 0) also holds the BF16 K/V rows [0, b) of its attention layers in
    // the `kv_slots` slots after the pair (Device::end_kv_slots), so it holds slots [slot, slot + 2 + kv_slots).
    uint32_t kv_slots = 0;
    // Pool slots the node's device state holds (0 without one).
    uint32_t held_slots() const { return slot == kNoSlot ? 0 : cls != 0 ? 2 + kv_slots : 1; }
    bool stated() const { return slot != kNoSlot || host; }
    uint64_t start() const { return depth - key.size(); }
};

class PrefixCache {
public:
    // The tree of `plan`'s model: a state pool of plan.prefix_state_slots + plan.max_sessions slots (D6′: capture and
    // track slots draw from one pool); the host tier (D9) is on when the plan has a snapshot arena (`arena` over
    // model.arena, ledger M3), whose images carry `layout_id`. Destroying the tree makes no device call (it dies
    // with its model, D2).
    PrefixCache(Device &device, DeviceModel &model, const ModelPlan &plan, ArenaAllocator *arena,
                const std::array<uint8_t, 32> &layout_id);
    PrefixCache(const PrefixCache &) = delete;
    PrefixCache &operator=(const PrefixCache &) = delete;

    struct Match {
        PrefixNode *node = nullptr;  // the deepest stated node within the cap (null: miss)
        uint64_t h = 0;              // its depth (0 on a miss)
        uint64_t k = 0;              // tokens of whole pages of the query that match the tree (D4 (d))
    };
    // D3: walks the whole pages of tokens[0, n) under `ns`. A stated node counts when its depth is <= `cap`
    // (floor64(n − 1) for a lookup) and, with `device_only`, its state is in a slot. With `draft_chunk` > 0 (a
    // drafting session's lookup, D15⁗) it counts only when it also holds a draft state at a depth that is a multiple
    // of draft_chunk (the prefill chunk: a cold call boundary) whose draft_next is tokens[depth]. An end state
    // (PrefixNode::cls, D15⁗′) counts only for a drafting lookup, at a depth b off that grid with draft_next ==
    // tokens[b] and a class κ = cls whose row-bit signature equals the lookup's own class κ′ = min(s + C, n) − s
    // (s = floor_C(b), C = draft_chunk): Device::row_sig(model, s, κ) == row_sig(model, s, κ′) (D15⁗″; the default
    // device's σ = class); a signature mismatch is a miss there. D17: every alternative edge is walked; the deepest
    // counting state wins, the one whose state was used last among equally deep ones, and `k` is the deepest page
    // match over all of them. With `pages` (a session's page ids, at least n / 64) only the session's own path is
    // walked: device edges whose pages are those ids.
    Match match(const CacheNamespace &ns, const uint32_t *tokens, uint64_t n, uint64_t cap, bool device_only,
                uint64_t draft_chunk = 0, const uint32_t *pages = nullptr) const;
    // +1 / -1 lock on `node` and every ancestor below the root.
    void lock(PrefixNode *node);
    void unlock(PrefixNode *node);
    // Marks `node` and its ancestors used at `clock`, and the node's own state.
    void touch(PrefixNode *node, uint64_t clock);
    // The page ids of rows [0, node->depth) along the path of a device node (empty for null or a root).
    std::vector<uint32_t> path_pages(const PrefixNode *node) const;

    // A state slot: the lowest free one, else the slot of the LRU unlocked state, end states first, then (c) states
    // (D7, O112); an end state touched at `keep` (the capturing ticket's clock, 0: none) ranks by its kind, so a
    // ticket's later capture (its prompt-end decode capture) does not take the end state it just captured ahead of
    // older states. False when every slot is a session's track slot or holds a state on a locked path.
    bool take_slot(uint32_t *slot, uint64_t keep = 0);
    // D15⁗′, O112: `count` (>= 2) adjacent free slots from `*slot` (the lowest such run): an end state's pair and its
    // chunk-0 K/V slots (PrefixNode::kv_slots). Else it evicts, first (m210 D15⁗″ ResumeExact) the superseded end
    // states on the capturing session's path tokens[0, len) of namespace `ns` (`superseded_ends`: the earlier prompt
    // ends above the state it resumes, which its hit's path lock alone keeps), oldest state first; then O247 (S10
    // EndPairHorizon), with `horizon` > 0 (the capturing session's hit node's prev_state_clock: its previous use of
    // the state it hit; O259 ShortCache: a hit-less prompt below the prefill chunk passes its own ticket's clock), the
    // stale states (`stale_states`: unlocked, last used before `horizon`, so idle longer than
    // the session's own reuse interval, any session's, each path's deepest state other than an end state excepted),
    // oldest state first; then LRU unlocked end states, then the capturing session's own LRU unlocked states on its
    // path below its deepest one (`own_states`; with `pages`, the session's page ids, its own path among alternative
    // edges, D17), until such a run is free, never another state; nothing is evicted and false is returned when
    // freeing all of them leaves no such run. `horizon` 0 (no hit, an end past chunk 0) and no superseded end state is
    // O112's order.
    bool take_run(uint32_t *slot, uint32_t count, const CacheNamespace &ns, const uint32_t *tokens, uint64_t len,
                  const uint32_t *pages = nullptr, uint64_t horizon = 0);
    void free_slot(uint32_t slot);
    // Frees `count` slots from `slot` (a run take_run gave that no node took).
    void free_run(uint32_t slot, uint32_t count);
    // Whether a state over tokens[0, len) with the inserting session's `pages` may be attached (M210S-39): every
    // device node already on the path holds, for the rows it covers within [0, len), exactly those page ids (a host
    // node on the way is replaced by the inserter's pages, and an unlocked end-state leaf over other pages is dropped
    // for them (O112), so neither conflicts). A state attached to another session's KV rows would not be reproducible
    // by any cold run. Always true with alternative edges on (D17): the insert puts the session's own pages beside
    // the other ones instead.
    bool own_path(const CacheNamespace &ns, const uint32_t *tokens, uint64_t len, const uint32_t *pages) const;
    // Inserts the state in `slot` at tokens[0, len) (len a positive multiple of 64) with the inserting session's
    // `pages` (page i holds rows [64 i, 64 i + 64), at least len / 64 of them): existing nodes are shared, a partly
    // matching edge is split (the split parent gets no state, F4), a host node on the way is dropped (or, when the
    // key ends there, takes the session's pages and `slot`), an unlocked end-state leaf over other pages is dropped
    // (O112, a state eviction), and new edges retain their pages. `draft_next` set marks the state a draft state
    // (D15⁗) with that next id; `cls` > 0 marks it an end state (PrefixNode::cls) holding the run of 2 + `kv_slots`
    // slots from `slot`. True when the tree took `slot` (and its run); false (the caller keeps or frees them) when the
    // node already held a device state or, with the tree unchanged, when !own_path (callers check own_path first to
    // count it and spare a slot). D17 (alternative edges on): the insert follows the edges over the session's own
    // pages (an edge whose first page is the session's is its own as far as their pages agree, split there); where
    // none is, an unlocked end-state leaf whose whole key the session's tokens repeat yields (O112), and the rest
    // becomes a new edge with the session's pages beside the edges over other pages (host nodes included), which stay.
    bool insert(const CacheNamespace &ns, const uint32_t *tokens, uint64_t len, const uint32_t *pages, uint32_t slot,
                CaptureKind kind, uint64_t clock, std::optional<uint32_t> draft_next = std::nullopt,
                uint64_t cls = 0, uint32_t kv_slots = 0);
    // D7's evict-before-CAPACITY hook: evicts LRU unlocked leaves (a stated one spills to the host tier when it is on),
    // end-state leaves first (O112), until `need` pages are free; whether they are.
    bool make_free(uint64_t need);
    // D3′ admission: free + evictable − `unbound` >= `want` (evictable = pages only the tree holds, in unlocked
    // nodes).
    bool admits(uint64_t want, uint64_t unbound) const;
    uint64_t evictable_pages() const;
    // D9 arena priority (ledger CR11): drops the LRU unlocked host entry; false when none is left.
    bool drop_host_lru();
    // H3: whether `bytes` of arena could fit once every droppable (unlocked) host entry were dropped (fragmentation
    // aside); callers check it before drop_host_lru so a request that can never fit destroys nothing.
    bool could_fit_after_drops(uint64_t bytes) const;
    // D9 load-back of a host node (locked by the caller, so eviction keeps its path): its own-edge pages from the
    // pool (evicting first), its state into a pool slot; the node is then a device node. CERE_ERR_CAPACITY (node
    // unchanged) when no slot or no pages are obtainable.
    cere_status reload(PrefixNode *node);
    // Whether a session holds a lock (D11).
    bool locked() const;
    // D11: drops every device and host entry (session track slots stay).
    void flush();
    // m210 D17: alternative edges on or off (cere_model_set_prefix_alt_edges, before the model's first ticket).
    void set_alt_edges(bool on) { alt_edges_ = on; }
    bool alt_edges() const { return alt_edges_; }
    // Latched when the device refused a page retain or release (see retain_pages).
    bool pool_faulted() const { return pool_fault_; }

    const PrefixStats &stats() const { return stats_; }
    PrefixStats &stats() { return stats_; }
    // The namespace root of `ns`, null when the namespace holds no node (tests and ktests walk the tree from it).
    const PrefixNode *root(const CacheNamespace &ns) const;
    uint64_t slots() const { return slot_used_.size(); }
    uint64_t free_slots() const;

private:
    template <typename F>
    void each(F &&f) const;
    // The LRU eligible node: by path use (leaves), or with `state_order` by state use with end states (but one
    // touched at `keep`), then (c) states first (D7, O112).
    PrefixNode *lru(bool (*eligible)(const PrefixNode &), bool state_order, uint64_t keep = 0) const;
    // The device states on the device path tokens[0, len) of `ns` (with `pages`, the session's own path, D17),
    // shallowest first.
    std::vector<PrefixNode *> path_states(const CacheNamespace &ns, const uint32_t *tokens, uint64_t len,
                                          const uint32_t *pages) const;
    // O112: the unlocked device states other than end states on that path, without the deepest such state on it
    // (locked or not): the ones an end-state pair of that path may take.
    std::vector<PrefixNode *> own_states(const CacheNamespace &ns, const uint32_t *tokens, uint64_t len,
                                         const uint32_t *pages) const;
    // m210 D15⁗″ (ResumeExact): the end states on that path other than its deepest state (a resumed call's hit) that
    // no session hit itself (locked only by hits below them): a conversation's earlier prompt ends, which a hit at
    // its previous prompt's end keeps locked turn after turn.
    std::vector<PrefixNode *> superseded_ends(const CacheNamespace &ns, const uint32_t *tokens, uint64_t len,
                                              const uint32_t *pages) const;
    // O247: the unlocked device states last used before `horizon` (none for 0) that are end states or have a device
    // state other than an end state deeper on their path: the ones an end-state pair takes first.
    std::vector<PrefixNode *> stale_states(uint64_t horizon) const;
    void evict_leaf(PrefixNode *node);
    void evict_state(PrefixNode *node);
    // Frees the node's state slot (both of an end state's) and clears its state marks (slot, draft, cls).
    void free_state(PrefixNode *node);
    bool spill(PrefixNode *node);
    void drop_host_children(PrefixNode *node);
    // O112 / D15⁗″: erases a yielding end-state subtree (prefix_cache.cpp end_yields) deepest first.
    void drop_end_chain(PrefixNode *node);
    void erase(PrefixNode *node);
    void remove(PrefixNode *node);
    // The device refused a page reference change (a double release, an unallocated page): the tree's accounting and
    // the pool disagree, so the flag below latches (one stderr line) and the host answers CERE_ERR_INTERNAL to the
    // model's later calls (DM-3).
    void retain_pages(const uint32_t *ids, uint64_t n);
    void release_pages(const uint32_t *ids, uint64_t n);
    void pool_fault(const char *op, cere_status st);
    std::vector<uint32_t> path_tokens(const PrefixNode *node) const;

    Device &device_;
    DeviceModel &model_;
    cere_qwen4_exp_config config_;
    ArenaAllocator *arena_;  // null: no host tier
    uint32_t spill_kinds_;   // the kinds of a host-tier image: 0-8, plus 10 (MTP rows) on a model with drafts
    std::array<uint8_t, 32> layout_id_;
    std::map<CacheNamespace, std::unique_ptr<PrefixNode>> roots_;
    std::vector<uint8_t> slot_used_;
    uint64_t seq_ = 0;
    PrefixStats stats_;
    bool alt_edges_ = false;  // D17
    bool pool_fault_ = false;
};

}  // namespace cere::host
