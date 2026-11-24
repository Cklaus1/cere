"""K16 (tasks/m210-spec.md R1, R3′, ledger R): the native prefix cache on the real A4 model, GPU0, container only.

Suite `model_k16` (`native_dev.py test --suite model_k16`; `--suite model` runs it too). `K16Prefix` creates one
model as `test_model_k15_snapshot.py` does (`oracle.create_model`): the build's `max_context_tokens` 200,000, four
sessions over a 200,000-token paged KV pool, the 4 GiB `snapshot_host_bytes` arena (the D9 host tier lives in it)
and `prefix_state_slots` 16 (D6 default), so the fixed-state pool holds 16 + 4 slots (D6′). `K16Eviction` creates
its own model afterwards (two models never coexist on GPU0): context 16,384, two sessions over a 32,768-token pool,
`prefix_state_slots` 2 (the "2-slot pool"; with the two sessions' track slots the pool has 4 indices) and no arena
(an evicted leaf is deleted, D7). Prompts are synthetic and deterministic (the chat template around numbered records,
cut in token space; a different first record gives a prompt whose first page differs); token IDs stay in memory.

Every gate here is native-vs-native and bit-exact (D8a "cached = cold-split"); D8b is reported only. No K16 result
is SGLang parity or a G1 result. The tree's prefix cache is flushed before every case (`cere_ktest_prefix_flush`,
D11) and the counters (`cere_ktest_prefix_stats`) are compared as deltas within a case.

The cold-split reference is an un-namespaced session (it never looks up or captures, D2) driven by
`cere_ktest_session_prefill_split` with the call ends a cache-on ticket of the same prompt makes (D4 (b′): its calls
end at h + 4,096 k and at each split point), then teacher-forced DECODE feeds of its greedy tokens.

Cases (R1 K16 (1)-(8), R3′ (2′)/(6″)):
* `test_checkpoint_equals_cold_split` (1), D5: for b in 4,032, 4,096, 4,160 and 9,024 (n 4,033, 5,000, 4,161, 9,025)
  a cache-on cold PREFILL ticket of tokens[0, n) on a namespaced session captures at b (4,032 and 4,160: (a′); 4,096:
  the (a) call end; 9,024: (a′) after two 4,096-row calls); a fresh session's prefix attach of tokens[0, b + 1) hits
  h = b and its state digest equals the cold-split session's digest after its call ending at b, bit-exact. b 64
  (n 200, M210S-39): the tree first holds a 192-token path that shares tokens[0, 100) and branches inside the second
  page, so the ticket's (d) capture at 64 and (a′) capture at 192 land on a path whose page 0 another ticket computed:
  both are refused and counted (`capture_page_mismatch`, not `capture_skipped`), nothing is stored, and a later ticket
  of tokens[0, 65) reports `cached_tokens` 0.
  Every b: the cold-split run continued through the ticket's later call ends scores the ticket's token bit-exactly.
* `test_hit_equals_cold_split` (2), (2′), (3), (8): for h in 64, 4,096, 8,192 and 9,024 the tree is seeded by a
  ticket of tokens[0, h + 1) ((a′) = h); then with n = h + 33 (no split after h): a hit ticket of 32 tokens reports
  `cached_tokens` = h and its tokens and `cere_token_score` rows equal the cold-split run [0, h)[h, n) + 31 DECODE
  rows; a prefix attach leaves the session at position h with its prompt open (the next call is PREFILL, (2′)) and
  its digest equals the cold-split digest at h ((8)); its PREFILL [h, n) and 32 traced DECODE steps (token, every
  score field, the `forward_trace`-contract taps) equal the cold-split run's. (3) D8b: the cold single-call run of
  tokens[0, n) and 32 greedy steps against the cold-split run: greedy agreement and the first score reported only.
* `test_namespaces_isolated` (4): a namespace-A ticket's checkpoint at 4,096 is invisible to namespace B (attach h 0,
  ticket `cached_tokens` 0) and to an un-namespaced ticket (no lookup, no capture), and hit by namespace A.
* `K16Eviction.test_eviction_two_slot_pool` (5): four checkpoints fill the pool; after a hit touches the oldest, a
  fifth capture evicts the LRU state (the second) and only it misses. With the four states on one path locked by a
  session's hit, a capture is skipped (`capture_skipped` + 1, nothing inserted, the ticket still OK); released, the
  same capture evicts the path's least recently used state (the shallowest: D7 orders states by their own use) and the
  path still hits at the deepest state while the evicted one misses.
* `test_host_tier_round_trip` (6), D9: a checkpoint at 4,096 is evicted to the host tier by 20 later captures (one
  per pool slot: `host_spills` + 1); a hit on it loads it back (`host_loads` + 1, h 4,096), its digest equals the
  cold-split digest at 4,096 and its PREFILL tail and 32 traced DECODE steps equal the cold-split run's.
* `test_state_copy` (6′), ledger M4: `copy_fixed_state` -> `load_fixed_state` of a session after a call ending at
  4,096 into a fresh session (`cere_ktest_prefix_state_copy`): digests equal; both finish the prompt and 64 traced
  DECODE steps are bit-identical.
* `test_snapshot_restore_of_hit` (6″): a session that hit at 4,096 and finished its prompt is snapshotted (M2.11) and
  restored into a fresh session: digests equal and 64 traced DECODE steps are bit-identical. The prompt's end leaves
  every token fed, so the hit first commits its predicted token unfed (`cere_ktest_session_commit_next`), the shape a
  ticket leaves; an image taken before that (token_count == fed) is refused (CERE_ERR_ARGUMENT, HardenHost H1) and
  the refused copy then restores the committed image.
* `test_repeat_determinism` (7): flush, seed ticket of tokens[0, 9,025), attach digest at 9,024 and a 32-token hit
  ticket, twice: scores, `cached_tokens` and digests are bit-identical across the repeats.
* `test_boundary_and_aligned_hits` (8): a ticket of tokens[0, 10,024) with W 132 `capture_position` 9,024 (D4 (c):
  9,024 - 0 >= 8,192, tail 1,000 <= 1,024); the next turn (tokens[0, 9,024) + a 600-token other tail) hits at 9,024
  (attach and ticket `cached_tokens`) with the cold-split digest at 9,024. An n % 64 == 0 prompt (4,096) repeated
  back to back: the repeat reports `cached_tokens` floor64(n - 1) = 4,032 and the same 8 tokens and scores as the
  first ticket (same call layout [0, 4,032)[4,032, 4,096)); the hit's digest equals the cold-split digest at 4,032.
* `K16Spec.test_drafting_hits` (9), m210 D15⁗ (ledger CR8): a third model (context 16,384, two sessions, a
  65,536-token pool, 8 state slots) with the MTP draft over the pinned artifact's draft token map (mounted by
  native_dev.py). Greedy 64-token tickets of the 10,024-token prompt: an un-namespaced NO_SPECULATION ticket is the
  plain cache-off reference; a cold drafting ticket (namespace A) reports `cached_tokens` 0, inserts its two draft
  states (its call ends 4,096 and 8,192) and its scores equal plain; a hot drafting ticket hits at 8,192 (`hits` + 1)
  and its scores equal plain and cold bit-exactly, and its verify_steps and accepted_drafts equal the cold ticket's
  (a wrong MTP draft state would keep the tokens and lose accepted drafts). A branch prompt (tokens[0, 8,192) + 1,200
  other tokens) falls back to the draft state at 4,096 and equals its plain and cold (namespace B) runs, accepted
  drafts included. A NO_SPECULATION ticket hits the drafting state at 8,192 and its first score equals the
  cold-split run [0, 4,096)[4,096, 8,192)[8,192, 9,984)[9,984, 10,024). NO_SPECULATION states (namespace C) are no
  drafting hit: the drafting ticket there reports `cached_tokens` 0 and equals plain. End states stay off on this
  model (the default, as on every shipped profile), so its counts are the stage-1 ones.
* `K16End.test_end_state_hits` (11), m210 D15⁗′ stage 2a: a drafting model with end states on
  (`cere_ktest_prefix_end_state`; context 73,728, two sessions, a 200,000-token pool, 8 state slots, the served draft
  map). For n 4,544, 18,368 and 72,000 (b = floor64(n - 1) inside
  the cold call [s, e), s >= 4,096; classes e - s 448, 1,984, 2,368): the cold drafting ticket inserts n // 4,096 draft
  states and the end state at b; the hot ticket hits at b and its scores, verify_steps and accepted_drafts equal the
  un-namespaced (cache-off) drafting run's, and a 1-token hot ticket's `cere_ktest_draft_digest` equals the cache-off
  one. At 4,544: a 5,196-token prompt (same cold call, class 1,100) misses the end state (cached_tokens 4,096) and
  equals its cache-off run (score and digest); the κ mutant (the end state's class forced to 1,100 through
  `cere_ktest_prefix_end_class`, so that prompt hits it at b) must differ from the cache-off digest.
* `K16SpecHost.test_host_tier_drafting_hits` (12), Sessions O211: K16 (9)'s drafting model with the 4 GiB arena (the
  D9 host tier) and 2 prefix slots. After a cold drafting ticket (namespace A) and its never-spilled hot hit at 8,192,
  drafting tickets of two other prompts (namespaces B and C) spill A's draft states (`host_spills` + >= 2); the next
  drafting ticket of the prompt in namespace A loads them back (`host_loads` + >= 1), reports `cached_tokens` 8,192,
  and its scores equal the plain run's and the device hit's bit-exactly, its verify_steps and accepted_drafts the
  device hit's (a host image without the edge's MTP rows would be no drafting hit, or lose accepted drafts).
* `K16End` (13), m210 D15⁗″ ResumeExact: on K16 (11)'s model, a drafting lookup hits the deepest end state on its
  path whose row-bit signature σ(s, κ) (`cere_ktest_prefix_row_sig`) equals its own class's, and a resumed call
  captures its own end state. `test_chain` (K16Chain: 17,000 -> 17,450 -> 18,000 -> 19,000, one cold call from
  16,384), `test_chunk0_chain` (1,200 -> 1,700 -> 2,040 -> 2,600: dense chunk-0 calls, then the K9 path) and
  `test_short_tail` (K16ShortTail: 18,374 -> 18,376, a resume of 8 rows): each turn of one conversation hits the
  previous turn's end state when their σ are equal (else the grid state: the draft state at s, none at s = 0) and
  its score and draft digest equal the cache-off run's. `test_row_sig` (K16RowSig, S5): for s 0 and 4,096 and every
  class at both ends of every row range in which no per-call choice changes (`cere_ktest_row_choice_bounds`), a cold
  ticket of s + κ captures its end state, the ticket of s + κ_ref (κ_ref: the largest tested class of the same σ)
  hits it and equals its cache-off run. `test_sigma_mutant` (S2): s 4,096 and the class pairs 218 / 478 and 96 / 160
  (other moe_shared_gate_up split ranges: σ differ) miss and equal cache-off; with the end state's class forced to
  the lookup's (`cere_ktest_prefix_end_class`) the resume differs from cache-off iff SIGMA_MUTANT_DIFFERS.
* Device memory (D6, D6′): the model's ALLOCATE (cudaMemGetInfo delta), the lowest GPU0 free value sampled after
  every case and the executor's peak (free before ALLOCATE minus that floor) are reported against D6′'s projection
  (≈ 81,028 MiB at 16 slots) and the 10,240 MiB free floor, recorded only.

Native entry points (the contract with the kernel track; `oracle.call` appends the stream; 0 = success, otherwise a
`cere_status`; "host" buffers are CPU memory; types of `cere_executor.h`). The model handle comes from
`cere_ktest_model_create/put/seal` (stated in `test_model_k10_long.py`, called by `oracle.create_model`); the
session entries `cere_ktest_session_open`, `cere_ktest_session_feed`, `cere_ktest_session_commit_next`,
`cere_ktest_session_snapshot`, `cere_ktest_session_restore`, `cere_ktest_session_close` and
`cere_ktest_state_digest` are stated in `test_model_k15_snapshot.py`. K16's own (tasks/m210-spec.md Interfaces):

    int cere_ktest_session_prefill_split(uint64_t session, const uint32_t *tokens /*host [n]*/, uint64_t n,
                                         const uint64_t *call_ends /*host [k]*/, uint64_t k,
                                         cere_token_score *scores /*host [1]*/, cudaStream_t stream);
        The cold-split reference: tokens[0, n) through Device::step in calls ending at call_ends (strictly
        increasing in (fed, n]); an unbound un-namespaced session is bound first; a session that consumed
        tokens[0, f) with its prompt open continues at f. The call ending at n scores the next token into scores[0]
        (uncommitted: teacher-forced DECODE feeds continue).
    int cere_ktest_prefix_attach(uint64_t session, const uint32_t *tokens /*host [n]*/, uint64_t n,
                                 uint64_t *out /*host [2]*/, cudaStream_t stream);
        The first prefill round's lookup, lock, page bind and load_slot on a fresh session without a step: the
        session has consumed tokens[0, h) with its prompt open; out = {h, k}.
    int cere_ktest_session_device_state(uint64_t session, uint64_t *out /*host [4]*/, cudaStream_t stream);
        out = {position, prompt open, pages bound, lazy}.
    int cere_ktest_prefix_state_copy(uint64_t source, uint64_t destination, cudaStream_t stream);
        capture_state of the source (consumed tokens[0, p) of an open prompt, p % 64 == 0) into a pool slot, then
        bind_pages (the source's [0, p / 64) shared) and load_slot into the fresh destination.
    int cere_ktest_prefix_stats(uint64_t model, uint64_t *out /*host [11]*/, cudaStream_t stream);
        out = {hits, captures, capture_skipped, inserts, state_evictions, leaf_evictions, host_spills, host_loads,
        host_drops, free KV pages, capture_page_mismatch}.
    int cere_ktest_session_set_namespace(uint64_t session, const uint8_t *namespace_id /*host [16]*/,
                                         cudaStream_t stream);
        cere_session_set_cache_namespace (before the first submit).
    int cere_ktest_prefix_flush(uint64_t model, cudaStream_t stream);
        cere_model_flush_prefix_cache.
    int cere_ktest_prefix_end_state(uint64_t model, uint32_t on, cudaStream_t stream);
        cere_model_set_prefix_end_state (m210 D15⁗′ stage 2a; before the model's first ticket, else CERE_ERR_BUSY).
    int cere_ktest_session_prefill_cached(uint64_t session, const uint32_t *tokens /*host [n]*/, uint64_t n,
                                          uint64_t max_new_tokens, uint64_t capture_position,
                                          cere_token_score *scores /*host [max_new_tokens]*/,
                                          uint64_t *out_counts /*host [3]*/, cudaStream_t stream);
        One greedy PREFILL work item with W 132 (0 = none) on an empty session: scores of the committed tokens,
        out_counts = {committed tokens, finish, cached_tokens}.
    int cere_ktest_session_prefill_cached_flags(uint64_t session, const uint32_t *tokens /*host [n]*/, uint64_t n,
                                                uint64_t max_new_tokens, uint64_t flags,
                                                cere_token_score *scores /*host [max_new_tokens]*/,
                                                uint64_t *out_counts /*host [6]*/, cudaStream_t stream);
        K16 (9): the same without W 132 and with `flags` as the work's cere_work.flags (0: a drafting PREFILL on a
        model with a draft; CERE_WORK_NO_SPECULATION): out_counts = {committed tokens, finish, cached_tokens,
        verify_steps, accepted_drafts, verified_drafts}.
    int cere_ktest_forward_layout_probe(uint64_t model, const uint32_t *tokens /*host [n]*/, uint64_t n,
                                        const uint64_t *ends /*host [ka + kb]*/, uint64_t ka, uint64_t kb,
                                        uint64_t w0, uint64_t replace, uint64_t arena_bytes,
                                        uint8_t *names /*host [names_capacity]*/, uint64_t names_capacity,
                                        uint64_t *digests /*host [2][entry_capacity][n - w0]*/,
                                        uint64_t entry_capacity, cere_token_score *scores /*host [2]*/,
                                        uint64_t *out_counts /*host [3]*/, cudaStream_t stream);
        K16 (10), m210 D15⁗′: tokens[0, n) through CudaDevice::forward on a fresh session in layout A (calls ending
        at ends[0, ka)), then on another in layout B (ends[ka, ka + kb)), both ending the same calls at or below w0;
        per I and P point occurrence of a call (forward order) the 64-bit row digests of the window rows [w0, n).
        With `replace` 1, B's P points are replaced by A's rows after their digest (each kernel then sees A's
        inputs). out_counts = {entries, name bytes, arena bytes a replace needs}.
    int cere_ktest_state_image(uint64_t session, uint64_t draft, uint8_t *dst /*host [capacity]*/, uint64_t capacity,
                               uint64_t *table /*host [3 * table_capacity]*/, uint64_t table_capacity,
                               uint64_t *out_counts /*host [2]*/, cudaStream_t stream);
        The bytes cere_ktest_state_digest (draft 0) or cere_ktest_draft_digest (draft 1) hashes; table = per section
        {kind, offset, bytes}; out_counts = {sections, image bytes} (also on CERE_ERR_CAPACITY).
    int cere_ktest_prefix_end_class(uint64_t model, const uint8_t *namespace_id /*host [16]*/,
                                    const uint32_t *tokens /*host [len]*/, uint64_t len, uint64_t cls,
                                    uint64_t *old /*host [1]*/, cudaStream_t stream);
        K16 (11)'s κ mutant: the end state at exactly tokens[0, len) under the namespace gets class `cls` in the tree
        (old = its class before; the device state is untouched); CERE_ERR_ARGUMENT when no end state is there.
    int cere_ktest_prefix_row_sig(uint64_t model, uint64_t s, uint64_t cls, uint64_t *out /*host [1]*/,
                                  cudaStream_t stream);
        K16 (13), m210 D15⁗″: out = the device's row-bit signature σ(s, cls) (CudaDevice::row_sig); a drafting lookup
        of class κ′ hits an end state of class κ in the same cold call iff σ(s, κ) == σ(s, κ′).
    int cere_ktest_row_choice_bounds(uint64_t cap, uint64_t *out /*host [capacity]*/, uint64_t capacity,
                                     uint64_t *count /*host [1]*/, cudaStream_t stream);
        K16 (13) K16RowSig: the ascending row counts in [65, cap] at which a per-call choice of a PREFILL call changes
        (K2 buckets, the pinned plans' range ends, P3b buckets, the dense K6 limit), plus 65; *count of them.
"""
from __future__ import annotations

import contextlib
import functools
import re
import unittest

import torch
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

import oracle

ROW = "K16"
MAX_CONTEXT = 200_000  # the build max (support.cpp)
SESSIONS = 4
POOL_TOKENS = 200_000  # the shipped profiles' kv_pool_tokens
ARENA_BYTES = 4 << 30  # the dev snapshot config's arena (native_dev.py memlock); D9's host tier lives in it
STATE_SLOTS = 16  # D6 default
STATE_SLOT_BYTES = 59_031_808  # D6′: one fixed state, padded
D6_PROJECTED_MIB = 81_028  # D6′ projection at 16 slots
FREE_FLOOR_MIB = 10_240  # ledger M1
PAGE_TOKENS = 64  # host/kv_pool.h kKvPageTokens
CASE_CONTEXT = 16_384  # every K16Prefix session
PROMPT_TOKENS = 10_024
HIT_STEPS = 32
COPY_STEPS = 64
HIT_TAIL = 33  # n = h + 33: floor64(n - 1) = h, so a hit ticket has no split after h
# (1) D5: (b, n, the cache-on ticket's call ends after the 4,096-row chunks).
CHECKPOINTS = ((64, 200, (64, 192, 200)), (4_032, 4_033, (4_032, 4_033)), (4_096, 5_000, (4_096, 4_992, 5_000)),
               (4_160, 4_161, (4_160, 4_161)), (9_024, 9_025, (9_024, 9_025)))
BRANCH_SHARED = 100  # (1) b 64: the branch prompt shares tokens[0, 100)
HITS = (64, 4_096, 8_192, 9_024)
BOUNDARY = 9_024
BOUNDARY_PROMPT = 10_024
BOUNDARY_TAIL = 600
ALIGNED_PROMPT = 4_096
ALIGNED_STEPS = 8
HOST_HIT = 4_096
EVICT_CONTEXT = 16_384
EVICT_SESSIONS = 2
EVICT_POOL_TOKENS = 32_768
EVICT_SLOTS = 2
EVICT_SESSION_CONTEXT = 1_024
NAMES_CAPACITY = 1 << 16
ENTRY_CAPACITY = 512
VALUES_CAPACITY = 4 << 20
# K16 (9), m210 D15⁗: the drafting model (the served draft map, as test_profile_forward.py reads it), its 4,096-row
# prefill chunk (min(context, kMaxChunkTokens)) and the 10,024-token prompt's draft states (its cold call ends).
DRAFT_MAP = "/draft_token_map.u32le"  # native_dev.py DRAFT_MAP_MOUNT
SPEC_CONTEXT = 16_384
SPEC_SESSIONS = 2
SPEC_POOL_TOKENS = 65_536
SPEC_SLOTS = 8
SPEC_PROMPT = 10_024
SPEC_STEPS = 64
SPEC_DRAFT_STATES = (4_096, 8_192)
SPEC_A_PRIME = 9_984  # floor64(SPEC_PROMPT - 1): a NO_SPECULATION ticket's (a′) split point
SPEC_BRANCH_TAIL = 1_200
SPEC_BRANCH_SKIP = 200  # the variant's leading chat-template rows are skipped, so its first token is a record's
# K16 (12), Sessions O211: the drafting model with the D9 host tier (the 4 GiB arena) and 2 prefix slots, so two other
# namespaces' draft states (SPEC_HOST_EVICTORS, below) spill namespace A's to the host tier.
SPEC_HOST_SLOTS = 2
CERE_WORK_NO_SPECULATION = 0x2
# K16 (10), m210 D15⁗′: the prefill call-layout probe. Layout A is the cache-off one (calls end at multiples of the
# device's 4,096-row prefill chunk, then at n); layout B also ends a call at h (a hot hit resuming at h).
LAYOUT_CONTEXT = 32_768
LAYOUT_SESSIONS = 2
LAYOUT_POOL_TOKENS = 65_536
LAYOUT_CHUNK = 4_096
LAYOUT_DEPTHS = (1_216, 4_544, 18_368)  # Scorecard G6R-1's 1.2K / 4.5K / 18K hot prompts
LAYOUT_CONTROL = 8_192  # a multiple of the chunk: A == B
# K16 (11), m210 D15⁗′ stage 2a: end states on a drafting model. The prompts' b = floor64(n − 1) lie strictly inside
# a cold call [s, e) with s >= 4,096 (classes 448, 1,984 and 2,368); END_MISS shares the 4.5K prompt's tokens[0, b + 1)
# in the same cold call with class 1,100 (another K2 pin range and P3b bucket). PrefixCapture G1: chunk-0 end states
# (s = 0: a dense cold call of n <= 2,048 rows that also saves its BF16 K/V rows [0, b)): G6's 1.2K hot shape (1,200:
# 48 resumed rows), a 12-row resume (1,100) and the dense limit (2,040: 56 rows). END_DENSE_OVER (a 2,100-row call,
# the K9 path) captures no end state, so its hot ticket misses.
END_CONTEXT = 73_728
END_SESSIONS = 2
END_POOL_TOKENS = 200_000
END_SLOTS = 8
END_CHUNK = 4_096
END_STEPS = 64
END_PROMPTS = (4_544, 18_368, 72_000, 1_200, 1_100, 2_040)
END_MISS = 5_196
END_DENSE_OVER = 2_100
# K16 (13), m210 D15⁗″ ResumeExact: one conversation per chain (namespace A, prompts extending each other, none a
# multiple of 64, so no prompt-end decode capture). CHAIN: one cold call from 16,384 (classes 616, 1,066, 1,616,
# 2,616: no split-K pin, one σ). CHUNK0_CHAIN: dense chunk-0 calls (classes 1,200, 1,700, 2,040), then 2,600 (the K9
# path: σ differs, no end-state hit). SHORT_TAIL: a resume of 8 rows (classes 1,990 -> 1,992). ROW_SIG_STARTS:
# (s, κ cap) of K16RowSig: chunk 0 up to the dense limit (no chunk-0 end state above it, PrefixCapture G1) and the
# second chunk up to C. SIGMA_MUTANTS: (κ, κ′) at s = 4,096 in different moe_shared_gate_up split ranges (193-256
# split 8 / 473-544 split 3; 65-96 split 9 / 129-160 split 6). Not 65-96 / 97-100 (split 9 / split 8): that resume
# equals cache-off (model_k16 2026-11-02T06:45Z), so σ is conservative there (a miss, never a wrong hit).
CHAIN = (17_000, 17_450, 18_000, 19_000)
CHUNK0_CHAIN = (1_200, 1_700, 2_040, 2_600)
SHORT_TAIL = (18_374, 18_376)
ROW_SIG_MIN = 65  # CudaDevice::kRowSigMinRows: an end state lies at b >= s + 64, so its class is at least 65
CHUNK0_DENSE = 2_048  # k6::kMaxContext: a chunk-0 call of at most this many rows is dense and keeps an end state
ROW_SIG_STARTS = ((0, CHUNK0_DENSE), (END_CHUNK, END_CHUNK))
ROW_SIG_BOUNDS = 256  # capacity of cere_ktest_row_choice_bounds' output
SIGMA_MUTANTS = ((218, 478), (96, 160))
LAYOUT_TAIL = 64  # n = h + 64
LAYOUT_ENTRIES = 4_096
LAYOUT_NAMES = 1 << 18
LAYOUT_HEADROOM = 16 << 30  # replace mode runs when A's recorded window rows leave this much device memory free
LAYOUT_TABLE = 16
LAYOUT_LAYERS = 48
LAYOUT_KV_WIDTH = 512  # FP8 bytes of one token's K (or V) in one attention layer (model.h kKvWidth)
CERE_LEVER_K2_SMALL_M = 1  # -v19's k2_gemm
ATTN_K, ATTN_V, QSA_KEYS, QSA_RING, GDN_CONV, GDN_SSM = range(1, 7)  # host/snapshot_format.h section kinds
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

CERE_FINISH_LENGTH = 1
CERE_FINISH_STOP_TOKEN = 2
CERE_ERR_ARGUMENT = -1  # cere_executor.h
# cere_ktest_prefix_stats fields
HITS_AT, CAPTURES, SKIPPED, INSERTS, STATE_EVICTIONS, LEAF_EVICTIONS, HOST_SPILLS, HOST_LOADS, HOST_DROPS, FREE, \
    PAGE_MISMATCH = range(11)

NS_A = bytes(range(16))
NS_B = bytes(range(16, 32))
NS_C = bytes(range(32, 48))
SPEC_HOST_EVICTORS = (NS_B, NS_C)  # K16 (12)

SENTINEL = "<<RECORDS>>"
QUESTION = "\n\nCopy every record above, one per line, in the same order, without any commentary."
COLORS = ("red", "green", "blue", "amber", "violet", "silver", "black")
ANIMALS = ("otter", "heron", "lynx", "badger", "falcon", "marten", "ibex", "crane", "viper", "bison", "gecko")


@functools.cache
def tokenizer():
    return get_tokenizer(str(oracle.MODEL_DIR))


@functools.cache
def synthetic_ids(total: int, first_record: int = 0) -> tuple[int, ...]:
    """Exactly `total` prompt IDs: the chat template around numbered records from `first_record` (cut in token
    space) and a request to copy them (as test_model_k15_snapshot.py)."""
    tok = tokenizer()
    text = tok.apply_chat_template([{"role": "user", "content": SENTINEL + QUESTION}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    if text.count(SENTINEL) != 1:
        raise oracle.OracleError("the rendered chat template does not hold the record sentinel exactly once")
    head, tail = text.split(SENTINEL)
    head_ids = list(tok.encode(head, add_special_tokens=False))
    tail_ids = list(tok.encode(tail, add_special_tokens=False))
    need = total - len(head_ids) - len(tail_ids)
    records = "\n".join(f"Record {i:05d}: the {COLORS[i % len(COLORS)]} {ANIMALS[i % len(ANIMALS)]} weighs "
                         f"{(i * 37) % 101 + 1} kg." for i in range(first_record, first_record + max(need // 8, 1)))
    body_ids = list(tok.encode(records, add_special_tokens=False))
    if need <= 0 or len(body_ids) < need:
        raise oracle.OracleError(f"synthetic prompt of {total} tokens: {len(body_ids)} record tokens for {need}")
    return tuple(head_ids + body_ids[:need] + tail_ids)


def variant(index: int, total: int = 400) -> list[int]:
    """A prompt whose first page differs from the main prompt's and from every other variant's."""
    return list(synthetic_ids(total, 20_000 + 500 * index))


# -- ktest entries -------------------------------------------------------------------------------------------------


def session_open(model: int, max_context_tokens: int) -> int:
    out = torch.zeros(1, dtype=torch.int64)
    oracle.call("cere_ktest_session_open", model, max_context_tokens, out)
    return int(out.item())


def close(handle: int) -> None:
    oracle.call("cere_ktest_session_close", handle)


def set_namespace(session: int, namespace: bytes) -> None:
    oracle.call("cere_ktest_session_set_namespace", session, torch.tensor(list(namespace), dtype=torch.uint8))


def flush(model: int) -> None:
    oracle.call("cere_ktest_prefix_flush", model)


def stats(model: int) -> list[int]:
    out = torch.zeros(11, dtype=torch.int64)
    oracle.call("cere_ktest_prefix_stats", model, out)
    return [int(v) for v in out.tolist()]


def row_sig(model: int, s: int, cls: int) -> int:
    """K16 (13): the device's row-bit signature σ(s, cls) of the model."""
    out = torch.zeros(1, dtype=torch.int64)
    oracle.call("cere_ktest_prefix_row_sig", model, s, cls, out)
    return int(out.item())


def row_choice_bounds(cap: int) -> list[int]:
    """K16 (13) K16RowSig: the row counts in [ROW_SIG_MIN, cap] at which a per-call choice changes, plus ROW_SIG_MIN."""
    out = torch.zeros(ROW_SIG_BOUNDS, dtype=torch.int64)
    count = torch.zeros(1, dtype=torch.int64)
    oracle.call("cere_ktest_row_choice_bounds", cap, out, ROW_SIG_BOUNDS, count)
    return [int(v) for v in out[:int(count.item())].tolist()]


def digest(session: int) -> torch.Tensor:
    out = torch.zeros(32, dtype=torch.uint8)
    oracle.call("cere_ktest_state_digest", session, out)
    return out


def cached_ticket(session: int, tokens: list[int], max_new: int, capture_position: int = 0
                  ) -> tuple[torch.Tensor, int, int]:
    """One greedy PREFILL ticket: (score rows of the committed tokens, finish, cached_tokens)."""
    scores = torch.full((max_new, 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_session_prefill_cached", session, torch.tensor(tokens, dtype=torch.int32), len(tokens),
                max_new, capture_position, scores, counts)
    committed, finish, cached = (int(v) for v in counts.tolist())
    return scores[:committed].clone(), finish, cached


class SpecRun:
    """A PREFILL ticket's committed score rows (int32 [committed, 4]), cached_tokens and speculation counters."""

    def __init__(self, scores: torch.Tensor, cached: int, verify_steps: int, accepted: int):
        self.scores = scores
        self.cached = cached
        self.verify_steps = verify_steps
        self.accepted = accepted

    def accept_length(self) -> float:
        """Committed tokens per verify step (tasks/m29-spec.md D13 ii); 0.0 without a verify step."""
        return round(len(self.scores) / self.verify_steps, 4) if self.verify_steps else 0.0


def spec_ticket(session: int, tokens: list[int], max_new: int, flags: int) -> SpecRun:
    """One greedy PREFILL ticket with `flags` (0: a drafting PREFILL, m210 D15⁗)."""
    scores = torch.full((max_new, 4), -1, dtype=torch.int32)
    counts = torch.zeros(6, dtype=torch.int64)
    oracle.call("cere_ktest_session_prefill_cached_flags", session, torch.tensor(tokens, dtype=torch.int32),
                len(tokens), max_new, flags, scores, counts)
    committed, _, cached, steps, accepted, _ = (int(v) for v in counts.tolist())
    return SpecRun(scores[:committed].clone(), cached, steps, accepted)


def prefill_split(session: int, tokens: list[int], call_ends: tuple[int, ...]) -> torch.Tensor:
    """The cold-split calls; the score row (int32 [4]) when the last end is len(tokens)."""
    score = torch.full((1, 4), -1, dtype=torch.int32)
    oracle.call("cere_ktest_session_prefill_split", session, torch.tensor(tokens, dtype=torch.int32), len(tokens),
                torch.tensor(call_ends, dtype=torch.int64), len(call_ends), score)
    return score[0].clone()


def attach(session: int, tokens: list[int]) -> tuple[int, int]:
    out = torch.zeros(2, dtype=torch.int64)
    oracle.call("cere_ktest_prefix_attach", session, torch.tensor(tokens, dtype=torch.int32), len(tokens), out)
    return int(out[0]), int(out[1])


def device_state(session: int) -> tuple[int, int, int, int]:
    out = torch.zeros(4, dtype=torch.int64)
    oracle.call("cere_ktest_session_device_state", session, out)
    position, prompt_open, pages, lazy = (int(v) for v in out.tolist())
    return position, prompt_open, pages, lazy


def state_copy(source: int, destination: int) -> None:
    oracle.call("cere_ktest_prefix_state_copy", source, destination)


def layout_ends(n: int, h: int) -> tuple[list[int], list[int], int]:
    """K16 (10): (A's call ends, B's, w0): A the cache-off layout, B also ending a call at h, w0 the last end both
    share (0: none)."""
    cold = list(range(LAYOUT_CHUNK, n, LAYOUT_CHUNK)) + [n]
    return cold, sorted(set(cold) | {h}), max([0] + [e for e in cold if e <= h])


def layout_probe(model: int, tokens: list[int], a: list[int], b: list[int], w0: int, replace: int = 0,
                 arena_bytes: int = 0) -> tuple[list[str], torch.Tensor, torch.Tensor, int]:
    """K16 (10): (point names, digests int64 [2, entries, n - w0], scores int32 [2, 4], arena bytes a replace needs)."""
    names = torch.zeros(LAYOUT_NAMES, dtype=torch.uint8)
    digests = torch.zeros((2, LAYOUT_ENTRIES, len(tokens) - w0), dtype=torch.int64)
    scores = torch.full((2, 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_forward_layout_probe", model, torch.tensor(tokens, dtype=torch.int32), len(tokens),
                torch.tensor(a + b, dtype=torch.int64), len(a), len(b), w0, replace, arena_bytes, names, LAYOUT_NAMES,
                digests, LAYOUT_ENTRIES, scores, counts)
    entries, name_bytes, need = (int(v) for v in counts.tolist())
    labels = bytes(names[:name_bytes].tolist()).decode().split("\0")[:-1]
    return labels, digests[:, :entries].clone(), scores, need


def state_image(session: int) -> tuple[list[tuple[int, int, int]], torch.Tensor]:
    """K16 (10): the session's state image (cere_ktest_state_digest's bytes) and its sections (kind, offset, bytes)."""
    table = torch.zeros(3 * LAYOUT_TABLE, dtype=torch.int64)
    counts = torch.zeros(2, dtype=torch.int64)
    oracle.call_status("cere_ktest_state_image", session, 0, None, 0, table, LAYOUT_TABLE, counts)  # sizes only
    image = torch.zeros(int(counts[1]), dtype=torch.uint8)
    oracle.call("cere_ktest_state_image", session, 0, image, image.numel(), table, LAYOUT_TABLE, counts)
    rows = table[:3 * int(counts[0])].view(-1, 3).tolist()
    return [(int(k), int(o), int(s)) for k, o, s in rows], image


def layer_list(layers) -> str:
    """Sorted layer numbers as ranges ("0-2,5"); "" for none."""
    out: list[list[int]] = []
    for x in sorted(set(layers)):
        if out and out[-1][1] == x - 1:
            out[-1][1] = x
        else:
            out.append([x, x])
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


def point_layers(names: list[str], mask: torch.Tensor) -> str:
    """The layers (48 = the final mixer) of the entries `mask` selects."""
    return layer_list(int(m.group(1)) if (m := LAYER_RE.search(name)) else LAYOUT_LAYERS
                      for name, flag in zip(names, mask.tolist()) if flag)



def feed(session: int, tokens: list[int], *, trace: bool = False) -> tuple[torch.Tensor, tuple | None]:
    """Teacher-forced DECODE feed (as test_model_k15_snapshot.py): score rows and, with `trace`, the taps."""
    ids = torch.tensor(tokens, dtype=torch.int32)
    scores = torch.full((len(tokens), 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    if not trace:
        oracle.call("cere_ktest_session_feed", session, ids, len(tokens), 0, 0, None, 0, None, 0, None, 0, scores,
                    counts)
        return scores, None
    names = torch.zeros(NAMES_CAPACITY, dtype=torch.uint8)
    shapes = torch.zeros((ENTRY_CAPACITY, 3), dtype=torch.int64)
    values = oracle.empty((VALUES_CAPACITY,), torch.uint8)
    oracle.call("cere_ktest_session_feed", session, ids, len(tokens), 0, 0, names, NAMES_CAPACITY, shapes,
                ENTRY_CAPACITY, values, VALUES_CAPACITY, scores, counts)
    entries, name_bytes, value_bytes = (int(v) for v in counts.tolist())
    if entries == 0:
        raise oracle.OracleError("a traced feed tapped no entry")
    return scores, (names[:name_bytes].clone(), shapes[:entries].clone(), values[:value_bytes].cpu())


def snapshot_keep(session: int) -> int:
    length = torch.zeros(1, dtype=torch.int64)
    handle = torch.zeros(1, dtype=torch.int64)
    oracle.call("cere_ktest_session_snapshot", session, None, 0, length, handle)
    return int(handle.item())


def restore(session: int, snapshot_handle: int) -> None:
    oracle.call("cere_ktest_session_restore", session, snapshot_handle, None, 0)


def restore_status(session: int, snapshot_handle: int) -> int:
    """cere_ktest_session_restore from the snapshot handle; the raw status."""
    return oracle.call_status("cere_ktest_session_restore", session, snapshot_handle, None, 0)


def commit_next(session: int, token: int) -> None:
    """The fully fed session's committed, unfed `token` (cere_ktest_session_commit_next): a ticket's shape."""
    oracle.call("cere_ktest_session_commit_next", session, token)


class Run:
    """Greedy teacher-forced DECODE rows: score rows and (traced) the taps of every step."""

    def __init__(self):
        self.scores: list[torch.Tensor] = []
        self.names: list[torch.Tensor] = []
        self.shapes: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []


def greedy(session: int, first: int, steps: int, *, trace: bool = True) -> Run:
    """`steps` greedy DECODE rows starting from the token `first`."""
    run = Run()
    token = first
    for _ in range(steps):
        scores, taps = feed(session, [token], trace=trace)
        run.scores.append(scores[0].clone())
        if taps is not None:
            run.names.append(taps[0])
            run.shapes.append(taps[1])
            run.values.append(taps[2])
        token = int(scores[0, 0].item())
    return run


def check_runs(tc: unittest.TestCase, case: str, run: Run, reference: Run) -> None:
    oracle.check_count(tc, ROW, f"{case} steps", len(run.scores), expected=len(reference.scores))
    oracle.check_bit_exact(tc, ROW, f"{case} scores", torch.stack(run.scores), torch.stack(reference.scores))
    oracle.check_bit_exact(tc, ROW, f"{case} tap names", torch.cat(run.names), torch.cat(reference.names))
    oracle.check_bit_exact(tc, ROW, f"{case} tap shapes", torch.cat(run.shapes), torch.cat(reference.shapes))
    oracle.check_bit_exact(tc, ROW, f"{case} tap values", torch.cat(run.values), torch.cat(reference.values))


def first_token(score: torch.Tensor) -> int:
    return int(score[0].item())


def logit(score: torch.Tensor) -> float:
    return float(score[2:3].view(torch.float32).item())


class Sessions(unittest.TestCase):
    """Session bookkeeping shared by the K16 classes: every handle is closed in tearDown, newest first."""

    model = 0
    cache = True  # False: a cache-off model, which has no tree to flush (cere_ktest_prefix_flush: UNSUPPORTED)

    def setUp(self):
        self.handles: list[int] = []
        if self.cache:
            flush(self.model)

    def tearDown(self):
        self.close_all()

    def close_all(self) -> None:
        while self.handles:
            close(self.handles.pop())

    @contextlib.contextmanager
    def case(self, **params):
        """A subTest whose sessions are closed when it ends, so a failed case leaves no hit lock or session."""
        with self.subTest(**params):
            try:
                yield
            finally:
                self.close_all()

    def open(self, context: int, namespace: bytes | None = None) -> int:
        session = session_open(self.model, context)
        self.handles.append(session)
        if namespace is not None:
            set_namespace(session, namespace)
        return session

    def drop(self, handle: int) -> None:
        self.handles.remove(handle)
        close(handle)

    def ticket(self, context: int, tokens: list[int], max_new: int = 1, capture_position: int = 0,
               namespace: bytes | None = NS_A) -> tuple[torch.Tensor, int, int]:
        """A cache-on (namespaced) ticket on a fresh session, released afterwards."""
        session = self.open(context, namespace)
        out = cached_ticket(session, tokens, max_new, capture_position)
        self.drop(session)
        return out

    def attached(self, context: int, tokens: list[int], namespace: bytes = NS_A) -> tuple[int, int, int]:
        """(session, h, k) of a fresh namespaced session's prefix attach."""
        session = self.open(context, namespace)
        h, k = attach(session, tokens)
        return session, h, k

    def hit_at(self, context: int, tokens: list[int], expected: int, case: str) -> int:
        """A fresh namespaced session's attach, asserted to hit at `expected`; the session."""
        session, h, _ = self.attached(context, tokens)
        oracle.check_count(self, ROW, f"{case} h", h, expected=expected)
        return session


class K16Prefix(Sessions):
    @classmethod
    def setUpClass(cls):
        cls.prompt = list(synthetic_ids(PROMPT_TOKENS))
        torch.cuda.synchronize(oracle.DEVICE)
        torch.cuda.empty_cache()
        cls.free_before, cls.total = torch.cuda.mem_get_info(oracle.DEVICE)
        cls.model = oracle.create_model(MAX_CONTEXT, max_sessions=SESSIONS, kv_pool_tokens=POOL_TOKENS,
                                        snapshot_host_bytes=ARENA_BYTES, prefix_state_slots=STATE_SLOTS)
        cls.free_after, _ = torch.cuda.mem_get_info(oracle.DEVICE)
        cls.free_floor = cls.free_after

    @classmethod
    def tearDownClass(cls):
        mib = 1 << 20
        oracle.report_row(ROW, "device memory (D6)", phase="diag", gpu0_total_mib=cls.total // mib,
                          free_before_allocate_mib=cls.free_before // mib,
                          allocate_taken_mib=(cls.free_before - cls.free_after) // mib,
                          executor_peak_mib=(cls.free_before - cls.free_floor) // mib,
                          gpu0_free_floor_mib=cls.free_floor // mib,
                          gpu0_used_peak_mib=(cls.total - cls.free_floor) // mib,
                          state_pool_bytes=STATE_SLOT_BYTES * (STATE_SLOTS + SESSIONS), state_slots=STATE_SLOTS,
                          sessions=SESSIONS, d6_projected_mib=D6_PROJECTED_MIB, d6_free_floor_mib=FREE_FLOOR_MIB)
        oracle.release_model(cls.model)

    def tearDown(self):
        super().tearDown()
        free, _ = torch.cuda.mem_get_info(oracle.DEVICE)
        type(self).free_floor = min(type(self).free_floor, free)

    def test_checkpoint_equals_cold_split(self):
        for b, n, ends in CHECKPOINTS:
            with self.case(b=b):
                flush(self.model)
                tokens = self.prompt[:n]
                if b == 64:
                    # (d): the tree holds a 192-token path sharing tokens[0, 100) and no state at 64.
                    branch = tokens[:BRANCH_SHARED] + variant(0)[BRANCH_SHARED:n]
                    self.assertTrue(branch[64:128] != tokens[64:128], f"{ROW} b64: the branch shares page 1")
                    self.ticket(CASE_CONTEXT, branch)
                before = stats(self.model)
                scores, finish, cached = self.ticket(CASE_CONTEXT, tokens)
                oracle.check_count(self, ROW, f"D5 b{b} cached_tokens", cached, expected=0)
                stopped = first_token(scores[0]) in oracle.stop_token_ids()
                oracle.check_count(self, ROW, f"D5 b{b} finish", finish,
                                   expected=CERE_FINISH_STOP_TOKEN if stopped else CERE_FINISH_LENGTH)
                after = stats(self.model)
                cold = self.open(CASE_CONTEXT)
                if b == 64:
                    # M210S-39: the (d) capture at 64 and the (a′) capture at 192 land on a path whose page 0 is
                    # the branch ticket's, not the pages this ticket computed: skipped, counted, never stored.
                    oracle.check_count(self, ROW, "D5 b64 (d) captures", after[CAPTURES] - before[CAPTURES],
                                       expected=0)
                    oracle.check_count(self, ROW, "D5 b64 (d) page mismatches",
                                       after[PAGE_MISMATCH] - before[PAGE_MISMATCH], expected=2)
                    oracle.check_count(self, ROW, "D5 b64 (d) slot skips", after[SKIPPED] - before[SKIPPED], expected=0)
                    _, _, later = self.ticket(CASE_CONTEXT, tokens[:b + 1])
                    oracle.check_count(self, ROW, "D5 b64 (d) later cached_tokens", later, expected=0)
                    prefill_split(cold, tokens, (b,))
                    # Diagnostic: does the call layout of rows [0, 64) change the state at 64?
                    halves = self.open(CASE_CONTEXT)
                    prefill_split(halves, tokens, (32, 64))
                    oracle.report_row(ROW, "D5 b64 layout sensitivity", phase="diag",
                                      split_32_64_equals_64=bool(torch.equal(digest(halves), digest(cold))))
                    self.drop(halves)
                else:
                    oracle.report_row(ROW, f"D5 b{b} captures", phase="diag",
                                      captures=after[CAPTURES] - before[CAPTURES],
                                      skipped=after[SKIPPED] - before[SKIPPED])
                    hit = self.hit_at(CASE_CONTEXT, tokens[:b + 1], b, f"D5 b{b}")
                    prefill_split(cold, tokens, (b,))
                    oracle.check_bit_exact(self, ROW, f"D5 b{b} digest", digest(hit), digest(cold))
                    self.drop(hit)
                # Splits never depend on what was stored: the ticket's score is the cold-split run's.
                score = prefill_split(cold, tokens, ends[1:])
                oracle.check_bit_exact(self, ROW, f"D5 b{b} ticket score", scores[0], score)
                self.drop(cold)

    def test_hit_equals_cold_split(self):
        for h in HITS:
            with self.case(h=h):
                flush(self.model)
                n = h + HIT_TAIL
                tokens = self.prompt[:n]
                self.ticket(CASE_CONTEXT, tokens[:h + 1])
                # The cold-split run: [0, h)[h, n) + 32 traced DECODE rows; its digest at h.
                cold = self.open(CASE_CONTEXT)
                prefill_split(cold, tokens, (h,))
                cold_digest = digest(cold)
                cold_score = prefill_split(cold, tokens, (n,))
                cold_run = greedy(cold, first_token(cold_score), HIT_STEPS)
                cold_rows = torch.stack([cold_score] + cold_run.scores[:HIT_STEPS - 1])
                # The ticket commits up to the first stop token (the prompt is cut mid-record); the teacher-forced
                # cold rows continue past it.
                stops = set(oracle.stop_token_ids())
                committed = next((i + 1 for i, row in enumerate(cold_rows) if first_token(row) in stops), HIT_STEPS)
                cold_rows = cold_rows[:committed]
                # (2) the hit ticket.
                scores, finish, cached = self.ticket(CASE_CONTEXT, tokens, HIT_STEPS)
                oracle.check_count(self, ROW, f"D8a h{h} cached_tokens", cached, expected=h)
                oracle.check_count(self, ROW, f"D8a h{h} finish", finish,
                                   expected=CERE_FINISH_LENGTH if committed == HIT_STEPS else CERE_FINISH_STOP_TOKEN)
                oracle.check_count(self, ROW, f"D8a h{h} committed", len(scores), expected=committed)
                oracle.check_bit_exact(self, ROW, f"D8a h{h} ticket tokens",
                                       torch.tensor(scores[:, 0].tolist(), dtype=torch.int32),
                                       torch.tensor(cold_rows[:, 0].tolist(), dtype=torch.int32))
                oracle.check_bit_exact(self, ROW, f"D8a h{h} ticket scores", scores, cold_rows)
                # (2′) the attach leaves the prompt open at h; (8) its digest; then PREFILL [h, n) and 32 steps.
                hit = self.hit_at(CASE_CONTEXT, tokens, h, f"D8a h{h}")
                position, prompt_open, pages, lazy = device_state(hit)
                oracle.check_count(self, ROW, f"(2′) h{h} position", position, expected=h)
                oracle.check_count(self, ROW, f"(2′) h{h} prompt open", prompt_open, expected=1)
                oracle.check_count(self, ROW, f"(2′) h{h} lazy", lazy, expected=1)
                oracle.check_count(self, ROW, f"(2′) h{h} pages", pages, expected=CASE_CONTEXT // PAGE_TOKENS)
                oracle.check_bit_exact(self, ROW, f"(8) h{h} digest", digest(hit), cold_digest)
                hit_score = prefill_split(hit, tokens, (n,))
                oracle.check_bit_exact(self, ROW, f"D8a h{h} PREFILL score", hit_score, cold_score)
                check_runs(self, f"D8a h{h}", greedy(hit, first_token(hit_score), HIT_STEPS), cold_run)
                self.drop(hit)
                # (3) D8b, reported only: the cold single call [0, n) against the cold-split run.
                single = self.open(CASE_CONTEXT)
                single_score = prefill_split(single, tokens, (n,))
                single_run = greedy(single, first_token(single_score), HIT_STEPS, trace=False)
                single_tokens = [first_token(single_score)] + [first_token(s) for s in single_run.scores]
                split_tokens = [first_token(cold_score)] + [first_token(s) for s in cold_run.scores]
                leading = next((i for i, (a, c) in enumerate(zip(single_tokens, split_tokens)) if a != c),
                               len(split_tokens))
                oracle.report_row(ROW, f"D8b h{h}", phase="diag", hit=h, steps=HIT_STEPS,
                                  leading_agreement=leading,
                                  equal_tokens=sum(a == c for a, c in zip(single_tokens, split_tokens)),
                                  first_score_bit_equal=bool(torch.equal(single_score, cold_score)),
                                  first_logit_delta=logit(single_score) - logit(cold_score))
                self.drop(single)
                self.drop(cold)

    def test_namespaces_isolated(self):
        n = HOST_HIT + HIT_TAIL
        tokens = self.prompt[:n]
        self.ticket(CASE_CONTEXT, tokens[:HOST_HIT + 1])
        other, h, k = self.attached(CASE_CONTEXT, tokens, NS_B)
        oracle.check_count(self, ROW, "namespace B attach h", h, expected=0)
        oracle.check_count(self, ROW, "namespace B attach k", k, expected=0)
        self.drop(other)
        _, _, cached = self.ticket(CASE_CONTEXT, tokens, namespace=NS_B)
        oracle.check_count(self, ROW, "namespace B cached_tokens", cached, expected=0)
        before = stats(self.model)
        _, finish, cached = self.ticket(CASE_CONTEXT, tokens, namespace=None)
        after = stats(self.model)
        oracle.check_count(self, ROW, "un-namespaced cached_tokens", cached, expected=0)
        oracle.check_count(self, ROW, "un-namespaced finish", finish, expected=CERE_FINISH_LENGTH)
        oracle.check_count(self, ROW, "un-namespaced hits", after[HITS_AT] - before[HITS_AT], expected=0)
        oracle.check_count(self, ROW, "un-namespaced captures", after[CAPTURES] - before[CAPTURES], expected=0)
        _, _, cached = self.ticket(CASE_CONTEXT, tokens)
        oracle.check_count(self, ROW, "namespace A cached_tokens", cached, expected=HOST_HIT)

    def test_host_tier_round_trip(self):
        n = HOST_HIT + HIT_TAIL
        tokens = variant(1, n + 64)[:n]
        self.ticket(CASE_CONTEXT, tokens[:HOST_HIT + 1])
        cold = self.open(CASE_CONTEXT)
        prefill_split(cold, tokens, (HOST_HIT,))
        cold_digest = digest(cold)
        cold_score = prefill_split(cold, tokens, (n,))
        cold_run = greedy(cold, first_token(cold_score), HIT_STEPS)
        self.drop(cold)
        # One capture per pool slot: the 4,096 state (the LRU one) is evicted by the last and spills (D9).
        before = stats(self.model)
        for i in range(STATE_SLOTS + SESSIONS):
            self.ticket(CASE_CONTEXT, variant(2 + i)[:PAGE_TOKENS + 1])
        after = stats(self.model)
        oracle.check_count(self, ROW, "host tier evictions", after[STATE_EVICTIONS] - before[STATE_EVICTIONS],
                           expected=1)
        oracle.check_count(self, ROW, "host tier spills", after[HOST_SPILLS] - before[HOST_SPILLS], expected=1)
        hit = self.hit_at(CASE_CONTEXT, tokens, HOST_HIT, "host tier")
        loaded = stats(self.model)
        oracle.check_count(self, ROW, "host tier loads", loaded[HOST_LOADS] - after[HOST_LOADS], expected=1)
        oracle.check_bit_exact(self, ROW, "host tier digest", digest(hit), cold_digest)
        hit_score = prefill_split(hit, tokens, (n,))
        oracle.check_bit_exact(self, ROW, "host tier PREFILL score", hit_score, cold_score)
        check_runs(self, "host tier", greedy(hit, first_token(hit_score), HIT_STEPS), cold_run)

    def test_state_copy(self):
        n = 4_161
        tokens = self.prompt[:n]
        source = self.open(CASE_CONTEXT)
        prefill_split(source, tokens, (HOST_HIT,))
        destination = self.open(CASE_CONTEXT)
        state_copy(source, destination)
        oracle.check_bit_exact(self, ROW, "(6′) digest", digest(destination), digest(source))
        position, prompt_open, _, _ = device_state(destination)
        oracle.check_count(self, ROW, "(6′) position", position, expected=HOST_HIT)
        oracle.check_count(self, ROW, "(6′) prompt open", prompt_open, expected=1)
        source_score = prefill_split(source, tokens, (n,))
        copy_score = prefill_split(destination, tokens, (n,))
        oracle.check_bit_exact(self, ROW, "(6′) PREFILL score", copy_score, source_score)
        source_run = greedy(source, first_token(source_score), COPY_STEPS)
        check_runs(self, "(6′)", greedy(destination, first_token(copy_score), COPY_STEPS), source_run)

    def test_snapshot_restore_of_hit(self):
        n = 4_161
        tokens = self.prompt[:n]
        self.ticket(CASE_CONTEXT, tokens[:HOST_HIT + 1])
        hit = self.hit_at(CASE_CONTEXT, tokens, HOST_HIT, "(6″)")
        score = prefill_split(hit, tokens, (n,))
        hit_digest = digest(hit)
        unfed = snapshot_keep(hit)  # every token fed: the image has token_count == fed
        self.handles.append(unfed)
        copy = self.open(CASE_CONTEXT)
        oracle.check_count(self, ROW, "(6″) restore of a token_count == fed image", restore_status(copy, unfed),
                           expected=CERE_ERR_ARGUMENT)
        commit_next(hit, first_token(score))
        handle = snapshot_keep(hit)
        self.handles.append(handle)
        restore(copy, handle)
        oracle.check_bit_exact(self, ROW, "(6″) digest", digest(copy), hit_digest)
        hit_run = greedy(hit, first_token(score), COPY_STEPS)
        check_runs(self, "(6″)", greedy(copy, first_token(score), COPY_STEPS), hit_run)

    def test_repeat_determinism(self):
        h = 9_024
        tokens = self.prompt[:h + HIT_TAIL]
        runs = []
        for _ in range(2):
            flush(self.model)
            seed, _, _ = self.ticket(CASE_CONTEXT, tokens[:h + 1])
            hit = self.hit_at(CASE_CONTEXT, tokens, h, "(7)")
            hit_digest = digest(hit)
            self.drop(hit)
            scores, _, cached = self.ticket(CASE_CONTEXT, tokens, HIT_STEPS)
            runs.append((seed, hit_digest, scores, cached))
        (seed0, digest0, scores0, cached0), (seed1, digest1, scores1, cached1) = runs
        oracle.check_bit_exact(self, ROW, "(7) seed scores", seed1, seed0)
        oracle.check_bit_exact(self, ROW, "(7) digest", digest1, digest0)
        oracle.check_bit_exact(self, ROW, "(7) hit scores", scores1, scores0)
        oracle.check_count(self, ROW, "(7) cached_tokens", cached1, expected=cached0)
        oracle.check_count(self, ROW, "(7) cached_tokens h", cached0, expected=h)

    def test_boundary_and_aligned_hits(self):
        # D4 (c): the message boundary at 9,024, tail 1,000 <= 1,024; the next turn branches at 9,024.
        self.ticket(CASE_CONTEXT, self.prompt[:BOUNDARY_PROMPT], capture_position=BOUNDARY)
        turn = self.prompt[:BOUNDARY] + variant(40, BOUNDARY_TAIL + 64)[:BOUNDARY_TAIL]
        self.assertTrue(turn[BOUNDARY:BOUNDARY + PAGE_TOKENS] != self.prompt[BOUNDARY:BOUNDARY + PAGE_TOKENS],
                        f"{ROW} boundary: the next turn shares the page after the boundary")
        hit = self.hit_at(CASE_CONTEXT, turn, BOUNDARY, "(8) boundary")
        cold = self.open(CASE_CONTEXT)
        prefill_split(cold, turn, (BOUNDARY,))
        oracle.check_bit_exact(self, ROW, "(8) boundary digest", digest(hit), digest(cold))
        self.drop(cold)
        self.drop(hit)
        _, _, cached = self.ticket(CASE_CONTEXT, turn)
        oracle.check_count(self, ROW, "(8) boundary cached_tokens", cached, expected=BOUNDARY)
        # n % 64 == 0, back to back: the repeat hits at floor64(n - 1).
        flush(self.model)
        tokens = self.prompt[:ALIGNED_PROMPT]
        aligned = ALIGNED_PROMPT - PAGE_TOKENS
        first, _, cached = self.ticket(CASE_CONTEXT, tokens, ALIGNED_STEPS)
        oracle.check_count(self, ROW, "(8) aligned first cached_tokens", cached, expected=0)
        repeat, _, cached = self.ticket(CASE_CONTEXT, tokens, ALIGNED_STEPS)
        oracle.check_count(self, ROW, "(8) aligned repeat cached_tokens", cached, expected=aligned)
        oracle.check_bit_exact(self, ROW, "(8) aligned repeat scores", repeat, first)
        hit = self.hit_at(CASE_CONTEXT, tokens, aligned, "(8) aligned")
        cold = self.open(CASE_CONTEXT)
        prefill_split(cold, tokens, (aligned,))
        oracle.check_bit_exact(self, ROW, "(8) aligned digest", digest(hit), digest(cold))


class K16Eviction(Sessions):
    @classmethod
    def setUpClass(cls):
        cls.model = oracle.create_model(EVICT_CONTEXT, max_sessions=EVICT_SESSIONS,
                                        kv_pool_tokens=EVICT_POOL_TOKENS, prefix_state_slots=EVICT_SLOTS)

    @classmethod
    def tearDownClass(cls):
        oracle.release_model(cls.model)

    def probe(self, tokens: list[int]) -> int:
        """h of a fresh namespaced session's attach (the session is released)."""
        session, h, _ = self.attached(EVICT_SESSION_CONTEXT, tokens)
        self.drop(session)
        return h

    def test_eviction_two_slot_pool(self):
        pool = EVICT_SLOTS + EVICT_SESSIONS  # track slots draw from the same pool (D6′)
        one = PAGE_TOKENS + 1
        prompts = [variant(50 + i)[:one] for i in range(pool + 1)]
        # LRU: fill the pool, touch the oldest, one more capture evicts the second.
        start = stats(self.model)
        for tokens in prompts[:pool]:
            self.ticket(EVICT_SESSION_CONTEXT, tokens)
        full = stats(self.model)
        oracle.check_count(self, ROW, "(5) captures to fill", full[CAPTURES] - start[CAPTURES], expected=pool)
        oracle.check_count(self, ROW, "(5) evictions to fill", full[STATE_EVICTIONS] - start[STATE_EVICTIONS],
                           expected=0)
        oracle.check_count(self, ROW, "(5) touch oldest", self.probe(prompts[0]), expected=PAGE_TOKENS)
        self.ticket(EVICT_SESSION_CONTEXT, prompts[pool])
        evicted = stats(self.model)
        oracle.check_count(self, ROW, "(5) evictions", evicted[STATE_EVICTIONS] - full[STATE_EVICTIONS], expected=1)
        oracle.check_count(self, ROW, "(5) LRU evicted", self.probe(prompts[1]), expected=0)
        for i in (0, *range(2, pool + 1)):
            oracle.check_count(self, ROW, f"(5) kept {i}", self.probe(prompts[i]), expected=PAGE_TOKENS)
        # All states locked: four states on one path, held by a hit; a capture is skipped.
        flush(self.model)
        path = variant(60)
        for depth in range(1, pool + 1):
            _, _, cached = self.ticket(EVICT_SESSION_CONTEXT, path[:depth * PAGE_TOKENS + 1])
            oracle.check_count(self, ROW, f"(5) path {depth} cached_tokens", cached,
                               expected=(depth - 1) * PAGE_TOKENS)
        holder = self.open(EVICT_SESSION_CONTEXT, NS_A)
        h, _ = attach(holder, path[:pool * PAGE_TOKENS + 44])
        oracle.check_count(self, ROW, "(5) lock holder h", h, expected=pool * PAGE_TOKENS)
        before = stats(self.model)
        other = variant(61)[:one]
        _, finish, _ = self.ticket(EVICT_SESSION_CONTEXT, other)
        after = stats(self.model)
        oracle.check_count(self, ROW, "(5) locked finish", finish, expected=CERE_FINISH_LENGTH)
        oracle.check_count(self, ROW, "(5) capture skipped", after[SKIPPED] - before[SKIPPED], expected=1)
        oracle.check_count(self, ROW, "(5) locked captures", after[CAPTURES] - before[CAPTURES], expected=0)
        oracle.check_count(self, ROW, "(5) locked evictions", after[STATE_EVICTIONS] - before[STATE_EVICTIONS],
                           expected=0)
        self.drop(holder)
        oracle.check_count(self, ROW, "(5) skipped not inserted", self.probe(other), expected=0)
        # Released: the same capture evicts the path's least recently used state (D7, by each state's own use): the
        # shallowest, since every deeper ticket used the state above it and the holder used the deepest last.
        released = stats(self.model)
        self.ticket(EVICT_SESSION_CONTEXT, other)
        again = stats(self.model)
        oracle.check_count(self, ROW, "(5) released captures", again[CAPTURES] - released[CAPTURES], expected=1)
        oracle.check_count(self, ROW, "(5) released evictions",
                           again[STATE_EVICTIONS] - released[STATE_EVICTIONS], expected=1)
        oracle.check_count(self, ROW, "(5) path after eviction", self.probe(path[:pool * PAGE_TOKENS + 44]),
                           expected=pool * PAGE_TOKENS)
        oracle.check_count(self, ROW, "(5) LRU state evicted", self.probe(path[:one]), expected=0)


class K16Spec(Sessions):
    """K16 (9), m210 D15⁗ (ledger CR8): drafting sessions on a model with the MTP draft (the served draft map)."""

    @classmethod
    def setUpClass(cls):
        with open(DRAFT_MAP, "rb") as f:
            draft_map = torch.frombuffer(bytearray(f.read()), dtype=torch.int32)
        cls.prompt = list(synthetic_ids(SPEC_PROMPT))
        cls.model = oracle.create_model(SPEC_CONTEXT, max_sessions=SPEC_SESSIONS, kv_pool_tokens=SPEC_POOL_TOKENS,
                                        prefix_state_slots=SPEC_SLOTS, draft_token_map=draft_map)

    @classmethod
    def tearDownClass(cls):
        oracle.release_model(cls.model)

    def request(self, tokens: list[int], namespace: bytes | None, *, drafting: bool) -> SpecRun:
        """One greedy PREFILL ticket on a fresh session (namespaced unless `namespace` is None), released after."""
        session = self.open(SPEC_CONTEXT, namespace)
        run = spec_ticket(session, tokens, SPEC_STEPS, 0 if drafting else CERE_WORK_NO_SPECULATION)
        self.drop(session)
        return run

    def same(self, case: str, run: SpecRun, reference: SpecRun) -> None:
        oracle.check_bit_exact(self, ROW, f"{case} scores", run.scores, reference.scores)

    def test_drafting_hits(self):
        prompt = self.prompt
        plain = self.request(prompt, None, drafting=False)  # cache off: no lookup, no capture, no draft
        before = stats(self.model)
        cold = self.request(prompt, NS_A, drafting=True)
        seeded = stats(self.model)
        oracle.check_count(self, ROW, "(9) cold cached_tokens", cold.cached, expected=0)
        oracle.check_count(self, ROW, "(9) cold draft states", seeded[INSERTS] - before[INSERTS],
                           expected=len(SPEC_DRAFT_STATES))
        oracle.check_count(self, ROW, "(9) cold drafts verified", int(cold.verify_steps > 0), expected=1)
        self.same("(9) cold == plain", cold, plain)
        hot = self.request(prompt, NS_A, drafting=True)
        oracle.check_count(self, ROW, "(9) hot cached_tokens", hot.cached, expected=SPEC_DRAFT_STATES[-1])
        oracle.check_count(self, ROW, "(9) hot hits", stats(self.model)[HITS_AT] - seeded[HITS_AT], expected=1)
        self.same("(9) hot == plain", hot, plain)
        self.same("(9) hot == cold", hot, cold)
        oracle.check_count(self, ROW, "(9) hot verify_steps", hot.verify_steps, expected=cold.verify_steps)
        oracle.check_count(self, ROW, "(9) hot accepted_drafts", hot.accepted, expected=cold.accepted)
        oracle.report_row(ROW, "(9) accept length", phase="diag", cold_verify_steps=cold.verify_steps,
                          cold_accepted_drafts=cold.accepted, hot_verify_steps=hot.verify_steps,
                          hot_accepted_drafts=hot.accepted, cold_accept_length=cold.accept_length(),
                          hot_accept_length=hot.accept_length(), hot_cached_tokens=hot.cached)

        # The token at 8,192 differs: the state there drafted over another next id, so the hit falls back to 4,096.
        branch = prompt[:SPEC_DRAFT_STATES[-1]] + variant(70, SPEC_BRANCH_TAIL + SPEC_BRANCH_SKIP)[SPEC_BRANCH_SKIP:]
        oracle.check_count(self, ROW, "(9) branch token differs",
                           int(branch[SPEC_DRAFT_STATES[-1]] != prompt[SPEC_DRAFT_STATES[-1]]), expected=1)
        branch_plain = self.request(branch, None, drafting=False)
        branch_cold = self.request(branch, NS_B, drafting=True)
        branch_hot = self.request(branch, NS_A, drafting=True)
        oracle.check_count(self, ROW, "(9) branch cached_tokens", branch_hot.cached, expected=SPEC_DRAFT_STATES[0])
        self.same("(9) branch hot == plain", branch_hot, branch_plain)
        self.same("(9) branch hot == cold", branch_hot, branch_cold)
        oracle.check_count(self, ROW, "(9) branch accepted_drafts", branch_hot.accepted, expected=branch_cold.accepted)

        # A NO_SPECULATION ticket hits a drafting state (it then splits at (a′) = 9,984): its first score equals the
        # cold-split run's.
        no_spec = self.request(prompt, NS_A, drafting=False)
        oracle.check_count(self, ROW, "(9) NO_SPECULATION on draft state", no_spec.cached,
                           expected=SPEC_DRAFT_STATES[-1])
        split = self.open(SPEC_CONTEXT)
        score = prefill_split(split, prompt, (*SPEC_DRAFT_STATES, SPEC_A_PRIME, len(prompt)))
        oracle.check_bit_exact(self, ROW, "(9) NO_SPECULATION hit == cold split", no_spec.scores[0], score)
        self.drop(split)

        # NO_SPECULATION states (namespace C) are never drafting hits; a drafting session there runs cold.
        self.request(prompt, NS_C, drafting=False)
        unhit = self.request(prompt, NS_C, drafting=True)
        oracle.check_count(self, ROW, "(9) NO_SPECULATION state not a drafting hit", unhit.cached, expected=0)
        self.same("(9) unhit == plain", unhit, plain)
        oracle.check_count(self, ROW, "(9) unhit accepted_drafts", unhit.accepted, expected=cold.accepted)


class K16SpecHost(Sessions):
    """K16 (12), Sessions O211: the D9 host tier on a drafting model keeps draft states (kind 10, the MTP rows)."""

    @classmethod
    def setUpClass(cls):
        with open(DRAFT_MAP, "rb") as f:
            draft_map = torch.frombuffer(bytearray(f.read()), dtype=torch.int32)
        cls.prompt = list(synthetic_ids(SPEC_PROMPT))
        cls.model = oracle.create_model(SPEC_CONTEXT, max_sessions=SPEC_SESSIONS, kv_pool_tokens=SPEC_POOL_TOKENS,
                                        prefix_state_slots=SPEC_HOST_SLOTS, snapshot_host_bytes=ARENA_BYTES,
                                        draft_token_map=draft_map)

    @classmethod
    def tearDownClass(cls):
        oracle.release_model(cls.model)

    def request(self, tokens: list[int], namespace: bytes | None, *, drafting: bool) -> SpecRun:
        session = self.open(SPEC_CONTEXT, namespace)
        run = spec_ticket(session, tokens, SPEC_STEPS, 0 if drafting else CERE_WORK_NO_SPECULATION)
        self.drop(session)
        return run

    def test_host_tier_drafting_hits(self):
        prompt = self.prompt
        plain = self.request(prompt, None, drafting=False)
        cold = self.request(prompt, NS_A, drafting=True)
        oracle.check_bit_exact(self, ROW, "(12) cold scores == plain", cold.scores, plain.scores)
        hot = self.request(prompt, NS_A, drafting=True)  # the never-spilled hit
        oracle.check_count(self, ROW, "(12) device hit cached_tokens", hot.cached, expected=SPEC_DRAFT_STATES[-1])
        before = stats(self.model)
        for index, namespace in enumerate(SPEC_HOST_EVICTORS):
            self.request(variant(index, SPEC_PROMPT), namespace, drafting=True)
        spilled = stats(self.model)
        oracle.check_count(self, ROW, "(12) draft states spilled", int(spilled[HOST_SPILLS] - before[HOST_SPILLS] >= 2),
                           expected=1)
        reloaded = self.request(prompt, NS_A, drafting=True)
        after = stats(self.model)
        oracle.check_count(self, ROW, "(12) host hit cached_tokens", reloaded.cached, expected=SPEC_DRAFT_STATES[-1])
        oracle.check_count(self, ROW, "(12) host loads", int(after[HOST_LOADS] - spilled[HOST_LOADS] >= 1), expected=1)
        oracle.check_bit_exact(self, ROW, "(12) host hit scores == plain", reloaded.scores, plain.scores)
        oracle.check_bit_exact(self, ROW, "(12) host hit scores == device hit", reloaded.scores, hot.scores)
        oracle.check_count(self, ROW, "(12) host hit verify_steps", reloaded.verify_steps, expected=hot.verify_steps)
        oracle.check_count(self, ROW, "(12) host hit accepted_drafts", reloaded.accepted, expected=hot.accepted)
        oracle.report_row(ROW, "(12) host tier drafting hit", phase="diag", host_spills=spilled[HOST_SPILLS] -
                          before[HOST_SPILLS], host_loads=after[HOST_LOADS] - spilled[HOST_LOADS],
                          device_hit_accepted=hot.accepted, host_hit_accepted=reloaded.accepted,
                          device_hit_verify_steps=hot.verify_steps, host_hit_verify_steps=reloaded.verify_steps)


class K16Layout(Sessions):
    """K16 (10), m210 D15⁗′ (Scorecard G6R-1): which kernels make a prefill call layout change the output. Per depth h
    (n = h + 64), layout A is the cache-off one and layout B also ends a call at h, as a hot hit resuming at h does.
    The probe's cascade (every point as computed), its isolation (B's P points replaced by A's rows, so each kernel
    sees A's inputs) and the per-layer state after n are diag rows; the control (h a multiple of the chunk, A == B)
    is bit-exact. -v19's k2_gemm lever (small_m), no draft."""

    cache = False

    @classmethod
    def setUpClass(cls):
        cls.prompt = list(synthetic_ids(max(LAYOUT_DEPTHS + (LAYOUT_CONTROL,)) + LAYOUT_TAIL))
        # Cache off (no prefix_state_slots): eager page binding, so device.forward runs on a fresh session.
        cls.model = oracle.create_model(LAYOUT_CONTEXT, max_sessions=LAYOUT_SESSIONS,
                                        kv_pool_tokens=LAYOUT_POOL_TOKENS,
                                        model_levers=oracle.levers(k2_gemm=CERE_LEVER_K2_SMALL_M))

    @classmethod
    def tearDownClass(cls):
        oracle.release_model(cls.model)

    def test_control_layouts_equal(self):
        n = LAYOUT_CONTROL + LAYOUT_TAIL
        a, b, w0 = layout_ends(n, LAYOUT_CONTROL)
        oracle.check_count(self, ROW, "(10) control layouts", int(a == b), expected=1)
        _, digests, scores, _ = layout_probe(self.model, self.prompt[:n], a, b, w0)
        oracle.check_bit_exact(self, ROW, "(10) control digests", digests[1], digests[0])
        oracle.check_bit_exact(self, ROW, "(10) control scores", scores[1], scores[0])

    def test_layout_divergence(self):
        for h in LAYOUT_DEPTHS:
            with self.case(depth=h):
                n = h + LAYOUT_TAIL
                tokens = self.prompt[:n]
                a, b, w0 = layout_ends(n, h)
                names, digests, scores, need = layout_probe(self.model, tokens, a, b, w0)
                differs = digests[0] != digests[1]
                cut = h - w0
                before, after = differs[:, :cut].any(1), differs[:, cut:].any(1)
                first = [names[int(m.nonzero()[0])] if bool(m.any()) else "" for m in (before, after)]
                oracle.report_row(ROW, f"(10) h {h} cascade", phase="diag", n=n, w0=w0, calls_a=len(a),
                                  calls_b=len(b), entries=len(names), differ_before_h=int(before.sum()),
                                  differ_from_h=int(after.sum()), first_before_h=first[0], first_from_h=first[1],
                                  layers_from_h=point_layers(names, after),
                                  scores_equal=int(torch.equal(scores[0], scores[1])), replace_arena_bytes=need)
                self.isolate(h, tokens, a, b, w0, need)
                self.states(h, tokens, a, b, names)

    def isolate(self, h: int, tokens: list[int], a: list[int], b: list[int], w0: int, need: int) -> None:
        """The replace run: per point (layer-free name) the layers whose B rows differ before h and from h."""
        free, _ = torch.cuda.mem_get_info(oracle.DEVICE)
        if need + LAYOUT_HEADROOM > free:
            oracle.report_row(ROW, f"(10) h {h} isolate", phase="diag", run=0, arena_mib=need >> 20,
                              free_mib=free >> 20)
            return
        names, digests, scores, _ = layout_probe(self.model, tokens, a, b, w0, replace=1, arena_bytes=need)
        differs = digests[0] != digests[1]
        cut = h - w0
        before, after = differs[:, :cut].any(1), differs[:, cut:].any(1)
        points: dict[str, list[int]] = {}
        for i, name in enumerate(names):
            points.setdefault(LAYER_RE.sub(".layers.*.", name), []).append(i)
        oracle.report_row(ROW, f"(10) h {h} isolate", phase="diag", run=1, arena_mib=need >> 20,
                          differ_before_h=int(before.sum()), differ_from_h=int(after.sum()),
                          scores_equal=int(torch.equal(scores[0], scores[1])))
        for point, rows in points.items():
            index = torch.tensor(rows)
            if bool(before[index].any()) or bool(after[index].any()):
                select = [names[i] for i in rows]
                oracle.report_row(ROW, f"(10) h {h} isolate point", phase="diag", point=point,
                                  layers_before_h=point_layers(select, before[index]),
                                  layers_from_h=point_layers(select, after[index]))

    def states(self, h: int, tokens: list[int], a: list[int], b: list[int], names: list[str]) -> None:
        """Each layout's session state after n, per section kind: the layers (model numbering) whose bytes differ and,
        for K and V, the first row that differs."""
        attention = sorted({int(m.group(1)) for name in names if ".self_attn." in name
                            and (m := LAYER_RE.search(name))})
        gdn = [layer for layer in range(LAYOUT_LAYERS) if layer not in attention]
        runs = []
        for ends in (a, b):
            session = self.open(LAYOUT_CONTEXT)
            score = prefill_split(session, tokens, tuple(ends))
            runs.append((score, *state_image(session)))
            self.drop(session)
        (score_a, table, image_a), (score_b, _, image_b) = runs
        oracle.report_row(ROW, f"(10) h {h} state", phase="diag", attention_layers=layer_list(attention),
                          scores_equal=int(torch.equal(score_a, score_b)))
        for kind, offset, size in table:
            if size == 0:  # e.g. the QSA ring when n is a multiple of the compress ratio
                continue
            layers = attention if kind in (ATTN_K, ATTN_V, QSA_KEYS, QSA_RING) else gdn if kind in (
                GDN_CONV, GDN_SSM) else [LAYOUT_LAYERS]
            x = image_a[offset:offset + size].view(len(layers), -1)
            y = image_b[offset:offset + size].view(len(layers), -1)
            per_layer = (x != y).any(1)
            fields = {"kind": kind, "layers_differ": layer_list(l for l, d in zip(layers, per_layer.tolist()) if d)}
            if kind in (ATTN_K, ATTN_V) and bool(per_layer.any()):
                rows = (x != y).view(len(layers), len(tokens), LAYOUT_KV_WIDTH).any(2).any(0)
                fields["first_row_differ"] = int(rows.nonzero()[0])
                fields["rows_differ"] = int(rows.sum())
            oracle.report_row(ROW, f"(10) h {h} state section", phase="diag", **fields)


class K16End(Sessions):
    """K16 (11), m210 D15⁗′ stage 2a: end states on a drafting model (the served draft map). Per prompt length n
    (b = floor64(n − 1) strictly inside the cold call [s, e), class κ = e − s; s = floor_C(b) >= 4,096, or s = 0 for a
    dense chunk-0 call whose end state also holds its BF16 K/V rows [0, b), PrefixCapture G1): the cold drafting ticket
    inserts its draft states (the call ends below n) and the end state at b; the hot ticket of the same prompt hits at
    b and its tokens, scores and speculation counters equal the cache-off (un-namespaced) drafting run's bit-exactly; a
    1-token ticket's draft digest (state + MTP entry + proposals at the call end) equals the cache-off one. At 4.5K: a
    prompt sharing tokens[0, b + 1) with class 1,100 misses the end state (cached_tokens = s) and equals its cache-off
    run; the κ mutant (the end state's class forced to 1,100, so that prompt hits it and resumes it with the wrong
    class) must differ from its cache-off digest.

    K16 (13), m210 D15⁗″ ResumeExact: a lookup hits an end state whose σ (the device's row-bit signature at its class)
    equals the lookup class's, and a resumed call captures its own end state, so a conversation hits its previous
    turn's end (`test_chain`, `test_chunk0_chain`, `test_short_tail`); `test_row_sig` checks σ against every row range
    of the second chunk and of chunk 0, `test_sigma_mutant` that a σ mismatch would change the bits."""
    LEVERS: dict[str, int] = {}  # the model's slot choices, lever -> CERE_LEVER_* (empty: null selection, every default)
    WORK_FLAGS = 0  # the tickets' cere_work.flags (0: a drafting greedy PREFILL, MTP drafts, fixed policy)
    # The κ mutant's digest differs from cache-off (1) when the wrong class changes the bits: the default kernels
    # pick split-K attention and cuBLASLt buckets by class. A lever set whose prefill is row-partition invariant at
    # both classes resumes the wrong class bit for bit (0), which the subclass states.
    KAPPA_MUTANT_DIFFERS = 1
    # K16 (13): a resume across moe_shared_gate_up split ranges (SIGMA_MUTANTS) differs from cache-off (1) when the
    # split changes the rows' bits; 0 for a lever set whose prefill GEMMs split no K.
    SIGMA_MUTANT_DIFFERS = 1

    @classmethod
    def setUpClass(cls):
        with open(cls.draft_map_path(), "rb") as f:
            draft_map = torch.frombuffer(bytearray(f.read()), dtype=torch.int32)
        cls.prompt = list(synthetic_ids(max(END_PROMPTS)))
        cls.model = oracle.create_model(END_CONTEXT, max_sessions=END_SESSIONS, kv_pool_tokens=END_POOL_TOKENS,
                                        prefix_state_slots=END_SLOTS, draft_token_map=draft_map,
                                        model_levers=oracle.levers(**cls.LEVERS) if cls.LEVERS else None)
        cls.addClassCleanup(oracle.release_model, cls.model)
        oracle.call("cere_ktest_prefix_end_state", cls.model, 1)  # the opt-in key; off by default

    @classmethod
    def draft_map_path(cls) -> str:
        """The served draft token map (u32 ids): the pinned artifact's, mounted by native_dev.py."""
        return DRAFT_MAP

    def request(self, tokens: list[int], namespace: bytes | None, steps: int = END_STEPS
                ) -> tuple[SpecRun, torch.Tensor]:
        """One greedy drafting PREFILL ticket on a fresh session and its draft digest after; released after."""
        session = self.open(END_CONTEXT, namespace)
        run = spec_ticket(session, tokens, steps, self.WORK_FLAGS)
        out = torch.zeros(32, dtype=torch.uint8)
        oracle.call("cere_ktest_draft_digest", session, out)
        self.drop(session)
        return run, out

    def same(self, case: str, run: SpecRun, reference: SpecRun) -> None:
        oracle.check_bit_exact(self, ROW, f"{case} scores", run.scores, reference.scores)
        oracle.check_count(self, ROW, f"{case} verify_steps", run.verify_steps, expected=reference.verify_steps)
        oracle.check_count(self, ROW, f"{case} accepted_drafts", run.accepted, expected=reference.accepted)

    def test_end_state_hits(self):
        for n in END_PROMPTS:
            with self.case(n=n):
                flush(self.model)
                self.length(n)

    def test_chunk0_dense_over(self):
        """PrefixCapture G1: a chunk-0 cold call above the dense limit (the K9 path) saves no K/V run, so it captures
        no end state: the cold ticket inserts nothing and the hot one misses (cached_tokens 0); both equal the
        cache-off run."""
        flush(self.model)
        tokens = self.prompt[:END_DENSE_OVER]
        plain, _ = self.request(tokens, None)
        before = stats(self.model)
        cold, _ = self.request(tokens, NS_A)
        oracle.check_count(self, ROW, "11 dense over cold inserts", stats(self.model)[INSERTS] - before[INSERTS],
                           expected=0)
        hot, _ = self.request(tokens, NS_A)
        oracle.check_count(self, ROW, "11 dense over hot cached_tokens", hot.cached, expected=0)
        self.same("11 dense over cold == plain", cold, plain)
        self.same("11 dense over hot == plain", hot, plain)

    def test_chain(self):
        self.conversation("chain", CHAIN)

    def test_chunk0_chain(self):
        self.conversation("chunk0 chain", CHUNK0_CHAIN)

    def test_short_tail(self):
        self.conversation("short tail", SHORT_TAIL)

    def conversation(self, label: str, turns: tuple[int, ...]) -> None:
        """K16 (13): the turns of one conversation (namespace A, each prompt extends the previous one) from an empty
        tree. A turn hits the deepest state on its path the rule allows: an end state whose σ(s, κ) equals σ(s, κ′)
        (κ′ = min(s + C, n) − s, the lookup's class) or a draft state at a multiple of C strictly inside an earlier
        prompt; its scores, verify steps, accepted drafts and draft digest equal the cache-off run's. A turn captures
        its end state at b = floor64(n − 1) when b lies past its hit and strictly inside its last call [s, e), and,
        at s = 0, the call is dense (PrefixCapture G1)."""
        flush(self.model)
        ends: list[tuple[int, int, int]] = []  # (b, s, κ) of the end states the turns captured
        longest = 0  # the longest earlier prompt
        for i, n in enumerate(turns):
            tokens = self.prompt[:n]
            case = f"13 {label} turn {i} n {n}"
            plain, plain_digest = self.request(tokens, None)
            hits = [b for b, s, k in ends
                    if b < n and row_sig(self.model, s, k) == row_sig(self.model, s, min(s + END_CHUNK, n) - s)]
            expected = max(hits + [max(range(0, min(n, longest), END_CHUNK), default=0)])
            before = stats(self.model)
            hot, hot_digest = self.request(tokens, NS_A)
            after = stats(self.model)
            oracle.check_count(self, ROW, f"{case} cached_tokens", hot.cached, expected=expected)
            self.same(f"{case} hot == plain", hot, plain)
            oracle.check_bit_exact(self, ROW, f"{case} digest", hot_digest, plain_digest)
            b = (n - 1) // PAGE_TOKENS * PAGE_TOKENS
            s = b // END_CHUNK * END_CHUNK
            k = min(s + END_CHUNK, n) - s
            if b > max(expected, s) and (s > 0 or k <= CHUNK0_DENSE):
                ends.append((b, s, k))
            longest = max(longest, n)
            oracle.report_row(ROW, f"{case} turn", phase="diag", cached=hot.cached,
                              hits=after[HITS_AT] - before[HITS_AT],
                              inserts=after[INSERTS] - before[INSERTS], b=b, cls=k, sig=row_sig(self.model, s, k),
                              hot_accept_length=hot.accept_length(), plain_accept_length=plain.accept_length())

    def test_row_sig(self):
        """K16RowSig (S5): σ leaves a choice out only if it never changes a row's bits. For s in ROW_SIG_STARTS and
        every class κ at both ends of every row range without a per-call choice change, a cold ticket of s + κ
        captures its end state at b = floor64(s + κ − 1); the ticket of s + κ_ref (κ_ref: the largest tested class of
        σ(s, κ)) hits it at b and equals its cache-off run (scores and draft digest). Equality to one reference over
        each state's own rows [s, b) gives equality between any two σ-equal classes."""
        for s, cap in ROW_SIG_STARTS:
            bounds = row_choice_bounds(cap)
            classes = sorted({k for i, lo in enumerate(bounds)
                              for k in (lo, bounds[i + 1] - 1 if i + 1 < len(bounds) else cap)})
            sig = {k: row_sig(self.model, s, k) for k in classes}
            ref = {sig[k]: k for k in classes}  # ascending: the largest class of each σ
            plains: dict[int, tuple[SpecRun, torch.Tensor]] = {}
            pairs = 0
            for k in classes:
                k_ref = ref[sig[k]]
                if k == k_ref:
                    continue
                with self.case(s=s, cls=k):
                    case = f"13 row sig s {s} cls {k} ref {k_ref}"
                    if k_ref not in plains:
                        plains[k_ref] = self.request(self.prompt[:s + k_ref], None, 1)
                    plain, plain_digest = plains[k_ref]
                    flush(self.model)
                    self.request(self.prompt[:s + k], NS_A, 1)
                    hot, hot_digest = self.request(self.prompt[:s + k_ref], NS_A, 1)
                    oracle.check_count(self, ROW, f"{case} cached_tokens", hot.cached,
                                       expected=(s + k - 1) // PAGE_TOKENS * PAGE_TOKENS)
                    oracle.check_bit_exact(self, ROW, f"{case} scores", hot.scores, plain.scores)
                    oracle.check_bit_exact(self, ROW, f"{case} digest", hot_digest, plain_digest)
                    pairs += 1
            oracle.report_row(ROW, f"13 row sig s {s}", phase="diag", classes=len(classes), signatures=len(ref),
                              pairs=pairs)

    def test_sigma_mutant(self):
        """S2: for (κ, κ′) of SIGMA_MUTANTS at s = C (different moe_shared_gate_up split ranges) σ differ, so the
        ticket of s + κ′ misses the end state of s + κ (cached s, the draft state) and equals cache-off; with the end
        state's class forced to κ′ it hits and resumes rows [s, b) computed under κ's split, which differs from
        cache-off iff SIGMA_MUTANT_DIFFERS."""
        s = END_CHUNK
        ns = torch.tensor(list(NS_A), dtype=torch.uint8)
        for k, k_q in SIGMA_MUTANTS:
            with self.case(cls=k, lookup=k_q):
                case = f"13 sigma {k} to {k_q}"
                oracle.check_count(self, ROW, f"{case} sig equal",
                                   int(row_sig(self.model, s, k) == row_sig(self.model, s, k_q)), expected=0)
                tokens = self.prompt[:s + k]
                b = (s + k - 1) // PAGE_TOKENS * PAGE_TOKENS
                q = self.prompt[:s + k_q]
                plain, plain_digest = self.request(q, None, 1)
                flush(self.model)
                self.request(tokens, NS_A, 1)
                miss, miss_digest = self.request(q, NS_A, 1)
                oracle.check_count(self, ROW, f"{case} miss cached_tokens", miss.cached, expected=s)
                oracle.check_bit_exact(self, ROW, f"{case} miss scores", miss.scores, plain.scores)
                oracle.check_bit_exact(self, ROW, f"{case} miss digest", miss_digest, plain_digest)
                flush(self.model)
                self.request(tokens, NS_A, 1)
                old = torch.zeros(1, dtype=torch.int64)
                oracle.call("cere_ktest_prefix_end_class", self.model, ns, torch.tensor(tokens[:b], dtype=torch.int32),
                            b, k_q, old)
                oracle.check_count(self, ROW, f"{case} mutant old class", int(old.item()), expected=k)
                mutant, mutant_digest = self.request(q, NS_A, 1)
                oracle.check_count(self, ROW, f"{case} mutant cached_tokens", mutant.cached, expected=b)
                differs = int(not torch.equal(mutant_digest, plain_digest))
                oracle.check_count(self, ROW, f"{case} mutant differs", differs, expected=self.SIGMA_MUTANT_DIFFERS)

    def length(self, n: int) -> None:
        tokens = self.prompt[:n]
        b = (n - 1) // PAGE_TOKENS * PAGE_TOKENS
        s = b // END_CHUNK * END_CHUNK
        case = f"11 n {n}"  # GPU1 lab rows (model_v22) keep only LAB_REPORT_STRING labels: no leading "(", no "κ"
        plain, _ = self.request(tokens, None)
        _, plain_digest = self.request(tokens, None, 1)
        before = stats(self.model)
        cold, _ = self.request(tokens, NS_A)
        seeded = stats(self.model)
        oracle.check_count(self, ROW, f"{case} cold cached_tokens", cold.cached, expected=0)
        # The draft states below n, the end state at b and, when n % 64 == 0, the prompt-end decode capture at n.
        oracle.check_count(self, ROW, f"{case} cold inserts", seeded[INSERTS] - before[INSERTS],
                           expected=n // END_CHUNK + 1 + int(n % PAGE_TOKENS == 0))
        self.same(f"{case} cold == plain", cold, plain)
        hot, _ = self.request(tokens, NS_A)
        oracle.check_count(self, ROW, f"{case} hot cached_tokens", hot.cached, expected=b)
        oracle.check_count(self, ROW, f"{case} hot hits", stats(self.model)[HITS_AT] - seeded[HITS_AT], expected=1)
        self.same(f"{case} hot == plain", hot, plain)
        once, hot_digest = self.request(tokens, NS_A, 1)
        oracle.check_count(self, ROW, f"{case} hot digest cached_tokens", once.cached, expected=b)
        oracle.check_bit_exact(self, ROW, f"{case} hot digest", hot_digest, plain_digest)
        oracle.report_row(ROW, f"{case} end state", phase="diag", b=b, s=s, cls=min(s + END_CHUNK, n) - s,
                          hot_accept_length=hot.accept_length(), plain_accept_length=plain.accept_length())
        if n != END_PROMPTS[0]:
            return
        # κ mismatch: a longer prompt in the same cold call (class END_MISS - s) shares tokens[0, b + 1): it misses the
        # end state at b (its class is n - s) and takes the draft state at s. It captures its own end state, so the
        # mutant below starts from a re-seeded tree.
        q = self.prompt[:END_MISS]
        q_plain, q_digest = self.request(q, None, 1)
        miss, miss_digest = self.request(q, NS_A, 1)
        oracle.check_count(self, ROW, "11 kappa mismatch cached_tokens", miss.cached, expected=s)
        oracle.check_bit_exact(self, ROW, "11 kappa mismatch scores", miss.scores, q_plain.scores)
        oracle.check_bit_exact(self, ROW, "11 kappa mismatch digest", miss_digest, q_digest)
        # The κ mutant: the end state at b of a re-seeded tree gets q's class, so q hits it and resumes it with the
        # wrong class; that run differs from q's cache-off one iff KAPPA_MUTANT_DIFFERS.
        flush(self.model)
        self.request(tokens, NS_A, 1)
        old = torch.zeros(1, dtype=torch.int64)
        ns = torch.tensor(list(NS_A), dtype=torch.uint8)
        oracle.call("cere_ktest_prefix_end_class", self.model, ns, torch.tensor(tokens[:b], dtype=torch.int32), b,
                    END_MISS - s, old)
        oracle.check_count(self, ROW, "11 mutant old class", int(old.item()), expected=min(s + END_CHUNK, n) - s)
        mutant, mutant_digest = self.request(q, NS_A, 1)
        oracle.check_count(self, ROW, "11 mutant cached_tokens", mutant.cached, expected=b)
        oracle.check_count(self, ROW, "11 mutant differs", int(not torch.equal(mutant_digest, q_digest)),
                           expected=self.KAPPA_MUTANT_DIFFERS)


if __name__ == "__main__":
    unittest.main()
