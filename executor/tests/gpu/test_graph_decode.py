"""P1 (tasks/m23-spec.md P1, M23P1-01..08): the decode graph equals the eager decode step bit for bit on the real A4
model, container only.

Suite `graph` (`native_dev.py --gpu 1 test --suite graph`, a GPU1 suite: every row is native-vs-native and
bit-exact, no timing, memory floor or GPU0 identity). Models are created, uploaded and sealed through
`oracle.create_model`; each class owns one model and releases it before the next class starts. The prompts are
seeded synthetic ids (no text); token IDs stay in memory and the report rows carry counts and SHA-256 digests only.
The rows hold on either MoE backend of the build under test (M23P1-10): the default P3b build (M23-D7) captures P3b's
block, a `--moe-backend k7` build K7's.

Each case runs `cere_ktest_forward_decode_rows` twice on the same model with the same prompts: once with the
model's decode graphs switched off (`cere_ktest_decode_graph` mode 0, every step eager) and once switched on (mode
1). Every session joins at round 0 and leaves at STEPS, so every round is one decode step of all R rows (R = 1: the
single-row DECODE forward, decode graph 0; R >= 2: forward_decode_rows, decode graph R), and every step after the
prefill is eligible (greedy, unconstrained, text-only, no taps, after the first token). Gates per case: the score
rows (the prefill's and every decode's top-2, hence the greedy tokens) of the graph run equal the eager run's bit
for bit; the eager run moves no decode-graph counter; the graph run's replays grow by exactly the decode steps and
its captures plus in-place updates by at least one. The cases (M23P1-08) put the decode positions across the
4-token QSA block, the 64-token page, the 2,048-token dense / sparse switch (K9's top-k saturation: above 2,048
visible tokens the selection drops blocks) and up to the profile context:
* `GraphRowsK6.test_rows_<R>` (R = 1 .. 4): a 2,048-token context, so attention is K6 over the FP8 cache
  (full_attention_step); prompts PROMPTS_K6, the first R of them. `GraphRowsK6.test_default_eager`: a model created
  with null levers (decode_step at its default, eager) has its decode graphs off; a query (mode -1) and a refused
  mode (2) leave the state unchanged.
* `GraphRowsQsa.test_rows_<R>` (R = 1 .. 4): a 4,096-token context, so attention is K6 dense up to position 2,047
  and the K9 QSA path after it (indexer, compress-store, select, sparse attention); prompts PROMPTS_QSA, the first
  R of them (session 0 and 1 cross 2,048, so their key flips and the graph is captured again mid-run).
* `GraphLong.test_rows_1_at_profile_context`: the D12 profile's 200,000-token plan (pool 200,000), one session whose
  STEPS decodes end at position 199,999. The model is created with the decode_step lever CERE_LEVER_DECODE_GRAPH
  (tasks/m23-spec.md PI-02), so the lever is checked too: the first query reports the graphs enabled.
* `GraphSlotReuse.test_slot_reused_by_another_context` (P1R-01): a one-session 2,048-token plan, so every session
  takes slot 0. A session of context REUSE_SHORT is teacher-forced (`cere_ktest_session_feed`: the prompt's PREFILL,
  then STEPS single-row DECODE forwards of synthetic ids) with the graphs off; with them on, a session of the plan's
  context runs the same way (it captures decode graph 0), then the REUSE_SHORT session again. The slot carves its
  recurrent state by the session's context, so the third session's score rows equal the first's bit for bit only
  when it does not replay the second session's graph: its STEPS decodes are run by a graph and capture or update
  it at least once.
* `GraphOption.test_malformed_option_refused`: `cere_ktest_model_create_device` with P1's former device-id option
  `decode_graph` (any value; the decode_step lever replaced it) is CERE_ERR_ARGUMENT, an unknown option (the parse
  fails before any device work); `cere_ktest_decode_graph` on a null handle is CERE_ERR_ARGUMENT.
* `GraphHotSwap.test_swap_equals_a_fresh_model` (SelfTune ST2 (c), hot swap by plan rebuild): a 2,048-token,
  two-session model created with null levers is swapped through SWAPS (eager -> graph -> spec_graph ->
  spec_graph + small_m -> eager) by `cere_ktest_model_set_selection` (K8's, the host's `cere_model_set_selection`),
  with a two-row decode run (PROMPTS_K6's first two) after each. The swap moves no counter and sets the graph flags
  (`cere_ktest_spec_graph` -1 queries the spec flag); a graph selection's run captures anew (the swap dropped every
  graph) and replays every step, an eager run moves nothing; every run's score rows equal a fresh model created
  with that selection bit for bit. A k2_weights swap (fixed at create) is CERE_ERR_UNSUPPORTED and changes nothing.

Native entry points (K8's `cere_ktest_model_*` and `cere_ktest_forward_decode_rows` as stated in
`test_model_k8_forward.py`, K15's `cere_ktest_session_open`, `cere_ktest_session_feed` (max_new_tokens 0, no names)
and `cere_ktest_session_close` as stated in `test_model_k15_snapshot.py`; `oracle.call` appends the stream; 0 =
success, otherwise a `cere_status`; "host" buffers are CPU memory). P1's own:

    int cere_ktest_decode_graph(uint64_t model, int32_t mode, uint64_t *out /*host [4]*/,
                                cudaStream_t /*stream*/);
        tasks/m23-spec.md P1 (M23P1-07/08): mode 0 or 1 switches the model's decode graphs off or on for its later
        steps (no step may run concurrently); -1 leaves them as they are. out after the switch: {enabled, graphs
        instantiated, executables updated in place, steps run by a graph}, the counters from the model's
        INSTANTIATE. Any other mode, a null out or an unknown handle: CERE_ERR_ARGUMENT, nothing changed.
"""

from __future__ import annotations

import hashlib
import unittest
from typing import NamedTuple

import torch

import oracle

ROW = "P1"
STEPS = 24
SEED = 23
# Synthetic ids below the vocabulary's special tokens.
ID_LIMIT = 150_000
CERE_ERR_ARGUMENT = -1
CERE_LEVER_DECODE_GRAPH = 1
CERE_LEVER_DECODE_SPEC_GRAPH = 2
CERE_LEVER_K2_SMALL_M = 1
CERE_LEVER_K2_WEIGHTS_FP8 = 1

CONTEXT_K6 = 2_048  # k6::kMaxContext: K6 attention at every position
# positions 2,024 .. 2,047 (the context's end), 60 .. 83 (page 64), 1,020 .. 1,043 (page 1,024), 5 .. 28
PROMPTS_K6 = (2_024, 60, 1_020, 5)
CONTEXT_QSA = 4_096  # above k6::kMaxContext: K9 QSA from position 2,048
# positions 2,030 .. 2,053 and 2,044 .. 2,067 (the 2,048 switch and page), 120 .. 143 (page 128), 3 .. 26
PROMPTS_QSA = (2_030, 2_044, 120, 3)
SESSIONS = 4
CONTEXT_LONG = 200_000  # native-g0.toml max_context_tokens and kv_pool_tokens (the D12 profile plan)
PROMPT_LONG = CONTEXT_LONG - STEPS  # decodes at positions 199,976 .. 199,999
REUSE_SHORT = 1_024  # the reusing session's context: half the plan's, so its slot map ends 8 KiB earlier
# the plan-context session's prompt; the short session's decodes at positions 500 .. 523 (page 512)
PROMPTS_REUSE = (700, 500)
# GraphHotSwap's selections, in swap order (name, registry lever -> CERE_LEVER_* value): eager -> graph -> spec_graph
# -> spec_graph with small_m -> eager.
SWAPS = (("eager", {}), ("graph", {"decode_step": CERE_LEVER_DECODE_GRAPH}),
         ("spec_graph", {"decode_step": CERE_LEVER_DECODE_SPEC_GRAPH}),
         ("spec_graph+small_m", {"decode_step": CERE_LEVER_DECODE_SPEC_GRAPH, "k2_gemm": CERE_LEVER_K2_SMALL_M}),
         ("eager", {}))


class Counters(NamedTuple):
    enabled: int
    captures: int
    updates: int
    replays: int


def decode_graph(handle: int, mode: int) -> Counters:
    """`cere_ktest_decode_graph`: switch (mode 0 / 1) or query (-1) the model's decode graphs."""
    out = torch.zeros(4, dtype=torch.int64)
    oracle.call("cere_ktest_decode_graph", handle, mode, out)
    return Counters(*(int(v) for v in out.tolist()))


def spec_graph(handle: int) -> int:
    """`cere_ktest_spec_graph` query (-1): whether the model's spec (verify / draft) graphs are on."""
    out = torch.zeros(7, dtype=torch.int64)
    oracle.call("cere_ktest_spec_graph", handle, -1, out)
    return int(out[0])


def selection(**values: int) -> torch.Tensor:
    """`oracle.levers(**values)` as the uint8 buffer `cere_ktest_model_set_selection` reads."""
    return torch.frombuffer(bytearray(bytes(oracle.levers(**values))), dtype=torch.uint8)


def prompts_of(lengths: tuple) -> list:
    """Seeded synthetic prompts of `lengths` tokens (in memory only)."""
    gen = torch.Generator().manual_seed(SEED)
    return [torch.randint(0, ID_LIMIT, (n,), generator=gen, dtype=torch.int64).tolist() for n in lengths]


def decode_rows(handle: int, prompts: list) -> torch.Tensor:
    """`cere_ktest_forward_decode_rows` greedy, untapped: every session joins at round 0 and leaves at STEPS. Returns
    the score rows, int32 [S, STEPS + 1, 4] (cere_token_score bytes)."""
    sessions = len(prompts)
    flat = torch.tensor([t for p in prompts for t in p], dtype=torch.int32)
    lens = torch.tensor([len(p) for p in prompts], dtype=torch.int64)
    join = torch.zeros(sessions, dtype=torch.int64)
    leave = torch.full((sessions,), STEPS, dtype=torch.int64)
    scores = torch.full((sessions * (STEPS + 1), 4), -1, dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int64)
    oracle.call("cere_ktest_forward_decode_rows", handle, sessions, flat, lens, join, leave, None, scores,
                None, 0, None, 0, None, 0, counts)
    return scores.view(sessions, STEPS + 1, 4)


def run_session(handle: int, context: int, prompt: list, forced: list) -> torch.Tensor:
    """A session of `context` on `handle`, teacher-forced through `cere_ktest_session_feed`: the PREFILL of `prompt`,
    then one DECODE per id of `forced`; closed before return. The score rows, int32 [1 + len(forced), 4]: the
    prefill's (it predicts the first decode's token), then each decode's."""
    out = torch.zeros(1, dtype=torch.int64)
    oracle.call("cere_ktest_session_open", handle, context, out)
    session = int(out[0])
    try:
        rows = []
        for tokens in (prompt, forced):
            ids = torch.tensor(tokens, dtype=torch.int32)
            scores = torch.full((len(tokens), 4), -1, dtype=torch.int32)
            oracle.call("cere_ktest_session_feed", session, ids, len(tokens), 0, 0, None, 0, None, 0, None, 0, scores,
                        torch.zeros(3, dtype=torch.int64))
            rows.append(scores)
    finally:
        oracle.call("cere_ktest_session_close", session)
    return torch.cat([rows[0][:1], rows[1]])


class _GraphCase(unittest.TestCase):
    CONTEXT = 0
    SESSIONS = 1
    POOL = 0
    DECODE_STEP: int | None = None  # the desc's decode_step lever; None = null levers
    PROMPTS: tuple = ()
    handle = 0

    @classmethod
    def setUpClass(cls):
        levers = None if cls.DECODE_STEP is None else oracle.levers(decode_step=cls.DECODE_STEP)
        cls.handle = oracle.create_model(cls.CONTEXT, max_sessions=cls.SESSIONS, kv_pool_tokens=cls.POOL,
                                         model_levers=levers)
        cls.addClassCleanup(oracle.release_model, cls.handle)
        cls.initial = decode_graph(cls.handle, -1)

    def graph_equals_eager(self, rows: int) -> None:
        case = f"{type(self).__name__}/ctx{self.CONTEXT}/r{rows}"
        prompts = prompts_of(self.PROMPTS[:rows])
        before = decode_graph(self.handle, 0)
        eager = decode_rows(self.handle, prompts)
        after_eager = decode_graph(self.handle, -1)
        self.assertEqual(after_eager, before, f"{case}: the eager run moved a decode-graph counter")
        decode_graph(self.handle, 1)
        graph = decode_rows(self.handle, prompts)
        after = decode_graph(self.handle, -1)
        self.assertTrue(bool((eager[..., 0] >= 0).all()), f"{case}: an eager score row is unwritten")
        replays = after.replays - after_eager.replays
        made = after.captures - after_eager.captures + after.updates - after_eager.updates
        oracle.report_row(ROW, case, rows=rows, steps=STEPS, prompt_tokens=sum(len(p) for p in prompts),
                          last_position=max(len(p) for p in prompts) + STEPS - 1, replays=replays,
                          captures=after.captures - after_eager.captures,
                          updates=after.updates - after_eager.updates,
                          eager_sha256=hashlib.sha256(eager.numpy().tobytes()).hexdigest(),
                          graph_sha256=hashlib.sha256(graph.numpy().tobytes()).hexdigest())
        oracle.check_bit_exact(self, ROW, case, graph, eager)
        self.assertEqual(replays, STEPS, f"{case}: steps run by a graph")
        self.assertTrue(made >= 1, f"{case}: no graph was captured")


class GraphRowsK6(_GraphCase):
    CONTEXT = CONTEXT_K6
    SESSIONS = SESSIONS
    PROMPTS = PROMPTS_K6

    def test_default_eager(self):
        """decode_step defaults to eager (a model created with null levers): graphs off; a query (-1) and a refused
        mode (2) change nothing."""
        self.assertEqual(self.initial.enabled, 0, "decode_step defaults to eager")
        state = decode_graph(self.handle, -1)
        with self.assertRaisesRegex(oracle.OracleError, f"status {CERE_ERR_ARGUMENT}$"):
            oracle.call("cere_ktest_decode_graph", self.handle, 2, torch.zeros(4, dtype=torch.int64))
        self.assertEqual(decode_graph(self.handle, -1), state, "a query or a refused mode changed the state")

    def test_rows_1(self):
        self.graph_equals_eager(1)

    def test_rows_2(self):
        self.graph_equals_eager(2)

    def test_rows_3(self):
        self.graph_equals_eager(3)

    def test_rows_4(self):
        self.graph_equals_eager(4)


class GraphRowsQsa(_GraphCase):
    CONTEXT = CONTEXT_QSA
    SESSIONS = SESSIONS
    PROMPTS = PROMPTS_QSA

    def test_rows_1(self):
        self.graph_equals_eager(1)

    def test_rows_2(self):
        self.graph_equals_eager(2)

    def test_rows_3(self):
        self.graph_equals_eager(3)

    def test_rows_4(self):
        self.graph_equals_eager(4)


class GraphLong(_GraphCase):
    CONTEXT = CONTEXT_LONG
    POOL = CONTEXT_LONG
    DECODE_STEP = CERE_LEVER_DECODE_GRAPH
    PROMPTS = (PROMPT_LONG,)

    def test_rows_1_at_profile_context(self):
        self.assertEqual(self.initial.enabled, 1, "the decode_step lever reached the model")
        self.graph_equals_eager(1)


class GraphSlotReuse(_GraphCase):
    CONTEXT = CONTEXT_K6

    def test_slot_reused_by_another_context(self):
        """P1R-01: a session reusing slot 0 with another context does not replay the previous session's graph."""
        case = f"GraphSlotReuse/ctx{REUSE_SHORT}-after-ctx{CONTEXT_K6}"
        long_prompt, short_prompt = prompts_of(PROMPTS_REUSE)
        forced = prompts_of((STEPS,))[0]
        before = decode_graph(self.handle, 0)
        eager = run_session(self.handle, REUSE_SHORT, short_prompt, forced)
        self.assertEqual(decode_graph(self.handle, -1), before, f"{case}: the eager run moved a decode-graph counter")
        decode_graph(self.handle, 1)
        run_session(self.handle, CONTEXT_K6, long_prompt, forced)
        mid = decode_graph(self.handle, -1)
        self.assertEqual(mid.replays - before.replays, STEPS, f"{case}: the plan-context session's graph replays")
        graph = run_session(self.handle, REUSE_SHORT, short_prompt, forced)
        after = decode_graph(self.handle, -1)
        self.assertTrue(bool((eager[..., 0] >= 0).all()), f"{case}: an eager score row is unwritten")
        replays = after.replays - mid.replays
        made = after.captures - mid.captures + after.updates - mid.updates
        oracle.report_row(ROW, case, rows=1, steps=STEPS, prompt_tokens=len(short_prompt),
                          last_position=len(short_prompt) + STEPS - 1, replays=replays,
                          captures=after.captures - mid.captures, updates=after.updates - mid.updates,
                          eager_sha256=hashlib.sha256(eager.numpy().tobytes()).hexdigest(),
                          graph_sha256=hashlib.sha256(graph.numpy().tobytes()).hexdigest())
        oracle.check_bit_exact(self, ROW, case, graph, eager)
        self.assertEqual(replays, STEPS, f"{case}: steps run by a graph")
        self.assertTrue(made >= 1, f"{case}: the reusing session replayed the previous session's graph")


class GraphOption(unittest.TestCase):
    def test_malformed_option_refused(self):
        for device_id in ("0,decode_graph=on", "0,decode_graph=off", "0,decode_graph=maybe", "0,decode_graph="):
            with self.subTest(device_id=device_id):
                with self.assertRaisesRegex(oracle.OracleError, f"status {CERE_ERR_ARGUMENT}$"):
                    oracle.create_model(CONTEXT_K6, device_id=device_id)
        # an unknown (null) handle and an unknown mode are refused by the P1 export, nothing changed
        with self.assertRaisesRegex(oracle.OracleError, f"status {CERE_ERR_ARGUMENT}$"):
            oracle.call("cere_ktest_decode_graph", 0, -1, torch.zeros(4, dtype=torch.int64))
        with self.assertRaisesRegex(oracle.OracleError, f"status {CERE_ERR_ARGUMENT}$"):
            oracle.call("cere_ktest_decode_graph", 0, 2, torch.zeros(4, dtype=torch.int64))


class GraphHotSwap(unittest.TestCase):
    def test_swap_equals_a_fresh_model(self):
        """SelfTune ST2 (c), hot swap by plan rebuild (D4/E6): one model created with null levers is swapped
        (`cere_ktest_model_set_selection`, the host's `cere_model_set_selection`) through SWAPS, with a decode run of
        two rows after each. A swap captures nothing and drops every graph: the run after a graph selection captures
        anew (no replay of the previous selection's graph) and replays every step; after an eager one no counter
        moves. Each run's score rows equal those of a fresh model created with that selection, bit for bit. A refused
        swap (k2_weights is fixed at create: UNSUPPORTED) changes nothing."""
        prompts = prompts_of(PROMPTS_K6[:2])
        swapped = []
        handle = oracle.create_model(CONTEXT_K6, max_sessions=2)
        try:
            for name, values in SWAPS:
                before = decode_graph(handle, -1)
                oracle.call("cere_ktest_model_set_selection", handle, selection(**values))
                state = decode_graph(handle, -1)
                graphed = values.get("decode_step", 0) != 0
                self.assertEqual(state.enabled, int(graphed), f"{name}: the swap set decode graphs")
                self.assertEqual(state[1:], before[1:], f"{name}: the swap itself moved a counter")
                self.assertEqual(spec_graph(handle), int(values.get("decode_step", 0) == CERE_LEVER_DECODE_SPEC_GRAPH),
                                 f"{name}: the swap set spec graphs")
                scores = decode_rows(handle, prompts)
                after = decode_graph(handle, -1)
                captures, replays = after.captures - state.captures, after.replays - state.replays
                if graphed:
                    self.assertEqual(replays, STEPS, f"{name}: steps run by a graph")
                    self.assertTrue(captures >= 1, f"{name}: the run replayed a graph of the previous selection")
                else:
                    self.assertEqual(after, state, f"{name}: an eager run moved a decode-graph counter")
                swapped.append((name, values, scores, captures, replays))
            refused = oracle.call_status("cere_ktest_model_set_selection", handle,
                                         selection(**dict(SWAPS[-1][1], k2_weights=CERE_LEVER_K2_WEIGHTS_FP8)))
            self.assertEqual(refused, oracle.CERE_ERR_UNSUPPORTED, "a k2_weights swap is refused")
            self.assertEqual(decode_graph(handle, -1), after, "the refused swap changed the graphs")
        finally:
            oracle.release_model(handle)
        fresh = {}
        for name, values in dict(SWAPS).items():
            ref = oracle.create_model(CONTEXT_K6, max_sessions=2, model_levers=oracle.levers(**values))
            try:
                fresh[name] = decode_rows(ref, prompts)
            finally:
                oracle.release_model(ref)
        for i, (name, values, scores, captures, replays) in enumerate(swapped):
            case = f"GraphHotSwap/{i}-{name}"
            levers = ",".join(f"{k}={v}" for k, v in sorted(values.items())) or "default"
            oracle.report_row(ROW, case, rows=2, steps=STEPS, levers=levers, captures=captures,
                              replays=replays, swapped_sha256=hashlib.sha256(scores.numpy().tobytes()).hexdigest(),
                              fresh_sha256=hashlib.sha256(fresh[name].numpy().tobytes()).hexdigest())
            self.assertTrue(bool((scores[..., 0] >= 0).all()), f"{case}: a score row is unwritten")
            oracle.check_bit_exact(self, ROW, case, scores, fresh[name])


if __name__ == "__main__":
    unittest.main()
