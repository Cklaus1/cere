"""SelfTune ST1 launch-sequence equality (tasks/selftune-spec.md ST1, E3): the kernel nodes of every CUDA graph the
forward captures, and the eager launches of prefill (ST1d) and of the MTP drafts (ST1e), recorded per case so two
libraries' launch sequences can be compared without a profiler.

One real A4 model with MTP drafts (`test_model_k8_forward.create_model`, CONTEXT tokens, SESSIONS sessions, decode and
spec graphs switched on by their ktests) runs the cases below in this fixed order, in one test method (a graph
already captured for a key is replayed, not instantiated again, so the order is part of each case's record):
* `decode/feed`: one session teacher-forced through `cere_ktest_session_feed` (PREFILL, then STEPS single-row
  DECODE forwards: decode graph 0, `run_layers`).
* `decode/dense/r<R>` (R = 1 .. SESSIONS, BC2's row cap): `cere_ktest_forward_decode_rows` of R sessions whose
  STEPS decodes stay below 2,048 (K6 attention over the FP8 cache; decode graph R, `run_rows`).
* `decode/qsa/r<R>` (R = 1 .. SESSIONS): the same with prompts whose decodes cross position 2,048 (the K9 QSA path;
  the dense bit of each row's key flips mid-run, so the graph is captured or updated again).
* `verify/m<m>` (m = 1 .. SPEC_DRAFT_TOKENS, cere_ktest_session_verify's limit of 3 drafts): one prefilled session's
  chain of m rows (verify graph m), through CudaDevice::verify_rows.
* `verify/chain/m<m>` (m = 2 .. CHAIN_ROWS, O279 D-b7): one drafting session's greedy
  `cere_ktest_session_speculate_chain` call with m - 1 caller drafts (speculate_rows: chains past 3 drafts take the
  model's undo pool), up to TreeSpec's longest lookup chain of 8 drafts; the rest of a long chain's verify graphs
  are the R-row speculate calls of `draft/r<R>`.
* `draft/r<R>` (R = 1 .. DRAFT_ROWS): R drafting sessions, ROUNDS greedy `cere_ktest_session_speculate_mtp` rounds
  with max_commit cycling 1 .. SPEC_DRAFT_TOKENS (MTP draft graphs, and the verify graphs of the R-row calls).
* `draft/chain/<path>/m<m>` (ST1e, m = 2 .. CHAIN_ROWS): one drafting session's speculate_chain call whose m - 1
  caller drafts are the prompt's greedy continuation (taken from a plain session), so all of them are accepted and
  the MTP draft-extend runs over m committed rows (1 + TreeSpec's 8 drafts at most), then its layout::kDraftSteps
  (2) decode steps. Recorded with the eager launches (mode 3). `graph`: spec graphs on, a prompt of CHAIN_PROMPT
  tokens (draft_graph, the captured draft of m rows); `qsa`: spec graphs on, DRAFT_QSA_PROMPT tokens, past
  2,048 (spec_draft_graph_fits refuses, so the eager mtp::forward loop of CudaDevice::draft); `eager`: spec graphs
  off, CHAIN_PROMPT tokens (the same eager loop at dense positions).
* `draft/rows/<path>/r<R>` (ST1e, R in DRAFT_BATCH: 1, 4 and BC2's 16): R drafting sessions, ROUNDS
  speculate_mtp rounds with max_commit cycling 1 .. SPEC_DRAFT_TOKENS, recorded with the eager launches: R = 1 is
  CudaDevice::draft; R >= 2 is mtp::draft_rows (P11R-21: one MTP forward per pass over every session's rows) while
  the rounds' committed rows fit its 16 extend rows (R = 4 at any commit, R = 16 at commit 1), else draft per
  session. Paths as above: `dense` (spec graphs on, DENSE_PROMPTS), `qsa` (spec graphs on, past 2,048) and `eager`
  (spec graphs off, DENSE_PROMPTS).

Prefill runs eagerly, so its cases record every eager launch too (recorder mode 3). A second model (`PrefillLaunches`:
PREFILL_CONTEXT tokens, drafting sessions, created with a prefill slice of SLICE_MS armed by
`cere_ktest_next_prefill_slice`, so it keeps the MixedPD park region as production's `prefill_slice_ms` gives it)
runs these cases through `cere_ktest_session_prefill_sliced` (one PREFILL step as the scheduler's solo_step makes
it, chunk by chunk, with the segment planned from a fixed cost per layer):
* `prefill/dense/short` (under 2,048 tokens, one partial chunk), `prefill/qsa/short` (past 2,048: the K9 sparse
  path), `prefill/qsa/chunk` (exactly one 4,096-token chunk) and `prefill/qsa/multi` (three chunks), each whole (no
  slice).
* `prefill/sliced/first`: a call sliced at SLICE_MS with SPLIT_US per layer pauses after its first segment
  (model::segment::segment_layers: layer 12 of 48); `prefill/sliced/decode`: another session's DECODE while the call
  is parked (the decode round between segments); `prefill/sliced/resume`: the call resumes from that non-zero layer
  and pauses again; `prefill/sliced/rest`: its remaining segments to the call's end.
* `prefill/sliced/whole`: the same budget at WHOLE_US per layer fits the whole call, which runs as one segment.
A third model (`ImagePrefillLaunches`, the vision model of `test_model_k15_snapshot`) records `prefill/image`: one
image prompt's PREFILL (vision encode and the image splice).

Per case, `cere_ktest_graph_launches` records every graph instantiation and in-place update (CUPTI driver
callbacks): each graph's nodes in node order (a kernel's symbol, grid, block, dynamic shared memory and programmatic
stream serialization attribute; copies and memsets by shape) and its edges with their dependency type (a PDL launch
is a programmatic edge); in a prefill case each eager kernel launch, copy, memset and graph launch as a record of its
own in issue order, in the same node format. No pointer, argument or token id is recorded, so equal forwards give
equal text in any process. The row (ST1L, one per case) carries the text's SHA-256, the SHA-256 prefix of each
record, the node, kernel, programmatic-edge and eager-launch counts, the decode / spec graph counter deltas and the
model's five arena sizes (`cere_ktest_model_arena`: workspace, derived, session, pool and staging bytes of its memory
plan, O279 D-b8); `scripts/native/launch_sequence.py` records a baseline fixture from a report and checks a later
report against it (the comparison of two libraries is that script's, never this suite's). This suite checks the
recorder itself: a case records graphs exactly when the graph counters moved (instantiations or updates), at least
one graph record per counted graph step, a prefill case at least one eager launch, and the record counts add up.

Native entry points (oracle.call appends the stream; 0 = success, otherwise a cere_status; "host" buffers are CPU
memory):

    int cere_ktest_graph_launches(int64_t mode, uint8_t *out, uint64_t capacity, uint64_t *counts,
                                  cudaStream_t);
        mode 1 clears the records and starts recording graphs (CERE_ERR_UNSUPPORTED without libcupti.so.13 or a
        driver entry point, CERE_ERR_BUSY when another CUPTI subscriber holds the process); mode 3 the same with the
        eager launches too; mode 0 stops (the records stay); mode 2 reads: counts (host uint64 [4]) = {records,
        instantiations, in-place updates, text bytes} and, when out is not null, the first min(capacity, text
        bytes) bytes of the records' text. A failed walk or eager record makes mode 2 CERE_ERR_INTERNAL; any other
        mode or a null counts in mode 2: CERE_ERR_ARGUMENT.
    int cere_ktest_model_arena(uint64_t model, uint64_t *out, cudaStream_t);
        out (host uint64 [5]) = the ktest model's {workspace_bytes, derived_bytes, session_bytes, pool_bytes,
        staging_bytes} (device/memory_plan.h ModelLayout); a null handle or out: CERE_ERR_ARGUMENT.
    int cere_ktest_next_prefill_slice(uint64_t slice_ms, cudaStream_t);
        arms the prefill slice (ms; 0 disarms) the next cere_ktest_model_create* sets on its executor before the
        model's create (cere_executor_set_prefill_slice); that create takes it. Above CERE_MAX_PREFILL_SLICE_MS:
        CERE_ERR_ARGUMENT.
    int cere_ktest_session_prefill_sliced(uint64_t session, const uint32_t *tokens, uint64_t n, uint64_t budget_ms,
                                          uint64_t ms_per_layer_us, uint64_t *out, cere_token_score *scores,
                                          cudaStream_t stream);
        one PREFILL step of the session's prompt tokens[0, n) (host) from its fed tokens, through CudaDevice::step
        with a StepSlice of budget_ms (0: none), the model's park cost fixed at ms_per_layer_us / 1,000 ms per layer
        first; out (host uint64 [4]) = {tokens consumed, paused, the paused call's next layer, fed tokens}; the step
        that ends the prompt writes scores[0] (host [1]).
"""

from __future__ import annotations

import contextlib
import hashlib
import unittest

import torch

import oracle
import test_graph_decode as graph_decode
import test_model_k8_forward as k8
import test_model_k15_snapshot as k15

ROW = "ST1L"
CONTEXT = 4_096  # above k6::kMaxContext: positions from 2,048 run the K9 QSA path
SESSIONS = 16  # BC2's decode row cap
STEPS = graph_decode.STEPS  # decodes per decode case (forward_decode_rows' leave round)
SEED = 20261111
DENSE_PROMPTS = tuple(40 + 11 * i for i in range(SESSIONS))  # decodes end below 2,048
QSA_PROMPTS = tuple(2_030 + i for i in range(SESSIONS))  # decodes cross 2,048
FEED_PROMPT = 61
VERIFY_PROMPT = 61
CHAIN_ROWS = 9  # TreeSpec's longest lookup chain: 8 drafts (DraftPolicy::kMaxTailDrafts, SPEC_TAIL_MAX)
CHAIN_PROMPT = 83
DRAFT_ROWS = 4
DRAFT_PROMPTS = (37, 211, 512, 1_029)
ROUNDS = 8
DRAFT_BATCH = (1, 4, 16)  # draft rows: one session, draft_rows at any commit, BC2's row cap
DRAFT_QSA_PROMPT = 2_100  # every draft row past 2,048: no draft graph fits
ARENA = ("workspace_bytes", "derived_bytes", "session_bytes", "pool_bytes", "staging_bytes")
PREFILL_CONTEXT = 12_288  # three 4,096-token chunks fit
PREFILL_SESSIONS = 2
CHUNK = 4_096  # CudaDevice's prefill chunk (k5::kChunkTokens)
SHORT_PROMPTS = (("dense", 1_000), ("qsa", 3_000))  # under one chunk, below / past 2,048
CHUNK_PROMPT = CHUNK
MULTI_PROMPT = 9_000  # three chunks
SLICE_MS = 200  # production's prefill_slice_ms (-v58)
SPLIT_US = 12_500  # 12.5 ms per layer: segments of 12 layers (48 + 2 tail layers at <= 16 per 200 ms)
WHOLE_US = 2_000  # 2 ms per layer: the whole call (50 x 2 ms) fits 200 ms
SPLIT_LAYERS = (12, 24, 36)  # the paused call's next layer after each segment at SPLIT_US
SLICED_PROMPT = 3_000
DECODE_PROMPT = 61


def graph_launches(mode: int) -> tuple:
    """Start (1) or stop (0) the recorder, or read it (2): (counts, text)."""
    counts = torch.zeros(4, dtype=torch.int64)
    if mode != 2:
        oracle.call("cere_ktest_graph_launches", mode, None, 0, counts)
        return (), ""
    oracle.call("cere_ktest_graph_launches", 2, None, 0, counts)
    size = int(counts[3])
    out = torch.zeros(max(size, 1), dtype=torch.uint8)
    oracle.call("cere_ktest_graph_launches", 2, out, size, counts)
    return tuple(int(v) for v in counts.tolist()), bytes(out[:size].tolist()).decode("ascii")


def next_prefill_slice(slice_ms: int) -> None:
    """Arms the prefill slice the next ktest model create sets on its executor."""
    oracle.call("cere_ktest_next_prefill_slice", slice_ms)


def prefill_sliced(session: int, ids: list[int], budget_ms: int, ms_per_layer_us: int) -> tuple:
    """One PREFILL step of `ids` from the session's fed tokens: (consumed, paused, parked layer, fed)."""
    out = torch.zeros(4, dtype=torch.int64)
    score = torch.full((1, 4), -1, dtype=torch.int32)
    oracle.call("cere_ktest_session_prefill_sliced", session, torch.tensor(ids, dtype=torch.int32), len(ids),
                budget_ms, ms_per_layer_us, out, score)
    return tuple(int(v) for v in out.tolist())


def prefill_whole(session: int, ids: list[int]) -> int:
    """`ids` through unsliced PREFILL steps (one per chunk); the number of steps."""
    steps, fed = 0, 0
    while fed < len(ids):
        consumed, paused, _, fed = prefill_sliced(session, ids, 0, 0)
        if consumed == 0 or paused:
            raise oracle.OracleError(f"an unsliced prefill step consumed {consumed} (paused {paused})")
        steps += 1
    return steps


def model_arena(handle: int) -> dict:
    """The model's five arena sizes (ARENA order)."""
    out = torch.zeros(len(ARENA), dtype=torch.int64)
    oracle.call("cere_ktest_model_arena", handle, out)
    return dict(zip(ARENA, (int(v) for v in out.tolist())))


def summary(text: str) -> dict:
    """The record-level shape of a recorder text: records, nodes, kernels, programmatic edges, record digests."""
    records, nodes, kernels, programmatic = [], 0, 0, 0
    for line in text.splitlines(keepends=True):
        if line.startswith("record "):
            records.append([line])
            nodes += int(line.split(" nodes=")[1].split()[0])
            continue
        records[-1].append(line)
        kernels += " kernel " in line
        programmatic += line.startswith("edge ") and " type=1 " in line
    digests = [hashlib.sha256("".join(r[1:]).encode()).hexdigest()[:12] for r in records]
    return {"records": len(records), "nodes": nodes, "kernels": kernels, "programmatic_edges": programmatic,
            "record_sha256": ",".join(digests)}


class _Recorded(unittest.TestCase):
    """The per-case recording of every class below (`handle`: the class's ktest model)."""

    handle = 0

    def counters(self) -> tuple:
        return graph_decode.decode_graph(self.handle, -1) + k8.spec_graph(self.handle, -1)

    def recorded(self, case: str, run, *, eager: bool = False) -> None:
        """Runs `run()` under the recorder (with the eager launches when `eager`) and reports the case's row; checks
        the recorder against the counters."""
        before = self.counters()
        graph_launches(3 if eager else 1)
        run()
        graph_launches(0)
        after = self.counters()
        (records, instantiates, updates, size), text = graph_launches(2)
        # decode (enabled, captures, updates, replays), spec (on, verify captures / updates / replays, draft
        # captures / updates / replays)
        delta = [a - b for a, b in zip(after, before)]
        steps_made = delta[1] + delta[2] + delta[5] + delta[6] + delta[8] + delta[9]
        shape = summary(text)
        launches = records - instantiates - updates
        oracle.report_row(ROW, case, launches_sha256=hashlib.sha256(text.encode()).hexdigest(), text_bytes=size,
                          instantiates=instantiates, updates=updates, launches=launches, decode_captures=delta[1],
                          decode_updates=delta[2], decode_replays=delta[3], verify_captures=delta[5],
                          verify_updates=delta[6], verify_replays=delta[7], draft_captures=delta[8],
                          draft_updates=delta[9], draft_replays=delta[10], **shape, **model_arena(self.handle))
        self.assertEqual(shape["records"], records, f"{case}: record headers in the text")
        self.assertEqual(launches, sum(" launch " in line or " graph_launch " in line
                                       for line in text.splitlines() if line.startswith("record ")),
                         f"{case}: eager records")
        self.assertEqual(launches > 0, eager, f"{case}: eager records iff recorded with them")
        graphs = instantiates + updates
        self.assertEqual(graphs > 0, steps_made > 0, f"{case}: graphs recorded iff a graph was made")
        self.assertTrue(graphs >= steps_made, f"{case}: fewer graph records than graph steps made")
        if records:
            self.assertTrue(shape["kernels"] > 0, f"{case}: records without kernel nodes")


class GraphLaunches(_Recorded):
    @classmethod
    def setUpClass(cls):
        cls.vocab = int(oracle.text_config()["vocab_size"])
        model = k8.create_model(CONTEXT, SESSIONS, drafts=k8.SPEC_DRAFT_TOKENS)
        cls.addClassCleanup(k8.release_model, model.handle)
        k8.upload_and_seal(model)
        cls.handle = model.handle
        cls.addClassCleanup(graph_launches, 0)

    def decode_feed(self) -> None:
        gen = torch.Generator().manual_seed(SEED)
        ids = k8.synthetic_ids(gen, self.vocab, FEED_PROMPT + STEPS)
        graph_decode.run_session(self.handle, CONTEXT, ids[:FEED_PROMPT], ids[FEED_PROMPT:])

    def verify_chains(self) -> None:
        gen = torch.Generator().manual_seed(SEED + 1)
        with k8.opened_session(self.handle) as session:
            k8.session_feed(session, k8.synthetic_ids(gen, self.vocab, VERIFY_PROMPT))
            for m in range(1, k8.SPEC_DRAFT_TOKENS + 1):
                chain = k8.synthetic_ids(gen, self.vocab, m)
                self.recorded(f"verify/m{m}", lambda: k8.session_verify(session, chain, 1))

    def verify_long_chains(self) -> None:
        gen = torch.Generator().manual_seed(SEED + 4)
        with k8.opened_session(self.handle, draft=True) as session:
            token = int(k8.session_feed(session, k8.synthetic_ids(gen, self.vocab, CHAIN_PROMPT))[0, 0])
            for m in range(2, CHAIN_ROWS + 1):
                drafts = k8.synthetic_ids(gen, self.vocab, m - 1)
                got = {}

                def run() -> None:
                    got["s"] = k8.session_speculate_chain([session], [token], [drafts], [m])

                self.recorded(f"verify/chain/m{m}", run)
                token = int(got["s"].scores[0, got["s"].committed[0] - 1, 0])

    def draft_rounds(self, rows: int) -> None:
        gen = torch.Generator().manual_seed(SEED + 2)
        with contextlib.ExitStack() as stack:
            sessions, tokens = [], []
            for r in range(rows):
                session = stack.enter_context(k8.opened_session(self.handle, draft=True))
                prefill = k8.session_feed(session, k8.synthetic_ids(gen, self.vocab, DRAFT_PROMPTS[r]))
                sessions.append(session)
                tokens.append(int(prefill[0, 0]))

            def rounds() -> None:
                for i in range(ROUNDS):
                    commit = i % k8.SPEC_DRAFT_TOKENS + 1
                    got = k8.session_speculate_mtp(sessions, tokens, [commit - 1] * rows, [commit] * rows)
                    for r in range(rows):
                        tokens[r] = int(got.scores[r, got.committed[r] - 1, 0])

            self.recorded(f"draft/r{rows}", rounds)

    def draft_chains(self, path: str, prompt_tokens: int) -> None:
        """`draft/chain/<path>/m<m>`: every chain's drafts are the greedy continuation, so the draft-extend runs over
        m committed rows."""
        gen = torch.Generator().manual_seed(SEED + 5)
        prompt = k8.synthetic_ids(gen, self.vocab, prompt_tokens)
        with k8.opened_session(self.handle) as plain:
            truth = [int(k8.session_feed(plain, prompt)[0, 0])]
            while len(truth) < CHAIN_ROWS:
                truth.append(int(k8.session_feed(plain, truth[-1:])[0, 0]))
        for m in range(2, CHAIN_ROWS + 1):
            with k8.opened_session(self.handle, draft=True) as session:
                k8.session_feed(session, prompt)
                got = {}

                def run() -> None:
                    got["s"] = k8.session_speculate_chain([session], truth[:1], [truth[1:m]], [m])

                self.recorded(f"draft/chain/{path}/m{m}", run, eager=True)
                self.assertEqual(got["s"].committed[0], m, f"draft/chain/{path}/m{m}: committed rows")

    def draft_batch(self, path: str, lengths: tuple) -> None:
        """`draft/rows/<path>/r<R>`: R drafting sessions' speculate_mtp rounds (`lengths`: their prompts). One
        unrecorded cycle of commits 1 .. SPEC_DRAFT_TOKENS runs first: whether a verify graph of R rows replays as
        captured or is updated in place depends on the sessions an earlier case left it holding, which varied
        between two runs of one library (draft/rows/dense/r1 after draft/chain/qsa/m9: 0 or 1 verify updates), so the
        recorded rounds start from graphs these sessions already ran."""
        for rows in DRAFT_BATCH:
            gen = torch.Generator().manual_seed(SEED + 6)
            with contextlib.ExitStack() as stack:
                sessions, tokens = [], []
                for r in range(rows):
                    session = stack.enter_context(k8.opened_session(self.handle, draft=True))
                    prefill = k8.session_feed(session, k8.synthetic_ids(gen, self.vocab, lengths[r]))
                    sessions.append(session)
                    tokens.append(int(prefill[0, 0]))

                def rounds(count: int = ROUNDS) -> None:
                    for i in range(count):
                        commit = i % k8.SPEC_DRAFT_TOKENS + 1
                        got = k8.session_speculate_mtp(sessions, tokens, [commit - 1] * rows, [commit] * rows)
                        for r in range(rows):
                            tokens[r] = int(got.scores[r, got.committed[r] - 1, 0])

                rounds(k8.SPEC_DRAFT_TOKENS)
                self.recorded(f"draft/rows/{path}/r{rows}", rounds, eager=True)

    def test_launch_sequences(self):
        self.addCleanup(graph_decode.decode_graph, self.handle, graph_decode.decode_graph(self.handle, -1).enabled)
        self.addCleanup(k8.spec_graph, self.handle, k8.spec_graph(self.handle, -1)[0])
        graph_decode.decode_graph(self.handle, 1)
        k8.spec_graph(self.handle, 1)
        self.recorded("decode/feed", self.decode_feed)
        for path, lengths in (("dense", DENSE_PROMPTS), ("qsa", QSA_PROMPTS)):
            gen = torch.Generator().manual_seed(SEED + 3)
            prompts = [k8.synthetic_ids(gen, self.vocab, n) for n in lengths]
            for rows in range(1, SESSIONS + 1):
                self.recorded(f"decode/{path}/r{rows}",
                              lambda: graph_decode.decode_rows(self.handle, prompts[:rows]))
        self.verify_chains()
        self.verify_long_chains()
        for rows in range(1, DRAFT_ROWS + 1):
            self.draft_rounds(rows)
        qsa_drafts = tuple(DRAFT_QSA_PROMPT + i for i in range(SESSIONS))
        self.draft_chains("graph", CHAIN_PROMPT)
        self.draft_chains("qsa", DRAFT_QSA_PROMPT)
        self.draft_batch("dense", DENSE_PROMPTS)
        self.draft_batch("qsa", qsa_drafts)
        k8.spec_graph(self.handle, 0)
        self.draft_chains("eager", CHAIN_PROMPT)
        self.draft_batch("eager", DENSE_PROMPTS)

    def test_refused_modes(self):
        """A mode other than 0 / 1 / 2 / 3, or a read without counts, is CERE_ERR_ARGUMENT; so is an arena read of
        a null handle, and a prefill slice above CERE_MAX_PREFILL_SLICE_MS."""
        counts = torch.zeros(4, dtype=torch.int64)
        for mode, buffer in ((4, counts), (-1, counts), (2, None)):
            with self.subTest(mode=mode):
                self.assertEqual(oracle.call_status("cere_ktest_graph_launches", mode, None, 0, buffer),
                                 graph_decode.CERE_ERR_ARGUMENT)
        self.assertEqual(oracle.call_status("cere_ktest_model_arena", 0, torch.zeros(5, dtype=torch.int64)),
                         graph_decode.CERE_ERR_ARGUMENT)
        self.assertTrue(all(v > 0 for k, v in model_arena(self.handle).items() if k != "staging_bytes"))
        self.assertEqual(oracle.call_status("cere_ktest_next_prefill_slice", 1 << 40), graph_decode.CERE_ERR_ARGUMENT)


class PrefillLaunches(_Recorded):
    @classmethod
    def setUpClass(cls):
        cls.vocab = int(oracle.text_config()["vocab_size"])
        next_prefill_slice(SLICE_MS)
        cls.addClassCleanup(next_prefill_slice, 0)
        model = k8.create_model(PREFILL_CONTEXT, PREFILL_SESSIONS, drafts=k8.SPEC_DRAFT_TOKENS)
        cls.addClassCleanup(k8.release_model, model.handle)
        k8.upload_and_seal(model)
        cls.handle = model.handle
        cls.addClassCleanup(graph_launches, 0)

    def whole(self, case: str, ids: list[int], steps: int) -> None:
        with k8.opened_session(self.handle, draft=True) as session:
            got = {}
            self.recorded(case, lambda: got.update(steps=prefill_whole(session, ids)), eager=True)
            self.assertEqual(got["steps"], steps, f"{case}: prefill steps")

    def sliced(self, gen: torch.Generator) -> None:
        ids = k8.synthetic_ids(gen, self.vocab, SLICED_PROMPT)
        decode_ids = k8.synthetic_ids(gen, self.vocab, DECODE_PROMPT + 1)
        with k8.opened_session(self.handle) as other, k8.opened_session(self.handle, draft=True) as session:
            k8.session_feed(other, decode_ids[:DECODE_PROMPT])
            got = {}
            self.recorded("prefill/sliced/first",
                          lambda: got.update(first=prefill_sliced(session, ids, SLICE_MS, SPLIT_US)), eager=True)
            self.assertEqual(got["first"], (0, 1, SPLIT_LAYERS[0], 0))
            self.recorded("prefill/sliced/decode", lambda: k8.session_feed(other, decode_ids[DECODE_PROMPT:]),
                          eager=True)
            self.recorded("prefill/sliced/resume",
                          lambda: got.update(resume=prefill_sliced(session, ids, SLICE_MS, SPLIT_US)), eager=True)
            self.assertEqual(got["resume"], (0, 1, SPLIT_LAYERS[1], 0))

            def rest() -> None:
                got["rest"] = [prefill_sliced(session, ids, SLICE_MS, SPLIT_US) for _ in SPLIT_LAYERS[1:]]

            self.recorded("prefill/sliced/rest", rest, eager=True)
            self.assertEqual(got["rest"], [(0, 1, SPLIT_LAYERS[2], 0), (SLICED_PROMPT, 0, 0, SLICED_PROMPT)])
        with k8.opened_session(self.handle, draft=True) as session:
            self.recorded("prefill/sliced/whole",
                          lambda: got.update(whole=prefill_sliced(session, ids, SLICE_MS, WHOLE_US)), eager=True)
            self.assertEqual(got["whole"], (SLICED_PROMPT, 0, 0, SLICED_PROMPT))

    def test_prefill_launch_sequences(self):
        self.addCleanup(graph_decode.decode_graph, self.handle, graph_decode.decode_graph(self.handle, -1).enabled)
        self.addCleanup(k8.spec_graph, self.handle, k8.spec_graph(self.handle, -1)[0])
        graph_decode.decode_graph(self.handle, 1)
        k8.spec_graph(self.handle, 1)
        gen = torch.Generator().manual_seed(SEED + 5)
        for path, n in SHORT_PROMPTS:
            self.whole(f"prefill/{path}/short", k8.synthetic_ids(gen, self.vocab, n), 1)
        self.whole("prefill/qsa/chunk", k8.synthetic_ids(gen, self.vocab, CHUNK_PROMPT), 1)
        self.whole("prefill/qsa/multi", k8.synthetic_ids(gen, self.vocab, MULTI_PROMPT), -(-MULTI_PROMPT // CHUNK))
        self.sliced(gen)


class ImagePrefillLaunches(_Recorded):
    @classmethod
    def setUpClass(cls):
        cls.ids, cls.offset, cls.pixels, _ = k15.image_prompt()
        cls.handle = oracle.create_model(k15.IMAGE_CONTEXT, max_sessions=1, vision=True)
        cls.addClassCleanup(oracle.release_model, cls.handle)
        cls.addClassCleanup(graph_launches, 0)

    def test_image_prefill_launch_sequence(self):
        session = k15.session_open(self.handle, k15.IMAGE_CONTEXT)
        self.addCleanup(k15.close, session)
        got = {}
        self.recorded("prefill/image",
                      lambda: got.update(scores=k15.prefill_images(session, self.ids, self.offset, self.pixels, 1)),
                      eager=True)
        self.assertEqual(got["scores"][0].shape[0], 1, "image prefill committed")


if __name__ == "__main__":
    unittest.main()
