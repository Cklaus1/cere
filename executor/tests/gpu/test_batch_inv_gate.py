"""BC0 (tasks/batch-concurrency-spec.md BC-D1 (a)): the batch invariance gate on the real model, suite `batch_inv`.

Every request of `batch_inv_plan.requests(rows)` runs alone (the solo reference, M = 1) and in the seeded
compositions of `batch_inv_plan.schedules` through the real rounds (`cere_ktest_compose`: cere_executor_submit /
cere_executor_poll, one run_round per poll): sessions opened in a random order (slots differ), joins mid-decode, the
long prompt joining late (its prefill chunks interleave with decode rounds), a prefix-cache hit at 4,096 behind a
namespaced primer, plain rows beside drafting rows, greedy and seeded-sampled rows, prompts past 2,048 positions (K9),
and decode / spec graphs switched between polls. The last seed's compositions run under the shipped decode share
(S1 G4, tasks/beat-sglang-spec.md: native-g0-spec's `decode_share_pct` 200, `cere_ktest_decode_share`), so waiting
prompts prefill back to back while decode rows wait and a lone pending prompt's prefill rounds alternate with runs of
decode rounds; the other seeds keep the share off (one decode round per chunk). A request's committed token ids,
`cere_token_score` rows and logprobs rows must equal its solo run bit for bit, and so must its finish and
cached_tokens.

Profiles (the K8 model `/model` with the served draft map, so not -v52's artifact: the gate is about the code path):
* `BatchInvProd`: -v52's levers (k2_gemm small_m, decode_step spec_graph, verify_attention batched), 4 draft tokens,
  32 prefix state slots with end states and alternative edges on, max_sessions 4 (-v52's rows): M <= 4.
* `BatchInvDev`: the same with max_sessions 16: M <= 16.

Cells (batch_inv_plan.CELLS), every one gated: `greedy`, `sampled_plain` and `sampled_drafting` (F11, report-only until
BC1; BC-D4 prices every draft pick with the model's fixed `round_costs`). Also gated: every sampled drafting request
repeats its own solo run after the other traffic (`solo repeat`; the deleted wall-clock EMA `spec_costs` was model-wide).

Coverage self-checks (gated, so the gate cannot pass vacuously): some decode round held `rows` rows, some held plain
and drafting rows together, some prefill chunk ran beside decode-ready rows, some run under the share showed a round
the share off never runs (a prefill chunk right after another beside decode-ready rows, or a decode round right after
another while a prompt was pending: which one depends on the round walls), and the hit request's solo cached_tokens is
4,096.

Mutants (`BatchInvProd.test_mutants`, BC0 acceptance: >= 2 planted mutants turn the gate RED; every candidate's status
is a `mutant` row, and a bit-equal one is replaced by the next): each runs the greedy dense / sparse requests batched
(joins, graphs toggled) and compares them with the solo runs.
* M1 `k2_cublaslt`: k2_gemm = cublaslt (cere_ktest_model_set_selection) for the solo and the batched runs (its
  per-bucket algorithms by m, k2_gemm.h); also `BatchInvDev.test_mutant_k2_cublaslt_m16` at M <= 16, report-only.
* M2 `prefill_chunk`: the batched run's prefill chunk 1,024 rows (the solo runs keep the executor's policy): a
  prompt's chunk partition changes its bits (D8 (b)).
* M1 replacement candidates, batched run only: `verify_per_row` (verify attention per row, the solo runs' batched;
  bit-equal by design) and `prefill_p4` (P4 tensor-core prefill attention for calls of >= 17 rows, the solo runs'
  CUDA-core K6/K9).

Baselines (BC0 (c)): `BatchInvDev.test_graph_churn` (graph recaptures per 1,000 rounds at C = 16, BC-D7) and
`PlanLayoutV52` (the -v52 plan per R 4 / 8 / 12 / 16, host arithmetic).

    int cere_ktest_compose(uint64_t model, uint64_t requests, const uint64_t *sessions /*host [N] K15 handles*/,
                           const uint32_t *prompts, const uint64_t *prompt_lens, const uint64_t *max_new_tokens,
                           const uint32_t *flags, const cere_sampling_params *sampling, const uint32_t *top_logprobs,
                           const uint64_t *join, uint64_t poll_seed, const int8_t *graph_modes,
                           uint64_t graph_mode_count, uint64_t prefill_chunk_tokens, uint64_t max_polls,
                           uint32_t *out_tokens, cere_token_score *out_scores, cere_token_logprobs *out_logprobs,
                           uint64_t out_stride, uint64_t *out_results /*[6 N]*/, uint32_t *out_rounds /*[4 max_polls]*/,
                           uint64_t *out_polls /*[1]*/, cudaStream_t stream);
    int cere_ktest_model_set_selection(uint64_t model, const cere_slot_selection *selection, cudaStream_t stream);
    int cere_ktest_prefix_alt_edges(uint64_t model, uint32_t on, cudaStream_t stream);
    int cere_ktest_decode_share(uint64_t model, uint32_t share_pct, cudaStream_t stream);
    int cere_ktest_plan_layout_slots(int64_t max_sessions, int64_t max_context_tokens, int64_t kv_pool_tokens,
                                     int64_t drafts, int64_t prefix_state_slots, uint64_t *out /*host [11]*/,
                                     cudaStream_t stream);
"""

from __future__ import annotations

import time
import unittest

import torch

import batch_inv_plan as bp
import oracle
from test_model_k16_prefix import NS_A, close, flush, session_open, set_namespace, stats, synthetic_ids
from test_model_k8_forward import TOKEN_LOGPROBS_BYTES, sampling_bytes

ROW = "BC0"
DRAFT_MAP = "/draft_token_map.u32le"  # native_dev.py DRAFT_MAP_MOUNT
PREFIX_SLOTS = 32  # -v52 prefix_state_slots
PRIMER_RECORD = 90_000
HIT_TAIL_RECORD = 95_000
MAX_POLLS = 8192
MUTANT_CHUNK = 1024
K2_CUBLASLT = 0  # CERE_LEVER_K2_CUBLASLT
V52_LEVERS = {"k2_gemm": 1, "decode_step": 2, "verify_attention": 1}  # small_m, spec_graph, batched
MUTANT_KINDS = ("dense", "sparse")  # the mutants' requests: greedy, no hit, no late long prompt
CAPTURE_MEDIAN_MS = 9.1  # F13: the median graph capture
DECODE_SHARE_PCT = 200  # S1 G4: configs/runtime/native-g0-spec.toml decode_share_pct, on the last seed's compositions


def levers_tensor(**values: int) -> torch.Tensor:
    return torch.frombuffer(bytearray(bytes(oracle.levers(**values))), dtype=torch.uint8)


def graph_counters(model: int) -> tuple[list[int], list[int]]:
    """cere_ktest_decode_graph's [4] and cere_ktest_spec_graph's [7] counters, the graphs left as they are."""
    decode = torch.zeros(4, dtype=torch.int64)
    spec = torch.zeros(7, dtype=torch.int64)
    oracle.call("cere_ktest_decode_graph", model, -1, decode)
    oracle.call("cere_ktest_spec_graph", model, -1, spec)
    return [int(v) for v in decode.tolist()], [int(v) for v in spec.tolist()]


class Profile(unittest.TestCase):
    rows = 0

    @classmethod
    def setUpClass(cls):
        with open(DRAFT_MAP, "rb") as f:
            draft_map = torch.frombuffer(bytearray(f.read()), dtype=torch.int32)
        cls.levers = oracle.levers(**V52_LEVERS)
        cls.model = oracle.create_model(bp.CONTEXT, max_sessions=cls.rows, prefix_state_slots=PREFIX_SLOTS,
                                        draft_token_map=draft_map, model_levers=cls.levers)
        oracle.call("cere_ktest_prefix_end_state", cls.model, 1)
        oracle.call("cere_ktest_prefix_alt_edges", cls.model, 1)
        cls.vocab = int(oracle.text_config()["vocab_size"])
        cls.reqs = bp.requests(cls.rows)
        cls.primer = list(synthetic_ids(bp.PRIMER_TOKENS, PRIMER_RECORD))
        cls.prompts = [cls.prompt_of(r) for r in cls.reqs]
        cls.solo = {}

    @classmethod
    def tearDownClass(cls):
        oracle.release_model(cls.model)

    @classmethod
    def prompt_of(cls, r: bp.Request) -> list[int]:
        if r.kind == "hit":
            # D15⁗: a drafting state at h is keyed with the next id tokens[h] (its MTP row took it), so the hit
            # prompt shares the primer's tokens[0, h] and diverges after.
            tail = r.prompt_tokens - bp.HIT_PREFIX - 1
            return cls.primer[:bp.HIT_PREFIX + 1] + list(synthetic_ids(tail + 200, HIT_TAIL_RECORD))[-tail:]
        return list(synthetic_ids(r.prompt_tokens, r.record))

    # -- the driver ----------------------------------------------------------------------------------------------

    def compose(self, members, open_order, joins, poll_seed=1, graph_modes=(), chunk=0):
        """One composition after a flushed prefix tree (and the primer, when the hit request is a member): per member
        index an Outcome, and the predicted rounds."""
        flush(self.model)
        if any(self.reqs[i].kind == "hit" for i in members):
            self.prime()
        handles = {}
        try:
            for i in open_order:
                handles[i] = session_open(self.model, bp.CONTEXT)
                set_namespace(handles[i], NS_A)
            return self._compose(members, [handles[i] for i in members], joins, poll_seed, graph_modes, chunk)
        finally:
            for h in handles.values():
                close(h)

    def prime(self):
        h = session_open(self.model, bp.CONTEXT)
        try:
            set_namespace(h, NS_A)
            primer = bp.Request("primer", "dense", bp.PRIMER_TOKENS, bp.PRIMER_MAX_NEW, False, True, PRIMER_RECORD)
            self._compose_raw([primer], [self.primer], [h], [0], 1, (), 0)
        finally:
            close(h)

    def _compose(self, members, handles, joins, poll_seed, graph_modes, chunk):
        out, rounds = self._compose_raw([self.reqs[i] for i in members], [self.prompts[i] for i in members],
                                        handles, joins, poll_seed, graph_modes, chunk)
        return dict(zip(members, out)), rounds

    def _compose_raw(self, reqs, prompts, handles, joins, poll_seed, graph_modes, chunk):
        n = len(reqs)
        stride = max(r.max_new for r in reqs)
        specs = [(bp.SAMPLING_TEMPERATURE, bp.SAMPLING_TOP_P, 0, 0.0, 0.0, 1.0, bp.SAMPLING_SEED + r.record, True)
                 if r.sampled else None for r in reqs]
        tokens = torch.full((n, stride), -1, dtype=torch.int32)
        scores = torch.full((n, stride, 4), -1, dtype=torch.int32)
        logprobs = torch.full((n, stride, TOKEN_LOGPROBS_BYTES // 4), -1, dtype=torch.int32)
        results = torch.zeros(6 * n, dtype=torch.int64)
        rounds = torch.zeros(4 * MAX_POLLS, dtype=torch.int32)
        polls = torch.zeros(1, dtype=torch.int64)
        modes = torch.tensor(list(graph_modes) or [-1], dtype=torch.int8)
        oracle.call("cere_ktest_compose", self.model, n, torch.tensor(handles, dtype=torch.int64),
                    torch.tensor([t for p in prompts for t in p], dtype=torch.int32),
                    torch.tensor([len(p) for p in prompts], dtype=torch.int64),
                    torch.tensor([r.max_new for r in reqs], dtype=torch.int64),
                    torch.tensor([r.flags for r in reqs], dtype=torch.int32), sampling_bytes(specs, self.vocab),
                    torch.tensor([bp.TOP_LOGPROBS if r.sampled else 0 for r in reqs], dtype=torch.int32),
                    torch.tensor(list(joins), dtype=torch.int64), poll_seed, modes, len(graph_modes), chunk,
                    MAX_POLLS, tokens, scores, logprobs, stride, results, rounds, polls)
        out = []
        for i, r in enumerate(reqs):
            committed, finish, cached = (int(v) for v in results[6 * i:6 * i + 3].tolist())
            if not 0 < committed <= r.max_new:
                raise oracle.OracleError(f"{r.name}: committed {committed} outside (0, {r.max_new}]")
            out.append(bp.Outcome(committed, finish, cached, tokens[i, :committed].numpy().tobytes(),
                                  scores[i, :committed].numpy().tobytes(),
                                  logprobs[i, :committed].numpy().tobytes() if r.sampled else b""))
        used = int(polls.item())
        return out, [tuple(int(v) for v in rounds[4 * p:4 * p + 4].tolist()) for p in range(used)]

    def solo_run(self, i: int) -> bp.Outcome:
        return self.compose([i], [i], [0])[0][i]

    def reference(self, i: int) -> bp.Outcome:
        if i not in self.solo:
            self.solo[i] = self.solo_run(i)
        return self.solo[i]

    def compare_run(self, run: bp.Run, label: str, chunk: int = 0):
        got, rounds = self.compose(run.members, run.open_order, run.joins, run.poll_seed, run.graph_modes, chunk)
        compared = []
        for i in run.members:
            diff = bp.differences(self.reference(i), got[i])
            r = self.reqs[i]
            compared.append((r.cell, f"{r.name}@{label}", diff))
            if diff:
                oracle.report_row(ROW, f"{self.rows}r {label} {r.name}", phase="diff", cell=r.cell,
                                  differences=bp.field(diff))
        return compared, bp.coverage(rounds)

    # -- mutants -------------------------------------------------------------------------------------------------

    def mutant_members(self) -> list[int]:
        return [i for i, r in enumerate(self.reqs) if r.cell == "greedy" and r.kind in MUTANT_KINDS][:self.rows]

    def mutant(self, name: str, chunk: int = 0, levers: dict | None = None, solo: dict | None = None) -> str:
        """One batched run of mutant_members (joins every third poll, graphs toggled) under `chunk` / `levers` (the
        batched run only), compared with `solo` (default: the references); its status row; RED or GREEN."""
        members = self.mutant_members()
        solo = solo if solo is not None else {i: self.reference(i) for i in members}
        joins = tuple(0 if k % 2 == 0 else 3 * k for k in range(len(members)))
        if levers is not None:
            oracle.call("cere_ktest_model_set_selection", self.model, levers_tensor(**levers))
        try:
            got, _ = self.compose(members, members[::-1], joins, 7, (3, 0, 1, 2, 3, 3, -1), chunk)
        finally:
            if levers is not None:
                oracle.call("cere_ktest_model_set_selection", self.model, levers_tensor(**V52_LEVERS))
        differing = [self.reqs[i].name for i in members if bp.differences(solo[i], got[i])]
        status = "RED" if differing else "GREEN"
        oracle.report_row(ROW, f"{self.rows}r mutant {name}", phase="mutant", status=status, compared=len(members),
                          differing=len(differing), requests=bp.field(differing))
        return status

    def mutant_k2_cublaslt(self) -> str:
        """M1: k2_gemm = cublaslt for the solo and the batched runs (its per-bucket algorithms by m)."""
        values = dict(V52_LEVERS, k2_gemm=K2_CUBLASLT)
        oracle.call("cere_ktest_model_set_selection", self.model, levers_tensor(**values))
        try:
            solo = {i: self.solo_run(i) for i in self.mutant_members()}
        finally:
            oracle.call("cere_ktest_model_set_selection", self.model, levers_tensor(**V52_LEVERS))
        return self.mutant("k2_cublaslt", levers=values, solo=solo)

    # -- the gate ------------------------------------------------------------------------------------------------

    def run_gate(self):
        hit = next(i for i, r in enumerate(self.reqs) if r.kind == "hit")
        before = stats(self.model)
        cached = self.reference(hit).cached
        after = stats(self.model)
        oracle.report_row(ROW, f"{self.rows}r hit solo prefix", phase="diag", hits=after[0] - before[0],
                          captures=after[1] - before[1], inserts=after[3] - before[3],
                          capture_skipped=after[2] - before[2], cached_tokens=cached)
        oracle.check_count(self, ROW, f"{self.rows}r hit solo cached_tokens", cached, expected=bp.HIT_PREFIX)
        drafting_sampled = [i for i, r in enumerate(self.reqs) if r.cell == "sampled_drafting"]
        first = {i: self.reference(i) for i in drafting_sampled}
        compared, cover, share_runs = [], [], 0
        for seed in bp.SEEDS:
            share = DECODE_SHARE_PCT if seed == bp.SEEDS[-1] else 0
            oracle.call("cere_ktest_decode_share", self.model, share)
            try:
                for k, run in enumerate(bp.schedules(self.reqs, self.rows, seed)):
                    c, cov = self.compare_run(run, f"s{seed}r{k}" + ("share" if share else ""))
                    compared += c
                    cover.append(cov)
                    share_runs += bool(share) and cov["prefill_run_beside_decode"] + cov["decode_run_beside_prompt"] > 0
                    oracle.report_row(ROW, f"{self.rows}r s{seed}r{k} coverage", phase="coverage", decode_share=share,
                                      **cov)
            finally:
                oracle.call("cere_ktest_decode_share", self.model, 0)
        repeats = sum(bool(bp.differences(first[i], self.solo_run(i))) for i in drafting_sampled)
        oracle.report_row(ROW, f"{self.rows}r sampled_drafting solo repeat", phase="f11", requests=len(first),
                          differing=repeats)
        verdict = bp.verdict(compared)
        for cell, v in verdict.items():
            oracle.report_row(ROW, f"{self.rows}r {cell}", phase="verdict", status=v["status"],
                              compared=v["compared"], differing=len(v["differing"]),
                              requests=bp.field(v["differing"]))
        oracle.check_count(self, ROW, f"{self.rows}r max decode rows", max(c["max_rows"] for c in cover),
                           expected=self.rows)
        self.assertTrue(sum(c["mixed_rounds"] for c in cover) > 0, "no decode round mixed plain and drafting rows")
        self.assertTrue(sum(c["prefill_beside_decode"] for c in cover) > 0, "no prefill chunk beside decode rows")
        self.assertTrue(share_runs > 0, "no run under the share ran a round the share off never runs")
        oracle.check_count(self, ROW, f"{self.rows}r sampled_drafting solo repeat differing", repeats, expected=0)
        for cell, v in verdict.items():
            oracle.check_count(self, ROW, f"{self.rows}r {cell} differing", len(v["differing"]), expected=0)


class BatchInvProd(Profile):
    """The prod profile (-v52's levers, 4 rows): M <= 4, and the planted mutants."""
    rows = 4

    def test_invariance(self):
        self.run_gate()

    def test_mutants(self):
        """BC0 acceptance: at least two planted mutants turn the gate RED (each candidate's status is a row; a
        bit-equal candidate is reported GREEN and the others replace it)."""
        red = [name for name, status in (("k2_cublaslt", self.mutant_k2_cublaslt()),
                                         ("prefill_chunk", self.mutant("prefill_chunk", chunk=MUTANT_CHUNK)),
                                         ("verify_per_row", self.mutant("verify_per_row",
                                                                        levers=dict(V52_LEVERS, verify_attention=0))),
                                         ("prefill_p4", self.mutant("prefill_p4",
                                                                    levers=dict(V52_LEVERS, prefill_attention=1))))
               if status == "RED"]
        oracle.report_row(ROW, f"{self.rows}r mutants red", phase="mutant", red=len(red), names=bp.field(red))
        self.assertTrue(len(red) >= 2, f"mutants RED: {red}")


V52_CONTEXT = 262_144  # configs/runtime/native-g0-spec-v52.toml max_context_tokens, kv_pool_tokens, prefix slots
V52_POOL = 524_288
V52_DRAFTS = 4
PLAN_ROWS = (4, 8, 12, 16)


class PlanLayoutV52(unittest.TestCase):
    """BC0 (c): the -v52 plan's device arena per R (cere_ktest_plan_layout_slots, host arithmetic, prod's pool and
    slots), for BC2's memory fit (Q2): rows `BC0 plan R<r>` with device_mib and its delta to R 4."""

    def test_plan_layout_v52(self):
        keys = ("device_bytes", "pinned_bytes", "pool_end", "undo_offset", "undo_bytes", "workspace_rows",
                "spec_sampler_offset", "spec_sampler_bytes", "verify_rows", "state_bytes", "state_slots")
        base = None
        for rows in PLAN_ROWS:
            out = torch.zeros(len(keys), dtype=torch.int64)
            oracle.call("cere_ktest_plan_layout_slots", rows, V52_CONTEXT, V52_POOL, V52_DRAFTS, PREFIX_SLOTS, out)
            got = dict(zip(keys, (int(v) for v in out.tolist())))
            self.assertEqual(got["state_slots"], PREFIX_SLOTS + rows)
            base = got["device_bytes"] if base is None else base
            oracle.report_row(ROW, f"plan R{rows}", phase="plan", device_mib=got["device_bytes"] / 2**20,
                              delta_mib=(got["device_bytes"] - base) / 2**20, pinned_mib=got["pinned_bytes"] / 2**20,
                              state_mib=got["state_bytes"] * got["state_slots"] / 2**20,
                              undo_mib=got["undo_bytes"] / 2**20, sampler_mib=got["spec_sampler_bytes"] / 2**20,
                              verify_rows=got["verify_rows"], workspace_rows=got["workspace_rows"])


class BatchInvDev(Profile):
    """The dev profile (-v52's levers, 16 rows): M <= 16."""
    rows = 16

    def test_invariance(self):
        self.run_gate()

    def test_mutant_k2_cublaslt_m16(self):
        """M1 at M <= 16 (report-only: buckets 5-8 and 9-16 pin their own cuBLASLt algorithms)."""
        self.mutant_k2_cublaslt()

    def test_graph_churn(self):
        """BC0 (c), BC-D7: graph recaptures per 1,000 rounds at C = 16 under joins and leaves (graphs left as the
        levers set them). Recaptures = graphs instantiated + executables updated in place, per kind (decode, verify,
        draft); `round_ms` is the run's wall time per round, so BC-D7's bar (recaptures x the 9.1 ms median capture
        > 2 % of round time) is read off the row."""
        run = bp.schedules(self.reqs, self.rows, bp.SEEDS[0])[0]
        self.compose(run.members, run.open_order, run.joins, run.poll_seed)  # warm: first captures per row count
        decode0, spec0 = graph_counters(self.model)
        start = time.perf_counter()
        _, rounds = self.compose(run.members, run.open_order, run.joins, run.poll_seed)
        wall_ms = (time.perf_counter() - start) * 1e3
        decode1, spec1 = graph_counters(self.model)
        cov = bp.coverage(rounds)
        ran = max(cov["decode_rounds"] + cov["prefill_rounds"], 1)  # S1 G3: a poll may report without a round
        per = 1000 / ran
        recaptures = {"decode": (decode1[1] - decode0[1]) + (decode1[2] - decode0[2]),
                      "verify": (spec1[1] - spec0[1]) + (spec1[2] - spec0[2]),
                      "draft": (spec1[4] - spec0[4]) + (spec1[5] - spec0[5])}
        oracle.report_row(ROW, "graph churn C16", phase="churn", polls=len(rounds), rounds=ran,
                          decode_rounds=cov["decode_rounds"], max_rows=cov["max_rows"], round_ms=wall_ms / ran,
                          decode_graph_steps=decode1[3] - decode0[3], verify_graph_calls=spec1[3] - spec0[3],
                          draft_graph_runs=spec1[6] - spec0[6],
                          **{f"{k}_recaptures": v for k, v in recaptures.items()},
                          **{f"{k}_per_1000_rounds": v * per for k, v in recaptures.items()},
                          capture_share=sum(recaptures.values()) * CAPTURE_MEDIAN_MS / wall_ms)
        self.assertTrue(cov["decode_rounds"] > 0, "no decode round")


if __name__ == "__main__":
    unittest.main()
