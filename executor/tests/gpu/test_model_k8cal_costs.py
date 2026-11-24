"""K8 speculation cost calibration (S4 of the 2026-10-18 test audit): K8Speculation's report-only timing tests,
moved out of every per-merge `model_k8` class and run nightly or on demand on the small-M model only
(`native_dev.py --gpu {gpu} test --suite model_k8cal`, pooled; when: `native_dev.ON_DEMAND_SUITES`). Their rows are
the SpecCosts calibration input (executor/cuda/src/host/spec_calibrate.h and spec_lookup.h cite the "A3 window t" and
"A3D window t" rows) and gate nothing: a test fails only when a speculated score differs from the plain run's or a
window has no timed round, the bit-identity K8Speculation already gates per merge.

K8SpecCostsSmallM: K8Speculation's drafting model (test_model_k8_forward.py: `speculative_draft_tokens` =
SPEC_DRAFT_TOKENS, SPEC_SESSIONS sessions, SPEC_CONTEXT) created with the `k2_gemm` lever at `small_m`
(CERE_LEVER_K2_SMALL_M), K8SpeculationSmallM's model; "plain" is K8Speculation's teacher-forced plain run:
* `test_spec_window_costs` (tasks/m29a-spec.md A3; O95: the same windows `cere_model_seal` measures for the adaptive
  policy's tables, tasks/m23-spec.md SC-01, reported here as a cross-check): per verify window
  t = 1..4 and parity prompt, SPEC_COST_ROUNDS greedy rounds of one drafting session through
  `cere_ktest_session_speculate_mtp` with t - 1 MTP proposals, each round timed on the host: every committed score
  equals the plain run's; the median round ms (cold first round excluded) and the accept length per window (report
  rows "A3 window t").
* `test_spec_deep_window_costs` (DraftAccept, SpecCosts windows 5..9): per verify window t = 4..CHAIN_DRAFTS + 1 and
  parity prompt, SPEC_COST_ROUNDS greedy rounds of t - 1 caller drafts = the plain run's next tokens through
  `cere_ktest_session_speculate_chain` (every draft accepted), each round timed on the host: every committed score
  equals the plain run's; the median round ms per window (report rows "A3D window t").
* `test_p11_spec_cycle` (tasks/m23-spec.md P11R-18): the c1 speculation cycle on P11_CYCLE_IDS
  (`parity_v1` prompts), P11_CYCLE_TOKENS greedy tokens each, per drafts per chain n in P11_CYCLE_DRAFTS (the
  speculate_rows draft_count, max_commit n + 1, on this SPEC_DRAFT_TOKENS model) through
  `cere_ktest_session_speculate_mtp`: every committed score equals the plain run's. Reported per n: rounds, accept
  length and tok/s (committed / summed call wall); on the first prompt the GPU ms of a round, of a verify alone at m
  = n + 1 (`cere_ktest_session_verify`, which runs no MTP draft, on the true tokens, scores equal plain's), their
  difference (the MTP draft: P11_CYCLE_MTP_FORWARDS forwards whatever n) and the host/sync rest of a round's wall;
  plus a plain DECODE step's wall and GPU ms. GPU ms = the summed CUDA records (kernels, copies, sets) under
  torch.profiler (CUPTI); walls come from unprofiled calls.

Token IDs and text stay in memory; report rows carry counts, scalars and digests only.
"""
from __future__ import annotations

import contextlib
import functools
import statistics
import time
import unittest

import torch
from torch.profiler import ProfilerActivity, profile

import oracle
import test_model_k8_forward as k8
from test_model_k8_forward import (ROW, SPEC_CHAIN_MAX, SPEC_DECODE, SPEC_SESSIONS, PlainRun, opened_session,
                                   parity_prompts, plain_greedy, session_feed, session_speculate_chain,
                                   session_speculate_mtp, session_verify)

# P11 lever 2 (m23 design note P11R-18): test_p11_spec_cycle's prompts and sizes.
P11_CYCLE_IDS = ("p01", "p04", "p07", "p10", "p13", "p16", "p19", "p22")
P11_CYCLE_TOKENS = 128
P11_CYCLE_PLAIN_STEPS = 32
P11_CYCLE_VERIFY_ROUNDS = 8
P11_CYCLE_DRAFTS = (1, 2, 3)  # drafts per verify chain: speculative_draft_tokens 2, 3, 4
P11_CYCLE_MTP_FORWARDS = 3  # CudaDevice::draft: the draft-extend and 2 MTP decode steps, whatever the draft count
SPEC_COST_ROUNDS = 48  # m29a design note A3: timed speculation rounds per window and prompt
UNDO_SNAPSHOTS = 3  # model::layout::kUndoSnapshots: the undo snapshots one drafting session adds to the model's pool
# CudaModel::max_chain_drafts on this model, 12: min(kMaxChainDrafts, verify_rows - 1, max_sessions x kUndoSnapshots);
# a longer chain is refused (row status CERE_ERR_ARGUMENT)
CHAIN_DRAFTS = min(SPEC_CHAIN_MAX, SPEC_SESSIONS * UNDO_SNAPSHOTS)


def gpu_ms(fn) -> tuple[object, float]:
    """fn()'s result and the GPU ms of every CUDA record (kernels, copies, sets) it ran, under torch.profiler (CUPTI)."""
    if ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise oracle.OracleError("torch.profiler has no CUDA activity (CUPTI) in this image")
    torch.cuda.synchronize(oracle.DEVICE)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize(oracle.DEVICE)
    ns = sum(e.duration_ns() for e in prof.profiler.kineto_results.events()
             if e.device_type() == torch.autograd.DeviceType.CUDA)
    return out, ns / 1e6


class K8SpecCostsSmallM(unittest.TestCase):
    """The SpecCosts calibration timing tests on K8SpeculationSmallM's model (module docstring)."""

    LEVERS = {"k2_gemm": k8.CERE_LEVER_K2_SMALL_M}
    setUpClass = classmethod(k8.K8Speculation.setUpClass.__func__)
    draft_token_map = classmethod(k8.K8Speculation.draft_token_map.__func__)
    drafting_session = k8.K8Speculation.drafting_session

    def spec_cycle(self, prompt: list[int], truth: PlainRun, n: int, profiled: int) -> list[tuple]:
        """`prompt` speculated to P11_CYCLE_TOKENS tokens at n drafts per chain, every committed score checked
        against `truth`: per round (wall s, GPU ms or None, committed); rounds 1 .. `profiled` run under the
        profiler (GPU ms set, their walls include its overhead)."""
        with contextlib.ExitStack() as stack:
            session, prefill = self.drafting_session(stack, prompt)
            oracle.check_bit_exact(self, ROW, f"P11 n={n} prefill", prefill, truth.prefill)
            at, out = 0, []
            while at < P11_CYCLE_TOKENS:
                commit = min(n + 1, P11_CYCLE_TOKENS - at)
                call = functools.partial(session_speculate_mtp, [session], [truth.tokens[at]], [commit - 1], [commit])
                start = time.perf_counter()
                got, ms = gpu_ms(call) if 1 <= len(out) <= profiled else (call(), None)
                wall = time.perf_counter() - start
                c, a = got.committed[0], got.accepted[0]
                case = f"P11 n={n} round {len(out)} (position {len(prompt) + at})"
                self.assertTrue(1 <= c <= commit and a == c - 1, f"{ROW} {case}: committed {c} accepted {a}")
                oracle.check_bit_exact(self, ROW, f"{case} scores", got.scores[0, :c], truth.scores[at:at + c])
                out.append((wall, ms, c))
                at += c
            return out

    def verify_alone(self, prompt: list[int], truth: PlainRun, n: int) -> tuple[float, float]:
        """Median wall ms and mean GPU ms of a verify alone at m = n + 1 on the true tokens (all n drafts accepted,
        no MTP draft), P11_CYCLE_VERIFY_ROUNDS unprofiled then as many profiled; scores equal plain's."""
        walls, gpus = [], []
        with opened_session(self.handle, draft=True) as session:
            session_feed(session, prompt)
            for i in range(2 * P11_CYCLE_VERIFY_ROUNDS):
                at = i * (n + 1)
                chain = truth.tokens[at:at + n + 1]
                call = functools.partial(session_verify, session, chain, n + 1)
                start = time.perf_counter()
                if i < P11_CYCLE_VERIFY_ROUNDS:
                    verified = call()
                    walls.append(time.perf_counter() - start)
                else:
                    verified, ms = gpu_ms(call)
                    gpus.append(ms)
                oracle.check_bit_exact(self, ROW, f"P11 verify n={n} call {i} scores", verified,
                                       truth.scores[at:at + n + 1])
        return statistics.median(walls) * 1e3, statistics.mean(gpus)

    def test_p11_spec_cycle(self):
        """P11R-18 (report-only timing): the c1 speculation cycle per drafts per chain (module docstring)."""
        prompts = [ids for _, ids, _ in parity_prompts(P11_CYCLE_IDS)]
        truth = [plain_greedy(self.handle, ids, P11_CYCLE_TOKENS) for ids in prompts]
        steps = P11_CYCLE_PLAIN_STEPS
        with opened_session(self.handle) as plain:
            session_feed(plain, prompts[0])
            walls = []
            for i in range(steps):
                start = time.perf_counter()
                session_feed(plain, truth[0].tokens[i:i + 1])
                walls.append(time.perf_counter() - start)
            _, plain_gpu = gpu_ms(lambda: session_feed(plain, truth[0].tokens[steps:steps + 1]))
        plain_wall = statistics.median(walls) * 1e3
        oracle.report_row(ROW, "P11 cycle plain c1", step_wall_ms=round(plain_wall, 3),
                          step_gpu_ms=round(plain_gpu, 3), tok_s=round(1e3 / plain_wall, 2))
        for n in P11_CYCLE_DRAFTS:
            with self.subTest(drafts=n):
                runs = [self.spec_cycle(ids, t, n, P11_CYCLE_VERIFY_ROUNDS if p == 0 else 0)
                        for p, (ids, t) in enumerate(zip(prompts, truth))]
                rounds = [r for run in runs for r in run]
                timed = [r for r in rounds if r[1] is None]
                profiled = [r[1] for r in rounds if r[1] is not None]
                committed = sum(r[2] for r in rounds)
                round_wall = statistics.median(r[0] for r in timed) * 1e3
                round_gpu = statistics.mean(profiled)
                verify_wall, verify_gpu = self.verify_alone(prompts[0], truth[0], n)
                oracle.report_row(ROW, f"P11 cycle n={n}", rounds=len(rounds), committed=committed,
                                  accept_length=round(committed / len(rounds), 4),
                                  tok_s=round(sum(r[2] for r in timed) / sum(r[0] for r in timed), 2),
                                  round_wall_ms=round(round_wall, 3), round_gpu_ms=round(round_gpu, 3),
                                  verify_wall_ms=round(verify_wall, 3), verify_gpu_ms=round(verify_gpu, 3),
                                  draft_gpu_ms=round(round_gpu - verify_gpu, 3),
                                  host_ms=round(round_wall - round_gpu, 3))

    def test_spec_window_costs(self):
        """A3 (tasks/m29a-spec.md): the round cost by verify window t = 1..4 (O95: the seal-time tables measure the
        same windows, tasks/m23-spec.md SC-01). One
        drafting session per parity prompt runs SPEC_COST_ROUNDS greedy rounds of t - 1 MTP proposals (one
        speculate_rows call each, timed on the host; the call is synchronous) fed the plain run's tokens: every
        committed score equals the plain run's. Reported: the median ms per window over the rounds after each
        session's first, and the window's accept length."""
        samples = {t: [] for t in range(1, 5)}
        counts = {t: [0, 0] for t in range(1, 5)}  # committed, rounds
        for pid, prompt, _ in parity_prompts():
            truth = plain_greedy(self.handle, prompt, SPEC_DECODE)
            for t in range(1, 5):
                with self.subTest(prompt=pid, window=t), contextlib.ExitStack() as stack:
                    session, prefill = self.drafting_session(stack, prompt)
                    oracle.check_bit_exact(self, ROW, f"A3 {pid} t={t} prefill", prefill, truth.prefill)
                    at = 0
                    for round_ in range(SPEC_COST_ROUNDS):
                        commit = min(t, SPEC_DECODE - at)
                        started = time.perf_counter()
                        got = session_speculate_mtp([session], [truth.tokens[at]], [commit - 1], [commit])
                        ms = (time.perf_counter() - started) * 1e3
                        c = got.committed[0]
                        case = f"A3 {pid} t={t} round {round_}"
                        self.assertTrue(1 <= c <= commit, f"{ROW} {case}: committed {c}")
                        oracle.check_bit_exact(self, ROW, f"{case} scores", got.scores[0, :c],
                                               truth.scores[at:at + c])
                        if round_ > 0 and commit == t:
                            samples[t].append(ms)
                        counts[t][0] += c
                        counts[t][1] += 1
                        at += c
        for t in range(1, 5):
            self.assertTrue(samples[t], f"{ROW} A3 t={t}: no timed round")
            oracle.report_row(ROW, f"A3 window {t}", rounds=len(samples[t]),
                              median_ms=round(statistics.median(samples[t]), 3),
                              mean_ms=round(statistics.fmean(samples[t]), 3),
                              accept_length=round(counts[t][0] / counts[t][1], 4))

    def test_spec_deep_window_costs(self):
        """DraftAccept (SpecCosts windows 5..9): the round cost by verify window t = 4..CHAIN_DRAFTS + 1 of deep chains
        (this model's longest chain; Chains, O123, raised kMaxChainDrafts past it). One drafting
        session per parity prompt runs SPEC_COST_ROUNDS greedy rounds of t - 1 caller drafts = the plain run's next
        tokens (cere_ktest_session_speculate_chain, every draft accepted, so the round commits t tokens and extends
        t rows: the dearest round of the window), timed on the host; every committed score equals the plain run's.
        Reported: the median ms per window over the rounds after each session's first (window 4 measured the same
        way, to scale against the A3 table)."""
        windows = range(4, CHAIN_DRAFTS + 2)
        samples = {t: [] for t in windows}
        for pid, prompt, _ in parity_prompts():
            truth = plain_greedy(self.handle, prompt, SPEC_DECODE)
            for t in windows:
                with self.subTest(prompt=pid, window=t), contextlib.ExitStack() as stack:
                    session, prefill = self.drafting_session(stack, prompt)
                    oracle.check_bit_exact(self, ROW, f"A3D {pid} t={t} prefill", prefill, truth.prefill)
                    at = 0
                    for round_ in range(SPEC_COST_ROUNDS):
                        commit = min(t, SPEC_DECODE - at)
                        if commit < 2:
                            break
                        drafts = [list(truth.tokens[at + 1:at + commit])]
                        started = time.perf_counter()
                        got = session_speculate_chain([session], [truth.tokens[at]], drafts, [commit])
                        ms = (time.perf_counter() - started) * 1e3
                        c = got.committed[0]
                        case = f"A3D {pid} t={t} round {round_}"
                        self.assertEqual(c, commit, f"{ROW} {case}: committed")
                        oracle.check_bit_exact(self, ROW, f"{case} scores", got.scores[0, :c],
                                               truth.scores[at:at + c])
                        if round_ > 0 and commit == t:
                            samples[t].append(ms)
                        at += c
        for t in windows:
            self.assertTrue(samples[t], f"{ROW} A3D t={t}: no timed round")
            oracle.report_row(ROW, f"A3D window {t}", rounds=len(samples[t]),
                              median_ms=round(statistics.median(samples[t]), 3),
                              mean_ms=round(statistics.fmean(samples[t]), 3))


if __name__ == "__main__":
    unittest.main()
