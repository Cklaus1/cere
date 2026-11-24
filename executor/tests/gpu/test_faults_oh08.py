"""OH-08 (tasks/ops-hardening-spec.md): the executor's failure paths, driven on the real A4 model by the compile-time
fault-injection hooks (executor/cuda/src/fault/fault_inject.h), container only.

Suite `faults` (`native_dev.py test --suite faults --build-dir D`, D a `native_dev.py build --fault-injection
--build-dir D` library: the default build has no hook and no `cere_ktest_fault_refusals`, so the suite fails there,
it never skips). Each case is one child process (`python3 test_faults_oh08.py <case> <result.json>`) with
`CERE_FAULT_INJECT=<site>:<n>` set, which the library reads once: the n-th pass of the site fails, every other pass
runs normally, and the hit logs `cere: fault_inject site=<site> hit=<n>`. A child loads its own model through
`oracle.create_model` (a sticky fault leaves the process's CUDA context unusable for good, so no case shares a
process with another) and writes its counts and score rows to the result file (token IDs stay out of the report
rows); the parent asserts. The prompts are seeded synthetic ids (`test_graph_decode.prompts_of`): a PROMPT-token
PREFILL, then STEPS teacher-forced DECODEs (`test_graph_decode.run_session`). Each case fails on the code before the
fix it guards:
* `test_mf2_failure_after_launch_breaks_the_session` (MF-2, site forward_launch): a forward whose launches all ran
  returns CERE_ERR_CAPACITY (not INTERNAL, the session position unchanged). Before MF-2 the session stayed usable;
  now it is broken: the same feed again is CERE_ERR_INTERNAL. A fresh session afterwards scores bit-identically to
  the reference session run before the fault. Its `cere_ktest_fault_refusals` is the healthy control: not faulted,
  a session is created, the sealed model's seal is CERE_ERR_ARGUMENT, the release succeeds.
* `test_dm2_sticky_fault_poisons_the_device` (DM-2, site forward_sticky): a trap kernel on the forward's stream
  (cudaErrorLaunchFailure, sticky). The feed is CERE_ERR_INTERNAL, `cere: device_poisoned ordinal=0` is logged once,
  and the executor refuses new work: faulted, session create and seal CERE_ERR_INTERNAL; the model and the executor
  still release.
* `test_mf1_failed_seal_is_final` (MF-1, site seal): INSTANTIATE fails after instantiate_model rewrote the MoE
  block scales in place. The seal is CERE_ERR_INTERNAL; a second seal is CERE_ERR_INTERNAL too (before MF-1 it
  instantiated again, over the rewritten scales, and succeeded); the model releases.
* `test_dm5_failed_graph_instantiation_latches_eager` (DM-5, site graph_instantiate): the decode graph's
  instantiation fails at the first decode. The session's rows equal the eager run's bit for bit,
  `cere: decode_graph_instantiate_failed` is logged once and the key is latched: no capture, update or replay for the
  rest of the session (before DM-5 the next step captured again). A session of another context (another key)
  captures once and replays every step, its rows equal to its eager run's.
* `test_kgm2_error_after_the_moe_fork_joins_it` (KGM-2, site moe_fork): with decode graphs on, the first
  shared-expert fork (the first decode step, inside the graph capture; the PREFILL's PROMPT rows are above
  k2s::kMaxM and do not fork) is followed by an error. The fork is joined before the error returns, so the capture
  ends cleanly and the step reruns eagerly: the feed succeeds, its rows equal the eager run's bit for bit and the
  remaining STEPS - 1 steps replay the graph captured at step 2. It guards the recovery, not the pre-KGM-2 code:
  graph_step reruns a step that failed inside a capture eagerly whether or not the fork was joined, so a no-op
  ~SharedForkScope passes it too (OH-08 mutant run, 2026-10-22); outside a capture the missing join is a race, not a
  deterministic result.
* `test_f3_device_bound_fault_breaks_the_session` (KernelsAttention F3, site bound_fault): the faulted session's
  PREFILL finds the K9 device-bound fault word set (a clamped out-of-range device position). The feed is
  CERE_ERR_INTERNAL with `cere: device_bound_fault code=1` logged once, the word is cleared, the session is broken
  (the same feed again is CERE_ERR_INTERNAL), the device is not poisoned, and a fresh session afterwards scores
  bit-identically to the reference session run before the fault. Before F3 no word existed: the clamped call
  succeeded silently.
Native entry points: K8's `cere_ktest_model_*` (stated in `test_model_k8_forward.py`), K15's
`cere_ktest_session_open`, `cere_ktest_session_feed` (max_new_tokens 0, no names) and `cere_ktest_session_close`
(stated in `test_model_k15_snapshot.py`), P1's `cere_ktest_decode_graph` (stated in `test_graph_decode.py`);
`oracle.call` appends the stream; 0 = success, otherwise a `cere_status`; "host" buffers are CPU memory. OH-08's own
(fault-injection builds only):
    int cere_ktest_fault_refusals(uint64_t model, uint64_t session, int64_t *out /*host [4]*/,
                                  cudaStream_t /*stream*/);
        Never touches the stream. `session` (0: none), a K15 session of the model, is released first; out =
        {Device::faulted(), cere_session_create's status (a created session is released), cere_model_seal's status,
        the status of releasing the model and then its executor}; the handle is gone when out[3] is 0.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

import oracle
import test_graph_decode as graph

ROW = "OH08"
CONTEXT = graph.CONTEXT_K6  # one session of the plan's context; K6 attention at every position
SHORT = graph.REUSE_SHORT  # another session context: another decode-graph key
PROMPT = 40  # PREFILL rows above k2s::kMaxM (16): the PREFILL never forks the shared expert
STEPS = graph.STEPS
CHILD_TIMEOUT_S = 1_800
CERE_ERR_ARGUMENT = -1
CERE_ERR_CAPACITY = -3
CERE_ERR_INTERNAL = -6
FAULT_ENV = "CERE_FAULT_INJECT"
# The forward_launch pass that fails in the MF-2 case: the one after the reference session's PREFILL and STEPS
# DECODEs, i.e. the faulted session's PREFILL.
MF2_HIT = 1 + STEPS + 1
BOUND_HIT = MF2_HIT  # the bound_fault pass after the reference session: the faulted session's PREFILL


def tokens() -> tuple[list, list]:
    prompt, forced = graph.prompts_of((PROMPT, STEPS))
    return prompt, forced


def open_session(handle: int, context: int) -> int:
    out = torch.zeros(1, dtype=torch.int64)
    oracle.call("cere_ktest_session_open", handle, context, out)
    return int(out[0])


def feed_status(session: int, ids: list, *, synchronize: bool = True) -> int:
    """One teacher-forced `cere_ktest_session_feed` of `ids`; its status, not raised."""
    scores = torch.full((len(ids), 4), -1, dtype=torch.int32)
    return oracle.call_status("cere_ktest_session_feed", session, torch.tensor(ids, dtype=torch.int32), len(ids), 0,
                              0, None, 0, None, 0, None, 0, scores, torch.zeros(3, dtype=torch.int64),
                              synchronize=synchronize)


def fault_refusals(handle: int, session: int) -> list:
    """`cere_ktest_fault_refusals`: releases `session` (0: none), probes the executor, releases the model."""
    out = torch.zeros(4, dtype=torch.int64)
    status = oracle.call_status("cere_ktest_fault_refusals", handle, session, out, synchronize=False)
    if status != 0:
        raise oracle.OracleError(f"cere_ktest_fault_refusals returned status {status}")
    return [int(v) for v in out.tolist()]


def rows(scores: torch.Tensor) -> list:
    return scores.tolist()


# -- child cases: each runs in its own process with CERE_FAULT_INJECT set, returns its result ------------------------


def child_mf2() -> dict:
    handle = oracle.create_model(CONTEXT)
    prompt, forced = tokens()
    reference = graph.run_session(handle, CONTEXT, prompt, forced)
    session = open_session(handle, CONTEXT)
    first = feed_status(session, prompt)
    again = feed_status(session, prompt)
    oracle.call("cere_ktest_session_close", session)
    after = graph.run_session(handle, CONTEXT, prompt, forced)
    return {"first": first, "again": again, "reference": rows(reference), "after": rows(after),
            "refusals": fault_refusals(handle, 0)}


def child_dm2() -> dict:
    handle = oracle.create_model(CONTEXT)
    prompt, _ = tokens()
    session = open_session(handle, CONTEXT)
    status = feed_status(session, prompt, synchronize=False)
    return {"status": status, "refusals": fault_refusals(handle, session)}


def child_mf1() -> dict:
    model_buf, entries, keep = oracle.a4_model_desc(CONTEXT)
    handle = oracle.allocate(model_buf)
    first = oracle.upload_and_seal_status(handle, entries)
    second = oracle.call_status("cere_ktest_model_seal", handle)
    released = oracle.call_status("cere_ktest_model_release", handle)
    del keep
    return {"first": first, "second": second, "released": released}


def child_dm5() -> dict:
    handle = oracle.create_model(CONTEXT)
    prompt, forced = tokens()
    graph.decode_graph(handle, 0)
    eager = graph.run_session(handle, CONTEXT, prompt, forced)
    eager_short = graph.run_session(handle, SHORT, prompt, forced)
    before = graph.decode_graph(handle, 1)
    faulted = graph.run_session(handle, CONTEXT, prompt, forced)
    mid = graph.decode_graph(handle, -1)
    recovered = graph.run_session(handle, SHORT, prompt, forced)
    after = graph.decode_graph(handle, -1)
    oracle.release_model(handle)
    return {"eager": rows(eager), "faulted": rows(faulted), "eager_short": rows(eager_short),
            "recovered": rows(recovered), "before": list(before), "mid": list(mid), "after": list(after)}


def child_kgm2() -> dict:
    handle = oracle.create_model(CONTEXT)
    prompt, forced = tokens()
    before = graph.decode_graph(handle, 1)
    forked = graph.run_session(handle, CONTEXT, prompt, forced)
    after = graph.decode_graph(handle, 0)
    eager = graph.run_session(handle, CONTEXT, prompt, forced)
    oracle.release_model(handle)
    return {"forked": rows(forked), "eager": rows(eager), "before": list(before), "after": list(after)}


def child_bound() -> dict:
    handle = oracle.create_model(CONTEXT)
    prompt, forced = tokens()
    reference = graph.run_session(handle, CONTEXT, prompt, forced)
    session = open_session(handle, CONTEXT)
    first = feed_status(session, prompt)
    again = feed_status(session, prompt)
    oracle.call("cere_ktest_session_close", session)
    after = graph.run_session(handle, CONTEXT, prompt, forced)
    return {"first": first, "again": again, "reference": rows(reference), "after": rows(after),
            "refusals": fault_refusals(handle, 0)}


CHILDREN = {"mf2": child_mf2, "dm2": child_dm2, "mf1": child_mf1, "dm5": child_dm5, "kgm2": child_kgm2,
            "bound": child_bound}


def child_main(case: str, out: str) -> None:
    """Runs one child case and writes its result; exits without interpreter teardown (after a sticky fault torch's
    CUDA state is unusable and the executor reset the device)."""
    result = CHILDREN[case]()
    Path(out).write_text(json.dumps(result))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


# -- parent ------------------------------------------------------------------------------------------------------


def run_child(case: str, fault: str) -> tuple[dict, list]:
    """Runs child `case` under CERE_FAULT_INJECT=`fault`; (its result, its `cere: ` stderr lines). Its output is
    echoed. The fault-injection ktest must be in the library under test (a default build fails here)."""
    oracle.ktest("cere_ktest_fault_refusals")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "result.json"
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), case, str(out)],
                              env=dict(os.environ, **{FAULT_ENV: fault}), capture_output=True, text=True,
                              timeout=CHILD_TIMEOUT_S)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0 or not out.is_file():
            raise oracle.OracleError(f"fault case {case} ({FAULT_ENV}={fault}) exited {proc.returncode}: "
                                     f"{proc.stderr.strip()[-2000:]}")
        result = json.loads(out.read_text())
    return result, [line for line in proc.stderr.splitlines() if line.startswith("cere: ")]


def count(lines: list, prefix: str) -> int:
    return sum(1 for line in lines if line.startswith(prefix))


def scores(values: list) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


class Faults(unittest.TestCase):
    def hit_once(self, case: str, lines: list, site: str, hit: int) -> None:
        oracle.check_count(self, ROW, f"{case}/fault_hits", count(lines, f"cere: fault_inject site={site} hit={hit}"),
                           expected=1)
        oracle.check_count(self, ROW, f"{case}/fault_lines", count(lines, "cere: fault_inject "), expected=1)

    def test_mf2_failure_after_launch_breaks_the_session(self):
        case = "MF-2"
        result, lines = run_child("mf2", f"forward_launch:{MF2_HIT}")
        self.hit_once(case, lines, "forward_launch", MF2_HIT)
        oracle.check_count(self, ROW, f"{case}/first_feed", result["first"], expected=CERE_ERR_CAPACITY)
        oracle.check_count(self, ROW, f"{case}/broken_session_feed", result["again"], expected=CERE_ERR_INTERNAL)
        oracle.check_bit_exact(self, ROW, f"{case}/fresh_session_rows", scores(result["after"]),
                               scores(result["reference"]))
        faulted, created, sealed, released = result["refusals"]
        oracle.check_count(self, ROW, f"{case}/control_faulted", faulted, expected=0)
        oracle.check_count(self, ROW, f"{case}/control_session_create", created, expected=0)
        oracle.check_count(self, ROW, f"{case}/control_seal_sealed", sealed, expected=CERE_ERR_ARGUMENT)
        oracle.check_count(self, ROW, f"{case}/control_release", released, expected=0)
        oracle.check_count(self, ROW, f"{case}/device_poisoned_lines", count(lines, "cere: device_poisoned"),
                           expected=0)

    def test_dm2_sticky_fault_poisons_the_device(self):
        case = "DM-2"
        result, lines = run_child("dm2", "forward_sticky:1")
        self.hit_once(case, lines, "forward_sticky", 1)
        oracle.check_count(self, ROW, f"{case}/feed", result["status"], expected=CERE_ERR_INTERNAL)
        oracle.check_count(self, ROW, f"{case}/device_poisoned_lines",
                           count(lines, "cere: device_poisoned ordinal=0 error=cudaErrorLaunchFailure"), expected=1)
        faulted, created, sealed, released = result["refusals"]
        oracle.check_count(self, ROW, f"{case}/faulted", faulted, expected=1)
        oracle.check_count(self, ROW, f"{case}/session_create", created, expected=CERE_ERR_INTERNAL)
        oracle.check_count(self, ROW, f"{case}/seal", sealed, expected=CERE_ERR_INTERNAL)
        oracle.check_count(self, ROW, f"{case}/release", released, expected=0)

    def test_mf1_failed_seal_is_final(self):
        case = "MF-1"
        result, lines = run_child("mf1", "seal:1")
        self.hit_once(case, lines, "seal", 1)
        oracle.check_count(self, ROW, f"{case}/seal", result["first"], expected=CERE_ERR_INTERNAL)
        oracle.check_count(self, ROW, f"{case}/seal_again", result["second"], expected=CERE_ERR_INTERNAL)
        oracle.check_count(self, ROW, f"{case}/release", result["released"], expected=0)

    def test_dm5_failed_graph_instantiation_latches_eager(self):
        case = "DM-5"
        result, lines = run_child("dm5", "graph_instantiate:1")
        self.hit_once(case, lines, "graph_instantiate", 1)
        oracle.check_count(self, ROW, f"{case}/instantiate_failed_lines",
                           count(lines, "cere: decode_graph_instantiate_failed rows=1 "), expected=1)
        oracle.check_bit_exact(self, ROW, f"{case}/latched_rows", scores(result["faulted"]), scores(result["eager"]))
        before, mid, after = (graph.Counters(*result[k]) for k in ("before", "mid", "after"))
        oracle.check_count(self, ROW, f"{case}/latched_captures", mid.captures - before.captures, expected=0)
        oracle.check_count(self, ROW, f"{case}/latched_updates", mid.updates - before.updates, expected=0)
        oracle.check_count(self, ROW, f"{case}/latched_replays", mid.replays - before.replays, expected=0)
        oracle.check_bit_exact(self, ROW, f"{case}/other_key_rows", scores(result["recovered"]),
                               scores(result["eager_short"]))
        oracle.check_count(self, ROW, f"{case}/other_key_captures", after.captures - mid.captures, expected=1)
        oracle.check_count(self, ROW, f"{case}/other_key_replays", after.replays - mid.replays, expected=STEPS)

    def test_kgm2_error_after_the_moe_fork_joins_it(self):
        case = "KGM-2"
        result, lines = run_child("kgm2", "moe_fork:1")
        self.hit_once(case, lines, "moe_fork", 1)
        oracle.check_bit_exact(self, ROW, f"{case}/rows", scores(result["forked"]), scores(result["eager"]))
        before, after = graph.Counters(*result["before"]), graph.Counters(*result["after"])
        oracle.check_count(self, ROW, f"{case}/captures", after.captures - before.captures, expected=1)
        oracle.check_count(self, ROW, f"{case}/replays", after.replays - before.replays, expected=STEPS - 1)

    def test_f3_device_bound_fault_breaks_the_session(self):
        case = "F3"
        result, lines = run_child("bound", f"bound_fault:{BOUND_HIT}")
        self.hit_once(case, lines, "bound_fault", BOUND_HIT)
        oracle.check_count(self, ROW, f"{case}/first_feed", result["first"], expected=CERE_ERR_INTERNAL)
        oracle.check_count(self, ROW, f"{case}/bound_fault_lines", count(lines, "cere: device_bound_fault code=1"),
                           expected=1)
        oracle.check_count(self, ROW, f"{case}/broken_session_feed", result["again"], expected=CERE_ERR_INTERNAL)
        oracle.check_bit_exact(self, ROW, f"{case}/fresh_session_rows", scores(result["after"]),
                               scores(result["reference"]))
        faulted, created, sealed, released = result["refusals"]
        oracle.check_count(self, ROW, f"{case}/faulted", faulted, expected=0)
        oracle.check_count(self, ROW, f"{case}/session_create", created, expected=0)
        oracle.check_count(self, ROW, f"{case}/release", released, expected=0)
        oracle.check_count(self, ROW, f"{case}/device_poisoned_lines", count(lines, "cere: device_poisoned"),
                           expected=0)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in CHILDREN:
        child_main(sys.argv[1], sys.argv[2])
    unittest.main()
