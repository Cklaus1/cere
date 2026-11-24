"""M3.2N G-1 (tasks/m32n-spec.md D4, R4): two executors in one process, one per card (GPU0 + GPU1).

Suite `twocard_d4` (`native_dev.py test --suite twocard_d4`, or `twocard` with the O47 module; a G1 window: the
container sees exactly GPU0 then GPU1 as CUDA ordinals 0 and 1, native_dev.TWO_CARD_ATTRIBUTION). Each executor is
the real A4 model with the MTP draft layer (`oracle.create_model(..., device_id="<ordinal>", draft_token_map=...)`)
holding its own private PLE copy, so both load at once: ALLOCATE, UPLOAD and INSTANTIATE of the two run on two threads.
Inputs are synthetic seeded token ids (no prompt text); a repeated block lets MTP drafts be accepted.

* `test_runner_per_device`: after both loads, the P3b runner slot of each ordinal was built on that ordinal and serves
  it, and each slot refuses the other card (`cere_ktest_p3b_runner_binding`, CERE_ERR_INTERNAL): D4's runner per
  device ordinal; a runner shared across ordinals (the pre-D4 process-global runner) is refused, so a second card's
  INSTANTIATE fails instead of driving the first card's runner.
* `test_concurrent_equals_single`: per card a single-executor run (the other executor idle) of PREFILL + DECODE
  (CERE_WORK_NO_SPECULATION) and of PREFILL + DECODE + VERIFY (flags 0: MTP drafts, at least one verify step), then
  the same requests on both cards at the same time (two threads): each card's tokens and scores equal its single run
  bit for bit, and the two cards equal each other (one SKU, deterministic kernels).
"""
from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import torch

import oracle

ROW = "R4"
MAX_CONTEXT = 4_096
PROMPT_TOKENS = 1_024
MAX_NEW = 48
DRAFT_MAP = "/draft_token_map.u32le"  # native_dev.py DRAFT_MAP_MOUNT
ORDINALS = (0, 1)
CERE_ERR_INTERNAL = -6
CERE_WORK_NO_SPECULATION = 0x2
SPEC_FLAGS = 0  # MTP drafts (the draft layer is loaded)
FINISHES = (1, 2)  # length, stop token


def synthetic_ids(n: int) -> list[int]:
    """`n` deterministic in-vocabulary ids (no text, no tokenizer): a 64-id block repeated, so drafts can be accepted."""
    block = [1_000 + (i * 7_919) % 30_000 for i in range(64)]
    return [block[i % len(block)] for i in range(n)]


def runner_binding(ordinal: int, device: int) -> tuple[int, int]:
    """(status of the runner slot `ordinal` with `device` current, the ordinal it was built on or -1)."""
    out = torch.full((2,), -2, dtype=torch.int64)
    oracle.call("cere_ktest_p3b_runner_binding", ordinal, device, out)
    return int(out[0]), int(out[1])


def generate_flags(handle: int, ids: list[int], flags: int) -> dict:
    """`cere_ktest_generate_flags`: tokens and scores (int32), committed, finish and verify steps."""
    prompt = torch.tensor(ids, dtype=torch.int32)
    tokens = torch.full((MAX_NEW,), -1, dtype=torch.int32)
    scores = torch.full((MAX_NEW, 4), -1, dtype=torch.int32)
    result = torch.zeros(6, dtype=torch.int64)
    oracle.call("cere_ktest_generate_flags", handle, prompt, len(ids), MAX_NEW, flags, tokens, scores, result)
    committed, finish, _, steps, accepted, _ = (int(v) for v in result.tolist())
    if not 0 <= committed <= MAX_NEW:
        raise oracle.OracleError(f"committed_tokens {committed} outside [0, {MAX_NEW}]")
    return {"tokens": tokens[:committed].clone(), "scores": scores[:committed].clone(), "committed": committed,
            "finish": finish, "steps": steps, "accepted": accepted}


def run_threads(calls: list) -> tuple[list, list]:
    """Runs the zero-argument `calls` at the same time, one thread each (ctypes drops the GIL in the call): (results,
    exceptions), both in call order (None where none)."""
    start = threading.Barrier(len(calls))

    def run(call):
        start.wait()
        return call()

    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = [pool.submit(run, call) for call in calls]
    errors = [f.exception() for f in futures]
    return [None if e is not None else f.result() for f, e in zip(futures, errors)], errors


def on_threads(calls: list) -> list:
    """`run_threads`' results, re-raising the first failure."""
    results, errors = run_threads(calls)
    for e in errors:
        if e is not None:
            raise e
    return results


def same(a: dict, b: dict) -> bool:
    return (a["committed"] == b["committed"] and a["finish"] == b["finish"] and torch.equal(a["tokens"], b["tokens"])
            and torch.equal(a["scores"], b["scores"]))


class TwoCardExecutors(unittest.TestCase):
    """Two executors, one per CUDA ordinal, loaded concurrently in this process."""

    @classmethod
    def setUpClass(cls):
        if torch.cuda.device_count() != len(ORDINALS):
            raise oracle.OracleError(f"{torch.cuda.device_count()} CUDA devices visible, not GPU0 + GPU1")
        with open(DRAFT_MAP, "rb") as f:
            cls.draft_map = torch.frombuffer(bytearray(f.read()), dtype=torch.int32)
        loads = [lambda o=o: oracle.create_model(MAX_CONTEXT, device_id=str(o), draft_token_map=cls.draft_map)
                 for o in ORDINALS]
        started = time.monotonic()
        handles, errors = run_threads(loads)
        cls.load_ms = int((time.monotonic() - started) * 1000)
        for handle in handles:
            if handle is not None:  # a load that succeeded while the other failed is still released
                cls.addClassCleanup(oracle.release_model, handle)
        for e in errors:
            if e is not None:
                raise e
        cls.handles = dict(zip(ORDINALS, handles))
        oracle.report_row(ROW, "concurrent load", ordinals=len(ORDINALS), load_ms=cls.load_ms, passed=True)

    def test_runner_per_device(self):
        bindings = {(o, d): runner_binding(o, d) for o in ORDINALS for d in ORDINALS}
        own = all(bindings[(o, o)] == (0, o) for o in ORDINALS)
        refused = all(bindings[(o, d)] == (CERE_ERR_INTERNAL, o) for o in ORDINALS for d in ORDINALS if o != d)
        oracle.report_row(ROW, "runner per device", **{f"slot{o}_on{d}_status": s for (o, d), (s, _) in bindings.items()},
                          **{f"slot{o}_built_on": bindings[(o, o)][1] for o in ORDINALS}, passed=own and refused)
        for o in ORDINALS:
            self.assertEqual(bindings[(o, o)], (0, o), f"{ROW}: ordinal {o}'s runner is not its own")
        for (o, d), got in bindings.items():
            if o != d:
                self.assertEqual(got, (CERE_ERR_INTERNAL, o),
                                 f"{ROW}: ordinal {o}'s runner was not refused with device {d} current")

    def test_concurrent_equals_single(self):
        ids = synthetic_ids(PROMPT_TOKENS)
        for name, flags in (("decode", CERE_WORK_NO_SPECULATION), ("verify", SPEC_FLAGS)):
            with self.subTest(work=name):
                single = {o: generate_flags(self.handles[o], ids, flags) for o in ORDINALS}
                both = dict(zip(ORDINALS, on_threads(
                    [lambda o=o: generate_flags(self.handles[o], ids, flags) for o in ORDINALS])))
                per_card = all(same(single[o], both[o]) for o in ORDINALS)
                across = same(single[0], single[1])
                verified = flags == CERE_WORK_NO_SPECULATION or all(single[o]["steps"] > 0 for o in ORDINALS)
                oracle.report_row(ROW, f"concurrent equals single {name}",
                                  **{f"committed{o}": single[o]["committed"] for o in ORDINALS},
                                  **{f"finish{o}": single[o]["finish"] for o in ORDINALS},
                                  **{f"verify_steps{o}": single[o]["steps"] for o in ORDINALS},
                                  **{f"accepted{o}": single[o]["accepted"] for o in ORDINALS},
                                  per_card_equal=per_card, cards_equal=across, passed=per_card and across and verified)
                for o in ORDINALS:
                    self.assertIn(single[o]["finish"], FINISHES, f"{ROW}: card {o} finish {single[o]['finish']}")
                    self.assertTrue(single[o]["committed"] >= 2, f"{ROW}: card {o}: no DECODE step ran")
                    self.assertTrue(same(single[o], both[o]),
                                    f"{ROW}: card {o}'s concurrent {name} run differs from its single run")
                self.assertTrue(verified, f"{ROW}: a speculative run took no verify step")
                self.assertTrue(across, f"{ROW}: GPU0 and GPU1 differ on the same {name} request")


if __name__ == "__main__":
    unittest.main()
