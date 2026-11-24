"""BC0 (tasks/batch-concurrency-spec.md BC-D1 (a)): the batch invariance gate's plan and verdicts (stdlib only).

The GPU suite `batch_inv` (test_batch_inv_gate.py) runs, per profile, every request of `requests(rows)` alone (the
solo reference, M = 1) and in seeded compositions (`schedules`): random session-open order (so slots differ), random
joins (rows join and leave mid-decode), a long prompt joining late (its prefill chunks interleave with decode rounds),
one prefix-cache hit, plain rows beside drafting rows, greedy and seeded-sampled rows, dense prompts and prompts past
2,048 positions (the K9 sparse regime), and decode / spec graphs switched between polls. Each request's committed
token ids, `cere_token_score` rows and logprobs rows must equal its solo run bit for bit (`differences`).

Cells (`CELLS`), every one gated: `greedy` (every greedy request), `sampled_plain` (seeded-sampled requests on
non-drafting sessions, ledger CR4) and `sampled_drafting` (seeded-sampled requests on drafting sessions with the flags
cered sends, `WORK_LOOKUP_DRAFTS` only; report-only until BC1, F11: their lookup picks were priced by the model-wide
wall-clock EMA `spec_costs`, deleted by BC-D4, so they are invariant by construction).

The work flags mirror cered's `draft_flags` (crates/cere-runtime/src/native/serve.rs) under the prod profile
(-v52: drafter mtp+lookup, draft_policy adaptive): a greedy drafting ticket LOOKUP | ADAPTIVE, a sampled drafting
ticket LOOKUP, a plain row NO_SPECULATION (a request cered serves without speculation, e.g. `ignore_eos`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# cere_executor.h cere_work.flags
WORK_NO_SPECULATION = 0x2
WORK_LOOKUP_DRAFTS = 0x4
WORK_ADAPTIVE_DRAFTS = 0x10

SEEDS = (20261024, 20261025, 20261026)  # BC-D1: three seeds of compositions per profile
SAMPLING_TEMPERATURE = 0.6  # the benchmark's sampling (llm-subagent-benchmark §6)
SAMPLING_TOP_P = 0.95
SAMPLING_SEED = 20261024  # request i's seed is SAMPLING_SEED + i
TOP_LOGPROBS = 5
HIT_PREFIX = 4096  # the primer's capture point (K16: a namespaced cold PREFILL of 5,000 tokens captures at 4,096)
PRIMER_TOKENS = 5000
PRIMER_MAX_NEW = 4
LONG_JOIN = (2, 8)  # the long prompt joins at a poll in this range, so its prefill chunks meet decode rounds
JOIN_MAX = 12  # a late-joining request's poll (early enough that every row of a run decodes together at some round)
GRAPH_MODES = (-1, 0, 1, 2, 3)  # per poll: leave, or bit 0 decode graphs, bit 1 spec graphs
GRAPH_PERIOD = 7  # graph_modes entries per run (cycled over the polls)


@dataclass(frozen=True)
class Request:
    """One request of the gate: a synthetic prompt of `prompt_tokens` (records from `record`; the `hit` request is the
    primer's first HIT_PREFIX tokens plus a fresh tail), `max_new` tokens, greedy or seeded-sampled, drafting or a
    plain row; `kind` dense (< 2,048 positions), sparse (past 2,048), long (several prefill chunks, joins late) or
    hit (a prefix-cache hit at HIT_PREFIX)."""
    name: str
    kind: str
    prompt_tokens: int
    max_new: int
    sampled: bool
    drafting: bool
    record: int

    @property
    def flags(self) -> int:
        if not self.drafting:
            return WORK_NO_SPECULATION
        return WORK_LOOKUP_DRAFTS | (0 if self.sampled else WORK_ADAPTIVE_DRAFTS)

    @property
    def cell(self) -> str:
        if not self.sampled:
            return "greedy"
        return "sampled_drafting" if self.drafting else "sampled_plain"


# (name, kind, prompt tokens, max_new, sampled, drafting); the first PROD_REQUESTS serve the prod profile (4 rows),
# all of them the dev profile (16 rows).
_TABLE = (
    ("g_draft_dense", "dense", 600, 128, False, True),
    ("g_plain_dense", "dense", 300, 96, False, False),
    ("g_draft_sparse", "sparse", 2400, 128, False, True),
    ("g_plain_sparse", "sparse", 2350, 96, False, False),
    ("g_long", "long", 5200, 64, False, True),
    ("g_hit", "hit", HIT_PREFIX + 600, 96, False, True),
    ("s_plain_dense", "dense", 400, 96, True, False),
    ("s_draft_dense", "dense", 700, 128, True, True),
    ("s_plain_sparse", "sparse", 2300, 96, True, False),
    ("s_draft_sparse", "sparse", 2450, 128, True, True),
    ("g_draft_dense2", "dense", 900, 128, False, True),
    ("g_plain_dense2", "dense", 200, 96, False, False),
    ("g_draft_dense3", "dense", 450, 160, False, True),
    ("s_draft_dense2", "dense", 500, 128, True, True),
    ("g_draft_sparse2", "sparse", 2600, 128, False, True),
    ("s_plain_dense2", "dense", 350, 96, True, False),
)
PROD_REQUESTS = 10
CONTEXT = 8192  # every prompt + max_new fits; > 2,048 so sparse prompts take the K9 path


def requests(rows: int) -> list[Request]:
    """The profile's requests: PROD_REQUESTS for rows <= 4, all 16 for more rows."""
    if rows < 2:
        raise ValueError(f"the gate composes at least 2 rows, not {rows}")
    count = PROD_REQUESTS if rows <= 4 else len(_TABLE)
    return [Request(name, kind, tokens, new, sampled, drafting, 1000 * (i + 1))
            for i, (name, kind, tokens, new, sampled, drafting) in enumerate(_TABLE[:count])]


@dataclass(frozen=True)
class Run:
    """One composition: requests `members` (indices into the request list) opened in `open_order`, member i
    submitted before poll joins[i]; the poll seed and the graph modes of cere_ktest_compose."""
    members: tuple[int, ...]
    open_order: tuple[int, ...]
    joins: tuple[int, ...]
    poll_seed: int
    graph_modes: tuple[int, ...]


def schedules(reqs: list[Request], rows: int, seed: int) -> list[Run]:
    """The seed's compositions: the requests shuffled into groups of `rows` (the last one filled up with other
    requests, so every run holds `rows` requests and every request runs at least once). Joins: the long request at
    a poll in LONG_JOIN, every other one at poll 0 or (half of them, seeded) at a poll in [1, JOIN_MAX], at least one
    at 0."""
    if rows < 2 or rows > len(reqs):
        raise ValueError(f"rows {rows} outside [2, {len(reqs)}]")
    rng = random.Random(seed)
    order = list(range(len(reqs)))
    rng.shuffle(order)
    groups = [order[i:i + rows] for i in range(0, len(order), rows)]
    if len(groups[-1]) < rows:
        rest = [i for i in order if i not in groups[-1]]
        groups[-1] += rng.sample(rest, rows - len(groups[-1]))
    runs = []
    for group in groups:
        joins = []
        for i in group:
            if reqs[i].kind == "long":
                joins.append(rng.randint(*LONG_JOIN))
            else:
                joins.append(0 if rng.random() < 0.5 else rng.randint(1, JOIN_MAX))
        if 0 not in (j for j, i in zip(joins, group) if reqs[i].kind != "long"):
            joins[next(k for k, i in enumerate(group) if reqs[i].kind != "long")] = 0
        opened = list(group)
        rng.shuffle(opened)
        runs.append(Run(tuple(group), tuple(opened), tuple(joins), rng.getrandbits(63) | 1,
                        tuple(rng.choice(GRAPH_MODES) for _ in range(GRAPH_PERIOD))))
    return runs


@dataclass(frozen=True)
class Outcome:
    """One request's result: committed tokens, finish, cached_tokens and the bytes of its token ids, score rows and
    logprobs rows (b"" when it has none)."""
    committed: int
    finish: int
    cached: int
    tokens: bytes
    scores: bytes
    logprobs: bytes


def differences(solo: Outcome, batched: Outcome) -> list[str]:
    """The fields in which `batched` differs from its solo run (empty = bit-equal); token-level fields name the first
    differing token."""
    out = []
    for name in ("committed", "finish", "cached"):
        if getattr(solo, name) != getattr(batched, name):
            out.append(f"{name} {getattr(solo, name)} vs {getattr(batched, name)}")
    for name in ("tokens", "scores", "logprobs"):
        a, b = getattr(solo, name), getattr(batched, name)
        if a == b:
            continue
        row = len(a) // solo.committed if solo.committed else 1
        at = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        out.append(f"{name} differ from token {at // max(row, 1)}")
    return out


FIELD_MAX = 120  # native_dev.LAB_REPORT_STRING: <= 128 characters, no `;`


def field(items: list[str]) -> str:
    """A report row string of `items` (difference texts or request labels): joined by ` + `, cut to FIELD_MAX
    characters, `none` when empty."""
    text = " + ".join(items) or "none"
    return text if len(text) <= FIELD_MAX else text[:FIELD_MAX - 4] + " etc"


CELLS = ("greedy", "sampled_plain", "sampled_drafting")


def verdict(compared: list[tuple[str, str, list[str]]]) -> dict[str, dict]:
    """Per cell: {status GREEN/RED, compared, differing (request@run names)} over (cell, label, differences)."""
    out = {c: {"status": "GREEN", "compared": 0, "differing": []} for c in CELLS}
    for cell, label, diff in compared:
        entry = out[cell]
        entry["compared"] += 1
        if diff:
            entry["differing"].append(label)
            entry["status"] = "RED"
    return out


def coverage(rounds: list[tuple[int, int, int, int]]) -> dict[str, int]:
    """What a run's predicted rounds (cere_ktest_compose out_rounds, one per poll: kind 0 none (S1 G3: also a poll that
    reports an earlier round) / 1 prefill / 2 decode, plain rows, drafting rows, prompt pending) exercised: the most
    decode rows in one round, decode rounds mixing plain and drafting rows, prefill chunks holding decode rows back,
    and decode rounds per row count. Two only the decode share produces (S1 G4; one decode round per chunk never does):
    a prefill chunk right after another while decode rows are ready (FIFO while two or more prompts wait) and a decode
    round right after another while a prompt is pending (the share of the last waiting prompt)."""
    out = {"max_rows": 0, "mixed_rounds": 0, "prefill_beside_decode": 0, "prefill_run_beside_decode": 0,
           "decode_run_beside_prompt": 0, "decode_rounds": 0, "prefill_rounds": 0}
    last = 0  # the kind of the last round that ran (a kind-0 poll runs none)
    for kind, plain, drafting, prompt in rounds:
        if kind == 2:
            out["decode_rounds"] += 1
            out["max_rows"] = max(out["max_rows"], plain + drafting)
            out["mixed_rounds"] += plain != 0 and drafting != 0
            out["decode_run_beside_prompt"] += last == 2 and prompt != 0
            key = f"rows_{plain + drafting}"
            out[key] = out.get(key, 0) + 1
        elif kind == 1:
            out["prefill_rounds"] += 1
            out["prefill_beside_decode"] += plain + drafting != 0
            out["prefill_run_beside_decode"] += last == 1 and plain + drafting != 0
        last = kind or last
    return out
