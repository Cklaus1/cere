"""D12's speculation cell (tasks/m23-spec.md P11R-16): the torch-free classifier of the CUPTI records of a profiled
speculation pass (test_profile_forward.py `test_profile_spec_*`; CPU tests in tests/test_d12_spec.py).

A record is (start ns, duration ns, name): a kernel, memcpy or memset of the process, in start order.

* Rounds: the caller launches a torch fill (`FillFunctor`) right before each `cere_ktest_session_speculate_mtp` call;
  `rounds` cuts the records at those marks and drops them.
* Phases of a round (CudaDevice::speculate_rows, device/cuda_device.cpp): verify (the ids upload, the target forward
  over the 1 + drafts rows, lm_head and top-2 per row, ending at the scores' device-to-host copy), commit (the
  device-to-device undo restores of a greedy rejection right after it, model/forward.cpp verify_commit; none when
  every draft is accepted), the MTP forwards (CudaDevice::draft: the draft-extend over the committed ids, then the
  decode steps; each starts with the embedding gather and ends at `argmax_map_kernel`, model/mtp.cu) and the tail
  (the proposals copies).
* Groups by kernel name: `k2_dense` (cuBLASLt's GEMMs: every unnamed kernel; and the in-tree K2 kernels, the
  `cere::kernels::` namespaces in K2_NAMESPACES), `lm_head` (a run of dense GEMMs right before a top-2 or the MTP
  argmax), `moe` (`cere::kernels::` other than K2 and `hc_*`: K0 quantize, K7/K13; P3b: `tensorrt_llm` or a grouped
  CUTLASS GEMM, `GroupProblemShape`), `k3` (`hc_*`), `attn_gdn` (K5, K6, K9, P4),
  `sampler` (K1's top-2, K12, K13s, the MTP argmax), `copy` (memcpy, memset), `other` (K1 norms/rope/embedding, K4 PLE,
  the MTP norms, torch).
"""
from __future__ import annotations

import collections
from typing import NamedTuple

Record = tuple[int, int, str]  # start ns, duration ns, name

ROUND_MARK = "FillFunctor"
MTP_END = "argmax_map_kernel"
D2H = "Memcpy DtoH"
D2D = "Memcpy DtoD"
GROUPS = ("k2_dense", "lm_head", "moe", "k3", "attn_gdn", "sampler", "copy", "other")
# The in-tree K2 dense kernels' namespaces under `cere::kernels::`: small-M (P11), gemvx-order rows (P11R-21), MXFP8
# fast (P11R-24), FP8 weights (k2_fp8) and the MXFP8 upcast (k2_mxfp8). test_profile_forward.family reads it too;
# tests/test_d12_spec.py checks it against the executor's sources (O144: k2x, k2f and k2m were counted as MoE).
K2_NAMESPACES = ("k2s", "k2v", "k2x", "k2f", "k2m")


def group(name: str) -> str:
    """The coarse group of one record name (module docstring)."""
    if name.startswith(("Memcpy", "Memset")):
        return "copy"
    if "GroupProblemShape" in name or "tensorrt_llm" in name or "cere::p3b::" in name:
        return "moe"
    if "cere::kernels::" in name:
        if any(f"::{k}::" in name for k in K2_NAMESPACES):
            return "k2_dense"
        return "k3" if "::hc_" in name else "moe"
    if any(f"cere::{k}::" in name for k in ("k5", "k6", "k9", "p4")):
        return "attn_gdn"
    if any(f"cere::{k}::" in name for k in ("k12", "k13s")):
        return "sampler"
    if "cere::k1::" in name:
        return "sampler" if "top2_kernel" in name else "other"
    if "cere::model::mtp::" in name:
        return "sampler" if MTP_END in name else "other"
    if "cere::" in name or "at::native" in name:
        return "other"
    return "k2_dense"


def _ends_head(name: str) -> bool:
    return ("cere::k1::" in name and "top2_kernel" in name) or MTP_END in name


def labelled(records: list[Record]) -> list[Record]:
    """`records` with each name replaced by its group; a run of dense GEMMs that ends right before a K1 top-2 or the
    MTP argmax is lm_head."""
    groups = [group(n) for _, _, n in records]
    for i, (_, _, name) in enumerate(records):
        if not _ends_head(name):
            continue
        j = i - 1
        while j >= 0 and groups[j] == "k2_dense":
            groups[j] = "lm_head"
            j -= 1
    return [(s, d, g) for (s, d, _), g in zip(records, groups)]


def group_ns(records: list[Record]) -> tuple[dict[str, int], collections.Counter]:
    """(summed duration ns, record count) per group present in `records`."""
    ns: dict[str, int] = {}
    count: collections.Counter = collections.Counter()
    for _, d, g in labelled(records):
        ns[g] = ns.get(g, 0) + d
        count[g] += 1
    return ns, count


def rounds(records: list[Record], expected: int) -> list[list[Record]]:
    """The records of each round, cut at the round marks (dropped); refuses another mark count, records before the
    first mark or an empty round."""
    marks = [i for i, (_, _, n) in enumerate(records) if ROUND_MARK in n]
    if len(marks) != expected:
        raise ValueError(f"{len(marks)} round marks, expected {expected}")
    if marks[0] != 0:
        raise ValueError(f"{marks[0]} records before the first round mark")
    out = []
    for a, b in zip(marks, marks[1:] + [len(records)]):
        if b == a + 1:
            raise ValueError(f"round {len(out)} is empty")
        out.append(records[a + 1:b])
    return out


class Phases(NamedTuple):
    verify: list[Record]
    commit: list[Record]
    mtp: list[list[Record]]
    tail: list[Record]


def phases(records: list[Record], forwards: int) -> Phases:
    """One round's records split into verify, commit, the MTP forwards and the tail (module docstring)."""
    names = [n for _, _, n in records]
    v = next((i for i, n in enumerate(names) if n.startswith(D2H)), None)
    if v is None:
        raise ValueError("a round without the verify scores' device-to-host copy")
    c = v + 1
    while c < len(names) and names[c].startswith(D2D):
        c += 1
    ends = [i for i in range(c, len(names)) if MTP_END in names[i]]
    if len(ends) != forwards:
        raise ValueError(f"{len(ends)} MTP forwards, expected {forwards}")
    starts = [c] + [e + 1 for e in ends[:-1]]
    return Phases(records[:v + 1], records[v + 1:c], [records[s:e + 1] for s, e in zip(starts, ends)],
                  records[ends[-1] + 1:])


def busy_ns(records: list[Record]) -> int:
    """The summed record durations."""
    return sum(d for _, d, _ in records)


def span_ns(records: list[Record]) -> int:
    """First start to last end."""
    if not records:
        return 0
    return max(s + d for s, d, _ in records) - records[0][0]


def idle_ns(records: list[Record]) -> int:
    """The span not covered by any record (the host and sync gaps between launches [INFERENCE])."""
    covered, end = 0, None
    for s, d, _ in sorted(records):
        if end is None or s >= end:
            covered += d
            end = s + d
        elif s + d > end:
            covered += s + d - end
            end = s + d
    return span_ns(sorted(records)) - covered


def exclusive_by(items: list[tuple[int, int, str]]) -> dict[str, int]:
    """ns per key of (start ns, duration ns, key) items, each covered instant given to the earliest-started item still
    running (P11R-21: a programmatic dependent launch starts while its predecessor drains and waits for it, so its
    duration holds the predecessor's tail; a summed duration counts that overlap twice). The values add up to the span
    less idle_ns."""
    out: dict[str, int] = {}
    end = None
    for s, d, key in sorted(items):
        lo = s if end is None else max(s, end)
        if s + d > lo:
            out[key] = out.get(key, 0) + s + d - lo
        end = s + d if end is None else max(end, s + d)
    return out


def exclusive_ns(records: list[Record]) -> dict[str, int]:
    """exclusive_by over the records' groups."""
    return exclusive_by(labelled(records))


def copy_idle_ns(records: list[Record]) -> int:
    """The part of idle_ns next to a copy: each gap whose record before (the one reaching furthest so far) or after is
    a memcpy or memset (P11R-20: a CUDA graph's copy nodes against its kernel nodes)."""
    total, end, last = 0, None, ""
    for s, d, name in sorted(records):
        if end is not None and s > end and (group(last) == "copy" or group(name) == "copy"):
            total += s - end
        if end is None or s + d > end:
            end, last = s + d, name
    return total


def gap_ns(before: list[Record], after: list[Record]) -> int:
    """From the last end of `before` to the first start of `after` (both non-empty)."""
    return after[0][0] - max(s + d for s, d, _ in before)


def distinct_experts(rows: list[list[int]], window: int) -> float:
    """The mean number of distinct experts over every `window` consecutive rows (each row: one position's routed
    expert ids of one layer): what one verify step over `window` rows reads, against `window` 1 for plain decode."""
    if not 1 <= window <= len(rows):
        raise ValueError(f"window {window} outside 1..{len(rows)}")
    sizes = [len(set().union(*rows[i:i + window])) for i in range(len(rows) - window + 1)]
    return sum(sizes) / len(sizes)



def segments_of(p: Phases) -> list[tuple[str, list[Record]]]:
    """A round's phases in order: verify, commit, mtp0 .. mtp<n-1>, tail."""
    return [("verify", p.verify), ("commit", p.commit), *[(f"mtp{k}", f) for k, f in enumerate(p.mtp)],
            ("tail", p.tail)]


def round_fields(rounds: list[Phases]) -> dict[str, float]:
    """Per-round means: the round's span, busy and idle ms; per phase its span ms, busy ms, idle ms (and the part of it
    next to a copy, copy_idle_ns), launches, per group us (summed, `_us`, and exclusive_ns, `_excl_us`) and the gap ms
    before it (from the previous non-empty phase's end; none for verify or an empty phase). The phase spans and the
    gaps tile the round's span."""
    acc: dict[str, float] = collections.defaultdict(float)
    for p in rounds:
        acc["span_ms"] += span_ns([r for _, seg in segments_of(p) for r in seg]) / 1e6
        acc["busy_ms"] += busy_ns([r for _, seg in segments_of(p) for r in seg]) / 1e6
        acc["idle_ms"] += idle_ns([r for _, seg in segments_of(p) for r in seg]) / 1e6
        prev: list[Record] | None = None
        for name, seg in segments_of(p):
            acc[f"{name}_launches"] += len(seg)
            acc[f"{name}_span_ms"] += span_ns(seg) / 1e6
            acc[f"{name}_busy_ms"] += busy_ns(seg) / 1e6
            acc[f"{name}_idle_ms"] += idle_ns(seg) / 1e6
            acc[f"{name}_copy_idle_ms"] += copy_idle_ns(seg) / 1e6
            if not seg:
                continue
            if prev is not None:
                acc[f"gap_before_{name}_ms"] += gap_ns(prev, seg) / 1e6
            for g, t in group_ns(seg)[0].items():
                acc[f"{name}_{g}_us"] += t / 1e3
            for g, t in exclusive_ns(seg).items():
                acc[f"{name}_{g}_excl_us"] += t / 1e3
            prev = seg
    return {k: v / len(rounds) for k, v in acc.items()}


def draft_fields(rounds: list[Phases]) -> dict[str, float]:
    """Per-round means over the draft as one window (every MTP forward and the tail: a round of R rows drafts the R
    chains together, one MTP forward per pass over every chain's rows (CudaDevice::draft_rows), then the proposals
    copies and a host sync, tasks/m23-spec.md P11R-21): `draft_span_ms`, `draft_busy_ms`, `draft_idle_ms`,
    `draft_launches` and `draft_<group>_us`."""
    acc: dict[str, float] = collections.defaultdict(float)
    for p in rounds:
        seg = [r for f in p.mtp for r in f] + p.tail
        acc["draft_launches"] += len(seg)
        acc["draft_span_ms"] += span_ns(seg) / 1e6
        acc["draft_busy_ms"] += busy_ns(seg) / 1e6
        acc["draft_idle_ms"] += idle_ns(seg) / 1e6
        for g, t in group_ns(seg)[0].items():
            acc[f"draft_{g}_us"] += t / 1e3
    return {k: v / len(rounds) for k, v in acc.items()}