"""D1 teacher-forced KL calculus (tasks/m214-spec.md D1; C1).

Pure Python in fp64 (no torch, no numpy, no CUDA; neither numpy nor torch is on the CI host): the GPU case
`test_teacher_forced_kl` of `test_numerics_b8_layers.py` computes the per-row statistics on the device (logits =
`model` rows x lm_head^T accumulated in fp32, log-softmax in fp64) and hands them here as `RowStats`; `row_stats`
below is the same per-row definition over host logits, which the CPU test `tests/test_numerics_kl_offline.py` drives
on synthetic arrays. Nothing here sees a token id or text: a row's top-1 ids are compared for equality only.

Conventions:
* Every distribution pair is (reference, candidate) = (SGLang's teacher-forced pass, the other side): KL(ref || cand).
  Arm (a): candidate = native `forward_trace`; yardstick (y): candidate = SGLang's prompt pass, rows [0, P).
* Row r predicts token r + 1. Prompt rows are [0, P), completion rows [P, T); row P - 1 is the `last-prompt-row`
  (the first-token row). Only prompt rows 0 .. P - 2 have a target (the prompt's own next token) for NLL.
* Labels are descriptive, never gates and never a G1 verdict; a NaN anywhere fails (KLError).
"""
from __future__ import annotations

import math
import statistics
from typing import NamedTuple

YARD_FLOOR = 1e-6  # nats: a yard KL mean below it is replaced by it and the row carries yard_zero
RATIO_LIMIT = 2.0  # a CERE choice (Strata kept MMQ at 1.15)
FLIP_MARGIN = 0.5  # nats: flips whose SGLang top-2 margin is below it are counted apart
SE_FACTOR = 2.0
BETTER = "better"
WORSE = "worse-approximation"
ROUNDING = "rounding-class"
LARGE = "different-rounding-large"
LABELS = (BETTER, WORSE, ROUNDING, LARGE)


class KLError(AssertionError):
    """A NaN or malformed input: the D1 measure fails (an AssertionError, so unittest reports a FAIL)."""


class RowStats(NamedTuple):
    kl: float  # KL(ref || cand) in nats
    ref_top1: int
    cand_top1: int
    ref_margin: float  # top-1 minus top-2 log-prob, nats
    cand_margin: float
    ref_nll: float | None = None  # -log p(target); None for a row without a target
    cand_nll: float | None = None


def _finite(value: float, what: str) -> float:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise KLError(f"{what} is {value!r}")
    return value


def log_softmax(logits) -> list:
    """fp64 log-softmax of one logit row."""
    xs = [_finite(v, "logit") for v in logits]
    if not xs:
        raise KLError("empty logit row")
    top = max(xs)
    lse = top + math.log(math.fsum(math.exp(v - top) for v in xs))
    return [v - lse for v in xs]


def _top2(lp: list) -> tuple:
    if len(lp) < 2:
        raise KLError("a distribution needs two tokens for a top-2 margin")
    first = max(range(len(lp)), key=lambda i: (lp[i], -i))
    second = max((i for i in range(len(lp)) if i != first), key=lambda i: (lp[i], -i))
    return first, lp[first] - lp[second]


def row_stats(ref_logits, cand_logits, target: int | None = None) -> RowStats:
    """The per-row definition over host logits (the GPU case computes the same on the device)."""
    ref, cand = log_softmax(ref_logits), log_softmax(cand_logits)
    if len(ref) != len(cand):
        raise KLError(f"vocab {len(cand)} vs reference {len(ref)}")
    kl = math.fsum(math.exp(r) * (r - c) for r, c in zip(ref, cand))
    ref_top1, ref_margin = _top2(ref)
    cand_top1, cand_margin = _top2(cand)
    nll = (None, None) if target is None else (-ref[target], -cand[target])
    return RowStats(kl, ref_top1, cand_top1, ref_margin, cand_margin, *nll)


def p99(values: list) -> float:
    """Nearest-rank 99th percentile."""
    ordered = sorted(values)
    return ordered[max(math.ceil(0.99 * len(ordered)) - 1, 0)]


def paired(diffs: list) -> tuple:
    """(mean, standard error) of paired differences; SE = sample stdev / sqrt(n), NaN (a failure) below 2 rows."""
    if not diffs:
        raise KLError("no paired NLL rows")
    mean = math.fsum(diffs) / len(diffs)
    se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) >= 2 else math.nan
    return mean, se


def label(ratio: float, dnll: float, se: float) -> str:
    """D1's labels, assigned in order; total over every (ratio, dNLL) quadrant."""
    if dnll < -SE_FACTOR * se:
        return BETTER
    if dnll > SE_FACTOR * se:
        return WORSE
    return ROUNDING if ratio <= RATIO_LIMIT else LARGE


def _check(rows: list, what: str) -> None:
    for k, r in enumerate(rows):
        for field in ("kl", "ref_margin", "cand_margin"):
            _finite(getattr(r, field), f"{what} row {k} {field}")
        for field in ("ref_nll", "cand_nll"):
            if getattr(r, field) is not None:
                _finite(getattr(r, field), f"{what} row {k} {field}")


def _kl(rows: list) -> dict:
    kls = [r.kl for r in rows]
    return {"mean": math.fsum(kls) / len(kls), "median": statistics.median(kls), "p99": p99(kls)}


def prompt_summary(prompt_rows: list, completion_rows: list, yard_rows: list) -> dict:
    """One prompt: arm (a) prompt rows [0, P) and completion rows [P, T), yardstick rows [0, P) -> the per-prompt
    fields, the `last-prompt-row` fields and the label (numbers only)."""
    if not prompt_rows or len(yard_rows) != len(prompt_rows):
        raise KLError(f"{len(prompt_rows)} prompt rows, {len(yard_rows)} yardstick rows")
    if not completion_rows:
        raise KLError("no completion rows")
    _check(prompt_rows, "prompt")
    _check(completion_rows, "completion")
    _check(yard_rows, "yardstick")
    targeted = prompt_rows[:-1]
    if any(r.ref_nll is None or r.cand_nll is None for r in targeted):
        raise KLError("a prompt row 0 .. P-2 has no NLL")
    if prompt_rows[-1].ref_nll is not None or any(r.ref_nll is not None for r in completion_rows):
        raise KLError("NLL is taken over prompt rows 0 .. P-2 only")
    dnll, se = paired([r.cand_nll - r.ref_nll for r in targeted])
    _finite(dnll, "dNLL")
    _finite(se, f"dNLL SE over {len(targeted)} row(s)")
    kl = _kl(prompt_rows)
    yard = _kl(yard_rows)["mean"]
    yard_zero = yard < YARD_FLOOR
    ratio = kl["mean"] / (YARD_FLOOR if yard_zero else yard)
    flips = [r for r in prompt_rows if r.ref_top1 != r.cand_top1]
    same = len(prompt_rows) - len(flips)
    completion = _kl(completion_rows)
    last = prompt_rows[-1]
    return {"rows": len(prompt_rows), "completion_rows": len(completion_rows),
            "same_top1_pct": 100.0 * same / len(prompt_rows),
            "kl_mean": kl["mean"], "kl_median": kl["median"], "kl_p99": kl["p99"],
            "completion_kl_mean": completion["mean"], "completion_kl_median": completion["median"],
            "completion_kl_p99": completion["p99"],
            "yard_kl_mean": yard, "yard_zero": yard_zero, "ratio": ratio,
            "flips": len(flips), "flips_low_margin": sum(r.ref_margin < FLIP_MARGIN for r in flips),
            "nll_rows": len(targeted), "dnll": dnll, "dnll_se": se, "label": label(ratio, dnll, se),
            "last": {"kl": last.kl, "ref_margin": last.ref_margin, "native_margin": last.cand_margin,
                     "top1_equal": last.ref_top1 == last.cand_top1}}


def aggregate(prompts: list) -> dict:
    """The aggregate over every prompt's rows: `prompts` is a list of (prompt rows, completion rows, yard rows)."""
    if not prompts:
        raise KLError("no prompts")
    rows = [r for p, _, _ in prompts for r in p]
    completion = [r for _, c, _ in prompts for r in c]
    yard_rows = [r for _, _, y in prompts for r in y]
    summaries = [prompt_summary(*p) for p in prompts]
    dnll, se = paired([r.cand_nll - r.ref_nll for p, _, _ in prompts for r in p[:-1]])
    kl = _kl(rows)
    yard = _kl(yard_rows)["mean"]
    yard_zero = yard < YARD_FLOOR
    ratio = kl["mean"] / (YARD_FLOOR if yard_zero else yard)
    flips = [r for r in rows if r.ref_top1 != r.cand_top1]
    return {"prompts": len(prompts), "rows": len(rows), "completion_rows": len(completion),
            "same_top1_pct": 100.0 * (len(rows) - len(flips)) / len(rows),
            "kl_mean": kl["mean"], "kl_median": kl["median"], "kl_p99": kl["p99"],
            "completion_kl_mean": _kl(completion)["mean"], "yard_kl_mean": yard, "yard_zero": yard_zero,
            "ratio": ratio, "flips": len(flips), "flips_low_margin": sum(r.ref_margin < FLIP_MARGIN for r in flips),
            "nll_rows": sum(s["nll_rows"] for s in summaries), "dnll": dnll, "dnll_se": se,
            "label": label(ratio, dnll, se),
            "last_prompt_row_flips": sum(not s["last"]["top1_equal"] for s in summaries),
            **{f"labels_{lab.replace('-', '_')}": sum(s["label"] == lab for s in summaries) for lab in LABELS}}
