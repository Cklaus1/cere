"""D9Q quality-gate calculus (tasks/m23-spec.md D9Q (b), [REVISED: M23-D9Q-b3v2], [REVISED: M23-D9Q-b2v3],
[REVISED: M23-D9Q-screen], [REVISED: M23-D9Q-b3v3], [REVISED: owner 2026-10-16 screen tiers]).

Pure Python in fp64 (no torch, no numpy, no CUDA): the GPU suite `test_quality_gate.py` builds `kl_measure.RowStats`
per row (reference R = P's teacher-forced prefill arm, candidate = one lever set's arm), grouped per prompt, and hands
them here; `native_dev.py quality-baseline` and `quality-verdict` replay the gate from the suite's report rows. CPU
tests: `tests/test_quality_offline.py`.

Conventions (D9Q-b2v3, pre-registered at PREREG_COMMIT before any quality_heldout_v2 GPU data; never retuned):
* Arms: `pre` (teacher-forced prefill) and `dec` (forced DECODE) over the same rows; row r predicts token r + 1.
  Every RowStats is (R, X.arm): kl = KL(R || X.arm), ref_* are R's, cand_* are X.arm's.
* Parts: `main` (the selected shape's prompts, clustered by source file or directory) and `long` (the 16-prompt
  long-context suite, never pooled with main).
* (b1) mean KL(R || C) / max(mean KL(R || P), Y) <= RATIO_LIMIT, Y = mean KL(R || P.dec) of the part.
  (b2v3) d_i = mean over prompt i's targeted rows of NLL(C) - NLL(P); CR1 cluster-robust variance
  V = G/(G-1) sum_g (S_g - n_g mean)^2 / n^2, UCB = mean + t(0.95, G-1) sqrt(V); PASS iff UCB < DELTA on both arms.
  (b3 v3, `b3`) confident flips (R's top-2 margin >= FLIP_MARGIN), one-sided exact binomial test at ALPHA. A
  prefill-neutral C (its pre arm equals R: KL 0, no flip; a decode-only lever) keeps b3 v2 on both arms: the
  discordant flips, C only vs P only, against pi = 1/2. Any other C: pre arm cf(C.pre) against cf(P.dec) (the
  yardstick run Y; P.pre is R, so P has no pre flip of its own), dec arm the discordant flips, each against
  pi = B3V3_PI = rho / (1 + rho), rho = RATIO_LIMIT (b1's tolerance carried to flip counts).
* Long suite (detection-only, n = k <= LONG_PROMPTS prompts, S1f): BLOCK iff mean - t(0.95, k-1) s/sqrt(k) > DELTA on
  either arm, else INCONCLUSIVE iff mean > LONG_INCONCLUSIVE on either arm, else `long_clear(k)`.
* The v1 b2 rule (row-level iid SE) stays for replaying the retained v1 verdicts only (`b2_v1`).
* Screen ([REVISED: M23-D9Q-screen], owner decision 2026-10-15, pre-registered at SCREEN_PREREG_COMMIT): a candidate
  run on SCREEN_N shape-SHAPE prompts (`screen_select`, stratified by source file) plus the whole long suite and every
  control; per arm the same CR1 bound: FAIL iff LCB > SCREEN_FAIL_LCB on an arm, PASS iff UCB < SCREEN_PASS_UCB with
  b1 and b3 on both arms, else ESCALATE to the full b2v3 run, which alone decides (b) then.
* Screen tiers ([REVISED: owner 2026-10-16 screen tiers]; only output-changing levers are screened, bit-identical
  ones are covered by identity): triage per candidate on TRIAGE_N prompts (`triage_select`, the screen's bound and
  thresholds; FAIL drops it, PASS or ESCALATE keeps it; mode `triage`, never a verdict), then the SCREEN_N screen of
  the bundle (every kept candidate on together) decides; on a bundle FAIL or ESCALATE the bundle's top-K prompts by
  dNLL rerun per candidate (mode `subset`, `attribution`: diagnostic only, never a verdict).
Nothing here sees a token id or text: top-1 ids are compared for equality only and never returned.
"""
from __future__ import annotations

import hashlib
import math
import random
import statistics

import kl_measure

ROW = "D9Q"
ARMS = ("pre", "dec")
PARTS = ("main", "long")
SIDES = ("candidate", "predecessor")
RATIO_LIMIT = 2.0
FLIP_MARGIN = 0.5  # nats: R's top-2 margin at or above which a top-1 flip is confident
ALPHA = 0.01  # b3: one-sided level of the exact binomial test on the confident flips (v2 and v3)
DELTA = 0.01  # b2v3: nats/token per arm (owner 2026-10-15)
B2V3_ALPHA = 0.05  # one-sided level of the cluster-robust t bound
DE_RATIO = 1.2  # directory clusters iff DE_dir > DE_RATIO x DE_file ("materially larger")
LONG_PROMPTS = 16  # the long suite's target size (8 + 8); S1f: it runs at the k <= 16 that pass, labelled n=k
LONG_INCONCLUSIVE = 0.005  # nats/token: delta / 2
LONG_MIN = 2  # the prompt-level bounds need two prompts
FRAMING_NLL_MAX = 4.0  # nats/token: R's mean NLL over the scored rows
FRAMING_SPECIAL_MAX = 0.10  # share of scored rows whose R top-1 is a special or added token
SEED = 20261015
SIM_REPLICATES = 10_000
OUTER = 200
INNER = 1_000
SHIFTS = (0.0, 0.005, 0.01, 0.05)
FAIL_POWER = 0.95  # calibration check: a shape qualifies iff the upper end of P(FAIL | +0.01)'s interval >= it
TIE = 0.05  # |P_S(PASS | 0) - P_L(PASS | 0)| below it is a tie
PASS_FLOOR = 0.5  # both shapes below it: REPORT to the owner
SHAPES = ("L", "S")
TIE_SHAPE = "L"
SHAPE_MIN_PROMPTS = {"S": 5_200}  # S is eligible only with this many valid prompts
SE_FACTOR, DNLL_FLOOR, DNLL_LIMIT = 2.0, 0.001, 0.005  # the v1 b2 rule, replay only (`b2_v1`)
PROMPT_SET = "quality_heldout_v2"
# [REVISED: M23-D9Q-b3v3] "gate changed after a miss" (owner 2026-10-16; b3 v2, fixed at b1f97f3e before any held-out
# run, failed every prefill-changing set in the pre arm by construction, O93): B3V3_PREREG_COMMIT is b3 v3's
# pre-registration commit (spec row D9Q-b3v3 with its calibration report), pinned by the commit after it; every verdict
# records it.
B3V3_PI = (2, 3)  # rho / (1 + rho), rho = RATIO_LIMIT, as an exact fraction
B3V3_PREREG_COMMIT: str | None = "d3bd9cdacfdddf36195423f48e10fd4f0b7a98f4"
# the last pre-registration commit (f7edf698 S1, 47d3e32b S1c in full, 88c1fefa S1d calibration check, aefef4db S1e
# idiom-filtered P3 and long refill, b3dfd868 S1f refill from the main files' free runs, long suite at n=k), before
# any v2 data
PREREG_COMMIT = "b3dfd868818df8995f0d45c104e7ddfe8de0ee0d"
# The shape decision (`shape_decision` over both baselines), committed after the baseline and before any candidate:
# quality_baseline-20261015T105120Z (S4, db2b1f61): shape L (S ineligible), cluster level file (rule 3)
SHAPE: str | None = "L"
CLUSTER_LEVEL: str | None = "file"
SUMMARY_KEYS = ("rows", "kl_mean", "kl_median", "kl_p99", "same_top1_pct", "flips", "confident_flips", "nll_rows",
                "nll_mean")
PAIRED_KEYS = ("nll_rows", "dnll", "dnll_se", "confident_flips_candidate_only", "confident_flips_predecessor_only")
PROMPT_KEYS = ("part", "file", "dir", "x")  # a `prompt <id>` row also holds `<set>_<arm>`: that prompt's d_i
FRAMING_KEYS = ("rows", "ref_nll_mean", "special_top1_share")
LEVELS = ("file", "dir")
NEGATIVE = "neg_bf16_partial"  # must FAIL the main gate
SENSITIVITY = "neg_bf16_logits"  # reported beside it, not required to fail
INJ_TARGET = "inj_target_0p05"  # P's rows, target probability scaled by e^-INJ_C: must FAIL b2v3, BLOCK the long suite
INJ_C = 0.05
INJ_SPREAD = {"inj_spread_0p05": 0.05, "inj_spread_0p01": 0.01}  # e_f x_i + c; 0p05 must FAIL b2v3, 0p01 report-only
INJ_SPREAD_REQUIRED = "inj_spread_0p05"
SELF_CONSISTENCY = "p_repeat"  # a second null-lever model: must PASS the main gate, long suite clear
CONTROL_SETS = (NEGATIVE, SENSITIVITY, INJ_TARGET, SELF_CONSISTENCY)
D9_SERIES_RUNS = 3
FIRST_DIVERGENCE = "first divergence earlier"  # D9's criterion that D9Q makes report-only
# [REVISED: M23-D9Q-screen] (owner decision 2026-10-15): the 300-prompt screen with escalation; the full run is used
# only for grey-zone results. SCREEN_PREREG_COMMIT is the screen's pre-registration commit (spec row D9Q-screen),
# pinned by the commit after it; every screen report records it and every screen verdict prints it.
SCREEN_N = 300
SCREEN_SEED = 20261015
SCREEN_TAG = "D9Q-screen"  # the hash domain of `screen_select`
SCREEN_MODE = "screen"  # the corpus row's mode of a screen run (a full candidate run is "candidate")
SCREEN_PASS_UCB = 0.005  # nats/token: PASS iff UCB < it on both arms (with b1 and b3)
SCREEN_FAIL_LCB = DELTA  # FAIL iff LCB > it on either arm
SCREEN_PREREG_COMMIT: str | None = "122e1c4ec9489741cdebb736766593d4cc532d49"
# [REVISED: owner 2026-10-16 screen tiers]: triage per candidate (TRIAGE_N prompts, the screen's seed, bound and
# thresholds; a triage report never satisfies a verdict), the SCREEN_N screen on the bundle, and on a bundle FAIL or
# ESCALATE the attribution rerun of the bundle's top ATTRIBUTION_K prompts per candidate (diagnostic only).
TRIAGE_N = 50
TRIAGE_MODE = "triage"  # the corpus row's mode of a triage run
TRIAGE_DECISION = {"FAIL": "DROP", "ESCALATE": "KEEP", "PASS": "KEEP"}
SUBSET_MODE = "subset"  # the corpus row's mode of a prompt-id subset run (the attribution rerun)
ATTRIBUTION_K = 16
# fp4local design note D11, FL-16 (S2 tier 1, report-only: a checkpoint swap's corpus is P's own completions): a
# candidate checkpoint as the set instead of a lever set. The set is named by the checkpoint; its model has null levers
# and is read from the directory in CANDIDATE_DIR_ENV (native_dev.py mounts it read-only), placed by the families file
# CHECKPOINTS names (repository-relative). P stays the /model checkpoint.
CANDIDATE_DIR_ENV = "CERE_QUALITY_CANDIDATE_DIR"
CHECKPOINTS = {"fp4local": "configs/artifacts/qwen4_exp-nvfp4-plenvfp4.families.json"}
# k2-prefill design note G4 (opt-in): the frame's `reference` names P's checkpoint instead of the pinned one;
# native_dev.py mounts it at /model, and P, SELF_CONSISTENCY, the controls and the lever sets are placed by the
# families file it names here (s3: the S3 checkpoint, whose MXFP8 dense weights the k2_gemm small_m_a8 lever reads).
REFERENCES = {"s3": "configs/artifacts/qwen4_exp-nvfp4-plenvfp4-mxfp8.families.json"}


class QualityError(AssertionError):
    """Malformed or misaligned input: the D9Q measure fails (an AssertionError, so unittest reports a FAIL)."""


# -- Student t (no scipy) ---------------------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularized incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10_000):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            return h
    raise QualityError(f"incomplete beta ({a}, {b}, {x}) did not converge")


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, nu: float) -> float:
    """P(T <= t) for Student's t with nu degrees of freedom."""
    if not nu > 0:
        raise QualityError(f"t_cdf: {nu} degrees of freedom")
    tail = 0.5 * _betainc(nu / 2.0, 0.5, nu / (nu + t * t))
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p: float, nu: float) -> float:
    """The p quantile of Student's t (bisection on `t_cdf` to 1e-12)."""
    if not 0.0 < p < 1.0:
        raise QualityError(f"t_quantile: p {p}")
    if p < 0.5:
        return -t_quantile(1.0 - p, nu)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, nu) < p:
        hi *= 2.0
    while hi - lo > 1e-12:
        mid = (lo + hi) / 2.0
        if t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# -- the cluster-robust bound, ICC and design effect -----------------------------------------------------------
def _finite_list(values: list, what: str) -> list:
    return [kl_measure._finite(v, what) for v in values]


def clusters(d: list, labels: list) -> list:
    """[(n_g, S_g)] per cluster label, in order of first appearance."""
    if len(d) != len(labels):
        raise QualityError(f"{len(d)} values, {len(labels)} cluster labels")
    at: dict = {}
    out: list = []
    for v, g in zip(d, labels):
        if g not in at:
            at[g] = len(out)
            out.append([0, 0.0])
        out[at[g]][0] += 1
        out[at[g]][1] += v
    return [(n, s) for n, s in out]


def cr_variance(sums: list) -> tuple:
    """(mean, CR1 variance of the mean) from cluster sums [(n_g, S_g)]."""
    g = len(sums)
    if g < 2:
        raise QualityError(f"a cluster-robust bound needs at least 2 clusters, got {g}")
    n = sum(k for k, _ in sums)
    mean = math.fsum(s for _, s in sums) / n
    v = g / (g - 1) * math.fsum((s - k * mean) ** 2 for k, s in sums) / (n * n)
    return mean, v


def cr_bound(d: list, labels: list) -> dict:
    """Rule (1) of D9Q-b2v3 for per-prompt values `d` clustered by `labels`: mean, CR1 SE, t(0.95, G-1) UCB, the
    two-sided 95 % CI (t(0.975, G-1)) and the perplexity change exp(.) - 1 of each."""
    d = _finite_list(d, "per-prompt dNLL")
    sums = clusters(d, labels)
    mean, v = cr_variance(sums)
    g = len(sums)
    se = math.sqrt(v)
    t95, t975 = t_quantile(1.0 - B2V3_ALPHA, g - 1), t_quantile(0.975, g - 1)
    lo, hi = mean - t975 * se, mean + t975 * se
    return {"n": len(d), "clusters": g, "mean": mean, "sd": statistics.stdev(d) if len(d) > 1 else 0.0, "se": se,
            "t95": t95, "ucb": mean + t95 * se, "ci_lo": lo, "ci_hi": hi, "ppl": math.expm1(mean),
            "ppl_lo": math.expm1(lo), "ppl_hi": math.expm1(hi)}


def icc(d: list, labels: list) -> float:
    """One-way ANOVA intraclass correlation (MSB - MSW) / (MSB + (n0 - 1) MSW), unequal cluster sizes; 0.0 when
    both mean squares are zero."""
    groups: dict = {}
    for v, g in zip(d, labels):
        groups.setdefault(g, []).append(v)
    k, n = len(groups), len(d)
    if k < 2 or n <= k:
        raise QualityError(f"ICC needs at least 2 clusters and more prompts than clusters ({n} prompts, {k} clusters)")
    mean = math.fsum(d) / n
    ssb = math.fsum(len(xs) * (math.fsum(xs) / len(xs) - mean) ** 2 for xs in groups.values())
    ssw = math.fsum((x - math.fsum(xs) / len(xs)) ** 2 for xs in groups.values() for x in xs)
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - math.fsum(len(xs) ** 2 for xs in groups.values()) / n) / (k - 1)
    den = msb + (n0 - 1) * msw
    return 0.0 if den == 0.0 else (msb - msw) / den


def design_effect(d: list, labels: list) -> float:
    """V_CR1 / (s^2 / n); 1.0 when the values do not vary."""
    _, v = cr_variance(clusters(d, labels))
    s2 = statistics.variance(d)
    return 1.0 if s2 == 0.0 else v / (s2 / len(d))


def dependence(d: list, files: list, dirs: list) -> dict:
    """ICC, design effect and cluster count at file and directory level. The ICC is None (undefined, report-only)
    when a level has fewer than 2 clusters or no cluster with two prompts: a triage's 50 prompts are often 50 files
    (tasks/fp4local-spec.md FL-16, S2 tier 1). Nothing gates on the ICC; rule (3) reads the design effects."""
    return {level: {"icc": icc(d, labels) if 2 <= len(set(labels)) < len(d) else None,
                    "design_effect": design_effect(d, labels),
                    "clusters": len(set(labels))} for level, labels in (("file", files), ("dir", dirs))}


def _icc_text(v: float | None, digits: int) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def cluster_level(dep: dict) -> str:
    """Rule (3): directories iff DE_dir > DE_RATIO x DE_file, else files."""
    return "dir" if dep["dir"]["design_effect"] > DE_RATIO * dep["file"]["design_effect"] else "file"


def b2v3(d: list, labels: list) -> dict:
    out = cr_bound(d, labels)
    out["b2"] = out["ucb"] < DELTA
    return out


def long_suite(d: list) -> dict:
    """One arm of the long suite: prompt-level t, one-sided 95 % lower bound."""
    d = _finite_list(d, "long-suite dNLL")
    if len(d) < 2:
        raise QualityError(f"the long suite needs at least 2 prompts, got {len(d)}")
    mean, sd = math.fsum(d) / len(d), statistics.stdev(d)
    lcb = mean - t_quantile(1.0 - B2V3_ALPHA, len(d) - 1) * sd / math.sqrt(len(d))
    return {"n": len(d), "mean": mean, "sd": sd, "lcb": lcb, "ppl": math.expm1(mean), "block": lcb > DELTA,
            "inconclusive": mean > LONG_INCONCLUSIVE}


def long_clear(n: int) -> str:
    """The long suite's clear wording (never "equivalent" or "safe")."""
    return f"no regression detected (detection-only, n={n})"


def long_outcome(arms: dict) -> str:
    """BLOCK iff either arm's lower bound > DELTA; else INCONCLUSIVE iff either mean > LONG_INCONCLUSIVE; else clear."""
    if any(arms[a]["block"] for a in ARMS):
        return "BLOCK"
    if any(arms[a]["inconclusive"] for a in ARMS):
        return "INCONCLUSIVE"
    return long_clear(arms[ARMS[0]]["n"])


# -- the v1 rule, replay only ---------------------------------------------------------------------------------
def b2_v1(dnll: float, se: float) -> dict:
    """The v1 b2 rule (row-level iid SE): dNLL <= min(max(SE_FACTOR SE, DNLL_FLOOR), DNLL_LIMIT)."""
    bound = min(max(SE_FACTOR * se, DNLL_FLOOR), DNLL_LIMIT)
    return {"dnll": dnll, "dnll_se": se, "dnll_bound": bound, "b2": dnll <= bound}


def replay_b2_v1(report: dict, name: str) -> dict:
    """{arm: b2_v1} from the `{name} {arm} paired` rows of a quality_heldout_v1 suite report."""
    out = {}
    for arm in ARMS:
        rows = [r for r in report.get("rows") or [] if r.get("row") == ROW and r.get("case") == f"{name} {arm} paired"]
        if len(rows) != 1:
            raise QualityError(f"report has {len(rows)} rows {name} {arm} paired")
        out[arm] = b2_v1(rows[0]["dnll"], rows[0]["dnll_se"])
    return out


# -- per-arm statistics ----------------------------------------------------------------------------------------
def arm_summary(rows: list) -> dict:
    """Aggregates of one (lever set, arm) against R: scalars only."""
    if not rows:
        raise QualityError("no rows")
    for k, r in enumerate(rows):
        kl_measure._finite(r.kl, f"row {k} kl")
        kl_measure._finite(r.ref_margin, f"row {k} ref_margin")
        if r.cand_nll is not None:
            kl_measure._finite(r.cand_nll, f"row {k} cand_nll")
    kls = [r.kl for r in rows]
    flips = [r for r in rows if r.ref_top1 != r.cand_top1]
    nll = [r.cand_nll for r in rows if r.cand_nll is not None]
    return {"rows": len(rows), "kl_mean": math.fsum(kls) / len(kls), "kl_median": statistics.median(kls),
            "kl_p99": kl_measure.p99(kls), "same_top1_pct": 100.0 * (len(rows) - len(flips)) / len(rows),
            "flips": len(flips), "confident_flips": sum(r.ref_margin >= FLIP_MARGIN for r in flips),
            "nll_rows": len(nll), "nll_mean": math.fsum(nll) / len(nll) if nll else 0.0}


def exact_tail(b: int, n: int, pi: tuple = (1, 2)) -> float:
    """P(X >= b) for X ~ Binomial(n, pi), pi = num / den, exact (integer sums; 1.0 for n = 0)."""
    num, den = pi
    if not 0 <= b <= n or not 0 < num < den:
        raise QualityError(f"exact_tail({b}, {n}, {pi}): need 0 <= b <= n and 0 < pi < 1")
    return sum(math.comb(n, k) * num ** k * (den - num) ** (n - k) for k in range(b, n + 1)) / den ** n


def paired(cand_prompts: list, pred_prompts: list) -> dict:
    """Per prompt (one RowStats list per prompt, both against one R over the same rows): `d` the per-prompt mean
    dNLL (b2v3), the row-level dNLL with its iid SE (report-only) and the discordant confident flips (b3).
    Misaligned prompts or rows, or a prompt without a target, fail (QualityError)."""
    if len(cand_prompts) != len(pred_prompts):
        raise QualityError(f"{len(cand_prompts)} candidate prompts, {len(pred_prompts)} predecessor prompts")
    diffs, d = [], []
    cand_only = pred_only = 0
    for i, (cand_rows, pred_rows) in enumerate(zip(cand_prompts, pred_prompts)):
        if len(cand_rows) != len(pred_rows):
            raise QualityError(f"prompt {i}: {len(cand_rows)} candidate rows, {len(pred_rows)} predecessor rows")
        mine = []
        for k, (c, p) in enumerate(zip(cand_rows, pred_rows)):
            if c.ref_top1 != p.ref_top1 or c.ref_nll != p.ref_nll or c.ref_margin != p.ref_margin:
                raise QualityError(f"prompt {i} row {k}: the candidate and the predecessor rows are not against one "
                                   "reference row")
            if (c.cand_nll is None) != (p.cand_nll is None):
                raise QualityError(f"prompt {i} row {k}: targeted on one side only")
            if c.cand_nll is not None:
                mine.append(c.cand_nll - p.cand_nll)
            if c.ref_margin >= FLIP_MARGIN:
                c_flip, p_flip = c.cand_top1 != c.ref_top1, p.cand_top1 != p.ref_top1
                cand_only += c_flip and not p_flip
                pred_only += p_flip and not c_flip
        if not mine:
            raise QualityError(f"prompt {i}: no targeted row")
        diffs += mine
        d.append(math.fsum(mine) / len(mine))
    dnll, se = kl_measure.paired(diffs)
    kl_measure._finite(dnll, "dNLL")
    kl_measure._finite(se, f"dNLL SE over {len(diffs)} row(s)")
    return {"nll_rows": len(diffs), "dnll": dnll, "dnll_se": se, "d": d,
            "confident_flips_candidate_only": cand_only, "confident_flips_predecessor_only": pred_only}


def arm_input(cand_prompts: list, pred_prompts: list) -> dict:
    """One arm's gate input: both summaries (over every row) and the paired statistics (per prompt)."""
    return {"candidate": arm_summary([r for rows in cand_prompts for r in rows]),
            "predecessor": arm_summary([r for rows in pred_prompts for r in rows]),
            "paired": paired(cand_prompts, pred_prompts)}


def null_values(pred_dec_prompts: list) -> list:
    """x_i = mean over prompt i's targeted rows of NLL(P.dec) - NLL(R): P's own decode-vs-prefill difference."""
    out = []
    for i, rows in enumerate(pred_dec_prompts):
        x = [r.cand_nll - r.ref_nll for r in rows if r.cand_nll is not None]
        if not x:
            raise QualityError(f"prompt {i}: no targeted row")
        out.append(math.fsum(x) / len(x))
    return out


def framing(ref_rows: list, special_ids) -> dict:
    """R's mean NLL over the scored rows and the share of rows whose R top-1 is in `special_ids`."""
    nll = [r.ref_nll for r in ref_rows if r.ref_nll is not None]
    if not nll:
        raise QualityError("framing: no targeted row")
    return {"rows": len(ref_rows), "ref_nll_mean": math.fsum(nll) / len(nll),
            "special_top1_share": sum(r.ref_top1 in special_ids for r in ref_rows) / len(ref_rows)}


def framing_ok(f: dict) -> bool:
    return f["ref_nll_mean"] <= FRAMING_NLL_MAX and f["special_top1_share"] <= FRAMING_SPECIAL_MAX


# -- the gates -----------------------------------------------------------------------------------------------
def prefill_neutral(arms: dict) -> bool:
    """C's pre arm equals R (KL 0, no top-1 flip): C changes nothing before decode (a decode-only lever)."""
    pre = arms["pre"]["candidate"]
    return pre["kl_mean"] == 0.0 and pre["flips"] == 0


def b3(arms: dict, arm: str) -> dict:
    """b3 v3 on `arm`: (b, c) and pi of the one-sided exact binomial test, FAIL iff P(X >= b; n = b + c, pi) < ALPHA.
    Prefill-neutral C: v2 (discordant flips C only vs P only, pi 1/2). Else pre: cf(C.pre) vs cf(P.dec), dec: the
    discordant flips, both at B3V3_PI."""
    pair = arms[arm]["paired"]
    b, c = pair["confident_flips_candidate_only"], pair["confident_flips_predecessor_only"]
    versus = "C only vs P only"
    if prefill_neutral(arms):
        rule, pi = "v2", (1, 2)
    else:
        rule, pi = "v3", B3V3_PI
        if arm == "pre":
            b, c = arms["pre"]["candidate"]["confident_flips"], arms["dec"]["predecessor"]["confident_flips"]
            versus = "C.pre vs P.dec"
    p = exact_tail(b, b + c, pi)
    return {"b3_rule": rule, "b3_versus": versus, "b3_pi": list(pi), "b3_b": b, "b3_c": c, "flip_p": p,
            "b3": p >= ALPHA}


def _b1_b3(arms: dict, arm: str, yardstick: float) -> dict:
    cand, pred, pair = arms[arm]["candidate"], arms[arm]["predecessor"], arms[arm]["paired"]
    scale = max(pred["kl_mean"], yardstick)
    ratio = cand["kl_mean"] / scale
    return {"kl_candidate": cand["kl_mean"], "kl_predecessor": pred["kl_mean"], "yardstick": yardstick,
            "ratio": ratio, "kl_bound": RATIO_LIMIT * scale, "b1": ratio <= RATIO_LIMIT,
            "row_dnll": pair["dnll"], "row_dnll_se": pair["dnll_se"], "nll_rows": pair["nll_rows"],
            "confident_flips_candidate": cand["confident_flips"], "confident_flips_predecessor": pred["confident_flips"],
            "flips_candidate_only": pair["confident_flips_candidate_only"],
            "flips_predecessor_only": pair["confident_flips_predecessor_only"], **b3(arms, arm),
            "same_top1_pct_candidate": cand["same_top1_pct"], "same_top1_pct_predecessor": pred["same_top1_pct"]}


def _yardstick(arms: dict) -> float:
    if set(arms) != set(ARMS):
        raise QualityError(f"arms {sorted(arms)}, expected {list(ARMS)}")
    y = arms["dec"]["predecessor"]["kl_mean"]
    if not y > 0.0:
        raise QualityError(f"yardstick KL(R || P.dec) is {y!r}: the decode path equals the reference, so the gate has "
                           "no noise scale")
    return y


def gate(arms: dict, labels: list, files: list, dirs: list) -> dict:
    """The main gate over both arms, clustered by `labels`: {"yardstick", "arms", "failures", "pass"}; every arm also
    carries its dependence (ICC, DE) at file and directory level."""
    y = _yardstick(arms)
    verdicts, failures = {}, []
    for arm in ARMS:
        d = arms[arm]["paired"]["d"]
        v = {**_b1_b3(arms, arm, y), **b2v3(d, labels), "dependence": dependence(d, files, dirs)}
        v["pass"] = v["b1"] and v["b2"] and v["b3"]
        verdicts[arm] = v
        if not v["b1"]:
            failures.append(f"{arm} b1: KL {v['kl_candidate']:.6g} / max(P {v['kl_predecessor']:.6g}, Y {y:.6g}) = "
                            f"{v['ratio']:.4g} > {RATIO_LIMIT}")
        if not v["b2"]:
            failures.append(f"{arm} b2v3: UCB95 {v['ucb']:.6g} (mean {v['mean']:.6g}, n {v['n']}, G {v['clusters']}) "
                            f">= {DELTA}")
        if not v["b3"]:
            failures.append(f"{arm} b3: {b3_text(v)} < {ALPHA}")
    return {"yardstick": y, "arms": verdicts, "failures": failures, "pass": not failures}


def long_gate(arms: dict) -> dict:
    """The long suite: per arm the prompt-level bounds and b3 flips (reported), and the outcome."""
    y = _yardstick(arms)
    verdicts = {arm: {**_b1_b3(arms, arm, y), **long_suite(arms[arm]["paired"]["d"])} for arm in ARMS}
    return {"yardstick": y, "arms": verdicts, "outcome": long_outcome(verdicts)}


def b3_text(v: dict) -> str:
    """One arm's b3 numbers in words."""
    num, den = v["b3_pi"]
    return (f"{v['b3_b']} vs {v['b3_c']} confident flips ({v['b3_versus']}, {v['b3_rule']}), one-sided exact p "
            f"{v['flip_p']:.3g} at pi {num}/{den}")


def spread_control(x: list, files: list, c: float) -> list:
    """d_i = e_f x_i + c, e_f = +-1 per source file in order of first appearance from
    random.Random(SEED).getrandbits(1) (1 -> +1)."""
    rng = random.Random(SEED)
    sign: dict = {}
    for f in files:
        if f not in sign:
            sign[f] = 1.0 if rng.getrandbits(1) else -1.0
    return [sign[f] * v + c for v, f in zip(x, files)]


# -- the screen ([REVISED: M23-D9Q-screen]) -------------------------------------------------------------------
def _hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest(), "big")


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def screen_select(prompts: list, seed: int = SCREEN_SEED, n: int = SCREEN_N) -> list:
    """The screen's prompts: `n` of `prompts` [(id, file)], a pure function of the ids, their files and `seed`.
    Systematic sampling over the files: the files ordered by sha256(f"{SCREEN_TAG}:{seed}:file:{file}"), file f taking
    the units [n C_{f-1}, n C_f) of n N (C the cumulative prompt counts), one seat at every s + k N (k = 0..n-1,
    s = sha256(f"{SCREEN_TAG}:{seed}:start") mod N); a file then holds the floor or the ceiling of its proportional
    share n n_f / N and every prompt has inclusion probability n / N. A file's seats go to its prompts in the order
    sha256(f"{SCREEN_TAG}:{seed}:prompt:{id}"). Returns the chosen ids in input order."""
    ids = [pid for pid, _ in prompts]
    total = len(ids)
    if len(set(ids)) != total:
        raise QualityError("screen selection: duplicate prompt ids")
    if not 0 < n <= total:
        raise QualityError(f"screen selection: {n} of {total} prompts")
    by_file: dict = {}
    for pid, f in prompts:
        by_file.setdefault(f, []).append(pid)
    start = _hash(f"{SCREEN_TAG}:{seed}:start") % total
    chosen: set = set()
    lo = 0
    for f in sorted(by_file, key=lambda f: _hash(f"{SCREEN_TAG}:{seed}:file:{f}")):
        hi = lo + n * len(by_file[f])
        seats = _ceil_div(hi - start, total) - _ceil_div(lo - start, total)
        chosen.update(sorted(by_file[f], key=lambda p: _hash(f"{SCREEN_TAG}:{seed}:prompt:{p}"))[:seats])
        lo = hi
    return [pid for pid in ids if pid in chosen]


def screen_frame() -> dict:
    """The `screen` of a screen run's stdin frame (native_dev.py `test --suite quality --screen`)."""
    return {"n": SCREEN_N, "seed": SCREEN_SEED, "prereg_commit": SCREEN_PREREG_COMMIT}


def triage_select(prompts: list) -> list:
    """[REVISED: owner 2026-10-16 screen tiers]: a candidate's triage prompts, `screen_select` of `prompts` at
    SCREEN_SEED with n = TRIAGE_N (a pure function of the ids, their files and the seed, as the screen's)."""
    return screen_select(prompts, seed=SCREEN_SEED, n=TRIAGE_N)


def triage_frame() -> dict:
    """The `screen` of a triage run's stdin frame (native_dev.py `test --suite quality --triage`)."""
    return {"tier": TRIAGE_MODE, "n": TRIAGE_N, "seed": SCREEN_SEED, "prereg_commit": SCREEN_PREREG_COMMIT}


def subset_frame(ids: list) -> dict:
    """The `screen` of a prompt-id subset run's stdin frame (native_dev.py `test --suite quality --prompt-ids`): the
    main prompt ids it runs, in fixture order."""
    return {"tier": SUBSET_MODE, "ids": list(ids)}


def checkpoint_families(checkpoint: str | None, lever_set: str | None) -> str | None:
    """The families file (CHECKPOINTS) of a frame's candidate `checkpoint`, None without one; a checkpoint run names
    its set by the checkpoint (`lever_set` == `checkpoint`), anything else fails."""
    if checkpoint is None:
        return None
    if checkpoint not in CHECKPOINTS:
        raise QualityError(f"{checkpoint!r} is not a candidate checkpoint ({'|'.join(CHECKPOINTS)})")
    if lever_set != checkpoint:
        raise QualityError(f"a candidate checkpoint run names its set by the checkpoint: lever set {lever_set!r}, "
                           f"checkpoint {checkpoint!r}")
    return CHECKPOINTS[checkpoint]


def reference_families(reference: str | None) -> str | None:
    """The families file (REFERENCES) of a frame's `reference` checkpoint (P at /model), None without one."""
    if reference is None:
        return None
    if reference not in REFERENCES:
        raise QualityError(f"{reference!r} is not a reference checkpoint ({'|'.join(REFERENCES)})")
    return REFERENCES[reference]


def lever_sets(registry: dict) -> dict:
    """{name: [(lever, impl id), ...]}: the lever sets suite `quality` runs ([REVISED: owner 2026-10-16 screen tiers]),
    every non-empty combination of the registry's `d9_batch` implementations with at most one per lever, named by
    their ids joined by `+` in registry order (a single implementation, a triage candidate; two or more, a bundle of
    KEEP candidates, e.g. the bundle re-screened without a culprit). The suite runs the CUDA library, so an
    implementation whose `arch` lacks sm_120a (k2_gemm tail_split: Vulkan only, screened on AMD,
    tasks/vulkan-amd-spec.md D7) is in no set."""
    singles = [(lever["name"], impl["id"]) for lever in registry["levers"] for impl in lever["impls"]
               if impl.get("d9_batch") and "sm_120a" in impl.get("arch", ["sm_120a"])]
    if not singles:
        raise QualityError("the registry lists no d9_batch implementation")
    out = {}
    for mask in range(1, 1 << len(singles)):
        chosen = [s for k, s in enumerate(singles) if mask >> k & 1]
        if len({lever for lever, _ in chosen}) == len(chosen):
            out["+".join(impl for _, impl in chosen)] = chosen
    return dict(sorted(out.items(), key=lambda kv: (len(kv[1]), [singles.index(s) for s in kv[1]])))


def tier_corpus(mode: str, n: int) -> dict:
    """The tier fields of a run's corpus row (mode SCREEN_MODE, TRIAGE_MODE or SUBSET_MODE; `n` its main prompts;
    none for a baseline or full candidate run)."""
    if mode == SCREEN_MODE:
        return {"screen_n": n, "screen_seed": SCREEN_SEED, "screen_prereg_commit": SCREEN_PREREG_COMMIT}
    if mode == TRIAGE_MODE:
        return {"triage_n": n, "screen_seed": SCREEN_SEED, "screen_prereg_commit": SCREEN_PREREG_COMMIT}
    if mode == SUBSET_MODE:
        return {"subset_n": n}
    return {}


def _require_screen_pin() -> None:
    if SCREEN_PREREG_COMMIT is None:
        raise QualityError("the screen pre-registration is not pinned (quality_measure.SCREEN_PREREG_COMMIT): no "
                           "screen run and no screen verdict")


def run_mode(screen: dict | None, lever_set: str | None, main_ids: list, long_n: int) -> str:
    """The suite's run mode from its frame's `screen`, lever set, main prompt ids and long prompt count: `baseline`
    (no lever set, no screen), `candidate` (a lever set, no screen), SCREEN_MODE (a lever set on SCREEN_N main
    prompts, `screen_frame()`) or TRIAGE_MODE (TRIAGE_N, `triage_frame()`), both at the pinned pre-registration and
    with LONG_MIN..LONG_PROMPTS long prompts, or SUBSET_MODE (a lever set on exactly the frame's ids, no long suite);
    anything else fails."""
    if screen is None:
        mode = "baseline" if lever_set is None else "candidate"
    else:
        mode = screen.get("tier", SCREEN_MODE) if isinstance(screen, dict) else None
        if mode == SCREEN_MODE:
            _require_screen_pin()
            ok = screen == screen_frame() and len(main_ids) == SCREEN_N
        elif mode == TRIAGE_MODE:
            _require_screen_pin()
            ok = screen == triage_frame() and len(main_ids) == TRIAGE_N
        elif mode == SUBSET_MODE:
            ids = screen.get("ids")
            ok = (isinstance(ids, list) and len(ids) > 0 and len(set(ids)) == len(ids)
                  and screen == subset_frame(ids) and list(main_ids) == ids)
        else:
            ok = False
        if lever_set is None or not ok:
            raise QualityError(f"the frame's screen {screen} (lever set {lever_set!r}, {len(main_ids)} main prompts) "
                               f"is not {screen_frame()} or {triage_frame()} of a lever set on {SCREEN_N} or "
                               f"{TRIAGE_N} main prompts, nor {subset_frame(['<id>', '...'])} of a lever set on "
                               "exactly those main prompts")
    if not (long_n == 0 if mode == SUBSET_MODE else LONG_MIN <= long_n <= LONG_PROMPTS):
        raise QualityError(f"a {mode} run with {long_n} long prompts (a subset run has none, any other "
                           f"{LONG_MIN}..{LONG_PROMPTS})")
    return mode


def screen_outcome(bounds: dict, *, b1_b3: bool) -> dict:
    """The screen's decision from both arms' rule (1) bounds (`cr_bound` keys mean, se, t95, ucb); the LCB is the same
    one-sided t(0.95, G-1) bound below the mean. FAIL iff LCB > SCREEN_FAIL_LCB on either arm; PASS iff UCB <
    SCREEN_PASS_UCB on both arms and `b1_b3` (b1 and b3 hold on both arms); else ESCALATE (the full b2v3 run)."""
    arms = {a: {"ucb": bounds[a]["ucb"], "lcb": bounds[a]["mean"] - bounds[a]["t95"] * bounds[a]["se"]}
            for a in ARMS}
    if any(v["lcb"] > SCREEN_FAIL_LCB for v in arms.values()):
        outcome = "FAIL"
    elif b1_b3 and all(v["ucb"] < SCREEN_PASS_UCB for v in arms.values()):
        outcome = "PASS"
    else:
        outcome = "ESCALATE"
    return {"outcome": outcome, "arms": arms}


def screen_gate(g: dict) -> dict:
    """`screen_outcome` of a main `gate` result."""
    return screen_outcome(g["arms"], b1_b3=all(g["arms"][a]["b1"] and g["arms"][a]["b3"] for a in ARMS))


def escalate_command(name: str) -> str:
    """The full b2v3 run a screen ESCALATE sends lever set `name` to."""
    return f"CERE_LIVE=1 python3 scripts/live/native_dev.py test --suite quality --shape {SHAPE} --lever-set {name}"


def subset_command(name: str, ids: list) -> str:
    """The attribution rerun of candidate `name` on exactly the main prompts `ids` ([REVISED: owner 2026-10-16 screen
    tiers])."""
    return f"{escalate_command(name)} --prompt-ids {','.join(ids)}"


# -- report rows ---------------------------------------------------------------------------------------------
def report_rows(name: str, part: str, arms: dict) -> list:
    """The suite's report rows for lever set `name` on `part`: per arm the two summaries and the paired statistics
    (the per-prompt d_i go to the `prompt <id>` rows, `prompt_rows`)."""
    out = []
    for arm in ARMS:
        for side in SIDES:
            out.append({"row": ROW, "case": f"{name} {part} {arm} {side}", **arms[arm][side]})
        out.append({"row": ROW, "case": f"{name} {part} {arm} paired",
                    **{k: v for k, v in arms[arm]["paired"].items() if k != "d"}})
    return out


def prompt_rows(part: str, pids: list, files: list, dirs: list, x: list, arms_by_set: dict) -> list:
    """One `prompt <id>` row per prompt of `part`: its fixture file and directory index, its null value x_i and, per
    lever set and arm, its d_i (`arms_by_set`: {set: arms with paired d lists})."""
    out = []
    for i, pid in enumerate(pids):
        row = {"row": ROW, "case": f"prompt {pid}", "part": part, "file": files[i], "dir": dirs[i], "x": x[i]}
        for name, arms in arms_by_set.items():
            for arm in ARMS:
                row[f"{name}_{arm}"] = arms[arm]["paired"]["d"][i]
        out.append(row)
    return out


def by_case(rows: list) -> dict:
    out: dict = {}
    for r in rows:
        if r.get("row") == ROW:
            case = r.get("case")
            if case in out:
                raise QualityError(f"report row {case!r} twice")
            out[case] = r
    return out


def prompts_by_part(cases: dict) -> dict:
    """{part: [`prompt <id>` rows in report order]}; every row holds PROMPT_KEYS and a known part."""
    out: dict = {part: [] for part in PARTS}
    for case, r in cases.items():
        if not str(case).startswith("prompt "):
            continue
        missing = [k for k in PROMPT_KEYS if k not in r]
        if missing or r["part"] not in PARTS:
            raise QualityError(f"row {case!r}: missing {missing} or part {r.get('part')!r} not in {PARTS}")
        out[r["part"]].append(r)
    return out


def pick(cases: dict, case: str, keys: tuple) -> dict:
    r = cases.get(case)
    if r is None:
        raise QualityError(f"report lacks row {case!r}")
    missing = [k for k in keys if k not in r]
    if missing:
        raise QualityError(f"row {case!r} lacks {missing}")
    return {k: r[k] for k in keys}


def gate_input(cases: dict, prompts: dict, name: str, part: str) -> dict:
    """Inverse of `report_rows` + `prompt_rows` (other rows and extra keys such as `utc` are ignored); a missing row
    or a prompt without the set's d_i fails."""
    out = {}
    for arm in ARMS:
        key = f"{name}_{arm}"
        if any(key not in p for p in prompts[part]):
            raise QualityError(f"a {part} prompt row lacks {key}")
        out[arm] = {**{side: pick(cases, f"{name} {part} {arm} {side}", SUMMARY_KEYS) for side in SIDES},
                    "paired": {**pick(cases, f"{name} {part} {arm} paired", PAIRED_KEYS),
                               "d": [p[key] for p in prompts[part]]}}
    return out


def corpus_row(cases: dict, prompts: dict, set_sha256: str) -> dict:
    """The report's corpus row, checked: PROMPT_SET, PREREG_COMMIT, the fixture digest, a shape, a prompt row for
    every main prompt and LONG_MIN..LONG_PROMPTS long prompts (none in a SUBSET_MODE run); a screen or triage run's
    (mode SCREEN_MODE / TRIAGE_MODE) the pinned screen pre-registration and seed, and every tier run's `tier_corpus`
    n = its main prompts."""
    c = pick(cases, "corpus", ("prompt_set", "prereg_commit", "set_sha256", "shape", "mode", "sequences", "long"))
    if (c["prompt_set"], c["prereg_commit"], c["set_sha256"]) != (PROMPT_SET, PREREG_COMMIT, set_sha256):
        raise QualityError(f"the quality report records (prompt set, pre-registration, set digest) "
                           f"{(c['prompt_set'], c['prereg_commit'], c['set_sha256'])}, expected "
                           f"{(PROMPT_SET, PREREG_COMMIT, set_sha256)}")
    long_ok = c["long"] == 0 if c["mode"] == SUBSET_MODE else LONG_MIN <= c["long"] <= LONG_PROMPTS
    if c["shape"] not in SHAPES or not long_ok:
        raise QualityError(f"the quality report's shape {c['shape']!r} / long prompts {c['long']} are not one of "
                           f"{SHAPES} / {LONG_MIN}..{LONG_PROMPTS} (a {SUBSET_MODE} run: 0)")
    if (len(prompts["main"]), len(prompts["long"])) != (c["sequences"], c["long"]):
        raise QualityError(f"{len(prompts['main'])} main and {len(prompts['long'])} long prompt rows, the corpus row "
                           f"says {c['sequences']} and {c['long']}")
    if c["mode"] in (SCREEN_MODE, TRIAGE_MODE):
        _require_screen_pin()
    want = tier_corpus(c["mode"], c["sequences"])
    if want:
        got = pick(cases, "corpus", tuple(want))
        if got != want:
            raise QualityError(f"the {c['mode']} report records {got}, expected {want}")
    return c


def run_check(rows: list, set_sha256: str, level: str | None = None) -> dict:
    """A suite report's controls, framing and null values, replayed. `level`: the frozen cluster level (a candidate
    run); None decides it here by rule (3) on the run's own null values (a baseline). problems == [] iff the run is
    valid. A SUBSET_MODE report holds no controls (`subset_check`)."""
    cases = by_case(rows)
    prompts = prompts_by_part(cases)
    corpus = corpus_row(cases, prompts, set_sha256)
    if corpus["mode"] == SUBSET_MODE:
        raise QualityError(f"a {SUBSET_MODE} report (the attribution rerun) holds no controls and is never a verdict")
    files, dirs = [p["file"] for p in prompts["main"]], [p["dir"] for p in prompts["main"]]
    x, x_long = [p["x"] for p in prompts["main"]], [p["x"] for p in prompts["long"]]
    null_dep = dependence(x, files, dirs)
    if level is None:
        level = cluster_level(null_dep)
    elif level not in LEVELS:
        raise QualityError(f"cluster level {level!r} is not one of {LEVELS}")
    labels = files if level == "file" else dirs
    gates = {name: gate(gate_input(cases, prompts, name, "main"), labels, files, dirs) for name in CONTROL_SETS}
    longs = {name: long_gate(gate_input(cases, prompts, name, "long")) for name in CONTROL_SETS}
    spread = {}
    for name, c in INJ_SPREAD.items():
        spread[name] = {"main": b2v3(spread_control(x, files, c), labels),
                        "literal": b2v3([v + c for v in x], labels),
                        "long": long_suite(spread_control(x_long, list(range(len(x_long))), c))}
    frames = {part: pick(cases, f"framing {part}", FRAMING_KEYS) for part in PARTS}
    problems = []
    if gates[NEGATIVE]["pass"]:
        problems.append(f"{NEGATIVE} passed the main gate (must fail)")
    if any(gates[INJ_TARGET]["arms"][a]["b2"] for a in ARMS):
        problems.append(f"{INJ_TARGET} passed b2v3 on an arm (must fail both)")
    if longs[INJ_TARGET]["outcome"] != "BLOCK":
        problems.append(f"{INJ_TARGET} did not BLOCK the long suite ({longs[INJ_TARGET]['outcome']})")
    if spread[INJ_SPREAD_REQUIRED]["main"]["b2"]:
        problems.append(f"{INJ_SPREAD_REQUIRED} passed b2v3 (must fail): UCB95 "
                        f"{spread[INJ_SPREAD_REQUIRED]['main']['ucb']:.6g} < {DELTA}")
    if not gates[SELF_CONSISTENCY]["pass"]:
        problems.append(f"{SELF_CONSISTENCY} failed the main gate (must pass): "
                        + "; ".join(gates[SELF_CONSISTENCY]["failures"]))
    if longs[SELF_CONSISTENCY]["outcome"] != long_clear(len(x_long)):
        problems.append(f"{SELF_CONSISTENCY} long suite: {longs[SELF_CONSISTENCY]['outcome']} (must be clear)")
    screen = None
    if corpus["mode"] in (SCREEN_MODE, TRIAGE_MODE):  # the screen's own decision on its controls
        screen = {name: screen_gate(gates[name]) for name in (INJ_TARGET, SELF_CONSISTENCY)}
        for name, must in ((INJ_TARGET, "FAIL"), (SELF_CONSISTENCY, "PASS")):
            if screen[name]["outcome"] != must:
                problems.append(f"{name} did not {must} the screen ({screen[name]['outcome']})")
    problems += framing_problems(frames)
    return {"corpus": {k: corpus[k] for k in ("shape", "mode", "sequences", "long")}, "level": level,
            "null": {"n": len(x), "mean": math.fsum(x) / len(x), "sd": statistics.stdev(x), "dependence": null_dep},
            "x": x, "labels": labels, "files": files, "dirs": dirs, "gates": gates, "longs": longs,
            "spread": spread, "framing": frames, "screen": screen, "problems": problems, "cases": cases,
            "prompts": prompts}


def framing_problems(frames: dict) -> list:
    """The `framing <part>` rows ({part: FRAMING_KEYS}) that fail `framing_ok`, as problems."""
    return [f"framing {part}: R's NLL {f['ref_nll_mean']:.4g} (<= {FRAMING_NLL_MAX}), special top-1 share "
            f"{f['special_top1_share']:.4g} (<= {FRAMING_SPECIAL_MAX})"
            for part, f in frames.items() if not framing_ok(f)]


# -- attribution ([REVISED: owner 2026-10-16 screen tiers]): diagnostic only, never a verdict ----------------------
D_KEYS = (*ARMS, "both")  # per arm, and "both": the two arms' mean


def _mean(values: list) -> float:
    return math.fsum(values) / len(values)


def _prompt_d(prompts: list, name: str) -> dict:
    """{id: {arm: d_i, "both": the arms' mean}} of lever set `name` over `prompt <id>` rows, in report order."""
    out = {}
    for p in prompts:
        missing = [f"{name}_{arm}" for arm in ARMS if f"{name}_{arm}" not in p]
        if missing:
            raise QualityError(f"row {p['case']!r} lacks {missing}")
        d = dict(zip(ARMS, _finite_list([float(p[f"{name}_{arm}"]) for arm in ARMS], f"{p['case']} {name} d")))
        out[p["case"][len("prompt "):]] = d | {"both": _mean(list(d.values()))}
    return out


def subset_check(rows: list, set_sha256: str) -> dict:
    """A SUBSET_MODE report (the suite's prompt-id subset run: P and one lever set on exactly the frame's main
    prompts, no long suite, no control) replayed: its set, ids, per-prompt d ({id: D_KEYS}), mean d (D_KEYS) and
    framing; problems == [] iff R's framing holds."""
    cases = by_case(rows)
    prompts = prompts_by_part(cases)
    corpus = corpus_row(cases, prompts, set_sha256)
    if corpus["mode"] != SUBSET_MODE:
        raise QualityError(f"the report is mode {corpus['mode']}, not a {SUBSET_MODE} run")
    suffix = " main pre paired"
    names = sorted(str(c)[:-len(suffix)] for c in cases if str(c).endswith(suffix))
    if len(names) != 1:
        raise QualityError(f"a {SUBSET_MODE} report holds one lever set, not {names}")
    [name] = names
    gate_input(cases, prompts, name, "main")  # every summary and paired row of the set is there
    d = _prompt_d(prompts["main"], name)
    framing = pick(cases, "framing main", FRAMING_KEYS)
    return {"corpus": {k: corpus[k] for k in ("shape", "mode", "sequences", "long")}, "set": name, "ids": list(d),
            "d": d, "mean": {k: _mean([v[k] for v in d.values()]) for k in D_KEYS}, "framing": framing,
            "problems": framing_problems({"main": framing})}


def _bundle_d(bundle: str, quality: dict, set_sha256: str) -> dict:
    """The bundle's per-prompt d ({id: D_KEYS}, report order) from a valid passing screen or full candidate run."""
    if quality.get("suite") != "quality" or quality.get("result") != "pass":
        raise QualityError(f"the bundle report is suite {quality.get('suite')!r} result {quality.get('result')!r}, "
                           "not a passing quality suite run")
    checked = run_check(quality.get("rows") or [], set_sha256, CLUSTER_LEVEL)
    if checked["corpus"]["mode"] not in (SCREEN_MODE, "candidate"):
        raise QualityError(f"the bundle report is mode {checked['corpus']['mode']}, not a {SCREEN_MODE} or full "
                           "candidate run")
    if checked["problems"]:
        raise QualityError("the bundle run is invalid: " + "; ".join(checked["problems"]))
    return _prompt_d(checked["prompts"]["main"], bundle)


def _top_ids(d: dict, k: int) -> list:
    if not 0 < k <= len(d):
        raise QualityError(f"top {k} of {len(d)} prompts")
    return [pid for pid, _ in sorted(d.items(), key=lambda kv: -kv[1]["both"])[:k]]  # stable: ties in report order


def attribution_ids(bundle: str, quality: dict, set_sha256: str, k: int = ATTRIBUTION_K) -> list:
    """The top-`k` main prompt ids of a bundle report (a valid screen or full candidate run of lever set `bundle`)
    by the bundle's dNLL, the mean of the prompt's two arms' d_i, highest first (ties in report order). Ids only:
    the prompts every candidate reruns (`test --suite quality --lever-set C --prompt-ids`)."""
    return _top_ids(_bundle_d(bundle, quality, set_sha256), k)


def attribution(bundle: str, quality: dict, candidates: list, set_sha256: str, k: int = ATTRIBUTION_K) -> dict:
    """The attribution of a bundle FAIL or ESCALATE (diagnostic only: `verdict` None, never a verdict): the bundle
    report's top-`k` prompts (`attribution_ids`) and each candidate's SUBSET_MODE rerun on exactly them
    (`candidates`: suite reports whose lever sets split the bundle's implementations, one candidate on at a time).
    Per candidate its mean dNLL on them (D_KEYS), highest first; `sum` their sum; `interaction` the bundle's own mean
    on them minus `sum` (the bundle's mean is on the prompts its own dNLL selected, so it reads high)."""
    d = _bundle_d(bundle, quality, set_sha256)
    ids = _top_ids(d, k)
    bundle_mean = {key: _mean([d[pid][key] for pid in ids]) for key in D_KEYS}
    rows = []
    for doc in candidates:
        if doc.get("suite") != "quality" or doc.get("result") != "pass":
            raise QualityError(f"a candidate report is suite {doc.get('suite')!r} result {doc.get('result')!r}, not "
                               "a passing quality suite run")
        c = subset_check(doc.get("rows") or [], set_sha256)
        if c["problems"]:
            raise QualityError(f"the {c['set']} rerun is invalid: " + "; ".join(c["problems"]))
        if sorted(c["ids"]) != sorted(ids):
            raise QualityError(f"the {c['set']} rerun ran {len(c['ids'])} prompts that are not the bundle's top {k}")
        rows.append({"set": c["set"], "utc": doc.get("utc"), "ktest_sha256": doc.get("ktest_sha256"),
                     "mean": c["mean"]})
    parts = [p for r in rows for p in r["set"].split("+")]
    if not rows or len(set(parts)) != len(parts) or sorted(parts) != sorted(bundle.split("+")):
        raise QualityError(f"the candidates {[r['set'] for r in rows]} do not split the bundle {bundle} (each of its "
                           "implementations on in exactly one candidate rerun)")
    total = {key: math.fsum(r["mean"][key] for r in rows) for key in D_KEYS}
    return {"row": ROW, "mode": "attribution", "verdict": None, "bundle": bundle, "k": k, "ids": ids,
            "bundle_utc": quality.get("utc"), "bundle_mean": bundle_mean,
            "candidates": sorted(rows, key=lambda r: -r["mean"]["both"]), "sum": total,
            "interaction": {key: bundle_mean[key] - total[key] for key in D_KEYS}}


# -- the operating characteristic (rule (5)) -------------------------------------------------------------------
def _sums_by_label(values: list, labels: list) -> list:
    return clusters(values, labels)


def _centred(sums: list) -> list:
    n = sum(k for k, _ in sums)
    m = math.fsum(s for _, s in sums) / n
    return [(k, s - k * m) for k, s in sums]


def pass_rate(sums: list, mu: float, replicates: int, rng: random.Random, t95: float) -> float:
    """The fraction of `replicates` cluster resamples (G whole clusters with replacement from `sums`, centred values,
    each shifted by `mu`) whose rule (1) UCB < DELTA."""
    g = len(sums)
    passed = 0
    for _ in range(replicates):
        draw = [sums[rng.randrange(g)] for _ in range(g)]
        n = sum(k for k, _ in draw)
        m = math.fsum(s for _, s in draw) / n
        v = g / (g - 1) * math.fsum((s - k * m) ** 2 for k, s in draw) / (n * n)
        passed += m + mu + t95 * math.sqrt(v) < DELTA
    return passed / replicates


def operating_characteristic(x: list, labels: list, shape: str, *, replicates: int = SIM_REPLICATES,
                             outer: int = OUTER, inner: int = INNER) -> dict:
    """Rule (5) for one shape: per shift, P(PASS | mu) with its Monte Carlo SE and the 2.5-97.5 percentile interval of
    the `outer` cluster-bootstrap estimates (`outer_pass[mu]`: the per-replicate estimates, kept for the shape
    difference)."""
    sums = _centred(_sums_by_label(x, labels))
    g = len(sums)
    if g < 2:
        raise QualityError(f"the simulation needs at least 2 clusters, got {g}")
    t95 = t_quantile(1.0 - B2V3_ALPHA, g - 1)
    raw = _sums_by_label(x, labels)
    outer_pass: dict = {mu: [] for mu in SHIFTS}
    rng_outer = random.Random(f"{SEED}:outer:{shape}")
    for o in range(outer):
        boot = _centred([raw[rng_outer.randrange(g)] for _ in range(g)])
        for mu in SHIFTS:
            outer_pass[mu].append(pass_rate(boot, mu, inner, random.Random(f"{SEED}:outer:{shape}:{o}:{mu!r}"), t95))
    out = {"shape": shape, "clusters": g, "replicates": replicates, "outer": outer, "inner": inner, "shifts": {}}
    for mu in SHIFTS:
        p = pass_rate(sums, mu, replicates, random.Random(f"{SEED}:{shape}:{mu!r}"), t95)
        lo, hi = percentile_interval(outer_pass[mu])
        out["shifts"][repr(mu)] = {"mu": mu, "p_pass": p, "p_fail": 1.0 - p, "mc_se": math.sqrt(p * (1 - p) / replicates),
                                   "pass_lo": lo, "pass_hi": hi}
    out["outer_pass0"] = outer_pass[0.0]
    return out


def percentile_interval(values: list) -> tuple:
    """(2.5th, 97.5th) percentiles, linear interpolation between order statistics."""
    xs = sorted(values)
    if not xs:
        raise QualityError("no values for an interval")

    def q(p: float) -> float:
        h = (len(xs) - 1) * p
        lo = math.floor(h)
        return xs[lo] + (h - lo) * (xs[min(lo + 1, len(xs) - 1)] - xs[lo])
    return q(0.025), q(0.975)


def calibration(oc: dict) -> dict:
    """The +0.01 calibration check (owner ruling 2026-10-15): P(FAIL | +0.01)'s point estimate, MC SE and 95 %
    interval (the upper end is 1 - the lower end of P(PASS | +0.01)); qualifies iff the upper end >= FAIL_POWER."""
    r = oc["shifts"][repr(0.01)]
    lo, hi = 1.0 - r["pass_hi"], 1.0 - r["pass_lo"]
    return {"p_fail": r["p_fail"], "mc_se": r["mc_se"], "fail_lo": lo, "fail_hi": hi, "ok": hi >= FAIL_POWER}


def shape_decision(shapes: dict) -> dict:
    """Rule (5)'s choice. `shapes`: {shape: {"corpus_ok", "valid", "controls_ok", "oc"}} for SHAPES; a shape below its
    SHAPE_MIN_PROMPTS has no baseline ("oc" None) and does not qualify. Returns {"choice": shape | None, "report":
    reason | None, "qualifies", "calibration", "p_pass0", "delta", "delta_lo", "delta_hi", "tie"} (the shape
    comparison only when both shapes were measured; "both below PASS_FLOOR" likewise needs both)."""
    if set(shapes) != set(SHAPES):
        raise QualityError(f"shape decision needs {SHAPES}, got {sorted(shapes)}")
    qualifies, p0, cal = {}, {}, {}
    for s, v in shapes.items():
        enough = v["valid"] >= SHAPE_MIN_PROMPTS.get(s, 0)
        if v.get("oc") is None:
            if enough:
                raise QualityError(f"shape {s} has {v['valid']} valid prompts but no baseline")
            cal[s], p0[s], qualifies[s] = None, None, False
            continue
        if not enough:
            raise QualityError(f"shape {s} has {v['valid']} valid prompts (< {SHAPE_MIN_PROMPTS[s]}): no baseline")
        cal[s] = calibration(v["oc"])
        p0[s] = v["oc"]["shifts"][repr(0.0)]["p_pass"]
        qualifies[s] = bool(v["corpus_ok"] and v["controls_ok"] and cal[s]["ok"])
    out = {"qualifies": qualifies, "calibration": cal, "p_pass0": p0, "delta": None, "delta_lo": None,
           "delta_hi": None, "tie": None, "choice": None, "report": None}
    both = all(p is not None for p in p0.values())
    if both:
        diffs = [b - a for a, b in zip(shapes["L"]["oc"]["outer_pass0"], shapes["S"]["oc"]["outer_pass0"],
                                       strict=True)]
        lo, hi = percentile_interval(diffs)
        delta = p0["S"] - p0["L"]
        out |= {"delta": delta, "delta_lo": lo, "delta_hi": hi, "tie": abs(delta) < TIE or lo <= 0.0 <= hi}
    if both and all(p < PASS_FLOOR for p in p0.values()):
        out["report"] = f"both shapes have P(PASS | 0) < {PASS_FLOOR}"
    elif all(qualifies.values()):
        out["choice"] = TIE_SHAPE if out["tie"] else max(SHAPES, key=lambda s: p0[s])
    elif any(qualifies.values()):
        out["choice"] = next(s for s in SHAPES if qualifies[s])
    else:
        out["report"] = "no shape qualifies"
    return out


def baseline(reports: dict, valid: dict, set_sha256: str, long_suite_counts: dict | None = None, *,
             replicates: int = SIM_REPLICATES, outer: int = OUTER, inner: int = INNER) -> dict:
    """Rule (5) from the baseline runs: `reports` {shape: `test --suite quality` report (no lever set) | None for a
    shape below SHAPE_MIN_PROMPTS}, `valid` {shape: valid prompts of the fixture}. Per measured shape: the run's
    checks (its own cluster level by rule (3)), the controls and the simulated operating characteristic; then the
    shape decision. Returns {"shapes", "decision", "long_suite"}."""
    if set(reports) != set(SHAPES) or set(valid) != set(SHAPES):
        raise QualityError(f"the baseline needs {SHAPES}, got {sorted(reports)} / {sorted(valid)}")
    shapes = {}
    for s in SHAPES:
        rep = reports[s]
        if rep is None:
            shapes[s] = {"valid": valid[s], "corpus_ok": False, "controls_ok": False, "oc": None}
            continue
        if rep.get("suite") != "quality":
            raise QualityError(f"shape {s}: not a quality suite report (suite {rep.get('suite')!r})")
        checked = run_check(rep.get("rows") or [], set_sha256)
        if (checked["corpus"]["shape"], checked["corpus"]["mode"]) != (s, "baseline"):
            raise QualityError(f"shape {s}: the report is shape {checked['corpus']['shape']} mode "
                               f"{checked['corpus']['mode']}, expected {s} baseline")
        ok = not checked["problems"] and rep.get("result") == "pass"
        shapes[s] = {"valid": valid[s], "corpus_ok": True, "controls_ok": ok,
                     "report": "valid" if ok else "invalid: " + "; ".join(checked["problems"] or [
                         f"suite result {rep.get('result')!r}"]),
                     "utc": rep.get("utc"), "ktest_sha256": rep.get("ktest_sha256"),
                     "check": {k: checked[k] for k in ("corpus", "level", "null", "gates", "longs", "spread",
                                                       "framing", "problems")},
                     "oc": operating_characteristic(checked["x"], checked["labels"], s, replicates=replicates,
                                                    outer=outer, inner=inner)}
    return {"prereg_commit": PREREG_COMMIT, "set_sha256": set_sha256, "shapes": shapes,
            "decision": shape_decision(shapes), "long_suite": long_suite_counts}


# -- the verdict ---------------------------------------------------------------------------------------------
def _part_a(name: str, d9: dict | None) -> dict:
    """(a) the D9 envelope part of a `native_parity.py --d9-verdict` report (first divergence report-only)."""
    if d9 is None:
        return {"status": "missing"}
    levers = (d9.get("candidate") or {}).get("levers") or {}
    if sorted(levers.values()) != sorted(name.split("+")):
        raise QualityError(f"the D9 verdict's candidate levers {levers} are not the set {name}")
    envelope = [f for f in d9.get("failures", []) if not f.startswith(FIRST_DIVERGENCE)]
    runs = d9.get("candidate_runs")
    status = "fail" if envelope else "pass" if runs == D9_SERIES_RUNS else "incomplete"
    return {"status": status, "generated_at": d9.get("generated_at"), "candidate_runs": runs,
            "candidate_within": d9.get("candidate_within"), "predecessor_within": d9.get("predecessor_within"),
            "envelope_failures": envelope, "first_divergence_earlier": list(d9.get("earlier_ids", []))}


NEVER_A_VERDICT = {TRIAGE_MODE: "triage only drops or keeps a candidate",
                   SUBSET_MODE: "the attribution rerun is diagnostic only"}


def _refuse_never_a_verdict(quality: dict) -> None:
    """[REVISED: owner 2026-10-16 screen tiers]: a triage or subset report never satisfies a verdict or a default flip."""
    mode = next((r.get("mode") for r in quality.get("rows") or []
                 if r.get("row") == ROW and r.get("case") == "corpus"), None)
    if mode in NEVER_A_VERDICT:
        raise QualityError(f"a {mode} report is never a verdict ([REVISED: owner 2026-10-16 screen tiers]: "
                           f"{NEVER_A_VERDICT[mode]}; a default flip needs the bundle's screen PASS, or its escalated "
                           "full run, on exactly the shipped set)")


def _replay(name: str, quality: dict, set_sha256: str, mode: str, selection: list | None) -> tuple:
    """(checked, main gate, long gate) of lever set `name` replayed from a passing `quality` report of `mode`
    ("candidate", SCREEN_MODE or TRIAGE_MODE) on the decided SHAPE at CLUSTER_LEVEL whose required controls all
    behave; `selection`: the ids its main prompts must be (a screen's or triage's), None for a full run."""
    if SHAPE is None or CLUSTER_LEVEL is None:
        raise QualityError("no shape decision is committed (quality_measure.SHAPE / CLUSTER_LEVEL): the baseline and "
                           "the owner's review come first")
    if mode != "candidate":
        _require_screen_pin()
    if B3V3_PREREG_COMMIT is None:
        raise QualityError("the b3 v3 pre-registration is not pinned (quality_measure.B3V3_PREREG_COMMIT): no verdict")
    if quality.get("suite") != "quality":
        raise QualityError(f"not a quality suite report (suite {quality.get('suite')!r})")
    rows = quality.get("rows") or []
    checked = run_check(rows, set_sha256, CLUSTER_LEVEL)
    if (checked["corpus"]["shape"], checked["corpus"]["mode"]) != (SHAPE, mode):
        raise QualityError(f"the quality report is shape {checked['corpus']['shape']} mode "
                           f"{checked['corpus']['mode']}, expected {SHAPE} {mode} (a screen report is judged as a "
                           "screen, a full run's without it)")
    if checked["problems"]:
        raise QualityError("the run is invalid, no verdict (fix, record, rerun; never retune): "
                           + "; ".join(checked["problems"]))
    if quality.get("result") != "pass":
        raise QualityError(f"the quality suite run is {quality.get('result')!r}, not pass")
    cases, prompts = checked["cases"], checked["prompts"]
    if selection is not None and [p["case"][len("prompt "):] for p in prompts["main"]] != list(selection):
        raise QualityError(f"the report's main prompts are not the {mode} selection (quality_measure.{mode}_select "
                           f"of the fixture's shape {SHAPE} prompts)")
    measured = gate(gate_input(cases, prompts, name, "main"), checked["labels"], checked["files"], checked["dirs"])
    return checked, measured, long_gate(gate_input(cases, prompts, name, "long"))


def _controls(checked: dict) -> dict:
    return {k: checked[k] for k in ("gates", "longs", "spread", "framing", "null", "screen", "problems")}


def _part_b(name: str, quality: dict, set_sha256: str, screen: list | None = None) -> dict:
    """(b) the main gate and the long suite replayed from a passing candidate `quality` report on the decided SHAPE
    at CLUSTER_LEVEL whose required controls all behave (else no verdict). `screen`: the screen selection the report's
    main prompts must be ([REVISED: M23-D9Q-screen]; the report a screen run's, (b) from `screen_gate`: fail on FAIL,
    escalate on ESCALATE); None: a full candidate run under the b2v3 rule. A triage or subset report is refused
    ([REVISED: owner 2026-10-16 screen tiers])."""
    _refuse_never_a_verdict(quality)
    checked, measured, long = _replay(name, quality, set_sha256, "candidate" if screen is None else SCREEN_MODE,
                                      screen)
    clear = long["outcome"] == long_clear(long["arms"][ARMS[0]]["n"])
    decided = None
    if screen is None:
        status = "fail" if not measured["pass"] or long["outcome"] == "BLOCK" else "pass" if clear else "owner"
    else:
        decided = screen_gate(measured)
        status = ("fail" if decided["outcome"] == "FAIL" or long["outcome"] == "BLOCK" else "escalate"
                  if decided["outcome"] == "ESCALATE" else "pass" if clear else "owner")
    return {"status": status, "utc": quality.get("utc"), "prereg_commit": PREREG_COMMIT,
            "b3v3_commit": B3V3_PREREG_COMMIT,
            "set_sha256": set_sha256, "ktest_sha256": quality.get("ktest_sha256"), "shape": SHAPE,
            "level": CLUSTER_LEVEL, "gate": measured, "long": long, "screen": decided, "controls": _controls(checked)}


def triage(name: str, quality: dict, set_sha256: str, selection: list) -> dict:
    """[REVISED: owner 2026-10-16 screen tiers]: candidate `name`'s triage from a passing TRIAGE_MODE report whose
    main prompts are `selection` (`triage_select`): the screen's decision (`screen_gate`, the same bound and
    thresholds) on its n prompts; FAIL -> DROP, PASS or ESCALATE -> KEEP (into the bundle). Never a verdict (`verdict`
    None; `verdict` refuses a triage report); the long suite is reported beside it."""
    _require_screen_pin()
    checked, measured, long = _replay(name, quality, set_sha256, TRIAGE_MODE, selection)
    decided = screen_gate(measured)
    return {"row": ROW, "set": name, "mode": TRIAGE_MODE, "verdict": None, "label": f"triage n={len(selection)}",
            "n": len(selection), "seed": SCREEN_SEED, "screen_prereg_commit": SCREEN_PREREG_COMMIT,
            "outcome": decided["outcome"], "decision": TRIAGE_DECISION[decided["outcome"]], "screen": decided,
            "quality_utc": quality.get("utc"), "prereg_commit": PREREG_COMMIT, "b3v3_commit": B3V3_PREREG_COMMIT,
            "set_sha256": set_sha256, "ktest_sha256": quality.get("ktest_sha256"), "shape": SHAPE,
            "level": CLUSTER_LEVEL, "gate": measured, "long": long, "controls": _controls(checked)}


# (c)'s per-set evidence ([REVISED: review-wave-fg M2/M3]): the K8 suite and class that run K8Speculation on the set's
# levers (executor/tests/gpu/test_model_k8_forward.py; the P4 sets' classes are test_model_k8p4_spec.py's, suite
# model_k8p4 since S1 of the 2026-10-18 test audit) and the spec profile R3 must run (configs/runtime/*.toml).
K8_CLASSES = {"small_m": "K8SpeculationSmallM", "p4": "K8SpeculationP4", "p4+small_m": "K8SpeculationSmallMP4"}
K8_SUITES = {"small_m": "model_k8", "p4": "model_k8p4", "p4+small_m": "model_k8p4"}
SPEC_PROFILES = {"small_m": "native-g0-qwen38-flash-next-v16"}


def _set_key(name: str) -> str:
    return "+".join(sorted(name.split("+")))


def _ktest_status(name: str, spec: dict | None) -> str:
    """The K8 suite passes only with result pass, no skip and the set's K8Speculation* class run with every test ok."""
    if spec is None:
        return "missing"
    cls = (spec.get("classes") or {}).get(K8_CLASSES.get(_set_key(name), ""))
    ok = (spec.get("result") == "pass" and spec.get("skipped") == 0 and isinstance(cls, dict)
          and cls.get("ok", 0) > 0 and cls.get("skipped") == 0 and cls.get("failed") == 0)
    return "pass" if ok else "fail"


def _part_c(name: str, spec: dict | None, r3: dict | None, library: str | None) -> dict:
    """(c) greedy spec == plain: the set's GPU1 K8 suite report (K8_SUITES: the set's K8Speculation* class, no skip) and
    GPU0 R3 on the set's spec profile, both on the candidate library `library` where the report records one."""
    suite = K8_SUITES.get(_set_key(name), "model_k8")
    if spec is not None and spec.get("suite") != suite:
        raise QualityError(f"the spec report is suite {spec.get('suite')!r}, not {name}'s {suite}")
    if r3 is not None and (r3.get("phase") != "G" or "R3" not in (r3.get("rows") or {})):
        raise QualityError("the R3 report is not a `native_spec.py --phase G` report holding row R3")
    profile = None if r3 is None else (r3.get("configs") or {}).get("spec_profile_id")
    if r3 is not None and (profile is None or profile != SPEC_PROFILES.get(_set_key(name))):
        raise QualityError(f"the R3 report ran spec profile {profile!r}, not {name}'s "
                           f"{SPEC_PROFILES.get(_set_key(name))!r}")
    for what, doc in ((suite, spec), ("R3", r3)):
        got = None if doc is None else doc.get("library_sha256")
        if got is not None and got != library:
            raise QualityError(f"the {what} report ran library {got}, not the candidate library {library}")
    ktest = _ktest_status(name, spec)
    r3_status = "pending" if r3 is None else "pass" if str(r3["rows"]["R3"].get("pass")) == "True" else "fail"
    parts = (ktest, r3_status)
    status = "fail" if "fail" in parts else "pass" if parts == ("pass", "pass") else "incomplete"
    return {"status": status, "ktest": ktest, "ktest_utc": None if spec is None else spec.get("utc"),
            "ktest_tests": None if spec is None else spec.get("tests"),
            "ktest_suite": suite, "ktest_class": K8_CLASSES.get(_set_key(name)), "library_sha256": library,
            "r3": r3_status,
            "r3_utc": None if r3 is None else r3.get("utc"), "r3_spec_profile_id": profile,
            "r3_line": None if r3 is None else r3["rows"]["R3"].get("line")}


def throughput(name: str, sweep: dict | None) -> dict | None:
    """The perf sweep row whose swept levers are exactly `name`'s implementations (report-only)."""
    if sweep is None:
        return None
    want = sorted(name.split("+"))
    for r in sweep.get("rows") or []:
        changing = sorted(r.get("changing") or [])
        if changing == want:
            return {"decode_r1_spec": r.get("decode_r1_spec"), "swept": r.get("swept"), "utc": sweep.get("utc")}
    raise QualityError(f"the perf sweep has no row whose numerics-changing implementations are {want}")


def verdict(name: str, quality: dict, *, d9: dict | None, spec: dict | None, r3: dict | None, set_sha256: str,
            sweep: dict | None = None, screen: list | None = None) -> dict:
    """D9Q for lever set `name`: FAIL if a part fails (a long-suite BLOCK fails (b)); PASS iff (a), (b) and (c) pass;
    OWNER if (b)'s long suite is INCONCLUSIVE and nothing fails; else INCOMPLETE. `screen`: the screen selection
    ([REVISED: M23-D9Q-screen]); a screen ESCALATE that nothing fails is ESCALATE (the full b2v3 run decides (b))."""
    parts = {"a": _part_a(name, d9), "b": _part_b(name, quality, set_sha256, screen),
             "c": _part_c(name, spec, r3, quality.get("library_sha256"))}
    statuses = [p["status"] for p in parts.values()]
    overall = ("FAIL" if "fail" in statuses else "ESCALATE" if "escalate" in statuses
               else "PASS" if statuses == ["pass"] * 3
               else "OWNER" if "owner" in statuses and "incomplete" not in statuses and "missing" not in statuses
               else "INCOMPLETE")
    screened = None if screen is None else {"label": f"screen n={len(screen)}", "n": len(screen), "seed": SCREEN_SEED,
                                            "prereg_commit": SCREEN_PREREG_COMMIT,
                                            "outcome": parts["b"]["screen"]["outcome"]}
    return {"row": ROW, "set": name, "verdict": overall, "screen": screened, **parts,
            "throughput": throughput(name, sweep)}


# -- markdown ------------------------------------------------------------------------------------------------
MAIN_HEAD = ["| set | arm | d̄ (nats/token) | 95 % CI | UCB95 | ppl change (CI) | n | G | ICC file / dir | DE file / dir "
             "| b2v3 | KL ratio (b1) | flips b vs c, p (b3) | pass |", "|---|---|---|---|---|---|---|---|---|---|---|---|"
             "---|---|"]
LONG_HEAD = ["| set | arm | d̄ | sd | LCB95 | ppl change | flips b vs c, p (b3) | outcome |",
             "|---|---|---|---|---|---|---|---|"]


def _pct(v: float) -> str:
    return f"{100 * v:+.3f} %"


def main_lines(label: str, g: dict) -> list:
    out = []
    for arm in ARMS:
        v = g["arms"][arm]
        dep = v["dependence"]
        out.append(f"| {label} | {arm} | {v['mean']:+.6f} | [{v['ci_lo']:+.6f}, {v['ci_hi']:+.6f}] | {v['ucb']:+.6f} | "
                   f"{_pct(v['ppl'])} [{_pct(v['ppl_lo'])}, {_pct(v['ppl_hi'])}] | {v['n']} | {v['clusters']} | "
                   f"{_icc_text(dep['file']['icc'], 3)} / {_icc_text(dep['dir']['icc'], 3)} | "
                   f"{dep['file']['design_effect']:.2f} / "
                   f"{dep['dir']['design_effect']:.2f} | {v['b2']} | {v['ratio']:.3f} ({v['b1']}) | "
                   f"{_flips(v)} ({v['b3']}) | "
                   f"{v['pass']} |")
    return out


def long_lines(label: str, g: dict) -> list:
    out = []
    for arm in ARMS:
        v = g["arms"][arm]
        out.append(f"| {label} | {arm} | {v['mean']:+.6f} | {v['sd']:.4f} | {v['lcb']:+.6f} | {_pct(v['ppl'])} | "
                   f"{_flips(v)} | {g['outcome']} |")
    return out


def _flips(v: dict) -> str:
    return f"{v['b3_b']} vs {v['b3_c']} ({v['b3_versus']}, {v['b3_rule']}), {v['flip_p']:.3g}"


def controls_lines(checked: dict) -> list:
    lines = ["Main shape:", "", *MAIN_HEAD]
    for name, note in ((NEGATIVE, "must fail"), (INJ_TARGET, "must fail b2v3"), (SELF_CONSISTENCY, "must pass"),
                       (SENSITIVITY, "report-only")):
        lines += main_lines(f"{name} ({note})", checked["gates"][name])
    lines += ["", "| injected spread | d̄ | 95 % CI | UCB95 | b2v3 | literal d̄ / UCB95 (report-only) | long d̄ / LCB95 |",
              "|---|---|---|---|---|---|---|"]
    for name in INJ_SPREAD:
        s = checked["spread"][name]
        note = "must fail" if name == INJ_SPREAD_REQUIRED else "boundary, report-only"
        lines.append(f"| {name} ({note}) | {s['main']['mean']:+.6f} | [{s['main']['ci_lo']:+.6f}, "
                     f"{s['main']['ci_hi']:+.6f}] | {s['main']['ucb']:+.6f} | {s['main']['b2']} | "
                     f"{s['literal']['mean']:+.6f} / {s['literal']['ucb']:+.6f} | {s['long']['mean']:+.6f} / "
                     f"{s['long']['lcb']:+.6f} |")
    lines += ["", "Long suite:", "", *LONG_HEAD]
    for name in CONTROL_SETS:
        lines += long_lines(name, checked["longs"][name])
    lines += ["", "| framing | rows | R's mean NLL (<= 4.0) | special top-1 share (<= 0.10) |", "|---|---|---|---|"]
    for part, f in checked["framing"].items():
        lines.append(f"| {part} | {f['rows']} | {f['ref_nll_mean']:.4f} | {f['special_top1_share']:.4f} |")
    if checked.get("screen") is not None:
        lines += ["", "Screen decision of the controls: " + ", ".join(
            f"{name} {s['outcome']} (must {'FAIL' if name == INJ_TARGET else 'PASS'})"
            for name, s in checked["screen"].items())]
    lines += ["", "Run validity: " + ("all required controls behave and the framing checks pass" if not
                                     checked["problems"] else "INVALID: " + "; ".join(checked["problems"]))]
    return lines


def screen_lines(doc: dict) -> list:
    """A screen verdict's decision paragraph (none for a full run)."""
    s, b = doc["screen"], doc["b"]
    if s is None:
        return []
    arms = "; ".join(f"{a} UCB95 {v['ucb']:+.6f}, LCB95 {v['lcb']:+.6f}" for a, v in b["screen"]["arms"].items())
    out = [f"Screen ([REVISED: M23-D9Q-screen], owner decision 2026-10-15): {s['label']} of {PROMPT_SET} shape "
           f"{b['shape']} (`screen_select`, seed {s['seed']}, stratified by source file), pre-registration "
           f"`{s['prereg_commit']}`. Per arm the b2v3 CR1 bound: FAIL iff LCB95 > {SCREEN_FAIL_LCB} on an arm; PASS "
           f"iff UCB95 < {SCREEN_PASS_UCB} on both arms with b1 and b3; else ESCALATE. Outcome: **{s['outcome']}** "
           f"({arms}).", ""]
    if s["outcome"] == "ESCALATE":
        out += [f"ESCALATE is not a PASS: the full b2v3 run decides (b): `{escalate_command(doc['set'])}`, then "
                f"`quality-verdict --lever-set {doc['set']}` (without `--screen`) on its report.", ""]
    return out


def markdown(doc: dict, paths: dict) -> str:
    """The verdict's report page: numbers, statuses and input paths only."""
    a, b, c = doc["a"], doc["b"], doc["c"]
    s = doc.get("screen")
    lines = [f"# D9Q {doc['set']}: {doc['verdict']}" + ("" if s is None else f" ({s['label']}: {s['outcome']})"), "",
             "tasks/m23-spec.md D9Q with [REVISED: M23-D9Q-b2v3] and [REVISED: M23-D9Q-b3v3]: PASS iff (a) D9 envelope, "
             "(b) the main b2v3 gate (with b1, b3 v3) and a long suite that does not BLOCK, and (c) greedy spec == "
             "plain; an INCONCLUSIVE long suite returns the default decision to the owner.", "",
             f"Pre-registration `{b['prereg_commit']}` (b3 v3 `{b['b3v3_commit']}`). Library: libcere_ktest.so sha256 "
             f"`{b['ktest_sha256']}`. Shape {b['shape']}, clusters by {b['level']}. Inputs: "
             + ", ".join(f"{k} `{v}`" for k, v in paths.items()), "", f"## (b) quality: {b['status']}", "",
             *screen_lines(doc),
             (f"Main gate on {PROMPT_SET} shape {b['shape']}: PASS iff the one-sided 95 % cluster-robust (CR1, t(G-1)) "
              f"upper bound of the mean per-prompt dNLL < {DELTA} nats/token on both arms, with b1 (ratio <= "
              f"{RATIO_LIMIT}) and b3 v3 (alpha {ALPHA}, pi {B3V3_PI[0]}/{B3V3_PI[1]} unless prefill-neutral)."
              if s is None else
              "The b2v3 gate's numbers on the screen sample (its b2v3 / pass columns are report-only here: the screen "
              "decision above decides (b))."), "", *MAIN_HEAD, *main_lines(doc["set"], b["gate"]), "",
             *[f"- {'FAIL' if s is None else 'b2v3 on the sample'}: {f}" for f in b["gate"]["failures"]], "",
             f"Long-context suite (separate, never pooled, detection-only): BLOCK iff LCB95 > {DELTA} on an arm, "
             f"INCONCLUSIVE iff d̄ > {LONG_INCONCLUSIVE}. Outcome: **{b['long']['outcome']}**.", "", *LONG_HEAD,
             *long_lines(doc["set"], b["long"]), ""]
    t = doc.get("throughput")
    lines += ["Throughput beside the quality numbers (not part of PASS/FAIL): "
              + ("no perf sweep given" if t is None else f"decode_r1_spec {t['decode_r1_spec']} tok/s "
                 f"(perf sweep {t['utc']})"), "", "### Controls (same run)", "", *controls_lines(b["controls"]), ""]
    lines += [f"## (a) envelope: {a['status']}", ""]
    if a["status"] != "missing":
        lines += [f"Candidate runs {a['candidate_runs']}/{D9_SERIES_RUNS}, within the envelope "
                  f"{a['candidate_within']} (predecessor {a['predecessor_within']}/{D9_SERIES_RUNS}); envelope "
                  f"failures: {'; '.join(a['envelope_failures']) or 'none'}. First divergence earlier (report-only): "
                  f"{' '.join(a['first_divergence_earlier']) or 'none'}.", ""]
    lines += [f"## (c) lossless speculation: {c['status']}", "",
              f"{c['ktest_suite']} {c['ktest_class']} (GPU1, no skip): {c['ktest']} ({c['ktest_tests']} tests, "
              f"{c['ktest_utc']});"
              f" R3 (GPU0): {c['r3']}" + (f" ({c['r3_line']}, {c['r3_spec_profile_id']}, {c['r3_utc']})"
                                          if c["r3"] != "pending" else " (runs on the set's spec profile)")
              + f"; candidate library `{c['library_sha256']}`", ""]
    return "\n".join(lines)


def triage_markdown(doc: dict, paths: dict) -> str:
    """A triage's report page ([REVISED: owner 2026-10-16 screen tiers]): numbers, the decision and input paths."""
    arms = "; ".join(f"{a} UCB95 {v['ucb']:+.6f}, LCB95 {v['lcb']:+.6f}" for a, v in doc["screen"]["arms"].items())
    return "\n".join([
        f"# D9Q triage {doc['set']}: {doc['decision']} ({doc['label']}: {doc['outcome']})", "",
        "[REVISED: owner 2026-10-16 screen tiers]: per output-changing candidate, the screen's per-arm b2v3 CR1 bound "
        f"on {doc['label']} of {PROMPT_SET} shape {doc['shape']} (`triage_select`, seed {doc['seed']}, stratified by "
        f"source file), pre-registration `{doc['screen_prereg_commit']}`: FAIL iff LCB95 > {SCREEN_FAIL_LCB} on an "
        f"arm; PASS iff UCB95 < {SCREEN_PASS_UCB} on both arms with b1 and b3; else ESCALATE. FAIL drops the "
        f"candidate; PASS or ESCALATE keeps it for the bundle. Outcome: **{doc['outcome']}** ({arms}).", "",
        "**Never a verdict**: a triage report never satisfies `quality-verdict` or a default flip. The bundle of every "
        f"kept candidate is screened at n={SCREEN_N} (`test --suite quality --shape {doc['shape']} --lever-set "
        "BUNDLE --screen`, then `quality-verdict --screen`).", "",
        f"Pre-registration `{doc['prereg_commit']}` (b3 v3 `{doc['b3v3_commit']}`). Library: libcere_ktest.so sha256 "
        f"`{doc['ktest_sha256']}`. Clusters by {doc['level']}. Inputs: "
        + ", ".join(f"{k} `{v}`" for k, v in paths.items()), "",
        "The b2v3 gate's numbers on the triage sample (its b2v3 / pass columns are report-only).", "", *MAIN_HEAD,
        *main_lines(doc["set"], doc["gate"]), "",
        f"Long-context suite (report-only here): **{doc['long']['outcome']}**.", "", *LONG_HEAD,
        *long_lines(doc["set"], doc["long"]), "", "### Controls (same run)", "", *controls_lines(doc["controls"])])


def attribution_markdown(doc: dict, paths: dict) -> str:
    """An attribution's report page ([REVISED: owner 2026-10-16 screen tiers]): means, ids and input paths only."""
    def cells(v: dict) -> str:
        return " | ".join(f"{v[k]:+.6f}" for k in D_KEYS)
    lines = [f"# D9Q attribution {doc['bundle']} (top {doc['k']} prompts, diagnostic only)", "",
             "[REVISED: owner 2026-10-16 screen tiers]: on a bundle FAIL or ESCALATE, the bundle report's top "
             f"{doc['k']} main prompts by dNLL (the mean of both arms' d_i) rerun per candidate, one candidate on at a "
             "time (`test --suite quality --lever-set C --prompt-ids IDS`). **Never a verdict**: the bundle is "
             f"re-screened at n={SCREEN_N} without the culprit(s). The bundle's mean is on the prompts its own dNLL "
             "selected, so it reads high against the reruns.", "",
             "Inputs: " + ", ".join(f"{k} `{v}`" for k, v in paths.items()), "",
             "Prompts: " + " ".join(doc["ids"]), "",
             "| lever set | d̄ pre | d̄ dec | d̄ both |", "|---|---|---|---|",
             *[f"| {r['set']} | {cells(r['mean'])} |" for r in doc["candidates"]],
             f"| sum of the candidates | {cells(doc['sum'])} |",
             f"| bundle {doc['bundle']} | {cells(doc['bundle_mean'])} |",
             f"| interaction (bundle - sum) | {cells(doc['interaction'])} |"]
    return "\n".join(lines)


def baseline_markdown(doc: dict) -> str:
    """The baseline (every eligible shape) and the shape decision: numbers, statuses and input paths only."""
    dec = doc["decision"]
    long = doc.get("long_suite") or {}
    lines = ["# D9Q-b2v3 baseline and shape decision", "",
             f"Pre-registration `{PREREG_COMMIT}`. No candidate data. Every probability below is a simulation "
             "estimate with its Monte Carlo SE and a 95 % interval from an outer cluster bootstrap of the baseline "
             f"({OUTER} x {INNER}), not a guarantee. Null values x = NLL(P.dec) - NLL(R) per prompt; the prefill arm "
             "has no null spread of its own (P.pre is R, p_repeat is bit-identical), so x is its surrogate and both "
             "arms share these numbers.", ""]
    if long:
        lines += [f"Long suite (detection-only, n={long['n']} of {LONG_PROMPTS}): {long['survivors']} original cuts "
                  f"survive ({long['survivors_overlapping_main']} overlap main-set text, allowed for originals), "
                  f"{long['refills']} refills (per class {long['per_class']}).", ""]
    for s in SHAPES:
        v = doc["shapes"][s]
        if v.get("oc") is None:
            lines += [f"## Shape {s}: ineligible", "",
                      f"Valid prompts {v['valid']} < {SHAPE_MIN_PROMPTS.get(s, 0)}: no baseline, does not qualify.", ""]
            continue
        n = v["check"]["null"]
        lines += [f"## Shape {s}: {v['report']} (library sha256 `{v['ktest_sha256']}`)", "",
                  f"Valid prompts {v['valid']} (frozen n {n['n']}); corpus checks {v['corpus_ok']}. Null x: mean "
                  f"{n['mean']:+.6f}, sd {n['sd']:.4f}; ICC file {_icc_text(n['dependence']['file']['icc'], 4)}, dir "
                  f"{_icc_text(n['dependence']['dir']['icc'], 4)}; design effect file "
                  f"{n['dependence']['file']['design_effect']:.3f}"
                  f" (G {n['dependence']['file']['clusters']}), dir {n['dependence']['dir']['design_effect']:.3f} (G "
                  f"{n['dependence']['dir']['clusters']}); cluster level (rule 3): **{v['check']['level']}**.", "",
                  "| shift μ | P̂(PASS) | P̂(FAIL) | MC SE | 95 % interval of P̂(PASS) |", "|---|---|---|---|---|"]
        for r in v["oc"]["shifts"].values():
            lines.append(f"| {r['mu']:+.3f} | {r['p_pass']:.4f} | {r['p_fail']:.4f} | {r['mc_se']:.4f} | "
                         f"[{r['pass_lo']:.4f}, {r['pass_hi']:.4f}] |")
        lines += ["", *controls_lines(v["check"]), ""]
    lines += ["## Decision (rule 5)", "",
              "| shape | +0.01 calibration check: P̂(FAIL \\| +0.01) | MC SE | 95 % interval | upper end >= "
              f"{FAIL_POWER} | P̂(PASS \\| 0) | qualifies |", "|---|---|---|---|---|---|---|"]
    for s in SHAPES:
        c = dec["calibration"][s]
        if c is None:
            lines.append(f"| {s} | - | - | - | - | - | {dec['qualifies'][s]} (ineligible) |")
        else:
            lines.append(f"| {s} | {c['p_fail']:.4f} | {c['mc_se']:.4f} | [{c['fail_lo']:.4f}, {c['fail_hi']:.4f}] | "
                         f"{c['ok']} | {dec['p_pass0'][s]:.4f} | {dec['qualifies'][s]} |")
    lines += ["", "Both arms share x, so the check reads the same on both."
              + ("" if dec["delta"] is None else f" Difference P̂_S - P̂_L of P(PASS | 0) {dec['delta']:+.4f}, 95 % "
                 f"interval [{dec['delta_lo']:+.4f}, {dec['delta_hi']:+.4f}]; tie {dec['tie']}."), "",
              f"**Choice: {dec['choice']}**" if dec["choice"] else f"**REPORT to the owner: {dec['report']}**", ""]
    return "\n".join(lines)
