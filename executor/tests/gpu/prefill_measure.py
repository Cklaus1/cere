"""M2.2p prefill-step measure and patch-attribution calculus (tasks/m22p-spec.md D4, D5; C1, C2).

Pure Python (no torch, no numpy, no CUDA): the GPU suite `test_numerics_prefill.py` feeds it distances, margins and
host lists, and passes `oracle.rel_l2` as `dist` where whole tensors are compared on the device; the CPU test
`tests/test_prefill_measure.py` drives it on synthetic arrays. Nothing here sees a token id or text.

Conventions:
* A tap key is `(layer, name)` in the native forward order (`model` uses layer -1 and comes last); `name` is the
  layer-free suffix (`self_attn.o_proj`, `mlp.gate`, ...). `d[key]` / `limits[key]` of None, or a key missing from
  `d`, means the name is `absent` from the dump (D3: reported, never a failure).
* A `top1` score is `cere_token_score`: an object with `logit` and `runner_up_logit` attributes, or a 4-sequence
  (id, runner_up, logit, runner_up_logit).
"""
from __future__ import annotations

import math

TARGETS = ("p12", "p13")
MIN_DM = 0.125  # one BF16 logit ulp at |logit| 16-32 (m22p design note D5, M22PS-15)
OUTSIDE = "outside the blocks"  # P-all gives m <= 0: embedding, PLE, combine gates, final mixer, lm_head
NONE = "none"


def _flat(x) -> list:
    """Row-major floats of a scalar sequence or a sequence of rows."""
    out = []
    for v in x:
        if isinstance(v, (int, float)):
            out.append(float(v))
        else:
            out.extend(float(w) for w in v)
    return out


def rel_l2(a, b) -> float:
    """||a - b||_2 / ||b||_2 (b is the reference); inf when a holds a non-finite value."""
    av, bv = _flat(a), _flat(b)
    if len(av) != len(bv):
        raise ValueError(f"{len(av)} values vs reference {len(bv)}")
    if not all(math.isfinite(v) for v in av):
        return math.inf
    denom = math.sqrt(math.fsum(v * v for v in bv))
    if denom == 0.0 or not math.isfinite(denom):
        raise ValueError(f"reference norm is {denom}; rel-L2 is undefined")
    return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(av, bv))) / denom


# -- D4 ---------------------------------------------------------------------------------------------------------------


def pairwise_spread(passes, dist=rel_l2) -> float:
    """e = max over frame pairs f < g of dist(pass f, pass g) (the later frame is the reference): the reference's own
    run-to-run spread of one tap at the same M."""
    passes = list(passes)
    if len(passes) < 2:
        raise ValueError(f"a spread needs >= 2 frames, got {len(passes)}")
    return max(float(dist(passes[f], passes[g])) for f in range(len(passes)) for g in range(f + 1, len(passes)))


def limit(e) -> float:
    """The D4 bar: `oracle.calibrated_limit(e, FLOOR_BLOCK)` = max(1e-2, 2e)."""
    import oracle  # executor/tests/gpu/oracle.py on the GPU; a stand-in in the CPU test

    return oracle.calibrated_limit(float(e), oracle.FLOOR_BLOCK)


def _measured(d, limits, key):
    value, bar = d.get(key), limits.get(key)
    if value is None or bar is None:
        return None
    return float(value), float(bar)


def _exceeds(value: float, bar: float) -> bool:
    return not (math.isfinite(value) and value <= bar)


def first_exceeding(order, d, limits):
    """The first (layer, name) in forward order with d > limit, or None (`none`): {layer, name, d, limit, ratio}."""
    for key in order:
        m = _measured(d, limits, key)
        if m is not None and _exceeds(*m):
            value, bar = m
            return {"layer": key[0], "name": key[1], "d": value, "limit": bar, "ratio": value / bar}
    return None


def per_name(order, d, limits) -> dict:
    """name -> {exceeding, measured, absent, max_ratio (None if never measured)}, names in first-seen order."""
    out: dict = {}
    for key in order:
        row = out.setdefault(key[1], {"exceeding": 0, "measured": 0, "absent": 0, "max_ratio": None})
        m = _measured(d, limits, key)
        if m is None:
            row["absent"] += 1
            continue
        value, bar = m
        row["measured"] += 1
        row["exceeding"] += _exceeds(value, bar)
        ratio = value / bar
        row["max_ratio"] = ratio if row["max_ratio"] is None else max(row["max_ratio"], ratio)
    return out


def excess(ratios, targets=TARGETS) -> dict:
    """Profile, not a gate: per target prompt and tap, its d/limit over the max d/limit of every non-target prompt at
    the same tap. `ratios`: pid -> {key: d/limit or None}. None where the target or every other prompt is absent, or
    the others' max is 0. Targets missing from `ratios` are left out."""
    others = [pid for pid in ratios if pid not in targets]
    out: dict = {}
    for target in targets:
        if target not in ratios:
            continue
        row = {}
        for key, value in ratios[target].items():
            seen = [ratios[o][key] for o in others if ratios[o].get(key) is not None]
            denom = max(seen) if seen else None
            row[key] = None if value is None or not denom else float(value) / float(denom)
        out[target] = row
    return out


def last_row_d(native, ref, dist=rel_l2) -> float:
    """dist of row P-1 only (whole-tensor rel-L2 averages it away, M22PS-09)."""
    return float(dist(native[-1], ref[-1]))


def margin_projection(h_native, h_ref, w_a, w_b) -> float:
    """(h_native - h_ref) . (w_A - w_B) at `model`, row P-1: the first-order margin change of native's error on the
    reference's top-2 lm_head rows (held in memory)."""
    hn, hr, wa, wb = _flat(h_native), _flat(h_ref), _flat(w_a), _flat(w_b)
    if not len(hn) == len(hr) == len(wa) == len(wb):
        raise ValueError(f"lengths {len(hn)}, {len(hr)}, {len(wa)}, {len(wb)} differ")
    return math.fsum((x - y) * (a - b) for x, y, a, b in zip(hn, hr, wa, wb))


def _ranked(logits) -> list:
    """Expert indices best-first, ties to the lowest id (FP32 softmax is monotone: the routed order)."""
    return sorted(range(len(logits)), key=lambda i: (-float(logits[i]), i))


def topk_set(logits, k: int) -> frozenset:
    return frozenset(_ranked(logits)[:k])


def routed_set_first_divergence(layers, k: int = 10):
    """`layers`: [(layer, native router logits [rows][experts], reference `mlp.gate` logits [rows][experts])] in
    forward order. The first (layer, row) where the top-k sets differ: {layer, row, differing (experts in native's
    set not in the reference's), gap (reference k-th minus (k+1)-th logit)}, or None."""
    for layer, native_rows, ref_rows in layers:
        if len(native_rows) != len(ref_rows):
            raise ValueError(f"layer {layer}: {len(native_rows)} native rows vs {len(ref_rows)} reference rows")
        for row, (n, r) in enumerate(zip(native_rows, ref_rows)):
            if len(n) != len(r) or len(r) <= k:
                raise ValueError(f"layer {layer} row {row}: {len(n)} vs {len(r)} experts for top-{k}")
            ns, ranked = topk_set(n, k), _ranked(r)
            rs = frozenset(ranked[:k])
            if ns != rs:
                gap = float(r[ranked[k - 1]]) - float(r[ranked[k]])
                return {"layer": layer, "row": row, "differing": len(ns - rs), "gap": gap}
    return None


# -- D5 ---------------------------------------------------------------------------------------------------------------


def _score(top1) -> tuple:
    if hasattr(top1, "logit"):
        return float(top1.logit), float(top1.runner_up_logit)
    _, _, logit, runner = top1
    return float(logit), float(runner)


def signed_margin(target_logit, top1) -> float:
    """m = target logit - best other logit, from `forward_patch`'s target logit and top-2 score. The target is the
    top-1 iff its logit equals the top-1 logit (then the best other is the runner-up; a tie gives 0, which is not a
    pick); otherwise the best other is the top-1, whether the target is the runner-up or outside the top-2."""
    t = float(target_logit)
    best, runner = _score(top1)
    if not (math.isfinite(t) and math.isfinite(best) and math.isfinite(runner)):
        raise ValueError("non-finite logit")
    if t > best:
        raise ValueError("target logit exceeds the top-1 logit")
    return t - runner if t == best else t - best


def l_star(prefix_margins):
    """L* = the smallest L with m(P-prefix(L)) > 0, or None. `prefix_margins`: {L: m} or a list indexed by L."""
    items = prefix_margins.items() if isinstance(prefix_margins, dict) else enumerate(prefix_margins)
    positive = [L for L, m in items if m > 0]
    return min(positive) if positive else None


def attributed_class(p0: float, p_all: float, class_margins: dict) -> str:
    """D5: P-all m <= 0 -> OUTSIDE; else the class whose P-class alone gives m > 0 (largest m if several); else the
    class with the largest dm = m - p0 if |dm| >= MIN_DM; else NONE."""
    if p_all <= 0:
        return OUTSIDE
    positive = [(m, name) for name, m in class_margins.items() if m > 0]
    if positive:
        return max(positive)[1]
    if not class_margins:
        return NONE
    name, m = max(class_margins.items(), key=lambda t: t[1] - p0)
    return name if abs(m - p0) >= MIN_DM else NONE


def top_singles(single_margins: dict, p0: float, n: int = 10) -> list:
    """The n (layer, name) P-single patches with the largest dm, [{key, m, dm}] best-first (ties: forward key order)."""
    rows = [{"key": key, "m": float(m), "dm": float(m) - p0} for key, m in single_margins.items()]
    rows.sort(key=lambda r: -r["dm"])
    return rows[:n]


def rank_screens(screens: dict, p0: dict) -> list:
    """D7 screen ranking. `screens`: name -> {target: m}; `p0`: target -> m without a patch. Rows best-first by the
    best dm over the targets, then by name: {screen, dm {target: dm}, best_dm, turns_positive, positive (targets
    with m > 0, comma-joined)}."""
    rows = []
    for name, margins in screens.items():
        missing = [t for t in margins if t not in p0]
        if missing:
            raise ValueError(f"screen {name}: no P0 margin for {missing}")
        dm = {t: float(m) - float(p0[t]) for t, m in margins.items()}
        positive = [t for t, m in margins.items() if m > 0]
        rows.append({"screen": name, "dm": dm, "best_dm": max(dm.values()) if dm else 0.0,
                     "turns_positive": bool(positive), "positive": ",".join(positive)})
    rows.sort(key=lambda r: (-r["best_dm"], r["screen"]))
    return rows
