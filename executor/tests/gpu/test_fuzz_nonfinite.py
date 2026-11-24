"""OH-09 (tasks/ops-hardening-spec.md): a seeded NaN/Inf property suite over the router and sampler ktests, container
only.

Suite `fuzz` (`native_dev.py [--gpu 1] test --suite fuzz`, nightly-capable: no model load, no timing, no memory
figure; ON_DEMAND_SUITES). Every trial draws its rows from one seeded CPU generator (SEED), so a run is reproducible
and a failing trial is named by its index in the report rows; the rows hold counts only (no logits, no ids). Each
invariant is one violation count over all trials, which must be 0 (`oracle.check_count`).

Router (KGM-1, `cere_ktest_moe_router`, stated in `test_kernels_k7_moe.py`): ROUTER_TRIALS batches of 1..8 BF16
logit rows of the model's [num_experts] (scale 0.5..8), each row of one kind (ROW_KINDS: finite, sprinkled NaN, +Inf
or -Inf, a mix, all NaN, all -Inf, fewer than K numbers among NaNs). Per row:
* `router_ids_valid`: K distinct ids, each in [0, E);
* `router_ids_ranked`: slot k is the k-th id of the order "logits descending, NaN as -Inf; ties to the lower id"
  (k7_moe.cu: NaN ranks below every number, and NaN and -Inf tie);
* `router_weights_finite`: every weight finite and >= 0;
* `router_weights_zero`: a row whose softmax denominator is not a finite positive number (a NaN or +Inf logit, or
  every logit -Inf) has every weight exactly 0;
* `router_weights_sum`: any other row's weights sum to 1 within 1e-5.

Sampler (SS-1/SS-2, `cere_ktest_sample_probs`, `cere_ktest_sample_draw`, `cere_ktest_sample_top2_f32`, stated in
`test_kernels_k12_sampler.py`): SAMPLER_TRIALS fp32 processed rows of a vocabulary from VOCABS (scale 0.5..8, one
kind as above plus several +Inf), sampled with a random temperature, top_k, top_p and min_p. With "no mass" = no
+Inf and every non-NaN logit -Inf:
* `sampler_probs_finite`: every probability finite and in [0, 1]; the lse never NaN, -Inf exactly for no mass;
* `sampler_nan_is_minus_inf`: unless no mass, probs and lse are bit-identical to the same row with every NaN set to
  -Inf (a NaN carries no mass);
* `sampler_posinf_takes_all`: with a +Inf logit only +Inf ids have probability, all of them the same value;
* `sampler_no_mass_one_hot`: no mass puts p = 1 on the lowest NaN id (id 0 without one) and keeps only it;
* `sampler_draws_valid`: each of DRAWS draws is in [0, vocab), kept by the D3 mask and of probability > 0;
* `sampler_probs_sum`: the probabilities sum to 1 within 1e-5;
* `sampler_top2_order`: top2 is the first two ids of the order "NaN, then numbers descending (+Inf first); ties to
  the lower id" (K1's greedy order).
"""
from __future__ import annotations

import math
import unittest

import torch

import oracle
from test_kernels_k12_sampler import _kept, _native_draw, _native_probs, _native_top2

ROW = "OH09"
SEED = 20261023
ROUTER_TRIALS = 64
SAMPLER_TRIALS = 96
DRAWS = 32
VOCABS = (4_096, 4_099, 32_768)
ROW_KINDS = ("finite", "nan", "posinf", "neginf", "mixed", "all_nan", "all_neginf", "few_numbers")
SAMPLER_KINDS = ROW_KINDS + ("posinf_many",)
SUM_TOL = 1e-5
NAN = float("nan")
INF = float("inf")


def sprinkle(row: torch.Tensor, kind: str, gen: torch.Generator, keep_numbers: int) -> torch.Tensor:
    """`row` (fp32, CPU) with the non-finite entries of `kind`; `keep_numbers`: how many numbers `few_numbers` keeps."""
    n = row.numel()

    def some(value: float, most: int) -> None:
        count = int(torch.randint(1, most + 1, (1,), generator=gen))
        row[torch.randperm(n, generator=gen)[:count]] = value

    if kind == "nan":
        some(NAN, 8)
    elif kind == "posinf":
        some(INF, 1)
    elif kind == "posinf_many":
        some(INF, 4)
    elif kind == "neginf":
        some(-INF, n // 2)
    elif kind == "mixed":
        some(-INF, 8)
        some(NAN, 4)
        if int(torch.randint(0, 2, (1,), generator=gen)):
            some(INF, 2)
    elif kind == "all_nan":
        row.fill_(NAN)
    elif kind == "all_neginf":
        row.fill_(-INF)
        if int(torch.randint(0, 2, (1,), generator=gen)):
            some(NAN, 3)
    elif kind == "few_numbers":
        numbers = row[torch.randperm(n, generator=gen)[:keep_numbers]].clone()
        row.fill_(NAN)
        row[torch.randperm(n, generator=gen)[:keep_numbers]] = numbers
    return row


def random_row(n: int, kind: str, gen: torch.Generator, keep_numbers: int) -> torch.Tensor:
    scale = 0.5 + 7.5 * float(torch.rand((1,), generator=gen))
    return sprinkle(torch.randn(n, generator=gen) * scale, kind, gen, keep_numbers)


def router_order(values: list) -> list:
    """Logits descending with NaN as -Inf; ties to the lower id."""
    return sorted(range(len(values)), key=lambda e: (INF if math.isnan(values[e]) else -values[e], e))


def greedy_order(values: list) -> list:
    """NaN first, then numbers descending (+Inf first); ties to the lower id."""
    return sorted(range(len(values)), key=lambda e: (not math.isnan(values[e]),
                                                     0.0 if math.isnan(values[e]) else -values[e], e))


class Violations:
    """Per-invariant violation counts and the first trial that violated each."""

    def __init__(self, names: tuple):
        self.count = dict.fromkeys(names, 0)
        self.first: dict[str, int] = {}

    def add(self, name: str, trial: int, bad: bool) -> None:
        if bad:
            self.count[name] += 1
            self.first.setdefault(name, trial)

    def check(self, tc: unittest.TestCase, trials: int) -> None:
        for name, bad in self.count.items():
            oracle.report_row(ROW, name, phase="fuzz", seed=SEED, trials=trials, first_trial=self.first.get(name, -1))
            oracle.check_count(tc, ROW, name, bad, expected=0)


ROUTER_CHECKS = ("router_ids_valid", "router_ids_ranked", "router_weights_finite", "router_weights_zero",
                 "router_weights_sum")
SAMPLER_CHECKS = ("sampler_probs_finite", "sampler_nan_is_minus_inf", "sampler_posinf_takes_all",
                  "sampler_no_mass_one_hot", "sampler_draws_valid", "sampler_probs_sum", "sampler_top2_order")


def pick(options: tuple, gen: torch.Generator):
    return options[int(torch.randint(0, len(options), (1,), generator=gen))]


def sampler_case(gen: torch.Generator, vocab: int) -> dict:
    return {"temperature": 0.3 + 1.2 * float(torch.rand((1,), generator=gen)), "top_k": pick((1, 5, 20, vocab), gen),
            "top_p": pick((0.5, 0.9, 1.0), gen), "min_p": pick((0.0, 0.05), gen)}


class NonFiniteFuzz(unittest.TestCase):
    def test_router_kgm1_invariants(self):
        cfg = oracle.text_config()
        num_experts, top_k = cfg["num_experts"], cfg["num_experts_per_tok"]
        gen = torch.Generator().manual_seed(SEED)
        v = Violations(ROUTER_CHECKS)
        rows_run = 0
        for trial in range(ROUTER_TRIALS):
            m = int(torch.randint(1, 9, (1,), generator=gen))
            kinds = [ROW_KINDS[int(torch.randint(0, len(ROW_KINDS), (1,), generator=gen))] for _ in range(m)]
            logits = torch.stack([random_row(num_experts, kind, gen, top_k - 1) for kind in kinds])
            logits = logits.to(torch.bfloat16)
            ids = oracle.empty((m, top_k), torch.int32)
            weights = oracle.empty((m, top_k), torch.float32)
            oracle.call("cere_ktest_moe_router", logits.to(oracle.DEVICE).contiguous(), ids, weights, m, num_experts,
                        top_k)
            ids, weights, values = ids.cpu(), weights.cpu(), logits.float()
            rows_run += m
            for r in range(m):
                row_ids, row_w, row = ids[r].tolist(), weights[r], values[r]
                v.add("router_ids_valid", trial,
                      len(set(row_ids)) != top_k or not all(0 <= e < num_experts for e in row_ids))
                v.add("router_ids_ranked", trial, row_ids != router_order(row.tolist())[:top_k])
                v.add("router_weights_finite", trial, not bool((torch.isfinite(row_w) & (row_w >= 0)).all()))
                degenerate = bool(torch.isnan(row).any() or torch.isposinf(row).any() or torch.isneginf(row).all())
                if degenerate:
                    v.add("router_weights_zero", trial, not bool((row_w == 0).all()))
                else:
                    v.add("router_weights_sum", trial, not abs(float(row_w.double().sum()) - 1.0) <= SUM_TOL)
        oracle.report_row(ROW, "router", phase="fuzz", seed=SEED, trials=ROUTER_TRIALS, rows=rows_run,
                          num_experts=num_experts, top_k=top_k)
        v.check(self, ROUTER_TRIALS)

    def test_sampler_ss1_invariants(self):
        gen = torch.Generator().manual_seed(SEED + 1)
        v = Violations(SAMPLER_CHECKS)
        for trial in range(SAMPLER_TRIALS):
            vocab = VOCABS[int(torch.randint(0, len(VOCABS), (1,), generator=gen))]
            kind = SAMPLER_KINDS[int(torch.randint(0, len(SAMPLER_KINDS), (1,), generator=gen))]
            row = random_row(vocab, kind, gen, 3)
            case = sampler_case(gen, vocab)
            us = torch.rand(DRAWS, generator=gen, dtype=torch.float64)
            nan = torch.isnan(row)
            posinf = torch.isposinf(row)
            no_mass = not bool(posinf.any()) and bool(torch.isneginf(row[~nan]).all())
            processed = row.to(oracle.DEVICE).contiguous()
            probs, mask, lse = _native_probs(processed, case)
            probs, kept, lse = probs.cpu(), _kept(mask, vocab).cpu(), float(lse.item())
            v.add("sampler_probs_finite", trial,
                  not bool((torch.isfinite(probs) & (probs >= 0) & (probs <= 1)).all()) or math.isnan(lse)
                  or (lse == -INF) != no_mass)
            if no_mass:
                only = int(nan.nonzero()[0]) if bool(nan.any()) else 0
                v.add("sampler_no_mass_one_hot", trial,
                      probs[only].item() != 1.0 or int((probs != 0).sum()) != 1
                      or kept.nonzero().flatten().tolist() != [only])
            else:
                want, _, want_lse = _native_probs(row.masked_fill(nan, -INF).to(oracle.DEVICE).contiguous(), case)
                v.add("sampler_nan_is_minus_inf", trial,
                      not torch.equal(probs.view(torch.int32), want.cpu().view(torch.int32))
                      or float(want_lse.item()) != lse)
                v.add("sampler_probs_sum", trial, not abs(float(probs.double().sum()) - 1.0) <= SUM_TOL)
            if bool(posinf.any()):
                held = probs[posinf]
                v.add("sampler_posinf_takes_all", trial,
                      bool((probs[~posinf] != 0).any()) or not bool((held == held[0]).all()))
            drawn = _native_draw(processed, case, us.to(oracle.DEVICE)).cpu().long()
            in_range = bool(((drawn >= 0) & (drawn < vocab)).all())
            v.add("sampler_draws_valid", trial,
                  not in_range or not bool(kept[drawn].all()) or not bool((probs[drawn] > 0).all()))
            top2, _ = _native_top2(processed)
            v.add("sampler_top2_order", trial, top2.cpu().tolist() != greedy_order(row.tolist())[:2])
        oracle.report_row(ROW, "sampler", phase="fuzz", seed=SEED + 1, trials=SAMPLER_TRIALS, draws=DRAWS)
        v.check(self, SAMPLER_TRIALS)


if __name__ == "__main__":
    unittest.main()
