"""K6 / row K13S (tasks/m29-spec.md D6, K6, C11, M29S2-13; ledger M8, ledger table R K13S): the phase-S
accept/resample kernel `kernels/k13s_spec_accept.{h,cu}` (SGLang's target-only rule with thresholds 1.0).

Suite `kernels_spec` (GPU1 lane: `native_dev.py --gpu 1 test --suite kernels_spec`). Inputs: 64 fixed rows of the
real vocab (248,320) built by K12's own sampler stage (`cere_ktest_sample_probs`: fp32 probabilities and the D3 keep
mask) from seeded synthetic logits: peaked, Zipf and near-uniform logits under top_k/top_p/min_p truncations, with the
draft at rank 1, the middle rank and the last rank of the kept set, or outside it (the most probable truncated id),
plus four rows where the kept set is the draft alone (p(d) = 1). Every case reports row "K13S". Checks:
- distribution (2^20 coins per case, seed fixed, keys 0..2^20 - 1): the committed first token's empirical
  distribution vs p (the kept mass renormalized): chi-square over bins of expected count >= 5 (the draft alone, the
  top ids alone, the rest grouped in id order), p >= ALPHA / 64 (Bonferroni); zero committed ids outside the keep
  mask; a draft outside the support never accepted; p(d) = 1 always accepted; a repeat is bit-identical;
- exact rule (10,000 seeds per case, key fixed): a host reference (torch on CPU for the counter, Python floats for the
  fp64 sums, both IEEE-exact) of the D6 rule: u = uniform(seed, key, 4, 5), accept iff kept(d) and
  u < fp64(p[d]) / S, else the inverse CDF of uniform(seed, key, 6, 7) over the kept ids without d (k12 draw's chunked
  fp64 sums), equals the kernel's decision and committed id bit for bit;
- chain (`cere_ktest_spec_accept`, 2,000 seeds per chain): the committed ids and count of whole chains (3 drafts with
  all-accept coverage, a draft outside the support at row 1, no draft) equal the host reference, the bonus row being
  K12's plain draw (channels 0, 1 at key0 + n).
Report rows hold scalars only (case names, counts, p values); never ids.

Native entry points (oracle.py calling convention: device pointers `void *`, integers `int64_t`, the stream last;
0 = success). words = ceil(vocab / 32).

    int cere_ktest_spec_row_trials(const void *probs, const void *mask, int64_t vocab, int64_t draft,
                                   int64_t first_seed, int64_t seed_step, int64_t key0, int64_t key_step,
                                   int64_t trials, void *ids, void *accepted, cudaStream_t stream);
        probs f32 [vocab], mask uint32 [words] (K12's outputs); trial t: seed first_seed + t * seed_step, key
        key0 + t * key_step (uint32); ids int32 [trials] (committed id), accepted uint8 [trials] (k13s::row_trials).

    int cere_ktest_spec_accept(const void *probs, const void *masks, int64_t vocab, const void *drafts, int64_t n,
                               int64_t seed, int64_t key0, int64_t chains, void *committed, void *counts,
                               cudaStream_t stream);
        probs f32 [n + 1, vocab], masks uint32 [n + 1, words], drafts int32 [n]; chain c runs k13s::spec_accept with
        seed + c and key0; committed int32 [chains, n + 1], counts int32 [chains].

    int cere_ktest_sample_probs(...)  (test_kernels_k12_sampler.py's docstring).
"""
from __future__ import annotations

import functools
import math
import unittest

import torch

import oracle

ROW = "K13S"
VOCAB = oracle.D6_VOCAB
COINS = 1 << 20
REF_SEEDS = 10_000
CHAINS = 2_000
ALPHA = 1e-3
CASES = 64
# seeds with a non-zero high word (the counter mixes both 32-bit halves)
DIST_SEED = 0x2545F4914F6CDD1D
REF_SEED = 0x5851F42D4C957F2D
CHAIN_SEED = 0x14057B7EF767814F
CONFIGS = {  # (temperature, top_k, top_p, min_p)
    "k20p95": (1.0, 20, 0.95, 0.0),
    "t07k50p90": (0.7, 50, 0.9, 0.0),
    "p95": (1.0, VOCAB, 0.95, 0.0),
    "t13p80": (1.3, VOCAB, 0.8, 0.0),
    "k40minp05": (1.0, 40, 1.0, 0.05),
}
KINDS = ("peaked", "zipf", "near_uniform")
DRAFTS = ("top1", "mid", "last", "outside")


# -- inputs ---------------------------------------------------------------------------------------------------------


def _logits(kind: str, seed: int) -> torch.Tensor:
    g = oracle.generator(seed)
    noise = torch.randn(VOCAB, generator=g, device=oracle.DEVICE, dtype=torch.float32)
    if kind == "peaked":
        return noise * 3.0
    if kind == "zipf":
        rank = torch.randperm(VOCAB, generator=g, device=oracle.DEVICE).float() + 1.0
        return -1.1 * torch.log(rank) + 0.01 * noise
    if kind == "near_uniform":
        return noise * 0.05
    if kind == "spike":
        out = noise * 0.5
        out[int(torch.randint(VOCAB, (1,), generator=g, device=oracle.DEVICE))] = 40.0
        return out
    raise oracle.OracleError(f"unknown logits kind {kind}")


def _probs(kind: str, seed: int, params: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    processed = _logits(kind, seed).contiguous()
    probs = oracle.empty((VOCAB,), torch.float32)
    mask = oracle.empty(((VOCAB + 31) // 32,), torch.int32)
    lse = oracle.empty((1,), torch.float64)
    oracle.call("cere_ktest_sample_probs", processed, VOCAB, *params, probs, mask, lse)
    return probs, mask


def _draft(probs: torch.Tensor, kept: torch.Tensor, which: str) -> int:
    if which == "outside":
        masked = torch.where(kept, torch.full_like(probs, -1.0), probs)
        return int(torch.argmax(masked))
    ids = torch.nonzero(kept).reshape(-1)
    order = torch.argsort(-probs[ids].double(), stable=True)
    rank = {"top1": 0, "mid": ids.numel() // 2, "last": ids.numel() - 1}[which]
    return int(ids[order[rank]])


@functools.cache
def _cases() -> list[dict]:
    """The 64 fixed rows: kinds x configs x drafts (60), then four rows whose kept set is the draft alone."""
    cases = []
    seed = 1000
    for kind in KINDS:
        for cname, params in CONFIGS.items():
            for which in DRAFTS:
                cases.append({"name": f"{kind}_{cname}_{which}", "kind": kind, "seed": seed, "params": params,
                              "draft": which})
                seed += 1
    for kind, params in (("peaked", (1.0, 1, 1.0, 0.0)), ("zipf", (1.0, 1, 1.0, 0.0)),
                         ("near_uniform", (1.0, 1, 1.0, 0.0)), ("spike", (1.0, VOCAB, 0.5, 0.0))):
        cases.append({"name": f"{kind}_single_pd1", "kind": kind, "seed": seed, "params": params, "draft": "top1"})
        seed += 1
    if len(cases) != CASES:
        raise oracle.OracleError(f"{len(cases)} cases, expected {CASES}")
    return cases


@functools.cache
def _row(index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """(probs, mask, kept, draft) of case `index`."""
    case = _cases()[index]
    probs, mask = _probs(case["kind"], case["seed"], case["params"])
    kept = oracle.d6_kept(mask)
    return probs, mask, kept, _draft(probs, kept, case["draft"])


def _trials(probs, mask, draft: int, first_seed: int, seed_step: int, key0: int, key_step: int, n: int):
    ids = oracle.empty((n,), torch.int32)
    accepted = oracle.empty((n,), torch.uint8)
    oracle.call("cere_ktest_spec_row_trials", probs, mask, VOCAB, draft, first_seed, seed_step, key0, key_step, n,
                ids, accepted)
    return ids, accepted


# -- distribution helpers -------------------------------------------------------------------------------------------


def _chi2_sf(x: float, df: int) -> float:
    return float(torch.special.gammaincc(torch.tensor(df / 2.0, dtype=torch.float64),
                                         torch.tensor(x / 2.0, dtype=torch.float64)))


def _bins(probs: torch.Tensor, kept: torch.Tensor, draft: int, n: int) -> tuple[torch.Tensor, list[float]]:
    """bin of each id (-1 outside the support) and the expected counts: the draft alone and up to 30 top ids alone
    (expected >= 5), the rest grouped in id order into up to 20 groups, each group merged forward until >= 5."""
    ids = torch.nonzero(kept).reshape(-1)
    mass = probs[ids].double()
    mass = mass / mass.sum()
    of = torch.full((VOCAB,), -1, dtype=torch.int64, device=probs.device)
    expected: list[float] = []
    alone = torch.zeros(ids.numel(), dtype=torch.bool, device=probs.device)
    order = torch.argsort(-mass, stable=True)
    for r in order[:30].tolist():
        if float(mass[r]) * n >= 5.0:
            alone[r] = True
    if bool(kept[draft]):
        r = int((ids < draft).sum())
        if float(mass[r]) * n >= 5.0:
            alone[r] = True
    for r in torch.nonzero(alone).reshape(-1).tolist():
        of[ids[r]] = len(expected)
        expected.append(float(mass[r]) * n)
    rest_ids = ids[~alone]
    rest_mass = mass[~alone]
    rest_total = float(rest_mass.sum()) * n
    if rest_ids.numel() == 0 or rest_total <= 0.0:
        return of, expected
    groups = max(1, min(20, int(rest_total // 5.0)))
    before = torch.cumsum(rest_mass, 0) - rest_mass
    group = torch.clamp(((before + rest_mass / 2.0) / rest_mass.sum() * groups).floor().long(), max=groups - 1)
    gmass = torch.zeros(groups, dtype=torch.float64, device=probs.device).index_add_(0, group, rest_mass * n)
    merged, acc, base = [], 0.0, len(expected)
    out_of = []
    for g in range(groups):
        acc += float(gmass[g])
        out_of.append(len(merged))
        if acc >= 5.0:
            merged.append(acc)
            acc = 0.0
    if acc > 0.0:
        if merged:
            merged[-1] += acc
            out_of = [min(m, len(merged) - 1) for m in out_of]
        else:
            merged.append(acc)
    of[rest_ids] = torch.tensor(out_of, dtype=torch.int64, device=probs.device)[group] + base
    expected.extend(merged)
    return of, expected


def _gof(of: torch.Tensor, expected: list[float], got: torch.Tensor) -> tuple[int, float]:
    if len(expected) < 2:
        return len(expected), 1.0
    bins = of[got.long()]
    observed = torch.bincount(bins[bins >= 0], minlength=len(expected)).double().tolist()
    x = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    return len(expected), _chi2_sf(x, len(expected) - 1)


class K13SpecAccept(unittest.TestCase):
    def test_distribution_support_and_repeat(self):
        failing = 0
        alpha = ALPHA / CASES  # Bonferroni over the 64 cases
        for index, case in enumerate(_cases()):
            name = case["name"]
            with self.subTest(case=name):
                probs, mask, kept, draft = _row(index)
                ids, accepted = _trials(probs, mask, draft, DIST_SEED + index, 0, 0, 1, COINS)
                again, again_accepted = _trials(probs, mask, draft, DIST_SEED + index, 0, 0, 1, COINS)
                oracle.check_bit_exact(self, ROW, f"repeat_ids_{name}", again, ids)
                oracle.check_bit_exact(self, ROW, f"repeat_accepted_{name}", again_accepted, accepted)
                outside = int((~kept[ids.long()]).sum())
                oracle.check_count(self, ROW, f"outside_support_{name}", outside, expected=0)
                flags = accepted.long()
                bad_flag = int(((flags == 1) & (ids.long() != draft)).sum()) + int((flags > 1).sum())
                oracle.check_count(self, ROW, f"accepted_is_draft_{name}", bad_flag, expected=0)
                taken = int(flags.sum())
                if not bool(kept[draft]):
                    oracle.check_count(self, ROW, f"outside_draft_accepted_{name}", taken, expected=0)
                    oracle.check_count(self, ROW, f"outside_draft_committed_{name}",
                                       int((ids.long() == draft).sum()), expected=0)
                if int(kept.sum()) == 1 and bool(kept[draft]):
                    oracle.check_count(self, ROW, f"pd1_rejected_{name}", COINS - taken, expected=0)
                of, expected = _bins(probs, kept, draft, COINS)
                bins, p = _gof(of, expected, ids)
                passed = bins < 2 or p >= alpha
                failing += not passed
                oracle.report_row(ROW, f"first_token_gof_{name}", phase="gof", bins=bins, p=p, alpha=alpha,
                                  kept=int(kept.sum()), accepted=taken, coins=COINS, testable=bins >= 2,
                                  passed=passed)
        oracle.check_count(self, ROW, "first_token_gof_failing", failing, expected=0)

    def test_non_finite_rows_hardening(self):
        """SS-4: rows of NaN / +inf logits reach the accept rule as k12 probabilities (never NaN, one kept id at
        least): every trial commits a kept id, a draft that is the only kept id is always accepted, any other draft
        is rejected and resampled to the kept id."""
        nan, inf = float("nan"), float("inf")
        params = (1.0, 20, 0.95, 0.0)
        rows = {"all_nan": torch.full((VOCAB,), nan, device=oracle.DEVICE),
                "all_minus_inf": torch.full((VOCAB,), -inf, device=oracle.DEVICE)}
        nan_row = _logits("peaked", 77)
        nan_row[11] = nan
        rows["nan_among_numbers"] = nan_row
        inf_row = _logits("peaked", 78)
        inf_row[23] = inf
        rows["inf_among_numbers"] = inf_row
        for name, row in rows.items():
            with self.subTest(case=name):
                probs = oracle.empty((VOCAB,), torch.float32)
                mask = oracle.empty(((VOCAB + 31) // 32,), torch.int32)
                lse = oracle.empty((1,), torch.float64)
                oracle.call("cere_ktest_sample_probs", row.contiguous(), VOCAB, *params, probs, mask, lse)
                kept = oracle.d6_kept(mask)
                self.assertTrue(bool(torch.isfinite(probs).all()), name)
                self.assertTrue(int(kept.sum()) >= 1, name)
                if name != "nan_among_numbers":
                    only = int(torch.nonzero(kept).reshape(-1)[0])
                    self.assertEqual(int(kept.sum()), 1)
                    self.assertEqual(only, 23 if name == "inf_among_numbers" else 0)
                    for draft, accepts in ((only, True), (only + 1, False)):
                        ids, accepted = _trials(probs, mask, draft, DIST_SEED, 0, 0, 1, 256)
                        self.assertEqual(set(ids.tolist()), {only})
                        self.assertEqual(set(accepted.tolist()), {1 if accepts else 0})
                else:
                    ids, accepted = _trials(probs, mask, int(torch.nonzero(kept).reshape(-1)[0]), DIST_SEED, 0, 0, 1,
                                            256)
                    self.assertTrue(bool(kept[ids.long()].all()))
                    self.assertFalse(bool(kept[11]))

    def test_exact_rule_vs_host_reference(self):
        rejected_total = 0
        for index, case in enumerate(_cases()):
            name = case["name"]
            with self.subTest(case=name):
                probs, mask, kept, draft = _row(index)
                key = 7_919 * (index + 1)
                ids, accepted = _trials(probs, mask, draft, REF_SEED + index * REF_SEEDS, 1, key, 0, REF_SEEDS)
                ref = oracle.D6RowRef(probs, kept, draft)
                seeds = [REF_SEED + index * REF_SEEDS + t for t in range(REF_SEEDS)]
                want_ids, want_flags = oracle.d6_ref_row(ref, seeds, [key] * REF_SEEDS)
                got_flags = accepted.cpu().long().tolist()
                got_ids = ids.cpu().long().tolist()
                decision = sum(int(a != b) for a, b in zip(got_flags, want_flags))
                committed = sum(int(a != b) for a, b in zip(got_ids, want_ids))
                rejected = REF_SEEDS - sum(want_flags)
                rejected_total += rejected
                oracle.check_count(self, ROW, f"ref_decision_{name}", decision, expected=0)
                oracle.check_count(self, ROW, f"ref_committed_{name}", committed, expected=0)
                oracle.report_row(ROW, f"ref_coverage_{name}", phase="reported", seeds=REF_SEEDS, rejected=rejected)
        self.assertTrue(rejected_total > 0, f"{ROW}: the host reference saw no rejection")

    def test_chain_vs_host_reference(self):
        # case index per verify row (row j < n's draft is that case's draft; the last row is the bonus row): peaked
        # rank-1 drafts with a p(d) = 1 row (all-accept coverage), a draft outside the support at row 1, no draft
        chains = {
            "chain3_top1": [0, 60, 4, 8],
            "chain3_outside_row1": [1, 3, 21, 9],
            "chain0_bonus": [5],
        }
        for name, rows in chains.items():
            with self.subTest(chain=name):
                n = len(rows) - 1
                data = [_row(i) for i in rows]
                probs = torch.stack([d[0] for d in data]).contiguous()
                masks = torch.stack([d[1] for d in data]).contiguous()
                drafts = torch.tensor([data[j][3] for j in range(n)] or [0], dtype=torch.int32,
                                      device=oracle.DEVICE)
                key0 = 4_242 + 17 * n
                committed = oracle.empty((CHAINS, n + 1), torch.int32)
                counts = oracle.empty((CHAINS,), torch.int32)
                oracle.call("cere_ktest_spec_accept", probs, masks, VOCAB, drafts if n > 0 else None, n,
                            CHAIN_SEED, key0, CHAINS, committed, counts)
                refs = [oracle.D6RowRef(data[j][0], data[j][2], data[j][3]) for j in range(n)]
                bonus_ref = oracle.D6RowRef(data[n][0], data[n][2], None)
                seeds = [CHAIN_SEED + c for c in range(CHAINS)]
                per_row = [oracle.d6_ref_row(refs[j], seeds, [key0 + j] * CHAINS) for j in range(n)]
                bonus = [bonus_ref.draw(u) for u in oracle.d6_uniforms(seeds, [key0 + n] * CHAINS, 0, 1)]
                got_counts = counts.cpu().long().tolist()
                got = committed.cpu().long().tolist()
                wrong = 0
                full = 0
                for c in range(CHAINS):
                    want = []
                    for j in range(n):
                        ids, flags = per_row[j]
                        want.append(ids[c])
                        if not flags[c]:
                            break
                    else:
                        want.append(bonus[c])
                        full += 1
                    wrong += int(got_counts[c] != len(want) or got[c][:len(want)] != want)
                oracle.check_count(self, ROW, f"chain_{name}", wrong, expected=0)
                oracle.report_row(ROW, f"chain_coverage_{name}", phase="reported", chains=CHAINS, drafts=n,
                                  all_accepted=full)
                if name == "chain3_outside_row1":
                    oracle.check_count(self, ROW, f"chain_past_outside_{name}",
                                       sum(int(k > 2) for k in got_counts), expected=0)
                else:
                    self.assertTrue(full > 0, f"{ROW} {name}: no chain reached the bonus row")


if __name__ == "__main__":
    unittest.main()
