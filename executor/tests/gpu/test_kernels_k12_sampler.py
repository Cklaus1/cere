"""K12 (tasks/m27-spec.md C11, I10; tasks/m28-spec.md C5, C9; ledger table R K12): the M2.7 device sampler kernels
and M2.8's vocab mask vs SGLang's sampler pieces.

Every case reports row "K12". The oracle is the C3 fixture `tests/fixtures/native/sampler_cases.json` (written by
`scripts/native/gen_sampler_fixtures.py` in the pinned image from SGLang's penalties, `Sampler` mask code and
softmax; its absence fails the suite) plus `host/sampler_ref` (C2), reached through `cere_ktest_sample_ref_keep_mask`.
Checks (counts of violations, each must be 0, or bit-exact):
- `process` (mask_dev null for M2.7's cases; a case with a `vocab_mask` runs `cere_ktest_process_masked`): per logit
  |native - SGLang| <= 2^-22 max(1, |x|) + count |f| 2^-23 (D2 arithmetic), the ids the vocab mask clears exactly
  -inf (the fixture's oracle applied SGLang's grammar-mask kernel, so every check below is post-mask); a repeat is
  bit-identical;
- `top2_f32` equals the oracle's greedy (id, runner_up) on every T = 0 case and first-index-wins on ties;
- the kernel's own fp32 probabilities match exp(q - lse(q)) within 1e-6 rel plus half the smallest fp32 subnormal,
  where q = fp64(fp32(processed) / fp32(T)) is SGLang's own fp32 quotient (`logits.div_(T)`, a correctly rounded
  tensor division) and lse(q) its fp64 log-sum-exp (an fp64 quotient moves exp by |x/T| 2^-24 rel, over 1e-6 once
  |x/T| > 16; a probability below 2^-126 is subnormal in fp32), and the kernel's keep mask equals
  `sampler_ref::keep_mask` of **those** probabilities bit for bit (no boundary exclusion);
- draws for the fixture's 4,096 u values per case equal the oracle's ids except within 1e-6 of a CDF boundary when
  the kernel's mask equals the oracle's; when the masks differ (a top_p/min_p boundary id, reported as
  `mask_vs_sglang`) the kernel's draws equal the inverse CDF over its own mask and the oracle disagreement is reported;
- seeded draws (N = 200,000, seeds 0..N-1, key = the case index, channels (0, 1)): the id equals the inverse CDF of
  the u this suite computes from its own murmur_hash32 port (bit-exact counter, D4) except within 1e-6 of a CDF
  boundary; zero draws outside the kernel's mask; T1 (equal-mass CDF bins) and T2 (truncated top-K marginals)
  chi-square GOF against the oracle's distribution, each p >= 1e-4 (C2's construction; power >= 0.99 at w >= 0.019);
- top logprobs (N = 20 and 0): each within 1e-5 of the fp64 oracle, the listed ids a valid top-N;
- `count_commit` increments exactly the committed ids;
- M2.8 masked process (C5): `process` with a random, a single-bit and a partial-last-word mask (padding bits set)
  then `top2_f32` equals torch's masked argmax and `sampler_ref::process` with the same mask; the -inf ids are exactly
  the clear bits of xgrammar 0.2.1 `testing.bitmask_to_bool_mask` (vendored below as `_bitmask_to_bool_mask`: the
  suite rules allow no xgrammar import; the generator checks SGLang's applied mask against the real function); an
  all-ones mask equals mask_dev null bit for bit on every M2.7 case (M2.7's cases keep the null path above);
- M2.8 masked draws (C9): the masked cases run every check above (seeded N = 200,000 with T1/T2 p >= 1e-4 against
  the masked oracle), plus zero seeded draws outside the vocab mask and a cleared id's committed logprob -inf;
- stage time (D12 at R6): forward.cpp sample_row's sampled kernels (process, top2_f32, probabilities, keep_mask,
  draw_seeded, finalize, count_commit) with S1 (T 1.0, top_k 20, top_p 0.95, min_p 0) on the vocab-248,320 rows Z and
  T, timed per decode step by CUDA events over STAGE_STEPS steps; the median step is at most STAGE_BUDGET_US[rows]
  at rows 1 and 4, every committed id is in the kernel's own S1 keep mask and each row's counts hold one per step.
  Budget: the live R6 greedy step is 1 / 52.3 tok/s = 19.1 ms and D12 (sampled >= 95 % of greedy tok/s) leaves
  5 % of it, 0.95 ms, for the sampled stage per decode step. Rows 1 gets half of that, 475 us per row-step: the other
  half covers the per-step host work outside the kernels (the sampling-parameter and logit_bias staging) and event
  noise. Rows 4 gets the whole 0.95 ms for its four row stages together: R6 may decode its 4 prompts as one rows-4
  step, which is no shorter than the rows-1 greedy step, so four stages must fit the same 5 %.
Report rows hold scalars only (case names, counts, p values, microseconds); never ids.

Native entry points (oracle.py calling convention: device pointers `void *`, integers `int64_t`, `float` scalars, the
stream last; 0 = success). All tensors are contiguous; `words` = ceil(vocab / 32).

    int cere_ktest_sample_process(const void *logits, int64_t vocab, const void *counts, float frequency_penalty,
                                  float presence_penalty, float repetition_penalty, const void *bias_ids,
                                  const void *bias_values, int64_t bias_count, void *processed, cudaStream_t stream);
        logits bf16 [vocab]; counts uint32 [vocab] or null (no output tokens); bias_ids int32 / bias_values f32
        [bias_count] or null; processed f32 [vocab]: k12 `process` with mask_dev = null (D2).

    int cere_ktest_process_masked(const void *logits, int64_t vocab, const void *counts, float frequency_penalty,
                                  float presence_penalty, float repetition_penalty, const void *mask,
                                  const void *bias_ids, const void *bias_values, int64_t bias_count, void *processed,
                                  void *ref_processed, cudaStream_t stream);
        As cere_ktest_sample_process with mask_dev = mask (uint32 [words], non-null; id j allowed iff bit j % 32 of
        word j / 32). ref_processed f32 [vocab] or null: when non-null, `sampler_ref::process` with the same mask on
        host copies of the inputs (M2.8, tasks/m28-spec.md C5).

    int cere_ktest_sample_top2_f32(const void *values, int64_t vocab, void *ids, void *scores, cudaStream_t stream);
        values f32 [vocab]; ids int32 [2] (first maximum, first maximum without it); scores f32 [2].

    int cere_ktest_sample_probs(const void *processed, int64_t vocab, float temperature, int64_t top_k, float top_p,
                                float min_p, void *probs, void *mask, void *lse, cudaStream_t stream);
        processed f32 [vocab]; probs f32 [vocab] (the kernel's fp32 probabilities of processed / T); mask uint32
        [words] (D3 keep mask, bit i % 32 of word i / 32 = kept); lse f64 [1].

    int cere_ktest_sample_draw(const void *processed, int64_t vocab, float temperature, int64_t top_k, float top_p,
                               float min_p, const void *us, int64_t n, void *ids, cudaStream_t stream);
        us f64 [n] in [0, 1); ids int32 [n]: the kernel's draw (D4 inverse CDF over its own mask) for each u.

    int cere_ktest_sample_seeded(const void *processed, int64_t vocab, float temperature, int64_t top_k, float top_p,
                                 float min_p, int64_t first_seed, int64_t n_seeds, int64_t key, void *ids,
                                 cudaStream_t stream);
        ids int32 [n_seeds]: draw j uses seed first_seed + j at `key`, channels (0, 1) (the kernel's own counter).

    int cere_ktest_sample_top_logprobs(const void *processed, int64_t vocab, float temperature, int64_t committed,
                                       int64_t n, void *committed_logprob, void *top_ids, void *top_logprobs,
                                       cudaStream_t stream);
        temperature 0 = log_softmax(processed), else log_softmax(processed / T) (D7); committed_logprob f32 [1];
        top_ids int32 [n], top_logprobs f32 [n] (descending, ties to the lower id); n <= 20.

    int cere_ktest_sample_count_commit(void *counts, const void *ids, int64_t n, cudaStream_t stream);
        counts uint32 [vocab]; ids int32 [n]: count_commit for each id in order.

    int cere_ktest_sample_ref_keep_mask(const void *probs, int64_t vocab, int64_t top_k, float top_p, float min_p,
                                        void *mask, cudaStream_t stream);
        `host/sampler_ref` keep_mask on a host copy of probs f32 [vocab]; mask uint32 [words].

    int cere_ktest_sample_stage_time(const void *logits, int64_t vocab, int64_t rows, void *counts,
                                     float temperature, int64_t top_k, float top_p, float min_p, int64_t seed,
                                     int64_t steps, void *step_us, void *ids, cudaStream_t stream);
        logits bf16 [rows, vocab]; counts uint32 [rows, vocab] (row r's out_counts; penalties 0, repetition 1);
        rows <= 64, steps <= 1000. Runs 3 untimed warm-up steps, then `steps` timed steps; a step runs, for each row r,
        sample_row's sampled kernels with draw key = step * rows + r (warm-up steps: key r). step_us host f32 [steps]
        (each step's CUDA-event span, microseconds); ids int32 [steps, rows] (each timed step's committed ids).
"""
from __future__ import annotations

import base64
import functools
import json
import math
import statistics
import unittest

import torch

import oracle

ROW = "K12"
FIXTURE = oracle.SRC / "tests/fixtures/native/sampler_cases.json"
SCHEMA = "cere-sampler-cases-v1"
SEEDED_DRAWS = 200_000
ALPHA = 1e-4
BAND = 1e-6
FP32_SUBNORMAL_HALF_ULP = 2.0 ** -150
LOGPROB_TOL = 1e-5
STAGE_STEPS = 50
STAGE_WARMUP = 3  # cere_ktest_sample_stage_time's untimed warm-up steps
S1 = {"name": "stage_S1", "temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0}
STAGE_BUDGET_US = {1: 475.0, 4: 950.0}  # per decode step (docstring: D12 at R6 with the 19.1 ms greedy step)
MASK32 = 0xFFFFFFFF
MASK64 = (1 << 64) - 1


@functools.cache
def _fixture() -> dict:
    if not FIXTURE.is_file():
        raise oracle.OracleError(f"{FIXTURE} is absent (C3: run gen_sampler_fixtures.py in the pinned image)")
    doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise oracle.OracleError(f"{FIXTURE} schema {doc.get('schema')!r} is not {SCHEMA}")
    return doc


@functools.cache
def _row(name: str) -> torch.Tensor:
    raw = bytearray(base64.b64decode(_fixture()["rows"][name]))
    return torch.frombuffer(raw, dtype=torch.int16).view(torch.bfloat16).to(oracle.DEVICE).contiguous()


def _mask_words(text: str) -> torch.Tensor:
    raw = bytearray(base64.b64decode(text))
    return torch.frombuffer(raw, dtype=torch.int32).to(oracle.DEVICE).contiguous()


def _kept(words: torch.Tensor, vocab: int) -> torch.Tensor:
    ids = torch.arange(vocab, device=words.device)
    return ((words.long()[ids // 32] >> (ids % 32)) & 1).bool()


def _bitmask_to_bool_mask(words: torch.Tensor, vocab: int) -> torch.Tensor:
    """xgrammar 0.2.1 `testing.bitmask_to_bool_mask` for one row (vendored: `bool_mask[:, i] = (bit_mask[:, i // 32]
    & (1 << (i % 32))) != 0` for i < vocab_size), an independent formulation of `_kept`."""
    ids = torch.arange(vocab, device=words.device)
    return (words.long()[ids // 32] & (torch.ones_like(ids) << (ids % 32))) != 0


def _vocab_allowed(case: dict, vocab: int) -> torch.Tensor | None:
    """M2.8: the case's vocab mask as bools (None for M2.7's unmasked cases)."""
    return _bitmask_to_bool_mask(_mask_words(case["vocab_mask"]), vocab) if "vocab_mask" in case else None


def _words(allowed: list[int], vocab: int, *, pad_ones: bool) -> torch.Tensor:
    """ceil(vocab / 32) int32 words with bit i % 32 of word i // 32 set for each allowed id (the xgrammar layout);
    `pad_ones` also sets the bits past `vocab` in a partial last word (every consumer ignores them)."""
    words = [0] * ((vocab + 31) // 32)
    for i in allowed:
        words[i // 32] |= 1 << (i % 32)
    if pad_ones and vocab % 32:
        words[-1] |= MASK32 ^ ((1 << (vocab % 32)) - 1)
    return torch.tensor([w - (1 << 32) if w >= 1 << 31 else w for w in words], dtype=torch.int32,
                        device=oracle.DEVICE)


def _torch_masked(case: dict, logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """torch's masked logits in fp32, SGLang's order: penalties (x - (f c + p), then x < 0 ? x r : x / r on counted
    ids), the vocab mask (cleared ids -inf, `apply_vocab_mask`), then logit_bias (`sampling_batch_info.py`)."""
    x = logits.float().clone()
    for i, n in case["counts"]:
        acc = torch.tensor(case["frequency_penalty"], dtype=torch.float32) * float(n)
        y = x[i] - (acc.to(x.device) + torch.tensor(case["presence_penalty"], dtype=torch.float32, device=x.device))
        r = torch.tensor(case["repetition_penalty"], dtype=torch.float32, device=x.device)
        x[i] = y * r if float(y) < 0.0 else y / r
    x = x.masked_fill(~allowed, -math.inf)
    for i, b in case["logit_bias"]:
        x[i] = x[i] + torch.tensor(b, dtype=torch.float32, device=x.device)
    return x


def _oracle_processed(case: dict, vocab: int) -> torch.Tensor:
    out = _row(case["row"]).double().clone()
    for i, value in case["processed"]:
        out[i] = value
    allowed = _vocab_allowed(case, vocab)
    if allowed is not None:
        out[~allowed] = -math.inf  # the fixture omits the ids SGLang's grammar mask cleared (C9)
    return out


def _oracle_scaled(case: dict, vocab: int) -> tuple[torch.Tensor, float]:
    """q = fp64 of SGLang's fp32 quotient processed / T (`logits.div_(temperatures)`: a tensor divisor, so a
    correctly rounded fp32 division like the kernel's `__fdiv_rn`) and lse(q) in fp64."""
    x = _oracle_processed(case, vocab).float()
    q = (x / torch.full_like(x, float(case["temperature"]))).double()
    return q, float(torch.logsumexp(q, 0))


def _counts(case: dict, vocab: int) -> torch.Tensor:
    counts = torch.zeros(vocab, dtype=torch.int32, device=oracle.DEVICE)
    for i, n in case["counts"]:
        counts[i] = n
    return counts


# -- native entry points ----------------------------------------------------------------------------------------


def _bias(case: dict):
    n = len(case["logit_bias"])
    bias_ids = torch.tensor([i for i, _ in case["logit_bias"]], dtype=torch.int32, device=oracle.DEVICE) if n else None
    bias_values = torch.tensor([b for _, b in case["logit_bias"]], dtype=torch.float32,
                               device=oracle.DEVICE) if n else None
    return bias_ids, bias_values, n


def _native_process(case: dict) -> torch.Tensor:
    """M2.7's cases: mask_dev null (cere_ktest_sample_process); M2.8's masked cases: their vocab_mask."""
    if "vocab_mask" in case:
        return _native_process_masked(case, _mask_words(case["vocab_mask"]))[0]
    logits = _row(case["row"])
    vocab = logits.numel()
    counts = _counts(case, vocab) if case["counts"] else None
    bias_ids, bias_values, n = _bias(case)
    out = oracle.empty((vocab,), torch.float32)
    oracle.call("cere_ktest_sample_process", logits, vocab, counts, float(case["frequency_penalty"]),
                float(case["presence_penalty"]), float(case["repetition_penalty"]), bias_ids, bias_values, n, out)
    return out


def _native_process_masked(case: dict, mask: torch.Tensor, logits: torch.Tensor | None = None,
                           ref: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    """cere_ktest_process_masked on the case's inputs (or `logits`) with `mask`; with ref, sampler_ref's output too."""
    logits = _row(case["row"]) if logits is None else logits
    vocab = logits.numel()
    counts = _counts(case, vocab) if case["counts"] else None
    bias_ids, bias_values, n = _bias(case)
    out = oracle.empty((vocab,), torch.float32)
    ref_out = oracle.empty((vocab,), torch.float32) if ref else None
    oracle.call("cere_ktest_process_masked", logits, vocab, counts, float(case["frequency_penalty"]),
                float(case["presence_penalty"]), float(case["repetition_penalty"]), mask, bias_ids, bias_values, n,
                out, ref_out)
    return out, ref_out


def _native_top2(values: torch.Tensor):
    ids = oracle.empty((2,), torch.int32)
    scores = oracle.empty((2,), torch.float32)
    oracle.call("cere_ktest_sample_top2_f32", values, values.numel(), ids, scores)
    return ids, scores


def _params(case: dict) -> tuple:
    return (float(case["temperature"]), int(case["top_k"]), float(case["top_p"]), float(case["min_p"]))


def _native_probs(processed: torch.Tensor, case: dict):
    vocab = processed.numel()
    probs = oracle.empty((vocab,), torch.float32)
    mask = oracle.empty(((vocab + 31) // 32,), torch.int32)
    lse = oracle.empty((1,), torch.float64)
    oracle.call("cere_ktest_sample_probs", processed, vocab, *_params(case), probs, mask, lse)
    return probs, mask, lse


def _native_draw(processed: torch.Tensor, case: dict, us: torch.Tensor) -> torch.Tensor:
    ids = oracle.empty((us.numel(),), torch.int32)
    oracle.call("cere_ktest_sample_draw", processed, processed.numel(), *_params(case), us, us.numel(), ids)
    return ids


def _native_seeded(processed: torch.Tensor, case: dict, n: int, key: int) -> torch.Tensor:
    ids = oracle.empty((n,), torch.int32)
    oracle.call("cere_ktest_sample_seeded", processed, processed.numel(), *_params(case), 0, n, key, ids)
    return ids


def _native_logprobs(processed: torch.Tensor, temperature: float, committed: int, n: int):
    committed_lp = oracle.empty((1,), torch.float32)
    top_ids = oracle.empty((max(n, 1),), torch.int32)
    top_lps = oracle.empty((max(n, 1),), torch.float32)
    oracle.call("cere_ktest_sample_top_logprobs", processed, processed.numel(), temperature, committed, n,
                committed_lp, top_ids, top_lps)
    return committed_lp, top_ids, top_lps


def _native_count_commit(counts: torch.Tensor, ids: torch.Tensor) -> None:
    oracle.call("cere_ktest_sample_count_commit", counts, ids, ids.numel())


def _native_stage_time(row: str, rows: int, seed: int) -> tuple[list[float], torch.Tensor, torch.Tensor]:
    one = _row(row)
    vocab = one.numel()
    logits = one.repeat(rows).contiguous()
    counts = torch.zeros(rows * vocab, dtype=torch.int32, device=oracle.DEVICE)
    step_us = torch.zeros(STAGE_STEPS, dtype=torch.float32)
    ids = oracle.empty((STAGE_STEPS * rows,), torch.int32)
    oracle.call("cere_ktest_sample_stage_time", logits, vocab, rows, counts, *_params(S1), seed, STAGE_STEPS, step_us,
                ids)
    return step_us.tolist(), ids, counts.view(rows, vocab)


def _ref_keep_mask(probs: torch.Tensor, case: dict) -> torch.Tensor:
    vocab = probs.numel()
    mask = oracle.empty(((vocab + 31) // 32,), torch.int32)
    _, top_k, top_p, min_p = _params(case)
    oracle.call("cere_ktest_sample_ref_keep_mask", probs, vocab, top_k, top_p, min_p, mask)
    return mask


# -- D4 counter (port of murmur_hash.py:20-48,92-101, as scripts/native/gen_sampler_fixtures.py) ----------------


def _mul32(a: torch.Tensor, c: int) -> torch.Tensor:
    """(a * c) mod 2^32 for a in [0, 2^32) without int64 overflow (16-bit halves of c)."""
    return (a * (c & 0xFFFF) + (((a * (c >> 16)) & 0xFFFF) << 16)) & MASK32


def _mix(h: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    k = _mul32(k, 0xCC9E2D51)
    k = ((k << 15) | (k >> 17)) & MASK32
    k = _mul32(k, 0x1B873593)
    h = h ^ k
    h = ((h << 13) | (h >> 19)) & MASK32
    return (h * 5 + 0xE6546B64) & MASK32


def _murmur(seeds: torch.Tensor, key: int, c: int) -> torch.Tensor:
    """murmur_hash32(seed, key, c) for int64 seeds in [0, 2^31) (the low word is the seed, the high word 0)."""
    h = _mix(torch.zeros_like(seeds), seeds & MASK32)
    h = _mix(h, torch.zeros_like(seeds))
    h = _mix(h, torch.full_like(seeds, key & MASK32))
    h = _mix(h, torch.full_like(seeds, c))
    h = h ^ 16
    h = h ^ (h >> 16)
    h = _mul32(h, 0x85EBCA6B)
    h = h ^ (h >> 13)
    h = _mul32(h, 0xC2B2AE35)
    return h ^ (h >> 16)


def _seeded_us(n: int, key: int) -> torch.Tensor:
    seeds = torch.arange(n, dtype=torch.int64, device=oracle.DEVICE)
    hi, lo = _murmur(seeds, key, 0), _murmur(seeds, key, 1)
    # x = hi * 2^32 + lo; u = (x >> 11) * 2^-53 = (hi * 2^21 + (lo >> 11)) * 2^-53, exact in fp64.
    return (hi * (1 << 21) + (lo >> 11)).double() * 2.0 ** -53


def _splitmix_us(draw_seed: int, count: int) -> torch.Tensor:
    xs = [0, MASK64]
    state = draw_seed
    while len(xs) < count:
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        xs.append(z ^ (z >> 31))
    return torch.tensor([(x >> 11) * 2.0 ** -53 for x in xs], dtype=torch.float64, device=oracle.DEVICE)


# -- distribution helpers ---------------------------------------------------------------------------------------


def _cdf(probs: torch.Tensor, kept: torch.Tensor):
    ids = torch.nonzero(kept).reshape(-1)
    sums = torch.cumsum(probs[ids].double(), 0)
    return ids, sums


def _bad_draws(got: torch.Tensor, us: torch.Tensor, probs: torch.Tensor, kept: torch.Tensor,
               expected: torch.Tensor | None = None) -> int:
    """Draws that differ from the inverse CDF over `kept` (or from `expected`) other than within BAND of a boundary
    between the two positions."""
    ids, sums = _cdf(probs, kept)
    target = us * sums[-1]
    pos = torch.clamp(torch.searchsorted(sums, target, right=True), max=ids.numel() - 1)
    want = ids[pos] if expected is None else expected.long()
    differ = torch.nonzero(got.long() != want).reshape(-1).tolist()
    bad = 0
    for j in differ:
        a = int(torch.searchsorted(ids, got[j].long()))
        b = int(torch.searchsorted(ids, want[j]))
        inside = a < ids.numel() and int(ids[a]) == int(got[j]) and b < ids.numel() and int(ids[b]) == int(want[j])
        lo, hi = min(a, b), max(a, b)
        near = inside and bool(((sums[lo:hi] - target[j]).abs() <= BAND).all())
        bad += not near
    return bad


def _chi2_sf(x: float, df: int) -> float:
    return float(torch.special.gammaincc(torch.tensor(df / 2.0, dtype=torch.float64),
                                         torch.tensor(x / 2.0, dtype=torch.float64)))


def _t1_bins(mass: list[float], n: int) -> tuple[list[int], list[float]]:
    big = min(len(mass), 20)
    raw, raw_mass, before = [], [0.0] * big, 0.0
    for m in mass:
        b = min(big - 1, int(math.floor((before + m / 2.0) * big)))
        raw.append(b)
        raw_mass[b] += m
        before += m
    merged, expected, acc = [], [], 0.0
    for b in range(big):
        acc += raw_mass[b] * n
        merged.append(len(expected))
        if acc >= 5.0:
            expected.append(acc)
            acc = 0.0
    if acc > 0.0:
        if not expected:
            expected.append(0.0)
        expected[-1] += acc
        merged = [min(m, len(expected) - 1) for m in merged]
    return [merged[b] for b in raw], expected


def _t2_bins(mass: list[float], n: int) -> tuple[list[int], list[float]]:
    order = sorted(range(len(mass)), key=lambda j: (-mass[j], j))
    of, expected, pooled = [0] * len(mass), [], []
    for r, j in enumerate(order):
        if r < min(len(mass), 19) and mass[j] * n >= 5.0:
            of[j] = len(expected)
            expected.append(mass[j] * n)
        else:
            pooled.append(j)
    pooled_mass = sum(mass[j] for j in pooled) * n
    if pooled_mass > 0.0:
        for j in pooled:
            of[j] = len(expected)
        expected.append(pooled_mass)
    return of, expected


def _gof(of: list[int], expected: list[float], positions: torch.Tensor) -> tuple[int, float]:
    if len(expected) < 2:
        return len(expected), 1.0
    bins = torch.tensor(of, dtype=torch.int64, device=positions.device)[positions]
    observed = torch.bincount(bins, minlength=len(expected)).double().tolist()
    x = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    return len(expected), _chi2_sf(x, len(expected) - 1)


class K12Sampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _fixture()["cases"]

    def _sampled(self):
        return [(i, c) for i, c in enumerate(self.cases) if c["temperature"] > 0.0]

    def test_process_d2_tolerance(self):
        for case in self.cases:
            with self.subTest(case=case["name"]):
                native = _native_process(case)
                vocab = native.numel()
                x = _row(case["row"]).double()
                want = _oracle_processed(case, vocab)
                counts = _counts(case, vocab).double()
                tol = 2.0 ** -22 * torch.clamp(x.abs(), min=1.0) + counts * abs(case["frequency_penalty"]) * 2.0 ** -23
                got = native.double()
                inf = torch.isinf(want) | torch.isinf(got)
                finite_bad = ((got - want).abs() > tol) & ~inf
                bad = int((finite_bad | (inf & (got != want))).sum())
                oracle.check_count(self, ROW, f"process_{case['name']}", bad, expected=0)
                oracle.check_bit_exact(self, ROW, f"process_{case['name']}_repeat", _native_process(case), native)

    def test_top2_f32_greedy_and_ties(self):
        for case in self.cases:
            if case["temperature"] != 0.0:
                continue
            with self.subTest(case=case["name"]):
                ids, _ = _native_top2(_native_process(case))
                want = torch.tensor([case["greedy"]["id"], case["greedy"]["runner_up"]], dtype=torch.int32,
                                    device=oracle.DEVICE)
                oracle.check_bit_exact(self, ROW, f"top2_f32_{case['name']}", ids, want)
        values = torch.zeros(4096, dtype=torch.float32, device=oracle.DEVICE)
        values[1000] = values[3000] = values[3500] = 2.0
        ids, scores = _native_top2(values)
        oracle.check_bit_exact(self, ROW, "top2_f32_ties_first_index",
                               ids, torch.tensor([1000, 3000], dtype=torch.int32, device=oracle.DEVICE))
        oracle.check_bit_exact(self, ROW, "top2_f32_ties_scores",
                               scores, torch.tensor([2.0, 2.0], dtype=torch.float32, device=oracle.DEVICE))

    def _nonfinite_row(self, **entries: float) -> torch.Tensor:
        """A seeded N(0, 1) fp32 row of 4096 with `i<id>` entries overwritten (NaN, +-inf)."""
        row = torch.randn(4096, generator=oracle.generator(20260515), device=oracle.DEVICE)
        for name, value in entries.items():
            row[int(name[1:])] = value
        return row

    def test_non_finite_logits_hardening(self):
        """SS-1/SS-2/SS-4 (hardening): NaN and +inf logits never poison the row. NaN carries no probability mass, +inf
        is the largest finite float, a row without mass puts p = 1 on its lowest NaN id (id 0 without one); NaN
        ranks above +inf in top2 and in the top-n order (K1's greedy order). Finite rows are untouched."""
        nan, inf = float("nan"), float("inf")
        case = {"temperature": 0.7, "top_k": 50, "top_p": 0.95, "min_p": 0.0}
        vocab = 4096
        us = torch.linspace(0.0, 0.999, 64, dtype=torch.float64, device=oracle.DEVICE)
        # NaN among numbers: the NaN id gets 0, the others the probabilities of the same row with that id at -inf.
        probs, mask, lse = _native_probs(self._nonfinite_row(i10=nan), case)
        want, _, want_lse = _native_probs(self._nonfinite_row(i10=-inf), case)
        oracle.check_bit_exact(self, ROW, "probs_nan_is_minus_inf", probs, want)
        oracle.check_bit_exact(self, ROW, "lse_nan_is_minus_inf", lse, want_lse)
        self.assertEqual(probs[10].item(), 0.0)
        ids = _native_draw(self._nonfinite_row(i10=nan), case, us)
        self.assertNotIn(10, ids.tolist())
        self.assertTrue(bool(_kept(mask, vocab)[ids.long()].all()))
        # +inf: it takes all the mass (ties split it).
        probs, mask, _ = _native_probs(self._nonfinite_row(i77=inf), case)
        self.assertEqual(probs[77].item(), 1.0)
        self.assertEqual(int((probs != 0).sum()), 1)
        self.assertEqual(set(_native_draw(self._nonfinite_row(i77=inf), case, us).tolist()), {77})
        both = self._nonfinite_row(i77=inf, i90=inf)
        probs, mask, _ = _native_probs(both, case)
        self.assertEqual(probs[77].item(), probs[90].item())
        oracle.check_count(self, ROW, "both_inf_split_half", int(abs(probs[77].item() - 0.5) > 1e-6), expected=0)
        self.assertEqual(set(_native_draw(both, case, us).tolist()), {77, 90})
        # No mass: all NaN, NaN among -inf, all -inf.
        nan_among = torch.full((vocab,), -inf, device=oracle.DEVICE)
        nan_among[9] = nan_among[5] = nan
        for name, row, only in (("all_nan", torch.full((vocab,), nan, device=oracle.DEVICE), 0),
                                ("nan_among_minus_inf", nan_among, 5),
                                ("all_minus_inf", torch.full((vocab,), -inf, device=oracle.DEVICE), 0)):
            with self.subTest(case=name):
                probs, mask, lse = _native_probs(row, case)
                self.assertEqual(probs[only].item(), 1.0)
                self.assertEqual(int((probs != 0).sum()), 1)
                self.assertEqual(lse.item(), -inf)
                self.assertEqual(_kept(mask, vocab).nonzero().flatten().tolist(), [only])
                self.assertEqual(set(_native_draw(row, case, us).tolist()), {only})
        # top2: NaN above +inf, NaNs by the lower id (K1's order).
        ids, _ = _native_top2(self._nonfinite_row(i3=inf, i9=nan, i5=nan))
        self.assertEqual(ids.tolist(), [5, 9])
        ids, _ = _native_top2(self._nonfinite_row(i3=inf, i8=nan))
        self.assertEqual(ids.tolist(), [8, 3])
        # top-n order: the NaN entry first, then the numbers descending (no kNone ids), the NaN's logprob NaN.
        row = self._nonfinite_row(i5=nan)
        _, top_ids, top_lps = _native_logprobs(row, 0.7, 5, 4)
        order = torch.argsort(row.nan_to_num(nan=-inf), descending=True, stable=True)[:3]
        self.assertEqual(top_ids.tolist(), [5, *order.tolist()])
        self.assertTrue(bool(torch.isnan(top_lps[0])))
        self.assertTrue(bool(torch.isfinite(top_lps[1:]).all()))

    def test_probabilities_and_mask(self):
        for _, case in self._sampled():
            with self.subTest(case=case["name"]):
                processed = _native_process(case)
                probs, mask, _ = _native_probs(processed, case)
                q, lse = _oracle_scaled(case, probs.numel())
                want = torch.exp(q - lse)
                bad = int(((probs.double() - want).abs() > BAND * want + FP32_SUBNORMAL_HALF_ULP).sum())
                oracle.check_count(self, ROW, f"probs_rel_1e-6_{case['name']}", bad, expected=0)
                oracle.check_bit_exact(self, ROW, f"mask_vs_sampler_ref_{case['name']}", mask,
                                       _ref_keep_mask(probs, case))
                differ = int((_kept(mask, probs.numel()) != _kept(_mask_words(case["mask"]), probs.numel())).sum())
                oracle.report_row(ROW, f"mask_vs_sglang_{case['name']}", phase="reported", differing=differ)

    def test_draw_for_fixture_us(self):
        for _, case in self._sampled():
            with self.subTest(case=case["name"]):
                processed = _native_process(case)
                probs, mask, _ = _native_probs(processed, case)
                us = _splitmix_us(int(case["draw_seed"]), len(case["draw_ids"]))
                got = _native_draw(processed, case, us)
                kept = _kept(mask, probs.numel())
                expected = torch.tensor(case["draw_ids"], dtype=torch.int64, device=oracle.DEVICE)
                if bool((kept == _kept(_mask_words(case["mask"]), probs.numel())).all()):
                    bad = _bad_draws(got, us, probs, kept, expected)
                else:
                    bad = _bad_draws(got, us, probs, kept)
                    oracle.report_row(ROW, f"draw_vs_sglang_{case['name']}", phase="reported",
                                      differing=_bad_draws(got, us, probs, kept, expected))
                oracle.check_count(self, ROW, f"draw_exact_{case['name']}", bad, expected=0)

    def test_seeded_draws_counter_and_distribution(self):
        failing = 0
        for index, case in self._sampled():
            if not case["distributional"]:
                continue
            with self.subTest(case=case["name"]):
                processed = _native_process(case)
                probs, mask, _ = _native_probs(processed, case)
                vocab = probs.numel()
                kept = _kept(mask, vocab)
                got = _native_seeded(processed, case, SEEDED_DRAWS, index)
                outside = int((~kept[got.long()]).sum())
                oracle.check_count(self, ROW, f"seeded_outside_mask_{case['name']}", outside, expected=0)
                allowed = _vocab_allowed(case, vocab)
                if allowed is not None:  # C9: no seeded draw outside the vocab mask (no boundary-band excuse)
                    escaped = int((~allowed[got.long()]).sum())
                    oracle.check_count(self, ROW, f"seeded_outside_vocab_mask_{case['name']}", escaped, expected=0)
                bad = _bad_draws(got, _seeded_us(SEEDED_DRAWS, index), probs, kept)
                oracle.check_count(self, ROW, f"seeded_counter_{case['name']}", bad, expected=0)
                okept = _kept(_mask_words(case["mask"]), vocab)
                ids = torch.nonzero(okept).reshape(-1)
                q, lse = _oracle_scaled(case, vocab)
                mass = torch.exp(q[ids] - lse)
                mass = (mass / mass.sum()).tolist()
                position = torch.full((vocab,), -1, dtype=torch.int64, device=oracle.DEVICE)
                position[ids] = torch.arange(ids.numel(), device=oracle.DEVICE)
                positions = position[got.long()]
                positions = positions[positions >= 0]  # a draw outside the oracle's mask lies in C2's boundary band
                for test, (of, expected) in (("T1", _t1_bins(mass, positions.numel())),
                                             ("T2", _t2_bins(mass, positions.numel()))):
                    bins, p = _gof(of, expected, positions)
                    passed = bins < 2 or p >= ALPHA
                    failing += not passed
                    oracle.report_row(ROW, f"seeded_{test}_{case['name']}", phase="gof", bins=bins, p=p,
                                      testable=bins >= 2, passed=passed)
        oracle.check_count(self, ROW, "seeded_gof_failing", failing, expected=0)

    def test_top_logprobs(self):
        for case in self.cases:
            with self.subTest(case=case["name"]):
                processed = _native_process(case)
                t = float(case["temperature"])
                vocab = processed.numel()
                want_all = _oracle_processed(case, vocab) / (t if t else 1.0) - case["lse"]
                committed = int(torch.argmax(want_all))
                committed_lp, top_ids, top_lps = _native_logprobs(processed, t, committed, min(20, vocab))
                bad = int(abs(float(committed_lp[0]) - float(want_all[committed])) > LOGPROB_TOL)
                want = want_all[top_ids.long()]
                finite = torch.isfinite(want)
                bad += int(((top_lps.double() - want).abs() > LOGPROB_TOL)[finite].sum())
                bad += int((torch.isfinite(top_lps[~finite])).sum())
                outside = want_all.clone()
                outside[top_ids.long()] = -math.inf
                bad += int((want < float(outside.max()) - 2 * LOGPROB_TOL).sum())
                bad += int(top_ids.unique().numel() != top_ids.numel())
                oracle.check_count(self, ROW, f"top_logprobs_{case['name']}", bad, expected=0)
                zero_lp, none_ids, _ = _native_logprobs(processed, t, committed, 0)  # top_logprobs: 0 writes no entry
                oracle.check_bit_exact(self, ROW, f"top_logprobs_zero_{case['name']}", none_ids,
                                       torch.full((1,), -1, dtype=torch.int32, device=oracle.DEVICE))
                oracle.check_bit_exact(self, ROW, f"top_logprobs_zero_committed_{case['name']}", zero_lp, committed_lp)
                allowed = _vocab_allowed(case, vocab)
                if allowed is not None and not bool(allowed.all()):  # C9: logprobs are post-mask
                    cleared = int(torch.nonzero(~allowed)[0])
                    cleared_lp, _, _ = _native_logprobs(processed, t, cleared, 0)
                    oracle.check_count(self, ROW, f"top_logprobs_cleared_committed_{case['name']}",
                                       int(not bool(torch.isneginf(cleared_lp).all())), expected=0)

    def test_count_commit(self):
        counts = torch.zeros(4096, dtype=torch.int32, device=oracle.DEVICE)
        ids = torch.tensor([5, 5, 7, 4095, 5], dtype=torch.int32, device=oracle.DEVICE)
        _native_count_commit(counts, ids)
        want = torch.zeros_like(counts)
        want[5], want[7], want[4095] = 3, 1, 1
        oracle.check_bit_exact(self, ROW, "count_commit", counts, want)

    def test_masked_process_c5(self):
        """C5: random, single-bit and partial-last-word masks; process then top2_f32 vs torch and sampler_ref."""
        z = _row("Z")
        gen = torch.Generator().manual_seed(2808)
        random_words = torch.randint(-(1 << 31), 1 << 31, ((z.numel() + 31) // 32,), generator=gen,
                                     dtype=torch.int64).to(torch.int32)
        single = [123_457]
        tail = [3, 500] + [i for i in range(992, 1000) if i != 995]
        small = [0, 3, 4, 9, 15]
        plain = {"counts": [], "frequency_penalty": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
                 "logit_bias": []}
        cases = [
            ("random", z, random_words.to(oracle.DEVICE),
             {"counts": [[7, 3], [1000, 1], [123_456, 2]], "frequency_penalty": 0.5, "presence_penalty": 0.3,
              "repetition_penalty": 1.1, "logit_bias": [[42, 5.0], [99_999, -3.0]]}),
            ("single_bit", z, _words(single, z.numel(), pad_ones=False), plain),
            ("partial_last_word", z[:1000].contiguous(), _words(tail, 1000, pad_ones=True),
             {"counts": [[993, 2], [3, 1]], "frequency_penalty": 0.4, "presence_penalty": 0.2,
              "repetition_penalty": 1.3, "logit_bias": [[997, 0.5]]}),
            ("small_vocab_half_word", _row("s_ties"), _words(small, _row("s_ties").numel(), pad_ones=True), plain),
        ]
        for name, logits, mask, inputs in cases:
            with self.subTest(mask=name):
                case = dict(inputs, row=None)
                vocab = logits.numel()
                native, ref = _native_process_masked(case, mask, logits=logits, ref=True)
                allowed = _bitmask_to_bool_mask(mask, vocab)
                oracle.check_bit_exact(self, ROW, f"masked_{name}_bit_order_vs_kept", allowed, _kept(mask, vocab))
                cleared = ~allowed | torch.isneginf(logits.float())
                oracle.check_count(self, ROW, f"masked_{name}_neg_inf_vs_xgrammar_bits",
                                   int((torch.isneginf(native) != cleared).sum()), expected=0)
                x = logits.double()
                counts = _counts(case, vocab).double()
                tol = 2.0 ** -22 * torch.clamp(x.abs(), min=1.0) + counts * abs(case["frequency_penalty"]) * 2.0 ** -23
                got, want = native.double(), ref.double()
                inf = torch.isinf(want) | torch.isinf(got)
                bad = int(((((got - want).abs() > tol) & ~inf) | (inf & (got != want))).sum())
                oracle.check_count(self, ROW, f"masked_{name}_vs_sampler_ref", bad, expected=0)
                reference = _torch_masked(case, logits, allowed)
                ids, _ = _native_top2(native)
                first = int(torch.argmax(reference))
                oracle.check_count(self, ROW, f"masked_{name}_top2_vs_torch_argmax", int(int(ids[0]) != first),
                                   expected=0)
                oracle.check_count(self, ROW, f"masked_{name}_top2_vs_sampler_ref",
                                   int(int(ids[0]) != int(torch.argmax(ref))), expected=0)
                oracle.check_count(self, ROW, f"masked_{name}_top2_allowed", int(not bool(allowed[int(ids[0])])),
                                   expected=0)
                if int(allowed.sum()) >= 2:
                    without = reference.clone()
                    without[first] = -math.inf
                    oracle.check_count(self, ROW, f"masked_{name}_runner_up_vs_torch",
                                       int(int(ids[1]) != int(torch.argmax(without))), expected=0)

    def test_all_ones_mask_equals_null(self):
        """C5: on every M2.7 case an all-ones mask (padding bits set) is bit-identical to mask_dev null."""
        for case in self.cases:
            if "vocab_mask" in case:
                continue
            with self.subTest(case=case["name"]):
                vocab = _row(case["row"]).numel()
                ones = torch.full(((vocab + 31) // 32,), -1, dtype=torch.int32, device=oracle.DEVICE)
                masked, _ = _native_process_masked(case, ones)
                oracle.check_bit_exact(self, ROW, f"all_ones_mask_{case['name']}", masked, _native_process(case))

    def test_stage_time_d12_budget(self):
        for row in ("Z", "T"):
            logits = _row(row)
            vocab = logits.numel()
            processed = oracle.empty((vocab,), torch.float32)
            oracle.call("cere_ktest_sample_process", logits, vocab, None, 0.0, 0.0, 1.0, None, None, 0, processed)
            _, mask, _ = _native_probs(processed, S1)
            kept = _kept(mask, vocab)
            for rows in (1, 4):
                with self.subTest(row=row, rows=rows):
                    step_us, ids, counts = _native_stage_time(row, rows, 1234)
                    median = statistics.median(step_us)
                    budget = STAGE_BUDGET_US[rows]
                    case = f"stage_time_{row}_rows{rows}"
                    oracle.report_row(ROW, case, phase="perf", steps=STAGE_STEPS, median_us=median,
                                      min_us=min(step_us), max_us=max(step_us), budget_us=budget,
                                      passed=median <= budget)
                    outside = int((~kept[ids.long()]).sum())
                    oracle.check_count(self, ROW, f"{case}_outside_mask", outside, expected=0)
                    per_row = counts.long().sum(dim=1).tolist()
                    oracle.check_count(self, ROW, f"{case}_counts", sum(n != STAGE_WARMUP + STAGE_STEPS
                                                                       for n in per_row), expected=0)
                    self.assertTrue(median <= budget,
                                    f"{ROW} {case}: median stage step {median:.1f} us over the D12 budget {budget} us")


if __name__ == "__main__":
    unittest.main()
