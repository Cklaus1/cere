"""The M2.7 sampler oracle (tasks/m27-spec.md C3): SGLang's own sampler pieces on synthetic logit rows, run in the
pinned image on the CPU (no GPU, no network, Triton interpreter). Two modes.

Generate (writes tests/fixtures/native/sampler_cases.json):

    docker run --rm --network none --user UID:GID -e HOME=/tmp -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 \
      --entrypoint python3 -v REPO:/src:ro -v REPO/tests/fixtures/native:/out IMAGE \
      /src/scripts/native/gen_sampler_fixtures.py --out /out --image IMAGE

Self-check (part of gate G0; reads the committed fixture, writes nothing):

    docker run --rm --network none --user UID:GID -e HOME=/tmp -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 \
      --entrypoint python3 -v REPO:/src:ro IMAGE \
      /src/scripts/native/gen_sampler_fixtures.py --selfcheck /src/tests/fixtures/native/sampler_cases.json

IMAGE is `bench_backend.IMAGE` (name@sha256:digest). `TRITON_INTERPRET=1` must be set before `import triton` (F10);
the script also sets it before importing SGLang. The self-check compares SGLang's interpreted `murmur_hash32` with the
pure-Python port below on 10^4 (seed, key, c) triples, regenerates every oracle output in memory and compares it
with the committed file; it prints `SAMPLER FIXTURES SELFCHECK PASS` or `SAMPLER FIXTURES SELFCHECK FAIL <names>`
and exits 0/1.

Oracles (`$S` = the image's `sglang/srt`, single-request semantics, F4):

* penalties: SGLang's accumulation, per preloaded output occurrence `scatter_add_` of f (frequency), `scatter_` of
  p (presence) and r (repetition) (`penaltylib/*_penalty.py`), the additive tensor `(0 - F) - P` added once, then
  `apply_scaling_penalties` (the undecorated function of `repetition_penalty.py:9-15`), then the dense `logit_bias`
  add (`sampling_batch_info.py:266-300`), all fp32 on the fp32 cast of the BF16 row (`logits_processor.py:831`);
* probabilities: `logits.div_(T)` then `torch.softmax` fp32 (`sampler.py:193-211`); the fp64 log-sum-exp of
  processed/T is recorded (the C2 logprob oracle, D7);
* mask, min_p = 0: `Sampler._compute_sampling_mask_from_probs` (`sampler.py:302-329`) on the fp32 probabilities cast
  to fp64 and divided by their fp64 sum (D3: masses summed in fp64). The division removes the normalization error of
  torch's fp32 softmax over 248,320 ids, whose probabilities sum to 1 +- ~1e-5: unnormalized, top_p 1.0 drops a
  tail of ~6e-6 mass and top_p 0.5 moves the cut by ~4e-6, both outside C2's 1e-6 boundary band. Scaling keeps
  the order and the draws (u*S over the kept set), so only the top_p cut moves;
* mask, min_p > 0: the sequential flashinfer order (`sampler.py:274-279`) composed from the torch pieces of
  `top_k_top_p_min_p_sampling_from_probs_torch` (`sampler.py:567-612`) in fp64: top-k zero + renorm, top-p zero +
  renorm, min_p zero, with a stable descending sort;
* ties: torch's descending sort does not promise an order among equal probabilities, D3 ranks ties to the lower
  id; a tie group the oracle keeps only partly is replaced by the same number of its lowest ids
  (`tie_groups_canonicalized` per case);
* draw: the D4 inverse CDF over the canonical mask in id order (fp64 running sum of the fp32 oracle probabilities
  against u*S, clamp to the last kept id) for 4,096 u = ((x >> 11) * 2^-53), x = 0, 2^64 - 1, then splitmix64 from
  `draw_seed`;
* greedy (T = 0): `torch.argmax` (first maximum) and the runner-up (first maximum with the winner excluded); the
  recorded log-sum-exp is that of the processed logits;
* `murmur_hash32` (`kernels/ops/sampling/murmur_hash.py:104-121`) under the Triton interpreter;
* `neg_inf_logprob` (D7, Q7): a `ChoiceLogprobs` with one -inf top entry rendered as the unary response is
  (FastAPI `jsonable_encoder` + `JSONResponse`, the route returns the pydantic model) and as a stream chunk is
  (`model_dump_json`, `serving_chat.py:2460`); per path the rendered text or the exception type, and whether
  the text is valid JSON;
* vocab mask (M2.8, tasks/m28-spec.md C5/C9; cases with a `vocab_mask`): after the penalties and before the
  `logit_bias` add (`sampling_batch_info.py` apply_logits_bias), SGLang's grammar-mask kernel
  `apply_token_bitmask_inplace_triton` (the xgrammar backend's apply_vocab_mask) under the Triton interpreter; the
  ids it sets to -inf must be exactly those `xgrammar.testing.bitmask_to_bool_mask` clears (the A5 bit order), and
  they must still be -inf after the bias. Everything downstream (probabilities, mask, draws, greedy, lse) is the
  unmasked path's on the masked logits.

File `sampler_cases.json` (`schema` cere-sampler-cases-v1, `image_digest`, `sglang_version`, `torch_version`,
`generator`, `vocab` 248,320): `rows` {name: base64 of the little-endian BF16 row}: Z (Zipf-like `-1.1 ln(rank)`,
shuffled ids), F (flat 0), T (64 ids share the max 3.0, the rest N(0,1)), M (Z with every odd id -inf), and the
small rows `s_ties`, `s_peak` (16 ids); `murmur` [[seed, key, c, h]] (seed as a decimal string); `cases` [{name,
row, temperature, top_k (vocab size = all), top_p, min_p, frequency_penalty, presence_penalty, repetition_penalty,
counts [[id, n]], logit_bias [[id, bias]], distributional (bool), vocab_mask (M2.8's masked cases only, appended after
M2.7's: base64 of little-endian u32 words, ceil(vocab / 32) of them, id i allowed iff bit i % 32 of word i / 32 is set;
bits past the vocab are ignored), processed [[id, fp32 value]] for every id whose processed logit is not the input
(in a masked case, ids the vocab_mask clears are omitted: they are -inf), lse (fp64), mask (base64 of little-endian
u32 words, bit i % 32 of word i / 32 set = kept; sampled cases), tie_groups_canonicalized, greedy {id, runner_up}
(T = 0 cases), draw_seed (decimal string), draw_ids [4,096 ids]}]; `neg_inf_logprob` {unary, stream: {text | error,
valid_json}}. No text: every row and case is synthetic; ids are integers.

The rows, cases, the murmur port, splitmix64, the u map, the inverse CDF and the encodings are pure Python
(tests/test_gen_sampler_fixtures.py runs them on the host); torch and SGLang are imported only by the oracle.
"""
from __future__ import annotations

import argparse
import base64
import bisect
import json
import math
import os
import random
import struct
import sys
from pathlib import Path

SCHEMA = "cere-sampler-cases-v1"
GENERATOR = "scripts/native/gen_sampler_fixtures.py"
VOCAB = 248320
SEED = 20261010
DRAW_US = 4096
MURMUR_ROWS = 1250  # x 8 channels = 10^4 triples
MASK32 = 0xFFFFFFFF
MASK64 = (1 << 64) - 1
NEG_INF = float("-inf")
SMALL_VOCAB = 16


# ---- pure-Python ports (host-testable) ----

def _rotl32(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & MASK32


def _mix(h: int, k: int) -> int:
    k = (k * 0xCC9E2D51) & MASK32
    k = _rotl32(k, 15)
    k = (k * 0x1B873593) & MASK32
    h ^= k
    h = _rotl32(h, 13)
    return (h * 5 + 0xE6546B64) & MASK32


def murmur_hash32(seed: int, key: int, c: int) -> int:
    """`murmur_hash32_kernel` (murmur_hash.py:20-48,76-101) for one (seed, position, column)."""
    seed &= MASK64
    h = _mix(0, seed & MASK32)
    h = _mix(h, seed >> 32)
    h = _mix(h, key & MASK32)
    h = _mix(h, c & MASK32)
    h ^= 16
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & MASK32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & MASK32
    return h ^ (h >> 16)


def splitmix64(state: int) -> tuple[int, int]:
    """One splitmix64 step: (next state, output)."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def draw_xs(draw_seed: int, count: int = DRAW_US) -> list[int]:
    """The u bit patterns of the exact draw check: 0, 2^64 - 1, then splitmix64 outputs from `draw_seed`."""
    xs = [0, MASK64]
    state = draw_seed
    while len(xs) < count:
        state, out = splitmix64(state)
        xs.append(out)
    return xs


def u_from_x(x: int) -> float:
    """D4 / ledger M8: u = (x >> 11) * 2^-53 in [0, 1) (exact in fp64)."""
    return (x >> 11) * 2.0 ** -53


def inverse_cdf(probs: list[float], kept: list[int], u: float) -> int:
    """D4 draw over `kept` (ascending ids): the first id whose fp64 running sum of probabilities exceeds u*S."""
    sums = []
    acc = 0.0
    for i in kept:
        acc += probs[i]
        sums.append(acc)
    return kept_cdf_pick(kept, sums, u)


def kept_cdf_pick(kept: list[int], sums: list[float], u: float) -> int:
    t = u * sums[-1]
    j = bisect.bisect_right(sums, t)
    return kept[min(j, len(kept) - 1)]


def to_bf16_bits(value: float) -> int:
    """Round-to-nearest-even fp32 -> BF16 (the cast of `Tensor.to(torch.bfloat16)`), as the u16 bit pattern."""
    if value == NEG_INF:
        return 0xFF80
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return (bits >> 16) & 0xFFFF


def from_bf16_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def to_f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def encode_bf16_row(bits: list[int]) -> str:
    return base64.b64encode(struct.pack(f"<{len(bits)}H", *bits)).decode("ascii")


def decode_bf16_row(text: str) -> list[int]:
    raw = base64.b64decode(text)
    return list(struct.unpack(f"<{len(raw) // 2}H", raw))


def mask_words(allowed: list[int], vocab: int, pad_ones: bool = False) -> list[int]:
    """ceil(vocab / 32) u32 words, id i set in bit i % 32 of word i // 32 (the xgrammar layout); `pad_ones` also sets
    the bits past `vocab` in a partial last word (ignored by every consumer)."""
    words = [0] * ((vocab + 31) // 32)
    for i in allowed:
        words[i // 32] |= 1 << (i % 32)
    if pad_ones and vocab % 32:
        words[-1] |= MASK32 ^ ((1 << (vocab % 32)) - 1)
    return words


def mask_allowed(words: list[int], vocab: int) -> list[int]:
    """The ids < vocab whose bit is set (xgrammar `testing.bitmask_to_bool_mask` semantics)."""
    return [i for i in range(vocab) if words[i // 32] >> (i % 32) & 1]


def encode_words(words: list[int]) -> str:
    return base64.b64encode(struct.pack(f"<{len(words)}I", *words)).decode("ascii")


def decode_words(text: str) -> list[int]:
    raw = base64.b64decode(text)
    return list(struct.unpack(f"<{len(raw) // 4}I", raw))


def encode_mask(kept: list[int], vocab: int) -> str:
    return encode_words(mask_words(kept, vocab))


def decode_mask(text: str, vocab: int) -> list[int]:
    raw = base64.b64decode(text)
    words = struct.unpack(f"<{len(raw) // 4}I", raw)
    return [i for i in range(vocab) if words[i // 32] >> (i % 32) & 1]


def canonical_ties(kept: list[int], probs: list[float]) -> tuple[list[int], int]:
    """D3 ties to the lower id: a probability value the mask keeps only partly keeps its lowest ids instead."""
    keep = set(kept)
    groups: dict[float, list[int]] = {}
    for i, p in enumerate(probs):
        groups.setdefault(p, []).append(i)
    fixed = 0
    for ids in groups.values():
        inside = sum(1 for i in ids if i in keep)
        if 0 < inside < len(ids) and any(i in keep for i in ids[inside:]):
            for i in ids:
                keep.discard(i)
            keep.update(ids[:inside])
            fixed += 1
    return sorted(keep), fixed


def rows() -> dict[str, list[int]]:
    """The synthetic rows as BF16 bit patterns (C3 inputs)."""
    rng = random.Random(SEED)
    ids = list(range(VOCAB))
    rng.shuffle(ids)
    z = [0] * VOCAB
    for rank, i in enumerate(ids, start=1):
        z[i] = to_bf16_bits(to_f32(-1.1 * math.log(rank)))
    flat = [to_bf16_bits(0.0)] * VOCAB
    ties = [to_bf16_bits(to_f32(min(rng.gauss(0.0, 1.0), 2.5))) for _ in range(VOCAB)]  # the 64 ties stay the top
    for i in rng.sample(range(VOCAB), 64):
        ties[i] = to_bf16_bits(3.0)
    masked = [0xFF80 if i % 2 else b for i, b in enumerate(z)]
    s_ties = [to_bf16_bits(v) for v in (1.5, 2.0, 2.0, -1.0, 0.5, 2.0, 0.0, -3.0, 1.5, 0.25, -0.5, 1.0, 2.0, -2.0,
                                        0.75, 1.5)]
    s_peak = [to_bf16_bits(v) for v in (-4.0, -3.0, 6.0, -2.5, -4.0, -1.0, -3.5, 0.0, -2.0, NEG_INF, -4.0, -1.5,
                                        -3.0, NEG_INF, -0.5, -2.0)]
    return {"Z": z, "F": flat, "T": ties, "M": masked, "s_ties": s_ties, "s_peak": s_peak}


def _case(name, row, t, top_k, top_p, min_p, **extra) -> dict:
    vocab = SMALL_VOCAB if row.startswith("s_") else VOCAB
    case = {"name": name, "row": row, "temperature": t, "top_k": vocab if top_k is None else top_k, "top_p": top_p,
            "min_p": min_p, "frequency_penalty": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
            "counts": [], "logit_bias": [], "distributional": not row.startswith("s_")}
    case.update(extra)
    return case


MASKED_CASES = ("Z-G1", "Z-G2", "T-G3", "Z-G4", "Z-G5", "s_ties-G-greedy", "s_peak-G-greedy-bias", "s_ties-G-k3")


def masked_cases(row_bits: dict[str, list[int]]) -> list[dict]:
    """M2.8's masked cases (tasks/m28-spec.md C9; names MASKED_CASES): a `vocab_mask` (base64 of the u32 words) applied
    after the penalties and before `logit_bias`. Random half (Z-G1), sparse 2,048 ids (Z-G2), half the 64 ties plus
    10 % of the rest (T-G3, sequential min_p), penalties plus a bias on a masked id (Z-G4), a single id (Z-G5), and
    16-id rows whose one partial word has every bit past the vocab set (s_ties-G-greedy) or whose bias lands on a
    masked id (s_peak-G-greedy-bias)."""
    rng = random.Random(f"{SEED}-vocab-mask")
    z = [from_bf16_bits(b) for b in row_bits["Z"]]
    t = [from_bf16_bits(b) for b in row_bits["T"]]
    top1 = max(range(VOCAB), key=lambda i: (z[i], -i))
    half = [i for i in range(VOCAB) if rng.random() < 0.5]
    sparse = sorted(rng.sample(range(VOCAB), 2048))
    ties = [i for i in range(VOCAB) if t[i] == 3.0]
    tie_set = set(ties)
    t_allowed = sorted(set(ties[0::2]) | {i for i in range(VOCAB) if i not in tie_set and rng.random() < 0.1})
    g4 = [i for i in range(VOCAB) if i != top1 and rng.random() < 0.5]
    best = max(g4, key=lambda i: (z[i], -i))
    counts = [[i, 1 + rng.randrange(5)] for i in sorted(set(rng.sample(g4, 48)) | {top1})]
    lowest = min(range(VOCAB), key=lambda i: (z[i], i))

    def big(allowed: list[int]) -> str:
        return encode_words(mask_words(allowed, VOCAB))

    def small(allowed: list[int], pad_ones: bool = False) -> str:
        return encode_words(mask_words(allowed, SMALL_VOCAB, pad_ones))

    return [
        _case("Z-G1", "Z", 1.0, None, 1.0, 0.0, vocab_mask=big(half)),
        _case("Z-G2", "Z", 1.0, 20, 0.95, 0.0, vocab_mask=big(sparse)),
        _case("T-G3", "T", 1.0, 20, 0.95, 0.05, vocab_mask=big(t_allowed)),
        _case("Z-G4", "Z", 1.0, 20, 1.0, 0.0, frequency_penalty=0.5, presence_penalty=0.3, repetition_penalty=1.1,
              counts=counts, logit_bias=[[top1, 5.0], [best, -3.0]], vocab_mask=big(g4)),
        _case("Z-G5", "Z", 1.0, None, 1.0, 0.0, vocab_mask=big([lowest])),
        _case("s_ties-G-greedy", "s_ties", 0.0, None, 1.0, 0.0,
              vocab_mask=small([i for i in range(SMALL_VOCAB) if i not in (1, 2, 5, 12)], pad_ones=True)),
        _case("s_peak-G-greedy-bias", "s_peak", 0.0, None, 1.0, 0.0, logit_bias=[[7, 100.0], [14, -1.0]],
              vocab_mask=small([i for i in range(SMALL_VOCAB) if i not in (2, 7)])),
        _case("s_ties-G-k3", "s_ties", 1.0, 3, 1.0, 0.0, vocab_mask=small([0, 3, 4, 6, 8, 9, 15])),
    ]


def cases(row_bits: dict[str, list[int]]) -> list[dict]:
    """The 24 distributional cases {Z, F, T, M} x {P1..P6} plus the small exact cases (C3), then M2.8's masked cases
    (appended, so M2.7's case indices and draw seeds are unchanged)."""
    out = []
    for row in ("Z", "F", "T", "M"):
        logits = [from_bf16_bits(b) for b in row_bits[row]]
        top1 = max(range(VOCAB), key=lambda i: (logits[i], -i))
        rng = random.Random(f"{SEED}-{row}")
        counted = sorted(set(rng.sample(range(VOCAB), 48)) | {top1})
        counts = [[i, 1 + rng.randrange(5)] for i in counted]
        out += [
            _case(f"{row}-P1", row, 1.0, None, 1.0, 0.0),
            _case(f"{row}-P2", row, 0.3, 20, 0.95, 0.0),
            _case(f"{row}-P3", row, 1.7, None, 0.5, 0.0),
            _case(f"{row}-P4", row, 1.0, 20, 0.95, 0.05),
            _case(f"{row}-P5", row, 0.7, None, 0.8, 0.05),
            _case(f"{row}-P6", row, 1.0, 20, 1.0, 0.0, frequency_penalty=0.5, presence_penalty=0.3,
                  repetition_penalty=1.1, counts=counts, logit_bias=[[top1, -5.0]]),
        ]
    out += [
        _case("s_ties-greedy", "s_ties", 0.0, None, 1.0, 0.0),
        _case("s_ties-greedy-pen", "s_ties", 0.0, None, 1.0, 0.0, frequency_penalty=0.5, presence_penalty=0.25,
              repetition_penalty=1.5, counts=[[1, 2], [2, 1], [12, 3]], logit_bias=[[3, 4.0], [15, -1.0]]),
        _case("s_ties-k3", "s_ties", 1.0, 3, 1.0, 0.0),
        _case("s_ties-p", "s_ties", 0.8, None, 0.6, 0.0),
        _case("s_ties-seq", "s_ties", 1.2, 5, 0.9, 0.3),
        _case("s_peak-greedy", "s_peak", 0.0, None, 1.0, 0.0),
        _case("s_peak-k1", "s_peak", 1.0, 1, 1.0, 0.0),
        _case("s_peak-all", "s_peak", 2.0, None, 1.0, 0.0, logit_bias=[[2, -100.0], [7, 100.0]]),
        _case("s_peak-minp", "s_peak", 1.0, None, 1.0, 0.5),
    ]
    out += masked_cases(row_bits)
    for i, case in enumerate(out):
        case["draw_seed"] = str(splitmix64(SEED * 1000 + i)[1])
    return out


# ---- oracle (pinned image) ----

def _sglang():
    os.environ.setdefault("TRITON_INTERPRET", "1")
    if os.environ["TRITON_INTERPRET"] != "1":
        raise SystemExit("TRITON_INTERPRET must be 1 (F10)")
    import torch
    from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32 as triton_murmur
    from sglang.srt.layers.sampler import Sampler
    from sglang.srt.sampling.penaltylib import repetition_penalty

    scaling = repetition_penalty.apply_scaling_penalties
    scaling = getattr(scaling, "_torchdynamo_orig_callable", scaling)
    return torch, triton_murmur, Sampler, scaling


def oracle_murmur(torch, triton_murmur) -> list[list]:
    rng = random.Random(f"{SEED}-murmur")
    seeds = [rng.getrandbits(64) for _ in range(MURMUR_ROWS)]
    keys = [rng.getrandbits(32) for _ in range(MURMUR_ROWS)]
    signed = [s - (1 << 64) if s >= 1 << 63 else s for s in seeds]
    hashed = triton_murmur(torch.tensor(signed, dtype=torch.int64), torch.tensor(keys, dtype=torch.int64),
                           torch.arange(8, dtype=torch.int64))
    table = hashed.to(torch.int64).tolist()
    return [[str(seeds[r]), keys[r], c, table[r][c] & MASK32] for r in range(MURMUR_ROWS) for c in range(8)]


def oracle_vocab_mask(torch, y, text: str):
    """M2.8 (tasks/m28-spec.md C9): SGLang's grammar mask on `y` [1, vocab] in place, the kernel its xgrammar backend
    applies (`apply_token_bitmask_inplace_triton`, `xgrammar_backend.py` apply_vocab_mask; interpreted on the CPU),
    between the penalties and `logit_bias` (`sampling_batch_info.py` apply_logits_bias). Returns the allowed ids as a
    bool tensor from `xgrammar.testing.bitmask_to_bool_mask` (the A5 bit order) after checking that exactly the other
    ids became -inf and the allowed ones are unchanged."""
    from sglang.kernels.ops.grammar.bitmask_ops import apply_token_bitmask_inplace_triton
    from xgrammar.testing import bitmask_to_bool_mask

    vocab = y.shape[1]
    words = decode_words(text)
    bitmask = torch.tensor([[w - (1 << 32) if w >= 1 << 31 else w for w in words]], dtype=torch.int32)
    before = y.clone()
    apply_token_bitmask_inplace_triton(y, bitmask)
    allowed = bitmask_to_bool_mask(bitmask, vocab)[0]
    if not (bool(torch.isneginf(y[0][~allowed]).all()) and torch.equal(y[0][allowed], before[0][allowed])):
        raise SystemExit("vocab_mask: SGLang's applied mask differs from xgrammar bitmask_to_bool_mask")
    return allowed


def oracle_case(torch, Sampler, scaling, case: dict, bits: list[int]) -> dict:
    from types import SimpleNamespace

    vocab = len(bits)
    x = torch.tensor([from_bf16_bits(b) for b in bits], dtype=torch.float32).view(1, vocab)
    freq = torch.zeros_like(x)
    pres = torch.zeros_like(x)
    rep = torch.ones_like(x)
    f = torch.tensor([[case["frequency_penalty"]]], dtype=torch.float32)
    p = torch.tensor([[case["presence_penalty"]]], dtype=torch.float32)
    r = torch.tensor([[case["repetition_penalty"]]], dtype=torch.float32)
    for token, n in case["counts"]:
        index = torch.tensor([[token]])
        for _ in range(n):
            freq.scatter_add_(1, index, f)
            pres.scatter_(1, index, p)
            rep.scatter_(1, index, r)
    y = x.clone()
    if case["counts"]:
        acc = torch.zeros_like(x)
        acc.sub_(freq)
        acc.sub_(pres)
        y.add_(acc)
        scaling(y, rep)
    allowed = None
    if "vocab_mask" in case:
        allowed = oracle_vocab_mask(torch, y, case["vocab_mask"])
    if case["logit_bias"]:
        bias = torch.zeros_like(x)
        for token, b in case["logit_bias"]:
            bias[0, token] = b
        y.add_(bias)
    if allowed is not None and not bool(torch.isneginf(y[0][~allowed]).all()):
        raise SystemExit(f"{case['name']}: a masked id is not -inf after logit_bias")
    processed = [[i, v] for i, (v, v0) in enumerate(zip(y[0].tolist(), x[0].tolist()))
                 if v != v0 and (allowed is None or allowed[i])]
    out = {"processed": processed}
    t = case["temperature"]
    if t == 0.0:
        out["lse"] = float(torch.logsumexp(y[0].double(), 0))
        first = int(torch.argmax(y[0]))
        rest = y[0].clone()
        rest[first] = NEG_INF
        out["greedy"] = {"id": first, "runner_up": int(torch.argmax(rest))}
        return out
    out["lse"] = float(torch.logsumexp(y[0].double() / t, 0))
    probs32 = torch.softmax(y / t, dim=-1)
    probs = probs32.double()
    top_k = min(case["top_k"], vocab)
    if case["min_p"] == 0.0:
        info = SimpleNamespace(sampling_mask_max_top_k=0, top_ks=torch.tensor([top_k]),
                               top_ps=torch.tensor([case["top_p"]], dtype=torch.float64), need_min_p_sampling=False,
                               min_ps=torch.tensor([0.0], dtype=torch.float64))
        normalized = probs / probs.sum()  # fp32 softmax normalization error removed (docstring, mask min_p = 0)
        idx, _, keep, _ = Sampler._compute_sampling_mask_from_probs(None, normalized, info)
        kept = sorted(int(i) for i in idx[0][keep[0]].tolist())
    else:
        q, idx = probs.sort(dim=-1, descending=True, stable=True)
        q = q.clone()
        q[0, top_k:] = 0.0
        q /= q.sum()
        q[(torch.cumsum(q, -1) - q) > case["top_p"]] = 0.0
        q /= q.sum()
        q[q < q[0, 0] * case["min_p"]] = 0.0
        kept = sorted(int(i) for i in idx[0][q[0] > 0].tolist())
    plist = probs32[0].tolist()
    kept = [i for i in kept if plist[i] > 0.0]
    kept, fixed = canonical_ties(kept, plist)
    sums = []
    acc = 0.0
    for i in kept:
        acc += plist[i]
        sums.append(acc)
    out["mask"] = encode_mask(kept, vocab)
    out["tie_groups_canonicalized"] = fixed
    out["draw_ids"] = [kept_cdf_pick(kept, sums, u_from_x(xv)) for xv in draw_xs(int(case["draw_seed"]))]
    return out


def oracle_neg_inf() -> dict:
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionTokenLogprob, ChoiceLogprobs, TopLogprob

    lp = ChoiceLogprobs(content=[ChatCompletionTokenLogprob(
        token="a", bytes=[97], logprob=-0.5,
        top_logprobs=[TopLogprob(token="a", bytes=[97], logprob=-0.5), TopLogprob(token="b", bytes=[98],
                                                                                 logprob=NEG_INF)])])

    def rendered(render) -> dict:
        try:
            text = render()
        except Exception as exc:  # the rendering failure is the recorded outcome
            return {"error": type(exc).__name__, "valid_json": False}
        text = text.decode("utf-8") if isinstance(text, bytes) else text
        try:
            json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
            valid = True
        except ValueError:
            valid = False
        return {"text": text, "valid_json": valid}

    return {"unary": rendered(lambda: JSONResponse(content=jsonable_encoder(lp)).body),
            "stream": rendered(lp.model_dump_json)}


def build() -> dict:
    torch, triton_murmur, Sampler, scaling = _sglang()
    import sglang

    row_bits = rows()
    doc_cases = []
    for case in cases(row_bits):
        doc_cases.append({**case, **oracle_case(torch, Sampler, scaling, case, row_bits[case["row"]])})
    return {"schema": SCHEMA, "generator": GENERATOR, "sglang_version": sglang.__version__,
            "torch_version": torch.__version__, "vocab": VOCAB,
            "rows": {name: encode_bf16_row(b) for name, b in row_bits.items()},
            "murmur": oracle_murmur(torch, triton_murmur), "cases": doc_cases, "neg_inf_logprob": oracle_neg_inf()}


def selfcheck(path: Path) -> int:
    torch, triton_murmur, _, _ = _sglang()
    problems = [f"murmur {seed}/{key}/{c}" for seed, key, c, h in oracle_murmur(torch, triton_murmur)
                if murmur_hash32(int(seed), key, c) != h][:5]
    committed = json.loads(path.read_text(encoding="utf-8"))
    fresh = build()
    for name in ("schema", "vocab", "rows", "murmur", "neg_inf_logprob", "sglang_version"):
        if committed.get(name) != fresh[name]:
            problems.append(name)
    old = {c["name"]: c for c in committed.get("cases", [])}
    for case in fresh["cases"]:
        if old.get(case["name"]) != case:
            problems.append(f"case {case['name']}")
    if problems:
        print("SAMPLER FIXTURES SELFCHECK FAIL " + " ".join(problems))
        return 1
    print(f"SAMPLER FIXTURES SELFCHECK PASS murmur {len(fresh['murmur'])} cases {len(fresh['cases'])}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, help="tests/fixtures/native (generate mode)")
    p.add_argument("--image", help="the pinned image reference with digest (generate mode, recorded)")
    p.add_argument("--selfcheck", type=Path, help="the committed sampler_cases.json (self-check mode)")
    args = p.parse_args()
    if args.selfcheck:
        return selfcheck(args.selfcheck)
    if not (args.out and args.image and "@sha256:" in args.image):
        p.error("generate mode needs --out and --image NAME@sha256:DIGEST")
    doc = build()
    doc = {"schema": doc.pop("schema"), "image_digest": args.image.split("@", 1)[1], **doc}
    (args.out / "sampler_cases.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
