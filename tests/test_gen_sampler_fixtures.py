"""tasks/m27-spec.md C3: the host-runnable parts of scripts/native/gen_sampler_fixtures.py (the oracle itself runs in
the pinned image; this checks the pure-Python port, the u map, the inverse CDF, the encodings and the inputs), plus
M2.8's masked cases (tasks/m28-spec.md C5, C9)."""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import random
import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/native/gen_sampler_fixtures.py"


def load():
    spec = importlib.util.spec_from_file_location("gen_sampler_fixtures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load()


def murmur3_x86_32(data: bytes, seed: int = 0) -> int:
    """Textbook MurmurHash3_x86_32 over bytes (an independent formulation of the Triton kernel)."""
    c1, c2, mask = 0xCC9E2D51, 0x1B873593, 0xFFFFFFFF
    h = seed
    for (k,) in struct.iter_unpack("<I", data):
        k = (k * c1) & mask
        k = ((k << 15) | (k >> 17)) & mask
        k = (k * c2) & mask
        h ^= k
        h = ((h << 13) | (h >> 19)) & mask
        h = (h * 5 + 0xE6546B64) & mask
    h ^= len(data)
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & mask
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & mask
    return h ^ (h >> 16)


class PurePorts(unittest.TestCase):
    def test_murmur_equals_textbook_murmur3_over_seed_key_column(self):
        rng = random.Random(7)
        for _ in range(2000):
            seed, key, c = rng.getrandbits(64), rng.getrandbits(32), rng.randrange(8)
            data = struct.pack("<IIII", seed & 0xFFFFFFFF, seed >> 32, key, c)
            self.assertEqual(gen.murmur_hash32(seed, key, c), murmur3_x86_32(data))

    def test_splitmix64_known_answer(self):
        state, out = gen.splitmix64(0)
        self.assertEqual(out, 0xE220A8397B1DCDAF)
        self.assertEqual(gen.splitmix64(state)[1], 0x6E789E6AA1B965F4)

    def test_u_map_edges(self):
        self.assertEqual(gen.u_from_x(0), 0.0)
        self.assertEqual(gen.u_from_x((1 << 64) - 1), 1.0 - 2.0 ** -53)
        self.assertEqual(gen.u_from_x(1 << 63), 0.5)
        xs = gen.draw_xs(5)
        self.assertEqual(xs[:2], [0, (1 << 64) - 1])
        self.assertEqual(len(gen.draw_xs(123)), gen.DRAW_US)
        self.assertEqual(gen.draw_xs(123), gen.draw_xs(123))

    def test_inverse_cdf_id_order_unnormalized_and_clamped(self):
        probs = [0.5, 0.0, 0.25, 0.125, 0.125]
        kept = [0, 2, 3]  # S = 0.875
        self.assertEqual(gen.inverse_cdf(probs, kept, 0.0), 0)
        self.assertEqual(gen.inverse_cdf(probs, kept, 0.5), 0)  # t = 0.4375 < 0.5
        self.assertEqual(gen.inverse_cdf(probs, kept, 0.6), 2)  # t = 0.525
        self.assertEqual(gen.inverse_cdf(probs, kept, 0.86), 3)
        self.assertEqual(gen.inverse_cdf(probs, kept, 1.0 - 2.0 ** -53), 3)
        self.assertEqual(gen.kept_cdf_pick([4, 9], [0.25, 0.5], 1.0), 9)  # u*S >= S clamps to the last kept id

    def test_bf16_round_to_nearest_even(self):
        self.assertEqual(gen.to_bf16_bits(1.0), 0x3F80)
        self.assertEqual(gen.to_bf16_bits(float("-inf")), 0xFF80)
        self.assertEqual(gen.from_bf16_bits(0xFF80), float("-inf"))
        one_plus_half_ulp = struct.unpack("<f", struct.pack("<I", 0x3F808000))[0]
        self.assertEqual(gen.to_bf16_bits(one_plus_half_ulp), 0x3F80)  # tie to even
        just_above = struct.unpack("<f", struct.pack("<I", 0x3F808001))[0]
        self.assertEqual(gen.to_bf16_bits(just_above), 0x3F81)
        self.assertEqual(gen.decode_bf16_row(gen.encode_bf16_row([1, 0xFF80, 0x3F80])), [1, 0xFF80, 0x3F80])

    def test_mask_encoding_round_trip(self):
        kept = [0, 31, 32, 63, 100, 248319]
        text = gen.encode_mask(kept, gen.VOCAB)
        self.assertEqual(len(gen.base64.b64decode(text)), gen.VOCAB // 8)
        self.assertEqual(gen.decode_mask(text, gen.VOCAB), kept)

    def test_canonical_ties_keeps_lowest_ids_of_a_partly_kept_group(self):
        probs = [0.1, 0.3, 0.3, 0.3, 0.0]
        self.assertEqual(gen.canonical_ties([1, 3], probs), ([1, 2], 1))
        self.assertEqual(gen.canonical_ties([1, 2], probs), ([1, 2], 0))
        self.assertEqual(gen.canonical_ties([0, 1, 2, 3], probs), ([0, 1, 2, 3], 0))

    def test_vocab_mask_words_xgrammar_bit_order(self):
        # xgrammar 0.2.1 `testing.bitmask_to_bool_mask`: bool_mask[:, i] = (bit_mask[:, i // 32] & (1 << (i % 32))) != 0
        allowed = [0, 5, 31, 32, 39]
        words = gen.mask_words(allowed, 40)
        self.assertEqual(words, [(1 << 0) | (1 << 5) | (1 << 31), (1 << 0) | (1 << 7)])
        self.assertEqual([i for i in range(40) if words[i // 32] & (1 << (i % 32)) != 0], allowed)
        self.assertEqual(gen.mask_allowed(words, 40), allowed)
        padded = gen.mask_words(allowed, 40, pad_ones=True)  # bits past the vocab are set and ignored
        self.assertEqual(padded[1], words[1] | 0xFFFFFF00)
        self.assertEqual(gen.mask_allowed(padded, 40), allowed)
        self.assertEqual(gen.decode_words(gen.encode_words(padded)), padded)
        self.assertEqual(gen.encode_words(gen.mask_words([0, 31, 32, 63, 100, 248319], gen.VOCAB)),
                         gen.encode_mask([0, 31, 32, 63, 100, 248319], gen.VOCAB))


class Inputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = gen.rows()
        cls.cases = gen.cases(cls.rows)

    def test_row_shapes(self):
        for name in ("Z", "F", "T", "M"):
            self.assertEqual(len(self.rows[name]), gen.VOCAB, name)
        for name in ("s_ties", "s_peak"):
            self.assertEqual(len(self.rows[name]), gen.SMALL_VOCAB, name)
        z = [gen.from_bf16_bits(b) for b in self.rows["Z"]]
        self.assertEqual(max(z), 0.0)
        self.assertAlmostEqual(min(z), -1.1 * math.log(gen.VOCAB), delta=0.1)
        self.assertNotEqual(z.index(0.0), 0)  # ids are shuffled
        m = [gen.from_bf16_bits(b) for b in self.rows["M"]]
        self.assertTrue(all(math.isinf(v) for v in m[1::2]))
        self.assertEqual(m[0::2], z[0::2])
        self.assertEqual(set(self.rows["F"]), {0})
        t = [gen.from_bf16_bits(b) for b in self.rows["T"]]
        self.assertEqual(t.count(3.0), 64)
        self.assertLess(max(v for v in t if v != 3.0), 3.0)

    def test_cases(self):
        names = [c["name"] for c in self.cases]
        self.assertEqual(len(names), len(set(names)))
        dist = [c for c in self.cases if c["distributional"] and "vocab_mask" not in c]
        self.assertEqual(len(dist), 24)
        self.assertEqual({c["row"] for c in dist}, {"Z", "F", "T", "M"})
        for c in dist:
            self.assertGreater(c["temperature"], 0.0)
        p6 = {c["row"]: c for c in dist if c["name"].endswith("-P6")}
        z = [gen.from_bf16_bits(b) for b in self.rows["Z"]]
        top1 = z.index(max(z))
        self.assertEqual(p6["Z"]["logit_bias"], [[top1, -5.0]])
        self.assertIn(top1, [i for i, _ in p6["Z"]["counts"]])
        self.assertEqual((p6["Z"]["frequency_penalty"], p6["Z"]["presence_penalty"], p6["Z"]["repetition_penalty"]),
                         (0.5, 0.3, 1.1))
        self.assertTrue(all(c["top_k"] == gen.SMALL_VOCAB for c in self.cases
                            if c["row"].startswith("s_") and c["name"].endswith("greedy")))
        self.assertEqual(len({c["draw_seed"] for c in self.cases}), len(self.cases))

    def test_masked_cases_appended_after_m27s(self):
        names = [c["name"] for c in self.cases]
        masked = [c for c in self.cases if "vocab_mask" in c]
        self.assertEqual([c["name"] for c in masked], list(gen.MASKED_CASES))
        self.assertEqual(names[-len(masked):], list(gen.MASKED_CASES))  # M2.7's indices and draw seeds unchanged
        for c in masked:
            vocab = gen.SMALL_VOCAB if c["row"].startswith("s_") else gen.VOCAB
            words = gen.decode_words(c["vocab_mask"])
            self.assertEqual(len(words), (vocab + 31) // 32, c["name"])
            allowed = gen.mask_allowed(words, vocab)
            self.assertTrue(allowed, c["name"])
            self.assertEqual(c["distributional"], not c["row"].startswith("s_"), c["name"])
        by = {c["name"]: c for c in masked}
        z = [gen.from_bf16_bits(b) for b in self.rows["Z"]]
        top1 = z.index(max(z))
        self.assertEqual(gen.mask_allowed(gen.decode_words(by["Z-G5"]["vocab_mask"]), gen.VOCAB), [z.index(min(z))])
        g4 = by["Z-G4"]
        self.assertNotIn(top1, gen.mask_allowed(gen.decode_words(g4["vocab_mask"]), gen.VOCAB))
        self.assertIn([top1, 5.0], g4["logit_bias"])  # bias on a masked id: stays -inf (mask before bias)
        self.assertIn(top1, [i for i, _ in g4["counts"]])
        greedy = gen.decode_words(by["s_ties-G-greedy"]["vocab_mask"])
        self.assertEqual(greedy[0] >> gen.SMALL_VOCAB, 0xFFFF)  # partial last word: bits past the vocab set
        self.assertEqual(by["s_ties-G-greedy"]["temperature"], 0.0)

    def test_m27_case_inputs_equal_the_committed_fixture(self):
        path = ROOT / "tests/fixtures/native/sampler_cases.json"
        committed = {c["name"]: c for c in json.loads(path.read_text(encoding="utf-8"))["cases"]}
        for c in self.cases:
            if "vocab_mask" in c:
                continue
            self.assertEqual({k: committed[c["name"]][k] for k in c}, c, c["name"])


class Static(unittest.TestCase):
    def test_heavy_imports_are_local_and_interpreter_is_forced(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        top = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        top |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
        self.assertFalse(top & {"torch", "triton", "sglang", "numpy", "fastapi"}, top)
        doc = ast.get_docstring(tree)
        self.assertIn("--network none", doc)
        self.assertIn("TRITON_INTERPRET=1", doc)
        self.assertIn("SAMPLER FIXTURES SELFCHECK PASS", doc)
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertLess(source.index('os.environ.setdefault("TRITON_INTERPRET", "1")'),
                        source.index("from sglang.kernels.ops.sampling.murmur_hash import"))


if __name__ == "__main__":
    unittest.main()
