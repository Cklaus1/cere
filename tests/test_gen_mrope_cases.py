"""M2.13 C8: the pure layout helper and case list of scripts/native/gen_mrope_cases.py (no torch/sglang in CI: the
generator imports them inside `main`), and the committed fixture's shape."""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts/native/gen_mrope_cases.py"
FIXTURE = ROOT / "tests/fixtures/native/mrope_cases.json"
IMAGE, START, END = 248056, 248053, 248054

spec = importlib.util.spec_from_file_location("gen_mrope_cases", GEN)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


class LayoutTest(unittest.TestCase):
    def test_image_is_start_pads_end(self):
        ids, images = gen.build_layout([("text", 2), ("image", (1, 4, 6)), ("text", 1)], IMAGE, START, END, 2)
        self.assertEqual(ids, [gen.TEXT_ID] * 2 + [START] + [IMAGE] * 6 + [END, gen.TEXT_ID])
        self.assertEqual(images, [{"offset": 3, "grid": [1, 4, 6], "tokens": 6}])

    def test_adjacent_images_share_no_rows(self):
        ids, images = gen.build_layout([("image", (2, 2, 2)), ("image", (1, 2, 4))], IMAGE, START, END, 2)
        self.assertEqual([i["offset"] for i in images], [1, 5])
        self.assertEqual([i["tokens"] for i in images], [2, 2])
        self.assertEqual(ids, [START, IMAGE, IMAGE, END, START, IMAGE, IMAGE, END])

    def test_grid_not_a_multiple_of_the_merge_size_is_refused(self):
        with self.assertRaises(ValueError):
            gen.build_layout([("image", (1, 3, 4))], IMAGE, START, END, 2)

    def test_case_list_covers_c8(self):
        counts = [sum(kind == "image" for kind, _ in segs) for _, segs in gen.CASES]
        self.assertIn(0, counts)
        self.assertEqual(set(range(5)), set(counts))
        self.assertTrue(any(segs[0][0] == "image" for _, segs in gen.CASES))
        self.assertTrue(any(segs[-1][0] == "image" for _, segs in gen.CASES))
        self.assertTrue(any(a[0] == b[0] == "image" for _, segs in gen.CASES for a, b in zip(segs, segs[1:])))
        for _, segs in gen.CASES:
            for kind, value in segs:
                if kind == "image":
                    t, h, w = value
                    self.assertTrue(64 <= t * (h // 2) * (w // 2) <= 1024, value)  # D4 per-image token bounds


class FixtureTest(unittest.TestCase):
    def test_fixture_matches_the_case_list(self):
        doc = json.loads(FIXTURE.read_text())
        self.assertEqual(doc["schema"], gen.SCHEMA)
        self.assertEqual([c["name"] for c in doc["cases"]], [name for name, _ in gen.CASES])
        self.assertLess(FIXTURE.stat().st_size, 1 << 20)
        for case, (_, segs) in zip(doc["cases"], gen.CASES):
            ids, images = gen.build_layout(segs, doc["image_token_id"], doc["vision_start_token_id"],
                                           doc["vision_end_token_id"], doc["spatial_merge_size"])
            self.assertEqual(case["input_len"], len(ids))
            self.assertEqual(case["images"], images)
            self.assertEqual(len(case["positions"]), 3)
            self.assertTrue(all(len(row) == len(ids) for row in case["positions"]))
            self.assertEqual(case["rope_delta"], max(map(max, case["positions"])) + 1 - len(ids))
            self.assertLessEqual(case["rope_delta"], 0)


if __name__ == "__main__":
    unittest.main()
