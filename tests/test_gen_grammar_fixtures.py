"""M2.8 C4/D5: static checks of the committed grammar oracle fixtures (scripts/native/gen_grammar_fixtures.py, run in
the pinned image). The fixtures hold request fields, schemas, counts and SHA-256 digests only."""
from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/native"
LOCK = ROOT / "configs/deployments/qwen38-flash-next.lock.json"
HEX64 = re.compile(r"[0-9a-f]{64}")
HEADER = {"schema", "image_digest", "sglang_version", "generator", "source_tokenizer_sha256", "template_sha256",
          "xgrammar_version", "torch_version", "vocab_size", "mask_words", "stop_token_ids"}
CASE_KEYS = {"name", "request", "sglang_extra", "outcome", "error", "key_type", "key_sha256", "instance_sha256",
             "path_len", "path_sha256", "stop_token", "masks", "open_mask_sha256", "masks_sha256"}
THINKING_KEYS = {"name", "request", "sglang_extra", "outcome", "key_type", "key_sha256", "think_len", "think_sha256",
                 "path_len", "path_sha256", "stop_token", "steps_sha256", "masks", "all_ones_masks",
                 "open_mask_sha256", "think_end_mask_sha256", "masks_sha256"}
REQUEST_FIELDS = {"response_format", "regex", "ebnf", "tools_json", "tool_choice", "parallel_tool_calls"}
VARIANT_KEYS = {"name", "keyword", "format", "schema", "conforming_sha256", "violating_sha256", "compiled", "warned",
                "accepted", "rejected", "pass"}
CANDIDATES = set("type properties required additionalProperties items prefixItems enum const anyOf allOf $ref $defs "
                 "minItems maxItems minLength maxLength pattern format minimum maximum exclusiveMinimum "
                 "exclusiveMaximum".split())
ANNOTATIONS = {"title", "description", "default"}
BOUND_KEYWORDS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
BOUND_VARIANT_KEYS = {"name", "type", "form", "schema", "conforming_sha256", "violating_sha256", "compiled", "warned",
                      "accepted", "rejected", "pass"}
COMBINATION_KEYS = {"name", "group", "schema", "conforming_sha256", "violating_sha256", "compiled", "warned",
                    "accepted", "rejected", "pass"}
NUMBER_RANGE_KEYS = {"name", "schema", "conforming_sha256", "violating_sha256", "compiled", "accepted", "rejected",
                     "stock_pass", "pass"}


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class GrammarFixturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load("grammar_cases.json")
        cls.keywords = load("grammar_keywords.json")
        cls.lock = json.loads(LOCK.read_text())

    def check_header(self, doc: dict, schema: str):
        self.assertLessEqual(HEADER, set(doc))
        self.assertEqual(doc["schema"], schema)
        self.assertEqual(doc["generator"], "scripts/native/gen_grammar_fixtures.py")
        self.assertEqual(doc["image_digest"], "sha256:" + self.lock["backend_build_sha256"])
        self.assertEqual(doc["source_tokenizer_sha256"], self.lock["tokenizer_sha256"])
        self.assertEqual(doc["template_sha256"], self.lock["template_sha256"])
        self.assertEqual(doc["xgrammar_version"], "0.2.1")
        self.assertEqual((doc["vocab_size"], doc["mask_words"]), (248320, 7760))
        self.assertTrue(doc["stop_token_ids"] and doc["stop_token_ids"] == sorted(doc["stop_token_ids"]))

    def test_headers(self):
        self.check_header(self.cases, "cere-grammar-cases-v1")
        self.check_header(self.keywords, "cere-grammar-keywords-v1")
        self.assertEqual(self.cases["stop_token_ids"], self.keywords["stop_token_ids"])

    def test_g1_to_g9_and_forty_keyword_schemas_hold_parity(self):
        names = [c["name"] for c in self.cases["cases"]]
        self.assertEqual(names, [f"G{i}" for i in range(1, 10)] + [f"K{i:02d}" for i in range(1, 41)] + ["T1"])
        for c in self.cases["cases"][:-1]:
            self.assertLessEqual(set(c), CASE_KEYS, c["name"])
            self.assertLessEqual(set(c["request"]), REQUEST_FIELDS, c["name"])
            for raw in c["request"].values():
                json.loads(raw) if isinstance(raw, str) else self.assertIsInstance(raw, bool)
            if c["outcome"] == "refused":
                self.assertEqual(c["error"]["type"], "invalid_value", c["name"])
                continue
            self.assertEqual(c["outcome"], "ok", c["name"])
            self.assertIn(c["key_type"], ("json", "regex", "ebnf", "structural_tag"), c["name"])
            for k in ("key_sha256", "instance_sha256", "path_sha256", "open_mask_sha256", "masks_sha256"):
                self.assertRegex(c[k], HEX64, (c["name"], k))
            self.assertGreater(c["path_len"], 0)
            self.assertEqual(c["masks"], c["path_len"], c["name"])  # open + every non-final step
            self.assertIn(c["stop_token"], self.cases["stop_token_ids"], c["name"])

    def test_case_kinds_follow_the_spec_list(self):
        by = {c["name"]: c for c in self.cases["cases"]}
        self.assertEqual(json.loads(by["G1"]["request"]["response_format"]), {"type": "json_object"})
        self.assertEqual([by[g]["key_type"] for g in ("G1", "G2", "G3", "G4", "G5")],
                         ["json", "json", "json", "regex", "ebnf"])
        self.assertEqual(json.loads(by["G6"]["request"]["tool_choice"]), "required")
        self.assertEqual(len(json.loads(by["G6"]["request"]["tools_json"])), 2)
        self.assertEqual(json.loads(by["G7"]["request"]["tool_choice"])["function"]["name"], "g")
        self.assertNotIn("tool_choice", by["G9"]["request"])
        self.assertTrue(any(t["function"].get("strict") for t in json.loads(by["G9"]["request"]["tools_json"])))
        self.assertEqual(by["G8"]["sglang_extra"], {"stream": True})
        self.assertEqual(by["G8"]["request"], by["G2"]["request"])
        for k in ("key_sha256", "masks_sha256", "path_sha256"):
            self.assertEqual(by["G8"][k], by["G2"][k])
        for c in self.cases["cases"][9:]:
            rf = json.loads(c["request"]["response_format"])
            self.assertEqual((rf["type"], set(rf["json_schema"])), ("json_schema", {"name", "schema"}))

    def test_thinking_case_defers_g2_past_the_reasoning(self):
        """D15 / ledger CR20: T1 is G2's schema opened with enable_thinking; the reasoning masks are all-ones, the
        think-end step's mask is G2's open mask, and G2's path follows."""
        by = {c["name"]: c for c in self.cases["cases"]}
        t1, g2 = by["T1"], by["G2"]
        self.assertEqual(set(t1), THINKING_KEYS)
        self.assertEqual(t1["request"], {**g2["request"], "enable_thinking": True})
        self.assertEqual(t1["sglang_extra"], {"chat_template_kwargs": {"enable_thinking": True}})
        self.assertEqual((t1["outcome"], t1["key_type"], t1["key_sha256"]), ("ok", "json", g2["key_sha256"]))
        for k in ("think_sha256", "path_sha256", "steps_sha256", "open_mask_sha256", "think_end_mask_sha256",
                  "masks_sha256"):
            self.assertRegex(t1[k], HEX64, k)
        self.assertGreaterEqual(t1["think_len"], 3)  # synthetic text, a grammar token, an EOS, the think-end ids
        self.assertEqual(t1["all_ones_masks"], t1["think_len"])  # the open mask and every reasoning step's
        self.assertEqual(t1["masks"], t1["think_len"] + t1["path_len"])
        self.assertEqual((t1["path_len"], t1["path_sha256"], t1["stop_token"]),
                         (g2["path_len"], g2["path_sha256"], g2["stop_token"]))
        self.assertEqual(t1["think_end_mask_sha256"], g2["open_mask_sha256"])
        self.assertNotEqual(t1["open_mask_sha256"], g2["open_mask_sha256"])

    def test_enforcement_matrix(self):
        doc = self.keywords
        self.assertEqual(doc["dialect"], "json-schema-2020-12/xgrammar-0.2.1")
        self.assertEqual(set(doc["candidates"]), CANDIDATES)
        self.assertTrue(doc["keywords"], "the passing keyword set is empty")
        self.assertTrue(doc["formats"], "the passing format set is empty")
        self.assertEqual(set(doc["keywords"]) | set(doc["failed_keywords"]), CANDIDATES)
        self.assertFalse(set(doc["keywords"]) & set(doc["failed_keywords"]))
        self.assertNotIn("oneOf", doc["keywords"])
        self.assertLessEqual(set(doc["annotations"]), ANNOTATIONS)
        variants = doc["variants"]
        by_kw: dict = {}
        by_fmt: dict = {}
        for v in variants:
            self.assertEqual(set(v), VARIANT_KEYS, v["name"])
            self.assertTrue(v["conforming_sha256"] and v["violating_sha256"], v["name"])
            for d in v["conforming_sha256"] + v["violating_sha256"]:
                self.assertRegex(d, HEX64)
            self.assertEqual(len(v["accepted"]), len(v["conforming_sha256"]))
            self.assertEqual(len(v["rejected"]), len(v["violating_sha256"]))
            self.assertEqual(v["pass"], v["compiled"] and not v["warned"] and all(v["accepted"]) and all(v["rejected"]),
                             v["name"])
            if v["format"] is None:
                self.assertIn(json.dumps(v["keyword"]) + ":", json.dumps(v["schema"]), v["name"])
                by_kw.setdefault(v["keyword"], []).append(v["pass"])
            else:
                self.assertEqual(v["schema"], {"type": "string", "format": v["format"]})
                by_fmt.setdefault(v["format"], []).append(v["pass"])
        self.assertEqual(set(by_kw), (CANDIDATES - {"format"}) | ANNOTATIONS)
        self.assertEqual(sorted([k for k, p in by_kw.items() if all(p) and k not in ANNOTATIONS | BOUND_KEYWORDS]
                                + (["format"] if doc["formats"] else [])), doc["keywords"])
        self.assertEqual(sorted(a for a in ANNOTATIONS if all(by_kw[a])), doc["annotations"])
        self.assertEqual(sorted(f for f, p in by_fmt.items() if all(p)), doc["formats"])
        self.assertEqual(sorted(f for f, p in by_fmt.items() if not all(p)), doc["failed_formats"])

    def test_bound_forms(self):
        """Numeric bounds are value-restricted: only the matrix-proven (type, form) pairs are advertised."""
        doc = self.keywords
        self.assertFalse(BOUND_KEYWORDS & set(doc["keywords"]))
        forms: dict = {}
        for v in doc["bound_variants"]:
            self.assertEqual(set(v), BOUND_VARIANT_KEYS, v["name"])
            self.assertIn(v["type"], ("integer", "number"))
            self.assertEqual(v["form"], sorted(v["form"]))
            self.assertLessEqual(set(v["form"]), BOUND_KEYWORDS)
            self.assertEqual(set(v["schema"]), {"type"} | set(v["form"]), v["name"])
            self.assertEqual(v["schema"]["type"], v["type"])
            self.assertTrue(v["conforming_sha256"] and v["violating_sha256"], v["name"])
            for d in v["conforming_sha256"] + v["violating_sha256"]:
                self.assertRegex(d, HEX64)
            self.assertEqual(v["pass"], v["compiled"] and not v["warned"] and all(v["accepted"]) and all(v["rejected"]),
                             v["name"])
            forms.setdefault((v["type"], tuple(v["form"])), []).append(v["pass"])
        self.assertEqual(len({f for _, f in forms}), 8)
        self.assertEqual({t: sorted(list(f) for (tt, f), p in forms.items() if tt == t and all(p))
                          for t in ("integer", "number")}, doc["bound_forms"])
        self.assertTrue(doc["bound_forms"]["integer"], "no integer bound form passes")
        want = {}
        for kw in sorted(BOUND_KEYWORDS):
            per = {t: [f for f in fs if kw in f] for t, fs in doc["bound_forms"].items() if any(kw in f for f in fs)}
            if per:
                want[kw] = per
        self.assertEqual(doc["value_restricted_keywords"], want)

    def test_combination_matrix(self):
        """Z2a: group A records the combinations xgrammar drops a keyword in (refused by the walker); every group B
        form the walker accepts is enforced."""
        doc = self.keywords
        variants = doc["combination_variants"]
        names = [v["name"] for v in variants]
        self.assertEqual(len(names), len(set(names)))
        for v in variants:
            typed_enum = ":type+enum:" in v["name"]
            self.assertEqual(set(v), COMBINATION_KEYS | ({"same_grammar_without_type"} if typed_enum else set()),
                             v["name"])
            if typed_enum and v["group"] == "B":
                # O113: the compiled grammar is exactly the enum's (the `type` beside it adds nothing).
                self.assertTrue(v["same_grammar_without_type"], v["name"])
            self.assertEqual(v["group"], v["name"][0], v["name"])
            self.assertTrue(v["conforming_sha256"] and v["violating_sha256"], v["name"])
            for d in v["conforming_sha256"] + v["violating_sha256"]:
                self.assertRegex(d, HEX64)
            self.assertEqual(len(v["accepted"]), len(v["conforming_sha256"]))
            self.assertEqual(len(v["rejected"]), len(v["violating_sha256"]))
            self.assertEqual(v["pass"], v["compiled"] and not v["warned"] and all(v["accepted"]) and all(v["rejected"]),
                             v["name"])
        self.assertEqual(doc["combinations_enforced"], sorted(v["name"] for v in variants if v["pass"]))
        self.assertEqual(doc["combinations_unenforced"], sorted(v["name"] for v in variants if not v["pass"]))
        self.assertEqual([v["name"] for v in variants if v["group"] == "B" and not v["pass"]], [])
        for prefix in ("A:string:", "A:$ref", "A:object+anyOf", "A:integer:maximum+anyOf", "A:typeless:",
                       "A:required-not-in-properties", "B:string:", "B:$ref", "B:const", "B:enum", "B:anyOf",
                       "B:required-", "B:type-array:", "A:type+enum:", "B:type+enum:"):
            self.assertTrue(any(n.startswith(prefix) for n in names), prefix)

    def test_number_range_variants(self):
        """B7 [REVISED: owner 2026-10-18 number bounds]: every inclusive number range (TC-64's [0, 10], fractional,
        negative and nested forms) is enforced exactly by the grammar sidecar, which xgrammar's own compile is not."""
        variants = self.keywords["number_range_variants"]
        names = [v["name"] for v in variants]
        self.assertEqual(len(names), len(set(names)))
        for prefix in ("number:[0,10]", "number:[0.5,", "number:[-7.25,", "number:[-3,", "object:", "array:", "$defs",
                       "anyOf"):
            self.assertTrue(any(n.startswith(prefix) for n in names), prefix)
        tc64 = {"type": "number", "minimum": 0, "maximum": 10}
        self.assertEqual(variants[0]["schema"], tc64)
        for v in variants:
            self.assertEqual(set(v), NUMBER_RANGE_KEYS, v["name"])
            self.assertTrue(v["conforming_sha256"] and v["violating_sha256"], v["name"])
            for d in v["conforming_sha256"] + v["violating_sha256"]:
                self.assertRegex(d, HEX64)
            self.assertEqual(len(v["accepted"]), len(v["conforming_sha256"]))
            self.assertEqual(len(v["rejected"]), len(v["violating_sha256"]))
            self.assertEqual(v["pass"], v["compiled"] and all(v["accepted"]) and all(v["rejected"]), v["name"])
            self.assertTrue(v["pass"], v["name"])
            # Each variant has a violating instance xgrammar 0.2.1's own number rule accepts (the leak).
            self.assertFalse(v["stock_pass"], v["name"])
        # TC-64's instances (digests): 0 10 3.5 10.0 1e1 0.0 accepted; -1 10.5 11 1e2 -0.1 rejected.
        digests = {hashlib.sha256(s.encode()).hexdigest() for s in ("0", "10", "3.5", "10.0", "1e1", "0.0")}
        self.assertLessEqual(digests, set(variants[0]["conforming_sha256"]))
        digests = {hashlib.sha256(s.encode()).hexdigest() for s in ("-1", "10.5", "11", "1e2", "-0.1")}
        self.assertLessEqual(digests, set(variants[0]["violating_sha256"]))

    def test_no_text_fields(self):
        """Only schemas, request fields, counts and digests: no prompt, output, instance or token-id lists."""
        for doc in (self.cases, self.keywords):
            for section in ("cases", "variants", "bound_variants", "combination_variants", "number_range_variants"):
                for c in doc.get(section, []):
                    for k in c:
                        self.assertNotIn(k, {"text", "prompt", "content", "output", "instance", "ids", "path",
                                             "messages", "bitmask", "mask"}, c["name"])


if __name__ == "__main__":
    unittest.main()
