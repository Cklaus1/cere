"""M2.12 D11 (tasks/m212-spec.md): the committed oracle fixtures `thinking_cases.json` and
`token_control_cases.json`, written by `scripts/native/gen_api_fixtures.py` in the pinned image (CPU,
`--network none`). The generator needs SGLang, so CI checks what it wrote: the provenance header agrees with the
native lock and the M2.4 fixtures, and the recorded verdicts are self-consistent (a stream's deltas concatenate to
the unary split, the suppression set is `{eos}` for this artifact, `reasoning_tokens` counts through `</think>`)."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "native"
LOCK = ROOT / "configs" / "deployments" / "qwen38-flash-next-native.lock.json"
THINK_END = 248069  # `</think>` in the pinned tokenizer
THINK_START = 248068  # `<think>` in the pinned tokenizer
EOS = 248046  # `<|im_end|>`, the tokenizer's eos_token_id


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class GenApiFixtures(unittest.TestCase):
    def setUp(self):
        self.thinking = load("thinking_cases.json")
        self.controls = load("token_control_cases.json")

    def cases(self, doc: dict, kind: str) -> list:
        return [c for c in doc["cases"] if c["kind"] == kind]

    def test_provenance_matches_the_lock_and_the_m24_fixtures(self):
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        stop = load("stop_cases.json")
        for doc, schema in ((self.thinking, "cere-thinking-cases-v1"),
                            (self.controls, "cere-token-control-cases-v1")):
            self.assertEqual(doc["schema"], schema)
            self.assertEqual(doc["generator"], "scripts/native/gen_api_fixtures.py")
            self.assertEqual(doc["source_tokenizer_sha256"], lock["tokenizer_sha256"])
            self.assertEqual(doc["template_sha256"], lock["template_sha256"])
            for key in ("image_digest", "sglang_version"):
                self.assertEqual(doc[key], stop[key], key)
        self.assertEqual(self.thinking["default_chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(self.controls["eos_token_ids"], [248044, EOS])

    def test_resolve_cases_cover_every_request_per_message_set(self):
        resolve = self.cases(self.thinking, "resolve")
        sets = sorted(self.thinking["messages"])
        names = {c["name"].split("/")[0] for c in resolve}
        self.assertEqual(len(resolve), len(names) * len(sets))
        for c in resolve:
            self.assertIn(c["messages"], sets)
            expect = c["expect"]
            if "error" in expect:
                self.assertIn(expect["error"]["stage"], ("validate", "render"), c["name"])
                continue
            self.assertIsInstance(expect["force_reasoning"], bool, c["name"])
            # The pinned template ends a thinking-off prompt with an empty think block, a thinking-on one with
            # `<think>\n` only (D4).
            closed = THINK_END in expect["ids"][-3:]
            self.assertEqual(closed, not expect["rendered_kwargs"]["enable_thinking"], c["name"])

    def test_preserve_thinking_false_drops_only_the_earlier_think_block(self):
        """tasks/api-lenient-spec.md AL-2 (a)/(b), F6/F7 on the pinned template: `preserve_thinking: true` renders the
        ids of its absence; `false` deletes exactly one span, the think block of the assistant turn before the last
        user query (`<think>` ... `</think>\\n\\n`), and keeps the tool-call turn's block after it."""
        ids = {c["name"]: c["expect"]["ids"] for c in self.cases(self.thinking, "resolve") if "ids" in c["expect"]}
        history = "reasoning_history"
        absent, kept = ids[f"enable_true/{history}"], ids[f"preserve_true/{history}"]
        dropped = ids[f"preserve_false/{history}"]
        self.assertEqual(kept, absent)
        # Without an assistant turn there is nothing to drop; `multi_turn`'s plain assistant turn renders an empty
        # think block under the default (`preserve_thinking` true) and none under false, so it differs.
        for name in ("single_user", "system_user"):
            self.assertEqual(ids[f"preserve_false/{name}"], ids[f"enable_true/{name}"], name)
            self.assertEqual(ids[f"preserve_false_thinking_off/{name}"], ids[f"absent/{name}"], name)
        self.assertNotEqual(ids["preserve_false/multi_turn"], ids["enable_true/multi_turn"])
        start = next(i for i, (a, b) in enumerate(zip(absent, dropped)) if a != b)
        removed = len(absent) - len(dropped)
        self.assertGreater(removed, 0)
        self.assertEqual(absent[:start] + absent[start + removed:], dropped)
        span = absent[start:start + removed]
        self.assertEqual((span[0], span.count(THINK_START), span.count(THINK_END)), (THINK_START, 1, 1))
        self.assertIn(THINK_END, dropped[start:])

    def test_forced_reasoning_streams_concatenate_to_the_unary_split(self):
        cases = self.cases(self.thinking, "reasoning")
        self.assertGreaterEqual(len(cases), 10)
        for c in cases:
            self.assertTrue(c["force_reasoning"])
            unary = list(c["unary"])
            if "<think>" in c["text"] and not c["text"].startswith("<think>"):
                # SGLang's stream strips the first `<think>` anywhere, `detect_and_parse` only leading ones.
                unary[0] = unary[0].replace("<think>", "", 1)
            for s in c["streams"]:
                self.assertEqual(s["cuts"][0], 0)
                self.assertEqual(s["cuts"][-1], len(c["text"]))
                parts = s["deltas"] + [s["finish"]]
                got = ["".join(p[0] for p in parts), "".join(p[1] for p in parts)]
                self.assertEqual(got, unary, f"{c['name']} {s['chunking']}")

    def test_reasoning_tokens_count_through_the_think_end(self):
        for c in self.cases(self.thinking, "reasoning_tokens"):
            ids, counts = c["ids"], c["reasoning_tokens"]
            self.assertEqual(c["think_end_ids"], [THINK_END])
            self.assertEqual(len(counts), len(ids))
            end = ids.index(THINK_END) + 1 if THINK_END in ids else len(ids)
            self.assertEqual(counts, [min(i + 1, end) for i in range(len(ids))], c["name"])

    def test_token_controls(self):
        (suppression,) = self.cases(self.controls, "suppression")
        self.assertEqual(suppression["set"], [EOS])  # C8 pin for this artifact
        params = {c["name"]: c["expect"] for c in self.cases(self.controls, "params")}
        self.assertEqual(params["stop_ids_dupes"]["stop_token_ids"], [3, 7])
        self.assertIsNone(params["stop_ids_empty"]["stop_token_ids"])
        self.assertEqual(params["min_over_budget"], {"error": "ValueError"})
        self.assertEqual(params["min_negative"], {"error": "ValueError"})
        finish = {c["name"]: c["expect"] for c in self.cases(self.controls, "finish")}
        # D9: `ignore_eos` ends no output on a token, model or request id; the budget does.
        for name in ("ignore_eos_runs_through", "ignore_eos_alone"):
            self.assertEqual(finish[name]["finish_type"], "length", name)
        self.assertEqual(finish["request_id_ends"], {"finish_type": "stop", "matched": 11, "finished_len": 3})


if __name__ == "__main__":
    unittest.main()
