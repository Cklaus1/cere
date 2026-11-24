"""SelfTune ST1 (tasks/selftune-spec.md E3): scripts/native/launch_sequence.py records the ST1L rows of a graph suite
report as the baseline fixture and checks a later report against it."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "native"))

import launch_sequence as ls  # noqa: E402

LIBRARY = "eb84f6509942f811dd2ae9ff8048fb415784332667aae2749899e6af93ea91e2"


def row(case: str, **fields) -> dict:
    return {"row": "ST1L", "case": case, "utc": "2026-11-12T00:00:00Z", "launches_sha256": "a" * 64,
            "records": 13, "kernels": 900, **fields}


def report(rows: list, verdict: str = "OK") -> dict:
    return {"verdict": verdict, "failures": 0, "errors": 0, "library_sha256": LIBRARY,
            "rows": [{"row": "P1", "case": "GraphRowsK6/ctx2048/r1", "replays": 24}, *rows]}


class LaunchSequence(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.fixture = self.dir / "launch_sequence.json"

    def write(self, data: dict, name: str = "native-test-graph-20261112T000000Z.json") -> Path:
        path = self.dir / name
        path.write_text(json.dumps(data))
        return path

    def test_record_then_check_equal(self):
        rows = [row("decode/dense/r1"), row("verify/m1", records=1)]
        data = ls.record(self.write(report(rows)), self.fixture)
        self.assertEqual(data["library_sha256"], LIBRARY)
        self.assertEqual(sorted(data["cases"]), ["decode/dense/r1", "verify/m1"])
        self.assertNotIn("utc", data["cases"]["verify/m1"])
        later = report([dict(r, utc="2026-11-13T00:00:00Z") for r in rows])
        self.assertEqual(ls.check(self.write(later, "later.json"), self.fixture)[1], [])
        self.assertEqual(ls.main(["check", str(self.dir / "later.json"), "--fixture", str(self.fixture)]), 0)

    def test_check_lists_every_difference(self):
        ls.record(self.write(report([row("decode/dense/r1"), row("verify/m1")])), self.fixture)
        later = self.write(report([row("decode/dense/r1", kernels=901), row("draft/r1")]), "later.json")
        self.assertEqual(ls.check(later, self.fixture)[1], [
            "verify/m1: missing from the report", "draft/r1: not in the baseline",
            "decode/dense/r1: kernels: 900 -> 901"])
        self.assertEqual(ls.main(["check", str(later), "--fixture", str(self.fixture)]), 1)

    def test_arena_sizes_compare_exactly(self):
        arena = dict(workspace_bytes=4096, derived_bytes=512, session_bytes=256, pool_bytes=1024, staging_bytes=0)
        ls.record(self.write(report([row("verify/m5", **arena)])), self.fixture)
        later = self.write(report([row("verify/m5", **dict(arena, pool_bytes=1280))]), "later.json")
        self.assertEqual(ls.check(later, self.fixture)[1], ["verify/m5: pool_bytes: 1024 -> 1280"])

    def test_record_refuses_a_failed_or_empty_report(self):
        with self.assertRaisesRegex(ls.LaunchError, "did not pass"):
            ls.record(self.write(report([row("verify/m1")], verdict="FAIL")), self.fixture)
        with self.assertRaisesRegex(ls.LaunchError, "no ST1L rows"):
            ls.record(self.write(report([])), self.fixture)
        with self.assertRaisesRegex(ls.LaunchError, "repeated case"):
            ls.record(self.write(report([row("verify/m1"), row("verify/m1")])), self.fixture)
        self.assertFalse(self.fixture.exists())

    def test_malformed_fixture_exits_2(self):
        self.fixture.write_text(json.dumps({"schema": 0}))
        path = self.write(report([row("verify/m1")]))
        self.assertEqual(ls.main(["check", str(path), "--fixture", str(self.fixture)]), 2)


if __name__ == "__main__":
    unittest.main()
