#!/usr/bin/env python3
"""SelfTune ST1 launch-sequence equality (tasks/selftune-spec.md ST1, E3): record or check the baseline of the
launch-sequence suite (`executor/tests/gpu/test_graph_launches.py`, row ST1L, run by `native_dev.py test --suite
graph`).

    python3 scripts/native/launch_sequence.py record REPORT [--fixture PATH]
    python3 scripts/native/launch_sequence.py check REPORT [--fixture PATH]

REPORT is the suite run's `bench/reports/native-test-graph-<utc>.json`. Its ST1L rows (one per case: the recorder
text's SHA-256, per-record digests, node / kernel / programmatic-edge counts, the decode / verify / draft graph
counter deltas and the model's five arena sizes, O279 D-b8) are the launch sequence of the report's library.
`record` writes them as the baseline fixture (default FIXTURE) with the report's library_sha256 and file name; the
report must carry a passing graph suite.
`check` compares every case's fields with the fixture's: exit 0 when every case is equal, 1 listing each case
missing, extra or different (field: baseline -> report), 2 when the report or fixture is malformed. Standard library
only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/native/launch_sequence.json"
ROW = "ST1L"
SCHEMA = 1
# Row fields that are not part of the launch sequence.
UNCOMPARED = frozenset({"row", "case", "utc"})


class LaunchError(ValueError):
    pass


def report_cases(report: dict) -> dict[str, dict]:
    """The report's ST1L rows: case -> compared fields."""
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise LaunchError("the report has no rows list")
    cases: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("row") != ROW:
            continue
        case = row.get("case")
        if not isinstance(case, str) or case in cases:
            raise LaunchError(f"ST1L row with a missing or repeated case: {case!r}")
        cases[case] = {k: v for k, v in sorted(row.items()) if k not in UNCOMPARED}
    if not cases:
        raise LaunchError(f"the report has no {ROW} rows (not a graph suite run with test_graph_launches.py?)")
    return cases


def load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LaunchError(f"{path}: {e}") from None
    if not isinstance(data, dict):
        raise LaunchError(f"{path}: not a JSON object")
    return data


def record(report_path: Path, fixture: Path) -> dict:
    report = load(report_path)
    if report.get("verdict") != "OK" or report.get("failures") or report.get("errors"):
        raise LaunchError(f"{report_path}: the graph suite did not pass (verdict {report.get('verdict')!r})")
    library = report.get("library_sha256")
    if not isinstance(library, str) or len(library) != 64:
        raise LaunchError(f"{report_path}: no library_sha256")
    data = {"schema": SCHEMA, "row": ROW, "library_sha256": library, "report": report_path.name,
            "cases": report_cases(report)}
    fixture.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return data


def differences(baseline: dict[str, dict], cases: dict[str, dict]) -> list[str]:
    out = [f"{case}: missing from the report" for case in baseline if case not in cases]
    out += [f"{case}: not in the baseline" for case in cases if case not in baseline]
    for case, want in baseline.items():
        got = cases.get(case)
        if got is None:
            continue
        for field in sorted(set(want) | set(got)):
            if want.get(field) != got.get(field):
                out.append(f"{case}: {field}: {want.get(field)!r} -> {got.get(field)!r}")
    return out


def check(report_path: Path, fixture: Path) -> tuple[dict, list[str]]:
    base = load(fixture)
    if base.get("schema") != SCHEMA or base.get("row") != ROW or not isinstance(base.get("cases"), dict):
        raise LaunchError(f"{fixture}: not a schema {SCHEMA} {ROW} fixture")
    report = load(report_path)
    return report, differences(base["cases"], report_cases(report))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=("record", "check"))
    p.add_argument("report", type=Path)
    p.add_argument("--fixture", type=Path, default=FIXTURE)
    args = p.parse_args(argv)
    try:
        if args.command == "record":
            data = record(args.report, args.fixture)
            print(f"launch_sequence: recorded {len(data['cases'])} cases of library {data['library_sha256'][:8]} "
                  f"in {args.fixture}")
            return 0
        report, diff = check(args.report, args.fixture)
    except LaunchError as e:
        print(f"launch_sequence: {e}", file=sys.stderr)
        return 2
    for line in diff:
        print(f"launch_sequence: {line}")
    library = str(report.get("library_sha256", ""))[:8]
    print(f"launch_sequence: {'EQUAL' if not diff else 'DIFFERENT'} (library {library}, {len(diff)} differences)")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
