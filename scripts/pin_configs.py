"""The committed runtime configs' hash pins: one generated file, `tests/fixtures/config_pins.json`.

    python3 scripts/pin_configs.py --write   # regenerate the file (after lock_native.py, or any config edit)
    python3 scripts/pin_configs.py --check   # exit 1 when the file differs from the configs (wired into make validate)

The file holds, for every `configs/runtime/*.toml`:
  * `config_hash`: `cere_contracts.canonical_digest` of the parsed TOML, which the Rust loader's `config_hash` must
    equal (crates/cere-runtime/tests/config_replicas.rs, config_native.rs and config_rev2.rs read it, as do
    tests/test_cere_contracts.py and tests/test_contracts.py);
and `masked_toml_sha256`: native-g0.toml's digest with the fields `-v3` .. `-v8` may change removed
(tests/test_native_deployment.py `masked`), so any other change to it moves this pin.

The pins still catch unintended config changes: a config edit fails `--check` (and the tests) until `--write`
regenerates the file, and the regenerated file is part of the reviewed diff. Edit the file only through `--write`.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
from cere_contracts import canonical_digest  # noqa: E402

PINS = "tests/fixtures/config_pins.json"
CONFIG_DIR = "configs/runtime"
MASKED_TOMLS = ("configs/runtime/native-g0.toml",)
WRITE = "python3 scripts/pin_configs.py --write"


def compute(root: Path = ROOT) -> dict:
    """The pins of the configs under `root`, in the file's shape."""
    from test_native_deployment import canonical_sha256, masked  # the masked-field rule lives with its tests

    configs = {p.relative_to(root).as_posix(): tomllib.loads(p.read_text())
               for p in sorted((root / CONFIG_DIR).glob("*.toml"))}
    return {
        "generated_by": WRITE,
        "config_hash": {rel: canonical_digest(doc) for rel, doc in configs.items()},
        "masked_toml_sha256": {rel: canonical_sha256(masked("toml", configs[rel])) for rel in MASKED_TOMLS},
    }


def render(pins: dict) -> str:
    return json.dumps(pins, indent=2) + "\n"


def load(root: Path = ROOT) -> dict:
    """The committed pin file."""
    return json.loads((root / PINS).read_text())


def drift(root: Path = ROOT) -> list[str]:
    """One line per difference between the committed pin file and the configs; empty when they agree."""
    path = root / PINS
    want = compute(root)
    if not path.is_file():
        return [f"{PINS} is missing"]
    try:
        have = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"{PINS} is not JSON: {e}"]
    lines = []
    for section in ("config_hash", "masked_toml_sha256"):
        old, new = have.get(section) or {}, want[section]
        for rel in sorted(old.keys() | new.keys()):
            if rel not in new:
                lines.append(f"{section} {rel}: pinned, but the config is gone")
            elif rel not in old:
                lines.append(f"{section} {rel}: not pinned (now {new[rel]})")
            elif old[rel] != new[rel]:
                lines.append(f"{section} {rel}: pinned {old[rel]}, now {new[rel]}")
    if not lines and path.read_text() != render(want):
        lines.append(f"{PINS} is not in the form `--write` writes (edited by hand?)")
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help=f"regenerate {PINS}")
    mode.add_argument("--check", action="store_true", help=f"exit 1 when {PINS} differs from the configs")
    p.add_argument("--root", type=Path, default=ROOT, help="repository root (default: this checkout)")
    args = p.parse_args(argv)
    lines = drift(args.root)
    if args.write:
        if lines:
            (args.root / PINS).write_text(render(compute(args.root)))
        print(f"pin_configs: wrote {PINS} ({len(lines)} change(s))" if lines else f"pin_configs: {PINS} up to date")
        for line in lines:
            print(f"  {line}")
        return 0
    if not lines:
        print(f"pin_configs: {PINS} matches every {CONFIG_DIR} config")
        return 0
    for line in lines:
        print(f"pin_configs: DRIFT: {line}")
    print(f"pin_configs: {PINS} is stale. If every change above is intended (a relock after lock_native.py, or a "
          f"reviewed config edit), run `{WRITE}` and commit the file; otherwise revert the config change.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
