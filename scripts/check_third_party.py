"""Third-party and provenance check for the native executor (tasks/m22-spec.md B3, ground rule "Third-party code").

    python3 scripts/check_third_party.py [--root DIR] [--third-party FILE] [--build-json FILE]

Checks, with the Python stdlib only:
  * `executor/THIRD_PARTY.json` against `schemas/third_party.schema.json` (every entry has name, version, source,
    revision, license, files and relation `linked`|`math_reference`|`vendored`; names unique);
  * every `vendored` entry (tasks/m23-spec.md C8) has `root` (under `executor/third_party/`), `pins` and a non-empty
    `license_files` (each one of its `files`), and only vendored entries have them; every listed file exists under
    its root and matches its line of the pins file (wheel RECORD format: path,sha256=<urlsafe base64>,size), every
    pin names a listed file, and every file under `executor/third_party/` is a listed file or a pins file;
  * every source file under `executor/cuda/`: each `math_reference: <name>` line of its leading comment block names a
    `math_reference` entry of THIRD_PARTY.json;
  * `executor/PROVENANCE.json`, when present, against `schemas/provenance.schema.json`;
  * with `--build-json`: the file against `schemas/native_build.schema.json`, every static library is a file of a
    `linked` entry, and every `NEEDED` name is a listed system library or a file of a `linked` entry.
Prints one line per violation and exits 1 if there is any, else 0. Wired into `make validate`.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
SOURCE_SUFFIXES = {".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp"}
MATH_REFERENCE = re.compile(r"\bmath_reference:(.*)")
COMMENT_LINE = re.compile(r"^\s*(//|/\*|\*|\*/)")


# -- stdlib subset of JSON Schema 2020-12 (the keywords our schemas use) --------------------------------------------


def _type_ok(value, kind: str) -> bool:
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return type(value) is int
    if kind == "boolean":
        return isinstance(value, bool)
    raise ValueError(f"unsupported schema type {kind}")


SUPPORTED = {"$schema", "$id", "title", "description", "type", "additionalProperties", "required", "properties",
             "const", "enum", "pattern", "minLength", "maxLength", "minItems", "uniqueItems", "items", "minimum"}


def validate(schema: dict, value, path: str = "$") -> list[str]:
    unknown = set(schema) - SUPPORTED
    if unknown:
        raise ValueError(f"schema keywords {sorted(unknown)} are not supported by check_third_party.py")
    errors: list[str] = []
    if "type" in schema and not _type_ok(value, schema["type"]):
        return [f"{path}: expected {schema['type']}"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match {schema['pattern']}")
    if type(value) is int and "minimum" in schema and value < schema["minimum"]:
        errors.append(f"{path}: below {schema['minimum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if schema.get("uniqueItems"):
            seen = [json.dumps(v, sort_keys=True) for v in value]
            if len(set(seen)) != len(seen):
                errors.append(f"{path}: items are not unique")
        if "items" in schema:
            for i, item in enumerate(value):
                errors += validate(schema["items"], item, f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing {key}")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}: unexpected property {key}")
        for key, sub in props.items():
            if key in value:
                errors += validate(sub, value[key], f"{path}.{key}")
    return errors


def schema_errors(name: str, value) -> list[str]:
    return validate(json.loads((SCHEMAS / f"{name}.schema.json").read_text()), value)


# -- checks ---------------------------------------------------------------------------------------------------------


def load_json(path: Path, problems: list[str]):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        problems.append(f"{path}: {e}")
        return None


def check_inventory(doc) -> list[str]:
    problems = [f"THIRD_PARTY.json {e}" for e in schema_errors("third_party", doc)]
    if problems:
        return problems
    names = [e["name"] for e in doc["entries"]]
    for name in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"THIRD_PARTY.json: duplicate entry name {name}")
    return problems


def header_references(text: str) -> list[str]:
    """`math_reference:` values in the leading comment block (lines before the first non-comment, non-blank line)."""
    refs = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not (in_block or COMMENT_LINE.match(line)):
            break
        if stripped.startswith("/*"):
            in_block = True
        m = MATH_REFERENCE.search(line)
        if m:
            refs.append(m.group(1).strip().rstrip("*/").strip())
        if "*/" in stripped:
            in_block = False
    return refs


def check_sources(cuda_dir: Path, doc, root: Path) -> list[str]:
    allowed = {e["name"] for e in doc["entries"] if e["relation"] == "math_reference"}
    problems = []
    if not cuda_dir.is_dir():
        return problems
    for path in sorted(p for p in cuda_dir.rglob("*") if p.is_file() and p.suffix in SOURCE_SUFFIXES):
        rel = path.relative_to(root).as_posix()
        for ref in header_references(path.read_text(errors="replace")):
            if not ref:
                problems.append(f"{rel}: empty math_reference")
            elif ref not in allowed:
                problems.append(f"{rel}: math_reference {ref} is not a math_reference entry of THIRD_PARTY.json")
    return problems


VENDOR_DIR = "executor/third_party"
VENDORED_FIELDS = ("root", "pins", "license_files")
RECORD_LINE = re.compile(r"([^,]+),sha256=([A-Za-z0-9_-]{43}),([0-9]+)")


def record_digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def read_pins(path: Path, rel: str, problems: list[str]) -> dict[str, tuple[str, int]]:
    pins: dict[str, tuple[str, int]] = {}
    try:
        text = path.read_text()
    except OSError as e:
        problems.append(f"{rel}: {e}")
        return pins
    for n, line in enumerate(text.splitlines(), 1):
        m = RECORD_LINE.fullmatch(line)
        if not m:
            problems.append(f"{rel}:{n}: not a RECORD line (path,sha256=<urlsafe base64>,size)")
        elif m.group(1) in pins:
            problems.append(f"{rel}:{n}: duplicate pin {m.group(1)}")
        else:
            pins[m.group(1)] = (m.group(2), int(m.group(3)))
    return pins


def check_vendored(root: Path, doc) -> list[str]:
    problems = []
    owned: set[str] = set()
    pins_by_root: dict[str, dict[str, tuple[str, int]]] = {}
    pinned_by_root: dict[str, set[str]] = {}
    for e in doc["entries"]:
        present = [f for f in VENDORED_FIELDS if f in e]
        if e["relation"] != "vendored":
            problems += [f"THIRD_PARTY.json {e['name']}: {f} is for vendored entries only" for f in present]
            continue
        missing = [f for f in VENDORED_FIELDS if f not in e]
        if missing:
            problems += [f"THIRD_PARTY.json {e['name']}: vendored entry without {f}" for f in missing]
            continue
        vroot = e["root"]
        if not vroot.startswith(VENDOR_DIR + "/") or "/" in vroot[len(VENDOR_DIR) + 1:]:
            problems.append(f"THIRD_PARTY.json {e['name']}: root {vroot} is not a directory of {VENDOR_DIR}/")
            continue
        pins_rel = f"{vroot}/{e['pins']}"
        if vroot not in pins_by_root:
            pins_by_root[vroot] = read_pins(root / pins_rel, pins_rel, problems)
            pinned_by_root[vroot] = set()
            owned.add(pins_rel)
        pins = pins_by_root[vroot]
        for lic in e["license_files"]:
            if lic not in e["files"]:
                problems.append(f"THIRD_PARTY.json {e['name']}: licence file {lic} is not one of its files")
        for f in e["files"]:
            rel = f"{vroot}/{f}"
            if rel in owned:
                problems.append(f"THIRD_PARTY.json {e['name']}: {rel} is listed twice")
                continue
            owned.add(rel)
            pinned_by_root[vroot].add(f)
            path = root / rel
            if not path.is_file() or path.is_symlink():
                problems.append(f"{rel}: listed by {e['name']} but not a regular file")
                continue
            if f not in pins:
                problems.append(f"{rel}: no pin in {pins_rel}")
                continue
            data = path.read_bytes()
            if pins[f] != (record_digest(data), len(data)):
                problems.append(f"{rel}: content differs from its pin in {pins_rel}")
    for vroot, pins in pins_by_root.items():
        for f in sorted(set(pins) - pinned_by_root[vroot]):
            problems.append(f"{vroot}: pin {f} names no listed file")
    base = root / VENDOR_DIR
    if base.is_dir():
        for path in sorted(p for p in base.rglob("*") if p.is_file() or p.is_symlink()):
            rel = path.relative_to(root).as_posix()
            if rel not in owned:
                problems.append(f"{rel}: vendored file without a THIRD_PARTY.json vendored entry")
    return problems


def check_build(build, doc) -> list[str]:
    problems = [f"build.json {e}" for e in schema_errors("native_build", build)]
    if problems:
        return problems
    linked = {f for e in doc["entries"] if e["relation"] == "linked" for f in e["files"]}
    system = {s["file"] for s in doc["system_libraries"]}
    for lib in build["static_libs"]:
        if lib not in linked:
            problems.append(f"build.json: static library {lib} is not a file of a linked THIRD_PARTY.json entry")
    for out in build["outputs"]:
        for lib in out["needed"]:
            if lib not in system and lib not in linked:
                problems.append(f"build.json: {out['file']} NEEDED {lib} is not in THIRD_PARTY.json")
    return problems


def check(root: Path, third_party: Path | None = None, build_json: Path | None = None) -> list[str]:
    problems: list[str] = []
    doc = load_json(third_party or root / "executor/THIRD_PARTY.json", problems)
    if doc is None:
        return problems
    problems += check_inventory(doc)
    if problems:
        return problems
    problems += check_sources(root / "executor/cuda", doc, root)
    problems += check_vendored(root, doc)
    provenance = root / "executor/PROVENANCE.json"
    if provenance.exists():
        prov = load_json(provenance, problems)
        if prov is not None:
            problems += [f"PROVENANCE.json {e}" for e in schema_errors("provenance", prov)]
    if build_json is not None:
        build = load_json(build_json, problems)
        if build is not None:
            problems += check_build(build, doc)
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT, help="repository root to check (default: this checkout)")
    p.add_argument("--third-party", type=Path, help="THIRD_PARTY.json to use instead of ROOT/executor/")
    p.add_argument("--build-json", type=Path, help="a build/native/build.json to check")
    args = p.parse_args(argv)
    problems = check(args.root, args.third_party, args.build_json)
    for line in problems:
        print(line)
    if problems:
        return 1
    print("third-party check: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
