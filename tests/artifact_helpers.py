"""Shared helpers for the v2 artifact tests (M2.1 B2/B3): the positive example and the invalid-table op applier.

`examples/artifact` has exactly one producer, `cerec prepare --name fixture-model` (R3); it is the positive v2 artifact.

Artifact-case ops (tests/fixtures/artifact_invalid.json, applied in order to a copy of the example):
* `{"op":"shard","path":P,"edit":E,"reseal":bool}` rebuilds shard P from its parsed tensors (data-offset order,
  data bytes kept). E is `{"rename":[old,new]}`, `{"drop":name}` (tensor and its bytes), `{"set":{"tensor":name,
  "dtype"?:D,"shape"?:S}}` or `{"metadata":obj|null}`. The new header is compact sorted JSON with contiguous offsets.
  reseal=true updates P's files[] entry (bytes, sha256, safetensors facts) and `model_hash`, nothing else.
* `{"op":"flip_byte","path":P,"offset":int}` xors 0xff at the offset (negative counts from the end).
* `manifest_set` / `manifest_append` / `manifest_add` (`pointer`, `value` | `delta`), then canonical re-encode.
  Pointer elements: int index, str key, or `{"field": value}` selecting the array element whose field equals value.
* `{"op":"manifest_raw","transform":"reindent4"|"unsort_keys"|"duplicate_key"}` non-canonical re-encodes;
  `{"op":"manifest_raw","transform":"replace","old":S,"new":S}` replaces the first `old` in the canonical bytes.
* `{"op":"add_file","path":P,"hex":H}`, `{"op":"symlink","path":P}`, `{"op":"fifo","path":P}`.
The expected digest is `expected_manifest_sha256` when given, else the sha256 of the final manifest.json bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cere_contracts import canonical_json, manifest_identity, parse_json  # noqa: E402
from model_inventory import dump_header  # noqa: E402

EXAMPLE = ROOT / "examples/artifact"
TABLE = ROOT / "tests/fixtures/artifact_invalid.json"
MODEL_SRC = ROOT / "tests/fixtures/model_src"
MODEL_FAMILIES = ROOT / "tests/fixtures/model_src.families.json"
TREES = ".cache/huggingface/trees"


def inventory_case_src(dest: Path, case: dict) -> None:
    """Copy tests/fixtures/model_src to `dest` with its HF tree file replaced by the case's `tree_name`/`tree_text`."""
    shutil.copytree(MODEL_SRC, dest, copy_function=shutil.copyfile)
    for old in (dest / TREES).iterdir():
        old.unlink()
    (dest / TREES / case["tree_name"]).write_text(case["tree_text"])


def copy_example(dest: Path, src: Path | None = None) -> str:
    """Copy the artifact (files writable, so cases can mutate it) and return its manifest digest."""
    shutil.copytree(src or EXAMPLE, dest, copy_function=shutil.copyfile)
    for d, _dirs, _files in os.walk(dest):
        os.chmod(d, 0o755)
    return hashlib.sha256((dest / "manifest.json").read_bytes()).hexdigest()


def _select(container, key):
    if isinstance(key, dict):
        (field, value), = key.items()
        return next(i for i, x in enumerate(container) if x.get(field) == value)
    return key


def _parent(doc, pointer):
    node = doc
    for key in pointer[:-1]:
        node = node[_select(node, key)]
    return node, _select(node, pointer[-1])


def _rebuild_shard(path: Path, edit: dict) -> None:
    raw = path.read_bytes()
    h = dump_header(path)
    tensors = [dict(t, data=raw[t["offset"]:t["offset"] + t["bytes"]]) for t in h["tensors"]]
    metadata = h["metadata"]
    if "rename" in edit:
        old, new = edit["rename"]
        next(t for t in tensors if t["name"] == old)["name"] = new
    elif "drop" in edit:
        tensors.remove(next(t for t in tensors if t["name"] == edit["drop"]))
    elif "set" in edit:
        t = next(t for t in tensors if t["name"] == edit["set"]["tensor"])
        t.update({k: v for k, v in edit["set"].items() if k in ("dtype", "shape")})
    elif "metadata" in edit:
        metadata = edit["metadata"]
    else:
        raise ValueError(f"unknown shard edit {edit}")
    header, pos = {}, 0
    for t in tensors:
        header[t["name"]] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [pos, pos + len(t["data"])]}
        pos += len(t["data"])
    if metadata is not None:
        header["__metadata__"] = metadata
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(t["data"] for t in tensors))


def _reseal(root: Path, manifest: dict, rel: str) -> None:
    raw = (root / rel).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    entry = next(f for f in manifest["files"] if f["path"] == rel)
    entry.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    entry["safetensors"] = {"header_bytes": 8 + n, "header_sha256": hashlib.sha256(raw[:8 + n]).hexdigest(),
                            "tensors": len([k for k in json.loads(raw[8:8 + n]) if k != "__metadata__"])}
    manifest["model_hash"] = manifest_identity(manifest["files"])["model_hash"]


def _unsorted(value):
    if isinstance(value, dict):
        return {k: _unsorted(value[k]) for k in sorted(value, reverse=True)}
    if isinstance(value, list):
        return [_unsorted(x) for x in value]
    return value


def apply_ops(root: Path, ops: list) -> None:
    mpath = root / "manifest.json"
    for op in ops:
        kind = op["op"]
        if kind == "shard":
            manifest = parse_json(mpath.read_bytes())
            _rebuild_shard(root / op["path"], op["edit"])
            if op["reseal"]:
                _reseal(root, manifest, op["path"])
                mpath.write_bytes(canonical_json(manifest))
        elif kind == "flip_byte":
            p = root / op["path"]
            b = bytearray(p.read_bytes())
            b[op["offset"]] ^= 0xFF
            p.write_bytes(bytes(b))
        elif kind in ("manifest_set", "manifest_append", "manifest_add"):
            manifest = parse_json(mpath.read_bytes())
            node, key = _parent(manifest, op["pointer"])
            if kind == "manifest_set":
                node[key] = op["value"]
            elif kind == "manifest_append":
                node[key].append(op["value"])
            else:
                node[key] += op["delta"]
            mpath.write_bytes(canonical_json(manifest))
        elif kind == "manifest_raw":
            manifest = parse_json(mpath.read_bytes())
            transform = op["transform"]
            if transform == "reindent4":
                data = (json.dumps(manifest, sort_keys=True, indent=4, ensure_ascii=True) + "\n").encode()
            elif transform == "unsort_keys":
                data = (json.dumps(_unsorted(manifest), indent=2, ensure_ascii=True, separators=(",", ": "))
                        + "\n").encode()
            elif transform == "duplicate_key":
                line = b'  "quality_mode": "layout_only",\n'
                canon = canonical_json(manifest)
                assert canon.count(line) == 1
                data = canon.replace(line, line + line)
            elif transform == "replace":
                canon = canonical_json(manifest)
                assert op["old"].encode() in canon, op["old"]
                data = canon.replace(op["old"].encode(), op["new"].encode(), 1)
            else:
                raise ValueError(f"unknown manifest transform {transform}")
            mpath.write_bytes(data)
        elif kind == "add_file":
            p = root / op["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(bytes.fromhex(op["hex"]))
        elif kind == "symlink":
            p = root / op["path"]
            outside = root.parent / ("outside-" + op["path"].replace("/", "_"))
            os.replace(p, outside)
            os.symlink(outside.resolve(), p)
        elif kind == "fifo":
            p = root / op["path"]
            if p.exists():
                p.unlink()
            os.mkfifo(p)
        else:
            raise ValueError(f"unknown op {kind}")


def prepare_case(dest: Path, case: dict, src: Path | None = None) -> str:
    """Copy the artifact to `dest`, apply the case ops, return the digest to verify against."""
    copy_example(dest, src)
    apply_ops(dest, case["ops"])
    return case.get("expected_manifest_sha256") or hashlib.sha256((dest / "manifest.json").read_bytes()).hexdigest()
